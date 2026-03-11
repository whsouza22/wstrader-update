"""
WS Trader — Engine de 3 Redes Neurais para Double Top/Bottom
═════════════════════════════════════════════════════════════
Contém 3 modelos de ML independentes que votam Win/Loss de padrões DT:

  IA 1 (Geradora):     XGBoost           — analisa 40 features DT → prediz Win/Loss
  IA 2 (Validadora):   LightGBM          — confirma ou rejeita a IA 1
  IA 3 (Validadora 2): MLP 128→64→32     — captura padrões não-lineares

COMO FUNCIONA:
  1. O bot (WS_AUTO_AI_BULLEX.py) detecta um Double Top/Bottom
  2. Chama extract_features() do ws_adaptive_brain.py → vetor de 40 floats
  3. Chama predict_dt(features) deste arquivo → 3 NNs votam Win ou Loss
  4. Se 2+ NNs votam Win com confiança mínima → APROVADO (sem filtros hardcoded)

FEATURES USADAS (40 — DT_FEATURE_NAMES):
  f0-f25:  Features originais (geometria, rejeição, momentum, trend, etc.)
  f26-f39: Features de CONTEXTO/REGIME (a IA aprende sozinha):
    f26 range_compression     — mercado morto? compressão de range
    f27 ema_slope_20          — inclinação EMA20 (tendência curta)
    f28 trend_consistency     — % velas alinhadas com tendência
    f29 market_efficiency     — direcional vs ruído
    f30 alternation_rate      — alternância de cor (choppy?)
    f31 body_avg_vs_range     — corpo médio vs range médio
    f32 dist_to_local_extreme — distância ao extremo local (esticado?)
    f33 wick_density          — densidade de rejeição recente
    f34 adr_relative          — ADR relativo (volatilidade)
    f35 retests_at_level      — retestes no nível
    f36 space_to_opposite     — espaço livre para reversão
    f37 momentum_long         — momentum 10 velas (contexto amplo)
    f38 candle_uniformity     — uniformidade (estável vs caótico)
    f39 time_since_impulse    — tempo desde último impulso forte

TREINAMENTO:
  - Offline: python train_neural_network.py (usa CSVs de candles_5000/ e candles_deep/)
  - Modelo salvo em: ~/.wstrader/reversal_tf_{ativo}.pkl (válido 365 dias)
  - IA 1: XGBoost (300 trees, depth=6, colsample=0.8)
  - IA 2: LightGBM (300 trees, depth=6, histograma)
  - IA 3: MLP (128→64→32, relu, adam, early_stopping)
  - Backward compat: modelos treinados com 26 features continuam funcionando

MÉTODOS PRINCIPAIS:
  - feed_dt_features(features, result) — alimenta 40 features + Win(1)/Loss(0)
  - predict_dt(features) — prediz Win/Loss, retorna {win, prob_win, confidence, votes}
  - train_all() — treina os 3 modelos com dados acumulados
"""

import os, sys, time, pickle, logging, threading
import numpy as np
import pandas as pd

log = logging.getLogger("ReversalAI")

# ═══════════════════════════════════════════════════════
#  CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════
ATR_PERIOD          = 14
MIN_CANDLES         = 30       # Mín. velas para analisar
FUTURE_CANDLES      = 1        # Expiração: 1 minuto
MIN_BARS_BETWEEN    = 3        # Cooldown entre sinais
CANDLE_COUNT        = 200      # Velas no gráfico

# ── ML ──
MIN_SAMPLES_ML      = 60       # Amostras mínimas para treinar
RETRAIN_EVERY       = 20       # Retreino a cada N novos dados
TRAINING_WINDOW     = 2_000_000  # Máx. dados de treino
VALIDATION_SPLIT    = 0.20
MIN_VALIDATION_ACC  = 0.505    # Acurácia mín. para ativar (pure ML)

# ── Confiança mínima ──
AI1_CONF_MIN        = 52.0     # IA 1 precisa >= 52%
AI2_CONF_MIN        = 51.0     # IA 2 precisa >= 51%
AI3_CONF_MIN        = 51.0     # IA 3 (MLP) precisa >= 51%
AI3_MIN_SAMPLES     = 500      # MLP precisa de pelo menos 500 amostras

# ── Persistência ──
_user_data_dir = os.path.join(os.path.expanduser("~"), ".wstrader")
os.makedirs(_user_data_dir, exist_ok=True)
MODEL_PERSIST_FILE    = os.path.join(_user_data_dir, "reversal_tf_{broker}.pkl")
MODEL_PERSIST_MAX_AGE = 365 * 24 * 3600   # 365 dias — modelo treinado NUNCA expira
GITHUB_MODEL_URL = os.getenv(
    "WS_MODEL_URL",
    "https://raw.githubusercontent.com/whsouza22/wstrader-update/main/models/reversal_tf_{broker}.pkl"
)

# ═══════════════════════════════════════════════════════
#  FEATURES  — 32 features puras de mercado (usadas por feed_candles)
# ═══════════════════════════════════════════════════════
FEATURE_NAMES = [
    # ── Vela Atual (5) ──
    "body_atr",              # Corpo da vela / ATR
    "range_atr",             # Range da vela / ATR
    "upper_wick_pct",        # Pavio superior (% do range)
    "lower_wick_pct",        # Pavio inferior (% do range)
    "close_position",        # Posição do close: 0=low, 1=high

    # ── Velas Recentes (5) ──
    "body_prev1_atr",        # Corpo anterior / ATR
    "body_prev2_atr",        # 2ª anterior / ATR
    "bull_pct_5",            # % bullish nas últimas 5
    "bull_pct_10",           # % bullish nas últimas 10
    "consecutive_dir",       # Velas consecutivas mesma direção

    # ── Momentum (5) ──
    "momentum_5_atr",        # Momentum 5 velas / ATR
    "momentum_10_atr",       # Momentum 10 velas / ATR
    "momentum_20_atr",       # Momentum 20 velas / ATR
    "max_body_5_atr",        # Maior corpo nas últimas 5 / ATR
    "acceleration",          # 2ª metade vs 1ª metade

    # ── RSI (3) ──
    "rsi_value",             # RSI normalizado (0–1)
    "rsi_speed",             # Velocidade do RSI
    "rsi_from_50",           # Distância do neutro

    # ── Contexto (4) ──
    "price_vs_ma20",         # Preço vs MA20 / ATR
    "price_vs_ma50",         # Preço vs MA50 / ATR
    "bb_position",           # Posição Bollinger Bands
    "atr_change",            # Mudança de volatilidade

    # ── Micro-Estrutura (5) — adicionadas pela análise ──
    "wick_rejection_ratio",  # Pavio superior vs inferior (rejections)
    "body_vs_wick",          # Corpo / (corpo + pavios) → dominância
    "range_percentile",      # Percentil do range nas últimas 30
    "close_vs_prev_range",   # Close atual vs range anterior
    "trend_alignment",       # Momentum alinhado com MA? (+1/−1)

    # ── Contexto de Confirmação (5) — IA aprende padrões de reversão sozinha ──
    "stretch_up_score",      # Score de esticada pra cima (0-7) — quanto mais alto, mais esticado
    "stretch_dn_score",      # Score de esticada pra baixo (0-7)
    "stretch_vs_wick_top",   # Interação: stretch_up × pavio superior → rejeição vendedores
    "stretch_vs_wick_bot",   # Interação: stretch_dn × pavio inferior → rejeição compradores
    "stretch_vs_color",      # Vela reversa ao stretch? +1=reversa, -1=continuação, 0=neutro
]

# ═══════════════════════════════════════════════════════
#  DT FEATURES  — 40 features (26 originais + 14 contexto/regime)
#  IA aprende TUDO sozinha — sem filtros hardcoded
# ═══════════════════════════════════════════════════════
DT_FEATURE_NAMES = [
    "wick_ratio",            # f0  Rejection wick size                   0–1
    "close_position",        # f1  Close position in candle              0–1
    "macro_against_pct",     # f2  % candles contra trade (15 velas)     0–1
    "depth_ratio",           # f3  Pattern depth / ATR (/25)             0–1
    "symmetry",              # f4  Temporal symmetry                     0–1
    "span",                  # f5  Pattern width (/200)                  0–1
    "macro_body_avg",        # f6  Corpo médio 10 velas / ATR             0–1
    "progress_pct",          # f7  % of RS→Neck path traveled           0–1
    "ema_score",             # f8  EMA8-EMA20/ATR dir-adjusted         -1 to +1
    "momentum",              # f9  Momentum 5 candles dir-adjusted     -1 to +1
    "zone_score",            # f10 Historical level touches (/10)       0–1
    "arm_wr",                # f11 Historical WR for asset+type+mode    0–1
    "approach_decay",        # f12 Momentum deceleration at approach     0–1
    "rejection_quality",     # f13 Rejection quality (wicks+confirm)     0–1
    "volatility_regime",     # f14 Volatility regime (0=expl,1=calm)     0–1
    "trend_strength",        # f15 EMA10 vs EMA50 trend dir-adjusted   -1 to +1
    "price_vs_ema50",        # f16 Price distance from EMA50 dir-adj   -1 to +1
    "consecutive_against",   # f17 Consecutive candles against trade     0–1
    "body_dominance",        # f18 Net body pressure in trade dir      -1 to +1
    "macro_drift_against",   # f19 Drift contra trade 15 velas / ATR  -1 to +1
    "rsi_dir_adjusted",      # f20 RSI(14) ajustado pela direção       0–1
    "candle_range_ratio",    # f21 Tamanho da vela RS / ATR             0–1
    "acceleration",          # f22 Aceleração do momentum              -1 to +1
    "n_pivots_at_level",     # f23 Pivots no nível (2=DT, 3+=desgast)   0–1
    "body_conviction",       # f24 Convicção corpo 5 velas (0=doji,1=forte) 0–1
    "micro_range_ratio",     # f25 Atividade recente vs ATR (0=morto)   0–1
    # ═══ CONTEXTO/REGIME — IA aprende sozinha o que importa ═══
    "range_compression",     # f26 Range recente vs range médio          0–1
    "ema_slope_20",          # f27 Inclinação EMA20 dir-adjusted        -1 to +1
    "trend_consistency",     # f28 % velas alinhadas com tendência       0–1
    "market_efficiency",     # f29 Move líquido / move total             0–1
    "alternation_rate",      # f30 Taxa alternância de cor (ruído)       0–1
    "body_avg_vs_range",     # f31 Corpo médio vs range médio            0–1
    "dist_to_local_extreme", # f32 Distância ao extremo local            0–1
    "wick_density",          # f33 Densidade de rejeição recente         0–1
    "adr_relative",          # f34 ADR relativo (volatilidade)           0–1
    "retests_at_level",      # f35 Retestes no nível                     0–1
    "space_to_opposite",     # f36 Espaço livre para reversão            0–1
    "momentum_long",         # f37 Momentum 10 velas (contexto amplo)  -1 to +1
    "candle_uniformity",     # f38 Uniformidade (estável vs caótico)     0–1
    "time_since_impulse",    # f39 Tempo desde último impulso forte      0–1
]


# ═══════════════════════════════════════════════════════
#  REVERSAL AI  —  IA ML Pura (3 modelos)
# ═══════════════════════════════════════════════════════
class ReversalAI:

    def __init__(self, broker: str = "iq"):
        self.broker = broker
        self.history = []              # Alias para compatibilidade
        self._train_data = []          # [{"f": [30], "l": 0/1, "ts": float}]
        self._ai1 = None               # GradientBoosting (Geradora)
        self._ai2 = None               # LightGBM (Validadora)
        self._ai3 = None               # MLP Neural Network (Validadora 2)
        self._ai1_ready = False
        self._ai2_ready = False
        self._ai3_ready = False
        self._ai1_val = 0.0
        self._ai2_val = 0.0
        self._ai3_val = 0.0
        self._new_samples = 0
        self._loaded_n_samples = 0     # amostras do modelo carregado do disco
        self._locked_signals = {}      # {asset: {key: sig}}
        self._processed_candles = {}   # {asset: set(keys)}
        self._lock = threading.Lock()

        self._try_load_persisted_model()

    # ──────────────────────────────────────────────
    #  INDICADORES
    # ──────────────────────────────────────────────

    def _atr(self, highs, lows, closes, period=ATR_PERIOD):
        """Average True Range."""
        n = len(closes)
        if n < period + 1:
            rng = highs[:n] - lows[:n]
            return float(np.mean(rng)) if len(rng) > 0 else 1e-8
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i],
                        abs(highs[i] - closes[i - 1]),
                        abs(lows[i] - closes[i - 1]))
        return max(float(np.mean(tr[-period:])), 1e-8)

    def _rsi(self, closes, period=14):
        """RSI normalizado 0–1."""
        if len(closes) < period + 1:
            return 0.5
        deltas = np.diff(closes[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_g = float(np.mean(gains))
        avg_l = float(np.mean(losses))
        if avg_l < 1e-10:
            return 1.0
        rs = avg_g / avg_l
        return float(np.clip(1 - 1 / (1 + rs), 0, 1))

    def _bb_position(self, closes, idx, period=20, num_std=2.0):
        """Posição nas Bollinger Bands: −1.5 a +1.5."""
        start = max(0, idx - period + 1)
        w = closes[start:idx + 1]
        if len(w) < period:
            return 0.0
        ma = np.mean(w)
        sd = np.std(w)
        if sd < 1e-10:
            return 0.0
        pos = (closes[idx] - ma) / (num_std * sd)
        return float(np.clip(pos, -1.5, 1.5))

    # ──────────────────────────────────────────────
    #  PADRÕES DE VELA (8 clássicos)
    # ──────────────────────────────────────────────

    def _detect_candle_patterns(self, O, H, L, C, idx, atr):
        """Detecta 8 padrões de vela clássicos no idx."""
        def body(i): return abs(C[i] - O[i])
        def rng(i): return max(H[i] - L[i], 1e-10)
        def bull(i): return C[i] > O[i]
        def upper_wick(i): return H[i] - max(O[i], C[i])
        def lower_wick(i): return min(O[i], C[i]) - L[i]

        r = rng(idx)
        b = body(idx)
        uw = upper_wick(idx)
        lw = lower_wick(idx)
        b_pct = b / r

        # 1. ENGOLFO (Engulfing)
        engulfing = 0.0
        if idx >= 1:
            b1 = body(idx - 1)
            if b > b1 and b1 > 0:
                if bull(idx) and not bull(idx - 1):
                    if C[idx] > O[idx - 1] and O[idx] <= C[idx - 1]:
                        engulfing = min(b / b1, 3.0)
                elif not bull(idx) and bull(idx - 1):
                    if O[idx] >= C[idx - 1] and C[idx] < O[idx - 1]:
                        engulfing = -min(b / b1, 3.0)

        # 2. MARTELO (Hammer)
        hammer = 0.0
        if b_pct < 0.35 and lw > 2 * b and uw < b * 1.1:
            hammer = lw / r

        # 3. ESTRELA CADENTE (Shooting Star)
        shooting = 0.0
        if b_pct < 0.35 and uw > 2 * b and lw < b * 1.1:
            shooting = uw / r

        # 4. PIN BAR
        pin_bar = 0.0
        if uw / r > 0.60:
            pin_bar = -(uw / r)
        elif lw / r > 0.60:
            pin_bar = lw / r

        # 5. DOJI
        doji = 0.0
        if b_pct < 0.10:
            doji = 1.0
        elif b_pct < 0.20:
            doji = 0.5

        # 6. MORNING / EVENING STAR (3 velas)
        morning_evening = 0.0
        if idx >= 2:
            b2 = body(idx - 2)
            b1 = body(idx - 1)
            r1 = rng(idx - 1)
            if b1 / r1 < 0.30 and b2 / atr > 0.15 and b / atr > 0.15:
                if not bull(idx - 2) and bull(idx):
                    morning_evening = min(b / atr, 3.0)
                elif bull(idx - 2) and not bull(idx):
                    morning_evening = -min(b / atr, 3.0)

        # 7. THREE SOLDIERS / CROWS
        three = 0.0
        if idx >= 2:
            all_bull = all(bull(idx - j) for j in range(3))
            all_bear = all(not bull(idx - j) for j in range(3))
            if all_bull or all_bear:
                avg_b = sum(body(idx - j) for j in range(3)) / (3 * atr)
                if avg_b > 0.12:
                    three = min(avg_b, 3.0) if all_bull else -min(avg_b, 3.0)

        # 8. INSIDE BAR
        inside = 0.0
        if idx >= 1:
            if H[idx] <= H[idx - 1] and L[idx] >= L[idx - 1]:
                ratio = rng(idx) / rng(idx - 1)
                inside = max(0, 1.0 - ratio)

        # Nomes para display
        names = []
        if abs(engulfing) >= 1.0:
            names.append("ENGOLFO " + ("↑" if engulfing > 0 else "↓"))
        if hammer >= 0.5:
            names.append("MARTELO")
        if shooting >= 0.5:
            names.append("ESTRELA ↓")
        if abs(pin_bar) >= 0.6:
            names.append("PIN BAR")
        if doji >= 0.5:
            names.append("DOJI")
        if abs(morning_evening) >= 0.5:
            names.append("MORNING☆" if morning_evening > 0 else "EVENING☆")
        if abs(three) >= 0.2:
            names.append("3 SOLDIERS" if three > 0 else "3 CROWS")
        if inside >= 0.3:
            names.append("INSIDE")

        return {
            "engulfing": round(float(engulfing), 3),
            "hammer": round(float(hammer), 3),
            "shooting_star": round(float(shooting), 3),
            "pin_bar": round(float(np.clip(pin_bar, -1, 1)), 3),
            "doji_star": round(float(doji), 1),
            "morning_evening": round(float(morning_evening), 3),
            "three_soldiers": round(float(three), 3),
            "inside_bar": round(float(inside), 3),
            "names": names,
        }

    # ──────────────────────────────────────────────
    #  RSI
    # ──────────────────────────────────────────────

    def _rsi_analysis(self, closes, idx):
        """RSI: valor, velocidade, distância do 50."""
        rsi_val = self._rsi(closes[:idx + 1])

        # Velocidade
        rsi_speed = 0.0
        if idx >= 3:
            rsi_prev = self._rsi(closes[:idx - 2])
            rsi_speed = float(np.clip(rsi_val - rsi_prev, -0.5, 0.5))

        # Distância do neutro
        rsi_from_50 = abs(rsi_val - 0.5) * 2      # 0 = no 50, 1 = extremo

        # Label para dashboard
        rsi_pct = round(rsi_val * 100)
        if rsi_val > 0.70:
            label = f"RSI {rsi_pct} ⚠ OVERBOUGHT"
        elif rsi_val < 0.30:
            label = f"RSI {rsi_pct} ⚠ OVERSOLD"
        else:
            label = f"RSI {rsi_pct}"

        return {
            "value": round(float(rsi_val), 3),
            "speed": round(float(rsi_speed), 3),
            "from_50": round(float(rsi_from_50), 3),
            "label": label,
        }

    # ──────────────────────────────────────────────
    #  FEATURES  (30)
    # ──────────────────────────────────────────────

    def _extract_features(self, df, idx, atr, patterns, rsi_info):
        """32 features puras de mercado — sem regras de topo/fundo."""
        O = df["open"].values
        H = df["high"].values
        L = df["low"].values
        C = df["close"].values

        r = max(H[idx] - L[idx], 1e-10)
        b = abs(C[idx] - O[idx])

        # ── Vela Atual (5) ──
        body_atr    = b / atr
        range_atr   = r / atr
        upper_wick  = (H[idx] - max(O[idx], C[idx])) / r
        lower_wick  = (min(O[idx], C[idx]) - L[idx]) / r
        close_pos   = (C[idx] - L[idx]) / r

        # ── Velas Recentes (5) ──
        body_prev1 = abs(C[idx - 1] - O[idx - 1]) / atr if idx >= 1 else 0
        body_prev2 = abs(C[idx - 2] - O[idx - 2]) / atr if idx >= 2 else 0

        n5 = min(5, idx + 1)
        bull5 = sum(1 for j in range(idx - n5 + 1, idx + 1) if C[j] > O[j])
        bull_pct_5 = bull5 / n5

        n10 = min(10, idx + 1)
        bull10 = sum(1 for j in range(idx - n10 + 1, idx + 1) if C[j] > O[j])
        bull_pct_10 = bull10 / n10

        # Consecutivas na mesma direção
        is_bull = C[idx] > O[idx]
        consec = 0
        for j in range(idx, max(idx - 10, -1), -1):
            if j < 0:
                break
            if (C[j] > O[j]) == is_bull:
                consec += 1
            else:
                break

        # ── Momentum (5) ──
        mom5  = (C[idx] - C[max(0, idx - 5)]) / atr
        mom10 = (C[idx] - C[max(0, idx - 10)]) / atr
        mom20 = (C[idx] - C[max(0, idx - 20)]) / atr

        max_body_5 = max(abs(C[j] - O[j]) for j in range(max(0, idx - 4), idx + 1)) / atr

        mid = max(0, idx - 10)
        h1 = abs(C[mid] - C[max(0, idx - 20)]) / atr
        h2 = abs(C[idx] - C[mid]) / atr
        accel = h2 / h1 if h1 > 0.01 else 1.0

        # ── Contexto (4) ──
        ma20 = np.mean(C[max(0, idx - 19):idx + 1])
        ma50 = np.mean(C[max(0, idx - 49):idx + 1]) if idx >= 49 else ma20
        pv20 = (C[idx] - ma20) / atr
        pv50 = (C[idx] - ma50) / atr
        bb   = self._bb_position(C, idx)

        atr_rec = self._atr(
            H[max(0, idx - 6):idx + 1],
            L[max(0, idx - 6):idx + 1],
            C[max(0, idx - 6):idx + 1], period=7)
        atr_old = self._atr(
            H[max(0, idx - 20):max(1, idx - 6)],
            L[max(0, idx - 20):max(1, idx - 6)],
            C[max(0, idx - 20):max(1, idx - 6)], period=7) if idx > 20 else atr
        atr_chg = atr_rec / atr_old if atr_old > 1e-10 else 1.0

        # ── Micro-Estrutura (5) — novas features ──
        # 1) Wick Rejection Ratio: pavio sup vs inf (>0 = rejeição em cima)
        uw_abs = H[idx] - max(O[idx], C[idx])
        lw_abs = min(O[idx], C[idx]) - L[idx]
        wick_total = uw_abs + lw_abs
        wick_reject = (uw_abs - lw_abs) / wick_total if wick_total > 1e-10 else 0.0

        # 2) Body vs Wick: quanto do range é corpo (dominância direcional)
        body_vs_wick = b / r if r > 1e-10 else 0.0

        # 3) Range Percentile: quão grande é esta vela vs últimas 30
        start_rng = max(0, idx - 29)
        ranges_30 = [H[j] - L[j] for j in range(start_rng, idx + 1)]
        if len(ranges_30) > 1:
            range_pctile = float(np.searchsorted(np.sort(ranges_30), r)) / len(ranges_30)
        else:
            range_pctile = 0.5

        # 4) Close vs Previous Range: close atual vs range da vela anterior
        if idx >= 1:
            prev_range = max(H[idx-1] - L[idx-1], 1e-10)
            close_vs_prev_rng = (C[idx] - C[idx-1]) / prev_range
            close_vs_prev_rng = float(np.clip(close_vs_prev_rng, -3.0, 3.0))
        else:
            close_vs_prev_rng = 0.0

        # 5) Trend Alignment: momentum alinhado com posição vs MA20?
        # +1 se momento e posição concordam, -1 se discordam
        mom_dir = 1.0 if mom5 > 0 else (-1.0 if mom5 < 0 else 0.0)
        ma_dir = 1.0 if pv20 > 0 else (-1.0 if pv20 < 0 else 0.0)
        trend_align = mom_dir * ma_dir  # +1=alinhado, -1=divergente

        # ── Contexto de Confirmação (5) — IA aprende padrões sozinha ──
        # Calcular stretch scores inline (mesma lógica de _is_stretched)
        rsi_val = rsi_info["value"]
        s_up = 0
        if bb > 0.4:    s_up += 1
        if bb > 0.7:    s_up += 1
        if rsi_val > 0.58: s_up += 1
        if rsi_val > 0.70: s_up += 1
        if pv20 > 0.5:  s_up += 1
        if pv50 > 0.8:  s_up += 1
        if mom10 > 0.5:  s_up += 1

        s_dn = 0
        if bb < -0.4:    s_dn += 1
        if bb < -0.7:    s_dn += 1
        if rsi_val < 0.42: s_dn += 1
        if rsi_val < 0.30: s_dn += 1
        if pv20 < -0.5:  s_dn += 1
        if pv50 < -0.8:  s_dn += 1
        if mom10 < -0.5:  s_dn += 1

        # Interação stretch × pavio (a IA aprende se pavio = rejeição)
        stretch_vs_wick_top = s_up * upper_wick   # stretch UP + pavio sup = vendedores rejeitando
        stretch_vs_wick_bot = s_dn * lower_wick   # stretch DOWN + pavio inf = compradores rejeitando

        # Cor da vela relativa ao stretch (a IA aprende se cor importa)
        is_bearish = C[idx] < O[idx]
        if s_up >= 2 or s_dn >= 2:
            if s_up > s_dn:
                # Esticado pra cima: vela vermelha = reversão, verde = continuação
                stretch_vs_color = 1.0 if is_bearish else -1.0
            else:
                # Esticado pra baixo: vela verde = reversão, vermelha = continuação
                stretch_vs_color = 1.0 if not is_bearish else -1.0
        else:
            stretch_vs_color = 0.0

        return [
            # Vela Atual (5)
            body_atr, range_atr, upper_wick, lower_wick, close_pos,
            # Velas Recentes (5)
            body_prev1, body_prev2, bull_pct_5, bull_pct_10, consec,
            # Momentum (5)
            mom5, mom10, mom20, max_body_5, accel,
            # RSI (3)
            rsi_info["value"], rsi_info["speed"], rsi_info["from_50"],
            # Contexto (4)
            pv20, pv50, bb, atr_chg,
            # Micro-Estrutura (5)
            wick_reject, body_vs_wick, range_pctile, close_vs_prev_rng, trend_align,
            # Contexto de Confirmação (5)
            float(s_up), float(s_dn), stretch_vs_wick_top, stretch_vs_wick_bot, stretch_vs_color,
        ]

    # ──────────────────────────────────────────────
    #  VOLUME SINTÉTICO (para OTC sem volume real)
    # ──────────────────────────────────────────────

    @staticmethod
    def _synthetic_volume(O, H, L, C, idx, atr):
        """Calcula volume sintético baseado em atividade de preço.

        Componentes:
          1. Range / ATR → vela grande = mais atividade
          2. Body % → corpo grande = convicção direcional forte
          3. Wick total → pavios grandes = muita rejeição/atividade
          4. Gap vs anterior → gap grande = ordem forte no abertura
          5. Variação vs média 10 → consistência de atividade

        Retorna: volume sintético (int ~50-300, comparável a volume real)
                 vol_ratio (float, 1.0 = média)
        """
        if atr < 1e-10:
            return 100, 1.0

        r = max(H[idx] - L[idx], 1e-10)
        b = abs(C[idx] - O[idx])

        # 1. Range intensity (primary driver)
        range_ratio = r / atr

        # 2. Body conviction (0-1)
        body_pct = b / r

        # 3. Wick activity (more wicks = more two-way trading)
        uw = H[idx] - max(O[idx], C[idx])
        lw = min(O[idx], C[idx]) - L[idx]
        wick_pct = (uw + lw) / r

        # 4. Gap from previous close
        gap = abs(C[idx] - C[max(0, idx - 1)]) / atr if idx > 0 else 0

        # Synthetic volume score
        # Range is king, body adds conviction, wicks add activity, gap adds urgency
        syn_raw = range_ratio * (0.4 + 0.3 * body_pct + 0.2 * wick_pct + 0.1 * min(gap, 2.0))

        # Calculate rolling average for ratio
        window = min(20, idx + 1)
        if window > 1:
            syn_hist = []
            for j in range(max(0, idx - window + 1), idx + 1):
                rj = max(H[j] - L[j], 1e-10)
                bj = abs(C[j] - O[j])
                bp = bj / rj
                uwj = H[j] - max(O[j], C[j])
                lwj = min(O[j], C[j]) - L[j]
                wp = (uwj + lwj) / rj
                gj = abs(C[j] - C[max(0, j - 1)]) / atr if j > 0 else 0
                syn_hist.append(rj / atr * (0.4 + 0.3 * bp + 0.2 * wp + 0.1 * min(gj, 2.0)))
            avg = sum(syn_hist) / len(syn_hist) if syn_hist else syn_raw
        else:
            avg = syn_raw

        vol_ratio = syn_raw / avg if avg > 1e-10 else 1.0

        # Normalize to integer volume (similar scale to real volume)
        syn_vol = int(syn_raw * 100)

        return syn_vol, round(vol_ratio, 2)

    # ──────────────────────────────────────────────
    #  DETECÇÃO DE PREÇO ESTICADO (Reversal Zone)
    # ──────────────────────────────────────────────

    def _is_stretched(self, feats):
        """Detecta se o preço está esticado (overextended).

        Retorna (score_up, score_down):
          score_up  >= 2 → preço esticado pra CIMA → considerar PUT
          score_down >= 2 → preço esticado pra BAIXO → considerar CALL
        """
        bb   = feats[FEATURE_NAMES.index("bb_position")]
        rsi  = feats[FEATURE_NAMES.index("rsi_value")]
        pv20 = feats[FEATURE_NAMES.index("price_vs_ma20")]
        mom10 = feats[FEATURE_NAMES.index("momentum_10_atr")]
        pv50 = feats[FEATURE_NAMES.index("price_vs_ma50")]

        # ── Preço esticado para CIMA (overbought zone) ──
        score_up = 0
        if bb > 0.4:   score_up += 1
        if bb > 0.7:   score_up += 1   # Bem acima da BB
        if rsi > 0.58:  score_up += 1
        if rsi > 0.70:  score_up += 1   # RSI overbought
        if pv20 > 0.5:  score_up += 1   # Acima da MA20
        if pv50 > 0.8:  score_up += 1   # Bem acima da MA50
        if mom10 > 0.5:  score_up += 1   # Momentum alto

        # ── Preço esticado para BAIXO (oversold zone) ──
        score_dn = 0
        if bb < -0.4:   score_dn += 1
        if bb < -0.7:   score_dn += 1   # Bem abaixo da BB
        if rsi < 0.42:  score_dn += 1
        if rsi < 0.30:  score_dn += 1   # RSI oversold
        if pv20 < -0.5:  score_dn += 1   # Abaixo da MA20
        if pv50 < -0.8:  score_dn += 1   # Bem abaixo da MA50
        if mom10 < -0.5:  score_dn += 1   # Momentum baixo

        return score_up, score_dn

    # ──────────────────────────────────────────────
    #  PREDIÇÃO
    # ──────────────────────────────────────────────

    def _predict_ai1(self, fv_df):
        """IA 1 (Geradora): GradientBoosting → P(up)."""
        if not self._ai1:
            return None
        try:
            return float(self._ai1.predict_proba(fv_df)[0][1])
        except Exception:
            return None

    def _predict_ai2(self, fv_df):
        """IA 2 (Validadora): LightGBM → P(up)."""
        if not self._ai2:
            return None
        try:
            return float(self._ai2.predict_proba(fv_df)[0][1])
        except Exception:
            return None

    def _predict_ai3(self, fv_df):
        """IA 3 (Validadora 2): MLP Neural Network → P(up)."""
        if not self._ai3:
            return None
        try:
            return float(self._ai3.predict_proba(fv_df)[0][1])
        except Exception:
            return None

    # ──────────────────────────────────────────────
    #  ANÁLISE PRINCIPAL
    # ──────────────────────────────────────────────

    def analyze_candles(self, df, asset: str = "", collect_data: bool = True) -> list:
        """Analisa velas com ML puro.

        - Coleta dados a CADA vela (para treino)
        - Emite sinal SOMENTE quando IA 1 + IA 2 concordam
        - Sinais TRAVADOS: uma vez emitido, nunca muda
        """
        signals = []
        n = len(df)
        if n < MIN_CANDLES + 5:
            return signals

        O = df["open"].values
        H = df["high"].values
        L = df["low"].values
        C = df["close"].values
        atr = self._atr(H, L, C)
        if atr < 1e-10:
            return signals

        # ═══ TRAVAMENTO ═══
        if asset not in self._locked_signals:
            self._locked_signals[asset] = {}
        locked = self._locked_signals[asset]

        if asset not in self._processed_candles:
            self._processed_candles[asset] = set()
        processed = self._processed_candles[asset]

        idx_to_key = {}
        for i, t in enumerate(df.index):
            if hasattr(t, 'strftime'):
                idx_to_key[i] = t.strftime("%Y-%m-%d %H:%M")

        last_sig_idx = -MIN_BARS_BETWEEN - 1

        for idx in range(MIN_CANDLES, n):
            key = idx_to_key.get(idx)

            # ── Sinal TRAVADO ──
            if key and key in locked:
                sig_copy = dict(locked[key])
                sig_copy["idx"] = idx
                sig_copy.pop("_feats", None)

                # Resolver resultado
                if sig_copy.get("result") is None and idx + FUTURE_CANDLES < n:
                    fc = C[idx + FUTURE_CANDLES]
                    ec = sig_copy["entry_price"]
                    d  = sig_copy["direction"]
                    if fc == ec:
                        sig_copy["result"] = "tie"
                    elif d == "CALL":
                        sig_copy["result"] = "win" if fc > ec else "loss"
                    else:
                        sig_copy["result"] = "win" if fc < ec else "loss"
                    locked[key] = dict(sig_copy)

                signals.append(sig_copy)
                last_sig_idx = idx
                continue

            # Já processou?
            if key and key in processed:
                continue
            if key:
                processed.add(key)

            # ── Features ──
            local_atr = self._atr(H[:idx + 1], L[:idx + 1], C[:idx + 1])
            if local_atr < 1e-10:
                continue

            patterns = self._detect_candle_patterns(O, H, L, C, idx, local_atr)
            rsi_info = self._rsi_analysis(C, idx)
            feats = self._extract_features(df, idx, local_atr, patterns, rsi_info)
            if feats is None:
                continue

            # ── Coleta de dados (TODA vela → treino) ──
            if collect_data and idx + FUTURE_CANDLES < n:
                fc = C[idx + FUTURE_CANDLES]
                if fc != C[idx]:   # skip ties
                    label = 1 if fc > C[idx] else 0
                    self._record(feats, label)

            # ── Cooldown ──
            if idx - last_sig_idx < MIN_BARS_BETWEEN:
                continue

            # ── ML Prediction ──
            ml_active = self._ai1_ready and self._ai2_ready
            if not ml_active:
                continue

            try:
                fv = pd.DataFrame([feats], columns=FEATURE_NAMES)

                # IA 1 (Geradora)
                p1 = self._predict_ai1(fv)
                if p1 is None:
                    continue

                # IA 2 (Validadora)
                p2 = self._predict_ai2(fv)
                if p2 is None:
                    continue

                # IA 3 (MLP Neural Network — Validadora 2)
                p3 = None
                if self._ai3_ready:
                    try:
                        fv_scaled = pd.DataFrame(
                            self._ai3_scaler.transform(fv),
                            columns=FEATURE_NAMES
                        )
                        p3 = self._predict_ai3(fv_scaled)
                    except Exception:
                        p3 = None

                # Direção + Confiança
                ai1_call = p1 > 0.5
                ai2_call = p2 > 0.5
                ai1_conf = (p1 if ai1_call else 1 - p1) * 100
                ai2_conf = (p2 if ai2_call else 1 - p2) * 100
                ai1_dir = "CALL" if ai1_call else "PUT"
                ai2_dir = "CALL" if ai2_call else "PUT"

                # IA 3 (se disponível)
                ai3_dir = None
                ai3_conf = 0.0
                if p3 is not None:
                    ai3_call = p3 > 0.5
                    ai3_conf = (p3 if ai3_call else 1 - p3) * 100
                    ai3_dir = "CALL" if ai3_call else "PUT"

                # Sistema de votação: maioria decide (2 de 3)
                votes = {"CALL": 0, "PUT": 0}
                votes[ai1_dir] += 1
                votes[ai2_dir] += 1
                if ai3_dir is not None:
                    votes[ai3_dir] += 1

                direction = "CALL" if votes["CALL"] > votes["PUT"] else "PUT"
                n_agree = votes[direction]

                # Sem IA 3: as 2 devem concordar (como antes)
                # Com IA 3: pelo menos 2 de 3 devem concordar
                if ai3_dir is None:
                    if ai1_dir != ai2_dir:
                        continue
                else:
                    if n_agree < 2:
                        continue

                # Confiança mínima (cada IA que votou na direção)
                if ai1_dir == direction and ai1_conf < AI1_CONF_MIN:
                    continue
                if ai2_dir == direction and ai2_conf < AI2_CONF_MIN:
                    continue
                if ai3_dir == direction and ai3_conf < AI3_CONF_MIN:
                    continue

                # Confiança ponderada
                if ai3_dir is not None:
                    confidence = ai1_conf * 0.40 + ai2_conf * 0.30 + ai3_conf * 0.30
                else:
                    confidence = ai1_conf * 0.6 + ai2_conf * 0.4
            except Exception:
                continue

            # ── REVERSAL ONLY: só entrar quando preço está esticado ──
            score_up, score_dn = self._is_stretched(feats)
            stretch_min = 2   # Mínimo de score para considerar esticado

            if score_up < stretch_min and score_dn < stretch_min:
                continue  # Preço NÃO está esticado — sem entrada

            # Preço esticado para CIMA → só aceitar PUT (reversão pra baixo)
            if score_up >= stretch_min and direction != "PUT":
                continue

            # Preço esticado para BAIXO → só aceitar CALL (reversão pra cima)
            if score_dn >= stretch_min and direction != "CALL":
                continue

            stretch_score = max(score_up, score_dn)
            stretch_dir = "UP" if score_up >= score_dn else "DOWN"

            # ── FILTRO: Wick Rejection (bloqueia LOSS) ──
            # Análise profunda mostrou: wick rejeição >= 30% = +4.4% WR
            # PUT precisa pavio superior (vendedores rejeitando)
            # CALL precisa pavio inferior (compradores rejeitando)
            uwk = feats[FEATURE_NAMES.index("upper_wick_pct")]
            lwk = feats[FEATURE_NAMES.index("lower_wick_pct")]
            wick_min = 0.08  # Mínimo 8% de pavio de rejeição (suave)
            if direction == "PUT" and uwk < wick_min:
                continue  # PUT sem pavio superior = sem rejeição de vendedores
            if direction == "CALL" and lwk < wick_min:
                continue  # CALL sem pavio inferior = sem rejeição de compradores

            # ── FILTRO: Zona morta de confiança (60-65% = 40.1% WR!) ──
            if 59.5 <= confidence <= 65.5:
                continue  # Zona de confiança com WR péssimo

            # ── FILTRO: Micro-tendência — consec == 1 é ruído (52.4% WR) ──
            # Análise mostrou: consec >= 2 = ~62% WR vs consec == 1 = 52.4%
            # Precisamos de pelo menos 2 velas consecutivas para confirmar direção
            consec_val = feats[FEATURE_NAMES.index("consecutive_dir")]
            if consec_val < 2:
                continue  # Sem direção clara, entrada ruidosa

            # ── FILTRO: Aceleração moderada = zona de LOSS (44.2% WR) ──
            # accel 0.05~0.50 = momentum CRESCENDO moderado = reversão falha
            # Desacelerando ou forte aceleração (exaustão) = OK
            accel_val = feats[FEATURE_NAMES.index("acceleration")]
            if 0.05 < accel_val < 0.50:
                continue  # Momentum ainda acelerando, reversão prematura

            # ── FILTRO: Volume Sintético (OTC sem volume real) ──
            # Análise mostrou: vol_ratio 1.2-2.0 = 70.4% WR (atividade alta)
            # vol_ratio < 0.5 = 54.5% WR (mercado parado = ruído)
            syn_vol, vol_ratio = self._synthetic_volume(O, H, L, C, idx, local_atr)
            if vol_ratio < 0.5:
                continue  # Mercado muito parado, sinal é ruído

            # ── Resultado real ──
            result = None
            if idx + FUTURE_CANDLES < n:
                fc = C[idx + FUTURE_CANDLES]
                ec = C[idx]
                if fc == ec:
                    result = "tie"
                elif direction == "CALL":
                    result = "win" if fc > ec else "loss"
                else:
                    result = "win" if fc < ec else "loss"

            # ── Montar sinal ──
            sig = {
                "idx": idx,
                "direction": direction,
                "confidence": round(confidence, 1),
                "ai1_conf": round(ai1_conf, 1),
                "ai1_dir": ai1_dir,
                "ai2_conf": round(ai2_conf, 1),
                "ai2_dir": ai2_dir,
                "ai3_conf": round(ai3_conf, 1) if ai3_dir else 0.0,
                "ai3_dir": ai3_dir or "",
                "n_agree": n_agree,
                "result": result,
                "entry_price": round(float(C[idx]), 6),
                "ml_active": True,
                "skipped": False,
                "skip_reason": "",
                "patterns": patterns.get("names", []),
                "rsi_value": rsi_info["value"],
                "rsi_label": rsi_info["label"],
                "momentum": round(float((C[idx] - C[max(0, idx - 10)]) / local_atr), 2),
                "stretch_score": stretch_score,
                "stretch_dir": stretch_dir,
                "syn_vol": syn_vol,
                "vol_ratio": vol_ratio,
            }
            if hasattr(df.index[idx], "strftime"):
                sig["time"] = df.index[idx].strftime("%H:%M")

            # Travar sinal
            if key:
                locked[key] = dict(sig)

            signals.append(sig)
            last_sig_idx = idx

        # ── Auto-retrain DESATIVADO ──
        # Modelo treinado offline (train_neural_network.py) é PROTEGIDO.
        # Não sobreescrever com auto-retrain online de poucos dados.
        # if (collect_data
        #         and self._new_samples >= RETRAIN_EVERY
        #         and len(self._train_data) >= MIN_SAMPLES_ML):
        #     self._retrain()
        #     self._new_samples = 0

        # ── Limpeza ──
        if len(locked) > 500:
            for k in sorted(locked.keys())[:-300]:
                del locked[k]
        if len(processed) > 1000:
            keep = sorted(processed)[-500:]
            processed.clear()
            processed.update(keep)

        return signals

    # ──────────────────────────────────────────────
    #  S/R Zones (para o gráfico)
    # ──────────────────────────────────────────────

    def get_stall_zones(self, df):
        """Retorna suportes e resistências simples (local min/max)."""
        n = len(df)
        if n < 30:
            return []

        H = df["high"].values
        L = df["low"].values
        C = df["close"].values
        atr = self._atr(H, L, C)

        zones = []
        window = 5
        scan_start = max(window, n - 80)

        for i in range(scan_start, n - window):
            # Local high → resistência
            if H[i] == max(H[max(0, i - window):i + window + 1]):
                zones.append({
                    "level": round(float(H[i]), 6),
                    "type": "resistencia",
                    "strength": 1,
                    "rejection": 0,
                    "range_atr": 0,
                })
            # Local low → suporte
            if L[i] == min(L[max(0, i - window):i + window + 1]):
                zones.append({
                    "level": round(float(L[i]), 6),
                    "type": "suporte",
                    "strength": 1,
                    "rejection": 0,
                    "range_atr": 0,
                })

        if not zones:
            return []

        # Merge próximos
        merge_dist = atr * 0.5
        zones.sort(key=lambda z: z["level"])
        merged = []
        cur = zones[0]
        for z in zones[1:]:
            if abs(z["level"] - cur["level"]) <= merge_dist:
                cur["strength"] += 1
            else:
                merged.append(cur)
                cur = z
        merged.append(cur)
        return merged[-20:]

    # Alias
    def get_wick_zones(self, df):
        return self.get_stall_zones(df)

    # ──────────────────────────────────────────────
    #  RECORD + TRAIN
    # ──────────────────────────────────────────────

    def _record(self, feats, label):
        """Registra dados para treino."""
        self._train_data.append({"f": feats, "l": label, "ts": time.time()})
        self._new_samples += 1
        self.history = self._train_data    # Alias

    def _retrain(self):
        """Treina IA 1 (XGBoost) + IA 2 (LightGBM) + IA 3 (MLP 128→64→32).
        IA aprende TUDO sozinha — sem filtros hardcoded.
        40 features incluem regime/contexto/qualidade."""
        try:
            # ── IA 1: XGBoost (melhor que GradientBoosting) ──
            try:
                from xgboost import XGBClassifier
                _xgb_ok = True
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                _xgb_ok = False

            # ── IA 2: LightGBM ──
            try:
                from lightgbm import LGBMClassifier
                _lgbm_ok = True
            except ImportError:
                from sklearn.ensemble import ExtraTreesClassifier
                _lgbm_ok = False

            # Auto-detectar formato: DT (40 features) ou genérico (32) ou legacy (26)
            nf_dt = len(DT_FEATURE_NAMES)
            nf_gen = len(FEATURE_NAMES)
            data_dt = [s for s in self._train_data[-TRAINING_WINDOW:]
                       if len(s["f"]) == nf_dt]
            # Backward compat: aceitar 26 features (legacy) — pad com 0.5 para 40
            _LEGACY_NF = 26
            data_legacy = [s for s in self._train_data[-TRAINING_WINDOW:]
                           if len(s["f"]) == _LEGACY_NF and nf_dt != _LEGACY_NF]
            if data_legacy and len(data_dt) < MIN_SAMPLES_ML:
                # Pad legacy 26→40 com valores neutros (0.5)
                for s in data_legacy:
                    s["f"] = list(s["f"]) + [0.5] * (nf_dt - _LEGACY_NF)
                data_dt.extend(data_legacy)
                log.info(f"  Legacy compat: {len(data_legacy)} amostras 26→{nf_dt} features (padded)")

            data_gen = [s for s in self._train_data[-TRAINING_WINDOW:]
                        if len(s["f"]) == nf_gen]

            # Priorizar DT se houver dados suficientes
            if len(data_dt) >= MIN_SAMPLES_ML:
                data = data_dt
                feature_names = DT_FEATURE_NAMES
                nf = nf_dt
                self._n_features_trained = nf
                log.info(f"  Treino DT: {len(data)} amostras ({nf} features)")
            elif len(data_gen) >= MIN_SAMPLES_ML:
                data = data_gen
                feature_names = FEATURE_NAMES
                nf = nf_gen
                self._n_features_trained = nf
                log.info(f"  Treino genérico: {len(data)} amostras ({nf} features)")
            else:
                log.info(f"Aguardando dados (DT={len(data_dt)} gen={len(data_gen)}/{MIN_SAMPLES_ML})")
                return

            X = np.array([s["f"] for s in data])
            y = np.array([s["l"] for s in data])
            if len(np.unique(y)) < 2:
                return

            # Subamostrar se dataset muito grande
            MAX_TRAIN = 200_000
            if len(X) > MAX_TRAIN:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(X), MAX_TRAIN, replace=False)
                idx.sort()
                X = X[idx]
                y = y[idx]
                log.info(f"  Subamostrado: {len(data)} → {MAX_TRAIN} amostras")

            X_df = pd.DataFrame(X, columns=feature_names)
            n = len(X)

            split = int(n * (1 - VALIDATION_SPLIT))
            if split < MIN_SAMPLES_ML or (n - split) < 5:
                Xt, yt = X_df, y
                Xv, yv = None, None
            else:
                Xt, yt = X_df.iloc[:split], y[:split]
                Xv, yv = X_df.iloc[split:], y[split:]

            # Peso exponencial (dados recentes valem mais)
            w = np.exp(np.linspace(-2.0, 0.0, len(Xt)))

            # Balanceamento de classes
            n_pos = int(np.sum(yt == 1))
            n_neg = int(np.sum(yt == 0))
            if n_pos > 0 and n_neg > 0 and n_pos > 2 * n_neg:
                class_ratio = n_pos / n_neg
                class_w = np.ones(len(yt))
                class_w[yt == 0] = min(class_ratio, 6.0)
                w = w * class_w
                log.info(f"  Class balance: {n_pos}W/{n_neg}L → LOSSes ×{min(class_ratio, 6.0):.1f}")

            # ── IA 1: XGBoost (melhor regularização, mais rápido, feature importance precisa) ──
            if _xgb_ok:
                ai1 = XGBClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    min_child_weight=10, gamma=0.1,
                    reg_alpha=0.1, reg_lambda=1.0,
                    random_state=42, verbosity=0,
                    eval_metric='logloss', use_label_encoder=False)
                ai1.fit(Xt, yt, sample_weight=w)
                log.info(f"  IA 1: XGBoost (300 trees, depth=6)")
            else:
                ai1 = GradientBoostingClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.06,
                    subsample=0.8, min_samples_leaf=15, random_state=42)
                ai1.fit(Xt, yt, sample_weight=w)
                log.info(f"  IA 1: GradientBoosting fallback (200 trees)")

            # ── HARD-EXAMPLE MINING ──
            try:
                _proba_train = ai1.predict_proba(Xt)[:, 1]
                _confidence = np.abs(_proba_train - 0.5)
                _hard_w = 1.0 + 2.0 * (1.0 - np.clip(_confidence * 2, 0, 1))
                _pred_train = (_proba_train >= 0.5).astype(int)
                _wrong = _pred_train != yt
                _hard_w[_wrong] *= 3.0
                w_hard = w * _hard_w
                _n_hard = int(np.sum(_wrong))
                log.info(f"  Hard-mining: {_n_hard}/{len(yt)} errados upweighted 3×")
            except Exception:
                w_hard = w

            # ── IA 2: LightGBM (diversidade: usa histograma vs exact split) ──
            if _lgbm_ok:
                ai2 = LGBMClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    min_child_samples=15, reg_alpha=0.1, reg_lambda=1.0,
                    random_state=99, verbose=-1)
                ai2.fit(Xt, yt, sample_weight=w_hard)
            else:
                ai2 = ExtraTreesClassifier(
                    n_estimators=100, max_depth=6,
                    min_samples_leaf=10, random_state=99)
                ai2.fit(Xt, yt, sample_weight=w_hard)

            # ── Validação IA1 + IA2 ──
            ai1_ok = True
            ai2_ok = True
            if Xv is not None:
                pred1 = (ai1.predict_proba(Xv)[:, 1] >= 0.5).astype(int)
                acc1 = float(np.mean(pred1 == yv))
                self._ai1_val = acc1
                pred2 = (ai2.predict_proba(Xv)[:, 1] >= 0.5).astype(int)
                acc2 = float(np.mean(pred2 == yv))
                self._ai2_val = acc2

                log.info(f"  IA 1: val={acc1:.1%} | IA 2: val={acc2:.1%} | n={n}")

                if acc1 < MIN_VALIDATION_ACC:
                    log.info(f"  ⚠ IA 1 ({acc1:.1%}) < {MIN_VALIDATION_ACC:.0%} → desativada")
                    ai1_ok = False
                if acc2 < MIN_VALIDATION_ACC:
                    log.info(f"  ⚠ IA 2 ({acc2:.1%}) < {MIN_VALIDATION_ACC:.0%} → desativada")
                    ai2_ok = False

            if ai1_ok:
                self._ai1 = ai1
                self._ai1_ready = True
            if ai2_ok:
                self._ai2 = ai2
                self._ai2_ready = True

            # ── IA 3: MLP 128→64→32 (captura interações não-lineares) ──
            ai3 = None
            ai3_ok = False
            if n >= AI3_MIN_SAMPLES:
                try:
                    from sklearn.neural_network import MLPClassifier
                    from sklearn.preprocessing import StandardScaler

                    scaler = StandardScaler()

                    # Balancear classes
                    _pos_idx = np.where(yt == 1)[0]
                    _neg_idx = np.where(yt == 0)[0]
                    if len(_pos_idx) > 2 * len(_neg_idx) and len(_neg_idx) >= 20:
                        _rng_mlp = np.random.RandomState(77)
                        _keep_pos = _rng_mlp.choice(_pos_idx, size=2 * len(_neg_idx), replace=False)
                        _bal_idx = np.sort(np.concatenate([_keep_pos, _neg_idx]))
                    else:
                        _bal_idx = np.arange(len(yt))

                    # Oversample hard examples
                    try:
                        _proba_mlp = ai1.predict_proba(Xt)[:, 1]
                        _pred_mlp = (_proba_mlp >= 0.5).astype(int)
                        _wrong_mlp = np.where(_pred_mlp != yt)[0]
                        _hard_in_bal = np.intersect1d(_bal_idx, _wrong_mlp)
                        if len(_hard_in_bal) >= 5:
                            _bal_idx = np.sort(np.concatenate([_bal_idx, _hard_in_bal]))
                            log.info(f"  MLP hard-mining: +{len(_hard_in_bal)} hard examples oversampled")
                    except Exception:
                        pass

                    Xt_mlp = Xt.iloc[_bal_idx]
                    yt_mlp = yt[_bal_idx]

                    Xt_scaled = pd.DataFrame(
                        scaler.fit_transform(Xt_mlp), columns=Xt.columns
                    )

                    # MLP mais profunda: 128→64→32 (captura interações complexas)
                    ai3 = MLPClassifier(
                        hidden_layer_sizes=(128, 64, 32),
                        activation='relu',
                        solver='adam',
                        alpha=0.01,
                        learning_rate='adaptive',
                        learning_rate_init=0.001,
                        max_iter=500,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=25,
                        random_state=77,
                        verbose=False,
                    )
                    ai3.fit(Xt_scaled, yt_mlp)
                    self._ai3_scaler = scaler

                    if Xv is not None:
                        Xv_scaled = pd.DataFrame(
                            scaler.transform(Xv), columns=Xv.columns
                        )
                        pred3 = (ai3.predict_proba(Xv_scaled)[:, 1] >= 0.5).astype(int)
                        acc3 = float(np.mean(pred3 == yv))
                        self._ai3_val = acc3
                        log.info(f"  IA 3 (MLP): val={acc3:.1%} | layers=(128,64,32)")
                        if acc3 < MIN_VALIDATION_ACC:
                            log.info(f"  ⚠ IA 3 ({acc3:.1%}) < {MIN_VALIDATION_ACC:.0%} → desativada")
                        else:
                            ai3_ok = True
                    else:
                        ai3_ok = True
                        self._ai3_val = 0.0

                except Exception as e3:
                    log.debug(f"  IA 3 (MLP) erro: {e3}")
            else:
                log.info(f"  IA 3 (MLP): aguardando dados ({n}/{AI3_MIN_SAMPLES})")

            if ai3_ok and ai3 is not None:
                self._ai3 = ai3
                self._ai3_ready = True

            # ── Resumo ──
            ativas = []
            if ai1_ok: ativas.append("IA 1")
            if ai2_ok: ativas.append("IA 2")
            if ai3_ok: ativas.append("IA 3")

            if len(ativas) >= 2:
                log.info(f"  ✓ {' + '.join(ativas)} ATIVAS | {n} amostras")
                self._persist_model()

                # Top features (IA 1)
                try:
                    imp = ai1.feature_importances_
                    for i in np.argsort(imp)[-5:][::-1]:
                        if i < nf:
                            log.info(f"    {feature_names[i]}: {imp[i]:.3f}")
                except Exception:
                    pass
            elif len(ativas) == 1:
                log.info(f"  ⚠ Apenas {ativas[0]} ativa")
            else:
                log.info(f"  ✗ Nenhuma IA ativa — aguardando mais dados")

        except Exception as e:
            log.error(f"Erro no treino: {e}")

    # ──────────────────────────────────────────────
    #  PERSISTÊNCIA
    # ──────────────────────────────────────────────

    def _persist_model(self):
        try:
            path = MODEL_PERSIST_FILE.replace("{broker}", self.broker)
            save_data = {
                "ai1": self._ai1,
                "ai2": self._ai2,
                "ai1_val": self._ai1_val,
                "ai2_val": self._ai2_val,
                "timestamp": time.time(),
                "n_samples": len(self._train_data),
                "n_features": getattr(self, '_n_features_trained', len(FEATURE_NAMES)),
            }
            # Salvar IA 3 + scaler se disponível
            if self._ai3 is not None:
                save_data["ai3"] = self._ai3
                save_data["ai3_val"] = self._ai3_val
                if hasattr(self, '_ai3_scaler'):
                    save_data["ai3_scaler"] = self._ai3_scaler
            with open(path, "wb") as f:
                pickle.dump(save_data, f)
            log.info(f"  Modelo salvo em {path}")
        except Exception as e:
            log.debug(f"Erro ao salvar: {e}")

    def _download_model_from_github(self, path: str) -> bool:
        """Baixa modelo NN pré-treinado do GitHub se não existir localmente."""
        try:
            import urllib.request
            import urllib.error
            _url = GITHUB_MODEL_URL.replace("{broker}", self.broker)
            log.info(f"🌐 Baixando modelo NN do GitHub para {self.broker}...")
            req = urllib.request.Request(_url, headers={
                "User-Agent": "WS-Trader-IA/1.0",
            })
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
            with open(path, "wb") as f:
                f.write(raw)
            size_kb = len(raw) / 1024
            log.info(f"✅ Modelo NN baixado do GitHub ({size_kb:.0f} KB)")
            return True
        except Exception as e:
            log.warning(f"⚠️ Falha ao baixar modelo do GitHub: {e}")
            return False

    def _try_load_persisted_model(self):
        try:
            path = MODEL_PERSIST_FILE.replace("{broker}", self.broker)
            if not os.path.exists(path):
                # Tentar baixar do GitHub
                self._download_model_from_github(path)
            if not os.path.exists(path):
                return
            with open(path, "rb") as f:
                data = pickle.load(f)
            age = time.time() - data.get("timestamp", 0)
            if age > MODEL_PERSIST_MAX_AGE:
                # Modelo expirado — tentar baixar versão mais nova do GitHub
                if self._download_model_from_github(path):
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                    age = time.time() - data.get("timestamp", 0)
                    if age > MODEL_PERSIST_MAX_AGE:
                        log.info("Modelo do GitHub também expirado — será retreinado")
                        return
                else:
                    return
            _saved_nf = data.get("n_features", 0)
            if _saved_nf not in (len(DT_FEATURE_NAMES), len(FEATURE_NAMES)):
                log.info("Modelo incompatível — será retreinado")
                os.remove(path)
                return
            self._n_features_trained = _saved_nf
            self._ai1 = data.get("ai1")
            self._ai2 = data.get("ai2")
            self._ai3 = data.get("ai3")
            self._ai1_val = data.get("ai1_val", 0)
            self._ai2_val = data.get("ai2_val", 0)
            self._ai3_val = data.get("ai3_val", 0)
            if data.get("ai3_scaler"):
                self._ai3_scaler = data["ai3_scaler"]
            n = data.get("n_samples", 0)
            self._loaded_n_samples = n
            if self._ai1:
                self._ai1_ready = True
            if self._ai2:
                self._ai2_ready = True
            if self._ai3:
                self._ai3_ready = True
            ia3_str = f", IA3={self._ai3_val:.1%}" if self._ai3_ready else ""
            log.info(f"✓ Modelo carregado ({n} amostras, "
                     f"IA1={self._ai1_val:.1%}, IA2={self._ai2_val:.1%}{ia3_str})")
        except Exception:
            pass

    def force_retrain(self):
        """Forçar retreino."""
        if len(self._train_data) >= MIN_SAMPLES_ML:
            log.info(f"Retreino forçado | {len(self._train_data)} amostras")
            self._retrain()
            return True
        return False

    # ──────────────────────────────────────────────
    #  PREDIÇÃO DIRETA (usada pelo bot DT)
    # ──────────────────────────────────────────────

    def predict_current(self, df) -> dict | None:
        """Predição ML para a ÚLTIMA vela do DataFrame.

        Retorna dict com dir, confidence, p1, p2, p3 ou None se ML inativo.
        NÃO aplica filtros de stretch/wick — apenas ML puro com votação 2/3.
        """
        n = len(df)
        if n < MIN_CANDLES + 5:
            return None
        if not (self._ai1_ready and self._ai2_ready):
            return None

        O = df["open"].values
        H = df["high"].values
        L = df["low"].values
        C = df["close"].values
        atr = self._atr(H, L, C)
        if atr < 1e-10:
            return None

        idx = n - 1
        rsi_info = self._rsi_analysis(C, idx)
        feats = self._extract_features(df, idx, atr, {}, rsi_info)
        if feats is None:
            return None

        try:
            fv = pd.DataFrame([feats], columns=FEATURE_NAMES)

            p1 = self._predict_ai1(fv)
            if p1 is None:
                return None

            p2 = self._predict_ai2(fv)
            if p2 is None:
                return None

            p3 = None
            if self._ai3_ready:
                try:
                    fv_scaled = pd.DataFrame(
                        self._ai3_scaler.transform(fv),
                        columns=FEATURE_NAMES
                    )
                    p3 = self._predict_ai3(fv_scaled)
                except Exception:
                    p3 = None

            # Direção + Confiança
            ai1_call = p1 > 0.5
            ai2_call = p2 > 0.5
            ai1_conf = (p1 if ai1_call else 1 - p1) * 100
            ai2_conf = (p2 if ai2_call else 1 - p2) * 100
            ai1_dir = "CALL" if ai1_call else "PUT"
            ai2_dir = "CALL" if ai2_call else "PUT"

            ai3_dir = None
            ai3_conf = 0.0
            if p3 is not None:
                ai3_call = p3 > 0.5
                ai3_conf = (p3 if ai3_call else 1 - p3) * 100
                ai3_dir = "CALL" if ai3_call else "PUT"

            # Votação: maioria decide (2 de 3)
            votes = {"CALL": 0, "PUT": 0}
            votes[ai1_dir] += 1
            votes[ai2_dir] += 1
            if ai3_dir is not None:
                votes[ai3_dir] += 1

            direction = "CALL" if votes["CALL"] > votes["PUT"] else "PUT"
            n_agree = votes[direction]

            # Sem IA 3: as 2 devem concordar
            if ai3_dir is None:
                if ai1_dir != ai2_dir:
                    return {"dir": None, "confidence": 0, "p1": p1, "p2": p2,
                            "p3": p3, "votes": 0, "reason": "IA1 e IA2 discordam"}
            else:
                if n_agree < 2:
                    return {"dir": None, "confidence": 0, "p1": p1, "p2": p2,
                            "p3": p3, "votes": n_agree, "reason": "Votação < 2/3"}

            # Confiança ponderada
            if ai3_dir is not None:
                confidence = ai1_conf * 0.40 + ai2_conf * 0.30 + ai3_conf * 0.30
            else:
                confidence = ai1_conf * 0.6 + ai2_conf * 0.4

            return {
                "dir": direction,
                "confidence": round(confidence, 1),
                "p1": round(p1, 4),
                "p2": round(p2, 4),
                "p3": round(p3, 4) if p3 is not None else None,
                "votes": n_agree,
                "reason": None,
            }
        except Exception:
            return None

    # ──────────────────────────────────────────────
    #  DT-SPECIFIC: Treino + Predição (15 features reais)
    #  Features vêm pré-extraídas via extract_features()
    #  do ws_adaptive_brain — mesma extração do retrain_with_fix.py
    # ──────────────────────────────────────────────

    def feed_dt_features(self, features, result: int):
        """Alimenta features DT pré-extraídas (40-float) com resultado Win=1/Loss=0.

        As features devem vir de ws_adaptive_brain.extract_features().
        Aceita 26 (legacy) ou 40 features. SEM retreinar.
        """
        if features is None:
            return 0
        feats = list(features) if not isinstance(features, list) else features
        nf_expected = len(DT_FEATURE_NAMES)
        # Backward compat: aceitar 26 legacy → pad com 0.5
        if len(feats) == 26 and nf_expected == 40:
            feats = feats + [0.5] * (nf_expected - 26)
        if len(feats) != nf_expected:
            return 0
        self._record(feats, result)
        return 1

    def predict_dt(self, features) -> dict | None:
        """Prediz Win/Loss para features DT (40-float).

        As features devem vir de ws_adaptive_brain.extract_features().
        Aceita 26 (legacy) ou 40 features.
        Retorna dict com prob_win, confidence, p1, p2, p3, votes ou None.
        """
        if not (self._ai1_ready and self._ai2_ready):
            return None

        if features is None:
            return None
        feats = list(features) if not isinstance(features, list) else features
        nf_expected = len(DT_FEATURE_NAMES)
        # Backward compat: aceitar 26 legacy → pad com 0.5
        if len(feats) == 26 and nf_expected == 40:
            feats = feats + [0.5] * (nf_expected - 26)
        # Aceitar modelo treinado com 26 features recebendo 40
        _trained_nf = getattr(self, '_n_features_trained', nf_expected)
        if _trained_nf == 26 and len(feats) == 40:
            feats = feats[:26]
            _feature_names = DT_FEATURE_NAMES[:26]
        elif len(feats) != nf_expected:
            return None
        else:
            _feature_names = DT_FEATURE_NAMES

        try:
            fv = pd.DataFrame([feats], columns=_feature_names)

            p1 = self._predict_ai1(fv)
            if p1 is None:
                return None

            p2 = self._predict_ai2(fv)
            if p2 is None:
                return None

            p3 = None
            if self._ai3_ready:
                try:
                    fv_scaled = pd.DataFrame(
                        self._ai3_scaler.transform(fv),
                        columns=_feature_names
                    )
                    p3 = self._predict_ai3(fv_scaled)
                except Exception:
                    p3 = None

            # Votação + Consenso Inteligente
            ai1_win = p1 > 0.5
            ai2_win = p2 > 0.5
            ai1_conf = (p1 if ai1_win else 1 - p1) * 100
            ai2_conf = (p2 if ai2_win else 1 - p2) * 100

            votes_win = int(ai1_win) + int(ai2_win)
            ai3_conf = 0.0
            if p3 is not None:
                ai3_win = p3 > 0.5
                ai3_conf = (p3 if ai3_win else 1 - p3) * 100
                votes_win += int(ai3_win)

            total_voters = 3 if p3 is not None else 2

            # Probabilidade de WIN ponderada
            if p3 is not None:
                prob_win = p1 * 0.40 + p2 * 0.30 + p3 * 0.30
                confidence = ai1_conf * 0.40 + ai2_conf * 0.30 + ai3_conf * 0.30
            else:
                prob_win = p1 * 0.6 + p2 * 0.4
                confidence = ai1_conf * 0.6 + ai2_conf * 0.4

            # ── TEMPERATURE SCALING ──
            # Reduz overconfidence: puxa probabilidades extremas para mais perto do centro.
            # T=1.3 → prob=0.97 vira ~0.94, prob=0.85 vira ~0.82
            _T = 1.3
            _log_odds = np.log(max(prob_win, 1e-6) / max(1 - prob_win, 1e-6))
            prob_win = float(1.0 / (1.0 + np.exp(-_log_odds / _T)))

            # ── CONSENSO INTELIGENTE ──
            # Quando modelos discordam muito (alta variância), a predição é INCERTA.
            # Penalizar nn_score proporcionalmente ao desvio padrão das probabilidades.
            # Ex: p1=0.66,p2=0.90,p3=0.95 → std=0.127 → penalty=0.047 → score=0.77 (BLOQUEADO)
            # Ex: p1=0.94,p2=0.92,p3=0.91 → std=0.012 → penalty=0    → score=0.92 (APROVADO)
            _all_probs = [p1, p2] + ([p3] if p3 is not None else [])
            _prob_std = float(np.std(_all_probs))
            _consensus_penalty = max(0.0, _prob_std - 0.08) * 1.0
            nn_score = max(0.0, prob_win - _consensus_penalty)

            is_win = votes_win > (total_voters / 2)

            return {
                "win": is_win,
                "prob_win": round(float(prob_win), 4),
                "nn_score": round(float(nn_score), 4),
                "consensus_penalty": round(float(_consensus_penalty), 4),
                "confidence": round(float(confidence), 1),
                "p1": round(float(p1), 4),
                "p2": round(float(p2), 4),
                "p3": round(float(p3), 4) if p3 is not None else None,
                "votes_win": votes_win,
                "total_voters": total_voters,
            }
        except Exception:
            return None

    def feed_candles(self, df, asset: str = ""):
        """Alimenta dados de treino a partir de velas (SEM retreinar).

        Apenas coleta amostras. O treino é feito por train_all().
        """
        n = len(df)
        if n < MIN_CANDLES + 5:
            return 0

        O = df["open"].values
        H = df["high"].values
        L = df["low"].values
        C = df["close"].values
        atr = self._atr(H, L, C)
        if atr < 1e-10:
            return 0

        added = 0
        for idx in range(MIN_CANDLES, n):
            if idx + FUTURE_CANDLES >= n:
                break
            rsi_info = self._rsi_analysis(C, idx)
            feats = self._extract_features(df, idx, atr, {}, rsi_info)
            if feats is None:
                continue
            fc = C[idx + FUTURE_CANDLES]
            if fc != C[idx]:
                label = 1 if fc > C[idx] else 0
                self._record(feats, label)
                added += 1

        return added

    def train_all(self, force=False):
        """Treina as 3 redes neurais com todos os dados coletados e persiste.

        Chamar UMA vez após feed_candles de todos os ativos.
        PROTEÇÃO: Se já tem modelo treinado offline (>1000 amostras), NÃO sobreescreve.
        Use force=True para retreino via train_neural_network.py.
        """
        # ── PROTEÇÃO: modelo treinado offline é sagrado ──
        if not force and self._ai1_ready and self._ai2_ready:
            _persist_path = MODEL_PERSIST_FILE.replace("{broker}", self.broker)
            if os.path.exists(_persist_path):
                try:
                    with open(_persist_path, "rb") as _pf:
                        _pdata = pickle.load(_pf)
                    _p_samples = _pdata.get("n_samples", 0)
                    if _p_samples >= 1000:
                        log.info(f"  🛡️ NN: Modelo offline PROTEGIDO ({_p_samples} amostras, "
                                 f"IA1={self._ai1_val:.1%} IA2={self._ai2_val:.1%}) — NÃO sobreescreve")
                        return True
                except Exception:
                    pass

        n = len(self._train_data)
        if n < MIN_SAMPLES_ML:
            log.info(f"  ⚠️ NN train_all: poucos dados ({n}/{MIN_SAMPLES_ML})")
            return False

        log.info(f"  🧠 NN: Treinando 3 modelos com {n} amostras...")
        self._retrain()

        ativas = []
        if self._ai1_ready: ativas.append(f"IA1={self._ai1_val:.1%}")
        if self._ai2_ready: ativas.append(f"IA2={self._ai2_val:.1%}")
        if self._ai3_ready: ativas.append(f"IA3={self._ai3_val:.1%}")

        if ativas:
            self._persist_model()
            log.info(f"  ✅ NN: {' | '.join(ativas)} — modelo salvo em disco")
            return True
        else:
            log.info(f"  ⚠️ NN: Nenhum modelo atingiu acurácia mínima")
            return False

    # ──────────────────────────────────────────────
    #  COMPATIBILIDADE (Bullex bot)
    # ──────────────────────────────────────────────

    @property
    def model(self):
        """Retorna modelo IA 1 (para verificação `model is None`)."""
        return self._ai1

    def save_stats_to_disk(self):
        """Salva stats no disco (compatibilidade com Bullex bot)."""
        try:
            import json
            path = os.path.join(_user_data_dir, f"ws_ai_stats_{self.broker}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.get_stats(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ──────────────────────────────────────────────
    #  STATS
    # ──────────────────────────────────────────────

    def get_stats(self) -> dict:
        n = len(self._train_data)
        # Se modelo foi carregado do disco (.pkl), usar n_samples salvo
        n_display = n if n > 0 else self._loaded_n_samples
        if not n_display and not self._ai1_ready:
            return {"ml": False, "samples": 0, "total": 0, "wr": 0, "ai1_ready": False, "ai2_ready": False}
        wins = sum(1 for s in self._train_data if s["l"] == 1) if n > 0 else 0
        return {
            "ml": self._ai1_ready and self._ai2_ready,
            "samples": n_display,
            "total": n_display,
            "wins": wins,
            "losses": n_display - wins if n > 0 else 0,
            "wr": round(wins / n * 100, 1) if n > 0 else 0,
            "ai1_val": round(self._ai1_val * 100, 1),
            "ai2_val": round(self._ai2_val * 100, 1),
            "ai3_val": round(self._ai3_val * 100, 1),
            "ai1_ready": self._ai1_ready,
            "ai2_ready": self._ai2_ready,
            "ai3_ready": self._ai3_ready,
        }


# ═══════════════════════════════════════════════════════
#  PER-ASSET INSTANCES
# ═══════════════════════════════════════════════════════
_instances: dict = {}  # {broker_or_asset: ReversalAI}
_instance_lock = threading.Lock()


def get_reversal_ai(broker: str = "iq") -> ReversalAI:
    """Retorna instância ReversalAI para o broker/ativo especificado."""
    if broker not in _instances:
        with _instance_lock:
            if broker not in _instances:
                _instances[broker] = ReversalAI(broker)
    return _instances[broker]
