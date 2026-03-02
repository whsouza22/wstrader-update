"""
WS Trader - IA ML Pura (Reversal-Only)

Dois modelos de Machine Learning independentes:
  IA 1 (Geradora):   GradientBoosting - analisa 32 features - preve CALL ou PUT
  IA 2 (Validadora): LightGBM         - confirma ou rejeita a IA 1

ESTRATEGIA: REVERSAL-ONLY
  - Entra SOMENTE quando o preco esta ESTICADO (overextended)
  - Preco la em cima -> PUT (reversao pra baixo)
  - Preco la embaixo -> CALL (reversao pra cima)
  - A IA ML confirma se os padroes de vela indicam reversao
  - Sem regras manuais de padrao de vela
  - A IA aprende sozinha os padroes a partir dos dados
  - 5 features de CONFIRMACAO (stretch × vela) permitem a IA
    descobrir sozinha quais padroes de vela revertem o preco
  - Entrada SOMENTE em extremos de preco
  - Sinal emitido quando as duas IAs concordam com a reversao
  - Retreino automatico a cada 20 novos dados
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
TRAINING_WINDOW     = 3000     # Máx. dados de treino
VALIDATION_SPLIT    = 0.20
MIN_VALIDATION_ACC  = 0.505    # Acurácia mín. para ativar (pure ML)

# ── Confiança mínima ──
AI1_CONF_MIN        = 52.0     # IA 1 precisa >= 52%
AI2_CONF_MIN        = 51.0     # IA 2 precisa >= 51%

# ── Persistência ──
_user_data_dir = os.path.join(os.path.expanduser("~"), ".wstrader")
os.makedirs(_user_data_dir, exist_ok=True)
MODEL_PERSIST_FILE    = os.path.join(_user_data_dir, "reversal_tf_{broker}.pkl")
MODEL_PERSIST_MAX_AGE = 12 * 3600   # 12 h

# ═══════════════════════════════════════════════════════
#  FEATURES  — 32 features puras de mercado
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
#  REVERSAL AI  —  IA ML Pura (2 modelos)
# ═══════════════════════════════════════════════════════
class ReversalAI:

    def __init__(self, broker: str = "iq"):
        self.broker = broker
        self.history = []              # Alias para compatibilidade
        self._train_data = []          # [{"f": [30], "l": 0/1, "ts": float}]
        self._ai1 = None               # GradientBoosting (Geradora)
        self._ai2 = None               # LightGBM (Validadora)
        self._ai1_ready = False
        self._ai2_ready = False
        self._ai1_val = 0.0
        self._ai2_val = 0.0
        self._new_samples = 0
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

                # Direção + Confiança
                ai1_call = p1 > 0.5
                ai2_call = p2 > 0.5
                ai1_conf = (p1 if ai1_call else 1 - p1) * 100
                ai2_conf = (p2 if ai2_call else 1 - p2) * 100
                ai1_dir = "CALL" if ai1_call else "PUT"
                ai2_dir = "CALL" if ai2_call else "PUT"

                # As duas IAs devem concordar
                if ai1_dir != ai2_dir:
                    continue
                if ai1_conf < AI1_CONF_MIN or ai2_conf < AI2_CONF_MIN:
                    continue

                direction = ai1_dir
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

        # ── Auto-retrain ──
        if (collect_data
                and self._new_samples >= RETRAIN_EVERY
                and len(self._train_data) >= MIN_SAMPLES_ML):
            self._retrain()
            self._new_samples = 0

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
        """Treina IA 1 (GradientBoosting) + IA 2 (LightGBM) com validação temporal."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            try:
                from lightgbm import LGBMClassifier
                _lgbm_ok = True
            except ImportError:
                from sklearn.ensemble import ExtraTreesClassifier
                _lgbm_ok = False

            nf = len(FEATURE_NAMES)
            data = [s for s in self._train_data[-TRAINING_WINDOW:]
                    if len(s["f"]) == nf]
            if len(data) < MIN_SAMPLES_ML:
                log.info(f"Aguardando dados ({len(data)}/{MIN_SAMPLES_ML})")
                return

            X = np.array([s["f"] for s in data])
            y = np.array([s["l"] for s in data])
            if len(np.unique(y)) < 2:
                return

            X_df = pd.DataFrame(X, columns=FEATURE_NAMES)
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

            # ── IA 1: GradientBoosting (Geradora) ──
            ai1 = GradientBoostingClassifier(
                n_estimators=120, max_depth=4, learning_rate=0.08,
                subsample=0.8, min_samples_leaf=10, random_state=42)
            ai1.fit(Xt, yt, sample_weight=w)

            # ── IA 2: LightGBM / ExtraTrees (Validadora) ──
            if _lgbm_ok:
                ai2 = LGBMClassifier(
                    n_estimators=120, max_depth=4, learning_rate=0.08,
                    subsample=0.8, min_child_samples=15,
                    random_state=99, verbose=-1)
                ai2.fit(Xt, yt, sample_weight=w)
            else:
                ai2 = ExtraTreesClassifier(
                    n_estimators=80, max_depth=6,
                    min_samples_leaf=10, random_state=99)
                ai2.fit(Xt, yt, sample_weight=w)

            # ── Validação ──
            ai1_ok = True
            ai2_ok = True
            if Xv is not None:
                # IA 1
                pred1 = (ai1.predict_proba(Xv)[:, 1] >= 0.5).astype(int)
                acc1 = float(np.mean(pred1 == yv))
                self._ai1_val = acc1
                # IA 2
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

            if ai1_ok and ai2_ok:
                log.info(f"  ✓ IA 1 + IA 2 ATIVAS | {n} amostras")
                self._persist_model()

                # Top features (IA 1)
                try:
                    imp = ai1.feature_importances_
                    for i in np.argsort(imp)[-5:][::-1]:
                        if i < nf:
                            log.info(f"    {FEATURE_NAMES[i]}: {imp[i]:.3f}")
                except Exception:
                    pass
            elif ai1_ok or ai2_ok:
                log.info(f"  ⚠ Apenas {'IA 1' if ai1_ok else 'IA 2'} ativa")
            else:
                log.info(f"  ✗ Ambas IAs abaixo do mínimo — aguardando mais dados")

        except Exception as e:
            log.error(f"Erro no treino: {e}")

    # ──────────────────────────────────────────────
    #  PERSISTÊNCIA
    # ──────────────────────────────────────────────

    def _persist_model(self):
        try:
            path = MODEL_PERSIST_FILE.replace("{broker}", self.broker)
            with open(path, "wb") as f:
                pickle.dump({
                    "ai1": self._ai1,
                    "ai2": self._ai2,
                    "ai1_val": self._ai1_val,
                    "ai2_val": self._ai2_val,
                    "timestamp": time.time(),
                    "n_samples": len(self._train_data),
                    "n_features": len(FEATURE_NAMES),
                }, f)
            log.info(f"  Modelo salvo em {path}")
        except Exception as e:
            log.debug(f"Erro ao salvar: {e}")

    def _try_load_persisted_model(self):
        try:
            path = MODEL_PERSIST_FILE.replace("{broker}", self.broker)
            if not os.path.exists(path):
                return
            with open(path, "rb") as f:
                data = pickle.load(f)
            age = time.time() - data.get("timestamp", 0)
            if age > MODEL_PERSIST_MAX_AGE:
                return
            if data.get("n_features", 0) != len(FEATURE_NAMES):
                log.info("Modelo incompatível — será retreinado")
                os.remove(path)
                return
            self._ai1 = data.get("ai1")
            self._ai2 = data.get("ai2")
            self._ai1_val = data.get("ai1_val", 0)
            self._ai2_val = data.get("ai2_val", 0)
            n = data.get("n_samples", 0)
            if self._ai1:
                self._ai1_ready = True
            if self._ai2:
                self._ai2_ready = True
            log.info(f"✓ Modelo carregado ({n} amostras, "
                     f"IA1={self._ai1_val:.1%}, IA2={self._ai2_val:.1%})")
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
        if not n:
            return {"ml": False, "samples": 0, "total": 0, "wr": 0, "ai1_ready": False, "ai2_ready": False}
        wins = sum(1 for s in self._train_data if s["l"] == 1)
        return {
            "ml": self._ai1_ready and self._ai2_ready,
            "samples": n,
            "total": n,
            "wins": wins,
            "losses": n - wins,
            "wr": round(wins / n * 100, 1),
            "ai1_val": round(self._ai1_val * 100, 1),
            "ai2_val": round(self._ai2_val * 100, 1),
            "ai1_ready": self._ai1_ready,
            "ai2_ready": self._ai2_ready,
        }


# ═══════════════════════════════════════════════════════
#  SINGLETON
# ═══════════════════════════════════════════════════════
_instance = None
_instance_lock = threading.Lock()


def get_reversal_ai(broker: str = "iq") -> ReversalAI:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ReversalAI(broker)
    return _instance
