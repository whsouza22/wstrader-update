# -*- coding: utf-8 -*-
"""
WS_ADAPTIVE_BRAIN — Cérebro Pensante para Trading
═══════════════════════════════════════════════════
Em vez de regras fixas (wick < 30% → BLOQUEIA), o cérebro:
  1. Extrai 15 features numéricas de cada sinal
  2. Compara com os 89K+ trades do treinamento (kNN + Regressão Logística)
  3. Calcula a probabilidade REAL de WIN baseada em trades SIMILARES
  4. Após cada trade, aprende o resultado (online learning)
  5. Roda em <1ms — tudo local, sem API

Features extraídas:
  f0:  wick_ratio       — tamanho do wick de rejeição (0-1)
  f1:  close_position   — posição do close na vela (0=bottom, 1=top)
  f2:  candles_ago      — quantas velas desde o toque (1-5)
  f3:  depth_ratio      — profundidade do padrão / ATR
  f4:  symmetry         — simetria temporal (0-1)
  f5:  span             — largura total do padrão (velas)
  f6:  shoulder_ratio   — similaridade dos toques (0.99-1.0)
  f7:  progress_pct     — % do caminho RS→Neck já andado (0-100)
  f8:  ema_diff         — EMA8-EMA20 normalizado pelo ATR
  f9:  momentum_score   — força do momentum (-1 a +1)
  f10: zone_touches     — quantas vezes nível foi tocado antes
  f11: arm_wr           — WR histórico do ativo+tipo+modo (0-1)
  f12: approach_decay   — desaceleração do momentum na chegada ao nível (0-1)
  f13: rejection_quality— qualidade da rejeição (wicks + corpo confirmação) (0-1)
  f14: volatility_regime— regime de volatilidade (0=explosivo, 1=calmo) (0-1)
"""
import json
import logging
import os
import numpy as np
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("WS_BRAIN")

# ═══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════
BRAIN_FILE = os.path.join(os.path.expanduser("~"), ".wstrader", "ws_brain_weights.json")
MIN_PROB_ENTRY = 0.62        # Probabilidade mínima para entrar (sobe conforme aprende)
MIN_SIMILAR_TRADES = 10      # Mínimo de trades similares para decidir
KNN_K = 30                   # Quantos vizinhos mais próximos considerar
LEARNING_RATE = 0.01         # Taxa de aprendizado online
HARD_BLOCK_CANDLES_AGO_0 = False  # Permite vela atual (candles_ago=0)


# ═══════════════════════════════════════════════════════════════
# EXTRAÇÃO DE FEATURES (transforma sinal em vetor numérico)
# ═══════════════════════════════════════════════════════════════
def extract_features(pat: dict, H, L, C_arr, O, n: int, atr: float,
                     hs_stats: dict, ativo: str) -> Optional[np.ndarray]:
    """Extrai 15 features numéricas de um sinal detectado.
    IMPORTANTE: Usa APENAS dados até o RS (sem velas futuras).
    Retorna None se dados insuficientes."""
    if n < 25 or atr <= 0:
        return None

    direction = pat.get("direction", "")
    rs_price = pat.get("right_shoulder", {}).get("price", 0)
    neckline = pat.get("neckline", 0)
    candles_ago = pat.get("candles_ago", 0)
    pat_type = pat.get("type", "")
    mode = pat.get("mode", "double_touch")

    # Índice local do RS dentro do array
    rs_idx = n - 1 - candles_ago
    if rs_idx < 0 or rs_idx >= n or rs_price == 0:
        return None

    # ── f0: wick_ratio (rejeição na vela do toque) ──
    h_rs = float(H[rs_idx])
    l_rs = float(L[rs_idx])
    c_rs = float(C_arr[rs_idx])
    o_rs = float(O[rs_idx])
    candle_range = h_rs - l_rs
    if candle_range < 1e-10:
        return None

    if direction == "PUT":
        wick = h_rs - max(c_rs, o_rs)
    else:
        wick = min(c_rs, o_rs) - l_rs
    wick_ratio = wick / candle_range

    # ── f1: close_position ──
    close_position = (c_rs - l_rs) / candle_range

    # ── f2: candles_ago → SUBSTITUÍDO por macro_against_pct ──
    # candles_ago era sempre 0 no treino (feature morta)
    # NOVO: % de velas contra a direção do trade nas últimas 15 antes do RS
    _macro_w = min(15, rs_idx)
    macro_against_pct = 0.5
    if _macro_w >= 3:
        _against_count = 0
        for _mi in range(rs_idx - _macro_w, rs_idx):
            _m_body = float(C_arr[_mi]) - float(O[_mi])
            if (direction == "CALL" and _m_body < 0) or (direction == "PUT" and _m_body > 0):
                _against_count += 1
        macro_against_pct = _against_count / _macro_w

    # ── f3-f6: geometria do padrão ──
    try:
        iL = pat["left_shoulder"]["idx"]
        iH = pat["head"]["idx"]
        iR = pat["right_shoulder"]["idx"]
        depth = pat.get("depth", abs(pat["head"]["price"] - rs_price))
        span = max(iR - iL, 1)
        d_left = max(iH - iL, 1)
        d_right = max(iR - iH, 1)
        symmetry = min(d_left, d_right) / max(d_left, d_right)
        depth_ratio = depth / atr
    except (KeyError, TypeError):
        return None

    # normalizar
    depth_ratio_norm = min(depth_ratio, 25.0) / 25.0
    span_norm = min(span, 200) / 200.0

    # ── f6: shoulder_ratio → SUBSTITUÍDO por macro_body_avg ──
    # shoulder_ratio era sempre 1.0 no DT (feature morta)
    # NOVO: tamanho médio dos corpos nas últimas 10 velas / ATR
    _body_w = min(10, rs_idx)
    macro_body_avg = 0.0
    if _body_w >= 2 and atr > 1e-10:
        _body_sum = 0.0
        for _bi in range(rs_idx - _body_w, rs_idx):
            _body_sum += abs(float(C_arr[_bi]) - float(O[_bi]))
        macro_body_avg = np.clip((_body_sum / _body_w) / atr, 0, 2.0) / 2.0

    # ── f7: progress_pct (quanto o preço andou do RS→Neck) ──
    # Usa preço no momento do RS (não preço atual que pode ser futuro)
    rs_to_neck = abs(neckline - rs_price) if neckline > 0 else 0
    if rs_to_neck > 0:
        cur_price = float(C_arr[min(rs_idx, n - 1)])
        dist_to_rs = abs(cur_price - rs_price)
        progress_pct = min(dist_to_rs / rs_to_neck, 1.0)
    else:
        progress_pct = 0.0

    # ── f8: ema_diff (tendência ATÉ o RS — sem dados futuros) ──
    _ema_end = rs_idx + 1  # inclusive do RS, exclusivo acima
    closes = [float(C_arr[i]) for i in range(max(0, _ema_end - 30), _ema_end)]
    ema8 = _ema(closes, 8)
    ema20 = _ema(closes, 20)
    if ema8 is not None and ema20 is not None:
        ema_diff = (ema8 - ema20) / atr
        if direction == "PUT":
            ema_score = np.clip(ema_diff / 3.0, -1, 1)
        else:
            ema_score = np.clip(-ema_diff / 3.0, -1, 1)
    else:
        ema_score = 0.0

    # ── f9: momentum (5 velas ANTES do RS — sem dados futuros) ──
    momentum = 0.0
    _mom_n = min(5, rs_idx)
    if _mom_n >= 2:
        for i in range(rs_idx - _mom_n, rs_idx):
            c_i = float(C_arr[i])
            o_i = float(O[i])
            body = (c_i - o_i) / atr
            if direction == "PUT":
                momentum -= body
            else:
                momentum += body
        momentum = np.clip(momentum / _mom_n, -1, 1)

    # ── f10: zone_touches (histórico de toques no nível — antes do padrão) ──
    tol = atr * 0.5
    try:
        iL_orig = pat["left_shoulder"]["idx"]
        iR_orig = pat["right_shoulder"]["idx"]
        span_candles = max(iR_orig - iL_orig, 1)
    except (KeyError, TypeError):
        span_candles = 20
    ls_idx_local = max(0, rs_idx - span_candles)
    start_idx = max(0, ls_idx_local - 100)
    touches = 0
    for i in range(start_idx, min(ls_idx_local, n)):
        if direction == "PUT" and abs(float(H[i]) - rs_price) <= tol:
            touches += 1
        elif direction == "CALL" and abs(float(L[i]) - rs_price) <= tol:
            touches += 1
    zone_score = min(touches, 10) / 10.0

    # ── f11: arm_wr (WR histórico) ──
    arm_key = f"{ativo}_{pat_type}_{mode}"
    arm = hs_stats.get("arms", {}).get(arm_key, {})
    total = arm.get("total", 0)
    wins = arm.get("wins", 0)
    arm_wr = wins / total if total >= 30 else 0.5

    # ── f12: approach_decay (desaceleração do momentum na chegada) ──
    approach_decay = 0.5
    if rs_idx >= 3:
        bodies_before = []
        for bi in range(max(0, rs_idx - 5), rs_idx):
            b = abs(float(C_arr[bi]) - float(O[bi]))
            bodies_before.append(b)
        if len(bodies_before) >= 2:
            body_rs = abs(c_rs - o_rs)
            body_avg_before = sum(bodies_before) / len(bodies_before)
            if body_avg_before > 1e-10:
                decay_ratio = body_rs / body_avg_before
                approach_decay = np.clip(1.0 - decay_ratio, 0.0, 1.0)

    # ── f13: rejection_quality (REDESENHADO — sem vela futura) ──
    # Agora baseado SOMENTE em dados disponíveis no momento da decisão:
    # 1. Wick ratio do RS (rejeição do nível) — 30%
    # 2. Corpo do RS contra a direção (red em PUT = vendedores rejeitaram) — 40%
    # 3. Doji detection: corpo muito pequeno = indecisão = ruim — 30%
    rq_components = []
    # Componente 1: wick de rejeição (já calculado)
    rq_components.append(wick_ratio * 0.3)

    # Componente 2: corpo do RS mostra rejeição?
    # PUT: close < open (vela vermelha/bearish) = vendedores dominaram = bom
    # CALL: close > open (vela verde/bullish) = compradores dominaram = bom
    body_rs = c_rs - o_rs
    body_abs = abs(body_rs)
    if candle_range > 1e-10:
        body_ratio = body_abs / candle_range  # quão grande é o corpo vs range

        if direction == "PUT":
            # Vela vermelha (close < open) = rejeição bearish = bom para PUT
            if body_rs < 0:
                # Corpo bearish — quanto maior melhor
                rs_body_score = min(body_ratio * 2, 1.0)
            else:
                # Corpo bullish = CONTRA a reversão PUT = ruim
                rs_body_score = 0.0
        else:  # CALL
            if body_rs > 0:
                rs_body_score = min(body_ratio * 2, 1.0)
            else:
                rs_body_score = 0.0

        # Componente 3: Doji = corpo < 20% do range = indecisão = penalizar
        # Doji não mostra convicção de rejeição
        if body_ratio < 0.20:
            doji_penalty = 1.0 - (body_ratio / 0.20)  # 1.0 = doji puro, 0 = normal
            # Pavio grande + doji = parcialmente OK (pin bar)
            if wick_ratio > 0.5:
                doji_penalty *= 0.3  # Pin bar com pavio grande: reduce penalty
        else:
            doji_penalty = 0.0
    else:
        rs_body_score = 0.0
        doji_penalty = 1.0

    rq_components.append(rs_body_score * 0.4)

    # Wicks de rejeição nas velas ANTES do RS (sem usar dados futuros)
    wick_reject_count = 0
    for wi in range(max(0, rs_idx - 4), rs_idx + 1):  # até RS inclusive, sem rs+1
        w_range = float(H[wi]) - float(L[wi])
        if w_range < 1e-10:
            continue
        if direction == "PUT":
            w_rej = float(H[wi]) - max(float(C_arr[wi]), float(O[wi]))
        else:
            w_rej = min(float(C_arr[wi]), float(O[wi])) - float(L[wi])
        if w_rej / w_range > 0.35:
            wick_reject_count += 1
    rq_components.append(min(wick_reject_count, 4) / 4.0 * 0.3)

    rejection_quality = sum(rq_components) * (1.0 - doji_penalty * 0.5)

    # ── f14: volatility_regime (ATÉ o RS — sem dados futuros) ──
    volatility_regime = 0.5
    if rs_idx >= 10:
        atr_recent_vals = [float(H[k]) - float(L[k]) for k in range(max(0, rs_idx - 5), rs_idx + 1)]
        atr_long_vals = [float(H[k]) - float(L[k]) for k in range(max(0, rs_idx - 50), rs_idx + 1)]
        atr_recent = sum(atr_recent_vals) / len(atr_recent_vals) if atr_recent_vals else atr
        atr_long = sum(atr_long_vals) / len(atr_long_vals) if atr_long_vals else atr
        if atr_long > 1e-10:
            vol_ratio = atr_recent / atr_long
            volatility_regime = np.clip(1.0 - (vol_ratio - 0.7) / 1.3, 0.0, 1.0)

    # ── f15: trend_strength (tendência de médio prazo — EMA10 vs EMA50) ──
    # Captura se o trade é a FAVOR ou CONTRA a tendência de 50 velas
    trend_strength = 0.0
    _trend_end = rs_idx + 1
    closes_long = [float(C_arr[i]) for i in range(max(0, _trend_end - 100), _trend_end)]
    ema10_long = _ema(closes_long, 10)
    ema50_long = _ema(closes_long, 50)
    if ema10_long is not None and ema50_long is not None and atr > 1e-10:
        trend_raw = (ema10_long - ema50_long) / atr
        if direction == "PUT":
            trend_strength = np.clip(-trend_raw / 5.0, -1, 1)
        else:
            trend_strength = np.clip(trend_raw / 5.0, -1, 1)

    # ── f16: price_vs_ema50 (distância do preço à EMA50) ──
    # Preço longe da EMA50 na direção do trade = sobreextensão = bom para reversão
    price_vs_ema50 = 0.0
    if ema50_long is not None and atr > 1e-10:
        dist_ema = (float(C_arr[rs_idx]) - ema50_long) / atr
        if direction == "PUT":
            price_vs_ema50 = np.clip(dist_ema / 5.0, -1, 1)
        else:
            price_vs_ema50 = np.clip(-dist_ema / 5.0, -1, 1)

    # ── f17: consecutive_against (velas consecutivas CONTRA o trade) ──
    # Conta quantas velas seguidas (do RS para trás) têm corpo contra a direção
    consecutive_against = 0
    for _ca_i in range(rs_idx - 1, max(rs_idx - 6, -1), -1):
        if _ca_i < 0:
            break
        _ca_body = float(C_arr[_ca_i]) - float(O[_ca_i])
        # PUT quer queda: corpo positivo (bullish) = contra
        # CALL quer alta: corpo negativo (bearish) = contra
        _is_against = (_ca_body > 0 and direction == "PUT") or \
                      (_ca_body < 0 and direction == "CALL")
        if _is_against:
            consecutive_against += 1
        else:
            break  # parou a sequência
    consecutive_against_norm = min(consecutive_against, 5) / 5.0

    # ── f18: body_dominance (pressão líquida na direção do trade) ──
    # Soma dos corpos a favor - contra, últimas 5 velas / ATR
    body_dominance = 0.0
    _bd_n = min(5, rs_idx)
    if _bd_n >= 1 and atr > 1e-10:
        _bd_sum = 0.0
        for _bd_i in range(rs_idx - _bd_n, rs_idx):
            _bd_body = float(C_arr[_bd_i]) - float(O[_bd_i])
            if direction == "PUT":
                _bd_sum -= _bd_body  # corpo bearish = positivo para PUT
            else:
                _bd_sum += _bd_body  # corpo bullish = positivo para CALL
        body_dominance = np.clip(_bd_sum / (_bd_n * atr), -1, 1)

    # ── f19: macro_drift_against — drift contra o trade nas últimas 15 velas / ATR ──
    macro_drift_against = 0.0
    _drift_w = min(15, rs_idx)
    if _drift_w >= 3 and atr > 1e-10:
        _drift_raw = float(C_arr[rs_idx]) - float(C_arr[max(0, rs_idx - _drift_w)])
        if direction == "CALL":
            macro_drift_against = np.clip(-_drift_raw / (atr * 5.0), -1, 1)
        else:
            macro_drift_against = np.clip(_drift_raw / (atr * 5.0), -1, 1)

    # ── f20: rsi_at_rs — RSI(14) ajustado pela direção ──
    rsi_value = 0.5
    _rsi_n = min(15, rs_idx + 1)
    if _rsi_n >= 5:
        _rsi_closes = [float(C_arr[i]) for i in range(rs_idx - _rsi_n + 1, rs_idx + 1)]
        _deltas = [_rsi_closes[j+1] - _rsi_closes[j] for j in range(len(_rsi_closes)-1)]
        _gains = [d if d > 0 else 0 for d in _deltas]
        _losses_rsi = [-d if d < 0 else 0 for d in _deltas]
        _avg_g = sum(_gains) / len(_gains) if _gains else 0
        _avg_l = sum(_losses_rsi) / len(_losses_rsi) if _losses_rsi else 0
        if _avg_l > 1e-10:
            _rs_val = _avg_g / _avg_l
            rsi_value = 1.0 - 1.0 / (1.0 + _rs_val)
        elif _avg_g > 0:
            rsi_value = 1.0
    # Ajustar pela direção: para PUT, RSI alto = bom (sobre-comprado)
    # para CALL, RSI baixo = bom (sobre-vendido)
    if direction == "PUT":
        rsi_dir_adjusted = rsi_value  # alto = bom para PUT
    else:
        rsi_dir_adjusted = 1.0 - rsi_value  # baixo = bom para CALL

    # ── f21: candle_range_vs_atr — tamanho da vela RS vs ATR ──
    candle_range_ratio = np.clip(candle_range / atr, 0, 3.0) / 3.0 if atr > 1e-10 else 0.5

    # ── f22: acceleration — aceleração/desaceleração do momentum ──
    # Compara momentum dos últimos 5 candles vs 5 anteriores
    acceleration = 0.0
    if rs_idx >= 10 and atr > 1e-10:
        _mom_recent = 0.0
        _mom_older = 0.0
        for _ai in range(rs_idx - 5, rs_idx):
            _ab = abs(float(C_arr[_ai]) - float(O[_ai]))
            _mom_recent += _ab
        for _ai in range(rs_idx - 10, rs_idx - 5):
            _ab = abs(float(C_arr[_ai]) - float(O[_ai]))
            _mom_older += _ab
        _mom_recent /= 5.0
        _mom_older /= 5.0
        if _mom_older > 1e-10:
            acceleration = np.clip((_mom_recent - _mom_older) / _mom_older, -1, 1)

    # ── f23: n_pivots_at_level — quantos pivots no mesmo nível de preço ──
    # DT normal = 2 pivots (LS + RS). 3+ indica nível já testado/desgastado.
    # NN aprende que mais pivots = menos chance de reversão.
    _n_piv_raw = pat.get("n_pivots_at_level", 2)
    n_pivots_norm = min(_n_piv_raw, 6) / 6.0

    features = np.array([
        wick_ratio,                # f0  — wick de rejeição no RS
        close_position,            # f1  — posição do close na vela RS
        macro_against_pct,         # f2  — % velas contra trade (15 velas) [ERA candles_ago]
        depth_ratio_norm,          # f3  — profundidade / ATR
        symmetry,                  # f4  — simetria temporal
        span_norm,                 # f5  — largura do padrão
        macro_body_avg,            # f6  — corpo médio 10 velas / ATR [ERA shoulder_ratio]
        progress_pct,              # f7  — % do caminho RS→Neck
        ema_score,                 # f8  — tendência ATÉ o RS (sem futuro)
        momentum,                  # f9  — momentum 5 velas ANTES do RS
        zone_score,                # f10 — toques históricos no nível
        arm_wr,                    # f11 — WR histórico
        approach_decay,            # f12 — desaceleração na chegada
        rejection_quality,         # f13 — QUALIDADE DA REJEIÇÃO (sem leak)
        volatility_regime,         # f14 — regime de vol ATÉ o RS (sem futuro)
        trend_strength,            # f15 — tendência EMA10 vs EMA50
        price_vs_ema50,            # f16 — distância preço→EMA50
        consecutive_against_norm,  # f17 — velas consecutivas contra (5)
        body_dominance,            # f18 — pressão direcional líquida
        macro_drift_against,       # f19 — drift contra trade (15 velas) / ATR
        rsi_dir_adjusted,          # f20 — RSI(14) ajustado pela direção
        candle_range_ratio,        # f21 — tamanho da vela RS / ATR
        acceleration,              # f22 — aceleração do momentum
        n_pivots_norm,             # f23 — pivots no nível (2=DT normal, 3+=desgastado)
    ], dtype=np.float64)

    return features


# ═══════════════════════════════════════════════════════════════
# CÉREBRO ADAPTATIVO
# ═══════════════════════════════════════════════════════════════
class AdaptiveBrain:
    """Cérebro que aprende. Combina:
    1. kNN no geometry_history (busca padrões similares → WR deles)
    2. Regressão logística treinada online (pesos ajustados a cada trade)
    3. Combinação ponderada dos dois (confiança adaptativa)
    """

    def __init__(self):
        self.n_features = 15
        # Pesos da regressão logística (iniciam neutros)
        self.weights = np.zeros(self.n_features)
        self.bias = 0.0
        # Histórico de trades com features (para kNN)
        self.memory: List[Tuple[np.ndarray, int]] = []  # (features, result)
        # Contadores
        self.total_decisions = 0
        self.total_approved = 0
        self.total_blocked = 0
        # Confiança no modelo logístico vs kNN (começa 50/50)
        self.logistic_weight = 0.3  # começa confiando mais no kNN
        # Normalização das features (calculado no pré-treino)
        self._feature_means = np.zeros(self.n_features)
        self._feature_stds = np.ones(self.n_features)
        self._loaded = False

    def load(self, hs_stats: dict):
        """Carrega pesos salvos + PRÉ-TREINA com os 89K trades do treinamento."""
        if self._loaded:
            return

        # Carregar pesos salvos (se existem de sessão anterior)
        _had_saved = False
        if os.path.exists(BRAIN_FILE):
            try:
                with open(BRAIN_FILE, "r") as f:
                    data = json.load(f)
                saved_weights = np.array(data.get("weights", [0.0] * self.n_features))
                if len(saved_weights) < self.n_features:
                    saved_weights = np.concatenate([saved_weights, np.zeros(self.n_features - len(saved_weights))])
                self.weights = saved_weights[:self.n_features]
                self.bias = data.get("bias", 0.0)
                self.logistic_weight = data.get("logistic_weight", 0.3)
                self.total_decisions = data.get("total_decisions", 0)
                self.total_approved = data.get("total_approved", 0)
                self.total_blocked = data.get("total_blocked", 0)
                fm = data.get("feature_means")
                fs = data.get("feature_stds")
                if fm and fs:
                    _fm = np.array(fm)
                    _fs = np.array(fs)
                    if len(_fm) < self.n_features:
                        _fm = np.concatenate([_fm, np.full(self.n_features - len(_fm), 0.5)])
                        _fs = np.concatenate([_fs, np.ones(self.n_features - len(_fs))])
                    self._feature_means = _fm[:self.n_features]
                    self._feature_stds = _fs[:self.n_features]
                saved_mem = data.get("memory", [])
                for m in saved_mem:
                    _mf = np.array(m["f"])
                    if len(_mf) < self.n_features:
                        _mf = np.concatenate([_mf, np.full(self.n_features - len(_mf), 0.5)])
                    self.memory.append((_mf[:self.n_features], m["r"]))
                _had_saved = True
                log.info(f"  🧠 Cérebro carregado: {len(self.memory)} trades em memória, "
                         f"{self.total_decisions} decisões totais")
            except Exception as e:
                log.warning(f"  ⚠️ Erro ao carregar cérebro: {e}")

        # ═══ PRÉ-TREINAR COM OS DADOS DO TREINAMENTO (89K trades) ═══
        # O treinamento já tem a resposta: quais padrões ganham e quais perdem.
        # O cérebro aprende AGORA, não espera trades ao vivo.
        # Sempre treina LR se pesos são zero (mesmo com arquivo salvo)
        needs_lr_train = np.linalg.norm(self.weights) < 0.01
        self._pretrain(hs_stats, skip_lr=not needs_lr_train)
        self._loaded = True

    def _pretrain(self, hs_stats: dict, skip_lr: bool = False):
        """Pré-treina o cérebro com TODOS os dados disponíveis:
        1. features_15  (amostras com 15 features REAIS do backtest)
        2. geometry_history (samples com features geométricas + resultado)
        3. arms (83K+ trades com WR por ativo/tipo/modo) → gera amostras sintéticas
        4. Treina a regressão logística com mini-batch SGD (múltiplas épocas)
        """
        geo_history = hs_stats.get("geometry_history", [])
        arms = hs_stats.get("arms", {})
        features_15_data = hs_stats.get("features_15", [])

        # ── ETAPA 0: Adicionar features_15 REAIS do backtest (PRIORIDADE) ──
        f15_added = 0
        for item in features_15_data:
            feats_list, result = item
            if result not in (0, 1):
                continue
            if len(feats_list) >= self.n_features:
                fvec = np.array(feats_list[:self.n_features], dtype=np.float64)
                self.memory.append((fvec, result))
                f15_added += 1

        # ── ETAPA 1: Adicionar geometry_history à memória ──
        geo_added = 0
        for g in geo_history:
            result = g.get("result", -1)
            if result not in (0, 1):
                continue
            geo_features = self._geo_to_features(g, hs_stats)
            if geo_features is not None:
                self.memory.append((geo_features, result))
                geo_added += 1

        # ── ETAPA 2: Gerar amostras sintéticas a partir dos arms (83K trades) ──
        # Cada arm tem wins/total. Geramos samples representativos variando
        # as features de geometria ao redor dos valores típicos do treinamento.
        synth_added = 0
        rng = np.random.RandomState(42)  # determinístico
        
        for arm_key, arm_data in arms.items():
            if "DOUBLE" not in arm_key:
                continue
            total = arm_data.get("total", 0)
            wins = arm_data.get("wins", 0)
            if total < 50:
                continue
            
            wr = wins / total
            ativo = arm_key.split("_DOUBLE")[0]
            is_top = "DOUBLE_TOP" in arm_key
            
            # Gerar N amostras proporcionais (max 50 por arm — total ~950)
            n_samples = min(50, max(10, total // 100))
            n_wins = int(n_samples * wr)
            n_losses = n_samples - n_wins
            
            for i in range(n_samples):
                result = 1 if i < n_wins else 0
                feat = self._generate_synthetic_sample(rng, wr, is_top, result)
                self.memory.append((feat, result))
                synth_added += 1

        log.info(f"  🧠 PRÉ-TREINO: +{f15_added} features_15_reais + {geo_added} geo_history "
                 f"+ {synth_added} sintéticos = {len(self.memory)} total em memória")

        # ── ETAPA 3: Treinar regressão logística (se não tinha pesos salvos) ──
        if not skip_lr and len(self.memory) >= 50:
            self._train_logistic()

    def _generate_synthetic_sample(self, rng, wr: float, is_top: bool, result: int) -> np.ndarray:
        """Gera uma amostra sintética baseada no WR do arm e resultado.
        Features não-geométricas são NEUTRAS (mesma distribuição para WIN/LOSS)
        para evitar data leakage. Modelo aprende apenas de geometria real."""

        # Features não-geométricas: mesma distribuição independente do resultado
        wick = rng.uniform(0.15, 0.65)
        close_pos = rng.uniform(0.20, 0.80)
        depth = rng.uniform(0.08, 0.50)
        sym = rng.uniform(0.20, 0.90)
        progress = rng.uniform(0.01, 0.20)
        momentum = rng.uniform(-0.3, 0.4)
        zone = rng.uniform(0.1, 0.5)

        return np.array([
            wick,                          # f0: wick_ratio (neutro)
            close_pos,                     # f1: close_position (neutro)
            0.1,                           # f2: candles_ago
            depth,                         # f3: depth_ratio_norm
            sym,                           # f4: symmetry
            rng.uniform(0.05, 0.30),       # f5: span_norm
            rng.uniform(0.9995, 1.0),      # f6: shoulder_ratio
            progress,                      # f7: progress_pct (neutro)
            rng.uniform(-0.3, 0.5),        # f8: ema (neutro)
            momentum,                      # f9: momentum (neutro)
            zone,                          # f10: zone_touches (neutro)
            wr,                            # f11: arm_wr (WR real do treino)
            rng.uniform(0.3, 0.7),         # f12: approach_decay (neutro)
            rng.uniform(0.3, 0.7),         # f13: rejection_quality (neutro)
            rng.uniform(0.3, 0.7),         # f14: volatility_regime (neutro)
            rng.uniform(-0.3, 0.3),        # f15: trend_strength (neutro)
            rng.uniform(-0.3, 0.3),        # f16: price_vs_ema50 (neutro)
        ], dtype=np.float64)

    def _train_logistic(self):
        """Treina a regressão logística com todos os dados da memória.
        Múltiplas épocas, mini-batch SGD."""
        if len(self.memory) < 50:
            return

        log.info(f"  🧠 Treinando regressão logística com {len(self.memory)} amostras...")
        
        X = np.array([m[0] for m in self.memory])
        y = np.array([m[1] for m in self.memory])
        
        # Normalizar features para estabilizar o treino
        self._feature_means = X.mean(axis=0)
        self._feature_stds = X.std(axis=0)
        self._feature_stds[self._feature_stds < 1e-6] = 1.0
        X_norm = (X - self._feature_means) / self._feature_stds
        
        # SGD com múltiplas épocas
        lr = 0.05
        n_epochs = 30
        n_samples = len(X_norm)
        
        for epoch in range(n_epochs):
            # Shuffle
            idx = np.random.permutation(n_samples)
            total_loss = 0.0
            
            for i in idx:
                z = np.dot(self.weights, X_norm[i]) + self.bias
                z = np.clip(z, -10, 10)
                pred = 1.0 / (1.0 + np.exp(-z))
                error = y[i] - pred
                
                # Update (cross-entropy gradient: (y - pred) * x)
                self.weights += lr * error * X_norm[i]
                self.bias += lr * error
                
                total_loss += -y[i] * np.log(pred + 1e-10) - (1 - y[i]) * np.log(1 - pred + 1e-10)
            
            # Decay learning rate
            lr *= 0.95

        # Accuracy no treino
        correct = 0
        for i in range(n_samples):
            z = np.dot(self.weights, X_norm[i]) + self.bias
            z = np.clip(z, -10, 10)
            pred = 1.0 / (1.0 + np.exp(-z))
            if (pred >= 0.5) == (y[i] == 1):
                correct += 1
        
        acc = correct / n_samples * 100
        log.info(f"  🧠 LR treinada: accuracy={acc:.1f}% | weights_norm={np.linalg.norm(self.weights):.2f}")

    def _geo_to_features(self, geo: dict, hs_stats: dict = None) -> Optional[np.ndarray]:
        """Converte uma entrada do geometry_history em vetor de features.
        IMPORTANTE: Features não-geométricas usam ruído NEUTRO (não correlacionam
        com resultado) para forçar o modelo a aprender apenas de geometria real."""
        try:
            span = geo.get("span", 0)
            sym = geo.get("symmetry", 0)
            depth = geo.get("depth_ratio", 0)
            sr = geo.get("shoulder_ratio", 1.0)
            is_top = geo.get("type", "").endswith("TOP")
            ativo = geo.get("ativo", "")

            # Buscar WR real do arm (se disponível)
            arm_wr = 0.89  # default
            if hs_stats:
                pat_type = "DOUBLE_TOP" if is_top else "DOUBLE_BOTTOM"
                arm_key = f"{ativo}_{pat_type}_double_touch"
                arm = hs_stats.get("arms", {}).get(arm_key, {})
                t = arm.get("total", 0)
                w = arm.get("wins", 0)
                if t >= 30:
                    arm_wr = w / t

            # Ruído determinístico baseado na geometria (NÃO no resultado)
            seed = int(abs(hash((span, round(depth, 4), round(sym, 4))))) % (2**31)
            rng = np.random.RandomState(seed)

            return np.array([
                rng.uniform(0.15, 0.65),       # f0: wick_ratio (neutro)
                rng.uniform(0.20, 0.80),       # f1: close_position (neutro)
                0.1,                           # f2: candles_ago
                min(depth, 25.0) / 25.0,       # f3: depth_ratio (REAL)
                sym,                           # f4: symmetry (REAL)
                min(span, 200) / 200.0,        # f5: span (REAL)
                sr,                            # f6: shoulder_ratio (REAL)
                rng.uniform(0.01, 0.15),       # f7: progress_pct (neutro)
                rng.uniform(-0.3, 0.5),        # f8: ema_score (neutro)
                rng.uniform(-0.3, 0.4),        # f9: momentum (neutro)
                rng.uniform(0.1, 0.5),         # f10: zone_touches (neutro)
                arm_wr,                        # f11: arm_wr (REAL)
                geo.get("approach_decay", rng.uniform(0.3, 0.7)),   # f12: approach_decay (REAL se disponível)
                geo.get("rejection_quality", rng.uniform(0.3, 0.7)), # f13: rejection_quality (REAL se disponível)
                geo.get("volatility_regime", rng.uniform(0.3, 0.7)), # f14: volatility_regime (REAL se disponível)
                rng.uniform(-0.3, 0.3),        # f15: trend_strength (neutro)
                rng.uniform(-0.3, 0.3),        # f16: price_vs_ema50 (neutro)
            ], dtype=np.float64)
        except Exception:
            return None

    def decide(self, features: np.ndarray, pat: dict) -> Tuple[bool, float, Dict]:
        """Decisão do cérebro: entrar ou não?

        Returns:
            (aprovado, probabilidade, detalhes)
        """
        self.total_decisions += 1
        candles_ago = pat.get("candles_ago", 0)
        details = {}

        # ══ BLOQUEIO DURO: vela não fechada (isso NUNCA funciona) ══
        if HARD_BLOCK_CANDLES_AGO_0 and candles_ago < 1:
            self.total_blocked += 1
            return False, 0.0, {"reason": "vela_nao_fechou", "prob": 0.0}

        # ══ 1. kNN: buscar trades similares na memória ══
        knn_prob, knn_n, knn_detail = self._knn_predict(features)

        # ══ 2. Regressão Logística: predição com pesos aprendidos ══
        logistic_prob = self._logistic_predict(features)

        # ══ 3. Combinar (peso adaptativo) ══
        if knn_n >= MIN_SIMILAR_TRADES:
            combined_prob = (
                (1 - self.logistic_weight) * knn_prob +
                self.logistic_weight * logistic_prob
            )
        else:
            # Sem suficientes trades similares: confia mais na logística
            combined_prob = logistic_prob

        # Threshold FIXO — brain já vem pré-treinado com 63K amostras reais
        threshold = MIN_PROB_ENTRY  # 0.62 sempre, sem fase permissiva

        approved = combined_prob >= threshold

        if approved:
            self.total_approved += 1
        else:
            self.total_blocked += 1

        details = {
            "prob": round(combined_prob, 4),
            "knn_prob": round(knn_prob, 4),
            "knn_n": knn_n,
            "logistic_prob": round(logistic_prob, 4),
            "log_weight": round(self.logistic_weight, 3),
            "approved": approved,
        }

        # Log
        emoji = "✅" if approved else "🚫"
        log.info(
            f"  {emoji} CÉREBRO: prob={combined_prob:.1%} "
            f"(kNN={knn_prob:.1%} n={knn_n} | LR={logistic_prob:.1%}) "
            f"mín={MIN_PROB_ENTRY:.0%}"
        )

        return approved, combined_prob, details

    def learn(self, features: np.ndarray, result: int):
        """Aprende com o resultado de um trade (online learning).
        result: 1=WIN, 0=LOSS
        """
        # Adicionar à memória
        self.memory.append((features.copy(), result))

        # Limitar memória (manter últimos 2000 trades reais)
        if len(self.memory) > 2500:
            self.memory = self.memory[-2000:]

        # Atualizar pesos da regressão logística (SGD com normalização)
        x_norm = (features - self._feature_means) / self._feature_stds
        pred = self._logistic_predict(features)
        error = result - pred
        self.weights += LEARNING_RATE * error * x_norm
        self.bias += LEARNING_RATE * error

        # Ajustar confiança logística vs kNN baseado em performance recente
        recent_trades = self.memory[-50:] if len(self.memory) >= 50 else self.memory
        if len(recent_trades) >= 20:
            knn_correct = 0
            lr_correct = 0
            for feat, res in recent_trades:
                knn_p, knn_n_check, _ = self._knn_predict(feat)
                lr_p = self._logistic_predict(feat)
                if knn_n_check >= 5:
                    knn_correct += 1 if (knn_p >= 0.5) == (res == 1) else 0
                lr_correct += 1 if (lr_p >= 0.5) == (res == 1) else 0

            total = len(recent_trades)
            knn_acc = knn_correct / total
            lr_acc = lr_correct / total
            # Ajustar peso: quem acerta mais ganha mais peso
            total_acc = knn_acc + lr_acc
            if total_acc > 0:
                self.logistic_weight = np.clip(lr_acc / total_acc, 0.1, 0.9)

        # Salvar
        self.save()

        log.info(
            f"  🧠 APRENDEU: {'WIN' if result else 'LOSS'} | "
            f"mem={len(self.memory)} | lr_weight={self.logistic_weight:.2f}"
        )

    def _knn_predict(self, features: np.ndarray) -> Tuple[float, int, str]:
        """kNN: encontra os K trades mais similares e retorna o WR deles."""
        if not self.memory:
            return 0.5, 0, "sem_memoria"

        # Calcular distâncias
        distances = []
        for mem_feat, mem_result in self.memory:
            diff = features - mem_feat
            # Peso diferente por feature
            # Geometria (f3-f6) tem peso alto: dados REAIS do treino
            # Market features (f0-f2, f7-f11): peso menor quando memória é geo_history
            weights = np.array([
                1.0,  # f0: wick_ratio
                0.8,  # f1: close_position
                0.8,  # f2: candles_ago
                3.0,  # f3: depth_ratio (CHAVE — dados reais do treino)
                2.5,  # f4: symmetry (CHAVE)
                2.0,  # f5: span (CHAVE)
                1.5,  # f6: shoulder_ratio (CHAVE)
                1.0,  # f7: progress_pct
                0.8,  # f8: ema_score
                0.8,  # f9: momentum
                0.8,  # f10: zone_touches
                2.0,  # f11: arm_wr (importante)
                2.5,  # f12: approach_decay (CHAVE — desaceleração)
                3.0,  # f13: rejection_quality (CHAVE — velas respeitando)
                2.0,  # f14: volatility_regime (regime de mercado)
            ])
            dist = np.sqrt(np.sum((diff * weights) ** 2))
            distances.append((dist, mem_result))

        distances.sort(key=lambda x: x[0])
        k = min(KNN_K, len(distances))
        neighbors = distances[:k]

        # WR ponderado pela distância (mais perto = mais peso)
        total_weight = 0.0
        weighted_wins = 0.0
        for dist, result in neighbors:
            w = 1.0 / (dist + 0.01)  # evita divisão por zero
            weighted_wins += w * result
            total_weight += w

        prob = weighted_wins / total_weight if total_weight > 0 else 0.5
        return prob, k, f"kNN_k={k}"

    def _logistic_predict(self, features: np.ndarray) -> float:
        """Regressão logística: sigmoid(w·x_norm + b)."""
        x_norm = (features - self._feature_means) / self._feature_stds
        z = np.dot(self.weights, x_norm) + self.bias
        z = np.clip(z, -10, 10)  # evita overflow
        return 1.0 / (1.0 + np.exp(-z))

    def save(self):
        """Salva pesos e memória em disco."""
        try:
            os.makedirs(os.path.dirname(BRAIN_FILE), exist_ok=True)
            # Salvar apenas os últimos 500 trades na memória (economia de disco)
            mem_save = [{"f": m[0].tolist(), "r": m[1]} for m in self.memory[-500:]]
            data = {
                "weights": self.weights.tolist(),
                "bias": self.bias,
                "logistic_weight": self.logistic_weight,
                "total_decisions": self.total_decisions,
                "total_approved": self.total_approved,
                "total_blocked": self.total_blocked,
                "feature_means": self._feature_means.tolist(),
                "feature_stds": self._feature_stds.tolist(),
                "memory": mem_save,
            }
            with open(BRAIN_FILE, "w") as f:
                json.dump(data, f)
        except Exception as e:
            log.warning(f"  ⚠️ Erro ao salvar cérebro: {e}")

    def get_stats(self) -> Dict:
        """Retorna estatísticas do cérebro."""
        return {
            "total_decisions": self.total_decisions,
            "total_approved": self.total_approved,
            "total_blocked": self.total_blocked,
            "memory_size": len(self.memory),
            "logistic_weight": round(self.logistic_weight, 3),
            "weights_norm": round(float(np.linalg.norm(self.weights)), 4),
        }


# ═══════════════════════════════════════════════════════════════
# INSTÂNCIA GLOBAL
# ═══════════════════════════════════════════════════════════════
brain = AdaptiveBrain()


def brain_decide(pat: dict, H, L, C_arr, O, n: int, atr: float,
                 hs_stats: dict, ativo: str) -> Tuple[bool, float, Dict]:
    """API principal: extrai features e pede decisão ao cérebro.

    Returns:
        (aprovado, probabilidade, detalhes)
    """
    # Carregar na primeira chamada
    brain.load(hs_stats)

    # Extrair features
    features = extract_features(pat, H, L, C_arr, O, n, atr, hs_stats, ativo)
    if features is None:
        log.warning("  ⚠️ CÉREBRO: Features inválidas — BLOQUEANDO")
        return False, 0.0, {"reason": "features_invalidas"}

    # Decidir
    approved, prob, details = brain.decide(features, pat)
    details["ativo"] = ativo
    details["direction"] = pat.get("direction", "")

    return approved, prob, details


def brain_learn(pat: dict, H, L, C_arr, O, n: int, atr: float,
                hs_stats: dict, ativo: str, result: int):
    """Ensina o cérebro com o resultado de um trade.
    result: 1=WIN, 0=LOSS"""
    features = extract_features(pat, H, L, C_arr, O, n, atr, hs_stats, ativo)
    if features is not None:
        brain.learn(features, result)


# ═══════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════
def _ema(data, period):
    """EMA simples."""
    if len(data) < period:
        return None
    multiplier = 2.0 / (period + 1)
    ema = float(data[0])
    for i in range(1, len(data)):
        ema = (float(data[i]) - ema) * multiplier + ema
    return ema
