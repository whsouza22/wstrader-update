# -*- coding: utf-8 -*-
"""
ws_continuation_ai.py — Motor de IA para Continuacao (Compressao + Breakout)
=============================================================================
Detecta padroes de compressao em velas ao vivo e usa o modelo ML treinado
(XGBoost + LightGBM) para filtrar entradas de alta probabilidade.

Estrategia:
  1. Impulso forte (3+ velas consecutivas na mesma direcao, range >= 0.3*ATR)
  2. Compressao (1-3 velas pequenas, range < thresh*ATR)
  3. Breakout candle fecha na direcao do impulso
  4. ML confirma com prob >= threshold → SINAL

Uso pelo bot:
  from ws_continuation_ai import ContinuationAI
  cont_ai = ContinuationAI()
  signals = cont_ai.scan(df)  # df com open/high/low/close

Uso pelo dashboard:
  Mesmo objeto — scan() retorna lista de sinais compativeis com o dashboard.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import time

log = logging.getLogger("CONT_AI")

_USER_DIR = os.path.join(os.path.expanduser("~"), ".wstrader")
_MODEL_FILE = os.path.join(_USER_DIR, "continuation_ml.pkl")


# ======================================================================
# INDICADORES VETORIZADOS
# ======================================================================

def _compute_atr(H, L, period=14):
    ranges = H - L
    n = len(H)
    atr = np.zeros(n)
    cs = np.cumsum(ranges)
    atr[period:] = (cs[period:] - cs[:n - period]) / period
    return atr


def _compute_ema(arr, period):
    ema = np.zeros_like(arr)
    if len(arr) < period:
        return ema
    ema[period - 1] = np.mean(arr[:period])
    k = 2.0 / (period + 1)
    for i in range(period, len(arr)):
        ema[i] = arr[i] * k + ema[i - 1] * (1 - k)
    return ema


def _compute_rsi(C, period=14):
    n = len(C)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    delta = np.diff(C)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


# ======================================================================
# DETECCAO DE PADRAO
# ======================================================================

def _detect_patterns(O, H, L, C, atr, min_impulse=3,
                     n_small_list=(1, 2, 3), thresh_list=(0.5, 0.6, 0.7)):
    """Detecta padroes de compressao. Retorna lista de dicts."""
    n = len(C)
    ranges = H - L

    is_bear_big = (C < O) & (ranges >= 0.3 * atr)
    is_bull_big = (C > O) & (ranges >= 0.3 * atr)

    bear_runs = np.zeros(n, dtype=np.int32)
    bull_runs = np.zeros(n, dtype=np.int32)
    if n > 0:
        bear_runs[0] = 1 if is_bear_big[0] else 0
        bull_runs[0] = 1 if is_bull_big[0] else 0
    for i in range(1, n):
        bear_runs[i] = (bear_runs[i - 1] + 1) if is_bear_big[i] else 0
        bull_runs[i] = (bull_runs[i - 1] + 1) if is_bull_big[i] else 0

    small_runs_dict = {}
    for th in thresh_list:
        is_small = ranges < th * atr
        sr = np.zeros(n, dtype=np.int32)
        if n > 0:
            sr[0] = 1 if is_small[0] else 0
        for i in range(1, n):
            sr[i] = (sr[i - 1] + 1) if is_small[i] else 0
        small_runs_dict[th] = sr

    patterns = []
    for ns in n_small_list:
        for th in thresh_list:
            sm_runs = small_runs_dict[th]
            start = max(min_impulse + ns + 1, 15)
            if start >= n:
                continue

            j = np.arange(start, n)
            small_ok = sm_runs[j - 1] >= ns

            imp_last = j - ns - 1
            imp_first = j - ns - min_impulse
            valid = imp_first >= 0

            bear_ok = np.zeros(len(j), dtype=bool)
            bear_ok[valid] = ((bear_runs[imp_last[valid]] >= min_impulse) &
                              (C[imp_first[valid]] > C[imp_last[valid]]))

            bull_ok = np.zeros(len(j), dtype=bool)
            bull_ok[valid] = ((bull_runs[imp_last[valid]] >= min_impulse) &
                              (C[imp_last[valid]] > C[imp_first[valid]]))

            for idx in j[small_ok & bear_ok]:
                patterns.append({
                    "j": int(idx),
                    "imp_start": int(idx - ns - min_impulse),
                    "imp_end": int(idx - ns),
                    "sm_start": int(idx - ns),
                    "sm_end": int(idx),
                    "direction": "PUT",
                    "n_small": ns,
                    "thresh": th,
                })

            for idx in j[small_ok & bull_ok]:
                patterns.append({
                    "j": int(idx),
                    "imp_start": int(idx - ns - min_impulse),
                    "imp_end": int(idx - ns),
                    "sm_start": int(idx - ns),
                    "sm_end": int(idx),
                    "direction": "CALL",
                    "n_small": ns,
                    "thresh": th,
                })

    # Filtrar sobrepostos (manter primeiro)
    if not patterns:
        return []
    patterns.sort(key=lambda p: p["j"])
    filtered = [patterns[0]]
    for p in patterns[1:]:
        if p["j"] >= filtered[-1]["j"] + 2:
            filtered.append(p)
    return filtered


# ======================================================================
# EXTRACAO DE FEATURES (33 features, identico ao treino)
# ======================================================================

def _extract_features(O, H, L, C, atr, ema8, ema20, ema50, rsi,
                      imp_start, imp_end, sm_start, sm_end, direction):
    """Extrai 33 features de um padrao. Usa candle de breakout (sm_end)."""
    n = len(C)
    ei = sm_end  # breakout candle — ja fechou no momento da entrada
    atr_val = atr[ei] if atr[ei] > 1e-10 else 1e-10
    is_put = 1 if direction == "PUT" else 0

    f = {}

    # --- IMPULSO ---
    imp_bodies = np.abs(C[imp_start:imp_end] - O[imp_start:imp_end])
    imp_ranges = H[imp_start:imp_end] - L[imp_start:imp_end]
    imp_drop = abs(C[imp_end - 1] - C[imp_start])

    f["imp_length"] = imp_end - imp_start
    f["imp_drop_atr"] = imp_drop / atr_val
    f["imp_avg_body_atr"] = np.mean(imp_bodies) / atr_val if len(imp_bodies) > 0 else 0
    f["imp_max_body_atr"] = np.max(imp_bodies) / atr_val if len(imp_bodies) > 0 else 0
    f["imp_body_consistency"] = np.min(imp_bodies) / (np.max(imp_bodies) + 1e-10) if len(imp_bodies) > 0 else 0
    f["imp_acceleration"] = (imp_bodies[-1] / (imp_bodies[0] + 1e-10)) if len(imp_bodies) >= 2 else 1.0

    # --- COMPRESSAO ---
    sm_ranges = H[sm_start:sm_end] - L[sm_start:sm_end]
    sm_bodies = np.abs(C[sm_start:sm_end] - O[sm_start:sm_end])

    f["sm_length"] = sm_end - sm_start
    f["sm_avg_range_atr"] = np.mean(sm_ranges) / atr_val if len(sm_ranges) > 0 else 0
    f["sm_min_range_atr"] = np.min(sm_ranges) / atr_val if len(sm_ranges) > 0 else 0
    f["sm_range_shrink"] = (sm_ranges[-1] / (sm_ranges[0] + 1e-10)) if len(sm_ranges) >= 2 else 1.0
    f["compression_ratio"] = (np.mean(sm_ranges) / (np.mean(imp_ranges) + 1e-10)) if len(imp_ranges) > 0 else 1.0
    f["sm_body_vs_range"] = np.mean(sm_bodies) / (np.mean(sm_ranges) + 1e-10) if len(sm_ranges) > 0 else 0.5

    if direction == "PUT":
        aligned = np.sum(C[sm_start:sm_end] < O[sm_start:sm_end])
    else:
        aligned = np.sum(C[sm_start:sm_end] > O[sm_start:sm_end])
    f["sm_aligned_pct"] = aligned / max(sm_end - sm_start, 1)

    # --- TENDENCIA/CONTEXTO ---
    f["ema8_vs_20"] = (ema8[ei] - ema20[ei]) / atr_val
    f["ema20_vs_50"] = (ema20[ei] - ema50[ei]) / atr_val
    f["price_vs_ema50"] = (C[ei] - ema50[ei]) / atr_val
    f["rsi"] = rsi[ei] / 100.0

    # --- BREAKOUT CANDLE ---
    brk_body = abs(C[ei] - O[ei])
    brk_range = H[ei] - L[ei]
    f["brk_body_atr"] = brk_body / atr_val
    f["brk_range_atr"] = brk_range / atr_val
    f["brk_body_ratio"] = brk_body / (brk_range + 1e-10)
    if is_put:
        f["brk_aligned"] = 1.0 if C[ei] < O[ei] else 0.0
        f["brk_force"] = (O[ei] - C[ei]) / atr_val
    else:
        f["brk_aligned"] = 1.0 if C[ei] > O[ei] else 0.0
        f["brk_force"] = (C[ei] - O[ei]) / atr_val

    # --- DIRECAO AJUSTADA ---
    if is_put:
        f["trend_alignment"] = -f["ema8_vs_20"]
        f["rsi_favorable"] = f["rsi"]
    else:
        f["trend_alignment"] = f["ema8_vs_20"]
        f["rsi_favorable"] = 1.0 - f["rsi"]

    # --- MOMENTUM ---
    lookback = min(10, ei - 1)
    if lookback >= 2:
        mom = C[ei] - C[ei - lookback]
        f["momentum_10"] = mom / atr_val
        f["momentum_aligned"] = (-mom / atr_val) if is_put else (mom / atr_val)
    else:
        f["momentum_10"] = 0
        f["momentum_aligned"] = 0

    # --- VOLATILIDADE ---
    if ei >= 50:
        f["vol_regime"] = np.std(C[ei - 10:ei]) / (np.std(C[ei - 50:ei]) + 1e-10)
    else:
        f["vol_regime"] = 1.0

    if ei >= 60:
        f["atr_relative"] = np.mean(H[ei - 10:ei] - L[ei - 10:ei]) / (np.mean(H[ei - 60:ei] - L[ei - 60:ei]) + 1e-10)
    else:
        f["atr_relative"] = 1.0

    # --- MERCADO ---
    if ei >= 20:
        hh = np.max(H[ei - 20:ei])
        ll = np.min(L[ei - 20:ei])
        f["price_position_20"] = (C[ei] - ll) / (hh - ll + 1e-10)
    else:
        f["price_position_20"] = 0.5

    if ei >= 10:
        changes = np.sum(np.diff(np.sign(C[ei - 10:ei] - O[ei - 10:ei])) != 0)
        f["alternation"] = changes / 9.0
    else:
        f["alternation"] = 0.5

    f["is_put"] = float(is_put)

    return f


# ======================================================================
# CLASSE PRINCIPAL
# ======================================================================

FEATURE_COLS = [
    "imp_length", "imp_drop_atr", "imp_avg_body_atr", "imp_max_body_atr",
    "imp_body_consistency", "imp_acceleration",
    "sm_length", "sm_avg_range_atr", "sm_min_range_atr", "sm_range_shrink",
    "compression_ratio", "sm_body_vs_range", "sm_aligned_pct",
    "ema8_vs_20", "ema20_vs_50", "price_vs_ema50", "rsi",
    "brk_body_atr", "brk_range_atr", "brk_body_ratio",
    "brk_aligned", "brk_force",
    "trend_alignment", "rsi_favorable",
    "momentum_10", "momentum_aligned",
    "vol_regime", "atr_relative",
    "price_position_20", "alternation",
    "is_put", "n_small_cfg", "thresh_cfg",
]


class ContinuationAI:
    """Motor de IA para continuacao. Carrega modelo, detecta padroes, prediz."""

    def __init__(self):
        self.model = None
        self.loaded = False
        self._load_model()

    def _load_model(self):
        """Carrega modelo treinado do disco."""
        if not os.path.exists(_MODEL_FILE):
            log.warning(f"Modelo continuation_ml.pkl nao encontrado em {_MODEL_FILE}")
            return
        try:
            with open(_MODEL_FILE, "rb") as f:
                data = pickle.load(f)
            self.model_type = data.get("type", "global")
            self.feature_cols = data.get("feature_cols", FEATURE_COLS)
            self.threshold = data.get("threshold", 0.50)
            self.exp = data.get("exp", 2)
            self.min_impulse = data.get("min_impulse", 3)
            self.stats = data.get("stats", {})
            if self.model_type == "per_asset":
                self.per_asset = data.get("per_asset", {})
                self.xgb = self.lgb = self.scaler = None
            else:
                self.per_asset = {}
                self.xgb = data["xgb"]
                self.lgb = data["lgb"]
                self.scaler = data["scaler"]
            self.loaded = True
            log.info(f"[CONT_AI] Modelo carregado: WR_teste={self.stats.get('test_wr', '?')}% "
                     f"threshold={self.threshold} WF={self.stats.get('walk_forward_wr', '?')}%")
        except Exception as e:
            log.error(f"[CONT_AI] Erro ao carregar modelo: {e}")

    def reload(self):
        """Recarrega modelo (caso retreinado)."""
        self._load_model()

    def _predict(self, x, ativo=""):
        """Prediz usando modelo per-asset ou global. Retorna (prob, p_xgb, p_lgb, threshold)."""
        am = None
        if self.per_asset and ativo:
            am = self.per_asset.get(ativo)
        if am:
            x_s = am["scaler"].transform(x)
            x_df = pd.DataFrame(x_s, columns=self.feature_cols)
            p_xgb = float(am["xgb"].predict_proba(x_df)[0, 1])
            p_lgb = float(am["lgb"].predict_proba(x_df)[0, 1])
            th = am.get("threshold", self.threshold)
        elif self.xgb is not None:
            x_s = self.scaler.transform(x)
            x_df = pd.DataFrame(x_s, columns=self.feature_cols)
            p_xgb = float(self.xgb.predict_proba(x_df)[0, 1])
            p_lgb = float(self.lgb.predict_proba(x_df)[0, 1])
            th = self.threshold
        else:
            return 0.5, 0.5, 0.5, self.threshold
        return (p_xgb + p_lgb) / 2.0, p_xgb, p_lgb, th

    def scan(self, df, ativo="", max_candles_ago=2):
        """Detecta padroes de compressao e retorna sinais filtrados pela ML.

        Args:
            df: DataFrame com colunas open, high, low, close (index = datetime)
            ativo: nome do ativo (para o dashboard)
            max_candles_ago: maximo de velas atras para considerar sinal 'live'

        Returns:
            Lista de sinais:
            [{
                "ativo": str,
                "direction": "PUT"|"CALL",
                "type": "CONTINUATION",
                "mode": "breakout_confirmation",
                "ml_prob": float,
                "ml_approved": bool,
                "entry_idx": int,
                "entry_price": float,
                "breakout_idx": int,
                "candles_ago": int,
                "scan_ts": float,
                "pattern": {...},  # detalhes do padrao
                "backtest": None | {"result": "win"|"loss", ...},
            }]
        """
        if not self.loaded:
            return []

        O = df["open"].values.astype(np.float64)
        H = df["high"].values.astype(np.float64)
        L = df["low"].values.astype(np.float64)
        C = df["close"].values.astype(np.float64)
        n = len(C)

        if n < 80:
            return []

        atr = _compute_atr(H, L)
        ema8 = _compute_ema(C, 8)
        ema20 = _compute_ema(C, 20)
        ema50 = _compute_ema(C, 50)
        rsi = _compute_rsi(C)

        patterns = _detect_patterns(
            O, H, L, C, atr,
            min_impulse=self.min_impulse,
            n_small_list=[1, 2, 3],
            thresh_list=[0.5, 0.6, 0.7],
        )

        signals = []
        seen_idx = set()

        for pat in patterns:
            j = pat["j"]  # breakout candle
            entry_idx = j  # entrada apos breakout fechar
            exit_idx = j + self.exp

            if entry_idx < 60 or entry_idx in seen_idx:
                continue
            seen_idx.add(entry_idx)

            candles_ago = n - 1 - j

            # Extrair features
            feats = _extract_features(
                O, H, L, C, atr, ema8, ema20, ema50, rsi,
                pat["imp_start"], pat["imp_end"],
                pat["sm_start"], pat["sm_end"],
                pat["direction"],
            )
            feats["n_small_cfg"] = pat["n_small"]
            feats["thresh_cfg"] = pat["thresh"]

            # Extended features (per-asset signal)
            is_put_b = pat["direction"] == "PUT"
            atr_j = max(atr[j], 1e-10)
            if j >= 5:
                feats["ema8_slope"] = (ema8[j] - ema8[j - 5]) / atr_j
                feats["ema20_slope"] = (ema20[j] - ema20[j - 5]) / atr_j
                feats["rsi_change5"] = (rsi[j] - rsi[j - 5]) / 100.0
            else:
                feats["ema8_slope"] = feats["ema20_slope"] = feats["rsi_change5"] = 0.0
            consec = 0
            for k in range(j, max(j - 10, -1), -1):
                if (C[k] < O[k]) == is_put_b:
                    consec += 1
                else:
                    break
            feats["consecutive_aligned"] = consec
            if j >= 5:
                bodies5 = np.abs(C[j - 5:j] - O[j - 5:j])
                ranges5 = H[j - 5:j] - L[j - 5:j]
                feats["body_pct_5"] = float(np.mean(bodies5 / (ranges5 + 1e-10)))
                feats["directional_pct_5"] = sum(
                    1 for kk in range(j - 5, j) if (C[kk] < O[kk]) == is_put_b
                ) / 5.0
            else:
                feats["body_pct_5"] = feats["directional_pct_5"] = 0.5

            # Montar vetor de features na ordem do modelo
            x = np.array([[feats.get(col, 0) for col in self.feature_cols]], dtype=np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=3.0, neginf=-3.0)

            # Predizer (per-asset ou global)
            prob, p_xgb, p_lgb, th = self._predict(x, ativo)

            # Filtro alta confianca: ambos modelos devem concordar
            both_agree = (p_xgb >= th) and (p_lgb >= th)

            approved = both_agree

            # Backtest: se temos velas suficientes apos a entrada
            bt = None
            if exit_idx < n:
                if pat["direction"] == "PUT":
                    win = C[exit_idx] < C[entry_idx]
                else:
                    win = C[exit_idx] > C[entry_idx]
                bt = {
                    "result": "win" if win else "loss",
                    "entry_idx": int(entry_idx),
                    "exit_idx": int(exit_idx),
                    "entry_price": round(float(C[entry_idx]), 6),
                    "exit_price": round(float(C[exit_idx]), 6),
                }

            # Timestamps dos pontos-chave
            df_index = df.index
            df_len = len(df_index)

            def _idx_to_ts(idx):
                if 0 <= idx < df_len:
                    ts = df_index[idx]
                    return int(ts.value // 10**9) if hasattr(ts, 'value') else 0
                return 0

            signal = {
                "ativo": ativo,
                "direction": pat["direction"],
                "type": "CONTINUATION",
                "mode": "breakout_confirmation",
                "ml_prob": round(prob, 4),
                "ml_approved": approved,
                "nn_approved": approved,  # compatibilidade com dashboard
                "nn_score": round(prob, 4),
                "confidence": round(prob * 100, 1),
                "entry_idx": int(entry_idx),
                "entry_price": round(float(C[entry_idx]), 6),
                "entry_ts": _idx_to_ts(entry_idx),
                "breakout_idx": int(j),
                "candles_ago": int(candles_ago),
                "scan_ts": time.time(),
                "exp_candles": self.exp,
                "impulse": {
                    "start_idx": pat["imp_start"],
                    "end_idx": pat["imp_end"],
                    "start_ts": _idx_to_ts(pat["imp_start"]),
                    "end_ts": _idx_to_ts(pat["imp_end"]),
                },
                "compression": {
                    "start_idx": pat["sm_start"],
                    "end_idx": pat["sm_end"],
                    "start_ts": _idx_to_ts(pat["sm_start"]),
                    "end_ts": _idx_to_ts(pat["sm_end"]),
                },
                "pattern": {
                    "type": "CONTINUATION",
                    "direction": pat["direction"],
                    "n_small": pat["n_small"],
                    "thresh": pat["thresh"],
                    "imp_start": pat["imp_start"],
                    "imp_end": pat["imp_end"],
                    "sm_start": pat["sm_start"],
                    "sm_end": pat["sm_end"],
                },
                "backtest": bt,
                "xgb_prob": round(p_xgb, 4),
                "lgb_prob": round(p_lgb, 4),
                "ia_prob": round(prob, 3),
                "both_agree": bool((p_xgb >= th) and (p_lgb >= th)),
                "regime_ok": bool(regime_ok),
                "regime": {
                    "alternation": round(alt, 3),
                    "compression": round(comp, 3),
                    "vol_regime": round(vol, 3),
                },
            }

            # Adicionar timestamps para compatibilidade com dashboard
            if bt:
                bt["entry_ts"] = _idx_to_ts(bt["entry_idx"])
                bt["exit_ts"] = _idx_to_ts(bt["exit_idx"])

            signals.append(signal)

        return signals

    def scan_live(self, df, ativo="", max_candles_ago=2):
        """Retorna apenas sinais LIVE (sem resultado ainda) aprovados pela ML."""
        all_signals = self.scan(df, ativo, max_candles_ago)
        live = []
        for s in all_signals:
            if s["backtest"] is None and s["ml_approved"] and s["candles_ago"] <= max_candles_ago:
                live.append(s)
        return live

    def predict_single(self, df, direction, imp_start, imp_end, sm_start, sm_end, ativo=""):
        """Prediz probabilidade para um padrao especifico (uso pelo bot)."""
        if not self.loaded:
            return 0.0, False

        O = df["open"].values.astype(np.float64)
        H = df["high"].values.astype(np.float64)
        L = df["low"].values.astype(np.float64)
        C = df["close"].values.astype(np.float64)

        atr = _compute_atr(H, L)
        ema8 = _compute_ema(C, 8)
        ema20 = _compute_ema(C, 20)
        ema50 = _compute_ema(C, 50)
        rsi_arr = _compute_rsi(C)

        feats = _extract_features(
            O, H, L, C, atr, ema8, ema20, ema50, rsi_arr,
            imp_start, imp_end, sm_start, sm_end, direction,
        )
        feats["n_small_cfg"] = sm_end - sm_start
        feats["thresh_cfg"] = 0.6

        # Extended features
        j = sm_end  # breakout candle
        is_put_b = direction == "PUT"
        atr_j = max(atr[j], 1e-10)
        if j >= 5:
            feats["ema8_slope"] = (ema8[j] - ema8[j - 5]) / atr_j
            feats["ema20_slope"] = (ema20[j] - ema20[j - 5]) / atr_j
            feats["rsi_change5"] = (rsi_arr[j] - rsi_arr[j - 5]) / 100.0
        else:
            feats["ema8_slope"] = feats["ema20_slope"] = feats["rsi_change5"] = 0.0
        consec = 0
        for k in range(j, max(j - 10, -1), -1):
            if (C[k] < O[k]) == is_put_b:
                consec += 1
            else:
                break
        feats["consecutive_aligned"] = consec
        if j >= 5:
            bodies5 = np.abs(C[j - 5:j] - O[j - 5:j])
            ranges5 = H[j - 5:j] - L[j - 5:j]
            feats["body_pct_5"] = float(np.mean(bodies5 / (ranges5 + 1e-10)))
            feats["directional_pct_5"] = sum(
                1 for kk in range(j - 5, j) if (C[kk] < O[kk]) == is_put_b
            ) / 5.0
        else:
            feats["body_pct_5"] = feats["directional_pct_5"] = 0.5

        x = np.array([[feats.get(col, 0) for col in self.feature_cols]], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=3.0, neginf=-3.0)

        prob, p_xgb, p_lgb, th = self._predict(x, ativo)
        both_agree = (p_xgb >= th) and (p_lgb >= th)

        return prob, both_agree

    def get_stats(self):
        """Retorna stats do modelo para exibicao."""
        if not self.loaded:
            return {"loaded": False}
        return {
            "loaded": True,
            "model_type": getattr(self, "model_type", "global"),
            "n_assets": self.stats.get("n_assets", 1),
            "threshold": self.threshold,
            "exp": self.exp,
            "min_impulse": self.min_impulse,
            "test_wr": self.stats.get("test_wr", 0),
            "test_trades": self.stats.get("test_trades", 0),
            "walk_forward_wr": self.stats.get("walk_forward_wr", 0),
            "walk_forward_trades": self.stats.get("walk_forward_trades", 0),
            "perm_test_wr": self.stats.get("perm_test_wr", 0),
        }
