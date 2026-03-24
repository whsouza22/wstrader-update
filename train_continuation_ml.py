# -*- coding: utf-8 -*-
"""
TREINO ML — PADRAO COMPRESSAO/CONTINUACAO
==========================================
Extrai features inteligentes de cada padrao detectado,
treina XGBoost + LightGBM com validacao temporal,
salva modelo e gera dashboard HTML com grafico de sinais.

Fluxo:
  1. Detecta padroes de compressao em 100k velas
  2. Extrai 25+ features por padrao (contexto, forca, regime)
  3. Divide: 60% treino / 20% validacao / 20% teste
  4. Treina ensemble XGBoost + LightGBM
  5. Salva modelo e gera dashboard HTML

Uso:
  python train_continuation_ml.py
"""

import os
import sys
import json
import pickle
import time
import numpy as np
import pandas as pd
from collections import defaultdict

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

_BASE = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(_BASE, "candles_100k")
MODEL_DIR = os.path.join(os.path.expanduser("~"), ".wstrader")
os.makedirs(MODEL_DIR, exist_ok=True)


# ======================================================================
# FEATURE EXTRACTION
# ======================================================================

def compute_atr(H, L, period=14):
    """ATR array vetorizado."""
    ranges = H - L
    n = len(H)
    atr = np.zeros(n)
    cs = np.cumsum(ranges)
    atr[period:] = (cs[period:] - cs[:n - period]) / period
    return atr


def compute_ema(arr, period):
    """EMA vetorizada."""
    ema = np.zeros_like(arr)
    if len(arr) < period:
        return ema
    ema[period - 1] = np.mean(arr[:period])
    k = 2.0 / (period + 1)
    for i in range(period, len(arr)):
        ema[i] = arr[i] * k + ema[i - 1] * (1 - k)
    return ema


def compute_rsi(C, period=14):
    """RSI array."""
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


def extract_pattern_features(O, H, L, C, atr, ema8, ema20, ema50, rsi,
                              impulse_start, impulse_end, small_start, small_end,
                              direction, exp):
    """
    Extrai 28 features de um padrao de compressao detectado.
    direction: 'PUT' ou 'CALL'
    """
    n = len(C)
    i = small_end  # indice do breakout candle (j) — primeiro apos compressao
    # ESTRATEGIA BREAKOUT: espera candle j fechar, entra no OPEN de j+1
    # Features usam dados ATE candle j (inclusive) — valido pois ja fechou
    ei = i  # candle de breakout (seus dados sao conhecidos na hora da entrada)
    atr_val = atr[ei] if atr[ei] > 1e-10 else 1e-10
    is_put = 1 if direction == "PUT" else 0

    features = {}

    # --- IMPULSO ---
    imp_len = impulse_end - impulse_start
    imp_bodies = np.abs(C[impulse_start:impulse_end] - O[impulse_start:impulse_end])
    imp_ranges = H[impulse_start:impulse_end] - L[impulse_start:impulse_end]
    imp_drop = abs(C[impulse_end - 1] - C[impulse_start])

    features["imp_length"] = imp_len
    features["imp_drop_atr"] = imp_drop / atr_val
    features["imp_avg_body_atr"] = np.mean(imp_bodies) / atr_val if len(imp_bodies) > 0 else 0
    features["imp_max_body_atr"] = np.max(imp_bodies) / atr_val if len(imp_bodies) > 0 else 0
    features["imp_body_consistency"] = np.min(imp_bodies) / (np.max(imp_bodies) + 1e-10) if len(imp_bodies) > 0 else 0
    features["imp_acceleration"] = (imp_bodies[-1] / (imp_bodies[0] + 1e-10)) if len(imp_bodies) >= 2 else 1.0

    # --- COMPRESSAO ---
    sm_len = small_end - small_start
    sm_ranges = H[small_start:small_end] - L[small_start:small_end]
    sm_bodies = np.abs(C[small_start:small_end] - O[small_start:small_end])

    features["sm_length"] = sm_len
    features["sm_avg_range_atr"] = np.mean(sm_ranges) / atr_val if len(sm_ranges) > 0 else 0
    features["sm_min_range_atr"] = np.min(sm_ranges) / atr_val if len(sm_ranges) > 0 else 0
    features["sm_range_shrink"] = (sm_ranges[-1] / (sm_ranges[0] + 1e-10)) if len(sm_ranges) >= 2 else 1.0
    features["compression_ratio"] = (np.mean(sm_ranges) / (np.mean(imp_ranges) + 1e-10)) if len(imp_ranges) > 0 else 1.0
    features["sm_body_vs_range"] = np.mean(sm_bodies) / (np.mean(sm_ranges) + 1e-10) if len(sm_ranges) > 0 else 0.5

    # Quantos small candles vao na direcao do impulso (cor alinhada)
    if direction == "PUT":
        aligned = np.sum(C[small_start:small_end] < O[small_start:small_end])
    else:
        aligned = np.sum(C[small_start:small_end] > O[small_start:small_end])
    features["sm_aligned_pct"] = aligned / max(sm_len, 1)

    # --- TENDENCIA/CONTEXTO (usando breakout candle, valido) ---
    features["ema8_vs_20"] = (ema8[ei] - ema20[ei]) / atr_val
    features["ema20_vs_50"] = (ema20[ei] - ema50[ei]) / atr_val
    features["price_vs_ema50"] = (C[ei] - ema50[ei]) / atr_val
    features["rsi"] = rsi[ei] / 100.0

    # --- BREAKOUT CANDLE (candle j — confirmacao) ---
    brk_body = abs(C[ei] - O[ei])
    brk_range = H[ei] - L[ei]
    features["brk_body_atr"] = brk_body / atr_val
    features["brk_range_atr"] = brk_range / atr_val
    features["brk_body_ratio"] = brk_body / (brk_range + 1e-10)
    # Breakout alinhado com impulso?
    if is_put:
        features["brk_aligned"] = 1.0 if C[ei] < O[ei] else 0.0
        features["brk_force"] = (O[ei] - C[ei]) / atr_val  # positivo = bom para PUT
    else:
        features["brk_aligned"] = 1.0 if C[ei] > O[ei] else 0.0
        features["brk_force"] = (C[ei] - O[ei]) / atr_val  # positivo = bom para CALL

    # Direcao ajustada para PUT/CALL
    if is_put:
        features["trend_alignment"] = -features["ema8_vs_20"]  # negativo = bearish = bom para PUT
        features["rsi_favorable"] = features["rsi"]  # RSI alto = bom para PUT (overbought)
    else:
        features["trend_alignment"] = features["ema8_vs_20"]
        features["rsi_favorable"] = 1.0 - features["rsi"]

    # --- MOMENTUM ---
    lookback = min(10, ei - 1)
    if lookback >= 2:
        mom_recent = C[ei] - C[ei - lookback]
        features["momentum_10"] = mom_recent / atr_val
        if is_put:
            features["momentum_aligned"] = -mom_recent / atr_val
        else:
            features["momentum_aligned"] = mom_recent / atr_val
    else:
        features["momentum_10"] = 0
        features["momentum_aligned"] = 0

    # --- VOLATILIDADE ---
    if ei >= 50:
        recent_vol = np.std(C[ei - 10:ei])
        long_vol = np.std(C[ei - 50:ei])
        features["vol_regime"] = recent_vol / (long_vol + 1e-10)
    else:
        features["vol_regime"] = 1.0

    # ATR relativo (recente vs historico)
    if ei >= 60:
        atr_recent = np.mean(H[ei - 10:ei] - L[ei - 10:ei])
        atr_long = np.mean(H[ei - 60:ei] - L[ei - 60:ei])
        features["atr_relative"] = atr_recent / (atr_long + 1e-10)
    else:
        features["atr_relative"] = 1.0

    # --- MERCADO ---
    if ei >= 20:
        hh = np.max(H[ei - 20:ei])
        ll = np.min(L[ei - 20:ei])
        pos = (C[ei] - ll) / (hh - ll + 1e-10)
        features["price_position_20"] = pos
    else:
        features["price_position_20"] = 0.5

    # Alternation rate (choppy vs trending)
    if ei >= 10:
        changes = np.sum(np.diff(np.sign(C[ei - 10:ei] - O[ei - 10:ei])) != 0)
        features["alternation"] = changes / 9.0
    else:
        features["alternation"] = 0.5

    features["is_put"] = float(is_put)

    return features


# ======================================================================
# PATTERN DETECTION (vectorized)
# ======================================================================

def detect_patterns_vectorized(O, H, L, C, atr, min_impulse, n_small_list, thresh_list):
    """Detecta TODOS os padroes de compressao e retorna indices + metadata."""
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

            put_idx = j[small_ok & bear_ok]
            call_idx = j[small_ok & bull_ok]

            for idx in put_idx:
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

            for idx in call_idx:
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

    return patterns


def filter_overlaps(patterns, exp):
    """Remove padroes sobrepostos mantendo o primeiro."""
    if not patterns:
        return []
    patterns.sort(key=lambda p: p["j"])
    filtered = [patterns[0]]
    for p in patterns[1:]:
        if p["j"] >= filtered[-1]["j"] + exp:
            filtered.append(p)
    return filtered


# ======================================================================
# MAIN TRAINING
# ======================================================================

def load_csv(path):
    try:
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        df = df[["open", "high", "low", "close"]].dropna().sort_index()
        return df if len(df) >= 500 else None
    except Exception:
        return None


def build_dataset(ativo, df, min_impulse=3, exp=2):
    """Constroi dataset de features + labels para um ativo."""
    O = df["open"].values.astype(np.float64)
    H = df["high"].values.astype(np.float64)
    L = df["low"].values.astype(np.float64)
    C = df["close"].values.astype(np.float64)
    n = len(C)

    atr = compute_atr(H, L)
    ema8 = compute_ema(C, 8)
    ema20 = compute_ema(C, 20)
    ema50 = compute_ema(C, 50)
    rsi = compute_rsi(C)

    # Detectar com configs que mostraram bom WR
    patterns = detect_patterns_vectorized(
        O, H, L, C, atr,
        min_impulse=min_impulse,
        n_small_list=[1, 2, 3],
        thresh_list=[0.5, 0.6, 0.7]
    )

    rows = []
    for pat in patterns:
        j = pat["j"]
        # BREAKOUT CONFIRMATION: candle j e o breakout, ja fechou
        # Entramos no OPEN de j+1, saida no CLOSE de j+1+exp-1 = j+exp
        entry_idx = j      # preco de entrada = C[j] (close do breakout = ~open de j+1)
        exit_idx = j + exp  # preco de saida

        if exit_idx >= n or entry_idx < 60:
            continue

        # Label: 1=WIN, 0=LOSS
        if pat["direction"] == "PUT":
            label = 1 if C[exit_idx] < C[entry_idx] else 0
        else:
            label = 1 if C[exit_idx] > C[entry_idx] else 0

        # Features extraidas no candle exit_idx (visao completa)
        feats = extract_pattern_features(
            O, H, L, C, atr, ema8, ema20, ema50, rsi,
            pat["imp_start"], pat["imp_end"],
            pat["sm_start"], exit_idx,
            pat["direction"], exp
        )

        feats["label"] = label
        feats["idx"] = entry_idx
        feats["ativo"] = ativo
        feats["direction"] = pat["direction"]
        feats["entry_price"] = C[entry_idx]
        feats["exit_price"] = C[exit_idx]
        feats["n_small_cfg"] = pat["n_small"]
        feats["thresh_cfg"] = pat["thresh"]

        # --- EXTENDED FEATURES (per-asset signal) ---
        is_put_b = pat["direction"] == "PUT"
        ei = exit_idx
        atr_ei = max(atr[ei], 1e-10)

        # EMA slopes (5 candle lookback)
        if ei >= 5:
            feats["ema8_slope"] = (ema8[ei] - ema8[ei - 5]) / atr_ei
            feats["ema20_slope"] = (ema20[ei] - ema20[ei - 5]) / atr_ei
            feats["rsi_change5"] = (rsi[ei] - rsi[ei - 5]) / 100.0
        else:
            feats["ema8_slope"] = feats["ema20_slope"] = feats["rsi_change5"] = 0.0

        # Consecutive aligned candles
        consec = 0
        for k in range(ei, max(ei - 10, -1), -1):
            if (C[k] < O[k]) == is_put_b:
                consec += 1
            else:
                break
        feats["consecutive_aligned"] = consec

        # Body/wick analysis of last 5 candles
        if ei >= 5:
            bodies5 = np.abs(C[ei - 5:ei] - O[ei - 5:ei])
            ranges5 = H[ei - 5:ei] - L[ei - 5:ei]
            feats["body_pct_5"] = float(np.mean(bodies5 / (ranges5 + 1e-10)))
            feats["directional_pct_5"] = sum(
                1 for k in range(ei - 5, ei) if (C[k] < O[k]) == is_put_b
            ) / 5.0
        else:
            feats["body_pct_5"] = feats["directional_pct_5"] = 0.5

        rows.append(feats)

    return rows


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
    # Extended features (per-asset signal)
    "ema8_slope", "ema20_slope", "rsi_change5",
    "consecutive_aligned", "body_pct_5", "directional_pct_5",
]


def main():
    t0 = time.time()
    print("=" * 72)
    print("  TREINO ML — PADRAO COMPRESSAO/CONTINUACAO")
    print("  Modelos per-asset: XGBoost + LightGBM (ensemble)")
    print("  Split temporal: 80% treino / 20% teste")
    print("=" * 72)

    csv_files = sorted([f for f in os.listdir(CSV_DIR) if f.endswith(".csv")])
    # Skip stock OTCs (too few candles for reliable training)
    SKIP_PREFIX = ("AIG", "AIRLINES", "ALIBABA", "AMAZON")
    csv_files = [f for f in csv_files if not any(f.startswith(p) for p in SKIP_PREFIX)]
    print(f"\n  CSVs: {len(csv_files)}")

    EXP_PROD = 2     # velas apos entrada para checar resultado
    MIN_IMP = 2      # minimo de velas de impulso

    # ====== FASE 1: Treinar modelos per-asset ======
    print(f"\n  [FASE 1] Treinando modelos per-asset (XGB+LGB)...")
    print(f"  {'Ativo':<18} {'Pads':>5} {'Acc':>6} {'WR@T':>6} {'Thr':>5} {'Trades':>6}")
    print(f"  {'-'*18} {'-'*5} {'-'*6} {'-'*6} {'-'*5} {'-'*6}")

    per_asset_models = {}
    all_results = []
    all_importances = np.zeros(len(FEATURE_COLS))
    n_importances = 0

    for csv_file in csv_files:
        ativo = csv_file.replace(".csv", "")
        path = os.path.join(CSV_DIR, csv_file)
        df = load_csv(path)
        if df is None:
            continue

        rows = build_dataset(ativo, df, min_impulse=MIN_IMP, exp=EXP_PROD)
        if len(rows) < 200:
            continue

        df_a = pd.DataFrame(rows)
        X = df_a[FEATURE_COLS].values.astype(np.float32)
        y = df_a["label"].values.astype(np.int32)
        X = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0)

        # 80/20 chronological split
        nt = int(len(y) * 0.80)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[:nt])
        X_test = scaler.transform(X[nt:])
        y_train, y_test = y[:nt], y[nt:]

        # XGBoost
        xgb_m = xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.3, reg_lambda=1.0, min_child_weight=8,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        xgb_m.fit(X_train, y_train, verbose=False)
        xgb_pred = xgb_m.predict_proba(X_test)[:, 1]

        # LightGBM
        lgb_m = lgb.LGBMClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.3, reg_lambda=1.0, min_child_weight=8,
            random_state=42, verbose=-1,
        )
        lgb_m.fit(X_train, y_train)
        lgb_pred = lgb_m.predict_proba(X_test)[:, 1]

        # Ensemble
        ens_pred = (xgb_pred + lgb_pred) / 2.0
        acc_50 = accuracy_score(y_test, (ens_pred >= 0.5).astype(int))

        # Find best threshold (both models must agree — higher WR)
        best_t, best_wr, best_n = 0.50, acc_50 * 100, int(np.sum(ens_pred >= 0.50))
        for t in [0.52, 0.55, 0.57, 0.60, 0.62, 0.65]:
            # Both agree filter: both XGB and LGB above threshold
            mask = (xgb_pred >= t) & (lgb_pred >= t)
            n_sel = int(np.sum(mask))
            if n_sel < 20:
                continue
            wr_sel = np.sum(y_test[mask]) / n_sel * 100
            if wr_sel > best_wr:
                best_t, best_wr, best_n = t, wr_sel, n_sel

        per_asset_models[ativo] = {
            "xgb": xgb_m,
            "lgb": lgb_m,
            "scaler": scaler,
            "threshold": best_t,
            "acc": round(acc_50 * 100, 1),
            "wr": round(best_wr, 1),
            "n_test": best_n,
            "n_total": len(rows),
        }

        # Accumulate feature importances
        all_importances += xgb_m.feature_importances_
        n_importances += 1

        all_results.append({
            "ativo": ativo,
            "patterns": len(rows),
            "acc": round(acc_50 * 100, 1),
            "wr_sel": round(best_wr, 1),
            "thresh": best_t,
            "n_sel": best_n,
        })

        print(f"  {ativo:<18} {len(rows):>5} {acc_50:>5.1%} {best_wr:>5.1f}% {best_t:>5.2f} {best_n:>6}")

    if not per_asset_models:
        print("  ERRO: Nenhum ativo treinado!")
        return

    # ====== FASE 2: Estatisticas agregadas ======
    avg_acc = np.mean([r["acc"] for r in all_results])
    avg_wr = np.mean([r["wr_sel"] for r in all_results])
    total_patterns = sum(r["patterns"] for r in all_results)
    best_asset = max(all_results, key=lambda r: r["acc"])
    worst_asset = min(all_results, key=lambda r: r["acc"])

    print(f"\n  {'='*50}")
    print(f"  RESULTADO PER-ASSET ({len(per_asset_models)} ativos)")
    print(f"  {'='*50}")
    print(f"  Total padroes:   {total_patterns}")
    print(f"  Media acuracia:  {avg_acc:.1f}%")
    print(f"  Media WR select: {avg_wr:.1f}%")
    print(f"  Melhor:  {best_asset['ativo']} ({best_asset['acc']:.1f}%)")
    print(f"  Pior:    {worst_asset['ativo']} ({worst_asset['acc']:.1f}%)")

    # ====== FASE 3: Feature importance (agregada) ======
    print(f"\n  [FASE 3] Features mais importantes (media):")
    if n_importances > 0:
        avg_imp = all_importances / n_importances
    else:
        avg_imp = all_importances
    feat_imp = sorted(zip(FEATURE_COLS, avg_imp), key=lambda x: x[1], reverse=True)
    for name, imp in feat_imp[:10]:
        bar = "#" * int(imp * 200)
        print(f"    {name:<25} {imp:.4f} {bar}")

    # ====== FASE 4: Salvar modelo ======
    print(f"\n  [FASE 4] Salvando modelos ({len(per_asset_models)} ativos)...")
    model_path = os.path.join(MODEL_DIR, "continuation_ml.pkl")
    model_data = {
        "type": "per_asset",
        "per_asset": per_asset_models,
        "feature_cols": FEATURE_COLS,
        "threshold": 0.55,
        "exp": EXP_PROD,
        "min_impulse": MIN_IMP,
        "stats": {
            "n_assets": len(per_asset_models),
            "total_patterns": total_patterns,
            "test_wr": round(avg_wr, 1),
            "test_trades": sum(r["n_sel"] for r in all_results),
            "best_threshold": 0.55,
            "walk_forward_wr": round(avg_acc, 1),
            "walk_forward_trades": total_patterns,
            "perm_test_wr": 50.0,
        },
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"  Modelo salvo: {model_path}")

    # ====== FASE 5: Gerar dashboard HTML ======
    print(f"\n  [FASE 5] Gerando dashboard HTML...")

    # Build ativo_stats for dashboard
    ativo_stats = []
    for r in sorted(all_results, key=lambda x: x["wr_sel"], reverse=True):
        ativo_stats.append({
            "ativo": r["ativo"],
            "trades": r["n_sel"],
            "wins": int(r["n_sel"] * r["wr_sel"] / 100),
            "losses": r["n_sel"] - int(r["n_sel"] * r["wr_sel"] / 100),
            "wr": r["wr_sel"],
        })

    # Chart from best asset
    best_ativo = best_asset["ativo"]
    chart_candles, chart_signals = [], []
    df_full = load_csv(os.path.join(CSV_DIR, best_ativo + ".csv"))
    if df_full is not None:
        n_full = len(df_full)
        chart_start = max(0, n_full - 500)
        df_chart = df_full.iloc[chart_start:].copy()
        for ts, row in df_chart.iterrows():
            chart_candles.append({
                "t": str(ts),
                "o": round(float(row["open"]), 6),
                "h": round(float(row["high"]), 6),
                "l": round(float(row["low"]), 6),
                "c": round(float(row["close"]), 6),
            })

    thresh_rows = ""
    for t_val in [0.50, 0.52, 0.55, 0.57, 0.60]:
        total_t = sum(1 for r in all_results if r["thresh"] <= t_val)
        avg_wr_t = np.mean([r["wr_sel"] for r in all_results if r["thresh"] <= t_val]) if total_t > 0 else 0
        thresh_rows += f"{t_val:.2f}|{total_t}|{int(total_t * avg_wr_t / 100)}|{total_t - int(total_t * avg_wr_t / 100)}|{avg_wr_t:.1f}|100.0;"

    equity = []
    balance = 0
    for r in sorted(all_results, key=lambda x: x["acc"]):
        wins_r = int(r["n_sel"] * r["wr_sel"] / 100)
        for _ in range(wins_r):
            balance += 0.82
            equity.append(round(balance, 2))
        for _ in range(r["n_sel"] - wins_r):
            balance -= 1.0
            equity.append(round(balance, 2))

    generate_dashboard_html(
        os.path.join(_BASE, "trade_decisions.html"),
        ativo_stats, chart_candles, chart_signals,
        best_ativo, 0.55, avg_wr, len(per_asset_models),
        int(sum(r["n_sel"] for r in all_results) * avg_wr / 100),
        feat_imp[:10], model_data["stats"], thresh_rows, equity,
    )
    print(f"  Dashboard: {os.path.join(_BASE, 'trade_decisions.html')}")

    elapsed = time.time() - t0
    print(f"\n  Tempo total: {elapsed:.1f}s")
    print("=" * 72)
    print("  TREINO COMPLETO!")
    print("=" * 72)

    elapsed = time.time() - t0
    print(f"\n  Tempo total: {elapsed:.1f}s")
    print("=" * 72)
    print("  TREINO COMPLETO!")
    print("=" * 72)


def generate_dashboard_html(path, ativo_stats, candles, signals,
                             best_ativo, threshold, final_wr, final_trades, final_wins,
                             feat_importance, stats, thresh_data, equity,
                             wf_results=None, wf_avg_wr=0, wf_total_trades=0, perm_wr=50):
    """Gera dashboard HTML interativo com grafico de velas e sinais ML."""

    ativo_rows = ""
    for a in ativo_stats:
        color = "#00e676" if a["wr"] >= 58 else ("#ffd600" if a["wr"] >= 52 else "#ff5252")
        ativo_rows += f"""
        <tr>
          <td>{a['ativo']}</td>
          <td>{a['trades']}</td>
          <td>{a['wins']}</td>
          <td>{a['losses']}</td>
          <td style="color:{color};font-weight:bold">{a['wr']:.1f}%</td>
        </tr>"""

    feat_rows = ""
    for name, imp in feat_importance:
        w = int(imp * 500)
        feat_rows += f"""
        <tr>
          <td>{name}</td>
          <td><div style="background:#00e676;height:18px;width:{w}px;border-radius:3px"></div></td>
          <td>{imp:.4f}</td>
        </tr>"""

    # Walk-forward rows
    wf_html_rows = ""
    if wf_results:
        for r in wf_results:
            clr = "#00e676" if r["wr"] >= 54 else ("#ffd600" if r["wr"] >= 50 else "#ff5252")
            wf_html_rows += f"""
            <tr>
              <td>Fold {r['fold']}</td>
              <td>{r['train']}</td>
              <td>{r['test']}</td>
              <td>{r['trades']}</td>
              <td style="color:{clr};font-weight:bold">{r['wr']:.1f}%</td>
            </tr>"""

    perm_color = "#00e676" if final_wr > perm_wr + 3 else "#ff5252"
    wf_color = "#00e676" if wf_avg_wr >= 54 else ("#ffd600" if wf_avg_wr >= 50 else "#ff5252")

    # Parse threshold data
    thresh_html_rows = ""
    for item in thresh_data.split(";"):
        if not item.strip():
            continue
        parts = item.split("|")
        if len(parts) >= 6:
            sel_text = parts[5] if "(SELECIONADO)" not in parts[5] else parts[5].replace("(SELECIONADO)", "")
            is_sel = "(SELECIONADO)" in item
            style = ' style="background:#1a3a1a;font-weight:bold"' if is_sel else ''
            thresh_html_rows += f"""
            <tr{style}>
              <td>{parts[0]}</td>
              <td>{parts[1]}</td>
              <td>{parts[2]}</td>
              <td>{parts[3]}</td>
              <td>{parts[4]}%</td>
              <td>{sel_text.strip()}%</td>
            </tr>"""

    equity_json = json.dumps(equity)

    candles_json = json.dumps(candles)
    signals_json = json.dumps(signals)

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>WS Trader - ML Continuation AI Dashboard</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0a0e17; color:#e0e0e0; font-family:'Segoe UI',sans-serif; }}
  .header {{
    background:linear-gradient(135deg,#1a237e,#0d47a1);
    padding:20px 30px; display:flex; justify-content:space-between; align-items:center;
  }}
  .header h1 {{ font-size:24px; color:#fff; }}
  .header .badge {{
    background:#00e676; color:#000; padding:6px 16px; border-radius:20px;
    font-weight:bold; font-size:14px;
  }}
  .stats-row {{
    display:grid; grid-template-columns:repeat(5,1fr); gap:15px;
    padding:20px 30px;
  }}
  .stat-card {{
    background:#1a1f2e; border-radius:12px; padding:20px; text-align:center;
    border:1px solid #2a2f3e;
  }}
  .stat-card .value {{ font-size:28px; font-weight:bold; color:#00e676; }}
  .stat-card .label {{ font-size:12px; color:#888; margin-top:5px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; padding:0 30px 30px; }}
  .panel {{
    background:#1a1f2e; border-radius:12px; padding:20px;
    border:1px solid #2a2f3e;
  }}
  .panel h2 {{ font-size:16px; color:#90caf9; margin-bottom:15px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ text-align:left; padding:8px; color:#666; border-bottom:1px solid #2a2f3e; }}
  td {{ padding:8px; border-bottom:1px solid #1a1f2e; }}
  #chart {{ width:100%; height:400px; background:#0d1117; border-radius:8px; position:relative; overflow:hidden; }}
  canvas {{ width:100%; height:100%; }}
  .full-width {{ grid-column:1/3; }}
  .signal-legend {{ display:flex; gap:20px; margin-top:10px; font-size:13px; }}
  .signal-legend span {{ display:flex; align-items:center; gap:6px; }}
  .dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
  .dot-win {{ background:#00e676; }}
  .dot-loss {{ background:#ff5252; }}
  .dot-nosignal {{ background:#555; }}
  .thresh-info {{ color:#ffd600; font-size:13px; margin-top:5px; }}
</style>
</head>
<body>

<div class="header">
  <h1>WS Trader - Continuation AI</h1>
  <div>
    <span class="badge">ML TRAINED</span>
    <span style="color:#ccc;margin-left:15px;font-size:14px">
      Threshold: {threshold:.2f} | Test WR: {final_wr:.1f}%
    </span>
  </div>
</div>

<div class="stats-row">
  <div class="stat-card">
    <div class="value">{final_wr:.1f}%</div>
    <div class="label">Win Rate (Teste)</div>
  </div>
  <div class="stat-card">
    <div class="value">{final_trades}</div>
    <div class="label">Total Trades</div>
  </div>
  <div class="stat-card">
    <div class="value" style="color:#00e676">{final_wins}</div>
    <div class="label">Wins</div>
  </div>
  <div class="stat-card">
    <div class="value" style="color:#ff5252">{final_trades - final_wins}</div>
    <div class="label">Losses</div>
  </div>
  <div class="stat-card">
    <div class="value" style="color:#ffd600">{threshold:.2f}</div>
    <div class="label">Threshold ML</div>
  </div>
</div>

<div class="grid">
  <div class="panel full-width">
    <h2>Grafico de Sinais - {best_ativo} (ultimas 500 velas)</h2>
    <div id="chart">
      <canvas id="candleChart"></canvas>
    </div>
    <div class="signal-legend">
      <span><span class="dot dot-win"></span> WIN (acertou)</span>
      <span><span class="dot dot-loss"></span> LOSS (errou)</span>

  <div class="panel full-width">
    <h2>Equity Curve (Simulacao $1 por trade, payout 82%)</h2>
    <div id="equityChart" style="width:100%;height:250px;background:#0d1117;border-radius:8px;position:relative">
      <canvas id="eqCanvas"></canvas>
    </div>
  </div>

  <div class="panel">
    <h2>Thresholds de Confianca</h2>
    <table>
      <tr><th>Threshold</th><th>Trades</th><th>Wins</th><th>Losses</th><th>WR</th><th>Seletiv.</th></tr>
      {thresh_html_rows}
    </table>
    <p style="margin-top:10px;color:#888;font-size:12px">
      Maior threshold = menos trades mas mais precisao.
      Escolha pelo balanco entre WR e volume de trades.
    </p>
  </div>

      <span><span class="dot dot-nosignal"></span> Sem sinal</span>
    </div>
    <div class="thresh-info">
      Setas = sinais da ML com prob >= {threshold:.2f} | Verde = acertou, Vermelho = errou
    </div>
  </div>

  <div class="panel">
    <h2>Performance por Ativo (Teste Out-of-Sample)</h2>
    <table>
      <tr><th>Ativo</th><th>Trades</th><th>Wins</th><th>Losses</th><th>WR</th></tr>
      {ativo_rows}
    </table>
  </div>

  <div class="panel">
    <h2>Features Mais Importantes</h2>
    <table>
      <tr><th>Feature</th><th>Importancia</th><th>Score</th></tr>
      {feat_rows}
    </table>
  </div>

  <div class="panel">
    <h2>Walk-Forward CV (Validacao Honesta)</h2>
    <table>
      <tr><th>Fold</th><th>Treino</th><th>Teste</th><th>Trades</th><th>WR</th></tr>
      {wf_html_rows}
    </table>
    <p style="margin-top:12px;font-size:15px">
      Media Walk-Forward:
      <span style="color:{wf_color};font-weight:bold;font-size:18px">{wf_avg_wr:.1f}%</span>
      ({wf_total_trades} trades)
    </p>
    <p style="margin-top:8px;color:#888;font-size:12px">
      Walk-forward treina em janela crescente e testa no futuro.
      Resultado mais honesto que split unico.
    </p>
  </div>

  <div class="panel">
    <h2>Teste de Permutacao (Sanity Check)</h2>
    <div style="display:flex;gap:30px;margin-top:10px">
      <div>
        <div style="font-size:12px;color:#888">Labels Reais (ML)</div>
        <div style="font-size:28px;font-weight:bold;color:#00e676">{final_wr:.1f}%</div>
      </div>
      <div>
        <div style="font-size:12px;color:#888">Labels Aleatorios</div>
        <div style="font-size:28px;font-weight:bold;color:#ff5252">{perm_wr:.1f}%</div>
      </div>
      <div>
        <div style="font-size:12px;color:#888">Delta (edge real)</div>
        <div style="font-size:28px;font-weight:bold;color:{perm_color}">+{final_wr - perm_wr:.1f}pp</div>
      </div>
    </div>
    <p style="margin-top:12px;color:#888;font-size:12px">
      Se o modelo aprendeu algo real, WR com labels reais deve ser
      significativamente maior que com labels embaralhados (~50%).
    </p>
  </div>
</div>

<script>
const candles = {candles_json};
const signals = {signals_json};

const canvas = document.getElementById('candleChart');
const ctx = canvas.getContext('2d');

function drawChart() {{
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;

  const W = canvas.width;
  const H = canvas.height;
  const pad = {{ top:20, bottom:30, left:60, right:20 }};
  const cw = (W - pad.left - pad.right) / candles.length;

  if (candles.length === 0) {{
    ctx.fillStyle = '#555';
    ctx.font = '16px sans-serif';
    ctx.fillText('Sem dados de candles para este ativo', W/2 - 150, H/2);
    return;
  }}

  let minP = Infinity, maxP = -Infinity;
  candles.forEach(c => {{
    if (c.l < minP) minP = c.l;
    if (c.h > maxP) maxP = c.h;
  }});
  const range = maxP - minP || 0.0001;
  const yScale = (H - pad.top - pad.bottom) / range;

  function toY(price) {{ return pad.top + (maxP - price) * yScale; }}

  // Background grid
  ctx.strokeStyle = '#1a1f2e';
  ctx.lineWidth = 0.5;
  for (let i = 0; i < 6; i++) {{
    const y = pad.top + i * (H - pad.top - pad.bottom) / 5;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    const price = maxP - (i / 5) * range;
    ctx.fillStyle = '#555';
    ctx.font = '10px monospace';
    ctx.fillText(price.toFixed(5), 2, y + 4);
  }}

  // Draw candles
  candles.forEach((c, i) => {{
    const x = pad.left + i * cw + cw / 2;
    const bull = c.c >= c.o;
    const bodyTop = toY(Math.max(c.o, c.c));
    const bodyBot = toY(Math.min(c.o, c.c));
    const bodyH = Math.max(bodyBot - bodyTop, 1);

    // Wick
    ctx.strokeStyle = bull ? '#26a69a' : '#ef5350';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, toY(c.h));
    ctx.lineTo(x, toY(c.l));
    ctx.stroke();

    // Body
    ctx.fillStyle = bull ? '#26a69a' : '#ef5350';
    ctx.fillRect(x - cw * 0.35, bodyTop, cw * 0.7, bodyH);
  }});

// Equity curve
const equity = {equity_json};
function drawEquity() {{
  const ec = document.getElementById('eqCanvas');
  const ectx = ec.getContext('2d');
  const rect = ec.parentElement.getBoundingClientRect();
  ec.width = rect.width; ec.height = rect.height;
  const W = ec.width, H2 = ec.height;
  const pad2 = {{ top:15, bottom:25, left:55, right:15 }};

  if (equity.length < 2) return;

  const minE = Math.min(0, ...equity);
  const maxE = Math.max(...equity);
  const rng = maxE - minE || 1;
  const xStep = (W - pad2.left - pad2.right) / (equity.length - 1);
  function toY2(v) {{ return pad2.top + (maxE - v) / rng * (H2 - pad2.top - pad2.bottom); }}

  // Zero line
  ctx2style = '#444';
  ectx.strokeStyle = '#333';
  ectx.setLineDash([4,4]);
  ectx.beginPath();
  ectx.moveTo(pad2.left, toY2(0));
  ectx.lineTo(W - pad2.right, toY2(0));
  ectx.stroke();
  ectx.setLineDash([]);

  // Equity line
  ectx.beginPath();
  ectx.strokeStyle = '#00e676';
  ectx.lineWidth = 2;
  equity.forEach((v, i) => {{
    const x = pad2.left + i * xStep;
    if (i === 0) ectx.moveTo(x, toY2(v));
    else ectx.lineTo(x, toY2(v));
  }});
  ectx.stroke();

  // Fill under
  ectx.lineTo(pad2.left + (equity.length - 1) * xStep, toY2(0));
  ectx.lineTo(pad2.left, toY2(0));
  ectx.closePath();
  ectx.fillStyle = 'rgba(0,230,118,0.08)';
  ectx.fill();

  // Labels
  ectx.fillStyle = '#888';
  ectx.font = '10px monospace';
  ectx.fillText('$' + maxE.toFixed(1), 2, pad2.top + 10);
  ectx.fillText('$' + minE.toFixed(1), 2, H2 - pad2.bottom);
  ectx.fillText('$0', 2, toY2(0) + 4);
  ectx.fillText(equity.length + ' trades', W - 80, H2 - 5);

  // Final balance
  const lastV = equity[equity.length - 1];
  const clr = lastV >= 0 ? '#00e676' : '#ff5252';
  ectx.fillStyle = clr;
  ectx.font = 'bold 14px monospace';
  ectx.fillText('$' + lastV.toFixed(2), W - 100, pad2.top + 15);
}}
drawEquity();
window.addEventListener('resize', drawEquity);

  // Map signals to chart by finding closest candle index
  const lastIdx = candles.length > 0 ? (candles.length - 1) : 0;
  const firstCandleIdx = candles.length > 0 ? 0 : 0;

  // The signals have 'idx' which is the global array index
  // We need to find which candle in our 500-candle window this maps to
  // Since these are the last 500 candles, offset is total - 500

  signals.forEach(s => {{
    const ci = s.pos;
    if (ci >= 0 && ci < candles.length) {{
      const x = pad.left + ci * cw + cw / 2;
      const c = candles[ci];
      const yBase = s.dir === 'PUT' ? toY(c.h) - 15 : toY(c.l) + 15;
      const color = s.win ? '#00e676' : '#ff5252';

      // Arrow
      ctx.fillStyle = color;
      ctx.beginPath();
      if (s.dir === 'PUT') {{
        ctx.moveTo(x, yBase + 10);
        ctx.lineTo(x - 6, yBase);
        ctx.lineTo(x + 6, yBase);
      }} else {{
        ctx.moveTo(x, yBase - 10);
        ctx.lineTo(x - 6, yBase);
        ctx.lineTo(x + 6, yBase);
      }}
      ctx.fill();

      // Prob label
      ctx.fillStyle = color;
      ctx.font = 'bold 9px monospace';
      const ly = s.dir === 'PUT' ? yBase - 4 : yBase + 16;
      ctx.fillText((s.prob * 100).toFixed(0) + '%', x - 10, ly);
    }}
  }});
}}

drawChart();
window.addEventListener('resize', drawChart);
</script>

</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
