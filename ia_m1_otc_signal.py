# ia_m1_otc_signal.py
# Requisitos: pip install pandas numpy scikit-learn

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


# -----------------------------
# 1) PROXIES (atividade/"volume" OTC)
# -----------------------------
def add_activity_proxies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Retornos
    out["ret_1"] = out["close"].pct_change()

    # Range
    out["range"] = out["high"] - out["low"]
    out["range_pct"] = out["range"] / out["close"].replace(0, np.nan)

    # True Range
    prev_close = out["close"].shift(1)
    out["true_range"] = np.maximum(
        out["high"] - out["low"],
        np.maximum((out["high"] - prev_close).abs(), (out["low"] - prev_close).abs())
    )

    # Estrutura do candle (rejeicao/indecisao)
    out["body"] = (out["close"] - out["open"]).abs()
    out["upper_wick"] = out["high"] - np.maximum(out["open"], out["close"])
    out["lower_wick"] = np.minimum(out["open"], out["close"]) - out["low"]
    out["wick_ratio"] = (out["upper_wick"] + out["lower_wick"]) / (out["body"] + 1e-9)

    # Pressao do candle (fecha perto do topo/fundo)
    out["pressure"] = (out["close"] - out["open"]) / (out["range"] + 1e-9)

    # Volatilidade curta (atividade)
    out["vol_10"] = out["ret_1"].rolling(10, min_periods=10).std()
    out["vol_20"] = out["ret_1"].rolling(20, min_periods=20).std()

    # "Volume proxy" = atividade + volatilidade
    out["act_raw"] = out["range_pct"].rolling(5, min_periods=5).mean() + out["vol_10"]

    # Normaliza para comparar com o "normal" recente
    mu = out["act_raw"].rolling(200, min_periods=40).mean()
    sd = out["act_raw"].rolling(200, min_periods=40).std()
    out["act_z"] = (out["act_raw"] - mu) / (sd + 1e-9)
    out["act_z"] = out["act_z"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out


# -----------------------------
# 2) FEATURES + ALVO (proximo candle)
# -----------------------------
def add_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ret_3"] = out["close"].pct_change(3)
    out["ret_5"] = out["close"].pct_change(5)

    out["ma_10"] = out["close"].rolling(10).mean()
    out["ma_20"] = out["close"].rolling(20).mean()
    out["ma_dist_10"] = (out["close"] - out["ma_10"]) / out["close"].replace(0, np.nan)
    out["ma_dist_20"] = (out["close"] - out["ma_20"]) / out["close"].replace(0, np.nan)

    # Chop (mercado picotado): alternancia de direcao recente
    sign_ret = np.sign(out["ret_1"].fillna(0))
    out["chop_10"] = (sign_ret.diff().abs() > 0).rolling(10).mean()

    # Alvo: proximo candle sobe?
    out["y"] = (out["close"].shift(-1) > out["close"]).astype(int)

    out = out.dropna().reset_index(drop=True)
    return out


# -----------------------------
# 3) MODELO (baseline forte + estavel)
# -----------------------------
def build_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=40,
            random_state=42,
            n_jobs=-1
        ))
    ])


# -----------------------------
# 4) FILTROS (o que voce pediu)
# -----------------------------
def signal_filters(row, cfg):
    """
    Retorna (ok:bool, reason:str)
    """
    # 1) Atividade minima (act_z > 0)
    if row["act_z"] <= cfg["min_act_z"]:
        return False, "LOW_ACTIVITY"

    # 2) Range precisa estar acima do normal curto
    if row["range_pct"] < row["range_pct_ma"]:
        return False, "SMALL_RANGE"

    # 3) Evitar candle muito indeciso (wick_ratio alto)
    if row["wick_ratio"] >= cfg["max_wick_ratio"]:
        return False, "INDECISION_WICKS"

    # 4) Evitar mercado picotado
    if row["chop_10"] >= cfg["max_chop"]:
        return False, "CHOPPY"

    return True, "OK"


# -----------------------------
# 5) TREINO + SINAIS + LOG + BACKTEST
# -----------------------------
def run_backtest_signals(df: pd.DataFrame, cfg: dict):
    """
    cfg principal:
      - prob_threshold: ex 0.65
      - min_act_z: ex 0.0
      - max_wick_ratio: ex 2.0
      - max_chop: ex 0.65
    """
    work = df.copy()

    # media curta de range_pct para comparar "range acima do normal"
    work["range_pct_ma"] = work["range_pct"].rolling(cfg["range_ma_window"]).mean()

    feature_cols = [
        "ret_1", "ret_3", "ret_5",
        "range_pct", "true_range",
        "wick_ratio", "pressure",
        "vol_10", "vol_20",
        "act_z",
        "ma_dist_10", "ma_dist_20",
        "chop_10"
    ]

    X = work[feature_cols].astype(float).values
    y = work["y"].astype(int).values

    # split temporal (sem misturar futuro no treino)
    split = int(len(work) * cfg["train_ratio"])
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = build_model()
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]  # P(UP)
    test = work.iloc[split:].copy().reset_index(drop=True)
    test["p_up"] = probs
    test["p_dn"] = 1.0 - test["p_up"]

    signals = []
    for _i, row in test.iterrows():
        # prob minima
        conf = max(row["p_up"], row["p_dn"])
        if conf < cfg["prob_threshold"]:
            signals.append(("NO_SIGNAL", conf, "LOW_CONF"))
            continue

        # filtros
        ok, reason = signal_filters(row, cfg)
        if not ok:
            signals.append(("NO_SIGNAL", conf, reason))
            continue

        direction = "UP" if row["p_up"] > row["p_dn"] else "DOWN"
        signals.append((direction, conf, "OK"))

    test["signal"] = [s[0] for s in signals]
    test["conf"] = [s[1] for s in signals]
    test["reason"] = [s[2] for s in signals]

    # Backtest simples: acerto no proximo candle
    test["y_true"] = y_test[:len(test)]
    test["hit"] = np.nan
    mask_up = test["signal"] == "UP"
    mask_dn = test["signal"] == "DOWN"

    test.loc[mask_up, "hit"] = (test.loc[mask_up, "y_true"] == 1).astype(int)
    test.loc[mask_dn, "hit"] = (test.loc[mask_dn, "y_true"] == 0).astype(int)

    taken = test.dropna(subset=["hit"])
    total_trades = len(taken)
    winrate = float(taken["hit"].mean()) if total_trades > 0 else 0.0

    print("\n=== RELATORIO ===")
    print(f"Trades: {total_trades} | Winrate: {winrate:.4f} | Threshold: {cfg['prob_threshold']}")
    print("Bloqueios (top 8):")
    print(test["reason"].value_counts().head(8))

    # Metricas gerais de classificacao (sem filtros, so IA bruta)
    y_pred_raw = (test["p_up"].values >= 0.5).astype(int)
    print("\n=== IA (bruta, sem filtro) ===")
    print(classification_report(test["y_true"].values, y_pred_raw, digits=4))

    # Logs uteis (ultimas linhas)
    cols_log = [
        "open", "high", "low", "close", "signal", "conf", "reason",
        "act_z", "range_pct", "range_pct_ma", "wick_ratio", "chop_10", "pressure", "p_up"
    ]
    print("\n=== ULTIMOS 15 LOGS ===")
    print(test[cols_log].tail(15).to_string(index=False))

    return test


# -----------------------------
# 6) EXEMPLO DE USO
# -----------------------------
if __name__ == "__main__":
    # CSV precisa ter: open, high, low, close
    df = pd.read_csv("candles_m1.csv")

    df = add_activity_proxies(df)
    df = add_features_and_target(df)

    cfg = {
        "train_ratio": 0.70,

        # (o principal) quanto maior, menos trades e mais seletivo
        "prob_threshold": 0.56,

        # filtros
        "min_act_z": -0.20,
        "range_ma_window": 20,
        "max_wick_ratio": 2.8,
        "max_chop": 0.80,
    }

    result = run_backtest_signals(df, cfg)

    # Se quiser salvar:
    # result.to_csv("signals_backtest.csv", index=False)
