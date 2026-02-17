import numpy as np
import pandas as pd


def candle_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["range"] = out["high"] - out["low"]
    out["body"] = (out["close"] - out["open"]).abs()
    out["upper_wick"] = out["high"] - np.maximum(out["open"], out["close"])
    out["lower_wick"] = np.minimum(out["open"], out["close"]) - out["low"]
    out["wick_ratio"] = (out["upper_wick"] + out["lower_wick"]) / (out["body"] + 1e-9)
    out["ret_1"] = out["close"].pct_change()
    out["vol_10"] = out["ret_1"].rolling(10, min_periods=10).std()
    out["range_pct"] = out["range"] / out["close"].replace(0, np.nan)
    return out


def activity_z(df: pd.DataFrame) -> pd.Series:
    tmp = df.copy()
    tmp["act_raw"] = tmp["range_pct"].rolling(5, min_periods=5).mean() + tmp["vol_10"]
    mu = tmp["act_raw"].rolling(200, min_periods=40).mean()
    sd = tmp["act_raw"].rolling(200, min_periods=40).std()
    act = (tmp["act_raw"] - mu) / (sd + 1e-9)
    return act.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def chop_score(df: pd.DataFrame, win: int = 10) -> pd.Series:
    s = np.sign(df["ret_1"].fillna(0))
    return ((s.diff().abs() > 0).rolling(win, min_periods=win).mean()).fillna(0.0)


def pass_filters(row, cfg):
    if row["act_z"] <= cfg["min_act_z"]:
        return False, "LOW_ACTIVITY"
    if row["chop"] >= cfg["max_chop"]:
        return False, "CHOPPY"
    if row["wick_ratio"] >= cfg["max_wick_ratio"]:
        return False, "INDECISION_WICKS"
    min_range_ratio = float(cfg.get("min_range_ratio", 1.0))
    if row["range_pct"] < (row["range_pct_ma"] * min_range_ratio):
        return False, "SMALL_RANGE"
    return True, "OK"
