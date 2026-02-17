def rejection_trigger(df, i, side, min_wick_ratio=1.7):
    if i < 1 or i >= len(df) - 1:
        return False

    c = df.iloc[i]
    nxt = df.iloc[i + 1]

    if c["wick_ratio"] < min_wick_ratio:
        return False

    if side == "SELL":
        if c["upper_wick"] <= c["lower_wick"]:
            return False
        if c["close"] > c["open"]:
            return False
        return nxt["low"] < c["low"]

    if side == "BUY":
        if c["lower_wick"] <= c["upper_wick"]:
            return False
        if c["close"] < c["open"]:
            return False
        return nxt["high"] > c["high"]

    return False


def break_retest_trigger(df, i, zone, side):
    if i < 2 or i >= len(df) - 1:
        return False
    prev = df.iloc[i - 1]
    cur = df.iloc[i]
    nxt = df.iloc[i + 1]

    if side == "BUY":
        if prev["close"] <= zone.high:
            return False
        if not (zone.low <= cur["low"] <= zone.high or zone.low <= cur["close"] <= zone.high):
            return False
        return nxt["high"] > cur["high"]

    if side == "SELL":
        if prev["close"] >= zone.low:
            return False
        if not (zone.low <= cur["high"] <= zone.high or zone.low <= cur["close"] <= zone.high):
            return False
        return nxt["low"] < cur["low"]

    return False
