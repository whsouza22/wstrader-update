from typing import Optional

from .filters import candle_stats, activity_z, chop_score, pass_filters
from .setups_m1 import rejection_trigger, break_retest_trigger
from .ai_filter import ai_confirms
from .structure_m5 import near_zone, zone_distance


def decide_m1(df_m1, zones_m5, p_up: float, cfg) -> Optional[dict]:
    if df_m1 is None or len(df_m1) < 5:
        return None

    stats = candle_stats(df_m1)
    stats["act_z"] = activity_z(stats)
    stats["chop"] = chop_score(stats, cfg.get("chop_window", 10))
    stats["range_pct_ma"] = stats["range_pct"].rolling(cfg["range_ma_window"], min_periods=10).mean()
    stats["range_ma"] = stats["range"].rolling(14, min_periods=14).mean()

    i = len(stats) - 2
    row = stats.iloc[i + 1]
    price = float(row["close"])

    zS = near_zone(price, zones_m5, "S")
    zR = near_zone(price, zones_m5, "R")

    max_zone_dist_atr = float(cfg.get("max_zone_dist_atr", 0.6))
    range_ma = float(row.get("range_ma", 0.0))
    dist_allow = max_zone_dist_atr * (range_ma if range_ma > 0 else 0.0)

    candidates = []
    if zR and zR.touches >= cfg["min_zone_touches"]:
        candidates.append(("SELL", zR))
    if zS and zS.touches >= cfg["min_zone_touches"]:
        candidates.append(("BUY", zS))

    if not candidates and dist_allow > 0.0:
        for z in zones_m5:
            if z.kind == "S":
                d = zone_distance(price, z)
                if d <= dist_allow:
                    candidates.append(("BUY", z))
                    break
            if z.kind == "R":
                d = zone_distance(price, z)
                if d <= dist_allow:
                    candidates.append(("SELL", z))
                    break

    if not candidates:
        return {"ok": False, "why": "OUTSIDE_ZONE"}

    ok, reason = pass_filters(row, cfg)
    if not ok:
        return {
            "ok": False,
            "why": reason,
            "act_z": row["act_z"],
            "chop": row["chop"],
            "wick": row["wick_ratio"],
            "range_pct": row["range_pct"],
        }

    for side, zone in candidates:
        trig = (
            rejection_trigger(stats, i, side, cfg["min_wick_ratio"])
            or break_retest_trigger(stats, i, zone, side)
        )
        if not trig:
            continue

        ai_ok, ai_reason, conf = ai_confirms(side, p_up, cfg["min_conf"])
        if not ai_ok:
            return {"ok": False, "why": ai_reason, "conf": conf, "side": side}

        dist = zone_distance(price, zone)
        return {
            "ok": True,
            "side": side,
            "conf": conf,
            "zone": zone.kind,
            "touches": zone.touches,
            "dist_zone": dist,
            "price": price,
            "act_z": row["act_z"],
            "chop": row["chop"],
            "wick": row["wick_ratio"],
            "range_pct": row["range_pct"],
            "why": "STRUCTURE+SETUP+AI"
        }

    return {"ok": False, "why": "NO_TRIGGER"}
