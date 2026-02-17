import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Zone:
    kind: str  # "S" suporte ou "R" resistencia
    low: float
    high: float
    touches: int


def pivots(df: pd.DataFrame, left: int = 2, right: int = 2):
    h = df["high"].values
    l = df["low"].values
    n = len(df)
    ph = np.zeros(n, dtype=bool)
    pl = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        if h[i] == np.max(h[i - left:i + right + 1]) and np.sum(h[i] == h[i - left:i + right + 1]) == 1:
            ph[i] = True
        if l[i] == np.min(l[i - left:i + right + 1]) and np.sum(l[i] == l[i - left:i + right + 1]) == 1:
            pl[i] = True
    return ph, pl


def build_zones(df: pd.DataFrame, max_zones: int = 10, tol_mult: float = 0.6) -> List[Zone]:
    if df is None or len(df) < 20:
        return []
    rng = (df["high"] - df["low"]).rolling(14, min_periods=14).mean()
    tol_base = float(rng.iloc[-1] if np.isfinite(rng.iloc[-1]) else rng.mean())
    tol = max(1e-9, tol_base * tol_mult)

    ph, pl = pivots(df)
    pts = []
    for i in np.where(ph)[0]:
        pts.append(("R", float(df["high"].iloc[i])))
    for i in np.where(pl)[0]:
        pts.append(("S", float(df["low"].iloc[i])))
    pts.sort(key=lambda x: x[1])

    zones: List[Zone] = []
    for kind, price in pts:
        z_low, z_high = price - tol, price + tol
        merged = False
        for z in zones:
            if z.kind != kind:
                continue
            if not (z_high < z.low or z_low > z.high):
                z.low = min(z.low, z_low)
                z.high = max(z.high, z_high)
                z.touches += 1
                merged = True
                break
        if not merged:
            zones.append(Zone(kind, z_low, z_high, 1))

    zones.sort(key=lambda z: z.touches, reverse=True)
    return zones[:max_zones]


def near_zone(price: float, zones: List[Zone], kind: Optional[str] = None) -> Optional[Zone]:
    for z in zones:
        if kind and z.kind != kind:
            continue
        if z.low <= price <= z.high:
            return z
    return None


def zone_distance(price: float, zone: Zone) -> float:
    if zone.low <= price <= zone.high:
        return 0.0
    if price < zone.low:
        return zone.low - price
    return price - zone.high
