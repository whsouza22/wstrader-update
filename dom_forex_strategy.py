# -*- coding: utf-8 -*-
"""
DOM FOREX PERFECT ZONES — Zonas de S/R + LTA/LTB + Confluência
Para Opções Binárias OTC M1 (IQ Option)

Baseado na metodologia DOM Forex Perfect Zones:
- Detecta ZONAS (retângulos) de Suporte e Resistência com Weight/Força
  (múltiplas reações, rejeições, clusters de preço)
- Identifica LTA (Linha de Tendência de Alta) e LTB (Linha de Tendência de Baixa)
- Analisa padrões de candle (rejeição, engolfo) nos níveis
- RSI como CONFLUÊNCIA (overbought/oversold reforça rejeição na zona)
- Confirmação de 2 velas: vela anterior toca zona, vela atual confirma direção
- Entrada SOMENTE quando há confluência de 2+ fatores

3 Setups:
1. REJEIÇÃO DIRETA: Preço toca zona forte + candle de rejeição + LTA/LTB
2. ROMPIMENTO: Preço rompe zona com corpo grande → continuação
3. PULLBACK PÓS-ROMPIMENTO: Rompe zona + volta nela + confirma

Zona sozinha NÃO é sinal. É LOCAL de decisão.
"""

import pandas as pd
import numpy as np
import math
import logging
from typing import Dict, Any, List, Optional, Tuple

log = logging.getLogger("WS_AUTO_AI")

# ═══════════════════════════════════════════════════════════════
# CONFIGURAÇÕES DA ESTRATÉGIA
# ═══════════════════════════════════════════════════════════════
SR_LOOKBACK = 200           # Velas para escanear S/R
SR_PIVOT_LEFT = 5           # Barras à esquerda para pivô
SR_PIVOT_RIGHT = 3          # Barras à direita para pivô
SR_CLUSTER_ATR = 0.5        # Tolerância de cluster em ATR (zona, não linha)
SR_MIN_TOUCHES = 2          # Toques mínimos para zona válida
SR_PROXIMITY_ATR = 0.50     # Preço dentro da zona (distância máx em ATR)
SR_STRONG_TOUCHES = 3       # Toques para zona "forte"
SR_VERY_STRONG = 5          # Toques para zona "muito forte"

LT_MIN_POINTS = 2           # Pontos mínimos para trendline
LT_TOLERANCE_ATR = 0.35     # Tolerância de toque na trendline
LT_MAX_BREAK_ATR = 0.5      # Máximo de violação
LT_PROXIMITY_ATR = 0.5      # Distância máxima da trendline

RSI_PERIOD = 14             # Período do RSI
RSI_OVERSOLD = 30           # RSI oversold (confluência CALL em suporte)
RSI_OVERBOUGHT = 70         # RSI overbought (confluência PUT em resistência)

MIN_CANDLES = 100           # Mínimo de velas para análise


# ═══════════════════════════════════════════════════════════════
# DETECÇÃO DE PIVÔS
# ═══════════════════════════════════════════════════════════════
def find_pivot_points(df: pd.DataFrame, left: int = SR_PIVOT_LEFT,
                      right: int = SR_PIVOT_RIGHT) -> List[Dict[str, Any]]:
    """
    Detecta pivot highs e pivot lows no DataFrame.
    Pivot high: máxima mais alta que as 'left' anteriores e 'right' seguintes.
    Pivot low: mínima mais baixa que as 'left' anteriores e 'right' seguintes.
    """
    pivots = []
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values

    for i in range(left, len(df) - right):
        # Pivot HIGH
        is_high = True
        for j in range(1, left + 1):
            if highs[i] < highs[i - j]:
                is_high = False
                break
        if is_high:
            for j in range(1, right + 1):
                if highs[i] < highs[i + j]:
                    is_high = False
                    break
        if is_high:
            pivots.append({
                "type": "high",
                "index": i,
                "price": float(highs[i]),
                "abs_index": i,
            })

        # Pivot LOW
        is_low = True
        for j in range(1, left + 1):
            if lows[i] > lows[i - j]:
                is_low = False
                break
        if is_low:
            for j in range(1, right + 1):
                if lows[i] > lows[i + j]:
                    is_low = False
                    break
        if is_low:
            pivots.append({
                "type": "low",
                "index": i,
                "price": float(lows[i]),
                "abs_index": i,
            })

    return pivots


# ═══════════════════════════════════════════════════════════════
# RSI (Relative Strength Index) — confluência em zonas S/R
# ═══════════════════════════════════════════════════════════════
def calculate_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> float:
    """
    Calcula RSI (Relative Strength Index) para confluência.
    RSI < 30 = oversold (favorece CALL em suporte)
    RSI > 70 = overbought (favorece PUT em resistência)
    """
    if len(df) < period + 2:
        return 50.0  # Neutro se dados insuficientes

    closes = df["close"].astype(float).values[-(period + 1):]
    deltas = np.diff(closes)

    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0

    if avg_loss < 1e-9:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def check_prev_candle_zone_touch(
    df: pd.DataFrame, sr_zone: Dict[str, Any], atr_val: float
) -> Dict[str, Any]:
    """
    Verifica se a VELA ANTERIOR (penúltima fechada) tocou a zona S/R.
    DOM Forex: não entra só pelo toque — espera a vela seguinte confirmar.

    Retorna:
      touched: bool — se a vela anterior tocou a borda da zona
      touch_type: str — 'wick' (pavio) ou 'body' (corpo)
      rejected: bool — se a vela anterior já mostrou rejeição
    """
    result = {"touched": False, "touch_type": "none", "rejected": False}

    if len(df) < 3 or not sr_zone:
        return result

    prev = df.iloc[-2]  # Penúltima vela (fechada)
    p_open = float(prev["open"])
    p_high = float(prev["high"])
    p_low = float(prev["low"])
    p_close = float(prev["close"])
    margin = atr_val * 0.2  # Margem de toque: 20% do ATR

    zone_low = sr_zone["zone_low"]
    zone_high = sr_zone["zone_high"]
    zone_type = sr_zone.get("type", "")

    if zone_type == "support":
        # Suporte: a low da vela anterior deve ter tocado a zona
        if p_low <= zone_high + margin:
            result["touched"] = True
            # Corpo tocou ou só pavio?
            body_low = min(p_open, p_close)
            if body_low <= zone_high + margin:
                result["touch_type"] = "body"
            else:
                result["touch_type"] = "wick"
            # Rejeição: fechou ACIMA da zona
            if p_close > zone_high:
                result["rejected"] = True

    elif zone_type == "resistance":
        # Resistência: a high da vela anterior deve ter tocado a zona
        if p_high >= zone_low - margin:
            result["touched"] = True
            body_high = max(p_open, p_close)
            if body_high >= zone_low - margin:
                result["touch_type"] = "body"
            else:
                result["touch_type"] = "wick"
            # Rejeição: fechou ABAIXO da zona
            if p_close < zone_low:
                result["rejected"] = True

    return result


# ═══════════════════════════════════════════════════════════════
# DETECÇÃO DE PERFECT ZONES (S/R)
# ═══════════════════════════════════════════════════════════════
def detect_sr_zones(df: pd.DataFrame, atr_val: float,
                    lookback: int = SR_LOOKBACK) -> List[Dict[str, Any]]:
    """
    Detecta zonas de Suporte e Resistência estilo DOM Forex Perfect Zones.

    Cada zona é um RETÂNGULO (range), não uma linha.
    Weight (W) = força da zona baseada em:
    - Quantidade de toques
    - Intensidade de rejeição
    - Recência
    - Falsos rompimentos (que voltaram = zona mais forte)

    Retorna lista de zonas ordenadas por Weight:
    [{"price": float, "zone_high": float, "zone_low": float,
      "touches": int, "weight": float, "type": str,
      "rejections": int, "false_breaks": int, "recency": float}]
    """
    if len(df) < 20:
        return []

    df_scan = df.iloc[-min(lookback, len(df)):]
    pivots = find_pivot_points(df_scan)

    if not pivots:
        return []

    cluster_tol = atr_val * SR_CLUSTER_ATR
    zones = []
    used = set()

    sorted_pivots = sorted(pivots, key=lambda p: p["price"])

    for i, p in enumerate(sorted_pivots):
        if i in used:
            continue
        cluster = [p]
        used.add(i)

        for j in range(i + 1, len(sorted_pivots)):
            if j in used:
                continue
            if abs(sorted_pivots[j]["price"] - p["price"]) <= cluster_tol:
                cluster.append(sorted_pivots[j])
                used.add(j)
            elif sorted_pivots[j]["price"] - p["price"] > cluster_tol * 2:
                break

        if len(cluster) >= SR_MIN_TOUCHES:
            prices = [c["price"] for c in cluster]
            zone_high = max(prices)
            zone_low = min(prices)
            avg_price = sum(prices) / len(prices)
            last_idx = max(c["index"] for c in cluster)
            total_candles = len(df_scan)
            recency = last_idx / max(total_candles - 1, 1)

            n_highs = sum(1 for c in cluster if c["type"] == "high")
            n_lows = sum(1 for c in cluster if c["type"] == "low")

            # Contar rejeições (toques com retorno forte)
            rejections = 0
            false_breaks = 0
            closes = df_scan["close"].astype(float).values
            highs_arr = df_scan["high"].astype(float).values
            lows_arr = df_scan["low"].astype(float).values

            for idx in range(max(0, last_idx - 50), min(total_candles - 1, last_idx + 1)):
                candle_h = highs_arr[idx]
                candle_l = lows_arr[idx]
                candle_c = closes[idx]

                touched = (candle_l <= zone_high + cluster_tol * 0.3 and
                           candle_h >= zone_low - cluster_tol * 0.3)
                if not touched:
                    continue

                # Rejeição = preço entrou na zona mas fechou fora
                if candle_c > zone_high or candle_c < zone_low:
                    rejections += 1

                # Falso rompimento = preço passou da zona mas voltou
                if idx + 1 < total_candles:
                    next_c = closes[idx + 1]
                    if candle_h > zone_high and next_c < zone_high:
                        false_breaks += 1
                    elif candle_l < zone_low and next_c > zone_low:
                        false_breaks += 1

            # ── WEIGHT (Força da zona) ──
            weight = len(cluster) * 1.0
            weight += rejections * 0.5
            weight += false_breaks * 0.8
            weight *= (0.7 + recency * 0.3)
            if n_highs > 0 and n_lows > 0:
                weight *= 1.2

            zone_type = "resistance" if n_highs >= n_lows else "support"

            zones.append({
                "price": avg_price,
                "zone_high": zone_high,
                "zone_low": zone_low,
                "range_pts": zone_high - zone_low,
                "touches": len(cluster),
                "weight": weight,
                "strength": weight,
                "type": zone_type,
                "last_touch_idx": last_idx,
                "recency": recency,
                "rejections": rejections,
                "false_breaks": false_breaks,
                "n_highs": n_highs,
                "n_lows": n_lows,
            })

    zones.sort(key=lambda z: z["weight"], reverse=True)
    return zones


def find_nearest_sr(zones: List[Dict[str, Any]], price: float,
                    atr_val: float, direction: str = "any") -> Optional[Dict[str, Any]]:
    """
    Encontra a zona S/R mais próxima do preço atual.
    Verifica se o preço está DENTRO da zona ou muito perto.
    """
    best = None
    best_dist = float("inf")

    for z in zones:
        dist_center = abs(price - z["price"])
        if price >= z["zone_low"] and price <= z["zone_high"]:
            dist_edge = 0.0
        elif price > z["zone_high"]:
            dist_edge = price - z["zone_high"]
        else:
            dist_edge = z["zone_low"] - price

        if dist_edge > atr_val * SR_PROXIMITY_ATR:
            continue

        if direction == "support" and price < z["zone_low"]:
            continue
        if direction == "resistance" and price > z["zone_high"]:
            continue

        # FIX: Quando preço está dentro ou muito perto da zona, a mesma zona
        # passava como support E resistance, permitindo PUT em suporte.
        # Agora filtra pelo tipo da zona quando preço está próximo.
        if dist_edge < atr_val * 0.5:  # Preço perto ou dentro da zona
            if direction == "support" and z["type"] == "resistance":
                continue  # Zona é resistência, não usar como suporte
            if direction == "resistance" and z["type"] == "support":
                continue  # Zona é suporte, não usar como resistência

        if dist_center < best_dist:
            best_dist = dist_center
            best = {
                **z,
                "distance": dist_edge,
                "distance_atr": dist_edge / max(atr_val, 1e-9),
                "inside_zone": dist_edge == 0.0,
            }

    return best


# ═══════════════════════════════════════════════════════════════
# ZONA ROMPIDA RECENTEMENTE (setup pullback pós-rompimento)
# ═══════════════════════════════════════════════════════════════
def find_recently_broken_zone(zones: List[Dict[str, Any]], df: pd.DataFrame,
                              atr_val: float, lookback: int = 10) -> Optional[Dict[str, Any]]:
    """
    Procura zona rompida recentemente onde o preço está voltando.
    Setup 3: PULLBACK PÓS-ROMPIMENTO (o mais seguro segundo Dom Forex).
    """
    if len(df) < lookback + 5:
        return None

    closes = df["close"].astype(float).values
    current_price = closes[-1]

    for z in zones:
        zone_high = z["zone_high"]
        zone_low = z["zone_low"]

        broke_up = False
        broke_down = False

        for i in range(-lookback, -2):
            c = closes[i]
            if c < zone_low and closes[i + 1] > zone_high:
                broke_up = True
            if c > zone_high and closes[i + 1] < zone_low:
                broke_down = True

        if broke_up and zone_low <= current_price <= zone_high + atr_val * 0.3:
            return {
                **z,
                "break_direction": "up",
                "setup_type": "pullback_after_breakout",
                "signal_dir": "CALL",
                "distance_atr": abs(current_price - z["price"]) / max(atr_val, 1e-9),
            }

        if broke_down and zone_low - atr_val * 0.3 <= current_price <= zone_high:
            return {
                **z,
                "break_direction": "down",
                "setup_type": "pullback_after_breakout",
                "signal_dir": "PUT",
                "distance_atr": abs(current_price - z["price"]) / max(atr_val, 1e-9),
            }

    return None


# ═══════════════════════════════════════════════════════════════
# DETECÇÃO DE LTA (Linha de Tendência de Alta)
# ═══════════════════════════════════════════════════════════════
def detect_lta(df: pd.DataFrame, atr_val: float,
               min_points: int = LT_MIN_POINTS) -> Optional[Dict[str, Any]]:
    """Detecta LTA conectando swing lows ascendentes."""
    if len(df) < 30:
        return None

    df_scan = df.iloc[-min(150, len(df)):]
    pivots = [p for p in find_pivot_points(df_scan, left=4, right=3) if p["type"] == "low"]

    if len(pivots) < min_points:
        return None

    n = len(df_scan)
    tol = atr_val * LT_TOLERANCE_ATR
    best_line = None
    best_score = 0

    for i in range(len(pivots)):
        for j in range(i + 1, len(pivots)):
            idx_i = pivots[i]["index"]
            idx_j = pivots[j]["index"]
            price_i = pivots[i]["price"]
            price_j = pivots[j]["price"]

            if idx_j == idx_i:
                continue
            slope = (price_j - price_i) / (idx_j - idx_i)
            if slope <= 0:
                continue

            intercept = price_i - slope * idx_i

            points_on = 0
            max_break = 0
            for p in pivots:
                expected = slope * p["index"] + intercept
                diff = p["price"] - expected
                if abs(diff) <= tol:
                    points_on += 1
                if diff < -tol:
                    max_break = max(max_break, abs(diff))

            if max_break > atr_val * LT_MAX_BREAK_ATR:
                continue

            if points_on >= min_points:
                recency = max(idx_i, idx_j) / max(n - 1, 1)
                score = points_on * 2 + recency * 3 + min(slope / max(atr_val, 1e-9), 1.0)

                if score > best_score:
                    best_score = score
                    current_val = slope * (n - 1) + intercept
                    current_price = float(df_scan["close"].iloc[-1])
                    proximity = abs(current_price - current_val) / max(atr_val, 1e-9)

                    best_line = {
                        "slope": slope, "intercept": intercept,
                        "points": points_on, "strength": score,
                        "current_value": current_val, "current_price": current_price,
                        "proximity": proximity, "max_break": max_break,
                        "direction": "up",
                    }

    return best_line


# ═══════════════════════════════════════════════════════════════
# DETECÇÃO DE LTB (Linha de Tendência de Baixa)
# ═══════════════════════════════════════════════════════════════
def detect_ltb(df: pd.DataFrame, atr_val: float,
               min_points: int = LT_MIN_POINTS) -> Optional[Dict[str, Any]]:
    """Detecta LTB conectando swing highs descendentes."""
    if len(df) < 30:
        return None

    df_scan = df.iloc[-min(150, len(df)):]
    pivots = [p for p in find_pivot_points(df_scan, left=4, right=3) if p["type"] == "high"]

    if len(pivots) < min_points:
        return None

    n = len(df_scan)
    tol = atr_val * LT_TOLERANCE_ATR
    best_line = None
    best_score = 0

    for i in range(len(pivots)):
        for j in range(i + 1, len(pivots)):
            idx_i = pivots[i]["index"]
            idx_j = pivots[j]["index"]
            price_i = pivots[i]["price"]
            price_j = pivots[j]["price"]

            if idx_j == idx_i:
                continue
            slope = (price_j - price_i) / (idx_j - idx_i)
            if slope >= 0:
                continue

            intercept = price_i - slope * idx_i

            points_on = 0
            max_break = 0
            for p in pivots:
                expected = slope * p["index"] + intercept
                diff = p["price"] - expected
                if abs(diff) <= tol:
                    points_on += 1
                if diff > tol:
                    max_break = max(max_break, abs(diff))

            if max_break > atr_val * LT_MAX_BREAK_ATR:
                continue

            if points_on >= min_points:
                recency = max(idx_i, idx_j) / max(n - 1, 1)
                score = points_on * 2 + recency * 3 + min(abs(slope) / max(atr_val, 1e-9), 1.0)

                if score > best_score:
                    best_score = score
                    current_val = slope * (n - 1) + intercept
                    current_price = float(df_scan["close"].iloc[-1])
                    proximity = abs(current_price - current_val) / max(atr_val, 1e-9)

                    best_line = {
                        "slope": slope, "intercept": intercept,
                        "points": points_on, "strength": score,
                        "current_value": current_val, "current_price": current_price,
                        "proximity": proximity, "max_break": max_break,
                        "direction": "down",
                    }

    return best_line


# ═══════════════════════════════════════════════════════════════
# ANÁLISE DE PADRÃO DE CANDLE (rejeição na zona)
# ═══════════════════════════════════════════════════════════════
def analyze_candle_pattern(df: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    """
    Analisa candle atual para sinais de rejeição.
    Essencial para confirmar que preço está reagindo à zona.
    """
    if len(df) < 3:
        return {"pattern": "none", "direction": "NEUTRAL", "strength": 0.0,
                "body_frac": 0.0, "rejection_wick": 0.0}

    c = df.iloc[-1]
    o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
    rng = h - l
    if rng < 1e-9:
        return {"pattern": "doji_flat", "direction": "NEUTRAL", "strength": 0.0,
                "body_frac": 0.0, "rejection_wick": 0.0}

    body = abs(cl - o)
    body_frac = body / rng
    upper_wick = h - max(o, cl)
    lower_wick = min(o, cl) - l
    upper_frac = upper_wick / rng
    lower_frac = lower_wick / rng

    is_bull = cl > o
    is_bear = cl < o

    pattern = "neutral"
    direction = "NEUTRAL"
    strength = 0.0

    # ── HAMMER / MARTELO (CALL) ──
    if lower_frac >= 0.55 and body_frac <= 0.40 and upper_frac <= 0.15:
        pattern = "hammer"
        direction = "CALL"
        strength = 0.5 + lower_frac * 0.3 + (0.1 if is_bull else 0.0)

    # ── SHOOTING STAR (PUT) ──
    elif upper_frac >= 0.55 and body_frac <= 0.40 and lower_frac <= 0.15:
        pattern = "shooting_star"
        direction = "PUT"
        strength = 0.5 + upper_frac * 0.3 + (0.1 if is_bear else 0.0)

    # ── ENGOLFO BULLISH (CALL) ──
    elif is_bull and body_frac >= 0.55:
        prev = df.iloc[-2]
        prev_body = abs(float(prev["close"]) - float(prev["open"]))
        if float(prev["close"]) < float(prev["open"]) and body > prev_body * 1.2:
            pattern = "bullish_engulfing"
            direction = "CALL"
            strength = 0.5 + body_frac * 0.3

    # ── ENGOLFO BEARISH (PUT) ──
    elif is_bear and body_frac >= 0.55:
        prev = df.iloc[-2]
        prev_body = abs(float(prev["close"]) - float(prev["open"]))
        if float(prev["close"]) > float(prev["open"]) and body > prev_body * 1.2:
            pattern = "bearish_engulfing"
            direction = "PUT"
            strength = 0.5 + body_frac * 0.3

    # ── REJEIÇÃO BULLISH ──
    elif is_bull and lower_frac >= 0.40 and body_frac >= 0.25:
        pattern = "bull_rejection"
        direction = "CALL"
        strength = 0.3 + lower_frac * 0.3

    # ── REJEIÇÃO BEARISH ──
    elif is_bear and upper_frac >= 0.40 and body_frac >= 0.25:
        pattern = "bear_rejection"
        direction = "PUT"
        strength = 0.3 + upper_frac * 0.3

    # ── DOJI ──
    elif body_frac <= 0.10:
        pattern = "doji"
        direction = "NEUTRAL"
        strength = 0.1

    # ── CANDLE FORTE ──
    elif body_frac >= 0.70:
        if is_bull:
            pattern = "strong_bull"
            direction = "CALL"
        else:
            pattern = "strong_bear"
            direction = "PUT"
        strength = 0.4 + body_frac * 0.2

    strength = min(1.0, max(0.0, strength))

    return {
        "pattern": pattern,
        "direction": direction,
        "strength": strength,
        "body_frac": body_frac,
        "rejection_wick": max(upper_frac, lower_frac),
        "upper_frac": upper_frac,
        "lower_frac": lower_frac,
        "is_bull": is_bull,
    }


# ═══════════════════════════════════════════════════════════════
# QUALIDADE DO MERCADO (só price action, zero indicadores)
# ═══════════════════════════════════════════════════════════════
def analyze_market_quality(df: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    """
    Analisa qualidade do mercado usando apenas price action:
    volatilidade, direcionalidade, choppiness.
    """
    if len(df) < 30:
        return {"quality": 0.5, "context": "unknown", "volatility": "normal",
                "directional": 0.5, "chop_frac": 0.5}

    closes = df["close"].astype(float).values
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values

    ranges_recent = highs[-10:] - lows[-10:]
    ranges_avg = highs[-50:] - lows[-50:]
    vol_ratio = float(np.mean(ranges_recent)) / max(float(np.mean(ranges_avg)), 1e-9)

    net_move = abs(closes[-1] - closes[-30])
    total_move = sum(abs(closes[i] - closes[i - 1]) for i in range(-29, 0))
    directional = net_move / max(total_move, 1e-9)

    flips = 0
    for i in range(-19, 0):
        d1 = 1 if closes[i] > closes[i - 1] else -1
        d2 = 1 if closes[i + 1] > closes[i] else -1 if i + 1 < 0 else 0
        if d1 != d2 and d2 != 0:
            flips += 1
    chop_frac = flips / 19.0

    quality = 0.55  # Base mais alto - Dom Forex funciona em mercado choppy (zonas S/R)
    if directional > 0.30:
        quality += 0.15
    elif directional < 0.10:
        quality -= 0.10  # Penalidade reduzida - choppy é normal em OTC M1
    if chop_frac < 0.40:
        quality += 0.10
    elif chop_frac > 0.60:
        quality -= 0.08  # Penalidade reduzida - zonas S/R funcionam em mercado choppy
    if 0.7 < vol_ratio < 1.3:
        quality += 0.05
    elif vol_ratio > 2.0:
        quality -= 0.15
    quality = max(0.0, min(1.0, quality))

    if vol_ratio > 1.5:
        volatility = "high"
    elif vol_ratio < 0.6:
        volatility = "low"
    else:
        volatility = "normal"

    if quality >= 0.60:
        context = "bom"
    elif quality >= 0.40:
        context = "neutro"
    else:
        context = "ruim"

    return {
        "quality": quality, "context": context, "volatility": volatility,
        "directional": directional, "chop_frac": chop_frac, "vol_ratio": vol_ratio,
    }


# ═══════════════════════════════════════════════════════════════
# CÁLCULO DE CONFLUÊNCIA (SEM INDICADORES — só zonas + LT + candle)
# ═══════════════════════════════════════════════════════════════
def calculate_confluence(
    sr_zone: Optional[Dict[str, Any]],
    lta: Optional[Dict[str, Any]],
    ltb: Optional[Dict[str, Any]],
    candle: Dict[str, Any],
    market: Dict[str, Any],
    direction: str,
    atr_val: float,
    pullback_zone: Optional[Dict[str, Any]] = None,
    df_m1: Optional[pd.DataFrame] = None,
    rsi_val: float = 50.0,
) -> Dict[str, Any]:
    """
    Calcula confluência — DOM Forex Perfect Zones.

    Confluências:
    1. Preço em ZONA S/R (peso principal — até 2.5 pts)
    2. LTA/LTB confirmando (até 2.0 pts)
    3. Padrão de candle de rejeição (até 1.0 pt)
    4. Pullback pós-rompimento (1.5 pts bônus)
    5. Qualidade de mercado (até 0.5 pt — filtra lixo)
    6. RSI (até 0.8 pt — oversold/overbought na zona)
    7. Toque na zona pela vela ANTERIOR (até 0.6 pt)
    """
    confluences = []
    total_weight = 0.0
    max_weight = 0.0

    # ── 1. ZONA S/R (Perfect Zone) ──
    sr_weight = 0.0
    edge_touch = False  # FIX: rastrear se candle TOCOU a borda da zona
    if sr_zone:
        dist_atr = sr_zone.get("distance_atr", 999)
        inside = sr_zone.get("inside_zone", False)

        if dist_atr <= SR_PROXIMITY_ATR or inside:
            sr_weight += 1.0
            touches = sr_zone.get("touches", 0)
            confluences.append(f"SR_{sr_zone['type']}({touches}t,W={sr_zone.get('weight',0):.1f})")

            # FIX: Verificar TOQUE NA BORDA + REJEIÇÃO (não entrar no meio da zona)
            # CALL em suporte: candle low toca zona, close acima = rejeição
            # PUT em resistência: candle high toca zona, close abaixo = rejeição
            if df_m1 is not None and len(df_m1) >= 1:
                c_last = df_m1.iloc[-1]
                c_close = float(c_last["close"])
                c_low = float(c_last["low"])
                c_high = float(c_last["high"])
                zone_low = sr_zone["zone_low"]
                zone_high = sr_zone["zone_high"]
                margin = atr_val * 0.3  # margem de toque

                if direction == "CALL":
                    # Candle low toca/penetra a zona de suporte?
                    wick_touches = c_low <= zone_high + margin
                    # Close ficou acima do suporte? (rejeição)
                    close_above = c_close >= zone_low
                    if wick_touches and close_above:
                        edge_touch = True
                        sr_weight += 0.4
                        confluences.append("toque_borda_sup")
                    elif inside and not close_above:
                        # Preço fechou ABAIXO do suporte = rompendo, não comprar
                        sr_weight -= 0.8
                        confluences.append("rompendo_sup")
                elif direction == "PUT":
                    # Candle high toca/penetra a zona de resistência?
                    wick_touches = c_high >= zone_low - margin
                    # Close ficou abaixo da resistência? (rejeição)
                    close_below = c_close <= zone_high
                    if wick_touches and close_below:
                        edge_touch = True
                        sr_weight += 0.4
                        confluences.append("toque_borda_res")
                    elif inside and not close_below:
                        # Preço fechou ACIMA da resistência = rompendo, não vender
                        sr_weight -= 0.8
                        confluences.append("rompendo_res")

                # Preço no meio da zona sem rejeição clara — penalidade leve
                if inside and not edge_touch:
                    candle_confirms = candle.get("direction") == direction and candle.get("strength", 0) >= 0.3
                    if not candle_confirms:
                        sr_weight -= 0.3
                        confluences.append("meio_zona_sem_rejeicao")
            else:
                # Sem df_m1, manter comportamento antigo
                if inside:
                    sr_weight += 0.3
                    confluences.append("dentro_zona")

            if touches >= SR_STRONG_TOUCHES:
                sr_weight += 0.5
                confluences.append(f"zona_forte({touches}t)")
            if touches >= SR_VERY_STRONG:
                sr_weight += 0.4
            if sr_zone.get("rejections", 0) >= 2:
                sr_weight += 0.3
                confluences.append(f"rejeicoes({sr_zone['rejections']})")
            if sr_zone.get("false_breaks", 0) >= 1:
                sr_weight += 0.3
                confluences.append(f"false_break({sr_zone['false_breaks']})")
            if sr_zone.get("recency", 0) >= 0.7:
                sr_weight += 0.2

    sr_weight = max(0.0, sr_weight)  # Garantir que não fique negativo
    total_weight += sr_weight
    max_weight += 2.5

    # ── 2. LTA / LTB ──
    lt_weight = 0.0
    lt_info = None
    lt_touch = False  # candle tocou a trendline?
    if direction == "CALL" and lta:
        lt_info = lta
    elif direction == "PUT" and ltb:
        lt_info = ltb

    if lt_info:
        prox = lt_info.get("proximity", 999)
        if prox <= LT_PROXIMITY_ATR:
            lt_weight += 1.0
            lt_type = "LTA" if direction == "CALL" else "LTB"
            confluences.append(f"{lt_type}({lt_info['points']}pts)")

            if lt_info["points"] >= 3:
                lt_weight += 0.5
                confluences.append(f"{lt_type}_forte")
            if prox <= LT_PROXIMITY_ATR * 0.5:
                lt_weight += 0.3
            if sr_weight > 0:
                lt_weight += 0.3
                confluences.append("zona+LT")

            # ── TOQUE NA TRENDLINE: candle wick toca a LTB/LTA ──
            # PUT na LTB: candle high toca/cruza LTB e close fica abaixo = rejeição
            # CALL na LTA: candle low toca/cruza LTA e close fica acima = rejeição
            if df_m1 is not None and len(df_m1) >= 1:
                c_last = df_m1.iloc[-1]
                c_close = float(c_last["close"])
                c_low = float(c_last["low"])
                c_high = float(c_last["high"])
                lt_val = lt_info["current_value"]
                lt_margin = atr_val * 0.3  # margem de toque

                if direction == "PUT":
                    # PUT: candle high toca/penetra a LTB?
                    if c_high >= lt_val - lt_margin:
                        lt_touch = True
                        lt_weight += 0.5
                        confluences.append("toque_LTB")
                        # Close ficou abaixo da LTB? (rejeição confirmada)
                        if c_close < lt_val:
                            lt_weight += 0.4
                            confluences.append("rejeicao_LTB")
                elif direction == "CALL":
                    # CALL: candle low toca/penetra a LTA?
                    if c_low <= lt_val + lt_margin:
                        lt_touch = True
                        lt_weight += 0.5
                        confluences.append("toque_LTA")
                        # Close ficou acima da LTA? (rejeição confirmada)
                        if c_close > lt_val:
                            lt_weight += 0.4
                            confluences.append("rejeicao_LTA")
    total_weight += lt_weight
    max_weight += 3.1  # Aumentado: 1.0 base + 0.5 forte + 0.3 prox + 0.3 zona+LT + 0.5 toque + 0.4 rejeição + 0.1 margem

    # ── 3. PADRÃO DE CANDLE ──
    candle_weight = 0.0
    if candle.get("direction") == direction and candle.get("strength", 0) >= 0.3:
        candle_weight = candle["strength"]
        confluences.append(f"candle_{candle['pattern']}")
    total_weight += candle_weight
    max_weight += 1.0

    # ── 4. PULLBACK PÓS-ROMPIMENTO ──
    pb_weight = 0.0
    if pullback_zone and pullback_zone.get("signal_dir") == direction:
        pb_weight = 1.5
        confluences.append(f"pullback_{pullback_zone['break_direction']}")
    total_weight += pb_weight
    max_weight += 1.5

    # ── 5. QUALIDADE DE MERCADO ──
    mkt_weight = 0.0
    mkt_quality = market.get("quality", 0.5)
    if mkt_quality >= 0.55:
        mkt_weight = min(0.5, (mkt_quality - 0.40) * 1.0)
    total_weight += mkt_weight
    max_weight += 0.5

    # ── 6. RSI CONFLUÊNCIA ──
    rsi_weight = 0.0
    if sr_zone:
        if direction == "CALL" and rsi_val <= RSI_OVERSOLD:
            # RSI oversold em suporte → forte confluência CALL (bounce provável)
            rsi_weight = 0.5 + min(0.3, (RSI_OVERSOLD - rsi_val) / 30.0)
            confluences.append(f"RSI_oversold({rsi_val:.0f})")
        elif direction == "PUT" and rsi_val >= RSI_OVERBOUGHT:
            # RSI overbought em resistência → forte confluência PUT
            rsi_weight = 0.5 + min(0.3, (rsi_val - RSI_OVERBOUGHT) / 30.0)
            confluences.append(f"RSI_overbought({rsi_val:.0f})")
        elif direction == "CALL" and rsi_val >= RSI_OVERBOUGHT:
            # RSI overbought mas querendo CALL? → penalizar
            rsi_weight = -0.3
            confluences.append(f"RSI_contra({rsi_val:.0f})")
        elif direction == "PUT" and rsi_val <= RSI_OVERSOLD:
            # RSI oversold mas querendo PUT? → penalizar (bounce provável)
            rsi_weight = -0.3
            confluences.append(f"RSI_contra({rsi_val:.0f})")
    total_weight += rsi_weight
    max_weight += 0.8

    # ── 7. CONFIRMAÇÃO DE 2 VELAS (toque anterior + candle atual confirma) ──
    prev_touch_weight = 0.0
    if df_m1 is not None and sr_zone and len(df_m1) >= 3:
        prev_touch = check_prev_candle_zone_touch(df_m1, sr_zone, atr_val)
        if prev_touch["touched"]:
            prev_touch_weight += 0.3
            confluences.append(f"prev_toque_{prev_touch['touch_type']}")
            if prev_touch["rejected"]:
                prev_touch_weight += 0.3
                confluences.append("prev_rejeicao")
        # Sem toque prévio: NÃO penalizar — muitas entradas boas são na primeira vela
        # que toca a zona. A penalidade estava bloqueando oportunidades válidas.
    total_weight += prev_touch_weight
    max_weight += 0.6

    # ── 8. PENALIDADE CONTRA-TENDÊNCIA + MOMENTUM ──
    # FIX: Se preço vem em tendência forte contra a direção do trade,
    # zona/LT é mais provável de romper. Penalidade FORTE para bloquear.
    trend_penalty = 0.0
    if df_m1 is not None and len(df_m1) >= 8:
        closes_recent = df_m1["close"].astype(float).values[-8:]
        highs_recent = df_m1["high"].astype(float).values[-8:]
        lows_recent = df_m1["low"].astype(float).values[-8:]
        # Contar quantas das últimas 7 velas fecharam em queda vs alta
        n_bear = sum(1 for i in range(1, len(closes_recent)) if closes_recent[i] < closes_recent[i-1])
        n_bull = len(closes_recent) - 1 - n_bear

        # ── Força do momentum: tamanho médio do corpo das velas na direção ──
        # Velas com corpo grande = momentum real, não só contagem
        body_sizes = []
        for i in range(1, len(closes_recent)):
            body = abs(closes_recent[i] - closes_recent[i-1])
            body_sizes.append(body)
        avg_body = sum(body_sizes) / max(len(body_sizes), 1)
        # ATR ratio: corpo médio vs ATR — se > 0.3 = velas grandes
        body_atr_ratio = avg_body / max(atr_val, 1e-9)

        # ── Contra-tendência básica (aplica com ou sem sr_zone) ──
        if direction == "CALL" and n_bear >= 5:
            # Tendência de baixa forte → suporte/LTA pode romper
            trend_penalty = 0.8 + (n_bear - 5) * 0.4  # 0.8 a 1.6
            if body_atr_ratio > 0.3:
                trend_penalty += 0.5  # Velas grandes = momentum real
            confluences.append(f"contra_tendencia({n_bear}/7 bear)")
        elif direction == "PUT" and n_bull >= 5:
            # Tendência de alta forte → resistência/LTB pode romper
            trend_penalty = 0.8 + (n_bull - 5) * 0.4  # 0.8 a 1.6
            if body_atr_ratio > 0.3:
                trend_penalty += 0.5
            confluences.append(f"contra_tendencia({n_bull}/7 bull)")

        # ── MOMENTUM EXTREMO: 7/7 velas = BLOQUEAR (penalidade máxima) ──
        if direction == "CALL" and n_bear >= 7:
            trend_penalty += 1.5
            confluences.append("momentum_extremo_bear")
        elif direction == "PUT" and n_bull >= 7:
            trend_penalty += 1.5
            confluences.append("momentum_extremo_bull")

        # ── V-SHAPE RECOVERY: queda forte + alta forte = rompimento provável ──
        # Detecta: preço caiu significativamente e depois recuperou tudo com velas fortes
        if len(df_m1) >= 20:
            closes_20 = df_m1["close"].astype(float).values[-20:]
            # Ponto mais alto e mais baixo nos últimos 20 candles
            max_price = max(closes_20)
            min_price = min(closes_20)
            idx_max = 0
            idx_min = 0
            for i, p in enumerate(closes_20):
                if p == max_price:
                    idx_max = i
                if p == min_price:
                    idx_min = i

            range_total = max_price - min_price
            current = closes_20[-1]

            if range_total > atr_val * 2.0:  # Movimento significativo
                if idx_max < idx_min < len(closes_20) - 3:
                    # Padrão V invertido: subiu → caiu → está subindo de novo
                    recovery = (current - min_price) / max(range_total, 1e-9)
                    if direction == "PUT" and recovery >= 0.7:
                        # Preço recuperou 70%+ da queda = momentum de alta forte
                        v_pen = 0.8 + (recovery - 0.7) * 2.0  # 0.8 a 1.4
                        trend_penalty += v_pen
                        confluences.append(f"V_recovery({recovery:.0%})")
                elif idx_min < idx_max < len(closes_20) - 3:
                    # Padrão V: caiu → subiu → está caindo de novo
                    drop = (max_price - current) / max(range_total, 1e-9)
                    if direction == "CALL" and drop >= 0.7:
                        v_pen = 0.8 + (drop - 0.7) * 2.0
                        trend_penalty += v_pen
                        confluences.append(f"V_drop({drop:.0%})")

        # ── EXAUSTÃO: preço chegou na zona APÓS movimento forte na MESMA direção ──
        # PUT em suporte após queda forte = exaustão (zona vai segurar, bounce provável)
        # CALL em resistência após alta forte = exaustão (zona vai segurar, queda provável)
        if sr_zone:
            zone_type = sr_zone.get("type", "")
            if direction == "PUT" and zone_type == "support" and n_bear >= 4:
                # PUT num suporte após 4+ velas de queda = exaustão → penalizar forte
                exhaustion_pen = 0.8 + (n_bear - 4) * 0.3  # 0.8 a 1.7
                trend_penalty += exhaustion_pen
                confluences.append(f"exaustao_suporte({n_bear}/7 bear)")
            elif direction == "CALL" and zone_type == "resistance" and n_bull >= 4:
                # CALL numa resistência após 4+ velas de alta = exaustão
                exhaustion_pen = 0.8 + (n_bull - 4) * 0.3
                trend_penalty += exhaustion_pen
                confluences.append(f"exaustao_resistencia({n_bull}/7 bull)")

    total_weight -= trend_penalty
    total_weight = max(0.0, total_weight)

    # ── SCORE FINAL ──
    score = total_weight / max(max_weight, 1e-9)
    score = min(0.95, max(0.0, score))

    confluence_count = 0
    if sr_weight > 0:
        confluence_count += 1
    if lt_weight > 0:
        confluence_count += 1
    if candle_weight > 0:
        confluence_count += 1
    if pb_weight > 0:
        confluence_count += 1
    if mkt_weight > 0:
        confluence_count += 1
    if rsi_weight > 0:
        confluence_count += 1
    if prev_touch_weight > 0:
        confluence_count += 1

    return {
        "score": score,
        "confluence_count": confluence_count,
        "confluences": confluences,
        "total_weight": total_weight,
        "max_weight": max_weight,
        "sr_weight": sr_weight,
        "lt_weight": lt_weight,
        "lt_touch": lt_touch,
        "candle_weight": candle_weight,
        "pb_weight": pb_weight,
        "mkt_weight": mkt_weight,
        "rsi_weight": rsi_weight,
        "prev_touch_weight": prev_touch_weight,
    }


# ═══════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL: dom_forex_signal
# ═══════════════════════════════════════════════════════════════
def dom_forex_signal(df_m1: pd.DataFrame, atr_val: float,
                     min_confluence: int = 2, min_score: float = 0.28) -> Dict[str, Any]:
    """
    Gera sinal baseado em DOM Forex Perfect Zones.

    Price action + RSI confluência:
    1. Detecta Perfect Zones (S/R com peso/força)
    2. Detecta LTA e LTB
    3. Analisa candle de rejeição na zona
    4. Verifica pullback pós-rompimento
    5. RSI como confluência (oversold/overbought na zona)
    6. Confirmação de 2 velas (toque anterior + candle atual)
    7. Calcula confluência para CALL e PUT
    8. Retorna melhor sinal se confluência >= 2

    3 Setups:
    1. REJEIÇÃO → preço toca zona + candle rejeição + LTA/LTB + RSI
    2. ROMPIMENTO → candle forte rompendo zona
    3. PULLBACK → zona rompida + preço volta + confirma
    """
    if len(df_m1) < MIN_CANDLES:
        return {
            "trade": False, "dir": "NEUTRAL", "score": 0.0,
            "reasons": [f"min_velas({len(df_m1)}<{MIN_CANDLES})"]
        }

    # ── 1. DETECTAR PERFECT ZONES ──
    sr_zones = detect_sr_zones(df_m1, atr_val)
    current_price = float(df_m1["close"].iloc[-1])

    # Zonas mantêm classificação histórica (n_highs vs n_lows)
    # NÃO reclassificar por posição do preço — isso causava PUT em suporte
    # quando preço caía ligeiramente abaixo do preço médio da zona.
    # O find_nearest_sr já filtra por tipo quando preço está perto.

    nearest_support = find_nearest_sr(sr_zones, current_price, atr_val, "support")
    nearest_resistance = find_nearest_sr(sr_zones, current_price, atr_val, "resistance")

    # ── 2. DETECTAR LTA / LTB ──
    lta = detect_lta(df_m1, atr_val)
    ltb = detect_ltb(df_m1, atr_val)

    # ── 3. PADRÃO DE CANDLE ──
    candle = analyze_candle_pattern(df_m1, atr_val)

    # ── 4. PULLBACK PÓS-ROMPIMENTO ──
    pullback_zone = find_recently_broken_zone(sr_zones, df_m1, atr_val)

    # ── 5. QUALIDADE DO MERCADO ──
    market = analyze_market_quality(df_m1, atr_val)

    # ── 5b. RSI ──
    rsi_val = calculate_rsi(df_m1)

    # ── 6. CALCULAR CONFLUÊNCIA ──
    conf_call = calculate_confluence(
        sr_zone=nearest_support, lta=lta, ltb=ltb,
        candle=candle, market=market,
        direction="CALL", atr_val=atr_val,
        pullback_zone=pullback_zone if (pullback_zone and pullback_zone.get("signal_dir") == "CALL") else None,
        df_m1=df_m1,
        rsi_val=rsi_val,
    )

    conf_put = calculate_confluence(
        sr_zone=nearest_resistance, lta=lta, ltb=ltb,
        candle=candle, market=market,
        direction="PUT", atr_val=atr_val,
        pullback_zone=pullback_zone if (pullback_zone and pullback_zone.get("signal_dir") == "PUT") else None,
        df_m1=df_m1,
        rsi_val=rsi_val,
    )

    # ── 7. DECIDIR MELHOR SINAL ──
    best_dir = "NEUTRAL"
    best_conf = None
    best_sr = None

    # DOM Forex: S/R ou LT forte é OBRIGATÓRIA
    # CALL precisa de suporte OU LTA forte próxima
    # PUT precisa de resistência OU LTB forte próxima
    call_has_sr = nearest_support is not None and conf_call["sr_weight"] > 0
    put_has_sr = nearest_resistance is not None and conf_put["sr_weight"] > 0

    # LTB/LTA forte e próxima pode SUBSTITUIR zona S/R horizontal
    # Requer: trendline detectada + proximidade + toque no candle (lt_weight alto)
    call_has_lt = (lta is not None
                   and lta.get("proximity", 999) <= LT_PROXIMITY_ATR
                   and conf_call["lt_weight"] >= 1.5)
    put_has_lt = (ltb is not None
                  and ltb.get("proximity", 999) <= LT_PROXIMITY_ATR
                  and conf_put["lt_weight"] >= 1.5)

    call_valid = ((call_has_sr or call_has_lt)
                  and conf_call["confluence_count"] >= min_confluence
                  and conf_call["score"] >= min_score)
    put_valid = ((put_has_sr or put_has_lt)
                 and conf_put["confluence_count"] >= min_confluence
                 and conf_put["score"] >= min_score)

    if call_valid and put_valid:
        if conf_call["score"] >= conf_put["score"]:
            best_dir = "CALL"
            best_conf = conf_call
            best_sr = nearest_support
        else:
            best_dir = "PUT"
            best_conf = conf_put
            best_sr = nearest_resistance
    elif call_valid:
        best_dir = "CALL"
        best_conf = conf_call
        best_sr = nearest_support
    elif put_valid:
        best_dir = "PUT"
        best_conf = conf_put
        best_sr = nearest_resistance

    # ── SEM SINAL ──
    if best_dir == "NEUTRAL":
        any_dir = "CALL" if conf_call["score"] >= conf_put["score"] else "PUT"
        any_conf = conf_call if any_dir == "CALL" else conf_put
        any_sr = nearest_support if any_dir == "CALL" else nearest_resistance
        return {
            "trade": False,
            "dir": any_dir,
            "score": any_conf["score"],
            "reasons": [
                f"conf_insuf({any_conf['confluence_count']}<{min_confluence})",
                *any_conf.get("confluences", [])
            ],
            "market_quality": market["quality"],
            "context": market["context"],
            "has_lt": bool(lta or ltb),
            "lt_confluence": max(
                (lta["strength"] if lta else 0),
                (ltb["strength"] if ltb else 0)
            ) / 10.0,
            "sr_proximity": any_sr["distance_atr"] if any_sr else 0.0,
            "sr_touches": any_sr["touches"] if any_sr else 0,
            "sr_weight": any_sr["weight"] if any_sr else 0.0,
            "sr_reason": any_sr["type"] if any_sr else "sem_sr",
            "sr_bonus": 0.0,
            "entry_confidence": candle.get("strength", 0.0),
            "candle_strength": candle.get("strength", 0.0),
            "effA": market.get("directional", 0.0),
            "rsi": rsi_val,
            "confluence_bonus": any_conf.get("total_weight", 0.0),
            "confluence_count": any_conf.get("confluence_count", 0),
        }

    # ── SINAL CONFIRMADO ──
    lt_active = lta if best_dir == "CALL" else ltb
    # Dom Forex: LT existe na direção = has_lt True (proximidade já pesa na confluência)
    has_lt = lt_active is not None

    # Tipo de setup
    setup_type = "rejeicao"
    if pullback_zone and pullback_zone.get("signal_dir") == best_dir:
        setup_type = "pullback"
    elif candle.get("pattern") in ("strong_bull", "strong_bear") and candle.get("strength", 0) >= 0.6:
        setup_type = "rompimento"

    # FIX: Entrada "rejeicao" EXIGE candle confirmando a direção
    # Sem rejeição no candle = preço no meio da zona sem reação = não entrar
    # RSI extremo pode substituir: RSI oversold em suporte ou overbought em resistência
    # Toque+rejeição na LTB/LTA também confirma (wick tocou trendline + close do lado certo)
    if setup_type == "rejeicao":
        candle_dir = candle.get("direction", "NEUTRAL")
        candle_str = candle.get("strength", 0.0)
        # Candle deve confirmar: direção certa OU pelo menos neutro com LT forte
        candle_ok = (candle_dir == best_dir and candle_str >= 0.3)
        lt_forte = (lt_active is not None and lt_active.get("points", 0) >= 3)
        # LTB/LTA com toque + rejeição no candle (lt_weight >= 1.9 = base + toque + rejeição)
        lt_touch_reject = (lt_active is not None and best_conf.get("lt_weight", 0) >= 1.9)
        # RSI extremo confirma: oversold em suporte (CALL) ou overbought em resistência (PUT)
        rsi_confirms = (
            (best_dir == "CALL" and rsi_val <= RSI_OVERSOLD) or
            (best_dir == "PUT" and rsi_val >= RSI_OVERBOUGHT)
        )
        if not candle_ok and not lt_forte and not rsi_confirms and not lt_touch_reject:
            # Sem confirmação de candle nem LT forte → bloquear
            best_conf["confluences"].append("sem_candle_rejeicao")
            return {
                "trade": False,
                "dir": best_dir,
                "score": best_conf["score"] * 0.7,  # Penalizar score
                "reasons": [
                    f"rejeicao_sem_candle({candle.get('pattern','?')})",
                    *best_conf.get("confluences", [])
                ],
                "market_quality": market["quality"],
                "context": market["context"],
                "has_lt": has_lt,
                "lt_confluence": lt_active["strength"] / 10.0 if lt_active else 0.0,
                "sr_proximity": best_sr["distance_atr"] if best_sr else 0.0,
                "sr_touches": best_sr["touches"] if best_sr else 0,
                "sr_weight": best_sr["weight"] if best_sr else 0.0,
                "sr_reason": best_sr["type"] if best_sr else "sem_sr",
                "sr_bonus": best_conf["sr_weight"],
                "entry_confidence": candle.get("strength", 0.0),
                "candle_strength": candle.get("strength", 0.0),
                "effA": market.get("directional", 0.0),
                "rsi": rsi_val,
                "confluence_bonus": best_conf.get("total_weight", 0.0),
                "confluence_count": best_conf.get("confluence_count", 0),
            }

    return {
        "trade": True,
        "dir": best_dir,
        "score": best_conf["score"],
        "setup_type": setup_type,
        # Perfect Zone
        "sr_proximity": best_sr["distance_atr"] if best_sr else 0.0,
        "sr_touches": best_sr["touches"] if best_sr else 0,
        "sr_weight": best_sr["weight"] if best_sr else 0.0,
        "sr_strength": best_sr["strength"] if best_sr else 0.0,
        "sr_reason": best_sr["type"] if best_sr else "sem_sr",
        "sr_bonus": best_conf["sr_weight"],
        "sr_rejections": best_sr["rejections"] if best_sr else 0,
        "sr_false_breaks": best_sr["false_breaks"] if best_sr else 0,
        "inside_zone": best_sr.get("inside_zone", False) if best_sr else False,
        # Trendline
        "has_lt": has_lt,
        "lt_confluence": lt_active["strength"] / 10.0 if lt_active else 0.0,
        "lt_proximity": lt_active["proximity"] if lt_active else 0.0,
        "lt_points": lt_active["points"] if lt_active else 0,
        # Candle
        "candle_pattern": candle["pattern"],
        "candle_strength": candle["strength"],
        # Mercado
        "market_quality": market["quality"],
        "context": market["context"],
        # Entrada
        "entry_confidence": candle.get("strength", 0.0),
        # Confluência
        "confluence_count": best_conf["confluence_count"],
        "confluence_bonus": best_conf["total_weight"],
        # Legados (LGBM / Bayesiano)
        "effA": market.get("directional", 0.0),
        "rsi": rsi_val,
        "retr": best_sr["distance_atr"] if best_sr else 0.0,
        "A_atr": best_conf.get("total_weight", 0.0),
        "flips": market.get("chop_frac", 0.5),
        "comp": 0.0,
        "late": 0.0,
        "distBreak": 0.0,
        "pb_len": best_conf["confluence_count"],
        "risk_atr": 0.0,
        "trend_strength": lt_active["strength"] if lt_active else 0.0,
        "trend_reason": f"{'LTA' if best_dir == 'CALL' else 'LTB'}_{lt_active['points']}pts" if lt_active else "sem_LT",
        # Razões
        "reasons": [
            f"PerfectZone_{best_dir}_{setup_type}",
            f"SR={best_sr['type']}({best_sr['touches']}t,W={best_sr['weight']:.1f})" if best_sr else "sem_SR",
            f"LT={'LTA' if best_dir == 'CALL' else 'LTB'}({lt_active['points']}pts)" if lt_active else "sem_LT",
            f"candle={candle['pattern']}({candle['strength']:.2f})",
            f"mkt={market['context']}({market['quality']:.2f})",
            f"RSI={rsi_val:.0f}",
            f"conf={best_conf['confluence_count']}",
        ],
    }
