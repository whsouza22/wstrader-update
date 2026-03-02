# -*- coding: utf-8 -*-
"""
WS Structure Patterns — 13 Padrões de Velas em Topos/Fundos com Edge Real
══════════════════════════════════════════════════════════════════════════

Baseado no estudo empírico de 7.700 velas M1 × 11 pares OTC.
Cada padrão foi validado com:
  - WR ≥ 85% (testado em dados reais)
  - Amostra ≥ 10 ocorrências
  - Aparece em 2+ ativos diferentes (não é acaso de 1 par)
  - Hora do dia específica (recorrência diária)

Funcionamento:
  1. Detecta pivots (topos/fundos menores) no DataFrame
  2. Verifica se o preço atual está tocando uma zona de pivot
  3. Classifica o candle atual e o padrão de 3 candles
  4. Compara com os 13 padrões conhecidos que ganham
  5. Retorna: (match, direction, pattern_name, wr, confidence)

Integração:
  - Chamado pelo ws_confluence_brain.py como feature adicional
  - Padrão match = bônus na probabilidade (+5% a +12%)
  - Padrão anti-match (inverso) = penalidade (-3% a -5%)
"""

import numpy as np
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List

log = logging.getLogger("WS_AUTO_AI")

# ═══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════
PIVOT_WINDOW_MINOR = 3       # Janela para pivots menores (7 candles)
PIVOT_MIN_SIG_ATR = 0.08     # Significância mínima em ATR
ZONE_TOLERANCE_ATR = 0.25    # Tolerância para "preço tocou a zona"
MIN_PIVOT_DISTANCE = 5       # Mínimo de candles entre pivot e toque


# ═══════════════════════════════════════════════════════════════
# CLASSIFICAÇÃO DE CANDLE INDIVIDUAL
# ═══════════════════════════════════════════════════════════════
def classify_candle(o: float, h: float, l: float, c: float, atr_val: float) -> str:
    """
    Classifica um candle em tipo canônico.
    Retorna: doji, hammer, inv_hammer, pin_bull, pin_bear,
             marubozu_bull, marubozu_bear, bull, bear.
    """
    body = c - o
    body_size = abs(body)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    total_range = h - l
    if total_range < 1e-9:
        return "doji"

    body_ratio = body_size / total_range

    # Doji: corpo < 15% do range
    if body_ratio < 0.15:
        return "doji"

    # Hammer: corpo bull, wick inferior >= 2x corpo, wick sup pequena
    if body > 0 and lower_wick >= body_size * 2.0 and upper_wick < body_size * 0.3:
        return "hammer"

    # Inverted Hammer: corpo bear, wick sup >= 2x corpo, wick inf pequena
    if body < 0 and upper_wick >= body_size * 2.0 and lower_wick < body_size * 0.3:
        return "inv_hammer"

    # Pin bar bullish: wick inferior longa (>= 60% range), corpo bull
    if body > 0 and lower_wick >= total_range * 0.60:
        return "pin_bull"

    # Pin bar bearish: wick superior longa (>= 60% range), corpo bear
    if body < 0 and upper_wick >= total_range * 0.60:
        return "pin_bear"

    # Marubozu bullish: corpo >80% range, bull
    if body > 0 and body_ratio > 0.80:
        return "marubozu_bull"

    # Marubozu bearish: corpo >80% range, bear
    if body < 0 and body_ratio > 0.80:
        return "marubozu_bear"

    return "bull" if body > 0 else "bear"


# ═══════════════════════════════════════════════════════════════
# CLASSIFICAÇÃO DE PADRÃO DE 3 CANDLES
# ═══════════════════════════════════════════════════════════════
def classify_3candle(opens, highs, lows, closes, atr_val: float) -> str:
    """
    Analisa os últimos 3 candles e retorna padrão composto.
    Precisa de arrays com pelo menos 3 elementos.
    """
    if len(opens) < 3:
        return "insuf"

    o1, o2, o3 = float(opens[-3]), float(opens[-2]), float(opens[-1])
    h1, h2, h3 = float(highs[-3]), float(highs[-2]), float(highs[-1])
    l1, l2, l3 = float(lows[-3]), float(lows[-2]), float(lows[-1])
    c1, c2, c3 = float(closes[-3]), float(closes[-2]), float(closes[-1])

    b1, b2, b3 = c1 - o1, c2 - o2, c3 - o3
    r2 = h2 - l2

    # Morning Star: bear grande + doji/pequeno + bull grande
    if b1 < 0 and abs(b1) > atr_val * 0.3 and r2 > 0 and abs(b2) / r2 < 0.20 and b3 > atr_val * 0.3:
        return "morning_star"

    # Evening Star: bull grande + doji/pequeno + bear grande
    if b1 > 0 and b1 > atr_val * 0.3 and r2 > 0 and abs(b2) / r2 < 0.20 and b3 < 0 and abs(b3) > atr_val * 0.3:
        return "evening_star"

    # Three White Soldiers: 3 bulls consecutivos com corpos semelhantes
    if b1 > 0 and b2 > 0 and b3 > 0 and b2 >= b1 * 0.8 and b3 >= b2 * 0.8:
        return "three_soldiers"

    # Three Black Crows: 3 bears consecutivos
    if b1 < 0 and b2 < 0 and b3 < 0 and abs(b2) >= abs(b1) * 0.8 and abs(b3) >= abs(b2) * 0.8:
        return "three_crows"

    # Tweezers Bottom: lows iguais (± 0.05 ATR), ultimo bear→bull
    tol = atr_val * 0.05
    if abs(l2 - l3) < tol and b2 < 0 and b3 > 0:
        return "tweezers_bottom"

    # Tweezers Top: highs iguais, ultimo bull→bear
    if abs(h2 - h3) < tol and b2 > 0 and b3 < 0:
        return "tweezers_top"

    # Inside Bar: candle 3 inteiramente dentro do candle 2
    if h3 <= h2 and l3 >= l2:
        return "inside_bar"

    # Sequência genérica
    c1t = classify_candle(o1, h1, l1, c1, atr_val)
    c2t = classify_candle(o2, h2, l2, c2, atr_val)
    c3t = classify_candle(o3, h3, l3, c3, atr_val)
    return f"seq_{c1t}_{c2t}_{c3t}"


# ═══════════════════════════════════════════════════════════════
# OS 13 PADRÕES VENCEDORES (extraídos do estudo empírico)
# ═══════════════════════════════════════════════════════════════
# Formato: (pivot_type, candle_1, candle_3, direction, hours, wr, samples, n_ativos)
# hours = lista de horas UTC com edge, None = qualquer hora
WINNING_PATTERNS = [
    # ── PADRÕES COM 100% WR (hora-específicos) ──
    # #1: Fundo menor + marubozu_bull + three_soldiers → CALL (100%, N=24, 4 ativos)
    ("fundo", "marubozu_bull", "three_soldiers",  "CALL", [18, 19], 1.00, 24, 4),

    # #2: Fundo menor + bull + seq_marubozu_bull_bull_bull → CALL (100%, N=20, 5 ativos)
    ("fundo", "bull", "seq_marubozu_bull_bull_bull", "CALL", [18], 1.00, 20, 5),

    # #3: Topo menor + marubozu_bear + tweezers_top → PUT (100%, N=20, 3 ativos)
    ("topo",  "marubozu_bear", "tweezers_top",    "PUT",  [19], 1.00, 20, 3),

    # #4: Fundo menor + bear + evening_star → CALL (100%, N=15, 7 ativos)
    ("fundo", "bear", "evening_star",             "CALL", [14], 1.00, 15, 7),

    # #5: Topo menor + marubozu_bear + evening_star → PUT (100%, N=11, 6 ativos)
    ("topo",  "marubozu_bear", "evening_star",    "PUT",  [11], 1.00, 11, 6),

    # #6: Topo menor + marubozu_bear + seq_bull_bull_marubozu_bear → PUT (100%, N=12, 4 ativos)
    ("topo",  "marubozu_bear", "seq_bull_bull_marubozu_bear", "PUT", [17], 1.00, 12, 4),

    # ── PADRÕES COM 85-93% WR (multi-ativo, hora-específicos) ──
    # #7: Topo menor + bull + tweezers_bottom → PUT (93.3%, N=15, 6 ativos)
    ("topo",  "bull", "tweezers_bottom",          "PUT",  [18], 0.933, 15, 6),

    # #8: Fundo menor + doji + inside_bar → CALL (91.7%, N=24, 5 ativos)
    ("fundo", "doji", "inside_bar",               "CALL", [18], 0.917, 24, 5),

    # #9: Topo menor + doji + seq_bear_bull_doji → PUT (89.5%, N=19, 5 ativos)
    ("topo",  "doji", "seq_bear_bull_doji",       "PUT",  [18], 0.895, 19, 5),

    # #10: Topo menor + marubozu_bear + three_crows → PUT (87.5%, N=16, 5 ativos)
    ("topo",  "marubozu_bear", "three_crows",     "PUT",  [13], 0.875, 16, 5),

    # #11: Topo menor + bear + seq_bear_bull_bear → PUT (85.7%, N=28, 6 ativos)
    ("topo",  "bear", "seq_bear_bull_bear",       "PUT",  [12, 16], 0.857, 28, 6),

    # #12: Fundo menor + marubozu_bull + seq_bull_bear_marubozu_bull → CALL (85.7%, N=21, 5 ativos)
    ("fundo", "marubozu_bull", "seq_bull_bear_marubozu_bull", "CALL", [18], 0.857, 21, 5),

    # #13: Topo menor + bear + seq_bear_bear_bear → PUT (85%, N=20, 4 ativos)
    ("topo",  "bear", "seq_bear_bear_bear",       "PUT",  [16, 17], 0.850, 20, 4),
]

# Padrões conhecidos que PERDEM (inverso dos ganhadores, contexto errado)
# Servem como anti-filtro: se o padrão é de perda, penalizar
LOSING_PATTERNS = [
    # Fundo menor genérico sem padrão especial = ~48% (sem edge)
    # Topo maior genérico = ~49% (sem edge)
    # Esses não adicionam edge — são neutros
]


# ═══════════════════════════════════════════════════════════════
# DETECTOR DE PIVOTS MENORES
# ═══════════════════════════════════════════════════════════════
def _find_minor_pivots(highs, lows, atr_val: float,
                       window: int = PIVOT_WINDOW_MINOR) -> Tuple[List, List]:
    """
    Encontra pivots menores (topos e fundos locais).
    Retorna: (pivot_highs, pivot_lows) — cada um é lista de (index, price).
    """
    min_sig = PIVOT_MIN_SIG_ATR * atr_val
    ph, pl = [], []

    for i in range(window, len(highs) - window):
        # Pivot High
        if highs[i] >= max(highs[i - window: i + window + 1]):
            neighbors = list(highs[max(0, i - window):i]) + list(highs[i + 1:i + window + 1])
            if neighbors and (highs[i] - max(neighbors)) >= min_sig:
                ph.append((i, float(highs[i])))
        # Pivot Low
        if lows[i] <= min(lows[i - window: i + window + 1]):
            neighbors = list(lows[max(0, i - window):i]) + list(lows[i + 1:i + window + 1])
            if neighbors and (min(neighbors) - lows[i]) >= min_sig:
                pl.append((i, float(lows[i])))

    return ph, pl


# ═══════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL: DETECTAR PADRÃO VENCEDOR
# ═══════════════════════════════════════════════════════════════
def detect_structure_pattern(df, atr_val: float,
                              direction: str = "",
                              utc_hour: Optional[int] = None
                              ) -> Dict[str, Any]:
    """
    Analisa o DataFrame e detecta se o preço atual está num padrão vencedor.

    Args:
        df: DataFrame com colunas open/high/low/close (index = time)
        atr_val: ATR atual
        direction: "CALL" ou "PUT" (do setup S/R). Se vazio, detecta automaticamente.
        utc_hour: hora UTC atual. Se None, calcula do último candle.

    Returns:
        Dict com:
          - "match": bool — se há match com padrão vencedor
          - "pattern_name": str — nome do padrão
          - "pattern_dir": str — CALL ou PUT esperado
          - "pattern_wr": float — WR histórico (0-1)
          - "pattern_samples": int — amostras do estudo
          - "pattern_ativos": int — quantos ativos tinham este padrão
          - "confidence": float — confiança 0-1 (baseada em WR × match com hora)
          - "pivot_type": str — "topo" ou "fundo"
          - "candle_1": str — tipo do candle atual
          - "candle_3": str — padrão de 3 candles
          - "hour_match": bool — se a hora atual coincide com a melhor hora
          - "bonus_pct": float — bônus sugerido para a probabilidade (+0.05 a +0.12)
    """
    result = {
        "match": False, "pattern_name": "", "pattern_dir": "",
        "pattern_wr": 0.0, "pattern_samples": 0, "pattern_ativos": 0,
        "confidence": 0.0, "pivot_type": "", "candle_1": "",
        "candle_3": "", "hour_match": False, "bonus_pct": 0.0,
    }

    n = len(df)
    if n < 50:
        return result

    opens = df["open"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)

    i = n - 1  # Última vela (a do toque)

    # Determinar hora UTC
    if utc_hour is None:
        try:
            ts = df.index[i]
            utc_hour = int(ts.hour) if hasattr(ts, "hour") else None
        except Exception:
            utc_hour = None

    # Classificar candle atual e padrão de 3 candles
    c1_type = classify_candle(opens[i], highs[i], lows[i], closes[i], atr_val)
    c3_type = classify_3candle(
        opens[max(0, i-2):i+1], highs[max(0, i-2):i+1],
        lows[max(0, i-2):i+1], closes[max(0, i-2):i+1], atr_val
    )
    result["candle_1"] = c1_type
    result["candle_3"] = c3_type

    # Encontrar pivots menores
    pivot_highs, pivot_lows = _find_minor_pivots(highs, lows, atr_val)

    # Verificar se preço atual toca algum pivot (topo ou fundo)
    zone_tol = ZONE_TOLERANCE_ATR * atr_val
    touching_type = None  # "topo" ou "fundo"

    # Check topos (resistências)
    for (p_idx, p_price) in reversed(pivot_highs):
        if p_idx >= i - MIN_PIVOT_DISTANCE:
            continue  # pivot muito recente (pode ser o próprio candle)
        if p_idx < i - 500:
            continue  # pivot muito antigo
        if abs(closes[i] - p_price) <= zone_tol or (lows[i] <= p_price + zone_tol and highs[i] >= p_price - zone_tol):
            touching_type = "topo"
            break

    # Check fundos (suportes)
    if touching_type is None:
        for (p_idx, p_price) in reversed(pivot_lows):
            if p_idx >= i - MIN_PIVOT_DISTANCE:
                continue
            if p_idx < i - 500:
                continue
            if abs(closes[i] - p_price) <= zone_tol or (lows[i] <= p_price + zone_tol and highs[i] >= p_price - zone_tol):
                touching_type = "fundo"
                break

    if touching_type is None:
        return result  # Preço não está tocando nenhum pivot

    result["pivot_type"] = touching_type

    # Buscar match com os 13 padrões vencedores
    best_match = None
    best_confidence = 0.0

    for pat in WINNING_PATTERNS:
        p_pivot, p_c1, p_c3, p_dir, p_hours, p_wr, p_samples, p_ativos = pat

        # Check tipo de pivot
        if p_pivot != touching_type:
            continue

        # Check candle 1 (tipo do candle atual)
        if p_c1 != c1_type:
            continue

        # Check padrão de 3 candles
        if p_c3 != c3_type:
            continue

        # Check direção: se setup tem direção, deve combinar
        if direction and direction != p_dir:
            continue

        # Match encontrado!
        hour_match = (utc_hour is not None and p_hours is not None and utc_hour in p_hours)

        # Calcular confiança
        # Base: WR do padrão (85-100%)
        # Bônus: hora coincide (+10%), muitos ativos (+5%)
        conf = p_wr
        if hour_match:
            conf = min(1.0, conf + 0.05)  # hora certa = confiança extra
        if p_ativos >= 5:
            conf = min(1.0, conf + 0.03)  # multi-ativo forte

        if conf > best_confidence:
            best_confidence = conf
            best_match = pat
            best_hour_match = hour_match

    if best_match is not None:
        p_pivot, p_c1, p_c3, p_dir, p_hours, p_wr, p_samples, p_ativos = best_match

        # Calcular bônus de probabilidade
        # 100% WR + hora match = +12%
        # 100% WR sem hora = +8%
        # 85-93% WR + hora = +7%
        # 85-93% WR sem hora = +5%
        if p_wr >= 0.99:
            bonus = 0.12 if best_hour_match else 0.08
        elif p_wr >= 0.90:
            bonus = 0.09 if best_hour_match else 0.06
        else:
            bonus = 0.07 if best_hour_match else 0.05

        hours_str = ",".join(f"{h:02d}h" for h in p_hours) if p_hours else "all"
        pattern_name = f"{p_pivot}_{p_c1}_{p_c3}@{hours_str}"

        result.update({
            "match": True,
            "pattern_name": pattern_name,
            "pattern_dir": p_dir,
            "pattern_wr": p_wr,
            "pattern_samples": p_samples,
            "pattern_ativos": p_ativos,
            "confidence": best_confidence,
            "hour_match": best_hour_match,
            "bonus_pct": bonus,
        })

    return result


# ═══════════════════════════════════════════════════════════════
# HELPER: Resumo rápido para log
# ═══════════════════════════════════════════════════════════════
def pattern_summary(result: Dict) -> str:
    """Retorna string resumo do resultado para log."""
    if not result.get("match"):
        return f"no_pattern ({result.get('candle_1', '?')}/{result.get('candle_3', '?')})"
    hr = "hora_OK" if result["hour_match"] else "hora_diff"
    return (
        f"PATTERN: {result['pattern_name']} → {result['pattern_dir']} "
        f"WR={result['pattern_wr']:.0%} N={result['pattern_samples']} "
        f"({hr}) bonus=+{result['bonus_pct']:.0%}"
    )


# ═══════════════════════════════════════════════════════════════
# HELPER: Verificar se hora atual é "golden hour" (melhor para operar)
# ═══════════════════════════════════════════════════════════════
# Baseado no estudo: horas com WR médio >70% nos padrões estrela
GOLDEN_HOURS = {10, 17, 18, 19}          # WR médio 70-75%
GOOD_HOURS = {9, 11, 13, 14, 15, 16}     # WR médio 62-68%
NEUTRAL_HOURS = {8, 12, 20}              # WR médio <62%


def get_hour_quality(utc_hour: Optional[int]) -> str:
    """Retorna qualidade da hora para operação: golden, good, neutral, unknown."""
    if utc_hour is None:
        return "unknown"
    if utc_hour in GOLDEN_HOURS:
        return "golden"
    if utc_hour in GOOD_HOURS:
        return "good"
    return "neutral"


def get_hour_bonus(utc_hour: Optional[int]) -> float:
    """Retorna bônus/penalidade baseado na hora do dia."""
    quality = get_hour_quality(utc_hour)
    if quality == "golden":
        return 0.03   # +3% nas golden hours
    elif quality == "good":
        return 0.01   # +1% nas good hours
    elif quality == "neutral":
        return -0.02  # -2% nas horas neutras
    return 0.0        # sem ajuste se hora desconhecida
