# -*- coding: utf-8 -*-
"""
WS CANDLE COLOR AI — IA de Padrões de Velas (Cor + Formação)
═══════════════════════════════════════════════════════════════

IA específica que analisa CORES e FORMAÇÕES de velas para calcular
a probabilidade da próxima vela ser na direção desejada.

Como funciona:
  1. Extrai sequência de cores (verde/vermelho) das últimas N velas
  2. Detecta PADRÕES CLÁSSICOS (engulfing, hammer, doji, soldiers, etc.)
  3. Analisa DOMINÂNCIA de cor recente (quem está no controle?)
  4. Calcula PROBABILIDADE baseada em dados históricos reais do ativo
  5. Retorna score 0-100% e se confirma ou não a direção desejada

Usado como confirmação ADICIONAL no sistema de confluências S/R.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple

log = logging.getLogger("WS_AUTO_AI")

# ═══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════
LOOKBACK_PATTERN   = 10   # Últimas 10 velas para padrões
LOOKBACK_DOMINANCE = 20   # Últimas 20 velas para dominância de cor
LOOKBACK_HISTORY   = 200  # Últimas 200 velas para calcular probabilidade histórica
                          # Antes era 120 → poucas amostras para padrões de 3 cores
MIN_PROB_CONFIRM   = 55.0 # Mínimo 55% para considerar confirmado
MIN_CANDLES_NEEDED = 30   # Mínimo de velas para rodar


# ═══════════════════════════════════════════════════════════════
# 1. CLASSIFICAR COR DE CADA VELA
# ═══════════════════════════════════════════════════════════════
def _classify_candles(df: pd.DataFrame) -> np.ndarray:
    """
    Classifica cada vela:
      +1 = verde (bullish: close > open)
       0 = doji  (close ≈ open, corpo < 10% do range)
      -1 = vermelha (bearish: close < open)
    """
    o = df["open"].astype(float).values
    c = df["close"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values

    colors = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        full_range = h[i] - l[i]
        body = abs(c[i] - o[i])

        if full_range > 0 and body / full_range < 0.10:
            colors[i] = 0  # doji
        elif c[i] > o[i]:
            colors[i] = 1  # verde
        elif c[i] < o[i]:
            colors[i] = -1  # vermelha
        else:
            colors[i] = 0  # doji
    return colors


# ═══════════════════════════════════════════════════════════════
# 2. DETECTAR PADRÕES CLÁSSICOS DE VELAS
# ═══════════════════════════════════════════════════════════════
def _detect_patterns(df: pd.DataFrame, atr_val: float) -> List[Dict[str, Any]]:
    """
    Detecta padrões de velas clássicos nos últimos candles.
    Retorna lista de padrões encontrados com direção e força.
    """
    patterns = []
    n = len(df)
    if n < 3 or atr_val <= 0:
        return patterns

    o = df["open"].astype(float).values
    c = df["close"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values

    # --- Último candle (i = -1) ---
    body_1 = c[-1] - o[-1]
    abs_body_1 = abs(body_1)
    full_1 = h[-1] - l[-1] if h[-1] > l[-1] else 0.0001
    upper_wick_1 = h[-1] - max(o[-1], c[-1])
    lower_wick_1 = min(o[-1], c[-1]) - l[-1]

    # --- Penúltimo candle (i = -2) ---
    body_2 = c[-2] - o[-2]
    abs_body_2 = abs(body_2)
    full_2 = h[-2] - l[-2] if h[-2] > l[-2] else 0.0001

    # ── HAMMER / PIN BAR (pavio inferior longo, corpo pequeno no topo) ──
    if lower_wick_1 / full_1 >= 0.60 and abs_body_1 / full_1 <= 0.30:
        strength = min(1.0, lower_wick_1 / atr_val)
        patterns.append({
            "name": "hammer",
            "dir": "CALL",
            "strength": strength,
            "candles": 1,
        })

    # ── SHOOTING STAR (pavio superior longo, corpo pequeno no fundo) ──
    if upper_wick_1 / full_1 >= 0.60 and abs_body_1 / full_1 <= 0.30:
        strength = min(1.0, upper_wick_1 / atr_val)
        patterns.append({
            "name": "shooting_star",
            "dir": "PUT",
            "strength": strength,
            "candles": 1,
        })

    # ── BULLISH ENGULFING (vela verde engolfa a vermelha anterior) ──
    if body_2 < 0 and body_1 > 0:  # vermelho → verde
        if c[-1] > o[-2] and o[-1] <= c[-2]:
            strength = min(1.0, abs_body_1 / (abs_body_2 + 0.0001))
            patterns.append({
                "name": "bullish_engulfing",
                "dir": "CALL",
                "strength": min(1.0, strength),
                "candles": 2,
            })

    # ── BEARISH ENGULFING (vela vermelha engolfa a verde anterior) ──
    if body_2 > 0 and body_1 < 0:  # verde → vermelho
        if c[-1] < o[-2] and o[-1] >= c[-2]:
            strength = min(1.0, abs_body_1 / (abs_body_2 + 0.0001))
            patterns.append({
                "name": "bearish_engulfing",
                "dir": "PUT",
                "strength": min(1.0, strength),
                "candles": 2,
            })

    # ── THREE WHITE SOLDIERS (3 verdes consecutivas com corpos crescentes) ──
    if n >= 3:
        b3 = c[-3] - o[-3]
        b2 = c[-2] - o[-2]
        b1 = c[-1] - o[-1]
        if b3 > 0 and b2 > 0 and b1 > 0:
            if c[-1] > c[-2] > c[-3]:
                avg_body = (b1 + b2 + b3) / 3.0
                strength = min(1.0, avg_body / atr_val * 2)
                patterns.append({
                    "name": "three_white_soldiers",
                    "dir": "CALL",
                    "strength": strength,
                    "candles": 3,
                })

    # ── THREE BLACK CROWS (3 vermelhas consecutivas com corpos crescentes) ──
    if n >= 3:
        b3 = c[-3] - o[-3]
        b2 = c[-2] - o[-2]
        b1 = c[-1] - o[-1]
        if b3 < 0 and b2 < 0 and b1 < 0:
            if c[-1] < c[-2] < c[-3]:
                avg_body = (abs(b1) + abs(b2) + abs(b3)) / 3.0
                strength = min(1.0, avg_body / atr_val * 2)
                patterns.append({
                    "name": "three_black_crows",
                    "dir": "PUT",
                    "strength": strength,
                    "candles": 3,
                })

    # ── DOJI após tendência (indecisão = possível reversão) ──
    if abs_body_1 / full_1 < 0.10 and full_1 >= atr_val * 0.5:
        # Doji com range significativo
        if n >= 4:
            # Checar tendência antes do doji
            recent_move = c[-2] - c[-4]
            if recent_move > atr_val * 0.5:
                patterns.append({
                    "name": "doji_topo",
                    "dir": "PUT",
                    "strength": 0.5,
                    "candles": 1,
                })
            elif recent_move < -atr_val * 0.5:
                patterns.append({
                    "name": "doji_fundo",
                    "dir": "CALL",
                    "strength": 0.5,
                    "candles": 1,
                })

    # ── MORNING STAR (vermelho grande + doji/pequeno + verde grande) ──
    if n >= 3:
        b3_abs = abs(c[-3] - o[-3])
        b2_abs = abs(c[-2] - o[-2])
        b1_abs = abs(c[-1] - o[-1])
        full_2_ms = h[-2] - l[-2] if h[-2] > l[-2] else 0.0001

        if (c[-3] - o[-3]) < 0 and b3_abs > atr_val * 0.3:  # 1ª: vermelha grande
            if b2_abs / full_2_ms < 0.25:  # 2ª: corpo pequeno (estrela)
                if (c[-1] - o[-1]) > 0 and b1_abs > atr_val * 0.3:  # 3ª: verde grande
                    patterns.append({
                        "name": "morning_star",
                        "dir": "CALL",
                        "strength": 0.8,
                        "candles": 3,
                    })

    # ── EVENING STAR (verde grande + doji/pequeno + vermelho grande) ──
    if n >= 3:
        b3_abs = abs(c[-3] - o[-3])
        b2_abs = abs(c[-2] - o[-2])
        b1_abs = abs(c[-1] - o[-1])
        full_2_es = h[-2] - l[-2] if h[-2] > l[-2] else 0.0001

        if (c[-3] - o[-3]) > 0 and b3_abs > atr_val * 0.3:  # 1ª: verde grande
            if b2_abs / full_2_es < 0.25:  # 2ª: corpo pequeno
                if (c[-1] - o[-1]) < 0 and b1_abs > atr_val * 0.3:  # 3ª: vermelha grande
                    patterns.append({
                        "name": "evening_star",
                        "dir": "PUT",
                        "strength": 0.8,
                        "candles": 3,
                    })

    return patterns


# ═══════════════════════════════════════════════════════════════
# 3. DOMINÂNCIA DE COR (quem controla o mercado?)
# ═══════════════════════════════════════════════════════════════
def _color_dominance(colors: np.ndarray, lookback: int = LOOKBACK_DOMINANCE) -> Dict[str, Any]:
    """
    Analisa dominância de cor nas últimas N velas.
    Retorna quem está no controle e com que força.
    """
    recent = colors[-lookback:] if len(colors) >= lookback else colors

    n_green = int(np.sum(recent == 1))
    n_red = int(np.sum(recent == -1))
    n_doji = int(np.sum(recent == 0))
    total = len(recent)

    green_pct = n_green / total * 100 if total > 0 else 50
    red_pct = n_red / total * 100 if total > 0 else 50

    # Dominância: quem tem mais velas?
    if green_pct >= 65:
        dominant = "CALL"
        dominance_strength = green_pct / 100.0
    elif red_pct >= 65:
        dominant = "PUT"
        dominance_strength = red_pct / 100.0
    else:
        dominant = "NEUTRAL"
        dominance_strength = max(green_pct, red_pct) / 100.0

    # Sequência atual: quantas velas consecutivas da mesma cor?
    streak = 0
    streak_dir = 0
    for i in range(len(recent) - 1, -1, -1):
        if recent[i] == 0:
            break
        if streak == 0:
            streak_dir = recent[i]
            streak = 1
        elif recent[i] == streak_dir:
            streak += 1
        else:
            break

    return {
        "dominant": dominant,
        "dominance_strength": round(dominance_strength, 2),
        "green_pct": round(green_pct, 1),
        "red_pct": round(red_pct, 1),
        "doji_count": n_doji,
        "streak": streak,
        "streak_dir": "CALL" if streak_dir == 1 else ("PUT" if streak_dir == -1 else "NEUTRAL"),
    }


# ═══════════════════════════════════════════════════════════════
# 4. CORPO DAS VELAS (compradores/vendedores estão fortes?)
# ═══════════════════════════════════════════════════════════════
def _body_strength_analysis(df: pd.DataFrame, atr_val: float,
                            lookback: int = 8) -> Dict[str, Any]:
    """
    Analisa a FORÇA dos corpos das últimas velas.
    Corpos grandes = convicção. Corpos pequenos = indecisão.
    """
    tail = df.tail(lookback)
    o = tail["open"].astype(float).values
    c = tail["close"].astype(float).values

    bodies = c - o  # positivo = verde, negativo = vermelho
    abs_bodies = np.abs(bodies)

    avg_body = float(np.mean(abs_bodies))
    body_atr_ratio = avg_body / atr_val if atr_val > 0 else 0

    # Força dos compradores vs vendedores
    bull_power = float(np.sum(np.maximum(bodies, 0)))  # soma dos corpos verdes
    bear_power = float(np.sum(np.maximum(-bodies, 0)))  # soma dos corpos vermelhos

    total_power = bull_power + bear_power
    if total_power > 0:
        bull_ratio = bull_power / total_power
        bear_ratio = bear_power / total_power
    else:
        bull_ratio = 0.5
        bear_ratio = 0.5

    # Último corpo comparado com média
    last_body_atr = abs_bodies[-1] / atr_val if atr_val > 0 else 0

    return {
        "avg_body_atr": round(body_atr_ratio, 3),
        "bull_ratio": round(bull_ratio, 2),
        "bear_ratio": round(bear_ratio, 2),
        "last_body_atr": round(last_body_atr, 3),
        "conviction": "strong" if body_atr_ratio >= 0.35 else "weak",
    }


# ═══════════════════════════════════════════════════════════════
# 5. PROBABILIDADE HISTÓRICA POR PADRÃO DE COR
# ═══════════════════════════════════════════════════════════════
def _historical_color_probability(df: pd.DataFrame, direction: str,
                                  lookback: int = LOOKBACK_HISTORY) -> Dict[str, Any]:
    """
    Calcula probabilidade REAL baseada em dados históricos:
    "Dado o padrão atual de cores, qual a probabilidade histórica
     de a próxima vela ser na direção desejada?"

    Método: Sliding window — encontra padrões similares no passado
    e conta quantas vezes a vela seguinte foi na direção esperada.
    """
    n = len(df)
    window = min(lookback, n)
    if window < 30:
        return {"prob": 50.0, "samples": 0, "method": "insufficient_data"}

    colors = _classify_candles(df.tail(window))

    # Padrão atual: últimas 3 cores
    if len(colors) < 5:
        return {"prob": 50.0, "samples": 0, "method": "insufficient_data"}

    pattern_len = 3
    current_pattern = tuple(colors[-pattern_len:])

    # Buscar padrão no histórico
    matches = 0
    hits = 0  # vela seguinte na direção desejada

    for i in range(pattern_len, len(colors) - 1):
        past_pattern = tuple(colors[i - pattern_len:i])
        if past_pattern == current_pattern:
            matches += 1
            next_color = colors[i]
            if direction == "CALL" and next_color == 1:
                hits += 1
            elif direction == "PUT" and next_color == -1:
                hits += 1

    if matches >= 3:
        prob = hits / matches * 100
        method = f"exact_match({matches})"
    else:
        # Fallback: padrão relaxado (últimas 2 cores)
        current_2 = tuple(colors[-2:])
        matches_2 = 0
        hits_2 = 0
        for i in range(2, len(colors) - 1):
            past_2 = tuple(colors[i - 2:i])
            if past_2 == current_2:
                matches_2 += 1
                next_color = colors[i]
                if direction == "CALL" and next_color == 1:
                    hits_2 += 1
                elif direction == "PUT" and next_color == -1:
                    hits_2 += 1

        if matches_2 >= 5:
            prob = hits_2 / matches_2 * 100
            method = f"relaxed_match({matches_2})"
            matches = matches_2
        else:
            # Último fallback: freq geral de cor
            if direction == "CALL":
                prob = float(np.sum(colors == 1)) / max(1, np.sum(colors != 0)) * 100
            else:
                prob = float(np.sum(colors == -1)) / max(1, np.sum(colors != 0)) * 100
            method = f"freq_geral"
            matches = len(colors)

    return {
        "prob": round(prob, 1),
        "samples": matches,
        "method": method,
    }


# ═══════════════════════════════════════════════════════════════
# 6. FUNÇÃO PRINCIPAL — predict_candle_color()
# ═══════════════════════════════════════════════════════════════
def predict_candle_color(df: pd.DataFrame, direction: str,
                         atr_val: float) -> Dict[str, Any]:
    """
    IA de Cores de Velas — Analisa padrões e retorna probabilidade.

    Args:
        df: DataFrame M1 com OHLC
        direction: "CALL" ou "PUT" (direção desejada pela S/R)
        atr_val: ATR atual

    Returns:
        {
            "confirmed": bool,       # Padrão de velas confirma a direção?
            "probability": float,     # 0-100% de probabilidade
            "pattern_name": str,      # Nome do padrão encontrado
            "pattern_strength": float,# Força do padrão 0-1
            "dominance": str,         # Quem domina: CALL/PUT/NEUTRAL
            "dominance_strength": float,
            "streak": int,            # Quantas velas seguidas da mesma cor
            "streak_dir": str,        # Direção da sequência
            "bull_ratio": float,      # Força compradores 0-1
            "bear_ratio": float,      # Força vendedores 0-1
            "hist_prob": float,       # Probabilidade histórica do padrão
            "body_conviction": str,   # "strong" ou "weak"
            "reason": str,            # Explicação detalhada
        }
    """
    result = {
        "confirmed": False,
        "probability": 50.0,
        "pattern_name": "none",
        "pattern_strength": 0.0,
        "dominance": "NEUTRAL",
        "dominance_strength": 0.5,
        "streak": 0,
        "streak_dir": "NEUTRAL",
        "bull_ratio": 0.5,
        "bear_ratio": 0.5,
        "hist_prob": 50.0,
        "body_conviction": "weak",
        "reason": "sem_dados",
    }

    if df is None or len(df) < MIN_CANDLES_NEEDED:
        return result

    try:
        # ── 1. Classificar cores ──
        colors = _classify_candles(df)

        # ── 2. Detectar padrões clássicos ──
        tail_df = df.tail(LOOKBACK_PATTERN)
        patterns = _detect_patterns(tail_df, atr_val)

        # Filtrar padrões na direção desejada
        aligned_patterns = [p for p in patterns if p["dir"] == direction]
        contrary_patterns = [p for p in patterns if p["dir"] != direction and p["dir"] != "NEUTRAL"]

        # Melhor padrão alinhado
        best_pattern = None
        if aligned_patterns:
            best_pattern = max(aligned_patterns, key=lambda p: p["strength"])

        # ── 3. Dominância de cor ──
        dominance = _color_dominance(colors)

        # ── 4. Força dos corpos ──
        body_ctx = _body_strength_analysis(df, atr_val)

        # ── 5. Probabilidade histórica ──
        hist = _historical_color_probability(df, direction)

        # ── 6. CALCULAR SCORE FINAL ──
        # Combina: padrão (40%) + dominância (20%) + histórico (30%) + corpo (10%)
        score = 50.0  # Base neutra

        # A) Padrão de vela encontrado (+/- até 20 pontos)
        if best_pattern:
            pattern_bonus = best_pattern["strength"] * 20.0
            score += pattern_bonus
            result["pattern_name"] = best_pattern["name"]
            result["pattern_strength"] = best_pattern["strength"]
        if contrary_patterns:
            worst = max(contrary_patterns, key=lambda p: p["strength"])
            score -= worst["strength"] * 15.0

        # B) Dominância de cor (+/- até 15 pontos)
        if dominance["dominant"] == direction:
            score += dominance["dominance_strength"] * 10.0
        elif dominance["dominant"] != "NEUTRAL":
            # Dominância CONTRA a direção do trade = penalidade forte
            # Quanto mais extrema a dominação, maior o risco
            dom_penalty = dominance["dominance_strength"] * 12.0
            if dominance["dominance_strength"] >= 0.75:
                dom_penalty *= 1.5  # 75%+ dominância = muito perigoso
            score -= dom_penalty

        # C) Sequência recente (streak) — LEITURA GRÁFICA REAL
        # CASCATA: 4+ velas consecutivas na MESMA direção = momentum FORTE
        # NÃO é "reversão provável" — é continuação provável!
        # Streak de 3 pode ser reversão. Streak de 5+ = cascata.
        if dominance["streak"] >= 5 and dominance["streak_dir"] != direction:
            # 5+ velas CONTRA o trade = CASCATA → bloqueia forte
            # O preço está em queda livre / subida livre
            score -= min(15.0, dominance["streak"] * 3.0)
        elif dominance["streak"] == 4 and dominance["streak_dir"] != direction:
            # 4 velas contra = move ainda forte → penalidade
            score -= 6.0
        elif dominance["streak"] == 3 and dominance["streak_dir"] != direction:
            # APENAS 3 velas contra = PODE ser reversão (em contexto S/R)
            # Bônus pequeno, mas só se dominância geral não é extrema
            if dominance["dominance_strength"] < 0.65:
                score += 4.0  # Possível reversão em range
            # Se dominância é forte, NÃO dar bônus (move vai continuar)
        elif dominance["streak"] >= 4 and dominance["streak_dir"] == direction:
            # 4+ velas a favor = pode estar exaurido (overextended)
            score -= 5.0

        # D) Probabilidade histórica (+/- até 15 pontos)
        hist_deviation = hist["prob"] - 50.0
        score += hist_deviation * 0.30  # 30% peso

        # E) Força do corpo (+/- até 5 pontos)
        if direction == "CALL":
            body_edge = (body_ctx["bull_ratio"] - 0.5) * 10.0
        else:
            body_edge = (body_ctx["bear_ratio"] - 0.5) * 10.0
        score += body_edge

        # Clamp
        score = max(15.0, min(85.0, score))

        # ── 7. DECISÃO ──
        confirmed = score >= MIN_PROB_CONFIRM

        # Montar motivo
        parts = []
        if best_pattern:
            parts.append(f"{best_pattern['name']}({best_pattern['strength']:.0%})")
        if dominance["dominant"] != "NEUTRAL":
            parts.append(f"dom={dominance['dominant']}({dominance['dominance_strength']:.0%})")
        if dominance["streak"] >= 2:
            parts.append(f"seq={dominance['streak']}{dominance['streak_dir'][0]}")
        parts.append(f"hist={hist['prob']:.0f}%({hist['method']})")
        if body_ctx["conviction"] == "strong":
            parts.append("corpo_forte")

        reason = "+".join(parts) if parts else "neutro"

        # ── Preencher resultado ──
        result.update({
            "confirmed": confirmed,
            "probability": round(score, 1),
            "dominance": dominance["dominant"],
            "dominance_strength": dominance["dominance_strength"],
            "streak": dominance["streak"],
            "streak_dir": dominance["streak_dir"],
            "bull_ratio": body_ctx["bull_ratio"],
            "bear_ratio": body_ctx["bear_ratio"],
            "hist_prob": hist["prob"],
            "body_conviction": body_ctx["conviction"],
            "reason": reason,
        })

    except Exception as e:
        log.warning(f"[CandleColorAI] Erro: {e}")
        result["reason"] = f"erro: {e}"

    return result
