# -*- coding: utf-8 -*-
"""
FINAL ENTRY GUARD — Última validação antes de entrar no trade
══════════════════════════════════════════════════════════════

Verifica condições de segurança que a estratégia S/R pode ter perdido:
  1. Candle atual não pode ser spike/anomalia
  2. Spread/distância da zona deve ser razoável
  3. Momentum contra a direção bloqueia
  4. Score mínimo absoluto de segurança
  5. Verifica se último candle já consumiu o bounce (late entry)
"""

import numpy as np
import logging

log = logging.getLogger("WS_AUTO_AI")


def validate_entry(df=None, direction=None, atr_val=0.0,
                   zone_price=None, zone_high=None, zone_low=None,
                   zone_type=None, original_score=0.0, **kwargs):
    """
    Guarda final: retorna valid=True/False com motivos detalhados.
    
    Args:
        df: DataFrame com candles (precisa ter pelo menos 5 candles)
        direction: "CALL" ou "PUT"
        atr_val: ATR atual do ativo
        zone_price: preço central da zona S/R
        zone_high/zone_low: limites da zona
        zone_type: "support" ou "resistance"
        original_score: score calculado pela estratégia
        **kwargs: campos extras do setup (zone_strength, sr_touches, etc.)
    
    Returns:
        dict com: valid, blocks, warnings, adjusted_score
    """
    blocks = []
    warnings = []
    adjusted_score = original_score

    # Se não tem dados suficientes, aprovar com warning
    if df is None or len(df) < 5 or atr_val <= 0:
        return {
            "valid": True,
            "blocks": [],
            "warnings": ["dados_insuficientes_para_guard"],
            "adjusted_score": original_score,
        }

    # ── Dados do último candle ──
    last = df.iloc[-1]
    o = float(last["open"])
    h = float(last["high"])
    l = float(last["low"])
    c = float(last["close"])
    full_range = h - l

    # ═══════════════════════════════════════════════════════════
    # GUARD 1: Spike / anomalia no último candle
    # ═══════════════════════════════════════════════════════════
    if full_range > atr_val * 2.5:
        blocks.append("spike_candle_anomalo")
        log.warning(f"[GUARD] Candle com range={full_range:.6f} > 2.5×ATR={atr_val*2.5:.6f}")

    # ═══════════════════════════════════════════════════════════
    # GUARD 2: Distância excessiva da zona
    # ═══════════════════════════════════════════════════════════
    if zone_price is not None and atr_val > 0:
        dist = abs(c - zone_price) / atr_val
        if dist > 1.0:
            blocks.append(f"longe_da_zona({dist:.2f}ATR)")
        elif dist > 0.6:
            warnings.append(f"dist_moderada({dist:.2f}ATR)")
            adjusted_score -= 0.05

    # ═══════════════════════════════════════════════════════════
    # GUARD 3: Momentum forte CONTRA a direção (3 candles)
    # ═══════════════════════════════════════════════════════════
    if len(df) >= 4 and direction:
        last3 = df.iloc[-3:]
        closes = last3["close"].to_numpy(float)
        opens = last3["open"].to_numpy(float)

        if direction == "CALL":
            # 3 candles bearish seguidos = momentum forte de queda
            all_bearish = all(closes[i] < opens[i] for i in range(len(closes)))
            body_sum = sum(abs(opens[i] - closes[i]) for i in range(len(closes)))
            if all_bearish and body_sum > atr_val * 1.5:
                blocks.append("momentum_forte_contra_CALL")
        else:
            # 3 candles bullish seguidos = momentum forte de alta
            all_bullish = all(closes[i] > opens[i] for i in range(len(closes)))
            body_sum = sum(abs(opens[i] - closes[i]) for i in range(len(closes)))
            if all_bullish and body_sum > atr_val * 1.5:
                blocks.append("momentum_forte_contra_PUT")

    # ═══════════════════════════════════════════════════════════
    # GUARD 4: Score mínimo absoluto de segurança
    # ═══════════════════════════════════════════════════════════
    if original_score < 0.45:
        blocks.append(f"score_muito_baixo({original_score:.2f})")

    # ═══════════════════════════════════════════════════════════
    # GUARD 5: Late entry — bounce já aconteceu
    # ═══════════════════════════════════════════════════════════
    if zone_price is not None and direction and atr_val > 0 and len(df) >= 3:
        prev = df.iloc[-2]
        prev_c = float(prev["close"])

        if direction == "CALL" and zone_type == "support":
            # Se preço já subiu > 0.8 ATR do suporte, bounce já foi
            if c - zone_price > atr_val * 0.8 and c > prev_c:
                blocks.append("late_entry_bounce_ja_foi")
        elif direction == "PUT" and zone_type == "resistance":
            # Se preço já caiu > 0.8 ATR da resistência, bounce já foi
            if zone_price - c > atr_val * 0.8 and c < prev_c:
                blocks.append("late_entry_bounce_ja_foi")

    # ═══════════════════════════════════════════════════════════
    # GUARD 6: Zone strength muito fraca com poucos toques
    # ═══════════════════════════════════════════════════════════
    zone_str = float(kwargs.get("zone_strength", 0.0))
    sr_touches = int(kwargs.get("sr_touches", 0))
    if zone_str < 0.45 and sr_touches < 3:
        blocks.append(f"zona_fraca(str={zone_str:.2f},tq={sr_touches})")

    # ═══════════════════════════════════════════════════════════
    # RESULTADO FINAL
    # ═══════════════════════════════════════════════════════════
    adjusted_score = max(0.0, min(1.0, adjusted_score))
    valid = len(blocks) == 0

    if blocks:
        log.info(f"[GUARD] BLOQUEADO: {', '.join(blocks)}")
    if warnings:
        log.info(f"[GUARD] Avisos: {', '.join(warnings)}")

    return {
        "valid": valid,
        "blocks": blocks,
        "warnings": warnings,
        "adjusted_score": adjusted_score,
    }
