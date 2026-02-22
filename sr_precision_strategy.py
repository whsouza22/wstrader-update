# -*- coding: utf-8 -*-
"""
SR + CANDLE FEATURES — Estratégia de Suporte/Resistência + Features Matemáticas

🎯 FILOSOFIA: SÓ S/R + FEATURES. A IA aprende o resto.
🧠 A IA (Bayesiano + LGBM) continua aprendendo por ativo no WS_AUTO_AI.py

PIPELINE:
1. SUPORTE/RESISTÊNCIA: Zona com 3+ toques → preço perto = sinal
2. FEATURES CANDLE: body_ratio, wick_ratios, close_pos, ret1/3/5, body_vs_ma20, range_vs_ma20
3. REJEIÇÃO: Padrão matemático de candle confirmando a zona
4. MOMENTUM: Anti-breakout — 3+ velas fortes contra penaliza

ENTRADA:
- Preço no SUPORTE → CALL (compradores defendem)
- Preço na RESISTÊNCIA → PUT (vendedores defendem)
- Candle de rejeição (pavio > corpo) dá bônus
- Momentum contra (3+ velas fortes) penaliza

SCORE (simples):
- Base: 0.60
- +0.04 por toque extra (acima de 3)
- +0.06 se candle tem rejeição (proporcional à qualidade)
- -0.08 se momentum forte contra
- Máximo: 0.85

A IA no WS_AUTO_AI.py recebe o sinal e decide se entra ou não baseado
no aprendizado histórico daquele ativo específico.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple

log = logging.getLogger("WS_AUTO_AI")

# ═══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════
SR_LOOKBACK = 200          # Velas para buscar S/R
SR_MIN_TOUCHES = 3         # Mínimo de toques para zona válida
SR_CLUSTER_ATR = 0.35      # Tolerância para agrupar toques (faixa)
SR_PROXIMITY_ATR = 0.50    # Preço deve estar a no máximo 0.50 ATR da zona
MIN_CANDLES = 80           # Mínimo de velas para análise

# Score — transparente e limpo (AJUSTADO para dar mais margem)
SCORE_BASE = 0.52              # Base: zona S/R com 3 toques (era 0.50 — ficava no limite)
SCORE_PER_TOUCH = 0.05         # +0.05 por toque extra (4t=0.57, 5t=0.62, 6t=0.67)
SCORE_MAX_TOUCH_BONUS = 0.20   # Máximo: até 7+ toques
SCORE_REJECTION_MAX = 0.10     # Candle de rejeição confirmando bounce (era 0.08)
SCORE_TREND_ALIGN = 0.05       # Bônus: trade COM a macro tendência (era 0.04)
SCORE_MACRO_AGAINST = 0.04     # Penalidade: contra macro tendência (era 0.06 — bloqueava muito)

# Trend filter
TREND_LOOKBACK = 20            # Velas para calcular tendência (micro)
TREND_SLOPE_THRESHOLD = 0.25   # slope > 0.25 ATR em 20 velas = tendência forte

# MACRO TREND — tendência dominante (lookback maior)
MACRO_TREND_LOOKBACK = 50      # 50 velas para macro tendência
MACRO_TREND_MQ_PENALTY = 0.25  # Redução no market_quality se contra macro forte

# Choppy / Wick filter (mercado ruim com muitos pavios)
CHOPPY_LOOKBACK = 10         # Últimas N velas para análise de pavios
CHOPPY_WICK_THRESHOLD = 0.60 # Candle é "de pavio" se body_ratio < 0.40 (pavio > 60%)
CHOPPY_MAX_RATIO = 0.70      # Se 70%+ das velas são de pavio → mercado choppy (era 0.60 — OTC tem muitos pavios)


# ═══════════════════════════════════════════════════════════════
# 0a. DETECÇÃO DE TENDÊNCIA (slope + higher-highs/lower-lows)
# ═══════════════════════════════════════════════════════════════
def _detect_trend(df: pd.DataFrame, atr_val: float, lookback: int = None) -> Tuple[str, float]:
    """
    Detecta tendência usando slope dos closes + higher-highs/lower-lows.
    
    Retorna: (trend_dir, trend_strength)
      - trend_dir: "up", "down", "neutral"
      - trend_strength: 0.0 a 1.0 (0 = sem tendência, 1 = tendência forte)
    """
    _lb = lookback or TREND_LOOKBACK
    if len(df) < _lb:
        return "neutral", 0.0
    
    closes = df["close"].astype(float).values[-_lb:]
    highs = df["high"].astype(float).values[-_lb:]
    lows = df["low"].astype(float).values[-_lb:]
    
    # 1. Slope linear dos closes (normalizado por ATR)
    x = np.arange(len(closes))
    slope = np.polyfit(x, closes, 1)[0]  # coeficiente angular
    slope_atr = slope * _lb / atr_val if atr_val > 0 else 0.0
    
    # 2. Higher-highs / Lower-lows (contar segmentos de 5 velas)
    seg = 4  # comparar em blocos
    hh_count = 0
    ll_count = 0
    for i in range(seg, len(closes), seg):
        if highs[i] > highs[i - seg]:
            hh_count += 1
        if lows[i] < lows[i - seg]:
            ll_count += 1
    
    # 3. Preço vs média das primeiras e últimas velas
    first_half = np.mean(closes[:_lb // 2])
    second_half = np.mean(closes[_lb // 2:])
    displacement = (second_half - first_half) / atr_val if atr_val > 0 else 0.0
    
    # Combinar sinais
    up_signals = 0
    down_signals = 0
    
    if slope_atr > TREND_SLOPE_THRESHOLD:
        up_signals += 2
    elif slope_atr < -TREND_SLOPE_THRESHOLD:
        down_signals += 2
    
    if hh_count >= 3:
        up_signals += 1
    if ll_count >= 3:
        down_signals += 1
    
    if displacement > 0.5:
        up_signals += 1
    elif displacement < -0.5:
        down_signals += 1
    
    # Determinar tendência
    strength = abs(slope_atr) / 1.0  # normalizar: 1 ATR de slope = força 1.0
    strength = min(1.0, strength)
    
    if up_signals >= 2 and up_signals > down_signals:
        return "up", strength
    elif down_signals >= 2 and down_signals > up_signals:
        return "down", strength
    
    return "neutral", strength * 0.3


# ═══════════════════════════════════════════════════════════════
# 0b. FILTRO DE MERCADO CHOPPY (muitos pavios = indecisão)
# ═══════════════════════════════════════════════════════════════
def _is_choppy_market(df: pd.DataFrame, atr_val: float) -> Tuple[bool, float]:
    """
    Detecta mercado choppy analisando as últimas CHOPPY_LOOKBACK velas.
    Mercado com muitos pavios = indecisão = ruim para S/R bounce.
    
    Critérios:
    - body_ratio < 0.40 → candle de pavio (corpo < 40% do range)
    - Se 60%+ das velas recentes são de pavio → mercado choppy
    
    Retorna: (is_choppy, wick_ratio) onde wick_ratio = % de velas com pavio grande
    """
    if len(df) < CHOPPY_LOOKBACK:
        return False, 0.0
    
    recent = df.tail(CHOPPY_LOOKBACK)
    wick_candles = 0
    
    for i in range(len(recent)):
        row = recent.iloc[i]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        total_range = h - l
        
        if total_range < atr_val * 0.05:
            wick_candles += 1  # Doji = pior que pavio
            continue
        
        body = abs(c - o)
        body_ratio = body / total_range
        
        if body_ratio < (1.0 - CHOPPY_WICK_THRESHOLD):
            # Corpo < 40% do range → pavio domina
            wick_candles += 1
    
    ratio = wick_candles / CHOPPY_LOOKBACK
    return ratio >= CHOPPY_MAX_RATIO, ratio


# ═══════════════════════════════════════════════════════════════
# 1. DETECÇÃO DE ZONAS S/R (SIMPLES)
# ═══════════════════════════════════════════════════════════════
def _find_sr_zones(df: pd.DataFrame, atr_val: float) -> List[Dict[str, Any]]:
    """
    Encontra zonas de suporte/resistência por clustering de pivôs.
    Simples: encontra pontos de reversão, agrupa os próximos, conta toques.
    """
    if len(df) < 30 or atr_val <= 0:
        return []

    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    n = len(df)
    
    # ── Encontrar pontos de reversão (pivot high/low com left=3, right=2) ──
    reversal_prices = []
    left, right = 3, 2
    
    for i in range(left, n - right):
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
            reversal_prices.append({"price": highs[i], "index": i, "type": "high"})
        
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
            reversal_prices.append({"price": lows[i], "index": i, "type": "low"})
    
    if len(reversal_prices) < 3:
        return []
    
    # ── Clustering: agrupar reversões próximas ──
    tolerance = atr_val * SR_CLUSTER_ATR
    reversal_prices.sort(key=lambda x: x["price"])
    
    zones = []
    used = [False] * len(reversal_prices)
    
    for i in range(len(reversal_prices)):
        if used[i]:
            continue
        
        cluster = [reversal_prices[i]]
        used[i] = True
        
        for j in range(i + 1, len(reversal_prices)):
            if used[j]:
                continue
            if abs(reversal_prices[j]["price"] - reversal_prices[i]["price"]) <= tolerance:
                cluster.append(reversal_prices[j])
                used[j] = True
        
        if len(cluster) >= SR_MIN_TOUCHES:
            avg_price = sum(p["price"] for p in cluster) / len(cluster)
            last_index = max(p["index"] for p in cluster)
            candles_ago = n - 1 - last_index
            
            # ── Classificar tipo de zona: suporte, resistência ou mista ──
            n_highs = sum(1 for p in cluster if p["type"] == "high")
            n_lows = sum(1 for p in cluster if p["type"] == "low")
            if n_lows > n_highs:
                zone_type = "support"    # Maioria de pivot LOWs → zona de suporte
            elif n_highs > n_lows:
                zone_type = "resistance" # Maioria de pivot HIGHs → zona de resistência
            else:
                zone_type = "mixed"      # Empate → determinar pelo contexto
            
            zones.append({
                "price": avg_price,
                "touches": len(cluster),
                "zone_high": avg_price + tolerance * 0.5,
                "zone_low": avg_price - tolerance * 0.5,
                "candles_ago": candles_ago,
                "last_index": last_index,
                "type": zone_type,
                "high_count": n_highs,
                "low_count": n_lows,
            })
    
    # Ordenar por número de toques (mais fortes primeiro)
    zones.sort(key=lambda z: z["touches"], reverse=True)
    return zones


# ═══════════════════════════════════════════════════════════════
# 2. ANÁLISE MATEMÁTICA DE CANDLE (features numéricas, não visuais)
# ═══════════════════════════════════════════════════════════════
def _candle_features(df: pd.DataFrame, atr_val: float) -> Dict[str, float]:
    """Calcula EMA do período dado. Retorna o valor atual."""
    if len(closes) < period:
        return float(closes[-1])
    
    multiplier = 2.0 / (period + 1)
    ema = float(closes[0])
    for i in range(1, len(closes)):
        ema = (closes[i] - ema) * multiplier + ema
    return ema


def _ema_filter(price: float, ema: float, atr_val: float
                ) -> Tuple[str, float]:
    """
    Determina direção permitida baseada na posição do preço vs EMA.
    
    LÓGICA:
    - Preço ACIMA da EMA = tendência de alta → preço pode SUBIR até resistência
    - Preço ABAIXO da EMA = tendência de baixa → preço pode CAIR até suporte
    - Preço no MEIO (±dead_zone) = sem tendência clara → NÃO OPERA
    
    Retorna: (allowed_direction, ema_strength)
    - "ABOVE" = preço acima EMA (tendência alta)
    - "BELOW" = preço abaixo EMA (tendência baixa)
    - "NEUTRAL" = zona morta
    """
    if atr_val <= 0:
        return "NEUTRAL", 0.0
    
    distance_atr = (price - ema) / atr_val
    
    if abs(distance_atr) <= EMA_DEAD_ZONE_ATR:
        return "NEUTRAL", abs(distance_atr)
    
    strength = min(1.0, abs(distance_atr) / 2.0)
    
    if distance_atr > 0:
        return "ABOVE", strength
    else:
        return "BELOW", strength


# ═══════════════════════════════════════════════════════════════
# 3. CANAL DE PREÇO (SIMPLES)
# ═══════════════════════════════════════════════════════════════
def _detect_channel(df: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    """
    Detecta canal de preço: linhas de suporte e resistência paralelas.
    Usa swing highs/lows com regressão linear.
    """
    no_channel = {
        "has_channel": False, "channel_type": "none",
        "price_position": 0.5, "signal": None, "quality": 0.0,
    }
    
    if len(df) < CHANNEL_LOOKBACK or atr_val <= 0:
        return no_channel
    
    data = df.tail(CHANNEL_LOOKBACK)
    highs = data["high"].astype(float).values
    lows = data["low"].astype(float).values
    closes = data["close"].astype(float).values
    n = len(data)
    margin = 3
    
    # Swing highs e lows
    sh, sl = [], []
    for i in range(margin, n - margin):
        if all(highs[i] >= highs[i - j] for j in range(1, margin + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, margin + 1)):
            sh.append((float(i), highs[i]))
        if all(lows[i] <= lows[i - j] for j in range(1, margin + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, margin + 1)):
            sl.append((float(i), lows[i]))
    
    if len(sh) < CHANNEL_MIN_TOUCHES or len(sl) < CHANNEL_MIN_TOUCHES:
        return no_channel
    
    try:
        h_slope, h_int = np.polyfit([s[0] for s in sh], [s[1] for s in sh], 1)
        l_slope, l_int = np.polyfit([s[0] for s in sl], [s[1] for s in sl], 1)
    except Exception:
        return no_channel
    
    # Verificar paralelismo básico
    slope_diff = abs(h_slope - l_slope)
    avg_slope = (abs(h_slope) + abs(l_slope)) / 2.0
    if avg_slope > atr_val * 0.001 and slope_diff > avg_slope * 0.65:
        return no_channel
    
    # Largura do canal
    idx = float(n - 1)
    upper = h_slope * idx + h_int
    lower = l_slope * idx + l_int
    width = upper - lower
    
    if width <= 0 or width < atr_val * 0.8:
        return no_channel
    
    # Qualidade: quantas velas violam o canal?
    violations = 0
    tol = width * 0.15
    for i in range(n):
        u = h_slope * i + h_int
        lo = l_slope * i + l_int
        if highs[i] > u + tol or lows[i] < lo - tol:
            violations += 1
    
    quality = max(0.0, min(1.0, 1.0 - violations / float(n) * 3.0))
    if quality < 0.35:
        return no_channel
    
    # Posição do preço no canal
    price = closes[-1]
    position = max(0.0, min(1.0, (price - lower) / width))
    
    # Sinal na borda
    signal = None
    if position <= CHANNEL_BOUNDARY:
        signal = "CALL"
    elif position >= (1.0 - CHANNEL_BOUNDARY):
        signal = "PUT"
    
    # Tipo
    mid_slope = (h_slope + l_slope) / 2.0
    s_norm = mid_slope / atr_val if atr_val > 0 else 0.0
    if abs(s_norm) < 0.003:
        ch_type = "horizontal"
    elif s_norm > 0:
        ch_type = "ascending"
    else:
        ch_type = "descending"
    
def _candle_features(df: pd.DataFrame, atr_val: float) -> Dict[str, float]:
    """
    Extrai features MATEMÁTICAS da última vela + contexto.
    Alinhado com pipeline de referência (OHLC -> features -> IA).
    
    Retorna dict com:
      body_ratio        — corpo / range total (0=doji, 1=sem pavio)
      upper_wick_ratio  — pavio sup / range total
      lower_wick_ratio  — pavio inf / range total
      close_position    — (close-low)/(high-low) → 1=fechou no topo, 0=no fundo
      body_vs_ma20      — corpo atual / média dos últimos 20 corpos
      range_vs_atr      — range total / ATR (>1 = expansão, <0.5 = compressão)
      range_vs_ma20     — range atual / média dos últimos 20 ranges
      direction_num     — +1 se bullish, -1 se bearish, 0 se doji
      absorption_bull   — absorção compradora (pavio inf grande + fechamento forte)
      absorption_bear   — absorção vendedora (pavio sup grande + fechamento fraco)
      body_strength     — quão forte é o corpo relativo ao ATR
      ret1              — retorno percentual do último candle
      ret3              — retorno percentual dos últimos 3 candles
      ret5              — retorno percentual dos últimos 5 candles
    """
    result = {
        "body_ratio": 0.0, "upper_wick_ratio": 0.0, "lower_wick_ratio": 0.0,
        "close_position": 0.5, "body_vs_ma20": 1.0, "range_vs_atr": 0.5,
        "range_vs_ma20": 1.0,
        "direction_num": 0.0, "absorption_bull": 0.0, "absorption_bear": 0.0,
        "body_strength": 0.0,
        "ret1": 0.0, "ret3": 0.0, "ret5": 0.0,
    }
    
    if len(df) < 2 or atr_val <= 0:
        return result
    
    candle = df.iloc[-1]
    o, h, l, c = float(candle["open"]), float(candle["high"]), float(candle["low"]), float(candle["close"])
    
    body = abs(c - o)
    total_range = h - l
    
    if total_range < atr_val * 0.02:
        return result  # vela insignificante
    
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    
    # --- Ratios fundamentais ---
    result["body_ratio"] = body / total_range
    result["upper_wick_ratio"] = upper_wick / total_range
    result["lower_wick_ratio"] = lower_wick / total_range
    result["close_position"] = (c - l) / total_range  # 1=topo, 0=fundo
    result["range_vs_atr"] = total_range / atr_val
    result["direction_num"] = 1.0 if c > o else (-1.0 if c < o else 0.0)
    result["body_strength"] = body / atr_val
    
    # --- Corpo vs média 20 (pipeline: body_vs_ma20) ---
    n_avg = min(20, len(df) - 1)
    if n_avg >= 3:
        bodies = []
        ranges = []
        for i in range(len(df) - n_avg - 1, len(df) - 1):
            _o = float(df.iloc[i]["open"])
            _h = float(df.iloc[i]["high"])
            _l = float(df.iloc[i]["low"])
            _c = float(df.iloc[i]["close"])
            bodies.append(abs(_c - _o))
            ranges.append(_h - _l)
        avg_body = np.mean(bodies) if bodies else body
        avg_range = np.mean(ranges) if ranges else total_range
        result["body_vs_ma20"] = body / avg_body if avg_body > 1e-9 else 1.0
        result["range_vs_ma20"] = total_range / avg_range if avg_range > 1e-9 else 1.0
    
    # --- Momentum: retornos percentuais (pipeline: ret1, ret3, ret5) ---
    closes = df["close"].astype(float).values
    if len(closes) >= 2:
        result["ret1"] = (closes[-1] / closes[-2] - 1.0) if closes[-2] > 1e-9 else 0.0
    if len(closes) >= 4:
        result["ret3"] = (closes[-1] / closes[-4] - 1.0) if closes[-4] > 1e-9 else 0.0
    if len(closes) >= 6:
        result["ret5"] = (closes[-1] / closes[-6] - 1.0) if closes[-6] > 1e-9 else 0.0
    
    # --- Absorção: quem ganhou a briga nesta vela ---
    if result["lower_wick_ratio"] > 0.0:
        result["absorption_bull"] = result["lower_wick_ratio"] * result["close_position"]
    if result["upper_wick_ratio"] > 0.0:
        result["absorption_bear"] = result["upper_wick_ratio"] * (1.0 - result["close_position"])
    
    return result


def _check_rejection(df: pd.DataFrame, direction: str, atr_val: float
                     ) -> Tuple[bool, str, float]:
    """
    Analisa rejeição usando features MATEMÁTICAS, não padrões visuais.
    
    Para CALL (suporte):
      - Quer absorção bull alta (pavio inf grande + close alto)
      - Engulfing = corpo atual cobre o anterior inteiro na direção certa
    
    Para PUT (resistência):
      - Quer absorção bear alta (pavio sup grande + close baixo)
      - Engulfing = corpo atual cobre o anterior inteiro na direção certa
    
    Retorna: (has_rejection, pattern_label, quality 0-1)
    Quality é contínua, não binária — baseada nos ratios reais.
    """
    if len(df) < 2 or atr_val <= 0:
        return False, "none", 0.0
    
    feat = _candle_features(df, atr_val)
    
    candle = df.iloc[-1]
    o, h, l, c = float(candle["open"]), float(candle["high"]), float(candle["low"]), float(candle["close"])
    total_range = h - l
    
    if total_range < atr_val * 0.05:
        return False, "doji_tiny", 0.0
    
    prev = df.iloc[-2]
    prev_o, prev_c = float(prev["open"]), float(prev["close"])
    
    if direction == "CALL":
        # === CALL: queremos evidência de compradores defendendo ===
        
        # 1. Absorção bullish forte (pavio inferior > 50% + close nos 60% superiores)
        absorption = feat["absorption_bull"]
        wick_r = feat["lower_wick_ratio"]
        body_r = feat["body_ratio"]
        close_pos = feat["close_position"]
        
        # Engulfing bullish (matemático: corpo bull cobre corpo bear anterior)
        if prev_c < prev_o and c > o and c > prev_o and o <= prev_c:
            # Quality: quanto maior absorção + corpo vs avg, melhor
            q = min(1.0, 0.55 + absorption * 0.3 + min(feat["body_vs_ma20"], 2.0) * 0.1)
            return True, "engulfing_bull", q
        
        # Hammer / Pin bar bull (matemático)
        if wick_r >= 0.45 and body_r <= 0.50 and close_pos >= 0.55:
            # Quality contínua: proporcional ao pavio e posição do close
            q = min(1.0, wick_r * 0.7 + close_pos * 0.3)
            label = "hammer" if c > o else "pin_bar_bull"
            return True, label, q
        
        # Rejeição moderada: pavio inferior > corpo, close acima do meio
        if wick_r > 0.30 and wick_r > body_r and close_pos >= 0.50:
            q = min(0.65, wick_r * 0.6 + close_pos * 0.2)
            return True, "wick_bull", q
    
    elif direction == "PUT":
        # === PUT: queremos evidência de vendedores defendendo ===
        
        absorption = feat["absorption_bear"]
        wick_r = feat["upper_wick_ratio"]
        body_r = feat["body_ratio"]
        close_pos = feat["close_position"]
        
        # Engulfing bearish (matemático)
        if prev_c > prev_o and c < o and o >= prev_c and c <= prev_o:
            q = min(1.0, 0.55 + absorption * 0.3 + min(feat["body_vs_ma20"], 2.0) * 0.1)
            return True, "engulfing_bear", q
        
        # Shooting star / Pin bar bear (matemático)
        if wick_r >= 0.45 and body_r <= 0.50 and close_pos <= 0.45:
            q = min(1.0, wick_r * 0.7 + (1.0 - close_pos) * 0.3)
            label = "shooting_star" if c < o else "pin_bar_bear"
            return True, label, q
        
        # Rejeição moderada: pavio superior > corpo, close abaixo do meio
        if wick_r > 0.30 and wick_r > body_r and close_pos <= 0.50:
            q = min(0.65, wick_r * 0.6 + (1.0 - close_pos) * 0.2)
            return True, "wick_bear", q
    
    return False, "none", 0.0


# ═══════════════════════════════════════════════════════════════
# 3. MOMENTUM CONTRA (anti-breakout simples)
# ═══════════════════════════════════════════════════════════════
def _check_momentum_against(df: pd.DataFrame, direction: str,
                             atr_val: float) -> bool:
    """
    Retorna True se tem 3+ velas fortes consecutivas INDO PARA a zona.
    Ex: 3 velas bearish fortes → preço chegando em suporte com força → pode romper.
    """
    if len(df) < 5 or atr_val <= 0:
        return False
    
    count = 0
    for i in range(len(df) - 1, max(len(df) - 6, 0), -1):
        o = float(df.iloc[i]["open"])
        c = float(df.iloc[i]["close"])
        body = abs(c - o)
        
        if body < atr_val * 0.15:
            break
        
        if direction == "CALL" and c < o:
            # Bearish candle indo para suporte com força
            count += 1
        elif direction == "PUT" and c > o:
            # Bullish candle indo para resistência com força
            count += 1
        else:
            break
    
    return count >= 3


# ═══════════════════════════════════════════════════════════════
# 4. FUNÇÃO PRINCIPAL: sr_precision_signal
# ═══════════════════════════════════════════════════════════════
def sr_precision_signal(df_m1: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    """
    Estratégia S/R + Candle Features.
    A IA (Bayesiano + LGBM) no WS_AUTO_AI.py aprende por ativo e valida.
    
    Retorna dict compatível com WS_AUTO_AI.
    precision_trade=True quando sinal é válido.
    """
    # ── Resultado neutro (template com TODOS os campos que WS_AUTO_AI lê) ──
    neutral = {
        "trade": False, "precision_trade": False,
        "dir": "NEUTRAL", "score": 0.0,
        "setup_type": "simple_sr",
        "sr_proximity": 0.0, "sr_touches": 0,
        "sr_weight": 0.0, "sr_reason": "sem_sinal",
        "sr_bonus": 0.0, "sr_rejections": 0,
        "sr_false_breaks": 0, "sr_strength": 0.0,
        "inside_zone": False,
        "has_lt": False, "lt_confluence": 0.0,
        "lt_proximity": 0.0, "lt_points": 0,
        "candle_pattern": "none", "candle_strength": 0.0,
        "market_quality": 0.5, "context": "neutro",
        "entry_confidence": 0.0,
        "confluence_count": 0, "confluence_bonus": 0.0,
        "m5_trend": "neutral", "m5_confirms": False,
        "breakout_risk": "low", "breakout_risk_score": 0.0,
        "approach_quality": 0.5, "approach_clean": True,
        "has_channel": False, "channel_type": "none",
        "channel_confirms": False, "channel_quality": 0.0,
        "channel_position": 0.5,
        "overext_favors": False, "overext_exhaustion": False,
        "overext_consecutive": 0, "overext_displacement": 0.0,
        "effA": 0.0, "rsi": 50.0, "retr": 0.0,
        "A_atr": 0.0, "flips": 0.5,
        "comp": 0.0, "late": 0.0, "distBreak": 0.0,
        "pb_len": 0, "risk_atr": 0.0,
        "trend_strength": 0.0, "trend_reason": "simple_sr",
        "retracement": 0.5, "pullback_candles": 2,
        "entry_confirmation": 0.5,
        "reasons": [],
    }
    
    # ── Dados suficientes? ──
    if df_m1 is None or len(df_m1) < MIN_CANDLES:
        nr = neutral.copy()
        nr["reasons"] = [f"min_velas({len(df_m1) if df_m1 is not None else 0}<{MIN_CANDLES})"]
        return nr
    
    closes = df_m1["close"].astype(float).values
    current_price = float(closes[-1])
    
    # ════════════════════════════════════════════════════════
    # PASSO 0: FILTRO CHOPPY — muitos pavios = mercado ruim
    # ════════════════════════════════════════════════════════
    is_choppy, wick_ratio = _is_choppy_market(df_m1, atr_val)
    # CHOPPY NÃO bloqueia mais — info vai no market_quality para IA decidir
    # (antes retornava score=0.00, agora apenas reduz market_quality)
    
    # ════════════════════════════════════════════════════════
    # PASSO 1: ZONAS S/R — preço perto de zona forte?
    # ════════════════════════════════════════════════════════
    lookback_df = df_m1.tail(SR_LOOKBACK)
    zones = _find_sr_zones(lookback_df, atr_val)
    
    if not zones:
        nr = neutral.copy()
        nr["reasons"] = ["sem_zonas_sr"]
        return nr
    
    # Encontrar zona ativa (preço perto + alinhada com EMA)
    active_zone = None
    zone_dir = None
    proximity = SR_PROXIMITY_ATR * atr_val
    
    # Média recente para desempate de zonas mistas
    recent_avg = float(closes[-40:].mean()) if len(closes) >= 40 else float(closes.mean())
    
    for z in zones:
        dist = abs(current_price - z["price"])
        if dist > proximity:
            continue
        
        # ── Determinar tipo da zona: SUPORTE ou RESISTÊNCIA ──
        # Primário: usar tipo dos pivôs (pivot LOWs = suporte, pivot HIGHs = resistência)
        # Secundário: posição relativa à média recente (desempate para zonas mistas)
        zone_type = z.get("type", "mixed")
        
        if zone_type == "support":
            candidate_dir = "CALL"   # Zona de suporte → compradores defendem → CALL
        elif zone_type == "resistance":
            candidate_dir = "PUT"    # Zona de resistência → vendedores defendem → PUT
        else:
            # Zona mista: usar posição relativa à média recente dos preços
            if z["price"] < recent_avg - atr_val * 0.10:
                candidate_dir = "CALL"  # Zona abaixo da média → suporte
            elif z["price"] > recent_avg + atr_val * 0.10:
                candidate_dir = "PUT"   # Zona acima da média → resistência
            else:
                continue  # Zona no meio, ambígua → pular
        
        # ── FILTRO DE APROXIMAÇÃO: preço deve vir do lado CORRETO ──
        # CALL em suporte: preço deve vir de CIMA (pullback) → avg_approach > zona
        # PUT em resistência: preço deve vir de BAIXO (rally) → avg_approach < zona
        # Se preço se aproxima do lado errado → zona age como OPOSTO → SKIP
        if len(closes) >= 8:
            approach_avg = float(np.mean(closes[-8:-1]))  # média 7 velas anteriores
            approach_thr = atr_val * 0.10
            
            if candidate_dir == "CALL" and approach_avg < z["price"] - approach_thr:
                # Preço vinha de BAIXO → subindo para a zona → é RESISTÊNCIA → SKIP
                continue
            elif candidate_dir == "PUT" and approach_avg > z["price"] + approach_thr:
                # Preço vinha de CIMA → caindo para a zona → é SUPORTE → SKIP
                continue
        
        # ── ZONA VÁLIDA: S/R type + approach confirmado ──
        active_zone = z
        zone_dir = candidate_dir
        active_zone["distance"] = dist
        active_zone["distance_atr"] = dist / atr_val if atr_val > 0 else 0
        break
    
    if active_zone is None:
        nr = neutral.copy()
        best = zones[0]
        best_dist = abs(current_price - best["price"]) / atr_val if atr_val > 0 else 99
        nr["sr_touches"] = best["touches"]
        best_type = best.get("type", "?")
        nr["reasons"] = [
            f"zona_longe(melhor={best['touches']}t,tipo={best_type},d={best_dist:.1f}ATR)",
        ]
        return nr
    
    # ════════════════════════════════════════════════════════
    # PASSO 2: FEATURES MATEMÁTICAS DO CANDLE + REJEIÇÃO
    # ════════════════════════════════════════════════════════
    candle_feat = _candle_features(df_m1, atr_val)
    has_rejection, rej_pattern, rej_quality = _check_rejection(
        df_m1, zone_dir, atr_val
    )
    
    # ════════════════════════════════════════════════════════
    # PASSO 2.5: TENDÊNCIA — detectar direção do mercado
    # ════════════════════════════════════════════════════════
    trend_dir, trend_strength = _detect_trend(df_m1, atr_val)
    
    # Verificar se sinal é contra a tendência (micro 20 velas)
    is_counter_trend = False
    if trend_dir == "up" and zone_dir == "PUT":
        is_counter_trend = True
    elif trend_dir == "down" and zone_dir == "CALL":
        is_counter_trend = True
    
    # MACRO TENDÊNCIA — 50 velas (tendência dominante)
    macro_trend_dir, macro_trend_strength = _detect_trend(df_m1, atr_val, lookback=MACRO_TREND_LOOKBACK)
    is_counter_macro = False
    if macro_trend_dir == "up" and zone_dir == "PUT":
        is_counter_macro = True
    elif macro_trend_dir == "down" and zone_dir == "CALL":
        is_counter_macro = True
    
    # ════════════════════════════════════════════════════════
    # PASSO 3: MOMENTUM CONTRA — breakout risk?
    # ════════════════════════════════════════════════════════
    momentum_against = _check_momentum_against(df_m1, zone_dir, atr_val)
    
    # ════════════════════════════════════════════════════════
    # SCORING — LIMPO & TRANSPARENTE
    # Cada componente contribui de forma clara e previsível
    # 3t=0.50 | 4t=0.55 | 5t=0.60 | 6t=0.65 | 7t=0.70
    # +rejeição: até +0.08 | +trend alinhado: +0.04
    # -macro contra: -0.06 | +fresca: +0.03
    # ════════════════════════════════════════════════════════
    score = SCORE_BASE  # 0.50
    reasons = []
    
    # Bônus: toques extras (zona mais forte = mais confiável)
    extra = max(0, active_zone["touches"] - SR_MIN_TOUCHES)
    touch_bonus = min(extra * SCORE_PER_TOUCH, SCORE_MAX_TOUCH_BONUS)
    if extra > 0:
        reasons.append(f"+toques({active_zone['touches']})")
    score += touch_bonus
    
    # Bônus: rejeição candle (confirmação de bounce na zona)
    if has_rejection:
        rej_bonus = SCORE_REJECTION_MAX * rej_quality
        score += rej_bonus
        reasons.append(f"rejeicao({rej_pattern},q={rej_quality:.2f},+{rej_bonus:.3f})")
    
    # Bônus: trade COM a macro tendência (vento a favor)
    _trade_with_macro = False
    if macro_trend_dir == "up" and zone_dir == "CALL":
        _trade_with_macro = True
    elif macro_trend_dir == "down" and zone_dir == "PUT":
        _trade_with_macro = True
    if _trade_with_macro and macro_trend_strength > 0.30:
        score += SCORE_TREND_ALIGN
        reasons.append(f"com_macro({macro_trend_dir},+{SCORE_TREND_ALIGN})")
    
    # Penalidade: contra a macro tendência FORTE
    if is_counter_macro and macro_trend_strength > 0.30:
        score -= SCORE_MACRO_AGAINST
        reasons.append(f"contra_macro({macro_trend_dir},str={macro_trend_strength:.2f},-{SCORE_MACRO_AGAINST})")
    
    # Bônus/penalidade: zona fresca ou antiga
    if active_zone["candles_ago"] <= 40:
        score += 0.03
        reasons.append("zona_fresca")
    elif active_zone["candles_ago"] > 120:
        score -= 0.03
        reasons.append("zona_antiga")
    
    # Clamp — range honesto
    score = max(0.30, min(0.85, score))
    
    # ════════════════════════════════════════════════════════
    # DECISÃO: score mínimo para enviar à IA
    # ════════════════════════════════════════════════════════
    min_score = 0.48  # Permissivo — IA é o filtro real
    is_trade = score >= min_score
    
    # Confluências
    conf_count = 1  # S/R sempre conta
    if has_rejection:
        conf_count += 1
    if active_zone["touches"] >= 5:
        conf_count += 1
    if _trade_with_macro:
        conf_count += 1
    
    # Entry confidence
    entry_conf = 0.50
    if has_rejection:
        entry_conf = max(entry_conf, rej_quality)
    
    # Market quality — contexto do mercado para IA usar
    mkt_quality = 0.60
    if not momentum_against:
        mkt_quality += 0.15
    if wick_ratio > 0.30:
        mkt_quality -= (wick_ratio - 0.30) * 0.5
    mkt_quality = min(0.90, mkt_quality)
    if is_counter_trend:
        mkt_quality -= 0.10
    if is_counter_macro and macro_trend_strength > 0.30:
        mkt_quality -= MACRO_TREND_MQ_PENALTY
    if _trade_with_macro:
        mkt_quality += 0.10  # Vento a favor melhora contexto
    mkt_quality = max(0.10, min(0.95, mkt_quality))
    context = "bom" if mkt_quality >= 0.60 else "neutro"
    
    # Setup type
    setup_type = "sr_bounce"
    if active_zone["touches"] >= 5:
        setup_type = "sr_forte"
    
    reasons_final = [
        f"SR_{zone_dir}_{setup_type}",
        f"zona={active_zone['touches']}t({active_zone.get('type','?')},d={active_zone['distance_atr']:.2f}ATR)",
        f"rejeicao={rej_pattern}(q={rej_quality:.2f})" if has_rejection else "rejeicao=NAO",
        f"momentum={'CONTRA' if momentum_against else 'OK'}",
        f"trend={trend_dir}(str={trend_strength:.2f}){'_CONTRA' if is_counter_trend else ''}",
        f"macro={macro_trend_dir}(str={macro_trend_strength:.2f}){'_CONTRA' if is_counter_macro else ''}",
        f"conf={conf_count}",
        *reasons,
    ]
    
    # ════════════════════════════════════════════════════════
    # RETORNO — compatível com WS_AUTO_AI (IA aprende com isso)
    # ════════════════════════════════════════════════════════
    return {
        "trade": is_trade,
        "precision_trade": is_trade,
        "dir": zone_dir,
        "score": score,
        "setup_type": setup_type,
        # S/R
        "sr_proximity": active_zone["distance_atr"],
        "sr_touches": active_zone["touches"],
        "sr_weight": float(active_zone["touches"]),
        "sr_strength": float(active_zone["touches"] * 10),
        "sr_reason": setup_type,
        "sr_bonus": float(active_zone["touches"]) * 0.5,
        "sr_rejections": 1 if has_rejection else 0,
        "sr_false_breaks": 0,
        "inside_zone": active_zone["distance_atr"] <= 0.15,
        # Trendline (removido — sem EMA/LT)
        "has_lt": False,
        "lt_confluence": 0.0,
        "lt_proximity": 0.0,
        "lt_points": 0,
        # Candle — padrão + features matemáticas para IA
        "candle_pattern": rej_pattern,
        "candle_strength": rej_quality,
        "candle_body_ratio": candle_feat["body_ratio"],
        "candle_upper_wick": candle_feat["upper_wick_ratio"],
        "candle_lower_wick": candle_feat["lower_wick_ratio"],
        "candle_close_pos": candle_feat["close_position"],
        "candle_body_vs_avg": candle_feat["body_vs_ma20"],
        "candle_range_vs_atr": candle_feat["range_vs_atr"],
        "candle_range_vs_ma20": candle_feat["range_vs_ma20"],
        "candle_absorption_bull": candle_feat["absorption_bull"],
        "candle_absorption_bear": candle_feat["absorption_bear"],
        "candle_body_strength": candle_feat["body_strength"],
        "candle_ret1": candle_feat["ret1"],
        "candle_ret3": candle_feat["ret3"],
        "candle_ret5": candle_feat["ret5"],
        # Mercado
        "market_quality": mkt_quality,
        "context": context,
        # Entrada
        "entry_confidence": entry_conf,
        "entry_confirmation": entry_conf,
        # Confluência
        "confluence_count": conf_count,
        "confluence_bonus": conf_count * 0.15,
        # M5 (removido — simplicidade)
        "m5_trend": "neutral",
        "m5_confirms": False,
        # Breakout risk
        "breakout_risk": "high" if momentum_against else "low",
        "breakout_risk_score": 0.8 if momentum_against else 0.1,
        # Approach
        "approach_quality": 0.6,
        "approach_clean": not momentum_against,
        # Canal (removido)
        "has_channel": False,
        "channel_type": "none",
        "channel_confirms": False,
        "channel_quality": 0.0,
        "channel_position": 0.5,
        # Overext (removido — simplicidade)
        "overext_favors": False,
        "overext_exhaustion": False,
        "overext_consecutive": 0,
        "overext_displacement": 0.0,
        # Legacy (IA Bayesiana/LGBM usa esses campos)
        "effA": 0.0,
        "rsi": 50.0,
        "retr": active_zone["distance_atr"],
        "A_atr": float(active_zone["touches"]),
        "flips": 0.5,
        "comp": 0.0,
        "late": 0.0,
        "distBreak": 0.0,
        "pb_len": conf_count,
        "risk_atr": 0.8 if momentum_against else 0.1,
        "trend_strength": 0.0,
        "trend_reason": f"sr_{setup_type}",
        "trend_dir_detected": trend_dir,
        "trend_strength_detected": trend_strength,
        "is_counter_trend": is_counter_trend,
        "macro_trend_dir": macro_trend_dir,
        "macro_trend_strength": macro_trend_strength,
        "is_counter_macro": is_counter_macro,
        "retracement": 0.5,
        "pullback_candles": 2,
        # Razões
        "reasons": reasons_final,
    }
