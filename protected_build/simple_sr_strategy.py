# -*- coding: utf-8 -*-
"""
ESTRATÉGIA S/R LIMPA — Do Zero
════════════════════════════════

O que um trader faz no gráfico:
  1. Marca zonas de SUPORTE (preço bateu e voltou várias vezes)
  2. Marca zonas de RESISTÊNCIA (preço bateu e voltou várias vezes)
  3. Espera o preço CHEGAR na zona
  4. Espera CONFIRMAÇÃO: pullback (voltou pra zona) ou turnaround (virou na zona)
  5. Entra na direção do bounce

Regras:
  - Zona precisa ter >=3 toques históricos
  - Preço precisa estar DENTRO ou MUITO PERTO da zona (< 0.5 ATR)
  - Último candle precisa CONFIRMAR reversão (wick rejeição ou candle de reversão)
  - Sem IA, sem ensemble, sem score complexo
  - Só lógica pura de S/R

Compatível com WS_AUTO_AI_BULLEX.py (mesmo formato de retorno dict).
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple

try:
    from ws_candle_color_ai import predict_candle_color
except ImportError:
    predict_candle_color = None

log = logging.getLogger("WS_AUTO_AI")

# ═══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════
SR_LOOKBACK    = 300    # Quantas velas olhar para trás para achar pivôs (5h no M1)
SR_MIN_TOUCHES = 3      # Mínimo de toques para zona ser válida (3 = zona confirmada)
SR_CLUSTER_ATR = 0.50   # Distância (em ATR) para agrupar toques na mesma zona
SR_MAX_DIST    = 0.30   # Preço MÁXIMO distância da zona (em ATR)
                        # 0.30 = preço DEVE estar tocando a zona (máx 0.30 ATR)
                        # Entrar SOMENTE quando o preço está no S/R
PIVOT_WINDOW   = 3      # Janela para detectar pivot high/low
                        # 3 = precisa ser extremo em 7 candles (7min no M1)
                        # OTC M1 tem micro-estrutura — 5 era largo demais
                        # Trader real marca pivôs visíveis, não espera swing perfeito
PIVOT_MIN_SIG  = 0.06   # Significância mínima do pivô em ATR (0.06 = filtro suave, aceita pivots reais em OTC 1min)
                        # Pivô precisa ser >= 0.15 ATR acima/abaixo dos vizinhos
                        # Elimina micro-pivôs que são apenas ruído de preço
MIN_CANDLES    = 50     # Mínimo de velas para rodar
RANGE_WINDOW   = 30     # Janela para calcular range (topo/fundo)
                        # 30 velas = 30 min no M1 → range ATUAL, não histórico
                        # Antes era 50 → picos antigos distorciam o range


# ═══════════════════════════════════════════════════════════════
# 1. DETECTAR PIVÔS (pontos de virada do preço)
# ═══════════════════════════════════════════════════════════════
def _find_pivots(highs: np.ndarray, lows: np.ndarray,
                 window: int = PIVOT_WINDOW,
                 atr_val: float = 0.0) -> Tuple[List[float], List[float]]:
    """
    Pivot High = ponto onde o preço fez um topo local SIGNIFICATIVO.
    Pivot Low  = ponto onde o preço fez um fundo local SIGNIFICATIVO.

    LEITURA GRÁFICA REAL:
      Um trader NÃO marca cada micro-swing. Ele marca APENAS os pontos
      onde o preço realmente PAROU e REVERTEU com força visível.

      Filtro de significância: o pivô precisa estar pelo menos
      PIVOT_MIN_SIG * ATR acima/abaixo dos candles vizinhos.
      Isso elimina ruído e mantém apenas zonas estruturais reais.
    """
    pivot_highs = []
    pivot_lows = []
    min_sig = PIVOT_MIN_SIG * atr_val if atr_val > 0 else 0.0

    for i in range(window, len(highs) - window):
        # Pivot high: este high é o MAIOR na janela
        if highs[i] >= max(highs[i - window : i + window + 1]):
            # Filtro de significância: o pivô precisa estar
            # significativamente acima dos vizinhos mais próximos
            if min_sig > 0:
                neighbors_h = list(highs[max(0, i-window):i]) + \
                              list(highs[i+1:i+window+1])
                if neighbors_h:
                    second_highest = max(neighbors_h)
                    if (highs[i] - second_highest) < min_sig:
                        continue  # Pivô insignificante — ruído
            pivot_highs.append(float(highs[i]))

        # Pivot low: este low é o MENOR na janela
        if lows[i] <= min(lows[i - window : i + window + 1]):
            # Filtro de significância: pivô precisa estar
            # significativamente abaixo dos vizinhos
            if min_sig > 0:
                neighbors_l = list(lows[max(0, i-window):i]) + \
                              list(lows[i+1:i+window+1])
                if neighbors_l:
                    second_lowest = min(neighbors_l)
                    if (second_lowest - lows[i]) < min_sig:
                        continue  # Pivô insignificante — ruído
            pivot_lows.append(float(lows[i]))

    return pivot_highs, pivot_lows


# ═══════════════════════════════════════════════════════════════
# 2. AGRUPAR PIVÔS EM ZONAS (clusters)
# ═══════════════════════════════════════════════════════════════
def _cluster_into_zones(levels: List[float],
                        tolerance: float) -> List[Dict[str, Any]]:
    """
    Agrupa preços próximos em ZONAS.
    Cada zona tem: price (média), zone_high, zone_low, touches.
    """
    if not levels:
        return []

    sorted_lvls = sorted(levels)
    clusters: List[List[float]] = []
    current = [sorted_lvls[0]]

    for lvl in sorted_lvls[1:]:
        if lvl - current[0] <= tolerance:
            current.append(lvl)
        else:
            clusters.append(current)
            current = [lvl]
    clusters.append(current)

    zones = []
    for c in clusters:
        if len(c) >= SR_MIN_TOUCHES:
            price = float(np.mean(c))
            zones.append({
                "price": price,
                "zone_high": float(max(c)) + tolerance * 0.25,
                "zone_low": float(min(c)) - tolerance * 0.25,
                "touches": len(c),
            })

    return sorted(zones, key=lambda z: z["touches"], reverse=True)


# ═══════════════════════════════════════════════════════════════
# 3. DETECTAR ZONAS S/R
# ═══════════════════════════════════════════════════════════════
def detect_sr_zones(df: pd.DataFrame, atr_val: float) -> List[Dict]:
    """
    Retorna lista de zonas S/R ordenadas por força (toques).

    LEITURA GRÁFICA REAL:
      Trader profissional separa SUPORTES (pontos onde o preço fez fundo
      e voltou) de RESISTÊNCIAS (pontos onde o preço fez topo e voltou).
      Depois verifica se algum nível serve como ambos (flip zone).

      Pivôs de high → zonas de RESISTÊNCIA
      Pivôs de low → zonas de SUPORTE
      Se uma zona de suporte e uma de resistência coincidem → MIXED (forte!)
    """
    if len(df) < 20:
        return []

    lookback_df = df.tail(SR_LOOKBACK)
    tolerance = SR_CLUSTER_ATR * atr_val

    highs = lookback_df["high"].astype(float).values
    lows = lookback_df["low"].astype(float).values

    pivot_highs, pivot_lows = _find_pivots(highs, lows, atr_val=atr_val)

    # ── CLUSTERING SEPARADO: suportes e resistências em clusters distintos ──
    # Isso evita misturar um topo com um fundo na mesma zona injustamente
    support_zones = _cluster_into_zones(pivot_lows, tolerance)
    resist_zones = _cluster_into_zones(pivot_highs, tolerance)

    # ── DYNAMIC S/R: Swing highs/lows recentes (últimas 30 velas) ──
    # Um trader SEMPRE vê os swings recentes como níveis — mesmo sem cluster.
    # Isso garante que SEMPRE existam zonas perto do preço atual.
    recent_n = min(30, len(lookback_df))
    recent_highs = lookback_df["high"].astype(float).values[-recent_n:]
    recent_lows = lookback_df["low"].astype(float).values[-recent_n:]
    recent_closes = lookback_df["close"].astype(float).values[-recent_n:]
    current_price = float(recent_closes[-1]) if len(recent_closes) > 0 else 0

    # Swing high/low recente (mini-pivots com window=2)
    for i in range(2, len(recent_highs) - 2):
        if recent_highs[i] >= max(recent_highs[max(0,i-2):i+3]):
            level = float(recent_highs[i])
            # Verificar se NÃO já está coberto por um cluster existente
            already_covered = any(abs(z["price"] - level) <= tolerance for z in resist_zones)
            if not already_covered:
                resist_zones.append({
                    "price": level,
                    "zone_high": level + atr_val * 0.10,
                    "zone_low": level - atr_val * 0.10,
                    "touches": 2,  # swing = pelo menos 2 toques implícitos
                    "type": "resistance",
                })

        if recent_lows[i] <= min(recent_lows[max(0,i-2):i+3]):
            level = float(recent_lows[i])
            already_covered = any(abs(z["price"] - level) <= tolerance for z in support_zones)
            if not already_covered:
                support_zones.append({
                    "price": level,
                    "zone_high": level + atr_val * 0.10,
                    "zone_low": level - atr_val * 0.10,
                    "touches": 2,
                    "type": "support",
                })

    # ── DYNAMIC S/R: EMAs como suporte/resistência dinâmico ──
    # Trader profissional: "EMA 20 é suporte em tendência de alta"
    if len(lookback_df) >= 50:
        all_closes = lookback_df["close"].astype(float).values
        ema20 = float(np.mean(all_closes[-20:]))
        ema50 = float(np.mean(all_closes[-50:]))

        for ema_level in [ema20, ema50]:
            dist_to_price = abs(ema_level - current_price)
            if dist_to_price <= 2.0 * atr_val:  # EMA está relevante (perto)
                # EMA abaixo do preço = suporte dinâmico
                # EMA acima do preço = resistência dinâmica
                ema_type = "support" if ema_level < current_price else "resistance"
                target_list = support_zones if ema_type == "support" else resist_zones
                already_covered = any(abs(z["price"] - ema_level) <= tolerance for z in target_list)
                if not already_covered:
                    target_list.append({
                        "price": ema_level,
                        "zone_high": ema_level + atr_val * 0.15,
                        "zone_low": ema_level - atr_val * 0.15,
                        "touches": 3,  # EMA = nível dinâmico confiável
                        "type": ema_type,
                    })

    # Marcar tipo explicitamente
    for z in support_zones:
        z["type"] = "support"
    for z in resist_zones:
        z["type"] = "resistance"

    # Merge: se um suporte e uma resistência coincidem (mesma faixa de preço),
    # fundir em "mixed" (flip zone = muito forte porque preço mudou de papel)
    merged_zones = []
    used_resist = set()

    for sz in support_zones:
        found_match = False
        for ri, rz in enumerate(resist_zones):
            if ri in used_resist:
                continue
            # Zonas coincidem se estão dentro de 1 tolerância uma da outra
            if abs(sz["price"] - rz["price"]) <= tolerance:
                # Fundir: combinar toques, usar média dos preços
                merged = {
                    "price": (sz["price"] + rz["price"]) / 2.0,
                    "zone_high": max(sz["zone_high"], rz["zone_high"]),
                    "zone_low": min(sz["zone_low"], rz["zone_low"]),
                    "touches": sz["touches"] + rz["touches"],
                    "type": "mixed",  # Flip zone = muito forte
                }
                merged_zones.append(merged)
                used_resist.add(ri)
                found_match = True
                break
        if not found_match:
            merged_zones.append(sz)

    # Adicionar resistências não usadas
    for ri, rz in enumerate(resist_zones):
        if ri not in used_resist:
            merged_zones.append(rz)

    return sorted(merged_zones, key=lambda z: z["touches"], reverse=True)


def find_nearest_sr(zones: List[Dict], price: float, atr_val: float,
                    zone_type: str = None) -> Optional[Dict]:
    """Encontra a zona mais próxima do preço atual."""
    best = None
    best_dist = float("inf")

    for z in zones:
        if zone_type and z.get("type") != zone_type and z.get("type") != "mixed":
            continue
        dist = abs(price - z["price"])
        if dist < best_dist:
            best_dist = dist
            best = z

    if best and best_dist <= SR_MAX_DIST * atr_val:
        return best
    return None


# ═══════════════════════════════════════════════════════════════
# 3b. FORÇA DA ZONA — Score inteligente para a IA
# ═══════════════════════════════════════════════════════════════
def calculate_zone_strength(df: pd.DataFrame, zone: Dict, atr_val: float) -> Dict[str, Any]:
    """
    Calcula a FORÇA REAL de uma zona S/R analisando como o preço reagiu
    historicamente a essa zona. Retorna score 0-1 e detalhes.
    
    Uma zona é FORTE quando:
      1. Tem muitos toques (quantas vezes o preço chegou lá e voltou)
      2. Os bounces foram LIMPOS (wicks grandes = rejeição forte)
      3. Houve toque RECENTE (zona ativa, não antiga)
      4. Houve tentativa de breakout que FALHOU (preço tentou romper e voltou)
      5. Bodies dos candles que tocaram são pequenos na zona (indecisão no nível)
    
    A IA usa este score para decidir:
      - zone_strength >= 0.70 → zona FORTE → pode entrar confiante
      - zone_strength 0.50-0.70 → zona MÉDIA → IA decide
      - zone_strength < 0.50 → zona FRACA → IA bloqueia ou reduz score
    """
    if len(df) < 20 or atr_val <= 0:
        return {"zone_strength": 0.3, "recent_touch": False, "clean_bounces": 0,
                "failed_breakouts": 0, "avg_rejection": 0.0, "details": "dados_insuf"}
    
    zone_price = float(zone["price"])
    zone_high = float(zone.get("zone_high", zone_price + atr_val * 0.2))
    zone_low = float(zone.get("zone_low", zone_price - atr_val * 0.2))
    touches = int(zone.get("touches", 0))
    ztype = zone.get("type", "mixed")
    
    # Percorrer candles e analisar reações na zona
    n = len(df)
    clean_bounces = 0          # bounces com wick grande (rejeição limpa)
    failed_breakouts = 0       # tentou romper mas fechou dentro
    total_wick_ratios = []     # wick ratios de candles que tocaram a zona
    recent_touch_idx = -1      # índice do toque mais recente
    touch_distances = []       # distância (em candles) de cada toque
    
    margin = atr_val * 0.20    # margem para considerar "tocou a zona"
    
    for i in range(max(0, n - SR_LOOKBACK), n):
        row = df.iloc[i]
        h = float(row["high"])
        l = float(row["low"])
        o = float(row["open"])
        c = float(row["close"])
        full_range = h - l if h - l > 0 else 0.0001
        body = abs(c - o)
        
        # Candle tocou a zona?
        touched_support = l <= zone_high + margin  # wick desceu até a zona
        touched_resistance = h >= zone_low - margin  # wick subiu até a zona
        
        if ztype == "support" and not touched_support:
            continue
        elif ztype == "resistance" and not touched_resistance:
            continue
        elif ztype == "mixed" and not (touched_support or touched_resistance):
            continue
        
        # Registrar toque
        candles_ago = n - 1 - i
        touch_distances.append(candles_ago)
        if recent_touch_idx < 0 or candles_ago < recent_touch_idx:
            recent_touch_idx = candles_ago
        
        # Analisar QUALIDADE do bounce
        if ztype in ("support", "mixed") and l <= zone_high + margin:
            lower_wick = min(o, c) - l
            wick_ratio = lower_wick / full_range
            total_wick_ratios.append(wick_ratio)
            held = c > zone_price - margin  # fechou acima da zona
            
            if wick_ratio >= 0.35 and held:
                clean_bounces += 1
            
            # Failed breakout: preço penetrou abaixo da zona mas fechou acima
            if l < zone_low and c > zone_price:
                failed_breakouts += 1
        
        elif ztype in ("resistance", "mixed") and h >= zone_low - margin:
            upper_wick = h - max(o, c)
            wick_ratio = upper_wick / full_range
            total_wick_ratios.append(wick_ratio)
            held = c < zone_price + margin
            
            if wick_ratio >= 0.35 and held:
                clean_bounces += 1
            
            if h > zone_high and c < zone_price:
                failed_breakouts += 1
    
    # ── CALCULAR SCORE DE FORÇA ──
    strength = 0.30  # base
    details = []
    
    # 1. Toques: mais toques = mais forte (max +0.20)
    touch_bonus = min(0.20, (touches - 2) * 0.05)
    strength += touch_bonus
    details.append(f"{touches}t(+{touch_bonus:.2f})")
    
    # 2. Bounces limpos: mostra rejeição real (max +0.20)
    if clean_bounces >= 3:
        strength += 0.20
        details.append(f"clean={clean_bounces}(+0.20)")
    elif clean_bounces >= 2:
        strength += 0.12
        details.append(f"clean={clean_bounces}(+0.12)")
    elif clean_bounces >= 1:
        strength += 0.06
        details.append(f"clean={clean_bounces}(+0.06)")
    
    # 3. Recência: toque recente = zona ativa (max +0.15)
    recent_touch = recent_touch_idx >= 0 and recent_touch_idx <= 20
    if recent_touch_idx >= 0 and recent_touch_idx <= 5:
        strength += 0.15
        details.append(f"recente={recent_touch_idx}v(+0.15)")
    elif recent_touch_idx >= 0 and recent_touch_idx <= 15:
        strength += 0.10
        details.append(f"recente={recent_touch_idx}v(+0.10)")
    elif recent_touch_idx >= 0 and recent_touch_idx <= 30:
        strength += 0.05
        details.append(f"recente={recent_touch_idx}v(+0.05)")
    else:
        details.append(f"antigo={recent_touch_idx}v")
    
    # 4. Failed breakouts: zona resistiu ataque = muito forte (max +0.15)
    if failed_breakouts >= 2:
        strength += 0.15
        details.append(f"failed_bk={failed_breakouts}(+0.15)")
    elif failed_breakouts >= 1:
        strength += 0.08
        details.append(f"failed_bk={failed_breakouts}(+0.08)")
    
    # 5. Qualidade média de rejeição (wick ratio médio)
    avg_rejection = float(np.mean(total_wick_ratios)) if total_wick_ratios else 0.0
    if avg_rejection >= 0.45:
        strength += 0.05
        details.append(f"avg_rej={avg_rejection:.2f}(+0.05)")
    
    strength = max(0.0, min(1.0, strength))
    
    return {
        "zone_strength": round(strength, 3),
        "recent_touch": recent_touch,
        "clean_bounces": clean_bounces,
        "failed_breakouts": failed_breakouts,
        "avg_rejection": round(avg_rejection, 3),
        "details": ",".join(details),
    }


# ═══════════════════════════════════════════════════════════════
# 3c. MOMENTUM — Pressão direcional das últimas velas
# ═══════════════════════════════════════════════════════════════
def _check_momentum_pressure(df: pd.DataFrame, direction: str,
                             atr_val: float) -> Dict[str, Any]:
    """
    Analisa o MOMENTUM das últimas 5 velas para detectar pressão de breakout.
    
    Para PUT (resistência): Se últimas velas são fortemente bullish → perigo
    Para CALL (suporte): Se últimas velas são fortemente bearish → perigo
    
    O bot precisa ser DINÂMICO: esperar o melhor momento nas velas.
    Se há pressão forte contra a direção do trade, é breakout, não bounce.
    
    Retorna:
      momentum_ok: bool — momentum compatível com trade?
      penalty: float — penalidade ao score (0.0 a -0.20)
      contra_count: int — candles consecutivos contra o trade
      reason: str
    """
    if len(df) < 7:
        return {"momentum_ok": True, "penalty": 0.0, "contra_count": 0,
                "body_accel": 0.0, "total_move_atr": 0.0, "reason": "dados_insuf"}
    
    # Analisar 5 velas ANTES da última (a última é o candle de confirmação)
    last_5 = df.iloc[-6:-1]
    
    consecutive_contra = 0
    max_consecutive_contra = 0
    total_contra = 0
    contra_bodies = []
    aligned_bodies = []
    
    for _, row in last_5.iterrows():
        o = float(row["open"])
        c = float(row["close"])
        body = abs(c - o)
        is_bull = c > o
        is_bear = c < o
        
        if direction == "PUT":
            # Para PUT: velas bullish são CONTRA (pressão de alta)
            if is_bull:
                consecutive_contra += 1
                total_contra += 1
                contra_bodies.append(body)
            else:
                max_consecutive_contra = max(max_consecutive_contra, consecutive_contra)
                consecutive_contra = 0
                if is_bear:
                    aligned_bodies.append(body)
        else:
            # Para CALL: velas bearish são CONTRA (pressão de baixa)
            if is_bear:
                consecutive_contra += 1
                total_contra += 1
                contra_bodies.append(body)
            else:
                max_consecutive_contra = max(max_consecutive_contra, consecutive_contra)
                consecutive_contra = 0
                if is_bull:
                    aligned_bodies.append(body)
    
    max_consecutive_contra = max(max_consecutive_contra, consecutive_contra)
    
    # Aceleração dos bodies contra (estão crescendo?)
    body_accel = 0.0
    if len(contra_bodies) >= 2:
        body_accel = (contra_bodies[-1] - contra_bodies[0]) / max(atr_val, 1e-9)
    
    # Movimento total nas últimas 5 velas
    first_open = float(last_5.iloc[0]["open"])
    last_close = float(last_5.iloc[-1]["close"])
    total_move = (last_close - first_open) / max(atr_val, 1e-9)
    
    # Média do tamanho dos bodies contra em relação ao ATR
    avg_contra = sum(contra_bodies) / max(len(contra_bodies), 1) / max(atr_val, 1e-9)
    
    # ── Calcular penalidade ──
    penalty = 0.0
    reasons = []
    momentum_ok = True
    
    # 1. Velas consecutivas contra = momentum forte (MAIS RESTRITIVO)
    if max_consecutive_contra >= 3:
        penalty -= 0.15
        momentum_ok = False
        reasons.append(f"seq_{max_consecutive_contra}v_contra")
    elif max_consecutive_contra >= 2 and total_contra >= 3:
        penalty -= 0.10
        reasons.append(f"seq_{max_consecutive_contra}v_contra(total={total_contra})")
    
    # 2. Bodies acelerando na direção contra = building pressure
    if body_accel > 0.3:
        penalty -= 0.05
        reasons.append(f"accel({body_accel:.1f})")
        # Aceleração EXTREMA = mercado ficando cada vez mais volátil
        if body_accel > 0.7:
            momentum_ok = False
    
    # 3. Movimento total grande contra a direção do trade (MAIS RESTRITIVO)
    if direction == "PUT" and total_move > 0.8:
        penalty -= 0.07
        if total_move > 1.5:
            momentum_ok = False
        reasons.append(f"alta_{total_move:.1f}ATR")
    elif direction == "CALL" and total_move < -0.8:
        penalty -= 0.07
        if total_move < -1.5:
            momentum_ok = False
        reasons.append(f"baixa_{abs(total_move):.1f}ATR")
    
    # 4. Bodies contra muito grandes (volatilidade direcional)
    if avg_contra > 0.40:
        penalty -= 0.05
        # Bodies MUITO grandes = mercado imprevisível (choppy)
        if avg_contra > 0.55:
            momentum_ok = False
        reasons.append(f"bodies({avg_contra:.2f}ATR)")

    # 5. IMPULSE ARRIVAL: vela anterior ao toque com corpo MUITO grande
    # = preço chegou à zona com força extrema, risco alto de rompimento
    prev_candle_body = abs(float(last_5.iloc[-1]["close"]) - float(last_5.iloc[-1]["open"]))
    prev_body_atr = prev_candle_body / max(atr_val, 1e-9)
    if prev_body_atr >= 0.80:
        penalty -= 0.05
        reasons.append(f"impulse_arrival({prev_body_atr:.2f}ATR)")
        if prev_body_atr >= 1.2:
            momentum_ok = False

    # 6. CONTINUATION PATTERN (flag/pennant):
    # Move forte (4+ velas contra com bodies grandes) seguido de 1 vela pequena
    # = o move provavelmente VAI CONTINUAR, não é exaustão
    last_candle = df.iloc[-1]
    last_body = abs(float(last_candle["close"]) - float(last_candle["open"]))
    last_body_atr = last_body / max(atr_val, 1e-9)
    if total_contra >= 4 and contra_bodies:
        avg_contra_body = sum(contra_bodies) / len(contra_bodies)
        avg_contra_body_atr = avg_contra_body / max(atr_val, 1e-9)
        # Pausa pequena após move forte com bodies grandes = continuation flag
        if last_body_atr < 0.20 and avg_contra_body_atr > 0.40:
            penalty -= 0.07
            reasons.append(f"continuation_flag(pause+{total_contra}v_fortes)")
            momentum_ok = False

    # 7. BODIES PERSISTENTES: bodies contra grandes E não diminuindo
    # = momentum ainda forte, zona pode ser rompida
    if len(contra_bodies) >= 3 and avg_contra > 0.40:
        half = len(contra_bodies) // 2
        first_avg = sum(contra_bodies[:half]) / max(half, 1)
        second_avg = sum(contra_bodies[half:]) / max(len(contra_bodies) - half, 1)
        if first_avg > 0 and second_avg >= first_avg * 0.85:
            penalty -= 0.03
            reasons.append("bodies_persistentes")

    return {
        "momentum_ok": momentum_ok,
        "penalty": max(-0.20, penalty),
        "contra_count": max_consecutive_contra,
        "body_accel": round(body_accel, 3),
        "total_move_atr": round(total_move, 2),
        "prev_body_atr": round(prev_body_atr, 3),
        "continuation_flag": total_contra >= 3 and last_body_atr < 0.25,
        "reason": ",".join(reasons) if reasons else "ok",
    }


# ═══════════════════════════════════════════════════════════════
# 3d. CASCATA / WATERFALL — Movimento direcional forte sem pausa
# ═══════════════════════════════════════════════════════════════
CASCADE_MIN_CANDLES = 5     # Mínimo de velas para considerar cascata
CASCADE_MIN_ATR = 1.5       # Movimento mínimo em ATR para cascata
CASCADE_BODY_RATIO = 0.30   # Corpo mínimo médio das velas (in % of range)


def _detect_cascade(df: pd.DataFrame, direction: str,
                    atr_val: float) -> Dict[str, Any]:
    """
    LEITURA GRÁFICA REAL — Cascata / Waterfall:

    Quando o preço cai (ou sobe) 5+ velas consecutivas com corpos grandes,
    é um MOVIMENTO DIRECIONAL FORTE. Qualquer "suporte" ou "resistência"
    encontrado no final desse movimento NÃO é confiável — é apenas onde
    o preço PAUSOU momentaneamente.

    REGRA DE TRADER REAL:
      - Preço caiu 5+ velas com corpo forte → NÃO entrar CALL
      - Preço subiu 5+ velas com corpo forte → NÃO entrar PUT
      - Esperar CONSOLIDAÇÃO (3+ velas laterais) antes de considerar reversal

    Olha as últimas 15 velas (não apenas 5 como o momentum filter).
    """
    result = {
        "is_cascade": False,
        "cascade_dir": "NONE",     # "UP" ou "DOWN"
        "cascade_candles": 0,
        "cascade_move_atr": 0.0,
        "cascade_blocks": False,   # True se bloqueia a direção do trade
        "cascade_reason": "",
    }

    if len(df) < 10 or atr_val <= 0:
        return result

    lookback = min(15, len(df))
    recent = df.iloc[-lookback:]

    # Contar sequências de velas na mesma direção
    max_bull_seq = 0
    max_bear_seq = 0
    bull_seq = 0
    bear_seq = 0
    bull_bodies = []
    bear_bodies = []
    # Track best sequences
    best_bull_start = 0
    best_bull_end = 0
    best_bear_start = 0
    best_bear_end = 0
    curr_bull_start = 0
    curr_bear_start = 0

    for i in range(len(recent)):
        row = recent.iloc[i]
        o = float(row["open"])
        c = float(row["close"])
        h = float(row["high"])
        l = float(row["low"])
        body = abs(c - o)
        full = h - l if h > l else 0.0001

        if c > o:  # bullish
            bull_seq += 1
            bull_bodies.append(body / full)
            if bull_seq > max_bull_seq:
                max_bull_seq = bull_seq
                best_bull_start = curr_bull_start
                best_bull_end = i
            bear_seq = 0
            bear_bodies = []
            curr_bear_start = i + 1
        elif c < o:  # bearish
            bear_seq += 1
            bear_bodies.append(body / full)
            if bear_seq > max_bear_seq:
                max_bear_seq = bear_seq
                best_bear_start = curr_bear_start
                best_bear_end = i
            bull_seq = 0
            bull_bodies = []
            curr_bull_start = i + 1
        else:  # doji — não quebra a sequência, mas não confirma
            pass

    # Verificar cascata de BAIXA (preço caindo = bloqueia CALL)
    if max_bear_seq >= CASCADE_MIN_CANDLES:
        s = best_bear_start
        e = best_bear_end
        move = abs(float(recent.iloc[s]["open"]) - float(recent.iloc[e]["close"]))
        move_atr = move / atr_val
        if move_atr >= CASCADE_MIN_ATR:
            result["is_cascade"] = True
            result["cascade_dir"] = "DOWN"
            result["cascade_candles"] = max_bear_seq
            result["cascade_move_atr"] = round(move_atr, 2)
            if direction == "CALL":
                result["cascade_blocks"] = True
                result["cascade_reason"] = (
                    f"CASCATA_BAIXA({max_bear_seq}v,{move_atr:.1f}ATR)→CALL_BLOQUEADO"
                )
            else:
                result["cascade_reason"] = (
                    f"cascata_baixa({max_bear_seq}v,{move_atr:.1f}ATR)→PUT_ok"
                )
            return result

    # Verificar cascata de ALTA (preço subindo = bloqueia PUT)
    if max_bull_seq >= CASCADE_MIN_CANDLES:
        s = best_bull_start
        e = best_bull_end
        move = abs(float(recent.iloc[e]["close"]) - float(recent.iloc[s]["open"]))
        move_atr = move / atr_val
        if move_atr >= CASCADE_MIN_ATR:
            result["is_cascade"] = True
            result["cascade_dir"] = "UP"
            result["cascade_candles"] = max_bull_seq
            result["cascade_move_atr"] = round(move_atr, 2)
            if direction == "PUT":
                result["cascade_blocks"] = True
                result["cascade_reason"] = (
                    f"CASCATA_ALTA({max_bull_seq}v,{move_atr:.1f}ATR)→PUT_BLOQUEADO"
                )
            else:
                result["cascade_reason"] = (
                    f"cascata_alta({max_bull_seq}v,{move_atr:.1f}ATR)→CALL_ok"
                )
            return result

    # Cascata mais relaxada: 4 velas com movimento > 2.0 ATR
    # (velas com corpo grande = move impulsivo)
    for check_dir, seq_count, bodies_list in [
        ("DOWN", bear_seq, bear_bodies), ("UP", bull_seq, bull_bodies)
    ]:
        if seq_count >= 4 and bodies_list:
            avg_body = sum(bodies_list) / len(bodies_list)
            if avg_body >= 0.45:  # corpos grandes = move impulsivo
                recent_4 = df.iloc[-4:]
                move_4 = abs(
                    float(recent_4.iloc[-1]["close"]) -
                    float(recent_4.iloc[0]["open"])
                )
                move_4_atr = move_4 / atr_val
                if move_4_atr >= 2.0:
                    result["is_cascade"] = True
                    result["cascade_dir"] = check_dir
                    result["cascade_candles"] = seq_count
                    result["cascade_move_atr"] = round(move_4_atr, 2)
                    blocks_dir = "CALL" if check_dir == "DOWN" else "PUT"
                    if direction == blocks_dir:
                        result["cascade_blocks"] = True
                        result["cascade_reason"] = (
                            f"impulso_{check_dir.lower()}({seq_count}v,body={avg_body:.0%},"
                            f"{move_4_atr:.1f}ATR)→{blocks_dir}_BLOQUEADO"
                        )
                    return result

    return result


# ═══════════════════════════════════════════════════════════════
# 4. CONFIRMAÇÃO: o preço RESPEITOU a zona? (Otimizado para M1)
# ═══════════════════════════════════════════════════════════════
def _check_zone_respect(df: pd.DataFrame, zone: Dict, direction: str,
                        atr_val: float) -> Dict[str, Any]:
    """
    Verifica se o ÚLTIMO candle (que acabou de fechar) confirma respeito à zona.
    
    IMPORTANTE para M1 binário:
      - Só olha o ÚLTIMO candle (-1). Candle -2 = 2 minutos atrás = muito tarde.
      - Verifica ANTI-LATE: se o close já está longe da zona (bounce já aconteceu),
        a entrada seria tarde demais → não confirma.
    
    Para CALL (suporte):
      - Wick inferior tocou a zona (low <= zone_high)
      - Close ACIMA da zona (preço não rompeu)
      - Close não muito longe da zona (bounce não acabou) → ANTI-LATE
    
    Para PUT (resistência):
      - Wick superior tocou a zona (high >= zone_low)
      - Close ABAIXO da zona (preço não rompeu)
      - Close não muito longe da zona (bounce não acabou) → ANTI-LATE
    
    Retorna dict com:
      confirmed: bool
      reason: str (o que confirmou)
      quality: float 0-1
    """
    if len(df) < 3:
        return {"confirmed": False, "reason": "sem_dados", "quality": 0.0}

    margin = atr_val * 0.15  # margem pequena
    # ANTI-LATE: se o close se afastou mais que 0.7 ATR da zona, o bounce já acabou
    max_bounce_dist = atr_val * 0.70

    # SÓ verificar o ÚLTIMO candle (que acabou de fechar)
    row = df.iloc[-1]
    o = float(row["open"])
    h = float(row["high"])
    l = float(row["low"])
    c = float(row["close"])
    body = abs(c - o)
    full_range = h - l if h > l else 0.0001

    if direction == "CALL":
        # === SUPORTE: preço bateu embaixo e voltou ===
        touched = l <= zone["zone_high"] + margin
        held = c > zone["price"] - margin

        # ANTI-LATE: close já muito acima da zona? Bounce já aconteceu
        dist_from_zone = c - zone["price"]
        if dist_from_zone > max_bounce_dist:
            return {"confirmed": False, "reason": "bounce_ja_foi", "quality": 0.0}

        lower_wick = min(o, c) - l
        wick_ratio = lower_wick / full_range if full_range > 0 else 0
        is_bullish = c > o

        if touched and held:
            quality = 0.40
            reasons = []
            has_rejection = False  # precisa de sinal REAL de rejeição

            if wick_ratio >= 0.35:
                quality += 0.25
                reasons.append("wick_rejeicao")
                has_rejection = True
            if is_bullish:
                quality += 0.15
                reasons.append("candle_bull")
                has_rejection = True
                # corpo_forte só conta quando alinhado com direção
                if body >= atr_val * 0.25:
                    quality += 0.10
                    reasons.append("corpo_forte")
            elif body >= atr_val * 0.30:
                # Candle BEARISH com corpo forte em suporte = pressão de queda!
                quality -= 0.10
                reasons.append("corpo_contra")

            if not has_rejection:
                # Sem wick_rejeicao E sem candle alinhado = sem rejeição real
                return {
                    "confirmed": False,
                    "reason": "+".join(reasons) + ",sem_rejeicao" if reasons else "sem_rejeicao",
                    "quality": max(0.0, quality),
                }

            return {
                "confirmed": True,
                "reason": "+".join(reasons) if reasons else "toque_basico",
                "quality": min(quality, 1.0),
            }

        # APPROACH: preço está PERTO da zona mas wick não tocou exatamente
        # NÃO confirma automaticamente — approach sozinho é insuficiente
        approach_dist = abs(c - zone["price"])
        if approach_dist <= atr_val * 0.25 and held:
            if len(df) >= 3:
                prev_c = float(df.iloc[-2]["close"])
                approaching = prev_c > c  # preço caiu em direção ao suporte
                if approaching or is_bullish:
                    return {
                        "confirmed": False,
                        "reason": "approach_zone_sem_rejeicao",
                        "quality": 0.20,
                    }

    else:
        # === RESISTÊNCIA: preço bateu em cima e voltou ===
        touched = h >= zone["zone_low"] - margin
        held = c < zone["price"] + margin

        # ANTI-LATE: close já muito abaixo da zona? Bounce já aconteceu
        dist_from_zone = zone["price"] - c
        if dist_from_zone > max_bounce_dist:
            return {"confirmed": False, "reason": "bounce_ja_foi", "quality": 0.0}

        upper_wick = h - max(o, c)
        wick_ratio = upper_wick / full_range if full_range > 0 else 0
        is_bearish = c < o

        if touched and held:
            quality = 0.40
            reasons = []
            has_rejection = False  # precisa de sinal REAL de rejeição

            if wick_ratio >= 0.35:
                quality += 0.25
                reasons.append("wick_rejeicao")
                has_rejection = True
            if is_bearish:
                quality += 0.15
                reasons.append("candle_bear")
                has_rejection = True
                # corpo_forte só conta quando alinhado com direção
                if body >= atr_val * 0.25:
                    quality += 0.10
                    reasons.append("corpo_forte")
            elif body >= atr_val * 0.30:
                # Candle BULLISH com corpo forte em resistência = pressão de alta!
                quality -= 0.10
                reasons.append("corpo_contra")

            if not has_rejection:
                # Sem wick_rejeicao E sem candle alinhado = sem rejeição real
                return {
                    "confirmed": False,
                    "reason": "+".join(reasons) + ",sem_rejeicao" if reasons else "sem_rejeicao",
                    "quality": max(0.0, quality),
                }

            return {
                "confirmed": True,
                "reason": "+".join(reasons) if reasons else "toque_basico",
                "quality": min(quality, 1.0),
            }

        # APPROACH: preço perto da zona de resistência
        # NÃO confirma automaticamente — approach sozinho é insuficiente
        approach_dist = abs(c - zone["price"])
        if approach_dist <= atr_val * 0.25 and held:
            if len(df) >= 3:
                prev_c = float(df.iloc[-2]["close"])
                approaching = prev_c < c  # preço subiu em direção à resistência
                if approaching or is_bearish:
                    return {
                        "confirmed": False,
                        "reason": "approach_zone_sem_rejeicao",
                        "quality": 0.20,
                    }

    return {"confirmed": False, "reason": "sem_confirmacao", "quality": 0.0}


# ═══════════════════════════════════════════════════════════════
# 5. VERIFICAR ROMPIMENTO (breakout = NÃO operar)
# ═══════════════════════════════════════════════════════════════
def _check_breakout(df: pd.DataFrame, zone: Dict, direction: str,
                    atr_val: float) -> bool:
    """
    Se o preço ROMPEU a zona com força, não operar bounce.
    
    CALL + suporte rompido pra baixo = breakout (não faz CALL)
    PUT + resistência rompida pra cima = breakout (não faz PUT)
    
    V2 — mais agressivo:
      - Checa 5 candles (não apenas 2)
      - Margem menor (0.15 ATR em vez de 0.3)
      - Se 2+ dos últimos 5 candles fecharam além da zona = breakout
      - Se QUALQUER candle fechou muito além (>0.5 ATR) = breakout forte
    """
    if len(df) < 3:
        return False

    lookback = min(5, len(df))
    margin_soft = atr_val * 0.15    # margem suave: 2 candles bastam
    margin_hard = atr_val * 0.50    # margem forte: 1 candle basta

    count_beyond = 0

    for i in range(-lookback, 0):
        c = float(df.iloc[i]["close"])

        if direction == "CALL":
            if c < zone["zone_low"] - margin_hard:
                return True  # 1 candle muito além = breakout confirmado
            if c < zone["zone_low"] - margin_soft:
                count_beyond += 1
        else:
            if c > zone["zone_high"] + margin_hard:
                return True  # 1 candle muito além = breakout confirmado
            if c > zone["zone_high"] + margin_soft:
                count_beyond += 1

    # 2+ candles fechando além da zona (margem suave) = breakout
    if count_beyond >= 2:
        return True

    return False


# ═══════════════════════════════════════════════════════════════
# 5b2. ESTRUTURA LOWER-HIGH / HIGHER-LOW — Continuação de tendência
# ═══════════════════════════════════════════════════════════════
def _check_market_structure(df: pd.DataFrame, direction: str,
                            atr_val: float) -> Dict[str, Any]:
    """
    Detecta LOWER HIGHS (LH) e HIGHER LOWS (HL) nas últimas 15 velas.
    
    Lógica de trader real:
    - Se preço FAZ LOWER HIGHS no SUPORTE → continuação de BAIXA → NÃO entrar CALL
      (cada tentativa de subir fica mais fraca = vendedores controlam)
    - Se preço FAZ HIGHER LOWS na RESISTÊNCIA → continuação de ALTA → NÃO entrar PUT
      (cada tentativa de cair fica mais fraca = compradores controlam)
    
    Retorna:
      structure_danger: bool — estrutura perigosa para esta direção?
      lh_count: int — quantos lower highs detectados
      hl_count: int — quantos higher lows detectados
      reason: str — descrição da detecção
    """
    if len(df) < 15:
        return {"structure_danger": False, "lh_count": 0, "hl_count": 0,
                "reason": "dados_insuf"}

    recent = df.tail(15)
    highs = recent["high"].astype(float).values
    lows = recent["low"].astype(float).values

    # ── Detectar SWING HIGHS (picos locais) ──
    swing_highs = []
    for i in range(1, len(highs) - 1):
        if highs[i] > highs[i-1] and highs[i] >= highs[i+1]:
            swing_highs.append(highs[i])
    # Adicionar último high se não detectou suficiente
    if len(swing_highs) < 2:
        # Usar máximas em janelas de 5 velas
        for start in range(0, len(highs) - 4, 5):
            window = highs[start:start+5]
            swing_highs.append(float(max(window)))

    # ── Detectar SWING LOWS (vales locais) ──
    swing_lows = []
    for i in range(1, len(lows) - 1):
        if lows[i] < lows[i-1] and lows[i] <= lows[i+1]:
            swing_lows.append(lows[i])
    if len(swing_lows) < 2:
        for start in range(0, len(lows) - 4, 5):
            window = lows[start:start+5]
            swing_lows.append(float(min(window)))

    # ── Contar LOWER HIGHS (cada swing high menor que o anterior) ──
    lh_count = 0
    if len(swing_highs) >= 2:
        for i in range(1, len(swing_highs)):
            if swing_highs[i] < swing_highs[i-1] - atr_val * 0.1:
                lh_count += 1

    # ── Contar HIGHER LOWS (cada swing low maior que o anterior) ──
    hl_count = 0
    if len(swing_lows) >= 2:
        for i in range(1, len(swing_lows)):
            if swing_lows[i] > swing_lows[i-1] + atr_val * 0.1:
                hl_count += 1

    # ── Avaliar perigo ──
    structure_danger = False
    reason = "sem_padrao"

    # CALL no SUPORTE mas preço faz LOWER HIGHS = continuação de BAIXA
    if direction == "CALL" and lh_count >= 2:
        structure_danger = True
        reason = f"LH_no_suporte({lh_count}LH)"

    # PUT na RESISTÊNCIA mas preço faz HIGHER LOWS = continuação de ALTA
    if direction == "PUT" and hl_count >= 2:
        structure_danger = True
        reason = f"HL_na_resistencia({hl_count}HL)"

    return {
        "structure_danger": structure_danger,
        "lh_count": lh_count,
        "hl_count": hl_count,
        "reason": reason,
    }


# ═══════════════════════════════════════════════════════════════
# 5b. FILTRO DE TENDÊNCIA — Não operar contra a tendência
# ═══════════════════════════════════════════════════════════════
def _check_trend_filter(df: pd.DataFrame, direction: str,
                        atr_val: float) -> Dict[str, Any]:
    """
    Detecta tendência clara usando EMA(20) slope + direção net das últimas 10 velas.
    
    Se preço está em UPTREND → NÃO entrar PUT (resistência vai romper)
    Se preço está em DOWNTREND → NÃO entrar CALL (suporte vai romper)
    
    Retorna:
      trend_ok: bool — trade é compatível com a tendência?
      trend_dir: str — "UP", "DOWN", "LATERAL"
      slope_atr: float — inclinação da EMA em relação ao ATR
      penalty: float — penalidade no score
    """
    if len(df) < 25:
        return {"trend_ok": True, "trend_dir": "LATERAL", "slope_atr": 0.0,
                "penalty": 0.0, "reason": "dados_insuf"}

    closes = df["close"].astype(float).values

    # ── EMA(20) slope ──
    ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().values
    # Slope = variação da EMA nos últimos 5 candles normalizada pelo ATR
    ema_now = ema20[-1]
    ema_5ago = ema20[-6] if len(ema20) >= 6 else ema20[0]
    slope = (ema_now - ema_5ago) / max(atr_val, 1e-9)

    # ── Direção net das últimas 10 velas ──
    last_10 = df.iloc[-10:]
    bulls = sum(1 for _, r in last_10.iterrows()
                if float(r["close"]) > float(r["open"]))
    bears = 10 - bulls

    # ── Movimento total das últimas 10 velas em ATR ──
    open_10 = float(last_10.iloc[0]["open"])
    close_now = float(last_10.iloc[-1]["close"])
    net_move = (close_now - open_10) / max(atr_val, 1e-9)

    # ── Higher highs / Lower lows check (últimas 5 velas) ──
    last_5_closes = closes[-5:]
    ascending = sum(1 for i in range(1, len(last_5_closes))
                    if last_5_closes[i] > last_5_closes[i-1])
    descending = sum(1 for i in range(1, len(last_5_closes))
                     if last_5_closes[i] < last_5_closes[i-1])

    # ── Preço relativo à EMA: acima ou abaixo consistentemente? ──
    # Se as últimas 8 velas estão ABAIXO da EMA → vendedores dominam
    last_8_below_ema = sum(1 for i in range(-8, 0) if closes[i] < ema20[i])
    last_8_above_ema = sum(1 for i in range(-8, 0) if closes[i] > ema20[i])

    # ── Classificar tendência — SENSÍVEL: detectar tendência mais cedo ──
    trend_dir = "LATERAL"
    penalty = 0.0
    trend_ok = True

    # UPTREND: Qualquer combinação clara de alta
    is_up = False
    if slope > 0.15 and bulls >= 6 and net_move > 0.5:
        is_up = True
    elif slope > 0.25 and net_move > 0.6:
        is_up = True
    elif net_move > 1.0 and ascending >= 3:
        is_up = True  # Movimento forte + closes ascendentes
    elif slope > 0.15 and ascending >= 4:
        is_up = True  # EMA subindo + 4 de 5 closes subindo
    elif slope > 0.10 and last_8_above_ema >= 7 and net_move > 0.3:
        is_up = True  # Preço consistentemente acima da EMA + EMA subindo
    elif net_move > 0.8 and bulls >= 6:
        is_up = True  # Movimento forte com maioria de velas verdes

    if is_up:
        trend_dir = "UP"
        if direction == "PUT":
            trend_ok = False
            penalty = -0.10

    # DOWNTREND: Qualquer combinação clara de baixa
    is_down = False
    if not is_up:
        if slope < -0.15 and bears >= 6 and net_move < -0.5:
            is_down = True
        elif slope < -0.25 and net_move < -0.6:
            is_down = True
        elif net_move < -1.0 and descending >= 3:
            is_down = True
        elif slope < -0.15 and descending >= 4:
            is_down = True
        elif slope < -0.10 and last_8_below_ema >= 7 and net_move < -0.3:
            is_down = True  # Preço consistentemente abaixo da EMA + EMA caindo
        elif net_move < -0.8 and bears >= 6:
            is_down = True  # Movimento forte com maioria de velas vermelhas

    if is_down:
        trend_dir = "DOWN"
        if direction == "CALL":
            trend_ok = False
            penalty = -0.10

    reason = f"ema_slope={slope:.2f},bulls={bulls},net={net_move:.1f}ATR"
    return {
        "trend_ok": trend_ok,
        "trend_dir": trend_dir,
        "slope_atr": round(slope, 3),
        "penalty": penalty,
        "reason": reason,
    }


# ═══════════════════════════════════════════════════════════════
# 5c. DETECÇÃO DE CANAIS (ascending/descending channels)
# ═══════════════════════════════════════════════════════════════
def _detect_channel(df: pd.DataFrame, atr_val: float,
                    lookback: int = 40) -> Dict[str, Any]:
    """
    Detecta canais de preço usando regressão linear nos highs e lows.

    Canal ascendente: highs e lows ambos subindo → linha superior = resistência dinâmica
    Canal descendente: highs e lows ambos caindo → linha inferior = suporte dinâmico
    Canal lateral: sem inclinação significativa

    Verifica se o preço está PERTO da borda do canal (top ou bottom).
    """
    result = {
        "has_channel": False,
        "channel_type": "none",     # "ascending", "descending", "lateral"
        "at_upper": False,          # Preço na borda superior
        "at_lower": False,          # Preço na borda inferior
        "channel_width_atr": 0.0,   # Largura do canal em ATRs
        "upper_price": 0.0,         # Preço da borda superior projetada
        "lower_price": 0.0,         # Preço da borda inferior projetada
        "slope_upper": 0.0,         # Inclinação da borda superior
        "slope_lower": 0.0,         # Inclinação da borda inferior
        "channel_quality": 0.0,     # 0-1: quão bem definido é o canal
        "channel_dir_signal": "NEUTRAL",  # CALL se na borda inferior, PUT se na superior
        "channel_bonus": 0.0,
        "channel_reason": "",
    }

    if len(df) < lookback + 5:
        return result

    slc = df.iloc[-lookback:]
    highs = slc["high"].astype(float).values
    lows = slc["low"].astype(float).values
    closes = slc["close"].astype(float).values
    n = len(highs)
    x = np.arange(n, dtype=float)

    # Regressão linear nos highs (borda superior) e lows (borda inferior)
    try:
        # Upper channel line (highs)
        slope_h, intercept_h = np.polyfit(x, highs, 1)
        fitted_h = slope_h * x + intercept_h
        residuals_h = highs - fitted_h
        r2_h = 1.0 - (np.var(residuals_h) / max(np.var(highs), 1e-12))

        # Lower channel line (lows)
        slope_l, intercept_l = np.polyfit(x, lows, 1)
        fitted_l = slope_l * x + intercept_l
        residuals_l = lows - fitted_l
        r2_l = 1.0 - (np.var(residuals_l) / max(np.var(lows), 1e-12))
    except Exception:
        return result

    # Projeções para a posição atual (última vela)
    upper_now = slope_h * (n - 1) + intercept_h
    lower_now = slope_l * (n - 1) + intercept_l
    current_price = float(closes[-1])
    channel_width = upper_now - lower_now

    if channel_width <= 0 or atr_val <= 0:
        return result

    width_atr = channel_width / atr_val

    # Qualidade mínima do canal
    avg_r2 = (max(0, r2_h) + max(0, r2_l)) / 2.0
    if avg_r2 < 0.30 or width_atr < 0.5:
        return result  # Canal mal definido ou muito estreito

    result["has_channel"] = True
    result["upper_price"] = round(upper_now, 6)
    result["lower_price"] = round(lower_now, 6)
    result["channel_width_atr"] = round(width_atr, 2)
    result["slope_upper"] = round(slope_h / atr_val, 4)
    result["slope_lower"] = round(slope_l / atr_val, 4)
    result["channel_quality"] = round(avg_r2, 3)

    # Tipo de canal
    slope_threshold = 0.002 * atr_val  # Mínimo para considerar inclinação
    both_up = slope_h > slope_threshold and slope_l > slope_threshold
    both_down = slope_h < -slope_threshold and slope_l < -slope_threshold

    if both_up:
        result["channel_type"] = "ascending"
    elif both_down:
        result["channel_type"] = "descending"
    else:
        result["channel_type"] = "lateral"

    # Proximidade das bordas (margem = 15% da largura do canal)
    margin = channel_width * 0.15
    dist_upper = upper_now - current_price
    dist_lower = current_price - lower_now

    if dist_upper <= margin and dist_upper >= -margin * 0.5:
        # Preço PERTO da borda SUPERIOR → possível rejeição para baixo
        result["at_upper"] = True
        result["channel_dir_signal"] = "PUT"
        result["channel_bonus"] = 0.06 * min(1.0, avg_r2 / 0.60)
        result["channel_reason"] = f"topo_canal({result['channel_type']},q={avg_r2:.2f})"
    elif dist_lower <= margin and dist_lower >= -margin * 0.5:
        # Preço PERTO da borda INFERIOR → possível bounce para cima
        result["at_lower"] = True
        result["channel_dir_signal"] = "CALL"
        result["channel_bonus"] = 0.06 * min(1.0, avg_r2 / 0.60)
        result["channel_reason"] = f"fundo_canal({result['channel_type']},q={avg_r2:.2f})"

    return result


# ═══════════════════════════════════════════════════════════════
# 5d. WICK REJECTION — Pavio forte na ponta (rejeição extrema)
# ═══════════════════════════════════════════════════════════════
def _check_wick_rejection(df: pd.DataFrame, direction: str,
                          atr_val: float, zone: Dict = None,
                          channel: Dict = None) -> Dict[str, Any]:
    """
    Analisa se a última vela (ou penúltima) tem pavio forte na ponta,
    indicando REJEIÇÃO do preço em uma zona importante.

    Pavio na ponta = o mercado tentou ir mas foi rejeitado = sinal forte.

    Checks:
    1. Wick ratio (pavio / range total) → >40% = rejeção forte
    2. Body é pequeno vs wick → pin bar / hammer / shooting star
    3. Wick na direção certa (ex: wick inferior longo → CALL)
    4. Se wick está NA ZONA S/R ou na borda do canal → bônus extra
    """
    result = {
        "has_wick_rejection": False,
        "wick_ratio": 0.0,
        "wick_atr": 0.0,
        "body_ratio": 0.0,
        "wick_pattern": "none",     # "hammer", "shooting_star", "pin_bar", "doji_wick"
        "wick_at_zone": False,      # Wick tocou na zona S/R?
        "wick_at_channel": False,   # Wick tocou a borda do canal?
        "wick_bonus": 0.0,
        "wick_reason": "",
    }

    if len(df) < 3 or atr_val <= 0:
        return result

    # Analisar as 2 últimas velas
    for idx in [-1, -2]:
        candle = df.iloc[idx]
        o = float(candle["open"])
        h = float(candle["high"])
        l = float(candle["low"])
        c = float(candle["close"])
        full_range = h - l
        if full_range < atr_val * 0.05:
            continue  # Vela muito pequena, ignorar

        body = abs(c - o)
        body_ratio = body / full_range

        if direction == "CALL":
            # Para CALL → queremos wick INFERIOR longo (tentou cair mas voltou)
            wick = min(o, c) - l
            wick_tip = l
        else:
            # Para PUT → queremos wick SUPERIOR longo (tentou subir mas voltou)
            wick = h - max(o, c)
            wick_tip = h

        wick_ratio = wick / full_range
        wick_atr = wick / atr_val

        # Wick precisa ser significativo
        if wick_ratio < 0.35 or wick_atr < 0.15:
            continue

        result["wick_ratio"] = round(wick_ratio, 3)
        result["wick_atr"] = round(wick_atr, 3)
        result["body_ratio"] = round(body_ratio, 3)

        # Classificar padrão
        if body_ratio < 0.10:
            result["wick_pattern"] = "doji_wick"
        elif wick_ratio >= 0.60 and body_ratio < 0.25:
            result["wick_pattern"] = "pin_bar"
        elif direction == "CALL" and wick_ratio >= 0.40:
            result["wick_pattern"] = "hammer"
        elif direction == "PUT" and wick_ratio >= 0.40:
            result["wick_pattern"] = "shooting_star"

        # Verificar se o pavio TOCOU a zona S/R
        if zone:
            zone_high = float(zone.get("zone_high", 0))
            zone_low = float(zone.get("zone_low", 0))
            zone_margin = atr_val * 0.10

            if direction == "CALL" and zone_low > 0:
                # Pavio inferior tocou a zona de suporte?
                if wick_tip <= zone_high + zone_margin:
                    result["wick_at_zone"] = True
            elif direction == "PUT" and zone_high > 0:
                # Pavio superior tocou a zona de resistência?
                if wick_tip >= zone_low - zone_margin:
                    result["wick_at_zone"] = True

        # Verificar se o pavio tocou a borda do canal
        if channel and channel.get("has_channel"):
            ch_margin = atr_val * 0.10
            if direction == "CALL" and channel.get("at_lower"):
                if wick_tip <= channel["lower_price"] + ch_margin:
                    result["wick_at_channel"] = True
            elif direction == "PUT" and channel.get("at_upper"):
                if wick_tip >= channel["upper_price"] - ch_margin:
                    result["wick_at_channel"] = True

        # Calcular bônus
        bonus = 0.0
        reasons = []

        if wick_ratio >= 0.60:
            bonus += 0.08
            reasons.append(f"wick_forte({wick_ratio:.0%})")
        elif wick_ratio >= 0.45:
            bonus += 0.05
            reasons.append(f"wick_medio({wick_ratio:.0%})")
        else:
            bonus += 0.02
            reasons.append(f"wick({wick_ratio:.0%})")

        if result["wick_at_zone"]:
            bonus += 0.04
            reasons.append("wick_na_zona")

        if result["wick_at_channel"]:
            bonus += 0.04
            reasons.append("wick_no_canal")

        if result["wick_pattern"] in ("pin_bar", "doji_wick"):
            bonus += 0.03
            reasons.append(result["wick_pattern"])

        result["has_wick_rejection"] = True
        result["wick_bonus"] = round(bonus, 3)
        result["wick_reason"] = "+".join(reasons)
        break  # Usar a primeira vela com wick significativo

    return result


# ═══════════════════════════════════════════════════════════════
# 5b. MULTI-TIMEFRAME — Confirmar zona M1 no M5 (trader real)
# ═══════════════════════════════════════════════════════════════
def check_m5_confirmation(df_m5: pd.DataFrame, zone_price: float,
                          zone_dir: str, atr_m5: float) -> Dict[str, Any]:
    """
    Trader experiente SEMPRE olha timeframe maior antes de entrar.
    Verifica se a zona de S/R do M1 também é relevante no M5.
    
    Retorna:
      m5_zone_exists: bool  — Existe zona S/R no M5 na mesma região?
      m5_trend_aligned: bool — A tendência M5 está a favor do trade?
      m5_rejection: bool — O M5 mostra rejeição recente na zona?
      m5_confluence: int — Pontos de confluência M5 (0-3)
    """
    result = {
        "m5_zone_exists": False,
        "m5_trend_aligned": False,
        "m5_trend_contra": False,
        "m5_trend_dir": "LATERAL",
        "m5_rejection": False,
        "m5_confluence": 0,
        "m5_reason": "sem_dados",
    }
    
    if df_m5 is None or len(df_m5) < 30:
        return result
    
    # Detectar zonas S/R no M5
    zones_m5 = detect_sr_zones(df_m5, atr_m5)
    
    # 1. Existe zona M5 na mesma região? (dentro de 1.5 ATR_m5)
    for z in zones_m5:
        dist = abs(z["price"] - zone_price) / atr_m5 if atr_m5 > 0 else 99
        if dist <= 1.5:
            result["m5_zone_exists"] = True
            result["m5_confluence"] += 1
            break
    
    # 2. Tendência M5 (SMA 20 vs SMA 50 simples)
    closes_m5 = df_m5["close"].astype(float).values
    if len(closes_m5) >= 50:
        sma20 = float(np.mean(closes_m5[-20:]))
        sma50 = float(np.mean(closes_m5[-50:]))
        m5_trend = "UP" if sma20 > sma50 else "DOWN"
        result["m5_trend_dir"] = m5_trend
        if (zone_dir == "CALL" and m5_trend == "UP") or \
           (zone_dir == "PUT" and m5_trend == "DOWN"):
            result["m5_trend_aligned"] = True
            result["m5_confluence"] += 1
        elif (zone_dir == "CALL" and m5_trend == "DOWN") or \
             (zone_dir == "PUT" and m5_trend == "UP"):
            result["m5_trend_contra"] = True
            result["m5_confluence"] -= 1  # Penalidade: M5 contra
    
    # 3. Rejeição recente no M5 (últimas 3 velas M5 = últimos 15 min)
    for i in range(-3, 0):
        if abs(i) > len(df_m5):
            continue
        row = df_m5.iloc[i]
        h, l, o, c = float(row["high"]), float(row["low"]), float(row["open"]), float(row["close"])
        full = h - l if h > l else 0.0001
        
        if zone_dir == "CALL":
            # Suporte: wick inferior longo + close acima da zona
            lower_wick = min(o, c) - l
            if lower_wick / full >= 0.35 and abs(l - zone_price) / atr_m5 <= 1.0:
                result["m5_rejection"] = True
                result["m5_confluence"] += 1
                break
        else:
            # Resistência: wick superior longo + close abaixo da zona
            upper_wick = h - max(o, c)
            if upper_wick / full >= 0.35 and abs(h - zone_price) / atr_m5 <= 1.0:
                result["m5_rejection"] = True
                result["m5_confluence"] += 1
                break
    
    parts = []
    if result["m5_zone_exists"]:
        parts.append("zona_m5")
    if result["m5_trend_aligned"]:
        parts.append("trend_m5")
    if result["m5_rejection"]:
        parts.append("rejeicao_m5")
    result["m5_reason"] = "+".join(parts) if parts else "sem_conf_m5"
    
    return result


# ═══════════════════════════════════════════════════════════════
# 5b2. MULTI-TIMEFRAME M15 — S/R de alta relevância
# ═══════════════════════════════════════════════════════════════
def check_m15_confirmation(df_m15: pd.DataFrame, zone_price: float,
                           zone_dir: str, atr_m15: float) -> Dict[str, Any]:
    """
    S/R no M15 é MUITO mais forte que M5.
    Um trader profissional sempre olha M15 para confirmar zonas de M1.
    Zona que existe no M1 E no M15 = zona institucional.

    Retorna:
      m15_zone_exists: bool  — Existe zona S/R no M15 na mesma região?
      m15_trend_aligned: bool — A tendência M15 está a favor?
      m15_rejection: bool — M15 mostra rejeição recente na zona?
      m15_confluence: int — Pontos de confluência M15 (0-3)
    """
    result = {
        "m15_zone_exists": False,
        "m15_trend_aligned": False,
        "m15_rejection": False,
        "m15_confluence": 0,
        "m15_reason": "sem_dados",
    }

    if df_m15 is None or len(df_m15) < 15:
        return result

    # Detectar zonas S/R no M15
    zones_m15 = detect_sr_zones(df_m15, atr_m15)

    # 1. Existe zona M15 na mesma região? (dentro de 2.0 ATR_m15)
    for z in zones_m15:
        dist = abs(z["price"] - zone_price) / atr_m15 if atr_m15 > 0 else 99
        if dist <= 2.0:
            result["m15_zone_exists"] = True
            result["m15_confluence"] += 1
            break

    # 2. Tendência M15 (SMA 10 vs SMA 30)
    closes_m15 = df_m15["close"].astype(float).values
    if len(closes_m15) >= 30:
        sma10 = float(np.mean(closes_m15[-10:]))
        sma30 = float(np.mean(closes_m15[-30:]))
        m15_trend = "UP" if sma10 > sma30 else "DOWN"
        if (zone_dir == "CALL" and m15_trend == "UP") or \
           (zone_dir == "PUT" and m15_trend == "DOWN"):
            result["m15_trend_aligned"] = True
            result["m15_confluence"] += 1

    # 3. Rejeição recente no M15 (últimas 2 velas M15 = últimos 30 min)
    for i in range(-2, 0):
        if abs(i) > len(df_m15):
            continue
        row = df_m15.iloc[i]
        h, l, o, c = float(row["high"]), float(row["low"]), float(row["open"]), float(row["close"])
        full = h - l if h > l else 0.0001

        if zone_dir == "CALL":
            lower_wick = min(o, c) - l
            if lower_wick / full >= 0.35 and abs(l - zone_price) / atr_m15 <= 1.5:
                result["m15_rejection"] = True
                result["m15_confluence"] += 1
                break
        else:
            upper_wick = h - max(o, c)
            if upper_wick / full >= 0.35 and abs(h - zone_price) / atr_m15 <= 1.5:
                result["m15_rejection"] = True
                result["m15_confluence"] += 1
                break

    parts = []
    if result["m15_zone_exists"]:
        parts.append("zona_m15")
    if result["m15_trend_aligned"]:
        parts.append("trend_m15")
    if result["m15_rejection"]:
        parts.append("rejeicao_m15")
    result["m15_reason"] = "+".join(parts) if parts else "sem_conf_m15"

    return result


# ═══════════════════════════════════════════════════════════════
# 5c. ESPAÇO ATÉ PRÓXIMO S/R — Tem room para lucrar?
# ═══════════════════════════════════════════════════════════════
def check_space_to_profit(zones: List[Dict], current_price: float,
                          zone_dir: str, atr_val: float) -> Dict[str, Any]:
    """
    Trader experiente mede: "Se eu entrar CALL aqui, qual a próxima
    resistência? Tem espaço suficiente para o preço ir?"
    
    Se a próxima zona oposta está a < 0.5 ATR → sem espaço → NÃO ENTRA.
    """
    min_space_atr = 0.5  # Mínimo de espaço necessário (em ATR)
    
    next_opposing_dist = float("inf")
    next_opposing_price = None
    
    for z in zones:
        zp = z["price"]
        if zone_dir == "CALL":
            # Procurar próxima RESISTÊNCIA acima do preço
            if zp > current_price:
                dist = (zp - current_price) / atr_val if atr_val > 0 else 99
                if dist < next_opposing_dist:
                    next_opposing_dist = dist
                    next_opposing_price = zp
        else:
            # Procurar próximo SUPORTE abaixo do preço
            if zp < current_price:
                dist = (current_price - zp) / atr_val if atr_val > 0 else 99
                if dist < next_opposing_dist:
                    next_opposing_dist = dist
                    next_opposing_price = zp
    
    has_space = next_opposing_dist >= min_space_atr
    
    return {
        "has_space": has_space,
        "space_atr": round(next_opposing_dist, 2),
        "next_sr_price": next_opposing_price,
        "space_reason": f"espaco={next_opposing_dist:.1f}ATR" if has_space 
                        else f"sem_espaco({next_opposing_dist:.1f}ATR)",
    }


# ═══════════════════════════════════════════════════════════════
# 6. SINAL PRINCIPAL — simple_sr_signal()
# ═══════════════════════════════════════════════════════════════
def simple_sr_signal(df_m1: pd.DataFrame, atr_val: float,
                     df_m5: pd.DataFrame = None, atr_m5: float = None,
                     df_m15: pd.DataFrame = None, atr_m15: float = None) -> Dict[str, Any]:
    """
    Função principal: analisa S/R e retorna sinal.
    
    Multi-timeframe: usa M1 para entrada, M5 e M15 para confirmar zonas.
    Zonas que existem em M1 + M5 + M15 = zonas institucionais muito fortes.
    
    Fluxo:
      1. Detectar zonas S/R
      2. Achar zona mais próxima do preço
      3. Determinar direção (suporte=CALL, resistência=PUT)
      4. Verificar se NÃO rompeu (breakout)
      5. Verificar CONFIRMAÇÃO (preço respeitou a zona)
      6. Se confirmou → trade=True
    
    Retorna dict compatível com WS_AUTO_AI_BULLEX.py
    """
    # Retorno neutro (sem sinal)
    neutral = {
        "trade": False, "dir": "NEUTRAL", "score": 0.0,
        "reasons": [], "sr_touches": 0, "sr_proximity": 0.0,
        "sr_weight": 0.0, "sr_strength": 0.0, "sr_reason": "",
        "sr_bonus": 0.0, "sr_rejections": 0, "inside_zone": False,
        "market_quality": 0.50, "context": 0.50, "entry_confidence": 0.0,
        "confluence_count": 0, "confluence_bonus": 0.0,
        "candle_pattern": "none", "candle_strength": 0.0, "candle_body_ratio": 0.0,
        "approach_force": 0.0, "approach_candles": 0, "approach_atr": 0.0,
        "is_impulse": False, "bounce_quality": 0.0, "bounce_confirmed": False,
        # Campos extras que o engine lê (defaults seguros)
        "retr": 0.0, "A_atr": 0.0, "effA": 0.0, "flips": 0.0,
        "pb_len": 0, "distBreak": 0.0, "late_ext": 0.0,
        "compression": 0.0, "ctx": "neutro",
        "has_lt": False, "lt_points": 0, "lt_confluence": 0.0,
    }

    if len(df_m1) < MIN_CANDLES:
        nr = neutral.copy()
        nr["reasons"] = ["velas_insuficientes"]
        return nr

    closes = df_m1["close"].astype(float).values
    current_price = float(closes[-1])

    # ── 1. DETECTAR ZONAS ──
    zones = detect_sr_zones(df_m1, atr_val)
    if not zones:
        nr = neutral.copy()
        nr["reasons"] = ["sem_zonas_sr"]
        return nr

    # ── 2. ACHAR ZONA MAIS PRÓXIMA ──
    # REGRA FUNDAMENTAL: preço deve estar no TOPO ou FUNDO do range.
    # NUNCA entrar no MEIO — isso é o que um trader humano faz.
    #
    # Detectar range: máxima e mínima das últimas N velas.
    # CALL só se preço está na parte INFERIOR do range (< 30%)
    # PUT só se preço está na parte SUPERIOR do range (> 70%)

    # ── Detectar extremos do range recente ──
    range_window = min(RANGE_WINDOW, len(df_m1))
    recent_highs = df_m1["high"].astype(float).values[-range_window:]
    recent_lows = df_m1["low"].astype(float).values[-range_window:]
    range_high = float(np.max(recent_highs))
    range_low = float(np.min(recent_lows))
    range_size = range_high - range_low if range_high > range_low else atr_val
    
    # Posição do preço no range: 0.0 = fundo, 1.0 = topo
    range_position = (current_price - range_low) / range_size if range_size > 0 else 0.5
    range_position = max(0.0, min(1.0, range_position))  # clipar

    active_zone = None
    zone_dir = None
    best_dist = float("inf")

    for z in zones:
        dist = abs(current_price - z["price"])
        max_dist = SR_MAX_DIST * atr_val
        if dist > max_dist:
            continue

        # Direção baseada na POSIÇÃO do preço vs zona (não tipo histórico)
        if z["price"] < current_price:
            candidate_dir = "CALL"   # zona ABAIXO do preço = suporte → esperar bounce UP
        elif z["price"] > current_price:
            candidate_dir = "PUT"    # zona ACIMA do preço = resistência → esperar rejeição DOWN
        else:
            # Preço exatamente na zona — usar tipo histórico como desempate
            ztype = z.get("type", "mixed")
            if ztype == "support":
                candidate_dir = "CALL"
            elif ztype == "resistance":
                candidate_dir = "PUT"
            else:
                continue

        # ══ FILTRO TOPO/FUNDO ESTRITO ══
        # SÓ opera no FUNDO do range (CALL) ou TOPO do range (PUT)
        # Nada no meio — sem pullback, sem throwback
        # A IA (LSTM) decide se o sinal será respeitado
        if candidate_dir == "CALL" and range_position > 0.30:
            continue  # Preço NÃO está no fundo absoluto → NÃO entrar
        if candidate_dir == "PUT" and range_position < 0.70:
            continue  # Preço NÃO está no topo absoluto → NÃO entrar

        # ══ FILTRO DISTÂNCIA ZERO: preco DEVE estar na zona ══
        # Só entra se o preco está TOCANDO o S/R (max 0.30 ATR)
        dist_in_atr = dist / atr_val if atr_val > 0 else 99
        if dist_in_atr > 0.30:
            continue  # Preço longe da zona → não entrar

        # Selecionar a zona MAIS PRÓXIMA do preço atual
        if dist < best_dist:
            best_dist = dist
            active_zone = z
            zone_dir = candidate_dir
            active_zone["distance"] = dist
            active_zone["distance_atr"] = dist_in_atr
            active_zone["range_position"] = range_position

    if active_zone is None:
        nr = neutral.copy()
        best = zones[0]
        best_dist = abs(current_price - best["price"]) / atr_val if atr_val > 0 else 99
        nr["sr_touches"] = best["touches"]
        nr["reasons"] = [f"zona_longe(melhor={best['touches']}t,d={best_dist:.1f}ATR)"]
        return nr

    # ── 2b. DETECTAR ZONAS AGRUPADAS (CLUSTER) ──
    # Se existem 2+ zonas S/R próximas uma da outra, isso forma uma
    # "mega zona" muito mais forte — o preço tende a respeitar mais.
    nearby_zones_count = 0
    nearby_zones_touches = 0
    cluster_bonus = 0.0
    active_price = active_zone["price"]
    cluster_dist = 1.5 * atr_val  # zona dentro de 1.5 ATR = cluster
    for z in zones:
        if z is active_zone:
            continue
        if abs(z["price"] - active_price) <= cluster_dist:
            nearby_zones_count += 1
            nearby_zones_touches += z["touches"]
    if nearby_zones_count >= 2:
        cluster_bonus = 0.15  # 3+ zonas agrupadas = mega zona
    elif nearby_zones_count == 1:
        cluster_bonus = 0.10  # 2 zonas próximas = zona forte

    # ── 3. OBSERVAÇÕES (dados para o raciocínio — NÃO bloqueiam) ──
    # O cérebro inteligente decide. Aqui apenas coletamos informações.
    breakout_detected = _check_breakout(df_m1, active_zone, zone_dir, atr_val)

    # ── 3b. TREND — observação, não bloqueio ──
    trend_ctx = _check_trend_filter(df_m1, zone_dir, atr_val)

    # ── 3c. ESTRUTURA DE MERCADO: Lower Highs / Higher Lows ──
    structure_ctx = _check_market_structure(df_m1, zone_dir, atr_val)

    # ── 4. CONFIRMAÇÃO: preço respeitou a zona? ──
    confirm = _check_zone_respect(df_m1, active_zone, zone_dir, atr_val)

    # ── 4b. FORÇA DA ZONA (para a IA) ──
    zs = calculate_zone_strength(df_m1, active_zone, atr_val)
    zone_str = float(zs["zone_strength"])

    # ── 4c. MOMENTUM: pressão das últimas velas ──
    momentum_ctx = _check_momentum_pressure(df_m1, zone_dir, atr_val)
    momentum_penalty = float(momentum_ctx["penalty"])

    # ── 4c2. CASCATA: movimento direcional forte sem pausa ──
    cascade_ctx = _detect_cascade(df_m1, zone_dir, atr_val)

    # ── 4d. CANAL: detectar canal de preço ──
    channel_ctx = _detect_channel(df_m1, atr_val)

    # ── 4e. WICK REJECTION: pavio forte na ponta (rejeição) ──
    wick_ctx = _check_wick_rejection(df_m1, zone_dir, atr_val,
                                      zone=active_zone, channel=channel_ctx)

    # ── 4f. MULTI-TIMEFRAME M5: confirmação no timeframe maior ──
    m5_ctx = check_m5_confirmation(
        df_m5, active_zone["price"], zone_dir,
        atr_m5 if atr_m5 else atr_val * 2.0  # fallback: ATR M5 ≈ 2x ATR M1
    )

    # ── 4f2. MULTI-TIMEFRAME M15: S/R de alta relevância ──
    m15_ctx = check_m15_confirmation(
        df_m15, active_zone["price"], zone_dir,
        atr_m15 if atr_m15 else atr_val * 3.5  # fallback: ATR M15 ≈ 3.5x ATR M1
    )

    # ── 4g. ESPAÇO ATÉ PRÓXIMO S/R: tem room para lucrar? ──
    space_ctx = check_space_to_profit(zones, current_price, zone_dir, atr_val)

    # ── 4h. CANDLE COLOR AI: padrões de velas confirmam a direção? ──
    candle_ai = {"confirmed": False, "probability": 50.0, "reason": "desativado"}
    if predict_candle_color is not None:
        try:
            candle_ai = predict_candle_color(df_m1, zone_dir, atr_val)
        except Exception as _e:
            log.warning(f"[CandleAI] Erro: {_e}")

    # ── 5. SCORE + CONFLUÊNCIA — COMO UM TRADER REAL PENSA ──
    # Um trader profissional de S/R NÃO entra só porque viu um candle bonito.
    # Ele CONTA CONFLUÊNCIAS: quantos fatores INDEPENDENTES confirmam?
    #
    # Confluências possíveis (cada uma = +1 ponto):
    #   1. Zona FORTE (strength >= 0.65, muitos toques limpos)
    #   2. Preço PERTO da zona (< 0.5 ATR)
    #   3. Rejeição confirmada (wick/candle na zona)
    #   4. Tendência a favor (não contra)
    #   5. Cluster S/R (2+ zonas agrupadas = mega zona)
    #   6. Canal definido com preço na borda
    #   7. Wick rejection forte (pavio longo na zona)
    #   8. SEM breakout (zona íntegra)
    #   9. Momentum NÃO agressivo contra
    #
    # REGRA: mínimo 3 confluências para gerar sinal.
    # Trader real: "Se não tem confluência, NÃO entra."

    dist_atr = active_zone["distance_atr"]
    touches = active_zone["touches"]

    # ── CONTAR CONFLUÊNCIAS ──
    confluence_points = 0
    confluence_details = []

    # 1. ZONA (qualquer zona real com toques)
    if zone_str >= 0.65 and touches >= 4:
        confluence_points += 2  # Zona MUITO forte = vale dobro
        confluence_details.append(f"zona_forte({touches}t,{zone_str:.0%})")
    elif zone_str >= 0.50 and touches >= 3:
        confluence_points += 1
        confluence_details.append(f"zona_ok({touches}t,{zone_str:.0%})")
    elif touches >= 2:
        confluence_points += 1  # Zona com 2 toques = ainda é zona real
        confluence_details.append(f"zona_nova({touches}t,{zone_str:.0%})")

    # 2. PREÇO PERTO DA ZONA (tocando ou quase)
    if dist_atr < 0.15:
        confluence_points += 2  # Preço DENTRO da zona = muito forte
        confluence_details.append(f"dentro({dist_atr:.2f}ATR)")
    elif dist_atr < 0.30:
        confluence_points += 1
        confluence_details.append(f"perto({dist_atr:.2f}ATR)")
    elif dist_atr < 0.50:
        confluence_points += 1
        confluence_details.append(f"proximo({dist_atr:.2f}ATR)")

    # 3. REJEIÇÃO CONFIRMADA (candle mostrou que zona segurou)
    if confirm["confirmed"] and confirm["quality"] >= 0.50:
        confluence_points += 1
        confluence_details.append(f"rejeicao({confirm['reason']})")

    # 4. TENDÊNCIA A FAVOR (ou neutra)
    if trend_ctx["trend_ok"]:
        confluence_points += 1
        confluence_details.append(f"trend_ok({trend_ctx['trend_dir']})")

    # 5. CLUSTER S/R (mega zona — múltiplas zonas agrupadas)
    if nearby_zones_count >= 1:
        confluence_points += 1
        confluence_details.append(f"cluster({nearby_zones_count+1}zonas)")

    # 6. CANAL (preço na borda do canal alinhado)
    if channel_ctx.get("has_channel"):
        ch_dir = channel_ctx.get("channel_dir_signal", "NEUTRAL")
        if ch_dir == zone_dir:
            confluence_points += 1
            confluence_details.append(f"canal({channel_ctx['channel_type']})")

    # 7. WICK REJECTION (pavio forte = rejeição profunda)
    if wick_ctx.get("has_wick_rejection") and float(wick_ctx.get("wick_bonus", 0)) > 0:
        confluence_points += 1
        confluence_details.append(f"wick({wick_ctx['wick_reason']})")

    # 8. MULTI-TIMEFRAME M5 (zona confirmada no M5 = muito forte)
    if m5_ctx["m5_zone_exists"]:
        confluence_points += 1
        confluence_details.append(f"m5_zona")
    if m5_ctx["m5_trend_aligned"]:
        confluence_points += 1
        confluence_details.append(f"m5_trend")
    if m5_ctx["m5_trend_contra"]:
        confluence_points -= 2  # M5 tendência CONTRA = penalidade FORTE
        confluence_details.append(f"m5_trend_CONTRA({m5_ctx['m5_trend_dir']})")
    if m5_ctx["m5_rejection"]:
        confluence_points += 1
        confluence_details.append(f"m5_rejeicao")

    # 8b. MULTI-TIMEFRAME M15 (zona M15 = INSTITUCIONAL, peso dobrado)
    if m15_ctx["m15_zone_exists"]:
        confluence_points += 2  # M15 zona vale DOBRO (S/R de alta relevância)
        confluence_details.append(f"M15_ZONA")
    if m15_ctx["m15_trend_aligned"]:
        confluence_points += 1
        confluence_details.append(f"m15_trend")
    if m15_ctx["m15_rejection"]:
        confluence_points += 1
        confluence_details.append(f"m15_rejeicao")

    # 9. ESPAÇO PARA LUCRO (próximo S/R oposto distância suficiente)
    if space_ctx["has_space"]:
        confluence_points += 1
        confluence_details.append(f"espaco({space_ctx['space_atr']}ATR)")

    # 10. CANDLE COLOR AI (padrões de velas confirmam a direção)
    if candle_ai["confirmed"] and candle_ai["probability"] >= 58.0:
        confluence_points += 1
        confluence_details.append(f"candle_ai({candle_ai['probability']:.0f}%,{candle_ai.get('pattern_name','')})")

    # ── SEM PENALIDADES — A IA LSTM DECIDE ──
    # Todas as informações (momentum, cascata, estrutura, etc.) são
    # passadas no setup para o Brain/LSTM usar como CONTEXTO.
    # Mas NÃO reduzem confluência nem bloqueiam.
    # A decisão de entrar ou não é 100% da IA baseada nas velas.
    penalties = []  # vazio — só para compatibilidade do log

    # Observações (informativas, sem penalidade)
    if breakout_detected:
        penalties.append("breakout(obs)")
    if not trend_ctx["trend_ok"]:
        penalties.append(f"trend({trend_ctx['trend_dir']},obs)")
    if cascade_ctx["cascade_blocks"]:
        penalties.append(f"cascata({cascade_ctx['cascade_reason']},obs)")
    if structure_ctx["structure_danger"]:
        penalties.append(f"estrutura({structure_ctx['reason']},obs)")

    # ── CALCULAR SCORE (simples — zona + proximidade) ──
    score = 0.30  # base
    reasons = []

    # Cada confluência adiciona ao score
    conf_score = confluence_points * 0.10
    score += conf_score
    reasons.append(f"confluencia={confluence_points}({'+'.join(confluence_details)})")
    if penalties:
        reasons.append(f"obs({','.join(penalties)})")

    # Bônus por zona forte
    if zone_str >= 0.80:
        score += 0.08
    elif zone_str >= 0.70:
        score += 0.05
    if touches >= 5:
        score += 0.05
    if dist_atr < 0.15:
        score += 0.05

    score = max(0.0, min(1.0, score))

    # ── 6. DECISÃO — ULTRA SIMPLES ──
    # Preço chegou na zona S/R (topo ou fundo)? → trade = True
    # A IA LSTM Candle Brain toma a decisão FINAL de entrar ou não.
    # Aqui só valida que existe uma zona real (2+ toques).

    if touches < 2:
        trade = False
        reasons.append("poucos_toques")
    else:
        trade = True  # Zona real com toques → IA LSTM decide

    if not trade:
        reasons.append("NO_TRADE")

    # ── 7. MARKET QUALITY (inclui força da zona) ──
    mkt_q = 0.50
    if touches >= 4:
        mkt_q += 0.10
    if touches >= 6:
        mkt_q += 0.05
    if confirm["confirmed"]:
        mkt_q += 0.15
    if dist_atr < 0.20:
        mkt_q += 0.08
    # Zona forte = contexto melhor
    if zone_str >= 0.65:
        mkt_q += 0.10
    elif zone_str >= 0.50:
        mkt_q += 0.05
    # Canal bem definido com wick = contexto excelente
    if channel_ctx.get("has_channel") and wick_ctx.get("has_wick_rejection"):
        mkt_q += 0.12
    elif channel_ctx.get("has_channel"):
        mkt_q += 0.05
    elif wick_ctx.get("has_wick_rejection"):
        mkt_q += 0.06
    mkt_q = min(0.95, mkt_q)

    # ── 8. PREENCHER RESULTADO ──
    touch_bonus = min(0.20, max(0, (touches - 2)) * 0.05)
    sr_weight = float(touches) + (1.0 if confirm["confirmed"] else 0.0)
    confluence = (1 + int(confirm["confirmed"]) + int(dist_atr < 0.20)
                  + int(zone_str >= 0.60)
                  + int(channel_ctx.get("has_channel", False))
                  + int(wick_ctx.get("has_wick_rejection", False)))

    result = neutral.copy()
    result.update({
        "trade": trade,
        "dir": zone_dir,
        "score": score,
        "reasons": reasons,
        "sr_touches": touches,
        "sr_proximity": max(0.0, 1.0 - dist_atr),
        "sr_weight": sr_weight,
        "sr_strength": min(1.0, touches / 7.0),
        "sr_reason": f"{touches}t_{active_zone.get('type', 'mixed')}",
        "sr_bonus": touch_bonus,
        "sr_rejections": 1 if confirm["confirmed"] else 0,
        "inside_zone": dist_atr < 0.15,
        "candle_strength": confirm["quality"],
        "bounce_quality": confirm["quality"],
        "bounce_confirmed": confirm["confirmed"],
        "market_quality": mkt_q,
        "context": mkt_q,
        "entry_confidence": min(score, mkt_q),
        "confluence_count": confluence,
        "confluence_bonus": 0.05 if confluence >= 2 else 0.0,
        # NOVO: Força da zona para a IA
        "zone_strength": zone_str,
        "zone_clean_bounces": zs["clean_bounces"],
        "zone_failed_breakouts": zs["failed_breakouts"],
        "zone_recent_touch": zs["recent_touch"],
        "zone_avg_rejection": zs["avg_rejection"],
        "zone_details": zs["details"],
        # MOMENTUM
        "momentum_ok": momentum_ctx["momentum_ok"],
        "momentum_penalty": momentum_penalty,
        "momentum_reason": momentum_ctx["reason"],
        "momentum_contra": momentum_ctx["contra_count"],
        "momentum_total_move_atr": momentum_ctx["total_move_atr"],
        "momentum_body_accel": momentum_ctx["body_accel"],
        "momentum_prev_body_atr": momentum_ctx.get("prev_body_atr", 0.0),
        "continuation_flag": momentum_ctx.get("continuation_flag", False),
        # TREND
        "trend_ok": trend_ctx["trend_ok"],
        "trend_dir": trend_ctx["trend_dir"],
        "trend_slope": trend_ctx["slope_atr"],
        "trend_reason": trend_ctx["reason"],
        # BREAKOUT
        "breakout_detected": breakout_detected,
        # CASCATA
        "is_cascade": cascade_ctx["is_cascade"],
        "cascade_dir": cascade_ctx["cascade_dir"],
        "cascade_candles": cascade_ctx["cascade_candles"],
        "cascade_move_atr": cascade_ctx["cascade_move_atr"],
        "cascade_blocks": cascade_ctx["cascade_blocks"],
        "cascade_reason": cascade_ctx["cascade_reason"],
        # CANAL
        "has_channel": channel_ctx.get("has_channel", False),
        "channel_type": channel_ctx.get("channel_type", "none"),
        "channel_at_upper": channel_ctx.get("at_upper", False),
        "channel_at_lower": channel_ctx.get("at_lower", False),
        "channel_width_atr": channel_ctx.get("channel_width_atr", 0.0),
        "channel_quality": channel_ctx.get("channel_quality", 0.0),
        "channel_dir_signal": channel_ctx.get("channel_dir_signal", "NEUTRAL"),
        "channel_bonus": float(channel_ctx.get("channel_bonus", 0.0)) if channel_ctx.get("has_channel") else 0.0,
        "channel_reason": channel_ctx.get("channel_reason", ""),
        # WICK REJECTION
        "has_wick_rejection": wick_ctx.get("has_wick_rejection", False),
        "wick_ratio": wick_ctx.get("wick_ratio", 0.0),
        "wick_atr": wick_ctx.get("wick_atr", 0.0),
        "wick_pattern": wick_ctx.get("wick_pattern", "none"),
        "wick_at_zone": wick_ctx.get("wick_at_zone", False),
        "wick_at_channel": wick_ctx.get("wick_at_channel", False),
        "wick_bonus": float(wick_ctx.get("wick_bonus", 0.0)) if wick_ctx.get("has_wick_rejection") else 0.0,
        "wick_reason": wick_ctx.get("wick_reason", ""),
        # ZONA — para o raciocínio inteligente de entrada
        "zone_price": float(active_zone["price"]),
        "zone_high": float(active_zone.get("zone_high", active_zone["price"] + atr_val * 0.2)),
        "zone_low": float(active_zone.get("zone_low", active_zone["price"] - atr_val * 0.2)),
        # zone_type reflete a FUNÇÃO ATUAL (baseada na posição do preço)
        "zone_type": "support" if zone_dir == "CALL" else ("resistance" if zone_dir == "PUT" else "mixed"),
        "zone_distance_atr": dist_atr,
        # CLUSTER S/R: zonas agrupadas
        "nearby_zones_count": nearby_zones_count,
        "nearby_zones_touches": nearby_zones_touches,
        "cluster_bonus": cluster_bonus,
        # POSIÇÃO NO RANGE: 0.0=fundo 1.0=topo — para debug/log
        "range_position": round(range_position, 2),
        # MULTI-TIMEFRAME M5
        "m5_zone_exists": m5_ctx["m5_zone_exists"],
        "m5_trend_aligned": m5_ctx["m5_trend_aligned"],
        "m5_trend_contra": m5_ctx["m5_trend_contra"],
        "m5_trend_dir": m5_ctx["m5_trend_dir"],
        "m5_rejection": m5_ctx["m5_rejection"],
        "m5_confluence": m5_ctx["m5_confluence"],
        "m5_reason": m5_ctx["m5_reason"],
        # MULTI-TIMEFRAME M15 (S/R institucional)
        "m15_zone_exists": m15_ctx["m15_zone_exists"],
        "m15_trend_aligned": m15_ctx["m15_trend_aligned"],
        "m15_rejection": m15_ctx["m15_rejection"],
        "m15_confluence": m15_ctx["m15_confluence"],
        "m15_reason": m15_ctx["m15_reason"],
        # ESPAÇO PARA LUCRO
        "has_space": space_ctx["has_space"],
        "space_atr": space_ctx["space_atr"],
        "space_reason": space_ctx["space_reason"],
        # CANDLE COLOR AI
        "candle_ai_confirmed": candle_ai["confirmed"],
        "candle_ai_probability": candle_ai["probability"],
        "candle_ai_pattern": candle_ai.get("pattern_name", "none"),
        "candle_ai_dominance": candle_ai.get("dominance", "NEUTRAL"),
        "candle_ai_streak": candle_ai.get("streak", 0),
        "candle_ai_reason": candle_ai.get("reason", ""),
        # ESTRUTURA DE MERCADO (LH/HL)
        "structure_danger": structure_ctx["structure_danger"],
        "structure_lh_count": structure_ctx["lh_count"],
        "structure_hl_count": structure_ctx["hl_count"],
        "structure_reason": structure_ctx["reason"],
        # CONFLUÊNCIA TOTAL
        "confluence_points": confluence_points,
        "confluence_details": confluence_details,
    })

    return result


# ═══════════════════════════════════════════════════════════════
# 7. BACKTEST — Testar a estratégia em dados históricos
# ═══════════════════════════════════════════════════════════════
def backtest_sr(df_full: pd.DataFrame, min_candles: int = 200,
                test_window: int = 120,
                feature_fn=None,
                exp_fn=None) -> Dict[str, Any]:
    """
    Roda a estratégia nos últimos `test_window` candles.
    Para cada vela, simula simple_sr_signal nos dados anteriores
    e verifica se a direção estava certa na expiração.
    
    feature_fn: opcional — callable(df, setup, atr) -> dict de features.
                Quando fornecido, coleta features reais para treinamento ML.
    exp_fn:     opcional — callable(setup, atr) -> int (minutos de expiração).
                Quando fornecido, olha N candles à frente (N = expiração).
                Se None, olha apenas 1 candle à frente (como antes).
    
    Retorna: total_signals, wins, losses, win_rate, passed
             + training_samples (se feature_fn fornecido)
    """
    n = len(df_full)
    if n < min_candles:
        return {"total_signals": 0, "wins": 0, "losses": 0,
                "win_rate": 0.0, "passed": False, "reason": "dados_insuficientes"}

    start_idx = max(min_candles, n - test_window - 1)
    # Reservar espaço para expiração máxima de 5 candles à frente
    end_idx = n - 6 if exp_fn is not None else n - 1

    wins = 0
    losses = 0
    empates = 0
    samples = [] if feature_fn is not None else None

    for i in range(start_idx, end_idx):
        df_hist = df_full.iloc[:i + 1].copy()
        if len(df_hist) < MIN_CANDLES:
            continue

        # ATR
        atr_sub = df_hist.tail(16)
        h = atr_sub["high"].to_numpy(float)
        l = atr_sub["low"].to_numpy(float)
        c = atr_sub["close"].to_numpy(float)
        if len(h) < 3:
            continue
        tr = np.maximum(h[1:] - l[1:],
                        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        atr_val = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))
        if atr_val <= 0:
            continue

        try:
            signal = simple_sr_signal(df_hist, atr_val)
        except Exception:
            continue

        if not signal.get("trade"):
            continue

        direction = signal.get("dir")
        if not direction or direction == "NEUTRAL":
            continue

        # ── Expiração dinâmica: quantos candles à frente olhar ──
        exp_candles = 1
        if exp_fn is not None:
            try:
                exp_candles = max(1, min(5, exp_fn(signal, atr_val)))
            except Exception:
                exp_candles = 1

        # Garantir dados suficientes à frente
        if i + exp_candles >= len(df_full):
            continue

        # Resultado: preço na expiração vs preço de entrada
        # Entry = open do candle seguinte (como no live)
        # Exit  = close do candle na expiração (N minutos depois)
        entry_price = float(df_full.iloc[i + 1]["open"])
        exit_price = float(df_full.iloc[i + exp_candles]["close"])

        if exit_price == entry_price:
            empates += 1
            continue

        won = (direction == "CALL" and exit_price > entry_price) or \
              (direction == "PUT" and exit_price < entry_price)

        if won:
            wins += 1
        else:
            losses += 1

        # Coletar features reais para treinamento ML
        if feature_fn is not None:
            try:
                feats = feature_fn(df_hist, signal, atr_val)
                if feats:
                    samples.append((feats, 1 if won else 0))
            except Exception:
                pass

    total = wins + losses
    win_rate = (wins / total * 100) if total > 0 else 0.0

    result = {
        "total_signals": total,
        "wins": wins,
        "losses": losses,
        "empates": empates,
        "win_rate": round(win_rate, 1),
        # Se < 4 sinais: estratégia muito seletiva → NÃO bloquear
        # Só bloquear quando temos PROVA de perda (4+ sinais e WR < 50%)
        "passed": total < 4 or win_rate >= 50.0,
        "reason": "ok" if total >= 4 else "seletivo_ok",
    }
    if samples is not None:
        result["training_samples"] = samples
    return result
