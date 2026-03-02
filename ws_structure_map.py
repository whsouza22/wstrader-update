"""
ws_structure_map.py — Mapeamento Estrutural de Preço (ZigZag + Regiões)
=======================================================================

Baseado na leitura visual de um trader profissional:
  1. Mapeia o ZigZag do preço (swing highs e swing lows)
  2. Cria REGIÕES a partir dos pontos de reversão
  3. Identifica a TENDÊNCIA ESTRUTURAL (HH/HL = alta, LH/LL = baixa)
  4. Quando preço TOCA uma região → sinaliza entrada na direção da estrutura

Conceito:
  - O preço se move em ondas (zigzag)
  - Cada onda cria um ponto de reversão
  - Esses pontos formam REGIÕES horizontais
  - Quando o preço retorna a uma região, é provável que reaja novamente
  - A direção da reação depende da ESTRUTURA (HH/HL vs LH/LL)

Exemplo do gráfico:
  - Preço sobe até 0.584 (topo) → cria REGIÃO de resistência
  - Preço cai até 0.555 (fundo) → cria REGIÃO de suporte
  - Preço sobe de novo até 0.575 (topo menor = LH)
  - Preço cai até 0.558 (fundo maior = HL)
  → Estrutura: consolidação/range
  → Regiões: 0.584, 0.575, 0.558, 0.555
  → Preço toca 0.558 → CALL (suporte + HL = compradores sustentando)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

log = logging.getLogger("ws_structure_map")


# ═══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════

# ZigZag: mínimo de variação para considerar um novo swing (em ATRs)
ZIGZAG_MIN_CHANGE_ATR = 0.40     # Swing precisa ter pelo menos 0.4 ATR

# Janela para detectar pivots (em candles)
SWING_WINDOW = 3                  # 3 candles de cada lado

# Tolerância para considerar que preço "tocou" uma região (em ATRs)
REGION_TOUCH_TOL_ATR = 0.25      # Preço dentro de 0.25 ATR da região

# Mínimo de candles entre pivots para evitar ruído
MIN_SWING_DISTANCE = 3

# Máximo de regiões a mapear
MAX_REGIONS = 12

# Mínimo de swings para classificar estrutura
MIN_SWINGS_FOR_STRUCTURE = 4

# Tolerância para agrupar swings em regiões (em ATRs)
CLUSTER_TOL_ATR = 0.30


# ═══════════════════════════════════════════════════════════════
# 1. ZIGZAG — Mapear swing highs e swing lows
# ═══════════════════════════════════════════════════════════════

def find_swings(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                atr_val: float, window: int = SWING_WINDOW
                ) -> List[Tuple[int, float, str]]:
    """
    Encontra swing highs e swing lows no preço.
    
    Retorna lista de (index, price, type) onde type = "H" ou "L"
    Ordenado por index.
    """
    n = len(highs)
    if n < window * 2 + 1:
        return []
    
    min_move = ZIGZAG_MIN_CHANGE_ATR * atr_val
    swings: List[Tuple[int, float, str]] = []
    
    # Fase 1: encontrar candidatos a swing high/low
    candidates = []
    
    for i in range(window, n - window):
        is_high = True
        is_low = True
        
        for j in range(i - window, i + window + 1):
            if j == i:
                continue
            if highs[j] > highs[i]:
                is_high = False
            if lows[j] < lows[i]:
                is_low = False
        
        if is_high:
            candidates.append((i, float(highs[i]), "H"))
        if is_low:
            candidates.append((i, float(lows[i]), "L"))
    
    if not candidates:
        return []
    
    # Fase 2: filtrar — alternar H/L e manter distância mínima
    candidates.sort(key=lambda x: x[0])
    
    filtered = [candidates[0]]
    for c in candidates[1:]:
        idx, price, stype = c
        last_idx, last_price, last_type = filtered[-1]
        
        # Se mesmo tipo, manter o mais extremo
        if stype == last_type:
            if stype == "H" and price > last_price:
                filtered[-1] = c
            elif stype == "L" and price < last_price:
                filtered[-1] = c
        else:
            # Tipo diferente — verificar distância mínima
            if abs(price - last_price) >= min_move and (idx - last_idx) >= MIN_SWING_DISTANCE:
                filtered.append(c)
    
    return filtered


# ═══════════════════════════════════════════════════════════════
# 2. REGIÕES — Agrupar swings em zonas horizontais
# ═══════════════════════════════════════════════════════════════

def build_regions(swings: List[Tuple[int, float, str]],
                  atr_val: float,
                  current_price: float = 0.0
                  ) -> List[Dict[str, Any]]:
    """
    Agrupa swing points em REGIÕES (zonas horizontais).
    
    Cada região tem:
      - price: preço médio da região
      - high/low: limites da região
      - touches: quantos swings tocaram essa região
      - type: "resistance", "support" ou "both"
      - last_touch_idx: índice do último swing que tocou
      - swing_types: lista dos tipos de swing ("H"/"L") que tocaram
    """
    if not swings:
        return []
    
    tol = CLUSTER_TOL_ATR * atr_val
    
    # Extrair preços e tipos
    prices_types = [(price, stype, idx) for idx, price, stype in swings]
    prices_types.sort(key=lambda x: x[0])
    
    # Agrupar em clusters
    clusters: List[List[Tuple[float, str, int]]] = []
    current_cluster = [prices_types[0]]
    
    for pt in prices_types[1:]:
        if pt[0] - current_cluster[0][0] <= tol:
            current_cluster.append(pt)
        else:
            clusters.append(current_cluster)
            current_cluster = [pt]
    clusters.append(current_cluster)
    
    # Converter clusters em regiões
    regions = []
    for cluster in clusters:
        prices = [p for p, _, _ in cluster]
        types = [t for _, t, _ in cluster]
        indices = [i for _, _, i in cluster]
        
        avg_price = float(np.mean(prices))
        margin = max(tol * 0.3, (max(prices) - min(prices)) * 0.5 + atr_val * 0.05)
        
        # Classificar tipo
        n_highs = types.count("H")
        n_lows = types.count("L")
        if n_highs > 0 and n_lows > 0:
            rtype = "both"  # Preço usado como suporte E resistência
        elif n_highs > n_lows:
            rtype = "resistance"
        else:
            rtype = "support"
        
        regions.append({
            "price": avg_price,
            "high": float(max(prices)) + atr_val * 0.05,
            "low": float(min(prices)) - atr_val * 0.05,
            "touches": len(cluster),
            "type": rtype,
            "last_touch_idx": max(indices),
            "swing_types": types,
            "n_highs": n_highs,
            "n_lows": n_lows,
        })
    
    # Ordenar por proximidade do preço atual (se fornecido)
    if current_price > 0:
        regions.sort(key=lambda r: abs(r["price"] - current_price))
    else:
        regions.sort(key=lambda r: r["touches"], reverse=True)
    
    return regions[:MAX_REGIONS]


# ═══════════════════════════════════════════════════════════════
# 3. ESTRUTURA — Classificar tendência (HH/HL/LH/LL)
# ═══════════════════════════════════════════════════════════════

def classify_structure(swings: List[Tuple[int, float, str]],
                       atr_val: float
                       ) -> Dict[str, Any]:
    """
    Classifica a estrutura do mercado baseada nos swings.
    
    HH + HL = UPTREND (tendência de alta)
    LH + LL = DOWNTREND (tendência de baixa)
    Misturado = RANGE (consolidação)
    
    Retorna:
      structure: "uptrend", "downtrend", "range"
      hh_count, hl_count, lh_count, ll_count: contagens
      last_swing_type: "H" ou "L"
      last_swing_price: preço do último swing
      expected_next: "H" ou "L" (próximo swing esperado)
      expected_dir: "CALL" ou "PUT" baseado na estrutura
      confidence: 0-1 (quão clara é a estrutura)
    """
    result = {
        "structure": "range",
        "hh_count": 0, "hl_count": 0,
        "lh_count": 0, "ll_count": 0,
        "last_swing_type": "",
        "last_swing_price": 0.0,
        "expected_next": "",
        "expected_dir": "",
        "confidence": 0.0,
        "trend_label": "",
    }
    
    if len(swings) < MIN_SWINGS_FOR_STRUCTURE:
        result["structure"] = "insufficient"
        return result
    
    # Separar highs e lows mantendo ordem
    highs_seq = [(idx, price) for idx, price, stype in swings if stype == "H"]
    lows_seq = [(idx, price) for idx, price, stype in swings if stype == "L"]
    
    min_diff = atr_val * 0.08  # diferença mínima para considerar HH/LH etc
    
    # Contar HH/LH
    hh, lh = 0, 0
    for i in range(1, len(highs_seq)):
        diff = highs_seq[i][1] - highs_seq[i-1][1]
        if diff > min_diff:
            hh += 1
        elif diff < -min_diff:
            lh += 1
    
    # Contar HL/LL
    hl, ll = 0, 0
    for i in range(1, len(lows_seq)):
        diff = lows_seq[i][1] - lows_seq[i-1][1]
        if diff > min_diff:
            hl += 1
        elif diff < -min_diff:
            ll += 1
    
    result["hh_count"] = hh
    result["hl_count"] = hl
    result["lh_count"] = lh
    result["ll_count"] = ll
    
    # Último swing
    last = swings[-1]
    result["last_swing_type"] = last[2]
    result["last_swing_price"] = last[1]
    
    # Próximo esperado (alternância H/L)
    result["expected_next"] = "L" if last[2] == "H" else "H"
    
    # Classificar estrutura
    up_score = hh + hl       # pontos para uptrend
    down_score = lh + ll     # pontos para downtrend
    total = max(1, up_score + down_score)
    
    if up_score >= 3 and up_score > down_score * 2:
        result["structure"] = "uptrend"
        result["trend_label"] = f"UP ({hh}HH+{hl}HL)"
        result["confidence"] = min(1.0, up_score / total)
    elif down_score >= 3 and down_score > up_score * 2:
        result["structure"] = "downtrend"
        result["trend_label"] = f"DOWN ({lh}LH+{ll}LL)"
        result["confidence"] = min(1.0, down_score / total)
    elif up_score >= 2 and up_score > down_score:
        result["structure"] = "uptrend_weak"
        result["trend_label"] = f"up_fraco ({hh}HH+{hl}HL vs {lh}LH+{ll}LL)"
        result["confidence"] = min(0.6, up_score / total)
    elif down_score >= 2 and down_score > up_score:
        result["structure"] = "downtrend_weak"
        result["trend_label"] = f"dn_fraco ({lh}LH+{ll}LL vs {hh}HH+{hl}HL)"
        result["confidence"] = min(0.6, down_score / total)
    else:
        result["structure"] = "range"
        result["trend_label"] = f"RANGE ({hh}HH+{hl}HL vs {lh}LH+{ll}LL)"
        result["confidence"] = 0.3
    
    # Direção esperada baseada na estrutura
    struct = result["structure"]
    if struct in ("uptrend", "uptrend_weak"):
        # Uptrend: esperar pullback ao suporte (HL) → CALL
        result["expected_dir"] = "CALL"
    elif struct in ("downtrend", "downtrend_weak"):
        # Downtrend: esperar reteste da resistência (LH) → PUT
        result["expected_dir"] = "PUT"
    else:
        # Range: depende de qual região toca
        result["expected_dir"] = ""  # decidido pela região
    
    return result


# ═══════════════════════════════════════════════════════════════
# 4. DETECÇÃO PRINCIPAL — Preço está tocando uma região?
# ═══════════════════════════════════════════════════════════════

def detect_structure_touch(df, atr_val: float,
                           lookback: int = 200
                           ) -> Dict[str, Any]:
    """
    Análise completa: mapeia estrutura + regiões + detecta se está tocando.
    
    Args:
        df: DataFrame com open/high/low/close
        atr_val: ATR atual (para definir tolerâncias)
        lookback: candles para análise (default 200)
    
    Returns:
        Dict com:
          - "touch": bool — preço está tocando uma região?
          - "region": Dict da região sendo tocada (ou None)
          - "structure": Dict com classificação da estrutura
          - "direction": "CALL"/"PUT"/"" — direção sugerida
          - "confidence": 0-1
          - "regions_count": total de regiões mapeadas
          - "swings_count": total de swings encontrados
          - "reason": str — explicação da decisão
          - "bonus_pct": float — bônus sugerido (+0.05 a +0.15)
    """
    result = {
        "touch": False,
        "region": None,
        "structure": None,
        "direction": "",
        "confidence": 0.0,
        "regions_count": 0,
        "swings_count": 0,
        "reason": "sem_dados",
        "bonus_pct": 0.0,
    }
    
    if df is None or len(df) < 30:
        return result
    
    # Limitar lookback
    if len(df) > lookback:
        df = df.tail(lookback)
    
    opens = df["open"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    n = len(df)
    
    current_price = float(closes[-1])
    current_high = float(highs[-1])
    current_low = float(lows[-1])
    
    # ── 1. Encontrar swings ──
    swings = find_swings(highs, lows, closes, atr_val)
    result["swings_count"] = len(swings)
    
    if len(swings) < 3:
        result["reason"] = f"poucos_swings({len(swings)})"
        return result
    
    # ── 2. Classificar estrutura ──
    structure = classify_structure(swings, atr_val)
    result["structure"] = structure
    
    # ── 3. Construir regiões ──
    regions = build_regions(swings, atr_val, current_price)
    result["regions_count"] = len(regions)
    
    if not regions:
        result["reason"] = "sem_regioes"
        return result
    
    # ── 4. Verificar se preço toca alguma região ──
    touch_tol = REGION_TOUCH_TOL_ATR * atr_val
    touching_region = None
    
    for region in regions:
        r_high = region["high"] + touch_tol
        r_low = region["low"] - touch_tol
        
        # Preço atual (close, high ou low) está dentro da região expandida?
        price_in_region = (
            (r_low <= current_price <= r_high) or
            (r_low <= current_high <= r_high) or
            (r_low <= current_low <= r_high)
        )
        
        if price_in_region:
            touching_region = region
            break
    
    if touching_region is None:
        # Encontrar região mais próxima para log
        closest = regions[0]
        dist = abs(current_price - closest["price"]) / atr_val
        result["reason"] = f"fora_regiao (mais_prox={closest['price']:.5f} dist={dist:.2f}ATR)"
        return result
    
    result["touch"] = True
    result["region"] = touching_region
    
    # ── 5. Determinar direção ──
    region_type = touching_region["type"]
    struct_type = structure["structure"]
    struct_dir = structure["expected_dir"]
    
    # Lógica de direção:
    direction = ""
    confidence = 0.0
    reason_parts = []
    
    # A. Região é suporte → CALL (preço deve voltar a subir)
    # B. Região é resistência → PUT (preço deve voltar a cair)
    # C. Estrutura reforça: uptrend + suporte = CALL forte
    #                        downtrend + resistência = PUT forte
    
    if region_type == "support":
        direction = "CALL"
        confidence = 0.55
        reason_parts.append(f"toque_SUPORTE({touching_region['touches']}t)")
    elif region_type == "resistance":
        direction = "PUT"
        confidence = 0.55
        reason_parts.append(f"toque_RESISTENCIA({touching_region['touches']}t)")
    elif region_type == "both":
        # Região foi suporte e resistência — usar estrutura para decidir
        if current_price <= touching_region["price"]:
            direction = "CALL"  # abaixo da região = suporte
            reason_parts.append(f"toque_ZONA_MISTA_abaixo({touching_region['touches']}t)")
        else:
            direction = "PUT"   # acima da região = resistência
            reason_parts.append(f"toque_ZONA_MISTA_acima({touching_region['touches']}t)")
        confidence = 0.50
    
    # Bônus por estrutura alinhada
    if struct_dir == direction and struct_dir:
        confidence += 0.15
        reason_parts.append(f"estrutura_ALINHADA({structure['trend_label']})")
    elif struct_dir and struct_dir != direction:
        confidence -= 0.10
        reason_parts.append(f"estrutura_CONTRA({structure['trend_label']})")
    
    # Bônus por múltiplos toques (região provada)
    if touching_region["touches"] >= 4:
        confidence += 0.10
        reason_parts.append(f"regiao_FORTE({touching_region['touches']}t)")
    elif touching_region["touches"] >= 3:
        confidence += 0.05
        reason_parts.append(f"regiao_boa({touching_region['touches']}t)")
    
    # Bônus se é região "both" (suporte que virou resistência ou vice-versa)
    if region_type == "both":
        confidence += 0.05
        reason_parts.append("flip_zone")
    
    # Verificar se candle atual mostra rejeição na região
    body = abs(closes[-1] - opens[-1])
    upper_wick = highs[-1] - max(opens[-1], closes[-1])
    lower_wick = min(opens[-1], closes[-1]) - lows[-1]
    total_candle = highs[-1] - lows[-1]
    
    if total_candle > 0:
        if direction == "CALL" and lower_wick > body * 1.5 and lower_wick > total_candle * 0.4:
            confidence += 0.08
            reason_parts.append("rejeicao_wick_CALL")
        elif direction == "PUT" and upper_wick > body * 1.5 and upper_wick > total_candle * 0.4:
            confidence += 0.08
            reason_parts.append("rejeicao_wick_PUT")
    
    # "Último swing" — se o último swing foi contrário, é pullback
    if structure["last_swing_type"] == "H" and direction == "CALL":
        confidence += 0.05
        reason_parts.append("pullback_de_topo")
    elif structure["last_swing_type"] == "L" and direction == "PUT":
        confidence += 0.05
        reason_parts.append("pullback_de_fundo")
    
    # Limitar confiança
    confidence = max(0.0, min(1.0, confidence))
    
    # Calcular bônus de probabilidade
    if confidence >= 0.80:
        bonus = 0.15   # Toque em região forte + estrutura alinhada + rejeição
    elif confidence >= 0.70:
        bonus = 0.12
    elif confidence >= 0.60:
        bonus = 0.08
    elif confidence >= 0.50:
        bonus = 0.05
    else:
        bonus = 0.0
        direction = ""   # Confiança muito baixa → não sugerir direção
    
    result["direction"] = direction
    result["confidence"] = confidence
    result["reason"] = " | ".join(reason_parts)
    result["bonus_pct"] = bonus
    
    return result


# ═══════════════════════════════════════════════════════════════
# 5. RESUMO PARA LOG
# ═══════════════════════════════════════════════════════════════

def structure_map_summary(result: Dict) -> str:
    """Retorna string resumo para log."""
    if not result.get("touch"):
        return f"MAP: sem toque ({result.get('reason', '?')}) [{result.get('swings_count', 0)} swings, {result.get('regions_count', 0)} regiões]"
    
    region = result.get("region", {})
    struct = result.get("structure", {})
    return (
        f"MAP: TOQUE {result['direction']} "
        f"região={region.get('price', 0):.5f} "
        f"({region.get('type', '?')}, {region.get('touches', 0)}t) "
        f"conf={result['confidence']:.0%} "
        f"bonus=+{result['bonus_pct']:.0%} "
        f"| EST: {struct.get('trend_label', '?')} "
        f"| {result['reason']}"
    )


# ═══════════════════════════════════════════════════════════════
# 6. VERIFICAR ALINHAMENTO COM SETUP EXISTENTE
# ═══════════════════════════════════════════════════════════════

def check_alignment(map_result: Dict, setup_dir: str) -> Dict[str, Any]:
    """
    Verifica se o mapeamento estrutural está ALINHADO com o setup S/R.
    
    Se ambos concordam (mesmo direction) → reforço forte.
    Se discordam → cautela.
    
    Returns:
        aligned: bool
        bonus: float (bônus adicional se alinhado)
        detail: str
    """
    if not map_result.get("touch") or not map_result.get("direction"):
        return {"aligned": False, "bonus": 0.0, "detail": "sem_mapa"}
    
    map_dir = map_result["direction"]
    map_conf = map_result["confidence"]
    
    if map_dir == setup_dir:
        # ALINHADO: mapa estrutural + S/R concordam
        bonus = min(0.10, map_result["bonus_pct"] * 0.7)
        return {
            "aligned": True,
            "bonus": bonus,
            "detail": f"ALINHADO({map_dir} conf={map_conf:.0%})",
        }
    else:
        # CONTRA: mapa diz uma direção, S/R diz outra
        return {
            "aligned": False,
            "bonus": -0.05,
            "detail": f"CONTRA(mapa={map_dir} vs setup={setup_dir})",
        }


# ═══════════════════════════════════════════════════════════════
# TESTE LOCAL
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("ws_structure_map — Mapeamento Estrutural de Preço")
    print("=" * 60)
    
    # Simular dados
    np.random.seed(42)
    n = 200
    prices = np.cumsum(np.random.randn(n) * 0.001) + 1.0
    
    import pandas as pd
    df = pd.DataFrame({
        "open": prices,
        "high": prices + np.random.rand(n) * 0.002,
        "low": prices - np.random.rand(n) * 0.002,
        "close": prices + np.random.randn(n) * 0.001,
    })
    
    atr = 0.005
    
    # Encontrar swings
    swings = find_swings(
        df["high"].values.astype(float),
        df["low"].values.astype(float),
        df["close"].values.astype(float),
        atr
    )
    print(f"\nSwings encontrados: {len(swings)}")
    for idx, price, stype in swings[-10:]:
        print(f"  [{idx:3d}] {'▲' if stype == 'H' else '▼'} {price:.5f}")
    
    # Estrutura
    struct = classify_structure(swings, atr)
    print(f"\nEstrutura: {struct['trend_label']}")
    print(f"  HH={struct['hh_count']} HL={struct['hl_count']} "
          f"LH={struct['lh_count']} LL={struct['ll_count']}")
    print(f"  Direção esperada: {struct['expected_dir']}")
    print(f"  Confiança: {struct['confidence']:.0%}")
    
    # Regiões
    regions = build_regions(swings, atr, float(df["close"].iloc[-1]))
    print(f"\nRegiões mapeadas: {len(regions)}")
    for r in regions[:6]:
        print(f"  {r['type']:12s} @ {r['price']:.5f} "
              f"[{r['low']:.5f}-{r['high']:.5f}] "
              f"{r['touches']}t")
    
    # Detecção completa
    result = detect_structure_touch(df, atr)
    print(f"\n{structure_map_summary(result)}")
