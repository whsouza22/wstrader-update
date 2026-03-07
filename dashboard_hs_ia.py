"""
dashboard_hs_ia.py — Dashboard IA Double Touch (Duplo Toque) — SOMENTE LEITURA
============================================================================
Servidor HTTP local que:
  1. Lê dados de velas do cache do bot (ws_dashboard_cache.json)
  2. NÃO conecta ao broker — apenas visualiza sinais
  3. Detecta TODOS os padrões Double Touch históricos nos dados do cache
  4. Backtest: verifica se cada padrão deu WIN ou LOSS (3 velas após entrada)
  5. Treina a IA com os resultados (aprende quais setups são bons)
  6. Mostra: gráfico, padrões, sinais, win rate

IMPORTANTE: O dashboard NÃO faz trades. O bot (WS_AUTO_AI_BULLEX.py) é
a ÚNICA fonte de dados e a ÚNICA conexão ao broker.
O dashboard é SOMENTE VISUALIZAÇÃO.

Uso:
  python dashboard_hs_ia.py                   (porta padrão 8899)
  python dashboard_hs_ia.py --port 9999       (porta customizada)

Acesse: http://localhost:8899
"""

import os, sys, json, time, logging, argparse, threading
import numpy as np
import pandas as pd
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("HS_IA")

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
DEFAULT_PORT = 8899
N_CANDLES = 900         # 900 velas M1 = 15h de dados
EXP_CANDLES = 3         # Expiração: 3 velas (3 min) para verificar WIN/LOSS
MIN_PAYOUT = 80
MAX_ASSETS = 20
PIVOT_WINDOW = 5

# ── Persistência / Retrain semanal ──
_USER_DIR = os.path.join(os.path.expanduser("~"), ".wstrader")
IA_PERSIST_FILE = os.path.join(_USER_DIR, "hs_ia_dashboard_stats.json")
TRAIN_CONTROL_FILE = os.path.join(_USER_DIR, "hs_ia_train_control.json")

# ── Cache compartilhado: o BOT escreve, dashboard apenas LÊ ──
_DASHBOARD_CACHE_FILE = os.path.join(_USER_DIR, "ws_dashboard_cache.json")

# ── Premium Gate ──
_PREMIUM_PRODUCT_ID = "prod_U4ZxrEEApDg2Hb"   # PREMIUM — acesso total
_PRO_PRODUCT_ID     = "prod_S4t8FQuUptWQ6R"   # PRO
_DEMO_PRODUCT_ID    = "prod_U3CRqZJMVigJAK"   # DEMO
_stripe_product     = os.environ.get("STRIPE_PRODUCT_ID", "")
_IS_PREMIUM = (_stripe_product == _PREMIUM_PRODUCT_ID)
_IS_PRO     = (_stripe_product == _PRO_PRODUCT_ID)
_IS_PAID    = _IS_PREMIUM or _IS_PRO

# ══════════════════════════════════════════════════════════════════
# DETECÇÃO DE PIVOTS
# ══════════════════════════════════════════════════════════════════
def detect_pivots(H, L, window=5):
    n = len(H)
    ph, pl = [], []
    edge_min = 2
    for i in range(window, n - edge_min):
        rw = min(window, n - 1 - i)
        is_ph = True
        for j in range(1, window + 1):
            if H[i] <= H[i - j]:
                is_ph = False; break
        if is_ph:
            for j in range(1, rw + 1):
                if H[i] <= H[i + j]:
                    is_ph = False; break
        if is_ph:
            ph.append((i, float(H[i])))
        is_pl = True
        for j in range(1, window + 1):
            if L[i] >= L[i - j]:
                is_pl = False; break
        if is_pl:
            for j in range(1, rw + 1):
                if L[i] >= L[i + j]:
                    is_pl = False; break
        if is_pl:
            pl.append((i, float(L[i])))
    return ph, pl


def _extract_geometry(pat, atr_val):
    """Extrai features geométricas de um padrão H&S para a IA aprender."""
    try:
        iL = pat["left_shoulder"]["idx"]
        iH = pat["head"]["idx"]
        iR = pat["right_shoulder"]["idx"]
        span = iR - iL
        depth = pat.get("depth", 0)
        neck = pat.get("neckline", 0)
        v1 = pat.get("valley1", {}).get("price", neck)
        v2 = pat.get("valley2", {}).get("price", neck)
        d_left = iH - iL
        d_right = iR - iH
        symmetry = min(d_left, d_right) / max(d_left, d_right) if max(d_left, d_right) > 0 else 0
        depth_ratio = depth / atr_val if atr_val > 0 else 0
        neck_align = abs(v1 - v2) / atr_val if atr_val > 0 else 0
        return {
            "span": span,
            "symmetry": round(symmetry, 4),
            "depth_ratio": round(depth_ratio, 4),
            "neck_align": round(neck_align, 4),
        }
    except Exception:
        return None


def ia_pattern_quality(pat, atr_val, geo_history=None):
    """IA que APRENDE da geometria dos padrões — sem regras hardcoded.
    Compara features geométricas do padrão atual contra o perfil
    estatístico dos padrões que deram WIN no histórico.
    Retorna fator 0.50-1.0. Quando não há dados suficientes, retorna 1.0 (neutro).
    """
    geo = _extract_geometry(pat, atr_val)
    if geo is None or geo_history is None:
        return 1.0
    if len(geo_history) < 10:
        return 1.0
    win_geos = [g for g in geo_history if g.get("result") == 1]
    if len(win_geos) < 5:
        return 1.0
    features = ["span", "symmetry", "depth_ratio", "neck_align"]
    score_sum = 0.0
    n_feat = 0
    for feat in features:
        win_vals = [g[feat] for g in win_geos if feat in g]
        if len(win_vals) < 3:
            continue
        mean_w = sum(win_vals) / len(win_vals)
        variance = sum((v - mean_w) ** 2 for v in win_vals) / len(win_vals)
        std_w = variance ** 0.5 if variance > 0 else mean_w * 0.3
        if std_w < 0.001:
            std_w = 0.001
        current_val = geo.get(feat, mean_w)
        distance = abs(current_val - mean_w) / std_w
        feat_score = max(0.50, 1.0 - distance * 0.12)
        score_sum += feat_score
        n_feat += 1
    if n_feat == 0:
        return 1.0
    final = score_sum / n_feat
    return round(max(0.50, min(1.0, final)), 4)


# ══════════════════════════════════════════════════════════════════
# DETECÇÃO H&S COMPLETA (todos os padrões históricos)
# ══════════════════════════════════════════════════════════════════
def detect_all_hs(H, L, C, O, pivot_highs, pivot_lows, atr):
    """Detecta TODOS os padrões H&S/iH&S no histórico de 900 velas.
    Inclui validações: cabeça não pode ter sido rompida.
    Simetria temporal: braços não podem diferir mais de 3:1."""
    patterns = []
    n = len(H)
    tol = atr * 1.5
    min_depth = atr * 1.0
    min_spacing = 8
    max_span = 100
    trend_lookback = 30
    symmetry_min = 0.90
    temporal_sym_min = 0.30  # braço curto >= 30% do braço longo (max 3.3:1)
    seen_heads = set()

    # ── MODO 1: H&S Clássico (3 pivot highs) ──
    for i in range(len(pivot_highs) - 2):
        iL, pL = pivot_highs[i]
        iH, pH = pivot_highs[i + 1]
        iR, pR = pivot_highs[i + 2]
        if pH <= pL or pH <= pR: continue
        if abs(pL - pR) > tol: continue
        if iH - iL < min_spacing or iR - iH < min_spacing: continue
        if iR - iL > max_span: continue
        # Simetria temporal: braços devem ser proporcionais
        d_left = iH - iL
        d_right = iR - iH
        if min(d_left, d_right) / max(d_left, d_right) < temporal_sym_min: continue
        shoulder_avg = (pL + pR) / 2
        head_depth = pH - shoulder_avg
        if head_depth < min_depth: continue
        if min(pL, pR) / max(pL, pR) < symmetry_min: continue
        if iL >= trend_lookback:
            if float(C[iL]) <= float(C[iL - trend_lookback]): continue
        # Validação: cabeça não foi rompida
        if iH + 1 < iR + 1:
            if float(max(H[iH+1:iR+1])) >= pH: continue
        v1_region = L[iL:iH + 1]
        v1_rel = int(np.argmin(v1_region)); v1_idx = iL + v1_rel; v1_price = float(v1_region[v1_rel])
        v2_region = L[iH:iR + 1]
        v2_rel = int(np.argmin(v2_region)); v2_idx = iH + v2_rel; v2_price = float(v2_region[v2_rel])
        neckline = (v1_price + v2_price) / 2
        if abs(v1_price - v2_price) > atr * 0.5: continue
        neck_slope = (v2_price - v1_price) / max(1, v2_idx - v1_idx)
        seen_heads.add(("H", iH))
        patterns.append({
            "type": "HEAD_SHOULDERS", "direction": "PUT", "mode": "classic",
            "left_shoulder": {"idx": int(iL), "price": round(float(pL), 6)},
            "head": {"idx": int(iH), "price": round(float(pH), 6)},
            "right_shoulder": {"idx": int(iR), "price": round(float(pR), 6)},
            "valley1": {"idx": int(v1_idx), "price": round(v1_price, 6)},
            "valley2": {"idx": int(v2_idx), "price": round(v2_price, 6)},
            "neckline": round(neckline, 6),
            "neck_slope": round(neck_slope, 8),
            "depth": round(float(head_depth), 6),
            "target": round(neckline - head_depth, 6),
            "stop": round(float(pH), 6),
            "entry_idx": int(iR),
            "entry_price": round(float(C[int(iR)]), 6),
        })

    # ── MODO 1: iH&S Clássico (3 pivot lows) ──
    for i in range(len(pivot_lows) - 2):
        iL, pL = pivot_lows[i]
        iH, pH = pivot_lows[i + 1]
        iR, pR = pivot_lows[i + 2]
        if pH >= pL or pH >= pR: continue
        if abs(pL - pR) > tol: continue
        if iH - iL < min_spacing or iR - iH < min_spacing: continue
        if iR - iL > max_span: continue
        # Simetria temporal
        d_left = iH - iL
        d_right = iR - iH
        if min(d_left, d_right) / max(d_left, d_right) < temporal_sym_min: continue
        shoulder_avg = (pL + pR) / 2
        head_depth = shoulder_avg - pH
        if head_depth < min_depth: continue
        if min(pL, pR) / max(pL, pR) < symmetry_min: continue
        if iL >= trend_lookback:
            if float(C[iL]) >= float(C[iL - trend_lookback]): continue
        if iH + 1 < iR + 1:
            if float(min(L[iH+1:iR+1])) <= pH: continue
        v1_region = H[iL:iH + 1]
        v1_rel = int(np.argmax(v1_region)); v1_idx = iL + v1_rel; v1_price = float(v1_region[v1_rel])
        v2_region = H[iH:iR + 1]
        v2_rel = int(np.argmax(v2_region)); v2_idx = iH + v2_rel; v2_price = float(v2_region[v2_rel])
        neckline = (v1_price + v2_price) / 2
        if abs(v1_price - v2_price) > atr * 0.5: continue
        neck_slope = (v2_price - v1_price) / max(1, v2_idx - v1_idx)
        seen_heads.add(("L", iH))
        patterns.append({
            "type": "INV_HEAD_SHOULDERS", "direction": "CALL", "mode": "classic",
            "left_shoulder": {"idx": int(iL), "price": round(float(pL), 6)},
            "head": {"idx": int(iH), "price": round(float(pH), 6)},
            "right_shoulder": {"idx": int(iR), "price": round(float(pR), 6)},
            "valley1": {"idx": int(v1_idx), "price": round(v1_price, 6)},
            "valley2": {"idx": int(v2_idx), "price": round(v2_price, 6)},
            "neckline": round(neckline, 6),
            "neck_slope": round(neck_slope, 8),
            "depth": round(float(head_depth), 6),
            "target": round(neckline + head_depth, 6),
            "stop": round(float(pH), 6),
            "entry_idx": int(iR),
            "entry_price": round(float(C[int(iR)]), 6),
        })

    # ── MODO 2: H&S Tempo Real (PUT) ──
    rt_scan = 50
    for i in range(len(pivot_highs) - 1):
        iL, pL = pivot_highs[i]
        iH, pH = pivot_highs[i + 1]
        if ("H", iH) in seen_heads: continue
        if pH <= pL or iH - iL < min_spacing: continue
        head_depth = pH - pL
        if head_depth < min_depth: continue
        if iL >= trend_lookback:
            if float(C[iL]) <= float(C[iL - trend_lookback]): continue
        search_start = iH + min_spacing
        if search_start >= n: continue
        # Limitar busca: máx 3x a distância do braço esquerdo
        d_left = iH - iL
        search_end = min(n, iH + int(d_left * 3.5))
        region = H[search_start:search_end]
        if len(region) < 2: continue
        local_max_rel = int(np.argmax(region))
        iR = search_start + local_max_rel
        pR = float(H[iR])
        if abs(pL - pR) > tol or pR >= pH: continue
        if min(pL, pR) / max(pL, pR) < symmetry_min: continue
        if iR - iL > max_span: continue
        # Validar que iR é um pivot real (não apenas argmax da região)
        _pivot_check = min(3, iR - search_start, n - 1 - iR)
        if _pivot_check < 2: continue
        _is_pivot = all(H[iR] >= H[iR - j] for j in range(1, _pivot_check + 1)) and \
                    all(H[iR] >= H[iR + j] for j in range(1, min(_pivot_check + 1, n - iR)))
        if not _is_pivot: continue
        # Simetria temporal
        d_right = iR - iH
        if min(d_left, d_right) / max(d_left, d_right) < temporal_sym_min: continue
        # Validação cabeça
        if float(max(H[iH+1:n])) >= pH: continue
        v1_region = L[iL:iH + 1]
        v1_rel = int(np.argmin(v1_region)); v1_idx = iL + v1_rel; v1_price = float(v1_region[v1_rel])
        v2_region = L[iH:min(iR + 1, n)]
        v2_rel = int(np.argmin(v2_region)); v2_idx = iH + v2_rel; v2_price = float(v2_region[v2_rel])
        neckline = (v1_price + v2_price) / 2
        if abs(v1_price - v2_price) > atr * 0.5: continue
        neck_slope = (v2_price - v1_price) / max(1, v2_idx - v1_idx)
        patterns.append({
            "type": "HEAD_SHOULDERS", "direction": "PUT", "mode": "realtime",
            "left_shoulder": {"idx": int(iL), "price": round(float(pL), 6)},
            "head": {"idx": int(iH), "price": round(float(pH), 6)},
            "right_shoulder": {"idx": int(iR), "price": round(float(pR), 6)},
            "valley1": {"idx": int(v1_idx), "price": round(v1_price, 6)},
            "valley2": {"idx": int(v2_idx), "price": round(v2_price, 6)},
            "neckline": round(neckline, 6),
            "neck_slope": round(neck_slope, 8),
            "depth": round(float(head_depth), 6),
            "target": round(neckline - head_depth, 6),
            "stop": round(float(pH), 6),
            "entry_idx": int(iR),
            "entry_price": round(float(C[int(iR)]), 6),
        })

    # ── MODO 2: iH&S Tempo Real (CALL) ──
    for i in range(len(pivot_lows) - 1):
        iL, pL = pivot_lows[i]
        iH, pH = pivot_lows[i + 1]
        if ("L", iH) in seen_heads: continue
        if pH >= pL or iH - iL < min_spacing: continue
        head_depth = pL - pH
        if head_depth < min_depth: continue
        if iL >= trend_lookback:
            if float(C[iL]) >= float(C[iL - trend_lookback]): continue
        search_start = iH + min_spacing
        if search_start >= n: continue
        # Limitar busca: máx 3x a distância do braço esquerdo
        d_left = iH - iL
        search_end = min(n, iH + int(d_left * 3.5))
        region = L[search_start:search_end]
        if len(region) < 2: continue
        local_min_rel = int(np.argmin(region))
        iR = search_start + local_min_rel
        pR = float(L[iR])
        if abs(pL - pR) > tol or pR <= pH: continue
        if min(pL, pR) / max(pL, pR) < symmetry_min: continue
        if iR - iL > max_span: continue
        # Validar que iR é um pivot real (não apenas argmin da região)
        _pivot_check = min(3, iR - search_start, n - 1 - iR)
        if _pivot_check < 2: continue
        _is_pivot = all(L[iR] <= L[iR - j] for j in range(1, _pivot_check + 1)) and \
                    all(L[iR] <= L[iR + j] for j in range(1, min(_pivot_check + 1, n - iR)))
        if not _is_pivot: continue
        # Simetria temporal
        d_right = iR - iH
        if min(d_left, d_right) / max(d_left, d_right) < temporal_sym_min: continue
        if float(min(L[iH+1:n])) <= pH: continue
        v1_region = H[iL:iH + 1]
        v1_rel = int(np.argmax(v1_region)); v1_idx = iL + v1_rel; v1_price = float(v1_region[v1_rel])
        v2_region = H[iH:min(iR + 1, n)]
        v2_rel = int(np.argmax(v2_region)); v2_idx = iH + v2_rel; v2_price = float(v2_region[v2_rel])
        neckline = (v1_price + v2_price) / 2
        if abs(v1_price - v2_price) > atr * 0.5: continue
        neck_slope = (v2_price - v1_price) / max(1, v2_idx - v1_idx)
        patterns.append({
            "type": "INV_HEAD_SHOULDERS", "direction": "CALL", "mode": "realtime",
            "left_shoulder": {"idx": int(iL), "price": round(float(pL), 6)},
            "head": {"idx": int(iH), "price": round(float(pH), 6)},
            "right_shoulder": {"idx": int(iR), "price": round(float(pR), 6)},
            "valley1": {"idx": int(v1_idx), "price": round(v1_price, 6)},
            "valley2": {"idx": int(v2_idx), "price": round(v2_price, 6)},
            "neckline": round(neckline, 6),
            "neck_slope": round(neck_slope, 8),
            "depth": round(float(head_depth), 6),
            "target": round(neckline + head_depth, 6),
            "stop": round(float(pH), 6),
            "entry_idx": int(iR),
            "entry_price": round(float(C[int(iR)]), 6),
        })

    return patterns


# ══════════════════════════════════════════════════════════════════
# DETECÇÃO DOUBLE TOUCH (Duplo Toque)
# ══════════════════════════════════════════════════════════════════
def detect_double_touch(H, L, C_arr, O, pivot_highs, pivot_lows, atr, n,
                        max_candles_ago=9999, training=False):
    """Detecta Duplo Toque: preço toca o MESMO nível 2x + rejeição (wick).
    Double Top (PUT): 2 toques em resistência + wick rejeição → preço cai
    Double Bottom (CALL): 2 toques em suporte + wick rejeição → preço sobe
    """
    patterns = []
    tol = atr * 0.4
    min_spacing = 8
    max_spacing = 60
    min_depth = atr * 1.0
    min_candle_range = atr * 0.20

    # ═══ DOUBLE TOP (PUT) ═══
    for i, (idx1, price1) in enumerate(pivot_highs):
        if training or max_candles_ago >= 9999:
            for j_idx in range(i + 1, len(pivot_highs)):
                idx2, price2 = pivot_highs[j_idx]
                spacing = idx2 - idx1
                if spacing < min_spacing or spacing > max_spacing:
                    continue
                if abs(price1 - price2) > tol:
                    continue
                v_reg = L[idx1:idx2 + 1]
                if len(v_reg) < 3:
                    continue
                v_rel = int(np.argmin(v_reg))
                v_idx = idx1 + v_rel
                v_price = float(v_reg[v_rel])
                touch_level = max(float(price1), float(price2))
                depth = touch_level - v_price
                if depth < min_depth:
                    continue
                patterns.append({
                    "type": "DOUBLE_TOP", "direction": "PUT", "mode": "double_touch",
                    "left_shoulder": {"idx": int(idx1), "price": round(float(price1), 6)},
                    "head": {"idx": int(v_idx), "price": round(float(touch_level), 6)},
                    "right_shoulder": {"idx": int(idx2), "price": round(float(price2), 6)},
                    "valley1": {"idx": int(v_idx), "price": round(v_price, 6)},
                    "valley2": {"idx": int(v_idx), "price": round(v_price, 6)},
                    "neckline": round(v_price, 6),
                    "neck_slope": 0.0,
                    "depth": round(depth, 6),
                    "target": round(v_price, 6),
                    "stop": round(touch_level + atr * 0.3, 6),
                    "entry_idx": int(idx2),
                    "entry_price": round(float(C_arr[int(idx2)]), 6),
                    "candles_ago": n - 1 - idx2,
                })
        else:
            if n - 1 - idx1 < min_spacing:
                continue
            j_start = max(idx1 + min_spacing, n - 1 - max_candles_ago)
            for j in range(j_start, n):
                h_j, c_j, o_j, l_j = float(H[j]), float(C_arr[j]), float(O[j]), float(L[j])
                candle_range = h_j - l_j
                if candle_range < min_candle_range:
                    continue
                if h_j < price1 - tol or h_j > price1 + tol:
                    continue
                wick_up = h_j - max(c_j, o_j)
                if wick_up < candle_range * 0.35:
                    continue
                if c_j > l_j + candle_range * 0.40:
                    continue
                v_reg = L[idx1:j + 1]
                if len(v_reg) < 3:
                    continue
                v_rel = int(np.argmin(v_reg))
                v_idx = idx1 + v_rel
                v_price = float(v_reg[v_rel])
                touch_level = max(float(price1), h_j)
                depth = touch_level - v_price
                if depth < min_depth:
                    continue
                patterns.append({
                    "type": "DOUBLE_TOP", "direction": "PUT", "mode": "double_touch",
                    "left_shoulder": {"idx": int(idx1), "price": round(float(price1), 6)},
                    "head": {"idx": int(v_idx), "price": round(float(touch_level), 6)},
                    "right_shoulder": {"idx": int(j), "price": round(float(h_j), 6)},
                    "valley1": {"idx": int(v_idx), "price": round(v_price, 6)},
                    "valley2": {"idx": int(v_idx), "price": round(v_price, 6)},
                    "neckline": round(v_price, 6),
                    "neck_slope": 0.0,
                    "depth": round(depth, 6),
                    "target": round(v_price, 6),
                    "stop": round(touch_level + atr * 0.3, 6),
                    "entry_idx": int(j),
                    "entry_price": round(c_j, 6),
                    "candles_ago": n - 1 - j,
                })

    # ═══ DOUBLE BOTTOM (CALL) ═══
    for i, (idx1, price1) in enumerate(pivot_lows):
        if training or max_candles_ago >= 9999:
            for j_idx in range(i + 1, len(pivot_lows)):
                idx2, price2 = pivot_lows[j_idx]
                spacing = idx2 - idx1
                if spacing < min_spacing or spacing > max_spacing:
                    continue
                if abs(price1 - price2) > tol:
                    continue
                p_reg = H[idx1:idx2 + 1]
                if len(p_reg) < 3:
                    continue
                p_rel = int(np.argmax(p_reg))
                p_idx = idx1 + p_rel
                p_price = float(p_reg[p_rel])
                touch_level = min(float(price1), float(price2))
                depth = p_price - touch_level
                if depth < min_depth:
                    continue
                patterns.append({
                    "type": "DOUBLE_BOTTOM", "direction": "CALL", "mode": "double_touch",
                    "left_shoulder": {"idx": int(idx1), "price": round(float(price1), 6)},
                    "head": {"idx": int(p_idx), "price": round(float(touch_level), 6)},
                    "right_shoulder": {"idx": int(idx2), "price": round(float(price2), 6)},
                    "valley1": {"idx": int(p_idx), "price": round(p_price, 6)},
                    "valley2": {"idx": int(p_idx), "price": round(p_price, 6)},
                    "neckline": round(p_price, 6),
                    "neck_slope": 0.0,
                    "depth": round(depth, 6),
                    "target": round(p_price, 6),
                    "stop": round(touch_level - atr * 0.3, 6),
                    "entry_idx": int(idx2),
                    "entry_price": round(float(C_arr[int(idx2)]), 6),
                    "candles_ago": n - 1 - idx2,
                })
        else:
            if n - 1 - idx1 < min_spacing:
                continue
            j_start = max(idx1 + min_spacing, n - 1 - max_candles_ago)
            for j in range(j_start, n):
                h_j, c_j, o_j, l_j = float(H[j]), float(C_arr[j]), float(O[j]), float(L[j])
                candle_range = h_j - l_j
                if candle_range < min_candle_range:
                    continue
                if l_j > price1 + tol or l_j < price1 - tol:
                    continue
                wick_down = min(c_j, o_j) - l_j
                if wick_down < candle_range * 0.35:
                    continue
                if c_j < h_j - candle_range * 0.40:
                    continue
                p_reg = H[idx1:j + 1]
                if len(p_reg) < 3:
                    continue
                p_rel = int(np.argmax(p_reg))
                p_idx = idx1 + p_rel
                p_price = float(p_reg[p_rel])
                touch_level = min(float(price1), l_j)
                depth = p_price - touch_level
                if depth < min_depth:
                    continue
                patterns.append({
                    "type": "DOUBLE_BOTTOM", "direction": "CALL", "mode": "double_touch",
                    "left_shoulder": {"idx": int(idx1), "price": round(float(price1), 6)},
                    "head": {"idx": int(p_idx), "price": round(float(touch_level), 6)},
                    "right_shoulder": {"idx": int(j), "price": round(float(l_j), 6)},
                    "valley1": {"idx": int(p_idx), "price": round(p_price, 6)},
                    "valley2": {"idx": int(p_idx), "price": round(p_price, 6)},
                    "neckline": round(p_price, 6),
                    "neck_slope": 0.0,
                    "depth": round(depth, 6),
                    "target": round(p_price, 6),
                    "stop": round(touch_level - atr * 0.3, 6),
                    "entry_idx": int(j),
                    "entry_price": round(c_j, 6),
                    "candles_ago": n - 1 - j,
                })

    return patterns


# ══════════════════════════════════════════════════════════════════
# BACKTEST: verificar WIN/LOSS de cada padrão
# ══════════════════════════════════════════════════════════════════
def backtest_pattern(pat, C, O, H, L, n):
    """Verifica se o padrão H&S resultaria em WIN ou LOSS.
    
    Regra: entra no CLOSE da vela de confirmação (entry_idx = ombro direito).
    Verifica o close EXP_CANDLES velas depois.
    PUT: WIN se close < entry_price
    CALL: WIN se close > entry_price
    
    Também verifica guards do bot:
    - Preço não pode estar acima da cabeça (PUT) ou abaixo (CALL)
    - Preço não pode estar longe demais do ombro D
    """
    entry_idx = pat.get("entry_idx", pat["right_shoulder"]["idx"])
    
    if entry_idx >= n or entry_idx < 0:
        return None  # sem dados para verificar
    
    exit_idx = entry_idx + EXP_CANDLES
    if exit_idx >= n:
        return None  # padrão muito recente, sem resultado ainda

    # Entrada no close da vela de confirmação (ombro direito)
    entry_price = float(C[entry_idx])
    exit_price = float(C[exit_idx - 1])  # close da última vela
    
    head_price = pat["head"]["price"]
    rs_price = pat["right_shoulder"]["price"]
    
    # Verificar guards do bot
    if pat["direction"] == "PUT":
        # Guard: preço acima da cabeça = inválido
        if entry_price >= head_price:
            return {"result": "skip", "reason": "acima_cabeca"}
        win = exit_price < entry_price
    else:  # CALL
        if entry_price <= head_price:
            return {"result": "skip", "reason": "abaixo_cabeca"}
        win = exit_price > entry_price
    
    return {
        "result": "win" if win else "loss",
        "entry_price": round(entry_price, 6),
        "exit_price": round(exit_price, 6),
        "entry_idx": entry_idx,
        "exit_idx": exit_idx - 1,
        "pips": round(abs(exit_price - entry_price), 6),
    }


# ══════════════════════════════════════════════════════════════════
# IA SIMPLES: aprende quais setups dão WIN
# ══════════════════════════════════════════════════════════════════
class HS_IA:
    """IA que aprende padrões H&S por ativo e tipo."""
    
    def __init__(self):
        self.stats = {}  # {ativo: {type: {wins, total, features...}}}
        self.global_stats = {"wins": 0, "total": 0, "by_type": {}}
        self.geometry_history = []  # IA aprende geometria dos padrões
    
    def learn(self, ativo, pat, result, atr_val=0):
        """Registra resultado de um padrão."""
        if result["result"] not in ("win", "loss"):
            return
        
        key = f"{ativo}_{pat['type']}_{pat['mode']}"
        if key not in self.stats:
            self.stats[key] = {"wins": 0, "total": 0, "patterns": []}
        
        self.stats[key]["total"] += 1
        if result["result"] == "win":
            self.stats[key]["wins"] += 1
        
        # Stats globais
        self.global_stats["total"] += 1
        if result["result"] == "win":
            self.global_stats["wins"] += 1
        
        t = pat["type"]
        if t not in self.global_stats["by_type"]:
            self.global_stats["by_type"][t] = {"wins": 0, "total": 0}
        self.global_stats["by_type"][t]["total"] += 1
        if result["result"] == "win":
            self.global_stats["by_type"][t]["wins"] += 1
        
        # Guardar features para análise
        depth_atr = pat.get("depth", 0)
        self.stats[key]["patterns"].append({
            "result": result["result"],
            "depth": depth_atr,
            "mode": pat["mode"],
            "entry_price": result.get("entry_price", 0),
            "exit_price": result.get("exit_price", 0),
        })

        # ── IA: armazenar geometria para aprendizado contínuo ──
        geo = _extract_geometry(pat, atr_val)
        if geo is not None:
            geo["result"] = 1 if result["result"] == "win" else 0
            geo["ativo"] = ativo
            geo["type"] = pat.get("type", "HS")
            geo["source"] = "backtest"  # marcado como backtest (learn vem do treino)
            self.geometry_history.append(geo)
            if len(self.geometry_history) > 300:
                self.geometry_history = self.geometry_history[-300:]
    
    def predict(self, ativo, pat):
        """Prediz probabilidade de WIN para um setup — com fallback hierárquico ponderado."""
        import math
        key = f"{ativo}_{pat['type']}_{pat['mode']}"
        data = self.stats.get(key, None)
        if data and data["total"] >= 3:
            return (data["wins"] + 2) / (data["total"] + 4)  # Bayesian

        # Fallback hierárquico ponderado (não para no primeiro raso)
        pat_type = pat.get("type", "HS")
        pat_mode = pat.get("mode", "classic")

        # F1: mesmo tipo + modo
        f1_w, f1_t = 0, 0
        for k, v in self.stats.items():
            if f"_{pat_type}_{pat_mode}" in k:
                f1_w += v.get("wins", 0)
                f1_t += v.get("total", 0)
        # F2: mesmo tipo qualquer modo
        f2_w, f2_t = 0, 0
        for k, v in self.stats.items():
            if f"_{pat_type}_" in k:
                f2_w += v.get("wins", 0)
                f2_t += v.get("total", 0)
        # F3: global
        f3_t = self.global_stats.get("total", 0)
        f3_w = self.global_stats.get("wins", 0)

        candidates = []
        if f1_t >= 3:
            candidates.append(((f1_w + 2) / (f1_t + 4), f1_t))
        if f2_t >= 5:
            candidates.append(((f2_w + 2) / (f2_t + 4), f2_t))
        if f3_t >= 10:
            candidates.append(((f3_w + 2) / (f3_t + 4), f3_t))

        if candidates:
            total_weight = sum(math.sqrt(n) for _, n in candidates)
            return sum(p * math.sqrt(n) for p, n in candidates) / total_weight

        return 0.5  # sem NENHUM dado
    
    def get_summary(self):
        """Resumo global para o dashboard."""
        total = self.global_stats["total"]
        wins = self.global_stats["wins"]
        wr = (wins / total * 100) if total > 0 else 0
        
        by_type = {}
        for t, d in self.global_stats["by_type"].items():
            by_type[t] = {
                "wins": d["wins"], "total": d["total"],
                "wr": round(d["wins"] / d["total"] * 100, 1) if d["total"] > 0 else 0
            }
        
        # Top / worst ativos
        asset_stats = {}
        for key, d in self.stats.items():
            parts = key.split("_")
            ativo = parts[0]
            if ativo not in asset_stats:
                asset_stats[ativo] = {"wins": 0, "total": 0}
            asset_stats[ativo]["wins"] += d["wins"]
            asset_stats[ativo]["total"] += d["total"]
        
        for a in asset_stats:
            t = asset_stats[a]["total"]
            w = asset_stats[a]["wins"]
            asset_stats[a]["wr"] = round(w / t * 100, 1) if t > 0 else 0
        
        ia_level = _get_ia_level(total)
        return {
            "total": total, "wins": wins, "wr": round(wr, 1),
            "by_type": by_type,
            "by_asset": asset_stats,
            "ia_level": ia_level,
        }

    def get_training_stats(self):
        """Retorna stats detalhados para o bot importar no startup.
        Formato compatível com ai_predict() do bot: {arms: {key: {wins, total}}}"""
        arms = {}
        for key, d in self.stats.items():
            # key = "ATIVO_TYPE_MODE" no dashboard → converter para "ATIVO_TYPE" no bot
            parts = key.split("_")
            if len(parts) >= 2:
                bot_key = f"{parts[0]}_{parts[1]}"  # ATIVO_TYPE
            else:
                bot_key = key
            if bot_key not in arms:
                arms[bot_key] = {"wins": 0, "total": 0}
            arms[bot_key]["wins"] += d["wins"]
            arms[bot_key]["total"] += d["total"]
        return {
            "meta": {"total": self.global_stats["total"]},
            "arms": arms,
        }

    def save_to_disk(self, filepath=None):
        """Salva stats da IA para disco (persistência entre reinícios)."""
        filepath = filepath or IA_PERSIST_FILE
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            data = {"stats": {}, "global_stats": self.global_stats}
            for k, v in self.stats.items():
                entry = {"wins": v["wins"], "total": v["total"]}
                data["stats"][k] = entry
            # IA: persistir geometria aprendida
            data["geometry_history"] = self.geometry_history[-300:]
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(f"Erro salvando IA: {e}")

    def load_from_disk(self, filepath=None):
        """Carrega stats da IA do disco."""
        filepath = filepath or IA_PERSIST_FILE
        try:
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.get("stats", {}).items():
                    self.stats[k] = {"wins": v.get("wins", 0), "total": v.get("total", 0), "patterns": []}
                self.global_stats = data.get("global_stats", {"wins": 0, "total": 0, "by_type": {}})
                # IA: carregar geometria aprendida
                self.geometry_history = data.get("geometry_history", [])
                log.info(f"[IA] Stats carregadas do disco: {self.global_stats['total']} padrões | WR={self.global_stats['wins']/max(1,self.global_stats['total'])*100:.1f}% | geo={len(self.geometry_history)}")
                return True
        except Exception as e:
            log.warning(f"Erro carregando IA: {e}")
        return False


# ── Controle de retrain semanal ──
def _load_train_control():
    """Carrega dados de controle de treino."""
    try:
        if os.path.exists(TRAIN_CONTROL_FILE):
            with open(TRAIN_CONTROL_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_train_control():
    """Salva controle de treino com data/semana atual."""
    try:
        os.makedirs(os.path.dirname(TRAIN_CONTROL_FILE), exist_ok=True)
        now = datetime.now()
        iso = now.isocalendar()
        with open(TRAIN_CONTROL_FILE, "w") as f:
            json.dump({"iso_year": iso[0], "iso_week": iso[1], "date": now.isoformat()}, f)
        log.info(f"[RETRAIN] Controle salvo: semana {iso[1]}/{iso[0]}")
    except Exception as e:
        log.warning(f"Erro salvando controle de treino: {e}")


def _need_retrain():
    """Verifica se precisa retreinar (nova semana ISO)."""
    ctrl = _load_train_control()
    if not ctrl:
        return True
    now = datetime.now().isocalendar()
    return ctrl.get("iso_week") != now[1] or ctrl.get("iso_year") != now[0]


def _get_ia_level(n_total: int) -> dict:
    """Retorna nível da IA baseado no total de amostras."""
    if n_total == 0:
        return {"num": 1, "nome": "Iniciante", "emoji": "🌱", "cor": "#94a3b8"}
    elif n_total <= 10:
        return {"num": 2, "nome": "Aprendendo", "emoji": "📚", "cor": "#f59e0b"}
    elif n_total <= 30:
        return {"num": 3, "nome": "Calibrando", "emoji": "⚙️", "cor": "#3b82f6"}
    elif n_total <= 80:
        return {"num": 4, "nome": "Experiente", "emoji": "🧠", "cor": "#8b5cf6"}
    elif n_total <= 200:
        return {"num": 5, "nome": "Avançada", "emoji": "🎯", "cor": "#10b981"}
    else:
        return {"num": 6, "nome": "Expert", "emoji": "🏆", "cor": "#10b981"}


# ══════════════════════════════════════════════════════════════════
# DATA (leitura do cache do bot — SEM conexão ao broker)
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
# CACHE GLOBAL
# ══════════════════════════════════════════════════════════════════
_lock = threading.Lock()
_scanning = False  # True durante o scan pesado
_ia = HS_IA()
_selected_ativo = ""  # ativo selecionado no frontend
_cache = {
    "assets_data": {},          # {ativo: DataFrame}
    "assets_patterns": {},      # {ativo: [patterns with results]}
    "ia_summary": {},
    "payouts": {},
    "last_update": 0,
    "connected": False,
    "error": None,
    "scan_count": 0,
    "live_signals": [],         # sinais EM TEMPO REAL (padrões sem resultado ainda)
}

# ── Trades reais recebidos via POST do bot ──
_real_trades_lock = threading.Lock()
_real_trades: list = []  # entradas reais feitas pelo bot
_REAL_TRADES_MAX = 100


def _load_bot_trade_logs() -> list:
    """Lê os arquivos de log de trades reais do bot (todas as corretoras)."""
    entries = []
    for suffix in ("iq", "bullex", "casatrader"):
        fpath = os.path.join(_USER_DIR, f"ws_live_trades_{suffix}.json")
        try:
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for t in data.get("trades", []):
                    if t.get("status") in ("win", "loss", "entry", "tie"):
                        entries.append({
                            "ativo": t.get("ativo", "?"),
                            "dir": t.get("dir", "?"),
                            "result": t["status"],
                            "price": t.get("entry_price") or 0,
                            "stake": t.get("stake", 0),
                            "profit": t.get("resultado", 0),
                            "time": (t.get("time", "")[-8:-3] if t.get("time") else ""),
                            "ts": t.get("ts", 0),
                            "broker": t.get("broker", suffix),
                        })
        except Exception:
            pass
    # Deduplicar: se há um resultado (win/loss/tie) para um entry do mesmo ativo, manter só o resultado
    deduped = []
    seen_results = set()  # (ativo, ts_approx) já resolvidos
    # Primeiro pass: coletar todos os resultados
    for e in entries:
        if e["result"] in ("win", "loss", "tie"):
            seen_results.add(e["ativo"] + "_" + str(int(e.get("ts", 0) // 300)))
    # Segundo pass: filtrar entries que já têm resultado
    for e in entries:
        if e["result"] == "entry":
            key = e["ativo"] + "_" + str(int(e.get("ts", 0) // 300))
            if key in seen_results:
                continue  # pular entry duplicado — já temos o resultado
        deduped.append(e)
    # Ordenar por timestamp decrescente
    deduped.sort(key=lambda x: x.get("ts", 0), reverse=True)
    return deduped[:_REAL_TRADES_MAX]


def _load_candles_from_cache():
    """Lê dados de velas do cache compartilhado (escrito pelo bot).
    Retorna dict {ativo: DataFrame} e {ativo: payout}."""
    assets_data = {}
    payouts = {}
    try:
        if not os.path.exists(_DASHBOARD_CACHE_FILE):
            return assets_data, payouts
        with open(_DASHBOARD_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for ativo, adata in data.get("assets", {}).items():
            candles = adata.get("candles", [])
            if len(candles) < 50:
                continue
            df = pd.DataFrame(candles)
            df.rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close"}, inplace=True)
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.dropna(subset=["time", "open", "high", "low", "close"])
            df.set_index("time", inplace=True)
            df.sort_index(inplace=True)
            if "volume" not in df.columns:
                df["volume"] = 0
            assets_data[ativo] = df
            payouts[ativo] = adata.get("payout", 0)
    except Exception as e:
        log.debug(f"Erro ao ler cache do bot: {e}")
    return assets_data, payouts


def _signal_scan_thread():
    """Thread de detecção de sinais: roda a cada ~55s usando dados em cache.
    NÃO faz chamadas ao broker — usa dados já atualizados pelas outras threads.
    Mantém live_signals sempre frescos para o bot (scan_ts < 60s).
    """
    while True:
        try:
            # Esperar até segundo :35 do minuto (sinal pronto antes de :45)
            now_s = time.time() % 60
            wait_to_35 = (35 - now_s) % 60
            if wait_to_35 < 3:
                wait_to_35 += 60  # não executar imediatamente se já passou
            time.sleep(wait_to_35)

            with _lock:
                ad_copy = dict(_cache["assets_data"])
                payouts = dict(_cache.get("payouts", {}))

            if not ad_copy:
                continue  # nenhum dado ainda — aguardar heavy scan

            fresh_signals = []

            for ativo, df in ad_copy.items():
                if df is None or len(df) < 100:
                    continue

                H = df["high"].values
                L = df["low"].values
                C = df["close"].values
                O = df["open"].values
                n = len(C)

                # ATR
                atr_vals = [float(H[k] - L[k]) for k in range(max(0, n - 14), n)]
                atr = float(np.mean(atr_vals)) if atr_vals else 0.001

                # Detectar pivots e Double Touch (somente DT)
                ph, pl = detect_pivots(H, L, PIVOT_WINDOW)
                all_hs = detect_double_touch(H, L, C, O, ph, pl, atr, n, max_candles_ago=1)

                for pat in all_hs:
                    bt = backtest_pattern(pat, C, O, H, L, n)
                    if bt is None:
                        # Padrão recente sem resultado = sinal LIVE
                        entry_idx = pat.get("entry_idx", pat["right_shoulder"]["idx"] + 1)
                        pat["entry_pending"] = entry_idx >= n
                        rs_idx = pat["right_shoulder"]["idx"]
                        pat["candles_ago"] = max(0, n - 1 - rs_idx)
                        pat["scan_ts"] = time.time()  # timestamp FRESCO
                        ia_prob = _ia.predict(ativo, pat)
                        _pq = ia_pattern_quality(pat, atr, _ia.geometry_history)
                        ia_prob = round(ia_prob * _pq, 3)
                        pat["ia_prob"] = ia_prob
                        pat["ativo"] = ativo
                        fresh_signals.append(pat)

            # Atualizar cache com sinais frescos
            with _lock:
                _cache["live_signals"] = fresh_signals

            n_sig = len(fresh_signals)
            if n_sig > 0:
                log.info(f"[SIGNAL-SCAN] {n_sig} sinais frescos detectados (scan_ts atualizado)")
            else:
                log.debug("[SIGNAL-SCAN] Nenhum sinal live neste minuto")

        except Exception as e:
            log.debug(f"Signal scan error: {e}")
            time.sleep(10)


def _update_thread():
    """Thread principal: lê dados do cache do bot, detecta padrões, backtest, treina IA.
    O dashboard NÃO conecta ao broker — apenas lê o cache escrito pelo bot.
    Retrain semanal: limpa IA e retreina do zero 1x por semana.
    """
    global _ia, _scanning
    _first_cycle = True
    _last_cache_ts = 0
    
    while True:
        try:
            # ── Ler dados do cache do bot (NÃO conecta ao broker) ──
            if not os.path.exists(_DASHBOARD_CACHE_FILE):
                log.info("Aguardando bot escrever cache (ws_dashboard_cache.json)...")
                with _lock:
                    _cache["connected"] = False
                    _cache["error"] = "Aguardando bot iniciar scan..."
                time.sleep(10)
                continue

            # Verificar se cache foi atualizado
            try:
                _cache_mtime = os.path.getmtime(_DASHBOARD_CACHE_FILE)
            except Exception:
                _cache_mtime = 0
            if _cache_mtime <= _last_cache_ts:
                # Cache não mudou — aguardar
                time.sleep(2)
                continue
            _last_cache_ts = _cache_mtime

            # ── Retrain semanal (1x por semana) ──
            if _first_cycle:
                _first_cycle = False
                if _need_retrain():
                    log.info("=" * 60)
                    log.info("[RETRAIN] Nova semana detectada — LIMPANDO IA e retreinando do zero!")
                    log.info("=" * 60)
                    _ia = HS_IA()
                    try:
                        if os.path.exists(IA_PERSIST_FILE):
                            os.remove(IA_PERSIST_FILE)
                            log.info("[RETRAIN] Stats antigos removidos do disco")
                    except Exception:
                        pass
                else:
                    loaded = _ia.load_from_disk()
                    if loaded:
                        log.info("[IA] Stats da semana carregados — continuando acumulação")
                    else:
                        log.info("[IA] Sem stats salvos — treinando do zero")

            # ── Carregar dados do cache do bot ──
            bot_assets, payouts = _load_candles_from_cache()
            if not bot_assets:
                log.warning("Cache do bot vazio ou sem ativos")
                with _lock:
                    _cache["connected"] = False
                    _cache["error"] = "Cache do bot sem dados"
                time.sleep(10)
                continue

            with _lock:
                _cache["connected"] = True
                _cache["error"] = None
                _cache["payouts"] = payouts

            assets_patterns = {}
            live_signals = []
            _ia_new = HS_IA()

            log.info(f"Processando {len(bot_assets)} ativos do cache do bot...")
            _scanning = True

            for ativo, df in bot_assets.items():
                if df is None or len(df) < 100:
                    continue

                with _lock:
                    _cache["assets_data"][ativo] = df

                H = df["high"].values
                L = df["low"].values
                C = df["close"].values
                O = df["open"].values
                n = len(C)

                atr_vals = [float(H[k] - L[k]) for k in range(max(0, n-14), n)]
                atr = np.mean(atr_vals) if atr_vals else 0.001

                ph, pl = detect_pivots(H, L, PIVOT_WINDOW)
                all_hs = detect_double_touch(H, L, C, O, ph, pl, atr, n, max_candles_ago=9999, training=True)

                patterns_with_results = []
                for pat in all_hs:
                    bt = backtest_pattern(pat, C, O, H, L, n)
                    if bt is None:
                        entry_idx = pat.get("entry_idx", pat["right_shoulder"]["idx"] + 1)
                        pat["entry_pending"] = entry_idx >= n
                        rs_idx = pat["right_shoulder"]["idx"]
                        pat["candles_ago"] = max(0, n - 1 - rs_idx)
                        pat["scan_ts"] = time.time()
                        ia_prob = _ia.predict(ativo, pat)
                        _pq = ia_pattern_quality(pat, atr, _ia.geometry_history)
                        ia_prob = round(ia_prob * _pq, 3)
                        pat["ia_prob"] = ia_prob
                        pat["ativo"] = ativo
                        live_signals.append(pat)
                        patterns_with_results.append({**pat, "backtest": None, "ia_prob": ia_prob})
                    elif bt["result"] in ("win", "loss"):
                        _ia_new.learn(ativo, pat, bt, atr)
                        ia_prob = _ia.predict(ativo, pat)
                        patterns_with_results.append({**pat, "backtest": bt, "ia_prob": round(ia_prob, 3)})

                if patterns_with_results:
                    assets_patterns[ativo] = patterns_with_results
                    _w = sum(1 for p in patterns_with_results if (p.get('backtest') or {}).get('result') == 'win')
                    _l = sum(1 for p in patterns_with_results if (p.get('backtest') or {}).get('result') == 'loss')
                    _lv = sum(1 for p in patterns_with_results if p.get('backtest') is None)
                    log.info(f"  {ativo}: {len(all_hs)} padrões | {_w}W / {_l}L | Live: {_lv}")

            _scanning = False

            _ia = _ia_new
            summary = _ia.get_summary()

            _ia.save_to_disk()
            if summary.get("total", 0) > 0:
                _save_train_control()

            with _lock:
                _cache["assets_patterns"] = assets_patterns
                _cache["ia_summary"] = summary
                _cache["live_signals"] = live_signals
                _cache["last_update"] = time.time()
                _cache["scan_count"] += 1

            log.info(f"[IA] Total: {summary['total']} padrões | WR: {summary['wr']:.1f}% | "
                     f"Live: {len(live_signals)} sinais")

        except Exception as e:
            _scanning = False
            log.error(f"Erro: {e}", exc_info=True)
            with _lock:
                _cache["error"] = str(e)
                _cache["connected"] = False

        # Sleep 3s e verificar novamente se cache foi atualizado  
        time.sleep(3)


# ══════════════════════════════════════════════════════════════════
# JSON BUILDER
# ══════════════════════════════════════════════════════════════════
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, pd.Timestamp): return obj.isoformat()
        return super().default(obj)


def build_api_data():
    """Retorna dados completos para o frontend."""
    with _lock:
        ad = _cache["assets_data"]
        ap = _cache["assets_patterns"]
        summary = _cache["ia_summary"]
        live = _cache["live_signals"]
        payouts = _cache["payouts"]
        scan_count = _cache["scan_count"]
    
    charts = {}
    for ativo, df in ad.items():
        last_120 = df.tail(120)
        candles = []
        for ts, row in last_120.iterrows():
            candles.append({
                "t": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                "o": round(float(row["open"]), 6),
                "h": round(float(row["high"]), 6),
                "l": round(float(row["low"]), 6),
                "c": round(float(row["close"]), 6),
            })
        
        pats_data = ap.get(ativo, [])
        n_total = len(df)
        offset = n_total - 120  # para mapear índices do padrão para o gráfico
        
        # Mapear padrões para coordenadas do gráfico
        mapped_pats = []
        for p in pats_data:
            mp = dict(p)
            # Ajustar índices relativos aos 120 candles exibidos
            for key in ["left_shoulder", "head", "right_shoulder", "valley1", "valley2"]:
                if key in mp and mp[key]:
                    mp[key] = dict(mp[key])
                    mp[key]["chart_idx"] = mp[key]["idx"] - offset
            if "entry_idx" in mp:
                mp["entry_chart_idx"] = mp["entry_idx"] - offset
            if mp.get("backtest") and "entry_idx" in mp["backtest"]:
                mp["backtest"] = dict(mp["backtest"])
                mp["backtest"]["entry_chart_idx"] = mp["backtest"]["entry_idx"] - offset
                mp["backtest"]["exit_chart_idx"] = mp["backtest"]["exit_idx"] - offset
            mapped_pats.append(mp)
        
        charts[ativo] = {
            "candles": candles,
            "patterns": mapped_pats,
            "payout": payouts.get(ativo, 0),
            "n_candles": n_total,
        }
    
    # Live signals com IA prob
    live_mapped = []
    for s in live:
        live_mapped.append({
            "ativo": s.get("ativo", "?"),
            "type": s["type"],
            "direction": s["direction"],
            "mode": s.get("mode", "?"),
            "ia_prob": s.get("ia_prob", 0.5),
            "head_price": s["head"]["price"],
            "rs_price": s["right_shoulder"]["price"],
            "neckline": s.get("neckline", 0),
            "entry_pending": s.get("entry_pending", True),
            "candles_ago": s.get("candles_ago", 99),
            "scan_ts": s.get("scan_ts", 0),
            "target": s.get("target", 0),
            "stop": s.get("stop", 0),
        })
    
    # Broker entries: APENAS trades REAIS feitos pelo bot (lidos dos logs)
    broker_entries = _load_bot_trade_logs()
    # Mesclar com trades recebidos via POST (tempo real), sem duplicar
    # Cria set de chaves únicas dos trades já lidos do arquivo
    _seen = set()
    for be in broker_entries:
        _seen.add((be.get("ativo",""), int(be.get("ts", 0) // 60)))
    with _real_trades_lock:
        for rt in _real_trades:
            key = (rt.get("ativo",""), int(rt.get("ts", 0) // 60))
            if key not in _seen:
                broker_entries.append(rt)
                _seen.add(key)
    broker_entries.sort(key=lambda x: x.get("ts", 0), reverse=True)
    broker_entries = broker_entries[:50]

    # IA training stats para o bot importar no startup
    ia_training_stats = {}
    try:
        ia_training_stats = _ia.get_training_stats()
    except Exception:
        pass

    return {
        "charts": charts,
        "summary": summary,
        "live_signals": live_mapped,
        "broker_entries": broker_entries,
        "ia_training_stats": ia_training_stats,
        "scan_count": scan_count,
        "last_update": datetime.now().strftime("%H:%M:%S"),
        "is_premium": _IS_PREMIUM,
        "is_pro": _IS_PRO,
        "is_paid": _IS_PAID,
    }


# ══════════════════════════════════════════════════════════════════
# HTML DASHBOARD
# ══════════════════════════════════════════════════════════════════
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WS Trader — IA H&S Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4/dist/lightweight-charts.standalone.production.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg-primary:#0b0f19;--bg-secondary:#111827;--bg-card:#1a2035;--bg-hover:#1e2a45;
  --border:#1e293b;--border-light:#334155;
  --text-primary:#f1f5f9;--text-secondary:#94a3b8;--text-muted:#64748b;
  --accent:#3b82f6;--accent-glow:rgba(59,130,246,0.25);
  --green:#10b981;--green-bg:rgba(16,185,129,0.12);--green-glow:rgba(16,185,129,0.3);
  --red:#ef4444;--red-bg:rgba(239,68,68,0.12);--red-glow:rgba(239,68,68,0.3);
  --orange:#f59e0b;--orange-bg:rgba(245,158,11,0.12);--orange-glow:rgba(245,158,11,0.3);
  --purple:#8b5cf6;--purple-bg:rgba(139,92,246,0.12);
  --glass:rgba(30,41,59,0.5);
  --radius:12px;--radius-sm:8px;--radius-full:9999px;
}
body{background:var(--bg-primary);color:var(--text-primary);font-family:'Inter',system-ui,sans-serif;overflow:hidden;height:100vh;display:flex;flex-direction:column}
.icon-svg{display:inline-block;vertical-align:middle;width:14px;height:14px;fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}

/* ── HEADER ── */
.top-bar{background:linear-gradient(135deg,#111827 0%,#0f172a 100%);padding:12px 24px;display:flex;align-items:center;gap:16px;border-bottom:1px solid var(--border);flex-shrink:0;backdrop-filter:blur(12px)}
.logo{display:flex;align-items:center;gap:10px}
.logo-icon{width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,#3b82f6,#8b5cf6);display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:800;color:#fff;box-shadow:0 4px 15px rgba(59,130,246,0.3)}
.logo h1{font-size:15px;font-weight:700;color:var(--text-primary);letter-spacing:-0.3px}
.logo .sub{font-size:10px;color:var(--text-muted);font-weight:400;letter-spacing:0.5px}
.top-badges{display:flex;gap:8px;margin-left:20px}
.tbadge{padding:4px 12px;border-radius:var(--radius-full);font-size:11px;font-weight:600;border:1px solid var(--border);display:inline-flex;align-items:center;gap:5px}
.tbadge.online{background:var(--green-bg);border-color:var(--green);color:var(--green)}
.tbadge.scanning{background:var(--orange-bg);border-color:var(--orange);color:var(--orange);animation:tblink 2s infinite}
.tbadge.err{background:var(--red-bg);border-color:var(--red);color:var(--red)}
@keyframes tblink{0%,100%{opacity:1}50%{opacity:.5}}
.top-right{margin-left:auto;display:flex;align-items:center;gap:14px}
.top-time{font-size:12px;color:var(--text-secondary);font-weight:600;font-variant-numeric:tabular-nums;display:flex;align-items:center;gap:5px}
.clock-dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 1.5s infinite}
.candle-timer{display:inline-flex;align-items:center;gap:6px;background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-full);padding:4px 12px 4px 6px;font-size:11px;font-weight:700;color:var(--orange);font-variant-numeric:tabular-nums}
.candle-timer .ct-ring{position:relative;width:24px;height:24px}
.candle-timer .ct-ring svg{transform:rotate(-90deg)}
.candle-timer .ct-ring-bg{fill:none;stroke:var(--border);stroke-width:3}
.candle-timer .ct-ring-fg{fill:none;stroke:var(--orange);stroke-width:3;stroke-linecap:round;transition:stroke-dashoffset .3s linear,stroke .3s}
.candle-timer .ct-secs{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:800;color:var(--text-primary)}
.candle-timer .ct-label{font-size:10px;color:var(--text-muted);font-weight:500}
.candle-timer.urgent .ct-ring-fg{stroke:var(--red)}
.candle-timer.urgent{color:var(--red);animation:tblink 1s infinite}

/* ── STATS ROW ── */
.stats-row{display:flex;gap:10px;padding:10px 24px;background:var(--bg-secondary);border-bottom:1px solid var(--border);flex-shrink:0;flex-wrap:wrap}
.st{display:flex;flex-direction:column;background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-sm);padding:8px 16px;min-width:100px;position:relative;overflow:hidden}
.st::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.st.blue::before{background:linear-gradient(90deg,var(--accent),var(--purple))}
.st.green::before{background:var(--green)}
.st.red::before{background:var(--red)}
.st.yellow::before{background:var(--orange)}
.st .lbl{font-size:9px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.8px;margin-bottom:2px;display:flex;align-items:center;gap:4px}
.st .val{font-size:22px;font-weight:800;letter-spacing:-0.5px}
.st .val.blue{color:var(--accent)}
.st .val.green{color:var(--green)}
.st .val.red{color:var(--red)}
.st .val.yellow{color:var(--orange)}

/* ── LAYOUT ── */
.content{display:flex;flex:1;overflow:hidden;min-height:0}

/* ── LEFT SIDEBAR ── */
.sidebar{width:280px;min-width:280px;background:var(--bg-secondary);border-right:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden}
.sidebar-top{padding:12px 16px;background:var(--bg-card);border-bottom:1px solid var(--border)}
.sidebar-title{font-size:12px;font-weight:700;color:var(--text-secondary);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;display:flex;align-items:center;gap:6px}
.search-box{position:relative}
.search-box input{width:100%;background:var(--bg-primary);border:1px solid var(--border);border-radius:var(--radius-sm);padding:8px 12px 8px 34px;color:var(--text-primary);font-size:12px;font-family:inherit;outline:none;transition:border .2s}
.search-box input:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-glow)}
.search-box input::placeholder{color:var(--text-muted)}
.search-box .search-icon{position:absolute;left:10px;top:50%;transform:translateY(-50%);color:var(--text-muted);display:flex}
.asset-list{flex:1;overflow-y:auto;padding:4px 8px}
.asset-list::-webkit-scrollbar{width:4px}
.asset-list::-webkit-scrollbar-track{background:transparent}
.asset-list::-webkit-scrollbar-thumb{background:var(--border-light);border-radius:4px}
.asset-item{display:flex;align-items:center;padding:10px 12px;cursor:pointer;border-radius:var(--radius-sm);margin:2px 0;border:1px solid transparent;transition:all .15s}
.asset-item:hover{background:var(--bg-hover);border-color:var(--border)}
.asset-item.active{background:var(--accent-glow);border-color:var(--accent)}
.asset-item .a-left{flex:1;display:flex;flex-direction:column;gap:2px}
.asset-item .a-name{font-size:13px;font-weight:700;color:var(--text-primary);display:flex;align-items:center;gap:6px}
.asset-item .a-meta{font-size:10px;color:var(--text-muted);display:flex;gap:8px;align-items:center}
.asset-item .a-right{display:flex;flex-direction:column;align-items:flex-end;gap:3px}
.asset-item .a-payout{font-size:11px;font-weight:700;color:var(--green)}
.wr-pill{font-size:10px;padding:2px 8px;border-radius:var(--radius-full);font-weight:700}
.wr-pill.good{background:var(--green-bg);color:var(--green)}
.wr-pill.mid{background:var(--orange-bg);color:var(--orange)}
.wr-pill.bad{background:var(--red-bg);color:var(--red)}
.live-dot{width:8px;height:8px;border-radius:50%;background:var(--orange);display:inline-block;animation:pulse 1.5s infinite;box-shadow:0 0 6px var(--orange-glow)}
.ia-badge{display:inline-flex;align-items:center;gap:3px;background:var(--purple-bg);color:var(--purple);font-size:9px;font-weight:700;padding:2px 6px;border-radius:var(--radius-full)}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.7)}}

/* ── CENTER: CHART ── */
.main-area{flex:1;display:flex;flex-direction:column;overflow:hidden}
.chart-toolbar{padding:10px 20px;background:var(--bg-secondary);border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;flex-shrink:0}
.chart-toolbar .ct-left{display:flex;align-items:center;gap:12px}
.chart-toolbar .ct-name{font-weight:800;color:var(--text-primary);font-size:18px;letter-spacing:-0.3px}
.chart-toolbar .ct-payout{background:var(--green-bg);color:var(--green);padding:3px 10px;border-radius:var(--radius-full);font-size:12px;font-weight:700}
.chart-toolbar .ct-dir{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:var(--radius-full);font-size:11px;font-weight:700}
.ct-dir.put{background:var(--red-bg);color:var(--red)}
.ct-dir.call{background:var(--green-bg);color:var(--green)}
.ct-right{display:flex;align-items:center;gap:12px}
.ct-info{font-size:11px;color:var(--text-muted);display:flex;align-items:center;gap:5px}
.ia-entry-icon{display:inline-flex;align-items:center;gap:6px;background:linear-gradient(135deg,#8b5cf6,#6366f1);color:#fff;padding:6px 14px;border-radius:var(--radius-full);font-size:12px;font-weight:700;box-shadow:0 4px 15px rgba(139,92,246,0.3)}
#main-chart-box{flex:1;min-height:0;position:relative}
.pat-footer{padding:8px 20px;background:var(--bg-card);border-top:1px solid var(--border);max-height:120px;overflow-y:auto;font-size:11px;flex-shrink:0}
.pat-row{display:flex;align-items:center;justify-content:space-between;padding:4px 0;border-bottom:1px solid var(--border)}
.pat-row:last-child{border:none}
.pat-row .pr-type{color:var(--text-secondary);font-weight:500;display:flex;align-items:center;gap:4px}
.pat-row .pr-ia{color:var(--purple);font-weight:700;display:flex;align-items:center;gap:3px}
.pat-row .pr-res{font-weight:700;display:flex;align-items:center;gap:4px}
.pr-res.win{color:var(--green)} .pr-res.loss{color:var(--red)} .pr-res.live{color:var(--orange)} .pr-res.skip{color:var(--text-muted)}
.empty-state{display:flex;flex-direction:column;align-items:center;justify-content:center;flex:1;color:var(--text-muted);gap:16px}
.empty-state .e-icon{opacity:.25}
.empty-state .e-text{font-size:15px;font-weight:500}
.empty-state .e-sub{font-size:12px;max-width:300px;text-align:center;line-height:1.6}

/* ── RIGHT SIDEBAR: LIVE + RESULTS ── */
.right-panel{width:320px;min-width:320px;background:var(--bg-secondary);border-left:1px solid var(--border);display:flex;flex-direction:column;min-height:0}
.rp-section{flex:0 0 auto;border-bottom:1px solid var(--border)}
.rp-section.entries-section{flex:1 1 0px;display:flex;flex-direction:column;overflow:hidden;border-bottom:none;min-height:0}
.rp-header{padding:10px 14px;display:flex;align-items:center;gap:8px;font-size:11px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.8px;background:var(--bg-card);flex-shrink:0}
.rp-body{padding:8px 10px;max-height:120px;overflow-y:auto}
.rp-body.entries-body{padding:8px 10px;max-height:none;flex:1 1 0px;overflow-y:auto;min-height:0}
.rp-body::-webkit-scrollbar{width:3px}
.rp-body::-webkit-scrollbar-thumb{background:var(--border-light);border-radius:3px}
.signal-card{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-sm);padding:10px 12px;margin:4px 0;cursor:pointer;transition:all .15s}
.signal-card:hover{border-color:var(--accent);transform:translateY(-1px)}
.signal-card .sc-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.signal-card .sc-name{font-weight:700;font-size:12px;color:var(--text-primary)}
.signal-card .sc-dir{font-size:10px;font-weight:700;padding:2px 8px;border-radius:var(--radius-full)}
.sc-dir.put{background:var(--red-bg);color:var(--red)} .sc-dir.call{background:var(--green-bg);color:var(--green)}
.signal-card .sc-bottom{display:flex;justify-content:space-between;align-items:center}
.signal-card .sc-type{font-size:10px;color:var(--text-muted)}
.signal-card .sc-prob{display:flex;align-items:center;gap:4px;font-size:11px;font-weight:700;color:var(--purple)}
.prob-bar{width:40px;height:4px;border-radius:2px;background:var(--bg-primary);overflow:hidden}
.prob-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--purple),var(--accent))}
.result-row{display:flex;align-items:center;padding:8px 10px;border-radius:var(--radius-sm);margin:3px 0;font-size:11px;background:var(--bg-card);border:1px solid var(--border);gap:8px}
.result-row .rr-ativo{flex:1;font-weight:600;color:var(--text-primary)}
.result-row .rr-dir{font-size:10px;font-weight:700}
.result-row .rr-price{font-size:9px;color:var(--text-muted);font-variant-numeric:tabular-nums}
.result-row .rr-profit{font-size:10px;font-weight:700;font-variant-numeric:tabular-nums}
.result-row .rr-res{font-weight:800;font-size:12px;display:flex;align-items:center;gap:3px}
.result-row.win{border-left:3px solid var(--green)} .result-row.win .rr-res{color:var(--green)} .result-row.win .rr-profit{color:var(--green)}
.result-row.loss{border-left:3px solid var(--red)} .result-row.loss .rr-res{color:var(--red)} .result-row.loss .rr-profit{color:var(--red)}
.result-row.entry{border-left:3px solid var(--purple)} .result-row.entry .rr-res{color:var(--purple)} .result-row.entry .rr-profit{color:var(--purple)}
.result-row .rr-broker{font-size:8px;color:var(--text-muted);text-transform:uppercase;font-weight:600;letter-spacing:0.5px}

/* ── FOOTER ── */
.footer{text-align:center;padding:6px;color:var(--text-muted);font-size:10px;border-top:1px solid var(--border);flex-shrink:0;background:var(--bg-secondary);font-weight:500;display:flex;align-items:center;justify-content:center;gap:6px}
</style>
</head>
<body>

<!-- SVG Icons (hidden sprite) -->
<svg style="display:none" xmlns="http://www.w3.org/2000/svg">
  <symbol id="i-search" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></symbol>
  <symbol id="i-zap" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></symbol>
  <symbol id="i-chart" viewBox="0 0 24 24"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></symbol>
  <symbol id="i-activity" viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></symbol>
  <symbol id="i-brain" viewBox="0 0 24 24"><path d="M12 2a7 7 0 0 1 7 7c0 2.38-1.19 4.47-3 5.74V17a2 2 0 0 1-2 2h-4a2 2 0 0 1-2-2v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 0 1 7-7z"/><line x1="9" y1="22" x2="15" y2="22"/><line x1="10" y1="19" x2="10" y2="22"/><line x1="14" y1="19" x2="14" y2="22"/></symbol>
  <symbol id="i-trending" viewBox="0 0 24 24"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></symbol>
  <symbol id="i-clock" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></symbol>
  <symbol id="i-target" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></symbol>
  <symbol id="i-check" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></symbol>
  <symbol id="i-x" viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></symbol>
  <symbol id="i-wifi" viewBox="0 0 24 24"><path d="M5 12.55a11 11 0 0 1 14.08 0"/><path d="M1.42 9a16 16 0 0 1 21.16 0"/><path d="M8.53 16.11a6 6 0 0 1 6.95 0"/><circle cx="12" cy="20" r="1"/></symbol>
  <symbol id="i-layers" viewBox="0 0 24 24"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></symbol>
  <symbol id="i-arrow-down" viewBox="0 0 24 24"><line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/></symbol>
  <symbol id="i-arrow-up" viewBox="0 0 24 24"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></symbol>
  <symbol id="i-candlestick" viewBox="0 0 24 24"><rect x="4" y="8" width="4" height="10" rx="1"/><line x1="6" y1="4" x2="6" y2="8"/><line x1="6" y1="18" x2="6" y2="22"/><rect x="14" y="4" width="4" height="10" rx="1"/><line x1="16" y1="1" x2="16" y2="4"/><line x1="16" y1="14" x2="16" y2="20"/></symbol>
</svg>

<!-- TOP BAR -->
<div class="top-bar">
  <div class="logo">
    <div class="logo-icon">W</div>
    <div>
      <h1>WS Trader</h1>
      <div class="sub">IA Head & Shoulders</div>
    </div>
  </div>
  <div class="top-badges">
    <span class="tbadge scanning" id="badge-status"><svg class="icon-svg" style="width:12px;height:12px"><use href="#i-wifi"/></svg> Conectando...</span>
    <span class="tbadge" id="badge-scan" style="border-color:var(--accent);color:var(--accent)"><svg class="icon-svg" style="width:12px;height:12px"><use href="#i-layers"/></svg> Scan #0</span>
  </div>
  <div class="top-right">
    <div class="candle-timer" id="candle-timer">
      <div class="ct-ring">
        <svg width="24" height="24" viewBox="0 0 24 24"><circle class="ct-ring-bg" cx="12" cy="12" r="10"/><circle class="ct-ring-fg" id="ct-ring-fg" cx="12" cy="12" r="10" stroke-dasharray="62.83" stroke-dashoffset="0"/></svg>
        <span class="ct-secs" id="ct-secs">60</span>
      </div>
      <span><span id="ct-countdown">0:60</span><br><span class="ct-label">Vela M1</span></span>
    </div>
    <span class="top-time" id="live-clock"><span class="clock-dot"></span> --:--:--</span>
    <span style="font-size:10px;color:var(--text-muted)" id="last-update">--</span>
  </div>
</div>

<!-- STATS ROW -->
<div class="stats-row" id="stats-bar">
  <div class="st blue"><div class="lbl"><svg class="icon-svg" style="width:10px;height:10px"><use href="#i-layers"/></svg> Padroes</div><div class="val blue" id="st-total">0</div></div>
  <div class="st green"><div class="lbl"><svg class="icon-svg" style="width:10px;height:10px"><use href="#i-target"/></svg> Win Rate</div><div class="val green" id="st-wr">0%</div></div>
  <div class="st green"><div class="lbl"><svg class="icon-svg" style="width:10px;height:10px"><use href="#i-check"/></svg> Wins</div><div class="val green" id="st-wins">0</div></div>
  <div class="st red"><div class="lbl"><svg class="icon-svg" style="width:10px;height:10px"><use href="#i-x"/></svg> Losses</div><div class="val red" id="st-losses">0</div></div>
  <div class="st blue"><div class="lbl"><svg class="icon-svg" style="width:10px;height:10px"><use href="#i-arrow-down"/></svg> H&S PUT</div><div class="val blue" id="st-hs" style="font-size:14px">-</div></div>
  <div class="st blue"><div class="lbl"><svg class="icon-svg" style="width:10px;height:10px"><use href="#i-arrow-up"/></svg> iH&S CALL</div><div class="val blue" id="st-ihs" style="font-size:14px">-</div></div>
  <div class="st yellow"><div class="lbl"><svg class="icon-svg" style="width:10px;height:10px"><use href="#i-zap"/></svg> Ao Vivo</div><div class="val yellow" id="st-live">0</div></div>
  <div class="st" id="st-ia-level-box" style="border-color:var(--text-muted)"><div class="lbl"><svg class="icon-svg" style="width:10px;height:10px"><use href="#i-brain"/></svg> IA Nível</div><div class="val" id="st-ia-level" style="font-size:13px;color:var(--text-muted)">🌱 1 - Iniciante</div></div>
</div>

<!-- MAIN CONTENT -->
<div class="content">

  <!-- LEFT SIDEBAR: ASSET LIST -->
  <div class="sidebar">
    <div class="sidebar-top">
      <div class="sidebar-title"><svg class="icon-svg" style="width:13px;height:13px"><use href="#i-candlestick"/></svg> Ativos</div>
      <div class="search-box">
        <span class="search-icon"><svg class="icon-svg" style="width:13px;height:13px"><use href="#i-search"/></svg></span>
        <input type="text" id="asset-search" placeholder="Buscar ativo..." oninput="filterAssets(this.value)">
      </div>
    </div>
    <div class="asset-list" id="asset-list"></div>
  </div>

  <!-- CENTER: CHART -->
  <div class="main-area">
    <div class="chart-toolbar" id="chart-toolbar" style="display:none">
      <div class="ct-left">
        <span class="ct-name" id="ct-name">--</span>
        <span class="ct-payout" id="ct-payout"></span>
        <span class="ct-dir" id="ct-dir" style="display:none"></span>
      </div>
      <div class="ct-right">
        <span class="ct-info" id="ct-info"><svg class="icon-svg" style="width:12px;height:12px"><use href="#i-candlestick"/></svg></span>
        <span class="ia-entry-icon" id="ia-entry" style="display:none">
          <svg class="icon-svg" style="width:16px;height:16px;stroke:#fff"><use href="#i-brain"/></svg>
          <span id="ia-entry-txt">IA 85%</span>
        </span>
      </div>
    </div>
    <div id="main-chart-box">
      <div class="empty-state" id="empty-state">
        <div class="e-icon"><svg style="width:64px;height:64px;stroke:var(--text-muted);fill:none;stroke-width:1.5"><use href="#i-candlestick"/></svg></div>
        <div class="e-text">Selecione um ativo na lista</div>
        <div class="e-sub">Clique em um ativo para visualizar o grafico de velas com os padroes H&S detectados pela IA</div>
      </div>
    </div>
    <div class="pat-footer" id="pat-footer" style="display:none"></div>
  </div>

  <!-- RIGHT SIDEBAR: LIVE + RESULTS -->
  <div class="right-panel">
    <div class="rp-section">
      <div class="rp-header"><svg class="icon-svg" style="width:13px;height:13px;stroke:var(--orange)"><use href="#i-zap"/></svg> Sinais ao Vivo</div>
      <div class="rp-body" id="live-list">
        <div style="color:var(--text-muted);font-size:11px;text-align:center;padding:20px 0">Aguardando sinais...</div>
      </div>
    </div>
    <div class="rp-section entries-section">
      <div class="rp-header"><svg class="icon-svg" style="width:13px;height:13px;stroke:var(--green)"><use href="#i-activity"/></svg> Entradas na Corretora</div>
      <div class="rp-body entries-body" id="results-list">
        <div style="color:var(--text-muted);font-size:11px;text-align:center;padding:20px 0">Sem entradas ainda</div>
      </div>
    </div>
  </div>

</div>

<div class="footer"><svg class="icon-svg" style="width:11px;height:11px"><use href="#i-activity"/></svg> WS Trader v5.2 — IA H&S — Velas ao vivo a cada 1s</div>

<script>
let mainChart = null, mainSeries = null, selectedAtivo = null, latestData = null, candleData = [], allAtivos = [];
let firstRender = true;

function parseTime(t) {
  if (typeof t === 'number') return t;
  if (typeof t === 'string' && /^\d+$/.test(t)) return parseInt(t, 10);
  /* ISO string — append Z if no timezone to force UTC */
  if (typeof t === 'string' && t.indexOf('Z') < 0 && t.indexOf('+') < 0 && !/\d{2}:\d{2}$/.test(t.slice(-5))) t = t + 'Z';
  var d = new Date(t); return Math.floor(d.getTime() / 1000);
}

/* Live clock + candle countdown */
var CIRC = 2 * Math.PI * 10; /* 62.83 */
function tickClock() {
  var now = new Date();
  var h = String(now.getHours()).padStart(2,'0');
  var m = String(now.getMinutes()).padStart(2,'0');
  var s = String(now.getSeconds()).padStart(2,'0');
  document.getElementById('live-clock').innerHTML = '<span class="clock-dot"></span> ' + h + ':' + m + ':' + s;

  /* Candle countdown: segundos restantes no minuto atual */
  var secsLeft = 60 - now.getSeconds();
  if (secsLeft === 60) secsLeft = 0; /* exatamente :00 = vela nova */
  var pct = secsLeft / 60;
  var offset = CIRC * (1 - pct);
  document.getElementById('ct-secs').textContent = secsLeft;
  document.getElementById('ct-countdown').textContent = '0:' + String(secsLeft).padStart(2,'0');
  document.getElementById('ct-ring-fg').setAttribute('stroke-dashoffset', offset.toFixed(2));
  var timer = document.getElementById('candle-timer');
  if (secsLeft <= 10 && secsLeft > 0) { timer.classList.add('urgent'); } else { timer.classList.remove('urgent'); }
}
setInterval(tickClock, 1000);
tickClock();

function filterAssets(query) {
  var q = query.toLowerCase().trim();
  document.querySelectorAll('.asset-item').forEach(function(el) {
    var name = (el.getAttribute('data-ativo') || '').toLowerCase();
    el.style.display = (!q || name.includes(q)) ? '' : 'none';
  });
}

function initChart() {
  var el = document.getElementById('main-chart-box');
  if (mainChart) { mainChart.remove(); mainChart = null; mainSeries = null; }
  document.getElementById('empty-state').style.display = 'none';
  mainChart = LightweightCharts.createChart(el, {
    width: el.clientWidth, height: el.clientHeight,
    layout: { background: { color: '#0b0f19' }, textColor: '#64748b', fontFamily: 'Inter, system-ui, sans-serif' },
    grid: { vertLines: { color: '#1e293b' }, horzLines: { color: '#1e293b' } },
    crosshair: { mode: 0, vertLine: { color: 'rgba(59,130,246,0.3)', width: 1 }, horzLine: { color: 'rgba(59,130,246,0.3)', width: 1 } },
    timeScale: { timeVisible: true, secondsVisible: false, borderColor: '#1e293b' },
    rightPriceScale: { borderColor: '#1e293b' },
  });
  mainSeries = mainChart.addCandlestickSeries({
    upColor: '#10b981', downColor: '#ef4444',
    wickUpColor: '#10b981', wickDownColor: '#ef4444',
    borderVisible: false,
  });
  mainChart.timeScale().subscribeVisibleTimeRangeChange(function() { requestAnimationFrame(drawHSOverlay); });
  new ResizeObserver(function() {
    if (mainChart) { mainChart.applyOptions({ width: el.clientWidth, height: el.clientHeight }); requestAnimationFrame(drawHSOverlay); }
  }).observe(el);
}

function selectAsset(ativo) {
  if (selectedAtivo !== ativo || !mainChart) {
    selectedAtivo = ativo;
    firstRender = true;
    initChart();
    /* Reiniciar streaming de velas para novo ativo */
    if (typeof startLiveCandles === 'function') startLiveCandles();
  }
  document.querySelectorAll('.asset-item').forEach(function(el) { el.classList.remove('active'); });
  var itemEl = document.getElementById('ai-' + ativo.replace(/[^a-zA-Z0-9]/g, '_'));
  if (itemEl) { itemEl.classList.add('active'); itemEl.scrollIntoView({ block: 'nearest' }); }
  if (latestData) renderChart(latestData);
}

function renderChart(data) {
  if (!selectedAtivo || !mainChart || !mainSeries) return;
  var cdata = (data.charts || {})[selectedAtivo];
  if (!cdata) return;

  // Toolbar
  document.getElementById('chart-toolbar').style.display = '';
  document.getElementById('ct-name').textContent = selectedAtivo;
  document.getElementById('ct-payout').textContent = cdata.payout + '%';
  document.getElementById('ct-info').innerHTML = '<svg class="icon-svg" style="width:12px;height:12px"><use href="#i-candlestick"/></svg> ' + cdata.n_candles + ' velas';

  // IA entry badge
  var livePats = (cdata.patterns || []).filter(function(p) { return !p.backtest; });
  var iaEntry = document.getElementById('ia-entry');
  var ctDir = document.getElementById('ct-dir');
  if (livePats.length > 0) {
    var best = livePats.reduce(function(a, b) { return (a.ia_prob || 0) > (b.ia_prob || 0) ? a : b; });
    iaEntry.style.display = '';
    document.getElementById('ia-entry-txt').textContent = 'IA ' + ((best.ia_prob||0.5)*100).toFixed(0) + '%';
    ctDir.style.display = '';
    ctDir.className = 'ct-dir ' + (best.direction === 'PUT' ? 'put' : 'call');
    ctDir.innerHTML = '<svg class="icon-svg" style="width:12px;height:12px"><use href="#i-arrow-' + (best.direction==='PUT'?'down':'up') + '"/></svg> ' + best.direction;
  } else {
    iaEntry.style.display = 'none';
    ctDir.style.display = 'none';
  }

  // Candles — setData no primeiro render ou troca de ativo.
  // Depois, usa update() para manter em sincronia sem resetar o gráfico.
  // Se o streaming live atualizou recentemente, NÃO sobrescrever com dados do scan (stale).
  var _skipCandleUpdate = (_liveUpdateTs > 0 && (Date.now() - _liveUpdateTs) < 15000);
  var newCandles = (cdata.candles || []).map(function(c) {
    return { time: parseTime(c.t), open: c.o, high: c.h, low: c.l, close: c.c };
  });
  var needFullSet = firstRender || candleData.length === 0;
  if (!needFullSet && newCandles.length > 0 && candleData.length > 0) {
    /* Trocar apenas se ativo mudou (timestamps muito diferentes) */
    var lastOld = candleData[0] ? candleData[0].time : 0;
    var lastNew = newCandles[0] ? newCandles[0].time : 0;
    if (Math.abs(lastOld - lastNew) > 120) needFullSet = true;
  }
  if (needFullSet) {
    candleData = newCandles;
    mainSeries.setData(candleData);
  } else if (newCandles.length > 0 && !_skipCandleUpdate) {
    /* Atualizar as últimas 5 velas via update() — só se live stream NÃO atualizou recentemente */
    var lastFive = newCandles.slice(-5);
    lastFive.forEach(function(bar) {
      mainSeries.update(bar);
      var found = false;
      for (var i = candleData.length - 1; i >= 0; i--) {
        if (candleData[i].time === bar.time) { candleData[i] = bar; found = true; break; }
      }
      if (!found) candleData.push(bar);
    });
  }
  if (firstRender) {
    mainChart.timeScale().fitContent();
    firstRender = false;
  }
  setTimeout(drawHSOverlay, 100);

  // Pattern list footer
  var patEl = document.getElementById('pat-footer');
  var visible = (cdata.patterns || []).filter(function(p) { return p.right_shoulder && p.right_shoulder.chart_idx >= 0; });
  if (visible.length > 0) {
    patEl.style.display = '';
    patEl.innerHTML = visible.map(function(p) {
      var bt = p.backtest;
      var cls = 'live', icoRef = '#i-clock', txt = 'LIVE';
      if (bt) {
        if (bt.result === 'win') { cls = 'win'; icoRef = '#i-check'; txt = 'WIN'; }
        else if (bt.result === 'loss') { cls = 'loss'; icoRef = '#i-x'; txt = 'LOSS'; }
        else { cls = 'skip'; icoRef = '#i-arrow-down'; txt = 'SKIP'; }
      }
      var typeName = p.type === 'DOUBLE_TOP' ? 'DT \u25BC' : p.type === 'DOUBLE_BOTTOM' ? 'DB \u25B2' : (p.type === 'HEAD_SHOULDERS' ? 'H&S' : 'iH&S');
      return '<div class="pat-row"><span class="pr-type"><svg class="icon-svg" style="width:11px;height:11px"><use href="#i-activity"/></svg> ' + typeName + ' ' + p.mode + '</span><span class="pr-ia"><svg class="icon-svg" style="width:11px;height:11px;stroke:var(--purple)"><use href="#i-brain"/></svg> ' + ((p.ia_prob||0.5)*100).toFixed(0) + '%</span><span class="pr-res ' + cls + '"><svg class="icon-svg" style="width:11px;height:11px"><use href="' + icoRef + '"/></svg> ' + txt + '</span></div>';
    }).join('');
  } else {
    patEl.style.display = 'none';
  }
}

function drawHSOverlay() {
  var box = document.getElementById('main-chart-box');
  if (!box) return;
  var canvas = document.getElementById('hs-overlay');
  if (!canvas) {
    canvas = document.createElement('canvas');
    canvas.id = 'hs-overlay';
    canvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:10';
    box.appendChild(canvas);
  }
  var r = box.getBoundingClientRect();
  var dpr = window.devicePixelRatio || 1;
  canvas.width = r.width * dpr; canvas.height = r.height * dpr;
  canvas.style.width = r.width + 'px'; canvas.style.height = r.height + 'px';
  var ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, r.width, r.height);
  if (!latestData || !selectedAtivo || !mainChart || !mainSeries || !candleData.length) return;
  var cd = (latestData.charts || {})[selectedAtivo];
  if (!cd || !cd.patterns) return;
  var ts = mainChart.timeScale();
  function gx(i) { if (i < 0 || i >= candleData.length) return null; return ts.timeToCoordinate(candleData[i].time); }
  function gy(p) { return mainSeries.priceToCoordinate(p); }

  cd.patterns.forEach(function(pat) {
    var ls = pat.left_shoulder, hd = pat.head, rs = pat.right_shoulder, v1 = pat.valley1, v2 = pat.valley2;
    if (!ls || !hd || !rs) return;
    var lsi = ls.chart_idx, hdi = hd.chart_idx, rsi = rs.chart_idx;
    var v1i = v1 ? v1.chart_idx : -1, v2i = v2 ? v2.chart_idx : -1;
    if (lsi < 0 || hdi < 0 || rsi < 0) return;
    var lsx = gx(lsi), lsy = gy(ls.price), hdx = gx(hdi), hdy = gy(hd.price), rsx = gx(rsi), rsy = gy(rs.price);
    if ([lsx,lsy,hdx,hdy,rsx,rsy].some(function(v){return v===null||isNaN(v)})) return;

    var isDT = pat.mode === 'double_touch';
    var isBear = pat.type === 'HEAD_SHOULDERS' || pat.type === 'DOUBLE_TOP';
    var mainC = isDT ? '#a855f7' : (isBear ? '#ef4444' : '#10b981');
    var mainCa = isDT ? 'rgba(168,85,247,0.15)' : (isBear ? 'rgba(239,68,68,0.15)' : 'rgba(16,185,129,0.15)');

    // ── Fill area (shoulder-to-shoulder shape) ──
    var hasV = v1i >= 0 && v2i >= 0;
    var v1x, v1y, v2x, v2y;
    if (hasV) {
      v1x = gx(v1i); v1y = gy(v1.price); v2x = gx(v2i); v2y = gy(v2.price);
      if ([v1x,v1y,v2x,v2y].some(function(v){return v===null||isNaN(v)})) hasV = false;
    }

    if (hasV) {
      ctx.fillStyle = mainCa; ctx.beginPath();
      if (isDT) {
        // DT: draw simple shape Touch1 → Valley → Touch2
        ctx.moveTo(lsx, lsy); ctx.lineTo(v1x, v1y); ctx.lineTo(rsx, rsy);
        ctx.closePath(); ctx.fill();
        // Lines: Touch1 → Valley → Touch2
        ctx.strokeStyle = mainC; ctx.lineWidth = 2.5; ctx.setLineDash([]); ctx.globalAlpha = 0.9;
        ctx.beginPath(); ctx.moveTo(lsx, lsy); ctx.lineTo(v1x, v1y); ctx.lineTo(rsx, rsy);
        ctx.stroke(); ctx.globalAlpha = 1;
        // Horizontal touch level line (resistance/support)
        var touchY = gy(pat.head.price);
        if (touchY !== null && !isNaN(touchY)) {
          var tL = gx(Math.max(0, lsi - 3)) || lsx;
          var tR = gx(Math.min(candleData.length - 1, rsi + 5)) || rsx;
          ctx.strokeStyle = isDT ? 'rgba(168,85,247,0.7)' : 'rgba(59,130,246,0.7)';
          ctx.lineWidth = 2; ctx.setLineDash([8, 4]);
          ctx.beginPath(); ctx.moveTo(tL, touchY); ctx.lineTo(tR, touchY); ctx.stroke(); ctx.setLineDash([]);
        }
      } else {
        ctx.moveTo(lsx, lsy); ctx.lineTo(v1x, v1y); ctx.lineTo(hdx, hdy); ctx.lineTo(v2x, v2y); ctx.lineTo(rsx, rsy);
        ctx.closePath(); ctx.fill();

        // ── Lines: LS -> V1 -> Head -> V2 -> RS ──
        ctx.strokeStyle = mainC; ctx.lineWidth = 2.5; ctx.setLineDash([]); ctx.globalAlpha = 0.9;
        ctx.beginPath(); ctx.moveTo(lsx, lsy); ctx.lineTo(v1x, v1y); ctx.lineTo(hdx, hdy); ctx.lineTo(v2x, v2y); ctx.lineTo(rsx, rsy);
        ctx.stroke(); ctx.globalAlpha = 1;
      }

      // ── Neckline: dashed line through valleys ──
      ctx.strokeStyle = 'rgba(59,130,246,0.7)'; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(v1x, v1y); ctx.lineTo(v2x, v2y);
      var ndx = v2x - v1x, ndy = v2y - v1y, nl = Math.sqrt(ndx*ndx + ndy*ndy);
      if (nl > 0) ctx.lineTo(v2x + (ndx/nl)*140, v2y + (ndy/nl)*140);
      ctx.stroke(); ctx.setLineDash([]);

      // ── Valley dots ──
      ctx.fillStyle = 'rgba(59,130,246,0.8)';
      ctx.beginPath(); ctx.arc(v1x, v1y, 4, 0, Math.PI*2); ctx.fill();
      ctx.beginPath(); ctx.arc(v2x, v2y, 4, 0, Math.PI*2); ctx.fill();
    } else {
      // No valleys, just draw LS -> Head -> RS
      ctx.strokeStyle = mainC; ctx.lineWidth = 2.5; ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(lsx, lsy); ctx.lineTo(hdx, hdy); ctx.lineTo(rsx, rsy); ctx.stroke();
    }

    // ── Shoulder circles (white ring) ──
    [{ x: lsx, y: lsy }, { x: rsx, y: rsy }].forEach(function(pt) {
      ctx.fillStyle = mainC; ctx.beginPath(); ctx.arc(pt.x, pt.y, 6, 0, Math.PI*2); ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(pt.x, pt.y, 6, 0, Math.PI*2); ctx.stroke();
    });

    // ── Head: ORANGE for H&S, PURPLE for DT ──
    var headCol = isDT ? '#a855f7' : '#f59e0b';
    ctx.shadowColor = headCol; ctx.shadowBlur = 12;
    ctx.fillStyle = headCol; ctx.beginPath(); ctx.arc(hdx, hdy, 9, 0, Math.PI*2); ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(hdx, hdy, 9, 0, Math.PI*2); ctx.stroke();
    // Label inside
    ctx.fillStyle = '#000'; ctx.font = 'bold 11px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(isDT ? 'DT' : 'H', hdx, hdy);
    ctx.textBaseline = 'alphabetic';

    // ── Labels ──
    ctx.font = '600 10px Inter, sans-serif'; ctx.textAlign = 'center';
    ctx.fillStyle = mainC;
    ctx.fillText(isDT ? 'Toque 1' : 'Ombro E', lsx, isBear ? lsy - 14 : lsy + 18);
    ctx.fillText(isDT ? 'Toque 2' : 'Ombro D', rsx, isBear ? rsy - 14 : rsy + 18);
    ctx.fillStyle = isDT ? '#a855f7' : '#f59e0b'; ctx.font = '700 11px Inter, sans-serif';
    ctx.fillText(isDT ? (isBear ? 'RESIST\xCANCIA' : 'SUPORTE') : 'CABECA', hdx, isBear ? hdy - 18 : hdy + 22);

    if (hasV) {
      ctx.fillStyle = 'rgba(59,130,246,0.7)'; ctx.font = '600 9px Inter, sans-serif';
      ctx.fillText('Neckline', (v1x+v2x)/2, isBear ? Math.max(v1y,v2y)+14 : Math.min(v1y,v2y)-8);
    }

    // ── Entry marker at confirmation candle (Close price) ──
    var entryChartIdx = (pat.entry_chart_idx != null) ? pat.entry_chart_idx : rsi;
    if (entryChartIdx >= 0 && entryChartIdx < candleData.length) {
      // Usar preço real de entrada (Close) em vez do preço do ombro (High/Low)
      var entryPrice = pat.entry_price || (pat.backtest && pat.backtest.entry_price) || rs.price;
      var ex = gx(entryChartIdx), ey = gy(entryPrice);
      if (ex !== null && ey !== null && !isNaN(ex) && !isNaN(ey)) {
        // Entry dashed line at CLOSE price level (preço real de entrada)
        var esleft = gx(Math.max(0, rsi - 2));
        var esright = gx(Math.min(candleData.length - 1, entryChartIdx + 12));
        if (esleft && esright) {
          ctx.setLineDash([5, 3]); ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.moveTo(esleft, ey); ctx.lineTo(esright, ey); ctx.stroke(); ctx.setLineDash([]);
        }
        // Modern entry icon (circle with arrow) — deslocado 1 vela à direita para não sobrepor Ombro D
        var entryDrawX = gx(Math.min(candleData.length - 1, entryChartIdx + 1)) || ex;
        ctx.shadowColor = '#f59e0b'; ctx.shadowBlur = 10;
        ctx.fillStyle = '#f59e0b'; ctx.beginPath(); ctx.arc(entryDrawX, ey, 10, 0, Math.PI*2); ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(entryDrawX, ey, 10, 0, Math.PI*2); ctx.stroke();
        // Arrow inside circle
        ctx.fillStyle = '#000'; ctx.font = 'bold 12px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(isBear ? '\u25BC' : '\u25B2', entryDrawX, ey);
        ctx.textBaseline = 'alphabetic';
        // ENTRADA label com preço
        ctx.fillStyle = '#f59e0b'; ctx.font = '700 10px Inter, sans-serif'; ctx.textAlign = 'left';
        ctx.fillText('ENTRADA ' + entryPrice.toFixed(5), entryDrawX + 16, ey + 4);
      }
    }

    // ── Stop line ──
    if (pat.stop) {
      var stopY = gy(pat.stop);
      if (stopY !== null && !isNaN(stopY)) {
        var sL = gx(Math.max(0, rsi - 1)), sR = gx(Math.min(candleData.length - 1, rsi + 12));
        if (sL && sR) {
          ctx.setLineDash([3, 3]); ctx.strokeStyle = 'rgba(239,68,68,0.6)'; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(sL, stopY); ctx.lineTo(sR, stopY); ctx.stroke(); ctx.setLineDash([]);
          ctx.fillStyle = 'rgba(239,68,68,0.7)'; ctx.font = '600 9px Inter, sans-serif'; ctx.textAlign = 'left';
          ctx.fillText('STOP', rsx + 14, stopY + (isBear ? -6 : 14));
        }
      }
    }

    // ── Target line ──
    if (pat.target) {
      var targetY = gy(pat.target);
      if (targetY !== null && !isNaN(targetY)) {
        var tL = gx(Math.max(0, rsi - 1)), tR = gx(Math.min(candleData.length - 1, rsi + 12));
        if (tL && tR) {
          ctx.setLineDash([3, 3]); ctx.strokeStyle = 'rgba(16,185,129,0.6)'; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(tL, targetY); ctx.lineTo(tR, targetY); ctx.stroke(); ctx.setLineDash([]);
          ctx.fillStyle = 'rgba(16,185,129,0.7)'; ctx.font = '600 9px Inter, sans-serif'; ctx.textAlign = 'left';
          ctx.fillText('META', rsx + 14, targetY + (isBear ? 14 : -6));
        }
      }
    }

    // ── Direction arrow + label ──
    ctx.fillStyle = isBear ? '#ef4444' : '#10b981';
    ctx.beginPath();
    if (isBear) { ctx.moveTo(rsx, rsy + 34); ctx.lineTo(rsx - 7, rsy + 24); ctx.lineTo(rsx + 7, rsy + 24); }
    else { ctx.moveTo(rsx, rsy - 34); ctx.lineTo(rsx - 7, rsy - 24); ctx.lineTo(rsx + 7, rsy - 24); }
    ctx.closePath(); ctx.fill();
    ctx.font = '800 11px Inter, sans-serif'; ctx.textAlign = 'left';
    if (isBear) { ctx.fillStyle = '#ef4444'; ctx.fillText('PUT', rsx + 12, rsy + 33); }
    else { ctx.fillStyle = '#10b981'; ctx.fillText('CALL', rsx + 12, rsy - 27); }

    // ── Result badge ──
    var bt = pat.backtest, btxt = 'LIVE', bcol = '#f59e0b', bbg = 'rgba(245,158,11,0.2)';
    if (bt) {
      if (bt.result === 'win') { btxt = 'WIN'; bcol = '#10b981'; bbg = 'rgba(16,185,129,0.2)'; }
      else if (bt.result === 'loss') { btxt = 'LOSS'; bcol = '#ef4444'; bbg = 'rgba(239,68,68,0.2)'; }
    }
    var bw = ctx.measureText(btxt).width + 16;
    var bx = hdx - bw/2, by = isBear ? hdy - 40 : hdy + 30;
    ctx.fillStyle = bbg;
    ctx.beginPath();
    ctx.roundRect(bx, by - 10, bw, 18, 6);
    ctx.fill();
    ctx.fillStyle = bcol; ctx.font = '800 11px Inter, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(btxt, hdx, by + 3);

    // ── IA prob below result ──
    ctx.fillStyle = '#8b5cf6'; ctx.font = '700 10px Inter, sans-serif';
    ctx.fillText('IA ' + ((pat.ia_prob||0.5)*100).toFixed(0) + '%', hdx, by + (isBear ? -14 : 20));

    // ── Pattern name ──
    ctx.fillStyle = 'rgba(255,255,255,0.5)'; ctx.font = '600 9px Inter, sans-serif';
    ctx.fillText(isDT ? (isBear ? 'Double Top' : 'Double Bottom') : (isBear ? 'Head & Shoulders' : 'Inverse H&S'), hdx, by + (isBear ? -28 : 34));
  });
}

function buildSidebar(data) {
  var list = document.getElementById('asset-list');
  var charts = data.charts || {};
  var s = data.summary || {};
  var byAsset = s.by_asset || {};
  var liveSet = new Set((data.live_signals || []).map(function(l){return l.ativo}));

  allAtivos = Object.keys(charts).sort(function(a, b) {
    var aL = liveSet.has(a) ? 1 : 0, bL = liveSet.has(b) ? 1 : 0;
    if (aL !== bL) return bL - aL;
    var aW = (byAsset[a] || {}).wr || 0, bW = (byAsset[b] || {}).wr || 0;
    if (aW !== bW) return bW - aW;
    return a.localeCompare(b);
  });

  list.innerHTML = allAtivos.map(function(ativo) {
    var cd = charts[ativo], ad = byAsset[ativo] || {};
    var pats = cd.patterns || [];
    var wins = pats.filter(function(p){return (p.backtest||{}).result === 'win'}).length;
    var losses = pats.filter(function(p){return (p.backtest||{}).result === 'loss'}).length;
    var live = pats.filter(function(p){return !p.backtest}).length;
    var total = wins + losses;
    var wr = total > 0 ? (wins / total * 100) : 0;
    var wrCls = wr >= 60 ? 'good' : wr >= 45 ? 'mid' : 'bad';
    var isAct = selectedAtivo === ativo ? ' active' : '';
    var dot = liveSet.has(ativo) ? '<span class="live-dot"></span>' : '';
    var iaMax = 0;
    pats.forEach(function(p) { if ((p.ia_prob||0) > iaMax) iaMax = p.ia_prob; });
    var iaBadge = iaMax > 0.5 ? '<span class="ia-badge"><svg class="icon-svg" style="width:10px;height:10px;stroke:var(--purple)"><use href="#i-brain"/></svg> ' + (iaMax*100).toFixed(0) + '%</span>' : '';
    var id = ativo.replace(/[^a-zA-Z0-9]/g, '_');
    return '<div class="asset-item' + isAct + '" id="ai-' + id + '" data-ativo="' + ativo + '" onclick="selectAsset(\'' + ativo + '\')">' +
      '<div class="a-left">' +
        '<span class="a-name">' + ativo + ' ' + dot + '</span>' +
        '<span class="a-meta"><svg class="icon-svg" style="width:10px;height:10px"><use href="#i-chart"/></svg> ' + pats.length + ' padroes' + (live > 0 ? ' &middot; <span style="color:var(--orange)">' + live + ' live</span>' : '') + ' ' + iaBadge + '</span>' +
      '</div>' +
      '<div class="a-right">' +
        '<span class="a-payout">' + (cd.payout||0) + '%</span>' +
        (total > 0 ? '<span class="wr-pill ' + wrCls + '">' + wr.toFixed(0) + '%</span>' : '') +
      '</div></div>';
  }).join('');

  if (!selectedAtivo && allAtivos.length > 0) selectAsset(allAtivos[0]);
}

function buildLivePanel(data) {
  var el = document.getElementById('live-list');
  var sigs = data.live_signals || [];
  if (sigs.length === 0) {
    el.innerHTML = '<div style="color:var(--text-muted);font-size:11px;text-align:center;padding:20px 0"><svg class="icon-svg" style="width:16px;height:16px;opacity:.4"><use href="#i-zap"/></svg><br>Aguardando sinais...</div>';
    return;
  }
  el.innerHTML = sigs.map(function(sig) {
    var cls = sig.direction === 'PUT' ? 'put' : 'call';
    var prob = ((sig.ia_prob||0.5)*100).toFixed(0);
    var dirIcon = sig.direction === 'PUT' ? '#i-arrow-down' : '#i-arrow-up';
    return '<div class="signal-card" onclick="selectAsset(\'' + sig.ativo + '\')">' +
      '<div class="sc-top"><span class="sc-name">' + sig.ativo + '</span><span class="sc-dir ' + cls + '"><svg class="icon-svg" style="width:10px;height:10px"><use href="' + dirIcon + '"/></svg> ' + sig.direction + '</span></div>' +
      '<div class="sc-bottom"><span class="sc-type"><svg class="icon-svg" style="width:10px;height:10px"><use href="#i-activity"/></svg> ' + (sig.type==='HEAD_SHOULDERS'?'H&S':'iH&S') + ' ' + sig.mode + '</span>' +
      '<span class="sc-prob"><svg class="icon-svg" style="width:12px;height:12px;stroke:var(--purple)"><use href="#i-brain"/></svg> ' + prob + '%' +
      '<span class="prob-bar"><span class="prob-fill" style="width:' + prob + '%"></span></span></span></div></div>';
  }).join('');
}

function buildResultsPanel(data) {
  var el = document.getElementById('results-list');
  var entries = data.broker_entries || [];

  /* Somente entradas REAIS — sem fallback de backtest */
  entries = entries.slice(0, 50);

  if (entries.length === 0) {
    el.innerHTML = '<div style="color:var(--text-muted);font-size:11px;text-align:center;padding:20px 0"><svg class="icon-svg" style="width:16px;height:16px;opacity:.4"><use href="#i-activity"/></svg><br>Sem entradas reais ainda<br><span style="font-size:10px;opacity:.6">Inicie o bot para ver os trades aqui</span></div>';
    return;
  }
  el.innerHTML = entries.map(function(r) {
    var cls = r.result === 'win' ? 'win' : r.result === 'entry' ? 'entry' : 'loss';
    var icoRef = r.result === 'win' ? '#i-check' : r.result === 'entry' ? '#i-clock' : '#i-x';
    var priceStr = r.price ? parseFloat(r.price).toFixed(5) : '';
    var dirIcon = r.dir === 'PUT' ? '#i-arrow-down' : '#i-arrow-up';
    var profitStr = '';
    if (r.profit && r.profit !== 0) {
      var pf = parseFloat(r.profit);
      profitStr = '<span class="rr-profit">' + (pf > 0 ? '+' : '') + pf.toFixed(2) + '</span>';
    }
    var brokerStr = r.broker ? '<span class="rr-broker">' + r.broker + '</span>' : '';
    return '<div class="result-row ' + cls + '" onclick="selectAsset(\'' + r.ativo + '\')" style="cursor:pointer">' +
      '<span class="rr-ativo">' + r.ativo + '</span>' +
      '<span class="rr-dir" style="color:' + (r.dir==='PUT'?'var(--red)':'var(--green)') + '"><svg class="icon-svg" style="width:10px;height:10px"><use href="' + dirIcon + '"/></svg> ' + r.dir + '</span>' +
      (priceStr ? '<span class="rr-price">' + priceStr + '</span>' : '') +
      profitStr +
      '<span class="rr-res"><svg class="icon-svg" style="width:12px;height:12px"><use href="' + icoRef + '"/></svg> ' + r.result.toUpperCase() + '</span>' +
      '</div>';
  }).join('');
}

function updateDashboard(data) {
  latestData = data;
  var s = data.summary || {};
  document.getElementById('st-total').textContent = s.total || 0;
  var wr = s.wr || 0;
  var wrEl = document.getElementById('st-wr');
  wrEl.textContent = wr.toFixed(1) + '%';
  wrEl.className = 'val ' + (wr >= 60 ? 'green' : wr >= 45 ? 'yellow' : 'red');
  document.getElementById('st-wins').textContent = s.wins || 0;
  document.getElementById('st-losses').textContent = (s.total || 0) - (s.wins || 0);
  var hs = (s.by_type || {}).HEAD_SHOULDERS;
  var ihs = (s.by_type || {}).INV_HEAD_SHOULDERS;
  document.getElementById('st-hs').textContent = hs ? hs.wr + '% (' + hs.total + ')' : '-';
  document.getElementById('st-ihs').textContent = ihs ? ihs.wr + '% (' + ihs.total + ')' : '-';
  document.getElementById('st-live').textContent = (data.live_signals || []).length;
  // IA Level
  var lvl = s.ia_level || {num:1, nome:'Iniciante', emoji:'\ud83c\udf31', cor:'#94a3b8'};
  var iaLvlEl = document.getElementById('st-ia-level');
  var iaLvlBox = document.getElementById('st-ia-level-box');
  iaLvlEl.textContent = lvl.emoji + ' ' + lvl.num + ' - ' + lvl.nome;
  iaLvlEl.style.color = lvl.cor;
  iaLvlBox.style.borderColor = lvl.cor;
  document.getElementById('badge-scan').innerHTML = '<svg class="icon-svg" style="width:12px;height:12px"><use href="#i-layers"/></svg> Scan #' + (data.scan_count || 0);
  document.getElementById('badge-status').innerHTML = '<svg class="icon-svg" style="width:12px;height:12px"><use href="#i-wifi"/></svg> Online';
  document.getElementById('badge-status').className = 'tbadge online';
  document.getElementById('last-update').textContent = 'Dados: ' + (data.last_update || '--');

  buildSidebar(data);
  buildLivePanel(data);
  buildResultsPanel(data);

  if (selectedAtivo) renderChart(data);
}

async function fetchData() {
  try {
    var r = await fetch('/api/data');
    if (!r.ok) return;
    var data = await r.json();
    updateDashboard(data);
  } catch(e) {
    document.getElementById('badge-status').innerHTML = '<svg class="icon-svg" style="width:12px;height:12px"><use href="#i-wifi"/></svg> Offline';
    document.getElementById('badge-status').className = 'tbadge err';
  }
}

fetchData();
setInterval(fetchData, 5000);

/* ═══ LIVE CANDLE STREAMING — atualiza a cada 5s ═══ */
var _liveInterval = null;
var _lastCandleTime = 0;
var _liveFetching = false;
var _liveFetchStart = 0;
var _liveUpdateTs = 0;  /* timestamp do último update live bem-sucedido */
var _liveErrorCount = 0;
function startLiveCandles() {
  if (_liveInterval) clearInterval(_liveInterval);
  _liveErrorCount = 0;
  _liveInterval = setInterval(async function() {
    if (!selectedAtivo || !mainSeries || !mainChart) return;
    /* Safety: se _liveFetching ficou true por >15s, forçar reset */
    if (_liveFetching) {
      if (Date.now() - _liveFetchStart > 15000) { _liveFetching = false; }
      else return;
    }
    _liveFetchStart = Date.now();
    _liveFetching = true;
    try {
      var r = await fetch('/api/live_candles?ativo=' + encodeURIComponent(selectedAtivo));
      if (!r.ok) { _liveFetching = false; return; }
      var d = await r.json();
      if (!d.candles || d.candles.length === 0 || d.ativo !== selectedAtivo) { _liveFetching = false; return; }
      d.candles.forEach(function(c) {
        var t = parseTime(c.t);
        if (isNaN(t) || t <= 0) return;
        var bar = { time: t, open: c.o, high: c.h, low: c.l, close: c.c };
        mainSeries.update(bar);
        if (t > _lastCandleTime && _lastCandleTime > 0) {
          requestAnimationFrame(drawHSOverlay);
        }
        _lastCandleTime = Math.max(_lastCandleTime, t);
        var found = false;
        for (var i = candleData.length - 1; i >= 0; i--) {
          if (candleData[i].time === t) { candleData[i] = bar; found = true; break; }
        }
        if (!found) candleData.push(bar);
      });
      _liveUpdateTs = Date.now();
      _liveErrorCount = 0;
    } catch(e) {
      _liveErrorCount++;
    }
    _liveFetching = false;
  }, 5000);
}
/* Iniciar streaming de velas automaticamente */
startLiveCandles();

</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════
# HTTP SERVER
# ══════════════════════════════════════════════════════════════════
class HSHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silenciar logs HTTP

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/trade":
            # Bot envia trade real em tempo real
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                if body.get("ativo") and body.get("result") in ("win", "loss", "entry", "tie"):
                    with _real_trades_lock:
                        new_status = body["result"]
                        new_entry = {
                            "ativo": body["ativo"],
                            "dir": body.get("dir", "?"),
                            "result": new_status,
                            "price": body.get("price", 0),
                            "stake": body.get("stake", 0),
                            "profit": body.get("profit", 0),
                            "time": body.get("time", ""),
                            "ts": body.get("ts", time.time()),
                            "broker": body.get("broker", "?"),
                        }
                        # Se é resultado, atualizar o entry existente do mesmo ativo
                        if new_status in ("win", "loss", "tie"):
                            updated = False
                            for i in range(len(_real_trades) - 1, -1, -1):
                                if _real_trades[i]["ativo"] == body["ativo"] and _real_trades[i]["result"] == "entry":
                                    _real_trades[i]["result"] = new_status
                                    _real_trades[i]["profit"] = body.get("profit", 0)
                                    _real_trades[i]["ts"] = body.get("ts", time.time())
                                    updated = True
                                    break
                            if not updated:
                                _real_trades.append(new_entry)
                        else:
                            _real_trades.append(new_entry)
                        if len(_real_trades) > _REAL_TRADES_MAX:
                            del _real_trades[:-_REAL_TRADES_MAX]
                self.send_response(200)
                self._cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        qs = parse_qs(urlparse(self.path).query)
        
        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
        
        elif path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._cors_headers()
            self.end_headers()
            try:
                data = build_api_data()
                self.wfile.write(json.dumps(data, cls=NpEncoder).encode("utf-8"))
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        elif path == "/api/live_candles":
            # Retorna velas em tempo real — lê arquivo live do bot (streaming 1s)
            global _selected_ativo
            ativo = (qs.get("ativo") or [""])[0]
            if ativo:
                _selected_ativo = ativo
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._cors_headers()
            self.end_headers()
            try:
                candles = []
                # PRIORIDADE 1: Ler arquivo live do bot (streaming real-time 1s)
                _live_file = os.path.join(_USER_DIR, "ws_live_candles.json")
                _used_live = False
                if os.path.exists(_live_file):
                    _max_retries = 2
                    for _retry in range(_max_retries):
                        try:
                            _lf_age = time.time() - os.path.getmtime(_live_file)
                            if _lf_age < 30:  # Arquivo com menos de 30s = fresco
                                with open(_live_file, "r") as _lf:
                                    _raw = _lf.read()
                                if _raw:
                                    _live_data = json.loads(_raw)
                                    _asset_candles = _live_data.get("assets", {}).get(ativo, [])
                                    if _asset_candles:
                                        for _c in _asset_candles:
                                            _ts = _c.get("t", 0)
                                            candles.append({
                                                "t": int(_ts),
                                                "o": _c.get("o", 0),
                                                "h": _c.get("h", 0),
                                                "l": _c.get("l", 0),
                                                "c": _c.get("c", 0),
                                            })
                                        _used_live = True
                                break  # sucesso, sair do retry
                        except (json.JSONDecodeError, PermissionError):
                            # Arquivo sendo escrito pelo bot — retry
                            import time as _time_mod
                            _time_mod.sleep(0.05)
                        except Exception:
                            break

                # FALLBACK: cache antigo (DataFrame)
                if not candles:
                    with _lock:
                        df = _cache["assets_data"].get(ativo)
                    if df is not None and len(df) > 0:
                        last_n = df.tail(5)
                        for ts, row in last_n.iterrows():
                            candles.append({
                                "t": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                                "o": round(float(row["open"]), 6),
                                "h": round(float(row["high"]), 6),
                                "l": round(float(row["low"]), 6),
                                "c": round(float(row["close"]), 6),
                            })

                if candles:
                    self.wfile.write(json.dumps({"candles": candles, "ativo": ativo}).encode("utf-8"))
                else:
                    self.wfile.write(b'{"candles":[],"ativo":""}')
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        
        else:
            self.send_response(404)
            self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Dashboard IA H&S")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Porta (default: {DEFAULT_PORT})")
    args = parser.parse_args()
    
    log.info(f"{'='*60}")
    log.info(f"  🧠 WS Trader — IA H&S Dashboard (SOMENTE LEITURA)")
    log.info(f"  📊 Dados: lidos do cache do bot (sem conexão ao broker)")
    log.info(f"  🌐 http://localhost:{args.port}")
    log.info(f"{'='*60}")
    
    # Thread de dados (lê cache do bot a cada ~10s — detecta padrões + treino IA)
    t = threading.Thread(target=_update_thread, daemon=True)
    t.start()
    
    # Thread de detecção rápida de sinais (a cada ~55s — mantém live_signals frescos)
    t_sig = threading.Thread(target=_signal_scan_thread, daemon=True)
    t_sig.start()
    
    # SEM thread de atualização rápida de velas (não conecta ao broker)
    
    # HTTP server THREADING com reuse_address — live candles não trava!
    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        allow_reuse_address = True
        allow_reuse_port = True
        daemon_threads = True
    
    server = ThreadedHTTPServer(("0.0.0.0", args.port), HSHandler)
    log.info(f"Dashboard THREADED iniciado na porta {args.port}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Dashboard encerrado.")
        server.server_close()


if __name__ == "__main__":
    main()
