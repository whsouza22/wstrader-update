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

import os, sys, json, time, logging, argparse, threading, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass
import numpy as np
import pandas as pd
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Optional
from urllib.parse import urlparse, parse_qs

try:
    from ws_reversal_ai import ReversalAI
    from ws_adaptive_brain import extract_features
    _HAS_NN = True
except ImportError:
    _HAS_NN = False

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

# ── NN models per-asset (carregados 1x) ──
_nn_models = {}  # {ativo: ReversalAI}
_NN_MIN_PROB = 0.80

def _load_nn_model(ativo):
    """Carrega modelo NN per-ativo se disponível."""
    if not _HAS_NN:
        return None
    if ativo in _nn_models:
        return _nn_models[ativo]
    try:
        rai = ReversalAI(ativo)
        if rai._ai1_ready:
            _nn_models[ativo] = rai
            log.info(f"[NN] Modelo carregado para {ativo}")
            return rai
    except Exception:
        pass
    return None

def _nn_predict_pattern(ativo, pat, H, L, C, O, n, atr):
    """Roda NN no padrão. Retorna dict {approved, count_above, p1, p2, p3} ou None."""
    rai = _load_nn_model(ativo)
    if rai is None or n < 50:
        return None
    try:
        candles_ago = pat.get("candles_ago", 0)
        rs_idx_local = n - 1 - candles_ago
        if rs_idx_local < 0:
            rs_idx_local = 0
        win_start = max(0, rs_idx_local - 55)
        win_end = min(n, rs_idx_local + 2)  # Inclui até 1 vela pós-RS para features
        H_w = H[win_start:win_end]
        L_w = L[win_start:win_end]
        C_w = C[win_start:win_end]
        O_w = O[win_start:win_end]
        n_w = len(H_w)
        atr_vals = [float(H_w[k] - L_w[k]) for k in range(max(0, n_w - 14), n_w)]
        atr_local = float(np.mean(atr_vals)) if atr_vals else atr
        if atr_local <= 0:
            atr_local = atr
        pat_copy = dict(pat)
        pat_copy["candles_ago"] = max(0, n_w - 1 - (rs_idx_local - win_start))
        feats = extract_features(pat_copy, H_w, L_w, C_w, O_w, n_w, atr_local, {}, ativo)
        if feats is None:
            return None
        pred = rai.predict_dt(feats)
        if pred is None:
            return None
        p1 = pred.get("p1", 0)
        p2 = pred.get("p2", 0)
        p3 = pred.get("p3")
        count_above = sum([
            round(p1, 2) >= _NN_MIN_PROB,
            round(p2, 2) >= _NN_MIN_PROB,
            round(p3, 2) >= _NN_MIN_PROB if p3 is not None else False
        ])
        nn_score = pred.get("nn_score", pred.get("prob_win", 0))
        return {
            "approved": nn_score >= _NN_MIN_PROB,
            "count_above": count_above,
            "nn_score": round(nn_score, 3),
            "p1": round(p1, 3),
            "p2": round(p2, 3),
            "p3": round(p3, 3) if p3 is not None else None,
            "prob_win": round(pred.get("prob_win", 0), 3),
        }
    except Exception:
        return None


def _build_price_prediction(pat, C, H, L, atr, n, nn_result=None):
    """Constrói previsão de preço-alvo para o sinal.
    Calcula velocidade média recente do par por minuto e estima
    se 1 minuto basta para atingir um movimento mínimo favorável,
    ou se precisa de 2 minutos.
    Retorna dict com prediction_2m pronto para o frontend.
    """
    try:
        direction = pat.get("direction", "CALL")
        rs_idx = pat.get("right_shoulder", {}).get("idx", n - 1)
        entry_idx = pat.get("entry_idx", rs_idx + 1)
        if entry_idx >= n:
            entry_idx = n - 1
        current_price = float(C[entry_idx]) if entry_idx < n else float(C[-1])

        # ATR recente (últimos 14 candles = 14 min)
        atr_base = float(atr) if atr > 0 else 0.001

        # Velocidade média por candle (últimos 10 candles M1)
        look = min(10, n - 1)
        moves = []
        for k in range(n - look, n):
            moves.append(abs(float(C[k]) - float(C[k - 1])))
        avg_move_1m = float(np.mean(moves)) if moves else atr_base * 0.4

        # Profundidade do padrão → impulso esperado
        depth = float(pat.get("depth", 0))
        depth_ratio = depth / atr_base if atr_base > 0 else 0

        # NN score bonus: NN mais alto → previsão mais confiante
        nn_score = 0.5
        if nn_result and isinstance(nn_result, dict):
            nn_score = float(nn_result.get("nn_score", 0.5))

        # Fator de impulso: combina profundidade + confiança
        impulse = 0.3 + depth_ratio * 0.15 + nn_score * 0.4
        impulse = min(impulse, 1.2)  # cap

        # Movimento esperado em 1 min e 2 min
        expected_1m = avg_move_1m * impulse
        expected_2m = avg_move_1m * impulse * 1.7  # 2 candles com decaimento

        # Mínimo necessário: 30% do ATR para considerar viável
        min_move = atr_base * 0.30

        # Decisão de duração
        if expected_1m >= min_move and nn_score >= 0.85:
            smart_exp = 1
            projected_move = expected_1m
        else:
            smart_exp = 2
            projected_move = expected_2m

        if direction == "PUT":
            projected_price = current_price - projected_move
        else:
            projected_price = current_price + projected_move

        confidence = round(min(nn_score * impulse, 0.99), 2)

        return {
            "available": True,
            "minutes": smart_exp,
            "price": round(projected_price, 6),
            "current_price": round(current_price, 6),
            "confidence": confidence,
            "move_expected": round(projected_move, 6),
            "avg_speed_1m": round(avg_move_1m, 6),
            "min_move": round(min_move, 6),
            "smart_exp": smart_exp,
        }
    except Exception:
        return {"available": False}


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
        pL = pat["left_shoulder"]["price"]
        pR = pat["right_shoulder"]["price"]
        # DT: valley1==valley2 (mesmo ponto) → usar alinhamento dos toques
        if pat.get("mode") == "double_touch":
            neck_align = abs(pL - pR) / atr_val if atr_val > 0 else 0
        else:
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
    features = ["span", "symmetry", "depth_ratio"]
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
    """H&S removido: o dashboard mantém somente Double Touch."""
    return []

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
    min_spacing = 12
    max_spacing = 45
    min_depth = atr * 1.5  # detecção base — NN decide qualidade
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
                v_reg = L[idx1+1:idx2]
                if len(v_reg) < 1:
                    continue
                v_rel = int(np.argmin(v_reg))
                v_idx = idx1 + 1 + v_rel
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
                if j - idx1 > max_spacing:
                    continue
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
                v_reg = L[idx1+1:j]
                if len(v_reg) < 1:
                    continue
                v_rel = int(np.argmin(v_reg))
                v_idx = idx1 + 1 + v_rel
                v_price = float(v_reg[v_rel])
                touch_level = max(float(price1), h_j)
                depth = touch_level - v_price
                if depth < min_depth:
                    continue
                d_left = v_idx - idx1
                d_right = j - v_idx
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
                p_reg = H[idx1+1:idx2]
                if len(p_reg) < 1:
                    continue
                p_rel = int(np.argmax(p_reg))
                p_idx = idx1 + 1 + p_rel
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
                if j - idx1 > max_spacing:
                    continue
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
                p_reg = H[idx1+1:j]
                if len(p_reg) < 1:
                    continue
                p_rel = int(np.argmax(p_reg))
                p_idx = idx1 + 1 + p_rel
                p_price = float(p_reg[p_rel])
                touch_level = min(float(price1), l_j)
                depth = p_price - touch_level
                if depth < min_depth:
                    continue
                d_left = p_idx - idx1
                d_right = j - p_idx
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
    """IA que aprende padrões DT por ativo e tipo."""
    
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
            geo["type"] = pat.get("type", "DOUBLE_TOP")
            geo["source"] = "backtest"  # marcado como backtest (learn vem do treino)
            self.geometry_history.append(geo)
            if len(self.geometry_history) > 300:
                self.geometry_history = self.geometry_history[-300:]
    
    def predict(self, ativo, pat):
        """Prediz probabilidade de WIN para um setup — com fallback hierárquico ponderado."""
        key = f"{ativo}_{pat.get('type', 'DOUBLE_TOP')}_{pat.get('mode', 'double_touch')}"
        item = self.stats.get(key)
        if item and item.get("total", 0) > 0:
            wins = item.get("wins", 0)
            total = item.get("total", 0)
            return round((wins + 2) / (total + 4), 4)

        pat_type = pat.get("type", "DOUBLE_TOP")
        pat_mode = pat.get("mode", "double_touch")
        candidates = []

        same_mode_wins = 0
        same_mode_total = 0
        same_type_wins = 0
        same_type_total = 0
        for stat_key, stat_val in self.stats.items():
            if f"_{pat_type}_{pat_mode}" in stat_key:
                same_mode_wins += stat_val.get("wins", 0)
                same_mode_total += stat_val.get("total", 0)
            if f"_{pat_type}_" in stat_key:
                same_type_wins += stat_val.get("wins", 0)
                same_type_total += stat_val.get("total", 0)

        if same_mode_total >= 3:
            candidates.append(((same_mode_wins + 2) / (same_mode_total + 4), same_mode_total))
        if same_type_total >= 5:
            candidates.append(((same_type_wins + 2) / (same_type_total + 4), same_type_total))

        global_total = self.global_stats.get("total", 0)
        global_wins = self.global_stats.get("wins", 0)
        if global_total >= 10:
            candidates.append(((global_wins + 2) / (global_total + 4), global_total))

        if candidates:
            import math
            total_weight = sum(math.sqrt(max(1, total)) for _, total in candidates)
            blended = sum(prob * math.sqrt(max(1, total)) for prob, total in candidates) / total_weight
            return round(blended, 4)

        return 0.5

    def get_summary(self):
        total = int(self.global_stats.get("total", 0) or 0)
        wins = int(self.global_stats.get("wins", 0) or 0)
        by_type = {}
        for pat_type, data in (self.global_stats.get("by_type") or {}).items():
            pt_total = int(data.get("total", 0) or 0)
            pt_wins = int(data.get("wins", 0) or 0)
            by_type[pat_type] = {
                "wins": pt_wins,
                "total": pt_total,
                "wr": round((pt_wins / pt_total) * 100, 1) if pt_total > 0 else 0.0,
            }

        by_asset = {}
        for stat_key, data in self.stats.items():
            parts = stat_key.split("_")
            ativo = parts[0] if parts else stat_key
            bucket = by_asset.setdefault(ativo, {"wins": 0, "total": 0, "live": 0})
            bucket["wins"] += int(data.get("wins", 0) or 0)
            bucket["total"] += int(data.get("total", 0) or 0)

        for data in by_asset.values():
            total_asset = data.get("total", 0)
            wins_asset = data.get("wins", 0)
            data["wr"] = round((wins_asset / total_asset) * 100, 1) if total_asset > 0 else 0.0

        return {
            "total": total,
            "wins": wins,
            "wr": round((wins / total) * 100, 1) if total > 0 else 0.0,
            "by_type": by_type,
            "by_asset": by_asset,
            "ia_level": _get_ia_level(total),
        }

    def get_training_stats(self):
        return {
            "meta": {"total": int(self.global_stats.get("total", 0) or 0), "wins": int(self.global_stats.get("wins", 0) or 0)},
            "arms": self.stats,
        }

    def save_to_disk(self):
        try:
            os.makedirs(os.path.dirname(IA_PERSIST_FILE), exist_ok=True)
            payload = {
                "stats": self.stats,
                "global_stats": self.global_stats,
                "geometry_history": self.geometry_history[-300:],
            }
            with open(IA_PERSIST_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception:
            pass

    def load_from_disk(self):
        try:
            if not os.path.exists(IA_PERSIST_FILE):
                return False
            with open(IA_PERSIST_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.stats = payload.get("stats", {}) if isinstance(payload.get("stats"), dict) else {}
            self.global_stats = payload.get("global_stats", {"wins": 0, "total": 0, "by_type": {}})
            self.geometry_history = payload.get("geometry_history", []) if isinstance(payload.get("geometry_history"), list) else []
            return True
        except Exception:
            return False

    def seed_from_bot_stats(self):
        return None


def _load_train_control():
    try:
        if not os.path.exists(TRAIN_CONTROL_FILE):
            return {}
        with open(TRAIN_CONTROL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_train_control():
    try:
        os.makedirs(os.path.dirname(TRAIN_CONTROL_FILE), exist_ok=True)
        now = datetime.now()
        iso = now.isocalendar()
        with open(TRAIN_CONTROL_FILE, "w", encoding="utf-8") as f:
            json.dump({"iso_year": iso[0], "iso_week": iso[1], "date": now.isoformat()}, f, ensure_ascii=False)
    except Exception:
        pass


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
        return {"num": 1, "nome": "Iniciante", "emoji": "🌱", "cor": "#6b7280"}
    elif n_total <= 10:
        return {"num": 2, "nome": "Aprendendo", "emoji": "📚", "cor": "#ff6a00"}
    elif n_total <= 30:
        return {"num": 3, "nome": "Calibrando", "emoji": "⚙️", "cor": "#ff6a00"}
    elif n_total <= 80:
        return {"num": 4, "nome": "Experiente", "emoji": "🧠", "cor": "#a855f7"}
    elif n_total <= 200:
        return {"num": 5, "nome": "Avançada", "emoji": "🎯", "cor": "#00e676"}
    else:
        return {"num": 6, "nome": "Expert", "emoji": "🏆", "cor": "#00e676"}


# ══════════════════════════════════════════════════════════════════
# LIVE BROKER — conexão INDEPENDENTE para o dashboard
# ══════════════════════════════════════════════════════════════════
_BROKER_TYPE = os.getenv("BROKER_TYPE", "bullex").strip().lower()
_LIVE_ASSETS = [
    "EURNZD-OTC", "GBPCHF-OTC", "EURAUD-OTC",
    "USDCAD-OTC", "AUDNZD-OTC", "GBPAUD-OTC",
]
_LIVE_TF = 60  # M1
_LIVE_N_CANDLES = 100  # candles para exibir no gráfico
_MIN_VISIBLE_PATTERNS = max(1, int(os.getenv("WS_DASH_MIN_VISIBLE_PATTERNS", "4")))
_LIVE_BROKER_REF = [None]  # referência mutável para reconexões
_LIVE_CONNECTED = threading.Event()


def _connect_live_broker():
    """Conecta ao broker para leitura de candles (SOMENTE LEITURA — sem trades)."""
    # Carregar credenciais do .env do usuário (mesmo arquivo que o login salva)
    _env_file = os.path.join(_USER_DIR, ".env")
    if os.path.exists(_env_file):
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=_env_file, override=True)
            log.info(f"Dashboard: credenciais carregadas de {_env_file}")
        except ImportError:
            # Fallback manual se dotenv não estiver instalado
            try:
                with open(_env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line and not line.startswith("#"):
                            k, v = line.split("=", 1)
                            os.environ[k.strip()] = v.strip()
            except Exception:
                pass

    # Re-ler BROKER_TYPE após carregar .env
    _bt = os.getenv("BROKER_TYPE", "").strip().lower()

    # Auto-detectar broker pelas credenciais disponíveis
    if not _bt:
        if os.getenv("IQ_EMAIL") and (os.getenv("IQ_PASS") or os.getenv("IQ_PASSWORD")):
            _bt = "iq_option"
        elif os.getenv("CASATRADER_EMAIL") and os.getenv("CASATRADER_PASS"):
            _bt = "casatrader"
        elif os.getenv("BULLUX_EMAIL") or os.getenv("BULLEX_EMAIL"):
            _bt = "bullex"
        else:
            _bt = _BROKER_TYPE  # fallback ao padrão do módulo

    if _bt == "casatrader":
        from casatraderapi.stable_api import Casa_Trader as _BrokerCls
        _email = os.getenv("CASATRADER_EMAIL", "")
        _senha = os.getenv("CASATRADER_PASS", "")
    elif _bt == "iq_option":
        from iqoptionapi.stable_api import IQ_Option as _BrokerCls
        _email = os.getenv("IQ_EMAIL", "")
        _senha = os.getenv("IQ_PASS", "") or os.getenv("IQ_PASSWORD", "")
    else:
        from bullexapi.stable_api import Bullex as _BrokerCls
        _email = os.getenv("BULLUX_EMAIL", "") or os.getenv("BULLEX_EMAIL", "")
        _senha = os.getenv("BULLUX_PASS", "") or os.getenv("BULLEX_PASS", "")

    if not _email or not _senha:
        log.warning("Dashboard: credenciais do broker não encontradas — usando cached data")
        return None

    try:
        log.info(f"Dashboard: Conectando ao {_bt}...")
        bx = _BrokerCls(_email, _senha)
        check, reason = bx.connect()
        if check is False:
            log.warning(f"Dashboard: Falha na conexão — {reason}")
            return None
        for _ in range(12):
            if bx.check_connect():
                break
            time.sleep(1.5)
        if not bx.check_connect():
            log.warning("Dashboard: Timeout na conexão")
            return None
        bx.change_balance("PRACTICE")  # dashboard = somente leitura
        try:
            bx.update_ACTIVES_OPCODE()
        except Exception:
            pass
        time.sleep(2)  # aguardar servidor initializar before requesting data
        log.info("Dashboard: Conectado ao broker (somente leitura)")
        return bx
    except Exception as e:
        log.warning(f"Dashboard: Erro ao conectar — {e}")
        return None


def _read_live_candles_from_file():
    """Lê velas live do arquivo escrito pelo bot.
    Retorna dict {ativo: [candles]} ou {} se arquivo stale/inexistente."""
    _live_file = os.path.join(_USER_DIR, "ws_live_candles.json")
    try:
        if not os.path.exists(_live_file):
            return {}
        age = time.time() - os.path.getmtime(_live_file)
        if age > 120:  # aceita até 2 min (bot pode pausar entre scans)
            return {}
        with open(_live_file, "r") as f:
            data = json.load(f)
        return data.get("assets", {})
    except Exception:
        return {}


def _live_broker_thread():
    """Thread que alimenta o dashboard com dados de velas.
    Lê EXCLUSIVAMENTE dos arquivos escritos pelo bot.
    NUNCA abre conexão própria ao broker (evita conflito de WebSocket).
    """
    _last_log_ts = 0

    while True:
        try:
            _has_data = False

            # ── 1. Ler cache completo do bot (500 velas por ativo) ──
            if os.path.exists(_DASHBOARD_CACHE_FILE):
                try:
                    bot_snapshot = _load_dashboard_cache_snapshot()
                    bot_assets, payouts = _load_candles_from_cache()
                    if bot_assets:
                        bot_patterns = {}
                        for _snap_ativo, _snap_data in (bot_snapshot.get("assets", {}) or {}).items():
                            _patterns = _snap_data.get("patterns", [])
                            if isinstance(_patterns, list):
                                bot_patterns[_snap_ativo] = _patterns
                        bot_live_signals = bot_snapshot.get("live_signals", []) if isinstance(bot_snapshot.get("live_signals"), list) else []
                        bot_summary = bot_snapshot.get("summary", {}) if isinstance(bot_snapshot.get("summary"), dict) else {}
                        bot_ts = float(bot_snapshot.get("ts", 0) or 0)
                        bot_analysis = str(bot_snapshot.get("analysis_source", "")) == "bot"
                        with _lock:
                            # Substituir ativos — se bot mudou de par, removemos os antigos
                            _cache["assets_data"] = bot_assets
                            _cache["payouts"] = payouts
                            if bot_analysis:
                                _cache["assets_patterns"] = bot_patterns
                                _cache["live_signals"] = bot_live_signals
                                _cache["signal_history"] = _merge_signal_history(_cache.get("signal_history", {}), bot_live_signals)
                                _cache["selected_assets"] = list(bot_snapshot.get("selected_assets", []) or [])
                                _cache["ia_summary"] = bot_summary
                                _cache["analysis_source"] = "bot"
                                if bot_ts and bot_ts != _cache.get("bot_cache_ts", 0):
                                    _cache["scan_count"] += 1
                                    _cache["bot_cache_ts"] = bot_ts
                            else:
                                _cache["analysis_source"] = "dashboard"
                        _has_data = True
                except Exception:
                    pass

            # ── 2. Ler candles live do arquivo do bot e MERGE no cache ──
            live_assets = _read_live_candles_from_file()
            if live_assets:
                _has_data = True
                # Merge: atualizar DataFrames com candles live frescos
                with _lock:
                    for _la_ativo, _la_candles in live_assets.items():
                        if _la_ativo not in _cache["assets_data"]:
                            continue
                        _df = _cache["assets_data"][_la_ativo]
                        if _df is None or len(_df) == 0:
                            continue
                        for _lc in _la_candles:
                            _lt = _lc.get("t", 0)
                            if _lt <= 0:
                                continue
                            _ts = pd.Timestamp(_lt, unit="s", tz="UTC")
                            try:
                                _ts = _ts.tz_localize(None)
                            except Exception:
                                pass
                            if _ts in _df.index:
                                _df.at[_ts, "open"] = _lc.get("o", _df.at[_ts, "open"])
                                _df.at[_ts, "high"] = _lc.get("h", _df.at[_ts, "high"])
                                _df.at[_ts, "low"] = _lc.get("l", _df.at[_ts, "low"])
                                _df.at[_ts, "close"] = _lc.get("c", _df.at[_ts, "close"])
                            else:
                                _new_row = pd.DataFrame(
                                    [{"open": _lc.get("o", 0), "high": _lc.get("h", 0),
                                      "low": _lc.get("l", 0), "close": _lc.get("c", 0), "volume": 0}],
                                    index=[_ts]
                                )
                                _new_row.index.name = "time"
                                _df = pd.concat([_df, _new_row])
                                _df.sort_index(inplace=True)
                                _df = _df.tail(120)  # manter últimas 120 velas
                                _cache["assets_data"][_la_ativo] = _df

            # ── 3. Atualizar status de conexão ──
            with _lock:
                if _has_data or _cache["assets_data"]:
                    _cache["connected"] = True
                    _cache["error"] = None
                else:
                    _cache["error"] = "Aguardando dados do bot..."

            # Log periódico (a cada 60s)
            if time.time() - _last_log_ts > 60:
                n_assets = len(_cache.get("assets_data", {}))
                has_live = bool(live_assets)
                log.info(f"Dashboard live: {n_assets} ativos carregados, live={'sim' if has_live else 'não'}")
                _last_log_ts = time.time()

        except Exception as e:
            log.debug(f"Live broker thread error: {e}")

        time.sleep(1)


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
    "signal_history": {},       # {ativo: [sinais do bot preservados no gráfico]}
    "selected_assets": [],      # ativos realmente selecionados pelo bot para o scan atual
    "analysis_source": "dashboard",
    "bot_cache_ts": 0,
}

_SIGNAL_HISTORY_MAX_PER_ASSET = 16
_SIGNAL_HISTORY_MAX_AGE_SEC = 6 * 60 * 60


def _signal_identity_key(signal: dict) -> str:
    if not isinstance(signal, dict):
        return ""
    ativo = str(signal.get("ativo", "") or "")
    direction = str(signal.get("direction", "") or "")
    sig_type = str(signal.get("type", "") or "")
    mode = str(signal.get("mode", "") or "")
    rs_ts = int(((signal.get("right_shoulder") or {}).get("ts", 0)) or 0)
    entry_ts = int(signal.get("entry_ts", 0) or 0)
    return "|".join([ativo, direction, sig_type, mode, str(rs_ts), str(entry_ts)])


def _merge_signal_history(existing_history: dict, incoming_signals: list) -> dict:
    merged = {}
    now_ts = time.time()

    if isinstance(existing_history, dict):
        for ativo, patterns in existing_history.items():
            if isinstance(patterns, list):
                merged[str(ativo)] = [dict(item) for item in patterns if isinstance(item, dict)]

    for signal in incoming_signals or []:
        if not isinstance(signal, dict):
            continue
        ativo = str(signal.get("ativo", "") or "")
        if not ativo:
            continue
        merged.setdefault(ativo, []).append(dict(signal))

    normalized = {}
    for ativo, patterns in merged.items():
        dedup = {}
        for pattern in patterns:
            pattern_ts = float(pattern.get("scan_ts", 0) or ((pattern.get("right_shoulder") or {}).get("ts", 0) or pattern.get("entry_ts", 0) or 0))
            if pattern_ts and (now_ts - pattern_ts) > _SIGNAL_HISTORY_MAX_AGE_SEC:
                continue
            dedup[_signal_identity_key(pattern)] = dict(pattern)
        ordered = list(dedup.values())
        ordered.sort(
            key=lambda item: float(item.get("scan_ts", 0) or ((item.get("right_shoulder") or {}).get("ts", 0) or item.get("entry_ts", 0) or 0)),
            reverse=True,
        )
        if ordered:
            normalized[ativo] = ordered[:_SIGNAL_HISTORY_MAX_PER_ASSET]
    return normalized

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
                            "decision_id": t.get("decision_id"),
                            "order_id": t.get("order_id"),
                            "nn_approved": t.get("nn_approved"),
                            "nn_p1": t.get("nn_p1"),
                            "nn_p2": t.get("nn_p2"),
                            "nn_p3": t.get("nn_p3"),
                        })
        except Exception:
            pass
    # Deduplicar: se há um resultado (win/loss/tie) para um entry do mesmo ativo, manter só o resultado
    deduped = []
    seen_results = set()  # (ativo+dir, ts_approx) já resolvidos
    # Primeiro pass: coletar todos os resultados
    for e in entries:
        if e["result"] in ("win", "loss", "tie"):
            seen_results.add(e["ativo"] + "_" + e.get("dir", "") + "_" + str(int(e.get("ts", 0) // 300)))
    # Segundo pass: filtrar entries que já têm resultado
    for e in entries:
        if e["result"] == "entry":
            key = e["ativo"] + "_" + e.get("dir", "") + "_" + str(int(e.get("ts", 0) // 300))
            if key in seen_results:
                continue  # pular entry duplicado — já temos o resultado
        deduped.append(e)
    # Ordenar por timestamp decrescente
    deduped.sort(key=lambda x: x.get("ts", 0), reverse=True)
    return deduped[:_REAL_TRADES_MAX]


def _pattern_reference_ts(pattern: dict) -> int:
    if not isinstance(pattern, dict):
        return 0

    candidates = [
        pattern.get("broker_ts"),
        pattern.get("entry_ts"),
        pattern.get("scan_ts"),
    ]
    right_shoulder = pattern.get("right_shoulder")
    head = pattern.get("head")
    if isinstance(right_shoulder, dict):
        candidates.append(right_shoulder.get("ts"))
    if isinstance(head, dict):
        candidates.append(head.get("ts"))

    for candidate in candidates:
        try:
            ts = int(float(candidate or 0))
        except Exception:
            ts = 0
        if ts > 0:
            return ts
    return 0


def _find_matching_active_entry(pattern: dict, entries: list) -> Optional[dict]:
    if not isinstance(pattern, dict) or not isinstance(entries, list):
        return None

    direction = str(pattern.get("direction", "") or "")
    reference_ts = _pattern_reference_ts(pattern)
    best_entry = None
    best_delta = None

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("result") not in ("entry", "win", "loss", "tie"):
            continue
        if str(entry.get("dir", "") or "") != direction:
            continue

        try:
            entry_ts = int(float(entry.get("ts", 0) or 0))
        except Exception:
            entry_ts = 0

        delta = abs(entry_ts - reference_ts) if reference_ts > 0 and entry_ts > 0 else 0
        if best_entry is None or best_delta is None or delta < best_delta:
            best_entry = entry
            best_delta = delta

    return best_entry


def _select_primary_chart_patterns(patterns: list) -> list:
    if not isinstance(patterns, list) or not patterns:
        return []

    def _rank_key(pat: dict):
        is_live = 1 if pat and not pat.get("backtest") else 0
        is_active = 1 if pat and pat.get("signal_active") is not False else 0
        ts = _pattern_reference_ts(pat if isinstance(pat, dict) else {})
        ia_prob = float((pat or {}).get("ia_prob", 0.0) or 0.0)
        nn_approved = 1 if (pat or {}).get("nn_approved") is True else 0
        return (is_live, is_active, nn_approved, ts, ia_prob)

    ranked = sorted(
        [pat for pat in patterns if isinstance(pat, dict)],
        key=_rank_key,
        reverse=True,
    )
    return ranked[:1]


def _load_dashboard_cache_snapshot() -> dict:
    try:
        if not os.path.exists(_DASHBOARD_CACHE_FILE):
            return {}
        with open(_DASHBOARD_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        log.debug(f"Erro ao ler snapshot do bot: {e}")
        return {}


def _load_candles_from_cache():
    """Lê dados de velas do cache compartilhado (escrito pelo bot).
    Retorna dict {ativo: DataFrame} e {ativo: payout}."""
    assets_data = {}
    payouts = {}
    try:
        data = _load_dashboard_cache_snapshot()
        if not data:
            return assets_data, payouts
        for ativo, adata in data.get("assets", {}).items():
            candles = adata.get("candles", [])
            if len(candles) < 20:
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


def _count_visible_patterns_in_tail(patterns: list, total_candles: int, tail_size: int) -> int:
    if not patterns or total_candles <= 0:
        return 0
    tail_start = max(0, total_candles - max(1, int(tail_size)))
    visible = 0
    for pat in patterns:
        rs_idx = (pat.get("right_shoulder") or {}).get("idx", -1)
        if tail_start <= rs_idx < total_candles:
            visible += 1
    return visible


def _signal_scan_thread():
    """Thread de detecção de sinais: roda a cada ~55s usando dados em cache.
    NÃO faz chamadas ao broker — usa dados já atualizados pelas outras threads.
    Mantém live_signals sempre frescos para o bot (scan_ts < 60s).
    """
    while True:
        try:
            with _lock:
                _use_bot_analysis = _cache.get("analysis_source") == "bot"
            if _use_bot_analysis:
                time.sleep(5)
                continue

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
                if df is None or len(df) < 20:
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
                all_hs = detect_double_touch(H, L, C, O, ph, pl, atr, n, max_candles_ago=3)

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
                        # NN prediction
                        _nn_res = _nn_predict_pattern(ativo, pat, H, L, C, O, n, atr)
                        if _nn_res is not None:
                            pat["nn_approved"] = _nn_res["approved"]
                            pat["nn_count"] = _nn_res["count_above"]
                            pat["nn_score"] = _nn_res.get("nn_score", 0)
                            pat["nn_p1"] = _nn_res["p1"]
                            pat["nn_p2"] = _nn_res["p2"]
                            pat["nn_p3"] = _nn_res["p3"]
                        else:
                            pat["nn_approved"] = None
                        # Previsão de preço-alvo + duração inteligente
                        pat["prediction_2m"] = _build_price_prediction(pat, C, H, L, atr, n, _nn_res)
                        # Gravar timestamps nos pontos-chave
                        _sig_df_index = df.index
                        _sig_df_len = len(_sig_df_index)
                        for _sk in ("left_shoulder", "head", "right_shoulder", "valley1", "valley2"):
                            _sp = pat.get(_sk)
                            if _sp and "idx" in _sp and 0 <= _sp["idx"] < _sig_df_len:
                                _sp["ts"] = int(_sig_df_index[_sp["idx"]].timestamp()) if hasattr(_sig_df_index[_sp["idx"]], 'timestamp') else 0
                        if "entry_idx" in pat and 0 <= pat["entry_idx"] < _sig_df_len:
                            pat["entry_ts"] = int(_sig_df_index[pat["entry_idx"]].timestamp()) if hasattr(_sig_df_index[pat["entry_idx"]], 'timestamp') else 0
                        # 100% IA: só mostra sinais aprovados pela rede neural
                        if pat.get("nn_approved") is True:
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
    """Thread principal: processa padrões, backtest, treina IA.
    Usa dados do live broker thread OU do cache do bot (fallback).
    """
    global _ia, _scanning
    _first_cycle = True
    _last_cache_ts = 0
    _last_process_ts = 0
    _bot_mode_logged = False
    
    while True:
        try:
            # ── Fonte de dados: live broker OU cache do bot (fallback) ──
            _has_live_data = False
            with _lock:
                _has_live_data = bool(_cache["assets_data"])

            if not _has_live_data:
                # Sem dados do live broker — tentar cache do bot
                if not os.path.exists(_DASHBOARD_CACHE_FILE):
                    log.info("Aguardando conexão live ou cache do bot...")
                    with _lock:
                        _cache["error"] = "Aguardando conexão ao broker..."
                    time.sleep(5)
                    continue

                try:
                    _cache_mtime = os.path.getmtime(_DASHBOARD_CACHE_FILE)
                except Exception:
                    _cache_mtime = 0
                if _cache_mtime > _last_cache_ts:
                    _last_cache_ts = _cache_mtime
                    bot_assets, payouts = _load_candles_from_cache()
                    if bot_assets:
                        with _lock:
                            for ativo, df in bot_assets.items():
                                _cache["assets_data"][ativo] = df
                            _cache["payouts"].update(payouts)
                            _cache["connected"] = True
                            _cache["error"] = None
                else:
                    time.sleep(2)
                    continue

            # Não processar mais de 1x a cada 30s
            if time.time() - _last_process_ts < 30:
                time.sleep(5)
                continue
            _last_process_ts = time.time()

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
                # Sempre enriquecer com stats do bot principal (63k+ amostras)
                _ia.seed_from_bot_stats()

            # ── Usar dados já em cache (preenchidos pelo live broker ou cache bot) ──
            with _lock:
                current_assets = dict(_cache["assets_data"])
                current_payouts = dict(_cache["payouts"])

            with _lock:
                _use_bot_analysis = _cache.get("analysis_source") == "bot" and bool(_cache.get("assets_data"))

            if _use_bot_analysis:
                _scanning = False
                if not _bot_mode_logged:
                    log.info("[DASHBOARD] Usando análise do bot como fonte única; scanner local pesado em standby")
                    _bot_mode_logged = True
                time.sleep(5)
                continue
            _bot_mode_logged = False

            if not current_assets:
                time.sleep(5)
                continue

            assets_patterns = {}
            live_signals = []
            _ia_new = HS_IA()
            # ── Seed a partir da base do bot (63k+ amostras estáveis) ──
            _ia_new.seed_from_bot_stats()
            # ── Herdar geometria aprendida (não perder histórico visual) ──
            if _ia and _ia.geometry_history:
                _ia_new.geometry_history = list(_ia.geometry_history)

            log.info(f"Processando {len(current_assets)} ativos...")
            _scanning = True

            for ativo, df in current_assets.items():
                if df is None or len(df) < 20:
                    continue

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
                        # NN prediction for live signal
                        _nn_res = _nn_predict_pattern(ativo, pat, H, L, C, O, n, atr)
                        if _nn_res is not None:
                            pat["nn_approved"] = _nn_res["approved"]
                            pat["nn_count"] = _nn_res["count_above"]
                            pat["nn_score"] = _nn_res.get("nn_score", 0)
                            pat["nn_p1"] = _nn_res["p1"]
                            pat["nn_p2"] = _nn_res["p2"]
                            pat["nn_p3"] = _nn_res["p3"]
                        else:
                            pat["nn_approved"] = None
                        # Previsão de preço-alvo + duração inteligente
                        pat["prediction_2m"] = _build_price_prediction(pat, C, H, L, atr, n, _nn_res)
                        # 100% IA: só mostra sinais aprovados pela rede neural
                        if pat.get("nn_approved") is True:
                            live_signals.append(pat)
                        patterns_with_results.append({**pat, "backtest": None, "ia_prob": ia_prob})
                    elif bt["result"] in ("win", "loss"):
                        _ia_new.learn(ativo, pat, bt, atr)
                        ia_prob = _ia.predict(ativo, pat)
                        patterns_with_results.append({**pat, "backtest": bt, "ia_prob": round(ia_prob, 3)})

                # ── Gravar timestamps nos pontos-chave para mapeamento estável ──
                _df_index = df.index
                _df_len = len(_df_index)
                for _pr in patterns_with_results:
                    for _pkey in ("left_shoulder", "head", "right_shoulder", "valley1", "valley2"):
                        _pt = _pr.get(_pkey)
                        if _pt and "idx" in _pt:
                            _pi = _pt["idx"]
                            if 0 <= _pi < _df_len:
                                _pt["ts"] = int(_df_index[_pi].timestamp()) if hasattr(_df_index[_pi], 'timestamp') else 0
                    if "entry_idx" in _pr:
                        _ei = _pr["entry_idx"]
                        if 0 <= _ei < _df_len:
                            _pr["entry_ts"] = int(_df_index[_ei].timestamp()) if hasattr(_df_index[_ei], 'timestamp') else 0
                    _bt = _pr.get("backtest")
                    if _bt:
                        for _bk in ("entry_idx", "exit_idx"):
                            _bi = _bt.get(_bk, -1)
                            if 0 <= _bi < _df_len:
                                _bt[_bk.replace("idx", "ts")] = int(_df_index[_bi].timestamp()) if hasattr(_df_index[_bi], 'timestamp') else 0

                if patterns_with_results:
                    assets_patterns[ativo] = patterns_with_results
                    _w = sum(1 for p in patterns_with_results if (p.get('backtest') or {}).get('result') == 'win')
                    _l = sum(1 for p in patterns_with_results if (p.get('backtest') or {}).get('result') == 'loss')
                    _lv = sum(1 for p in patterns_with_results if p.get('backtest') is None)
                    _visible = _count_visible_patterns_in_tail(patterns_with_results, n, _LIVE_N_CANDLES)
                    log.info(
                        f"  {ativo}: total={len(all_hs)} | visíveis={_visible} | {_w}W / {_l}L | Live: {_lv}"
                    )

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

            log.info(f"[IA] WR: {summary['wr']:.1f}% | "
                     f"Live: {len(live_signals)} sinais")

        except Exception as e:
            _scanning = False
            log.error(f"Erro: {e}", exc_info=True)
            with _lock:
                _cache["error"] = str(e)

        # Sleep 10s e verificar novamente se cache foi atualizado  
        time.sleep(10)


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
        signal_history = _cache.get("signal_history", {})
        payouts = _cache["payouts"]
        scan_count = _cache["scan_count"]
        analysis_source = _cache.get("analysis_source", "dashboard")

    # ── Ler velas live do bot para merge em tempo real ──
    _live_candles_by_asset = {}
    try:
        _live_file = os.path.join(_USER_DIR, "ws_live_candles.json")
        if os.path.exists(_live_file):
            _lf_age = time.time() - os.path.getmtime(_live_file)
            if _lf_age < 15:
                with open(_live_file, "r") as _lf:
                    _ld = json.load(_lf)
                _live_candles_by_asset = _ld.get("assets", {})
    except Exception:
        pass
    
    _live_by_asset = {}
    _history_by_asset = {}
    for _sig in live:
        if not isinstance(_sig, dict):
            continue
        _sig_asset = str(_sig.get("ativo", "") or "")
        if not _sig_asset:
            continue
        _live_by_asset.setdefault(_sig_asset, []).append(_sig)

    _authoritative_live = analysis_source == "bot" or bool(_live_by_asset)
    if _authoritative_live:
        if isinstance(signal_history, dict):
            for _asset_name, _patterns in signal_history.items():
                if isinstance(_patterns, list):
                    _history_by_asset[str(_asset_name)] = [dict(_item) for _item in _patterns if isinstance(_item, dict)]

    charts = {}
    for ativo, df in ad.items():
        last_120 = df.tail(_LIVE_N_CANDLES)
        candles = []
        for ts, row in last_120.iterrows():
            candles.append({
                "t": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                "o": round(float(row["open"]), 6),
                "h": round(float(row["high"]), 6),
                "l": round(float(row["low"]), 6),
                "c": round(float(row["close"]), 6),
            })

        # ── Merge: atualizar/adicionar velas live no final ──
        _asset_live = _live_candles_by_asset.get(ativo, [])
        if _asset_live and candles:
            for _lc in _asset_live:
                _lt = _lc.get("t", 0)
                if _lt <= 0:
                    continue
                _bar = {
                    "t": int(_lt),
                    "o": _lc.get("o", 0),
                    "h": _lc.get("h", 0),
                    "l": _lc.get("l", 0),
                    "c": _lc.get("c", 0),
                }
                # Procurar candle existente com mesmo timestamp
                _found = False
                for _ci in range(len(candles) - 1, max(len(candles) - 10, -1), -1):
                    _ct = candles[_ci].get("t", "")
                    # Comparar: live usa int epoch, cache usa ISO string
                    if isinstance(_ct, str) and _ct:
                        try:
                            _ct_epoch = int(pd.Timestamp(_ct).timestamp())
                        except Exception:
                            continue
                    else:
                        _ct_epoch = int(_ct) if _ct else 0
                    if _ct_epoch == int(_lt):
                        candles[_ci] = _bar
                        _found = True
                        break
                if not _found:
                    candles.append(_bar)

        # ── Normalizar timestamps: tudo como epoch int ──
        for _ci in range(len(candles)):
            _ct = candles[_ci]["t"]
            if isinstance(_ct, str):
                try:
                    candles[_ci]["t"] = int(pd.Timestamp(_ct).timestamp())
                except Exception:
                    pass
            elif isinstance(_ct, float):
                candles[_ci]["t"] = int(_ct)

        # Ordenar e deduplicar por timestamp antes de mapear chart_idx.
        # O frontend ordena as velas de novo; se o backend mapear os padrões
        # com a lista fora de ordem, o overlay desenha nos pontos errados.
        _candles_by_ts = {}
        for _candle in candles:
            _cts = _candle.get("t")
            if isinstance(_cts, int) and _cts > 0:
                _candles_by_ts[_cts] = _candle
        candles = [_candles_by_ts[_cts] for _cts in sorted(_candles_by_ts.keys())]

        _asset_live_patterns = [dict(_item) for _item in _live_by_asset.get(ativo, []) if isinstance(_item, dict)]
        if _asset_live_patterns:
            pats_data = _asset_live_patterns
        else:
            pats_data = _history_by_asset.get(ativo, []) if _authoritative_live else ap.get(ativo, [])
        _active_signal_keys = {_signal_identity_key(_sig) for _sig in _live_by_asset.get(ativo, [])}

        # ── Lookup: epoch → chart_idx ──
        _ts_to_ci = {}
        for _ci, _candle in enumerate(candles):
            _ts_to_ci[int(_candle["t"])] = _ci

        # Mapear padrões para coordenadas do gráfico
        mapped_pats = []
        for p in pats_data:
            mp = dict(p)
            mp["overlay_authoritative"] = _authoritative_live
            mp["signal_active"] = _signal_identity_key(mp) in _active_signal_keys if _authoritative_live else (not mp.get("backtest"))
            for key in ["left_shoulder", "head", "right_shoulder", "valley1", "valley2"]:
                if key in mp and mp[key]:
                    mp[key] = dict(mp[key])
                    _pts = mp[key].get("ts", 0)
                    if _pts and int(_pts) in _ts_to_ci:
                        mp[key]["chart_idx"] = _ts_to_ci[int(_pts)]
                    else:
                        mp[key]["chart_idx"] = -1
            if "entry_idx" in mp:
                _ets = mp.get("entry_ts", 0)
                if _ets and int(_ets) in _ts_to_ci:
                    mp["entry_chart_idx"] = _ts_to_ci[int(_ets)]
                else:
                    mp["entry_chart_idx"] = -1
            if mp.get("backtest") and "entry_idx" in mp["backtest"]:
                mp["backtest"] = dict(mp["backtest"])
                _bet = mp["backtest"].get("entry_ts", 0)
                _bxt = mp["backtest"].get("exit_ts", 0)
                mp["backtest"]["entry_chart_idx"] = _ts_to_ci.get(int(_bet), -1) if _bet else -1
                mp["backtest"]["exit_chart_idx"] = _ts_to_ci.get(int(_bxt), -1) if _bxt else -1
            # Só incluir padrões com Toque 2 visível no gráfico
            rs_ci = mp.get("right_shoulder", {}).get("chart_idx", -1)
            if rs_ci >= 0 and rs_ci < len(candles):
                mapped_pats.append(mp)
        if _authoritative_live and mapped_pats:
            mapped_pats = _select_primary_chart_patterns(mapped_pats)
        
        charts[ativo] = {
            "candles": candles,
            "patterns": mapped_pats,
            "payout": payouts.get(ativo, 0),
            "n_candles": len(candles),
            "visible_patterns": len(mapped_pats),
            "live_patterns": len(_live_by_asset.get(ativo, [])) if _authoritative_live else sum(1 for p in mapped_pats if not p.get("backtest")),
            "meets_min_patterns": len(mapped_pats) >= _MIN_VISIBLE_PATTERNS,
            "overlay_mode": "bot_live_only" if _authoritative_live else "dashboard_mixed",
        }
    
    # Broker entries: APENAS trades REAIS feitos pelo bot (lidos dos logs)
    broker_entries = _load_bot_trade_logs()

    # ── Sincronizar NN: se o engine entrou (entry), o NN APROVOU ──
    # Dashboard e engine avaliam NN em momentos diferentes → podem divergir.
    # Se existe uma entrada ativa do engine, usar o resultado do engine.
    _active_entries_by_asset = {}
    for be in broker_entries:
        if be.get("result") in ("entry", "win", "loss", "tie"):
            _active_entries_by_asset.setdefault(be.get("ativo", ""), []).append(be)

    # Corrigir nn_approved nos padrões do gráfico também
    for ativo_key, chart_data in charts.items():
        _asset_entries = _active_entries_by_asset.get(ativo_key, [])
        for mp in chart_data.get("patterns", []):
            if not mp.get("backtest"):  # só padrões live
                _be_match = _find_matching_active_entry(mp, _asset_entries)
                if _be_match:
                    mp["nn_approved"] = True
                    mp["broker_status"] = _be_match.get("result")
                    mp["broker_ts"] = _be_match.get("ts")
                    if _be_match.get("price"):
                        mp["broker_entry_price"] = round(float(_be_match.get("price") or 0), 6)

    # Live signals com IA prob
    live_mapped = []
    for s in live:
        _mapped_signal = dict(s) if isinstance(s, dict) else {}
        _mapped_signal["ativo"] = _mapped_signal.get("ativo", "?")
        _mapped_signal["overlay_authoritative"] = _authoritative_live
        live_mapped.append(_mapped_signal)
    
    # Broker entries: APENAS trades REAIS feitos pelo bot (lidos dos logs)
    # (já carregado acima para sincronizar NN nos gráficos)
    for lm in live_mapped:
        _be_match = _find_matching_active_entry(lm, _active_entries_by_asset.get(lm["ativo"], []))
        if _be_match:
            lm["nn_approved"] = True
            lm["broker_status"] = _be_match.get("result")
            lm["broker_ts"] = _be_match.get("ts")
            if _be_match.get("price"):
                lm["broker_entry_price"] = round(float(_be_match.get("price") or 0), 6)
                lm["entry_price"] = round(float(_be_match.get("price") or 0), 6)
            if _be_match.get("nn_p1") is not None:
                lm["nn_p1"] = _be_match["nn_p1"]
                lm["nn_p2"] = _be_match.get("nn_p2")
                lm["nn_p3"] = _be_match.get("nn_p3")
    # Mesclar com trades recebidos via POST (tempo real), sem duplicar
    # Dedup por decision_id/order_id (primário) + ativo+dir bucket (fallback)
    _seen_ids = set()
    _seen_keys = set()
    for be in broker_entries:
        if be.get("decision_id"):
            _seen_ids.add(be["decision_id"])
        if be.get("order_id") is not None:
            _seen_ids.add(str(be["order_id"]))
        _seen_keys.add((be.get("ativo",""), be.get("dir",""), int(be.get("ts", 0) // 300)))
    with _real_trades_lock:
        for rt in _real_trades:
            # Skip se já temos este trade por ID
            if rt.get("decision_id") and rt["decision_id"] in _seen_ids:
                continue
            if rt.get("order_id") is not None and str(rt["order_id"]) in _seen_ids:
                continue
            key = (rt.get("ativo",""), rt.get("dir",""), int(rt.get("ts", 0) // 300))
            if key not in _seen_keys:
                broker_entries.append(rt)
                _seen_keys.add(key)
                if rt.get("decision_id"):
                    _seen_ids.add(rt["decision_id"])
    # Consolidar: se há win/loss para um ativo+dir, remover entry duplicado
    _resolved = set()
    for be in broker_entries:
        if be.get("result") in ("win", "loss", "tie"):
            _resolved.add((be.get("ativo",""), be.get("dir",""), int(be.get("ts", 0) // 300)))
    broker_entries = [be for be in broker_entries if not (be.get("result") == "entry" and (be.get("ativo",""), be.get("dir",""), int(be.get("ts", 0) // 300)) in _resolved)]
    broker_entries.sort(key=lambda x: x.get("ts", 0), reverse=True)
    broker_entries = broker_entries[:50]

    # IA training stats para o bot importar no startup
    ia_training_stats = {}
    try:
        ia_training_stats = _ia.get_training_stats()
    except Exception:
        pass

    # ── NN Stats per-ativo (lidos dos arquivos de stats salvos pelo bot) ──
    nn_per_asset = {}
    _top_assets = ["NZDJPY-OTC", "GBPAUD-OTC", "USDCAD-OTC", "EURNZD-OTC"]
    for _nn_asset in _top_assets:
        _nn_file = os.path.join(_USER_DIR, f"ws_ai_stats_{_nn_asset}.json")
        if os.path.exists(_nn_file):
            try:
                with open(_nn_file, "r", encoding="utf-8") as _nf:
                    _nn_data = json.load(_nf)
                nn_per_asset[_nn_asset] = {
                    "samples": _nn_data.get("samples", 0),
                    "ml": _nn_data.get("ml", False),
                    "ai1_val": _nn_data.get("ai1_val", 0),
                    "ai2_val": _nn_data.get("ai2_val", 0),
                    "ai3_val": _nn_data.get("ai3_val", 0),
                    "ai1_ready": _nn_data.get("ai1_ready", False),
                }
            except Exception:
                pass

    return {
        "charts": charts,
        "summary": summary,
        "live_signals": live_mapped,
        "selected_assets": list(_cache.get("selected_assets", []) or []),
        "broker_entries": broker_entries,
        "ia_training_stats": ia_training_stats,
        "nn_per_asset": nn_per_asset,
        "scan_count": scan_count,
        "min_visible_patterns": _MIN_VISIBLE_PATTERNS,
        "last_update": datetime.now().strftime("%H:%M:%S"),
        "connected": _cache.get("connected", False),
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
<title>WS Trader — IA Double Touch</title>
<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4/dist/lightweight-charts.standalone.production.js"></script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=Sora:wght@500;600;700;800&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
    --bg-primary:#020617;--bg-secondary:#07101d;--bg-card:#0b1628;--bg-hover:#112038;
    --border:rgba(148,163,184,0.16);--border-light:rgba(148,163,184,0.24);
    --text-primary:#e5eefb;--text-secondary:#9fb0c8;--text-muted:#6f819b;
    --accent:#f97316;--accent-glow:rgba(249,115,22,0.22);
  --green:#00e676;--green-bg:rgba(0,230,118,0.10);--green-glow:rgba(0,230,118,0.25);
  --red:#ff3d57;--red-bg:rgba(255,61,87,0.10);--red-glow:rgba(255,61,87,0.25);
    --orange:#fb923c;--orange-bg:rgba(251,146,60,0.12);--orange-glow:rgba(251,146,60,0.24);
    --purple:#7c93ff;--purple-bg:rgba(124,147,255,0.12);
    --cyan:#38bdf8;--cyan-bg:rgba(56,189,248,0.12);
    --glass:rgba(10,16,29,0.68);
    --radius:14px;--radius-sm:10px;--radius-full:9999px;
}
body{background:radial-gradient(circle at 12% -8%,rgba(56,189,248,0.16),transparent 28%),radial-gradient(circle at 88% 0%,rgba(249,115,22,0.12),transparent 26%),linear-gradient(180deg,#020617 0%,#050b16 48%,#030712 100%);color:var(--text-primary);font-family:'IBM Plex Sans',system-ui,sans-serif;overflow:hidden;height:100vh;display:flex;flex-direction:column}
.icon-svg{display:inline-block;vertical-align:middle;width:14px;height:14px;fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}

/* ── HEADER ── */
.top-bar{background:linear-gradient(135deg,rgba(8,16,30,0.94) 0%,rgba(10,18,33,0.9) 55%,rgba(15,24,42,0.86) 100%);padding:12px 24px;display:flex;align-items:center;gap:16px;border-bottom:1px solid var(--border);flex-shrink:0;backdrop-filter:blur(18px)}
.logo{display:flex;align-items:center;gap:10px}
.logo-icon{width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,#f97316,#38bdf8);display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:800;color:#fff;box-shadow:0 10px 30px rgba(56,189,248,0.18)}
.logo h1{font-size:15px;font-weight:700;color:var(--text-primary);letter-spacing:-0.3px;font-family:'Sora',sans-serif}
.logo .sub{font-size:10px;color:var(--text-muted);font-weight:400;letter-spacing:0.5px}
.top-badges{display:flex;gap:8px;margin-left:20px}
.tbadge{padding:4px 12px;border-radius:var(--radius-full);font-size:11px;font-weight:600;border:1px solid var(--border);display:inline-flex;align-items:center;gap:5px;cursor:pointer}
.tbadge.online{background:var(--green-bg);border-color:var(--green);color:var(--green)}
.tbadge.scanning{background:var(--orange-bg);border-color:var(--orange);color:var(--orange);animation:tblink 2s infinite}
.tbadge.err{background:var(--red-bg);border-color:var(--red);color:var(--red)}
@keyframes tblink{0%,100%{opacity:1}50%{opacity:.5}}
.top-right{margin-left:auto;display:flex;align-items:center;gap:14px}
.top-time{font-size:12px;color:var(--text-secondary);font-weight:600;font-variant-numeric:tabular-nums;display:flex;align-items:center;gap:5px}
.clock-dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 1.5s infinite}
.candle-timer{display:inline-flex;align-items:center;gap:6px;background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-full);padding:4px 12px 4px 6px;font-size:11px;font-weight:700;color:var(--orange);font-variant-numeric:tabular-nums;cursor:pointer}
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
.st.blue::before{background:linear-gradient(90deg,var(--accent),#ff8c33)}
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
.main-area{flex:1;display:flex;flex-direction:column;overflow:hidden;padding:12px 14px 14px;background:linear-gradient(180deg,rgba(6,10,20,0.48),rgba(6,10,20,0.18))}
.chart-toolbar{padding:14px 18px;background:linear-gradient(180deg,rgba(10,18,32,0.94),rgba(10,18,32,0.82));border:1px solid rgba(56,189,248,0.12);border-bottom:none;display:flex;justify-content:space-between;align-items:center;flex-shrink:0;border-radius:22px 22px 0 0;backdrop-filter:blur(16px);box-shadow:0 18px 44px rgba(2,6,23,0.34)}
.chart-toolbar .ct-left{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.chart-toolbar .ct-name{font-weight:800;color:var(--text-primary);font-size:20px;letter-spacing:-0.5px;font-family:'Sora',sans-serif}
.chart-toolbar .ct-payout{background:rgba(0,230,118,0.12);color:var(--green);padding:5px 12px;border-radius:var(--radius-full);font-size:12px;font-weight:700;border:1px solid rgba(0,230,118,0.18)}
.chart-toolbar .ct-dir{display:inline-flex;align-items:center;gap:4px;padding:5px 12px;border-radius:var(--radius-full);font-size:11px;font-weight:700;border:1px solid transparent}
.ct-dir.put{background:var(--red-bg);color:var(--red)}
.ct-dir.call{background:var(--green-bg);color:var(--green)}
.ct-right{display:flex;align-items:center;gap:10px;flex-wrap:wrap;justify-content:flex-end}
.ct-info,.ct-chip{font-size:11px;color:var(--text-secondary);display:inline-flex;align-items:center;gap:6px;padding:7px 12px;border-radius:var(--radius-full);background:rgba(255,255,255,0.03);border:1px solid rgba(148,163,184,0.12)}
.ct-chip.good{color:#86efac;border-color:rgba(0,230,118,0.18);background:rgba(0,230,118,0.08)}
.ct-chip.warn{color:#fdba74;border-color:rgba(249,115,22,0.18);background:rgba(249,115,22,0.08)}
.ia-entry-icon{display:inline-flex;align-items:center;gap:8px;background:linear-gradient(135deg,#f97316,#fb923c);color:#fff;padding:8px 14px;border-radius:var(--radius-full);font-size:12px;font-weight:700;box-shadow:0 10px 26px rgba(249,115,22,0.24)}
#main-chart-box{flex:1;min-height:0;position:relative;background:linear-gradient(180deg,rgba(7,17,31,0.98),rgba(4,9,18,0.98));border-left:1px solid rgba(56,189,248,0.12);border-right:1px solid rgba(56,189,248,0.12);overflow:hidden}
#main-chart-box::before{content:'';position:absolute;inset:0;background:radial-gradient(circle at top right,rgba(56,189,248,0.10),transparent 28%),radial-gradient(circle at bottom left,rgba(249,115,22,0.10),transparent 32%);pointer-events:none;z-index:0}
#main-chart-box::after{content:'';position:absolute;inset:0;border:1px solid rgba(255,255,255,0.03);pointer-events:none;z-index:0}
.pat-footer{padding:10px 20px;background:linear-gradient(180deg,rgba(9,15,28,0.98),rgba(7,12,23,0.98));border:1px solid rgba(56,189,248,0.12);border-top:none;max-height:126px;overflow-y:auto;font-size:11px;flex-shrink:0;border-radius:0 0 22px 22px;box-shadow:0 18px 44px rgba(2,6,23,0.34)}
.pat-row{display:flex;align-items:center;justify-content:space-between;padding:4px 0;border-bottom:1px solid var(--border)}
.pat-row:last-child{border:none}
.pat-row .pr-type{color:var(--text-secondary);font-weight:500;display:flex;align-items:center;gap:4px}
.pat-row .pr-ia{color:var(--purple);font-weight:700;display:flex;align-items:center;gap:3px}
.pat-row .pr-res{font-weight:700;display:flex;align-items:center;gap:4px}
.pr-res.win{color:var(--green)} .pr-res.loss{color:var(--red)} .pr-res.live{color:var(--orange)} .pr-res.skip{color:var(--text-muted)}
.empty-state{display:flex;flex-direction:column;align-items:center;justify-content:center;flex:1;color:var(--text-muted);gap:16px;position:relative;z-index:1}
.empty-state .e-icon{opacity:.32;filter:drop-shadow(0 0 18px rgba(56,189,248,0.16))}
.empty-state .e-text{font-size:17px;font-weight:600;color:#d9e7fb;font-family:'Sora',sans-serif}
.empty-state .e-sub{font-size:12px;max-width:340px;text-align:center;line-height:1.7;color:var(--text-secondary)}

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
.prob-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--accent),#ff8c33)}
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

/* ── DECISION MODAL ── */
.dm-overlay{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:radial-gradient(circle at top,rgba(19,33,58,0.55),rgba(2,6,16,0.92) 58%,rgba(2,4,12,0.97));z-index:9999;justify-content:center;align-items:center;backdrop-filter:blur(14px)}
.dm-overlay.open{display:flex}
.dm-box{background:linear-gradient(180deg,rgba(9,14,25,0.99),rgba(5,9,18,0.98));border:1px solid rgba(125,211,252,0.14);border-radius:24px;max-width:1000px;width:95%;max-height:94vh;overflow-y:auto;padding:22px 22px 16px;position:relative;box-shadow:0 30px 100px rgba(0,0,0,0.62),inset 0 1px 0 rgba(255,255,255,0.04)}
.dm-close{position:absolute;top:10px;right:14px;background:none;border:none;color:var(--text-muted);font-size:20px;cursor:pointer;z-index:1}
.dm-close:hover{color:var(--red)}
.dm-shell{display:grid;gap:10px}
.dm-headline{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:10px;padding-right:24px}
.dm-headcopy{display:flex;flex-direction:column;gap:6px;min-width:0}
.dm-overline{font-size:9px;font-weight:800;letter-spacing:1.4px;text-transform:uppercase;color:#93c5fd}
.dm-title{font-size:18px;font-weight:800;color:#f8fafc;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.dm-meta{font-size:10px;color:var(--text-muted)}
.dm-direction{display:inline-flex;align-items:center;justify-content:center;padding:3px 9px;border-radius:999px;font-size:10px;font-weight:800;letter-spacing:0.4px}
.dm-direction.call{color:var(--green);background:rgba(0,230,118,0.12);border:1px solid rgba(0,230,118,0.26)}
.dm-direction.put{color:var(--red);background:rgba(255,61,87,0.12);border:1px solid rgba(255,61,87,0.24)}
.dm-score-badge{display:flex;flex-direction:column;align-items:flex-end;justify-content:center;gap:4px;min-width:124px;padding:10px 12px;border-radius:16px;background:linear-gradient(180deg,rgba(255,255,255,0.05),rgba(255,255,255,0.02));border:1px solid rgba(255,255,255,0.08)}
.dm-score-badge span{font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--text-muted)}
.dm-score-badge strong{font-size:20px;line-height:1;font-weight:800}
.dm-score-badge.good strong{color:var(--green)}
.dm-score-badge.bad strong{color:var(--red)}
.dm-score-badge.warn strong{color:#f59e0b}
.dm-pipeline{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:10px}
.dm-step{display:flex;flex-direction:column;align-items:flex-start;padding:8px 10px;border-radius:12px;font-size:10px;min-width:98px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06)}
.dm-step.pass{background:rgba(0,230,118,0.08);border:1px solid rgba(0,230,118,0.22);color:#b6f6d0}
.dm-step.fail{background:rgba(255,61,87,0.08);border:1px solid rgba(255,61,87,0.18);color:#ffc4cf}
.dm-step .sv{font-size:13px;font-weight:800;margin-top:3px;color:#f8fafc}
.dm-arrow{color:rgba(125,211,252,0.32);font-size:14px}
.dm-hero{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(220px,0.86fr);gap:10px;margin-bottom:10px}
.dm-focus{background:linear-gradient(180deg,rgba(12,19,34,0.96),rgba(8,13,24,0.98));border:1px solid rgba(125,211,252,0.1);border-radius:16px;padding:12px}
.dm-focus.prime{background:linear-gradient(135deg,rgba(18,33,61,0.98),rgba(9,16,29,0.98));border-color:rgba(125,211,252,0.16)}
.dm-focus-head{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:8px}
.dm-focus h4{font-size:13px;color:#f8fafc;margin-bottom:3px}
.dm-focus p{font-size:10px;color:var(--text-muted);line-height:1.35}
.dm-kpis{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
.dm-kpi{display:flex;flex-direction:column;gap:4px;padding:8px 10px;border-radius:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.05)}
.dm-kpi .kl{font-size:9px;font-weight:700;letter-spacing:0.8px;text-transform:uppercase;color:var(--text-muted)}
.dm-kpi .kv{font-size:13px;font-weight:800;color:#f8fafc}
.dm-pillrow{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}
.dm-pill{display:inline-flex;align-items:center;gap:6px;padding:5px 8px;border-radius:999px;font-size:10px;font-weight:700;border:1px solid rgba(255,255,255,0.07);background:rgba(255,255,255,0.03);color:#e5e7eb}
.dm-pill.good{color:var(--green);border-color:rgba(0,230,118,0.22);background:rgba(0,230,118,0.08)}
.dm-pill.bad{color:var(--red);border-color:rgba(255,61,87,0.2);background:rgba(255,61,87,0.08)}
.dm-pill.warn{color:#f59e0b;border-color:rgba(245,158,11,0.18);background:rgba(245,158,11,0.08)}
.dm-pill.neutral{color:#dbeafe;border-color:rgba(147,197,253,0.18);background:rgba(59,130,246,0.09)}
.dm-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:10px}
.dm-sec{background:linear-gradient(180deg,rgba(14,18,30,0.96),rgba(11,14,24,0.96));border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:11px}
.dm-sec h4{font-size:11px;color:#7dd3fc;margin-bottom:8px;letter-spacing:0.5px;text-transform:uppercase}
.dm-row{display:flex;justify-content:space-between;padding:4px 0;font-size:11px;border-bottom:1px solid rgba(255,255,255,0.04);gap:10px}
.dm-row:last-child{border-bottom:none}
.dm-row .dl{color:var(--text-muted)}
.dm-row .dv{font-weight:600;text-align:right}
.dv.good{color:var(--green)}.dv.bad{color:var(--red)}.dv.warn{color:#f59e0b}.dv.neutral{color:var(--purple)}
.dm-protect{padding:11px 12px;border-radius:12px;background:linear-gradient(180deg,rgba(11,18,31,0.96),rgba(8,12,22,0.98));border:1px solid rgba(255,255,255,0.05)}
.dm-protect strong{display:block;font-size:11px;font-weight:800;letter-spacing:0.4px;color:#e2e8f0;margin-bottom:4px}
.dm-protect p{font-size:10px;line-height:1.4;color:var(--text-muted)}
.dm-blurline{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04)}
.dm-blurline:last-child{border-bottom:none}
.dm-blurkey{font-size:10px;color:var(--text-muted)}
.dm-blurval{font-size:11px;font-weight:800;color:#cbd5e1;letter-spacing:1.2px;filter:blur(2px);user-select:none}
.dm-softnote{margin-top:8px;font-size:9px;color:#94a3b8;line-height:1.35}
.dm-bar{display:flex;align-items:center;gap:8px;margin:8px 0}
.dm-bar .bl{font-size:10px;color:var(--text-muted);min-width:52px;font-weight:700;letter-spacing:0.6px;text-transform:uppercase}
.dm-bar .bbg{flex:1;height:9px;background:rgba(255,255,255,0.05);border-radius:999px;overflow:hidden}
.dm-bar .bf{height:100%;border-radius:999px;box-shadow:0 0 18px rgba(255,255,255,0.12)}
.dm-bar .bv{font-size:11px;font-weight:800;min-width:40px;text-align:right}
.dm-result{margin-top:10px;padding:10px;border-radius:12px;font-size:14px;font-weight:800;text-align:center;letter-spacing:0.3px}
.dm-nomatch{color:var(--text-muted);text-align:center;padding:34px 22px;font-size:13px;line-height:1.5}
@media (max-height: 820px){.dm-box{max-height:96vh;max-width:1020px;padding:18px 18px 14px}.dm-shell{gap:8px}.dm-headline{margin-bottom:8px;padding-right:20px}.dm-pipeline{margin-bottom:8px}.dm-step{min-width:92px;padding:7px 9px}.dm-hero{grid-template-columns:minmax(0,1.45fr) minmax(220px,0.85fr);gap:8px;margin-bottom:8px}.dm-focus,.dm-sec,.dm-protect{padding:10px}.dm-row{padding:3px 0}.dm-focus p{font-size:9.5px;line-height:1.28}.dm-result{margin-top:8px;padding:9px;font-size:13px}}
@media (max-width: 760px){.dm-headline{flex-direction:column;padding-right:20px}.dm-score-badge{align-items:flex-start}.dm-hero{grid-template-columns:1fr}.dm-kpis{grid-template-columns:1fr}}
@media (max-width: 1100px){.chart-toolbar{padding:12px 14px}.ct-right{justify-content:flex-start}.main-area{padding:10px}.sidebar{width:250px;min-width:250px}.right-panel{width:300px;min-width:300px}}
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
      <div class="sub">IA Double Touch</div>
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

<!-- STATS ROW — REMOVIDO -->
<div class="stats-row" id="stats-bar" style="display:none">
  <div class="st blue"><div class="val blue" id="st-total">-</div></div>
  <div class="st"><div class="val" id="st-wr">0%</div></div>
  <div class="st"><div class="val" id="st-wins">0</div></div>
  <div class="st"><div class="val" id="st-losses">0</div></div>
  <div class="st"><div class="val" id="st-hs">-</div></div>
  <div class="st"><div class="val" id="st-ihs">-</div></div>
  <div class="st"><div class="val" id="st-live">0</div></div>
  <div class="st" id="st-ia-level-box"><div class="val" id="st-ia-level">-</div></div>
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
                <span class="ct-chip" id="ct-patterns"><svg class="icon-svg" style="width:12px;height:12px"><use href="#i-layers"/></svg> 0 padroes</span>
                <span class="ct-chip" id="ct-nn-chip" style="display:none"></span>
                        <span class="ct-chip" id="ct-ai-chip" style="display:none"></span>
            <span class="ct-chip" id="ct-pred-chip" style="display:none"></span>
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
                <div class="e-sub">Abra um ativo para ver o grafico neural em tempo real, com velas, niveis-chave e leitura profissional dos padroes detectados.</div>
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
      <div class="rp-header" style="display:flex;align-items:center;justify-content:space-between"><span><svg class="icon-svg" style="width:13px;height:13px;stroke:var(--green)"><use href="#i-activity"/></svg> Entradas na Corretora</span><button onclick="clearTrades()" style="background:var(--red);color:#fff;border:none;border-radius:4px;padding:2px 8px;font-size:10px;cursor:pointer;opacity:.7" onmouseover="this.style.opacity=1" onmouseout="this.style.opacity=.7">Limpar</button></div>
      <div class="rp-body entries-body" id="results-list">
        <div style="color:var(--text-muted);font-size:11px;text-align:center;padding:20px 0">Sem entradas ainda</div>
      </div>
    </div>
  </div>

</div>

<div class="footer"><svg class="icon-svg" style="width:11px;height:11px"><use href="#i-activity"/></svg> WS Trader v5.6 — IA Double Touch — Velas ao vivo a cada 1s</div>

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
        layout: { background: { color: '#07111f' }, textColor: '#90a4bf', fontFamily: 'IBM Plex Sans, system-ui, sans-serif' },
        grid: { vertLines: { color: 'rgba(56,189,248,0.06)' }, horzLines: { color: 'rgba(255,255,255,0.045)' } },
        crosshair: {
            mode: 1,
            vertLine: { color: 'rgba(56,189,248,0.24)', width: 1, labelBackgroundColor: '#0f172a' },
            horzLine: { color: 'rgba(249,115,22,0.22)', width: 1, labelBackgroundColor: '#111827' }
        },
        timeScale: { timeVisible: true, secondsVisible: false, borderColor: 'rgba(56,189,248,0.14)', rightOffset: 6, barSpacing: 11, minBarSpacing: 7, lockVisibleTimeRangeOnResize: true },
        rightPriceScale: { borderColor: 'rgba(56,189,248,0.14)', scaleMargins: { top: 0.08, bottom: 0.08 } },
        watermark: { visible: true, text: selectedAtivo ? selectedAtivo.replace('-OTC', '') : 'WS Trader', color: 'rgba(56,189,248,0.07)', fontSize: 34, horzAlign: 'left', vertAlign: 'top' },
  });
  mainSeries = mainChart.addCandlestickSeries({
        upColor: '#22c55e', downColor: '#fb7185',
        wickUpColor: '#34d399', wickDownColor: '#fb7185',
        borderUpColor: '#16a34a', borderDownColor: '#e11d48',
        borderVisible: true,
  });
  mainChart.timeScale().subscribeVisibleTimeRangeChange(function() { requestAnimationFrame(drawHSOverlay); });
  new ResizeObserver(function() {
    if (mainChart) { mainChart.applyOptions({ width: el.clientWidth, height: el.clientHeight }); requestAnimationFrame(drawHSOverlay); }
  }).observe(el);
}

function selectAsset(ativo) {
  if (selectedAtivo !== ativo || !mainChart) {
    selectedAtivo = ativo;
    candleData = [];  /* Limpar velas do ativo anterior */
    _lastCandleTime = 0;
    firstRender = true;
    initChart();
    /* Reiniciar streaming de velas para novo ativo */
    if (typeof startLiveCandles === 'function') startLiveCandles();
  }
  document.querySelectorAll('.asset-item').forEach(function(el) { el.classList.remove('active'); });
  var itemEl = document.getElementById('ai-' + ativo.replace(/[^a-zA-Z0-9]/g, '_'));
  if (itemEl) { itemEl.classList.add('active'); itemEl.scrollIntoView({ block: 'nearest' }); }
  /* Buscar dados frescos para o novo ativo */
  fetchData();
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
    if (mainChart) {
        mainChart.applyOptions({ watermark: { visible: true, text: selectedAtivo.replace('-OTC', ''), color: 'rgba(56,189,248,0.07)', fontSize: 34, horzAlign: 'left', vertAlign: 'top' } });
    }

  // IA entry badge
    var assetLiveSignals = (data.live_signals || []).filter(function(sig) { return sig.ativo === selectedAtivo; });
    var livePats = assetLiveSignals.length > 0 ? assetLiveSignals : [];
        var visiblePatterns = (cdata.patterns || []).filter(function(p) { return p.right_shoulder && p.right_shoulder.chart_idx >= 0; });
        function patternRankTs(pat) {
            var candidates = [
                pat && pat.broker_ts,
                pat && pat.entry_ts,
                pat && pat.scan_ts,
                pat && pat.right_shoulder && pat.right_shoulder.ts,
                pat && pat.head && pat.head.ts
            ];
            for (var i = 0; i < candidates.length; i++) {
                var ts = Number(candidates[i] || 0);
                if (ts > 0) return ts;
            }
            return 0;
        }
        function choosePrimaryPattern(patterns) {
            if (!patterns || !patterns.length) return [];
            var ranked = patterns.slice().sort(function(a, b) {
                var aLive = a && !a.backtest ? 1 : 0;
                var bLive = b && !b.backtest ? 1 : 0;
                if (aLive !== bLive) return bLive - aLive;
                var aActive = a && a.signal_active === false ? 0 : 1;
                var bActive = b && b.signal_active === false ? 0 : 1;
                if (aActive !== bActive) return bActive - aActive;
                var aTs = patternRankTs(a);
                var bTs = patternRankTs(b);
                if (aTs !== bTs) return bTs - aTs;
                return Number(b && b.ia_prob || 0) - Number(a && a.ia_prob || 0);
            });
            return ranked.length ? [ranked[0]] : [];
        }
        var renderPatterns = choosePrimaryPattern(visiblePatterns);
  var iaEntry = document.getElementById('ia-entry');
  var ctDir = document.getElementById('ct-dir');
    var ctPatterns = document.getElementById('ct-patterns');
    var ctNnChip = document.getElementById('ct-nn-chip');
                var ctAiChip = document.getElementById('ct-ai-chip');
        var ctPredChip = document.getElementById('ct-pred-chip');
        ctPatterns.innerHTML = '<svg class="icon-svg" style="width:12px;height:12px"><use href="#i-layers"/></svg> ' + renderPatterns.length + ' padrao ativo';
  if (livePats.length > 0) {
    var best = livePats.reduce(function(a, b) { return (a.ia_prob || 0) > (b.ia_prob || 0) ? a : b; });
    var timing = best.timing_hint || {};
        var aiSummary = getAiConsensusSummary(best);
    var timingLabel = '';
    if (timing && timing.available) {
      timingLabel = timing.action === 'now' ? ' · AGORA' : ' · ESPERA';
    }
    iaEntry.style.display = '';
    document.getElementById('ia-entry-txt').textContent = 'IA ' + ((best.ia_prob||0.5)*100).toFixed(0) + '%' + timingLabel;
    ctDir.style.display = '';
    ctDir.className = 'ct-dir ' + (best.direction === 'PUT' ? 'put' : 'call');
    ctDir.innerHTML = '<svg class="icon-svg" style="width:12px;height:12px"><use href="#i-arrow-' + (best.direction==='PUT'?'down':'up') + '"/></svg> ' + best.direction;
        ctNnChip.style.display = '';
        ctNnChip.className = 'ct-chip ' + (best.nn_approved === true ? 'good' : 'warn');
        ctNnChip.innerHTML = '<svg class="icon-svg" style="width:12px;height:12px"><use href="#i-brain"/></svg> ' + (best.nn_approved === true ? 'NN liberou ' : 'NN monitor ') + (((best.nn_score||0)*100).toFixed(0)) + '%';
        ctAiChip.style.display = '';
        ctAiChip.className = 'ct-chip ' + aiSummary.cls;
        ctAiChip.innerHTML = '<svg class="icon-svg" style="width:12px;height:12px"><use href="#i-zap"/></svg> ' + aiSummary.text;
                if (best.prediction_2m && best.prediction_2m.available) {
                    var predMin = best.prediction_2m.smart_exp || best.prediction_2m.minutes || 2;
                    var predConf = Math.round((best.prediction_2m.confidence || 0) * 100);
                    ctPredChip.style.display = '';
                    ctPredChip.className = 'ct-chip ' + (predMin === 1 ? 'good' : 'warn');
                    ctPredChip.innerHTML = '<svg class="icon-svg" style="width:12px;height:12px"><use href="#i-clock"/></svg> EXP ' + predMin + 'M · ALVO ' + Number(best.prediction_2m.price || 0).toFixed(5) + ' · ' + predConf + '%';
                } else if (timing && timing.available) {
                    ctPredChip.style.display = '';
                    ctPredChip.className = 'ct-chip ' + (timing.action === 'now' ? 'good' : 'warn');
                    ctPredChip.innerHTML = '<svg class="icon-svg" style="width:12px;height:12px"><use href="#i-clock"/></svg> ' + (timing.action === 'now' ? 'TIMING AGORA' : 'TIMING ' + Math.round(Number(timing.wait_seconds || 0)) + 's');
                } else {
                    ctPredChip.style.display = 'none';
                }
  } else {
    iaEntry.style.display = 'none';
    ctDir.style.display = 'none';
        ctNnChip.style.display = 'none';
    ctAiChip.style.display = 'none';
                ctPredChip.style.display = 'none';
  }

  // Candles — SEMPRE usa setData para dados do /api/data (120 velas).
  // update() é reservado SOMENTE para live streaming (1-2 velas).
  var newCandles = (cdata.candles || []).map(function(c) {
    return { time: parseTime(c.t), open: c.o, high: c.h, low: c.l, close: c.c };
  }).filter(function(c) { return !isNaN(c.time) && c.time > 0; });
  newCandles.sort(function(a, b) { return a.time - b.time; });
  if (newCandles.length > 0) {
    candleData = newCandles;
    mainSeries.setData(candleData);
  }
  if (firstRender) {
    mainChart.timeScale().fitContent();
    firstRender = false;
  }
  setTimeout(drawHSOverlay, 100);

  // Pattern list footer
  var patEl = document.getElementById('pat-footer');
        if (renderPatterns.length > 0) {
    patEl.style.display = '';
                patEl.innerHTML = renderPatterns.map(function(p) {
      var bt = p.backtest;
    var cls = 'live', icoRef = '#i-clock', txt = (p.signal_active === false ? 'SINAL' : 'LIVE');
      if (bt) {
        if (bt.result === 'win') { cls = 'win'; icoRef = '#i-check'; txt = 'WIN'; }
        else if (bt.result === 'loss') { cls = 'loss'; icoRef = '#i-x'; txt = 'LOSS'; }
        else { cls = 'skip'; icoRef = '#i-arrow-down'; txt = 'SKIP'; }
      } else if (p.broker_status === 'win') { cls = 'win'; icoRef = '#i-check'; txt = 'WIN'; }
      else if (p.broker_status === 'loss') { cls = 'loss'; icoRef = '#i-x'; txt = 'LOSS'; }
      else if (p.broker_status === 'tie') { cls = 'skip'; icoRef = '#i-arrow-down'; txt = 'TIE'; }
    var typeName = p.type === 'DOUBLE_TOP' ? 'DT \u25BC' : 'DB \u25B2';
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
        var selectedLiveSignals = (latestData.live_signals || []).filter(function(sig) {
                return sig.ativo === selectedAtivo;
        });
        var selectedBrokerEntries = (latestData.broker_entries || []).filter(function(entry) {
                return entry.ativo === selectedAtivo;
        });

    function nearestChartIdxForTs(epochTs) {
        var target = Number(epochTs || 0);
        if (!(target > 0) || !candleData.length) return -1;
        var bestIdx = -1;
        var bestDelta = Number.MAX_SAFE_INTEGER;
        candleData.forEach(function(candle, idx) {
            var delta = Math.abs(Number(candle.time || 0) - target);
            if (delta < bestDelta) {
                bestDelta = delta;
                bestIdx = idx;
            }
        });
        return bestIdx;
    }

    function patternReferenceTs(pat) {
        var candidates = [
            pat && pat.broker_ts,
            pat && pat.entry_ts,
            pat && pat.scan_ts,
            pat && pat.right_shoulder && pat.right_shoulder.ts,
            pat && pat.head && pat.head.ts
        ];
        for (var i = 0; i < candidates.length; i++) {
            var ts = Number(candidates[i] || 0);
            if (ts > 0) return ts;
        }
        return 0;
    }

    function findActiveTradeForPattern(pat) {
        var refTs = patternReferenceTs(pat);
        var bestTrade = null;
        var bestDelta = Number.MAX_SAFE_INTEGER;
        for (var i = 0; i < selectedBrokerEntries.length; i++) {
            var trade = selectedBrokerEntries[i];
            if (trade.result !== 'entry') continue;
            if (trade.dir !== pat.direction) continue;
            var tradeTs = Number(trade.ts || 0);
            var delta = (tradeTs > 0 && refTs > 0) ? Math.abs(tradeTs - refTs) : 0;
            if (!bestTrade || delta < bestDelta) {
                bestTrade = trade;
                bestDelta = delta;
            }
        }
        return bestTrade;
    }

    function choosePrimaryOverlayPattern(patterns) {
        if (!patterns || !patterns.length) return [];
        var ranked = patterns.filter(function(p) {
            return p && p.right_shoulder && p.right_shoulder.chart_idx >= 0;
        }).sort(function(a, b) {
            var aLive = a && !a.backtest ? 1 : 0;
            var bLive = b && !b.backtest ? 1 : 0;
            if (aLive !== bLive) return bLive - aLive;
            var aActive = a && a.signal_active === false ? 0 : 1;
            var bActive = b && b.signal_active === false ? 0 : 1;
            if (aActive !== bActive) return bActive - aActive;
            var aTs = patternReferenceTs(a);
            var bTs = patternReferenceTs(b);
            if (aTs !== bTs) return bTs - aTs;
            return Number(b && b.ia_prob || 0) - Number(a && a.ia_prob || 0);
        });
        return ranked.length ? [ranked[0]] : [];
    }

  choosePrimaryOverlayPattern(cd.patterns).forEach(function(pat) {
    var ls = pat.left_shoulder, hd = pat.head, rs = pat.right_shoulder, v1 = pat.valley1, v2 = pat.valley2;
    if (!ls || !hd || !rs) return;
    var lsi = ls.chart_idx, hdi = hd.chart_idx, rsi = rs.chart_idx;
    var v1i = v1 ? v1.chart_idx : -1, v2i = v2 ? v2.chart_idx : -1;
        // Pular apenas se o ombro direito está fora do gráfico (padrão totalmente invisível)
        if (rsi < 0 || rsi >= candleData.length) return;
        // Clamp pontos à esquerda que estão fora do range (desenhar da borda)
        if (lsi < 0) lsi = 0;
        if (hdi < 0) hdi = 0;
        if (v1i >= 0 && v1i >= candleData.length) v1i = candleData.length - 1;
        if (v2i >= 0 && v2i >= candleData.length) v2i = candleData.length - 1;
    var lsx = gx(lsi), lsy = gy(ls.price), hdx = gx(hdi), hdy = gy(hd.price), rsx = gx(rsi), rsy = gy(rs.price);
    if ([lsx,lsy,hdx,hdy,rsx,rsy].some(function(v){return v===null||isNaN(v)})) return;

    var isDT = pat.mode === 'double_touch';
    var isBear = pat.type === 'DOUBLE_TOP';
        var isAuthoritative = pat.overlay_authoritative === true;

    // NN-based colors for live DT patterns (sem backtest)
    var nnApproved = pat.nn_approved;
    var isLive = !pat.backtest;
    var mainC, mainCa;
        if (isDT && isLive && isAuthoritative) {
            mainC = '#a855f7'; mainCa = 'rgba(168,85,247,0.18)';  // roxo = modelo live do bot
        } else if (isDT && isLive && nnApproved === true) {
      mainC = '#00e676'; mainCa = 'rgba(0,230,118,0.18)';  // verde = NN aprovou
    } else if (isDT && isLive && nnApproved === false) {
      mainC = '#ff3d57'; mainCa = 'rgba(255,61,87,0.12)';   // vermelho = NN rejeitou
    } else if (isDT && isLive) {
      mainC = '#6b7280'; mainCa = 'rgba(107,114,128,0.12)';  // cinza = sem modelo
    } else if (isDT) {
      mainC = '#a855f7'; mainCa = 'rgba(168,85,247,0.15)';   // roxo = histórico
    } else {
      mainC = isBear ? '#ff3d57' : '#00e676';
      mainCa = isBear ? 'rgba(255,61,87,0.15)' : 'rgba(0,230,118,0.15)';
    }

    // ── Fill area (shoulder-to-shoulder shape) ──
    var hasV = v1i >= 0 && v2i >= 0;
    var v1x, v1y, v2x, v2y;
        var dtPivotX = null, dtPivotY = null, dtTouchY = null, dtCenterX = null;
    if (hasV) {
      // Clamp valley indices dentro do range visível
      var v1ic = Math.max(0, Math.min(v1i, candleData.length - 1));
      var v2ic = Math.max(0, Math.min(v2i, candleData.length - 1));
      v1x = gx(v1ic); v1y = gy(v1.price); v2x = gx(v2ic); v2y = gy(v2.price);
      if ([v1x,v1y,v2x,v2y].some(function(v){return v===null||isNaN(v)})) hasV = false;

            // ── FIX: Garantir que V1 (pico/vale) fique ENTRE T1 e T2 visualmente ──
            // Se o pico detectado está na borda (mesma posição de T1 ou fora),
            // reposicionar para o ponto médio entre T1 e T2 para desenho correto (W/M).
            if (isDT && hasV) {
                var minVx = Math.min(lsx, rsx);
                var maxVx = Math.max(lsx, rsx);
                var margin = Math.max(10, (maxVx - minVx) * 0.15);
                if (v1x <= minVx + margin || v1x >= maxVx - margin) {
                    v1x = (lsx + rsx) / 2;  // Centralizar para forma W/M correta
                }
                if (v2x <= minVx + margin || v2x >= maxVx - margin) {
                    v2x = (lsx + rsx) / 2;
                }
            }

            if (isDT && hasV) {
                dtPivotX = v1x;
                dtPivotY = v1y;
                dtCenterX = (lsx + rsx) / 2;
                dtTouchY = (lsy + rsy) / 2;
                if (dtTouchY === null || isNaN(dtTouchY)) dtTouchY = null;
            }
    }

    if (hasV) {
      if (isDT) {
        // ═══════════════════════════════════════════════════════
        // DOUBLE TOP / DOUBLE BOTTOM — desenho padrao tecnico
        // DT = forma M (dois topos + vale)  DB = forma W (dois fundos + pico)
        // Referencia: Investopedia / TradingView canonical
        // ═══════════════════════════════════════════════════════

        // Nivel medio dos dois toques (resistencia / suporte)
        var touchMidY = (lsy + rsy) / 2;

        // ── 1. FILL sutil: triangulo entre linha T1-T2 e V-path ──
        ctx.save();
        ctx.globalAlpha = 0.10;
        ctx.fillStyle = mainC;
        ctx.beginPath();
        ctx.moveTo(lsx, lsy);
        ctx.lineTo(v1x, v1y);
        ctx.lineTo(rsx, rsy);
        ctx.closePath();
        ctx.fill();
        ctx.restore();

        // ── 2. NECKLINE: tracejado horizontal no nivel do vale/pico ──
        var nkExtL = gx(Math.max(0, lsi - 3)) || lsx;
        var nkExtR = gx(Math.min(candleData.length - 1, rsi + 6)) || rsx;
        ctx.save();
        ctx.strokeStyle = 'rgba(59,130,246,0.65)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 5]);
        ctx.beginPath();
        ctx.moveTo(nkExtL, v1y);
        ctx.lineTo(nkExtR, v1y);
        ctx.stroke();
        ctx.restore();

        // ── 3. RESISTENCIA/SUPORTE: tracejado horizontal no nivel dos toques ──
        var resExtL = gx(Math.max(0, lsi - 2)) || lsx;
        var resExtR = gx(Math.min(candleData.length - 1, rsi + 4)) || rsx;
        ctx.save();
        ctx.strokeStyle = 'rgba(168,85,247,0.55)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([8, 5]);
        ctx.beginPath();
        ctx.moveTo(resExtL, touchMidY);
        ctx.lineTo(resExtR, touchMidY);
        ctx.stroke();
        ctx.restore();

        // ── 4. V-PATH principal: T1 → Vale/Pico → T2  (a forma M ou W) ──
        ctx.save();
        ctx.strokeStyle = mainC;
        ctx.lineWidth = 2.5;
        ctx.setLineDash([]);
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(lsx, lsy);
        ctx.lineTo(v1x, v1y);
        ctx.lineTo(rsx, rsy);
        ctx.stroke();
        ctx.restore();

        // ── 5. MARCADORES: circulos nos 3 pontos-chave ──
        // T1 (toque 1)
        ctx.fillStyle = mainC;
        ctx.beginPath(); ctx.arc(lsx, lsy, 6, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(lsx, lsy, 6, 0, Math.PI * 2); ctx.stroke();

        // T2 (toque 2)
        ctx.fillStyle = mainC;
        ctx.beginPath(); ctx.arc(rsx, rsy, 6, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(rsx, rsy, 6, 0, Math.PI * 2); ctx.stroke();

        // Vale/Pico central (pivot)
        ctx.save();
        ctx.shadowColor = 'rgba(59,130,246,0.5)'; ctx.shadowBlur = 8;
        ctx.fillStyle = 'rgba(59,130,246,0.95)';
        ctx.beginPath(); ctx.arc(v1x, v1y, 5, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(v1x, v1y, 5, 0, Math.PI * 2); ctx.stroke();

        // ── 6. BADGE "DT" ou "DB" ──
        var badgeCX = (lsx + rsx) / 2;
        var badgeCY = (touchMidY + v1y) / 2;
        var badgeText = isBear ? 'DT' : 'DB';
        ctx.font = '700 10px Inter, sans-serif'; ctx.textAlign = 'center';
        var badgeW = ctx.measureText(badgeText).width + 14;
        ctx.fillStyle = 'rgba(168,85,247,0.90)';
        ctx.beginPath(); ctx.roundRect(badgeCX - badgeW / 2, badgeCY - 9, badgeW, 18, 8); ctx.fill();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.roundRect(badgeCX - badgeW / 2, badgeCY - 9, badgeW, 18, 8); ctx.stroke();
        ctx.fillStyle = '#fff'; ctx.textBaseline = 'middle';
        ctx.fillText(badgeText, badgeCX, badgeCY + 1);
        ctx.textBaseline = 'alphabetic';

      } else {
        // ── H&S path (legado, desativado) ──
        ctx.fillStyle = mainCa; ctx.beginPath();
        ctx.moveTo(lsx, lsy); ctx.lineTo(v1x, v1y); ctx.lineTo(hdx, hdy); ctx.lineTo(v2x, v2y); ctx.lineTo(rsx, rsy);
        ctx.closePath(); ctx.fill();
        ctx.strokeStyle = mainC; ctx.lineWidth = 2.5; ctx.setLineDash([]); ctx.globalAlpha = 0.9;
        ctx.beginPath(); ctx.moveTo(lsx, lsy); ctx.lineTo(v1x, v1y); ctx.lineTo(hdx, hdy); ctx.lineTo(v2x, v2y); ctx.lineTo(rsx, rsy);
        ctx.stroke(); ctx.globalAlpha = 1;

        ctx.strokeStyle = 'rgba(59,130,246,0.7)'; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
        ctx.beginPath(); ctx.moveTo(v1x, v1y); ctx.lineTo(v2x, v2y);
        var ndx = v2x - v1x, ndy = v2y - v1y, nl = Math.sqrt(ndx * ndx + ndy * ndy);
        if (nl > 0) ctx.lineTo(v2x + (ndx / nl) * 140, v2y + (ndy / nl) * 140);
        ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(59,130,246,0.8)';
        ctx.beginPath(); ctx.arc(v1x, v1y, 4, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(v2x, v2y, 4, 0, Math.PI * 2); ctx.fill();

        [{ x: lsx, y: lsy }, { x: rsx, y: rsy }].forEach(function(pt) {
          ctx.fillStyle = mainC; ctx.beginPath(); ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2); ctx.fill();
          ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2); ctx.stroke();
        });
        var headCol = '#a855f7';
        ctx.shadowColor = headCol; ctx.shadowBlur = 12;
        ctx.fillStyle = headCol; ctx.beginPath(); ctx.arc(hdx, hdy, 9, 0, Math.PI * 2); ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(hdx, hdy, 9, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = '#000'; ctx.font = 'bold 11px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('H', hdx, hdy);
        ctx.textBaseline = 'alphabetic';
      }
    } else {
      // No valleys: minimal fallback
      ctx.strokeStyle = mainC; ctx.lineWidth = 2.5; ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(lsx, lsy); ctx.lineTo(hdx, hdy); ctx.lineTo(rsx, rsy); ctx.stroke();
      [{ x: lsx, y: lsy }, { x: rsx, y: rsy }].forEach(function(pt) {
        ctx.fillStyle = mainC; ctx.beginPath(); ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2); ctx.stroke();
      });
    }

    // ── Offsets para labels e setas ──
    var neckAvgY = hasV ? v1y : (lsy + rsy) / 2;
    var touchAvgY = (lsy + rsy) / 2;
        var patDepthPx = (isDT && hasV) ? (Math.abs(touchAvgY - v1y) || 40) : (Math.abs(hdy - neckAvgY) || 40);
    var oU = Math.max(8, patDepthPx * 0.12);
        var arrowOff = oU * 2;
        var zoneLabelX = isDT ? ((lsx + rsx) / 2) : hdx;
        var zoneLabelY = isDT ? touchAvgY : hdy;

    // ── Labels: Toque 1, Toque 2, Resistencia/Suporte, Neckline ──
    ctx.font = '600 10px Inter, sans-serif'; ctx.textAlign = 'center';
    ctx.fillStyle = mainC;
    ctx.fillText('Toque 1', lsx, isBear ? lsy - oU : lsy + oU + 4);
    ctx.fillText('Toque 2', rsx, isBear ? rsy - oU : rsy + oU + 4);
    ctx.fillStyle = '#a855f7'; ctx.font = '700 11px Inter, sans-serif';
        ctx.fillText(isBear ? 'RESIST\xCANCIA' : 'SUPORTE', zoneLabelX, isBear ? zoneLabelY - oU - 2 : zoneLabelY + oU + 6);
    if (isDT && hasV) {
      ctx.fillStyle = 'rgba(59,130,246,0.8)'; ctx.font = '600 9px Inter, sans-serif';
      ctx.fillText('Neckline', v1x, isBear ? v1y + 14 : v1y - 10);
    }

    // ── NN Label (badge) para padrões live ──
    if (isDT && isLive) {
      var nnLabel, nnColor;
      if (nnApproved === true) { nnLabel = '\u2705 NN ' + ((pat.nn_score||0)*100).toFixed(0) + '%'; nnColor = '#00e676'; }
      else if (nnApproved === false) { nnLabel = '\u274C NN ' + ((pat.nn_score||0)*100).toFixed(0) + '%'; nnColor = '#ff3d57'; }
      else { nnLabel = '\u2754 SEM MODELO'; nnColor = '#6b7280'; }
            var nnAnchorX = isDT ? zoneLabelX : hdx;
            var nnAnchorY = isDT ? zoneLabelY : hdy;
            var nnY = isBear ? nnAnchorY - oU - 14 : nnAnchorY + oU + 18;
      ctx.font = 'bold 10px Inter, sans-serif'; ctx.textAlign = 'center';
      var tw = ctx.measureText(nnLabel).width + 10;
      ctx.fillStyle = 'rgba(0,0,0,0.7)';
      ctx.beginPath();
            ctx.roundRect(nnAnchorX - tw/2, nnY - 9, tw, 16, 4);
      ctx.fill();
      ctx.fillStyle = nnColor;
            ctx.fillText(nnLabel, nnAnchorX, nnY + 2);

    var aiSummary = getAiConsensusSummary(pat);
    var aiColor = aiSummary.cls === 'good' ? '#86efac' : '#fdba74';
    var aiY = isBear ? nnY - 18 : nnY + 18;
    var aiWidth = ctx.measureText(aiSummary.text).width + 10;
    ctx.fillStyle = 'rgba(0,0,0,0.72)';
    ctx.beginPath();
    ctx.roundRect(nnAnchorX - aiWidth/2, aiY - 9, aiWidth, 16, 4);
    ctx.fill();
    ctx.fillStyle = aiColor;
    ctx.fillText(aiSummary.text, nnAnchorX, aiY + 2);
    }

    // ── Entry marker at confirmation candle (Close price) ──
        var timingHint = pat.timing_hint || {};
        var activeTrade = findActiveTradeForPattern(pat);
        var waitingSignal = !!(isLive && !activeTrade && timingHint.available && timingHint.action && timingHint.action !== 'now');
        var shouldDrawEntry = true;
        var entryLabel = (!activeTrade && isLive) ? 'SINAL' : 'ENTRADA';
        var entryChartIdx = (pat.entry_chart_idx != null) ? pat.entry_chart_idx : rsi;
        var entryPrice = pat.entry_price || (pat.backtest && pat.backtest.entry_price) || rs.price;
        if (activeTrade && Number(activeTrade.price || 0) > 0) {
            entryPrice = Number(activeTrade.price || 0);
            entryChartIdx = nearestChartIdxForTs(activeTrade.ts);
            if (entryChartIdx < 0) entryChartIdx = candleData.length - 1;
            entryLabel = 'ENTRADA REAL';
        } else if (waitingSignal) {
            shouldDrawEntry = false;
        }
        if (shouldDrawEntry && entryChartIdx >= 0 && entryChartIdx < candleData.length) {
      var ex = gx(entryChartIdx), ey = gy(entryPrice);
      if (ex !== null && ey !== null && !isNaN(ex) && !isNaN(ey)) {
        var esleft = gx(Math.max(0, rsi - 2));
        var esright = gx(Math.min(candleData.length - 1, entryChartIdx + 12));
        if (esleft && esright) {
          ctx.setLineDash([5, 3]); ctx.strokeStyle = '#ff6a00'; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.moveTo(esleft, ey); ctx.lineTo(esright, ey); ctx.stroke(); ctx.setLineDash([]);
        }
        var entryDrawX = gx(Math.min(candleData.length - 1, entryChartIdx + 1)) || ex;
        ctx.shadowColor = '#ff6a00'; ctx.shadowBlur = 10;
        ctx.fillStyle = '#ff6a00'; ctx.beginPath(); ctx.arc(entryDrawX, ey, 10, 0, Math.PI*2); ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(entryDrawX, ey, 10, 0, Math.PI*2); ctx.stroke();
        ctx.fillStyle = '#000'; ctx.font = 'bold 12px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(isBear ? '\u25BC' : '\u25B2', entryDrawX, ey);
        ctx.textBaseline = 'alphabetic';
        ctx.fillStyle = '#ff6a00'; ctx.font = '700 10px Inter, sans-serif'; ctx.textAlign = 'left';
                ctx.fillText(entryLabel + ' ' + entryPrice.toFixed(5), entryDrawX + 16, ey + 4);
      }
    }

        if (waitingSignal) {
            var waitText = timingHint.action === 'wait_retest_zone'
                ? 'AGUARDAR RETESTE'
                : 'AGUARDAR ' + Math.round(Number(timingHint.wait_seconds || 0)) + 's';
            var waitY = isBear ? rsy + arrowOff + 20 : rsy - arrowOff - 20;
            ctx.font = 'bold 10px Inter, sans-serif';
            ctx.textAlign = 'center';
            var waitWidth = ctx.measureText(waitText).width + 14;
            ctx.fillStyle = 'rgba(15,23,42,0.82)';
            ctx.beginPath();
            ctx.roundRect(rsx - waitWidth / 2, waitY - 10, waitWidth, 18, 5);
            ctx.fill();
            ctx.strokeStyle = 'rgba(249,115,22,0.35)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.roundRect(rsx - waitWidth / 2, waitY - 10, waitWidth, 18, 5);
            ctx.stroke();
            ctx.fillStyle = '#fdba74';
            ctx.fillText(waitText, rsx, waitY + 2);
            ctx.textAlign = 'left';
        }

    // ── Stop line ──
    if (pat.stop) {
      var stopY = gy(pat.stop);
      if (stopY !== null && !isNaN(stopY)) {
        var sL = gx(Math.max(0, rsi - 1)), sR = gx(Math.min(candleData.length - 1, rsi + 12));
        if (sL && sR) {
          ctx.setLineDash([3, 3]); ctx.strokeStyle = 'rgba(255,61,87,0.6)'; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(sL, stopY); ctx.lineTo(sR, stopY); ctx.stroke(); ctx.setLineDash([]);
          ctx.fillStyle = 'rgba(255,61,87,0.7)'; ctx.font = '600 9px Inter, sans-serif'; ctx.textAlign = 'left';
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
          ctx.setLineDash([3, 3]); ctx.strokeStyle = 'rgba(0,230,118,0.6)'; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(tL, targetY); ctx.lineTo(tR, targetY); ctx.stroke(); ctx.setLineDash([]);
          ctx.fillStyle = 'rgba(0,230,118,0.7)'; ctx.font = '600 9px Inter, sans-serif'; ctx.textAlign = 'left';
          ctx.fillText('META', rsx + 14, targetY + (isBear ? 14 : -6));
        }
      }
    }

    // ── Direction arrow + label (usa direction real, pode ser invertido) ──
    var actualDir = pat.direction || (isBear ? 'PUT' : 'CALL');
    var isActualPut = actualDir === 'PUT';
    var wasInverted = pat.inverted === true;
    var directionColor = waitingSignal ? '#f59e0b' : (isActualPut ? '#ff3d57' : '#00e676');
    var directionLabel = waitingSignal ? ('SINAL ' + actualDir) : actualDir;
    if (wasInverted) { directionLabel = '\u{1F504} ' + directionLabel; directionColor = '#f59e0b'; }
    ctx.fillStyle = directionColor;
    ctx.beginPath();
    if (isActualPut) { ctx.moveTo(rsx, rsy + arrowOff); ctx.lineTo(rsx - 7, rsy + arrowOff - 10); ctx.lineTo(rsx + 7, rsy + arrowOff - 10); }
    else { ctx.moveTo(rsx, rsy - arrowOff); ctx.lineTo(rsx - 7, rsy - arrowOff + 10); ctx.lineTo(rsx + 7, rsy - arrowOff + 10); }
    ctx.closePath(); ctx.fill();
    ctx.font = '800 11px Inter, sans-serif'; ctx.textAlign = 'left';
    if (isActualPut) { ctx.fillStyle = directionColor; ctx.fillText(directionLabel, rsx + 12, rsy + arrowOff - 1); }
    else { ctx.fillStyle = directionColor; ctx.fillText(directionLabel, rsx + 12, rsy - arrowOff + 5); }

    // ── Result badge + IA prob + Pattern name ──
    var bt = pat.backtest;
    if (bt) {
      var res = bt.result, rC, rL;
      if (res === 'win') { rC = '#00e676'; rL = 'WIN'; }
      else if (res === 'loss') { rC = '#ff3d57'; rL = 'LOSS'; }
      else { rC = '#6b7280'; rL = 'LIVE'; }
      var rx = rsx + 20, ry = isBear ? rsy + arrowOff + 16 : rsy - arrowOff - 20;
      ctx.font = 'bold 10px Inter, sans-serif'; ctx.textAlign = 'left';
      var rtw = ctx.measureText(rL).width + 12;
      ctx.fillStyle = rC; ctx.globalAlpha = 0.2;
      ctx.beginPath(); ctx.roundRect(rx, ry - 9, rtw, 16, 4); ctx.fill();
      ctx.globalAlpha = 1; ctx.fillStyle = rC;
      ctx.fillText(rL, rx + 6, ry + 2);
      // IA prob
      if (pat.ia_prob) {
        ctx.fillStyle = '#a855f7';
        ctx.fillText('IA ' + (pat.ia_prob * 100).toFixed(0) + '%', rx + rtw + 6, ry + 2);
      }
      // Pattern name
      var pName = isDT ? (isBear ? 'Double Top' : 'Double Bottom') : (isBear ? 'H&S' : 'IH&S');
      ctx.fillStyle = 'rgba(255,255,255,0.4)'; ctx.font = '600 9px Inter, sans-serif';
      ctx.fillText(pName, rx, ry + 14);
    }
  });

    if (selectedLiveSignals.length > 0) {
        var bestLiveSignal = selectedLiveSignals.reduce(function(a, b) {
            var aScore = Math.max(a.nn_score || 0, a.ia_prob || 0);
            var bScore = Math.max(b.nn_score || 0, b.ia_prob || 0);
            return aScore >= bScore ? a : b;
        });
        var pred = bestLiveSignal.prediction_2m || {};
        var predY = gy(pred.price);
        var lastX = gx(candleData.length - 1);
        var prevX = gx(Math.max(0, candleData.length - 2));
        if (pred.available && predY !== null && !isNaN(predY) && lastX !== null && !isNaN(lastX)) {
            var step = (prevX !== null && !isNaN(prevX)) ? Math.max(18, lastX - prevX) : 24;
            var startX = Math.max(18, lastX - step * 0.35);
            var endX = Math.min(r.width - 16, lastX + step * 2.2);
            var isPutPred = bestLiveSignal.direction === 'PUT';
            var predColor = isPutPred ? 'rgba(255,138,76,0.98)' : 'rgba(46,214,167,0.98)';
            var predGlow = isPutPred ? 'rgba(255,138,76,0.30)' : 'rgba(46,214,167,0.28)';
            var predFill = isPutPred ? 'rgba(33,16,10,0.82)' : 'rgba(7,32,28,0.82)';
            var predBorder = isPutPred ? 'rgba(255,170,120,0.45)' : 'rgba(125,255,223,0.38)';
            var labelText = 'ALVO ' + (pred.smart_exp || pred.minutes || 2) + 'M  ' + bestLiveSignal.direction + '  ' + Number(pred.price || 0).toFixed(5);

            ctx.save();
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.shadowColor = predGlow;
            ctx.shadowBlur = 16;
            ctx.strokeStyle = predGlow;
            ctx.lineWidth = 7;
            ctx.setLineDash([12, 8]);
            ctx.beginPath();
            ctx.moveTo(startX, predY);
            ctx.lineTo(endX, predY);
            ctx.stroke();

            ctx.shadowBlur = 0;
            ctx.strokeStyle = predColor;
            ctx.lineWidth = 2.5;
            ctx.setLineDash([10, 6]);
            ctx.beginPath();
            ctx.moveTo(startX, predY);
            ctx.lineTo(endX, predY);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.fillStyle = predGlow;
            ctx.beginPath();
            ctx.arc(lastX, predY, 10, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = predColor;
            ctx.beginPath();
            ctx.arc(lastX, predY, 4.5, 0, Math.PI * 2);
            ctx.fill();

            ctx.strokeStyle = 'rgba(255,255,255,0.92)';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(lastX, predY, 4.5, 0, Math.PI * 2);
            ctx.stroke();

            ctx.font = '700 10px IBM Plex Sans, system-ui, sans-serif';
            var labelWidth = Math.max(132, Math.ceil(ctx.measureText(labelText).width) + 18);
            var labelHeight = 24;
            var labelX = Math.min(r.width - labelWidth - 10, endX - labelWidth + 8);
            var labelY = Math.max(10, Math.min(r.height - labelHeight - 10, predY - 18));

            ctx.fillStyle = predFill;
            ctx.beginPath();
            ctx.roundRect(labelX, labelY, labelWidth, labelHeight, 8);
            ctx.fill();

            ctx.strokeStyle = predBorder;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.roundRect(labelX, labelY, labelWidth, labelHeight, 8);
            ctx.stroke();

            ctx.fillStyle = predColor;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.fillText(labelText, labelX + 10, labelY + (labelHeight / 2) + 0.5);
            ctx.restore();

            var currentPrice = Number(pred.current_price || 0);
            if (currentPrice > 0) {
                var curY = gy(currentPrice);
                if (curY !== null && !isNaN(curY)) {
                    ctx.strokeStyle = 'rgba(255,255,255,0.16)';
                    ctx.lineWidth = 1;
                    ctx.setLineDash([3, 5]);
                    ctx.beginPath();
                    ctx.moveTo(lastX, curY);
                    ctx.lineTo(lastX, predY);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }
            }
        }
    }
}

function buildSidebar(data) {
  var list = document.getElementById('asset-list');
  var charts = data.charts || {};
  var s = data.summary || {};
  var byAsset = s.by_asset || {};
  var liveSet = new Set((data.live_signals || []).map(function(l){return l.ativo}));
    var selectedSet = new Set(data.selected_assets || []);
    var minPatterns = data.min_visible_patterns || 4;

  allAtivos = Object.keys(charts).sort(function(a, b) {
        var aSelected = selectedSet.has(a) ? 1 : 0, bSelected = selectedSet.has(b) ? 1 : 0;
        if (aSelected !== bSelected) return bSelected - aSelected;
    var aL = liveSet.has(a) ? 1 : 0, bL = liveSet.has(b) ? 1 : 0;
    if (aL !== bL) return bL - aL;
        var aEligible = (charts[a] || {}).meets_min_patterns ? 1 : 0;
        var bEligible = (charts[b] || {}).meets_min_patterns ? 1 : 0;
        if (aEligible !== bEligible) return bEligible - aEligible;
        var aP = (charts[a] || {}).visible_patterns || 0;
        var bP = (charts[b] || {}).visible_patterns || 0;
        if (aP !== bP) return bP - aP;
    var aW = (byAsset[a] || {}).wr || 0, bW = (byAsset[b] || {}).wr || 0;
    if (aW !== bW) return bW - aW;
    return a.localeCompare(b);
  });

    var prioritizedAtivos = allAtivos.filter(function(ativo) {
        var cd = charts[ativo] || {};
        return selectedSet.has(ativo) || liveSet.has(ativo) || cd.meets_min_patterns;
    });
    var displayAtivos = prioritizedAtivos.length > 0 ? prioritizedAtivos : allAtivos;

    list.innerHTML = displayAtivos.map(function(ativo) {
    var cd = charts[ativo], ad = byAsset[ativo] || {};
    var pats = cd.patterns || [];
    var wins = pats.filter(function(p){return (p.backtest||{}).result === 'win'}).length;
    var losses = pats.filter(function(p){return (p.backtest||{}).result === 'loss'}).length;
        var visibleCount = cd.visible_patterns || pats.length;
        var live = cd.live_patterns || pats.filter(function(p){return !p.backtest}).length;
    var total = wins + losses;
    var wr = total > 0 ? (wins / total * 100) : 0;
    var wrCls = wr >= 60 ? 'good' : wr >= 45 ? 'mid' : 'bad';
    var isAct = selectedAtivo === ativo ? ' active' : '';
    var dot = liveSet.has(ativo) ? '<span class="live-dot"></span>' : '';
        var scanBadge = selectedSet.has(ativo) ? ' <span style="color:var(--cyan)">scan</span>' : '';
    var iaMax = 0;
    pats.forEach(function(p) { if ((p.ia_prob||0) > iaMax) iaMax = p.ia_prob; });
    var iaBadge = iaMax > 0.5 ? '<span class="ia-badge"><svg class="icon-svg" style="width:10px;height:10px;stroke:var(--purple)"><use href="#i-brain"/></svg> ' + (iaMax*100).toFixed(0) + '%</span>' : '';
        var lowBadge = visibleCount < minPatterns ? ' <span style="color:var(--orange)">min ' + minPatterns + '</span>' : '';
    var id = ativo.replace(/[^a-zA-Z0-9]/g, '_');
    return '<div class="asset-item' + isAct + '" id="ai-' + id + '" data-ativo="' + ativo + '" onclick="selectAsset(\'' + ativo + '\')">' +
      '<div class="a-left">' +
        '<span class="a-name">' + ativo + ' ' + dot + '</span>' +
                '<span class="a-meta"><svg class="icon-svg" style="width:10px;height:10px"><use href="#i-chart"/></svg> ' + visibleCount + ' padroes' + lowBadge + scanBadge + (live > 0 ? ' &middot; <span style="color:var(--orange)">' + live + ' live</span>' : '') + ' ' + iaBadge + '</span>' +
      '</div>' +
      '<div class="a-right">' +
        '<span class="a-payout">' + (cd.payout||0) + '%</span>' +
        (total > 0 ? '<span class="wr-pill ' + wrCls + '">' + wr.toFixed(0) + '%</span>' : '') +
      '</div></div>';
  }).join('');

    if ((!selectedAtivo || displayAtivos.indexOf(selectedAtivo) < 0) && displayAtivos.length > 0) {
        selectAsset(displayAtivos[0]);
    }
}

function getAiConsensusSummary(sig) {
    var consensus = sig && sig.ai_consensus || {};
    var gpt = sig && sig.gpt || {};
    var shadow = sig && sig.shadow_pattern_lib || {};
    var bayesProb = consensus.bayes_prob != null ? Number(consensus.bayes_prob || 0) : Number(sig && sig.ia_prob || 0);
    var parts = ['BAYES ' + (bayesProb * 100).toFixed(0) + '%'];
    var cls = consensus.final_ok === false ? 'warn' : 'good';
    if (gpt.available) {
        if (gpt.approved === true) parts.push('GEN OK ' + Math.round(Number(gpt.confidence || 0)) + '%');
        else if (gpt.approved === false) parts.push('GEN DISC ' + Math.round(Number(gpt.confidence || 0)) + '%');
    } else {
        parts.push('GEN PEND');
        if (cls !== 'warn') cls = 'warn';
    }
    if (shadow.available) {
        parts.push(shadow.agreement ? 'LIB OK' : 'LIB DIVERG');
        if (!shadow.agreement && cls === 'good') cls = 'warn';
    }
    return { text: parts.join(' · '), cls: cls };
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
    var invertBadge = sig.inverted ? '<span style="color:#f59e0b;font-size:10px;font-weight:700"> \uD83D\uDD04 INV</span>' : '';
    var resultBadge = '';
    if (sig.broker_status === 'win') { resultBadge = '<span style="background:#00e676;color:#000;font-size:10px;font-weight:700;padding:1px 6px;border-radius:4px;margin-left:4px">\u2705 WIN</span>'; }
    else if (sig.broker_status === 'loss') { resultBadge = '<span style="background:#ff3d57;color:#fff;font-size:10px;font-weight:700;padding:1px 6px;border-radius:4px;margin-left:4px">\u274C LOSS</span>'; }
    else if (sig.broker_status === 'entry') { resultBadge = '<span style="background:#f59e0b;color:#000;font-size:10px;font-weight:700;padding:1px 6px;border-radius:4px;margin-left:4px">\u23F3 ABERTO</span>'; }
    var nnBadge = '';
        var aiSummary = getAiConsensusSummary(sig);
    if (sig.nn_approved === true) { nnBadge = '<span style="color:#00e676;font-size:10px;font-weight:700"> \u2705 NN ' + ((sig.nn_score||0)*100).toFixed(0) + '%</span>'; }
    else if (sig.nn_approved === false) { nnBadge = '<span style="color:#ff3d57;font-size:10px;font-weight:700"> \u274C NN ' + ((sig.nn_score||0)*100).toFixed(0) + '%</span>'; }
    else { nnBadge = '<span style="color:#6b7280;font-size:10px"> \u2754</span>'; }
    var predBadge = '';
    if (sig.prediction_2m && sig.prediction_2m.available) {
        var pExp = sig.prediction_2m.smart_exp || sig.prediction_2m.minutes || 2;
        var pPrice = Number(sig.prediction_2m.price || 0).toFixed(5);
        var pConf = Math.round((sig.prediction_2m.confidence || 0) * 100);
        predBadge = '<div style="margin-top:4px;font-size:10px;font-weight:700;color:' + (pExp === 1 ? '#00e676' : '#38bdf8') + '"><svg class="icon-svg" style="width:10px;height:10px"><use href="#i-clock"/></svg> EXP ' + pExp + 'M \u2192 ' + pPrice + ' (' + pConf + '%)</div>';
    }
    var metaBadge = '';
    if (sig.target) {
        var metaPrice = Number(sig.target).toFixed(5);
        var metaColor = sig.direction === 'PUT' ? '#ff3d57' : '#00e676';
        metaBadge = '<div style="margin-top:3px;font-size:10px;font-weight:700;color:' + metaColor + '"><svg class="icon-svg" style="width:10px;height:10px;stroke:' + metaColor + '"><use href="#i-target"/></svg> META ' + metaPrice + '</div>';
    }
    return '<div class="signal-card" onclick="selectAsset(\'' + sig.ativo + '\')">' +
      '<div class="sc-top"><span class="sc-name">' + sig.ativo + nnBadge + invertBadge + resultBadge + '</span><span class="sc-dir ' + cls + '"><svg class="icon-svg" style="width:10px;height:10px"><use href="' + dirIcon + '"/></svg> ' + sig.direction + '</span></div>' +
    '<div class="sc-bottom"><span class="sc-type"><svg class="icon-svg" style="width:10px;height:10px"><use href="#i-activity"/></svg> ' + (sig.type==='DOUBLE_TOP'?'DT \u25BC':'DB \u25B2') + ' ' + sig.mode + '</span>' +
      '<span class="sc-prob"><svg class="icon-svg" style="width:12px;height:12px;stroke:var(--purple)"><use href="#i-brain"/></svg> ' + prob + '%' +
            '<span class="prob-bar"><span class="prob-fill" style="width:' + prob + '%"></span></span></span></div>' +
            predBadge + metaBadge +
            '<div style="margin-top:4px;display:flex;gap:6px;flex-wrap:wrap"><span style="font-size:10px;font-weight:700;color:' + (aiSummary.cls === 'good' ? '#86efac' : '#fdba74') + '">' + aiSummary.text + '</span></div></div>';
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
    /* Dedup client-side: se win/loss existe para ativo+dir, remover entry */
  var resolvedKeys = {};
  entries.forEach(function(r) {
    if (r.result === 'win' || r.result === 'loss' || r.result === 'tie') {
      resolvedKeys[(r.ativo||'') + '|' + (r.dir||'') + '|' + Math.floor((r.ts||0)/300)] = true;
    }
  });
  entries = entries.filter(function(r) {
    if (r.result === 'entry') {
      var dk = (r.ativo||'') + '|' + (r.dir||'') + '|' + Math.floor((r.ts||0)/300);
      if (resolvedKeys[dk]) return false;
    }
    return true;
  });
  el.innerHTML = entries.map(function(r, idx) {
    var cls = r.result === 'win' ? 'win' : r.result === 'entry' ? 'entry' : 'loss';
    var icoRef = r.result === 'win' ? '#i-check' : r.result === 'entry' ? '#i-clock' : '#i-x';
    var priceStr = r.price ? parseFloat(r.price).toFixed(5) : '';
    var dirIcon = r.dir === 'PUT' ? '#i-arrow-down' : '#i-arrow-up';
    var profitStr = '';
    if (r.profit && r.profit !== 0) {
      var pf = parseFloat(r.profit);
      profitStr = '<span class="rr-profit">' + (pf > 0 ? '+' : '') + pf.toFixed(2) + '</span>';
    }
        return '<div class="result-row ' + cls + '" onclick="showDecisionPopupByIndex(' + idx + ')" style="cursor:pointer" title="Clique para abrir o painel neural">' +
      '<span class="rr-ativo">' + r.ativo + '</span>' +
      '<span class="rr-dir" style="color:' + (r.dir==='PUT'?'var(--red)':'var(--green)') + '"><svg class="icon-svg" style="width:10px;height:10px"><use href="' + dirIcon + '"/></svg> ' + r.dir + '</span>' +
      (priceStr ? '<span class="rr-price">' + priceStr + '</span>' : '') +
      profitStr +
      '<span class="rr-res"><svg class="icon-svg" style="width:12px;height:12px"><use href="' + icoRef + '"/></svg> ' + r.result.toUpperCase() + '</span>' +
      '<span style="font-size:10px;color:var(--purple);margin-left:4px" title="Ver decisões">🧠</span>' +
      '</div>';
  }).join('');
}

function updateDashboard(data) {
  latestData = data;
  var s = data.summary || {};
  // NN Per-Ativo: mostrar amostras treinadas por ativo
  var nn = data.nn_per_asset || {};
  var nnKeys = Object.keys(nn);
  if (nnKeys.length > 0) {
    var nnParts = nnKeys.map(function(a) {
      var d = nn[a];
      var name = a.replace('-OTC','');
      return name + '=' + (d.samples || 0);
    });
    document.getElementById('st-total').textContent = nnParts.join(' | ');
  } else {
    document.getElementById('st-total').textContent = '-';
  }
  var wr = s.wr || 0;
  var wrEl = document.getElementById('st-wr');
  wrEl.textContent = wr.toFixed(1) + '%';
  wrEl.className = 'val ' + (wr >= 60 ? 'green' : wr >= 45 ? 'yellow' : 'red');
  document.getElementById('st-wins').textContent = s.wins || 0;
  document.getElementById('st-losses').textContent = (s.total || 0) - (s.wins || 0);
    var dt = (s.by_type || {}).DOUBLE_TOP;
    var db = (s.by_type || {}).DOUBLE_BOTTOM;
  document.getElementById('st-hs').textContent = dt ? dt.wr + '% (' + dt.total + ')' : '-';
  document.getElementById('st-ihs').textContent = db ? db.wr + '% (' + db.total + ')' : '-';
  document.getElementById('st-live').textContent = (data.live_signals || []).length;
  // IA Level — mostrar status NN per-ativo
  var iaLvlEl = document.getElementById('st-ia-level');
  var iaLvlBox = document.getElementById('st-ia-level-box');
  var nnReady = 0;
  var nnTotal = 0;
  var nnK = Object.keys(nn);
  for (var ni = 0; ni < nnK.length; ni++) {
    nnTotal++;
    if (nn[nnK[ni]].ml) nnReady++;
  }
  if (nnReady > 0) {
    iaLvlEl.textContent = '\ud83c\udfc6 NN ' + nnReady + '/' + nnTotal + ' ativos';
    iaLvlEl.style.color = '#00e676';
    iaLvlBox.style.borderColor = '#00e676';
  } else if (nnTotal > 0) {
    iaLvlEl.textContent = '\u26a0\ufe0f NN carregando...';
    iaLvlEl.style.color = '#ff6a00';
    iaLvlBox.style.borderColor = '#ff6a00';
  } else {
    var lvl = s.ia_level || {num:1, nome:'Iniciante', emoji:'\ud83c\udf31', cor:'#6b7280'};
    iaLvlEl.textContent = lvl.emoji + ' ' + lvl.num + ' - ' + lvl.nome;
    iaLvlEl.style.color = lvl.cor;
    iaLvlBox.style.borderColor = lvl.cor;
  }
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
setInterval(fetchData, 15000);

/* ═══ LIVE CANDLE STREAMING — atualiza a cada 2s ═══ */
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
      if (Date.now() - _liveFetchStart > 5000) { _liveFetching = false; }
      else return;
    }
    _liveFetchStart = Date.now();
    _liveFetching = true;
    try {
      var r = await fetch('/api/live_candles?ativo=' + encodeURIComponent(selectedAtivo));
      if (!r.ok) { _liveFetching = false; return; }
      var d = await r.json();
      if (!d.candles || d.candles.length === 0 || d.ativo !== selectedAtivo) { _liveFetching = false; return; }
      /* Filtrar e ordenar velas live */
      var liveBars = d.candles.map(function(c) {
        return { time: parseTime(c.t), open: c.o, high: c.h, low: c.l, close: c.c };
      }).filter(function(c) { return !isNaN(c.time) && c.time > 0; });
      liveBars.sort(function(a, b) { return a.time - b.time; });
      /* Atualizar as últimas 5 velas via update() para manter gráfico fresco */
      var barsToUpdate = liveBars.slice(-5);
      barsToUpdate.forEach(function(bar) {
        try { mainSeries.update(bar); } catch(e) { /* timestamp fora de ordem — ignorar */ }
        _lastCandleTime = Math.max(_lastCandleTime, bar.time);
        var found = false;
        for (var i = candleData.length - 1; i >= Math.max(0, candleData.length - 10); i--) {
          if (candleData[i].time === bar.time) { candleData[i] = bar; found = true; break; }
        }
        if (!found) candleData.push(bar);
      });
      /* Sempre redesenhar overlay após atualizar velas */
      requestAnimationFrame(drawHSOverlay);
      _liveUpdateTs = Date.now();
      _liveErrorCount = 0;
    } catch(e) {
      _liveErrorCount++;
    }
    _liveFetching = false;
  }, 2000);
}
/* Iniciar streaming de velas automaticamente */
startLiveCandles();

/* ═══ REDRAW PERIÓDICO — redesenha overlay a cada 30s para manter sincronizado ═══ */
setInterval(function() {
  if (mainChart && mainSeries && selectedAtivo) {
    requestAnimationFrame(drawHSOverlay);
  }
}, 30000);

async function clearTrades() {
  try {
    await fetch('/api/clear_trades', {method:'POST'});
        _decisionsCache = [];
        _decLastFetch = 0;
    var el = document.getElementById('results-list');
    if (el) el.innerHTML = '<div style="color:var(--text-muted);font-size:11px;text-align:center;padding:20px 0">Entradas limpas</div>';
  } catch(e) {}
}

// ═══ DECISION MODAL ═══
var _decisionsCache = [];
var _decLastFetch = 0;

function _fetchDecisions(cb, force) {
  var now = Date.now();
    if (!force && _decisionsCache.length > 0 && now - _decLastFetch < 15000) { cb(_decisionsCache); return; }
    fetch('/api/decisions?_=' + now).then(function(r){return r.json()}).then(function(data){
    _decisionsCache = data || [];
    _decLastFetch = Date.now();
    cb(_decisionsCache);
  }).catch(function(){ cb(_decisionsCache); });
}

function _pct(v){return v!=null?(v*100).toFixed(0)+'%':'N/A'}
function _pct1(v){return v!=null?(v*100).toFixed(1)+'%':'N/A'}
function _f2(v){return v!=null?v.toFixed(2):'N/A'}
function _f4(v){return v!=null?v.toFixed(4):'N/A'}
function _f6(v){return v!=null?v.toFixed(6):'N/A'}
function _money(v){var n=Number(v||0); if(!isFinite(n)) n=0; return (n>0?'+':'') + n.toFixed(2)}
function _vc(v,g,b){return v>=g?'good':v<=b?'bad':'warn'}
function _bc(v,g,m){return v>=g?'var(--green)':v>=m?'#f59e0b':'var(--red)'}
function _bar(lbl,val,color){
  var w=Math.min(100,(val||0)*100);
  return '<div class="dm-bar"><span class="bl">'+lbl+'</span><div class="bbg"><div class="bf" style="width:'+w+'%;background:'+color+'"></div></div><span class="bv" style="color:'+color+'">'+_pct(val)+'</span></div>';
}

function _findDecisionForTrade(decs, trade) {
    if (!trade || !decs || !decs.length) return null;
    var best = null;
    var bestDelta = Number.MAX_SAFE_INTEGER;
    var tradeClock = trade.time || '';
    var tradeTs = Number(trade.ts || 0);
    for (var i = decs.length - 1; i >= 0; i--) {
        var dd = decs[i];
        if (trade.decision_id && dd.decision_id && dd.decision_id === trade.decision_id) return dd;
        if (trade.order_id != null && dd.order_id != null && Number(dd.order_id) === Number(trade.order_id)) return dd;
        if (dd.ativo !== trade.ativo || dd.direcao !== trade.dir) continue;
        if (tradeClock && dd.time && dd.time.indexOf(tradeClock) >= 0) return dd;
        var ddTs = Number(dd.ts || 0);
        if (tradeTs > 0 && ddTs > 0) {
            var delta = Math.abs(ddTs - tradeTs);
            if (delta < bestDelta) {
                bestDelta = delta;
                best = dd;
            }
        } else if (!best) {
            best = dd;
        }
    }
    if (best && bestDelta <= 900) return best;
    return tradeTs > 0 ? null : best;
}

function showDecisionPopupByIndex(index) {
    var entries = (latestData && latestData.broker_entries) || [];
    var trade = entries[index];
    if (!trade) {
        _renderDecisionModal(null, '?', null);
        return;
    }
    _fetchDecisions(function(decs) {
        var found = _findDecisionForTrade(decs, trade);
        if (found) {
            _renderDecisionModal(found, trade.ativo, trade);
            return;
        }
        _fetchDecisions(function(fresh) {
            _renderDecisionModal(_findDecisionForTrade(fresh, trade), trade.ativo, trade);
        }, true);
    }, false);
}

function _renderDecisionModal(d, ativo, trade) {
  var ov = document.getElementById('dm-overlay');
  var box = document.getElementById('dm-content');
  if (!d) {
        if (trade) {
            var tradeResColor = trade.result === 'win' ? 'var(--green)' : trade.result === 'loss' ? 'var(--red)' : '#f59e0b';
            var tradeDirCls = trade.dir === 'CALL' ? 'call' : 'put';
            box.innerHTML = '<div class="dm-shell"><div class="dm-headline"><div class="dm-headcopy"><span class="dm-overline">Resumo reservado</span><div class="dm-title"><span style="font-size:20px">🧾</span><strong>'+trade.ativo+'</strong><span class="dm-direction '+tradeDirCls+'">'+trade.dir+'</span></div><div class="dm-meta">'+(trade.time || 'Sem horario')+' · broker '+(trade.broker || '?')+'</div></div><div class="dm-score-badge warn"><span>Status</span><strong style="color:'+tradeResColor+'">'+(trade.result || 'entry').toUpperCase()+'</strong></div></div>' +
                '<div class="dm-grid">' +
                '<div class="dm-sec"><h4>Resumo do trade</h4>' +
                '<div class="dm-row"><span class="dl">Ativo</span><span class="dv neutral">'+trade.ativo+'</span></div>' +
                '<div class="dm-row"><span class="dl">Direcao</span><span class="dv '+(trade.dir === 'CALL' ? 'good' : 'bad')+'">'+trade.dir+'</span></div>' +
                '<div class="dm-row"><span class="dl">Broker</span><span class="dv neutral">'+(trade.broker || '?')+'</span></div>' +
                '<div class="dm-row"><span class="dl">Entrada</span><span class="dv neutral">'+(trade.price ? Number(trade.price).toFixed(5) : 'N/A')+'</span></div>' +
                '<div class="dm-row"><span class="dl">Resultado</span><span class="dv" style="color:'+tradeResColor+'">'+(trade.result || 'entry').toUpperCase()+'</span></div>' +
                '<div class="dm-row"><span class="dl">P/L</span><span class="dv" style="color:'+tradeResColor+'">'+_money(trade.profit)+'</span></div>' +
                '</div>' +
                '<div class="dm-protect"><strong>Camada neural protegida</strong><p>Os detalhes internos da validacao ficam ocultos ate a sincronizacao completa. Assim o painel continua elegante e evita expor a logica proprietaria antes da hora.</p></div>' +
                '</div></div>';
        } else {
            box.innerHTML = '<div class="dm-nomatch">Nenhum painel neural encontrado para <b>'+ativo+'</b><br><br><span style="font-size:11px;color:var(--text-muted)">O dashboard ainda nao encontrou um registro detalhado para esse trade.</span></div>';
        }
    ov.classList.add('open');
    return;
  }
    var nn = d.nn||{}, gpt = d.gpt||{}, geo = d.geometry||{}, pat = d.pattern||{}, consensus = d.ai_consensus||{}, shadow = d.shadow_pattern_lib||{};
  var status = d.status||'entry';
    var patLabel = d.pat_type==='DOUBLE_TOP' ? 'Double Top' : d.pat_type==='DOUBLE_BOTTOM' ? 'Double Bottom' : (d.pat_type||'Padrão');
    var geoScore = d.geom_score || 0;
    var iaProb = d.ia_prob || 0;
        var bayesProb = consensus.bayes_prob != null ? consensus.bayes_prob : iaProb;
    var nnAvailable = nn.available !== false && (nn.nn_score != null || nn.prob_win != null);
    var nnScore = nn.nn_score != null ? nn.nn_score : null;
    var probWin = nn.prob_win != null ? nn.prob_win : nnScore;
    var nnScoreLabel = nnAvailable ? _pct1(nnScore) : 'Indisponível';
    var probWinLabel = nnAvailable ? _pct1(probWin) : 'Indisponível';
    var nnReason = nn.reason || '';
    var nnSource = nn.source || (nnAvailable ? 'scan/live' : 'não informado');
    var expMin = d.exp_min != null ? d.exp_min : (gpt.exp_minutes != null ? gpt.exp_minutes : '?');
    var nnStateClass = !nnAvailable ? 'warn' : nn.approved===true ? 'good' : nn.approved===false ? 'bad' : 'warn';
        var gptStateClass = !gpt.available ? 'neutral' : gpt.approved===true ? 'good' : gpt.approved===false ? 'warn' : 'neutral';
        var gptStateText = !gpt.available ? 'Nao executada' : gpt.approved===true ? ('Confirmou ' + Math.round(Number(gpt.confidence || 0)) + '%') : ('Discordou ' + Math.round(Number(gpt.confidence || 0)) + '%');
        var shadowStateClass = !shadow.available ? 'neutral' : shadow.agreement ? 'good' : 'warn';
        var shadowStateText = !shadow.available ? 'Sem leitura' : (shadow.agreement ? 'Biblioteca em acordo' : 'Biblioteca divergente');
        var finalStateClass = consensus.final_ok===false ? 'bad' : consensus.gpt_ok===true || consensus.shadow_agreement===true ? 'good' : 'warn';
        var finalStateText = consensus.reason || 'Sem resumo de consenso';
        var nnStateText = !nnAvailable ? (nn.state_text || 'NN indisponível') : (nn.approved===true ? 'Confluencia validada' : nn.approved===false ? 'Confluencia recusada' : 'Sem veredito');
        var qualityBlend = Math.max(0, Math.min(1, (((nnAvailable ? nnScore : 0) || 0) * 0.56) + (iaProb * 0.24) + (Math.min(1, geoScore) * 0.20)));
        var qualityLabel = qualityBlend >= 0.86 ? 'Elite' : qualityBlend >= 0.72 ? 'Forte' : qualityBlend >= 0.58 ? 'Estavel' : 'Agressivo';
        var timingLabel = (d.wick_pct||0) >= 30 ? 'Entrada precisa' : (d.wick_pct||0) >= 15 ? 'Timing valido' : 'Timing curto';
        var structureState = geoScore >= 0.9 ? 'Estrutura premium' : geoScore >= 0.75 ? 'Estrutura consistente' : 'Estrutura observada';
  var geoOk = (geo.symmetry||0)>=0.40 && (geo.span||0)>=12 && (geo.depth_ratio||0)>=2.0;
  var nnMetricClass = nnAvailable ? _vc(nnScore,0.8,0.5) : 'neutral';
  var steps = [
        {n:'Padrão',v:patLabel,p:true},
                {n:'Estrutura',v:qualityLabel,p:geoOk},
                {n:'IA Base',v:_pct(iaProb),p:iaProb>=0.55},
                {n:'Confluencia',v:(nnAvailable ? _pct(nnScore) : 'NN indisponível'),p:nnAvailable && nn.approved===true}
  ];
  var pipe = steps.map(function(s,i){
    var cls = s.p?'pass':'fail';
    return (i>0?'<span class="dm-arrow">→</span>':'')+'<div class="dm-step '+cls+'"><span>'+s.n+'</span><span class="sv">'+s.v+'</span></div>';
  }).join('');

  // Result badge
  var resColor = status==='win'?'var(--green)':status==='loss'?'var(--red)':'#f59e0b';
  var resText = status==='win'?'WIN':status==='loss'?'LOSS':'⏳';
  var resultado = d.resultado!=null?(d.resultado>0?'+':'')+_f2(d.resultado):'';
    var dirCls = d.direcao === 'CALL' ? 'call' : 'put';
    var statePillCls = status==='win' ? 'good' : status==='loss' ? 'bad' : 'warn';

        var html = '<div class="dm-shell">';
        html += '<div class="dm-headline"><div class="dm-headcopy"><span class="dm-overline">Analise reservada da entrada</span><div class="dm-title"><span style="font-size:20px">'+(status==='win'?'✅':status==='loss'?'❌':'⏳')+'</span><strong>'+d.ativo+'</strong><span class="dm-direction '+dirCls+'">'+d.direcao+'</span></div><div class="dm-meta">'+(d.time||'Sem horario')+' · '+(d.mode||'Modo nao informado')+' · exp '+expMin+' min</div></div><div class="dm-score-badge '+nnStateClass+'"><span>Confluencia IA</span><strong>'+nnScoreLabel+'</strong></div></div>';
        html += '<div class="dm-pipeline">'+pipe+'</div>';
        html += '<div class="dm-hero">';
        html += '<div class="dm-focus prime"><div class="dm-focus-head"><div><h4>Resumo executivo</h4><p>Leitura profissional da entrada com auditoria visual, sem expor a engenharia interna da estrategia.</p></div><div class="dm-score-badge '+nnStateClass+'"><span>Prob Win</span><strong>'+probWinLabel+'</strong></div></div>';
        html += '<div class="dm-kpis">';
        html += '<div class="dm-kpi"><span class="kl">Veredito IA</span><span class="kv '+nnStateClass+'">'+nnStateText+'</span></div>';
        html += '<div class="dm-kpi"><span class="kl">Forca do setup</span><span class="kv '+_vc(qualityBlend,0.8,0.6)+'">'+qualityLabel+'</span></div>';
        html += '<div class="dm-kpi"><span class="kl">Contexto base</span><span class="kv '+_vc(iaProb,0.7,0.5)+'">'+_pct1(iaProb)+'</span></div>';
        html += '<div class="dm-kpi"><span class="kl">Timing</span><span class="kv '+_vc(d.wick_pct||0,30,15)+'">'+timingLabel+'</span></div>';
        html += '</div></div>';
        html += '<div class="dm-focus"><h4>Resumo da execucao</h4><div class="dm-pillrow"><span class="dm-pill '+nnStateClass+'">'+nnStateText+'</span><span class="dm-pill neutral">EXP '+expMin+'m</span><span class="dm-pill neutral">'+(d.mode||'Sem modo')+'</span><span class="dm-pill '+statePillCls+'">'+resText+'</span></div>';
        html += '<div class="dm-row"><span class="dl">Entrada</span><span class="dv neutral">'+_f6(d.entry_price)+'</span></div>';
        html += '<div class="dm-row"><span class="dl">Amostras IA</span><span class="dv neutral">'+(d.ia_samples||0)+'</span></div>';
        html += '<div class="dm-row"><span class="dl">Fonte NN</span><span class="dv neutral">'+nnSource+'</span></div>';
        if (nnReason) html += '<div class="dm-row"><span class="dl">Motivo NN</span><span class="dv neutral">'+nnReason+'</span></div>';
        html += '<div class="dm-row"><span class="dl">Estrutura</span><span class="dv '+_vc(geoScore,0.9,0.7)+'">'+structureState+'</span></div>';
        html += '<div class="dm-row"><span class="dl">Broker</span><span class="dv neutral">'+(d.broker || '?')+'</span></div>';
        html += '</div>';
        html += '</div>';
        html += '<div class="dm-grid">';
        html += '<div class="dm-sec"><h4>Leitura liberada</h4>';
        html += '<div class="dm-row"><span class="dl">Padrão</span><span class="dv neutral">'+patLabel+' '+(d.direcao==='PUT'?'(PUT)':'(CALL)')+'</span></div>';
        html += '<div class="dm-row"><span class="dl">Confluencia IA</span><span class="dv '+nnMetricClass+'">'+nnScoreLabel+'</span></div>';
        html += '<div class="dm-row"><span class="dl">Probabilidade</span><span class="dv '+(nnAvailable ? _vc(probWin,0.8,0.5) : 'neutral')+'">'+probWinLabel+'</span></div>';
        html += '<div class="dm-row"><span class="dl">Bayes final</span><span class="dv '+_vc(bayesProb,0.7,0.55)+'">'+_pct1(bayesProb)+'</span></div>';
        html += '<div class="dm-row"><span class="dl">IA generativa</span><span class="dv '+gptStateClass+'">'+gptStateText+'</span></div>';
        html += '<div class="dm-row"><span class="dl">Biblioteca shadow</span><span class="dv '+shadowStateClass+'">'+shadowStateText+'</span></div>';
        html += '<div class="dm-row"><span class="dl">Consenso final</span><span class="dv '+finalStateClass+'">'+finalStateText+'</span></div>';
        html += '<div class="dm-row"><span class="dl">Score geometrico</span><span class="dv '+_vc(geoScore,0.9,0.7)+'">'+_f4(geoScore)+'</span></div>';
        html += '<div class="dm-row"><span class="dl">Rejeicao</span><span class="dv '+_vc(d.wick_pct||0,30,15)+'">'+(d.wick_pct||0)+'%</span></div>';
        html += '</div>';
        html += '<div class="dm-sec"><h4>Faixa operacional</h4>';
        html += '<div class="dm-row"><span class="dl">Entrada</span><span class="dv neutral">'+_f6(d.entry_price)+'</span></div>';
        html += '<div class="dm-row"><span class="dl">Neckline</span><span class="dv" style="color:#60a5fa">'+_f6(pat.neckline)+'</span></div>';
        html += '<div class="dm-row"><span class="dl">Target</span><span class="dv good">'+_f6(pat.target)+'</span></div>';
        html += '<div class="dm-row"><span class="dl">ATR</span><span class="dv neutral">'+_f6(d.atr)+'</span></div>';
        html += '<div class="dm-row"><span class="dl">Resultado</span><span class="dv '+statePillCls+'">'+(d.resultado != null ? resultado : 'Aguardando')+'</span></div>';
        html += '</div>';
        html += '<div class="dm-protect"><strong>Detalhamento interno protegido</strong><p>Pesos, thresholds finos e a formulacao completa da decisao permanecem ocultos para preservar a estrategia.</p>';
        html += '<div class="dm-blurline"><span class="dm-blurkey">Arquitetura interna</span><span class="dm-blurval">private ensemble hidden</span></div>';
        html += '<div class="dm-blurline"><span class="dm-blurkey">Logica de validacao</span><span class="dm-blurval">multi-layer guarded flow</span></div>';
        html += '<div class="dm-blurline"><span class="dm-blurkey">Regra de liberacao</span><span class="dm-blurval">proprietary adaptive gate</span></div>';
        html += '<div class="dm-softnote">Visao publica reduzida para auditoria operacional.</div></div>';
        html += '</div>';

  // Result
  if (d.resultado != null) {
    var rbg = d.resultado>0?'var(--green-bg)':'var(--red-bg)';
    var rborder = d.resultado>0?'rgba(0,230,118,0.3)':'rgba(255,61,87,0.3)';
    html += '<div class="dm-result" style="background:'+rbg+';border:1px solid '+rborder+';color:'+resColor+'">'+(d.resultado>0?'✅ WIN +':'❌ LOSS ')+_f2(d.resultado)+'</div>';
  } else {
    html += '<div class="dm-result" style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.3);color:#f59e0b">⏳ Aguardando resultado...</div>';
  }

    html += '</div>';

  box.innerHTML = html;
  ov.classList.add('open');
}

function closeDecisionModal() {
  document.getElementById('dm-overlay').classList.remove('open');
}

</script>

<!-- Decision Modal Overlay -->
<div class="dm-overlay" id="dm-overlay" onclick="if(event.target===this)closeDecisionModal()">
  <div class="dm-box">
    <button class="dm-close" onclick="closeDecisionModal()">✕</button>
    <div id="dm-content"></div>
  </div>
</div>

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
        if path == "/api/clear_trades":
            with _real_trades_lock:
                _real_trades.clear()
            # Limpar também os arquivos persistentes para que as entradas não voltem
            for _sfx in ("iq", "bullex", "casatrader"):
                _tf = os.path.join(_USER_DIR, f"ws_live_trades_{_sfx}.json")
                try:
                    if os.path.exists(_tf):
                        with open(_tf, "w", encoding="utf-8") as _wf:
                            json.dump({"trades": [], "updated": time.time()}, _wf)
                except Exception:
                    pass
            self.send_response(200)
            self._cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        elif path == "/api/trade":
            # Bot envia trade real em tempo real
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                if body.get("ativo") and body.get("result") in ("win", "loss", "entry", "tie"):
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
                        "decision_id": body.get("decision_id"),
                        "order_id": body.get("order_id"),
                    }
                    # 1) Atualizar in-memory
                    with _real_trades_lock:
                        if new_status in ("win", "loss", "tie"):
                            updated = False
                            for i in range(len(_real_trades) - 1, -1, -1):
                                same_decision = body.get("decision_id") and _real_trades[i].get("decision_id") == body.get("decision_id")
                                same_order = body.get("order_id") is not None and _real_trades[i].get("order_id") == body.get("order_id")
                                fallback_match = _real_trades[i]["ativo"] == body["ativo"] and _real_trades[i]["dir"] == body.get("dir", "?") and _real_trades[i]["result"] == "entry"
                                if same_decision or same_order or fallback_match:
                                    _real_trades[i]["result"] = new_status
                                    _real_trades[i]["profit"] = body.get("profit", 0)
                                    # NÃO alterar ts — manter timestamp original da entrada para dedup funcionar
                                    if body.get("decision_id"):
                                        _real_trades[i]["decision_id"] = body.get("decision_id")
                                    if body.get("order_id") is not None:
                                        _real_trades[i]["order_id"] = body.get("order_id")
                                    updated = True
                                    break
                            if not updated:
                                _real_trades.append(new_entry)
                        else:
                            _real_trades.append(new_entry)
                        if len(_real_trades) > _REAL_TRADES_MAX:
                            del _real_trades[:-_REAL_TRADES_MAX]
                    # 2) Persistir no arquivo (sobrevive a reinícios)
                    _broker = body.get("broker", "bullex")
                    if _broker not in ("iq", "bullex", "casatrader"):
                        _broker = "bullex"
                    _fpath = os.path.join(_USER_DIR, f"ws_live_trades_{_broker}.json")
                    try:
                        _existing = []
                        if os.path.exists(_fpath):
                            with open(_fpath, "r", encoding="utf-8") as _rf:
                                _fdata = json.load(_rf)
                                _existing = _fdata.get("trades", [])
                        _rec = {
                            "ts": new_entry["ts"],
                            "time": body.get("time", "") if len(body.get("time", "")) > 10 else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "ativo": new_entry["ativo"],
                            "dir": new_entry["dir"],
                            "status": new_status,
                            "resultado": new_entry["profit"],
                            "entry_price": new_entry["price"],
                            "stake": new_entry["stake"],
                            "broker": _broker,
                            "decision_id": new_entry.get("decision_id"),
                            "order_id": new_entry.get("order_id"),
                        }
                        if new_status in ("win", "loss", "tie"):
                            _upd = False
                            for _j in range(len(_existing) - 1, -1, -1):
                                same_decision = body.get("decision_id") and _existing[_j].get("decision_id") == body.get("decision_id")
                                same_order = body.get("order_id") is not None and _existing[_j].get("order_id") == body.get("order_id")
                                fallback_match = _existing[_j].get("ativo") == new_entry["ativo"] and _existing[_j].get("dir") == new_entry["dir"] and _existing[_j].get("status") == "entry"
                                if same_decision or same_order or fallback_match:
                                    _existing[_j]["status"] = new_status
                                    _existing[_j]["resultado"] = new_entry["profit"]
                                    _existing[_j]["ts"] = new_entry["ts"]
                                    _existing[_j]["time"] = _rec["time"]
                                    if body.get("decision_id"):
                                        _existing[_j]["decision_id"] = body.get("decision_id")
                                    if body.get("order_id") is not None:
                                        _existing[_j]["order_id"] = body.get("order_id")
                                    _upd = True
                                    break
                            if not _upd:
                                _existing.append(_rec)
                        else:
                            _existing.append(_rec)
                        if len(_existing) > _REAL_TRADES_MAX:
                            _existing = _existing[-_REAL_TRADES_MAX:]
                        with open(_fpath, "w", encoding="utf-8") as _wf:
                            json.dump({"trades": _existing, "updated": time.time()}, _wf, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
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
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
        
        elif path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
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
            self.send_header("Cache-Control", "no-store")
            self._cors_headers()
            self.end_headers()
            try:
                candles = []

                # PRIORIDADE 1: Ler arquivo live do bot
                _live_file = os.path.join(_USER_DIR, "ws_live_candles.json")
                if os.path.exists(_live_file):
                    _max_retries = 2
                    for _retry in range(_max_retries):
                        try:
                            _lf_age = time.time() - os.path.getmtime(_live_file)
                            if _lf_age < 120:  # aceita até 2 min
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
                                break
                        except (json.JSONDecodeError, PermissionError):
                            import time as _time_mod
                            _time_mod.sleep(0.05)
                        except Exception:
                            break

                # FALLBACK: cache DataFrame
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
        
        elif path == "/decisions":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            try:
                _html_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_decisions.html")
                if os.path.exists(_html_file):
                    with open(_html_file, "r", encoding="utf-8") as f:
                        self.wfile.write(f.read().encode("utf-8"))
                else:
                    self.wfile.write(b'<h1>trade_decisions.html nao encontrado</h1>')
            except Exception as e:
                self.wfile.write(f'<h1>Erro: {e}</h1>'.encode("utf-8"))

        elif path == "/api/decisions":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self._cors_headers()
            self.end_headers()
            try:
                _dec_file = os.path.join(_USER_DIR, "ws_trade_decisions.json")
                if os.path.exists(_dec_file):
                    with open(_dec_file, "r", encoding="utf-8-sig") as f:
                        self.wfile.write(f.read().encode("utf-8"))
                else:
                    self.wfile.write(b'[]')
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Dashboard IA DT")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Porta (default: {DEFAULT_PORT})")
    args = parser.parse_args()
    
    log.info(f"{'='*60}")
    log.info(f"  🧠 WS Trader — IA Double Touch Dashboard")
    log.info(f"  📊 Dados: leitura passiva dos arquivos do bot (sem conexão própria)")
    log.info(f"  🌐 http://localhost:{args.port}")
    log.info(f"{'='*60}")
    
    # Thread LIVE: conecta ao broker independentemente e busca candles
    t_live = threading.Thread(target=_live_broker_thread, daemon=True)
    t_live.start()
    
    # Thread de dados (lê cache do bot + dados live — detecta padrões + treino IA)
    t = threading.Thread(target=_update_thread, daemon=True)
    t.start()
    
    # Thread de detecção rápida de sinais (a cada ~55s — mantém live_signals frescos)
    t_sig = threading.Thread(target=_signal_scan_thread, daemon=True)
    t_sig.start()
    
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
