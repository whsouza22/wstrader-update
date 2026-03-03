# -*- coding: utf-8 -*-
"""
WS_AUTO_AI_BULLEX — Motor de Trading com Reversal AI PURA
═══════════════════════════════════════════════════════════
✅ ÚNICA estratégia: Reversal AI (GradientBoosting, 40 features)
✅ Auto-seleciona ativo com MAIOR acurácia
✅ Retreina a cada 5 minutos com janela deslizante e SALVA modelo
✅ Analisa no segundo :50 para entrar na virada do candle (:00)
✅ Expiração fixa de 1 minuto
✅ Confiança da IA decide a entrada
✅ Suporta: Bullex, CasaTrader, IQ Option
"""

import os
import sys
import time
import json
import logging
import random

# ── Fix Windows console Unicode (emojis) ──
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from datetime import date as _date_cls, datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# ===================== DETECÇÃO AUTOMÁTICA DA CORRETORA =====================
BROKER_TYPE = os.getenv("BROKER_TYPE", "bullex").strip().lower()

if BROKER_TYPE == "casatrader":
    from casatraderapi.stable_api import Casa_Trader as BrokerAPI
    import casatraderapi.constants as _broker_consts
    _BROKER_TAG = "WS_CASATRADER"
    _BROKER_LABEL = "CasaTrader"
elif BROKER_TYPE == "iq_option":
    from iqoptionapi.stable_api import IQ_Option as BrokerAPI
    import iqoptionapi.constants as _broker_consts
    _BROKER_TAG = "WS_IQ"
    _BROKER_LABEL = "IQ Option"
else:  # bullex (padrão)
    BROKER_TYPE = "bullex"
    from bullexapi.stable_api import Bullex as BrokerAPI
    import bullexapi.constants as _broker_consts
    _BROKER_TAG = "WS_BULLEX"
    _BROKER_LABEL = "Bullex"

# ═══ REVERSAL AI — ÚNICA ESTRATÉGIA ═══
from ws_reversal_ai import ReversalAI, FEATURE_NAMES, MIN_SAMPLES_ML

# ===================== CONFIG =====================
if BROKER_TYPE == "casatrader":
    EMAIL = os.getenv("CASATRADER_EMAIL", "")
    SENHA = os.getenv("CASATRADER_PASS", "")
    CONTA = os.getenv("CASATRADER_CONTA", "PRACTICE")
elif BROKER_TYPE == "iq_option":
    EMAIL = os.getenv("IQ_EMAIL", "")
    SENHA = os.getenv("IQ_PASS", "") or os.getenv("IQ_PASSWORD", "")
    CONTA = os.getenv("IQ_CONTA", "PRACTICE")
else:
    EMAIL = os.getenv("BULLUX_EMAIL", "") or os.getenv("BULLEX_EMAIL", "")
    SENHA = os.getenv("BULLUX_PASS", "") or os.getenv("BULLEX_PASS", "")
    CONTA = os.getenv("BULLUX_CONTA", os.getenv("BULLEX_CONTA", "PRACTICE"))

# Guarda de plano: só libera REAL se produto for PRO
_PRO_PRODUCT_ID = "prod_S4t8FQuUptWQ6R"
_DEMO_PRODUCT_ID = "prod_U3CRqZJMVigJAK"
_PREMIUM_PRODUCT_ID = "prod_U4ZxrEEApDg2Hb"   # PREMIUM — acesso total
_stripe_prod = os.environ.get("STRIPE_PRODUCT_ID", "")
if _stripe_prod in (_PRO_PRODUCT_ID, _PREMIUM_PRODUCT_ID):
    _plan = "PREMIUM" if _stripe_prod == _PREMIUM_PRODUCT_ID else "PRO"
    logging.getLogger(__name__).info(f"✅ Plano {_plan} — conta REAL liberada")
else:
    CONTA = "PRACTICE"
    _plan_label = "DEMO" if _stripe_prod == _DEMO_PRODUCT_ID else "DESCONHECIDO"
    logging.getLogger(__name__).info(f"🔒 Plano {_plan_label} (product: {_stripe_prod}) — forçando conta PRACTICE")

# ── Timeframes e velas ──
TF_M1 = 60
N_M1 = int(os.getenv("WS_N_M1", "900"))  # 900 candles = 15h de dados

# ── Payout / Assets ──
PAYOUT_MINIMO = int(os.getenv("WS_PAYOUT_MIN", "80"))
NUM_ATIVOS = int(os.getenv("WS_NUM_ATIVOS", "20"))
PAYOUT_REFRESH_SEC = int(os.getenv("WS_PAYOUT_REFRESH", "180"))

# ── Expiração FIXA 3 minutos (H&S Cabeça e Ombro) ──
EXP_FIXA = 3

# ── Stake / Banca ──
VALOR_MINIMO = float(os.getenv("WS_VALOR_MINIMO", "3"))
STAKE_FIXA = float(os.getenv("WS_STAKE", "5"))
PERCENT_BANCA = float(os.getenv("WS_PERCENT_BANCA", "1.0"))
META_LUCRO_PERCENT = float(os.getenv("WS_META_LUCRO", "1.5"))
STOP_LOSS_PERCENT = float(os.getenv("WS_STOP_LOSS", "3.0"))
USE_DYNAMIC_STAKE = (os.getenv("WS_DYNAMIC_STAKE", "1").strip() == "1")

# ── Reversal AI config ──
CONFIDENCE_MIN = float(os.getenv('WS_CONF_MIN', "40.0"))       # Confiança mínima da IA para entrar
RETRAIN_INTERVAL_MIN = int(os.getenv("WS_RETRAIN_MIN", "5"))     # Retreinar a cada 5 minutos
ANALYZE_AT_SECOND = int(os.getenv("WS_ANALYZE_SEC", "45"))      # Analisar no segundo :45 (antes da vela fechar, scan ~12s, entra na virada :00)
COOLDOWN_AFTER_TRADE = int(os.getenv("WS_COOLDOWN", "180"))      # Cooldown global após cada trade (3 min)
MIN_WR_ATIVO = float(os.getenv("WS_MIN_WR", "80.0"))            # WR mínimo para selecionar ativo

# ── Variáveis para Engine / IA ──
DECIDIR_ANTES_FECHAR_SEC = int(os.getenv("WS_DECIDIR_ANTES_FECHAR", "12"))
IA_ON = True  # IA SEMPRE ativa para H&S
AI_STATS_FILE = os.path.join(os.path.expanduser("~"), ".wstrader", "ws_ai_stats_hs.json")
AI_MIN_SAMPLES = 5
AI_CONF_MIN = 0.3
AI_MIN_PROB = 0.40
HORARIO_INICIO_MIN = 90    # 1h30 da manhã (1*60 + 30)
HORARIO_FIM_MIN    = 1080  # 18h00 (18*60)
MAX_DIST_OMBRO_ATR = 0.5  # Distância máx do OmbroD em ATR — se preço já se afastou demais, não entrar
MAX_DIST_NECKLINE_ATR = 0.25  # Distância máx ALÉM da neckline — se preço já ultrapassou muito, é tarde demais

# ── Ativos fixos — melhores ativos (OTC + REAL com volume) ──
# Ranking: EURJPY-OTC 56.7% | AUDCAD-OTC 56.1% | EURGBP 55.0%
# EURUSD 50.8% | USDCHF 50.7% | EURGBP-OTC 50.0% | EURJPY 50.0%
FIXED_ASSETS = {
    "iq": [
        "EURJPY-OTC", "AUDCAD-OTC", "EURGBP-OTC",
        "EURGBP", "EURUSD", "USDCHF", "EURJPY",
    ],
    "bullex": [
        "EURJPY-OTC", "AUDCAD-OTC", "EURGBP-OTC",
        "EURGBP", "EURUSD", "USDCHF", "EURJPY",
    ],
    "casatrader": [
        "EURJPY-OTC", "AUDCAD-OTC", "EURGBP-OTC",
        "EURGBP", "EURUSD", "USDCHF", "EURJPY",
    ],
}

# ── Diretórios ──
_broker_suffix = BROKER_TYPE.replace("iq_option", "iq")
_user_data_dir = os.path.join(os.path.expanduser("~"), ".wstrader")
os.makedirs(_user_data_dir, exist_ok=True)

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [%(levelname)s] [{_BROKER_TAG}] %(message)s"
)
log = logging.getLogger(_BROKER_TAG)


class C:
    G = "\033[92m"
    R = "\033[91m"
    Y = "\033[93m"
    B = "\033[94m"
    Z = "\033[0m"


def paint(s: str, color: str) -> str:
    return f"{color}{s}{C.Z}"


# ═══════════════════════════════════════════════════════════════
# UTILIDADES + H&S DETECTION
# ═══════════════════════════════════════════════════════════════
cooldown = {}  # {ativo: timestamp}


def _safe_load_json(filepath):
    """Carrega JSON de forma segura."""
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"meta": {"total": 0}, "arms": {}}


def _safe_save_json(filepath, data):
    """Salva JSON de forma segura."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
# CONTROLE DE TREINO — MEMÓRIA PERMANENTE (NUNCA RESETA)
# A IA ACUMULA conhecimento para sempre. Cada vez que liga,
# carrega do disco e treina APENAS ativos que ainda não têm dados.
# ═══════════════════════════════════════════════════════════════
_TRAIN_CONTROL_FILE = os.path.join(os.path.expanduser("~"), ".wstrader", "hs_bot_train_control.json")


def _need_retrain_bot():
    """Retorna sempre False — IA NUNCA reseta. Memória permanente."""
    return False


def _save_retrain_control():
    """Salva timestamp do último treino (apenas informativo)."""
    try:
        os.makedirs(os.path.dirname(_TRAIN_CONTROL_FILE), exist_ok=True)
        now = datetime.now()
        iso = now.isocalendar()
        with open(_TRAIN_CONTROL_FILE, "w") as f:
            json.dump({"iso_year": iso[0], "iso_week": iso[1], "date": now.isoformat(),
                       "mode": "permanent_memory"}, f)
        log.info(paint(f"[TREINO] Controle salvo: {now.strftime('%d/%m/%Y %H:%M')}", C.G))
    except Exception:
        pass


def _get_ia_level(n_total: int) -> tuple:
    """Retorna (nivel_numero, nivel_nome, emoji) baseado no total de amostras."""
    if n_total == 0:
        return (1, "Iniciante", "🌱")
    elif n_total <= 10:
        return (2, "Aprendendo", "📚")
    elif n_total <= 30:
        return (3, "Calibrando", "⚙️")
    elif n_total <= 80:
        return (4, "Experiente", "🧠")
    elif n_total <= 200:
        return (5, "Avançada", "🎯")
    else:
        return (6, "Expert", "🏆")


# ═══════════════════════════════════════════════════════════════
# DETECÇÃO H&S — DIRETO DA CORRETORA (SEM DASHBOARD)
# ═══════════════════════════════════════════════════════════════


def detect_pivots(H, L, window=5):
    """Detecta pivot highs e pivot lows diretamente dos arrays OHLC."""
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


def detect_all_hs(H, L, C_arr, O, pivot_highs, pivot_lows, atr):
    """Detecta TODOS os padrões H&S/iH&S no histórico de velas.
    Inclui validações: cabeça não pode ter sido rompida."""
    patterns = []
    n = len(H)
    tol = atr * 1.5
    min_depth = atr * 1.0
    min_spacing = 8
    max_span = 100
    trend_lookback = 30
    symmetry_min = 0.90
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
        shoulder_avg = (pL + pR) / 2
        head_depth = pH - shoulder_avg
        if head_depth < min_depth: continue
        if min(pL, pR) / max(pL, pR) < symmetry_min: continue
        if iL >= trend_lookback:
            if float(C_arr[iL]) <= float(C_arr[iL - trend_lookback]): continue
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
            "entry_idx": int(iR) + 1,
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
        shoulder_avg = (pL + pR) / 2
        head_depth = shoulder_avg - pH
        if head_depth < min_depth: continue
        if min(pL, pR) / max(pL, pR) < symmetry_min: continue
        if iL >= trend_lookback:
            if float(C_arr[iL]) >= float(C_arr[iL - trend_lookback]): continue
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
            "entry_idx": int(iR) + 1,
        })

    # ── MODO 2: H&S Tempo Real (PUT) ──
    for i in range(len(pivot_highs) - 1):
        iL, pL = pivot_highs[i]
        iH, pH = pivot_highs[i + 1]
        if ("H", iH) in seen_heads: continue
        if pH <= pL or iH - iL < min_spacing: continue
        head_depth = pH - pL
        if head_depth < min_depth: continue
        if iL >= trend_lookback:
            if float(C_arr[iL]) <= float(C_arr[iL - trend_lookback]): continue
        search_start = iH + min_spacing
        if search_start >= n: continue
        region = H[search_start:n]
        if len(region) < 2: continue
        local_max_rel = int(np.argmax(region))
        iR = search_start + local_max_rel
        pR = float(H[iR])
        if abs(pL - pR) > tol or pR >= pH: continue
        if min(pL, pR) / max(pL, pR) < symmetry_min: continue
        if iR - iL > max_span: continue
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
            "entry_idx": int(iR) + 1,
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
            if float(C_arr[iL]) >= float(C_arr[iL - trend_lookback]): continue
        search_start = iH + min_spacing
        if search_start >= n: continue
        region = L[search_start:n]
        if len(region) < 2: continue
        local_min_rel = int(np.argmin(region))
        iR = search_start + local_min_rel
        pR = float(L[iR])
        if abs(pL - pR) > tol or pR <= pH: continue
        if min(pL, pR) / max(pL, pR) < symmetry_min: continue
        if iR - iL > max_span: continue
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
            "entry_idx": int(iR) + 1,
        })

    return patterns


def backtest_pattern(pat, C, O, H, L, n):
    """Verifica se o padrão H&S resultaria em WIN ou LOSS.
    Regra: entra na abertura da vela entry_idx na direção pat['direction'].
    Verifica o close EXP_CANDLES velas depois.
    PUT: WIN se close < entry_price
    CALL: WIN se close > entry_price
    Retorna None se padrão é LIVE (sem resultado ainda)."""
    entry_idx = pat.get("entry_idx", pat["right_shoulder"]["idx"] + 1)
    if entry_idx >= n or entry_idx < 0:
        return None  # sem dados para verificar
    exit_idx = entry_idx + EXP_FIXA  # EXP_FIXA candles de expiração
    if exit_idx >= n:
        return None  # padrão muito recente, sem resultado ainda
    entry_price = float(O[entry_idx])
    exit_price = float(C[exit_idx - 1])
    head_price = pat["head"]["price"]
    if pat["direction"] == "PUT":
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


def _train_ia_from_history(bx, hs_stats: dict) -> dict:
    """Treina IA a partir do histórico de velas — MEMÓRIA PERMANENTE.
    Busca N_M1 candles para cada ativo top, detecta todos os padrões H&S,
    faz backtest de cada um, e ACUMULA WIN/LOSS nos stats existentes.
    NUNCA limpa dados anteriores — apenas ADICIONA novos padrões.
    Ativos que já possuem dados são re-treinados para ACUMULAR mais."""
    ativos = obter_top_ativos_otc(bx)
    if not ativos:
        log.warning(paint("⚠️ Nenhum ativo para treino — IA mantém memória anterior", C.Y))
        return hs_stats

    # Identificar quais ativos já têm dados suficientes
    existing_arms = hs_stats.get("arms", {})
    ativos_novos = []
    ativos_retreino = []
    for ativo in ativos:
        # Verificar se ALGUM arm desse ativo existe com dados
        has_data = False
        for arm_key in existing_arms:
            if arm_key.startswith(f"{ativo}_") and existing_arms[arm_key].get("total", 0) >= 5:
                has_data = True
                break
        if has_data:
            ativos_retreino.append(ativo)
        else:
            ativos_novos.append(ativo)

    if ativos_novos:
        log.info(paint(f"🆕 Ativos NOVOS para treinar: {', '.join(ativos_novos)}", C.B))
    if ativos_retreino:
        log.info(paint(f"📚 Ativos com memória (acumulando): {', '.join(ativos_retreino)}", C.G))

    # Treinar TODOS — novos do zero, existentes acumulam
    all_ativos = ativos_novos + ativos_retreino
    log.info(paint(f"🏋️ Treinando IA com {len(all_ativos)} ativos ({N_M1} velas cada)...", C.B))

    total_wins = 0
    total_losses = 0
    total_patterns = 0

    for ativo in all_ativos:
        try:
            df = get_candles_df(bx, ativo, TF_M1, N_M1)
            if df is None or len(df) < 100:
                continue

            H = df["high"].values
            L = df["low"].values
            C_arr = df["close"].values
            O = df["open"].values
            n = len(H)

            # ATR (14 períodos)
            atr_vals = [float(H[k] - L[k]) for k in range(max(0, n - 14), n)]
            atr = float(np.mean(atr_vals)) if atr_vals else 0.001
            if atr <= 0:
                continue

            # Detectar pivots e H&S
            ph, pl = detect_pivots(H, L, window=5)
            all_hs = detect_all_hs(H, L, C_arr, O, ph, pl, atr)

            if not all_hs:
                continue

            _w, _l = 0, 0
            for pat in all_hs:
                bt = backtest_pattern(pat, C_arr, O, H, L, n)
                if bt is not None and bt["result"] in ("win", "loss"):
                    # ACUMULAR na IA — nunca sobrescrever
                    pat_type = pat.get("type", "HS")
                    mode = pat.get("mode", "classic")
                    arm = f"{ativo}_{pat_type}_{mode}"
                    if "arms" not in hs_stats:
                        hs_stats["arms"] = {}
                    if arm not in hs_stats["arms"]:
                        hs_stats["arms"][arm] = {"wins": 0, "total": 0}
                    hs_stats["arms"][arm]["total"] += 1
                    if bt["result"] == "win":
                        hs_stats["arms"][arm]["wins"] += 1
                        _w += 1
                    else:
                        _l += 1

                    # Stats globais (meta.total)
                    meta = hs_stats.setdefault("meta", {"total": 0})
                    meta["total"] = meta.get("total", 0) + 1

                    total_patterns += 1

            if _w + _l > 0:
                log.info(f"  {ativo}: {len(all_hs)} padrões | {_w}W / {_l}L")
            total_wins += _w
            total_losses += _l

        except Exception as e:
            log.debug(f"Erro treinando {ativo}: {e}")
            continue

    _n_total = hs_stats.get("meta", {}).get("total", 0)
    _lvl_num, _lvl_nome, _lvl_emoji = _get_ia_level(_n_total)
    wr = (total_wins / max(total_wins + total_losses, 1)) * 100

    log.info(paint("=" * 50, C.G))
    log.info(paint(f"🏋️ TREINO CONCLUÍDO — MEMÓRIA PERMANENTE!", C.G))
    log.info(paint(f"  📊 Sessão: {total_patterns} padrões | {total_wins}W / {total_losses}L | WR: {wr:.1f}%", C.G))
    log.info(paint(f"  🧠 IA TOTAL: {_n_total} amostras acumuladas | Nível {_lvl_num}: {_lvl_emoji} {_lvl_nome}", C.G))
    log.info(paint(f"  💾 Memória NUNCA é apagada — quanto mais roda, mais aprende!", C.G))
    log.info(paint("=" * 50, C.G))

    print(f">>> IA: Treinada! {_n_total} amostras acumuladas | Nível {_lvl_num} ({_lvl_nome}) | WR: {wr:.1f}%", flush=True)

    # Salvar no disco — PERMANENTE
    _safe_save_json(AI_STATS_FILE, hs_stats)
    _save_retrain_control()

    return hs_stats


def escolher_melhor_setup_local(bx, cooldown_map: dict, hs_stats: dict):
    """Detecta H&S em TEMPO REAL — busca candles DIRETO da corretora.
    Sem dashboard, sem JSON, sem delay. Detecção local pura.
    Returns (best_trade, best_any)."""
    ativos = obter_top_ativos_otc(bx)
    if not ativos:
        log.warning(paint("⚠️ Nenhum ativo OTC disponível", C.Y))
        return None, None

    best_trade = None
    best_any = None
    _total_patterns = 0

    for ativo in ativos:
        # Cooldown individual
        if ativo in cooldown_map:
            elapsed = time.time() - cooldown_map[ativo]
            if elapsed < COOLDOWN_AFTER_TRADE:
                continue

        # ── Buscar candles DIRETO da corretora ──
        df = get_candles_df(bx, ativo, TF_M1, N_M1)
        if df is None or len(df) < 100:
            continue

        H = df["high"].values
        L = df["low"].values
        C_arr = df["close"].values
        O = df["open"].values
        n = len(H)

        # ATR (14 períodos) — IGUAL ao dashboard
        atr_vals = [float(H[k] - L[k]) for k in range(max(0, n - 14), n)]
        atr = float(np.mean(atr_vals)) if atr_vals else 0.001
        if atr <= 0:
            continue

        # ── Detectar pivots e padrões H&S ──
        pivot_highs, pivot_lows = detect_pivots(H, L, window=5)
        patterns = detect_all_hs(H, L, C_arr, O, pivot_highs, pivot_lows, atr)

        if not patterns:
            continue

        # ── Filtrar: só padrões LIVE (sem resultado ainda) — IGUAL ao dashboard ──
        live_patterns = []
        for pat in patterns:
            bt = backtest_pattern(pat, C_arr, O, H, L, n)
            if bt is None:
                # Padrão recente sem resultado = sinal LIVE
                entry_idx = pat.get("entry_idx", pat["right_shoulder"]["idx"] + 1)
                pat["entry_pending"] = entry_idx >= n
                pat["candles_ago"] = max(0, n - 1 - pat["right_shoulder"]["idx"])
                live_patterns.append(pat)

        if not live_patterns:
            continue

        _total_patterns += len(live_patterns)

        for pat in live_patterns:
            direction = pat["direction"]
            pat_type = pat["type"]
            mode = pat.get("mode", "classic")

            # IA predict — IGUAL ao dashboard
            ia_prob = ai_predict_hs(ativo, pat, hs_stats)
            ia_n = hs_stats.get("arms", {}).get(f"{ativo}_{pat_type}_{mode}", {}).get("total", 0)

            # Score: se IA tem dados usa, senão 0.5
            score = ia_prob if ia_n >= AI_MIN_SAMPLES else 0.5

            setup = {
                "dir": direction,
                "type": pat_type,
                "mode": mode,
                "confidence": round(score * 100, 1),
                "pattern": pat,
            }

            log.info(paint(
                f"  📊 H&S LOCAL: {ativo} | {pat_type} {direction} ({mode}) | "
                f"score={score:.2f} | entry_idx={pat['entry_idx']} | n={n}",
                C.B
            ))

            if best_any is None or score > best_any[0]:
                best_any = (score, ativo, setup, atr)

            # Filtro: IA não bloqueia se poucos dados
            if ia_n >= AI_MIN_SAMPLES and ia_prob < AI_MIN_PROB:
                continue

            if best_trade is None or score > best_trade[0]:
                best_trade = (score, ativo, setup, atr)

    if _total_patterns > 0:
        log.info(paint(f"  🔍 Scan local: {_total_patterns} padrão(ões) H&S recente(s) encontrado(s)", C.G))
    return best_trade, best_any


def wait_until_minus(tf, seconds_before):
    """Espera até `seconds_before` segundos antes do fechamento do candle."""
    while True:
        s = tf - (time.time() % tf)
        if s <= seconds_before:
            return
        time.sleep(min(s - seconds_before, 1.0))



def ai_predict_hs(ativo, pat, stats_ai):
    """IA prediction para setup H&S — IGUAL ao dashboard.
    Usa key com mode e fallback para stats globais."""
    arms = stats_ai.get("arms", {})
    # Key específica: ativo_type_mode
    key = f"{ativo}_{pat.get('type', 'HS')}_{pat.get('mode', 'classic')}"
    data = arms.get(key, None)
    if data and data.get("total", 0) >= 3:
        return data["wins"] / data["total"]
    # Fallback: stats globais do tipo (qualquer ativo com mesmo type)
    pat_type = pat.get("type", "HS")
    g_wins, g_total = 0, 0
    for k, v in arms.items():
        if f"_{pat_type}_" in k or k.endswith(f"_{pat_type}"):
            g_wins += v.get("wins", 0)
            g_total += v.get("total", 0)
    if g_total >= 5:
        return g_wins / g_total
    return 0.5  # sem dados


def ai_predict(ativo, setup, stats_ai):
    """IA prediction para setup H&S (compatibilidade)."""
    arm = f"{ativo}_{setup.get('type', 'HS')}_{setup.get('mode', 'classic')}"
    arm_data = stats_ai.get("arms", {}).get(arm, {"wins": 0, "total": 0})
    n = arm_data.get("total", 0)
    w = arm_data.get("wins", 0)
    prob = w / max(n, 1) if n > 0 else 0.5
    conf = min(n / 10.0, 1.0)
    return {"prob": prob, "n_arm": n, "conf": conf}


def ai_update(ativo, setup, result_value, stats_ai):
    """Atualiza IA stats após trade H&S — com mode na key."""
    if "arms" not in stats_ai:
        stats_ai["arms"] = {}
    arm = f"{ativo}_{setup.get('type', 'HS')}_{setup.get('mode', 'classic')}"
    if arm not in stats_ai["arms"]:
        stats_ai["arms"][arm] = {"wins": 0, "total": 0}
    stats_ai["arms"][arm]["total"] += 1
    if result_value > 0:
        stats_ai["arms"][arm]["wins"] += 1
    meta = stats_ai.setdefault("meta", {"total": 0})
    meta["total"] = meta.get("total", 0) + 1


# ═══════════════════════════════════════════════════════════════
# BROKER CONNECTION
# ═══════════════════════════════════════════════════════════════
_MAX_CONNECT_RETRIES = 10
_CONNECT_RETRY_BASE_DELAY = 10
_CONNECT_RETRY_MAX_DELAY = 120


def conectar_broker() -> BrokerAPI:
    """Conecta ao broker com retry automático e backoff exponencial."""
    if not EMAIL or not SENHA:
        raise RuntimeError(f"Defina credenciais para {_BROKER_LABEL} nas variáveis de ambiente.")

    delay = _CONNECT_RETRY_BASE_DELAY
    for attempt in range(1, _MAX_CONNECT_RETRIES + 1):
        try:
            log.info(f"Conectando à {_BROKER_LABEL}... (tentativa {attempt}/{_MAX_CONNECT_RETRIES})")
            bx = BrokerAPI(EMAIL, SENHA)
            check, reason = bx.connect()

            if check is False or check == False:
                reason_str = str(reason) if reason else ""
                reason_lower = reason_str.lower()
                if any(kw in reason_lower for kw in ["invalid", "credentials", "password", "unauthorized", "403", "incorrect", "wrong"]):
                    raise RuntimeError(f"SENHA_INCORRETA: Credenciais inválidas para {_BROKER_LABEL}.")
                elif "2FA" in reason_str:
                    raise RuntimeError(f"2FA_REQUIRED: {_BROKER_LABEL} requer verificação em duas etapas.")
                else:
                    raise ConnectionError(f"Falha na conexão: {reason_str}")

            for _ in range(12):
                if bx.check_connect():
                    break
                time.sleep(1.5)

            if not bx.check_connect():
                raise ConnectionError(f"Timeout na conexão com a {_BROKER_LABEL}.")

            bx.change_balance(CONTA)
            # Atualizar ACTIVES dinamicamente para reconhecer todos os pares OTC
            try:
                bx.update_ACTIVES_OPCODE()
                log.info("ACTIVES atualizados dinamicamente")
            except Exception:
                pass
            time.sleep(2)
            try:
                bal = bx.get_balance()
                if bal is not None:
                    log.info(f"Conectado | Saldo: {bal:.2f} | Conta: {CONTA}")
                else:
                    log.info(f"Conectado | Conta: {CONTA} (saldo será carregado em breve)")
            except Exception:
                log.info(f"Conectado | Conta: {CONTA}")
            return bx

        except Exception as e:
            if attempt >= _MAX_CONNECT_RETRIES:
                log.error(paint(f"❌ Falha após {_MAX_CONNECT_RETRIES} tentativas: {e}", C.R))
                raise RuntimeError(f"Falha na conexão com a {_BROKER_LABEL} após {_MAX_CONNECT_RETRIES} tentativas.")
            log.warning(paint(f"⚠️ Tentativa {attempt} falhou ({e}). Retry em {delay}s...", C.Y))
            time.sleep(delay)
            delay = min(delay * 2, _CONNECT_RETRY_MAX_DELAY)


def ensure_connected(bx: Optional[BrokerAPI]) -> BrokerAPI:
    """Garante conexão ativa. Se caiu, reconecta."""
    if bx is None:
        return conectar_broker()
    try:
        if bx.check_connect():
            return bx
    except Exception:
        pass
    log.warning(paint("Conexão caiu. Reconectando...", C.Y))
    try:
        bx.connect()
        for _ in range(12):
            if bx.check_connect():
                bx.change_balance(CONTA)
                log.info("Reconectado.")
                return bx
            time.sleep(1.5)
    except Exception:
        pass
    return conectar_broker()


def safe_call(bx: BrokerAPI, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        if any(kw in msg for kw in ["10054", "forçado o cancelamento", "goodbye", "10053"]):
            log.error(paint(f"Erro de conexão: {e}", C.R))
            ensure_connected(bx)
            return fn(*args, **kwargs)
        raise


# ═══════════════════════════════════════════════════════════════
# CANDLES
# ═══════════════════════════════════════════════════════════════
def get_candles_df(bx: BrokerAPI, ativo: str, timeframe: int, n: int,
                   end_ts: Optional[float] = None) -> Optional[pd.DataFrame]:
    try:
        if end_ts is None:
            end_ts = time.time()
        candles = safe_call(bx, bx.get_candles, ativo, timeframe, n, end_ts)
        if not candles or isinstance(candles, int):
            return None

        df = pd.DataFrame(candles)
        if "from" in df.columns and "time" not in df.columns:
            df.rename(columns={"from": "time"}, inplace=True)
        if "min" in df.columns:
            df.rename(columns={"min": "low"}, inplace=True)
        if "max" in df.columns:
            df.rename(columns={"max": "high"}, inplace=True)
        if "time" not in df.columns:
            return None

        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        needed = ["open", "high", "low", "close"]
        for col in needed:
            if col not in df.columns:
                return None
        df = df[needed].dropna().sort_index()
        if len(df) < 50:
            return None
        return df
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# ATIVOS OTC / PAYOUT
# ═══════════════════════════════════════════════════════════════
_cache_ativos: List[str] = []
_cache_ativos_ts: float = 0.0


def obter_top_ativos_otc(bx: BrokerAPI) -> List[str]:
    global _cache_ativos, _cache_ativos_ts
    now = time.time()
    if _cache_ativos and (now - _cache_ativos_ts) < PAYOUT_REFRESH_SEC:
        return _cache_ativos

    try:
        dados = safe_call(bx, bx.get_all_open_time)
        turbo = dados.get("turbo", {})
    except Exception:
        return []

    abertos = [a for a, info in turbo.items() if info.get("open", False)]
    abertos_otc = [a for a in abertos if "-OTC" in a.upper()]
    if not abertos_otc:
        abertos_otc = abertos

    # Filtrar apenas FOREX (pares de moedas válidos)
    _FOREX_CURRENCIES = {"EUR","USD","GBP","JPY","AUD","NZD","CAD","CHF","SEK","NOK",
                          "DKK","PLN","HUF","TRY","MXN","ZAR","SGD","HKD","CZK","BRL",
                          "INR","RUB","THB","ILS","XOF"}

    def _is_forex(name):
        base = name.replace("-OTC", "").replace("-otc", "")
        if any(c in base for c in ":_") or len(base) != 6:
            return False
        return base[:3].upper() in _FOREX_CURRENCIES and base[3:].upper() in _FOREX_CURRENCIES

    abertos_otc = [a for a in abertos_otc if _is_forex(a)]

    # Filtrar por payout
    try:
        all_profit = safe_call(bx, bx.get_all_profit)
    except Exception:
        all_profit = {}

    filtrados = []
    for a in abertos_otc:
        try:
            profit = all_profit.get(a, {}).get("turbo", 0)
            payout = int(profit * 100) if profit else 0
        except Exception:
            payout = 0
        if payout >= PAYOUT_MINIMO:
            filtrados.append((a, payout))

    filtrados.sort(key=lambda x: x[1], reverse=True)
    top = [a for a, _ in filtrados[:NUM_ATIVOS]]
    _cache_ativos = top
    _cache_ativos_ts = now
    log.info(f"TOP ativos OTC: {top}")
    return top


# ═══════════════════════════════════════════════════════════════
# GESTÃO DE BANCA
# ═══════════════════════════════════════════════════════════════
def calcular_stake(bx: BrokerAPI) -> float:
    if not USE_DYNAMIC_STAKE:
        return float(max(VALOR_MINIMO, STAKE_FIXA))
    try:
        saldo = float(bx.get_balance())
        stake = (saldo * PERCENT_BANCA) / 100.0
        return float(max(VALOR_MINIMO, stake))
    except Exception:
        return float(max(VALOR_MINIMO, STAKE_FIXA))


def verificar_meta(saldo_inicial: float, saldo_atual: float) -> Tuple[bool, float]:
    lucro = saldo_atual - saldo_inicial
    lucro_pct = (lucro / saldo_inicial) * 100.0
    if lucro_pct >= META_LUCRO_PERCENT:
        return True, lucro_pct
    if lucro_pct <= -STOP_LOSS_PERCENT:
        return True, lucro_pct
    return False, lucro_pct


# ═══════════════════════════════════════════════════════════════
# LIVE TRADE LOG (para dashboard)
# ═══════════════════════════════════════════════════════════════
LIVE_LOG_FILE = os.path.join(_user_data_dir, f"ws_live_trades_{_broker_suffix}.json")
_LIVE_LOG_MAX = 100


def _log_live_trade(ativo: str, direcao: str, resultado: Optional[float],
                    entry_price: Optional[float], stake: float,
                    confidence: float = 0.0, status: str = "entry"):
    """Grava trade no log para consumo pelo dashboard."""
    try:
        trades = []
        if os.path.exists(LIVE_LOG_FILE):
            with open(LIVE_LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                trades = data.get("trades", [])
        record = {
            "ts": time.time(),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ativo": ativo,
            "dir": direcao,
            "status": status,
            "resultado": resultado,
            "entry_price": entry_price,
            "stake": stake,
            "exp_min": EXP_FIXA,
            "brain_score": confidence,
            "dot_prob": 0.0,
            "broker": _broker_suffix,
        }
        trades.append(record)
        if len(trades) > _LIVE_LOG_MAX:
            trades = trades[-_LIVE_LOG_MAX:]
        with open(LIVE_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump({"trades": trades, "updated": time.time()}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
# ORDEM + RESULTADO
# ═══════════════════════════════════════════════════════════════
def enviar_ordem(bx: BrokerAPI, ativo: str, direcao: str, stake: float) -> Optional[Tuple[str, int]]:
    """Envia ordem (TURBO → DIGITAL fallback). Expiração fixa 1 minuto."""
    d = "call" if direcao == "CALL" else "put"
    valor = float(max(VALOR_MINIMO, stake))

    # TURBO
    try:
        ok, op_id = safe_call(bx, bx.buy, valor, ativo, d, EXP_FIXA)
        if ok and op_id:
            return ("turbo", int(op_id))
        log.warning(paint(f"[ORDEM] TURBO falhou ok={ok} id={op_id}", C.Y))
    except Exception as e:
        log.warning(paint(f"[ORDEM] TURBO exc: {e}", C.Y))

    # DIGITAL fallback
    try:
        ok, op_id = safe_call(bx, bx.buy_digital_spot, ativo, valor, d, EXP_FIXA)
        if ok and op_id:
            return ("digital", int(op_id))
        log.warning(paint(f"[ORDEM] DIGITAL falhou ok={ok} id={op_id}", C.Y))
    except Exception as e:
        log.warning(paint(f"[ORDEM] DIGITAL exc: {e}", C.Y))

    return None


def wait_result(bx: BrokerAPI, op_type: str, op_id: int) -> float:
    """Aguarda resultado do trade."""
    while True:
        try:
            if op_type == "turbo":
                win, res = safe_call(bx, bx.check_win_v4, op_id)
                return float(res)
            else:
                res = safe_call(bx, bx.get_digital_spot_profit_after_sale, op_id)
                if isinstance(res, (int, float)):
                    return float(res)
        except Exception:
            ensure_connected(bx)
        time.sleep(0.25)


# ── Aliases para compatibilidade com ws_auto_ai_engine ──
verificar_meta_atingida = verificar_meta
calcular_stake_dinamico = calcular_stake


# ═══════════════════════════════════════════════════════════════
# TIMING — esperar segundo :45 (antes da vela fechar) + entrar :00
# ═══════════════════════════════════════════════════════════════
def seconds_to_next(tf: int) -> float:
    now = time.time()
    return tf - (now % tf)


def wait_until_second(target_second: int = 45):
    """Espera até o segundo :45 do minuto atual (antes da vela fechar)."""
    while True:
        now = time.time()
        current_second = int(now % 60)
        if current_second == target_second:
            return
        if current_second > target_second:
            # Já passou, espera próximo minuto
            wait = 60 - current_second + target_second
        else:
            wait = target_second - current_second
        # Sleep grosso até 1s antes, depois fino
        if wait > 1.5:
            time.sleep(wait - 1.0)
        else:
            time.sleep(0.05)


def wait_candle_open():
    """Espera até a virada da vela (:00) para executar ordem.
    Usa spin-lock fino nos últimos 50ms para precisão."""
    now = time.time()
    sec_in_candle = now % 60
    s = 60 - sec_in_candle
    # Se já estamos nos primeiros 5s do candle, entra direto
    if sec_in_candle < 5:
        return
    log.info(paint(f"  ⏱️ Aguardando virada :00 ({s:.0f}s)...", C.B))
    # Sleep grosso até 50ms antes
    if s > 0.05:
        time.sleep(s - 0.05)
    # Spin-lock fino
    target = now + s
    while time.time() < target:
        pass


# ═══════════════════════════════════════════════════════════════
# MAIN — LOOP PRINCIPAL (SOMENTE CABEÇA E OMBROS)
# ═══════════════════════════════════════════════════════════════
def main():
    bx: Optional[BrokerAPI] = None
    bx = ensure_connected(bx)

    # Inicializar ReversalAI (para stats e compatibilidade)
    reversal_ai = ReversalAI("unified")

    # ── Carregar / Treinar IA — MEMÓRIA PERMANENTE ──
    # A IA NUNCA perde memória. Carrega do disco e ACUMULA.
    log.info(paint("🧠 Carregando memória da IA H&S...", C.B))

    # SEMPRE carregar stats existentes do disco (memória permanente)
    hs_stats = _safe_load_json(AI_STATS_FILE)
    _n_total = hs_stats.get("meta", {}).get("total", 0)

    if _n_total > 0:
        _lvl_num, _lvl_nome, _lvl_emoji = _get_ia_level(_n_total)
        log.info(paint(f"💾 IA carregada do disco! {_n_total} amostras acumuladas | Nível {_lvl_num}: {_lvl_emoji} {_lvl_nome}", C.G))
        print(f">>> IA: Memória carregada! {_n_total} amostras | Nível {_lvl_num} ({_lvl_nome})", flush=True)
    else:
        log.info(paint("🌱 Primeira execução — treinando IA do zero...", C.Y))

    # SEMPRE treinar para ACUMULAR mais dados (novos ativos + padrões recentes)
    log.info(paint("🏋️ Treinando IA (acumulando novos padrões na memória)...", C.B))
    hs_stats = _train_ia_from_history(bx, hs_stats)

    log.info("=" * 60)
    log.info(paint(f"🚀 WS TRADER — Cabeça e Ombros (H&S) ({_BROKER_LABEL})", C.G))
    log.info("=" * 60)
    log.info(f"✅ Estratégia: SOMENTE Cabeça e Ombros (H&S)")
    log.info(f"✅ Corretora: {_BROKER_LABEL} ({BROKER_TYPE})")
    log.info(f"✅ Expiração: {EXP_FIXA} minuto(s)")
    log.info(f"✅ Horário: 1:30 às 18:00")
    log.info(f"✅ Sinais: Detecção LOCAL (direto da corretora, sem delay)")
    log.info(f"✅ Memória: PERMANENTE — IA nunca perde conhecimento")
    log.info(f"✅ IA: ATIVA — acumula padrões H&S a cada execução")
    log.info("=" * 60)

    # Saldo inicial
    try:
        bal = bx.get_balance()
        saldo_inicial = float(bal) if bal is not None else 0.0
        if saldo_inicial == 0:
            time.sleep(3)
            bal = bx.get_balance()
            saldo_inicial = float(bal) if bal is not None else 1000.0
        log.info(paint(f"💰 SALDO INICIAL: {saldo_inicial:.2f} | META: {META_LUCRO_PERCENT:.1f}% | STOP: {STOP_LOSS_PERCENT:.1f}%", C.G))
        if USE_DYNAMIC_STAKE:
            log.info(paint(f"� GESTÃO: {PERCENT_BANCA:.1f}% da banca por operação", C.B))
        else:
            log.info(paint(f"� GESTÃO: Stake fixo de {STAKE_FIXA:.2f}", C.B))
    except Exception as e:
        log.warning(f"⚠️ Saldo não obtido: {e}")
        saldo_inicial = 1000.0

    total_trades = 0
    total_wins = 0
    _current_day = _date_cls.today()
    _last_trade_time = 0.0

    print(f"\n>>> IA: Iniciado | Exp: {EXP_FIXA}min | Sinais: Detecção LOCAL", flush=True)

    # Exportar stats iniciais para o UI
    reversal_ai.save_stats_to_disk()

    # ═══ LOOP PRINCIPAL — SINAIS DO DASHBOARD H&S ═══
    while True:
        try:
            bx = ensure_connected(bx)

            # ── Reset diário ──
            _today = _date_cls.today()
            if _today != _current_day:
                log.info(paint(f"\n🌅 NOVO DIA! {_current_day} → {_today}", C.G))
                _current_day = _today
                total_trades = 0
                total_wins = 0
                cooldown.clear()
                try:
                    bal = bx.get_balance()
                    saldo_inicial = float(bal) if bal is not None else saldo_inicial
                except Exception:
                    pass
                log.info(paint(f"💰 Novo saldo inicial: {saldo_inicial:.2f}", C.G))

            # ── Verificar horario de operacao (1:30 - 18:00) ──
            _now = datetime.now()
            _minutos_atual = _now.hour * 60 + _now.minute
            if _minutos_atual < HORARIO_INICIO_MIN or _minutos_atual >= HORARIO_FIM_MIN:
                log.info(paint("=" * 60, C.Y))
                log.info(paint(
                    f"FORA DO HORARIO DE OPERACAO ({_now.hour}:{_now.minute:02d})",
                    C.Y
                ))
                log.info(paint(
                    "Horario disponivel: 1:30 ate 18:00",
                    C.Y
                ))
                log.info(paint(
                    "Fora desse horario o mercado OTC nao atende os requisitos",
                    C.Y
                ))
                log.info(paint(
                    "da IA para operacoes seguras. Aguarde o proximo horario.",
                    C.Y
                ))
                log.info(paint("=" * 60, C.Y))
                print(f">>> IA: FORA DO HORARIO -- Bot disponivel das 1:30 as 18:00. Fora desse horario o mercado nao atende os requisitos da IA. Aguardando...", flush=True)
                time.sleep(60)
                continue

            # ── Verificar Meta / Stop Loss ──
            try:
                saldo_atual = float(bx.get_balance())
                atingiu, lucro_pct = verificar_meta(saldo_inicial, saldo_atual)
                if atingiu:
                    if lucro_pct >= 0:
                        log.info(paint(f"🏆 META ATINGIDA! Lucro: {lucro_pct:.2f}%  — IA encerrada.", C.G))
                        print(f">>> 🏆 META ATINGIDA! Lucro: {lucro_pct:.2f}%", flush=True)
                    else:
                        log.info(paint(f"🛑 STOP LOSS! Perda: {lucro_pct:.2f}%  — IA encerrada.", C.R))
                        print(f">>> 🛑 STOP LOSS! Perda: {lucro_pct:.2f}%", flush=True)
                    return
            except Exception:
                pass

            # ── Cooldown global (3 min após cada trade) ──
            if _last_trade_time > 0:
                elapsed = time.time() - _last_trade_time
                remaining = COOLDOWN_AFTER_TRADE - elapsed
                if remaining > 0:
                    mins = int(remaining) // 60
                    secs = int(remaining) % 60
                    log.info(paint(f"  ⏳ Cooldown: {mins}m{secs:02d}s restantes", C.B))
                    s = seconds_to_next(TF_M1)
                    time.sleep(min(s + 1, 30))
                    continue

            # ═══ ESPERAR SEGUNDO :45 PARA SCANEAR H&S ═══
            log.info(paint(f"\n⏰ Esperando :{ANALYZE_AT_SECOND:02d} para scanear H&S direto da corretora...", C.B))
            wait_until_second(ANALYZE_AT_SECOND)

            # ═══ DETECÇÃO LOCAL H&S — DIRETO DA CORRETORA ═══
            log.info(paint("\n🔍 Scaneando H&S em tempo real (direto da corretora)...", C.B))
            best_trade, best_any = escolher_melhor_setup_local(bx, cooldown, hs_stats)

            if not best_trade:
                if best_any:
                    _, a, setup, _ = best_any
                    log.info(paint(
                        f"  ⏸️ Sinal em formação: {a} | aguardando confirmação",
                        C.Y
                    ))
                else:
                    log.info(paint("  ⏸️ Nenhum H&S recente. Próximo candle.", C.Y))
                s = seconds_to_next(TF_M1)
                time.sleep(min(s + 1, 30))
                continue

            # ═══ H&S ENCONTRADO → ENTRAR ═══
            sc, ativo, setup, atr_val = best_trade
            direcao = setup["dir"]
            pat_type = setup["type"]

            # ═══ IA: PROBABILIDADE LOCAL — IGUAL AO DASHBOARD ═══
            pat_data = setup.get("pattern", setup)
            ia_prob = ai_predict_hs(ativo, pat_data, hs_stats)
            _arm_key = f"{ativo}_{pat_type}_{setup.get('mode', 'classic')}"
            ia_samples = hs_stats.get("arms", {}).get(_arm_key, {}).get("total", 0)

            # Se IA sem dados: 0.5 (neutro)
            if ia_samples < AI_MIN_SAMPLES:
                ia_prob = 0.5

            log.info(paint(
                f"  🤖 IA H&S: {ativo} | prob={ia_prob:.2f} | amostras={ia_samples}",
                C.B
            ))

            # Bloqueio: só se IA tem dados suficientes E prob é ruim
            if ia_samples >= AI_MIN_SAMPLES and ia_prob < AI_MIN_PROB:
                log.info(paint(
                    f"  🚫 IA BLOQUEOU entrada: {ativo} {direcao} | prob={ia_prob:.2f}",
                    C.Y
                ))
                print(f">>> IA: Entrada BLOQUEADA {ativo} {direcao} prob={ia_prob:.2f}", flush=True)
                s = seconds_to_next(TF_M1)
                time.sleep(min(s + 1, 30))
                continue

            # ═══ GUARD H&S: VALIDAR PREÇO vs CABEÇA/NECKLINE ═══
            _head_price = setup["pattern"]["head"]["price"]
            _neckline = setup["pattern"].get("neckline", 0)
            _rs_price = setup["pattern"]["right_shoulder"]["price"]
            _guard_ok = True
            try:
                _price_df = get_candles_df(bx, ativo, TF_M1, 3)
                if _price_df is not None and len(_price_df) >= 1:
                    _cur = float(_price_df["close"].iloc[-1])
                    if direcao == "PUT":
                        # H&S: preço NÃO pode estar acima da cabeça
                        if _cur >= _head_price:
                            log.info(paint(
                                f"  🚫 GUARD: Preço ({_cur:.6f}) >= Cabeça ({_head_price:.6f}) — SKIP",
                                C.Y
                            ))
                            _guard_ok = False
                        # PUT: preço deve estar ABAIXO ou próximo do neckline
                        elif _neckline > 0 and _cur > _neckline + (_head_price - _neckline) * 0.5:
                            log.info(paint(
                                f"  🚫 GUARD: Preço ({_cur:.6f}) muito acima do Neckline ({_neckline:.6f}) — SKIP",
                                C.Y
                            ))
                            _guard_ok = False
                    else:  # CALL (iH&S)
                        if _cur <= _head_price:
                            log.info(paint(
                                f"  🚫 GUARD: Preço ({_cur:.6f}) <= Cabeça ({_head_price:.6f}) — SKIP",
                                C.Y
                            ))
                            _guard_ok = False
                        elif _neckline > 0 and _cur < _neckline - (_neckline - _head_price) * 0.5:
                            log.info(paint(
                                f"  🚫 GUARD: Preço ({_cur:.6f}) muito abaixo do Neckline ({_neckline:.6f}) — SKIP",
                                C.Y
                            ))
                            _guard_ok = False
                    if _guard_ok:
                        log.info(paint(
                            f"  ✅ GUARD OK: {ativo} {direcao} | Preço={_cur:.6f} | "
                            f"Neck={_neckline:.6f} | Cabeça={_head_price:.6f}",
                            C.G
                        ))
            except Exception as _ge:
                log.warning(paint(f"  ⚠️ GUARD: Não validou preço ({_ge})", C.Y))

            if not _guard_ok:
                print(f">>> IA: GUARD bloqueou {ativo} {direcao} — preço fora da zona H&S", flush=True)
                s = seconds_to_next(TF_M1)
                time.sleep(min(s + 1, 30))
                continue

            # Calcular stake
            stake = calcular_stake(bx)

            # ═══ ENTRAR IMEDIATAMENTE (na seta do dashboard) ═══
            _log_live_trade(ativo, direcao, None, None, stake,
                            confidence=ia_prob * 100, status="entry")

            op = enviar_ordem(bx, ativo, direcao, stake)
            if not op:
                log.warning(paint(f"  ❌ Falha na ordem: {ativo}", C.R))
                s = seconds_to_next(TF_M1)
                time.sleep(min(s + 1, 30))
                continue

            op_type, op_id = op
            log.info(paint(
                f"  ✅ ENTRADA: {ativo} {direcao} | Stake={stake:.2f} | Tipo={op_type} | "
                f"IA={ia_prob:.0%} | Amostras={ia_samples}",
                C.G if direcao == "CALL" else C.R
            ))
            print(f">>> IA: Entrada {ativo} {direcao} stake={stake:.2f} prob={ia_prob:.2f}", flush=True)

            # ═══ AGUARDAR RESULTADO ═══
            res = wait_result(bx, op_type, op_id)
            total_trades += 1

            if res > 0:
                total_wins += 1
                _live_status = "win"
                log.info(paint(f"  ✅ WIN +{res:.2f}$", C.G))
                print(f">>> RESULTADO: WIN {ativo} {direcao} +{res:.2f}", flush=True)
            elif res < 0:
                _live_status = "loss"
                log.info(paint(f"  ❌ LOSS {res:.2f}$", C.R))
                print(f">>> RESULTADO: LOSS {ativo} {direcao} {res:.2f}", flush=True)
            else:
                _live_status = "tie"
                log.info(paint(f"  ⚪ EMPATE", C.B))
                print(f">>> RESULTADO: EMPATE {ativo}", flush=True)

            _log_live_trade(ativo, direcao, res, None, stake,
                            confidence=ia_prob * 100, status=_live_status)

            # ── IA: aprender com o resultado ──
            ai_update(ativo, setup, res, hs_stats)
            _n_amostras = sum(v.get("total", 0) for v in hs_stats.get("arms", {}).values())
            _lvl_num, _lvl_nome, _lvl_emoji = _get_ia_level(_n_amostras)
            log.info(paint(
                f"  🤖 IA atualizada: {ativo} | resultado={'WIN' if res > 0 else 'LOSS' if res < 0 else 'EMPATE'} | "
                f"prob_antes={ia_prob:.2f} | amostras={_n_amostras} | Nível {_lvl_num}: {_lvl_emoji} {_lvl_nome}",
                C.B
            ))
            _safe_save_json(AI_STATS_FILE, hs_stats)
            # Salvar controle de retrain SOMENTE quando IA tem amostras reais
            if _n_amostras > 0:
                _save_retrain_control()

            # ── Estatísticas ──
            wr = (total_wins / max(1, total_trades)) * 100
            try:
                saldo_now = float(bx.get_balance())
                lucro = saldo_now - saldo_inicial
                meta_val = saldo_inicial * META_LUCRO_PERCENT / 100.0
                log.info(paint(
                    f"  � Sessão: {total_trades} trades | {total_wins}W | WR={wr:.1f}% | "
                    f"Lucro: {'+' if lucro >= 0 else ''}{lucro:.2f}",
                    C.G if lucro >= 0 else C.R
                ))
                print(f">>> IA: {total_trades} trades | WR={wr:.1f}% | {'+' if lucro >= 0 else ''}{lucro:.2f}", flush=True)
            except Exception:
                log.info(f"  � Sessão: {total_trades} trades | {total_wins}W | WR={wr:.1f}%")

            # Exportar stats para o UI
            reversal_ai.save_stats_to_disk()

            # Cooldown
            _last_trade_time = time.time()
            cooldown[ativo] = time.time()

        except KeyboardInterrupt:
            log.info(paint("\n⏹️ IA encerrada pelo usuário.", C.Y))
            print(">>> IA encerrada.", flush=True)
            break
        except Exception as e:
            log.error(paint(f"❌ Erro no loop: {e}", C.R))
            import traceback
            log.error(traceback.format_exc())
            time.sleep(5)
            continue


if __name__ == "__main__":
    main()
