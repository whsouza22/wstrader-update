# -*- coding: utf-8 -*-
"""
WS_AUTO_AI_BULLEX — Pernada B (M1) para BULLEX
✅ Candles FECHADOS (evita sinal fora da hora)
✅ Anti-lateral + Anti-esticado
✅ Filtro de SUPORTE/RESISTÊNCIA FORTE (usa >=200 velas e considera várias regiões)
✅ IA ENSEMBLE: Bayesiano + LightGBM (Gradient Boosting) para decisões mais inteligentes
✅ Execução real (TURBO -> DIGITAL fallback)

Requisitos:
pip install bullexapi pandas numpy lightgbm
"""

import os
import time
import math
import json
import logging
import pickle
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from bullexapi.stable_api import Bullex

# DOM Forex Strategy (Perfect Zones)
from dom_forex_strategy import dom_forex_signal

# LightGBM para Gradient Boosting
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LGBM_AVAILABLE = False

# ===================== CONFIG =====================
EMAIL = os.getenv("BULLEX_EMAIL", "") or "wstrader@wstrader.onmicrosoft.com"
SENHA = os.getenv("BULLEX_PASS", "") or "P152030@w"
CONTA = os.getenv("BULLEX_CONTA", "PRACTICE")

TF_M1 = 60
DECIDIR_ANTES_FECHAR_SEC = int(os.getenv("WS_DECIDIR_ANTES_FECHAR", "10"))
N_M1 = int(os.getenv("WS_N_M1", "340"))

PAYOUT_MINIMO = int(os.getenv("WS_PAYOUT_MIN", "80"))
NUM_ATIVOS = int(os.getenv("WS_NUM_ATIVOS", "12"))
PAYOUT_REFRESH_SEC = int(os.getenv("WS_PAYOUT_REFRESH", "180"))

EXP_FIXA = int(os.getenv("WS_EXP_MIN", "5"))
VALOR_MINIMO = float(os.getenv("WS_VALOR_MINIMO", "3"))
STAKE_FIXA = float(os.getenv("WS_STAKE", "5"))

# ===================== GESTÃO DE BANCA =====================
PERCENT_BANCA = float(os.getenv("WS_PERCENT_BANCA", "1.0"))  # 1% da banca por operação
META_LUCRO_PERCENT = float(os.getenv("WS_META_LUCRO", "1.5"))  # para com 1.5% de lucro
STOP_LOSS_PERCENT = float(os.getenv("WS_STOP_LOSS", "3.0"))  # para com 3% de perda (opcional)
USE_DYNAMIC_STAKE = (os.getenv("WS_DYNAMIC_STAKE", "1").strip() == "1")  # usar % da banca

COOLDOWN_ATIVO = int(os.getenv("WS_COOLDOWN_ATIVO", "60"))  # 60 seg de cooldown base
COOLDOWN_LOSS_ATIVO = int(os.getenv("WS_COOLDOWN_LOSS", "300"))  # 5 min após LOSS no ativo
MAX_CONSECUTIVE_LOSS = int(os.getenv("WS_MAX_CONSEC_LOSS", "1"))  # Pausa após 1 loss
RETRAIN_ON_LOSS = (os.getenv("WS_RETRAIN_ON_LOSS", "1").strip() == "1")  # Retreinar e CONTINUAR após loss
PAUSE_AFTER_LOSS_SECONDS = int(os.getenv("WS_PAUSE_AFTER_LOSS", "120"))  # Pausa de 2 min após loss antes de continuar
RETRAIN_PENALTY = float(os.getenv("WS_RETRAIN_PENALTY", "0.25"))  # Penalidade no retreino (25%)

# ===================== IA (ONLINE) - APRENDIZADO ADAPTATIVO =====================
IA_ON = (os.getenv("WS_AI_ON", "1").strip() == "1")  # LIGADO: aprende bloqueando losses
AI_STATS_FILE = os.getenv("WS_AI_FILE", "ws_ai_stats_bullex.json")
AI_MIN_SAMPLES = int(os.getenv("WS_AI_MIN_SAMPLES", "15"))   # 15 trades para começar a bloquear
AI_MIN_PROB = float(os.getenv("WS_AI_MIN_PROB", "0.45"))     # probabilidade mínima (bayesiana)
AI_MIN_WINRATE = float(os.getenv("WS_AI_MIN_WINRATE", "0.42"))  # bloqueia se winrate < 42%
AI_CONF_MIN = float(os.getenv("WS_AI_CONF_MIN", "0.50"))     # confiança mínima na decisão

# ===================== LIGHTGBM ENSEMBLE =====================
LGBM_ON = (os.getenv("WS_LGBM_ON", "1").strip() == "1") and LGBM_AVAILABLE  # LightGBM ativo
LGBM_MODEL_FILE = os.getenv("WS_LGBM_FILE", "ws_lgbm_model_bullex.pkl")
LGBM_DATA_FILE = os.getenv("WS_LGBM_DATA", "ws_lgbm_data_bullex.json")
LGBM_MIN_SAMPLES = int(os.getenv("WS_LGBM_MIN_SAMPLES", "30"))  # Mínimo de amostras para treinar
LGBM_RETRAIN_EVERY = int(os.getenv("WS_LGBM_RETRAIN", "10"))   # Retreina a cada N trades
LGBM_MIN_PROB = float(os.getenv("WS_LGBM_MIN_PROB", "0.55"))   # Probabilidade mínima do LGBM
ENSEMBLE_MODE = os.getenv("WS_ENSEMBLE_MODE", "both")  # "both" = ambos devem concordar, "any" = qualquer um

# ===================== MODO DA IA =====================
# "learning" = IA tem controle total, filtros de score relaxados (mais trades, aprende mais rápido)
# "strict"   = IA + filtros rigorosos (menos trades, mais conservador)
IA_MODE = os.getenv("WS_IA_MODE", "learning").strip().lower()  # PADRÃO: learning

# ===================== PERNADA B (RELAXADO PARA PERMITIR MAIS ENTRADAS) =====================
IMPULSO_MIN_ATR = float(os.getenv("WS_IMPULSO_MIN_ATR", "0.50"))  # MUITO RELAXADO: 0.50 ATR
IMPULSO_JANELA_MIN = int(os.getenv("WS_IMP_JMIN", "3"))  # mínimo 3 velas
IMPULSO_JANELA_MAX = int(os.getenv("WS_IMP_JMAX", "15"))  # máximo 15 velas

PULLBACK_MIN = int(os.getenv("WS_PB_MIN", "1"))
PULLBACK_MAX = int(os.getenv("WS_PB_MAX", "6"))  # aumentado para 6

RETR_MIN = float(os.getenv("WS_RETR_MIN", "0.10"))  # MUITO RELAXADO: 10%
RETR_MAX = float(os.getenv("WS_RETR_MAX", "0.85"))  # até 85%

BREAK_MARGIN_ATR = float(os.getenv("WS_BREAK_MARGIN_ATR", "0.01"))  # margem mínima
MAX_BREAK_DISTANCE_ATR = float(os.getenv("WS_MAX_BREAK_DIST_ATR", "0.40"))  # distância maior

# ===================== ANTI-LATERAL (MUITO RELAXADO) =====================
MIN_EFF_A = float(os.getenv("WS_MIN_EFF_A", "0.40"))  # RELAXADO: apenas 40% eficiência

CHOP_LOOKBACK = int(os.getenv("WS_CHOP_LB", "28"))
MAX_COLOR_FLIPS_FRAC = float(os.getenv("WS_MAX_FLIPS", "0.80"))  # permite mais choppiness
MIN_NET_GROSS_EFF = float(os.getenv("WS_MIN_NETGROSS", "0.10"))  # muito relaxado

COMP_LOOKBACK = int(os.getenv("WS_COMP_LB", "18"))
MIN_RANGE_ATR = float(os.getenv("WS_MIN_RANGE_ATR", "0.50"))  # MUITO RELAXADO: 0.50 ATR

LATE_LOOKBACK = int(os.getenv("WS_LATE_LB", "18"))
MAX_LATE_EXT_ATR = float(os.getenv("WS_MAX_LATE_EXT_ATR", "12.0"))  # permite extensão maior

# ===================== QUALIDADE DO GATILHO (RELAXADO) =====================
MIN_BODY_FRAC_BREAK = float(os.getenv("WS_MIN_BODY_FRAC", "0.10"))  # apenas 10% de corpo
MAX_WICK_AGAINST = float(os.getenv("WS_MAX_WICK_AGAINST", "0.75"))  # permite mais pavio

# ===================== SCORE (DEPENDE DO IA_MODE) =====================
# No modo "learning": filtros de score MUITO relaxados, IA decide
# No modo "strict": filtros rigorosos + IA
if IA_MODE == "learning":
    GATE_MIN_SCORE = float(os.getenv("WS_GATE_MIN", "0.35"))   # Score mínimo RELAXADO
    GATE_SOFT_SCORE = float(os.getenv("WS_GATE_SOFT", "0.25")) # Soft skip RELAXADO
    GATE_CONTEXT_BAD_BLOCK = False  # NÃO bloqueia por contexto ruim - IA decide
else:  # strict
    GATE_MIN_SCORE = float(os.getenv("WS_GATE_MIN", "0.75"))   # Score mínimo rigoroso
    GATE_SOFT_SCORE = float(os.getenv("WS_GATE_SOFT", "0.65")) # Soft skip rigoroso
    GATE_CONTEXT_BAD_BLOCK = True  # Bloquear se contexto for ruim

# ===================== ANTI-SPIKE =====================
SPIKE_RANGE_ATR = float(os.getenv("WS_SPIKE_RANGE_ATR", "1.35"))
SPIKE_WICK_FRAC = float(os.getenv("WS_SPIKE_WICK_FRAC", "0.62"))
SPIKE_COOLDOWN_MIN = int(os.getenv("WS_SPIKE_COOLDOWN_MIN", "6"))

# ===================== FILTRO S/R FORTE (AJUSTADO) =====================
SR_LOOKBACK = int(os.getenv("WS_SR_LOOKBACK", "220"))
SR_CLUSTER_ATR = float(os.getenv("WS_SR_CLUSTER_ATR", "0.45"))
SR_MIN_TOUCHES_STRONG = int(os.getenv("WS_SR_MIN_TOUCHES", "3"))

SR_TOP_LEVELS = int(os.getenv("WS_SR_TOP_LEVELS", "6"))
SR_CHECK_NEAR = int(os.getenv("WS_SR_CHECK_NEAR", "2"))
SR_BLOCK_DIST_ATR = float(os.getenv("WS_SR_BLOCK_ATR", "0.65"))

# ===================== LOG =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [WS_BULLEX] %(message)s"
)
log = logging.getLogger("WS_BULLEX")

class C:
    G = "\033[92m"
    R = "\033[91m"
    Y = "\033[93m"
    B = "\033[94m"
    Z = "\033[0m"

def paint(s: str, color: str) -> str:
    return f"{color}{s}{C.Z}"

def dir_color(direction: str) -> str:
    return C.G if direction == "CALL" else (C.R if direction == "PUT" else C.Y)

_cache_ativos: List[str] = []
_cache_ativos_ts: float = 0.0

cooldown: Dict[str, float] = {}
cooldown_spike: Dict[str, float] = {}
cooldown_loss: Dict[str, float] = {}  # Cooldown após LOSS em ativo específico
consecutive_losses: Dict[str, int] = {}  # Contador de losses consecutivos por ativo
global_consecutive_losses: int = 0  # Losses consecutivos globais

# LightGBM globals
lgbm_model: Any = None  # Modelo LightGBM treinado
lgbm_data: List[Dict] = []  # Dados de treino acumulados
lgbm_trade_count: int = 0  # Contador para retreino

# ===================== TEMPO (candle fechado) =====================
def seconds_to_next(tf: int) -> float:
    now = time.time()
    return tf - (now % tf)

def wait_until_minus(tf: int, seconds_before: int):
    while True:
        s = seconds_to_next(tf)
        if s <= seconds_before:
            return
        time.sleep(min(0.25, max(0.04, s - seconds_before)))

def wait_for_next_open(tf: int):
    s = seconds_to_next(tf)
    time.sleep(s + 0.12)

def end_ts_closed(tf: int) -> float:
    now = time.time()
    return now - (now % tf) - 1  # garante candle fechado

# ===================== BULLEX CONNECTION =====================
def conectar_bullex() -> Bullex:
    if not EMAIL or not SENHA:
        raise RuntimeError("Defina BULLEX_EMAIL e BULLEX_PASS nas variáveis de ambiente.")
    log.info("Conectando à Bullex...")
    bx = Bullex(EMAIL, SENHA)
    bx.connect()

    for _ in range(12):
        if bx.check_connect():
            break
        time.sleep(1.5)

    if not bx.check_connect():
        raise RuntimeError("Falha na conexão com a Bullex.")

    bx.change_balance(CONTA)
    try:
        log.info(f"Conectado | Saldo: {bx.get_balance():.2f} | Conta: {CONTA}")
    except Exception:
        log.info(f"Conectado | Conta: {CONTA}")

    return bx

def ensure_connected(bx: Optional[Bullex]) -> Bullex:
    if bx is None:
        return conectar_bullex()
    try:
        if bx.check_connect():
            return bx
    except Exception:
        pass

    log.warning(paint("Conexão caiu. Tentando reconectar...", C.Y))
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

    return conectar_bullex()

def safe_call(bx: Bullex, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        if ("10054" in msg) or ("forçado o cancelamento" in msg) or ("goodbye" in msg) or ("10053" in msg):
            log.error(paint(f"Erro de conexão: {e}", C.R))
            ensure_connected(bx)
            return fn(*args, **kwargs)
        raise

# ===================== CANDLES =====================
def get_candles_df(bx: Bullex, ativo: str, timeframe: int, n: int, end_ts: Optional[float] = None) -> Optional[pd.DataFrame]:
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

        need_min = max(220, SR_LOOKBACK + 20)
        if len(df) < need_min:
            return None
        return df
    except Exception:
        return None

# ===================== ATIVOS / PAYOUT =====================
def obter_top_ativos_otc(bx: Bullex) -> List[str]:
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

    # Pega payouts
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
    log.info(f"TOP ativos: {top}")
    return top

# ===================== HELPERS =====================
def atr(df: pd.DataFrame, period: int = 14) -> float:
    sub = df.tail(period + 2)
    h = sub["high"].to_numpy(float)
    l = sub["low"].to_numpy(float)
    c = sub["close"].to_numpy(float)
    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1]))
    )
    return float(np.mean(tr[-period:]))

# ===================== LINHA DE TENDÊNCIA (LTA/LTB) =====================
def detect_trendline(df: pd.DataFrame, lookback: int, direction: str) -> Optional[Tuple[float, float]]:
    if len(df) < lookback:
        return None

    sub = df.tail(lookback)

    if direction == "CALL":
        pivots = []
        lows = sub["low"].to_numpy(float)

        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                pivots.append((i, lows[i]))

        if len(pivots) < 2:
            return None

        if pivots[-1][1] <= pivots[0][1]:
            return None

        x = np.array([p[0] for p in pivots])
        y = np.array([p[1] for p in pivots])
        slope, intercept = np.polyfit(x, y, 1)

        if slope <= 0:
            return None

        return (float(slope), float(intercept))

    else:  # PUT
        pivots = []
        highs = sub["high"].to_numpy(float)

        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                pivots.append((i, highs[i]))

        if len(pivots) < 2:
            return None

        if pivots[-1][1] >= pivots[0][1]:
            return None

        x = np.array([p[0] for p in pivots])
        y = np.array([p[1] for p in pivots])
        slope, intercept = np.polyfit(x, y, 1)

        if slope >= 0:
            return None

        return (float(slope), float(intercept))

def check_trendline_confluence(df: pd.DataFrame, pb_high: float, pb_low: float,
                                direction: str, atr_val: float) -> Dict[str, Any]:
    trendline = detect_trendline(df.tail(50), 50, direction)

    if trendline is None:
        return {"has_trendline": False, "confluence": 0.0, "distance": 999.0}

    slope, intercept = trendline
    x_pb = len(df.tail(50)) - 1
    lt_value = slope * x_pb + intercept

    if direction == "CALL":
        distance = abs(pb_low - lt_value) / max(atr_val, 1e-9)
        if distance < 0.3:
            return {"has_trendline": True, "confluence": 1.0, "distance": distance, "lt_value": lt_value}
        elif distance < 0.6:
            return {"has_trendline": True, "confluence": 0.6, "distance": distance, "lt_value": lt_value}
        else:
            return {"has_trendline": True, "confluence": 0.2, "distance": distance, "lt_value": lt_value}
    else:
        distance = abs(pb_high - lt_value) / max(atr_val, 1e-9)
        if distance < 0.3:
            return {"has_trendline": True, "confluence": 1.0, "distance": distance, "lt_value": lt_value}
        elif distance < 0.6:
            return {"has_trendline": True, "confluence": 0.6, "distance": distance, "lt_value": lt_value}
        else:
            return {"has_trendline": True, "confluence": 0.2, "distance": distance, "lt_value": lt_value}

def candle_dir(row: pd.Series) -> int:
    o = float(row["open"]); c = float(row["close"])
    return 1 if c > o else (-1 if c < o else 0)

def candle_range(row: pd.Series) -> float:
    return float(row["high"]) - float(row["low"])

def wick_fractions(row: pd.Series) -> Dict[str, float]:
    o = float(row["open"]); c = float(row["close"])
    h = float(row["high"]); l = float(row["low"])
    rng = max(h - l, 1e-9)
    upper = h - max(o, c)
    lower = min(o, c) - l
    body  = abs(c - o)
    return {"rng": rng, "upper_frac": upper / rng, "lower_frac": lower / rng, "body_frac": body / rng}

def is_spike_wicky(row: pd.Series, atr_val: float) -> bool:
    w = wick_fractions(row)
    if candle_range(row) < SPIKE_RANGE_ATR * atr_val:
        return False
    return (w["upper_frac"] >= SPIKE_WICK_FRAC) or (w["lower_frac"] >= SPIKE_WICK_FRAC)

def leg_efficiency(df_leg: pd.DataFrame) -> float:
    if len(df_leg) < 3:
        return 0.0
    closes = df_leg["close"].to_numpy(float)
    net = abs(closes[-1] - closes[0])
    gross = np.sum(np.abs(np.diff(closes))) + 1e-9
    return float(net / gross)

def chop_stats(df_m1: pd.DataFrame, lookback: int) -> Tuple[float, float]:
    sub = df_m1.tail(lookback)
    if len(sub) < 6:
        return 1.0, 0.0
    dirs = [candle_dir(sub.iloc[i]) for i in range(len(sub))]
    dirs2 = [d for d in dirs if d != 0]
    if len(dirs2) < 6:
        return 1.0, 0.0
    flips = 0
    for i in range(1, len(dirs2)):
        if dirs2[i] != dirs2[i-1]:
            flips += 1
    flips_frac = flips / max(1, (len(dirs2) - 1))
    eff = leg_efficiency(sub)
    return float(flips_frac), float(eff)

def compression_ratio(df_m1: pd.DataFrame, atr_val: float, lookback: int) -> float:
    sub = df_m1.tail(lookback)
    if len(sub) < 6:
        return 0.0
    rng = float(sub["high"].max() - sub["low"].min())
    return float(rng / max(atr_val, 1e-9))

def late_extension_atr(df_m1: pd.DataFrame, atr_val: float, lookback: int) -> float:
    sub = df_m1.tail(lookback)
    if len(sub) < 6:
        return 0.0
    o0 = float(sub["open"].iloc[0])
    cN = float(sub["close"].iloc[-1])
    return float(abs(cN - o0) / max(atr_val, 1e-9))

# ===================== S/R FORTE =====================
def _cluster_levels(levels: List[float], tol: float) -> List[Tuple[float, int]]:
    if not levels:
        return []
    levels = sorted(levels)
    clusters = []
    cur = [levels[0]]
    for x in levels[1:]:
        if abs(x - cur[-1]) <= tol:
            cur.append(x)
        else:
            clusters.append((float(np.mean(cur)), len(cur)))
            cur = [x]
    clusters.append((float(np.mean(cur)), len(cur)))
    clusters.sort(key=lambda t: t[1], reverse=True)
    return clusters

def strong_sr_levels_last200(df_m1: pd.DataFrame, atr_val: float) -> Tuple[List[Tuple[float,int]], List[Tuple[float,int]]]:
    sub = df_m1.tail(SR_LOOKBACK)
    h = sub["high"].to_numpy(float)
    l = sub["low"].to_numpy(float)

    highs: List[float] = []
    lows: List[float] = []

    k = 2
    for i in range(k, len(sub) - k):
        if h[i] == np.max(h[i-k:i+k+1]):
            highs.append(float(h[i]))
        if l[i] == np.min(l[i-k:i+k+1]):
            lows.append(float(l[i]))

    tol_price = max(atr_val * SR_CLUSTER_ATR, 1e-9)

    res = _cluster_levels(highs, tol_price)
    sup = _cluster_levels(lows, tol_price)

    res = [(lvl, n) for (lvl, n) in res if n >= SR_MIN_TOUCHES_STRONG]
    sup = [(lvl, n) for (lvl, n) in sup if n >= SR_MIN_TOUCHES_STRONG]
    return res, sup

def pick_top_levels(levels: List[Tuple[float,int]], top_n: int) -> List[Tuple[float,int]]:
    return sorted(levels, key=lambda t: t[1], reverse=True)[:top_n]

def nearest_k(levels: List[Tuple[float,int]], price: float, k: int) -> List[Tuple[float,int,float]]:
    arr = [(lvl, touches, abs(lvl - price)) for (lvl, touches) in levels]
    arr.sort(key=lambda x: x[2])
    return arr[:k]

def sr_block_directional_multi(df_m1: pd.DataFrame, atr_val: float, direction: str) -> Optional[str]:
    res, sup = strong_sr_levels_last200(df_m1, atr_val)
    price = float(df_m1["close"].iloc[-1])
    atr_safe = max(atr_val, 1e-9)

    res = pick_top_levels(res, SR_TOP_LEVELS)
    sup = pick_top_levels(sup, SR_TOP_LEVELS)

    if direction == "CALL" and res:
        above = [(lvl,t) for (lvl,t) in res if lvl >= price]
        if not above:
            return None
        cand = nearest_k(above, price, SR_CHECK_NEAR)
        for lvl, touches, dist_abs in cand:
            dist_atr = dist_abs / atr_safe
            zone = min(SR_BLOCK_DIST_ATR + 0.10 * max(0, touches - SR_MIN_TOUCHES_STRONG), 1.30)
            if dist_atr <= zone:
                return f"bloqueado_RES(nivel={lvl:.6f},toques={touches},dist={dist_atr:.2f}ATR<=zona{zone:.2f})"
        return None

    if direction == "PUT" and sup:
        below = [(lvl,t) for (lvl,t) in sup if lvl <= price]
        if not below:
            return None
        cand = nearest_k(below, price, SR_CHECK_NEAR)
        for lvl, touches, dist_abs in cand:
            dist_atr = dist_abs / atr_safe
            zone = min(SR_BLOCK_DIST_ATR + 0.10 * max(0, touches - SR_MIN_TOUCHES_STRONG), 1.30)
            if dist_atr <= zone:
                return f"bloqueado_SUP(nivel={lvl:.6f},toques={touches},dist={dist_atr:.2f}ATR<=zona{zone:.2f})"
        return None

    return None

def sr_pingpong_zone(df_m1: pd.DataFrame, atr_val: float) -> Optional[str]:
    res, sup = strong_sr_levels_last200(df_m1, atr_val)
    if not res or not sup:
        return None
    price = float(df_m1["close"].iloc[-1])
    atr_safe = max(atr_val, 1e-9)

    res_near = sorted([(lvl,t) for (lvl,t) in res], key=lambda x: abs(x[0]-price))[:2]
    sup_near = sorted([(lvl,t) for (lvl,t) in sup], key=lambda x: abs(x[0]-price))[:2]

    above = [(lvl,t) for (lvl,t) in res_near if lvl >= price]
    below = [(lvl,t) for (lvl,t) in sup_near if lvl <= price]
    if not above or not below:
        return None

    r_lvl, r_t = min(above, key=lambda x: abs(x[0]-price))
    s_lvl, s_t = min(below, key=lambda x: abs(x[0]-price))

    corridor_atr = abs(r_lvl - s_lvl) / atr_safe
    if corridor_atr <= 0.85:
        return f"pingpong(corredor={corridor_atr:.2f}ATR sup={s_lvl:.6f} res={r_lvl:.6f})"
    return None

# ===================== VALIDAÇÃO DE ENTRADA =====================
def validate_entry_quality(df_m1: pd.DataFrame, atr_val: float, direction: str, entry_price: float, pb_high: float, pb_low: float) -> Dict[str, Any]:
    if len(df_m1) < 5:
        return {"valid": False, "confidence": 0.0, "reason": "dados_insuficientes"}

    last_candle = df_m1.iloc[-1]
    open_price = float(last_candle["open"])
    close_price = float(last_candle["close"])
    high_price = float(last_candle["high"])
    low_price = float(last_candle["low"])

    candle_range = high_price - low_price
    body = abs(close_price - open_price)
    body_ratio = body / max(candle_range, 1e-9)

    if body_ratio < 0.25:
        return {"valid": False, "confidence": 0.0, "reason": f"candle_fraco(body={body_ratio:.2f})"}

    if direction == "CALL":
        stop_loss = pb_low - (0.15 * atr_val)
        risk = entry_price - stop_loss
        target_1 = entry_price + (risk * 1.5)
        recent_highs = df_m1.tail(20)["high"].to_numpy(float)
        max_recent = float(np.max(recent_highs))
        if target_1 > max_recent * 1.005:
            confidence = 0.75
        else:
            confidence = 0.55
    else:
        stop_loss = pb_high + (0.15 * atr_val)
        risk = stop_loss - entry_price
        target_1 = entry_price - (risk * 1.5)
        recent_lows = df_m1.tail(20)["low"].to_numpy(float)
        min_recent = float(np.min(recent_lows))
        if target_1 < min_recent * 0.995:
            confidence = 0.75
        else:
            confidence = 0.55

    risk_atr = risk / max(atr_val, 1e-9)
    if risk_atr < 0.2 or risk_atr > 2.0:
        return {"valid": False, "confidence": 0.0, "reason": f"risco_inadequado({risk_atr:.2f}ATR)"}

    last_3_closes = df_m1.tail(3)["close"].to_numpy(float)
    momentum = abs(last_3_closes[-1] - last_3_closes[0]) / max(atr_val, 1e-9)
    if momentum < 0.10:
        confidence *= 0.80

    last_3 = df_m1.tail(3)
    aligned = 0
    for _, row in last_3.iterrows():
        c = float(row["close"])
        o = float(row["open"])
        if (direction == "CALL" and c > o) or (direction == "PUT" and c < o):
            aligned += 1

    alignment_ratio = aligned / 3.0
    if alignment_ratio >= 0.67:
        confidence *= 1.15
    elif alignment_ratio < 0.34:
        confidence *= 0.80

    confidence = min(1.0, max(0.0, confidence))
    if confidence < 0.40:
        return {"valid": False, "confidence": confidence, "reason": f"confianca_baixa({confidence:.2f})"}

    return {
        "valid": True,
        "confidence": float(confidence),
        "risk_atr": float(risk_atr),
        "risk_reward": 1.5,
        "momentum": float(momentum),
        "body_ratio": float(body_ratio),
        "alignment": float(alignment_ratio),
        "reason": "entrada_validada"
    }

def validate_trend_continuation(df_m1: pd.DataFrame, impulso_dir: str, pb_end_idx: int) -> Dict[str, Any]:
    if len(df_m1) < pb_end_idx + 30:
        return {"valid": True, "reason": "dados_insuficientes", "strength": 0.5}

    pre_impulse = df_m1.iloc[max(0, pb_end_idx - 25):pb_end_idx - 5]
    if len(pre_impulse) < 10:
        return {"valid": True, "reason": "contexto_curto", "strength": 0.5}

    closes = pre_impulse["close"].to_numpy(float)
    price_change = closes[-1] - closes[0]
    price_change_pct = abs(price_change) / max(closes[0], 1e-9)

    if impulso_dir == "PUT":
        if price_change > 0:
            if price_change_pct > 0.015:
                return {"valid": False, "reason": "contra_tendencia_forte_alta", "strength": 0.2}
            return {"valid": True, "reason": "contra_tendencia_fraca", "strength": 0.4}
        return {"valid": True, "reason": "continuacao_queda", "strength": min(1.0, price_change_pct * 50)}
    else:
        if price_change < 0:
            if price_change_pct > 0.015:
                return {"valid": False, "reason": "contra_tendencia_forte_baixa", "strength": 0.2}
            return {"valid": True, "reason": "contra_tendencia_fraca", "strength": 0.4}
        return {"valid": True, "reason": "continuacao_alta", "strength": min(1.0, price_change_pct * 50)}

def analyze_market_context(df_m1: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    if len(df_m1) < 50:
        return {"quality": 0.0, "context": "insuficiente"}

    recent = df_m1.tail(20)
    closes = recent["close"].to_numpy(float)
    highs = recent["high"].to_numpy(float)
    lows = recent["low"].to_numpy(float)

    bullish = sum(1 for i in range(len(recent)) if closes[i] > recent["open"].iloc[i])
    bearish = sum(1 for i in range(len(recent)) if closes[i] < recent["open"].iloc[i])
    directional_bias = abs(bullish - bearish) / len(recent)

    ranges = [highs[i] - lows[i] for i in range(len(recent))]
    avg_range = np.mean(ranges)
    std_range = np.std(ranges)
    volatility_consistency = 1.0 - min(1.0, std_range / max(avg_range, 1e-9))

    hh_count = sum(1 for i in range(5, len(recent)) if highs[i] > max(highs[i-5:i]))
    ll_count = sum(1 for i in range(5, len(recent)) if lows[i] < min(lows[i-5:i]))
    structure_quality = max(hh_count, ll_count) / max(1, len(recent) - 5)

    last_10 = closes[-10:]
    price_momentum = abs(last_10[-1] - last_10[0]) / (max(atr_val, 1e-9) * 10)
    price_momentum = min(1.0, price_momentum)

    body_ratios = []
    for i in range(len(recent)):
        o = recent["open"].iloc[i]
        c = recent["close"].iloc[i]
        h = recent["high"].iloc[i]
        l = recent["low"].iloc[i]
        body = abs(c - o)
        total = h - l
        if total > 1e-9:
            body_ratios.append(body / total)
    avg_body_ratio = np.mean(body_ratios) if body_ratios else 0.0

    quality = (
        directional_bias * 0.25 +
        volatility_consistency * 0.20 +
        structure_quality * 0.25 +
        price_momentum * 0.15 +
        avg_body_ratio * 0.15
    )

    return {
        "quality": float(quality),
        "directional_bias": float(directional_bias),
        "volatility_consistency": float(volatility_consistency),
        "structure_quality": float(structure_quality),
        "price_momentum": float(price_momentum),
        "body_ratio": float(avg_body_ratio),
        "context": "excelente" if quality > 0.70 else ("bom" if quality > 0.55 else ("mediano" if quality > 0.40 else "ruim"))
    }

# ===================== GESTÃO DE BANCA =====================
def calcular_stake_dinamico(bx: Bullex, base_stake: float) -> float:
    if not USE_DYNAMIC_STAKE:
        return float(max(VALOR_MINIMO, base_stake))
    try:
        saldo = float(bx.get_balance())
        stake = (saldo * PERCENT_BANCA) / 100.0
        return float(max(VALOR_MINIMO, stake))
    except Exception:
        return float(max(VALOR_MINIMO, base_stake))

def verificar_meta_atingida(saldo_inicial: float, saldo_atual: float) -> Tuple[bool, float]:
    lucro = saldo_atual - saldo_inicial
    lucro_percent = (lucro / saldo_inicial) * 100.0
    if lucro_percent >= META_LUCRO_PERCENT:
        return True, lucro_percent
    if lucro_percent <= -STOP_LOSS_PERCENT:
        return True, lucro_percent
    return False, lucro_percent

# ===================== IA ONLINE (Bayes + UCB) =====================
def _safe_load_json(path: str) -> Dict[str, Any]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"meta": {"total": 0}, "arms": {}}

def _safe_save_json(path: str, data: Dict[str, Any]):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ===================== LIGHTGBM - GRADIENT BOOSTING =====================

def lgbm_extract_features(setup: Dict[str, Any], df_m1: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Extrai features numéricas do setup para o LightGBM."""
    score = float(setup.get("score", 0.0))
    retr = float(setup.get("retr", 0.0))
    A_atr = float(setup.get("A_atr", 0.0))
    effA = float(setup.get("effA", 0.0))
    flips = float(setup.get("flips", 0.0))
    pb_len = float(setup.get("pb_len", 0))
    distBreak = float(setup.get("distBreak", 0.0))
    late_ext = float(setup.get("late_ext", 0.0))
    compression = float(setup.get("compression", 0.0))
    market_quality = float(setup.get("market_quality", 0.5))
    entry_conf = float(setup.get("entry_confidence", 0.5))
    
    ctx = str(setup.get("ctx", "neutro"))
    ctx_score = 1.0 if ctx == "bom" else (0.5 if ctx == "neutro" else 0.0)
    
    dir_str = str(setup.get("dir", "NEUTRAL"))
    dir_enc = 1.0 if dir_str == "CALL" else (-1.0 if dir_str == "PUT" else 0.0)
    
    features = np.array([
        score, retr, A_atr, effA, flips, pb_len, distBreak,
        late_ext, compression, market_quality, entry_conf, ctx_score, dir_enc
    ], dtype=np.float32)
    
    return features

def lgbm_load_model():
    """Carrega modelo LightGBM do disco se existir."""
    global lgbm_model
    if not LGBM_ON or lgbm_model is not None:
        return
    
    try:
        if os.path.exists(LGBM_MODEL_FILE):
            with open(LGBM_MODEL_FILE, "rb") as f:
                lgbm_model = pickle.load(f)
            log.info(f"[LGBM] Modelo carregado de {LGBM_MODEL_FILE}")
    except Exception as e:
        log.warning(f"[LGBM] Erro ao carregar modelo: {e}")
        lgbm_model = None

def lgbm_save_model():
    """Salva modelo LightGBM no disco."""
    global lgbm_model
    if lgbm_model is None:
        return
    
    try:
        with open(LGBM_MODEL_FILE, "wb") as f:
            pickle.dump(lgbm_model, f)
        log.info(f"[LGBM] Modelo salvo em {LGBM_MODEL_FILE}")
    except Exception as e:
        log.warning(f"[LGBM] Erro ao salvar modelo: {e}")

def lgbm_load_data():
    """Carrega dados de treino do disco."""
    global lgbm_data
    try:
        if os.path.exists(LGBM_DATA_FILE):
            with open(LGBM_DATA_FILE, "r", encoding="utf-8") as f:
                lgbm_data = json.load(f)
            log.info(f"[LGBM] {len(lgbm_data)} amostras carregadas")
    except Exception:
        lgbm_data = []

def lgbm_save_data():
    """Salva dados de treino no disco."""
    global lgbm_data
    try:
        with open(LGBM_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(lgbm_data, f, ensure_ascii=False)
    except Exception:
        pass

def lgbm_add_sample(setup: Dict[str, Any], result: float):
    """Adiciona uma amostra de treino (setup + resultado)."""
    global lgbm_data, lgbm_trade_count
    if not LGBM_ON or result == 0:
        return
    
    features = lgbm_extract_features(setup).tolist()
    label = 1 if result > 0 else 0
    
    lgbm_data.append({"features": features, "label": label})
    lgbm_trade_count += 1
    
    if len(lgbm_data) > 1000:
        lgbm_data = lgbm_data[-1000:]
    
    lgbm_save_data()
    
    if lgbm_trade_count >= LGBM_RETRAIN_EVERY and len(lgbm_data) >= LGBM_MIN_SAMPLES:
        lgbm_train()
        lgbm_trade_count = 0

def lgbm_train():
    """Treina ou retreina o modelo LightGBM com os dados acumulados."""
    global lgbm_model, lgbm_data
    
    if not LGBM_ON or lgb is None or len(lgbm_data) < LGBM_MIN_SAMPLES:
        return
    
    try:
        X = np.array([d["features"] for d in lgbm_data], dtype=np.float32)
        y = np.array([d["label"] for d in lgbm_data], dtype=np.int32)
        
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 15,
            "max_depth": 5,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "min_child_samples": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1,
            "force_col_wise": True,
        }
        
        lgbm_model = lgb.LGBMClassifier(**params)
        lgbm_model.fit(X, y)
        
        lgbm_save_model()
        
        preds = lgbm_model.predict(X)
        acc = (preds == y).mean() * 100
        log.info(paint(f"[LGBM] Modelo treinado! Amostras={len(lgbm_data)} | Acc={acc:.1f}%", C.G))
        
    except Exception as e:
        log.warning(f"[LGBM] Erro no treino: {e}")

def lgbm_predict(setup: Dict[str, Any]) -> Tuple[float, bool]:
    """Prediz probabilidade de WIN usando LightGBM."""
    global lgbm_model
    
    if not LGBM_ON or lgbm_model is None:
        return 0.5, False
    
    try:
        features = lgbm_extract_features(setup).reshape(1, -1)
        proba = lgbm_model.predict_proba(features)[0]
        return float(proba[1]), True
    except Exception as e:
        log.warning(f"[LGBM] Erro na predição: {e}")
        return 0.5, False

def ensemble_predict(ativo: str, setup: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combina predições do Bayesiano e LightGBM com GATEs inteligentes.
    Modelo profissional: prioriza ZONA (S/R + LT) sobre candle.
    """
    # Predição Bayesiana
    bayes_pred = ai_predict(ativo, setup, stats)
    bayes_prob = float(bayes_pred["prob"])
    bayes_conf = float(bayes_pred["conf"])
    n_arm = int(bayes_pred["n_arm"])
    prior = float(bayes_pred.get("prior", 0.50))
    
    # Predição LightGBM
    lgbm_prob, lgbm_available = lgbm_predict(setup)
    
    # ── Extrair features de zona (DOM Forex) ──
    sc         = float(setup.get("score", 0.0))
    ctx_val    = float(setup.get("market_quality", 0.40))
    entry_conf_val = float(setup.get("entry_confidence", 0.50))
    sr_prox    = float(setup.get("sr_proximity", 0.0))
    sr_tq      = int(setup.get("sr_touches", 0))
    sr_w       = float(setup.get("sr_weight", 0.0))
    has_lt     = bool(setup.get("has_lt", False))
    lt_pts     = int(setup.get("lt_points", setup.get("pb_len", 0)))
    lt_conf    = float(setup.get("lt_confluence", 0.0))
    conf_count = int(setup.get("confluence_count", 0))

    # ── Conceitos de zona ──
    sr_forte = sr_prox > 0.60 and sr_tq >= 4
    sr_basico = sr_tq >= 3 and sr_w >= 4.0
    zona_forte = (sr_tq >= 3 and sr_w >= 4.0) or (has_lt and lt_pts >= 3)
    setup_forte = sc >= 0.55 and ctx_val >= 0.55 and conf_count >= 2
    
    if not lgbm_available or not LGBM_ON:
        reason_detail = "lgbm_off" if not LGBM_ON else "lgbm_unavailable"
        
        if n_arm >= AI_MIN_SAMPLES:
            should_trade = (bayes_prob >= 0.52) and (bayes_conf >= AI_CONF_MIN)
            reason_suffix = f"hist,prob={bayes_prob:.2f},n={n_arm}"
        else:
            # ── WARMUP INTELIGENTE W1-W7 ──
            if ctx_val < 0.30 and not zona_forte:
                should_trade = False
                reason_suffix = f"W1_ctx_pessimo={ctx_val:.2f}"
            elif sc >= 0.45 and ctx_val >= 0.40:
                should_trade = True
                reason_suffix = f"W2_sc_ctx_ok(sc={sc:.2f},ctx={ctx_val:.2f})"
            elif zona_forte and sc >= 0.40:
                should_trade = True
                reason_suffix = f"W3_zona_forte(sc={sc:.2f},sr={sr_tq}t,lt={lt_pts}pts)"
            elif sc >= 0.40 and sr_forte:
                should_trade = True
                reason_suffix = f"W4_score_sr_ok(sc={sc:.2f},sr={sr_tq}t)"
            elif sc >= 0.45 and entry_conf_val >= 0.40:
                should_trade = True
                reason_suffix = f"W5_sc_candle_ok(sc={sc:.2f},ec={entry_conf_val:.2f})"
            elif bayes_prob >= 0.50 and sc >= 0.42:
                should_trade = True
                reason_suffix = f"W6_prior_score(prob={bayes_prob:.2f},sc={sc:.2f})"
            elif IA_MODE == "learning" and sc >= 0.40 and ctx_val >= 0.40:
                should_trade = True
                reason_suffix = f"W7_learning(sc={sc:.2f},ctx={ctx_val:.2f})"
            else:
                should_trade = False
                reason_suffix = f"fraco(sc={sc:.2f},prob={bayes_prob:.2f},ctx={ctx_val:.2f})"
        
        return {
            "should_trade": should_trade,
            "bayes_prob": bayes_prob,
            "lgbm_prob": 0.5,
            "ensemble_prob": bayes_prob,
            "reason": f"bayes_only({reason_detail},{reason_suffix})",
            "bayes_conf": bayes_conf,
            "n_arm": n_arm
        }
    
    # ── Ensemble ponderado ──
    bayes_weight = min(1.0, n_arm / AI_MIN_SAMPLES) * 0.5 + 0.25
    lgbm_weight = 1.0 - bayes_weight
    ensemble_prob = bayes_prob * bayes_weight + lgbm_prob * lgbm_weight
    
    if ENSEMBLE_MODE == "both":
        bayes_ok = bayes_prob >= AI_MIN_PROB
        lgbm_ok = lgbm_prob >= LGBM_MIN_PROB
        should_trade = bayes_ok and lgbm_ok
        reason = f"both(B={bayes_prob:.2f},L={lgbm_prob:.2f})"
    elif ENSEMBLE_MODE == "any":
        bayes_ok = bayes_prob >= AI_MIN_PROB
        lgbm_ok = lgbm_prob >= LGBM_MIN_PROB
        should_trade = (bayes_ok or lgbm_ok) and ensemble_prob >= 0.55
        reason = f"any(B={bayes_prob:.2f},L={lgbm_prob:.2f},ens={ensemble_prob:.2f})"
    else:
        min_ens_weighted = max(AI_MIN_PROB, 0.58)
        should_trade = ensemble_prob >= min_ens_weighted
        reason = f"weighted(ens={ensemble_prob:.2f},min={min_ens_weighted:.2f})"
    
    # ── WARMUP COM LGBM ──
    if n_arm < AI_MIN_SAMPLES:
        warmup_threshold = LGBM_MIN_PROB
        
        # Threshold dinâmico baseado no contexto
        if ctx_val < 0.40:
            min_ens = 0.58
        elif ctx_val < 0.50:
            min_ens = 0.54
        else:
            min_ens = 0.50
        
        if entry_conf_val < 0.50:
            min_ens += 0.02
        
        # S/R forte relaxa threshold
        if sr_forte:
            min_ens -= 0.06
        elif sr_basico:
            min_ens -= 0.03

        # REGRA #1: LGBM muito negativo
        if lgbm_prob < 0.30:
            should_trade = False
            reason = f"warmup_danger(L={lgbm_prob:.2f}<0.30)"
        # REGRA #2: Ambos negativos
        elif bayes_prob < 0.50 and lgbm_prob < 0.50:
            should_trade = False
            reason = f"warmup_consenso_neg(B={bayes_prob:.2f},L={lgbm_prob:.2f})"
        # REGRA #3: Ambos positivos
        elif bayes_prob >= 0.58 and lgbm_prob >= 0.58:
            should_trade = True
            reason = f"warmup_consenso_ok(B={bayes_prob:.2f},L={lgbm_prob:.2f})"
        # REGRA #4: Bayes forte + LGBM neutro
        elif bayes_prob >= 0.63 and lgbm_prob >= 0.50:
            should_trade = True
            reason = f"warmup_bayes_forte(B={bayes_prob:.2f},L={lgbm_prob:.2f})"
        # REGRA #5: LGBM forte + Bayes neutro
        elif lgbm_prob >= 0.62 and bayes_prob >= 0.50:
            should_trade = True
            reason = f"warmup_lgbm_forte(B={bayes_prob:.2f},L={lgbm_prob:.2f})"
        # REGRA #6: Ensemble atinge threshold dinâmico
        elif ensemble_prob >= min_ens:
            should_trade = True
            reason = f"warmup_ens_ctx(B={bayes_prob:.2f},L={lgbm_prob:.2f},ens={ensemble_prob:.2f},min={min_ens:.2f})"
        else:
            should_trade = False
            reason = f"warmup_fraco(B={bayes_prob:.2f},L={lgbm_prob:.2f},ens={ensemble_prob:.2f},min={min_ens:.2f})"
    
    return {
        "should_trade": should_trade,
        "bayes_prob": bayes_prob,
        "lgbm_prob": lgbm_prob,
        "ensemble_prob": ensemble_prob,
        "reason": reason,
        "bayes_conf": bayes_conf,
        "n_arm": n_arm
    }

# ===================== IA BAYESIANA =====================

def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def _bucket(x: float, step: float, lo: float, hi: float) -> int:
    x = _clip(x, lo, hi)
    return int(round((x - lo) / step))

def ai_make_key(ativo: str, setup: Dict[str, Any]) -> str:
    d = str(setup.get("dir", "NEUTRAL"))
    sc = float(setup.get("score", 0.0))
    pb = int(setup.get("pb_len", 0))
    retr = float(setup.get("retr", 0.0))
    Aatr = float(setup.get("A_atr", 0.0))
    effA = float(setup.get("effA", 0.0))
    flips = float(setup.get("flips", 0.0))
    distBreak = float(setup.get("distBreak", 0.0))

    b_sc = _bucket(sc, 0.04, 0.40, 1.00)
    b_re = _bucket(retr, 0.06, 0.10, 0.80)
    b_A  = _bucket(Aatr, 0.40, 0.60, 6.00)
    b_eff = _bucket(effA, 0.05, 0.05, 0.50)
    b_flip = _bucket(flips, 0.10, 0.0, 0.80)
    b_dist = _bucket(distBreak, 0.05, 0.0, 0.50)

    return f"{d}|sc{b_sc}|pb{pb}|re{b_re}|A{b_A}|eff{b_eff}|fl{b_flip}|dst{b_dist}"

def ai_prior_from_setup(setup: Dict[str, Any]) -> float:
    """
    Prior PROFISSIONAL baseado em zonas de alta probabilidade.
    Prioriza: ZONA (S/R + LT) sobre candle.
    """
    sc = float(setup.get("score", 0.0))
    ctx = float(setup.get("market_quality", 0.40))
    sr_prox = float(setup.get("sr_proximity", 0.0))
    sr_tq = int(setup.get("sr_touches", 0))
    sr_w = float(setup.get("sr_weight", 0.0))
    has_lt = bool(setup.get("has_lt", False))
    lt_pts = int(setup.get("lt_points", setup.get("pb_len", 0)))
    lt_conf = float(setup.get("lt_confluence", 0.0))
    candle_str = float(setup.get("candle_strength", setup.get("entry_confidence", 0.0)))
    conf_count = int(setup.get("confluence_count", 0))
    effA = float(setup.get("effA", 0.0))

    # Base no score
    p = 0.48 + (sc - 0.40) * 0.40

    # 1. ZONA S/R - fator mais importante
    if sr_tq >= 5 and sr_w >= 8.0:
        p += 0.08
    elif sr_tq >= 3 and sr_w >= 4.0:
        p += 0.05
    elif sr_tq >= 2:
        p += 0.02

    # 2. TRENDLINE alinhada
    if has_lt and lt_pts >= 4:
        p += 0.06
    elif has_lt and lt_pts >= 2:
        p += 0.03

    # 3. Contexto de mercado
    if ctx >= 0.70:
        p += 0.04
    elif ctx >= 0.55:
        p += 0.02
    elif ctx < 0.35:
        p -= 0.04

    # 4. Confluência alta
    if conf_count >= 5:
        p += 0.04
    elif conf_count >= 4:
        p += 0.02

    # 5. Candle (bônus, não requisito)
    if candle_str >= 0.60:
        p += 0.03
    elif candle_str >= 0.30:
        p += 0.01

    # 6. Direcionalidade
    if effA > 0.20:
        p += 0.02
    elif effA < 0.05:
        p -= 0.02

    return _clip(p, 0.42, 0.78)

def ai_predict(ativo: str, setup: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, float]:
    key = ai_make_key(ativo, setup)
    arms = stats.setdefault("arms", {})
    meta = stats.setdefault("meta", {"total": 0})

    total = int(meta.get("total", 0))
    arm = arms.get(key)

    prior = ai_prior_from_setup(setup)

    if arm is None:
        a = 2.0 * prior
        b = 2.0 * (1.0 - prior)
        n = 0
        arms[key] = {"a": a, "b": b, "n": n}
        bayes_mean = a / (a + b)
        ucb01 = 1.0
        conf = float(_clip(0.55 + float(setup.get("score", 0.0)) * 0.30, 0.0, 0.90))
        return {"prob": float(bayes_mean), "bayes": float(bayes_mean), "ucb01": float(ucb01),
                "conf": float(conf), "n_arm": 0, "total": total, "key": key, "prior": prior}

    a = float(arm.get("a", 1.0))
    b = float(arm.get("b", 1.0))
    n = int(arm.get("n", 0))

    bayes_mean = a / (a + b)

    if n <= 0:
        ucb01 = 1.0
    else:
        bonus = math.sqrt(2.0 * math.log(max(2, total + 1)) / max(1, n))
        ucb01 = _clip(bayes_mean + bonus, 0.0, 1.0)

    conf = _clip(n / (n + 10.0), 0.0, 0.99)

    w = _clip(n / (n + 25.0), 0.0, 1.0)
    prob = (1.0 - w) * prior + w * bayes_mean
    prob = _clip(prob, 0.0, 1.0)

    return {"prob": float(prob), "bayes": float(bayes_mean), "ucb01": float(ucb01),
            "conf": float(conf), "n_arm": n, "total": total, "key": key, "prior": prior}

def ai_update(ativo: str, setup: Dict[str, Any], pnl: float, stats: Dict[str, Any]):
    if pnl == 0:
        return

    key = ai_make_key(ativo, setup)
    arms = stats.setdefault("arms", {})
    meta = stats.setdefault("meta", {"total": 0})
    patterns = stats.setdefault("patterns", {})

    arm = arms.get(key)
    if arm is None:
        prior = ai_prior_from_setup(setup)
        arms[key] = {"a": 2.0 * prior, "b": 2.0 * (1.0 - prior), "n": 0}
        arm = arms[key]

    a = float(arm.get("a", 1.0))
    b = float(arm.get("b", 1.0))
    n = int(arm.get("n", 0))

    if pnl > 0:
        a += 1.0
    else:
        b += 1.0

    n += 1
    arm["a"], arm["b"], arm["n"] = a, b, n
    meta["total"] = int(meta.get("total", 0)) + 1

    pattern = patterns.get(key)
    if pattern is None:
        patterns[key] = {"trades": 0, "wins": 0, "losses": 0}
        pattern = patterns[key]

    pattern["trades"] += 1
    if pnl > 0:
        pattern["wins"] += 1
    else:
        pattern["losses"] += 1

def ai_retrain_on_loss(ativo: str, setup: Dict[str, Any], stats: Dict[str, Any]):
    """
    RETREINO SEVERO: Aplica penalidade extra no padrão que causou LOSS.
    
    - Aumenta 'b' (falhas) no Bayesian em proporção ao RETRAIN_PENALTY
    - Marca padrão como arriscado para evitar entrada imediata igual
    - Salva imediatamente para persistir aprendizado
    """
    key = ai_make_key(ativo, setup)
    arms = stats.setdefault("arms", {})
    patterns = stats.setdefault("patterns", {})
    
    arm = arms.get(key)
    if arm is None:
        prior = ai_prior_from_setup(setup)
        arms[key] = {"a": 2.0 * prior, "b": 2.0 * (1.0 - prior), "n": 0}
        arm = arms[key]
    
    # PENALIDADE SEVERA: aumenta 'b' (falhas) significativamente
    penalty_factor = max(1, int(RETRAIN_PENALTY * 10))  # Ex: 0.25 -> 2.5 -> 2 falhas extras
    arm["b"] = float(arm.get("b", 1.0)) + penalty_factor
    
    # Marcar padrão como "queimado" temporariamente
    pattern = patterns.get(key)
    if pattern is None:
        patterns[key] = {"trades": 0, "wins": 0, "losses": 0, "burned_until": 0}
        pattern = patterns[key]
    
    # Queimar padrão por 30 minutos (evita repetir mesmo erro)
    pattern["burned_until"] = time.time() + 1800  # 30 min
    pattern["last_loss_time"] = time.time()
    
    # Log detalhado do retreino
    new_prob = arm.get("a", 1.0) / (arm.get("a", 1.0) + arm["b"])
    log.warning(f"[AI-RETRAIN] {ativo} key={key[:30]}... | penalidade={penalty_factor} | nova_prob={new_prob:.2f}")

# ===================== PERNADA B =====================
def pernada_b(df_m1: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    if len(df_m1) < 240:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["poucas_velas"]}

    context = analyze_market_context(df_m1, atr_val)
    market_quality = float(context.get("quality", 0.0))

    flips_frac, eff_zone = chop_stats(df_m1, CHOP_LOOKBACK)
    comp = compression_ratio(df_m1, atr_val, COMP_LOOKBACK)
    late_ext = late_extension_atr(df_m1, atr_val, LATE_LOOKBACK)

    decision = df_m1.iloc[-1]
    q = wick_fractions(decision)

    if q["body_frac"] < MIN_BODY_FRAC_BREAK:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
                "reasons": [f"gatilho_fraco(body={q['body_frac']:.2f})"]}

    ping = sr_pingpong_zone(df_m1, atr_val)
    if ping:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": [ping]}

    best = None

    for pb_len in range(PULLBACK_MIN, PULLBACK_MAX + 1):
        pb = df_m1.iloc[-(pb_len + 1):-1]
        if len(pb) != pb_len:
            continue

        for w in range(IMPULSO_JANELA_MIN, IMPULSO_JANELA_MAX + 1):
            imp = df_m1.iloc[-(pb_len + 1 + w):-(pb_len + 1)]
            if len(imp) != w:
                continue

            top = float(imp["high"].max())
            bot = float(imp["low"].min())
            size_A = top - bot

            if size_A < IMPULSO_MIN_ATR * atr_val:
                continue

            start = float(imp["open"].iloc[0])
            end = float(imp["close"].iloc[-1])
            move = end - start

            dir_impulso_A = "PUT" if move < 0 else ("CALL" if move > 0 else "NEUTRAL")
            if dir_impulso_A == "NEUTRAL":
                continue

            eff_A = leg_efficiency(imp)
            if eff_A < MIN_EFF_A:
                continue

            contra = 0
            for _, r in pb.iterrows():
                d = candle_dir(r)
                if dir_impulso_A == "PUT" and d == 1:
                    contra += 1
                if dir_impulso_A == "CALL" and d == -1:
                    contra += 1

            if contra < max(1, int(math.ceil(pb_len * 0.50))):
                continue

            impulse_start_idx = len(df_m1) - (pb_len + 1 + w)
            trend_validation = validate_trend_continuation(df_m1, dir_impulso_A, impulse_start_idx)
            trend_strength = float(trend_validation.get("strength", 0.0)) if trend_validation.get("valid", False) else 0.0

            if dir_impulso_A == "PUT":
                pb_high = float(pb["high"].max())
                retr = (pb_high - bot) / max(size_A, 1e-9)
            else:
                pb_low = float(pb["low"].min())
                retr = (top - pb_low) / max(size_A, 1e-9)

            if retr < RETR_MIN or retr > RETR_MAX:
                continue

            c1 = float(decision["close"])
            dir_entrada = dir_impulso_A

            blk_sr = sr_block_directional_multi(df_m1, atr_val, dir_entrada)
            if blk_sr:
                continue

            pb_high = float(pb["high"].max())
            pb_low = float(pb["low"].min())

            if dir_entrada == "CALL":
                if not (c1 > pb_low + BREAK_MARGIN_ATR * atr_val):
                    continue
                if q["upper_frac"] > MAX_WICK_AGAINST:
                    continue

                dist = (c1 - pb_low) / max(atr_val, 1e-9)
                if dist > MAX_BREAK_DISTANCE_ATR:
                    continue

                entry_validation = validate_entry_quality(df_m1, atr_val, "CALL", c1, pb_high, pb_low)
                if not entry_validation.get("valid", False):
                    continue

                entry_confidence = float(entry_validation.get("confidence", 0.0))
                risk_atr = float(entry_validation.get("risk_atr", 0.0))
                entry_momentum = float(entry_validation.get("momentum", 0.0))
                entry_alignment = float(entry_validation.get("alignment", 0.0))

                lt_conf = check_trendline_confluence(df_m1, pb_high, pb_low, "CALL", atr_val)
                lt_confluence = float(lt_conf.get("confluence", 0.0))
                has_lt = lt_conf.get("has_trendline", False)

                score = 0.40  # Base aumentada
                impulso_score = min(0.12, (size_A / atr_val - IMPULSO_MIN_ATR) * 0.06)
                score += impulso_score
                eff_score = min(0.15, max(0, (eff_A - MIN_EFF_A) * 0.35))
                score += eff_score

                if 0.30 <= retr <= 0.50:
                    retr_score = 0.12
                elif 0.25 <= retr <= 0.60:
                    retr_score = 0.05
                else:
                    retr_score = max(-0.08, -(abs(retr - 0.40) * 0.20))
                score += retr_score

                if 2 <= pb_len <= 4:
                    pb_score = 0.06
                elif pb_len == 1 or pb_len == 5:
                    pb_score = 0.02
                else:
                    pb_score = -0.02
                score += pb_score

                if flips_frac > 0.50:
                    chop_penalty = min(0.20, (flips_frac - 0.50) * 0.50)
                    score -= chop_penalty

                # Contexto de mercado - MUITO IMPORTANTE
                if market_quality > 0.65:
                    ctx_score = 0.22
                elif market_quality > 0.55:
                    ctx_score = 0.12
                elif market_quality > 0.45:
                    ctx_score = 0.0
                else:
                    ctx_score = -0.30  # Penalidade SEVERA para contexto ruim
                score += ctx_score

                trend_score = min(0.08, trend_strength * 0.12)
                score += trend_score

                if entry_confidence > 0.65:
                    entry_score = 0.25
                elif entry_confidence > 0.55:
                    entry_score = 0.15
                elif entry_confidence > 0.48:
                    entry_score = 0.08
                else:
                    entry_score = -0.05
                score += entry_score

                momentum_score = min(0.10, entry_momentum * 0.08)
                score += momentum_score

                if entry_alignment >= 0.67:
                    align_score = 0.08
                elif entry_alignment >= 0.34:
                    align_score = 0.03
                else:
                    align_score = -0.03
                score += align_score

                if has_lt and lt_confluence > 0.8:
                    score += 0.25
                elif has_lt and lt_confluence > 0.5:
                    score += 0.15
                elif has_lt and lt_confluence > 0.2:
                    score += 0.05

                if risk_atr > 1.3:
                    risk_penalty = 0.10
                elif risk_atr > 1.0:
                    risk_penalty = 0.05
                elif risk_atr < 0.30:
                    risk_penalty = 0.05
                else:
                    risk_penalty = 0.0
                score -= risk_penalty

                perfect_count = 0
                if market_quality > 0.60: perfect_count += 1
                if eff_A > 0.70: perfect_count += 1
                if 0.30 <= retr <= 0.50: perfect_count += 1
                if entry_confidence > 0.65: perfect_count += 1
                if entry_alignment >= 0.67: perfect_count += 1
                if lt_confluence > 0.8: perfect_count += 1

                if perfect_count >= 5:
                    confluence_bonus = 0.20
                elif perfect_count >= 4:
                    confluence_bonus = 0.12
                elif perfect_count >= 3:
                    confluence_bonus = 0.05
                else:
                    confluence_bonus = 0.0

                score += confluence_bonus
                score = float(max(0.0, min(0.95, score)))

                # Filtros RIGOROSOS para evitar LOSS
                if score < 0.55:
                    continue
                if market_quality < 0.42:  # Bloquear contexto ruim
                    continue
                if entry_confidence < 0.45:
                    continue

                setup = {
                    "trade": True, "dir": "CALL", "score": score,
                    "pb_len": pb_len, "retr": float(retr),
                    "A_atr": float(size_A / max(atr_val, 1e-9)),
                    "effA": float(eff_A),
                    "flips": float(flips_frac),
                    "comp": float(comp),
                    "late": float(late_ext),
                    "distBreak": float(dist),
                    "market_quality": float(market_quality),
                    "context": str(context.get("context", "?")),
                    "confluence_bonus": float(confluence_bonus),
                    "trend_strength": float(trend_strength),
                    "trend_reason": str(trend_validation.get("reason", "?")),
                    "entry_confidence": float(entry_confidence),
                    "entry_momentum": float(entry_momentum),
                    "entry_alignment": float(entry_alignment),
                    "risk_atr": float(risk_atr),
                    "lt_confluence": float(lt_confluence),
                    "has_lt": has_lt,
                    "reasons": [
                        "pernadaB_CALL",
                        f"A={size_A/atr_val:.2f}ATR", f"retr={retr:.2f}", f"pb={pb_len}",
                        f"effA={eff_A:.2f}",
                        f"ctx={context.get('context','?')}({market_quality:.2f})",
                        f"entry_conf={entry_confidence:.2f}",
                        f"LTA={lt_confluence:.2f}" if has_lt else "sem_LTA"
                    ]
                }
            else:  # PUT
                if not (c1 < pb_high - BREAK_MARGIN_ATR * atr_val):
                    continue
                if q["lower_frac"] > MAX_WICK_AGAINST:
                    continue

                dist = (pb_high - c1) / max(atr_val, 1e-9)
                if dist > MAX_BREAK_DISTANCE_ATR:
                    continue

                entry_validation = validate_entry_quality(df_m1, atr_val, "PUT", c1, pb_high, pb_low)
                if not entry_validation.get("valid", False):
                    continue

                entry_confidence = float(entry_validation.get("confidence", 0.0))
                risk_atr = float(entry_validation.get("risk_atr", 0.0))
                entry_momentum = float(entry_validation.get("momentum", 0.0))
                entry_alignment = float(entry_validation.get("alignment", 0.0))

                lt_conf = check_trendline_confluence(df_m1, pb_high, pb_low, "PUT", atr_val)
                lt_confluence = float(lt_conf.get("confluence", 0.0))
                has_lt = lt_conf.get("has_trendline", False)

                score = 0.40  # Base aumentada
                impulso_score = min(0.12, (size_A / atr_val - IMPULSO_MIN_ATR) * 0.06)
                score += impulso_score
                eff_score = min(0.15, max(0, (eff_A - MIN_EFF_A) * 0.35))
                score += eff_score

                if 0.30 <= retr <= 0.50:
                    retr_score = 0.12
                elif 0.25 <= retr <= 0.60:
                    retr_score = 0.05
                else:
                    retr_score = max(-0.08, -(abs(retr - 0.40) * 0.20))
                score += retr_score

                if 2 <= pb_len <= 4:
                    pb_score = 0.06
                elif pb_len == 1 or pb_len == 5:
                    pb_score = 0.02
                else:
                    pb_score = -0.02
                score += pb_score

                if flips_frac > 0.50:
                    chop_penalty = min(0.20, (flips_frac - 0.50) * 0.50)
                    score -= chop_penalty

                # Contexto de mercado - MUITO IMPORTANTE
                if market_quality > 0.65:
                    ctx_score = 0.22
                elif market_quality > 0.55:
                    ctx_score = 0.12
                elif market_quality > 0.45:
                    ctx_score = 0.0
                else:
                    ctx_score = -0.30  # Penalidade SEVERA para contexto ruim
                score += ctx_score

                trend_score = min(0.08, trend_strength * 0.12)
                score += trend_score

                if entry_confidence > 0.65:
                    entry_score = 0.25
                elif entry_confidence > 0.55:
                    entry_score = 0.15
                elif entry_confidence > 0.48:
                    entry_score = 0.08
                else:
                    entry_score = -0.05
                score += entry_score

                momentum_score = min(0.10, entry_momentum * 0.08)
                score += momentum_score

                if entry_alignment >= 0.67:
                    align_score = 0.08
                elif entry_alignment >= 0.34:
                    align_score = 0.03
                else:
                    align_score = -0.03
                score += align_score

                if has_lt and lt_confluence > 0.8:
                    score += 0.25
                elif has_lt and lt_confluence > 0.5:
                    score += 0.15
                elif has_lt and lt_confluence > 0.2:
                    score += 0.05

                if risk_atr > 1.3:
                    risk_penalty = 0.10
                elif risk_atr > 1.0:
                    risk_penalty = 0.05
                elif risk_atr < 0.30:
                    risk_penalty = 0.05
                else:
                    risk_penalty = 0.0
                score -= risk_penalty

                perfect_count = 0
                if market_quality > 0.60: perfect_count += 1
                if eff_A > 0.70: perfect_count += 1
                if 0.30 <= retr <= 0.50: perfect_count += 1
                if entry_confidence > 0.65: perfect_count += 1
                if entry_alignment >= 0.67: perfect_count += 1
                if lt_confluence > 0.8: perfect_count += 1

                if perfect_count >= 5:
                    confluence_bonus = 0.20
                elif perfect_count >= 4:
                    confluence_bonus = 0.12
                elif perfect_count >= 3:
                    confluence_bonus = 0.05
                else:
                    confluence_bonus = 0.0

                score += confluence_bonus
                score = float(max(0.0, min(0.95, score)))

                # Filtros RIGOROSOS para evitar LOSS
                if score < 0.55:
                    continue
                if market_quality < 0.42:  # Bloquear contexto ruim
                    continue
                if entry_confidence < 0.45:
                    continue

                setup = {
                    "trade": True, "dir": "PUT", "score": score,
                    "pb_len": pb_len, "retr": float(retr),
                    "A_atr": float(size_A / max(atr_val, 1e-9)),
                    "effA": float(eff_A),
                    "flips": float(flips_frac),
                    "comp": float(comp),
                    "late": float(late_ext),
                    "distBreak": float(dist),
                    "market_quality": float(market_quality),
                    "context": str(context.get("context", "?")),
                    "confluence_bonus": float(confluence_bonus),
                    "trend_strength": float(trend_strength),
                    "trend_reason": str(trend_validation.get("reason", "?")),
                    "entry_confidence": float(entry_confidence),
                    "entry_momentum": float(entry_momentum),
                    "entry_alignment": float(entry_alignment),
                    "risk_atr": float(risk_atr),
                    "lt_confluence": float(lt_confluence),
                    "has_lt": has_lt,
                    "reasons": [
                        "pernadaB_PUT",
                        f"A={size_A/atr_val:.2f}ATR", f"retr={retr:.2f}", f"pb={pb_len}",
                        f"effA={eff_A:.2f}",
                        f"ctx={context.get('context','?')}({market_quality:.2f})",
                        f"entry_conf={entry_confidence:.2f}",
                        f"LTB={lt_confluence:.2f}" if has_lt else "sem_LTB"
                    ]
                }

            if best is None or setup["score"] > best["score"]:
                best = setup

    if best is None:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["sem_pernadaB_valida"]}

    block_final = sr_block_directional_multi(df_m1, atr_val, best["dir"])
    if block_final:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": [block_final]}

    return best

# ===================== ESCOLHER MELHOR SETUP =====================
def escolher_melhor_setup(bx: Bullex, ativos: List[str]):
    best_trade = None
    best_any = None
    
    log.info(paint(f"Analisando {len(ativos)} ativos...", C.B))

    for a in ativos:
        if a in cooldown and (time.time() - cooldown[a]) < COOLDOWN_ATIVO:
            continue
        if a in cooldown_spike and (time.time() - cooldown_spike[a]) < (SPIKE_COOLDOWN_MIN * 60):
            continue
        # Cooldown especial após LOSS no ativo
        if a in cooldown_loss and (time.time() - cooldown_loss[a]) < COOLDOWN_LOSS_ATIVO:
            continue
        # Bloquear ativo com muitos losses consecutivos
        if consecutive_losses.get(a, 0) >= MAX_CONSECUTIVE_LOSS:
            continue

        df = get_candles_df(bx, a, TF_M1, N_M1, end_ts=end_ts_closed(TF_M1))
        if df is None:
            continue

        atr_val = atr(df, 14)
        last_closed = df.iloc[-1]

        if is_spike_wicky(last_closed, atr_val):
            cooldown_spike[a] = time.time()
            continue

        setup = dom_forex_signal(df, atr_val)

        sc_any = float(setup.get("score", 0.0))
        cand_any = (sc_any, a, setup, float(atr_val))
        if best_any is None or cand_any[0] > best_any[0]:
            best_any = cand_any

        if setup.get("trade"):
            cand_trade = (float(setup["score"]), a, setup, float(atr_val))
            if best_trade is None or cand_trade[0] > best_trade[0]:
                best_trade = cand_trade
                log.info(paint(f"  🎯 {a}: {setup['dir']} score={setup['score']:.2f}", dir_color(setup['dir'])))

    return best_trade, best_any

# ===================== ORDEM =====================
def enviar_ordem(bx: Bullex, ativo: str, direcao: str, stake: float) -> Optional[Tuple[str, int]]:
    d = "call" if direcao == "CALL" else "put"
    valor = float(max(VALOR_MINIMO, stake))

    # TURBO
    try:
        ok, op_id = safe_call(bx, bx.buy, valor, ativo, d, int(EXP_FIXA))
        if ok and op_id:
            return ("turbo", int(op_id))
        log.warning(paint(f"[ORDEM-FAIL] TURBO ok={ok} op_id={op_id}", C.Y))
    except Exception as e:
        log.warning(paint(f"[ORDEM-EXC] TURBO {e}", C.Y))

    # DIGITAL
    try:
        ok, op_id = safe_call(bx, bx.buy_digital_spot, ativo, valor, d, int(EXP_FIXA))
        if ok and op_id:
            return ("digital", int(op_id))
        log.warning(paint(f"[ORDEM-FAIL] DIGITAL ok={ok} op_id={op_id}", C.Y))
    except Exception as e:
        log.warning(paint(f"[ORDEM-EXC] DIGITAL {e}", C.Y))

    return None

def wait_result(bx: Bullex, op_type: str, op_id: int) -> float:
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

# ===================== MAIN =====================
def main():
    bx: Optional[Bullex] = None
    bx = ensure_connected(bx)

    log.info("=" * 60)
    log.info("🚀 WS_AUTO_AI_BULLEX — Pernada B (M1) + ENSEMBLE IA")
    log.info("=" * 60)
    log.info("✅ Analisa TODOS os ativos de uma vez")
    log.info("✅ Entra quando aparecer sinal confirmado")
    log.info("✅ IA ENSEMBLE: Bayesiano + LightGBM (Gradient Boosting)")
    log.info("=" * 60)
    
    # Mostrar modo da IA
    if IA_MODE == "learning":
        log.info(paint("🧠 MODO: LEARNING - IA tem CONTROLE TOTAL, filtros relaxados", C.G))
        log.info(paint("   → Score mínimo muito baixo, IA decide quando entrar", C.B))
        log.info(paint("   → Contexto ruim NÃO bloqueia, IA aprende sozinha", C.B))
    else:
        log.info(paint("🔒 MODO: STRICT - IA + filtros rigorosos", C.Y))
        log.info(paint("   → Score mínimo alto, filtros conservadores", C.B))

    stats = _safe_load_json(AI_STATS_FILE) if IA_ON else {"meta": {"total": 0}, "arms": {}}
    if IA_ON:
        log.info(f"[BAYES] ON | file={AI_STATS_FILE} | min_samples={AI_MIN_SAMPLES} | min_prob={AI_MIN_PROB:.2f}")
    
    # Carregar LightGBM
    if LGBM_ON:
        lgbm_load_data()
        lgbm_load_model()
        log.info(paint(f"[LGBM] ON | mode={ENSEMBLE_MODE} | min_prob={LGBM_MIN_PROB:.2f} | samples={len(lgbm_data)}", C.B))
        if len(lgbm_data) >= LGBM_MIN_SAMPLES and lgbm_model is None:
            lgbm_train()
    else:
        log.info("[LGBM] OFF - usando apenas Bayesiano")
    
    # Modo de operação após LOSS
    if RETRAIN_ON_LOSS:
        log.info(paint(f"🔄 MODO: RETRAIN & CONTINUE - Após LOSS pausa {PAUSE_AFTER_LOSS_SECONDS}s, retreina e continua", C.G))
    else:
        log.info(paint("⏹️ MODO: STOP após LOSS - Bot para e precisa reiniciar manualmente", C.Y))

    try:
        saldo_inicial = float(bx.get_balance())
        log.info(paint(f"💰 SALDO INICIAL: {saldo_inicial:.2f} | META: {META_LUCRO_PERCENT:.1f}% (={saldo_inicial * META_LUCRO_PERCENT / 100:.2f})", C.G))
        if USE_DYNAMIC_STAKE:
            log.info(paint(f"📊 GESTÃO: {PERCENT_BANCA:.1f}% da banca por operação (stake dinâmico)", C.B))
        else:
            log.info(paint(f"📊 GESTÃO: Stake fixo de {STAKE_FIXA:.2f}", C.B))
    except Exception:
        saldo_inicial = 1000.0

    total = 0
    wins = 0

    while True:
        bx = ensure_connected(bx)

        try:
            saldo_atual = float(bx.get_balance())
            deve_parar, lucro_percent = verificar_meta_atingida(saldo_inicial, saldo_atual)
            if deve_parar:
                lucro_abs = saldo_atual - saldo_inicial
                if lucro_percent >= META_LUCRO_PERCENT:
                    log.info(paint(f"🎯 META ATINGIDA! Lucro: {lucro_abs:.2f} ({lucro_percent:.2f}%) | Bot desligando...", C.G))
                else:
                    log.info(paint(f"🛑 STOP LOSS! Perda: {lucro_abs:.2f} ({lucro_percent:.2f}%) | Bot desligando...", C.R))
                break
        except Exception as e:
            log.warning(f"Erro ao verificar meta: {e}")

        ativos = obter_top_ativos_otc(bx)
        if not ativos:
            log.warning("Sem ativos com payout mínimo. Tentando em 10s...")
            time.sleep(10)
            continue

        wait_until_minus(TF_M1, DECIDIR_ANTES_FECHAR_SEC)

        best_trade, best_any = escolher_melhor_setup(bx, ativos)

        if not best_trade:
            if best_any:
                sc, at, st, _av = best_any
                log.info(paint(
                    f"[SKIP] nenhum setup passou | melhor={at} score={sc:.2f} | {','.join(st.get('reasons', []))}",
                    C.Y
                ))
                cooldown[at] = time.time()
            else:
                log.info(paint("[SKIP] nenhum ativo analisável no minuto", C.Y))

            wait_for_next_open(TF_M1)
            continue

        score, ativo, setup, atr_val = best_trade
        score = float(score)

        if score < GATE_SOFT_SCORE:
            log.info(paint(
                f"[SKIP] {ativo} | score={score:.2f} | {','.join(setup.get('reasons', []))}",
                C.Y
            ))
            wait_for_next_open(TF_M1)
            cooldown[ativo] = time.time()
            continue

        if score < GATE_MIN_SCORE:
            log.info(paint(
                f"[SOFT-SKIP] {ativo} | score={score:.2f} | {','.join(setup.get('reasons', []))}",
                C.B
            ))
            wait_for_next_open(TF_M1)
            cooldown[ativo] = time.time()
            continue

        final_dir = str(setup["dir"])
        log.info(paint(
            f"[SINAL-HARD] {ativo} -> {final_dir} | score={score:.2f} | ATR={atr_val:.6f} | {','.join(setup.get('reasons', []))}",
            dir_color(final_dir)
        ))

        if IA_ON:
            # ENSEMBLE: Combina Bayesiano + LightGBM
            ens = ensemble_predict(ativo, setup, stats)
            bayes_prob = float(ens["bayes_prob"])
            lgbm_prob = float(ens["lgbm_prob"])
            ensemble_prob = float(ens["ensemble_prob"])
            should_trade = bool(ens["should_trade"])
            ens_reason = str(ens["reason"])
            n_arm = int(ens.get("n_arm", 0))
            
            # Log do ensemble
            if LGBM_ON and lgbm_model is not None:
                log.info(paint(
                    f"[ENSEMBLE] {ativo} {final_dir} | Bayes={bayes_prob:.2f} | LGBM={lgbm_prob:.2f} | Ens={ensemble_prob:.2f} | {ens_reason}",
                    C.B
                ))
            else:
                log.info(paint(
                    f"[BAYES] {ativo} {final_dir} | prob={bayes_prob:.2f} (n={n_arm}) | {ens_reason}",
                    C.B
                ))
            
            # Decisão do ensemble
            if not should_trade:
                log.info(paint(f"[IA-SKIP] {ativo} {final_dir} | {ens_reason}", C.Y))
                wait_for_next_open(TF_M1)
                cooldown[ativo] = time.time()
                continue
            
            # ══════════ GATEs PROFISSIONAIS ══════════
            ctx_val = float(setup.get("market_quality", 0.40))
            entry_conf_val = float(setup.get("entry_confidence", 0.50))
            sr_prox_gate = float(setup.get("sr_proximity", 0.0))
            sr_tq_gate = int(setup.get("sr_touches", 0))
            sr_forte = sr_prox_gate > 0.60 and sr_tq_gate >= 4
            sr_basico = sr_tq_gate >= 3 and float(setup.get("sr_weight", 0.0)) >= 4.0
            confluence_count = int(setup.get("confluence_bonus", 0) > 0.04) + int(setup.get("has_lt", False)) + int(sr_prox_gate > 0.30)
            lt_conf = float(setup.get("lt_confluence", 0.0))
            
            # GATE 1: Contexto ruim BLOQUEIA a menos que ensemble alto + SR forte
            if ctx_val < 0.40 and not ((sr_forte or sr_basico) and ensemble_prob >= 0.58):
                log.info(paint(f"[CTX-GATE] {ativo} {final_dir} | ctx_ruim={ctx_val:.2f} | ens={ensemble_prob:.2f}", C.Y))
                wait_for_next_open(TF_M1)
                cooldown[ativo] = time.time()
                continue
            
            # GATE 2: Contexto mediano + sem zona confirmada = precisa score alto
            score_gate2 = float(setup.get("score", 0.0))
            has_lt_g2 = bool(setup.get("has_lt", False))
            if ctx_val < 0.50 and not sr_forte and not sr_basico and not has_lt_g2 and score_gate2 < 0.55:
                log.info(paint(f"[CTX-GATE] {ativo} {final_dir} | ctx_med+sem_zona | ctx={ctx_val:.2f},sc={score_gate2:.2f}", C.Y))
                wait_for_next_open(TF_M1)
                cooldown[ativo] = time.time()
                continue
            
            # GATE 3: Trendline fraca sem S/R forte = não opera
            if lt_conf < 0.5 and not setup.get("has_lt", False) and sr_prox_gate < 0.40:
                log.info(paint(f"[TREND-GATE] {ativo} {final_dir} | sem_tendência_forte+sem_SR | lt={lt_conf:.2f},sr={sr_prox_gate:.2f}", C.Y))
                wait_for_next_open(TF_M1)
                cooldown[ativo] = time.time()
                continue
            
            # GATE 4: Foco na ZONA - só bloqueia se TUDO é fraco
            score_val_gate = float(setup.get("score", 0.0))
            sr_tq_g4 = int(setup.get("sr_touches", 0))
            sr_w_g4 = float(setup.get("sr_weight", 0.0))
            lt_pts_g4 = int(setup.get("lt_points", setup.get("pb_len", 0)))
            has_lt_g4 = bool(setup.get("has_lt", False))
            zona_forte = (sr_tq_g4 >= 3 and sr_w_g4 >= 4.0) or (has_lt_g4 and lt_pts_g4 >= 3)
            setup_ok = score_val_gate >= 0.45 and ctx_val >= 0.50
            setup_forte = score_val_gate >= 0.55 and ctx_val >= 0.55 and confluence_count >= 2
            ens_gate_thr = 0.52 if not LGBM_ON else 0.60
            if entry_conf_val < 0.30 and not zona_forte and not setup_forte and ensemble_prob < ens_gate_thr:
                log.info(paint(f"[CONF-GATE] {ativo} {final_dir} | tudo_fraco | conf={entry_conf_val:.2f},zona={zona_forte},sc={score_val_gate:.2f},ens={ensemble_prob:.2f}", C.Y))
                wait_for_next_open(TF_M1)
                cooldown[ativo] = time.time()
                continue

        wait_for_next_open(TF_M1)

        stake = calcular_stake_dinamico(bx, STAKE_FIXA)
        log.info(paint(f"[{ativo}] 💵 Stake calculado: {stake:.2f}", C.B))

        op = enviar_ordem(bx, ativo, final_dir, stake)

        if not op:
            log.error(paint(f"[{ativo}] ❌ falhou enviar ordem (TURBO/DIGITAL).", C.R))
            cooldown[ativo] = time.time()
            continue

        op_type, op_id = op
        log.info(paint(
            f"[{ativo}] ✅ ORDEM ENVIADA {final_dir} exp={EXP_FIXA}m ({op_type}) | stake={stake:.2f}",
            dir_color(final_dir)
        ))

        res = wait_result(bx, op_type, op_id)

        total += 1
        global global_consecutive_losses
        
        should_pause_and_continue = False  # Flag para pausar e continuar após LOSS
        
        if res > 0:
            wins += 1
            log.info(paint(f"[{ativo}] ✅ WIN {res:.2f}$", C.G))
            # Reset counters após WIN
            consecutive_losses[ativo] = 0
            global_consecutive_losses = 0
        elif res < 0:
            log.info(paint(f"[{ativo}] ❌ LOSS {res:.2f}$", C.R))
            # Incrementar contadores de LOSS
            consecutive_losses[ativo] = consecutive_losses.get(ativo, 0) + 1
            global_consecutive_losses += 1
            # Aplicar cooldown especial após LOSS
            cooldown_loss[ativo] = time.time()
            
            # RETREINO SEVERO: aplicar penalidade extra no padrão
            if IA_ON:
                ai_retrain_on_loss(ativo, setup, stats)
                log.warning(paint(f"[RETRAIN] Padrão penalizado com {RETRAIN_PENALTY:.0%} - IA aprendendo com erro", C.Y))
            
            # RETRAIN_ON_LOSS: pausar, retreinar e continuar automaticamente
            if RETRAIN_ON_LOSS:
                should_pause_and_continue = True
                log.warning(paint(f"[RETRAIN] IA retreinada - pausando {PAUSE_AFTER_LOSS_SECONDS}s antes de continuar", C.Y))
            elif global_consecutive_losses >= MAX_CONSECUTIVE_LOSS:
                should_pause_and_continue = True
                log.warning(paint(f"[PAUSE] {global_consecutive_losses} losses consecutivos - pausando {PAUSE_AFTER_LOSS_SECONDS}s", C.Y))
        else:
            log.info(paint(f"[{ativo}] ⚪ EMPATE {res:.2f}$", C.B))

        if IA_ON:
            ai_update(ativo, setup, res, stats)
            _safe_save_json(AI_STATS_FILE, stats)
        
        # Adiciona amostra ao LightGBM para aprendizado
        if LGBM_ON:
            lgbm_add_sample(setup, res)
        
        # PAUSAR e CONTINUAR automaticamente após LOSS (não para o bot)
        if should_pause_and_continue:
            log.info(paint("=" * 60, C.Y))
            log.info(paint("⏸️ PAUSANDO APÓS LOSS - IA RETREINADA (BAYES + LGBM)", C.Y))
            log.info(paint(f"📊 RESUMO: trades={total} wins={wins} acc={(wins/max(1,total))*100:.1f}%", C.Y))
            log.info(paint(f"⏳ Esperando {PAUSE_AFTER_LOSS_SECONDS}s antes de continuar...", C.Y))
            log.info(paint("=" * 60, C.Y))
            time.sleep(PAUSE_AFTER_LOSS_SECONDS)
            global_consecutive_losses = 0  # Reset após pausa
            log.info(paint("\n▶️ RETOMANDO OPERAÇÕES - IA atualizada pronta!\n", C.G))
            continue  # Continua o loop principal

        acc = (wins / max(1, total)) * 100.0

        try:
            saldo_atual = float(bx.get_balance())
            lucro_atual = saldo_atual - saldo_inicial
            lucro_percent_atual = (lucro_atual / saldo_inicial) * 100.0
            falta_meta = (saldo_inicial * META_LUCRO_PERCENT / 100.0) - lucro_atual

            if lucro_percent_atual >= 0:
                log.info(paint(f"📊 GLOBAL: trades={total} wins={wins} acc={acc:.2f}%", C.G))
                log.info(paint(f"💰 SALDO: {saldo_atual:.2f} | LUCRO: +{lucro_atual:.2f} ({lucro_percent_atual:.2f}%) | FALTA: {falta_meta:.2f} para meta\n", C.G))
            else:
                log.info(paint(f"📊 GLOBAL: trades={total} wins={wins} acc={acc:.2f}%", C.Y))
                log.info(paint(f"💰 SALDO: {saldo_atual:.2f} | PERDA: {lucro_atual:.2f} ({lucro_percent_atual:.2f}%)\n", C.Y))
        except Exception:
            log.info(f"📊 GLOBAL: trades={total} wins={wins} acc={acc:.2f}%\n")

        cooldown[ativo] = time.time()

if __name__ == "__main__":
    main()
