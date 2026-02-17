# -*- coding: utf-8 -*-
"""
WS_AUTO_AI_OPTIMIZED — Versão otimizada com IA mais inteligente
✅ Filtros mais seletivos e inteligentes
✅ IA aprende padrões ruins mais rápido
✅ Score mais rigoroso com confluências
✅ Validação multinível de entrada
"""

import os
import time
import math
import json
import logging
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from iqoptionapi.stable_api import IQ_Option

try:
    import talib
    TALIB_AVAILABLE = True
except Exception:
    talib = None
    TALIB_AVAILABLE = False

if not TALIB_AVAILABLE:
    raise RuntimeError("TA-Lib obrigatorio: instale TA-Lib para executar o WS_AUTO_AI_OPTIMIZED.")

# ===================== CONFIG =====================
EMAIL = os.getenv("IQ_EMAIL", "") or "wstrader@wstrader.onmicrosoft.com"
SENHA = os.getenv("IQ_PASS", "") or "P152030@w"
CONTA = os.getenv("IQ_CONTA", "REAL")

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
PERCENT_BANCA = float(os.getenv("WS_PERCENT_BANCA", "1.0"))
META_LUCRO_PERCENT = float(os.getenv("WS_META_LUCRO", "1.5"))
STOP_LOSS_PERCENT = float(os.getenv("WS_STOP_LOSS", "3.0"))
USE_DYNAMIC_STAKE = (os.getenv("WS_DYNAMIC_STAKE", "1").strip() == "1")

COOLDOWN_ATIVO = int(os.getenv("WS_COOLDOWN_ATIVO", "20"))

# ===================== IA (ONLINE) - OTIMIZADA =====================
IA_ON = (os.getenv("WS_AI_ON", "1").strip() == "1")
AI_STATS_FILE = os.getenv("WS_AI_FILE", "ws_ai_stats_m1_optimized.json")
AI_MIN_SAMPLES = int(os.getenv("WS_AI_MIN_SAMPLES", "10"))  # REDUZIDO: aprende mais rápido
AI_MIN_PROB = float(os.getenv("WS_AI_MIN_PROB", "0.50"))  # AUMENTADO: mais rigoroso
AI_MIN_WINRATE = float(os.getenv("WS_AI_MIN_WINRATE", "0.48"))  # AUMENTADO: bloqueia abaixo de 48%
AI_CONF_MIN = float(os.getenv("WS_AI_CONF_MIN", "0.55"))  # AUMENTADO: mais confiança necessária
AI_CONFIRM_MIN = float(os.getenv("WS_AI_CONFIRM_MIN", "0.55"))  # CONFIRMA ENTRADA

# ⭐ NOVO: Sistema de bloqueio progressivo
AI_FAST_BLOCK_LOSSES = int(os.getenv("WS_FAST_BLOCK", "3"))  # Bloqueia após 3 losses consecutivos

# ===================== PERNADA B (MAIS SELETIVO) =====================
IMPULSO_MIN_ATR = float(os.getenv("WS_IMPULSO_MIN_ATR", "0.65"))  # AUMENTADO: impulso mais forte
IMPULSO_JANELA_MIN = int(os.getenv("WS_IMP_JMIN", "3"))
IMPULSO_JANELA_MAX = int(os.getenv("WS_IMP_JMAX", "12"))  # REDUZIDO: impulsos mais recentes

PULLBACK_MIN = int(os.getenv("WS_PB_MIN", "1"))
PULLBACK_MAX = int(os.getenv("WS_PB_MAX", "5"))  # REDUZIDO: pullbacks mais curtos

RETR_MIN = float(os.getenv("WS_RETR_MIN", "0.20"))  # AUMENTADO: retração mínima maior
RETR_MAX = float(os.getenv("WS_RETR_MAX", "0.75"))  # REDUZIDO: não deixa retrair demais

BREAK_MARGIN_ATR = float(os.getenv("WS_BREAK_MARGIN_ATR", "0.02"))  # margem maior
MAX_BREAK_DISTANCE_ATR = float(os.getenv("WS_MAX_BREAK_DIST_ATR", "0.35"))  # REDUZIDO: quebra mais próxima

# ===================== ANTI-LATERAL (MAIS RIGOROSO) =====================
MIN_EFF_A = float(os.getenv("WS_MIN_EFF_A", "0.50"))  # AUMENTADO: eficiência mínima maior

CHOP_LOOKBACK = int(os.getenv("WS_CHOP_LB", "28"))
MAX_COLOR_FLIPS_FRAC = float(os.getenv("WS_MAX_FLIPS", "0.65"))  # REDUZIDO: menos choppiness permitido
MIN_NET_GROSS_EFF = float(os.getenv("WS_MIN_NETGROSS", "0.15"))  # AUMENTADO: mais direcionalidade

COMP_LOOKBACK = int(os.getenv("WS_COMP_LB", "18"))
MIN_RANGE_ATR = float(os.getenv("WS_MIN_RANGE_ATR", "0.70"))  # AUMENTADO: range mínimo maior

LATE_LOOKBACK = int(os.getenv("WS_LATE_LB", "18"))
MAX_LATE_EXT_ATR = float(os.getenv("WS_MAX_LATE_EXT_ATR", "9.0"))  # REDUZIDO: menos extensão tardia

# ===================== QUALIDADE DO GATILHO (MAIS RIGOROSO) =====================
MIN_BODY_FRAC_BREAK = float(os.getenv("WS_MIN_BODY_FRAC", "0.20"))  # AUMENTADO: corpo mínimo maior
MAX_WICK_AGAINST = float(os.getenv("WS_MAX_WICK_AGAINST", "0.60"))  # REDUZIDO: menos pavio contra

# ===================== SCORE (MAIS RIGOROSO) =====================
GATE_MIN_SCORE = float(os.getenv("WS_GATE_MIN", "0.60"))  # AUMENTADO: score mínimo maior
GATE_SOFT_SCORE = float(os.getenv("WS_GATE_SOFT", "0.55"))  # AUMENTADO: soft score maior

# ===================== ANTI-SPIKE =====================
SPIKE_RANGE_ATR = float(os.getenv("WS_SPIKE_RANGE_ATR", "1.35"))
SPIKE_WICK_FRAC = float(os.getenv("WS_SPIKE_WICK_FRAC", "0.62"))
SPIKE_COOLDOWN_MIN = int(os.getenv("WS_SPIKE_COOLDOWN_MIN", "6"))

# ===================== FILTRO S/R FORTE =====================
SR_LOOKBACK = int(os.getenv("WS_SR_LOOKBACK", "220"))
SR_CLUSTER_ATR = float(os.getenv("WS_SR_CLUSTER_ATR", "0.45"))
SR_MIN_TOUCHES_STRONG = int(os.getenv("WS_SR_MIN_TOUCHES", "3"))
SR_TOP_LEVELS = int(os.getenv("WS_SR_TOP_LEVELS", "6"))
SR_CHECK_NEAR = int(os.getenv("WS_SR_CHECK_NEAR", "2"))
SR_BLOCK_DIST_ATR = float(os.getenv("WS_SR_BLOCK_ATR", "0.65"))
SR_HARD_BLOCK_ATR = float(os.getenv("WS_SR_HARD_BLOCK_ATR", "0.70"))

# ===================== PADROES FORTES =====================
PATTERN_STRONG_MIN = float(os.getenv("WS_PATTERN_STRONG_MIN", "0.80"))

# ===================== TENDENCIA =====================
TREND_MIN_STRENGTH = float(os.getenv("WS_TREND_MIN", "0.30"))

# ===================== LOG =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [WS_AUTO_AI] %(message)s"
)
log = logging.getLogger("WS_AUTO_AI")

class C:
    G = "\033[92m"
    R = "\033[91m"
    Y = "\033[93m"
    B = "\033[94m"
    M = "\033[95m"  # NOVO: Magenta para alertas IA
    Z = "\033[0m"

def paint(s: str, color: str) -> str:
    return f"{color}{s}{C.Z}"

def dir_color(direction: str) -> str:
    return C.G if direction == "CALL" else (C.R if direction == "PUT" else C.Y)

_cache_ativos: List[str] = []
_cache_ativos_ts: float = 0.0

cooldown: Dict[str, float] = {}
cooldown_spike: Dict[str, float] = {}

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
    return now - (now % tf) - 1

# ===================== PATCH WEBSOCKET =====================
def patch_websocket_on_close():
    try:
        from iqoptionapi.ws.client import WebsocketClient
        if getattr(WebsocketClient, "__WS_PATCHED__", False):
            return
        old = WebsocketClient.on_close

        def on_close_compat(self, *args, **kwargs):
            try:
                return old(self)
            except TypeError:
                try:
                    return old(self, *args, **kwargs)
                except Exception:
                    return None
            except Exception:
                return None

        WebsocketClient.on_close = on_close_compat
        WebsocketClient.__WS_PATCHED__ = True
        log.info("Patch aplicado: WebsocketClient.on_close compatível.")
    except Exception as e:
        log.warning(f"Patch websocket falhou: {e}")

# ===================== IQ OPTION =====================
def conectar_iq() -> IQ_Option:
    if not EMAIL or not SENHA:
        raise RuntimeError("Defina IQ_EMAIL e IQ_PASS nas variáveis de ambiente.")
    patch_websocket_on_close()
    log.info("Conectando à IQ Option...")
    iq = IQ_Option(EMAIL, SENHA)
    iq.connect()

    for _ in range(12):
        if iq.check_connect():
            break
        time.sleep(1.5)

    if not iq.check_connect():
        raise RuntimeError("Falha na conexão com a IQ Option.")

    iq.change_balance(CONTA)
    try:
        log.info(f"Conectado | Saldo: {iq.get_balance():.2f} | Conta: {CONTA}")
    except Exception:
        log.info(f"Conectado | Conta: {CONTA}")

    return iq

def ensure_connected(iq: Optional[IQ_Option]) -> IQ_Option:
    if iq is None:
        return conectar_iq()
    try:
        if iq.check_connect():
            return iq
    except Exception:
        pass

    log.warning(paint("Conexão caiu. Tentando reconectar...", C.Y))
    try:
        iq.connect()
        for _ in range(12):
            if iq.check_connect():
                iq.change_balance(CONTA)
                log.info("Reconectado.")
                return iq
            time.sleep(1.5)
    except Exception:
        pass

    return conectar_iq()

def safe_call(iq: IQ_Option, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        if ("10054" in msg) or ("forçado o cancelamento" in msg) or ("goodbye" in msg) or ("10053" in msg):
            log.error(paint(f"Erro de conexão: {e}", C.R))
            ensure_connected(iq)
            return fn(*args, **kwargs)
        raise

# ===================== CANDLES =====================
def get_candles_df(iq: IQ_Option, ativo: str, timeframe: int, n: int, end_ts: Optional[float] = None) -> Optional[pd.DataFrame]:
    try:
        if end_ts is None:
            end_ts = time.time()

        candles = safe_call(iq, iq.get_candles, ativo, timeframe, n, end_ts)
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
def obter_top_ativos_otc(iq: IQ_Option) -> List[str]:
    global _cache_ativos, _cache_ativos_ts
    now = time.time()
    if _cache_ativos and (now - _cache_ativos_ts) < PAYOUT_REFRESH_SEC:
        return _cache_ativos

    try:
        dados = safe_call(iq, iq.get_all_open_time)
        turbo = dados.get("turbo", {})
    except Exception:
        return []

    abertos = [a for a, info in turbo.items() if info.get("open", False)]
    abertos_otc = [a for a in abertos if "-OTC" in a.upper()]
    if not abertos_otc:
        abertos_otc = abertos

    filtrados = []
    for a in abertos_otc:
        try:
            payout = safe_call(iq, iq.get_digital_payout, a)
            payout = int(payout) if payout is not None else 0
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

# ===================== PADRÕES DE VELAS =====================
_TALIB_STRONG_PATTERNS = [
    ("CDLENGULFING", 1.0),
    ("CDLMORNINGSTAR", 1.0),
    ("CDLEVENINGSTAR", 1.0),
    ("CDL3WHITESOLDIERS", 1.0),
    ("CDL3BLACKCROWS", 1.0),
    ("CDLPIERCING", 0.8),
    ("CDLDARKCLOUDCOVER", 0.8),
    ("CDLHAMMER", 0.7),
    ("CDLSHOOTINGSTAR", 0.7),
    ("CDLHANGINGMAN", 0.6),
]

def _analyze_candle_patterns_talib(df: pd.DataFrame, lookback: int = 5) -> Dict[str, Any]:
    if len(df) < lookback + 5:
        return {"score": 0.0, "direction": "NEUTRAL", "patterns": []}

    recent = df.tail(max(lookback + 5, 12))
    o = recent["open"].to_numpy(float)
    h = recent["high"].to_numpy(float)
    l = recent["low"].to_numpy(float)
    c = recent["close"].to_numpy(float)

    patterns = []
    call_strength = 0.0
    put_strength = 0.0

    for name, weight in _TALIB_STRONG_PATTERNS:
        fn = getattr(talib, name, None) if TALIB_AVAILABLE else None
        if fn is None:
            continue
        try:
            res = fn(o, h, l, c)
        except Exception:
            continue
        if res is None or len(res) == 0:
            continue

        for i in range(-lookback, 0):
            val = int(res[i])
            if val == 0:
                continue
            strength = min(1.0, abs(val) / 100.0) * weight
            signal = "CALL" if val > 0 else "PUT"

            recency = 1.0 + ((lookback + i) / max(1, lookback - 1)) * 0.5
            if signal == "CALL":
                call_strength += strength * recency
            else:
                put_strength += strength * recency

            if i == -1:
                patterns.append({"pattern": name, "signal": signal, "strength": float(strength)})

    total_strength = call_strength + put_strength
    if total_strength < 0.1:
        return {"score": 0.0, "direction": "NEUTRAL", "patterns": patterns}

    if call_strength > put_strength:
        confidence = call_strength / total_strength
        return {
            "score": float(confidence),
            "direction": "CALL",
            "patterns": patterns,
            "call_strength": float(call_strength),
            "put_strength": float(put_strength),
        }

    confidence = put_strength / total_strength
    return {
        "score": float(confidence),
        "direction": "PUT",
        "patterns": patterns,
        "call_strength": float(call_strength),
        "put_strength": float(put_strength),
    }

def analyze_strong_pattern_last(df: pd.DataFrame, min_score: float = PATTERN_STRONG_MIN) -> Dict[str, Any]:
    if len(df) < 2:
        return {"score": 0.0, "direction": "NEUTRAL", "patterns": []}

    recent = df.tail(12)
    o = recent["open"].to_numpy(float)
    h = recent["high"].to_numpy(float)
    l = recent["low"].to_numpy(float)
    c = recent["close"].to_numpy(float)

    best_score = 0.0
    best_dir = "NEUTRAL"
    best_name = None
    patterns = []

    for name, weight in _TALIB_STRONG_PATTERNS:
        fn = getattr(talib, name, None)
        if fn is None:
            continue
        try:
            res = fn(o, h, l, c)
        except Exception:
            continue
        if res is None or len(res) == 0:
            continue

        val = int(res[-1])
        if val == 0:
            continue
        strength = min(1.0, abs(val) / 100.0) * weight
        signal = "CALL" if val > 0 else "PUT"
        patterns.append({"pattern": name, "signal": signal, "strength": float(strength)})

        if strength >= min_score and strength > best_score:
            best_score = float(strength)
            best_dir = signal
            best_name = name

    if best_score < min_score:
        return {"score": 0.0, "direction": "NEUTRAL", "patterns": patterns}

    return {
        "score": float(best_score),
        "direction": best_dir,
        "pattern": best_name,
        "patterns": patterns,
    }

def identify_candle_pattern(row: pd.Series, prev_row: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Identifica padrões clássicos de candlestick.
    Retorna: {pattern: str, signal: str (CALL/PUT/NEUTRAL), strength: float (0-1)}
    """
    o = float(row["open"])
    h = float(row["high"])
    l = float(row["low"])
    c = float(row["close"])
    
    body = abs(c - o)
    total_range = h - l
    
    if total_range < 1e-9:
        return {"pattern": "none", "signal": "NEUTRAL", "strength": 0.0}
    
    body_ratio = body / total_range
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    upper_shadow_ratio = upper_shadow / total_range
    lower_shadow_ratio = lower_shadow / total_range
    
    # DOJI - Indecisão
    if body_ratio < 0.10:
        return {"pattern": "doji", "signal": "NEUTRAL", "strength": 0.3}
    
    # HAMMER - Alta (corpo pequeno, pavio inferior longo)
    if body_ratio < 0.30 and lower_shadow_ratio > 0.60 and upper_shadow_ratio < 0.10:
        if c > o:  # Bullish hammer
            return {"pattern": "hammer_bull", "signal": "CALL", "strength": 0.75}
        else:
            return {"pattern": "hammer_bear", "signal": "CALL", "strength": 0.65}
    
    # SHOOTING STAR - Baixa (corpo pequeno, pavio superior longo)
    if body_ratio < 0.30 and upper_shadow_ratio > 0.60 and lower_shadow_ratio < 0.10:
        if c < o:  # Bearish shooting star
            return {"pattern": "shooting_star_bear", "signal": "PUT", "strength": 0.75}
        else:
            return {"pattern": "shooting_star_bull", "signal": "PUT", "strength": 0.65}
    
    # MARUBOZU - Forte (corpo grande, pouco pavio)
    if body_ratio > 0.85:
        if c > o:
            return {"pattern": "marubozu_bull", "signal": "CALL", "strength": 0.80}
        else:
            return {"pattern": "marubozu_bear", "signal": "PUT", "strength": 0.80}
    
    # SPINNING TOP - Indecisão (corpo pequeno, pavios em ambos os lados)
    if body_ratio < 0.30 and upper_shadow_ratio > 0.30 and lower_shadow_ratio > 0.30:
        return {"pattern": "spinning_top", "signal": "NEUTRAL", "strength": 0.2}
    
    # Padrões de duas velas (se prev_row fornecido)
    if prev_row is not None:
        prev_o = float(prev_row["open"])
        prev_h = float(prev_row["high"])
        prev_l = float(prev_row["low"])
        prev_c = float(prev_row["close"])
        prev_body = abs(prev_c - prev_o)
        
        # ENGULFING BULLISH - Alta forte
        if prev_c < prev_o and c > o:  # vela anterior bearish, atual bullish
            if c > prev_o and o < prev_c and body > prev_body * 0.90:
                return {"pattern": "engulfing_bull", "signal": "CALL", "strength": 0.90}
        
        # ENGULFING BEARISH - Baixa forte
        if prev_c > prev_o and c < o:  # vela anterior bullish, atual bearish
            if c < prev_o and o > prev_c and body > prev_body * 0.90:
                return {"pattern": "engulfing_bear", "signal": "PUT", "strength": 0.90}
        
        # PIERCING LINE - Alta
        if prev_c < prev_o and c > o:  # anterior bearish, atual bullish
            if o < prev_c and c > (prev_o + prev_c) / 2 and c < prev_o:
                return {"pattern": "piercing_line", "signal": "CALL", "strength": 0.70}
        
        # DARK CLOUD COVER - Baixa
        if prev_c > prev_o and c < o:  # anterior bullish, atual bearish
            if o > prev_c and c < (prev_o + prev_c) / 2 and c > prev_o:
                return {"pattern": "dark_cloud", "signal": "PUT", "strength": 0.70}
    
    # Vela comum
    if c > o:
        return {"pattern": "bullish_common", "signal": "CALL", "strength": 0.40}
    elif c < o:
        return {"pattern": "bearish_common", "signal": "PUT", "strength": 0.40}
    else:
        return {"pattern": "none", "signal": "NEUTRAL", "strength": 0.0}

def analyze_candle_patterns_sequence(df: pd.DataFrame, lookback: int = 5) -> Dict[str, Any]:
    """
    Analisa sequência de padrões de velas nas últimas N velas.
    Retorna score agregado e direção predominante.
    """
    return _analyze_candle_patterns_talib(df, lookback)

    if len(df) < lookback:
        return {"score": 0.0, "direction": "NEUTRAL", "patterns": []}
    
    recent = df.tail(lookback)
    patterns = []
    call_strength = 0.0
    put_strength = 0.0
    
    for i in range(len(recent)):
        prev_row = recent.iloc[i-1] if i > 0 else None
        curr_row = recent.iloc[i]
        
        pattern_data = identify_candle_pattern(curr_row, prev_row)
        patterns.append(pattern_data)
        
        signal = pattern_data["signal"]
        strength = pattern_data["strength"]
        
        # Peso maior para padrões mais recentes
        weight = 1.0 + (i / lookback) * 0.5
        
        if signal == "CALL":
            call_strength += strength * weight
        elif signal == "PUT":
            put_strength += strength * weight
    
    total_strength = call_strength + put_strength
    if total_strength < 0.1:
        return {"score": 0.0, "direction": "NEUTRAL", "patterns": patterns}
    
    if call_strength > put_strength:
        confidence = call_strength / total_strength
        return {
            "score": float(confidence),
            "direction": "CALL",
            "patterns": patterns,
            "call_strength": float(call_strength),
            "put_strength": float(put_strength)
        }
    else:
        confidence = put_strength / total_strength
        return {
            "score": float(confidence),
            "direction": "PUT",
            "patterns": patterns,
            "call_strength": float(call_strength),
            "put_strength": float(put_strength)
        }

# ===================== IA PREDITIVA (30 VELAS) =====================
def predict_direction_ml(df: pd.DataFrame, lookback: int = 30) -> Dict[str, Any]:
    """
    IA Preditiva: Analisa últimas 30 velas para prever direção.
    Usa múltiplos indicadores e padrões para predição.
    """
    if len(df) < lookback + 5:
        return {"predicted": "NEUTRAL", "confidence": 0.0, "reason": "dados_insuficientes"}
    
    recent = df.tail(lookback)
    
    # 1. ANÁLISE DE MOMENTUM (últimas 30 velas)
    closes = recent["close"].to_numpy(float)
    highs = recent["high"].to_numpy(float)
    lows = recent["low"].to_numpy(float)
    
    # Tendência de preço
    price_change = closes[-1] - closes[0]
    price_trend = "CALL" if price_change > 0 else ("PUT" if price_change < 0 else "NEUTRAL")
    price_strength = abs(price_change) / max(closes[0], 1e-9)
    
    # 2. ANÁLISE DE HIGHER HIGHS / LOWER LOWS
    hh_count = 0
    ll_count = 0
    for i in range(5, len(recent)):
        if highs[i] > max(highs[i-5:i]):
            hh_count += 1
        if lows[i] < min(lows[i-5:i]):
            ll_count += 1
    
    structure_signal = "CALL" if hh_count > ll_count else ("PUT" if ll_count > hh_count else "NEUTRAL")
    structure_strength = abs(hh_count - ll_count) / max(1, lookback - 5)
    
    # 3. ANÁLISE DE MOMENTUM RECENTE (últimas 10 velas)
    recent_10 = closes[-10:]
    recent_momentum = (recent_10[-1] - recent_10[0]) / max(recent_10[0], 1e-9)
    momentum_signal = "CALL" if recent_momentum > 0 else ("PUT" if recent_momentum < 0 else "NEUTRAL")
    momentum_strength = abs(recent_momentum)
    
    # 4. ANÁLISE DE VELAS DIRECIONAIS
    bullish_candles = sum(1 for i in range(len(recent)) if closes[i] > recent["open"].iloc[i])
    bearish_candles = sum(1 for i in range(len(recent)) if closes[i] < recent["open"].iloc[i])
    
    candle_signal = "CALL" if bullish_candles > bearish_candles else ("PUT" if bearish_candles > bullish_candles else "NEUTRAL")
    candle_strength = abs(bullish_candles - bearish_candles) / len(recent)
    
    # 5. ANÁLISE DE VOLATILIDADE (estável = melhor predição)
    ranges = [highs[i] - lows[i] for i in range(len(recent))]
    volatility_cv = np.std(ranges) / max(np.mean(ranges), 1e-9)  # Coeficiente de variação
    volatility_bonus = 1.0 - min(1.0, volatility_cv)  # Baixa volatilidade = mais confiável
    
    # 6. ANÁLISE DE SUPORTE/RESISTÊNCIA PRÓXIMA
    current_price = closes[-1]
    recent_highs = highs[-10:]
    recent_lows = lows[-10:]
    
    resistance_distance = (max(recent_highs) - current_price) / max(current_price, 1e-9)
    support_distance = (current_price - min(recent_lows)) / max(current_price, 1e-9)
    
    sr_signal = "PUT" if resistance_distance < 0.005 else ("CALL" if support_distance < 0.005 else "NEUTRAL")
    sr_strength = 0.5 if sr_signal != "NEUTRAL" else 0.0
    
    # 7. PADRÕES DE VELAS (últimas 5 velas)
    pattern_analysis = analyze_candle_patterns_sequence(df, lookback=5)
    pattern_signal = pattern_analysis["direction"]
    pattern_strength = pattern_analysis["score"]
    
    # AGREGAÇÃO INTELIGENTE COM PESOS
    weights = {
        "price_trend": 0.15,
        "structure": 0.20,
        "momentum": 0.20,
        "candles": 0.15,
        "sr": 0.10,
        "patterns": 0.20
    }
    
    call_score = 0.0
    put_score = 0.0
    
    if price_trend == "CALL":
        call_score += price_strength * weights["price_trend"] * 10
    elif price_trend == "PUT":
        put_score += price_strength * weights["price_trend"] * 10
    
    if structure_signal == "CALL":
        call_score += structure_strength * weights["structure"] * 10
    elif structure_signal == "PUT":
        put_score += structure_strength * weights["structure"] * 10
    
    if momentum_signal == "CALL":
        call_score += momentum_strength * weights["momentum"] * 10
    elif momentum_signal == "PUT":
        put_score += momentum_strength * weights["momentum"] * 10
    
    if candle_signal == "CALL":
        call_score += candle_strength * weights["candles"] * 10
    elif candle_signal == "PUT":
        put_score += candle_strength * weights["candles"] * 10
    
    if sr_signal == "CALL":
        call_score += sr_strength * weights["sr"] * 10
    elif sr_signal == "PUT":
        put_score += sr_strength * weights["sr"] * 10
    
    if pattern_signal == "CALL":
        call_score += pattern_strength * weights["patterns"] * 10
    elif pattern_signal == "PUT":
        put_score += pattern_strength * weights["patterns"] * 10
    
    # Aplica bônus de volatilidade estável
    call_score *= (0.5 + volatility_bonus * 0.5)
    put_score *= (0.5 + volatility_bonus * 0.5)
    
    # DECISÃO FINAL
    total_score = call_score + put_score
    
    if total_score < 0.5:
        return {
            "predicted": "NEUTRAL",
            "confidence": 0.0,
            "reason": "sinais_fracos",
            "details": {
                "call_score": float(call_score),
                "put_score": float(put_score),
                "volatility_bonus": float(volatility_bonus)
            }
        }
    
    if call_score > put_score:
        confidence = call_score / total_score
        if confidence < 0.58:  # Mínimo 58% de confiança
            return {"predicted": "NEUTRAL", "confidence": float(confidence), "reason": "confianca_baixa"}
        
        return {
            "predicted": "CALL",
            "confidence": float(confidence),
            "reason": "ml_prediction",
            "details": {
                "call_score": float(call_score),
                "put_score": float(put_score),
                "price_trend": price_trend,
                "structure": structure_signal,
                "momentum": momentum_signal,
                "pattern": pattern_signal,
                "volatility_bonus": float(volatility_bonus),
                "bullish_candles": int(bullish_candles),
                "bearish_candles": int(bearish_candles)
            }
        }
    else:
        confidence = put_score / total_score
        if confidence < 0.58:  # Mínimo 58% de confiança
            return {"predicted": "NEUTRAL", "confidence": float(confidence), "reason": "confianca_baixa"}
        
        return {
            "predicted": "PUT",
            "confidence": float(confidence),
            "reason": "ml_prediction",
            "details": {
                "call_score": float(call_score),
                "put_score": float(put_score),
                "price_trend": price_trend,
                "structure": structure_signal,
                "momentum": momentum_signal,
                "pattern": pattern_signal,
                "volatility_bonus": float(volatility_bonus),
                "bullish_candles": int(bullish_candles),
                "bearish_candles": int(bearish_candles)
            }
        }

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

    else:  # PUT
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

def sr_block_any_strong(df_m1: pd.DataFrame, atr_val: float) -> Optional[str]:
    res, sup = strong_sr_levels_last200(df_m1, atr_val)
    price = float(df_m1["close"].iloc[-1])
    atr_safe = max(atr_val, 1e-9)

    levels = res + sup
    if not levels:
        return None

    for lvl, touches in levels:
        dist_atr = abs(lvl - price) / atr_safe
        zone = min(SR_HARD_BLOCK_ATR + 0.10 * max(0, touches - SR_MIN_TOUCHES_STRONG), 1.50)
        if dist_atr <= zone:
            return f"bloqueado_SR_FORTE(nivel={lvl:.6f},toques={touches},dist={dist_atr:.2f}ATR<=zona{zone:.2f})"
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
    # ⭐ MAIS RIGOROSO: bloqueia corredores menores
    if corridor_atr <= 1.10:  # AUMENTADO de 0.85 para 1.10
        return f"pingpong(corredor={corridor_atr:.2f}ATR sup={s_lvl:.6f} res={r_lvl:.6f})"
    return None

# ===================== PROJEÇÃO E VALIDAÇÃO DE ENTRADA (MELHORADA) =====================
def validate_entry_quality(df_m1: pd.DataFrame, atr_val: float, direction: str, entry_price: float, pb_high: float, pb_low: float) -> Dict[str, Any]:
    """⭐ VALIDAÇÃO MELHORADA: Mais rigorosa e inteligente"""
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

    # ⭐ MAIS FLEXIVEL: corpo mínimo reduzido
    if body_ratio < 0.20:
        return {"valid": False, "confidence": 0.0, "reason": f"candle_fraco(body={body_ratio:.2f})"}

    if direction == "CALL":
        stop_loss = pb_low - (0.15 * atr_val)
        risk = entry_price - stop_loss
        target_1 = entry_price + (risk * 1.5)

        recent_highs = df_m1.tail(20)["high"].to_numpy(float)
        max_recent = float(np.max(recent_highs))

        if target_1 > max_recent * 1.005:
            confidence = 0.80  # AUMENTADO de 0.75
        else:
            confidence = 0.60  # AUMENTADO de 0.55

    else:  # PUT
        stop_loss = pb_high + (0.15 * atr_val)
        risk = stop_loss - entry_price
        target_1 = entry_price - (risk * 1.5)

        recent_lows = df_m1.tail(20)["low"].to_numpy(float)
        min_recent = float(np.min(recent_lows))

        if target_1 < min_recent * 0.995:
            confidence = 0.80  # AUMENTADO
        else:
            confidence = 0.60  # AUMENTADO

    risk_atr = risk / max(atr_val, 1e-9)

    # ⭐ MAIS RIGOROSO: faixa de risco mais estreita
    if risk_atr < 0.25 or risk_atr > 1.8:  # AJUSTADO de 0.2-2.0 para 0.25-1.8
        return {"valid": False, "confidence": 0.0, "reason": f"risco_inadequado({risk_atr:.2f}ATR)"}

    # MOMENTUM
    last_3_closes = df_m1.tail(3)["close"].to_numpy(float)
    momentum = abs(last_3_closes[-1] - last_3_closes[0]) / max(atr_val, 1e-9)

    # ⭐ MAIS RIGOROSO: momentum mínimo aumentado
    if momentum < 0.15:  # AUMENTADO de 0.10 para 0.15
        confidence *= 0.75  # PENALIZA MAIS

    # ALINHAMENTO
    last_3 = df_m1.tail(3)
    aligned = 0
    for _, row in last_3.iterrows():
        c = float(row["close"])
        o = float(row["open"])
        if (direction == "CALL" and c > o) or (direction == "PUT" and c < o):
            aligned += 1

    alignment_ratio = aligned / 3.0
    if alignment_ratio >= 0.67:
        confidence *= 1.20  # BÔNUS AUMENTADO
    elif alignment_ratio < 0.34:
        confidence *= 0.75  # PENALIZA MAIS

    confidence = min(1.0, max(0.0, confidence))

    # ⭐ MAIS FLEXIVEL: confidence mínima reduzida
    if confidence < 0.45:
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

# ===================== VALIDAÇÃO DE CONTINUAÇÃO DE TENDÊNCIA (MELHORADA) =====================
def validate_trend_continuation(df_m1: pd.DataFrame, impulso_dir: str, pb_end_idx: int) -> Dict[str, Any]:
    """⭐ MAIS RIGOROSA: Bloqueia contra-tendências fortes"""
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
            # ⭐ MAIS RIGOROSO: bloqueia tendências contrárias menores
            if price_change_pct > 0.010:  # REDUZIDO de 1.5% para 1.0%
                return {"valid": False, "reason": "contra_tendencia_forte_alta", "strength": 0.2}
            return {"valid": True, "reason": "contra_tendencia_fraca", "strength": 0.4}
        return {"valid": True, "reason": "continuacao_queda", "strength": min(1.0, price_change_pct * 50)}

    else:  # CALL
        if price_change < 0:
            # ⭐ MAIS RIGOROSO
            if price_change_pct > 0.010:  # REDUZIDO de 1.5% para 1.0%
                return {"valid": False, "reason": "contra_tendencia_forte_baixa", "strength": 0.2}
            return {"valid": True, "reason": "contra_tendencia_fraca", "strength": 0.4}
        return {"valid": True, "reason": "continuacao_alta", "strength": min(1.0, price_change_pct * 50)}

# ===================== ANÁLISE INTELIGENTE DE CONTEXTO (MELHORADA) =====================
def analyze_market_context(df_m1: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    """⭐ ANÁLISE MAIS RIGOROSA E INTELIGENTE"""
    if len(df_m1) < 50:
        return {"quality": 0.0, "context": "insuficiente"}

    recent = df_m1.tail(20)
    closes = recent["close"].to_numpy(float)
    highs = recent["high"].to_numpy(float)
    lows = recent["low"].to_numpy(float)

    # 1. MOMENTUM DIRECIONAL
    bullish = sum(1 for i in range(len(recent)) if closes[i] > recent["open"].iloc[i])
    bearish = sum(1 for i in range(len(recent)) if closes[i] < recent["open"].iloc[i])
    directional_bias = abs(bullish - bearish) / len(recent)

    # 2. VOLATILIDADE ORDENADA
    ranges = [highs[i] - lows[i] for i in range(len(recent))]
    avg_range = np.mean(ranges)
    std_range = np.std(ranges)
    volatility_consistency = 1.0 - min(1.0, std_range / max(avg_range, 1e-9))

    # 3. HIGHER HIGHS / LOWER LOWS
    hh_count = sum(1 for i in range(5, len(recent)) if highs[i] > max(highs[i-5:i]))
    ll_count = sum(1 for i in range(5, len(recent)) if lows[i] < min(lows[i-5:i]))
    structure_quality = max(hh_count, ll_count) / max(1, len(recent) - 5)

    # 4. MOMENTUM DE PREÇO
    last_10 = closes[-10:]
    price_momentum = abs(last_10[-1] - last_10[0]) / (max(atr_val, 1e-9) * 10)
    price_momentum = min(1.0, price_momentum)

    # 5. LIMPEZA DOS CANDLES
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

    # ⭐ SCORE FINAL MAIS EXIGENTE (pesos ajustados)
    quality = (
        directional_bias * 0.30 +  # AUMENTADO: mais peso na direção
        volatility_consistency * 0.20 +
        structure_quality * 0.25 +
        price_momentum * 0.15 +
        avg_body_ratio * 0.10  # REDUZIDO: menos peso no corpo
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
def calcular_stake_dinamico(iq: IQ_Option, base_stake: float) -> float:
    if not USE_DYNAMIC_STAKE:
        return float(max(VALOR_MINIMO, base_stake))

    try:
        saldo = float(iq.get_balance())
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

# ===================== IA ONLINE (MELHORADA E OTIMIZADA) =====================
def _safe_load_json(path: str) -> Dict[str, Any]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"meta": {"total": 0}, "arms": {}, "patterns": {}, "streaks": {}}  # NOVO: streaks

def _safe_save_json(path: str, data: Dict[str, Any]):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def _bucket(x: float, step: float, lo: float, hi: float) -> int:
    x = _clip(x, lo, hi)
    return int(round((x - lo) / step))

def ai_make_key(ativo: str, setup: Dict[str, Any]) -> str:
    """⭐ CHAVE OTIMIZADA: Mais granular e precisa"""
    d = str(setup.get("dir", "NEUTRAL"))
    sc = float(setup.get("score", 0.0))
    pb = int(setup.get("pb_len", 0))
    retr = float(setup.get("retr", 0.0))
    Aatr = float(setup.get("A_atr", 0.0))
    effA = float(setup.get("effA", 0.0))
    flips = float(setup.get("flips", 0.0))
    distBreak = float(setup.get("distBreak", 0.0))
    market_q = float(setup.get("market_quality", 0.0))
    entry_conf = float(setup.get("entry_confidence", 0.0))

    # ⭐ Buckets mais refinados
    b_sc = _bucket(sc, 0.03, 0.50, 1.00)  # REDUZIDO: mais granular
    b_re = _bucket(retr, 0.05, 0.20, 0.75)  # AJUSTADO: nova faixa
    b_A  = _bucket(Aatr, 0.30, 0.65, 6.00)  # AJUSTADO
    b_eff = _bucket(effA, 0.06, 0.50, 1.00)  # REDUZIDO: mais granular
    b_flip = _bucket(flips, 0.08, 0.0, 0.65)  # AJUSTADO: nova faixa
    b_dist = _bucket(distBreak, 0.04, 0.0, 0.35)  # REDUZIDO
    b_mkt = _bucket(market_q, 0.08, 0.0, 1.00)  # NOVO: qualidade de mercado
    b_entry = _bucket(entry_conf, 0.08, 0.0, 1.00)  # NOVO: confiança de entrada

    return f"{d}|sc{b_sc}|pb{pb}|re{b_re}|A{b_A}|eff{b_eff}|fl{b_flip}|dst{b_dist}|mkt{b_mkt}|ent{b_entry}"

def ai_prior_from_setup(setup: Dict[str, Any]) -> float:
    """⭐ PRIOR MELHORADO: Mais conservador e inteligente"""
    sc = float(setup.get("score", 0.0))
    effA = float(setup.get("effA", 0.0))
    flips = float(setup.get("flips", 0.5))
    retr = float(setup.get("retr", 0.5))
    distBreak = float(setup.get("distBreak", 0.2))
    market_q = float(setup.get("market_quality", 0.0))
    entry_conf = float(setup.get("entry_confidence", 0.0))

    # ⭐ BASE MAIS CONSERVADORA
    p = 0.45 + (sc - 0.50) * 0.30  # REDUZIDO: prior mais baixo

    # ⭐ AJUSTES MAIS EXIGENTES
    if effA > 0.75:
        p += 0.06  # AUMENTADO
    elif effA < 0.55:
        p -= 0.05

    if flips < 0.30:
        p += 0.07  # AUMENTADO
    elif flips > 0.50:
        p -= 0.06

    if 0.30 <= retr <= 0.50:
        p += 0.06  # AUMENTADO
    elif retr < 0.20 or retr > 0.65:
        p -= 0.05

    if distBreak < 0.12:
        p += 0.05  # AUMENTADO
    elif distBreak > 0.22:
        p -= 0.04

    # ⭐ NOVO: Qualidade de mercado tem grande impacto
    if market_q > 0.65:
        p += 0.08
    elif market_q < 0.35:
        p -= 0.08

    # ⭐ NOVO: Confiança de entrada
    if entry_conf > 0.70:
        p += 0.07
    elif entry_conf < 0.45:
        p -= 0.06

    return _clip(p, 0.35, 0.70)  # AJUSTADO: faixa mais baixa

def ai_predict(ativo: str, setup: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, float]:
    key = ai_make_key(ativo, setup)
    arms = stats.setdefault("arms", {})
    meta = stats.setdefault("meta", {"total": 0})

    total = int(meta.get("total", 0))
    arm = arms.get(key)

    prior = ai_prior_from_setup(setup)

    if arm is None:
        # ⭐ INICIA COM PRIOR MAIS CONSERVADOR
        a = 1.5 * prior  # REDUZIDO de 2.0
        b = 1.5 * (1.0 - prior)
        n = 0
        arms[key] = {"a": a, "b": b, "n": n}
        bayes_mean = a / (a + b)
        ucb01 = 1.0
        conf = float(_clip(0.50 + float(setup.get("score", 0.0)) * 0.25, 0.0, 0.85))  # REDUZIDO
        return {"prob": float(bayes_mean), "bayes": float(bayes_mean), "ucb01": float(ucb01),
                "conf": float(conf), "n_arm": 0, "total": total, "key": key, "prior": prior}

    a = float(arm.get("a", 1.0))
    b = float(arm.get("b", 1.0))
    n = int(arm.get("n", 0))

    bayes_mean = a / (a + b)

    # UCB
    if n <= 0:
        ucb01 = 1.0
    else:
        bonus = math.sqrt(2.0 * math.log(max(2, total + 1)) / max(1, n))
        ucb01 = _clip(bayes_mean + bonus, 0.0, 1.0)

    # ⭐ CONFIANÇA MAIS EXIGENTE
    conf = _clip(n / (n + 12.0), 0.0, 0.98)  # AUMENTADO de 10 para 12

    # ⭐ PROB FINAL: Mais peso no histórico real
    w = _clip(n / (n + 20.0), 0.0, 1.0)  # REDUZIDO de 25 para 20
    prob = (1.0 - w) * prior + w * bayes_mean
    prob = _clip(prob, 0.0, 1.0)

    return {"prob": float(prob), "bayes": float(bayes_mean), "ucb01": float(ucb01),
            "conf": float(conf), "n_arm": n, "total": total, "key": key, "prior": prior}

def ai_update(ativo: str, setup: Dict[str, Any], pnl: float, stats: Dict[str, Any]):
    """⭐ UPDATE MELHORADO: Rastreia streaks e padrões ruins"""
    if pnl == 0:
        return

    key = ai_make_key(ativo, setup)
    arms = stats.setdefault("arms", {})
    meta = stats.setdefault("meta", {"total": 0})
    patterns = stats.setdefault("patterns", {})
    streaks = stats.setdefault("streaks", {})  # ⭐ NOVO: tracking de sequências

    arm = arms.get(key)
    if arm is None:
        prior = ai_prior_from_setup(setup)
        arms[key] = {"a": 1.5 * prior, "b": 1.5 * (1.0 - prior), "n": 0}
        arm = arms[key]

    # Atualiza Bayesian
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

    # Rastreamento de padrão
    pattern = patterns.get(key)
    if pattern is None:
        patterns[key] = {"trades": 0, "wins": 0, "losses": 0}
        pattern = patterns[key]

    pattern["trades"] += 1
    if pnl > 0:
        pattern["wins"] += 1
    else:
        pattern["losses"] += 1

    # ⭐ NOVO: Tracking de streak (sequência de losses)
    streak = streaks.get(key)
    if streak is None:
        streaks[key] = {"current_losses": 0, "max_losses": 0}
        streak = streaks[key]

    if pnl < 0:
        streak["current_losses"] += 1
        streak["max_losses"] = max(streak["max_losses"], streak["current_losses"])
    else:
        streak["current_losses"] = 0

def ai_should_block_pattern(ativo: str, setup: Dict[str, Any], stats: Dict[str, Any]) -> Tuple[bool, str]:
    """⭐ BLOQUEIO INTELIGENTE E AGRESSIVO COM PADRÕES RUINS"""
    key = ai_make_key(ativo, setup)
    patterns = stats.get("patterns", {})
    streaks = stats.get("streaks", {})
    pattern = patterns.get(key)
    streak = streaks.get(key)

    # ⭐ BLOQUEIO RÁPIDO: 3 losses consecutivos
    if streak and streak["current_losses"] >= AI_FAST_BLOCK_LOSSES:
        return True, f"🚫FAST_BLOCK({streak['current_losses']}losses_consecutivos)"

    # Fase 1: APRENDIZADO
    if pattern is None or pattern["trades"] < AI_MIN_SAMPLES:
        trades_count = pattern["trades"] if pattern else 0
        return False, f"learning({trades_count}/{AI_MIN_SAMPLES})"

    # Fase 2: AVALIAÇÃO RIGOROSA
    winrate = pattern["wins"] / max(1, pattern["trades"])

    # ⭐ BLOQUEIO MAIS AGRESSIVO
    if winrate < AI_MIN_WINRATE:
        return True, f"🚫BLOCKED_WR={winrate:.0%}({pattern['wins']}W/{pattern['losses']}L)"

    # ⭐ NOVO: Bloqueia se teve muitos losses seguidos no histórico
    if streak and streak["max_losses"] >= 5:
        return True, f"🚫BLOCKED_MAX_STREAK={streak['max_losses']}losses"

    return False, f"✅approved_wr={winrate:.0%}({pattern['wins']}W/{pattern['losses']}L)"

# ===================== PERNADA B (MAIS SELETIVA) =====================
def pernada_b(df_m1: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    if len(df_m1) < 240:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["poucas_velas"]}

    # ⭐ IA PREDITIVA - ANÁLISE DE 30 VELAS
    ml_prediction = predict_direction_ml(df_m1, lookback=30)
    ml_direction = ml_prediction.get("predicted", "NEUTRAL")
    ml_confidence = float(ml_prediction.get("confidence", 0.0))
    ml_reason = ml_prediction.get("reason", "unknown")
    
    # ⭐ PADRÃO FORTE NA VELA ANTERIOR (TA-Lib)
    pattern_analysis = analyze_strong_pattern_last(df_m1, min_score=PATTERN_STRONG_MIN)
    pattern_direction = pattern_analysis.get("direction", "NEUTRAL")
    pattern_score = float(pattern_analysis.get("score", 0.0))
    if pattern_score < PATTERN_STRONG_MIN:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
                "reasons": [f"❌padrao_fraco({pattern_score:.2f})"]}

    # ⭐ CONTEXTO DE MERCADO AGORA É BLOQUEANTE
    context = analyze_market_context(df_m1, atr_val)
    market_quality = float(context.get("quality", 0.0))

    # ⭐ BLOQUEIA CONTEXTO RUIM (mais flexivel)
    if market_quality < 0.20:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
                "reasons": [f"❌contexto_ruim(quality={market_quality:.2f},ctx={context.get('context','?')})"]}

    flips_frac, eff_zone = chop_stats(df_m1, CHOP_LOOKBACK)
    comp = compression_ratio(df_m1, atr_val, COMP_LOOKBACK)
    late_ext = late_extension_atr(df_m1, atr_val, LATE_LOOKBACK)

    decision = df_m1.iloc[-1]
    q = wick_fractions(decision)

    if q["body_frac"] < MIN_BODY_FRAC_BREAK:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
                "reasons": [f"❌gatilho_fraco(body={q['body_frac']:.2f})"]}

    hard_sr = sr_block_any_strong(df_m1, atr_val)
    if hard_sr:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": [f"❌{hard_sr}"]}

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

            # ⭐ MAIS RIGOROSO: pullback precisa ter mais velas contra
            if contra < max(1, int(math.ceil(pb_len * 0.60))):  # AUMENTADO de 0.50 para 0.60
                continue

            impulse_start_idx = len(df_m1) - (pb_len + 1 + w)
            trend_validation = validate_trend_continuation(df_m1, dir_impulso_A, impulse_start_idx)

            # ⭐ AGORA BLOQUEIA se não validar tendência
            if not trend_validation.get("valid", False):
                continue

            trend_strength = float(trend_validation.get("strength", 0.0))
            if trend_strength < TREND_MIN_STRENGTH:
                continue

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

            if dir_entrada != pattern_direction:
                continue

            if dir_entrada != ml_direction or ml_confidence < AI_CONFIRM_MIN:
                continue

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

                # ⭐ SCORE MAIS RIGOROSO: BASE MENOR
                score = 0.40  # AUMENTADO de 0.35

                # 1. IMPULSO (máx +0.15) - AUMENTADO
                impulso_score = min(0.15, (size_A / atr_val - IMPULSO_MIN_ATR) * 0.08)
                score += impulso_score

                # 2. EFICIÊNCIA (máx +0.20) - AUMENTADO - MUITO IMPORTANTE
                eff_score = min(0.20, max(0, (eff_A - MIN_EFF_A) * 0.40))
                score += eff_score

                # 3. RETRAÇÃO IDEAL (máx +0.12) - AUMENTADO
                if 0.30 <= retr <= 0.50:
                    retr_score = 0.12
                elif 0.25 <= retr <= 0.60:
                    retr_score = 0.06
                else:
                    retr_score = max(-0.08, -(abs(retr - 0.40) * 0.20))  # PENALIZA MAIS
                score += retr_score

                # 4. PULLBACK (máx +0.06)
                if 2 <= pb_len <= 3:
                    pb_score = 0.06
                elif pb_len == 1 or pb_len == 4:
                    pb_score = 0.03
                else:
                    pb_score = 0.0
                score += pb_score

                # 5. CHOPPINESS - PENALIZA MAIS
                if flips_frac > 0.50:
                    chop_penalty = min(0.20, (flips_frac - 0.50) * 0.60)  # AUMENTADO
                    score -= chop_penalty

                # 6. QUALIDADE DE CONTEXTO (máx +0.22) - AUMENTADO - CRÍTICO
                if market_quality > 0.65:
                    ctx_score = 0.22
                elif market_quality > 0.50:
                    ctx_score = 0.12
                elif market_quality > 0.35:
                    ctx_score = 0.05
                else:
                    ctx_score = -0.10  # PENALIZA FORTE
                score += ctx_score

                # 7. TENDÊNCIA (máx +0.10) - AUMENTADO
                trend_score = min(0.15, trend_strength * 0.20)
                score += trend_score

                # 8. QUALIDADE DA ENTRADA (máx +0.28) - AUMENTADO - MUITO CRÍTICO
                if entry_confidence > 0.70:
                    entry_score = 0.28
                elif entry_confidence > 0.60:
                    entry_score = 0.18
                elif entry_confidence > 0.50:
                    entry_score = 0.10
                else:
                    entry_score = -0.08  # PENALIZA
                score += entry_score

                # 9. MOMENTUM (máx +0.12) - AUMENTADO
                momentum_score = min(0.12, entry_momentum * 0.10)
                score += momentum_score

                # 10. ALINHAMENTO (máx +0.10) - AUMENTADO
                if entry_alignment >= 0.67:
                    align_score = 0.10
                elif entry_alignment >= 0.34:
                    align_score = 0.04
                else:
                    align_score = -0.05  # PENALIZA
                score += align_score

                # ⭐ BÔNUS LT AUMENTADO
                if has_lt and lt_confluence > 0.8:
                    score += 0.30  # AUMENTADO de 0.25
                elif has_lt and lt_confluence > 0.5:
                    score += 0.18  # AUMENTADO de 0.15
                elif has_lt and lt_confluence > 0.2:
                    score += 0.06  # AUMENTADO de 0.05

                # 11. RISCO
                if risk_atr > 1.3:
                    risk_penalty = 0.12  # AUMENTADO
                elif risk_atr > 1.0:
                    risk_penalty = 0.06
                elif risk_atr < 0.30:
                    risk_penalty = 0.06
                else:
                    risk_penalty = 0.0
                score -= risk_penalty

                # 12. CONFLUÊNCIA PERFEITA - Bônus AUMENTADO
                perfect_count = 0
                if market_quality > 0.65: perfect_count += 1
                if eff_A > 0.75: perfect_count += 1
                if 0.30 <= retr <= 0.50: perfect_count += 1
                if entry_confidence > 0.70: perfect_count += 1
                if entry_alignment >= 0.67: perfect_count += 1
                if lt_confluence > 0.8: perfect_count += 1

                if perfect_count >= 5:
                    confluence_bonus = 0.25  # AUMENTADO de 0.20
                elif perfect_count >= 4:
                    confluence_bonus = 0.15  # AUMENTADO
                elif perfect_count >= 3:
                    confluence_bonus = 0.08  # AUMENTADO
                else:
                    confluence_bonus = 0.0

                score += confluence_bonus
                
                # 13. ⭐ BÔNUS IA PREDITIVA (30 VELAS) - NOVO
                if ml_direction == "CALL" and ml_confidence > 0.60:
                    ml_bonus = min(0.20, ml_confidence * 0.25)  # Até +0.20
                    score += ml_bonus
                    log.info(paint(f"  🤖 ML prediz CALL (conf={ml_confidence:.2f}) +{ml_bonus:.2f}", C.M))
                elif ml_direction == "PUT":
                    # Penaliza se ML prevê PUT mas estamos tentando CALL
                    ml_penalty = min(0.15, ml_confidence * 0.20)
                    score -= ml_penalty
                    log.info(paint(f"  ⚠️ ML prediz PUT (conf={ml_confidence:.2f}) -{ml_penalty:.2f}", C.Y))
                
                # 14. ⭐ BÔNUS PADRÕES DE VELAS - NOVO
                if pattern_direction == "CALL" and pattern_score >= PATTERN_STRONG_MIN:
                    pattern_bonus = min(0.15, pattern_score * 0.20)
                    score += pattern_bonus
                    log.info(paint(f"  📊 Padrões indicam CALL (score={pattern_score:.2f}) +{pattern_bonus:.2f}", C.G))
                elif pattern_direction == "PUT":
                    pattern_penalty = min(0.12, pattern_score * 0.15)
                    score -= pattern_penalty
                    log.info(paint(f"  ⚠️ Padrões indicam PUT (score={pattern_score:.2f}) -{pattern_penalty:.2f}", C.Y))

                score = float(max(0.0, min(0.98, score)))

                # ⭐ FILTROS MAIS RIGOROSOS
                if score < 0.55:  # AUMENTADO de 0.48 para 0.55
                    continue

                if market_quality < 0.35:  # AUMENTADO de 0.25 para 0.35
                    continue

                if entry_confidence < 0.50:  # AUMENTADO de 0.40 para 0.50
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
                    "ml_prediction": ml_direction,
                    "ml_confidence": float(ml_confidence),
                    "pattern_direction": pattern_direction,
                    "pattern_score": float(pattern_score),
                    "reasons": [
                        "✅CALL_B",
                        f"A={size_A/atr_val:.2f}ATR", f"retr={retr:.2f}", f"pb={pb_len}",
                        f"effA={eff_A:.2f}",
                        f"chop={flips_frac:.2f}",
                        f"ctx={context.get('context','?')}({market_quality:.2f})",
                        f"conf={confluence_bonus:.2f}",
                        f"entry={entry_confidence:.2f}",
                        f"⭐LTA={lt_confluence:.2f}" if has_lt else "sem_LTA",
                        f"🤖ML={ml_direction}({ml_confidence:.2f})",
                        f"📊Pattern={pattern_direction}({pattern_score:.2f})"
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

                # ⭐ MESMO SISTEMA DE SCORE RIGOROSO PARA PUT
                score = 0.40

                impulso_score = min(0.15, (size_A / atr_val - IMPULSO_MIN_ATR) * 0.08)
                score += impulso_score

                eff_score = min(0.20, max(0, (eff_A - MIN_EFF_A) * 0.40))
                score += eff_score

                if 0.30 <= retr <= 0.50:
                    retr_score = 0.12
                elif 0.25 <= retr <= 0.60:
                    retr_score = 0.06
                else:
                    retr_score = max(-0.08, -(abs(retr - 0.40) * 0.20))
                score += retr_score

                if 2 <= pb_len <= 3:
                    pb_score = 0.06
                elif pb_len == 1 or pb_len == 4:
                    pb_score = 0.03
                else:
                    pb_score = 0.0
                score += pb_score

                if flips_frac > 0.50:
                    chop_penalty = min(0.20, (flips_frac - 0.50) * 0.60)
                    score -= chop_penalty

                if market_quality > 0.65:
                    ctx_score = 0.22
                elif market_quality > 0.50:
                    ctx_score = 0.12
                elif market_quality > 0.35:
                    ctx_score = 0.05
                else:
                    ctx_score = -0.10
                score += ctx_score

                trend_score = min(0.10, trend_strength * 0.15)
                score += trend_score

                if entry_confidence > 0.70:
                    entry_score = 0.28
                elif entry_confidence > 0.60:
                    entry_score = 0.18
                elif entry_confidence > 0.50:
                    entry_score = 0.10
                else:
                    entry_score = -0.08
                score += entry_score

                momentum_score = min(0.12, entry_momentum * 0.10)
                score += momentum_score

                if entry_alignment >= 0.67:
                    align_score = 0.10
                elif entry_alignment >= 0.34:
                    align_score = 0.04
                else:
                    align_score = -0.05
                score += align_score

                if has_lt and lt_confluence > 0.8:
                    score += 0.30
                elif has_lt and lt_confluence > 0.5:
                    score += 0.18
                elif has_lt and lt_confluence > 0.2:
                    score += 0.06

                if risk_atr > 1.3:
                    risk_penalty = 0.12
                elif risk_atr > 1.0:
                    risk_penalty = 0.06
                elif risk_atr < 0.30:
                    risk_penalty = 0.06
                else:
                    risk_penalty = 0.0
                score -= risk_penalty

                perfect_count = 0
                if market_quality > 0.65: perfect_count += 1
                if eff_A > 0.75: perfect_count += 1
                if 0.30 <= retr <= 0.50: perfect_count += 1
                if entry_confidence > 0.70: perfect_count += 1
                if entry_alignment >= 0.67: perfect_count += 1
                if lt_confluence > 0.8: perfect_count += 1

                if perfect_count >= 5:
                    confluence_bonus = 0.25
                elif perfect_count >= 4:
                    confluence_bonus = 0.15
                elif perfect_count >= 3:
                    confluence_bonus = 0.08
                else:
                    confluence_bonus = 0.0

                score += confluence_bonus
                
                # 13. ⭐ BÔNUS IA PREDITIVA (30 VELAS) - NOVO
                if ml_direction == "PUT" and ml_confidence > 0.60:
                    ml_bonus = min(0.20, ml_confidence * 0.25)  # Até +0.20
                    score += ml_bonus
                    log.info(paint(f"  🤖 ML prediz PUT (conf={ml_confidence:.2f}) +{ml_bonus:.2f}", C.M))
                elif ml_direction == "CALL":
                    # Penaliza se ML prevê CALL mas estamos tentando PUT
                    ml_penalty = min(0.15, ml_confidence * 0.20)
                    score -= ml_penalty
                    log.info(paint(f"  ⚠️ ML prediz CALL (conf={ml_confidence:.2f}) -{ml_penalty:.2f}", C.Y))
                
                # 14. ⭐ BÔNUS PADRÕES DE VELAS - NOVO
                if pattern_direction == "PUT" and pattern_score >= PATTERN_STRONG_MIN:
                    pattern_bonus = min(0.15, pattern_score * 0.20)
                    score += pattern_bonus
                    log.info(paint(f"  📊 Padrões indicam PUT (score={pattern_score:.2f}) +{pattern_bonus:.2f}", C.R))
                elif pattern_direction == "CALL":
                    pattern_penalty = min(0.12, pattern_score * 0.15)
                    score -= pattern_penalty
                    log.info(paint(f"  ⚠️ Padrões indicam CALL (score={pattern_score:.2f}) -{pattern_penalty:.2f}", C.Y))

                score = float(max(0.0, min(0.98, score)))

                if score < 0.55:
                    continue

                if market_quality < 0.35:
                    continue

                if entry_confidence < 0.50:
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
                    "ml_prediction": ml_direction,
                    "ml_confidence": float(ml_confidence),
                    "pattern_direction": pattern_direction,
                    "pattern_score": float(pattern_score),
                    "reasons": [
                        "✅PUT_B",
                        f"A={size_A/atr_val:.2f}ATR", f"retr={retr:.2f}", f"pb={pb_len}",
                        f"effA={eff_A:.2f}",
                        f"chop={flips_frac:.2f}",
                        f"ctx={context.get('context','?')}({market_quality:.2f})",
                        f"conf={confluence_bonus:.2f}",
                        f"entry={entry_confidence:.2f}",
                        f"⭐LTB={lt_confluence:.2f}" if has_lt else "sem_LTB",
                        f"🤖ML={ml_direction}({ml_confidence:.2f})",
                        f"📊Pattern={pattern_direction}({pattern_score:.2f})"
                    ]
                }

            if best is None or setup["score"] > best["score"]:
                best = setup

    if best is None:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["sem_pernadaB_valida"]}

    block_final = sr_block_directional_multi(df_m1, atr_val, best["dir"])
    if block_final:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": [block_final]}

    hard_sr_final = sr_block_any_strong(df_m1, atr_val)
    if hard_sr_final:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": [hard_sr_final]}

    return best

# ===================== ESCOLHER MELHOR SETUP DO MINUTO =====================
def escolher_melhor_setup(iq: IQ_Option, ativos: List[str]):
    best_trade = None
    best_any = None

    for a in ativos:
        if a in cooldown and (time.time() - cooldown[a]) < COOLDOWN_ATIVO:
            continue
        if a in cooldown_spike and (time.time() - cooldown_spike[a]) < (SPIKE_COOLDOWN_MIN * 60):
            continue

        df = get_candles_df(iq, a, TF_M1, N_M1, end_ts=end_ts_closed(TF_M1))
        if df is None:
            continue

        atr_val = atr(df, 14)
        last_closed = df.iloc[-1]

        if is_spike_wicky(last_closed, atr_val):
            cooldown_spike[a] = time.time()
            continue

        setup = pernada_b(df, atr_val)

        sc_any = float(setup.get("score", 0.0))
        cand_any = (sc_any, a, setup, float(atr_val))
        if best_any is None or cand_any[0] > best_any[0]:
            best_any = cand_any

        if setup.get("trade"):
            cand_trade = (float(setup["score"]), a, setup, float(atr_val))
            if best_trade is None or cand_trade[0] > best_trade[0]:
                best_trade = cand_trade

    return best_trade, best_any

# ===================== ORDEM =====================
def enviar_ordem(iq: IQ_Option, ativo: str, direcao: str, stake: float) -> Optional[Tuple[str, int]]:
    d = "call" if direcao == "CALL" else "put"
    valor = float(max(VALOR_MINIMO, stake))

    try:
        ok, op_id = safe_call(iq, iq.buy, valor, ativo, d, int(EXP_FIXA))
        if ok and op_id:
            return ("turbo", int(op_id))
        log.warning(paint(f"[ORDEM-FAIL] TURBO ok={ok} op_id={op_id}", C.Y))
    except Exception as e:
        log.warning(paint(f"[ORDEM-EXC] TURBO {e}", C.Y))

    try:
        ok, op_id = safe_call(iq, iq.buy_digital_spot, ativo, valor, d, int(EXP_FIXA))
        if ok and op_id:
            return ("digital", int(op_id))
        log.warning(paint(f"[ORDEM-FAIL] DIGITAL ok={ok} op_id={op_id}", C.Y))
    except Exception as e:
        log.warning(paint(f"[ORDEM-EXC] DIGITAL {e}", C.Y))

    return None

def wait_result(iq: IQ_Option, op_type: str, op_id: int) -> float:
    while True:
        try:
            if op_type == "turbo":
                ok, res = safe_call(iq, iq.check_win_v4, op_id)
                if ok:
                    return float(res)
            else:
                res = safe_call(iq, iq.check_win_digital_v2, op_id)
                if isinstance(res, (int, float)):
                    return float(res)
        except Exception:
            ensure_connected(iq)
        time.sleep(0.25)

# ===================== MAIN =====================
def main():
    iq: Optional[IQ_Option] = None
    iq = ensure_connected(iq)

    log.info(paint("🚀 WS_AUTO_AI_OPTIMIZED - Versão Otimizada com IA Inteligente", C.M))
    log.info("✅ Filtros mais seletivos e rigorosos")
    log.info("✅ IA aprende e bloqueia padrões ruins rapidamente")
    log.info("✅ Score mais exigente com confluências")

    stats = _safe_load_json(AI_STATS_FILE) if IA_ON else {"meta": {"total": 0}, "arms": {}, "patterns": {}, "streaks": {}}
    if IA_ON:
        log.info(paint(f"🤖 IA=ON | file={AI_STATS_FILE} | min_samples={AI_MIN_SAMPLES} | min_prob={AI_MIN_PROB:.2f} | conf_min={AI_CONF_MIN:.2f}", C.M))
        log.info(paint(f"⚡ FAST_BLOCK: {AI_FAST_BLOCK_LOSSES} losses consecutivos = bloqueio imediato", C.M))

    try:
        saldo_inicial = float(iq.get_balance())
        log.info(paint(f"💰 SALDO INICIAL: {saldo_inicial:.2f} | META: {META_LUCRO_PERCENT:.1f}%", C.G))
        if USE_DYNAMIC_STAKE:
            log.info(paint(f"📊 GESTÃO: {PERCENT_BANCA:.1f}% da banca por operação", C.B))
        else:
            log.info(paint(f"📊 GESTÃO: Stake fixo de {STAKE_FIXA:.2f}", C.B))
    except Exception:
        saldo_inicial = 1000.0

    total = 0
    wins = 0

    while True:
        iq = ensure_connected(iq)

        try:
            saldo_atual = float(iq.get_balance())
            deve_parar, lucro_percent = verificar_meta_atingida(saldo_inicial, saldo_atual)
            if deve_parar:
                lucro_abs = saldo_atual - saldo_inicial
                if lucro_percent >= META_LUCRO_PERCENT:
                    log.info(paint(f"🎯 META ATINGIDA! Lucro: {lucro_abs:.2f} ({lucro_percent:.2f}%)", C.G))
                else:
                    log.info(paint(f"🛑 STOP LOSS! Perda: {lucro_abs:.2f} ({lucro_percent:.2f}%)", C.R))
                break
        except Exception as e:
            log.warning(f"Erro ao verificar meta: {e}")

        ativos = obter_top_ativos_otc(iq)
        if not ativos:
            log.warning("Sem ativos com payout mínimo. Tentando em 10s...")
            time.sleep(10)
            continue

        wait_until_minus(TF_M1, DECIDIR_ANTES_FECHAR_SEC)

        best_trade, best_any = escolher_melhor_setup(iq, ativos)

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
            f"[SINAL-HARD] {ativo} -> {final_dir} | score={score:.2f} | ATR={atr_val:.6f}",
            dir_color(final_dir)
        ))
        log.info(paint(f"  📋 {' | '.join(setup.get('reasons', []))}", C.B))

        # ===================== IA FILTRO OTIMIZADO =====================
        if IA_ON:
            pred = ai_predict(ativo, setup, stats)
            prob = float(pred["prob"])
            bayes = float(pred["bayes"])
            ucb01 = float(pred["ucb01"])
            conf = float(pred["conf"])
            n_arm = int(pred["n_arm"])

            # ⭐ BLOQUEIO INTELIGENTE DE PADRÕES
            should_block, block_reason = ai_should_block_pattern(ativo, setup, stats)
            if should_block:
                log.info(paint(f"[🚫 IA-BLOCK] {ativo} {final_dir} | {block_reason}", C.R))
                wait_for_next_open(TF_M1)
                cooldown[ativo] = time.time()
                continue

            if n_arm < AI_MIN_SAMPLES:
                log.info(paint(
                    f"[🤖 IA] {ativo} {final_dir} | {block_reason} | prob={prob:.2f} conf={conf:.2f}",
                    C.M
                ))
            else:
                log.info(paint(
                    f"[🤖 IA] {ativo} {final_dir} | {block_reason} | bayes={bayes:.2f}(n={n_arm}) prob={prob:.2f} conf={conf:.2f}",
                    C.M
                ))

            # ⭐ FILTRO MAIS RIGOROSO
            if n_arm >= AI_MIN_SAMPLES:
                if (prob < AI_MIN_PROB) or (conf < AI_CONF_MIN):
                    log.info(paint(f"[❌ IA-SKIP] {ativo} {final_dir} | prob={prob:.2f} conf={conf:.2f} n={n_arm}", C.Y))
                    wait_for_next_open(TF_M1)
                    cooldown[ativo] = time.time()
                    continue
            else:
                # Warmup mais rigoroso
                if prob < (AI_MIN_PROB - 0.05):  # REDUZIDO de 0.07 para 0.05
                    log.info(paint(f"[❌ IA-SKIP] {ativo} {final_dir} | warmup_prob={prob:.2f} n={n_arm}", C.Y))
                    wait_for_next_open(TF_M1)
                    cooldown[ativo] = time.time()
                    continue

        wait_for_next_open(TF_M1)

        stake = calcular_stake_dinamico(iq, STAKE_FIXA)
        log.info(paint(f"[{ativo}] 💵 Stake: {stake:.2f}", C.B))

        op = enviar_ordem(iq, ativo, final_dir, stake)

        if not op:
            log.error(paint(f"[{ativo}] ❌ Falha ao enviar ordem", C.R))
            cooldown[ativo] = time.time()
            continue

        op_type, op_id = op
        log.info(paint(
            f"[{ativo}] ✅ ORDEM {final_dir} exp={EXP_FIXA}m ({op_type}) stake={stake:.2f}",
            dir_color(final_dir)
        ))

        res = wait_result(iq, op_type, op_id)

        total += 1
        if res > 0:
            wins += 1
            log.info(paint(f"[{ativo}] ✅ WIN {res:.2f}$", C.G))
        elif res < 0:
            log.info(paint(f"[{ativo}] ❌ LOSS {res:.2f}$", C.R))
        else:
            log.info(paint(f"[{ativo}] ⚪ EMPATE {res:.2f}$", C.B))

        if IA_ON:
            ai_update(ativo, setup, res, stats)
            _safe_save_json(AI_STATS_FILE, stats)

        acc = (wins / max(1, total)) * 100.0

        try:
            saldo_atual = float(iq.get_balance())
            lucro_atual = saldo_atual - saldo_inicial
            lucro_percent_atual = (lucro_atual / saldo_inicial) * 100.0
            falta_meta = (saldo_inicial * META_LUCRO_PERCENT / 100.0) - lucro_atual

            if lucro_percent_atual >= 0:
                log.info(paint(f"📊 TRADES={total} WINS={wins} ACC={acc:.2f}%", C.G))
                log.info(paint(f"💰 SALDO={saldo_atual:.2f} | LUCRO=+{lucro_atual:.2f}({lucro_percent_atual:.2f}%) | FALTA={falta_meta:.2f}\n", C.G))
            else:
                log.info(paint(f"📊 TRADES={total} WINS={wins} ACC={acc:.2f}%", C.Y))
                log.info(paint(f"💰 SALDO={saldo_atual:.2f} | PERDA={lucro_atual:.2f}({lucro_percent_atual:.2f}%)\n", C.Y))
        except Exception:
            log.info(f"📊 TRADES={total} WINS={wins} ACC={acc:.2f}%\n")

        cooldown[ativo] = time.time()

if __name__ == "__main__":
    main()
