# -*- coding: utf-8 -*-
"""
WS_AUTO_AI — Pernada B (M1) com:
✅ Candles FECHADOS (evita sinal fora da hora)
✅ Anti-lateral + Anti-esticado
✅ Filtro de SUPORTE/RESISTÊNCIA FORTE (usa >=200 velas e considera várias regiões)
✅ IA online simples (Bayes + UCB) aprendendo com seus próprios resultados (salva em JSON)
✅ Execução real (TURBO -> DIGITAL fallback)

Requisitos:
pip install iqoptionapi pandas numpy
"""

import os
import sys
import time
import math
import json
import logging
import ctypes
import threading
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from iqoptionapi.stable_api import IQ_Option

# Importa analisador de loss
try:
    from loss_analyzer import get_loss_analyzer
    LOSS_ANALYZER_ENABLED = True
    LOSS_ANALYZER = get_loss_analyzer(os.getenv("BACKEND_URL", "http://localhost:8000"))
except ImportError:
    LOSS_ANALYZER_ENABLED = False
    LOSS_ANALYZER = None

# Sistema de aprendizado do Firebase DESABILITADO 
# A IA agora usa APENAS o arquivo JSON local (ws_ai_stats_m1.json)
# O Firebase é usado apenas para ANÁLISE de losses, não para filtrar trades
AI_LEARNING_ENABLED = False
AI_LEARNING = None

def _enable_console_colors():
    if os.name != "nt":
        return
    try:
        import colorama
        colorama.just_fix_windows_console()
        return
    except Exception:
        pass
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass

_enable_console_colors()

# ===================== CONFIG =====================
EMAIL = os.getenv("IQ_EMAIL", "")
SENHA = os.getenv("IQ_PASS", "")
CONTA = os.getenv("IQ_CONTA", "PRACTICE").strip().upper()

# Validação de credenciais
if not EMAIL or not SENHA:
    print("[WS_AUTO_AI] ERRO - Credenciais nao fornecidas via variaveis de ambiente", flush=True)
    print("[WS_AUTO_AI] ERRO - Configure IQ_EMAIL e IQ_PASS no aplicativo", flush=True)
    sys.exit(1)

TF_M1 = 60
DECIDIR_ANTES_FECHAR_SEC = int(os.getenv("WS_DECIDIR_ANTES_FECHAR", "10"))
N_M1 = int(os.getenv("WS_N_M1", "340"))

PAYOUT_MINIMO = int(os.getenv("WS_PAYOUT_MIN", "80"))
NUM_ATIVOS = int(os.getenv("WS_NUM_ATIVOS", "12"))
PAYOUT_REFRESH_SEC = int(os.getenv("WS_PAYOUT_REFRESH", "240"))  # 4 minutos

EXP_FIXA = int(os.getenv("WS_EXP_MIN", "5"))  # FIXO 5 minutos
VALOR_MINIMO = float(os.getenv("WS_VALOR_MINIMO", "3"))
STAKE_FIXA = float(os.getenv("WS_STAKE", "5"))

# ===================== GESTÃO DE BANCA =====================
PERCENT_BANCA = float(os.getenv("WS_PERCENT_BANCA", "1.0"))  # 1% da banca por operação
META_LUCRO_PERCENT = float(os.getenv("WS_META_LUCRO", "1.5"))  # para com 1.5% de lucro
STOP_LOSS_PERCENT = float(os.getenv("WS_STOP_LOSS", "3.0"))  # para com 3% de perda (opcional)
USE_DYNAMIC_STAKE = (os.getenv("WS_DYNAMIC_STAKE", "1").strip() == "1")  # usar % da banca

COOLDOWN_ATIVO = int(os.getenv("WS_COOLDOWN_ATIVO", "20"))  # seg

# ===================== IA (ONLINE) - APRENDIZADO RÁPIDO COM LOSS =====================
# 🎯 APRENDE RÁPIDO: Loss tem peso maior, bloqueia padrões ruins rapidamente
IA_ON = (os.getenv("WS_AI_ON", "1").strip() == "1")  # LIGADO: aprende e melhora
AI_STATS_FILE = os.getenv("WS_AI_FILE", "ws_ai_stats_m1.json")
AI_MIN_SAMPLES = int(os.getenv("WS_AI_MIN_SAMPLES", "3"))    # 🔥 RÁPIDO: 3 trades já decide
AI_MIN_PROB = float(os.getenv("WS_AI_MIN_PROB", "0.48"))     # Equilibrado: 48% mínimo
AI_CONF_MIN = float(os.getenv("WS_AI_CONF_MIN", "0.55"))     # Confiança razoável: 55%
AI_MIN_WINRATE = float(os.getenv("WS_AI_MIN_WINRATE", "0.45"))  # Bloqueia se < 45%
AI_LOSS_WEIGHT = float(os.getenv("WS_AI_LOSS_WEIGHT", "2.5"))  # 🔥 Loss vale 2.5x mais que win

# 🎯 LTA/LTB COM SEGUNDO TOQUE - Estratégia principal
LT_ENABLED = True  # Ativar detecção de LTA/LTB
LT_MIN_TOUCHES = 2  # Mínimo 2 toques para confirmar linha
LT_WICK_REJECTION_MIN = 0.40  # Pavio de rejeição mínimo 40% da vela
LT_BONUS_SCORE = 0.20  # Bônus no score quando toca LT com rejeição

# Confluência OPCIONAL - só bloqueia se muito ruim
MIN_CONFLUENCE_SIGNALS = int(os.getenv("WS_MIN_CONFLUENCE", "2"))  # Mínimo 2 de 6
BLOCK_LOW_CONFLUENCE = False  # NÃO bloqueia - LTA/LTB é suficiente

# ===================== PADRÃO IMPULSO-CORREÇÃO (EQUILIBRADO) =====================
# Impulsos normais, foco em LTA/LTB com segundo toque
IMPULSO_MIN_ATR = float(os.getenv("WS_IMPULSO_MIN_ATR", "0.60"))  # Equilibrado: 0.6 ATR
IMPULSO_JANELA_MIN = int(os.getenv("WS_IMP_JMIN", "3"))  # mínimo 3 velas
IMPULSO_JANELA_MAX = int(os.getenv("WS_IMP_JMAX", "18"))  # máximo 18 velas

PULLBACK_MIN = int(os.getenv("WS_PB_MIN", "2"))  # mínimo 2 velas
PULLBACK_MAX = int(os.getenv("WS_PB_MAX", "6"))  # máximo 6 velas

# RETRAÇÕES FIBONACCI CORRETAS PARA ELLIOTT WAVE:
# - Onda 2: Retrai 50-61.8% da Onda 1 (ideal: 61.8%)
# - Onda 4: Retrai 23.6-38.2% da Onda 3 (ideal: 38.2%)
RETR_MIN = float(os.getenv("WS_RETR_MIN", "0.382"))  # Mínimo 38.2% (Fibonacci)
RETR_MAX = float(os.getenv("WS_RETR_MAX", "0.618"))  # Máximo 61.8% (Fibonacci)

BREAK_MARGIN_ATR = float(os.getenv("WS_BREAK_MARGIN_ATR", "0.05"))  # margem de rompimento
MAX_BREAK_DISTANCE_ATR = float(os.getenv("WS_MAX_BREAK_DIST_ATR", "0.45"))  # distância máxima

# ===================== ANTI-LATERAL (SÓ MERCADO HORRÍVEL) =====================
# Bloqueia APENAS mercado muito lateral/caótico
MIN_EFF_A = float(os.getenv("WS_MIN_EFF_A", "0.30"))  # Equilibrado: 30%

CHOP_LOOKBACK = int(os.getenv("WS_CHOP_LB", "28"))
MAX_COLOR_FLIPS_FRAC = float(os.getenv("WS_MAX_FLIPS", "0.70"))  # Só bloqueia se > 70% choppy
MIN_NET_GROSS_EFF = float(os.getenv("WS_MIN_NETGROSS", "0.15"))  # 15% mínimo

COMP_LOOKBACK = int(os.getenv("WS_COMP_LB", "18"))
MIN_RANGE_ATR = float(os.getenv("WS_MIN_RANGE_ATR", "0.60"))  # Equilibrado: 0.6 ATR

LATE_LOOKBACK = int(os.getenv("WS_LATE_LB", "18"))
MAX_LATE_EXT_ATR = float(os.getenv("WS_MAX_LATE_EXT_ATR", "10.0"))  # Máx 10 ATR

# ===================== QUALIDADE DO GATILHO (FOCO EM PAVIOS DE REJEIÇÃO) =====================
# Aceita velas com pavios de rejeição (mostrando força)
MIN_BODY_FRAC_BREAK = float(os.getenv("WS_MIN_BODY_FRAC", "0.20"))  # Corpo pequeno OK se tem pavio de rejeição
MAX_WICK_AGAINST = float(os.getenv("WS_MAX_WICK_AGAINST", "0.50"))  # Pavio contra até 50%
MIN_WICK_REJECTION = float(os.getenv("WS_MIN_WICK_REJECT", "0.35"))  # Pavio de rejeição mínimo 35%

# ===================== SCORE (EQUILIBRADO) =====================
# Score normal, bônus para LTA/LTB com segundo toque
GATE_MIN_SCORE = float(os.getenv("WS_GATE_MIN", "0.42"))  # Equilibrado: 0.42
GATE_SOFT_SCORE = float(os.getenv("WS_GATE_SOFT", "0.35"))  # Soft: 0.35

# FILTROS DE CONTEXTO (EQUILIBRADOS - FOCO EM LTA/LTB)
# LTA/LTB com segundo toque + pavio de rejeição é suficiente
MIN_CONTEXT_QUALITY = float(os.getenv("WS_MIN_CTX_QUALITY", "0.25"))  # Baixo - LTA/LTB compensa
MIN_ENTRY_CONFIDENCE = float(os.getenv("WS_MIN_ENTRY_CONF", "0.35"))  # Baixo - LTA/LTB compensa
MIN_CONFLUENCE_FOR_WEAK_CONTEXT = float(os.getenv("WS_MIN_CONFL", "0.00"))  # LTA/LTB é a confluência

# 🚨 FILTROS CRÍTICOS (SÓ BLOQUEIA MERCADO HORRÍVEL)
MIN_ALIGNMENT_RATIO = float(os.getenv("WS_MIN_ALIGN", "0.25"))  # Equilibrado
MAX_CONSOLIDATION_CONFIDENCE = float(os.getenv("WS_MAX_CONSOL_CONF", "0.85"))  # Só bloqueia se > 85% consolidação
BLOCK_CALL_NEAR_RESISTANCE = False  # DESLIGADO - pode entrar CALL em resistência se LTA confirma
BLOCK_PUT_NEAR_SUPPORT = False  # DESLIGADO - pode entrar PUT em suporte se LTB confirma
MIN_MOMENTUM_ALIGNMENT = False  # DESLIGADO - LTA/LTB é suficiente

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
SR_BLOCK_DIST_ATR = float(os.getenv("WS_SR_BLOCK_ATR", "0.45"))  # REDUZIDO: mais rigoroso, bloqueia mais perto

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
    return now - (now % tf) - 1  # garante candle fechado

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

        # precisa ser grande o bastante pro SR + filtros
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

def calculate_atr_trailing_stops(df: pd.DataFrame, period: int = 14, multiplier: float = 2.0, high_low: bool = False) -> Tuple[float, int]:
    """
    Calcula ATR Trailing Stops para filtro de tendência.
    Retorna (atr_ts_value, position) onde:
    - position = 1: tendência de alta (pode CALL, não PUT)
    - position = -1: tendência de baixa (pode PUT, não CALL)
    
    Se preço > ATR TS: position = 1 (CALL permitido)
    Se preço < ATR TS: position = -1 (PUT permitido)
    """
    if len(df) < period + 1:
        return 0.0, 0
    
    # Calcular ATR usando RMA (Running Moving Average)
    df_copy = df.copy()
    h = df_copy["high"].values
    l = df_copy["low"].values
    c = df_copy["close"].values
    
    # True Range
    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1]))
    )
    
    # RMA (EMA com alpha = 1/period)
    alpha = 1.0 / period
    atr_values = np.zeros(len(tr))
    atr_values[0] = tr[0]
    for i in range(1, len(tr)):
        atr_values[i] = alpha * tr[i] + (1 - alpha) * atr_values[i-1]
    
    atr_current = atr_values[-1] * multiplier
    
    # Usar high/low ou close
    if high_low:
        h_ref = h[1:]
        l_ref = l[1:]
    else:
        h_ref = c[1:]
        l_ref = c[1:]
    
    # Calcular ATR Trailing Stop recursivamente
    atr_ts = np.zeros(len(c) - 1)
    pos = np.zeros(len(c) - 1, dtype=int)
    
    # Inicializar primeiro valor
    if c[1] > c[0]:
        atr_ts[0] = h_ref[0] - atr_current
        pos[0] = 1
    else:
        atr_ts[0] = l_ref[0] + atr_current
        pos[0] = -1
    
    # Calcular para o resto
    for i in range(1, len(atr_ts)):
        prev_atr_ts = atr_ts[i-1]
        curr_close = c[i+1]
        prev_close = c[i]
        
        # Lógica do ATR TS
        if curr_close > prev_atr_ts and prev_close > prev_atr_ts:
            # Tendência de alta continua
            atr_ts[i] = max(prev_atr_ts, h_ref[i] - atr_current)
        elif curr_close < prev_atr_ts and prev_close < prev_atr_ts:
            # Tendência de baixa continua
            atr_ts[i] = min(prev_atr_ts, l_ref[i] + atr_current)
        elif curr_close > prev_atr_ts:
            # Mudou para alta
            atr_ts[i] = h_ref[i] - atr_current
        else:
            # Mudou para baixa
            atr_ts[i] = l_ref[i] + atr_current
        
        # Determinar posição (tendência)
        if prev_close < prev_atr_ts and curr_close > atr_ts[i]:
            pos[i] = 1  # Virou alta
        elif prev_close > prev_atr_ts and curr_close < atr_ts[i]:
            pos[i] = -1  # Virou baixa
        else:
            pos[i] = pos[i-1]  # Mantém posição anterior
    
    return float(atr_ts[-1]), int(pos[-1])

# ===================== ESTRUTURA DE MERCADO - TENDENCIA MAIOR =====================
def detect_market_structure_trend(df: pd.DataFrame, lookback: int = 100, swing_window: int = 5) -> Dict[str, Any]:
    """
    Detecta a tendencia MAIOR do mercado usando estrutura de topos e fundos.

    Analisa os ultimos 'lookback' candles para identificar:
    - Swing Highs (topos) e Swing Lows (fundos)
    - Higher Highs + Higher Lows = UPTREND (so permite CALL)
    - Lower Highs + Lower Lows = DOWNTREND (so permite PUT)
    - Misto = LATERAL (sem entrada)

    Retorna:
    - trend: "UP", "DOWN", "LATERAL"
    - strength: 0.0 a 1.0 (forca da tendencia)
    - allowed_direction: "CALL", "PUT", "NONE"
    - swing_highs: lista de (indice, preco)
    - swing_lows: lista de (indice, preco)
    """
    if len(df) < lookback:
        return {"trend": "UNKNOWN", "strength": 0.0, "allowed_direction": "NONE", "swing_highs": [], "swing_lows": []}

    sub = df.tail(lookback)
    highs = sub["high"].to_numpy(float)
    lows = sub["low"].to_numpy(float)
    closes = sub["close"].to_numpy(float)

    swing_highs = []  # (indice, preco)
    swing_lows = []   # (indice, preco)

    # Encontra Swing Highs (topos locais)
    for i in range(swing_window, len(highs) - swing_window):
        is_swing_high = True
        for j in range(1, swing_window + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs.append((i, highs[i]))

    # Encontra Swing Lows (fundos locais)
    for i in range(swing_window, len(lows) - swing_window):
        is_swing_low = True
        for j in range(1, swing_window + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break
        if is_swing_low:
            swing_lows.append((i, lows[i]))

    # Precisa de pelo menos 3 swings para determinar tendencia
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return {"trend": "LATERAL", "strength": 0.0, "allowed_direction": "NONE", "swing_highs": swing_highs, "swing_lows": swing_lows}

    # Analisa os ultimos 4 swing highs
    recent_highs = [sh[1] for sh in swing_highs[-4:]]
    higher_highs = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1])
    lower_highs = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] < recent_highs[i-1])

    # Analisa os ultimos 4 swing lows
    recent_lows = [sl[1] for sl in swing_lows[-4:]]
    higher_lows = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] > recent_lows[i-1])
    lower_lows = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] < recent_lows[i-1])

    # Determina tendencia
    total_comparisons = min(len(recent_highs) - 1, len(recent_lows) - 1)
    if total_comparisons == 0:
        return {"trend": "LATERAL", "strength": 0.0, "allowed_direction": "NONE", "swing_highs": swing_highs, "swing_lows": swing_lows}

    # UPTREND: Higher Highs AND Higher Lows
    up_score = (higher_highs + higher_lows) / (total_comparisons * 2)

    # DOWNTREND: Lower Highs AND Lower Lows
    down_score = (lower_highs + lower_lows) / (total_comparisons * 2)

    # Adiciona confirmacao de preco atual vs swing points
    current_price = closes[-1]
    last_swing_high = swing_highs[-1][1] if swing_highs else current_price
    last_swing_low = swing_lows[-1][1] if swing_lows else current_price

    # Se preco atual esta abaixo do ultimo swing high E fazendo lower highs/lows = DOWNTREND forte
    if current_price < last_swing_high and down_score >= 0.5:
        down_score += 0.15

    # Se preco atual esta acima do ultimo swing low E fazendo higher highs/lows = UPTREND forte
    if current_price > last_swing_low and up_score >= 0.5:
        up_score += 0.15

    # Decisao final - retorna informacao para o AGENTE IA decidir
    if up_score >= 0.55 and up_score > down_score + 0.12:
        return {
            "trend": "UP",
            "strength": min(1.0, up_score),
            "suggested_direction": "CALL",  # Sugestao para o agente
            "swing_highs": swing_highs,
            "swing_lows": swing_lows,
            "up_score": up_score,
            "down_score": down_score
        }
    elif down_score >= 0.55 and down_score > up_score + 0.12:
        return {
            "trend": "DOWN",
            "strength": min(1.0, down_score),
            "suggested_direction": "PUT",  # Sugestao para o agente
            "swing_highs": swing_highs,
            "swing_lows": swing_lows,
            "up_score": up_score,
            "down_score": down_score
        }
    else:
        # Mercado misto - agente deve analisar com cuidado
        return {
            "trend": "MIXED",
            "strength": max(up_score, down_score),
            "suggested_direction": "ANALYZE",  # Agente decide
            "swing_highs": swing_highs,
            "swing_lows": swing_lows,
            "up_score": up_score,
            "down_score": down_score
        }

# ===================== LINHA DE TENDÊNCIA (LTA/LTB) =====================
def detect_trendline(df: pd.DataFrame, lookback: int, direction: str) -> Optional[Tuple[float, float]]:
    """
    Detecta linha de tendência (LTA para alta, LTB para baixa).
    Retorna (slope, intercept) ou None se não encontrar.
    """
    if len(df) < lookback:
        return None

    sub = df.tail(lookback)

    if direction == "CALL":
        # LTA - conecta mínimas ascendentes (suporte)
        pivots = []
        lows = sub["low"].to_numpy(float)

        # Encontra pivôs de baixa (mínimas locais)
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                pivots.append((i, lows[i]))

        # Precisa de pelo menos 2 pivôs ascendentes
        if len(pivots) < 2:
            return None

        # Verifica se há tendência ascendente nos pivôs
        if pivots[-1][1] <= pivots[0][1]:
            return None

        # Calcula linha de tendência
        x = np.array([p[0] for p in pivots])
        y = np.array([p[1] for p in pivots])
        slope, intercept = np.polyfit(x, y, 1)

        # Só aceita se slope positivo (ascendente)
        if slope <= 0:
            return None

        return (float(slope), float(intercept))

    else:  # PUT
        # LTB - conecta máximas descendentes (resistência)
        pivots = []
        highs = sub["high"].to_numpy(float)

        # Encontra pivôs de alta (máximas locais)
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                pivots.append((i, highs[i]))

        # Precisa de pelo menos 2 pivôs descendentes
        if len(pivots) < 2:
            return None

        # Verifica se há tendência descendente nos pivôs
        if pivots[-1][1] >= pivots[0][1]:
            return None

        # Calcula linha de tendência
        x = np.array([p[0] for p in pivots])
        y = np.array([p[1] for p in pivots])
        slope, intercept = np.polyfit(x, y, 1)

        # Só aceita se slope negativo (descendente)
        if slope >= 0:
            return None

        return (float(slope), float(intercept))

# ===================== LTA/LTB COM SEGUNDO TOQUE + PAVIO DE REJEIÇÃO =====================
def detect_lt_second_touch(df: pd.DataFrame, atr_val: float, lookback: int = 50) -> Dict[str, Any]:
    """
    🎯 ESTRATÉGIA PRINCIPAL: Detecta LTA/LTB com SEGUNDO TOQUE e pavio de rejeição
    
    - Identifica a linha de tendência (LTA para alta, LTB para baixa)
    - Conta quantas vezes tocou a linha
    - Verifica se tem pavio de rejeição no último toque (força)
    - Olha para trás se o movimento anterior foi forte
    
    Retorna:
    - has_lt: True se encontrou linha de tendência válida
    - lt_type: "LTA" ou "LTB"
    - touches: número de toques na linha
    - has_rejection: True se tem pavio de rejeição no último toque
    - rejection_strength: força do pavio de rejeição (0-1)
    - signal: "CALL" para LTA com rejeição, "PUT" para LTB com rejeição, None se inválido
    - score_bonus: bônus para adicionar ao score (0 a 0.25)
    - reason: descrição do sinal encontrado
    """
    result = {
        "has_lt": False,
        "lt_type": None,
        "touches": 0,
        "has_rejection": False,
        "rejection_strength": 0.0,
        "signal": None,
        "score_bonus": 0.0,
        "reason": ""
    }
    
    if len(df) < lookback or atr_val <= 0:
        return result
    
    sub = df.tail(lookback)
    closes = sub["close"].to_numpy(float)
    highs = sub["high"].to_numpy(float)
    lows = sub["low"].to_numpy(float)
    opens = sub["open"].to_numpy(float)
    n = len(sub)
    
    # ⚠️ FILTRO CRÍTICO: VERIFICA POSIÇÃO NO RANGE
    # Só entra se estiver no EXTREMO (suporte ou resistência), não no MEIO!
    range_high = float(np.max(highs))
    range_low = float(np.min(lows))
    range_size = range_high - range_low
    
    if range_size < 1e-9:
        return result
    
    # Última vela (potencial toque)
    last_idx = n - 1
    last_close = closes[last_idx]
    last_high = highs[last_idx]
    last_low = lows[last_idx]
    
    # Calcula posição no range (0 = suporte, 1 = resistência, 0.5 = meio)
    position_in_range = (last_close - range_low) / range_size
    
    # 🚫 BLOQUEIA SE ESTÁ NO MEIO DO RANGE (entre 30% e 70%)
    # CALL só permitido se está perto do suporte (< 35%)
    # PUT só permitido se está perto da resistência (> 65%)
    MIN_EXTREME_POSITION = 0.35  # Deve estar nos 35% extremos do range
    last_open = opens[last_idx]
    last_range = last_high - last_low
    
    if last_range < 1e-9:
        return result
    
    # Calcula corpo e pavios da última vela
    last_body = abs(last_close - last_open)
    last_body_top = max(last_close, last_open)
    last_body_bottom = min(last_close, last_open)
    last_upper_wick = last_high - last_body_top
    last_lower_wick = last_body_bottom - last_low
    
    # ===================== DETECTA LTA (LINHA DE TENDÊNCIA DE ALTA) =====================
    # LTA conecta MÍNIMAS ascendentes - suporte dinâmico
    lta_pivots = []
    for i in range(2, n - 2):
        # Pivô de baixa: mínima local
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            lta_pivots.append((i, lows[i]))
    
    # Verifica LTA
    if len(lta_pivots) >= 2:
        # Verifica se pivôs são ascendentes
        ascending = all(lta_pivots[j][1] < lta_pivots[j+1][1] for j in range(len(lta_pivots)-1))
        
        if ascending or (lta_pivots[-1][1] > lta_pivots[0][1]):
            # Calcula linha de tendência
            x_pts = np.array([p[0] for p in lta_pivots])
            y_pts = np.array([p[1] for p in lta_pivots])
            slope, intercept = np.polyfit(x_pts, y_pts, 1)
            
            if slope > 0:  # LTA válida (ascendente)
                # Valor da LTA na última vela
                lt_value = slope * last_idx + intercept
                
                # Conta toques na LTA (mínimas que tocaram a linha)
                touches = 0
                tolerance = atr_val * 0.35  # 35% do ATR de tolerância
                for i in range(n):
                    expected = slope * i + intercept
                    if abs(lows[i] - expected) <= tolerance:
                        touches += 1
                
                # Verifica se última vela toca a LTA
                distance_to_lt = abs(last_low - lt_value) / atr_val
                
                if distance_to_lt <= 0.4 and touches >= LT_MIN_TOUCHES:
                    # 🚫 FILTRO CRÍTICO: CALL só se está PERTO DO SUPORTE!
                    # Se está no meio do range (> 35%), NÃO entra CALL
                    if position_in_range > MIN_EXTREME_POSITION:
                        # Está no meio ou perto da resistência - NÃO FAZ CALL
                        result["reason"] = f"⚠️ BLOQUEIO: CALL mas preço no meio do range ({position_in_range:.0%})"
                        return result
                    
                    # TOQUE CONFIRMADO! Verifica pavio de rejeição
                    # Para CALL em LTA: pavio INFERIOR grande = rejeição da queda
                    rejection_strength = last_lower_wick / last_range
                    has_rejection = rejection_strength >= LT_WICK_REJECTION_MIN
                    
                    # Verifica se fechou de alta (corpo verde)
                    is_bullish = last_close > last_open
                    
                    # Olha para trás: movimento anterior foi forte?
                    lookback_momentum = 8
                    if last_idx >= lookback_momentum:
                        prev_move = closes[last_idx - 1] - closes[last_idx - lookback_momentum]
                        momentum_was_down = prev_move < -atr_val * 0.3  # Veio de queda
                    else:
                        momentum_was_down = False
                    
                    if has_rejection and (is_bullish or rejection_strength >= 0.50):
                        result["has_lt"] = True
                        result["lt_type"] = "LTA"
                        result["touches"] = touches
                        result["has_rejection"] = True
                        result["rejection_strength"] = rejection_strength
                        result["signal"] = "CALL"
                        result["score_bonus"] = LT_BONUS_SCORE if touches >= 3 else LT_BONUS_SCORE * 0.7
                        result["reason"] = f"LTA {touches}° toque | Pavio rejeição {rejection_strength:.0%} | {'✅ Alta' if is_bullish else '➡️ Rejeitou'} | Pos={position_in_range:.0%}(suporte)"
                        
                        # Bônus extra se veio de queda e rejeitou (pullback no suporte)
                        if momentum_was_down:
                            result["score_bonus"] += 0.05
                            result["reason"] += " | Pullback confirmado"
                        
                        return result
    
    # ===================== DETECTA LTB (LINHA DE TENDÊNCIA DE BAIXA) =====================
    # LTB conecta MÁXIMAS descendentes - resistência dinâmica
    ltb_pivots = []
    for i in range(2, n - 2):
        # Pivô de alta: máxima local
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            ltb_pivots.append((i, highs[i]))
    
    # Verifica LTB
    if len(ltb_pivots) >= 2:
        # Verifica se pivôs são descendentes
        descending = all(ltb_pivots[j][1] > ltb_pivots[j+1][1] for j in range(len(ltb_pivots)-1))
        
        if descending or (ltb_pivots[-1][1] < ltb_pivots[0][1]):
            # Calcula linha de tendência
            x_pts = np.array([p[0] for p in ltb_pivots])
            y_pts = np.array([p[1] for p in ltb_pivots])
            slope, intercept = np.polyfit(x_pts, y_pts, 1)
            
            if slope < 0:  # LTB válida (descendente)
                # Valor da LTB na última vela
                lt_value = slope * last_idx + intercept
                
                # Conta toques na LTB (máximas que tocaram a linha)
                touches = 0
                tolerance = atr_val * 0.35
                for i in range(n):
                    expected = slope * i + intercept
                    if abs(highs[i] - expected) <= tolerance:
                        touches += 1
                
                # Verifica se última vela toca a LTB
                distance_to_lt = abs(last_high - lt_value) / atr_val
                
                if distance_to_lt <= 0.4 and touches >= LT_MIN_TOUCHES:
                    # 🚫 FILTRO CRÍTICO: PUT só se está PERTO DA RESISTÊNCIA!
                    # Se está no meio do range (< 65%), NÃO entra PUT
                    if position_in_range < (1.0 - MIN_EXTREME_POSITION):
                        # Está no meio ou perto do suporte - NÃO FAZ PUT
                        result["reason"] = f"⚠️ BLOQUEIO: PUT mas preço no meio do range ({position_in_range:.0%})"
                        return result
                    
                    # TOQUE CONFIRMADO! Verifica pavio de rejeição
                    # Para PUT em LTB: pavio SUPERIOR grande = rejeição da alta
                    rejection_strength = last_upper_wick / last_range
                    has_rejection = rejection_strength >= LT_WICK_REJECTION_MIN
                    
                    # Verifica se fechou de baixa (corpo vermelho)
                    is_bearish = last_close < last_open
                    
                    # Olha para trás: movimento anterior foi forte?
                    lookback_momentum = 8
                    if last_idx >= lookback_momentum:
                        prev_move = closes[last_idx - 1] - closes[last_idx - lookback_momentum]
                        momentum_was_up = prev_move > atr_val * 0.3  # Veio de alta
                    else:
                        momentum_was_up = False
                    
                    if has_rejection and (is_bearish or rejection_strength >= 0.50):
                        result["has_lt"] = True
                        result["lt_type"] = "LTB"
                        result["touches"] = touches
                        result["has_rejection"] = True
                        result["rejection_strength"] = rejection_strength
                        result["signal"] = "PUT"
                        result["score_bonus"] = LT_BONUS_SCORE if touches >= 3 else LT_BONUS_SCORE * 0.7
                        result["reason"] = f"LTB {touches}° toque | Pavio rejeição {rejection_strength:.0%} | {'✅ Baixa' if is_bearish else '➡️ Rejeitou'} | Pos={position_in_range:.0%}(resistência)"
                        
                        # Bônus extra se veio de alta e rejeitou (pullback na resistência)
                        if momentum_was_up:
                            result["score_bonus"] += 0.05
                            result["reason"] += " | Pullback confirmado"
                        
                        return result
    
    return result

def check_trendline_confluence(df: pd.DataFrame, pb_high: float, pb_low: float,
                                direction: str, atr_val: float) -> Dict[str, Any]:
    """
    Verifica se o pullback tocou/respeitou a linha de tendência.
    Retorna score de confluência com a LT.
    """
    # Tenta detectar linha de tendência nas últimas 30-50 velas
    trendline = detect_trendline(df.tail(50), 50, direction)

    if trendline is None:
        return {"has_trendline": False, "confluence": 0.0, "distance": 999.0}

    slope, intercept = trendline

    # Calcula valor da LT na posição do pullback (última vela)
    x_pb = len(df.tail(50)) - 1  # posição do pullback
    lt_value = slope * x_pb + intercept

    if direction == "CALL":
        # Para CALL, pullback deve tocar a LTA (suporte)
        # Verifica se a mínima do pullback está próxima da LTA
        distance = abs(pb_low - lt_value) / max(atr_val, 1e-9)

        # Se tocou a LTA (distância < 0.3 ATR), excelente confluência
        if distance < 0.3:
            return {"has_trendline": True, "confluence": 1.0, "distance": distance, "lt_value": lt_value}
        # Próximo mas não tocou
        elif distance < 0.6:
            return {"has_trendline": True, "confluence": 0.6, "distance": distance, "lt_value": lt_value}
        # Muito longe da LTA
        else:
            return {"has_trendline": True, "confluence": 0.2, "distance": distance, "lt_value": lt_value}

    else:  # PUT
        # Para PUT, pullback deve tocar a LTB (resistência)
        # Verifica se a máxima do pullback está próxima da LTB
        distance = abs(pb_high - lt_value) / max(atr_val, 1e-9)

        # Se tocou a LTB
        if distance < 0.3:
            return {"has_trendline": True, "confluence": 1.0, "distance": distance, "lt_value": lt_value}
        elif distance < 0.6:
            return {"has_trendline": True, "confluence": 0.6, "distance": distance, "lt_value": lt_value}
        else:
            return {"has_trendline": True, "confluence": 0.2, "distance": distance, "lt_value": lt_value}

def check_sr_confluence(df: pd.DataFrame, pb_low: float, pb_high: float, direction: str, atr_val: float) -> Dict[str, Any]:
    """
    Verifica se o pullback tocou uma zona de Suporte/Resistencia.
    Para CALL: pullback deve ter tocado um suporte
    Para PUT: pullback deve ter tocado uma resistencia

    Retorna:
    - has_sr: True se tocou zona SR
    - confluence: 0.0 a 1.0 (forca da confluencia)
    - sr_level: nivel do SR tocado
    """
    res, sup = strong_sr_levels_last200(df, atr_val)
    atr_safe = max(atr_val, 1e-9)

    if direction == "CALL":
        # Para CALL, pullback deve ter tocado SUPORTE (pb_low proximo de um suporte)
        if not sup:
            return {"has_sr": False, "confluence": 0.0, "sr_level": 0.0}

        for lvl, touches in sup:
            distance = abs(pb_low - lvl) / atr_safe
            # Se o fundo do pullback esta dentro de 0.5 ATR do suporte = confluencia forte
            if distance <= 0.3:
                return {"has_sr": True, "confluence": 1.0, "sr_level": lvl, "touches": touches, "distance": distance}
            elif distance <= 0.6:
                return {"has_sr": True, "confluence": 0.7, "sr_level": lvl, "touches": touches, "distance": distance}
            elif distance <= 1.0:
                return {"has_sr": True, "confluence": 0.4, "sr_level": lvl, "touches": touches, "distance": distance}

        return {"has_sr": False, "confluence": 0.0, "sr_level": 0.0}

    else:  # PUT
        # Para PUT, pullback deve ter tocado RESISTENCIA (pb_high proximo de uma resistencia)
        if not res:
            return {"has_sr": False, "confluence": 0.0, "sr_level": 0.0}

        for lvl, touches in res:
            distance = abs(pb_high - lvl) / atr_safe
            # Se o topo do pullback esta dentro de 0.5 ATR da resistencia = confluencia forte
            if distance <= 0.3:
                return {"has_sr": True, "confluence": 1.0, "sr_level": lvl, "touches": touches, "distance": distance}
            elif distance <= 0.6:
                return {"has_sr": True, "confluence": 0.7, "sr_level": lvl, "touches": touches, "distance": distance}
            elif distance <= 1.0:
                return {"has_sr": True, "confluence": 0.4, "sr_level": lvl, "touches": touches, "distance": distance}

        return {"has_sr": False, "confluence": 0.0, "sr_level": 0.0}

# ===================== AGENTE IA - ANÁLISE CONTEXTUAL INTELIGENTE =====================
def ai_agent_analyze(df: pd.DataFrame, setup: Dict[str, Any], proposed_direction: str,
                     market_trend: str, suggested_direction: str, trend_strength: float,
                     up_score: float, down_score: float, atr_val: float) -> Dict[str, Any]:
    """
    🤖 AGENTE IA - Analisa o contexto COMPLETO das últimas 100 velas e decide:

    1. A Pernada B está correta? (impulso + pullback válidos)
    2. A direção proposta faz sentido com o contexto?
    3. Há confluência suficiente? (SR + LT + Tendência)
    4. Qual a probabilidade de WIN?

    O Agente pode:
    - APROVAR o trade na direção proposta
    - REJEITAR o trade (não opera)
    - SOBRESCREVER a direção (se detectar erro)

    Retorna:
    - approved: bool
    - reason: str
    - confidence: float (0-1)
    - override_direction: str (opcional, se mudar direção)
    """

    # Extrai dados do setup
    score = float(setup.get("score", 0.0))
    has_lt = bool(setup.get("has_lt", False))
    lt_confluence = float(setup.get("lt_confluence", 0.0))
    entry_confidence = float(setup.get("entry_confidence", 0.0))
    market_quality = float(setup.get("market_quality", 0.0))

    # Verifica confluência com SR
    pb_low = float(setup.get("pb_low", 0.0))
    pb_high = float(setup.get("pb_high", 0.0))
    sr_check = check_sr_confluence(df, pb_low, pb_high, proposed_direction, atr_val)
    has_sr = sr_check.get("has_sr", False)
    sr_confluence = sr_check.get("confluence", 0.0)

    # Verifica LT
    lt_check = check_trendline_confluence(df, pb_high, pb_low, proposed_direction, atr_val)
    has_trendline = lt_check.get("has_trendline", False)

    # ===================== ANÁLISE CONTEXTUAL DAS 100 VELAS =====================
    sub = df.tail(100)
    closes = sub["close"].to_numpy(float)
    highs = sub["high"].to_numpy(float)
    lows = sub["low"].to_numpy(float)

    # Calcula tendência de curto prazo (últimas 20 velas)
    short_trend = "NEUTRAL"
    if len(closes) >= 20:
        short_ma = np.mean(closes[-20:])
        long_ma = np.mean(closes[-50:]) if len(closes) >= 50 else short_ma
        current_price = closes[-1]

        if current_price > short_ma > long_ma:
            short_trend = "UP"
        elif current_price < short_ma < long_ma:
            short_trend = "DOWN"

    # Conta candles verdes vs vermelhos nas últimas 30 velas
    green_count = 0
    red_count = 0
    for i in range(-30, 0):
        if i < -len(closes):
            continue
        if closes[i] > sub["open"].iloc[i]:
            green_count += 1
        else:
            red_count += 1

    momentum_ratio = green_count / max(1, green_count + red_count)

    # ===================== DECISÃO DO AGENTE =====================
    confidence = 0.5  # Começa neutro
    reasons = []

    # 1. CONFLUÊNCIA: Deve ter pelo menos 1 (SR ou LT)
    has_any_confluence = has_sr or has_trendline or has_lt
    if has_any_confluence:
        confidence += 0.15
        if has_sr:
            reasons.append("SR")
        if has_trendline or has_lt:
            reasons.append("LT")
    else:
        # Sem confluência mas score alto ainda pode passar
        if score >= 0.75:
            confidence += 0.05
            reasons.append("HighScore")
        else:
            return {
                "approved": False,
                "reason": "Sem confluência (SR/LT)",
                "confidence": 0.3
            }

    # 2. TENDÊNCIA: Verifica alinhamento
    trend_aligned = False

    if proposed_direction == "CALL":
        # CALL precisa de tendência de alta ou reversão de baixa com suporte
        if market_trend == "UP" or suggested_direction == "CALL":
            trend_aligned = True
            confidence += 0.15
            reasons.append(f"Trend={market_trend}")
        elif market_trend == "MIXED" and has_sr and momentum_ratio > 0.45:
            # Mercado misto mas tem suporte e momentum não é totalmente negativo
            trend_aligned = True
            confidence += 0.08
            reasons.append("MixedOK")
        elif market_trend == "DOWN":
            # CALL em downtrend - precisa de forte suporte
            if has_sr and sr_confluence >= 0.7:
                trend_aligned = True
                confidence += 0.05
                reasons.append("ReversalSR")
            else:
                return {
                    "approved": False,
                    "reason": f"CALL contra tendência DOWN (up={up_score:.2f} down={down_score:.2f})",
                    "confidence": 0.25
                }
    else:  # PUT
        # PUT precisa de tendência de baixa ou reversão de alta com resistência
        if market_trend == "DOWN" or suggested_direction == "PUT":
            trend_aligned = True
            confidence += 0.15
            reasons.append(f"Trend={market_trend}")
        elif market_trend == "MIXED" and has_sr and momentum_ratio < 0.55:
            # Mercado misto mas tem resistência e momentum não é totalmente positivo
            trend_aligned = True
            confidence += 0.08
            reasons.append("MixedOK")
        elif market_trend == "UP":
            # PUT em uptrend - precisa de forte resistência
            if has_sr and sr_confluence >= 0.7:
                trend_aligned = True
                confidence += 0.05
                reasons.append("ReversalSR")
            else:
                return {
                    "approved": False,
                    "reason": f"PUT contra tendência UP (up={up_score:.2f} down={down_score:.2f})",
                    "confidence": 0.25
                }

    # 3. QUALIDADE DO SETUP
    if score >= 0.80:
        confidence += 0.10
        reasons.append("A+")
    elif score >= 0.70:
        confidence += 0.05
        reasons.append("A")

    if entry_confidence >= 0.60:
        confidence += 0.05

    # 4. MOMENTUM de curto prazo
    if proposed_direction == "CALL" and short_trend == "UP":
        confidence += 0.05
    elif proposed_direction == "PUT" and short_trend == "DOWN":
        confidence += 0.05

    # Limita confiança máxima
    confidence = min(0.95, confidence)

    # Decisão final
    reason_str = "+".join(reasons) if reasons else "Base"

    if confidence >= 0.55:
        return {
            "approved": True,
            "reason": reason_str,
            "confidence": confidence
        }
    else:
        return {
            "approved": False,
            "reason": f"Confiança baixa ({confidence:.0%}): {reason_str}",
            "confidence": confidence
        }

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

# ===================== S/R FORTE (múltiplas regiões) =====================
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
    """
    Retorna (resistencias, suportes) com base nas últimas SR_LOOKBACK velas (>=200).
    """
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
    """
    Bloqueia CALL perto de resistência forte acima.
    Bloqueia PUT perto de suporte forte abaixo.
    Considera as SR_CHECK_NEAR regiões mais próximas e as SR_TOP_LEVELS mais fortes.
    """
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

def check_approaching_sr(df_m1: pd.DataFrame, atr_val: float, direction: str) -> Optional[str]:
    """
    FILTRO DE TIMING: Evita entradas quando preço está SE APROXIMANDO de S/R forte.

    - Para PUT: Se preço está SUBINDO em direção a uma RESISTÊNCIA forte, espera tocar/rejeitar
    - Para CALL: Se preço está DESCENDO em direção a um SUPORTE forte, espera tocar/rejeitar

    Isso melhora o timing evitando entradas prematuras.
    """
    if len(df_m1) < 10:
        return None

    res, sup = strong_sr_levels_last200(df_m1, atr_val)
    price = float(df_m1["close"].iloc[-1])
    atr_safe = max(atr_val, 1e-9)

    # Calcula momentum das últimas 3-5 velas
    recent = df_m1.tail(5)
    momentum = float(recent["close"].iloc[-1] - recent["close"].iloc[0])
    momentum_atr = momentum / atr_safe

    if direction == "PUT":
        # Para PUT: verifica se preço está SUBINDO em direção a uma resistência
        if momentum_atr > 0.3 and res:  # Momentum positivo significativo
            above = [(lvl, t) for (lvl, t) in res if lvl > price]
            if above:
                # Pega resistência mais próxima acima
                nearest_res = min(above, key=lambda x: x[0] - price)
                dist_to_res = (nearest_res[0] - price) / atr_safe
                touches = nearest_res[1]

                # Se está a menos de 0.8 ATR da resistência e subindo forte
                if dist_to_res < 0.8 and touches >= 2:
                    return f"⏳aguardar_toque_RES(dist={dist_to_res:.2f}ATR,toques={touches},mom={momentum_atr:.2f})"

    elif direction == "CALL":
        # Para CALL: verifica se preço está DESCENDO em direção a um suporte
        if momentum_atr < -0.3 and sup:  # Momentum negativo significativo
            below = [(lvl, t) for (lvl, t) in sup if lvl < price]
            if below:
                # Pega suporte mais próximo abaixo
                nearest_sup = max(below, key=lambda x: x[0])
                dist_to_sup = (price - nearest_sup[0]) / atr_safe
                touches = nearest_sup[1]

                # Se está a menos de 0.8 ATR do suporte e descendo forte
                if dist_to_sup < 0.8 and touches >= 2:
                    return f"⏳aguardar_toque_SUP(dist={dist_to_sup:.2f}ATR,toques={touches},mom={momentum_atr:.2f})"

    return None

# ===================== FILTRO: SÓ ENTRA PERTO DE S/R OU LT (NÃO NO MEIO!) =====================
def check_price_at_key_level(df_m1: pd.DataFrame, atr_val: float, direction: str) -> Dict[str, Any]:
    """
    🎯 FILTRO CRÍTICO: Só permite entrada quando o preço está PERTO de um nível importante!
    
    - Para CALL: preço deve estar PERTO de um suporte ou LTA (não no meio do caminho)
    - Para PUT: preço deve estar PERTO de uma resistência ou LTB (não no meio)
    
    Retorna:
    - at_key_level: True se está perto de um nível importante
    - level_type: "SUPORTE", "RESISTÊNCIA", "LTA", "LTB" ou None
    - distance_atr: distância até o nível em ATRs
    - reason: descrição do nível
    """
    result = {
        "at_key_level": False,
        "level_type": None,
        "distance_atr": 999.0,
        "reason": "no_meio_do_caminho"
    }
    
    if len(df_m1) < 20 or atr_val <= 0:
        return result
    
    price = float(df_m1["close"].iloc[-1])
    low = float(df_m1["low"].iloc[-1])
    high = float(df_m1["high"].iloc[-1])
    atr_safe = max(atr_val, 1e-9)
    
    # Distância máxima para considerar "perto" de um nível
    MAX_DISTANCE_ATR = 0.5  # Deve estar a menos de 0.5 ATR do nível
    
    # 1. VERIFICA S/R
    res, sup = strong_sr_levels_last200(df_m1, atr_val)
    
    if direction == "CALL":
        # Para CALL: deve estar perto de um SUPORTE
        if sup:
            for lvl, touches in sup:
                dist = abs(low - lvl) / atr_safe
                if dist < MAX_DISTANCE_ATR and touches >= 2:
                    result["at_key_level"] = True
                    result["level_type"] = "SUPORTE"
                    result["distance_atr"] = dist
                    result["reason"] = f"✅perto_SUP({lvl:.5f},dist={dist:.2f}ATR,toques={touches})"
                    return result
        
        # 2. Verifica LTA
        lt_check = detect_lt_second_touch(df_m1, atr_val, lookback=50)
        if lt_check.get("has_lt") and lt_check.get("lt_type") == "LTA":
            result["at_key_level"] = True
            result["level_type"] = "LTA"
            result["distance_atr"] = 0.0  # Já está tocando
            result["reason"] = f"✅{lt_check.get('reason', 'LTA')}"
            return result
    
    elif direction == "PUT":
        # Para PUT: deve estar perto de uma RESISTÊNCIA
        if res:
            for lvl, touches in res:
                dist = abs(high - lvl) / atr_safe
                if dist < MAX_DISTANCE_ATR and touches >= 2:
                    result["at_key_level"] = True
                    result["level_type"] = "RESISTÊNCIA"
                    result["distance_atr"] = dist
                    result["reason"] = f"✅perto_RES({lvl:.5f},dist={dist:.2f}ATR,toques={touches})"
                    return result
        
        # 2. Verifica LTB
        lt_check = detect_lt_second_touch(df_m1, atr_val, lookback=50)
        if lt_check.get("has_lt") and lt_check.get("lt_type") == "LTB":
            result["at_key_level"] = True
            result["level_type"] = "LTB"
            result["distance_atr"] = 0.0  # Já está tocando
            result["reason"] = f"✅{lt_check.get('reason', 'LTB')}"
            return result
    
    # Se chegou aqui, está no MEIO DO CAMINHO - não é bom!
    result["reason"] = "🚫no_meio(longe_de_SR_e_LT)"
    return result

def sr_pingpong_zone(df_m1: pd.DataFrame, atr_val: float) -> Optional[str]:
    """
    Se tiver suporte e resistência próximos ao preço, vira "corredor" -> evita operar.
    """
    res, sup = strong_sr_levels_last200(df_m1, atr_val)
    if not res or not sup:
        return None
    price = float(df_m1["close"].iloc[-1])
    atr_safe = max(atr_val, 1e-9)

    # pega níveis mais próximos por distância (não por força)
    res_near = sorted([(lvl,t) for (lvl,t) in res], key=lambda x: abs(x[0]-price))[:2]
    sup_near = sorted([(lvl,t) for (lvl,t) in sup], key=lambda x: abs(x[0]-price))[:2]

    # melhor resistência acima e melhor suporte abaixo
    above = [(lvl,t) for (lvl,t) in res_near if lvl >= price]
    below = [(lvl,t) for (lvl,t) in sup_near if lvl <= price]
    if not above or not below:
        return None

    r_lvl, r_t = min(above, key=lambda x: abs(x[0]-price))
    s_lvl, s_t = min(below, key=lambda x: abs(x[0]-price))

    corridor_atr = abs(r_lvl - s_lvl) / atr_safe
    # corredor curto => mercado batendo em paredes (reduzido de 1.60 para 0.85 para bloquear menos)
    if corridor_atr <= 0.85:
        return f"pingpong(corredor={corridor_atr:.2f}ATR sup={s_lvl:.6f} res={r_lvl:.6f})"
    return None

# ===================== DETECÇÃO DE ZONAS DE REVERSÃO (CRÍTICO!) =====================
def detect_mini_consolidation(df_m1: pd.DataFrame, atr_val: float, lookback: int = 30) -> Dict[str, Any]:
    """
    Detecta MINI-LATERALIZAÇÃO dentro de tendência.
    Bloqueia entradas quando preço está em range pequeno/consolidação.
    """
    if len(df_m1) < lookback:
        return {"is_consolidation": False, "reason": "dados_insuficientes"}
    
    recent = df_m1.tail(lookback)
    highs = recent["high"].to_numpy(float)
    lows = recent["low"].to_numpy(float)
    closes = recent["close"].to_numpy(float)
    
    # 1. RANGE VERTICAL (quanto o preço variou)
    price_range = float(np.max(highs) - np.min(lows))
    range_atr = price_range / max(atr_val, 1e-9)
    
    # Se range < 1.5 ATR em 30 velas = CONSOLIDAÇÃO
    if range_atr < 1.5:
        return {"is_consolidation": True, "reason": f"range_pequeno({range_atr:.2f}ATR)", "confidence": 0.90}
    
    # 2. TESTA SE ESTÁ EM CANAL HORIZONTAL (mini-lateral)
    # Divide em 3 partes e verifica se preço fica "preso"
    third = len(recent) // 3
    part1_range = float(np.max(highs[:third]) - np.min(lows[:third]))
    part2_range = float(np.max(highs[third:2*third]) - np.min(lows[third:2*third]))
    part3_range = float(np.max(highs[2*third:]) - np.min(lows[2*third:]))
    
    avg_part_range = (part1_range + part2_range + part3_range) / 3.0
    avg_part_range_atr = avg_part_range / max(atr_val, 1e-9)
    
    # Se cada parte tem range pequeno = lateral
    if avg_part_range_atr < 0.8:
        return {"is_consolidation": True, "reason": f"lateral_horizontal({avg_part_range_atr:.2f}ATR)", "confidence": 0.85}
    
    # 3. VERIFICA CHOPINESS (quantas vezes muda de direção)
    price_changes = np.diff(closes)
    direction_changes = 0
    for i in range(1, len(price_changes)):
        if (price_changes[i] > 0 and price_changes[i-1] < 0) or \
           (price_changes[i] < 0 and price_changes[i-1] > 0):
            direction_changes += 1

    chop_ratio = direction_changes / len(price_changes)

    # Se muda muito de direção (>70%) = choppy/lateral
    if chop_ratio > 0.70:
        return {"is_consolidation": True, "reason": f"choppy({chop_ratio:.2f})", "confidence": 0.75}
    
    return {"is_consolidation": False, "reason": "mercado_direcional", "confidence": 0.0}

def detect_reversal_zone(df_m1: pd.DataFrame, atr_val: float, direction: str) -> Dict[str, Any]:
    """
    Detecta se está em ZONA DE REVERSÃO ao invés de continuação.
    BLOQUEIA entradas em:
    1. Tendência muito estendida (over-extended)
    2. Formação de padrão de reversão (duplo fundo, duplo topo)
    3. Divergência entre preço e momentum
    4. Aproximação de suporte/resistência histórico forte
    """
    if len(df_m1) < 50:
        return {"is_reversal": False, "reason": "dados_insuficientes"}
    
    price = float(df_m1["close"].iloc[-1])
    recent = df_m1.tail(50)
    
    # 1. VERIFICA EXTENSÃO DA TENDÊNCIA (Bollinger extremo)
    closes = recent["close"].to_numpy(float)
    sma_20 = float(np.mean(closes[-20:]))
    std_20 = float(np.std(closes[-20:]))
    
    bb_upper = sma_20 + (2.5 * std_20)
    bb_lower = sma_20 - (2.5 * std_20)
    
    # Se preço está fora das Bollinger Bands (over-extended)
    if direction == "PUT" and price < bb_lower:
        # Querendo entrar PUT mas preço já está muito baixo = REVERSÃO PROVÁVEL
        return {"is_reversal": True, "reason": f"over_extended_baixa(price={price:.6f}<BB_lower={bb_lower:.6f})", "confidence": 0.85}
    
    if direction == "CALL" and price > bb_upper:
        # Querendo entrar CALL mas preço já está muito alto = REVERSÃO PROVÁVEL
        return {"is_reversal": True, "reason": f"over_extended_alta(price={price:.6f}>BB_upper={bb_upper:.6f})", "confidence": 0.85}
    
    # 2. DETECTA PADRÃO DE REVERSÃO (Duplo Fundo / Duplo Topo)
    lows = recent["low"].to_numpy(float)
    highs = recent["high"].to_numpy(float)
    
    if direction == "PUT":
        # Procura duplo fundo (sinal de reversão para alta)
        last_20_lows = lows[-20:]
        min_low_1 = float(np.min(last_20_lows[:10]))
        min_low_2 = float(np.min(last_20_lows[10:]))
        
        # Se formou duplo fundo (dois fundos similares) = REVERSÃO
        if abs(min_low_1 - min_low_2) < (0.5 * atr_val):
            return {"is_reversal": True, "reason": f"duplo_fundo(low1={min_low_1:.6f},low2={min_low_2:.6f})", "confidence": 0.75}
    
    if direction == "CALL":
        # Procura duplo topo (sinal de reversão para baixa)
        last_20_highs = highs[-20:]
        max_high_1 = float(np.max(last_20_highs[:10]))
        max_high_2 = float(np.max(last_20_highs[10:]))
        
        # Se formou duplo topo (dois topos similares) = REVERSÃO
        if abs(max_high_1 - max_high_2) < (0.5 * atr_val):
            return {"is_reversal": True, "reason": f"duplo_topo(high1={max_high_1:.6f},high2={max_high_2:.6f})", "confidence": 0.75}
    
    # 3. DIVERGÊNCIA SIMPLES (preço vs momentum)
    last_10 = df_m1.tail(10)
    price_change = float(last_10["close"].iloc[-1] - last_10["close"].iloc[0])
    
    # Calcula momentum simples (soma das mudanças de preço)
    price_diffs = np.diff(last_10["close"].to_numpy(float))
    momentum = float(np.sum(price_diffs))
    
    if direction == "PUT":
        # Se preço caindo MAS momentum desacelerando = DIVERGÊNCIA (possível reversão)
        if price_change < 0 and momentum > price_change * 0.5:
            return {"is_reversal": True, "reason": f"divergencia_alta(price_ch={price_change:.6f},momentum={momentum:.6f})", "confidence": 0.65}
    
    if direction == "CALL":
        # Se preço subindo MAS momentum desacelerando = DIVERGÊNCIA (possível reversão)
        if price_change > 0 and momentum < price_change * 0.5:
            return {"is_reversal": True, "reason": f"divergencia_baixa(price_ch={price_change:.6f},momentum={momentum:.6f})", "confidence": 0.65}
    
    # 4. VERIFICA DISTÂNCIA DE MÍNIMAS/MÁXIMAS HISTÓRICAS (50 velas)
    if direction == "PUT":
        # Se preço está muito perto da mínima histórica = ZONA DE REVERSÃO
        hist_low = float(np.min(lows))
        dist_to_low = abs(price - hist_low) / max(atr_val, 1e-9)
        
        if dist_to_low < 0.5:  # Muito perto do fundo histórico
            return {"is_reversal": True, "reason": f"proximo_minima_historica(dist={dist_to_low:.2f}ATR)", "confidence": 0.80}
    
    if direction == "CALL":
        # Se preço está muito perto da máxima histórica = ZONA DE REVERSÃO
        hist_high = float(np.max(highs))
        dist_to_high = abs(price - hist_high) / max(atr_val, 1e-9)
        
        if dist_to_high < 0.5:  # Muito perto do topo histórico
            return {"is_reversal": True, "reason": f"proximo_maxima_historica(dist={dist_to_high:.2f}ATR)", "confidence": 0.80}
    
    return {"is_reversal": False, "reason": "zona_continuacao_ok", "confidence": 0.0}

# ===================== PROJEÇÃO E VALIDAÇÃO DE ENTRADA =====================
def validate_entry_quality(df_m1: pd.DataFrame, atr_val: float, direction: str, entry_price: float, pb_high: float, pb_low: float) -> Dict[str, Any]:
    """
    Valida a qualidade da entrada projetando alvos e analisando risco/retorno.
    Retorna score de confiança e razão de risco/retorno.
    """
    if len(df_m1) < 5:
        return {"valid": False, "confidence": 0.0, "reason": "dados_insuficientes"}

    # 1. ANÁLISE DO CANDLE DE ENTRADA
    last_candle = df_m1.iloc[-1]
    open_price = float(last_candle["open"])
    close_price = float(last_candle["close"])
    high_price = float(last_candle["high"])
    low_price = float(last_candle["low"])

    candle_range = high_price - low_price
    body = abs(close_price - open_price)
    body_ratio = body / max(candle_range, 1e-9)

    # Candle de entrada deve ter corpo razoável (reduzido de 0.35 para 0.25)
    if body_ratio < 0.25:
        return {"valid": False, "confidence": 0.0, "reason": f"candle_fraco(body={body_ratio:.2f})"}

    # 2. PROJEÇÃO DE ALVO E STOP
    if direction == "CALL":
        # Stop abaixo da mínima do pullback
        stop_loss = pb_low - (0.15 * atr_val)
        risk = entry_price - stop_loss

        # Alvo: projeta 1.5x o risco (R:R 1:1.5)
        target_1 = entry_price + (risk * 1.5)

        # Verifica se há espaço para o alvo (sem resistência próxima)
        recent_highs = df_m1.tail(20)["high"].to_numpy(float)
        max_recent = float(np.max(recent_highs))

        # Se alvo muito próximo de máximas recentes, pode ser arriscado
        if target_1 > max_recent * 1.005:  # alvo acima das máximas
            confidence = 0.75
        else:
            confidence = 0.55

    else:  # PUT
        # Stop acima da máxima do pullback
        stop_loss = pb_high + (0.15 * atr_val)
        risk = stop_loss - entry_price

        # Alvo: projeta 1.5x o risco (R:R 1:1.5)
        target_1 = entry_price - (risk * 1.5)

        # Verifica espaço para o alvo
        recent_lows = df_m1.tail(20)["low"].to_numpy(float)
        min_recent = float(np.min(recent_lows))

        if target_1 < min_recent * 0.995:  # alvo abaixo das mínimas
            confidence = 0.75
        else:
            confidence = 0.55

    # 3. RAZÃO RISCO/RETORNO
    risk_atr = risk / max(atr_val, 1e-9)

    # Risco muito grande ou muito pequeno é ruim (relaxado de 0.3-1.5 para 0.2-2.0)
    if risk_atr < 0.2 or risk_atr > 2.0:
        return {"valid": False, "confidence": 0.0, "reason": f"risco_inadequado({risk_atr:.2f}ATR)"}

    # 4. MOMENTUM DO BREAKOUT
    last_3_closes = df_m1.tail(3)["close"].to_numpy(float)
    momentum = abs(last_3_closes[-1] - last_3_closes[0]) / max(atr_val, 1e-9)
    
    # Calcula direção do momentum
    raw_momentum = last_3_closes[-1] - last_3_closes[0]
    if direction == "CALL":
        momentum_direction = "correct" if raw_momentum > 0 else "wrong"
    else:  # PUT
        momentum_direction = "correct" if raw_momentum < 0 else "wrong"

    # Momentum fraco = breakout fraco (reduzido de 0.15 para 0.10)
    if momentum < 0.10:
        confidence *= 0.80  # penaliza menos

    # 5. ALINHAMENTO DE VELAS (últimas 5 velas para melhor análise)
    last_5 = df_m1.tail(5)
    aligned = 0
    for _, row in last_5.iterrows():
        c = float(row["close"])
        o = float(row["open"])
        if (direction == "CALL" and c > o) or (direction == "PUT" and c < o):
            aligned += 1

    alignment_ratio = aligned / 5.0  # Agora usa 5 velas (como no Firebase)
    if alignment_ratio >= 0.60:  # 3 de 5 velas alinhadas
        confidence *= 1.15  # bônus
    elif alignment_ratio < 0.40:  # menos de 2/5
        confidence *= 0.75  # penaliza mais

    # Confidence final
    confidence = min(1.0, max(0.0, confidence))

    # Score mínimo para aprovar (AUMENTADO de 0.40 para 0.45)
    if confidence < 0.45:
        return {"valid": False, "confidence": confidence, "reason": f"confianca_baixa({confidence:.2f})", 
                "momentum_direction": momentum_direction, "alignment": alignment_ratio}

    return {
        "valid": True,
        "confidence": float(confidence),
        "risk_atr": float(risk_atr),
        "risk_reward": 1.5,
        "momentum": float(momentum),
        "momentum_direction": momentum_direction,
        "body_ratio": float(body_ratio),
        "alignment": float(alignment_ratio),
        "reason": "entrada_validada"
    }

# ===================== VALIDAÇÃO DE CONTINUAÇÃO DE TENDÊNCIA =====================
def validate_trend_continuation(df_m1: pd.DataFrame, impulso_dir: str, pb_end_idx: int) -> Dict[str, Any]:
    """
    Valida se a entrada está em continuação de tendência.
    SUAVIZADO: Não bloqueia totalmente, apenas reduz score se contra tendência forte.
    """
    if len(df_m1) < pb_end_idx + 30:
        return {"valid": True, "reason": "dados_insuficientes", "strength": 0.5}

    # Pega as 15-25 velas ANTES da pernada A começar
    pre_impulse = df_m1.iloc[max(0, pb_end_idx - 25):pb_end_idx - 5]

    if len(pre_impulse) < 10:
        return {"valid": True, "reason": "contexto_curto", "strength": 0.5}

    closes = pre_impulse["close"].to_numpy(float)

    # Análise de tendência predominante
    price_change = closes[-1] - closes[0]
    price_change_pct = abs(price_change) / max(closes[0], 1e-9)

    if impulso_dir == "PUT":
        # Para PUT, queremos tendência de queda ANTES do impulso A
        if price_change > 0:
            # Preço estava subindo antes - só bloqueia se for tendência FORTE
            if price_change_pct > 0.015:  # Subida maior que 1.5%
                return {"valid": False, "reason": "contra_tendencia_forte_alta", "strength": 0.2}
            # Subida fraca, permite mas com score reduzido
            return {"valid": True, "reason": "contra_tendencia_fraca", "strength": 0.4}
        # Confirma tendência de queda
        return {"valid": True, "reason": "continuacao_queda", "strength": min(1.0, price_change_pct * 50)}

    else:  # CALL
        # Para CALL, queremos tendência de alta ANTES do impulso A
        if price_change < 0:
            # Preço estava caindo antes - só bloqueia se for tendência FORTE
            if price_change_pct > 0.015:  # Queda maior que 1.5%
                return {"valid": False, "reason": "contra_tendencia_forte_baixa", "strength": 0.2}
            # Queda fraca, permite mas com score reduzido
            return {"valid": True, "reason": "contra_tendencia_fraca", "strength": 0.4}
        # Confirma tendência de alta
        return {"valid": True, "reason": "continuacao_alta", "strength": min(1.0, price_change_pct * 50)}

# ===================== ANÁLISE INTELIGENTE DE CONTEXTO (SEM INDICADORES TRADICIONAIS) =====================
def analyze_market_context(df_m1: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    """
    Análise contextual inteligente do mercado sem depender de indicadores tradicionais.
    Foca em padrões de price action puros e estrutura de mercado.
    """
    if len(df_m1) < 50:
        return {"quality": 0.0, "context": "insuficiente"}

    # 1. MOMENTUM DIRECIONAL (últimas 20 velas)
    recent = df_m1.tail(20)
    closes = recent["close"].to_numpy(float)
    highs = recent["high"].to_numpy(float)
    lows = recent["low"].to_numpy(float)

    # Conta velas na direção vs contra direção
    bullish = sum(1 for i in range(len(recent)) if closes[i] > recent["open"].iloc[i])
    bearish = sum(1 for i in range(len(recent)) if closes[i] < recent["open"].iloc[i])
    directional_bias = abs(bullish - bearish) / len(recent)  # 0-1, quanto maior mais direcional

    # 2. VOLATILIDADE ORDENADA (mercado respeitando movimentos)
    ranges = [highs[i] - lows[i] for i in range(len(recent))]
    avg_range = np.mean(ranges)
    std_range = np.std(ranges)
    volatility_consistency = 1.0 - min(1.0, std_range / max(avg_range, 1e-9))

    # 3. HIGHER HIGHS / LOWER LOWS (estrutura de tendência)
    hh_count = sum(1 for i in range(5, len(recent)) if highs[i] > max(highs[i-5:i]))
    ll_count = sum(1 for i in range(5, len(recent)) if lows[i] < min(lows[i-5:i]))
    structure_quality = max(hh_count, ll_count) / max(1, len(recent) - 5)

    # 4. MOMENTUM DE PREÇO (velocidade da última perna)
    last_10 = closes[-10:]
    price_momentum = abs(last_10[-1] - last_10[0]) / (max(atr_val, 1e-9) * 10)
    price_momentum = min(1.0, price_momentum)  # normaliza

    # 5. LIMPEZA DOS CANDLES (menos wicks, mais body)
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

    # SCORE FINAL DE QUALIDADE (0-1)
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
def calcular_stake_dinamico(iq: IQ_Option, base_stake: float) -> float:
    """
    Calcula stake dinâmico baseado em % da banca atual.
    Se USE_DYNAMIC_STAKE=True, usa PERCENT_BANCA% do saldo.
    """
    if not USE_DYNAMIC_STAKE:
        return float(max(VALOR_MINIMO, base_stake))

    try:
        saldo = float(iq.get_balance())
        stake = (saldo * PERCENT_BANCA) / 100.0
        return float(max(VALOR_MINIMO, stake))
    except Exception:
        return float(max(VALOR_MINIMO, base_stake))

def verificar_meta_atingida(saldo_inicial: float, saldo_atual: float) -> Tuple[bool, float]:
    """
    Verifica se atingiu a meta de lucro ou stop loss.
    Retorna (deve_parar, lucro_percent).
    """
    lucro = saldo_atual - saldo_inicial
    lucro_percent = (lucro / saldo_inicial) * 100.0

    if lucro_percent >= META_LUCRO_PERCENT:
        return True, lucro_percent

    if lucro_percent <= -STOP_LOSS_PERCENT:
        return True, lucro_percent

    return False, lucro_percent


# ===================== 🎯 SISTEMA DE CONFLUÊNCIA ULTRA SELETIVO =====================
def calculate_confluence_score(df: pd.DataFrame, direction: str, setup: Dict[str, Any], atr_val: float) -> Tuple[int, List[str], bool]:
    """
    🎯 CALCULA PONTUAÇÃO DE CONFLUÊNCIA (0-6 sinais)
    
    CADA SINAL QUE ALINHA = +1 PONTO
    
    Sinais verificados:
    1. TENDÊNCIA (EMA 21 > EMA 50 para CALL, < para PUT)
    2. PREÇO vs EMA (preço acima/abaixo da EMA 21)
    3. RSI (> 50 para CALL, < 50 para PUT) 
    4. SUPORTE/RESISTÊNCIA (perto de zona favorável)
    5. VELA DE CONFIRMAÇÃO (corpo forte na direção)
    6. MERCADO DIRECIONAL (não está lateral/choppy)
    
    Retorna: (score, reasons, should_enter)
    """
    score = 0
    reasons = []
    
    if len(df) < 60:
        return 0, ["dados_insuficientes"], False
    
    try:
        closes = df['close'].to_numpy(float)
        highs = df['high'].to_numpy(float)
        lows = df['low'].to_numpy(float)
        opens = df['open'].to_numpy(float)
        
        current_close = closes[-1]
        current_open = opens[-1]
        current_high = highs[-1]
        current_low = lows[-1]
        
        # ===== 1. TENDÊNCIA (EMA 21 vs EMA 50) =====
        ema21 = pd.Series(closes).ewm(span=21, adjust=False).mean().iloc[-1]
        ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().iloc[-1]
        
        if direction == "CALL":
            if ema21 > ema50:
                score += 1
                reasons.append("✅EMA21>EMA50")
            else:
                reasons.append("❌EMA21<EMA50")
        else:  # PUT
            if ema21 < ema50:
                score += 1
                reasons.append("✅EMA21<EMA50")
            else:
                reasons.append("❌EMA21>EMA50")
        
        # ===== 2. PREÇO vs EMA 21 =====
        if direction == "CALL":
            if current_close > ema21:
                score += 1
                reasons.append("✅Preço>EMA21")
            else:
                reasons.append("❌Preço<EMA21")
        else:
            if current_close < ema21:
                score += 1
                reasons.append("✅Preço<EMA21")
            else:
                reasons.append("❌Preço>EMA21")
        
        # ===== 3. RSI (Período 14) =====
        delta = pd.Series(closes).diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 0.0001)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        if direction == "CALL":
            if rsi_val > 50 and rsi_val < 75:  # RSI bullish mas não overbought
                score += 1
                reasons.append(f"✅RSI={rsi_val:.0f}>50")
            elif rsi_val > 75:
                reasons.append(f"⚠️RSI={rsi_val:.0f}(overbought)")
            else:
                reasons.append(f"❌RSI={rsi_val:.0f}<50")
        else:
            if rsi_val < 50 and rsi_val > 25:  # RSI bearish mas não oversold
                score += 1
                reasons.append(f"✅RSI={rsi_val:.0f}<50")
            elif rsi_val < 25:
                reasons.append(f"⚠️RSI={rsi_val:.0f}(oversold)")
            else:
                reasons.append(f"❌RSI={rsi_val:.0f}>50")
        
        # ===== 4. SUPORTE/RESISTÊNCIA =====
        lookback_sr = min(100, len(df))
        recent_high = np.max(highs[-lookback_sr:])
        recent_low = np.min(lows[-lookback_sr:])
        price_range = recent_high - recent_low
        
        dist_to_support = current_close - recent_low
        dist_to_resistance = recent_high - current_close
        
        if direction == "CALL":
            # CALL é bom perto de suporte
            if dist_to_support < price_range * 0.25:  # nos 25% inferiores
                score += 1
                reasons.append("✅Perto_Suporte")
            elif dist_to_resistance < price_range * 0.15:  # muito perto de resistência
                reasons.append("❌Perto_Resistência")
            else:
                reasons.append("⚪Meio_Range")
        else:
            # PUT é bom perto de resistência
            if dist_to_resistance < price_range * 0.25:  # nos 25% superiores
                score += 1
                reasons.append("✅Perto_Resistência")
            elif dist_to_support < price_range * 0.15:  # muito perto de suporte
                reasons.append("❌Perto_Suporte")
            else:
                reasons.append("⚪Meio_Range")
        
        # ===== 5. VELA DE CONFIRMAÇÃO =====
        body = abs(current_close - current_open)
        candle_range = current_high - current_low
        body_ratio = body / candle_range if candle_range > 0 else 0
        
        is_bullish_candle = current_close > current_open
        is_bearish_candle = current_close < current_open
        
        if direction == "CALL":
            if is_bullish_candle and body_ratio > 0.55:
                score += 1
                reasons.append(f"✅Vela_Bullish({body_ratio:.0%})")
            elif is_bearish_candle:
                reasons.append(f"❌Vela_Bearish")
            else:
                reasons.append(f"⚪Vela_Fraca({body_ratio:.0%})")
        else:
            if is_bearish_candle and body_ratio > 0.55:
                score += 1
                reasons.append(f"✅Vela_Bearish({body_ratio:.0%})")
            elif is_bullish_candle:
                reasons.append(f"❌Vela_Bullish")
            else:
                reasons.append(f"⚪Vela_Fraca({body_ratio:.0%})")
        
        # ===== 6. MERCADO DIRECIONAL (não lateral) =====
        # Verifica as últimas 20 velas para detectar lateralidade
        last_20_range = np.max(highs[-20:]) - np.min(lows[-20:])
        directional_ratio = last_20_range / (atr_val * 20) if atr_val > 0 else 1
        
        # Conta mudanças de cor (choppiness)
        color_changes = 0
        for i in range(-19, 0):
            prev_bullish = closes[i-1] > opens[i-1]
            curr_bullish = closes[i] > opens[i]
            if prev_bullish != curr_bullish:
                color_changes += 1
        
        chop_ratio = color_changes / 19
        
        if chop_ratio < 0.45 and directional_ratio > 0.5:  # Mercado direcional
            score += 1
            reasons.append(f"✅Direcional(chop={chop_ratio:.0%})")
        elif chop_ratio > 0.60:
            reasons.append(f"❌Lateral(chop={chop_ratio:.0%})")
        else:
            reasons.append(f"⚪Neutro(chop={chop_ratio:.0%})")
        
    except Exception as e:
        return 0, [f"erro_calculo:{str(e)[:30]}"], False
    
    # ===== DECISÃO FINAL =====
    # 🎯 SÓ ENTRA SE TIVER MÍNIMO DE CONFLUÊNCIA
    should_enter = score >= MIN_CONFLUENCE_SIGNALS
    
    return score, reasons, should_enter


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

def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def _bucket(x: float, step: float, lo: float, hi: float) -> int:
    x = _clip(x, lo, hi)
    return int(round((x - lo) / step))

def ai_make_key(ativo: str, setup: Dict[str, Any]) -> str:
    """
    Chave compacta MELHORADA para análise mais inteligente:
    - direção
    - score (buckets mais refinados)
    - pb_len (tamanho do pullback)
    - retr (retração)
    - A_atr (força do impulso)
    - effA (eficiência do impulso A - NOVO)
    - flips (chopiness - NOVO)
    - distBreak (distância da quebra - NOVO)
    """
    d = str(setup.get("dir", "NEUTRAL"))
    sc = float(setup.get("score", 0.0))
    pb = int(setup.get("pb_len", 0))
    retr = float(setup.get("retr", 0.0))
    Aatr = float(setup.get("A_atr", 0.0))
    effA = float(setup.get("effA", 0.0))
    flips = float(setup.get("flips", 0.0))
    distBreak = float(setup.get("distBreak", 0.0))

    # Buckets mais refinados para melhor precisão
    b_sc = _bucket(sc, 0.04, 0.40, 1.00)        # 0.40..1.00, passo menor
    b_re = _bucket(retr, 0.06, 0.10, 0.80)      # 0.10..0.80, mais granular
    b_A  = _bucket(Aatr, 0.40, 0.60, 6.00)      # 0.60..6.00
    b_eff = _bucket(effA, 0.08, 0.50, 1.00)     # eficiência NOVO
    b_flip = _bucket(flips, 0.10, 0.0, 0.80)    # chopiness NOVO
    b_dist = _bucket(distBreak, 0.05, 0.0, 0.50) # distância da quebra NOVO

    return f"{d}|sc{b_sc}|pb{pb}|re{b_re}|A{b_A}|eff{b_eff}|fl{b_flip}|dst{b_dist}"

def ai_prior_from_setup(setup: Dict[str, Any]) -> float:
    """
    Prior INTELIGENTE baseado em múltiplos fatores (não apenas score).
    Analisa confluência de sinais para melhor estimativa inicial.
    """
    sc = float(setup.get("score", 0.0))
    effA = float(setup.get("effA", 0.0))
    flips = float(setup.get("flips", 0.5))
    retr = float(setup.get("retr", 0.5))
    distBreak = float(setup.get("distBreak", 0.2))

    # Base no score
    p = 0.50 + (sc - 0.50) * 0.35

    # Ajustes inteligentes por confluência:
    # 1. Eficiência alta do impulso A aumenta confiança
    if effA > 0.70:
        p += 0.04
    elif effA < 0.60:
        p -= 0.03

    # 2. Baixo chopiness (mercado direcional) aumenta confiança
    if flips < 0.35:
        p += 0.05
    elif flips > 0.55:
        p -= 0.04

    # 3. Retração ideal (0.3-0.5) aumenta confiança
    if 0.30 <= retr <= 0.50:
        p += 0.04
    elif retr < 0.20 or retr > 0.65:
        p -= 0.03

    # 4. Quebra próxima e limpa aumenta confiança
    if distBreak < 0.15:
        p += 0.03
    elif distBreak > 0.25:
        p -= 0.02

    return _clip(p, 0.40, 0.75)

def ai_predict(ativo: str, setup: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, float]:
    """
    Retorna: prob, bayes_mean, ucb01, conf, n_arm, total
    """
    key = ai_make_key(ativo, setup)
    arms = stats.setdefault("arms", {})
    meta = stats.setdefault("meta", {"total": 0})

    total = int(meta.get("total", 0))
    arm = arms.get(key)

    prior = ai_prior_from_setup(setup)

    if arm is None:
        # inicia com Beta fraca baseada no prior (a+b = 2)
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

    # UCB simples em [0..1]
    if n <= 0:
        ucb01 = 1.0
    else:
        bonus = math.sqrt(2.0 * math.log(max(2, total + 1)) / max(1, n))
        ucb01 = _clip(bayes_mean + bonus, 0.0, 1.0)

    # confiança cresce com amostras
    conf = _clip(n / (n + 10.0), 0.0, 0.99)

    # prob final mistura prior + bayes, mas com peso maior no bayes quando tem histórico
    # 🔥 APRENDIZADO RÁPIDO: n + 10 (era 25) = converge mais rápido para o histórico real
    w = _clip(n / (n + 10.0), 0.0, 1.0)
    prob = (1.0 - w) * prior + w * bayes_mean
    prob = _clip(prob, 0.0, 1.0)

    return {"prob": float(prob), "bayes": float(bayes_mean), "ucb01": float(ucb01),
            "conf": float(conf), "n_arm": n, "total": total, "key": key, "prior": prior}

def ai_update(ativo: str, setup: Dict[str, Any], pnl: float, stats: Dict[str, Any]):
    """
    🎯 APRENDIZADO RÁPIDO COM LOSS:
    - pnl > 0 => sucesso (peso 1.0)
    - pnl < 0 => falha (peso AI_LOSS_WEIGHT = 2.5x)
    - pnl = 0 => ignora
    
    Com AI_LOSS_WEIGHT=2.5, cada loss vale como 2.5 losses!
    Isso faz a IA aprender muito mais rápido a evitar padrões perdedores.
    """
    if pnl == 0:
        return

    key = ai_make_key(ativo, setup)
    arms = stats.setdefault("arms", {})
    meta = stats.setdefault("meta", {"total": 0})

    arm = arms.get(key)
    if arm is None:
        prior = ai_prior_from_setup(setup)
        arms[key] = {"a": 2.0 * prior, "b": 2.0 * (1.0 - prior), "n": 0, "wins": 0, "losses": 0}
        arm = arms[key]

    a = float(arm.get("a", 1.0))
    b = float(arm.get("b", 1.0))
    n = int(arm.get("n", 0))
    wins = int(arm.get("wins", 0))
    losses = int(arm.get("losses", 0))

    if pnl > 0:
        # WIN: peso normal (1.0)
        a += 1.0
        wins += 1
    else:
        # LOSS: peso MAIOR (AI_LOSS_WEIGHT = 2.5x)
        # Isso faz a IA aprender MUITO mais rápido com losses!
        b += AI_LOSS_WEIGHT
        losses += 1

    n += 1
    arm["a"], arm["b"], arm["n"] = a, b, n
    arm["wins"], arm["losses"] = wins, losses
    meta["total"] = int(meta.get("total", 0)) + 1
    
    # Log do aprendizado
    winrate = wins / max(n, 1)
    prob_atual = a / (a + b)
    log.info(f"🧠 IA APRENDEU: {key} | {'✅WIN' if pnl > 0 else '❌LOSS'} | W/L={wins}/{losses} ({winrate:.0%}) | prob={prob_atual:.0%}")


# ===================== DETECÇÃO CORRETA DE ELLIOTT WAVE (5 ONDAS) =====================
def detect_elliott_waves(df: pd.DataFrame, atr_val: float, lookback: int = 100) -> Dict[str, Any]:
    """
    Detecta padrão de Elliott Wave COMPLETO usando as últimas 'lookback' velas.
    
    ONDAS DE IMPULSO (5 ondas):
    - Onda 1: Primeiro movimento na direção da tendência
    - Onda 2: Correção (retrai 50-61.8% da Onda 1, NUNCA mais de 100%)
    - Onda 3: Extensão forte (geralmente a maior, mínimo 1.618x da Onda 1)
    - Onda 4: Correção menor (retrai 38.2% da Onda 3, NÃO sobrepõe Onda 1)
    - Onda 5: Movimento final (pode ser menor que Onda 3)
    
    REGRAS DE ELLIOTT:
    1. Onda 2 NUNCA retrai mais de 100% da Onda 1
    2. Onda 3 NUNCA é a menor entre 1, 3, 5
    3. Onda 4 NUNCA entra no território de preço da Onda 1
    
    NÍVEIS FIBONACCI:
    - 38.2% (0.382): Retração comum da Onda 4
    - 50.0% (0.500): Retração média
    - 61.8% (0.618): Retração máxima ideal da Onda 2
    - 161.8% (1.618): Extensão mínima da Onda 3
    """
    
    result = {
        "valid": False,
        "direction": "NEUTRAL",
        "current_wave": 0,
        "entry_point": None,
        "waves": {},
        "confidence": 0.0,
        "reason": ""
    }
    
    if len(df) < lookback:
        result["reason"] = f"dados_insuficientes({len(df)}<{lookback})"
        return result
    
    sub = df.tail(lookback)
    closes = sub["close"].to_numpy(float)
    highs = sub["high"].to_numpy(float)
    lows = sub["low"].to_numpy(float)
    opens = sub["open"].to_numpy(float)
    
    # Encontra pivôs de alta e baixa (swing highs/lows)
    swing_highs = []
    swing_lows = []
    
    for i in range(3, len(sub) - 3):
        # Swing High: ponto mais alto que os 3 vizinhos de cada lado
        if highs[i] >= max(highs[i-3:i]) and highs[i] >= max(highs[i+1:i+4]):
            swing_highs.append((i, highs[i]))
        # Swing Low: ponto mais baixo que os 3 vizinhos de cada lado
        if lows[i] <= min(lows[i-3:i]) and lows[i] <= min(lows[i+1:i+4]):
            swing_lows.append((i, lows[i]))
    
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        result["reason"] = f"pivots_insuficientes(highs={len(swing_highs)},lows={len(swing_lows)})"
        return result
    
    # Determina direção da tendência predominante
    first_price = closes[0]
    last_price = closes[-1]
    mid_price = closes[len(closes)//2]
    
    trend_direction = "NEUTRAL"
    if last_price > first_price and mid_price > first_price:
        trend_direction = "UP"
    elif last_price < first_price and mid_price < first_price:
        trend_direction = "DOWN"
    
    if trend_direction == "NEUTRAL":
        result["reason"] = "sem_tendencia_clara"
        return result
    
    # ============ DETECÇÃO DE ONDAS PARA TENDÊNCIA DE ALTA ============
    if trend_direction == "UP":
        # Procura sequência: Low1 → High1 → Low2 → High3 → Low4 → High5
        sorted_lows = sorted(swing_lows, key=lambda x: x[0])
        sorted_highs = sorted(swing_highs, key=lambda x: x[0])
        
        best_pattern = None
        best_score = 0
        
        for i, (idx1, wave1_start) in enumerate(sorted_lows[:-2]):
            # Onda 1: Primeiro impulso de alta
            candidates_h1 = [h for h in sorted_highs if h[0] > idx1]
            if not candidates_h1:
                continue
            
            for idx_h1, wave1_end in candidates_h1[:3]:  # Testa até 3 candidatos
                wave1_size = wave1_end - wave1_start
                if wave1_size < atr_val * 0.5:  # Onda 1 mínima: 0.5 ATR
                    continue
                
                # Onda 2: Correção (deve ser mínima após Onda 1)
                candidates_l2 = [l for l in sorted_lows if l[0] > idx_h1]
                if not candidates_l2:
                    continue
                
                for idx_l2, wave2_end in candidates_l2[:3]:
                    wave2_retrace = (wave1_end - wave2_end) / max(wave1_size, 1e-9)
                    
                    # REGRA 1: Onda 2 não pode retrair mais de 100% da Onda 1
                    if wave2_end < wave1_start:
                        continue  # Violação!
                    
                    # Retração ideal: 50-61.8%
                    if wave2_retrace < 0.38 or wave2_retrace > 0.90:
                        continue  # Retração fora do range ideal
                    
                    # Onda 3: Extensão (deve ser maior que Onda 1)
                    candidates_h3 = [h for h in sorted_highs if h[0] > idx_l2]
                    if not candidates_h3:
                        continue
                    
                    for idx_h3, wave3_end in candidates_h3[:3]:
                        wave3_size = wave3_end - wave2_end
                        
                        # REGRA 2: Onda 3 não pode ser a menor (idealmente >= 1.618x Onda 1)
                        if wave3_size < wave1_size:
                            continue  # Onda 3 muito pequena
                        
                        # Onda 4: Correção menor
                        candidates_l4 = [l for l in sorted_lows if l[0] > idx_h3]
                        if not candidates_l4:
                            continue
                        
                        for idx_l4, wave4_end in candidates_l4[:3]:
                            wave4_retrace = (wave3_end - wave4_end) / max(wave3_size, 1e-9)
                            
                            # REGRA 3: Onda 4 não pode sobrepor Onda 1
                            if wave4_end < wave1_end:
                                continue  # Violação! Onda 4 entrou no território da Onda 1
                            
                            # Retração ideal da Onda 4: 23.6-50%
                            if wave4_retrace < 0.20 or wave4_retrace > 0.60:
                                continue
                            
                            # Calcula score do padrão
                            score = 0.0
                            
                            # Bônus por retração de Onda 2 próxima de 61.8%
                            if 0.55 <= wave2_retrace <= 0.70:
                                score += 0.25
                            elif 0.45 <= wave2_retrace <= 0.75:
                                score += 0.15
                            
                            # Bônus por Onda 3 >= 1.618x Onda 1
                            wave3_extension = wave3_size / max(wave1_size, 1e-9)
                            if wave3_extension >= 1.618:
                                score += 0.30
                            elif wave3_extension >= 1.3:
                                score += 0.15
                            
                            # Bônus por retração de Onda 4 próxima de 38.2%
                            if 0.35 <= wave4_retrace <= 0.45:
                                score += 0.20
                            elif 0.25 <= wave4_retrace <= 0.50:
                                score += 0.10
                            
                            # Bônus por padrão recente (mais próximo do final)
                            recency = idx_l4 / len(sub)
                            score += recency * 0.15
                            
                            if score > best_score:
                                best_score = score
                                best_pattern = {
                                    "wave1": {"start": wave1_start, "end": wave1_end, "size": wave1_size, "idx_start": idx1, "idx_end": idx_h1},
                                    "wave2": {"end": wave2_end, "retrace": wave2_retrace, "idx": idx_l2},
                                    "wave3": {"end": wave3_end, "size": wave3_size, "extension": wave3_extension, "idx": idx_h3},
                                    "wave4": {"end": wave4_end, "retrace": wave4_retrace, "idx": idx_l4},
                                    "wave5_target": wave4_end + wave1_size,  # Onda 5 projetada
                                    "score": score
                                }
        
        if best_pattern and best_score >= 0.35:
            result["valid"] = True
            result["direction"] = "CALL"
            result["waves"] = best_pattern
            result["confidence"] = min(1.0, best_score + 0.35)
            
            # Determina onda atual
            current_close = closes[-1]
            if current_close > best_pattern["wave3"]["end"]:
                result["current_wave"] = 5
                result["entry_point"] = None  # Já está na Onda 5, tarde demais
                result["reason"] = "onda_5_em_andamento"
            elif current_close > best_pattern["wave2"]["end"]:
                result["current_wave"] = 3
                result["entry_point"] = "CALL"  # MELHOR ENTRADA: Onda 3
                result["reason"] = "onda_3_continuacao"
            elif current_close > best_pattern["wave1"]["start"]:
                result["current_wave"] = 1
                result["entry_point"] = None  # Aguardar Onda 2 completar
                result["reason"] = "aguardar_onda_2"
    
    # ============ DETECÇÃO DE ONDAS PARA TENDÊNCIA DE BAIXA ============
    elif trend_direction == "DOWN":
        sorted_lows = sorted(swing_lows, key=lambda x: x[0])
        sorted_highs = sorted(swing_highs, key=lambda x: x[0])
        
        best_pattern = None
        best_score = 0
        
        for i, (idx1, wave1_start) in enumerate(sorted_highs[:-2]):
            # Onda 1: Primeiro impulso de baixa
            candidates_l1 = [l for l in sorted_lows if l[0] > idx1]
            if not candidates_l1:
                continue
            
            for idx_l1, wave1_end in candidates_l1[:3]:
                wave1_size = wave1_start - wave1_end
                if wave1_size < atr_val * 0.5:
                    continue
                
                # Onda 2: Correção (alta)
                candidates_h2 = [h for h in sorted_highs if h[0] > idx_l1]
                if not candidates_h2:
                    continue
                
                for idx_h2, wave2_end in candidates_h2[:3]:
                    wave2_retrace = (wave2_end - wave1_end) / max(wave1_size, 1e-9)
                    
                    # REGRA 1: Onda 2 não pode retrair mais de 100%
                    if wave2_end > wave1_start:
                        continue
                    
                    if wave2_retrace < 0.38 or wave2_retrace > 0.90:
                        continue
                    
                    # Onda 3: Extensão de baixa
                    candidates_l3 = [l for l in sorted_lows if l[0] > idx_h2]
                    if not candidates_l3:
                        continue
                    
                    for idx_l3, wave3_end in candidates_l3[:3]:
                        wave3_size = wave2_end - wave3_end
                        
                        # REGRA 2: Onda 3 não pode ser a menor
                        if wave3_size < wave1_size:
                            continue
                        
                        # Onda 4: Correção de alta
                        candidates_h4 = [h for h in sorted_highs if h[0] > idx_l3]
                        if not candidates_h4:
                            continue
                        
                        for idx_h4, wave4_end in candidates_h4[:3]:
                            wave4_retrace = (wave4_end - wave3_end) / max(wave3_size, 1e-9)
                            
                            # REGRA 3: Onda 4 não pode sobrepor Onda 1
                            if wave4_end > wave1_end:
                                continue
                            
                            if wave4_retrace < 0.20 or wave4_retrace > 0.60:
                                continue
                            
                            score = 0.0
                            
                            if 0.55 <= wave2_retrace <= 0.70:
                                score += 0.25
                            elif 0.45 <= wave2_retrace <= 0.75:
                                score += 0.15
                            
                            wave3_extension = wave3_size / max(wave1_size, 1e-9)
                            if wave3_extension >= 1.618:
                                score += 0.30
                            elif wave3_extension >= 1.3:
                                score += 0.15
                            
                            if 0.35 <= wave4_retrace <= 0.45:
                                score += 0.20
                            elif 0.25 <= wave4_retrace <= 0.50:
                                score += 0.10
                            
                            recency = idx_h4 / len(sub)
                            score += recency * 0.15
                            
                            if score > best_score:
                                best_score = score
                                best_pattern = {
                                    "wave1": {"start": wave1_start, "end": wave1_end, "size": wave1_size, "idx_start": idx1, "idx_end": idx_l1},
                                    "wave2": {"end": wave2_end, "retrace": wave2_retrace, "idx": idx_h2},
                                    "wave3": {"end": wave3_end, "size": wave3_size, "extension": wave3_extension, "idx": idx_l3},
                                    "wave4": {"end": wave4_end, "retrace": wave4_retrace, "idx": idx_h4},
                                    "wave5_target": wave4_end - wave1_size,
                                    "score": score
                                }
        
        if best_pattern and best_score >= 0.35:
            result["valid"] = True
            result["direction"] = "PUT"
            result["waves"] = best_pattern
            result["confidence"] = min(1.0, best_score + 0.35)
            
            current_close = closes[-1]
            if current_close < best_pattern["wave3"]["end"]:
                result["current_wave"] = 5
                result["entry_point"] = None
                result["reason"] = "onda_5_em_andamento"
            elif current_close < best_pattern["wave2"]["end"]:
                result["current_wave"] = 3
                result["entry_point"] = "PUT"
                result["reason"] = "onda_3_continuacao"
            elif current_close < best_pattern["wave1"]["start"]:
                result["current_wave"] = 1
                result["entry_point"] = None
                result["reason"] = "aguardar_onda_2"
    
    return result


# ===================== PADRÃO IMPULSO-CORREÇÃO-CONTINUAÇÃO (Ondas 1-2-3 ou 3-4-5 de Elliott) =====================
def pernada_b(df_m1: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    """
    Detecta padrão de continuação de tendência baseado em Ondas de Elliott:
    
    ESTRUTURA DO PADRÃO:
    1. IMPULSO (Onda 1 ou 3): Movimento forte na direção da tendência
    2. CORREÇÃO/PULLBACK (Onda 2 ou 4): Retração temporária (pullback)
    3. CONTINUAÇÃO (Entrada na Onda 3 ou 5): Retomada da tendência principal
    
    EXEMPLOS:
    - TENDÊNCIA DE ALTA: Impulso↑ → Pullback↓ → Rompimento↑ (CALL)
    - TENDÊNCIA DE BAIXA: Impulso↓ → Pullback↑ → Rompimento↓ (PUT)
    
    NOTA: NÃO é a "Onda B" corretiva ABC, mas sim uma continuação impulsiva!
    """
    if len(df_m1) < 240:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["poucas_velas"]}

    # ⭐ ANÁLISE DE TENDÊNCIA MAIOR COM ATR TRAILING STOP (CRÍTICO!)
    # Aumentado para 20 períodos e multiplicador 2.5 para ser mais rigoroso
    atr_ts_value, atr_position = calculate_atr_trailing_stops(df_m1, period=20, multiplier=2.5, high_low=False)
    
    # ⭐ ANÁLISE DE ESTRUTURA DE MERCADO (TOPOS E FUNDOS) - Aumentado para 150 velas
    market_structure = detect_market_structure_trend(df_m1, lookback=150, swing_window=5)
    market_trend = market_structure.get("trend", "LATERAL")
    market_allowed = market_structure.get("allowed_direction", "NONE")
    trend_strength = market_structure.get("strength", 0.0)
    
    # 📊 Intensidade recente (últimas 20 velas)
    recent_20 = df_m1.tail(20)
    if len(recent_20) >= 10:
        red_ratio = float((recent_20["close"] < recent_20["open"]).mean())
        green_ratio = float((recent_20["close"] > recent_20["open"]).mean())
    else:
        red_ratio = 0.0
        green_ratio = 0.0
    
    # 🚫 BLOQUEIO RIGOROSO: NÃO OPERA CONTRA A TENDÊNCIA MAIOR
    # Se ATR TS indica tendência de baixa (position=-1), BLOQUEIA CALL
    # Se ATR TS indica tendência de alta (position=1), BLOQUEIA PUT
    # Se estrutura confirma tendência DOWN, BLOQUEIA CALL
    # Se estrutura confirma tendência UP, BLOQUEIA PUT
    
    # 🛑 NOVO FILTRO: BLOQUEIA CONTRA-TENDÊNCIA FORTE (EVITA PULLBACK EM TENDÊNCIA)
    # Calcula força da tendência pelos últimos candles
    last_20 = df_m1.tail(20)
    trend_momentum = (last_20['close'].iloc[-1] - last_20['close'].iloc[0]) / (atr_val * 20)
    
    # Se tendência forte de QUEDA (-) e sinal é CALL, BLOQUEIA
    if trend_momentum < -0.3 and market_trend == "DOWN":
        # Verifica se é pullback temporário (últimas 5 velas)
        last_5 = df_m1.tail(5)
        recent_bounce = (last_5['close'].iloc[-1] - last_5['close'].iloc[0]) / atr_val
        if recent_bounce > 0.5:  # Pullback detectado
            return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
                    "reasons": [f"🚫pullback_em_downtrend_forte(trend_mom={trend_momentum:.2f},bounce={recent_bounce:.2f})"]}
    
    # Se tendência forte de ALTA (+) e sinal é PUT, BLOQUEIA  
    if trend_momentum > 0.3 and market_trend == "UP":
        # Verifica se é pullback temporário (últimas 5 velas)
        last_5 = df_m1.tail(5)
        recent_drop = (last_5['close'].iloc[0] - last_5['close'].iloc[-1]) / atr_val
        if recent_drop > 0.5:  # Pullback detectado
            return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
                    "reasons": [f"🚫pullback_em_uptrend_forte(trend_mom={trend_momentum:.2f},drop={recent_drop:.2f})"]}
    
    # ANÁLISE INTELIGENTE DE CONTEXTO
    context = analyze_market_context(df_m1, atr_val)
    market_quality = float(context.get("quality", 0.0))

    # BLOQUEIO: Detecta MINI-CONSOLIDAÇÃO/LATERALIZAÇÃO (🚨 MAIS RIGOROSO)
    consolidation_check = detect_mini_consolidation(df_m1, atr_val, lookback=30)
    if consolidation_check.get("is_consolidation", False):
        # Bloqueia se confiança >= 0.70 (REDUZIDO de 0.88 - mais restritivo)
        if consolidation_check.get("confidence", 0.0) >= MAX_CONSOLIDATION_CONFIDENCE:
            return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
                    "reasons": [f"🚫 consolidacao_detectada({consolidation_check.get('reason','?')},conf={consolidation_check.get('confidence',0):.0%})"]}

    # ⭐ ANÁLISE DE ELLIOTT WAVE COMPLETA (100 VELAS)
    # Detecta padrão de 5 ondas correto com regras de Fibonacci
    elliott_analysis = detect_elliott_waves(df_m1, atr_val, lookback=100)
    elliott_valid = elliott_analysis.get("valid", False)
    elliott_direction = elliott_analysis.get("direction", "NEUTRAL")
    elliott_entry = elliott_analysis.get("entry_point", None)
    elliott_confidence = elliott_analysis.get("confidence", 0.0)
    elliott_current_wave = elliott_analysis.get("current_wave", 0)
    elliott_reason = elliott_analysis.get("reason", "")

    # Direção sugerida pela estrutura de mercado (NÃO BLOQUEANTE)
    # A decisão final será feita pelo agent IA após análise contextual
    suggested_direction = None

    # Detecta direção pelo ATR
    if atr_position == 1:
        suggested_direction = "CALL"
    elif atr_position == -1:
        suggested_direction = "PUT"

    # Se estrutura de mercado sugere direção diferente, registra mas não bloqueia
    if market_allowed != "NONE" and suggested_direction != market_allowed:
        # Conflito - mas deixa o agent decidir
        pass

    # Permite ambas as direções - o filtro agora é contextual via agent
    allowed_directions = {"CALL", "PUT"}

    # NÃO bloqueia mais por contexto ruim (apenas registra)
    # if market_quality < 0.15:
    #     return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
    #             "reasons": [f"contexto_ruim(quality={market_quality:.2f},ctx={context.get('context','?')})"]}

    flips_frac, eff_zone = chop_stats(df_m1, CHOP_LOOKBACK)
    comp = compression_ratio(df_m1, atr_val, COMP_LOOKBACK)
    late_ext = late_extension_atr(df_m1, atr_val, LATE_LOOKBACK)

    # FILTROS MUITO RELAXADOS - apenas para evitar situações extremas
    # if flips_frac > MAX_COLOR_FLIPS_FRAC and eff_zone < MIN_NET_GROSS_EFF:
    #     return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
    #             "reasons": [f"lateral_chop(flips={flips_frac:.2f},eff={eff_zone:.2f})"]}

    # if comp < MIN_RANGE_ATR:
    #     return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
    #             "reasons": [f"lateral_compress(range={comp:.2f}ATR/{COMP_LOOKBACK})"]}

    # if late_ext > MAX_LATE_EXT_ATR:
    #     return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
    #             "reasons": [f"entrada_atrasada(ext={late_ext:.2f}ATR/{LATE_LOOKBACK})"]}

    decision = df_m1.iloc[-1]
    q = wick_fractions(decision)

    if q["body_frac"] < MIN_BODY_FRAC_BREAK:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
                "reasons": [f"gatilho_fraco(body={q['body_frac']:.2f})"]}

    # corredor de SR perto do preço => evita operar
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

            # IDENTIFICAÇÃO CORRETA DO PADRÃO DE ELLIOTT:
            # - Impulso (Onda 1/3): Movimento forte
            # - Pullback (Onda 2/4): Correção temporária
            # - A DIREÇÃO DO IMPULSO define a TENDÊNCIA PRINCIPAL
            # 
            # Se impulso foi de QUEDA (move < 0):
            #   → Tendência = BAIXA
            #   → Pullback sobe (contra-tendência)
            #   → Entrada PUT quando rompe o pullback para baixo
            #
            # Se impulso foi de ALTA (move > 0):
            #   → Tendência = ALTA  
            #   → Pullback desce (contra-tendência)
            #   → Entrada CALL quando rompe o pullback para cima
            
            dir_impulso = "PUT" if move < 0 else ("CALL" if move > 0 else "NEUTRAL")
            if dir_impulso == "NEUTRAL":
                continue
            
            # 🚫 VALIDAÇÃO CRÍTICA: Impulso DEVE estar alinhado com tendência maior!
            # Se impulso é CALL mas tendência permite só PUT → PULA
            # Se impulso é PUT mas tendência permite só CALL → PULA
            if dir_impulso not in allowed_directions:
                continue  # Impulso contra a tendência maior - ignora!

            eff_A = leg_efficiency(imp)
            if eff_A < MIN_EFF_A:
                continue

            # VALIDAÇÃO DO PULLBACK (Onda 2 ou 4):
            # O pullback deve mover-se CONTRA o impulso principal
            # 
            # Se impulso foi de QUEDA (PUT):
            #   → Pullback deve ter velas de ALTA (correção para cima)
            # Se impulso foi de ALTA (CALL):
            #   → Pullback deve ter velas de QUEDA (correção para baixo)
            
            contra = 0
            for _, r in pb.iterrows():
                d = candle_dir(r)
                # Se impulso foi PUT (queda), conta velas CALL no pullback
                if dir_impulso == "PUT" and d == 1:
                    contra += 1
                # Se impulso foi CALL (alta), conta velas PUT no pullback
                if dir_impulso == "CALL" and d == -1:
                    contra += 1

            # Pullback precisa ter pelo menos 50% das velas contra o impulso
            if contra < max(1, int(math.ceil(pb_len * 0.50))):
                continue

            # VALIDAÇÃO DE CONTINUAÇÃO DE TENDÊNCIA
            impulse_start_idx = len(df_m1) - (pb_len + 1 + w)
            trend_validation = validate_trend_continuation(df_m1, dir_impulso, impulse_start_idx)

            # Força da tendência (não bloqueante)
            trend_strength = float(trend_validation.get("strength", 0.0)) if trend_validation.get("valid", False) else 0.0

            # CÁLCULO DA RETRAÇÃO DO PULLBACK (Fibonacci)
            if dir_impulso == "PUT":
                # Impulso de queda → pullback sobe → mede quanto subiu
                pb_high = float(pb["high"].max())
                retr = (pb_high - bot) / max(size_A, 1e-9)
            else:  # dir_impulso == "CALL"
                # Impulso de alta → pullback desce → mede quanto desceu
                pb_low = float(pb["low"].min())
                retr = (top - pb_low) / max(size_A, 1e-9)

            if retr < RETR_MIN or retr > RETR_MAX:
                continue

            c1 = float(decision["close"])

            # ENTRADA: Continuação na direção do impulso (Onda 3 ou 5)
            # - Impulso PUT → Entrada PUT quando rompe pullback para baixo
            # - Impulso CALL → Entrada CALL quando rompe pullback para cima
            dir_entrada = dir_impulso
            
            # 🚫 BLOQUEIO CRÍTICO: Verifica se a direção está permitida pela tendência maior
            if dir_entrada not in allowed_directions:
                continue  # Pula este setup - contra tendência!
            
            # 🚫 BLOQUEIO POR TENDÊNCIA RECENTE (loss recorrente no Firebase)
            # Evita CALL quando o mercado está bearish e >60% velas vermelhas recentes
            if dir_entrada == "CALL" and market_trend == "DOWN" and red_ratio >= 0.60:
                continue
            # Evita PUT quando o mercado está bullish e >60% velas verdes recentes
            if dir_entrada == "PUT" and market_trend == "UP" and green_ratio >= 0.60:
                continue
            
            # 🛑 FILTRO ADICIONAL: Bloqueia CALL em downtrend forte mesmo que allowed
            if dir_entrada == "CALL" and market_trend == "DOWN" and trend_strength > 0.6:
                # Verifica se as últimas 10 velas têm tendência de queda clara
                last_10_change = (df_m1['close'].iloc[-1] - df_m1['close'].iloc[-10]) / atr_val
                if last_10_change < -2.0:  # Queda maior que 2 ATRs nas últimas 10 velas
                    continue  # Bloqueia CALL em queda forte
            
            # 🛑 FILTRO ADICIONAL: Bloqueia PUT em uptrend forte mesmo que allowed
            if dir_entrada == "PUT" and market_trend == "UP" and trend_strength > 0.6:
                # Verifica se as últimas 10 velas têm tendência de alta clara
                last_10_change = (df_m1['close'].iloc[-1] - df_m1['close'].iloc[-10]) / atr_val
                if last_10_change > 2.0:  # Alta maior que 2 ATRs nas últimas 10 velas
                    continue  # Bloqueia PUT em alta forte

            # bloqueio SR forte (múltiplas regiões)
            blk_sr = sr_block_directional_multi(df_m1, atr_val, dir_entrada)
            if blk_sr:
                continue

            # FILTRO DE TIMING: Evita entrada quando preço se aproxima de S/R forte
            approaching_sr = check_approaching_sr(df_m1, atr_val, dir_entrada)
            if approaching_sr:
                continue  # Espera o preço tocar/rejeitar o S/R antes de entrar

            # 🎯🎯🎯 FILTRO CRÍTICO: SÓ ENTRA PERTO DE S/R OU LT (NÃO NO MEIO!) 🎯🎯🎯
            # Isso evita entradas no "meio do caminho" como no loss do EUR/JPY
            key_level_check = check_price_at_key_level(df_m1, atr_val, dir_entrada)
            if not key_level_check.get("at_key_level", False):
                # Preço está no MEIO DO CAMINHO - não entra!
                continue

            # Detecta ZONA DE REVERSÃO
            reversal_check = detect_reversal_zone(df_m1, atr_val, dir_entrada)
            if reversal_check.get("is_reversal", False):
                # BLOQUEIA se confiança de reversão >= 65%
                if reversal_check.get("confidence", 0.0) >= 0.65:
                    continue  # Pula - está em zona de reversão!

            # Calcula os extremos do pullback (necessário para ambas direções)
            pb_high = float(pb["high"].max())
            pb_low = float(pb["low"].min())

            # LÓGICA DE ROMPIMENTO (CONTINUAÇÃO - Onda 3 ou 5)
            if dir_entrada == "CALL":
                # CALL: Impulso de ALTA → Pullback desce → Rompe para CIMA (continuação)
                if not (c1 > pb_low + BREAK_MARGIN_ATR * atr_val):
                    continue
                if q["upper_frac"] > MAX_WICK_AGAINST:
                    continue

                dist = (c1 - pb_low) / max(atr_val, 1e-9)
                if dist > MAX_BREAK_DISTANCE_ATR:
                    continue

                # 🚨 FILTRO DESATIVADO: BLOQUEIA CALL PERTO DE RESISTÊNCIA
                # if BLOCK_CALL_NEAR_RESISTANCE:
                #     res_levels, _ = strong_sr_levels_last200(df_m1, atr_val)
                #     if res_levels:
                #         for res_lvl, res_touches in res_levels:
                #             dist_to_res = (res_lvl - c1) / max(atr_val, 1e-9)
                #             if 0 < dist_to_res < 0.5 and res_touches >= 2:
                #                 continue

                # VALIDAÇÃO DE QUALIDADE DA ENTRADA (NOVO)
                entry_validation = validate_entry_quality(df_m1, atr_val, "CALL", c1, pb_high, pb_low)
                if not entry_validation.get("valid", False):
                    continue  # Pula entrada se não passar na validação
                
                # 🚨 FILTRO DESATIVADO: VERIFICA ALINHAMENTO MÍNIMO
                entry_alignment = float(entry_validation.get("alignment", 0.0))
                # if entry_alignment < MIN_ALIGNMENT_RATIO:
                #     continue  # DESATIVADO - não bloqueia por alinhamento
                
                # 🚨 FILTRO DESATIVADO: VERIFICA MOMENTUM NA DIREÇÃO CORRETA
                # if MIN_MOMENTUM_ALIGNMENT:
                #     entry_momentum_dir = entry_validation.get("momentum_direction", "neutral")
                #     if entry_momentum_dir == "wrong":
                #         continue  # DESATIVADO - não bloqueia por momentum
                
                # DESATIVADO: Exigia corpo forte contra tendência
                body_ratio = float(entry_validation.get("body_ratio", 0.0))
                # if market_trend == "DOWN" and red_ratio >= 0.60 and body_ratio < 0.70:
                #     continue

                entry_confidence = float(entry_validation.get("confidence", 0.0))
                risk_atr = float(entry_validation.get("risk_atr", 0.0))
                entry_momentum = float(entry_validation.get("momentum", 0.0))

                # ⭐ DETECÇÃO DE LTA COM SEGUNDO TOQUE + PAVIO DE REJEIÇÃO (ESTRATÉGIA PRINCIPAL)
                lt_second_touch = detect_lt_second_touch(df_m1, atr_val, lookback=50)
                has_lt_signal = lt_second_touch.get("has_lt", False)
                lt_signal_type = lt_second_touch.get("lt_type", None)
                lt_signal_dir = lt_second_touch.get("signal", None)
                lt_touches = lt_second_touch.get("touches", 0)
                lt_rejection = lt_second_touch.get("rejection_strength", 0.0)
                lt_score_bonus = lt_second_touch.get("score_bonus", 0.0)
                lt_reason = lt_second_touch.get("reason", "")
                
                # Se LTA detectada com segundo toque e direção é CALL = BÔNUS GRANDE!
                lt_call_bonus = 0.0
                if has_lt_signal and lt_signal_type == "LTA" and lt_signal_dir == "CALL":
                    lt_call_bonus = lt_score_bonus  # 0.20 a 0.25 de bônus

                # CONFLUÊNCIA COM LINHA DE TENDÊNCIA (LTA para CALL)
                lt_conf = check_trendline_confluence(df_m1, pb_high, pb_low, "CALL", atr_val)
                lt_confluence = float(lt_conf.get("confluence", 0.0))
                has_lt = lt_conf.get("has_trendline", False)

                # CONFLUÊNCIA COM SUPORTE/RESISTÊNCIA (Suporte para CALL)
                sr_conf = check_sr_confluence(df_m1, pb_low, pb_high, "CALL", atr_val)
                sr_confluence = float(sr_conf.get("confluence", 0.0))
                has_sr = sr_conf.get("has_sr", False)

                # ⚠️ CONFLUÊNCIA OPCIONAL: Bônus se tiver LTA OU Suporte (não bloqueia mais)
                # if not has_lt and not has_sr:
                #     continue  # DESATIVADO - permite entrada sem confluência

                # 🚫 BLOQUEIO CRÍTICO: Se Elliott Wave indica direção OPOSTA com alta confiança, BLOQUEIA!
                if elliott_valid and elliott_direction == "PUT" and elliott_confidence >= 0.70:
                    continue  # BLOQUEIA CALL quando Elliott indica PUT forte!

                # Score base com CONFLUÊNCIA INTELIGENTE (CALL)
                score = 0.50  # base reduzida, pois agora temos validação de entrada
                score += min(0.18, (size_A / atr_val - IMPULSO_MIN_ATR) * 0.10)
                score += min(0.12, (eff_A - MIN_EFF_A) * 0.45)
                score += min(0.12, (RETR_MAX - retr) * 0.16)
                score += 0.03 if pb_len >= 2 else 0.00
                score -= min(0.10, max(0.0, (flips_frac - 0.30) * 0.25))

                # ⭐ BÔNUS POR ELLIOTT WAVE COMPLETO (NOVO)
                # Se detectamos padrão Elliott válido E direção coincide → BÔNUS GRANDE
                if elliott_valid and elliott_direction == "CALL" and elliott_entry == "CALL":
                    score += 0.25  # +25% por padrão Elliott confirmado
                    score += elliott_confidence * 0.15  # até +15% pela confiança do Elliott
                elif elliott_valid and elliott_direction == "CALL":
                    score += 0.10  # +10% se Elliott é CALL mas entry_point diferente
                
                # Se Elliott indica direção OPOSTA (mas confiança baixa), penaliza forte
                if elliott_valid and elliott_direction == "PUT":
                    score -= 0.30  # -30% por conflito com Elliott (aumentado de -15%)

                # BÔNUS POR QUALIDADE DE CONTEXTO
                score += market_quality * 0.12

                # BÔNUS POR FORÇA DA TENDÊNCIA
                score += trend_strength * 0.06

                # BÔNUS POR QUALIDADE DA ENTRADA (NOVO - IMPORTANTE)
                score += entry_confidence * 0.20  # até +20% pela confiança da entrada
                score += entry_momentum * 0.08    # até +8% pelo momentum
                score += entry_alignment * 0.06   # até +6% pelo alinhamento

                # ⭐ BÔNUS POR CONFLUÊNCIA COM LINHA DE TENDÊNCIA (LTA)
                if has_lt and lt_confluence > 0.8:
                    score += 0.20  # BÔNUS GRANDE se tocou perfeitamente a LTA
                elif has_lt and lt_confluence > 0.5:
                    score += 0.12  # Bônus médio se próximo da LTA
                elif has_lt and lt_confluence > 0.2:
                    score += 0.05  # Bônus pequeno

                # ⭐ BÔNUS POR CONFLUÊNCIA COM SUPORTE
                if has_sr and sr_confluence > 0.8:
                    score += 0.20  # BÔNUS GRANDE se tocou perfeitamente o suporte
                elif has_sr and sr_confluence > 0.5:
                    score += 0.12  # Bônus médio
                elif has_sr and sr_confluence > 0.2:
                    score += 0.05  # Bônus pequeno

                # ⭐⭐⭐ BÔNUS ESPECIAL: LTA COM SEGUNDO TOQUE + PAVIO REJEIÇÃO (ESTRATÉGIA PRINCIPAL) ⭐⭐⭐
                if lt_call_bonus > 0:
                    score += lt_call_bonus  # +0.20 a +0.25 por LTA com segundo toque e rejeição

                # PENALIZAÇÃO POR RISCO ALTO
                if risk_atr > 1.0:
                    score -= 0.05

                # BÔNUS POR CONFLUÊNCIA DE SINAIS
                confluence_bonus = 0.0
                if (market_quality > 0.60 and eff_A > 0.65 and 0.30 <= retr <= 0.50 and
                    entry_confidence > 0.65 and entry_alignment > 0.60):
                    confluence_bonus += 0.12  # setup perfeito
                elif market_quality > 0.50 and eff_A > 0.60 and entry_confidence > 0.55:
                    confluence_bonus += 0.06  # bom setup

                score += confluence_bonus
                score = float(max(0.0, min(1.0, score)))

                # Armazena informações de reversão para log
                reversal_info = detect_reversal_zone(df_m1, atr_val, "CALL")

                setup = {
                    "trade": True, "dir": "CALL", "score": score,
                    # campos para IA:
                    "pb_len": pb_len, "retr": float(retr),
                    "A_atr": float(size_A / max(atr_val, 1e-9)),
                    "effA": float(eff_A),
                    "flips": float(flips_frac),
                    "comp": float(comp),
                    "late": float(late_ext),
                    "distBreak": float(dist),
                    # contexto de mercado:
                    "market_quality": float(market_quality),
                    "context": str(context.get("context", "?")),
                    "confluence_bonus": float(confluence_bonus),
                    "trend_strength": float(trend_strength),
                    "trend_reason": str(trend_validation.get("reason", "?")),
                    # NOVO: informações de tendência maior
                    "atr_position": int(atr_position),
                    "atr_ts_value": float(atr_ts_value),
                    "market_trend": str(market_trend),
                    "market_allowed": str(market_allowed),
                    # NOVO: informações de reversão
                    "reversal_check": str(reversal_info.get("reason", "ok")),
                    "reversal_confidence": float(reversal_info.get("confidence", 0.0)),
                    # validação de entrada (NOVO):
                    "entry_confidence": float(entry_confidence),
                    "entry_momentum": float(entry_momentum),
                    "entry_alignment": float(entry_alignment),
                    "risk_atr": float(risk_atr),
                    # confluência LT:
                    "lt_confluence": float(lt_confluence),
                    "has_lt": has_lt,
                    # ⭐ NOVO: LTA com segundo toque
                    "lt_second_touch": has_lt_signal and lt_signal_type == "LTA",
                    "lt_touches": lt_touches,
                    "lt_rejection": lt_rejection,
                    "lt_bonus": lt_call_bonus,
                    "lt_reason": lt_reason,
                    # NOVO: Elliott Wave Analysis
                    "elliott_valid": bool(elliott_valid),
                    "elliott_direction": str(elliott_direction),
                    "elliott_entry": str(elliott_entry) if elliott_entry else "N/A",
                    "elliott_confidence": float(elliott_confidence),
                    "elliott_wave": int(elliott_current_wave),
                    "reasons": [
                        "Elliott_Wave_CALL(Onda3/5)",
                        f"🎯TREND={market_trend}(ATR={atr_position})",
                        f"Impulso={size_A/atr_val:.2f}ATR", f"retr={retr:.2f}", f"pb={pb_len}",
                        f"effA={eff_A:.2f}",
                        f"chop={flips_frac:.2f}/{eff_zone:.2f}",
                        f"comp={comp:.2f}ATR",
                        f"late={late_ext:.2f}ATR",
                        f"distBreak={dist:.2f}ATR",
                        f"ctx={context.get('context','?')}({market_quality:.2f})",
                        f"conf_bonus={confluence_bonus:.2f}",
                        f"trend={trend_validation.get('reason','?')}({trend_strength:.3f})",
                        f"entry_conf={entry_confidence:.2f}",
                        f"entry_mom={entry_momentum:.2f}",
                        f"entry_align={entry_alignment:.2f}",
                        f"⭐LTA={lt_confluence:.2f}" if has_lt else "sem_LTA",
                        f"🎯LTA_2°TOQUE={lt_reason}(+{lt_call_bonus:.2f})" if lt_call_bonus > 0 else "",
                        f"🌊Elliott={elliott_direction}(wave{elliott_current_wave},conf={elliott_confidence:.2f})" if elliott_valid else "sem_Elliott"
                    ]
                }
            else:  # dir_entrada == "PUT"
                # PUT: Impulso de QUEDA → Pullback sobe → Rompe para BAIXO (continuação)
                pb_high = float(pb["high"].max())
                if not (c1 < pb_high - BREAK_MARGIN_ATR * atr_val):
                    continue
                if q["lower_frac"] > MAX_WICK_AGAINST:
                    continue

                dist = (pb_high - c1) / max(atr_val, 1e-9)
                if dist > MAX_BREAK_DISTANCE_ATR:
                    continue

                # 🚨 FILTRO DESATIVADO: BLOQUEIA PUT PERTO DE SUPORTE
                # if BLOCK_PUT_NEAR_SUPPORT:
                #     _, sup_levels = strong_sr_levels_last200(df_m1, atr_val)
                #     if sup_levels:
                #         for sup_lvl, sup_touches in sup_levels:
                #             dist_to_sup = (c1 - sup_lvl) / max(atr_val, 1e-9)
                #             if 0 < dist_to_sup < 0.5 and sup_touches >= 2:
                #                 continue

                # VALIDAÇÃO DE QUALIDADE DA ENTRADA (NOVO)
                entry_validation = validate_entry_quality(df_m1, atr_val, "PUT", c1, pb_high, pb_low)
                if not entry_validation.get("valid", False):
                    continue  # Pula entrada se não passar na validação
                
                # 🚨 FILTRO DESATIVADO: VERIFICA ALINHAMENTO MÍNIMO
                entry_alignment = float(entry_validation.get("alignment", 0.0))
                # if entry_alignment < MIN_ALIGNMENT_RATIO:
                #     continue  # DESATIVADO - não bloqueia por alinhamento
                
                # 🚨 FILTRO DESATIVADO: VERIFICA MOMENTUM NA DIREÇÃO CORRETA
                # if MIN_MOMENTUM_ALIGNMENT:
                #     entry_momentum_dir = entry_validation.get("momentum_direction", "neutral")
                #     if entry_momentum_dir == "wrong":
                #         continue  # DESATIVADO - não bloqueia por momentum
                
                # DESATIVADO: Exigia corpo forte contra tendência
                body_ratio = float(entry_validation.get("body_ratio", 0.0))
                # if market_trend == "UP" and green_ratio >= 0.60 and body_ratio < 0.70:
                #     continue

                entry_confidence = float(entry_validation.get("confidence", 0.0))
                risk_atr = float(entry_validation.get("risk_atr", 0.0))
                entry_momentum = float(entry_validation.get("momentum", 0.0))

                # ⭐ DETECÇÃO DE LTB COM SEGUNDO TOQUE + PAVIO DE REJEIÇÃO (ESTRATÉGIA PRINCIPAL)
                lt_second_touch = detect_lt_second_touch(df_m1, atr_val, lookback=50)
                has_lt_signal = lt_second_touch.get("has_lt", False)
                lt_signal_type = lt_second_touch.get("lt_type", None)
                lt_signal_dir = lt_second_touch.get("signal", None)
                lt_touches = lt_second_touch.get("touches", 0)
                lt_rejection = lt_second_touch.get("rejection_strength", 0.0)
                lt_score_bonus = lt_second_touch.get("score_bonus", 0.0)
                lt_reason = lt_second_touch.get("reason", "")
                
                # Se LTB detectada com segundo toque e direção é PUT = BÔNUS GRANDE!
                lt_put_bonus = 0.0
                if has_lt_signal and lt_signal_type == "LTB" and lt_signal_dir == "PUT":
                    lt_put_bonus = lt_score_bonus  # 0.20 a 0.25 de bônus

                # CONFLUÊNCIA COM LINHA DE TENDÊNCIA (LTB para PUT)
                lt_conf = check_trendline_confluence(df_m1, pb_high, pb_low, "PUT", atr_val)
                lt_confluence = float(lt_conf.get("confluence", 0.0))
                has_lt = lt_conf.get("has_trendline", False)

                # CONFLUÊNCIA COM SUPORTE/RESISTÊNCIA (Resistencia para PUT)
                sr_conf = check_sr_confluence(df_m1, pb_low, pb_high, "PUT", atr_val)
                sr_confluence = float(sr_conf.get("confluence", 0.0))
                has_sr = sr_conf.get("has_sr", False)

                # ⚠️ CONFLUÊNCIA OPCIONAL: Bônus se tiver LTB OU Resistencia (não bloqueia mais)
                # if not has_lt and not has_sr:
                #     continue  # DESATIVADO - permite entrada sem confluência

                # 🚫 BLOQUEIO CRÍTICO: Se Elliott Wave indica direção OPOSTA com alta confiança, BLOQUEIA!
                if elliott_valid and elliott_direction == "CALL" and elliott_confidence >= 0.70:
                    continue  # BLOQUEIA PUT quando Elliott indica CALL forte!

                # Score base com CONFLUÊNCIA INTELIGENTE (PUT)
                score = 0.50  # base reduzida, pois agora temos validação de entrada
                score += min(0.18, (size_A / atr_val - IMPULSO_MIN_ATR) * 0.10)
                score += min(0.12, (eff_A - MIN_EFF_A) * 0.45)
                score += min(0.12, (RETR_MAX - retr) * 0.16)
                score += 0.03 if pb_len >= 2 else 0.00
                score -= min(0.10, max(0.0, (flips_frac - 0.30) * 0.25))

                # ⭐ BÔNUS POR ELLIOTT WAVE COMPLETO (NOVO)
                # Se detectamos padrão Elliott válido E direção coincide → BÔNUS GRANDE
                if elliott_valid and elliott_direction == "PUT" and elliott_entry == "PUT":
                    score += 0.25  # +25% por padrão Elliott confirmado
                    score += elliott_confidence * 0.15  # até +15% pela confiança do Elliott
                elif elliott_valid and elliott_direction == "PUT":
                    score += 0.10  # +10% se Elliott é PUT mas entry_point diferente
                
                # Se Elliott indica direção OPOSTA (mas confiança baixa), penaliza forte
                if elliott_valid and elliott_direction == "CALL":
                    score -= 0.30  # -30% por conflito com Elliott (aumentado de -15%)

                # BÔNUS POR QUALIDADE DE CONTEXTO
                score += market_quality * 0.12

                # BÔNUS POR FORÇA DA TENDÊNCIA
                score += trend_strength * 0.06

                # BÔNUS POR QUALIDADE DA ENTRADA (NOVO - IMPORTANTE)
                score += entry_confidence * 0.20  # até +20% pela confiança da entrada
                score += entry_momentum * 0.08    # até +8% pelo momentum
                score += entry_alignment * 0.06   # até +6% pelo alinhamento

                # ⭐ BÔNUS POR CONFLUÊNCIA COM LINHA DE TENDÊNCIA (LTB)
                if has_lt and lt_confluence > 0.8:
                    score += 0.20  # BÔNUS GRANDE se tocou perfeitamente a LTB
                elif has_lt and lt_confluence > 0.5:
                    score += 0.12  # Bônus médio se próximo da LTB
                elif has_lt and lt_confluence > 0.2:
                    score += 0.05  # Bônus pequeno

                # ⭐ BÔNUS POR CONFLUÊNCIA COM RESISTÊNCIA
                if has_sr and sr_confluence > 0.8:
                    score += 0.20  # BÔNUS GRANDE se tocou perfeitamente a resistencia
                elif has_sr and sr_confluence > 0.5:
                    score += 0.12  # Bônus médio
                elif has_sr and sr_confluence > 0.2:
                    score += 0.05  # Bônus pequeno

                # ⭐⭐⭐ BÔNUS ESPECIAL: LTB COM SEGUNDO TOQUE + PAVIO REJEIÇÃO (ESTRATÉGIA PRINCIPAL) ⭐⭐⭐
                if lt_put_bonus > 0:
                    score += lt_put_bonus  # +0.20 a +0.25 por LTB com segundo toque e rejeição

                # PENALIZAÇÃO POR RISCO ALTO
                if risk_atr > 1.0:
                    score -= 0.05

                # BÔNUS POR CONFLUÊNCIA DE SINAIS
                confluence_bonus = 0.0
                if (market_quality > 0.60 and eff_A > 0.65 and 0.30 <= retr <= 0.50 and
                    entry_confidence > 0.65 and entry_alignment > 0.60):
                    confluence_bonus += 0.12  # setup perfeito
                elif market_quality > 0.50 and eff_A > 0.60 and entry_confidence > 0.55:
                    confluence_bonus += 0.06  # bom setup

                score += confluence_bonus
                score = float(max(0.0, min(1.0, score)))

                # Armazena informações de reversão para log
                reversal_info = detect_reversal_zone(df_m1, atr_val, "PUT")

                setup = {
                    "trade": True, "dir": "PUT", "score": score,
                    "pb_len": pb_len, "retr": float(retr),
                    "A_atr": float(size_A / max(atr_val, 1e-9)),
                    "effA": float(eff_A),
                    "flips": float(flips_frac),
                    "comp": float(comp),
                    "late": float(late_ext),
                    "distBreak": float(dist),
                    # contexto de mercado:
                    "market_quality": float(market_quality),
                    "context": str(context.get("context", "?")),
                    "confluence_bonus": float(confluence_bonus),
                    "trend_strength": float(trend_strength),
                    "trend_reason": str(trend_validation.get("reason", "?")),
                    # NOVO: informações de tendência maior
                    "atr_position": int(atr_position),
                    "atr_ts_value": float(atr_ts_value),
                    "market_trend": str(market_trend),
                    "market_allowed": str(market_allowed),
                    # NOVO: informações de reversão
                    "reversal_check": str(reversal_info.get("reason", "ok")),
                    "reversal_confidence": float(reversal_info.get("confidence", 0.0)),
                    # validação de entrada (NOVO):
                    "entry_confidence": float(entry_confidence),
                    "entry_momentum": float(entry_momentum),
                    "entry_alignment": float(entry_alignment),
                    "risk_atr": float(risk_atr),
                    # confluência LT:
                    "lt_confluence": float(lt_confluence),
                    "has_lt": has_lt,
                    # ⭐ NOVO: LTB com segundo toque
                    "lt_second_touch": has_lt_signal and lt_signal_type == "LTB",
                    "lt_touches": lt_touches,
                    "lt_rejection": lt_rejection,
                    "lt_bonus": lt_put_bonus,
                    "lt_reason": lt_reason,
                    # NOVO: Elliott Wave Analysis
                    "elliott_valid": bool(elliott_valid),
                    "elliott_direction": str(elliott_direction),
                    "elliott_entry": str(elliott_entry) if elliott_entry else "N/A",
                    "elliott_confidence": float(elliott_confidence),
                    "elliott_wave": int(elliott_current_wave),
                    "reasons": [
                        "Elliott_Wave_PUT(Onda3/5)",
                        f"🎯TREND={market_trend}(ATR={atr_position})",
                        f"Impulso={size_A/atr_val:.2f}ATR", f"retr={retr:.2f}", f"pb={pb_len}",
                        f"effA={eff_A:.2f}",
                        f"chop={flips_frac:.2f}/{eff_zone:.2f}",
                        f"comp={comp:.2f}ATR",
                        f"late={late_ext:.2f}ATR",
                        f"distBreak={dist:.2f}ATR",
                        f"ctx={context.get('context','?')}({market_quality:.2f})",
                        f"conf_bonus={confluence_bonus:.2f}",
                        f"trend={trend_validation.get('reason','?')}({trend_strength:.3f})",
                        f"entry_conf={entry_confidence:.2f}",
                        f"entry_mom={entry_momentum:.2f}",
                        f"entry_align={entry_alignment:.2f}",
                        f"⭐LTB={lt_confluence:.2f}" if has_lt else "sem_LTB",
                        f"🎯LTB_2°TOQUE={lt_reason}(+{lt_put_bonus:.2f})" if lt_put_bonus > 0 else "",
                        f"🌊Elliott={elliott_direction}(wave{elliott_current_wave},conf={elliott_confidence:.2f})" if elliott_valid else "sem_Elliott"
                    ]
                }

            if best is None or setup["score"] > best["score"]:
                best = setup

    if best is None:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["sem_padrao_Elliott_valido"]}

    # bloqueio final SR forte no momento do sinal
    block_final = sr_block_directional_multi(df_m1, atr_val, best["dir"])
    if block_final:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": [block_final]}

    # ===================== FILTROS DE QUALIDADE INTELIGENTES (NOVOS) =====================
    ctx_quality = float(best.get("market_quality", 0.0))
    entry_conf = float(best.get("entry_confidence", 0.0))
    confl_bonus = float(best.get("confluence_bonus", 0.0))
    lt_conf = float(best.get("lt_confluence", 0.0))
    has_lt = best.get("has_lt", False)

    # 1. BLOQUEAR APENAS CONTEXTO EXTREMAMENTE RUIM
    if ctx_quality < MIN_CONTEXT_QUALITY:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
                "reasons": [f"contexto_extremamente_ruim(quality={ctx_quality:.2f}<{MIN_CONTEXT_QUALITY})"]}

    # 2. BLOQUEAR APENAS ENTRADA MUITO FRACA
    if entry_conf < MIN_ENTRY_CONFIDENCE:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
                "reasons": [f"entrada_muito_fraca(conf={entry_conf:.2f}<{MIN_ENTRY_CONFIDENCE})"]}

    # 3. BLOQUEAR COMBINAÇÃO PERIGOSA: contexto ruim + entrada fraca + sem confluência
    # Este é o padrão que causou os 3 losses: ctx=0.33-0.39, entry=0.44, conf=0.00
    if ctx_quality < 0.40 and entry_conf < 0.48 and confl_bonus < 0.02:
        # Permite APENAS se tiver linha de tendência muito forte (LT > 0.6)
        if not (has_lt and lt_conf > 0.6):
            return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
                    "reasons": [f"⚠️tudo_ruim(ctx={ctx_quality:.2f},entry={entry_conf:.2f},confl={confl_bonus:.2f},sem_LT_forte)"]}

    # 4. BLOQUEAR SCORE INFLADO SEM QUALIDADE (padrão dos losses: score=1.0 mas tudo ruim)
    if best["score"] > 0.85:
        # Se score alto mas TUDO indica problema, é entrada falsa
        if ctx_quality < 0.35 and entry_conf < 0.46 and confl_bonus == 0.0:
            if not (has_lt and lt_conf > 0.7):
                return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
                        "reasons": [f"⚠️score_inflado_setup_ruim(score={best['score']:.2f},ctx={ctx_quality:.2f},entry={entry_conf:.2f})"]}

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

    # TURBO
    try:
        ok, op_id = safe_call(iq, iq.buy, valor, ativo, d, int(EXP_FIXA))
        if ok and op_id:
            return ("turbo", int(op_id))
        log.warning(paint(f"[ORDEM-FAIL] TURBO ok={ok} op_id={op_id}", C.Y))
    except Exception as e:
        log.warning(paint(f"[ORDEM-EXC] TURBO {e}", C.Y))

    # DIGITAL
    try:
        ok, op_id = safe_call(iq, iq.buy_digital_spot, ativo, valor, d, int(EXP_FIXA))
        if ok and op_id:
            return ("digital", int(op_id))
        log.warning(paint(f"[ORDEM-FAIL] DIGITAL ok={ok} op_id={op_id}", C.Y))
    except Exception as e:
        log.warning(paint(f"[ORDEM-EXC] DIGITAL {e}", C.Y))

    return None

def _call_with_timeout(func, args: tuple, timeout_s: float):
    """
    Executa func(*args) em thread separada, retorna None se ultrapassar timeout_s.
    """
    result_box: Dict[str, Any] = {}
    def runner():
        try:
            result_box["value"] = func(*args)
        except Exception as e:
            result_box["error"] = e
    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return None  # timeout
    if "error" in result_box:
        raise result_box["error"]
    return result_box.get("value")

def wait_result(iq: IQ_Option, op_type: str, op_id: int) -> float:
    """
    Aguarda resultado da operação com timeout para evitar travamentos.
    Retorna o PNL (lucro/prejuízo) da operação.
    """
    import threading
    
    # Normalizar order_id (pode vir como lista)
    if isinstance(op_id, list) and op_id:
        op_id = op_id[0]
    try:
        op_id = int(op_id)
    except:
        pass
    
    # Log para aparecer no chat
    sys.stdout.write(f"[WS_AUTO_AI] Aguardando resultado da operacao (exp={EXP_FIXA}min)...\n")
    sys.stdout.flush()
    log.info(f"[WS] Verificando resultado de operacao ID={op_id} tipo={op_type}...")
    
    # Esperar tempo da expiração + margem (EXP_FIXA em minutos + 60s de margem)
    wait_time = (EXP_FIXA * 60) + 60  # segundos
    time.sleep(wait_time)
    
    # Agora sim, verificar resultado com timeout curto
    max_attempts = 20  # 20 tentativas = 10 segundos
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        
        try:
            if op_type == "turbo":
                # Tentar socket_option_closed primeiro (mais confiável)
                if hasattr(iq, 'socket_option_closed') and isinstance(iq.socket_option_closed, dict):
                    for oid, profit_data in iq.socket_option_closed.items():
                        oid_int = None
                        if isinstance(oid, list) and oid:
                            oid = oid[0]
                        try:
                            oid_int = int(oid)
                        except:
                            pass
                        
                        if oid == op_id or oid_int == op_id:
                            profit = float(profit_data) if isinstance(profit_data, (int, float)) else float(profit_data.get("profit", 0))
                            status_msg = 'WIN' if profit > 0 else 'LOSS' if profit < 0 else 'EMPATE'
                            # Não enviar duplicado - será enviado no main
                            log.info(f"[WS] Resultado verificado (socket): {status_msg} ${profit:.2f}")
                            return profit
                
                # Fallback: check_win_v4 com timeout
                result = _call_with_timeout(iq.check_win_v4, (op_id,), 2.0)
                if result is not None:
                    # result pode ser tupla (status, profit) ou só float
                    if isinstance(result, tuple) and len(result) >= 2:
                        ok, profit = result[0], result[1]
                        if ok:
                            profit = float(profit)
                            status_msg = 'WIN' if profit > 0 else 'LOSS' if profit < 0 else 'EMPATE'
                            # Não enviar duplicado - será enviado no main
                            log.info(f"[WS] Resultado verificado (v4): {status_msg} ${profit:.2f}")
                            return profit
                    elif isinstance(result, (int, float)):
                        profit = float(result)
                        status_msg = 'WIN' if profit > 0 else 'LOSS' if profit < 0 else 'EMPATE'
                        # Não enviar duplicado - será enviado no main
                        log.info(f"[WS] Resultado verificado (v4): {status_msg} ${profit:.2f}")
                        return profit
            
            else:  # digital
                # check_win_digital_v2 com timeout
                result = _call_with_timeout(iq.check_win_digital_v2, (op_id,), 2.0)
                if result is not None:
                    if isinstance(result, tuple) and len(result) >= 2:
                        status, profit = result[0], result[1]
                        if status == "equal":
                            # Não enviar duplicado - será enviado no main
                            log.info(f"[WS] Resultado: EMPATE (devolucao)")
                            return 0.0
                        profit = float(profit)
                        status_msg = 'WIN' if profit > 0 else 'LOSS'
                        # Não enviar duplicado - será enviado no main
                        log.info(f"[WS] Resultado verificado (digital): {status_msg} ${profit:.2f}")
                        return profit
                    elif isinstance(result, (int, float)):
                        profit = float(result)
                        status_msg = 'WIN' if profit > 0 else 'LOSS' if profit < 0 else 'EMPATE'
                        # Não enviar duplicado - será enviado no main
                        log.info(f"[WS] Resultado verificado (digital): {status_msg} ${profit:.2f}")
                        return profit
        
        except Exception as e:
            if attempt % 5 == 0:  # Log a cada 5 tentativas
                log.info(f"[WS] Tentativa {attempt}/{max_attempts} - erro: {e}")
        
        time.sleep(0.5)
    
    # Se chegou aqui, timeout total
    sys.stdout.write(f"[WS_AUTO_AI] AVISO: Timeout ao verificar resultado - considerando empate\n")
    sys.stdout.flush()
    log.warning(f"[WS] TIMEOUT ao verificar resultado apos {max_attempts} tentativas")
    ensure_connected(iq)
    return 0.0  # Considerar empate em caso de timeout

# ===================== MAIN =====================
def main():
    # Aviso simples: operar envolve risco.
    iq: Optional[IQ_Option] = None
    iq = ensure_connected(iq)

    # 🎯 MODO LTA/LTB COM SEGUNDO TOQUE + PAVIO REJEIÇÃO
    log.info(paint("=" * 65, C.G))
    log.info(paint("🎯 ESTRATÉGIA LTA/LTB - SEGUNDO TOQUE + PAVIO REJEIÇÃO 🎯", C.G))
    log.info(paint("=" * 65, C.G))
    log.info(paint("📈 LTA (Linha Tendência Alta): CALL no 2° toque com pavio inferior", C.B))
    log.info(paint("📉 LTB (Linha Tendência Baixa): PUT no 2° toque com pavio superior", C.B))
    log.info(paint(f"📊 Mínimo de toques na linha: {LT_MIN_TOUCHES}", C.B))
    log.info(paint(f"📊 Pavio de rejeição mínimo: {LT_WICK_REJECTION_MIN:.0%} da vela", C.B))
    log.info(paint(f"📊 Bônus no score: +{LT_BONUS_SCORE:.0%}", C.B))
    log.info(paint(f"📊 Score mínimo entrada: {GATE_MIN_SCORE:.0%}", C.B))
    log.info(paint(f"⚠️  SÓ BLOQUEIA MERCADO HORRÍVEL (lateral extremo)", C.Y))
    log.info(paint("=" * 65, C.G))

    log.info("Iniciando: Pernada B (M1) | LTA/LTB 2° TOQUE | EXECUÇÃO ON")

    stats = _safe_load_json(AI_STATS_FILE) if IA_ON else {"meta": {"total": 0}, "arms": {}}
    if IA_ON:
        log.info(f"IA=ON | file={AI_STATS_FILE} | min_samples={AI_MIN_SAMPLES} | min_prob={AI_MIN_PROB:.2f} | conf_min={AI_CONF_MIN:.2f}")

    # 🧠 INICIALIZA APRENDIZADO DO FIREBASE
    if AI_LEARNING_ENABLED and AI_LEARNING:
        try:
            log.info("🧠 AI Learning: Carregando aprendizado do Firebase...")
            AI_LEARNING.refresh_learning(force=True)
            log.info(AI_LEARNING.get_learning_summary())
        except Exception as e:
            log.warning(f"⚠️ Erro ao carregar aprendizado: {e}")

    # GESTÃO DE BANCA (NOVO)
    try:
        saldo_inicial = float(iq.get_balance())
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
        iq = ensure_connected(iq)

        # VERIFICAR SE ATINGIU META OU STOP (NOVO)
        try:
            saldo_atual = float(iq.get_balance())
            deve_parar, lucro_percent = verificar_meta_atingida(saldo_inicial, saldo_atual)
            if deve_parar:
                lucro_abs = saldo_atual - saldo_inicial
                if lucro_percent >= META_LUCRO_PERCENT:
                    log.info(paint(f"🎯 META ATINGIDA! Lucro: {lucro_abs:.2f} ({lucro_percent:.2f}%) | Parando operação.", C.G))
                else:
                    log.info(paint(f"🛑 STOP LOSS! Perda: {lucro_abs:.2f} ({lucro_percent:.2f}%) | Parando operação.", C.R))
                break
        except Exception as e:
            log.warning(f"Erro ao verificar meta: {e}")

        ativos = obter_top_ativos_otc(iq)
        if not ativos:
            log.warning("Sem ativos com payout mínimo. Tentando em 10s...")
            time.sleep(10)
            continue

        # decide no fim do candle, analisando candle FECHADO
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

        # ===================== AGENTE IA - ANÁLISE CONTEXTUAL COMPLETA =====================
        # O Agente analisa as últimas 100 velas e decide se o trade é válido
        df_ativo = get_candles_df(iq, ativo, TF_M1, N_M1, end_ts=end_ts_closed(TF_M1))
        if df_ativo is not None and len(df_ativo) >= 100:
            # Coleta contexto para o Agente
            market_struct = detect_market_structure_trend(df_ativo, lookback=100, swing_window=5)
            trend = market_struct.get("trend", "UNKNOWN")
            suggested_dir = market_struct.get("suggested_direction", "ANALYZE")
            trend_strength = market_struct.get("strength", 0.0)
            up_sc = market_struct.get("up_score", 0.0)
            down_sc = market_struct.get("down_score", 0.0)

            # Agente IA analisa e decide
            agent_decision = ai_agent_analyze(
                df=df_ativo,
                setup=setup,
                proposed_direction=final_dir,
                market_trend=trend,
                suggested_direction=suggested_dir,
                trend_strength=trend_strength,
                up_score=up_sc,
                down_score=down_sc,
                atr_val=atr_val
            )

            if not agent_decision["approved"]:
                log.info(paint(
                    f"[AGENT-REJECT] {ativo} {final_dir} | {agent_decision['reason']} | Sugestao={suggested_dir}",
                    C.Y
                ))
                wait_for_next_open(TF_M1)
                cooldown[ativo] = time.time()
                continue

            # Se agente sugerir direcao diferente, usa a do agente
            if agent_decision.get("override_direction"):
                old_dir = final_dir
                final_dir = agent_decision["override_direction"]
                log.info(paint(
                    f"[AGENT-OVERRIDE] {ativo} | {old_dir} -> {final_dir} | {agent_decision['reason']}",
                    C.B
                ))

            log.info(paint(
                f"[AGENT-OK] {ativo} {final_dir} | {agent_decision['reason']} | conf={agent_decision.get('confidence', 0):.0%}",
                C.G
            ))

        # ===================== SINAL CONFIRMADO PELO AGENTE =====================
        log.info(paint(
            f"[SINAL-HARD] {ativo} -> {final_dir} | score={score:.2f} | ATR={atr_val:.6f} | {','.join(setup.get('reasons', []))}",
            dir_color(final_dir)
        ))

        # ===================== 🎯 FILTRO DE CONFLUÊNCIA ULTRA SELETIVO =====================
        # Este é o filtro MAIS IMPORTANTE - só entra se múltiplos sinais alinharem
        if BLOCK_LOW_CONFLUENCE:
            confluence_score, confluence_reasons, should_enter = calculate_confluence_score(
                df_ativo, final_dir, setup, atr_val
            )
            
            log.info(paint(
                f"[🎯 CONFLUÊNCIA] {ativo} {final_dir} | Score: {confluence_score}/6 | {' | '.join(confluence_reasons[:4])}",
                C.G if should_enter else C.Y
            ))
            
            if not should_enter:
                log.info(paint(
                    f"[🚫 BLOQUEIO-CONFLUÊNCIA] {ativo} {final_dir} | Confluência {confluence_score}/6 < {MIN_CONFLUENCE_SIGNALS} necessários",
                    C.R
                ))
                # Salva para IA aprender que este setup foi bloqueado
                setup["confluence_score"] = confluence_score
                setup["confluence_blocked"] = True
                wait_for_next_open(TF_M1)
                cooldown[ativo] = time.time()
                continue
            
            # Adiciona confluência ao setup para IA aprender
            setup["confluence_score"] = confluence_score
            setup["confluence_reasons"] = confluence_reasons

        # ===================== 🧠 FILTRO DE APRENDIZADO (FIREBASE) =====================
        # Consulta losses anteriores e bloqueia setups que causaram prejuízo
        if AI_LEARNING_ENABLED and AI_LEARNING:
            # Prepara contexto de mercado e entrada para análise
            market_ctx = {}
            entry_qual = {}
            
            # Busca contexto do setup
            if df_ativo is not None and len(df_ativo) >= 20:
                try:
                    recent = df_ativo.tail(20)
                    green_count = (recent['close'] > recent['open']).sum()
                    red_count = len(recent) - green_count
                    
                    # Detecta consolidação
                    std_ratio = df_ativo['close'].tail(20).std() / df_ativo['close'].mean()
                    is_consolidating = std_ratio < 0.001
                    
                    # Calcula ATR
                    tr_calc = df_ativo['high'] - df_ativo['low']
                    atr_mean = tr_calc.tail(14).mean()
                    volatility = "high" if atr_val > atr_mean * 1.5 else "low"
                    
                    # Suporte/Resistência
                    resistance = df_ativo['high'].tail(50).max()
                    support = df_ativo['low'].tail(50).min()
                    current_price = df_ativo['close'].iloc[-1]
                    near_resistance = abs(current_price - resistance) / current_price < 0.001
                    near_support = abs(current_price - support) / current_price < 0.001
                    
                    market_ctx = {
                        "trend": "bullish" if green_count > red_count else "bearish" if red_count > green_count else "neutral",
                        "green_candles": int(green_count),
                        "red_candles": int(red_count),
                        "is_consolidating": bool(is_consolidating),
                        "volatility": volatility,
                        "near_resistance": bool(near_resistance),
                        "near_support": bool(near_support)
                    }
                    
                    # Qualidade da entrada
                    prev_5 = df_ativo.tail(6).iloc[:-1]  # 5 velas antes da entrada
                    if final_dir == "CALL":
                        aligned = (prev_5['close'] > prev_5['open']).sum()
                    else:
                        aligned = (prev_5['close'] < prev_5['open']).sum()
                    
                    alignment_ratio = aligned / 5.0
                    momentum = df_ativo['close'].iloc[-1] - df_ativo['close'].iloc[-6]
                    momentum_correct = (final_dir == "CALL" and momentum > 0) or (final_dir == "PUT" and momentum < 0)
                    
                    entry_candle = df_ativo.iloc[-1]
                    body = abs(entry_candle['close'] - entry_candle['open'])
                    range_candle = entry_candle['high'] - entry_candle['low']
                    body_ratio = body / range_candle if range_candle > 0 else 0
                    
                    entry_qual = {
                        "alignment_ratio": float(alignment_ratio),
                        "momentum_direction": "correct" if momentum_correct else "wrong",
                        "entry_quality": "strong" if body_ratio > 0.7 else "weak"
                    }
                except Exception as e:
                    log.warning(f"⚠️ Erro ao coletar contexto para aprendizado: {e}")
            
            # Verifica se deve bloquear baseado no aprendizado
            should_block, block_reason = AI_LEARNING.should_block_trade(
                ativo=ativo,
                direction=final_dir,
                score=score,
                conf=float(setup.get("score", 0.5)),
                market_context=market_ctx,
                entry_quality=entry_qual
            )
            
            if should_block:
                log.info(paint(
                    f"[🧠 AI-LEARN-BLOCK] {ativo} {final_dir} | {block_reason}",
                    C.Y
                ))
                wait_for_next_open(TF_M1)
                cooldown[ativo] = time.time()
                continue
            
            # Obtém penalidade e motivos (para log)
            penalty, penalty_reasons = AI_LEARNING.get_penalty_for_setup(
                ativo, final_dir, market_ctx, entry_qual
            )
            
            if penalty > 0.10:  # Log se penalidade significativa
                log.info(paint(
                    f"[🧠 AI-LEARN] {ativo} {final_dir} | penalty={penalty:.0%} | {'; '.join(penalty_reasons)}",
                    C.B
                ))

        # ===================== IA FILTRO =====================
        if IA_ON:
            pred = ai_predict(ativo, setup, stats)
            prob = float(pred["prob"])
            bayes = float(pred["bayes"])
            ucb01 = float(pred["ucb01"])
            conf = float(pred["conf"])
            n_arm = int(pred["n_arm"])
            
            # 🧠 APLICAR PENALIDADE DO AI LEARNING NA PROBABILIDADE
            prob_original = prob
            if AI_LEARNING and penalty > 0:
                # Reduz a probabilidade proporcional à penalidade
                prob = prob * (1 - penalty)
                log.info(paint(
                    f"[🧠 AI-LEARN] Prob ajustada: {prob_original:.2f} → {prob:.2f} (penalty={penalty:.0%})",
                    C.B
                ))

            if n_arm < AI_MIN_SAMPLES:
                log.info(paint(
                    f"[IA] {ativo} {final_dir} | conf={conf:.2f} | bayes={bayes:.2f} (n={n_arm}) | ucb01={ucb01:.2f} (n_arm={n_arm}) | warmup(prob={prob:.2f},conf={conf:.2f},n={n_arm})",
                    C.B
                ))
            else:
                log.info(paint(
                    f"[IA] {ativo} {final_dir} | conf={conf:.2f} | bayes={bayes:.2f} (n={n_arm}) | ucb01={ucb01:.2f} (n_arm={n_arm}) | prob={prob:.2f}",
                    C.B
                ))

            # regra: só bloqueia de verdade quando tem histórico suficiente
            if n_arm >= AI_MIN_SAMPLES:
                if (prob < AI_MIN_PROB) or (conf < AI_CONF_MIN):
                    log.info(paint(f"[IA-SKIP] {ativo} {final_dir} | prob={prob:.2f} conf={conf:.2f} n={n_arm}", C.Y))
                    wait_for_next_open(TF_M1)
                    cooldown[ativo] = time.time()
                    continue
            else:
                # warmup: não trava demais, mas se prob for muito baixa, pula
                if prob < (AI_MIN_PROB - 0.07):
                    log.info(paint(f"[IA-SKIP] {ativo} {final_dir} | warmup_prob={prob:.2f} n={n_arm}", C.Y))
                    wait_for_next_open(TF_M1)
                    cooldown[ativo] = time.time()
                    continue

        # entra na abertura do próximo M1
        wait_for_next_open(TF_M1)

        # STAKE DINÂMICO BASEADO NA BANCA (NOVO)
        stake = calcular_stake_dinamico(iq, STAKE_FIXA)
        log.info(paint(f"[{ativo}] 💵 Stake calculado: {stake:.2f}", C.B))

        op = enviar_ordem(iq, ativo, final_dir, stake)

        if not op:
            log.error(paint(f"[{ativo}] ❌ falhou enviar ordem (TURBO/DIGITAL).", C.R))
            cooldown[ativo] = time.time()
            continue

        op_type, op_id = op
        
        # Mensagem para aparecer no chat
        sys.stdout.write(f"[WS_AUTO_AI] ENTRADA: {ativo} {final_dir} | Stake: ${stake:.2f} | Exp: {EXP_FIXA}min\n")
        sys.stdout.flush()
        
        log.info(paint(
            f"[{ativo}] ✅ ORDEM ENVIADA {final_dir} exp={EXP_FIXA}m ({op_type}) | stake={stake:.2f}",
            dir_color(final_dir)
        ))

        res = wait_result(iq, op_type, op_id)

        total += 1
        if res > 0:
            wins += 1
            # Formato único para detecção sem duplicação
            sys.stdout.write(f">>> RESULTADO: WIN {res:.2f}\n")
            sys.stdout.flush()
            log.info(paint(f"[{ativo}] ✅ WIN {res:.2f}$", C.G))
        elif res < 0:
            # Formato único para detecção sem duplicação
            sys.stdout.write(f">>> RESULTADO: LOSS {res:.2f}\n")
            sys.stdout.flush()
            log.info(paint(f"[{ativo}] ❌ LOSS {res:.2f}$", C.R))
            
            # 🔍 ANÁLISE DE LOSS
            if LOSS_ANALYZER_ENABLED and LOSS_ANALYZER:
                try:
                    log.info("🔍 Iniciando análise de loss...")
                    # Aguarda 1 segundo para API estabilizar após resultado
                    time.sleep(1)
                    # Executa em thread separada para não bloquear
                    analysis_thread = threading.Thread(
                        target=LOSS_ANALYZER.analyze_loss,
                        args=(iq, op_id, ativo, final_dir, abs(res), setup),
                        daemon=True
                    )
                    analysis_thread.start()
                except Exception as e:
                    log.error(f"⚠️ Erro ao iniciar análise de loss: {e}")
        else:
            # Formato único para detecção sem duplicação
            sys.stdout.write(f">>> RESULTADO: EMPATE {res:.2f}\n")
            sys.stdout.flush()
            log.info(paint(f"[{ativo}] ⚪ EMPATE {res:.2f}$", C.B))

        # update IA após resultado
        if IA_ON:
            ai_update(ativo, setup, res, stats)
            _safe_save_json(AI_STATS_FILE, stats)

        acc = (wins / max(1, total)) * 100.0

        # EXIBIR PROGRESSO EM RELAÇÃO À META (NOVO)
        try:
            saldo_atual = float(iq.get_balance())
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
