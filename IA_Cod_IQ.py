# -*- coding: utf-8 -*-
"""
WS_AUTO_AI — Pernada B (M1) com:
✅ Candles FECHADOS (evita sinal fora da hora)
✅ Anti-lateral + Anti-esticado
✅ Filtro de SUPORTE/RESISTÊNCIA FORTE (usa >=200 velas e considera várias regiões)
✅ IA online simples (Bayes + UCB) aprendendo com seus próprios resultados (salva em JSON)
✅ Execução real (TURBO -> DIGITAL fallback)
✅ Análise de Loss com envio para Firebase
✅ Projeção de tendência (Trend Projection AI)

Requisitos:
pip install iqoptionapi pandas numpy requests
"""

import os
import time
import math
import json
import logging
import requests
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from iqoptionapi.stable_api import IQ_Option

# ===================== DETECTOR DE PADRÕES PROFISSIONAL =====================
try:
    from pattern_detector import (
        analyze_patterns,
        detect_candlestick_patterns,
        PatternType
    )
    PATTERN_DETECTOR_AVAILABLE = True
    print("[OK PADROES] Detector de 13 padroes de candlestick (80%+) ativado!")
except ImportError as e:
    PATTERN_DETECTOR_AVAILABLE = False
    print(f"[AVISO PADROES] Detector nao disponivel: {e}")

# ===================== MEMORIA INTELIGENTE (DESATIVADA - SUBSTITUIDA PELO CNN) =====================
# try:
#     from smart_memory import get_smart_memory, should_block_trade, record_trade_result
#     SMART_MEMORY_AVAILABLE = True
#     print("[OK MEMORIA] Sistema de memoria inteligente ativado!")
# except ImportError as e:
#     SMART_MEMORY_AVAILABLE = False
#     print(f"[AVISO MEMORIA] Memoria inteligente nao disponivel: {e}")
SMART_MEMORY_AVAILABLE = False
print("[INFO] Memoria inteligente DESATIVADA - usando novo sistema CNN")

# ===================== NOVO SISTEMA: REGIME FILTER + CNN + RISK CONTROL =====================
try:
    from regime_filter import RegimeFilter
    REGIME_FILTER_AVAILABLE = True
    print("[OK] Regime Filter ativado!")
except ImportError as e:
    REGIME_FILTER_AVAILABLE = False
    print(f"[AVISO] Regime Filter nao disponivel: {e}")

try:
    from neural_model import TradingCNN
    CNN_AVAILABLE = True
    print("[OK] Modelo CNN ativado!")
except ImportError as e:
    CNN_AVAILABLE = False
    print(f"[AVISO] Modelo CNN nao disponivel: {e}")

try:
    from risk_control import RiskControl
    RISK_CONTROL_AVAILABLE = True
    print("[OK] Risk Control ativado!")
except ImportError as e:
    RISK_CONTROL_AVAILABLE = False
    print(f"[AVISO] Risk Control nao disponivel: {e}")

# Instancias globais do novo sistema
regime_filter = RegimeFilter() if REGIME_FILTER_AVAILABLE else None
trading_cnn = TradingCNN() if CNN_AVAILABLE else None
risk_control = RiskControl() if RISK_CONTROL_AVAILABLE else None

# ===================== BACKEND URL =====================
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ===================== CONFIG =====================
EMAIL = os.getenv("IQ_EMAIL", "") or "wstrader@wstrader.onmicrosoft.com"
SENHA = os.getenv("IQ_PASS", "") or "P152030@w"
CONTA = os.getenv("IQ_CONTA", "REAL")

TF_M1 = 60
TF_M5 = 300  # Timeframe de 5 minutos para analise de tendencia
DECIDIR_ANTES_FECHAR_SEC = int(os.getenv("WS_DECIDIR_ANTES_FECHAR", "10"))
N_M1 = int(os.getenv("WS_N_M1", "340"))
N_M5 = int(os.getenv("WS_N_M5", "48"))  # 48 candles de 5min = 4 horas de contexto

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

COOLDOWN_ATIVO = int(os.getenv("WS_COOLDOWN_ATIVO", "20"))  # seg

# ===================== IA (ONLINE) - APRENDIZADO ADAPTATIVO MELHORADO =====================
IA_ON = (os.getenv("WS_AI_ON", "1").strip() == "1")  # LIGADO: aprende bloqueando losses
USE_BOS = (os.getenv("WS_USE_BOS", "0").strip() == "1")  # DESLIGADO: não usar BOS/CHOCH por padrão
AI_STATS_FILE = os.getenv("WS_AI_FILE", "ws_ai_stats_m1.json")
AI_MIN_SAMPLES = int(os.getenv("WS_AI_MIN_SAMPLES", "8"))    # REDUZIDO: 8 trades para aprender mais rápido
AI_MIN_PROB = float(os.getenv("WS_AI_MIN_PROB", "0.42"))     # AJUSTADO: probabilidade mínima
AI_MIN_WINRATE = float(os.getenv("WS_AI_MIN_WINRATE", "0.70"))  # BLOQUEIA: winrate < 70%
AI_CONF_MIN = float(os.getenv("WS_AI_CONF_MIN", "0.45"))     # AJUSTADO: confiança mínima

# ===================== CONSENSO DE 4 AGENTES (PADRÕES DE VELA) =====================
CANDLE_AGENTS_ON = (os.getenv("WS_CANDLE_AGENTS", "1").strip() == "1")
CANDLE_AGENT_MIN_TALIB = float(os.getenv("WS_CANDLE_AGENT_MIN_TALIB", "0.80"))
CANDLE_AGENT_MIN_REL = float(os.getenv("WS_CANDLE_AGENT_MIN_REL", "0.70"))

# Confirmação de padrão pela IA (prob/conf) e bloqueio por estatística de padrão
AI_CONFIRM_PATTERN = (os.getenv("WS_AI_CONFIRM_PATTERN", "1").strip() == "1")
AI_BLOCK_PATTERN_STATS = (os.getenv("WS_AI_BLOCK_PATTERN_STATS", "0").strip() == "1")

# ===== NOVOS PARÂMETROS DE APRENDIZADO AVANÇADO =====
AI_LOSS_WEIGHT = float(os.getenv("WS_AI_LOSS_WEIGHT", "2.5"))    # Losses pesam 2.5x mais (aprende rápido dos erros)
AI_RECENT_MEMORY = int(os.getenv("WS_AI_RECENT_MEMORY", "10"))   # Últimos 10 trades pesam 2x mais
AI_DECAY_RATE = float(os.getenv("WS_AI_DECAY_RATE", "0.98"))     # Decaimento temporal (trades antigos valem menos)
AI_CONSECUTIVE_LOSS_BLOCK = int(os.getenv("WS_AI_CONSEC_LOSS", "2"))  # Bloqueia após 2 losses seguidas no padrão
AI_ADAPTIVE_PRIOR = (os.getenv("WS_AI_ADAPTIVE_PRIOR", "1").strip() == "1")  # Prior adaptativo ligado
AI_GLOBAL_WINRATE_INFLUENCE = float(os.getenv("WS_AI_GLOBAL_WR", "0.35"))  # AUMENTADO: 35% influencia da performance global

# ===================== PERNADA B (RIGOROSO PARA 80%+ WIN RATE) =====================
IMPULSO_MIN_ATR = float(os.getenv("WS_IMPULSO_MIN_ATR", "1.20"))  # RIGOROSO: exige impulso significativo de 1.20 ATR
IMPULSO_JANELA_MIN = int(os.getenv("WS_IMP_JMIN", "3"))  # mínimo 3 velas
IMPULSO_JANELA_MAX = int(os.getenv("WS_IMP_JMAX", "15"))  # máximo 15 velas

PULLBACK_MIN = int(os.getenv("WS_PB_MIN", "1"))
PULLBACK_MAX = int(os.getenv("WS_PB_MAX", "6"))  # aumentado para 6

RETR_MIN = float(os.getenv("WS_RETR_MIN", "0.30"))  # RIGOROSO: minimo 30% (zona Fibonacci)
RETR_MAX = float(os.getenv("WS_RETR_MAX", "0.62"))  # RIGOROSO: maximo 62% (Golden Ratio)

BREAK_MARGIN_ATR = float(os.getenv("WS_BREAK_MARGIN_ATR", "0.01"))  # margem mínima
MAX_BREAK_DISTANCE_ATR = float(os.getenv("WS_MAX_BREAK_DIST_ATR", "0.40"))  # distância maior

# ===================== ANTI-LATERAL (RIGOROSO) =====================
MIN_EFF_A = float(os.getenv("WS_MIN_EFF_A", "0.65"))  # RIGOROSO: exige 65% eficiencia (impulsos limpos)

CHOP_LOOKBACK = int(os.getenv("WS_CHOP_LB", "28"))
MAX_COLOR_FLIPS_FRAC = float(os.getenv("WS_MAX_FLIPS", "0.80"))  # permite mais choppiness
MIN_NET_GROSS_EFF = float(os.getenv("WS_MIN_NETGROSS", "0.10"))  # muito relaxado

COMP_LOOKBACK = int(os.getenv("WS_COMP_LB", "18"))
MIN_RANGE_ATR = float(os.getenv("WS_MIN_RANGE_ATR", "0.50"))  # MUITO RELAXADO: 0.50 ATR

LATE_LOOKBACK = int(os.getenv("WS_LATE_LB", "18"))
MAX_LATE_EXT_ATR = float(os.getenv("WS_MAX_LATE_EXT_ATR", "12.0"))  # permite extensão maior

# ===================== QUALIDADE DO GATILHO (RIGOROSO) =====================
MIN_BODY_FRAC_BREAK = float(os.getenv("WS_MIN_BODY_FRAC", "0.40"))  # RIGOROSO: exige 40% de corpo (vela forte)
MAX_WICK_AGAINST = float(os.getenv("WS_MAX_WICK_AGAINST", "0.35"))  # RIGOROSO: maximo 35% de pavio contra

# ===================== SCORE (RIGOROSO PARA 80%+ WIN RATE) =====================
GATE_MIN_SCORE = float(os.getenv("WS_GATE_MIN", "0.68"))  # RIGOROSO: score minimo de 68%
GATE_SOFT_SCORE = float(os.getenv("WS_GATE_SOFT", "0.60"))  # RIGOROSO: score soft de 60%

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
SR_BLOCK_DIST_ATR = float(os.getenv("WS_SR_BLOCK_ATR", "1.20"))  # RIGOROSO: exige 1.20 ATR de distancia de S/R

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
    """Garante conexão com a IQ Option - NUNCA desiste de reconectar"""
    if iq is None:
        return conectar_iq()
    try:
        if iq.check_connect():
            return iq
    except Exception:
        pass

    # Loop infinito de reconexão - só para se atingir meta/stop
    max_attempts = 999  # Praticamente infinito
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        log.warning(paint(f"Conexão caiu. Tentativa {attempt} de reconexão...", C.Y))
        
        try:
            # Tenta reconectar o objeto existente
            iq.connect()
            for _ in range(15):
                if iq.check_connect():
                    iq.change_balance(CONTA)
                    log.info(paint(f"✅ Reconectado com sucesso após {attempt} tentativa(s)!", C.G))
                    return iq
                time.sleep(1.5)
        except Exception as e:
            log.warning(f"Erro na tentativa {attempt}: {e}")
        
        # Se não conseguiu, tenta criar nova conexão
        try:
            log.info(paint("Criando nova conexão...", C.Y))
            new_iq = conectar_iq()
            if new_iq and new_iq.check_connect():
                log.info(paint("✅ Nova conexão estabelecida!", C.G))
                return new_iq
        except Exception as e:
            log.warning(f"Erro ao criar nova conexão: {e}")
        
        # Aguarda antes de tentar novamente (aumenta gradualmente)
        wait_time = min(5 + (attempt * 2), 30)  # Máximo 30 segundos
        log.info(paint(f"Aguardando {wait_time}s antes de tentar novamente...", C.Y))
        time.sleep(wait_time)
    
    # Se chegou aqui, tenta última vez
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


# ===================== COLETA M5 PARA TENDENCIA =====================
def get_candles_m5(iq: IQ_Option, ativo: str, n: int = 48) -> Optional[pd.DataFrame]:
    """
    Coleta candles de 5 minutos para analise de tendencia macro.

    Args:
        iq: Conexao IQ Option
        ativo: Nome do ativo
        n: Numero de candles (48 = 4 horas)

    Returns:
        DataFrame com candles M5 ou None se falhar
    """
    try:
        end_ts = time.time()
        candles = safe_call(iq, iq.get_candles, ativo, TF_M5, n, end_ts)
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
        return df if len(df) >= 20 else None
    except Exception:
        return None


# ===================== NOVO PIPELINE: CNN + REGIME FILTER + RISK CONTROL =====================
def new_pipeline_decide(
    iq: IQ_Option,
    ativo: str,
    df_m1: pd.DataFrame,
    atr_val: float,
    payout: int = 80
) -> Tuple[bool, str, Dict]:
    """
    Novo pipeline de decisao com:
    1. Coleta M5 para tendencia
    2. Regime Filter (pre-modelo)
    3. CNN com 3 classes
    4. Risk Control (pos-modelo)

    Returns:
        Tuple[bool, str, Dict]: (pode_operar, direcao, detalhes)
        direcao: "CALL", "PUT", ou "NO_TRADE"
    """
    details = {
        "pipeline": "CNN_V2",
        "stages": {}
    }

    # ===== STAGE 1: COLETA M5 =====
    df_m5 = None
    m5_direction = "NEUTRAL"
    m5_strength = 0.0

    if REGIME_FILTER_AVAILABLE and regime_filter:
        df_m5 = get_candles_m5(iq, ativo, N_M5)
        if df_m5 is not None:
            m5_direction, m5_strength = regime_filter.get_m5_direction(df_m5)
        details["stages"]["m5"] = {
            "direction": m5_direction,
            "strength": m5_strength,
            "candles": len(df_m5) if df_m5 is not None else 0
        }
        log.info(paint(f"[M5] {ativo} | Tendencia: {m5_direction} | Forca: {m5_strength:.2f}", C.B))

    # ===== STAGE 2: REGIME FILTER =====
    if REGIME_FILTER_AVAILABLE and regime_filter:
        blocked, reason, rf_details = regime_filter.should_block(
            df_m1, df_m5, atr_val, payout
        )
        details["stages"]["regime_filter"] = {
            "blocked": blocked,
            "reason": reason,
            "details": rf_details
        }

        if blocked:
            log.info(paint(f"[REGIME] {ativo} | BLOQUEADO: {reason}", C.Y))
            return False, "NO_TRADE", details

    # ===== STAGE 3: MODELO CNN =====
    cnn_class = "NO_TRADE"
    cnn_prob = 0.33
    cnn_confidence = 0.0

    if CNN_AVAILABLE and trading_cnn:
        prediction = trading_cnn.predict(df_m1)
        cnn_class = prediction["class"]
        cnn_prob = prediction["probability"]
        cnn_confidence = prediction["confidence"]

        details["stages"]["cnn"] = {
            "class": cnn_class,
            "probability": cnn_prob,
            "confidence": cnn_confidence,
            "raw_probs": prediction["raw_probs"]
        }

        log.info(paint(
            f"[CNN] {ativo} | Classe: {cnn_class} | Prob: {cnn_prob:.2f} | Conf: {cnn_confidence:.2f}",
            C.G if cnn_class != "NO_TRADE" else C.Y
        ))

        # Se CNN diz NO_TRADE, para aqui
        if cnn_class == "NO_TRADE":
            return False, "NO_TRADE", details
    else:
        # Fallback: usa direcao do M5
        if m5_direction == "BULLISH":
            cnn_class = "CALL"
            cnn_prob = 0.5 + m5_strength * 0.2
        elif m5_direction == "BEARISH":
            cnn_class = "PUT"
            cnn_prob = 0.5 + m5_strength * 0.2
        else:
            return False, "NO_TRADE", details

    # ===== STAGE 4: VALIDACAO M5 vs CNN =====
    # Verifica se CNN concorda com M5
    if m5_direction != "NEUTRAL":
        if cnn_class == "CALL" and m5_direction != "BULLISH":
            log.info(paint(f"[CONFLITO] CNN={cnn_class} vs M5={m5_direction} | Bloqueando", C.Y))
            details["stages"]["conflict"] = f"CNN={cnn_class} vs M5={m5_direction}"
            return False, "NO_TRADE", details
        if cnn_class == "PUT" and m5_direction != "BEARISH":
            log.info(paint(f"[CONFLITO] CNN={cnn_class} vs M5={m5_direction} | Bloqueando", C.Y))
            details["stages"]["conflict"] = f"CNN={cnn_class} vs M5={m5_direction}"
            return False, "NO_TRADE", details

    # ===== STAGE 5: RISK CONTROL =====
    if RISK_CONTROL_AVAILABLE and risk_control:
        can_trade, rc_reason, rc_details = risk_control.should_trade({
            "class": cnn_class,
            "probability": cnn_prob,
            "raw_probs": details.get("stages", {}).get("cnn", {}).get("raw_probs", [0.33, 0.33, 0.34]),
            "confidence": cnn_confidence
        })

        details["stages"]["risk_control"] = {
            "can_trade": can_trade,
            "reason": rc_reason,
            "threshold": rc_details.get("current_threshold", 0.60),
            "consecutive_losses": rc_details.get("consecutive_losses", 0)
        }

        if not can_trade:
            log.info(paint(f"[RISK] {ativo} | BLOQUEADO: {rc_reason}", C.Y))
            return False, "NO_TRADE", details

    # ===== APROVADO =====
    log.info(paint(
        f"[PIPELINE OK] {ativo} {cnn_class} | M5={m5_direction} | CNN={cnn_prob:.2f} | Conf={cnn_confidence:.2f}",
        C.G
    ))

    return True, cnn_class, details


def on_trade_result_cnn(ativo: str, direction: str, win: bool, df_m1: pd.DataFrame):
    """
    Atualiza o sistema CNN apos resultado do trade.

    Args:
        ativo: Nome do ativo
        direction: "CALL" ou "PUT"
        win: True se ganhou
        df_m1: DataFrame com candles no momento da entrada
    """
    # Atualiza modelo CNN
    if CNN_AVAILABLE and trading_cnn:
        trading_cnn.add_training_sample(df_m1, direction, win)

    # Atualiza Risk Control
    if RISK_CONTROL_AVAILABLE and risk_control:
        risk_control.on_result(win)

    # Log
    result_str = "WIN" if win else "LOSS"
    color = C.G if win else C.R
    log.info(paint(f"[CNN LEARN] {ativo} {direction} | {result_str}", color))


def format_last_candles(df: pd.DataFrame, n: int = 10) -> str:
    """Formata as últimas N velas para log."""
    try:
        tail = df.tail(n)
        parts = []
        for idx, row in tail.iterrows():
            parts.append(
                f"{idx:%H:%M:%S} O:{row['open']:.6f} H:{row['high']:.6f} L:{row['low']:.6f} C:{row['close']:.6f}"
            )
        return " | ".join(parts)
    except Exception:
        return "(falha ao formatar velas)"

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

def analyze_recent_momentum(df: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    """
    Analisa força das últimas velas para indicar continuação ou reversão.
    Retorna direção, força e sinais de exaustão.
    """
    result = {
        "direction": "NEUTRAL",
        "strength": 0.0,
        "continuation_ok": False,
        "reversal_ok_call": False,
        "reversal_ok_put": False,
        "last3_cont_call": False,
        "last3_cont_put": False,
        "last3_rev_call": False,
        "last3_rev_put": False,
        "reason": ""
    }

    if df is None or len(df) < 20:
        return result

    atr_safe = max(atr_val, 1e-9)
    closes = df["close"].tail(15).to_numpy(float)
    opens = df["open"].tail(15).to_numpy(float)

    # Direção e força geral (últimas 15 velas)
    move_15 = (closes[-1] - closes[0]) / atr_safe
    result["direction"] = "BULLISH" if move_15 > 0 else ("BEARISH" if move_15 < 0 else "NEUTRAL")
    result["strength"] = min(1.0, abs(move_15) / 2.0)

    # Continuação: últimas 5 e 10 velas na mesma direção
    move_10 = (closes[-1] - closes[-10]) / atr_safe
    move_5 = (closes[-1] - closes[-5]) / atr_safe
    same_dir = (move_10 > 0 and move_5 > 0) or (move_10 < 0 and move_5 < 0)
    if same_dir and abs(move_5) >= 0.35 and abs(move_10) >= 0.6:
        result["continuation_ok"] = True
        result["reason"] = "continuation"

    # Exaustão: últimas 5 velas perdendo força vs. 10 anteriores
    prev10 = (closes[-5] - closes[-15]) / atr_safe
    last5 = move_5
    body_sizes = np.abs(closes - opens)
    small_bodies = np.mean(body_sizes[-3:]) < (np.mean(body_sizes[-10:]) * 0.6)
    weakening = abs(last5) <= (abs(prev10) * 0.35)

    if weakening and small_bodies:
        # Se estava subindo e perdeu força -> favorece PUT
        if prev10 > 0:
            result["reversal_ok_put"] = True
            result["reason"] = "exhaustion_up"
        # Se estava caindo e perdeu força -> favorece CALL
        elif prev10 < 0:
            result["reversal_ok_call"] = True
            result["reason"] = "exhaustion_down"

    # ===== ESTRUTURA DAS ÚLTIMAS 3 VELAS =====
    last3_close = closes[-3:]
    last3_open = opens[-3:]
    last3_body = np.abs(last3_close - last3_open)
    last3_dir = [1 if last3_close[i] > last3_open[i] else (-1 if last3_close[i] < last3_open[i] else 0) for i in range(3)]
    body_avg_10 = np.mean(body_sizes[-10:]) if len(body_sizes) >= 10 else np.mean(body_sizes)
    small_last3 = np.mean(last3_body) < (body_avg_10 * 0.7)

    # Continuação: 3 velas na mesma direção com corpo decente
    if all(d > 0 for d in last3_dir) and np.mean(last3_body) >= body_avg_10 * 0.6:
        result["last3_cont_call"] = True
    if all(d < 0 for d in last3_dir) and np.mean(last3_body) >= body_avg_10 * 0.6:
        result["last3_cont_put"] = True

    # Reversão: últimas 3 velas com indecisão/mistura + corpos menores
    mixed = (last3_dir.count(1) >= 1 and last3_dir.count(-1) >= 1)
    if mixed and small_last3:
        if prev10 > 0:
            result["last3_rev_put"] = True
        elif prev10 < 0:
            result["last3_rev_call"] = True

    return result

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

# ===================== DETECÇÃO DE MERCADO LATERAL / ZIG-ZAG =====================
def detect_zigzag_lateral_market(df_m1: pd.DataFrame, atr_val: float, lookback: int = 15) -> Dict[str, Any]:
    """
    Detecta padrão de zig-zag/mercado lateral onde o preço oscila sem tendência clara.
    
    Método:
    1. Identifica swing highs e swing lows nos últimos N candles
    2. Conta quantas vezes a direção muda (reversões)
    3. Calcula o tamanho médio dos swings em ATR
    4. Se muitas reversões pequenas = mercado lateral/choppy
    
    Retorna:
        Dict com is_lateral, reversal_count, avg_swing_atr, reason
    """
    if len(df_m1) < lookback + 2:
        return {"is_lateral": False, "reversal_count": 0, "avg_swing_atr": 0, "reason": "dados_insuficientes"}
    
    atr_safe = max(atr_val, 1e-9)
    df_recent = df_m1.tail(lookback).reset_index(drop=True)
    
    highs = df_recent["high"].to_numpy(float)
    lows = df_recent["low"].to_numpy(float)
    closes = df_recent["close"].to_numpy(float)
    
    # Identificar pontos de swing (pivot highs e pivot lows)
    # Um swing high é quando high[i] > high[i-1] e high[i] > high[i+1]
    # Um swing low é quando low[i] < low[i-1] e low[i] < low[i+1]
    
    swing_points = []  # Lista de (index, tipo, preço) onde tipo = 'H' ou 'L'
    
    for i in range(1, len(df_recent) - 1):
        # Swing High
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            swing_points.append((i, 'H', highs[i]))
        # Swing Low
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            swing_points.append((i, 'L', lows[i]))
    
    # Ordenar por índice
    swing_points.sort(key=lambda x: x[0])
    
    if len(swing_points) < 3:
        return {"is_lateral": False, "reversal_count": len(swing_points), "avg_swing_atr": 0, "reason": "poucos_swings"}
    
    # Contar reversões de direção
    # Uma reversão acontece quando temos H -> L ou L -> H em sequência
    reversal_count = 0
    swing_sizes = []
    
    for i in range(1, len(swing_points)):
        prev_type = swing_points[i-1][1]
        curr_type = swing_points[i][1]
        prev_price = swing_points[i-1][2]
        curr_price = swing_points[i][2]
        
        # Se mudou de H para L ou de L para H = reversão
        if prev_type != curr_type:
            reversal_count += 1
            swing_size = abs(curr_price - prev_price) / atr_safe
            swing_sizes.append(swing_size)
    
    avg_swing_atr = np.mean(swing_sizes) if swing_sizes else 0
    
    # Calcular range total do período
    range_total = (np.max(highs) - np.min(lows)) / atr_safe
    
    # Calcular "choppiness" - se o preço está oscilando muito dentro de um range pequeno
    # Muitas reversões (>= 4) + swings pequenos (< 0.8 ATR médio) + range apertado (< 2 ATR) = lateral
    
    is_lateral = False
    reason = ""
    
    # Condição 1: Muitas reversões com swings pequenos
    if reversal_count >= 4 and avg_swing_atr < 0.8:
        is_lateral = True
        reason = f"zigzag_muitas_reversoes(rev={reversal_count},swing_medio={avg_swing_atr:.2f}ATR)"
    
    # Condição 2: Range apertado com várias reversões (mercado lateral clássico)
    elif reversal_count >= 3 and range_total < 2.0:
        is_lateral = True
        reason = f"mercado_lateral(rev={reversal_count},range={range_total:.2f}ATR)"
    
    # Condição 3: Swings muito pequenos indicando indecisão
    elif reversal_count >= 3 and avg_swing_atr < 0.5:
        is_lateral = True
        reason = f"indecisao(rev={reversal_count},swing_medio={avg_swing_atr:.2f}ATR)"
    
    # Análise adicional: verificar se há tendência nos últimos candles
    if not is_lateral:
        # Calcular direção predominante
        price_change = (closes[-1] - closes[0]) / atr_safe
        
        # Se muitas reversões mas pouca progressão líquida = choppy
        if reversal_count >= 3 and abs(price_change) < 0.5:
            is_lateral = True
            reason = f"sem_progressao(rev={reversal_count},move_liquido={price_change:.2f}ATR)"
    
    return {
        "is_lateral": is_lateral,
        "reversal_count": reversal_count,
        "avg_swing_atr": avg_swing_atr,
        "range_total_atr": range_total if 'range_total' in dir() else 0,
        "reason": reason
    }


# ===================== SISTEMA DE APRENDIZADO DE TOPOS E FUNDOS =====================
# A IA APRENDE A ESTRUTURA DO MERCADO PARA IDENTIFICAR PONTOS IDEAIS DE ENTRADA
# 
# CONCEITOS FUNDAMENTAIS:
# - HH (Higher High) = Topo mais alto que o anterior → Tendência de ALTA
# - HL (Higher Low) = Fundo mais alto que o anterior → Tendência de ALTA
# - LH (Lower High) = Topo mais baixo que o anterior → Tendência de BAIXA
# - LL (Lower Low) = Fundo mais baixo que o anterior → Tendência de BAIXA
#
# PONTOS IDEAIS DE ENTRADA:
# - Em ALTA: Entrar no reteste do HL (fundo mais alto) → CALL
# - Em BAIXA: Entrar no reteste do LH (topo mais baixo) → PUT
#
# BREAK OF STRUCTURE (BOS):
# - Rompimento de um topo/fundo anterior indica mudança de tendência
# ============================================================================

def identify_swing_points(df_m1: pd.DataFrame, lookback: int = 30, strength: int = 2) -> Dict[str, Any]:
    """
    Identifica topos (swing highs) e fundos (swing lows) significativos.
    
    Args:
        df_m1: DataFrame com dados OHLC
        lookback: Número de candles para analisar
        strength: Quantos candles de cada lado devem ser menores/maiores (força do pivot)
    
    Retorna:
        Dict com swing_highs, swing_lows, estrutura classificada
    """
    if len(df_m1) < lookback:
        return {"swing_highs": [], "swing_lows": [], "valid": False}
    
    df = df_m1.tail(lookback).reset_index(drop=True)
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)
    
    swing_highs = []  # Lista de (índice, preço)
    swing_lows = []   # Lista de (índice, preço)
    
    # Identificar swing highs e lows com força configurável
    for i in range(strength, len(df) - strength):
        # Verifica se é um swing high (pico local)
        is_swing_high = True
        for j in range(1, strength + 1):
            if highs[i] <= highs[i-j] or highs[i] <= highs[i+j]:
                is_swing_high = False
                break
        
        if is_swing_high:
            swing_highs.append((i, highs[i]))
        
        # Verifica se é um swing low (vale local)
        is_swing_low = True
        for j in range(1, strength + 1):
            if lows[i] >= lows[i-j] or lows[i] >= lows[i+j]:
                is_swing_low = False
                break
        
        if is_swing_low:
            swing_lows.append((i, lows[i]))
    
    return {
        "swing_highs": swing_highs,
        "swing_lows": swing_lows,
        "closes": closes,
        "valid": len(swing_highs) >= 2 and len(swing_lows) >= 2
    }


def analyze_market_structure(df_m1: pd.DataFrame, atr_val: float, lookback: int = 30) -> Dict[str, Any]:
    """
    SISTEMA INTELIGENTE DE ANÁLISE DE ESTRUTURA DE MERCADO
    
    Analisa topos e fundos para determinar:
    1. Tipo de tendência (ALTA, BAIXA, LATERAL)
    2. Qualidade da estrutura (força)
    3. Ponto atual do preço na estrutura
    4. Ponto ideal de entrada
    
    REGRAS DE ESTRUTURA:
    - ALTA: HH + HL (Higher Highs + Higher Lows)
    - BAIXA: LH + LL (Lower Highs + Lower Lows)
    - LATERAL: Mix sem padrão claro
    
    Retorna:
        Dict completo com análise da estrutura
    """
    result = {
        "trend": "NEUTRAL",
        "trend_strength": 0.0,
        "structure_type": "undefined",
        "swing_highs": [],
        "swing_lows": [],
        "last_hh": None,
        "last_hl": None,
        "last_lh": None,
        "last_ll": None,
        "ideal_entry_zone": None,
        "ideal_direction": None,
        "entry_quality": 0.0,
        "price_position": "middle",
        "bos_detected": False,
        "bos_direction": None,
        "reasons": [],
        "valid": False
    }
    
    if len(df_m1) < lookback:
        return result
    
    atr_safe = max(atr_val, 1e-9)
    
    # Identificar swing points
    swings = identify_swing_points(df_m1, lookback, strength=2)
    if not swings["valid"]:
        return result
    
    swing_highs = swings["swing_highs"]
    swing_lows = swings["swing_lows"]
    closes = swings["closes"]
    current_price = closes[-1]
    
    result["swing_highs"] = swing_highs
    result["swing_lows"] = swing_lows
    result["valid"] = True
    
    # ===== ANÁLISE DE TOPOS (Higher Highs vs Lower Highs) =====
    hh_count = 0  # Higher Highs
    lh_count = 0  # Lower Highs
    
    for i in range(1, len(swing_highs)):
        prev_high = swing_highs[i-1][1]
        curr_high = swing_highs[i][1]
        
        if curr_high > prev_high:
            hh_count += 1
            result["last_hh"] = curr_high
        else:
            lh_count += 1
            result["last_lh"] = curr_high
    
    # ===== ANÁLISE DE FUNDOS (Higher Lows vs Lower Lows) =====
    hl_count = 0  # Higher Lows
    ll_count = 0  # Lower Lows
    
    for i in range(1, len(swing_lows)):
        prev_low = swing_lows[i-1][1]
        curr_low = swing_lows[i][1]
        
        if curr_low > prev_low:
            hl_count += 1
            result["last_hl"] = curr_low
        else:
            ll_count += 1
            result["last_ll"] = curr_low
    
    # ===== CLASSIFICAR ESTRUTURA =====
    total_highs = hh_count + lh_count
    total_lows = hl_count + ll_count
    
    # Tendência de ALTA: Maioria HH + HL
    if total_highs > 0 and total_lows > 0:
        hh_ratio = hh_count / total_highs if total_highs > 0 else 0
        hl_ratio = hl_count / total_lows if total_lows > 0 else 0
        lh_ratio = lh_count / total_highs if total_highs > 0 else 0
        ll_ratio = ll_count / total_lows if total_lows > 0 else 0
        
        # TENDÊNCIA DE ALTA: HH > 50% E HL > 50%
        if hh_ratio >= 0.5 and hl_ratio >= 0.5:
            result["trend"] = "BULLISH"
            result["trend_strength"] = (hh_ratio + hl_ratio) / 2
            result["structure_type"] = "uptrend_HH_HL"
            result["reasons"].append(f"Estrutura_ALTA(HH={hh_count},HL={hl_count})")
            
            # Ponto ideal de entrada em ALTA: Reteste do último HL (fundo mais alto)
            if result["last_hl"] is not None:
                hl_dist = abs(current_price - result["last_hl"]) / atr_safe
                if hl_dist <= 0.8:  # Preço próximo do último HL
                    result["ideal_entry_zone"] = result["last_hl"]
                    result["ideal_direction"] = "CALL"
                    result["entry_quality"] = max(0, 1 - (hl_dist / 0.8))
                    result["price_position"] = "near_hl"
                    result["reasons"].append(f"Preço_proximo_HL(dist={hl_dist:.2f}ATR)→CALL_IDEAL")
        
        # TENDÊNCIA DE BAIXA: LH > 50% E LL > 50%
        elif lh_ratio >= 0.5 and ll_ratio >= 0.5:
            result["trend"] = "BEARISH"
            result["trend_strength"] = (lh_ratio + ll_ratio) / 2
            result["structure_type"] = "downtrend_LH_LL"
            result["reasons"].append(f"Estrutura_BAIXA(LH={lh_count},LL={ll_count})")
            
            # Ponto ideal de entrada em BAIXA: Reteste do último LH (topo mais baixo)
            if result["last_lh"] is not None:
                lh_dist = abs(current_price - result["last_lh"]) / atr_safe
                if lh_dist <= 0.8:  # Preço próximo do último LH
                    result["ideal_entry_zone"] = result["last_lh"]
                    result["ideal_direction"] = "PUT"
                    result["entry_quality"] = max(0, 1 - (lh_dist / 0.8))
                    result["price_position"] = "near_lh"
                    result["reasons"].append(f"Preço_proximo_LH(dist={lh_dist:.2f}ATR)→PUT_IDEAL")
        
        # ESTRUTURA MISTA/LATERAL
        else:
            result["trend"] = "NEUTRAL"
            result["trend_strength"] = 0.3
            result["structure_type"] = "mixed_no_clear_trend"
            result["reasons"].append(f"Estrutura_MISTA(HH={hh_count},LH={lh_count},HL={hl_count},LL={ll_count})")
    
    # ===== DETECTAR BREAK OF STRUCTURE (BOS) =====
    # BOS acontece quando o preço rompe um topo/fundo significativo
    
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        last_significant_high = swing_highs[-2][1]  # Penúltimo topo
        last_significant_low = swing_lows[-2][1]    # Penúltimo fundo
        
        # BOS de ALTA: Preço rompeu o último topo significativo
        if current_price > last_significant_high:
            high_break = (current_price - last_significant_high) / atr_safe
            if high_break >= 0.3:  # Rompimento significativo
                result["bos_detected"] = True
                result["bos_direction"] = "BULLISH"
                result["reasons"].append(f"BOS_ALTA(rompeu_topo={last_significant_high:.6f},break={high_break:.2f}ATR)")
        
        # BOS de BAIXA: Preço rompeu o último fundo significativo
        elif current_price < last_significant_low:
            low_break = (last_significant_low - current_price) / atr_safe
            if low_break >= 0.3:  # Rompimento significativo
                result["bos_detected"] = True
                result["bos_direction"] = "BEARISH"
                result["reasons"].append(f"BOS_BAIXA(rompeu_fundo={last_significant_low:.6f},break={low_break:.2f}ATR)")
    
    # ===== POSIÇÃO DO PREÇO NA ESTRUTURA =====
    if swing_highs and swing_lows:
        highest_swing = max([h[1] for h in swing_highs])
        lowest_swing = min([l[1] for l in swing_lows])
        swing_range = highest_swing - lowest_swing
        
        if swing_range > 0:
            position_pct = (current_price - lowest_swing) / swing_range
            result["position_pct"] = position_pct  # Salvar para uso externo
            
            if position_pct >= 0.8:
                result["price_position"] = "top_of_range"
                result["reasons"].append(f"Preço_no_TOPO({position_pct:.0%})")
            elif position_pct <= 0.2:
                result["price_position"] = "bottom_of_range"
                result["reasons"].append(f"Preço_no_FUNDO({position_pct:.0%})")
            elif 0.4 <= position_pct <= 0.6:
                result["price_position"] = "middle_of_range"
                result["reasons"].append(f"Preço_no_MEIO({position_pct:.0%})")
    
    # ===== ANÁLISE DO MOVIMENTO RECENTE (últimos 10 candles) =====
    # Detecta se o preço já se moveu muito em uma direção (extensão)
    df_recent = df_m1.tail(10)
    if len(df_recent) >= 10:
        recent_high = float(df_recent["high"].max())
        recent_low = float(df_recent["low"].min())
        recent_open = float(df_recent["open"].iloc[0])
        recent_close = float(df_recent["close"].iloc[-1])
        
        # Movimento total em ATRs
        move_from_high = (recent_high - recent_close) / atr_safe
        move_from_low = (recent_close - recent_low) / atr_safe
        total_move = (recent_close - recent_open) / atr_safe
        
        result["recent_move_atr"] = total_move
        result["move_from_high"] = move_from_high
        result["move_from_low"] = move_from_low
        
        # EXTENSÃO DE BAIXA: Preço caiu muito e está perto da mínima
        if total_move < -1.5 and move_from_low < 0.5:
            result["extended_move"] = "BEARISH_EXTENDED"
            result["extended_severity"] = abs(total_move)
            result["reasons"].append(f"BAIXA_ESTENDIDA({total_move:.1f}ATR_perto_minima)")
        
        # EXTENSÃO DE ALTA: Preço subiu muito e está perto da máxima
        elif total_move > 1.5 and move_from_high < 0.5:
            result["extended_move"] = "BULLISH_EXTENDED"
            result["extended_severity"] = total_move
            result["reasons"].append(f"ALTA_ESTENDIDA({total_move:.1f}ATR_perto_maxima)")
    
    return result


def detect_wave_cycle(df_m1: pd.DataFrame, atr_val: float, lookback: int = 30) -> Dict[str, Any]:
    """
    DETECÇÃO DE CICLO DE ONDAS DO MERCADO
    
    O preço se move em ondas:
    1. IMPULSO (movimento forte na direção)
    2. CONSOLIDAÇÃO (lateralização/correção)
    3. IMPULSO (continuação)
    4. CONSOLIDAÇÃO
    ... e assim por diante
    
    REGRAS:
    - Entrar no início do IMPULSO (após consolidação)
    - NÃO entrar no MEIO do impulso (já perdeu o timing)
    - NÃO entrar na CONSOLIDAÇÃO (mercado indefinido)
    
    Retorna:
        Dict com fase atual, direção provável, qualidade de entrada
    """
    result = {
        "wave_phase": "UNKNOWN",  # IMPULSO_UP, IMPULSO_DOWN, CONSOLIDACAO, INICIO_IMPULSO
        "wave_direction": "NEUTRAL",  # BULLISH, BEARISH
        "impulse_strength": 0.0,
        "consolidation_quality": 0.0,
        "entry_timing": "RUIM",  # BOM, MEDIO, RUIM
        "should_wait": False,
        "reason": "",
        "valid": False
    }
    
    if len(df_m1) < lookback:
        return result
    
    atr_safe = max(atr_val, 1e-9)
    df = df_m1.tail(lookback).copy()
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    
    # Dividir em 3 segmentos para detectar padrão de onda
    seg_size = lookback // 3
    
    # Segmento 1 (mais antigo)
    seg1_close = closes[:seg_size]
    seg1_move = (seg1_close[-1] - seg1_close[0]) / atr_safe
    seg1_range = (max(highs[:seg_size]) - min(lows[:seg_size])) / atr_safe
    
    # Segmento 2 (meio)
    seg2_close = closes[seg_size:2*seg_size]
    seg2_move = (seg2_close[-1] - seg2_close[0]) / atr_safe
    seg2_range = (max(highs[seg_size:2*seg_size]) - min(lows[seg_size:2*seg_size])) / atr_safe
    
    # Segmento 3 (mais recente)
    seg3_close = closes[2*seg_size:]
    seg3_move = (seg3_close[-1] - seg3_close[0]) / atr_safe
    seg3_range = (max(highs[2*seg_size:]) - min(lows[2*seg_size:])) / atr_safe
    
    # Detectar padrão de onda
    # CONSOLIDAÇÃO = range pequeno e movimento pequeno
    is_consolidation = lambda m, r: abs(m) < 0.5 and r < 1.0
    # IMPULSO = movimento significativo
    is_impulse = lambda m: abs(m) > 0.8
    
    seg1_is_impulse = is_impulse(seg1_move)
    seg2_is_consolidation = is_consolidation(seg2_move, seg2_range)
    seg3_is_impulse = is_impulse(seg3_move)
    seg3_is_consolidation = is_consolidation(seg3_move, seg3_range)
    
    result["valid"] = True
    
    # PADRÃO 1: Impulso -> Consolidação -> ??? (momento de decisão)
    if seg1_is_impulse and seg2_is_consolidation:
        if seg3_is_consolidation:
            # Ainda em consolidação - AGUARDAR rompimento
            result["wave_phase"] = "CONSOLIDACAO"
            result["wave_direction"] = "BULLISH" if seg1_move > 0 else "BEARISH"
            result["entry_timing"] = "RUIM"
            result["should_wait"] = True
            result["reason"] = "aguardar_rompimento_consolidacao"
        elif seg3_is_impulse:
            if (seg1_move > 0 and seg3_move > 0) or (seg1_move < 0 and seg3_move < 0):
                # CONTINUAÇÃO - Impulso na mesma direção
                result["wave_phase"] = "IMPULSO_CONTINUACAO"
                result["wave_direction"] = "BULLISH" if seg3_move > 0 else "BEARISH"
                result["impulse_strength"] = abs(seg3_move)
                # Se já está no MEIO do impulso, timing ruim
                result["entry_timing"] = "RUIM" if abs(seg3_move) > 1.5 else "MEDIO"
                result["reason"] = "impulso_em_andamento" if abs(seg3_move) > 1.5 else "inicio_impulso"
            else:
                # REVERSÃO - Impulso na direção oposta
                result["wave_phase"] = "INICIO_IMPULSO"
                result["wave_direction"] = "BULLISH" if seg3_move > 0 else "BEARISH"
                result["impulse_strength"] = abs(seg3_move)
                result["entry_timing"] = "BOM" if abs(seg3_move) < 1.0 else "MEDIO"
                result["reason"] = "inicio_nova_onda"
    
    # PADRÃO 2: Consolidação recente - mercado indefinido
    elif seg3_is_consolidation:
        result["wave_phase"] = "CONSOLIDACAO"
        result["consolidation_quality"] = 1.0 - seg3_range  # Quanto menor o range, mais "puro" a consolidação
        result["entry_timing"] = "RUIM"
        result["should_wait"] = True
        result["reason"] = "mercado_em_consolidacao"
    
    # PADRÃO 3: Impulso forte em andamento
    elif seg3_is_impulse:
        result["wave_phase"] = "IMPULSO_FORTE" if abs(seg3_move) > 1.5 else "IMPULSO"
        result["wave_direction"] = "BULLISH" if seg3_move > 0 else "BEARISH"
        result["impulse_strength"] = abs(seg3_move)
        # Timing depende de quanto já se moveu
        if abs(seg3_move) > 2.0:
            result["entry_timing"] = "RUIM"
            result["reason"] = "impulso_muito_estendido"
        elif abs(seg3_move) > 1.5:
            result["entry_timing"] = "MEDIO"
            result["reason"] = "impulso_em_andamento"
        else:
            result["entry_timing"] = "BOM"
            result["reason"] = "inicio_impulso"
    
    return result


def get_ideal_entry_point(structure: Dict[str, Any], direction: str, current_price: float, atr_val: float) -> Dict[str, Any]:
    """
    Com base na estrutura de mercado, determina se o ponto atual é ideal para entrada.
    
    REGRAS DE ENTRADA IDEAL:
    - CALL em ALTA: Entrar próximo do HL (Higher Low) - PULLBACK para suporte
    - PUT em BAIXA: Entrar próximo do LH (Lower High) - THROWBACK para resistência
    - NUNCA entrar no meio do range sem confirmação
    - NUNCA comprar no topo do range em alta / vender no fundo em baixa
    
    Retorna:
        Dict com is_ideal, quality, reason
    """
    result = {
        "is_ideal_entry": False,
        "entry_quality": 0.0,
        "reason": "",
        "recommendation": "AGUARDAR",
        "structure_aligned": False,
        "extended_block": False
    }
    
    if not structure.get("valid", False):
        result["reason"] = "estrutura_invalida"
        return result
    
    atr_safe = max(atr_val, 1e-9)
    trend = structure.get("trend", "NEUTRAL")
    price_position = structure.get("price_position", "middle")
    ideal_direction = structure.get("ideal_direction")
    
    # ===== VERIFICAR EXTENSÃO DO MOVIMENTO (BLOQUEIO CRÍTICO) =====
    # Se o preço já se moveu muito na direção do trade, NÃO ENTRAR
    extended_move = structure.get("extended_move", None)
    extended_severity = structure.get("extended_severity", 0)
    position_pct = structure.get("position_pct", 0.5)
    
    # PUT após baixa estendida = BLOQUEAR (vender no fundo)
    if direction == "PUT" and extended_move == "BEARISH_EXTENDED":
        result["is_ideal_entry"] = False
        result["entry_quality"] = 0.05
        result["reason"] = f"PUT_BLOQUEADO_BAIXA_ESTENDIDA({extended_severity:.1f}ATR)"
        result["recommendation"] = "NAO_ENTRAR"
        result["extended_block"] = True
        return result
    
    # CALL após alta estendida = BLOQUEAR (comprar no topo)
    if direction == "CALL" and extended_move == "BULLISH_EXTENDED":
        result["is_ideal_entry"] = False
        result["entry_quality"] = 0.05
        result["reason"] = f"CALL_BLOQUEADO_ALTA_ESTENDIDA({extended_severity:.1f}ATR)"
        result["recommendation"] = "NAO_ENTRAR"
        result["extended_block"] = True
        return result
    
    # ===== VERIFICAR POSIÇÃO EXTREMA NO RANGE =====
    # PUT quando preço está no fundo do range (< 25%) = BLOQUEAR
    if direction == "PUT" and position_pct <= 0.25:
        result["is_ideal_entry"] = False
        result["entry_quality"] = 0.10
        result["reason"] = f"PUT_BLOQUEADO_NO_FUNDO({position_pct:.0%}_do_range)"
        result["recommendation"] = "NAO_ENTRAR"
        result["extended_block"] = True
        return result
    
    # CALL quando preço está no topo do range (> 75%) = BLOQUEAR
    if direction == "CALL" and position_pct >= 0.75:
        result["is_ideal_entry"] = False
        result["entry_quality"] = 0.10
        result["reason"] = f"CALL_BLOQUEADO_NO_TOPO({position_pct:.0%}_do_range)"
        result["recommendation"] = "NAO_ENTRAR"
        result["extended_block"] = True
        return result
    entry_quality = structure.get("entry_quality", 0.0)
    bos_detected = structure.get("bos_detected", False)
    bos_direction = structure.get("bos_direction")
    
    # ===== VERIFICAR ALINHAMENTO ESTRUTURA + DIREÇÃO =====
    
    # CALL em tendência de ALTA
    if direction == "CALL" and trend == "BULLISH":
        result["structure_aligned"] = True
        
        # Ponto IDEAL: Preço próximo do HL (fundo mais alto) = PULLBACK
        if ideal_direction == "CALL" and entry_quality >= 0.5:
            result["is_ideal_entry"] = True
            result["entry_quality"] = entry_quality
            result["reason"] = f"CALL_ideal_no_HL(pullback_quality={entry_quality:.0%})"
            result["recommendation"] = "ENTRAR"
        
        # BOS de alta = força extra
        elif bos_detected and bos_direction == "BULLISH":
            result["is_ideal_entry"] = True
            result["entry_quality"] = 0.75
            result["reason"] = "CALL_apos_BOS_ALTA(rompimento_confirmado)"
            result["recommendation"] = "ENTRAR"
        
        # Preço no fundo do range em tendência de alta = bom ponto
        elif price_position == "bottom_of_range":
            result["is_ideal_entry"] = True
            result["entry_quality"] = 0.70
            result["reason"] = "CALL_no_fundo_do_range(zona_de_suporte)"
            result["recommendation"] = "ENTRAR"
        
        # Preço no topo em tendência de alta = NÃO IDEAL (aguardar pullback)
        elif price_position == "top_of_range":
            result["is_ideal_entry"] = False
            result["entry_quality"] = 0.20
            result["reason"] = "CALL_no_topo_AGUARDAR_PULLBACK"
            result["recommendation"] = "AGUARDAR"
        
        # Preço no meio = parcialmente ok
        else:
            result["is_ideal_entry"] = entry_quality >= 0.4
            result["entry_quality"] = max(0.4, entry_quality)
            result["reason"] = f"CALL_no_meio_estrutura(quality={entry_quality:.0%})"
            result["recommendation"] = "ENTRAR_COM_CAUTELA" if result["is_ideal_entry"] else "AGUARDAR"
    
    # PUT em tendência de BAIXA
    elif direction == "PUT" and trend == "BEARISH":
        result["structure_aligned"] = True
        
        # Ponto IDEAL: Preço próximo do LH (topo mais baixo) = THROWBACK
        if ideal_direction == "PUT" and entry_quality >= 0.5:
            result["is_ideal_entry"] = True
            result["entry_quality"] = entry_quality
            result["reason"] = f"PUT_ideal_no_LH(throwback_quality={entry_quality:.0%})"
            result["recommendation"] = "ENTRAR"
        
        # BOS de baixa = força extra
        elif bos_detected and bos_direction == "BEARISH":
            result["is_ideal_entry"] = True
            result["entry_quality"] = 0.75
            result["reason"] = "PUT_apos_BOS_BAIXA(rompimento_confirmado)"
            result["recommendation"] = "ENTRAR"
        
        # Preço no topo do range em tendência de baixa = bom ponto
        elif price_position == "top_of_range":
            result["is_ideal_entry"] = True
            result["entry_quality"] = 0.70
            result["reason"] = "PUT_no_topo_do_range(zona_de_resistencia)"
            result["recommendation"] = "ENTRAR"
        
        # Preço no fundo em tendência de baixa = NÃO IDEAL (aguardar throwback)
        elif price_position == "bottom_of_range":
            result["is_ideal_entry"] = False
            result["entry_quality"] = 0.20
            result["reason"] = "PUT_no_fundo_AGUARDAR_THROWBACK"
            result["recommendation"] = "AGUARDAR"
        
        # Preço no meio = parcialmente ok
        else:
            result["is_ideal_entry"] = entry_quality >= 0.4
            result["entry_quality"] = max(0.4, entry_quality)
            result["reason"] = f"PUT_no_meio_estrutura(quality={entry_quality:.0%})"
            result["recommendation"] = "ENTRAR_COM_CAUTELA" if result["is_ideal_entry"] else "AGUARDAR"
    
    # CALL em tendência de BAIXA = CONTRA TENDÊNCIA (cuidado!)
    elif direction == "CALL" and trend == "BEARISH":
        result["structure_aligned"] = False
        
        # Só permite se BOS de alta (reversão)
        if bos_detected and bos_direction == "BULLISH":
            result["is_ideal_entry"] = True
            result["entry_quality"] = 0.60
            result["reason"] = "CALL_reversao_BOS_ALTA(possivel_mudanca_tendencia)"
            result["recommendation"] = "ENTRAR_COM_CAUTELA"
        
        # Preço no fundo extremo pode indicar reversão
        elif price_position == "bottom_of_range":
            result["is_ideal_entry"] = False
            result["entry_quality"] = 0.35
            result["reason"] = "CALL_contra_tendencia_no_fundo(aguardar_confirmacao)"
            result["recommendation"] = "AGUARDAR_CONFIRMACAO"
        
        else:
            result["is_ideal_entry"] = False
            result["entry_quality"] = 0.15
            result["reason"] = "CALL_CONTRA_TENDENCIA_BAIXA"
            result["recommendation"] = "NAO_ENTRAR"
    
    # PUT em tendência de ALTA = CONTRA TENDÊNCIA (cuidado!)
    elif direction == "PUT" and trend == "BULLISH":
        result["structure_aligned"] = False
        
        # Só permite se BOS de baixa (reversão)
        if bos_detected and bos_direction == "BEARISH":
            result["is_ideal_entry"] = True
            result["entry_quality"] = 0.60
            result["reason"] = "PUT_reversao_BOS_BAIXA(possivel_mudanca_tendencia)"
            result["recommendation"] = "ENTRAR_COM_CAUTELA"
        
        # Preço no topo extremo pode indicar reversão
        elif price_position == "top_of_range":
            result["is_ideal_entry"] = False
            result["entry_quality"] = 0.35
            result["reason"] = "PUT_contra_tendencia_no_topo(aguardar_confirmacao)"
            result["recommendation"] = "AGUARDAR_CONFIRMACAO"
        
        else:
            result["is_ideal_entry"] = False
            result["entry_quality"] = 0.15
            result["reason"] = "PUT_CONTRA_TENDENCIA_ALTA"
            result["recommendation"] = "NAO_ENTRAR"
    
    # MERCADO LATERAL/NEUTRO
    else:
        result["structure_aligned"] = False
        result["is_ideal_entry"] = False
        result["entry_quality"] = 0.25
        result["reason"] = "ESTRUTURA_NEUTRA_SEM_DIRECAO_CLARA"
        result["recommendation"] = "AGUARDAR_DEFINICAO"
    
    return result


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

    # Momentum fraco = breakout fraco (reduzido de 0.15 para 0.10)
    if momentum < 0.10:
        confidence *= 0.80  # penaliza menos

    # 5. ALINHAMENTO DE VELAS
    last_3 = df_m1.tail(3)
    aligned = 0
    for _, row in last_3.iterrows():
        c = float(row["close"])
        o = float(row["open"])
        if (direction == "CALL" and c > o) or (direction == "PUT" and c < o):
            aligned += 1

    alignment_ratio = aligned / 3.0
    if alignment_ratio >= 0.67:  # 2 de 3 velas alinhadas
        confidence *= 1.15  # bônus
    elif alignment_ratio < 0.34:
        confidence *= 0.80  # penaliza

    # Confidence final
    confidence = min(1.0, max(0.0, confidence))

    # Score mínimo para aprovar (reduzido de 0.50 para 0.40)
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

# ===================== VALIDAÇÃO DE MOMENTUM DE CURTO PRAZO (1-5 MIN) =====================
def validate_short_term_momentum(df_m1: pd.DataFrame, direction: str, atr_val: float) -> Dict[str, Any]:
    """
    NOVO: Valida se o momentum de curto prazo (últimas 1-5 velas) confirma a direção.
    Esta é uma verificação CRÍTICA para evitar entradas contra a tendência imediata.
    
    Args:
        df_m1: DataFrame com dados M1
        direction: "CALL" ou "PUT" - direção pretendida
        atr_val: ATR atual
    
    Returns:
        Dict com: valid, confidence, reason, momentum_score
    """
    if len(df_m1) < 10:
        return {"valid": False, "confidence": 0.0, "reason": "dados_insuficientes", "momentum_score": 0.0}
    
    # ===== ANÁLISE DAS ÚLTIMAS 5 VELAS (período crítico) =====
    last_5 = df_m1.tail(5)
    last_3 = df_m1.tail(3)
    last_1 = df_m1.tail(1)
    
    closes_5 = last_5["close"].to_numpy(float)
    opens_5 = last_5["open"].to_numpy(float)
    highs_5 = last_5["high"].to_numpy(float)
    lows_5 = last_5["low"].to_numpy(float)
    
    # 1. DIREÇÃO DAS ÚLTIMAS 5 VELAS
    bullish_candles = sum(1 for i in range(len(closes_5)) if closes_5[i] > opens_5[i])
    bearish_candles = 5 - bullish_candles
    
    # 2. MOMENTUM DE PREÇO (variação nas últimas 5 velas)
    price_change_5 = closes_5[-1] - closes_5[0]
    price_change_pct = price_change_5 / max(atr_val, 1e-9)
    
    # 3. DIREÇÃO DAS ÚLTIMAS 3 VELAS (mais recente)
    closes_3 = last_3["close"].to_numpy(float)
    opens_3 = last_3["open"].to_numpy(float)
    bullish_3 = sum(1 for i in range(len(closes_3)) if closes_3[i] > opens_3[i])
    
    # 4. ÚLTIMA VELA (gatilho imediato)
    last_close = float(last_1["close"].iloc[0])
    last_open = float(last_1["open"].iloc[0])
    last_high = float(last_1["high"].iloc[0])
    last_low = float(last_1["low"].iloc[0])
    last_bullish = last_close > last_open
    
    # 5. ANÁLISE DE HIGHER HIGHS / LOWER LOWS nas últimas 5
    making_higher_highs = all(highs_5[i] >= highs_5[i-1] - 0.1*atr_val for i in range(1, len(highs_5)))
    making_lower_lows = all(lows_5[i] <= lows_5[i-1] + 0.1*atr_val for i in range(1, len(lows_5)))
    
    # ===== ESTRATÉGIA PERNADA B = REVERSÃO =====
    # A Pernada B detecta fim de correção, então é NORMAL que:
    # - CALL apareça após velas de BAIXA (fim da correção de baixa)
    # - PUT apareça após velas de ALTA (fim da correção de alta)
    # 
    # O que queremos verificar é se há SINAIS DE REVERSÃO, não continuação!
    
    if direction == "CALL":
        # Para CALL em reversão, verificamos:
        # - A correção está perdendo força (velas menores, menos bearish)
        # - Há sinais de recuperação (última vela verde, rejeição de mínima)
        
        momentum_score = 0.50  # Base neutra (50%)
        reasons = []
        
        # 1. Se as últimas velas são bearish, isso é ESPERADO para reversão CALL
        if bearish_candles >= 3:
            momentum_score += 0.10  # OK - setup de reversão normal
            reasons.append("correcao_detectada")
        
        # 2. SINAIS DE REVERSÃO - última vela é bullish (início da reversão)
        if last_bullish:
            momentum_score += 0.25
            reasons.append("gatilho_alta")
        else:
            # Última vela bearish mas PEQUENA = exaustão
            last_body = abs(last_close - last_open)
            if last_body < 0.4 * atr_val:
                momentum_score += 0.10
                reasons.append("exaustao_vendedores")
            # Última vela com pavio inferior grande = rejeição de baixa
            lower_wick = min(last_close, last_open) - last_low
            if lower_wick > last_body:
                momentum_score += 0.15
                reasons.append("rejeicao_baixa")
        
        # 3. Últimas 3 velas mostram reversão?
        if bullish_3 >= 2:
            momentum_score += 0.15
            reasons.append("3v_revertendo")
        elif bullish_3 == 1:
            momentum_score += 0.05
        
        # 4. Verificar se preço parou de cair (estabilização)
        last_2_change = closes_5[-1] - closes_5[-3]
        if last_2_change > 0:
            momentum_score += 0.10
            reasons.append("preco_recuperando")
        elif abs(last_2_change) < 0.2 * atr_val:
            momentum_score += 0.05
            reasons.append("preco_estabilizando")
        
    else:  # PUT
        # Para PUT em reversão, verificamos:
        # - A correção de alta está perdendo força
        # - Há sinais de queda (última vela vermelha, rejeição de máxima)
        
        momentum_score = 0.50  # Base neutra (50%)
        reasons = []
        
        # 1. Se as últimas velas são bullish, isso é ESPERADO para reversão PUT
        if bullish_candles >= 3:
            momentum_score += 0.10  # OK - setup de reversão normal
            reasons.append("correcao_detectada")
        
        # 2. SINAIS DE REVERSÃO - última vela é bearish (início da reversão)
        if not last_bullish:
            momentum_score += 0.25
            reasons.append("gatilho_baixa")
        else:
            # Última vela bullish mas PEQUENA = exaustão
            last_body = abs(last_close - last_open)
            if last_body < 0.4 * atr_val:
                momentum_score += 0.10
                reasons.append("exaustao_compradores")
            # Última vela com pavio superior grande = rejeição de alta
            upper_wick = last_high - max(last_close, last_open)
            if upper_wick > last_body:
                momentum_score += 0.15
                reasons.append("rejeicao_alta")
        
        # 3. Últimas 3 velas mostram reversão?
        bearish_3 = 3 - bullish_3
        if bearish_3 >= 2:
            momentum_score += 0.15
            reasons.append("3v_revertendo")
        elif bearish_3 == 1:
            momentum_score += 0.05
        
        # 4. Verificar se preço parou de subir (estabilização)
        last_2_change = closes_5[-1] - closes_5[-3]
        if last_2_change < 0:
            momentum_score += 0.10
            reasons.append("preco_caindo")
        elif abs(last_2_change) < 0.2 * atr_val:
            momentum_score += 0.05
            reasons.append("preco_estabilizando")
    
    # Normaliza score entre 0 e 1
    momentum_score = max(0.0, min(1.0, momentum_score))
    
    # ===== DECISÃO FINAL =====
    # Como a lógica agora começa em 0.50 (neutra), qualquer sinal positivo passa
    MIN_MOMENTUM_SCORE = 0.40  # 40% = tem pelo menos algum sinal de reversão
    
    if momentum_score >= MIN_MOMENTUM_SCORE:
        return {
            "valid": True,
            "confidence": momentum_score,
            "reason": f"reversao_ok({','.join(reasons)})",
            "momentum_score": momentum_score,
            "bullish_5": bullish_candles,
            "price_change": price_change_pct
        }
    else:
        return {
            "valid": False,
            "confidence": momentum_score,
            "reason": f"sem_sinal_reversao({','.join(reasons)})",
            "momentum_score": momentum_score,
            "bullish_5": bullish_candles,
            "price_change": price_change_pct
        }

# ===================== SISTEMA DE DECISÃO INTELIGENTE - CONFLUÊNCIA OBRIGATÓRIA =====================

def read_market_context(df_m1: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    """
    🎯 LER O CENÁRIO DO MERCADO PRIMEIRO - Antes de qualquer decisão!
    
    CONHECIMENTO REAL DE OPÇÕES BINÁRIAS (M1):
    
    1. IDENTIFICAR O CONTEXTO:
       - TENDÊNCIA FORTE = Operar a favor (pullback)
       - TENDÊNCIA FRACA = Cuidado, pode reverter
       - LATERAL/RANGE = NÃO OPERAR ou operar nas extremidades
       - PÓS-MOVIMENTO FORTE = ESPERAR correção
    
    2. O QUE FUNCIONA DE VERDADE:
       - Pullback em tendência (38-62% de retração)
       - Rompimento de consolidação COM FORÇA
       - Reversão em S/R forte COM CONFIRMAÇÃO
       - A FAVOR da tendência principal
    
    3. O QUE NÃO FUNCIONA (evitar):
       - Operar contra tendência forte
       - Entrar no meio do range
       - Entrar após candle muito grande (exaustão)
       - Entrar durante consolidação
       - Entrar sem confirmação de vela
    """
    context = {
        "can_trade": False,  # Pode operar?
        "market_type": "INDEFINIDO",  # TENDENCIA_ALTA, TENDENCIA_BAIXA, LATERAL, CONSOLIDACAO
        "trend_strength": 0.0,  # 0-1
        "recommended_direction": "NONE",  # CALL, PUT, NONE
        "best_setup": "NONE",  # PULLBACK, ROMPIMENTO, REVERSAO, NONE
        "wait_reason": "",
        "confidence": 0.0,
        
        # Detalhes da análise
        "is_trending": False,
        "trend_direction": "NONE",
        "is_pullback": False,
        "pullback_quality": 0.0,
        "is_breakout": False,
        "breakout_direction": "NONE",
        "is_at_sr": False,
        "sr_type": "NONE",
        "is_exhausted": False,
        "is_consolidating": False,
        "last_candle_confirms": False,
    }
    
    if len(df_m1) < 30:
        context["wait_reason"] = "DADOS_INSUFICIENTES"
        return context
    
    try:
        atr_safe = max(atr_val, 1e-9)
        
        highs = df_m1["high"].tail(30).values.astype(float)
        lows = df_m1["low"].tail(30).values.astype(float)
        closes = df_m1["close"].tail(30).values.astype(float)
        opens = df_m1["open"].tail(30).values.astype(float)
        
        current_price = closes[-1]
        current_open = opens[-1]
        current_high = highs[-1]
        current_low = lows[-1]
        
        # ═══════════════════════════════════════════════════════════════
        # 1. IDENTIFICAR TIPO DE MERCADO
        # ═══════════════════════════════════════════════════════════════
        
        # Movimento das últimas 20 velas
        move_20 = closes[-1] - closes[-20]
        move_20_atr = move_20 / atr_safe
        
        # Movimento das últimas 10 velas
        move_10 = closes[-1] - closes[-10]
        move_10_atr = move_10 / atr_safe
        
        # Movimento das últimas 5 velas
        move_5 = closes[-1] - closes[-5]
        move_5_atr = move_5 / atr_safe
        
        # Range das últimas 10 velas
        range_10 = max(highs[-10:]) - min(lows[-10:])
        range_10_atr = range_10 / atr_safe
        
        # Contar velas por direção (últimas 10)
        green_10 = sum(1 for i in range(-10, 0) if closes[i] > opens[i])
        red_10 = 10 - green_10
        
        # ===== DETECTAR TENDÊNCIA =====
        # Tendência forte = movimento consistente + maioria das velas na mesma direção
        
        if move_20_atr > 2.0 and green_10 >= 7:
            context["market_type"] = "TENDENCIA_ALTA"
            context["is_trending"] = True
            context["trend_direction"] = "CALL"
            context["trend_strength"] = min(1.0, abs(move_20_atr) / 4)
            
        elif move_20_atr < -2.0 and red_10 >= 7:
            context["market_type"] = "TENDENCIA_BAIXA"
            context["is_trending"] = True
            context["trend_direction"] = "PUT"
            context["trend_strength"] = min(1.0, abs(move_20_atr) / 4)
            
        elif range_10_atr < 1.5:
            context["market_type"] = "CONSOLIDACAO"
            context["is_consolidating"] = True
            context["trend_strength"] = 0.2
            
        else:
            context["market_type"] = "LATERAL"
            context["trend_strength"] = 0.3
        
        # ═══════════════════════════════════════════════════════════════
        # 2. DETECTAR PULLBACK (retração em tendência)
        # ═══════════════════════════════════════════════════════════════
        
        if context["is_trending"]:
            # Em tendência de ALTA, pullback = preço caiu um pouco
            if context["trend_direction"] == "CALL":
                # Swing high e low das últimas 15 velas
                swing_high = max(highs[-15:-3])
                swing_low = min(lows[-15:-3])
                swing_range = swing_high - swing_low
                
                if swing_range > atr_safe * 0.5:
                    retrace = (swing_high - current_price) / swing_range
                    # Pullback ideal: 38% a 62% de retração
                    if 0.30 <= retrace <= 0.70:
                        context["is_pullback"] = True
                        context["pullback_quality"] = 1.0 - abs(retrace - 0.50) * 2
                        context["best_setup"] = "PULLBACK"
                        context["recommended_direction"] = "CALL"
            
            # Em tendência de BAIXA, pullback = preço subiu um pouco
            elif context["trend_direction"] == "PUT":
                swing_high = max(highs[-15:-3])
                swing_low = min(lows[-15:-3])
                swing_range = swing_high - swing_low
                
                if swing_range > atr_safe * 0.5:
                    retrace = (current_price - swing_low) / swing_range
                    # Pullback ideal: 38% a 62% de retração
                    if 0.30 <= retrace <= 0.70:
                        context["is_pullback"] = True
                        context["pullback_quality"] = 1.0 - abs(retrace - 0.50) * 2
                        context["best_setup"] = "PULLBACK"
                        context["recommended_direction"] = "PUT"
        
        # ═══════════════════════════════════════════════════════════════
        # 3. DETECTAR ROMPIMENTO DE CONSOLIDAÇÃO
        # ═══════════════════════════════════════════════════════════════
        
        # Range das últimas 8 velas (possível consolidação)
        consol_high = max(highs[-8:-1])  # Exclui última vela
        consol_low = min(lows[-8:-1])
        consol_range = consol_high - consol_low
        
        # Se range pequeno e última vela rompeu
        if consol_range < atr_safe * 1.5:
            # Rompeu para cima?
            if current_high > consol_high + atr_safe * 0.2 and closes[-1] > consol_high:
                context["is_breakout"] = True
                context["breakout_direction"] = "CALL"
                context["best_setup"] = "ROMPIMENTO"
                context["recommended_direction"] = "CALL"
            
            # Rompeu para baixo?
            elif current_low < consol_low - atr_safe * 0.2 and closes[-1] < consol_low:
                context["is_breakout"] = True
                context["breakout_direction"] = "PUT"
                context["best_setup"] = "ROMPIMENTO"
                context["recommended_direction"] = "PUT"
        
        # ═══════════════════════════════════════════════════════════════
        # 4. DETECTAR SUPORTE/RESISTÊNCIA
        # ═══════════════════════════════════════════════════════════════
        
        # Máxima e mínima das últimas 30 velas
        range_high = max(highs)
        range_low = min(lows)
        range_size = range_high - range_low
        
        if range_size > 0:
            position = (current_price - range_low) / range_size
            
            # No suporte (< 15% do range)
            if position < 0.15:
                context["is_at_sr"] = True
                context["sr_type"] = "SUPORTE"
                if context["best_setup"] == "NONE":
                    context["best_setup"] = "REVERSAO"
                    context["recommended_direction"] = "CALL"
            
            # Na resistência (> 85% do range)
            elif position > 0.85:
                context["is_at_sr"] = True
                context["sr_type"] = "RESISTENCIA"
                if context["best_setup"] == "NONE":
                    context["best_setup"] = "REVERSAO"
                    context["recommended_direction"] = "PUT"
        
        # ═══════════════════════════════════════════════════════════════
        # 5. DETECTAR EXAUSTÃO (movimento forte demais = esperar)
        # ═══════════════════════════════════════════════════════════════
        
        # Última vela muito grande?
        last_body = abs(closes[-1] - opens[-1])
        avg_body = np.mean([abs(closes[i] - opens[i]) for i in range(-10, -1)])
        
        if last_body > avg_body * 2.5:
            context["is_exhausted"] = True
        
        # Movimento das últimas 3 velas muito forte?
        if abs(move_5_atr) > 2.5:
            context["is_exhausted"] = True
        
        # ═══════════════════════════════════════════════════════════════
        # 6. CONFIRMAÇÃO DA ÚLTIMA VELA
        # ═══════════════════════════════════════════════════════════════
        
        last_is_green = closes[-1] > opens[-1]
        last_body_ratio = last_body / (current_high - current_low) if (current_high - current_low) > 0 else 0
        
        # Vela de confirmação: corpo > 50% do range, não é doji
        is_confirmation_candle = last_body_ratio > 0.50
        
        if context["recommended_direction"] == "CALL" and last_is_green and is_confirmation_candle:
            context["last_candle_confirms"] = True
        elif context["recommended_direction"] == "PUT" and not last_is_green and is_confirmation_candle:
            context["last_candle_confirms"] = True
        
        # ═══════════════════════════════════════════════════════════════
        # 7. DECISÃO: PODE OPERAR?
        # ═══════════════════════════════════════════════════════════════
        
        # NÃO OPERAR SE:
        if context["is_consolidating"] and not context["is_breakout"]:
            context["can_trade"] = False
            context["wait_reason"] = "CONSOLIDACAO_SEM_ROMPIMENTO"
            context["recommended_direction"] = "NONE"
            
        elif context["is_exhausted"]:
            context["can_trade"] = False
            context["wait_reason"] = "MOVIMENTO_EXAUSTO_ESPERAR_CORRECAO"
            context["recommended_direction"] = "NONE"
            
        elif context["market_type"] == "LATERAL" and not context["is_at_sr"]:
            context["can_trade"] = False
            context["wait_reason"] = "MERCADO_LATERAL_SEM_SR"
            context["recommended_direction"] = "NONE"
            
        elif context["best_setup"] == "NONE":
            context["can_trade"] = False
            context["wait_reason"] = "SEM_SETUP_CLARO"
            
        # PODE OPERAR SE:
        elif context["is_pullback"] and context["last_candle_confirms"]:
            context["can_trade"] = True
            context["confidence"] = 0.7 + context["pullback_quality"] * 0.2
            
        elif context["is_breakout"]:
            context["can_trade"] = True
            context["confidence"] = 0.75
            
        elif context["is_at_sr"] and context["last_candle_confirms"]:
            context["can_trade"] = True
            context["confidence"] = 0.65
            
        elif context["is_pullback"] and not context["last_candle_confirms"]:
            context["can_trade"] = False
            context["wait_reason"] = "PULLBACK_SEM_CONFIRMACAO_VELA"
            
        elif context["is_at_sr"] and not context["last_candle_confirms"]:
            context["can_trade"] = False
            context["wait_reason"] = "SR_SEM_CONFIRMACAO_VELA"
            
        else:
            context["can_trade"] = False
            context["wait_reason"] = "AGUARDANDO_SETUP"
    
    except Exception as e:
        context["wait_reason"] = f"ERRO: {str(e)[:30]}"
    
    return context


def detect_consolidation(df_m1: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    """
    🎯 DETECTA CONSOLIDAÇÃO (MERCADO LATERAL)
    
    Consolidação = preço se movendo em range pequeno após tendência
    
    REGRA DE OURO:
    - Durante consolidação = NÃO ENTRA!
    - Espera ROMPER a consolidação
    - Rompeu para baixo = PUT
    - Rompeu para cima = CALL
    """
    result = {
        "is_consolidating": False,
        "consolidation_high": 0.0,
        "consolidation_low": 0.0,
        "consolidation_range": 0.0,
        "consolidation_candles": 0,
        "broke_up": False,
        "broke_down": False,
        "breakout_strength": 0.0,
        "should_wait": False,
        "trend_before_consolidation": "NEUTRAL",  # ALTA ou BAIXA
    }
    
    if len(df_m1) < 20:
        return result
    
    try:
        highs = df_m1["high"].tail(20).values.astype(float)
        lows = df_m1["low"].tail(20).values.astype(float)
        closes = df_m1["close"].tail(20).values.astype(float)
        opens = df_m1["open"].tail(20).values.astype(float)
        current_price = closes[-1]
        current_high = highs[-1]
        current_low = lows[-1]
        
        atr_safe = max(atr_val, 1e-9)
        
        # ===== 1. DETECTAR TENDÊNCIA ANTERIOR (primeiras 10 velas) =====
        first_10_move = closes[9] - closes[0]
        if first_10_move > atr_safe * 1.0:
            result["trend_before_consolidation"] = "ALTA"
        elif first_10_move < -atr_safe * 1.0:
            result["trend_before_consolidation"] = "BAIXA"
        
        # ===== 2. ANALISAR ÚLTIMAS 8 VELAS PARA CONSOLIDAÇÃO =====
        last_8_highs = highs[-8:]
        last_8_lows = lows[-8:]
        last_8_closes = closes[-8:]
        
        consolidation_high = float(max(last_8_highs))
        consolidation_low = float(min(last_8_lows))
        consolidation_range = consolidation_high - consolidation_low
        
        result["consolidation_high"] = consolidation_high
        result["consolidation_low"] = consolidation_low
        result["consolidation_range"] = consolidation_range
        
        # ===== 3. VERIFICAR SE É CONSOLIDAÇÃO =====
        # Consolidação = range pequeno (< 1.5 ATR) e velas pequenas
        is_small_range = consolidation_range < atr_safe * 1.5
        
        # Verificar se as velas estão "presas" no range
        candles_in_range = 0
        for i in range(-8, 0):
            if lows[i] >= consolidation_low - atr_safe * 0.2 and highs[i] <= consolidation_high + atr_safe * 0.2:
                candles_in_range += 1
        
        result["consolidation_candles"] = candles_in_range
        
        # É consolidação se:
        # - Range pequeno (< 1.5 ATR)
        # - Maioria das velas dentro do range (>= 6 de 8)
        if is_small_range and candles_in_range >= 6:
            result["is_consolidating"] = True
            result["should_wait"] = True  # ESPERAR ROMPIMENTO!
        
        # ===== 4. VERIFICAR ROMPIMENTO =====
        # A última vela rompeu a consolidação?
        
        # Rompimento para CIMA
        if current_high > consolidation_high + atr_safe * 0.2:
            if current_close := closes[-1] > consolidation_high:
                result["broke_up"] = True
                result["should_wait"] = False  # Rompeu! Pode entrar CALL
                result["breakout_strength"] = (current_high - consolidation_high) / atr_safe
        
        # Rompimento para BAIXO
        if current_low < consolidation_low - atr_safe * 0.2:
            if closes[-1] < consolidation_low:
                result["broke_down"] = True
                result["should_wait"] = False  # Rompeu! Pode entrar PUT
                result["breakout_strength"] = (consolidation_low - current_low) / atr_safe
        
        # ===== 5. VERIFICAR SE ESTÁ NO MEIO DA CONSOLIDAÇÃO =====
        if result["is_consolidating"] and not result["broke_up"] and not result["broke_down"]:
            # Preço ainda dentro da consolidação = ESPERAR
            mid_consolidation = (consolidation_high + consolidation_low) / 2
            if consolidation_low + atr_safe * 0.3 < current_price < consolidation_high - atr_safe * 0.3:
                result["should_wait"] = True  # No meio = definitivamente esperar
                
    except Exception as e:
        pass
    
    return result


def analyze_trend_lines(df_m1: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    """
    🎯 ANÁLISE DE LTA (Linha de Tendência de Alta) e LTB (Linha de Tendência de Baixa)
    
    Detecta linhas de tendência usando regressão linear dos topos e fundos.
    """
    result = {
        "has_lta": False,  # Linha de Tendência de Alta (fundos ascendentes)
        "has_ltb": False,  # Linha de Tendência de Baixa (topos descendentes)
        "trend_type": "LATERAL",  # ALTA, BAIXA, LATERAL
        "trend_strength": 0.0,  # 0-1
        "lta_slope": 0.0,
        "ltb_slope": 0.0,
        "price_vs_lta": "above",  # above, below, at
        "price_vs_ltb": "below",  # above, below, at
        "broke_lta": False,  # Rompeu LTA para baixo (bearish)
        "broke_ltb": False,  # Rompeu LTB para cima (bullish)
    }
    
    if len(df_m1) < 20:
        return result
    
    try:
        highs = df_m1["high"].tail(20).values.astype(float)
        lows = df_m1["low"].tail(20).values.astype(float)
        closes = df_m1["close"].tail(20).values.astype(float)
        current_price = closes[-1]
        
        # Encontrar topos e fundos locais
        tops = []
        bottoms = []
        
        for i in range(2, len(highs) - 2):
            # Topo local
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                tops.append((i, highs[i]))
            # Fundo local
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                bottoms.append((i, lows[i]))
        
        # Analisar LTA (fundos ascendentes)
        if len(bottoms) >= 2:
            # Regressão linear dos fundos
            x_bottoms = np.array([b[0] for b in bottoms])
            y_bottoms = np.array([b[1] for b in bottoms])
            if len(x_bottoms) >= 2:
                slope_lta, intercept_lta = np.polyfit(x_bottoms, y_bottoms, 1)
                result["lta_slope"] = slope_lta / atr_val  # Normalizado
                
                # LTA válida se fundos estão subindo
                if slope_lta > atr_val * 0.01:  # Inclinação positiva significativa
                    result["has_lta"] = True
                    
                    # Onde o preço está em relação à LTA projetada?
                    lta_at_current = intercept_lta + slope_lta * (len(highs) - 1)
                    if current_price > lta_at_current + atr_val * 0.3:
                        result["price_vs_lta"] = "above"
                    elif current_price < lta_at_current - atr_val * 0.3:
                        result["price_vs_lta"] = "below"
                        result["broke_lta"] = True  # ROMPEU A LTA PARA BAIXO!
                    else:
                        result["price_vs_lta"] = "at"
        
        # Analisar LTB (topos descendentes)
        if len(tops) >= 2:
            # Regressão linear dos topos
            x_tops = np.array([t[0] for t in tops])
            y_tops = np.array([t[1] for t in tops])
            if len(x_tops) >= 2:
                slope_ltb, intercept_ltb = np.polyfit(x_tops, y_tops, 1)
                result["ltb_slope"] = slope_ltb / atr_val  # Normalizado
                
                # LTB válida se topos estão descendo
                if slope_ltb < -atr_val * 0.01:  # Inclinação negativa significativa
                    result["has_ltb"] = True
                    
                    # Onde o preço está em relação à LTB projetada?
                    ltb_at_current = intercept_ltb + slope_ltb * (len(highs) - 1)
                    if current_price < ltb_at_current - atr_val * 0.3:
                        result["price_vs_ltb"] = "below"
                    elif current_price > ltb_at_current + atr_val * 0.3:
                        result["price_vs_ltb"] = "above"
                        result["broke_ltb"] = True  # ROMPEU A LTB PARA CIMA!
                    else:
                        result["price_vs_ltb"] = "at"
        
        # Determinar tipo de tendência
        if result["has_lta"] and not result["has_ltb"]:
            result["trend_type"] = "ALTA"
            result["trend_strength"] = min(1.0, abs(result["lta_slope"]) * 2)
        elif result["has_ltb"] and not result["has_lta"]:
            result["trend_type"] = "BAIXA"
            result["trend_strength"] = min(1.0, abs(result["ltb_slope"]) * 2)
        elif result["has_lta"] and result["has_ltb"]:
            # Canal ou triângulo
            if result["lta_slope"] > result["ltb_slope"]:
                result["trend_type"] = "TRIANGULO_ALTA"
            else:
                result["trend_type"] = "TRIANGULO_BAIXA"
            result["trend_strength"] = 0.5
        else:
            result["trend_type"] = "LATERAL"
            result["trend_strength"] = 0.3
            
    except Exception as e:
        pass
    
    return result


def find_support_resistance(df_m1: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    """
    🎯 ENCONTRA SUPORTES E RESISTÊNCIAS REAIS
    
    Baseado em zonas onde o preço tocou múltiplas vezes (não simples min/max).
    """
    result = {
        "supports": [],  # Lista de níveis de suporte
        "resistances": [],  # Lista de níveis de resistência
        "nearest_support": 0.0,
        "nearest_resistance": 0.0,
        "at_support": False,
        "at_resistance": False,
        "broke_support": False,
        "broke_resistance": False,
    }
    
    if len(df_m1) < 30:
        return result
    
    try:
        highs = df_m1["high"].tail(30).values.astype(float)
        lows = df_m1["low"].tail(30).values.astype(float)
        closes = df_m1["close"].tail(30).values.astype(float)
        current_price = closes[-1]
        
        # Criar zonas de preço (dividir range em 20 partes)
        price_max = max(highs)
        price_min = min(lows)
        price_range = price_max - price_min
        zone_size = price_range / 20 if price_range > 0 else atr_val
        
        # Contar toques em cada zona
        zone_touches = {}
        for i in range(len(highs)):
            # Marca zonas onde o preço "tocou" (high ou low)
            high_zone = int((highs[i] - price_min) / zone_size) if zone_size > 0 else 0
            low_zone = int((lows[i] - price_min) / zone_size) if zone_size > 0 else 0
            
            zone_touches[high_zone] = zone_touches.get(high_zone, 0) + 1
            zone_touches[low_zone] = zone_touches.get(low_zone, 0) + 1
        
        # Zonas com 3+ toques são suporte/resistência
        significant_zones = {k: v for k, v in zone_touches.items() if v >= 3}
        
        current_zone = int((current_price - price_min) / zone_size) if zone_size > 0 else 0
        
        for zone, touches in significant_zones.items():
            zone_price = price_min + (zone + 0.5) * zone_size
            
            if zone < current_zone:
                result["supports"].append(zone_price)
            elif zone > current_zone:
                result["resistances"].append(zone_price)
        
        # Encontrar suporte/resistência mais próximos
        if result["supports"]:
            result["nearest_support"] = max(result["supports"])
            # Está no suporte?
            if abs(current_price - result["nearest_support"]) < atr_val * 0.5:
                result["at_support"] = True
            # Rompeu suporte para baixo?
            if current_price < result["nearest_support"] - atr_val * 0.3:
                result["broke_support"] = True
        
        if result["resistances"]:
            result["nearest_resistance"] = min(result["resistances"])
            # Está na resistência?
            if abs(current_price - result["nearest_resistance"]) < atr_val * 0.5:
                result["at_resistance"] = True
            # Rompeu resistência para cima?
            if current_price > result["nearest_resistance"] + atr_val * 0.3:
                result["broke_resistance"] = True
                
    except Exception as e:
        pass
    
    return result


# ===================== SISTEMA DE CONFLUENCIA PARA 80%+ WIN RATE =====================
def smart_entry_decision_v2(
    direction: str,
    setup: Dict[str, Any],
    df_m1: pd.DataFrame,
    atr_val: float,
    ai_stats: Dict[str, Any] = None,
    ai_prediction: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    DECISAO DE ENTRADA RIGOROSA COM SISTEMA DE CONFLUENCIA

    Exige MINIMO 4 de 6 criterios para entrar:
    1. Padrao de candlestick 80%+ (OBRIGATORIO)
    2. Estrutura de mercado favoravel (OBRIGATORIO)
    3. Momentum de curto prazo confirmado (OBRIGATORIO)
    4. Distancia segura de S/R (OBRIGATORIO)
    5. Tendencia geral favoravel - EMA (DESEJAVEL)
    6. IA historico aprova - WR >= 55% (DESEJAVEL)

    Retorna: Dict com allow_entry, confluences, reasons, blocks
    """
    result = {
        "allow_entry": False,
        "final_score": 0,
        "confidence": 0.0,
        "reasons": [],
        "blocks": [],
        "warnings": [],
        "confluences": 0,
        "max_confluences": 7,
        "decision_reason": "",
        "detected_patterns": [],
        "candle_agents": [],
        "market_type": "CONFLUENCIA",
        "override_direction": None
    }

    ativo = setup.get("ativo", "ATIVO")

    if df_m1 is None or len(df_m1) < 60:
        result["decision_reason"] = "DADOS_INSUFICIENTES"
        result["blocks"].append("DADOS_INSUFICIENTES")
        return result

    # ===== CRITERIO 0: MEMORIA INTELIGENTE (BLOQUEIO PREVENTIVO) =====
    if SMART_MEMORY_AVAILABLE:
        try:
            memory_blocked, memory_reason = should_block_trade(ativo, direction)
            if memory_blocked:
                result["blocks"].append(f"MEMORIA:{memory_reason}")
                result["decision_reason"] = f"MEMORIA_BLOCK:{memory_reason}"
                logging.info(f"[MEMORIA] {ativo} {direction} BLOQUEADO: {memory_reason}")
                return result  # BLOQUEIO IMEDIATO pela memoria
        except Exception as e:
            logging.debug(f"[MEMORIA] Erro ao verificar: {e}")

    atr_safe = max(atr_val, 1e-9)
    confluences = 0.0

    # ===== CRITERIO 1: PADRAO DE CANDLESTICK 80%+ (OBRIGATORIO) =====
    padrao_80 = None

    if setup.get("padrao"):
        # Padrao ja veio no setup
        conf = setup.get("confiabilidade", 0)
        if conf >= 0.80:
            padrao_80 = {
                "nome": setup.get("padrao"),
                "confiabilidade": conf,
                "direcao": setup.get("dir", direction)
            }

    if not padrao_80 and PATTERN_DETECTOR_AVAILABLE:
        try:
            result_patterns = analyze_patterns(df=df_m1, direcao_sinal=direction, atr_val=atr_safe)
            padroes_candle = result_patterns.get("padroes_candlestick", [])

            for p in padroes_candle:
                if p.get("direcao") == direction and p.get("confiabilidade", 0) >= 0.80:
                    if not padrao_80 or p.get("confiabilidade", 0) > padrao_80.get("confiabilidade", 0):
                        padrao_80 = p
        except Exception as e:
            logging.debug(f"[CONFLUENCIA] Erro ao detectar padroes: {e}")

    if padrao_80:
        confluences += 1
        result["reasons"].append(f"PADRAO:{padrao_80['nome']}({padrao_80['confiabilidade']*100:.0f}%)")
        result["detected_patterns"].append(padrao_80)
    else:
        result["blocks"].append("SEM_PADRAO_80+")
        result["decision_reason"] = "SEM_PADRAO_80+"
        logging.info(f"[CONFLUENCIA] {ativo} {direction} BLOQUEADO: Sem padrao 80%+")
        return result  # BLOQUEIO IMEDIATO - padrao e obrigatorio

    # ===== CRITERIO 1B: CONSENSO DE 4 AGENTES (OBRIGATORIO) =====
    if CANDLE_AGENTS_ON:
        agents_result = candle_agents_consensus(direction, df_m1, atr_safe)
        result["candle_agents"] = agents_result.get("agents", [])
        if not agents_result.get("allow", False):
            result["blocks"].append(f"AGENTES_DIVERGENTES:{agents_result.get('reason', 'divergent')}")
            result["decision_reason"] = f"AGENTES_DIVERGENTES:{agents_result.get('reason', 'divergent')}"
            logging.info(f"[CONFLUENCIA] {ativo} {direction} BLOQUEADO: {agents_result.get('reason')}")
            return result
        confluences += 1
        result["reasons"].append("AGENTES_4x_OK")

    # ===== CRITERIO 2: ESTRUTURA DE MERCADO (OBRIGATORIO) =====
    try:
        # Analisa ultimas 40 velas para estrutura
        df_estrutura = df_m1.tail(40)
        highs = df_estrutura["high"].values
        lows = df_estrutura["low"].values
        closes = df_estrutura["close"].values

        # Detecta Higher Highs/Higher Lows ou Lower Highs/Lower Lows
        recent_high = max(highs[-10:])
        recent_low = min(lows[-10:])
        prev_high = max(highs[-20:-10])
        prev_low = min(lows[-20:-10])

        if direction == "CALL":
            # Para CALL: quer Higher Lows (fundos ascendentes)
            estrutura_ok = recent_low > prev_low * 0.999  # margem de 0.1%
        else:
            # Para PUT: quer Lower Highs (topos descendentes)
            estrutura_ok = recent_high < prev_high * 1.001

        if estrutura_ok:
            confluences += 1
            result["reasons"].append(f"ESTRUTURA:{'HL' if direction == 'CALL' else 'LH'}_OK")
        else:
            result["blocks"].append(f"ESTRUTURA_CONTRA:{direction}")
            result["decision_reason"] = "ESTRUTURA_CONTRA"
            logging.info(f"[CONFLUENCIA] {ativo} {direction} BLOQUEADO: Estrutura de mercado contra")
            return result  # BLOQUEIO IMEDIATO

    except Exception as e:
        result["blocks"].append("ESTRUTURA_ERRO")
        logging.debug(f"[CONFLUENCIA] Erro ao analisar estrutura: {e}")
        return result

    # ===== CRITERIO 3: MOMENTUM DE CURTO PRAZO (OBRIGATORIO) =====
    try:
        # Ultimas 5 velas
        df_momentum = df_m1.tail(5)

        if direction == "CALL":
            # Para CALL: quer mais velas de alta ou recuperacao
            velas_alta = sum(1 for _, row in df_momentum.iterrows() if row["close"] > row["open"])
            ultima_alta = df_momentum.iloc[-1]["close"] > df_momentum.iloc[-1]["open"]
            momentum_ok = velas_alta >= 2 or ultima_alta
        else:
            # Para PUT: quer mais velas de baixa ou queda
            velas_baixa = sum(1 for _, row in df_momentum.iterrows() if row["close"] < row["open"])
            ultima_baixa = df_momentum.iloc[-1]["close"] < df_momentum.iloc[-1]["open"]
            momentum_ok = velas_baixa >= 2 or ultima_baixa

        if momentum_ok:
            confluences += 1
            result["reasons"].append(f"MOMENTUM:CONFIRMADO")
        else:
            result["blocks"].append(f"MOMENTUM_FRACO")
            result["decision_reason"] = "MOMENTUM_FRACO"
            logging.info(f"[CONFLUENCIA] {ativo} {direction} BLOQUEADO: Momentum nao confirma")
            return result  # BLOQUEIO IMEDIATO

    except Exception as e:
        result["blocks"].append("MOMENTUM_ERRO")
        return result

    # ===== CRITERIO 4: DISTANCIA S/R (OBRIGATORIO) =====
    try:
        sr_blocked = sr_block_directional_multi(df_m1, atr_safe, direction)

        if not sr_blocked:
            confluences += 1
            result["reasons"].append("SR:SEGURO")
        else:
            result["blocks"].append(f"SR_PROXIMO")
            result["decision_reason"] = "SR_PROXIMO"
            logging.info(f"[CONFLUENCIA] {ativo} {direction} BLOQUEADO: S/R muito proximo")
            return result  # BLOQUEIO IMEDIATO

    except Exception as e:
        result["warnings"].append("SR_ERRO_CHECK")
        # Nao bloqueia, apenas avisa
        confluences += 0.5  # Da meio ponto se nao conseguiu verificar

    # ===== CRITERIO 5: TENDENCIA GERAL - EMA (DESEJAVEL) =====
    try:
        closes = df_m1["close"].values

        if len(closes) >= 50:
            ema20 = np.mean(closes[-20:])
            ema50 = np.mean(closes[-50:])

            if direction == "CALL":
                tendencia_ok = ema20 > ema50
            else:
                tendencia_ok = ema20 < ema50

            if tendencia_ok:
                confluences += 1
                result["reasons"].append(f"TENDENCIA:EMA_OK")
            else:
                result["warnings"].append("TENDENCIA_CONTRA")
        else:
            result["warnings"].append("TENDENCIA_SEM_DADOS")

    except Exception as e:
        result["warnings"].append("TENDENCIA_ERRO")

    # ===== CRITERIO 6: IA HISTORICO (DESEJAVEL) =====
    try:
        if ai_stats:
            key = ai_make_key(ativo, setup)
            pattern_data = ai_stats.get("patterns", {}).get(key, {})

            total_trades = pattern_data.get("trades", 0)
            wins = pattern_data.get("wins", 0)

            if total_trades >= 5:
                wr = wins / total_trades
                if wr >= 0.55:
                    confluences += 1
                    result["reasons"].append(f"IA:WR={wr*100:.0f}%")
                else:
                    result["warnings"].append(f"IA_WR_BAIXO:{wr*100:.0f}%")
            elif total_trades == 0:
                # Primeira vez - da meio ponto (fase de aprendizado)
                confluences += 0.5
                result["reasons"].append("IA:LEARNING")
        else:
            confluences += 0.5  # Sem dados de IA = meio ponto
            result["warnings"].append("IA_SEM_DADOS")

    except Exception as e:
        result["warnings"].append("IA_ERRO")

    # ===== DECISAO FINAL =====
    result["confluences"] = confluences

    # MINIMO 4.5 confluencias (4 obrigatorios + 0.5 desejavel)
    if confluences >= 4.5:
        # Calcula score final baseado nas confluencias
        base_score = 50
        conf_bonus = confluences * 8
        pattern_bonus = padrao_80.get("confiabilidade", 0.80) * 10

        final_score = min(95, int(base_score + conf_bonus + pattern_bonus))

        result["allow_entry"] = True
        result["final_score"] = final_score
        result["confidence"] = padrao_80.get("confiabilidade", 0.80)
        result["decision_reason"] = f"CONFLUENCIA:{confluences:.1f}/6"
        result["override_direction"] = padrao_80.get("direcao") if padrao_80.get("direcao") != direction else None

        logging.info(f"[CONFLUENCIA] {ativo} {direction} APROVADO! Confluencias={confluences:.1f}/6 Score={final_score}")
    else:
        result["decision_reason"] = f"CONFLUENCIA_INSUFICIENTE:{confluences:.1f}/4.5"
        result["blocks"].append(f"CONFLUENCIA:{confluences:.1f}/4.5")
        logging.info(f"[CONFLUENCIA] {ativo} {direction} REJEITADO: Confluencias={confluences:.1f}/4.5 (minimo 4.5)")

    return result


def _agent_from_analyze_patterns(df_m1: pd.DataFrame, direction: str, atr_val: float) -> Dict[str, Any]:
    if not PATTERN_DETECTOR_AVAILABLE:
        return {"agent": "analyze_patterns", "direction": "NEUTRAL", "confidence": 0.0, "reason": "detector_off"}
    try:
        result_patterns = analyze_patterns(df=df_m1, direcao_sinal=direction, atr_val=atr_val)
        padroes_candle = result_patterns.get("padroes_candlestick", [])
        if not padroes_candle:
            return {"agent": "analyze_patterns", "direction": "NEUTRAL", "confidence": 0.0, "reason": "no_pattern"}
        best = max(padroes_candle, key=lambda p: float(p.get("confiabilidade", 0.0)))
        conf = float(best.get("confiabilidade", 0.0))
        if conf < CANDLE_AGENT_MIN_TALIB:
            return {"agent": "analyze_patterns", "direction": "NEUTRAL", "confidence": conf, "reason": "low_conf"}
        return {
            "agent": "analyze_patterns",
            "direction": best.get("direcao", "NEUTRAL"),
            "confidence": conf,
            "reason": best.get("nome", "pattern")
        }
    except Exception as e:
        return {"agent": "analyze_patterns", "direction": "NEUTRAL", "confidence": 0.0, "reason": f"error:{e}"}


def _agent_from_talib_direct(df_m1: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    if not PATTERN_DETECTOR_AVAILABLE:
        return {"agent": "talib_direct", "direction": "NEUTRAL", "confidence": 0.0, "reason": "detector_off"}
    try:
        patterns = detect_candlestick_patterns(df_m1, atr_val)
        if not patterns:
            return {"agent": "talib_direct", "direction": "NEUTRAL", "confidence": 0.0, "reason": "no_pattern"}
        best = max(patterns, key=lambda p: float(p.confiabilidade))
        conf = float(best.confiabilidade)
        if conf < CANDLE_AGENT_MIN_TALIB:
            return {"agent": "talib_direct", "direction": "NEUTRAL", "confidence": conf, "reason": "low_conf"}
        return {
            "agent": "talib_direct",
            "direction": best.direcao,
            "confidence": conf,
            "reason": best.nome
        }
    except Exception as e:
        return {"agent": "talib_direct", "direction": "NEUTRAL", "confidence": 0.0, "reason": f"error:{e}"}


def _agent_from_manual_pattern(df_m1: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    try:
        if df_m1 is None or len(df_m1) < 3:
            return {"agent": "manual_pattern", "direction": "NEUTRAL", "confidence": 0.0, "reason": "insufficient_data"}
        last = df_m1.iloc[-2]
        prev = df_m1.iloc[-3]
        pattern = identify_candlestick_pattern(last, prev, atr_val)
        conf = float(pattern.get("reliability", 0.0))
        if conf < CANDLE_AGENT_MIN_REL:
            return {"agent": "manual_pattern", "direction": "NEUTRAL", "confidence": conf, "reason": "low_rel"}
        return {
            "agent": "manual_pattern",
            "direction": pattern.get("direction", "NEUTRAL"),
            "confidence": conf,
            "reason": pattern.get("name", "pattern")
        }
    except Exception as e:
        return {"agent": "manual_pattern", "direction": "NEUTRAL", "confidence": 0.0, "reason": f"error:{e}"}


def _agent_from_last3(df_m1: pd.DataFrame) -> Dict[str, Any]:
    try:
        if df_m1 is None or len(df_m1) < 4:
            return {"agent": "last3_momentum", "direction": "NEUTRAL", "confidence": 0.0, "reason": "insufficient_data"}
        tail = df_m1.tail(3)
        ups = sum(1 for _, row in tail.iterrows() if row["close"] > row["open"])
        downs = 3 - ups
        last = tail.iloc[-1]
        last_dir = "CALL" if last["close"] > last["open"] else "PUT" if last["close"] < last["open"] else "NEUTRAL"
        if ups >= 2 and last_dir == "CALL":
            return {"agent": "last3_momentum", "direction": "CALL", "confidence": ups / 3.0, "reason": "bullish_3"}
        if downs >= 2 and last_dir == "PUT":
            return {"agent": "last3_momentum", "direction": "PUT", "confidence": downs / 3.0, "reason": "bearish_3"}
        return {"agent": "last3_momentum", "direction": "NEUTRAL", "confidence": max(ups, downs) / 3.0, "reason": "no_clear"}
    except Exception as e:
        return {"agent": "last3_momentum", "direction": "NEUTRAL", "confidence": 0.0, "reason": f"error:{e}"}


def candle_agents_consensus(direction: str, df_m1: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    agents = [
        _agent_from_analyze_patterns(df_m1, direction, atr_val),
        _agent_from_talib_direct(df_m1, atr_val),
        _agent_from_manual_pattern(df_m1, atr_val),
        _agent_from_last3(df_m1)
    ]

    mismatches = []
    for agent in agents:
        if agent.get("direction") != direction:
            mismatches.append(f"{agent.get('agent')}={agent.get('direction')}")

    allow = len(mismatches) == 0
    reason = "OK" if allow else "divergent:" + ",".join(mismatches)
    return {"allow": allow, "agents": agents, "reason": reason}


def smart_entry_decision(
    direction: str,
    setup: Dict[str, Any],
    momentum_check: Dict[str, Any],
    trend_check: Dict[str, Any],
    projection_check: Dict[str, Any],
    chart_analysis: Dict[str, Any],
    ai_prediction: Dict[str, Any] = None,
    df_m1: pd.DataFrame = None,
    atr_val: float = 0.0,
    ai_stats: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    LOGICA SIMPLIFICADA: SÓ ENTRA SE DETECTAR PADRÃO 80%+!
    
    PADRÕES 80%+:
    - Engolfo (85%)
    - 3 Soldados/Corvos (88%)
    - Morning/Evening Star (85%)
    - Hammer/Shooting Star (82%)
    - Marubozu (80%)
    - Doji Star (80%)
    """
    result = {
        "allow_entry": False,
        "final_score": 0.0,
        "confidence": 0.0,
        "reasons": [],
        "warnings": [],
        "ia_analysis": "",
        "decision_reason": "",
        "market_context": None,
        "is_consolidating": False,
        "override_direction": None,
        "detected_patterns": [],
        "market_type": "INDEFINIDO",
        "best_setup": "NONE"
    }
    
    ativo = setup.get("ativo", "ATIVO")
    
    # ═══════════════════════════════════════════════════════════════
    # SE JÁ VEIO COM PADRÃO DETECTADO NO SETUP = ENTRA DIRETO!
    # ═══════════════════════════════════════════════════════════════
    
    if setup.get("padrao"):
        padrao_nome = setup.get("padrao", "PADRAO")
        conf = setup.get("confiabilidade", 0.80)
        direcao = setup.get("dir", direction)
        
        # Confirmação pela IA (prob/conf) quando há amostras suficientes
        if AI_CONFIRM_PATTERN and ai_prediction:
            n_arm = int(ai_prediction.get("n_arm", 0))
            prob = float(ai_prediction.get("prob", 0.0))
            conf_ai = float(ai_prediction.get("conf", 0.0))
            if n_arm >= AI_MIN_SAMPLES and (prob < AI_MIN_PROB or conf_ai < AI_CONF_MIN):
                logging.info(f"[IA CONFIRMA] Padrão {padrao_nome} {direcao} bloqueado: prob={prob:.2f} conf={conf_ai:.2f} n={n_arm}")
                result["decision_reason"] = f"IA_CONFIRMA_NEGATIVA:{padrao_nome}"
                return result
        
        # Bloqueio por estatística do padrão (opcional)
        if AI_BLOCK_PATTERN_STATS and ai_stats:
            is_blocked, reason = ai_should_block_candlestick(padrao_nome, direcao, ai_stats)
            if is_blocked:
                logging.info(f"[IA BLOQUEIO] Padrão {padrao_nome} {direcao} bloqueado: {reason}")
                result["decision_reason"] = f"BLOQUEIO_IA:{padrao_nome}"
                return result
        
        logging.info(f"[ENTRADA] {ativo} | {padrao_nome} {direcao} ({conf*100:.0f}%) - APROVADO!")
        
        result["allow_entry"] = True
        result["final_score"] = int(conf * 100)
        result["confidence"] = conf
        result["decision_reason"] = f"PADRAO:{padrao_nome}"
        result["reasons"] = [f"PADRAO_{padrao_nome}({conf*100:.0f}%)"]
        result["detected_patterns"] = [{"nome": padrao_nome, "confiabilidade": conf, "direcao": direcao}]
        result["best_setup"] = padrao_nome
        result["market_type"] = "PADRAO_80%+"
        result["override_direction"] = direcao if direcao != direction else None
        return result
    
    # ═══════════════════════════════════════════════════════════════
    # SE NÃO VEIO COM PADRÃO, ANALISA NOVAMENTE
    # ═══════════════════════════════════════════════════════════════
    
    if df_m1 is None or len(df_m1) < 15:
        result["decision_reason"] = "DADOS_INSUFICIENTES"
        return result
    
    atr_safe = max(atr_val, 1e-9)
    
    if not PATTERN_DETECTOR_AVAILABLE:
        result["decision_reason"] = "DETECTOR_INDISPONIVEL"
        return result
    
    try:
        result_call = analyze_patterns(df=df_m1, direcao_sinal="CALL", atr_val=atr_safe)
        result_put = analyze_patterns(df=df_m1, direcao_sinal="PUT", atr_val=atr_safe)
        
        padroes_candle_call = result_call.get("padroes_candlestick", [])
        padroes_candle_put = result_put.get("padroes_candlestick", [])
        
        # Buscar padrão CALL 80%+
        padrao_call = None
        for p in padroes_candle_call:
            if p.get("direcao") == "CALL" and p.get("confiabilidade", 0) >= 0.80:
                if not padrao_call or p.get("confiabilidade", 0) > padrao_call.get("confiabilidade", 0):
                    padrao_call = p
        
        # Buscar padrão PUT 80%+
        padrao_put = None
        for p in padroes_candle_put:
            if p.get("direcao") == "PUT" and p.get("confiabilidade", 0) >= 0.80:
                if not padrao_put or p.get("confiabilidade", 0) > padrao_put.get("confiabilidade", 0):
                    padrao_put = p
        
        # SE DETECTOU PADRÃO 80%+ = ENTRADA!
        
        if padrao_call:
            # Confirmação pela IA (prob/conf) quando há amostras suficientes
            if AI_CONFIRM_PATTERN and ai_prediction:
                n_arm = int(ai_prediction.get("n_arm", 0))
                prob = float(ai_prediction.get("prob", 0.0))
                conf_ai = float(ai_prediction.get("conf", 0.0))
                if n_arm >= AI_MIN_SAMPLES and (prob < AI_MIN_PROB or conf_ai < AI_CONF_MIN):
                    logging.info(f"[IA CONFIRMA] Padrão {padrao_call.get('nome','')} CALL bloqueado: prob={prob:.2f} conf={conf_ai:.2f} n={n_arm}")
                    padrao_call = None
            # Bloqueio por estatística do padrão (opcional)
            if padrao_call and AI_BLOCK_PATTERN_STATS and ai_stats:
                is_blocked, reason = ai_should_block_candlestick(padrao_call.get('nome', ''), "CALL", ai_stats)
                if is_blocked:
                    logging.info(f"[IA BLOQUEIO] Padrão {padrao_call['nome']} CALL bloqueado: {reason}")
                    padrao_call = None
            
            if padrao_call:
                conf = padrao_call.get("confiabilidade", 0.8)
                logging.info(f"[PADRAO 80%+] {ativo} | {padrao_call['nome']} CALL ({conf*100:.0f}%) - ENTRADA!")
                
                result["allow_entry"] = True
                result["final_score"] = int(conf * 100)
                result["confidence"] = conf
                result["decision_reason"] = f"PADRAO:{padrao_call['nome']}"
                result["reasons"] = [f"PADRAO_{padrao_call['nome']}({conf*100:.0f}%)"]
                result["detected_patterns"] = [padrao_call]
                result["best_setup"] = padrao_call['nome']
                result["market_type"] = "PADRAO_80%+"
                result["override_direction"] = "CALL" if direction != "CALL" else None
                return result
        
        if padrao_put:
            # Confirmação pela IA (prob/conf) quando há amostras suficientes
            if AI_CONFIRM_PATTERN and ai_prediction:
                n_arm = int(ai_prediction.get("n_arm", 0))
                prob = float(ai_prediction.get("prob", 0.0))
                conf_ai = float(ai_prediction.get("conf", 0.0))
                if n_arm >= AI_MIN_SAMPLES and (prob < AI_MIN_PROB or conf_ai < AI_CONF_MIN):
                    logging.info(f"[IA CONFIRMA] Padrão {padrao_put.get('nome','')} PUT bloqueado: prob={prob:.2f} conf={conf_ai:.2f} n={n_arm}")
                    padrao_put = None
            # Bloqueio por estatística do padrão (opcional)
            if padrao_put and AI_BLOCK_PATTERN_STATS and ai_stats:
                is_blocked, reason = ai_should_block_candlestick(padrao_put.get('nome', ''), "PUT", ai_stats)
                if is_blocked:
                    logging.info(f"[IA BLOQUEIO] Padrão {padrao_put['nome']} PUT bloqueado: {reason}")
                    padrao_put = None
            
            if padrao_put:
                conf = padrao_put.get("confiabilidade", 0.8)
                logging.info(f"[PADRAO 80%+] {ativo} | {padrao_put['nome']} PUT ({conf*100:.0f}%) - ENTRADA!")
                
                result["allow_entry"] = True
                result["final_score"] = int(conf * 100)
                result["confidence"] = conf
                result["decision_reason"] = f"PADRAO:{padrao_put['nome']}"
                result["reasons"] = [f"PADRAO_{padrao_put['nome']}({conf*100:.0f}%)"]
                result["detected_patterns"] = [padrao_put]
                result["best_setup"] = padrao_put['nome']
                result["market_type"] = "PADRAO_80%+"
                result["override_direction"] = "PUT" if direction != "PUT" else None
                return result
            
    except Exception as e:
        logging.error(f"[PATTERN] Erro: {e}")
    
    # Nenhum padrão 80%+ detectado = NÃO ENTRA
    result["decision_reason"] = "SEM_PADRAO"
    return result


# ===================== IA TRADER PROFISSIONAL - ANÁLISE GRÁFICA AVANÇADA =====================
def identify_candlestick_pattern(candle: pd.Series, prev_candle: pd.Series, atr_val: float) -> Dict[str, Any]:
    """
    Identifica padrões de candlestick como um trader profissional.
    Retorna o padrão identificado e sua força/direção.
    """
    o, h, l, c = float(candle["open"]), float(candle["high"]), float(candle["low"]), float(candle["close"])
    po, ph, pl, pc = float(prev_candle["open"]), float(prev_candle["high"]), float(prev_candle["low"]), float(prev_candle["close"])
    
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    candle_range = h - l
    
    prev_body = abs(pc - po)
    is_green = c > o
    prev_is_green = pc > po
    
    pattern = {"name": "none", "direction": "NEUTRAL", "strength": 0.0, "reliability": 0.0}
    
    # Evita divisão por zero
    if candle_range < atr_val * 0.01:
        return pattern
    
    body_ratio = body / candle_range
    upper_ratio = upper_wick / candle_range
    lower_ratio = lower_wick / candle_range
    
    # ===== PADRÕES DE REVERSÃO BULLISH (CALL) =====
    
    # 1. HAMMER (Martelo) - Forte sinal de reversão bullish
    if lower_ratio > 0.6 and body_ratio < 0.3 and upper_ratio < 0.1:
        pattern = {"name": "hammer", "direction": "CALL", "strength": 0.85, "reliability": 0.75}
    
    # 2. BULLISH ENGULFING (Engolfo de Alta)
    elif is_green and not prev_is_green and c > po and o < pc and body > prev_body * 1.2:
        pattern = {"name": "bullish_engulfing", "direction": "CALL", "strength": 0.90, "reliability": 0.80}
    
    # 3. MORNING STAR hint (última vela verde após doji)
    elif is_green and body_ratio > 0.5 and prev_body < atr_val * 0.3:
        pattern = {"name": "morning_star_hint", "direction": "CALL", "strength": 0.70, "reliability": 0.65}
    
    # 4. PIERCING LINE (Linha Perfurante)
    elif is_green and not prev_is_green and o < pl and c > (po + pc) / 2:
        pattern = {"name": "piercing_line", "direction": "CALL", "strength": 0.75, "reliability": 0.70}
    
    # 5. BULLISH PIN BAR
    elif lower_ratio > 0.65 and body_ratio < 0.25 and is_green:
        pattern = {"name": "bullish_pin_bar", "direction": "CALL", "strength": 0.88, "reliability": 0.78}
    
    # ===== PADRÕES DE REVERSÃO BEARISH (PUT) =====
    
    # 6. SHOOTING STAR (Estrela Cadente) - Forte sinal de reversão bearish
    elif upper_ratio > 0.6 and body_ratio < 0.3 and lower_ratio < 0.1:
        pattern = {"name": "shooting_star", "direction": "PUT", "strength": 0.85, "reliability": 0.75}
    
    # 7. BEARISH ENGULFING (Engolfo de Baixa)
    elif not is_green and prev_is_green and c < po and o > pc and body > prev_body * 1.2:
        pattern = {"name": "bearish_engulfing", "direction": "PUT", "strength": 0.90, "reliability": 0.80}
    
    # 8. EVENING STAR hint (última vela vermelha após doji)
    elif not is_green and body_ratio > 0.5 and prev_body < atr_val * 0.3:
        pattern = {"name": "evening_star_hint", "direction": "PUT", "strength": 0.70, "reliability": 0.65}
    
    # 9. DARK CLOUD COVER (Nuvem Negra)
    elif not is_green and prev_is_green and o > ph and c < (po + pc) / 2:
        pattern = {"name": "dark_cloud", "direction": "PUT", "strength": 0.75, "reliability": 0.70}
    
    # 10. BEARISH PIN BAR
    elif upper_ratio > 0.65 and body_ratio < 0.25 and not is_green:
        pattern = {"name": "bearish_pin_bar", "direction": "PUT", "strength": 0.88, "reliability": 0.78}
    
    # ===== PADRÕES DE CONTINUAÇÃO =====
    
    # 11. MARUBOZU BULLISH (vela cheia sem sombras = força compradora)
    elif is_green and body_ratio > 0.85:
        pattern = {"name": "marubozu_bullish", "direction": "CALL", "strength": 0.80, "reliability": 0.72}
    
    # 12. MARUBOZU BEARISH (vela cheia sem sombras = força vendedora)
    elif not is_green and body_ratio > 0.85:
        pattern = {"name": "marubozu_bearish", "direction": "PUT", "strength": 0.80, "reliability": 0.72}
    
    # ===== PADRÕES DE INDECISÃO =====
    
    # 13. DOJI (indecisão - esperar próxima vela)
    elif body_ratio < 0.1:
        pattern = {"name": "doji", "direction": "NEUTRAL", "strength": 0.0, "reliability": 0.50}
    
    # 14. SPINNING TOP (indecisão)
    elif body_ratio < 0.3 and upper_ratio > 0.3 and lower_ratio > 0.3:
        pattern = {"name": "spinning_top", "direction": "NEUTRAL", "strength": 0.0, "reliability": 0.45}
    
    return pattern


def professional_chart_analysis(df_m1: pd.DataFrame, direction: str, atr_val: float, exp_minutes: int = 5) -> Dict[str, Any]:
    """
    🧠 IA TRADER PROFISSIONAL - Análise Gráfica Completa
    
    Analisa como um trader experiente:
    1. Padrões de candlestick
    2. Estrutura de mercado (topos/fundos)
    3. Força e momentum
    4. Zonas de suporte/resistência
    5. Probabilidade de WIN em 1-5 minutos
    
    Retorna:
    - valid: Se a entrada é recomendada
    - win_probability: Probabilidade estimada de WIN (0-100%)
    - confidence: Confiança na análise
    - reasons: Lista de motivos
    """
    result = {
        "valid": True,
        "win_probability": 50.0,
        "confidence": 0.0,
        "pattern": None,
        "structure": None,
        "reasons": [],
        "warnings": [],
        "recommendation": "NEUTRAL"
    }
    
    if len(df_m1) < 30:
        result["reasons"].append("dados_insuficientes")
        return result
    
    try:
        # 1. ANÁLISE DE CANDLESTICK (última e penúltima vela)
        last_candle = df_m1.iloc[-1]
        prev_candle = df_m1.iloc[-2]
        pattern = identify_candlestick_pattern(last_candle, prev_candle, atr_val)
        result["pattern"] = pattern
        
        # 2. ANÁLISE DE ESTRUTURA DE MERCADO
        structure = analyze_market_structure(df_m1, atr_val)
        result["structure"] = structure
        
        # 3. CALCULAR PROBABILIDADE BASE
        win_prob = 50.0  # Base neutra
        confidence = 0.5
        
        # ===== AJUSTES BASEADOS EM PADRÃO DE CANDLESTICK =====
        if pattern["name"] != "none":
            pattern_dir = pattern["direction"]
            pattern_strength = pattern["strength"]
            pattern_reliability = pattern["reliability"]
            
            if pattern_dir == direction:
                # Padrão confirma a direção - AUMENTA probabilidade
                win_prob += pattern_strength * 20
                confidence += pattern_reliability * 0.2
                result["reasons"].append(f"padrao_{pattern['name']}_favoravel")
            elif pattern_dir != "NEUTRAL" and pattern_dir != direction:
                # Padrão CONTRA a direção - REDUZ probabilidade
                win_prob -= pattern_strength * 25
                confidence += pattern_reliability * 0.15
                result["warnings"].append(f"padrao_{pattern['name']}_contrario")
            else:
                # Padrão neutro (doji, spinning top)
                win_prob -= 5
                result["warnings"].append("padrao_indecisao")
        
        # ===== AJUSTES BASEADOS EM ESTRUTURA DE MERCADO =====
        struct_bias = structure["bias"]
        struct_strength = structure["strength"]
        
        if struct_bias == direction:
            # Estrutura confirma direção
            win_prob += struct_strength * 15
            confidence += 0.15
            result["reasons"].append(f"estrutura_{structure['structure']}_favoravel")
        elif struct_bias != "NEUTRAL" and struct_bias != direction:
            # Estrutura contra direção
            win_prob -= struct_strength * 20
            confidence += 0.1
            result["warnings"].append(f"estrutura_contra_{structure['structure']}")
        
        # ===== AJUSTES BASEADOS EM MOMENTUM =====
        momentum = structure["momentum"]
        if direction == "CALL" and momentum > 0.5:
            win_prob += 8
            result["reasons"].append("momentum_bullish_forte")
        elif direction == "CALL" and momentum < -0.5:
            win_prob -= 12
            result["warnings"].append("momentum_bearish_contra")
        elif direction == "PUT" and momentum < -0.5:
            win_prob += 8
            result["reasons"].append("momentum_bearish_forte")
        elif direction == "PUT" and momentum > 0.5:
            win_prob -= 12
            result["warnings"].append("momentum_bullish_contra")
        
        # ===== AJUSTES BASEADOS EM POSIÇÃO NO RANGE =====
        if structure["near_top"]:
            if direction == "CALL":
                win_prob -= 15
                result["warnings"].append("proximo_resistencia")
            else:  # PUT
                win_prob += 10
                result["reasons"].append("reversao_no_topo")
        
        if structure["near_bottom"]:
            if direction == "PUT":
                win_prob -= 15
                result["warnings"].append("proximo_suporte")
            else:  # CALL
                win_prob += 10
                result["reasons"].append("reversao_no_fundo")
        
        # ===== ANÁLISE DE SEQUÊNCIA DE VELAS (últimas 5) =====
        last_5 = df_m1.tail(5)
        green_count = (last_5["close"] > last_5["open"]).sum()
        red_count = 5 - green_count
        
        # Detectar exaustão (muitas velas na mesma direção)
        if green_count >= 4 and direction == "CALL":
            win_prob -= 10
            result["warnings"].append("possivel_exaustao_alta")
        elif red_count >= 4 and direction == "PUT":
            win_prob -= 10
            result["warnings"].append("possivel_exaustao_baixa")
        
        # Confirmação de força
        if green_count >= 3 and direction == "CALL":
            win_prob += 5
            result["reasons"].append("sequencia_bullish")
        elif red_count >= 3 and direction == "PUT":
            win_prob += 5
            result["reasons"].append("sequencia_bearish")
        
        # ===== ANÁLISE DO TAMANHO DAS VELAS (força do movimento) =====
        bodies = abs(df_m1["close"].tail(5) - df_m1["open"].tail(5))
        avg_body = bodies.mean()
        last_body = abs(last_candle["close"] - last_candle["open"])
        
        if last_body > avg_body * 1.5:
            # Vela grande = movimento forte
            is_green = last_candle["close"] > last_candle["open"]
            if (is_green and direction == "CALL") or (not is_green and direction == "PUT"):
                win_prob += 8
                result["reasons"].append("vela_forte_favoravel")
            else:
                win_prob -= 10
                result["warnings"].append("vela_forte_contraria")
        
        # ===== AJUSTE PARA TIMEFRAME 1-5 MIN =====
        # Em timeframes curtos, reversões são mais prováveis
        if exp_minutes <= 2:
            # Reduz confiança para expiração muito curta
            confidence *= 0.9
            if win_prob > 65:
                win_prob = min(win_prob, 70)  # Cap para não ser overconfident
        
        # ===== DECISÃO FINAL =====
        win_prob = max(25, min(85, win_prob))  # Limita entre 25% e 85% (RELAXADO de 15%)
        confidence = max(0.3, min(0.95, confidence))
        
        # Recomendação
        if win_prob >= 65:
            result["recommendation"] = "FORTE_" + direction
        elif win_prob >= 55:
            result["recommendation"] = "MODERADO_" + direction
        elif win_prob >= 45:
            result["recommendation"] = "NEUTRO"
        elif win_prob >= 35:
            result["recommendation"] = "EVITAR"
        else:
            result["recommendation"] = "NAO_OPERAR"
        
        # Validação final - RELAXADA
        # Só invalida se probabilidade MUITO baixa (< 35%, antes era < 45%)
        if win_prob < 35:
            result["valid"] = False
            result["reasons"].append(f"probabilidade_baixa_{win_prob:.0f}pct")
        
        # Só invalida se MUITOS alertas E probabilidade baixa
        if len(result["warnings"]) >= 4 and win_prob < 45:  # RELAXADO de 3 para 4 warnings
            result["valid"] = False
            result["reasons"].append("muitos_alertas")
        
        result["win_probability"] = win_prob
        result["confidence"] = confidence
        
        # ===== ANÁLISE DE EXTENSÃO E PULLBACK (PARA DECISÃO INTELIGENTE) =====
        closes = df_m1["close"].tail(30).to_numpy(float)
        highs = df_m1["high"].tail(30).to_numpy(float)
        lows = df_m1["low"].tail(30).to_numpy(float)
        
        # Calcular movimento recente em ATRs (últimos 10 candles)
        recent_high = float(np.max(highs[-10:]))
        recent_low = float(np.min(lows[-10:]))
        current_price = closes[-1]
        
        # Movimento total recente
        if current_price > closes[-10]:
            # Movimento de alta
            recent_move_atr = (current_price - recent_low) / max(atr_val, 1e-9)
        else:
            # Movimento de baixa
            recent_move_atr = -((recent_high - current_price) / max(atr_val, 1e-9))
        
        result["recent_move_atr"] = recent_move_atr
        
        # Detectar se está em extremo (suporte ou resistência baseado em topos/fundos)
        range_30 = recent_high - recent_low
        position_in_range = (current_price - recent_low) / max(range_30, 1e-9)
        
        is_at_extreme = False
        extreme_type = "none"
        
        # Próximo do fundo (< 15% do range) = suporte
        if position_in_range < 0.15:
            is_at_extreme = True
            extreme_type = "support"
        # Próximo do topo (> 85% do range) = resistência
        elif position_in_range > 0.85:
            is_at_extreme = True
            extreme_type = "resistance"
        
        result["is_at_extreme"] = is_at_extreme
        result["extreme_type"] = extreme_type
        result["position_in_range"] = position_in_range
        
        # Qualidade do pullback
        # Bom pullback: retração de 38-62% do movimento anterior
        if len(closes) >= 20:
            swing_high = float(np.max(highs[-20:-5]))
            swing_low = float(np.min(lows[-20:-5]))
            swing_range = swing_high - swing_low
            
            if swing_range > atr_val * 0.5:  # Movimento significativo
                if direction == "CALL":
                    # Para CALL, queremos pullback de alta (preço retraiu e agora sobe)
                    retrace = (swing_high - current_price) / swing_range
                    if 0.30 <= retrace <= 0.70:
                        pullback_quality = 1.0 - abs(retrace - 0.50)  # Melhor em 50%
                    else:
                        pullback_quality = max(0, 0.5 - abs(retrace - 0.50))
                else:  # PUT
                    # Para PUT, queremos pullback de baixa (preço subiu e agora cai)
                    retrace = (current_price - swing_low) / swing_range
                    if 0.30 <= retrace <= 0.70:
                        pullback_quality = 1.0 - abs(retrace - 0.50)
                    else:
                        pullback_quality = max(0, 0.5 - abs(retrace - 0.50))
            else:
                pullback_quality = 0.5  # Movimento pequeno, neutro
        else:
            pullback_quality = 0.5
        
        result["pullback_quality"] = pullback_quality
        
    except Exception as e:
        result["reasons"].append(f"erro_analise: {str(e)[:30]}")
    
    return result


def validate_immediate_trend(df_m1: pd.DataFrame, direction: str, atr_val: float) -> Dict[str, Any]:
    """
    NOVO: Verifica se a tendência imediata (últimos 10 candles) suporta a direção.
    Análise de EMA rápida vs preço + contagem de velas.
    """
    if len(df_m1) < 15:
        return {"valid": True, "reason": "dados_insuficientes", "trend_aligned": False, 
                "trend_direction": "NEUTRAL", "trend_strength": 0.0}
    
    closes = df_m1["close"].tail(15).to_numpy(float)
    opens = df_m1["open"].tail(15).to_numpy(float)
    
    # Contar velas verdes e vermelhas (últimas 10)
    green_count = sum(1 for i in range(-10, 0) if closes[i] > opens[i])
    red_count = 10 - green_count
    
    # Força da tendência baseada na contagem
    trend_strength = max(green_count, red_count) / 10.0
    trend_direction = "BULLISH" if green_count > red_count else "BEARISH" if red_count > green_count else "NEUTRAL"
    
    # EMA de 5 períodos (muito rápida)
    ema_5 = closes.copy()
    multiplier = 2 / (5 + 1)
    for i in range(1, len(ema_5)):
        ema_5[i] = (closes[i] * multiplier) + (ema_5[i-1] * (1 - multiplier))
    
    # Preço atual vs EMA5
    current_price = closes[-1]
    current_ema = ema_5[-1]
    
    # Direção da EMA (subindo ou descendo?)
    ema_slope = ema_5[-1] - ema_5[-5]
    ema_slope_pct = ema_slope / max(atr_val, 1e-9)
    
    base_result = {
        "trend_direction": trend_direction,
        "trend_strength": trend_strength,
        "ema_slope": ema_slope_pct,
        "green_count": green_count,
        "red_count": red_count
    }
    
    if direction == "CALL":
        # Para CALL: preço acima da EMA5 e EMA subindo
        price_above_ema = current_price > current_ema
        ema_rising = ema_slope_pct > 0.03  # RELAXADO de 0.05
        
        if price_above_ema and ema_rising:
            return {**base_result, "valid": True, "reason": "tendencia_alta", "trend_aligned": True}
        elif price_above_ema or ema_rising:
            return {**base_result, "valid": True, "reason": "tendencia_neutra", "trend_aligned": False}
        else:
            # Preço abaixo da EMA e EMA caindo = CONTRA a direção
            # RELAXADO: só bloqueia se tendência MUITO forte contra (de -0.15 para -0.35)
            if ema_slope_pct < -0.35:  # EMA caindo MUITO forte
                return {**base_result, "valid": False, "reason": "contra_tendencia_forte", "trend_aligned": False}
            return {**base_result, "valid": True, "reason": "tendencia_incerta", "trend_aligned": False}
    
    else:  # PUT
        # Para PUT: preço abaixo da EMA5 e EMA caindo
        price_below_ema = current_price < current_ema
        ema_falling = ema_slope_pct < -0.03  # RELAXADO de -0.05
        
        if price_below_ema and ema_falling:
            return {**base_result, "valid": True, "reason": "tendencia_baixa", "trend_aligned": True}
        elif price_below_ema or ema_falling:
            return {**base_result, "valid": True, "reason": "tendencia_neutra", "trend_aligned": False}
        else:
            # Preço acima da EMA e EMA subindo = CONTRA a direção
            # RELAXADO: só bloqueia se tendência MUITO forte contra (de 0.15 para 0.35)
            if ema_slope_pct > 0.35:  # EMA subindo MUITO forte
                return {**base_result, "valid": False, "reason": "contra_tendencia_forte", "trend_aligned": False}
            return {**base_result, "valid": True, "reason": "tendencia_incerta", "trend_aligned": False}


# ===================== PROJEÇÃO INTELIGENTE DE EXPIRAÇÃO =====================
def smart_expiration_projection(df_m1: pd.DataFrame, direction: str, atr_val: float) -> Dict[str, Any]:
    """
    🧠 IA INTELIGENTE - Projeta o melhor tempo de expiração (1, 2, 3 ou 5 minutos)
    
    PRIORIZA TEMPO MAIS CURTO quando win rates são iguais ou próximos!
    
    Analisa onde o preço estará em cada intervalo e escolhe a expiração com maior
    probabilidade de WIN. Se empatar, escolhe o mais curto.
    
    Retorna:
    - best_expiration: Melhor tempo de expiração (1, 2, 3 ou 5)
    - projections: Dicionário com projeção para cada tempo
    - confidence: Confiança na projeção
    - reason: Motivo da escolha
    """
    result = {
        "best_expiration": 1,  # Default = mais curto
        "projections": {},
        "confidence": 0.5,
        "reason": "default",
        "projected_win": {}
    }
    
    if len(df_m1) < 30:
        return result
    
    try:
        closes = df_m1["close"].tail(30).to_numpy(float)
        highs = df_m1["high"].tail(30).to_numpy(float)
        lows = df_m1["low"].tail(30).to_numpy(float)
        current_price = closes[-1]
        
        # Volatilidade média (range das últimas 10 velas)
        ranges = highs[-10:] - lows[-10:]
        avg_range = float(np.mean(ranges))
        
        # Regressão Linear para projetar tendência
        x = np.arange(20)
        y = closes[-20:]
        slope, intercept = np.polyfit(x, y, 1)
        
        # Velocidade do movimento recente (últimos 5 candles)
        recent_move = closes[-1] - closes[-5]
        move_per_candle = recent_move / 5
        
        # Calcular força do momentum (sem usar RSI, apenas movimento)
        momentum_strength = abs(move_per_candle) / max(atr_val, 1e-9)
        
        # Analisar suportes e resistências próximos
        recent_high = float(np.max(highs[-20:]))
        recent_low = float(np.min(lows[-20:]))
        
        # Distância até suporte/resistência
        dist_to_resistance = recent_high - current_price
        dist_to_support = current_price - recent_low
        
        # Projetar para cada expiração (1, 2, 3, 5 minutos)
        # IMPORTANTE: Ordem do mais curto para o mais longo
        expirations = [1, 2, 3, 5]
        best_exp = 1  # Começa com o mais curto
        best_score = -999
        best_win_prob = 0
        
        # Armazenar scores e win probs para comparar
        scores_by_exp = {}
        
        for exp in expirations:
            # Projeção baseada em tendência linear
            projected_price = intercept + slope * (20 + exp)
            
            # Projeção baseada em momentum recente
            momentum_projection = current_price + (move_per_candle * exp)
            
            # Média ponderada das projeções (60% tendência, 40% momentum)
            final_projection = projected_price * 0.6 + momentum_projection * 0.4
            
            # Movimento esperado
            expected_move = final_projection - current_price
            expected_move_pct = (expected_move / current_price) * 100
            
            # Verificar se projeção está na direção correta
            if direction == "CALL":
                is_favorable = expected_move > 0
                # Para CALL, verificar se tem espaço até resistência
                space_available = dist_to_resistance / max(atr_val, 1e-9)
            else:  # PUT
                is_favorable = expected_move < 0
                # Para PUT, verificar se tem espaço até suporte
                space_available = dist_to_support / max(atr_val, 1e-9)
            
            # Calcular pontuação para esta expiração
            score = 0
            
            # 1. Direção favorável (+40 pontos)
            if is_favorable:
                score += 40
            
            # 2. Magnitude do movimento esperado (+30 pontos máx)
            move_magnitude = abs(expected_move_pct)
            score += min(30, move_magnitude * 10)
            
            # 3. Espaço disponível até S/R (+20 pontos máx)
            if space_available > 1.0:
                score += 20
            elif space_available > 0.5:
                score += 10
            else:
                score -= 10  # Penalidade se S/R muito próximo
            
            # 4. Consistência com momentum (+10 pontos)
            if momentum_strength > 0.3:
                if (direction == "CALL" and move_per_candle > 0) or (direction == "PUT" and move_per_candle < 0):
                    score += 10
            
            # 5. BÔNUS para tempo mais curto (menos tempo = menos risco)
            # Quanto mais curto, mais bônus
            time_bonus = (6 - exp) * 3  # 1min=+15, 2min=+12, 3min=+9, 5min=+3
            score += time_bonus
            
            # Calcular probabilidade de WIN
            win_probability = min(85, max(30, 50 + score * 0.35))
            
            result["projections"][exp] = {
                "projected_price": round(final_projection, 6),
                "expected_move_pct": round(expected_move_pct, 4),
                "is_favorable": is_favorable,
                "score": score,
                "win_probability": round(win_probability, 1)
            }
            result["projected_win"][exp] = round(win_probability, 1)
            scores_by_exp[exp] = {"score": score, "win_prob": win_probability}
            
            # Atualizar melhor expiração
            # REGRA: Se win_prob igual ou muito próximo (diferença < 3%), escolhe o mais curto
            if score > best_score:
                # Verifica se é significativamente melhor (> 5 pontos)
                if score > best_score + 5:
                    best_score = score
                    best_exp = exp
                    best_win_prob = win_probability
                else:
                    # Scores próximos - mantém o mais curto
                    if exp < best_exp:
                        best_score = score
                        best_exp = exp
                        best_win_prob = win_probability
        
        # DECISÃO FINAL: Escolher melhor expiração baseado em win rate e volatilidade
        win_1m = result["projected_win"].get(1, 0)
        win_2m = result["projected_win"].get(2, 0)
        win_3m = result["projected_win"].get(3, 0)
        win_5m = result["projected_win"].get(5, 0)
        
        # Calcular volatilidade para decidir expiração
        volatility = np.std(closes[-10:]) / np.mean(closes[-10:]) * 100
        
        # REGRA DE EXPIRAÇÃO: SEMPRE 1 MINUTO
        # Motivo: M1 é o timeframe de análise, então entrada em 1 min é mais precisa
        # A IA já analisa a vela fechada, então o movimento esperado é na próxima vela
        
        best_exp = 1
        best_win_prob = win_1m
        
        result["best_expiration"] = best_exp
        result["volatility"] = volatility
        result["confidence"] = min(0.9, 0.4 + (best_score / 100) * 0.5)
        
        # Razão da escolha
        best_proj = result["projections"][best_exp]
        result["reason"] = f"exp_{best_exp}min(win={best_proj['win_probability']:.0f}%_score={best_proj['score']:.0f})"
        
    except Exception as e:
        result["reason"] = f"erro: {str(e)[:30]}"
    
    return result


# ===================== ANÁLISE DE PROJEÇÃO DE TENDÊNCIA =====================
def analyze_trend_projection(df_m1: pd.DataFrame, direction: str, atr_val: float, exp_minutes: int = 5) -> Dict[str, Any]:
    """
    IA de Projeção de Tendência
    Analisa se a tendência atual projeta para ganhar nos próximos X minutos.
    Usa múltiplos indicadores:
    - Regressão linear para projetar preço
    - Força do momentum
    - Volatilidade esperada
    - Probabilidade de reversão
    """
    result = {
        "valid": True,
        "projection_aligned": False,
        "projected_direction": "NEUTRO",
        "confidence": 0.0,
        "expected_move": 0.0,
        "reversal_risk": 0.0,
        "reason": ""
    }
    
    if len(df_m1) < 30:
        result["reason"] = "dados_insuficientes"
        return result
    
    try:
        closes = df_m1["close"].tail(30).to_numpy(float)
        highs = df_m1["high"].tail(30).to_numpy(float)
        lows = df_m1["low"].tail(30).to_numpy(float)
        
        # 1. Regressão Linear nos últimos 15 candles para projetar tendência
        x = np.arange(15)
        y = closes[-15:]
        slope, intercept = np.polyfit(x, y, 1)
        
        # Projeta o preço para os próximos exp_minutes
        projected_price = intercept + slope * (15 + exp_minutes)
        current_price = closes[-1]
        expected_move = (projected_price - current_price) / current_price * 100
        
        # 2. Direção projetada
        if expected_move > 0.01:
            projected_direction = "CALL"
        elif expected_move < -0.01:
            projected_direction = "PUT"
        else:
            projected_direction = "NEUTRO"
        
        # 3. Força do Momentum (RSI simplificado)
        gains = []
        losses = []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i-1]
            if diff > 0:
                gains.append(diff)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(diff))
        
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # 4. Volatilidade e risco de reversão
        volatility = np.std(closes[-10:]) / np.mean(closes[-10:]) * 100
        
        # Verifica se está em zona de sobrecompra/sobrevenda
        reversal_risk = 0.0
        if rsi > 70:
            reversal_risk = (rsi - 70) / 30  # Risco de reversão para baixo
        elif rsi < 30:
            reversal_risk = (30 - rsi) / 30  # Risco de reversão para cima
        
        # 5. Verifica sequência de velas na mesma direção (exaustão)
        last_5_dirs = [1 if closes[i] > closes[i-1] else -1 for i in range(-5, 0)]
        same_direction = abs(sum(last_5_dirs))
        if same_direction >= 4:
            reversal_risk += 0.2  # Aumenta risco se 4+ velas na mesma direção
        
        # 6. Calcula confiança
        confidence = 0.5
        
        # Ajusta confiança baseado na clareza da tendência
        trend_clarity = abs(slope) / (atr_val / 15)  # Slope normalizado pelo ATR
        confidence += min(0.3, trend_clarity * 0.15)
        
        # Reduz confiança se volatilidade alta
        if volatility > 0.5:
            confidence -= 0.1
        
        # Reduz confiança se risco de reversão alto
        confidence -= reversal_risk * 0.2
        
        confidence = max(0.0, min(1.0, confidence))
        
        # 7. Verifica alinhamento com direção do sinal
        projection_aligned = projected_direction == direction.upper()
        
        # 8. Validação final
        if reversal_risk > 0.5:
            if (direction.upper() == "CALL" and rsi > 70) or (direction.upper() == "PUT" and rsi < 30):
                result["valid"] = False
                result["reason"] = f"reversao_iminente_rsi={rsi:.1f}"
        
        if not projection_aligned and confidence > 0.6:
            # Projeção clara contra a direção
            result["valid"] = False
            result["reason"] = f"projecao_contraria_{projected_direction}"
        
        result.update({
            "projection_aligned": projection_aligned,
            "projected_direction": projected_direction,
            "confidence": confidence,
            "expected_move": expected_move,
            "reversal_risk": reversal_risk,
            "rsi": rsi,
            "volatility": volatility,
            "slope": slope
        })
        
        if result["valid"]:
            result["reason"] = f"proj={projected_direction}_conf={confidence:.2f}_rsi={rsi:.1f}"
        
    except Exception as e:
        result["reason"] = f"erro_projecao: {str(e)[:30]}"
    
    return result


# ===================== RSI SWING REJECTION - ESTRATÉGIA DE BAIXO LOSS =====================
def rsi_swing_rejection(df_m1: pd.DataFrame, direction: str) -> Dict[str, Any]:
    """
    RSI Swing Rejection - Estratégia profissional de baixo loss.
    
    Usada por traders profissionais para reduzir losses.
    Só entra quando RSI confirma com padrão de rejeição.
    
    BULLISH Swing Rejection (para CALL):
    1. RSI cai abaixo de 30 (oversold)
    2. RSI volta acima de 30
    3. RSI faz outro dip MAS não volta abaixo de 30
    4. RSI rompe o topo anterior = SINAL DE COMPRA
    
    BEARISH Swing Rejection (para PUT):
    1. RSI sobe acima de 70 (overbought)
    2. RSI volta abaixo de 70
    3. RSI faz outro pico MAS não volta acima de 70
    4. RSI rompe o fundo anterior = SINAL DE VENDA
    
    Returns:
        Dict com valid, reason, rsi_signal, confidence
    """
    result = {
        "valid": False,  # Por padrão não válido - precisa confirmar
        "rsi_signal": "NEUTRAL",
        "confidence": 0.0,
        "rsi_current": 50.0,
        "rsi_previous": 50.0,
        "swing_detected": False,
        "reason": ""
    }
    
    if len(df_m1) < 20:
        result["reason"] = "dados_insuficientes"
        return result
    
    try:
        # Calcular RSI dos últimos 20 candles
        closes = df_m1["close"].tail(20).to_numpy(float)
        
        # Calcular RSI para cada candle
        rsi_values = []
        gains_sum = 0
        losses_sum = 0
        period = 14
        
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i-1]
            gain = diff if diff > 0 else 0
            loss = abs(diff) if diff < 0 else 0
            
            if i < period:
                gains_sum += gain
                losses_sum += loss
                if i == period - 1:
                    avg_gain = gains_sum / period
                    avg_loss = losses_sum / period
            else:
                avg_gain = (avg_gain * 13 + gain) / 14
                avg_loss = (avg_loss * 13 + loss) / 14
            
            if i >= period - 1:
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                rsi_values.append(rsi)
        
        if len(rsi_values) < 5:
            result["reason"] = "rsi_insuficiente"
            return result
        
        rsi_current = rsi_values[-1]
        rsi_prev = rsi_values[-2]
        rsi_prev2 = rsi_values[-3]
        rsi_prev3 = rsi_values[-4] if len(rsi_values) >= 4 else rsi_prev2
        
        result["rsi_current"] = round(rsi_current, 1)
        result["rsi_previous"] = round(rsi_prev, 1)
        
        # ===== BULLISH SWING REJECTION (para CALL) =====
        if direction == "CALL":
            # Verificar padrão de swing rejection bullish
            # 1. RSI esteve abaixo de 30 recentemente
            was_oversold = any(r < 30 for r in rsi_values[-6:-2])
            
            # 2. RSI voltou acima de 30
            recovered = rsi_prev2 > 30
            
            # 3. RSI fez dip mas não voltou abaixo de 30
            higher_low = rsi_prev < rsi_prev2 and rsi_prev > 30
            
            # 4. RSI atual quebrando pra cima
            breaking_up = rsi_current > rsi_prev
            
            # Padrão clássico de swing rejection
            if was_oversold and recovered and higher_low and breaking_up:
                result["valid"] = True
                result["rsi_signal"] = "BULLISH_SWING_REJECTION"
                result["swing_detected"] = True
                result["confidence"] = 0.8
                result["reason"] = f"rsi_swing_bullish({rsi_current:.1f})"
            
            # Padrão simplificado: RSI saindo de oversold com momentum
            elif rsi_prev < 35 and rsi_current > rsi_prev and rsi_current - rsi_prev > 3:
                result["valid"] = True
                result["rsi_signal"] = "BULLISH_MOMENTUM"
                result["confidence"] = 0.6
                result["reason"] = f"rsi_bullish_momentum({rsi_current:.1f})"
            
            # RSI neutro mas não contra
            elif 40 <= rsi_current <= 60:
                result["valid"] = True  # Permite entrada, RSI neutro
                result["rsi_signal"] = "NEUTRAL"
                result["confidence"] = 0.4
                result["reason"] = f"rsi_neutro({rsi_current:.1f})"
            
            # RSI contra (overbought) - BLOQUEIA
            elif rsi_current > 70:
                result["valid"] = False
                result["rsi_signal"] = "OVERBOUGHT_BLOCK"
                result["reason"] = f"rsi_overbought_bloqueia_call({rsi_current:.1f})"
            
            # RSI subindo - OK
            elif rsi_current > rsi_prev:
                result["valid"] = True
                result["rsi_signal"] = "BULLISH"
                result["confidence"] = 0.5
                result["reason"] = f"rsi_subindo({rsi_current:.1f})"
            
            else:
                result["reason"] = f"rsi_sem_confirmacao({rsi_current:.1f})"
        
        # ===== BEARISH SWING REJECTION (para PUT) =====
        elif direction == "PUT":
            # Verificar padrão de swing rejection bearish
            # 1. RSI esteve acima de 70 recentemente
            was_overbought = any(r > 70 for r in rsi_values[-6:-2])
            
            # 2. RSI voltou abaixo de 70
            recovered = rsi_prev2 < 70
            
            # 3. RSI fez pico mas não voltou acima de 70
            lower_high = rsi_prev > rsi_prev2 and rsi_prev < 70
            
            # 4. RSI atual quebrando pra baixo
            breaking_down = rsi_current < rsi_prev
            
            # Padrão clássico de swing rejection
            if was_overbought and recovered and lower_high and breaking_down:
                result["valid"] = True
                result["rsi_signal"] = "BEARISH_SWING_REJECTION"
                result["swing_detected"] = True
                result["confidence"] = 0.8
                result["reason"] = f"rsi_swing_bearish({rsi_current:.1f})"
            
            # Padrão simplificado: RSI saindo de overbought com momentum
            elif rsi_prev > 65 and rsi_current < rsi_prev and rsi_prev - rsi_current > 3:
                result["valid"] = True
                result["rsi_signal"] = "BEARISH_MOMENTUM"
                result["confidence"] = 0.6
                result["reason"] = f"rsi_bearish_momentum({rsi_current:.1f})"
            
            # RSI neutro mas não contra
            elif 40 <= rsi_current <= 60:
                result["valid"] = True  # Permite entrada, RSI neutro
                result["rsi_signal"] = "NEUTRAL"
                result["confidence"] = 0.4
                result["reason"] = f"rsi_neutro({rsi_current:.1f})"
            
            # RSI contra (oversold) - BLOQUEIA
            elif rsi_current < 30:
                result["valid"] = False
                result["rsi_signal"] = "OVERSOLD_BLOCK"
                result["reason"] = f"rsi_oversold_bloqueia_put({rsi_current:.1f})"
            
            # RSI descendo - OK
            elif rsi_current < rsi_prev:
                result["valid"] = True
                result["rsi_signal"] = "BEARISH"
                result["confidence"] = 0.5
                result["reason"] = f"rsi_descendo({rsi_current:.1f})"
            
            else:
                result["reason"] = f"rsi_sem_confirmacao({rsi_current:.1f})"
        
    except Exception as e:
        result["reason"] = f"erro_rsi: {str(e)[:30]}"
    
    return result


# ===================== FIREBASE - ENVIO DE LOSS PARA ANÁLISE =====================
def _sanitize_for_json(obj):
    """
    Converte tipos NumPy para tipos Python nativos para serialização JSON.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.bool_, np.integer)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif hasattr(obj, 'item'):  # Para qualquer outro tipo NumPy escalar
        return obj.item()
    else:
        return obj

def send_loss_to_firebase(
    iq: IQ_Option,
    order_id: int,
    ativo: str,
    direction: str,
    stake: float,
    setup: Dict[str, Any],
    momentum_data: Dict[str, Any] = None,
    trend_data: Dict[str, Any] = None,
    projection_data: Dict[str, Any] = None,
    ai_prediction: Dict[str, Any] = None,
    chart_analysis: Dict[str, Any] = None
) -> bool:
    """
    Envia dados de LOSS para o Firebase para análise futura.
    Captura contexto completo: setup, momentum, tendência, projeção, predição IA, análise gráfica.
    """
    try:
        # Captura candles recentes para análise
        candles_data = {}
        try:
            candles = iq.get_candles(ativo, 60, 50, time.time())
            if candles and not isinstance(candles, int):
                df = pd.DataFrame(candles)
                if not df.empty:
                    # Análise de mercado
                    recent = df.tail(20)
                    green_count = (recent['close'] > recent['open']).sum()
                    red_count = len(recent) - green_count
                    
                    # Tendência
                    trend = "bullish" if green_count > red_count else "bearish" if red_count > green_count else "neutral"
                    
                    candles_data = {
                        "count": len(df),
                        "last_10_closes": df['close'].tail(10).tolist(),
                        "last_10_opens": df['open'].tail(10).tolist(),
                        "trend": trend,
                        "green_candles": int(green_count),
                        "red_candles": int(red_count)
                    }
        except Exception as e:
            log.warning(f"Erro ao capturar candles para análise: {e}")
        
        # Prepara dados para enviar
        analysis_data = {
            "order_id": str(order_id),
            "timestamp": datetime.now().isoformat(),
            "asset": ativo,
            "direction": direction.upper(),
            "stake": float(stake),
            "market_context": {
                "trend": candles_data.get("trend", "unknown"),
                "green_candles": candles_data.get("green_candles", 0),
                "red_candles": candles_data.get("red_candles", 0),
            },
            "entry_quality": {
                "score": setup.get("score", 0),
                "reasons": setup.get("reasons", []),
                "perna_a_dir": setup.get("perna_a_dir", ""),
                "corpo_ratio": setup.get("corpo_ratio", 0)
            },
            "setup": {
                "perna_a": setup.get("perna_a", {}),
                "perna_b": setup.get("perna_b", {}),
                "score": setup.get("score", 0),
                "reasons": setup.get("reasons", []),
                "atr": setup.get("atr_val", 0)
            },
            "momentum_analysis": momentum_data or {},
            "trend_analysis": trend_data or {},
            "projection_analysis": projection_data or {},
            "ai_prediction": ai_prediction or {},
            "chart_analysis": chart_analysis or {},  # ANÁLISE GRÁFICA PROFISSIONAL
            "candles_data": candles_data,
            "ai_analysis": _generate_loss_analysis(
                ativo, direction, setup, momentum_data, trend_data, projection_data, candles_data, chart_analysis
            )
        }
        
        # Sanitiza dados para JSON (converte tipos NumPy para Python nativos)
        analysis_data = _sanitize_for_json(analysis_data)
        
        # Envia para o backend
        endpoint = f"{BACKEND_URL}/api/loss/analyze"
        response = requests.post(endpoint, json=analysis_data, timeout=10)
        
        if response.status_code == 200:
            log.info(paint(f"[FIREBASE] Loss enviado para analise: {ativo} {direction}", C.B))
            return True
        else:
            log.warning(f"[FIREBASE] Erro ao enviar loss: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        log.warning("[FIREBASE] Backend nao disponivel - loss nao enviado")
        return False
    except Exception as e:
        log.warning(f"[FIREBASE] Erro ao enviar loss: {e}")
        return False


# ===================== FIREBASE - ENVIO DE WIN PARA ANÁLISE =====================
def send_win_to_firebase(
    iq: IQ_Option,
    order_id: int,
    ativo: str,
    direction: str,
    stake: float,
    profit: float,
    setup: Dict[str, Any],
    momentum_data: Dict[str, Any] = None,
    trend_data: Dict[str, Any] = None,
    projection_data: Dict[str, Any] = None,
    ai_prediction: Dict[str, Any] = None,
    chart_analysis: Dict[str, Any] = None
) -> bool:
    """
    Envia dados de WIN para o Firebase para análise e aprendizado.
    Captura contexto completo: setup, momentum, tendência, projeção, predição IA, análise gráfica.
    """
    try:
        # Captura candles recentes para análise
        candles_data = {}
        try:
            candles = iq.get_candles(ativo, 60, 50, time.time())
            if candles and not isinstance(candles, int):
                df = pd.DataFrame(candles)
                if not df.empty:
                    # Análise de mercado
                    recent = df.tail(20)
                    green_count = (recent['close'] > recent['open']).sum()
                    red_count = len(recent) - green_count
                    
                    # Tendência
                    trend = "bullish" if green_count > red_count else "bearish" if red_count > green_count else "neutral"
                    
                    candles_data = {
                        "count": len(df),
                        "last_10_closes": df['close'].tail(10).tolist(),
                        "last_10_opens": df['open'].tail(10).tolist(),
                        "trend": trend,
                        "green_candles": int(green_count),
                        "red_candles": int(red_count)
                    }
        except Exception as e:
            log.warning(f"Erro ao capturar candles para análise WIN: {e}")
        
        # Prepara dados para enviar
        analysis_data = {
            "order_id": str(order_id),
            "timestamp": datetime.now().isoformat(),
            "result": "WIN",
            "asset": ativo,
            "direction": direction.upper(),
            "stake": float(stake),
            "profit": float(profit),
            "market_context": {
                "trend": candles_data.get("trend", "unknown"),
                "green_candles": candles_data.get("green_candles", 0),
                "red_candles": candles_data.get("red_candles", 0),
            },
            "entry_quality": {
                "score": setup.get("score", 0),
                "reasons": setup.get("reasons", []),
                "perna_a_dir": setup.get("perna_a_dir", ""),
                "corpo_ratio": setup.get("corpo_ratio", 0),
                "late_ext": setup.get("late_ext", 0),
                "context_score": setup.get("context_score", 0),
                "entry_momentum": setup.get("entry_momentum", 0),
                "entry_confidence": setup.get("entry_confidence", 0),
                "entry_alignment": setup.get("entry_alignment", 0)
            },
            "setup": {
                "score": setup.get("score", 0),
                "reasons": setup.get("reasons", []),
                "atr": setup.get("atr_val", 0),
                "pb_len": setup.get("pb_len", 0),
                "retr": setup.get("retr", 0),
                "effA": setup.get("effA", 0),
                "flips": setup.get("flips", 0),
                "comp": setup.get("comp", 0),
                "late": setup.get("late", 0),
                "distBreak": setup.get("distBreak", 0),
                "lt_confluence": setup.get("lt_confluence", 0)
            },
            "momentum_analysis": momentum_data or {},
            "trend_analysis": trend_data or {},
            "projection_analysis": projection_data or {},
            "ai_prediction": ai_prediction or {},
            "chart_analysis": chart_analysis or {},
            "candles_data": candles_data,
            "win_analysis": _generate_win_analysis(
                ativo, direction, profit, setup, momentum_data, trend_data, projection_data, candles_data, chart_analysis
            )
        }
        
        # Sanitiza dados para JSON (converte tipos NumPy para Python nativos)
        analysis_data = _sanitize_for_json(analysis_data)
        
        # Envia para o backend
        endpoint = f"{BACKEND_URL}/api/win/analyze"
        response = requests.post(endpoint, json=analysis_data, timeout=10)
        
        if response.status_code == 200:
            log.info(paint(f"[FIREBASE] Win enviado para analise: {ativo} {direction} +${profit:.2f}", C.G))
            return True
        else:
            log.warning(f"[FIREBASE] Erro ao enviar win: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        log.warning("[FIREBASE] Backend nao disponivel - win nao enviado")
        return False
    except Exception as e:
        log.warning(f"[FIREBASE] Erro ao enviar win: {e}")
        return False


def _generate_win_analysis(
    ativo: str,
    direction: str,
    profit: float,
    setup: Dict[str, Any],
    momentum_data: Dict[str, Any],
    trend_data: Dict[str, Any],
    projection_data: Dict[str, Any],
    candles_data: Dict[str, Any],
    chart_analysis: Dict[str, Any] = None
) -> str:
    """Gera análise textual do WIN para identificar padrões de sucesso."""
    success_factors = []
    
    # Analisa fatores de sucesso
    score = setup.get("score", 0)
    if score >= 0.8:
        success_factors.append(f"Score alto: {score:.2f}")
    
    # Verifica momentum
    if momentum_data:
        mom_score = momentum_data.get("momentum_score", 0)
        if mom_score >= 0.55:
            success_factors.append(f"Momentum forte: {mom_score:.0%}")
        if momentum_data.get("valid", False):
            success_factors.append("Momentum validado")
    
    # Verifica tendência
    if trend_data:
        if trend_data.get("trend_aligned", False):
            success_factors.append("Tendencia alinhada com direcao")
    
    # Verifica projeção
    if projection_data:
        if projection_data.get("projection_aligned", False):
            success_factors.append("Projecao alinhada com direcao")
        conf = projection_data.get("confidence", 0)
        if conf >= 0.6:
            success_factors.append(f"Alta confianca na projecao: {conf:.0%}")
    
    # Verifica contexto de mercado
    if candles_data:
        trend = candles_data.get("trend", "")
        if (trend == "bullish" and direction.upper() == "CALL") or \
           (trend == "bearish" and direction.upper() == "PUT"):
            success_factors.append(f"Operou a favor da tendencia ({trend})")
    
    # Verifica análise gráfica
    if chart_analysis:
        win_prob = chart_analysis.get("win_probability", 50)
        if win_prob >= 60:
            success_factors.append(f"Win probability alta: {win_prob:.0f}%")
        
        pattern = chart_analysis.get("pattern", {})
        if pattern and pattern.get("name", "none") != "none":
            pattern_dir = pattern.get("direction", "NEUTRAL")
            if pattern_dir == direction.upper():
                success_factors.append(f"Padrao {pattern.get('name')} confirmou direcao")
    
    # Verifica setup específico
    entry_mom = setup.get("entry_momentum", 0)
    if entry_mom >= 0.5:
        success_factors.append(f"Entry momentum bom: {entry_mom:.0%}")
    
    late_ext = setup.get("late_ext", 0)
    if late_ext <= 0.5:
        success_factors.append(f"Entrada no timing certo: late={late_ext:.2f}ATR")
    
    context_score = setup.get("context_score", 0)
    if context_score >= 0.5:
        success_factors.append(f"Contexto favoravel: {context_score:.0%}")
    
    # Monta análise
    if not success_factors:
        success_factors.append("WIN dentro da variancia normal")
    
    analysis = f"""
ANALISE DE WIN - {ativo}
==================================================
Direcao: {direction}
Lucro: +${profit:.2f}
Score: {score:.2f}

FATORES DE SUCESSO:
"""
    for i, factor in enumerate(success_factors, 1):
        analysis += f"\n{i}. {factor}"
    
    analysis += "\n\nPADRAO VENCEDOR - Replicar em entradas futuras!"
    
    return analysis


def _generate_loss_analysis(
    ativo: str,
    direction: str,
    setup: Dict[str, Any],
    momentum_data: Dict[str, Any],
    trend_data: Dict[str, Any],
    projection_data: Dict[str, Any],
    candles_data: Dict[str, Any],
    chart_analysis: Dict[str, Any] = None
) -> str:
    """Gera análise textual do loss para aprendizado futuro."""
    problems = []
    recommendations = []
    
    # Verifica problemas de momentum
    if momentum_data:
        if not momentum_data.get("valid", True):
            problems.append(f"Momentum invalido: {momentum_data.get('reason', 'desconhecido')}")
            recommendations.append("Verificar confirmacao de momentum antes da entrada")
        
        mom_score = momentum_data.get("momentum_score", 0)
        if mom_score < 0.5:
            problems.append(f"Momentum fraco: score={mom_score:.2f}")
    
    # Verifica problemas de tendência
    if trend_data:
        if not trend_data.get("valid", True):
            problems.append(f"Tendencia contraria: {trend_data.get('reason', 'desconhecido')}")
            recommendations.append("Evitar entradas contra a tendencia de curto prazo")
        
        if not trend_data.get("trend_aligned", False):
            problems.append("Tendencia nao alinhada com direcao")
    
    # Verifica problemas de projeção
    if projection_data:
        if not projection_data.get("valid", True):
            problems.append(f"Projecao desfavoravel: {projection_data.get('reason', 'desconhecido')}")
            recommendations.append("Verificar projecao de tendencia antes de operar")
        
        if not projection_data.get("projection_aligned", False):
            proj_dir = projection_data.get("projected_direction", "?")
            problems.append(f"Projecao na direcao oposta: {proj_dir}")
        
        reversal_risk = projection_data.get("reversal_risk", 0)
        if reversal_risk > 0.3:
            problems.append(f"Alto risco de reversao: {reversal_risk:.2f}")
            recommendations.append("Evitar entradas em zonas de sobrecompra/sobrevenda")
    
    # Verifica contexto de mercado
    if candles_data:
        trend = candles_data.get("trend", "")
        if trend == "bullish" and direction.upper() == "PUT":
            problems.append("Operou PUT em tendencia bullish")
            recommendations.append("Evitar PUT quando mercado esta claramente bullish")
        elif trend == "bearish" and direction.upper() == "CALL":
            problems.append("Operou CALL em tendencia bearish")
            recommendations.append("Evitar CALL quando mercado esta claramente bearish")
    
    # ===== ANÁLISE GRÁFICA PROFISSIONAL =====
    if chart_analysis:
        win_prob = chart_analysis.get("win_probability", 50)
        pattern = chart_analysis.get("pattern", {})
        structure = chart_analysis.get("structure", {})
        
        # Probabilidade baixa
        if win_prob < 50:
            problems.append(f"Probabilidade de WIN baixa: {win_prob:.0f}%")
            recommendations.append(f"Aguardar setups com WIN >= 55%")
        
        # Padrão de candlestick contra
        if pattern and pattern.get("direction") not in ["NEUTRAL", direction.upper()]:
            problems.append(f"Padrao de candle {pattern.get('name', '?')} indica {pattern.get('direction', '?')}")
            recommendations.append("Respeitar padroes de candlestick de reversao")
        
        # Estrutura de mercado contra
        if structure and structure.get("bias") not in ["NEUTRAL", direction.upper()]:
            problems.append(f"Estrutura de mercado ({structure.get('structure', '?')}) favorece {structure.get('bias', '?')}")
            recommendations.append("Operar a favor da estrutura de mercado")
        
        # Warnings da análise gráfica
        chart_warnings = chart_analysis.get("warnings", [])
        for warn in chart_warnings[:3]:
            problems.append(f"Alerta grafico: {warn}")
    
    # Monta análise
    analysis = f"""
ANALISE DE LOSS - {ativo}
{'='*50}
Direcao: {direction.upper()}
Score: {setup.get('score', 0):.2f}
WIN Probability: {chart_analysis.get('win_probability', '?') if chart_analysis else '?'}%

PROBLEMAS IDENTIFICADOS:
"""
    
    if problems:
        for i, prob in enumerate(problems, 1):
            analysis += f"\n{i}. {prob}"
    else:
        analysis += "\n- Loss dentro da normalidade (variacao de mercado)"
    
    analysis += "\n\nRECOMENDACOES:"
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            analysis += f"\n{i}. {rec}"
    else:
        analysis += "\n- Manter estrategia atual"
    
    return analysis

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

# ===================== IA ONLINE (Bayes + UCB) - VERSÃO INTELIGENTE =====================
def _safe_load_json(path: str) -> Dict[str, Any]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"meta": {"total": 0, "global_wins": 0, "global_losses": 0}, "arms": {}, "patterns": {}, "recent_trades": {}}

def _safe_save_json(path: str, data: Dict[str, Any]):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _apply_temporal_decay(stats: Dict[str, Any]):
    """
    NOVO: Aplica decaimento temporal aos parâmetros Bayesianos.
    Resultados antigos valem menos, permitindo adaptação mais rápida.
    """
    arms = stats.get("arms", {})
    for key, arm in arms.items():
        a = float(arm.get("a", 1.0))
        b = float(arm.get("b", 1.0))
        n = int(arm.get("n", 0))
        
        # Aplica decaimento suave (mantém proporção mas reduz magnitude)
        decay = AI_DECAY_RATE
        # Não deixa os valores ficarem muito pequenos
        new_a = max(1.0, a * decay)
        new_b = max(1.0, b * decay)
        
        arm["a"] = new_a
        arm["b"] = new_b
    
    log.info(f"[AI] Decaimento temporal aplicado ({AI_DECAY_RATE}x) - IA adaptada!")

def _get_global_winrate(stats: Dict[str, Any]) -> float:
    """
    NOVO: Calcula winrate global de todos os trades.
    Usado para ajustar o prior adaptativamente.
    """
    meta = stats.get("meta", {})
    global_wins = int(meta.get("global_wins", 0))
    global_losses = int(meta.get("global_losses", 0))
    total = global_wins + global_losses
    
    if total < 10:  # Precisa de mínimo de dados
        return 0.50  # Neutro
    
    return global_wins / total

def _get_recent_performance(recent_trades: list) -> Dict[str, float]:
    """
    NOVO: Analisa performance recente de um padrão.
    Retorna winrate recente e tendência.
    """
    if not recent_trades or len(recent_trades) < 3:
        return {"recent_winrate": 0.50, "trend": 0.0, "streak": 0}
    
    wins = sum(recent_trades)
    recent_wr = wins / len(recent_trades)
    
    # Calcula tendência (melhorando ou piorando?)
    first_half = recent_trades[:len(recent_trades)//2]
    second_half = recent_trades[len(recent_trades)//2:]
    
    wr_first = sum(first_half) / max(1, len(first_half))
    wr_second = sum(second_half) / max(1, len(second_half))
    trend = wr_second - wr_first  # Positivo = melhorando
    
    # Streak atual
    streak = 0
    last_result = recent_trades[-1]
    for r in reversed(recent_trades):
        if r == last_result:
            streak += 1
        else:
            break
    streak = streak if last_result == 1 else -streak  # Negativo se losses
    
    return {"recent_winrate": recent_wr, "trend": trend, "streak": streak}

def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def _bucket(x: float, step: float, lo: float, hi: float) -> int:
    x = _clip(x, lo, hi)
    return int(round((x - lo) / step))

def ai_make_key(ativo: str, setup: Dict[str, Any]) -> str:
    """
    Chave compacta GLOBAL (não separada por ativo ou padrão) para análise:
    - direção
    - score (buckets mais refinados)
    - pb_len (tamanho do pullback)
    - retr (retração)
    - A_atr (força do impulso)
    - effA (eficiência do impulso A)
    - flips (chopiness)
    - distBreak (distância da quebra)
    
    NOTA: Não inclui ativo ou padrões de vela na chave para manter aprendizado GLOBAL
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
    b_eff = _bucket(effA, 0.08, 0.50, 1.00)     # eficiência
    b_flip = _bucket(flips, 0.10, 0.0, 0.80)    # chopiness
    b_dist = _bucket(distBreak, 0.05, 0.0, 0.50) # distância da quebra

    return f"{d}|sc{b_sc}|pb{pb}|re{b_re}|A{b_A}|eff{b_eff}|fl{b_flip}|dst{b_dist}"

def ai_prior_from_setup(setup: Dict[str, Any], stats: Dict[str, Any] = None) -> float:
    """
    Prior INTELIGENTE baseado em múltiplos fatores + ADAPTATIVO.
    MELHORIAS:
    - Analisa confluência de sinais
    - Ajusta baseado na performance global (se disponível)
    - Aprende padrões de sucesso
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
        p += 0.05
    elif effA < 0.55:
        p -= 0.04

    # 2. Baixo chopiness (mercado direcional) aumenta confiança
    if flips < 0.30:
        p += 0.06
    elif flips > 0.55:
        p -= 0.05

    # 3. Retração ideal (0.3-0.5) aumenta confiança
    if 0.30 <= retr <= 0.50:
        p += 0.05
    elif retr < 0.18 or retr > 0.68:
        p -= 0.04

    # 4. Quebra próxima e limpa aumenta confiança
    if distBreak < 0.12:
        p += 0.04
    elif distBreak > 0.28:
        p -= 0.03
    
    # ===== MELHORIA: PRIOR ADAPTATIVO BASEADO EM PERFORMANCE GLOBAL =====
    if AI_ADAPTIVE_PRIOR and stats is not None:
        global_wr = _get_global_winrate(stats)
        
        # Ajusta prior baseado em como estamos performando globalmente
        # Se ganhando muito: ligeiro aumento no prior (mercado favorável)
        # Se perdendo muito: reduz prior (mercado desfavorável)
        if global_wr > 0.55:  # Performance boa
            adjustment = (global_wr - 0.50) * AI_GLOBAL_WINRATE_INFLUENCE
            p += adjustment
        elif global_wr < 0.45:  # Performance ruim
            adjustment = (0.50 - global_wr) * AI_GLOBAL_WINRATE_INFLUENCE
            p -= adjustment

    return _clip(p, 0.38, 0.78)

def ai_predict(ativo: str, setup: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, float]:
    """
    VERSÃO MELHORADA - Previsão mais inteligente
    
    Retorna: prob, bayes_mean, ucb01, conf, n_arm, total, recent_info
    
    MELHORIAS:
    - Considera memória recente com peso extra
    - Detecta tendências de melhora/piora
    - Prior adaptativo baseado em performance global
    """
    key = ai_make_key(ativo, setup)
    arms = stats.setdefault("arms", {})
    meta = stats.setdefault("meta", {"total": 0, "global_wins": 0, "global_losses": 0})
    recent_trades = stats.get("recent_trades", {})

    total = int(meta.get("total", 0))
    arm = arms.get(key)

    # Prior ADAPTATIVO (passa stats para ajustar baseado em performance global)
    prior = ai_prior_from_setup(setup, stats)

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
                "conf": float(conf), "n_arm": 0, "total": total, "key": key, "prior": prior,
                "recent_wr": 0.50, "trend": 0.0}

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

    # confiança cresce com amostras (mais rápido agora)
    conf = _clip(n / (n + 8.0), 0.0, 0.99)  # Reduzido de 10 para 8

    # ===== MELHORIA: CONSIDERA PERFORMANCE RECENTE =====
    recent = recent_trades.get(key, [])
    recent_info = _get_recent_performance(recent)
    recent_wr = recent_info["recent_winrate"]
    trend = recent_info["trend"]
    
    # Mistura prior + bayes + performance recente
    # Performance recente tem peso extra quando temos dados
    w_bayes = _clip(n / (n + 20.0), 0.0, 0.7)  # Até 70% peso no bayes
    w_recent = _clip(len(recent) / (len(recent) + 5.0), 0.0, 0.25) if recent else 0.0  # Até 25% peso nos recentes
    w_prior = 1.0 - w_bayes - w_recent
    
    prob = w_prior * prior + w_bayes * bayes_mean + w_recent * recent_wr
    
    # Ajusta baseado na tendência (se está melhorando ou piorando)
    if trend > 0.15:  # Tendência de melhora
        prob = min(1.0, prob + 0.03)
    elif trend < -0.15:  # Tendência de piora
        prob = max(0.0, prob - 0.04)
    
    prob = _clip(prob, 0.0, 1.0)

    return {"prob": float(prob), "bayes": float(bayes_mean), "ucb01": float(ucb01),
            "conf": float(conf), "n_arm": n, "total": total, "key": key, "prior": prior,
            "recent_wr": float(recent_wr), "trend": float(trend)}

def ai_update(ativo: str, setup: Dict[str, Any], pnl: float, stats: Dict[str, Any]):
    """
    VERSÃO MELHORADA - Aprendizado mais inteligente e rápido
    
    pnl > 0 => sucesso
    pnl < 0 => falha
    pnl = 0 => ignora

    MELHORIAS:
    - Losses pesam AI_LOSS_WEIGHT vezes mais (aprende rápido dos erros)
    - Memória de curto prazo (últimos trades pesam mais)
    - Decaimento temporal (trades antigos valem menos)
    - Rastreamento de sequência de losses
    """
    if pnl == 0:
        return

    key = ai_make_key(ativo, setup)
    arms = stats.setdefault("arms", {})
    meta = stats.setdefault("meta", {"total": 0, "global_wins": 0, "global_losses": 0})
    patterns = stats.setdefault("patterns", {})
    recent_trades = stats.setdefault("recent_trades", {})  # NOVO: memória recente

    arm = arms.get(key)
    if arm is None:
        prior = ai_prior_from_setup(setup, stats)  # Passa stats para prior adaptativo
        arms[key] = {"a": 2.0 * prior, "b": 2.0 * (1.0 - prior), "n": 0}
        arm = arms[key]

    # Atualiza Bayesian com PESO DIFERENCIADO
    a = float(arm.get("a", 1.0))
    b = float(arm.get("b", 1.0))
    n = int(arm.get("n", 0))

    # ===== MELHORIA 1: LOSSES PESAM MAIS =====
    if pnl > 0:
        a += 1.0  # Win normal
        meta["global_wins"] = int(meta.get("global_wins", 0)) + 1
    else:
        b += AI_LOSS_WEIGHT  # Loss pesa 2.5x mais - APRENDE RÁPIDO DOS ERROS!
        meta["global_losses"] = int(meta.get("global_losses", 0)) + 1

    n += 1
    arm["a"], arm["b"], arm["n"] = a, b, n
    meta["total"] = int(meta.get("total", 0)) + 1

    # ===== MELHORIA 2: MEMÓRIA DE CURTO PRAZO =====
    # Rastreia últimos trades por padrão para detectar sequências
    recent = recent_trades.get(key, [])
    recent.append(1 if pnl > 0 else 0)
    # Mantém apenas os últimos AI_RECENT_MEMORY trades
    if len(recent) > AI_RECENT_MEMORY:
        recent = recent[-AI_RECENT_MEMORY:]
    recent_trades[key] = recent

    # NOVO: Rastreamento de padrão para bloqueio inteligente
    pattern = patterns.get(key)
    if pattern is None:
        patterns[key] = {"trades": 0, "wins": 0, "losses": 0, "consecutive_losses": 0, "last_result": None}
        pattern = patterns[key]

    pattern["trades"] += 1
    if pnl > 0:
        pattern["wins"] += 1
        pattern["consecutive_losses"] = 0  # Reset sequência de losses
        pattern["last_result"] = "win"
    else:
        pattern["losses"] += 1
        pattern["consecutive_losses"] = int(pattern.get("consecutive_losses", 0)) + 1
        pattern["last_result"] = "loss"
    
    # ===== NOVO: APRENDIZADO DE PADRÕES DE CANDLESTICK =====
    # Extrai padrões do setup e atualiza aprendizado
    detected_patterns = setup.get("patterns", [])
    direction = setup.get("final_direction", setup.get("dir", ""))
    is_win = pnl > 0
    
    for p in detected_patterns:
        pattern_name = p.get("name", "") or p.get("nome", "")
        if pattern_name and pattern_name != "none":
            ai_update_candlestick_learning(pattern_name, direction, is_win, stats)
    
    # ===== MELHORIA 3: DECAIMENTO TEMPORAL (aplica periodicamente) =====
    total_trades = int(meta.get("total", 0))
    if total_trades > 0 and total_trades % 50 == 0:  # A cada 50 trades
        _apply_temporal_decay(stats)

def ai_should_block_pattern(ativo: str, setup: Dict[str, Any], stats: Dict[str, Any]) -> Tuple[bool, str]:
    """
    VERSÃO MELHORADA - Bloqueio mais inteligente
    
    CRITÉRIOS DE BLOQUEIO:
    1. Winrate consistentemente baixo (< AI_MIN_WINRATE)
    2. Sequência de losses consecutivas (>= AI_CONSECUTIVE_LOSS_BLOCK)
    3. Tendência de piora forte nos últimos trades
    
    PERMITE:
    - Fase de aprendizado (poucos dados)
    - Padrões com winrate aceitável
    - Padrões em recuperação (tendência de melhora)

    Returns:
        (should_block, reason)
    """
    key = ai_make_key(ativo, setup)
    patterns = stats.get("patterns", {})
    recent_trades_dict = stats.get("recent_trades", {})
    pattern = patterns.get(key)

    # Fase 1: APRENDIZADO - permite tudo (sem histórico suficiente)
    if pattern is None or pattern["trades"] < AI_MIN_SAMPLES:
        trades_count = pattern["trades"] if pattern else 0
        return False, f"learning({trades_count}/{AI_MIN_SAMPLES})"

    # ===== BLOQUEIO 1: SEQUÊNCIA DE LOSSES CONSECUTIVAS =====
    consecutive_losses = int(pattern.get("consecutive_losses", 0))
    if consecutive_losses >= AI_CONSECUTIVE_LOSS_BLOCK:
        return True, f"blocked_streak={consecutive_losses}losses_seguidas"

    # Fase 2: AVALIAÇÃO - analisa performance real
    winrate = pattern["wins"] / max(1, pattern["trades"])

    # ===== BLOQUEIO 2: WINRATE MUITO BAIXO =====
    if winrate < AI_MIN_WINRATE:
        return True, f"blocked_wr={winrate:.0%}({pattern['wins']}W/{pattern['losses']}L)"
    
    # ===== ANÁLISE DE TENDÊNCIA RECENTE =====
    recent = recent_trades_dict.get(key, [])
    if len(recent) >= 5:
        recent_info = _get_recent_performance(recent)
        recent_wr = recent_info["recent_winrate"]
        trend = recent_info["trend"]
        
        # ===== BLOQUEIO 3: TENDÊNCIA DE PIORA FORTE =====
        # Mesmo com winrate geral OK, se últimos trades são ruins, bloqueia temporariamente
        if recent_wr < 0.30 and trend < -0.15:
            return True, f"blocked_recent_bad(wr_recent={recent_wr:.0%},trend={trend:.2f})"
        
        # ===== PERMITE: PADRÃO EM RECUPERAÇÃO =====
        # Se tendência de melhora, dá mais uma chance mesmo com winrate na zona de risco
        if 0.35 <= winrate < AI_MIN_WINRATE and trend > 0.10:
            return False, f"recovery_chance_wr={winrate:.0%}(trend_up={trend:.2f})"

    # Permite: padrão tem performance aceitável
    return False, f"approved_wr={winrate:.0%}({pattern['wins']}W/{pattern['losses']}L)"

# ===== NOVO: APRENDIZADO ESPECÍFICO DE PADRÕES DE CANDLESTICK =====
def ai_update_candlestick_learning(pattern_name: str, direction: str, is_win: bool, stats: Dict[str, Any]):
    """
    NOVO: Aprende especificamente quais padrões de candlestick funcionam.
    Armazena estatísticas por nome de padrão + direção.
    """
    if not pattern_name or pattern_name == "none":
        return
    
    candle_patterns = stats.setdefault("candle_patterns", {})
    key = f"{pattern_name}_{direction}"
    
    if key not in candle_patterns:
        candle_patterns[key] = {
            "wins": 0, 
            "losses": 0, 
            "total": 0,
            "consecutive_losses": 0,
            "last_results": []  # Últimos 10 resultados
        }
    
    p = candle_patterns[key]
    p["total"] += 1
    
    if is_win:
        p["wins"] += 1
        p["consecutive_losses"] = 0
    else:
        p["losses"] += 1
        p["consecutive_losses"] = p.get("consecutive_losses", 0) + 1
    
    # Mantém últimos 10 resultados
    p["last_results"].append(1 if is_win else 0)
    if len(p["last_results"]) > 10:
        p["last_results"] = p["last_results"][-10:]
    
    # Log quando padrão tem resultados suficientes
    if p["total"] >= 3:
        wr = p["wins"] / p["total"]
        log.info(f"[🧠 IA PADROES] {key}: {p['wins']}W/{p['losses']}L ({wr:.0%})")

def ai_should_block_candlestick(pattern_name: str, direction: str, stats: Dict[str, Any]) -> Tuple[bool, str]:
    """
    NOVO: Verifica se um padrão de candlestick específico deve ser bloqueado.
    """
    if not pattern_name or pattern_name == "none":
        return False, "no_pattern"
    
    candle_patterns = stats.get("candle_patterns", {})
    key = f"{pattern_name}_{direction}"
    p = candle_patterns.get(key)
    
    if not p or p["total"] < 5:  # Mínimo 5 trades para avaliar
        return False, f"learning({p['total'] if p else 0}/5)"
    
    wr = p["wins"] / max(1, p["total"])
    
    # Bloqueia se winrate < 40%
    if wr < 0.40:
        return True, f"blocked_pattern({wr:.0%})"
    
    # Bloqueia se 3+ losses consecutivas
    if p.get("consecutive_losses", 0) >= 3:
        return True, f"blocked_streak({p['consecutive_losses']}losses)"
    
    # Verifica tendência recente (últimos 5)
    recent = p.get("last_results", [])[-5:]
    if len(recent) >= 5:
        recent_wr = sum(recent) / len(recent)
        if recent_wr < 0.20:  # Últimos 5 muito ruins
            return True, f"blocked_recent({recent_wr:.0%})"
    
    return False, f"approved({wr:.0%})"

def ai_get_report(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    NOVO: Gera relatório do estado da IA para monitoramento.
    Útil para acompanhar o aprendizado em tempo real.
    """
    meta = stats.get("meta", {})
    patterns = stats.get("patterns", {})
    arms = stats.get("arms", {})
    
    total = int(meta.get("total", 0))
    global_wins = int(meta.get("global_wins", 0))
    global_losses = int(meta.get("global_losses", 0))
    global_wr = global_wins / max(1, global_wins + global_losses)
    
    # Conta padrões por status
    blocked_patterns = 0
    approved_patterns = 0
    learning_patterns = 0
    
    for key, p in patterns.items():
        if p["trades"] < AI_MIN_SAMPLES:
            learning_patterns += 1
        elif p["wins"] / max(1, p["trades"]) < AI_MIN_WINRATE:
            blocked_patterns += 1
        else:
            approved_patterns += 1
    
    # Top 5 melhores e piores padrões
    pattern_list = []
    for key, p in patterns.items():
        if p["trades"] >= 3:  # Mínimo 3 trades
            wr = p["wins"] / max(1, p["trades"])
            pattern_list.append({"key": key, "trades": p["trades"], "wins": p["wins"], 
                                "losses": p["losses"], "winrate": wr})
    
    pattern_list.sort(key=lambda x: x["winrate"], reverse=True)
    top_5 = pattern_list[:5] if len(pattern_list) >= 5 else pattern_list
    worst_5 = pattern_list[-5:] if len(pattern_list) >= 5 else []
    
    report = {
        "total_trades": total,
        "global_wins": global_wins,
        "global_losses": global_losses,
        "global_winrate": f"{global_wr:.1%}",
        "total_patterns": len(patterns),
        "patterns_learning": learning_patterns,
        "patterns_approved": approved_patterns,
        "patterns_blocked": blocked_patterns,
        "top_5_patterns": top_5,
        "worst_5_patterns": worst_5,
        "ai_settings": {
            "min_samples": AI_MIN_SAMPLES,
            "min_winrate": f"{AI_MIN_WINRATE:.0%}",
            "loss_weight": AI_LOSS_WEIGHT,
            "consecutive_loss_block": AI_CONSECUTIVE_LOSS_BLOCK,
            "adaptive_prior": AI_ADAPTIVE_PRIOR
        }
    }
    
    return report

def ai_log_status(stats: Dict[str, Any]):
    """
    NOVO: Loga status resumido da IA para acompanhamento.
    """
    report = ai_get_report(stats)
    log.info(f"[AI STATUS] Total: {report['total_trades']} trades | "
             f"WR Global: {report['global_winrate']} | "
             f"Padrões: {report['patterns_approved']}✓ {report['patterns_blocked']}✗ {report['patterns_learning']}⏳")
    
    # NOVO: Mostra padrões de candlestick aprendidos
    candle_patterns = stats.get("candle_patterns", {})
    if candle_patterns:
        log.info(f"[AI CANDLESTICK] {len(candle_patterns)} padrões aprendidos:")
        # Ordena por total de trades
        sorted_patterns = sorted(candle_patterns.items(), key=lambda x: x[1]["total"], reverse=True)
        for key, p in sorted_patterns[:5]:  # Top 5
            wr = p["wins"] / max(1, p["total"])
            status = "✓" if wr >= 0.50 else ("⚠" if wr >= 0.40 else "✗")
            log.info(f"   {status} {key}: {p['wins']}W/{p['losses']}L ({wr:.0%})")

# ===================== PERNADA B =====================
def pernada_b(df_m1: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    if len(df_m1) < 240:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["poucas_velas"]}

    # ANÁLISE INTELIGENTE DE CONTEXTO (DESATIVADO TEMPORARIAMENTE PARA TESTE)
    context = analyze_market_context(df_m1, atr_val)
    market_quality = float(context.get("quality", 0.0))

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

            # CORREÇÃO: A direção da PERNADA A define a tendência principal
            # PUT = impulso de queda (A vai pra baixo)
            # CALL = impulso de alta (A vai pra cima)
            dir_impulso_A = "PUT" if move < 0 else ("CALL" if move > 0 else "NEUTRAL")
            if dir_impulso_A == "NEUTRAL":
                continue

            eff_A = leg_efficiency(imp)
            if eff_A < MIN_EFF_A:
                continue

            # CORREÇÃO CRÍTICA: O pullback deve ser CONTRA a pernada A
            # Se A foi de QUEDA (PUT), o pullback deve ter velas de ALTA (contra=CALL)
            # Se A foi de ALTA (CALL), o pullback deve ter velas de QUEDA (contra=PUT)
            contra = 0
            for _, r in pb.iterrows():
                d = candle_dir(r)
                # Se impulso foi PUT (queda), conta velas CALL no pullback
                if dir_impulso_A == "PUT" and d == 1:
                    contra += 1
                # Se impulso foi CALL (alta), conta velas PUT no pullback
                if dir_impulso_A == "CALL" and d == -1:
                    contra += 1

            # Pullback precisa ter velas contra a pernada A (reduzido de 67% para 50%)
            if contra < max(1, int(math.ceil(pb_len * 0.50))):
                continue

            # VALIDAÇÃO INTELIGENTE: Verifica continuação de tendência (NÃO BLOQUEANTE)
            impulse_start_idx = len(df_m1) - (pb_len + 1 + w)
            trend_validation = validate_trend_continuation(df_m1, dir_impulso_A, impulse_start_idx)

            # Se não validar tendência, apenas não dá bônus (não bloqueia mais)
            trend_strength = float(trend_validation.get("strength", 0.0)) if trend_validation.get("valid", False) else 0.0

            # Calcula a retração do pullback
            if dir_impulso_A == "PUT":
                # Se A foi de queda, pullback sobe, medimos quanto subiu
                pb_high = float(pb["high"].max())
                retr = (pb_high - bot) / max(size_A, 1e-9)
            else:  # dir_impulso_A == "CALL"
                # Se A foi de alta, pullback desce, medimos quanto desceu
                pb_low = float(pb["low"].min())
                retr = (top - pb_low) / max(size_A, 1e-9)

            if retr < RETR_MIN or retr > RETR_MAX:
                continue

            c1 = float(decision["close"])

            # ENTRADA é na direção da PERNADA A (continuação da tendência)
            # Se A foi PUT (queda), entramos PUT quando rompe o pullback pra baixo
            # Se A foi CALL (alta), entramos CALL quando rompe o pullback pra cima
            dir_entrada = dir_impulso_A

            # bloqueio SR forte (múltiplas regiões)
            blk_sr = sr_block_directional_multi(df_m1, atr_val, dir_entrada)
            if blk_sr:
                continue

            # Calcula os extremos do pullback (necessário para ambas direções)
            pb_high = float(pb["high"].max())
            pb_low = float(pb["low"].min())

            # LÓGICA DE ROMPIMENTO CORRIGIDA
            if dir_entrada == "CALL":
                # CALL: Impulso A foi de ALTA, pullback desceu, agora rompe pullback pra CIMA
                if not (c1 > pb_low + BREAK_MARGIN_ATR * atr_val):
                    continue
                if q["upper_frac"] > MAX_WICK_AGAINST:
                    continue

                dist = (c1 - pb_low) / max(atr_val, 1e-9)
                if dist > MAX_BREAK_DISTANCE_ATR:
                    continue

                # VALIDAÇÃO DE QUALIDADE DA ENTRADA (NOVO)
                entry_validation = validate_entry_quality(df_m1, atr_val, "CALL", c1, pb_high, pb_low)
                if not entry_validation.get("valid", False):
                    continue  # Pula entrada se não passar na validação

                entry_confidence = float(entry_validation.get("confidence", 0.0))
                risk_atr = float(entry_validation.get("risk_atr", 0.0))
                entry_momentum = float(entry_validation.get("momentum", 0.0))
                entry_alignment = float(entry_validation.get("alignment", 0.0))

                # CONFLUÊNCIA COM LINHA DE TENDÊNCIA (LTA para CALL)
                lt_conf = check_trendline_confluence(df_m1, pb_high, pb_low, "CALL", atr_val)
                lt_confluence = float(lt_conf.get("confluence", 0.0))
                has_lt = lt_conf.get("has_trendline", False)

                # Score base CONSERVADOR - Sistema Inteligente de Pontuação
                score = 0.35  # BASE MUITO REDUZIDA - precisa MERECER pontos

                # 1. IMPULSO (máx +0.12)
                impulso_score = min(0.12, (size_A / atr_val - IMPULSO_MIN_ATR) * 0.06)
                score += impulso_score

                # 2. EFICIÊNCIA (máx +0.15) - MUITO IMPORTANTE
                eff_score = min(0.15, max(0, (eff_A - MIN_EFF_A) * 0.35))
                score += eff_score

                # 3. RETRAÇÃO IDEAL (máx +0.10)
                if 0.30 <= retr <= 0.50:
                    retr_score = 0.10  # perfeito
                elif 0.25 <= retr <= 0.60:
                    retr_score = 0.05  # bom
                else:
                    retr_score = max(-0.05, -(abs(retr - 0.40) * 0.15))  # penaliza extremos
                score += retr_score

                # 4. PULLBACK (máx +0.05)
                if 2 <= pb_len <= 4:
                    pb_score = 0.05
                elif pb_len == 1 or pb_len == 5:
                    pb_score = 0.02
                else:
                    pb_score = 0.0
                score += pb_score

                # 5. CHOPPINESS - PENALIZA FORTE (máx -0.15)
                if flips_frac > 0.60:
                    chop_penalty = min(0.15, (flips_frac - 0.60) * 0.50)
                    score -= chop_penalty

                # 6. QUALIDADE DE CONTEXTO (máx +0.18) - CRÍTICO
                if market_quality > 0.60:
                    ctx_score = 0.18
                elif market_quality > 0.45:
                    ctx_score = 0.10
                elif market_quality > 0.30:
                    ctx_score = 0.03
                else:
                    ctx_score = -0.08  # PENALIZA contexto ruim
                score += ctx_score

                # 7. TENDÊNCIA (máx +0.08)
                trend_score = min(0.08, trend_strength * 0.12)
                score += trend_score

                # 8. QUALIDADE DA ENTRADA (máx +0.25) - MUITO CRÍTICO
                if entry_confidence > 0.65:
                    entry_score = 0.25
                elif entry_confidence > 0.55:
                    entry_score = 0.15
                elif entry_confidence > 0.48:
                    entry_score = 0.08
                else:
                    entry_score = -0.05  # PENALIZA entrada fraca
                score += entry_score

                # 9. MOMENTUM (máx +0.10)
                momentum_score = min(0.10, entry_momentum * 0.08)
                score += momentum_score

                # 10. ALINHAMENTO (máx +0.08)
                if entry_alignment >= 0.67:
                    align_score = 0.08
                elif entry_alignment >= 0.34:
                    align_score = 0.03
                else:
                    align_score = -0.03  # PENALIZA desalinhamento
                score += align_score

                # ⭐ BÔNUS POR CONFLUÊNCIA COM LINHA DE TENDÊNCIA (LTA) - MUITO IMPORTANTE!
                if has_lt and lt_confluence > 0.8:
                    score += 0.25  # BÔNUS GRANDE se tocou perfeitamente a LTA
                elif has_lt and lt_confluence > 0.5:
                    score += 0.15  # Bônus médio se próximo da LTA
                elif has_lt and lt_confluence > 0.2:
                    score += 0.05  # Bônus pequeno

                # 11. RISCO (máx -0.10)
                if risk_atr > 1.3:
                    risk_penalty = 0.10
                elif risk_atr > 1.0:
                    risk_penalty = 0.05
                elif risk_atr < 0.30:
                    risk_penalty = 0.05  # risco muito pequeno também é ruim
                else:
                    risk_penalty = 0.0
                score -= risk_penalty

                # 12. CONFLUÊNCIA PERFEITA - Bônus ENORME (máx +0.20)
                perfect_count = 0
                if market_quality > 0.60: perfect_count += 1
                if eff_A > 0.70: perfect_count += 1
                if 0.30 <= retr <= 0.50: perfect_count += 1
                if entry_confidence > 0.65: perfect_count += 1
                if entry_alignment >= 0.67: perfect_count += 1
                if lt_confluence > 0.8: perfect_count += 1

                if perfect_count >= 5:
                    confluence_bonus = 0.20  # SETUP EXCEPCIONAL
                elif perfect_count >= 4:
                    confluence_bonus = 0.12  # SETUP ÓTIMO
                elif perfect_count >= 3:
                    confluence_bonus = 0.05  # SETUP BOM
                else:
                    confluence_bonus = 0.0

                score += confluence_bonus

                # LIMITA SCORE
                score = float(max(0.0, min(0.95, score)))  # máx 0.95 (nunca 1.0)

                # ✅ FILTROS MÍNIMOS - IA VAI APRENDER E BLOQUEAR PADRÕES RUINS
                # Apenas sanity checks básicos - IA aprende o resto
                if score < 0.48:  # só bloqueia setups extremamente fracos
                    continue

                if market_quality < 0.25:  # só bloqueia contextos horríveis
                    continue

                if entry_confidence < 0.40:  # só bloqueia entradas muito fracas
                    continue

                # IA vai bloquear padrões específicos que dão loss consistente

                setup = {
                    "trade": True, "dir": "CALL", "score": score,
                    # campos para IA:
                    "pb_len": pb_len, "retr": float(retr),
                    "A_atr": float(size_A / max(atr_val, 1e-9)),
                    "effA": float(eff_A),
                    "flips": float(flips_frac),
                    "comp": float(comp),
                    "late": float(late_ext),
                    "late_ext": float(late_ext),  # NOVO: para smart_entry_decision
                    "distBreak": float(dist),
                    # contexto de mercado:
                    "market_quality": float(market_quality),
                    "context_score": float(market_quality),  # NOVO: para smart_entry_decision
                    "context": str(context.get("context", "?")),
                    "confluence_bonus": float(confluence_bonus),
                    "trend_strength": float(trend_strength),
                    "trend_reason": str(trend_validation.get("reason", "?")),
                    # validação de entrada (NOVO):
                    "entry_confidence": float(entry_confidence),
                    "entry_momentum": float(entry_momentum),
                    "entry_alignment": float(entry_alignment),
                    "risk_atr": float(risk_atr),
                    # confluência LT:
                    "lt_confluence": float(lt_confluence),
                    "has_lt": has_lt,
                    "reasons": [
                        "pernadaB_CALL",
                        f"A={size_A/atr_val:.2f}ATR", f"retr={retr:.2f}", f"pb={pb_len}",
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
                        f"⭐LTA={lt_confluence:.2f}" if has_lt else "sem_LTA"
                    ]
                }
            else:  # dir_entrada == "PUT"
                # PUT: Impulso A foi de QUEDA, pullback subiu, agora rompe pullback pra BAIXO
                pb_high = float(pb["high"].max())
                if not (c1 < pb_high - BREAK_MARGIN_ATR * atr_val):
                    continue
                if q["lower_frac"] > MAX_WICK_AGAINST:
                    continue

                dist = (pb_high - c1) / max(atr_val, 1e-9)
                if dist > MAX_BREAK_DISTANCE_ATR:
                    continue

                # VALIDAÇÃO DE QUALIDADE DA ENTRADA (NOVO)
                entry_validation = validate_entry_quality(df_m1, atr_val, "PUT", c1, pb_high, pb_low)
                if not entry_validation.get("valid", False):
                    continue  # Pula entrada se não passar na validação

                entry_confidence = float(entry_validation.get("confidence", 0.0))
                risk_atr = float(entry_validation.get("risk_atr", 0.0))
                entry_momentum = float(entry_validation.get("momentum", 0.0))
                entry_alignment = float(entry_validation.get("alignment", 0.0))

                # CONFLUÊNCIA COM LINHA DE TENDÊNCIA (LTB para PUT)
                lt_conf = check_trendline_confluence(df_m1, pb_high, pb_low, "PUT", atr_val)
                lt_confluence = float(lt_conf.get("confluence", 0.0))
                has_lt = lt_conf.get("has_trendline", False)

                # Score base CONSERVADOR - Sistema Inteligente de Pontuação (PUT)
                score = 0.35  # BASE MUITO REDUZIDA - precisa MERECER pontos

                # 1. IMPULSO (máx +0.12)
                impulso_score = min(0.12, (size_A / atr_val - IMPULSO_MIN_ATR) * 0.06)
                score += impulso_score

                # 2. EFICIÊNCIA (máx +0.15) - MUITO IMPORTANTE
                eff_score = min(0.15, max(0, (eff_A - MIN_EFF_A) * 0.35))
                score += eff_score

                # 3. RETRAÇÃO IDEAL (máx +0.10)
                if 0.30 <= retr <= 0.50:
                    retr_score = 0.10  # perfeito
                elif 0.25 <= retr <= 0.60:
                    retr_score = 0.05  # bom
                else:
                    retr_score = max(-0.05, -(abs(retr - 0.40) * 0.15))  # penaliza extremos
                score += retr_score

                # 4. PULLBACK (máx +0.05)
                if 2 <= pb_len <= 4:
                    pb_score = 0.05
                elif pb_len == 1 or pb_len == 5:
                    pb_score = 0.02
                else:
                    pb_score = 0.0
                score += pb_score

                # 5. CHOPPINESS - PENALIZA FORTE (máx -0.15)
                if flips_frac > 0.60:
                    chop_penalty = min(0.15, (flips_frac - 0.60) * 0.50)
                    score -= chop_penalty

                # 6. QUALIDADE DE CONTEXTO (máx +0.18) - CRÍTICO
                if market_quality > 0.60:
                    ctx_score = 0.18
                elif market_quality > 0.45:
                    ctx_score = 0.10
                elif market_quality > 0.30:
                    ctx_score = 0.03
                else:
                    ctx_score = -0.08  # PENALIZA contexto ruim
                score += ctx_score

                # 7. TENDÊNCIA (máx +0.08)
                trend_score = min(0.08, trend_strength * 0.12)
                score += trend_score

                # 8. QUALIDADE DA ENTRADA (máx +0.25) - MUITO CRÍTICO
                if entry_confidence > 0.65:
                    entry_score = 0.25
                elif entry_confidence > 0.55:
                    entry_score = 0.15
                elif entry_confidence > 0.48:
                    entry_score = 0.08
                else:
                    entry_score = -0.05  # PENALIZA entrada fraca
                score += entry_score

                # 9. MOMENTUM (máx +0.10)
                momentum_score = min(0.10, entry_momentum * 0.08)
                score += momentum_score

                # 10. ALINHAMENTO (máx +0.08)
                if entry_alignment >= 0.67:
                    align_score = 0.08
                elif entry_alignment >= 0.34:
                    align_score = 0.03
                else:
                    align_score = -0.03  # PENALIZA desalinhamento
                score += align_score

                # ⭐ BÔNUS POR CONFLUÊNCIA COM LINHA DE TENDÊNCIA (LTB) - MUITO IMPORTANTE!
                if has_lt and lt_confluence > 0.8:
                    score += 0.25  # BÔNUS GRANDE se tocou perfeitamente a LTB
                elif has_lt and lt_confluence > 0.5:
                    score += 0.15  # Bônus médio se próximo da LTB
                elif has_lt and lt_confluence > 0.2:
                    score += 0.05  # Bônus pequeno

                # 11. RISCO (máx -0.10)
                if risk_atr > 1.3:
                    risk_penalty = 0.10
                elif risk_atr > 1.0:
                    risk_penalty = 0.05
                elif risk_atr < 0.30:
                    risk_penalty = 0.05  # risco muito pequeno também é ruim
                else:
                    risk_penalty = 0.0
                score -= risk_penalty

                # 12. CONFLUÊNCIA PERFEITA - Bônus ENORME (máx +0.20)
                perfect_count = 0
                if market_quality > 0.60: perfect_count += 1
                if eff_A > 0.70: perfect_count += 1
                if 0.30 <= retr <= 0.50: perfect_count += 1
                if entry_confidence > 0.65: perfect_count += 1
                if entry_alignment >= 0.67: perfect_count += 1
                if lt_confluence > 0.8: perfect_count += 1

                if perfect_count >= 5:
                    confluence_bonus = 0.20  # SETUP EXCEPCIONAL
                elif perfect_count >= 4:
                    confluence_bonus = 0.12  # SETUP ÓTIMO
                elif perfect_count >= 3:
                    confluence_bonus = 0.05  # SETUP BOM
                else:
                    confluence_bonus = 0.0

                score += confluence_bonus

                # LIMITA SCORE
                score = float(max(0.0, min(0.95, score)))  # máx 0.95 (nunca 1.0)

                # ✅ FILTROS MÍNIMOS - IA VAI APRENDER E BLOQUEAR PADRÕES RUINS
                # Apenas sanity checks básicos - IA aprende o resto
                if score < 0.48:  # só bloqueia setups extremamente fracos
                    continue

                if market_quality < 0.25:  # só bloqueia contextos horríveis
                    continue

                if entry_confidence < 0.40:  # só bloqueia entradas muito fracas
                    continue

                # IA vai bloquear padrões específicos que dão loss consistente

                setup = {
                    "trade": True, "dir": "PUT", "score": score,
                    "pb_len": pb_len, "retr": float(retr),
                    "A_atr": float(size_A / max(atr_val, 1e-9)),
                    "effA": float(eff_A),
                    "flips": float(flips_frac),
                    "comp": float(comp),
                    "late": float(late_ext),
                    "late_ext": float(late_ext),  # NOVO: para smart_entry_decision
                    "distBreak": float(dist),
                    # contexto de mercado:
                    "market_quality": float(market_quality),
                    "context_score": float(market_quality),  # NOVO: para smart_entry_decision
                    "context": str(context.get("context", "?")),
                    "confluence_bonus": float(confluence_bonus),
                    "trend_strength": float(trend_strength),
                    "trend_reason": str(trend_validation.get("reason", "?")),
                    # validação de entrada (NOVO):
                    "entry_confidence": float(entry_confidence),
                    "entry_momentum": float(entry_momentum),
                    "entry_alignment": float(entry_alignment),
                    "risk_atr": float(risk_atr),
                    # confluência LT:
                    "lt_confluence": float(lt_confluence),
                    "has_lt": has_lt,
                    "reasons": [
                        "pernadaB_PUT",
                        f"A={size_A/atr_val:.2f}ATR", f"retr={retr:.2f}", f"pb={pb_len}",
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
                        f"⭐LTB={lt_confluence:.2f}" if has_lt else "sem_LTB"
                    ]
                }

            if best is None or setup["score"] > best["score"]:
                best = setup

    if best is None:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["sem_pernadaB_valida"]}

    # bloqueio final SR forte no momento do sinal
    block_final = sr_block_directional_multi(df_m1, atr_val, best["dir"])
    if block_final:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": [block_final]}

    return best

# ===================== ESCOLHER MELHOR SETUP DO MINUTO =====================
def escolher_melhor_setup(iq: IQ_Option, ativos: List[str]):
    """
    LÓGICA SIMPLIFICADA:
    1. Se detectar padrão 80%+ = ENTRA DIRETO!
    2. Se não tiver padrão = aguarda
    """
    best_trade = None
    best_any = None

    for a in ativos:
        if a in cooldown and (time.time() - cooldown[a]) < COOLDOWN_ATIVO:
            continue
        if a in cooldown_spike and (time.time() - cooldown_spike[a]) < (SPIKE_COOLDOWN_MIN * 60):
            continue

        df = get_candles_df(iq, a, TF_M1, N_M1, end_ts=end_ts_closed(TF_M1))
        if df is None or len(df) < 15:
            continue

        atr_val = atr(df, 14)
        last_closed = df.iloc[-1]

        if is_spike_wicky(last_closed, atr_val):
            cooldown_spike[a] = time.time()
            continue

        # ═══════════════════════════════════════════════════════════════
        # NOVA LÓGICA: DETECTAR PADRÃO 80%+ PRIMEIRO!
        # Se tiver padrão = força trade=True com score alto!
        # ═══════════════════════════════════════════════════════════════
        
        if PATTERN_DETECTOR_AVAILABLE:
            try:
                # Analisa padrões CALL e PUT
                result_call = analyze_patterns(df=df, direcao_sinal="CALL", atr_val=float(atr_val))
                result_put = analyze_patterns(df=df, direcao_sinal="PUT", atr_val=float(atr_val))
                
                padroes_call = result_call.get("padroes_candlestick", [])
                padroes_put = result_put.get("padroes_candlestick", [])
                
                # Busca melhor padrão CALL 80%+
                melhor_call = None
                for p in padroes_call:
                    if p.get("direcao") == "CALL" and p.get("confiabilidade", 0) >= 0.80:
                        if not melhor_call or p.get("confiabilidade", 0) > melhor_call.get("confiabilidade", 0):
                            melhor_call = p
                
                # Busca melhor padrão PUT 80%+
                melhor_put = None
                for p in padroes_put:
                    if p.get("direcao") == "PUT" and p.get("confiabilidade", 0) >= 0.80:
                        if not melhor_put or p.get("confiabilidade", 0) > melhor_put.get("confiabilidade", 0):
                            melhor_put = p
                
                # Se encontrou padrão 80%+ CALL
                if melhor_call:
                    conf = melhor_call.get("confiabilidade", 0.8)
                    nome = melhor_call.get("nome", "PADRAO")
                    # ===== CONFLUÊNCIAS OBRIGATÓRIAS =====
                    sr_block = sr_block_directional_multi(df, atr_val, "CALL")
                    if sr_block:
                        log.info(paint(f"[⛔ SR] {a} | CALL bloqueado: {sr_block}", C.Y))
                        continue
                    pingpong = sr_pingpong_zone(df, atr_val)
                    if pingpong:
                        log.info(paint(f"[⛔ SR] {a} | CALL bloqueado: {pingpong}", C.Y))
                        continue

                    estrutura = analyze_market_structure(df, atr_val, lookback=40)
                    if not estrutura.get("valid"):
                        log.info(paint(f"[⛔ ESTRUTURA] {a} | CALL bloqueado: estrutura inválida", C.Y))
                        continue

                    trend_dir = estrutura.get("trend")
                    trend_ok = (trend_dir == "BULLISH" and estrutura.get("trend_strength", 0) >= 0.45)
                    bos_ok = (estrutura.get("bos_detected") and estrutura.get("bos_direction") == "BULLISH") if USE_BOS else True
                    ideal_ok = (estrutura.get("ideal_direction") == "CALL" and estrutura.get("entry_quality", 0) >= 0.35)

                    # Não operar contra tendência forte (CALL contra baixa)
                    if trend_dir == "BEARISH" and USE_BOS and not bos_ok:
                        log.info(paint(f"[⛔ ESTRUTURA] {a} | CALL bloqueado: tendência BAIXA forte", C.Y))
                        continue

                    # Momentum curto (últimas 15 velas) não pode ser fortemente contrário
                    closes_15 = df["close"].tail(15).to_numpy(float)
                    down_count = sum(1 for i in range(14) if closes_15[i+1] < closes_15[i])
                    if down_count >= 11:
                        log.info(paint(f"[⛔ MOMENTUM] {a} | CALL bloqueado: 15 velas em baixa", C.Y))
                        continue

                    # Evita CALL no topo da estrutura
                    if estrutura.get("price_position") == "top_of_range" and USE_BOS and not bos_ok:
                        log.info(paint(f"[⛔ ESTRUTURA] {a} | CALL bloqueado: topo de range", C.Y))
                        continue

                    # Exige pelo menos 1 confluência estrutural
                    if not (trend_ok or bos_ok or ideal_ok):
                        log.info(paint(f"[⛔ ESTRUTURA] {a} | CALL bloqueado: trend_ok={trend_ok} bos_ok={bos_ok} ideal_ok={ideal_ok}", C.Y))
                        continue

                    # ===== MOMENTUM (15 velas + estrutura últimas 3) =====
                    momo = analyze_recent_momentum(df, atr_val)
                    is_continuation = "MARUBOZU" in nome
                    if is_continuation and not (momo.get("continuation_ok", False) and momo.get("last3_cont_call", False)):
                        log.info(paint(f"[⛔ MOMENTUM] {a} | CALL bloqueado: sem continuação/3velas ({momo.get('reason','')})", C.Y))
                        continue
                    if (not is_continuation) and momo.get("direction") == "BEARISH" and not (momo.get("reversal_ok_call", False) and momo.get("last3_rev_call", False)):
                        log.info(paint(f"[⛔ MOMENTUM] {a} | CALL bloqueado: sem exaustão/3velas ({momo.get('reason','')})", C.Y))
                        continue

                    log.info(paint(f"[🎯 PADRÃO] {a} | {nome} CALL ({conf*100:.0f}%) - Verificando CNN...", C.B))

                    # ===== NOVO PIPELINE: CNN + REGIME FILTER + RISK CONTROL =====
                    payout = obter_payout_ativo(iq, a) if 'obter_payout_ativo' in dir() else 80
                    pipeline_ok, pipeline_dir, pipeline_details = new_pipeline_decide(iq, a, df, atr_val, payout)

                    if not pipeline_ok or pipeline_dir != "CALL":
                        log.info(paint(f"[CNN] {a} | CALL bloqueado pelo pipeline: {pipeline_dir}", C.Y))
                        continue

                    log.info(paint(f"[✅ PIPELINE] {a} | {nome} CALL ({conf*100:.0f}%) - APROVADO!", C.G))
                    log.info(paint(f"[📈 ULTIMAS 10 VELAS] {a} | {format_last_candles(df, 10)}", C.B))

                    # Cria setup com trade=True
                    setup_pattern = {
                        "trade": True,
                        "dir": "CALL",
                        "score": conf * 100,
                        "ativo": a,
                        "padrao": nome,
                        "confiabilidade": conf,
                        "pipeline": pipeline_details
                    }
                    cand_trade = (conf * 100, a, setup_pattern, float(atr_val))
                    if best_trade is None or cand_trade[0] > best_trade[0]:
                        best_trade = cand_trade
                    continue  # Já encontrou padrão, não precisa analisar pernada
                
                # Se encontrou padrão 80%+ PUT
                if melhor_put:
                    conf = melhor_put.get("confiabilidade", 0.8)
                    nome = melhor_put.get("nome", "PADRAO")
                    # ===== CONFLUÊNCIAS OBRIGATÓRIAS =====
                    sr_block = sr_block_directional_multi(df, atr_val, "PUT")
                    if sr_block:
                        log.info(paint(f"[⛔ SR] {a} | PUT bloqueado: {sr_block}", C.Y))
                        continue
                    pingpong = sr_pingpong_zone(df, atr_val)
                    if pingpong:
                        log.info(paint(f"[⛔ SR] {a} | PUT bloqueado: {pingpong}", C.Y))
                        continue

                    estrutura = analyze_market_structure(df, atr_val, lookback=40)
                    if not estrutura.get("valid"):
                        log.info(paint(f"[⛔ ESTRUTURA] {a} | PUT bloqueado: estrutura inválida", C.Y))
                        continue

                    trend_dir = estrutura.get("trend")
                    trend_ok = (trend_dir == "BEARISH" and estrutura.get("trend_strength", 0) >= 0.45)
                    bos_ok = (estrutura.get("bos_detected") and estrutura.get("bos_direction") == "BEARISH") if USE_BOS else True
                    ideal_ok = (estrutura.get("ideal_direction") == "PUT" and estrutura.get("entry_quality", 0) >= 0.35)

                    # Não operar contra tendência forte (PUT contra alta)
                    if trend_dir == "BULLISH" and USE_BOS and not bos_ok:
                        log.info(paint(f"[⛔ ESTRUTURA] {a} | PUT bloqueado: tendência ALTA forte", C.Y))
                        continue

                    # Momentum curto (últimas 15 velas) não pode ser fortemente contrário
                    closes_15 = df["close"].tail(15).to_numpy(float)
                    up_count = sum(1 for i in range(14) if closes_15[i+1] > closes_15[i])
                    if up_count >= 11:
                        log.info(paint(f"[⛔ MOMENTUM] {a} | PUT bloqueado: 15 velas em alta", C.Y))
                        continue

                    # Evita PUT no fundo da estrutura
                    if estrutura.get("price_position") == "bottom_of_range" and USE_BOS and not bos_ok:
                        log.info(paint(f"[⛔ ESTRUTURA] {a} | PUT bloqueado: fundo de range", C.Y))
                        continue

                    # Exige pelo menos 1 confluência estrutural
                    if not (trend_ok or bos_ok or ideal_ok):
                        log.info(paint(f"[⛔ ESTRUTURA] {a} | PUT bloqueado: trend_ok={trend_ok} bos_ok={bos_ok} ideal_ok={ideal_ok}", C.Y))
                        continue

                    # ===== MOMENTUM (15 velas + estrutura últimas 3) =====
                    momo = analyze_recent_momentum(df, atr_val)
                    is_continuation = "MARUBOZU" in nome
                    if is_continuation and not (momo.get("continuation_ok", False) and momo.get("last3_cont_put", False)):
                        log.info(paint(f"[⛔ MOMENTUM] {a} | PUT bloqueado: sem continuação/3velas ({momo.get('reason','')})", C.Y))
                        continue
                    if (not is_continuation) and momo.get("direction") == "BULLISH" and not (momo.get("reversal_ok_put", False) and momo.get("last3_rev_put", False)):
                        log.info(paint(f"[⛔ MOMENTUM] {a} | PUT bloqueado: sem exaustão/3velas ({momo.get('reason','')})", C.Y))
                        continue

                    log.info(paint(f"[🎯 PADRÃO] {a} | {nome} PUT ({conf*100:.0f}%) - Verificando CNN...", C.B))

                    # ===== NOVO PIPELINE: CNN + REGIME FILTER + RISK CONTROL =====
                    payout = obter_payout_ativo(iq, a) if 'obter_payout_ativo' in dir() else 80
                    pipeline_ok, pipeline_dir, pipeline_details = new_pipeline_decide(iq, a, df, atr_val, payout)

                    if not pipeline_ok or pipeline_dir != "PUT":
                        log.info(paint(f"[CNN] {a} | PUT bloqueado pelo pipeline: {pipeline_dir}", C.Y))
                        continue

                    log.info(paint(f"[✅ PIPELINE] {a} | {nome} PUT ({conf*100:.0f}%) - APROVADO!", C.G))
                    log.info(paint(f"[📈 ULTIMAS 10 VELAS] {a} | {format_last_candles(df, 10)}", C.B))

                    # Cria setup com trade=True
                    setup_pattern = {
                        "trade": True,
                        "dir": "PUT",
                        "score": conf * 100,
                        "ativo": a,
                        "padrao": nome,
                        "confiabilidade": conf,
                        "pipeline": pipeline_details
                    }
                    cand_trade = (conf * 100, a, setup_pattern, float(atr_val))
                    if best_trade is None or cand_trade[0] > best_trade[0]:
                        best_trade = cand_trade
                    continue  # Já encontrou padrão, não precisa analisar pernada
                    
            except Exception as e:
                log.warning(f"[PATTERN] Erro ao analisar {a}: {e}")

        # Se não encontrou padrão, usa setup tradicional
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
def enviar_ordem(iq: IQ_Option, ativo: str, direcao: str, stake: float, expiracao: int = None) -> Optional[Tuple[str, int]]:
    """
    Envia ordem para a corretora.
    expiracao: tempo de expiração em minutos (se None, usa EXP_FIXA)
    """
    d = "call" if direcao == "CALL" else "put"
    valor = float(max(VALOR_MINIMO, stake))
    exp_time = expiracao if expiracao else int(EXP_FIXA)

    # TURBO
    try:
        ok, op_id = safe_call(iq, iq.buy, valor, ativo, d, exp_time)
        if ok and op_id:
            return ("turbo", int(op_id))
        log.warning(paint(f"[ORDEM-FAIL] TURBO ok={ok} op_id={op_id}", C.Y))
    except Exception as e:
        log.warning(paint(f"[ORDEM-EXC] TURBO {e}", C.Y))

    # DIGITAL
    try:
        ok, op_id = safe_call(iq, iq.buy_digital_spot, ativo, valor, d, exp_time)
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
    # Aviso simples: operar envolve risco.
    iq: Optional[IQ_Option] = None
    iq = ensure_connected(iq)

    log.info("Iniciando: Pernada B (M1) | Anti-lateral + SR forte + IA Contextual MELHORADA | EXECUÇÃO ON")

    stats = _safe_load_json(AI_STATS_FILE) if IA_ON else {"meta": {"total": 0, "global_wins": 0, "global_losses": 0}, "arms": {}, "patterns": {}, "recent_trades": {}}
    if IA_ON:
        log.info(f"🧠 IA=ON | file={AI_STATS_FILE}")
        log.info(f"   └─ min_samples={AI_MIN_SAMPLES} | min_prob={AI_MIN_PROB:.2f} | min_wr={AI_MIN_WINRATE:.0%}")
        log.info(f"   └─ loss_weight={AI_LOSS_WEIGHT}x | consec_loss_block={AI_CONSECUTIVE_LOSS_BLOCK}")
        log.info(f"   └─ adaptive_prior={AI_ADAPTIVE_PRIOR} | decay_rate={AI_DECAY_RATE}")
        # Exibe status inicial da IA
        ai_log_status(stats)

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
                    f"[IA] {at} | IA analisando mercado - aguardando melhor momento",
                    C.Y
                ))
                cooldown[at] = time.time()
            else:
                log.info(paint("[IA] Analisando mercado - nenhuma oportunidade no momento", C.Y))

            wait_for_next_open(TF_M1)
            continue

        score, ativo, setup, atr_val = best_trade
        score = float(score)

        if score < GATE_SOFT_SCORE:
            log.info(paint(
                f"[IA] {ativo} | IA aguardando - condições não ideais",
                C.Y
            ))
            wait_for_next_open(TF_M1)
            cooldown[ativo] = time.time()
            continue

        if score < GATE_MIN_SCORE:
            log.info(paint(
                f"[IA] {ativo} | IA analisando - buscando melhor entrada",
                C.B
            ))
            wait_for_next_open(TF_M1)
            cooldown[ativo] = time.time()
            continue

        final_dir = str(setup["dir"])
        log.info(paint(
            f"[IA] {ativo} -> {final_dir} | IA identificou possível entrada",
            dir_color(final_dir)
        ))

        # ===================== IA FILTRO =====================
        if IA_ON:
            pred = ai_predict(ativo, setup, stats)
            prob = float(pred["prob"])
            bayes = float(pred["bayes"])
            ucb01 = float(pred["ucb01"])
            conf = float(pred["conf"])
            n_arm = int(pred["n_arm"])

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

        # ===================== VALIDAÇÃO DE MOMENTUM DE CURTO PRAZO (1-5 MIN) =====================
        # Obtém dados frescos para validação de momentum
        df_fresh = get_candles_df(iq, ativo, TF_M1, N_M1, end_ts=end_ts_closed(TF_M1))
        if df_fresh is None:
            log.warning(f"[{ativo}] Falha ao obter dados para validação de momentum")
            wait_for_next_open(TF_M1)
            cooldown[ativo] = time.time()
            continue
        
        # ===== COLETA TODAS AS ANÁLISES PRIMEIRO =====
        momentum_check = validate_short_term_momentum(df_fresh, final_dir, atr_val)
        trend_check = validate_immediate_trend(df_fresh, final_dir, atr_val)
        
        # ===== PROJEÇÃO INTELIGENTE DE EXPIRAÇÃO =====
        # A IA analisa para 1, 2, 3 e 5 minutos e escolhe a melhor
        exp_projection = smart_expiration_projection(df_fresh, final_dir, atr_val)
        best_exp = exp_projection.get("best_expiration", EXP_FIXA)
        exp_confidence = exp_projection.get("confidence", 0.5)
        exp_reason = exp_projection.get("reason", "default")
        projected_wins = exp_projection.get("projected_win", {})
        
        # Log da projeção de expiração
        volatility = exp_projection.get("volatility", 0)
        vol_level = "ALTA" if volatility > 0.3 else ("MÉDIA" if volatility > 0.15 else "BAIXA")
        log.info(paint(
            f"[🎯 IA] {ativo} | Exp={best_exp}min | Vol={volatility:.2f}% ({vol_level}) | Wins: 1m={projected_wins.get(1,0):.0f}% 2m={projected_wins.get(2,0):.0f}% 3m={projected_wins.get(3,0):.0f}% 5m={projected_wins.get(5,0):.0f}%",
            C.B
        ))
        
        projection_check = analyze_trend_projection(df_fresh, final_dir, atr_val, best_exp)
        chart_analysis = professional_chart_analysis(df_fresh, final_dir, atr_val, best_exp)
        
        # ===== USA SISTEMA DE CONFLUENCIA PARA 80%+ WIN RATE =====
        ai_prediction_data = {}
        if IA_ON:
            ai_prediction_data = {
                "prob": float(pred.get("prob", 0)),
                "bayes": float(pred.get("bayes", 0)),
                "ucb01": float(pred.get("ucb01", 0)),
                "conf": float(pred.get("conf", 0)),
                "n_arm": int(pred.get("n_arm", 0))
            }

        # NOVO: Usa sistema de confluencia (exige 4/6 criterios)
        decision = smart_entry_decision_v2(
            direction=final_dir,
            setup=setup,
            df_m1=df_fresh,
            atr_val=atr_val,
            ai_stats=stats if IA_ON else None,
            ai_prediction=ai_prediction_data
        )

        # LOG do sistema de confluencia
        if decision.get("confluences", 0) > 0:
            log.info(paint(
                f"[CONFLUENCIA] {ativo} | {decision.get('confluences', 0):.1f}/6 criterios atendidos",
                C.B
            ))
        
        # ===== LOG DETALHADO DA DECISÃO =====
        decision_score = decision.get("final_score", 0)
        decision_reasons = decision.get("reasons", [])
        decision_blocks = decision.get("blocks", [])
        decision_warnings = decision.get("warnings", [])
        allow_entry = decision.get("allow_entry", False)
        is_consolidating = decision.get("is_consolidating", False)
        market_type = decision.get("market_type", "INDEFINIDO")
        best_setup = decision.get("best_setup", "NONE")
        
        # NOVO: Se o padrão de vela indicar direção diferente, usar a do padrão!
        override_dir = decision.get("override_direction")
        if override_dir and override_dir != final_dir:
            log.info(paint(
                f"[PADROES] {ativo} | Direcao invertida pelo padrao: {final_dir} -> {override_dir}",
                C.Y
            ))
            final_dir = override_dir
        
        # Cor baseada na decisão
        dec_color = C.G if allow_entry else C.R
        
        # LOG DO CONTEXTO DO MERCADO (NOVO!)
        log.info(paint(
            f"[📊 CONTEXTO] {ativo} | Mercado: {market_type} | Setup: {best_setup}",
            C.B
        ))
        
        # LOG ESPECIAL PARA CONSOLIDAÇÃO
        if is_consolidating:
            log.info(paint(
                f"[⏸️ IA] {ativo} | CONSOLIDAÇÃO DETECTADA | Aguardando ROMPIMENTO...",
                C.Y
            ))
        
        # Mensagem amigável da IA
        if allow_entry:
            log.info(paint(
                f"[🧠 IA] {ativo} {final_dir} | Setup: {best_setup} | Confiança: {decision_score:.0f}%",
                dec_color
            ))
            if decision_reasons:
                log.info(paint(f"[IA] Motivos: {', '.join(decision_reasons[:4])}", C.G))
        else:
            reason_text = decision.get("decision_reason", "aguardando")
            log.info(paint(
                f"[🧠 IA] {ativo} | IA aguardando | {reason_text}",
                dec_color
            ))
        
        # ===== SE NÃO APROVADO, PULA =====
        if not allow_entry:
            # Mostrar motivo do bloqueio para debug
            block_reason = decision.get("decision_reason", "desconhecido")
            if decision_warnings:
                log.info(paint(f"[IA] {ativo} | Avisos: {decision_warnings[:3]}", C.Y))
            wait_for_next_open(TF_M1)
            cooldown[ativo] = time.time()
            continue
        
        # ===== TUDO ALINHADO - ENTRADA APROVADA =====
        log.info(paint(
            f"[✅ IA ANALISOU] {ativo} {final_dir} | {best_setup} | Confiança: {decision_score:.0f}% | Expiração: {best_exp}min",
            C.G
        ))

        # entra na abertura do próximo M1
        wait_for_next_open(TF_M1)

        # STAKE DINÂMICO BASEADO NA BANCA (NOVO)
        stake = calcular_stake_dinamico(iq, STAKE_FIXA)
        log.info(paint(f"[{ativo}] Valor: ${stake:.2f} | Expiração: {best_exp} minutos", C.B))

        # Usa a expiração inteligente calculada pela IA
        op = enviar_ordem(iq, ativo, final_dir, stake, best_exp)

        if not op:
            log.error(paint(f"[{ativo}] ❌ Erro ao enviar ordem - tentando novamente", C.R))
            cooldown[ativo] = time.time()
            continue

        op_type, op_id = op
        log.info(paint(
            f"[{ativo}] ✅ ORDEM {final_dir} | Exp: {best_exp}min | Valor: ${stake:.2f} | Win proj: {projected_wins.get(best_exp, 50):.0f}%",
            dir_color(final_dir)
        ))

        # Registra trade no Risk Control
        if RISK_CONTROL_AVAILABLE and risk_control:
            risk_control.on_trade_opened()

        res = wait_result(iq, op_type, op_id)

        total += 1
        if res > 0:
            wins += 1
            log.info(paint(f"[{ativo}] ✅ WIN {res:.2f}$", C.G))
            
            # ===================== ENVIAR WIN PARA FIREBASE =====================
            # Envia dados completos do WIN para análise e identificação de padrões vencedores
            try:
                # Extrai informações do chart_analysis
                win_probability = float(chart_analysis.get("win_probability", 50.0))
                chart_confidence = float(chart_analysis.get("confidence", 0.5))
                pattern_info = chart_analysis.get("pattern", {})
                structure_info = chart_analysis.get("structure", {})
                recommendation = chart_analysis.get("recommendation", "NEUTRAL")
                
                # Adiciona análise gráfica profissional aos dados
                chart_analysis_data = {
                    "win_probability": win_probability,
                    "confidence": chart_confidence,
                    "pattern": pattern_info,
                    "structure": structure_info,
                    "recommendation": recommendation,
                    "reasons": chart_analysis.get("reasons", []),
                    "warnings": chart_analysis.get("warnings", [])
                }
                
                send_win_to_firebase(
                    iq=iq,
                    order_id=op_id,
                    ativo=ativo,
                    direction=final_dir,
                    stake=stake,
                    profit=res,
                    setup=setup,
                    momentum_data=momentum_check,
                    trend_data=trend_check,
                    projection_data=projection_check,
                    ai_prediction=ai_prediction_data if IA_ON else None,
                    chart_analysis=chart_analysis_data
                )
            except Exception as e:
                log.warning(f"[FIREBASE] Erro ao enviar win: {e}")

            # ===================== ATUALIZA CNN (NOVO SISTEMA) =====================
            try:
                on_trade_result_cnn(ativo, final_dir, win=True, df_m1=df_fresh)
            except Exception as e:
                log.debug(f"[CNN] Erro ao registrar win: {e}")

        elif res < 0:
            log.info(paint(f"[{ativo}] ❌ LOSS {res:.2f}$", C.R))
            
            # ===================== ENVIAR LOSS PARA FIREBASE =====================
            # Envia dados completos do loss para análise futura e aprendizado
            try:
                # Extrai informações do chart_analysis
                win_probability = float(chart_analysis.get("win_probability", 50.0))
                chart_confidence = float(chart_analysis.get("confidence", 0.5))
                pattern_info = chart_analysis.get("pattern", {})
                structure_info = chart_analysis.get("structure", {})
                recommendation = chart_analysis.get("recommendation", "NEUTRAL")
                
                # Adiciona análise gráfica profissional aos dados
                chart_analysis_data = {
                    "win_probability": win_probability,
                    "confidence": chart_confidence,
                    "pattern": pattern_info,
                    "structure": structure_info,
                    "recommendation": recommendation,
                    "reasons": chart_analysis.get("reasons", []),
                    "warnings": chart_analysis.get("warnings", [])
                }
                
                send_loss_to_firebase(
                    iq=iq,
                    order_id=op_id,
                    ativo=ativo,
                    direction=final_dir,
                    stake=stake,
                    setup=setup,
                    momentum_data=momentum_check,
                    trend_data=trend_check,
                    projection_data=projection_check,
                    ai_prediction=ai_prediction_data if IA_ON else None,
                    chart_analysis=chart_analysis_data
                )
            except Exception as e:
                log.warning(f"[FIREBASE] Erro ao enviar loss: {e}")

            # ===================== ATUALIZA CNN (NOVO SISTEMA) =====================
            try:
                on_trade_result_cnn(ativo, final_dir, win=False, df_m1=df_fresh)
            except Exception as e:
                log.debug(f"[CNN] Erro ao registrar loss: {e}")

        else:
            log.info(paint(f"[{ativo}] ⚪ EMPATE {res:.2f}$", C.B))

        # update IA após resultado
        if IA_ON:
            # NOVO: Adiciona padrões detectados ao setup para aprendizado!
            # Isso permite que a IA aprenda quais padrões funcionam
            detected_patterns = []
            
            # Padrões do chart_analysis
            pattern_info = chart_analysis.get("pattern", {})
            if pattern_info.get("name") and pattern_info.get("name") != "none":
                detected_patterns.append({
                    "name": pattern_info.get("name", ""),
                    "direction": pattern_info.get("direction", ""),
                    "strength": pattern_info.get("strength", 0),
                    "reliability": pattern_info.get("reliability", 0)
                })
            
            # Padrões do decision (se houver)
            decision_patterns = decision.get("detected_patterns", [])
            for p in decision_patterns:
                if isinstance(p, dict) and p.get("nome"):
                    detected_patterns.append({
                        "name": p.get("nome", ""),
                        "direction": p.get("direcao", ""),
                        "strength": p.get("confiabilidade", 0),
                        "reliability": p.get("confiabilidade", 0)
                    })
            
            # Adiciona ao setup para o ai_make_key usar
            setup["patterns"] = detected_patterns
            setup["final_direction"] = final_dir
            setup["result"] = "win" if res > 0 else "loss"
            
            ai_update(ativo, setup, res, stats)
            _safe_save_json(AI_STATS_FILE, stats)
            
            # LOG STATUS DA IA A CADA 5 TRADES (NOVO)
            if total % 5 == 0:
                ai_log_status(stats)

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
