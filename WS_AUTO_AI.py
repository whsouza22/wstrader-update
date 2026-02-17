# -*- coding: utf-8 -*-
"""
WS_AUTO_AI — DOM Forex Perfect Zones (M1) com:
✅ Candles FECHADOS (evita sinal fora da hora)
✅ Anti-lateral + Anti-esticado
✅ Filtro de SUPORTE/RESISTÊNCIA FORTE (usa >=200 velas e considera várias regiões)
✅ IA ENSEMBLE: Bayesiano + LightGBM (Gradient Boosting) para decisões mais inteligentes
✅ Execução real (TURBO -> DIGITAL fallback)

Requisitos:
pip install iqoptionapi pandas numpy lightgbm
"""

import os
import time
import math
import json
import logging
import pickle
import threading
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# ===================== MULTI-BROKER SUPPORT =====================
# Detecta qual corretora usar via variável de ambiente BROKER_TYPE
BROKER_TYPE = os.getenv("BROKER_TYPE", "iq_option").lower().strip()

if BROKER_TYPE == "bullex":
    from bullexapi.stable_api import Bullex as BrokerAPI
    _BROKER_NAME = "Bullex"
elif BROKER_TYPE == "casatrader":
    from casatraderapi.stable_api import Casa_Trader as BrokerAPI
    _BROKER_NAME = "CasaTrader"
else:
    from iqoptionapi.stable_api import IQ_Option as BrokerAPI
    _BROKER_NAME = "IQ Option"
    BROKER_TYPE = "iq_option"

# DOM Forex Strategy (Perfect Zones) — para IQ Option
from dom_forex_strategy import dom_forex_signal

# LightGBM para Gradient Boosting
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LGBM_AVAILABLE = False

# Claude API Calibrator - DESATIVADO (IA local é suficiente)
# Para reativar: mudar CLAUDE_CALIBRATE_ON para True
try:
    from ai_claude_calibrator import calibrate_after_backtest, CLAUDE_AVAILABLE, CLAUDE_CALIBRATE_ON
    CLAUDE_CALIBRATE_ON = False  # Forçado OFF - IA local faz calibração melhor
except ImportError:
    CLAUDE_AVAILABLE = False
    CLAUDE_CALIBRATE_ON = False
    def calibrate_after_backtest(*a, **kw): return None

# CNN Pattern Detector (leve, sem TensorFlow)
try:
    from cnn_pattern_detector import get_cnn, LightCNN
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False
    def get_cnn(*a, **kw): return None

# ===================== CONFIG =====================
# Credenciais por broker
if BROKER_TYPE == "bullex":
    EMAIL = os.getenv("BULLUX_EMAIL", "") or os.getenv("IQ_EMAIL", "")
    SENHA = os.getenv("BULLUX_PASS", "") or os.getenv("IQ_PASS", "") or os.getenv("IQ_PASSWORD", "")
    CONTA = os.getenv("BULLUX_CONTA", "PRACTICE")
elif BROKER_TYPE == "casatrader":
    EMAIL = os.getenv("CASATRADER_EMAIL", "") or os.getenv("IQ_EMAIL", "")
    SENHA = os.getenv("CASATRADER_PASS", "") or os.getenv("IQ_PASS", "") or os.getenv("IQ_PASSWORD", "")
    CONTA = os.getenv("CASATRADER_CONTA", "PRACTICE")
else:
    EMAIL = os.getenv("IQ_EMAIL", "")
    SENHA = os.getenv("IQ_PASS", "") or os.getenv("IQ_PASSWORD", "")
    CONTA = os.getenv("IQ_CONTA", "PRACTICE")

TF_M1 = 60
DECIDIR_ANTES_FECHAR_SEC = int(os.getenv("WS_DECIDIR_ANTES_FECHAR", "10"))
N_M1 = int(os.getenv("WS_N_M1", "340"))

PAYOUT_MINIMO = int(os.getenv("WS_PAYOUT_MIN", "80"))
NUM_ATIVOS = int(os.getenv("WS_NUM_ATIVOS", "12"))
PAYOUT_REFRESH_SEC = int(os.getenv("WS_PAYOUT_REFRESH", "180"))

EXP_FIXA = int(os.getenv("WS_EXP_MIN", "1"))
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
RETRAIN_ON_LOSS = (os.getenv("WS_RETRAIN_ON_LOSS", "0").strip() == "1")  # DESATIVADO: usar backtest em vez de retreino
BACKTEST_ON_LOSS = (os.getenv("WS_BACKTEST_ON_LOSS", "1").strip() == "1")  # ATIVADO: backtest 30min após LOSS para recalibrar
PAUSE_AFTER_LOSS_SECONDS = int(os.getenv("WS_PAUSE_AFTER_LOSS", "60"))  # Pausa de 1 min após loss antes de continuar
RETRAIN_PENALTY = float(os.getenv("WS_RETRAIN_PENALTY", "0.25"))  # Penalidade ao retreinar padrão

# ===================== IA (ONLINE) - APRENDIZADO ADAPTATIVO =====================
IA_ON = (os.getenv("WS_AI_ON", "1").strip() == "1")  # LIGADO: aprende bloqueando losses
# Arquivos por broker para não conflitar
_broker_suffix = {"iq_option": "m1", "bullex": "bullex", "casatrader": "casatrader"}.get(BROKER_TYPE, "m1")
AI_STATS_FILE = os.getenv("WS_AI_FILE", f"ws_ai_stats_{_broker_suffix}.json")
AI_MIN_SAMPLES = int(os.getenv("WS_AI_MIN_SAMPLES", "15"))   # 15 trades para começar a bloquear
AI_MIN_PROB = float(os.getenv("WS_AI_MIN_PROB", "0.55"))     # probabilidade mínima (bayesiana) - AUMENTADO para 55%
AI_MIN_WINRATE = float(os.getenv("WS_AI_MIN_WINRATE", "0.42"))  # bloqueia se winrate < 42%
AI_CONF_MIN = float(os.getenv("WS_AI_CONF_MIN", "0.50"))     # confiança mínima na decisão

# ===================== LIGHTGBM ENSEMBLE =====================
LGBM_ON = (os.getenv("WS_LGBM_ON", "1").strip() == "1") and LGBM_AVAILABLE  # LightGBM ativo
LGBM_MODEL_FILE = os.getenv("WS_LGBM_FILE", f"ws_lgbm_model_{_broker_suffix}.pkl")
LGBM_DATA_FILE = os.getenv("WS_LGBM_DATA", f"ws_lgbm_data_{_broker_suffix}.json")
LGBM_N_FEATURES = 14  # Número esperado de features (score,retr,A_atr,effA,flips,pb_len,distBreak,late_ext,compression,market_quality,entry_conf,ctx_score,dir_enc,rsi_norm)
LGBM_MIN_SAMPLES = int(os.getenv("WS_LGBM_MIN_SAMPLES", "30"))  # Mínimo de amostras para treinar
LGBM_RETRAIN_EVERY = int(os.getenv("WS_LGBM_RETRAIN", "10"))   # Retreina a cada N trades
LGBM_MIN_PROB = float(os.getenv("WS_LGBM_MIN_PROB", "0.58"))   # Probabilidade mínima do LGBM - balanceado
LGBM_WARMUP_PROB = float(os.getenv("WS_LGBM_WARMUP_PROB", "0.55"))  # Threshold durante warmup (55% - balanceado)
ENSEMBLE_MODE = os.getenv("WS_ENSEMBLE_MODE", "weighted")  # "weighted" = média ponderada (mais confiável)

# ===================== CNN PATTERN DETECTOR =====================
# CNN DESATIVADO - não tem dados suficientes e retorna sempre 0.50
# Para reativar: mudar "0" para "1" abaixo
CNN_ON = False  # Forçado OFF - sem dados/amostras para ser útil
CNN_MIN_PROB = float(os.getenv("WS_CNN_MIN_PROB", "0.55"))    # Prob mínima CNN para confirmar
CNN_WEIGHT = float(os.getenv("WS_CNN_WEIGHT", "0.20"))         # Peso da CNN no ensemble (20%)
CNN_VETO_THRESHOLD = float(os.getenv("WS_CNN_VETO", "0.30"))   # CNN < 0.30 = veto (bloqueia trade)

# ===================== FILTROS DE QUALIDADE ENSEMBLE =====================
# Exigência mínima de ensemble baseada na qualidade do contexto
ENS_MIN_CTX_RUIM = float(os.getenv("WS_ENS_MIN_CTX_RUIM", "0.65"))  # ctx < 0.40 precisa ensemble >= 0.65 (MAIS RIGOROSO)
ENS_MIN_CTX_MED  = float(os.getenv("WS_ENS_MIN_CTX_MED",  "0.60"))  # ctx 0.40-0.50 precisa ensemble >= 0.60
ENS_MIN_CTX_BOM  = float(os.getenv("WS_ENS_MIN_CTX_BOM",  "0.55"))  # ctx >= 0.50 precisa ensemble >= 0.55

# ===================== MODO DA IA =====================
# "learning" = IA tem controle total, filtros de score relaxados (mais trades, aprende mais rápido)
# "strict"   = IA + filtros rigorosos (menos trades, mais conservador)
IA_MODE = os.getenv("WS_IA_MODE", "learning").strip().lower()  # PADRÃO: learning

# ===================== FILTROS DE QUALIDADE EXTRA =====================
# REQUIRE_TRENDLINE: SEMPRE ativado - tendência confirmada é essencial
_trendline_default = "0"  # Desativado: LTB/LTA agora é confluência, não filtro obrigatório
REQUIRE_TRENDLINE = (os.getenv("WS_REQUIRE_TRENDLINE", _trendline_default).strip() == "1")
# USE_TRENDLINE_FILTER: variável dinâmica que será ajustada pelo backtest
USE_TRENDLINE_FILTER = REQUIRE_TRENDLINE  # Começa com o valor de REQUIRE_TRENDLINE
# MIN_ENTRY_EFF: direcionalidade mínima do mercado (DOM Forex: net_move/total_move ~0.05-0.25 em OTC)
MIN_ENTRY_EFF = float(os.getenv("WS_MIN_ENTRY_EFF", "0.10"))  # 10% — compatível com DOM Forex em OTC M1
# MIN_CONFLUENCE: mínimo de fatores confluentes para operar
MIN_CONFLUENCE = int(os.getenv("WS_MIN_CONFLUENCE", "2"))  # NOVO: mínimo 2 confluências
# BACKTEST_MIN_WINRATE: taxa mínima do backtest para operar (evita mercado ruim)
MIN_CONFLUENCE = int(os.getenv("WS_MIN_CONFLUENCE", "2"))  # Mínimo 2 confluências para entrar
BACKTEST_MIN_WINRATE = float(os.getenv("WS_BACKTEST_MIN_WINRATE", "0.52"))  # 52% mínimo (acima de sorte/50%)

# ===================== AUTO-AJUSTE DE FILTROS =====================
# Contadores para auto-ajuste quando muitos skips consecutivos
MAX_CONSECUTIVE_SKIPS = int(os.getenv("WS_MAX_CONSEC_SKIPS", "8"))  # Após 8 skips, relaxa filtros
AUTO_RELAX_ON_SKIPS = (os.getenv("WS_AUTO_RELAX", "0").strip() == "1")  # DESATIVADO: não relaxa trendline

# ===================== IA APRENDE COM BACKTEST (HÍBRIDO COM HISTÓRICO) =====================
# Se ativado, a IA (Bayesian + LightGBM) também aprende com os sinais simulados do backtest
AI_LEARN_FROM_BACKTEST = (os.getenv("WS_AI_LEARN_BACKTEST", "1").strip() == "1")  # ATIVADO: IA aprende com backtest
AI_BACKTEST_WEIGHT = float(os.getenv("WS_AI_BACKTEST_WEIGHT", "0.5"))  # Peso do backtest (0.5 = metade do peso de trades reais)

# HISTÓRICO DE BACKTEST (acumula com decay temporal)
BACKTEST_HISTORY_FILE = os.getenv("WS_BACKTEST_HISTORY_FILE", f"ws_backtest_history_{_broker_suffix}.json")
BACKTEST_HISTORY_MAX_SAMPLES = int(os.getenv("WS_BACKTEST_MAX_SAMPLES", "500"))  # Máximo de amostras no histórico
BACKTEST_HISTORY_DECAY_HOURS = float(os.getenv("WS_BACKTEST_DECAY_HOURS", "6.0"))  # Após 6h, peso começa a decair
BACKTEST_HISTORY_MIN_WEIGHT = float(os.getenv("WS_BACKTEST_MIN_WEIGHT", "0.2"))  # Peso mínimo de amostras antigas (20%)
BACKTEST_USE_ACCUMULATED = (os.getenv("WS_BACKTEST_ACCUMULATE", "1").strip() == "1")  # ATIVADO: usa histórico acumulado


# ===================== SCORE (DEPENDE DO IA_MODE) =====================
# Ambos os modos agora exigem confluência e tendência forte
if IA_MODE == "learning":
    GATE_MIN_SCORE = float(os.getenv("WS_GATE_MIN", "0.42"))   # Score mínimo (Dom Forex: zona + confluência)
    GATE_SOFT_SCORE = float(os.getenv("WS_GATE_SOFT", "0.35")) # Soft skip — mais entradas
    GATE_CONTEXT_BAD_BLOCK = True  # BLOQUEIA se contexto for ruim
    GATE_CONTEXT_VERY_BAD = 0.20  # Contexto mínimo aceitável (relaxado — backtest ajusta)
else:  # strict
    GATE_MIN_SCORE = float(os.getenv("WS_GATE_MIN", "0.60"))   # Score mínimo rigoroso
    GATE_SOFT_SCORE = float(os.getenv("WS_GATE_SOFT", "0.50")) # Soft skip rigoroso
    GATE_CONTEXT_BAD_BLOCK = True  # Bloquear se contexto for ruim
    GATE_CONTEXT_VERY_BAD = 0.40  # Limiar de contexto mais rigoroso

# ===================== ANTI-SPIKE =====================
SPIKE_RANGE_ATR = float(os.getenv("WS_SPIKE_RANGE_ATR", "1.35"))
SPIKE_WICK_FRAC = float(os.getenv("WS_SPIKE_WICK_FRAC", "0.62"))
SPIKE_COOLDOWN_MIN = int(os.getenv("WS_SPIKE_COOLDOWN_MIN", "6"))

# ===================== FILTRO S/R FORTE (AJUSTADO) =====================
SR_LOOKBACK = int(os.getenv("WS_SR_LOOKBACK", "220"))
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
cooldown_loss: Dict[str, float] = {}  # Cooldown após LOSS em ativo específico
consecutive_losses: Dict[str, int] = {}  # Contador de losses consecutivos por ativo
global_consecutive_losses: int = 0  # Losses consecutivos globais

# ===================== FILTROS POR ATIVO =====================
# Cada ativo tem seus próprios filtros calibrados pelo backtest
filtros_por_ativo: Dict[str, Dict[str, Any]] = {}
# Estrutura: {
#   "EURUSD-OTC": {"min_ctx": 0.40, "min_score": 0.55, "taxa": 0.65, "sinais": 8, "habilitado": True},
#   ...
# }

# Ativos que já foram analisados no backtest (para detectar mudanças)
ativos_analisados_backtest: List[str] = []

# LightGBM globals
lgbm_model: Any = None  # Modelo LightGBM treinado
lgbm_data: List[Dict] = []  # Dados de treino acumulados
lgbm_trade_count: int = 0  # Contador para retreino
lgbm_reliable: bool = False  # Se True, modelo LGBM tem validação >= 55% (confiável)
lgbm_val_accuracy: float = 0.0  # Última accuracy de validação medida

# CNN Pattern Detector globals
cnn_model: Any = None  # Instância do LightCNN

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
        if BROKER_TYPE == "bullex":
            from bullexapi.ws.client import WebsocketClient
        elif BROKER_TYPE == "casatrader":
            from casatraderapi.ws.client import WebsocketClient
        else:
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

# ===================== CONEXÃO COM CORRETORA =====================
def conectar_iq(max_retries: int = 5) -> BrokerAPI:
    if not EMAIL or not SENHA:
        raise RuntimeError(f"Defina credenciais para {_BROKER_NAME} nas variáveis de ambiente.")
    patch_websocket_on_close()
    
    for attempt in range(1, max_retries + 1):
        try:
            log.info(f"Conectando à {_BROKER_NAME}... (tentativa {attempt}/{max_retries})")
            iq = BrokerAPI(EMAIL, SENHA)
            iq.connect()

            for _ in range(15):
                if iq.check_connect():
                    break
                time.sleep(1.5)

            if not iq.check_connect():
                raise ConnectionError("check_connect() retornou False")

            iq.change_balance(CONTA)
            try:
                log.info(f"Conectado | Saldo: {iq.get_balance():.2f} | Conta: {CONTA}")
            except Exception:
                log.info(f"Conectado | Conta: {CONTA}")

            return iq
        except Exception as e:
            log.warning(paint(f"Tentativa {attempt}/{max_retries} falhou: {e}", C.Y))
            if attempt < max_retries:
                wait_time = min(10 * attempt, 30)  # 10s, 20s, 30s, 30s...
                log.info(paint(f"Aguardando {wait_time}s antes de tentar novamente...", C.Y))
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Falha na conexão após {max_retries} tentativas: {e}")

def ensure_connected(iq: Optional[BrokerAPI]) -> BrokerAPI:
    if iq is None:
        return conectar_iq()
    try:
        if iq.check_connect():
            return iq
    except Exception:
        pass

    log.info(paint("Conexão instável. Reconectando...", C.Y))
    for attempt in range(1, 4):  # 3 tentativas de reconexão rápida
        try:
            iq.connect()
            for _ in range(15):
                if iq.check_connect():
                    iq.change_balance(CONTA)
                    log.info(paint("Reconectado com sucesso.", C.G))
                    return iq
                time.sleep(1.5)
        except Exception as e:
            log.warning(paint(f"Reconexão rápida {attempt}/3 falhou: {e}", C.Y))
            time.sleep(3 * attempt)  # 3s, 6s, 9s

    log.warning(paint("Reconexão rápida falhou. Criando nova conexão...", C.Y))
    return conectar_iq()

def _call_with_timeout(fn, timeout_sec, *args, **kwargs):
    """Executa uma função com timeout. Retorna (resultado, sucesso)."""
    result = [None]
    error = [None]
    completed = threading.Event()
    
    def _worker():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as e:
            error[0] = e
        finally:
            completed.set()
    
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    
    if completed.wait(timeout=timeout_sec):
        if error[0]:
            raise error[0]
        return result[0]
    else:
        log.warning(paint(f"⏱️ TIMEOUT ({timeout_sec}s) em {fn.__name__ if hasattr(fn, '__name__') else fn}", C.R))
        raise TimeoutError(f"Timeout de {timeout_sec}s em {fn}")

def safe_call(iq: BrokerAPI, fn, *args, timeout=45, **kwargs):
    try:
        return _call_with_timeout(fn, timeout, *args, **kwargs)
    except TimeoutError:
        log.warning(paint(f"⏱️ safe_call timeout ({timeout}s) em {getattr(fn, '__name__', '?')} - reconectando...", C.Y))
        try:
            ensure_connected(iq)
        except Exception:
            pass
        try:
            return _call_with_timeout(fn, timeout, *args, **kwargs)
        except TimeoutError:
            log.warning(paint(f"⏱️ safe_call: segundo timeout - retornando None", C.Y))
            return None
        except Exception:
            return None
    except Exception as e:
        msg = str(e).lower()
        if ("10054" in msg) or ("forçado o cancelamento" in msg) or ("goodbye" in msg) or ("10053" in msg):
            log.warning(paint(f"Erro de conexão (recuperável): {e}", C.Y))
            try:
                ensure_connected(iq)
                return fn(*args, **kwargs)
            except Exception:
                return None
        raise

# ===================== CANDLES =====================
def get_candles_df(iq: BrokerAPI, ativo: str, timeframe: int, n: int, end_ts: Optional[float] = None) -> Optional[pd.DataFrame]:
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
def obter_top_ativos_otc(iq: BrokerAPI) -> List[str]:
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
            payout = safe_call(iq, iq.get_digital_payout, a, 10, timeout=15)
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

# ===================== GESTÃO DE BANCA =====================
def calcular_stake_dinamico(iq: BrokerAPI, base_stake: float) -> float:
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
    """
    Extrai features numéricas do setup para o LightGBM.
    Features:
    - score, retr, A_atr, effA, flips, pb_len, distBreak
    - market_quality, late_ext, compression
    - dir_encoded (CALL=1, PUT=-1)
    - rsi (RSI confluência)
    """
    # Features básicas do setup
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
    rsi = float(setup.get("rsi", 50.0))
    
    # Contexto do setup
    ctx = str(setup.get("ctx", "neutro"))
    ctx_score = 1.0 if ctx == "bom" else (0.5 if ctx == "neutro" else 0.0)
    
    # Direção encodada
    dir_str = str(setup.get("dir", "NEUTRAL"))
    dir_enc = 1.0 if dir_str == "CALL" else (-1.0 if dir_str == "PUT" else 0.0)
    
    # RSI normalizado (0-1 range para LGBM)
    rsi_norm = rsi / 100.0
    
    features = np.array([
        score, retr, A_atr, effA, flips, pb_len, distBreak,
        late_ext, compression, market_quality, entry_conf, ctx_score, dir_enc, rsi_norm
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
    """
    Adiciona uma amostra de treino LIVE (setup + resultado).
    result > 0 = WIN (1), result < 0 = LOSS (0), result = 0 = ignora
    """
    global lgbm_data, lgbm_trade_count
    if not LGBM_ON or result == 0:
        return
    
    features = lgbm_extract_features(setup).tolist()
    label = 1 if result > 0 else 0
    
    lgbm_data.append({"features": features, "label": label, "source": "live", "timestamp": time.time()})
    lgbm_trade_count += 1
    
    # Limita dados a últimas 1000 amostras
    if len(lgbm_data) > 1000:
        lgbm_data = lgbm_data[-1000:]
    
    lgbm_save_data()
    
    # Retreina se atingiu threshold
    if lgbm_trade_count >= LGBM_RETRAIN_EVERY and len(lgbm_data) >= LGBM_MIN_SAMPLES:
        lgbm_train()
        lgbm_trade_count = 0

def lgbm_train():
    """Treina ou retreina o modelo LightGBM com os dados acumulados."""
    global lgbm_model, lgbm_data, lgbm_reliable, lgbm_val_accuracy
    
    if not LGBM_ON or lgb is None or len(lgbm_data) < LGBM_MIN_SAMPLES:
        return
    
    try:
        # ===== LIMPEZA DE DADOS ANTIGOS =====
        # Remove amostras com mais de 12 horas - padrões OTC mudam rápido
        MAX_DATA_AGE_HOURS = 12
        cutoff_time = time.time() - (MAX_DATA_AGE_HOURS * 3600)
        
        old_count = len(lgbm_data)
        lgbm_data = [d for d in lgbm_data if d.get("timestamp", time.time()) >= cutoff_time]
        removed = old_count - len(lgbm_data)
        
        if removed > 0:
            log.info(paint(f"[LGBM] Limpeza: removidas {removed} amostras antigas (>{MAX_DATA_AGE_HOURS}h)", C.Y))
            lgbm_save_data()
        
        # Verifica se ainda tem amostras suficientes após limpeza
        if len(lgbm_data) < LGBM_MIN_SAMPLES:
            log.warning(paint(f"[LGBM] Após limpeza, apenas {len(lgbm_data)} amostras (mín={LGBM_MIN_SAMPLES}). Modelo desabilitado.", C.Y))
            lgbm_reliable = False
            lgbm_val_accuracy = 0.0
            lgbm_model = None
            return
        
        # Migração de features: amostras antigas com 13 features recebem rsi_norm=0.5 (neutro)
        for d in lgbm_data:
            if len(d["features"]) < LGBM_N_FEATURES:
                d["features"].extend([0.5] * (LGBM_N_FEATURES - len(d["features"])))
            elif len(d["features"]) > LGBM_N_FEATURES:
                d["features"] = d["features"][:LGBM_N_FEATURES]
        
        X = np.array([d["features"] for d in lgbm_data], dtype=np.float32)
        y = np.array([d["label"] for d in lgbm_data], dtype=np.int32)
        
        # Parâmetros otimizados para trading - ANTI-OVERFITTING
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 8,           # REDUZIDO: menos complexidade = menos overfitting
            "max_depth": 3,            # REDUZIDO: árvores mais rasas generalizam melhor
            "learning_rate": 0.03,     # REDUZIDO: aprendizado mais lento = mais robusto
            "n_estimators": 80,        # REDUZIDO: menos árvores
            "min_child_samples": 10,   # AUMENTADO: precisa mais exemplos por folha
            "subsample": 0.7,          # REDUZIDO: mais dropout de amostras
            "colsample_bytree": 0.7,   # REDUZIDO: mais dropout de features
            "reg_alpha": 0.5,          # AUMENTADO: mais regularização L1
            "reg_lambda": 0.5,         # AUMENTADO: mais regularização L2
            "verbose": -1,
            "force_col_wise": True,
        }
        
        lgbm_model = lgb.LGBMClassifier(**params)
        
        # VALIDAÇÃO REAL: divide dados em treino/teste para medir accuracy real
        n_samples = len(X)
        if n_samples >= 50:
            # 80/20 split - últimas amostras como validação (mais recentes)
            split_idx = int(n_samples * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            lgbm_model.fit(X_train, y_train)
            
            # Accuracy no treino
            preds_train = lgbm_model.predict(X_train)
            acc_train = (preds_train == y_train).mean() * 100
            
            # Accuracy na VALIDAÇÃO (accuracy real!)
            preds_val = lgbm_model.predict(X_val)
            acc_val = (preds_val == y_val).mean() * 100
            
            lgbm_val_accuracy = acc_val
            
            log.info(paint(f"[LGBM] Modelo treinado! Amostras={n_samples} | Treino={acc_train:.1f}% | Val={acc_val:.1f}% (real)", C.G))
            
            # ===== AUTO-DISABLE: se validação < 50%, modelo está chutando =====
            # (55% era muito restritivo com amostras pequenas - val oscila demais)
            if acc_val < 50.0:
                lgbm_reliable = False
                log.warning(paint(f"[LGBM] ⚠️ Val={acc_val:.1f}% < 50% → LGBM DESABILITADO (usando só Bayes)", C.Y))
                log.warning(paint(f"[LGBM] Modelo não consegue prever melhor que moeda. Ignorando predições LGBM.", C.Y))
            else:
                lgbm_reliable = True
                log.info(paint(f"[LGBM] ✅ Val={acc_val:.1f}% ≥ 50% → LGBM CONFIÁVEL, usando no ensemble", C.G))
            
            # Retreina com TODOS os dados para o modelo final (só se confiável)
            if lgbm_reliable:
                lgbm_model.fit(X, y)
            else:
                # Modelo não confiável - mantém treinado mas marcado como unreliable
                lgbm_model.fit(X, y)
        else:
            # Poucas amostras: treina tudo mas marca como não confiável
            lgbm_model.fit(X, y)
            preds = lgbm_model.predict(X)
            acc = (preds == y).mean() * 100
            lgbm_reliable = False  # Sem validação = não confiável
            lgbm_val_accuracy = 0.0
            log.info(paint(f"[LGBM] Modelo treinado! Amostras={n_samples} | Acc={acc:.1f}% (sem validação - LGBM não confiável)", C.Y))
        
        lgbm_save_model()
        
    except Exception as e:
        log.warning(f"[LGBM] Erro no treino: {e}")
        lgbm_reliable = False
        lgbm_val_accuracy = 0.0

def lgbm_predict(setup: Dict[str, Any]) -> Tuple[float, bool]:
    """
    Prediz probabilidade de WIN usando LightGBM.
    
    Returns:
        (probabilidade, modelo_disponivel)
    """
    global lgbm_model
    
    if not LGBM_ON or lgbm_model is None:
        return 0.5, False
    
    try:
        features = lgbm_extract_features(setup).reshape(1, -1)
        proba = lgbm_model.predict_proba(features)[0]
        # proba[1] = probabilidade de WIN (classe 1)
        return float(proba[1]), True
    except Exception as e:
        log.warning(f"[LGBM] Erro na predição: {e}")
        return 0.5, False

# ===================== ENSEMBLE: BAYESIANO + LIGHTGBM + CNN =====================

def ensemble_predict(ativo: str, setup: Dict[str, Any], stats: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Combina predições do Bayesiano, LightGBM e CNN para decisão mais robusta.
    
    Modes:
    - "both": Ambos devem aprovar (mais conservador)
    - "any": Qualquer um aprova (mais agressivo)
    - "weighted": Média ponderada das probabilidades
    
    Returns:
        {
            "should_trade": bool,
            "bayes_prob": float,
            "lgbm_prob": float,
            "cnn_prob": float,
            "ensemble_prob": float,
            "reason": str
        }
    """
    # Predição Bayesiana
    bayes_pred = ai_predict(ativo, setup, stats)
    bayes_prob = float(bayes_pred["prob"])
    bayes_conf = float(bayes_pred["conf"])
    n_arm = int(bayes_pred["n_arm"])
    
    # Predição LightGBM
    lgbm_prob, lgbm_available = lgbm_predict(setup)
    
    # Predição CNN Pattern Detector
    cnn_prob = 0.5
    cnn_available = False
    cnn_class = "NO_TRADE"
    if CNN_ON and cnn_model is not None and df is not None:
        try:
            direction_hint = str(setup.get("dir", ""))
            cnn_pred = cnn_model.predict(df, direction_hint=direction_hint)
            cnn_class = str(cnn_pred.get("class", "NO_TRADE"))
            cnn_reliable = bool(cnn_pred.get("reliable", False))
            
            # Converter classe CNN para probabilidade alinhada com a direção
            raw_probs = cnn_pred.get("raw_probs", [0.33, 0.33, 0.34])
            if direction_hint == "CALL":
                cnn_prob = float(raw_probs[0])  # P(CALL)
            elif direction_hint == "PUT":
                cnn_prob = float(raw_probs[1])  # P(PUT)
            else:
                cnn_prob = 0.5
            
            cnn_available = True
            
            # CNN VETO: se CNN diz fortemente que é NO_TRADE ou direção oposta
            if cnn_reliable and cnn_prob < CNN_VETO_THRESHOLD:
                return {
                    "should_trade": False,
                    "bayes_prob": bayes_prob,
                    "lgbm_prob": lgbm_prob,
                    "cnn_prob": cnn_prob,
                    "ensemble_prob": 0.0,
                    "reason": f"cnn_veto(cnn={cnn_prob:.2f}<{CNN_VETO_THRESHOLD:.2f},class={cnn_class})",
                    "bayes_conf": bayes_conf,
                    "n_arm": n_arm
                }
        except Exception as e:
            log.warning(f"[CNN] Erro na predição: {e}")
            cnn_prob = 0.5
            cnn_available = False
    
    # Se LightGBM não disponível OU não confiável, usa só Bayesiano com regras inteligentes
    if not lgbm_available or not LGBM_ON or not lgbm_reliable:
        reason_detail = "lgbm_off" if not LGBM_ON else ("lgbm_unreliable" if not lgbm_reliable else "lgbm_unavailable")
        
        if n_arm >= AI_MIN_SAMPLES:
            # Com histórico suficiente, confia no Bayesiano (threshold levemente relaxado)
            should_trade = (bayes_prob >= 0.52) and (bayes_conf >= AI_CONF_MIN)
            reason_suffix = f"hist,prob={bayes_prob:.2f},n={n_arm}"
        else:
            # WARMUP SEM LGBM: bayes_prob com n=0 é apenas o prior
            # Trader profissional: decide pela ZONA (S/R + LT + score), não pelo prior
            sc = float(setup.get("score", 0.0))
            ctx_val = float(setup.get("market_quality", 0.40))
            entry_conf = float(setup.get("entry_confidence", 0.50))
            sr_prox = float(setup.get("sr_proximity", 0.0))
            sr_tq = int(setup.get("sr_touches", 0))
            sr_w = float(setup.get("sr_weight", 0.0))
            has_lt = bool(setup.get("has_lt", False))
            lt_pts = int(setup.get("lt_points", setup.get("pb_len", 0)))
            sr_forte = sr_tq >= 3 and sr_w >= 4.0
            zona_forte = sr_forte or (has_lt and lt_pts >= 3)
            
            # W1: Contexto péssimo sem zona forte → bloqueia
            if ctx_val < 0.30 and not zona_forte:
                should_trade = False
                reason_suffix = f"ctx_pessimo={ctx_val:.2f}"
            # W2: Score + contexto bons (critério principal do backtest calibrado)
            elif sc >= 0.45 and ctx_val >= 0.40:
                should_trade = True
                reason_suffix = f"score_ctx_ok(sc={sc:.2f},ctx={ctx_val:.2f})"
            # W3: Zona forte + score razoável → permite (trader profissional entra aqui)
            elif zona_forte and sc >= 0.40:
                should_trade = True
                reason_suffix = f"zona_forte(sc={sc:.2f},sr={sr_tq}t,lt={lt_pts}pts)"
            # W4: Score bom + S/R forte → permite
            elif sc >= 0.40 and sr_forte:
                should_trade = True
                reason_suffix = f"score_sr_ok(sc={sc:.2f},sr={sr_tq}t)"
            # W5: Score bom + candle confirmando → permite
            elif sc >= 0.45 and entry_conf >= 0.40:
                should_trade = True
                reason_suffix = f"score_candle_ok(sc={sc:.2f},ec={entry_conf:.2f})"
            # W6: Prior razoável + score decente → permite
            elif bayes_prob >= 0.50 and sc >= 0.42:
                should_trade = True
                reason_suffix = f"prior_score_ok(prob={bayes_prob:.2f},sc={sc:.2f})"
            # W7: Modo learning - permite setups com score mínimo para aprender
            elif IA_MODE == "learning" and sc >= 0.40 and ctx_val >= 0.40:
                should_trade = True
                reason_suffix = f"learning_ok(sc={sc:.2f},ctx={ctx_val:.2f})"
            else:
                should_trade = False
                reason_suffix = f"fraco(sc={sc:.2f},prob={bayes_prob:.2f},ctx={ctx_val:.2f})"
        
        # CNN pode ajudar mesmo sem LGBM: ajusta ensemble_prob
        ens_prob = bayes_prob
        if cnn_available and cnn_prob != 0.5:
            ens_prob = bayes_prob * (1.0 - CNN_WEIGHT) + cnn_prob * CNN_WEIGHT
        
        return {
            "should_trade": should_trade,
            "bayes_prob": bayes_prob,
            "lgbm_prob": 0.5,
            "cnn_prob": cnn_prob,
            "ensemble_prob": ens_prob,
            "reason": f"bayes_only({reason_detail},{reason_suffix})" + (f",cnn={cnn_prob:.2f}" if cnn_available else ""),
            "bayes_conf": bayes_conf,
            "n_arm": n_arm
        }
    
    # Calcula probabilidade ensemble (média ponderada com CNN)
    # Bayesiano tem peso maior se tiver mais amostras
    bayes_weight = min(1.0, n_arm / AI_MIN_SAMPLES) * 0.5 + 0.25  # 0.25 a 0.75
    lgbm_weight = 1.0 - bayes_weight
    
    # Integra CNN no ensemble (reduz pesos proporcionalmente)
    if cnn_available and cnn_prob != 0.5:
        cnn_w = CNN_WEIGHT  # 0.20 padrão
        bayes_weight *= (1.0 - cnn_w)
        lgbm_weight *= (1.0 - cnn_w)
        ensemble_prob = bayes_prob * bayes_weight + lgbm_prob * lgbm_weight + cnn_prob * cnn_w
    else:
        ensemble_prob = bayes_prob * bayes_weight + lgbm_prob * lgbm_weight
    
    # Sufixo CNN para logs
    cnn_suffix = f",C={cnn_prob:.2f}" if cnn_available else ""
    
    # Decisão baseada no modo
    if ENSEMBLE_MODE == "both":
        # Ambos devem aprovar
        bayes_ok = bayes_prob >= AI_MIN_PROB
        lgbm_ok = lgbm_prob >= LGBM_MIN_PROB
        should_trade = bayes_ok and lgbm_ok
        reason = f"both(B={bayes_prob:.2f},L={lgbm_prob:.2f}{cnn_suffix})"
    elif ENSEMBLE_MODE == "any":
        # Qualquer um aprova, MAS ensemble deve ser forte
        bayes_ok = bayes_prob >= AI_MIN_PROB
        lgbm_ok = lgbm_prob >= LGBM_MIN_PROB
        should_trade = (bayes_ok or lgbm_ok) and ensemble_prob >= 0.55
        reason = f"any(B={bayes_prob:.2f},L={lgbm_prob:.2f},ens={ensemble_prob:.2f}{cnn_suffix})"
    else:  # weighted
        # Usa prob ensemble com threshold mais alto
        min_ens_weighted = max(AI_MIN_PROB, 0.58)
        should_trade = ensemble_prob >= min_ens_weighted
        reason = f"weighted(ens={ensemble_prob:.2f},min={min_ens_weighted:.2f}{cnn_suffix})"
    
    # Warmup: se poucos dados no Bayesiano, IA decide com base no cenário
    if n_arm < AI_MIN_SAMPLES:
        warmup_threshold = LGBM_WARMUP_PROB if IA_MODE == "learning" else LGBM_MIN_PROB
        if lgbm_available:
            # Pegar contexto do setup para ajustar exigência
            ctx_val = float(setup.get("market_quality", 0.40))
            entry_conf_val = float(setup.get("entry_confidence", 0.50))
            sr_prox = float(setup.get("sr_proximity", 0.0))
            sr_tq = int(setup.get("sr_touches", 0))
            
            # Threshold dinâmico de ensemble baseado no contexto
            if ctx_val < 0.40:  # contexto ruim
                min_ens = ENS_MIN_CTX_RUIM  # 0.58
            elif ctx_val < 0.50:  # contexto mediano
                min_ens = ENS_MIN_CTX_MED   # 0.54
            else:  # contexto bom
                min_ens = ENS_MIN_CTX_BOM   # 0.50
            
            # entry_conf baixo (0.44) penaliza mais
            if entry_conf_val < 0.50:
                min_ens += 0.02
            
            # S/R PROXIMITY BENEFIT: pullback perto de S/R forte relaxa threshold
            # (estilo Candle Mind: S/R forte = mais confiança na entrada)
            if sr_prox > 0.60 and sr_tq >= 4:
                min_ens -= 0.06  # suporte/resistência muito forte
            elif sr_prox > 0.40 and sr_tq >= 3:
                min_ens -= 0.03  # suporte/resistência moderado
            
            # REGRA #1: LGBM muito confiante que vai PERDER
            if lgbm_prob < 0.30:
                should_trade = False
                reason = f"warmup_danger(L={lgbm_prob:.2f}<0.30)"
            # REGRA #2: Ambos negativos = consenso de LOSS, bloquear
            elif bayes_prob < 0.50 and lgbm_prob < 0.50:
                should_trade = False
                reason = f"warmup_consenso_neg(B={bayes_prob:.2f},L={lgbm_prob:.2f})"
            # REGRA #3: Ambos modelos concordam positivamente (ambos >= 0.58)
            elif bayes_prob >= 0.58 and lgbm_prob >= 0.58:
                should_trade = True
                reason = f"warmup_consenso_ok(B={bayes_prob:.2f},L={lgbm_prob:.2f})"
            # REGRA #4: Bayes alto (>=0.63) E LGBM não contra (>=0.50)
            elif bayes_prob >= 0.63 and lgbm_prob >= 0.50:
                should_trade = True
                reason = f"warmup_bayes_forte(B={bayes_prob:.2f},L={lgbm_prob:.2f})"
            # REGRA #5: LGBM alto (>=0.62) E Bayes não contra (>=0.50)
            elif lgbm_prob >= 0.62 and bayes_prob >= 0.50:
                should_trade = True
                reason = f"warmup_lgbm_forte(B={bayes_prob:.2f},L={lgbm_prob:.2f})"
            # REGRA #6: Ensemble atinge threshold dinâmico (ajustado por contexto)
            elif ensemble_prob >= min_ens:
                should_trade = True
                reason = f"warmup_ens_ctx(B={bayes_prob:.2f},L={lgbm_prob:.2f},ens={ensemble_prob:.2f},min={min_ens:.2f})"
            else:
                # Cenário fraco demais
                should_trade = False
                reason = f"warmup_fraco(B={bayes_prob:.2f},L={lgbm_prob:.2f},ens={ensemble_prob:.2f},min={min_ens:.2f})"
        else:
            should_trade = True
            reason = f"warmup_no_lgbm"
    
    return {
        "should_trade": should_trade,
        "bayes_prob": bayes_prob,
        "lgbm_prob": lgbm_prob,
        "cnn_prob": cnn_prob,
        "ensemble_prob": ensemble_prob,
        "reason": reason,
        "bayes_conf": bayes_conf,
        "n_arm": n_arm
    }

# ===================== IA BAYESIANA (ORIGINAL) =====================

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
    b_eff = _bucket(effA, 0.05, 0.05, 0.50)     # direcionalidade DOM Forex (OTC ~0.05-0.25)
    b_flip = _bucket(flips, 0.10, 0.0, 0.80)    # chopiness NOVO
    b_dist = _bucket(distBreak, 0.05, 0.0, 0.50) # distância da quebra NOVO

    return f"{d}|sc{b_sc}|pb{pb}|re{b_re}|A{b_A}|eff{b_eff}|fl{b_flip}|dst{b_dist}"

def ai_prior_from_setup(setup: Dict[str, Any]) -> float:
    """
    Prior PROFISSIONAL baseado em zonas de alta probabilidade.
    Trader profissional: prioriza ZONA (S/R + LT) sobre candle.
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

    # Base no score (range mais amplo para refletir melhor a qualidade)
    p = 0.48 + (sc - 0.40) * 0.40

    # 1. ZONA S/R - fator mais importante para trader profissional
    if sr_tq >= 5 and sr_w >= 8.0:
        p += 0.08  # S/R muito forte (5+ toques, peso alto)
    elif sr_tq >= 3 and sr_w >= 4.0:
        p += 0.05  # S/R forte
    elif sr_tq >= 2:
        p += 0.02  # S/R básico

    # 2. TRENDLINE alinhada - segundo fator mais importante
    if has_lt and lt_pts >= 4:
        p += 0.06  # LT muito forte (4+ pontos)
    elif has_lt and lt_pts >= 2:
        p += 0.03  # LT presente

    # 3. Contexto de mercado
    if ctx >= 0.70:
        p += 0.04  # mercado excelente
    elif ctx >= 0.55:
        p += 0.02  # mercado bom
    elif ctx < 0.35:
        p -= 0.04  # mercado ruim

    # 4. Confluência alta = muitas confirmações
    if conf_count >= 5:
        p += 0.04
    elif conf_count >= 4:
        p += 0.02

    # 5. Candle padrão (bônus, não requisito)
    if candle_str >= 0.60:
        p += 0.03  # candle forte confirmando
    elif candle_str >= 0.30:
        p += 0.01  # candle moderado
    # Candle 0.0 (neutral) = sem bônus, mas NÃO penaliza

    # 6. Direcionalidade (DOM Forex)
    if effA > 0.20:
        p += 0.02
    elif effA < 0.05:
        p -= 0.02

    return _clip(p, 0.42, 0.78)

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
    w = _clip(n / (n + 25.0), 0.0, 1.0)
    prob = (1.0 - w) * prior + w * bayes_mean
    prob = _clip(prob, 0.0, 1.0)

    return {"prob": float(prob), "bayes": float(bayes_mean), "ucb01": float(ucb01),
            "conf": float(conf), "n_arm": n, "total": total, "key": key, "prior": prior}

def ai_update(ativo: str, setup: Dict[str, Any], pnl: float, stats: Dict[str, Any]):
    """
    pnl > 0 => sucesso
    pnl < 0 => falha
    pnl = 0 => ignora

    ATUALIZA: Bayesian (a, b, n) + Rastreamento de padrão (wins, losses, trades)
    """
    if pnl == 0:
        return

    key = ai_make_key(ativo, setup)
    arms = stats.setdefault("arms", {})
    meta = stats.setdefault("meta", {"total": 0})
    patterns = stats.setdefault("patterns", {})  # NOVO: rastreamento de padrões

    arm = arms.get(key)
    if arm is None:
        prior = ai_prior_from_setup(setup)
        arms[key] = {"a": 2.0 * prior, "b": 2.0 * (1.0 - prior), "n": 0}
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

    # NOVO: Rastreamento de padrão para bloqueio inteligente
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
    # Isso reduz a probabilidade bayesiana do padrão
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

def ai_learn_from_backtest_signal(ativo: str, setup: Dict[str, Any], win: bool, stats: Dict[str, Any]):
    """
    APRENDIZADO COM BACKTEST: Atualiza a IA com sinais simulados do backtest.
    Usa peso reduzido (AI_BACKTEST_WEIGHT) para não dominar trades reais.
    """
    if not AI_LEARN_FROM_BACKTEST:
        return
    
    key = ai_make_key(ativo, setup)
    arms = stats.setdefault("arms", {})
    patterns = stats.setdefault("patterns", {})
    
    arm = arms.get(key)
    if arm is None:
        prior = ai_prior_from_setup(setup)
        arms[key] = {"a": 2.0 * prior, "b": 2.0 * (1.0 - prior), "n": 0}
        arm = arms[key]
    
    # Aplicar peso do backtest (menor que trades reais)
    weight = AI_BACKTEST_WEIGHT
    
    if win:
        arm["a"] = float(arm.get("a", 1.0)) + weight
    else:
        arm["b"] = float(arm.get("b", 1.0)) + weight
    
    arm["n"] = int(arm.get("n", 0)) + 1
    
    # Atualizar padrão
    pattern = patterns.get(key)
    if pattern is None:
        patterns[key] = {"trades": 0, "wins": 0, "losses": 0, "backtest_samples": 0}
        pattern = patterns[key]
    
    pattern["backtest_samples"] = pattern.get("backtest_samples", 0) + 1
    if win:
        pattern["wins"] = pattern.get("wins", 0) + 1
    else:
        pattern["losses"] = pattern.get("losses", 0) + 1

def ai_learn_from_backtest_batch(sinais: List[Dict], stats: Dict[str, Any]):
    """
    Processa um lote de sinais do backtest e atualiza a IA.
    Só processa sinais de qualidade (score >= 0.55, ctx >= 0.40).
    Retorna quantidade de sinais processados.
    """
    if not AI_LEARN_FROM_BACKTEST or not sinais:
        return 0
    
    count = 0
    skipped = 0
    for sinal in sinais:
        ativo = sinal.get("ativo", "")
        win = sinal.get("win", False)
        score = sinal.get("score", 0.0)
        ctx = sinal.get("ctx", 0.0)
        effA = sinal.get("effA", 0.0)
        
        # ===== FILTRO DE QUALIDADE =====
        # Só aprende com sinais que teriam passado nos filtros reais
        if score < 0.55 or ctx < 0.40 or effA < 0.25:
            skipped += 1
            continue
        
        # Reconstruir setup completo para gerar a key e features corretas
        setup_backtest = {
            "dir": sinal.get("direcao", "CALL"),
            "score": score,
            "market_quality": ctx,
            "effA": effA,
            "has_lt": sinal.get("has_lt", False),
            "retracement": sinal.get("retr", 0.5),
            "pullback_candles": sinal.get("pb", 2),
            "entry_confirmation": sinal.get("entry_conf", 0.5),
            "lt_confluence": sinal.get("lt_conf", 0),
        }
        
        ai_learn_from_backtest_signal(ativo, setup_backtest, win, stats)
        
        # Também adiciona ao LightGBM se disponível
        if LGBM_ON and lgb is not None:
            lgbm_add_sample_from_backtest(setup_backtest, win)
        
        count += 1
    
    if skipped > 0:
        log.info(paint(f"[BACKTEST-LEARN] Filtrou {skipped} sinais fracos, aprendeu {count} de qualidade", C.Y))
    
    return count

def lgbm_add_sample_from_backtest(setup: Dict[str, Any], win: bool):
    """
    Adiciona amostra do backtest ao LightGBM SOMENTE se for de qualidade.
    Filtra sinais fracos (score baixo, contexto ruim) que poluem o treinamento.
    """
    global lgbm_data
    if not LGBM_ON:
        return
    
    # ===== FILTRO DE QUALIDADE =====
    # Só aprende com sinais que teriam passado nos filtros reais
    score = setup.get("score", 0.0)
    ctx = setup.get("market_quality", 0.0)
    effA = setup.get("effA", 0.0)
    
    # Requer: score >= 0.55 E contexto >= 0.40 E effA >= 0.25
    if score < 0.55 or ctx < 0.40 or effA < 0.25:
        return  # Sinal fraco -- não polui o treinamento
    
    features = lgbm_extract_features(setup).tolist()
    label = 1 if win else 0
    
    # Adiciona com flag de backtest e timestamp
    lgbm_data.append({"features": features, "label": label, "source": "backtest", "timestamp": time.time()})
    
    # Limita dados
    if len(lgbm_data) > 1000:
        lgbm_data = lgbm_data[-1000:]
    
    lgbm_save_data()

# ===================== HISTÓRICO DE BACKTEST COM DECAY TEMPORAL =====================
# Armazena sinais de backtests anteriores para aumentar assertividade

backtest_history: List[Dict] = []

def backtest_history_load():
    """Carrega histórico de backtests do disco."""
    global backtest_history
    try:
        if os.path.exists(BACKTEST_HISTORY_FILE):
            with open(BACKTEST_HISTORY_FILE, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if not raw or len(raw) < 10:
                log.warning(f"[BACKTEST-HIST] Arquivo vazio ou corrompido ({len(raw)} bytes) - resetando")
                backtest_history = []
                return
            backtest_history = json.loads(raw)
            if not isinstance(backtest_history, list):
                log.warning(f"[BACKTEST-HIST] Formato inválido (não é lista) - resetando")
                backtest_history = []
                return
            log.info(f"[BACKTEST-HIST] Carregado histórico com {len(backtest_history)} amostras")
    except (json.JSONDecodeError, ValueError) as e:
        log.warning(f"[BACKTEST-HIST] JSON corrompido: {e} - resetando arquivo")
        backtest_history = []
        # Salvar arquivo limpo para evitar erro repetido
        try:
            with open(BACKTEST_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
        except Exception:
            pass
    except Exception as e:
        backtest_history = []
        log.warning(f"[BACKTEST-HIST] Erro ao carregar: {e}")

def backtest_history_save():
    """Salva histórico de backtests no disco (escrita atômica para evitar corrupção)."""
    global backtest_history
    try:
        # Escrita atômica: salva em arquivo temporário e depois renomeia
        tmp_file = BACKTEST_HISTORY_FILE + ".tmp"
        data = json.dumps(backtest_history, ensure_ascii=False)
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        # Renomear é atômico no Windows (sobrescreve)
        if os.path.exists(BACKTEST_HISTORY_FILE):
            os.remove(BACKTEST_HISTORY_FILE)
        os.rename(tmp_file, BACKTEST_HISTORY_FILE)
    except Exception as e:
        log.warning(f"[BACKTEST-HIST] Erro ao salvar: {e}")
        # Limpar tmp se ficou
        try:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        except Exception:
            pass

def backtest_history_calculate_weight(timestamp: float) -> float:
    """
    Calcula o peso de uma amostra baseado na idade (decay temporal).
    - Amostras recentes (< DECAY_HOURS): peso = 1.0
    - Amostras antigas: peso decai linearmente até MIN_WEIGHT
    """
    age_seconds = time.time() - timestamp
    age_hours = age_seconds / 3600
    
    if age_hours <= BACKTEST_HISTORY_DECAY_HOURS:
        return 1.0
    
    # Decay linear após DECAY_HOURS horas
    # Em 24h, chega ao peso mínimo
    decay_range = 24.0 - BACKTEST_HISTORY_DECAY_HOURS
    excess_hours = age_hours - BACKTEST_HISTORY_DECAY_HOURS
    decay_factor = min(1.0, excess_hours / decay_range)
    
    weight = 1.0 - (decay_factor * (1.0 - BACKTEST_HISTORY_MIN_WEIGHT))
    return max(BACKTEST_HISTORY_MIN_WEIGHT, weight)

def backtest_history_add_signals(sinais: List[Dict]):
    """
    Adiciona sinais do backtest atual ao histórico acumulado.
    Remove amostras antigas se exceder o limite.
    """
    global backtest_history
    
    if not BACKTEST_USE_ACCUMULATED:
        return
    
    timestamp = time.time()
    
    # Adicionar novos sinais com timestamp
    for sinal in sinais:
        sinal_with_ts = sinal.copy()
        sinal_with_ts["timestamp"] = timestamp
        backtest_history.append(sinal_with_ts)
    
    # Remover amostras muito antigas (>48h) e limitar tamanho
    cutoff_time = time.time() - (48 * 3600)  # 48 horas
    backtest_history = [s for s in backtest_history if s.get("timestamp", 0) > cutoff_time]
    
    # Limitar ao máximo de amostras (mantém as mais recentes)
    if len(backtest_history) > BACKTEST_HISTORY_MAX_SAMPLES:
        backtest_history = sorted(backtest_history, key=lambda x: x.get("timestamp", 0), reverse=True)
        backtest_history = backtest_history[:BACKTEST_HISTORY_MAX_SAMPLES]
    
    backtest_history_save()
    log.info(f"[BACKTEST-HIST] Histórico atualizado: {len(backtest_history)} amostras")

def backtest_history_get_weighted_signals() -> List[Tuple[Dict, float]]:
    """
    Retorna todos os sinais do histórico com seus pesos calculados.
    Returns: Lista de (sinal, peso)
    """
    global backtest_history
    
    if not BACKTEST_USE_ACCUMULATED or not backtest_history:
        return []
    
    weighted_signals = []
    for sinal in backtest_history:
        ts = sinal.get("timestamp", time.time())
        weight = backtest_history_calculate_weight(ts)
        weighted_signals.append((sinal, weight))
    
    return weighted_signals

def backtest_history_analyze() -> Dict[str, Any]:
    """
    Analisa o histórico de backtest e retorna estatísticas.
    """
    global backtest_history
    
    if not backtest_history:
        return {"total": 0, "wins": 0, "winrate": 0, "weighted_winrate": 0}
    
    total = len(backtest_history)
    wins = sum(1 for s in backtest_history if s.get("win", False))
    
    # Calcular winrate ponderado pelo tempo
    total_weight = 0
    wins_weight = 0
    for sinal in backtest_history:
        ts = sinal.get("timestamp", time.time())
        weight = backtest_history_calculate_weight(ts)
        total_weight += weight
        if sinal.get("win", False):
            wins_weight += weight
    
    weighted_winrate = (wins_weight / total_weight) if total_weight > 0 else 0
    
    return {
        "total": total,
        "wins": wins,
        "winrate": wins / total if total > 0 else 0,
        "weighted_winrate": weighted_winrate,
        "total_weight": total_weight
    }

def ai_learn_from_accumulated_history(stats: Dict[str, Any]) -> int:
    """
    Treina a IA com o histórico acumulado de backtests, aplicando peso temporal.
    Usa peso decrescente para amostras antigas.
    
    Returns: Número de amostras processadas
    """
    global backtest_history
    
    if not AI_LEARN_FROM_BACKTEST or not BACKTEST_USE_ACCUMULATED:
        return 0
    
    if not backtest_history:
        backtest_history_load()
    
    if not backtest_history:
        return 0
    
    count = 0
    for sinal, time_weight in backtest_history_get_weighted_signals():
        ativo = sinal.get("ativo", "")
        win = sinal.get("win", False)
        
        # Peso final = peso do backtest * peso temporal
        final_weight = AI_BACKTEST_WEIGHT * time_weight
        
        # Reconstruir setup
        setup_backtest = {
            "dir": sinal.get("direcao", "CALL"),
            "score": sinal.get("score", 0.5),
            "market_quality": sinal.get("ctx", 0.5),
            "effA": sinal.get("effA", 0.3),
            "has_lt": sinal.get("has_lt", False),
            "retracement": sinal.get("retr", 0.5),
            "pullback_candles": sinal.get("pb", 2),
            "entry_confirmation": sinal.get("entry_conf", 0.5),
            "lt_confluence": sinal.get("lt_conf", 0),
        }
        
        # Atualizar Bayesian com peso ajustado
        key = ai_make_key(ativo, setup_backtest)
        arms = stats.setdefault("arms", {})
        
        arm = arms.get(key)
        if arm is None:
            prior = ai_prior_from_setup(setup_backtest)
            arms[key] = {"a": 2.0 * prior, "b": 2.0 * (1.0 - prior), "n": 0}
            arm = arms[key]
        
        if win:
            arm["a"] = float(arm.get("a", 1.0)) + final_weight
        else:
            arm["b"] = float(arm.get("b", 1.0)) + final_weight
        
        arm["n"] = int(arm.get("n", 0)) + 1
        count += 1
    
    return count

def ai_should_block_pattern(ativo: str, setup: Dict[str, Any], stats: Dict[str, Any]) -> Tuple[bool, str]:
    """
    DECISÃO INTELIGENTE: Bloqueia padrão SOMENTE se histórico mostra losses consistentes.
    Permite padrões durante fase de aprendizado e padrões com winrate aceitável.

    Returns:
        (should_block, reason)
    """
    key = ai_make_key(ativo, setup)
    patterns = stats.get("patterns", {})
    pattern = patterns.get(key)

    # Fase 0: PADRÃO QUEIMADO - bloqueia temporariamente após LOSS com retreino
    if pattern and pattern.get("burned_until", 0) > time.time():
        remaining = int(pattern["burned_until"] - time.time())
        return True, f"burned({remaining}s_restantes)"

    # Fase 1: APRENDIZADO - permite tudo (sem histórico suficiente)
    if pattern is None or pattern["trades"] < AI_MIN_SAMPLES:
        trades_count = pattern["trades"] if pattern else 0
        return False, f"learning({trades_count}/{AI_MIN_SAMPLES})"

    # Fase 2: AVALIAÇÃO - analisa performance real
    winrate = pattern["wins"] / max(1, pattern["trades"])

    # Bloqueia SOMENTE se winrate consistentemente BAIXO
    if winrate < AI_MIN_WINRATE:
        return True, f"blocked_wr={winrate:.0%}({pattern['wins']}W/{pattern['losses']}L)"

    # Permite: padrão tem performance aceitável ou está em zona cinza
    return False, f"approved_wr={winrate:.0%}({pattern['wins']}W/{pattern['losses']}L)"


# ===================== DETECT SETUP (DOM FOREX PERFECT ZONES) =====================
def detect_setup(df: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    """Estratégia DOM Forex Perfect Zones (price action puro)."""
    return dom_forex_signal(df, atr_val)


# ===================== ESCOLHER MELHOR SETUP DO MINUTO (TODOS OS ATIVOS) =====================
def escolher_melhor_setup(iq: BrokerAPI, ativos: List[str]):
    """
    Analisa TODOS os ativos de uma vez e retorna o melhor setup.
    Se encontrar sinal confirmado em qualquer ativo, entra.
    """
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

        df = get_candles_df(iq, a, TF_M1, N_M1, end_ts=end_ts_closed(TF_M1))
        if df is None:
            continue

        atr_val = atr(df, 14)
        last_closed = df.iloc[-1]

        if is_spike_wicky(last_closed, atr_val):
            cooldown_spike[a] = time.time()
            continue

        setup = detect_setup(df, atr_val)

        sc_any = float(setup.get("score", 0.0))
        cand_any = (sc_any, a, setup, float(atr_val), df)
        if best_any is None or cand_any[0] > best_any[0]:
            best_any = cand_any

        if setup.get("trade"):
            cand_trade = (float(setup["score"]), a, setup, float(atr_val), df)
            if best_trade is None or cand_trade[0] > best_trade[0]:
                best_trade = cand_trade
                log.info(paint(f"  🎯 {a}: {setup['dir']} score={setup['score']:.2f}", dir_color(setup['dir'])))

    return best_trade, best_any

# ===================== ORDEM =====================
def enviar_ordem(iq: BrokerAPI, ativo: str, direcao: str, stake: float) -> Optional[Tuple[str, int]]:
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

def wait_result(iq: BrokerAPI, op_type: str, op_id: int) -> float:
    # Timeout total = EXP_FIXA * 60 + 30s margem (ex: M1=90s, M5=330s)
    _max_wait = int(EXP_FIXA) * 60 + 30
    _check_timeout = int(EXP_FIXA) * 60 + 15  # Timeout por tentativa de check
    _wr_start = time.time()
    _attempts = 0
    while True:
        elapsed = time.time() - _wr_start
        if elapsed > _max_wait:
            log.warning(paint(f"⏱️ wait_result TIMEOUT {_max_wait}s - assumindo LOSS", C.R))
            return 0.0
        try:
            if op_type == "turbo":
                result = safe_call(iq, iq.check_win_v4, op_id, timeout=_check_timeout)
                if result is not None:
                    if isinstance(result, tuple):
                        ok, res = result
                        if ok:
                            return float(res)
                    elif isinstance(result, (int, float)):
                        return float(result)
            else:
                result = safe_call(iq, iq.check_win_digital_v2, op_id, timeout=_check_timeout)
                if result is not None:
                    if isinstance(result, tuple):
                        ok, res = result
                        if ok and isinstance(res, (int, float)):
                            return float(res)
                    elif isinstance(result, (int, float)):
                        return float(result)
        except Exception as e:
            _attempts += 1
            log.warning(paint(f"⏱️ wait_result erro (tentativa {_attempts}): {e}", C.Y))
            if _attempts >= 3:
                log.warning(paint(f"⏱️ wait_result: 3 erros seguidos - assumindo não resolvido", C.R))
                return 0.0
            ensure_connected(iq)
        time.sleep(0.5)

# ===================== BACKTEST AUTOMÁTICO =====================
# Arquivo para salvar estado do auto-tuner
AUTO_TUNER_FILE = os.path.join(os.path.dirname(__file__), "auto_tuner_state.json")

def backtest_antes_de_operar(iq: BrokerAPI, ativos: List[str], n_candles: int = 90) -> Dict[str, Any]:
    """
    Executa backtest nos últimos N minutos para calibrar filtros automaticamente.
    CALCULA FILTROS INDIVIDUAIS POR ATIVO!
    
    Retorna:
        {
            "sinais": int,
            "wins": int,
            "losses": int,
            "taxa_acerto": float,
            "calibrado": bool,
            "ajustes": list,
            "filtros_por_ativo": dict
        }
    """
    global GATE_CONTEXT_VERY_BAD, GATE_MIN_SCORE, filtros_por_ativo
    
    log.info("=" * 60)
    log.info(paint("🧠 BACKTEST INTELIGENTE - Analisando padrões dos últimos 90min...", C.G))
    log.info(paint("   📊 Calculando filtros INDIVIDUAIS por ativo!", C.B))
    log.info("=" * 60)
    
    # DOM Forex exige MÍNIMO 150 velas (otimizado)
    candles_necessarios = max(200, n_candles)
    
    # Guardar filtros originais
    ctx_original = GATE_CONTEXT_VERY_BAD
    score_original = GATE_MIN_SCORE
    
    # ===== FASE 1: COLETAR TODOS OS SINAIS POR ATIVO =====
    log.info(paint("📊 Fase 1: Coletando sinais por ativo...", C.B))
    sinais_por_ativo: Dict[str, List[Dict]] = {}
    _bt_start = time.time()
    _BT_TIMEOUT = 120  # máximo 2 min para backtest inteiro
    _bt_timed_out = False
    
    for ativo in ativos[:5]:  # máx 5 ativos (otimizado)
        if time.time() - _bt_start > _BT_TIMEOUT:
            log.warning(paint(f"⏱️ Backtest timeout ({_BT_TIMEOUT}s) - usando dados já coletados", C.Y))
            _bt_timed_out = True
            break
        sinais_por_ativo[ativo] = []
        try:
            df = get_candles_df(iq, ativo, TF_M1, candles_necessarios + 50)
            if df is None or len(df) < 160:
                continue
            
            for i in range(160, len(df) - 2, 2):  # step=2 para performance
                if time.time() - _bt_start > _BT_TIMEOUT:
                    _bt_timed_out = True
                    break
                df_window = df.iloc[:i].copy().reset_index(drop=True)
                
                atr_val = atr(df_window, 14)
                if atr_val < 1e-9:
                    continue
                
                setup = detect_setup(df_window, atr_val)
                
                if setup.get("trade"):
                    score = setup.get("score", 0)
                    ctx = setup.get("market_quality", 0)
                    direcao = setup.get("dir")
                    effA = setup.get("effA", 0)
                    has_lt = setup.get("has_lt", False)
                    retr = setup.get("retracement", 0.5)
                    pb = setup.get("pullback_candles", 2)
                    entry_conf = setup.get("entry_confirmation", 0.5)
                    lt_conf = setup.get("lt_confluence", 0)
                    
                    # Simular resultado
                    vela_entrada = df.iloc[i]
                    vela_resultado = df.iloc[i + 1]
                    
                    if direcao == "CALL":
                        win = vela_resultado["close"] > vela_entrada["open"]
                    else:
                        win = vela_resultado["close"] < vela_entrada["open"]
                    
                    sinais_por_ativo[ativo].append({
                        "ativo": ativo,
                        "direcao": direcao,
                        "score": float(score),
                        "ctx": float(ctx),
                        "effA": float(effA),
                        "has_lt": bool(has_lt),
                        "retr": float(retr),
                        "pb": int(pb),
                        "entry_conf": float(entry_conf),
                        "lt_conf": float(lt_conf),
                        "win": bool(win)
                    })
        except Exception:
            continue
    
    # Combinar todos os sinais para análise global
    todos_sinais = []
    for ativo, sinais in sinais_por_ativo.items():
        todos_sinais.extend(sinais)
    
    total_raw = len(todos_sinais)
    if total_raw == 0:
        min_ctx_floor = max(GATE_CONTEXT_VERY_BAD, 0.40)
        min_score_floor = max(GATE_MIN_SCORE, 0.60)
        log.warning(paint("⚠️ Nenhum sinal encontrado no período - mantendo filtros rígidos", C.Y))
        GATE_CONTEXT_VERY_BAD = min_ctx_floor
        GATE_MIN_SCORE = min_score_floor
        # Inicializar todos os ativos com piso rígido (sem auto-relax)
        for ativo in ativos:
            filtros_por_ativo[ativo] = {"min_ctx": min_ctx_floor, "min_score": min_score_floor, "taxa": 0.50, "sinais": 0, "habilitado": True}
        return {"sinais": 0, "wins": 0, "losses": 0, "taxa_acerto": 0, "calibrado": True, "ajustes": [], "filtros_por_ativo": filtros_por_ativo, "use_trendline": REQUIRE_TRENDLINE}
    
    wins_raw = sum(1 for s in todos_sinais if s["win"])
    taxa_raw = wins_raw / total_raw
    log.info(f"   Total de sinais: {total_raw} | WINs: {wins_raw} ({taxa_raw*100:.1f}%)")
    
    # ===== ANÁLISE DE TRENDLINE (has_lt) =====
    # Verificar se exigir trendline realmente melhora os resultados
    sinais_com_lt = [s for s in todos_sinais if s.get("has_lt", False)]
    sinais_sem_lt = [s for s in todos_sinais if not s.get("has_lt", False)]
    
    winrate_com_lt = sum(1 for s in sinais_com_lt if s["win"]) / len(sinais_com_lt) if sinais_com_lt else 0
    winrate_sem_lt = sum(1 for s in sinais_sem_lt if s["win"]) / len(sinais_sem_lt) if sinais_sem_lt else 0
    
    # Decidir se usa filtro de trendline automaticamente
    # Se REQUIRE_TRENDLINE estiver ativo, nunca desativar por backtest.
    use_trendline_filter = REQUIRE_TRENDLINE
    if len(sinais_com_lt) >= 3:
        # Se trendline melhora significativamente (>5%), usar
        if winrate_com_lt > winrate_sem_lt + 0.05:
            use_trendline_filter = True
            log.info(paint(f"   📈 Trendline ATIVADO: com_LT={winrate_com_lt*100:.0f}% ({len(sinais_com_lt)}) vs sem_LT={winrate_sem_lt*100:.0f}% ({len(sinais_sem_lt)})", C.G))
        else:
            if REQUIRE_TRENDLINE:
                use_trendline_filter = True
                log.info(paint(f"   📊 Trendline MANTIDO (obrigatório): com_LT={winrate_com_lt*100:.0f}% vs sem_LT={winrate_sem_lt*100:.0f}%", C.B))
            else:
                use_trendline_filter = False
                log.info(paint(f"   📊 Trendline DESATIVADO: com_LT={winrate_com_lt*100:.0f}% vs sem_LT={winrate_sem_lt*100:.0f}% (diferença <5%)", C.Y))
    else:
        # Poucos sinais com trendline
        if REQUIRE_TRENDLINE:
            use_trendline_filter = True
            log.info(paint(f"   ⚠️ Trendline MANTIDO (obrigatório): poucos sinais com LT ({len(sinais_com_lt)})", C.B))
        else:
            use_trendline_filter = False
            log.info(paint(f"   ⚠️ Trendline DESATIVADO: poucos sinais com LT ({len(sinais_com_lt)})", C.Y))
    
    # ===== FASE 2: ANALISAR PADRÕES WINs vs LOSSes =====
    log.info(paint("🔍 Fase 2: Analisando padrões de WIN vs LOSS...", C.B))
    
    wins_list = [s for s in todos_sinais if s["win"]]
    losses_list = [s for s in todos_sinais if not s["win"]]
    
    if wins_list:
        avg_ctx_win = sum(s["ctx"] for s in wins_list) / len(wins_list)
        avg_score_win = sum(s["score"] for s in wins_list) / len(wins_list)
        min_ctx_win = min(s["ctx"] for s in wins_list)
        min_score_win = min(s["score"] for s in wins_list)
        log.info(f"   📗 WINs: ctx_médio={avg_ctx_win:.2f} score_médio={avg_score_win:.2f}")
    else:
        avg_ctx_win, avg_score_win = 0.40, 0.50
        min_ctx_win, min_score_win = 0.30, 0.42
    
    if losses_list:
        avg_ctx_loss = sum(s["ctx"] for s in losses_list) / len(losses_list)
        avg_score_loss = sum(s["score"] for s in losses_list) / len(losses_list)
        log.info(f"   📕 LOSSes: ctx_médio={avg_ctx_loss:.2f} score_médio={avg_score_loss:.2f}")
    else:
        avg_ctx_loss, avg_score_loss = 0.40, 0.55
    
    # ===== FASE 3: CALCULAR PONTO DE CORTE IDEAL =====
    log.info(paint("🎯 Fase 3: Calculando filtros ideais...", C.B))
    
    # ESTRATÉGIA: Testar TODAS as combinações de filtros e encontrar a que
    # maximiza a taxa de acerto mantendo pelo menos 3 sinais.
    # O backtest é o CALIBRADOR — deve encontrar os melhores filtros!
    
    best_ctx = 0.30
    best_score = 0.40
    best_taxa = taxa_raw
    best_count = total_raw
    
    # Testar combinações de filtros do mais rigoroso ao mais leve
    ctx_range = [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25]
    score_range = [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
    
    for ctx_test in ctx_range:
        for score_test in score_range:
            filtered = [s for s in todos_sinais if s["ctx"] >= ctx_test and s["score"] >= score_test]
            n = len(filtered)
            if n < 3:  # Precisa de pelo menos 3 sinais para ser estatisticamente relevante
                continue
            w = sum(1 for s in filtered if s["win"])
            taxa = w / n
            
            # Aceitar se: taxa melhor, OU mesma taxa com mais sinais
            if taxa > best_taxa or (taxa == best_taxa and n > best_count):
                best_taxa = taxa
                best_ctx = ctx_test
                best_score = score_test
                best_count = n
    
    ctx_ideal = best_ctx
    score_ideal = best_score
    
    log.info(f"   📐 Melhor combinação encontrada: ctx≥{ctx_ideal:.2f} score≥{score_ideal:.2f}")
    log.info(f"      → {best_count} sinais | {best_taxa*100:.1f}% WR (original: {taxa_raw*100:.1f}%)")
    
    # ===== FASE 4: VALIDAR E APLICAR FILTROS CALIBRADOS =====
    log.info(paint("📈 Fase 4: Validando filtros calibrados...", C.B))
    
    sinais_filtrados = [s for s in todos_sinais if s["ctx"] >= ctx_ideal and s["score"] >= score_ideal]
    total_filtrado = len(sinais_filtrados)
    wins_filtrado = sum(1 for s in sinais_filtrados if s["win"])
    taxa_filtrado = wins_filtrado / total_filtrado if total_filtrado > 0 else 0
    
    log.info(f"   Após filtro: {total_filtrado} sinais | {wins_filtrado} WINs ({taxa_filtrado*100:.1f}%)")
    
    # Se o backtest não conseguiu melhorar a taxa acima de 55%, aumentar filtros
    if taxa_filtrado < 0.55 and total_filtrado >= 3:
        log.info(paint(f"   ⚠️ Taxa ainda < 55% ({taxa_filtrado*100:.1f}%) - tentando filtros mais rigorosos...", C.Y))
        # Tentar subir os filtros para encontrar subset com melhor taxa
        for ctx_up in [ctx_ideal + 0.05, ctx_ideal + 0.10, ctx_ideal + 0.15]:
            for score_up in [score_ideal + 0.05, score_ideal + 0.10, score_ideal + 0.15]:
                sinais_test = [s for s in todos_sinais if s["ctx"] >= ctx_up and s["score"] >= score_up]
                if len(sinais_test) >= 2:
                    wins_test = sum(1 for s in sinais_test if s["win"])
                    taxa_test = wins_test / len(sinais_test)
                    if taxa_test >= 0.55:
                        ctx_ideal = ctx_up
                        score_ideal = score_up
                        total_filtrado = len(sinais_test)
                        wins_filtrado = wins_test
                        taxa_filtrado = taxa_test
                        log.info(paint(f"   ✅ Encontrado: ctx≥{ctx_ideal:.2f} score≥{score_ideal:.2f} → {taxa_filtrado*100:.1f}% ({total_filtrado} sinais)", C.G))
                        break
            if taxa_filtrado >= 0.55:
                break
    
    # ===== APLICAR FILTROS GLOBAIS =====
    GATE_CONTEXT_VERY_BAD = ctx_ideal
    GATE_MIN_SCORE = score_ideal
    
    ajustes = []
    if ctx_ideal != ctx_original:
        ajustes.append(f"Contexto: {ctx_original:.2f} → {ctx_ideal:.2f}")
    if score_ideal != score_original:
        ajustes.append(f"Score: {score_original:.2f} → {score_ideal:.2f}")
    
    # ===== FASE 6: CALCULAR FILTROS INDIVIDUAIS POR ATIVO =====
    log.info(paint("🎯 Fase 6: Calculando filtros por ativo...", C.B))
    
    for ativo, sinais in sinais_por_ativo.items():
        if len(sinais) < 2:
            # Poucos sinais - usar filtros globais mas habilitar
            filtros_por_ativo[ativo] = {
                "min_ctx": ctx_ideal,
                "min_score": score_ideal,
                "taxa": 0.50,
                "sinais": len(sinais),
                "habilitado": True,
                "motivo": "poucos_sinais"
            }
            continue
        
        wins_ativo = sum(1 for s in sinais if s["win"])
        taxa_ativo = wins_ativo / len(sinais)
        
        # ===== GRID SEARCH POR ATIVO =====
        # Encontrar melhor combinação de filtros para este ativo específico
        best_ctx_a = ctx_ideal
        best_score_a = score_ideal
        best_taxa_a = taxa_ativo
        best_n_a = len(sinais)
        
        for ctx_test in ctx_range:
            for score_test in score_range:
                filtered = [s for s in sinais if s["ctx"] >= ctx_test and s["score"] >= score_test]
                n = len(filtered)
                if n < 2:
                    continue
                w = sum(1 for s in filtered if s["win"])
                taxa = w / n
                if taxa > best_taxa_a or (taxa == best_taxa_a and n > best_n_a):
                    best_taxa_a = taxa
                    best_ctx_a = ctx_test
                    best_score_a = score_test
                    best_n_a = n
        
        ctx_ativo = best_ctx_a
        score_ativo = best_score_a
        
        # Decidir se ativo está habilitado
        habilitado = True
        motivo = "ok"
        
        if best_taxa_a < 0.42:
            # Taxa muito baixa mesmo com filtros — desabilitar ativo
            habilitado = False
            motivo = f"taxa_baixa_{best_taxa_a*100:.0f}%"
        elif best_taxa_a < 0.50:
            motivo = f"taxa_moderada_{best_taxa_a*100:.0f}%"
        
        filtros_por_ativo[ativo] = {
            "min_ctx": ctx_ativo,
            "min_score": score_ativo,
            "taxa": best_taxa_a,
            "sinais": len(sinais),
            "wins": wins_ativo,
            "habilitado": habilitado,
            "motivo": motivo
        }
        
        # Log
        status = "✅" if habilitado else "⛔"
        log.info(f"   {status} {ativo}: {len(sinais)} sinais | {taxa_ativo*100:.0f}% | ctx≥{ctx_ativo:.2f} score≥{score_ativo:.2f}")
    
    # ===== RESULTADO FINAL =====
    log.info("=" * 60)
    log.info(paint(f"🎯 RESULTADO DO BACKTEST INTELIGENTE:", C.G))
    log.info(f"   Sinais analisados: {total_raw}")
    log.info(f"   Taxa original: {taxa_raw*100:.1f}%")
    log.info(f"   → Taxa com filtros: {taxa_filtrado*100:.1f}% ({total_filtrado} sinais)")
    log.info(f"   Filtros finais: ctx≥{ctx_ideal:.2f} score≥{score_ideal:.2f}")
    
    if ajustes:
        log.info(paint("   📐 Ajustes:", C.Y))
        for aj in ajustes:
            log.info(f"      • {aj}")
    
    # CONDIÇÃO: Calibrado se backtest conseguiu filtros com taxa >= 50%
    calibrado = (taxa_filtrado >= 0.50 and total_filtrado >= 2) or (taxa_raw >= 0.55 and total_raw >= 3)
    
    if calibrado:
        if taxa_filtrado >= 0.60:
            log.info(paint("   ✅ PRONTO PARA OPERAR - taxa excelente!", C.G))
        elif taxa_filtrado >= 0.55:
            log.info(paint("   ✅ PRONTO PARA OPERAR!", C.G))
        else:
            log.info(paint("   ⚠️ Taxa moderada - operando com cautela", C.Y))
    else:
        log.info(paint("   ⛔ Mercado MUITO difícil - filtros rigorosos aplicados", C.R))
    
    log.info("=" * 60)
    
    # ===== SALVAR SINAIS NO HISTÓRICO ACUMULADO =====
    if BACKTEST_USE_ACCUMULATED and todos_sinais:
        try:
            backtest_history_add_signals(todos_sinais)
            hist_stats = backtest_history_analyze()
            log.info(paint(f"📚 HISTÓRICO ACUMULADO: {hist_stats['total']} amostras | WR={hist_stats['weighted_winrate']*100:.1f}% (ponderado)", C.B))
        except Exception as e:
            log.warning(f"Erro ao salvar histórico: {e}")
    
    # ===== IA APRENDE COM O BACKTEST (ATUAL + HISTÓRICO) =====
    if AI_LEARN_FROM_BACKTEST and IA_ON:
        try:
            # Carregar stats da IA
            stats_backtest = _safe_load_json(AI_STATS_FILE)
            if stats_backtest is None:
                stats_backtest = {"meta": {"total": 0}, "arms": {}, "patterns": {}}
            
            n_learned_current = 0
            n_learned_history = 0
            
            # 1. Aprender com sinais do backtest ATUAL (peso normal)
            if todos_sinais:
                n_learned_current = ai_learn_from_backtest_batch(todos_sinais, stats_backtest)
            
            # 2. Aprender com HISTÓRICO ACUMULADO (peso temporal decrescente)
            if BACKTEST_USE_ACCUMULATED:
                n_learned_history = ai_learn_from_accumulated_history(stats_backtest)
            
            # Salvar stats atualizados
            _safe_save_json(AI_STATS_FILE, stats_backtest)
            
            # Retreinar LightGBM se tiver amostras suficientes
            if LGBM_ON and len(lgbm_data) >= LGBM_MIN_SAMPLES:
                lgbm_train()
            
            log.info(paint(f"🧠 IA APRENDEU: {n_learned_current} sinais atuais + {n_learned_history} do histórico (decay temporal)", C.G))
        except Exception as e:
            import traceback
            log.warning(f"Erro ao treinar IA com backtest: {e}")
            log.warning(f"Traceback: {traceback.format_exc()}")
    elif not IA_ON:
        log.warning(paint("⚠️ IA desligada (WS_AI_ON=0) - backtest não treina IA!", C.Y))
    
    # Salvar estado
    try:
        tuner_state = {
            "last_update": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "raw_signals": total_raw,
            "raw_accuracy": taxa_raw,
            "filtered_signals": total_filtrado,
            "filtered_accuracy": taxa_filtrado,
            "thresholds": {"min_score": GATE_MIN_SCORE, "min_context": GATE_CONTEXT_VERY_BAD},
            "analysis": {
                "avg_ctx_win": avg_ctx_win if wins_list else None,
                "avg_ctx_loss": avg_ctx_loss if losses_list else None,
                "avg_score_win": avg_score_win if wins_list else None,
                "avg_score_loss": avg_score_loss if losses_list else None
            },
            "filtros_por_ativo": {k: v for k, v in filtros_por_ativo.items()},
            "ai_learned_from_backtest": AI_LEARN_FROM_BACKTEST
        }
        with open(AUTO_TUNER_FILE, "w", encoding="utf-8") as f:
            json.dump(tuner_state, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    
    backtest_return = {
        "sinais": total_filtrado,
        "wins": wins_filtrado,
        "losses": total_filtrado - wins_filtrado,
        "taxa_acerto": taxa_filtrado,
        "calibrado": calibrado,
        "ajustes": ajustes,
        "filtros_por_ativo": filtros_por_ativo,
        "use_trendline": use_trendline_filter
    }
    
    # ===== CALIBRAÇÃO COM CLAUDE - DESATIVADO (IA local faz isso) =====
    # Para reativar: setar CLAUDE_CALIBRATE_ON = True no topo do arquivo
    if False and CLAUDE_AVAILABLE and CLAUDE_CALIBRATE_ON:
        try:
            log.info(paint("🧠 [CLAUDE] Analisando resultados do backtest para recalibrar...", C.B))
            claude_result = calibrate_after_backtest(backtest_return)
            if claude_result and claude_result.get("applied"):
                applied = claude_result["applied"]
                log.info(paint(f"🎯 [CLAUDE] {len(applied)} parâmetros ajustados automaticamente | Confiança: {claude_result.get('confidence',0):.0%}", C.G))
                for param, (old, new) in applied.items():
                    log.info(paint(f"   🔧 {param}: {old} → {new}", C.G))
            elif claude_result:
                log.info(paint(f"ℹ️ [CLAUDE] Análise: {claude_result.get('analysis', 'OK')} | Sem ajustes necessários", C.B))
        except Exception as e:
            log.warning(f"[CLAUDE] Erro na calibração: {e}")
    
    return backtest_return

# ===================== MAIN =====================
def main():
    iq: Optional[BrokerAPI] = None
    iq = ensure_connected(iq)

    global MIN_CONFLUENCE
    if "MIN_CONFLUENCE" not in globals():
        MIN_CONFLUENCE = int(os.getenv("WS_MIN_CONFLUENCE", "2"))

    log.info("=" * 60)
    log.info(f"🚀 WS_AUTO_AI — DOM Forex Perfect Zones (M1) + ENSEMBLE IA [{_BROKER_NAME}]")
    log.info("=" * 60)
    log.info("✅ Analisa TODOS os ativos de uma vez")
    log.info("✅ Entra quando aparecer sinal confirmado")
    log.info("✅ IA ENSEMBLE: Bayesiano + LightGBM (Gradient Boosting)")
    if CLAUDE_AVAILABLE and CLAUDE_CALIBRATE_ON:
        log.info("✅ CLAUDE CALIBRATOR: Recalibração inteligente pós-backtest")
    log.info("=" * 60)
    
    # Mostrar modo da IA
    if IA_MODE == "learning":
        log.info(paint("🧠 MODO: LEARNING COM CONFLUÊNCIA - Tendência + Zona + Confiança obrigatórios", C.G))
        log.info(paint(f"   → Score mínimo: {GATE_MIN_SCORE:.2f} | Confluência mínima: {MIN_CONFLUENCE}", C.B))
        log.info(paint(f"   → Contexto mínimo: {GATE_CONTEXT_VERY_BAD:.2f} | Ensemble: {ENSEMBLE_MODE}", C.B))
        log.info(paint(f"   → Trendline: {'OBRIGATÓRIA' if REQUIRE_TRENDLINE else 'opcional'} | EffA mínima: {MIN_ENTRY_EFF:.2f}", C.B))
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
        # Mostrar status de confiabilidade
        if lgbm_reliable:
            log.info(paint(f"[LGBM] ✅ Modelo CONFIÁVEL (Val={lgbm_val_accuracy:.1f}%)", C.G))
        else:
            log.info(paint(f"[LGBM] ⚠️ Modelo NÃO confiável → usando APENAS Bayes até melhorar", C.Y))
    else:
        log.info("[LGBM] OFF - usando apenas Bayesiano")
    
    # Carregar CNN Pattern Detector
    if CNN_ON:
        cnn_model = get_cnn(data_dir=".")
        cnn_stats = cnn_model.get_stats()
        log.info(paint(f"[CNN] ON | amostras={cnn_stats['total_samples']} | treinos={cnn_stats['total_trained']} | peso={CNN_WEIGHT:.0%}", C.B))
        if cnn_stats["reliable"]:
            log.info(paint(f"[CNN] ✅ Modelo treinado (WR={cnn_stats['win_rate']*100:.1f}%)", C.G))
        else:
            log.info(paint(f"[CNN] ⚠️ Aquecendo ({cnn_stats['total_samples']}/30 amostras)", C.Y))
    else:
        log.info("[CNN] OFF")
    
    # Carregar histórico de backtest acumulado
    if BACKTEST_USE_ACCUMULATED:
        backtest_history_load()
        hist_stats = backtest_history_analyze()
        if hist_stats["total"] > 0:
            log.info(paint(f"📚 HISTÓRICO CARREGADO: {hist_stats['total']} amostras | WR={hist_stats['winrate']*100:.1f}% | WR_ponderado={hist_stats['weighted_winrate']*100:.1f}%", C.G))
        else:
            log.info(paint("📚 HISTÓRICO: Vazio (será preenchido com backtests)", C.B))
    
    # Modo de operação após LOSS
    if BACKTEST_ON_LOSS:
        log.info(paint(f"🔄 MODO: BACKTEST ON LOSS - Após LOSS faz backtest 30min, recalibra filtros e continua", C.G))
    elif RETRAIN_ON_LOSS:
        log.info(paint(f"🔄 MODO: RETRAIN & CONTINUE - Após LOSS pausa {PAUSE_AFTER_LOSS_SECONDS}s, retreina e continua", C.G))
    else:
        log.info(paint("⏹️ MODO: STOP após LOSS - Bot para e precisa reiniciar manualmente", C.Y))

    # IA aprende com backtest
    if AI_LEARN_FROM_BACKTEST:
        mode_hist = "+ HISTÓRICO ACUMULADO" if BACKTEST_USE_ACCUMULATED else ""
        log.info(paint(f"🧠 IA APRENDE COM BACKTEST: ON | peso={AI_BACKTEST_WEIGHT:.1f}x {mode_hist}", C.G))
    else:
        log.info(paint("🧠 IA aprende apenas com trades reais", C.B))

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

    # ========== BACKTEST INTELIGENTE ANTES DE OPERAR ==========
    mercado_ok = True  # Flag para indicar se o mercado está bom
    ultima_verificacao_mercado = time.time()
    INTERVALO_REVERIFICACAO = 120  # Re-verificar mercado a cada 2 minutos se estiver ruim
    mercado_tentativas_falhas = 0  # Contador de re-verificações que falharam
    
    # Variável global para controlar filtro de trendline dinamicamente
    global USE_TRENDLINE_FILTER, ativos_analisados_backtest
    consecutive_skips = 0  # Contador de skips consecutivos para auto-relax
    
    try:
        ativos_backtest = obter_top_ativos_otc(iq)
        if ativos_backtest:
            backtest_result = backtest_antes_de_operar(iq, ativos_backtest, n_candles=90)
            taxa_backtest = backtest_result.get("taxa_acerto", 0.0)
            
            # IMPORTANTE: Salvar quais ativos foram analisados no backtest
            ativos_analisados_backtest = list(ativos_backtest)
            
            # Atualizar USE_TRENDLINE_FILTER baseado no backtest
            USE_TRENDLINE_FILTER = backtest_result.get("use_trendline", REQUIRE_TRENDLINE)
            lt_status = "ATIVADO" if USE_TRENDLINE_FILTER else "DESATIVADO"
            log.info(paint(f"📊 Filtro de Trendline: {lt_status} (automático pelo backtest)", C.B))
            
            # Mostrar filtros finais que serão usados
            log.info(paint(f"🎯 FILTROS ATIVOS: ctx≥{GATE_CONTEXT_VERY_BAD:.2f} score≥{GATE_MIN_SCORE:.2f}", C.G))
            
            if taxa_backtest < BACKTEST_MIN_WINRATE:
                # Em vez de bloquear totalmente, operar com cautela extra
                mercado_ok = True  # PERMITIR operar mesmo com mercado difícil
                log.warning(paint(f"⚠️ MERCADO DIFÍCIL: Taxa do backtest: {taxa_backtest*100:.1f}% - OPERANDO com filtros adaptativos", C.Y))
                log.info(paint("   → IA vai selecionar APENAS os melhores sinais disponíveis", C.B))
            elif backtest_result["calibrado"]:
                log.info(paint("✅ Filtros otimizados - iniciando operações!", C.G))
            else:
                log.info(paint("⚠️ Mercado difícil mas filtros calibrados - operando com cautela", C.Y))
    except Exception as e:
        log.warning(f"Erro no backtest inicial: {e}")
    # =========================================================

    _last_loop_activity = time.time()  # Watchdog: última atividade do loop

    while True:
        # ===== WATCHDOG: detectar loop travado =====
        loop_start = time.time()
        if loop_start - _last_loop_activity > 180:  # 3 min sem atividade (ajustado para M1)
            log.warning(paint("⚠️ WATCHDOG: Loop parado há 3+ min - reconectando...", C.Y))
            try:
                iq = ensure_connected(iq)
            except Exception as e:
                log.warning(f"Watchdog reconexão falhou: {e}")
                time.sleep(15)
                continue
        _last_loop_activity = loop_start
        # ============================================

        iq = ensure_connected(iq)

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

        # ========== VERIFICAR SE O MERCADO ESTÁ BOM ==========
        if not mercado_ok:
            # Re-verificar mercado periodicamente
            if time.time() - ultima_verificacao_mercado >= INTERVALO_REVERIFICACAO:
                log.info(paint("🔄 Re-verificando condições do mercado...", C.B))
                try:
                    ativos_reverif = obter_top_ativos_otc(iq)
                    if ativos_reverif:
                        backtest_reverif = backtest_antes_de_operar(iq, ativos_reverif, n_candles=90)
                        taxa_reverif = backtest_reverif.get("taxa_acerto", 0.0)
                        ultima_verificacao_mercado = time.time()
                        
                        # IMPORTANTE: Atualizar ativos analisados
                        ativos_analisados_backtest.clear()
                        ativos_analisados_backtest.extend(ativos_reverif)
                        
                        # Atualizar USE_TRENDLINE_FILTER com resultado do backtest
                        USE_TRENDLINE_FILTER = backtest_reverif.get("use_trendline", REQUIRE_TRENDLINE)
                        lt_status = "ATIVADO" if USE_TRENDLINE_FILTER else "DESATIVADO"
                        log.info(paint(f"📊 Filtro de Trendline: {lt_status}", C.B))
                        
                        if taxa_reverif >= BACKTEST_MIN_WINRATE:
                            mercado_ok = True
                            mercado_tentativas_falhas = 0
                            log.info(paint(f"✅ MERCADO MELHOROU! Taxa: {taxa_reverif*100:.1f}% - Retomando operações!", C.G))
                        else:
                            mercado_tentativas_falhas += 1
                            # Após 2 tentativas, forçar operação com cautela
                            if mercado_tentativas_falhas >= 2:
                                mercado_ok = True
                                log.warning(paint(f"🔄 Mercado difícil ({taxa_reverif*100:.1f}%) mas IA vai operar com os melhores sinais", C.Y))
                                log.info(paint("   → Filtros adaptativos ativados - IA seleciona apenas sinais fortes", C.B))
                            else:
                                log.warning(paint(f"⚠️ Mercado ainda difícil: {taxa_reverif*100:.1f}% - próxima verificação em {INTERVALO_REVERIFICACAO//60} min", C.Y))
                except Exception as e:
                    log.warning(f"Erro ao re-verificar mercado: {e}")
            
            # Se mercado não está ok, esperar
            if not mercado_ok:
                time.sleep(60)  # Esperar 1 min antes de verificar novamente
                continue
        # =====================================================

        ativos = obter_top_ativos_otc(iq)
        if not ativos:
            log.warning("Sem ativos com payout mínimo. Tentando em 10s...")
            time.sleep(10)
            continue

        # ===================== VERIFICAR NOVOS ATIVOS =====================
        # Detectar se há ativos novos que não foram analisados no backtest
        novos_ativos = [a for a in ativos if a not in ativos_analisados_backtest]
        if novos_ativos:
            log.info(paint(f"🔄 {len(novos_ativos)} NOVOS ATIVOS detectados: {novos_ativos[:3]}...", C.Y))
            log.info(paint("   → Fazendo backtest nos novos ativos antes de operar...", C.B))
            try:
                backtest_novos = backtest_antes_de_operar(iq, novos_ativos, n_candles=90)
                # Adicionar aos ativos já analisados
                ativos_analisados_backtest.extend(novos_ativos)
                log.info(paint(f"✅ Backtest concluído para novos ativos | Taxa: {backtest_novos.get('taxa_acerto', 0)*100:.1f}%", C.G))
            except Exception as e:
                log.warning(f"Erro ao fazer backtest em novos ativos: {e}")
                # Mesmo com erro, adiciona para não ficar em loop
                ativos_analisados_backtest.extend(novos_ativos)
        # ===================================================================

        wait_until_minus(TF_M1, DECIDIR_ANTES_FECHAR_SEC)

        best_trade, best_any = escolher_melhor_setup(iq, ativos)

        if not best_trade:
            if best_any:
                sc, at, st, _av, _df = best_any
                log.info(paint(
                    f"[SKIP] nenhum setup passou | melhor={at} score={sc:.2f} | {','.join(st.get('reasons', []))}",
                    C.Y
                ))
                cooldown[at] = time.time()
            else:
                log.info(paint("[SKIP] nenhum ativo analisável no minuto", C.Y))

            wait_for_next_open(TF_M1)
            continue

        score, ativo, setup, atr_val, df_candles = best_trade
        score = float(score)

        # ===================== FILTROS POR ATIVO =====================
        # Verificar se ativo está habilitado e usar filtros específicos
        if ativo in filtros_por_ativo:
            filtro_ativo = filtros_por_ativo[ativo]
            
            # Verificar se ativo está desabilitado
            if not filtro_ativo.get("habilitado", True):
                log.info(paint(
                    f"[ATIVO-SKIP] {ativo} DESABILITADO | {filtro_ativo.get('motivo', '?')} | sinais={filtro_ativo.get('sinais', 0)}",
                    C.R
                ))
                consecutive_skips += 1
                wait_for_next_open(TF_M1)
                cooldown[ativo] = time.time()
                continue
            
            # Usar filtros específicos do ativo
            min_score_ativo = filtro_ativo.get("min_score", GATE_MIN_SCORE)
            min_ctx_ativo = filtro_ativo.get("min_ctx", GATE_CONTEXT_VERY_BAD)
            ctx = setup.get("market_quality", 0)
            
            if ctx < min_ctx_ativo:
                log.info(paint(
                    f"[CTX-SKIP] {ativo} | ctx={ctx:.2f}<{min_ctx_ativo:.2f} (específico) | score={score:.2f}",
                    C.Y
                ))
                consecutive_skips += 1
                # Auto-relax: se muitos skips por CTX, reduzir min_ctx do ativo
                if AUTO_RELAX_ON_SKIPS and consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                    filtro_ativo["min_ctx"] = max(0.25, filtro_ativo["min_ctx"] - 0.05)
                    consecutive_skips = 0
                    log.info(paint(f"🔄 AUTO-RELAX: Reduzindo min_ctx de {ativo} para {filtro_ativo['min_ctx']:.2f}", C.G))
                wait_for_next_open(TF_M1)
                cooldown[ativo] = time.time()
                continue
            
            if score < min_score_ativo:
                log.info(paint(
                    f"[SCORE-SKIP] {ativo} | score={score:.2f}<{min_score_ativo:.2f} (específico)",
                    C.Y
                ))
                consecutive_skips += 1
                # Auto-relax: se muitos skips por SCORE, reduzir min_score do ativo
                if AUTO_RELAX_ON_SKIPS and consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                    filtro_ativo["min_score"] = max(0.38, filtro_ativo["min_score"] - 0.03)
                    consecutive_skips = 0
                    log.info(paint(f"🔄 AUTO-RELAX: Reduzindo min_score de {ativo} para {filtro_ativo['min_score']:.2f}", C.G))
                wait_for_next_open(TF_M1)
                cooldown[ativo] = time.time()
                continue
        else:
            # Ativo não tem filtros específicos - usar globais COM verificação de contexto
            ctx = setup.get("market_quality", 0)
            
            # Verificar contexto ANTES de tudo
            if ctx < GATE_CONTEXT_VERY_BAD:
                log.info(paint(
                    f"[CTX-SKIP] {ativo} | ctx={ctx:.2f}<{GATE_CONTEXT_VERY_BAD:.2f} (global) | score={score:.2f}",
                    C.Y
                ))
                consecutive_skips += 1
                wait_for_next_open(TF_M1)
                cooldown[ativo] = time.time()
                continue
            
            if score < GATE_SOFT_SCORE:
                log.info(paint(
                    f"[SKIP] {ativo} | score={score:.2f} | {','.join(setup.get('reasons', []))}",
                    C.Y
                ))
                consecutive_skips += 1
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

        # ===================== FILTROS DE QUALIDADE EXTRA =====================
        # Filtro 1: Exigir linha de tendência (LTA/LTB) confirmando a direção
        # Usa USE_TRENDLINE_FILTER (ajustado dinamicamente pelo backtest)
        if USE_TRENDLINE_FILTER and not setup.get("has_lt", False):
            dir_tipo = "LTA" if setup.get("dir") == "CALL" else "LTB"
            ctx_lt = setup.get("market_quality", 0)
            
            # OVERRIDE: No modo LEARNING, se score >= 0.90 E contexto bom E confluência >= 3, permite entrada sem LT
            if IA_MODE == "learning" and score >= 0.90 and ctx_lt >= 0.55 and setup.get("sr_proximity", 0) > 0.40:
                log.info(paint(
                    f"[LT-OVERRIDE] {ativo} | sem_{dir_tipo} mas score={score:.2f} ctx={ctx_lt:.2f} SR={setup.get('sr_proximity',0):.2f} | PERMITINDO",
                    C.G
                ))
            else:
                if IA_MODE == "learning" and score >= 0.90 and ctx_lt < 0.55:
                    log.info(paint(
                        f"[LT-SKIP] {ativo} | sem_{dir_tipo} | score={score:.2f} mas ctx={ctx_lt:.2f}<0.42 (mercado ruim)",
                        C.Y
                    ))
                else:
                    log.info(paint(
                        f"[LT-SKIP] {ativo} | sem_{dir_tipo} | score={score:.2f} | {','.join(setup.get('reasons', []))}",
                        C.Y
                    ))
                # Auto-relax: se muitos skips seguidos por LT, desativar
                consecutive_skips += 1
                if AUTO_RELAX_ON_SKIPS and consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                    USE_TRENDLINE_FILTER = False
                    consecutive_skips = 0
                    log.info(paint(f"🔄 AUTO-RELAX: Desativando filtro de Trendline após {MAX_CONSECUTIVE_SKIPS} skips", C.G))
                wait_for_next_open(TF_M1)
                cooldown[ativo] = time.time()
                continue
        
        # Filtro 2: Eficiência mínima do setup
        # OTC M1 tem eficiência direcional naturalmente baixa
        # Se score é alto (>= 0.55), aceitar effA menor (0.05 vs 0.10)
        effA = setup.get("effA", 0.0)
        eff_threshold = MIN_ENTRY_EFF * 0.5 if score >= 0.55 else MIN_ENTRY_EFF
        if effA < eff_threshold:
            log.info(paint(
                f"[EFF-SKIP] {ativo} | effA={effA:.2f}<{eff_threshold:.2f} | score={score:.2f}",
                C.Y
            ))
            wait_for_next_open(TF_M1)
            cooldown[ativo] = time.time()
            continue

        final_dir = str(setup["dir"])
        sinal_invertido = False  # Flag para não poluir IA com sinais invertidos
        
        # Info sobre filtros usados
        if ativo in filtros_por_ativo:
            filtro = filtros_por_ativo[ativo]
            filtro_info = f"[{ativo} filtros: ctx≥{filtro.get('min_ctx', 0):.2f} score≥{filtro.get('min_score', 0):.2f} taxa={filtro.get('taxa', 0)*100:.0f}%]"
            log.info(paint(filtro_info, C.B))
        
        log.info(paint(
            f"[SINAL-HARD] {ativo} -> {final_dir} | score={score:.2f} | ATR={atr_val:.6f} | {','.join(setup.get('reasons', []))}",
            dir_color(final_dir)
        ))

        if IA_ON:
            # ENSEMBLE: Combina Bayesiano + LightGBM + CNN
            ens = ensemble_predict(ativo, setup, stats, df=df_candles)
            bayes_prob = float(ens["bayes_prob"])
            lgbm_prob = float(ens["lgbm_prob"])
            cnn_prob_val = float(ens.get("cnn_prob", 0.5))
            ensemble_prob = float(ens["ensemble_prob"])
            should_trade = bool(ens["should_trade"])
            ens_reason = str(ens["reason"])
            n_arm = int(ens.get("n_arm", 0))
            
            # Sufixo CNN para logs
            cnn_log = f" | CNN={cnn_prob_val:.2f}" if CNN_ON and cnn_model is not None else ""
            
            # Log do ensemble
            if LGBM_ON and lgbm_model is not None and lgbm_reliable:
                log.info(paint(
                    f"[ENSEMBLE] {ativo} {final_dir} | Bayes={bayes_prob:.2f} | LGBM={lgbm_prob:.2f}{cnn_log} | Ens={ensemble_prob:.2f} | {ens_reason}",
                    C.B
                ))
            elif LGBM_ON and not lgbm_reliable:
                log.info(paint(
                    f"[BAYES-ONLY] {ativo} {final_dir} | prob={bayes_prob:.2f} (n={n_arm}){cnn_log} | LGBM desabilitado (Val={lgbm_val_accuracy:.1f}%<50%) | {ens_reason}",
                    C.Y
                ))
            else:
                log.info(paint(
                    f"[BAYES] {ativo} {final_dir} | prob={bayes_prob:.2f} (n={n_arm}){cnn_log} | {ens_reason}",
                    C.B
                ))
            
            # Decisão do ensemble
            if not should_trade:
                if lgbm_prob < 0.30:
                    log.info(paint(f"[IA-BLOCK] {ativo} {final_dir} | LGBM={lgbm_prob:.2f}<0.30 = PERIGO | {ens_reason}", C.R))
                else:
                    log.info(paint(f"[IA-SKIP] {ativo} {final_dir} | ensemble fraco | {ens_reason}", C.Y))
                wait_for_next_open(TF_M1)
                cooldown[ativo] = time.time()
                continue
            
            # GATE EXTRA: Verificação rigorosa do contexto vs ensemble vs confluência
            ctx_val = float(setup.get("market_quality", 0.40))
            entry_conf_val = float(setup.get("entry_confidence", 0.50))
            sr_prox_gate = float(setup.get("sr_proximity", 0.0))
            sr_tq_gate = int(setup.get("sr_touches", 0))
            sr_forte = sr_prox_gate > 0.60 and sr_tq_gate >= 4  # S/R forte = critério mais exigente
            sr_basico = sr_tq_gate >= 3 and float(setup.get("sr_weight", 0.0)) >= 4.0  # S/R com zona confirmada
            confluence_count = int(setup.get("confluence_bonus", 0) > 0.04) + int(setup.get("has_lt", False)) + int(sr_prox_gate > 0.30)
            lt_conf = float(setup.get("lt_confluence", 0.0))
            
            # GATE 1: Contexto ruim BLOQUEIA a menos que ensemble muito alto + SR forte
            if ctx_val < 0.40 and not ((sr_forte or sr_basico) and ensemble_prob >= 0.58):
                log.info(paint(f"[CTX-GATE] {ativo} {final_dir} | ctx_ruim={ctx_val:.2f} | ens={ensemble_prob:.2f}", C.Y))
                wait_for_next_open(TF_M1)
                cooldown[ativo] = time.time()
                continue
            # GATE 2: Contexto mediano + sem zona confirmada = precisa score alto
            # Trader profissional: zona confirmada (S/R 3+ toques) compensa candle fraco
            score_gate2 = float(setup.get("score", 0.0))
            has_lt_g2 = bool(setup.get("has_lt", False))
            if ctx_val < 0.50 and not sr_forte and not sr_basico and not has_lt_g2 and score_gate2 < 0.55:
                log.info(paint(f"[CTX-GATE] {ativo} {final_dir} | ctx_med+sem_zona | ctx={ctx_val:.2f},sc={score_gate2:.2f}", C.Y))
                wait_for_next_open(TF_M1)
                cooldown[ativo] = time.time()
                continue
            # GATE 3: Trendline fraca (<0.5) sem S/R forte = não opera
            if lt_conf < 0.5 and not setup.get("has_lt", False) and sr_prox_gate < 0.40:
                log.info(paint(f"[TREND-GATE] {ativo} {final_dir} | sem_tendência_forte+sem_SR | lt={lt_conf:.2f},sr={sr_prox_gate:.2f}", C.Y))
                wait_for_next_open(TF_M1)
                cooldown[ativo] = time.time()
                continue
            # GATE 4: Verificação PROFISSIONAL - foco na ZONA, não no candle
            # Trader profissional: candle neutro em zona forte = ENTRADA VÁLIDA
            # Só bloqueia se TUDO é fraco: zona fraca + candle fraco + ensemble fraco
            score_val_gate = float(setup.get("score", 0.0))
            sr_tq_g4 = int(setup.get("sr_touches", 0))
            sr_w_g4 = float(setup.get("sr_weight", 0.0))
            lt_pts_g4 = int(setup.get("lt_points", setup.get("pb_len", 0)))
            has_lt_g4 = bool(setup.get("has_lt", False))
            
            # Zona forte = S/R com toques + LT alinhada (trader profissional entra aqui)
            zona_forte = (sr_tq_g4 >= 3 and sr_w_g4 >= 4.0) or (has_lt_g4 and lt_pts_g4 >= 3)
            # Setup de qualidade = score bom + contexto aceitável
            setup_ok = score_val_gate >= 0.45 and ctx_val >= 0.50
            # Setup forte = alta confluência
            setup_forte = score_val_gate >= 0.55 and ctx_val >= 0.55 and confluence_count >= 2
            
            # SÓ BLOQUEIA se: candle fraco E zona fraca E setup fraco E ensemble baixo
            ens_gate_threshold = 0.52 if (not lgbm_reliable or not LGBM_ON) else 0.60
            if entry_conf_val < 0.30 and not zona_forte and not setup_forte and ensemble_prob < ens_gate_threshold:
                log.info(paint(f"[CONF-GATE] {ativo} {final_dir} | tudo_fraco | conf={entry_conf_val:.2f},zona={zona_forte},sc={score_val_gate:.2f},ens={ensemble_prob:.2f}", C.Y))
                wait_for_next_open(TF_M1)
                cooldown[ativo] = time.time()
                continue
            
            # LOG S/R quando detectado
            if sr_prox_gate > 0:
                sr_reason_log = str(setup.get("sr_reason", "?"))
                log.info(paint(f"[SR-ZONE] {ativo} {final_dir} | {sr_reason_log} | bonus={setup.get('sr_bonus',0):.2f}", C.G if sr_forte else C.B))

        wait_for_next_open(TF_M1)

        stake = calcular_stake_dinamico(iq, STAKE_FIXA)
        log.info(paint(f"[{ativo}] 💵 Stake calculado: {stake:.2f}", C.B))

        op = enviar_ordem(iq, ativo, final_dir, stake)

        if not op:
            log.error(paint(f"[{ativo}] ❌ falhou enviar ordem (TURBO/DIGITAL).", C.R))
            cooldown[ativo] = time.time()
            continue

        # Resetar contador de skips - uma operação foi feita
        consecutive_skips = 0
        
        op_type, op_id = op
        log.info(paint(
            f"[{ativo}] ✅ ORDEM ENVIADA {final_dir} exp={EXP_FIXA}m ({op_type}) | stake={stake:.2f}",
            dir_color(final_dir)
        ))

        res = wait_result(iq, op_type, op_id)

        total += 1
        global global_consecutive_losses
        
        should_pause_and_continue = False  # Flag para pausar e continuar após LOSS
        
        if res > 0:
            wins += 1
            log.info(paint(f"[{ativo}] ✅ WIN {res:.2f}$", C.G))
            # Reset counters após WIN
            consecutive_losses[ativo] = 0
            global_consecutive_losses = 0
            
            # RECOMPENSAR ATIVO QUE DEU WIN - relaxar filtros levemente
            if ativo in filtros_por_ativo:
                filtro = filtros_por_ativo[ativo]
                # Relaxar filtros levemente (mas não muito)
                filtro["min_ctx"] = max(0.35, filtro.get("min_ctx", 0.40) - 0.01)
                filtro["min_score"] = max(0.50, filtro.get("min_score", 0.55) - 0.01)
                filtro["taxa"] = min(1.0, filtro.get("taxa", 0.50) + 0.05)  # Aumentar taxa estimada
                filtro["habilitado"] = True  # Garantir que está habilitado
                filtro["motivo"] = "ok_win"
        elif res < 0:
            log.info(paint(f"[{ativo}] ❌ LOSS {res:.2f}$", C.R))
            # Incrementar contadores de LOSS
            consecutive_losses[ativo] = consecutive_losses.get(ativo, 0) + 1
            global_consecutive_losses += 1
            # Aplicar cooldown especial após LOSS
            cooldown_loss[ativo] = time.time()
            
            # PENALIZAR FILTROS DO ATIVO ESPECÍFICO QUE DEU LOSS
            if ativo in filtros_por_ativo:
                filtro = filtros_por_ativo[ativo]
                # Apertar filtros do ativo que deu LOSS
                filtro["min_ctx"] = min(0.50, filtro.get("min_ctx", 0.40) + 0.03)
                filtro["min_score"] = min(0.65, filtro.get("min_score", 0.55) + 0.03)
                filtro["taxa"] = max(0.0, filtro.get("taxa", 0.50) - 0.10)  # Reduzir taxa estimada
                
                # Se taxa ficar muito baixa, desabilitar ativo
                if filtro["taxa"] < 0.35:
                    filtro["habilitado"] = False
                    filtro["motivo"] = f"desabilitado_loss_consec_{consecutive_losses[ativo]}"
                    log.warning(paint(f"⛔ {ativo} DESABILITADO após LOSS! Taxa estimada muito baixa", C.R))
                else:
                    log.info(paint(f"🔧 {ativo} filtros apertados: ctx≥{filtro['min_ctx']:.2f} score≥{filtro['min_score']:.2f}", C.Y))
            
            # RETREINO SEVERO: aplicar penalidade extra no padrão (apenas se não usar backtest)
            if IA_ON and not BACKTEST_ON_LOSS:
                ai_retrain_on_loss(ativo, setup, stats)
                log.warning(paint(f"[RETRAIN] Padrão penalizado com {RETRAIN_PENALTY:.0%} - IA aprendendo com erro", C.Y))
            
            # BACKTEST_ON_LOSS: fazer backtest de 30 min para recalibrar filtros
            if BACKTEST_ON_LOSS:
                log.info(paint("="*60, C.Y))
                log.info(paint("🔄 RECALIBRANDO FILTROS COM BACKTEST DE 30 MINUTOS...", C.Y))
                log.info(paint("="*60, C.Y))
                try:
                    # Usar ativos atuais para o backtest pós-LOSS
                    ativos_recalibrar = obter_top_ativos_otc(iq)
                    backtest_result_loss = backtest_antes_de_operar(iq, ativos_recalibrar, n_candles=90)  # 90 candles (otimizado)
                    taxa_backtest_loss = backtest_result_loss.get("taxa_acerto", 0.0)
                    
                    # IMPORTANTE: Atualizar ativos analisados (backtest completo, não incremental)
                    ativos_analisados_backtest.clear()
                    ativos_analisados_backtest.extend(ativos_recalibrar)
                    
                    # Atualizar USE_TRENDLINE_FILTER com resultado do backtest pós-LOSS
                    USE_TRENDLINE_FILTER = backtest_result_loss.get("use_trendline", REQUIRE_TRENDLINE)
                    lt_status = "ATIVADO" if USE_TRENDLINE_FILTER else "DESATIVADO"
                    log.info(paint(f"📊 Filtro de Trendline: {lt_status}", C.B))
                    
                    # Verificar se mercado ainda está bom
                    if taxa_backtest_loss < BACKTEST_MIN_WINRATE:
                        mercado_ok = False
                        log.warning(paint(f"⛔ MERCADO RUIM APÓS LOSS! Taxa: {taxa_backtest_loss*100:.1f}% < {BACKTEST_MIN_WINRATE*100:.1f}%", C.R))
                        log.warning(paint("   → Aguardando mercado melhorar antes de continuar...", C.Y))
                    elif taxa_backtest_loss < 0.55:
                        # Mercado mediano - apertar filtros globais como proteção extra
                        mercado_ok = True
                        log.warning(paint(f"⚠️ Mercado MEDIANO após recalibração: {taxa_backtest_loss*100:.1f}% - filtros mais rigorosos", C.Y))
                    else:
                        mercado_ok = True
                        log.info(paint(f"✅ Mercado OK após recalibração: {taxa_backtest_loss*100:.1f}%", C.G))
                    ultima_verificacao_mercado = time.time()
                except Exception as e:
                    log.error(f"Erro no backtest pós-LOSS: {e}")
                should_pause_and_continue = True
                log.info(paint(f"⏳ Filtros recalibrados - pausando {PAUSE_AFTER_LOSS_SECONDS}s antes de continuar...", C.Y))
            # RETRAIN_ON_LOSS: pausar, retreinar e continuar automaticamente (fallback)
            elif RETRAIN_ON_LOSS:
                should_pause_and_continue = True
                log.warning(paint(f"[RETRAIN] IA retreinada - pausando {PAUSE_AFTER_LOSS_SECONDS}s antes de continuar", C.Y))
            elif global_consecutive_losses >= MAX_CONSECUTIVE_LOSS:
                should_pause_and_continue = True
                log.warning(paint(f"[PAUSE] {global_consecutive_losses} losses consecutivos - pausando {PAUSE_AFTER_LOSS_SECONDS}s", C.Y))
        else:
            log.info(paint(f"[{ativo}] ⚪ EMPATE {res:.2f}$", C.B))

        # Só treina IA se o sinal NÃO foi invertido (para não poluir o aprendizado)
        if sinal_invertido:
            log.info(paint(f"[🔄 NO-TRAIN] Sinal foi INVERTIDO - NÃO treinando IA com este resultado", C.Y))
        else:
            if IA_ON:
                ai_update(ativo, setup, res, stats)
                _safe_save_json(AI_STATS_FILE, stats)
            
            # Adiciona amostra ao LightGBM para aprendizado
            if LGBM_ON:
                lgbm_add_sample(setup, res)
            
            # Adiciona amostra ao CNN Pattern Detector para aprendizado
            if CNN_ON and cnn_model is not None and df_candles is not None:
                try:
                    cnn_model.add_sample(df_candles, final_dir, win=(res > 0))
                except Exception as e:
                    log.warning(f"[CNN] Erro ao adicionar amostra: {e}")
        
        # PAUSAR e CONTINUAR automaticamente após LOSS (não para o bot)
        if should_pause_and_continue:
            # Escalar pausa com losses consecutivos: 60s, 120s, 180s...
            pause_multiplier = max(1, global_consecutive_losses)
            pause_time = min(PAUSE_AFTER_LOSS_SECONDS * pause_multiplier, 300)  # máx 5 min
            log.info(paint("=" * 60, C.Y))
            if BACKTEST_ON_LOSS:
                log.info(paint("⏸️ PAUSANDO APÓS LOSS - FILTROS RECALIBRADOS VIA BACKTEST", C.Y))
            else:
                log.info(paint("⏸️ PAUSANDO APÓS LOSS - IA RETREINADA (BAYES + LGBM)", C.Y))
            log.info(paint(f"📊 RESUMO: trades={total} wins={wins} acc={(wins/max(1,total))*100:.1f}%", C.Y))
            log.info(paint(f"⏳ Esperando {pause_time}s antes de continuar (x{pause_multiplier})...", C.Y))
            log.info(paint("=" * 60, C.Y))
            time.sleep(pause_time)
            global_consecutive_losses = 0  # Reset após pausa
            log.info(paint("\n▶️ RETOMANDO OPERAÇÕES - Filtros ajustados!\n", C.G))
            continue  # Continua o loop principal

        acc = (wins / max(1, total)) * 100.0

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
    _max_restarts = 5
    _restart_count = 0
    while True:
        try:
            main()
            _restart_count = 0  # Reset após execução bem-sucedida
        except (RuntimeError, ConnectionError, OSError) as e:
            _restart_count += 1
            log.warning(paint(f"⚠️ Conexão perdida: {e}", C.Y))
            if _restart_count >= _max_restarts:
                log.error(paint(f"❌ Máximo de {_max_restarts} restarts atingido. Encerrando.", C.R))
                break
            wait_sec = min(15 * _restart_count, 60)  # 15s, 30s, 45s, 60s
            log.info(paint(f"🔄 Reiniciando em {wait_sec}s... ({_restart_count}/{_max_restarts})", C.Y))
            time.sleep(wait_sec)
        except KeyboardInterrupt:
            log.info("Bot encerrado pelo usuário.")
            break
        except Exception as e:
            _restart_count += 1
            log.warning(paint(f"⚠️ Erro recuperável: {e}", C.Y))
            if _restart_count >= _max_restarts:
                log.error(paint(f"❌ Máximo de {_max_restarts} restarts atingido. Encerrando.", C.R))
                break
            wait_sec = min(20 * _restart_count, 90)  # 20s, 40s, 60s, 80s, 90s
            log.info(paint(f"🔄 Reiniciando em {wait_sec}s... ({_restart_count}/{_max_restarts})", C.Y))
            time.sleep(wait_sec)
