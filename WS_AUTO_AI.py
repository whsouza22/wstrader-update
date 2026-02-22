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
    import bullexapi.constants as _broker_constants
    _BROKER_NAME = "Bullex"
elif BROKER_TYPE == "casatrader":
    from casatraderapi.stable_api import Casa_Trader as BrokerAPI
    import casatraderapi.constants as _broker_constants
    _BROKER_NAME = "CasaTrader"
else:
    from iqoptionapi.stable_api import IQ_Option as BrokerAPI
    import iqoptionapi.constants as _broker_constants
    _BROKER_NAME = "IQ Option"
    BROKER_TYPE = "iq_option"

# SR Precision Strategy — S/R + Candle Features
from sr_precision_strategy import sr_precision_signal

# AI Loss Memory — IA generativa que analisa LOSSes e salva motivos
try:
    from ai_loss_memory import analyze_and_save_loss, get_loss_summary
    LOSS_MEMORY_ON = True
except ImportError:
    LOSS_MEMORY_ON = False
    log.warning("ai_loss_memory não encontrado — análise de LOSSes desabilitada")

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
NUM_ATIVOS = int(os.getenv("WS_NUM_ATIVOS", "40"))             # Pool inicial (busca 40)
NUM_ATIVOS_OPERAR = int(os.getenv("WS_NUM_ATIVOS_OPERAR", "10"))  # TOP N para operar (ranking por WR)
BACKTEST_MIN_WR_OPERAR = float(os.getenv("WS_BACKTEST_MIN_WR_OPERAR", "0.75"))  # SÓ opera ativos com WR >= 75% no backtest
PAYOUT_REFRESH_SEC = int(os.getenv("WS_PAYOUT_REFRESH", "180"))

EXP_FIXA = int(os.getenv("WS_EXP_MIN", "1"))
VALOR_MINIMO = float(os.getenv("WS_VALOR_MINIMO", "3"))
STAKE_FIXA = float(os.getenv("WS_STAKE", "5"))

# ===================== GESTÃO DE BANCA =====================
PERCENT_BANCA = float(os.getenv("WS_PERCENT_BANCA", "1.0"))  # 1% da banca por operação
META_LUCRO_PERCENT = float(os.getenv("WS_META_LUCRO", "1.5"))  # para com 1.5% de lucro
STOP_LOSS_PERCENT = float(os.getenv("WS_STOP_LOSS", "3.0"))  # para com 3% de perda (opcional)
USE_DYNAMIC_STAKE = (os.getenv("WS_DYNAMIC_STAKE", "1").strip() == "1")  # usar % da banca

COOLDOWN_ATIVO = int(os.getenv("WS_COOLDOWN_ATIVO", "120"))  # 2 min de cooldown base (era 5min — muito restritivo)
COOLDOWN_LOSS_ATIVO = int(os.getenv("WS_COOLDOWN_LOSS", "120"))  # 2 min após LOSS no ativo (era 5min)
MAX_CONSECUTIVE_LOSS = int(os.getenv("WS_MAX_CONSEC_LOSS", "3"))  # Pausa após 3 losses consecutivos (era 1 — bloqueava tudo)
MAX_SESSION_LOSSES_NO_WIN = int(os.getenv("WS_MAX_SESSION_LOSS_NO_WIN", "6"))  # Parar bot se 6+ LOSSes sem nenhum WIN (era 3)
RETRAIN_ON_LOSS = (os.getenv("WS_RETRAIN_ON_LOSS", "0").strip() == "1")  # DESATIVADO: usar backtest em vez de retreino
BACKTEST_ON_LOSS = (os.getenv("WS_BACKTEST_ON_LOSS", "1").strip() == "1")  # ATIVADO: backtest 30min após LOSS para recalibrar
PAUSE_AFTER_LOSS_SECONDS = int(os.getenv("WS_PAUSE_AFTER_LOSS", "30"))  # Pausa de 30s após loss (era 60s)
RETRAIN_PENALTY = float(os.getenv("WS_RETRAIN_PENALTY", "0.15"))  # Penalidade ao retreinar padrão (era 0.25)

# ===================== IA (ONLINE) - APRENDIZADO ADAPTATIVO =====================
IA_ON = (os.getenv("WS_AI_ON", "1").strip() == "1")  # LIGADO: aprende bloqueando losses
# Arquivos por broker para não conflitar
_broker_suffix = {"iq_option": "m1", "bullex": "bullex", "casatrader": "casatrader"}.get(BROKER_TYPE, "m1")
AI_STATS_FILE = os.getenv("WS_AI_FILE", f"ws_ai_stats_{_broker_suffix}.json")
AI_MIN_SAMPLES = int(os.getenv("WS_AI_MIN_SAMPLES", "20"))   # 20 trades para começar a bloquear (era 15 — dava pouca margem)
AI_MIN_PROB = float(os.getenv("WS_AI_MIN_PROB", "0.52"))     # probabilidade mínima bayesiana (era 0.55 — muito restritivo)
AI_MIN_WINRATE = float(os.getenv("WS_AI_MIN_WINRATE", "0.38"))  # bloqueia se winrate < 38% (era 42%)
AI_CONF_MIN = float(os.getenv("WS_AI_CONF_MIN", "0.40"))     # confiança mínima na decisão (era 0.50)

# ===================== LIGHTGBM ENSEMBLE =====================
LGBM_ON = (os.getenv("WS_LGBM_ON", "1").strip() == "1") and LGBM_AVAILABLE  # LightGBM ativo
LGBM_MODEL_FILE = os.getenv("WS_LGBM_FILE", f"ws_lgbm_model_{_broker_suffix}.pkl")
LGBM_DATA_FILE = os.getenv("WS_LGBM_DATA", f"ws_lgbm_data_{_broker_suffix}.json")
LGBM_N_FEATURES = 22  # Features v4: 12 base + 6 candle math + 4 pipeline (ret1,ret3,ret5,range_vs_ma20)
LGBM_MIN_SAMPLES = int(os.getenv("WS_LGBM_MIN_SAMPLES", "30"))  # Mínimo de amostras para treinar
LGBM_RETRAIN_EVERY = int(os.getenv("WS_LGBM_RETRAIN", "10"))   # Retreina a cada N trades
LGBM_MIN_PROB = float(os.getenv("WS_LGBM_MIN_PROB", "0.54"))   # Probabilidade mínima do LGBM (era 0.58 — muito restritivo)
LGBM_WARMUP_PROB = float(os.getenv("WS_LGBM_WARMUP_PROB", "0.52"))  # Threshold durante warmup (era 0.55)
ENSEMBLE_MODE = os.getenv("WS_ENSEMBLE_MODE", "weighted")  # "weighted" = média ponderada (mais confiável)

# ===================== CNN PATTERN DETECTOR =====================
# CNN DESATIVADO - não tem dados suficientes e retorna sempre 0.50
# Para reativar: mudar "0" para "1" abaixo
CNN_ON = False  # Forçado OFF - sem dados/amostras para ser útil
CNN_MIN_PROB = float(os.getenv("WS_CNN_MIN_PROB", "0.55"))    # Prob mínima CNN para confirmar
CNN_WEIGHT = float(os.getenv("WS_CNN_WEIGHT", "0.20"))         # Peso da CNN no ensemble (20%)
CNN_VETO_THRESHOLD = float(os.getenv("WS_CNN_VETO", "0.30"))   # CNN < 0.30 = veto (bloqueia trade)

# ===================== FILTROS DE QUALIDADE ENSEMBLE (SIMPLIFICADO) =====================
# Thresholds reduzidos — IA filtra inteligentemente, não precisa bloquear tanto
ENS_MIN_CTX_RUIM = float(os.getenv("WS_ENS_MIN_CTX_RUIM", "0.55"))  # ctx < 0.40 precisa ensemble >= 0.55 (era 0.65)
ENS_MIN_CTX_MED  = float(os.getenv("WS_ENS_MIN_CTX_MED",  "0.52"))  # ctx 0.40-0.50 precisa ensemble >= 0.52 (era 0.60)
ENS_MIN_CTX_BOM  = float(os.getenv("WS_ENS_MIN_CTX_BOM",  "0.50"))  # ctx >= 0.50 precisa ensemble >= 0.50 (era 0.55)

# ===================== MODO DA IA =====================
# "learning" = IA tem controle total, filtros de score relaxados (mais trades, aprende mais rápido)
# "strict"   = IA + filtros rigorosos (menos trades, mais conservador)
IA_MODE = os.getenv("WS_IA_MODE", "learning").strip().lower()  # PADRÃO: learning

# ===================== FILTROS DE QUALIDADE EXTRA =====================
# MIN_CONFLUENCE: mínimo 2 confluências para operar (S/R + candle)
MIN_CONFLUENCE = 2  # Mínimo 2 confluências para entrar
BACKTEST_MIN_WINRATE = float(os.getenv("WS_BACKTEST_MIN_WINRATE", "0.40"))  # 40% mínimo (realista para OTC com poucas amostras)

# ===================== EXPERT GATE — SÓ OPERA QUANDO IA É EXPERT =====================
# A IA só faz operações REAIS quando atingir nível Expert.
# Até lá, roda backtest para aprender e simula sinais (sem arriscar dinheiro).
EXPERT_ONLY_TRADING = False                   # DESATIVADO — opera desde o início, IA filtra
EXPERT_MIN_TRADES = 50                        # Mínimo de trades (backtest+live) para Expert
EXPERT_MIN_WINRATE = 65.0                     # WR mínimo para Expert (%)

# ===================== EARLY SESSION GUARD =====================
# DESATIVADO — era muito agressivo, pausava 3min após 2 losses impedindo o bot de operar
EARLY_SESSION_GUARD_ON = False  # DESATIVADO (era True — bloqueava o bot cedo demais)
EARLY_SESSION_MAX_LOSSES = int(os.getenv("WS_EARLY_LOSSES", "4"))  # Após 4 losses sem win = pausa (era 2)

# ===================== LOSS PENALTY SYSTEM =====================
# DESATIVADO — criava efeito cascata: 1 loss → filtros sobem → bot para de operar
# A IA ensemble já filtra por qualidade, não precisa de penalidade extra
LOSS_PENALTY_ON = False  # DESATIVADO (era True — principal causa do bot ficar inativo)
LOSS_PENALTY_SCORE_PER_LEVEL = 0.01   # Valor residual se reativado
LOSS_PENALTY_CTX_PER_LEVEL = 0.01     # Valor residual se reativado
LOSS_PENALTY_MAX_LEVEL = 2            # Máximo 2 níveis se reativado
LOSS_PENALTY_DECAY_ON_WIN = 2         # 2 níveis removidos por WIN (mais rápido)

# ===================== AUTO-AJUSTE DE FILTROS =====================
# Contadores para auto-ajuste quando muitos skips consecutivos
MAX_CONSECUTIVE_SKIPS = int(os.getenv("WS_MAX_CONSEC_SKIPS", "8"))  # Tracking de skips
AUTO_RELAX_ON_SKIPS = False  # DESATIVADO: NUNCA relaxa trendline — LT é obrigatória

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
# SIMPLIFICADO: thresholds mais baixos para permitir mais entradas
# A IA ensemble (Bayes + LGBM) filtra os sinais ruins — não precisa bloquear no gate
if IA_MODE == "learning":
    GATE_MIN_SCORE = float(os.getenv("WS_GATE_MIN", "0.38"))   # Score mínimo (era 0.42 — muitos sinais bons bloqueados)
    GATE_SOFT_SCORE = float(os.getenv("WS_GATE_SOFT", "0.30")) # Soft skip (era 0.35)
    GATE_CONTEXT_BAD_BLOCK = True  # BLOQUEIA se contexto for ruim
    GATE_CONTEXT_VERY_BAD = 0.15  # Contexto mínimo (era 0.20 — OTC tem ctx baixo naturalmente)
else:  # strict
    GATE_MIN_SCORE = float(os.getenv("WS_GATE_MIN", "0.50"))   # Score mínimo rigoroso (era 0.60)
    GATE_SOFT_SCORE = float(os.getenv("WS_GATE_SOFT", "0.42")) # Soft skip rigoroso (era 0.50)
    GATE_CONTEXT_BAD_BLOCK = True  # Bloquear se contexto for ruim
    GATE_CONTEXT_VERY_BAD = 0.30  # Limiar de contexto (era 0.40)

# ===================== PREVISÃO DA PRÓXIMA VELA (CANDLE PREDICTOR) =====================
# Analisa as últimas N velas fechadas para prever a direção da próxima
# Roda ANTES da vela abrir → entrada IMEDIATA sem delay
CANDLE_PREDICT_ON = False  # DESATIVADO — bloqueava 80%+ dos sinais (predict_contra)
CANDLE_PREDICT_LOOKBACK = int(os.getenv("WS_CANDLE_PREDICT_LB", "10"))    # Últimas 10 velas
CANDLE_PREDICT_MIN_SCORE = float(os.getenv("WS_CANDLE_PREDICT_MIN", "0.02"))  # Score mínimo: DEVE ser a favor (era -0.15)
CANDLE_PREDICT_MIN_CONF = float(os.getenv("WS_CANDLE_PREDICT_MIN_CONF", "0.55"))  # Confiança mínima (rejeita 0.50-0.54 = ruído)

# ===================== ATR VOLATILITY GATE (NOVO) =====================
# Filter ALL entries based on ATR regime — blocks trades in dangerous markets
# ATR(14) / ATR(50) ratio: sweet spot is 0.5 - 1.8
ATR_GATE_ON = (os.getenv("WS_ATR_GATE", "1").strip() == "1")  # ATIVADO
ATR_GATE_MAX_RATIO = float(os.getenv("WS_ATR_GATE_MAX", "2.20"))  # ATR recente > 2.2x = volátil demais (era 1.80 — bloqueava mercados normais OTC)
ATR_GATE_MIN_RATIO = float(os.getenv("WS_ATR_GATE_MIN", "0.25"))  # ATR recente < 0.25x = morto (era 0.40 — bloqueava OTC calmos)
ATR_GATE_MAX_CANDLE = float(os.getenv("WS_ATR_GATE_MAX_CANDLE", "3.00"))  # Candle > 3x ATR = spike (era 2.50)
ATR_GATE_CHOP_FLIPS = int(os.getenv("WS_ATR_GATE_CHOP_FLIPS", "10"))  # 10+ flips = choppy (era 8 — OTC tem muitos flips naturalmente)

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

# Silenciar pings do websocket (DEBUG "Sending ping" a cada 25s é muito verboso)
logging.getLogger("websocket").setLevel(logging.WARNING)

# ===================== EXCEÇÃO DE META =====================
class MetaAtingidaException(Exception):
    """Exceção levantada quando a meta diária é atingida. Impede reinício automático."""
    pass

class C:
    G = "\033[92m"
    R = "\033[91m"
    Y = "\033[93m"
    B = "\033[94m"
    C = "\033[96m"  # Cyan
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
last_trade_dir: Dict[str, Tuple[str, float]] = {}  # Última direção operada por ativo: {ativo: ("CALL"/"PUT", timestamp)}
DIR_FLIP_COOLDOWN = 180  # 3 min se inverter direção (era 10min — bloqueava reversões legítimas)
global_consecutive_losses: int = 0  # Losses consecutivos globais
loss_penalty_level: int = 0  # Nível de penalidade (0=sem, 1..4 progressivo)

# ===================== FILTROS POR ATIVO =====================
# Cada ativo tem seus próprios filtros calibrados pelo backtest
filtros_por_ativo: Dict[str, Dict[str, Any]] = {}
# Estrutura: {
#   "EURUSD-OTC": {"min_ctx": 0.40, "min_score": 0.55, "taxa": 0.65, "sinais": 8, "habilitado": True},
#   ...
# }

# Ativos que já foram analisados no backtest (para detectar mudanças)
ativos_analisados_backtest: List[str] = []

# Ativos selecionados para operar (TOP N por WR do backtest)
_ativos_operando: List[str] = []

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


def predict_next_candle(iq: 'BrokerAPI', ativo: str, direcao: str, atr_val: float) -> Tuple[bool, str, float]:
    """
    Prevê a direção da PRÓXIMA vela analisando as últimas N velas FECHADAS.
    
    Roda ANTES da vela abrir → entrada IMEDIATA na abertura, sem delay.
    Usa 6 fatores ponderados:
      1. Momentum das últimas 3 velas (direção dos corpos)
      2. Posição do close da última vela no range
      3. Crescimento dos corpos (aceleração)
      4. Rejeição por pavios (defesa compradora/vendedora)
      5. Micro-tendência (média últimas 2 vs 3 anteriores)
      6. Sequência de velas contínuas na mesma direção
    
    Retorna: (confirma: bool, razao: str, confiança: float 0-1)
    """
    if not CANDLE_PREDICT_ON:
        return True, "predict_off", 0.50
    
    try:
        n = CANDLE_PREDICT_LOOKBACK
        candles = iq.get_candles(ativo, 60, n, end_ts_closed(60))
        if not candles or len(candles) < 5:
            return True, "predict_nodata", 0.50
        
        opens   = [float(c['open'])  for c in candles]
        highs   = [float(c['max'])   for c in candles]
        lows    = [float(c['min'])   for c in candles]
        closes  = [float(c['close']) for c in candles]
        
        score = 0.0
        factors = []
        
        # ── Fator 1: Momentum das últimas 3 velas ──
        bull_count = sum(1 for i in range(-3, 0) if closes[i] > opens[i])
        bear_count = 3 - bull_count
        if direcao == "CALL":
            momentum = (bull_count - bear_count) / 3.0
        else:
            momentum = (bear_count - bull_count) / 3.0
        score += momentum * 0.25
        factors.append(f"mom={momentum:+.2f}")
        
        # ── Fator 2: Posição do close da última vela no range ──
        last_range = highs[-1] - lows[-1]
        if last_range > 0:
            close_pos = (closes[-1] - lows[-1]) / last_range
            if direcao == "CALL":
                close_score = (close_pos - 0.5) * 0.30
            else:
                close_score = (0.5 - close_pos) * 0.30
            score += close_score
            factors.append(f"cl={'↑' if close_pos > 0.5 else '↓'}{close_pos:.2f}")
        
        # ── Fator 3: Crescimento dos corpos (aceleração) ──
        bodies = [abs(closes[i] - opens[i]) for i in range(-3, 0)]
        if bodies[0] > 0 and atr_val > 0:
            body_growth = bodies[-1] / max(bodies[0], atr_val * 0.01)
            last_bullish = closes[-1] > opens[-1]
            favor = (direcao == "CALL" and last_bullish) or (direcao == "PUT" and not last_bullish)
            if body_growth > 1.5 and favor:
                score += 0.15
                factors.append("accel+")
            elif body_growth > 1.5 and not favor:
                score -= 0.10
                factors.append("accel-")
        
        # ── Fator 4: Rejeição por pavios na última vela ──
        last_body = abs(closes[-1] - opens[-1])
        upper_wick = highs[-1] - max(closes[-1], opens[-1])
        lower_wick = min(closes[-1], opens[-1]) - lows[-1]
        
        if direcao == "CALL" and lower_wick > last_body * 0.5:
            score += 0.12
            factors.append("wick_buy")
        elif direcao == "PUT" and upper_wick > last_body * 0.5:
            score += 0.12
            factors.append("wick_sell")
        elif direcao == "CALL" and upper_wick > last_body * 0.8:
            score -= 0.08
            factors.append("wick_contra")
        elif direcao == "PUT" and lower_wick > last_body * 0.8:
            score -= 0.08
            factors.append("wick_contra")
        
        # ── Fator 5: Micro-tendência (EMA curta vs longa) ──
        if len(closes) >= 5:
            avg_old = sum(closes[-5:-2]) / 3.0
            avg_new = sum(closes[-2:]) / 2.0
            micro = (avg_new - avg_old) / atr_val if atr_val > 0 else 0.0
            if direcao == "CALL":
                score += max(-0.15, min(0.15, micro * 0.30))
            else:
                score += max(-0.15, min(0.15, -micro * 0.30))
            factors.append(f"μ={'↑' if micro > 0 else '↓'}{abs(micro):.2f}")
        
        # ── Fator 6: Sequência contínua (3+ velas mesma dir) ──
        seq = 0
        for i in range(len(closes) - 1, 0, -1):
            is_bull = closes[i] > opens[i]
            match = (direcao == "CALL" and is_bull) or (direcao == "PUT" and not is_bull)
            if match:
                seq += 1
            else:
                break
        if seq >= 3:
            score += 0.10
            factors.append(f"seq={seq}")
        elif seq == 0:
            # Última vela contra → potencial reversão
            # Checar se penúltima era a favor (pullback)
            if len(closes) >= 2:
                pen_bull = closes[-2] > opens[-2]
                pen_match = (direcao == "CALL" and pen_bull) or (direcao == "PUT" and not pen_bull)
                if pen_match:
                    score -= 0.05
                    factors.append("pullback")
        
        # ── Decisão final ──
        confidence = max(0.0, min(1.0, 0.5 + score))
        detail = ",".join(factors)
        
        if score < CANDLE_PREDICT_MIN_SCORE:
            return False, f"predict_contra({detail},sc={score:+.3f})", confidence
        
        # NOVO: Rejeitar confiança muito baixa (0.50-0.54 = ruído, não sinal)
        if confidence < CANDLE_PREDICT_MIN_CONF:
            return False, f"predict_low_conf({detail},sc={score:+.3f},conf={confidence:.2f}<{CANDLE_PREDICT_MIN_CONF})", confidence
        
        return True, f"predict_ok({detail},sc={score:+.3f})", confidence
        
    except Exception as e:
        return True, f"predict_error({e})", 0.50

# ===================== PATCH WEBSOCKET =====================
# ===================== ATR VOLATILITY GATE =====================
def atr_volatility_gate(df: 'pd.DataFrame', atr_val: float) -> Tuple[bool, str]:
    """
    Filtra entradas baseado no regime de volatilidade ATR.
    
    Bloqueia quando:
    1. ATR recente >> ATR histórico (mercado explosivo, zonas não seguram)
    2. ATR recente << ATR histórico (mercado morto, spread come lucro)
    3. Candle atual é spike (range > 2.5x ATR)
    4. Mercado está choppy (7+ flips em 10 velas)
    
    Retorna: (ok: bool, razão: str)
    """
    if not ATR_GATE_ON:
        return True, "atr_gate_off"
    
    if df is None or len(df) < 50:
        return True, "atr_gate_nodata"
    
    try:
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        closes = df["close"].astype(float).values
        
        ranges = highs - lows
        
        # ATR recente (14 velas) vs ATR histórico (50 velas)
        atr_recent = float(np.mean(ranges[-14:]))
        atr_hist = float(np.mean(ranges[-50:]))
        
        if atr_hist < 1e-9:
            return True, "atr_gate_zero_hist"
        
        ratio = atr_recent / atr_hist
        
        # 1. Volatilidade MUITO ALTA → zonas S/R serão rompidas
        if ratio > ATR_GATE_MAX_RATIO:
            return False, f"atr_gate_ALTA({ratio:.2f}>{ATR_GATE_MAX_RATIO})"
        
        # 2. Mercado MORTO → spread come o lucro, movimentos sem continuidade
        if ratio < ATR_GATE_MIN_RATIO:
            return False, f"atr_gate_MORTO({ratio:.2f}<{ATR_GATE_MIN_RATIO})"
        
        # 3. Candle atual é SPIKE → não entrar no caos
        last_range = ranges[-1]
        if atr_val > 0 and last_range > ATR_GATE_MAX_CANDLE * atr_val:
            return False, f"atr_gate_SPIKE({last_range/atr_val:.1f}x>{ATR_GATE_MAX_CANDLE}x)"
        
        # 4. Choppiness check: muitos flips = sem direção (mercado OTC lateral)
        if len(closes) >= 12:
            flips = 0
            for i in range(-10, -1):
                if (closes[i] > closes[i-1]) != (closes[i-1] > closes[i-2]):
                    flips += 1
            if flips >= ATR_GATE_CHOP_FLIPS:
                return False, f"atr_gate_CHOP({flips}flips>={ATR_GATE_CHOP_FLIPS})"
        
        return True, f"atr_gate_ok(ratio={ratio:.2f})"
    
    except Exception as e:
        return True, f"atr_gate_error({e})"


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
                return old(self, *args, **kwargs)
            except TypeError:
                try:
                    return old(self)
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
            check, reason = iq.connect()

            if check != True:
                raise ConnectionError(f"connect() retornou False: {reason}")

            for _ in range(15):
                if iq.check_connect():
                    break
                time.sleep(1.5)

            if not iq.check_connect():
                raise ConnectionError("check_connect() retornou False após connect()")

            # Atualizar ACTIVES com todos os ativos do servidor (inclui novos OTC)
            try:
                iq.update_ACTIVES_OPCODE()
                log.info(paint("✅ ACTIVES atualizados do servidor (assets dinâmicos)", C.G))
            except Exception as e:
                log.warning(paint(f"⚠️ Falha ao atualizar ACTIVES dinâmicos: {e}", C.Y))

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
        new_iq = conectar_iq()
        update_heartbeat_ref(new_iq)
        return new_iq
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
                    update_heartbeat_ref(iq)
                    log.info(paint("Reconectado com sucesso.", C.G))
                    return iq
                time.sleep(1.5)
        except Exception as e:
            log.warning(paint(f"Reconexão rápida {attempt}/3 falhou: {e}", C.Y))
            time.sleep(3 * attempt)  # 3s, 6s, 9s

    log.warning(paint("Reconexão rápida falhou. Criando nova conexão...", C.Y))
    new_iq = conectar_iq()
    update_heartbeat_ref(new_iq)
    return new_iq

# ===================== HEARTBEAT KEEPALIVE =====================
_heartbeat_thread: Optional[threading.Thread] = None
_heartbeat_stop = threading.Event()
_heartbeat_iq_ref: Optional[BrokerAPI] = None

def _heartbeat_worker():
    """Thread que envia heartbeat periódico para manter WebSocket vivo."""
    global _heartbeat_iq_ref
    while not _heartbeat_stop.is_set():
        try:
            iq = _heartbeat_iq_ref
            if iq and hasattr(iq, 'api') and iq.check_connect():
                hb_time = int(time.time() * 1000)
                iq.api.heartbeat(hb_time)
        except Exception:
            pass  # silencioso - não travar por heartbeat
        _heartbeat_stop.wait(60)  # a cada 60s (complementa ping_interval do websocket)

def start_heartbeat(iq: BrokerAPI):
    """Inicia thread de heartbeat keepalive."""
    global _heartbeat_thread, _heartbeat_iq_ref
    _heartbeat_iq_ref = iq
    if _heartbeat_thread and _heartbeat_thread.is_alive():
        return  # já rodando
    _heartbeat_stop.clear()
    _heartbeat_thread = threading.Thread(target=_heartbeat_worker, daemon=True, name="heartbeat-keepalive")
    _heartbeat_thread.start()
    log.info(paint("💓 Heartbeat keepalive iniciado (25s)", C.G))

def stop_heartbeat():
    """Para thread de heartbeat."""
    _heartbeat_stop.set()

def update_heartbeat_ref(iq: BrokerAPI):
    """Atualiza referência do broker no heartbeat (após reconexão)."""
    global _heartbeat_iq_ref
    _heartbeat_iq_ref = iq

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
        if ("10054" in msg) or ("forçado o cancelamento" in msg) or ("goodbye" in msg) or ("10053" in msg) or ("connection" in msg and "lost" in msg) or ("websocket" in msg and "closed" in msg):
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

        # precisa ser grande o bastante pro SR + filtros (DOM Forex e Precision)
        need_min = max(220, SR_LOOKBACK + 20, 140)  # 140 = PRECISION_MIN_CANDLES + margem
        if len(df) < need_min:
            return None
        return df
    except Exception:
        return None

# ===================== ATIVOS / PAYOUT =====================
_payout_cache: Dict[str, Tuple[int, float]] = {}  # {ativo: (payout, timestamp)}
PAYOUT_CACHE_INDIVIDUAL_SEC = 120  # cache individual por ativo: 2min

def _rank_score(filtro: Dict[str, Any]) -> float:
    """Score para ranking de ativos: WR ponderado pelo nº de sinais.
    Ativos com 0 sinais = rank 0. 1 sinal vale pouco, 5+ vale bastante."""
    wr = filtro.get("taxa", 0)
    n = filtro.get("sinais", 0)
    if n <= 0:
        return 0.0  # Sem dados = sem ranking
    # Peso logarítmico: 1 sinal=0.29, 2=0.46, 3=0.58, 5=0.75, 10=1.0
    import math
    confidence = min(1.0, math.log2(n + 1) / math.log2(11))
    return wr * confidence

def obter_top_ativos_otc(iq: BrokerAPI) -> List[str]:
    global _cache_ativos, _cache_ativos_ts
    now = time.time()
    if _cache_ativos and (now - _cache_ativos_ts) < PAYOUT_REFRESH_SEC:
        return _cache_ativos

    try:
        dados = safe_call(iq, iq.get_all_open_time)
        turbo = dados.get("turbo", {})
    except Exception:
        return _cache_ativos if _cache_ativos else []

    abertos = [a for a, info in turbo.items() if info.get("open", False)]
    abertos_otc = [a for a in abertos if "-OTC" in a.upper()]
    if not abertos_otc:
        abertos_otc = abertos

    # FILTRAR ativos que existem no OP_code.ACTIVES (evitar "not found on consts")
    # Após update_ACTIVES_OPCODE(), o dicionário inclui todos os ativos do servidor.
    try:
        actives_dict = _broker_constants.ACTIVES
        n_antes = len(abertos_otc)
        abertos_otc = [a for a in abertos_otc if a in actives_dict]
        n_removidos = n_antes - len(abertos_otc)
        if n_removidos > 0:
            log.info(paint(f"🔧 Filtrados {n_removidos} ativos sem OP_code (restam {len(abertos_otc)} válidos)", C.B))
    except Exception:
        pass  # Se falhar, continua sem filtro (compatibilidade)

    # === FILTRO DE SEGURANÇA: só aceitar ativos que existem no OP_code.ACTIVES ===
    # Sem isso, get_candles() falha ("not found on consts") e desperdiça vagas no TOP
    known_actives = getattr(_broker_constants, 'ACTIVES', {})
    invalid_assets = [a for a in abertos_otc if a not in known_actives]
    if invalid_assets:
        log.warning(paint(f"⚠️ {len(invalid_assets)} ativos ignorados (sem OP_code): {invalid_assets[:5]}{'...' if len(invalid_assets) > 5 else ''}", C.Y))
    abertos_otc = [a for a in abertos_otc if a in known_actives]

    # === MÉTODO RÁPIDO: get_all_profit() retorna payouts de TODOS os ativos em 1 chamada ===
    all_profits = None
    try:
        all_profits = safe_call(iq, iq.get_all_profit, timeout=15)
    except Exception:
        pass

    filtrados = []

    if all_profits:
        # Usar payouts do get_all_profit (turbo) — instantâneo, sem timeout por ativo
        for a in abertos_otc:
            try:
                profit_info = all_profits.get(a, {})
                turbo_profit = profit_info.get("turbo", 0)
                payout = int(turbo_profit * 100) if turbo_profit else 0
                if payout > 0:
                    _payout_cache[a] = (payout, now)
            except Exception:
                cached = _payout_cache.get(a)
                payout = cached[0] if cached else 0
            
            if payout >= PAYOUT_MINIMO:
                filtrados.append((a, payout))
        
        log.info(f"Payouts obtidos via get_all_profit (batch) - {len(filtrados)} ativos válidos")
    else:
        # FALLBACK: get_digital_payout por ativo (lento, mas funciona)
        log.warning(paint("⚠️ get_all_profit falhou - usando fallback por ativo", C.Y))
        timeout_count = 0
        max_timeouts = 3

        for a in abertos_otc:
            cached = _payout_cache.get(a)
            if cached and (now - cached[1]) < PAYOUT_CACHE_INDIVIDUAL_SEC:
                payout = cached[0]
            else:
                if timeout_count >= max_timeouts:
                    payout = cached[0] if cached else 0
                else:
                    try:
                        payout = safe_call(iq, iq.get_digital_payout, a, 5, timeout=6)
                        payout = int(payout) if payout is not None else 0
                        _payout_cache[a] = (payout, now)
                    except TimeoutError:
                        timeout_count += 1
                        payout = cached[0] if cached else 0
                    except Exception:
                        payout = cached[0] if cached else 0

            if payout >= PAYOUT_MINIMO:
                filtrados.append((a, payout))

        if timeout_count >= max_timeouts:
            log.warning(paint(f"⚠️ {timeout_count} timeouts ao verificar payouts - usando cache parcial", C.Y))

    filtrados.sort(key=lambda x: x[1], reverse=True)
    top = [a for a, _ in filtrados[:NUM_ATIVOS]]
    
    # Só atualiza cache global se conseguiu dados válidos
    if top:
        _cache_ativos = top
        _cache_ativos_ts = now
    elif _cache_ativos:
        log.warning(paint("⚠️ Nenhum ativo novo encontrado - mantendo cache anterior", C.Y))
        return _cache_ativos
    
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
    Extrai features v4 do setup para o LightGBM.
    Usa dados REAIS que o Precision S/R e DOM Forex geram.
    Features (22):
    - score: qualidade do sinal (0-1)
    - sr_touches_n: toques na zona S/R normalizado (0-1, /100)
    - sr_rejections_n: rejeições normalizadas (0-1, /60)
    - sr_false_breaks_n: false breaks normalizados (0-1, /15)
    - candle_str: força do candle de confirmação (0-1)
    - mkt_qual: qualidade do mercado (0-1)
    - confluence_cnt_n: contagem de confluências normalizada (0-1, /6)
    - effA: eficiência direcional (0-1)
    - approach_n: velocidade de aproximação normalizada (-1 a 1)
    - has_lt: trendline presente (0 ou 1)
    - m5_align: M5 alinhado com direção (1=favor, 0=neutro, -1=contra)
    - dir_enc: direção (1=CALL, -1=PUT)
    --- Features matemáticas do candle (pipeline) ---
    - candle_body_ratio: corpo/range (0=doji, 1=marubozu)
    - candle_close_pos: (close-low)/(high-low) (0=fundo, 1=topo)
    - candle_body_vs_avg: corpo vs média 20 (body_vs_ma20)
    - candle_range_vs_atr: range/ATR (expansão/compressão)
    - candle_absorption_bull: absorção compradora
    - candle_absorption_bear: absorção vendedora
    --- Pipeline momentum features (v4) ---
    - candle_ret1: retorno % 1 candle
    - candle_ret3: retorno % 3 candles
    - candle_ret5: retorno % 5 candles
    - candle_range_vs_ma20: range vs média 20 ranges
    """
    score = float(setup.get("score", 0.0))
    
    # S/R zone features (normalizadas 0-1)
    sr_touches = float(setup.get("sr_touches", 0))
    sr_touches_n = min(sr_touches / 100.0, 1.0)
    
    sr_rejections = float(setup.get("sr_rejections", 0))
    sr_rejections_n = min(sr_rejections / 60.0, 1.0)
    
    sr_false_breaks = float(setup.get("sr_false_breaks", 0))
    sr_false_breaks_n = min(sr_false_breaks / 15.0, 1.0)
    
    # Candle e mercado
    candle_str = float(setup.get("candle_strength", setup.get("entry_confidence", 0.5)))
    mkt_qual = float(setup.get("market_quality", 0.5))
    
    # Confluência
    conf_count = float(setup.get("confluence_count", setup.get("pb_len", 0)))
    conf_count_n = min(conf_count / 6.0, 1.0)
    
    # Eficiência direcional
    effA = float(setup.get("effA", 0.0))
    
    # Approach speed (normalizada)
    approach = float(setup.get("sr_proximity", setup.get("retr", 0.0)))
    approach_n = max(-1.0, min(1.0, approach))
    
    # Trendline
    has_lt = 1.0 if setup.get("has_lt", False) else 0.0
    
    # M5 alignment
    reasons = setup.get("reasons", [])
    reasons_str = ",".join(str(r) for r in reasons)
    dir_str = str(setup.get("dir", "NEUTRAL"))
    if "M5_bullish" in reasons_str and dir_str == "CALL":
        m5_align = 1.0
    elif "M5_bearish" in reasons_str and dir_str == "PUT":
        m5_align = 1.0
    elif "M5_contra" in reasons_str:
        m5_align = -1.0
    elif "M5_neutral" in reasons_str:
        m5_align = 0.0
    else:
        m5_align = 0.0
    
    # Direção
    dir_enc = 1.0 if dir_str == "CALL" else (-1.0 if dir_str == "PUT" else 0.0)
    
    # Features matemáticas do candle (v3)
    c_body_ratio = float(setup.get("candle_body_ratio", 0.0))
    c_close_pos = float(setup.get("candle_close_pos", 0.5))
    c_body_vs_avg = min(float(setup.get("candle_body_vs_avg", 1.0)), 3.0) / 3.0  # normaliza 0-1
    c_range_vs_atr = min(float(setup.get("candle_range_vs_atr", 0.5)), 3.0) / 3.0  # normaliza 0-1
    c_absorption_bull = float(setup.get("candle_absorption_bull", 0.0))
    c_absorption_bear = float(setup.get("candle_absorption_bear", 0.0))
    
    # Features do pipeline de referência (v4: momentum + range_vs_ma20)
    c_ret1 = max(-0.5, min(0.5, float(setup.get("candle_ret1", 0.0))))   # clamp ±50%
    c_ret3 = max(-0.5, min(0.5, float(setup.get("candle_ret3", 0.0))))
    c_ret5 = max(-0.5, min(0.5, float(setup.get("candle_ret5", 0.0))))
    c_range_vs_ma20 = min(float(setup.get("candle_range_vs_ma20", 1.0)), 3.0) / 3.0  # normaliza 0-1
    
    features = np.array([
        score, sr_touches_n, sr_rejections_n, sr_false_breaks_n,
        candle_str, mkt_qual, conf_count_n, effA,
        approach_n, has_lt, m5_align, dir_enc,
        c_body_ratio, c_close_pos, c_body_vs_avg, c_range_vs_atr,
        c_absorption_bull, c_absorption_bear,
        c_ret1, c_ret3, c_ret5, c_range_vs_ma20
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

# ===================== EXPERT GATE — VERIFICAÇÃO DE NÍVEL DA IA =====================

def _is_ia_expert() -> Tuple[bool, str, int, float]:
    """
    Verifica se a IA atingiu nível Expert (mesmos critérios do chat_screen_new).
    
    Critérios Expert:
      - total_trades >= EXPERT_MIN_TRADES (50)
      - win_rate >= EXPERT_MIN_WINRATE (65%)
    
    total_trades = LGBM amostras (backtest + live)
    
    Returns:
        (is_expert, phase_name, total_trades, win_rate)
    """
    global lgbm_data
    
    if not EXPERT_ONLY_TRADING:
        return True, "Expert (gate off)", 0, 0.0
    
    total_trades = len(lgbm_data) if lgbm_data else 0
    if total_trades == 0:
        return False, "Iniciante", 0, 0.0
    
    wins = sum(1 for s in lgbm_data if isinstance(s, dict) and s.get("label") == 1)
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
    
    if total_trades >= EXPERT_MIN_TRADES and win_rate >= EXPERT_MIN_WINRATE:
        return True, "Expert", total_trades, win_rate
    elif total_trades >= 25 and win_rate >= 58:
        return False, "Avançado", total_trades, win_rate
    elif total_trades >= 10 and win_rate >= 52:
        return False, "Intermediário", total_trades, win_rate
    else:
        return False, "Iniciante", total_trades, win_rate


def _simulate_candle_result(iq, ativo: str, direcao: str) -> float:
    """
    Espera a vela M1 fechar e verifica se a direção estava correta.
    Retorna +1.0 para WIN simulado, -1.0 para LOSS simulado, 0.0 para empate/erro.
    """
    try:
        # Esperar a vela fechar (M1 = 60s)
        time.sleep(62)  # 60s + 2s margem para vela consolidar
        
        df = get_candles_df(iq, ativo, TF_M1, 3, end_ts=end_ts_closed(TF_M1))
        if df is None or len(df) < 2:
            return 0.0
        
        # A vela que acabou de fechar
        last = df.iloc[-1]
        o = float(last["open"])
        c = float(last["close"])
        
        if direcao == "CALL":
            return 1.0 if c > o else (-1.0 if c < o else 0.0)
        else:  # PUT
            return 1.0 if c < o else (-1.0 if c > o else 0.0)
    except Exception as e:
        log.warning(f"[SIM] Erro ao simular resultado: {e}")
        return 0.0

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
            # Com histórico suficiente, confia no Bayesiano
            should_trade = (bayes_prob >= 0.50) and (bayes_conf >= AI_CONF_MIN)
            reason_suffix = f"hist,prob={bayes_prob:.2f},n={n_arm}"
        else:
            # WARMUP SEM LGBM — SIMPLIFICADO
            # Sem dados suficientes da IA, confia na zona S/R (score já filtrou)
            # O GATE 1 (quality) já garantiu score >= 0.42 e ctx >= 0.20
            sc = float(setup.get("score", 0.0))
            ctx_val = float(setup.get("market_quality", 0.40))
            sr_tq = int(setup.get("sr_touches", 0))
            
            # REGRA SIMPLES: se passou pelo gate de qualidade, PERMITE
            # A IA vai aprender com os resultados reais
            if ctx_val < 0.15:
                should_trade = False
                reason_suffix = f"ctx_min={ctx_val:.2f}"
            elif sc >= 0.42:
                should_trade = True
                reason_suffix = f"warmup_ok(sc={sc:.2f},ctx={ctx_val:.2f},sr={sr_tq}t)"
            elif bayes_prob >= 0.50 and sc >= 0.38:
                should_trade = True
                reason_suffix = f"prior_ok(prob={bayes_prob:.2f},sc={sc:.2f})"
            elif IA_MODE == "learning" and sc >= 0.38:
                should_trade = True
                reason_suffix = f"learning(sc={sc:.2f})"
            else:
                should_trade = False
                reason_suffix = f"fraco(sc={sc:.2f},prob={bayes_prob:.2f})"
        
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
        bayes_ok = bayes_prob >= AI_MIN_PROB
        lgbm_ok = lgbm_prob >= LGBM_MIN_PROB
        should_trade = bayes_ok and lgbm_ok
        reason = f"both(B={bayes_prob:.2f},L={lgbm_prob:.2f}{cnn_suffix})"
    elif ENSEMBLE_MODE == "any":
        bayes_ok = bayes_prob >= AI_MIN_PROB
        lgbm_ok = lgbm_prob >= LGBM_MIN_PROB
        should_trade = (bayes_ok or lgbm_ok) and ensemble_prob >= 0.50
        reason = f"any(B={bayes_prob:.2f},L={lgbm_prob:.2f},ens={ensemble_prob:.2f}{cnn_suffix})"
    else:  # weighted
        # Threshold SIMPLIFICADO — era 0.58 (bloqueava muitos sinais bons)
        min_ens_weighted = max(AI_MIN_PROB, 0.52)
        should_trade = ensemble_prob >= min_ens_weighted
        reason = f"weighted(ens={ensemble_prob:.2f},min={min_ens_weighted:.2f}{cnn_suffix})"
    
    # Warmup SIMPLIFICADO: regras claras e permissivas
    if n_arm < AI_MIN_SAMPLES:
        if lgbm_available:
            # REGRA #1: LGBM muito confiante que vai PERDER → bloqueia
            if lgbm_prob < 0.30:
                should_trade = False
                reason = f"warmup_danger(L={lgbm_prob:.2f}<0.30)"
            # REGRA #2: Ambos negativos (<0.45) → consenso de LOSS
            elif bayes_prob < 0.45 and lgbm_prob < 0.45:
                should_trade = False
                reason = f"warmup_neg(B={bayes_prob:.2f},L={lgbm_prob:.2f})"
            # REGRA #3: Ensemble >= 0.50 → PERMITE (mais permissivo no warmup)
            elif ensemble_prob >= 0.50:
                should_trade = True
                reason = f"warmup_ok(B={bayes_prob:.2f},L={lgbm_prob:.2f},ens={ensemble_prob:.2f})"
            # REGRA #4: LGBM ou Bayes positivo (>=0.55) → PERMITE
            elif lgbm_prob >= 0.55 or bayes_prob >= 0.55:
                should_trade = True
                reason = f"warmup_one_ok(B={bayes_prob:.2f},L={lgbm_prob:.2f})"
            else:
                should_trade = False
                reason = f"warmup_fraco(B={bayes_prob:.2f},L={lgbm_prob:.2f},ens={ensemble_prob:.2f})"
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
    Foco: ZONA S/R + candle features + momentum.
    """
    sc = float(setup.get("score", 0.0))
    ctx = float(setup.get("market_quality", 0.40))
    sr_prox = float(setup.get("sr_proximity", 0.0))
    sr_tq = int(setup.get("sr_touches", 0))
    sr_w = float(setup.get("sr_weight", 0.0))
    candle_str = float(setup.get("candle_strength", setup.get("entry_confidence", 0.0)))
    conf_count = int(setup.get("confluence_count", 0))

    # Base no score (range mais amplo para refletir melhor a qualidade)
    p = 0.48 + (sc - 0.40) * 0.40

    # 1. ZONA S/R - fator mais importante
    if sr_tq >= 5 and sr_w >= 8.0:
        p += 0.08  # S/R muito forte (5+ toques, peso alto)
    elif sr_tq >= 3 and sr_w >= 4.0:
        p += 0.05  # S/R forte
    elif sr_tq >= 2:
        p += 0.02  # S/R básico

    # 2. Contexto de mercado
    if ctx >= 0.70:
        p += 0.04  # mercado excelente
    elif ctx >= 0.55:
        p += 0.02  # mercado bom
    elif ctx < 0.35:
        p -= 0.04  # mercado ruim

    # 3. Confluência alta = muitas confirmações
    if conf_count >= 5:
        p += 0.04
    elif conf_count >= 4:
        p += 0.02

    # 4. Candle — usa absorção matemática em vez de só pattern name
    if candle_str >= 0.60:
        p += 0.03  # candle forte confirmando
    elif candle_str >= 0.30:
        p += 0.01  # candle moderado
    # Bônus absorção: se direção = CALL, absorption_bull alto → mais confiança
    dir_str = str(setup.get("dir", "NEUTRAL"))
    if dir_str == "CALL":
        abs_val = float(setup.get("candle_absorption_bull", 0.0))
    elif dir_str == "PUT":
        abs_val = float(setup.get("candle_absorption_bear", 0.0))
    else:
        abs_val = 0.0
    if abs_val >= 0.25:
        p += 0.02  # absorção forte na direção certa

    # 5. Momentum (pipeline: ret1/ret3) — momentum favorável à direção
    ret1 = float(setup.get("candle_ret1", 0.0))
    ret3 = float(setup.get("candle_ret3", 0.0))
    if dir_str == "CALL" and ret1 > 0 and ret3 > 0:
        p += 0.02  # momentum recente a favor (CALL + retornos positivos)
    elif dir_str == "PUT" and ret1 < 0 and ret3 < 0:
        p += 0.02  # momentum recente a favor (PUT + retornos negativos)
    elif dir_str == "CALL" and ret3 < -0.005:
        p -= 0.02  # momentum forte contra (CALL mas caindo)
    elif dir_str == "PUT" and ret3 > 0.005:
        p -= 0.02  # momentum forte contra (PUT mas subindo)

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
    
    # Aplicar peso do backtest — WINs têm peso 1.5x maior para IA aprender mais com acertos
    weight_win = AI_BACKTEST_WEIGHT * 1.5  # WIN pesa mais
    weight_loss = AI_BACKTEST_WEIGHT * 0.7  # LOSS pesa menos (evita pessimismo)
    
    if win:
        arm["a"] = float(arm.get("a", 1.0)) + weight_win
    else:
        arm["b"] = float(arm.get("b", 1.0)) + weight_loss
    
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
    Aprende com sinais que tenham score e contexto mínimos razoáveis.
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
        
        # ===== FILTRO DE QUALIDADE (RELAXADO para aprender mais) =====
        # Score >= 0.25 E contexto >= 0.30 — sinais mínimos viáveis
        # IA precisa aprender tanto wins quanto losses para calibrar
        if score < 0.25 or ctx < 0.30:
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
    
    # ===== FILTRO DE QUALIDADE (RELAXADO) =====
    # Aprende com sinais viáveis — IA precisa de WINs e LOSSes para calibrar
    score = setup.get("score", 0.0)
    ctx = setup.get("market_quality", 0.0)
    
    # Score >= 0.25 E contexto >= 0.30
    if score < 0.25 or ctx < 0.30:
        return  # Sinal muito fraco
    
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
    Evita duplicação verificando sinais já existentes.
    """
    global backtest_history
    
    if not BACKTEST_USE_ACCUMULATED:
        return
    
    timestamp = time.time()
    
    # Cria set de sinais existentes para deduplicação rápida
    existing_keys = set()
    for s in backtest_history:
        # Chave única: ativo + direção + resultado + features principais
        key = (
            s.get("ativo", ""),
            s.get("dir", ""),
            s.get("resultado", ""),
            str(s.get("features", ""))
        )
        existing_keys.add(key)
    
    # Adicionar apenas sinais novos (não duplicados) com timestamp
    added = 0
    for sinal in sinais:
        key = (
            sinal.get("ativo", ""),
            sinal.get("dir", ""),
            sinal.get("resultado", ""),
            str(sinal.get("features", ""))
        )
        if key not in existing_keys:
            sinal_with_ts = sinal.copy()
            sinal_with_ts["timestamp"] = timestamp
            backtest_history.append(sinal_with_ts)
            existing_keys.add(key)
            added += 1
    
    # Remover amostras muito antigas (>48h) e limitar tamanho
    cutoff_time = time.time() - (48 * 3600)  # 48 horas
    backtest_history = [s for s in backtest_history if s.get("timestamp", 0) > cutoff_time]
    
    # Limitar ao máximo de amostras (mantém as mais recentes)
    if len(backtest_history) > BACKTEST_HISTORY_MAX_SAMPLES:
        backtest_history = sorted(backtest_history, key=lambda x: x.get("timestamp", 0), reverse=True)
        backtest_history = backtest_history[:BACKTEST_HISTORY_MAX_SAMPLES]
    
    if added > 0:
        backtest_history_save()
    log.info(f"[BACKTEST-HIST] Histórico atualizado: {len(backtest_history)} amostras (+{added} novos)")

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
    skipped = 0
    for sinal, time_weight in backtest_history_get_weighted_signals():
        ativo = sinal.get("ativo", "")
        win = sinal.get("win", False)
        score = sinal.get("score", 0.0)
        ctx = sinal.get("ctx", 0.0)
        
        # Filtro de qualidade — só aprende de sinais com score/ctx razoável
        if score < 0.25 or ctx < 0.30:
            skipped += 1
            continue
        
        # Para LOSSes, exigir qualidade maior (evita aprender lixo)
        if not win and (score < 0.35 or ctx < 0.40):
            skipped += 1
            continue
        
        # Peso final = peso do backtest * peso temporal
        # WINs pesam 1.5x mais para evitar pessimismo
        weight_mult = 1.5 if win else 0.7
        final_weight = AI_BACKTEST_WEIGHT * time_weight * weight_mult
        
        # Reconstruir setup
        setup_backtest = {
            "dir": sinal.get("direcao", "CALL"),
            "score": score,
            "market_quality": ctx,
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


# ===================== DETECT SETUP (SR + CANDLE FEATURES) =====================
def detect_setup(df: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    """
    Estratégia S/R + Candle Features.
    Detecta zonas de suporte/resistência e analisa features matemáticas do candle.
    A IA (Bayesiano + LGBM) valida depois.
    """
    try:
        result = sr_precision_signal(df, atr_val)
        if result.get("precision_trade", False):
            log.info(paint(
                f"  ⭐ SR SIGNAL: {result['dir']} score={result['score']:.2f} "
                f"| zona={result.get('sr_touches', 0)}t "
                f"| candle={result.get('candle_pattern', '?')} "
                f"| risk={result.get('breakout_risk', '?')}",
                C.G
            ))
        return result
    except Exception as e:
        log.warning(f"[SR] Erro: {e}")
        return {"trade": False, "precision_trade": False, "dir": "NEUTRAL",
                "score": 0.0, "reasons": [f"erro_{e}"]}


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

def backtest_antes_de_operar(iq: BrokerAPI, ativos: List[str], n_candles: int = 90, skip_global_filters: bool = False) -> Dict[str, Any]:
    """
    Executa backtest nos últimos N minutos para calibrar filtros automaticamente.
    CALCULA FILTROS INDIVIDUAIS POR ATIVO!
    
    skip_global_filters: se True, NÃO altera GATE_CONTEXT_VERY_BAD/GATE_MIN_SCORE
                         (usado em backtests incrementais de novos ativos)
    
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
    log.info(paint(f"📊 Fase 1: Coletando sinais de {len(ativos)} ativos...", C.B))
    sinais_por_ativo: Dict[str, List[Dict]] = {}
    _bt_start = time.time()
    _BT_TIMEOUT = 300  # máximo 5 min para backtest (40 ativos)
    _bt_timed_out = False
    
    for ativo in ativos:  # processa TODOS os ativos do pool
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
            filtros_por_ativo[ativo] = {"min_ctx": min_ctx_floor, "min_score": min_score_floor, "taxa": 0.0, "sinais": 0, "habilitado": True}
        return {"sinais": 0, "wins": 0, "losses": 0, "taxa_acerto": 0, "calibrado": True, "ajustes": [], "filtros_por_ativo": filtros_por_ativo}
    
    wins_raw = sum(1 for s in todos_sinais if s["win"])
    taxa_raw = wins_raw / total_raw
    log.info(f"   Total de sinais: {total_raw} | WINs: {wins_raw} ({taxa_raw*100:.1f}%)")
    
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
    
    # ===== FASE 3: CALIBRAR PELO EQUILÍBRIO (MEDIANA) =====
    log.info(paint("🎯 Fase 3: Calibrando pelo equilíbrio (mediana)...", C.B))
    # Ordenar todos os sinais por score e contexto
    sorted_ctx = sorted([s["ctx"] for s in todos_sinais])
    sorted_score = sorted([s["score"] for s in todos_sinais])
    # Mediana
    med_ctx = sorted_ctx[len(sorted_ctx)//2] if sorted_ctx else 0.40
    med_score = sorted_score[len(sorted_score)//2] if sorted_score else 0.40
    # Buscar ponto de equilíbrio: maior número de sinais com WR >= 50% e não muito abaixo da mediana
    best_ctx = med_ctx
    best_score = med_score
    best_balance = 0
    best_n = 0
    best_wr = 0
    for delta in [0.00, -0.02, -0.04, -0.06, -0.08, -0.10]:
        ctx_test = max(0.20, med_ctx + delta)
        score_test = max(0.20, med_score + delta)
        filtered = [s for s in todos_sinais if s["ctx"] >= ctx_test and s["score"] >= score_test]
        n = len(filtered)
        if n < 3:
            continue
        w = sum(1 for s in filtered if s["win"])
        wr = w / n
        # Equilíbrio: (WR - 0.5) * n (maximiza quantidade com WR >= 50%)
        balance = (wr - 0.5) * n
        if wr >= 0.5 and balance > best_balance:
            best_balance = balance
            best_ctx = ctx_test
            best_score = score_test
            best_n = n
            best_wr = wr
    ctx_ideal = best_ctx
    score_ideal = best_score
    log.info(f"   📐 Filtros calibrados: ctx≥{ctx_ideal:.2f} score≥{score_ideal:.2f} (mediana)")
    log.info(f"      → {best_n} sinais | {best_wr*100:.1f}% WR (original: {taxa_raw*100:.1f}%)")
    
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
    # Limitar para não ficar absurdamente restritivo com poucas amostras
    if not skip_global_filters:
        GATE_CONTEXT_VERY_BAD = min(ctx_ideal, 0.50)   # Cap: não subir ctx acima de 0.50
        GATE_MIN_SCORE = min(score_ideal, 0.55)          # Cap: não subir score acima de 0.55
    else:
        log.info(paint("   ℹ️ Backtest incremental — filtros globais NÃO alterados", C.B))
    
    ajustes = []
    if ctx_ideal != ctx_original:
        ajustes.append(f"Contexto: {ctx_original:.2f} → {ctx_ideal:.2f}")
    if score_ideal != score_original:
        ajustes.append(f"Score: {score_original:.2f} → {score_ideal:.2f}")
    
    # ===== FASE 6: CALCULAR FILTROS INDIVIDUAIS POR ATIVO =====
    log.info(paint("🎯 Fase 6: Calculando filtros por ativo...", C.B))
    
    # CAPS para filtros por ativo — evita thresholds absurdos com poucas amostras
    PER_ASSET_MAX_CTX = 0.55   # Nunca exigir ctx acima disso por ativo (era 0.65)
    PER_ASSET_MAX_SCORE = 0.65  # Nunca exigir score acima disso por ativo (era 0.75)
    PER_ASSET_MIN_SIGNALS_DISABLE = 5  # Mínimo de sinais para poder desabilitar um ativo (era 3)

    for ativo, sinais in sinais_por_ativo.items():
        if len(sinais) < 2:
            # Poucos sinais - usar filtros globais mas habilitar
            # Taxa real: 0 sinais = 0.0, 1 sinal = resultado real (0 ou 1)
            if len(sinais) == 1:
                _taxa_poucos = 1.0 if sinais[0]["win"] else 0.0
            else:
                _taxa_poucos = 0.0  # 0 sinais = sem dados
            filtros_por_ativo[ativo] = {
                "min_ctx": min(ctx_ideal, PER_ASSET_MAX_CTX),
                "min_score": min(score_ideal, PER_ASSET_MAX_SCORE),
                "taxa": _taxa_poucos,
                "sinais": len(sinais),
                "habilitado": True,
                "motivo": "poucos_sinais"
            }
            continue

        wins_ativo = sum(1 for s in sinais if s["win"])
        taxa_ativo = wins_ativo / len(sinais)

        # ===== CALIBRAÇÃO POR ATIVO USANDO MEDIANA/EQUILÍBRIO =====
        ctxs = sorted(s["ctx"] for s in sinais)
        scores = sorted(s["score"] for s in sinais)
        # Mediana
        med_ctx = ctxs[len(ctxs)//2] if ctxs else ctx_ideal
        med_score = scores[len(scores)//2] if scores else score_ideal

        # CAP: com poucas amostras (<5), limitar filtros para não ficarem absurdos
        if len(sinais) < 5:
            med_ctx = min(med_ctx, PER_ASSET_MAX_CTX)
            med_score = min(med_score, PER_ASSET_MAX_SCORE)

        # Aplicar filtros medianos
        filtrados = [s for s in sinais if s["ctx"] >= med_ctx and s["score"] >= med_score]
        n_filtrados = len(filtrados)
        wins_filtrados = sum(1 for s in filtrados if s["win"])
        taxa_filtrada = wins_filtrados / n_filtrados if n_filtrados > 0 else 0.0

        # Ajustar habilitação — REQUER mínimo de sinais para desabilitar
        habilitado = True
        motivo = "ok"
        if len(sinais) >= PER_ASSET_MIN_SIGNALS_DISABLE and taxa_filtrada < 0.15:
            habilitado = False
            motivo = f"taxa_baixa_{taxa_filtrada*100:.0f}%"
        elif taxa_filtrada < 0.25:
            motivo = f"taxa_moderada_{taxa_filtrada*100:.0f}%"

        filtros_por_ativo[ativo] = {
            "min_ctx": med_ctx,
            "min_score": med_score,
            "taxa": taxa_filtrada,
            "sinais": n_filtrados,
            "wins": wins_filtrados,
            "habilitado": habilitado,
            "motivo": motivo
        }

        # Log
        status = "✅" if habilitado else "⛔"
        log.info(f"   {status} {ativo}: {n_filtrados} sinais | {taxa_filtrada*100:.0f}% | ctx≥{med_ctx:.2f} score≥{med_score:.2f}")
    
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
    # SÓ adiciona sinais de backtests com WR razoável (>=40%)
    # e SÓ os sinais FILTRADOS (que passaram ctx/score) — evita poluir com LOSSes
    sinais_para_historico = sinais_filtrados if sinais_filtrados else []
    backtest_wr_ok = taxa_raw >= 0.40  # WR mínimo para considerar backtest válido
    
    if BACKTEST_USE_ACCUMULATED and sinais_para_historico and backtest_wr_ok:
        try:
            backtest_history_add_signals(sinais_para_historico)
            hist_stats = backtest_history_analyze()
            log.info(paint(f"📚 HISTÓRICO ACUMULADO: {hist_stats['total']} amostras | WR={hist_stats['weighted_winrate']*100:.1f}% (ponderado)", C.B))
        except Exception as e:
            log.warning(f"Erro ao salvar histórico: {e}")
    elif BACKTEST_USE_ACCUMULATED:
        hist_stats = backtest_history_analyze()
        if taxa_raw < 0.40:
            log.info(paint(f"📚 HISTÓRICO: {hist_stats['total']} amostras | WR_backtest={taxa_raw*100:.0f}%<40% → NÃO aprendeu (evita poluir)", C.Y))
        else:
            log.info(paint(f"📚 HISTÓRICO ACUMULADO: {hist_stats['total']} amostras | WR={hist_stats['weighted_winrate']*100:.1f}% (ponderado)", C.B))
    
    # ===== IA APRENDE COM O BACKTEST (ATUAL + HISTÓRICO) =====
    # SÓ aprende se backtest tem WR razoável — evita aprender LOSSes ruins
    if AI_LEARN_FROM_BACKTEST and IA_ON:
        try:
            # Carregar stats da IA
            stats_backtest = _safe_load_json(AI_STATS_FILE)
            if stats_backtest is None:
                stats_backtest = {"meta": {"total": 0}, "arms": {}, "patterns": {}}
            
            n_learned_current = 0
            n_learned_history = 0
            
            # 1. Aprender com sinais do backtest ATUAL — SÓ se WR >= 40%
            if sinais_para_historico and backtest_wr_ok:
                n_learned_current = ai_learn_from_backtest_batch(sinais_para_historico, stats_backtest)
            elif not backtest_wr_ok and todos_sinais:
                log.info(paint(f"[BACKTEST-LEARN] WR={taxa_raw*100:.0f}%<40% → IA NÃO aprendeu (protege modelo)", C.Y))
            
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
        "total_sinais": total_raw  # total de sinais brutos (antes de filtrar)
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
    start_heartbeat(iq)  # Keepalive WebSocket

    global MIN_CONFLUENCE
    global lgbm_data, lgbm_model, lgbm_reliable, lgbm_val_accuracy
    global loss_penalty_level
    MIN_CONFLUENCE = 2  # Mínimo 2 confluências para entrar

    log.info("=" * 60)
    log.info(f"🚀 WS_AUTO_AI — S/R + Candle Features (M1) + ENSEMBLE IA [{_BROKER_NAME}]")
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
        log.info(paint(f"   → Confluência: {MIN_CONFLUENCE} (S/R+candle)", C.B))
    else:
        log.info(paint("🔒 MODO: STRICT - IA + filtros rigorosos", C.Y))
        log.info(paint("   → Score mínimo alto, filtros conservadores", C.B))

    stats = _safe_load_json(AI_STATS_FILE) if IA_ON else {"meta": {"total": 0}, "arms": {}}
    if IA_ON:
        log.info(f"[BAYES] ON | file={AI_STATS_FILE} | min_samples={AI_MIN_SAMPLES} | min_prob={AI_MIN_PROB:.2f}")
    
    # Carregar LightGBM
    if LGBM_ON:
        lgbm_load_data()
        # ===== MIGRAÇÃO: limpar dados antigos com features v1 (14 features) =====
        if lgbm_data and len(lgbm_data) > 0:
            sample_len = len(lgbm_data[0].get("features", []))
            if sample_len != LGBM_N_FEATURES:
                log.info(paint(f"[LGBM] \u26a0\ufe0f Dados antigos ({sample_len} features vs {LGBM_N_FEATURES} esperadas) \u2192 RESETANDO para features v2", C.Y))
                lgbm_data = []
                lgbm_model = None
                lgbm_reliable = False
                lgbm_val_accuracy = 0.0
                lgbm_save_data()
                log.info(paint(f"[LGBM] \u2705 Dados limpos - IA vai reaprender com features melhores", C.G))
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

    # Expert Gate status
    if EXPERT_ONLY_TRADING:
        _exp_ok, _exp_phase, _exp_t, _exp_wr = _is_ia_expert()
        if _exp_ok:
            log.info(paint(f"🏆 EXPERT GATE: IA é EXPERT ({_exp_t} trades, WR={_exp_wr:.0f}%) → OPERANDO REAL", C.G))
        else:
            log.info(paint(f"🎓 EXPERT GATE: IA é {_exp_phase} ({_exp_t} trades, WR={_exp_wr:.0f}%) → MODO SIMULAÇÃO (só backtest + simulação)", C.Y))
            log.info(paint(f"   → Precisa: {EXPERT_MIN_TRADES} trades com WR>={EXPERT_MIN_WINRATE:.0f}% para operar REAL", C.Y))
    else:
        log.info(paint("🔓 EXPERT GATE: DESATIVADO — operando real desde o início", C.B))

    try:
        saldo_inicial = float(iq.get_balance())
        log.info(paint(f"💰 SALDO INICIAL: {saldo_inicial:.2f} | META: {META_LUCRO_PERCENT:.1f}% (={saldo_inicial * META_LUCRO_PERCENT / 100:.2f})", C.G))
        if USE_DYNAMIC_STAKE:
            log.info(paint(f"📊 GESTÃO: {PERCENT_BANCA:.1f}% da banca por operação (stake dinâmico)", C.B))
        else:
            log.info(paint(f"📊 GESTÃO: Stake fixo de {STAKE_FIXA:.2f}", C.B))
        tick_status = "ATIVADO" if CANDLE_PREDICT_ON else "DESATIVADO"
        log.info(paint(f"🔮 CANDLE PREDICTOR: {tick_status} | lookback={CANDLE_PREDICT_LOOKBACK} velas | min_score={CANDLE_PREDICT_MIN_SCORE}", C.G if CANDLE_PREDICT_ON else C.Y))
    except Exception:
        saldo_inicial = 1000.0

    total = 0
    wins = 0
    session_total_losses_no_win = 0  # Contador de losses na sessão sem nenhum WIN (nunca reseta exceto com WIN)

    # ========== BACKTEST INTELIGENTE ANTES DE OPERAR ==========
    mercado_ok = True  # Flag para indicar se o mercado está bom
    ultima_verificacao_mercado = time.time()
    INTERVALO_REVERIFICACAO = 120  # Re-verificar mercado a cada 2 minutos se estiver ruim
    mercado_tentativas_falhas = 0  # Contador de re-verificações que falharam
    MAX_MERCADO_RETRIES = 3  # Máximo de tentativas antes de forçar retomada (reduzido para não travar)
    _early_guard_fired = False  # Early Session Guard só dispara 1x por sessão
    
    global ativos_analisados_backtest, _ativos_operando
    consecutive_skips = 0  # Contador de skips consecutivos para auto-relax
    
    try:
        ativos_backtest = obter_top_ativos_otc(iq)
        if ativos_backtest:
            log.info(paint(f"🔍 Pool inicial: {len(ativos_backtest)} ativos com payout≥{PAYOUT_MINIMO}%", C.B))
            backtest_result = backtest_antes_de_operar(iq, ativos_backtest, n_candles=90)
            taxa_backtest = backtest_result.get("taxa_acerto", 0.0)
            
            # IMPORTANTE: Salvar quais ativos foram analisados no backtest
            ativos_analisados_backtest = list(ativos_backtest)
            
            # ===== RANKING: SÓ seleciona ativos com WR >= 75% no backtest =====
            _fpa = backtest_result.get("filtros_por_ativo", {})
            if _fpa:
                _ranked = sorted(
                    _fpa.items(),
                    key=lambda x: _rank_score(x[1]),
                    reverse=True
                )
                # FILTRO PRINCIPAL: SÓ ativos com WR >= 75% e pelo menos 2 sinais
                _top = [
                    a for a, f in _ranked
                    if f.get("habilitado", True)
                    and f.get("taxa", 0) >= BACKTEST_MIN_WR_OPERAR
                    and f.get("sinais", 0) >= 2
                ][:NUM_ATIVOS_OPERAR]
                
                # Se nenhum ativo atingiu 75%, pegar os melhores com WR >= 60% (fallback)
                if not _top:
                    log.warning(paint(f"⚠️ Nenhum ativo com WR≥{BACKTEST_MIN_WR_OPERAR*100:.0f}%! Usando fallback WR≥60%...", C.Y))
                    _top = [
                        a for a, f in _ranked
                        if f.get("habilitado", True)
                        and f.get("taxa", 0) >= 0.60
                        and f.get("sinais", 0) >= 2
                    ][:NUM_ATIVOS_OPERAR]
                
                if _top:
                    _ativos_operando = _top
                    _cache_ativos = _top
                    _cache_ativos_ts = time.time()
                    log.info("=" * 60)
                    log.info(paint(f"🏆 {len(_top)} ATIVOS COM WR≥{BACKTEST_MIN_WR_OPERAR*100:.0f}% SELECIONADOS:", C.G))
                    for _i, (a, f) in enumerate([x for x in _ranked if x[0] in _top], 1):
                        _wr = f.get("taxa", 0) * 100
                        _ns = f.get("sinais", 0)
                        _rs = _rank_score(f)
                        log.info(paint(f"   {_i}. {a}: WR={_wr:.0f}% ({_ns} sinais) rank={_rs:.2f}", C.G))
                    log.info("=" * 60)
                else:
                    log.warning(paint("⚠️ Nenhum ativo passou nos filtros de WR! Usando pool por payout...", C.R))
            # ===============================================================
            
            # Mostrar filtros finais que serão usados
            log.info(paint(f"🎯 FILTROS ATIVOS: ctx≥{GATE_CONTEXT_VERY_BAD:.2f} score≥{GATE_MIN_SCORE:.2f}", C.G))
            
            # ===== IA COM APRENDIZADO → NÃO PAUSAR, APENAS CALIBRAR =====
            # Se a IA tem dados suficientes (Bayes ou LGBM), ela filtra cada trade
            # individualmente. O backtest apenas calibra filtros, nunca bloqueia tudo.
            ia_tem_dados = (
                (IA_ON and stats.get("meta", {}).get("total", 0) >= AI_MIN_SAMPLES) or
                (LGBM_ON and len(lgbm_data) >= LGBM_MIN_SAMPLES)
            )
            
            if ia_tem_dados:
                # IA tem aprendizado → operar SEMPRE, backtest só calibra filtros
                if taxa_backtest < 0.40:
                    log.warning(paint(f"⚠️ Backtest fraco ({taxa_backtest*100:.1f}%) mas IA tem aprendizado → operando com IA-VETO", C.Y))
                    log.info(paint("   → IA filtra cada sinal individualmente (Bayes + LGBM)", C.B))
                elif backtest_result["calibrado"]:
                    log.info(paint("✅ Filtros otimizados + IA com aprendizado - iniciando operações!", C.G))
                else:
                    log.info(paint("⚠️ Mercado difícil mas IA tem aprendizado - operando com cautela", C.Y))
                # mercado_ok permanece True → bot opera
            else:
                # IA SEM dados → backtest decide se pausa
                ia_min_wr = BACKTEST_MIN_WINRATE
                n_sinais_bt = backtest_result.get("total_sinais", 0)
                if n_sinais_bt < 10:
                    ia_min_wr = max(0.30, ia_min_wr - 0.10)
                    log.info(paint(f"📊 Poucas amostras no backtest ({n_sinais_bt}) → threshold reduzido para {ia_min_wr*100:.0f}%", C.Y))
                
                if taxa_backtest < ia_min_wr:
                    mercado_ok = False
                    log.warning(paint(f"⛔ MERCADO RUIM (IA sem dados): Taxa {taxa_backtest*100:.1f}% < {ia_min_wr*100:.0f}%", C.R))
                    log.info(paint("   → PAUSANDO até mercado melhorar (re-verifica a cada 2min)", C.Y))
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
                stop_heartbeat()
                if lucro_percent >= META_LUCRO_PERCENT:
                    log.info(paint(f"🎯 META ATINGIDA! Lucro: {lucro_abs:.2f} ({lucro_percent:.2f}%) | Parando operação.", C.G))
                    raise MetaAtingidaException(f"Meta atingida: {lucro_percent:.2f}%")
                else:
                    log.info(paint(f"🛑 STOP LOSS! Perda: {lucro_abs:.2f} ({lucro_percent:.2f}%) | Parando operação.", C.R))
                    raise MetaAtingidaException(f"Stop loss: {lucro_percent:.2f}%")
        except MetaAtingidaException:
            raise  # propagar para fora do loop
        except Exception as e:
            log.warning(f"Erro ao verificar meta: {e}")

        # ========== VERIFICAR SE O MERCADO ESTÁ BOM ==========
        if not mercado_ok:
            # Se a IA ganhou dados suficientes durante a sessão, liberar imediatamente
            ia_tem_dados_agora = (
                (IA_ON and stats.get("meta", {}).get("total", 0) >= AI_MIN_SAMPLES) or
                (LGBM_ON and len(lgbm_data) >= LGBM_MIN_SAMPLES)
            )
            if ia_tem_dados_agora:
                mercado_ok = True
                mercado_tentativas_falhas = 0
                log.info(paint("🧠 IA tem aprendizado suficiente → liberando operações (IA filtra individualmente)", C.G))
            # Re-verificar mercado periodicamente
            elif time.time() - ultima_verificacao_mercado >= INTERVALO_REVERIFICACAO:
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
                        
                        ia_min_wr_rev = BACKTEST_MIN_WINRATE
                        n_sinais_rev = backtest_reverif.get("total_sinais", 0)
                        if n_sinais_rev < 10:
                            ia_min_wr_rev = max(0.30, ia_min_wr_rev - 0.10)
                        
                        if taxa_reverif >= ia_min_wr_rev:
                            mercado_ok = True
                            mercado_tentativas_falhas = 0
                            log.info(paint(f"✅ MERCADO MELHOROU! Taxa: {taxa_reverif*100:.1f}% (min={ia_min_wr_rev*100:.0f}%) - Retomando operações!", C.G))
                        else:
                            mercado_tentativas_falhas += 1
                            if mercado_tentativas_falhas >= 2 and taxa_reverif >= 0.28:
                                mercado_ok = True
                                mercado_tentativas_falhas = 0
                                log.warning(paint(f"🔄 Mercado moderado ({taxa_reverif*100:.1f}%) → operando com IA-VETO ativo", C.Y))
                            elif mercado_tentativas_falhas >= MAX_MERCADO_RETRIES:
                                mercado_ok = True
                                mercado_tentativas_falhas = 0
                                log.warning(paint(
                                    f"⚠️ MAX RETRIES ({MAX_MERCADO_RETRIES}x) atingido — retomando com filtros rigorosos "
                                    f"(taxa={taxa_reverif*100:.1f}%)", C.Y
                                ))
                            else:
                                log.warning(paint(f"⚠️ Mercado ruim: {taxa_reverif*100:.1f}% (tentativa {mercado_tentativas_falhas}/{MAX_MERCADO_RETRIES}) - aguardando...", C.Y))
                except Exception as e:
                    log.warning(f"Erro ao re-verificar mercado: {e}")
            
            # Se mercado não está ok, esperar
            if not mercado_ok:
                time.sleep(60)  # Esperar 1 min antes de verificar novamente
                continue
        # =====================================================

        # Usar lista rankeada se disponível, senão busca padrão
        if _ativos_operando:
            ativos = list(_ativos_operando)
        else:
            ativos = obter_top_ativos_otc(iq)
        if not ativos:
            log.warning("Sem ativos com payout mínimo. Tentando em 10s...")
            time.sleep(10)
            continue

        # ===================== VERIFICAR NOVOS ATIVOS =====================
        # Cooldown: só verificar a cada 10 minutos (evita storm de backtests)
        _NOVO_ATIVO_COOLDOWN = 600  # 10 min
        if not hasattr(main, '_last_novo_check'):
            main._last_novo_check = time.time()  # type: ignore
        
        if time.time() - main._last_novo_check >= _NOVO_ATIVO_COOLDOWN:  # type: ignore
            main._last_novo_check = time.time()  # type: ignore
            _todos_disponiveis = obter_top_ativos_otc(iq)
            novos_ativos = [a for a in _todos_disponiveis if a not in ativos_analisados_backtest]
            if novos_ativos:
                log.info(paint(f"🔄 {len(novos_ativos)} NOVOS ATIVOS detectados: {novos_ativos[:3]}...", C.Y))
                log.info(paint("   → Fazendo backtest nos novos ativos (filtros globais preservados)...", C.B))
                try:
                    backtest_novos = backtest_antes_de_operar(iq, novos_ativos, n_candles=90, skip_global_filters=True)
                    ativos_analisados_backtest.extend(novos_ativos)
                    log.info(paint(f"✅ Backtest concluído para novos ativos | Taxa: {backtest_novos.get('taxa_acerto', 0)*100:.1f}%", C.G))
                    
                    # Re-ranquear TODOS os ativos conhecidos e atualizar top operando
                    _fpa_all = filtros_por_ativo
                    if _fpa_all:
                        _ranked_all = sorted(
                            _fpa_all.items(),
                            key=lambda x: _rank_score(x[1]),
                            reverse=True
                        )
                        # SÓ ativos com WR >= 75%
                        _new_top = [
                            a for a, f in _ranked_all
                            if f.get("habilitado", True)
                            and f.get("taxa", 0) >= BACKTEST_MIN_WR_OPERAR
                            and f.get("sinais", 0) >= 2
                        ][:NUM_ATIVOS_OPERAR]
                        # Fallback: WR >= 60% se nenhum atingiu 75%
                        if not _new_top:
                            _new_top = [
                                a for a, f in _ranked_all
                                if f.get("habilitado", True)
                                and f.get("taxa", 0) >= 0.60
                                and f.get("sinais", 0) >= 2
                            ][:NUM_ATIVOS_OPERAR]
                        if _new_top and _new_top != _ativos_operando:
                            _ativos_operando = _new_top
                            ativos = list(_ativos_operando)
                            log.info(paint(f"🏆 TOP {len(_new_top)} ATUALIZADO: {_new_top}", C.G))
                except Exception as e:
                    log.warning(f"Erro ao fazer backtest em novos ativos: {e}")
                    ativos_analisados_backtest.extend(novos_ativos)
        # ===================================================================

        wait_until_minus(TF_M1, DECIDIR_ANTES_FECHAR_SEC)
        t_antes_analise = time.time()

        best_trade, best_any = escolher_melhor_setup(iq, ativos)

        # GUARD: Verifica se não passou do tempo (entrada atrasada)
        # Se a análise demorou demais e o candle já fechou + novo abriu,
        # a entrada seria no início do candle seguinte (fora do timing).
        tempo_analise = time.time() - t_antes_analise
        seg_restantes = TF_M1 - (time.time() % TF_M1)
        if seg_restantes > DECIDIR_ANTES_FECHAR_SEC + 5:
            # Já estamos no início do PRÓXIMO candle — entrada seria atrasada
            log.warning(paint(
                f"⏰ Análise demorou {tempo_analise:.1f}s, candle já virou. Pulando entrada.", C.Y
            ))
            wait_for_next_open(TF_M1)
            continue

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

        # ===================== ATR VOLATILITY GATE (NOVO) =====================
        # Bloqueia QUALQUER entrada quando o mercado está em regime perigoso
        # Roda ANTES de tudo — se ATR não é ideal, não entra nem em precision
        atr_ok, atr_reason = atr_volatility_gate(df_candles, atr_val)
        if not atr_ok:
            log.info(paint(
                f"[ATR-GATE] {ativo} {setup.get('dir','?')} | {atr_reason} | "
                f"score={score:.2f} | Mercado em regime perigoso → SKIP",
                C.R
            ))
            wait_for_next_open(TF_M1)
            cooldown[ativo] = time.time()
            continue
        
        # ===================== EARLY SESSION GUARD (NOVO) =====================
        # Se começo da sessão (0 wins) e já tem N losses, pausa protegendo banca
        # Dispara apenas 1x por sessão para não travar o bot
        if EARLY_SESSION_GUARD_ON and not _early_guard_fired and wins == 0 and total >= EARLY_SESSION_MAX_LOSSES:
            _early_acc = (wins / max(1, total)) * 100
            if _early_acc < 40:
                _early_guard_fired = True  # NÃO disparar novamente nesta sessão
                log.warning(paint(
                    f"🛡️ EARLY SESSION GUARD: {total} trades, {wins} wins ({_early_acc:.0f}%) — "
                    f"Sem win na sessão! Pausando 3min e recalibrando...",
                    C.R
                ))
                # Forçar recalibração via backtest
                try:
                    ativos_recal = obter_top_ativos_otc(iq)
                    if ativos_recal:
                        backtest_early = backtest_antes_de_operar(iq, ativos_recal, n_candles=90)
                        taxa_early = backtest_early.get("taxa_acerto", 0.0)
                        ativos_analisados_backtest.clear()
                        ativos_analisados_backtest.extend(ativos_recal)
                        if taxa_early < 0.50:
                            # Pausa de 3min mas NÃO bloqueia mercado permanentemente
                            log.warning(paint(f"⛔ Mercado difícil ({taxa_early*100:.0f}%) — recalibrado, retomando com filtros rigorosos", C.R))
                        else:
                            log.info(paint(f"✅ Mercado OK ({taxa_early*100:.0f}%) — filtros recalibrados", C.G))
                except Exception as e:
                    log.warning(f"Erro no early session backtest: {e}")
                time.sleep(180)  # Pausa de 3 minutos
                continue

        # ══════════════════════════════════════════════════════════
        # PIPELINE DE ENTRADA v2 — LIMPO & TRANSPARENTE
        # 3 gates: QUALITY → IA DECISION → EXECUTE
        # ══════════════════════════════════════════════════════════

        final_dir = str(setup["dir"])
        sinal_invertido = False

        # ── GATE 0: DIRECTION FLIP PROTECTION ──
        # Evita entrar CALL depois de PUT (ou vice-versa) no mesmo ativo em < 10 min
        if ativo in last_trade_dir:
            _prev_dir, _prev_ts = last_trade_dir[ativo]
            _flip_elapsed = time.time() - _prev_ts
            if _prev_dir != final_dir and _flip_elapsed < DIR_FLIP_COOLDOWN:
                _flip_wait = int(DIR_FLIP_COOLDOWN - _flip_elapsed)
                log.info(paint(
                    f"[DIR-FLIP] {ativo} | último={_prev_dir} agora={final_dir} | "
                    f"inversão bloqueada (falta {_flip_wait}s de {DIR_FLIP_COOLDOWN}s)",
                    C.Y
                ))
                cooldown[ativo] = time.time()
                continue

        # ── GATE 1: QUALIDADE MÍNIMA (SIMPLIFICADO) ──
        # Score = qualidade da zona S/R + candle + trend
        # Context = condição do mercado
        # FILOSOFIA: thresholds FIXOS e BAIXOS. A IA ensemble filtra os ruins.
        ctx_val = float(setup.get("market_quality", 0.40))
        _pen_info = f" [PEN-L{loss_penalty_level}]" if loss_penalty_level > 0 else ""

        # Thresholds SIMPLES — sem penalidade progressiva que trava o bot
        _quality_min_score = 0.42  # Mínimo fixo (era 0.50 + penalty até 0.62)
        _quality_min_ctx = 0.20    # Mínimo fixo (era 0.35 + penalty até 0.50)

        if score < _quality_min_score:
            log.info(paint(
                f"[QUALITY] {ativo} {final_dir} | score={score:.2f}<{_quality_min_score:.2f}{_pen_info} | "
                f"{','.join(setup.get('reasons', []))}",
                C.Y
            ))
            consecutive_skips += 1
            cooldown[ativo] = time.time()
            continue

        if ctx_val < _quality_min_ctx:
            log.info(paint(
                f"[QUALITY] {ativo} {final_dir} | ctx={ctx_val:.2f}<{_quality_min_ctx:.2f}{_pen_info} | SKIP",
                C.Y
            ))
            consecutive_skips += 1
            cooldown[ativo] = time.time()
            continue

        # Ativo bloqueado pelo backtest (taxa muito baixa)
        # SIMPLIFICADO: score >= 0.55 bypassa bloqueio (era 0.72 — impossível)
        if ativo in filtros_por_ativo and not filtros_por_ativo[ativo].get("habilitado", True):
            if score < 0.55:
                log.info(paint(
                    f"[BLOCKED] {ativo} {final_dir} | backtest desabilitou "
                    f"({filtros_por_ativo[ativo].get('motivo','?')}) | SKIP",
                    C.R
                ))
                cooldown[ativo] = time.time()
                continue

        # Log do sinal que passou pelo gate de qualidade
        sr_touches_log = int(setup.get("sr_touches", 0))
        macro_t = str(setup.get("macro_trend_dir", "neutral"))
        macro_s = float(setup.get("macro_trend_strength", 0))
        candle_p = str(setup.get("candle_pattern", "none"))
        log.info(paint(
            f"[SINAL] {ativo} → {final_dir} | score={score:.2f} ctx={ctx_val:.2f} | "
            f"S/R={sr_touches_log}t candle={candle_p} macro={macro_t}({macro_s:.2f}){_pen_info} | "
            f"{','.join(setup.get('reasons', []))}",
            dir_color(final_dir)
        ))

        # ── GATE 2: IA DECISION ──
        # IA ensemble (Bayes + LGBM + CNN) é o filtro inteligente
        # Aprende quais setups funcionam e quais não
        if IA_ON:
            ens = ensemble_predict(ativo, setup, stats, df=df_candles)
            bayes_prob = float(ens["bayes_prob"])
            lgbm_prob = float(ens["lgbm_prob"])
            cnn_prob_val = float(ens.get("cnn_prob", 0.5))
            ensemble_prob = float(ens["ensemble_prob"])
            should_trade = bool(ens["should_trade"])
            ens_reason = str(ens["reason"])
            n_arm = int(ens.get("n_arm", 0))

            cnn_log = f" CNN={cnn_prob_val:.2f}" if CNN_ON and cnn_model is not None else ""

            log.info(paint(
                f"[IA] {ativo} {final_dir} | Bayes={bayes_prob:.2f}(n={n_arm}) "
                f"LGBM={lgbm_prob:.2f}{cnn_log} Ens={ensemble_prob:.2f} | {ens_reason}",
                C.G if should_trade else C.Y
            ))

            # LOSS PENALTY no ensemble: DESATIVADO (LOSS_PENALTY_ON=False)
            # Não bloqueia mais — a IA ensemble já pondera qualidade

            # IA diz não → respeitar
            if not should_trade:
                log.info(paint(
                    f"[IA-SKIP] {ativo} {final_dir} | {ens_reason} | SKIP",
                    C.Y
                ))
                cooldown[ativo] = time.time()
                continue

        # ── SESSION GUARD: muitos losses sem win → parar ──
        if total >= MAX_SESSION_LOSSES_NO_WIN and wins == 0:
            log.warning(paint(
                f"[SESSION] {ativo} {final_dir} | {total} trades sem WIN — sessão ruim, pausando 2min",
                C.R
            ))
            time.sleep(120)
            continue

        # ── LOG SETUP FINAL ──
        sr_reason_log = str(setup.get("sr_reason", "sr_bounce"))
        log.info(paint(
            f"✅ [SETUP-OK] {ativo} {final_dir} | {sr_reason_log} {sr_touches_log}t | "
            f"score={score:.2f} ctx={ctx_val:.2f} conf={setup.get('confluence_count', 1)}",
            C.G
        ))

        wait_for_next_open(TF_M1)

        # ── EXPERT GATE (opcional — desativado por padrão) ──
        if EXPERT_ONLY_TRADING:
            _expert_ok, _expert_phase, _expert_trades, _expert_wr = _is_ia_expert()
            if not _expert_ok:
                log.info(paint(
                    f"🎓 [SIM] IA={_expert_phase} ({_expert_trades}t, WR={_expert_wr:.0f}%) | "
                    f"Simulando {ativo} {final_dir}",
                    C.B
                ))
                sim_res = _simulate_candle_result(iq, ativo, final_dir)
                if sim_res > 0:
                    log.info(paint(f"🎓 [SIM-WIN] {ativo} {final_dir} ✅", C.G))
                elif sim_res < 0:
                    log.info(paint(f"🎓 [SIM-LOSS] {ativo} {final_dir} ❌", C.R))
                # Treinar IA com resultado simulado
                if sim_res != 0 and not sinal_invertido:
                    if IA_ON:
                        ai_update(ativo, setup, sim_res, stats)
                        _safe_save_json(AI_STATS_FILE, stats)
                    if LGBM_ON:
                        lgbm_add_sample(setup, sim_res)
                    if CNN_ON and cnn_model is not None and df_candles is not None:
                        try:
                            cnn_model.add_sample(df_candles, final_dir, win=(sim_res > 0))
                        except Exception:
                            pass
                cooldown[ativo] = time.time()
                last_trade_dir[ativo] = (final_dir, time.time())
                consecutive_skips = 0
                continue

        # IA é Expert → OPERAR DE VERDADE
        stake = calcular_stake_dinamico(iq, STAKE_FIXA)
        log.info(paint(f"[{ativo}] 💵 Stake calculado: {stake:.2f}", C.B))

        op = enviar_ordem(iq, ativo, final_dir, stake)

        if not op:
            log.error(paint(f"[{ativo}] ❌ falhou enviar ordem (TURBO/DIGITAL).", C.R))
            cooldown[ativo] = time.time()
            continue

        # Registrar direção operada (proteção contra inversão rápida)
        last_trade_dir[ativo] = (final_dir, time.time())

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
            # Enviar resultado para stdout (capturado pela tela)
            print(f">>> RESULTADO: WIN {res:.2f}", flush=True)
            # Reset counters após WIN
            consecutive_losses[ativo] = 0
            global_consecutive_losses = 0
            session_total_losses_no_win = 0  # Reset: teve WIN na sessão
            
            # LOSS PENALTY DECAY: reduzir penalidade após WIN
            if LOSS_PENALTY_ON and loss_penalty_level > 0:
                _old_pen = loss_penalty_level
                loss_penalty_level = max(0, loss_penalty_level - LOSS_PENALTY_DECAY_ON_WIN)
                log.info(paint(
                    f"📉 PENALTY DECAY: L{_old_pen} → L{loss_penalty_level} (WIN reduz exigência)",
                    C.G
                ))
            
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
            # Enviar resultado para stdout (capturado pela tela)
            print(f">>> RESULTADO: LOSS {abs(res):.2f}", flush=True)
            
            # 🧠 IA LOSS MEMORY: Analisar e salvar motivo do LOSS
            if LOSS_MEMORY_ON:
                try:
                    _extra = {
                        "broker": BROKER_TYPE,
                        "account_type": "DEMO" if os.getenv("WS_ACCOUNT_TYPE", "PRACTICE") != "REAL" else "REAL",
                    }
                    analyze_and_save_loss(ativo, final_dir, res, setup, stats, _extra)
                except Exception as _lm_err:
                    log.warning(f"[LOSS_MEMORY] Erro: {_lm_err}")
            
            # Incrementar contadores de LOSS
            consecutive_losses[ativo] = consecutive_losses.get(ativo, 0) + 1
            global_consecutive_losses += 1
            session_total_losses_no_win += 1
            # Aplicar cooldown especial após LOSS
            cooldown_loss[ativo] = time.time()
            
            # LOSS PENALTY: aumentar nível de penalidade (IA fica mais exigente)
            if LOSS_PENALTY_ON:
                _old_pen = loss_penalty_level
                loss_penalty_level = min(LOSS_PENALTY_MAX_LEVEL, loss_penalty_level + 1)
                _extra_score = loss_penalty_level * LOSS_PENALTY_SCORE_PER_LEVEL
                _extra_ctx = loss_penalty_level * LOSS_PENALTY_CTX_PER_LEVEL
                log.warning(paint(
                    f"📈 PENALTY UP: L{_old_pen} → L{loss_penalty_level} | "
                    f"score +{_extra_score:.2f} ctx +{_extra_ctx:.2f} "
                    f"(IA mais exigente para próximas entradas)",
                    C.R
                ))
            
            # KILL SWITCH SUAVIZADO: Se tomou N losses sem NENHUM win, PAUSA (não mata o bot)
            if session_total_losses_no_win >= MAX_SESSION_LOSSES_NO_WIN and wins == 0:
                log.warning(paint(f"🛑 SESSION GUARD: {session_total_losses_no_win} LOSSes consecutivos SEM WIN na sessão!", C.R))
                log.warning(paint(f"   → Pausando 3 minutos para mercado se estabilizar...", C.Y))
                time.sleep(180)  # Pausa 3 min em vez de matar o bot
                session_total_losses_no_win = 0  # Reset para tentar novamente
                log.info(paint("▶️ Retomando após pausa de 3 minutos...", C.G))
            
            # PENALIZAR FILTROS DO ATIVO ESPECÍFICO QUE DEU LOSS — SUAVIZADO
            if ativo in filtros_por_ativo:
                filtro = filtros_por_ativo[ativo]
                # Apertar LEVEMENTE (era +0.03 → agora +0.01)
                filtro["min_ctx"] = min(0.45, filtro.get("min_ctx", 0.40) + 0.01)
                filtro["min_score"] = min(0.55, filtro.get("min_score", 0.50) + 0.01)
                filtro["taxa"] = max(0.0, filtro.get("taxa", 0.50) - 0.05)
                
                # Só desabilita se taxa MUITO baixa (era 0.35 → agora 0.15)
                if filtro["taxa"] < 0.15:
                    filtro["habilitado"] = False
                    filtro["motivo"] = f"desabilitado_loss_consec_{consecutive_losses[ativo]}"
                    log.warning(paint(f"⛔ {ativo} DESABILITADO após LOSS repetido! Taxa={filtro['taxa']*100:.0f}%", C.R))
                else:
                    log.info(paint(f"🔧 {ativo} filtros ajustados: ctx≥{filtro['min_ctx']:.2f} score≥{filtro['min_score']:.2f}", C.Y))
            
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
                    
                    # Verificar se mercado ainda está bom (threshold dinâmico com IA)
                    # Se IA tem aprendizado, NUNCA pausar — apenas calibrar
                    ia_tem_dados_loss = (
                        (IA_ON and stats.get("meta", {}).get("total", 0) >= AI_MIN_SAMPLES) or
                        (LGBM_ON and len(lgbm_data) >= LGBM_MIN_SAMPLES)
                    )
                    if ia_tem_dados_loss:
                        # IA tem dados → operar, apenas logar status
                        if taxa_backtest_loss < 0.40:
                            log.warning(paint(f"⚠️ Backtest pós-LOSS fraco ({taxa_backtest_loss*100:.1f}%) mas IA filtra individualmente", C.Y))
                        else:
                            log.info(paint(f"✅ Mercado OK após recalibração: {taxa_backtest_loss*100:.1f}% + IA com aprendizado", C.G))
                        mercado_ok = True
                    else:
                        ia_min_wr_loss = BACKTEST_MIN_WINRATE
                        if taxa_backtest_loss < ia_min_wr_loss:
                            mercado_ok = False
                            log.warning(paint(f"⛔ MERCADO RUIM APÓS LOSS (IA sem dados)! Taxa: {taxa_backtest_loss*100:.1f}% < {ia_min_wr_loss*100:.1f}%", C.R))
                            log.warning(paint("   → Aguardando mercado melhorar antes de continuar...", C.Y))
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
        except MetaAtingidaException as e:
            stop_heartbeat()
            log.info(paint(f"✅ Bot encerrado: {e}", C.G))
            break  # META ou STOP LOSS - NÃO reiniciar
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
            stop_heartbeat()
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
