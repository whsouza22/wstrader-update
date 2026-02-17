# -*- coding: utf-8 -*-
"""
WS_AUTO_AI — Pernada B (M1) com:
✅ Candles FECHADOS (evita sinal fora da hora)
✅ Anti-lateral + Anti-esticado
✅ Filtro de SUPORTE/RESISTÊNCIA FORTE (usa >=200 velas e considera várias regiões)
✅ IA online simples (Bayes + UCB) aprendendo com seus próprios resultados (salva em JSON)
✅ Execução real (TURBO -> DIGITAL fallback)

Requisitos:
pip install casatraderapi pandas numpy
"""

# ========== SUPRIMIR MENSAGENS DO TENSORFLOW (ANTES DE IMPORTAR) ==========
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suprime mensagens TF
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desativa oneDNN warnings

import sys
import time
import math
import json
import logging
import ctypes
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pickle
import numpy as np
import pandas as pd
import talib
from casatraderapi.stable_api import Casa_Trader
from dom_forex_strategy import dom_forex_signal

# LightGBM para Gradient Boosting
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LGBM_AVAILABLE = False

# ===================== FLAGS GLOBAIS (ANTES DOS IMPORTS OPCIONAIS) =====================
PRE_USE_SKLEARN_ONLY = (os.getenv("WS_USE_SKLEARN_ONLY", "0").strip() == "1")
PRE_USE_CATBOOST_ONLY = (os.getenv("WS_USE_CATBOOST_ONLY", "0").strip() == "1")
PRE_USE_RF_PROXY_ONLY = (os.getenv("WS_USE_RF_PROXY_ONLY", "0").strip() == "1")
PRE_USE_STRUCT_HYBRID = (os.getenv("WS_USE_STRUCT_HYBRID", "0").strip() == "1")
_PRE_DISABLE_OTHERS = PRE_USE_SKLEARN_ONLY or PRE_USE_CATBOOST_ONLY or PRE_USE_RF_PROXY_ONLY or PRE_USE_STRUCT_HYBRID
ENABLE_REGIME_FILTER = (os.getenv("WS_ENABLE_REGIME_FILTER", "1").strip() == "1") and (not _PRE_DISABLE_OTHERS)
ENABLE_CNN = (os.getenv("WS_ENABLE_CNN", "1").strip() == "1") and (not _PRE_DISABLE_OTHERS)
ENABLE_RISK_CONTROL = (os.getenv("WS_ENABLE_RISK_CONTROL", "1").strip() == "1") and (not _PRE_DISABLE_OTHERS)
ENABLE_LOSS_ANALYZER = (os.getenv("WS_ENABLE_LOSS_ANALYZER", "1").strip() == "1") and (not _PRE_DISABLE_OTHERS)
ENABLE_AUTO_TUNER = (os.getenv("WS_ENABLE_AUTO_TUNER", "1").strip() == "1") and (not _PRE_DISABLE_OTHERS)
ENABLE_CONSENSO = (os.getenv("WS_ENABLE_CONSENSO", "1").strip() == "1") and (not _PRE_DISABLE_OTHERS)

# ===================== SKLEARN (OPCIONAL) =====================
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except Exception as e:
    SKLEARN_AVAILABLE = False
    print(f"[AVISO] Sklearn nao disponivel: {e}")

# ===================== CATBOOST (OPCIONAL) =====================
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception as e:
    CATBOOST_AVAILABLE = False
    print(f"[AVISO] CatBoost nao disponivel: {e}")

# IMPORTANTE: Define MODO_SR_SIMPLES ANTES de importar ia_autoconhecimento
MODO_SR_SIMPLES = (os.getenv("WS_MODO_SR_SIMPLES", "0").strip() == "1")  # DESLIGADO - Usando Pernada B + CNN
IA_AUTO_IN_SR = (os.getenv("WS_IA_AUTO_IN_SR", "0").strip() == "1")  # Default: desabilitado no modo S/R

# Sistema de Auto-Conhecimento da IA
# No modo S/R simplificado, usa 4 agentes em vez de IA-AUTO
if MODO_SR_SIMPLES and not IA_AUTO_IN_SR:
    # Não importa o ia_autoconhecimento no modo S/R
    IA_AUTOCONHECIMENTO_ON = False
    ia_autoconhecimento = None
    registrar_trade = None
    pode_entrar = None
    print("[INFO] IA-AUTO desabilitado no modo S/R (usa sistema de 4 agentes)")
else:
    try:
        from ia_autoconhecimento import ia_autoconhecimento, registrar_trade, pode_entrar
        IA_AUTOCONHECIMENTO_ON = True
    except ImportError:
        IA_AUTOCONHECIMENTO_ON = False
        ia_autoconhecimento = None
        registrar_trade = None
        pode_entrar = None

# ===================== NOVO SISTEMA: REGIME FILTER + CNN + RISK CONTROL =====================
if ENABLE_REGIME_FILTER:
    try:
        from regime_filter import RegimeFilter
        REGIME_FILTER_AVAILABLE = True
        print("[OK] Regime Filter ativado!")
    except ImportError as e:
        REGIME_FILTER_AVAILABLE = False
        print(f"[AVISO] Regime Filter nao disponivel: {e}")
else:
    REGIME_FILTER_AVAILABLE = False
    print("[INFO] Regime Filter desabilitado")

if ENABLE_CNN:
    try:
        from neural_model import TradingCNN
        CNN_AVAILABLE = True
        print("[OK] Modelo CNN ativado!")
    except ImportError as e:
        CNN_AVAILABLE = False
        print(f"[AVISO] Modelo CNN nao disponivel: {e}")
else:
    CNN_AVAILABLE = False
    print("[INFO] Modelo CNN desabilitado")

if ENABLE_RISK_CONTROL:
    try:
        from risk_control import RiskControl
        RISK_CONTROL_AVAILABLE = True
        print("[OK] Risk Control ativado!")
    except ImportError as e:
        RISK_CONTROL_AVAILABLE = False
        print(f"[AVISO] Risk Control nao disponivel: {e}")
else:
    RISK_CONTROL_AVAILABLE = False
    print("[INFO] Risk Control desabilitado")

# Loss Analyzer (opcional)
if ENABLE_LOSS_ANALYZER:
    try:
        from loss_analyzer import LossAnalyzer
        LOSS_ANALYZER_AVAILABLE = True
        print("[OK] Loss Analyzer ativado!")
    except ImportError as e:
        LOSS_ANALYZER_AVAILABLE = False
        print(f"[AVISO] Loss Analyzer nao disponivel: {e}")
else:
    LOSS_ANALYZER_AVAILABLE = False
    print("[INFO] Loss Analyzer desabilitado")

# OpenAI (opcional) para motivo de loss
try:
    from openai import OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
    openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=15.0) if OPENAI_API_KEY else None
    OPENAI_AVAILABLE = openai_client is not None
    if OPENAI_AVAILABLE:
        print("[OK] OpenAI disponível para análise de loss")
    else:
        print("[AVISO] OpenAI sem API key (OPENAI_API_KEY)")
except Exception as e:
    OPENAI_AVAILABLE = False
    openai_client = None
    print(f"[AVISO] OpenAI nao disponivel: {e}")

# AI Auto-Fixer - Sistema inteligente de análise e auto-correção
try:
    from ai_auto_fixer import get_ai_fixer, ai_should_enter, ai_analyze_loss, ai_record_win
    AI_FIXER_AVAILABLE = True
    ai_fixer = get_ai_fixer(auto_apply=False)
    print("[OK] AI Auto-Fixer ativado!")
except ImportError as e:
    AI_FIXER_AVAILABLE = False
    ai_fixer = None
    print(f"[INFO] AI Auto-Fixer não disponível: {e}")

# Importa AutoTuner
if ENABLE_AUTO_TUNER:
    try:
        from auto_tuner import get_tuner, AutoTuner
        AUTO_TUNER_AVAILABLE = True
        print("[OK] Auto-Tuner ativado!")
    except ImportError as e:
        AUTO_TUNER_AVAILABLE = False
        print(f"[AVISO] Auto-Tuner nao disponivel: {e}")
else:
    AUTO_TUNER_AVAILABLE = False
    print("[INFO] Auto-Tuner desabilitado")

# Importa Sistema de Consenso Multi-Agente
if ENABLE_CONSENSO:
    try:
        from multi_agent_consensus import get_consenso, SistemaConsenso
        CONSENSO_AVAILABLE = True
        print("[OK] Sistema de Consenso Multi-Agente (5 agentes) ativado!")
    except ImportError as e:
        CONSENSO_AVAILABLE = False
        print(f"[AVISO] Sistema de Consenso nao disponivel: {e}")
else:
    CONSENSO_AVAILABLE = False
    print("[INFO] Sistema de Consenso desabilitado")

# Importa Analisador de LOSS (IA Generativa)
try:
    from loss_analyzer_ai import get_loss_analyzer, LossAnalyzerAI
    LOSS_ANALYZER_AVAILABLE = True
    print("[OK] Analisador de LOSS (IA Generativa) ativado!")
except ImportError as e:
    LOSS_ANALYZER_AVAILABLE = False
    print(f"[AVISO] Analisador de LOSS nao disponivel: {e}")

# Importa pipeline de estrutura (M5 + M1)
try:
    from core.structure_m5 import build_zones
    from core.decision import decide_m1
    CORE_STRUCT_AVAILABLE = True
except ImportError as e:
    CORE_STRUCT_AVAILABLE = False
    print(f"[AVISO] Pipeline estrutura M5/M1 nao disponivel: {e}")

# Instancias globais do novo sistema
regime_filter = RegimeFilter() if REGIME_FILTER_AVAILABLE else None
trading_cnn = TradingCNN() if CNN_AVAILABLE else None
risk_control = RiskControl() if RISK_CONTROL_AVAILABLE else None
auto_tuner = get_tuner() if AUTO_TUNER_AVAILABLE else None
sistema_consenso = get_consenso(trading_cnn) if CONSENSO_AVAILABLE else None
loss_analyzer = get_loss_analyzer() if LOSS_ANALYZER_AVAILABLE else None

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
EMAIL = os.getenv("CASATRADER_EMAIL", "") or "wstrader@wstrader.onmicrosoft.com"
SENHA = os.getenv("CASATRADER_PASS", "") or "P152030@w"
CONTA = os.getenv("CASATRADER_CONTA", "REAL")

# Validação de credenciais
if not EMAIL or not SENHA:
    print("[WS_AUTO_AI] ERRO - Credenciais nao fornecidas via variaveis de ambiente", flush=True)
    print("[WS_AUTO_AI] ERRO - Configure CASATRADER_EMAIL e CASATRADER_PASS no aplicativo", flush=True)
    sys.exit(1)

TF_M1 = 60
TF_M5 = 300  # Timeframe de 5 minutos para analise de tendencia
DECIDIR_ANTES_FECHAR_SEC = int(os.getenv("WS_DECIDIR_ANTES_FECHAR", "10"))
N_M1 = int(os.getenv("WS_N_M1", "340"))
N_M5 = int(os.getenv("WS_N_M5", "240"))  # 240 candles de 5min = 20 horas de contexto

PAYOUT_MINIMO = int(os.getenv("WS_PAYOUT_MIN", "80"))
NUM_ATIVOS = int(os.getenv("WS_NUM_ATIVOS", "12"))
PAYOUT_REFRESH_SEC = int(os.getenv("WS_PAYOUT_REFRESH", "180"))

# ===================== EXPIRAÇÃO ANTI-MANIPULAÇÃO =====================
# Expiração variável entre 3-5 minutos, evitando minutos "redondos" da hora
EXP_MIN = int(os.getenv("WS_EXP_MIN", "3"))      # Expiração mínima (minutos)
EXP_MAX = int(os.getenv("WS_EXP_MAX", "5"))      # Expiração máxima (minutos)
EXP_FIXA = EXP_MAX  # Fallback para código legado
EVITAR_MINUTOS_REDONDOS = (os.getenv("WS_EVITAR_REDONDOS", "1").strip() == "1")

# ===================== EXPIRAÇÃO AUTO 1-5 =====================
EXP_MODE = os.getenv("WS_EXP_MODE", "AUTO_1_5").strip().upper()
EXP_FORCE = os.getenv("WS_EXP_FORCE", "").strip()
EXP_STATS_FILE = os.getenv("WS_EXP_STATS_FILE", "ws_exp_stats.json")
EXP_MIN_SAMPLES = int(os.getenv("WS_EXP_MIN_SAMPLES", "6"))

def _parse_exp_allowed(raw: str) -> List[int]:
    out: List[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            v = int(p)
        except Exception:
            continue
        if v > 0:
            out.append(v)
    if not out:
        return [1, 2, 3, 4, 5]
    return sorted(list(set(out)))

EXP_ALLOWED = _parse_exp_allowed(os.getenv("WS_EXP_ALLOWED", "1,2,3,4,5"))

# Minutos "perigosos" da hora (mais manipulação)
MINUTOS_MANIPULADOS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

VALOR_MINIMO = float(os.getenv("WS_VALOR_MINIMO", "3"))
STAKE_FIXA = float(os.getenv("WS_STAKE", "5"))

# ===================== GESTÃO DE BANCA =====================
PERCENT_BANCA = float(os.getenv("WS_PERCENT_BANCA", "1.0"))  # 1% da banca por operação
META_LUCRO_PERCENT = float(os.getenv("WS_META_LUCRO", "1.5"))  # para com 1.5% de lucro
STOP_LOSS_PERCENT = float(os.getenv("WS_STOP_LOSS", "3.0"))  # para com 3% de perda (opcional)
USE_DYNAMIC_STAKE = (os.getenv("WS_DYNAMIC_STAKE", "1").strip() == "1")  # usar % da banca

COOLDOWN_ATIVO = int(os.getenv("WS_COOLDOWN_ATIVO", "20"))  # seg

# ===================== DIAGNÓSTICO / DIREÇÃO =====================
DEBUG_REASONS = (os.getenv("WS_DEBUG_REASONS", "0").strip() == "1")
M5_ALIGN_BLOCK = (os.getenv("WS_M5_ALIGN_BLOCK", "0").strip() == "1")  # DESLIGADO - modo simples
M5_ALIGN_MIN_STRENGTH = float(os.getenv("WS_M5_ALIGN_MIN", "0.25"))
LOG_RANKED = (os.getenv("WS_LOG_RANKED", "0").strip() == "1")
RANKED_TOP = int(os.getenv("WS_LOG_RANKED_TOP", "3"))
FALLBACK_SIMPLE = (os.getenv("WS_FALLBACK_SIMPLE", "0").strip() == "1")
FALLBACK_MIN_SCORE = float(os.getenv("WS_FALLBACK_MIN_SCORE", "0.42"))
M5_NEUTRAL_BLOCK = (os.getenv("WS_M5_NEUTRAL_BLOCK", "0").strip() == "1")
M5_NEUTRAL_MIN_SCORE = float(os.getenv("WS_M5_NEUTRAL_MIN_SCORE", "0.45"))
CNN_STRICT = (os.getenv("WS_CNN_STRICT", "0").strip() == "1")
ENTRAR_IMEDIATO = (os.getenv("WS_ENTRAR_IMEDIATO", "1").strip() == "1")
SIMPLE_TREND_MODE = (os.getenv("WS_SIMPLE_TREND", "1").strip() == "1")
USE_M5_INDICATORS = (os.getenv("WS_USE_M5_INDICATORS", "1").strip() == "1")
USE_CANDLE_MID = (os.getenv("WS_USE_CANDLE_MID", "0").strip() == "1")
CM_MAX_DIST_ATR = float(os.getenv("WS_CM_MAX_DIST_ATR", "0.35"))
CM_ZIGZAG_MIN = float(os.getenv("WS_CM_ZIGZAG_MIN", "0.45"))
CM_TREND_MIN = float(os.getenv("WS_CM_TREND_MIN", "0.20"))
CM_CONFIRM_CANDLE = (os.getenv("WS_CM_CONFIRM_CANDLE", "1").strip() == "1")
M5_ENTRY_MAX_DELAY = float(os.getenv("WS_M5_ENTRY_MAX_DELAY", "60.0"))  # max segundos após virada para entrar
RISK_DISABLE_PAUSE = os.getenv("WS_RISK_DISABLE_PAUSE", "0").strip() == "1"
SR_ENTER_ON_OPEN = (os.getenv("WS_SR_ENTRAR_ABERTURA", "1").strip() == "1")
SR_ENTRY_MAX_DIST_ATR = float(os.getenv("WS_SR_ENTRY_MAX_DIST_ATR", "0.25"))
SR_STREAK_BLOCK_BARS = int(os.getenv("WS_SR_STREAK_BLOCK_BARS", "6"))
SR_STREAK_BLOCK_RATIO = float(os.getenv("WS_SR_STREAK_BLOCK_RATIO", "0.9"))

# ===================== RELATÓRIO DE LOSS =====================
LOSS_REPORT_ON = (os.getenv("WS_LOSS_REPORT", "1").strip() == "1")
LOSS_REPORT_FILE = os.getenv("WS_LOSS_REPORT_FILE", "loss_reports.json")
LOSS_REPORT_MAX = int(os.getenv("WS_LOSS_REPORT_MAX", "500"))

# ===================== MODO SIMPLIFICADO: S/R + CNN =====================
# MODO_SR_SIMPLES e IA_AUTO_IN_SR já definidos no início do arquivo (antes dos imports)

# ===================== IA (ONLINE) =====================
IA_ON = (os.getenv("WS_AI_ON", "1").strip() == "1")  # LIGADO: aprende bloqueando losses
AI_STATS_FILE = os.getenv("WS_AI_FILE", "ws_ai_stats_m1.json")

# ===================== MODO DA IA =====================
IA_MODE = os.getenv("WS_IA_MODE", "learning").strip().lower()

# ===================== LIGHTGBM ENSEMBLE =====================
LGBM_ON = (os.getenv("WS_LGBM_ON", "1").strip() == "1") and LGBM_AVAILABLE
LGBM_MODEL_FILE = os.getenv("WS_LGBM_FILE", "ws_lgbm_model_casatrader.pkl")
LGBM_DATA_FILE = os.getenv("WS_LGBM_DATA", "ws_lgbm_data_casatrader.json")
LGBM_N_FEATURES = 14
LGBM_MIN_SAMPLES = int(os.getenv("WS_LGBM_MIN_SAMPLES", "30"))
LGBM_RETRAIN_EVERY = int(os.getenv("WS_LGBM_RETRAIN", "10"))
LGBM_MIN_PROB = float(os.getenv("WS_LGBM_MIN_PROB", "0.58"))
LGBM_WARMUP_PROB = float(os.getenv("WS_LGBM_WARMUP_PROB", "0.55"))
ENSEMBLE_MODE = os.getenv("WS_ENSEMBLE_MODE", "weighted")

# ===================== FILTROS DE QUALIDADE ENSEMBLE =====================
ENS_MIN_CTX_RUIM = float(os.getenv("WS_ENS_MIN_CTX_RUIM", "0.65"))
ENS_MIN_CTX_MED  = float(os.getenv("WS_ENS_MIN_CTX_MED",  "0.60"))
ENS_MIN_CTX_BOM  = float(os.getenv("WS_ENS_MIN_CTX_BOM",  "0.55"))
AI_RESET_ON_START = os.getenv("WS_AI_RESET_ON_START", "0").strip() == "1"
AI_PRETRAIN_ON = os.getenv("WS_AI_PRETRAIN_ON", "0").strip() == "1"
AI_PRETRAIN_CANDLES = int(os.getenv("WS_AI_PRETRAIN_CANDLES", "900"))
AI_PRETRAIN_RESET = os.getenv("WS_AI_PRETRAIN_RESET", "0").strip() == "1"
AI_PRETRAIN_MAX_TRADES = int(os.getenv("WS_AI_PRETRAIN_MAX_TRADES", "500"))
AI_MIN_SAMPLES = int(os.getenv("WS_AI_MIN_SAMPLES", "15"))   # 15 trades para começar a bloquear
AI_MIN_PROB = float(os.getenv("WS_AI_MIN_PROB", "0.45"))     # probabilidade mínima (bayesiana)
AI_MIN_WINRATE = float(os.getenv("WS_AI_MIN_WINRATE", "0.42"))  # bloqueia se winrate < 42%
AI_CONF_MIN = float(os.getenv("WS_AI_CONF_MIN", "0.50"))     # confiança mínima na decisão

# ===================== SKLEARN ONLY =====================
USE_SKLEARN_ONLY = (os.getenv("WS_USE_SKLEARN_ONLY", "0").strip() == "1")
SKLEARN_MIN_PROB = float(os.getenv("WS_SKLEARN_MIN_PROB", "0.58"))
SKLEARN_TRAIN_CANDLES = int(os.getenv("WS_SKLEARN_TRAIN_CANDLES", "900"))
SKLEARN_TRAIN_MAX = int(os.getenv("WS_SKLEARN_TRAIN_MAX", "2000"))
SKLEARN_RETRAIN_SEC = int(os.getenv("WS_SKLEARN_RETRAIN_SEC", "3600"))

USE_CATBOOST_ONLY = (os.getenv("WS_USE_CATBOOST_ONLY", "0").strip() == "1")
CATBOOST_MIN_PROB = float(os.getenv("WS_CATBOOST_MIN_PROB", "0.58"))  # 58% threshold balanceado
CATBOOST_TRAIN_CANDLES = int(os.getenv("WS_CATBOOST_TRAIN_CANDLES", "900"))
CATBOOST_TRAIN_MAX = int(os.getenv("WS_CATBOOST_TRAIN_MAX", "2000"))
CATBOOST_RETRAIN_SEC = int(os.getenv("WS_CATBOOST_RETRAIN_SEC", "900"))  # 15 minutos
CATBOOST_DEPTH = int(os.getenv("WS_CATBOOST_DEPTH", "6"))
CATBOOST_LR = float(os.getenv("WS_CATBOOST_LR", "0.1"))
CATBOOST_ITERS = int(os.getenv("WS_CATBOOST_ITERS", "300"))
CATBOOST_MIN_SAMPLES = int(os.getenv("WS_CATBOOST_MIN_SAMPLES", "80"))
CATBOOST_BYPASS_ON_WEAK = (os.getenv("WS_CATBOOST_BYPASS", "1").strip() == "1")
CATBOOST_BYPASS_MIN_SCORE = float(os.getenv("WS_CATBOOST_BYPASS_MIN_SCORE", "0.70"))

USE_RF_PROXY_ONLY = (os.getenv("WS_USE_RF_PROXY_ONLY", "0").strip() == "1")
RF_PROXY_MIN_PROB = float(os.getenv("WS_RF_PROXY_MIN_PROB", "0.56"))
RF_PROXY_MIN_ROWS = int(os.getenv("WS_RF_PROXY_MIN_ROWS", "260"))
RF_PROXY_MIN_ACT_Z = float(os.getenv("WS_RF_PROXY_MIN_ACT_Z", "-0.20"))
RF_PROXY_MAX_WICK = float(os.getenv("WS_RF_PROXY_MAX_WICK", "2.8"))
RF_PROXY_MAX_CHOP = float(os.getenv("WS_RF_PROXY_MAX_CHOP", "0.80"))
RF_PROXY_RANGE_MA_WINDOW = int(os.getenv("WS_RF_PROXY_RANGE_MA_WINDOW", "20"))

USE_STRUCT_HYBRID = (os.getenv("WS_USE_STRUCT_HYBRID", "0").strip() == "1")
STRUCT_M5_MAX_ZONES = int(os.getenv("WS_STRUCT_M5_MAX_ZONES", "10"))
STRUCT_M5_TOL_MULT = float(os.getenv("WS_STRUCT_M5_TOL_MULT", "0.60"))
STRUCT_MIN_ZONE_TOUCHES = int(os.getenv("WS_STRUCT_MIN_ZONE_TOUCHES", "2"))
STRUCT_MIN_CONF = float(os.getenv("WS_STRUCT_MIN_CONF", "0.62"))
STRUCT_MIN_WICK_RATIO = float(os.getenv("WS_STRUCT_MIN_WICK_RATIO", "1.7"))
STRUCT_MAX_WICK_RATIO = float(os.getenv("WS_STRUCT_MAX_WICK_RATIO", "2.2"))
STRUCT_MIN_ACT_Z = float(os.getenv("WS_STRUCT_MIN_ACT_Z", "-0.10"))
STRUCT_MAX_CHOP = float(os.getenv("WS_STRUCT_MAX_CHOP", "0.70"))
STRUCT_RANGE_MA_WINDOW = int(os.getenv("WS_STRUCT_RANGE_MA_WINDOW", "20"))
STRUCT_CHOP_WINDOW = int(os.getenv("WS_STRUCT_CHOP_WINDOW", "10"))

if USE_SKLEARN_ONLY or USE_CATBOOST_ONLY or USE_RF_PROXY_ONLY or USE_STRUCT_HYBRID:
    IA_ON = False
    IA_AUTOCONHECIMENTO_ON = False

# ===================== REDE NEURAL DE PUNIÇÃO =====================
PUNISH_ON = (os.getenv("WS_PUNISH_ON", "1").strip() == "1")
PUNISH_MODEL_FILE = os.getenv("WS_PUNISH_FILE", "ws_punish_model.json")
PUNISH_LR = float(os.getenv("WS_PUNISH_LR", "0.05"))
PUNISH_SCALE = float(os.getenv("WS_PUNISH_SCALE", "0.5"))

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

# ===================== QUALIDADE DO GATILHO (AJUSTADO) =====================
MIN_BODY_FRAC_BREAK = float(os.getenv("WS_MIN_BODY_FRAC", "0.10"))  # apenas 10% de corpo
MAX_WICK_AGAINST = float(os.getenv("WS_MAX_WICK_AGAINST", "0.75"))  # permite mais pavio

# ===================== SCORE (BALANCEADO) =====================
GATE_MIN_SCORE = float(os.getenv("WS_GATE_MIN", "0.50"))  # Base relaxada: IA aprende
GATE_SOFT_SCORE = float(os.getenv("WS_GATE_SOFT", "0.45"))  # Permite aprendizado inicial

# FILTROS DE CONTEXTO (SELETIVOS - bloqueia apenas extremos ruins)
MIN_CONTEXT_QUALITY = float(os.getenv("WS_MIN_CTX_QUALITY", "0.28"))  # bloqueia apenas ctx MUITO ruim < 0.28
MIN_ENTRY_CONFIDENCE = float(os.getenv("WS_MIN_ENTRY_CONF", "0.42"))  # bloqueia apenas entry muito fraca < 0.42
MIN_CONFLUENCE_FOR_WEAK_CONTEXT = float(os.getenv("WS_MIN_CONFL", "0.06"))  # exige confluência se ctx ruim

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
SR_BLOCK_DIST_ATR = float(os.getenv("WS_SR_BLOCK_ATR", "0.65"))  # reduzido de 0.95 para 0.65 (menos bloqueio)
SR_TB_ON = (os.getenv("WS_SR_TOPBOTTOM", "1").strip() == "1")
SR_TB_LOOKBACK = int(os.getenv("WS_SR_TB_LOOKBACK", "30"))
SR_TB_MAX_DIST_ATR = float(os.getenv("WS_SR_TB_MAX_DIST_ATR", "0.60"))
SR_TREND_STRONG_BARS = int(os.getenv("WS_SR_TREND_STRONG_BARS", "7"))
SR_TREND_STRONG_PCT = float(os.getenv("WS_SR_TREND_STRONG_PCT", "0.35"))

# ===================== S/R BREAKOUT (ESTRUTURA FORTE) =====================
SR_BREAKOUT_ON = (os.getenv("WS_SR_BREAKOUT", "1").strip() == "1")
SR_BREAK_LOOKBACK = int(os.getenv("WS_SR_BREAK_LOOKBACK", "40"))
SR_BREAK_MIN_TOUCHES = int(os.getenv("WS_SR_BREAK_MIN_TOUCHES", "4"))
SR_BREAK_ATR = float(os.getenv("WS_SR_BREAK_ATR", "0.15"))
SR_BREAK_MAX_DIST_ATR = float(os.getenv("WS_SR_BREAK_MAX_DIST_ATR", "0.80"))

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
    M = "\033[95m"  # Magenta
    Z = "\033[0m"

def paint(s: str, color: str) -> str:
    return f"{color}{s}{C.Z}"

def dir_color(direction: str) -> str:
    return C.G if direction == "CALL" else (C.R if direction == "PUT" else C.Y)

def _debug_reject(debug: Optional[Dict[str, int]], key: str):
    if debug is None:
        return
    debug[key] = int(debug.get(key, 0)) + 1

def _summarize_rejects(debug: Optional[Dict[str, int]], top: int = 4) -> str:
    if not debug:
        return ""
    pairs = sorted(debug.items(), key=lambda x: x[1], reverse=True)[:max(1, top)]
    return " | rejeicoes=" + ", ".join([f"{k}:{v}" for k, v in pairs])

_cache_ativos: List[str] = []
_cache_ativos_ts: float = 0.0

# LightGBM globals
lgbm_model: Any = None
lgbm_data: List[Dict] = []
lgbm_trade_count: int = 0
lgbm_reliable: bool = False
lgbm_val_accuracy: float = 0.0

cooldown: Dict[str, float] = {}
cooldown_spike: Dict[str, float] = {}
# Rastreia losses consecutivos por (ativo, direção)
loss_streak_per_asset: Dict[str, int] = {}  # chave: "ATIVO_DIR", valor: n losses seguidos
MAX_CONSECUTIVE_LOSS_SAME_DIR = 2  # Bloqueia após 2 losses no mesmo ativo+direção

# ===================== FILTRO ADAPTATIVO - ENTRA SOMENTE EM ALTA PROBABILIDADE =====================
# Rastreia winrate por ativo e ajusta thresholds dinamicamente
ADAPTIVE_FILTER_ON = (os.getenv("WS_ADAPTIVE_FILTER", "1").strip() == "1")  # LIGADO por padrão
ADAPTIVE_MIN_TRADES = int(os.getenv("WS_ADAPTIVE_MIN_TRADES", "3"))  # Mínimo de trades para calcular winrate
ADAPTIVE_MIN_WINRATE = float(os.getenv("WS_ADAPTIVE_MIN_WINRATE", "0.40"))  # Bloqueia ativo se winrate < 40%
ADAPTIVE_GLOBAL_LOSS_STREAK_BLOCK = int(os.getenv("WS_GLOBAL_LOSS_STREAK", "3"))  # Pausa após 3 losses globais seguidos
ADAPTIVE_COOLDOWN_AFTER_LOSS = int(os.getenv("WS_COOLDOWN_AFTER_LOSS", "120"))  # Cooldown 2min após loss no ativo
ADAPTIVE_ML_PROB_BOOST = float(os.getenv("WS_ML_PROB_BOOST", "0.05"))  # Aumenta threshold ML após loss (5%)
ADAPTIVE_PATTERN_MIN_SCORE = float(os.getenv("WS_PATTERN_MIN_SCORE", "0.82"))  # Score mínimo de padrão para entrar
ADAPTIVE_ML_MIN_PROB = float(os.getenv("WS_ADAPTIVE_ML_MIN", "0.58"))  # Probabilidade ML mínima 58%

# Dicionários de rastreamento
adaptive_stats: Dict[str, Dict[str, int]] = {}  # {"ATIVO": {"wins": 0, "losses": 0}}
adaptive_global_loss_streak: int = 0  # Contador global de losses consecutivos
adaptive_pause_until: float = 0.0  # Timestamp até quando pausar entradas
adaptive_ml_boost: float = 0.0  # Boost atual no threshold ML (reseta após WIN)

# ===================== SISTEMA DE APRENDIZADO DE LOSS =====================
# Guarda contextos de LOSSes para evitar entradas similares
loss_memory: List[Dict] = []  # Lista de contextos de LOSS
LOSS_MEMORY_MAX = 50  # Máximo de registros na memória
LOSS_MEMORY_EXPIRY_HOURS = 4  # Expira após 4 horas
LOSS_SIMILARITY_ML_RANGE = 0.10  # Range de ML para considerar similar (±10%)

def load_loss_memory():
    """Carrega memória de LOSS do arquivo"""
    global loss_memory
    try:
        mem_path = Path.home() / ".wstrader" / "loss_memory.json"
        if mem_path.exists():
            with open(mem_path, "r") as f:
                data = json.load(f)
                # Filtra apenas registros válidos (não expirados)
                now = time.time()
                expiry_seconds = LOSS_MEMORY_EXPIRY_HOURS * 3600
                loss_memory = [
                    r for r in data 
                    if now - r.get("timestamp", 0) < expiry_seconds
                ]
                log.info(f"[LOSS-MEMORY] Carregados {len(loss_memory)} padrões de LOSS")
    except Exception as e:
        loss_memory = []

def save_loss_memory():
    """Salva memória de LOSS em arquivo"""
    try:
        mem_path = Path.home() / ".wstrader" / "loss_memory.json"
        mem_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mem_path, "w") as f:
            json.dump(loss_memory[-LOSS_MEMORY_MAX:], f, indent=2)
    except Exception:
        pass

def add_loss_to_memory(ativo: str, pattern: str, direction: str, ml_prob: float, pattern_score: float):
    """Adiciona um LOSS à memória para aprendizado"""
    global loss_memory
    
    # Extrai hora do dia (0-23)
    hora = datetime.now().hour
    
    loss_context = {
        "timestamp": time.time(),
        "ativo": ativo,
        "pattern": pattern,  # ex: "engolfo_baixa", "harami_alta"
        "direction": direction,  # CALL ou PUT
        "ml_prob": ml_prob,
        "pattern_score": pattern_score,
        "hora": hora
    }
    
    loss_memory.append(loss_context)
    
    # Limita tamanho
    if len(loss_memory) > LOSS_MEMORY_MAX:
        loss_memory = loss_memory[-LOSS_MEMORY_MAX:]
    
    save_loss_memory()
    log.info(paint(f"[LOSS-LEARN] 📚 Aprendendo: {ativo} {direction} | padrão={pattern} | ml={ml_prob:.2f} | hora={hora}h", C.B))

def is_similar_to_loss(ativo: str, pattern: str, direction: str, ml_prob: float, hora: int) -> Tuple[bool, str]:
    """Verifica se setup atual é similar a um LOSS passado"""
    if not loss_memory:
        return False, ""
    
    now = time.time()
    expiry_seconds = LOSS_MEMORY_EXPIRY_HOURS * 3600
    
    for loss in loss_memory:
        # Verifica se não expirou
        if now - loss.get("timestamp", 0) > expiry_seconds:
            continue
        
        # Verifica similaridade:
        # 1. Mesmo ativo
        # 2. Mesma direção
        # 3. Padrão contém substring similar (engolfo, harami, etc)
        # 4. ML_prob em range similar (±10%)
        same_ativo = loss["ativo"] == ativo
        same_dir = loss["direction"] == direction
        
        # Extrai tipo base do padrão (engolfo, harami, martelo, etc)
        loss_pattern_base = loss["pattern"].split("_")[0] if "_" in loss["pattern"] else loss["pattern"]
        current_pattern_base = pattern.split("_")[0] if "_" in pattern else pattern
        similar_pattern = loss_pattern_base == current_pattern_base
        
        # ML em range similar
        ml_diff = abs(loss["ml_prob"] - ml_prob)
        similar_ml = ml_diff <= LOSS_SIMILARITY_ML_RANGE
        
        # Hora similar (±1 hora)
        hora_diff = abs(loss["hora"] - hora)
        similar_hora = hora_diff <= 1 or hora_diff >= 23  # Considera virada de dia
        
        # Se 4 de 5 critérios são similares, bloqueia
        similarity_count = sum([same_ativo, same_dir, similar_pattern, similar_ml, similar_hora])
        
        if similarity_count >= 4:
            loss_time = datetime.fromtimestamp(loss["timestamp"]).strftime("%H:%M")
            return True, f"similar_loss({loss_time}|{loss['pattern']}|ml={loss['ml_prob']:.2f})"
    
    return False, ""

# Carrega memória de LOSS ao iniciar
load_loss_memory()

def get_adaptive_winrate(ativo: str) -> Tuple[float, int]:
    """Retorna (winrate, total_trades) do ativo"""
    if ativo not in adaptive_stats:
        return 1.0, 0  # Assume 100% se não tem histórico
    s = adaptive_stats[ativo]
    total = s.get("wins", 0) + s.get("losses", 0)
    if total == 0:
        return 1.0, 0
    return s.get("wins", 0) / total, total

def update_adaptive_stats(ativo: str, is_win: bool):
    """Atualiza estatísticas adaptativas após resultado"""
    global adaptive_global_loss_streak, adaptive_pause_until, adaptive_ml_boost
    
    if ativo not in adaptive_stats:
        adaptive_stats[ativo] = {"wins": 0, "losses": 0}
    
    if is_win:
        adaptive_stats[ativo]["wins"] += 1
        adaptive_global_loss_streak = 0  # Reseta streak global
        adaptive_ml_boost = 0.0  # Reseta boost do ML
    else:
        adaptive_stats[ativo]["losses"] += 1
        adaptive_global_loss_streak += 1
        adaptive_ml_boost = min(0.20, adaptive_ml_boost + ADAPTIVE_ML_PROB_BOOST)  # Aumenta threshold
        
        # Pausa após sequência de losses globais
        if adaptive_global_loss_streak >= ADAPTIVE_GLOBAL_LOSS_STREAK_BLOCK:
            adaptive_pause_until = time.time() + 180  # Pausa 3 minutos
            log.info(paint(f"[ADAPTIVE] ⚠️ {adaptive_global_loss_streak} losses seguidos - PAUSANDO 3 min", C.R))

def can_trade_adaptive(ativo: str, pattern_score: float, ml_prob: float, pattern: str = "", direction: str = "", log_obj=None) -> Tuple[bool, str]:
    """Verifica se pode entrar baseado no filtro adaptativo"""
    global adaptive_pause_until, adaptive_ml_boost
    
    if not ADAPTIVE_FILTER_ON:
        return True, "adaptive_desligado"
    
    _log = log_obj or log
    
    # 0. Verifica similaridade com LOSSes anteriores (Aprendizado)
    hora_atual = datetime.now().hour
    is_similar, similar_reason = is_similar_to_loss(ativo, pattern, direction, ml_prob, hora_atual)
    if is_similar:
        return False, f"🧠{similar_reason}"
    
    # 1. Verifica pausa global
    if time.time() < adaptive_pause_until:
        remaining = int(adaptive_pause_until - time.time())
        return False, f"pausa_global_{remaining}s"
    
    # 2. Verifica winrate do ativo
    winrate, total = get_adaptive_winrate(ativo)
    if total >= ADAPTIVE_MIN_TRADES and winrate < ADAPTIVE_MIN_WINRATE:
        return False, f"winrate_baixo_{winrate:.0%}(n={total})"
    
    # 3. Exige score de padrão mais alto
    if pattern_score < ADAPTIVE_PATTERN_MIN_SCORE:
        return False, f"padrão_fraco_{pattern_score:.2f}<{ADAPTIVE_PATTERN_MIN_SCORE}"
    
    # 4. Verifica ML com boost adaptativo (mínimo 62%, aumenta após cada loss)
    required_ml_prob = ADAPTIVE_ML_MIN_PROB + adaptive_ml_boost
    if ml_prob < required_ml_prob:
        return False, f"ml_prob_{ml_prob:.2f}<{required_ml_prob:.2f}(boost={adaptive_ml_boost:.2f})"
    
    return True, f"ok_wr={winrate:.0%}_ml={ml_prob:.2f}"

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
    """Espera a vela fechar - aguarda até 0.05s após virada para garantir dados"""
    s = seconds_to_next(tf)
    time.sleep(s + 0.05)  # Reduzido para entrada mais rápida

def end_ts_closed(tf: int) -> float:
    """Retorna timestamp da última vela FECHADA (garantido)"""
    now = time.time()
    # Calcula o início da vela atual, depois volta uma vela inteira
    inicio_vela_atual = now - (now % tf)
    # Retorna o fim da vela ANTERIOR (= início da atual - 1 segundo)
    return inicio_vela_atual - 1

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

# ===================== CasaTrader =====================
def conectar_casatrader() -> Casa_Trader:
    if not EMAIL or not SENHA:
        raise RuntimeError("Defina CASATRADER_EMAIL e CASATRADER_PASS nas variáveis de ambiente.")
    patch_websocket_on_close()
    log.info("Conectando à CasaTrader...")
    iq = Casa_Trader(EMAIL, SENHA)
    iq.connect()

    for _ in range(12):
        if iq.check_connect():
            break
        time.sleep(1.5)

    if not iq.check_connect():
        raise RuntimeError("Falha na conexão com a CasaTrader.")

    iq.change_balance(CONTA)
    try:
        log.info(f"Conectado | Saldo: {iq.get_balance():.2f} | Conta: {CONTA}")
    except Exception:
        log.info(f"Conectado | Conta: {CONTA}")

    return iq

def ensure_connected(iq: Optional[Casa_Trader]) -> Casa_Trader:
    if iq is None:
        return conectar_casatrader()
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

    return conectar_casatrader()

def safe_call(iq: Casa_Trader, fn, *args, **kwargs):
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
def get_candles_df(iq: Casa_Trader, ativo: str, timeframe: int, n: int, end_ts: Optional[float] = None) -> Optional[pd.DataFrame]:
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
        if USE_M5_INDICATORS and timeframe == TF_M5:
            need_min = max(40, N_M5)
        else:
            need_min = max(220, SR_LOOKBACK + 20)
        if len(df) < need_min:
            return None
        return df
    except Exception:
        return None

# ===================== COLETA M5 PARA TENDENCIA =====================
def get_candles_m5(iq: Casa_Trader, ativo: str, n: int = 48) -> Optional[pd.DataFrame]:
    """Coleta candles de 5 minutos para analise de tendencia macro."""
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
def new_pipeline_decide(iq: Casa_Trader, ativo: str, df_m1: pd.DataFrame, atr_val: float, payout: int = 80) -> Tuple[bool, str, Dict]:
    """
    Novo pipeline de decisao com:
    1. Coleta M5 para tendencia
    2. Regime Filter (pre-modelo)
    3. CNN com 3 classes
    4. Risk Control (pos-modelo)
    """
    details = {"pipeline": "CNN_V2", "stages": {}}

    # ===== STAGE 1: COLETA M5 =====
    df_m5 = None
    m5_direction = "NEUTRAL"
    m5_strength = 0.0

    if REGIME_FILTER_AVAILABLE and regime_filter:
        df_m5 = get_candles_m5(iq, ativo, N_M5)
        if df_m5 is not None:
            m5_direction, m5_strength = regime_filter.get_m5_direction(df_m5)
        details["stages"]["m5"] = {"direction": m5_direction, "strength": m5_strength}
        log.info(paint(f"[M5] {ativo} | Tendencia: {m5_direction} | Forca: {m5_strength:.2f}", C.B))

    # ===== STAGE 2: REGIME FILTER =====
    if REGIME_FILTER_AVAILABLE and regime_filter:
        blocked, reason, rf_details = regime_filter.should_block(df_m1, df_m5, atr_val, payout)
        details["stages"]["regime_filter"] = {"blocked": blocked, "reason": reason}

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

        details["stages"]["cnn"] = {"class": cnn_class, "probability": cnn_prob, "confidence": cnn_confidence}
        log.info(paint(f"[CNN] {ativo} | Classe: {cnn_class} | Prob: {cnn_prob:.2f} | Conf: {cnn_confidence:.2f}",
                      C.G if cnn_class != "NO_TRADE" else C.Y))

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
    if m5_direction != "NEUTRAL":
        if cnn_class == "CALL" and m5_direction != "BULLISH":
            log.info(paint(f"[CONFLITO] CNN={cnn_class} vs M5={m5_direction} | Bloqueando", C.Y))
            return False, "NO_TRADE", details
        if cnn_class == "PUT" and m5_direction != "BEARISH":
            log.info(paint(f"[CONFLITO] CNN={cnn_class} vs M5={m5_direction} | Bloqueando", C.Y))
            return False, "NO_TRADE", details

    # ===== STAGE 5: RISK CONTROL =====
    if RISK_CONTROL_AVAILABLE and risk_control:
        can_trade, rc_reason, rc_details = risk_control.should_trade({
            "class": cnn_class,
            "probability": cnn_prob,
            "raw_probs": details.get("stages", {}).get("cnn", {}).get("raw_probs", [0.33, 0.33, 0.34]),
            "confidence": cnn_confidence
        })
        details["stages"]["risk_control"] = {"can_trade": can_trade, "reason": rc_reason}

        if not can_trade:
            log.info(paint(f"[RISK] {ativo} | BLOQUEADO: {rc_reason}", C.Y))
            return False, "NO_TRADE", details

    # ===== APROVADO =====
    log.info(paint(f"[PIPELINE OK] {ativo} {cnn_class} | M5={m5_direction} | CNN={cnn_prob:.2f}", C.G))
    return True, cnn_class, details


def on_trade_result_cnn(ativo: str, direction: str, win: bool, df_m1: pd.DataFrame, profit: float = 0):
    """
    Atualiza todos os sistemas de IA apos resultado do trade:
    - CNN: adiciona amostra de treinamento
    - RiskControl: ajusta threshold e cooldown
    - AutoTuner: ajusta TODOS os parametros dinamicamente
    """
    # 1. Atualiza CNN
    if CNN_AVAILABLE and trading_cnn:
        trading_cnn.add_training_sample(df_m1, direction, win)

    # 2. Atualiza Risk Control
    if RISK_CONTROL_AVAILABLE and risk_control:
        risk_control.on_result(win)

    # 3. NOVO: Atualiza AutoTuner (auto-ajuste de parametros)
    if AUTO_TUNER_AVAILABLE and auto_tuner:
        from datetime import datetime
        hour = datetime.now().hour
        new_params = auto_tuner.on_trade_result(win=win, profit=profit, hour=hour)

        # Sincroniza threshold do RiskControl com AutoTuner
        if RISK_CONTROL_AVAILABLE and risk_control:
            new_threshold = new_params.get("base_threshold", 0.55)
            hour_adj = auto_tuner.get_hour_adjustment(hour)
            risk_control.current_threshold = min(0.80, max(0.40, new_threshold + hour_adj))

        # Log do auto-ajuste
        log.info(paint(
            f"[AUTO-TUNE] tol={new_params.get('tolerancia_atr', 0):.3f} | "
            f"toques={new_params.get('min_toques', 2)} | "
            f"thresh={new_params.get('base_threshold', 0.55):.2f} | "
            f"WR={auto_tuner.total_wins}/{auto_tuner.total_trades}",
            C.M
        ))

    result_str = "WIN" if win else "LOSS"
    color = C.G if win else C.R
    log.info(paint(f"[CNN LEARN] {ativo} {direction} | {result_str}", color))


# ===================== ATIVOS / PAYOUT =====================
def obter_top_ativos_otc(iq: Casa_Trader) -> List[str]:
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

def simple_trend_dir(df: pd.DataFrame, lookback: int = 20) -> Tuple[str, float]:
    """Direção e força de tendência simples (M1)."""
    closes = df["close"].tail(lookback)
    if len(closes) < 5:
        return "NEUTRAL", 0.0
    slope = float(closes.iloc[-1] - closes.iloc[0])
    atr_val = atr(df, 14)
    strength = abs(slope) / max(atr_val, 1e-9)
    if strength < CM_TREND_MIN:
        return "NEUTRAL", strength
    return ("CALL" if slope > 0 else "PUT"), strength

def strategy_candle_mid(df_m1: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    """Estratégia Candle Mid (M1): S/R forte + zigzag + tendência confirma."""
    if len(df_m1) < max(220, SR_LOOKBACK):
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["poucas_velas"]}

    price = float(df_m1["close"].iloc[-1])
    res, sup = strong_sr_levels_last200(df_m1, atr_val)

    # Nível mais próximo
    nearest_res = min(res, key=lambda t: abs(t[0] - price)) if res else None
    nearest_sup = min(sup, key=lambda t: abs(t[0] - price)) if sup else None

    near_res = False
    near_sup = False
    dist_res_atr = 999.0
    dist_sup_atr = 999.0

    if nearest_res:
        dist_res_atr = abs(nearest_res[0] - price) / max(atr_val, 1e-9)
        near_res = dist_res_atr <= CM_MAX_DIST_ATR
    if nearest_sup:
        dist_sup_atr = abs(price - nearest_sup[0]) / max(atr_val, 1e-9)
        near_sup = dist_sup_atr <= CM_MAX_DIST_ATR

    flips_frac, _eff = chop_stats(df_m1, 14)
    zigzag_ok = flips_frac >= CM_ZIGZAG_MIN

    trend_dir, trend_strength = simple_trend_dir(df_m1, 20)

    last = df_m1.iloc[-1]
    last_dir = "CALL" if last["close"] > last["open"] else ("PUT" if last["close"] < last["open"] else "NEUTRAL")

    reasons = [f"zigzag={flips_frac:.2f}", f"trend={trend_dir}({trend_strength:.2f})"]

    if not zigzag_ok:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["sem_zigzag"]}

    # Entrada em resistência -> PUT (com tendência confirmando ou neutra)
    if near_res and trend_dir in ("PUT", "NEUTRAL"):
        if CM_CONFIRM_CANDLE and last_dir != "PUT":
            return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["vela_confirma_put"]}
        score = 0.55 + (CM_MAX_DIST_ATR - dist_res_atr) * 0.20 + min(0.20, trend_strength * 0.10)
        return {
            "trade": True,
            "dir": "PUT",
            "score": float(min(0.95, max(0.0, score))),
            "sr_type": "RES",
            "sr_level": float(nearest_res[0]) if nearest_res else None,
            "reasons": ["candle_mid_put", f"res_dist={dist_res_atr:.2f}ATR"] + reasons
        }

    # Entrada em suporte -> CALL (com tendência confirmando ou neutra)
    if near_sup and trend_dir in ("CALL", "NEUTRAL"):
        if CM_CONFIRM_CANDLE and last_dir != "CALL":
            return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["vela_confirma_call"]}
        score = 0.55 + (CM_MAX_DIST_ATR - dist_sup_atr) * 0.20 + min(0.20, trend_strength * 0.10)
        return {
            "trade": True,
            "dir": "CALL",
            "score": float(min(0.95, max(0.0, score))),
            "sr_type": "SUP",
            "sr_level": float(nearest_sup[0]) if nearest_sup else None,
            "reasons": ["candle_mid_call", f"sup_dist={dist_sup_atr:.2f}ATR"] + reasons
        }

    return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["sem_sr_proximo"]}

# ===================== INDICADORES M5 =====================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def strategy_m5_indicators(df_m5: pd.DataFrame) -> Dict[str, Any]:
    if df_m5 is None or len(df_m5) < 30:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["poucas_velas_m5"]}

    close = df_m5["close"]
    ma16 = close.rolling(16).mean()
    macd_line, signal_line, hist = macd(close, 12, 26, 9)
    rsi14 = rsi(close, 14)

    # Tendência (slope da MA16 nos últimos 6 candles)
    ma_tail = ma16.tail(6).to_numpy(float)
    trend_slope = float(ma_tail[-1] - ma_tail[0])
    atr_m5 = atr(df_m5, 14)
    slope_thr = max(atr_m5 * 0.05, abs(ma_tail[-1]) * 0.0001)
    if trend_slope > slope_thr:
        trend_dir = "CALL"
    elif trend_slope < -slope_thr:
        trend_dir = "PUT"
    else:
        trend_dir = "NEUTRAL"

    last = df_m5.iloc[-1]
    ma = float(ma16.iloc[-1])
    m = float(macd_line.iloc[-1])
    s = float(signal_line.iloc[-1])
    h = float(hist.iloc[-1])
    r = float(rsi14.iloc[-1])
    c = float(last["close"])

    reasons = []
    score = 0.0

    # Confirmação de vela: última vela DEVE ser na direção do trade
    last_candle_green = float(last["close"]) > float(last["open"])
    last_candle_red = float(last["close"]) < float(last["open"])

    if c > ma and m > s and r > 52 and trend_dir in ("CALL", "NEUTRAL"):
        reasons.append("MA16_OK")
        reasons.append("MACD_OK")
        reasons.append("RSI_OK")
        reasons.append("TREND_OK")
        score = min(1.0, 0.5 + (r - 50) / 100 + abs(h) * 10)
        if not last_candle_green:
            reasons.append("VELA_NAO_CONFIRMA")
            return {"trade": False, "dir": "CALL", "score": score, "reasons": reasons}
        return {"trade": True, "dir": "CALL", "score": score, "reasons": reasons}

    if c < ma and m < s and r < 48 and trend_dir in ("PUT", "NEUTRAL"):
        reasons.append("MA16_OK")
        reasons.append("MACD_OK")
        reasons.append("RSI_OK")
        reasons.append("TREND_OK")
        score = min(1.0, 0.5 + (50 - r) / 100 + abs(h) * 10)
        if not last_candle_red:
            reasons.append("VELA_NAO_CONFIRMA")
            return {"trade": False, "dir": "PUT", "score": score, "reasons": reasons}
        return {"trade": True, "dir": "PUT", "score": score, "reasons": reasons}

    return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["sem_sinal_m5"]}

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

def candle_pattern_signal(df: pd.DataFrame) -> Tuple[str, str]:
    """Retorna (dir, pattern) com base em padrões simples das últimas 2 velas."""
    if len(df) < 2:
        return "NEUTRAL", "sem_dados"

    c1 = df.iloc[-1]
    c0 = df.iloc[-2]

    o1, c1c, h1, l1 = float(c1["open"]), float(c1["close"]), float(c1["high"]), float(c1["low"])
    o0, c0c, h0, l0 = float(c0["open"]), float(c0["close"]), float(c0["high"]), float(c0["low"])

    # Engolfo
    if c1c > o1 and c0c < o0 and c1c >= o0 and o1 <= c0c:
        return "CALL", "engolfo_alta"
    if c1c < o1 and c0c > o0 and c1c <= o0 and o1 >= c0c:
        return "PUT", "engolfo_baixa"

    # Pinbar simples
    w = wick_fractions(c1)
    if w["lower_frac"] > 0.55 and w["body_frac"] < 0.35:
        return "CALL", "pinbar_alta"
    if w["upper_frac"] > 0.55 and w["body_frac"] < 0.35:
        return "PUT", "pinbar_baixa"

    return "NEUTRAL", "sem_padrao"

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
    Retorna (resistencias, suportes) com POLARIDADE DINÂMICA.
    Combina highs+lows e classifica com base no preço atual:
    - Nível ACIMA do preço → Resistência
    - Nível ABAIXO do preço → Suporte
    """
    sub = df_m1.tail(SR_LOOKBACK)
    h = sub["high"].to_numpy(float)
    l = sub["low"].to_numpy(float)
    current_price = float(sub["close"].iloc[-1])

    highs: List[float] = []
    lows: List[float] = []

    k = 2
    for i in range(k, len(sub) - k):
        if h[i] == np.max(h[i-k:i+k+1]):
            highs.append(float(h[i]))
        if l[i] == np.min(l[i-k:i+k+1]):
            lows.append(float(l[i]))

    tol_price = max(atr_val * SR_CLUSTER_ATR, 1e-9)

    # POLARIDADE: combina TODOS os níveis (highs+lows) numa pool única
    all_levels = highs + lows
    combined = _cluster_levels(all_levels, tol_price)

    # Classifica com base no preço atual
    res: List[Tuple[float, int]] = []
    sup: List[Tuple[float, int]] = []

    for lvl, n in combined:
        if n < SR_MIN_TOUCHES_STRONG:
            continue
        if lvl > current_price:
            res.append((lvl, n))
        elif lvl < current_price:
            sup.append((lvl, n))
        # lvl == current_price: ignora (estamos exatamente no nível)

    return res, sup

def strong_sr_levels_recent(df_m1: pd.DataFrame, atr_val: float, lookback: int = 60, k: int = 1) -> Tuple[List[Tuple[float,int]], List[Tuple[float,int]]]:
    sub = df_m1.tail(lookback)
    if len(sub) < max(20, lookback // 2):
        return [], []
    h = sub["high"].to_numpy(float)
    l = sub["low"].to_numpy(float)
    current_price = float(sub["close"].iloc[-1])

    highs: List[float] = []
    lows: List[float] = []

    for i in range(k, len(sub) - k):
        if h[i] == np.max(h[i-k:i+k+1]):
            highs.append(float(h[i]))
        if l[i] == np.min(l[i-k:i+k+1]):
            lows.append(float(l[i]))

    tol_price = max(atr_val * SR_CLUSTER_ATR, 1e-9)
    all_levels = highs + lows
    combined = _cluster_levels(all_levels, tol_price)

    res: List[Tuple[float, int]] = []
    sup: List[Tuple[float, int]] = []

    for lvl, n in combined:
        if lvl > current_price:
            res.append((lvl, n))
        elif lvl < current_price:
            sup.append((lvl, n))

    return res, sup

def pick_top_levels(levels: List[Tuple[float,int]], top_n: int) -> List[Tuple[float,int]]:
    return sorted(levels, key=lambda t: t[1], reverse=True)[:top_n]

def nearest_k(levels: List[Tuple[float,int]], price: float, k: int) -> List[Tuple[float,int,float]]:
    arr = [(lvl, touches, abs(lvl - price)) for (lvl, touches) in levels]
    arr.sort(key=lambda x: x[2])
    return arr[:k]

def _detect_breakout_pattern_sr(df: pd.DataFrame, atr_safe: float, lookback: int = 40, k: int = 2) -> Tuple[bool, bool]:
    sub = df.tail(max(lookback, k * 2 + 1))
    if len(sub) < (k * 2 + 5):
        return False, False
    h = sub["high"].to_numpy(float)
    l = sub["low"].to_numpy(float)

    swing_highs = []
    swing_lows = []
    for i in range(k, len(sub) - k):
        if h[i] == np.max(h[i-k:i+k+1]):
            swing_highs.append(float(h[i]))
        if l[i] == np.min(l[i-k:i+k+1]):
            swing_lows.append(float(l[i]))

    topos_desc = False
    if len(swing_highs) >= 3:
        ultimos = swing_highs[-3:]
        min_diff = atr_safe * 0.3
        if (ultimos[0] - ultimos[1]) >= min_diff and (ultimos[1] - ultimos[2]) >= min_diff:
            topos_desc = True
        else:
            total_drop = ultimos[0] - ultimos[-1]
            if total_drop >= atr_safe * 1.0:
                topos_desc = True

    fundos_asc = False
    if len(swing_lows) >= 3:
        ultimos = swing_lows[-3:]
        min_diff = atr_safe * 0.3
        if (ultimos[1] - ultimos[0]) >= min_diff and (ultimos[2] - ultimos[1]) >= min_diff:
            fundos_asc = True
        else:
            total_rise = ultimos[-1] - ultimos[0]
            if total_rise >= atr_safe * 1.0:
                fundos_asc = True

    return topos_desc, fundos_asc

def _nearest_opposite_dist_atr(res: List[Tuple[float,int]], sup: List[Tuple[float,int]], price: float, atr_safe: float) -> Tuple[float, float]:
    nearest_res = min([abs(lvl - price) for lvl, _t in res if lvl >= price], default=9999.0)
    nearest_sup = min([abs(lvl - price) for lvl, _t in sup if lvl <= price], default=9999.0)
    return nearest_res / atr_safe, nearest_sup / atr_safe

# ===================== IA RF PROXY (OTC) =====================
def add_volume_proxies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ret_1"] = out["close"].pct_change()
    out["range"] = out["high"] - out["low"]
    out["range_pct"] = out["range"] / out["close"].replace(0, np.nan)

    prev_close = out["close"].shift(1)
    out["true_range"] = np.maximum(
        out["high"] - out["low"],
        np.maximum(
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs()
        )
    )

    out["body"] = (out["close"] - out["open"]).abs()
    out["upper_wick"] = out["high"] - np.maximum(out["open"], out["close"])
    out["lower_wick"] = np.minimum(out["open"], out["close"]) - out["low"]
    out["wick_ratio"] = (out["upper_wick"] + out["lower_wick"]) / (out["body"] + 1e-9)

    out["pressure"] = (out["close"] - out["open"]) / (out["range"] + 1e-9)

    out["vol_10"] = out["ret_1"].rolling(10, min_periods=10).std()
    out["vol_20"] = out["ret_1"].rolling(20, min_periods=20).std()

    out["vol_proxy_raw"] = (
        out["range_pct"].rolling(5, min_periods=5).mean() +
        out["vol_10"]
    )

    mu = out["vol_proxy_raw"].rolling(200, min_periods=40).mean()
    sd = out["vol_proxy_raw"].rolling(200, min_periods=40).std()
    out["act_z"] = (out["vol_proxy_raw"] - mu) / (sd + 1e-9)
    out["act_z"] = out["act_z"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["vol_proxy"] = out["act_z"]

    return out


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ret_3"] = out["close"].pct_change(3)
    out["ret_5"] = out["close"].pct_change(5)

    out["ma_10"] = out["close"].rolling(10).mean()
    out["ma_20"] = out["close"].rolling(20).mean()
    out["ma_dist_10"] = (out["close"] - out["ma_10"]) / out["close"]
    out["ma_dist_20"] = (out["close"] - out["ma_20"]) / out["close"]

    sign_ret = np.sign(out["ret_1"].fillna(0))
    out["chop_10"] = (sign_ret.diff().abs() > 0).rolling(10, min_periods=10).mean()

    out["y"] = (out["close"].shift(-1) > out["close"]).astype(int)

    out = out.dropna().reset_index(drop=True)
    return out


def build_rf_proxy_model() -> "Pipeline":
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=40,
            random_state=42,
            n_jobs=-1
        ))
    ])


def rf_proxy_predict_p_up(df_m1: pd.DataFrame) -> Optional[float]:
    if df_m1 is None or len(df_m1) < RF_PROXY_MIN_ROWS:
        return None
    try:
        df_feat = add_volume_proxies(df_m1)
        df_feat = make_features(df_feat)
    except Exception:
        return None

    feature_cols = [
        "ret_1", "ret_3", "ret_5",
        "range_pct", "true_range",
        "wick_ratio", "pressure",
        "vol_10", "vol_20", "act_z",
        "ma_dist_10", "ma_dist_20",
        "chop_10"
    ]

    if len(df_feat) < 120:
        return None

    split = int(len(df_feat) * 0.7)
    if split < 60 or (len(df_feat) - split) < 10:
        return None

    X = df_feat[feature_cols].values
    y = df_feat["y"].values
    X_train, y_train = X[:split], y[:split]
    X_last = X[-1:]

    model = build_rf_proxy_model()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_last)[0]
    return float(probs[1])


def setup_rf_proxy(df_m1: pd.DataFrame, prob_threshold: float = 0.60, min_rows: int = 260) -> Dict[str, Any]:
    if df_m1 is None or len(df_m1) < min_rows:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["poucas_velas"]}

    try:
        df_feat = add_volume_proxies(df_m1)
        df_feat = make_features(df_feat)
    except Exception:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["features_fail"]}

    feature_cols = [
        "ret_1", "ret_3", "ret_5",
        "range_pct", "true_range",
        "wick_ratio", "pressure",
        "vol_10", "vol_20", "act_z",
        "ma_dist_10", "ma_dist_20",
        "chop_10"
    ]

    if len(df_feat) < 120:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["poucas_features"]}

    split = int(len(df_feat) * 0.7)
    if split < 60 or (len(df_feat) - split) < 10:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["split_insuf"]}

    df_feat["range_pct_ma"] = df_feat["range_pct"].rolling(RF_PROXY_RANGE_MA_WINDOW, min_periods=10).mean()
    X = df_feat[feature_cols].values
    y = df_feat["y"].values

    X_train, y_train = X[:split], y[:split]
    X_last = X[-1:]
    row_last = df_feat.iloc[-1]

    model = build_rf_proxy_model()
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_last)[0]
    p_dn, p_up = float(probs[0]), float(probs[1])
    max_p = max(p_up, p_dn)

    act_z = float(row_last.get("act_z", 0.0))
    wick_ratio = float(row_last.get("wick_ratio", 0.0))
    pressure = float(row_last.get("pressure", 0.0))
    chop_10 = float(row_last.get("chop_10", 0.0))
    range_pct = float(row_last.get("range_pct", 0.0))
    range_pct_ma = float(row_last.get("range_pct_ma", 0.0))

    if act_z <= RF_PROXY_MIN_ACT_Z:
        return {"trade": False, "dir": "NEUTRAL", "score": max_p, "reasons": [f"baixa_atividade(act_z={act_z:.2f})"]}
    if range_pct < range_pct_ma:
        return {"trade": False, "dir": "NEUTRAL", "score": max_p, "reasons": ["range_pequeno"]}
    if wick_ratio >= RF_PROXY_MAX_WICK:
        return {"trade": False, "dir": "NEUTRAL", "score": max_p, "reasons": ["indecisao_wick"]}
    if chop_10 >= RF_PROXY_MAX_CHOP:
        return {"trade": False, "dir": "NEUTRAL", "score": max_p, "reasons": ["mercado_picotado"]}
    if max_p < prob_threshold:
        return {"trade": False, "dir": "NEUTRAL", "score": max_p, "reasons": [f"baixa_conf({max_p:.2f})"]}

    direction = "CALL" if p_up >= p_dn else "PUT"
    reasons = [f"rf_proxy(p_up={p_up:.2f},act_z={act_z:.2f},wick={wick_ratio:.2f},chop={chop_10:.2f})"]

    return {
        "trade": True,
        "dir": direction,
        "score": max_p,
        "reasons": reasons,
        "rf_p_up": p_up,
        "rf_p_dn": p_dn,
        "act_z": act_z,
        "wick_ratio": wick_ratio,
        "chop_10": chop_10,
    }

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

# ===================== SETUP SIMPLIFICADO: S/R + CNN =====================
# Importa AutoTuner para parametros dinamicos
try:
    from auto_tuner import get_tuner
    AUTO_TUNER_AVAILABLE = True
except ImportError:
    AUTO_TUNER_AVAILABLE = False

# ===================== ESTRATÉGIA: TENDÊNCIA + PADRÕES TA-LIB + IA =====================
def setup_trend_candle(df_m1: pd.DataFrame, atr_val: float, tuner_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Estratégia baseada nos 13 PADRÕES MAIS CONFIÁVEIS do TA-Lib (>80% eficácia):
    
    BULLISH (CALL):
    1. Three White Soldiers (~83%)
    2. Bullish Engulfing (~82%)
    3. Morning Star (~81%)
    4. Piercing Line (~80%)
    5. Bullish Harami (~80%)
    6. Hammer (~80%)
    
    BEARISH (PUT):
    7. Three Black Crows (~84%)
    8. Bearish Engulfing (~82%)
    9. Evening Star (~81%)
    10. Dark Cloud Cover (~80%)
    11. Bearish Harami (~80%)
    12. Shooting Star (~80%)
    13. Hanging Man (~80%)
    
    CatBoost (IA) confirma entrada depois.
    """
    no_trade = {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": []}

    if len(df_m1) < 30:
        return {**no_trade, "reasons": ["dados_insuficientes"]}

    atr_safe = max(float(atr_val), 1e-9)

    # Arrays para TA-Lib
    op = df_m1["open"].to_numpy(float)
    hi = df_m1["high"].to_numpy(float)
    lo = df_m1["low"].to_numpy(float)
    cl = df_m1["close"].to_numpy(float)

    # ===== DETECÇÃO DE SUPORTE/RESISTÊNCIA (TOPOS E FUNDOS) =====
    # Identifica níveis de S/R usando últimas 20 velas
    sr_window = min(20, len(hi) - 1)
    if sr_window >= 5:
        recent_highs = hi[-sr_window:]
        recent_lows = lo[-sr_window:]
        
        # Encontra topos (resistência) - máximas locais
        resistance_level = np.max(recent_highs)
        # Encontra fundos (suporte) - mínimas locais
        support_level = np.min(recent_lows)
        
        # Preço atual
        current_price = cl[-1]
        price_range = resistance_level - support_level
        
        # Margem de 5% do range para considerar "perto" de S/R
        sr_margin = price_range * 0.05 if price_range > 0 else 0.0001
        
        near_resistance = current_price >= (resistance_level - sr_margin)
        near_support = current_price <= (support_level + sr_margin)
    else:
        near_resistance = False
        near_support = False
        resistance_level = 0
        support_level = 0

    # ===== VERIFICAÇÃO DE TENDÊNCIA PRÉVIA (para validar padrões de reversão) =====
    # Tendência das últimas 7 velas (exclui a atual) - threshold aumentado para filtrar ruído
    if len(cl) >= 8:
        trend_closes = cl[-8:-1]  # 7 velas anteriores
        trend_highs = hi[-8:-1]
        trend_lows = lo[-8:-1]
        trend_change = (trend_closes[-1] - trend_closes[0]) / (trend_closes[0] + 1e-9)
        
        # Threshold aumentado: 0.15% mínimo de movimento para considerar tendência
        prior_trend_up = trend_change > 0.0015  # Alta de pelo menos 0.15%
        prior_trend_down = trend_change < -0.0015  # Queda de pelo menos 0.15%
        
        # Verificar se é mercado em RANGE (lateral) - não operar
        range_total = max(trend_highs) - min(trend_lows)
        atr_check = atr_safe if atr_safe > 0 else range_total / 5
        is_ranging = range_total < atr_check * 2  # Range muito pequeno = lateralizado
    else:
        prior_trend_up = False
        prior_trend_down = False
        is_ranging = True  # Sem dados suficientes = assume range

    # ===== OS 13 PADRÕES MAIS CONFIÁVEIS (>80% eficácia) =====
    
    # BULLISH (6 padrões)
    three_white = talib.CDL3WHITESOLDIERS(op, hi, lo, cl)    # ~83%
    engulfing = talib.CDLENGULFING(op, hi, lo, cl)            # ~82%
    morning_star = talib.CDLMORNINGSTAR(op, hi, lo, cl)       # ~81%
    piercing = talib.CDLPIERCING(op, hi, lo, cl)              # ~80%
    harami = talib.CDLHARAMI(op, hi, lo, cl)                  # ~80%
    hammer = talib.CDLHAMMER(op, hi, lo, cl)                  # ~80%
    
    # BEARISH (7 padrões)
    three_black = talib.CDL3BLACKCROWS(op, hi, lo, cl)        # ~84%
    # engulfing já calculado acima                             # ~82%
    evening_star = talib.CDLEVENINGSTAR(op, hi, lo, cl)       # ~81%
    dark_cloud = talib.CDLDARKCLOUDCOVER(op, hi, lo, cl)      # ~80%
    # harami já calculado acima                                # ~80%
    shooting_star = talib.CDLSHOOTINGSTAR(op, hi, lo, cl)     # ~80%
    hanging_man = talib.CDLHANGINGMAN(op, hi, lo, cl)         # ~80%

    # Última vela e vela anterior (para verificar força do engolfo)
    last = df_m1.iloc[-1]
    last_open = float(last["open"])
    last_close = float(last["close"])
    last_high = float(last["high"])
    last_low = float(last["low"])
    
    prev = df_m1.iloc[-2] if len(df_m1) >= 2 else last
    prev_open = float(prev["open"])
    prev_close = float(prev["close"])
    prev_body = abs(prev_close - prev_open)
    
    candle_green = last_close > last_open
    candle_red = last_close < last_open
    current_body = abs(last_close - last_open)
    total_range = last_high - last_low
    
    # Verificar se engolfo é forte (corpo atual > 75% do corpo anterior)
    engolfo_forte = current_body >= prev_body * 0.75 if prev_body > 0 else False

    # BLOQUEIO: Mercado em range/lateralizado - não operar
    if is_ranging:
        return {**no_trade, "reasons": ["mercado_range_lateral"]}

    # Detecta padrões na última vela
    padrao_call = []
    padrao_put = []

    # CALL patterns (valor > 0) - Ordem por eficácia
    if three_white[-1] > 0: padrao_call.append("3_soldados")      # 83%
    # Engolfo de ALTA: tendência prévia de BAIXA + engolfo forte
    if engulfing[-1] > 0 and prior_trend_down and engolfo_forte: padrao_call.append("engolfo_alta")  # 82%
    if morning_star[-1] > 0: padrao_call.append("estrela_manha")  # 81%
    if piercing[-1] > 0: padrao_call.append("piercing")           # 80%
    if harami[-1] > 0: padrao_call.append("harami_alta")          # 80%
    if hammer[-1] > 0: padrao_call.append("martelo")              # 80%

    # PUT patterns - Ordem por eficácia
    if three_black[-1] != 0: padrao_put.append("3_corvos")        # 84%
    # Engolfo de BAIXA: tendência prévia de ALTA + engolfo forte
    if engulfing[-1] < 0 and prior_trend_up and engolfo_forte: padrao_put.append("engolfo_baixa")  # 82%
    if evening_star[-1] != 0: padrao_put.append("estrela_noite")  # 81%
    if dark_cloud[-1] != 0: padrao_put.append("nuvem_negra")      # 80%
    if harami[-1] < 0: padrao_put.append("harami_baixa")          # 80%
    if shooting_star[-1] != 0: padrao_put.append("estrela_cad")   # 80%
    if hanging_man[-1] != 0: padrao_put.append("enforcado")       # 80%

    # ===== VERIFICAÇÃO DE SUPORTE/RESISTÊNCIA =====
    # CALL perto de resistência = perigoso (pode reverter para baixo)
    # PUT perto de suporte = perigoso (pode reverter para cima)
    sr_block_call = near_resistance
    sr_block_put = near_support
    
    # Mas: CALL em suporte com padrão de alta = ÓTIMO (reversão de alta)
    # PUT em resistência com padrão de baixa = ÓTIMO (reversão de baixa)
    sr_boost_call = near_support and len(padrao_call) > 0
    sr_boost_put = near_resistance and len(padrao_put) > 0

    # ===== DECISÃO FINAL =====
    has_call_pattern = len(padrao_call) > 0 and candle_green
    has_put_pattern = len(padrao_put) > 0 and candle_red
    
    # Bloqueia se estiver em zona de S/R contrária
    if has_call_pattern and sr_block_call and not sr_boost_call:
        return {**no_trade, "reasons": [f"call_perto_resistencia({'+'.join(padrao_call)})"]}
    if has_put_pattern and sr_block_put and not sr_boost_put:
        return {**no_trade, "reasons": [f"put_perto_suporte({'+'.join(padrao_put)})"]}

    if not has_call_pattern and not has_put_pattern:
        if len(padrao_call) > 0:
            return {**no_trade, "reasons": [f"call_vela_vermelha({'+'.join(padrao_call)})"]}
        elif len(padrao_put) > 0:
            return {**no_trade, "reasons": [f"put_vela_verde({'+'.join(padrao_put)})"]}
        else:
            return {**no_trade, "reasons": ["sem_padrao_forte"]}

    # Score baseado na eficácia do padrão
    score = 0.70  # Base alta para padrões >80%
    trade_dir = "NEUTRAL"
    sinais = []

    if has_call_pattern:
        trade_dir = "CALL"
        sinais.extend(padrao_call)
        sinais.append("verde")
        
        # Bônus por padrões mais fortes
        if "3_soldados" in padrao_call:          # 83%
            score += 0.13
        elif "engolfo_alta" in padrao_call:      # 82%
            score += 0.12
        elif "estrela_manha" in padrao_call:     # 81%
            score += 0.11
        else:                                     # 80%
            score += 0.10
        
        # Bônus por CALL em suporte (reversão de alta confirmada)
        if sr_boost_call:
            score += 0.05
            sinais.append("suporte")

    elif has_put_pattern:
        trade_dir = "PUT"
        sinais.extend(padrao_put)
        sinais.append("vermelho")
        
        if "3_corvos" in padrao_put:             # 84%
            score += 0.14
        elif "engolfo_baixa" in padrao_put:      # 82%
            score += 0.12
        elif "estrela_noite" in padrao_put:      # 81%
            score += 0.11
        else:                                     # 80%
            score += 0.10
        
        # Bônus por PUT em resistência (reversão de baixa confirmada)
        if sr_boost_put:
            score += 0.05
            sinais.append("resistencia")

    # Filtro: Vela muito esticada (> 3 ATR) - risco alto
    if total_range > atr_safe * 3.0:
        return {**no_trade, "reasons": ["vela_muito_esticada"]}

    score = min(1.0, score)

    return {
        "trade": True,
        "dir": trade_dir,
        "score": score,
        "strategy": "PATTERN_CANDLE",
        "reasons": [f"PATTERN({trade_dir},sinais={'+'.join(sinais)})"]
    }


def setup_sr_simples(df_m1: pd.DataFrame, atr_val: float, tuner_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Setup SIMPLIFICADO baseado apenas em Suporte/Resistência.

    Usa parametros do AutoTuner para auto-ajuste:
    - tolerancia_atr: quanto de "folga" para considerar que tocou
    - min_toques: minimo de toques para S/R valido
    - max_range_atr: filtro de vela esticada
    - min_wick_frac: minimo de pavio para rejeicao

    CONDICOES:
    - Toque no S/R (dentro da tolerancia)
    - Fechou do lado certo (acima suporte / abaixo resistencia)
    - Sinal de reversao (pavio OU cor OU padrao)
    - NAO ENTRA CONTRA TENDENCIA FORTE (ultimas velas)
    """
    if len(df_m1) < 50:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["poucas_velas"]}

    # Obtem parametros do AutoTuner ou usa defaults
    if tuner_params is None:
        if AUTO_TUNER_AVAILABLE:
            tuner_params = get_tuner().get_params()
        else:
            tuner_params = {
                "tolerancia_atr": 0.25,
                "min_toques": 2,
                "max_range_atr": 1.50,
                "min_wick_frac": 0.35,
            }

    TOLERANCIA_ATR = tuner_params.get("tolerancia_atr", 0.25)
    MIN_TOUCHES = tuner_params.get("min_toques", 2)
    MAX_RANGE_ATR = tuner_params.get("max_range_atr", 1.50)
    MIN_WICK_FRAC = tuner_params.get("min_wick_frac", 0.35)
    MAX_RANGE_ATR = max(MAX_RANGE_ATR, 2.0)
    allowed_touches = (2, 3, 4)

    res, sup = strong_sr_levels_recent(df_m1, atr_val, lookback=60, k=1)
    atr_safe = max(atr_val, 1e-9)
    nearest_res_dist_atr, nearest_sup_dist_atr = _nearest_opposite_dist_atr(res, sup, float(df_m1["close"].iloc[-1]), atr_safe)

    # Dados do ultimo candle fechado
    last_candle = df_m1.iloc[-1]
    candle_high = float(last_candle["high"])
    candle_low = float(last_candle["low"])
    candle_close = float(last_candle["close"])
    candle_open = float(last_candle["open"])
    w = wick_fractions(last_candle)
    range_atr = candle_range(last_candle) / atr_safe

    # === TOPO/FUNDO RECENTES (NOVO) ===
    # Usa máximas/mínimas recentes como referência de topo/fundo
    tb_lb = max(10, min(SR_TB_LOOKBACK, len(df_m1)))
    last_tb = df_m1.tail(tb_lb)
    recent_high = float(last_tb["high"].max())
    recent_low = float(last_tb["low"].min())
    dist_top_atr = abs(candle_high - recent_high) / atr_safe
    dist_bot_atr = abs(candle_low - recent_low) / atr_safe

    # Filtro: vela esticada (parametro dinamico)
    if range_atr > MAX_RANGE_ATR:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["vela_esticada"]}

    # === VERIFICACAO DE TENDENCIA LOCAL (7 velas) ===
    last_7 = df_m1.tail(7)
    velas_alta = sum(1 for _, r in last_7.iterrows() if r['close'] > r['open'])
    velas_baixa = 7 - velas_alta

    # Variacao de preco nas ultimas 7 velas
    preco_7_atras = float(df_m1.iloc[-7]['close'])
    variacao_pct = ((candle_close - preco_7_atras) / preco_7_atras) * 100 if preco_7_atras > 0 else 0

    # Tendencia forte de ALTA: N velas verdes OU subiu mais de X%
    tendencia_alta_forte = (velas_alta >= SR_TREND_STRONG_BARS) and (variacao_pct > SR_TREND_STRONG_PCT)

    # Tendencia forte de BAIXA: N velas vermelhas OU caiu mais de X%
    tendencia_baixa_forte = (velas_baixa >= SR_TREND_STRONG_BARS) and (variacao_pct < -SR_TREND_STRONG_PCT)

    if variacao_pct > (SR_TREND_STRONG_PCT * 2.0):
        tendencia_alta_forte = True
    if variacao_pct < (-SR_TREND_STRONG_PCT * 2.0):
        tendencia_baixa_forte = True

    # === VERIFICACAO DE TENDENCIA MÉDIO PRAZO (20 velas) ===
    # Detecta movimentos fortes que o filtro de 7 velas não pega (ex: rally de 30min)
    if len(df_m1) >= 20:
        preco_20_atras = float(df_m1.iloc[-20]['close'])
        variacao_20 = candle_close - preco_20_atras
        variacao_20_atr = abs(variacao_20) / atr_safe
        # Se preço moveu mais de 3 ATR em 20 velas = tendência forte de médio prazo
        if variacao_20 > 0 and variacao_20_atr >= 3.0:
            tendencia_alta_forte = True  # Reforça: não entra PUT contra rally
        elif variacao_20 < 0 and variacao_20_atr >= 3.0:
            tendencia_baixa_forte = True  # Reforça: não entra CALL contra queda

    # === BLOQUEIO POR SEQUÊNCIA FORTE DE VELAS DA MESMA COR ===
    streak_n = max(3, SR_STREAK_BLOCK_BARS)
    last_n = df_m1.tail(streak_n)
    up_n = sum(1 for _, r in last_n.iterrows() if r['close'] > r['open'])
    down_n = streak_n - up_n
    max_ratio = max(up_n, down_n) / max(1, streak_n)
    if max_ratio >= SR_STREAK_BLOCK_RATIO:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": [f"streak_forte_{max_ratio:.0%}"]}

    # Padrao de velas simples
    pad_dir, pad_name = candle_pattern_signal(df_m1)

    # Tolerancia em valor absoluto
    tolerancia = atr_safe * TOLERANCIA_ATR

    # === TOPOS E FUNDOS RECENTES ===
    def _recent_swings(df: pd.DataFrame, lookback: int = 30, k: int = 2) -> Tuple[Optional[float], Optional[float]]:
        sub = df.tail(max(lookback, k * 2 + 1))
        if len(sub) < (k * 2 + 3):
            return None, None
        h = sub["high"].to_numpy(float)
        l = sub["low"].to_numpy(float)
        last_high = None
        last_low = None
        for i in range(k, len(sub) - k):
            if h[i] == np.max(h[i-k:i+k+1]):
                last_high = float(h[i])
            if l[i] == np.min(l[i-k:i+k+1]):
                last_low = float(l[i])
        return last_high, last_low

    last_swing_high, last_swing_low = _recent_swings(df_m1, lookback=30, k=2)
    swing_tol = atr_safe * max(0.30, TOLERANCIA_ATR * 1.4)

    def _recent_break(level: float, side: str, lookback: int = 12) -> bool:
        sub = df_m1.tail(max(lookback, 3))
        if len(sub) < 3:
            return False
        break_atr = max(atr_safe * 0.15, 1e-9)
        if side == "SUP":
            return any(float(c) < (level - break_atr) for c in sub["close"])
        return any(float(c) > (level + break_atr) for c in sub["close"])

    # === FIX 15: DETECÇÃO DE PADRÃO DE ROMPIMENTO ===
    # Topos descendentes perto de suporte = triângulo descendente → rompimento para baixo
    # Fundos ascendentes perto de resistência = triângulo ascendente → rompimento para cima
    def _detect_breakout_pattern(df: pd.DataFrame, lookback: int = 40, k: int = 2):
        """
        Detecta padrões de triângulo que indicam rompimento iminente.
        Returns: (topos_descendentes: bool, fundos_ascendentes: bool)
        """
        sub = df.tail(max(lookback, k * 2 + 1))
        if len(sub) < (k * 2 + 5):
            return False, False
        h = sub["high"].to_numpy(float)
        l = sub["low"].to_numpy(float)

        # Coleta todos os swing highs e swing lows
        swing_highs = []
        swing_lows = []
        for i in range(k, len(sub) - k):
            if h[i] == np.max(h[i-k:i+k+1]):
                swing_highs.append(float(h[i]))
            if l[i] == np.min(l[i-k:i+k+1]):
                swing_lows.append(float(l[i]))

        # Topos descendentes: últimos 3+ highs cada vez menores
        topos_desc = False
        if len(swing_highs) >= 3:
            ultimos = swing_highs[-3:]
            # Cada topo menor que o anterior (com tolerância mínima de 0.3 ATR)
            min_diff = atr_safe * 0.3
            if (ultimos[0] - ultimos[1]) >= min_diff and (ultimos[1] - ultimos[2]) >= min_diff:
                topos_desc = True
            # Também aceita 2 de 3 descendentes com diferença significativa
            elif len(swing_highs) >= 3:
                total_drop = ultimos[0] - ultimos[-1]
                if total_drop >= atr_safe * 1.0:
                    # Queda total de pelo menos 1 ATR nos topos
                    topos_desc = True

        # Fundos ascendentes: últimos 3+ lows cada vez maiores
        fundos_asc = False
        if len(swing_lows) >= 3:
            ultimos = swing_lows[-3:]
            min_diff = atr_safe * 0.3
            if (ultimos[1] - ultimos[0]) >= min_diff and (ultimos[2] - ultimos[1]) >= min_diff:
                fundos_asc = True
            elif len(swing_lows) >= 3:
                total_rise = ultimos[-1] - ultimos[0]
                if total_rise >= atr_safe * 1.0:
                    fundos_asc = True

        return topos_desc, fundos_asc

    topos_descendentes, fundos_ascendentes = _detect_breakout_pattern_sr(df_m1, atr_safe, lookback=SR_BREAK_LOOKBACK, k=2)

    best_setup = {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": []}

    # === VERIFICA SUPORTE → CALL ===
    # BLOQUEIO: Nao entra CALL se tendencia forte de BAIXA!
    if tendencia_baixa_forte:
        # Tendencia forte de baixa - nao entra CALL (seria contra tendencia)
        pass  # Pula verificacao de suporte
    else:
        for sup_lvl, sup_toques in sup:
            if sup_toques < MIN_TOUCHES:
                continue
            if sup_lvl > candle_close:
                continue

            # 1. TOQUE: minima do candle perto do suporte
            dist_suporte = abs(candle_low - sup_lvl)
            tocou_suporte = dist_suporte <= tolerancia

            if not tocou_suporte:
                continue

            eff_toques = sup_toques + 1
            if eff_toques < MIN_TOUCHES:
                continue
            if eff_toques not in allowed_touches:
                continue

            if _recent_break(sup_lvl, "SUP"):
                continue

            # 2. FECHOU ACIMA: nao rompeu o suporte
            fechou_acima = candle_close > sup_lvl

            # 2.1 TOPO/FUNDO: suporte alinhado a fundo recente (suave)
            swing_ok = True
            if last_swing_low is not None and abs(sup_lvl - last_swing_low) > swing_tol:
                swing_ok = False

            # 3. SINAL DE REVERSAO (mais permissivo):
            #    - Candle de alta (verde)
            #    - Pavio inferior significativo
            #    - Padrao de alta detectado
            pavio_inferior = w["lower_frac"] >= MIN_WICK_FRAC
            candle_alta = candle_close > candle_open
            padrao_alta = (pad_dir == "CALL")
            tem_sinal = candle_alta or pavio_inferior or padrao_alta

            # EXAUSTÃO S/R: muitos toques = nível fraco, não forte
            # 2-4 toques = nível confirmado (bom)
            # 5+ toques = nível exaurido (provável rompimento)
            if eff_toques >= 8:
                continue  # Nível exaurido - NÃO opera, espera rompimento

            # FIX 15: PADRÃO DE ROMPIMENTO - Topos descendentes em suporte
            # Se os topos estão cada vez menores, o suporte vai romper → NÃO compra
            if topos_descendentes and eff_toques >= 4:
                continue  # Triângulo descendente → rompimento para baixo iminente

            # Score: toques moderados são bons, excesso penaliza
            if fechou_acima and tem_sinal:
                # Toques: 2=bom, 3-4=ótimo, 5=cuidado
                toques_score = min(0.15, eff_toques * 0.05) if eff_toques <= 4 else 0.05
                bonus_tb = 0.05 if (SR_TB_ON and dist_bot_atr <= SR_TB_MAX_DIST_ATR) else 0.0
                penalty_tb = 0.05 if (SR_TB_ON and dist_bot_atr > SR_TB_MAX_DIST_ATR) else 0.0
                penalty_swing = 0.05 if (SR_TB_ON and not swing_ok) else 0.0
                score = 0.55 + toques_score + (0.05 if pavio_inferior else 0) + (0.05 if candle_alta else 0) + bonus_tb - penalty_tb - penalty_swing
                score = min(1.0, score)

                sinais = []
                if pavio_inferior: sinais.append("pavio")
                if candle_alta: sinais.append("verde")
                if padrao_alta: sinais.append(pad_name)
                if eff_toques >= 5: sinais.append("exausto")
                if SR_TB_ON and dist_bot_atr <= SR_TB_MAX_DIST_ATR: sinais.append("fundo")
                if SR_TB_ON and dist_bot_atr > SR_TB_MAX_DIST_ATR: sinais.append("longe_fundo")
                if SR_TB_ON and not swing_ok: sinais.append("fundo_distante")

                if not best_setup["trade"] or score > best_setup["score"]:
                    best_setup = {
                        "trade": True,
                        "dir": "CALL",
                        "score": score,
                        "sr_type": "SUPORTE",
                        "sr_level": sup_lvl,
                        "sr_toques": eff_toques,
                        "sr_dist_atr": dist_suporte / atr_safe,
                        "opposite_dist_atr": nearest_res_dist_atr,
                        "reasons": [f"SUP(lvl={sup_lvl:.5f},toques={sup_toques},sinais={'+'.join(sinais)})"]
                    }

    # === VERIFICA RESISTENCIA → PUT ===
    # BLOQUEIO: Nao entra PUT se tendencia forte de ALTA!
    if tendencia_alta_forte:
        # Tendencia forte de alta - nao entra PUT (seria contra tendencia)
        pass  # Pula verificacao de resistencia
    else:
        for res_lvl, res_toques in res:
            if res_toques < MIN_TOUCHES:
                continue
            if res_lvl < candle_close:
                continue

            # 1. TOQUE: maxima do candle perto da resistencia
            dist_resistencia = abs(candle_high - res_lvl)
            tocou_resistencia = dist_resistencia <= tolerancia

            if not tocou_resistencia:
                continue

            eff_toques = res_toques + 1
            if eff_toques < MIN_TOUCHES:
                continue
            if eff_toques not in allowed_touches:
                continue

            if _recent_break(res_lvl, "RES"):
                continue

            # 2. FECHOU ABAIXO: nao rompeu a resistencia
            fechou_abaixo = candle_close < res_lvl

            # 2.1 TOPO/FUNDO: resistência alinhada a topo recente (suave)
            swing_ok = True
            if last_swing_high is not None and abs(res_lvl - last_swing_high) > swing_tol:
                swing_ok = False

            # 3. SINAL DE REVERSAO (mais permissivo):
            #    - Candle de baixa (vermelho)
            #    - Pavio superior significativo
            #    - Padrao de baixa detectado
            pavio_superior = w["upper_frac"] >= MIN_WICK_FRAC
            candle_baixa = candle_close < candle_open
            padrao_baixa = (pad_dir == "PUT")
            tem_sinal = candle_baixa or pavio_superior or padrao_baixa

            # EXAUSTÃO S/R: muitos toques = nível fraco, não forte
            if eff_toques >= 8:
                continue  # Nível exaurido - NÃO opera, espera rompimento

            # FIX 15: PADRÃO DE ROMPIMENTO - Fundos ascendentes em resistência
            # Se os fundos estão cada vez maiores, a resistência vai romper → NÃO vende
            if fundos_ascendentes and eff_toques >= 4:
                continue  # Triângulo ascendente → rompimento para cima iminente

            # Score: toques moderados são bons, excesso penaliza
            if fechou_abaixo and tem_sinal:
                toques_score = min(0.15, eff_toques * 0.05) if eff_toques <= 4 else 0.05
                bonus_tb = 0.05 if (SR_TB_ON and dist_top_atr <= SR_TB_MAX_DIST_ATR) else 0.0
                penalty_tb = 0.05 if (SR_TB_ON and dist_top_atr > SR_TB_MAX_DIST_ATR) else 0.0
                penalty_swing = 0.05 if (SR_TB_ON and not swing_ok) else 0.0
                score = 0.55 + toques_score + (0.05 if pavio_superior else 0) + (0.05 if candle_baixa else 0) + bonus_tb - penalty_tb - penalty_swing
                score = min(1.0, score)

                sinais = []
                if pavio_superior: sinais.append("pavio")
                if candle_baixa: sinais.append("vermelho")
                if padrao_baixa: sinais.append(pad_name)
                if eff_toques >= 5: sinais.append("exausto")
                if SR_TB_ON and dist_top_atr <= SR_TB_MAX_DIST_ATR: sinais.append("topo")
                if SR_TB_ON and dist_top_atr > SR_TB_MAX_DIST_ATR: sinais.append("longe_topo")
                if SR_TB_ON and not swing_ok: sinais.append("topo_distante")

                if not best_setup["trade"] or score > best_setup["score"]:
                    best_setup = {
                        "trade": True,
                        "dir": "PUT",
                        "score": score,
                        "sr_type": "RESISTENCIA",
                        "sr_level": res_lvl,
                        "sr_toques": eff_toques,
                        "sr_dist_atr": dist_resistencia / atr_safe,
                        "opposite_dist_atr": nearest_sup_dist_atr,
                        "reasons": [f"RES(lvl={res_lvl:.5f},toques={res_toques},sinais={'+'.join(sinais)})"]
                    }

    # === VERIFICAÇÃO DE CONFLITO S/R ===
    # Se vai entrar PUT perto de resistência, verifica se há suporte forte mais PRÓXIMO
    # Se vai entrar CALL perto de suporte, verifica se há resistência forte mais PRÓXIMA
    if best_setup["trade"]:
        setup_dir = best_setup["dir"]
        setup_sr_dist = best_setup.get("sr_dist_atr", 999)

        if setup_dir == "PUT" and sup:
            # Verifica se há suporte forte abaixo, mais próximo que a resistência
            for sup_lvl, sup_toques in sup:
                if sup_lvl < candle_close:
                    dist_sup = abs(candle_low - sup_lvl) / atr_safe
                    setup_res_dist = abs(candle_high - best_setup.get("sr_level", 0)) / atr_safe
                    # Se suporte está mais próximo (ou igualmente próximo) E tem toques comparáveis
                    if dist_sup <= setup_res_dist * 1.5 and sup_toques >= best_setup.get("sr_toques", 1) * 0.5:
                        best_setup = {"trade": False, "dir": "NEUTRAL", "score": 0.0,
                                      "reasons": [f"conflito_SR(SUP_prox={sup_lvl:.5f},toques={sup_toques},dist={dist_sup:.2f}ATR)"]}
                        break

        elif setup_dir == "CALL" and res:
            # Verifica se há resistência forte acima, mais próxima que o suporte
            for res_lvl, res_toques in res:
                if res_lvl > candle_close:
                    dist_res = abs(candle_high - res_lvl) / atr_safe
                    setup_sup_dist = abs(candle_low - best_setup.get("sr_level", 0)) / atr_safe
                    # Se resistência está mais próxima (ou igualmente próxima) E tem toques comparáveis
                    if dist_res <= setup_sup_dist * 1.5 and res_toques >= best_setup.get("sr_toques", 1) * 0.5:
                        best_setup = {"trade": False, "dir": "NEUTRAL", "score": 0.0,
                                      "reasons": [f"conflito_SR(RES_prox={res_lvl:.5f},toques={res_toques},dist={dist_res:.2f}ATR)"]}
                        break

    # Se nao encontrou setup, retorna info sobre tendencia
    if not best_setup["trade"]:
        if tendencia_alta_forte:
            best_setup["reasons"] = ["tendencia_alta_forte"]
        elif tendencia_baixa_forte:
            best_setup["reasons"] = ["tendencia_baixa_forte"]

    return best_setup


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
    # corredor curto => mercado batendo em paredes (relaxado)
    if corridor_atr <= 0.60:
        return f"pingpong(corredor={corridor_atr:.2f}ATR sup={s_lvl:.6f} res={r_lvl:.6f})"
    return None

def setup_sr_breakout(df_m1: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    if len(df_m1) < 60:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["poucas_velas"]}

    res, sup = strong_sr_levels_last200(df_m1, atr_val)
    atr_safe = max(atr_val, 1e-9)
    price = float(df_m1["close"].iloc[-1])
    prev_close = float(df_m1["close"].iloc[-2])

    nearest_res_dist_atr, nearest_sup_dist_atr = _nearest_opposite_dist_atr(res, sup, price, atr_safe)

    topos_desc, fundos_asc = _detect_breakout_pattern_sr(df_m1, atr_safe, lookback=SR_BREAK_LOOKBACK, k=2)
    break_atr = atr_safe * SR_BREAK_ATR

    best = {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": []}

    if topos_desc and sup:
        below = [(lvl, t) for (lvl, t) in sup if lvl <= price and t >= SR_BREAK_MIN_TOUCHES]
        if below:
            lvl, touches = max(below, key=lambda x: x[1])
            dist_atr = abs(price - lvl) / atr_safe
            if dist_atr <= SR_BREAK_MAX_DIST_ATR and prev_close >= (lvl - break_atr) and price < (lvl - break_atr):
                score = min(1.0, 0.60 + min(0.20, touches * 0.04) + min(0.20, (dist_atr / SR_BREAK_MAX_DIST_ATR) * 0.2))
                best = {
                    "trade": True,
                    "dir": "PUT",
                    "score": score,
                    "sr_type": "BREAK_SUP",
                    "sr_level": lvl,
                    "sr_toques": touches,
                    "sr_dist_atr": dist_atr,
                    "opposite_dist_atr": nearest_res_dist_atr,
                    "reasons": [f"BRK_SUP(lvl={lvl:.5f},toques={touches},dist={dist_atr:.2f}ATR)"]
                }

    if fundos_asc and res:
        above = [(lvl, t) for (lvl, t) in res if lvl >= price and t >= SR_BREAK_MIN_TOUCHES]
        if above:
            lvl, touches = max(above, key=lambda x: x[1])
            dist_atr = abs(price - lvl) / atr_safe
            if dist_atr <= SR_BREAK_MAX_DIST_ATR and prev_close <= (lvl + break_atr) and price > (lvl + break_atr):
                score = min(1.0, 0.60 + min(0.20, touches * 0.04) + min(0.20, (dist_atr / SR_BREAK_MAX_DIST_ATR) * 0.2))
                cand = {
                    "trade": True,
                    "dir": "CALL",
                    "score": score,
                    "sr_type": "BREAK_RES",
                    "sr_level": lvl,
                    "sr_toques": touches,
                    "sr_dist_atr": dist_atr,
                    "opposite_dist_atr": nearest_sup_dist_atr,
                    "reasons": [f"BRK_RES(lvl={lvl:.5f},toques={touches},dist={dist_atr:.2f}ATR)"]
                }
                if (not best["trade"]) or (cand["score"] > best["score"]):
                    best = cand

    return best

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

# ===================== GESTÃO DE BANCA =====================
def calcular_stake_dinamico(iq: Casa_Trader, base_stake: float) -> float:
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

def _exp_bucket(stats: Dict[str, Any], scope: str, ativo: Optional[str], direcao: Optional[str]) -> Dict[str, Any]:
    if scope == "global":
        return stats.setdefault("global", {})
    assets = stats.setdefault("assets", {})
    a = assets.setdefault(ativo or "?", {})
    return a.setdefault(direcao or "?", {})

def _exp_pick_best(bucket: Dict[str, Any], allowed: List[int]) -> Optional[Tuple[int, float, int]]:
    best_exp = None
    best_wr = -1.0
    best_trades = 0
    for exp in allowed:
        b = bucket.get(str(exp), {})
        wins = int(b.get("wins", 0))
        losses = int(b.get("losses", 0))
        trades = wins + losses
        if trades <= 0:
            continue
        wr = wins / max(1, trades)
        if (wr > best_wr) or (wr == best_wr and trades > best_trades):
            best_exp = int(exp)
            best_wr = float(wr)
            best_trades = int(trades)
    if best_exp is None:
        return None
    return best_exp, best_wr, best_trades

def _choose_exp_for_trade(stats: Dict[str, Any], ativo: str, direcao: str) -> Tuple[int, str]:
    if EXP_FORCE in ("3", "5"):
        return int(EXP_FORCE), "force"
    if EXP_MODE == "FIXED_5":
        return 5, "fixed_5"
    if EXP_MODE == "FIXED_3":
        return 3, "fixed_3"

    allowed = EXP_ALLOWED if EXP_ALLOWED else [EXP_MIN, EXP_MAX]

    asset_bucket = _exp_bucket(stats, "asset", ativo, direcao)
    best_asset = _exp_pick_best(asset_bucket, allowed)
    if best_asset and best_asset[2] >= EXP_MIN_SAMPLES:
        return int(best_asset[0]), f"asset_wr={best_asset[1]:.2f} trades={best_asset[2]}"

    global_bucket = _exp_bucket(stats, "global", None, None)
    best_global = _exp_pick_best(global_bucket, allowed)
    if best_global and best_global[2] >= EXP_MIN_SAMPLES:
        return int(best_global[0]), f"global_wr={best_global[1]:.2f} trades={best_global[2]}"

    return int(max(allowed)), "default"

def _update_exp_stats(stats: Dict[str, Any], ativo: str, direcao: str, exp_min: int, pnl: float):
    if pnl == 0:
        return
    exp_key = str(int(exp_min))

    global_bucket = _exp_bucket(stats, "global", None, None)
    gb = global_bucket.setdefault(exp_key, {"wins": 0, "losses": 0})

    asset_bucket = _exp_bucket(stats, "asset", ativo, direcao)
    ab = asset_bucket.setdefault(exp_key, {"wins": 0, "losses": 0})

    if pnl > 0:
        gb["wins"] = int(gb.get("wins", 0)) + 1
        ab["wins"] = int(ab.get("wins", 0)) + 1
    else:
        gb["losses"] = int(gb.get("losses", 0)) + 1
        ab["losses"] = int(ab.get("losses", 0)) + 1

def _update_asset_stats(stats: Dict[str, Any], ativo: str, pnl: float) -> Dict[str, Any]:
    if pnl == 0:
        return stats.get("assets", {}).get(ativo, {})
    assets = stats.setdefault("assets", {})
    a = assets.setdefault(ativo, {"wins": 0, "losses": 0, "trades": 0})
    if pnl > 0:
        a["wins"] = int(a.get("wins", 0)) + 1
    else:
        a["losses"] = int(a.get("losses", 0)) + 1
    a["trades"] = int(a.get("wins", 0)) + int(a.get("losses", 0))
    return a

def _reset_ai_stats(path: str):
    """Limpa histórico da IA para não poluir o aprendizado."""
    _safe_save_json(path, {"meta": {"total": 0}, "arms": {}, "patterns": {}})

def _pretrain_ai(iq: Casa_Trader, ativos: List[str], stats: Dict[str, Any]):
    """Pré-treina a IA usando histórico recente (ex.: 900 velas por ativo)."""
    if not IA_ON or not AI_PRETRAIN_ON:
        return

    if AI_PRETRAIN_RESET:
        _reset_ai_stats(AI_STATS_FILE)
        log.info(paint("[IA] Histórico resetado para pré-treino", C.B))

    total_updates = 0
    for a in ativos:
        df = get_candles_df(iq, a, TF_M1, AI_PRETRAIN_CANDLES, end_ts=end_ts_closed(TF_M1))
        if df is None or len(df) < 50:
            continue

        start_idx = max(50, min(240, len(df) - 2))
        for i in range(start_idx, len(df) - 1):
            window = df.iloc[:i+1]
            atr_val = atr(window, 14)

            if MODO_SR_SIMPLES:
                tuner_params = auto_tuner.get_params() if AUTO_TUNER_AVAILABLE and auto_tuner else None
                setup = setup_sr_simples(window, atr_val, tuner_params)
            elif USE_CANDLE_MID:
                setup = strategy_candle_mid(window, atr_val)
            elif USE_M5_INDICATORS:
                setup = dom_forex_signal(window, atr_val)
            else:
                setup = dom_forex_signal(window, atr_val)

            if not setup.get("trade"):
                continue

            entry_price = float(window["close"].iloc[-1])
            next_close = float(df["close"].iloc[i+1])
            win = (setup["dir"] == "CALL" and next_close > entry_price) or (
                setup["dir"] == "PUT" and next_close < entry_price
            )
            pnl = 1.0 if win else -1.0
            ai_update(a, setup, pnl, stats)
            # Também adiciona ao LightGBM se disponível
            if LGBM_ON and lgb is not None:
                lgbm_add_sample_from_backtest(setup, win)
            total_updates += 1

            if total_updates >= AI_PRETRAIN_MAX_TRADES:
                break

        if total_updates >= AI_PRETRAIN_MAX_TRADES:
            break

    _safe_save_json(AI_STATS_FILE, stats)
    log.info(paint(f"[IA] Pré-treino concluído: {total_updates} updates", C.B))

    # Treina LGBM com dados do backtest
    if LGBM_ON and lgb is not None and len(lgbm_data) >= LGBM_MIN_SAMPLES:
        lgbm_train()

# ===================== SKLEARN ONLY (MODEL) =====================
_sk_model: Optional[RandomForestClassifier] = None
_sk_scaler: Optional[StandardScaler] = None
_sk_last_train_ts: Optional[float] = None

_cb_model: Optional[CatBoostClassifier] = None
_cb_last_train_ts: Optional[float] = None

def _sk_features_from_window(window: pd.DataFrame) -> List[float]:
    last = window.iloc[-1]
    prev = window.iloc[-2] if len(window) >= 2 else last
    atr_val = atr(window, 14)
    atr_safe = max(float(atr_val), 1e-9)

    body = float(last["close"] - last["open"])
    rng = float(last["high"] - last["low"])
    upper = float(last["high"] - max(last["open"], last["close"]))
    lower = float(min(last["open"], last["close"]) - last["low"])

    ret1 = (float(last["close"]) - float(prev["close"])) / max(1e-9, float(prev["close"]))

    closes = window["close"].to_numpy(float)
    if len(closes) >= 6:
        ret5 = (float(closes[-1]) - float(closes[-6])) / max(1e-9, float(closes[-6]))
    else:
        ret5 = 0.0

    if len(closes) >= 11:
        ret10 = (float(closes[-1]) - float(closes[-11])) / max(1e-9, float(closes[-11]))
        std10 = float(np.std(closes[-10:])) / atr_safe
    else:
        ret10 = 0.0
        std10 = 0.0

    if len(closes) >= 10:
        x = np.arange(10, dtype=float)
        y = closes[-10:]
        slope = float(np.polyfit(x, y, 1)[0]) / atr_safe
    else:
        slope = 0.0

    recent = window.tail(min(30, len(window)))
    dist_top = (float(recent["high"].max()) - float(last["close"])) / atr_safe
    dist_bot = (float(last["close"]) - float(recent["low"].min())) / atr_safe

    # TA-Lib features: apenas padrões de vela (SEM EMA/ADX)
    cl_arr = window["close"].to_numpy(float)
    hi_arr = window["high"].to_numpy(float)
    lo_arr = window["low"].to_numpy(float)
    op_arr = window["open"].to_numpy(float)

    # OS 13 PADRÕES MAIS CONFIÁVEIS (>80% eficácia)
    engulf = talib.CDLENGULFING(op_arr, hi_arr, lo_arr, cl_arr)       # 82%
    hamm = talib.CDLHAMMER(op_arr, hi_arr, lo_arr, cl_arr)            # 80%
    shoot = talib.CDLSHOOTINGSTAR(op_arr, hi_arr, lo_arr, cl_arr)     # 80%
    mstar = talib.CDLMORNINGSTAR(op_arr, hi_arr, lo_arr, cl_arr)      # 81%
    estar = talib.CDLEVENINGSTAR(op_arr, hi_arr, lo_arr, cl_arr)      # 81%
    tw = talib.CDL3WHITESOLDIERS(op_arr, hi_arr, lo_arr, cl_arr)      # 83%
    tb = talib.CDL3BLACKCROWS(op_arr, hi_arr, lo_arr, cl_arr)         # 84%
    pier = talib.CDLPIERCING(op_arr, hi_arr, lo_arr, cl_arr)          # 80%
    dc = talib.CDLDARKCLOUDCOVER(op_arr, hi_arr, lo_arr, cl_arr)      # 80%
    harami = talib.CDLHARAMI(op_arr, hi_arr, lo_arr, cl_arr)          # 80%
    hanging = talib.CDLHANGINGMAN(op_arr, hi_arr, lo_arr, cl_arr)     # 80%

    # Normaliza padrões: >0 = bullish, <0 = bearish, 0 = nada
    p_engulf = 1.0 if engulf[-1] > 0 else (-1.0 if engulf[-1] < 0 else 0.0)
    p_hammer = 1.0 if hamm[-1] > 0 else 0.0
    p_shoot = -1.0 if shoot[-1] != 0 else 0.0
    p_mstar = 1.0 if mstar[-1] > 0 else 0.0
    p_estar = -1.0 if estar[-1] != 0 else 0.0
    p_3w = 1.0 if tw[-1] > 0 else 0.0
    p_3b = -1.0 if tb[-1] != 0 else 0.0
    p_pier = 1.0 if pier[-1] > 0 else 0.0
    p_dc = -1.0 if dc[-1] != 0 else 0.0
    p_harami = 1.0 if harami[-1] > 0 else (-1.0 if harami[-1] < 0 else 0.0)
    p_hanging = -1.0 if hanging[-1] != 0 else 0.0

    return [
        body / atr_safe,
        rng / atr_safe,
        upper / atr_safe,
        lower / atr_safe,
        ret1,
        ret5,
        ret10,
        std10,
        slope,
        dist_top,
        dist_bot,
        # Padrões TA-Lib (13 mais confiáveis - SEM EMA/ADX)
        p_engulf,
        p_hammer,
        p_shoot,
        p_mstar,
        p_estar,
        p_3w,
        p_3b,
        p_pier,
        p_dc,
        p_harami,
        p_hanging,
    ]

def _sk_build_dataset(df: pd.DataFrame, require_setup_trade: bool = True) -> Tuple[List[List[float]], List[int]]:
    X: List[List[float]] = []
    y: List[int] = []
    if df is None or len(df) < 60:
        return X, y

    start_idx = max(30, min(120, len(df) - 2))
    for i in range(start_idx, len(df) - 1):
        window = df.iloc[:i+1]
        if require_setup_trade and (USE_SKLEARN_ONLY or USE_CATBOOST_ONLY):
            atr_val = atr(window, 14)
            tuner_params = auto_tuner.get_params() if AUTO_TUNER_AVAILABLE and auto_tuner else None
            setup = setup_sr_simples(window, atr_val, tuner_params)
            if not setup.get("trade"):
                continue
        feats = _sk_features_from_window(window)
        next_close = float(df["close"].iloc[i+1])
        cur_close = float(df["close"].iloc[i])
        label = 1 if next_close > cur_close else 0
        X.append(feats)
        y.append(label)
        if len(X) >= SKLEARN_TRAIN_MAX:
            break
    return X, y

def _train_sklearn_model(iq: Casa_Trader, ativos: List[str]):
    global _sk_model, _sk_scaler
    if not USE_SKLEARN_ONLY or not SKLEARN_AVAILABLE:
        return

    X_all: List[List[float]] = []
    y_all: List[int] = []

    for a in ativos:
        df = get_candles_df(iq, a, TF_M1, SKLEARN_TRAIN_CANDLES, end_ts=end_ts_closed(TF_M1))
        if df is None or len(df) < 60:
            continue
        X, y = _sk_build_dataset(df, require_setup_trade=True)
        X_all.extend(X)
        y_all.extend(y)

    if len(X_all) < 200:
        log.info(paint("[SKLEARN] Base insuficiente para treino", C.Y))
        return

    scaler = StandardScaler()
    Xs = scaler.fit_transform(np.asarray(X_all, dtype=float))
    model = RandomForestClassifier(
        n_estimators=240,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(Xs, np.asarray(y_all, dtype=int))

    _sk_model = model
    _sk_scaler = scaler
    global _sk_last_train_ts
    _sk_last_train_ts = time.time()
    sr_msg = " (S/R apenas)" if USE_SKLEARN_ONLY else ""
    log.info(paint(f"[SKLEARN] Modelo treinado: samples={len(X_all)}{sr_msg}", C.G))

def _sk_predict(window: pd.DataFrame) -> Tuple[str, float]:
    if _sk_model is None or _sk_scaler is None:
        return "NEUTRAL", 0.0
    feats = _sk_features_from_window(window)
    X = _sk_scaler.transform(np.asarray([feats], dtype=float))
    proba = _sk_model.predict_proba(X)[0]
    prob_call = float(proba[1])
    prob_put = float(proba[0])
    if prob_call >= prob_put:
        return "CALL", prob_call
    return "PUT", prob_put

def _train_catboost_model(iq: Casa_Trader, ativos: List[str]):
    global _cb_model, _cb_last_train_ts
    if not USE_CATBOOST_ONLY or not CATBOOST_AVAILABLE:
        return

    X_all: List[List[float]] = []
    y_all: List[int] = []

    for a in ativos:
        df = get_candles_df(iq, a, TF_M1, CATBOOST_TRAIN_CANDLES, end_ts=end_ts_closed(TF_M1))
        if df is None or len(df) < 60:
            continue
        X, y = _sk_build_dataset(df, require_setup_trade=False)
        X_all.extend(X)
        y_all.extend(y)
        if len(X_all) >= CATBOOST_TRAIN_MAX:
            break

    if len(X_all) < CATBOOST_MIN_SAMPLES:
        log.info(paint("[CATBOOST] Base insuficiente para treino", C.Y))
        return

    pos = sum(1 for v in y_all if v == 1)
    neg = len(y_all) - pos
    if pos == 0 or neg == 0:
        log.info(paint("[CATBOOST] Classe única, treino ignorado", C.Y))
        return

    class_weights = [1.0, float(neg) / float(pos)]

    model = CatBoostClassifier(
        iterations=CATBOOST_ITERS,
        depth=CATBOOST_DEPTH,
        learning_rate=CATBOOST_LR,
        loss_function="Logloss",
        eval_metric="AUC",
        class_weights=class_weights,
        verbose=False,
        random_seed=42,
    )
    model.fit(np.asarray(X_all, dtype=float), np.asarray(y_all, dtype=int))

    _cb_model = model
    _cb_last_train_ts = time.time()
    log.info(paint(f"[CATBOOST] Modelo treinado: samples={len(X_all)} (TA-Lib+Preço - SEM EMA/ADX)", C.G))

def _cb_predict(window: pd.DataFrame) -> Tuple[str, float]:
    if _cb_model is None:
        return "NEUTRAL", 0.0
    feats = _sk_features_from_window(window)
    proba = _cb_model.predict_proba(np.asarray([feats], dtype=float))[0]
    prob_call = float(proba[1])
    prob_put = float(proba[0])
    if prob_call >= prob_put:
        return "CALL", prob_call
    return "PUT", prob_put

# ===================== REDE NEURAL DE PUNIÇÃO (LOGISTIC) =====================
_punish_model: Optional[Dict[str, Any]] = None

def _punish_features(setup: Dict[str, Any]) -> List[float]:
    # features normalizadas (0..1)
    return [
        _clip(float(setup.get("score", 0.0)), 0.0, 1.0),
        _clip(float(setup.get("retr", 0.0)), 0.0, 1.0),
        _clip(float(setup.get("A_atr", 0.0)) / 6.0, 0.0, 1.0),
        _clip(float(setup.get("effA", 0.0)), 0.0, 1.0),
        _clip(float(setup.get("flips", 0.0)), 0.0, 1.0),
        _clip(float(setup.get("distBreak", 0.0)) / 2.0, 0.0, 1.0),
        _clip(float(setup.get("entry_confidence", 0.0)), 0.0, 1.0),
        _clip(float(setup.get("market_quality", 0.0)), 0.0, 1.0)
    ]

def _punish_get_model() -> Dict[str, Any]:
    global _punish_model
    if _punish_model is not None:
        return _punish_model
    n_features = 8
    model = {"w": [0.0] * n_features, "b": 0.0, "n": 0}
    try:
        if os.path.exists(PUNISH_MODEL_FILE):
            with open(PUNISH_MODEL_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict) and isinstance(loaded.get("w"), list):
                    model = loaded
    except Exception:
        pass
    _punish_model = model
    return model

def _punish_save_model(model: Dict[str, Any]):
    try:
        with open(PUNISH_MODEL_FILE, "w", encoding="utf-8") as f:
            json.dump(model, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))

def _punish_predict(setup: Dict[str, Any]) -> float:
    if not PUNISH_ON:
        return 0.0
    model = _punish_get_model()
    w = model.get("w", [])
    b = float(model.get("b", 0.0))
    x = _punish_features(setup)
    if len(w) != len(x):
        w = [0.0] * len(x)
        model["w"] = w
    z = b
    for wi, xi in zip(w, x):
        z += float(wi) * float(xi)
    return float(_sigmoid(z))

def _punish_update(setup: Dict[str, Any], is_loss: bool):
    if not PUNISH_ON:
        return
    model = _punish_get_model()
    w = model.get("w", [])
    b = float(model.get("b", 0.0))
    x = _punish_features(setup)
    if len(w) != len(x):
        w = [0.0] * len(x)
    y = 1.0 if is_loss else 0.0
    z = b
    for wi, xi in zip(w, x):
        z += float(wi) * float(xi)
    pred = _sigmoid(z)
    err = (pred - y)
    lr = max(0.001, PUNISH_LR)
    w = [float(wi) - lr * err * float(xi) for wi, xi in zip(w, x)]
    b = float(b) - lr * err
    model["w"] = w
    model["b"] = b
    model["n"] = int(model.get("n", 0)) + 1
    _punish_save_model(model)

def _append_loss_report(path: str, entry: Dict[str, Any], max_items: int = 500):
    try:
        data = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    data = loaded
                elif isinstance(loaded, dict) and isinstance(loaded.get("items"), list):
                    data = loaded["items"]
        data.append(entry)
        if len(data) > max_items:
            data = data[-max_items:]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"items": data}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _build_loss_report(
    iq: Casa_Trader,
    ativo: str,
    direction: str,
    stake: float,
    pnl: float,
    setup: Dict[str, Any],
    atr_val: float,
    score: float
) -> Dict[str, Any]:
    now = datetime.now().isoformat()
    report = {
        "timestamp": now,
        "ativo": ativo,
        "direction": direction,
        "stake": float(stake),
        "pnl": float(pnl),
        "exp_min": int(EXP_FIXA),
        "score": float(score),
        "atr": float(atr_val),
        "sr_type": setup.get("sr_type"),
        "sr_level": setup.get("sr_level"),
        "sr_toques": setup.get("sr_toques"),
        "sr_dist_atr": setup.get("sr_dist_atr"),
        "reasons": setup.get("reasons", [])
    }

    # Tenta capturar velas e gerar motivo
    try:
        tf_use = TF_M5 if USE_M5_INDICATORS else TF_M1
        n_use = max(60, N_M5) if USE_M5_INDICATORS else 60
        df_loss = get_candles_df(iq, ativo, tf_use, n_use, end_ts=time.time())
        if df_loss is not None and len(df_loss) >= 10:
            # Heurísticas simples
            last_10 = df_loss.tail(10)
            greens = sum(1 for _, r in last_10.iterrows() if r["close"] > r["open"])
            reds = 10 - greens
            trend = "bullish" if greens > reds else "bearish" if reds > greens else "neutral"

            streak_n = max(3, SR_STREAK_BLOCK_BARS)
            last_n = df_loss.tail(streak_n)
            up_n = sum(1 for _, r in last_n.iterrows() if r["close"] > r["open"])
            down_n = streak_n - up_n
            streak_ratio = max(up_n, down_n) / max(1, streak_n)

            entry_mismatch = (
                (direction == "CALL" and trend == "bearish") or
                (direction == "PUT" and trend == "bullish")
            )

            motivos = []
            if entry_mismatch:
                motivos.append("contra_tendencia_curta")
            if streak_ratio >= SR_STREAK_BLOCK_RATIO:
                motivos.append(f"sequencia_forte_{streak_ratio:.0%}")

            report.update({
                "market_trend_10": trend,
                "greens_10": int(greens),
                "reds_10": int(reds),
                "streak_ratio": float(streak_ratio),
                "motivos": motivos
            })

            # LossAnalyzer (se disponível) para contexto extra
            if LOSS_ANALYZER_AVAILABLE:
                analyzer = LossAnalyzer()
                mc = analyzer.analyze_market_context(df_loss)
                eq = analyzer.analyze_entry_quality(df_loss, direction)
                report["market_context"] = mc
                report["entry_quality"] = eq
                report["ai_summary"] = analyzer.generate_ai_analysis(mc, eq, ativo, direction, stake)

            # OpenAI (generativo) para motivo do loss
            ai_payload = {
                "ativo": ativo,
                "direction": direction,
                "score": float(score),
                "atr": float(atr_val),
                "sr_type": setup.get("sr_type"),
                "sr_level": setup.get("sr_level"),
                "sr_toques": setup.get("sr_toques"),
                "sr_dist_atr": setup.get("sr_dist_atr"),
                "reasons": setup.get("reasons", []),
                "market_trend_10": report.get("market_trend_10"),
                "greens_10": report.get("greens_10"),
                "reds_10": report.get("reds_10"),
                "streak_ratio": report.get("streak_ratio"),
                "motivos": report.get("motivos", []),
                "market_context": report.get("market_context", {}),
                "entry_quality": report.get("entry_quality", {})
            }
            ai_result = _analyze_loss_with_openai(ai_payload)
            report["ai_loss_reason"] = ai_result
    except Exception:
        pass

    return report

def _analyze_loss_with_openai(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not OPENAI_AVAILABLE or openai_client is None:
        return {"motivo": "openai_indisponivel", "sugestao": ""}

    try:
        prompt = (
            "Analise este LOSS em opções binárias e retorne JSON com: "
            "motivo (curto), causa_principal, sinais_negativos (lista), sugestao (curta).\n"
            f"DADOS: {json.dumps(payload, ensure_ascii=False)}"
        )

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "Você é um analista de trading. Responda APENAS em JSON válido."},
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content.strip()
        try:
            return json.loads(content)
        except Exception:
            return {"motivo": content, "sugestao": ""}
    except Exception as e:
        return {"motivo": f"erro_openai:{e}", "sugestao": ""}

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
    b_flip = _bucket(flips, 0.10, 0.0, 0.80)    # chopiness
    b_dist = _bucket(distBreak, 0.05, 0.0, 0.50) # distância da quebra

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
        punish_prob = _punish_predict(setup)
        prob_adj = _clip(float(bayes_mean) * (1.0 - PUNISH_SCALE * punish_prob), 0.0, 1.0)
        return {"prob": float(prob_adj), "bayes": float(bayes_mean), "ucb01": float(ucb01),
            "conf": float(conf), "n_arm": 0, "total": total, "key": key, "prior": prior,
            "punish": float(punish_prob)}

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

    punish_prob = _punish_predict(setup)
    prob_adj = _clip(prob * (1.0 - PUNISH_SCALE * punish_prob), 0.0, 1.0)

    return {"prob": float(prob_adj), "bayes": float(bayes_mean), "ucb01": float(ucb01),
            "conf": float(conf), "n_arm": n, "total": total, "key": key, "prior": prior,
            "punish": float(punish_prob)}

def ai_update(ativo: str, setup: Dict[str, Any], pnl: float, stats: Dict[str, Any]):
    """
    pnl > 0 => sucesso
    pnl < 0 => falha
    pnl = 0 => ignora
    """
    if pnl == 0:
        return

    key = ai_make_key(ativo, setup)
    arms = stats.setdefault("arms", {})
    meta = stats.setdefault("meta", {"total": 0})

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

    # Atualiza rede neural de punição
    _punish_update(setup, is_loss=(pnl < 0))

# ===================== LIGHTGBM - GRADIENT BOOSTING =====================

def lgbm_extract_features(setup: Dict[str, Any], df_m1: Optional[pd.DataFrame] = None) -> np.ndarray:
    """
    Extrai features numéricas do setup para o LightGBM.
    14 features: score, retr, A_atr, effA, flips, pb_len, distBreak,
                 late_ext, compression, market_quality, entry_conf, ctx_score, dir_enc, rsi_norm
    """
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

    ctx = str(setup.get("ctx", "neutro"))
    ctx_score = 1.0 if ctx == "bom" else (0.5 if ctx == "neutro" else 0.0)

    dir_str = str(setup.get("dir", "NEUTRAL"))
    dir_enc = 1.0 if dir_str == "CALL" else (-1.0 if dir_str == "PUT" else 0.0)

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

def lgbm_add_sample_from_backtest(setup: Dict[str, Any], win: bool):
    """
    Adiciona amostra do backtest ao LightGBM SOMENTE se for de qualidade.
    """
    global lgbm_data
    if not LGBM_ON:
        return

    score = setup.get("score", 0.0)
    ctx = setup.get("market_quality", 0.0)
    effA = setup.get("effA", 0.0)

    if score < 0.55 or ctx < 0.40 or effA < 0.25:
        return

    features = lgbm_extract_features(setup).tolist()
    label = 1 if win else 0

    lgbm_data.append({"features": features, "label": label, "source": "backtest", "timestamp": time.time()})

    if len(lgbm_data) > 1000:
        lgbm_data = lgbm_data[-1000:]

    lgbm_save_data()

def lgbm_train():
    """Treina ou retreina o modelo LightGBM com os dados acumulados."""
    global lgbm_model, lgbm_data, lgbm_reliable, lgbm_val_accuracy

    if not LGBM_ON or lgb is None or len(lgbm_data) < LGBM_MIN_SAMPLES:
        return

    try:
        # Limpeza de dados antigos (>12h)
        MAX_DATA_AGE_HOURS = 12
        cutoff_time = time.time() - (MAX_DATA_AGE_HOURS * 3600)

        old_count = len(lgbm_data)
        lgbm_data = [d for d in lgbm_data if d.get("timestamp", time.time()) >= cutoff_time]
        removed = old_count - len(lgbm_data)

        if removed > 0:
            log.info(paint(f"[LGBM] Limpeza: removidas {removed} amostras antigas (>{MAX_DATA_AGE_HOURS}h)", C.Y))
            lgbm_save_data()

        if len(lgbm_data) < LGBM_MIN_SAMPLES:
            log.warning(paint(f"[LGBM] Após limpeza, apenas {len(lgbm_data)} amostras (mín={LGBM_MIN_SAMPLES}). Modelo desabilitado.", C.Y))
            lgbm_reliable = False
            lgbm_val_accuracy = 0.0
            lgbm_model = None
            return

        # Migração de features
        for d in lgbm_data:
            if len(d["features"]) < LGBM_N_FEATURES:
                d["features"].extend([0.5] * (LGBM_N_FEATURES - len(d["features"])))
            elif len(d["features"]) > LGBM_N_FEATURES:
                d["features"] = d["features"][:LGBM_N_FEATURES]

        X = np.array([d["features"] for d in lgbm_data], dtype=np.float32)
        y = np.array([d["label"] for d in lgbm_data], dtype=np.int32)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 8,
            "max_depth": 3,
            "learning_rate": 0.03,
            "n_estimators": 80,
            "min_child_samples": 10,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.5,
            "reg_lambda": 0.5,
            "verbose": -1,
            "force_col_wise": True,
        }

        lgbm_model = lgb.LGBMClassifier(**params)

        n_samples = len(X)
        if n_samples >= 50:
            split_idx = int(n_samples * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            lgbm_model.fit(X_train, y_train)

            preds_train = lgbm_model.predict(X_train)
            acc_train = (preds_train == y_train).mean() * 100

            preds_val = lgbm_model.predict(X_val)
            acc_val = (preds_val == y_val).mean() * 100

            lgbm_val_accuracy = acc_val

            log.info(paint(f"[LGBM] Modelo treinado! Amostras={n_samples} | Treino={acc_train:.1f}% | Val={acc_val:.1f}% (real)", C.G))

            if acc_val < 50.0:
                lgbm_reliable = False
                log.warning(paint(f"[LGBM] ⚠️ Val={acc_val:.1f}% < 50% → LGBM DESABILITADO (usando só Bayes)", C.Y))
            else:
                lgbm_reliable = True
                log.info(paint(f"[LGBM] ✅ Val={acc_val:.1f}% ≥ 50% → LGBM CONFIÁVEL", C.G))

            if lgbm_reliable:
                lgbm_model.fit(X, y)
            else:
                lgbm_model.fit(X, y)
        else:
            lgbm_model.fit(X, y)
            preds = lgbm_model.predict(X)
            acc = (preds == y).mean() * 100
            lgbm_reliable = False
            lgbm_val_accuracy = 0.0
            log.info(paint(f"[LGBM] Modelo treinado! Amostras={n_samples} | Acc={acc:.1f}% (sem validação)", C.Y))

        lgbm_save_model()

    except Exception as e:
        log.warning(f"[LGBM] Erro no treino: {e}")
        lgbm_reliable = False
        lgbm_val_accuracy = 0.0

def lgbm_predict(setup: Dict[str, Any]) -> Tuple[float, bool]:
    """
    Prediz probabilidade de WIN usando LightGBM.
    Returns: (probabilidade, modelo_disponivel)
    """
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

# ===================== ENSEMBLE PREDICT (MODELO PROFISSIONAL) =====================
def ensemble_predict(ativo: str, setup: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combina predições do Bayesiano e LightGBM com GATEs inteligentes.
    Modelo profissional: prioriza ZONA (S/R + LT + score), não candle.
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
    
    # Se LightGBM não disponível ou não confiável, usa só Bayesiano
    if not lgbm_available or not LGBM_ON or not lgbm_reliable:
        reason_detail = "lgbm_off" if not LGBM_ON else ("lgbm_unreliable" if not lgbm_reliable else "lgbm_unavailable")
        
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
    
    # ── Ensemble ponderado Bayes + LGBM ──
    bayes_weight = min(1.0, n_arm / AI_MIN_SAMPLES) * 0.5 + 0.25
    lgbm_weight = 1.0 - bayes_weight
    ensemble_prob = bayes_prob * bayes_weight + lgbm_prob * lgbm_weight
    
    # Decisão baseada no modo
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
    else:  # weighted
        min_ens_weighted = max(AI_MIN_PROB, 0.58)
        should_trade = ensemble_prob >= min_ens_weighted
        reason = f"weighted(ens={ensemble_prob:.2f},min={min_ens_weighted:.2f})"
    
    # ── WARMUP COM LGBM ──
    if n_arm < AI_MIN_SAMPLES:
        warmup_threshold = LGBM_WARMUP_PROB if IA_MODE == "learning" else LGBM_MIN_PROB
        
        # Threshold dinâmico baseado no contexto
        if ctx_val < 0.40:
            min_ens = ENS_MIN_CTX_RUIM
        elif ctx_val < 0.50:
            min_ens = ENS_MIN_CTX_MED
        else:
            min_ens = ENS_MIN_CTX_BOM
        
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
    
    # ══════════ GATEs PROFISSIONAIS ══════════
    if should_trade:
        # GATE 1: Contexto ruim BLOQUEIA a menos que SR forte + ensemble alto
        if ctx_val < 0.40 and not ((sr_forte or sr_basico) and ensemble_prob >= 0.58):
            should_trade = False
            reason = f"G1_ctx_ruim(ctx={ctx_val:.2f},ens={ensemble_prob:.2f})"
        
        # GATE 2: Contexto mediano + sem zona confirmada = precisa score alto
        elif ctx_val < 0.50 and not sr_forte and not sr_basico and not has_lt and sc < 0.55:
            should_trade = False
            reason = f"G2_ctx_med_sem_zona(ctx={ctx_val:.2f},sc={sc:.2f})"
        
        # GATE 3: Trendline fraca sem S/R forte = não opera
        elif lt_conf < 0.5 and not has_lt and sr_prox < 0.40:
            should_trade = False
            reason = f"G3_sem_tendencia(lt={lt_conf:.2f},sr={sr_prox:.2f})"
        
        # GATE 4: Foco na ZONA — só bloqueia se TUDO é fraco
        ens_gate_thr = 0.52 if (not lgbm_reliable or not LGBM_ON) else 0.60
        if entry_conf_val < 0.30 and not zona_forte and not setup_forte and ensemble_prob < ens_gate_thr:
            should_trade = False
            reason = f"G4_tudo_fraco(ec={entry_conf_val:.2f},zona={zona_forte},sc={sc:.2f},ens={ensemble_prob:.2f})"
    
    return {
        "should_trade": should_trade,
        "bayes_prob": bayes_prob,
        "lgbm_prob": lgbm_prob,
        "ensemble_prob": ensemble_prob,
        "reason": reason,
        "bayes_conf": bayes_conf,
        "n_arm": n_arm
    }

# ===================== PERNADA B =====================
def pernada_b(df_m1: pd.DataFrame, atr_val: float, debug: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    if len(df_m1) < 240:
        _debug_reject(debug, "poucas_velas")
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["poucas_velas"], "debug": debug}

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
        _debug_reject(debug, "gatilho_fraco")
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
                "reasons": [f"gatilho_fraco(body={q['body_frac']:.2f})"], "debug": debug}

    # corredor de SR perto do preço => evita operar
    ping = sr_pingpong_zone(df_m1, atr_val)
    if ping:
        _debug_reject(debug, "sr_pingpong")
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": [ping], "debug": debug}

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
                _debug_reject(debug, "impulso_pequeno")
                continue

            start = float(imp["open"].iloc[0])
            end = float(imp["close"].iloc[-1])
            move = end - start

            # CORREÇÃO: A direção da PERNADA A define a tendência principal
            # PUT = impulso de queda (A vai pra baixo)
            # CALL = impulso de alta (A vai pra cima)
            dir_impulso_A = "PUT" if move < 0 else ("CALL" if move > 0 else "NEUTRAL")
            if dir_impulso_A == "NEUTRAL":
                _debug_reject(debug, "impulso_neutro")
                continue

            eff_A = leg_efficiency(imp)
            if eff_A < MIN_EFF_A:
                _debug_reject(debug, "effA_baixa")
                continue

            # ===================== VERIFICAÇÃO CRÍTICA: CONTEXTO MAIOR =====================
            # Verifica se existe um movimento MAIOR na direção OPOSTA ANTES do impulso
            # Se existir, o impulso identificado é apenas uma CORREÇÃO, não a tendência principal
            impulse_start_idx = len(df_m1) - (pb_len + 1 + w)
            if impulse_start_idx > 30:  # Precisa ter pelo menos 30 velas antes
                # Analisa as 30 velas ANTES do impulso
                pre_impulse = df_m1.iloc[impulse_start_idx - 30:impulse_start_idx]
                if len(pre_impulse) >= 20:
                    pre_high = float(pre_impulse["high"].max())
                    pre_low = float(pre_impulse["low"].min())
                    pre_start = float(pre_impulse["open"].iloc[0])
                    pre_end = float(pre_impulse["close"].iloc[-1])
                    pre_move = pre_end - pre_start
                    pre_size = pre_high - pre_low

                    # Se o movimento ANTES do impulso é MAIOR e na direção OPOSTA
                    # então o impulso identificado é apenas uma correção
                    if dir_impulso_A == "CALL":
                        # Impulso de ALTA identificado - verifica se vem de uma QUEDA maior
                        if pre_move < 0 and pre_size > size_A * 2.0:
                            # Movimento de queda ANTES é maior que o impulso de alta
                            # Isso significa que o impulso de alta é apenas uma CORREÇÃO
                            _debug_reject(debug, "correcao_pre_impulso")
                            continue
                    else:  # dir_impulso_A == "PUT"
                        # Impulso de QUEDA identificado - verifica se vem de uma ALTA maior
                        if pre_move > 0 and pre_size > size_A * 2.0:
                            # Movimento de alta ANTES é maior que o impulso de queda
                            # Isso significa que o impulso de queda é apenas uma CORREÇÃO
                            _debug_reject(debug, "correcao_pre_impulso")
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

            # Pullback precisa ter velas contra a pernada A (reduzido para 30%)
            if contra < max(1, int(math.ceil(pb_len * 0.30))):
                _debug_reject(debug, "pullback_contra_fraco")
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
                _debug_reject(debug, "retr_fora")
                continue

            c1 = float(decision["close"])

            # ENTRADA é na direção da PERNADA A (continuação da tendência)
            # Se A foi PUT (queda), entramos PUT quando rompe o pullback pra baixo
            # Se A foi CALL (alta), entramos CALL quando rompe o pullback pra cima
            dir_entrada = dir_impulso_A

            # bloqueio SR forte (múltiplas regiões)
            blk_sr = sr_block_directional_multi(df_m1, atr_val, dir_entrada)
            if blk_sr:
                _debug_reject(debug, "sr_block")
                continue

            # Calcula os extremos do pullback (necessário para ambas direções)
            pb_high = float(pb["high"].max())
            pb_low = float(pb["low"].min())

            # LÓGICA DE ROMPIMENTO CORRIGIDA
            if dir_entrada == "CALL":
                # CALL: Impulso A foi de ALTA, pullback desceu, agora rompe pullback pra CIMA
                if not (c1 > pb_low + BREAK_MARGIN_ATR * atr_val):
                    _debug_reject(debug, "break_fail_call")
                    continue
                if q["upper_frac"] > MAX_WICK_AGAINST:
                    _debug_reject(debug, "wick_contra_call")
                    continue

                dist = (c1 - pb_low) / max(atr_val, 1e-9)
                if dist > MAX_BREAK_DISTANCE_ATR:
                    _debug_reject(debug, "dist_break_call")
                    continue

                # ===================== VERIFICAÇÃO DE TOPO =====================
                # Não entra CALL se preço está muito perto do TOPO (máxima das últimas 60 velas)
                max_60 = float(df_m1.tail(60)["high"].max())
                dist_topo = (max_60 - c1) / max(atr_val, 1e-9)
                if dist_topo < 1.0:  # Menos de 1.0 ATR do topo = muito perto!
                    _debug_reject(debug, "perto_topo")
                    continue

                # Verifica se o preço atual está ABAIXO do topo do impulso A
                # Se c1 >= top, estamos no topo ou acima - NÃO ENTRAR
                if c1 >= top * 0.998:  # Preço muito perto do topo do impulso = NÃO ENTRAR
                    _debug_reject(debug, "topo_impulso")
                    continue

                # Verifica se o impulso já está "cansado" (muitas velas desde o início)
                if w + pb_len > 10:  # Mais de 10 velas desde o início do impulso
                    # Verifica se as últimas velas perderam força
                    last_5 = df_m1.tail(5)
                    bullish_count = sum(1 for _, row in last_5.iterrows() if float(row["close"]) > float(row["open"]))
                    if bullish_count < 3:  # Menos de 3 velas de alta nas últimas 5 = perda de força
                        _debug_reject(debug, "forca_fraca_call")
                        continue

                # VALIDAÇÃO DE QUALIDADE DA ENTRADA (NOVO)
                entry_validation = validate_entry_quality(df_m1, atr_val, "CALL", c1, pb_high, pb_low)
                if not entry_validation.get("valid", False):
                    _debug_reject(debug, "entry_invalid_call")
                    continue  # Pula entrada se não passar na validação

                entry_confidence = float(entry_validation.get("confidence", 0.0))
                risk_atr = float(entry_validation.get("risk_atr", 0.0))
                entry_momentum = float(entry_validation.get("momentum", 0.0))
                entry_alignment = float(entry_validation.get("alignment", 0.0))

                # CONFLUÊNCIA COM LINHA DE TENDÊNCIA (LTA para CALL)
                lt_conf = check_trendline_confluence(df_m1, pb_high, pb_low, "CALL", atr_val)
                lt_confluence = float(lt_conf.get("confluence", 0.0))
                has_lt = lt_conf.get("has_trendline", False)

                # Score base com CONFLUÊNCIA INTELIGENTE (CALL)
                score = 0.50  # base reduzida, pois agora temos validação de entrada
                score += min(0.18, (size_A / atr_val - IMPULSO_MIN_ATR) * 0.10)
                score += min(0.12, (eff_A - MIN_EFF_A) * 0.45)
                score += min(0.12, (RETR_MAX - retr) * 0.16)
                score += 0.03 if pb_len >= 2 else 0.00
                score -= min(0.10, max(0.0, (flips_frac - 0.30) * 0.25))

                # BÔNUS POR QUALIDADE DE CONTEXTO
                score += market_quality * 0.12

                # BÔNUS POR FORÇA DA TENDÊNCIA
                score += trend_strength * 0.06

                # BÔNUS POR QUALIDADE DA ENTRADA (NOVO - IMPORTANTE)
                score += entry_confidence * 0.20  # até +20% pela confiança da entrada
                score += entry_momentum * 0.08    # até +8% pelo momentum
                score += entry_alignment * 0.06   # até +6% pelo alinhamento

                # ⭐ BÔNUS POR CONFLUÊNCIA COM LINHA DE TENDÊNCIA (LTA) - MUITO IMPORTANTE!
                if has_lt and lt_confluence > 0.8:
                    score += 0.25  # BÔNUS GRANDE se tocou perfeitamente a LTA
                elif has_lt and lt_confluence > 0.5:
                    score += 0.15  # Bônus médio se próximo da LTA
                elif has_lt and lt_confluence > 0.2:
                    score += 0.05  # Bônus pequeno

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
                    _debug_reject(debug, "break_fail_put")
                    continue
                if q["lower_frac"] > MAX_WICK_AGAINST:
                    _debug_reject(debug, "wick_contra_put")
                    continue

                dist = (pb_high - c1) / max(atr_val, 1e-9)
                if dist > MAX_BREAK_DISTANCE_ATR:
                    _debug_reject(debug, "dist_break_put")
                    continue

                # ===================== VERIFICAÇÃO DE FUNDO =====================
                # Não entra PUT se preço está muito perto do FUNDO (mínima das últimas 60 velas)
                min_60 = float(df_m1.tail(60)["low"].min())
                dist_fundo = (c1 - min_60) / max(atr_val, 1e-9)
                if dist_fundo < 1.0:  # Menos de 1.0 ATR do fundo = muito perto!
                    _debug_reject(debug, "perto_fundo")
                    continue

                # Verifica se o preço atual está ACIMA do fundo do impulso A
                # Se c1 <= bot, estamos no fundo ou abaixo - NÃO ENTRAR
                if c1 <= bot * 1.002:  # Preço muito perto do fundo do impulso = NÃO ENTRAR
                    _debug_reject(debug, "fundo_impulso")
                    continue

                # Verifica se o impulso já está "cansado" (muitas velas desde o início)
                if w + pb_len > 10:  # Mais de 10 velas desde o início do impulso
                    # Verifica se as últimas velas perderam força
                    last_5 = df_m1.tail(5)
                    bearish_count = sum(1 for _, row in last_5.iterrows() if float(row["close"]) < float(row["open"]))
                    if bearish_count < 3:  # Menos de 3 velas de queda nas últimas 5 = perda de força
                        _debug_reject(debug, "forca_fraca_put")
                        continue

                # VALIDAÇÃO DE QUALIDADE DA ENTRADA (NOVO)
                entry_validation = validate_entry_quality(df_m1, atr_val, "PUT", c1, pb_high, pb_low)
                if not entry_validation.get("valid", False):
                    _debug_reject(debug, "entry_invalid_put")
                    continue  # Pula entrada se não passar na validação

                entry_confidence = float(entry_validation.get("confidence", 0.0))
                risk_atr = float(entry_validation.get("risk_atr", 0.0))
                entry_momentum = float(entry_validation.get("momentum", 0.0))
                entry_alignment = float(entry_validation.get("alignment", 0.0))

                # CONFLUÊNCIA COM LINHA DE TENDÊNCIA (LTB para PUT)
                lt_conf = check_trendline_confluence(df_m1, pb_high, pb_low, "PUT", atr_val)
                lt_confluence = float(lt_conf.get("confluence", 0.0))
                has_lt = lt_conf.get("has_trendline", False)

                # Score base com CONFLUÊNCIA INTELIGENTE (PUT)
                score = 0.50  # base reduzida, pois agora temos validação de entrada
                score += min(0.18, (size_A / atr_val - IMPULSO_MIN_ATR) * 0.10)
                score += min(0.12, (eff_A - MIN_EFF_A) * 0.45)
                score += min(0.12, (RETR_MAX - retr) * 0.16)
                score += 0.03 if pb_len >= 2 else 0.00
                score -= min(0.10, max(0.0, (flips_frac - 0.30) * 0.25))

                # BÔNUS POR QUALIDADE DE CONTEXTO
                score += market_quality * 0.12

                # BÔNUS POR FORÇA DA TENDÊNCIA
                score += trend_strength * 0.06

                # BÔNUS POR QUALIDADE DA ENTRADA (NOVO - IMPORTANTE)
                score += entry_confidence * 0.20  # até +20% pela confiança da entrada
                score += entry_momentum * 0.08    # até +8% pelo momentum
                score += entry_alignment * 0.06   # até +6% pelo alinhamento

                # ⭐ BÔNUS POR CONFLUÊNCIA COM LINHA DE TENDÊNCIA (LTB) - MUITO IMPORTANTE!
                if has_lt and lt_confluence > 0.8:
                    score += 0.25  # BÔNUS GRANDE se tocou perfeitamente a LTB
                elif has_lt and lt_confluence > 0.5:
                    score += 0.15  # Bônus médio se próximo da LTB
                elif has_lt and lt_confluence > 0.2:
                    score += 0.05  # Bônus pequeno

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
        _debug_reject(debug, "sem_pernadaB_valida")
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["sem_pernadaB_valida"], "debug": debug}

    # bloqueio final SR forte no momento do sinal
    block_final = sr_block_directional_multi(df_m1, atr_val, best["dir"])
    if block_final:
        _debug_reject(debug, "sr_block_final")
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": [block_final], "debug": debug}

    # ===================== FILTROS DE QUALIDADE INTELIGENTES (NOVOS) =====================
    ctx_quality = float(best.get("market_quality", 0.0))
    entry_conf = float(best.get("entry_confidence", 0.0))
    confl_bonus = float(best.get("confluence_bonus", 0.0))
    lt_conf = float(best.get("lt_confluence", 0.0))
    has_lt = best.get("has_lt", False)

    # 1. BLOQUEAR APENAS CONTEXTO EXTREMAMENTE RUIM
    if ctx_quality < MIN_CONTEXT_QUALITY:
        _debug_reject(debug, "contexto_extremo")
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
            "reasons": [f"contexto_extremamente_ruim(quality={ctx_quality:.2f}<{MIN_CONTEXT_QUALITY})"], "debug": debug}

    # 2. BLOQUEAR APENAS ENTRADA MUITO FRACA
    if entry_conf < MIN_ENTRY_CONFIDENCE:
        _debug_reject(debug, "entrada_fraca")
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
            "reasons": [f"entrada_muito_fraca(conf={entry_conf:.2f}<{MIN_ENTRY_CONFIDENCE})"], "debug": debug}

    # 3. BLOQUEAR COMBINAÇÃO PERIGOSA: contexto ruim + entrada fraca + sem confluência
    # Este é o padrão que causou os 3 losses: ctx=0.33-0.39, entry=0.44, conf=0.00
    if ctx_quality < 0.40 and entry_conf < 0.48 and confl_bonus < 0.02:
        # Permite APENAS se tiver linha de tendência muito forte (LT > 0.6)
        if not (has_lt and lt_conf > 0.6):
            return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
                    "reasons": [f"⚠️tudo_ruim(ctx={ctx_quality:.2f},entry={entry_conf:.2f},confl={confl_bonus:.2f},sem_LT_forte)"], "debug": debug}

    # 4. BLOQUEAR SCORE INFLADO SEM QUALIDADE (padrão dos losses: score=1.0 mas tudo ruim)
    if best["score"] > 0.85:
        # Se score alto mas TUDO indica problema, é entrada falsa
        if ctx_quality < 0.35 and entry_conf < 0.46 and confl_bonus == 0.0:
            if not (has_lt and lt_conf > 0.7):
                return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
                        "reasons": [f"⚠️score_inflado_setup_ruim(score={best['score']:.2f},ctx={ctx_quality:.2f},entry={entry_conf:.2f})"], "debug": debug}

    if debug is not None:
        best["debug"] = debug
    return best

def fallback_simple_setup(df_m1: pd.DataFrame, atr_val: float, m5_dir: str, m5_strength: float) -> Dict[str, Any]:
    if len(df_m1) < 60:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["fb_poucas_velas"]}

    closes = df_m1["close"]
    ema_fast = closes.ewm(span=10, adjust=False).mean()
    ema_slow = closes.ewm(span=30, adjust=False).mean()

    slope = float(ema_fast.iloc[-1] - ema_fast.iloc[-5])

    if ema_fast.iloc[-1] > ema_slow.iloc[-1] and slope > 0:
        trend_dir = "CALL"
    elif ema_fast.iloc[-1] < ema_slow.iloc[-1] and slope < 0:
        trend_dir = "PUT"
    else:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["fb_sem_tendencia"]}

    last = df_m1.iloc[-1]
    q = wick_fractions(last)
    if q["body_frac"] < MIN_BODY_FRAC_BREAK:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["fb_gatilho_fraco"]}

    last_5 = df_m1.tail(5)
    if trend_dir == "CALL":
        dir_count = sum(1 for _, row in last_5.iterrows() if float(row["close"]) > float(row["open"]))
    else:
        dir_count = sum(1 for _, row in last_5.iterrows() if float(row["close"]) < float(row["open"]))

    if dir_count < 3:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["fb_momentum_fraco"]}

    if m5_dir != "NEUTRAL" and m5_strength >= M5_ALIGN_MIN_STRENGTH:
        if (trend_dir == "CALL" and m5_dir != "BULLISH") or (trend_dir == "PUT" and m5_dir != "BEARISH"):
            return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": [f"fb_conflito_m5({m5_dir}:{m5_strength:.2f})"]}

    score = 0.42
    score += min(0.15, abs(slope) / max(atr_val, 1e-9)) * 0.10
    score += min(0.20, (dir_count / 5.0)) * 0.10
    score += min(0.20, float(m5_strength)) * 0.10
    score = float(max(0.0, min(1.0, score)))

    if score < FALLBACK_MIN_SCORE:
        return {"trade": False, "dir": "NEUTRAL", "score": score, "reasons": ["fb_score_baixo"]}

    return {
        "trade": True,
        "dir": trend_dir,
        "score": score,
        "reasons": [
            "fallback_simple",
            f"ema_slope={slope:.6f}",
            f"mom={dir_count}/5",
            f"m5={m5_dir}({m5_strength:.2f})"
        ]
    }

def simple_trend_setup(df_m1: pd.DataFrame, df_m5: Optional[pd.DataFrame], atr_val: float) -> Dict[str, Any]:
    if df_m5 is None or len(df_m5) < 30 or len(df_m1) < 20:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["simple_sem_dados"]}

    closes_m5 = df_m5["close"]
    ema_fast_m5 = closes_m5.ewm(span=10, adjust=False).mean()
    ema_slow_m5 = closes_m5.ewm(span=30, adjust=False).mean()

    if ema_fast_m5.iloc[-1] > ema_slow_m5.iloc[-1]:
        trend_dir = "CALL"
    elif ema_fast_m5.iloc[-1] < ema_slow_m5.iloc[-1]:
        trend_dir = "PUT"
    else:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["simple_sem_tendencia"]}

    last_3 = df_m1.tail(3)
    if trend_dir == "CALL":
        conf = sum(1 for _, r in last_3.iterrows() if float(r["close"]) > float(r["open"]))
    else:
        conf = sum(1 for _, r in last_3.iterrows() if float(r["close"]) < float(r["open"]))

    if conf < 2:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["simple_sem_confirmacao"]}

    # Padrão de velas (últimas 2)
    p_dir, p_name = candle_pattern_signal(df_m1)
    if p_dir != "NEUTRAL" and p_dir != trend_dir:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": [f"padrao_contra({p_name})"]}

    # Regiões simples (suporte/resistência recente)
    last_n = df_m1.tail(30)
    support = float(last_n["low"].min())
    resist = float(last_n["high"].max())
    price = float(df_m1["close"].iloc[-1])
    if trend_dir == "CALL":
        if (price - support) / max(atr_val, 1e-9) > 1.2:
            return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["longe_suporte"]}
    else:
        if (resist - price) / max(atr_val, 1e-9) > 1.2:
            return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["longe_resistencia"]}

    return {
        "trade": True,
        "dir": trend_dir,
        "score": 0.60,
        "reasons": ["simple_trend", f"m5_dir={trend_dir}", f"m1_conf={conf}/3", f"padrao={p_name}"]
    }

# ===================== PRÉ-BUSCA PARALELA DE DADOS (OTIMIZAÇÃO) =====================
def _prefetch_candles_parallel(iq: Casa_Trader, ativos: List[str], tf: int, n_candles: int) -> Dict[str, pd.DataFrame]:
    """
    Busca dados de vários ativos em paralelo para acelerar a análise.
    Retorna um dict {ativo: DataFrame} com os dados pré-carregados.
    """
    result = {}
    end_ts = end_ts_closed(tf)
    
    def fetch_one(ativo: str) -> Tuple[str, Optional[pd.DataFrame]]:
        try:
            df = get_candles_df(iq, ativo, tf, n_candles, end_ts=end_ts)
            return (ativo, df)
        except Exception:
            return (ativo, None)
    
    # Usa no máximo 5 threads para não sobrecarregar a API
    max_workers = min(5, len(ativos))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, a): a for a in ativos}
        for future in as_completed(futures):
            ativo, df = future.result()
            if df is not None:
                result[ativo] = df
    
    return result

# ===================== ESCOLHER MELHOR SETUP DO MINUTO =====================
def escolher_melhor_setup(iq: Casa_Trader, ativos: List[str]):
    best_trade = None
    best_any = None
    ranked_any: List[Tuple[float, str, str]] = []
    
    # Filtra ativos em cooldown ANTES da busca paralela
    ativos_disponiveis = []
    for a in ativos:
        if a in cooldown and (time.time() - cooldown[a]) < COOLDOWN_ATIVO:
            continue
        if a in cooldown_spike and (time.time() - cooldown_spike[a]) < (SPIKE_COOLDOWN_MIN * 60):
            continue
        ativos_disponiveis.append(a)
    
    if not ativos_disponiveis:
        return None, None, []
    
    # Busca dados em paralelo (muito mais rápido!)
    tf_use = TF_M5 if USE_M5_INDICATORS else TF_M1
    n_use = max(N_M5, 240) if USE_M5_INDICATORS else N_M1
    dados_ativos = _prefetch_candles_parallel(iq, ativos_disponiveis, tf_use, n_use)

    for a in ativos_disponiveis:
        df = dados_ativos.get(a)
        if df is None:
            continue

        atr_val = atr(df, 14)
        last_closed = df.iloc[-1]

        m5_dir, m5_strength = "NEUTRAL", 0.0
        # Só coleta M5 se NÃO estiver no modo simplificado
        if not MODO_SR_SIMPLES:
            if (M5_ALIGN_BLOCK or FALLBACK_SIMPLE) and REGIME_FILTER_AVAILABLE and regime_filter:
                df_m5 = get_candles_m5(iq, a, N_M5)
                if df_m5 is not None:
                    m5_dir, m5_strength = regime_filter.get_m5_direction(df_m5)
                    m5_color = C.G if m5_dir == "BULLISH" else (C.R if m5_dir == "BEARISH" else C.Y)
                    log.debug(paint(f"[M5 SYNC] {a} | {m5_dir} | Força: {m5_strength:.2f}", m5_color))

        if is_spike_wicky(last_closed, atr_val):
            cooldown_spike[a] = time.time()
            continue

        dbg = {} if DEBUG_REASONS else None

        # ===== MODO ESTRUTURA HIBRIDA (M5 + M1 + IA) =====
        if USE_STRUCT_HYBRID:
            if not CORE_STRUCT_AVAILABLE:
                setup = {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["struct_indisponivel"]}
            else:
                df_m5 = get_candles_m5(iq, a, N_M5)
                zones = build_zones(df_m5, max_zones=STRUCT_M5_MAX_ZONES, tol_mult=STRUCT_M5_TOL_MULT) if df_m5 is not None else []
                p_up = rf_proxy_predict_p_up(df)
                cfg_struct = {
                    "min_act_z": STRUCT_MIN_ACT_Z,
                    "max_chop": STRUCT_MAX_CHOP,
                    "max_wick_ratio": STRUCT_MAX_WICK_RATIO,
                    "min_wick_ratio": STRUCT_MIN_WICK_RATIO,
                    "min_zone_touches": STRUCT_MIN_ZONE_TOUCHES,
                    "min_conf": STRUCT_MIN_CONF,
                    "range_ma_window": STRUCT_RANGE_MA_WINDOW,
                    "chop_window": STRUCT_CHOP_WINDOW,
                    "min_range_ratio": STRUCT_MIN_RANGE_RATIO,
                    "max_zone_dist_atr": STRUCT_MAX_ZONE_DIST_ATR,
                }

                if p_up is None:
                    setup = {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["ia_sem_prob"]}
                else:
                    dec = decide_m1(df, zones, p_up, cfg_struct)
                    if dec and dec.get("ok"):
                        side = dec.get("side")
                        dir_trade = "CALL" if side == "BUY" else "PUT"
                        conf = float(dec.get("conf", 0.0))
                        setup = {
                            "trade": True,
                            "dir": dir_trade,
                            "score": conf,
                            "reasons": [
                                f"hybrid({dec.get('why')})",
                                f"zone={dec.get('zone')}",
                                f"touches={dec.get('touches')}",
                                f"dist={dec.get('dist_zone', 0.0):.5f}",
                                f"act_z={dec.get('act_z', 0.0):.2f}",
                                f"chop={dec.get('chop', 0.0):.2f}",
                                f"wick={dec.get('wick', 0.0):.2f}",
                                f"range%={dec.get('range_pct', 0.0):.5f}",
                                f"p_up={p_up:.2f}"
                            ]
                        }
                    else:
                        why = "sem_setup" if not dec else str(dec.get("why", "sem_setup"))
                        score = float(dec.get("conf", 0.0)) if dec else 0.0
                        setup = {"trade": False, "dir": "NEUTRAL", "score": score, "reasons": [why]}
        # ===== MODO IA RF PROXY (M1) =====
        elif USE_RF_PROXY_ONLY:
            setup = setup_rf_proxy(df, prob_threshold=RF_PROXY_MIN_PROB, min_rows=RF_PROXY_MIN_ROWS)
            setup["rf_proxy"] = True
        # ===== MODO CANDLE MID (M1) =====
        elif USE_CANDLE_MID:
            setup = strategy_candle_mid(df, atr_val)
        # ===== MODO M5 PERNADA B =====
        elif USE_M5_INDICATORS:
            setup = dom_forex_signal(df, atr_val)
            setup["m5_mode"] = True
        # ===== MODO PADRÃO DE VELAS TA-LIB + IA =====
        elif MODO_SR_SIMPLES:
            tuner_params = auto_tuner.get_params() if AUTO_TUNER_AVAILABLE and auto_tuner else None
            setup = setup_trend_candle(df, atr_val, tuner_params)
            if setup.get("trade"):
                # Mostra timestamp da vela analisada para debug
                last_candle_time = pd.to_datetime(df.iloc[-1].name, unit='s') if hasattr(df.iloc[-1], 'name') else "?"
                log.info(paint(
                    f"[PADRÃO-VELA] {a} | {setup['dir']} | score={setup.get('score',0):.2f} | vela={last_candle_time} | {setup.get('reasons',[''])[0]}",
                    C.G if setup['dir'] == 'CALL' else C.R
                ))
        elif SIMPLE_TREND_MODE:
            setup = simple_trend_setup(df, df_m5, atr_val)
        else:
            setup = dom_forex_signal(df, atr_val)
            if (not setup.get("trade")) and FALLBACK_SIMPLE:
                fb = fallback_simple_setup(df, atr_val, m5_dir, m5_strength)
                if fb.get("trade"):
                    setup = fb
                    setup["fallback"] = "simple"

        sc_any = float(setup.get("score", 0.0))
        cand_any = (sc_any, a, setup, float(atr_val))
        ranked_any.append((float(sc_any), a, str(setup.get("dir", "NEUTRAL"))))
        if best_any is None or cand_any[0] > best_any[0]:
            best_any = cand_any

        if setup.get("trade"):
            # ===== MODO IA RF PROXY / HIBRIDO: sem consenso =====
            if USE_RF_PROXY_ONLY or USE_STRUCT_HYBRID:
                cnn_confirma = True
            # ===== MODO SIMPLIFICADO: CONSENSO DE 6 AGENTES =====
            elif MODO_SR_SIMPLES:
                # Sistema de consenso: 6 agentes (5+ precisam concordar)
                cnn_confirma = False  # Default: precisa do consenso

                if (USE_SKLEARN_ONLY or USE_CATBOOST_ONLY) and not (CONSENSO_AVAILABLE and sistema_consenso):
                    # ML só bypassa consenso se consenso não estiver disponível
                    cnn_confirma = True
                    log.info(paint(f"[ML] {a} | consenso indisponível, usando apenas ML", C.B))
                elif CONSENSO_AVAILABLE and sistema_consenso:
                    try:
                        # Prepara níveis S/R para o sistema de consenso
                        res_levels, sup_levels = strong_sr_levels_last200(df, atr_val)
                        sr_levels_dict = {
                            "resistencias": res_levels,
                            "suportes": sup_levels
                        }

                        # Analisa com os 6 agentes
                        resultado_consenso = sistema_consenso.analisar(df, atr_val, sr_levels=sr_levels_dict)

                        if resultado_consenso["trade"]:
                            # Consenso atingido!
                            direcao_consenso = resultado_consenso["direcao"]
                            confianca = resultado_consenso["confianca_media"]

                            # Verifica se a direção do consenso bate com o setup S/R
                            if direcao_consenso == setup["dir"]:
                                cnn_confirma = True
                                # Salva votos no setup para analise posterior de LOSS
                                setup["consenso_votos"] = resultado_consenso["votos"]
                                setup["consenso_confianca"] = confianca
                                log.info(paint(f"[✅ CONSENSO 6 AGENTES] {a} | {direcao_consenso} | Conf: {confianca:.2f}", C.G))
                                # Log detalhado dos votos
                                votos_str = " | ".join([f"{v.nome}:{v.voto.value}" for v in resultado_consenso["votos"]])
                                log.info(paint(f"    Votos: {votos_str}", C.B))
                            else:
                                cnn_confirma = False
                                log.info(paint(f"[❌ CONSENSO DIVERGE] {a} | Setup={setup['dir']} vs Consenso={direcao_consenso}", C.R))
                        else:
                            # Sem consenso
                            cnn_confirma = False
                            motivo = resultado_consenso["motivo"]
                            log.info(paint(f"[❌ SEM CONSENSO] {a} | {motivo[:80]}", C.Y))

                    except Exception as e:
                        log.debug(f"[CONSENSO] Erro: {e}")
                        cnn_confirma = False  # Se erro, não entra

                # Fallback: se consenso não disponível, usa apenas CNN
                elif CNN_AVAILABLE and trading_cnn:
                    try:
                        sr_dir = "BULLISH" if setup["dir"] == "CALL" else "BEARISH"
                        pred = trading_cnn.predict(df, m5_direction=sr_dir, m5_strength=0.5)
                        cnn_class = pred["class"]
                        cnn_prob = pred["probability"]

                        if cnn_class == setup["dir"]:
                            cnn_confirma = True
                            log.info(paint(f"[✅ CNN] {a} | {setup['dir']} | Prob: {cnn_prob:.2f}", C.G))
                        elif cnn_class == "NO_TRADE":
                            cnn_confirma = True
                            log.info(paint(f"[⚠️ CNN INCERTO] {a} | {setup['dir']} - entrando", C.Y))
                        else:
                            cnn_confirma = False
                            log.info(paint(f"[❌ CNN BLOQUEIA] {a} | Setup={setup['dir']} vs CNN={cnn_class}", C.R))
                    except Exception as e:
                        log.debug(f"[CNN] Erro: {e}")
                        cnn_confirma = False

            else:
                # ===== MODO M5 PERNADA B =====
                if USE_M5_INDICATORS or USE_CANDLE_MID:
                    # Usa consenso + CNN no M5 (como solicitado)
                    cnn_confirma = False
                    if (USE_SKLEARN_ONLY or USE_CATBOOST_ONLY) and not (CONSENSO_AVAILABLE and sistema_consenso):
                        cnn_confirma = True
                        log.info(paint(f"[ML] {a} | consenso indisponível (M5), usando apenas ML", C.B))
                    elif CONSENSO_AVAILABLE and sistema_consenso:
                        try:
                            res_levels, sup_levels = strong_sr_levels_last200(df, atr_val)
                            sr_levels_dict = {
                                "resistencias": res_levels,
                                "suportes": sup_levels
                            }
                            resultado_consenso = sistema_consenso.analisar(df, atr_val, sr_levels=sr_levels_dict)

                            if resultado_consenso["trade"]:
                                direcao_consenso = resultado_consenso["direcao"]
                                confianca = resultado_consenso["confianca_media"]

                                if direcao_consenso == setup["dir"]:
                                    cnn_confirma = True
                                    setup["consenso_votos"] = resultado_consenso["votos"]
                                    setup["consenso_confianca"] = confianca
                                    log.info(paint(f"[✅ CONSENSO M5] {a} | {direcao_consenso} | Conf: {confianca:.2f}", C.G))
                                    votos_str = " | ".join([f"{v.nome}:{v.voto.value}" for v in resultado_consenso["votos"]])
                                    log.info(paint(f"    Votos: {votos_str}", C.B))
                                else:
                                    cnn_confirma = False
                                    log.info(paint(f"[❌ CONSENSO M5 DIVERGE] {a} | Setup={setup['dir']} vs Consenso={direcao_consenso}", C.R))
                            else:
                                cnn_confirma = False
                                motivo = resultado_consenso["motivo"]
                                log.info(paint(f"[❌ SEM CONSENSO M5] {a} | {motivo[:80]}", C.Y))

                                # Fallback: usa CNN quando não há consenso
                                if CNN_AVAILABLE and trading_cnn:
                                    try:
                                        pred = trading_cnn.predict(df)
                                        cnn_class = pred["class"]
                                        cnn_prob = pred["probability"]

                                        if cnn_class == setup["dir"]:
                                            cnn_confirma = True
                                            log.info(paint(f"[✅ CNN] {a} {cnn_class} | Prob: {cnn_prob:.2f}", C.G))
                                        elif cnn_class == "NO_TRADE":
                                            cnn_confirma = not CNN_STRICT
                                            log.info(paint(f"[⚠️ CNN INCERTO] {a} | {setup['dir']} - entrando", C.Y))
                                        else:
                                            cnn_confirma = False
                                            log.info(paint(f"[❌ CNN BLOQUEIA] {a} | Setup={setup['dir']} vs CNN={cnn_class}", C.R))
                                    except Exception as e:
                                        log.debug(f"[CNN] Erro: {e}")
                                        cnn_confirma = False
                                else:
                                    # Sem CNN: libera se score forte
                                    if float(setup.get("score", 0.0)) >= 0.70:
                                        cnn_confirma = True
                                        log.info(paint(f"[⚠️ LIBERADO] {a} | score alto sem consenso/CNN", C.Y))

                        except Exception as e:
                            log.debug(f"[CONSENSO M5] Erro: {e}")
                            cnn_confirma = False
                    elif CNN_AVAILABLE and trading_cnn:
                        try:
                            pred = trading_cnn.predict(df)
                            cnn_class = pred["class"]
                            cnn_prob = pred["probability"]

                            if cnn_class == setup["dir"]:
                                cnn_confirma = True
                                log.info(paint(f"[✅ CNN] {a} {cnn_class} | Prob: {cnn_prob:.2f}", C.G))
                            elif cnn_class == "NO_TRADE":
                                cnn_confirma = not CNN_STRICT
                                log.info(paint(f"[⚠️ CNN INCERTO] {a} | {setup['dir']} - entrando", C.Y))
                            else:
                                cnn_confirma = not CNN_STRICT
                                log.info(paint(f"[❌ CNN BLOQUEIA] {a} | Setup={setup['dir']} vs CNN={cnn_class}", C.R))
                        except Exception as e:
                            log.debug(f"[CNN] Erro: {e}")
                            cnn_confirma = False
                else:
                    # ===== MODO ANTIGO (M1) =====
                    # Validação M5 alignment (modo antigo, não M5 indicators)
                    if M5_ALIGN_BLOCK and REGIME_FILTER_AVAILABLE and regime_filter:
                        setup["m5_dir"] = m5_dir
                        setup["m5_strength"] = float(m5_strength)

                        if m5_dir == "NEUTRAL":
                            if M5_NEUTRAL_BLOCK and setup.get("score", 0) < M5_NEUTRAL_MIN_SCORE:
                                setup["trade"] = False
                                setup.setdefault("reasons", []).append("m5_neutro_nao_opera")
                                log.info(paint(f"[❌ M5 NEUTRO] {a} | score={setup.get('score', 0):.2f} < {M5_NEUTRAL_MIN_SCORE:.2f} - NÃO opera", C.Y))
                                _debug_reject(dbg, "m5_neutro")
                                continue
                            log.info(paint(f"[⚠️ M5 NEUTRO] {a} | score={setup.get('score', 0):.2f} - usa direção M1", C.Y))
                            m5_dir = setup["dir"].replace("CALL", "BULLISH").replace("PUT", "BEARISH")
                            m5_strength = max(0.20, m5_strength)

                        if setup["dir"] == "CALL" and m5_dir != "BULLISH":
                            setup["trade"] = False
                            setup.setdefault("reasons", []).append(f"m5_contra({m5_dir})")
                            log.info(paint(f"[❌ M5 CONFLITO] {a} CALL vs M5={m5_dir} - BLOQUEADO", C.R))
                            _debug_reject(dbg, "m5_conflito")
                            continue
                        if setup["dir"] == "PUT" and m5_dir != "BEARISH":
                            setup["trade"] = False
                            setup.setdefault("reasons", []).append(f"m5_contra({m5_dir})")
                            log.info(paint(f"[❌ M5 CONFLITO] {a} PUT vs M5={m5_dir} - BLOQUEADO", C.R))
                            _debug_reject(dbg, "m5_conflito")
                            continue

                        log.info(paint(f"[✅ M5 OK] {a} {setup['dir']} alinhado com M5={m5_dir} (força={m5_strength:.2f})", C.G))

                    # CNN SINCRONIZADA COM M5
                    cnn_confirma = True
                    if CNN_AVAILABLE and trading_cnn:
                        try:
                            pred = trading_cnn.predict(df, m5_direction=m5_dir, m5_strength=m5_strength)
                            cnn_class = pred["class"]
                            cnn_prob = pred["probability"]

                            if cnn_class == setup["dir"]:
                                cnn_confirma = True
                                log.info(paint(f"[✅ CNN] {a} {cnn_class} | Prob: {cnn_prob:.2f}", C.G))
                            elif cnn_class == "NO_TRADE":
                                cnn_confirma = not CNN_STRICT
                            else:
                                cnn_confirma = not CNN_STRICT
                        except Exception as e:
                            log.debug(f"[CNN] Erro: {e}")
                            cnn_confirma = True

            if cnn_confirma:
                cand_trade = (float(setup["score"]), a, setup, float(atr_val))
                if best_trade is None or cand_trade[0] > best_trade[0]:
                    best_trade = cand_trade

    ranked_any.sort(key=lambda x: x[0], reverse=True)
    return best_trade, best_any, ranked_any

# ===================== EXPIRAÇÃO ANTI-MANIPULAÇÃO =====================
def calcular_expiracao_segura(allowed_exps: Optional[List[int]] = None, prefer_exp: Optional[int] = None) -> int:
    """
    Calcula expiração entre EXP_MIN e EXP_MAX evitando minutos "redondos".

    Lógica:
    - Minutos redondos (:00, :05, :10, etc.) têm mais manipulação
    - Calcula qual expiração (3, 4 ou 5 min) evita expirar em minuto redondo
    - Se não conseguir evitar, usa a expiração que cai no minuto MENOS perigoso

    Returns:
        int: Expiração em minutos (3, 4 ou 5)
    """
    from datetime import datetime

    allowed = allowed_exps if allowed_exps else list(range(EXP_MIN, EXP_MAX + 1))
    allowed = sorted([int(x) for x in allowed if int(x) > 0])
    if not allowed:
        allowed = [EXP_MAX]

    if not EVITAR_MINUTOS_REDONDOS:
        if prefer_exp in allowed:
            return int(prefer_exp)
        return int(allowed[-1])  # Usa expiração máxima se desativado

    agora = datetime.now()
    minuto_atual = agora.minute
    segundo_atual = agora.second

    # Testa cada expiração possível
    melhor_exp = EXP_MAX
    menor_risco = 999

    for exp in allowed:
        # Calcula em qual minuto a operação vai expirar
        minuto_expiracao = (minuto_atual + exp) % 60

        # Verifica se é minuto "perigoso"
        if minuto_expiracao in MINUTOS_MANIPULADOS:
            # Quão perto está de ser um minuto redondo completo?
            # Se estamos no segundo 30+, a manipulação já pode ter começado
            risco = 10  # Alto risco
        else:
            # Distância do minuto redondo mais próximo
            dist_anterior = min((minuto_expiracao - m) % 60 for m in MINUTOS_MANIPULADOS if (minuto_expiracao - m) % 60 > 0)
            dist_posterior = min((m - minuto_expiracao) % 60 for m in MINUTOS_MANIPULADOS if (m - minuto_expiracao) % 60 > 0)
            risco = -min(dist_anterior, dist_posterior)  # Negativo = melhor (mais longe)

        if prefer_exp is not None and exp == int(prefer_exp):
            risco = max(0, risco - 1)

        if risco < menor_risco:
            menor_risco = risco
            melhor_exp = exp

    # Se todos são arriscados, adiciona aleatoriedade para dificultar previsão
    if menor_risco >= 10:
        import random
        melhor_exp = random.choice([EXP_MIN, EXP_MIN + 1, EXP_MAX])
        log.debug(paint(f"[ANTI-MANIP] Todos minutos perigosos, usando exp={melhor_exp}min (aleatório)", C.Y))
    else:
        log.debug(paint(f"[ANTI-MANIP] Exp={melhor_exp}min evita minuto redondo", C.G))

    return melhor_exp


def deve_esperar_entrada() -> Tuple[bool, int]:
    """
    Verifica se deve esperar para entrar (evita últimos segundos da vela).

    Returns:
        (deve_esperar, segundos_para_esperar)
    """
    from datetime import datetime

    agora = datetime.now()
    segundo = agora.second

    # Evita entrar nos últimos 5 segundos de uma vela (spike de manipulação)
    if segundo >= 55:
        return True, 60 - segundo + 2  # Espera até segundo 2 da próxima vela

    # Evita entrar nos primeiros 3 segundos (spread alto)
    if segundo <= 3:
        return True, 4 - segundo

    return False, 0


# ===================== ORDEM =====================
def enviar_ordem(iq: Casa_Trader, ativo: str, direcao: str, stake: float, exp_min: int) -> Optional[Tuple[str, int]]:
    d = "call" if direcao == "CALL" else "put"
    valor = float(max(VALOR_MINIMO, stake))

    exp_segura = int(exp_min)
    log.info(paint(f"[EXP] Usando {exp_segura}min (anti-manipulação)", C.B))

    # TURBO
    try:
        ok, op_id = safe_call(iq, iq.buy, valor, ativo, d, int(exp_segura))
        if ok and op_id:
            return ("turbo", int(op_id))
        log.warning(paint(f"[ORDEM-FAIL] TURBO ok={ok} op_id={op_id}", C.Y))
    except Exception as e:
        log.warning(paint(f"[ORDEM-EXC] TURBO {e}", C.Y))

    # DIGITAL
    try:
        ok, op_id = safe_call(iq, iq.buy_digital_spot, ativo, valor, d, int(exp_segura))
        if ok and op_id:
            return ("digital", int(op_id))
        log.warning(paint(f"[ORDEM-FAIL] DIGITAL ok={ok} op_id={op_id}", C.Y))
    except Exception as e:
        log.warning(paint(f"[ORDEM-EXC] DIGITAL {e}", C.Y))

    return None

def wait_result(iq: Casa_Trader, op_type: str, op_id: int) -> float:
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
    iq: Optional[Casa_Trader] = None
    iq = ensure_connected(iq)

    if USE_STRUCT_HYBRID:
        log.info(paint("🎯 MODO HIBRIDO: M5 Estrutura + M1 Setup + IA | Timeframe M1 | EXECUÇÃO ON", C.G))
    elif USE_RF_PROXY_ONLY:
        log.info(paint("🎯 MODO RF PROXY: Atividade OTC + RandomForest | Timeframe M1 | EXECUÇÃO ON", C.G))
    elif MODO_SR_SIMPLES:
        ml_name = "CATBOOST" if USE_CATBOOST_ONLY else ("SKLEARN" if USE_SKLEARN_ONLY else "CNN")
        log.info(paint(f"🎯 MODO PADRÃO VELAS: TA-Lib + {ml_name} (sem tendência) | Exp=1min | Timeframe M1 | EXECUÇÃO ON", C.G))
    else:
        if USE_CANDLE_MID:
            log.info("Iniciando: CANDLE MID (M1) | S/R forte + ZigZag + Tendência | EXECUÇÃO ON")
        elif USE_M5_INDICATORS:
            log.info("Iniciando: M5 PERNADA B | EXECUÇÃO ON")
        else:
            log.info("Iniciando: Pernada B (M1) | Anti-lateral + SR forte + IA Contextual | EXECUÇÃO ON")

    if IA_ON and AI_RESET_ON_START:
        _reset_ai_stats(AI_STATS_FILE)
        log.info(paint("[IA] Histórico resetado ao iniciar", C.B))
    stats = _safe_load_json(AI_STATS_FILE) if IA_ON else {"meta": {"total": 0}, "arms": {}}
    if IA_ON:
        _pretrain_ai(iq, obter_top_ativos_otc(iq), stats)
    if IA_ON:
        log.info(f"IA=ON | file={AI_STATS_FILE} | min_samples={AI_MIN_SAMPLES} | min_prob={AI_MIN_PROB:.2f} | conf_min={AI_CONF_MIN:.2f}")

    # Carregar LightGBM
    if LGBM_ON:
        lgbm_load_data()
        lgbm_load_model()
        log.info(paint(f"[LGBM] ON | mode={ENSEMBLE_MODE} | min_prob={LGBM_MIN_PROB:.2f} | samples={len(lgbm_data)}", C.B))
        if len(lgbm_data) >= LGBM_MIN_SAMPLES and lgbm_model is None:
            lgbm_train()
        if lgbm_reliable:
            log.info(paint(f"[LGBM] ✅ Modelo CONFIÁVEL (Val={lgbm_val_accuracy:.1f}%)", C.G))
        else:
            log.info(paint(f"[LGBM] ⚠️ Modelo NÃO confiável → usando APENAS Bayes até melhorar", C.Y))
    else:
        log.info("[LGBM] OFF - usando apenas Bayesiano")

    exp_stats = _safe_load_json(EXP_STATS_FILE)
    log.info(paint(
        f"[EXP-MODE] mode={EXP_MODE} allowed={EXP_ALLOWED} force={EXP_FORCE or 'none'} min_samples={EXP_MIN_SAMPLES}",
        C.B
    ))

    if USE_SKLEARN_ONLY:
        _train_sklearn_model(iq, obter_top_ativos_otc(iq))

    if USE_CATBOOST_ONLY:
        _train_catboost_model(iq, obter_top_ativos_otc(iq))

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

        # SKLEARN: retreino periódico
        if USE_SKLEARN_ONLY and SKLEARN_RETRAIN_SEC > 0:
            now_ts = time.time()
            if (_sk_last_train_ts is None) or ((now_ts - _sk_last_train_ts) >= SKLEARN_RETRAIN_SEC):
                log.info(paint("[SKLEARN] Retreino periódico iniciado", C.B))
                _train_sklearn_model(iq, obter_top_ativos_otc(iq))

        # CATBOOST: retreino periódico
        if USE_CATBOOST_ONLY and CATBOOST_RETRAIN_SEC > 0:
            now_ts = time.time()
            if (_cb_last_train_ts is None) or ((now_ts - _cb_last_train_ts) >= CATBOOST_RETRAIN_SEC):
                log.info(paint("[CATBOOST] Retreino periódico iniciado", C.B))
                _train_catboost_model(iq, obter_top_ativos_otc(iq))

        # VERIFICAR PAUSA DO RISK CONTROL (antes de qualquer análise)
        if RISK_CONTROL_AVAILABLE and risk_control:
            from datetime import datetime as _dt
            _now = _dt.now()
            if (not RISK_DISABLE_PAUSE) and risk_control.pause_until and _now < risk_control.pause_until:
                _remaining = (risk_control.pause_until - _now).seconds // 60
                log.info(paint(f"[RISK] ⏸️ Em pausa automática ({_remaining}min restantes) - aguardando...", C.Y))
                time.sleep(60)
                continue
            if risk_control.cooldown_until and _now < risk_control.cooldown_until:
                _remaining = (risk_control.cooldown_until - _now).seconds
                log.info(paint(f"[RISK] ⏳ Em cooldown ({_remaining}s restantes) - aguardando...", C.Y))
                time.sleep(min(30, _remaining + 1))
                continue

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

        tf_use = TF_M5 if USE_M5_INDICATORS else TF_M1
        
        # ESPERA SINCRONIZADA: Aguarda vela fechar completamente
        secs_to_next = seconds_to_next(tf_use)
        if secs_to_next > 0.5:
            time.sleep(secs_to_next + 0.15)  # 150ms após virada para garantir dados atualizados

        best_trade, best_any, ranked_any = escolher_melhor_setup(iq, ativos)

        if not best_trade:
            if best_any:
                sc, at, st, _av = best_any
                rej_summary = _summarize_rejects(st.get("debug") if isinstance(st, dict) else None) if DEBUG_REASONS else ""
                log.info(paint(
                    f"[SKIP] nenhum setup passou | melhor={at} score={sc:.2f} | {','.join(st.get('reasons', []))}{rej_summary}",
                    C.Y
                ))
                if LOG_RANKED and ranked_any:
                    top_n = ranked_any[:max(1, RANKED_TOP)]
                    resumo = "; ".join([f"{a}:{s:.2f}({d})" for s, a, d in top_n])
                    log.info(paint(f"[RANK] top={len(top_n)} | {resumo}", C.B))
                cooldown[at] = time.time()
            else:
                log.info(paint("[SKIP] nenhum ativo analisável no minuto", C.Y))

            continue

        score, ativo, setup, atr_val = best_trade
        score = float(score)

        if score < GATE_SOFT_SCORE:
            log.info(paint(
                f"[SKIP] {ativo} | score={score:.2f} | {','.join(setup.get('reasons', []))}",
                C.Y
            ))
            cooldown[ativo] = time.time()
            continue

        if score < GATE_MIN_SCORE:
            log.info(paint(
                f"[SOFT-SKIP] {ativo} | score={score:.2f} | {','.join(setup.get('reasons', []))}",
                C.B
            ))
            cooldown[ativo] = time.time()
            continue

        final_dir = str(setup["dir"])

        # ===================== PROTEÇÃO: LOSSES CONSECUTIVOS MESMO ATIVO+DIREÇÃO =====================
        streak_key = f"{ativo}_{final_dir}"
        streak_count = loss_streak_per_asset.get(streak_key, 0)
        if streak_count >= MAX_CONSECUTIVE_LOSS_SAME_DIR:
            log.info(paint(
                f"[BLOQUEIO] {ativo} {final_dir} | {streak_count} losses consecutivos nesta direção - BLOQUEADO",
                C.R
            ))
            cooldown[ativo] = time.time()
            continue

        log.info(paint(
            f"[PADRÃO] {ativo} -> {final_dir} | score={score:.2f} | sinais={'+'.join(setup.get('reasons', []))} (aguardando ML...)",
            dir_color(final_dir)
        ))

        # ===================== AUTO-CONHECIMENTO (NOVO!) =====================
        # Verifica se a IA aprendeu a evitar este setup
        contexto_trade = {
            "tendencia": setup.get("reasons", [""])[0].split(",")[0] if setup.get("reasons") else "lateral",
            "volatilidade": "alta" if atr_val > 0.001 else "normal",
            "pernada_a": int(setup.get("pernada_a", 0)) if isinstance(setup.get("pernada_a"), (int, float)) else 0,
            "pernada_b": int(setup.get("pernada_b", 0)) if isinstance(setup.get("pernada_b"), (int, float)) else 0,
            "retraction": float(setup.get("retr", 0.5)),
            "score": score,
            "atr": atr_val,
            "motivos": setup.get("reasons", [])
        }

        # IA-AUTO: aprende com losses passados e bloqueia setups com padrão ruim
        if IA_AUTOCONHECIMENTO_ON and (not USE_M5_INDICATORS) and (not MODO_SR_SIMPLES) and pode_entrar is not None:
            verificacao = pode_entrar(ativo, final_dir, contexto_trade)
            if not verificacao["pode"]:
                log.info(paint(f"[IA-AUTO] {ativo} {final_dir} | BLOQUEADO: {verificacao['motivo']}", C.R))
                cooldown[ativo] = time.time()
                continue
            else:
                log.info(paint(f"[IA-AUTO] {ativo} {final_dir} | {verificacao['motivo']}", C.G))

        # ===================== CATBOOST ONLY =====================
        # CATBOOST é CONFIRMAÇÃO após padrões de candlestick
        # Padrões fortes (score >= 0.80) têm prioridade - ML é soft confirmation
        if USE_CATBOOST_ONLY:
            df_cb = get_candles_df(iq, ativo, tf_use, max(60, N_M1), end_ts=end_ts_closed(tf_use))
            if df_cb is None or len(df_cb) < 60:
                log.info(paint(f"[CATBOOST-SKIP] {ativo} | dados insuficientes (padrão: {setup.get('reasons', [''])[0]})", C.Y))
                cooldown[ativo] = time.time()
                continue
            cb_dir, cb_prob = _cb_predict(df_cb)
            
            # Se padrão é forte (>= 0.80), ML só precisa não discordar fortemente
            pattern_score = setup.get("score", 0.0)
            is_strong_pattern = pattern_score >= 0.80
            
            # Extrai nome do padrão para aprendizado
            pattern_reasons = setup.get("reasons", [])
            pattern_name = pattern_reasons[0] if pattern_reasons else "unknown"
            
            # ===================== FILTRO ADAPTATIVO - ALTA ASSERTIVIDADE =====================
            # Verifica antes de qualquer entrada se atende critérios de alta probabilidade
            ml_prob_for_check = cb_prob if cb_dir == final_dir else (1 - cb_prob)
            setup["ml_prob"] = ml_prob_for_check  # Guarda para aprendizado de LOSS
            can_enter, adaptive_reason = can_trade_adaptive(ativo, pattern_score, ml_prob_for_check, pattern_name, final_dir, log)
            if not can_enter:
                log.info(paint(f"[ADAPTIVE-BLOCK] {ativo} {final_dir} | {adaptive_reason}", C.R))
                cooldown[ativo] = time.time() + ADAPTIVE_COOLDOWN_AFTER_LOSS
                continue
            
            if is_strong_pattern:
                # Padrão forte: AINDA requer ML mínimo + não pode ter ML forte contra
                ml_agrees = (cb_dir == final_dir)
                ml_strongly_against = (cb_dir != final_dir and cb_prob >= 0.60)
                
                # NOVO: mesmo padrão forte precisa de ML >= CATBOOST_MIN_PROB quando concorda
                if ml_agrees and cb_prob < CATBOOST_MIN_PROB:
                    log.info(paint(f"[CATBOOST-BLOCK] {ativo} | padrão_forte={pattern_score:.2f} mas ML={cb_dir}({cb_prob:.2f}) FRACO < {CATBOOST_MIN_PROB:.2f}", C.Y))
                    cooldown[ativo] = time.time()
                    continue
                
                if ml_strongly_against:
                    log.info(paint(f"[CATBOOST-BLOCK] {ativo} | padrão={final_dir}(score={pattern_score:.2f}) vs ML={cb_dir}({cb_prob:.2f}) FORTE contra", C.Y))
                    cooldown[ativo] = time.time()
                    continue
                else:
                    log.info(paint(f"[CATBOOST-OK] {ativo} {final_dir} | padrão_forte={pattern_score:.2f} | ML={cb_dir}({cb_prob:.2f}) | {adaptive_reason} ✓", C.G))
            else:
                # Padrão fraco: exige confirmação ML
                if cb_prob < CATBOOST_MIN_PROB:
                    log.info(paint(f"[CATBOOST-BLOCK] {ativo} {cb_dir} | prob={cb_prob:.2f} < {CATBOOST_MIN_PROB:.2f} (padrão fraco + ML não confirma)", C.Y))
                    cooldown[ativo] = time.time()
                    continue
                if cb_dir != final_dir:
                    log.info(paint(f"[CATBOOST-BLOCK] {ativo} | padrão={final_dir} vs ML={cb_dir} (direções divergentes)", C.Y))
                    cooldown[ativo] = time.time()
                    continue
                log.info(paint(f"[CATBOOST-OK] {ativo} {cb_dir} | prob={cb_prob:.2f} | {adaptive_reason} ✓ (padrão + ML confirmados)", C.G))

        # ===================== SKLEARN ONLY =====================
        if USE_SKLEARN_ONLY:
            df_sk = get_candles_df(iq, ativo, tf_use, max(60, N_M1), end_ts=end_ts_closed(tf_use))
            if df_sk is None or len(df_sk) < 60:
                log.info(paint(f"[SKLEARN-SKIP] {ativo} | dados insuficientes", C.Y))
                cooldown[ativo] = time.time()
                continue
            sk_dir, sk_prob = _sk_predict(df_sk)
            if sk_prob < SKLEARN_MIN_PROB:
                log.info(paint(f"[SKLEARN-SKIP] {ativo} {sk_dir} | prob={sk_prob:.2f} < {SKLEARN_MIN_PROB:.2f}", C.Y))
                cooldown[ativo] = time.time()
                continue
            if sk_dir != final_dir:
                log.info(paint(f"[SKLEARN-SKIP] {ativo} | modelo={sk_dir} vs setup={final_dir}", C.Y))
                cooldown[ativo] = time.time()
                continue
            log.info(paint(f"[SKLEARN] {ativo} {sk_dir} | prob={sk_prob:.2f}", C.G))

        # ===================== IA FILTRO (MODELO PROFISSIONAL) =====================
        if IA_ON:
            ens = ensemble_predict(ativo, setup, stats)
            bayes_prob = float(ens["bayes_prob"])
            lgbm_prob = float(ens.get("lgbm_prob", 0.5))
            ensemble_prob = float(ens["ensemble_prob"])
            should_trade = bool(ens["should_trade"])
            ens_reason = str(ens["reason"])
            n_arm = int(ens.get("n_arm", 0))
            bayes_conf = float(ens.get("bayes_conf", 0.0))

            lgbm_log = f" | L={lgbm_prob:.2f}" if lgbm_prob != 0.5 else ""
            log.info(paint(
                f"[ENSEMBLE] {ativo} {final_dir} | B={bayes_prob:.2f} (n={n_arm}){lgbm_log} | ens={ensemble_prob:.2f} | {ens_reason}",
                C.B
            ))

            if not should_trade:
                log.info(paint(f"[IA-SKIP] {ativo} {final_dir} | {ens_reason}", C.Y))
                cooldown[ativo] = time.time()
                continue

        # M5: só entra se estiver bem próximo da virada (0,5,10,15...)
        if USE_M5_INDICATORS:
            candle_age = time.time() % tf_use
            if candle_age > M5_ENTRY_MAX_DELAY:
                log.info(paint(
                    f"[{ativo}] ⏳ sinal atrasado ({candle_age:.2f}s pós-virada) -> aguardando próxima vela",
                    C.Y
                ))
                wait_for_next_open(tf_use)
                cooldown[ativo] = time.time()
                continue

        # MODO SIMPLIFICADO: pode esperar a virada da vela para entrar na abertura
        # MODO NORMAL: espera abertura do próximo candle
        if USE_M5_INDICATORS:
            log.info(paint(f"[{ativo}] ✅ ENTRADA NA VIRADA (M5 indicadores)", C.G))
        elif MODO_SR_SIMPLES:
            log.info(paint(f"[{ativo}] ⚡ ENTRADA (padrão + ML confirmados)", C.G))
        elif USE_STRUCT_HYBRID:
            log.info(paint(f"[{ativo}] ⚡ ENTRADA (híbrido confirmado)", C.G))
        elif USE_RF_PROXY_ONLY:
            log.info(paint(f"[{ativo}] ⚡ ENTRADA (RF proxy confirmado)", C.G))
        else:
            wait_for_next_open(tf_use)

        # Guard: evita entrar se o preço já se afastou demais do S/R
        # Guard: BLOQUEIA se preço CRUZOU o nível S/R (suporte virou resistência ou vice-versa)
        if (not USE_M5_INDICATORS) and MODO_SR_SIMPLES and setup.get("sr_level") is not None:
            try:
                atr_safe = max(float(atr_val), 1e-9)
                df_now = get_candles_df(iq, ativo, tf_use, 3, end_ts=time.time())
                if df_now is not None and len(df_now) > 0:
                    last_close = float(df_now.iloc[-1]["close"])
                    sr_lvl = float(setup.get("sr_level"))
                    dist_atr = abs(last_close - sr_lvl) / atr_safe

                    # NOVO: Verifica se preço CRUZOU o nível S/R
                    if final_dir == "CALL" and last_close < sr_lvl:
                        # CALL em suporte mas preço agora está ABAIXO do suporte → suporte rompido!
                        log.info(paint(
                            f"[{ativo}] ⛔ SUPORTE ROMPIDO na entrada! Preço={last_close:.5f} < S/R={sr_lvl:.5f} - CALL CANCELADO",
                            C.R
                        ))
                        cooldown[ativo] = time.time()
                        continue
                    if final_dir == "PUT" and last_close > sr_lvl:
                        # PUT em resistência mas preço agora está ACIMA da resistência → resistência rompida!
                        log.info(paint(
                            f"[{ativo}] ⛔ RESISTÊNCIA ROMPIDA na entrada! Preço={last_close:.5f} > S/R={sr_lvl:.5f} - PUT CANCELADO",
                            C.R
                        ))
                        cooldown[ativo] = time.time()
                        continue

                    if dist_atr > SR_ENTRY_MAX_DIST_ATR:
                        log.info(paint(
                            f"[{ativo}] ⛔ S/R distante na entrada (dist={dist_atr:.2f}ATR > {SR_ENTRY_MAX_DIST_ATR:.2f})",
                            C.Y
                        ))
                        cooldown[ativo] = time.time()
                        continue
            except Exception as e:
                log.debug(f"[S/R] Erro ao checar distância de entrada: {e}")

        # Guard: bloqueia entrada em sequência forte de velas da mesma cor
        # DESATIVADO no modo tendência (queremos seguir sequências fortes)
        if (not USE_M5_INDICATORS) and MODO_SR_SIMPLES and setup.get("strategy") != "TREND_CANDLE":
            try:
                streak_n = max(3, SR_STREAK_BLOCK_BARS)
                df_now = get_candles_df(iq, ativo, tf_use, streak_n, end_ts=time.time())
                if df_now is not None and len(df_now) >= streak_n:
                    last_n = df_now.tail(streak_n)
                    up_n = sum(1 for _, r in last_n.iterrows() if r["close"] > r["open"])
                    down_n = streak_n - up_n
                    max_ratio = max(up_n, down_n) / max(1, streak_n)
                    if max_ratio >= SR_STREAK_BLOCK_RATIO:
                        log.info(paint(
                            f"[{ativo}] ⛔ sequência forte detectada ({max_ratio:.0%} mesma cor) - bloqueado",
                            C.Y
                        ))
                        cooldown[ativo] = time.time()
                        continue
            except Exception as e:
                log.debug(f"[S/R] Erro ao checar sequência forte: {e}")

        # STAKE DINÂMICO BASEADO NA BANCA (NOVO)
        stake = calcular_stake_dinamico(iq, STAKE_FIXA)
        log.info(paint(f"[{ativo}] 💵 Stake calculado: {stake:.2f}", C.B))

        saldo_antes = None
        try:
            saldo_antes = float(iq.get_balance())
        except Exception:
            saldo_antes = None

        # ANTI-SPIKE: Evita entrar nos últimos segundos da vela (manipulação)
        # NO MODO S/R/RF: Não espera anti-spike (queremos entrada rápida na abertura)
        if not (MODO_SR_SIMPLES or USE_RF_PROXY_ONLY or USE_STRUCT_HYBRID):
            esperar, segundos = deve_esperar_entrada()
            if esperar:
                log.info(paint(f"[{ativo}] ⏳ Anti-spike: aguardando {segundos}s para evitar manipulação", C.Y))
                time.sleep(segundos)

        # ===================== EXPIRAÇÃO =====================
        exp_prefer, exp_reason = _choose_exp_for_trade(exp_stats, ativo, final_dir)
        # MODO TENDÊNCIA: expiração FIXA de 1 minuto (seguindo tendência forte)
        if MODO_SR_SIMPLES:
            exp_prefer = 1
            exp_reason = "trend_1min"

        exp_usada = calcular_expiracao_segura(EXP_ALLOWED, exp_prefer)
        if exp_usada != exp_prefer:
            log.info(paint(
                f"[EXP] Preferida={exp_prefer}min ({exp_reason}) -> ajustada={exp_usada}min (anti-manip)",
                C.Y
            ))
        else:
            log.info(paint(f"[EXP] Preferida={exp_usada}min ({exp_reason})", C.B))

        op = enviar_ordem(iq, ativo, final_dir, stake, exp_usada)

        if not op:
            log.error(paint(f"[{ativo}] ❌ falhou enviar ordem (TURBO/DIGITAL).", C.R))
            cooldown[ativo] = time.time()
            continue

        op_type, op_id = op
        log.info(paint(
            f"[{ativo}] ✅ ORDEM ENVIADA {final_dir} exp={exp_usada}m ({op_type}) | stake={stake:.2f}",
            dir_color(final_dir)
        ))

        # Registra trade no Risk Control
        if RISK_CONTROL_AVAILABLE and risk_control:
            risk_control.on_trade_opened()

        res = wait_result(iq, op_type, op_id)

        if res == 0 and saldo_antes is not None:
            try:
                saldo_depois = float(iq.get_balance())
                delta = saldo_depois - saldo_antes
                if delta < -0.01:
                    res = float(delta)
                    log.info(paint(f"[{ativo}] ⚠️ ajuste: empate virou LOSS {res:.2f}$ (saldo)", C.Y))
            except Exception:
                pass

        total += 1
        # Atualiza rastreador de losses consecutivos por ativo+direção
        streak_key = f"{ativo}_{final_dir}"
        if res > 0:
            wins += 1
            loss_streak_per_asset[streak_key] = 0  # Reseta streak no WIN
            update_adaptive_stats(ativo, is_win=True)  # FILTRO ADAPTATIVO - atualiza winrate
            log.info(paint(f"[{ativo}] ✅ WIN {res:.2f}$ | Boost ML resetado", C.G))
        elif res < 0:
            loss_streak_per_asset[streak_key] = loss_streak_per_asset.get(streak_key, 0) + 1
            update_adaptive_stats(ativo, is_win=False)  # FILTRO ADAPTATIVO - atualiza winrate
            
            # ==== APRENDIZADO DE LOSS - Memoriza contexto para evitar repetições ====
            pattern_reasons_loss = setup.get("reasons", [])
            pattern_name_loss = pattern_reasons_loss[0] if pattern_reasons_loss else "unknown"
            ml_prob_loss = setup.get("ml_prob", 0.0)
            pattern_score_loss = setup.get("score", 0.0)
            add_loss_to_memory(ativo, pattern_name_loss, final_dir, ml_prob_loss, pattern_score_loss)
            
            wr, n = get_adaptive_winrate(ativo)
            log.info(paint(f"[{ativo}] ❌ LOSS {res:.2f}$ | streak={loss_streak_per_asset[streak_key]} | winrate={wr:.0%}(n={n}) | ML boost={adaptive_ml_boost:.2f}", C.R))
        else:
            log.info(paint(f"[{ativo}] ⚪ EMPATE {res:.2f}$", C.B))

        try:
            _update_exp_stats(exp_stats, ativo, final_dir, exp_usada, res)
            _safe_save_json(EXP_STATS_FILE, exp_stats)
        except Exception:
            pass

        if res < 0 and LOSS_REPORT_ON:
            try:
                report = _build_loss_report(iq, ativo, final_dir, stake, res, setup, atr_val, score)
                _append_loss_report(LOSS_REPORT_FILE, report, LOSS_REPORT_MAX)
                log.info(paint(f"[LOSS_REPORT] ✅ salvo em {LOSS_REPORT_FILE}", C.B))
            except Exception as e:
                log.debug(f"[LOSS_REPORT] Erro ao salvar: {e}")

        # ===================== ANALISADOR DE LOSS (IA GENERATIVA) =====================
        if res < 0 and LOSS_ANALYZER_AVAILABLE and loss_analyzer:
            try:
                df_loss = get_candles_df(iq, ativo, tf_use, 50, end_ts=end_ts_closed(tf_use))
                if df_loss is not None:
                    # Coleta votos dos agentes para analise
                    agentes_votos = {}
                    if setup.get("consenso_votos"):
                        for v in setup["consenso_votos"]:
                            agentes_votos[v.nome] = v.voto.value

                    # Analisa o loss
                    loss_ctx = loss_analyzer.analyze_loss(
                        df_m1=df_loss,
                        direcao=final_dir,
                        preco_entrada=float(df_loss.iloc[-1]["close"]),
                        preco_saida=float(df_loss.iloc[-1]["close"]) + (res / stake) * 0.0001,
                        prejuizo=res,
                        agentes_votos=agentes_votos,
                        atr_val=float(atr_val),
                        ativo=ativo
                    )

                    log.info(paint(f"[LOSS-AI] Categoria: {loss_ctx.categoria_loss}", C.M))
                    log.info(paint(f"[LOSS-AI] Motivo: {loss_ctx.motivo_detalhado}", C.M))
                    log.info(paint(f"[LOSS-AI] Licao: {loss_ctx.licao_aprendida}", C.B))

            except Exception as e:
                log.debug(f"[LOSS-AI] Erro ao analisar: {e}")

        # update IA após resultado
        if IA_ON:
            _update_asset_stats(stats, ativo, res)
            ai_update(ativo, setup, res, stats)
            _safe_save_json(AI_STATS_FILE, stats)

        # Adiciona amostra ao LightGBM para aprendizado
        if LGBM_ON:
            lgbm_add_sample(setup, res)

        # ===================== ATUALIZA CNN (NOVO SISTEMA) =====================
        try:
            n_use = N_M5 if USE_M5_INDICATORS else N_M1
            df_trade = get_candles_df(iq, ativo, tf_use, n_use, end_ts=end_ts_closed(tf_use))
            if df_trade is not None:
                on_trade_result_cnn(ativo, final_dir, win=(res > 0), df_m1=df_trade, profit=res)
        except Exception as e:
            log.debug(f"[CNN] Erro ao registrar resultado: {e}")

        # ===================== AUTO-CONHECIMENTO - APRENDER COM RESULTADO =====================
        # Desabilitado no modo S/R (usa AutoTuner em vez disso)
        if IA_AUTOCONHECIMENTO_ON and not MODO_SR_SIMPLES and registrar_trade is not None:
            resultado_str = "WIN" if res > 0 else "LOSS" if res < 0 else "EMPATE"
            registrar_trade(ativo, final_dir, resultado_str, res, contexto_trade)
            log.info(paint(f"[IA-AUTO] Aprendendo: {ativo} {final_dir} = {resultado_str}", C.M))

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

        if IA_ON:
            asset_stats = stats.get("assets", {}).get(ativo)
            if asset_stats:
                a_wins = int(asset_stats.get("wins", 0))
                a_losses = int(asset_stats.get("losses", 0))
                a_trades = int(asset_stats.get("trades", a_wins + a_losses))
                a_acc = (a_wins / max(1, a_trades)) * 100.0
                a_color = C.G if a_acc >= 50.0 else C.Y
                log.info(paint(f"📊 ATIVO {ativo}: trades={a_trades} wins={a_wins} acc={a_acc:.2f}%", a_color))

        cooldown[ativo] = time.time()

if __name__ == "__main__":
    main()
