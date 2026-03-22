# -*- coding: utf-8 -*-
"""
WS_AUTO_AI_BULLEX — Motor Principal de Trading Automatizado
═══════════════════════════════════════════════════════════════
Bot completo que opera Double Top/Bottom + Head & Shoulders em opções binárias.
Expiração fixa 1 minuto. Suporta: Bullex, CasaTrader, IQ Option.

ESTRATÉGIA:
  - Escaneia velas no segundo :50 (10s antes da virada)
  - Detecta padrões DT/H&S com pivots e geometria
  - Cada padrão passa por 6 Guards + IA Bayesiana antes de entrar
  - Entrada na virada do candle (:00) com expiração de 1min

FLUXO DE ENTRADA (escolher_melhor_setup_local):
  1. detect_pivots()       → encontra pivot highs/lows (window=5)
  2. detect_double_touch() → detecta Double Top (PUT) / Double Bottom (CALL)
  3. detect_all_hs()       → detecta Head & Shoulders
  4. Deduplicação de padrões
  5. ai_predict_hs()       → Bayesian brain (ws_adaptive_brain.py, 63K+ samples)
  6. ia_pattern_quality()  → score de geometria (depth, symmetry, span)
  7. Guard 1: POSIÇÃO      → preço deve estar perto do RS (< 60% ATR)
  8. Guard 2: DEPTH RATIO  → profundidade < 25 ATR (padrões muito profundos = ruim)
  9. Guard 3: SYMMETRY     → faixa 0.45-0.60 bloqueada (pior WR)
  10. Guard 4: MOMENTUM    → preço deve ter voltado na direção certa
  11. Guard 5: PROXIMIDADE → preço não pode estar já perto do alvo
  12. Guard 6: NN 3 MODELOS → extract_features() → predict_dt() (Win/Loss)
  13. Decisão → wait :00 → enviar ordem

MÓDULOS USADOS:
  - ws_reversal_ai.py     → 3 NNs (GradientBoosting + LightGBM + MLP)
  - ws_adaptive_brain.py  → extract_features() + Bayesian kNN brain

FUNÇÕES PRINCIPAIS:
  Detecção:
    - detect_pivots(H, L)                    → (pivot_highs, pivot_lows)
    - detect_double_touch(H,L,C,O,ph,pl,...) → lista de DT patterns
    - detect_all_hs(H,L,C,O,ph,pl,atr)      → lista de H&S patterns
    - detect_early_hs(...)                   → H&S em formação

  Análise:
    - backtest_pattern(pat, C, O, H, L, n)   → {result: "win"/"loss"}
    - _extract_geometry(pat, atr)             → dict com métricas geométricas
    - ia_pattern_quality(pat, atr, stats)     → score 0-100 de qualidade
    - ai_predict_hs(ativo, pat, stats)        → predição Bayesiana

  Execução:
    - main() / _main_inner()                 → loop principal do bot
    - escolher_melhor_setup_local(...)        → seleciona melhor DT com Guards
    - enviar_ordem(bx, ativo, dir, stake)     → envia ordem ao broker
    - wait_result(bx, op_type, op_id)         → aguarda resultado

  Treino online:
    - _train_ia_from_history(bx, hs_stats)   → treina NN com dados do brain/CSVs

  Seleção de ativos:
    - _pick_top_dt_assets(hs_stats, n_top)   → top 3 ativos por WR DT
    - obter_top_ativos_otc(bx)               → lista ativos OTC disponíveis

CONFIGURAÇÃO:
  - 3 ativos fixos: USDZAR-OTC, USDHKD-OTC, NZDJPY-OTC
  - Meta: +3% do saldo, Stop: -10%
  - Stake: 2% do saldo (mín $2)
  - EXP_FIXA = 1 (expiração 1 minuto)
"""

import os
import sys
import time
import json
import pickle
import logging
import random
import threading

# ── Garantir que o diretório do script está no sys.path ──
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

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

from numpy_pickle_compat import patch_numpy_pickle_compat

patch_numpy_pickle_compat()

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
from ws_reversal_ai import (
    ReversalAI,
    FEATURE_NAMES,
    get_ws_user_data_dir,
    get_reversal_model_persist_path,
    find_existing_reversal_model_path,
)
from ws_adaptive_brain import extract_features

# ═══ IA 4 — FILTRO DE CONTEXTO (tabela pré-computada do backtest) ═══
from ws_context_filter import context_lookup, format_context_log

try:
    from tradingpatterns import tradingpatterns as _shadow_patterns_lib
    _HAS_SHADOW_PATTERN_LIB = True
except Exception:
    _shadow_patterns_lib = None
    _HAS_SHADOW_PATTERN_LIB = False

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
LIVE_SCAN_N_M1 = max(180, int(os.getenv("WS_LIVE_SCAN_CANDLES", "300")))
SHADOW_PATTERN_LIB_ENABLED = os.getenv("WS_SHADOW_PATTERN_LIB", "1").strip().lower() not in {"0", "false", "no", "off"}
SHADOW_PATTERN_LIB_WINDOW = max(3, int(os.getenv("WS_SHADOW_PATTERN_LIB_WINDOW", "3")))
SHADOW_PATTERN_LIB_THRESHOLD = float(os.getenv("WS_SHADOW_PATTERN_LIB_THRESHOLD", "0.05"))
SHADOW_PATTERN_LIB_MAX_BAR_DISTANCE = max(1, int(os.getenv("WS_SHADOW_PATTERN_LIB_MAX_BAR_DISTANCE", "4")))

# ── Payout / Assets ──
PAYOUT_MINIMO = int(os.getenv("WS_PAYOUT_MIN", "80"))   # 80%+ payout → com WR 90%+ é lucrativo
PAYOUT_REFRESH_SEC = int(os.getenv("WS_PAYOUT_REFRESH", "180"))

NUM_ATIVOS = max(1, int(os.getenv("WS_NUM_ATIVOS", "4")))
SCAN_NUM_ATIVOS = max(1, min(NUM_ATIVOS, int(os.getenv("WS_SCAN_ATIVOS", str(NUM_ATIVOS)))))

# ── Expiração FIXA 2 minutos (alinhada com treino) ──
# O modelo foi treinado com EXP_FIXA=3 + bug off-by-one (C[exit_idx-1])
# que na prática checava C[entry_idx + 2] = 2 candles após entrada.
# Agora com o off-by-one corrigido (C[exit_idx]):
#   EXP_FIXA=2 → exit = C[entry_idx + 2] → MESMO resultado do treino
# Live: broker recebe 2 min = 2 candles M1 → alinhado com treino
EXP_FIXA = 2
EXP_EARLY = 5  # delay=0: EXP=5 → 91.4% WR

# ── URL da base de treino no GitHub (auto-download semanal) ──
# O desenvolvedor sobe ws_ai_base_training.json toda semana
# Os clientes baixam automaticamente na inicialização
GITHUB_TRAINING_URL = os.getenv(
    "WS_TRAINING_URL",
    "https://raw.githubusercontent.com/whsouza22/wstrader-update/main/ws_ai_base_training.json"
)
GITHUB_ENTRY_GUARD_API_URL = os.getenv(
    "WS_ENTRY_GUARD_API_URL",
    "https://api.github.com/repos/whsouza22/wstrader-update/contents/models_entry_guard"
)
GITHUB_ENTRY_GUARD_MANIFEST_URL = os.getenv(
    "WS_ENTRY_GUARD_MANIFEST_URL",
    "https://raw.githubusercontent.com/whsouza22/wstrader-update/main/models_entry_guard/manifest.json"
)
GITHUB_ENTRY_GUARD_RAW_URL = os.getenv(
    "WS_ENTRY_GUARD_RAW_URL",
    "https://raw.githubusercontent.com/whsouza22/wstrader-update/main/models_entry_guard/{file_name}"
)
BASE_TRAINING_LOCAL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ws_ai_base_training.json"
)

# ── Stake / Banca ──
VALOR_MINIMO = float(os.getenv("WS_VALOR_MINIMO", "3"))
STAKE_FIXA = float(os.getenv("WS_STAKE", "5"))
PERCENT_BANCA = float(os.getenv("WS_PERCENT_BANCA", "1.0"))
# Stake inteligente: $1/$2/$3 baseado na confiança da IA
STAKE_NIVEL_1 = 1.0   # prob_win < 85% → stake mínimo $1
STAKE_NIVEL_2 = 2.0   # 85% ≤ prob_win < 92% → stake médio $2
STAKE_NIVEL_3 = 3.0   # prob_win ≥ 92% → stake máximo $3
META_LUCRO_PERCENT = float(os.getenv("WS_META_LUCRO", "4.5"))
STOP_LOSS_PERCENT = float(os.getenv("WS_STOP_LOSS", "3.0"))
USE_DYNAMIC_STAKE = (os.getenv("WS_DYNAMIC_STAKE", "1").strip() == "1")

# ── MODO IA: Guards 1-5 desativados — somente as 3 NNs decidem (treinadas com 63K+ samples) ──
_GUARDS_DISABLED = True  # Guards 1-5 OFF — as 3 IAs (89.7% acc) já sabem filtrar

# ── WS Trader 2.0: DESATIVADO — NN com 40 features é a única decisão ──
_DECISION_ENGINE_ENABLED = False  # Desativado: regime/quality/risk são features f26-f39 no NN
_recent_trade_results = []        # mantido para log/estatísticas
_consecutive_losses = 0           # contador de losses consecutivos para adaptar threshold

# ── Reversal AI config ──
CONFIDENCE_MIN = float(os.getenv('WS_CONF_MIN', "40.0"))       # Confiança mínima da IA para entrar
ANALYZE_AT_SECOND = int(os.getenv("WS_ANALYZE_SEC", "30"))      # Analisar no segundo :30 e, se liberar, entrar direto na virada :00
COOLDOWN_AFTER_TRADE = int(os.getenv("WS_COOLDOWN", "180"))      # Cooldown global após cada trade (3 min)
MIN_WR_ATIVO = float(os.getenv("WS_MIN_WR", "80.0"))            # WR mínimo para selecionar ativo
MAX_ENTRY_DELAY_SEC = float(os.getenv("WS_MAX_ENTRY_DELAY_SEC", "6.0"))
MAX_LIVE_SIGNAL_CANDLES = max(1, int(os.getenv("WS_MAX_LIVE_SIGNAL_CANDLES", "1")))
DT_LATE_PROGRESS_PCT = float(os.getenv("WS_DT_LATE_PROGRESS_PCT", "60.0"))
DT_LATE_MIN_TARGET_ATR = float(os.getenv("WS_DT_LATE_MIN_TARGET_ATR", "3.5"))
ENABLE_CONTEXT_FILTER = True    # Filtro de contexto baseado em backtest
DT_ENTRY_AT_TURN = (os.getenv("WS_DT_ENTRY_AT_TURN", "1").strip() == "1")
DT_GRAPH_SIGNAL_ENTRY = (os.getenv("WS_DT_GRAPH_SIGNAL_ENTRY", "1").strip() == "1")
DT_GRAPH_NN_ONLY_TEST = (os.getenv("WS_DT_GRAPH_NN_ONLY_TEST", "1").strip() == "1")
DT_GRAPH_TIMING_WAIT_MAX_SEC = max(2.0, float(os.getenv("WS_DT_GRAPH_TIMING_WAIT_MAX_SEC", "35.0")))
DT_GRAPH_TIMING_POLL_SEC = max(0.2, float(os.getenv("WS_DT_GRAPH_TIMING_POLL_SEC", "0.5")))
DT_GRAPH_TIMING_DIRECT_NN_MIN = float(os.getenv("WS_DT_GRAPH_TIMING_DIRECT_NN_MIN", "0.88"))
DT_GRAPH_TIMING_DIRECT_PARTIAL_NN_MIN = float(os.getenv("WS_DT_GRAPH_TIMING_DIRECT_PARTIAL_NN_MIN", "0.88"))
DT_GRAPH_TIMING_DIRECT_MAX_PROGRESS = float(os.getenv("WS_DT_GRAPH_TIMING_DIRECT_MAX_PROGRESS", "0.35"))
DT_GRAPH_TIMING_DIRECT_PARTIAL_MAX_PROGRESS = float(os.getenv("WS_DT_GRAPH_TIMING_DIRECT_PARTIAL_MAX_PROGRESS", "0.30"))
DT_GRAPH_TIMING_DIRECT_MAX_DIST_ATR = float(os.getenv("WS_DT_GRAPH_TIMING_DIRECT_MAX_DIST_ATR", "0.45"))
DT_GRAPH_TIMING_DIRECT_PARTIAL_MAX_DIST_ATR = float(os.getenv("WS_DT_GRAPH_TIMING_DIRECT_PARTIAL_MAX_DIST_ATR", "0.38"))
DT_GRAPH_TIMING_DIRECT_ZONE_BUFFER_ATR = float(os.getenv("WS_DT_GRAPH_TIMING_DIRECT_ZONE_BUFFER_ATR", "0.06"))
DT_GRAPH_TIMING_DIRECT_PARTIAL_ZONE_BUFFER_ATR = float(os.getenv("WS_DT_GRAPH_TIMING_DIRECT_PARTIAL_ZONE_BUFFER_ATR", "0.04"))
DT_GRAPH_TIMING_DIRECT_TOUCH_MIN = float(os.getenv("WS_DT_GRAPH_TIMING_DIRECT_TOUCH_MIN", "0.54"))
DT_GRAPH_TIMING_DIRECT_PARTIAL_TOUCH_MIN = float(os.getenv("WS_DT_GRAPH_TIMING_DIRECT_PARTIAL_TOUCH_MIN", "0.58"))
DT_GRAPH_TIMING_SOFT_RELEASE_NN_MIN = float(os.getenv("WS_DT_GRAPH_TIMING_SOFT_RELEASE_NN_MIN", "0.76"))
DT_GRAPH_TIMING_SOFT_RELEASE_MAX_PROGRESS = float(os.getenv("WS_DT_GRAPH_TIMING_SOFT_RELEASE_MAX_PROGRESS", "60.0"))
DT_GRAPH_TIMING_SOFT_RELEASE_MIN_TARGET_ATR = float(os.getenv("WS_DT_GRAPH_TIMING_SOFT_RELEASE_MIN_TARGET_ATR", "3.0"))
DT_GRAPH_TIMING_SOFT_RELEASE_MAX_DIST_ATR = float(os.getenv("WS_DT_GRAPH_TIMING_SOFT_RELEASE_MAX_DIST_ATR", "1.45"))
DT_GRAPH_TIMING_BARRIER_LOOKBACK = int(os.getenv("WS_DT_GRAPH_TIMING_BARRIER_LOOKBACK", "10"))
DT_GRAPH_TIMING_BARRIER_NEAR_ATR = float(os.getenv("WS_DT_GRAPH_TIMING_BARRIER_NEAR_ATR", "0.40"))
DT_GRAPH_TIMING_BARRIER_WAIT_SEC = float(os.getenv("WS_DT_GRAPH_TIMING_BARRIER_WAIT_SEC", "8.0"))
DT_FALSE_MOVE_WAIT_SEC = max(1.0, float(os.getenv("WS_DT_FALSE_MOVE_WAIT_SEC", "4.0")))
DT_FALSE_MOVE_WICK_PCT = float(os.getenv("WS_DT_FALSE_MOVE_WICK_PCT", "68.0"))
DT_FALSE_MOVE_TOUCH_MAX = float(os.getenv("WS_DT_FALSE_MOVE_TOUCH_MAX", "0.62"))
FULL_PATTERN_STUDY_LOCAL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "backtest_reports",
    "full_pattern_value_full",
    "full_pattern_value_study.json",
)
FULL_PATTERN_ASSET_WR_MIN = float(os.getenv("WS_FULL_PATTERN_ASSET_WR_MIN", "85.0"))
FULL_PATTERN_TOP_ASSETS = max(1, int(os.getenv("WS_FULL_PATTERN_TOP_ASSETS", "4")))
DT_STUDY_PROGRESS_PREMIUM = float(os.getenv("WS_DT_STUDY_PROGRESS_PREMIUM", "0.08"))
DT_STUDY_TOUCH_PREMIUM = float(os.getenv("WS_DT_STUDY_TOUCH_PREMIUM", "0.86"))
DT_STUDY_RELEASE_PROB_GAP = float(os.getenv("WS_DT_STUDY_RELEASE_PROB_GAP", "0.03"))
DT_STUDY_RELEASE_MIN_PROB = float(os.getenv("WS_DT_STUDY_RELEASE_MIN_PROB", "0.74"))


def _normalize_dt_live_profile(value) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "": "standard",
        "default": "standard",
        "normal": "standard",
        "padrao": "standard",
        "padrão": "standard",
        "balanced": "standard",
        "moderado": "moderate",
        "medio": "moderate",
        "médio": "moderate",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in {"standard", "moderate", "elite"}:
        return "standard"
    return normalized


DT_LIVE_PROFILE = _normalize_dt_live_profile(os.getenv("WS_DT_LIVE_PROFILE", "standard"))
DT_NN_MIN_FLOOR = max(0.30, min(0.98, float(os.getenv("WS_DT_NN_MIN_FLOOR", "0.40"))))

# ── Variáveis para Engine / IA ──
DECIDIR_ANTES_FECHAR_SEC = int(os.getenv("WS_DECIDIR_ANTES_FECHAR", "12"))
IA_ON = True  # IA SEMPRE ativa para H&S
AI_STATS_FILE = os.path.join(os.path.expanduser("~"), ".wstrader", "ws_ai_stats_hs.json")
AI_MIN_SAMPLES = 5
AI_CONF_MIN = 0.3
AI_MIN_PROB = 0.55  # CORRIGIDO: era 0.40 (permitia entradas com 40% prob = moeda)
DT_BAYES_FINAL_MIN = max(0.75, min(0.95, float(os.getenv("WS_DT_BAYES_FINAL_MIN", "0.75"))))
HORARIO_INICIO_MIN = 90    # 1h30 da manhã (1*60 + 30)
HORARIO_FIM_MIN    = 1080  # 18h00 (18*60)
MAX_DIST_OMBRO_ATR = 0.5  # CORRIGIDO: era 1.0 (muito longe do ombro D = entrada ruim)
MAX_DIST_NECKLINE_ATR = 0.25  # Distância máx ALÉM da neckline — se preço já ultrapassou muito, é tarde demais


# ═══════════════════════════════════════════════════════════════
# PARÂMETROS FIXOS — valores que rodaram bem (NN≥80%, EXP=2, cooldown=2min)
# ═══════════════════════════════════════════════════════════════
def _analyze_macro_trend(guard_df, atr_val, direcao):
    """Analisa tendência macro das últimas 15 velas.
    Retorna dict com métricas de tendência e se deve bloquear.
    """
    result = {
        "block": False,
        "reason": "",
        "consecutive": 0,
        "bearish_pct": 0.5,
        "drift_atr": 0.0,
        "macro_penalty": 0.0,
    }
    if guard_df is None or len(guard_df) < 15 or atr_val <= 0:
        return result

    closes = guard_df["close"].values
    opens = guard_df["open"].values
    n = len(closes)

    # Últimas 15 velas
    _w = min(15, n)
    c_w = closes[-_w:]
    o_w = opens[-_w:]

    # Velas bearish/bullish
    bearish = [c_w[i] < o_w[i] for i in range(_w)]
    bearish_pct = sum(bearish) / _w
    bullish_pct = 1.0 - bearish_pct

    # Consecutivas na mesma direção (de trás pra frente)
    consec = 0
    if _w >= 2:
        last_bear = bearish[-1]
        for i in range(_w - 1, -1, -1):
            if bearish[i] == last_bear:
                consec += 1
            else:
                break

    # Drift total (variação de preço / ATR)
    drift = (float(c_w[-1]) - float(c_w[0])) / atr_val

    # EMA rápida slope (últimas 10 velas)
    _ema_w = min(10, n)
    ema_closes = [float(closes[-(i+1)]) for i in range(_ema_w)][::-1]
    if len(ema_closes) >= 5:
        ema_first_half = sum(ema_closes[:len(ema_closes)//2]) / (len(ema_closes)//2)
        ema_second_half = sum(ema_closes[len(ema_closes)//2:]) / (len(ema_closes) - len(ema_closes)//2)
        slope = (ema_second_half - ema_first_half) / atr_val
    else:
        slope = 0.0

    result["consecutive"] = consec
    result["bearish_pct"] = bearish_pct
    result["drift_atr"] = drift

    # ── Classificar tendência e calcular penalidade ──
    is_call = direcao == "CALL"
    is_put = direcao == "PUT"

    # Score de força contra o trade (0-1)
    against_score = 0.0
    if is_call:
        # CALL contra tendência de baixa
        if drift < -1.0:
            against_score += min(abs(drift) / 5.0, 0.4)
        if bearish_pct >= 0.65:
            against_score += (bearish_pct - 0.65) * 2.0
        if consec >= 4 and bearish[-1]:
            against_score += min((consec - 3) * 0.08, 0.3)
    elif is_put:
        # PUT contra tendência de alta
        if drift > 1.0:
            against_score += min(abs(drift) / 5.0, 0.4)
        if bullish_pct >= 0.65:
            against_score += (bullish_pct - 0.65) * 2.0
        if consec >= 4 and not bearish[-1]:
            against_score += min((consec - 3) * 0.08, 0.3)

    against_score = min(against_score, 1.0)

    # Penalidade ao nn_score (proporcional à força contra)
    # against_score 0.3+ começa a penalizar, 0.6+ penaliza forte
    if against_score >= 0.3:
        result["macro_penalty"] = (against_score - 0.2) * 0.25

    # Hard block: tendência MUITO forte contra o trade
    # 7+ consecutivas na direção oposta OU 80%+ velas contra + drift > 2 ATR
    if is_call:
        if consec >= 7 and bearish[-1]:
            result["block"] = True
            result["reason"] = f"{consec} velas vermelhas consecutivas"
        elif bearish_pct >= 0.80 and drift < -2.0:
            result["block"] = True
            result["reason"] = f"{bearish_pct:.0%} bearish + queda {abs(drift):.1f}ATR"
        elif drift < -3.5:
            result["block"] = True
            result["reason"] = f"queda forte {abs(drift):.1f}ATR em {_w} velas"
    elif is_put:
        if consec >= 7 and not bearish[-1]:
            result["block"] = True
            result["reason"] = f"{consec} velas verdes consecutivas"
        elif bullish_pct >= 0.80 and drift > 2.0:
            result["block"] = True
            result["reason"] = f"{bullish_pct:.0%} bullish + subida {abs(drift):.1f}ATR"
        elif drift > 3.5:
            result["block"] = True
            result["reason"] = f"subida forte {abs(drift):.1f}ATR em {_w} velas"

    return result


def _snapshot_pattern_point(point: Optional[dict], df_index) -> Optional[dict]:
    if not point:
        return None
    snap = {
        "idx": int(point.get("idx", -1)),
        "price": round(float(point.get("price", 0.0) or 0.0), 6),
    }
    idx = snap["idx"]
    if 0 <= idx < len(df_index):
        try:
            snap["ts"] = int(df_index[idx].value // 10**9)
        except Exception:
            snap["ts"] = 0
    return snap


def _serialize_dashboard_pattern(ativo: str, pat: dict, df, ia_prob: float,
                                 backtest: Optional[dict] = None,
                                 nn_pred: Optional[dict] = None,
                                 scan_ts: Optional[float] = None,
                                 market_regime: Optional[dict] = None,
                                 entry_guard_pred: Optional[dict] = None,
                                 touch_continuation: Optional[dict] = None,
                                 prediction_2m: Optional[dict] = None,
                                 timing_hint: Optional[dict] = None,
                                 shadow_pattern_lib: Optional[dict] = None,
                                 gpt_result: Optional[dict] = None,
                                 ai_consensus: Optional[dict] = None) -> dict:
    df_index = df.index
    snap = {
        "ativo": ativo,
        "type": str(pat.get("type", "")),
        "direction": str(pat.get("direction", "")),
        "mode": str(pat.get("mode", "double_touch")),
        "neckline": round(float(pat.get("neckline", 0.0) or 0.0), 6),
        "neck_slope": round(float(pat.get("neck_slope", 0.0) or 0.0), 8),
        "depth": round(float(pat.get("depth", 0.0) or 0.0), 6),
        "target": round(float(pat.get("target", 0.0) or 0.0), 6),
        "stop": round(float(pat.get("stop", 0.0) or 0.0), 6),
        "entry_idx": int(pat.get("entry_idx", -1) or -1),
        "entry_price": round(float(pat.get("entry_price", 0.0) or 0.0), 6),
        "candles_ago": int(pat.get("candles_ago", 0) or 0),
        "entry_pending": bool(pat.get("entry_pending", False)),
        "ia_prob": round(float(ia_prob or 0.0), 3),
        "scan_ts": float(scan_ts or 0.0),
        "left_shoulder": _snapshot_pattern_point(pat.get("left_shoulder"), df_index),
        "head": _snapshot_pattern_point(pat.get("head"), df_index),
        "right_shoulder": _snapshot_pattern_point(pat.get("right_shoulder"), df_index),
        "valley1": _snapshot_pattern_point(pat.get("valley1"), df_index),
        "valley2": _snapshot_pattern_point(pat.get("valley2"), df_index),
    }
    entry_idx = snap["entry_idx"]
    if 0 <= entry_idx < len(df_index):
        try:
            snap["entry_ts"] = int(df_index[entry_idx].value // 10**9)
        except Exception:
            snap["entry_ts"] = 0

    if nn_pred is not None:
        snap["nn_approved"] = bool(nn_pred.get("approved", False))
        snap["nn_score"] = round(float(nn_pred.get("nn_score", nn_pred.get("prob_win", 0.0)) or 0.0), 3)
        snap["nn_p1"] = round(float(nn_pred.get("p1", 0.0) or 0.0), 3)
        snap["nn_p2"] = round(float(nn_pred.get("p2", 0.0) or 0.0), 3)
        p3 = nn_pred.get("p3")
        snap["nn_p3"] = round(float(p3), 3) if p3 is not None else None
        snap["nn_trained_metrics"] = nn_pred.get("trained_metrics")
    else:
        snap["nn_approved"] = None

    if entry_guard_pred is not None:
        snap["direction_alignment_2m"] = entry_guard_pred.get("direction_alignment_2m")

    if touch_continuation is not None:
        snap["touch_continuation"] = touch_continuation

    if prediction_2m is not None:
        snap["prediction_2m"] = prediction_2m

    if timing_hint is not None:
        snap["timing_hint"] = timing_hint

    if shadow_pattern_lib is not None:
        snap["shadow_pattern_lib"] = shadow_pattern_lib

    if gpt_result is not None:
        snap["gpt"] = gpt_result

    if ai_consensus is not None:
        snap["ai_consensus"] = ai_consensus

    if market_regime is not None:
        snap["market_regime"] = {
            "ok": bool(market_regime.get("ok", True)),
            "score": round(float(market_regime.get("score", 0.0) or 0.0), 3),
            "reason": str(market_regime.get("reason", "")),
        }

    if backtest is not None:
        bt = {
            "result": str(backtest.get("result", "")),
            "entry_price": round(float(backtest.get("entry_price", 0.0) or 0.0), 6),
            "exit_price": round(float(backtest.get("exit_price", 0.0) or 0.0), 6),
            "entry_idx": int(backtest.get("entry_idx", -1) or -1),
            "exit_idx": int(backtest.get("exit_idx", -1) or -1),
            "pips": round(float(backtest.get("pips", 0.0) or 0.0), 6),
        }
        if 0 <= bt["entry_idx"] < len(df_index):
            try:
                bt["entry_ts"] = int(df_index[bt["entry_idx"]].value // 10**9)
            except Exception:
                bt["entry_ts"] = 0
        if 0 <= bt["exit_idx"] < len(df_index):
            try:
                bt["exit_ts"] = int(df_index[bt["exit_idx"]].value // 10**9)
            except Exception:
                bt["exit_ts"] = 0
        snap["backtest"] = bt
    else:
        snap["backtest"] = None

    return snap


def _build_ai_cotrader_consensus(mode: str,
                                 ia_prob: float,
                                 nn_pred: Optional[dict] = None,
                                 gpt_result: Optional[dict] = None,
                                 shadow_pattern_lib: Optional[dict] = None) -> dict:
    mode_name = str(mode or "classic")
    bayes_prob = float(ia_prob or 0.0)
    bayes_min = DT_BAYES_FINAL_MIN if mode_name == "double_touch" else AI_MIN_PROB
    bayes_known = bayes_prob > 0 and abs(bayes_prob - 0.5) > 1e-9
    # DT: IA neural (entry_guard) já decide — Bayes não bloqueia
    if mode_name == "double_touch":
        bayes_ok = True
    else:
        bayes_ok = True if not bayes_known else bayes_prob >= bayes_min

    nn_available = isinstance(nn_pred, dict)
    nn_score = None
    nn_ok = None
    if nn_available:
        nn_score = float(nn_pred.get("nn_score", nn_pred.get("prob_win", 0.0)) or 0.0)
        if nn_pred.get("approved") is not None:
            nn_ok = bool(nn_pred.get("approved"))

    gpt_available = isinstance(gpt_result, dict) and bool(gpt_result.get("available"))
    gpt_ok = None
    gpt_conf = None
    gpt_reason = ""
    if isinstance(gpt_result, dict):
        if gpt_result.get("approved") is not None:
            gpt_ok = bool(gpt_result.get("approved"))
        if gpt_result.get("confidence") is not None:
            gpt_conf = float(gpt_result.get("confidence") or 0.0)
        gpt_reason = str(gpt_result.get("reason", "") or "")

    shadow_available = isinstance(shadow_pattern_lib, dict) and bool(shadow_pattern_lib.get("available"))
    shadow_agreement = None
    if shadow_available and shadow_pattern_lib.get("agreement") is not None:
        shadow_agreement = bool(shadow_pattern_lib.get("agreement"))

    gpt_blocks = False  # Context filter bloqueia diretamente via _all_guards_ok
    final_ok = bool(bayes_ok)

    if not bayes_ok:
        reason = f"Bayes abaixo do piso ({bayes_prob:.0%} < {bayes_min:.0%})"
        blocking_actor = "bayes"
    elif gpt_blocks:
        reason = f"IA generativa discordou com alta confianca ({gpt_conf:.0f}%)"
        blocking_actor = "gpt"
    else:
        parts = [f"Bayes {bayes_prob:.0%}"]
        if gpt_available:
            if gpt_ok is True:
                parts.append(f"Gen aprovou {gpt_conf:.0f}%")
            elif gpt_ok is False:
                parts.append(f"Gen advisory {gpt_conf:.0f}%")
        if shadow_available:
            parts.append("Shadow ok" if shadow_agreement else "Shadow divergente")
        reason = " | ".join(parts)
        blocking_actor = None

    return {
        "mode": mode_name,
        "bayes_prob": round(bayes_prob, 4),
        "bayes_min": round(bayes_min, 4),
        "bayes_ok": bayes_ok,
        "nn_available": nn_available,
        "nn_score": round(nn_score, 4) if nn_score is not None else None,
        "nn_ok": nn_ok,
        "gpt_available": gpt_available,
        "gpt_ok": gpt_ok,
        "gpt_confidence": round(gpt_conf, 2) if gpt_conf is not None else None,
        "shadow_available": shadow_available,
        "shadow_agreement": shadow_agreement,
        "final_ok": final_ok,
        "blocking_actor": blocking_actor,
        "reason": reason,
        "gpt_reason": gpt_reason,
    }


def _force_dt_entry_at_turn(timing_hint: Optional[dict], nn_pred: Optional[dict] = None) -> Optional[dict]:
    if not DT_ENTRY_AT_TURN or not isinstance(timing_hint, dict) or not timing_hint.get("available"):
        return timing_hint
    if timing_hint.get("action") not in {"wait"}:
        return timing_hint

    nn_score = None
    if isinstance(nn_pred, dict):
        nn_score = float(nn_pred.get("nn_score", nn_pred.get("prob_win", 0.0)) or 0.0)
        if nn_score < DT_NN_MIN_FLOOR:
            return timing_hint

    forced_reason = str(timing_hint.get("reason", "timing convertido para entrada na virada"))
    current_second = float(time.time() % 60.0)
    seconds_to_turn = max(0.0, 60.0 - current_second)
    if current_second <= max(0.8, min(MAX_ENTRY_DELAY_SEC, 2.0)):
        return {
            **timing_hint,
            "available": True,
            "action": "now",
            "label": "agora_virada",
            "wait_seconds": 0.0,
            "force_entry_at_turn": True,
            "reason": f"virada :00 confirmada | {forced_reason}",
        }

    return {
        **timing_hint,
        "available": True,
        "action": "wait",
        "label": "aguardar_virada",
        "wait_seconds": round(seconds_to_turn, 2),
        "force_entry_at_turn": True,
        "reason": f"aguardando virada :00 | {forced_reason}",
    }


def _build_shadow_dt_library_comparison(pat: dict, df: Optional[pd.DataFrame], atr_val: float) -> dict:
    pat = pat or {}
    result = {
        "enabled": SHADOW_PATTERN_LIB_ENABLED,
        "library": "tradingpatterns",
        "available": False,
        "agreement": False,
        "reason": None,
    }

    if not SHADOW_PATTERN_LIB_ENABLED:
        result["reason"] = "shadow library desabilitada"
        return result
    if not _HAS_SHADOW_PATTERN_LIB:
        result["reason"] = "tradingpatterns indisponivel"
        return result
    if df is None or len(df) < max(12, SHADOW_PATTERN_LIB_WINDOW + 4):
        result["reason"] = "candles insuficientes"
        return result

    pat_type = str(pat.get("type", "") or "")
    direction = str(pat.get("direction", "") or "")
    if pat_type not in {"DOUBLE_TOP", "DOUBLE_BOTTOM"} or direction not in {"PUT", "CALL"}:
        result["reason"] = "padrao interno nao eh DT/DB"
        return result

    try:
        shadow_df = pd.DataFrame({
            "Open": pd.Series(df["open"].astype(float).values, index=df.index),
            "High": pd.Series(df["high"].astype(float).values, index=df.index),
            "Low": pd.Series(df["low"].astype(float).values, index=df.index),
            "Close": pd.Series(df["close"].astype(float).values, index=df.index),
        }, index=df.index)
        shadow_out = _shadow_patterns_lib.detect_double_top_bottom(
            shadow_df.copy(),
            window=SHADOW_PATTERN_LIB_WINDOW,
            threshold=SHADOW_PATTERN_LIB_THRESHOLD,
        )
        if "double_pattern" not in shadow_out.columns:
            result["reason"] = "biblioteca nao retornou coluna double_pattern"
            return result

        candidates = []
        for bar_idx, label in enumerate(shadow_out["double_pattern"].tolist()):
            if not isinstance(label, str) or not label:
                continue
            label_norm = label.strip().lower()
            if label_norm == "double top":
                lib_type = "DOUBLE_TOP"
                lib_direction = "PUT"
            elif label_norm == "double bottom":
                lib_type = "DOUBLE_BOTTOM"
                lib_direction = "CALL"
            else:
                continue
            try:
                ts_val = int(df.index[bar_idx].value // 10**9)
            except Exception:
                ts_val = 0
            candidates.append({
                "label": label,
                "type": lib_type,
                "direction": lib_direction,
                "bar_idx": int(bar_idx),
                "ts": ts_val,
            })

        result["available"] = True
        result["candidate_count"] = len(candidates)
        result["internal_type"] = pat_type
        result["internal_direction"] = direction
        result["window"] = SHADOW_PATTERN_LIB_WINDOW
        result["threshold"] = SHADOW_PATTERN_LIB_THRESHOLD

        if not candidates:
            result["reason"] = "biblioteca nao encontrou DT/DB neste recorte"
            return result

        internal_idx = int((pat.get("right_shoulder") or {}).get("idx", pat.get("entry_idx", -1)) or -1)
        if internal_idx < 0:
            internal_idx = int(pat.get("entry_idx", -1) or -1)

        nearest = min(
            candidates,
            key=lambda item: (abs(int(item.get("bar_idx", -1)) - internal_idx), -int(item.get("bar_idx", -1))),
        )
        nearest_distance = abs(int(nearest.get("bar_idx", -1)) - internal_idx) if internal_idx >= 0 else 999
        agreement = bool(
            nearest.get("type") == pat_type
            and nearest.get("direction") == direction
            and nearest_distance <= SHADOW_PATTERN_LIB_MAX_BAR_DISTANCE
        )

        result.update({
            "agreement": agreement,
            "internal_bar_idx": internal_idx,
            "nearest": nearest,
            "nearest_distance_bars": int(nearest_distance),
            "recent_match": int(nearest.get("bar_idx", -1)) >= max(0, len(df) - SHADOW_PATTERN_LIB_MAX_BAR_DISTANCE - 2),
            "reason": (
                f"biblioteca concorda em {nearest_distance} barra(s)"
                if agreement else
                f"biblioteca diverge: {nearest.get('type')} {nearest.get('direction')} @ {nearest_distance} barra(s)"
            ),
        })
        return result
    except Exception as exc:
        result["reason"] = f"falha no shadow detector: {exc}"
        return result


def _get_session_params(guard_df=None, atr_val=0.0):
    """Retorna parâmetros de sessão conforme o perfil ao vivo selecionado."""
    if DT_LIVE_PROFILE == "elite":
        return {
            "profile": "elite",
            "nn_min_prob": max(0.82, DT_NN_MIN_FLOOR),
            "exp_minutes": 2,
            "cooldown_sec": 4 * 60,
        }
    if DT_LIVE_PROFILE == "moderate":
        return {
            "profile": "moderate",
            "nn_min_prob": max(0.78, DT_NN_MIN_FLOOR),
            "exp_minutes": 2,
            "cooldown_sec": 3 * 60,
        }
    return {
        "profile": "standard",
        "nn_min_prob": DT_NN_MIN_FLOOR,
        "exp_minutes": 2,
        "cooldown_sec": 2 * 60,
    }


def _compute_smart_exp(C, H, L, n, atr_val, nn_score, pat_data):
    """Calcula duração ideal (1 ou 2 min) com base na velocidade do preço.
    Se o movimento médio por candle M1 supera 30% do ATR E a NN está ≥85%,
    1 minuto é suficiente → mais rápido, menor exposição ao risco.
    Caso contrário, 2 minutos (padrão, alinhado com treino).
    """
    try:
        if nn_score is None or nn_score < 0.92:
            return EXP_FIXA  # 2 min (padrão — NN não é confiante o bastante para 1min)

        look = min(10, n - 1)
        if look < 3:
            return EXP_FIXA

        moves = []
        for k in range(n - look, n):
            moves.append(abs(float(C[k]) - float(C[k - 1])))
        avg_move = float(np.mean(moves)) if moves else 0

        if atr_val <= 0:
            return EXP_FIXA

        # Profundidade do padrão → impulso esperado
        depth = float((pat_data or {}).get("depth", 0))
        depth_ratio = depth / atr_val

        impulse = 0.3 + depth_ratio * 0.15 + nn_score * 0.4
        expected_1m = avg_move * min(impulse, 1.2)
        min_move = atr_val * 0.30

        if expected_1m >= min_move:
            return 1  # 1 minuto basta
        return EXP_FIXA  # 2 min (padrão)
    except Exception:
        return EXP_FIXA


def _dt_geometry_scan_filter(geo: Optional[dict], geom_score: Optional[float]) -> dict:
    def _between(value: float, low: float, high: float) -> bool:
        return low <= float(value) <= high

    def _study_profile(current_geo: Optional[dict]) -> dict:
        if current_geo is None:
            return {
                "positive_hits": 0,
                "negative_hits": 0,
                "hard_block": False,
                "win_reasons": [],
                "loss_reasons": [],
                "hard_reasons": [],
                "reason": "geometria indisponivel",
            }

        depth_ratio = float(current_geo.get("depth_ratio", 0.0) or 0.0)
        symmetry = float(current_geo.get("symmetry", 0.0) or 0.0)
        neck_align = float(current_geo.get("neck_align", 0.0) or 0.0)
        d_left = int(current_geo.get("d_left", 0) or 0)
        d_right = int(current_geo.get("d_right", 0) or 0)
        shoulder_ratio = float(current_geo.get("shoulder_ratio", 0.0) or 0.0)

        win_reasons = []
        loss_reasons = []
        hard_reasons = []

        if depth_ratio >= 5.796:
            win_reasons.append("depth_ratio muito forte")
        elif depth_ratio >= 3.783:
            win_reasons.append("depth_ratio acima do corredor win")
        if depth_ratio <= 2.838:
            loss_reasons.append("depth_ratio muito raso")

        if neck_align >= 0.364:
            win_reasons.append("neck_align alto")
        elif neck_align >= 0.196:
            win_reasons.append("neck_align alinhado")
        if neck_align <= 0.117:
            loss_reasons.append("neck_align fraco")

        if d_left >= 9 and d_left <= 12:
            win_reasons.append("d_left no corredor 9-12")
        elif d_left >= 25 and d_left <= 33:
            win_reasons.append("d_left longo controlado")
        if d_left <= 6:
            loss_reasons.append("d_left curto demais")
        elif d_left <= 8:
            loss_reasons.append("d_left curto")

        if d_right <= 6:
            win_reasons.append("d_right rapido")
        elif d_right >= 19 and d_right <= 25:
            win_reasons.append("d_right no corredor 19-25")
        if _between(d_right, 8, 10):
            loss_reasons.append("d_right fraco 8-10")
        elif _between(d_right, 15, 19):
            loss_reasons.append("d_right fraco 15-19")
        elif d_right >= 33:
            loss_reasons.append("d_right tardio demais")

        if symmetry <= 0.214:
            win_reasons.append("simetria baixa com edge")
        elif _between(symmetry, 0.444, 0.529):
            win_reasons.append("simetria media vencedora")
        elif _between(symmetry, 0.731, 0.857):
            win_reasons.append("simetria alta controlada")
        if _between(symmetry, 0.290, 0.364):
            loss_reasons.append("simetria fraca")
        elif symmetry >= 0.857:
            loss_reasons.append("simetria perfeita demais")

        if _between(shoulder_ratio, 0.999790, 0.999910):
            win_reasons.append("ombros alinhados no corredor win")
        if shoulder_ratio >= 0.999980:
            loss_reasons.append("ombros perfeitos demais")

        if d_left <= 6 and d_right <= 6:
            hard_reasons.append("d_left e d_right curtos ao mesmo tempo")
        if _between(neck_align, 0.0651, 0.1310) and _between(shoulder_ratio, 0.998870, 0.999790):
            hard_reasons.append("neck_align fraco com ombros desalinhados")
        if symmetry >= 0.769 and d_right <= 6:
            hard_reasons.append("simetria alta demais com d_right rapido")
        if symmetry >= 0.769 and depth_ratio <= 2.838:
            hard_reasons.append("simetria alta demais com depth_ratio raso")

        reason_parts = []
        if win_reasons:
            reason_parts.append(f"win_hits={len(win_reasons)}")
        if loss_reasons:
            reason_parts.append(f"loss_hits={len(loss_reasons)}")
        if hard_reasons:
            reason_parts.append("loss_corridor=" + ", ".join(hard_reasons[:2]))

        return {
            "positive_hits": len(win_reasons),
            "negative_hits": len(loss_reasons),
            "hard_block": bool(hard_reasons),
            "win_reasons": win_reasons,
            "loss_reasons": loss_reasons,
            "hard_reasons": hard_reasons,
            "reason": " | ".join(reason_parts) if reason_parts else "sem sinais geométricos relevantes",
        }

    profile = DT_LIVE_PROFILE
    if geo is None:
        return {"ok": profile != "elite", "reason": "geometria indisponível", "profile": profile}

    geom_score = float(geom_score or 0.0)
    study_profile = _study_profile(geo)
    strong_geo_hits = int(study_profile.get("positive_hits", 0) or 0)
    negative_geo_hits = int(study_profile.get("negative_hits", 0) or 0)

    if study_profile.get("hard_block"):
        return {
            "ok": False,
            "hard_block": True,
            "reason": f"loss-profile geometrico | {study_profile.get('reason')}",
            "profile": profile,
            "strong_geo_hits": strong_geo_hits,
            "negative_geo_hits": negative_geo_hits,
            "study_profile": study_profile,
        }

    if profile == "elite":
        if (geom_score >= 0.82 and strong_geo_hits >= 2 and negative_geo_hits == 0) or strong_geo_hits >= 3:
            return {
                "ok": True,
                "reason": f"elite ok | geom={geom_score:.2f} | win_hits={strong_geo_hits} | loss_hits={negative_geo_hits} | {study_profile.get('reason')}",
                "profile": profile,
                "strong_geo_hits": strong_geo_hits,
                "negative_geo_hits": negative_geo_hits,
                "study_profile": study_profile,
            }
        return {
            "ok": False,
            "reason": f"elite rejeitou | geom={geom_score:.2f} | win_hits={strong_geo_hits} | loss_hits={negative_geo_hits} | {study_profile.get('reason')}",
            "profile": profile,
            "strong_geo_hits": strong_geo_hits,
            "negative_geo_hits": negative_geo_hits,
            "study_profile": study_profile,
        }

    if profile == "moderate":
        if (geom_score >= 0.78 and strong_geo_hits >= 1 and negative_geo_hits <= 1) or strong_geo_hits >= 2:
            return {
                "ok": True,
                "reason": f"moderate ok | geom={geom_score:.2f} | win_hits={strong_geo_hits} | loss_hits={negative_geo_hits} | {study_profile.get('reason')}",
                "profile": profile,
                "strong_geo_hits": strong_geo_hits,
                "negative_geo_hits": negative_geo_hits,
                "study_profile": study_profile,
            }
        return {
            "ok": False,
            "reason": f"moderate rejeitou | geom={geom_score:.2f} | win_hits={strong_geo_hits} | loss_hits={negative_geo_hits} | {study_profile.get('reason')}",
            "profile": profile,
            "strong_geo_hits": strong_geo_hits,
            "negative_geo_hits": negative_geo_hits,
            "study_profile": study_profile,
        }

    return {
        "ok": True,
        "reason": f"standard advisory | geom={geom_score:.2f} | win_hits={strong_geo_hits} | loss_hits={negative_geo_hits} | {study_profile.get('reason')}",
        "profile": profile,
        "strong_geo_hits": strong_geo_hits,
        "negative_geo_hits": negative_geo_hits,
        "study_profile": study_profile,
    }


def _validate_dt_entry_region(pat: dict, current_price: float, atr_val: float) -> dict:
    try:
        direction = str(pat.get("direction", ""))
        touch_price = float((pat.get("right_shoulder") or {}).get("price", current_price) or current_price)
        neckline = float(pat.get("neckline", current_price) or current_price)
        atr_base = max(float(atr_val or 0.0), abs(float(current_price)) * 0.0005, 1e-6)
        dist_touch_atr = abs(float(current_price) - touch_price) / atr_base

        if direction == "CALL":
            move_total = max(neckline - touch_price, atr_base * 0.12)
            progress_pct = max(0.0, float(current_price) - touch_price) / move_total
            overshoot_neck_atr = max(0.0, float(current_price) - neckline) / atr_base
            wrong_side_atr = max(0.0, touch_price - float(current_price)) / atr_base
        else:
            move_total = max(touch_price - neckline, atr_base * 0.12)
            progress_pct = max(0.0, touch_price - float(current_price)) / move_total
            overshoot_neck_atr = max(0.0, neckline - float(current_price)) / atr_base
            wrong_side_atr = max(0.0, float(current_price) - touch_price) / atr_base

        if wrong_side_atr > 0.12:
            return {
                "ok": False,
                "ideal": False,
                "reason": f"preco rompeu o lado errado do 2o toque ({wrong_side_atr:.2f}ATR)",
                "dist_touch_atr": round(float(dist_touch_atr), 4),
                "progress_pct": round(float(progress_pct), 4),
                "overshoot_neck_atr": round(float(overshoot_neck_atr), 4),
            }

        if overshoot_neck_atr > MAX_DIST_NECKLINE_ATR:
            return {
                "ok": False,
                "ideal": False,
                "reason": f"preco passou da neckline ({overshoot_neck_atr:.2f}ATR > {MAX_DIST_NECKLINE_ATR:.2f})",
                "dist_touch_atr": round(float(dist_touch_atr), 4),
                "progress_pct": round(float(progress_pct), 4),
                "overshoot_neck_atr": round(float(overshoot_neck_atr), 4),
            }

        ideal = True
        reason = f"regiao ok | touch={dist_touch_atr:.2f}ATR | progress={progress_pct:.0%}"

        if dist_touch_atr > _DT_ENTRY_TOUCH_BAND_ATR:
            return {
                "ok": False,
                "ideal": False,
                "reason": (
                    f"longe do 2o toque ({dist_touch_atr:.2f}ATR > {_DT_ENTRY_TOUCH_BAND_ATR:.2f})"
                ),
                "dist_touch_atr": round(float(dist_touch_atr), 4),
                "progress_pct": round(float(progress_pct), 4),
                "overshoot_neck_atr": round(float(overshoot_neck_atr), 4),
            }

        if progress_pct > _DT_ENTRY_PROGRESS_MAX:
            return {
                "ok": False,
                "ideal": False,
                "reason": (
                    f"fora da zona ideal ({progress_pct:.0%} > {_DT_ENTRY_PROGRESS_MAX:.0%})"
                ),
                "dist_touch_atr": round(float(dist_touch_atr), 4),
                "progress_pct": round(float(progress_pct), 4),
                "overshoot_neck_atr": round(float(overshoot_neck_atr), 4),
            }

        return {
            "ok": True,
            "ideal": True,
            "reason": reason,
            "dist_touch_atr": round(float(dist_touch_atr), 4),
            "progress_pct": round(float(progress_pct), 4),
            "overshoot_neck_atr": round(float(overshoot_neck_atr), 4),
        }
    except Exception:
        return {"ok": False, "ideal": False, "reason": "falha ao validar regiao de entrada"}


def _detect_dt_false_move_wait(wick_pct: float,
                               touch_continuation: Optional[dict],
                               entry_region: Optional[dict],
                               entry_guard_pred: Optional[dict],
                               win_geometry_alignment: Optional[dict]) -> dict:
    touch_continuation = touch_continuation or {}
    entry_region = entry_region or {}
    entry_guard_pred = entry_guard_pred or {}
    win_geometry_alignment = win_geometry_alignment or {}

    wick_pct = float(wick_pct or 0.0)
    touch_strength = float(touch_continuation.get("strength", 0.0) or 0.0)
    partial = bool(touch_continuation.get("partial"))
    ideal = bool(entry_region.get("ideal"))
    prob_now = float(entry_guard_pred.get("prob_now", 0.0) or 0.0)
    accuracy = float(entry_guard_pred.get("accuracy", 0.0) or 0.0)
    precision = float(entry_guard_pred.get("precision", 0.0) or 0.0)
    auc = float(entry_guard_pred.get("auc", 0.0) or 0.0)
    weak_model = (
        accuracy < _ENTRY_GUARD_MIN_ACC
        or precision < _ENTRY_GUARD_MIN_PREC
        or auc < _ENTRY_GUARD_MIN_AUC
    )
    win_geom_reason = str(win_geometry_alignment.get("reason", "") or "")
    base_insufficient = "base geometrica insuficiente" in win_geom_reason.lower()

    reasons = []
    wait_seconds = 0.0

    if partial:
        reasons.append("continuidade parcial")
        wait_seconds = max(wait_seconds, DT_FALSE_MOVE_WAIT_SEC)

    if wick_pct >= max(78.0, DT_FALSE_MOVE_WICK_PCT + 10.0):
        reasons.append(f"wick extremo ({wick_pct:.0f}%)")
        wait_seconds = max(wait_seconds, DT_FALSE_MOVE_WAIT_SEC + 1.0)
    elif wick_pct >= DT_FALSE_MOVE_WICK_PCT and (not ideal or touch_strength <= max(0.68, DT_FALSE_MOVE_TOUCH_MAX + 0.04)):
        reasons.append(f"wick alto ({wick_pct:.0f}%)")
        wait_seconds = max(wait_seconds, DT_FALSE_MOVE_WAIT_SEC)

    if not ideal and touch_strength <= DT_FALSE_MOVE_TOUCH_MAX and prob_now < 0.90:
        reasons.append(
            f"fora da zona ideal com continuidade curta (touch={touch_strength:.2f}, eg={prob_now:.2f})"
        )
        wait_seconds = max(wait_seconds, DT_FALSE_MOVE_WAIT_SEC)

    if base_insufficient and wick_pct >= 60.0:
        reasons.append("sem base geometrica forte com wick alto")
        wait_seconds = max(wait_seconds, DT_FALSE_MOVE_WAIT_SEC)

    if weak_model and wick_pct >= 60.0:
        reasons.append(
            f"modelo fraco com wick alto (acc={accuracy:.1%}, prec={precision:.1%}, auc={auc:.3f})"
        )
        wait_seconds = max(wait_seconds, DT_FALSE_MOVE_WAIT_SEC)

    if not reasons:
        return {"wait": False, "seconds": 0.0, "reason": "sem indicio de falso movimento"}

    return {
        "wait": True,
        "seconds": round(float(wait_seconds or DT_FALSE_MOVE_WAIT_SEC), 2),
        "reason": " | ".join(reasons[:3]),
    }


def _dt_profile_runtime_filter(direction: str,
                               geo: Optional[dict],
                               geom_score: Optional[float],
                               wick_pct: float,
                               entry_guard_prob: float,
                               progress_pct: float,
                               target_room_atr: float,
                               touch_strength: float,
                               macro_ctx: Optional[dict],
                               win_signature: Optional[dict] = None,
                               study_multifactor: Optional[dict] = None) -> dict:
    profile = DT_LIVE_PROFILE
    if profile == "standard":
        return {"ok": True, "reason": "standard sem hard guard", "profile": profile, "score": None}

    geom_score = float(geom_score or 0.0)
    wick_pct = float(wick_pct or 0.0)
    entry_guard_prob = float(entry_guard_prob or 0.0)
    progress_pct = float(progress_pct or 0.0)
    target_room_atr = float(target_room_atr or 0.0)
    touch_strength = float(touch_strength or 0.0)
    geo = geo or {}
    macro_ctx = macro_ctx or {}
    win_signature = win_signature or {}
    study_multifactor = study_multifactor or {}

    if macro_ctx.get("block"):
        return {
            "ok": False,
            "reason": f"macro contra forte: {macro_ctx.get('reason', 'tendencia contraria')}",
            "profile": profile,
            "score": 0.0,
        }

    study_geo = _dt_geometry_scan_filter(geo, geom_score)
    strong_geo_hits = int(study_geo.get("strong_geo_hits", 0) or 0)
    negative_geo_hits = int(study_geo.get("negative_geo_hits", 0) or 0)

    if study_geo.get("hard_block"):
        return {
            "ok": False,
            "reason": study_geo.get("reason", "loss-profile geometrico"),
            "profile": profile,
            "score": 0.0,
            "strong_geo_hits": strong_geo_hits,
            "negative_geo_hits": negative_geo_hits,
            "direction": direction,
        }

    if study_multifactor.get("hard_block"):
        return {
            "ok": False,
            "reason": study_multifactor.get("reason", "loss-profile multifator"),
            "profile": profile,
            "score": 0.0,
            "strong_geo_hits": strong_geo_hits,
            "negative_geo_hits": negative_geo_hits,
            "direction": direction,
        }

    score = 0.0
    reasons = []
    positive_multi_hits = int(study_multifactor.get("positive_hits", 0) or 0)
    negative_multi_hits = int(study_multifactor.get("negative_hits", 0) or 0)

    if positive_multi_hits >= 4:
        score += 1.5
    elif positive_multi_hits >= 3:
        score += 1.0
    elif positive_multi_hits >= 2:
        score += 0.5
    else:
        reasons.append(f"multifator premium insuficiente ({positive_multi_hits})")

    if negative_multi_hits >= 2:
        score -= 1.0
        reasons.append(f"multifator em loss-case ({negative_multi_hits})")
    elif negative_multi_hits == 1:
        score -= 0.5
        reasons.append("1 loss-case multifator")

    if study_multifactor.get("trigger_release"):
        score += 0.5

    if strong_geo_hits >= 3:
        score += 1.0
    elif strong_geo_hits >= 2:
        score += 0.5
    else:
        reasons.append(f"geometria win insuficiente ({strong_geo_hits})")

    if negative_geo_hits >= 2:
        score -= 1.0
        reasons.append(f"geometria em loss-profile ({negative_geo_hits})")
    elif negative_geo_hits == 1:
        score -= 0.5
        reasons.append("1 alerta de loss-profile")

    if wick_pct >= 35.0:
        score += 1.0
    elif wick_pct >= 20.0 or (progress_pct <= 20.0 and touch_strength >= 0.62):
        score += 0.5
    else:
        reasons.append(f"wick fraco ({wick_pct:.1f}%)")

    if progress_pct <= 35.0:
        score += 1.0
    elif progress_pct <= 55.0:
        score += 0.5
    else:
        reasons.append(f"entrada adiantada demais ({progress_pct:.0f}% do caminho)")

    if target_room_atr >= 4.0:
        score += 1.0
    elif target_room_atr >= 3.0:
        score += 0.5
    else:
        reasons.append(f"alvo curto ({target_room_atr:.1f}ATR)")

    if entry_guard_prob >= 0.84:
        score += 1.0
    elif entry_guard_prob >= 0.78:
        score += 0.5
    else:
        reasons.append(f"entry_guard baixo ({entry_guard_prob:.2f})")

    if win_signature.get("matched"):
        if win_signature.get("tier") == "tight":
            score += 1.0
        else:
            score += 0.5
    elif entry_guard_prob < 0.90:
        reasons.append("sem win signature")

    if touch_strength >= 0.62:
        score += 1.0
    elif touch_strength >= 0.55:
        score += 0.5
    else:
        reasons.append(f"continuidade fraca ({touch_strength:.2f})")

    macro_penalty = float(macro_ctx.get("macro_penalty", 0.0) or 0.0)
    if macro_penalty >= 0.08:
        score -= 1.0
        reasons.append(f"macro_penalty={macro_penalty:.2f}")
    elif macro_penalty >= 0.04:
        score -= 0.5

    if profile == "standard":
        return {
            "ok": True,
            "reason": (
                f"standard advisory | score={score:.1f}/4.0 | geom_hits={strong_geo_hits} | loss_hits={negative_geo_hits} | multi_hits={positive_multi_hits} | multi_loss={negative_multi_hits} | "
                f"wick={wick_pct:.0f}% | progress={progress_pct:.0f}% | alvo={target_room_atr:.1f}ATR | touch={touch_strength:.2f} | eg={entry_guard_prob:.2f}"
            ),
            "profile": profile,
            "score": round(score, 3),
            "strong_geo_hits": strong_geo_hits,
            "macro_penalty": round(macro_penalty, 3),
            "direction": direction,
        }

    min_score = 5.0 if profile == "elite" else 4.0
    ok = score >= min_score
    reason = (
        f"{profile} score={score:.1f}/{min_score:.1f} | geom_hits={strong_geo_hits} | loss_hits={negative_geo_hits} | multi_hits={positive_multi_hits} | multi_loss={negative_multi_hits} | "
        f"wick={wick_pct:.0f}% | progress={progress_pct:.0f}% | alvo={target_room_atr:.1f}ATR | touch={touch_strength:.2f} | eg={entry_guard_prob:.2f}"
    )
    if not ok and reasons:
        reason += " | " + "; ".join(reasons[:3])

    return {
        "ok": ok,
        "reason": reason,
        "profile": profile,
        "score": round(score, 3),
        "strong_geo_hits": strong_geo_hits,
        "macro_penalty": round(macro_penalty, 3),
        "direction": direction,
    }


# ── Ativos fixos — melhores ativos (OTC + REAL com volume) ──
# Ranking: EURJPY-OTC 56.7% | AUDCAD-OTC 56.1% | EURGBP 55.0%
# EURUSD 50.8% | USDCHF 50.7% | EURGBP-OTC 50.0% | EURJPY 50.0%
# TOP 20 pares OTC — TODOS com 100% WR no backtest de 5000 velas (modo classic)
# Selecionados por ranking: WR desc + volume de padrões desc
FIXED_ASSETS = {
    "iq": [
        "GBPCAD-OTC", "USDJPY-OTC", "AUDNZD-OTC", "USDCAD-OTC", "USDCHF-OTC",
        "CADCHF-OTC", "EURAUD-OTC", "EURJPY-OTC", "EURNZD-OTC", "GBPCHF-OTC",
        "GBPNZD-OTC", "GBPUSD-OTC", "NZDJPY-OTC", "USDHKD-OTC", "AUDCHF-OTC",
        "AUDJPY-OTC", "GBPAUD-OTC", "USDZAR-OTC", "AUDCAD-OTC", "EURCAD-OTC",
    ],
    "bullex": [
        "GBPCAD-OTC", "USDJPY-OTC", "AUDNZD-OTC", "USDCAD-OTC", "USDCHF-OTC",
        "CADCHF-OTC", "EURAUD-OTC", "EURJPY-OTC", "EURNZD-OTC", "GBPCHF-OTC",
        "GBPNZD-OTC", "GBPUSD-OTC", "NZDJPY-OTC", "USDHKD-OTC", "AUDCHF-OTC",
        "AUDJPY-OTC", "GBPAUD-OTC", "USDZAR-OTC", "AUDCAD-OTC", "EURCAD-OTC",
    ],
    "casatrader": [
        "GBPCAD-OTC", "USDJPY-OTC", "AUDNZD-OTC", "USDCAD-OTC", "USDCHF-OTC",
        "CADCHF-OTC", "EURAUD-OTC", "EURJPY-OTC", "EURNZD-OTC", "GBPCHF-OTC",
        "GBPNZD-OTC", "GBPUSD-OTC", "NZDJPY-OTC", "USDHKD-OTC", "AUDCHF-OTC",
        "AUDJPY-OTC", "GBPAUD-OTC", "USDZAR-OTC", "AUDCAD-OTC", "EURCAD-OTC",
    ],
}

# ── Diretórios ──
_broker_suffix = BROKER_TYPE.replace("iq_option", "iq")
_user_data_dir = get_ws_user_data_dir()
os.makedirs(_user_data_dir, exist_ok=True)
_full_pattern_study_cache = None
_full_pattern_study_assets_cache = None


def _bundled_data_dir(folder_name: str) -> str:
    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, folder_name)


def _copy_bundled_file_if_missing(file_name: str, folder_name: str) -> bool:
    try:
        target_path = os.path.join(_user_data_dir, file_name)
        if os.path.exists(target_path):
            return True
        bundled_path = os.path.join(_bundled_data_dir(folder_name), file_name)
        if not os.path.exists(bundled_path):
            return False
        with open(bundled_path, "rb") as src, open(target_path, "wb") as dst:
            dst.write(src.read())
        log.info(paint(f"📦 Arquivo embarcado copiado: {file_name}", C.G))
        return True
    except Exception as ex:
        log.warning(paint(f"⚠️ Falha ao copiar arquivo embarcado {file_name}: {ex}", C.Y))
        return False


def _load_full_pattern_value_study() -> dict:
    global _full_pattern_study_cache
    if isinstance(_full_pattern_study_cache, dict):
        return _full_pattern_study_cache

    payload = {}
    try:
        if os.path.exists(FULL_PATTERN_STUDY_LOCAL):
            with open(FULL_PATTERN_STUDY_LOCAL, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                payload = data
    except Exception:
        payload = {}

    _full_pattern_study_cache = payload
    return payload


def _get_full_pattern_qualified_assets(min_wr: Optional[float] = None) -> List[str]:
    global _full_pattern_study_assets_cache
    threshold = float(FULL_PATTERN_ASSET_WR_MIN if min_wr is None else min_wr)
    if isinstance(_full_pattern_study_assets_cache, list) and min_wr is None:
        return list(_full_pattern_study_assets_cache)

    payload = _load_full_pattern_value_study()
    assets_payload = payload.get("assets", {}) if isinstance(payload, dict) else {}
    ranked = []
    if isinstance(assets_payload, dict):
        for asset, stats in assets_payload.items():
            wr = float((stats or {}).get("win_rate", 0.0) or 0.0)
            if wr >= threshold:
                ranked.append((str(asset), wr))
    ranked.sort(key=lambda item: item[1], reverse=True)
    qualified = [asset for asset, _ in ranked]

    if min_wr is None:
        _full_pattern_study_assets_cache = list(qualified)
    return qualified


def _get_full_pattern_top_assets(limit: Optional[int] = None) -> List[str]:
    max_items = FULL_PATTERN_TOP_ASSETS if limit is None else max(1, int(limit))
    return _get_full_pattern_qualified_assets()[:max_items]


def _filter_assets_by_full_pattern_study(assets: List[str]) -> List[str]:
    qualified = set(_get_full_pattern_qualified_assets())
    if not qualified:
        return list(assets)
    filtered = [asset for asset in assets if asset in qualified]
    return filtered if filtered else list(assets)


def _classify_dt_signal_candle(pat: dict, df: Optional[pd.DataFrame]) -> dict:
    if df is None or len(df) < 1:
        return {
            "label": "unknown",
            "body_ratio": 0.0,
            "rejection_wick": 0.0,
            "close_pos": 0.0,
        }

    try:
        row = df.iloc[-1]
        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])
        candle_range = max(high_price - low_price, 1e-8)
        body_ratio = abs(close_price - open_price) / candle_range
        direction = str(pat.get("direction", "CALL"))

        if direction == "CALL":
            rejection_wick = max(min(open_price, close_price) - low_price, 0.0) / candle_range
            close_pos = (close_price - low_price) / candle_range
            aligned_body = close_price > open_price
        else:
            rejection_wick = max(high_price - max(open_price, close_price), 0.0) / candle_range
            close_pos = (high_price - close_price) / candle_range
            aligned_body = close_price < open_price

        if body_ratio < 0.18 and rejection_wick >= 0.35:
            label = "pin_rejection"
        elif body_ratio < 0.12:
            label = "doji_indecision"
        elif aligned_body and body_ratio >= 0.45 and close_pos >= 0.70:
            label = "full_body_confirm"
        elif rejection_wick >= 0.35 and close_pos >= 0.60:
            label = "wick_rejection_confirm"
        elif aligned_body and body_ratio >= 0.25:
            label = "body_confirm"
        else:
            label = "weak_or_mixed"

        return {
            "label": label,
            "body_ratio": round(float(body_ratio), 4),
            "rejection_wick": round(float(rejection_wick), 4),
            "close_pos": round(float(close_pos), 4),
        }
    except Exception:
        return {
            "label": "unknown",
            "body_ratio": 0.0,
            "rejection_wick": 0.0,
            "close_pos": 0.0,
        }


def _dt_multifactor_study_profile(pat: dict,
                                  df: Optional[pd.DataFrame],
                                  entry_region: Optional[dict],
                                  touch_continuation: Optional[dict],
                                  entry_guard_pred: Optional[dict] = None) -> dict:
    entry_region = entry_region or {}
    touch_continuation = touch_continuation or {}
    entry_guard_pred = entry_guard_pred or {}

    candle_profile = _classify_dt_signal_candle(pat, df)
    signal_label = str(candle_profile.get("label", "unknown"))
    touch_state = "partial" if touch_continuation.get("partial") else ("confirmed" if touch_continuation.get("matched") else "missing")
    if entry_region.get("ok") and entry_region.get("ideal"):
        entry_state = "ideal"
    elif entry_region.get("ok"):
        entry_state = "outside_ideal"
    else:
        entry_state = "invalid"

    progress_pct = float(entry_region.get("progress_pct", 0.0) or 0.0)
    touch_strength = float(touch_continuation.get("strength", 0.0) or 0.0)
    prob_now = float(entry_guard_pred.get("prob_now", 0.0) or 0.0) if isinstance(entry_guard_pred, dict) else 0.0
    threshold = float(entry_guard_pred.get("threshold", 0.0) or 0.0) if isinstance(entry_guard_pred, dict) else 0.0

    positive_cases = []
    negative_cases = []
    hard_reasons = []

    if signal_label == "body_confirm" and entry_state == "ideal":
        positive_cases.append("body_confirm + ideal")
    if touch_state == "confirmed" and entry_state == "ideal":
        positive_cases.append("confirmed + ideal")
    if signal_label == "full_body_confirm" and touch_state == "confirmed":
        positive_cases.append("full_body_confirm + confirmed")
    if progress_pct <= DT_STUDY_PROGRESS_PREMIUM:
        positive_cases.append("progress precoce <= 8%")
    if touch_strength >= DT_STUDY_TOUCH_PREMIUM:
        positive_cases.append("touch_strength premium")

    if signal_label == "weak_or_mixed" and entry_state == "outside_ideal":
        negative_cases.append("weak_or_mixed + outside_ideal")
        hard_reasons.append("vela fraca fora da zona ideal")
    if signal_label == "weak_or_mixed" and touch_state == "partial":
        negative_cases.append("weak_or_mixed + partial")
        hard_reasons.append("vela fraca com continuidade parcial")
    if signal_label == "wick_rejection_confirm" and touch_state == "partial":
        negative_cases.append("wick_rejection_confirm + partial")

    premium = bool(len(positive_cases) >= 2 or (signal_label == "full_body_confirm" and touch_state == "confirmed"))
    trigger_release = bool(
        premium
        and touch_state == "confirmed"
        and not touch_continuation.get("partial")
        and prob_now >= max(DT_STUDY_RELEASE_MIN_PROB, threshold - DT_STUDY_RELEASE_PROB_GAP)
    )
    score_boost = 0.0
    if len(positive_cases) >= 4:
        score_boost = 0.05
    elif len(positive_cases) >= 3:
        score_boost = 0.04
    elif len(positive_cases) >= 2:
        score_boost = 0.025

    reason_parts = []
    if positive_cases:
        reason_parts.append("win_cases=" + ", ".join(positive_cases[:3]))
    if negative_cases:
        reason_parts.append("loss_cases=" + ", ".join(negative_cases[:2]))
    if trigger_release:
        reason_parts.append(f"release_prob={prob_now:.1%}")
    if not reason_parts:
        reason_parts.append("sem match forte no estudo multifator")

    return {
        "ok": not bool(hard_reasons),
        "premium": premium,
        "hard_block": bool(hard_reasons),
        "trigger_release": trigger_release,
        "score_boost": round(float(score_boost), 4),
        "positive_hits": int(len(positive_cases)),
        "negative_hits": int(len(negative_cases)),
        "positive_cases": positive_cases,
        "negative_cases": negative_cases,
        "hard_reasons": hard_reasons,
        "signal_candle_class": signal_label,
        "touch_state": touch_state,
        "entry_region_state": entry_state,
        "body_ratio": candle_profile.get("body_ratio", 0.0),
        "rejection_wick": candle_profile.get("rejection_wick", 0.0),
        "close_pos": candle_profile.get("close_pos", 0.0),
        "reason": " | ".join(reason_parts),
    }


def _seed_bundled_models() -> None:
    for folder_name, prefix in (("models", "reversal_tf_"), ("models_entry_guard", "entry_guard_")):
        try:
            bundled_dir = _bundled_data_dir(folder_name)
            if not os.path.isdir(bundled_dir):
                continue
            copied = 0
            for file_name in os.listdir(bundled_dir):
                if not file_name.startswith(prefix) or not file_name.endswith(".pkl"):
                    continue
                if _copy_bundled_file_if_missing(file_name, folder_name):
                    copied += 1
            if copied:
                log.info(paint(f"📦 Modelos embarcados disponíveis de {folder_name}: {copied}", C.G))
        except Exception as ex:
            log.warning(paint(f"⚠️ Falha ao inicializar modelos embarcados de {folder_name}: {ex}", C.Y))

_ENTRY_GUARD_ENABLED = (os.getenv("WS_ENTRY_GUARD", "1").strip() == "1")
_ENTRY_GUARD_POOL_SIZE = max(NUM_ATIVOS, int(os.getenv("WS_ENTRY_GUARD_POOL", "31")))
_ENTRY_GUARD_MIN_ACC = float(os.getenv("WS_ENTRY_GUARD_MIN_ACC", "0.85"))
_ENTRY_GUARD_MIN_AUC = float(os.getenv("WS_ENTRY_GUARD_MIN_AUC", "0.72"))
_ENTRY_GUARD_MIN_PREC = float(os.getenv("WS_ENTRY_GUARD_MIN_PREC", "0.92"))
_ENTRY_GUARD_MIN_SAMPLES = max(500, int(os.getenv("WS_ENTRY_GUARD_MIN_SAMPLES", "1800")))
_ENTRY_GUARD_FALLBACK_MIN_ACC = float(os.getenv("WS_ENTRY_GUARD_FALLBACK_MIN_ACC", "0.73"))
_ENTRY_GUARD_FALLBACK_MIN_AUC = float(os.getenv("WS_ENTRY_GUARD_FALLBACK_MIN_AUC", "0.74"))
_ENTRY_GUARD_FALLBACK_MIN_PREC = float(os.getenv("WS_ENTRY_GUARD_FALLBACK_MIN_PREC", "0.90"))
_ENTRY_GUARD_TRIGGER_MIN_PROB = max(0.75, min(0.95, float(os.getenv("WS_ENTRY_GUARD_TRIGGER_MIN_PROB", "0.75"))))
_ENTRY_GUARD_TRIGGER_STRONG_PROB = max(_ENTRY_GUARD_TRIGGER_MIN_PROB, min(0.98, float(os.getenv("WS_ENTRY_GUARD_TRIGGER_STRONG_PROB", "0.82"))))
_ENTRY_WIN_SIG_TOUCH_ATR = float(os.getenv("WS_ENTRY_WIN_SIG_TOUCH_ATR", "0.25"))
_ENTRY_WIN_SIG_PROGRESS = float(os.getenv("WS_ENTRY_WIN_SIG_PROGRESS", "0.15"))
_ENTRY_WIN_SIG_TIGHT_TOUCH_ATR = float(os.getenv("WS_ENTRY_WIN_SIG_TIGHT_TOUCH_ATR", "0.18"))
_ENTRY_WIN_SIG_TIGHT_PROGRESS = float(os.getenv("WS_ENTRY_WIN_SIG_TIGHT_PROGRESS", "0.10"))
_ENTRY_WIN_SIG_BONUS = float(os.getenv("WS_ENTRY_WIN_SIG_BONUS", "0.06"))
_ENTRY_WIN_SIG_TIGHT_BONUS = float(os.getenv("WS_ENTRY_WIN_SIG_TIGHT_BONUS", "0.09"))
_DT_WIN_GEOMETRY_MIN_SAMPLES = max(12, int(os.getenv("WS_DT_WIN_GEOMETRY_MIN_SAMPLES", "18")))
_DT_ENTRY_TOUCH_BAND_ATR = min(
    MAX_DIST_OMBRO_ATR,
    max(_ENTRY_WIN_SIG_TOUCH_ATR, float(os.getenv("WS_DT_ENTRY_TOUCH_BAND_ATR", "0.32")))
)
_DT_ENTRY_PROGRESS_MAX = max(
    _ENTRY_WIN_SIG_PROGRESS,
    min(0.45, float(os.getenv("WS_DT_ENTRY_PROGRESS_MAX", "0.22")))
)
_ASSET_SELECTION_CANDLES = max(120, min(LIVE_SCAN_N_M1, int(os.getenv("WS_ASSET_SELECTION_CANDLES", "180"))))
_entry_guard_cache: Dict[str, Optional[dict]] = {}

# ── Cache compartilhado com o Dashboard (o bot escreve, dashboard lê) ──
_DASHBOARD_CACHE_FILE = os.path.join(_user_data_dir, "ws_dashboard_cache.json")

# ── Cache LIVE de velas: bot escreve a cada 1s (streaming real-time), dashboard lê ──
_LIVE_CANDLE_FILE = os.path.join(_user_data_dir, "ws_live_candles.json")

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


def _fmt_pct(value: Optional[float], digits: int = 0, fallback: str = "n/a") -> str:
    try:
        return f"{float(value):.{digits}%}"
    except (TypeError, ValueError):
        return fallback


def _fmt_num(value: Optional[float], digits: int = 2, fallback: str = "n/a") -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return fallback


def _ensure_dashboard_server():
    """Garante que o dashboard HTTP esteja ativo na porta 8899."""
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        sock.connect(("127.0.0.1", 8899))
        sock.close()
        log.info(paint("📊 Dashboard DT já rodando na porta 8899", C.G))
        return
    except Exception:
        pass

    def _run_dashboard():
        try:
            import argparse
            import importlib

            dash_mod = importlib.import_module("dashboard_hs_ia")
            _orig_parse = argparse.ArgumentParser.parse_args

            def _fake_parse(self, args=None, namespace=None):
                return _orig_parse(self, args=[], namespace=namespace)

            argparse.ArgumentParser.parse_args = _fake_parse
            try:
                dash_mod.main()
            finally:
                argparse.ArgumentParser.parse_args = _orig_parse
        except Exception as ex:
            log.warning(paint(f"⚠️ Falha ao iniciar dashboard: {ex}", C.Y))

    threading.Thread(target=_run_dashboard, daemon=True, name="dashboard-hs-ia").start()
    log.info(paint("📊 Dashboard DT iniciado automaticamente na porta 8899", C.G))


def _entry_guard_model_path(ativo: str) -> str:
    return os.path.join(_user_data_dir, f"entry_guard_{ativo}.pkl")


def _reversal_model_path(ativo: str) -> str:
    path = find_existing_reversal_model_path(ativo)
    return path if os.path.exists(path) else get_reversal_model_persist_path(ativo)


def _load_entry_guard_bundle(ativo: str) -> Optional[dict]:
    if ativo in _entry_guard_cache:
        return _entry_guard_cache[ativo]

    model_path = _entry_guard_model_path(ativo)
    if not os.path.exists(model_path):
        _copy_bundled_file_if_missing(os.path.basename(model_path), "models_entry_guard")
    if not os.path.exists(model_path):
        _entry_guard_cache[ativo] = None
        return None

    try:
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        if isinstance(bundle, dict) and bundle.get("models"):
            _entry_guard_cache[ativo] = bundle
            return bundle
    except Exception as ex:
        log.warning(paint(f"⚠️ EntryGuard {ativo}: falha ao carregar modelo ({ex})", C.Y))

    _entry_guard_cache[ativo] = None
    return None


def _get_entry_guard_live_threshold(bundle: dict) -> float:
    metrics = bundle.get("metrics", {}) if isinstance(bundle, dict) else {}
    recommended = metrics.get("recommended_threshold", {}) if isinstance(metrics, dict) else {}
    base_threshold = float(recommended.get("threshold", 0.65) or 0.65)
    profile_bump = float(os.getenv("WS_ENTRY_GUARD_PROFILE_BUMP", "0.00") or 0.0)
    return max(0.50, min(0.95, base_threshold + profile_bump))


def _detect_entry_win_signature(feature_map: dict) -> dict:
    if not isinstance(feature_map, dict):
        return {"matched": False, "tier": None, "bonus": 0.0, "reason": "sem features"}

    touch_atr = float(feature_map.get("entry_to_touch_atr", 999.0) or 999.0)
    progress_pct = float(feature_map.get("progress_pct", 999.0) or 999.0)

    if touch_atr <= _ENTRY_WIN_SIG_TIGHT_TOUCH_ATR and progress_pct <= _ENTRY_WIN_SIG_TIGHT_PROGRESS:
        return {
            "matched": True,
            "tier": "tight",
            "bonus": round(float(_ENTRY_WIN_SIG_TIGHT_BONUS), 4),
            "historical_wr": 0.9978,
            "reason": f"touch={touch_atr:.3f}ATR <= {_ENTRY_WIN_SIG_TIGHT_TOUCH_ATR:.2f} | progress={progress_pct:.1%} <= {_ENTRY_WIN_SIG_TIGHT_PROGRESS:.0%}",
        }

    if touch_atr <= _ENTRY_WIN_SIG_TOUCH_ATR and progress_pct <= _ENTRY_WIN_SIG_PROGRESS:
        return {
            "matched": True,
            "tier": "core",
            "bonus": round(float(_ENTRY_WIN_SIG_BONUS), 4),
            "historical_wr": 0.9951,
            "reason": f"touch={touch_atr:.3f}ATR <= {_ENTRY_WIN_SIG_TOUCH_ATR:.2f} | progress={progress_pct:.1%} <= {_ENTRY_WIN_SIG_PROGRESS:.0%}",
        }

    return {
        "matched": False,
        "tier": None,
        "bonus": 0.0,
        "historical_wr": None,
        "reason": f"touch={touch_atr:.3f}ATR | progress={progress_pct:.1%}",
    }


def _entry_guard_past_win_trigger(pred: Optional[dict], touch_continuation: Optional[dict]) -> dict:
    if not isinstance(pred, dict):
        return {"triggered": False, "reason": "entry_guard indisponivel", "min_prob": None}

    prob_now = float(pred.get("prob_now", 0.0) or 0.0)
    threshold = float(pred.get("threshold", 0.0) or 0.0)
    _delay_raw = pred.get("delay_candles", 99)
    delay_candles = 99 if _delay_raw is None else int(_delay_raw)
    min_prob = max(0.50, threshold)
    strong_prob_floor = max(min_prob, 0.78)
    touch_strength = float(((touch_continuation or {}).get("strength", 0.0) or 0.0))
    win_signature = pred.get("win_signature") if isinstance(pred.get("win_signature"), dict) else {}
    feature_map = pred.get("feature_map") if isinstance(pred.get("feature_map"), dict) else {}
    entry_to_touch_atr = float(feature_map.get("entry_to_touch_atr", 999.0) or 999.0)
    progress_pct = float(feature_map.get("progress_pct", 999.0) or 999.0)
    outside_touch_band = entry_to_touch_atr > _DT_ENTRY_TOUCH_BAND_ATR
    outside_progress_band = progress_pct > _DT_ENTRY_PROGRESS_MAX
    edge = max(0.0, prob_now - min_prob)
    premium_zone_prob_floor = min(0.92, max(strong_prob_floor, min_prob + 0.20))

    if win_signature.get("matched") and prob_now >= min_prob:
        return {
            "triggered": True,
            "reason": f"win_signature {win_signature.get('tier')} + prob={prob_now:.1%}",
            "min_prob": round(min_prob, 4),
        }

    if outside_touch_band or outside_progress_band:
        if prob_now < premium_zone_prob_floor:
            return {
                "triggered": False,
                "reason": (
                    "fora da zona ideal sem prob premium | "
                    f"prob={prob_now:.1%} < {premium_zone_prob_floor:.1%} | "
                    f"touch={entry_to_touch_atr:.2f}ATR | progress={progress_pct:.0%}"
                ),
                "min_prob": round(min_prob, 4),
            }
        if edge < 0.20:
            return {
                "triggered": False,
                "reason": (
                    "fora da zona ideal com edge curto | "
                    f"edge={edge:.1%} | touch={entry_to_touch_atr:.2f}ATR | progress={progress_pct:.0%}"
                ),
                "min_prob": round(min_prob, 4),
            }
        if touch_strength < 0.58:
            return {
                "triggered": False,
                "reason": (
                    "fora da zona ideal com continuidade fraca | "
                    f"touch_strength={touch_strength:.2f} | touch={entry_to_touch_atr:.2f}ATR | progress={progress_pct:.0%}"
                ),
                "min_prob": round(min_prob, 4),
            }

    if prob_now >= max(strong_prob_floor, _ENTRY_GUARD_TRIGGER_STRONG_PROB - 0.02) and delay_candles <= 1 and touch_strength >= 0.55:
        return {
            "triggered": True,
            "reason": f"prob forte={prob_now:.1%} | delay={delay_candles} | touch={touch_strength:.2f}",
            "min_prob": round(min_prob, 4),
        }

    if prob_now >= strong_prob_floor and delay_candles == 0 and touch_strength >= 0.52:
        return {
            "triggered": True,
            "reason": f"timing exato | prob={prob_now:.1%} | touch={touch_strength:.2f}",
            "min_prob": round(min_prob, 4),
        }

    if prob_now >= min_prob and delay_candles == 0 and touch_strength >= 0.48:
        return {
            "triggered": True,
            "reason": f"timing aceitavel | prob={prob_now:.1%} | touch={touch_strength:.2f}",
            "min_prob": round(min_prob, 4),
        }

    missing = []
    if prob_now < min_prob:
        missing.append(f"prob={prob_now:.1%} < min={min_prob:.1%}")
    if delay_candles > 1:
        missing.append(f"delay={delay_candles}")
    if touch_strength < 0.48:
        missing.append(f"touch={touch_strength:.2f} < 0.48")

    return {
        "triggered": False,
        "reason": "sem trigger de win passado | " + (" | ".join(missing) if missing else f"combinacao fraca de timing (delay={delay_candles}, touch={touch_strength:.2f})"),
        "min_prob": round(min_prob, 4),
    }


def _entry_guard_quality_risk_filter(pred: Optional[dict],
                                     touch_continuation: Optional[dict],
                                     entry_region: Optional[dict],
                                     win_geometry_alignment: Optional[dict]) -> dict:
    if not isinstance(pred, dict):
        return {"ok": True, "reason": "entry_guard indisponivel"}

    touch_continuation = touch_continuation or {}
    entry_region = entry_region or {}
    win_geometry_alignment = win_geometry_alignment or {}

    accuracy = float(pred.get("accuracy", 0.0) or 0.0)
    precision = float(pred.get("precision", 0.0) or 0.0)
    auc = float(pred.get("auc", 0.0) or 0.0)
    prob_now = float(pred.get("prob_now", 0.0) or 0.0)
    threshold = float(pred.get("threshold", 0.0) or 0.0)
    partial = bool(touch_continuation.get("partial"))
    ideal = bool(entry_region.get("ideal"))
    win_geom_reason = str(win_geometry_alignment.get("reason", "") or "")
    base_insufficient = "base geometrica insuficiente" in win_geom_reason.lower()
    strong_win_base = bool(win_geometry_alignment.get("ok")) and not base_insufficient
    weak_model = (
        accuracy < _ENTRY_GUARD_MIN_ACC
        or precision < _ENTRY_GUARD_MIN_PREC
        or auc < _ENTRY_GUARD_MIN_AUC
    )

    if partial and ideal:
        return {
            "ok": False,
            "reason": "continuidade parcial na zona ideal - aguardar confirmacao completa",
        }

    if partial and not ideal and weak_model:
        premium_prob = max(0.85, threshold + 0.25)
        if strong_win_base and precision >= _ENTRY_GUARD_MIN_PREC and prob_now >= premium_prob:
            return {
                "ok": True,
                "reason": (
                    f"parcial fora da zona liberado por base forte | prec={precision:.1%} | prob={prob_now:.1%}"
                ),
            }
        return {
            "ok": False,
            "reason": (
                "continuidade parcial fora da zona com modelo fraco | "
                f"acc={accuracy:.1%} | prec={precision:.1%} | auc={auc:.3f}"
            ),
        }

    if not ideal and weak_model and base_insufficient and prob_now < max(0.81, threshold + 0.18):
        return {
            "ok": False,
            "reason": (
                "fora da zona ideal sem base historica forte no ativo | "
                f"prob={prob_now:.1%} | acc={accuracy:.1%} | prec={precision:.1%}"
            ),
        }

    return {
        "ok": True,
        "reason": (
            f"qualidade ok | acc={accuracy:.1%} | prec={precision:.1%} | auc={auc:.3f}"
        ),
    }


def _training_alignment_check(pred: Optional[dict], trigger: Optional[dict]) -> dict:
    if not isinstance(pred, dict):
        return {"ok": False, "reason": "sem modelo entry_guard treinado"}

    trigger = trigger or {}
    train_samples = int(pred.get("train_samples", 0) or 0)
    _delay_raw = pred.get("delay_candles", 99)
    delay_candles = 99 if _delay_raw is None else int(_delay_raw)
    recommended_threshold = float(pred.get("recommended_threshold", pred.get("threshold", 0.0)) or 0.0)
    prob_now = float(pred.get("prob_now", 0.0) or 0.0)
    timing_mode = str(pred.get("timing_mode", "delay_aware") or "delay_aware")

    if train_samples < 1000:
        return {
            "ok": False,
            "reason": f"treino insuficiente ({train_samples} amostras)",
        }

    if delay_candles > 2:
        return {
            "ok": False,
            "reason": f"fora da janela ensinada (delay={delay_candles})",
        }

    if not trigger.get("triggered"):
        return {
            "ok": False,
            "reason": "trigger de win passado nao confirmou",
        }

    live_floor = max(0.50, recommended_threshold or float(pred.get("threshold", 0.0) or 0.0))
    if prob_now < live_floor:
        return {
            "ok": False,
            "reason": f"prob={prob_now:.1%} abaixo do piso treinado {live_floor:.1%}",
        }

    return {
        "ok": True,
        "reason": (
            f"alinhado ao treino | timing={timing_mode} | amostras={train_samples} | "
            f"delay={delay_candles} | prob={prob_now:.1%}"
        ),
    }


def _validate_dt_win_geometry_alignment(ativo: str, pat: dict, atr_val: float,
                                        hs_stats: dict, entry_guard_pred: Optional[dict] = None) -> dict:
    geo = _extract_geometry(pat, atr_val)
    if geo is None:
        return {"ok": False, "reason": "geometria indisponivel para alinhamento"}

    study_geo = _dt_geometry_scan_filter(geo, None)
    if study_geo.get("hard_block"):
        return {
            "ok": False,
            "hard_block": True,
            "reason": study_geo.get("reason", "loss-profile geometrico"),
            "geometry_hits": None,
            "study_profile": study_geo.get("study_profile"),
        }

    pat_type = str(pat.get("type", ""))
    all_geo = hs_stats.get("geometry_history", []) if isinstance(hs_stats, dict) else []
    same_asset_wins = [
        g for g in all_geo
        if g.get("result") == 1 and g.get("source") != "live"
        and g.get("ativo") == ativo and g.get("type") == pat_type
    ]
    pooled_wins = [
        g for g in all_geo
        if g.get("result") == 1 and g.get("source") != "live"
        and g.get("type") == pat_type
    ]
    win_geos = same_asset_wins if len(same_asset_wins) >= _DT_WIN_GEOMETRY_MIN_SAMPLES else pooled_wins
    if len(win_geos) < _DT_WIN_GEOMETRY_MIN_SAMPLES:
        soft_ok = int(study_geo.get("negative_geo_hits", 0) or 0) <= 1
        return {
            "ok": soft_ok,
            "reason": f"base geometrica insuficiente ({len(win_geos)} wins) | {study_geo.get('reason', 'sem study profile')}",
            "geometry_hits": None,
            "study_profile": study_geo.get("study_profile"),
        }

    feature_specs = [
        ("depth_ratio", 2.4, 0.08),
        ("span", 2.6, 1.5),
        ("d_left", 2.4, 1.0),
        ("d_right", 2.4, 1.0),
        ("shoulder_ratio", 2.8, 0.00025),
    ]
    checks = 0
    hits = 0
    reasons = []

    for feat, z_limit, min_std in feature_specs:
        vals = [float(g.get(feat)) for g in win_geos if g.get(feat) is not None]
        if len(vals) < 8:
            continue
        mean_val = sum(vals) / len(vals)
        variance = sum((v - mean_val) ** 2 for v in vals) / len(vals)
        std_val = max(variance ** 0.5, min_std)
        current_val = float(geo.get(feat, mean_val))
        z_score = abs(current_val - mean_val) / std_val
        checks += 1
        if z_score <= z_limit:
            hits += 1
        else:
            reasons.append(f"{feat} fora do win-profile ({current_val:.4f})")

    feature_map = entry_guard_pred.get("feature_map", {}) if isinstance(entry_guard_pred, dict) else {}
    if isinstance(feature_map, dict) and feature_map:
        entry_to_touch_atr = float(feature_map.get("entry_to_touch_atr", 999.0) or 999.0)
        progress_pct = float(feature_map.get("progress_pct", 999.0) or 999.0)

        checks += 1
        if entry_to_touch_atr <= _DT_ENTRY_TOUCH_BAND_ATR:
            hits += 1
        else:
            reasons.append(f"touch distante do win-profile ({entry_to_touch_atr:.2f}ATR)")

        checks += 1
        if progress_pct <= _DT_ENTRY_PROGRESS_MAX:
            hits += 1
        else:
            reasons.append(f"progress fora do win-profile ({progress_pct:.0%})")

    if checks <= 0:
        return {"ok": True, "reason": "sem checks suficientes de geometria", "geometry_hits": None}

    min_hits = min(checks, 4) if checks <= 4 else 5
    ok = hits >= min_hits
    detail = f"hits={hits}/{checks} | base={len(win_geos)} wins"
    if ok:
        return {
            "ok": True,
            "reason": f"geometria alinhada aos wins | {detail} | {study_geo.get('reason', '')}",
            "geometry_hits": hits,
            "geometry_checks": checks,
            "study_profile": study_geo.get("study_profile"),
        }
    return {
        "ok": False,
        "reason": f"geometria fora do win-profile | {detail} | {'; '.join(reasons[:3])} | {study_geo.get('reason', '')}",
        "geometry_hits": hits,
        "geometry_checks": checks,
        "study_profile": study_geo.get("study_profile"),
    }


def _describe_entry_guard_direction_alignment(pred: Optional[dict]) -> dict:
    if not isinstance(pred, dict):
        return {
            "aligned": False,
            "margin": None,
            "reason": "entry_guard indisponivel",
        }

    prob_now = float(pred.get("prob_now", 0.0) or 0.0)
    threshold = float(pred.get("threshold", 0.0) or 0.0)
    margin = round(prob_now - threshold, 4)
    directional_floor = 0.50
    if prob_now >= directional_floor:
        return {
            "aligned": True,
            "margin": margin,
            "reason": (
                f"prob direcional={prob_now:.1%} >= {directional_floor:.0%} | "
                f"thr treino={threshold:.1%} | edge={margin:+.1%}"
            ),
        }

    return {
        "aligned": False,
        "margin": margin,
        "reason": (
            f"prob direcional={prob_now:.1%} < {directional_floor:.0%} | "
            f"thr treino={threshold:.1%} | edge={margin:.1%}"
        ),
    }


def _detect_dt_touch_continuation_signal(pat: dict, df: Optional[pd.DataFrame], atr_val: float) -> dict:
    if df is None or len(df) < 2:
        return {"matched": False, "strength": 0.0, "reason": "sem velas suficientes"}

    try:
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        last_idx = len(df) - 1
        rs_idx = int((pat.get("right_shoulder") or {}).get("idx", last_idx) or last_idx)
        rs_idx = max(0, min(rs_idx, last_idx))
        signal_idx = last_idx
        prev_idx = max(0, signal_idx - 1)
        touch_price = float((pat.get("right_shoulder") or {}).get("price", closes[signal_idx]) or closes[signal_idx])
        atr_base = max(float(atr_val or 0.0), abs(float(closes[signal_idx])) * 0.0005, 1e-6)

        sig_open = float(opens[signal_idx])
        sig_high = float(highs[signal_idx])
        sig_low = float(lows[signal_idx])
        sig_close = float(closes[signal_idx])
        sig_range = max(sig_high - sig_low, 1e-8)
        sig_body = abs(sig_close - sig_open)
        sig_body_pct = sig_body / sig_range
        prev_close = float(closes[prev_idx])

        touch_high = float(highs[rs_idx])
        touch_low = float(lows[rs_idx])
        touch_close = float(closes[rs_idx])
        touch_open = float(opens[rs_idx])
        touch_range = max(touch_high - touch_low, 1e-8)

        direction = str(pat.get("direction", ""))
        is_call = direction == "CALL"
        close_pos = ((sig_close - sig_low) / sig_range) if is_call else ((sig_high - sig_close) / sig_range)
        same_candle = signal_idx == rs_idx
        prev_close_ok = sig_close >= prev_close if is_call else sig_close <= prev_close

        if is_call:
            touch_confirmed = abs(touch_low - touch_price) <= atr_base * 0.40 or abs(min(sig_open, sig_close) - touch_price) <= atr_base * 0.50
            touch_wick = max(min(touch_open, touch_close) - touch_low, 0.0) / touch_range
            continuation_ok = sig_close > sig_open and close_pos >= 0.58 and sig_body_pct >= 0.22
            if same_candle:
                continuation_ok = continuation_ok and (touch_wick >= 0.25 or (close_pos >= 0.70 and sig_body_pct >= 0.30))
            else:
                continuation_ok = continuation_ok and (sig_close >= touch_close or prev_close_ok or close_pos >= 0.72)
            label = "suporte"
            candle_label = "verde"
        else:
            touch_confirmed = abs(touch_high - touch_price) <= atr_base * 0.40 or abs(max(sig_open, sig_close) - touch_price) <= atr_base * 0.50
            touch_wick = max(touch_high - max(touch_open, touch_close), 0.0) / touch_range
            continuation_ok = sig_close < sig_open and close_pos >= 0.58 and sig_body_pct >= 0.22
            if same_candle:
                continuation_ok = continuation_ok and (touch_wick >= 0.25 or (close_pos >= 0.70 and sig_body_pct >= 0.30))
            else:
                continuation_ok = continuation_ok and (sig_close <= touch_close or prev_close_ok or close_pos >= 0.72)
            label = "resistencia"
            candle_label = "vermelha"

        strength = min(1.0, max(0.0, sig_body_pct * 0.5 + close_pos * 0.35 + touch_wick * 0.35))
        matched = bool(touch_confirmed and continuation_ok)
        partial_continuation = bool(
            touch_confirmed and not matched and (
                (sig_body_pct >= 0.22 and close_pos >= 0.55)
                or (same_candle and touch_wick >= 0.25)
                or ((not same_candle) and (prev_close_ok and sig_body_pct >= 0.22))
            )
        )
        if partial_continuation:
            matched = True
            strength = max(strength, 0.58)

        if matched:
            if partial_continuation:
                reason = f"2o toque no {label} + continuidade parcial valida (entry_guard decide o timing)"
            elif same_candle:
                reason = f"2o toque no {label} + vela {candle_label} de rejeicao/continuidade"
            else:
                reason = f"2o toque no {label} + vela {candle_label} de continuidade apos o toque"
        else:
            reason = f"toque={touch_confirmed} | continuidade={continuation_ok} | body={sig_body_pct:.0%}"

        return {
            "matched": matched,
            "partial": partial_continuation,
            "strength": round(float(strength), 4),
            "same_candle": same_candle,
            "signal_idx": int(signal_idx),
            "touch_idx": int(rs_idx),
            "reason": reason,
        }
    except Exception:
        return {"matched": False, "strength": 0.0, "reason": "falha ao ler continuidade"}


def _build_dt_prediction_2m(pat: dict, df: Optional[pd.DataFrame], atr_val: float,
                            entry_guard_pred: Optional[dict], touch_continuation: Optional[dict]) -> dict:
    if df is None or len(df) < 1:
        return {"available": False, "reason": "sem candle atual"}
    if not isinstance(entry_guard_pred, dict):
        return {"available": False, "reason": "entry_guard indisponivel"}
    direction_alignment = entry_guard_pred.get("direction_alignment_2m") if isinstance(entry_guard_pred.get("direction_alignment_2m"), dict) else {}
    if not direction_alignment.get("aligned"):
        return {"available": False, "reason": direction_alignment.get("reason", "direcao 2m desalinhada")}
    if not isinstance(touch_continuation, dict) or not touch_continuation.get("matched"):
        return {"available": False, "reason": "continuidade no 2o toque nao confirmada"}

    try:
        current_price = float(df["close"].values[-1])
        atr_base = max(float(atr_val or 0.0), abs(current_price) * 0.0005, 1e-6)
        neckline = float(pat.get("neckline", current_price) or current_price)
        target = float(pat.get("target", neckline) or neckline)
        direction = str(pat.get("direction", "CALL"))
        prob_now = float(entry_guard_pred.get("prob_now", 0.0) or 0.0)
        threshold = float(entry_guard_pred.get("threshold", 0.0) or 0.0)
        strength = float(touch_continuation.get("strength", 0.0) or 0.0)
        edge = max(0.0, prob_now - threshold)
        max_move = atr_base * (0.55 + min(edge * 2.5, 0.35) + min(strength * 0.35, 0.20))

        if direction == "CALL":
            toward_neck = neckline - current_price
            raw_move = toward_neck if toward_neck > 0 else max_move
            projected_price = current_price + min(max_move, max(raw_move, atr_base * 0.22))
            projected_price = min(projected_price, max(target, neckline, current_price + atr_base * 0.22))
        else:
            toward_neck = current_price - neckline
            raw_move = toward_neck if toward_neck > 0 else max_move
            projected_price = current_price - min(max_move, max(raw_move, atr_base * 0.22))
            projected_price = max(projected_price, min(target, neckline, current_price - atr_base * 0.22))

        move_atr = abs(projected_price - current_price) / atr_base
        return {
            "available": True,
            "minutes": 2,
            "direction": direction,
            "price": round(float(projected_price), 6),
            "current_price": round(float(current_price), 6),
            "move_atr": round(float(move_atr), 3),
            "confidence": round(float(max(prob_now, threshold)), 4),
            "reason": f"projecao 2m rumo a neckline com edge={edge:.1%}",
        }
    except Exception:
        return {"available": False, "reason": "falha ao montar projecao 2m"}


def _detect_dt_counter_barrier(pat: dict,
                               df: Optional[pd.DataFrame],
                               atr_val: float,
                               current_price: Optional[float] = None) -> dict:
    if df is None or len(df) < 5:
        return {"near": False, "reason": "sem candles para barreira"}

    try:
        direction = str(pat.get("direction", "") or "")
        if direction not in {"CALL", "PUT"}:
            return {"near": False, "reason": "direcao invalida"}

        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        closes = df["close"].astype(float).values
        n = len(closes)
        cur_price = float(current_price if current_price is not None else closes[-1])
        atr_base = max(float(atr_val or 0.0), abs(cur_price) * 0.0005, 1e-6)
        rs_idx = int((pat.get("right_shoulder") or {}).get("idx", n - 2) or n - 2)
        start = max(1, min(n - 3, max(rs_idx - 1, n - max(6, DT_GRAPH_TIMING_BARRIER_LOOKBACK))))
        end = max(start + 1, n - 1)

        barrier_price = None
        barrier_idx = None
        label = "resistencia" if direction == "CALL" else "suporte"

        if direction == "PUT":
            for idx in range(start, end):
                low_val = float(lows[idx])
                if low_val >= cur_price:
                    continue
                if low_val <= float(lows[idx - 1]) and low_val <= float(lows[min(idx + 1, n - 1)]):
                    if barrier_price is None or low_val > barrier_price:
                        barrier_price = low_val
                        barrier_idx = idx
            if barrier_price is None:
                recent_slice = lows[start:end]
                if len(recent_slice) > 0:
                    candidate = float(np.max(recent_slice[recent_slice < cur_price])) if np.any(recent_slice < cur_price) else None
                    if candidate is not None:
                        barrier_price = candidate
                        barrier_idx = start + int(np.where(recent_slice == candidate)[0][-1])
        else:
            for idx in range(start, end):
                high_val = float(highs[idx])
                if high_val <= cur_price:
                    continue
                if high_val >= float(highs[idx - 1]) and high_val >= float(highs[min(idx + 1, n - 1)]):
                    if barrier_price is None or high_val < barrier_price:
                        barrier_price = high_val
                        barrier_idx = idx
            if barrier_price is None:
                recent_slice = highs[start:end]
                if len(recent_slice) > 0:
                    candidate = float(np.min(recent_slice[recent_slice > cur_price])) if np.any(recent_slice > cur_price) else None
                    if candidate is not None:
                        barrier_price = candidate
                        barrier_idx = start + int(np.where(recent_slice == candidate)[0][0])

        if barrier_price is None:
            return {"near": False, "reason": f"sem {label} recente a frente"}

        distance_atr = abs(cur_price - barrier_price) / atr_base
        return {
            "near": bool(distance_atr <= DT_GRAPH_TIMING_BARRIER_NEAR_ATR),
            "label": label,
            "price": round(float(barrier_price), 6),
            "distance_atr": round(float(distance_atr), 4),
            "barrier_idx": int(barrier_idx) if barrier_idx is not None else None,
            "reason": f"{label} proximo a {distance_atr:.2f}ATR",
        }
    except Exception:
        return {"near": False, "reason": "falha ao mapear barreira"}


def _build_dt_graph_timing_hint(pat: dict,
                                touch_continuation: Optional[dict],
                                entry_region: Optional[dict],
                                prediction_2m: Optional[dict],
                                nn_pred: Optional[dict] = None,
                                df: Optional[pd.DataFrame] = None,
                                current_price: Optional[float] = None,
                                atr_val: float = 0.0) -> dict:
    """Timing simplificado: Bayesiana decide, sem bloqueios nem esperas."""
    touch_continuation = touch_continuation or {}
    entry_region = entry_region or {}
    nn_pred = nn_pred or {}

    direction = str(pat.get("direction", ""))
    zone = "suporte" if direction == "CALL" else "resistencia"
    strength = float(touch_continuation.get("strength", 0.0) or 0.0)
    partial = bool(touch_continuation.get("partial"))
    same_candle = bool(touch_continuation.get("same_candle"))
    matched = bool(touch_continuation.get("matched"))
    dist_touch_atr = float(entry_region.get("dist_touch_atr", 0.0) or 0.0) if entry_region else 0.0
    progress_pct = float(entry_region.get("progress_pct", 0.0) or 0.0) if entry_region else 0.0
    nn_score = float(nn_pred.get("nn_score", nn_pred.get("prob_win", 0.0)) or 0.0)

    # Sem bloqueios — Bayes decide. Timing sempre disponivel.
    reason_parts = [f"Bayes decide | {zone}"]
    if matched:
        reason_parts.append(f"touch={strength:.2f}")
    if nn_score > 0:
        reason_parts.append(f"nn={nn_score:.0%}")
    if dist_touch_atr > 0:
        reason_parts.append(f"dist={dist_touch_atr:.2f}ATR")

    return {
        "available": True,
        "action": "now",
        "label": "agora_bayes",
        "wait_seconds": 0.0,
        "zone": zone,
        "direction": direction,
        "touch_strength": round(float(strength), 4),
        "partial": partial,
        "same_candle": same_candle,
        "nn_score": round(float(nn_score), 4) if nn_pred else None,
        "dist_touch_atr": round(float(dist_touch_atr), 4),
        "progress_pct": round(float(progress_pct), 4),
        "reason": " | ".join(reason_parts),
    }


def _wait_for_dt_graph_timing_window(bx: BrokerAPI,
                                     ativo: str,
                                     pat: dict,
                                     atr_val: float,
                                     timing_hint: Optional[dict],
                                     current_price: Optional[float] = None) -> dict:
    timing_hint = timing_hint or {}
    action = str(timing_hint.get("action", "") or "")

    if action not in {"wait", "wait_retest_zone"}:
        return {
            "ok": True,
            "reason": "sem espera adicional",
            "price": current_price,
            "entry_region": _validate_dt_entry_region(pat, float(current_price), atr_val) if current_price is not None else None,
            "touch_continuation": None,
            "df": None,
            "waited": 0.0,
        }

    max_wait = min(DT_GRAPH_TIMING_WAIT_MAX_SEC, max(1.0, float(timing_hint.get("wait_seconds", 0.0) or 0.0)))
    started = time.time()
    last_price = float(current_price) if current_price is not None else None
    last_region = _validate_dt_entry_region(pat, last_price, atr_val) if last_price is not None else None
    last_touch = None
    last_df = None

    while time.time() - started < max_wait:
        try:
            _cur_rt, _closed_df = get_realtime_entry_snapshot(bx, ativo, TF_M1, closed_n=6)
            if _cur_rt is not None:
                last_price = float(_cur_rt)
        except Exception:
            _closed_df = None

        try:
            last_df = get_candles_df(bx, ativo, TF_M1, 60, min_len=50)
            if last_df is None:
                last_df = get_last_closed_candles_df(bx, ativo, TF_M1, 60, min_len=50)
        except Exception:
            last_df = _closed_df

        if last_price is not None:
            last_region = _validate_dt_entry_region(pat, float(last_price), atr_val)
        if last_df is not None and len(last_df) >= 2:
            last_touch = _detect_dt_touch_continuation_signal(pat, last_df, atr_val)

        if action == "wait_retest_zone":
            if last_region and last_region.get("ok") and last_region.get("ideal"):
                return {
                    "ok": True,
                    "reason": "preco retornou a zona ideal do 2o toque",
                    "price": last_price,
                    "entry_region": last_region,
                    "touch_continuation": last_touch,
                    "df": last_df,
                    "waited": round(time.time() - started, 2),
                }
        else:
            if last_touch and last_touch.get("matched") and not last_touch.get("partial") and last_region and last_region.get("ideal"):
                return {
                    "ok": True,
                    "reason": "continuidade confirmou dentro da zona ideal",
                    "price": last_price,
                    "entry_region": last_region,
                    "touch_continuation": last_touch,
                    "df": last_df,
                    "waited": round(time.time() - started, 2),
                }

        time.sleep(DT_GRAPH_TIMING_POLL_SEC)

    fail_reason = "preco nao voltou para a zona ideal a tempo" if action == "wait_retest_zone" else "confirmacao nao amadureceu na zona ideal"
    return {
        "ok": False,
        "reason": fail_reason,
        "price": last_price,
        "entry_region": last_region,
        "touch_continuation": last_touch,
        "df": last_df,
        "waited": round(time.time() - started, 2),
    }


def _maybe_soft_release_dt_graph_timing(setup: Optional[dict],
                                        nn_pred: Optional[dict] = None) -> dict:
    setup = setup or {}
    nn_pred = nn_pred or {}
    entry_region = (setup.get("entry_region") or {}) if isinstance(setup, dict) else {}
    touch_continuation = (setup.get("touch_continuation") or {}) if isinstance(setup, dict) else {}
    study_multifactor = (setup.get("study_multifactor") or {}) if isinstance(setup, dict) else {}
    quality_risk = (setup.get("quality_risk") or {}) if isinstance(setup, dict) else {}
    live_metrics = (setup.get("live_metrics") or {}) if isinstance(setup, dict) else {}
    entry_guard_pred = (setup.get("entry_guard_pre_pred") or {}) if isinstance(setup, dict) else {}
    counter_barrier = (live_metrics.get("counter_barrier") or {}) if isinstance(live_metrics, dict) else {}

    if not entry_region.get("ok"):
        return {"ok": False, "reason": "regiao invalida apos espera"}

    if entry_region.get("ideal"):
        return {"ok": False, "reason": "zona ideal voltou; soft release desnecessario"}

    if not touch_continuation.get("matched"):
        return {"ok": False, "reason": "continuidade nao confirmada apos espera"}

    if study_multifactor.get("hard_block"):
        return {"ok": False, "reason": study_multifactor.get("reason", "loss-profile multifator")}

    nn_score = float(nn_pred.get("nn_score", nn_pred.get("prob_win", 0.0)) or 0.0)
    raw_progress = live_metrics.get("progress_pct", entry_region.get("progress_pct", 0.0))
    progress_pct = float(raw_progress or 0.0)
    if progress_pct <= 1.0:
        progress_pct *= 100.0
    target_room_atr = float(live_metrics.get("target_room_atr", 0.0) or 0.0)
    dist_touch_atr = float(entry_region.get("dist_touch_atr", 0.0) or 0.0)
    prob_now = float(entry_guard_pred.get("prob_now", 0.0) or 0.0)
    precision = float(entry_guard_pred.get("precision", 0.0) or 0.0)
    premium = bool(study_multifactor.get("premium") or study_multifactor.get("trigger_release"))
    partial = bool(touch_continuation.get("partial"))
    same_candle = bool(touch_continuation.get("same_candle"))
    wick_pct = float(live_metrics.get("wick_pct", 0.0) or 0.0)

    if isinstance(counter_barrier, dict) and counter_barrier.get("near"):
        return {
            "ok": False,
            "reason": counter_barrier.get("reason", "barreira proxima contra a entrada"),
        }

    if same_candle and wick_pct >= 35.0:
        return {
            "ok": False,
            "reason": f"pavio alto no 2o toque ({wick_pct:.0f}%) sem preenchimento da regiao",
        }

    if partial:
        partial_release_ok = bool(
            nn_score >= DT_GRAPH_TIMING_DIRECT_PARTIAL_NN_MIN
            and progress_pct <= DT_GRAPH_TIMING_SOFT_RELEASE_MAX_PROGRESS
            and dist_touch_atr <= DT_GRAPH_TIMING_SOFT_RELEASE_MAX_DIST_ATR
            and target_room_atr >= max(3.5, DT_GRAPH_TIMING_SOFT_RELEASE_MIN_TARGET_ATR)
            and prob_now >= 0.78
        )
        if not partial_release_ok:
            return {"ok": False, "reason": f"continuidade parcial | forca curta ({float(touch_continuation.get('strength', 0.0) or 0.0):.2f})"}

    if isinstance(quality_risk, dict) and quality_risk.get("ok") is False:
        quality_override = bool(
            nn_score >= DT_GRAPH_TIMING_DIRECT_NN_MIN
            and prob_now >= 0.78
            and target_room_atr >= DT_GRAPH_TIMING_SOFT_RELEASE_MIN_TARGET_ATR
        )
        if not quality_override:
            return {"ok": False, "reason": quality_risk.get("reason", "quality_risk bloqueou")}

    if nn_score < DT_GRAPH_TIMING_SOFT_RELEASE_NN_MIN and not premium:
        return {
            "ok": False,
            "reason": f"NN abaixo do piso de soft release ({nn_score:.2f} < {DT_GRAPH_TIMING_SOFT_RELEASE_NN_MIN:.2f})",
        }

    if progress_pct > DT_GRAPH_TIMING_SOFT_RELEASE_MAX_PROGRESS:
        return {
            "ok": False,
            "reason": f"progress alto demais ({progress_pct:.0f}% > {DT_GRAPH_TIMING_SOFT_RELEASE_MAX_PROGRESS:.0f}%)",
        }

    if target_room_atr < DT_GRAPH_TIMING_SOFT_RELEASE_MIN_TARGET_ATR:
        return {
            "ok": False,
            "reason": f"alvo restante curto ({target_room_atr:.1f}ATR < {DT_GRAPH_TIMING_SOFT_RELEASE_MIN_TARGET_ATR:.1f}ATR)",
        }

    if dist_touch_atr > DT_GRAPH_TIMING_SOFT_RELEASE_MAX_DIST_ATR:
        return {
            "ok": False,
            "reason": f"distancia ao 2o toque alta ({dist_touch_atr:.2f}ATR > {DT_GRAPH_TIMING_SOFT_RELEASE_MAX_DIST_ATR:.2f}ATR)",
        }

    if prob_now < 0.50 and not premium and nn_score < 0.80:
        return {
            "ok": False,
            "reason": f"entry_guard muito fraco para soft release (prob={prob_now:.1%})",
        }

    if precision and precision < 0.88 and not premium and nn_score < 0.80:
        return {
            "ok": False,
            "reason": f"precisao historica insuficiente para soft release ({precision:.1%})",
        }

    return {
        "ok": True,
        "reason": (
            "soft release do timing DT | "
            f"NN={nn_score:.0%} | progress={progress_pct:.0f}% | alvo={target_room_atr:.1f}ATR | touch={dist_touch_atr:.2f}ATR"
            + (" | multifator premium" if premium else "")
        ),
        "nn_score": round(nn_score, 4),
        "progress_pct": round(progress_pct, 2),
        "target_room_atr": round(target_room_atr, 3),
        "dist_touch_atr": round(dist_touch_atr, 4),
        "premium": premium,
    }


def _build_entry_guard_features(ativo: str, pat: dict, df: Optional[pd.DataFrame],
                                atr_val: float, hs_stats: dict):
    if df is None or len(df) < 40:
        return None, None

    try:
        _H = df["high"].values
        _L = df["low"].values
        _C = df["close"].values
        _O = df["open"].values
        _n = len(_H)
        _candidate_idx = _n - 1
        _rs_idx = int(pat.get("right_shoulder", {}).get("idx", _candidate_idx))
        _rs_idx = max(0, min(_rs_idx, _candidate_idx))
        _delay = max(0, _candidate_idx - _rs_idx)

        _win_start = max(0, _candidate_idx - 110)
        _win_end = _candidate_idx + 1
        _H_win = _H[_win_start:_win_end]
        _L_win = _L[_win_start:_win_end]
        _C_win = _C[_win_start:_win_end]
        _O_win = _O[_win_start:_win_end]
        _n_win = len(_H_win)
        if _n_win < 25:
            return None, None

        _atr_local_vals = [float(_H_win[k] - _L_win[k]) for k in range(max(0, _n_win - 14), _n_win)]
        _atr_local = float(np.mean(_atr_local_vals)) if _atr_local_vals else atr_val
        if _atr_local <= 0:
            _atr_local = atr_val
        if _atr_local <= 0:
            _atr_local = max(abs(float(_C_win[-1])) * 0.0005, 1e-6)

        _pat_copy = dict(pat)
        _pat_copy["candles_ago"] = _delay
        _base_feats = extract_features(_pat_copy, _H_win, _L_win, _C_win, _O_win, _n_win,
                                       _atr_local, hs_stats, ativo)
        if _base_feats is None:
            return None, None

        _geo = _extract_geometry(pat, _atr_local) or {}
        _entry_price = float(_C[_candidate_idx])
        _open = float(_O[_candidate_idx])
        _high = float(_H[_candidate_idx])
        _low = float(_L[_candidate_idx])
        _range = max(_high - _low, 1e-8)
        _body = abs(_entry_price - _open)
        _upper_wick = _high - max(_entry_price, _open)
        _lower_wick = min(_entry_price, _open) - _low
        _wick_rejection = (_lower_wick if pat.get("direction") == "CALL" else _upper_wick) / _range
        _touch_price = float(pat.get("right_shoulder", {}).get("price", _entry_price))
        _neckline = float(pat.get("neckline", _entry_price))
        _left_idx = int(pat.get("left_shoulder", {}).get("idx", _rs_idx))
        _head_idx = int(pat.get("head", {}).get("idx", _rs_idx))
        _shoulder_left = float(pat.get("left_shoulder", {}).get("price", _touch_price))
        _shoulder_ratio = min(_shoulder_left, _touch_price) / max(_shoulder_left, _touch_price) if max(_shoulder_left, _touch_price) > 0 else 0.0
        _span = max(1, _rs_idx - _left_idx)

        _extra_feats = [
            round(_delay / 2.0, 6),
            round(abs(_entry_price - _neckline) / _atr_local, 6),
            round(abs(_entry_price - _touch_price) / _atr_local, 6),
            round(_range / _atr_local, 6),
            round(_body / _atr_local, 6),
            round(max(0.0, _wick_rejection), 6),
            round(max(0, _head_idx - _left_idx) / 40.0, 6),
            round(max(0, _rs_idx - _head_idx) / 40.0, 6),
            round(float(_geo.get("shoulder_ratio", _shoulder_ratio)), 6),
            round(float(_geo.get("depth_ratio", 0.0)), 6),
            round(_span / 60.0, 6),
        ]

        return [float(v) for v in _base_feats] + _extra_feats, {
            "delay_candles": _delay,
            "atr_local": round(float(_atr_local), 6),
        }
    except Exception:
        return None, None


def _estimate_entry_guard_score(ativo: str, pat: dict, df: Optional[pd.DataFrame],
                                atr_val: float, hs_stats: dict):
    if not _ENTRY_GUARD_ENABLED:
        return None, None

    bundle = _load_entry_guard_bundle(ativo)
    if bundle is None:
        return None, None

    feats, extra = _build_entry_guard_features(ativo, pat, df, atr_val, hs_stats)
    if feats is None:
        return None, None

    try:
        feature_names = bundle.get("feature_names", [])
        if feature_names and len(feats) != len(feature_names):
            return None, None

        feature_map = {}
        if feature_names:
            feature_map = {
                name: round(float(value), 6)
                for name, value in zip(feature_names, feats)
            }
        win_signature = _detect_entry_win_signature(feature_map)

        X = np.asarray([feats], dtype=float)
        models = bundle.get("models", {})
        rf_prob = float(models["rf"].predict_proba(X)[0][1])
        gb_prob = float(models["gb"].predict_proba(X)[0][1])
        lr_prob = float(models["lr"].predict_proba(X)[0][1])
        prob_now = float(rf_prob * 0.45 + gb_prob * 0.35 + lr_prob * 0.20)
        threshold = _get_entry_guard_live_threshold(bundle)
        metrics = bundle.get("metrics", {})
        recommended = metrics.get("recommended_threshold", {}) if isinstance(metrics, dict) else {}

        pred = {
            "approved": prob_now >= threshold,
            "prob_now": round(prob_now, 4),
            "threshold": round(threshold, 4),
            "recommended_threshold": round(float(recommended.get("threshold", threshold) or threshold), 4),
            "accuracy": round(float(metrics.get("accuracy", 0.0) or 0.0), 4),
            "auc": round(float(metrics.get("auc", 0.0) or 0.0), 4),
            "precision": round(float(recommended.get("precision", 0.0) or 0.0), 4),
            "recall": round(float(recommended.get("recall", 0.0) or 0.0), 4),
            "train_samples": int(metrics.get("train_samples", bundle.get("samples", 0)) or 0),
            "delay_candles": int((extra or {}).get("delay_candles", 0)),
            "p_rf": round(rf_prob, 4),
            "p_gb": round(gb_prob, 4),
            "p_lr": round(lr_prob, 4),
            "feature_map": feature_map,
            "win_signature": win_signature,
        }
        pred["direction_alignment_2m"] = _describe_entry_guard_direction_alignment(pred)
        return pred["prob_now"], pred
    except Exception:
        return None, None


def _rank_assets_by_entry_guard() -> List[Tuple[str, float, float, float, int]]:
    ranked = []
    try:
        for file_name in os.listdir(_user_data_dir):
            if not (file_name.startswith("entry_guard_") and file_name.endswith(".pkl")):
                continue
            ativo = file_name[len("entry_guard_"):-4]
            if not os.path.exists(_reversal_model_path(ativo)):
                continue
            bundle = _load_entry_guard_bundle(ativo)
            if bundle is None:
                continue
            metrics = bundle.get("metrics", {})
            recommended = metrics.get("recommended_threshold", {}) if isinstance(metrics, dict) else {}
            ranked.append((
                ativo,
                float(metrics.get("accuracy", 0.0) or 0.0),
                float(metrics.get("auc", 0.0) or 0.0),
                float(recommended.get("precision", 0.0) or 0.0),
                int(bundle.get("samples", 0) or 0),
            ))
    except Exception:
        return []

    ranked.sort(key=lambda item: (item[1], item[2], item[3], item[4]), reverse=True)
    return ranked


def _is_high_accuracy_asset(acc: float, auc: float, precision: float, samples: int) -> bool:
    return bool(
        acc >= _ENTRY_GUARD_MIN_ACC
        and auc >= _ENTRY_GUARD_MIN_AUC
        and precision >= _ENTRY_GUARD_MIN_PREC
        and samples >= _ENTRY_GUARD_MIN_SAMPLES
    )


def _is_conservative_fallback_asset(acc: float, auc: float, precision: float, samples: int) -> bool:
    return bool(
        acc >= _ENTRY_GUARD_FALLBACK_MIN_ACC
        and auc >= _ENTRY_GUARD_FALLBACK_MIN_AUC
        and precision >= _ENTRY_GUARD_FALLBACK_MIN_PREC
        and samples >= _ENTRY_GUARD_MIN_SAMPLES
    )


def _score_live_asset_candidate(bx: BrokerAPI, ativo: str, acc: float, payout: int = 0) -> Optional[dict]:
    """Ranqueia ativos para a varredura dinâmica usando accuracy, padrões e lateralidade."""
    df = get_candles_df(bx, ativo, TF_M1, _ASSET_SELECTION_CANDLES, min_len=90)
    if df is None or len(df) < 90:
        return None

    try:
        H = df["high"].values
        L = df["low"].values
        C_arr = df["close"].values
        O = df["open"].values
        n = len(H)
        atr_vals = [float(H[k] - L[k]) for k in range(max(0, n - 14), n)]
        atr = float(np.mean(atr_vals)) if atr_vals else 0.0
        if atr <= 0:
            return None

        pivot_highs, pivot_lows = detect_pivots(H, L, window=5)

        visible_patterns_raw = detect_double_touch(
            H, L, C_arr, O, pivot_highs, pivot_lows, atr, n,
            max_candles_ago=9999, training=True,
        )
        visible_start = max(0, n - 120)
        visible_count = 0
        for pat in visible_patterns_raw:
            rs_idx = int((pat.get("right_shoulder") or {}).get("idx", -1))
            if rs_idx >= visible_start:
                visible_count += 1

        live_patterns_raw = detect_double_touch(
            H, L, C_arr, O, pivot_highs, pivot_lows, atr, n,
            max_candles_ago=MAX_LIVE_SIGNAL_CANDLES,
        )

        fresh_candidates = []
        for pat in live_patterns_raw:
            bt = backtest_pattern(pat, C_arr, O, H, L, n)
            if bt is not None:
                continue
            candles_ago = max(0, n - 1 - pat["right_shoulder"]["idx"])
            if candles_ago > MAX_LIVE_SIGNAL_CANDLES:
                continue
            if pat.get("mode") == "realtime":
                continue
            if pat.get("mode") not in ("early", "double_touch") and candles_ago > 2:
                continue
            fresh_candidates.append((pat["type"], pat["direction"], candles_ago))

        dedup = {}
        for pat_type, direction, candles_ago in fresh_candidates:
            key = f"{pat_type}_{direction}"
            if key not in dedup or candles_ago < dedup[key]:
                dedup[key] = candles_ago

        live_count = len(dedup)
        direction_count = len({direction for _, direction, _ in fresh_candidates}) if fresh_candidates else 0
        live_score = min(live_count / 2.0, 1.0)
        visible_score = min(visible_count / 6.0, 1.0)
        payout_score = min(max(float(payout) / 100.0, 0.0), 1.0)
        conflict_penalty = 0.08 if direction_count > 1 else 0.0

        selection_score = (
            float(acc) * 0.35
            + live_score * 0.30
            + visible_score * 0.20
            + payout_score * 0.15
            - conflict_penalty
        )

        return {
            "asset": ativo,
            "selection_score": round(float(selection_score), 4),
            "accuracy": round(float(acc), 4),
            "visible_count": int(visible_count),
            "live_count": int(live_count),
            "regime_ok": True,
            "regime_score": 0.0,
            "regime_reason": "",
            "payout": int(payout),
        }
    except Exception:
        return None


def _download_entry_guard_from_github(file_name: str, quiet: bool = False) -> bool:
    try:
        import urllib.request

        url = GITHUB_ENTRY_GUARD_RAW_URL.replace("{file_name}", file_name)
        req = urllib.request.Request(url, headers={"User-Agent": "WS-Trader-IA/1.0"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = resp.read()
        out_path = os.path.join(_user_data_dir, file_name)
        with open(out_path, "wb") as f:
            f.write(raw)
        size_kb = len(raw) / 1024.0
        log.info(paint(f"📥 EntryGuard baixado: {file_name} ({size_kb:.0f} KB)", C.G))
        return True
    except Exception as ex:
        if not quiet:
            log.warning(paint(f"⚠️ Falha ao baixar {file_name}: {ex}", C.Y))
        return False


def _entry_guard_fallback_file_names() -> List[str]:
    """Monta uma lista provável de entry guards quando a API de contents do GitHub falha."""
    candidates = []
    seen = set()

    def _add_asset(asset_name: str):
        asset_name = str(asset_name or "").strip()
        if not asset_name:
            return
        file_name = f"entry_guard_{asset_name}.pkl"
        if file_name not in seen:
            seen.add(file_name)
            candidates.append(file_name)

    for asset_name in FIXED_ASSETS.get(_broker_suffix, []):
        _add_asset(asset_name)

    for asset_name in _get_full_pattern_qualified_assets():
        _add_asset(asset_name)

    try:
        for file_name in os.listdir(_user_data_dir):
            if not (file_name.startswith("reversal_tf_") and file_name.endswith(".pkl")):
                continue
            if "BACKUP" in file_name or "unified" in file_name.lower():
                continue
            asset_name = file_name[len("reversal_tf_"):-4]
            if asset_name:
                _add_asset(asset_name)
    except Exception:
        pass

    return candidates


def _sync_entry_guard_models_from_github() -> List[str]:
    """Baixa todos os entry guards publicados no GitHub para o diretório do usuário."""
    payload = None
    file_names = []
    try:
        import urllib.request

        req = urllib.request.Request(GITHUB_ENTRY_GUARD_MANIFEST_URL, headers={"User-Agent": "WS-Trader-IA/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as ex:
        log.warning(paint(f"⚠️ Falha ao baixar manifesto EntryGuard: {ex}", C.Y))
        payload = None

    if isinstance(payload, dict):
        manifest_files = payload.get("files", [])
        for file_name in manifest_files if isinstance(manifest_files, list) else []:
            file_name = str(file_name or "")
            if file_name.startswith("entry_guard_") and file_name.endswith(".pkl"):
                file_names.append(file_name)
        if file_names:
            log.info(paint(
                f"🌐 EntryGuards: manifesto carregado com {len(file_names)} arquivo(s)",
                C.G
            ))

    if not file_names and isinstance(payload, list):
        for item in payload:
            file_name = str(item.get("name", ""))
            if file_name.startswith("entry_guard_") and file_name.endswith(".pkl"):
                file_names.append(file_name)

    if not file_names:
        try:
            import urllib.request

            req = urllib.request.Request(GITHUB_ENTRY_GUARD_API_URL, headers={"User-Agent": "WS-Trader-IA/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            if isinstance(payload, list):
                for item in payload:
                    file_name = str(item.get("name", ""))
                    if file_name.startswith("entry_guard_") and file_name.endswith(".pkl"):
                        file_names.append(file_name)
        except Exception as ex:
            log.info(paint(f"🌐 EntryGuards: API /contents indisponível, usando fallback local ({ex})", C.Y))

    if not file_names:
        try:
            bundled_dir = _bundled_data_dir("models_entry_guard")
            if os.path.isdir(bundled_dir):
                for file_name in os.listdir(bundled_dir):
                    if file_name.startswith("entry_guard_") and file_name.endswith(".pkl"):
                        file_names.append(file_name)
        except Exception:
            pass

    if not file_names:
        file_names = _entry_guard_fallback_file_names()
        if file_names:
            log.info(paint(
                f"🌐 EntryGuards: usando fallback por URL direta para {len(file_names)} ativo(s)",
                C.Y
            ))

    downloaded_assets = []
    for file_name in file_names:
        asset = file_name[len("entry_guard_"):-4]
        local_path = os.path.join(_user_data_dir, file_name)
        if not os.path.exists(local_path):
            if not _copy_bundled_file_if_missing(file_name, "models_entry_guard"):
                _download_entry_guard_from_github(file_name, quiet=bool(file_names))
        if os.path.exists(local_path):
            downloaded_assets.append(asset)

    if downloaded_assets:
        log.info(paint(
            f"🌐 EntryGuards disponíveis para o live: {len(downloaded_assets)} ativos ({', '.join(downloaded_assets[:6])}{'...' if len(downloaded_assets) > 6 else ''})",
            C.G
        ))
    return downloaded_assets



# ═══════════════════════════════════════════════════════════════
# UTILIDADES + H&S DETECTION
# ═══════════════════════════════════════════════════════════════
cooldown = {}  # {ativo: timestamp}

# ── DEDUP persistente em disco — sobrevive a reinícios do bot ──
_DEDUP_FILE = os.path.join(os.path.expanduser("~"), ".wstrader", "ws_last_entry.json")

# ═══════════════════════════════════════════════════════════════
# MEMÓRIA DE NÍVEIS DT — impede entrada no 3º toque
# Grava o nível (preço) do toque quando entra.
# Se preço voltar ao mesmo nível (3º toque), BLOQUEIA.
# ═══════════════════════════════════════════════════════════════
_DT_LEVEL_MEMORY_FILE = os.path.join(os.path.expanduser("~"), ".wstrader", "ws_dt_level_memory.json")
_dt_level_memory: Dict[str, list] = {}  # {ativo: [{"level": price, "dir": "CALL"/"PUT", "ts": timestamp}, ...]}
_DT_MEMORY_EXPIRY = 3 * 60 * 60  # 3 horas — nível expira após 3 horas
_DT_MEMORY_TOL_MULT = 1.5        # tolerância = ATR * 1.5 para considerar "mesma zona"


def _load_dt_level_memory() -> Dict[str, list]:
    """Carrega memória de níveis DT do disco."""
    try:
        if os.path.exists(_DT_LEVEL_MEMORY_FILE):
            with open(_DT_LEVEL_MEMORY_FILE, "r") as f:
                data = json.load(f)
            # Limpar expirados
            now = time.time()
            cleaned = {}
            for ativo, entries in data.items():
                valid = [e for e in entries if now - e.get("ts", 0) < _DT_MEMORY_EXPIRY]
                if valid:
                    cleaned[ativo] = valid
            return cleaned
    except Exception:
        pass
    return {}


def _save_dt_level_memory():
    """Salva memória de níveis DT em disco (sobrevive a reinícios)."""
    global _dt_level_memory
    try:
        os.makedirs(os.path.dirname(_DT_LEVEL_MEMORY_FILE), exist_ok=True)
        with open(_DT_LEVEL_MEMORY_FILE, "w") as f:
            json.dump(_dt_level_memory, f)
    except Exception:
        pass


def _memorize_dt_level(ativo: str, level: float, direction: str):
    """Grava um nível de toque na memória. Ignora se já existe nível similar."""
    global _dt_level_memory
    if ativo not in _dt_level_memory:
        _dt_level_memory[ativo] = []
    # Verificar se nível já está na memória (evitar duplicatas a cada scan)
    now = time.time()
    for e in _dt_level_memory[ativo]:
        if now - e.get("ts", 0) > _DT_MEMORY_EXPIRY:
            continue
        if e.get("dir") != direction:
            continue
        # Se nível já existe com diff < 0.1% do preço, não gravar duplicata
        if level > 0 and abs(e.get("level", 0) - level) / level < 0.001:
            return  # Já memorizado — não duplicar
    _dt_level_memory[ativo].append({
        "level": round(level, 6),
        "dir": direction,
        "ts": time.time(),
    })
    # Manter apenas últimos 10 por ativo
    _dt_level_memory[ativo] = _dt_level_memory[ativo][-10:]
    _save_dt_level_memory()
    log.info(paint(
        f"  💾 MEMÓRIA DT: Gravado nível {level:.6f} ({direction}) para {ativo}",
        C.G
    ))


def _is_dt_level_already_traded(ativo: str, rs_price: float, direction: str, atr: float) -> bool:
    """Verifica se já entrou num DT neste nível (impede 3º toque).
    Compara RS price com níveis memorizados usando tolerância ATR*0.6."""
    global _dt_level_memory
    entries = _dt_level_memory.get(ativo, [])
    if not entries:
        return False
    tol = atr * _DT_MEMORY_TOL_MULT
    now = time.time()
    for e in entries:
        if now - e.get("ts", 0) > _DT_MEMORY_EXPIRY:
            continue
        if e.get("dir") != direction:
            continue
        if abs(e.get("level", 0) - rs_price) <= tol:
            log.info(paint(
                f"  🚫 MEMÓRIA DT: Nível {rs_price:.6f} já operado! "
                f"(memória: {e['level']:.6f}, diff={abs(e['level'] - rs_price):.6f}, tol={tol:.6f}) "
                f"— BLOQUEANDO 3º toque",
                C.R
            ))
            return True
    return False

# ═══════════════════════════════════════════════════════════════
# MEMÓRIA DE DIREÇÃO — impede entrada CONTRA sinal recente
# Se bot entrou PUT num ativo, não pode entrar CALL logo depois
# (e vice-versa). A seta que acabou de sair prevalece.
# ═══════════════════════════════════════════════════════════════
_CONTRA_SIGNAL_EXPIRY = 30 * 60  # 30 min — sinal contrário bloqueado por 30 min
_last_entry_dir: Dict[str, dict] = {}  # {ativo: {"dir": "CALL"/"PUT", "ts": timestamp}}


def _is_contra_signal(ativo: str, new_dir: str) -> bool:
    """Verifica se a entrada é CONTRA um sinal recente no mesmo ativo.
    Ex: bot entrou PUT há 10 min, agora quer entrar CALL = BLOQUEADO."""
    entry = _last_entry_dir.get(ativo)
    if not entry:
        return False
    if time.time() - entry.get("ts", 0) > _CONTRA_SIGNAL_EXPIRY:
        return False  # expirou
    if entry.get("dir") == new_dir:
        return False  # mesma direção, não é contra
    # Direção oposta dentro do tempo → CONTRA
    elapsed = int(time.time() - entry["ts"])
    log.info(paint(
        f"  🚫 CONTRA SINAL: {ativo} {new_dir} bloqueado — "
        f"último sinal foi {entry['dir']} há {elapsed}s "
        f"(expira em {int(_CONTRA_SIGNAL_EXPIRY - elapsed)}s)",
        C.R
    ))
    return True


def _record_entry_dir(ativo: str, direction: str):
    """Grava direção da última entrada para bloquear sinais contrários."""
    _last_entry_dir[ativo] = {"dir": direction, "ts": time.time()}


# ── LOCK FILE — impede duas instâncias do bot rodando ao mesmo tempo ──
_LOCK_FILE = os.path.join(os.path.expanduser("~"), ".wstrader", "ws_bot.lock")
_lock_fh = None  # file handle mantido aberto durante execução

def _acquire_lock() -> bool:
    """Tenta adquirir lock exclusivo. Retorna True se conseguiu."""
    global _lock_fh
    try:
        os.makedirs(os.path.dirname(_LOCK_FILE), exist_ok=True)
        _lock_fh = open(_LOCK_FILE, "w")
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(_lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fh.write(str(os.getpid()))
        _lock_fh.flush()
        return True
    except (IOError, OSError):
        if _lock_fh:
            _lock_fh.close()
            _lock_fh = None
        return False

def _release_lock():
    """Libera o lock file."""
    global _lock_fh
    try:
        if _lock_fh:
            if os.name == "nt":
                import msvcrt
                try:
                    msvcrt.locking(_lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
            _lock_fh.close()
            _lock_fh = None
        if os.path.exists(_LOCK_FILE):
            os.remove(_LOCK_FILE)
    except Exception:
        pass

def _load_last_entry_key() -> str:
    """Carrega a chave do último trade. Expira em 10 minutos."""
    try:
        if os.path.exists(_DEDUP_FILE):
            with open(_DEDUP_FILE, "r") as f:
                data = json.load(f)
            if time.time() - data.get("ts", 0) < 120:  # 2 min
                return data.get("key", "")
    except Exception:
        pass
    return ""

def _save_last_entry_key(key: str):
    """Salva a chave do último trade em disco."""
    try:
        os.makedirs(os.path.dirname(_DEDUP_FILE), exist_ok=True)
        with open(_DEDUP_FILE, "w") as f:
            json.dump({"key": key, "ts": time.time()}, f)
    except Exception:
        pass


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
    edge_min = 1  # FIX DELAY: era 2, agora 1 — detecta RS 1 vela mais cedo
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
    """H&S removido: o engine opera somente Double Touch."""
    return []


def detect_early_hs(H, L, C_arr, O, pivot_highs, pivot_lows, atr, n):
    """H&S antecipado removido: o engine opera somente Double Touch."""
    return []


# ═══════════════════════════════════════════════════════════════
# DETECÇÃO DUPLO TOQUE (Double Top / Double Bottom)
# ═══════════════════════════════════════════════════════════════

def detect_double_touch(H, L, C_arr, O, pivot_highs, pivot_lows, atr, n,
                        max_candles_ago=9999, training=False):
    """Detecta Duplo Toque: preço toca o MESMO nível 2x + rejeição (wick).
    Double Top (PUT): 2 toques em resistência + wick rejeição → preço cai
    Double Bottom (CALL): 2 toques em suporte + wick rejeição → preço sobe
    """
    patterns = []
    tol = atr * 0.35
    min_spacing = 12
    max_spacing = 45
    _train_mode = bool(training or max_candles_ago >= 9999)
    min_depth = atr * 1.5  # alinhado entre treino e live (depth não afeta WR)
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
                _n_piv_at_lvl = sum(1 for _, p in pivot_highs if abs(p - touch_level) <= tol)
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
                    "target": round(v_price - depth, 6),
                    "stop": round(touch_level + atr * 0.3, 6),
                    "entry_idx": int(idx2),
                    "entry_price": round(float(C_arr[int(idx2)]), 6),
                    "candles_ago": n - 1 - idx2,
                    "n_pivots_at_level": _n_piv_at_lvl,
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
                d_left = v_idx - idx1
                d_right = j - v_idx
                _n_piv_at_lvl = sum(1 for _, p in pivot_highs if abs(p - touch_level) <= tol)
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
                    "target": round(v_price - depth, 6),
                    "stop": round(touch_level + atr * 0.3, 6),
                    "entry_idx": int(j),
                    "entry_price": round(c_j, 6),
                    "candles_ago": n - 1 - j,
                    "n_pivots_at_level": _n_piv_at_lvl,
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
                _n_piv_at_lvl = sum(1 for _, p in pivot_lows if abs(p - touch_level) <= tol)
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
                    "target": round(p_price + depth, 6),
                    "stop": round(touch_level - atr * 0.3, 6),
                    "entry_idx": int(idx2),
                    "entry_price": round(float(C_arr[int(idx2)]), 6),
                    "candles_ago": n - 1 - idx2,
                    "n_pivots_at_level": _n_piv_at_lvl,
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
                d_left = p_idx - idx1
                d_right = j - p_idx
                _n_piv_at_lvl = sum(1 for _, p in pivot_lows if abs(p - touch_level) <= tol)
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
                    "target": round(p_price + depth, 6),
                    "stop": round(touch_level - atr * 0.3, 6),
                    "entry_idx": int(j),
                    "entry_price": round(c_j, 6),
                    "candles_ago": n - 1 - j,
                    "n_pivots_at_level": _n_piv_at_lvl,
                })

    return patterns


def backtest_pattern(pat, C, O, H, L, n):
    """Verifica se o padrão H&S resultaria em WIN ou LOSS.
    Regra: entra no CLOSE da vela do ombro direito (delay=0).
    Verifica o close EXP candles depois: exit = C[entry_idx + EXP].
    EXP_FIXA=2 → checa 2 candles à frente (alinhado com treino original).
    PUT: WIN se close < entry_price
    CALL: WIN se close > entry_price
    Retorna None se padrão é LIVE (sem resultado ainda)."""
    entry_idx = pat.get("entry_idx", pat["right_shoulder"]["idx"])
    if entry_idx >= n or entry_idx < 0:
        return None  # sem dados para verificar
    # EXP dinâmica: early usa EXP_EARLY (delay≈0), DT/classic usa EXP_FIXA (delay≥1)
    _exp = EXP_EARLY if pat.get("mode") == "early" else EXP_FIXA
    exit_idx = entry_idx + _exp  # candles de expiração
    if exit_idx >= n:
        return None  # padrão muito recente, sem resultado ainda
    # Entrada no CLOSE do ombro direito
    entry_price = float(C[entry_idx])
    exit_price = float(C[exit_idx])
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
        "exit_idx": exit_idx,
        "pips": round(abs(exit_price - entry_price), 6),
    }


# ══════════════════════════════════════════════════════════════
# BASE DE TREINO PRÉ-TREINADA (GitHub auto-download)
# ══════════════════════════════════════════════════════════════
def _download_training_base() -> Optional[dict]:
    """Tenta baixar base de treino mais recente do GitHub.
    Retorna dict com a base ou None se falhar."""
    try:
        import urllib.request
        import urllib.error

        log.info(paint("🌐 Baixando base de treino do GitHub...", C.B))
        print(">>> IA: Baixando treinamento do servidor... (pode levar 1-2 min)", flush=True)
        req = urllib.request.Request(GITHUB_TRAINING_URL, headers={
            "User-Agent": "WS-Trader-IA/1.0",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            size_mb = len(raw) / (1024 * 1024)
            log.info(paint(f"📥 Download concluído: {size_mb:.1f} MB", C.G))
            print(f">>> IA: Download concluído! ({size_mb:.1f} MB)", flush=True)
            data = json.loads(raw.decode("utf-8"))
        remote_version = data.get("version", "")
        if remote_version:
            log.info(paint(f"✅ Base de treino encontrada: versão {remote_version}", C.G))
            return data
    except urllib.error.HTTPError as e:
        log.warning(paint(f"⚠️ Erro ao baixar treino do GitHub: HTTP {e.code}", C.Y))
        print(f">>> IA: Erro ao baixar treinamento (HTTP {e.code})", flush=True)
    except Exception as e:
        log.warning(paint(f"⚠️ Falha no download do treino: {e}", C.Y))
        print(f">>> IA: Falha no download do treinamento: {e}", flush=True)
    return None


def _load_or_download_training_base(hs_stats: dict) -> dict:
    """Carrega base de treino (local ou GitHub).
    SEMPRE verifica se há versão mais nova no GitHub.
    Se houver, baixa e substitui o arquivo local.
    
    A base NUNCA sobrescreve dados LIVE do cliente."""

    # ── Verificar versão local atual ──
    local_version = hs_stats.get("meta", {}).get("deep_train_version", "")

    # ── Tentar carregar base local (veio com o app ou download anterior) ──
    base_data = None
    base_version = ""
    if os.path.exists(BASE_TRAINING_LOCAL):
        try:
            with open(BASE_TRAINING_LOCAL, "r", encoding="utf-8") as f:
                base_data = json.load(f)
            base_version = base_data.get("version", "")
            log.info(paint(f"📂 Base local encontrada: versão {base_version}", C.G))
        except Exception:
            base_data = None

    # ── SEMPRE verificar GitHub por versão mais nova ──
    remote = _download_training_base()
    if remote:
        remote_version = remote.get("version", "")
        if base_data is None:
            # Não tinha base local — usar GitHub
            log.info(paint(f"🆕 Base de treino baixada do GitHub: {remote_version}", C.G))
            base_data = remote
            base_version = remote_version
        elif remote_version > base_version:
            # GitHub tem versão mais nova — substituir
            log.info(paint(
                f"🆕 Atualização disponível: {remote_version} (local: {base_version})",
                C.G
            ))
            print(f">>> IA: Atualizando treinamento: {base_version} → {remote_version}", flush=True)
            base_data = remote
            base_version = remote_version
        else:
            log.info(paint(f"✅ Base local já é a mais recente ({base_version})", C.G))

        # Salvar localmente (novo download ou atualização)
        if base_data is remote:
            try:
                with open(BASE_TRAINING_LOCAL, "w", encoding="utf-8") as f:
                    json.dump(remote, f, indent=2, ensure_ascii=False)
                log.info(paint("💾 Base salva localmente para uso offline", C.G))
            except Exception:
                pass
    elif base_data is None:
        log.info(paint("📂 Sem base de treino pré-treinada — usará treino local", C.Y))
        print(">>> IA: Sem base pré-treinada e sem internet — treino local", flush=True)
        return hs_stats

    # ── MERGE: base pré-treinada + dados LIVE do cliente ──
    base_version = base_data.get("version", "unknown")
    
    # Se já carregou esta versão antes, pular
    if local_version == base_version:
        log.info(paint(f"📂 Base versão {base_version} já carregada — usando memória existente", C.G))
        return hs_stats

    log.info(paint(f"🔄 Aplicando base de treino versão {base_version}...", C.B))

    # Preservar dados LIVE do cliente
    _live_arms = {}
    for arm_key, arm_data in hs_stats.get("arms", {}).items():
        lw = arm_data.get("live_wins", 0)
        lt = arm_data.get("live_total", 0)
        recent = arm_data.get("recent", [])
        if lt > 0 or recent:
            _live_arms[arm_key] = {"live_wins": lw, "live_total": lt, "recent": recent}

    _live_geo = [g for g in hs_stats.get("geometry_history", []) if g.get("source") == "live"]

    # Carregar base
    new_stats = {
        "meta": base_data.get("meta", {}),
        "arms": base_data.get("arms", {}),
        "geometry_history": base_data.get("geometry_history", []),
    }

    # Re-aplicar dados LIVE do cliente
    for arm_key, live_data in _live_arms.items():
        if arm_key in new_stats["arms"]:
            new_stats["arms"][arm_key]["live_wins"] = live_data["live_wins"]
            new_stats["arms"][arm_key]["live_total"] = live_data["live_total"]
            new_stats["arms"][arm_key]["recent"] = live_data.get("recent", [])
            # Somar live nos totais
            new_stats["arms"][arm_key]["wins"] += live_data["live_wins"]
            new_stats["arms"][arm_key]["total"] += live_data["live_total"]
        else:
            new_stats["arms"][arm_key] = {
                "wins": live_data["live_wins"],
                "total": live_data["live_total"],
                "live_wins": live_data["live_wins"],
                "live_total": live_data["live_total"],
                "recent": live_data.get("recent", []),
            }

    # Adicionar geometria live
    new_stats["geometry_history"].extend(_live_geo)

    # Marcar versão
    new_stats["meta"]["deep_train_version"] = base_version
    new_stats["meta"]["entry_model"] = "iR_close_confirmation"
    new_stats["meta"]["last_bt_ts"] = time.time()

    _n_geo = len(new_stats.get("geometry_history", []))
    log.info(paint(
        f"✅ Base de treino v{base_version} aplicada! Geometria: {_n_geo} padrões",
        C.G
    ))

    # Salvar
    _safe_save_json(AI_STATS_FILE, new_stats)
    return new_stats


def _train_ia_from_history(bx, hs_stats: dict) -> dict:
    """Treina IA a partir do histórico de velas — MEMÓRIA PERMANENTE.
    Busca N_M1 candles para cada ativo top, detecta todos os padrões H&S,
    faz backtest de cada um, e ACUMULA WIN/LOSS nos stats existentes.

    PROTEÇÃO: Não re-acumula se já treinou recentemente (< 2h).
    Sem isso, cada restart DUPLICA os dados de backtest,
    fazendo os resultados LIVE terem peso insignificante."""
    ativos = obter_top_ativos_otc(bx)
    if not ativos:
        log.warning(paint("⚠️ Nenhum ativo para treino — IA mantém memória anterior", C.Y))
        return hs_stats

    # ── PROTEÇÃO ANTI-DUPLICAÇÃO ──
    # Cada restart re-adicionava os MESMOS padrões históricos.
    # Fix: só re-treinar se passaram >2h OU se CSVs 5000 existem e nunca foram usados.
    _CSV_DIR_CHECK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candles_5000")
    _csvs_exist = os.path.isdir(_CSV_DIR_CHECK) and len(os.listdir(_CSV_DIR_CHECK)) > 0
    _trained_with_csv = hs_stats.get("meta", {}).get("trained_with_csv", False)
    _force_csv_retrain = _csvs_exist and not _trained_with_csv

    _last_bt_ts = hs_stats.get("meta", {}).get("last_bt_ts", 0)
    _hours_since = (time.time() - _last_bt_ts) / 3600
    if _last_bt_ts > 0 and _hours_since < 2.0 and not _force_csv_retrain:
        _n_total = hs_stats.get("meta", {}).get("total", 0)
        log.info(paint(
            f"Backtest recente ({_hours_since:.1f}h atrás) — usando memória existente "
            f"({_n_total} padrões DT)",
            C.G
        ))

        # NN: Modelos per-ativo são carregados offline (train_neural_network.py)
        # NUNCA treinar NN online — só usar modelos pré-treinados

        return hs_stats

    if _force_csv_retrain:
        log.info(paint("📂 CSVs de 5000 velas detectados! Forçando re-treino profundo...", C.G))

    # ── RESET dos dados de backtest (mantém LIVE intacto) ──
    # Antes de re-treinar, remove APENAS os dados de backtest anteriores,
    # preservando live_wins, live_total e recent[] intactos.
    for _arm_key, _arm_data in hs_stats.get("arms", {}).items():
        _live_w = _arm_data.get("live_wins", 0)
        _live_t = _arm_data.get("live_total", 0)
        _arm_data["wins"] = _live_w
        _arm_data["total"] = _live_t
        # recent[] é LIVE-only, não mexer

    # Reset geometry_history — manter apenas registros LIVE (source="live")
    _old_geo = hs_stats.get("geometry_history", [])
    hs_stats["geometry_history"] = [g for g in _old_geo if g.get("source") == "live"]

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

    # ── Verificar CSVs de 5000 velas (treino offline) ──
    _CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candles_5000")
    _has_csv_dir = os.path.isdir(_CSV_DIR)
    if _has_csv_dir:
        log.info(paint(f"📂 CSVs de 5000 velas encontrados em {_CSV_DIR} — usando para treino profundo!", C.G))
    log.info(paint(f"🏋️ Treinando IA com {len(all_ativos)} ativos...", C.B))

    total_wins = 0
    total_losses = 0
    total_patterns = 0

    for ativo in all_ativos:
        try:
            # ── Priorizar CSV de 5000 velas (6x mais dados) ──
            df = None
            _csv_path = os.path.join(_CSV_DIR, f"{ativo}.csv") if _has_csv_dir else ""
            if _has_csv_dir and os.path.exists(_csv_path):
                try:
                    df = pd.read_csv(_csv_path)
                    df["time"] = pd.to_datetime(df["time"])
                    df.set_index("time", inplace=True)
                    needed = ["open", "high", "low", "close"]
                    for col in needed:
                        if col not in df.columns:
                            df = None
                            break
                    if df is not None:
                        df = df[needed].dropna().sort_index()
                        if len(df) < 100:
                            df = None
                        else:
                            log.info(paint(f"  📂 {ativo}: {len(df)} velas do CSV", C.G))
                except Exception:
                    df = None

            # Fallback: download da corretora (900 velas)
            if df is None:
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

            # Detectar pivots e somente Double Touch
            ph, pl = detect_pivots(H, L, window=5)
            all_dt = detect_double_touch(H, L, C_arr, O, ph, pl, atr, n,
                                         max_candles_ago=9999, training=True)
            all_patterns = all_dt

            if not all_patterns:
                continue

            _w, _l = 0, 0
            for pat in all_patterns:
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

                    # ── IA: armazenar geometria para aprendizado ──
                    geo = _extract_geometry(pat, atr)
                    if geo is not None:
                        geo["result"] = 1 if bt["result"] == "win" else 0
                        geo["ativo"] = ativo
                        geo["type"] = pat.get("type", "HS")
                        geo["source"] = "backtest"
                        if "geometry_history" not in hs_stats:
                            hs_stats["geometry_history"] = []
                        hs_stats["geometry_history"].append(geo)

                    # Stats globais (meta.total + meta.wins)
                    meta = hs_stats.setdefault("meta", {"total": 0})
                    meta["total"] = meta.get("total", 0) + 1
                    if bt["result"] == "win":
                        meta["wins"] = meta.get("wins", 0) + 1

                    total_patterns += 1

            if _w + _l > 0:
                log.info(f"  {ativo}: {len(all_patterns)} padrões | {_w}W / {_l}L")
            total_wins += _w
            total_losses += _l

            # NN: Modelos per-ativo são treinados offline (train_neural_network.py)
            # NUNCA alimentar NN online

        except Exception as e:
            log.debug(f"Erro treinando {ativo}: {e}")
            continue

    # NN: Modelos per-ativo são treinados offline (train_neural_network.py)
    # NUNCA treinar NN online

    _n_total = hs_stats.get("meta", {}).get("total", 0)
    _n_geo = len(hs_stats.get("geometry_history", []))
    _n_geo_wins = sum(1 for g in hs_stats.get("geometry_history", []) if g.get("result") == 1)
    _lvl_num, _lvl_nome, _lvl_emoji = _get_ia_level(_n_total)
    wr = (total_wins / max(total_wins + total_losses, 1)) * 100

    log.info(paint("=" * 50, C.G))
    log.info(paint(f"🏋️ TREINO CONCLUÍDO — MEMÓRIA PERMANENTE!", C.G))
    log.info(paint(f"  📊 Sessão: {total_patterns} padrões | {total_wins}W / {total_losses}L | WR: {wr:.1f}%", C.G))
    log.info(paint(f"  🧠 IA TOTAL: Memória DT atualizada", C.G))
    log.info(paint(f"  📐 IA Geométrica: {_n_geo} padrões aprendidos ({_n_geo_wins} wins)", C.G))
    # NN: modelos per-ativo são carregados depois, offline
    log.info(paint(f"  🧠 NN: Modelos per-ativo treinados offline (train_neural_network.py)", C.B))
    log.info(paint(f"  💾 Memória NUNCA é apagada — quanto mais roda, mais aprende!", C.G))
    log.info(paint("=" * 50, C.G))

    log.info(paint(f"  📊 Treino DT: Sessão {total_patterns} padrões | WR: {wr:.1f}%", C.G))

    # Marcar timestamp do backtest para evitar re-acumulação
    hs_stats.setdefault("meta", {})["last_bt_ts"] = time.time()
    # Marcar se treinou com CSVs de 5000 velas
    _CSV_DIR_MARK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candles_5000")
    if os.path.isdir(_CSV_DIR_MARK) and len(os.listdir(_CSV_DIR_MARK)) > 0:
        hs_stats["meta"]["trained_with_csv"] = True
        _csv_count = len([f for f in os.listdir(_CSV_DIR_MARK) if f.endswith(".csv")])
        hs_stats["meta"]["csv_assets_count"] = _csv_count
        log.info(paint(f"📂 IA treinada com CSVs de 5000 velas ({_csv_count} ativos)", C.G))

    # Limitar geometry_history a 500 registros (backtest + live)
    if len(hs_stats.get("geometry_history", [])) > 500:
        # Manter todos os live + últimos N backtest
        _live_g = [g for g in hs_stats["geometry_history"] if g.get("source") == "live"]
        _bt_g = [g for g in hs_stats["geometry_history"] if g.get("source") != "live"]
        _max_bt = 500 - len(_live_g)
        hs_stats["geometry_history"] = _live_g + _bt_g[-_max_bt:]

    # Salvar no disco — PERMANENTE
    _safe_save_json(AI_STATS_FILE, hs_stats)
    _save_retrain_control()

    return hs_stats


def _write_dashboard_cache(dashboard_assets: dict, payouts: dict, live_signals: Optional[list] = None,
                           selected_assets: Optional[list] = None):
    """Escreve cache compartilhado para o dashboard (read-only).
    O bot é a ÚNICA fonte de dados — dashboard nunca conecta ao broker."""
    try:
        cache = {
            "ts": time.time(),
            "broker": BROKER_TYPE,
            "analysis_source": "bot",
            "assets": {},
            "live_signals": list(live_signals or []),
            "selected_assets": list(selected_assets or []),
            "summary": {
                "total": 0,
                "wins": 0,
                "wr": 0.0,
                "by_asset": {},
                "by_type": {},
            },
        }
        _total_done = 0
        _total_wins = 0
        for ativo, info in dashboard_assets.items():
            df = info.get("df")
            if df is None or len(df) < 10:
                continue
            patterns = list(info.get("patterns") or [])
            # Guardar últimas 500 velas (dashboard mostra 120, mas precisa de mais para H&S)
            tail = df.tail(500)
            candles = []
            for ts, row in tail.iterrows():
                candles.append({
                    "t": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                    "o": round(float(row["open"]), 6),
                    "h": round(float(row["high"]), 6),
                    "l": round(float(row["low"]), 6),
                    "c": round(float(row["close"]), 6),
                })
            cache["assets"][ativo] = {
                "candles": candles,
                "payout": payouts.get(ativo, 0),
                "patterns": patterns,
                "market_regime": info.get("market_regime") or {},
            }

            _wins = sum(1 for pat in patterns if (pat.get("backtest") or {}).get("result") == "win")
            _losses = sum(1 for pat in patterns if (pat.get("backtest") or {}).get("result") == "loss")
            _live = sum(1 for pat in patterns if not pat.get("backtest"))
            _done = _wins + _losses
            _wr = (_wins / max(_done, 1) * 100.0) if _done > 0 else 0.0
            cache["summary"]["by_asset"][ativo] = {
                "wins": _wins,
                "total": _done,
                "wr": round(_wr, 1),
                "live": _live,
            }
            for pat in patterns:
                _ptype = str(pat.get("type", "DOUBLE_TOP") or "DOUBLE_TOP")
                _bucket = cache["summary"]["by_type"].setdefault(_ptype, {"wins": 0, "total": 0, "wr": 0.0})
                _res = (pat.get("backtest") or {}).get("result")
                if _res in ("win", "loss"):
                    _bucket["total"] += 1
                    if _res == "win":
                        _bucket["wins"] += 1
            _total_done += _done
            _total_wins += _wins

        cache["summary"]["total"] = int(_total_done)
        cache["summary"]["wins"] = int(_total_wins)
        cache["summary"]["wr"] = round((_total_wins / max(_total_done, 1)) * 100.0, 1) if _total_done > 0 else 0.0
        for _ptype, _bucket in cache["summary"]["by_type"].items():
            _bucket["wr"] = round((_bucket["wins"] / max(_bucket["total"], 1)) * 100.0, 1) if _bucket["total"] > 0 else 0.0
        _safe_save_json(_DASHBOARD_CACHE_FILE, cache)
    except Exception as e:
        log.debug(f"Erro ao escrever cache dashboard: {e}")


def escolher_melhor_setup_local(bx, cooldown_map: dict, hs_stats: dict,
                                reversal_ai_map: Optional[dict] = None,
                                early_only: bool = False):
    """Detecta Double Touch em TEMPO REAL — multi-asset.
    Busca candles DIRETO da corretora. Sem dashboard, sem JSON, sem delay.
    Varre TODOS os ativos elegíveis e devolve TODOS os candidatos ordenados.

    Returns (best_trade, best_any)."""
    ativos = obter_top_ativos_otc(bx)
    if not ativos:
        log.warning(paint("⚠️ Nenhum ativo OTC disponível", C.Y))
        return None, None

    best_trade = None
    best_any = None
    all_candidates = []
    _total_patterns = 0
    _scan_start = time.time()
    _dashboard_assets = {}  # acumula dados para o dashboard
    _dashboard_live_signals = []

    for ativo in ativos:
        # IMPORTANTE: sempre varrer TODOS os ativos antes da decisão final.
        # Não encerrar o scan ao encontrar o 1º padrão, senão a IA considera
        # apenas um ativo e ignora setups melhores nos demais pares.
        # ── Buscar candles DIRETO da corretora ──
        df = get_candles_df(bx, ativo, TF_M1, LIVE_SCAN_N_M1)
        if df is None or len(df) < 100:
            continue

        # Acumular para o dashboard (SEMPRE, independente de padrão)
        _dashboard_assets[ativo] = {"df": df, "patterns": [], "market_regime": {}}

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

        _market_regime = {}
        _dashboard_assets[ativo]["market_regime"] = {}

        # ── Detectar pivots e padrões H&S ──
        pivot_highs, pivot_lows = detect_pivots(H, L, window=5)

        # Snapshot único para o dashboard: padrões visíveis do próprio bot.
        _dashboard_patterns_raw = detect_double_touch(
            H, L, C_arr, O, pivot_highs, pivot_lows, atr, n,
            max_candles_ago=9999, training=True,
        )
        _dashboard_visible_start = max(0, n - 120)
        _dashboard_patterns = []
        for _dpat in _dashboard_patterns_raw:
            _rs_idx = int((_dpat.get("right_shoulder") or {}).get("idx", -1))
            if _rs_idx < _dashboard_visible_start:
                continue
            _dbt = backtest_pattern(_dpat, C_arr, O, H, L, n)
            if _dbt is not None and _dbt.get("result") not in ("win", "loss"):
                continue
            _d_ia = ai_predict_hs(ativo, _dpat, hs_stats)
            _dashboard_patterns.append(
                _serialize_dashboard_pattern(
                    ativo,
                    _dpat,
                    df,
                    ia_prob=_d_ia,
                    backtest=_dbt,
                    scan_ts=_scan_start,
                    market_regime=None,
                )
            )
        _dashboard_assets[ativo]["patterns"] = _dashboard_patterns

        # ── SOMENTE Double Touch (H&S removido — DT tem WR melhor no live) ──
        # Live: aceitar apenas sinais muito recentes para evitar entrada em padrão velho.
        # A revalidação final antes da ordem confirma que o DT continua fresco.
        patterns = detect_double_touch(
            H, L, C_arr, O, pivot_highs, pivot_lows, atr, n,
            max_candles_ago=MAX_LIVE_SIGNAL_CANDLES,
        )

        if not patterns:
            continue

        _detected_recent = len(patterns)

        # ── Filtrar: só padrões LIVE (sem resultado ainda) — IGUAL ao dashboard ──
        live_patterns = []
        for pat in patterns:
            bt = backtest_pattern(pat, C_arr, O, H, L, n)
            if bt is None:
                # Padrão recente sem resultado = sinal LIVE
                entry_idx = pat.get("entry_idx", pat["right_shoulder"]["idx"] + 1)
                pat["entry_pending"] = entry_idx >= n
                pat["candles_ago"] = max(0, n - 1 - pat["right_shoulder"]["idx"])
                pat["scan_ts"] = _scan_start

                if pat["candles_ago"] > MAX_LIVE_SIGNAL_CANDLES:
                    log.info(paint(
                        f"  ⛔ SKIP: {ativo} candles_ago={pat['candles_ago']} > {MAX_LIVE_SIGNAL_CANDLES} "
                        f"(sinal antigo para live)",
                        C.Y
                    ))
                    continue

                # ═══ FIX LIVE #1: candles_ago=0 OK — scan :50 + entrada :00 ═══
                # Antes bloqueava candles_ago < 1, causando entrada 1 vela atrasada.
                # Agora aceita candles_ago=0: scan detecta às :50, aguarda :00 (vela
                # fecha), entra no OPEN da próxima vela — mesma lógica do dashboard.
                # A vela está 83% completa às :50, features suficientes para detecção.
                if False:  # DESATIVADO — sync corrigido para não atrasar 1 vela
                    pass

                # ═══ FIX LIVE #2: Somente modo CLASSIC ou EARLY em live ═══
                # Classic = 3 pivots confirmados por detect_pivots (window=5).
                # Early = LS+Head confirmados, RS por filtro de rejeição (delay≈0).
                # Realtime = argmax/argmin com apenas 2 barras de validação = FRACO.
                if pat.get("mode") == "realtime":
                    log.info(paint(
                        f"  ⛔ SKIP: {ativo} modo=realtime "
                        f"(somente classic/early com pivots confirmados em live)",
                        C.Y
                    ))
                    continue

                # ═══ FIX LIVE #3: REMOVIDO ═══
                # O filtro de proximidade do Ombro D foi testado em 6 backtests
                # e provado INEFICAZ (não melhora o WR). O edge do padrão
                # é determinado pelo delay, não pela posição do preço.

                # ═══ FIX LIVE #4: fallback legado para modos não-DT ═══
                # O bloqueio principal do live agora é candles_ago <= MAX_LIVE_SIGNAL_CANDLES.
                if pat.get("mode") not in ("early", "double_touch") and pat["candles_ago"] > 2:
                    log.info(paint(
                        f"  ⛔ SKIP: {ativo} candles_ago={pat['candles_ago']} > 2 "
                        f"(delay muito alto — WR degrada abaixo de 55%)",
                        C.Y
                    ))
                    continue

                live_patterns.append(pat)

        if not live_patterns:
            log.info(paint(
                f"  📭 {ativo}: {_detected_recent} DT recente(s) detectado(s), mas nenhum ficou fresco/live para entrada",
                C.Y
            ))
            continue

        # ── DEDUPLICAR: 1 padrão por tipo+direção, manter o mais recente ──
        _dedup = {}
        for p in live_patterns:
            _key = f"{p['type']}_{p['direction']}"
            _ago = p.get("candles_ago", 99)
            if _key not in _dedup or _ago < _dedup[_key].get("candles_ago", 99):
                _dedup[_key] = p
        live_patterns = list(_dedup.values())

        # ── CONFLITO: se mesmo ativo tem PUT + CALL, sinal ambíguo → SKIP ──
        # ── SEM FILTRO DISTÂNCIA — apenas IA decide ──
        _total_patterns += len(live_patterns)

        for pat in live_patterns:
            direction = pat["direction"]
            pat_type = pat["type"]
            mode = pat.get("mode", "classic")

            # ═══ MEMÓRIA DT: Bloquear nível já operado ═══
            # IA estatística legada — mantida só para log/contexto.
            ia_prob = ai_predict_hs(ativo, pat, hs_stats)
            ia_n = hs_stats.get("arms", {}).get(f"{ativo}_{pat_type}_{mode}", {}).get("total", 0)

            # IA Geométrica — mantida só para log/contexto.
            _pq, _ = ia_pattern_quality(pat, atr, hs_stats)
            _scan_geo = _extract_geometry(pat, atr)

            if mode == "double_touch":
                _scan_geo_guard = _dt_geometry_scan_filter(_scan_geo, _pq)
                log.info(paint(
                    f"  📐 GEOMETRIA {DT_LIVE_PROFILE.upper()}: {ativo} {direction} — {_scan_geo_guard['reason']}",
                    C.G if _scan_geo_guard.get("ok") else C.Y
                ))
                if not _scan_geo_guard.get("ok"):
                    log.info(paint(
                        f"  ⚠️ GEOMETRIA ADVISORY: {ativo} {direction} | {_scan_geo_guard['reason']} | Bayes decide",
                        C.Y
                    ))

                _entry_region = _validate_dt_entry_region(pat, float(C_arr[-1]), atr)
                log.info(paint(
                    f"  📍 REGIAO DE ENTRADA: {ativo} {direction} — {_entry_region['reason']}",
                    C.G if _entry_region.get("ideal", _entry_region.get("ok")) else C.Y
                ))
                if not _entry_region.get("ok"):
                    log.info(paint(
                        f"  ⚠️ REGIAO ADVISORY: {ativo} {direction} | {_entry_region['reason']} | Bayes decide",
                        C.Y
                    ))
            else:
                _entry_region = None

            # Score principal de seleção = NN. Fallback = IA estatística legada.
            _nn_pre_score, _nn_pre_pred, _nn_pre_reason = _estimate_dt_nn_score(
                ativo, pat, df, atr, hs_stats, reversal_ai_map, return_reason=True
            )
            _entry_guard_pre_score, _entry_guard_pre_pred = _estimate_entry_guard_score(
                ativo, pat, df, atr, hs_stats
            )
            _entry_win_signature = ((_entry_guard_pre_pred or {}).get("win_signature") or {}) if isinstance(_entry_guard_pre_pred, dict) else {}
            if DT_GRAPH_SIGNAL_ENTRY and _nn_pre_score is not None:
                score = float(_nn_pre_score)
            elif _nn_pre_score is not None and _entry_guard_pre_score is not None:
                score = float(_nn_pre_score) * 0.75 + float(_entry_guard_pre_score) * 0.25
            elif _nn_pre_score is not None:
                score = float(_nn_pre_score)
            elif _entry_guard_pre_score is not None:
                score = float(_entry_guard_pre_score) * 0.55 + float(ia_prob) * 0.45
            else:
                score = float(ia_prob)
            if _entry_win_signature.get("matched") and not DT_GRAPH_SIGNAL_ENTRY:
                score += float(_entry_win_signature.get("bonus", 0.0) or 0.0)
            score = max(0.0, min(float(score), 0.9999))

            _touch_continuation = _detect_dt_touch_continuation_signal(pat, df, atr)
            if not _touch_continuation.get("matched"):
                log.info(paint(
                    f"  ⚠️ CONTINUACAO ADVISORY: {ativo} {direction} | {_touch_continuation.get('reason')} | Bayes decide",
                    C.Y
                ))
            else:
                log.info(paint(
                    f"  ✅ 2o TOQUE + CONTINUACAO: {ativo} {direction} | {_touch_continuation.get('reason')}",
                    C.G
                ))

            _study_multifactor = _dt_multifactor_study_profile(
                pat,
                df,
                _entry_region,
                _touch_continuation,
                _entry_guard_pre_pred,
            )
            log.info(paint(
                f"  🧪 ESTUDO MULTIFATOR: {ativo} {direction} | {_study_multifactor.get('reason')}",
                C.G if _study_multifactor.get("premium") else (C.Y if _study_multifactor.get("ok", True) else C.R)
            ))
            if _study_multifactor.get("hard_block"):
                log.info(paint(
                    f"  ⚠️ MULTIFATOR ADVISORY: {ativo} {direction} | {_study_multifactor.get('reason')} | Bayes decide",
                    C.Y
                ))

            _study_score_boost = float(_study_multifactor.get("score_boost", 0.0) or 0.0)
            if _study_score_boost > 0:
                score = max(0.0, min(float(score) + _study_score_boost, 0.9999))

            # ═══ PORTÃO DE PERFEIÇÃO — Alinhado ao treino ═══
            # No treino NÃO existe nenhum guard de vela/toque.
            # detect_double_touch → backtest_pattern → WIN/LOSS.
            # A NN foi treinada com TODOS os tipos de vela.
            # Aqui apenas logamos para diagnóstico.
            _perf_signal = _study_multifactor.get("signal_candle_class", "unknown")
            _perf_touch = _study_multifactor.get("touch_state", "missing")

            if _perf_signal in ("weak_or_mixed", "doji_indecision", "pin_rejection"):
                log.info(paint(
                    f"  ⚠️ VELA ADVISORY: {ativo} {direction} | "
                    f"signal={_perf_signal} body={_study_multifactor.get('body_ratio', 0):.0%} | "
                    f"NN decide (treino inclui este tipo)",
                    C.Y
                ))

            # Guard 2: Toque — apenas advisory (no treino, o toque é confirmado pela
            # própria detecção de pivot matching, sem check de continuidade).
            # A NN recebe f1:close_position, f13:rejection_quality, f24:body_conviction
            # que já codificam a qualidade do toque.
            if _perf_touch == "missing":
                log.info(paint(
                    f"  ⚠️ TOQUE ADVISORY: {ativo} {direction} | "
                    f"touch={_perf_touch} | Continuidade não confirmada — NN decide",
                    C.Y
                ))

            # Guard 3: Região de entrada — ADVISORY (no treino não existe
            # filtro de distância; a NN já aprendeu a geometria via features).
            if _entry_region is not None and not _entry_region.get("ok"):
                log.info(paint(
                    f"  ⚠️ REGIÃO ADVISORY: {ativo} {direction} | "
                    f"{_entry_region.get('reason')} — NN decide",
                    C.Y
                ))

            _direction_alignment_2m = ((_entry_guard_pre_pred or {}).get("direction_alignment_2m") or {}) if isinstance(_entry_guard_pre_pred, dict) else {}
            if not _direction_alignment_2m.get("aligned"):
                _align_reason = _direction_alignment_2m.get("reason", "entry_guard sem confirmacao direcional 2m")
                log.info(paint(
                    f"  ⚠️ DIRECAO 2M ADVISORY: {ativo} {direction} | {_align_reason} | Bayes decide",
                    C.Y
                ))

            if _direction_alignment_2m.get("aligned"):
                log.info(paint(
                    f"  ✅ DIRECAO 2M ALINHADA: {ativo} {direction} | {_direction_alignment_2m.get('reason')}",
                    C.G
                ))

            _entry_trigger = _entry_guard_past_win_trigger(_entry_guard_pre_pred, _touch_continuation)
            if (not _entry_trigger.get("triggered")) and _study_multifactor.get("trigger_release"):
                _entry_trigger = {
                    "triggered": True,
                    "reason": f"study premium release | {_study_multifactor.get('reason')}",
                    "min_prob": _entry_trigger.get("min_prob"),
                }
            if not _entry_trigger.get("triggered"):
                log.info(paint(
                    f"  ⚠️ TRIGGER ADVISORY: {ativo} {direction} | {_entry_trigger.get('reason')} | Bayes decide",
                    C.Y
                ))

            if _entry_trigger.get("triggered"):
                log.info(paint(
                    f"  ✅ TRIGGER DE WIN PASSADO: {ativo} {direction} | {_entry_trigger.get('reason')}",
                    C.G
                ))

            _win_geometry_alignment = _validate_dt_win_geometry_alignment(
                ativo,
                pat,
                atr,
                hs_stats,
                _entry_guard_pre_pred,
            )
            if not _win_geometry_alignment.get("ok"):
                log.info(paint(
                    f"  ⚠️ GEOMETRIA DOS WINS ADVISORY: {ativo} {direction} | {_win_geometry_alignment.get('reason')} | Bayes decide",
                    C.Y
                ))

            if _win_geometry_alignment.get("ok"):
                log.info(paint(
                    f"  ✅ GEOMETRIA ALINHADA AOS WINS: {ativo} {direction} | {_win_geometry_alignment.get('reason')}",
                    C.G
                ))

            _quality_risk = _entry_guard_quality_risk_filter(
                _entry_guard_pre_pred,
                _touch_continuation,
                _entry_region,
                _win_geometry_alignment,
            )
            if not _quality_risk.get("ok"):
                log.info(paint(
                    f"  ⚠️ RISCO ADVISORY: {ativo} {direction} | {_quality_risk.get('reason')} | Bayes decide",
                    C.Y
                ))

            if _quality_risk.get("ok"):
                log.info(paint(
                    f"  ✅ QUALIDADE/RISCO OK: {ativo} {direction} | {_quality_risk.get('reason')}",
                    C.G
                ))

            _training_alignment = _training_alignment_check(_entry_guard_pre_pred, _entry_trigger)
            if not _training_alignment.get("ok"):
                log.info(paint(
                    f"  ⚠️ TREINO ADVISORY: {ativo} {direction} | {_training_alignment.get('reason')} | Bayes decide",
                    C.Y
                ))

            if _training_alignment.get("ok"):
                log.info(paint(
                    f"  ✅ ALINHADO AO TREINO: {ativo} {direction} | {_training_alignment.get('reason')}",
                    C.G
                ))

            _prediction_2m = _build_dt_prediction_2m(
                pat,
                df,
                atr,
                _entry_guard_pre_pred,
                _touch_continuation,
            )
            _shadow_pattern_lib = _build_shadow_dt_library_comparison(pat, df, atr)
            _gpt_scan_result = {
                "available": False,
                "approved": None,
                "confidence": None,
                "reason": "IA generativa aguardando validacao live",
                "source": None,
                "exp_minutes": None,
                "latency_ms": None,
                "stage": "scan",
            }
            if mode == "double_touch":
                if _nn_pre_pred is not None and bool(_nn_pre_pred.get("approved", False)) and ENABLE_CONTEXT_FILTER:
                    try:
                        # ── CONTEXT TABLE: consultar WR histórico do backtest ──
                        _ctx_geo = _extract_geometry(pat, atr)
                        import datetime as _dt_mod
                        _ctx_hour = _dt_mod.datetime.utcnow().hour
                        _ctx_result = context_lookup(
                            ativo=ativo, direcao=direction,
                            hour=_ctx_hour,
                            depth_ratio=_ctx_geo["depth_ratio"] if _ctx_geo else 0,
                            symmetry=_ctx_geo["symmetry"] if _ctx_geo else 0,
                        )
                        _ctx_log = format_context_log(_ctx_result)
                        log.info(paint(f"  {_ctx_log}", C.B))

                        _gpt_scan_result = {
                            "available": True,
                            "approved": _ctx_result["action"] != "block",
                            "confidence": _ctx_result["wr"],
                            "reason": _ctx_result["reason"],
                            "source": f"ctx_L{_ctx_result['level']}",
                            "exp_minutes": None,
                            "latency_ms": 0,
                            "stage": "scan",
                        }
                    except Exception as _ctx_scan_err:
                        _gpt_scan_result = {
                            "available": False,
                            "approved": None,
                            "confidence": None,
                            "reason": f"erro context scan: {_ctx_scan_err}",
                            "source": None,
                            "exp_minutes": None,
                            "latency_ms": None,
                            "stage": "scan",
                        }
                        log.info(paint(
                            f"  ⚠️ Context scan erro: {ativo} {direction} | {_ctx_scan_err}",
                            C.Y
                        ))
                elif _nn_pre_pred is None:
                    _gpt_scan_result["reason"] = "NN indisponivel no scan"
                elif not bool(_nn_pre_pred.get("approved", False)):
                    _gpt_scan_result["reason"] = "NN do scan nao aprovou"
                else:
                    _gpt_scan_result["reason"] = "Context filter desativado"

            _ai_consensus = _build_ai_cotrader_consensus(
                mode,
                ia_prob,
                _nn_pre_pred,
                _gpt_scan_result,
                _shadow_pattern_lib,
            )
            if not _ai_consensus.get("final_ok"):
                log.info(paint(
                    f"  ⛔ CONSENSO IA: {ativo} {direction} | {_ai_consensus.get('reason')}",
                    C.Y
                ))
                continue

            log.info(paint(
                f"  ✅ CONSENSO IA: {ativo} {direction} | {_ai_consensus.get('reason')}",
                C.G if _ai_consensus.get("gpt_ok") is not False else C.Y
            ))
            _timing_hint = _build_dt_graph_timing_hint(
                pat,
                _touch_continuation,
                _entry_region,
                _prediction_2m,
                _nn_pre_pred,
                df=df,
                current_price=float(df["close"].values[-1]) if df is not None and len(df) > 0 else None,
                atr_val=atr,
            )
            _timing_hint = _force_dt_entry_at_turn(_timing_hint, _nn_pre_pred)
            if _timing_hint.get("available"):
                _timing_label = "ENTRAR AGORA" if _timing_hint.get("action") == "now" else f"ESPERAR {_timing_hint.get('wait_seconds', 0):.1f}s"
                log.info(paint(
                    f"  ⏱️ TIMING DT: {ativo} {direction} | {_timing_hint.get('zone')} | {_timing_label} | {_timing_hint.get('reason')}",
                    C.G if _timing_hint.get("action") == "now" else C.Y
                ))

            setup = {
                "dir": direction,
                "type": pat_type,
                "mode": mode,
                "confidence": round(score * 100, 1),
                "pattern": pat,
                "last_close": float(C_arr[-1]),  # preço atual para guards (evita 2ª API call)
                "last_close_prev": float(C_arr[-2]) if n >= 2 else float(C_arr[-1]),
                "scan_geometry": _scan_geo,
                "geometry_score": round(float(_pq), 4),
                "nn_pre_score": round(float(_nn_pre_score), 4) if _nn_pre_score is not None else None,
                "nn_pre_pred": _nn_pre_pred,
                "nn_pre_reason": _nn_pre_reason,
                "entry_guard_pre_score": round(float(_entry_guard_pre_score), 4) if _entry_guard_pre_score is not None else None,
                "entry_guard_pre_pred": _entry_guard_pre_pred,
                "entry_win_signature": _entry_win_signature,
                "entry_region": _entry_region,
                "entry_trigger": _entry_trigger,
                "study_multifactor": _study_multifactor,
                "quality_risk": _quality_risk,
                "win_geometry_alignment": _win_geometry_alignment,
                "training_alignment": _training_alignment,
                "touch_continuation": _touch_continuation,
                "prediction_2m": _prediction_2m,
                "timing_hint": _timing_hint,
                "shadow_pattern_lib": _shadow_pattern_lib,
                "gpt_pre_result": _gpt_scan_result,
                "ai_consensus": _ai_consensus,
            }

            if _shadow_pattern_lib.get("available"):
                _shadow_color = C.G if _shadow_pattern_lib.get("agreement") else C.Y
                log.info(paint(
                    f"  🧪 SHADOW LIB DT: {ativo} {direction} | {_shadow_pattern_lib.get('reason')}",
                    _shadow_color
                ))

            _dashboard_live_signals.append(
                _serialize_dashboard_pattern(
                    ativo,
                    pat,
                    df,
                    ia_prob=ia_prob,
                    backtest=None,
                    nn_pred=_nn_pre_pred,
                    scan_ts=_scan_start,
                    market_regime=None,
                    entry_guard_pred=_entry_guard_pre_pred,
                    touch_continuation=_touch_continuation,
                    prediction_2m=_prediction_2m,
                    timing_hint=_timing_hint,
                    shadow_pattern_lib=_shadow_pattern_lib,
                    gpt_result=_gpt_scan_result,
                    ai_consensus=_ai_consensus,
                )
            )

            if _nn_pre_score is not None:
                _score_label = f"NN={_nn_pre_score:.2f}"
            else:
                _score_label = "fallback estatístico"
            _entry_guard_label = f"{_entry_guard_pre_score:.2f}" if _entry_guard_pre_score is not None else "n/a"
            _win_sig_label = _entry_win_signature.get("tier") if _entry_win_signature.get("matched") else "off"
            _entry_trigger_label = "on" if _entry_trigger.get("triggered") else "off"

            log.info(paint(
                f"  📊 DT LOCAL: {ativo} | {pat_type} {direction} ({mode}) | "
                f"score={score:.2f} ({_score_label} | geom={_pq:.2f}) | "
                f"entry_guard={_entry_guard_label} | win_sig={_win_sig_label} | trigger={_entry_trigger_label} | entry_idx={pat['entry_idx']} | n={n}",
                C.B
            ))

            if best_any is None or score > best_any[0]:
                best_any = (score, ativo, setup, atr)

            # ═══ TODOS OS CANDIDATOS — NN/IA avaliam cada um ═══
            all_candidates.append((score, ativo, setup, atr))

            if best_trade is None or score > best_trade[0]:
                best_trade = (score, ativo, setup, atr)

    if _total_patterns > 0:
        log.info(paint(f"  🔍 Scan local: {_total_patterns} padrão(ões) DT recente(s) encontrado(s)", C.G))

    # ── Escrever cache para o dashboard (atualiza a cada scan) ──
    try:
        _payouts = {}
        try:
            all_profit = safe_call(bx, bx.get_all_profit)
            for a in _dashboard_assets:
                p = all_profit.get(a, {}).get("turbo", 0)
                _payouts[a] = int(p * 100) if p and p <= 1 else int(p) if p else 0
        except Exception:
            pass
        _write_dashboard_cache(_dashboard_assets, _payouts, live_signals=_dashboard_live_signals, selected_assets=ativos)
    except Exception:
        pass

    # Ordenar candidatos por score (melhor primeiro) após scan completo dos ativos
    all_candidates.sort(key=lambda x: x[0], reverse=True)
    return all_candidates, best_any


def wait_until_minus(tf, seconds_before):
    """Espera até `seconds_before` segundos antes do fechamento do candle."""
    while True:
        s = tf - (time.time() % tf)
        if s <= seconds_before:
            return
        time.sleep(min(s - seconds_before, 1.0))



def _extract_geometry(pat, atr_val):
    """Extrai features geométricas de um padrão H&S para a IA aprender.
    Inclui features extras descobertas na análise de 5000 velas."""
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
        # Features extras da análise profunda (5000 velas)
        pL = pat["left_shoulder"]["price"]
        pR = pat["right_shoulder"]["price"]
        # DT: valley1==valley2 (mesmo ponto) → usar alinhamento dos toques
        if pat.get("mode") == "double_touch":
            neck_align = abs(pL - pR) / atr_val if atr_val > 0 else 0
        else:
            neck_align = abs(v1 - v2) / atr_val if atr_val > 0 else 0
        shoulder_ratio = min(pL, pR) / max(pL, pR) if max(pL, pR) > 0 else 0
        neck_slope_norm = abs(pat.get("neck_slope", 0)) / atr_val if atr_val > 0 else 0
        return {
            "span": span,
            "symmetry": round(symmetry, 4),
            "depth_ratio": round(depth_ratio, 4),
            "neck_align": round(neck_align, 4),
            "d_left": d_left,
            "d_right": d_right,
            "shoulder_ratio": round(shoulder_ratio, 6),
            "neck_slope_norm": round(neck_slope_norm, 6),
        }
    except Exception:
        return None


def ia_pattern_quality(pat, atr_val, stats_ai=None):
    """IA que APRENDE da geometria dos padrões + filtros empíricos (5000 velas).

    Combina:
    1. Perfil estatístico dos WINs (aprendizado adaptativo)
    2. Filtros empíricos da análise de 5000 velas por ativo

    Filtros empíricos descobertos (análise profunda):
    ─ depth_ratio ≤ 2.28  → WR 95.6% (n=45)
    ─ d_right ≤ 15        → WR 97.1% (n=34)
    ─ span ≤ 28           → WR 96.6% (n=29)
    ─ neck_slope_norm ≥ 0.019 → WR 100% (n=19)
    ─ d_left ≤ 12         → WR 100% (n=19)
    ─ shoulder_ratio ≥ 0.9999 → WR 100% (n=19)

    Retorna fator 0.50-1.0 + motivos.
    """
    # Duplo Toque: IA aprende geometria IGUAL ao dashboard.
    # Compara features geométricas contra perfil estatístico dos WINs.
    # Se não tem dados suficientes, retorna 1.0 (neutro).
    if pat.get("mode") == "double_touch":
        geo = _extract_geometry(pat, atr_val)
        if geo is None or stats_ai is None:
            return 1.0, []
        # Usar geometry_history do stats_ai (85K amostras)
        _all_geo = stats_ai.get("geometry_history", [])
        if len(_all_geo) < 10:
            return 1.0, []
        win_geos = [g for g in _all_geo if g.get("result") == 1]
        if len(win_geos) < 5:
            return 1.0, []
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
            return 1.0, []
        final = score_sum / n_feat
        return round(max(0.50, min(1.0, final)), 4), []

    geo = _extract_geometry(pat, atr_val)
    if geo is None or stats_ai is None:
        return 1.0, []

    motivos = []

    # ══════════════════════════════════════════════════════════
    # PARTE 1: Filtros empíricos (análise 5000 velas)
    # Cada filtro aprovado dá bônus; filtro violado penaliza.
    # ══════════════════════════════════════════════════════════
    empirical_bonus = 0.0
    empirical_checks = 0

    # depth_ratio <= 2.28 → WR 95.6%
    dr = geo.get("depth_ratio", 0)
    if dr <= 2.28:
        empirical_bonus += 1.0
    else:
        empirical_bonus += 0.0
        motivos.append(f"depth_ratio={dr:.2f}>2.28")
    empirical_checks += 1

    # d_right <= 15 → WR 97.1%
    d_r = geo.get("d_right", 99)
    if d_r <= 15:
        empirical_bonus += 1.0
    elif d_r <= 22:
        empirical_bonus += 0.5
    else:
        empirical_bonus += 0.0
        motivos.append(f"d_right={d_r}>22")
    empirical_checks += 1

    # span <= 28 → WR 96.6%
    sp = geo.get("span", 99)
    if sp <= 28:
        empirical_bonus += 1.0
    elif sp <= 40:
        empirical_bonus += 0.5
    else:
        empirical_bonus += 0.0
        motivos.append(f"span={sp}>40")
    empirical_checks += 1

    # neck_slope_norm >= 0.019 → WR 100%
    nsn = geo.get("neck_slope_norm", 0)
    if nsn >= 0.019:
        empirical_bonus += 1.0
    elif nsn >= 0.008:
        empirical_bonus += 0.5
    else:
        empirical_bonus += 0.0
    empirical_checks += 1

    # d_left <= 12 → WR 100%
    d_l = geo.get("d_left", 99)
    if d_l <= 12:
        empirical_bonus += 1.0
    elif d_l <= 20:
        empirical_bonus += 0.5
    else:
        empirical_bonus += 0.0
    empirical_checks += 1

    # shoulder_ratio >= 0.9999 → WR 100%
    sr = geo.get("shoulder_ratio", 0)
    if sr >= 0.9999:
        empirical_bonus += 1.0
    elif sr >= 0.9995:
        empirical_bonus += 0.5
    else:
        empirical_bonus += 0.0
    empirical_checks += 1

    # Empirical score: 0.0 a 1.0
    emp_score = empirical_bonus / empirical_checks if empirical_checks > 0 else 0.5

    # ══════════════════════════════════════════════════════════
    # PARTE 2: Perfil estatístico dos WINs (aprendizado)
    # ══════════════════════════════════════════════════════════
    _all_geo = stats_ai.get("geometry_history", [])
    geo_history = [g for g in _all_geo if g.get("source") != "live"]

    if len(geo_history) >= 10:
        win_geos = [g for g in geo_history if g.get("result") == 1]
        if len(win_geos) >= 5:
            features = ["span", "symmetry", "depth_ratio", "neck_align",
                        "d_left", "d_right", "shoulder_ratio", "neck_slope_norm"]
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
                if feat_score < 0.85:
                    motivos.append(f"{feat}={current_val:.2f}(avg={mean_w:.2f})")

            if n_feat > 0:
                learned_score = score_sum / n_feat
                # Blend: 50% empírico + 50% aprendido
                final = emp_score * 0.50 + learned_score * 0.50
                # Escala para range 0.50-1.0
                final = 0.50 + final * 0.50
                return round(max(0.50, min(1.0, final)), 4), motivos

    # ══════════════════════════════════════════════════════════
    # HARD BLOCK: padrão com ≥3 violações empíricas → BLOQUEAR (pq < 0.50)
    # Previne padrões gigantes (span=82, d_right=50) de passar.
    # ══════════════════════════════════════════════════════════
    if len(motivos) >= 3:
        return 0.40, motivos  # Abaixo de 0.50 → IA bloqueia entry

    # Sem dados de aprendizado suficientes → usa só empírico
    final = 0.50 + emp_score * 0.50
    return round(max(0.50, min(1.0, final)), 4), motivos


def ai_predict_hs(ativo, pat, stats_ai):
    """IA prediction para setup H&S — com suavização Bayesiana + WR recente.

    Fixes críticos:
    1. Bayesian smoothing Beta(2,2) — evita WR extremos com poucos dados
    2. Janela deslizante (últimos 30) — IA adapta a condições ATUAIS
    3. Blend 60% recente + 40% histórico — não congela no backtest
    4. Backtest limitado a 30 amostras — live data tem peso real
    5. Fallback hierárquico ponderado — NÃO para no primeiro fallback raso
    6. Fallback GLOBAL final usa WR real (87%+) em vez de 0.50 neutro
    """
    arms = stats_ai.get("arms", {})
    # Key específica: ativo_type_mode
    key = f"{ativo}_{pat.get('type', 'HS')}_{pat.get('mode', 'classic')}"
    data = arms.get(key, None)
    if data and data.get("total", 0) >= 3:
        # ── Limitar influência do backtest (máx 30 amostras) ──
        live_w = data.get("live_wins", 0)
        live_t = data.get("live_total", 0)
        bt_w = data["wins"] - live_w
        bt_t = data["total"] - live_t
        if bt_t > 30:
            scale = 30.0 / bt_t
            bt_w = round(bt_w * scale)
            bt_t = 30
        eff_w = bt_w + live_w
        eff_t = bt_t + live_t
        # Bayesian smoothing: prior Beta(2,2) — nunca retorna valores extremos
        bayesian_wr = (eff_w + 2) / (eff_t + 4)
        # ── Windowed WR: últimos 30 resultados LIVE ──
        recent = data.get("recent", [])
        if len(recent) >= 8:
            recent_wr = sum(recent) / len(recent)
            # Blend: 60% recente (adapta rápido) + 40% histórico (estabilidade)
            return round(recent_wr * 0.6 + bayesian_wr * 0.4, 4)
        return round(bayesian_wr, 4)

    # ────────────────────────────────────────────────────────────────
    # FALLBACK HIERÁRQUICO PONDERADO (não para no primeiro raso)
    # ────────────────────────────────────────────────────────────────
    pat_type = pat.get("type", "HS")
    pat_mode = pat.get("mode", "classic")

    # Nível 1: mesmo tipo + modo (ex: INV_HEAD_SHOULDERS_realtime)
    f1_w, f1_t = 0, 0
    for k, v in arms.items():
        if f"_{pat_type}_{pat_mode}" in k:
            f1_w += v.get("wins", 0)
            f1_t += v.get("total", 0)

    # Nível 2: mesmo tipo qualquer modo (ex: INV_HEAD_SHOULDERS_*)
    f2_w, f2_t = 0, 0
    for k, v in arms.items():
        if f"_{pat_type}_" in k:
            f2_w += v.get("wins", 0)
            f2_t += v.get("total", 0)

    # Nível 3: GLOBAL — todos os arms (H&S + INV juntos)
    f3_w, f3_t = 0, 0
    for k, v in arms.items():
        f3_w += v.get("wins", 0)
        f3_t += v.get("total", 0)

    # ── Blend ponderado por amostras: mais dados = mais peso ──
    candidates = []
    if f1_t >= 3:
        candidates.append(((f1_w + 2) / (f1_t + 4), f1_t))
    if f2_t >= 5:
        candidates.append(((f2_w + 2) / (f2_t + 4), f2_t))
    if f3_t >= 10:
        candidates.append(((f3_w + 2) / (f3_t + 4), f3_t))

    if candidates:
        # Weighted average: peso = sqrt(amostras) para balancear
        import math
        total_weight = sum(math.sqrt(n) for _, n in candidates)
        blended = sum(prob * math.sqrt(n) for prob, n in candidates) / total_weight
        return round(blended, 4)

    # ── Fallback final: usa meta global se disponível ──
    meta = stats_ai.get("meta", {})
    meta_total = meta.get("total", 0)
    meta_wins = meta.get("wins", 0)
    if meta_total >= 10:
        return round((meta_wins + 2) / (meta_total + 4), 4)

    return 0.5  # sem NENHUM dado — conservador


def ai_predict(ativo, setup, stats_ai):
    """IA prediction para setup H&S (compatibilidade) — com suavização Bayesiana."""
    arm = f"{ativo}_{setup.get('type', 'HS')}_{setup.get('mode', 'classic')}"
    arm_data = stats_ai.get("arms", {}).get(arm, {"wins": 0, "total": 0})
    n = arm_data.get("total", 0)
    w = arm_data.get("wins", 0)
    # Bayesian smoothing: prior Beta(2,2) → evita WR extremos
    prob = (w + 2) / (n + 4) if n > 0 else 0.5
    # Windowed WR blend (mais responsivo a condições atuais)
    recent = arm_data.get("recent", [])
    if len(recent) >= 8:
        recent_wr = sum(recent) / len(recent)
        prob = recent_wr * 0.6 + prob * 0.4
    conf = min(n / 10.0, 1.0)
    return {"prob": round(prob, 4), "n_arm": n, "conf": conf}


def ai_update(ativo, setup, result_value, stats_ai):
    """Atualiza IA stats após trade H&S — com tracking LIVE separado.

    Separa resultados LIVE dos de backtest para que:
    1. Backtest não afogue os resultados reais
    2. Janela recente reflita performance ATUAL
    3. IA adapte a mudanças de mercado em tempo real
    4. Geometria do padrão é armazenada para IA aprender
    """
    if "arms" not in stats_ai:
        stats_ai["arms"] = {}
    arm = f"{ativo}_{setup.get('type', 'HS')}_{setup.get('mode', 'classic')}"
    if arm not in stats_ai["arms"]:
        stats_ai["arms"][arm] = {"wins": 0, "total": 0}
    d = stats_ai["arms"][arm]
    d["total"] += 1
    if result_value > 0:
        d["wins"] += 1
    # ── Track LIVE results separadamente (não contaminado por backtest) ──
    d["live_total"] = d.get("live_total", 0) + 1
    if result_value > 0:
        d["live_wins"] = d.get("live_wins", 0) + 1
    # ── Janela deslizante: últimos 30 resultados LIVE ──
    recent = d.get("recent", [])
    recent.append(1 if result_value > 0 else 0)
    if len(recent) > 30:
        recent = recent[-30:]
    d["recent"] = recent
    meta = stats_ai.setdefault("meta", {"total": 0})
    meta["total"] = meta.get("total", 0) + 1
    if result_value > 0:
        meta["wins"] = meta.get("wins", 0) + 1

    # ── IA: Armazenar geometria do padrão para aprendizado contínuo ──
    pat = setup.get("pattern", setup)
    atr_val = setup.get("atr", 0)
    geo = _extract_geometry(pat, atr_val)
    if geo is not None:
        geo["result"] = 1 if result_value > 0 else 0
        geo["ativo"] = ativo
        geo["type"] = setup.get("type", "HS")
        geo["source"] = "live"
        if "geometry_history" not in stats_ai:
            stats_ai["geometry_history"] = []
        stats_ai["geometry_history"].append(geo)
        # Manter últimas 200 geometrias para não crescer infinitamente
        if len(stats_ai["geometry_history"]) > 200:
            stats_ai["geometry_history"] = stats_ai["geometry_history"][-200:]

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
                   end_ts: Optional[float] = None,
                   min_len: int = 50) -> Optional[pd.DataFrame]:
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
        if len(df) < min_len:
            return None
        return df
    except Exception:
        return None


def get_last_closed_candles_df(bx: BrokerAPI, ativo: str, timeframe: int, n: int,
                               min_len: int = 1) -> Optional[pd.DataFrame]:
    try:
        _tf_sec = int(timeframe) if int(timeframe) > 0 else 60
    except Exception:
        _tf_sec = 60
    _now = time.time()
    _last_closed_end = _now - (_now % _tf_sec) - 0.001
    return get_candles_df(bx, ativo, timeframe, n, end_ts=_last_closed_end, min_len=min_len)


def get_realtime_candles_df(bx: BrokerAPI, ativo: str, timeframe: int, n: int,
                            min_len: int = 1) -> Optional[pd.DataFrame]:
    """Lê velas do stream realtime já aberto, sem nova chamada histórica."""
    try:
        candles = safe_call(bx, bx.get_realtime_candles, ativo, timeframe)
        if not candles or not isinstance(candles, dict):
            return None

        rows = []
        for ts in sorted(candles.keys())[-max(2, n):]:
            candle = candles.get(ts) or {}
            rows.append({
                "time": pd.to_datetime(int(ts), unit="s"),
                "open": float(candle.get("open", 0) or 0),
                "high": float(candle.get("max", candle.get("high", 0)) or 0),
                "low": float(candle.get("min", candle.get("low", 0)) or 0),
                "close": float(candle.get("close", 0) or 0),
            })

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df.set_index("time", inplace=True)
        df = df[["open", "high", "low", "close"]].dropna().sort_index()
        if len(df) < min_len:
            return None
        return df
    except Exception:
        return None


def get_realtime_entry_snapshot(bx: BrokerAPI, ativo: str, timeframe: int,
                                closed_n: int = 6) -> Tuple[Optional[float], Optional[pd.DataFrame]]:
    """Retorna preço corrente e histórico fechado diretamente do stream realtime."""
    df = get_realtime_candles_df(bx, ativo, timeframe, max(2, closed_n + 1), min_len=1)
    if df is None or len(df) == 0:
        return None, None

    now = time.time()
    bucket_start = pd.to_datetime(int(now - (now % timeframe)), unit="s")
    current_price = float(df["close"].iloc[-1])

    if len(df) >= 2 and df.index[-1] >= bucket_start:
        closed_df = df.iloc[:-1].tail(closed_n)
    else:
        closed_df = df.tail(closed_n)

    if closed_df is not None and len(closed_df) == 0:
        closed_df = None
    return current_price, closed_df




def _estimate_dt_nn_score(ativo: str, pat: dict, df: Optional[pd.DataFrame], atr_val: float,
                          hs_stats: dict, reversal_ai_map: Optional[dict] = None,
                          return_reason: bool = False):
    if df is None:
        return (None, None, "df ausente") if return_reason else (None, None)
    if len(df) < 50:
        return (None, None, f"candles insuficientes ({len(df)}/50)") if return_reason else (None, None)
    if reversal_ai_map is None:
        return (None, None, "reversal_ai_map ausente") if return_reason else (None, None)

    _rai = reversal_ai_map.get(ativo)
    if _rai is None or not getattr(_rai, "_ai1_ready", False) or not getattr(_rai, "_ai2_ready", False):
        _reason = (
            "modelo não carregado"
            if _rai is None
            else f"modelo parcial IA1={'OK' if getattr(_rai, '_ai1_ready', False) else 'OFF'} IA2={'OK' if getattr(_rai, '_ai2_ready', False) else 'OFF'}"
        )
        return (None, None, _reason) if return_reason else (None, None)

    try:
        _dt_context_candles = 110  # Mesmo contexto usado no treino offline DT.
        _H = df["high"].values
        _L = df["low"].values
        _C = df["close"].values
        _O = df["open"].values
        _n = len(_H)
        _rs_idx = int(pat.get("right_shoulder", {}).get("idx", _n - 1))
        _rs_idx = max(0, min(_rs_idx, _n - 1))

        _win_start = max(0, _rs_idx - _dt_context_candles)
        _win_end = min(_n, _rs_idx + 2)  # Inclui até 1 vela pós-RS para features RE-CHECK
        _H_win = _H[_win_start:_win_end]
        _L_win = _L[_win_start:_win_end]
        _C_win = _C[_win_start:_win_end]
        _O_win = _O[_win_start:_win_end]
        _n_win = len(_H_win)
        if _n_win < 25:
            return (None, None, f"janela curta ({_n_win}/25)") if return_reason else (None, None)

        _atr_local_vals = [float(_H_win[k] - _L_win[k]) for k in range(max(0, _n_win - 14), _n_win)]
        _atr_local = float(np.mean(_atr_local_vals)) if _atr_local_vals else atr_val
        if _atr_local <= 0:
            _atr_local = atr_val

        _pat_copy = dict(pat)
        _pat_copy["candles_ago"] = max(0, _n_win - 1 - (_rs_idx - _win_start))
        _feats = extract_features(_pat_copy, _H_win, _L_win, _C_win, _O_win, _n_win,
                                  _atr_local, hs_stats, ativo)
        if _feats is None:
            return (None, None, "extract_features retornou None") if return_reason else (None, None)

        _pred = _rai.predict_dt(_feats)
        if _pred is None:
            return (None, None, "predict_dt retornou None") if return_reason else (None, None)

        _pred = dict(_pred)
        _pred.setdefault("approved", bool(_pred.get("win", False)))
        _pred.setdefault("available", True)
        _pred["trained_metrics"] = {
            "ai1_val": round(float(getattr(_rai, "_ai1_val", 0.0) or 0.0), 4) if getattr(_rai, "_ai1_ready", False) else None,
            "ai2_val": round(float(getattr(_rai, "_ai2_val", 0.0) or 0.0), 4) if getattr(_rai, "_ai2_ready", False) else None,
            "ai3_val": round(float(getattr(_rai, "_ai3_val", 0.0) or 0.0), 4) if getattr(_rai, "_ai3_ready", False) else None,
            "n_samples": int(getattr(_rai, "_loaded_n_samples", 0) or 0),
            "n_features": int(getattr(_rai, "_n_features_trained", len(_feats)) or len(_feats)),
            "context_candles": _dt_context_candles,
        }

        return (_pred.get("nn_score"), _pred, None) if return_reason else (_pred.get("nn_score"), _pred)
    except Exception as ex:
        return (None, None, f"erro em _estimate_dt_nn_score: {ex}") if return_reason else (None, None)


def _is_dt_nn_model_ready(reversal_ai_map: Optional[dict], ativo: str) -> bool:
    if not reversal_ai_map:
        return False
    _rai = reversal_ai_map.get(ativo)
    return bool(
        _rai is not None
        and getattr(_rai, "_ai1_ready", False)
        and getattr(_rai, "_ai2_ready", False)
    )


def _ensure_reversal_model_loaded(reversal_ai_map: Optional[dict], ativo: str) -> bool:
    if reversal_ai_map is None:
        return False

    if ativo not in reversal_ai_map:
        reversal_ai_map[ativo] = ReversalAI(ativo)

    _rai = reversal_ai_map.get(ativo)
    if _rai is None:
        return False
    if getattr(_rai, "_ws_load_logged", False):
        return _is_dt_nn_model_ready(reversal_ai_map, ativo)

    if _is_dt_nn_model_ready(reversal_ai_map, ativo):
        _rai.save_stats_to_disk()
        log.info(paint(
            f"  ✅ NN {ativo}: {_rai._loaded_n_samples} amostras | "
            f"IA1={_rai._ai1_val:.1%} IA2={_rai._ai2_val:.1%}"
            + (f" IA3={_rai._ai3_val:.1%}" if _rai._ai3_ready else ""),
            C.G
        ))
    elif getattr(_rai, "_ai1_ready", False) or getattr(_rai, "_ai2_ready", False):
        log.info(paint(
            f"  ⚠️ NN {ativo}: carga parcial do modelo "
            f"(IA1={'OK' if getattr(_rai, '_ai1_ready', False) else 'OFF'} "
            f"IA2={'OK' if getattr(_rai, '_ai2_ready', False) else 'OFF'}) — snapshot NN desativado",
            C.Y
        ))
    else:
        log.info(paint(f"  ⚠️ NN {ativo}: modelo não encontrado", C.Y))

    setattr(_rai, "_ws_load_logged", True)
    return _is_dt_nn_model_ready(reversal_ai_map, ativo)


# ═══════════════════════════════════════════════════════════════
# ATIVOS OTC / PAYOUT
# ═══════════════════════════════════════════════════════════════
_cache_ativos: List[str] = []
_cache_ativos_ts: float = 0.0
_top_dt_assets: List[str] = []  # TOP N ativos DT (N definido por benchmark)


def _pick_top_dt_assets(hs_stats: dict, n_top: int = 4) -> List[str]:
    """Retorna o pool ranqueado pelos modelos entry-guard treinados por ativo."""
    ranked = _rank_assets_by_entry_guard()
    if ranked:
        filtered_ranked = [
            item for item in ranked
            if _is_high_accuracy_asset(item[1], item[2], item[3], item[4])
        ]
        conservative_ranked = [
            item for item in ranked
            if _is_conservative_fallback_asset(item[1], item[2], item[3], item[4])
        ]
        if filtered_ranked:
            chosen_rank = filtered_ranked
        elif conservative_ranked:
            chosen_rank = conservative_ranked
        else:
            chosen_rank = []

        if not chosen_rank:
            log.warning(paint(
                "⚠️ Nenhum ativo passou nos filtros de qualidade do entry guard — varredura DT ficará pausada até haver ativos confiáveis",
                C.Y
            ))
            return []

        pool_size = min(len(chosen_rank), max(n_top, _ENTRY_GUARD_POOL_SIZE))
        top = [item[0] for item in chosen_rank[:pool_size]]
        if filtered_ranked:
            log.info(paint(
                f"🎯 Pool dinâmico: {len(filtered_ranked)} ativo(s) | acc>={_ENTRY_GUARD_MIN_ACC:.0%} | auc>={_ENTRY_GUARD_MIN_AUC:.2f} | prec>={_ENTRY_GUARD_MIN_PREC:.0%}",
                C.G
            ))
        elif conservative_ranked:
            log.warning(paint(
                f"⚠️ Nenhum ativo passou no filtro alto — usando pool conservador | acc>={_ENTRY_GUARD_FALLBACK_MIN_ACC:.0%} | auc>={_ENTRY_GUARD_FALLBACK_MIN_AUC:.2f} | prec>={_ENTRY_GUARD_FALLBACK_MIN_PREC:.0%}",
                C.Y
            ))
        else:
            log.warning(paint("⚠️ Pool de ativos vazio após filtros de qualidade", C.Y))
        for i, (asset, acc, auc, precision, samples) in enumerate(chosen_rank[:min(pool_size, len(chosen_rank))]):
            log.info(paint(
                f"🎯 ASSET #{i+1}: {asset} | acc={acc:.1%} | auc={auc:.3f} | prec={precision:.1%} | amostras={samples}",
                C.G
            ))
        return top

    top = _get_full_pattern_qualified_assets()[:n_top] or ["NZDJPY-OTC", "GBPAUD-OTC", "USDCAD-OTC", "EURNZD-OTC"][:n_top]
    for i, a in enumerate(top[:n_top]):
        log.info(paint(f"🎯 ASSET #{i+1}: {a} (fallback sem entry_guard treinado)", C.Y))
    return top


def obter_top_ativos_otc(bx: BrokerAPI) -> List[str]:
    global _cache_ativos, _cache_ativos_ts, _top_dt_assets
    # Congela os ativos escolhidos no boot para evitar rotação durante a sessão.
    if _cache_ativos:
        return _cache_ativos

    if not _top_dt_assets:
        _top_dt_assets = _pick_top_dt_assets({}, n_top=max(NUM_ATIVOS, _ENTRY_GUARD_POOL_SIZE))

    # Verificar quais estão abertos na corretora
    targets = list(_top_dt_assets)
    turbo = {}
    try:
        dados = safe_call(bx, bx.get_all_open_time)
        turbo = dados.get("turbo", {})
        abertos = [a for a in targets if a in turbo and turbo[a].get("open", False)]
        if not abertos:
            log.warning(paint(f"⚠️ Nenhum ativo OTC aberto — usando targets originais", C.Y))
        targets = abertos if abertos else targets
    except Exception:
        pass

    # Verificar payouts — remover apenas os com payout < mínimo
    good_targets = []
    payouts_map = {}
    try:
        all_profit = safe_call(bx, bx.get_all_profit)
        for t in targets:
            profit = all_profit.get(t, {}).get("turbo", 0)
            payout = int(profit * 100) if profit else 0
            payouts_map[t] = payout
            if payout < PAYOUT_MINIMO:
                log.warning(paint(f"⚠️ {t} payout={payout}% (mín={PAYOUT_MINIMO}%)", C.Y))
            else:
                log.info(paint(f"✅ {t} payout={payout}% OK", C.G))
                good_targets.append(t)
    except Exception:
        good_targets = targets

    # Se nenhum passou no payout, usar os targets mesmo assim
    if not good_targets:
        good_targets = targets
        for t in good_targets:
            payouts_map.setdefault(t, 0)

    ranked_metrics = {
        asset: (acc, auc, precision, samples)
        for asset, acc, auc, precision, samples in _rank_assets_by_entry_guard()
    }
    dynamic_rank = []
    for asset in good_targets:
        acc, auc, precision, samples = ranked_metrics.get(asset, (0.0, 0.0, 0.0, 0))
        analysis = _score_live_asset_candidate(bx, asset, acc, payout=payouts_map.get(asset, 0))
        if analysis is None:
            payout_score = min(max(float(payouts_map.get(asset, 0)) / 100.0, 0.0), 1.0)
            analysis = {
                "asset": asset,
                "selection_score": round(float(acc) * 0.45 + payout_score * 0.05, 4),
                "accuracy": round(float(acc), 4),
                "visible_count": 0,
                "live_count": 0,
                "regime_ok": True,
                "regime_score": 0.5,
                "regime_reason": "fallback sem leitura de candles",
                "payout": int(payouts_map.get(asset, 0)),
            }
        analysis["auc"] = round(float(auc), 4)
        analysis["precision"] = round(float(precision), 4)
        analysis["samples"] = int(samples)
        dynamic_rank.append(analysis)

    dynamic_rank.sort(
        key=lambda item: (
            item.get("selection_score", 0.0),
            item.get("live_count", 0),
            item.get("visible_count", 0),
            item.get("accuracy", 0.0),
            item.get("regime_score", 0.0),
            item.get("payout", 0),
        ),
        reverse=True,
    )

    _scan_count = SCAN_NUM_ATIVOS
    _cache_ativos = [item["asset"] for item in dynamic_rank[:_scan_count]] if dynamic_rank else good_targets[:_scan_count]
    _cache_ativos_ts = time.time()
    if dynamic_rank:
        log.info(paint(
            f"🎯 Rotação dinâmica DT: pool={len(dynamic_rank)} | selecionando {len(_cache_ativos)} ativo(s) por acc+padrões+lateral",
            C.G
        ))
        for idx, item in enumerate(dynamic_rank[:len(_cache_ativos)]):
            _regime_flag = "lateral" if item.get("regime_ok") else "não-lateral"
            log.info(paint(
                f"🎯 SCAN #{idx+1}: {item['asset']} | sel={item['selection_score']:.3f} | acc={item['accuracy']:.1%} | "
                f"entry={item['live_count']} | vis={item['visible_count']} | {_regime_flag}={item['regime_score']:.2f} | payout={item['payout']}%",
                C.G if item.get("regime_ok") else C.Y
            ))
    log.info(paint(f"🎯 ATIVOS EM VARREDURA ({len(_cache_ativos)}): {_cache_ativos}", C.G))
    return _cache_ativos


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
                    confidence: float = 0.0, status: str = "entry",
                    nn_data: dict = None,
                    decision_id: Optional[str] = None,
                    order_id: Optional[int] = None):
    """Grava trade no log para consumo pelo dashboard.
    Salva no arquivo JSON E envia POST ao dashboard (tempo real)."""
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
        "decision_id": decision_id,
        "order_id": int(order_id) if order_id is not None else None,
    }
    if nn_data:
        record["nn_approved"] = nn_data.get("approved", True)
        record["nn_available"] = nn_data.get("available")
        record["nn_source"] = nn_data.get("source")
        record["nn_reason"] = nn_data.get("reason")
        record["nn_p1"] = nn_data.get("p1", 0)
        record["nn_p2"] = nn_data.get("p2", 0)
        record["nn_p3"] = nn_data.get("p3")
        record["nn_score"] = nn_data.get("nn_score", 0)
        record["consensus_penalty"] = nn_data.get("consensus_penalty", 0)
    # ── 1) Salvar no arquivo JSON (persistência) ──
    try:
        trades = []
        if os.path.exists(LIVE_LOG_FILE):
            with open(LIVE_LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                trades = data.get("trades", [])
        # Se é resultado (win/loss/tie), atualizar o último "entry" do mesmo ativo
        if status in ("win", "loss", "tie"):
            updated = False
            for i in range(len(trades) - 1, -1, -1):
                same_decision = decision_id and trades[i].get("decision_id") == decision_id
                same_order = order_id is not None and trades[i].get("order_id") == int(order_id)
                fallback_match = trades[i].get("ativo") == ativo and trades[i].get("status") == "entry"
                if same_decision or same_order or fallback_match:
                    trades[i]["status"] = status
                    trades[i]["resultado"] = resultado
                    trades[i]["ts"] = record["ts"]
                    trades[i]["time"] = record["time"]
                    if decision_id:
                        trades[i]["decision_id"] = decision_id
                    if order_id is not None:
                        trades[i]["order_id"] = int(order_id)
                    updated = True
                    break
            if not updated:
                trades.append(record)
        else:
            trades.append(record)
        if len(trades) > _LIVE_LOG_MAX:
            trades = trades[-_LIVE_LOG_MAX:]
        with open(LIVE_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump({"trades": trades, "updated": time.time()}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    # ── 2) POST ao dashboard (tempo real, não bloqueia) ──
    try:
        import urllib.request
        payload = json.dumps({
            "ativo": ativo, "dir": direcao, "result": status,
            "price": entry_price or 0, "stake": stake,
            "profit": resultado or 0,
            "time": record["time"][-8:-3],
            "ts": record["ts"], "broker": _broker_suffix,
            "decision_id": decision_id,
            "order_id": int(order_id) if order_id is not None else None,
        }).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:8899/api/trade",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass  # Dashboard pode não estar rodando


# ═══════════════════════════════════════════════════════════════
# TRADE DECISION LOG (para HTML viewer)
# ═══════════════════════════════════════════════════════════════
_DECISION_LOG_FILE = os.path.join(_user_data_dir, "ws_trade_decisions.json")
_DECISION_LOG_MAX = 200


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
    return value


def _write_json_atomic(file_path: str, payload):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp_path = file_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, file_path)


def _save_trade_decision(decision: dict):
    """Salva decisão completa do trade para análise no HTML viewer."""
    try:
        decisions = []
        if os.path.exists(_DECISION_LOG_FILE):
            with open(_DECISION_LOG_FILE, "r", encoding="utf-8-sig") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    decisions = loaded
        decisions.append(_json_safe(decision))
        if len(decisions) > _DECISION_LOG_MAX:
            decisions = decisions[-_DECISION_LOG_MAX:]
        _write_json_atomic(_DECISION_LOG_FILE, decisions)
    except Exception as ex:
        log.warning(paint(f"[DECISION LOG] falha ao salvar: {ex}", C.Y))


def _update_trade_decision_result(ativo: str, direcao: str, resultado: float, status: str,
                                  decision_id: Optional[str] = None,
                                  order_id: Optional[int] = None):
    """Atualiza o resultado no último decision log do mesmo ativo/direção."""
    try:
        if not os.path.exists(_DECISION_LOG_FILE):
            return
        with open(_DECISION_LOG_FILE, "r", encoding="utf-8-sig") as f:
            decisions = json.load(f)
        for i in range(len(decisions) - 1, -1, -1):
            d = decisions[i]
            same_decision = decision_id and d.get("decision_id") == decision_id
            same_order = order_id is not None and d.get("order_id") == int(order_id)
            fallback_match = d.get("ativo") == ativo and d.get("direcao") == direcao and d.get("resultado") is None
            if same_decision or same_order or fallback_match:
                d["resultado"] = round(resultado, 2)
                d["status"] = status
                if decision_id:
                    d["decision_id"] = decision_id
                if order_id is not None:
                    d["order_id"] = int(order_id)
                break
        _write_json_atomic(_DECISION_LOG_FILE, decisions)
    except Exception as ex:
        log.warning(paint(f"[DECISION LOG] falha ao atualizar resultado: {ex}", C.Y))


# ═══════════════════════════════════════════════════════════════
# ORDEM + RESULTADO
# ═══════════════════════════════════════════════════════════════
def enviar_ordem(bx: BrokerAPI, ativo: str, direcao: str, stake: float, exp: int = None) -> Optional[Tuple[str, int]]:
    """Envia ordem (TURBO → DIGITAL fallback). Expiração em minutos."""
    d = "call" if direcao == "CALL" else "put"
    valor = float(max(VALOR_MINIMO, stake))
    exp_min = exp or EXP_FIXA

    # TURBO
    try:
        ok, op_id = safe_call(bx, bx.buy, valor, ativo, d, exp_min)
        if ok and op_id:
            return ("turbo", int(op_id))
        log.warning(paint(f"[ORDEM] TURBO falhou ok={ok} id={op_id}", C.Y))
    except Exception as e:
        log.warning(paint(f"[ORDEM] TURBO exc: {e}", C.Y))

    # DIGITAL fallback
    try:
        ok, op_id = safe_call(bx, bx.buy_digital_spot, ativo, valor, d, exp_min)
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

# Horário em HORAS (engine usa hora_atual = datetime.now().hour)
HORARIO_INICIO = HORARIO_INICIO_MIN // 60   # 90 // 60 = 1 (≈ 1h30)
HORARIO_FIM    = HORARIO_FIM_MIN    // 60   # 1080 // 60 = 18


def escolher_melhor_setup(bx, ativos_ignored=None):
    """Wrapper de compatibilidade para ws_auto_ai_engine.py.
    escolher_melhor_setup_local já busca ativos internamente."""
    _stats = _safe_load_json(AI_STATS_FILE)
    return escolher_melhor_setup_local(bx, cooldown, _stats)


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
    Se já estamos nos primeiros 15s do candle novo, entra IMEDIATAMENTE
    (scan pode atrasar — melhor entrar 10s atrasado que 1min atrasado).
    Usa spin-lock fino nos últimos 50ms para precisão."""
    now = time.time()
    sec_in_candle = now % 60
    s = 60 - sec_in_candle
    # Se já estamos nos primeiros 15s do candle, entra DIRETO (não espera +55s)
    if sec_in_candle < 15:
        if sec_in_candle > 2:
            log.info(paint(f"  ⚡ Entrando {sec_in_candle:.0f}s após virada (scan demorou)", C.Y))
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
# MAIN — LOOP PRINCIPAL (SOMENTE DOUBLE TOUCH)
# ═══════════════════════════════════════════════════════════════
def main():
    # ── LOCK: Impedir duas instâncias do bot rodando ao mesmo tempo ──
    if not _acquire_lock():
        log.warning(paint(
            "⚠️ BOT JÁ ESTÁ RODANDO (outra instância detectada) — ABORTANDO",
            C.Y
        ))
        print(">>> IA: ABORTADO — outra instância do bot já está ativa", flush=True)
        return
    try:
        _main_inner()
    finally:
        _release_lock()

def _main_inner():
    bx: Optional[BrokerAPI] = None
    _ensure_dashboard_server()
    _seed_bundled_models()
    bx = ensure_connected(bx)

    if _ENTRY_GUARD_ENABLED:
        _sync_entry_guard_models_from_github()

    # Inicializar ReversalAI POR ATIVO — cada ativo tem sua própria NN
    # Os modelos são carregados do disco (treinados offline via train_neural_network.py)
    # NUNCA treinar NN online — só usar os modelos pré-treinados
    reversal_ai_map = {}  # {ativo: ReversalAI} — preenchido após selecionar ativos

    # ── Carregar mapa de inversão per-asset (gerado offline) ──
    _INVERSION_MAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ws_inversion_map.json")
    _inversion_map = {}
    if os.path.exists(_INVERSION_MAP_FILE):
        try:
            with open(_INVERSION_MAP_FILE, "r", encoding="utf-8") as _imf:
                _inversion_map = json.load(_imf)
            log.info(paint(f"📊 Mapa de inversão carregado: {len(_inversion_map)} ativos", C.G))
            _n_keep = sum(1 for v in _inversion_map.values() if v.get("action") == "keep")
            _n_inv = sum(1 for v in _inversion_map.values() if v.get("action") == "invert")
            _n_skip = sum(1 for v in _inversion_map.values() if v.get("action") == "skip")
            log.info(paint(f"   MANTER={_n_keep} | INVERTER={_n_inv} | PULAR={_n_skip}", C.B))
        except Exception as _ime:
            log.warning(paint(f"⚠️ Erro ao carregar mapa de inversão: {_ime}", C.Y))

    # ── Carregar / Treinar IA — MEMÓRIA PERMANENTE ──
    # A IA NUNCA perde memória. Carrega do disco e ACUMULA.
    log.info(paint("🧠 Carregando memória da IA DT...", C.B))

    # SEMPRE carregar stats existentes do disco (memória permanente)
    hs_stats = _safe_load_json(AI_STATS_FILE)
    _n_total = hs_stats.get("meta", {}).get("total", 0)

    if _n_total > 0:
        log.info(paint("💾 IA carregada do disco! Memória DT OK", C.G))
    else:
        log.info(paint("🌱 Primeira execução da IA...", C.Y))

    # ── 100% NN: Modelos pré-treinados (PKL) + Context Table ──
    log.info(paint("✅ Modo 100% NN — modelos pré-treinados (sem retreino)", C.G))

    log.info("=" * 60)
    log.info(paint(f"🚀 WS TRADER — Double Touch / Multi-Asset ({_BROKER_LABEL})", C.G))
    log.info("=" * 60)

    # ── SELECIONAR MELHOR ATIVO DT (baseado nos entry-guard models) ──
    global _top_dt_assets
    _top_dt_assets = _pick_top_dt_assets(hs_stats, n_top=max(SCAN_NUM_ATIVOS, _ENTRY_GUARD_POOL_SIZE))

    log.info(f"✅ Estratégia: SOMENTE Double Touch (Duplo Toque)")
    log.info(f"✅ Pool ranqueado de ativos: {_top_dt_assets}")
    log.info(f"✅ Varredura simultânea de ativos: {SCAN_NUM_ATIVOS}")
    log.info(f"✅ Live scan candles: {LIVE_SCAN_N_M1}")
    log.info(f"✅ Corretora: {_BROKER_LABEL} ({BROKER_TYPE})")
    log.info(f"✅ Expiração: {EXP_FIXA} minuto(s)")
    log.info(f"✅ Sinais: Detecção LOCAL (direto da corretora, sem delay)")
    log.info(f"✅ Modelos NN: pré-treinados offline (PKL imutáveis)")
    log.info(f"✅ Context Table: filtro estatístico de 78k trades")
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

    global _consecutive_losses
    total_trades = 0
    total_wins = 0
    _consecutive_losses = 0
    _current_day = _date_cls.today()

    # ── Restaurar contadores W/L do dia (sobrevive a reinícios) ──
    try:
        if os.path.exists(LIVE_LOG_FILE):
            with open(LIVE_LOG_FILE, "r", encoding="utf-8") as f:
                _saved = json.load(f).get("trades", [])
            _hoje_str = _current_day.isoformat()
            for t in _saved:
                _t_date = t.get("time", "")[:10]
                if _t_date == _hoje_str and t.get("status") in ("win", "loss", "tie"):
                    total_trades += 1
                    if t["status"] == "win":
                        total_wins += 1
            if total_trades > 0:
                log.info(paint(f"📊 Restaurado: {total_trades} trades (W={total_wins}) do dia", C.G))
    except Exception:
        pass

    _last_trade_time = 0.0
    _last_entry_key = _load_last_entry_key()  # Dedup persistente: sobrevive a reinícios

    # ── Carregar memória de níveis DT (impede 3º toque) ──
    global _dt_level_memory
    _dt_level_memory = _load_dt_level_memory()
    if _dt_level_memory:
        _n_mem = sum(len(v) for v in _dt_level_memory.values())
        log.info(paint(f"📋 Memória DT carregada: {_n_mem} nível(is) em {len(_dt_level_memory)} ativo(s)", C.B))

    print(f"\n>>> IA: Iniciado | Exp: {EXP_FIXA}min | Sinais: Detecção LOCAL", flush=True)

    # ── Rede Neural per-ativo: verificar se modelos estão prontos ──
    _nn_ready_count = sum(1 for _nn_asset in reversal_ai_map if _is_dt_nn_model_ready(reversal_ai_map, _nn_asset))
    if _nn_ready_count > 0:
        _nn_details = []
        for _nn_a, _nn_r in reversal_ai_map.items():
            if _is_dt_nn_model_ready(reversal_ai_map, _nn_a):
                _nn_s = _nn_r.get_stats()
                _nn_details.append(f"{_nn_a}={_nn_s.get('samples',0)}")
        print(f">>> NN: {_nn_ready_count}/{len(reversal_ai_map)} modelos carregados | " + " | ".join(_nn_details), flush=True)
    else:
        log.info(paint(
            f"  ⚠️ NN: Nenhum modelo per-ativo encontrado — Guard NN desativado. "
            f"Execute train_neural_network.py para treinar.",
            C.Y
        ))
        print(f">>> NN: Nenhum modelo per-ativo — Guard NN desativado", flush=True)

    # ═══ THREAD LIVE CANDLES: Lê streaming real-time do dicionário interno ═══
    # A inscrição no stream é feita NO LOOP PRINCIPAL (após primeiro scan)
    # para não conflitar com get_candles. A thread APENAS lê o dict e salva.
    _stream_subscribed = set()  # ativos já inscritos (compartilhado com loop)
    _stream_ready = threading.Event()  # sinaliza quando stream está pronto

    def _live_candle_thread(broker_ref, subscribed_ref, ready_event):
        """Thread daemon que exporta velas em tempo real para o dashboard.
        Lê APENAS o dicionário real_time_candles (instantâneo, sem rede).
        Nunca chama get_candles nem start_candles_stream."""
        _live_interval = 5  # 5 segundos (dashboard puxa a cada 5s)

        while True:
            try:
                _bx = broker_ref[0]
                if _bx is None or not ready_event.is_set():
                    time.sleep(_live_interval)
                    continue
                _ativos = _cache_ativos or _top_dt_assets
                if not _ativos:
                    time.sleep(_live_interval)
                    continue

                live_data = {"ts": time.time(), "assets": {}}

                for _a in _ativos:
                    try:
                        if _a not in subscribed_ref:
                            continue
                        # Leitura instantânea do dicionário interno (populado via WebSocket)
                        _rt = _bx.get_realtime_candles(_a, TF_M1)
                        if _rt and isinstance(_rt, dict) and len(_rt) > 0:
                            _clist = []
                            for _ts in sorted(_rt.keys())[-5:]:
                                _c = _rt[_ts]
                                _clist.append({
                                    "t": int(_ts),
                                    "o": round(float(_c.get("open", 0)), 6),
                                    "h": round(float(_c.get("max", _c.get("high", 0))), 6),
                                    "l": round(float(_c.get("min", _c.get("low", 0))), 6),
                                    "c": round(float(_c.get("close", 0)), 6),
                                })
                            if _clist:
                                live_data["assets"][_a] = _clist
                    except Exception:
                        pass

                if live_data["assets"]:
                    try:
                        _tmp_file = _LIVE_CANDLE_FILE + ".tmp"
                        with open(_tmp_file, "w") as _f:
                            json.dump(live_data, _f)
                        os.replace(_tmp_file, _LIVE_CANDLE_FILE)
                    except Exception:
                        try:
                            with open(_LIVE_CANDLE_FILE, "w") as _f:
                                json.dump(live_data, _f)
                        except Exception:
                            pass

            except Exception:
                pass
            time.sleep(_live_interval)

    # Referência mutável para o broker (thread pode ver reconexões)
    _bx_ref = [bx]
    _live_thread = threading.Thread(
        target=_live_candle_thread,
        args=(_bx_ref, _stream_subscribed, _stream_ready),
        daemon=True
    )
    _live_thread.start()
    log.info(paint("📡 Thread Live Candles iniciada (aguardando inscrição no stream)", C.G))

    # ═══ LOOP PRINCIPAL — SINAIS DO DASHBOARD DT ═══
    while True:
        try:
            bx = ensure_connected(bx)
            _bx_ref[0] = bx  # Atualizar referência para thread live candles

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

            # ── Verificar horario de operacao (DESATIVADO PARA TESTE) ──
            # _now = datetime.now()
            # _minutos_atual = _now.hour * 60 + _now.minute
            # if _minutos_atual < HORARIO_INICIO_MIN or _minutos_atual >= HORARIO_FIM_MIN:
            #     ...  # horário desativado para teste

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

            # ═══ PRÉ-CACHE: escolher os ativos só no início da sessão ═══
            obter_top_ativos_otc(bx)

            # ── Inscrever ativos no stream de velas (só 1x, ANTES do :50) ──
            _target_ativo = _cache_ativos if _cache_ativos else _top_dt_assets
            if _target_ativo:
                for _asset_live in _target_ativo:
                    _ensure_reversal_model_loaded(reversal_ai_map, _asset_live)

            if _target_ativo:
                _new_streams = 0
                for _sub_a in _target_ativo:
                    if _sub_a not in _stream_subscribed:
                        try:
                            bx.start_candles_stream(_sub_a, TF_M1, 10)
                            _stream_subscribed.add(_sub_a)
                            _new_streams += 1
                            log.info(paint(f"  📡 Stream inscrito: {_sub_a}", C.G))
                        except Exception as _sub_e:
                            log.debug(f"  ⚠️ Stream falhou para {_sub_a}: {_sub_e}")
                if _stream_subscribed and not _stream_ready.is_set():
                    _stream_ready.set()
                    log.info(paint(f"  ✅ Live streaming ativo! {len(_stream_subscribed)} ativos inscritos", C.G))
                elif _new_streams > 0:
                    log.info(paint(f"  ✅ Streams atualizados: {len(_stream_subscribed)} ativos inscritos", C.G))

            # ═══ SCAN DOUBLE TOUCH — no segundo :50 ═══
            # Scan :50 → 10s para IA → entrada no :00 (CLOSE da vela = igual ao treino)
            wait_until_second(ANALYZE_AT_SECOND)

            log.info(paint(
                f"\n🔍 Scan DT em {_target_ativo} ({len(_target_ativo) if isinstance(_target_ativo, list) else 1} ativos, segundo :{ANALYZE_AT_SECOND:02d})...",
                C.B
            ))
            all_candidates, best_any = escolher_melhor_setup_local(
                bx, cooldown, hs_stats, reversal_ai_map=reversal_ai_map
            )

            if not all_candidates:
                if best_any:
                    _, a, setup, _ = best_any
                    log.info(paint(
                        f"  ⏸️ DT em formação: {a} | aguardando confirmação",
                        C.Y
                    ))
                else:
                    _ativos_str = ", ".join(_target_ativo) if isinstance(_target_ativo, list) else _target_ativo
                    log.info(paint(f"  ⏸️ Nenhum DT em {_ativos_str}. Próximo candle.", C.Y))
                s = seconds_to_next(TF_M1)
                time.sleep(min(s + 1, 30))
                continue

            # ═══ DT ENCONTRADO → TENTAR CADA CANDIDATO ═══
            log.info(paint(f"  🎯 {len(all_candidates)} candidato(s) para avaliar", C.B))
            _candidate_traded = False

            for _cand_idx, (sc, ativo, setup, atr_val) in enumerate(all_candidates):
                setup["atr"] = atr_val  # IA usa ATR para aprender geometria
                direcao = setup["dir"]
                pat_type = setup["type"]

                # ── DEDUP: mesmo padrão ou mesma direção recente → SKIP ──
                _pat = setup.get("pattern", {})
                _head_p = _pat.get("head", {}).get("price", 0)
                _rs_p = _pat.get("right_shoulder", {}).get("price", 0)
                _entry_key = f"{ativo}_{direcao}_{_head_p:.6f}_{_rs_p:.6f}"
                _entry_key_simple = f"{ativo}_{direcao}"  # Bloqueia mesma direção no mesmo ativo
                _last_key_simple = "_".join(_last_entry_key.split("_")[:2]) if _last_entry_key else ""

                # ── TIME DEDUP: Cooldown dinâmico por sessão ──
                _session_params = _get_session_params()
                _secs_since_trade = time.time() - _last_trade_time
                _min_trade_interval = _session_params["cooldown_sec"]
                if _last_trade_time > 0 and _secs_since_trade < _min_trade_interval:
                    _wait_remain = int(_min_trade_interval - _secs_since_trade)
                    log.info(paint(
                        f"  🚫 TIME DEDUP: Último trade há {int(_secs_since_trade)}s "
                        f"(mín={_min_trade_interval}s) — aguardar {_wait_remain}s",
                        C.Y
                    ))
                    break  # TIME DEDUP bloqueia TODOS os candidatos

                if _entry_key == _last_entry_key or _entry_key_simple == _last_key_simple:
                    log.info(paint(
                        f"  🚫 DEDUP: Mesmo padrão/direção já operado ({ativo} {direcao}) — SKIP",
                        C.Y
                    ))
                    continue  # tentar próximo candidato

                # ── CONTRA SINAL: ADVISORY (no treino cada padrão é
                # avaliado independentemente, sem conceito de direção anterior).
                if _is_contra_signal(ativo, direcao):
                    log.info(paint(
                        f"  ⚠️ CONTRA SINAL ADVISORY: {ativo} {direcao} — NN decide",
                        C.Y
                    ))

                # ═══ ANÁLISE COMPLETA: IA + Geometria + Posição ═══
                pat_data = setup.get("pattern", setup)
                ia_prob = ai_predict_hs(ativo, pat_data, hs_stats)
                _arm_key = f"{ativo}_{pat_type}_{setup.get('mode', 'classic')}"
                ia_samples = hs_stats.get("arms", {}).get(_arm_key, {}).get("total", 0)

                _is_dt_mode = setup.get("mode") == "double_touch"

                # ── IA GEOMÉTRICA: apenas LOG (geometria já é feature no NN) ──
                _pq, _pq_motivos = ia_pattern_quality(pat_data, atr_val, hs_stats)
                # NÃO multiplica ia_prob — a NN já tem geometria como features f0-f25

                log.info(paint(
                    f"  🧠 IA DT: {ativo} | prob={ia_prob:.2f} | "
                    f"geom={_pq:.2f} (log only) | amostras={ia_samples} | modo={'DT' if _is_dt_mode else 'HS'}",
                    C.B
                ))

                # ═══ DECISÃO 100% NN — sem guards fixos ═══
                _all_guards_ok = True
                _cur = None
                _geo = None
                _wick_pct = 0.0
                _nn_pred = None
                _nn_score = float(setup.get("nn_pre_score") or 0.0)
                _entry_guard_pred = setup.get("entry_guard_pre_pred")
                _guard_block_reason = None
                _head_price = setup["pattern"]["head"]["price"]
                _neckline = setup["pattern"].get("neckline", 0)
                _rs_price = setup["pattern"]["right_shoulder"]["price"]
                _target_price = setup["pattern"].get("target", 0)
                _ls_price = setup["pattern"]["left_shoulder"]["price"]

                # Usar preço do SCAN (momento exato da detecção, sem delay)
                _cur = setup.get("last_close")
                _guard_df = None
                _macro_ctx = None
                _gpt_result_payload = setup.get("gpt_pre_result") if isinstance(setup.get("gpt_pre_result"), dict) else {
                    "available": False,
                    "approved": None,
                    "confidence": None,
                    "reason": "IA generativa nao executada",
                    "source": None,
                    "exp_minutes": None,
                    "latency_ms": None,
                    "stage": "live",
                }
                _gpt_approved = _gpt_result_payload.get("approved")
                _gpt_conf = _gpt_result_payload.get("confidence")
                _gpt_reason = _gpt_result_payload.get("reason")
                _gpt_source = _gpt_result_payload.get("source")
                _gpt_ms = _gpt_result_payload.get("latency_ms")
                _gpt_exp = _gpt_result_payload.get("exp_minutes")
                try:
                    _guard_df = get_candles_df(bx, ativo, TF_M1, 60, min_len=50)
                    if _guard_df is None:
                        _guard_df = get_last_closed_candles_df(bx, ativo, TF_M1, 60, min_len=50)
                    if _guard_df is not None and len(_guard_df) >= 1 and _cur is None:
                        _cur = float(_guard_df["close"].values[-1])
                    if _is_dt_mode and _guard_df is not None and len(_guard_df) >= 15:
                        _macro_ctx = _analyze_macro_trend(_guard_df, atr_val, direcao)
                except Exception as _pe:
                    log.debug(f"  get_candles_df falhou: {_pe}")

                if _cur is None:
                    log.warning(paint(f"  ⚠️ Preço atual indisponível — SKIP", C.Y))
                    _guard_block_reason = "preço atual indisponível"
                    _all_guards_ok = False

                if _is_dt_mode and _cur is not None:
                    # ═══ DT: LOGGING RICO ═══
                    _dist_to_rs = abs(_cur - _rs_price)
                    _rs_to_neck = abs(_neckline - _rs_price)
                    _progress_pct = (_dist_to_rs / _rs_to_neck * 100) if _rs_to_neck > 0 else 0
                    _target_room_atr = abs(_target_price - _cur) / max(float(atr_val or 0.0), 1e-6) if _target_price else 0.0
                    _geo = _extract_geometry(pat_data, atr_val)
                    _geo_str = ""
                    if _geo:
                        _geo_str = (f"span={_geo['span']} sym={_geo['symmetry']:.2f} "
                                   f"depth={_geo['depth_ratio']:.2f} neck={_geo['neck_align']:.3f}")

                    _rs_idx = pat_data["right_shoulder"]["idx"]
                    _wick_pct = 0
                    try:
                        _guard_n = len(_guard_df)
                        if _rs_idx < _guard_n:
                            _rs_row = _guard_df.iloc[min(_rs_idx, _guard_n - 1)]
                        else:
                            _rs_row = _guard_df.iloc[-2] if _guard_n >= 2 else _guard_df.iloc[-1]
                        _body = abs(float(_rs_row["close"]) - float(_rs_row["open"]))
                        _range_candle = float(_rs_row["high"]) - float(_rs_row["low"])
                        if _range_candle > 0:
                            _wick_pct = round((1 - _body / _range_candle) * 100, 1)
                    except Exception:
                        pass

                    _valley_price = pat_data["valley1"]["price"]
                    _force_1t = abs(_ls_price - _valley_price) / (atr_val if atr_val > 0 else 1)

                    log.info(paint(
                        f"  📍 POSIÇÃO: Preço={_cur:.6f} | RS={_rs_price:.6f} | "
                        f"Neck={_neckline:.6f} | Target={_target_price:.6f}",
                        C.G
                    ))
                    log.info(paint(
                        f"  📐 GEOMETRIA: {_geo_str} | IA geom={_pq:.2f}",
                        C.B
                    ))
                    log.info(paint(
                        f"  📊 ANÁLISE: dist_RS={_dist_to_rs:.6f} ({_progress_pct:.0f}% do caminho) | "
                        f"wick={_wick_pct:.0f}% | alvo_restante={_target_room_atr:.1f}ATR | força={_force_1t:.1f}ATR",
                        C.G if _progress_pct < 50 else C.Y
                    ))

                    print(
                        f">>> DT: {ativo} {direcao} | Preço={_cur:.6f} RS={_rs_price:.6f} "
                        f"Neck={_neckline:.6f} Target={_target_price:.6f} | "
                        f"geom={_pq:.2f} prob={ia_prob:.2f} wick={_wick_pct:.0f}%",
                        flush=True
                    )

                    # ═══ NN OBRIGATÓRIA — ÚNICA DECISÃO ═══
                    # Sem NN = sem entrada. NN 2/3 ≥ dinâmico = APROVADO.
                    _dyn_params = _get_session_params(_guard_df, atr_val)
                    _smart_exp = _dyn_params["exp_minutes"]
                    _nn_approved = False
                    log.info(paint(
                        f"  ⚙️ SESSÃO DINÂMICA: NN_min={_dyn_params['nn_min_prob']:.0%} | "
                        f"EXP={_smart_exp}min | cooldown={_dyn_params['cooldown_sec']//60}min | perfil={_dyn_params.get('profile', 'balanced')}",
                        C.B
                    ))
                    if _all_guards_ok and _guard_df is not None and len(_guard_df) >= 1:
                        _g_H = _guard_df["high"].values
                        _g_L = _guard_df["low"].values
                        _g_C = _guard_df["close"].values
                        _g_O = _guard_df["open"].values
                        _g_n = len(_g_H)

                    _nn_pred = None
                    _nn_score = None
                    _nn_source = None
                    if _all_guards_ok and _is_dt_mode and setup.get("nn_pre_pred") is not None:
                        _nn_pred = setup.get("nn_pre_pred")
                        _nn_score = float(setup.get("nn_pre_score") or _nn_pred.get("nn_score", _nn_pred.get("prob_win", 0)) or 0)
                        _nn_source = "scan"
                    elif _all_guards_ok and _is_dt_mode and _guard_df is not None and len(_guard_df) >= 50 and _is_dt_nn_model_ready(reversal_ai_map, ativo):
                        _nn_live_score, _nn_live_pred, _nn_live_reason = _estimate_dt_nn_score(
                            ativo,
                            pat_data,
                            _guard_df,
                            atr_val,
                            hs_stats,
                            reversal_ai_map,
                            return_reason=True,
                        )
                        if _nn_live_pred is not None:
                            _nn_pred = _nn_live_pred
                            _nn_score = float(_nn_live_score or _nn_live_pred.get("nn_score", _nn_live_pred.get("prob_win", 0)) or 0)
                            _nn_source = "live"
                        else:
                            setup["nn_live_reason"] = _nn_live_reason

                    if _all_guards_ok and _is_dt_mode and _nn_pred is not None:
                        _NN_MIN_PROB = _dyn_params["nn_min_prob"]
                        _nn_prob = float(_nn_pred.get("prob_win", _nn_score) or 0)
                        _nn_penalty = float(_nn_pred.get("consensus_penalty", 0) or 0)
                        _nn_p1 = float(_nn_pred.get("p1", 0) or 0)
                        _nn_p2 = float(_nn_pred.get("p2", 0) or 0)
                        _nn_p3 = _nn_pred.get("p3")
                        _nn_p3_str = f" p3={_nn_p3:.2f}" if _nn_p3 is not None else ""
                        _nn_trained = (_nn_pred.get("trained_metrics") or {}) if isinstance(_nn_pred, dict) else {}
                        _train_suffix = ""
                        if _nn_trained:
                            _train_parts = []
                            _train_samples = int(_nn_trained.get("n_samples") or 0)
                            _train_ctx = int(_nn_trained.get("context_candles") or 0)
                            if _nn_trained.get("ai1_val") is not None:
                                _train_parts.append(f"IA1={float(_nn_trained.get('ai1_val')):.0%}")
                            if _nn_trained.get("ai2_val") is not None:
                                _train_parts.append(f"IA2={float(_nn_trained.get('ai2_val')):.0%}")
                            if _nn_trained.get("ai3_val") is not None:
                                _train_parts.append(f"IA3={float(_nn_trained.get('ai3_val')):.0%}")
                            if _train_samples > 0:
                                _train_parts.append(f"n={_train_samples}")
                            if _train_ctx > 0:
                                _train_parts.append(f"ctx={_train_ctx}")
                            if _train_parts:
                                _train_suffix = " | treino=" + " ".join(_train_parts)
                        # ═══ IA DECIDE: score >= threshold → APROVADO, senão BLOQUEADO ═══
                        if _nn_score >= _NN_MIN_PROB:
                            _nn_approved = True
                            log.info(paint(
                                f"  ✅ NN APROVADO ({_nn_source}): score={_nn_score:.0%} >= {_NN_MIN_PROB:.0%} "
                                f"(prob={_nn_prob:.0%} consenso=-{_nn_penalty:.2f}) | "
                                f"p1={_nn_p1:.2f} p2={_nn_p2:.2f}{_nn_p3_str}{_train_suffix}",
                                C.G
                            ))
                        else:
                            _nn_approved = False
                            log.info(paint(
                                f"  🚫 NN BLOQUEOU ({_nn_source}): score={_nn_score:.0%} < {_NN_MIN_PROB:.0%} "
                                f"(prob={_nn_prob:.0%} consenso=-{_nn_penalty:.2f}) | "
                                f"p1={_nn_p1:.2f} p2={_nn_p2:.2f}{_nn_p3_str}{_train_suffix}",
                                C.R
                            ))

                        _timing_hint_live = _build_dt_graph_timing_hint(
                            pat_data,
                            setup.get("touch_continuation"),
                            setup.get("entry_region"),
                            setup.get("prediction_2m"),
                            _nn_pred,
                            df=_guard_df,
                            current_price=float(_cur) if _cur is not None else None,
                            atr_val=atr_val,
                        )
                        _timing_hint_live = _force_dt_entry_at_turn(_timing_hint_live, _nn_pred)
                        if _timing_hint_live.get("available"):
                            setup["timing_hint"] = _timing_hint_live
                            _timing_label = "ENTRAR AGORA" if _timing_hint_live.get("action") == "now" else f"ESPERAR {_timing_hint_live.get('wait_seconds', 0):.1f}s"
                            log.info(paint(
                                f"  ⏱️ TIMING LIVE DT: {ativo} {direcao} | {_timing_hint_live.get('zone')} | {_timing_label} | {_timing_hint_live.get('reason')}",
                                C.G if _timing_hint_live.get("action") == "now" else C.Y
                            ))

                        if _nn_approved and _is_dt_mode and _ENTRY_GUARD_ENABLED and not DT_GRAPH_SIGNAL_ENTRY:
                            if isinstance(_entry_guard_pred, dict):
                                _direction_alignment_2m = (_entry_guard_pred.get("direction_alignment_2m") or {}) if isinstance(_entry_guard_pred.get("direction_alignment_2m"), dict) else {}
                                if _entry_guard_pred.get("approved"):
                                    log.info(paint(
                                        f"  ✅ ENTRY GUARD (scan): prob={_entry_guard_pred['prob_now']:.0%} >= {_entry_guard_pred['threshold']:.0%} | "
                                        f"delay={_entry_guard_pred['delay_candles']} velas | acc={_entry_guard_pred['accuracy']:.1%} | {_direction_alignment_2m.get('reason', 'direcao 2m alinhada')}",
                                        C.G
                                    ))
                                else:
                                    log.info(paint(
                                        f"  ⚠️ DIRECAO 2M ADVISORY (scan): prob={_entry_guard_pred['prob_now']:.0%} < {_entry_guard_pred['threshold']:.0%} | "
                                        f"delay={_entry_guard_pred['delay_candles']} velas | acc={_entry_guard_pred['accuracy']:.1%} | {_direction_alignment_2m.get('reason', 'entry_guard rejeitou')} | Bayes decide",
                                        C.Y
                                    ))
                            else:
                                log.info(paint(
                                    f"  ⚠️ ENTRY GUARD (scan): pré-score indisponível para {ativo} — mantendo decisão NN",
                                    C.Y
                                ))
                        elif _nn_approved and _is_dt_mode and DT_GRAPH_SIGNAL_ENTRY:
                            log.info(paint(
                                "  ✅ MODO GRAFICO DT: sinal do scan preservado; após o sinal somente a NN valida",
                                C.G
                            ))

                    elif _all_guards_ok and _is_dt_mode:
                        _nn_reason_parts = []
                        if setup.get("nn_pre_reason"):
                            _nn_reason_parts.append(f"scan={setup.get('nn_pre_reason')}")
                        if setup.get("nn_live_reason"):
                            _nn_reason_parts.append(f"live={setup.get('nn_live_reason')}")
                        _nn_reason_suffix = f" ({'; '.join(_nn_reason_parts)})" if _nn_reason_parts else ""
                        if DT_GRAPH_SIGNAL_ENTRY:
                            _nn_approved = True  # Bayes decide
                            log.info(paint(
                                f"  ⚠️ NN: indisponível para {ativo} no modo grafico{_nn_reason_suffix} | Bayes decide",
                                C.Y
                            ))
                        else:
                            _nn_approved = True
                            log.info(paint(
                                f"  ⚠️ NN: indisponível para {ativo}{_nn_reason_suffix} | Bayes decide",
                                C.Y
                            ))
                    elif _all_guards_ok:
                        _nn_approved = True  # Bayes decide
                        log.info(paint(
                            f"  ⚠️ NN: Dados insuficientes (<50 candles) | Bayes decide",
                            C.Y
                        ))

                    # ═══ IA 4 — FILTRO DE CONTEXTO (tabela backtest) ═══
                    # Consulta WR histórico do backtest por geometria/ativo/hora
                    if _nn_approved and _guard_df is not None and ENABLE_CONTEXT_FILTER:
                        try:
                            _ctx_live_geo = _extract_geometry(pat_data, atr_val)
                            import datetime as _dt_mod
                            _ctx_live_hour = _dt_mod.datetime.utcnow().hour
                            _ctx_live = context_lookup(
                                ativo=ativo, direcao=direcao,
                                hour=_ctx_live_hour,
                                depth_ratio=_ctx_live_geo["depth_ratio"] if _ctx_live_geo else 0,
                                symmetry=_ctx_live_geo["symmetry"] if _ctx_live_geo else 0,
                            )
                            _ctx_live_log = format_context_log(_ctx_live)
                            log.info(paint(f"  {_ctx_live_log}", C.B))

                            if _ctx_live["action"] == "block":
                                _guard_block_reason = _ctx_live["reason"]
                                _all_guards_ok = False
                                log.info(paint(
                                    f"  ⛔ Context BLOQUEOU: {_ctx_live['reason']}",
                                    C.R
                                ))

                            _gpt_result_payload = {
                                "available": True,
                                "approved": _ctx_live["action"] != "block",
                                "confidence": _ctx_live["wr"],
                                "reason": _ctx_live["reason"],
                                "source": f"ctx_L{_ctx_live['level']}",
                                "exp_minutes": None,
                                "latency_ms": 0,
                                "stage": "live",
                            }
                            _gpt_approved = _gpt_result_payload["approved"]
                            _gpt_conf = _ctx_live["wr"]
                            _gpt_reason = _ctx_live["reason"]
                            _gpt_source = f"ctx_L{_ctx_live['level']}"
                            _gpt_ms = 0
                        except Exception as _ctx_err:
                            _gpt_result_payload = {
                                "available": False,
                                "approved": None,
                                "confidence": None,
                                "reason": f"erro context live: {_ctx_err}",
                                "source": None,
                                "exp_minutes": None,
                                "latency_ms": None,
                                "stage": "live",
                            }
                            _gpt_approved = None
                            _gpt_conf = None
                            _gpt_reason = _gpt_result_payload["reason"]
                            _gpt_source = None
                            _gpt_ms = None
                            log.warning(paint(
                                f"  ⚠️ Context filter erro: {_ctx_err} — mantendo decisão NN",
                                C.Y
                            ))
                    elif _nn_approved and _guard_df is not None and _is_dt_mode:
                        log.info(paint(
                            "  ⚡ Context filter desativado no DT",
                            C.B
                        ))

                    _ai_consensus_live = _build_ai_cotrader_consensus(
                        setup.get("mode", "classic"),
                        ia_prob,
                        _nn_pred if _nn_pred is not None else setup.get("nn_pre_pred"),
                        _gpt_result_payload,
                        setup.get("shadow_pattern_lib"),
                    )
                    setup["gpt_pre_result"] = _gpt_result_payload
                    setup["ai_consensus"] = _ai_consensus_live
                    if not _ai_consensus_live.get("final_ok"):
                        _guard_block_reason = _ai_consensus_live.get("reason") or "consenso IA bloqueou"
                        log.info(paint(
                            f"  ⛔ CONSENSO IA: {_guard_block_reason}",
                            C.Y
                        ))
                        _all_guards_ok = False
                    else:
                        log.info(paint(
                            f"  ✅ CONSENSO IA: {_ai_consensus_live.get('reason')}",
                            C.G if _ai_consensus_live.get("gpt_ok") is not False else C.Y
                        ))

                    _counter_barrier = _detect_dt_counter_barrier(
                        pat_data,
                        _guard_df,
                        atr_val,
                        current_price=float(_cur) if _cur is not None else None,
                    )
                    setup["live_metrics"] = {
                        "geometry": _geo if _geo else _extract_geometry(pat_data, atr_val),
                        "wick_pct": float(_wick_pct or 0),
                        "nn_score": float(_nn_score or 0),
                        "entry_guard_prob": float(_entry_guard_pred.get("prob_now", 0)) if isinstance(_entry_guard_pred, dict) else float(setup.get("entry_guard_pre_score") or 0),
                        "progress_pct": round(float(_progress_pct), 2),
                        "target_room_atr": round(float(_target_room_atr), 3),
                        "counter_barrier": _counter_barrier,
                    }

                    if _all_guards_ok and _nn_approved and _is_dt_mode and DT_GRAPH_SIGNAL_ENTRY:
                        _graph_entry_region = _validate_dt_entry_region(pat_data, float(_cur), atr_val)
                        _graph_timing_live = _build_dt_graph_timing_hint(
                            pat_data,
                            setup.get("touch_continuation"),
                            _graph_entry_region,
                            setup.get("prediction_2m"),
                            _nn_pred,
                            df=_guard_df,
                            current_price=float(_cur) if _cur is not None else None,
                            atr_val=atr_val,
                        )
                        _graph_timing_live = _force_dt_entry_at_turn(_graph_timing_live, _nn_pred)
                        setup["graph_wait"] = {
                            "initial": {
                                "wait": False,
                                "seconds": 0.0,
                                "reason": "modo grafico DT: sem graph wait" if DT_GRAPH_NN_ONLY_TEST else "sem indicio de falso movimento",
                            },
                            "entry_region_live": _graph_entry_region,
                        }
                        setup["entry_region"] = _graph_entry_region
                        setup["timing_hint"] = _graph_timing_live

                        if not _graph_entry_region.get("ok"):
                            log.info(paint(
                                f"  ⚠️ DT GRAFICO — REGIÃO ADVISORY: {_graph_entry_region.get('reason', 'regiao invalida')} — NN decide",
                                C.Y
                            ))
                            # Advisory — não bloqueia (treino não filtra por região)
                        else:
                            log.info(paint(
                                f"  ✅ DT GRAFICO: Bayes decide — sem bloqueios de timing/regiao/falso movimento",
                                C.G
                            ))

                    if _all_guards_ok and _is_dt_mode and _cur is not None and not DT_GRAPH_SIGNAL_ENTRY:
                        _entry_region_live = _validate_dt_entry_region(pat_data, float(_cur), atr_val)
                        setup["entry_region"] = _entry_region_live
                        if not _entry_region_live.get("ok"):
                            log.info(paint(
                                f"  ⚠️ REGIÃO ADVISORY (LIVE): {ativo} {direcao} | {_entry_region_live.get('reason')} — NN decide",
                                C.Y
                            ))
                            # Advisory — não bloqueia (treino não filtra por região)

                    if _all_guards_ok and _nn_approved and _is_dt_mode and not DT_GRAPH_SIGNAL_ENTRY:
                        _win_geometry_alignment = setup.get("win_geometry_alignment")
                        if not isinstance(_win_geometry_alignment, dict):
                            _win_geometry_alignment = _validate_dt_win_geometry_alignment(
                                ativo,
                                pat_data,
                                atr_val,
                                hs_stats,
                                _entry_guard_pred,
                            )
                            setup["win_geometry_alignment"] = _win_geometry_alignment
                        if not _win_geometry_alignment.get("ok"):
                            log.info(paint(
                                f"  ⚠️ GEOMETRIA DOS WINS ADVISORY: {ativo} {direcao} | {_win_geometry_alignment.get('reason')} | Bayes decide",
                                C.Y
                            ))

                    if _all_guards_ok and _nn_approved and (not _is_dt_mode or not DT_GRAPH_SIGNAL_ENTRY) and _progress_pct >= DT_LATE_PROGRESS_PCT and _target_room_atr < DT_LATE_MIN_TARGET_ATR:
                        log.info(paint(
                            f"  ⚠️ DT TARDIO ADVISORY: {ativo} {direcao} | {float(_progress_pct):.0f}% do caminho, {_target_room_atr:.1f}ATR até alvo | Bayes decide",
                            C.Y
                        ))

                    # Elite guard desativado — Bayes decide
                    if _all_guards_ok and _nn_approved and _is_dt_mode:
                        setup["elite_guard"] = {
                            "ok": True,
                            "reason": "Bayes decide — sem guards rigidos",
                            "profile": DT_LIVE_PROFILE,
                            "score": None,
                        }
                        log.info(paint(
                            f"  ✅ BAYES DECIDE: sem guards rigidos para {ativo} {direcao}",
                            C.G
                        ))

                elif not _is_dt_mode and _cur is not None:
                    # ═══ H&S CLÁSSICO: Guards básicos ═══
                    if direcao == "PUT" and _cur >= _head_price:
                        log.info(paint(f"  🚫 GUARD HEAD: Preço ({_cur:.6f}) >= Cabeça ({_head_price:.6f})", C.Y))
                        _guard_block_reason = "preço acima da cabeça"
                        _all_guards_ok = False
                    elif direcao == "CALL" and _cur <= _head_price:
                        log.info(paint(f"  🚫 GUARD HEAD: Preço ({_cur:.6f}) <= Cabeça ({_head_price:.6f})", C.Y))
                        _guard_block_reason = "preço abaixo da cabeça"
                        _all_guards_ok = False

                    if _all_guards_ok:
                        if direcao == "PUT" and _cur > _rs_price:
                            log.info(paint(f"  🚫 BREAK GUARD: Preço ({_cur:.6f}) > Ombro D ({_rs_price:.6f})", C.Y))
                            _guard_block_reason = "preço acima do ombro direito"
                            _all_guards_ok = False
                        elif direcao == "CALL" and _cur < _rs_price:
                            log.info(paint(f"  🚫 BREAK GUARD: Preço ({_cur:.6f}) < Ombro D ({_rs_price:.6f})", C.Y))
                            _guard_block_reason = "preço abaixo do ombro direito"
                            _all_guards_ok = False

                    # IA filter somente para H&S
                    if _all_guards_ok and ia_prob < AI_MIN_PROB and ia_prob != 0.5:
                        log.info(paint(f"  🚫 IA BLOQUEOU: prob={ia_prob:.2f} < {AI_MIN_PROB}", C.Y))
                        _guard_block_reason = f"ia_prob={ia_prob:.2f} < {AI_MIN_PROB:.2f}"
                        _all_guards_ok = False

                    if _all_guards_ok:
                        log.info(paint(f"  ✅ GUARDS OK: Preço={_cur:.6f} | Head={_head_price:.6f} | RS={_rs_price:.6f}", C.G))

                if not _all_guards_ok:
                    if _guard_block_reason:
                        print(f">>> IA: GUARD bloqueou {ativo} {direcao} — {_guard_block_reason}", flush=True)
                    else:
                        print(f">>> IA: GUARD bloqueou {ativo} {direcao}", flush=True)
                    continue  # tentar próximo candidato

                # Calcular stake baseado no saldo (% da banca)
                stake = calcular_stake(bx)
                log.info(paint(
                    f"  💰 STAKE: ${stake:.2f} ({PERCENT_BANCA:.1f}% da banca)",
                    C.G
                ))

                # ═══ ENTRADA DT: timing guard controla a virada :00 ═══
                _is_early = setup.get("mode") == "early"
                _is_dt = setup.get("mode") == "double_touch"
                _candles_ago = pat_data.get("candles_ago", 99)
                _graph_scan_ts = float(pat_data.get("scan_ts", 0.0) or 0.0)
                _graph_signal_age_sec = max(0.0, time.time() - _graph_scan_ts) if _graph_scan_ts > 0 else None
                _mode_label = "EARLY" if _is_early else ("DOUBLE_TOUCH" if _is_dt else "CLASSIC")
                if _is_dt and DT_GRAPH_SIGNAL_ENTRY:
                    _latency_suffix = (
                        f" | latencia_scan={_graph_signal_age_sec:.2f}s"
                        if _graph_signal_age_sec is not None else ""
                    )
                    # ═══ FIX TIMING: SEMPRE entrar no :00 (abertura da vela).
                    # O treino usa CLOSE da vela → entrada no OPEN da próxima.
                    # Entrar no meio do minuto (:33) desfasa o timing real.
                    log.info(paint(
                        f"  ⏱️ {_mode_label} MODE: candles_ago={_candles_ago} → aguardando virada :00{_latency_suffix}",
                        C.Y
                    ))
                    wait_candle_open()
                else:
                    log.info(paint(
                        f"  ⏱️ {_mode_label} MODE: Aguardando virada :00 para entrada "
                        f"(candles_ago={_candles_ago})",
                        C.G
                    ))
                    wait_candle_open()

                # ═══ SALVAR NN DO SCAN (antes do RE-CHECK poder sobrescrever) ═══
                _scan_nn_score = _nn_score  # Guardar score original do scan

                # ═══ FIX IA: Re-extrair features com vela FECHADA e re-rodar NN ═══
                # Treino usa velas fechadas. Se candles_ago=0, a predição anterior usou
                # dados de vela incompleta. Agora que wait_candle_open() garantiu :00,
                # a vela do RS está fechada → re-extrair features e re-predizer.
                if _is_dt_mode and _nn_pred is not None and _candles_ago == 0:
                    try:
                        _fresh_df = get_candles_df(bx, ativo, TF_M1, 60, min_len=50)
                        if _fresh_df is None:
                            _fresh_df = get_last_closed_candles_df(bx, ativo, TF_M1, 60, min_len=50)
                        if _fresh_df is not None and len(_fresh_df) >= 50 and _is_dt_nn_model_ready(reversal_ai_map, ativo):
                            _nn2_score, _nn2_pred, _nn2_reason = _estimate_dt_nn_score(
                                ativo, pat_data, _fresh_df, atr_val, hs_stats,
                                reversal_ai_map, return_reason=True,
                            )
                            if _nn2_pred is not None:
                                _nn2_s = float(_nn2_score or _nn2_pred.get("nn_score", _nn2_pred.get("prob_win", 0)) or 0)
                                _nn2_p1 = float(_nn2_pred.get("p1", 0) or 0)
                                _nn2_p2 = float(_nn2_pred.get("p2", 0) or 0)
                                _nn2_p3 = _nn2_pred.get("p3")
                                _nn2_p3s = f" p3={_nn2_p3:.2f}" if _nn2_p3 is not None else ""
                                _nn2_pen = float(_nn2_pred.get("consensus_penalty", 0) or 0)
                                log.info(paint(
                                    f"  🔄 NN RE-CHECK (vela fechada): score={_nn2_s:.0%} "
                                    f"(anterior={_nn_score:.0%}) | p1={_nn2_p1:.2f} p2={_nn2_p2:.2f}{_nn2_p3s} consenso=-{_nn2_pen:.2f}",
                                    C.G if _nn2_s >= _NN_MIN_PROB else C.R
                                ))
                                # Atualizar predição com dados da vela fechada
                                _nn_pred = _nn2_pred
                                _nn_score = _nn2_s
                                _nn_prob = float(_nn2_pred.get("prob_win", _nn2_s) or 0)
                                _nn_penalty = _nn2_pen
                                _nn_p1 = _nn2_p1
                                _nn_p2 = _nn2_p2
                                _nn_p3 = _nn2_p3
                                _nn_source = "recheck"
                                # Re-avaliar aprovação com novo score
                                if _nn2_s >= _NN_MIN_PROB:
                                    _nn_approved = True
                                    log.info(paint(
                                        f"  ✅ NN RE-CHECK APROVADO: {_nn2_s:.0%} >= {_NN_MIN_PROB:.0%}",
                                        C.G
                                    ))
                                else:
                                    _nn_approved = False
                                    log.info(paint(
                                        f"  🚫 NN RE-CHECK BLOQUEOU: {_nn2_s:.0%} < {_NN_MIN_PROB:.0%}",
                                        C.R
                                    ))
                                # Atualizar preço atual com dado fresco
                                _cur = float(_fresh_df["close"].values[-1])
                            else:
                                log.info(paint(
                                    f"  ⚠️ NN RE-CHECK: falhou ({_nn2_reason}) — mantendo predição anterior",
                                    C.Y
                                ))
                    except Exception as _recheck_ex:
                        log.debug(f"  NN RE-CHECK erro: {_recheck_ex}")

                # ═══ INVERSÃO INTELIGENTE substitui adaptação de sessão ═══
                # (a lógica de inversão por faixa de NN decide tudo)
                _session_threshold = _NN_MIN_PROB

                # Re-avaliar aprovação com threshold adaptado
                if _nn_pred is not None and _nn_score < _session_threshold:
                    _nn_approved = False

                # ═══ BLOQUEIO FINAL: se NN não aprovou, cancelar entrada ═══
                if _is_dt_mode and not _nn_approved and _nn_pred is not None:
                    log.info(paint(
                        f"  🚫 IA BLOQUEOU ENTRADA: {ativo} {direcao} | NN score={_nn_score:.0%} < {_session_threshold:.0%}",
                        C.R
                    ))
                    print(f">>> IA: NN bloqueou {ativo} {direcao} — score={_nn_score:.0%} (min={_session_threshold:.0%})", flush=True)
                    continue  # próximo candidato

                if _is_dt and DT_GRAPH_SIGNAL_ENTRY and isinstance(setup.get("timing_hint"), dict) and setup["timing_hint"].get("available"):
                    _timing_hint = setup["timing_hint"]
                    log.info(paint(
                        f"  ⚡ TIMING DT: Bayes decide — entrada imediata | {_timing_hint.get('reason')}",
                        C.G
                    ))

                _live_entry_price = float(_cur) if _cur else float(pat_data.get("entry_price", 0))

                # ═══ VERIFICAÇÃO DE FORÇA DO BOUNCE (Smart Entry) ═══
                # Verifica se o preço está REALMENTE bounceando na zona antes de entrar.
                # Se o preço não está se afastando da zona, é sinal fraco.
                if _is_dt_mode and _live_entry_price and atr_val > 0:
                    _rs_price_check = float(pat_data.get("right_shoulder", {}).get("price", 0) or 0)
                    _neck_price = float(pat_data.get("neckline", 0) or 0)
                    _bounce_ok = True
                    _bounce_reason = ""

                    if _rs_price_check > 0 and _neck_price > 0:
                        _dist_to_zone = abs(_live_entry_price - _rs_price_check)
                        _dist_to_target = abs(_neck_price - _rs_price_check)
                        _progress = _dist_to_zone / _dist_to_target if _dist_to_target > 0 else 0

                        # Para CALL: preço deve estar ACIMA do RS (bounceando pra cima)
                        # Para PUT: preço deve estar ABAIXO do RS (bounceando pra baixo)
                        if direcao == "CALL":
                            _price_above_zone = _live_entry_price > _rs_price_check
                            _moving_toward_target = _live_entry_price < _neck_price
                        else:  # PUT
                            _price_above_zone = _live_entry_price < _rs_price_check
                            _moving_toward_target = _live_entry_price > _neck_price

                        # BLOQUEAR: preço está do lado ERRADO da zona (rompeu suporte/resistência)
                        if not _price_above_zone and _dist_to_zone > atr_val * 0.15:
                            _bounce_ok = False
                            _bounce_reason = f"preço rompeu a zona ({_dist_to_zone / atr_val:.2f}ATR do lado errado)"

                        # BLOQUEAR: preço já andou demais em direção ao target (entrada tardia)
                        if _progress > 0.60:
                            _bounce_ok = False
                            _bounce_reason = f"entrada tardia — preço já percorreu {_progress:.0%} do caminho ao target"

                    # Verificar CANDLE actual: última vela deve confirmar a direção
                    if _bounce_ok and _guard_df is not None and len(_guard_df) >= 3:
                        _last_c = float(_guard_df["close"].values[-1])
                        _last_o = float(_guard_df["open"].values[-1])
                        _last_h = float(_guard_df["high"].values[-1])
                        _last_l = float(_guard_df["low"].values[-1])
                        _last_range = _last_h - _last_l

                        if _last_range > 0:
                            if direcao == "CALL":
                                # Vela deve mostrar rejeição no fundo: wick inferior forte
                                _lower_wick = min(_last_o, _last_c) - _last_l
                                _wick_pct_check = _lower_wick / _last_range
                                _body_bullish = _last_c >= _last_o  # verde ou doji
                                # Bloquear se vela é fortemente bearish (vermelha corpuda) = bounce fraco
                                _body_bearish_pct = max(0, _last_o - _last_c) / _last_range
                                if _body_bearish_pct > 0.65:
                                    _bounce_ok = False
                                    _bounce_reason = f"última vela fortemente bearish ({_body_bearish_pct:.0%} corpo vermelho)"
                            else:  # PUT
                                # Vela deve mostrar rejeição no topo: wick superior forte
                                _upper_wick = _last_h - max(_last_o, _last_c)
                                _wick_pct_check = _upper_wick / _last_range
                                _body_bullish_pct = max(0, _last_c - _last_o) / _last_range
                                if _body_bullish_pct > 0.65:
                                    _bounce_ok = False
                                    _bounce_reason = f"última vela fortemente bullish ({_body_bullish_pct:.0%} corpo verde)"

                    if not _bounce_ok:
                        log.info(paint(
                            f"  🚫 BOUNCE FRACO BLOQUEOU: {ativo} {direcao} | {_bounce_reason} — CANCELADO",
                            C.R
                        ))
                        print(f">>> IA: BOUNCE FRACO bloqueou {ativo} {direcao} — {_bounce_reason}", flush=True)
                        _all_guards_ok = False
                    else:
                        log.info(paint(
                            f"  ✅ BOUNCE CONFIRMADO: {ativo} {direcao} | preço bounceando corretamente",
                            C.G
                        ))

                # Verificar se preço já ultrapassou neckline
                if not _GUARDS_DISABLED and _neckline > 0 and _live_entry_price and not _is_dt_mode:
                    if direcao == "CALL" and _live_entry_price >= _neckline:
                        log.info(paint(
                            f"  🚫 FINAL CHECK: Preço ({_live_entry_price:.6f}) já acima da Neckline ({_neckline:.6f}) → CANCELADO",
                            C.Y
                        ))
                        _all_guards_ok = False
                    elif direcao == "PUT" and _live_entry_price <= _neckline:
                        log.info(paint(
                            f"  🚫 FINAL CHECK: Preço ({_live_entry_price:.6f}) já abaixo da Neckline ({_neckline:.6f}) → CANCELADO",
                            C.Y
                        ))
                        _all_guards_ok = False

                if not _all_guards_ok:
                    print(f">>> IA: FINAL CHECK cancelou {ativo} {direcao}", flush=True)
                    continue  # tentar próximo candidato

                _nn_entry_data = None
                if _nn_pred is not None:
                    _nn_entry_data = {
                        "approved": True,
                        "available": True,
                        "source": _nn_source,
                        "p1": round(_nn_p1, 3),
                        "p2": round(_nn_p2, 3),
                        "p3": round(_nn_p3, 3) if _nn_p3 is not None else None,
                        "nn_score": round(_nn_score, 3),
                        "consensus_penalty": round(_nn_penalty, 3),
                        "trained_metrics": _nn_pred.get("trained_metrics"),
                    }
                _decision_id = f"{int(time.time() * 1000)}_{ativo}_{direcao}_{random.randint(1000, 9999)}"
                _decision_conf_pct = (_nn_score * 100) if _nn_pred is not None else (ia_prob * 100)
                _nn_reason_parts = []
                if setup.get("nn_pre_reason"):
                    _nn_reason_parts.append(f"scan={setup.get('nn_pre_reason')}")
                if setup.get("nn_live_reason"):
                    _nn_reason_parts.append(f"live={setup.get('nn_live_reason')}")
                _nn_reason_text = "; ".join(_nn_reason_parts) if _nn_reason_parts else None
                if _nn_entry_data is None:
                    _nn_entry_data = {
                        "approved": _nn_approved,
                        "available": False,
                        "source": _nn_source,
                        "reason": _nn_reason_text,
                    }
                _decision_geo = _extract_geometry(pat_data, atr_val) if pat_data else None
                _decision_payload = {
                    "decision_id": _decision_id,
                    "ts": time.time(),
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ativo": ativo,
                    "direcao": direcao,
                    "broker": _broker_suffix,
                    "pat_type": pat_type,
                    "mode": setup.get("mode", "classic"),
                    "entry_price": round(_live_entry_price, 6) if _live_entry_price else None,
                    "stake": round(stake, 2),
                    "exp_min": EXP_EARLY if _is_early else _smart_exp,
                    "resultado": None,
                    "status": "entry",
                    # ── Geometria ──
                    "geometry": {
                        "span": _decision_geo.get("span", 0) if _decision_geo else 0,
                        "symmetry": _decision_geo.get("symmetry", 0) if _decision_geo else 0,
                        "depth_ratio": _decision_geo.get("depth_ratio", 0) if _decision_geo else 0,
                        "neck_align": _decision_geo.get("neck_align", 0) if _decision_geo else 0,
                        "shoulder_ratio": _decision_geo.get("shoulder_ratio", 0) if _decision_geo else 0,
                    },
                    "geom_score": round(_pq, 4),
                    # ── Padrão ──
                    "pattern": {
                        "left_shoulder": round(_ls_price, 6),
                        "head": round(_head_price, 6),
                        "right_shoulder": round(_rs_price, 6),
                        "neckline": round(_neckline, 6),
                        "target": round(_target_price, 6),
                        "depth": round(pat_data.get("depth", 0), 6),
                    },
                    # ── IA Probabilística ──
                    "ia_prob": round(ia_prob, 4),
                    "ia_samples": ia_samples,
                    # ── NN (3 modelos) ──
                    "nn": {
                        "available": _nn_pred is not None,
                        "source": _nn_source,
                        "reason": _nn_reason_text,
                        "state_text": "Confluencia validada" if _nn_pred is not None and _nn_approved else ("Confluencia recusada" if _nn_pred is not None and _nn_approved is False else "NN indisponível"),
                        "p1": round(_nn_p1, 4) if _nn_pred else None,
                        "p2": round(_nn_p2, 4) if _nn_pred else None,
                        "p3": round(_nn_p3, 4) if _nn_pred and _nn_p3 is not None else None,
                        "prob_win": round(_nn_prob, 4) if _nn_pred else None,
                        "nn_score": round(_nn_score, 4) if _nn_pred else None,
                        "consensus_penalty": round(_nn_penalty, 4) if _nn_pred else None,
                        "approved": _nn_approved,
                        "trained_metrics": _nn_pred.get("trained_metrics") if _nn_pred else None,
                    },
                    "entry_guard": {
                        "approved": _entry_guard_pred.get("approved") if _entry_guard_pred else None,
                        "prob_now": round(_entry_guard_pred.get("prob_now", 0), 4) if _entry_guard_pred else None,
                        "threshold": round(_entry_guard_pred.get("threshold", 0), 4) if _entry_guard_pred else None,
                        "recommended_threshold": round(_entry_guard_pred.get("recommended_threshold", 0), 4) if _entry_guard_pred else None,
                        "accuracy": round(_entry_guard_pred.get("accuracy", 0), 4) if _entry_guard_pred else None,
                        "auc": round(_entry_guard_pred.get("auc", 0), 4) if _entry_guard_pred else None,
                        "precision": round(_entry_guard_pred.get("precision", 0), 4) if _entry_guard_pred else None,
                        "recall": round(_entry_guard_pred.get("recall", 0), 4) if _entry_guard_pred else None,
                        "delay_candles": _entry_guard_pred.get("delay_candles") if _entry_guard_pred else None,
                        "direction_alignment_2m": _entry_guard_pred.get("direction_alignment_2m") if _entry_guard_pred else None,
                        "touch_continuation": setup.get("touch_continuation"),
                        "prediction_2m": setup.get("prediction_2m"),
                        "timing_hint": setup.get("timing_hint"),
                        "entry_region": setup.get("entry_region"),
                        "graph_wait": setup.get("graph_wait"),
                        "win_signature": _entry_guard_pred.get("win_signature") if _entry_guard_pred else None,
                        "entry_trigger": setup.get("entry_trigger"),
                        "quality_risk": setup.get("quality_risk"),
                        "win_geometry_alignment": setup.get("win_geometry_alignment"),
                        "training_alignment": setup.get("training_alignment"),
                    },
                    "shadow_pattern_lib": setup.get("shadow_pattern_lib"),
                    "ai_consensus": setup.get("ai_consensus"),
                    # ── Context Filter ──
                    "context_filter": {
                        "available": bool(_gpt_result_payload.get("available")),
                        "approved": _gpt_approved,
                        "confidence": _gpt_conf,
                        "reason": _gpt_reason,
                        "source": _gpt_source,
                        "latency_ms": _gpt_ms,
                        "exp_minutes": _gpt_exp,
                        "stage": _gpt_result_payload.get("stage"),
                    },
                    # ── Contexto ──
                    "cur_price": round(float(_cur), 6) if _cur else None,
                    "atr": round(atr_val, 6),
                    "wick_pct": locals().get('_wick_pct', 0),
                    "elite_guard": setup.get("elite_guard"),
                    "progress_pct": round(float(setup.get("live_metrics", {}).get("progress_pct", 0)), 2) if setup.get("live_metrics") else None,
                    "target_room_atr": round(float(setup.get("live_metrics", {}).get("target_room_atr", 0)), 3) if setup.get("live_metrics") else None,
                }

                # ═══ PROTEÇÃO ANTI-STREAK: após 2+ losses consecutivos, exigir NN mais alto ═══
                if _consecutive_losses >= 2 and _nn_score is not None:
                    _streak_nn_min = 0.90  # exigir 90% após streak
                    if _nn_score < _streak_nn_min:
                        log.info(paint(
                            f"  🚫 STREAK GUARD: {ativo} {direcao} | {_consecutive_losses} losses consecutivos "
                            f"→ NN={_nn_score*100:.0f}% < {_streak_nn_min*100:.0f}% mínimo anti-streak — CANCELADO",
                            C.R
                        ))
                        print(f">>> IA: STREAK GUARD bloqueou {ativo} {direcao} — {_consecutive_losses} losses, NN={_nn_score*100:.0f}%", flush=True)
                        continue
                    else:
                        log.info(paint(
                            f"  ⚠️ STREAK MODE: {ativo} {direcao} | {_consecutive_losses} losses consecutivos "
                            f"— NN={_nn_score*100:.0f}% ≥ {_streak_nn_min*100:.0f}% → permitido com cautela",
                            C.Y
                        ))

                # ═══ SHADOW DIVERGENCE GUARD: se biblioteca diverge E NN < 85%, bloquear ═══
                _shadow_lib = setup.get("shadow_pattern_lib") or {}
                if _shadow_lib.get("available") and not _shadow_lib.get("agreement", True):
                    _shadow_nn_min = 0.85
                    if _nn_score is not None and _nn_score < _shadow_nn_min:
                        log.info(paint(
                            f"  🚫 SHADOW DIVERGE BLOQUEOU: {ativo} {direcao} | "
                            f"biblioteca diverge + NN={_nn_score*100:.0f}% < {_shadow_nn_min*100:.0f}% — CANCELADO",
                            C.R
                        ))
                        print(f">>> IA: SHADOW DIVERGE bloqueou {ativo} {direcao} — NN={_nn_score*100:.0f}%", flush=True)
                        continue

                _use_exp = EXP_EARLY if _is_early else _smart_exp
                # ═══ SMART EXP: 1 min se velocidade + NN permitem ═══
                if not _is_early and _is_dt_mode and _nn_score is not None and _guard_df is not None:
                    _smart_computed = _compute_smart_exp(
                        _g_C, _g_H, _g_L, _g_n, atr_val, _nn_score, pat_data
                    )
                    if _smart_computed != _use_exp:
                        log.info(paint(
                            f"  ⚡ SMART EXP: {_use_exp}min → {_smart_computed}min "
                            f"(NN={_nn_score*100:.0f}% velocidade ok)",
                            C.G if _smart_computed == 1 else C.B
                        ))
                        _use_exp = _smart_computed
                # ═══ STREAK MODE: forçar 2min após 2+ losses consecutivos ═══
                if _consecutive_losses >= 2 and _use_exp == 1:
                    log.info(paint(
                        f"  ⚠️ STREAK → EXP: forçando 2min (era 1min) — {_consecutive_losses} losses consecutivos",
                        C.Y
                    ))
                    _use_exp = EXP_FIXA
                _send_delay_sec = time.time() % 60
                if _is_dt and DT_GRAPH_SIGNAL_ENTRY:
                    if _graph_signal_age_sec is not None:
                        _timing_note = "virada :00 respeitada" if DT_ENTRY_AT_TURN else "guard de virada ignorado"
                        log.info(paint(
                            f"  ⏱️ DT GRAFICO: ordem enviada {(_graph_signal_age_sec):.2f}s após o inicio do scan ({_timing_note})",
                            C.G
                        ))
                elif _is_dt_mode and _send_delay_sec > MAX_ENTRY_DELAY_SEC:
                    log.info(paint(
                        f"  🚫 DT ATRASADO: ordem seria enviada {_send_delay_sec:.2f}s após a virada > {MAX_ENTRY_DELAY_SEC:.2f}s",
                        C.R
                    ))
                    print(
                        f">>> IA: FINAL CHECK cancelou {ativo} {direcao} — ordem atrasada {_send_delay_sec:.2f}s",
                        flush=True
                    )
                    continue

                # ═══ DECISÃO DE DIREÇÃO BASEADA NO MAPA OFFLINE ═══
                # Análise de milhares de padrões históricos por ativo mostrou:
                #   NN >= 80%: WR original 92-100% → SEMPRE ENTRAR ORIGINAL
                #   NN < 80%:  WR varia por ativo → consultar mapa per-asset
                # NUNCA inverter: dados comprovam que direção original é melhor.
                _direcao_original = direcao
                if _nn_score is not None:
                    _nn_zone_score = max(_scan_nn_score, _nn_score)
                    _nn_zone_pct = _nn_zone_score * 100
                    _nn_pct = _nn_score * 100

                    if _nn_zone_pct >= 80:
                        # NN alto (≥80%): WR original 92-100% comprovado → ENTRAR
                        log.info(paint(
                            f"  ✅ NN CONFIANTE: {ativo} {direcao} | NN={_nn_pct:.0f}% (zona={_nn_zone_pct:.0f}% ≥80% → entrar original)",
                            C.G
                        ))
                    else:
                        # NN baixo (<80%): consultar perfil do ativo
                        _inv_profile = _inversion_map.get(ativo, {})
                        _low_action = _inv_profile.get("low_nn_action", "skip")
                        _low_wr = _inv_profile.get("low_nn", {}).get("original_wr", 0.5)
                        _low_n = _inv_profile.get("low_nn", {}).get("samples", 0)

                        if _low_action == "keep":
                            # Ativo funciona bem mesmo com NN baixo
                            log.info(paint(
                                f"  ✅ NN BAIXO OK: {ativo} {direcao} | NN={_nn_pct:.0f}% | WR_orig={_low_wr:.0%} n={_low_n} → entrar",
                                C.G
                            ))
                        else:
                            # NN baixo + ativo não confiável nessa faixa
                            log.info(paint(
                                f"  🚫 NN BAIXO SKIP: {ativo} {direcao} | NN={_nn_pct:.0f}% | WR_orig={_low_wr:.0%} n={_low_n} → pular",
                                C.R
                            ))
                            print(f">>> IA: Skip {ativo} {direcao} — NN={_nn_pct:.0f}% baixo (WR={_low_wr:.0%})", flush=True)
                            continue

                # ═══ ATUALIZAR DASHBOARD COM DIREÇÃO REAL ═══
                _was_inverted = (direcao != _direcao_original)
                _decision_payload["direcao"] = direcao
                _decision_payload["direcao_original"] = _direcao_original
                _decision_payload["invertido"] = _was_inverted
                # Atualizar live_signals no cache do dashboard
                try:
                    _dash_file = _DASHBOARD_CACHE_FILE
                    if os.path.exists(_dash_file):
                        with open(_dash_file, "r", encoding="utf-8") as _df:
                            _dash_data = json.load(_df)
                        _modified = False
                        for _sig in _dash_data.get("live_signals", []):
                            if isinstance(_sig, dict) and _sig.get("ativo") == ativo:
                                _sig["direction"] = direcao
                                _sig["original_direction"] = _direcao_original
                                _sig["inverted"] = _was_inverted
                                _modified = True
                        if _modified:
                            _safe_save_json(_dash_file, _dash_data)
                except Exception:
                    pass

                op = enviar_ordem(bx, ativo, direcao, stake, exp=_use_exp)
                if not op:
                    log.warning(paint(f"  ❌ Falha na ordem: {ativo}", C.R))
                    continue  # tentar próximo candidato

                op_type, op_id = op
                _log_live_trade(ativo, direcao, None, _live_entry_price, stake,
                                confidence=_decision_conf_pct, status="entry",
                                nn_data=_nn_entry_data,
                                decision_id=_decision_id,
                                order_id=op_id)
                _decision_payload["order_id"] = int(op_id) if op_id is not None else None
                _save_trade_decision(_decision_payload)
                _last_entry_key = _entry_key  # Marcar padrão como operado
                _last_trade_time = time.time()  # Marcar tempo do trade para TIME DEDUP
                _save_last_entry_key(_entry_key)  # Persistir em disco

                # ═══ MEMÓRIA DT: Gravar nível APÓS entrada com sucesso ═══
                _rs_mem = pat_data.get("right_shoulder", {}).get("price", 0)
                if _rs_mem > 0:
                    _memorize_dt_level(ativo, _rs_mem, direcao)

                # ═══ GRAVAR DIREÇÃO: bloquear sinais contrários futuros ═══
                _record_entry_dir(ativo, direcao)
                log.info(paint(
                    f"  ✅ ENTRADA: {ativo} {direcao} @ {_live_entry_price or 0:.6f} | Stake={stake:.2f} | "
                    f"Tipo={op_type} | EXP={_use_exp}min | Modo={'EARLY' if _is_early else ('DT' if _is_dt else 'CLASSIC')} | "
                    f"NN={_fmt_pct(_nn_score) if _nn_pred is not None else 'indisponível'}"
                    f"{f' [{_nn_reason_text}]' if _nn_reason_text and _nn_pred is None else ''} | Amostras={ia_samples}",
                    C.G if direcao == "CALL" else C.R
                ))
                print(
                    f">>> IA: Entrada {ativo} {direcao} @{_live_entry_price or 0:.6f} "
                    f"stake={stake:.2f} nn={_fmt_num(_nn_score) if _nn_pred is not None else 'indisponível'}"
                    f"{f' [{_nn_reason_text}]' if _nn_reason_text and _nn_pred is None else ''}",
                    flush=True
                )

                # ═══ AGUARDAR RESULTADO ═══
                res = wait_result(bx, op_type, op_id)
                total_trades += 1

                if res > 0:
                    total_wins += 1
                    _live_status = "win"
                    _consecutive_losses = 0  # reset na sequência de losses
                    log.info(paint(f"  ✅ WIN +{res:.2f}$", C.G))
                    print(f">>> RESULTADO: WIN {ativo} {direcao} +{res:.2f}", flush=True)
                elif res < 0:
                    _live_status = "loss"
                    _consecutive_losses += 1
                    log.info(paint(f"  ❌ LOSS {res:.2f}$ (consecutivos: {_consecutive_losses})", C.R))
                    print(f">>> RESULTADO: LOSS {ativo} {direcao} {res:.2f}", flush=True)
                else:
                    _live_status = "tie"
                    log.info(paint(f"  ⚪ EMPATE", C.B))
                    print(f">>> RESULTADO: EMPATE {ativo}", flush=True)

                _log_live_trade(ativo, direcao, res, _live_entry_price, stake,
                                confidence=_decision_conf_pct, status=_live_status,
                                decision_id=_decision_id, order_id=op_id)

                # ═══ ATUALIZAR DECISION LOG com resultado ═══
                _update_trade_decision_result(
                    ativo, direcao, res, _live_status,
                    decision_id=_decision_id,
                    order_id=op_id,
                )

                # ── IA: aprender com o resultado ──
                ai_update(ativo, setup, res, hs_stats)

                # ── Resultado: tracking ──
                _trade_result_01 = 1 if res > 0 else 0
                _recent_trade_results.append(_trade_result_01)
                if len(_recent_trade_results) > 50:
                    del _recent_trade_results[:-50]
                _arm_key_res = f"{ativo}_{pat_type}_{setup.get('mode', 'classic')}"
                _n_arm = hs_stats.get("arms", {}).get(_arm_key_res, {}).get("total", 0)
                log.info(paint(
                    f"  🤖 IA atualizada: {ativo} | resultado={'WIN' if res > 0 else 'LOSS' if res < 0 else 'EMPATE'} | "
                    f"prob_antes={ia_prob:.2f} | amostras_ativo={_n_arm}",
                    C.B
                ))
                _safe_save_json(AI_STATS_FILE, hs_stats)
                # Salvar controle de retrain SOMENTE quando IA tem amostras reais
                if _n_arm > 0:
                    _save_retrain_control()

                # ── Estatísticas ──
                wr = (total_wins / max(1, total_trades)) * 100
                try:
                    saldo_now = float(bx.get_balance())
                    lucro = saldo_now - saldo_inicial
                    meta_val = saldo_inicial * META_LUCRO_PERCENT / 100.0
                    log.info(paint(
                        f"  📈 Sessão: {total_trades} trades | {total_wins}W | WR={wr:.1f}% | "
                        f"Lucro: {'+' if lucro >= 0 else ''}{lucro:.2f}",
                        C.G if lucro >= 0 else C.R
                    ))
                    print(f">>> IA: {total_trades} trades | WR={wr:.1f}% | {'+' if lucro >= 0 else ''}{lucro:.2f}", flush=True)
                except Exception:
                    log.info(f"  📈 Sessão: {total_trades} trades | {total_wins}W | WR={wr:.1f}%")

                # Exportar stats para o UI (per-ativo)
                for _rai_save in reversal_ai_map.values():
                    _rai_save.save_stats_to_disk()

                _candidate_traded = True
                break  # candidato aprovado e operado — sair do loop

            # ── Se nenhum candidato foi operado, esperar próximo candle ──
            if not _candidate_traded:
                s = seconds_to_next(TF_M1)
                time.sleep(min(s + 1, 30))

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
