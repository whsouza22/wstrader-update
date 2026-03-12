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
from ws_adaptive_brain import extract_features

# ═══ IA 4 — GUARD GENERATIVA (IA WS Generativa) ═══
from ws_generative_guard import gpt_guard_check

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
PAYOUT_MINIMO = int(os.getenv("WS_PAYOUT_MIN", "80"))   # 80%+ payout → com WR 90%+ é lucrativo
PAYOUT_REFRESH_SEC = int(os.getenv("WS_PAYOUT_REFRESH", "180"))

NUM_ATIVOS = int(os.getenv("WS_NUM_ATIVOS", "4"))  # TOP 4 ativos DT

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
META_LUCRO_PERCENT = float(os.getenv("WS_META_LUCRO", "10.0"))
STOP_LOSS_PERCENT = float(os.getenv("WS_STOP_LOSS", "3.0"))
USE_DYNAMIC_STAKE = (os.getenv("WS_DYNAMIC_STAKE", "1").strip() == "1")

# ── MODO IA: Guards 1-5 desativados — somente as 3 NNs decidem (treinadas com 63K+ samples) ──
_GUARDS_DISABLED = True  # Guards 1-5 OFF — as 3 IAs (89.7% acc) já sabem filtrar

# ── WS Trader 2.0: DESATIVADO — NN com 40 features é a única decisão ──
_DECISION_ENGINE_ENABLED = False  # Desativado: regime/quality/risk são features f26-f39 no NN
_recent_trade_results = []        # mantido para log/estatísticas

# ── Reversal AI config ──
CONFIDENCE_MIN = float(os.getenv('WS_CONF_MIN', "40.0"))       # Confiança mínima da IA para entrar
RETRAIN_INTERVAL_MIN = int(os.getenv("WS_RETRAIN_MIN", "5"))     # Retreinar a cada 5 minutos
ANALYZE_AT_SECOND = int(os.getenv("WS_ANALYZE_SEC", "45"))      # Analisar no segundo :45 (antes da vela fechar, scan ~12s, entra na virada :00)
COOLDOWN_AFTER_TRADE = int(os.getenv("WS_COOLDOWN", "180"))      # Cooldown global após cada trade (3 min)
MIN_WR_ATIVO = float(os.getenv("WS_MIN_WR", "80.0"))            # WR mínimo para selecionar ativo
MAX_ENTRY_DELAY_SEC = float(os.getenv("WS_MAX_ENTRY_DELAY_SEC", "1.2"))
ENABLE_GPT_DT_ADVISORY = (os.getenv("WS_GPT_DT", "0").strip() == "1")


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

# ── Variáveis para Engine / IA ──
DECIDIR_ANTES_FECHAR_SEC = int(os.getenv("WS_DECIDIR_ANTES_FECHAR", "12"))
IA_ON = True  # IA SEMPRE ativa para H&S
AI_STATS_FILE = os.path.join(os.path.expanduser("~"), ".wstrader", "ws_ai_stats_hs.json")
AI_MIN_SAMPLES = 5
AI_CONF_MIN = 0.3
AI_MIN_PROB = 0.55  # CORRIGIDO: era 0.40 (permitia entradas com 40% prob = moeda)
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


def _get_session_params(guard_df=None, atr_val=0.0):
    """Retorna parâmetros de sessão conforme o perfil ao vivo selecionado."""
    if DT_LIVE_PROFILE == "elite":
        return {
            "profile": "elite",
            "nn_min_prob": 0.88,
            "exp_minutes": 2,
            "cooldown_sec": 4 * 60,
        }
    if DT_LIVE_PROFILE == "moderate":
        return {
            "profile": "moderate",
            "nn_min_prob": 0.82,
            "exp_minutes": 2,
            "cooldown_sec": 3 * 60,
        }
    return {
        "profile": "standard",
        "nn_min_prob": 0.76,
        "exp_minutes": 2,
        "cooldown_sec": 2 * 60,
    }


def _get_dt_live_guard_params():
    """Parâmetros do guard DT conforme o perfil ao vivo selecionado."""
    if DT_LIVE_PROFILE == "elite":
        return {
            "scan": {
                "lookback": 4,
                "progress_max_pct": 18.0,
                "touch_tol_atr": 0.22,
                "close_tol_atr": 0.08,
                "wrong_side_tol_atr": 0.10,
                "wick_min_ratio": 0.30,
                "body_tol_atr": 0.05,
            },
            "final": {
                "lookback": 2,
                "progress_max_pct": 16.0,
                "touch_tol_atr": 0.16,
                "close_tol_atr": 0.06,
                "wrong_side_tol_atr": 0.08,
                "wick_min_ratio": 0.30,
                "body_tol_atr": 0.04,
            },
        }
    if DT_LIVE_PROFILE == "moderate":
        return {
            "scan": {
                "lookback": 4,
                "progress_max_pct": 22.0,
                "touch_tol_atr": 0.26,
                "close_tol_atr": 0.10,
                "wrong_side_tol_atr": 0.14,
                "wick_min_ratio": 0.25,
                "body_tol_atr": 0.06,
            },
            "final": {
                "lookback": 2,
                "progress_max_pct": 20.0,
                "touch_tol_atr": 0.18,
                "close_tol_atr": 0.07,
                "wrong_side_tol_atr": 0.09,
                "wick_min_ratio": 0.28,
                "body_tol_atr": 0.045,
            },
        }
    return {
        "scan": {
            "lookback": 5,
            "progress_max_pct": 28.0,
            "touch_tol_atr": 0.30,
            "close_tol_atr": 0.12,
            "wrong_side_tol_atr": 0.18,
            "wick_min_ratio": 0.20,
            "body_tol_atr": 0.08,
        },
        "final": {
            "lookback": 2,
            "progress_max_pct": 25.0,
            "touch_tol_atr": 0.20,
            "close_tol_atr": 0.08,
            "wrong_side_tol_atr": 0.10,
            "wick_min_ratio": 0.25,
            "body_tol_atr": 0.05,
        },
    }


def _dt_profile_runtime_filter(geo: Optional[dict], nn_pred: Optional[dict],
                               ia_prob: float, wick_pct: float) -> dict:
    """Filtro de assertividade ao vivo conforme o perfil DT.

    Objetivo: reduzir entradas medianas e privilegiar apenas as faixas que,
    na análise histórica, mostraram WR próximo ou acima de 90%.
    """
    if DT_LIVE_PROFILE == "standard":
        return {"ok": True, "reason": "perfil standard", "profile": "standard"}

    profile_thresholds = {
        "moderate": {
            "nn_score": 0.82,
            "prob_win": 0.80,
            "consensus_penalty": 0.05,
            "base_pair": 0.76,
            "p3": 0.72,
            "ia_prob": 0.67,
            "depth_ratio": 2.80,
            "d_right": 18,
            "d_left": 15,
            "span": 34,
            "shoulder_ratio": 0.985,
            "wick_pct": 28,
        },
        "elite": {
            "nn_score": 0.88,
            "prob_win": 0.86,
            "consensus_penalty": 0.03,
            "base_pair": 0.80,
            "p3": 0.78,
            "ia_prob": 0.70,
            "depth_ratio": 2.28,
            "d_right": 15,
            "d_left": 12,
            "span": 28,
            "shoulder_ratio": 0.9999,
            "wick_pct": 35,
        },
    }
    thresholds = profile_thresholds.get(DT_LIVE_PROFILE, profile_thresholds["moderate"])

    if geo is None:
        return {"ok": False, "reason": f"geometria indisponível no perfil {DT_LIVE_PROFILE}", "profile": DT_LIVE_PROFILE}
    if nn_pred is None:
        return {"ok": False, "reason": f"NN indisponível no perfil {DT_LIVE_PROFILE}", "profile": DT_LIVE_PROFILE}

    reasons = []
    nn_score = float(nn_pred.get("nn_score", nn_pred.get("prob_win", 0)) or 0)
    prob_win = float(nn_pred.get("prob_win", nn_score) or 0)
    consensus_penalty = float(nn_pred.get("consensus_penalty", 0) or 0)
    p1 = float(nn_pred.get("p1", 0) or 0)
    p2 = float(nn_pred.get("p2", 0) or 0)
    p3_raw = nn_pred.get("p3")
    p3 = float(p3_raw or 0) if p3_raw is not None else None

    if nn_score < thresholds["nn_score"]:
        reasons.append(f"nn_score={nn_score:.0%}<{thresholds['nn_score']:.0%}")
    if prob_win < thresholds["prob_win"]:
        reasons.append(f"prob_win={prob_win:.0%}<{thresholds['prob_win']:.0%}")
    if consensus_penalty > thresholds["consensus_penalty"]:
        reasons.append(f"consenso={consensus_penalty:.2f}>{thresholds['consensus_penalty']:.2f}")
    if min(p1, p2) < thresholds["base_pair"]:
        reasons.append(f"base fraca p1/p2={min(p1, p2):.0%}<{thresholds['base_pair']:.0%}")
    if p3 is not None and p3 < thresholds["p3"]:
        reasons.append(f"p3={p3:.0%}<{thresholds['p3']:.0%}")
    if ia_prob < thresholds["ia_prob"]:
        reasons.append(f"ia_prob={ia_prob:.0%}<{thresholds['ia_prob']:.0%}")

    if geo.get("depth_ratio", 99) > thresholds["depth_ratio"]:
        reasons.append(f"depth_ratio={geo.get('depth_ratio', 0):.2f}>{thresholds['depth_ratio']:.2f}")
    if geo.get("d_right", 99) > thresholds["d_right"]:
        reasons.append(f"d_right={geo.get('d_right', 0)}>{thresholds['d_right']}")
    if geo.get("d_left", 99) > thresholds["d_left"]:
        reasons.append(f"d_left={geo.get('d_left', 0)}>{thresholds['d_left']}")
    if geo.get("span", 99) > thresholds["span"]:
        reasons.append(f"span={geo.get('span', 0)}>{thresholds['span']}")
    if geo.get("shoulder_ratio", 0) < thresholds["shoulder_ratio"]:
        reasons.append(
            f"shoulder_ratio={geo.get('shoulder_ratio', 0):.6f}<{thresholds['shoulder_ratio']:.6f}"
        )
    if wick_pct < thresholds["wick_pct"]:
        reasons.append(f"wick={wick_pct:.0f}%<{thresholds['wick_pct']:.0f}%")

    if reasons:
        return {"ok": False, "reason": " | ".join(reasons), "profile": DT_LIVE_PROFILE}
    return {"ok": True, "reason": f"{DT_LIVE_PROFILE} ok", "profile": DT_LIVE_PROFILE}


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
_user_data_dir = os.path.join(os.path.expanduser("~"), ".wstrader")
os.makedirs(_user_data_dir, exist_ok=True)

_ENTRY_GUARD_ENABLED = (os.getenv("WS_ENTRY_GUARD", "1").strip() == "1")
_ENTRY_GUARD_POOL_SIZE = int(os.getenv("WS_ENTRY_GUARD_POOL", "12"))
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


def _ensure_dashboard_server():
    """Garante que o dashboard HTTP esteja ativo na porta 8899."""
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        sock.connect(("127.0.0.1", 8899))
        sock.close()
        log.info(paint("📊 Dashboard H&S já rodando na porta 8899", C.G))
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
    log.info(paint("📊 Dashboard H&S iniciado automaticamente na porta 8899", C.G))


def _entry_guard_model_path(ativo: str) -> str:
    return os.path.join(_user_data_dir, f"entry_guard_{ativo}.pkl")


def _reversal_model_path(ativo: str) -> str:
    return os.path.join(_user_data_dir, f"reversal_tf_{ativo}.pkl")


def _load_entry_guard_bundle(ativo: str) -> Optional[dict]:
    if ativo in _entry_guard_cache:
        return _entry_guard_cache[ativo]

    model_path = _entry_guard_model_path(ativo)
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
    profile_bump = {
        "standard": 0.00,
        "moderate": 0.03,
        "elite": 0.06,
    }.get(DT_LIVE_PROFILE, 0.00)
    return max(0.50, min(0.95, base_threshold + profile_bump))


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
        }
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
    """Detecta TODOS os padrões H&S/iH&S no histórico de velas.
    Inclui validações: cabeça não pode ter sido rompida.
    Simetria temporal: braços não podem diferir mais de 3:1."""
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
            "entry_idx": int(iR),
            "entry_price": round(float(C_arr[int(iR)]), 6),
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
            "entry_idx": int(iR),
            "entry_price": round(float(C_arr[int(iR)]), 6),
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
        # Limitar busca: máx 3x a distância do braço esquerdo (evita ombro muito distante)
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
        # Pelo menos 2 velas antes E depois devem ser menores
        _pivot_check = min(3, iR - search_start, n - 1 - iR)
        if _pivot_check < 2: continue
        _is_pivot = all(H[iR] >= H[iR - j] for j in range(1, _pivot_check + 1)) and \
                    all(H[iR] >= H[iR + j] for j in range(1, min(_pivot_check + 1, n - iR)))
        if not _is_pivot: continue
        # Simetria temporal
        d_right = iR - iH
        if min(d_left, d_right) / max(d_left, d_right) < temporal_sym_min: continue
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
            "entry_price": round(float(C_arr[int(iR)]), 6),
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
            "entry_price": round(float(C_arr[int(iR)]), 6),
        })

    return patterns


def detect_early_hs(H, L, C_arr, O, pivot_highs, pivot_lows, atr, n):
    """Detecção ANTECIPADA de H&S: LS+Head confirmados, RS pela vela mais recente.

    Em vez de esperar detect_pivots confirmar o Ombro D com 5 candles futuros,
    esta função identifica a formação do Ombro D no MOMENTO que acontece:
    - LS e Head: pivots confirmados (window=5, alta qualidade)
    - RS: última vela fechada que atinge a zona do LS e mostra rejeição (wick)

    No LIVE, o bot escaneia no :05 (5 seg após fechar a vela).
    Se a última vela fechada é um "RS candidato" com padrão de rejeição,
    entra IMEDIATAMENTE no turbo → delay ≈ 0.

    Isso replica a entrada do backtest (delay=0, WR 89%).
    """
    patterns = []
    if n < 30:
        return patterns

    tol = atr * 1.5
    min_depth = atr * 1.0
    min_spacing = 8
    max_span = 100
    trend_lookback = 30
    symmetry_min = 0.90
    temporal_sym_min = 0.30

    # Foco: última vela fechada = índice n-1
    # (No :05, a vela anterior acabou de fechar)
    rs_candidates = [n - 1]

    # ── H&S EARLY (PUT): LS + Head confirmados, RS na última vela ──
    for i in range(len(pivot_highs) - 1):
        iL, pL = pivot_highs[i]
        iH, pH = pivot_highs[i + 1]
        if pH <= pL:
            continue
        if iH - iL < min_spacing:
            continue
        head_depth = pH - pL
        if head_depth < min_depth:
            continue
        # Trend check: tendência prévia deve ser ALTA
        if iL >= trend_lookback:
            if float(C_arr[iL]) <= float(C_arr[iL - trend_lookback]):
                continue

        d_left = iH - iL

        for j in rs_candidates:
            if j <= iH + min_spacing:
                continue
            if j - iL > max_span:
                continue

            h_j = float(H[j])
            l_j = float(L[j])
            o_j = float(O[j])
            c_j = float(C_arr[j])

            # RS: HIGH perto do LS
            if abs(h_j - pL) > tol:
                continue
            if h_j >= pH:
                continue
            pR = h_j
            if min(pL, pR) / max(pL, pR) < symmetry_min:
                continue

            # Simetria temporal
            d_right = j - iH
            if min(d_left, d_right) / max(d_left, d_right) < temporal_sym_min:
                continue

            # Cabeça não pode ter sido quebrada
            if iH + 1 <= j:
                if float(max(H[iH + 1:j + 1])) >= pH:
                    continue

            # ══ FILTRO CRUCIAL: padrão de rejeição no RS ══
            # Sem pivot confirmado, a vela DEVE mostrar rejeição:
            # - Upper wick > body (rejeitou do topo)
            # - Upper wick > 25% do range total
            body = abs(c_j - o_j)
            total_range = h_j - l_j
            if total_range < 1e-7:
                continue
            upper_wick = h_j - max(c_j, o_j)
            if upper_wick <= body:
                continue
            if upper_wick <= total_range * 0.25:
                continue

            # Neckline
            v1_region = L[iL:iH + 1]
            v1_rel = int(np.argmin(v1_region))
            v1_idx = iL + v1_rel
            v1_price = float(v1_region[v1_rel])
            v2_region = L[iH:j + 1]
            v2_rel = int(np.argmin(v2_region))
            v2_idx = iH + v2_rel
            v2_price = float(v2_region[v2_rel])
            neckline = (v1_price + v2_price) / 2
            if abs(v1_price - v2_price) > atr * 0.5:
                continue
            neck_slope = (v2_price - v1_price) / max(1, v2_idx - v1_idx)

            patterns.append({
                "type": "HEAD_SHOULDERS", "direction": "PUT", "mode": "early",
                "left_shoulder": {"idx": int(iL), "price": round(float(pL), 6)},
                "head": {"idx": int(iH), "price": round(float(pH), 6)},
                "right_shoulder": {"idx": int(j), "price": round(float(pR), 6)},
                "valley1": {"idx": int(v1_idx), "price": round(v1_price, 6)},
                "valley2": {"idx": int(v2_idx), "price": round(v2_price, 6)},
                "neckline": round(neckline, 6),
                "neck_slope": round(neck_slope, 8),
                "depth": round(float(head_depth), 6),
                "target": round(neckline - head_depth, 6),
                "stop": round(float(pH), 6),
                "entry_idx": int(j),
                "entry_price": round(c_j, 6),
            })

    # ── iH&S EARLY (CALL): LS + Head confirmados, RS na última vela ──
    for i in range(len(pivot_lows) - 1):
        iL, pL = pivot_lows[i]
        iH, pH = pivot_lows[i + 1]
        if pH >= pL:
            continue
        if iH - iL < min_spacing:
            continue
        head_depth = pL - pH
        if head_depth < min_depth:
            continue
        if iL >= trend_lookback:
            if float(C_arr[iL]) >= float(C_arr[iL - trend_lookback]):
                continue

        d_left = iH - iL

        for j in rs_candidates:
            if j <= iH + min_spacing:
                continue
            if j - iL > max_span:
                continue

            h_j = float(H[j])
            l_j = float(L[j])
            o_j = float(O[j])
            c_j = float(C_arr[j])

            if abs(l_j - pL) > tol:
                continue
            if l_j <= pH:
                continue
            pR = l_j
            if min(pL, pR) / max(pL, pR) < symmetry_min:
                continue

            d_right = j - iH
            if min(d_left, d_right) / max(d_left, d_right) < temporal_sym_min:
                continue

            if iH + 1 <= j:
                if float(min(L[iH + 1:j + 1])) <= pH:
                    continue

            # Filtro de rejeição: lower wick > body
            body = abs(c_j - o_j)
            total_range = h_j - l_j
            if total_range < 1e-7:
                continue
            lower_wick = min(c_j, o_j) - l_j
            if lower_wick <= body:
                continue
            if lower_wick <= total_range * 0.25:
                continue

            v1_region = H[iL:iH + 1]
            v1_rel = int(np.argmax(v1_region))
            v1_idx = iL + v1_rel
            v1_price = float(v1_region[v1_rel])
            v2_region = H[iH:j + 1]
            v2_rel = int(np.argmax(v2_region))
            v2_idx = iH + v2_rel
            v2_price = float(v2_region[v2_rel])
            neckline = (v1_price + v2_price) / 2
            if abs(v1_price - v2_price) > atr * 0.5:
                continue
            neck_slope = (v2_price - v1_price) / max(1, v2_idx - v1_idx)

            patterns.append({
                "type": "INV_HEAD_SHOULDERS", "direction": "CALL", "mode": "early",
                "left_shoulder": {"idx": int(iL), "price": round(float(pL), 6)},
                "head": {"idx": int(iH), "price": round(float(pH), 6)},
                "right_shoulder": {"idx": int(j), "price": round(float(pR), 6)},
                "valley1": {"idx": int(v1_idx), "price": round(v1_price, 6)},
                "valley2": {"idx": int(v2_idx), "price": round(v2_price, 6)},
                "neckline": round(neckline, 6),
                "neck_slope": round(neck_slope, 8),
                "depth": round(float(head_depth), 6),
                "target": round(neckline + head_depth, 6),
                "stop": round(float(pH), 6),
                "entry_idx": int(j),
                "entry_price": round(c_j, 6),
            })

    return patterns


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
    tol = atr * 0.4
    min_spacing = 12
    max_spacing = 60
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
                if c_j > l_j + candle_range * 0.40:
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
                if c_j < h_j - candle_range * 0.40:
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

            # Detectar pivots e H&S + Duplo Toque
            ph, pl = detect_pivots(H, L, window=5)
            all_hs = detect_all_hs(H, L, C_arr, O, ph, pl, atr)
            all_dt = detect_double_touch(H, L, C_arr, O, ph, pl, atr, n,
                                         max_candles_ago=9999, training=True)
            all_patterns = all_hs + all_dt

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


def _write_dashboard_cache(dashboard_assets: dict, payouts: dict):
    """Escreve cache compartilhado para o dashboard (read-only).
    O bot é a ÚNICA fonte de dados — dashboard nunca conecta ao broker."""
    try:
        cache = {
            "ts": time.time(),
            "broker": BROKER_TYPE,
            "assets": {},
        }
        for ativo, info in dashboard_assets.items():
            df = info.get("df")
            if df is None or len(df) < 10:
                continue
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
            }
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

    for ativo in ativos:
        # IMPORTANTE: sempre varrer TODOS os ativos antes da decisão final.
        # Não encerrar o scan ao encontrar o 1º padrão, senão a IA considera
        # apenas um ativo e ignora setups melhores nos demais pares.
        # ── Buscar candles DIRETO da corretora ──
        df = get_candles_df(bx, ativo, TF_M1, N_M1)
        if df is None or len(df) < 100:
            continue

        # Acumular para o dashboard (SEMPRE, independente de padrão)
        _dashboard_assets[ativo] = {"df": df}

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

        # ── SOMENTE Double Touch (H&S removido — DT tem WR melhor no live) ──
        # max_candles_ago=3: permite detectar 2º toque até 3 velas atrás
        # (backtest_pattern já filtra os que já têm resultado — candles_ago > EXP_FIXA-1)
        patterns = detect_double_touch(H, L, C_arr, O, pivot_highs, pivot_lows, atr, n, max_candles_ago=3)

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

                # ═══ FIX LIVE #4: Aceitar APENAS candles_ago ≤ 2 ═══
                # delay=3+ → WR degrada. EXCEÇÃO: early/double_touch
                if pat.get("mode") not in ("early", "double_touch") and pat["candles_ago"] > 2:
                    log.info(paint(
                        f"  ⛔ SKIP: {ativo} candles_ago={pat['candles_ago']} > 2 "
                        f"(delay muito alto — WR degrada abaixo de 55%)",
                        C.Y
                    ))
                    continue

                # ═══ DT: Geometria fraca vira alerta, não bloqueio ═══
                # A NN já recebe span/symmetry/depth como features e decide sozinha.
                if pat.get("mode") == "double_touch":
                    _geo_check = _extract_geometry(pat, atr)
                    if _geo_check:
                        _sym = _geo_check.get("symmetry", 0)
                        _span = _geo_check.get("span", 0)
                        _depth = _geo_check.get("depth_ratio", 0)
                        if _sym < 0.40:
                            log.info(paint(
                                f"  ⚠️ GEO FRACA: {ativo} symmetry={_sym:.2f} < 0.40 "
                                f"(a NN vai decidir)",
                                C.Y
                            ))
                        if _span < 12:
                            log.info(paint(
                                f"  ⚠️ GEO FRACA: {ativo} span={_span} < 12 "
                                f"(a NN vai decidir)",
                                C.Y
                            ))
                        if _depth < 2.0:
                            log.info(paint(
                                f"  ⚠️ GEO FRACA: {ativo} depth_ratio={_depth:.2f} < 2.0 "
                                f"(a NN vai decidir)",
                                C.Y
                            ))

                live_patterns.append(pat)

        if not live_patterns:
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
        _directions = set(p["direction"] for p in live_patterns)
        if len(_directions) > 1:
            log.info(paint(
                f"  ⚠️ CONFLITO: {ativo} tem PUT + CALL simultâneos → SKIP (sinal ambíguo)",
                C.Y
            ))
            continue

        # ── SEM FILTRO DISTÂNCIA — apenas IA decide ──
        _total_patterns += len(live_patterns)

        for pat in live_patterns:
            direction = pat["direction"]
            pat_type = pat["type"]
            mode = pat.get("mode", "classic")

            # ═══ MEMÓRIA DT: Bloquear nível já operado ═══
            _rs_price_check = pat.get("right_shoulder", {}).get("price", 0)
            if _is_dt_level_already_traded(ativo, _rs_price_check, direction, atr):
                log.info(paint(
                    f"  🚫 NÍVEL JÁ OPERADO: {ativo} {direction} RS={_rs_price_check:.6f} "
                    f"— nível já operado (memória DT)",
                    C.R
                ))
                continue

            # ═══ MEMÓRIA DT: NÃO gravar aqui! ═══
            # Nível só é gravado APÓS a entrada ser executada com sucesso.
            # Se gravar na análise, o mesmo DT é bloqueado como 3º toque
            # no próximo scan (mesmo sem ter entrado).

            # IA estatística legada — mantida só para log/contexto.
            ia_prob = ai_predict_hs(ativo, pat, hs_stats)
            ia_n = hs_stats.get("arms", {}).get(f"{ativo}_{pat_type}_{mode}", {}).get("total", 0)

            # IA Geométrica — mantida só para log/contexto.
            _pq, _ = ia_pattern_quality(pat, atr, hs_stats)

            # Score principal de seleção = NN. Fallback = IA estatística legada.
            _nn_pre_score, _nn_pre_pred = _estimate_dt_nn_score(
                ativo, pat, df, atr, hs_stats, reversal_ai_map
            )
            _entry_guard_pre_score, _entry_guard_pre_pred = _estimate_entry_guard_score(
                ativo, pat, df, atr, hs_stats
            )
            if _entry_guard_pre_score is not None and _nn_pre_score is not None:
                score = (_entry_guard_pre_score * 0.60) + (float(_nn_pre_score) * 0.40)
            elif _entry_guard_pre_score is not None:
                score = float(_entry_guard_pre_score)
            else:
                score = float(_nn_pre_score) if _nn_pre_score is not None else ia_prob

            setup = {
                "dir": direction,
                "type": pat_type,
                "mode": mode,
                "confidence": round(score * 100, 1),
                "pattern": pat,
                "last_close": float(C_arr[-1]),  # preço atual para guards (evita 2ª API call)
                "last_close_prev": float(C_arr[-2]) if n >= 2 else float(C_arr[-1]),
                "nn_pre_score": round(float(_nn_pre_score), 4) if _nn_pre_score is not None else None,
                "nn_pre_pred": _nn_pre_pred,
                "entry_guard_pre_score": round(float(_entry_guard_pre_score), 4) if _entry_guard_pre_score is not None else None,
                "entry_guard_pre_pred": _entry_guard_pre_pred,
            }

            if _entry_guard_pre_score is not None and _nn_pre_score is not None:
                _score_label = f"EG={_entry_guard_pre_score:.2f} + NN={_nn_pre_score:.2f}"
            elif _entry_guard_pre_score is not None:
                _score_label = f"EG={_entry_guard_pre_score:.2f}"
            elif _nn_pre_score is not None:
                _score_label = f"NN={_nn_pre_score:.2f}"
            else:
                _score_label = "fallback estatístico"

            log.info(paint(
                f"  📊 H&S LOCAL: {ativo} | {pat_type} {direction} ({mode}) | "
                f"score={score:.2f} ({_score_label} | geom={_pq:.2f}) | entry_idx={pat['entry_idx']} | n={n}",
                C.B
            ))

            if best_any is None or score > best_any[0]:
                best_any = (score, ativo, setup, atr)

            # ═══ TODOS OS CANDIDATOS — NN/IA avaliam cada um ═══
            all_candidates.append((score, ativo, setup, atr))

            if best_trade is None or score > best_trade[0]:
                best_trade = (score, ativo, setup, atr)

    if _total_patterns > 0:
        log.info(paint(f"  🔍 Scan local: {_total_patterns} padrão(ões) H&S recente(s) encontrado(s)", C.G))

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
        _write_dashboard_cache(_dashboard_assets, _payouts)
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


def _estimate_dt_nn_score(ativo: str, pat: dict, df: Optional[pd.DataFrame], atr_val: float,
                          hs_stats: dict, reversal_ai_map: Optional[dict] = None):
    if df is None or len(df) < 50 or reversal_ai_map is None:
        return None, None

    _rai = reversal_ai_map.get(ativo)
    if _rai is None or not getattr(_rai, "_ai1_ready", False) or not getattr(_rai, "_ai2_ready", False):
        return None, None

    try:
        _H = df["high"].values
        _L = df["low"].values
        _C = df["close"].values
        _O = df["open"].values
        _n = len(_H)
        _rs_idx = int(pat.get("right_shoulder", {}).get("idx", _n - 1))
        _rs_idx = max(0, min(_rs_idx, _n - 1))

        _win_start = max(0, _rs_idx - 55)
        _win_end = min(_n, _rs_idx + 1)
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

        _pat_copy = dict(pat)
        _pat_copy["candles_ago"] = max(0, _n_win - 1 - (_rs_idx - _win_start))
        _feats = extract_features(_pat_copy, _H_win, _L_win, _C_win, _O_win, _n_win,
                                  _atr_local, hs_stats, ativo)
        if _feats is None:
            return None, None

        _pred = _rai.predict_dt(_feats)
        if _pred is None:
            return None, None

        return _pred.get("nn_score"), _pred
    except Exception:
        return None, None


def _validate_dt_entry_region(df: Optional[pd.DataFrame], direcao: str,
                              rs_price: float, neckline: float,
                              atr_val: float, cur_price: Optional[float],
                              lookback: int = 3,
                              progress_max_pct: float = 25.0,
                              touch_tol_atr: float = 0.20,
                              close_tol_atr: float = 0.08,
                              wrong_side_tol_atr: float = 0.10,
                              wick_min_ratio: float = 0.25,
                              body_tol_atr: float = 0.05) -> dict:
    if df is None or len(df) < 1:
        return {"ok": False, "reason": "dados de candles indisponíveis", "progress_pct": None}
    if cur_price is None or rs_price <= 0:
        return {"ok": False, "reason": "preço atual/RS inválido", "progress_pct": None}

    _atr_safe = atr_val if atr_val and atr_val > 0 else max(abs(rs_price) * 0.0005, 1e-6)
    _touch_tol = _atr_safe * touch_tol_atr
    _close_tol = _atr_safe * close_tol_atr
    _wrong_side_tol = _atr_safe * wrong_side_tol_atr
    _body_tol = _atr_safe * body_tol_atr

    _recent = df.tail(max(1, lookback))
    _touch_row = None
    _touch_reason = None

    for _i in range(len(_recent) - 1, -1, -1):
        _row = _recent.iloc[_i]
        _open = float(_row["open"])
        _high = float(_row["high"])
        _low = float(_row["low"])
        _close = float(_row["close"])
        _range = max(_high - _low, 1e-9)

        if direcao == "CALL":
            _touched = _low <= rs_price + _touch_tol
            _wick = min(_open, _close) - _low
            _wick_ok = _wick >= _range * wick_min_ratio
            _close_ok = _close >= rs_price - _close_tol
            _body_ok = _close >= _open - _body_tol
            if _touched:
                if not _close_ok:
                    _touch_reason = f"tocou suporte mas fechou abaixo do suporte ({_close:.6f})"
                elif not _body_ok:
                    _touch_reason = f"tocou suporte mas fechou vendedor ({_open:.6f}->{_close:.6f})"
                elif not _wick_ok:
                    _touch_reason = f"tocou suporte sem rejeição suficiente (wick={_wick / _range:.0%})"
                else:
                    _touch_row = _row
                    break
        else:
            _touched = _high >= rs_price - _touch_tol
            _wick = _high - max(_open, _close)
            _wick_ok = _wick >= _range * wick_min_ratio
            _close_ok = _close <= rs_price + _close_tol
            _body_ok = _close <= _open + _body_tol
            if _touched:
                if not _close_ok:
                    _touch_reason = f"tocou resistência mas fechou rompendo acima ({_close:.6f})"
                elif not _body_ok:
                    _touch_reason = f"tocou resistência mas fechou comprador ({_open:.6f}->{_close:.6f})"
                elif not _wick_ok:
                    _touch_reason = f"tocou resistência sem rejeição suficiente (wick={_wick / _range:.0%})"
                else:
                    _touch_row = _row
                    break

    if _touch_row is None:
        return {
            "ok": False,
            "reason": _touch_reason or f"nenhuma das últimas {len(_recent)} velas tocou e rejeitou o RS",
            "progress_pct": None,
        }

    _rs_to_neck = abs(neckline - rs_price)
    _dist_to_rs = abs(cur_price - rs_price)
    _progress_pct = (_dist_to_rs / _rs_to_neck * 100) if _rs_to_neck > 0 else 0.0
    _wrong_side = max(0.0, cur_price - rs_price) if direcao == "PUT" else max(0.0, rs_price - cur_price)

    if direcao == "PUT" and neckline > 0 and cur_price <= neckline:
        return {"ok": False, "reason": f"preço já perdeu a neckline ({cur_price:.6f} <= {neckline:.6f})", "progress_pct": _progress_pct}
    if direcao == "CALL" and neckline > 0 and cur_price >= neckline:
        return {"ok": False, "reason": f"preço já rompeu a neckline ({cur_price:.6f} >= {neckline:.6f})", "progress_pct": _progress_pct}
    if _wrong_side > _wrong_side_tol:
        if direcao == "PUT":
            _reason = f"preço ainda acima da resistência em {_wrong_side / _atr_safe:.2f} ATR"
        else:
            _reason = f"preço ainda abaixo do suporte em {_wrong_side / _atr_safe:.2f} ATR"
        return {"ok": False, "reason": _reason, "progress_pct": _progress_pct}
    if _progress_pct > progress_max_pct:
        return {
            "ok": False,
            "reason": f"preço já andou {_progress_pct:.0f}% do caminho RS→Neck",
            "progress_pct": _progress_pct,
        }

    return {
        "ok": True,
        "reason": "toque e rejeição confirmados",
        "progress_pct": _progress_pct,
        "touch_close": round(float(_touch_row["close"]), 6),
        "touch_open": round(float(_touch_row["open"]), 6),
        "touch_high": round(float(_touch_row["high"]), 6),
        "touch_low": round(float(_touch_row["low"]), 6),
    }


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
        pool_size = max(n_top, _ENTRY_GUARD_POOL_SIZE, NUM_ATIVOS * 3)
        top = [item[0] for item in ranked[:pool_size]]
        for i, (asset, acc, auc, precision, samples) in enumerate(ranked[:min(n_top, len(ranked))]):
            log.info(paint(
                f"🎯 ASSET #{i+1}: {asset} | acc={acc:.1%} | auc={auc:.3f} | prec={precision:.1%} | amostras={samples}",
                C.G
            ))
        return top

    top = ["NZDJPY-OTC", "GBPAUD-OTC", "USDCAD-OTC", "EURNZD-OTC"][:max(n_top, NUM_ATIVOS)]
    for i, a in enumerate(top[:n_top]):
        log.info(paint(f"🎯 ASSET #{i+1}: {a} (fallback sem entry_guard treinado)", C.Y))
    return top


def obter_top_ativos_otc(bx: BrokerAPI) -> List[str]:
    global _cache_ativos, _cache_ativos_ts, _top_dt_assets
    # Refresh a cada 5 min (ativos podem abrir/fechar)
    if _cache_ativos and (time.time() - _cache_ativos_ts) < 300:
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
    try:
        all_profit = safe_call(bx, bx.get_all_profit)
        for t in targets:
            profit = all_profit.get(t, {}).get("turbo", 0)
            payout = int(profit * 100) if profit else 0
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

    _cache_ativos = good_targets[:NUM_ATIVOS]
    _cache_ativos_ts = time.time()
    log.info(paint(f"🎯 TOP {len(_cache_ativos)} ATIVOS: {_cache_ativos}", C.G))
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
# MAIN — LOOP PRINCIPAL (SOMENTE CABEÇA E OMBROS)
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
    bx = ensure_connected(bx)

    if _ENTRY_GUARD_ENABLED:
        _sync_entry_guard_models_from_github()

    # Inicializar ReversalAI POR ATIVO — cada ativo tem sua própria NN
    # Os modelos são carregados do disco (treinados offline via train_neural_network.py)
    # NUNCA treinar NN online — só usar os modelos pré-treinados
    reversal_ai_map = {}  # {ativo: ReversalAI} — preenchido após selecionar ativos

    # ── Carregar / Treinar IA — MEMÓRIA PERMANENTE ──
    # A IA NUNCA perde memória. Carrega do disco e ACUMULA.
    log.info(paint("🧠 Carregando memória da IA H&S...", C.B))

    # SEMPRE carregar stats existentes do disco (memória permanente)
    hs_stats = _safe_load_json(AI_STATS_FILE)
    _n_total = hs_stats.get("meta", {}).get("total", 0)

    if _n_total > 0:
        log.info(paint("💾 IA carregada do disco! Memória DT OK", C.G))
    else:
        log.info(paint("🌱 Primeira execução da IA...", C.Y))

    # ── PASSO 1: Carregar base pré-treinada (local ou GitHub) ──
    # Se disponível, PULA o treino local (já vem treinada do desenvolvedor)
    hs_stats = _load_or_download_training_base(hs_stats)
    _n_after_base = hs_stats.get("meta", {}).get("total", 0)

    # ── PASSO 2: Treino local com CSVs (87K velas por ativo) ──
    # SEMPRE treina se CSVs existem e ainda não foram processados
    _CSV_DIR_CHECK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candles_5000")
    _csvs_exist = os.path.isdir(_CSV_DIR_CHECK) and len(os.listdir(_CSV_DIR_CHECK)) > 0
    _trained_csv = hs_stats.get("meta", {}).get("trained_with_csv", False)
    _has_base = hs_stats.get("meta", {}).get("deep_train_version", "") != ""

    if _csvs_exist and not _trained_csv:
        log.info(paint("🏋️ CSVs de treino profundo detectados — treinando IA com todos os ativos...", C.B))
        hs_stats = _train_ia_from_history(bx, hs_stats)
    elif _has_base:
        log.info(paint(
            f"✅ Base pré-treinada OK! PULANDO treino local",
            C.G
        ))
    else:
        log.info(paint("🏋️ Sem base pré-treinada — treinando IA localmente...", C.B))
        hs_stats = _train_ia_from_history(bx, hs_stats)

    log.info("=" * 60)
    log.info(paint(f"🚀 WS TRADER — Double Touch / Multi-Asset ({_BROKER_LABEL})", C.G))
    log.info("=" * 60)

    # ── SELECIONAR MELHOR ATIVO DT (baseado no treino + benchmark) ──
    global _top_dt_assets
    _top_dt_assets = _pick_top_dt_assets(hs_stats, n_top=max(NUM_ATIVOS, _ENTRY_GUARD_POOL_SIZE))

    # ── Carregar modelo NN per-ativo (treinado offline) ──
    for _asset_load in _top_dt_assets:
        reversal_ai_map[_asset_load] = ReversalAI(_asset_load)
        _rai = reversal_ai_map[_asset_load]
        if _rai._ai1_ready:
            _rai.save_stats_to_disk()  # Exportar stats para dashboard
            log.info(paint(
                f"  ✅ NN {_asset_load}: {_rai._loaded_n_samples} amostras | "
                f"IA1={_rai._ai1_val:.1%} IA2={_rai._ai2_val:.1%}"
                + (f" IA3={_rai._ai3_val:.1%}" if _rai._ai3_ready else ""),
                C.G
            ))
        else:
            log.info(paint(f"  ⚠️ NN {_asset_load}: modelo não encontrado", C.Y))

    log.info(f"✅ Estratégia: SOMENTE Double Touch (Duplo Toque)")
    log.info(f"✅ Pool ranqueado de ativos: {_top_dt_assets}")
    log.info(f"✅ Corretora: {_BROKER_LABEL} ({BROKER_TYPE})")
    log.info(f"✅ Expiração: {EXP_FIXA} minuto(s)")
    log.info(f"🎯 Perfil DT Live: {DT_LIVE_PROFILE}")
    log.info(f"✅ Sinais: Detecção LOCAL (direto da corretora, sem delay)")
    log.info(f"✅ Memória: PERMANENTE — IA nunca perde conhecimento")
    log.info(f"✅ IA: ATIVA — acumula padrões DT a cada execução")
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
    _nn_ready_count = sum(1 for _rai in reversal_ai_map.values() if _rai._ai1_ready)
    if _nn_ready_count > 0:
        _nn_details = []
        for _nn_a, _nn_r in reversal_ai_map.items():
            if _nn_r._ai1_ready:
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

    # ═══ LOOP PRINCIPAL — SINAIS DO DASHBOARD H&S ═══
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

            # ═══ PRÉ-CACHE: Atualizar ativos/payout ANTES do :50 (evita delay no scan) ═══
            _cache_ativos_ts = 0  # forçar refresh a cada ciclo (fora da janela de scan)
            obter_top_ativos_otc(bx)

            # ── Inscrever ativos no stream de velas (só 1x, ANTES do :50) ──
            _target_ativo = _cache_ativos if _cache_ativos else _top_dt_assets
            if _target_ativo and not _stream_ready.is_set():
                for _sub_a in _target_ativo:
                    if _sub_a not in _stream_subscribed:
                        try:
                            bx.start_candles_stream(_sub_a, TF_M1, 10)
                            _stream_subscribed.add(_sub_a)
                            log.info(paint(f"  📡 Stream inscrito: {_sub_a}", C.G))
                        except Exception as _sub_e:
                            log.debug(f"  ⚠️ Stream falhou para {_sub_a}: {_sub_e}")
                if _stream_subscribed:
                    _stream_ready.set()
                    log.info(paint(f"  ✅ Live streaming ativo! {len(_stream_subscribed)} ativos inscritos", C.G))

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

            # ═══ H&S ENCONTRADO → TENTAR CADA CANDIDATO ═══
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

                # ── CONTRA SINAL: bloqueia direção oposta no mesmo ativo ──
                if _is_contra_signal(ativo, direcao):
                    print(f">>> IA: CONTRA SINAL bloqueou {ativo} {direcao}", flush=True)
                    continue

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
                    f"  🧠 IA H&S: {ativo} | prob={ia_prob:.2f} | "
                    f"geom={_pq:.2f} (log only) | amostras={ia_samples} | modo={'DT' if _is_dt_mode else 'HS'}",
                    C.B
                ))

                # ═══ DECISÃO 100% NN — sem guards fixos ═══
                _all_guards_ok = True
                _cur = None
                _head_price = setup["pattern"]["head"]["price"]
                _neckline = setup["pattern"].get("neckline", 0)
                _rs_price = setup["pattern"]["right_shoulder"]["price"]
                _target_price = setup["pattern"].get("target", 0)
                _ls_price = setup["pattern"]["left_shoulder"]["price"]

                # Usar preço do SCAN (momento exato da detecção, sem delay)
                _cur = setup.get("last_close")
                _guard_df = None
                try:
                    _guard_df = get_candles_df(bx, ativo, TF_M1, 60)
                    if _guard_df is not None and len(_guard_df) >= 1 and _cur is None:
                        _cur = float(_guard_df["close"].values[-1])
                except Exception as _pe:
                    log.debug(f"  get_candles_df falhou: {_pe}")

                if _cur is None:
                    log.warning(paint(f"  ⚠️ Preço atual indisponível — SKIP", C.Y))
                    _all_guards_ok = False

                if _is_dt_mode and _cur is not None:
                    _dt_guard_cfg = _get_dt_live_guard_params()
                    # ═══ DT: LOGGING RICO ═══
                    _dist_to_rs = abs(_cur - _rs_price)
                    _rs_to_neck = abs(_neckline - _rs_price)
                    _progress_pct = (_dist_to_rs / _rs_to_neck * 100) if _rs_to_neck > 0 else 0
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
                        f"wick={_wick_pct:.0f}% | força={_force_1t:.1f}ATR",
                        C.G if _progress_pct < 50 else C.Y
                    ))

                    print(
                        f">>> DT: {ativo} {direcao} | Preço={_cur:.6f} RS={_rs_price:.6f} "
                        f"Neck={_neckline:.6f} Target={_target_price:.6f} | "
                        f"geom={_pq:.2f} prob={ia_prob:.2f} wick={_wick_pct:.0f}%",
                        flush=True
                    )

                    # Pré-filtro no scan: exige toque recente + rejeição mínima + preço no lado correto.
                    _pos_ok = True
                    _dt_scan_guard = _validate_dt_entry_region(
                        _guard_df,
                        direcao,
                        _rs_price,
                        _neckline,
                        atr_val,
                        float(_cur),
                        **_dt_guard_cfg["scan"],
                    )

                    if not _dt_scan_guard["ok"]:
                        log.info(paint(
                            f"  🚫 TOQUE/REGIÃO: {_dt_scan_guard['reason']} → SKIP",
                            C.R
                        ))
                        print(f">>> TOQUE: {ativo} {direcao} {_dt_scan_guard['reason']} → SKIP", flush=True)
                        _pos_ok = False
                    else:
                        log.info(paint(
                            f"  ✅ TOQUE CONFIRMADO: RS atingido + rejeição | preço {_dt_scan_guard['progress_pct']:.0f}% do caminho RS→Neck",
                            C.G
                        ))

                    if not _pos_ok:
                        _all_guards_ok = False

                    # ═══ NN OBRIGATÓRIA — ÚNICA DECISÃO ═══
                    # Sem NN = sem entrada. NN 2/3 ≥ dinâmico = APROVADO.
                    _dyn_params = _get_session_params(_guard_df, atr_val)
                    _smart_exp = _dyn_params["exp_minutes"]
                    _nn_approved = False
                    _entry_guard_pred = setup.get("entry_guard_pre_pred")

                    if not _all_guards_ok:
                        pass  # posição inválida → pular NN/IA
                    else:
                        log.info(paint(
                            f"  ⚙️ SESSÃO DINÂMICA: NN_min={_dyn_params['nn_min_prob']:.0%} | "
                            f"EXP={_smart_exp}min | cooldown={_dyn_params['cooldown_sec']//60}min | perfil={_dyn_params.get('profile', 'balanced')}",
                            C.B
                        ))
                    if _all_guards_ok and _guard_df is not None and len(_guard_df) >= 50:
                        _nn_pred = None
                        try:
                            _g_H = _guard_df["high"].values
                            _g_L = _guard_df["low"].values
                            _g_C = _guard_df["close"].values
                            _g_O = _guard_df["open"].values
                            _g_n = len(_g_H)

                            _candles_ago_orig = pat_data.get("candles_ago", 0)
                            _rs_idx_local = _g_n - 1 - _candles_ago_orig
                            if _rs_idx_local < 0:
                                _rs_idx_local = 0

                            _win_start = max(0, _rs_idx_local - 55)
                            _win_end = min(_g_n, _rs_idx_local + 1)
                            _g_H_win = _g_H[_win_start:_win_end]
                            _g_L_win = _g_L[_win_start:_win_end]
                            _g_C_win = _g_C[_win_start:_win_end]
                            _g_O_win = _g_O[_win_start:_win_end]
                            _g_n_win = len(_g_H_win)

                            _atr_local_vals = [float(_g_H_win[k] - _g_L_win[k])
                                               for k in range(max(0, _g_n_win - 14), _g_n_win)]
                            _atr_local = float(np.mean(_atr_local_vals)) if _atr_local_vals else atr_val
                            if _atr_local <= 0:
                                _atr_local = atr_val

                            _pat_copy = dict(pat_data)
                            _pat_copy["candles_ago"] = max(0, _g_n_win - 1 - (_rs_idx_local - _win_start))

                            _nn_feats = extract_features(
                                _pat_copy, _g_H_win, _g_L_win, _g_C_win, _g_O_win, _g_n_win,
                                _atr_local, hs_stats, ativo)
                            if _nn_feats is not None:
                                _rai_ativo = reversal_ai_map.get(ativo)
                                if _rai_ativo is not None:
                                    _nn_pred = _rai_ativo.predict_dt(_nn_feats)
                        except Exception:
                            pass

                        if _nn_pred is not None:
                            _nn_votes = _nn_pred.get("votes_win", 0)
                            _nn_total = _nn_pred.get("total_voters", 2)
                            _nn_p1 = _nn_pred.get("p1", 0)
                            _nn_p2 = _nn_pred.get("p2", 0)
                            _nn_p3 = _nn_pred.get("p3")
                            _nn_p3_str = f" p3={_nn_p3:.2f}" if _nn_p3 is not None else ""
                            _nn_prob = _nn_pred.get("prob_win", 0)
                            _nn_score = _nn_pred.get("nn_score", _nn_prob)
                            _nn_penalty = _nn_pred.get("consensus_penalty", 0)
                            _NN_MIN_PROB = _dyn_params["nn_min_prob"]

                            _nn_approved = _nn_score >= _NN_MIN_PROB

                            if _nn_approved:
                                log.info(paint(
                                    f"  ✅ NN APROVADO: score={_nn_score:.0%} "
                                    f"(prob={_nn_prob:.0%} consenso=-{_nn_penalty:.2f}) | "
                                    f"p1={_nn_p1:.2f} p2={_nn_p2:.2f}{_nn_p3_str} | "
                                    f"EXP={_smart_exp}min",
                                    C.G
                                ))
                            else:
                                log.info(paint(
                                    f"  🚫 NN REJEITOU: score={_nn_score:.0%} < {_NN_MIN_PROB:.0%} "
                                    f"(prob={_nn_prob:.0%} consenso=-{_nn_penalty:.2f}) | "
                                    f"p1={_nn_p1:.2f} p2={_nn_p2:.2f}{_nn_p3_str}",
                                    C.R
                                ))
                                print(
                                    f">>> NN REJEITOU {ativo} {direcao}: score={_nn_score:.0%} "
                                    f"(p1={_nn_p1:.2f} p2={_nn_p2:.2f}{_nn_p3_str})",
                                    flush=True
                                )

                            if _nn_approved and _is_dt_mode and DT_LIVE_PROFILE != "standard":
                                _profile_check = _dt_profile_runtime_filter(
                                    _geo,
                                    _nn_pred,
                                    ia_prob,
                                    _wick_pct,
                                )
                                _profile_tag = _profile_check.get("profile", DT_LIVE_PROFILE).upper()
                                if not _profile_check["ok"]:
                                    _nn_approved = False
                                    log.info(paint(
                                        f"  🚫 {_profile_tag} FILTER: {_profile_check['reason']}",
                                        C.R
                                    ))
                                    print(
                                        f">>> {_profile_tag} FILTER bloqueou {ativo} {direcao}: {_profile_check['reason']}",
                                        flush=True
                                    )
                                else:
                                    log.info(paint(
                                        f"  ✅ {_profile_tag} FILTER: setup alinhado ao perfil ao vivo",
                                        C.G
                                    ))

                            if _nn_approved and _is_dt_mode and _ENTRY_GUARD_ENABLED:
                                _entry_guard_score, _entry_guard_pred = _estimate_entry_guard_score(
                                    ativo, pat_data, _guard_df, atr_val, hs_stats
                                )
                                if _entry_guard_pred is not None:
                                    if _entry_guard_pred["approved"]:
                                        log.info(paint(
                                            f"  ✅ ENTRY GUARD: prob={_entry_guard_pred['prob_now']:.0%} >= {_entry_guard_pred['threshold']:.0%} | "
                                            f"delay={_entry_guard_pred['delay_candles']} velas | acc={_entry_guard_pred['accuracy']:.1%}",
                                            C.G
                                        ))
                                    else:
                                        _nn_approved = False
                                        log.info(paint(
                                            f"  🚫 ENTRY GUARD: prob={_entry_guard_pred['prob_now']:.0%} < {_entry_guard_pred['threshold']:.0%} | "
                                            f"delay={_entry_guard_pred['delay_candles']} velas | acc={_entry_guard_pred['accuracy']:.1%}",
                                            C.R
                                        ))
                                        print(
                                            f">>> ENTRY GUARD bloqueou {ativo} {direcao}: prob={_entry_guard_pred['prob_now']:.0%} < {_entry_guard_pred['threshold']:.0%}",
                                            flush=True
                                        )
                                else:
                                    log.info(paint(
                                        f"  ⚠️ ENTRY GUARD: modelo indisponível para {ativo} — mantendo decisão NN",
                                        C.Y
                                    ))
                        else:
                            log.info(paint(
                                f"  🚫 NN: Modelo per-ativo não disponível para {ativo} — BLOQUEADO",
                                C.R
                            ))
                            print(f">>> NN BLOQUEOU {ativo}: modelo não disponível", flush=True)
                    elif _all_guards_ok:
                        log.info(paint(
                            f"  🚫 NN: Dados insuficientes (<50 candles) — BLOQUEADO",
                            C.R
                        ))

                    # ═══ IA 4 — GUARD GENERATIVA (IA WS Generativa) ═══
                    # Camada consultiva: analisa 30 velas + geometria + scores NN
                    # MODO ADVISORY: loga opinião mas NÃO bloqueia se NN ≥ 80%
                    if _nn_approved and _guard_df is not None and (not _is_dt_mode or ENABLE_GPT_DT_ADVISORY):
                        try:
                            _gpt_result = gpt_guard_check(
                                ativo=ativo,
                                direcao=direcao,
                                pat_data=pat_data,
                                H=_g_H, L=_g_L, C=_g_C, O=_g_O, n=_g_n,
                                atr_val=atr_val,
                                nn_pred=_nn_pred,
                                cur_price=float(_cur) if _cur else 0,
                            )
                            _gpt_approved = _gpt_result["approved"]
                            _gpt_conf = _gpt_result["confidence"]
                            _gpt_reason = _gpt_result["reason"]
                            _gpt_source = _gpt_result["source"]
                            _gpt_ms = _gpt_result["latency_ms"]

                            _gpt_exp = _gpt_result.get("exp_minutes", _smart_exp)
                            if _gpt_approved:
                                _smart_exp = _gpt_exp
                                log.info(paint(
                                    f"  ✅ IA Gen. APROVOU: conf={_gpt_conf}% | EXP={_gpt_exp}min | "
                                    f"{_gpt_reason} ({_gpt_source}, {_gpt_ms}ms)",
                                    C.G
                                ))
                            else:
                                # ADVISORY: apenas loga, NÃO bloqueia — NN já aprovou
                                log.info(paint(
                                    f"  ⚠️ IA Gen. discorda: conf={_gpt_conf}% | "
                                    f"{_gpt_reason} ({_gpt_source}, {_gpt_ms}ms) "
                                    f"— ENTRADA MANTIDA (NN={_nn_score:.0%})",
                                    C.Y
                                ))
                                print(
                                    f">>> IA Gen. discorda {ativo} {direcao}: "
                                    f"{_gpt_reason} — ENTRADA MANTIDA (NN aprovado)",
                                    flush=True
                                )
                        except Exception as _gpt_err:
                            log.warning(paint(
                                f"  ⚠️ IA Gen. erro: {_gpt_err} — mantendo decisão NN",
                                C.Y
                            ))
                    elif _nn_approved and _guard_df is not None and _is_dt_mode:
                        log.info(paint(
                            "  ⚡ IA Gen. pulada no DT para preservar entrada na virada :00",
                            C.B
                        ))

                    if not _nn_approved:
                        _all_guards_ok = False

                elif not _is_dt_mode and _cur is not None:
                    # ═══ H&S CLÁSSICO: Guards básicos ═══
                    if direcao == "PUT" and _cur >= _head_price:
                        log.info(paint(f"  🚫 GUARD HEAD: Preço ({_cur:.6f}) >= Cabeça ({_head_price:.6f})", C.Y))
                        _all_guards_ok = False
                    elif direcao == "CALL" and _cur <= _head_price:
                        log.info(paint(f"  🚫 GUARD HEAD: Preço ({_cur:.6f}) <= Cabeça ({_head_price:.6f})", C.Y))
                        _all_guards_ok = False

                    if _all_guards_ok:
                        if direcao == "PUT" and _cur > _rs_price:
                            log.info(paint(f"  🚫 BREAK GUARD: Preço ({_cur:.6f}) > Ombro D ({_rs_price:.6f})", C.Y))
                            _all_guards_ok = False
                        elif direcao == "CALL" and _cur < _rs_price:
                            log.info(paint(f"  🚫 BREAK GUARD: Preço ({_cur:.6f}) < Ombro D ({_rs_price:.6f})", C.Y))
                            _all_guards_ok = False

                    # IA filter somente para H&S
                    if _all_guards_ok and ia_prob < AI_MIN_PROB and ia_prob != 0.5:
                        log.info(paint(f"  🚫 IA BLOQUEOU: prob={ia_prob:.2f} < {AI_MIN_PROB}", C.Y))
                        _all_guards_ok = False

                    if _all_guards_ok:
                        log.info(paint(f"  ✅ GUARDS OK: Preço={_cur:.6f} | Head={_head_price:.6f} | RS={_rs_price:.6f}", C.G))

                if not _all_guards_ok:
                    print(f">>> IA: GUARD bloqueou {ativo} {direcao}", flush=True)
                    continue  # tentar próximo candidato

                # Calcular stake baseado no saldo (% da banca)
                stake = calcular_stake(bx)
                log.info(paint(
                    f"  💰 STAKE: ${stake:.2f} ({PERCENT_BANCA:.1f}% da banca)",
                    C.G
                ))

                # ═══ ENTRADA NA VIRADA :00 (CLOSE da vela = igual ao treino) ═══
                _is_early = setup.get("mode") == "early"
                _is_dt = setup.get("mode") == "double_touch"
                _candles_ago = pat_data.get("candles_ago", 99)
                _mode_label = "EARLY" if _is_early else ("DOUBLE_TOUCH" if _is_dt else "CLASSIC")
                log.info(paint(
                    f"  ⏱️ {_mode_label} MODE: Aguardando virada :00 para entrada "
                    f"(candles_ago={_candles_ago})",
                    C.G
                ))
                wait_candle_open()

                _entry_delay_sec = time.time() % 60
                if _is_dt_mode and _entry_delay_sec > MAX_ENTRY_DELAY_SEC:
                    log.info(paint(
                        f"  🚫 DT ATRASADO: entrada {_entry_delay_sec:.2f}s após a virada > {MAX_ENTRY_DELAY_SEC:.2f}s",
                        C.R
                    ))
                    print(
                        f">>> IA: FINAL CHECK cancelou {ativo} {direcao} — entrada atrasada {_entry_delay_sec:.2f}s",
                        flush=True
                    )
                    continue

                # ═══ VALIDAÇÃO FINAL: preço ainda na zona do padrão? ═══
                _entry_ok = True
                # Usar preço ATUAL (scan :50), não o close histórico do entry_idx
                _live_entry_price = float(_cur) if _cur else float(pat_data.get("entry_price", 0))
                # Tentar obter preço mais fresco (após virada :00)
                try:
                    _fresh_df = get_candles_df(bx, ativo, TF_M1, 3, min_len=1)
                    if _fresh_df is not None and len(_fresh_df) >= 1:
                        _live_entry_price = float(_fresh_df["close"].values[-1])
                except Exception:
                    pass

                if _live_entry_price and _rs_price > 0 and _is_dt_mode:
                    _closed_guard_df = None
                    try:
                        _closed_guard_df = get_last_closed_candles_df(bx, ativo, TF_M1, 6, min_len=3)
                    except Exception:
                        _closed_guard_df = None

                    _dt_final_guard = _validate_dt_entry_region(
                        _closed_guard_df,
                        direcao,
                        _rs_price,
                        _neckline,
                        atr_val,
                        float(_live_entry_price),
                        **_dt_guard_cfg["final"],
                    )
                    if not _dt_final_guard["ok"]:
                        log.info(paint(
                            f"  🚫 FINAL DT: candle fechado inválido — {_dt_final_guard['reason']}",
                            C.R
                        ))
                        _entry_ok = False
                    else:
                        log.info(paint(
                            f"  ✅ FINAL DT: candle fechado confirmou toque/rejeição | preço {_dt_final_guard['progress_pct']:.0f}% do caminho",
                            C.G
                        ))

                # Verificar se preço já ultrapassou neckline
                if not _GUARDS_DISABLED and _entry_ok and _neckline > 0 and _live_entry_price and not _is_dt_mode:
                    if direcao == "CALL" and _live_entry_price >= _neckline:
                        log.info(paint(
                            f"  🚫 FINAL CHECK: Preço ({_live_entry_price:.6f}) já acima da Neckline ({_neckline:.6f}) → CANCELADO",
                            C.Y
                        ))
                        _entry_ok = False
                    elif direcao == "PUT" and _live_entry_price <= _neckline:
                        log.info(paint(
                            f"  🚫 FINAL CHECK: Preço ({_live_entry_price:.6f}) já abaixo da Neckline ({_neckline:.6f}) → CANCELADO",
                            C.Y
                        ))
                        _entry_ok = False

                if not _entry_ok:
                    print(f">>> IA: FINAL CHECK cancelou {ativo} {direcao} — preço se moveu demais", flush=True)
                    continue  # tentar próximo candidato

                _nn_entry_data = None
                if _nn_pred is not None:
                    _nn_entry_data = {
                        "approved": True,
                        "p1": round(_nn_p1, 3),
                        "p2": round(_nn_p2, 3),
                        "p3": round(_nn_p3, 3) if _nn_p3 is not None else None,
                        "nn_score": round(_nn_score, 3),
                        "consensus_penalty": round(_nn_penalty, 3),
                    }
                _decision_id = f"{int(time.time() * 1000)}_{ativo}_{direcao}_{random.randint(1000, 9999)}"
                _decision_conf_pct = (_nn_score * 100) if _nn_pred is not None else (ia_prob * 100)
                _log_live_trade(ativo, direcao, None, _live_entry_price, stake,
                                confidence=_decision_conf_pct, status="entry",
                                nn_data=_nn_entry_data,
                                decision_id=_decision_id)

                # ═══ SALVAR DECISÃO COMPLETA para HTML viewer ═══
                _decision_geo = _extract_geometry(pat_data, atr_val) if pat_data else None
                _save_trade_decision({
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
                        "p1": round(_nn_p1, 4) if _nn_pred else None,
                        "p2": round(_nn_p2, 4) if _nn_pred else None,
                        "p3": round(_nn_p3, 4) if _nn_pred and _nn_p3 is not None else None,
                        "prob_win": round(_nn_prob, 4) if _nn_pred else None,
                        "nn_score": round(_nn_score, 4) if _nn_pred else None,
                        "consensus_penalty": round(_nn_penalty, 4) if _nn_pred else None,
                        "approved": _nn_approved,
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
                    },
                    # ── IA WS Generativa ──
                    "gpt": {
                        "approved": locals().get('_gpt_approved'),
                        "confidence": locals().get('_gpt_conf'),
                        "reason": locals().get('_gpt_reason'),
                        "exp_minutes": locals().get('_gpt_exp'),
                    },
                    # ── Contexto ──
                    "cur_price": round(float(_cur), 6) if _cur else None,
                    "atr": round(atr_val, 6),
                    "wick_pct": locals().get('_wick_pct', 0),
                })

                _use_exp = EXP_EARLY if _is_early else _smart_exp
                op = enviar_ordem(bx, ativo, direcao, stake, exp=_use_exp)
                if not op:
                    log.warning(paint(f"  ❌ Falha na ordem: {ativo}", C.R))
                    continue  # tentar próximo candidato

                op_type, op_id = op
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
                    f"NN={_nn_score:.0%} | Amostras={ia_samples}",
                    C.G if direcao == "CALL" else C.R
                ))
                print(f">>> IA: Entrada {ativo} {direcao} @{_live_entry_price or 0:.6f} stake={stake:.2f} nn={_nn_score:.2f}", flush=True)

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
