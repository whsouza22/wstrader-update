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

NUM_ATIVOS = int(os.getenv("WS_NUM_ATIVOS", "3"))  # TOP 3 ativos DT

# ── Expiração FIXA 3 minutos (H&S Cabeça e Ombro) ──
EXP_FIXA = 3
EXP_EARLY = 5  # delay=0: EXP=5 → 91.4% WR (melhor que EXP=3 → 89.4% para entrada antecipada)

# ── URL da base de treino no GitHub (auto-download semanal) ──
# O desenvolvedor sobe ws_ai_base_training.json toda semana
# Os clientes baixam automaticamente na inicialização
GITHUB_TRAINING_URL = os.getenv(
    "WS_TRAINING_URL",
    "https://raw.githubusercontent.com/user/wstrader/main/ws_ai_base_training.json"
)
BASE_TRAINING_LOCAL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ws_ai_base_training.json"
)

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
AI_MIN_PROB = 0.55  # CORRIGIDO: era 0.40 (permitia entradas com 40% prob = moeda)
HORARIO_INICIO_MIN = 90    # 1h30 da manhã (1*60 + 30)
HORARIO_FIM_MIN    = 1080  # 18h00 (18*60)
MAX_DIST_OMBRO_ATR = 0.5  # CORRIGIDO: era 1.0 (muito longe do ombro D = entrada ruim)
MAX_DIST_NECKLINE_ATR = 0.25  # Distância máx ALÉM da neckline — se preço já ultrapassou muito, é tarde demais

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
_DT_MEMORY_EXPIRY = 60 * 60  # 60 min — nível expira após 1 hora
_DT_MEMORY_TOL_MULT = 0.6    # tolerância = ATR * 0.6 para considerar "mesmo nível"


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
    """Grava um nível de toque na memória após entrada."""
    global _dt_level_memory
    if ativo not in _dt_level_memory:
        _dt_level_memory[ativo] = []
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
    min_spacing = 8
    max_spacing = 60
    min_depth = atr * 1.0
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
                })
        else:
            if n - 1 - idx1 < min_spacing or n - 1 - idx1 > max_spacing:
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
                })
        else:
            if n - 1 - idx1 < min_spacing or n - 1 - idx1 > max_spacing:
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
                })

    return patterns


def backtest_pattern(pat, C, O, H, L, n):
    """Verifica se o padrão H&S resultaria em WIN ou LOSS.
    Regra: entra no CLOSE da vela do ombro direito (delay=0).
    Para mode='early', a entrada real em LIVE também é ~delay=0
    (scan :05 + entrada imediata no turbo).
    Verifica o close EXP candles depois.
    Early: EXP_EARLY (5 min) — otimizado para delay=0 (91.4% WR).
    Classic: EXP_FIXA (3 min) — otimizado para delay=3 (58.7% WR).
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


# ══════════════════════════════════════════════════════════════
# BASE DE TREINO PRÉ-TREINADA (GitHub auto-download)
# ══════════════════════════════════════════════════════════════
def _download_training_base() -> Optional[dict]:
    """Tenta baixar base de treino mais recente do GitHub.
    Retorna dict com a base ou None se falhar."""
    try:
        import urllib.request
        import urllib.error

        log.info(paint("🌐 Verificando base de treino no GitHub...", C.B))
        req = urllib.request.Request(GITHUB_TRAINING_URL, headers={
            "User-Agent": "WS-Trader-IA/1.0",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        remote_version = data.get("version", "")
        if remote_version:
            log.info(paint(f"✅ Base de treino encontrada: versão {remote_version}", C.G))
            return data
    except urllib.error.HTTPError as e:
        log.debug(f"GitHub training base HTTP error: {e.code}")
    except Exception as e:
        log.debug(f"GitHub training base download failed: {e}")
    return None


def _load_or_download_training_base(hs_stats: dict) -> dict:
    """Carrega base de treino (local ou GitHub).
    Prioridade:
    1) Base local já existente ws_ai_base_training.json
    2) Download do GitHub
    3) Fallback: usa stats existentes
    
    A base NUNCA sobrescreve dados LIVE do cliente."""

    # ── Verificar versão local atual ──
    local_version = hs_stats.get("meta", {}).get("deep_train_version", "")

    # ── Tentar carregar base local (veio com o app ou download anterior) ──
    base_data = None
    if os.path.exists(BASE_TRAINING_LOCAL):
        try:
            with open(BASE_TRAINING_LOCAL, "r", encoding="utf-8") as f:
                base_data = json.load(f)
            base_version = base_data.get("version", "")
            log.info(paint(f"📂 Base local encontrada: versão {base_version}", C.G))
        except Exception:
            base_data = None

    # ── Tentar GitHub se base local não existe ou é antiga ──
    if base_data is None:
        remote = _download_training_base()
        if remote:
            base_data = remote
            # Salvar localmente para próxima vez
            try:
                with open(BASE_TRAINING_LOCAL, "w", encoding="utf-8") as f:
                    json.dump(remote, f, indent=2, ensure_ascii=False)
                log.info(paint("💾 Base salva localmente para uso offline", C.G))
            except Exception:
                pass
    elif base_data:
        # Base local existe — checar se GitHub tem versão mais nova
        remote = _download_training_base()
        if remote:
            remote_version = remote.get("version", "")
            base_version = base_data.get("version", "")
            if remote_version > base_version:
                log.info(paint(
                    f"🆕 Nova versão disponível: {remote_version} (local: {base_version})",
                    C.G
                ))
                base_data = remote
                try:
                    with open(BASE_TRAINING_LOCAL, "w", encoding="utf-8") as f:
                        json.dump(remote, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass

    if not base_data:
        log.info(paint("📂 Sem base de treino pré-treinada — usará treino local", C.Y))
        return hs_stats

    # ── MERGE: base pré-treinada + dados LIVE do cliente ──
    base_version = base_data.get("version", "unknown")
    
    # Se já carregou esta versão antes, pular
    if local_version == base_version:
        _n = hs_stats.get("meta", {}).get("total", 0)
        log.info(paint(f"📂 Base versão {base_version} já carregada ({_n} amostras) — usando memória existente", C.G))
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

    _total = new_stats["meta"].get("total", 0)
    _n_geo = len(new_stats.get("geometry_history", []))
    log.info(paint(
        f"✅ Base de treino aplicada! {_total} amostras + {len(_live_geo)} live | "
        f"Geometria: {_n_geo} padrões",
        C.G
    ))
    print(f">>> IA: Base pré-treinada v{base_version} carregada! {_total} amostras", flush=True)

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
        _lvl_num, _lvl_nome, _lvl_emoji = _get_ia_level(_n_total)
        log.info(paint(
            f"Backtest recente ({_hours_since:.1f}h atrás) — usando memória existente "
            f"({_n_total} amostras | Nível {_lvl_num}: {_lvl_emoji} {_lvl_nome})",
            C.G
        ))
        print(f">>> IA: Memória recente OK! {_n_total} amostras | Nível {_lvl_num} ({_lvl_nome})", flush=True)
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

        except Exception as e:
            log.debug(f"Erro treinando {ativo}: {e}")
            continue

    _n_total = hs_stats.get("meta", {}).get("total", 0)
    _n_geo = len(hs_stats.get("geometry_history", []))
    _n_geo_wins = sum(1 for g in hs_stats.get("geometry_history", []) if g.get("result") == 1)
    _lvl_num, _lvl_nome, _lvl_emoji = _get_ia_level(_n_total)
    wr = (total_wins / max(total_wins + total_losses, 1)) * 100

    log.info(paint("=" * 50, C.G))
    log.info(paint(f"🏋️ TREINO CONCLUÍDO — MEMÓRIA PERMANENTE!", C.G))
    log.info(paint(f"  📊 Sessão: {total_patterns} padrões | {total_wins}W / {total_losses}L | WR: {wr:.1f}%", C.G))
    log.info(paint(f"  🧠 IA TOTAL: {_n_total} amostras acumuladas | Nível {_lvl_num}: {_lvl_emoji} {_lvl_nome}", C.G))
    log.info(paint(f"  📐 IA Geométrica: {_n_geo} padrões aprendidos ({_n_geo_wins} wins)", C.G))
    log.info(paint(f"  💾 Memória NUNCA é apagada — quanto mais roda, mais aprende!", C.G))
    log.info(paint("=" * 50, C.G))

    print(f">>> IA: Treinada! {_n_total} amostras | Geometria: {_n_geo} padrões | Nível {_lvl_num} ({_lvl_nome}) | WR: {wr:.1f}%", flush=True)

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


def escolher_melhor_setup_local(bx, cooldown_map: dict, hs_stats: dict, early_only: bool = False):
    """Detecta Double Touch em TEMPO REAL — ativo único.
    Busca candles DIRETO da corretora. Sem dashboard, sem JSON, sem delay.
    Opera SOMENTE no ativo com maior WR no treino DT.

    Returns (best_trade, best_any)."""
    ativos = obter_top_ativos_otc(bx)
    if not ativos:
        log.warning(paint("⚠️ Nenhum ativo OTC disponível", C.Y))
        return None, None

    best_trade = None
    best_any = None
    _total_patterns = 0
    _scan_start = time.time()
    _dashboard_assets = {}  # acumula dados para o dashboard

    for ativo in ativos:
        # ── EARLY EXIT: Se já encontrou padrão bom E faltam <5s para :00, parar ──
        _elapsed_sec = int(time.time() % 60)
        if best_trade is not None and _elapsed_sec >= 55:
            log.info(paint(
                f"  ⚡ Early exit: padrão encontrado + segundo :{_elapsed_sec:02d} — parando scan",
                C.G
            ))
            break
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
        patterns = detect_double_touch(H, L, C_arr, O, pivot_highs, pivot_lows, atr, n, max_candles_ago=1)

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

                # ═══ FIX LIVE #1: Padrão deve ser vela FECHADA (não formando) ═══
                # No segundo :40 a última vela ainda está formando.
                # Se candles_ago=0 = vela atual = NÃO CONFIRMADO.
                # IGUAL AO DASHBOARD: só opera vela confirmada (candles_ago >= 1).
                # candles_ago=0 causou 3 LOSS seguidos — vela não confirmada.
                _allow_ago_0 = pat.get("mode") in ("early",)
                if pat["candles_ago"] < 1 and not _allow_ago_0:
                    log.info(paint(
                        f"  ⛔ SKIP: {ativo} candles_ago={pat['candles_ago']} "
                        f"(vela formando — não confirmado)",
                        C.Y
                    ))
                    continue

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

            # ═══ MEMÓRIA DT: Bloquear 3º toque no mesmo nível ═══
            _rs_price_check = pat.get("right_shoulder", {}).get("price", 0)
            if _is_dt_level_already_traded(ativo, _rs_price_check, direction, atr):
                log.info(paint(
                    f"  🚫 3º TOQUE BLOQUEADO: {ativo} {direction} RS={_rs_price_check:.6f} "
                    f"— nível já operado (memória DT)",
                    C.R
                ))
                continue

            # IA predict — IGUAL ao dashboard (usa fallback global)
            ia_prob = ai_predict_hs(ativo, pat, hs_stats)
            ia_n = hs_stats.get("arms", {}).get(f"{ativo}_{pat_type}_{mode}", {}).get("total", 0)

            # IA Geométrica — IGUAL ao dashboard (multiplica prob * quality)
            _pq, _ = ia_pattern_quality(pat, atr, hs_stats)
            if _pq < 1.0:
                ia_prob = round(ia_prob * _pq, 4)

            # Score: usa ia_prob ajustada pela geometria
            score = ia_prob

            setup = {
                "dir": direction,
                "type": pat_type,
                "mode": mode,
                "confidence": round(score * 100, 1),
                "pattern": pat,
                "last_close": float(C_arr[-1]),  # preço atual para guards (evita 2ª API call)
                "last_close_prev": float(C_arr[-2]) if n >= 2 else float(C_arr[-1]),
            }

            log.info(paint(
                f"  📊 H&S LOCAL: {ativo} | {pat_type} {direction} ({mode}) | "
                f"score={score:.2f} (geom={_pq:.2f}) | entry_idx={pat['entry_idx']} | n={n}",
                C.B
            ))

            if best_any is None or score > best_any[0]:
                best_any = (score, ativo, setup, atr)

            # Filtro: IA bloqueia se prob muito baixa
            # DT também precisa de prob mínima para evitar entradas aleatórias.
            _dt_min_prob = 0.65  # DT: mínimo 65% (mais rigoroso que antes)
            _min_check = _dt_min_prob if mode == "double_touch" else AI_MIN_PROB
            if ia_prob < _min_check and ia_prob != 0.5:
                log.info(paint(
                    f"  🚫 IA PROB BAIXA: {ativo} {direction} prob={ia_prob:.2f} < {_min_check}",
                    C.Y
                ))
                continue

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

    return best_trade, best_any


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
        neck_align = abs(v1 - v2) / atr_val if atr_val > 0 else 0
        # Features extras da análise profunda (5000 velas)
        pL = pat["left_shoulder"]["price"]
        pR = pat["right_shoulder"]["price"]
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
        features = ["span", "symmetry", "depth_ratio", "neck_align"]
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
_top_dt_assets: List[str] = []  # TOP N ativos DT (N definido por benchmark)


def _pick_top_dt_assets(hs_stats: dict, n_top: int = 3) -> List[str]:
    """Analisa IA stats (arms) e retorna os TOP N ativos treinados com maior WR.
    Exige mínimo de 30 trades para confiança estatística.
    Fallback: lista fixa de ativos OTC populares."""
    from collections import defaultdict
    ativo_agg = defaultdict(lambda: {"wins": 0, "total": 0})
    for key, val in hs_stats.get("arms", {}).items():
        # Chaves: ATIVO_HEAD_SHOULDERS_classic, ATIVO_INV_HEAD_SHOULDERS_classic, etc.
        parts = key.split("_HEAD_SHOULDERS")
        if len(parts) < 2:
            parts = key.split("_INV_HEAD_SHOULDERS")
        if len(parts) < 2:
            continue
        ativo = parts[0]
        if ativo.endswith("_INV"):
            ativo = ativo[:-4]
        ativo_agg[ativo]["wins"] += val.get("wins", 0)
        ativo_agg[ativo]["total"] += val.get("total", 0)

    # Ordenar por WR (mínimo 30 trades para confiança)
    ranked = []
    for ativo, v in ativo_agg.items():
        if v["total"] < 30:
            continue
        wr = v["wins"] / v["total"]
        ranked.append((wr, v["total"], ativo))
    ranked.sort(key=lambda x: (-x[0], -x[1]))  # WR desc, volume desc

    top = [a for _, _, a in ranked[:n_top]]

    if not top:
        top = ["EURNZD-OTC", "GBPCHF-OTC", "EURAUD-OTC", "EURUSD-OTC", "GBPUSD-OTC"][:n_top]
        log.info(paint(f"🎯 TOP {n_top} ASSETS (fallback): {top}", C.Y))
    else:
        for i, a in enumerate(top):
            wr = ativo_agg[a]["wins"] / ativo_agg[a]["total"] * 100
            n_trades = ativo_agg[a]["total"]
            log.info(paint(
                f"🎯 ASSET #{i+1}: {a} (WR={wr:.1f}% | {n_trades} trades)",
                C.G
            ))
    return top


def obter_top_ativos_otc(bx: BrokerAPI) -> List[str]:
    global _cache_ativos, _cache_ativos_ts, _top_dt_assets
    # ── MODO MULTI-ASSET: opera nos TOP 3 pares do treino DT ──
    if _cache_ativos:
        return _cache_ativos

    if not _top_dt_assets:
        _top_dt_assets = ["EURNZD-OTC", "GBPCHF-OTC", "EURAUD-OTC"]

    # Verificar quais estão abertos na corretora
    targets = list(_top_dt_assets)
    try:
        dados = safe_call(bx, bx.get_all_open_time)
        turbo = dados.get("turbo", {})
        abertos = [a for a in targets if a in turbo and turbo[a].get("open", False)]
        if not abertos:
            log.warning(paint(f"⚠️ Nenhum dos TOP assets aberto — buscando fallback", C.Y))
            _broker_key = BROKER_TYPE.replace("iq_option", "iq")
            fixed = FIXED_ASSETS.get(_broker_key, FIXED_ASSETS.get("bullex", []))
            abertos = [a for a in fixed if a in turbo and turbo[a].get("open", False)][:3]
        targets = abertos if abertos else targets
    except Exception:
        pass

    # Verificar payouts
    try:
        all_profit = safe_call(bx, bx.get_all_profit)
        for t in targets:
            profit = all_profit.get(t, {}).get("turbo", 0)
            payout = int(profit * 100) if profit else 0
            if payout < PAYOUT_MINIMO:
                log.warning(paint(f"⚠️ {t} payout={payout}% (mín={PAYOUT_MINIMO}%)", C.Y))
            else:
                log.info(paint(f"✅ {t} payout={payout}% OK", C.G))
    except Exception:
        pass

    _cache_ativos = targets[:3]
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
                    confidence: float = 0.0, status: str = "entry"):
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
    }
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
                if trades[i].get("ativo") == ativo and trades[i].get("status") == "entry":
                    trades[i]["status"] = status
                    trades[i]["resultado"] = resultado
                    trades[i]["ts"] = record["ts"]
                    trades[i]["time"] = record["time"]
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
        _lvl_num, _lvl_nome, _lvl_emoji = _get_ia_level(_n_after_base)
        log.info(paint(
            f"✅ Base pré-treinada OK! {_n_after_base} amostras | "
            f"Nível {_lvl_num}: {_lvl_emoji} {_lvl_nome} — PULANDO treino local",
            C.G
        ))
        print(f">>> IA: Base pré-treinada! {_n_after_base} amostras | Nível {_lvl_num} ({_lvl_nome})", flush=True)
    else:
        log.info(paint("🏋️ Sem base pré-treinada — treinando IA localmente...", C.B))
        hs_stats = _train_ia_from_history(bx, hs_stats)

    log.info("=" * 60)
    log.info(paint(f"🚀 WS TRADER — Double Touch / Ativo Único ({_BROKER_LABEL})", C.G))
    log.info("=" * 60)

    # ── SELECIONAR MELHOR ATIVO DT (baseado no treino + benchmark) ──
    global _top_dt_assets
    _top_dt_assets = _pick_top_dt_assets(hs_stats, n_top=3)

    log.info(f"✅ Estratégia: SOMENTE Double Touch (Duplo Toque)")
    log.info(f"✅ TOP Ativos: {_top_dt_assets}")
    log.info(f"✅ Corretora: {_BROKER_LABEL} ({BROKER_TYPE})")
    log.info(f"✅ Expiração: {EXP_FIXA} minuto(s)")
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
    _last_trade_time = 0.0
    _last_entry_key = _load_last_entry_key()  # Dedup persistente: sobrevive a reinícios

    # ── Carregar memória de níveis DT (impede 3º toque) ──
    global _dt_level_memory
    _dt_level_memory = _load_dt_level_memory()
    if _dt_level_memory:
        _n_mem = sum(len(v) for v in _dt_level_memory.values())
        log.info(paint(f"📋 Memória DT carregada: {_n_mem} nível(is) em {len(_dt_level_memory)} ativo(s)", C.B))

    print(f"\n>>> IA: Iniciado | Exp: {EXP_FIXA}min | Sinais: Detecção LOCAL", flush=True)

    # Exportar stats iniciais para o UI
    reversal_ai.save_stats_to_disk()

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

            # ═══ SCAN DOUBLE TOUCH — ATIVO ÚNICO no segundo :50 ═══
            wait_until_second(50)

            # ── Inscrever ativos no stream de velas (só 1x, no loop principal) ──
            # start_candles_stream é bloqueante (~10s por ativo), mas aqui é seguro
            # pois roda ANTES do scan (sem conflito de WebSocket).
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

            log.info(paint(
                f"\n🔍 Scan DT em {_target_ativo} ({len(_target_ativo) if isinstance(_target_ativo, list) else 1} ativos, segundo :50)...",
                C.B
            ))
            best_trade, best_any = escolher_melhor_setup_local(bx, cooldown, hs_stats)

            if not best_trade:
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

            # ═══ H&S ENCONTRADO → ENTRAR ═══
            sc, ativo, setup, atr_val = best_trade
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

            # ── TIME DEDUP: Bloqueia qualquer entrada por 2 min após trade ──
            # Com 3 ativos, 2 min é suficiente para evitar overtrading.
            _secs_since_trade = time.time() - _last_trade_time
            _min_trade_interval = 2 * 60  # 2 minutos — rápido mas seguro
            if _last_trade_time > 0 and _secs_since_trade < _min_trade_interval:
                _wait_remain = int(_min_trade_interval - _secs_since_trade)
                log.info(paint(
                    f"  🚫 TIME DEDUP: Último trade há {int(_secs_since_trade)}s "
                    f"(mín={_min_trade_interval}s) — aguardar {_wait_remain}s",
                    C.Y
                ))
                s = seconds_to_next(TF_M1)
                time.sleep(min(s + 1, 30))
                continue

            if _entry_key == _last_entry_key or _entry_key_simple == _last_key_simple:
                log.info(paint(
                    f"  🚫 DEDUP: Mesmo padrão/direção já operado ({ativo} {direcao}) — SKIP",
                    C.Y
                ))
                s = seconds_to_next(TF_M1)
                time.sleep(min(s + 1, 30))
                continue

            # ═══ ANÁLISE COMPLETA: IA + Geometria + Posição ═══
            pat_data = setup.get("pattern", setup)
            ia_prob = ai_predict_hs(ativo, pat_data, hs_stats)
            _arm_key = f"{ativo}_{pat_type}_{setup.get('mode', 'classic')}"
            ia_samples = hs_stats.get("arms", {}).get(_arm_key, {}).get("total", 0)

            _is_dt_mode = setup.get("mode") == "double_touch"

            # ── IA GEOMÉTRICA: aprende perfil dos WINs (IGUAL dashboard) ──
            _pq, _pq_motivos = ia_pattern_quality(pat_data, atr_val, hs_stats)
            _ia_prob_orig = ia_prob
            if _pq < 1.0:
                ia_prob = round(ia_prob * _pq, 4)

            log.info(paint(
                f"  🧠 IA H&S: {ativo} | prob={_ia_prob_orig:.2f}→{ia_prob:.2f} | "
                f"geom={_pq:.2f} | amostras={ia_samples} | modo={'DT' if _is_dt_mode else 'HS'}",
                C.B
            ))

            # ═══ GUARDS + ANÁLISE DE POSIÇÃO ═══
            _all_guards_ok = True
            _cur = None
            _head_price = setup["pattern"]["head"]["price"]
            _neckline = setup["pattern"].get("neckline", 0)
            _rs_price = setup["pattern"]["right_shoulder"]["price"]
            _target_price = setup["pattern"].get("target", 0)
            _ls_price = setup["pattern"]["left_shoulder"]["price"]

            # Buscar preço atual (usa dados do scan como fallback)
            try:
                _guard_df = get_candles_df(bx, ativo, TF_M1, 60)
                if _guard_df is not None and len(_guard_df) >= 1:
                    _cur = float(_guard_df["close"].values[-1])
            except Exception as _pe:
                log.debug(f"  get_candles_df falhou: {_pe}")

            # Fallback: usar dados já buscados no scan
            if _cur is None:
                _cur = setup.get("last_close")
                if _cur:
                    log.debug(f"  Usando preço do scan: {_cur:.6f}")

            if _cur is None:
                log.warning(paint(f"  ⚠️ Preço atual indisponível — SKIP", C.Y))
                _all_guards_ok = False

            if _is_dt_mode and _cur is not None:
                # ═══ DT: LOGGING RICO — IGUAL AO DASHBOARD ═══
                # Mostrar RS, Neckline, Target, posição do preço
                _dist_to_rs = abs(_cur - _rs_price)
                _rs_to_neck = abs(_neckline - _rs_price)
                _rs_to_target = abs(_target_price - _rs_price) if _target_price > 0 else 0
                _progress_pct = (_dist_to_rs / _rs_to_neck * 100) if _rs_to_neck > 0 else 0
                _geo = _extract_geometry(pat_data, atr_val)
                _geo_str = ""
                if _geo:
                    _geo_str = (f"span={_geo['span']} sym={_geo['symmetry']:.2f} "
                               f"depth={_geo['depth_ratio']:.2f} neck={_geo['neck_align']:.3f}")

                # Análise de rejeição/Volume: verificar força da rejeição no 2º toque
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

                # Direção do movimento após rejeição
                _move_dir = "neutro"
                if direcao == "PUT" and _cur < _rs_price:
                    _move_dir = "✅ descendo (correto)"
                elif direcao == "PUT" and _cur >= _rs_price:
                    _move_dir = "⚠️ acima do RS"
                elif direcao == "CALL" and _cur > _rs_price:
                    _move_dir = "✅ subindo (correto)"
                elif direcao == "CALL" and _cur <= _rs_price:
                    _move_dir = "⚠️ abaixo do RS"

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
                    f"wick={_wick_pct:.0f}% | mov={_move_dir}",
                    C.G if _progress_pct < 50 else C.Y
                ))
                print(
                    f">>> DT: {ativo} {direcao} | Preço={_cur:.6f} RS={_rs_price:.6f} "
                    f"Neck={_neckline:.6f} Target={_target_price:.6f} | "
                    f"geom={_pq:.2f} prob={ia_prob:.2f} wick={_wick_pct:.0f}%",
                    flush=True
                )

                # ═══ IA STRUCTURE GUARD: Validação inteligente baseada em 89K samples ═══
                # A IA SABE quais padrões ganham. Estas regras vêm da análise estatística:
                # - Posição do preço: se já andou >35% do RS→Neck, entrada é "tarde"
                # - Depth ratio: <4.0 → WR=90%+, ≥4.0 → WR=77% (dados dos 200 geos)
                # - Symmetry 0.45-0.60: WR=72.7% (pior faixa, 22 samples)
                # - Movimento contrário: preço deve se afastar do RS (confirmando rejeição)
                _reject_reasons = []

                # GUARD IA 1: POSIÇÃO DO PREÇO — entrada deve ser PERTO do RS (toque)
                # Double Bottom CALL: preço subiu do RS (suporte). Se >25% = tarde.
                # Double Top PUT: preço desceu do RS (resistência). Se >25% = tarde.
                _max_progress = 10  # máximo 10% do caminho RS→Neck
                if _progress_pct > _max_progress:
                    _reject_reasons.append(
                        f"POSIÇÃO: {_progress_pct:.0f}% > {_max_progress}% (preço já longe do toque)")
                    log.info(paint(
                        f"  🚫 IA GUARD POSIÇÃO: Preço já andou {_progress_pct:.0f}% "
                        f"do RS→Neck (máx={_max_progress}%) — entrada TARDE demais",
                        C.Y
                    ))

                # GUARD IA 2: DEPTH RATIO — padrões profundos demais têm WR baixo
                # depth < 4.0 → WR 90%+ (n=90) | depth ≥ 6.0 → WR <80% (n=110)
                # depth ≥ 10.0 → padrão absurdo (span longo demais) → BLOQUEIO CRÍTICO
                if _geo and _geo.get("depth_ratio", 0) >= 6.0:
                    _dr = _geo["depth_ratio"]
                    if _dr >= 10.0:
                        # Depth absurdo = padrão inválido, tratar como CRÍTICO
                        _reject_reasons.append(f"POSIÇÃO: depth={_dr:.1f} ≥ 10 (padrão inválido)")
                        log.info(paint(
                            f"  🚫 IA GUARD DEPTH CRÍTICO: depth_ratio={_dr:.1f} ≥ 10.0 "
                            f"— padrão absurdamente profundo, BLOQUEIO",
                            C.R
                        ))
                    else:
                        _reject_reasons.append(f"DEPTH: {_dr:.1f} ≥ 6.0 (WR ~77%)")
                        log.info(paint(
                            f"  ⚠️ IA GUARD DEPTH: depth_ratio={_dr:.1f} ≥ 6.0 "
                            f"(89K samples: WR cai para ~77%)",
                            C.Y
                        ))

                # GUARD IA 3: SYMMETRY ZONE — faixa 0.45-0.60 é a PIOR (WR=72.7%)
                if _geo:
                    _sym = _geo.get("symmetry", 0)
                    if 0.45 <= _sym <= 0.60:
                        _reject_reasons.append(f"SYM: {_sym:.2f} na zona 0.45-0.60 (WR=72.7%)")
                        log.info(paint(
                            f"  ⚠️ IA GUARD SYM: symmetry={_sym:.2f} na zona perigosa "
                            f"(0.45-0.60 → WR=72.7% nos 89K samples)",
                            C.Y
                        ))

                # GUARD IA 4: MOVIMENTO — preço deve ter VOLTADO na direção certa
                # Confirma que a rejeição gerou movimento. Se preço está no lado errado, skip.
                _wrong_side = False
                if direcao == "PUT" and _cur > _rs_price:
                    _wrong_side = True
                elif direcao == "CALL" and _cur < _rs_price:
                    _wrong_side = True
                if _wrong_side:
                    _reject_reasons.append(f"LADO ERRADO: preço não saiu da região")
                    log.info(paint(
                        f"  🚫 IA GUARD LADO: Preço={_cur:.6f} ainda no lado "
                        f"ERRADO do RS={_rs_price:.6f} — rejeição não confirmada",
                        C.Y
                    ))

                # GUARD IA 5: PROXIMIDADE AO ALVO — preço já perto do target = movimento esgotado
                # Se o preço já andou a maior parte do caminho RS→Target, entrar é arriscado:
                # o movimento está quase completo e pode reverter (bounce no suporte/resistência).
                # Para PUT: preço perto do suporte (target) → pode voltar a subir.
                # Para CALL: preço perto da resistência (target) → pode voltar a cair.
                _target_check = _target_price if _target_price > 0 else _neckline
                if _target_check > 0 and _rs_to_neck > 0:
                    _dist_to_target = abs(_cur - _target_check)
                    _target_proximity_pct = (1 - _dist_to_target / _rs_to_neck) * 100
                    _max_target_proximity = 60  # se preço já percorreu >60% rumo ao alvo → BLOQUEAR
                    if _target_proximity_pct > _max_target_proximity:
                        _reject_reasons.append(
                            f"POSIÇÃO: preço a {_target_proximity_pct:.0f}% do alvo "
                            f"(máx={_max_target_proximity}%) — movimento esgotado")
                        log.info(paint(
                            f"  🚫 IA GUARD ALVO: Preço={_cur:.6f} já a {_target_proximity_pct:.0f}% "
                            f"do Target={_target_check:.6f} (máx={_max_target_proximity}%) "
                            f"— movimento quase esgotado, risco de bounce!",
                            C.R
                        ))

                # DECISÃO: 1 razão crítica = BLOCK; 2+ warnings = BLOCK
                _n_critical = sum(1 for r in _reject_reasons if "POSIÇÃO" in r or "LADO" in r)
                _n_warnings = len(_reject_reasons) - _n_critical

                if _n_critical > 0 or _n_warnings >= 2:
                    log.info(paint(
                        f"  ❌ IA REJEITOU: {len(_reject_reasons)} problema(s) — "
                        + " | ".join(_reject_reasons),
                        C.R
                    ))
                    print(
                        f">>> IA REJEITOU {ativo} {direcao}: "
                        + " | ".join(_reject_reasons),
                        flush=True
                    )
                    _all_guards_ok = False
                elif _reject_reasons:
                    # 1 warning sozinho: logar mas permitir entrada
                    log.info(paint(
                        f"  ⚠️ IA ATENÇÃO: {_reject_reasons[0]} (1 warning — PERMITIDO)",
                        C.Y
                    ))
                else:
                    log.info(paint(
                        f"  ✅ IA APROVADO: padrão com boa estrutura | "
                        f"pos={_progress_pct:.0f}% depth={_geo.get('depth_ratio', 0):.1f} "
                        f"sym={_geo.get('symmetry', 0):.2f} mov={_move_dir}",
                        C.G
                    ))

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
                s = seconds_to_next(TF_M1)
                time.sleep(min(s + 1, 30))
                continue

            # Calcular stake
            stake = calcular_stake(bx)

            # ═══ ENTRADA: EARLY/DT/FAST = imediata, CLASSIC lento = na virada :00 ═══
            _is_early = setup.get("mode") == "early"
            _is_dt = setup.get("mode") == "double_touch"
            _candles_ago = pat_data.get("candles_ago", 99)
            _fast_classic = (not _is_early) and (not _is_dt) and _candles_ago <= 1
            if _is_early or _is_dt or _fast_classic:
                _mode_label = "EARLY" if _is_early else ("DOUBLE_TOUCH" if _is_dt else "FAST CLASSIC")
                log.info(paint(
                    f"  ⚡ {_mode_label} MODE: Entrada IMEDIATA no :50 "
                    f"(candles_ago={_candles_ago}, delay≈{_candles_ago} velas)",
                    C.G
                ))
                # Não espera :00 — entra AGORA para reduzir delay
            else:
                # candles_ago >= 2: espera :00 para entrar
                wait_candle_open()

            # ═══ VALIDAÇÃO FINAL: preço ainda na zona do padrão? ═══
            # Para DT/EARLY: reusar preço dos guards (acabou de buscar, < 2s atrás)
            # Para CLASSIC: buscar preço novo (wait_candle_open pode ter levado 20s+)
            _entry_ok = True
            _live_entry_price = None
            try:
                if (_is_early or _is_dt or _fast_classic) and _cur is not None:
                    _live_entry_price = _cur  # Reusar preço do guard (economia de ~1s)
                else:
                    _final_df = get_candles_df(bx, ativo, TF_M1, 2)
                    if _final_df is not None and len(_final_df) >= 1:
                        _live_entry_price = float(_final_df["close"].values[-1])

                if _live_entry_price is None:
                    _live_entry_price = float(pat_data.get("entry_price", 0))

                    # Verificar se preço já ultrapassou neckline
                    if _neckline > 0:
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
            except Exception as _fe:
                log.warning(paint(f"  ⚠️ FINAL CHECK: Erro ({_fe}) — entrando mesmo assim", C.Y))

            if not _entry_ok:
                print(f">>> IA: FINAL CHECK cancelou {ativo} {direcao} — preço se moveu demais", flush=True)
                s = seconds_to_next(TF_M1)
                time.sleep(min(s + 1, 30))
                continue

            _log_live_trade(ativo, direcao, None, _live_entry_price, stake,
                            confidence=ia_prob * 100, status="entry")

            _use_exp = EXP_EARLY if _is_early else EXP_FIXA
            op = enviar_ordem(bx, ativo, direcao, stake, exp=_use_exp)
            if not op:
                log.warning(paint(f"  ❌ Falha na ordem: {ativo}", C.R))
                s = seconds_to_next(TF_M1)
                time.sleep(min(s + 1, 30))
                continue

            op_type, op_id = op
            _last_entry_key = _entry_key  # Marcar padrão como operado
            _last_trade_time = time.time()  # Marcar tempo do trade para TIME DEDUP
            _save_last_entry_key(_entry_key)  # Persistir em disco

            # ═══ MEMÓRIA DT: Gravar nível do toque para impedir 3º toque ═══
            if _is_dt:
                _touch_level = _rs_price  # nível do Right Shoulder (= zona do toque 2)
                _memorize_dt_level(ativo, _touch_level, direcao)
            log.info(paint(
                f"  ✅ ENTRADA: {ativo} {direcao} @ {_live_entry_price or 0:.6f} | Stake={stake:.2f} | "
                f"Tipo={op_type} | EXP={_use_exp}min | Modo={'EARLY' if _is_early else ('DT' if _is_dt else ('FAST' if _fast_classic else 'CLASSIC'))} | "
                f"IA={ia_prob:.0%} | Amostras={ia_samples}",
                C.G if direcao == "CALL" else C.R
            ))
            print(f">>> IA: Entrada {ativo} {direcao} @{_live_entry_price or 0:.6f} stake={stake:.2f} prob={ia_prob:.2f}", flush=True)

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
