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
import socket
import subprocess
import urllib.request

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
PAYOUT_MINIMO = int(os.getenv("WS_PAYOUT_MIN", "80"))
NUM_ATIVOS = int(os.getenv("WS_NUM_ATIVOS", "20"))
PAYOUT_REFRESH_SEC = int(os.getenv("WS_PAYOUT_REFRESH", "180"))

# ── Expiração FIXA 3 minutos (H&S Cabeça e Ombro) ──
EXP_FIXA = 3

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
AI_MIN_PROB = 0.40
HORARIO_INICIO_MIN = 90    # 1h30 da manhã (1*60 + 30)
HORARIO_FIM_MIN    = 1080  # 18h00 (18*60)
MAX_DIST_OMBRO_ATR = 0.5  # Distância máx do OmbroD em ATR — se preço já se afastou demais, não entrar
MAX_DIST_NECKLINE_ATR = 0.25  # Distância máx ALÉM da neckline — se preço já ultrapassou muito, é tarde demais

# ── Ativos fixos — melhores ativos (OTC + REAL com volume) ──
# Ranking: EURJPY-OTC 56.7% | AUDCAD-OTC 56.1% | EURGBP 55.0%
# EURUSD 50.8% | USDCHF 50.7% | EURGBP-OTC 50.0% | EURJPY 50.0%
FIXED_ASSETS = {
    "iq": [
        "EURJPY-OTC", "AUDCAD-OTC", "EURGBP-OTC",
        "EURGBP", "EURUSD", "USDCHF", "EURJPY",
    ],
    "bullex": [
        "EURJPY-OTC", "AUDCAD-OTC", "EURGBP-OTC",
        "EURGBP", "EURUSD", "USDCHF", "EURJPY",
    ],
    "casatrader": [
        "EURJPY-OTC", "AUDCAD-OTC", "EURGBP-OTC",
        "EURGBP", "EURUSD", "USDCHF", "EURJPY",
    ],
}

# ── Diretórios ──
_broker_suffix = BROKER_TYPE.replace("iq_option", "iq")
_user_data_dir = os.path.join(os.path.expanduser("~"), ".wstrader")
os.makedirs(_user_data_dir, exist_ok=True)

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
# CONTROLE DE RETRAIN SEMANAL (compartilhado entre corretoras)
# ═══════════════════════════════════════════════════════════════
_TRAIN_CONTROL_FILE = os.path.join(os.path.expanduser("~"), ".wstrader", "hs_bot_train_control.json")


def _need_retrain_bot():
    """Verifica se precisa limpar hs_stats (nova semana ISO)."""
    try:
        if os.path.exists(_TRAIN_CONTROL_FILE):
            with open(_TRAIN_CONTROL_FILE, "r") as f:
                ctrl = json.load(f)
            now = datetime.now().isocalendar()
            if ctrl.get("iso_week") == now[1] and ctrl.get("iso_year") == now[0]:
                return False
    except Exception:
        pass
    return True


def _save_retrain_control():
    """Marca semana atual como treinada."""
    try:
        os.makedirs(os.path.dirname(_TRAIN_CONTROL_FILE), exist_ok=True)
        now = datetime.now()
        iso = now.isocalendar()
        with open(_TRAIN_CONTROL_FILE, "w") as f:
            json.dump({"iso_year": iso[0], "iso_week": iso[1], "date": now.isoformat()}, f)
        log.info(paint(f"[RETRAIN] Controle salvo: semana {iso[1]}/{iso[0]}", C.G))
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
# DASHBOARD — INTEGRAÇÃO (fonte única de sinais H&S)
# ═══════════════════════════════════════════════════════════════
DASHBOARD_URL = "http://localhost:8899/api/data"
DASHBOARD_PORT = 8899


def _is_dashboard_running():
    """Verifica se o dashboard está rodando na porta 8899."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect(("127.0.0.1", DASHBOARD_PORT))
        s.close()
        return True
    except Exception:
        return False


def _ensure_dashboard_running():
    """Inicia o dashboard_hs_ia.py automaticamente se não estiver rodando."""
    if _is_dashboard_running():
        log.info(paint("✅ Dashboard IA H&S já está rodando na porta 8899", C.G))
        return True

    log.info(paint("🚀 Iniciando Dashboard IA H&S automaticamente...", C.B))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_script = os.path.join(script_dir, "dashboard_hs_ia.py")

    if not os.path.exists(dashboard_script):
        log.warning(paint("⚠️ dashboard_hs_ia.py não encontrado!", C.Y))
        return False

    try:
        # Herda BROKER_TYPE para que o dashboard use a mesma corretora
        env = os.environ.copy()
        subprocess.Popen(
            [sys.executable, dashboard_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        log.info(paint("⏳ Aguardando dashboard conectar e fazer primeiro scan...", C.B))
        # Espera até 60s para o dashboard subir e fazer o primeiro scan
        for i in range(60):
            time.sleep(1)
            if _is_dashboard_running():
                # Dashboard subiu, agora espera o primeiro scan completar
                time.sleep(10)
                log.info(paint("✅ Dashboard IA H&S iniciado com sucesso!", C.G))
                return True
        log.warning(paint("⚠️ Dashboard não iniciou a tempo (60s)", C.Y))
        return False
    except Exception as e:
        log.warning(paint(f"⚠️ Erro ao iniciar dashboard: {e}", C.Y))
        return False


def _fetch_dashboard_signals():
    """Busca sinais LIVE do dashboard IA H&S via API."""
    try:
        req = urllib.request.urlopen(DASHBOARD_URL, timeout=5)
        data = json.loads(req.read().decode("utf-8"))
        signals = data.get("live_signals", [])
        summary = data.get("summary", {})
        scan_count = data.get("scan_count", 0)
        return signals, summary, scan_count
    except Exception as e:
        log.warning(paint(f"⚠️ Erro ao acessar dashboard: {e}", C.Y))
        return [], {}, 0


def escolher_melhor_setup_dashboard(cooldown: dict):
    """Busca sinais do dashboard e seleciona o melhor setup H&S.
    SUBSTITUI a detecção local — fonte única de sinais é o dashboard.
    Returns (best_trade, best_any) no mesmo formato do antigo escolher_melhor_setup."""
    signals, summary, scan_count = _fetch_dashboard_signals()

    if not signals:
        return None, None

    best_trade = None
    best_any = None

    for sig in signals:
        ativo = sig.get("ativo", "")
        direction = sig.get("direction", "")
        pat_type = sig.get("type", "HS")
        mode = sig.get("mode", "classic")
        ia_prob = sig.get("ia_prob", 0.5)
        head_price = sig.get("head_price", 0)
        rs_price = sig.get("rs_price", 0)
        neckline = sig.get("neckline", 0)
        entry_pending = sig.get("entry_pending", True)
        scan_ts = sig.get("scan_ts", 0)
        target = sig.get("target", 0)
        stop = sig.get("stop", 0)

        if not ativo or not direction:
            continue

        # ── SOMENTE sinais com entrada pendente (vela de entrada AINDA NÃO existe) ──
        if not entry_pending:
            log.info(paint(f"  ⏭️ {ativo}: entrada já ocorreu — aguardando resultado", C.Y))
            continue

        # ── FRESHNESS CHECK: scan do dashboard não pode ter mais de 120s ──
        if scan_ts > 0 and (time.time() - scan_ts) > 120:
            log.info(paint(
                f"  ⏭️ {ativo}: scan antigo ({int(time.time() - scan_ts)}s) — SKIP",
                C.Y
            ))
            continue

        # Cooldown individual por ativo (3 min após trade nele)
        if ativo in cooldown:
            elapsed = time.time() - cooldown[ativo]
            if elapsed < COOLDOWN_AFTER_TRADE:
                continue

        setup = {
            "dir": direction,
            "type": pat_type,
            "mode": mode,
            "confidence": round(ia_prob * 100, 1),
            "pattern": {
                "type": pat_type,
                "direction": direction,
                "mode": mode,
                "head": {"price": head_price},
                "right_shoulder": {"price": rs_price},
                "neckline": neckline,
                "target": target,
                "stop": stop,
            },
        }
        score = ia_prob

        # Candidato (qualquer sinal LIVE pendente)
        if best_any is None or ia_prob > best_any[0]:
            best_any = (score, ativo, setup, 0.001)

        # Filtro: dashboard confirmou padrão e entrada é pendente
        best_trade = (score, ativo, setup, 0.001) if (best_trade is None or ia_prob > best_trade[0]) else best_trade

    return best_trade, best_any


def wait_until_minus(tf, seconds_before):
    """Espera até `seconds_before` segundos antes do fechamento do candle."""
    while True:
        s = tf - (time.time() % tf)
        if s <= seconds_before:
            return
        time.sleep(min(s - seconds_before, 1.0))



def ai_predict(ativo, setup, stats_ai):
    """IA prediction para setup H&S."""
    arm = f"{ativo}_{setup.get('type', 'HS')}"
    arm_data = stats_ai.get("arms", {}).get(arm, {"wins": 0, "total": 0})
    n = arm_data.get("total", 0)
    w = arm_data.get("wins", 0)
    prob = w / max(n, 1) if n > 0 else 0.5
    conf = min(n / 10.0, 1.0)
    return {"prob": prob, "n_arm": n, "conf": conf}


def ai_update(ativo, setup, result_value, stats_ai):
    """Atualiza IA stats após trade H&S."""
    if "arms" not in stats_ai:
        stats_ai["arms"] = {}
    arm = f"{ativo}_{setup.get('type', 'HS')}"
    if arm not in stats_ai["arms"]:
        stats_ai["arms"][arm] = {"wins": 0, "total": 0}
    stats_ai["arms"][arm]["total"] += 1
    if result_value > 0:
        stats_ai["arms"][arm]["wins"] += 1
    meta = stats_ai.setdefault("meta", {"total": 0})
    meta["total"] = meta.get("total", 0) + 1


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


def obter_top_ativos_otc(bx: BrokerAPI) -> List[str]:
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

    # Filtrar apenas FOREX (pares de moedas válidos)
    _FOREX_CURRENCIES = {"EUR","USD","GBP","JPY","AUD","NZD","CAD","CHF","SEK","NOK",
                          "DKK","PLN","HUF","TRY","MXN","ZAR","SGD","HKD","CZK","BRL",
                          "INR","RUB","THB","ILS","XOF"}

    def _is_forex(name):
        base = name.replace("-OTC", "").replace("-otc", "")
        if any(c in base for c in ":_") or len(base) != 6:
            return False
        return base[:3].upper() in _FOREX_CURRENCIES and base[3:].upper() in _FOREX_CURRENCIES

    abertos_otc = [a for a in abertos_otc if _is_forex(a)]

    # Filtrar por payout
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
    log.info(f"TOP ativos OTC: {top}")
    return top


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
    """Grava trade no log para consumo pelo dashboard."""
    try:
        trades = []
        if os.path.exists(LIVE_LOG_FILE):
            with open(LIVE_LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                trades = data.get("trades", [])
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
        trades.append(record)
        if len(trades) > _LIVE_LOG_MAX:
            trades = trades[-_LIVE_LOG_MAX:]
        with open(LIVE_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump({"trades": trades, "updated": time.time()}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
# ORDEM + RESULTADO
# ═══════════════════════════════════════════════════════════════
def enviar_ordem(bx: BrokerAPI, ativo: str, direcao: str, stake: float) -> Optional[Tuple[str, int]]:
    """Envia ordem (TURBO → DIGITAL fallback). Expiração fixa 1 minuto."""
    d = "call" if direcao == "CALL" else "put"
    valor = float(max(VALOR_MINIMO, stake))

    # TURBO
    try:
        ok, op_id = safe_call(bx, bx.buy, valor, ativo, d, EXP_FIXA)
        if ok and op_id:
            return ("turbo", int(op_id))
        log.warning(paint(f"[ORDEM] TURBO falhou ok={ok} id={op_id}", C.Y))
    except Exception as e:
        log.warning(paint(f"[ORDEM] TURBO exc: {e}", C.Y))

    # DIGITAL fallback
    try:
        ok, op_id = safe_call(bx, bx.buy_digital_spot, ativo, valor, d, EXP_FIXA)
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
    Usa spin-lock fino nos últimos 50ms para precisão."""
    now = time.time()
    sec_in_candle = now % 60
    s = 60 - sec_in_candle
    # Se já estamos nos primeiros 5s do candle, entra direto
    if sec_in_candle < 5:
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
    bx: Optional[BrokerAPI] = None
    bx = ensure_connected(bx)

    # Inicializar ReversalAI (para stats e compatibilidade)
    reversal_ai = ReversalAI("unified")

    # ── RESET TOTAL: sempre limpa stats e retreina do Dashboard ao iniciar ──
    log.info(paint("=" * 60, C.G))
    log.info(paint("🔄 RESET IA — Retreinando com 900 velas do Dashboard!", C.G))
    log.info(paint("=" * 60, C.G))
    hs_stats = {"meta": {"total": 0}, "arms": {}}
    _safe_save_json(AI_STATS_FILE, hs_stats)

    # ── Iniciar Dashboard IA H&S automaticamente (fonte única de sinais) ──
    _ensure_dashboard_running()

    # ── TREINAMENTO INICIAL: importar backtest do Dashboard ou arquivo local ──
    log.info(paint("🧠 Importando treinamento do Dashboard (900 velas × 20 ativos)...", C.B))
    print(">>> IA: Importando treinamento do dashboard...", flush=True)
    _train_imported = False

    # 1) Tentar carregar do arquivo do Dashboard diretamente (mais rápido, não depende do scan)
    _DASHBOARD_STATS_FILE = os.path.join(os.path.expanduser("~"), ".wstrader", "hs_ia_dashboard_stats.json")
    try:
        if os.path.exists(_DASHBOARD_STATS_FILE):
            with open(_DASHBOARD_STATS_FILE, "r", encoding="utf-8") as _df:
                _disk_data = json.load(_df)
            _disk_stats = _disk_data.get("stats", {})
            if _disk_stats:
                if "arms" not in hs_stats:
                    hs_stats["arms"] = {}
                _imported_count = 0
                for _stat_key, _stat_val in _disk_stats.items():
                    # Converter chave do dashboard (ATIVO_TYPE_MODE) → bot (ATIVO_TYPE)
                    _parts = _stat_key.split("_")
                    _bot_key = f"{_parts[0]}_{_parts[1]}" if len(_parts) >= 2 else _stat_key
                    if _bot_key not in hs_stats["arms"]:
                        hs_stats["arms"][_bot_key] = {"wins": 0, "total": 0}
                    _d_total = _stat_val.get("total", 0)
                    _d_wins = _stat_val.get("wins", 0)
                    hs_stats["arms"][_bot_key]["wins"] += _d_wins
                    hs_stats["arms"][_bot_key]["total"] += _d_total
                    if _d_total > 0:
                        _imported_count += 1
                hs_stats["meta"] = {"total": sum(v.get("total", 0) for v in hs_stats["arms"].values())}
                _n_total = hs_stats["meta"]["total"]
                if _n_total > 0:
                    _safe_save_json(AI_STATS_FILE, hs_stats)
                    _lvl_num, _lvl_nome, _lvl_emoji = _get_ia_level(_n_total)
                    log.info(paint(
                        f"✅ IA TREINADA com arquivo do Dashboard! {_imported_count} chaves | "
                        f"{_n_total} amostras | Nível {_lvl_num}: {_lvl_emoji} {_lvl_nome}",
                        C.G
                    ))
                    print(f">>> IA: Treinada! {_n_total} amostras | Nível {_lvl_num} ({_lvl_nome})", flush=True)
                    _train_imported = True
    except Exception as _fe:
        log.debug(f"Arquivo do dashboard não disponível: {_fe}")

    # 2) Se não importou do arquivo, tentar via API (dashboard pode já ter dados na memória)
    if not _train_imported:
        for _try in range(30):  # tenta por até ~2.5min (scan 900 velas pode demorar)
            try:
                _req = urllib.request.urlopen(DASHBOARD_URL, timeout=10)
                _data = json.loads(_req.read().decode("utf-8"))
                _training = _data.get("ia_training_stats", {})
                _train_arms = _training.get("arms", {})
                _train_total = _training.get("meta", {}).get("total", 0)
                if _train_total > 0 and _train_arms:
                    if "arms" not in hs_stats:
                        hs_stats["arms"] = {}
                    _imported_count = 0
                    for _arm_key, _arm_data in _train_arms.items():
                        if _arm_key not in hs_stats["arms"]:
                            hs_stats["arms"][_arm_key] = {"wins": 0, "total": 0}
                        _d_total = _arm_data.get("total", 0)
                        _b_total = hs_stats["arms"][_arm_key].get("total", 0)
                        if _d_total > _b_total:
                            hs_stats["arms"][_arm_key]["wins"] = _arm_data.get("wins", 0)
                            hs_stats["arms"][_arm_key]["total"] = _d_total
                            _imported_count += 1
                    hs_stats["meta"] = {"total": sum(v.get("total", 0) for v in hs_stats["arms"].values())}
                    _safe_save_json(AI_STATS_FILE, hs_stats)
                    _n_total = hs_stats["meta"]["total"]
                    _lvl_num, _lvl_nome, _lvl_emoji = _get_ia_level(_n_total)
                    log.info(paint(
                        f"✅ IA TREINADA com Dashboard API! {_imported_count} ativos | "
                        f"{_n_total} amostras | Nível {_lvl_num}: {_lvl_emoji} {_lvl_nome}",
                        C.G
                    ))
                    print(f">>> IA: Treinada! {_n_total} amostras | Nível {_lvl_num} ({_lvl_nome})", flush=True)
                    _train_imported = True
                    break
                else:
                    log.info(paint(f"  ⏳ Dashboard ainda sem dados ({_try+1}/30)... aguardando scan...", C.Y))
                    time.sleep(5)
            except Exception as _te:
                log.info(paint(f"  ⏳ Dashboard não respondeu ({_try+1}/30): {_te}", C.Y))
                time.sleep(5)

    if not _train_imported:
        _n_total = sum(v.get("total", 0) for v in hs_stats.get("arms", {}).values())
        if _n_total > 0:
            _lvl_num, _lvl_nome, _lvl_emoji = _get_ia_level(_n_total)
            log.info(paint(f"✅ IA H&S: Stats locais ({_n_total} amostras) | Nível {_lvl_num}: {_lvl_emoji} {_lvl_nome}", C.G))
            print(f">>> IA: Nível {_lvl_num} ({_lvl_nome}) — {_n_total} amostras", flush=True)
        else:
            log.info(paint("🌱 IA H&S: Nenhuma amostra — IA iniciando do zero.", C.Y))
            print(">>> IA: Sem amostras — aguardando primeiro scan do dashboard.", flush=True)

    log.info("=" * 60)
    log.info(paint(f"🚀 WS TRADER — Cabeça e Ombros (H&S) ({_BROKER_LABEL})", C.G))
    log.info("=" * 60)
    log.info(f"✅ Estratégia: SOMENTE Cabeça e Ombros (H&S)")
    log.info(f"✅ Corretora: {_BROKER_LABEL} ({BROKER_TYPE})")
    log.info(f"✅ Expiração: {EXP_FIXA} minuto(s)")
    log.info(f"✅ Horário: 1:30 às 18:00")
    log.info(f"✅ Sinais: Dashboard IA H&S (http://localhost:8899)")
    log.info(f"✅ Retrain: 1x por semana (compartilhado entre corretoras)")
    log.info(f"✅ IA: ATIVA — aprendendo padrões H&S")
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

    print(f"\n>>> IA: Iniciado | Exp: {EXP_FIXA}min | Sinais: Dashboard", flush=True)

    # Exportar stats iniciais para o UI
    reversal_ai.save_stats_to_disk()

    # ═══ LOOP PRINCIPAL — SINAIS DO DASHBOARD H&S ═══
    while True:
        try:
            bx = ensure_connected(bx)

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

            # ── Verificar horario de operacao (1:30 - 18:00) ──
            _now = datetime.now()
            _minutos_atual = _now.hour * 60 + _now.minute
            if _minutos_atual < HORARIO_INICIO_MIN or _minutos_atual >= HORARIO_FIM_MIN:
                log.info(paint("=" * 60, C.Y))
                log.info(paint(
                    f"FORA DO HORARIO DE OPERACAO ({_now.hour}:{_now.minute:02d})",
                    C.Y
                ))
                log.info(paint(
                    "Horario disponivel: 1:30 ate 18:00",
                    C.Y
                ))
                log.info(paint(
                    "Fora desse horario o mercado OTC nao atende os requisitos",
                    C.Y
                ))
                log.info(paint(
                    "da IA para operacoes seguras. Aguarde o proximo horario.",
                    C.Y
                ))
                log.info(paint("=" * 60, C.Y))
                print(f">>> IA: FORA DO HORARIO -- Bot disponivel das 1:30 as 18:00. Fora desse horario o mercado nao atende os requisitos da IA. Aguardando...", flush=True)
                time.sleep(60)
                continue

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

            # ── Cooldown global (3 min após cada trade) ──
            if _last_trade_time > 0:
                elapsed = time.time() - _last_trade_time
                remaining = COOLDOWN_AFTER_TRADE - elapsed
                if remaining > 0:
                    mins = int(remaining) // 60
                    secs = int(remaining) % 60
                    log.info(paint(f"  ⏳ Cooldown: {mins}m{secs:02d}s restantes", C.B))
                    s = seconds_to_next(TF_M1)
                    time.sleep(min(s + 1, 30))
                    continue

            # ═══ ESPERAR SEGUNDO :45 PARA BUSCAR SINAIS ═══
            log.info(paint(f"\n⏰ Esperando :{ANALYZE_AT_SECOND:02d} para buscar sinais do dashboard...", C.B))
            wait_until_second(ANALYZE_AT_SECOND)

            # ═══ BUSCAR SINAIS DO DASHBOARD (FONTE ÚNICA DE H&S) ═══
            if not _is_dashboard_running():
                log.warning(paint("⚠️ Dashboard offline — tentando reiniciar...", C.Y))
                _ensure_dashboard_running()
                time.sleep(5)
                continue

            log.info(paint("\n🔍 Buscando sinais do Dashboard IA H&S...", C.B))
            best_trade, best_any = escolher_melhor_setup_dashboard(cooldown)

            if not best_trade:
                if best_any:
                    _, a, setup, _ = best_any
                    log.info(paint(
                        f"  ⏸️ Sinal em formação: {a} | aguardando confirmação",
                        C.Y
                    ))
                else:
                    log.info(paint("  ⏸️ Nenhum sinal no dashboard. Próximo candle.", C.Y))
                s = seconds_to_next(TF_M1)
                time.sleep(min(s + 1, 30))
                continue

            # ═══ H&S ENCONTRADO NO DASHBOARD → ENTRAR ═══
            sc, ativo, setup, atr_val = best_trade
            direcao = setup["dir"]
            pat_type = setup["type"]

            # ═══ IA: PROBABILIDADE DO DASHBOARD (FONTE PRINCIPAL) ═══
            # Dashboard IA é a fonte real — analisa 900 velas com precisão
            dashboard_prob = setup.get("confidence", 50.0) / 100.0

            # Bot IA: secundária — só filtra quando tem dados suficientes
            pred = ai_predict(ativo, setup, hs_stats)
            bot_prob = pred["prob"]
            ia_samples = pred["n_arm"]
            ia_conf = pred["conf"]

            # ═══ PROBABILIDADE COMBINADA: Dashboard é a fonte principal ═══
            # Se bot tem amostras, combina 70% dashboard + 30% bot
            # Se bot não tem amostras, usa 100% dashboard
            if ia_samples >= AI_MIN_SAMPLES:
                ia_prob = dashboard_prob * 0.7 + bot_prob * 0.3
            else:
                ia_prob = dashboard_prob  # Dashboard é a fonte real

            log.info(paint(
                f"  🤖 IA H&S: {ativo} | prob_final={ia_prob:.2f} | "
                f"dashboard={dashboard_prob:.2f} | bot={bot_prob:.2f} | amostras={ia_samples}",
                C.B
            ))

            # Bloqueio: só bloqueia se AMBAS as IAs concordam que é ruim
            if ia_samples >= AI_MIN_SAMPLES and ia_prob < AI_MIN_PROB and dashboard_prob < 0.6:
                log.info(paint(
                    f"  🚫 IA BLOQUEOU entrada: {ativo} {direcao} | "
                    f"prob_final={ia_prob:.2f} | dashboard={dashboard_prob:.2f} | bot={bot_prob:.2f}",
                    C.Y
                ))
                print(f">>> IA: Entrada BLOQUEADA {ativo} {direcao} prob={ia_prob:.2f}", flush=True)
                s = seconds_to_next(TF_M1)
                time.sleep(min(s + 1, 30))
                continue

            # ═══ GUARD H&S: VALIDAR PREÇO vs CABEÇA/NECKLINE ═══
            _head_price = setup["pattern"]["head"]["price"]
            _neckline = setup["pattern"].get("neckline", 0)
            _rs_price = setup["pattern"]["right_shoulder"]["price"]
            _guard_ok = True
            try:
                _price_df = get_candles_df(bx, ativo, TF_M1, 3)
                if _price_df is not None and len(_price_df) >= 1:
                    _cur = float(_price_df["close"].iloc[-1])
                    if direcao == "PUT":
                        # H&S: preço NÃO pode estar acima da cabeça
                        if _cur >= _head_price:
                            log.info(paint(
                                f"  🚫 GUARD: Preço ({_cur:.6f}) >= Cabeça ({_head_price:.6f}) — SKIP",
                                C.Y
                            ))
                            _guard_ok = False
                        # PUT: preço deve estar ABAIXO ou próximo do neckline
                        elif _neckline > 0 and _cur > _neckline + (_head_price - _neckline) * 0.5:
                            log.info(paint(
                                f"  🚫 GUARD: Preço ({_cur:.6f}) muito acima do Neckline ({_neckline:.6f}) — SKIP",
                                C.Y
                            ))
                            _guard_ok = False
                    else:  # CALL (iH&S)
                        if _cur <= _head_price:
                            log.info(paint(
                                f"  🚫 GUARD: Preço ({_cur:.6f}) <= Cabeça ({_head_price:.6f}) — SKIP",
                                C.Y
                            ))
                            _guard_ok = False
                        elif _neckline > 0 and _cur < _neckline - (_neckline - _head_price) * 0.5:
                            log.info(paint(
                                f"  🚫 GUARD: Preço ({_cur:.6f}) muito abaixo do Neckline ({_neckline:.6f}) — SKIP",
                                C.Y
                            ))
                            _guard_ok = False
                    if _guard_ok:
                        log.info(paint(
                            f"  ✅ GUARD OK: {ativo} {direcao} | Preço={_cur:.6f} | "
                            f"Neck={_neckline:.6f} | Cabeça={_head_price:.6f}",
                            C.G
                        ))
            except Exception as _ge:
                log.warning(paint(f"  ⚠️ GUARD: Não validou preço ({_ge})", C.Y))

            if not _guard_ok:
                print(f">>> IA: GUARD bloqueou {ativo} {direcao} — preço fora da zona H&S", flush=True)
                s = seconds_to_next(TF_M1)
                time.sleep(min(s + 1, 30))
                continue

            # Calcular stake
            stake = calcular_stake(bx)

            # ═══ ENTRAR IMEDIATAMENTE (na seta do dashboard) ═══
            _log_live_trade(ativo, direcao, None, None, stake,
                            confidence=dashboard_prob * 100, status="entry")

            op = enviar_ordem(bx, ativo, direcao, stake)
            if not op:
                log.warning(paint(f"  ❌ Falha na ordem: {ativo}", C.R))
                s = seconds_to_next(TF_M1)
                time.sleep(min(s + 1, 30))
                continue

            op_type, op_id = op
            log.info(paint(
                f"  ✅ ENTRADA: {ativo} {direcao} | Stake={stake:.2f} | Tipo={op_type} | "
                f"Dashboard={dashboard_prob:.0%} | Final={ia_prob:.0%}",
                C.G if direcao == "CALL" else C.R
            ))
            print(f">>> IA: Entrada {ativo} {direcao} stake={stake:.2f} dashboard={dashboard_prob:.2f}", flush=True)

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

            _log_live_trade(ativo, direcao, res, None, stake,
                            confidence=dashboard_prob * 100, status=_live_status)

            # ── Notificar dashboard em tempo real (POST) ──
            if _live_status in ("win", "loss"):
                try:
                    _post_data = json.dumps({
                        "ativo": ativo, "dir": direcao, "result": _live_status,
                        "price": 0, "stake": stake, "profit": res,
                        "time": datetime.now().strftime("%H:%M"),
                        "ts": time.time(), "broker": _broker_suffix,
                    }).encode("utf-8")
                    _req = urllib.request.Request(
                        f"{DASHBOARD_URL.rsplit('/api/', 1)[0]}/api/trade",
                        data=_post_data,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    urllib.request.urlopen(_req, timeout=3)
                except Exception:
                    pass  # dashboard pode não estar rodando

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

            # Cooldown
            _last_trade_time = time.time()
            cooldown[ativo] = time.time()

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
