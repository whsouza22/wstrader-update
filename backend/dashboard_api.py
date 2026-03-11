"""
Dashboard API — Endpoints para o dashboard HTML do WS Trader
Integra com o FastAPI existente (main_stripe.py)
"""
import os
import sys
import json
import subprocess
import threading
import logging
import signal
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

logger = logging.getLogger("dashboard_api")

router = APIRouter()

# ─── PATHS ───
WSTRADER_DIR = os.path.join(os.path.expanduser("~"), ".wstrader")
DAILY_LOG = os.path.join(WSTRADER_DIR, "ws_daily_log.json")
ENV_FILE = os.path.join(WSTRADER_DIR, ".env")
PREFS_FILE = os.path.join(WSTRADER_DIR, "preferences.json")

# ─── BOT STATE ───
_bot_state = {
    "running": False,
    "process": None,
    "broker": "iq_option",
    "account": "DEMO",
    "pid": None,
    "wins": 0,
    "losses": 0,
    "profit": 0.0,
    "balance": 0.0,
    "operations": [],
}
_bot_lock = threading.Lock()


# ═══════════════════════════════════════════════════
#  MODELS
# ═══════════════════════════════════════════════════
class LoginRequest(BaseModel):
    email: str
    password: str
    broker: str = "iq_option"

class BotStartRequest(BaseModel):
    broker: str = "iq_option"
    account: str = "DEMO"
    email: str = ""
    password: str = ""

class ChatRequest(BaseModel):
    message: str
    email: str = ""
    broker: str = "iq_option"


# ═══════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════
def _load_daily_log() -> dict:
    default = {"version": 2, "days": {}}
    try:
        if os.path.exists(DAILY_LOG):
            with open(DAILY_LOG, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if "version" not in data and "date" in data and "brokers" in data:
                old_date = data["date"]
                old_brokers = data["brokers"]
                data = {"version": 2, "days": {old_date: {"brokers": old_brokers}}}
            return data
    except Exception:
        pass
    return default


def _get_today_stats(broker: str, account: str) -> dict:
    full = _load_daily_log()
    today = datetime.now().strftime("%Y-%m-%d")
    bk = broker.lower().replace(" ", "_")
    section = (
        full.get("days", {})
        .get(today, {})
        .get("brokers", {})
        .get(bk, {})
        .get(account, {})
    )
    return {
        "wins": section.get("wins", 0),
        "losses": section.get("losses", 0),
        "profit": section.get("profit", 0.0),
        "operations": section.get("operations", []),
    }


def _load_env_credentials() -> dict:
    """Carrega credenciais salvas do .env"""
    result = {"email": "", "password": "", "broker": "iq_option"}
    try:
        if os.path.exists(ENV_FILE):
            with open(ENV_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        k, v = line.split('=', 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k == 'IQ_EMAIL':
                            result['email'] = v
                        elif k == 'IQ_PASSWORD':
                            result['password'] = v
    except Exception:
        pass
    try:
        if os.path.exists(PREFS_FILE):
            with open(PREFS_FILE, 'r', encoding='utf-8') as f:
                prefs = json.load(f)
                result['broker'] = prefs.get('last_broker', 'iq_option')
    except Exception:
        pass
    return result


def _detect_base_path():
    """Detecta o diretório base (frozen vs script)"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════
#  SERVE DASHBOARD HTML
# ═══════════════════════════════════════════════════
@router.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the modern HTML dashboard"""
    base = _detect_base_path()
    html_path = os.path.join(base, "dashboard.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="dashboard.html não encontrado")
    with open(html_path, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())


# ═══════════════════════════════════════════════════
#  AUTH ENDPOINTS
# ═══════════════════════════════════════════════════
@router.get("/api/auth/saved")
def get_saved_auth():
    """Retorna credenciais salvas (sem senha)"""
    creds = _load_env_credentials()
    return {
        "has_credentials": bool(creds["email"]),
        "email": creds["email"],
        "broker": creds["broker"],
    }


@router.post("/api/login")
def api_login(data: LoginRequest):
    """
    Valida credenciais tentando conectar ao broker.
    Salva credenciais se sucesso.
    """
    email = data.email.strip()
    password = data.password
    broker = data.broker.strip().lower()

    if not email or not password:
        return {"success": False, "message": "Email e senha são obrigatórios."}

    # Tentar conectar ao broker para validar
    connected = False
    balance = 0.0

    if broker in ("iq_option", "iq"):
        try:
            from iqoptionapi.stable_api import IQ_Option
            api = IQ_Option(email, password)
            check, reason = api.connect()
            if check:
                connected = True
                balance = api.get_balance()
                api.change_balance("PRACTICE")
                balance = api.get_balance()
                api.disconnect()
            else:
                return {"success": False, "message": f"Falha na conexão: {reason}"}
        except Exception as e:
            return {"success": False, "message": f"Erro: {str(e)}"}

    elif broker == "bullex":
        try:
            from bullexapi.stable_api import Bullex
            api = Bullex(email, password)
            check = api.connect()
            if check:
                connected = True
                balance = api.get_balance()
                api.disconnect()
            else:
                return {"success": False, "message": "Falha na conexão Bullex."}
        except Exception as e:
            return {"success": False, "message": f"Erro: {str(e)}"}

    elif broker == "casatrader":
        try:
            from casatraderapi.stable_api import CasaTrader
            api = CasaTrader(email, password)
            check = api.connect()
            if check:
                connected = True
                balance = api.get_balance()
                api.disconnect()
            else:
                return {"success": False, "message": "Falha na conexão CasaTrader."}
        except Exception as e:
            return {"success": False, "message": f"Erro: {str(e)}"}
    else:
        return {"success": False, "message": f"Broker desconhecido: {broker}"}

    if connected:
        # Salvar credenciais
        try:
            os.makedirs(WSTRADER_DIR, exist_ok=True)
            lines = []
            lines.append(f'IQ_EMAIL={email}')
            lines.append(f'IQ_PASSWORD={password}')
            with open(ENV_FILE, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
        except Exception:
            pass

        # Salvar broker preferido
        try:
            prefs = {}
            if os.path.exists(PREFS_FILE):
                with open(PREFS_FILE, 'r', encoding='utf-8') as f:
                    prefs = json.load(f)
            prefs['last_broker'] = broker
            with open(PREFS_FILE, 'w', encoding='utf-8') as f:
                json.dump(prefs, f, indent=2)
        except Exception:
            pass

        with _bot_lock:
            _bot_state["balance"] = balance

        return {
            "success": True,
            "message": "Login realizado com sucesso!",
            "balance": balance,
            "broker": broker,
        }

    return {"success": False, "message": "Falha na autenticação."}


# ═══════════════════════════════════════════════════
#  BOT CONTROL ENDPOINTS
# ═══════════════════════════════════════════════════
@router.post("/api/bot/start")
def bot_start(data: BotStartRequest):
    """Inicia o bot de trading como subprocesso"""
    with _bot_lock:
        if _bot_state["running"]:
            return {"success": False, "message": "Bot já está rodando."}

    broker = data.broker.strip().lower()
    account = data.account.strip().upper()

    # Configurar environment
    env = os.environ.copy()
    env["IQ_EMAIL"] = data.email
    env["IQ_PASSWORD"] = data.password
    env["BROKER_TYPE"] = broker
    env["ACCOUNT_TYPE"] = account

    # Encontrar o script principal
    base = _detect_base_path()
    if getattr(sys, 'frozen', False):
        # Modo empacotado: usar o próprio executável com --run-bot
        exe = sys.executable
        cmd = [exe, "--run-bot", broker]
    else:
        # Modo dev: usar python + TelaPrincipal.py --run-bot
        py = sys.executable
        script = os.path.join(base, "TelaPrincipal.py")
        cmd = [py, script, "--run-bot", broker]

    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        with _bot_lock:
            _bot_state["running"] = True
            _bot_state["process"] = proc
            _bot_state["pid"] = proc.pid
            _bot_state["broker"] = broker
            _bot_state["account"] = account
            # Reset stats
            _bot_state["wins"] = 0
            _bot_state["losses"] = 0
            _bot_state["profit"] = 0.0
            _bot_state["operations"] = []

        # Thread para monitorar o processo
        threading.Thread(target=_monitor_bot, args=(proc,), daemon=True).start()

        return {"success": True, "message": "Bot iniciado.", "pid": proc.pid}
    except Exception as e:
        return {"success": False, "message": f"Erro ao iniciar bot: {str(e)}"}


def _monitor_bot(proc):
    """Monitora o subprocesso do bot, lê stderr para saldo/meta e detecta saída"""
    try:
        while proc.poll() is None:
            line = proc.stderr.readline()
            if not line:
                continue
            try:
                text = line.decode('utf-8', errors='replace').strip()
                # Buscar atualizações de stats no stderr do bot
                if "SALDO:" in text:
                    try:
                        val = float(text.split("SALDO:")[1].strip().split()[0])
                        with _bot_lock:
                            _bot_state["balance"] = val
                    except Exception:
                        pass
            except Exception:
                pass

        # Processo terminou - atualizar stats do arquivo
        _sync_stats_from_file()

    except Exception:
        pass
    finally:
        with _bot_lock:
            _bot_state["running"] = False
            _bot_state["process"] = None
            _bot_state["pid"] = None


def _sync_stats_from_file():
    """Sincroniza stats do estado com o arquivo daily log"""
    with _bot_lock:
        broker = _bot_state["broker"]
        account = _bot_state["account"]
    stats = _get_today_stats(broker, account)
    with _bot_lock:
        _bot_state["wins"] = stats["wins"]
        _bot_state["losses"] = stats["losses"]
        _bot_state["profit"] = stats["profit"]
        _bot_state["operations"] = stats.get("operations", [])


@router.post("/api/bot/stop")
def bot_stop():
    """Para o bot de trading"""
    with _bot_lock:
        if not _bot_state["running"]:
            return {"success": True, "message": "Bot não está rodando."}

        proc = _bot_state["process"]
        if proc:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass

        _bot_state["running"] = False
        _bot_state["process"] = None
        _bot_state["pid"] = None

    _sync_stats_from_file()
    return {"success": True, "message": "Bot parado."}


@router.get("/api/bot/status")
def bot_status():
    """Retorna status atual do bot + stats"""
    # Re-sincronizar com o arquivo a cada poll
    _sync_stats_from_file()

    with _bot_lock:
        # Verificar se processo ainda vive
        if _bot_state["process"] and _bot_state["process"].poll() is not None:
            _bot_state["running"] = False
            _bot_state["process"] = None
            _bot_state["pid"] = None

        return {
            "bot_running": _bot_state["running"],
            "broker": _bot_state["broker"],
            "account": _bot_state["account"],
            "wins": _bot_state["wins"],
            "losses": _bot_state["losses"],
            "profit": _bot_state["profit"],
            "balance": _bot_state["balance"],
            "operations": _bot_state["operations"][-20:],
        }


# ═══════════════════════════════════════════════════
#  STATS ENDPOINT
# ═══════════════════════════════════════════════════
@router.get("/api/stats")
def get_stats(broker: str = "iq_option", account: str = "DEMO"):
    """Retorna stats do dia para broker/conta"""
    stats = _get_today_stats(broker, account)
    return stats


@router.get("/api/stats/weekly")
def get_weekly_stats():
    """Retorna stats dos últimos 7 dias"""
    result = {}
    brokers = ["iq_option", "bullex", "casatrader"]
    full = _load_daily_log()
    all_days = full.get("days", {})

    for broker in brokers:
        broker_data = []
        for i in range(7):
            from datetime import timedelta
            day = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            day_info = all_days.get(day, {})
            total_profit = 0.0
            total_wins = 0
            total_losses = 0
            for acct in ["DEMO", "REAL"]:
                section = day_info.get("brokers", {}).get(broker, {}).get(acct, {})
                total_profit += section.get("profit", 0.0)
                total_wins += section.get("wins", 0)
                total_losses += section.get("losses", 0)
            broker_data.append({
                "date": day,
                "profit": total_profit,
                "wins": total_wins,
                "losses": total_losses,
            })
        result[broker] = list(reversed(broker_data))

    return result


# ═══════════════════════════════════════════════════
#  CHAT ENDPOINT
# ═══════════════════════════════════════════════════
@router.post("/api/chat")
def chat_message(data: ChatRequest):
    """Processa mensagem do chat"""
    msg = data.message.strip().lower()

    # Comandos básicos
    if msg in ("status", "estado"):
        with _bot_lock:
            running = _bot_state["running"]
            broker = _bot_state["broker"]
        stats = _get_today_stats(broker, data.broker)
        total = stats["wins"] + stats["losses"]
        wr = f'{(stats["wins"]/total*100):.1f}%' if total > 0 else '0%'
        return {
            "response": (
                f"{'🟢 IA Operando' if running else '🔴 IA Parada'}\n\n"
                f"📊 Wins: {stats['wins']} | Losses: {stats['losses']}\n"
                f"📈 Win Rate: {wr}\n"
                f"💰 Lucro: R$ {stats['profit']:.2f}"
            )
        }

    if msg in ("ajuda", "help"):
        return {
            "response": (
                "📋 <b>Comandos disponíveis:</b>\n\n"
                "• <b>status</b> — estado atual da IA\n"
                "• <b>ajuda</b> — lista de comandos\n"
                "• <b>iniciar</b> — iniciar a IA\n"
                "• <b>parar</b> — parar a IA\n"
                "• <b>saldo</b> — verificar saldo\n"
                "• <b>resultado</b> — resultado do dia"
            )
        }

    if msg in ("iniciar", "start"):
        return {"response": "Use o botão ▶ Iniciar IA no painel de controles."}

    if msg in ("parar", "stop"):
        return {"response": "Use o botão ⏸ Parar IA no painel de controles."}

    if msg in ("saldo", "balance"):
        with _bot_lock:
            bal = _bot_state["balance"]
        return {"response": f"💰 Saldo atual: R$ {bal:.2f}"}

    if msg in ("resultado", "result", "profit"):
        stats = _get_today_stats(data.broker, "DEMO")
        return {
            "response": (
                f"📊 <b>Resultado do Dia:</b>\n\n"
                f"✅ Wins: {stats['wins']}\n"
                f"❌ Losses: {stats['losses']}\n"
                f"💰 Lucro: R$ {stats['profit']:.2f}"
            )
        }

    return {
        "response": (
            "Não entendi o comando. Digite <b>ajuda</b> para ver os comandos disponíveis."
        )
    }
