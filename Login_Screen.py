import os
import sys
import flet as ft
import logging
import re
import webbrowser
import requests
from dotenv import load_dotenv
import time
import urllib.parse
import threading
import asyncio
from requests.exceptions import RequestException
try:
    from iqoptionapi.stable_api import IQ_Option
except ImportError:
    IQ_Option = None

try:
    from bullexapi.stable_api import IQ_Option as Bullex
except ImportError:
    Bullex = None

try:
    from casatraderapi.stable_api import IQ_Option as CasaTrader
except ImportError:
    CasaTrader = None
import json

# Configuração básica de logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Arquivo de preferências do usuário
USER_PREFS_FILE = os.path.join(os.path.expanduser("~"), ".wstrader", "preferences.json")

def load_language_from_file():
    """Carrega o idioma salvo do arquivo JSON"""
    try:
        if os.path.exists(USER_PREFS_FILE):
            with open(USER_PREFS_FILE, 'r', encoding='utf-8') as f:
                prefs = json.load(f)
                lang = prefs.get('language', 'PT')
                logger.info(f"[LOGIN] ✅ Idioma '{lang}' carregado de {USER_PREFS_FILE}")
                return lang
    except Exception as ex:
        logger.error(f"[LOGIN] ❌ Erro ao carregar idioma: {ex}")

    logger.info("[LOGIN] ✅ Usando idioma padrão: PT")
    return 'PT'

# Importar o gerenciador de licenças
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    if getattr(sys, "_MEIPASS", None) and sys._MEIPASS not in sys.path:
        sys.path.insert(0, sys._MEIPASS)

    from license_manager import validate_license, get_hardware_id
except ImportError:
    logger.warning("license_manager não encontrado. Sistema de licenças desabilitado.")
    validate_license = None
    get_hardware_id = None


# Função para obter o caminho do .env persistente do usuário
def get_user_env_path():
    env_dir = os.path.join(os.path.expanduser("~"), ".wstrader")
    os.makedirs(env_dir, exist_ok=True)
    return os.path.join(env_dir, ".env")


def _unprotect_env(env_file: str):
    """Remove atributos de proteção (ReadOnly/Hidden/System) antes de escrever."""
    try:
        if os.path.exists(env_file) and sys.platform == 'win32':
            import ctypes
            # Remover atributos ReadOnly(1) + Hidden(2) + System(4)
            ctypes.windll.kernel32.SetFileAttributesW(env_file, 0x80)  # NORMAL
    except Exception as ex:
        logger.debug(f"Não foi possível desproteger .env: {ex}")


def _protect_env(env_file: str):
    """Reaplica atributos de proteção (ReadOnly + Hidden + System) após escrever."""
    try:
        if os.path.exists(env_file) and sys.platform == 'win32':
            import ctypes
            # ReadOnly(1) + Hidden(2) + System(4) = 7
            ctypes.windll.kernel32.SetFileAttributesW(env_file, 0x01 | 0x02 | 0x04)
    except Exception as ex:
        logger.debug(f"Não foi possível proteger .env: {ex}")


def _ensure_env_exists():
    """Garante que o .env existe com campos de email/senha do usuário."""
    env_file = get_user_env_path()

    if not os.path.exists(env_file):
        # Criar arquivo com valores padrão (somente credenciais do usuário)
        content = (
            "# =============================================\n"
            "# WS Trader - Credenciais do Usuário\n"
            "# Este arquivo NÃO deve ser deletado\n"
            "# =============================================\n\n"
            "IQ_EMAIL=\nIQ_PASSWORD=\nIQ_PASS=\n\n"
            "STRIPE_PRODUCT_ID=\n"
        )
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(content)
        _protect_env(env_file)
        logger.info(f"✅ .env criado em {env_file}")

    return env_file


# Garante que o .env existe
_ensure_env_exists()

# Carrega configurações do .env
load_dotenv(dotenv_path=get_user_env_path())

PAYMENT_LINK = "https://wstrader.io/pricing.html"
CUSTOMER_PORTAL_LINK = "https://billing.stripe.com/p/login/00g6oQcPN8tl3VC9AA"

# Função para verificar a assinatura via API externa
def check_subscription(email, selected_lang):
    t = get_translation(selected_lang)
    try:
        # Tenta o backend LOCAL primeiro (retorna product_id)
        urls = [
            "http://127.0.0.1:8000/check_subscription",
            "https://api-wstrader.onrender.com/check_subscription",
        ]
        data = None
        for url in urls:
            try:
                response = requests.post(url, json={"email": email}, timeout=10)
                response.raise_for_status()
                data = response.json()
                if data.get("status") in ["active", "trial"]:
                    logger.info(f"Assinatura válida via {url}")
                    break
            except Exception as ex:
                logger.debug(f"Fallback: {url} falhou ({ex})")
                continue

        if data is None:
            return False, t['backend_config_error']

        if data.get("status") in ["active", "trial"]:
            logger.info(f"Assinatura válida encontrada para o email: {email}")
            # Salvar product_id no .env para controle de plano (DEMO vs PRO)
            product_id = data.get("product_id", "")
            if product_id:
                try:
                    _save_product_id(product_id)
                except Exception as ex:
                    logger.warning(f"Não foi possível salvar product_id: {ex}")
            else:
                logger.warning("⚠️ Backend não retornou product_id — plano pode ficar bloqueado")
            return True, data.get("message", "Assinatura ativa.")
        else:
            logger.warning(f"Assinatura inválida para o email: {email}. Motivo: {data.get('message')}")
            return False, data.get("message", t['backend_no_active_subscription'])
    except RequestException as e:
        logger.error(f"Erro ao verificar assinatura via API: {str(e)}")
        return False, t['backend_config_error']


def _save_product_id(product_id: str):
    """Salva o product_id do Stripe no .env para controle de plano."""
    env_file = get_user_env_path()  # ~/.wstrader/.env (mesmo path que o app lê)
    try:
        _unprotect_env(env_file)
        content = ""
        if os.path.exists(env_file):
            with open(env_file, "r", encoding="utf-8") as f:
                content = f.read()
        if "STRIPE_PRODUCT_ID" in content:
            content = re.sub(r'STRIPE_PRODUCT_ID=.*\n?', f'STRIPE_PRODUCT_ID={product_id}\n', content)
        else:
            content += f'STRIPE_PRODUCT_ID={product_id}\n'
        with open(env_file, "w", encoding="utf-8") as f:
            f.write(content)
        _protect_env(env_file)
        os.environ["STRIPE_PRODUCT_ID"] = product_id
        logger.info(f"Product ID salvo: {product_id}")
    except Exception as e:
        logger.error(f"Erro ao salvar product_id no .env: {e}")

# Função para verificar credenciais do broker (detecta automaticamente)
def check_iq_credentials(email, password, t):
    """Verifica credenciais contra o broker correto (IQ/Bullex/CasaTrader).
    Detecta pelo BROKER_TYPE env var ou pelo último broker usado (preferências).
    """
    try:
        # Detectar broker correto
        broker_type = os.getenv("BROKER_TYPE", "").lower().strip()
        
        # Se não há env var, tentar preferências do usuário
        if not broker_type:
            try:
                prefs_path = os.path.join(os.path.expanduser("~"), ".wstrader", "preferences.json")
                if os.path.exists(prefs_path):
                    with open(prefs_path, "r", encoding="utf-8") as f:
                        prefs = json.load(f)
                        broker_type = prefs.get("last_broker", "").lower().strip()
            except Exception:
                pass
        
        # Selecionar API do broker
        BrokerAPI = None
        broker_name = "Broker"
        if broker_type in ("bullex", "bullux"):
            BrokerAPI = Bullex
            broker_name = "Bullex"
        elif broker_type in ("casatrader", "casa_trader"):
            BrokerAPI = CasaTrader
            broker_name = "CasaTrader"
        else:
            BrokerAPI = IQ_Option
            broker_name = "IQ Option"
        
        if BrokerAPI is None:
            logger.warning(f"API do broker {broker_name} não encontrada — pulando verificação de credenciais")
            return True, ""  # Pular verificação se API não está instalada
        
        bx = BrokerAPI(email, password)
        connect_result = bx.connect()
        if not connect_result[0]:
            logger.error(f"Falha na conexão com {broker_name}: {connect_result[1]}")
            return False, t["invalid_credentials"]
        logger.info(f"Conexão com {broker_name} bem-sucedida para verificação de credenciais")
        
        # Desconectar imediatamente para não manter session aberta
        try:
            bx.disconnect()
        except Exception:
            pass
        
        return True, ""
    except Exception as e:
        logger.error(f"Erro ao verificar credenciais: {str(e)}")
        return False, t["invalid_credentials"]

# --- Traduções ---
def get_translation(lang):
    translations = {
        "PT": {
            "title": "Faça seu login",
            "email_label": "E-mail",
            "password_label": "Senha",
            "button_save": "Entrar",
            "no_account": "Ainda não tem conta?",
            "create_one": "Crie aqui",
            "access_denied": "Acesso Negado",
            "payment_required_title": "Assinatura Necessária",
            "payment_required_message": "Para acessar, é necessária uma assinatura ativa.",
            "make_payment": "Fazer Pagamento",
            "close_button": "Fechar",
            "google_error": "Erro ao autenticar com Google. Tente novamente ou verifique sua conexão.",
            "google_config_error": "Arquivo 'credentials.json' não encontrado. Verifique a pasta do projeto.",
            "save_success": "Credenciais salvas com sucesso!",
            "payment_error": "Não autorizado. Contate o suporte.",
            "credentials_required": "Por favor, insira email e senha válidos.",
            "invalid_email": "E-mail inválido. Verifique o formato.",
            "google_login": "Conectar com Google",
            "backend_not_registered": "E-mail não registrado no nosso sistema de pagamento.",
            "backend_no_active_subscription": "Nenhuma assinatura ativa ou em teste foi encontrada.",
            "backend_config_error": "Erro de conexão com o sistema de pagamento. Contate o suporte.",
            "trial_message": "Período de teste ativo até: ",
            "expired_trial": "Período de teste expirou em: ",
            "authenticating_backend": "Analisando a conta no Stripe...",
            "backend_connected": "Assinatura válida 🎉",
            "redirecting": "Tudo certo! Redirecionando...",
            "auth_error_title": "Erro de Autenticação",
            "google_browser_open": "Janela aberta no seu navegador para autenticar...",
            "google_authenticated": "Autenticado com Google 🎉",
            "backend_account_issue": "Problema na sua conta...",
            "manual_authenticated": "Autenticado com sucesso 🎉",
            "save_credentials_error": "Erro ao salvar credenciais. Tente novamente.",
            "google_iq_password_prompt": "Por favor, insira a senha da sua conta IQ Option para {email}",
            "iq_password_label": "Senha IQ Option",
            "iq_credentials_invalid": "Credenciais da IQ Option inválidas. Verifique o email e a senha.",
            "license_key_label": "Chave de Licença Gratuita",
            "license_key_hint": "Digite sua chave de licença",
            "license_title": "Ativação de Licença",
            "license_subtitle": "Digite sua chave de licença gratuita para continuar",
            "license_activate": "Ativar Licença",
            "license_validating": "Validando licença...",
            "license_invalid": "Chave de licença inválida",
            "license_limit_reached": "Esta chave já foi ativada",
            "license_already_used": "Esta chave já está em uso",
            "license_success": "Licença ativada com sucesso!",
            "license_info": "Versão gratuita.\nSua chave é exclusiva e vinculada a este computador.",
            "free_license_detected": "Licença FREE detectada - indo para autenticação da IA",
            "free_license_skip_stripe": "Licença FREE - pulando validação do Stripe",
            "free_license_verified": "Licença FREE verificada ✓",
            "free_license_activated": "Licença FREE ativada - indo para autenticação da IA",
            "redirecting_license": "Redirecionando para validação de licença...",
            "license_validated_skip": "Licença já validada anteriormente (lida do .env), pulando validação..."
            ,"paid_access_button": "Entrar (Assinante)"
            ,"lifetime_key_title": "Chave vitalícia"
            ,"lifetime_key_message": "Se você possui uma chave vitalícia, clique em **Sim** para inserir. Se não possui, clique em **Não** para voltar ao login."
            ,"lifetime_key_yes": "Sim, inserir chave"
            ,"lifetime_key_no": "Não"
        },
        "EN": {
            "title": "Login",
            "email_label": "Email",
            "password_label": "Password",
            "button_save": "Login",
            "no_account": "No account yet?",
            "create_one": "Create here",
            "access_denied": "Access Denied",
            "payment_required_title": "Subscription Required",
            "payment_required_message": "An active subscription is required to access.",
            "make_payment": "Make Payment",
            "close_button": "Close",
            "google_error": "Error authenticating with Google. Please try again or check your connection.",
            "google_config_error": "'credentials.json' file not found. Please check the project folder.",
            "save_success": "Credentials saved successfully!",
            "payment_error": "Not authorized. Contact support.",
            "credentials_required": "Please enter a valid email and password.",
            "invalid_email": "Invalid email. Please check the format.",
            "google_login": "Sign in with Google",
            "backend_not_registered": "Email not registered in our payment system.",
            "backend_no_active_subscription": "No active or trial subscription was found.",
            "backend_config_error": "Payment system connection error. Contact support.",
            "trial_message": "Trial period active until: ",
            "expired_trial": "Trial period expired on: ",
            "authenticating_backend": "Checking account with Stripe...",
            "backend_connected": "Subscription valid 🎉",
            "redirecting": "All set! Redirecting...",
            "auth_error_title": "Authentication Error",
            "google_browser_open": "Browser window opened for authentication...",
            "google_authenticated": "Authenticated with Google 🎉",
            "backend_account_issue": "Issue with your account...",
            "manual_authenticated": "Authenticated successfully 🎉",
            "save_credentials_error": "Error saving credentials. Please try again.",
            "google_iq_password_prompt": "Please enter the IQ Option password for {email}",
            "iq_password_label": "IQ Option Password",
            "iq_credentials_invalid": "IQ Option credentials invalid. Please check the email and password.",
            "license_key_label": "Free License Key",
            "license_key_hint": "Enter your license key",
            "license_title": "License Activation",
            "license_subtitle": "Enter your free license key to continue",
            "license_activate": "Activate License",
            "license_validating": "Validating license...",
            "license_invalid": "Invalid license key",
            "license_limit_reached": "This key has already been activated",
            "license_already_used": "This key is already in use",
            "license_success": "License activated successfully!",
            "license_info": "Free version.\nYour key is unique and bound to this computer.",
            "free_license_detected": "FREE license detected - going to AI authentication",
            "free_license_skip_stripe": "FREE license - skipping Stripe validation",
            "free_license_verified": "FREE license verified ✓",
            "free_license_activated": "FREE license activated - going to AI authentication",
            "redirecting_license": "Redirecting to license validation...",
            "license_validated_skip": "License already validated (read from .env), skipping validation..."
            ,"paid_access_button": "Enter (Subscriber)"
            ,"lifetime_key_title": "Lifetime key"
            ,"lifetime_key_message": "If you have a lifetime key, click **Yes** to enter it. If not, click **No** to return to login."
            ,"lifetime_key_yes": "Yes, enter key"
            ,"lifetime_key_no": "No"
        }
    }
    return translations.get(lang, translations["PT"])

# --- Funções Auxiliares ---
def show_simple_error_dialog(page, title, message, t, include_payment_button=False):
    actions = []
    if include_payment_button:
        actions.append(
            ft.ElevatedButton(
                content=ft.Text(t["make_payment"], color="#FFFFFF"),
                on_click=lambda e: (webbrowser.open_new(PAYMENT_LINK), page.close_dialog()),
                bgcolor="#1E88E5",
            )
        )
    actions.append(ft.ElevatedButton(content=ft.Text(t["close_button"]), on_click=lambda e: page.close_dialog()))

    dialog = ft.AlertDialog(
        title=ft.Text(title, color="#FFFFFF", weight=ft.FontWeight.BOLD, size=16),
        content=ft.Text(message, color="#FFFFFF", size=14, text_align=ft.TextAlign.CENTER),
        actions=actions,
        modal=True,
        shape=ft.RoundedRectangleBorder(radius=8),
        bgcolor=ft.LinearGradient(
            begin=ft.Alignment(-1, -1),
            end=ft.Alignment(1, 1),
            colors=["#1b263b", "#2a3b5b"]
        ),
        elevation=10
    )
    page.dialog = dialog
    dialog.open = True
    page.update()

def close_dialog(page):
    page.dialog.open = False
    page.update()

def go_to_payment(page):
    webbrowser.open_new(PAYMENT_LINK)
    if hasattr(page, 'dialog') and page.dialog:
        page.dialog.open = False
    page.update()

def load_credentials():
    try:
        email = os.getenv("IQ_EMAIL", "")
        password = os.getenv("IQ_PASSWORD", "")
        logger.info(f"📧 Credenciais carregadas: email={email}, senha={'*' * len(password) if password else '(vazia)'}")
        return {
            "iq_email": email,
            "iq_password": password
        }
    except Exception as e:
        logger.error(f"Erro ao carregar credenciais do .env: {e}")
        return {}

def save_credentials(email, password):
    try:
        env_file = get_user_env_path()
        lines = []
        email_updated = False
        password_updated = False

        logger.info(f"💾 Salvando credenciais: email={email}, senha={'*' * len(password)}")

        # Desproteger antes de escrever
        _unprotect_env(env_file)

        # Ler arquivo existente
        if os.path.exists(env_file):
            with open(env_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

        # Atualizar ou adicionar credenciais
        new_lines = []
        pass_updated = False
        for line in lines:
            if line.startswith('IQ_EMAIL='):
                new_lines.append(f'IQ_EMAIL={email}\n')
                email_updated = True
            elif line.startswith('IQ_PASSWORD='):
                new_lines.append(f'IQ_PASSWORD={password}\n')
                password_updated = True
            elif line.startswith('IQ_PASS='):
                new_lines.append(f'IQ_PASS={password}\n')
                pass_updated = True
            else:
                new_lines.append(line)

        # Adicionar se não existiam
        if not email_updated:
            new_lines.append(f'IQ_EMAIL={email}\n')
        if not password_updated:
            new_lines.append(f'IQ_PASSWORD={password}\n')
        if not pass_updated:
            new_lines.append(f'IQ_PASS={password}\n')

        # Salvar arquivo
        with open(env_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

        # Reproteger após escrever
        _protect_env(env_file)

        logger.info(f"✅ Credenciais atualizadas no .env: IQ_EMAIL={email}")

        # Recarregar variáveis de ambiente com override
        load_dotenv(dotenv_path=env_file, override=True)
        os.environ['IQ_EMAIL'] = email
        os.environ['IQ_PASSWORD'] = password
        os.environ['IQ_PASS'] = password

        logger.info(f"✅ Variáveis de ambiente atualizadas na memória")

    except Exception as e:
        logger.error(f"❌ Erro ao salvar credenciais no .env: {e}")
        raise Exception(f"Failed to save credentials: {str(e)}")

def save_license_data(license_key, license_type):
    """Salva dados da licença no .env (mesmo método usado para email/senha)"""
    try:
        env_file = get_user_env_path()
        _unprotect_env(env_file)
        content = ""
        if os.path.exists(env_file):
            with open(env_file, 'r', encoding='utf-8') as f:
                content = f.read()

        # Salvar LICENSE_VALID
        if "LICENSE_VALID" in content:
            content = re.sub(r'LICENSE_VALID=.*\n?', 'LICENSE_VALID=true\n', content)
        else:
            content += 'LICENSE_VALID=true\n'

        # Salvar LICENSE_KEY
        if "LICENSE_KEY" in content:
            content = re.sub(r'LICENSE_KEY=.*\n?', f'LICENSE_KEY={license_key}\n', content)
        else:
            content += f'LICENSE_KEY={license_key}\n'

        # Salvar LICENSE_TYPE
        if "LICENSE_TYPE" in content:
            content = re.sub(r'LICENSE_TYPE=.*\n?', f'LICENSE_TYPE={license_type}\n', content)
        else:
            content += f'LICENSE_TYPE={license_type}\n'

        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(content)
        _protect_env(env_file)
        logger.info(f"Licença salva no .env: LICENSE_TYPE={license_type}")
        load_dotenv(dotenv_path=env_file, override=True)
    except Exception as e:
        logger.error(f"Erro ao salvar licença no .env: {e}")

def clear_license_data():
    """Limpa dados de licença no .env para forçar revalidação"""
    try:
        env_file = get_user_env_path()
        _unprotect_env(env_file)
        content = ""
        if os.path.exists(env_file):
            with open(env_file, 'r', encoding='utf-8') as f:
                content = f.read()

        content = re.sub(r'LICENSE_VALID=.*\n?', 'LICENSE_VALID=false\n', content)
        content = re.sub(r'LICENSE_KEY=.*\n?', '', content)
        content = re.sub(r'LICENSE_TYPE=.*\n?', '', content)

        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(content)
        _protect_env(env_file)
        load_dotenv(dotenv_path=env_file, override=True)
        logger.info("Licença limpa no .env para revalidação")
    except Exception as e:
        logger.error(f"Erro ao limpar licença no .env: {e}")

def try_validate_saved_license(email: str):
    """Tenta validar licença salva (se houver) no Firebase."""
    try:
        license_key_env = os.getenv("LICENSE_KEY", "").strip().upper()
        if not license_key_env or not validate_license:
            return False, "", ""

        is_valid, error_message, user_data = validate_license(license_key_env, email=email)
        if is_valid:
            license_type = (user_data or {}).get("license_type", "FREE")
            save_license_data(license_key_env, license_type)
            return True, license_type, ""

        return False, "", error_message or ""
    except Exception as ex:
        logger.warning(f"Falha ao validar licença salva: {ex}")
        return False, "", ""

def is_valid_email(email):
    if not email:
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return False
    if len(email) > 254 or '..' in email or email.startswith('.') or email.endswith('.'):
        return False
    return True

def proceed_to_dashboard(page, email, password, t, is_google_auth=False):
    try:
        # Verificar credenciais da IQ Option antes de prosseguir
        is_valid, error_message = check_iq_credentials(email, password, t)
        if not is_valid:
            logger.error(f"Credenciais da IQ Option inválidas para o email: {email}")
            show_simple_error_dialog(page, t["auth_error_title"], error_message, t)
            page.go("/login")
            return

        save_credentials(email, password)
        try:
            if hasattr(page, 'session'):
                if hasattr(page.session, "set"):
                    page.session.set("credentials", {"email": email, "password": password, "bot_token": "dummy_token"})
                elif isinstance(page.session, dict):
                    page.session["credentials"] = {"email": email, "password": password, "bot_token": "dummy_token"}
        except Exception as e:
            logger.warning(f"Não foi possível salvar no session: {e}")

        page.snack_bar = ft.SnackBar(
            content=ft.Text(t["save_success"]),
            bgcolor="#4CAF50"
        )
        page.snack_bar.open = True
        page.update()
        page.go(f"/bot?email={urllib.parse.quote(email)}&password={urllib.parse.quote(password)}&bot_token=dummy_token&is_google_auth={is_google_auth}")
    except Exception as ex:
        logger.error(f"Erro durante o processo de login: {str(ex)}")
        show_simple_error_dialog(page, "Erro", t["save_credentials_error"], t)
        page.go("/login")

# --- Tela de Login ---
def login_screen(page: ft.Page):
    # Recarregar .env FORÇADO para refletir mudanças recentes (senha alterada, etc)
    try:
        env_file = get_user_env_path()
        # Limpar cache de variáveis antigas
        if "IQ_EMAIL" in os.environ:
            del os.environ["IQ_EMAIL"]
        if "IQ_PASSWORD" in os.environ:
            del os.environ["IQ_PASSWORD"]
        if "IQ_PASS" in os.environ:
            del os.environ["IQ_PASS"]
        if "STRIPE_PRODUCT_ID" in os.environ:
            del os.environ["STRIPE_PRODUCT_ID"]
        # Recarregar com override
        load_dotenv(dotenv_path=env_file, override=True)
        logger.info(f"🔄 .env recarregado na tela de login: {env_file}")
    except Exception as ex:
        logger.warning(f"Falha ao recarregar .env no login: {ex}")

    page.title = "WS Trader - Login"
    page.theme_mode = ft.ThemeMode.DARK
    page.window.width = 1200
    page.window.height = 750
    page.window.resizable = False
    page.window.maximizable = False
    page.window.minimizable = True
    page.padding = 0
    page.spacing = 0
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.bgcolor = ft.Colors.TRANSPARENT

    # Carregar idioma do arquivo JSON
    selected_lang = load_language_from_file()
    logger.info(f"✅ [LOGIN] Idioma FINAL selecionado: '{selected_lang}'")
    t = get_translation(selected_lang)
    credentials = load_credentials()
    error_text = ft.Ref[ft.Text]()

    def go_home(e):
        try:
            logger.info("Botão HOME clicado na tela de login - voltando para tela principal")
            # Mudar a rota e disparar route_change manualmente
            page.route = "/"
            logger.info("Chamando on_route_change manualmente")
            if hasattr(page, 'on_route_change') and page.on_route_change:
                page.on_route_change(page.route)
            logger.info("Navegação para tela principal concluída")
        except Exception as ex:
            logger.error(f"Erro ao navegar para home: {str(ex)}", exc_info=True)

    def update_save_button_state(e=None):
        logger.info("Updating save button state")
        email = email_field.value.strip() if email_field.value else ""
        password = password_field.value.strip() if password_field.value else ""
        is_valid = is_valid_email(email) and bool(password)
        login_button.disabled = not is_valid
        login_button.bgcolor = "#FF6A00" if is_valid else "#607D8B"
        key_icon_button.disabled = not is_valid
        if email and not is_valid_email(email):
            error_text.current.value = t["invalid_email"]
        else:
            error_text.current.value = ""
        page.update()

    async def save_credentials_action_async():
        logger.info("Save credentials action triggered")

        login_button.disabled = True
        loading_ring.visible = True
        error_text.current.value = ""
        page.update()

        try:
            email = email_field.value.strip()
            password = password_field.value.strip()

            if not is_valid_email(email) or not password:
                logger.error("Credenciais inválidas fornecidas.")
                show_simple_error_dialog(page, t["access_denied"], t["credentials_required"], t)
                page.go("/login")
                return

            # Validação APENAS via Stripe - sem chaves manuais
            # Usuários FREE têm vencimento até 2050 no Stripe
            logger.info("Validação via Stripe...")
            page.go(f"/authenticating?email={urllib.parse.quote(email)}&password={urllib.parse.quote(password)}&bot_token=dummy_token&is_google_auth=False")

        except Exception as ex:
            logger.error(f"Erro no login manual: {str(ex)}")
            show_simple_error_dialog(page, t["access_denied"], t["save_credentials_error"], t)
            page.go("/login")

        finally:
            login_button.disabled = False
            loading_ring.visible = False
            page.update()

    def save_credentials_action(e):
        page.run_task(save_credentials_action_async)

    # Verificar se já existem credenciais salvas
    has_saved_credentials = bool(credentials.get("iq_email") and credentials.get("iq_password"))
    
    email_field = ft.TextField(
        value=credentials.get("iq_email", ""),
        label=t["email_label"],
        border=ft.InputBorder.OUTLINE,
        bgcolor=None,
        border_color="#8B8E96",
        focused_border_color="#ECEFF1",
        height=45,
        text_size=14,
        content_padding=ft.Padding.symmetric(horizontal=15),
        cursor_color="#FFFFFF",
        prefix_icon="email_outlined",
        on_change=update_save_button_state,
        on_blur=update_save_button_state,
        on_focus=update_save_button_state,
        read_only=has_saved_credentials,
        disabled=has_saved_credentials
    )

    password_field = ft.TextField(
        value=credentials.get("iq_password", ""),
        label=t["password_label"],
        password=True,
        can_reveal_password=True,
        border=ft.InputBorder.OUTLINE,
        bgcolor=None,
        border_color="#8B8E96",
        focused_border_color="#ECEFF1",
        height=45,
        text_size=14,
        content_padding=ft.Padding.symmetric(horizontal=15),
        cursor_color="#FFFFFF",
        prefix_icon="lock_outline",
        on_change=update_save_button_state,
        on_blur=update_save_button_state,
        on_focus=update_save_button_state,
        read_only=has_saved_credentials,
        disabled=has_saved_credentials
    )

    login_button = ft.ElevatedButton(
        content=ft.Text(t["paid_access_button"], color="#050508", weight=ft.FontWeight.W_800),
        height=50,
        width=350,
        on_click=save_credentials_action,
        bgcolor="#607D8B",
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=8),
        ),
        disabled=True
    )

    lifetime_box = {"container": None, "backdrop": None}

    def handle_lifetime_yes(_e=None):
        if lifetime_box["container"] in page.overlay:
            page.overlay.remove(lifetime_box["container"])
        if lifetime_box["backdrop"] in page.overlay:
            page.overlay.remove(lifetime_box["backdrop"])
        page.update()
        email = email_field.value.strip()
        password = password_field.value.strip()
        if not is_valid_email(email) or not password:
            show_simple_error_dialog(page, t["access_denied"], t["credentials_required"], t)
            return
        logger.info(t["redirecting_license"])
        page.go(f"/license_validation?email={urllib.parse.quote(email)}&password={urllib.parse.quote(password)}")

    def handle_lifetime_no(_e=None):
        if lifetime_box["container"] in page.overlay:
            page.overlay.remove(lifetime_box["container"])
        if lifetime_box["backdrop"] in page.overlay:
            page.overlay.remove(lifetime_box["backdrop"])
        page.update()

    def open_lifetime_popup(_=None):
        if lifetime_box["container"] in page.overlay:
            return
        page.overlay.append(lifetime_box["backdrop"])
        page.overlay.append(lifetime_box["container"])
        page.update()

    loading_ring = ft.ProgressRing(visible=False, width=30, height=30, stroke_width=3, color="#FF6A00")

    no_account = ft.Row(
        controls=[
            ft.Text(t["no_account"], size=14, color="#8B8E96"),
            ft.GestureDetector(
                content=ft.Text(t["create_one"], size=14, color="#FF6A00", weight=ft.FontWeight.BOLD),
                on_tap=lambda e: go_to_payment(page),
                mouse_cursor=ft.MouseCursor.CLICK
            )
        ],
        alignment=ft.MainAxisAlignment.CENTER
    )
    
    # Link para alterar senha
    change_password_link = ft.Row(
        controls=[
            ft.GestureDetector(
                content=ft.Text("Alterar senha", size=13, color="#64748B", weight=ft.FontWeight.W_500),
                on_tap=lambda e: page.go("/change_password"),
                mouse_cursor=ft.MouseCursor.CLICK
            )
        ],
        alignment=ft.MainAxisAlignment.CENTER
    )

    # Botão de chave removido - agora usa apenas Stripe
    key_icon_button = ft.IconButton(
        icon=ft.Icons.KEY,
        icon_color="#FFFFFF",
        icon_size=18,
        tooltip="",
        on_click=None,
        disabled=True,
        visible=False,  # OCULTO - validação apenas via Stripe
    )

    header = ft.Container(
        content=ft.Row(
            controls=[
                ft.Container(expand=True),
                ft.Row(
                    controls=[
                        key_icon_button,
                        ft.IconButton(
                            icon=ft.Icons.HOME,
                            icon_color="#FFFFFF",
                            icon_size=18,
                            on_click=go_home
                        ),
                    ],
                    spacing=6,
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN
        ),
        padding=ft.Padding.only(top=10, left=20, right=20)
    )

    lifetime_instruction = ft.Container(
        alignment=ft.Alignment(0, 0),
        on_click=lambda e: None,
        content=ft.Container(
            width=360,
            height=170,
            padding=ft.padding.symmetric(horizontal=20, vertical=16),
            bgcolor="#1a202cCC",
            border=ft.border.all(1, "#2d3748"),
            border_radius=10,
            shadow=ft.BoxShadow(
                blur_radius=18,
                spread_radius=2,
                color="#00000080",
                offset=ft.Offset(0, 8),
            ),
            content=ft.Column(
                [
                    ft.Text(t["lifetime_key_title"], size=14, weight=ft.FontWeight.W_700, color="#FFFFFF"),
                    ft.Text(t["lifetime_key_message"], size=12, color="#A0A3AB"),
                    ft.Row(
                        [
                            ft.ElevatedButton(
                                content=ft.Text(t["lifetime_key_no"], color="#E5E7EB"),
                                on_click=handle_lifetime_no,
                                bgcolor="#374151",
                                style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=6)),
                            ),
                            ft.ElevatedButton(
                                content=ft.Text(t["lifetime_key_yes"], color="#FFFFFF"),
                                on_click=handle_lifetime_yes,
                                bgcolor="#FF6A00",
                                style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=6)),
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                ],
                spacing=10,
            ),
        ),
    )

    lifetime_backdrop = ft.Container(
        expand=True,
        bgcolor=ft.Colors.with_opacity(0.82, "#050508"),
        on_click=handle_lifetime_no,
    )

    lifetime_box["container"] = lifetime_instruction
    lifetime_box["backdrop"] = lifetime_backdrop

    login_form_container = ft.Container(
        width=400,
        padding=ft.Padding.symmetric(horizontal=50, vertical=30),
        bgcolor=None,
        content=ft.Column(
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=15,
            controls=[
                ft.Text("WS AI", size=32, weight=ft.FontWeight.BOLD, color="#FFFFFF"),
                ft.Text(t["title"], size=24, weight=ft.FontWeight.BOLD),
                ft.Container(height=30),
                email_field,
                password_field,
                ft.Container(height=10),
                login_button,
                no_account,
                change_password_link,
                ft.Container(height=15),
                loading_ring,
                ft.Text(ref=error_text, color="#EF5350", size=14, text_align=ft.TextAlign.CENTER)
            ]
        )
    )

    def view():
        return ft.View(
            route="/login",
            bgcolor=ft.Colors.TRANSPARENT,
            padding=0,
            spacing=0,
            controls=[
                ft.Container(
                    content=ft.Column(
                        controls=[
                            header,
                            ft.Container(
                                content=login_form_container,
                                expand=True
                            )
                        ],
                        expand=True,
                        spacing=0,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER
                    ),
                    expand=True,
                    padding=0,
                    gradient=ft.LinearGradient(
                        begin=ft.Alignment(-1, -1),
                        end=ft.Alignment(1, 1),
                        colors=["#050508", "#0a0c14"]
                    )
                )
            ]
        )

    page.views.clear()
    login_view = view()
    logger.info(f"Login view created with {len(login_view.controls)} controls")
    page.views.append(login_view)
    logger.info("Login screen view appended successfully")
    update_save_button_state(None)
    page.update()
    logger.info("Page updated")

# --- Tela de Autenticação com Efeito Typewriter ---
def show_auth_and_process(page: ft.Page):
    logger.info("╔════════════════════════════════════════╗")
    logger.info("║  TELA DE AUTENTICAÇÃO INICIALIZADA    ║")
    logger.info("╚════════════════════════════════════════╝")

    page.title = "WS Trader - Autenticando"
    page.theme_mode = ft.ThemeMode.DARK
    page.window.width = 1200
    page.window.height = 750
    page.window.resizable = False
    page.window.maximizable = False
    page.window.minimizable = True
    page.padding = 0
    page.spacing = 0
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.bgcolor = ft.Colors.TRANSPARENT

    # Carregar idioma das preferências
    selected_lang = load_language_from_file()
    t = get_translation(selected_lang)

    # Texto com efeito de digitação
    status_text = ft.Text(
        value="",
        size=28,
        weight=ft.FontWeight.BOLD,
        color="#FFFFFF",
        text_align=ft.TextAlign.CENTER,
    )

    # Função para efeito de digitação (assíncrona)
    async def type_writer(text, delay=0.06):
        status_text.value = ""
        page.update()
        for char in text:
            status_text.value += char
            page.update()
            await asyncio.sleep(delay)
        await asyncio.sleep(0.5)

    # Processo principal de autenticação
    async def perform_authentication():
        try:
            logger.info("=== INICIANDO AUTENTICAÇÃO ===")

            # Extrai parâmetros da rota
            try:
                parsed = urllib.parse.urlparse(page.route)
                params = urllib.parse.parse_qs(parsed.query)
                email = params.get("email", [""])[0]
                password = params.get("password", [""])[0]
                license_type = params.get("license_type", [""])[0]
            except Exception as e:
                logger.error(f"Erro ao parsear URL: {e}")
                await type_writer("Erro ao processar dados...")
                await asyncio.sleep(2)
                page.go("/login")
                return

            if not email or email == "pending":
                await type_writer("Dados inválidos...")
                await asyncio.sleep(2)
                page.go("/login")
                return

            # Verificar se é licença FREE - se for, pular validação do Stripe
            if license_type == "FREE":
                logger.info(f"✅ {t['free_license_skip_stripe']}")
                await type_writer(t["free_license_verified"])
                await asyncio.sleep(1.5)
            else:
                # Verificando assinatura no Stripe (somente para licenças pagas)
                await type_writer(t["authenticating_backend"])
                is_paid, reason = await asyncio.to_thread(check_subscription, email, selected_lang)

                if not is_paid:
                    await type_writer(t["backend_account_issue"])
                    await asyncio.sleep(2.5)
                    show_simple_error_dialog(
                        page,
                        t["payment_required_title"],
                        f"{t['payment_required_message']}\n\nMotivo: {reason}",
                        t,
                        include_payment_button=True
                    )
                    page.go("/login")
                    return

                await type_writer(t["backend_connected"])
            await asyncio.sleep(1.8)

            await type_writer(t["redirecting"])
            await asyncio.sleep(1.5)

            # Salvar credenciais
            try:
                save_credentials(email, password)
                if hasattr(page, 'session'):
                    if hasattr(page.session, "set"):
                        page.session.set("credentials", {"email": email, "password": password})
                    elif isinstance(page.session, dict):
                        page.session["credentials"] = {"email": email, "password": password}
            except Exception as e:
                logger.error(f"Erro ao salvar credenciais: {e}")

            page.snack_bar = ft.SnackBar(
                content=ft.Text(t["save_success"], color="white"),
                bgcolor="#4CAF50"
            )
            page.snack_bar.open = True
            page.update()
            await asyncio.sleep(1)

            # Redirecionar para o chat
            chat_route = f"/chat?email={urllib.parse.quote(email)}&password={urllib.parse.quote(password)}"
            page.go(chat_route)

        except Exception as ex:
            logger.error(f"Erro crítico na autenticação: {str(ex)}")
            await type_writer("Ocorreu um erro inesperado...")
            await asyncio.sleep(3)
            page.go("/login")

    # View corrigida com texto typewriter
    auth_view = ft.View(
        route="/authenticating",
        bgcolor=ft.Colors.TRANSPARENT,
        padding=0,
        spacing=0,
        controls=[
            ft.Container(
                content=status_text,
                alignment=ft.alignment.Alignment.CENTER,
                expand=True,
                padding=0,
                gradient=ft.LinearGradient(
                    begin=ft.Alignment(-1, -1),
                    end=ft.Alignment(1, 1),
                    colors=["#050508", "#0a0c14"]
                )
            )
        ]
    )

    page.views.clear()
    page.views.append(auth_view)
    page.update()

    logger.info("View de autenticação renderizada, iniciando autenticação...")

    # Inicia o processo de autenticação assíncrono
    page.run_task(perform_authentication)


# --- Tela de Validação de Licença ---
def license_validation_screen(page: ft.Page, email: str = None, password: str = None):
    """
    Tela intermediária para validação da chave de licença gratuita.
    Aparece APÓS login bem-sucedido, ANTES de acessar o dashboard.
    """
    logger.info("╔════════════════════════════════════════╗")
    logger.info("║  TELA DE VALIDAÇÃO DE LICENÇA         ║")
    logger.info("╚════════════════════════════════════════╝")

    # Carregar idioma
    selected_lang = load_language_from_file()
    t = get_translation(selected_lang)
    logger.info(f"✅ [LICENSE_VALIDATION] Idioma selecionado: '{selected_lang}'")

    # Se email/password não foram passados, extrair da URL
    if not email or not password:
        try:
            parsed = urllib.parse.urlparse(page.route)
            params = urllib.parse.parse_qs(parsed.query)
            email = params.get("email", [""])[0]
            password = params.get("password", [""])[0]
        except Exception as e:
            logger.error(f"Erro ao parsear parâmetros da URL: {e}")
            page.go("/login")
            return

    if not email or not password:
        logger.error("Email ou senha não fornecidos")
        page.go("/login")
        return

    logger.info(f"Validação de licença para: {email}")

    # Verificar se já tem licença válida salva no .env
    try:
        license_key_env = os.getenv("LICENSE_KEY", "").strip()

        if license_key_env:
            is_valid, validated_type, error_message = try_validate_saved_license(email)
            if is_valid and validated_type == "FREE":
                logger.info(f"✅ {t['free_license_detected']}")
                page.go(f"/authenticating?email={urllib.parse.quote(email)}&password={urllib.parse.quote(password)}&bot_token=dummy_token&is_google_auth=False&license_type=FREE")
                return
            logger.warning("Licença salva inválida no Firebase. Revalidação necessária.")
    except Exception as e:
        logger.warning(f"Erro ao verificar licença salva: {e}")

    page.title = "WS Trader - Ativação de Licença"
    page.theme_mode = ft.ThemeMode.DARK
    page.window.width = 1200
    page.window.height = 750
    page.window.resizable = False
    page.window.maximizable = False
    page.window.minimizable = True
    page.padding = 0
    page.spacing = 0
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.bgcolor = ft.Colors.TRANSPARENT

    try:
        if hasattr(page, 'session'):
            selected_lang = page.session.get("lang") or "PT"
        else:
            selected_lang = "PT"
    except:
        selected_lang = "PT"
    t = get_translation(selected_lang)

    # Elementos da UI
    license_key_field = ft.TextField(
        label=t["license_key_label"],
        hint_text=t["license_key_hint"],
        width=400,
        text_align=ft.TextAlign.CENTER,
        max_length=32,
        capitalization=ft.TextCapitalization.CHARACTERS,
        bgcolor="#2C2C2C",
        border_color="#FF6A00",
        focused_border_color="#FF6A00",
        label_style=ft.TextStyle(color="#FFFFFF"),
        text_style=ft.TextStyle(color="#FFFFFF", size=16, weight=ft.FontWeight.BOLD),
    )

    status_text = ft.Text(
        "",
        size=14,
        color="#FF0000",
        text_align=ft.TextAlign.CENTER,
        visible=False
    )

    loading_indicator = ft.ProgressRing(
        width=30,
        height=30,
        stroke_width=3,
    )

    loading_container = ft.Container(
        content=loading_indicator,
        alignment=ft.alignment.Alignment(0, 0),
        visible=False  # Inicialmente invisível
    )

    activate_button = ft.ElevatedButton(
        content=ft.Text(t["license_activate"], color="#FFFFFF", weight=ft.FontWeight.BOLD),
        bgcolor="#FF6A00",
        height=50,
        width=350,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=8),
        ),
        disabled=False
    )

    info_text = ft.Text(
        t["license_info"],
        size=12,
        color="#8B8E96",
        text_align=ft.TextAlign.CENTER,
    )

    hwid_text = ft.Text(
        "",
        size=10,
        color="#666666",
        text_align=ft.TextAlign.CENTER,
        selectable=True
    )

    def show_status(message, is_error=True):
        """Exibe mensagem de status"""
        status_text.value = message
        status_text.color = "#FF0000" if is_error else "#4CAF50"
        status_text.visible = True
        page.update()

    def hide_status():
        """Esconde mensagem de status"""
        status_text.visible = False
        page.update()

    def set_loading(loading):
        """Define estado de carregamento"""
        loading_container.visible = loading
        activate_button.disabled = loading
        license_key_field.disabled = loading
        page.update()

    def validate_key_format(key):
        """Valida formato da chave"""
        if not key:
            return False, t["credentials_required"]

        key = key.strip().upper()

        # Exigir exatamente 32 caracteres (chave completa)
        if len(key) != 32:
            return False, "Chave deve ter exatamente 32 caracteres"

        # Verificar se é alfanumérico (letras e números apenas)
        if not key.isalnum():
            return False, t["license_invalid"]

        return True, key

    def on_key_input_change(e):
        """Handler de mudança no input da chave"""
        value = e.control.value.upper().replace(" ", "") if e.control.value else ""

        # Não adicionar hífens - aceitar a chave como está
        e.control.value = value

        # Botão sempre habilitado
        activate_button.disabled = False
        activate_button.bgcolor = "#FF6A00"

        hide_status()
        page.update()

    async def on_activate_click(e):
        """Handler do botão de ativação"""
        if not validate_license:
            show_status("Sistema de licenças não disponível", is_error=True)
            return

        # Validar formato
        is_valid_format, result = validate_key_format(license_key_field.value)
        if not is_valid_format:
            show_status(f"{result}", is_error=True)
            return

        license_key = result
        hide_status()
        # Mostrar loading imediatamente
        show_status(t["license_validating"], is_error=False)
        set_loading(True)
        await asyncio.sleep(0)

        try:
            page.go(
                f"/license_checking?email={urllib.parse.quote(email)}"
                f"&password={urllib.parse.quote(password)}"
                f"&license_key={urllib.parse.quote(license_key)}"
            )
        finally:
            set_loading(False)

    # Configurar eventos
    activate_button.on_click = on_activate_click
    license_key_field.on_change = on_key_input_change
    license_key_field.on_submit = on_activate_click  # Permite Enter para ativar

    # Atualizar HWID text
    if get_hardware_id:
        hwid = get_hardware_id()
        hwid_text.value = f"Hardware ID: {hwid[:16]}... (para suporte)"

    # Container principal
    top_bar = ft.Container(
        height=44,
        padding=ft.padding.only(top=8, right=12, left=12),
        content=ft.Row(
            controls=[
                ft.Container(expand=True),
                ft.IconButton(
                    icon=ft.Icons.HOME,
                    icon_color="#FFFFFF",
                    icon_size=18,
                    on_click=lambda e: page.go("/login")
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        ),
    )

    content_column = ft.Column(
        controls=[
            ft.Text(
                "🔐",
                size=80,
            ),
            ft.Text(
                t["license_title"],
                size=28,
                weight=ft.FontWeight.BOLD,
                color="#FFFFFF"
            ),
            ft.Text(
                t["license_subtitle"],
                size=16,
                color="#8B8E96"
            ),
            ft.Container(height=10),
            info_text,
            ft.Container(height=30),
            license_key_field,
            ft.Container(height=10),
            status_text,
            ft.Container(height=20),
            ft.Stack(
                controls=[
                    activate_button,
                    loading_container
                ],
                width=350,
                height=50,
            ),
            ft.Container(height=30),
            hwid_text,
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=5
    )

    main_container = ft.Container(
        content=ft.Column(
            controls=[
                top_bar,
                ft.Container(content=content_column, expand=True),
            ],
            expand=True,
            spacing=0,
        ),
        alignment=ft.Alignment(0, 0),
        expand=True,
        gradient=ft.LinearGradient(
            begin=ft.Alignment(-1, -1),
            end=ft.Alignment(1, 1),
            colors=["#050508", "#0a0c14"]
        )
    )

    # View da tela de licença
    license_view = ft.View(
        route="/license_validation",
        bgcolor=ft.Colors.TRANSPARENT,
        padding=0,
        spacing=0,
        controls=[main_container]
    )

    page.views.clear()
    page.views.append(license_view)
    page.update()


def show_license_checking(page: ft.Page):
    """Tela intermediária de verificação da chave (estilo Stripe)."""
    page.title = "WS Trader - Verificando Licença"
    page.theme_mode = ft.ThemeMode.DARK
    page.window.width = 1200
    page.window.height = 750
    page.window.resizable = False
    page.window.maximizable = False
    page.window.minimizable = True
    page.padding = 0
    page.spacing = 0
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.bgcolor = ft.Colors.TRANSPARENT

    selected_lang = load_language_from_file()
    t = get_translation(selected_lang)

    status_text = ft.Text(
        value="",
        size=28,
        weight=ft.FontWeight.BOLD,
        color="#FFFFFF",
        text_align=ft.TextAlign.CENTER,
    )

    async def type_writer(text, delay=0.06):
        status_text.value = ""
        page.update()
        for char in text:
            status_text.value += char
            page.update()
            await asyncio.sleep(delay)
        await asyncio.sleep(0.5)

    async def perform_check():
        try:
            parsed = urllib.parse.urlparse(page.route)
            params = urllib.parse.parse_qs(parsed.query)
            email = params.get("email", [""])[0]
            password = params.get("password", [""])[0]
            license_key = params.get("license_key", [""])[0]

            if not license_key:
                await type_writer(t["license_invalid"])
                await asyncio.sleep(1.2)
                page.go(f"/license_validation?email={urllib.parse.quote(email)}&password={urllib.parse.quote(password)}")
                return

            await type_writer(t["license_validating"])

            is_valid, error_message, user_data = validate_license(license_key, email=email) if validate_license else (False, t["license_invalid"], None)

            if is_valid:
                await type_writer(f"{t['license_success']}")

                try:
                    save_license_data(license_key, user_data.get("license_type", "FREE"))
                except Exception as storage_error:
                    logger.warning(f"Erro ao salvar licença: {storage_error}")

                await asyncio.sleep(1.2)
                license_type = user_data.get("license_type", "")
                if license_type == "FREE":
                    page.go(
                        f"/authenticating?email={urllib.parse.quote(email)}&password={urllib.parse.quote(password)}"
                        f"&bot_token=dummy_token&is_google_auth=False&license_type=FREE"
                    )
                else:
                    page.go(
                        f"/authenticating?email={urllib.parse.quote(email)}&password={urllib.parse.quote(password)}"
                        f"&bot_token=dummy_token&is_google_auth=False"
                    )
                return

            if error_message:
                if "Limite de ativações" in error_message or "limit" in error_message.lower():
                    error_message = t["license_limit_reached"]
                elif "outro computador" in error_message or "another computer" in error_message.lower():
                    error_message = t["license_already_used"]
                elif "não encontrada" in error_message or "not found" in error_message.lower():
                    error_message = t["license_invalid"]

            await type_writer(f"{error_message or t['license_invalid']}")
            await asyncio.sleep(1.2)
            page.go(f"/license_validation?email={urllib.parse.quote(email)}&password={urllib.parse.quote(password)}")

        except Exception as ex:
            logger.error(f"Erro ao validar licença: {ex}", exc_info=True)
            await type_writer(f"{t['license_invalid']}")
            await asyncio.sleep(1.2)
            page.go("/login")

    checking_view = ft.View(
        route="/license_checking",
        bgcolor=ft.Colors.TRANSPARENT,
        padding=0,
        spacing=0,
        controls=[
            ft.Container(
                content=status_text,
                alignment=ft.alignment.Alignment.CENTER,
                expand=True,
                padding=0,
                gradient=ft.LinearGradient(
                    begin=ft.Alignment(-1, -1),
                    end=ft.Alignment(1, 1),
                    colors=["#050508", "#0a0c14"]
                )
            )
        ]
    )

    page.views.clear()
    page.views.append(checking_view)
    page.update()
    page.run_task(perform_check)


# --- Tela de Alteração de Senha ---
def change_password_screen(page: ft.Page):
    """
    Tela para o usuário alterar suas credenciais (email/senha).
    Mantém o mesmo padrão visual da tela de login.
    """
    logger.info("╔════════════════════════════════════════╗")
    logger.info("║  TELA DE ALTERAÇÃO DE SENHA           ║")
    logger.info("╚════════════════════════════════════════╝")

    # Carregar idioma
    selected_lang = load_language_from_file()
    t = get_translation(selected_lang)
    
    # Carregar credenciais atuais
    credentials = load_credentials()
    current_email = credentials.get("iq_email", "")

    error_text = ft.Ref[ft.Text]()
    success_text = ft.Ref[ft.Text]()

    def update_button_state(e=None):
        email = email_field.value.strip() if email_field.value else ""
        password = password_field.value.strip() if password_field.value else ""
        confirm = confirm_field.value.strip() if confirm_field.value else ""
        
        is_valid = is_valid_email(email) and len(password) >= 6 and password == confirm
        update_button.disabled = not is_valid
        update_button.bgcolor = "#FF6A00" if is_valid else "#607D8B"
        
        if email and not is_valid_email(email):
            error_text.current.value = "Email inválido"
        elif password and len(password) < 6:
            error_text.current.value = "Senha deve ter no mínimo 6 caracteres"
        elif password and confirm and password != confirm:
            error_text.current.value = "Senhas não coincidem"
        else:
            error_text.current.value = ""
        page.update()

    async def update_credentials_async():
        update_button.disabled = True
        error_text.current.value = ""
        page.update()

        try:
            new_email = email_field.value.strip()
            new_password = password_field.value.strip()

            if not is_valid_email(new_email):
                error_text.current.value = "Email inválido"
                return

            if len(new_password) < 6:
                error_text.current.value = "Senha deve ter no mínimo 6 caracteres"
                return

            # Ir imediatamente para tela de processamento
            page.go(f"/password_changing?email={urllib.parse.quote(new_email)}&password={urllib.parse.quote(new_password)}")
            return
            
        except Exception as ex:
            logger.error(f"❌ Erro ao atualizar credenciais: {str(ex)}")
            error_text.current.value = f"Erro ao salvar: {str(ex)}"

        finally:
            update_button.disabled = False
            page.update()

    def update_credentials_action(e):
        page.run_task(update_credentials_async)

    email_field = ft.TextField(
        value=current_email,
        label="Novo Email",
        border=ft.InputBorder.OUTLINE,
        bgcolor=None,
        border_color="#8B8E96",
        focused_border_color="#ECEFF1",
        height=45,
        text_size=14,
        content_padding=ft.Padding.symmetric(horizontal=15),
        cursor_color="#FFFFFF",
        prefix_icon="email_outlined",
        on_change=update_button_state,
    )

    def on_password_change(e):
        # Prevenir duplicação infinita ao colar
        if password_field.value and len(password_field.value) > 100:
            password_field.value = password_field.value[:100]
            page.update()
        update_button_state(e)
    
    def on_confirm_change(e):
        # Prevenir duplicação infinita ao colar
        if confirm_field.value and len(confirm_field.value) > 100:
            confirm_field.value = confirm_field.value[:100]
            page.update()
        update_button_state(e)

    password_field = ft.TextField(
        value="",
        label="Nova Senha",
        password=True,
        can_reveal_password=True,
        border=ft.InputBorder.OUTLINE,
        bgcolor=None,
        border_color="#8B8E96",
        focused_border_color="#ECEFF1",
        height=45,
        text_size=14,
        content_padding=ft.Padding.symmetric(horizontal=15),
        cursor_color="#FFFFFF",
        prefix_icon="lock_outline",
        on_change=on_password_change,
    )

    confirm_field = ft.TextField(
        value="",
        label="Confirmar Nova Senha",
        password=True,
        can_reveal_password=True,
        border=ft.InputBorder.OUTLINE,
        bgcolor=None,
        border_color="#8B8E96",
        focused_border_color="#ECEFF1",
        height=45,
        text_size=14,
        content_padding=ft.Padding.symmetric(horizontal=15),
        cursor_color="#FFFFFF",
        prefix_icon="lock_outline",
        on_change=on_confirm_change,
    )

    update_button = ft.ElevatedButton(
        content=ft.Text("Atualizar Credenciais", color="#050508", weight=ft.FontWeight.W_800),
        height=50,
        width=350,
        on_click=update_credentials_action,
        bgcolor="#607D8B",
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=8),
        ),
        disabled=True
    )

    loading_ring = ft.ProgressRing(visible=False, width=30, height=30, stroke_width=3, color="#FF6A00")

    back_to_login = ft.Row(
        controls=[
            ft.GestureDetector(
                content=ft.Row([
                    ft.Icon(ft.Icons.ARROW_BACK, color="#64748B", size=16),
                    ft.Text("Voltar para Login", size=14, color="#64748B", weight=ft.FontWeight.W_500),
                ], spacing=5),
                on_tap=lambda e: page.go("/login"),
                mouse_cursor=ft.MouseCursor.CLICK
            )
        ],
        alignment=ft.MainAxisAlignment.CENTER
    )

    form_container = ft.Container(
        width=400,
        padding=ft.Padding.symmetric(horizontal=50, vertical=30),
        bgcolor=None,
        content=ft.Column(
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=15,
            controls=[
                ft.Icon(ft.Icons.LOCK_RESET, color="#FF6A00", size=48),
                ft.Text("Alterar Credenciais", size=24, weight=ft.FontWeight.BOLD),
                ft.Text("Atualize seu email e senha", size=14, color="#8B8E96"),
                ft.Container(height=20),
                email_field,
                password_field,
                confirm_field,
                ft.Container(height=10),
                update_button,
                back_to_login,
                ft.Container(height=15),
                ft.Text(ref=error_text, color="#EF5350", size=14, text_align=ft.TextAlign.CENTER)
            ]
        )
    )

    def view():
        return ft.View(
            route="/change_password",
            controls=[
                ft.Container(
                    expand=True,
                    padding=0,
                    gradient=ft.LinearGradient(
                        begin=ft.Alignment(-1, -1),
                        end=ft.Alignment(1, 1),
                        colors=["#050508", "#0a0c14"]
                    ),
                    alignment=ft.alignment.Alignment(0, 0),
                    content=form_container
                )
            ],
            bgcolor=ft.Colors.TRANSPARENT,
            padding=0,
            spacing=0
        )

    page.views.clear()
    page.views.append(view())
    page.update()


# --- Tela de Processamento da Alteração de Senha ---
def password_changing_screen(page: ft.Page, email: str, password: str):
    """
    Tela que mostra o processamento da mudança de senha:
    1. Desconectando IAs
    2. Salvando novas credenciais
    3. Sucesso
    """
    logger.info("╔════════════════════════════════════════╗")
    logger.info("║  PROCESSANDO ALTERAÇÃO DE SENHA       ║")
    logger.info("╚════════════════════════════════════════╝")

    status_text = ft.Text(
        value="Desconectando todas as IAs...",
        size=20,
        color="#FFFFFF",
        text_align=ft.TextAlign.CENTER,
        weight=ft.FontWeight.W_400
    )
    
    status_icon = ft.ProgressRing(width=50, height=50, stroke_width=3, color="#FFFFFF")

    async def process_change():
        try:
            # Etapa 1: Desconectar bots
            await asyncio.sleep(0.5)
            
            try:
                import psutil
                logger.info("🔄 Desconectando TODAS as IAs ativas...")
                
                killed_count = 0
                pids_to_kill = []
                
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = proc.info.get('cmdline', [])
                        if cmdline and any('IA_Cod' in str(arg) for arg in cmdline):
                            pids_to_kill.append((proc.pid, proc))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                for pid, proc in pids_to_kill:
                    try:
                        proc.terminate()
                        killed_count += 1
                        logger.info(f"✅ IA terminada (PID: {pid})")
                    except:
                        try:
                            proc.kill()
                            logger.info(f"✅ IA forçada a fechar (PID: {pid})")
                        except:
                            pass
                
                await asyncio.sleep(1.5)
                
                # Segunda passada
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = proc.info.get('cmdline', [])
                        if cmdline and any('IA_Cod' in str(arg) for arg in cmdline):
                            proc.kill()
                    except:
                        pass
                
                # Atualizar status
                status_icon.visible = False
                if killed_count > 0:
                    status_text.value = f"✓ {killed_count} IA(s) desconectada(s)"
                else:
                    status_text.value = "ℹ Nenhuma IA ativa encontrada"
                    
                page.update()
                await asyncio.sleep(1)
                
            except Exception as ex:
                logger.warning(f"⚠️ Aviso ao desconectar IAs: {str(ex)}")
            
            # Etapa 2: Limpar variáveis de ambiente
            status_icon.visible = True
            status_text.value = "Limpando credenciais antigas..."
            page.update()
            await asyncio.sleep(0.5)
            
            import os
            for key in ["IQ_EMAIL", "IQ_PASSWORD", "IQ_PASS", "BULLUX_EMAIL", "BULLUX_PASS", "STRIPE_PRODUCT_ID"]:
                if key in os.environ:
                    del os.environ[key]
            
            # Remover STRIPE_PRODUCT_ID do arquivo .env para forçar re-check
            try:
                _env_path = get_user_env_path()
                if os.path.exists(_env_path):
                    with open(_env_path, 'r', encoding='utf-8') as f:
                        _lines = f.readlines()
                    _new_lines = [l for l in _lines if not l.startswith('STRIPE_PRODUCT_ID=')]
                    with open(_env_path, 'w', encoding='utf-8') as f:
                        f.writelines(_new_lines)
                    logger.info("🧹 STRIPE_PRODUCT_ID removido do .env")
            except Exception as _ex:
                logger.warning(f"⚠️ Falha ao remover STRIPE_PRODUCT_ID do .env: {_ex}")
            
            logger.info("🧹 Variáveis de ambiente antigas limpas")
            
            # Etapa 3: Salvar novas credenciais
            status_text.value = "Salvando novas credenciais..."
            page.update()
            await asyncio.sleep(0.5)
            
            save_credentials(email, password)
            logger.info("✅ Novas credenciais salvas")
            
            # Etapa 3.5: Verificar assinatura do novo email
            status_text.value = "Verificando assinatura..."
            page.update()
            try:
                _lang = load_language_from_file()
                is_paid, _reason = check_subscription(email, _lang)
                if is_paid:
                    logger.info(f"✅ Assinatura verificada para novo email: {email}")
                else:
                    logger.info(f"⚠️ Assinatura não encontrada para {email}: {_reason}")
            except Exception as _ex:
                logger.warning(f"⚠️ Falha ao verificar assinatura: {_ex}")
            await asyncio.sleep(0.5)
            
            # Etapa 4: Sucesso
            status_icon.visible = False
            status_text.value = "✓ Senha alterada com sucesso!"
            status_text.color = "#FFFFFF"
            page.update()
            await asyncio.sleep(1.5)
            
            # Redirecionar para login
            page.go("/login")
            
        except Exception as ex:
            logger.error(f"❌ Erro ao processar mudança: {str(ex)}")
            status_icon.visible = False
            status_text.value = f"✗ Erro: {str(ex)}"
            status_text.color = "#FFFFFF"
            page.update()
            await asyncio.sleep(3)
            page.go("/change_password")

    # Iniciar processamento automaticamente
    page.run_task(process_change)

    processing_view = ft.View(
        route="/password_changing",
        bgcolor=ft.Colors.TRANSPARENT,
        padding=0,
        spacing=0,
        controls=[
            ft.Container(
                content=ft.Column(
                    controls=[
                        status_icon,
                        ft.Container(height=20),
                        status_text
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=10
                ),
                alignment=ft.alignment.Alignment(0, 0),
                expand=True,
                padding=0,
                gradient=ft.LinearGradient(
                    begin=ft.alignment.Alignment(-1, -1),
                    end=ft.alignment.Alignment(1, 1),
                    colors=["#050508", "#0a0c14"]
                )
            )
        ]
    )

    page.views.clear()
    page.views.append(processing_view)
    page.update()


# --- Tela de Sucesso Alteração de Senha ---
def password_changed_success_screen(page: ft.Page):
    """Tela de sucesso após alteração de senha com efeito typewriter."""
    
    status_text = ft.Text(
        value="",
        size=28,
        weight=ft.FontWeight.BOLD,
        color="#FFFFFF",
        text_align=ft.TextAlign.CENTER
    )
    
    async def type_writer(text: str):
        """Efeito de digitação."""
        status_text.value = ""
        page.update()
        for char in text:
            status_text.value += char
            page.update()
            await asyncio.sleep(0.05)
        
        # Adicionar mensagem sobre reconexão
        await asyncio.sleep(1)
        reconnect_text = ft.Text(
            value="Você precisará conectar novamente com as novas credenciais.",
            size=16,
            color="#FFA500",
            text_align=ft.TextAlign.CENTER
        )
        success_view.controls[0].content.controls.append(reconnect_text)
        page.update()
        
        # Aguardar 3 segundos e voltar para login
        await asyncio.sleep(3)
        page.go("/login")
    
    success_view = ft.View(
        route="/password_changed_success",
        bgcolor=ft.Colors.TRANSPARENT,
        padding=0,
        spacing=0,
        controls=[
            ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Icon(ft.Icons.CHECK_CIRCLE_OUTLINE, color="#4CAF50", size=80),
                        ft.Container(height=20),
                        status_text
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=10
                ),
                alignment=ft.alignment.Alignment.CENTER,
                expand=True,
                padding=0,
                gradient=ft.LinearGradient(
                    begin=ft.Alignment(-1, -1),
                    end=ft.Alignment(1, 1),
                    colors=["#050508", "#0a0c14"]
                )
            )
        ]
    )
    
    page.views.clear()
    page.views.append(success_view)
    page.update()
    
    # Iniciar animação de digitação
    page.run_task(type_writer, "Sua senha foi alterada com sucesso!")


if __name__ == "__main__":

    ft.app(target=login_screen)