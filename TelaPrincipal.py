import os
import sys

# ===== FIX: google-api-core trava no OneDrive (packages_distributions) =====
# O google.api_core.__init__ chama check_python_version() que usa
# importlib.metadata.packages_distributions() — esta função enumera TODOS os
# arquivos de todos os pacotes instalados e chama path.exists() em cada um.
# No OneDrive isso trava porque precisa verificar arquivos na nuvem.
# Solução: substituir packages_distributions() por versão rápida ANTES de
# qualquer import do google.api_core.
if sys.version_info >= (3, 10):
    import importlib.metadata as _md
    _md.packages_distributions = lambda: {}

import flet as ft
import webbrowser
import logging
import urllib.parse
import requests
import subprocess
import json
import asyncio
import importlib
import socket
import ctypes
import atexit
import time
import re

# ===== FORÇAR ÍCONE DA TASKBAR NO WINDOWS =====
# O Flet/Flutter usa AppUserModelID do Flutter por padrão,
# fazendo a taskbar mostrar o ícone do Flet ao invés do WsTrader.
try:
    myappid = 'wstrader.plataforma.1.0'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except Exception:
    pass

# Arquivo para salvar preferências do usuário (incluindo idioma)
USER_PREFS_FILE = os.path.join(os.path.expanduser("~"), ".wstrader", "preferences.json")

def save_language(lang):
    """Salva o idioma escolhido em um arquivo JSON"""
    try:
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(USER_PREFS_FILE), exist_ok=True)

        # Ler preferências existentes ou criar novo
        prefs = {}
        if os.path.exists(USER_PREFS_FILE):
            with open(USER_PREFS_FILE, 'r', encoding='utf-8') as f:
                prefs = json.load(f)

        # Atualizar idioma
        prefs['language'] = lang

        # Salvar
        with open(USER_PREFS_FILE, 'w', encoding='utf-8') as f:
            json.dump(prefs, f, indent=2)

        logging.info(f"✅ Idioma '{lang}' salvo em {USER_PREFS_FILE}")
        return True
    except Exception as ex:
        logging.error(f"❌ Erro ao salvar idioma: {ex}")
        return False

def load_language():
    """Carrega o idioma salvo do arquivo JSON"""
    try:
        if os.path.exists(USER_PREFS_FILE):
            with open(USER_PREFS_FILE, 'r', encoding='utf-8') as f:
                prefs = json.load(f)
                lang = prefs.get('language', 'PT')
                logging.info(f"✅ Idioma '{lang}' carregado de {USER_PREFS_FILE}")
                return lang
    except Exception as ex:
        logging.error(f"❌ Erro ao carregar idioma: {ex}")

    logging.info("✅ Usando idioma padrão: PT")
    return 'PT'

RUN_BOT_MODE = "--run-bot" in sys.argv

# =========================
# Single instance lock
# =========================
_single_instance_socket = None

def _release_single_instance_lock():
    global _single_instance_socket
    try:
        if _single_instance_socket is not None:
            _single_instance_socket.close()
    except Exception:
        pass
    _single_instance_socket = None

def acquire_single_instance_lock():
    """Impede abrir mais de uma instância do app."""
    global _single_instance_socket
    if _single_instance_socket is not None:
        return True

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except Exception:
        pass
    try:
        s.bind(("127.0.0.1", 37291))
        s.listen(1)
        _single_instance_socket = s
        atexit.register(_release_single_instance_lock)
        return True
    except OSError:
        try:
            # Se não houver ninguém escutando de verdade, tenta novamente
            test = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test.settimeout(0.2)
            test.connect(("127.0.0.1", 37291))
            test.close()
        except Exception:
            try:
                time.sleep(0.2)
                s.bind(("127.0.0.1", 37291))
                s.listen(1)
                _single_instance_socket = s
                atexit.register(_release_single_instance_lock)
                return True
            except Exception:
                pass
        try:
            ctypes.windll.user32.MessageBoxW(0, "O WS Trader já está aberto.", "WS Trader", 0x10)
        except Exception:
            pass
        return False

if not RUN_BOT_MODE:
    try:
        from Login_Screen import login_screen, show_auth_and_process
        from trading_bot import bot_dashboard
        from chat_screen_new import chat_screen
        from tutorial_screen import tutorial_screen
    except ModuleNotFoundError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from Login_Screen import login_screen, show_auth_and_process
        from trading_bot import bot_dashboard
        from chat_screen_new import chat_screen
        from tutorial_screen import tutorial_screen
else:
    login_screen = None
    show_auth_and_process = None
    bot_dashboard = None
    chat_screen = None
    tutorial_screen = None


trading_bot = bot_dashboard if bot_dashboard is not None else None

# Suprimir logs de depuração do Flet
logging.getLogger('flet_core').setLevel(logging.WARNING)
logging.getLogger('flet').setLevel(logging.WARNING)

# Configurar o logging da aplicação - Apenas WARNINGS e ERRORS para o usuário final
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Caminho para o ícone da janela
WINDOW_ICON_FILENAME = "ws_ai_trader_corrigido.ico"

# Versão padrão do aplicativo
CURRENT_VERSION = "5.2" # H&S + warm-up 5min + guard preço longe + horário 6-18h

# URL do JSON para verificação de atualizações
VERSION_URL = "https://whsouza22.github.io/wstrader-update/version.json"
# GitHub Releases API - busca sempre a versão mais recente publicada
GITHUB_RELEASES_API = "https://api.github.com/repos/whsouza22/wstrader-update/releases/latest"

async def check_for_update(page: ft.Page, lang: str = "PT", status_column: ft.Column = None, progress_bar_control: ft.ProgressBar = None):
    """
    Verifica se há uma nova versão do aplicativo disponível e gerencia o processo de atualização.
    Exibe o status na UI de splash screen.
    """
    translations = {
        "PT": {
            "connecting": "Conectando ao servidor...",
            "connected": "Conectado. Verificando atualizações...",
            "update_found_prompt": "Nova versão encontrada! 🎉\nClique em 'Atualizar Agora' para instalar a versão {version}.",
            "updating_start_download": "🚀 Iniciando download da nova versão...",
            "update_progress_download": "⬇️ Baixando componentes: {progress:.1f}%",
            "update_success": "Download concluído! Por favor, feche este aplicativo para completar a atualização.\nReabra o app após a instalação. O instalador está na sua pasta de Downloads.",
            "closing_in": "Fechando em {seconds}...", # Nova tradução para o contador
            "update_done": "Versão {version} está atualizada.",
            "update_error": "Erro ao verificar/atualizar: {error}",
            "invalid_json": "JSON de atualização inválido ou inacessível. Status HTTP: {status}, Conteúdo: {content}",
            "version_label": "Versão {version}",
            "no_windows_update": "⚠️ Atualização automática apenas para Windows.",
            "close_app": "Fechar Aplicativo" # Mantido para compatibilidade, mas não usado diretamente
        },
        "EN": {
            "connecting": "Connecting to server...",
            "connected": "Connected. Checking for updates...",
            "update_found_prompt": "New version found! 🎉\nClick 'Update Now' to install version {version}.",
            "updating_start_download": "🚀 Starting new version download...",
            "update_progress_download": "⬇️ Downloading components: {progress:.1f}%",
            "update_success": "Download complete! Please close this application to complete the update.\nReopen the app after installation. The installer is in your Downloads folder.",
            "closing_in": "Closing in {seconds}...",
            "update_done": "Version {version} is up to date.",
            "update_error": "Error checking/updating: {error}",
            "invalid_json": "Invalid or inaccessible update JSON. HTTP Status: {status}, Content: {content}",
            "version_label": "Version {version}",
            "no_windows_update": "⚠️ Automatic update only for Windows.",
            "close_app": "Close Application"
        }
    }
    t = translations[lang]

    status_text_control = status_column.controls[0]
    retry_button = {"ref": None}

    async def update_splash_ui(text_message: str, show_progress_bar: bool = True, progress_value: float = None, show_button: bool = False, update_button_obj=None):
        """
        Atualiza os elementos da UI da splash screen (texto, barra de progresso, botão).
        """
        status_text_control.value = text_message
        
        progress_bar_control.visible = show_progress_bar
        if progress_value is not None:
            progress_bar_control.value = progress_value
        else:
            progress_bar_control.value = None

        if update_button_obj:
            if show_button and update_button_obj not in status_column.controls:
                status_column.controls.insert(2, update_button_obj) 
            elif not show_button and update_button_obj in status_column.controls:
                status_column.controls.remove(update_button_obj)
        
        page.update()
        await asyncio.sleep(0.01) # Cede o controle ao loop de eventos para renderização

    async def perform_update_action(latest_version, changelog, download_link, update_button_ref):
        """
        Atualização via PowerShell em background:
        1. Baixa o instalador silenciosamente (rápido, sem travar UI)
        2. Fecha o app
        3. Instala e reinicia
        """
        import tempfile

        logger.info(f"Iniciando atualizacao para versao: {latest_version}")

        temp_dir = os.path.join(tempfile.gettempdir(), "wstrader_update")
        os.makedirs(temp_dir, exist_ok=True)
        installer_path = os.path.join(temp_dir, "WsTrader_Update.exe")
        install_dir = os.environ.get('PROGRAMFILES', 'C:\\Program Files') + '\\WsTrader'
        app_exe = os.path.join(install_dir, 'WsTrader.exe')

        # Script PowerShell que faz TUDO: baixa, instala, reinicia
        ps_script = os.path.join(temp_dir, "wstrader_update.ps1")
        ps_content = f'''
$ErrorActionPreference = "SilentlyContinue"
# Baixar instalador
try {{
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri "{download_link}" -OutFile "{installer_path}" -UseBasicParsing
}} catch {{
    # Fallback para WebClient (mais rápido)
    (New-Object Net.WebClient).DownloadFile("{download_link}", "{installer_path}")
}}
# Matar app
Start-Sleep -Seconds 2
taskkill /F /IM WsTrader.exe /T 2>$null
taskkill /F /IM python.exe /T 2>$null
Start-Sleep -Seconds 2
# Instalar silenciosamente
& "{installer_path}" /S
Start-Sleep -Seconds 8
# Reiniciar se existe
if (Test-Path "{app_exe}") {{ Start-Process "{app_exe}" }}
# Limpar
Remove-Item "{ps_script}" -Force 2>$null
'''
        with open(ps_script, 'w', encoding='utf-8') as f:
            f.write(ps_content)

        # Lançar PowerShell em background (não espera, não bloqueia)
        subprocess.Popen(
            ['powershell', '-ExecutionPolicy', 'Bypass', '-WindowStyle', 'Hidden', '-File', ps_script],
            creationflags=0x08000000  # CREATE_NO_WINDOW
        )

        logger.info("Script de atualizacao lancado, fechando app...")

        # FECHAR IGUAL AO BOTÃO X: destroy() fecha a janela Flutter + Python
        try:
            await page.window.destroy()
        except Exception:
            pass

        # Garantia: se destroy() não matou tudo, força
        await asyncio.sleep(1.0)
        try:
            import ctypes
            ctypes.windll.kernel32.TerminateProcess(
                ctypes.windll.kernel32.GetCurrentProcess(), 0
            )
        except Exception:
            os._exit(0)

    async def handle_retry_click(e):
        e.control.disabled = True
        page.update()
        result = await check_for_update(page, lang, status_column, progress_bar_control)
        if result:
            page.go("/")
        else:
            # Re-enable button after failed retry
            e.control.disabled = False
            page.update()

    def _parse_version(v: str):
        parts = re.findall(r"\d+", v or "")
        return tuple(int(p) for p in parts) if parts else (0,)

    def _normalize_url(url: str) -> str:
        if not url:
            return url
        cleaned = url.replace("\n", "").replace(" ", "").strip()
        if cleaned.startswith("https//"):
            cleaned = cleaned.replace("https//", "https://", 1)
        if cleaned.startswith("http//"):
            cleaned = cleaned.replace("http//", "http://", 1)
        return cleaned

    try:
        await update_splash_ui(t["connecting"], show_progress_bar=True, progress_value=None, show_button=False)
        await asyncio.sleep(0.3) 

        await update_splash_ui(t["connected"], show_progress_bar=True, progress_value=None, show_button=False)
        await asyncio.sleep(0.2)

        # Executar request em thread separada para não bloquear a UI (evita "Working...")
        def _fetch_latest_version():
            """Busca a versão mais recente de TODAS as fontes e retorna a maior."""
            best_version = None
            best_download = None
            best_changelog = ""

            # === FONTE 1: version.json (GitHub Pages) ===
            try:
                for attempt in range(3):
                    try:
                        resp = requests.get(VERSION_URL, timeout=5)
                        resp.raise_for_status()
                        data = resp.json()
                        break
                    except Exception:
                        if attempt == 2:
                            data = {}
                        import time
                        time.sleep(1)

                # Suporta formato com array de versões
                if "versions" in data and isinstance(data["versions"], list):
                    for v_entry in data["versions"]:
                        v_str = v_entry.get("version", "")
                        v_url = v_entry.get("download_url") or v_entry.get("download_link") or v_entry.get("installer_url", "")
                        if v_str and v_url:
                            v_parsed = _parse_version(v_str)
                            if best_version is None or v_parsed > _parse_version(best_version):
                                best_version = v_str
                                best_download = v_url
                                best_changelog = v_entry.get("changelog", "")
                
                # Suporta formato flat (atual)
                if "version" in data:
                    v_str = data["version"]
                    v_url = data.get("download_url") or data.get("download_link") or data.get("installer_url", "")
                    if v_str and v_url:
                        v_parsed = _parse_version(v_str)
                        if best_version is None or v_parsed > _parse_version(best_version):
                            best_version = v_str
                            best_download = v_url
                            best_changelog = data.get("changelog", "")

            except Exception as e:
                logger.warning(f"Erro ao buscar version.json: {e}")

            # === FONTE 2: GitHub Releases API (sempre tem a última release) ===
            try:
                gh_resp = requests.get(GITHUB_RELEASES_API, timeout=10, headers={"Accept": "application/vnd.github.v3+json"})
                if gh_resp.status_code == 200:
                    gh_data = gh_resp.json()
                    gh_tag = gh_data.get("tag_name", "")
                    gh_version = gh_tag.lstrip("vV")  # Remove prefixo "v" ou "V"
                    
                    if gh_version:
                        gh_parsed = _parse_version(gh_version)
                        if best_version is None or gh_parsed > _parse_version(best_version):
                            # Busca URL do .exe nos assets da release
                            for asset in gh_data.get("assets", []):
                                name = asset.get("name", "")
                                if name.lower().endswith(".exe"):
                                    best_version = gh_version
                                    best_download = asset["browser_download_url"]
                                    best_changelog = gh_data.get("body", "")
                                    logger.info(f"GitHub Releases: encontrada v{gh_version} (mais recente)")
                                    break
            except Exception as e:
                logger.warning(f"Erro ao buscar GitHub Releases: {e}")

            return best_version, best_download, best_changelog

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _fetch_latest_version)
        latest_version, download_link, changelog = result

        download_link = _normalize_url(download_link) if download_link else None
        
        if not latest_version or not download_link:
            logger.warning("Nenhuma versão válida encontrada no servidor — continuando com versão atual")
            await update_splash_ui(
                t["update_done"].format(version=CURRENT_VERSION),
                show_progress_bar=False, progress_value=0.0, show_button=False
            )
            await asyncio.sleep(1.5)
            return True

        logger.info(f"Versão mais recente encontrada: {latest_version} (atual: {CURRENT_VERSION})")

        current_app_version = CURRENT_VERSION 

        if _parse_version(latest_version) > _parse_version(current_app_version):

            # Atualiza UI para mostrar que encontrou nova versao
            await update_splash_ui(
                f"Nova versao {latest_version} encontrada! Atualizando automaticamente...",
                show_progress_bar=True,
                progress_value=None,
                show_button=False
            )
            await asyncio.sleep(1.5)

            # Inicia atualizacao automatica sem precisar de clique
            await perform_update_action(latest_version, changelog, download_link, None)
            return False

        else:
            await update_splash_ui(t["update_done"].format(version=current_app_version), show_progress_bar=False, progress_value=0.0, show_button=False)
            await asyncio.sleep(1.5)
            return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Erro de rede ao verificar atualização: {str(e)}", exc_info=True)
        if not retry_button["ref"]:
            retry_button["ref"] = ft.Button(
                content=ft.Text("Tentar novamente", color="#FFFFFF"),
                on_click=handle_retry_click,
                bgcolor="#FF9800",
                width=200,
                height=38,
                style=ft.ButtonStyle(
                    padding=ft.Padding(16, 8, 16, 8),
                ),
            )
        else:
            # Update color and re-enable for network errors
            retry_button["ref"].bgcolor = "#FF9800"
            retry_button["ref"].disabled = False
        await update_splash_ui(t["update_error"].format(error="Erro de rede."), show_progress_bar=False, progress_value=0.0, show_button=True, update_button_obj=retry_button["ref"])
        await asyncio.sleep(3.0)
        return False
    except json.JSONDecodeError as e:
        logger.error(f"Erro ao decodificar JSON de atualização: {str(e)}", exc_info=True)
        if not retry_button["ref"]:
            retry_button["ref"] = ft.Button(
                content=ft.Text("Tentar novamente", color="#FFFFFF"),
                on_click=handle_retry_click,
                bgcolor="#2196F3",
                width=200,
                height=38,
                style=ft.ButtonStyle(
                    padding=ft.Padding(16, 8, 16, 8),
                ),
            )
        else:
            retry_button["ref"].bgcolor = "#2196F3"
            retry_button["ref"].disabled = False
        await update_splash_ui(t["update_error"].format(error="JSON de atualização inválido."), show_progress_bar=False, progress_value=0.0, show_button=True, update_button_obj=retry_button["ref"])
        await asyncio.sleep(3.0)
        return False
    except ValueError as e:
        logger.error(f"Erro na validação do JSON: {str(e)}", exc_info=True)
        if not retry_button["ref"]:
            retry_button["ref"] = ft.Button(
                content=ft.Text("Tentar novamente", color="#FFFFFF"),
                on_click=handle_retry_click,
                bgcolor="#2196F3",
                width=200,
                height=38,
                style=ft.ButtonStyle(
                    padding=ft.Padding(16, 8, 16, 8),
                ),
            )
        else:
            retry_button["ref"].bgcolor = "#2196F3"
            retry_button["ref"].disabled = False
        await update_splash_ui(t["update_error"].format(error=str(e)), show_progress_bar=False, progress_value=0.0, show_button=True, update_button_obj=retry_button["ref"])
        await asyncio.sleep(3.0)
        return False
    except Exception as e:
        logger.error(f"Erro inesperado durante a verificação de atualização: {str(e)}", exc_info=True)
        if not retry_button["ref"]:
            retry_button["ref"] = ft.Button(
                content=ft.Text("Tentar novamente", color="#FFFFFF"),
                on_click=handle_retry_click,
                bgcolor="#2196F3",
                width=200,
                height=38,
                style=ft.ButtonStyle(
                    padding=ft.Padding(16, 8, 16, 8),
                ),
            )
        else:
            retry_button["ref"].bgcolor = "#2196F3"
            retry_button["ref"].disabled = False
        await update_splash_ui(t["update_error"].format(error="Erro inesperado."), show_progress_bar=False, progress_value=0.0, show_button=True, update_button_obj=retry_button["ref"])
        await asyncio.sleep(3.0)
        return False

async def main(page: ft.Page):

    # Configurar ícone e janela
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
        img_dir = os.path.join(sys._MEIPASS, "img") if hasattr(sys, '_MEIPASS') else os.path.join(base_dir, "img")
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        img_dir = os.path.join(base_dir, "img")

    window_icon_abs_path = os.path.join(img_dir, WINDOW_ICON_FILENAME)

    # Definir ícone da janela
    if os.path.exists(window_icon_abs_path):
        page.window.icon = window_icon_abs_path
    else:
        logger.error(f"Ícone não encontrado: {window_icon_abs_path}")

    # Agora configura o resto da janela
    page.title = "WS Trader - Plataforma Inteligente"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#050508"

    # Configurações de janela (sintaxe alternativa para garantir travamento)
    page.window.width = 1200
    page.window.height = 750
    page.window.resizable = False
    page.window.maximizable = False
    page.window.minimizable = True

    page.padding = 0
    page.spacing = 0

    # ===== CLEANUP DE SUBPROCESSOS AO FECHAR =====
    def kill_all_bot_processes():
        """Mata todos os subprocessos de bots que podem estar rodando.
        Tenta psutil primeiro (preciso), depois fallback com taskkill (Windows)."""
        # 1) Tentar psutil (melhor método)
        try:
            import psutil
            current_pid = os.getpid()
            current_process = psutil.Process(current_pid)
            children = current_process.children(recursive=True)
            for child in reversed(children):  # mata filhos primeiro
                try:
                    child.kill()
                except Exception:
                    pass
            return  # psutil funcionou
        except Exception:
            pass

        # 2) Fallback: matar pela árvore de processos do Windows (funciona sem psutil)
        try:
            pid = os.getpid()
            subprocess.call(
                ['taskkill', '/F', '/T', '/PID', str(pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
            )
        except Exception:
            pass

    # ===== CLEANUP AO FECHAR (via atexit — simples e confiável) =====
    def _cleanup_on_exit():
        try:
            kill_all_bot_processes()
        except Exception:
            pass
        # Limpar apenas __pycache__ — memória da IA e credenciais são preservados
        try:
            import shutil
            base = os.path.dirname(os.path.abspath(__file__))
            pycache = os.path.join(base, "__pycache__")
            if os.path.isdir(pycache):
                try:
                    shutil.rmtree(pycache, ignore_errors=True)
                except Exception:
                    pass
            logger.info("🧹 Cleanup ao fechar (processos + pycache)")
        except Exception:
            pass
    atexit.register(_cleanup_on_exit)
    # =================================================================

    # Caminhos das outras imagens
    logo_abs_path = os.path.join(img_dir, "logo.png")
    logo_splash_abs_path = os.path.join(img_dir, "Wstrader_bot.png")
    bot_abs_path = os.path.join(img_dir, "Bot.png")
    brasil_flag_abs_path = os.path.join(img_dir, "brasil.png")
    america_flag_abs_path = os.path.join(img_dir, "america.png")

    logo_path = logo_abs_path if os.path.exists(logo_abs_path) else None
    logo_splash_path = logo_splash_abs_path if os.path.exists(logo_splash_abs_path) else None
    bot_path = bot_abs_path if os.path.exists(bot_abs_path) else None
    brasil_flag_path = brasil_flag_abs_path if os.path.exists(brasil_flag_abs_path) else None
    america_flag_path = america_flag_abs_path if os.path.exists(america_flag_abs_path) else None
    window_icon_path = window_icon_abs_path if os.path.exists(window_icon_abs_path) else None

    if not logo_splash_path:
        logo_splash_path = logo_path

    # ===================== SPLASH SCREEN MODERNA E LIMPA =====================

    loading_status_text = ft.Text(
        "Inicializando...",
        color="#8B8E96",
        size=13,
        text_align=ft.TextAlign.CENTER,
        weight=ft.FontWeight.W_500,
        opacity=1.0
    )

    loading_progress_bar = ft.ProgressBar(
        width=280,
        color="#FF6A00",
        bgcolor="#1a1f2e",
        value=None,
        bar_height=4,
    )

    # Imagem do bot limpa sem glow
    bot_image_container = ft.Container(
        content=ft.Image(src=logo_splash_path, width=200, height=200, opacity=0.95) if logo_splash_path else ft.Icon(ft.Icons.SMART_TOY, size=100, color="#FF6A00"),
        width=200,
        height=200,
        border_radius=100,
    )

    # Titulo elegante moderno
    title_text = ft.Text(
        "WS TRADER AI",
        size=32,
        weight=ft.FontWeight.W_800,
        color="#E8ECF4",
        text_align=ft.TextAlign.CENTER,
    )

    subtitle_text = ft.Text(
        "Inteligência Artificial para Trading",
        size=14,
        color="#8B8E96",
        opacity=0.90,
        text_align=ft.TextAlign.CENTER,
    )

    # Coluna de status - layout moderno
    loading_status_column = ft.Column(
        controls=[
            title_text,
            subtitle_text,
            ft.Container(height=35),
            ft.Container(
                content=loading_progress_bar,
                width=280,
                alignment=ft.Alignment(0, 0),
            ),
            ft.Container(height=12),
            loading_status_text,
        ],
        spacing=8,
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER
    )

    # View principal do loading - moderna e limpa
    loading_view = ft.View(
        route="/loading",
        bgcolor=ft.Colors.TRANSPARENT,
        padding=0,
        spacing=0,
        controls=[
            ft.Container(
                content=ft.Stack(
                    controls=[
                        # Fundo preto grafite
                        ft.Container(
                            expand=True,
                            gradient=ft.LinearGradient(
                                begin=ft.Alignment(-1, -1),
                                end=ft.Alignment(1, 1),
                                colors=["#050508", "#0a0c14"]
                            )
                        ),
                        # Conteudo central
                        ft.Container(
                            content=ft.Column(
                                controls=[
                                    bot_image_container,
                                    ft.Container(height=30),
                                    loading_status_column,
                                ],
                                alignment=ft.MainAxisAlignment.CENTER,
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                            alignment=ft.Alignment(0, 0),
                            expand=True,
                        ),
                        # Versão no rodapé com estilo moderno
                        ft.Container(
                            content=ft.Text(f"v{CURRENT_VERSION}", size=11, color="#8B8E96", opacity=0.7, weight=ft.FontWeight.W_400),
                            alignment=ft.Alignment(0, 1),
                            padding=ft.Padding.only(bottom=24),
                        ),
                    ],
                ),
                expand=True,
            )
        ]
    )

    page.views.append(loading_view)
    page.route = "/loading"
    page.update()

    # Iniciar backend em background
    async def init_backend_async():
        """Inicializa o backend em background"""
        if getattr(sys, 'frozen', False):
            backend_server_path = sys._MEIPASS
        else:
            backend_server_path = os.path.dirname(os.path.abspath(__file__))

        if backend_server_path not in sys.path:
            sys.path.insert(0, backend_server_path)

        try:
            from backend_server import start_backend_server
            start_backend_server()

            for i in range(10):
                try:
                    response = requests.get("http://127.0.0.1:8000/health", timeout=0.3)
                    if response.status_code == 200:
                        return
                except:
                    pass
                await asyncio.sleep(1)

            logger.warning("Backend não respondeu após 10s")

        except Exception as e:
            import traceback
            error_file = os.path.join(os.path.expanduser("~"), "wstrader_error.txt")
            with open(error_file, "w", encoding="utf-8") as f:
                f.write(f"ERRO BACKEND:\n{str(e)}\n\n{traceback.format_exc()}")
            logger.error(f"❌ Erro backend: {error_file}")

    # Iniciar backend sem bloquear
    page.run_task(init_backend_async)

    translations = {
        "PT": {
            "eyebrow": "— IA DE ÚLTIMA GERAÇÃO",
            "title_part1": "IA que ",
            "title_accent": "opera sozinha",
            "title_part2": " e evolui com o tempo.",
            "para_pre": "Conecte, defina sua meta e deixe a ",
            "para_bold": "IA WSTrader",
            "para_post": " trabalhar por você.",
            "buy": "Comprar agora",
            "contact": "Suporte",
            "stripe": "Gerenciar Conta",
            "login": "Entrar",
            "url_error": "⚠️ Erro ao processar os parâmetros da URL: {error}",
            "support_title": "Suporte WS Trader",
            "support_message": "Entre em contato com o suporte\natravés do email: wstrader@wstrader.onmicrosoft.com",
            "close_button": "Fechar",
            "navigation_error": "⚠️ Falha ao navegar para a tela da IA. Verifique os logs para mais detalhes.",
            "config_error": "⚠️ Credenciais não configuradas. Por favor, cadastre suas credenciais.",
            "version_label": "Versão {version}"
        },
        "EN": {
            "eyebrow": "— NEXT-GEN AI",
            "title_part1": "AI that ",
            "title_accent": "trades on its own",
            "title_part2": " and evolves over time.",
            "para_pre": "Connect, set your target and let ",
            "para_bold": "WSTrader AI",
            "para_post": " work for you.",
            "buy": "Buy now",
            "contact": "Support",
            "stripe": "Manage Account",
            "login": "Log in",
            "url_error": "⚠️ Error processing URL parameters: {error}",
            "support_title": "WS Trader Support",
            "support_message": "Contact support via email:\nwstrader@wstrader.onmicrosoft.com",
            "close_button": "Close",
            "navigation_error": "⚠️ Failed to navigate to the AI screen. Check the logs for more details.",
            "config_error": "⚠️ Credentials not configured. Please register your credentials.",
            "version_label": "Version {version}"
        }
    }
    default_lang = load_language()

    try:
        if hasattr(page, 'client_storage'):
            page.client_storage.set("lang", default_lang)
    except Exception:
        pass

    version_text = ft.Text(
        translations[default_lang]["version_label"].format(version=CURRENT_VERSION),
        size=12,
        color="#8B8E96",
        text_align=ft.TextAlign.RIGHT
    )


    def go_to_payment(e):
        webbrowser.open_new("https://wstrader.io/pricing.html")

    def go_to_stripe_portal(e):
        webbrowser.open_new("https://billing.stripe.com/p/login/00g6oQcPN8tl3VC9AA")

    def go_to_login(e):
        page.route = "/login"
        route_change(None)

    def show_contact_dialog(e):
        webbrowser.open_new("https://t.me/WftraderFlow_bot")

    header_stripe = ft.GestureDetector(
        content=ft.Text(translations[default_lang]["stripe"], color="#E8ECF4", size=14),
        on_tap=go_to_stripe_portal,
        mouse_cursor="click"
    )
    header_contact = ft.GestureDetector(
        content=ft.Text(translations[default_lang]["contact"], color="#E8ECF4", size=14),
        on_tap=show_contact_dialog,
        mouse_cursor="click"
    )
    
    def go_to_tutorial(e):
        page.route = "/tutorial"
        route_change(None)
    
    header_tutorial = ft.GestureDetector(
        content=ft.Text("Tutorial" if default_lang == "PT" else "Tutorial", color="#E8ECF4", size=14),
        on_tap=go_to_tutorial,
        mouse_cursor="click"
    )
    header_login = ft.GestureDetector(
        content=ft.Text(translations[default_lang]["login"], color="#E8ECF4", size=14),
        on_tap=go_to_login,
        mouse_cursor="click"
    )
    eyebrow_text = ft.Text(
        spans=[ft.TextSpan(
            translations[default_lang]["eyebrow"],
            style=ft.TextStyle(size=12, weight=ft.FontWeight.W_600, color="#FF9D2B", letter_spacing=2),
        )],
    )
    main_title = ft.Text(
        spans=[
            ft.TextSpan(translations[default_lang]["title_part1"],
                        style=ft.TextStyle(size=48, weight=ft.FontWeight.W_800, color="#E8ECF4")),
            ft.TextSpan(translations[default_lang]["title_accent"],
                        style=ft.TextStyle(size=48, weight=ft.FontWeight.W_800, color="#FF6A00")),
            ft.TextSpan(translations[default_lang]["title_part2"],
                        style=ft.TextStyle(size=48, weight=ft.FontWeight.W_800, color="#E8ECF4")),
        ],
    )
    main_paragraph = ft.Text(
        spans=[
            ft.TextSpan(translations[default_lang]["para_pre"],
                        style=ft.TextStyle(size=17, color="#8B8E96")),
            ft.TextSpan(translations[default_lang]["para_bold"],
                        style=ft.TextStyle(size=17, weight=ft.FontWeight.BOLD, color="#E8ECF4")),
            ft.TextSpan(translations[default_lang]["para_post"],
                        style=ft.TextStyle(size=17, color="#8B8E96")),
        ],
        width=520,
    )

    buy_button = ft.Button(
        content=ft.Text(translations[default_lang]["buy"], color="#050508", weight=ft.FontWeight.W_800),
        on_click=go_to_payment,
        bgcolor="#FF6A00",
        height=45,
        width=180,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=8),
        )
    )

    def update_language(e):
        lang = e.control.value if e else default_lang

        try:
            save_language(lang)
        except Exception:
            pass

        try:
            if hasattr(page, 'client_storage'):
                page.client_storage.set("lang", lang)
        except Exception:
            pass

        try:
            lang = str(lang).strip().upper()
            if lang not in translations:
                lang = "PT"

            t = translations[lang]
            eyebrow_text.spans[0].text = t["eyebrow"]
            main_title.spans[0].text = t["title_part1"]
            main_title.spans[1].text = t["title_accent"]
            main_title.spans[2].text = t["title_part2"]
            main_paragraph.spans[0].text = t["para_pre"]
            main_paragraph.spans[1].text = t["para_bold"]
            main_paragraph.spans[2].text = t["para_post"]
            buy_button.content.value = t["buy"]
            header_stripe.content.value = t["stripe"]
            header_contact.content.value = t["contact"]
            header_login.content.value = t["login"]
            version_text.value = t["version_label"].format(version=CURRENT_VERSION)
            page.update()
        except Exception as ex:
            logger.error(f"Erro ao atualizar idioma: {ex}")

    # Variável para controlar idioma selecionado
    current_language = {"value": default_lang}

    # Helper: cada ft.Image precisa de instância única (Flet 0.80 exige parent único)
    def _flag_img(src_path, w=24, h=16):
        """Cria uma nova instância de ft.Image para a bandeira (cada controle só pode ter 1 pai)."""
        if src_path:
            return ft.Image(src=src_path, width=w, height=h, fit="cover")
        return None

    # Botão texto inicial
    language_text = ft.Text("PT" if default_lang == "PT" else "EN", color="#E8ECF4", size=14)

    def change_to_pt(_):
        current_language["value"] = "PT"
        save_language("PT")

        class FakeControl:
            value = "PT"
        class FakeEvent:
            control = FakeControl()

        language_text.value = "PT"
        new_img = _flag_img(brasil_flag_path)
        if new_img:
            language_button_row.controls[0] = new_img
        page.update()
        update_language(FakeEvent())

    def change_to_en(_):
        current_language["value"] = "EN"
        save_language("EN")

        class FakeControl:
            value = "EN"
        class FakeEvent:
            control = FakeControl()

        language_text.value = "EN"
        new_img = _flag_img(america_flag_path)
        if new_img:
            language_button_row.controls[0] = new_img
        page.update()
        update_language(FakeEvent())

    # Row que contém bandeira + texto (instância própria da imagem)
    _btn_flag = _flag_img(brasil_flag_path) if default_lang == "PT" else _flag_img(america_flag_path)
    language_button_row = ft.Row(
        controls=[
            _btn_flag if _btn_flag else ft.Text("🌐", size=20),
            language_text
        ],
        spacing=8
    )

    # Criar PopupMenuButton com imagens de bandeiras (instâncias próprias)
    language_selector = ft.PopupMenuButton(
        content=ft.Row(
            controls=[
                language_button_row,
                ft.Icon(ft.Icons.ARROW_DROP_DOWN, color="#E8ECF4")
            ],
            spacing=5
        ),
        items=[
            ft.PopupMenuItem(
                content=ft.Row([
                    _flag_img(brasil_flag_path) or ft.Text("🇧🇷", size=20),
                    ft.Text("PT - Português", color="#E8ECF4")
                ], spacing=8),
                on_click=change_to_pt
            ),
            ft.PopupMenuItem(
                content=ft.Row([
                    _flag_img(america_flag_path) or ft.Text("🇺🇸", size=20),
                    ft.Text("EN - English", color="#E8ECF4")
                ], spacing=8),
                on_click=change_to_en
            )
        ],
        bgcolor="#0e1018"
    )

    logger.info(f"Language selector criado")

    logo_content = ft.Image(src=logo_path, width=40, height=40) if logo_path else ft.Text("Logo não encontrado", color="#FF0000")

    logo_section = ft.Row(
        controls=[
            logo_content,
            ft.Text("WS Trader", size=20, weight="bold", color="#E8ECF4")
        ],
        spacing=10
    )

    header = ft.Container(
        content=ft.Row(
            controls=[
                logo_section,
                ft.Row(
                    controls=[header_stripe, header_contact, header_tutorial, header_login, language_selector],
                    spacing=25,
                    alignment="end"
                )
            ],
            alignment="spaceBetween"
        ),
        padding=ft.Padding.symmetric(horizontal=30, vertical=20)
    )

    main_text_column = ft.Column(
        controls=[eyebrow_text, ft.Container(height=4), main_title, main_paragraph, ft.Container(height=8), buy_button],
        spacing=10,
        alignment="center",
        horizontal_alignment="start"
    )

    bot_content = ft.Image(src=bot_path, width=420, fit="contain") if bot_path else ft.Text("Imagem Bot não encontrada", color="#FF0000")

    content = ft.Row(
        controls=[
            ft.Container(content=main_text_column, padding=ft.Padding.only(left=80), expand=True),
            ft.Container(content=bot_content, padding=ft.Padding.only(right=40))
        ],
        vertical_alignment="center",
        alignment="spaceBetween",
        expand=True
    )

    footer = ft.Container(
        content=version_text,
        padding=ft.Padding.only(bottom=10, right=30),
        alignment=ft.Alignment(1, 1)
    )

    def home_view():
        logger.info("Rendering home view")
        return ft.View(
            route="/",
            bgcolor=ft.Colors.TRANSPARENT,
            padding=0,
            spacing=0,
            controls=[
                ft.Container(
                    content=ft.Column(controls=[header, content, footer], expand=True, spacing=0),
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

    def route_change(route):
        page.views.clear()

        if page.route == "/tutorial":
            page.views.append(ft.View(route="/tutorial", controls=[]))
            tutorial_screen(page)
        elif page.route == "/login":
            page.views.append(ft.View(route="/login", controls=[]))
            login_screen(page)
        elif page.route == "/change_password":
            page.views.append(ft.View(route="/change_password", controls=[]))
            from Login_Screen import change_password_screen
            change_password_screen(page)
        elif page.route.startswith("/password_changing"):
            _parsed = urllib.parse.urlparse(page.route)
            _params = urllib.parse.parse_qs(_parsed.query)
            _chg_email = _params.get("email", [""])[0]
            _chg_pass = _params.get("password", [""])[0]
            from Login_Screen import password_changing_screen
            password_changing_screen(page, _chg_email, _chg_pass)
        elif page.route == "/password_changed_success":
            page.views.append(ft.View(route="/password_changed_success", controls=[]))
            from Login_Screen import password_changed_success_screen
            password_changed_success_screen(page)
        elif page.route.startswith("/authenticating"):
            page.views.append(ft.View(route="/authenticating", controls=[]))
            show_auth_and_process(page)
        elif page.route.startswith("/chat"):
            try:
                from Login_Screen import load_credentials
                env_credentials = load_credentials()
                email = env_credentials.get("iq_email", "")
                password = env_credentials.get("iq_password", "")

                parsed_route = urllib.parse.urlparse(page.route)
                params = urllib.parse.parse_qs(parsed_route.query)
                email = email or params.get("email", [""])[0]
                password = password or params.get("password", [""])[0]

                if not email or not password:
                    logger.error("Credenciais não configuradas para chat.")
                    page.views.append(ft.View(route="/login", controls=[]))
                    login_screen(page)
                    page.snack_bar = ft.SnackBar(
                        content=ft.Text(
                            translations[default_lang]["config_error"],
                            color="#FFFFFF"
                        ),
                        bgcolor="#FF0000",
                        duration=5000
                    )
                    page.snack_bar.open = True
                    return

                logger.info(f"Chat: email={email}")
                page.views.append(ft.View(route="/chat", controls=[]))
                chat_screen(page, email, password)
            except Exception as e:
                logger.error(f"Error processing /chat route: {str(e)}", exc_info=True)
                page.views.append(ft.View(route="/login", controls=[]))
                login_screen(page)
                page.snack_bar = ft.SnackBar(
                    content=ft.Text(
                        translations[default_lang]["url_error"].format(error=str(e)),
                        color="#FFFFFF"
                    ),
                    bgcolor="#FF0000",
                    duration=5000
                )
                page.snack_bar.open = True
        elif page.route.startswith("/bot"):
            broker = "IQ Option"
            email = ""
            password = ""
            balance = 0.0
            bot_token = ""
            try:
                from Login_Screen import load_credentials
                env_credentials = load_credentials()
                email = env_credentials.get("iq_email", "")
                password = env_credentials.get("iq_password", "")

                parsed_route = urllib.parse.urlparse(page.route)
                params = urllib.parse.parse_qs(parsed_route.query)
                email = email or params.get("email", [""])[0]
                password = password or params.get("password", [""])[0]
                bot_token = params.get("bot_token", [""])[0]

                if not email or not password:
                    logger.error("Credenciais não configuradas.")
                    page.views.append(ft.View(route="/login", controls=[]))
                    login_screen(page)
                    page.snack_bar = ft.SnackBar(
                        content=ft.Text(
                            translations[default_lang]["config_error"],
                            color="#FFFFFF"
                        ),
                        bgcolor="#FF0000",
                        duration=5000
                    )
                    page.snack_bar.open = True
                    return

                logger.info(f"Dashboard: email={email}")
                page.views.append(ft.View(route="/bot", controls=[]))
                trading_bot(page, broker, email, password, balance, bot_token)
            except Exception as e:
                logger.error(f"Error processing /bot route: {str(e)}", exc_info=True)
                page.views.append(ft.View(route="/login", controls=[]))
                login_screen(page)
                page.snack_bar = ft.SnackBar(
                    content=ft.Text(
                        translations[default_lang]["url_error"].format(error=str(e)),
                        color="#FFFFFF"
                    ),
                    bgcolor="#FF0000",
                    duration=5000
                )

                page.snack_bar.open = True
        else:
            page.views.append(home_view())
        page.update()

    # Registrar o handler de mudança de rota
    page.on_route_change = route_change

    # Verificar atualização
    should_go_home = await check_for_update(page, "PT", loading_status_column, loading_progress_bar)

    # Navegar para home após verificação
    if should_go_home:
        page.views.clear()
        page.route = "/"
        page.views.append(home_view())
        page.update()
    else:
        # Se check_for_update retornou False (erro de rede, etc), navegar para home
        # A atualização em si fecha o app com os._exit(0), então esse else
        # só é alcançado em caso de erro - não deixar o app preso no splash
        await asyncio.sleep(2.0)
        page.views.clear()
        page.route = "/"
        page.views.append(home_view())
        page.update()

async def async_main(page: ft.Page):
    await main(page)

def run_bot_from_cli(argv):
    if "--run-bot" not in argv:
        return None

    # ========== RECONFIGURAR LOGGING PARA MODO BOT ==========
    # O basicConfig do TelaPrincipal configura level=WARNING,
    # mas o bot precisa de INFO para enviar SALDO/META via stderr.
    # basicConfig() só funciona 1x, então precisamos reconfigurar manualmente.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers:
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] [WS_AUTO_AI] %(message)s'))
    # ===========================================================

    idx = argv.index("--run-bot")
    if idx + 1 >= len(argv):
        logger.error("Missing broker after --run-bot")
        return 2

    broker_key = argv[idx + 1].strip().lower()
    
    # WS_AUTO_AI_BULLEX.py é o módulo principal unificado (S/R + IA ENSEMBLE)
    # Detecta o broker via BROKER_TYPE env var automaticamente
    os.environ["BROKER_TYPE"] = broker_key
    module_name = "WS_AUTO_AI_BULLEX"
    
    if not module_name:
        logger.error("Unknown broker key: %s", broker_key)
        return 2

    try:
        # No modo frozen, adicionar _internal ao path para encontrar os módulos IA
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
            internal_dir = os.path.join(base_dir, '_internal')
            if internal_dir not in sys.path:
                sys.path.insert(0, internal_dir)
        
        module = importlib.import_module(module_name)
    except Exception:
        logger.exception("Failed to import bot module: %s", module_name)
        return 1

    if not hasattr(module, "main"):
        logger.error("Bot module has no main(): %s", module_name)
        return 1

    try:
        module.main()
    except getattr(module, 'MetaAtingidaException', type(None)):
        logger.info("Bot encerrado: meta/stop atingido para %s", broker_key)
        return 0  # Saída normal - meta atingida
    except Exception:
        logger.exception("Bot execution failed: %s", broker_key)
        return 1

    return 0
if __name__ == "__main__":
    exit_code = run_bot_from_cli(sys.argv)
    if exit_code is not None:
        sys.exit(exit_code)

    if not acquire_single_instance_lock():
        sys.exit(0)

    # Define o diretório de assets para o Flet encontrar o ícone
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ft.run(async_main, view=ft.AppView.FLET_APP, assets_dir=base_dir)
