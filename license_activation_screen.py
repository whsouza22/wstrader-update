"""
Tela de Ativação de Licença - WS Trader
Interface para validação de chaves gratuitas
"""
import flet as ft
import logging
import os
import json
from license_manager import validate_license, get_hardware_id, get_machine_info

logger = logging.getLogger(__name__)

# Arquivo de preferências do usuário
USER_PREFS_FILE = os.path.join(os.path.expanduser("~"), ".wstrader", "preferences.json")

def load_language_from_file():
    """Carrega o idioma salvo do arquivo JSON"""
    try:
        if os.path.exists(USER_PREFS_FILE):
            with open(USER_PREFS_FILE, 'r', encoding='utf-8') as f:
                prefs = json.load(f)
                lang = prefs.get('language', 'PT')
                logger.info(f"[LICENSE] ✅ Idioma '{lang}' carregado de {USER_PREFS_FILE}")
                return lang
    except Exception as ex:
        logger.error(f"[LICENSE] ❌ Erro ao carregar idioma: {ex}")
    logger.info("[LICENSE] ✅ Usando idioma padrão: PT")
    return 'PT'

def get_license_translations(lang):
    """Retorna as traduções para a tela de licença"""
    translations = {
        "PT": {
            "title": "Ativação de Licença Gratuita",
            "label": "Chave de Licença Gratuita",
            "hint": "FREE-XXXXX-XXXXX-XXXXX",
            "button": "Ativar Licença",
            "info": "Esta é uma versão gratuita limitada.\nCada chave é única e vinculada a um único computador.\nA validação online é obrigatória a cada inicialização.",
            "hwid": "Hardware ID",
            "hwid_info": "(clique para mais info)",
            "system_info": "Informações do Sistema",
            "support_text": "Use essas informações ao entrar em contato com o suporte.",
            "close": "Fechar",
            "error_empty": "Digite uma chave de licença",
            "error_prefix": "Chave deve começar com FREE-",
            "error_format": "Formato inválido. Use: FREE-XXXXX-XXXXX-XXXXX",
            "error_length": "Cada segmento deve ter 5 caracteres",
            "success": "Licença ativada com sucesso!",
            "activation_info": "",
            "machine_info": "Informações da Máquina:"
        },
        "EN": {
            "title": "Free License Activation",
            "label": "Free License Key",
            "hint": "FREE-XXXXX-XXXXX-XXXXX",
            "button": "Activate License",
            "info": "This is a limited free version.\nEach key is unique and bound to a single computer.\nOnline validation is required at each startup.",
            "hwid": "Hardware ID",
            "hwid_info": "(click for more info)",
            "system_info": "System Information",
            "support_text": "Use this information when contacting support.",
            "close": "Close",
            "error_empty": "Enter a license key",
            "error_prefix": "Key must start with FREE-",
            "error_format": "Invalid format. Use: FREE-XXXXX-XXXXX-XXXXX",
            "error_length": "Each segment must have 5 characters",
            "success": "License activated successfully!",
            "activation_info": "",
            "machine_info": "Machine Information:"
        }
    }
    return translations.get(lang, translations["PT"])


def license_activation_screen(page: ft.Page, on_success_callback=None):
    """
    Tela de ativação de licença gratuita.

    Args:
        page: Página do Flet
        on_success_callback: Função a chamar quando ativação for bem-sucedida
    """

    # Carregar idioma
    selected_lang = load_language_from_file()
    t = get_license_translations(selected_lang)
    logger.info(f"✅ [LICENSE] Idioma selecionado: '{selected_lang}'")

    # Estado da tela
    is_loading = False

    # Elementos da UI
    license_key_input = ft.TextField(
        label=t["label"],
        hint_text=t["hint"],
        width=400,
        text_align=ft.TextAlign.CENTER,
        max_length=23,
        capitalization=ft.TextCapitalization.CHARACTERS,
        bgcolor="#2C2C2C",
        border_color="#FF681A",
        focused_border_color="#FF681A",
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
        visible=False
    )

    activate_button = ft.ElevatedButton(
        content=ft.Row(
            controls=[
                ft.Icon(ft.icons.VPN_KEY, color="#FFFFFF"),
                ft.Text(t["button"], color="#FFFFFF", weight=ft.FontWeight.BOLD),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=10,
        ),
        bgcolor="#FF681A",
        height=50,
        width=300,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=8),
        ),
        disabled=False
    )

    back_button = ft.TextButton(
        content=ft.Text(t.get("close", "Voltar"), color="#B0B0B0"),
        on_click=lambda e: page.go("/login")
    )

    hwid_text = ft.Text(
        "",
        size=10,
        color="#666666",
        text_align=ft.TextAlign.CENTER,
        selectable=True
    )

    info_text = ft.Text(
        t["info"],
        size=12,
        color="#B0B0B0",
        text_align=ft.TextAlign.CENTER,
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
        nonlocal is_loading
        is_loading = loading
        loading_indicator.visible = loading
        activate_button.disabled = loading
        license_key_input.disabled = loading
        page.update()

    def validate_key_format(key):
        """Valida formato da chave"""
        if not key:
            return False, t["error_empty"]

        key = key.strip().upper()

        # Formato esperado: FREE-XXXXX-XXXXX-XXXXX
        if not key.startswith("FREE-"):
            return False, t["error_prefix"]

        parts = key.split("-")
        if len(parts) != 4:
            return False, t["error_format"]

        if not all(len(p) == 5 or (i == 0 and p == "FREE") for i, p in enumerate(parts)):
            return False, t["error_length"]

        return True, key

    async def on_activate_click(e):
        """Handler do botão de ativação"""
        nonlocal is_loading

        if is_loading:
            return

        # Validar formato
        is_valid_format, result = validate_key_format(license_key_input.value)
        if not is_valid_format:
            show_status(f"❌ {result}", is_error=True)
            return

        license_key = result
        hide_status()
        set_loading(True)

        try:
            logger.info(f"Tentando ativar licença: {license_key[:10]}...")

            # Validar com o servidor
            is_valid, error_message, user_data = validate_license(license_key)

            if is_valid:
                # Sucesso!
                logger.info("✅ Licença ativada com sucesso!")

                show_status(
                    f"✅ {t['success']}",
                    is_error=False
                )

                # Salvar dados no client_storage
                try:
                    page.client_storage.set("license_valid", True)
                    page.client_storage.set("license_key", license_key)
                    page.client_storage.set("license_user_data", user_data)
                    page.client_storage.set("machine_info", get_machine_info())
                    page.client_storage.set("hwid", get_hardware_id())
                except Exception as storage_error:
                    logger.warning(f"Erro ao salvar no storage: {storage_error}")

                # Aguardar 2 segundos e chamar callback
                await ft.sleep(2)

                if on_success_callback:
                    on_success_callback(user_data)
            else:
                # Erro na validação
                logger.warning(f"❌ Erro na ativação: {error_message}")
                show_status(f"❌ {error_message}", is_error=True)

        except Exception as ex:
            logger.error(f"Erro ao ativar licença: {ex}", exc_info=True)
            show_status(f"❌ Erro inesperado: {str(ex)}", is_error=True)

        finally:
            set_loading(False)

    def on_key_input_change(e):
        """Handler de mudança no input da chave"""
        # Formatar automaticamente
        value = e.control.value.upper().replace(" ", "")

        # Adicionar hífens automaticamente
        if len(value) > 4 and "-" not in value[4:5]:
            parts = [value[i:i+5] for i in range(0, len(value), 5)]
            value = "-".join(parts)

        e.control.value = value
        hide_status()
        page.update()

    def show_hwid_info(e):
        """Mostra informações do HWID (para suporte)"""
        hwid = get_hardware_id()
        machine_info = get_machine_info()

        info_content = ft.Column(
            controls=[
                ft.Text(f"{t['hwid']} (HWID)", size=18, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                ft.Text(f"HWID: {hwid[:32]}...", size=12, selectable=True),
                ft.Divider(),
                ft.Text(t["machine_info"], size=14, weight=ft.FontWeight.BOLD),
            ],
            spacing=10,
            horizontal_alignment=ft.CrossAxisAlignment.START
        )

        for key, value in machine_info.items():
            info_content.controls.append(
                ft.Text(f"{key}: {value}", size=11, selectable=True)
            )

        info_content.controls.append(ft.Divider())
        info_content.controls.append(
            ft.Text(
                t["support_text"],
                size=10,
                color="#999999",
                italic=True
            )
        )

        dialog = ft.AlertDialog(
            title=ft.Text(t["system_info"]),
            content=info_content,
            actions=[
                ft.TextButton(t["close"], on_click=lambda e: close_dialog(dialog))
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        page.overlay.append(dialog)
        dialog.open = True
        page.update()

    def close_dialog(dialog):
        """Fecha um diálogo"""
        dialog.open = False
        page.update()

    # Configurar eventos
    activate_button.on_click = on_activate_click
    license_key_input.on_change = on_key_input_change

    # Atualizar HWID text
    hwid = get_hardware_id()
    hwid_text.value = f"{t['hwid']}: {hwid[:16]}... {t['hwid_info']}"

    hwid_button = ft.TextButton(
        content=ft.Row(
            controls=[
                ft.Icon(ft.icons.INFO_OUTLINE, size=16, color="#666666"),
                hwid_text
            ],
            spacing=5,
            alignment=ft.MainAxisAlignment.CENTER
        ),
        on_click=show_hwid_info
    )

    top_bar = ft.Container(
        height=40,
        padding=ft.padding.only(top=6, left=10, right=10),
        content=ft.Row(
            controls=[
                ft.IconButton(
                    icon=ft.Icons.ARROW_BACK,
                    icon_color="#B0B0B0",
                    icon_size=18,
                    on_click=lambda e: page.go("/login")
                ),
                ft.Container(expand=True),
                ft.IconButton(
                    icon=ft.Icons.HOME,
                    icon_color="#B0B0B0",
                    icon_size=18,
                    on_click=lambda e: page.go("/login")
                ),
                ft.IconButton(
                    icon=ft.Icons.CLOSE,
                    icon_color="#B0B0B0",
                    icon_size=18,
                    on_click=lambda e: page.go("/login")
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        ),
    )

    content_column = ft.Column(
        controls=[
            ft.Container(height=40),
            ft.Icon(
                ft.Icons.LOCK_PERSON,
                size=80,
                color="#FF681A"
            ),
            ft.Text(
                t["title"],
                size=28,
                weight=ft.FontWeight.BOLD,
                color="#FFFFFF"
            ),
            ft.Container(height=10),
            info_text,
            ft.Container(height=30),
            license_key_input,
            ft.Container(height=10),
            status_text,
            ft.Container(height=20),
            ft.Row(
                controls=[loading_indicator, activate_button],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=10
            ),
            ft.Container(height=8),
            back_button,
            ft.Container(height=30),
            hwid_button,
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=5
    )

    # Container principal
    main_container = ft.Container(
        content=ft.Column(
            controls=[
                top_bar,
                content_column,
            ],
            expand=True,
            spacing=0,
        ),
        alignment=ft.Alignment(0, 0),
        expand=True,
        gradient=ft.LinearGradient(
            begin=ft.Alignment(-1, -1),
            end=ft.Alignment(1, 1),
            colors=["#0E1114", "#111417"]
        )
    )

    # Limpar views e adicionar a tela de ativação
    page.views.clear()
    page.views.append(
        ft.View(
            route="/license_activation",
            bgcolor=ft.Colors.TRANSPARENT,
            padding=0,
            spacing=0,
            controls=[main_container]
        )
    )
    page.update()


def check_license_on_startup(page: ft.Page, on_valid_callback, on_invalid_callback):
    """
    Verifica a licença ao iniciar o aplicativo.

    Args:
        page: Página do Flet
        on_valid_callback: Função a chamar se a licença estiver válida
        on_invalid_callback: Função a chamar se a licença for inválida
    """
    try:
        # Tentar carregar licença do storage
        license_key = page.client_storage.get("license_key")

        if not license_key:
            logger.info("Nenhuma licença encontrada no storage")
            on_invalid_callback()
            return

        # Revalidar com o servidor
        logger.info("Revalidando licença...")
        is_valid, error_message, user_data = validate_license(license_key)

        if is_valid:
            logger.info("✅ Licença válida!")
            page.client_storage.set("license_valid", True)
            page.client_storage.set("license_user_data", user_data)
            on_valid_callback(user_data)
        else:
            logger.warning(f"❌ Licença inválida: {error_message}")
            page.client_storage.set("license_valid", False)
            on_invalid_callback(error_message)

    except Exception as e:
        logger.error(f"Erro ao verificar licença: {e}", exc_info=True)
        on_invalid_callback(f"Erro: {str(e)}")
