import flet as ft
import logging
import json

logger = logging.getLogger(__name__)

def tutorial_screen(page: ft.Page):
    """
    Tela de tutorial interativo com navegação lateral (carousel)
    mostrando o passo a passo para usar o WS Trader
    """
    
    # Carregar logo do WS Trader
    import os
    import sys
    
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
        img_dir = os.path.join(sys._MEIPASS, "img") if hasattr(sys, '_MEIPASS') else os.path.join(base_dir, "img")
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        img_dir = os.path.join(base_dir, "img")
    
    logo_path = os.path.join(img_dir, "logo.png")
    logo_exists = os.path.exists(logo_path)
    
    # Estado atual da página do tutorial
    current_page_index = {"value": 0}
    
    # Traduções
    translations = {
        "PT": {
            "title": "Como Usar o WS Trader",
            "skip": "Pular Tutorial",
            "next": "Próximo",
            "previous": "Anterior",
            "finish": "Começar",
            "page_indicator": "Página {current} de {total}",
            
            # Página 1 - Cadastro Stripe
            "step1_title": "1. Cadastre-se e Teste Grátis",
            "step1_text": (
                "Experimente 7 DIAS GRÁTIS antes de pagar!\n\n"
                "Como funciona:\n"
                "1. Clique no botão 'Comprar agora' abaixo\n"
                "2. Seu navegador abrirá a página de cadastro do Stripe\n"
                "3. Preencha seus dados (email, senha e cartão)\n"
                "4. Você terá 7 dias grátis para testar tudo!\n\n"
                "DICA IMPORTANTE: Use o MESMO email e senha\n"
                "que você usa na sua corretora (IQ Option ou Bullex)\n"
                "Isso facilita o login automático!"
            ),
            
            # Página 2 - Login
            "step2_title": "2. Faça Login no Sistema",
            "step2_text": (
                "Após criar sua conta no Stripe:\n\n"
                "• Clique em 'Entrar' na tela inicial\n"
                "• Use o MESMO email e senha do Stripe\n"
                "• O sistema faz a ponte entre você e a\n"
                "  corretora para operar de forma inteligente\n\n"
                "🔒 SEGURANÇA E PRIVACIDADE:\n"
                "• Sua senha fica SOMENTE no seu computador\n"
                "• Nós NÃO armazenamos senhas em servidores\n"
                "• Nós NÃO temos acesso à sua conta\n"
                "• Tudo é 100% local e privado"
            ),
            
            # Página 3 - Treinamento do Sistema
            "step3_title": "3. Treine o Sistema",
            "step3_text": (
                "O WS Trader já vem otimizado para operar:\n\n"
                "• Conecte sua conta à corretora\n"
                "• Ative o modo DEMO para treinar o sistema\n"
                "• A IA analisa suportes, resistências e tendências\n"
                "• O sistema aprende com cada operação\n\n"
                "DICA: Deixe o sistema operar em DEMO por algumas\n"
                "horas para calibrar com o mercado atual"
            ),
            
            # Página 4 - Chat e Otimização
            "step4_title": "4. Otimize seu Desempenho",
            "step4_text": (
                "Use o chat para melhorar seus resultados:\n\n"
                "• Acesse o chat na dashboard\n"
                "• Faça perguntas como:\n"
                "  - 'Qual o melhor horário para operar?'\n"
                "  - 'Como melhorar minha taxa de acerto?'\n"
                "  - 'Quais ativos estão performando melhor?'\n\n"
                "O sistema analisará seus dados e dará\n"
                "recomendações personalizadas!"
            ),
        },
        "EN": {
            "title": "How to Use WS Trader",
            "skip": "Skip Tutorial",
            "next": "Next",
            "previous": "Previous",
            "finish": "Get Started",
            "page_indicator": "Page {current} of {total}",
            
            # Page 1 - Stripe Registration
            "step1_title": "1. Sign Up and Try Free",
            "step1_text": (
                "Try 7 DAYS FREE before paying!\n\n"
                "How it works:\n"
                "1. Click the 'Buy now' button below\n"
                "2. Your browser will open Stripe's sign-up page\n"
                "3. Fill in your details (email, password, and card)\n"
                "4. You'll have 7 free days to test everything!\n\n"
                "IMPORTANT TIP: Use the SAME email and password\n"
                "you use on your broker (IQ Option or Bullex)\n"
                "This makes automatic login easier!"
            ),
            
            # Page 2 - Login
            "step2_title": "2. Login to the System",
            "step2_text": (
                "After creating your Stripe account:\n\n"
                "• Click 'Log in' on the home screen\n"
                "• Use the SAME email and password from Stripe\n"
                "• The system bridges you to the broker\n"
                "  for intelligent automated trading\n\n"
                "🔒 SECURITY & PRIVACY:\n"
                "• Your password stays ONLY on your computer\n"
                "• We do NOT store passwords on servers\n"
                "• We do NOT have access to your account\n"
                "• Everything is 100% local and private"
            ),
            
            # Page 3 - Training
            "step3_title": "3. Train the System",
            "step3_text": (
                "WS Trader comes optimized and ready to trade:\n\n"
                "• Connect your account to the broker\n"
                "• Activate DEMO mode to train the system\n"
                "• AI analyzes supports, resistances and trends\n"
                "• The system learns from each operation\n\n"
                "TIP: Let the system trade on DEMO for a few\n"
                "hours to calibrate with current market"
            ),
            
            # Page 4 - Chat and Optimization
            "step4_title": "4. Optimize Your Performance",
            "step4_text": (
                "Use the chat to improve your results:\n\n"
                "• Access the chat on the dashboard\n"
                "• Ask questions like:\n"
                "  - 'What's the best time to trade?'\n"
                "  - 'How to improve my win rate?'\n"
                "  - 'Which assets are performing better?'\n\n"
                "The system will analyze your data and give\n"
                "personalized recommendations!"
            ),
        }
    }
    
    # Detectar idioma
    def load_language_from_file():
        try:
            prefs_file = os.path.join(os.path.expanduser("~"), ".wstrader", "preferences.json")
            if os.path.exists(prefs_file):
                with open(prefs_file, "r", encoding="utf-8") as f:
                    prefs = json.load(f)
                    return prefs.get("language", "PT")
        except Exception:
            pass
        return "PT"

    lang = load_language_from_file()
    try:
        if hasattr(page, 'client_storage'):
            saved_lang = page.client_storage.get("lang")
            if saved_lang in translations:
                lang = saved_lang
    except Exception:
        pass
    
    t = translations[lang]
    
    # Conteúdo das páginas
    tutorial_pages = [
        {
            "title": t["step1_title"],
            "text": t["step1_text"],
            "image": "tutorial_1.png"  # Imagem para o passo 1
        },
        {
            "title": t["step2_title"],
            "text": t["step2_text"],
            "image": "tutorial_2.png"  # Imagem para o passo 2
        },
        {
            "title": t["step3_title"],
            "text": t["step3_text"],
            "image": "tutorial_3.png"  # Imagem para o passo 3
        },
        {
            "title": t["step4_title"],
            "text": t["step4_text"],
            "image": "tutorial_4.png"  # Imagem para o passo 4
        }
    ]
    
    # Controles de UI
    page_title = ft.Text(
        tutorial_pages[0]["title"],
        size=42,
        weight=ft.FontWeight.BOLD,
        color="#FFFFFF",
        text_align=ft.TextAlign.LEFT,
        font_family="Segoe UI"
    )
    
    page_text = ft.Text(
        tutorial_pages[0]["text"],
        size=17,
        color="#F0F2F5",
        text_align=ft.TextAlign.LEFT,
        width=560,
        font_family="Segoe UI"
    )
    
    # Caminho da imagem
    current_image_path = os.path.join(img_dir, tutorial_pages[0]["image"])
    image_exists = os.path.exists(current_image_path)
    
    page_image = ft.Image(
        src=current_image_path if image_exists else None,
        width=520,
        height=520,
        fit="contain"
    ) if image_exists else ft.Container(
        width=520,
        height=520
    )
    
    page_indicator = ft.Text(
        t["page_indicator"].format(current=1, total=len(tutorial_pages)),
        size=12,
        color="#CBD5E1",
        text_align=ft.TextAlign.CENTER
    )
    
    # Indicadores visuais (dots)
    def create_dots():
        dots = []
        for i in range(len(tutorial_pages)):
            dot = ft.Container(
                width=10 if i == current_page_index["value"] else 7,
                height=10 if i == current_page_index["value"] else 7,
                bgcolor="#FFFFFF" if i == current_page_index["value"] else "#4B5563",
                border_radius=5,
                animate=ft.Animation(250, ft.AnimationCurve.EASE_IN_OUT)
            )
            dots.append(dot)
        return ft.Row(
            controls=dots,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=8
        )
    
    dots_container = ft.Container(content=create_dots())
    
    def update_page_content():
        """Atualiza o conteúdo da página atual"""
        idx = current_page_index["value"]
        current = tutorial_pages[idx]
        
        page_title.value = current["title"]
        page_text.value = current["text"]
        
        # Atualizar imagem (sem card/container)
        image_path = os.path.join(img_dir, current["image"])
        if os.path.exists(image_path):
            page_image.src = image_path
            page_image.width = 520
            page_image.height = 520
        
        page_indicator.value = t["page_indicator"].format(
            current=idx + 1,
            total=len(tutorial_pages)
        )
        
        # Atualizar dots
        dots_container.content = create_dots()
        
        # Atualizar botões
        previous_button.visible = idx > 0
        next_button.visible = idx < len(tutorial_pages) - 1
        finish_button.visible = idx == len(tutorial_pages) - 1
        
        page.update()
    
    def go_to_next(e):
        """Vai para a próxima página"""
        if current_page_index["value"] < len(tutorial_pages) - 1:
            current_page_index["value"] += 1
            update_page_content()
    
    def go_to_previous(e):
        """Volta para a página anterior"""
        if current_page_index["value"] > 0:
            current_page_index["value"] -= 1
            update_page_content()

    def _navigate(route: str):
        """Navegação compatível com diferentes versões do Flet"""
        try:
            page.route = route
            if hasattr(page, "on_route_change") and page.on_route_change:
                page.on_route_change(None)
            page.update()
        except Exception:
            pass
    
    def skip_tutorial(e):
        """Pula o tutorial e vai para home"""
        logger.info("Usuário pulou o tutorial")
        _navigate("/")
    
    def finish_tutorial(e):
        """Finaliza o tutorial e vai para login"""
        logger.info("Tutorial concluído")
        # Marcar tutorial como visto
        try:
            if hasattr(page, 'client_storage'):
                page.client_storage.set("tutorial_completed", True)
        except:
            pass
        _navigate("/login")
    
    # Botões de navegação
    previous_button = ft.OutlinedButton(
        content=ft.Row([
            ft.Text("◀", color="#FFFFFF", size=16),
            ft.Text(t["previous"], color="#FFFFFF", size=14, weight=ft.FontWeight.W_500)
        ], spacing=8, alignment=ft.MainAxisAlignment.CENTER),
        on_click=go_to_previous,
        visible=False,
        height=48,
        width=140,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=10),
            side=ft.BorderSide(1.5, "#4B5563"),
            padding=ft.Padding(16, 10, 16, 10)
        )
    )
    
    next_button = ft.ElevatedButton(
        content=ft.Row([
            ft.Text(t["next"], color="#FFFFFF", size=14, weight=ft.FontWeight.W_500),
            ft.Text("▶", color="#FFFFFF", size=16)
        ], spacing=8, alignment=ft.MainAxisAlignment.CENTER),
        on_click=go_to_next,
        bgcolor="#F97316",
        color="#FFFFFF",
        height=48,
        width=140,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=10),
            padding=ft.Padding(18, 10, 18, 10)
        )
    )
    
    finish_button = ft.ElevatedButton(
        content=ft.Row([
            ft.Text(t["finish"], color="#FFFFFF", size=15, weight=ft.FontWeight.BOLD),
            ft.Text("✓", color="#FFFFFF", size=20)
        ], spacing=8, alignment=ft.MainAxisAlignment.CENTER),
        on_click=finish_tutorial,
        bgcolor="#F97316",
        color="#FFFFFF",
        visible=False,
        height=52,
        width=180,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=12),
            padding=ft.Padding(20, 12, 20, 12)
        )
    )
    
    skip_button = ft.OutlinedButton(
        content=ft.Text(t["skip"], color="#FFFFFF", size=14, weight=ft.FontWeight.W_500),
        on_click=skip_tutorial,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=8),
            side=ft.BorderSide(1.5, "#4B5563"),
            padding=ft.Padding(18, 10, 18, 10)
        )
    )
    
    # Bloco de texto (sem fundo/contorno, igual ao fundo da tela)
    text_block = ft.Container(
        content=ft.Column(
            controls=[
                page_title,
                ft.Container(height=16),
                page_text,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.START,
            spacing=0
        ),
        padding=ft.Padding(0, 0, 0, 0),
        height=400
    )
    
    # Layout principal
    content_row = ft.Row(
        controls=[
            # Lado esquerdo - Texto
            ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Container(expand=True),  # Spacer superior
                        text_block,
                        ft.Container(height=60),
                        dots_container,
                        ft.Container(expand=True),  # Spacer inferior
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.START,
                    spacing=0
                ),
                padding=ft.Padding(80, 40, 40, 40),
                expand=True
            ),
            # Lado direito - Imagem
            ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Container(expand=True),  # Spacer superior
                        page_image,
                        ft.Container(expand=True),  # Spacer inferior
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=0
                ),
                padding=ft.Padding(20, 20, 40, 20),
                alignment=ft.Alignment(0, 0)
            )
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        expand=True
    )
    
    # Botões de navegação na parte inferior
    navigation_buttons = ft.Container(
        content=ft.Row(
            controls=[
                previous_button,
                ft.Container(expand=True),  # Spacer
                next_button,
                finish_button
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            spacing=20
        ),
        padding=ft.Padding(40, 0, 40, 30),
        height=100
    )
    
    # Header com logo, título e botão de pular
    logo_section = ft.Row(
        controls=[
            ft.Image(src=logo_path, width=40, height=40) if logo_exists else ft.Container(width=40, height=40),
            ft.Text(
                t["title"],
                size=20,
                weight=ft.FontWeight.BOLD,
                color="#FFFFFF"
            )
        ],
        spacing=10
    )
    
    header = ft.Container(
        content=ft.Row(
            controls=[
                logo_section,
                skip_button
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN
        ),
        padding=ft.Padding(30, 20, 30, 20)
    )
    
    # Container principal
    main_container = ft.Container(
        content=ft.Column(
            controls=[
                header,
                content_row,
                navigation_buttons
            ],
            spacing=0,
            expand=True
        ),
        gradient=ft.LinearGradient(
            begin=ft.Alignment(-1, -1),
            end=ft.Alignment(1, 1),
            colors=["#0E1114", "#111417"]
        ),
        expand=True,
        padding=0
    )
    
    # Criar a view
    tutorial_view = ft.View(
        route="/tutorial",
        bgcolor=ft.Colors.TRANSPARENT,
        padding=0,
        spacing=0,
        controls=[main_container]
    )
    
    page.views.clear()
    page.views.append(tutorial_view)
    page.update()
    
    logger.info("Tutorial screen carregado com sucesso")
