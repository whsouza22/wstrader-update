# -*- coding: utf-8 -*-
"""
Dashboard do Bot de Trading - WS Trader
Interface limpa e simplificada com AUTO-REFRESH em tempo real
"""
import flet as ft
import logging
import threading
import asyncio
import os
import json
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Arquivo de preferências do usuário
USER_PREFS_FILE = os.path.join(os.path.expanduser("~"), ".wstrader", "preferences.json")

# ===================== PERSISTÊNCIA DIÁRIA =====================
DAILY_DATA_DIR = os.path.join(os.path.expanduser("~"), ".wstrader", "daily_data")

def _get_daily_file(broker: str = "iq_option") -> str:
    """Retorna caminho do arquivo de dados do dia para a corretora"""
    os.makedirs(DAILY_DATA_DIR, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    broker_key = broker.lower().replace(" ", "_")
    return os.path.join(DAILY_DATA_DIR, f"{broker_key}_{today}.json")

def load_daily_data(broker: str = "iq_option") -> dict:
    """Carrega dados do dia (operações, wins, losses, profit)"""
    fpath = _get_daily_file(broker)
    try:
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as ex:
        logger.error(f"Erro ao carregar dados diários: {ex}")
    return {"wins": 0, "losses": 0, "profit": 0.0, "operations": [], "broker": broker}

def save_daily_data(data: dict, broker: str = "iq_option"):
    """Salva dados do dia"""
    fpath = _get_daily_file(broker)
    try:
        os.makedirs(DAILY_DATA_DIR, exist_ok=True)
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except Exception as ex:
        logger.error(f"Erro ao salvar dados diários: {ex}")

def load_weekly_data() -> dict:
    """Carrega dados dos últimos 7 dias de todas as corretoras"""
    result = {}
    brokers = ["iq_option", "bullex", "casatrader"]
    try:
        if not os.path.exists(DAILY_DATA_DIR):
            return {b: [] for b in brokers}
        for broker in brokers:
            broker_data = []
            for i in range(7):
                day = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                fpath = os.path.join(DAILY_DATA_DIR, f"{broker}_{day}.json")
                if os.path.exists(fpath):
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            d = json.load(f)
                            broker_data.append({"date": day, "profit": d.get("profit", 0.0),
                                                "wins": d.get("wins", 0), "losses": d.get("losses", 0)})
                    except Exception:
                        broker_data.append({"date": day, "profit": 0.0, "wins": 0, "losses": 0})
                else:
                    broker_data.append({"date": day, "profit": 0.0, "wins": 0, "losses": 0})
            result[broker] = list(reversed(broker_data))  # mais antigo primeiro
    except Exception:
        result = {b: [] for b in brokers}
    return result

def load_language_from_file():
    """Carrega o idioma salvo do arquivo JSON"""
    try:
        if os.path.exists(USER_PREFS_FILE):
            with open(USER_PREFS_FILE, 'r', encoding='utf-8') as f:
                prefs = json.load(f)
                lang = prefs.get('language', 'PT')
                logger.info(f"[BOT] ✅ Idioma '{lang}' carregado de {USER_PREFS_FILE}")
                return lang
    except Exception as ex:
        logger.error(f"[BOT] ❌ Erro ao carregar idioma: {ex}")
    logger.info("[BOT] ✅ Usando idioma padrão: PT")
    return 'PT'

def get_bot_translations(lang):
    """Retorna as traduções para a tela da IA"""
    translations = {
        "PT": {
            "balance": "Saldo",
            "stopped": "IA Pausada",
            "running": "IA Operando",
            "broker": "Corretora",
            "account_type": "Tipo de Conta",
            "demo": "Demo (Treino)",
            "real": "Real",
            "start": "INICIAR IA",
            "stop": "PARAR IA",
            "settings": "Configurações",
            "wins": "Vitórias:",
            "losses": "Derrotas:",
            "rate": "Taxa:",
            "profit": "Lucro:",
            "exit": "Sair",
            "history": "Histórico de Operações",
            "empty": "Nenhuma operação realizada",
            "datetime": "Data/Hora",
            "pair": "Par",
            "direction": "Direção",
            "value": "Valor",
            "payout": "Payout",
            "result": "Resultado",
            "executing": "Em execução",
            "win": "WIN",
            "loss": "LOSS",
            "unavailable": "Indisponível",
            "stop_bot_first": "Pare a IA antes de sair!"
            ,"connection_failed": "Erro: conexão falhou"
            ,"ai_error": "Erro no sistema"
            ,"start_error": "Erro ao iniciar"
            ,"report": "Relatório"
            ,"accumulated": "Ganho Acumulado por Corretora"
            ,"weekly": "Ganhos Diários - Última Semana"
            ,"no_data": "Sem dados"
            ,"ai_learning": "Nível da IA"
            ,"ai_phase_warmup": "Iniciante"
            ,"ai_phase_bayes": "Intermediário"
            ,"ai_phase_lgbm": "Avançado"
            ,"ai_phase_full": "Expert"
            ,"ai_phase_desc_warmup": "IA aprendendo os padrões do mercado.\nPrecisa de 15 operações para evoluir."
            ,"ai_phase_desc_bayes": "IA filtrando sinais fracos.\nPrecisa de 30 operações para evoluir."
            ,"ai_phase_desc_lgbm": "IA otimizando decisões.\nPrecisa de 50+ operações para nível máximo."
            ,"ai_phase_desc_full": "IA calibrada com precisão máxima!\nTodos os módulos ativos."
            ,"ai_trades_label": "Operações acumuladas"
        },
        "EN": {
            "balance": "Balance",
            "stopped": "AI Paused",
            "running": "AI Operating",
            "broker": "Broker",
            "account_type": "Account Type",
            "demo": "Demo (Practice)",
            "real": "Real",
            "start": "START AI",
            "stop": "STOP AI",
            "settings": "Settings",
            "wins": "Wins:",
            "losses": "Losses:",
            "rate": "Rate:",
            "profit": "Profit:",
            "exit": "Exit",
            "history": "Operations History",
            "empty": "No operations performed",
            "datetime": "Date/Time",
            "pair": "Pair",
            "direction": "Direction",
            "value": "Value",
            "payout": "Payout",
            "result": "Result",
            "executing": "Executing",
            "win": "WIN",
            "loss": "LOSS",
            "unavailable": "Unavailable",
            "stop_bot_first": "Stop the AI before exiting!"
            ,"connection_failed": "Error: connection failed"
            ,"ai_error": "System error"
            ,"start_error": "Error starting"
            ,"report": "Report"
            ,"accumulated": "Accumulated Gain by Broker"
            ,"weekly": "Daily Gains - Last Week"
            ,"no_data": "No data"
            ,"ai_learning": "AI Level"
            ,"ai_phase_warmup": "Beginner"
            ,"ai_phase_bayes": "Intermediate"
            ,"ai_phase_lgbm": "Advanced"
            ,"ai_phase_full": "Expert"
            ,"ai_phase_desc_warmup": "AI learning market patterns.\nNeeds 15 operations to evolve."
            ,"ai_phase_desc_bayes": "AI filtering weak signals.\nNeeds 30 operations to evolve."
            ,"ai_phase_desc_lgbm": "AI optimizing decisions.\nNeeds 50+ operations for max level."
            ,"ai_phase_desc_full": "AI calibrated with maximum precision!\nAll modules active."
            ,"ai_trades_label": "Accumulated operations"
        }
    }
    return translations.get(lang, translations["PT"])

# Variável global para controlar a thread do bot
bot_thread = None
bot_should_stop = False
auto_refresh_task = None  # Task assíncrona para auto-refresh

def bot_dashboard(page: ft.Page, broker: str, email: str, password: str, balance: float, bot_token: str):
    """
    Dashboard do bot com seleção de corretora e tipo de conta

    Args:
        page: Página do Flet
        broker: Nome da corretora
        email: Email (não usado ainda)
        password: Senha (não usada ainda)
        balance: Saldo inicial
        bot_token: Token (não usado ainda)
    """
    logger.info(f"Carregando dashboard - Broker: {broker}, Email: {email}")

    # Carregar idioma
    selected_lang = load_language_from_file()
    t = get_bot_translations(selected_lang)
    logger.info(f"✅ [BOT] Idioma selecionado: '{selected_lang}'")

    # Configurações da janela (dimensões fixas - maior para tabela)
    page.title = f"WS Trader - {broker}"

    # Trava a janela completamente (sintaxe page.window.)
    page.window.width = 1200
    page.window.height = 750
    page.window.resizable = False
    page.window.maximizable = False
    page.window.minimizable = True

    # Estados
    bot_running = False
    selected_broker = broker or "IQ Option"
    selected_account = "DEMO"  # DEMO ou REAL

    # ===================== CARREGAR DADOS DO DIA =====================
    broker_key = selected_broker.lower().replace(" ", "_")
    daily_data = load_daily_data(broker_key)
    restored_wins = daily_data.get("wins", 0)
    restored_losses = daily_data.get("losses", 0)
    restored_profit = daily_data.get("profit", 0.0)
    restored_ops = daily_data.get("operations", [])
    
    # Estado do relatório
    report_visible = {"value": False}

    # Flag para forçar refresh da UI
    needs_update = {"flag": False, "last_update": datetime.now()}

    # =============== AUTO-REFRESH INTELIGENTE ===============
    async def auto_refresh_ui():
        """
        Auto-refresh inteligente da UI em tempo real.
        Atualiza a cada 500ms quando há mudanças pendentes.
        """
        global auto_refresh_task
        while bot_running:
            try:
                # Verifica se há atualização pendente
                if needs_update["flag"]:
                    # Atualiza a página
                    page.update()
                    needs_update["flag"] = False
                    needs_update["last_update"] = datetime.now()
                    logger.debug("🔄 UI atualizada automaticamente")

                # Aguarda 500ms antes de verificar novamente
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Erro no auto-refresh: {e}")
                await asyncio.sleep(1)

    def start_auto_refresh():
        """Inicia o auto-refresh assíncrono"""
        global auto_refresh_task
        try:
            auto_refresh_task = asyncio.create_task(auto_refresh_ui())
            logger.info("✅ Auto-refresh iniciado")
        except Exception as e:
            logger.error(f"Erro ao iniciar auto-refresh: {e}")
            # Fallback: usa page.run_task se create_task falhar
            try:
                page.run_task(auto_refresh_ui)
                logger.info("✅ Auto-refresh iniciado (fallback)")
            except Exception as e2:
                logger.error(f"Erro no fallback do auto-refresh: {e2}")

    def stop_auto_refresh():
        """Para o auto-refresh assíncrono"""
        global auto_refresh_task
        if auto_refresh_task:
            try:
                auto_refresh_task.cancel()
                logger.info("🛑 Auto-refresh parado")
            except Exception as e:
                logger.error(f"Erro ao parar auto-refresh: {e}")
            auto_refresh_task = None

    def mark_for_update():
        """Marca a UI para ser atualizada no próximo ciclo"""
        needs_update["flag"] = True

    # =============== CONTROLES DO BOT ===============

    # Saldo
    saldo_text = ft.Text(
        f"{t['balance']}: R$ {balance:.2f}",
        size=16,
        color="#4CAF50",
        weight=ft.FontWeight.BOLD
    )

    # Status (CORES SUAVES - SEM NEGRITO)
    status_text = ft.Text(
        t["stopped"],
        size=15,
        weight=ft.FontWeight.NORMAL,  # Sem negrito
        color="#F87171"  # Vermelho mais claro
    )

    # Ícone de status (bolinha suave)
    status_icon = ft.Icon(ft.Icons.CIRCLE, size=12, color="#F87171")

    # Seletor de Corretora (MODERNO)
    broker_dropdown = ft.Dropdown(
        label=t["broker"],
        label_style=ft.TextStyle(color="#9CA3AF", size=12),
        options=[
            ft.dropdown.Option("IQ Option"),
            ft.dropdown.Option("Bullex"),
        ],
        value=selected_broker,
        bgcolor="#1f2937",
        color="#E8EAF6",
        border_color="#3f4654",
        focused_border_color="#5B8DEF",
        border_radius=8,
        text_size=14,
        disabled=False
    )

    def on_broker_change(e):
        nonlocal selected_broker
        selected_broker = e.control.value
        logger.info(f"Corretora alterada para: {selected_broker}")
        page.title = f"WS Trader - {selected_broker}"
        page.update()

    broker_dropdown.on_change = on_broker_change

    # Seletor de Tipo de Conta (MODERNO)
    account_dropdown = ft.Dropdown(
        label=t["account_type"],
        label_style=ft.TextStyle(color="#9CA3AF", size=12),
        options=[
            ft.dropdown.Option("DEMO", text=t["demo"]),
            ft.dropdown.Option("REAL", text=t["real"]),
        ],
        value=selected_account,
        bgcolor="#1f2937",
        color="#E8EAF6",
        border_color="#3f4654",
        focused_border_color="#5B8DEF",
        border_radius=8,
        text_size=14,
        disabled=False
    )

    def on_account_change(e):
        nonlocal selected_account
        selected_account = e.control.value
        logger.info(f"Conta alterada para: {selected_account}")
        # Muda cor do saldo se for REAL
        if selected_account == "REAL":
            saldo_text.color = "#FF9800"  # Laranja para alertar
        else:
            saldo_text.color = "#4CAF50"  # Verde para demo
        page.update()

    account_dropdown.on_change = on_account_change

    # Botão Iniciar/Parar IA (mais quadrado)
    def toggle_bot(e):
        nonlocal bot_running
        global bot_thread, bot_should_stop

        if not bot_running:
            # Desabilita seletores enquanto bot roda
            broker_dropdown.disabled = True
            account_dropdown.disabled = True

            # Inicia o bot
            e.control.text = t["stop"]
            e.control.icon = ft.Icons.STOP_ROUNDED
            e.control.bgcolor = "#EF4444"  # Vermelho suave
            status_text.value = t["running"]
            status_text.color = "#34D399"  # Verde mais claro
            status_icon.color = "#34D399"
            exit_button.visible = False  # Esconde botão de sair
            bot_running = True
            logger.info(f"IA INICIADA - {selected_broker} - {selected_account}")

            # ✅ INICIA AUTO-REFRESH ASSÍNCRONO
            start_auto_refresh()

            # ✅ Iniciar thread do bot
            bot_should_stop = False

            # Importa e inicia o bot WS_AUTO_AI_ENGINE
            try:
                from ws_auto_ai_engine import TradingEngine, TradingConfig

                def bot_wrapper():
                    try:
                        logger.info("=" * 60)
                        logger.info("WS_AUTO_AI - Iniciando...")
                        logger.info(f"Conta: {selected_account}")
                        logger.info(f"Stake: R$ {balance * 0.01:.2f}")
                        logger.info("=" * 60)

                        # Configura o motor
                        config = TradingConfig()
                        config.EMAIL = email
                        config.SENHA = password
                        config.CONTA = "PRACTICE" if selected_account == "DEMO" else "REAL"
                        config.USE_DYNAMIC_STAKE = True
                        config.PERCENT_BANCA = 1.0  # 1% da banca

                        # Callback para atualizar logs na UI
                        def log_ui(msg: str):
                            logger.info(msg)

                        # Callback para atualizar estatísticas na UI (COM AUTO-REFRESH)
                        def stats_ui(stats: dict):
                            try:
                                # Atualiza estatísticas na UI
                                total_wins = restored_wins + stats.get('wins', 0)
                                total_losses = restored_losses + stats.get('losses', 0)
                                wins_text.value = str(total_wins)
                                losses_text.value = str(total_losses)

                                total_ops = total_wins + total_losses
                                win_rate = (total_wins / total_ops * 100) if total_ops > 0 else 0.0
                                winrate_text.value = f"{win_rate:.1f}%"

                                session_lucro = stats.get('lucro', 0.0)
                                total_lucro = restored_profit + session_lucro
                                profit_text.value = f"R$ {total_lucro:.2f}"
                                if total_lucro > 0:
                                    profit_text.color = "#10B981"
                                elif total_lucro < 0:
                                    profit_text.color = "#EF4444"

                                saldo = stats.get('saldo', 0.0)
                                saldo_text.value = f"{t['balance']}: R$ {saldo:.2f}"

                                # Armazena saldo atual na página para uso posterior
                                page._current_saldo = saldo

                                # Salva dados do dia
                                daily = load_daily_data(broker_key)
                                daily["wins"] = total_wins
                                daily["losses"] = total_losses
                                daily["profit"] = total_lucro
                                daily["broker"] = selected_broker
                                save_daily_data(daily, broker_key)

                                logger.info(f"Stats atualizados: W:{total_wins} L:{total_losses} Saldo:R${saldo:.2f}")

                                # ✅ Atualiza indicador de fase da IA
                                _update_ai_phase_ui()

                                # ✅ MARCA PARA AUTO-REFRESH (em vez de page.update() direto)
                                mark_for_update()
                            except Exception as e:
                                logger.error(f"Erro ao atualizar stats UI: {e}")

                        # Callback para adicionar operação na tabela (COM AUTO-REFRESH)
                        def add_operation_ui(op: dict):
                            try:
                                # Remove mensagem vazia se existir
                                if empty_message.visible:
                                    empty_message.visible = False

                                direction_icon = "CALL" if op['direction'] == "CALL" else "PUT"
                                direction_color = "#10B981" if op['direction'] == "CALL" else "#EF4444"

                                # Formata data/hora usando datetime do op ou now
                                time_str = op.get('timestamp', datetime.now()).strftime("%H:%M:%S")

                                new_row = ft.DataRow(
                                    cells=[
                                        ft.DataCell(ft.Text(time_str, size=12)),
                                        ft.DataCell(ft.Text(op['asset'], size=12)),
                                        ft.DataCell(ft.Text(f"{direction_icon}", color=direction_color, size=12)),
                                        ft.DataCell(ft.Text(f"R$ {op['stake']:.2f}", size=12)),
                                        ft.DataCell(ft.Text(f"{op.get('payout', 0)}%", size=12)),
                                        ft.DataCell(ft.Text(t["executing"], color="#F59E0B", size=12)),
                                    ],
                                    data=op.get('order_id')
                                )
                                operations_table.rows.insert(0, new_row)

                                # Salva operação no arquivo diário
                                try:
                                    daily = load_daily_data(broker_key)
                                    daily.setdefault("operations", []).append({
                                        "time": time_str,
                                        "asset": op['asset'],
                                        "direction": op['direction'],
                                        "stake": op['stake'],
                                        "payout": op.get('payout', 0),
                                        "result": "pending",
                                        "order_id": op.get('order_id'),
                                    })
                                    save_daily_data(daily, broker_key)
                                except Exception:
                                    pass

                                # Atualiza saldo atual também
                                try:
                                    if hasattr(page, '_current_saldo'):
                                        saldo_text.value = f"{t['balance']}: R$ {page._current_saldo:.2f}"
                                except Exception:
                                    pass

                                mark_for_update()
                                logger.info(f"Operação adicionada à tabela: {op['asset']} {op['direction']}")
                            except Exception as e:
                                logger.error(f"Erro ao adicionar operação na UI: {e}")

                        # Callback para atualizar resultado da operação (COM AUTO-REFRESH)
                        def update_operation_result(order_id: int, result: str, profit: float):
                            try:
                                for row in operations_table.rows:
                                    if row.data == order_id:
                                        result_cell = row.cells[5]
                                        if result == 'win':
                                            result_cell.content = ft.Text(f"{t['win']} (+R$ {profit:.2f})", color="#10B981", size=12)
                                        elif result == 'loss':
                                            result_cell.content = ft.Text(f"{t['loss']} (-R$ {abs(profit):.2f})", color="#EF4444", size=12)
                                        else:
                                            result_cell.content = ft.Text(f"{t['unavailable']}", color="#9CA3AF", size=12)

                                        logger.info(f"Resultado atualizado na UI: {result} | Lucro: R$ {profit:.2f}")

                                        # Salva resultado no diário
                                        try:
                                            daily = load_daily_data(broker_key)
                                            for op in daily.get("operations", []):
                                                if op.get("order_id") == order_id:
                                                    op["result"] = result
                                                    op["profit"] = profit
                                                    break
                                            save_daily_data(daily, broker_key)
                                        except Exception:
                                            pass

                                        mark_for_update()
                                        break
                            except Exception as e:
                                logger.error(f"Erro ao atualizar resultado na UI: {e}")

                        # Cria motor
                        engine = TradingEngine(config, log_ui, stats_ui, add_operation_ui, update_operation_result)

                        # Conecta
                        if not engine.conectar():
                            logger.error("❌ Falha ao conectar")
                            status_text.value = t["connection_failed"]
                            status_text.color = "#F87171"
                            status_icon.color = "#F87171"
                            page.update()
                            return

                        logger.info("✅ Conectado - Iniciando operação")

                        # Atualiza saldo inicial na UI
                        try:
                            saldo_text.value = f"{t['balance']}: R$ {engine.saldo_inicial:.2f}"
                            page.update()
                        except Exception:
                            pass

                        # Stop flag para controle
                        stop_flag = threading.Event()

                        # Loop principal
                        while not bot_should_stop and not stop_flag.is_set():
                            try:
                                # Aqui roda 1 ciclo do engine
                                engine.loop_principal(stop_flag)

                                # Se engine parou (meta/stop), sai do loop
                                if not engine.running:
                                    logger.info("Bot parou: meta atingida ou stop loss")
                                    break
                            except Exception as loop_ex:
                                logger.error(f"Erro no loop: {loop_ex}")
                                import traceback
                                logger.error(traceback.format_exc())
                                break

                        # Finaliza
                        try:
                            if engine.iq:
                                saldo_final = float(engine.iq.get_balance())
                                lucro = saldo_final - engine.saldo_inicial
                                logger.info(f"💵 Saldo final: R$ {saldo_final:.2f}")
                                logger.info(f"📊 Resultado: R$ {lucro:.2f}")

                                # Atualiza UI final
                                saldo_text.value = f"{t['balance']}: R$ {saldo_final:.2f}"
                                profit_text.value = f"R$ {lucro:.2f}"
                                if lucro > 0:
                                    profit_text.color = "#10B981"
                                elif lucro < 0:
                                    profit_text.color = "#EF4444"

                                # Mostra resumo
                                logger.info(f"📊 Total de operações: {engine.total_trades}")
                                logger.info(f"✅ Vitórias: {engine.wins}")
                                logger.info(f"❌ Derrotas: {engine.losses}")
                                if engine.total_trades > 0:
                                    wr = (engine.wins / engine.total_trades) * 100
                                    logger.info(f"📈 Win Rate: {wr:.1f}%")

                                page.update()
                                # IQ Option API não tem método disconnect, a conexão é limpa automaticamente
                        except Exception as e:
                            logger.error(f"Erro ao finalizar: {e}")

                        logger.info("Bot finalizado")

                        # Reseta estado da UI
                        status_text.value = t["stopped"]
                        status_text.color = "#F87171"
                        status_icon.color = "#F87171"
                        page.update()

                    except Exception as ex:
                        logger.error(f"Erro no bot: {ex}", exc_info=True)
                        status_text.value = t["ai_error"]
                        status_text.color = "#F87171"
                        status_icon.color = "#F87171"
                        page.update()

                bot_thread = threading.Thread(target=bot_wrapper, daemon=True)
                bot_thread.start()
                logger.info("Thread do bot iniciada com sucesso")
            except Exception as ex:
                logger.error(f"Erro ao iniciar bot: {ex}")
                status_text.value = t["start_error"]
                status_text.color = "#F87171"  # Vermelho mais claro
                status_icon.color = "#F87171"
                e.control.text = t["start"]
                e.control.icon = ft.Icons.PLAY_ARROW
                e.control.bgcolor = "#5B8DEF"
                bot_running = False
                broker_dropdown.disabled = False
                account_dropdown.disabled = False

        else:
            # Reabilita seletores
            broker_dropdown.disabled = False
            account_dropdown.disabled = False

            # Para o bot
            e.control.text = t["start"]
            e.control.icon = ft.Icons.PLAY_ARROW_ROUNDED
            e.control.bgcolor = "#5B8DEF"  # Azul suave
            status_text.value = t["stopped"]
            status_text.color = "#F87171"  # Vermelho mais claro
            status_icon.color = "#F87171"
            exit_button.visible = True  # Mostra botão de sair
            bot_running = False
            logger.info(f"IA PARADA - {selected_broker}")

            # ✅ PARA AUTO-REFRESH ASSÍNCRONO
            stop_auto_refresh()

            # ✅ Parar thread do bot
            bot_should_stop = True
            logger.info("Sinal de parada enviado para o bot")

        page.update()

    start_button = ft.ElevatedButton(
        content=ft.Text(t["start"]),
        icon=ft.Icons.PLAY_ARROW_ROUNDED,
        bgcolor="#5B8DEF",
        color="#FFFFFF",
        height=45,
        on_click=toggle_bot,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=10),
            elevation=0
        )
    )

    # ===================== INDICADOR DE FASE DA IA =====================
    def _get_ai_stats():
        """Lê dados de treinamento da IA do LGBM (LIVE + backtest filtrado).
        Retorna (total_trades, total_wr)."""
        try:
            _bsuffix = {"iq_option": "m1", "bullex": "bullex", "casatrader": "casatrader"}.get(broker_key, "m1")
            lgbm_file = f"ws_lgbm_data_{_bsuffix}.json"
            if os.path.exists(lgbm_file):
                with open(lgbm_file, "r", encoding="utf-8") as f:
                    samples = json.load(f)
                if isinstance(samples, list):
                    total = len(samples)
                    wins = sum(1 for s in samples if isinstance(s, dict) and s.get("label") == 1)
                    wr = (wins / total * 100.0) if total > 0 else 0.0
                    return total, wr
        except Exception:
            pass
        return 0, 0.0

    def _get_ai_phase(total, win_rate):
        """Retorna fase baseado em trades + win rate — cor por corretora"""
        broker_colors = {
            "iq_option":   ["#FFB74D", "#FF9800", "#F57C00", "#E65100"],
            "bullex":      ["#66BB6A", "#43A047", "#2E7D32", "#1B5E20"],
            "casatrader":  ["#64B5F6", "#42A5F5", "#1E88E5", "#1565C0"],
        }
        palette = broker_colors.get(broker_key, ["#F59E0B", "#FF9800", "#5B8DEF", "#10B981"])
        if total >= 50 and win_rate >= 65:
            return (t["ai_phase_full"], palette[3], 1.0, t["ai_phase_desc_full"])
        elif total >= 25 and win_rate >= 58:
            progress = min(1.0, (win_rate - 58) / 7.0 * 0.5 + total / 50.0 * 0.5)
            return (t["ai_phase_lgbm"], palette[2], progress, t["ai_phase_desc_lgbm"])
        elif total >= 10 and win_rate >= 52:
            progress = min(1.0, (win_rate - 52) / 6.0 * 0.5 + total / 25.0 * 0.5)
            return (t["ai_phase_bayes"], palette[1], progress, t["ai_phase_desc_bayes"])
        else:
            progress = 0.0
            if total > 0:
                progress = min(1.0, total / 30.0 * 0.5 + win_rate / 90.0 * 0.5)
            return (t["ai_phase_warmup"], palette[0], progress, t["ai_phase_desc_warmup"])

    ai_total, ai_wr = _get_ai_stats()
    ai_phase_name, ai_phase_color, ai_progress, ai_phase_desc = _get_ai_phase(ai_total, ai_wr)

    ai_phase_label = ft.Text(t["ai_learning"], size=11, color="#9CA3AF", weight=ft.FontWeight.W_500)
    ai_phase_text = ft.Text(ai_phase_name, size=13, weight=ft.FontWeight.BOLD, color=ai_phase_color)
    ai_trades_count = ft.Text(f"WR: {ai_wr:.0f}% | Treino: {ai_total}", size=10, color="#6B7280")
    ai_phase_description = ft.Text(ai_phase_desc, size=10, color="#6B7280", italic=True)
    ai_progress_bar = ft.ProgressBar(
        value=min(1.0, ai_progress),
        color=ai_phase_color,
        bgcolor="#1f2937",
        bar_height=4,
        border_radius=2,
    )

    ai_phase_card = ft.Container(
        content=ft.Column(
            controls=[
                ft.Row(
                    controls=[ai_phase_label, ai_phase_text],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                ai_progress_bar,
                ai_trades_count,
                ai_phase_description,
            ],
            spacing=4,
        ),
        padding=ft.padding.symmetric(horizontal=12, vertical=8),
        bgcolor="#1a1f2e",
        border_radius=8,
        border=ft.border.all(1, "#2a2f3e"),
    )

    def _update_ai_phase_ui():
        """Atualiza o indicador de fase da IA"""
        try:
            total, wr = _get_ai_stats()
            phase_name, phase_color, progress, phase_desc = _get_ai_phase(total, wr)
            ai_phase_text.value = phase_name
            ai_phase_text.color = phase_color
            ai_progress_bar.value = min(1.0, progress)
            ai_progress_bar.color = phase_color
            ai_trades_count.value = f"WR: {wr:.0f}% | Treino: {total}"
            ai_phase_description.value = phase_desc
        except Exception:
            pass

    # Estatísticas COMPACTAS (cores suaves e modernas) — restaura dados do dia
    wins_text = ft.Text(str(restored_wins), size=16, weight=ft.FontWeight.BOLD, color="#10B981")
    losses_text = ft.Text(str(restored_losses), size=16, weight=ft.FontWeight.BOLD, color="#EF4444")
    restored_wr = (restored_wins / (restored_wins + restored_losses) * 100) if (restored_wins + restored_losses) > 0 else 0
    winrate_text = ft.Text(f"{restored_wr:.0f}%", size=16, weight=ft.FontWeight.BOLD, color="#5B8DEF")
    profit_text = ft.Text(f"R$ {restored_profit:.2f}", size=16, weight=ft.FontWeight.BOLD,
                          color="#10B981" if restored_profit >= 0 else "#EF4444")

    stats_row = ft.Row(
        controls=[
            # Vitórias
            ft.Row(
                controls=[
                    ft.Text(t["wins"], size=12, color="#9CA3AF"),
                    wins_text
                ],
                spacing=6
            ),
            # Derrotas
            ft.Row(
                controls=[
                    ft.Text(t["losses"], size=12, color="#9CA3AF"),
                    losses_text
                ],
                spacing=6
            ),
            # Win Rate
            ft.Row(
                controls=[
                    ft.Text(t["rate"], size=12, color="#9CA3AF"),
                    winrate_text
                ],
                spacing=6
            ),
            # Lucro
            ft.Row(
                controls=[
                    ft.Text(t["profit"], size=12, color="#9CA3AF"),
                    profit_text
                ],
                spacing=6
            ),
        ],
        spacing=32,
        alignment=ft.MainAxisAlignment.START
    )

    # Botão de Sair (só visível quando bot NÃO está rodando)
    def go_back_to_home(e):
        if not bot_running:
            logger.info("Saindo do dashboard, voltando para tela inicial")
            page.push_route("/")
        else:
            logger.warning("Não é possível sair enquanto o bot está operando")
            page.snack_bar = ft.SnackBar(
                content=ft.Text(t["stop_bot_first"], color="#FFFFFF"),
                bgcolor="#F1A8A8"
            )
            page.snack_bar.open = True
            page.update()

    exit_button = ft.IconButton(
        icon=ft.Icons.LOGOUT_ROUNDED,
        icon_color="#F1E8E8",
        tooltip=t["exit"],
        on_click=go_back_to_home,
        visible=True  # Controlaremos visibilidade dinamicamente
    )

    # ===================== PAINEL DE RELATÓRIO =====================
    def _build_bar_chart():
        """Gráfico de barras usando Containers (compatível Flet 0.80)"""
        brokers_info = [
            ("IQ Option", "iq_option", "#FF9800"),
            ("Bullex", "bullex", "#10B981"),
            ("CasaTrader", "casatrader", "#5B8DEF"),
        ]
        values = []
        for display_name, bkey, color in brokers_info:
            d = load_daily_data(bkey)
            values.append(d.get("profit", 0.0))

        max_val = max(abs(v) for v in values) if any(v != 0 for v in values) else 1.0
        bar_max_h = 130

        bar_cols = []
        for i, (display_name, bkey, color) in enumerate(brokers_info):
            val = values[i]
            h = max(4, int(abs(val) / max_val * bar_max_h)) if val != 0 else 4
            bar_color = color if val >= 0 else "#EF4444"
            bar_cols.append(
                ft.Column([
                    ft.Text(f"R$ {val:.2f}", size=11, color=bar_color, weight=ft.FontWeight.W_600,
                            text_align=ft.TextAlign.CENTER),
                    ft.Container(width=55, height=h, bgcolor=bar_color,
                                 border_radius=ft.border_radius.only(top_left=4, top_right=4),
                                 animate=ft.Animation(400, ft.AnimationCurve.EASE_IN_OUT)),
                    ft.Text(display_name, size=10, color="#9CA3AF", text_align=ft.TextAlign.CENTER),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=4,
                   alignment=ft.MainAxisAlignment.END)
            )

        return ft.Container(
            content=ft.Row(bar_cols, spacing=30, alignment=ft.MainAxisAlignment.CENTER,
                           vertical_alignment=ft.CrossAxisAlignment.END),
            height=200, padding=ft.padding.only(top=10, bottom=10),
        )

    def _build_line_chart():
        """Tabela semanal por corretora (compatível Flet 0.80)"""
        weekly = load_weekly_data()
        today = datetime.now().date()
        days = [(today - timedelta(days=i)) for i in range(6, -1, -1)]

        colors = {"iq_option": "#FF9800", "bullex": "#10B981", "casatrader": "#5B8DEF"}
        names = {"iq_option": "IQ Option", "bullex": "Bullex", "casatrader": "CasaTrader"}

        # Cabeçalho com dias
        header_cells = [ft.Text("", size=10, color="#9CA3AF", width=80)]
        for d in days:
            header_cells.append(ft.Text(d.strftime("%d/%m"), size=10, color="#9CA3AF",
                                        text_align=ft.TextAlign.CENTER, width=65))

        rows = [ft.Row(header_cells, spacing=4)]

        for bkey in ["iq_option", "bullex", "casatrader"]:
            broker_list = weekly.get(bkey, [])
            date_map = {}
            for entry in broker_list:
                if isinstance(entry, dict):
                    date_map[entry.get("date", "")] = entry.get("profit", 0.0)
            row_cells = [ft.Text(names[bkey], size=10, color=colors[bkey],
                                  weight=ft.FontWeight.W_600, width=80)]
            for d in days:
                date_str = d.strftime("%Y-%m-%d")
                val = date_map.get(date_str, 0.0)
                cell_color = "#10B981" if val > 0 else "#EF4444" if val < 0 else "#555555"
                row_cells.append(
                    ft.Container(
                        content=ft.Text(f"{val:.0f}" if val != 0 else "-", size=10,
                                        color=cell_color, text_align=ft.TextAlign.CENTER),
                        width=65, height=30,
                        bgcolor="#1a2332" if val != 0 else "#1f2937",
                        border_radius=4,
                        alignment=ft.Alignment(0, 0),
                        border=ft.border.all(1, "#2a2f3e"),
                    )
                )
            rows.append(ft.Row(row_cells, spacing=4))

        return ft.Container(
            content=ft.Column(rows, spacing=4),
            padding=ft.padding.only(top=6, bottom=6),
        )

    # Legenda do gráfico de linha
    line_chart_legend = ft.Row(
        controls=[
            ft.Row([ft.Icon(ft.Icons.CIRCLE, size=8, color="#FF9800"), ft.Text("IQ Option", size=10, color="#9CA3AF")], spacing=4),
            ft.Row([ft.Icon(ft.Icons.CIRCLE, size=8, color="#10B981"), ft.Text("Bullex", size=10, color="#9CA3AF")], spacing=4),
            ft.Row([ft.Icon(ft.Icons.CIRCLE, size=8, color="#5B8DEF"), ft.Text("CasaTrader", size=10, color="#9CA3AF")], spacing=4),
        ],
        spacing=16,
        alignment=ft.MainAxisAlignment.CENTER,
    )

    # Container do relatório (inicialmente oculto)
    report_bar_container = ft.Container(content=_build_bar_chart(), padding=10)
    report_line_container = ft.Container(content=_build_line_chart(), padding=10)

    report_panel = ft.Container(
        content=ft.Column(
            controls=[
                ft.Text(t["accumulated"], size=14, weight=ft.FontWeight.W_600, color="#FF9800"),
                report_bar_container,
                ft.Divider(color="#3f4654", height=1),
                ft.Text(t["weekly"], size=14, weight=ft.FontWeight.W_600, color="#5B8DEF"),
                line_chart_legend,
                report_line_container,
            ],
            spacing=8,
        ),
        padding=16,
        bgcolor="#1a1f2e",
        border_radius=12,
        border=ft.border.all(1, "#3f4654"),
        visible=False,
        animate=ft.Animation(300, ft.AnimationCurve.EASE_IN_OUT),
    )

    def toggle_report(e):
        report_visible["value"] = not report_visible["value"]
        if report_visible["value"]:
            # Reconstrói os gráficos com dados atualizados
            report_bar_container.content = _build_bar_chart()
            report_line_container.content = _build_line_chart()
        report_panel.visible = report_visible["value"]
        page.update()

    report_button = ft.IconButton(
        icon=ft.Icons.BAR_CHART_ROUNDED,
        icon_color="#FF9800",
        tooltip=t["report"],
        on_click=toggle_report,
        icon_size=24,
    )

    # Header (SEM FUNDO - transparente)
    header = ft.Container(
        content=ft.Row(
            controls=[
                ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.CANDLESTICK_CHART, size=32, color="#5B8DEF"),
                        ft.Text(
                            "WS Trader",
                            size=26,
                            weight=ft.FontWeight.BOLD,
                            color="#E8EAF6"
                        ),
                    ],
                    spacing=12
                ),
                ft.Row(
                    controls=[
                        saldo_text,
                        report_button,
                        exit_button
                    ],
                    spacing=16
                )
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN
        ),
        padding=ft.padding.only(left=24, right=24, top=20, bottom=16)
    )

    # Painel Lateral de Controle (lado direito - BOTÃO NO FINAL)
    control_sidebar = ft.Container(
        content=ft.Column(
            controls=[
                # Status
                ft.Row(
                    controls=[
                        status_icon,
                        status_text
                    ],
                    spacing=8,
                    alignment=ft.MainAxisAlignment.CENTER
                ),
                ft.Divider(color="#3f4654", height=1),
                # Configurações
                ft.Text(t["settings"], size=13, weight=ft.FontWeight.W_500, color="#E8EAF6"),
                ft.Container(height=5),
                broker_dropdown,
                ft.Container(height=10),
                account_dropdown,
                ft.Container(height=10),
                # Indicador de Fase da IA
                ai_phase_card,
                ft.Container(height=10),
                # Botão Iniciar/Parar NO FINAL
                start_button,
            ],
            spacing=12,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH
        ),
        padding=20,
        bgcolor="#2a2f3e",
        border_radius=12,
        border=ft.border.all(1, "#3f4654"),
        width=280,
        height=350,  # Altura mínima para garantir que apareça
        alignment=ft.alignment.top_center
    )

    # Tabela de Histórico MODERNA (compacta e espaçosa)
    operations_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text(t["datetime"], size=12, weight=ft.FontWeight.W_600, color="#9CA3AF")),
            ft.DataColumn(ft.Text(t["pair"], size=12, weight=ft.FontWeight.W_600, color="#9CA3AF")),
            ft.DataColumn(ft.Text(t["direction"], size=12, weight=ft.FontWeight.W_600, color="#9CA3AF")),
            ft.DataColumn(ft.Text(t["value"], size=12, weight=ft.FontWeight.W_600, color="#9CA3AF")),
            ft.DataColumn(ft.Text(t["payout"], size=12, weight=ft.FontWeight.W_600, color="#9CA3AF")),
            ft.DataColumn(ft.Text(t["result"], size=12, weight=ft.FontWeight.W_600, color="#9CA3AF")),
        ],
        rows=[
            # Será preenchido dinamicamente
        ],
        border=ft.border.all(1, "#3f4654"),
        border_radius=8,
        heading_row_color="#1f2937",
        heading_row_height=40,
        data_row_max_height=45,
        data_row_min_height=45,
        column_spacing=80,
        horizontal_margin=40,
        data_text_style=ft.TextStyle(size=12, color="#D1D5DB"),
        expand=True  # Expande a tabela para ocupar todo o espaço disponível
    )

    # ===================== RESTAURAR OPERAÇÕES DO DIA =====================
    if restored_ops:
        for op in restored_ops:
            direction_icon = "CALL" if op.get('direction') == "CALL" else "PUT"
            direction_color = "#10B981" if op.get('direction') == "CALL" else "#EF4444"
            time_str = op.get('time', '--:--:--')
            res = op.get('result', 'pending')
            profit_val = op.get('profit', 0)
            if res == 'win':
                result_content = ft.Text(f"{t['win']} (+R$ {profit_val:.2f})", color="#10B981", size=12)
            elif res == 'loss':
                result_content = ft.Text(f"{t['loss']} (-R$ {abs(profit_val):.2f})", color="#EF4444", size=12)
            elif res == 'pending':
                result_content = ft.Text(t["executing"], color="#F59E0B", size=12)
            else:
                result_content = ft.Text(t["unavailable"], color="#9CA3AF", size=12)

            operations_table.rows.append(ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(time_str, size=12)),
                    ft.DataCell(ft.Text(op.get('asset', ''), size=12)),
                    ft.DataCell(ft.Text(direction_icon, color=direction_color, size=12)),
                    ft.DataCell(ft.Text(f"R$ {op.get('stake', 0):.2f}", size=12)),
                    ft.DataCell(ft.Text(f"{op.get('payout', 0)}%", size=12)),
                    ft.DataCell(result_content),
                ],
                data=op.get('order_id')
            ))
        logger.info(f"✅ {len(restored_ops)} operações restauradas do dia")

    # Container de mensagem quando vazio (no centro da tabela)
    empty_message = ft.Container(
        content=ft.Text(
            t["empty"],
            color="#64748B",
            text_align=ft.TextAlign.CENTER,
            size=13
        ),
        padding=80,
        alignment=ft.Alignment(0, 0),
        expand=True,
        visible=len(restored_ops) == 0  # Oculta se já tem operações
    )

    # ===================== CARDS DAS CORRETORAS =====================
    def _broker_card(name, icon_name, color, is_active=False):
        """Cria um card visual de corretora"""
        border_color = color if is_active else "#3f4654"
        bg = "#1a2332" if is_active else "#1f2937"
        badge_text = "ATIVO" if is_active else ""
        return ft.Container(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Icon(icon_name, size=20, color=color),
                            ft.Text(name, size=13, weight=ft.FontWeight.W_600, color="#E8EAF6"),
                        ],
                        spacing=8,
                    ),
                    ft.Text(badge_text, size=10, color=color, weight=ft.FontWeight.BOLD) if is_active else ft.Container(height=0),
                ],
                spacing=4,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.padding.symmetric(horizontal=16, vertical=10),
            bgcolor=bg,
            border_radius=10,
            border=ft.border.all(2 if is_active else 1, border_color),
            width=160,
            animate=ft.Animation(300, ft.AnimationCurve.EASE_IN_OUT),
        )
    
    broker_cards_row = ft.Row(
        controls=[
            _broker_card("IQ Option", ft.Icons.SHOW_CHART, "#FF9800", is_active=("iq" in broker_key)),
            _broker_card("Bullex", ft.Icons.BOLT, "#10B981", is_active=("bullex" in broker_key)),
            _broker_card("CasaTrader", ft.Icons.HOME_WORK, "#5B8DEF", is_active=("casa" in broker_key)),
        ],
        spacing=12,
        alignment=ft.MainAxisAlignment.START,
    )

    # Painel Principal (Tabela + Estatísticas - cores suaves)
    main_panel = ft.Container(
        content=ft.Column(
            controls=[
                # Cards das corretoras no topo
                ft.Container(
                    content=broker_cards_row,
                    padding=ft.padding.only(left=0, right=0, top=4, bottom=8)
                ),
                # Estatísticas compactas (SEM FUNDO)
                ft.Container(
                    content=stats_row,
                    padding=ft.padding.only(left=0, right=0, top=8, bottom=16)
                ),
                ft.Container(height=12),
                # Tabela de operações
                ft.Container(
                    content=ft.Column(
                        controls=[
                            ft.Text(
                                t["history"],
                                size=14,
                                weight=ft.FontWeight.BOLD,
                                color="#E8EAF6"
                            ),
                            ft.Container(height=10),
                            # Mostra tabela E mensagem vazia (controlados por visibilidade)
                            ft.Container(
                                content=ft.Stack(
                                    controls=[
                                        # Tabela sempre visível
                                        ft.Container(
                                            content=operations_table,
                                            expand=True
                                        ),
                                        # Mensagem vazia sobreposta no centro
                                        empty_message
                                    ],
                                    expand=True
                                ),
                                expand=True
                            )
                        ],
                        spacing=0,
                        expand=True
                    ),
                    padding=18,
                    border_radius=12,
                    expand=True
                ),
                # Painel de relatório (toggle)
                report_panel,
            ],
            spacing=0,
            expand=True
        ),
        expand=True
    )

    # Layout Principal (PROFISSIONAL - estilo trading)
    content = ft.Container(
        content=ft.Column(
            controls=[
                header,
                ft.Container(
                    content=ft.Row(
                        controls=[
                            ft.Container(
                                content=main_panel,
                                expand=True  # Main panel expande
                            ),
                            ft.Container(
                                content=control_sidebar,
                                width=280,  # Sidebar com largura fixa garantida
                                alignment=ft.alignment.top_center
                            ),
                        ],
                        spacing=12,
                        expand=True,
                        vertical_alignment=ft.CrossAxisAlignment.START,
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN  # Garante espaçamento
                    ),
                    padding=16,
                    expand=True
                )
            ],
            spacing=0,
            expand=True
        ),
        expand=True,
        padding=0,
        gradient=ft.LinearGradient(
            begin=ft.Alignment(-1, -1),
            end=ft.Alignment(1, 1),
            colors=["#0E1114", "#111417"]
        )
    )

    page.views.clear()
    page.views.append(
        ft.View(
            route="/bot",
            bgcolor=ft.Colors.TRANSPARENT,
            padding=0,
            spacing=0,
            controls=[content]
        )
    )
    page.update()

    # Força um segundo update para garantir renderização completa
    time.sleep(0.1)
    page.update()

    logger.info("Bot dashboard carregado com sucesso")
