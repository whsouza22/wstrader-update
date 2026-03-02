# -*- coding: utf-8 -*-
"""
Dashboard da IA de Trading - WS Trader
Interface limpa e simplificada com AUTO-REFRESH em tempo real
"""
import flet as ft
import logging
import threading
import asyncio
import os
import json
import time
import requests
from datetime import datetime, timedelta

# Carregar .env para ter STRIPE_PRODUCT_ID disponível
try:
    from dotenv import load_dotenv
    # Primeiro tenta o .env do usuário (~/.wstrader/.env) — onde o Login salva o product_id
    _user_env = os.path.join(os.path.expanduser("~"), ".wstrader", ".env")
    if os.path.exists(_user_env):
        load_dotenv(dotenv_path=_user_env, override=True)
    # Fallback: .env local do script
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_path):
        load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Arquivo de preferências do usuário
USER_PREFS_FILE = os.path.join(os.path.expanduser("~"), ".wstrader", "preferences.json")

# ===================== PERSISTÊNCIA DIÁRIA (ARQUIVO ÚNICO) =====================
UNIFIED_DAILY_FILE = os.path.join(os.path.expanduser("~"), ".wstrader", "ws_daily_log.json")

def _load_full_unified_bot() -> dict:
    """Carrega o arquivo unificado COMPLETO (todas as datas)."""
    default = {"version": 2, "days": {}}
    try:
        if os.path.exists(UNIFIED_DAILY_FILE):
            with open(UNIFIED_DAILY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Migrar formato v1 para v2
            if "version" not in data and "date" in data and "brokers" in data:
                old_date = data["date"]
                old_brokers = data["brokers"]
                data = {"version": 2, "days": {old_date: {"brokers": old_brokers}}}
            return data
    except Exception:
        pass
    return default

def _get_today_section(full: dict, broker: str, acct: str = "DEMO") -> dict:
    """Retorna seção do broker/conta para hoje."""
    today = datetime.now().strftime("%Y-%m-%d")
    day_data = full.get("days", {}).get(today, {"brokers": {}})
    bk = broker.lower().replace(" ", "_")
    section = day_data.get("brokers", {}).get(bk, {}).get(acct, {})
    return section

def load_daily_data(broker: str = "iq_option", acct: str = "DEMO") -> dict:
    """Carrega dados do dia do arquivo UNIFICADO"""
    try:
        full = _load_full_unified_bot()
        section = _get_today_section(full, broker, acct)
        return {
            "wins": section.get("wins", 0),
            "losses": section.get("losses", 0),
            "profit": section.get("profit", 0.0),
            "operations": section.get("operations", []),
            "broker": broker
        }
    except Exception as ex:
        logger.error(f"Erro ao carregar dados diários: {ex}")
    return {"wins": 0, "losses": 0, "profit": 0.0, "operations": [], "broker": broker}

def save_daily_data(data: dict, broker: str = "iq_option", acct: str = "DEMO"):
    """Salva dados do dia no arquivo UNIFICADO"""
    try:
        full = _load_full_unified_bot()
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in full.get("days", {}):
            full.setdefault("days", {})[today] = {"brokers": {}}
        day_data = full["days"][today]
        if "brokers" not in day_data:
            day_data["brokers"] = {}
        bk = broker.lower().replace(" ", "_")
        if bk not in day_data["brokers"]:
            day_data["brokers"][bk] = {}
        if acct not in day_data["brokers"][bk]:
            day_data["brokers"][bk][acct] = {
                "wins": 0, "losses": 0, "profit": 0.0,
                "meta_batida": False, "meta_valor": 0.0,
                "ganhos": 0.0, "entries": [], "operations": []
            }
        section = day_data["brokers"][bk][acct]
        section["wins"] = data.get("wins", 0)
        section["losses"] = data.get("losses", 0)
        section["profit"] = data.get("profit", 0.0)
        # Atualizar operations se presente
        if "operations" in data:
            section["operations"] = data["operations"]
        day_data["last_updated"] = datetime.now().isoformat()
        full["version"] = 2
        # Podar dias antigos (>60)
        all_dates = sorted(full["days"].keys())
        if len(all_dates) > 60:
            for old_date in all_dates[:-60]:
                del full["days"][old_date]
        # Escrita atômica
        os.makedirs(os.path.dirname(UNIFIED_DAILY_FILE), exist_ok=True)
        tmp_path = UNIFIED_DAILY_FILE + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(full, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(tmp_path, UNIFIED_DAILY_FILE)
        except OSError:
            import shutil
            shutil.move(tmp_path, UNIFIED_DAILY_FILE)
    except Exception as ex:
        logger.error(f"Erro ao salvar dados diários: {ex}")

def load_weekly_data() -> dict:
    """Carrega dados dos últimos 7 dias de todas as corretoras do arquivo UNIFICADO"""
    result = {}
    brokers = ["iq_option", "bullex", "casatrader"]
    try:
        full = _load_full_unified_bot()
        all_days = full.get("days", {})
        for broker in brokers:
            broker_data = []
            for i in range(7):
                day = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                day_info = all_days.get(day, {})
                # Somar DEMO + REAL para resumo semanal
                total_profit = 0.0
                total_wins = 0
                total_losses = 0
                for acct in ["DEMO", "REAL"]:
                    section = day_info.get("brokers", {}).get(broker, {}).get(acct, {})
                    total_profit += section.get("profit", 0.0)
                    total_wins += section.get("wins", 0)
                    total_losses += section.get("losses", 0)
                broker_data.append({"date": day, "profit": total_profit, "wins": total_wins, "losses": total_losses})
            result[broker] = list(reversed(broker_data))
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
                logger.info(f"[IA] ✅ Idioma '{lang}' carregado de {USER_PREFS_FILE}")
                return lang
    except Exception as ex:
        logger.error(f"[IA] ❌ Erro ao carregar idioma: {ex}")
    logger.info("[IA] ✅ Usando idioma padrão: PT")
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
            ,"ai_phase_desc_warmup": "IA H&S ativa — aprendendo padrões Cabeça e Ombros.\nEntrada imediata no Ombro D."
            ,"ai_phase_desc_bayes": "IA H&S filtrando sinais fracos.\nPrecisa de 30 operações para evoluir."
            ,"ai_phase_desc_lgbm": "IA H&S otimizando decisões.\nPrecisa de 50+ operações para nível máximo."
            ,"ai_phase_desc_full": "IA H&S calibrada com precisão máxima!\nTodos os módulos ativos."
            ,"ai_trades_label": "Operações acumuladas"
            ,"license_expired_title": "Assinatura Não Encontrada"
            ,"license_expired_msg": "Sua assinatura expirou ou não foi encontrada.\nPara continuar usando o WS Trader, renove sua assinatura."
            ,"license_renew": "Renovar Assinatura"
            ,"license_exit": "Sair"
            ,"license_checking": "Verificando assinatura..."
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
            ,"license_expired_title": "Subscription Not Found"
            ,"license_expired_msg": "Your subscription has expired or was not found.\nTo continue using WS Trader, please renew your subscription."
            ,"license_renew": "Renew Subscription"
            ,"license_exit": "Exit"
            ,"license_checking": "Checking subscription..."
        }
    }
    return translations.get(lang, translations["PT"])

# Variável global para controlar a thread da IA
bot_thread = None
bot_should_stop = False
auto_refresh_task = None  # Task assíncrona para auto-refresh

def bot_dashboard(page: ft.Page, broker: str, email: str, password: str, balance: float, bot_token: str):
    """
    Dashboard da IA com seleção de corretora e tipo de conta

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
    logger.info(f"✅ [IA] Idioma selecionado: '{selected_lang}'")

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

    # ===================== CONTROLE DE PLANO (DEMO vs PRO) =====================
    # Se o produto for o PRO, libera DEMO + REAL. Qualquer outro = só DEMO.
    PRO_PRODUCT_ID = "prod_S4t8FQuUptWQ6R"
    DEMO_PRODUCT_ID = "prod_U3CRqZJMVigJAK"
    _stripe_product = os.environ.get("STRIPE_PRODUCT_ID", "")
    is_demo_plan = (_stripe_product != PRO_PRODUCT_ID)
    if _stripe_product == PRO_PRODUCT_ID:
        logger.info("✅ Plano PRO — todas as contas liberadas")
    elif _stripe_product == DEMO_PRODUCT_ID:
        logger.info(f"🔒 Plano DEMO detectado (product: {_stripe_product}) — conta REAL desabilitada")
    else:
        logger.info(f"🔒 Plano DESCONHECIDO (product: {_stripe_product!r}) — conta REAL desabilitada")

    # ===================== CARREGAR DADOS DO DIA =====================
    broker_key = selected_broker.lower().replace(" ", "_")
    daily_data = load_daily_data(broker_key, selected_account)
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

    # =============== CONTROLES DA IA ===============

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
            ft.dropdown.Option("CasaTrader"),
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
        nonlocal selected_broker, broker_key
        selected_broker = e.control.value
        broker_key = selected_broker.lower().replace(" ", "_")
        logger.info(f"Corretora alterada para: {selected_broker}")
        page.title = f"WS Trader - {selected_broker}"
        page.update()

    broker_dropdown.on_change = on_broker_change

    # Seletor de Tipo de Conta (MODERNO)
    # Se plano DEMO, só mostra opção DEMO e desabilita troca
    _account_options = [
        ft.dropdown.Option("DEMO", text=t["demo"]),
    ]
    if not is_demo_plan:
        _account_options.append(ft.dropdown.Option("REAL", text=t["real"]))

    account_dropdown = ft.Dropdown(
        label=t["account_type"],
        label_style=ft.TextStyle(color="#9CA3AF", size=12),
        options=_account_options,
        value=selected_account,
        bgcolor="#1f2937",
        color="#E8EAF6",
        border_color="#3f4654",
        focused_border_color="#5B8DEF",
        border_radius=8,
        text_size=14,
        disabled=is_demo_plan,  # Trava no plano DEMO
        hint_text="🔒 Somente DEMO" if is_demo_plan else None,
    )

    def on_account_change(e):
        nonlocal selected_account
        new_value = e.control.value
        # Bloqueia conta REAL no plano Demo
        if is_demo_plan and new_value == "REAL":
            logger.warning("🔒 Tentativa de usar conta REAL com plano Demo — bloqueado")
            e.control.value = "DEMO"
            selected_account = "DEMO"
            page.update()
            return
        selected_account = new_value
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
                        # Guarda final: plano Demo SEMPRE usa PRACTICE
                        if is_demo_plan:
                            config.CONTA = "PRACTICE"
                        else:
                            config.CONTA = "PRACTICE" if selected_account == "DEMO" else "REAL"
                        config.BROKER_TYPE = broker_key  # Passa a corretora selecionada
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
                                daily = load_daily_data(broker_key, selected_account)
                                daily["wins"] = total_wins
                                daily["losses"] = total_losses
                                daily["profit"] = total_lucro
                                daily["broker"] = selected_broker
                                save_daily_data(daily, broker_key, selected_account)

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
                                    daily = load_daily_data(broker_key, selected_account)
                                    daily.setdefault("operations", []).append({
                                        "time": time_str,
                                        "asset": op['asset'],
                                        "direction": op['direction'],
                                        "stake": op['stake'],
                                        "payout": op.get('payout', 0),
                                        "result": "pending",
                                        "order_id": op.get('order_id'),
                                    })
                                    save_daily_data(daily, broker_key, selected_account)
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
                                            daily = load_daily_data(broker_key, selected_account)
                                            for op in daily.get("operations", []):
                                                if op.get("order_id") == order_id:
                                                    op["result"] = result
                                                    op["profit"] = profit
                                                    break
                                            save_daily_data(daily, broker_key, selected_account)
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
                                    logger.info("IA parou: meta atingida ou stop loss")
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

                        logger.info("IA finalizada")

                        # Reseta estado da UI
                        status_text.value = t["stopped"]
                        status_text.color = "#F87171"
                        status_icon.color = "#F87171"
                        page.update()

                    except Exception as ex:
                        logger.error(f"Erro na IA: {ex}", exc_info=True)
                        status_text.value = t["ai_error"]
                        status_text.color = "#F87171"
                        status_icon.color = "#F87171"
                        page.update()

                bot_thread = threading.Thread(target=bot_wrapper, daemon=True)
                bot_thread.start()
                logger.info("Thread da IA iniciada com sucesso")
            except Exception as ex:
                logger.error(f"Erro ao iniciar IA: {ex}")
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
    def _read_brain_training_stats_bot():
        """Lê stats da IA H&S (Cabeça e Ombros)."""
        try:
            import json as _json
            # IA H&S usa ws_ai_stats_hs.json
            hs_file = os.path.join(os.path.expanduser("~"), ".wstrader", "ws_ai_stats_hs.json")
            total, wins = 0, 0
            if os.path.exists(hs_file):
                with open(hs_file, "r") as _f:
                    data = _json.load(_f)
                arms = data.get("arms", {})
                for arm_data in arms.values():
                    total += arm_data.get("total", 0)
                    wins += arm_data.get("wins", 0)
            # IA H&S está sempre ativa — model_updates reflete experiência
            model_updates = max(total * 50, 1000)  # Sempre mostra como ativa
            return total, wins, model_updates
        except Exception:
            pass
        return 0, 0, 1000  # Retorna 1000 para sempre mostrar IA ativa

    def _get_ai_stats():
        """Lê dados de treinamento da IA (ReversalAI).
        Retorna (total_experience, combined_wr, training_samples, model_updates)."""
        total, wins, model_updates = _read_brain_training_stats_bot()
        wr = (wins / total * 100.0) if total > 0 else 0.0
        return total, wr, total, model_updates

    def _get_ai_phase(total_exp, combined_wr, model_updates=0):
        """Retorna fase baseado em experiência total — alinhado com Brain._get_experience_level"""
        broker_colors = {
            "iq_option":   ["#FFB74D", "#FF9800", "#F57C00", "#E65100"],
            "bullex":      ["#66BB6A", "#43A047", "#2E7D32", "#1B5E20"],
            "casatrader":  ["#64B5F6", "#42A5F5", "#1E88E5", "#1565C0"],
        }
        palette = broker_colors.get(broker_key, ["#F59E0B", "#FF9800", "#5B8DEF", "#10B981"])
        if total_exp >= 2000 or model_updates >= 40000:
            return (t["ai_phase_full"], palette[3], 1.0, t["ai_phase_desc_full"])
        elif total_exp >= 500 or model_updates >= 15000:
            progress = min(1.0, total_exp / 2000.0 * 0.6 + model_updates / 40000.0 * 0.4)
            return (t["ai_phase_lgbm"], palette[2], progress, t["ai_phase_desc_lgbm"])
        elif total_exp >= 100 or model_updates >= 5000:
            progress = min(1.0, total_exp / 500.0 * 0.6 + model_updates / 15000.0 * 0.4)
            return (t["ai_phase_bayes"], palette[1], progress, t["ai_phase_desc_bayes"])
        elif total_exp >= 30 or model_updates >= 1000:
            progress = min(1.0, total_exp / 100.0 * 0.6 + model_updates / 5000.0 * 0.4)
            return (t["ai_phase_warmup"], palette[0], progress, t["ai_phase_desc_warmup"])
        else:
            progress = 0.0
            if total_exp > 0:
                progress = min(1.0, total_exp / 30.0)
            return (t["ai_phase_warmup"], palette[0], progress, t["ai_phase_desc_warmup"])

    ai_total, ai_wr, ai_train, ai_updates = _get_ai_stats()
    ai_phase_name, ai_phase_color, ai_progress, ai_phase_desc = _get_ai_phase(ai_total, ai_wr, ai_updates)

    ai_phase_label = ft.Text(t["ai_learning"], size=11, color="#9CA3AF", weight=ft.FontWeight.W_500)
    ai_phase_text = ft.Text(ai_phase_name, size=13, weight=ft.FontWeight.BOLD, color=ai_phase_color)
    ai_trades_count = ft.Text(f"WR: {ai_wr:.0f}% | Amostras: {ai_train} | IA WS: ATIVO", size=10, color="#6B7280")
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
            total, wr, train, updates = _get_ai_stats()
            phase_name, phase_color, progress, phase_desc = _get_ai_phase(total, wr, updates)
            ai_phase_text.value = phase_name
            ai_phase_text.color = phase_color
            ai_progress_bar.value = min(1.0, progress)
            ai_progress_bar.color = phase_color
            ai_trades_count.value = f"WR: {wr:.0f}% | Amostras: {train} | IA WS: ATIVO"
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
            d = load_daily_data(bkey, selected_account)
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

    # ===================== OVERLAY DE LICENÇA EXPIRADA =====================
    license_overlay_visible = {"value": False}
    license_check_running = {"value": False}

    def _go_to_payment(e):
        """Abre página de pagamento do Stripe"""
        import webbrowser
        webbrowser.open_new("https://buy.stripe.com/fZe3e38sxfr28Le9AG")

    def _exit_app(e):
        """Sai do app e volta para tela inicial"""
        nonlocal bot_running
        global bot_should_stop
        # Para o bot se estiver rodando
        if bot_running:
            bot_running = False
            bot_should_stop = True
            stop_auto_refresh()
        page.route = "/"
        page.go("/")

    license_overlay = ft.Container(
        content=ft.Container(
            content=ft.Column(
                controls=[
                    ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, size=64, color="#FF9800"),
                    ft.Container(height=16),
                    ft.Text(
                        t.get("license_expired_title", "Assinatura Não Encontrada"),
                        size=24,
                        weight=ft.FontWeight.BOLD,
                        color="#FFFFFF",
                        text_align=ft.TextAlign.CENTER,
                    ),
                    ft.Container(height=12),
                    ft.Text(
                        t.get("license_expired_msg", "Sua assinatura expirou ou não foi encontrada.\nPara continuar usando o WS Trader, renove sua assinatura."),
                        size=14,
                        color="#D1D5DB",
                        text_align=ft.TextAlign.CENTER,
                    ),
                    ft.Container(height=24),
                    ft.ElevatedButton(
                        content=ft.Text(t.get("license_renew", "Renovar Assinatura"), weight=ft.FontWeight.BOLD),
                        bgcolor="#FF681A",
                        color="#FFFFFF",
                        height=48,
                        width=260,
                        on_click=_go_to_payment,
                        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)),
                    ),
                    ft.Container(height=10),
                    ft.TextButton(
                        content=ft.Text(t.get("license_exit", "Sair"), color="#9CA3AF", size=13),
                        on_click=_exit_app,
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=0,
            ),
            width=420,
            padding=40,
            bgcolor="#1a1f2e",
            border_radius=16,
            border=ft.border.all(2, "#FF9800"),
            shadow=[ft.BoxShadow(spread_radius=8, blur_radius=40, color=ft.Colors.with_opacity(0.3, "#000000"))],
        ),
        alignment=ft.Alignment(0, 0),
        expand=True,
        bgcolor=ft.Colors.with_opacity(0.85, "#0E1114"),
        visible=False,  # Começa invisível
    )

    def _show_license_block():
        """Mostra overlay de licença expirada e para o bot"""
        nonlocal bot_running
        global bot_should_stop

        if license_overlay_visible["value"]:
            return  # Já está visível

        logger.warning("⚠️ LICENÇA INVÁLIDA - Bloqueando acesso ao bot")

        # Para o bot se estiver rodando
        if bot_running:
            bot_running = False
            bot_should_stop = True
            stop_auto_refresh()
            status_text.value = t["stopped"]
            status_text.color = "#F87171"
            status_icon.color = "#F87171"
            start_button.text = t["start"]
            start_button.icon = ft.Icons.PLAY_ARROW_ROUNDED
            start_button.bgcolor = "#5B8DEF"
            broker_dropdown.disabled = False
            account_dropdown.disabled = False

        # Desabilita botão de iniciar
        start_button.disabled = True

        # Mostra overlay
        license_overlay.visible = True
        license_overlay_visible["value"] = True

        try:
            page.update()
        except Exception:
            pass

    def _check_license_validity() -> bool:
        """Verifica se a licença/assinatura é válida via Stripe API.
        Retorna True se válida, False se inválida."""
        try:
            url = "https://api-wstrader.onrender.com/check_subscription"
            response = requests.post(
                url,
                json={"email": email.strip().lower()},
                timeout=15
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("status") in ["active", "trial"] or data.get("valid") or data.get("is_active"):
                    logger.info("✅ Licença válida (verificação periódica)")
                    return True
                else:
                    logger.warning(f"❌ Licença inválida: {data.get('message', 'sem assinatura')}")
                    return False
            else:
                logger.warning(f"❌ Erro HTTP {response.status_code} ao verificar licença")
                return False
        except requests.exceptions.Timeout:
            logger.warning("⚠️ Timeout na verificação de licença - mantendo acesso")
            return True  # Em caso de timeout, não bloquear (falha aberta)
        except requests.exceptions.ConnectionError:
            logger.warning("⚠️ Sem conexão na verificação de licença - mantendo acesso")
            return True  # Sem internet, não bloquear
        except Exception as e:
            logger.error(f"⚠️ Erro na verificação de licença: {e}")
            return True  # Erro genérico, não bloquear

    async def _license_watchdog():
        """Tarefa assíncrona que verifica a licença periodicamente.
        - A cada 5 minutos verifica via API
        - Na virada da meia-noite faz verificação imediata
        - Se licença inválida, bloqueia o bot
        """
        CHECK_INTERVAL_SEC = 18000  # 5 horas
        last_check_date = datetime.now().strftime("%Y-%m-%d")

        # Aguarda 60s antes da primeira verificação (dar tempo do app iniciar)
        await asyncio.sleep(60)

        while True:
            try:
                if license_overlay_visible["value"]:
                    # Já está bloqueado, não precisa verificar mais
                    await asyncio.sleep(60)
                    continue

                now = datetime.now()
                current_date = now.strftime("%Y-%m-%d")

                # Detecta virada de meia-noite
                is_midnight_change = (current_date != last_check_date)

                if is_midnight_change:
                    logger.info("🕛 Meia-noite detectada - verificando licença...")
                    last_check_date = current_date

                # Verificação da licença em thread separada (não bloqueia UI)
                loop = asyncio.get_event_loop()
                is_valid = await loop.run_in_executor(None, _check_license_validity)

                if not is_valid:
                    # Licença inválida - bloquear!
                    _show_license_block()
                    return  # Para o watchdog

                # Aguarda próximo ciclo
                # Na virada de meia-noite, verifica mais rápido (a cada 30s por 5min)
                if is_midnight_change:
                    for _ in range(10):
                        await asyncio.sleep(30)
                        if license_overlay_visible["value"]:
                            return
                        is_valid2 = await loop.run_in_executor(None, _check_license_validity)
                        if not is_valid2:
                            _show_license_block()
                            return
                else:
                    await asyncio.sleep(CHECK_INTERVAL_SEC)

            except asyncio.CancelledError:
                logger.info("License watchdog cancelled")
                return
            except Exception as e:
                logger.error(f"Erro no license watchdog: {e}")
                await asyncio.sleep(60)  # Retry após 1 minuto em caso de erro

    # ===================== MONTAGEM FINAL DA VIEW =====================
    page.views.clear()
    page.views.append(
        ft.View(
            route="/bot",
            bgcolor=ft.Colors.TRANSPARENT,
            padding=0,
            spacing=0,
            controls=[
                ft.Stack(
                    controls=[
                        content,           # Dashboard principal
                        license_overlay,   # Overlay de licença (invisível até necessário)
                    ],
                    expand=True,
                )
            ]
        )
    )
    page.update()

    # Força um segundo update para garantir renderização completa
    time.sleep(0.1)
    page.update()

    # ✅ Iniciar watchdog de licença em background
    try:
        page.run_task(_license_watchdog)
        logger.info("✅ License watchdog iniciado (verifica a cada 5min + meia-noite)")
    except Exception as e:
        logger.error(f"Erro ao iniciar license watchdog: {e}")

    logger.info("Bot dashboard carregado com sucesso")
