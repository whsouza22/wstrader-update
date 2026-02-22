# -*- coding: utf-8 -*-
"""
WS TRADER AI — Chat moderno (estilo ChatGPT) + Painel lateral (brokers)
- UI responsiva, bubbles, avatar, markdown, typing indicator
- Botões IQ Option / Bullex lado a lado
- Botão do input alterna Enviar / Parar quando houver broker rodando
- Segurança: NÃO deixa chave OpenAI no código (use variável de ambiente)
"""

import flet as ft
import logging
import threading
import subprocess
import sys
import os
import json
import time
import re
from datetime import datetime, timedelta
from openai import OpenAI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Arquivo de preferências do usuário
USER_PREFS_FILE = os.path.join(os.path.expanduser("~"), ".wstrader", "preferences.json")
LOCKOUT_FILE = os.path.join(os.path.expanduser("~"), ".wstrader", "daily_lockout.json")

def load_language_from_file():
    """Carrega o idioma salvo do arquivo JSON"""
    try:
        if os.path.exists(USER_PREFS_FILE):
            with open(USER_PREFS_FILE, 'r', encoding='utf-8') as f:
                prefs = json.load(f)
                lang = prefs.get('language', 'PT')
                logger.info(f"[CHAT] ✅ Idioma '{lang}' carregado de {USER_PREFS_FILE}")
                return lang
    except Exception as ex:
        logger.error(f"[CHAT] ❌ Erro ao carregar idioma: {ex}")

    logger.info("[CHAT] ✅ Usando idioma padrão: PT")
    return 'PT'

def get_chat_translations(lang):
    """Retorna as traduções para a tela de chat"""
    translations = {
        "PT": {
            "chat_title": "WS TRADER — Chat",
            "chat_subtitle": "Assistente Inteligente de Operação de Trading",
            "typing": "Carregando",
            "status_box_title": "Atualizações",
            "process_starting": "Iniciando {broker}...",
            "file_not_found": "Arquivo não encontrado: {file}",
            "broker_disconnected": "{broker} desconectado.",
            "broker_start_error": "Erro ao iniciar {broker}: {error}",
            "broker_locked_today": "Corretora **{broker}** bloqueada hoje. Você pode operar novamente amanhã.",
            "panel_title": "Painel",
            "account_menu_title": "Tipo de Conta",
            "account_demo_label": "DEMO (Treino)",
            "account_real_label": "REAL",
            "broker_menu_title": "Selecionar Corretora",
            "broker_active": "{broker} já está ativa.",
            "account_changed_demo_verbose": "Conta alterada: **DEMO (Treino)**",
            "accuracy_title": "Acurácia do Dia",
            "accuracy_status": "Acurácia",
            "accuracy_no_data": "Sem dados",
            "accuracy_total": "Hoje: {total} operações",
            "accuracy_empty": "Nenhuma operação hoje",
            "accuracy_hint": "Estatísticas do dia atual",
            "accuracy_hint_empty": "Comece a operar para ver estatísticas",
            "wins_label": "Wins",
            "losses_label": "Loss",
            "connect_iq": "Conectar IQ Option",
            "connect_bullex": "Conectar Bullex",
            "disconnect_iq": "Desconectar IQ Option",
            "disconnect_bullex": "Desconectar Bullex",
            "send": "Enviar",
            "stop": "Parar IA",
            "placeholder": "Digite sua mensagem...",
            "home_tooltip": "Voltar ao início",
            "welcome_message": "Olá {email}! Sou a **WS Trader IA**.\n\nVamos iniciar sua operação.\nFaça uma pergunta para IA ou conecte na corretora para treinamento ou operação real.",
            "sidebar_commands": "Comandos",
            "sidebar_connection": "Conexão",
            "sidebar_account": "Conta",
            "sidebar_info": "Info",
            "cmd_connect_iq": "conectar iq option",
            "cmd_connect_bullex": "conectar bullex",
            "cmd_disconnect": "desconectar",
            "cmd_demo": "mudar para demo",
            "cmd_real": "mudar para real",
            "cmd_status": "status",
            "cmd_help": "ajuda",
            "cmd_exit": "sair",
            "help_commands": "**Comandos**\n\n**Conexão**\n- `conectar iq option`\n- `conectar bullex`\n- `desconectar`\n\n**Conta**\n- `mudar para demo`\n- `mudar para real`\n\n**Info**\n- `status`\n- `ajuda`\n- `sair` (voltar ao login)\n\nVocê também pode perguntar sobre: funcionamento da IA, filtros, estatísticas e plataforma.",
            "disconnect_first": "Desconecte primeiro para trocar a conta.",
            "account_changed_real": "Conta alterada: **REAL**",
            "account_real_warning": "Atenção: operando com dinheiro real.",
            "account_changed_demo": "Conta alterada: **DEMO**",
            "status_title": "**Status**",
            "status_iq": "- IQ: **{status}**",
            "status_bullex": "- Bullex: **{status}**",
            "status_account": "- Conta: **{account}**",
            "status_gains": "- Ganhos: **R$ {gains:.2f}**",
            "pill_iq": "IQ: {status}",
            "pill_bullex": "Bullex: {status}",
            "pill_account": "Conta: {account}",
            "pill_gains": "Ganhos: R$ {gains:.2f}",
            "pill_gains_positive": "Ganhos: +R$ {gains:.2f}",
            "status_online": "Online",
            "status_operating": "Operando",
            # IA operation messages
            "bot_starting": "Iniciando IA de trading...",
            "bot_connecting_iq": "Conectando a IQ Option...",
            "bot_connecting_bullex": "Conectando a Bullex...",
            "bot_websocket_connected": "Websocket conectado",
            "bot_connected_iq": "**Conectado a IQ Option**",
            "bot_connected_bullex": "**Conectado a Bullex**",
            "bot_real_warning": "**ATENCAO: Usando conta REAL**",
            "bot_demo_account": "Usando conta DEMO",
            "bot_started": "**IA iniciada! Aguardando sinais de trading...**",
            "bot_connected": "**Conectado**",
            "bot_operating": "**Operando**",
            "ia_goal_reached": "**Meta do dia alcançada!** Lucro: R$ {profit} ({percent}%)",
            "ia_goal_reached_simple": "**Meta do dia alcançada!** IA finalizada.",
            "bot_stop_loss": "**STOP LOSS!** Perda: R$ -{loss} ({percent}%)",
            "bot_stop_loss_simple": "**STOP LOSS!** IA finalizada.",
            "bot_connection_lost": "**Conexão perdida.** Reconectando automaticamente...",
            "bot_stop_loss_msg": "**STOP LOSS ATINGIDO**\n\nA IA parou de operar para proteger sua banca.",
            "bot_goal_msg": "**Meta do dia alcançada**\n\nParabéns! A IA pausou as operações por hoje.",
            "lockout_goal_message": "**Meta do dia alcançada.** Excelente trabalho! Agora só amanhã para operar novamente.",
            "lockout_stop_message": "**Stop do dia atingido.** Pausar hoje protege sua banca. Amanhã você pode operar novamente.",
            "stop_motivational_no_history": "Revise se você costuma operar às {hour}. Sem histórico suficiente nesse horário, treine primeiro antes de operar.",
            "stop_motivational_low_accuracy": "A acurácia neste horário ({hour}) está baixa ({accuracy:.1f}%). Recomendo treino antes de operar.",
            "stop_motivational_ok": "A acurácia neste horário ({hour}) está em {accuracy:.1f}% com {total} operações. Operar com cautela e manter disciplina é o ideal.",
            "best_hours_title": "**Melhores horários (baseado no seu histórico):**",
            "best_hours_item": "- {hour}: {win_rate:.1f}% ({trades} operações)",
            "best_hours_no_data": "Ainda não há dados por horário suficientes no arquivo para calcular os melhores horários. Continue operando em DEMO para gerar histórico.",
            "best_hours_insufficient": "Há poucos dados por horário para concluir com segurança. Continue treinando para gerar mais estatísticas.",
            "best_hours_note_low": "Observação: a base ainda é pequena; trate estes horários como referência inicial.",
            "accuracy_summary": "Acurácia atual: {accuracy:.1f}% (Wins: {wins}, Losses: {losses}, Total: {total}).",
            "accuracy_summary_no_data": "Ainda não há dados suficientes para calcular a acurácia. Continue operando em DEMO para gerar histórico.",
            "ai_learning": "Nível da IA",
            "ai_phase_warmup": "Iniciante",
            "ai_phase_bayes": "Intermediário",
            "ai_phase_lgbm": "Avançado",
            "ai_phase_full": "Expert",
            "ai_phase_desc_warmup": "IA aprendendo os padrões do mercado.\nPrecisa de 15 operações para evoluir.",
            "ai_phase_desc_bayes": "IA filtrando sinais fracos.\nPrecisa de 30 operações para evoluir.",
            "ai_phase_desc_lgbm": "IA otimizando decisões.\nPrecisa de 50+ operações para nível máximo.",
            "ai_phase_desc_full": "IA calibrada com precisão máxima!\nTodos os módulos ativos.",
            "ai_trades_label": "Operações acumuladas"
        },
        "EN": {
            "chat_title": "WS TRADER — Chat",
            "chat_subtitle": "Intelligent Assistant • Technical Support",
            "typing": "Loading",
            "status_box_title": "Updates",
            "process_starting": "Starting {broker}...",
            "file_not_found": "File not found: {file}",
            "broker_disconnected": "{broker} disconnected.",
            "broker_start_error": "Error starting {broker}: {error}",
            "broker_locked_today": "Broker **{broker}** is locked for today. You can operate again tomorrow.",
            "panel_title": "Panel",
            "account_menu_title": "Account Type",
            "account_demo_label": "DEMO (Practice)",
            "account_real_label": "REAL",
            "broker_menu_title": "Select Broker",
            "broker_active": "{broker} is already active.",
            "account_changed_demo_verbose": "Account changed: **DEMO (Practice)**",
            "accuracy_title": "Today's Accuracy",
            "accuracy_status": "Accuracy",
            "accuracy_no_data": "No data",
            "accuracy_total": "Today: {total} operations",
            "accuracy_empty": "No operations today",
            "accuracy_hint": "Today's statistics",
            "accuracy_hint_empty": "Start operating to see stats",
            "wins_label": "Wins",
            "losses_label": "Losses",
            "connect_iq": "Connect IQ Option",
            "connect_bullex": "Connect Bullex",
            "disconnect_iq": "Disconnect IQ Option",
            "disconnect_bullex": "Disconnect Bullex",
            "send": "Send",
            "stop": "Stop AI",
            "placeholder": "Type your message...",
            "home_tooltip": "Back to home",
            "welcome_message": "Hello {email}! I'm **WS Trader AI**.\n\nLet's get started.\nAsk AI a question or connect to the broker for training or real trading.",
            "sidebar_commands": "Commands",
            "sidebar_connection": "Connection",
            "sidebar_account": "Account",
            "sidebar_info": "Info",
            "cmd_connect_iq": "connect iq option",
            "cmd_connect_bullex": "connect bullex",
            "cmd_disconnect": "disconnect",
            "cmd_demo": "switch to demo",
            "cmd_real": "switch to real",
            "cmd_status": "status",
            "cmd_help": "help",
            "cmd_exit": "exit",
            "help_commands": "**Commands**\n\n**Connection**\n- `connect iq option`\n- `connect bullex`\n- `disconnect`\n\n**Account**\n- `switch to demo`\n- `switch to real`\n\n**Info**\n- `status`\n- `help`\n- `exit` (back to login)\n\nYou can also ask about: AI functionality, filters, statistics and platform.",
            "disconnect_first": "Disconnect first to change the account.",
            "account_changed_real": "Account changed: **REAL**",
            "account_real_warning": "Warning: operating with real money.",
            "account_changed_demo": "Account changed: **DEMO**",
            "status_title": "**Status**",
            "status_iq": "- IQ: **{status}**",
            "status_bullex": "- Bullex: **{status}**",
            "status_account": "- Account: **{account}**",
            "status_gains": "- Gains: **$ {gains:.2f}**",
            "pill_iq": "IQ: {status}",
            "pill_bullex": "Bullex: {status}",
            "pill_account": "Account: {account}",
            "pill_gains": "Gains: $ {gains:.2f}",
            "pill_gains_positive": "Gains: +$ {gains:.2f}",
            "status_online": "Online",
            "status_operating": "Operating",
            # IA operation messages
            "bot_starting": "Starting trading AI...",
            "bot_connecting_iq": "Connecting to IQ Option...",
            "bot_connecting_bullex": "Connecting to Bullex...",
            "bot_websocket_connected": "Websocket connected",
            "bot_connected_iq": "**Connected to IQ Option**",
            "bot_connected_bullex": "**Connected to Bullex**",
            "bot_real_warning": "**WARNING: Using REAL account**",
            "bot_demo_account": "Using DEMO account",
            "bot_started": "**AI started! Waiting for trading signals...**",
            "bot_connected": "**Connected**",
            "bot_operating": "**Operating**",
            "bot_goal_reached": "**GOAL REACHED!** Profit: $ {profit} ({percent}%)",
            "bot_goal_reached_simple": "**GOAL REACHED!** AI stopped.",
            "bot_stop_loss": "**STOP LOSS!** Loss: $ -{loss} ({percent}%)",
            "bot_stop_loss_simple": "**STOP LOSS!** AI stopped.",
            "bot_connection_lost": "**Connection lost.** Reconnecting automatically...",
            "bot_stop_loss_msg": "**STOP LOSS REACHED**\n\nThe AI stopped operating to protect your bankroll.",
            "bot_goal_msg": "**GOAL REACHED**\n\nCongratulations! The AI reached the daily goal and stopped operating.",
            "lockout_goal_message": "**Daily goal hit.** Great job! You can operate again tomorrow.",
            "lockout_stop_message": "**Daily stop reached.** Pausing today protects your bankroll. You can operate again tomorrow.",
            "stop_motivational_no_history": "Review whether you usually operate around {hour}. There is not enough history in this time slot, so train first before operating.",
            "stop_motivational_low_accuracy": "Accuracy at {hour} is low ({accuracy:.1f}%). I recommend training before operating.",
            "stop_motivational_ok": "Accuracy at {hour} is {accuracy:.1f}% across {total} operations. Operate cautiously and keep discipline.",
            "best_hours_title": "**Best hours (based on your history):**",
            "best_hours_item": "- {hour}: {win_rate:.1f}% ({trades} trades)",
            "best_hours_no_data": "There is not enough per-hour data in the stats file to compute best hours yet. Keep operating in DEMO to build history.",
            "best_hours_insufficient": "There is too little per-hour data to conclude safely. Keep training to generate more stats.",
            "best_hours_note_low": "Note: the sample is still small; treat these hours as a preliminary reference.",
            "accuracy_summary": "Current accuracy: {accuracy:.1f}% (Wins: {wins}, Losses: {losses}, Total: {total}).",
            "accuracy_summary_no_data": "There is not enough data to calculate accuracy yet. Keep operating in DEMO to build history.",
            "ai_learning": "AI Level",
            "ai_phase_warmup": "Beginner",
            "ai_phase_bayes": "Intermediate",
            "ai_phase_lgbm": "Advanced",
            "ai_phase_full": "Expert",
            "ai_phase_desc_warmup": "AI learning market patterns.\nNeeds 15 operations to evolve.",
            "ai_phase_desc_bayes": "AI filtering weak signals.\nNeeds 30 operations to evolve.",
            "ai_phase_desc_lgbm": "AI optimizing decisions.\nNeeds 50+ operations for max level.",
            "ai_phase_desc_full": "AI calibrated with maximum precision!\nAll modules active.",
            "ai_trades_label": "Accumulated operations"
        }
    }
    return translations.get(lang, translations["PT"])

def _today_str():
    return datetime.now().date().isoformat()

def _normalize_lockout_state(data):
    if not isinstance(data, dict):
        return None
    if "brokers" in data:
        return data
    # Compatibilidade com formato antigo
    active = bool(data.get("active")) and data.get("date") == _today_str()
    reason = data.get("reason", "goal")
    return {
        "date": data.get("date", _today_str()),
        "brokers": {
            "iq_option": {"active": active, "reason": reason},
            "bullex": {"active": active, "reason": reason}
        }
    }

def load_lockout_state():
    try:
        if os.path.exists(LOCKOUT_FILE):
            with open(LOCKOUT_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:  # Arquivo vazio
                    pass  # Vai retornar default
                else:
                    raw = json.loads(content)
                    data = _normalize_lockout_state(raw)
                    if data and data.get("date") == _today_str():
                        return data
    except Exception as ex:
        logger.debug(f"[CHAT] Arquivo lockout inválido, será recriado: {ex}")
    return {
        "date": _today_str(),
        "brokers": {
            "iq_option": {"active": False, "reason": "goal"},
            "bullex": {"active": False, "reason": "goal"}
        }
    }

def save_lockout_state(data):
    try:
        os.makedirs(os.path.dirname(LOCKOUT_FILE), exist_ok=True)
        with open(LOCKOUT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as ex:
        logger.error(f"[CHAT] ❌ Erro ao salvar lockout: {ex}")

# =========================
# OpenAI Configuration
# =========================
try:
    from config_keys import OPENAI_API_KEY
except ImportError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY, timeout=30.0, max_retries=3)  # Timeout de 30 segundos com 3 retries

SYSTEM_INSTRUCTIONS_PT = """Você é o assistente da WS Trader AI, uma plataforma de trading automatizado desenvolvida pela WS Trader (wstrader.io).

RESPONSABILIDADES:
1. Suporte aos usuários sobre a plataforma
2. Explicar como a IA de trading funciona e seu processo de treinamento
3. Explicar como a acurácia aumenta com o tempo através do aprendizado contínuo
4. Interpretar e explicar dados do arquivo JSON de estatísticas da IA
5. Ajudar com dúvidas sobre conexão com corretoras (IQ Option e Bullex)

LIMITAÇÕES:
- Não dar conselhos financeiros
- Não recomendar operações específicas
- Não prever mercado
- SEMPRE recomendar usar conta DEMO primeiro para treinar
- Focar em explicação técnica e educacional

=== COMO A IA FUNCIONA ===

ESTRATÉGIA BASE:
- Estratégia: Pernada B (Price Action)
- Timeframes: M1 para Bullex e M5 para IQ Option
- Filtros inteligentes: anti-lateral e anti-esticado
- Análise de Suporte/Resistência em tempo real
- Detecção de padrões de reversão

PROCESSO DE TREINAMENTO E APRENDIZADO:
1. FASE INICIAL (Conta DEMO):
   - A IA começa com padrões básicos de Price Action
   - Analisa cada operação realizada (WIN ou LOSS)
   - Registra contexto de mercado, horário, volatilidade
   - Acurácia inicial: aproximadamente 60-65%

2. APRENDIZADO CONTÍNUO:
   - A cada operação, a IA ajusta seus parâmetros internos
   - Identifica quais condições de mercado geram mais acertos
   - Aprende a evitar armadilhas (movimentos laterais, esticados)
   - Refina os pontos de entrada baseado em resultados anteriores
   - Acurácia após 100+ operações: 70-75%
   - Acurácia após 500+ operações: 75-82%

3. OTIMIZAÇÃO AVANÇADA:
   - A IA cria "memória" de padrões que funcionam melhor
   - Ajusta agressividade conforme momento do mercado
   - Melhora detecção de suporte/resistência
   - Acurácia madura (1000+ operações): 78-85%

ESTATÍSTICAS EM TEMPO REAL:
- Todas as operações são salvas em arquivos JSON
- Métricas: Total de operações, wins, losses, taxa de acerto
- Histórico de desempenho por sessão
- Análise de padrões mais lucrativos

=== ANÁLISE DE MELHORES HORÁRIOS ===

QUANDO O USUÁRIO PERGUNTAR SOBRE MELHORES HORÁRIOS:
1. Consulte o arquivo JSON de estatísticas da IA (ws_ai_stats_m1.json)
2. Analise a seção "patterns" que contém padrões no formato:
   "PUT|sc14|pb3|re5|A13|eff1|fl6|dst5": {
     "trades": 25,
     "wins": 18,
     "losses": 7,
     "by_hour": {
       "14": {"wins": 5, "losses": 1},
       "19": {"wins": 8, "losses": 2},
       "21": {"wins": 5, "losses": 4}
     }
   }
3. Para cada padrão, acesse o campo "by_hour" que contém estatísticas por hora (0-23)
4. Agrupe TODAS as operações por horário somando wins/losses de TODOS os padrões:
   - Para hora "19": some wins e losses de todos os padrões que operaram às 19h
   - win_rate = total_wins_hora / (total_wins_hora + total_losses_hora) * 100
5. Identifique os horários com:
   - Maior win rate (acima de 65%)
   - Número mínimo de operações (pelo menos 5 trades para ser confiável)
6. Apresente ao usuário:
   - Os 3-5 melhores horários para operar
   - Taxa de acerto em cada horário
   - Número total de operações naquele horário
   - Se possível, analise qual direção (CALL/PUT) funciona melhor comparando padrões

EXEMPLO DE RESPOSTA:
"Baseado nas estatísticas da sua IA, os melhores horários são:
1. 19:00 (78% de acerto, 25 operações)
2. 14:00 (74% de acerto, 18 operações)
3. 21:00 (72% de acerto, 15 operações)
4. 10:00 (68% de acerto, 12 operações)

ATENÇÃO: Esses dados são baseados no histórico da sua IA. Continue treinando para melhorar a precisão."

=== TRADES NECESSÁRIOS (RESPONDER QUANDO PERGUNTAR) ===
Use as estatísticas em JSON para informar quantos trades já existem e quantos faltam.

COMO LER O JSON:
- Total de trades: use bandit.total se existir, senão some global.w + global.l
- Número de padrões: use len(bandit.arms) ou len(by_key)
- Winrate: wins / (wins + losses) usando global.w/global.l

FAIXAS RECOMENDADAS:
Por padrão (Bayes/UCB):
- Warmup: 0–14 trades
- Confiável: 15+ trades
- Ótimo: 30+ trades

Por horário:
- Mínimo: 5 trades/hora
- Confiável: 15+ trades/hora
- Ótimo: 30+ trades/hora

Estimativa global:
- Funcional: ~300–500 trades
- Boa: ~800–1000 trades
- Excelente: ~2000+ trades

REGRA DE RESPOSTA:
1) Mostre o status atual (total trades, padrões e winrate)
2) Diga em qual nível ele está e quanto falta para o próximo
3) Recomende operar em DEMO para completar o histórico

FORMATO OBRIGATÓRIO (SEM EXPLICAÇÕES LONGAS):
- Responder SOMENTE com estatística + tabela, sem explicar como a IA funciona.
- Incluir o ciclo em ASCII (igual ao exemplo), a tabela de ações por horário/padrão e um resumo curto.
- Mencionar se há sinais novos (padrões sem histórico) e se horários bons já estão bloqueados/liberados.

DICA IMPORTANTE:
- Se houver poucos dados (menos de 50 operações totais), avise que a análise ainda é preliminar
- Para horários com menos de 5 operações, mencione que os dados são insuficientes
- Incentive o usuário a continuar operando em DEMO para gerar mais dados estatísticos
- Explique que mercados mudam e os melhores horários podem variar com o tempo
- Horários ruins (win rate < 50%) devem ser evitados

=== GERENCIAMENTO DE BANCA ===

META DIÁRIA E PROTEÇÃO:
- Meta diária: configurável de 0,5% a 4% (máximo)
- Padrão: 1,5% de lucro sobre a banca
- Quando atinge a meta, a IA PARA automaticamente
- Proteção: Stop Loss automático para evitar grandes perdas
- Cada entrada: 1% da banca atual
- Martingale: NÃO utiliza (cada operação é sempre 1% da banca)

REGRAS DE SEGURANÇA:
1. Sempre começar em conta DEMO
2. Treinar até atingir acurácia consistente acima de 70%
3. Apenas passar para REAL após no mínimo 200 operações em DEMO
4. Nunca operar com dinheiro que você não pode perder

=== COMO USAR ===

MODO DEMO (TREINO):
1. Conecte em conta DEMO/PRACTICE
2. Deixe a IA operar e aprender
3. Monitore a acurácia aumentando
4. Analise o gráfico de pizza no painel
5. Observe padrões e momentos de melhor desempenho

MODO REAL:
1. Só passe para REAL após dominar DEMO
2. Comece com valores pequenos
3. Respeite a meta configurada (máximo 4% ao dia)
4. Não force operações - confie na IA
5. Monitore estatísticas constantemente

=== REQUISITOS DO PC ===

"Qual computador preciso para rodar o WS Trader?"
Requisitos MÍNIMOS:
- Sistema: Windows 10 ou superior (64 bits)
- Processador: Intel i3 / AMD Ryzen 3 (ou equivalente)
- Memória RAM: 4 GB
- Armazenamento: 500 MB livres
- Internet: Conexão estável (mínimo 5 Mbps)

Requisitos RECOMENDADOS:
- Sistema: Windows 10/11 (64 bits)
- Processador: Intel i5 / AMD Ryzen 5 ou superior
- Memória RAM: 8 GB ou mais
- Armazenamento: 1 GB livre (SSD preferível)
- Internet: Conexão estável (10+ Mbps)

OBSERVAÇÕES:
- O app roda 100% local no seu PC, sem servidor externo
- Não precisa de placa de vídeo dedicada
- Funciona em notebooks e desktops
- Quanto mais estável a internet, melhor a conexão com a corretora
- O app consome pouca CPU e memória, pode rodar em segundo plano

=== APRENDIZADO EM TEMPO REAL ===

A IA do WS Trader aprende AO VIVO, em tempo real, durante as operações. Não existe um período separado de "treinamento" — ela já começa operando e aprendendo simultaneamente.

COMO FUNCIONA:
- A cada operação (WIN ou LOSS), a IA registra o contexto completo: padrão, horário, volatilidade, direção
- Ela ajusta seus parâmetros internos IMEDIATAMENTE após cada resultado
- Não precisa pausar, reiniciar ou esperar — o aprendizado é contínuo e automático
- Quanto mais operações, mais precisa ela fica naturalmente

EVOLUÇÃO NATURAL:
- Primeiras operações: IA começa com padrões base (~60-65% acurácia)
- Após 50-100 operações: já identifica padrões do mercado (~68-72%)
- Após 200-500 operações: bem calibrada para seu perfil (~72-78%)
- Após 500+ operações: maturidade alta (~78-85%)

DIFERENÇA DO WS TRADER:
- NÃO precisa de horas ou dias de treinamento prévio
- NÃO precisa de configuração manual complexa
- A IA aprende ENQUANTO opera — basta conectar e deixar rodar
- Cada operação torna a IA mais inteligente para a próxima
- Os dados de aprendizado ficam salvos e persistem entre sessões

=== RESPOSTAS A PERGUNTAS COMUNS ===

"Qual computador preciso?"
→ Qualquer PC com Windows 10+, 4 GB de RAM e internet estável. Não precisa de máquina potente — o app é leve e roda em notebooks comuns.

"Como a IA aprende?"
→ A IA aprende EM TEMPO REAL, ao vivo, durante cada operação. Ela analisa cada resultado (WIN/LOSS) e ajusta automaticamente seus parâmetros. Não precisa de um período separado de treinamento — basta conectar e ela já começa a aprender.

"Quanto tempo leva para a IA ficar precisa?"
→ A IA já começa operando e aprendendo ao mesmo tempo. A acurácia começa em ~60-65% e aumenta naturalmente. Após 100-200 operações atinge ~70-75%. Após 500+ pode chegar a 78-85%. Tudo acontece automaticamente enquanto opera.

"Posso acelerar o treinamento?"
→ Sim! Operando em conta DEMO todos os dias, a IA acumula experiência mais rápido. Como ela aprende ao vivo, quanto mais operar, mais rápido ela evolui. Recomendamos deixar rodando em DEMO por alguns dias.

"Por que começar em DEMO?"
→ DEMO permite que a IA aprenda sem riscos. Ela aprende exatamente igual em DEMO e REAL — a diferença é que em DEMO você não arrisca dinheiro real enquanto a IA está evoluindo.

"Precisa de placa de vídeo?"
→ Não! O WS Trader roda inteiramente na CPU. Qualquer computador básico com Windows 10+ consegue rodar sem problemas.

"Vocês têm acesso à minha conta/senha?"
→ NÃO. O WS Trader NÃO armazena senhas em servidores e NÃO tem acesso à sua conta na corretora. Sua senha fica guardada SOMENTE no seu computador, de forma 100% local e privada. O sistema apenas faz a ponte entre você e a corretora para operar de forma inteligente.

"E os termos de uso da corretora?"
→ Ao utilizar qualquer corretora (IQ Option, Bullex, CasaTrader), você deve sempre seguir os termos de uso da própria corretora. O WS Trader é uma ferramenta de análise e automação — a responsabilidade pelo uso da conta na corretora é do usuário, conforme os termos da corretora.

=== SEGURANÇA E PRIVACIDADE ===

SEMPRE que o usuário perguntar sobre segurança, senhas, privacidade ou acesso:
- Reforçar que NÃO armazenamos senhas em nenhum servidor
- Reforçar que NÃO temos acesso às contas dos clientes nas corretoras
- Explicar que o sistema funciona como uma PONTE — conecta localmente à corretora para operar
- Tudo roda 100% no computador do usuário
- Sempre orientar o usuário a seguir os TERMOS DE USO da corretora
- Nunca prometer que o sistema burla ou ignora regras da corretora

IMPORTANTE:
- Quando perguntado sobre o aprendizado da IA, SEMPRE explicar o processo gradual de aumento de acurácia
- Enfatizar a importância do treinamento em conta DEMO
- Mencionar que os dados de treino ficam salvos e ajudam a IA a melhorar continuamente
- O gráfico de pizza no painel mostra a acurácia atual baseada nas estatísticas
- Você TEM acesso às estatísticas injetadas no contexto ("ESTATÍSTICAS ATUAIS DA IA"). Use esses dados para responder sobre melhores horários e desempenho por hora; não diga que não tem acesso.

Site oficial: https://wstrader.io

Seja educado, profissional, didático e sempre incentive o uso responsável e educativo da plataforma."""

SYSTEM_INSTRUCTIONS_EN = """You are the assistant for WS Trader AI, an automated trading platform developed by WS Trader (wstrader.io).

RESPONSIBILITIES:
1. Support users about the platform
2. Explain how the trading system works and its training process
3. Explain how accuracy improves over time through continuous learning
4. Guide users about settings, best practices and safe usage
5. Help with broker connection questions (IQ Option and Bullex)

IMPORTANT:
- Do not promise guaranteed profit or certainty of gains
- Avoid financial advice; focus on how the platform works
- You DO have access to the injected stats context ("ESTATÍSTICAS ATUAIS DA IA"). Use it to answer best-hours questions and per-hour performance; do not say you lack access.

PC REQUIREMENTS:
Minimum: Windows 10+ (64-bit), Intel i3/Ryzen 3, 4GB RAM, 500MB disk, stable internet (5+ Mbps)
Recommended: Windows 10/11, Intel i5/Ryzen 5+, 8GB RAM, 1GB SSD, stable internet (10+ Mbps)
No dedicated GPU needed. Runs on laptops and desktops. Low CPU/memory usage.

REAL-TIME LEARNING:
The AI learns LIVE during operations — no separate training period needed.
- Each trade result (WIN/LOSS) immediately updates the AI parameters
- No pausing, restarting or waiting — learning is continuous and automatic
- More operations = more accuracy, naturally
- First trades: ~60-65% accuracy → After 200+: ~72-78% → After 500+: ~78-85%
- Data persists between sessions

SECURITY & PRIVACY:
- We do NOT store passwords on any server
- We do NOT have access to user broker accounts
- The system acts as a LOCAL BRIDGE — connects to the broker from the user's own computer
- Everything runs 100% on the user's machine
- Always guide users to follow the broker's TERMS OF SERVICE
- Never promise circumventing broker rules

WHEN ASKED ABOUT HOW MANY TRADES ARE NEEDED:
Use the JSON stats to report current totals and how many trades remain.

HOW TO READ THE JSON:
- Total trades: use bandit.total if present, otherwise global.w + global.l
- Pattern count: len(bandit.arms) or len(by_key)
- Win rate: wins / (wins + losses) using global.w/global.l

RECOMMENDED TIERS:
Default (Bayes/UCB):
- Warmup: 0–14 trades
- Reliable: 15+ trades
- Great: 30+ trades

Per-hour:
- Minimum: 5 trades/hour
- Reliable: 15+ trades/hour
- Great: 30+ trades/hour

Global estimate:
- Functional: ~300–500 trades
- Good: ~800–1000 trades
- Excellent: ~2000+ trades

RESPONSE RULE:
1) Show current status (total trades, patterns, win rate)
2) State current tier and what is missing for the next tier
3) Recommend DEMO training to build history
"""


def read_ai_stats():
    """
    Lê estatísticas da IA de um arquivo ÚNICO unificado.

    IMPORTANTE: Sistema híbrido para preservar dados entre atualizações:
    1. Arquivo do usuário (~/.wstrader/ws_ai_stats_m1.json) - PERSISTE entre atualizações
    2. Arquivo seed (pasta do executável) - IA pré-treinada que vem com o app

    IQ Option e Bullex usam o MESMO arquivo para memória compartilhada.

    Prioridade:
    1. ~/.wstrader/ws_ai_stats_m1.json (dados do usuário - preservados)
    2. ws_ai_stats_m1_seed.json (IA pré-treinada do executável)
    3. ws_ai_stats_m1.json (fallback no diretório do projeto)
    """
    stats = {}
    base_dir = os.path.dirname(__file__)
    user_data_dir = os.path.join(os.path.expanduser("~"), ".wstrader")

    # PRIORIDADE 1: Arquivo do usuário (preservado entre atualizações)
    user_path = os.path.join(user_data_dir, "ws_ai_stats_m1.json")
    if os.path.exists(user_path):
        try:
            with open(user_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                stats["unified"] = data
                logger.info(f"[STATS] ✅ Arquivo do usuário carregado: {user_path}")
                return stats
        except Exception as e:
            logger.error(f"Erro ao ler arquivo do usuário: {e}")

    # PRIORIDADE 2: Arquivo seed (IA pré-treinada)
    seed_path = os.path.join(base_dir, "ws_ai_stats_m1_seed.json")
    if os.path.exists(seed_path):
        try:
            with open(seed_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                stats["unified"] = data
                logger.info(f"[STATS] ✅ Arquivo seed carregado: {seed_path}")
                return stats
        except Exception as e:
            logger.error(f"Erro ao ler arquivo seed: {e}")

    # FALLBACK: Arquivo no projeto (compatibilidade)
    project_path = os.path.join(base_dir, "ws_ai_stats_m1.json")
    if os.path.exists(project_path):
        try:
            with open(project_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                stats["unified"] = data
                logger.info(f"[STATS] ✅ Arquivo do projeto carregado: {project_path}")
                return stats
        except Exception as e:
            logger.error(f"Erro ao ler arquivo do projeto: {e}")

    logger.warning("[STATS] ⚠️ Nenhum arquivo de estatísticas encontrado")
    return stats


def get_ai_response(user_message: str, conversation_history: list, system_instructions: str) -> str:
    if not client:
        if system_instructions == SYSTEM_INSTRUCTIONS_EN:
            return (
                "⚠️ OpenAI is not configured.\n\n"
                "Set the **OPENAI_API_KEY** environment variable and restart the app.\n"
                "Example (Windows PowerShell):\n"
                "`$env:OPENAI_API_KEY=\"YOUR_KEY\"`"
            )
        return (
            "⚠️ OpenAI não configurada.\n\n"
            "Defina a variável de ambiente **OPENAI_API_KEY** e reinicie o app.\n"
            "Exemplo (Windows PowerShell):\n"
            "`$env:OPENAI_API_KEY=\"SUA_CHAVE\"`"
        )

    try:
        ai_stats = read_ai_stats()
        stats_context = ""
        if ai_stats:
            stats_context = "\n\nESTATÍSTICAS ATUAIS DA IA:\n" + json.dumps(
                ai_stats, indent=2, ensure_ascii=False
            )

        messages = [{"role": "system", "content": system_instructions + stats_context}]
        messages.extend(conversation_history[-10:])
        messages.append({"role": "user", "content": user_message})

        # Ajuste de modelo: use o que você já vinha usando.
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )
        return resp.choices[0].message.content

    except Exception as e:
        logger.error(f"Erro ao obter resposta da OpenAI: {e}")
        return (
            f"❌ Erro ao processar sua mensagem: {str(e)}\n\n"
            "Tente novamente ou use: **ajuda**, **status**, **conectar iq option**."
        )


def chat_screen(page: ft.Page, email: str, password: str):
    logger.info(f"Carregando WS Trader AI para {email}")

    # Carregar idioma
    selected_lang = load_language_from_file()
    t = get_chat_translations(selected_lang)
    logger.info(f"✅ [CHAT] Idioma selecionado: '{selected_lang}'")
    system_instructions = SYSTEM_INSTRUCTIONS_EN if selected_lang == "EN" else SYSTEM_INSTRUCTIONS_PT

    # =========================
    # Estado
    # =========================
    conversation_history = []
    broker_connected = {"iq_option": False, "bullex": False, "casatrader": False}
    broker_processes = {"iq_option": None, "bullex": None, "casatrader": None}
    broker_connecting = {"value": False}
    account_type = {"value": "DEMO"}
    ganhos_acumulados = {"value": 0.0}
    meta_diaria = {"value": 0.0}  # Será calculado quando conectar (saldo * meta%)
    meta_batida_hoje = {"value": False}  # Controle para mostrar confete só uma vez
    
    # Arquivo para controle de meta batida por dia
    META_LOCKOUT_FILE = os.path.join(os.path.expanduser("~"), ".wstrader", "meta_lockout.json")
    
    # Meta batida SEPARADA por corretora E por tipo de conta (DEMO/REAL)
    meta_batida_broker = {
        "iq_option": {"DEMO": False, "REAL": False},
        "bullex": {"DEMO": False, "REAL": False},
        "casatrader": {"DEMO": False, "REAL": False}
    }
    ganhos_broker = {
        "iq_option": {"DEMO": 0.0, "REAL": 0.0},
        "bullex": {"DEMO": 0.0, "REAL": 0.0},
        "casatrader": {"DEMO": 0.0, "REAL": 0.0}
    }

    # === Helpers para acesso com separação DEMO/REAL ===
    def _get_ganhos(bk, acct_override=None):
        """Retorna ganhos do broker para conta especificada ou conta atual (DEMO/REAL)"""
        acct = acct_override or account_type["value"]
        d = ganhos_broker.get(bk)
        if isinstance(d, dict):
            return d.get(acct, 0.0)
        return 0.0

    def _set_ganhos(bk, val, acct_override=None):
        """Define ganhos do broker para conta especificada ou conta atual (DEMO/REAL)"""
        acct = acct_override or account_type["value"]
        if bk not in ganhos_broker or not isinstance(ganhos_broker[bk], dict):
            ganhos_broker[bk] = {"DEMO": 0.0, "REAL": 0.0}
        ganhos_broker[bk][acct] = val

    def _add_ganhos(bk, val, acct_override=None):
        """Soma valor aos ganhos do broker para conta especificada ou conta atual"""
        _set_ganhos(bk, _get_ganhos(bk, acct_override) + val, acct_override)

    def _get_meta_batida(bk, acct_override=None):
        """Verifica se meta foi batida para broker + conta especificada ou conta atual"""
        acct = acct_override or account_type["value"]
        d = meta_batida_broker.get(bk)
        if isinstance(d, dict):
            return d.get(acct, False)
        return False

    def _set_meta_batida(bk, val, acct_override=None):
        """Define meta batida para broker + conta especificada ou conta atual"""
        acct = acct_override or account_type["value"]
        if bk not in meta_batida_broker or not isinstance(meta_batida_broker[bk], dict):
            meta_batida_broker[bk] = {"DEMO": False, "REAL": False}
        meta_batida_broker[bk][acct] = val

    def _any_meta_batida_current():
        """Verifica se alguma meta foi batida para conta atual"""
        return any(_get_meta_batida(bk) for bk in ["iq_option", "bullex", "casatrader"])

    def _sum_ganhos_current():
        """Soma ganhos de todas corretoras para conta atual"""
        return sum(_get_ganhos(bk) for bk in ["iq_option", "bullex", "casatrader"])
    
    _meta_lockout_loaded = {"value": False}

    def load_meta_lockout_broker(broker_key: str = None):
        """Carrega o estado de meta batida do arquivo - SEPARADO POR CORRETORA E CONTA"""
        # CORREÇÃO: Se já carregou, usar dados em memória (evita sobrescrever ganhos acumulados)
        if _meta_lockout_loaded["value"]:
            if broker_key:
                return _get_meta_batida(broker_key)
            return _any_meta_batida_current()
        
        try:
            if os.path.exists(META_LOCKOUT_FILE):
                with open(META_LOCKOUT_FILE, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        return False
                    data = json.loads(content)
                    today = date.today().isoformat()
                    
                    # Se data é diferente, ignora (novo dia)
                    if data.get("date") != today:
                        return False
                    
                    # Carregar estado por corretora
                    brokers_data = data.get("brokers", {})
                    
                    # Formato novo - por corretora e conta (DEMO/REAL)
                    for bk in ["iq_option", "bullex", "casatrader"]:
                        bk_data = brokers_data.get(bk, {})
                        # Verificar se tem separação DEMO/REAL
                        if "DEMO" in bk_data or "REAL" in bk_data:
                            # Formato com separação DEMO/REAL
                            for acct in ["DEMO", "REAL"]:
                                acct_data = bk_data.get(acct, {})
                                meta_batida_broker[bk][acct] = acct_data.get("meta_batida", False)
                                # CORREÇÃO: Só carregar ganhos se ainda estiver zerado
                                # (daily_data JSON é a fonte autoritativa para ganhos)
                                saved_ganhos = acct_data.get("ganhos", 0.0)
                                if ganhos_broker[bk][acct] == 0.0 and saved_ganhos != 0.0:
                                    ganhos_broker[bk][acct] = saved_ganhos
                                if meta_batida_broker[bk][acct]:
                                    logger.info(f"[META] {bk} {acct} meta batida hoje! Ganhos: ${ganhos_broker[bk][acct]:.2f}")
                        else:
                            # Compatibilidade: formato antigo sem separação
                            old_meta = bk_data.get("meta_batida", False)
                            old_ganhos = bk_data.get("ganhos", 0.0)
                            if old_meta:
                                # Migrar para DEMO por padrão
                                meta_batida_broker[bk]["DEMO"] = True
                                if ganhos_broker[bk]["DEMO"] == 0.0 and old_ganhos != 0.0:
                                    ganhos_broker[bk]["DEMO"] = old_ganhos
                                logger.info(f"[META] Migrado {bk}: meta_batida DEMO Ganhos: ${ganhos_broker[bk]['DEMO']:.2f}")
                    _meta_lockout_loaded["value"] = True
                    
                    # Se broker_key especificado, retorna status específico para conta atual
                    if broker_key:
                        return _get_meta_batida(broker_key)
                    
                    # Retorna True se qualquer uma batida para conta atual
                    any_batida = _any_meta_batida_current()
                    if any_batida:
                        meta_batida_hoje["value"] = True
                        ganhos_acumulados["value"] = _sum_ganhos_current()
                    return any_batida
        except Exception as e:
            logger.debug(f"[META] Arquivo lockout inválido, será recriado: {e}")
        return False
    
    def save_meta_lockout_broker(broker_key: str = None):
        """Salva o estado de meta batida no arquivo - SEPARADO POR CORRETORA E CONTA"""
        try:
            os.makedirs(os.path.dirname(META_LOCKOUT_FILE), exist_ok=True)
            data = {
                "date": date.today().isoformat(),
                "timestamp": datetime.now().isoformat(),
                "brokers": {}
            }
            for bk in ["iq_option", "bullex", "casatrader"]:
                data["brokers"][bk] = {
                    "DEMO": {
                        "meta_batida": meta_batida_broker[bk].get("DEMO", False),
                        "ganhos": ganhos_broker[bk].get("DEMO", 0.0)
                    },
                    "REAL": {
                        "meta_batida": meta_batida_broker[bk].get("REAL", False),
                        "ganhos": ganhos_broker[bk].get("REAL", 0.0)
                    }
                }
            # Compatibilidade
            data["meta_batida"] = _any_meta_batida_current()
            data["ganhos"] = _sum_ganhos_current()
            data["meta_valor"] = meta_diaria["value"]
            
            with open(META_LOCKOUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            acct = account_type["value"]
            if broker_key:
                logger.info(f"[META] Lockout {broker_key} {acct} salvo: meta_batida={_get_meta_batida(broker_key)}")
            else:
                logger.info(f"[META] Lockout salvo ({acct}): iq={_get_meta_batida('iq_option')} bullex={_get_meta_batida('bullex')}")
        except Exception as e:
            logger.error(f"[META] Erro ao salvar lockout: {e}")
    
    def is_meta_locked_broker(broker_key: str):
        """Verifica se a meta já foi batida hoje PARA UMA CORRETORA + CONTA ESPECÍFICA DO BROKER"""
        load_meta_lockout_broker()
        # Usar broker_accounts para determinar a conta correta deste broker
        try:
            acct = broker_accounts.get(broker_key, account_type["value"])
        except Exception:
            acct = account_type["value"]
        return _get_meta_batida(broker_key, acct)
    
    # Funções de compatibilidade (mantidas para não quebrar código existente)
    def load_meta_lockout():
        return load_meta_lockout_broker()
    
    def save_meta_lockout():
        return save_meta_lockout_broker()
    
    def is_meta_locked_today():
        """Verifica se TODAS as metas foram batidas para conta atual"""
        load_meta_lockout_broker()
        return all(_get_meta_batida(bk) for bk in ["iq_option", "bullex", "casatrader"])

    # Estatísticas do DIA ATUAL POR CORRETORA (reseta a cada novo dia)
    from datetime import date
    daily_stats = {
        "date": date.today().isoformat(),
        "DEMO": {"wins": 0, "losses": 0},
        "REAL": {"wins": 0, "losses": 0}
    }
    
    # Estatísticas SEPARADAS por corretora E por conta (DEMO/REAL)
    daily_stats_broker = {
        "iq_option": {"DEMO": {"wins": 0, "losses": 0}, "REAL": {"wins": 0, "losses": 0}},
        "bullex": {"DEMO": {"wins": 0, "losses": 0}, "REAL": {"wins": 0, "losses": 0}},
        "casatrader": {"DEMO": {"wins": 0, "losses": 0}, "REAL": {"wins": 0, "losses": 0}}
    }

    # === Helpers para acesso stats com separação DEMO/REAL ===
    def _get_broker_wins(bk, acct_override=None):
        acct = acct_override or account_type["value"]
        return daily_stats_broker.get(bk, {}).get(acct, {}).get("wins", 0)

    def _get_broker_losses(bk, acct_override=None):
        acct = acct_override or account_type["value"]
        return daily_stats_broker.get(bk, {}).get(acct, {}).get("losses", 0)

    def _get_total_wins():
        acct = account_type["value"]
        return daily_stats.get(acct, {}).get("wins", 0)

    def _get_total_losses():
        acct = account_type["value"]
        return daily_stats.get(acct, {}).get("losses", 0)
    
    # Meta diária SEPARADA por corretora
    meta_diaria_broker = {
        "iq_option": 0.0,
        "bullex": 0.0,
        "casatrader": 0.0
    }

    def check_reset_daily_stats():
        """Verifica se mudou o dia e reseta as estatísticas se necessário"""
        today = date.today().isoformat()
        if daily_stats["date"] != today:
            logger.info(f"[DAILY] Novo dia detectado: {daily_stats['date']} -> {today}. Resetando estatísticas.")
            daily_stats["date"] = today
            daily_stats["DEMO"] = {"wins": 0, "losses": 0}
            daily_stats["REAL"] = {"wins": 0, "losses": 0}
            ganhos_acumulados["value"] = 0.0
            # Reset metas de todas as corretoras (DEMO e REAL)
            for bk in ["iq_option", "bullex", "casatrader"]:
                meta_batida_broker[bk] = {"DEMO": False, "REAL": False}
                ganhos_broker[bk] = {"DEMO": 0.0, "REAL": 0.0}
                daily_stats_broker[bk] = {"DEMO": {"wins": 0, "losses": 0}, "REAL": {"wins": 0, "losses": 0}}
                meta_diaria_broker[bk] = 0.0
            meta_batida_hoje["value"] = False  # Reset flag de meta batida
            save_meta_lockout()  # Salvar reset

    # ===================== SALVAR DADOS DIÁRIOS PARA RELATÓRIO =====================
    DAILY_REPORT_DIR = os.path.join(os.path.expanduser("~"), ".wstrader", "daily_data")

    def _restore_daily_stats_from_json():
        """Restaura wins/losses/ganhos do dia a partir dos JSONs salvos (para não zerar ao reiniciar)"""
        try:
            today_str = datetime.now().strftime("%Y-%m-%d")
            for acct in ["DEMO", "REAL"]:
                for bk in ["iq_option", "bullex", "casatrader"]:
                    fpath = os.path.join(DAILY_REPORT_DIR, f"{bk}_{today_str}_{acct}.json")
                    if os.path.exists(fpath):
                        try:
                            with open(fpath, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            saved_wins = data.get("wins", 0)
                            saved_losses = data.get("losses", 0)
                            saved_profit = data.get("profit", 0.0)
                            # Só restaurar se for maior que o atual (evitar regredir)
                            cur_w = daily_stats_broker[bk][acct]["wins"]
                            cur_l = daily_stats_broker[bk][acct]["losses"]
                            if saved_wins > cur_w:
                                daily_stats_broker[bk][acct]["wins"] = saved_wins
                            if saved_losses > cur_l:
                                daily_stats_broker[bk][acct]["losses"] = saved_losses
                            # Restaurar ganhos se zerado
                            if ganhos_broker[bk][acct] == 0.0 and saved_profit != 0.0:
                                ganhos_broker[bk][acct] = saved_profit
                            logger.info(f"[RESTORE] {bk} {acct}: W:{daily_stats_broker[bk][acct]['wins']} L:{daily_stats_broker[bk][acct]['losses']} P:R${ganhos_broker[bk][acct]:.2f}")
                        except Exception as ex:
                            logger.debug(f"[RESTORE] Erro ao ler {bk} {acct}: {ex}")
            # Atualizar totais para conta atual
            cur_acct = account_type["value"]
            daily_stats[cur_acct]["wins"] = sum(daily_stats_broker[bk][cur_acct]["wins"] for bk in ["iq_option", "bullex", "casatrader"])
            daily_stats[cur_acct]["losses"] = sum(daily_stats_broker[bk][cur_acct]["losses"] for bk in ["iq_option", "bullex", "casatrader"])
            # Determinar broker ativo (get_active_broker pode não existir ainda no startup)
            try:
                active = get_active_broker() or "iq_option"
            except Exception:
                active = next((bk for bk in ["iq_option", "bullex", "casatrader"] if broker_connected.get(bk, False)), "iq_option")
            ganhos_acumulados["value"] = _get_ganhos(active)
        except Exception as e:
            logger.debug(f"[RESTORE] Erro geral: {e}")

    # Restaurar histórico ao iniciar
    _restore_daily_stats_from_json()

    def _save_daily_report_data(bkey, wins, losses, profit, acct_override=None):
        """Salva dados do dia para alimentar os gráficos de relatório (separado por DEMO/REAL)"""
        try:
            os.makedirs(DAILY_REPORT_DIR, exist_ok=True)
            today_str = datetime.now().strftime("%Y-%m-%d")
            acct = acct_override or account_type["value"]
            fpath = os.path.join(DAILY_REPORT_DIR, f"{bkey}_{today_str}_{acct}.json")
            data = {}
            if os.path.exists(fpath):
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    data = {}
            data["wins"] = wins
            data["losses"] = losses
            data["profit"] = profit
            data["broker"] = bkey
            data["date"] = today_str
            data["account"] = acct
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"[REPORT] Dados salvos: {bkey} {acct} W:{wins} L:{losses} P:R${profit:.2f}")
        except Exception as e:
            logger.error(f"[REPORT] Erro ao salvar dados: {e}")

    # Arquivo para salvar LOSS para análise
    LOSS_LOG_FILE = os.path.join(os.path.expanduser("~"), ".wstrader", "loss_analysis.json")
    
    def save_loss_for_analysis(valor: float, ativo: str = "", direcao: str = "", padrao: str = "", tendencia: str = "", motivo: str = ""):
        """Salva informações do LOSS para análise posterior"""
        try:
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(LOSS_LOG_FILE), exist_ok=True)
            
            # Carregar dados existentes
            loss_data = []
            if os.path.exists(LOSS_LOG_FILE):
                try:
                    with open(LOSS_LOG_FILE, 'r', encoding='utf-8') as f:
                        loss_data = json.load(f)
                except:
                    loss_data = []
            
            # Adicionar novo LOSS
            loss_entry = {
                "timestamp": datetime.now().isoformat(),
                "data": date.today().isoformat(),
                "hora": datetime.now().strftime("%H:%M:%S"),
                "valor": valor,
                "ativo": ativo,
                "direcao": direcao,
                "padrao": padrao,
                "tendencia": tendencia,
                "motivo": motivo,
                "broker": "iq_option" if broker_connected.get("iq_option") else "bullex",
                "conta": account_type["value"]
            }
            loss_data.append(loss_entry)
            
            # Manter apenas últimos 500 registros
            if len(loss_data) > 500:
                loss_data = loss_data[-500:]
            
            # Salvar
            with open(LOSS_LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(loss_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[LOSS_ANALYSIS] 📊 LOSS salvo para análise: {ativo} {direcao} ${valor:.2f} | Padrão: {padrao}")
            
        except Exception as e:
            logger.error(f"[LOSS_ANALYSIS] Erro ao salvar LOSS: {e}")

    # Variável para guardar última operação (para capturar info do LOSS)
    ultima_operacao = {"ativo": "", "direcao": "", "padrao": "", "tendencia": "", "score": ""}

    def show_confetti_animation():
        """Mostra tela de celebracao quando bate a meta - Design Moderno"""
        valor = ganhos_acumulados['value']
        # Calcula porcentagem real: ganho / saldo_inicial * 100
        # saldo_inicial = meta_valor / (meta_pct / 100)
        active_bk = get_active_broker() or "iq_option"
        meta_val = meta_diaria_broker.get(active_bk, meta_diaria['value'])
        meta_pct = broker_goals.get(active_bk, 4.0)
        if meta_val > 0 and meta_pct > 0:
            saldo_inicial = meta_val / (meta_pct / 100.0)
            percent = (valor / saldo_inicial * 100) if saldo_inicial > 0 else meta_pct
        else:
            percent = meta_pct

        # Card central com design moderno
        goal_card = ft.Container(
            content=ft.Row([
                # Lado esquerdo - Imagem
                ft.Container(
                    content=ft.Image(
                        src="Img/meta_batida.png",
                        width=180,
                        height=180,
                        fit="contain",
                    ),
                    padding=ft.padding.all(20),
                ),
                # Lado direito - Texto moderno
                ft.Container(
                    content=ft.Column([
                        ft.Container(
                            content=ft.Row([
                                ft.Icon(ft.Icons.EMOJI_EVENTS_ROUNDED, color="#FFD700", size=28),
                                ft.Text(
                                    "META ALCANÇADA!",
                                    size=24,
                                    weight=ft.FontWeight.W_700,
                                    color="#FFFFFF",
                                ),
                            ], spacing=10),
                        ),
                        ft.Container(height=8),
                        ft.Text(
                            "Parabéns! Você atingiu sua meta diária.",
                            size=14,
                            color="#A8B1BD",
                            weight=ft.FontWeight.W_400,
                        ),
                        ft.Container(height=20),
                        # Valor do lucro
                        ft.Container(
                            content=ft.Column([
                                ft.Text("LUCRO DO DIA", size=11, color="#6B7280", weight=ft.FontWeight.W_600),
                                ft.Container(height=4),
                                ft.Row([
                                    ft.Text(
                                        f"+R$ {valor:.2f}",
                                        size=32,
                                        weight=ft.FontWeight.W_700,
                                        color="#52C97C",
                                    ),
                                    ft.Container(
                                        content=ft.Text(
                                            f"+{percent:.1f}%",
                                            size=14,
                                            weight=ft.FontWeight.W_600,
                                            color="#52C97C",
                                        ),
                                        bgcolor=ft.Colors.with_opacity(0.15, "#52C97C"),
                                        border_radius=6,
                                        padding=ft.padding.symmetric(horizontal=10, vertical=4),
                                    ),
                                ], spacing=12, alignment=ft.MainAxisAlignment.START),
                            ]),
                        ),
                        ft.Container(height=20),
                        # Status
                        ft.Container(
                            content=ft.Row([
                                ft.Container(
                                    width=8,
                                    height=8,
                                    bgcolor="#F59E0B",
                                    border_radius=4,
                                ),
                                ft.Text(
                                    "IA pausada até amanhã",
                                    size=13,
                                    color="#F59E0B",
                                    weight=ft.FontWeight.W_500,
                                ),
                            ], spacing=8),
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=0,
                    ),
                    padding=ft.padding.only(right=30, top=20, bottom=20),
                    expand=True,
                ),
            ],
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            width=520,
            bgcolor="#1A1E23",
            border_radius=16,
            border=ft.border.all(1, ft.Colors.with_opacity(0.1, ft.Colors.WHITE)),
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=40,
                color=ft.Colors.with_opacity(0.3, "#000000"),
            ),
        )

        # Botão fechar
        close_btn = ft.Container(
            content=ft.Text("Clique em qualquer lugar para fechar", size=12, color="#6B7280"),
            margin=ft.margin.only(top=20),
        )

        # Overlay de celebracao (moderno)
        confetti_overlay = ft.Container(
            content=ft.Column([
                goal_card,
                close_btn,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=ft.Colors.with_opacity(0.85, "#0E1114"),
            expand=True,
            alignment=ft.Alignment(0, 0),
            on_click=lambda e: close_confetti(),
        )

        def close_confetti():
            try:
                if confetti_overlay in page.overlay:
                    page.overlay.remove(confetti_overlay)
                    page.update()
            except:
                pass

        page.overlay.append(confetti_overlay)
        page.update()

        # Fechar automaticamente apos 30 segundos (ou clique do usuario)
        def auto_close():
            import time as t
            t.sleep(30)
            close_confetti()

        threading.Thread(target=auto_close, daemon=True).start()

    # Pubsub para comunicação entre threads
    def on_message(message):
        """Handler para mensagens pubsub"""
        try:
            fn, args, kwargs = message
            fn(*args, **kwargs)
            page.update()
        except Exception as e:
            logger.error(f"Erro no pubsub handler: {e}", exc_info=False)

    page.pubsub.subscribe(on_message)

    # =========================
    # Tema/cores (dark premium)
    # =========================
    COLORS = {
        "bg": "#0E1114",           # Fundo principal (preto grafite)
        "panel": "#181C20",        # Painéis laterais (mais claro que bg)
        "card": "#1E2328",         # Cards e containers (destaque moderado)
        "card2": "#1A1E23",        # Cards secundários
        "border": "#2A2F35",       # Bordas sutis
        "text": "#F0F2F5",         # Texto principal (branco quente)
        "muted": "#A8B1BD",        # Texto secundário (cinza claro)
        "muted2": "#8A95A3",       # Texto terciário
        "brand": "#FF7A1A",        # Laranja principal (mantido)
        "brand2": "#FF8F3D",       # Laranja hover
        "blue": "#5B9FD8",         # Azul mais suave
        "green": "#52C97C",        # Verde mais profissional
        "red": "#E85D5D",          # Vermelho mais suave
        "yellow": "#F59E0B",       # Amarelo para avisos
        "status_green": "#6EE7B7", # Status verde claro
        "status_red": "#F87171",   # Status vermelho claro
        "console_text": "#E5E7EB"  # Texto do console
    }

    GLASS_BG = ft.Colors.with_opacity(0.06, ft.Colors.WHITE)
    GLASS_BG_SOFT = ft.Colors.with_opacity(0.08, ft.Colors.WHITE)
    GLASS_BORDER = ft.Colors.with_opacity(0.10, ft.Colors.WHITE)


    # =========================
    # Page config
    # =========================
    page.title = t["chat_title"]
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    page.spacing = 0
    page.window.width = 1200
    page.window.height = 750
    page.window.resizable = True
    page.window.maximizable = True
    page.window.minimizable = True

    page.bgcolor = ft.LinearGradient(
        begin=ft.Alignment(-1, -1),
        end=ft.Alignment(1, 1),
        colors=["#0E1114", "#111417"]
    )

    # =========================
    # Helpers thread-safe
    # =========================
    def ui(fn, *args, **kwargs):
        """Envia atualização via pubsub para thread-safety"""
        try:
            # send_all só aceita a mensagem como argumento
            page.pubsub.send_all((fn, args, kwargs))
        except Exception as e:
            logger.error(f"Erro ao enviar via pubsub: {e}")
            # Fallback: tenta executar diretamente
            try:
                fn(*args, **kwargs)
                page.update()
            except Exception as e2:
                logger.error(f"Erro no fallback: {e2}")

    def any_broker_connected():
        return broker_connecting["value"] or broker_connected["iq_option"] or broker_connected["bullex"] or broker_connected["casatrader"]

    def _get_stats_source(ai_stats: dict):
        if ai_stats.get("unified"):
            return ai_stats.get("unified")
        if ai_stats.get("iq_option"):
            return ai_stats.get("iq_option")
        if ai_stats.get("bullex"):
            return ai_stats.get("bullex")
        return None

    def build_best_hours_message():
        ai_stats = read_ai_stats()
        stats = _get_stats_source(ai_stats or {})
        if not stats:
            return t["best_hours_no_data"]

        patterns = stats.get("patterns", {}) or {}
        if not patterns:
            return t["best_hours_no_data"]

        hourly = {}
        total_trades = 0
        for _, pdata in patterns.items():
            by_hour = pdata.get("by_hour") or {}
            for hour_str, vals in by_hour.items():
                try:
                    hour = int(hour_str)
                except Exception:
                    continue
                wins = int(vals.get("wins", 0))
                losses = int(vals.get("losses", 0))
                trades = wins + losses
                if trades == 0:
                    continue
                bucket = hourly.setdefault(hour, {"wins": 0, "losses": 0, "trades": 0})
                bucket["wins"] += wins
                bucket["losses"] += losses
                bucket["trades"] += trades
                total_trades += trades

        if not hourly:
            return t["best_hours_no_data"]

        min_trades_per_hour = 5
        ranked = []
        for hour, data in hourly.items():
            if data["trades"] < min_trades_per_hour:
                continue
            win_rate = (data["wins"] / data["trades"] * 100) if data["trades"] > 0 else 0
            ranked.append((hour, win_rate, data["trades"]))

        if not ranked:
            return t["best_hours_insufficient"]

        ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
        top = ranked[:5]

        lines = [t["best_hours_title"]]
        for hour, win_rate, trades in top:
            lines.append(t["best_hours_item"].format(hour=f"{hour:02d}:00", win_rate=win_rate, trades=trades))

        if total_trades < 50:
            lines.append("")
            lines.append(t["best_hours_note_low"])

        return "\n".join(lines)

    def build_accuracy_summary_message():
        ai_stats = read_ai_stats()
        stats = _get_stats_source(ai_stats or {})
        if not stats:
            return t["accuracy_summary_no_data"]

        wins = 0
        losses = 0
        total_trades = 0
        has_real_data = False

        patterns = stats.get("patterns", {})
        meta = stats.get("meta", {})

        if patterns:
            total_wins = 0
            total_losses = 0
            total_trades = 0

            for _, pattern_data in patterns.items():
                total_wins += pattern_data.get("wins", 0)
                total_losses += pattern_data.get("losses", 0)
                total_trades += pattern_data.get("trades", 0)

            wins = total_wins
            losses = total_losses

            meta_total = meta.get("total", 0)
            if meta_total > 0 and meta_total != total_trades:
                total_trades = meta_total

            has_real_data = True
        else:
            global_stats = stats.get("global", {})
            wins = global_stats.get("w", 0)
            losses = global_stats.get("l", 0)
            total_trades = wins + losses
            has_real_data = True if total_trades > 0 else False

        total = wins + losses
        if total == 0 or not has_real_data:
            return t["accuracy_summary_no_data"]

        accuracy = (wins / total) * 100
        return t["accuracy_summary"].format(
            accuracy=accuracy,
            wins=wins,
            losses=losses,
            total=total_trades if total_trades > 0 else total,
        )

    lockout_state = load_lockout_state()

    def is_lockout_active(broker_key: str):
        if lockout_state.get("date") != _today_str():
            return False
        return bool(lockout_state.get("brokers", {}).get(broker_key, {}).get("active"))

    def get_lockout_reason(broker_key: str):
        return lockout_state.get("brokers", {}).get(broker_key, {}).get("reason", "goal")

    def _hourly_stats():
        try:
            from operations_manager import OperationsManager
            ops_manager = OperationsManager(email)
            ops = ops_manager.get_all_operations() or []
        except Exception as ex:
            logger.warning(f"[CHAT] Sem histórico de operações: {ex}")
            return {"total": 0, "win_rate": 0.0}

        now = datetime.now()
        hour = now.hour
        wins = 0
        losses = 0
        for op in ops:
            ts = op.get("timestamp")
            result = op.get("result")
            if not ts or result not in ("win", "loss"):
                continue
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            if dt.hour == hour:
                if result == "win":
                    wins += 1
                else:
                    losses += 1
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0.0
        return {"total": total, "win_rate": win_rate}

    def build_stop_motivational_message():
        hour_str = datetime.now().strftime("%H:00")
        stats = _hourly_stats()
        if stats["total"] == 0:
            return t["stop_motivational_no_history"].format(hour=hour_str)
        if stats["win_rate"] < 55:
            return t["stop_motivational_low_accuracy"].format(hour=hour_str, accuracy=stats["win_rate"])
        return t["stop_motivational_ok"].format(hour=hour_str, accuracy=stats["win_rate"], total=stats["total"])

    def activate_daily_lockout(reason: str, broker_key: str):
        if account_type["value"] != "REAL":
            return
        lockout_state["date"] = _today_str()
        lockout_state.setdefault("brokers", {}).setdefault(broker_key, {})
        lockout_state["brokers"][broker_key].update({
            "active": True,
            "reason": reason,
        })
        save_lockout_state(lockout_state)

    def handle_lockout_message(reason: str):
        if reason == "goal":
            ui(add_ai_message, t["lockout_goal_message"])
        else:
            msg = t["lockout_stop_message"] + "\n\n" + build_stop_motivational_message()
            ui(add_ai_message, msg)

    def handle_broker_locked(broker_key: str):
        broker_label = {"iq_option": "IQ Option", "bullex": "Bullex", "casatrader": "CasaTrader"}.get(broker_key, broker_key)
        ui(add_ai_message, t["broker_locked_today"].format(broker=broker_label))

    # =========================
    # Top bar
    # =========================

    def go_home(_):
        """Volta para a tela de login"""
        if any_broker_connected():
            ui(add_status_message, t["disconnect_first"])
            return
        logger.info("Botão HOME clicado - voltando para login")

        # Desconecta processos silenciosamente usando cópia do dict
        for k, proc in list(broker_processes.items()):
            if proc and proc.poll() is None:
                try:
                    logger.info(f"Terminando processo {k}")
                    proc.terminate()
                except Exception as e:
                    logger.warning(f"Erro ao terminar {k}: {e}")
            broker_processes[k] = None
            broker_connected[k] = False

        # Limpa pubsub subscribers
        try:
            page.pubsub.unsubscribe_all()
        except Exception as e:
            logger.warning(f"Erro ao limpar pubsub: {e}")

        # Volta para login
        logger.info("Navegando para /login")
        page.go("/login")

    sidebar_visible = {"value": True}
    sidebar_ref = {"container": None}

    def toggle_sidebar(_=None):
        sidebar_visible["value"] = not sidebar_visible["value"]
        if sidebar_ref["container"]:
            sidebar_ref["container"].visible = sidebar_visible["value"]
        toggle_sidebar_btn.icon = ft.Icons.MENU_OPEN if sidebar_visible["value"] else ft.Icons.MENU
        page.update()

    toggle_sidebar_btn = ft.IconButton(
        icon=ft.Icons.MENU_OPEN,
        icon_color=COLORS["text"],
        bgcolor=COLORS["card"],
        tooltip="Mostrar/ocultar painel",
        on_click=toggle_sidebar,
        icon_size=18,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
    )

    toggle_broker_cards_ref = {"fn": lambda e: None}
    store_button_ref = {"button": None}
    logout_button_ref = {"button": None}

    def update_topbar_buttons_state():
        """Atualiza estado dos botões da topbar baseado na conexão"""
        is_connected = any_broker_connected()
        if store_button_ref["button"]:
            store_button_ref["button"].disabled = False
            store_button_ref["button"].icon_color = COLORS["muted"]
        if logout_button_ref["button"]:
            logout_button_ref["button"].disabled = is_connected
            logout_button_ref["button"].icon_color = COLORS["muted"] if not is_connected else "#555555"

    store_button = ft.IconButton(
        icon=ft.Icons.STORE,
        icon_color=COLORS["muted"],
        icon_size=18,
        tooltip="Corretoras",
        on_click=lambda e: toggle_broker_cards_ref["fn"](e),
        disabled=False,
    )
    report_icon_button = ft.IconButton(
        icon=ft.Icons.BAR_CHART_ROUNDED,
        icon_color=COLORS["muted"],
        icon_size=18,
        tooltip="Relatório",
        on_click=lambda e: toggle_report_ref["fn"](e),
        disabled=False,
    )
    toggle_report_ref = {"fn": lambda e: None}
    logout_button = ft.IconButton(
        icon=ft.Icons.LOGOUT_ROUNDED,
        icon_color=COLORS["muted"],
        icon_size=18,
        tooltip=t["home_tooltip"],
        on_click=go_home,
        disabled=False,
    )
    store_button_ref["button"] = store_button
    logout_button_ref["button"] = logout_button

    topbar = ft.Container(
        content=ft.Row(
            [
                ft.Row(
                    [
                        ft.Column(
                            [
                                ft.Text("WS TRADER", size=16, weight=ft.FontWeight.W_700, color=COLORS["text"]),
                                ft.Text(t["chat_subtitle"], size=11, color=COLORS["muted2"]),
                            ],
                            spacing=1,
                        ),
                    ],
                    spacing=10,
                ),
                ft.Row(
                    [
                        report_icon_button,
                        store_button,
                        logout_button,
                    ],
                    spacing=6,
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        ),
        padding=ft.padding.only(left=16, right=16, top=14, bottom=10),
    )

    # =========================
    # Chat list (ListView estilo chat)
    # =========================
    chat_list = ft.ListView(
        expand=True,
        spacing=14,
        padding=ft.padding.only(left=18, right=18, top=16, bottom=16),
        auto_scroll=True,
    )

    def bubble_user(text: str):
        return ft.Row(
            [
                ft.Container(
                    content=ft.Text(text, size=13, color="#0B0F17"),
                    bgcolor=COLORS["brand"],
                    padding=ft.padding.symmetric(horizontal=14, vertical=10),
                    border_radius=ft.border_radius.only(top_left=18, top_right=18, bottom_left=18, bottom_right=6),
                )
            ],
            alignment=ft.MainAxisAlignment.END,
        )

    def bubble_ai(text: str, show_connector: bool = False, connector_color: str = "#10B981"):
        """
        Cria mensagem da IA sem fundo.

        Args:
            text: Texto da mensagem
            show_connector: Se True, mostra linha vertical verde à esquerda
            connector_color: Cor da linha conectora (padrão: verde)
        """
        # Markdown dá um feel "ChatGPT"
        md = ft.Markdown(
            text,
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            code_theme="atom-one-dark",
            on_tap_link=lambda e: None,
        )
        md.code_style = ft.TextStyle(size=10)
        md.code_style_sheet = {
            "p": {"fontSize": "11px"},
            "body": {"fontSize": "11px"}
        }

        # Linha vertical verde (conectora)
        if show_connector:
            return ft.Row(
                [
                    # Linha vertical (mais fina)
                    ft.Container(
                        width=2,
                        height=40,
                        bgcolor=connector_color,
                        border_radius=2,
                    ),
                    # Conteúdo da mensagem (sem fundo)
                    ft.Container(
                        content=md,
                        padding=ft.padding.symmetric(horizontal=14, vertical=12),
                        expand=True,
                    ),
                ],
                spacing=10,
                alignment=ft.MainAxisAlignment.START,
            )
        else:
            # Sem linha conectora - apenas texto sem fundo
            return ft.Row(
                [
                    ft.Container(
                        content=md,
                        padding=ft.padding.symmetric(horizontal=14, vertical=12),
                        expand=True,
                    ),
                ],
                spacing=10,
                alignment=ft.MainAxisAlignment.START,
            )

    # Status box (atualizações durante operação)
    status_log = ft.ListView(
        height=140,
        spacing=6,
        auto_scroll=True,
        padding=ft.padding.symmetric(horizontal=12, vertical=8),
    )

    # typing indicator (pontinhos animados - otimizado)
    typing_text = ft.Text(
        "●",
        size=13,
        color=COLORS["brand"],
        weight=ft.FontWeight.W_600
    )

    typing_row = ft.Container(
        content=typing_text,
        padding=ft.padding.symmetric(horizontal=12, vertical=8),
        visible=False,
    )

    status_icon = ft.Icon(ft.Icons.PAUSE_CIRCLE_OUTLINED, size=16, color=COLORS["muted"])
    status_title_text = ft.Text("Parado", size=12, color=COLORS["text"], weight=ft.FontWeight.W_600)

    async def copy_status_log_async(text: str):
        """Copia texto para o clipboard usando a API do Flet"""
        try:
            await page.set_clipboard_async(text)
            ui(add_status_message, "✅ Mensagens copiadas!")
        except Exception as e:
            logger.error(f"Erro ao copiar (async): {e}")
            # Fallback com tkinter
            try:
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                root.clipboard_clear()
                root.clipboard_append(text)
                root.update()
                root.destroy()
                ui(add_status_message, "✅ Mensagens copiadas!")
            except Exception as e2:
                logger.error(f"Erro ao copiar (fallback): {e2}")
                ui(add_status_message, "❌ Não foi possível copiar.")

    def copy_status_log(e=None):
        lines = []
        if status_account_text.value:
            lines.append(status_account_text.value)
        for ctrl in status_log.controls:
            if isinstance(ctrl, ft.Text):
                if ctrl.value:
                    lines.append(ctrl.value)
            elif hasattr(ctrl, "content") and isinstance(ctrl.content, ft.Text):
                if ctrl.content.value:
                    lines.append(ctrl.content.value)
        text = "\n".join(lines)
        
        if not text.strip():
            ui(add_status_message, "ℹ️ Nenhuma mensagem para copiar.")
            return
            
        # Tentar usar a API assíncrona do Flet
        try:
            page.run_task(copy_status_log_async, text)
        except Exception as ex:
            logger.error(f"Erro ao iniciar cópia: {ex}")
            # Fallback direto com tkinter
            try:
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                root.clipboard_clear()
                root.clipboard_append(text)
                root.update()
                root.destroy()
                ui(add_status_message, "✅ Mensagens copiadas!")
            except Exception as e2:
                logger.error(f"Erro ao copiar: {e2}")
                ui(add_status_message, "❌ Não foi possível copiar.")

    copy_log_button = ft.IconButton(
        icon=ft.Icons.CONTENT_COPY,
        icon_color=COLORS["muted"],
        tooltip="Copiar entradas",
        on_click=copy_status_log,
        icon_size=16,
    )

    status_header = ft.Row(
        [status_icon, status_title_text, ft.Container(expand=True), copy_log_button],
        spacing=6,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
    status_divider = ft.Container(height=1, bgcolor=COLORS["border"], margin=ft.margin.symmetric(vertical=6))
    status_account_text = ft.Text("", size=11, color=COLORS["muted2"])
    status_account_row = ft.Container(
        content=status_account_text,
        padding=ft.padding.only(top=2, bottom=6),
    )

    wins_value_text = ft.Text("0", size=12, weight=ft.FontWeight.W_700, color=COLORS["text"])
    losses_value_text = ft.Text("0", size=12, weight=ft.FontWeight.W_700, color=COLORS["text"])
    win_rate_text = ft.Text("0%", size=12, weight=ft.FontWeight.W_700, color=COLORS["text"])
    wins_losses_box = ft.Container(
        content=ft.Row(
            [
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Container(width=6, height=6, bgcolor="#10B981", border_radius=3),
                            ft.Text(t["wins_label"], size=9, color=COLORS["muted"]),
                            wins_value_text,
                        ],
                        spacing=6,
                        alignment=ft.MainAxisAlignment.CENTER,
                    ),
                    padding=ft.padding.symmetric(horizontal=8, vertical=6),
                    bgcolor=COLORS["panel"],
                    border=ft.border.all(1, COLORS["border"]),
                    border_radius=8,
                ),
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Container(width=6, height=6, bgcolor="#EF4444", border_radius=3),
                            ft.Text(t["losses_label"], size=9, color=COLORS["muted"]),
                            losses_value_text,
                        ],
                        spacing=6,
                        alignment=ft.MainAxisAlignment.CENTER,
                    ),
                    padding=ft.padding.symmetric(horizontal=8, vertical=6),
                    bgcolor=COLORS["panel"],
                    border=ft.border.all(1, COLORS["border"]),
                    border_radius=8,
                ),
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Icon(ft.Icons.TRENDING_UP, size=12, color=COLORS["muted"]),
                            ft.Text("WR", size=9, color=COLORS["muted"]),
                            win_rate_text,
                        ],
                        spacing=6,
                        alignment=ft.MainAxisAlignment.CENTER,
                    ),
                    padding=ft.padding.symmetric(horizontal=8, vertical=6),
                    bgcolor=COLORS["panel"],
                    border=ft.border.all(1, COLORS["border"]),
                    border_radius=8,
                ),
            ],
            spacing=8,
        ),
        padding=ft.padding.only(bottom=8),
        visible=False,
    )

    status_box = ft.Container(
        content=ft.Column(
            [
                status_header,
                status_divider,
                status_account_row,
                wins_losses_box,
                status_log,
            ],
            spacing=0,
        ),
        padding=ft.padding.symmetric(horizontal=12, vertical=10),
        border_radius=12,
        bgcolor=COLORS["panel"],
        border=ft.border.all(1, COLORS["border"]),
        width=520,
        visible=False,
    )

    # Controle de animação otimizado
    typing_animation_running = {"value": False}
    typing_animation_thread = {"thread": None}

    def update_typing_text(value: str, color: str = None):
        typing_text.value = value
        if color:
            typing_text.color = color

    def update_status_header():
        if any_broker_connected():
            status_icon.name = ft.Icons.PLAY_CIRCLE_FILLED
            status_icon.color = COLORS["status_green"]
            status_title_text.value = t["status_operating"]
        else:
            status_icon.name = ft.Icons.PAUSE_CIRCLE_OUTLINED
            status_icon.color = COLORS["status_red"]
            status_title_text.value = "Parado" if selected_lang == "PT" else "Paused"

        update_wins_losses_box()
        _update_ai_phase_ui()

    def get_active_broker():
        """Retorna a corretora atualmente ativa/conectada"""
        for bk in ["iq_option", "bullex", "casatrader"]:
            if broker_connected.get(bk, False):
                return bk
        return None

    def update_status_account_context():
        # Usar valores da corretora ativa
        active_bk = get_active_broker()
        if active_bk:
            ganhos_valor = _get_ganhos(active_bk)
            meta_valor = meta_diaria_broker.get(active_bk, 0.0)
        else:
            ganhos_valor = 0.0
            meta_valor = 0.0
        
        conta_txt = t["pill_account"].format(account=account_type["value"])
        
        # Se meta ainda não foi calculada (0), não mostrar "Falta"
        if meta_valor <= 0:
            if ganhos_valor > 0:
                ganhos_txt = f"+R$ {ganhos_valor:.2f}"
            elif ganhos_valor < 0:
                ganhos_txt = f"R$ {ganhos_valor:.2f}"
            else:
                ganhos_txt = f"R$ 0.00"
        elif ganhos_valor >= meta_valor:
            ganhos_txt = f"✅ Meta batida! +R$ {ganhos_valor:.2f}"
        else:
            falta_meta = max(0, meta_valor - ganhos_valor)
            if ganhos_valor > 0:
                ganhos_txt = f"+R$ {ganhos_valor:.2f} | Falta: R$ {falta_meta:.2f}"
            elif ganhos_valor < 0:
                ganhos_txt = f"R$ {ganhos_valor:.2f} | Falta: R$ {falta_meta:.2f}"
            else:
                ganhos_txt = f"R$ 0.00 | Falta: R$ {falta_meta:.2f}"
        
        status_account_text.value = f"{conta_txt} | {ganhos_txt}"

    def update_wins_losses_box():
        # Usar estatísticas da corretora ativa (separado por DEMO/REAL)
        active_bk = get_active_broker()
        if active_bk:
            bk_acct = broker_accounts.get(active_bk, account_type["value"])
            wins_val = int(_get_broker_wins(active_bk, bk_acct))
            losses_val = int(_get_broker_losses(active_bk, bk_acct))
        else:
            wins_val = 0
            losses_val = 0
        total_val = wins_val + losses_val
        win_rate = (wins_val / total_val * 100.0) if total_val > 0 else 0.0
        wins_value_text.value = str(wins_val)
        losses_value_text.value = str(losses_val)
        win_rate_text.value = f"{win_rate:.0f}%"
        wins_losses_box.visible = any_broker_connected()

    def add_status_message(text: str, color: str = None):
        line_text = ft.Text(text, size=12, color=COLORS["muted"] if color is None else color)
        line_container = ft.Container(
            content=line_text,
            padding=ft.padding.symmetric(horizontal=8, vertical=6),
            border_radius=8,
            bgcolor=ft.Colors.with_opacity(0.08, ft.Colors.WHITE),
        )
        status_log.controls.append(line_container)
        # Limite simples
        if len(status_log.controls) > 300:
            status_log.controls = status_log.controls[-300:]
        update_status_header()
        update_status_account_context()
        # Verifica de forma mais robusta se o status_box já está na lista
        status_box_already_in_list = any(c is status_box for c in chat_list.controls)
        if not status_box_already_in_list:
            # Remove qualquer status_box duplicado antes de inserir
            chat_list.controls = [c for c in chat_list.controls if c is not status_box]
            if len(chat_list.controls) >= 1:
                chat_list.controls.insert(1, status_box)
            else:
                chat_list.controls.append(status_box)
        status_box.visible = True

        def _clear_highlight():
            try:
                line_container.bgcolor = "transparent"
                line_container.update()
            except Exception:
                pass

        threading.Timer(0.5, lambda: ui(_clear_highlight)).start()

    def animate_typing():
        """Anima os pontinhos de carregamento de forma otimizada"""
        dots = ["●", "●●", "●●●"]
        colors = [COLORS["brand"], COLORS["brand2"], "#FFC08A"]
        index = 0
        while typing_animation_running["value"]:
            try:
                ui(update_typing_text, dots[index], colors[index])
                index = (index + 1) % len(dots)
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Erro na animação: {e}")
                break

    def set_typing(on: bool):
        typing_row.visible = on
        if on:
            typing_animation_running["value"] = True
            if typing_animation_thread["thread"] is None or not typing_animation_thread["thread"].is_alive():
                t = threading.Thread(target=animate_typing, daemon=True)
                typing_animation_thread["thread"] = t
                t.start()
        else:
            typing_animation_running["value"] = False
            update_status_header()
        page.update()

    def add_user_message(text: str):
        chat_list.controls.append(bubble_user(text))
        logger.info(f"Mensagem do usuário adicionada: {text[:50]}...")
        # auto_scroll=True no ListView já faz o scroll automaticamente

    def add_ai_message(text: str, show_connector: bool = False, connector_color: str = "#10B981"):
        """
        Adiciona mensagem da IA ao chat.

        Args:
            text: Texto da mensagem
            show_connector: Se True, mostra linha vertical verde à esquerda
            connector_color: Cor da linha conectora
        """
        chat_list.controls.append(bubble_ai(text, show_connector=show_connector, connector_color=connector_color))
        logger.info(f"Mensagem da IA adicionada: {text[:50]}...")
        # auto_scroll=True no ListView já faz o scroll automaticamente

    # =========================
    # Sidebar (brokers + conta + status)
    # =========================
    def chip(text: str, icon: str, tone: str = "off"):
        text_ctrl = ft.Text(text, size=10, color=COLORS["muted"])
        icon_color = COLORS["muted"] if tone == "off" else COLORS["green"]
        icon_ctrl = ft.Icon(icon, size=12, color=icon_color)
        container = ft.Container(
            content=ft.Row([icon_ctrl, text_ctrl], spacing=6, alignment=ft.MainAxisAlignment.CENTER),
            padding=ft.padding.symmetric(horizontal=10, vertical=6),
            border_radius=999,
            bgcolor=GLASS_BG_SOFT,
            border=ft.border.all(1, GLASS_BORDER),
        )
        container.data = {"text": text_ctrl, "icon": icon_ctrl}
        return container

    iq_pill = chip(t["pill_iq"].format(status="OFF"), ft.Icons.RADIO_BUTTON_UNCHECKED, "off")
    bx_pill = chip(t["pill_bullex"].format(status="OFF"), ft.Icons.RADIO_BUTTON_UNCHECKED, "off")
    conta_pill = chip(t["pill_account"].format(account="DEMO"), ft.Icons.ACCOUNT_BALANCE_WALLET, "off")
    ganhos_pill = chip(t["pill_gains"].format(gains=0.00), ft.Icons.TRENDING_UP, "off")

    # Variáveis para os botões (serão criadas depois)
    btn_iq_ref = {"button": None}
    btn_bx_ref = {"button": None}
    btn_ct_ref = {"button": None}
    btn_toggle_account_ref = {"button": None}
    home_button_ref = {"button": None}
    selected_broker = {"value": None}

    def refresh_sidebar():
        iq_on = broker_connected["iq_option"]
        bx_on = broker_connected["bullex"]
        ct_on = broker_connected["casatrader"]
        iq_locked = (account_type["value"] == "REAL" and is_lockout_active("iq_option")) or is_meta_locked_broker("iq_option")
        bx_locked = (account_type["value"] == "REAL" and is_lockout_active("bullex")) or is_meta_locked_broker("bullex")
        ct_locked = (account_type["value"] == "REAL" and is_lockout_active("casatrader")) or is_meta_locked_broker("casatrader")
        selected = selected_broker["value"]
        iq_selected_locked = selected is not None and selected != "iq_option"
        bx_selected_locked = selected is not None and selected != "bullex"
        ct_selected_locked = selected is not None and selected != "casatrader"

        iq_pill.data["text"].value = t["pill_iq"].format(status="ON" if iq_on else "OFF")
        iq_pill.data["text"].color = COLORS["green"] if iq_on else COLORS["muted"]
        iq_pill.data["icon"].color = COLORS["green"] if iq_on else COLORS["muted"]
        iq_pill.bgcolor = GLASS_BG if iq_on else GLASS_BG_SOFT

        bx_pill.data["text"].value = t["pill_bullex"].format(status="ON" if bx_on else "OFF")
        bx_pill.data["text"].color = COLORS["green"] if bx_on else COLORS["muted"]
        bx_pill.data["icon"].color = COLORS["green"] if bx_on else COLORS["muted"]
        bx_pill.bgcolor = GLASS_BG if bx_on else GLASS_BG_SOFT

        conta_pill.data["text"].value = t["pill_account"].format(account=account_type['value'])
        conta_pill.data["text"].color = COLORS["muted"]
        conta_pill.data["icon"].color = COLORS["muted"]

        # Atualizar ganhos com cor dinâmica + falta para meta - POR CORRETORA
        active_bk = get_active_broker()
        if active_bk:
            bk_acct = broker_accounts.get(active_bk, account_type["value"])
            ganhos_valor = _get_ganhos(active_bk, bk_acct)
            meta_valor = meta_diaria_broker.get(active_bk, 0.0)
        else:
            # Sem broker ativo: mostrar dados da corretora com ganhos salvos (para não zerar visual)
            ganhos_valor = 0.0
            meta_valor = 0.0
            acct = account_type["value"]
            for bk in ["iq_option", "bullex", "casatrader"]:
                bk_ganhos = ganhos_broker.get(bk, {}).get(acct, 0.0) if isinstance(ganhos_broker.get(bk), dict) else 0.0
                if bk_ganhos != 0.0:
                    ganhos_valor += bk_ganhos
            # Verificar se alguma meta está batida para conta atual
            any_meta_batida = any(_get_meta_batida(bk) for bk in ["iq_option", "bullex", "casatrader"])
            if any_meta_batida and ganhos_valor > 0:
                ganhos_txt = f"✅ Meta batida! +R$ {ganhos_valor:.2f}"
                ganhos_pill.data["text"].color = COLORS["green"]
                ganhos_pill.data["icon"].color = COLORS["green"]
                ganhos_pill.data["text"].value = ganhos_txt
                # Pular lógica abaixo
                meta_valor = -1  # Flag para pular
        
        # Se meta ainda não foi calculada (0), não mostrar "Falta"
        if meta_valor == -1:
            pass  # Já tratado acima (meta batida sem broker ativo)
        elif meta_valor <= 0:
            if ganhos_valor > 0:
                ganhos_txt = f"+R$ {ganhos_valor:.2f}"
                ganhos_pill.data["text"].color = COLORS["green"]
                ganhos_pill.data["icon"].color = COLORS["green"]
            elif ganhos_valor < 0:
                ganhos_txt = f"R$ {ganhos_valor:.2f}"
                ganhos_pill.data["text"].color = COLORS["red"]
                ganhos_pill.data["icon"].color = COLORS["red"]
            else:
                ganhos_txt = f"R$ 0.00"
                ganhos_pill.data["text"].color = COLORS["muted"]
                ganhos_pill.data["icon"].color = COLORS["muted"]
        elif ganhos_valor >= meta_valor:
            # Meta batida!
            ganhos_txt = f"✅ Meta batida! +R$ {ganhos_valor:.2f}"
            ganhos_pill.data["text"].color = COLORS["green"]
            ganhos_pill.data["icon"].color = COLORS["green"]
        else:
            falta_meta = max(0, meta_valor - ganhos_valor)
            if ganhos_valor > 0:
                ganhos_txt = f"+R$ {ganhos_valor:.2f} | Falta: R$ {falta_meta:.2f}"
                ganhos_pill.data["text"].color = COLORS["green"]
                ganhos_pill.data["icon"].color = COLORS["green"]
            elif ganhos_valor < 0:
                ganhos_txt = f"R$ {ganhos_valor:.2f} | Falta: R$ {falta_meta:.2f}"
                ganhos_pill.data["text"].color = COLORS["red"]
                ganhos_pill.data["icon"].color = COLORS["red"]
            else:
                ganhos_txt = f"R$ 0.00 | Falta: R$ {falta_meta:.2f}"
                ganhos_pill.data["text"].color = COLORS["muted"]
                ganhos_pill.data["icon"].color = COLORS["muted"]
        
        ganhos_pill.data["text"].value = ganhos_txt

        update_wins_losses_box()

        # Desabilitar TODOS os botões quando QUALQUER corretora estiver conectada
        any_connected = any_broker_connected()
        if btn_iq_ref["button"]:
            btn_iq_ref["button"].disabled = (any_connected or iq_locked or iq_selected_locked) if hasattr(btn_iq_ref["button"], 'disabled') else None
            # Para container, remove/adiciona o on_click
            if isinstance(btn_iq_ref["button"], ft.Container):
                btn_iq_ref["button"].on_click = None if (any_connected or iq_locked or iq_selected_locked) else btn_connect_iq
                btn_iq_ref["button"].opacity = 0.5 if (any_connected or iq_locked or iq_selected_locked) else 1.0
                if isinstance(btn_iq_ref["button"].content, ft.Text):
                    btn_iq_ref["button"].content.value = "Conectado" if iq_on else "Conectar"
                    btn_iq_ref["button"].content.color = "#FFFFFF"
        if btn_bx_ref["button"]:
            btn_bx_ref["button"].disabled = (any_connected or bx_locked or bx_selected_locked) if hasattr(btn_bx_ref["button"], 'disabled') else None
            if isinstance(btn_bx_ref["button"], ft.Container):
                btn_bx_ref["button"].on_click = None if (any_connected or bx_locked or bx_selected_locked) else btn_connect_bx
                btn_bx_ref["button"].opacity = 0.5 if (any_connected or bx_locked or bx_selected_locked) else 1.0
                if isinstance(btn_bx_ref["button"].content, ft.Text):
                    btn_bx_ref["button"].content.value = "Conectado" if bx_on else "Conectar"
                    btn_bx_ref["button"].content.color = "#FFFFFF"
        if btn_ct_ref["button"]:
            btn_ct_ref["button"].disabled = (any_connected or ct_locked or ct_selected_locked) if hasattr(btn_ct_ref["button"], 'disabled') else None
            if isinstance(btn_ct_ref["button"], ft.Container):
                btn_ct_ref["button"].on_click = None if (any_connected or ct_locked or ct_selected_locked) else btn_connect_ct
                btn_ct_ref["button"].opacity = 0.5 if (any_connected or ct_locked or ct_selected_locked) else 1.0
                if isinstance(btn_ct_ref["button"].content, ft.Text):
                    btn_ct_ref["button"].content.value = "Conectado" if ct_on else "Conectar"
                    btn_ct_ref["button"].content.color = "#FFFFFF"
        if btn_toggle_account_ref["button"]:
            btn_toggle_account_ref["button"].disabled = any_connected

        if store_button_ref["button"]:
            store_button_ref["button"].disabled = any_connected
            store_button_ref["button"].opacity = 0.5 if any_connected else 1.0
            store_button_ref["button"].icon_color = COLORS["muted2"] if any_connected else COLORS["muted"]
        if logout_button_ref["button"]:
            logout_button_ref["button"].disabled = any_connected
            logout_button_ref["button"].opacity = 0.5 if any_connected else 1.0
            logout_button_ref["button"].icon_color = COLORS["muted2"] if any_connected else COLORS["muted"]

        # Desabilitar menus durante operação
        broker_menu.disabled = any_connected
        account_menu.disabled = any_connected

        # Atualizar opacidade visual dos menus
        if any_connected:
            broker_menu.content.opacity = 0.5
            account_menu.content.opacity = 0.5
        else:
            broker_menu.content.opacity = 1.0
            account_menu.content.opacity = 1.0

        # Desabilitar botão home durante operação
        if home_button_ref["button"]:
            home_button_ref["button"].disabled = any_connected
            home_button_ref["button"].opacity = 0.5 if any_connected else 1.0

        if not any_connected and len(status_log.controls) == 0:
            status_box.visible = False

        # GARANTIA: se nenhum broker conectado/conectando, cards devem estar visíveis
        if not any_connected and not broker_connecting.get("value", False):
            if not broker_cards_panel.visible:
                broker_cards_panel.visible = True
                try:
                    broker_cards_panel.update()
                except Exception:
                    pass

        update_status_account_context()

        # page.update() será chamado por quem chamar refresh_sidebar()

    # =========================
    # Botões / processos brokers
    # =========================
    # Controle para evitar dupla inicialização
    broker_starting = {"iq_option": False, "bullex": False, "casatrader": False}
    # Controle para evitar mensagens duplicadas de "Operando"
    operando_shown = {"iq_option": False, "bullex": False, "casatrader": False}
    # Controle para evitar cálculo duplicado de meta (separado do operando_shown)
    meta_calculated = {"iq_option": False, "bullex": False, "casatrader": False}
    # Contador de reconexões automáticas (máximo 3 tentativas)
    reconnect_count = {"iq_option": 0, "bullex": 0, "casatrader": 0}
    MAX_RECONNECT_ATTEMPTS = 5
    
    def run_broker_process(broker_key: str, script_name: str, env_vars: dict):
        # Determinar tipo de conta REAL para este broker específico (não usar account_type global)
        broker_acct = broker_accounts.get(broker_key, account_type["value"])
        logger.info(f"[{broker_key}] Conta do broker: {broker_acct} (global: {account_type['value']})")
        
        # PROTECAO: Verificar se meta já foi batida hoje PARA ESTA CORRETORA + CONTA
        if _get_meta_batida(broker_key, broker_acct):
            logger.info(f"[{broker_key}] Meta {broker_acct} do dia já foi batida! Bloqueando.")
            ui(show_confetti_animation)
            ui(add_status_message, f"Meta {broker_acct} alcançada nesta corretora!", COLORS["yellow"])
            ui(add_ai_message, f"**Meta {broker_acct} da {broker_key.upper()} já atingida!** 🎉\n\nVocê já bateu sua meta {broker_acct} na {broker_key.upper()} hoje. Tente a outra conta ou corretora.", True, "#10B981")
            return
        
        # PROTECAO: Evitar iniciar processo duplicado
        if broker_starting.get(broker_key, False):
            logger.warning(f"[{broker_key}] Processo ja esta sendo iniciado, ignorando...")
            return
        if broker_connected.get(broker_key, False):
            logger.warning(f"[{broker_key}] Broker ja esta conectado, ignorando...")
            return
        if broker_processes.get(broker_key) is not None:
            logger.warning(f"[{broker_key}] Processo ja existe, ignorando...")
            return
            
        broker_starting[broker_key] = True
        operando_shown[broker_key] = False  # Reset ao iniciar novo processo
        meta_calculated[broker_key] = False  # Reset cálculo de meta
        
        base_dir = os.path.dirname(__file__)
        is_frozen = getattr(sys, "frozen", False)
        script_path = os.path.join(base_dir, script_name)

        import shutil

        venv_python = os.path.join(base_dir, ".venv", "Scripts", "python.exe")
        path_python = shutil.which("python")

        def _is_usable_python(candidate: str) -> bool:
            if not candidate:
                return False
            try:
                if os.path.basename(candidate).lower() == "flet.exe":
                    return False
                result = subprocess.run(
                    [candidate, "-c", "import iqoptionapi"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5,
                )
                return result.returncode == 0
            except Exception:
                return False

        python_exe = None
        if os.path.exists(venv_python) and _is_usable_python(venv_python):
            python_exe = venv_python
        elif path_python and _is_usable_python(path_python):
            python_exe = path_python
        elif sys.executable and os.path.basename(sys.executable).lower() != "flet.exe":
            python_exe = sys.executable
        else:
            python_exe = venv_python if os.path.exists(venv_python) else sys.executable

        if is_frozen:
            cmd = [sys.executable, "--run-bot", broker_key]
            logger.info(f"Tentando iniciar {broker_key} via exe: {cmd}")
        else:
            logger.info(f"Tentando iniciar {broker_key} com script: {script_path}")
            if not os.path.exists(script_path):
                ui(add_status_message, t["file_not_found"].format(file=script_name), COLORS["red"])
                return
            cmd = [python_exe, "-u", script_path]  # -u = unbuffered
            logger.info(f"Python selecionado para bot: {python_exe}")

        env = os.environ.copy()
        # Silencia mensagens do TensorFlow no subprocesso
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"
        env["TF_ENABLE_ONEDNN_OPTS"] = "0"
        env["PYTHONWARNINGS"] = "ignore"  # Silencia warnings do Python/Keras
        env.update(env_vars or {})
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                bufsize=1,
                cwd=base_dir,
                env=env,
            )
            # Verifica falha imediata
            time.sleep(0.5)
            if proc.poll() is not None:
                err_out = (proc.stderr.read() or "").strip() if proc.stderr else ""
                std_out = (proc.stdout.read() or "").strip() if proc.stdout else ""
                broker_connected[broker_key] = False
                broker_processes[broker_key] = None
                broker_connecting["value"] = False
                broker_starting[broker_key] = False  # Reset flag
                # Restaurar cards das corretoras quando falha
                def _restore_on_fail():
                    broker_cards_panel.visible = True
                    try:
                        broker_cards_panel.update()
                    except Exception:
                        pass
                ui(_restore_on_fail)
                ui(refresh_sidebar)
                ui(update_send_button)
                ui(set_typing, False)
                detail = err_out or std_out or "Processo encerrou imediatamente."
                ui(add_status_message, t["broker_start_error"].format(broker=broker_key, error=detail), COLORS["red"])
                return
            broker_processes[broker_key] = proc
            broker_connected[broker_key] = True
            broker_connecting["value"] = False
            broker_starting[broker_key] = False  # Reset flag - processo iniciou com sucesso
            reconnect_count[broker_key] = 0  # Reset contador de reconexões quando conectar com sucesso
            # Restaurar histórico do dia (wins/losses/ganhos) do JSON salvo
            _restore_daily_stats_from_json()
            ui(refresh_sidebar)
            ui(update_status_header)
            ui(update_send_button)
            ui(set_typing, False)
            ui(add_status_message, f"IA {broker_key.replace('_', ' ').title()} iniciada.")
        except Exception as e:
            broker_connected[broker_key] = False
            broker_processes[broker_key] = None
            broker_connecting["value"] = False
            broker_starting[broker_key] = False  # Reset flag
            # Restaurar cards das corretoras quando dá erro
            def _restore_on_error():
                broker_cards_panel.visible = True
                try:
                    broker_cards_panel.update()
                except Exception:
                    pass
            ui(_restore_on_error)
            ui(refresh_sidebar)
            ui(update_send_button)
            ui(set_typing, False)
            ui(add_status_message, t["broker_start_error"].format(broker=broker_key, error=str(e)), COLORS["red"])
            return
        connection_lost = {"value": False}
        last_err_lines = []
        last_out_lines = []

        def _is_connection_lost(text: str) -> bool:
            low = (text or "").lower()
            # Erros de autenticação = desconectar definitivo
            if any(x in low for x in ["senha incorreta", "password incorrect", "invalid password", "authentication failed"]):
                return True
            # Máximo de restarts atingido = bot desistiu, desconectar
            if "máximo de" in low and "restarts atingido" in low:
                return True
            # Erros fatais de DNS (sem servidor)
            if "getaddrinfo failed" in low and "reconectando" not in low:
                return True
            # NÃO tratar timeout/reconexão como desconexão — o bot lida internamente
            # NÃO tratar "forçado o cancelamento" / WinError 10054 como fatal — o bot reconecta
            return False

        def _handle_connection_lost(source_text: str):
            if connection_lost["value"]:
                return
            connection_lost["value"] = True
            
            # Verificar se é erro de senha
            low = (source_text or "").lower()
            if "senha incorreta" in low or "password incorrect" in low or "invalid password" in low or "authentication failed" in low:
                ui(add_status_message, "Desconectado: Senha incorreta. Altere suas credenciais.", COLORS["red"])
                # Não reconectar se senha está errada
                try:
                    if proc and proc.poll() is None:
                        proc.terminate()
                except Exception:
                    pass
                broker_connected[broker_key] = False
                broker_processes[broker_key] = None
                broker_connecting["value"] = False
                broker_starting[broker_key] = False
                set_selected_broker(None)  # Desbloquear para permitir outra corretora
                ui(refresh_sidebar)
                ui(update_status_header)
                ui(update_send_button)
                ui(set_typing, False)
            else:
                # Conexão perdida - RECONECTAR AUTOMATICAMENTE!
                ui(add_status_message, t["bot_connection_lost"], COLORS["text"])
                
                try:
                    if proc and proc.poll() is None:
                        proc.terminate()
                except Exception:
                    pass
                
                broker_connected[broker_key] = False
                broker_processes[broker_key] = None
                broker_connecting["value"] = False
                broker_starting[broker_key] = False
                operando_shown[broker_key] = False  # Reset para mostrar "Operando" novamente
                meta_calculated[broker_key] = False  # Reset cálculo de meta
                ui(refresh_sidebar)
                ui(update_status_header)
                ui(update_send_button)
                ui(set_typing, False)
                
                # RECONEXÃO AUTOMÁTICA após 3 segundos (máximo 3 tentativas)
                def auto_reconnect():
                    time.sleep(3)
                    
                    # Verificar se não excedeu tentativas
                    if reconnect_count.get(broker_key, 0) >= MAX_RECONNECT_ATTEMPTS:
                        ui(add_status_message, f"Máximo de tentativas ({MAX_RECONNECT_ATTEMPTS}) atingido. Reconecte manualmente.", COLORS["red"])
                        reconnect_count[broker_key] = 0  # Reset para próxima vez
                        return
                    
                    reconnect_count[broker_key] = reconnect_count.get(broker_key, 0) + 1
                    attempt = reconnect_count[broker_key]
                    
                    connection_lost["value"] = False  # Reset flag
                    ui(add_status_message, f"Tentando reconectar... ({attempt}/{MAX_RECONNECT_ATTEMPTS})", COLORS["blue"])
                    
                    if broker_key == "iq_option":
                        connect_iq()
                    elif broker_key == "bullex":
                        connect_bullex()
                    elif broker_key == "casatrader":
                        connect_casatrader()
                
                threading.Thread(target=auto_reconnect, daemon=True).start()

        # Thread para capturar stderr - agora processa logs normais também
        def read_stderr():
            # Função para limpar códigos ANSI, emojis e Unicode escapado
            def clean_text(text):
                import re
                # Remove códigos de escape ANSI (cores)
                text = re.sub(r'\x1b\[[0-9;]*m', '', text)
                text = re.sub(r'\[\d+m', '', text)
                # Remove códigos Unicode escapados (\U0001f9e0, \u2514, etc)
                text = re.sub(r'\\U[0-9a-fA-F]{8}', '', text)
                text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
                # Remove emojis e caracteres especiais Unicode reais
                text = re.sub(r'[^\x00-\x7F]+', '', text)
                # Remove espaços extras
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            
            try:
                for err_line in proc.stderr:
                    err_msg = (err_line or "").strip()
                    if err_msg:
                        logger.info(f"[{broker_key} STDERR] {err_msg}")
                        last_err_lines.append(err_msg)
                        if len(last_err_lines) > 10:
                            last_err_lines.pop(0)

                        if _is_connection_lost(err_msg):
                            _handle_connection_lost(err_msg)
                            break

                        # Processar mensagens normais que vêm do stderr
                        clean = err_msg.split("[INFO]")[-1].strip()
                        clean = clean.replace("[WS_AUTO_AI] ", "").strip()
                        # Limpar códigos ANSI e emojis
                        clean = clean_text(clean)
                        
                        # Ignorar linhas técnicas que não interessam ao usuário
                        if not clean or len(clean) < 3:
                            continue
                        if clean.startswith("----") or clean.startswith("===="):
                            continue
                        if "min_samples=" in clean or "loss_weight=" in clean or "adaptive_prior=" in clean:
                            continue
                        if "TOP ativos:" in clean:
                            continue
                        # Ignorar logs técnicos de IA e momentum
                        if "[IA]" in clean and "bayes=" in clean:
                            continue
                        if "[MOMENTUM]" in clean:
                            continue
                        if "[MOMENTUM-SKIP]" in clean or "[TREND-SKIP]" in clean:
                            continue
                        if "[IA-SKIP]" in clean:
                            continue
                        if "[SKIP]" in clean and "nenhum" in clean.lower():
                            continue
                        if "cooldown" in clean.lower():
                            continue
                        if "warmup" in clean.lower():
                            continue
                        if "n_arm=" in clean or "ucb01=" in clean or "prob=" in clean:
                            continue
                        # Filtrar TODAS as mensagens técnicas do novo sistema de confluência
                        if "IA-DECISAO" in clean or "SMART-SKIP" in clean:
                            continue
                        if "CONFLUENCIA" in clean or "BLOQUEADO" in clean:
                            continue
                        if "ALERTA_TRADER" in clean or "PROJECAO_CONTRA" in clean:
                            continue
                        if "TENDENCIA_CONTRA" in clean or "projecao_" in clean.lower():
                            continue
                        if "estrutura_" in clean.lower() or "momentum_" in clean.lower():
                            continue
                        if "padrao_" in clean.lower():
                            continue
                        if "[ERROR]" in clean or "warning" in clean.lower():
                            continue
                        if "Traceback" in clean or "File " in clean:
                            continue
                        if "get_all_init" in clean or "late 30 sec" in clean:
                            continue
                        if "Processo" in clean and "encerrou" in clean:
                            continue
                        if "SCORE=" in clean:
                            continue
                        if "code=" in clean.lower():
                            continue

                        # Mensagens importantes - mostrar no chat com linguagem limpa
                        if "Iniciando" in clean and ("DOM Forex" in clean or "Pernada" in clean or "Perfect" in clean):
                            ui(add_status_message, "IA iniciada - Estrategia DOM Forex Perfect Zones", COLORS["text"])
                        elif "Conectando" in clean:
                            if broker_key == "bullex":
                                ui(add_status_message, t["bot_connecting_bullex"], COLORS["text"])
                            elif "IQ Option" in clean:
                                ui(add_status_message, t["bot_connecting_iq"], COLORS["text"])
                        elif "Websocket connected" in clean:
                            ui(add_status_message, t["bot_websocket_connected"], COLORS["text"])
                        elif "OK - Conectado" in clean or ("Conectado" in clean and "Saldo" in clean):
                            # Formato: Conectado | Saldo: 9923.59 | Conta: PRACTICE
                            saldo_match = re.search(r'Saldo[:\s]+([0-9.]+)', clean)
                            conta_match = re.search(r'Conta[:\s]+(\w+)', clean)
                            if saldo_match:
                                saldo = saldo_match.group(1)
                                conta = conta_match.group(1) if conta_match else "?"
                                tipo = "DEMO" if "PRACTICE" in conta.upper() or "DEMO" in conta.upper() else "REAL"
                                ui(add_status_message, f"Conectado | Saldo: ${saldo} | Conta: {tipo}", COLORS["text"])
                            elif broker_key == "bullex":
                                ui(add_status_message, t["bot_connected_bullex"], COLORS["text"])
                            else:
                                ui(add_status_message, t["bot_connected_iq"], COLORS["text"])
                        elif "ATENCAO - Usando conta REAL" in clean:
                            ui(add_status_message, t["bot_real_warning"], COLORS["red"])
                        elif "INFO - Usando conta DEMO" in clean:
                            ui(add_status_message, t["bot_demo_account"], COLORS["text"])
                        elif "SALDO INICIAL" in clean or ("SALDO:" in clean and "META:" in clean):
                            # Formato: SALDO INICIAL: 9923.59 | META: 10.0% (=992.36)
                            # Ou: 💰 SALDO: 11373.75 | META: 7.0%
                            # Calcular meta (1 vez por conexão)
                            if not meta_calculated.get(broker_key, False):
                                meta_calculated[broker_key] = True
                                operando_shown[broker_key] = True
                                saldo_match = re.search(r'SALDO[^:]*[:\s]+([0-9.]+)', clean)
                                if saldo_match:
                                    saldo = float(saldo_match.group(1))
                                    # Usar a meta configurada na interface (broker_goals)
                                    meta_pct = broker_goals.get(broker_key, 1.5)
                                    # Calcular meta em valor absoluto
                                    meta_valor_abs = saldo * (meta_pct / 100.0)
                                    meta_diaria["value"] = meta_valor_abs
                                    meta_diaria_broker[broker_key] = meta_valor_abs  # Salvar por corretora
                                    logger.info(f"[META] {broker_key}: Saldo: ${saldo:.2f} | Meta: {meta_pct}% = ${meta_valor_abs:.2f}")
                                    ui(add_status_message, f"Saldo: ${saldo:.2f} | Meta: {meta_pct}% (R$ {meta_valor_abs:.2f})", COLORS["text"])
                                    
                                    # Verificar se ganhos restaurados já batem a meta
                                    ganhos_atual = _get_ganhos(broker_key, broker_acct)
                                    if ganhos_atual > 0:
                                        logger.info(f"[META] {broker_key}: Ganhos restaurados: R${ganhos_atual:.2f} | Meta: R${meta_valor_abs:.2f}")
                                        if ganhos_atual >= meta_valor_abs and not _get_meta_batida(broker_key, broker_acct):
                                            _set_meta_batida(broker_key, True, broker_acct)
                                            meta_batida_hoje["value"] = True
                                            save_meta_lockout_broker()
                                            logger.info(f"[META] 🎉 {broker_key} {broker_acct} META JÁ BATIDA (restaurado)! R${ganhos_atual:.2f} >= R${meta_valor_abs:.2f}")
                                            # Parar o bot automaticamente
                                            def stop_after_meta_restore():
                                                import time as t
                                                t.sleep(0.5)
                                                nome_bk = {"iq_option": "IQ Option", "bullex": "Bullex", "casatrader": "CasaTrader"}.get(broker_key, broker_key)
                                                ui(add_status_message, f"Meta {nome_bk} já alcançada! 🎉 R${ganhos_atual:.2f} - IA pausada.", COLORS["text"])
                                                proc = broker_processes.get(broker_key)
                                                if proc and proc.poll() is None:
                                                    try:
                                                        proc.terminate()
                                                        broker_processes[broker_key] = None
                                                        broker_connected[broker_key] = False
                                                        broker_connecting["value"] = False
                                                        ui(refresh_sidebar)
                                                    except Exception as e:
                                                        logger.warning(f"Erro ao parar broker após meta restaurada: {e}")
                                            threading.Thread(target=stop_after_meta_restore, daemon=True).start()
                                        else:
                                            falta = meta_valor_abs - ganhos_atual
                                            ui(add_status_message, f"Continuando... Ganhos: R${ganhos_atual:.2f} | Falta: R${falta:.2f}", COLORS["text"])
                                    ui(refresh_sidebar)
                        elif "IA=ON" in clean:
                            ui(add_status_message, "IA ativa - Aprendizado ligado", COLORS["text"])
                        elif "AI STATUS" in clean:
                            # Formato: [AI STATUS] Total: 200 trades | WR Global: 50.0%
                            trades_match = re.search(r'Total[:\s]+(\d+)', clean)
                            wr_match = re.search(r'WR Global[:\s]+([0-9.]+%)', clean)
                            if trades_match:
                                trades = trades_match.group(1)
                                wr = wr_match.group(1) if wr_match else "?"
                                ui(add_status_message, f"IA: {trades} operacoes | Taxa de acerto: {wr}", COLORS["text"])
                        elif "BOT INICIADO" in clean or "Aguardando sinais" in clean:
                            ui(add_status_message, t["bot_started"], COLORS["text"])
                        elif "GESTO" in clean.upper() or "GESTAO" in clean:
                            # Formato: GESTÃO: 1.0% da banca por operação
                            pct_match = re.search(r'([0-9.]+%)', clean)
                            if pct_match:
                                pct = pct_match.group(1)
                                ui(add_status_message, f"Gestao: {pct} da banca por operacao", COLORS["text"])
                        
                        # DETECÇÃO DE ORDENS EXECUTADAS (APENAS SINAIS APROVADOS)
                        # NÃO mostrar [SINAL-HARD] - esses são apenas detecções, podem ser bloqueados
                        # NÃO mostrar entrada nem resultado para o usuário (apenas processa internamente)
                        
                        if "ORDEM ENVIADA" in clean or ("ORDEM" in clean and ("CALL" in clean or "PUT" in clean) and "stake" in clean.lower()):
                            # Formato: [ATIVO] ORDEM ENVIADA PUT exp=5m (turbo) | stake=99.53
                            # ou: [ATIVO] ✅ ORDEM CALL (turbo) stake=108.38
                            ativo_match = re.search(r'(\S+-OTC)', clean)
                            direcao_match = re.search(r'(?:ENVIADA\s+|ORDEM\s+)(CALL|PUT)', clean)
                            stake_match = re.search(r'stake=([0-9.]+)', clean)
                            if ativo_match and direcao_match:
                                ativo = ativo_match.group(1)
                                direcao = direcao_match.group(1)
                                stake = stake_match.group(1) if stake_match else "?"
                                direcao_pt = "COMPRA" if direcao == "CALL" else "VENDA"
                                # NÃO mostrar entrada na tela - apenas log interno
                                logger.info(f"[ENTRADA] {ativo} {direcao_pt} | Valor: ${stake}")
                                
                                # Salvar dados da operação para análise de LOSS
                                ultima_operacao["ativo"] = ativo
                                ultima_operacao["direcao"] = direcao
                        
                        # Capturar informações do padrão e tendência para análise de LOSS
                        elif "DECISAO: ENTRAR" in clean:
                            # Formato: [ATIVO] DECISAO: ENTRAR CALL | Sinal FORTE
                            direcao_match = re.search(r'ENTRAR\s+(CALL|PUT)', clean)
                            if direcao_match:
                                ultima_operacao["direcao"] = direcao_match.group(1)
                        
                        elif "PADRAO:" in clean and "->" not in clean:
                            # Formato: PADRAO: ENGOLFO_BAIXA (85%)
                            padrao_match = re.search(r'PADRAO:\s*(\S+)\s*\(', clean)
                            if padrao_match:
                                ultima_operacao["padrao"] = padrao_match.group(1)
                        
                        elif "TENDENCIA:" in clean:
                            # Formato: TENDENCIA: CONTRA (BAIXA) ou TENDENCIA: CONFIRMA (ALTA)
                            tendencia_match = re.search(r'TENDENCIA:\s*(\S+)', clean)
                            if tendencia_match:
                                ultima_operacao["tendencia"] = tendencia_match.group(1)
                        
                        elif "SCORE FINAL:" in clean:
                            # Formato: SCORE FINAL: 62%
                            score_match = re.search(r'SCORE FINAL:\s*(\d+%)', clean)
                            if score_match:
                                ultima_operacao["score"] = score_match.group(1)
                        
                        elif "GLOBAL:" in clean:
                            # Formato: GLOBAL: trades=3 wins=2 acc=66.67%
                            # NÃO mostrar na tela
                            pass
                        
                        elif "LUCRO:" in clean:
                            # Formato: SALDO: 10023.82 | LUCRO: +70.67 (0.71%)
                            # NÃO mostrar na tela
                            pass
                        
                        elif "PERDA:" in clean:
                            # NÃO mostrar na tela
                            pass
                        
                        # DETECÇÃO DE WIN/LOSS via STDERR — IGNORAR para evitar duplicação
                        # O resultado é contado APENAS via stdout ">>> RESULTADO:" (fonte única de verdade)
                        elif "WIN" in clean and re.search(r'WIN\s+[+-]?\d+[.,]\d+', clean):
                            # NÃO contar aqui — será contado via stdout ">>> RESULTADO: WIN"
                            logger.debug(f"[STDERR-SKIP] WIN detectado em stderr, ignorando (conta via stdout): {clean}")
                            pass
                        
                        elif "LOSS" in clean and re.search(r'LOSS\s+[+-]?\d+[.,]\d+', clean):
                            # NÃO contar aqui — será contado via stdout ">>> RESULTADO: LOSS"
                            # Apenas capturar dados do ativo para análise posterior
                            ativo_loss_match = re.search(r'\[(\S+-OTC)\]', clean)
                            if ativo_loss_match:
                                ultima_operacao["ativo"] = ativo_loss_match.group(1)
                            logger.debug(f"[STDERR-SKIP] LOSS detectado em stderr, ignorando (conta via stdout): {clean}")
                            pass
                        
                        elif "EMPATE" in clean:
                            ui(add_status_message, "Empate (devolucao)", COLORS["blue"])
                        
                        # Mostrar erros e dicas
                        elif "ERRO" in clean or "DICA" in clean:
                            # Ignorar erros de encoding
                            if "UnicodeEncodeError" not in err_msg and "charmap" not in err_msg:
                                if "DICA" in clean:
                                    ui(add_status_message, f"{clean}")
                                else:
                                    ui(add_status_message, f"{clean}", COLORS["red"])
                        elif any(token in clean.lower() for token in ["traceback", "exception", "modulenotfounderror", "importerror", "error"]):
                            ui(add_status_message, clean, COLORS["red"])
            except Exception as e:
                logger.error(f"Erro ao ler stderr: {e}")
                ui(add_status_message, f"Erro ao ler stderr: {e}", COLORS["red"])

        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()

        def watch_process():
            try:
                code = proc.wait()
                broker_connected[broker_key] = False
                broker_processes[broker_key] = None
                broker_connecting["value"] = False
                broker_starting[broker_key] = False  # Reset flag quando processo encerra
                # Desbloquear seleção de broker para permitir conectar outra corretora
                set_selected_broker(None)
                # Restaurar cards das corretoras quando processo encerra
                def _restore_cards():
                    broker_cards_panel.visible = True
                    try:
                        broker_cards_panel.update()
                    except Exception:
                        pass
                ui(_restore_cards)
                ui(refresh_sidebar)
                ui(update_status_header)
                ui(update_send_button)
                ui(set_typing, False)
                # Se processo morreu inesperadamente (não por meta, não por senha) → reconectar
                if code != 0 and code != 1 and not connection_lost["value"]:
                    logger.warning(f"Processo {broker_key} encerrou inesperadamente (code={code}). Reconectando...")
                    # Auto-reconectar quando processo morre sem ser por desconexão já tratada
                    def auto_reconnect_on_crash():
                        time.sleep(5)
                        if broker_connected.get(broker_key, False):
                            return  # Já reconectou
                        if reconnect_count.get(broker_key, 0) >= MAX_RECONNECT_ATTEMPTS:
                            ui(add_status_message, f"Máximo de tentativas ({MAX_RECONNECT_ATTEMPTS}) atingido.", COLORS["red"])
                            reconnect_count[broker_key] = 0
                            return
                        reconnect_count[broker_key] = reconnect_count.get(broker_key, 0) + 1
                        attempt = reconnect_count[broker_key]
                        connection_lost["value"] = False
                        ui(add_status_message, f"Reconectando automaticamente... ({attempt}/{MAX_RECONNECT_ATTEMPTS})", COLORS["blue"])
                        if broker_key == "iq_option":
                            connect_iq()
                        elif broker_key == "bullex":
                            connect_bullex()
                        elif broker_key == "casatrader":
                            connect_casatrader()
                    threading.Thread(target=auto_reconnect_on_crash, daemon=True).start()
            except Exception:
                pass

        threading.Thread(target=watch_process, daemon=True).start()

        # Ler stdout
        for line in proc.stdout:
            msg = (line or "").strip()
            if not msg:
                continue

            logger.info(f"[{broker_key}] {msg}")
            last_out_lines.append(msg)
            if len(last_out_lines) > 10:
                last_out_lines.pop(0)

            if _is_connection_lost(msg):
                _handle_connection_lost(msg)
                break

            # "limpa" log
            clean = msg.split("[INFO]")[-1].strip()
            clean = clean.replace("[WS_AUTO_AI] ", "").strip()

            # Simplificar mensagens - mostrar apenas essencial (com linha verde conectora)
            if "Conectado" in clean and "Saldo" in clean:
                # Extrair apenas saldo e conta
                parts = clean.split("|")
                saldo_info = [p.strip() for p in parts if "Saldo" in p or "Conta" in p]
                ui(add_status_message, t["bot_connected"] + " | " + " | ".join(saldo_info), COLORS["text"])
            elif "SALDO INICIAL" in clean or ("SALDO:" in clean and "META:" in clean):
                # Calcular meta a partir do stdout também (1 vez por conexão)
                if not meta_calculated.get(broker_key, False):
                    meta_calculated[broker_key] = True
                    operando_shown[broker_key] = True
                    saldo_match = re.search(r'SALDO[^:]*[:\s]+([0-9.]+)', clean)
                    if saldo_match:
                        saldo = float(saldo_match.group(1))
                        meta_pct = broker_goals.get(broker_key, 1.5)
                        meta_valor_abs = saldo * (meta_pct / 100.0)
                        meta_diaria["value"] = meta_valor_abs
                        meta_diaria_broker[broker_key] = meta_valor_abs
                        logger.info(f"[META-STDOUT] {broker_key}: Saldo: ${saldo:.2f} | Meta: {meta_pct}% = ${meta_valor_abs:.2f}")
                        ui(add_status_message, f"Saldo: ${saldo:.2f} | Meta: {meta_pct}% (R$ {meta_valor_abs:.2f})", COLORS["text"])
                        # Verificar se ganhos restaurados já batem a meta
                        ganhos_atual = _get_ganhos(broker_key, broker_acct)
                        if ganhos_atual > 0 and ganhos_atual >= meta_valor_abs and not _get_meta_batida(broker_key, broker_acct):
                            _set_meta_batida(broker_key, True, broker_acct)
                            meta_batida_hoje["value"] = True
                            save_meta_lockout_broker()
                            logger.info(f"[META] {broker_key} META JÁ BATIDA (restaurado)! R${ganhos_atual:.2f}")
                            nome_bk = {"iq_option": "IQ Option", "bullex": "Bullex", "casatrader": "CasaTrader"}.get(broker_key, broker_key)
                            ui(add_status_message, f"Meta {nome_bk} já alcançada! R${ganhos_atual:.2f}", COLORS["text"])
                        elif ganhos_atual > 0:
                            falta = meta_valor_abs - ganhos_atual
                            ui(add_status_message, f"Continuando... Ganhos: R${ganhos_atual:.2f} | Falta: R${falta:.2f}", COLORS["text"])
                        ui(refresh_sidebar)
                    else:
                        ui(add_status_message, t["bot_operating"], COLORS["text"])
            elif "ORDEM ENVIADA" in clean or "ENTRADA" in clean:
                # Mostrar apenas ordens que foram realmente executadas (não sinais bloqueados)
                # Formato: "[ATIVO] ORDEM ENVIADA CALL exp=5m (turbo) | stake=5.00"

                # Extrair ativo
                ativo_match = re.search(r'\[([A-Z]{6}(?:-OTC)?)\]', clean)
                if not ativo_match:
                    ativo_match = re.search(r'([A-Z]{6}(?:-OTC)?)', clean)
                par = ativo_match.group(1) if ativo_match else "PAR"

                # Procurar pela direção (CALL ou PUT)
                direcao_match = re.search(r'(CALL|PUT)', clean)
                direcao = direcao_match.group(1) if direcao_match else "?"

                # Procurar pelo valor/stake (ex: stake=5.00)
                valor_match = re.search(r'stake[=:\s]+(\d+[.,]\d+)', clean)
                if not valor_match:
                    valor_match = re.search(r'(?:R\$|USD|\$)\s*(\d+[.,]\d+)', clean)
                valor = f"R$ {valor_match.group(1)}" if valor_match else "R$ 5.00"

                # Mensagem compacta com linha conectora branca (ordem)
                mensagem = f"{par} {direcao} | {valor}"
                ui(add_status_message, mensagem, COLORS["text"])
            elif "META ATINGIDA" in clean or "META BATIDA" in clean:
                # Meta de lucro atingida - parar bot + popup
                meta_batida_hoje["value"] = True
                _set_meta_batida(broker_key, True, broker_acct)
                save_meta_lockout_broker()  # Salvar por broker
                ui(show_confetti_animation)
                nome_bk = {"iq_option": "IQ Option", "bullex": "Bullex", "casatrader": "CasaTrader"}.get(broker_key, broker_key)
                ui(add_status_message, f"Meta {broker_acct} {nome_bk} alcançada! 🎉 Bot parado.", COLORS["text"])
                # === PARAR O BOT AUTOMATICAMENTE ===
                def _stop_bot_meta_stderr():
                    import time as _t
                    _t.sleep(1)
                    _proc = broker_processes.get(broker_key)
                    if _proc and _proc.poll() is None:
                        try:
                            logger.info(f"[META] Parando bot {broker_key} após meta (stderr)...")
                            _proc.terminate()
                        except Exception as _e:
                            logger.warning(f"[META] Erro ao parar bot: {_e}")
                    broker_processes[broker_key] = None
                    broker_connected[broker_key] = False
                    broker_connecting["value"] = False
                    broker_starting[broker_key] = False
                    set_selected_broker(None)
                    def _restore_meta_stderr():
                        broker_cards_panel.visible = True
                        try:
                            broker_cards_panel.update()
                        except Exception:
                            pass
                    ui(_restore_meta_stderr)
                    ui(refresh_sidebar)
                    ui(update_status_header)
                    ui(update_send_button)
                threading.Thread(target=_stop_bot_meta_stderr, daemon=True).start()
            elif "STOP LOSS" in clean or "STOP-LOSS" in clean:
                # Stop loss atingido - parar bot
                perda_match = re.search(r'Perda:\s*-?(\d+[.,]\d+)', clean)
                percent_match = re.search(r'\((-?\d+[.,]\d+)%\)', clean)
                if perda_match and percent_match:
                    perda = perda_match.group(1).replace(',', '.')
                    percent = percent_match.group(1).replace(',', '.')
                    ui(add_status_message, t["bot_stop_loss"].format(loss=perda, percent=percent), COLORS["red"])
                else:
                    ui(add_status_message, t["bot_stop_loss_simple"], COLORS["red"])
                if broker_acct == "REAL" and not is_lockout_active(broker_key):
                    activate_daily_lockout("stop", broker_key)
                    handle_lockout_message("stop")
            elif ">>> RESULTADO:" in clean:
                # Processar apenas formato específico para evitar duplicação
                if "WIN" in clean:
                    # Formato: >>> RESULTADO: WIN 85.74
                    valor_match = re.search(r'WIN\s+([+-]?\d+[.,]\d+)', clean)
                    if valor_match:
                        valor = float(valor_match.group(1).replace(',', '.'))
                        # Atualizar valores POR CORRETORA E CONTA
                        _add_ganhos(broker_key, valor, broker_acct)
                        ganhos_acumulados["value"] = _get_ganhos(broker_key, broker_acct)  # Compatibilidade
                        # Atualizar estatísticas do dia
                        check_reset_daily_stats()
                        daily_stats_broker[broker_key][broker_acct]["wins"] += 1
                        daily_stats[broker_acct]["wins"] += 1
                        logger.info(f"[WIN] {broker_key} {broker_acct}: R$ {valor:.2f} | Acumulado: R$ {_get_ganhos(broker_key, broker_acct):.2f}")
                        _save_daily_report_data(broker_key, _get_broker_wins(broker_key, broker_acct), _get_broker_losses(broker_key, broker_acct), _get_ganhos(broker_key, broker_acct), broker_acct)
                        save_meta_lockout_broker(broker_key)  # Manter meta_lockout.json sincronizado
                        ui(refresh_sidebar)
                        ui(update_accuracy_chart)  # Atualizar gráfico após WIN
                        ui(_refresh_report_charts)
                        ui(_update_ai_phase_ui)
                        ui(add_status_message, f"WIN +R$ {valor:.2f}", COLORS["text"])
                        # Verificar se bateu a meta
                        meta_bk = meta_diaria_broker.get(broker_key, 0.0)
                        if meta_bk > 0 and _get_ganhos(broker_key, broker_acct) >= meta_bk and not _get_meta_batida(broker_key, broker_acct):
                            _set_meta_batida(broker_key, True, broker_acct)
                            meta_batida_hoje["value"] = True
                            save_meta_lockout_broker()
                            logger.info(f"[META] 🎉 {broker_key} {broker_acct} META BATIDA! Ganhos: R${_get_ganhos(broker_key, broker_acct):.2f} >= Meta: R${meta_bk:.2f}")
                            ui(show_confetti_animation)
                            nome_bk = {"iq_option": "IQ Option", "bullex": "Bullex", "casatrader": "CasaTrader"}.get(broker_key, broker_key)
                            ui(add_status_message, f"Meta {broker_acct} {nome_bk} alcançada! 🎉 Bot parado.", COLORS["text"])
                            # === PARAR O BOT AUTOMATICAMENTE ===
                            def _stop_bot_after_meta():
                                import time as _t
                                _t.sleep(1)  # Espera 1s para logs terminarem
                                _proc = broker_processes.get(broker_key)
                                if _proc and _proc.poll() is None:
                                    try:
                                        logger.info(f"[META] Parando bot {broker_key} após meta batida...")
                                        _proc.terminate()
                                    except Exception as _e:
                                        logger.warning(f"[META] Erro ao parar bot: {_e}")
                                broker_processes[broker_key] = None
                                broker_connected[broker_key] = False
                                broker_connecting["value"] = False
                                broker_starting[broker_key] = False
                                ui(set_selected_broker, None)
                                def _restore_after_meta():
                                    broker_cards_panel.visible = True
                                    try:
                                        broker_cards_panel.update()
                                    except Exception:
                                        pass
                                ui(_restore_after_meta)
                                ui(refresh_sidebar)
                                ui(update_status_header)
                                ui(update_send_button)
                            threading.Thread(target=_stop_bot_after_meta, daemon=True).start()
                elif "LOSS" in clean:
                    # Formato: >>> RESULTADO: LOSS -96.67
                    valor_match = re.search(r'LOSS\s+([+-]?\d+[.,]\d+)', clean)
                    if valor_match:
                        valor = float(valor_match.group(1).replace(',', '.'))
                        # Se o valor não é negativo, torná-lo negativo
                        if valor > 0:
                            valor = -valor
                        # Atualizar valores POR CORRETORA E CONTA
                        _add_ganhos(broker_key, valor, broker_acct)
                        ganhos_acumulados["value"] = _get_ganhos(broker_key, broker_acct)  # Compatibilidade
                        # Atualizar estatísticas do dia
                        check_reset_daily_stats()
                        daily_stats_broker[broker_key][broker_acct]["losses"] += 1
                        daily_stats[broker_acct]["losses"] += 1
                        logger.info(f"[LOSS] {broker_key} {broker_acct}: R$ {valor:.2f} | Acumulado: R$ {_get_ganhos(broker_key, broker_acct):.2f}")
                        _save_daily_report_data(broker_key, _get_broker_wins(broker_key, broker_acct), _get_broker_losses(broker_key, broker_acct), _get_ganhos(broker_key, broker_acct), broker_acct)
                        save_meta_lockout_broker(broker_key)  # Manter meta_lockout.json sincronizado
                        ui(refresh_sidebar)
                        ui(update_accuracy_chart)  # Atualizar gráfico após LOSS
                        ui(_refresh_report_charts)
                        ui(_update_ai_phase_ui)
                        ui(add_status_message, f"LOSS R$ {valor:.2f}", COLORS["text"])
                elif "EMPATE" in clean:
                    # Formato: >>> RESULTADO: EMPATE 0.00
                    ui(add_status_message, "EMPATE (devolucao)", COLORS["text"])
                # NOTA: Linhas "SALDO: | LUCRO:" e "SALDO: | PERDA:" são informativas
                # Não atualizamos ganhos aqui pois já foi feito em WIN/LOSS individual
                # (evita sobrescrever valores acumulados)
                # elif "SALDO:" in clean and "LUCRO:" in clean:
                #     pass  # Apenas informativo
                # elif "SALDO:" in clean and "PERDA:" in clean:
                #     pass  # Apenas informativo

                # Detectar STOP LOSS (com linha vermelha)
                if "STOP LOSS" in clean or "STOP!" in clean:
                    ui(add_status_message, t["bot_stop_loss_msg"], COLORS["red"])

                # Detectar META atingida
                if "META ATINGIDA" in clean or "META!" in clean:
                    meta_batida_hoje["value"] = True
                    _set_meta_batida(broker_key, True, broker_acct)
                    save_meta_lockout_broker()  # Salvar por broker
                    ui(show_confetti_animation)
                    nome_bk = {"iq_option": "IQ Option", "bullex": "Bullex", "casatrader": "CasaTrader"}.get(broker_key, broker_key)
                    ui(add_status_message, f"Meta {broker_acct} {nome_bk} alcançada! 🎉 Bot parado.", COLORS["text"])
                    # === PARAR O BOT AUTOMATICAMENTE ===
                    def _stop_bot_meta_stdout():
                        import time as _t
                        _t.sleep(1)
                        _proc = broker_processes.get(broker_key)
                        if _proc and _proc.poll() is None:
                            try:
                                logger.info(f"[META] Parando bot {broker_key} após meta (stdout)...")
                                _proc.terminate()
                            except Exception as _e:
                                logger.warning(f"[META] Erro ao parar bot: {_e}")
                        broker_processes[broker_key] = None
                        broker_connected[broker_key] = False
                        broker_connecting["value"] = False
                        broker_starting[broker_key] = False
                        set_selected_broker(None)
                        def _restore_meta_stdout():
                            broker_cards_panel.visible = True
                            try:
                                broker_cards_panel.update()
                            except Exception:
                                pass
                        ui(_restore_meta_stdout)
                        ui(refresh_sidebar)
                        ui(update_status_header)
                        ui(update_send_button)
                    threading.Thread(target=_stop_bot_meta_stdout, daemon=True).start()


    def connect_iq():
        set_selected_broker("iq_option")
        conta_txt = account_type["value"]
        # Mensagem removida - o bot já envia "Connecting to IQ Option..." pelo stderr
        # ui(add_ai_message, f"Conectando **IQ Option** ({conta_txt})…", True, "#10B981")

        broker_connecting["value"] = True
        ui(refresh_sidebar)
        ui(add_status_message, t["bot_connecting_iq"], COLORS["text"])

        # Verificar se meta já foi batida hoje NESTA CORRETORA + CONTA (IQ Option)
        if is_meta_locked_broker("iq_option"):
            broker_connecting["value"] = False
            set_selected_broker(None)  # Desbloquear para permitir outra corretora
            ui(refresh_sidebar)
            ui(show_confetti_animation)
            acct = account_type["value"]
            ui(add_status_message, f"Meta {acct} IQ Option já alcançada!", COLORS["yellow"])
            ui(add_ai_message, f"**Meta {acct} IQ Option já atingida!** 🎉\n\nVocê já bateu sua meta {acct} na IQ Option hoje. Tente a outra conta ou corretora.", True, "#10B981")
            return

        # NÃO resetar meta ao conectar DEMO - meta batida vale para o dia todo
        # (Removido o reset que estava aqui)

        if account_type["value"] == "REAL" and is_lockout_active("iq_option"):
            set_selected_broker(None)
            handle_broker_locked("iq_option")
            return

        # Recarregar credenciais do .env para pegar senha atualizada
        import os
        from dotenv import load_dotenv, dotenv_values

        def get_user_env_path():
            env_dir = os.path.join(os.path.expanduser("~"), ".wstrader")
            os.makedirs(env_dir, exist_ok=True)
            return os.path.join(env_dir, ".env")

        env_file = get_user_env_path()
        load_dotenv(dotenv_path=env_file, override=True)
        current_email = os.getenv("IQ_EMAIL", email)
        current_password = os.getenv("IQ_PASS", os.getenv("IQ_PASSWORD", password))

        if not current_email or not current_password:
            ui(add_status_message, "❌ Credenciais IQ Option ausentes. Salve email e senha no login.", COLORS["red"])
            return
        
        ws_env = {k: v for k, v in (dotenv_values(env_file) or {}).items() if k and k.startswith("WS_")}

        env = {
            "BROKER_TYPE": "iq_option",
            "IQ_EMAIL": current_email,
            "IQ_PASS": current_password,
            "IQ_PASSWORD": current_password,
            "IQ_CONTA": "PRACTICE" if broker_accounts["iq_option"] == "DEMO" else "REAL",
            "WS_META_LUCRO": f"{broker_goals['iq_option']:.1f}",
        }
        env.update(ws_env)
        broker_cards_panel.visible = False
        broker_cards_panel.update()
        logger.info(f"Conectando IQ Option com email: {current_email}")
        worker_thread = threading.Thread(target=run_broker_process, args=("iq_option", "WS_AUTO_AI.py", env), daemon=True)
        worker_thread.start()

    def connect_bullex():
        set_selected_broker("bullex")
        conta_txt = account_type["value"]
        # Mensagem removida - o bot já envia "Connecting to Bullex..." pelo stderr
        # ui(add_ai_message, f"Conectando **Bullex** ({conta_txt})…", True, "#10B981")

        broker_connecting["value"] = True
        ui(refresh_sidebar)
        ui(add_status_message, t["bot_connecting_bullex"], COLORS["text"])

        # Verificar se meta já foi batida hoje NESTA CORRETORA + CONTA (Bullex)
        if is_meta_locked_broker("bullex"):
            broker_connecting["value"] = False
            set_selected_broker(None)  # Desbloquear para permitir outra corretora
            ui(refresh_sidebar)
            ui(show_confetti_animation)
            acct = account_type["value"]
            ui(add_status_message, f"Meta {acct} Bullex já alcançada!", COLORS["yellow"])
            ui(add_ai_message, f"**Meta {acct} Bullex já atingida!** 🎉\n\nVocê já bateu sua meta {acct} na Bullex hoje. Tente a outra conta ou corretora.", True, "#10B981")
            return

        # NÃO resetar meta ao conectar DEMO - meta batida vale para o dia todo
        # (Removido o reset que estava aqui)

        if account_type["value"] == "REAL" and is_lockout_active("bullex"):
            set_selected_broker(None)
            handle_broker_locked("bullex")
            return

        # Recarregar credenciais do .env para pegar senha atualizada
        import os
        from dotenv import load_dotenv, dotenv_values

        def get_user_env_path():
            env_dir = os.path.join(os.path.expanduser("~"), ".wstrader")
            os.makedirs(env_dir, exist_ok=True)
            return os.path.join(env_dir, ".env")

        env_file = get_user_env_path()
        load_dotenv(dotenv_path=env_file, override=True)
        current_email = os.getenv("IQ_EMAIL", email)
        current_password = os.getenv("BULLUX_PASS", os.getenv("IQ_PASSWORD", password))

        if not current_email or not current_password:
            ui(add_status_message, "❌ Credenciais Bullex ausentes. Salve email e senha no login.", COLORS["red"])
            return
        
        ws_env = {k: v for k, v in (dotenv_values(env_file) or {}).items() if k and k.startswith("WS_")}

        env = {
            "BROKER_TYPE": "bullex",
            "BULLUX_EMAIL": current_email,
            "BULLUX_PASS": current_password,
            "IQ_PASSWORD": current_password,
            "BULLUX_CONTA": "PRACTICE" if broker_accounts["bullex"] == "DEMO" else "REAL",
            "WS_META_LUCRO": f"{broker_goals['bullex']:.1f}",
        }
        env.update(ws_env)
        broker_cards_panel.visible = False
        broker_cards_panel.update()
        logger.info(f"Conectando Bullex com email: {current_email}")
        worker_thread = threading.Thread(target=run_broker_process, args=("bullex", "WS_AUTO_AI.py", env), daemon=True)
        worker_thread.start()

    def connect_casatrader():
        set_selected_broker("casatrader")
        conta_txt = account_type["value"]

        broker_connecting["value"] = True
        ui(refresh_sidebar)
        ui(add_status_message, "Conectando CasaTrader...", COLORS["text"])

        # Verificar se meta já foi batida hoje NESTA CORRETORA + CONTA (CasaTrader)
        if is_meta_locked_broker("casatrader"):
            broker_connecting["value"] = False
            set_selected_broker(None)  # Desbloquear para permitir outra corretora
            ui(refresh_sidebar)
            ui(show_confetti_animation)
            acct = account_type["value"]
            ui(add_status_message, f"Meta {acct} CasaTrader já alcançada!", COLORS["yellow"])
            ui(add_ai_message, f"**Meta {acct} CasaTrader já atingida!** 🎉\n\nVocê já bateu sua meta {acct} na CasaTrader hoje. Tente a outra conta ou corretora.", True, "#10B981")
            return

        if account_type["value"] == "REAL" and is_lockout_active("casatrader"):
            set_selected_broker(None)
            handle_broker_locked("casatrader")
            return

        # Recarregar credenciais do .env
        import os
        from dotenv import load_dotenv, dotenv_values

        def get_user_env_path():
            env_dir = os.path.join(os.path.expanduser("~"), ".wstrader")
            os.makedirs(env_dir, exist_ok=True)
            return os.path.join(env_dir, ".env")

        env_file = get_user_env_path()
        load_dotenv(dotenv_path=env_file, override=True)
        current_email = os.getenv("CASATRADER_EMAIL", email)
        current_password = os.getenv("CASATRADER_PASS", password)

        if not current_email or not current_password:
            ui(add_status_message, "❌ Credenciais CasaTrader ausentes. Salve email e senha no login.", COLORS["red"])
            return
        
        ws_env = {k: v for k, v in (dotenv_values(env_file) or {}).items() if k and k.startswith("WS_")}

        env = {
            "BROKER_TYPE": "casatrader",
            "CASATRADER_EMAIL": current_email,
            "CASATRADER_PASS": current_password,
            "CASATRADER_CONTA": "PRACTICE" if broker_accounts["casatrader"] == "DEMO" else "REAL",
            "WS_META_LUCRO": f"{broker_goals['casatrader']:.1f}",
        }
        env.update(ws_env)
        broker_cards_panel.visible = False
        broker_cards_panel.update()
        logger.info(f"Conectando CasaTrader com email: {current_email}")
        worker_thread = threading.Thread(target=run_broker_process, args=("casatrader", "WS_AUTO_AI.py", env), daemon=True)
        worker_thread.start()

    def disconnect_all():
        stopped_any = False
        for k, proc in broker_processes.items():
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    stopped_any = True
                except Exception:
                    pass
            broker_processes[k] = None
            broker_connected[k] = False

        broker_connecting["value"] = False

        set_selected_broker(None)

        # Restaurar cards das corretoras
        broker_cards_panel.visible = True
        try:
            broker_cards_panel.update()
        except Exception:
            pass

        refresh_sidebar()
        update_send_button()
        page.update()
        ui(add_status_message, "✅ Desconectado." if stopped_any else "✅ Nenhum processo ativo.")

    # =========================
    # Input (estilo ChatGPT)
    # =========================
    def update_send_button():
        if any_broker_connected():
            send_btn.icon = ft.Icons.STOP_ROUNDED
            send_btn.bgcolor = COLORS["red"]
            send_btn.tooltip = t["stop"]
            send_btn.data = {"base": COLORS["red"], "hover": "#FF7B7B"}
        else:
            send_btn.icon = ft.Icons.ARROW_UPWARD_ROUNDED
            send_btn.bgcolor = COLORS["brand"]
            send_btn.tooltip = t["send"]
            send_btn.data = {"base": COLORS["brand"], "hover": COLORS["brand2"]}
        page.update()

    msg_tf = ft.TextField(
        hint_text=t["placeholder"],
        border=ft.InputBorder.NONE,
        text_style=ft.TextStyle(size=13, color=COLORS["text"]),
        hint_style=ft.TextStyle(size=13, color=COLORS["muted2"]),
        expand=True,
        content_padding=ft.padding.symmetric(vertical=6),
        height=44,
        multiline=False,
        on_submit=lambda e: None,  # a gente chama no handler
    )

    send_btn = ft.IconButton(
        icon=ft.Icons.ARROW_UPWARD_ROUNDED,
        icon_color="#FFFFFF",
        bgcolor=COLORS["brand"],
        tooltip=t["send"],
        on_click=lambda e: None,  # setado abaixo
        icon_size=18,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)),
    )
    send_btn.data = {"base": COLORS["brand"], "hover": COLORS["brand2"]}

    def handle_send_or_stop(_=None):
        send_btn.icon_size = 16
        page.update()
        threading.Timer(0.12, lambda: ui(setattr, send_btn, "icon_size", 18)).start()
        if any_broker_connected():
            disconnect_all()
            return
        send_message()

    send_btn.on_click = handle_send_or_stop

    def handle_send_hover(e):
        if e.data == "true":
            send_btn.bgcolor = send_btn.data.get("hover", COLORS["brand2"])
        else:
            send_btn.bgcolor = send_btn.data.get("base", COLORS["brand"])
        page.update()

    send_btn.on_hover = handle_send_hover

    input_box = ft.Container(
        content=ft.Row(
            [
                msg_tf,
                ft.Container(width=8),
                send_btn,
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=ft.padding.symmetric(horizontal=16, vertical=10),
        bgcolor=GLASS_BG_SOFT,
        border=ft.border.all(1, GLASS_BORDER),
        border_radius=20,
        height=60,
    )

    def send_message():
        text = (msg_tf.value or "").strip()
        if not text:
            return

        msg_tf.value = ""
        page.update()

        ui(add_user_message, text)
        lowered = text.lower()

        # comandos
        if "conectar iq" in lowered or "iq option" in lowered:
            if broker_connected["iq_option"]:
                ui(add_status_message, "✅ Já conectado à IQ Option.")
            else:
                connect_iq()
            refresh_sidebar()
            update_send_button()
            page.update()
            return

        if "conectar bullex" in lowered or "bullex" in lowered:
            if broker_connected["bullex"]:
                ui(add_status_message, "✅ Já conectado à Bullex.")
            else:
                connect_bullex()
            refresh_sidebar()
            update_send_button()
            page.update()
            return

        if "sair" in lowered or "logout" in lowered or "voltar login" in lowered:
            disconnect_all()
            ui(add_ai_message, "Saindo... Voltando para tela de login.")
            page.go("/login")
            return

        if "desconectar" in lowered:
            disconnect_all()
            return

        if "mudar para real" in lowered or "conta real" in lowered or "ativar real" in lowered:
            if any_broker_connected():
                ui(add_status_message, t["disconnect_first"])
            else:
                account_type["value"] = "REAL"
                ganhos_acumulados["value"] = _sum_ganhos_current()
                refresh_sidebar()
                ui(update_accuracy_chart)
                ui(_refresh_report_charts)
                page.update()
                update_status_header()
                ui(add_status_message, t["account_changed_real"])
                ui(add_status_message, t["account_real_warning"])
                if is_lockout_active("iq_option") and is_lockout_active("bullex") and is_lockout_active("casatrader"):
                    handle_lockout_message(get_lockout_reason("iq_option") or get_lockout_reason("bullex") or get_lockout_reason("casatrader"))
            return

        if "mudar para demo" in lowered or "conta demo" in lowered or "ativar demo" in lowered:
            if any_broker_connected():
                ui(add_status_message, t["disconnect_first"])
            else:
                account_type["value"] = "DEMO"
                ganhos_acumulados["value"] = _sum_ganhos_current()
                refresh_sidebar()
                ui(update_accuracy_chart)
                ui(_refresh_report_charts)
                page.update()
                update_status_header()
                ui(add_status_message, t["account_changed_demo"])
            return

        if "status" in lowered:
            iq_status = "ON" if broker_connected["iq_option"] else "OFF"
            bx_status = "ON" if broker_connected["bullex"] else "OFF"
            refresh_sidebar()
            page.update()
            ui(add_status_message, t["status_title"])
            ui(add_status_message, t["status_iq"].format(status=iq_status))
            ui(add_status_message, t["status_bullex"].format(status=bx_status))
            ui(add_status_message, t["status_account"].format(account=account_type['value']))
            ui(add_status_message, t["status_gains"].format(gains=ganhos_acumulados['value']))
            return

        if "ajuda" in lowered or "help" in lowered:
            ui(add_ai_message, t["help_commands"])
            return

        if (
            "horario" in lowered
            or "horário" in lowered
            or "melhor horario" in lowered
            or "melhor horário" in lowered
            or "best hours" in lowered
            or "best time" in lowered
        ):
            ui(add_ai_message, build_best_hours_message())
            return

        if (
            "acuracia" in lowered
            or "acurácia" in lowered
            or "accuracy" in lowered
            or "desempenho" in lowered
            or "performance" in lowered
            or "taxa de acerto" in lowered
        ):
            ui(add_ai_message, build_accuracy_summary_message())
            return

        # OpenAI (assíncrono)
        def worker():
            try:
                conversation_history.append({"role": "user", "content": text})
                resp = get_ai_response(text, conversation_history, system_instructions)
                conversation_history.append({"role": "assistant", "content": resp})
                ui(set_typing, False)
                ui(add_ai_message, resp)
            except Exception as e:
                ui(set_typing, False)
                ui(add_ai_message, f"❌ Erro ao processar. Tente novamente ou use `ajuda`.\n\nDetalhe: {str(e)}")

        set_typing(True)
        threading.Thread(target=worker, daemon=True).start()

    msg_tf.on_submit = lambda e: send_message()

    # =========================
    # Sidebar layout
    # =========================
    def make_action_button(label: str, icon_or_image, bgcolor: str, on_click, disabled=False, store_ref=None):
        # Se for string, é caminho de imagem; senão é ícone
        if isinstance(icon_or_image, str):
            icon_widget = ft.Image(src=icon_or_image, width=32, height=32)
        else:
            icon_widget = ft.Icon(icon_or_image, size=18, color=COLORS["text"])

        btn = ft.TextButton(
            content=ft.Row(
                [
                    icon_widget,
                    ft.Text(label, size=12, color=COLORS["text"], weight=ft.FontWeight.W_600),
                ],
                spacing=10,
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            on_click=on_click if not disabled else None,
            disabled=disabled,
            style=ft.ButtonStyle(
                bgcolor=bgcolor,
                shape=ft.RoundedRectangleBorder(radius=14),
                padding=ft.padding.symmetric(horizontal=14, vertical=12),
            ),
        )

        # Salvar referência se fornecida
        if store_ref is not None:
            store_ref["button"] = btn

        return ft.Container(
            content=btn,
            expand=True,
        )

    def make_image_button(image_path: str, on_click, disabled=False, store_ref=None, tooltip="", bgcolor=None):
        """Botão apenas com imagem (sem texto) - formato quadrado"""
        btn = ft.Container(
            content=ft.Image(src=image_path, width=35, height=35),
            width=50,
            height=50,
            bgcolor=bgcolor if bgcolor else COLORS["card"],
            border=ft.border.all(2, COLORS["border"]),
            border_radius=8,
            on_click=on_click if not disabled else None,
            tooltip=tooltip,
            ink=True,
            alignment=ft.alignment.Alignment(0, 0),
        )

        # Salvar referência se fornecida
        if store_ref is not None:
            store_ref["button"] = btn

        return btn

    def btn_connect_iq(e):
        if broker_connected["iq_option"]:
            ui(add_status_message, t["broker_active"].format(broker="IQ Option"))
        elif is_meta_locked_broker("iq_option"):
            acct = account_type["value"]
            ui(add_status_message, f"Meta {acct} IQ Option já alcançada! 🎉", COLORS["yellow"])
            ui(add_ai_message, f"**Meta {acct} IQ Option já atingida!** 🎉\n\nVocê já bateu sua meta {acct} na IQ Option hoje. Tente a outra conta ou corretora.", True, "#10B981")
            ui(show_confetti_animation)
        elif account_type["value"] == "REAL" and is_lockout_active("iq_option"):
            handle_broker_locked("iq_option")
        else:
            connect_iq()

    def btn_connect_bx(e):
        if broker_connected["bullex"]:
            ui(add_status_message, t["broker_active"].format(broker="Bullex"))
        elif is_meta_locked_broker("bullex"):
            acct = account_type["value"]
            ui(add_status_message, f"Meta {acct} Bullex já alcançada! 🎉", COLORS["yellow"])
            ui(add_ai_message, f"**Meta {acct} Bullex já atingida!** 🎉\n\nVocê já bateu sua meta {acct} na Bullex hoje. Tente a outra conta ou corretora.", True, "#10B981")
            ui(show_confetti_animation)
        elif account_type["value"] == "REAL" and is_lockout_active("bullex"):
            handle_broker_locked("bullex")
        else:
            connect_bullex()

    def btn_connect_ct(e):
        if broker_connected["casatrader"]:
            ui(add_status_message, t["broker_active"].format(broker="CasaTrader"))
        elif is_meta_locked_broker("casatrader"):
            acct = account_type["value"]
            ui(add_status_message, f"Meta {acct} CasaTrader já alcançada! 🎉", COLORS["yellow"])
            ui(add_ai_message, f"**Meta {acct} CasaTrader já atingida!** 🎉\n\nVocê já bateu sua meta {acct} na CasaTrader hoje. Tente a outra conta ou corretora.", True, "#10B981")
            ui(show_confetti_animation)
        elif account_type["value"] == "REAL" and is_lockout_active("casatrader"):
            handle_broker_locked("casatrader")
        else:
            connect_casatrader()

    def btn_disconnect(e):
        disconnect_all()

    def btn_toggle_account(e):
        if any_broker_connected():
            ui(add_status_message, t["disconnect_first"])
            return
        account_type["value"] = "REAL" if account_type["value"] == "DEMO" else "DEMO"
        ganhos_acumulados["value"] = 0.0
        refresh_sidebar()
        page.update()
        update_status_header()
        ui(add_status_message, t["account_changed_real"] if account_type["value"] == "REAL" else t["account_changed_demo"])

    # Funções para menu de corretora
    def select_iq_option(e):
        if account_type["value"] == "REAL" and is_lockout_active("iq_option"):
            handle_broker_locked("iq_option")
            return
        if not broker_connected["iq_option"]:
            connect_iq()
        else:
            ui(add_status_message, t["broker_active"].format(broker="IQ Option"))

    def select_bullex(e):
        if account_type["value"] == "REAL" and is_lockout_active("bullex"):
            handle_broker_locked("bullex")
            return
        if not broker_connected["bullex"]:
            connect_bullex()
        else:
            ui(add_status_message, t["broker_active"].format(broker="Bullex"))

    # Funções para menu de conta
    def select_demo(e):
        if any_broker_connected():
            ui(add_status_message, t["disconnect_first"])
            return
        account_type["value"] = "DEMO"
        ganhos_acumulados["value"] = 0.0
        refresh_sidebar()
        page.update()
        update_status_header()
        ui(add_status_message, t["account_changed_demo_verbose"])

    def select_real(e):
        if any_broker_connected():
            ui(add_status_message, t["disconnect_first"])
            return
        account_type["value"] = "REAL"
        ganhos_acumulados["value"] = 0.0
        refresh_sidebar()
        page.update()
        update_status_header()
        ui(add_status_message, t["account_changed_real"])

    def update_env_var(key: str, value: str):
        """Atualiza ou cria variável no .env"""
        try:
            env_path = os.path.join(os.path.dirname(__file__), ".env")
            lines = []
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            updated = False
            for i, line in enumerate(lines):
                if line.strip().startswith(f"{key}="):
                    lines[i] = f"{key}={value}\n"
                    updated = True
                    break
            if not updated:
                lines.append(f"{key}={value}\n")
            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        except Exception as e:
            logger.error(f"Erro ao atualizar .env: {e}")

    # Menu suspenso de Corretora
    broker_menu = ft.PopupMenuButton(
        content=ft.Container(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.BUSINESS, color=COLORS["muted"], size=18),
                    ft.Container(
                        content=ft.Text(
                            t["broker_menu_title"],
                            color=COLORS["text"],
                            size=13,
                            weight=ft.FontWeight.W_500,
                        ),
                        expand=True,
                    ),
                    ft.Icon(ft.Icons.KEYBOARD_ARROW_DOWN, color=COLORS["muted2"], size=18),
                ],
                spacing=10,
            ),
            padding=ft.padding.symmetric(horizontal=14, vertical=10),
        ),
        items=[
            ft.PopupMenuItem(
                content=ft.Container(
                    content=ft.Text("Selecione a Corretora", color=COLORS["muted2"], size=10, weight=ft.FontWeight.W_600),
                    padding=ft.padding.only(left=8, top=4, bottom=4),
                ),
                disabled=True,
            ),
            ft.PopupMenuItem(
                content=ft.Container(
                    content=ft.Row([
                        ft.Image(src=os.path.join(os.path.dirname(__file__), "img", "log_iq.png"), width=22, height=22),
                        ft.Text("IQ Option", color=COLORS["text"], size=13, weight=ft.FontWeight.W_500)
                    ], spacing=10),
                    padding=ft.padding.symmetric(horizontal=8, vertical=6),
                ),
                on_click=select_iq_option
            ),
            ft.PopupMenuItem(
                content=ft.Container(
                    content=ft.Row([
                        ft.Image(src=os.path.join(os.path.dirname(__file__), "img", "log_bullex.png"), width=22, height=22),
                        ft.Text("Bullex", color=COLORS["text"], size=13, weight=ft.FontWeight.W_500)
                    ], spacing=10),
                    padding=ft.padding.symmetric(horizontal=8, vertical=6),
                ),
                on_click=select_bullex
            ),
        ],
        bgcolor=GLASS_BG_SOFT,
    )

    # Menu suspenso de Conta
    account_menu = ft.PopupMenuButton(
        content=ft.Container(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.ACCOUNT_BALANCE_WALLET, color=COLORS["muted"], size=18),
                    ft.Container(
                        content=ft.Text(
                            t["account_menu_title"],
                            color=COLORS["text"],
                            size=13,
                            weight=ft.FontWeight.W_500,
                        ),
                        expand=True,
                    ),
                    ft.Icon(ft.Icons.KEYBOARD_ARROW_DOWN, color=COLORS["muted2"], size=18),
                ],
                spacing=10,
            ),
            padding=ft.padding.symmetric(horizontal=14, vertical=10),
        ),
        items=[
            ft.PopupMenuItem(
                content=ft.Container(
                    content=ft.Text("Selecione o Tipo de Conta", color=COLORS["muted2"], size=10, weight=ft.FontWeight.W_600),
                    padding=ft.padding.only(left=8, top=4, bottom=4),
                ),
                disabled=True,
            ),
            ft.PopupMenuItem(
                content=ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.PSYCHOLOGY, color="#10B981", size=20),
                        ft.Text(t["account_demo_label"], color=COLORS["text"], size=13, weight=ft.FontWeight.W_500)
                    ], spacing=10),
                    padding=ft.padding.symmetric(horizontal=8, vertical=6),
                ),
                on_click=select_demo
            ),
            ft.PopupMenuItem(
                content=ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.ACCOUNT_BALANCE, color="#3B82F6", size=20),
                        ft.Text(t["account_real_label"], color=COLORS["text"], size=13, weight=ft.FontWeight.W_500)
                    ], spacing=10),
                    padding=ft.padding.symmetric(horizontal=8, vertical=6),
                ),
                on_click=select_real
            ),
        ],
        bgcolor=GLASS_BG_SOFT,
    )

    def dropdown_wrap(ctrl):
        return ft.Container(
            content=ctrl,
            padding=ft.padding.all(0),
            bgcolor=ft.Colors.with_opacity(0.08, ft.Colors.WHITE),
            border=ft.border.all(1, ft.Colors.with_opacity(0.12, ft.Colors.WHITE)),
            border_radius=10,
        )

    def dropdown_section(title: str, icon: str, ctrl):
        return ft.Column(
            [
                ft.Row(
                    [
                        ft.Icon(icon, color=COLORS["muted"], size=16),
                        ft.Text(title, size=11, color=COLORS["muted"], weight=ft.FontWeight.W_600),
                    ],
                    spacing=8,
                ),
                dropdown_wrap(ctrl),
            ],
            spacing=6,
        )

    # Meta diária por corretora (0.5% a 4% máximo)
    broker_goals = {
        "iq_option": min(float(os.getenv("WS_META_LUCRO", "1.5")), 4.0),
        "bullex": min(float(os.getenv("WS_META_LUCRO", "1.5")), 4.0),
        "casatrader": min(float(os.getenv("WS_META_LUCRO", "1.5")), 4.0),
    }
    broker_accounts = {
        "iq_option": account_type["value"],
        "bullex": account_type["value"],
        "casatrader": account_type["value"],
    }
    account_toggle_refs = {
        "iq_option": {"demo": None, "real": None},
        "bullex": {"demo": None, "real": None},
        "casatrader": {"demo": None, "real": None},
    }

    def update_broker_card_lock():
        selected = selected_broker["value"]
        for broker_key in ["iq_option", "bullex", "casatrader"]:
            locked = selected is not None and broker_key != selected
            demo_btn = account_toggle_refs[broker_key].get("demo")
            real_btn = account_toggle_refs[broker_key].get("real")
            if demo_btn:
                demo_btn.on_click = None if locked else (lambda e, bk=broker_key: set_account_for_broker(bk, "DEMO"))
                demo_btn.opacity = 0.5 if locked else 1.0
            if real_btn:
                real_btn.on_click = None if locked else (lambda e, bk=broker_key: set_account_for_broker(bk, "REAL"))
                real_btn.opacity = 0.5 if locked else 1.0
        refresh_sidebar()
        page.update()

    def set_selected_broker(broker_key=None):
        selected_broker["value"] = broker_key
        update_broker_card_lock()

    def update_account_toggles(broker_key: str):
        """Atualiza visual dos toggles Demo/Real"""
        current = broker_accounts.get(broker_key, "DEMO")
        for key, value in [("demo", "DEMO"), ("real", "REAL")]:
            btn = account_toggle_refs[broker_key].get(key)
            if not btn:
                continue
            is_active = current == value
            if value == "DEMO":
                btn.bgcolor = "#EF4444" if is_active else ft.Colors.TRANSPARENT
            else:
                btn.bgcolor = "#10B981" if is_active else ft.Colors.TRANSPARENT
            if isinstance(btn.content, ft.Text):
                btn.content.color = "#FFFFFF" if is_active else COLORS["muted"]

    def set_account_for_broker(broker_key: str, value: str):
        if any_broker_connected():
            ui(add_status_message, t["disconnect_first"])
            return
        # NÃO travar outras corretoras ao mudar DEMO/REAL — só ao conectar (play)
        broker_accounts[broker_key] = value
        account_type["value"] = value
        ganhos_acumulados["value"] = 0.0
        refresh_sidebar()
        page.update()
        update_status_header()
        update_account_toggles(broker_key)
        ui(add_status_message, t["account_changed_real"] if value == "REAL" else t["account_changed_demo"])

    def make_account_toggle(broker_key: str):
        """Badge de status Demo/Real"""
        current = broker_accounts.get(broker_key, "DEMO")
        
        demo_btn = ft.Container(
            content=ft.Text("DEMO", size=9, weight=ft.FontWeight.W_700, 
                           color="#FFFFFF" if current == "DEMO" else COLORS["muted"]),
            padding=ft.padding.symmetric(horizontal=8, vertical=4),
            border_radius=4,
            bgcolor="#EF4444" if current == "DEMO" else ft.Colors.TRANSPARENT,
            on_click=lambda e: set_account_for_broker(broker_key, "DEMO"),
        )
        real_btn = ft.Container(
            content=ft.Text("REAL", size=9, weight=ft.FontWeight.W_700, 
                           color="#FFFFFF" if current == "REAL" else COLORS["muted"]),
            padding=ft.padding.symmetric(horizontal=8, vertical=4),
            border_radius=4,
            bgcolor="#10B981" if current == "REAL" else ft.Colors.TRANSPARENT,
            on_click=lambda e: set_account_for_broker(broker_key, "REAL"),
        )
        account_toggle_refs[broker_key]["demo"] = demo_btn
        account_toggle_refs[broker_key]["real"] = real_btn
        
        return ft.Row([demo_btn, real_btn], spacing=6)

    def make_goal_spinner(broker_key: str):
        """Spinner numérico para meta"""
        pct = broker_goals[broker_key]
        
        pct_text = ft.Text(
            f"{pct:.1f}%",
            size=10,
            weight=ft.FontWeight.W_600,
            color=COLORS["text"],
            width=36,
            text_align=ft.TextAlign.CENTER,
        )

        def increase_value(e):
            if broker_goals[broker_key] < 4:  # Máximo 4%
                broker_goals[broker_key] += 0.5
                pct_text.value = f"{broker_goals[broker_key]:.1f}%"
                pct_text.update()
                update_env_var("WS_META_LUCRO", f"{broker_goals[broker_key]:.1f}")

        def decrease_value(e):
            if broker_goals[broker_key] > 0.5:
                broker_goals[broker_key] -= 0.5
                pct_text.value = f"{broker_goals[broker_key]:.1f}%"
                pct_text.update()
                update_env_var("WS_META_LUCRO", f"{broker_goals[broker_key]:.1f}")

        up_btn = ft.Container(
            content=ft.Icon(ft.Icons.EXPAND_LESS, size=12, color=COLORS["muted"]),
            on_click=increase_value,
            padding=2,
        )
        down_btn = ft.Container(
            content=ft.Icon(ft.Icons.EXPAND_MORE, size=12, color=COLORS["muted"]),
            on_click=decrease_value,
            padding=2,
        )

        return ft.Row([
            ft.Text("Meta", size=9, color=COLORS["muted"]),
            ft.Container(
                content=ft.Row([
                    pct_text,
                    ft.Column([up_btn, down_btn], spacing=0),
                ], spacing=0),
                bgcolor=COLORS["panel"],
                border=ft.border.all(1, COLORS["border"]),
                border_radius=4,
                padding=ft.padding.symmetric(horizontal=4, vertical=1),
            ),
        ], spacing=4, alignment=ft.MainAxisAlignment.END)

    def make_broker_button(broker_key: str, name: str, logo_path: str, connect_handler, description: str):
        """Card compacto moderno"""
        
        # Referência para o ícone do botão
        connect_icon = ft.Icon(ft.Icons.PLAY_ARROW_ROUNDED, size=18, color=COLORS["brand"])
        
        # Função de hover - mesmo efeito do botão de sair
        def on_connect_hover(e):
            if e.data == "true":
                connect_icon.color = COLORS["muted"]  # Cor ao passar o mouse
                connect_btn.bgcolor = "#2A2A2A"  # Fundo escuro
            else:
                connect_icon.color = COLORS["brand"]  # Cor original
                connect_btn.bgcolor = None  # Sem fundo
            connect_btn.update()
        
        # Botão conectar - ícone play com hover
        connect_btn = ft.Container(
            content=connect_icon,
            padding=6,
            border_radius=6,
            on_click=connect_handler,
            on_hover=on_connect_hover,
        )
        if broker_key == "iq_option":
            btn_iq_ref["button"] = connect_btn
        elif broker_key == "bullex":
            btn_bx_ref["button"] = connect_btn
        elif broker_key == "casatrader":
            btn_ct_ref["button"] = connect_btn

        return ft.Container(
            content=ft.Column(
                [
                    # Header: Logo + Nome + Botão conectar
                    ft.Row(
                        [
                            ft.Row([
                                ft.Image(src=logo_path, width=18, height=18),
                                ft.Text(name, size=12, weight=ft.FontWeight.W_600, color=COLORS["text"]),
                            ], spacing=6),
                            connect_btn,
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    # Descrição detalhada
                    ft.Text(description, size=9, color=COLORS["muted"]),
                    ft.Container(height=12),
                    # Badge Demo/Real + Spinner Meta inline
                    ft.Row([
                        make_account_toggle(broker_key),
                        make_goal_spinner(broker_key),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ],
                spacing=2,
            ),
            padding=12,
            border_radius=8,
            bgcolor=COLORS["card"],
            border=ft.border.all(1, COLORS["border"]),
            width=210,
            height=130,
        )

    # Cards lado a lado alinhados à esquerda
    broker_cards_row = ft.Container(
        content=ft.Row(
            [
                make_broker_button(
                    "iq_option",
                    "IQ Option",
                    os.path.join(os.path.dirname(__file__), "img", "log_iq.png"),
                    btn_connect_iq,
                    "Corretora global com ativos OTC 24h e execução rápida",
                ),
                make_broker_button(
                    "bullex",
                    "Bullex",
                    os.path.join(os.path.dirname(__file__), "img", "log_bullex.png"),
                    btn_connect_bx,
                    "Plataforma digital estável com baixa latência",
                ),
                make_broker_button(
                    "casatrader",
                    "CasaTrader",
                    os.path.join(os.path.dirname(__file__), "img", "casatrader.png"),
                    btn_connect_ct,
                    "Plataforma brasileira com suporte local e OTC",
                ),
            ],
            spacing=12,
            alignment=ft.MainAxisAlignment.START,
            wrap=True,
        ),
        padding=0,
    )

    broker_cards_panel = ft.Container(
        content=broker_cards_row,
        padding=ft.padding.only(top=6, bottom=6),
        visible=True,
    )

    # ===================== INDICADOR DE FASE DA IA =====================
    _broker_suffix_map = {"iq_option": "m1", "bullex": "bullex", "casatrader": "casatrader"}

    def _read_ai_training_stats(suffix):
        """Lê dados de treinamento da IA do LGBM.
        Retorna (total_trades, total_wr, live_trades, live_wr).
        Total = LIVE + backtest filtrado (ambos contam para o aprendizado).
        Live = apenas trades reais (para transparência).
        """
        try:
            lgbm_file = os.path.join(os.path.dirname(__file__), f"ws_lgbm_data_{suffix}.json")
            if os.path.exists(lgbm_file):
                with open(lgbm_file, "r", encoding="utf-8") as f:
                    samples = json.load(f)
                if isinstance(samples, list):
                    total = len(samples)
                    wins = sum(1 for s in samples if isinstance(s, dict) and s.get("label") == 1)
                    total_wr = (wins / total * 100.0) if total > 0 else 0.0
                    # Stats LIVE separado
                    live = [s for s in samples if isinstance(s, dict) and s.get("source") != "backtest"]
                    live_total = len(live)
                    live_wins = sum(1 for s in live if s.get("label") == 1)
                    live_wr = (live_wins / live_total * 100.0) if live_total > 0 else 0.0
                    return total, total_wr, live_total, live_wr
        except Exception:
            pass
        return 0, 0.0, 0, 0.0

    def _get_ai_stats_chat(broker_key=None):
        """Lê stats da IA para a corretora ativa.
        Retorna (total_trades, total_wr, live_trades, live_wr).
        Combina dados LGBM + session do dia para melhor representação."""
        bk = broker_key or get_active_broker()
        if not bk:
            best_total, best_wr, best_live, best_lwr = 0, 0.0, 0, 0.0
            for suffix in ["m1", "bullex", "casatrader"]:
                total, wr, live, lwr = _read_ai_training_stats(suffix)
                if total > best_total:
                    best_total, best_wr, best_live, best_lwr = total, wr, live, lwr
            # Se LGBM está vazio, usar stats do dia como fallback
            if best_total == 0:
                acct = account_type["value"]
                session_wins = daily_stats.get(acct, {}).get("wins", 0)
                session_losses = daily_stats.get(acct, {}).get("losses", 0)
                session_total = session_wins + session_losses
                if session_total > 0:
                    session_wr = (session_wins / session_total) * 100.0
                    return session_total, session_wr, session_total, session_wr
            return best_total, best_wr, best_live, best_lwr
        suffix = _broker_suffix_map.get(bk, "m1")
        lgbm_total, lgbm_wr, lgbm_live, lgbm_lwr = _read_ai_training_stats(suffix)
        # Combinar com stats da sessão do dia para a corretora ativa
        acct = account_type["value"]
        session_wins = daily_stats_broker.get(bk, {}).get(acct, {}).get("wins", 0)
        session_losses = daily_stats_broker.get(bk, {}).get(acct, {}).get("losses", 0)
        session_total = session_wins + session_losses
        # Se LGBM tem dados, usar LGBM + adicionar session live
        if lgbm_total > 0:
            combined_live = lgbm_live + session_total
            combined_live_wins = int(lgbm_lwr * lgbm_live / 100.0) + session_wins if lgbm_live > 0 else session_wins
            combined_lwr = (combined_live_wins / combined_live * 100.0) if combined_live > 0 else lgbm_lwr
            return lgbm_total + session_total, lgbm_wr, combined_live, combined_lwr
        # Se LGBM vazio, usar apenas session
        if session_total > 0:
            session_wr = (session_wins / session_total) * 100.0
            return session_total, session_wr, session_total, session_wr
        return 0, 0.0, 0, 0.0

    def _get_ai_phase_chat(total_trades, win_rate, broker_key=None):
        """Retorna fase baseado em trades + win rate — cor por corretora"""
        broker_colors = {
            "iq_option":   ["#FFB74D", "#FF9800", "#F57C00", "#E65100"],
            "bullex":      ["#66BB6A", "#43A047", "#2E7D32", "#1B5E20"],
            "casatrader":  ["#64B5F6", "#42A5F5", "#1E88E5", "#1565C0"],
        }
        bk = broker_key or get_active_broker()
        palette = broker_colors.get(bk, ["#F59E0B", "#FF9800", "#5B8DEF", "#10B981"])

        # Expert: 50+ trades LIVE E win rate >= 65%
        if total_trades >= 50 and win_rate >= 65:
            return (t["ai_phase_full"], palette[3], 1.0, t["ai_phase_desc_full"])
        # Avançado: 25+ trades LIVE E win rate >= 58%
        elif total_trades >= 25 and win_rate >= 58:
            progress = min(1.0, (win_rate - 58) / 7.0 * 0.5 + total_trades / 50.0 * 0.5)
            return (t["ai_phase_lgbm"], palette[2], progress, t["ai_phase_desc_lgbm"])
        # Intermediário: 10+ trades LIVE E win rate >= 52%
        elif total_trades >= 10 and win_rate >= 52:
            progress = min(1.0, (win_rate - 52) / 6.0 * 0.5 + total_trades / 25.0 * 0.5)
            return (t["ai_phase_bayes"], palette[1], progress, t["ai_phase_desc_bayes"])
        # Iniciante: qualquer outro caso
        else:
            progress = 0.0
            if total_trades > 0:
                progress = min(1.0, total_trades / 30.0 * 0.5 + win_rate / 90.0 * 0.5)
            return (t["ai_phase_warmup"], palette[0], progress, t["ai_phase_desc_warmup"])

    _ai_total_init, _ai_wr_init, _ai_live_init, _ai_lwr_init = _get_ai_stats_chat()
    _ai_phase_name, _ai_phase_color, _ai_progress, _ai_phase_desc = _get_ai_phase_chat(_ai_total_init, _ai_wr_init)

    ai_phase_label = ft.Text(t["ai_learning"], size=11, color="#9CA3AF", weight=ft.FontWeight.W_500)
    ai_phase_text = ft.Text(_ai_phase_name, size=12, weight=ft.FontWeight.BOLD, color=_ai_phase_color)
    ai_trades_count = ft.Text(f"WR: {_ai_wr_init:.0f}% | Treino: {_ai_total_init} | Live: {_ai_live_init} ({_ai_lwr_init:.0f}%)", size=10, color="#6B7280")
    ai_phase_description = ft.Text(_ai_phase_desc, size=10, color="#6B7280", italic=True)
    ai_progress_bar = ft.ProgressBar(
        value=min(1.0, _ai_progress),
        color=_ai_phase_color,
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
                ft.Row(
                    controls=[ai_trades_count, ai_phase_description],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
            ],
            spacing=4,
        ),
        padding=ft.padding.symmetric(horizontal=12, vertical=8),
        bgcolor=COLORS["panel"],
        border_radius=8,
        border=ft.border.all(1, COLORS["border"]),
    )

    def _update_ai_phase_ui():
        """Atualiza o indicador de fase da IA"""
        try:
            total, wr, live, lwr = _get_ai_stats_chat()
            phase_name, phase_color, progress, phase_desc = _get_ai_phase_chat(total, wr)
            ai_phase_text.value = phase_name
            ai_phase_text.color = phase_color
            ai_progress_bar.value = min(1.0, progress)
            ai_progress_bar.color = phase_color
            ai_trades_count.value = f"WR: {wr:.0f}% | Treino: {total} | Live: {live} ({lwr:.0f}%)"
            ai_phase_description.value = phase_desc
        except Exception:
            pass

    def add_broker_cards_below_welcome():
        # Cards agora estão fixos no layout principal (entre chat_list e input_box)
        # Esta função é mantida por compatibilidade mas não precisa mais adicionar ao chat_list
        pass

    def toggle_broker_cards(e=None):
        broker_cards_panel.visible = not broker_cards_panel.visible
        broker_cards_panel.update()

    toggle_broker_cards_ref["fn"] = toggle_broker_cards

    # ===================== PAINEL DE RELATÓRIO (GRÁFICOS) =====================
    DAILY_DATA_DIR = DAILY_REPORT_DIR  # Reutilizar diretório já definido

    # Filtro DEMO/REAL independente para os gráficos do relatório
    report_acct_filter = {"value": account_type.get("value", "DEMO")}

    def _on_report_filter_change(selected):
        """Callback do toggle DEMO/REAL no painel de relatório"""
        report_acct_filter["value"] = selected
        # Atualizar visual dos botões
        _update_report_filter_btns()
        _refresh_report_charts()

    def _update_report_filter_btns():
        sel = report_acct_filter["value"]
        for btn_lbl, btn_ref in [("DEMO", report_btn_demo), ("REAL", report_btn_real)]:
            is_sel = (sel == btn_lbl)
            btn_ref.bgcolor = "#EF4444" if (is_sel and btn_lbl == "DEMO") else "#10B981" if (is_sel and btn_lbl == "REAL") else "#2a2f3e"
            btn_ref.content.color = "#FFFFFF" if is_sel else "#9CA3AF"
            btn_ref.content.weight = ft.FontWeight.W_700 if is_sel else ft.FontWeight.W_400
        try:
            report_filter_row.update()
        except Exception:
            pass

    report_btn_demo = ft.Container(
        content=ft.Text("DEMO", size=11, color="#FFFFFF", weight=ft.FontWeight.W_700,
                        text_align=ft.TextAlign.CENTER),
        bgcolor="#EF4444",
        border_radius=6,
        padding=ft.padding.symmetric(horizontal=14, vertical=5),
        on_click=lambda e: _on_report_filter_change("DEMO"),
        animate=ft.Animation(200, ft.AnimationCurve.EASE_IN_OUT),
    )

    report_btn_real = ft.Container(
        content=ft.Text("REAL", size=11, color="#9CA3AF", weight=ft.FontWeight.W_400,
                        text_align=ft.TextAlign.CENTER),
        bgcolor="#2a2f3e",
        border_radius=6,
        padding=ft.padding.symmetric(horizontal=14, vertical=5),
        on_click=lambda e: _on_report_filter_change("REAL"),
        animate=ft.Animation(200, ft.AnimationCurve.EASE_IN_OUT),
    )

    report_filter_row = ft.Row(
        controls=[
            ft.Text("Filtro:", size=11, color="#9CA3AF"),
            report_btn_demo,
            report_btn_real,
        ],
        spacing=8,
        alignment=ft.MainAxisAlignment.START,
    )

    def _load_monthly_report():
        """Carrega dados do mês inteiro por corretora, só dias com operação"""
        result = {}
        brokers = ["iq_option", "bullex", "casatrader"]
        acct = report_acct_filter["value"]
        active_days = set()
        try:
            if not os.path.exists(DAILY_DATA_DIR):
                return {b: {} for b in brokers}, []
            today = datetime.now().date()
            first_day = today.replace(day=1)
            num_days = (today - first_day).days + 1
            for broker in brokers:
                date_map = {}
                for i in range(num_days):
                    day = first_day + timedelta(days=i)
                    day_str = day.strftime("%Y-%m-%d")
                    fpath = os.path.join(DAILY_DATA_DIR, f"{broker}_{day_str}_{acct}.json")
                    if os.path.exists(fpath):
                        try:
                            with open(fpath, 'r', encoding='utf-8') as f:
                                d = json.load(f)
                                profit = d.get("profit", 0.0)
                                if profit != 0.0:
                                    date_map[day_str] = profit
                                    active_days.add(day_str)
                        except Exception:
                            pass
                result[broker] = date_map
        except Exception:
            result = {b: {} for b in brokers}
        sorted_days = sorted(active_days)
        return result, sorted_days

    def _build_report_bar_chart():
        """Gráfico de barras — Ganho ACUMULADO no mês por corretora"""
        brokers_info = [
            ("IQ Option", "iq_option", "#FF9800"),
            ("Bullex", "bullex", "#10B981"),
            ("CasaTrader", "casatrader", "#5B8DEF"),
        ]
        monthly, _ = _load_monthly_report()
        values = []
        for display_name, bkey, color in brokers_info:
            total = sum(monthly.get(bkey, {}).values())
            values.append(total)

        max_val = max(abs(v) for v in values) if any(v != 0 for v in values) else 1.0
        bar_max_h = 120

        bar_cols = []
        for i, (display_name, bkey, color) in enumerate(brokers_info):
            val = values[i]
            h = max(4, int(abs(val) / max_val * bar_max_h)) if val != 0 else 4
            bar_color = color if val >= 0 else "#EF4444"
            bar_cols.append(
                ft.Column([
                    ft.Text(f"R$ {val:.2f}", size=10, color=bar_color, weight=ft.FontWeight.W_600,
                            text_align=ft.TextAlign.CENTER),
                    ft.Container(width=50, height=h, bgcolor=bar_color,
                                 border_radius=ft.border_radius.only(top_left=4, top_right=4),
                                 animate=ft.Animation(400, ft.AnimationCurve.EASE_IN_OUT)),
                    ft.Text(display_name, size=9, color="#9CA3AF", text_align=ft.TextAlign.CENTER),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=4,
                   alignment=ft.MainAxisAlignment.END)
            )

        # Total geral acumulado
        total_geral = sum(values)
        total_color = "#10B981" if total_geral >= 0 else "#EF4444"

        return ft.Container(
            content=ft.Column([
                ft.Row(bar_cols, spacing=24, alignment=ft.MainAxisAlignment.CENTER,
                       vertical_alignment=ft.CrossAxisAlignment.END),
                ft.Container(height=4),
                ft.Text(f"Total Acumulado: R$ {total_geral:.2f}", size=11,
                        weight=ft.FontWeight.W_700, color=total_color,
                        text_align=ft.TextAlign.CENTER),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2),
            height=210, padding=ft.padding.only(top=10, bottom=6),
        )

    def _build_report_line_chart():
        """Tabela mensal por corretora — só dias com operação + total"""
        monthly, active_days = _load_monthly_report()
        if not active_days:
            return ft.Container(
                content=ft.Text("Nenhuma operação no mês", size=11, color="#9CA3AF",
                                text_align=ft.TextAlign.CENTER),
                padding=10,
            )

        colors_map = {"iq_option": "#FF9800", "bullex": "#10B981", "casatrader": "#5B8DEF"}
        names = {"iq_option": "IQ Option", "bullex": "Bullex", "casatrader": "CasaTrader"}

        # Cabeçalho com dias que tiveram operação
        header_cells = [ft.Text("", size=9, color="#9CA3AF", width=70)]
        for day_str in active_days:
            d = datetime.strptime(day_str, "%Y-%m-%d").date()
            header_cells.append(ft.Text(d.strftime("%d/%m"), size=9, color="#9CA3AF",
                                        text_align=ft.TextAlign.CENTER, width=55))
        # Coluna Total
        header_cells.append(ft.Text("Total", size=9, color="#FFFFFF",
                                    weight=ft.FontWeight.W_700,
                                    text_align=ft.TextAlign.CENTER, width=60))

        rows = [ft.Row(header_cells, spacing=2)]

        grand_total = 0.0
        for bkey in ["iq_option", "bullex", "casatrader"]:
            date_map = monthly.get(bkey, {})
            row_cells = [ft.Text(names[bkey], size=9, color=colors_map[bkey],
                                  weight=ft.FontWeight.W_600, width=70)]
            broker_total = 0.0
            for day_str in active_days:
                val = date_map.get(day_str, 0.0)
                broker_total += val
                cell_color = "#10B981" if val > 0 else "#EF4444" if val < 0 else "#555555"
                row_cells.append(
                    ft.Container(
                        content=ft.Text(f"{val:.0f}" if val != 0 else "-", size=9,
                                        color=cell_color, text_align=ft.TextAlign.CENTER),
                        width=55, height=28,
                        bgcolor="#1a2332" if val != 0 else "#1f2937",
                        border_radius=4,
                        alignment=ft.Alignment(0, 0),
                        border=ft.border.all(1, "#2a2f3e"),
                    )
                )
            grand_total += broker_total
            # Célula total da corretora
            t_color = "#10B981" if broker_total > 0 else "#EF4444" if broker_total < 0 else "#9CA3AF"
            row_cells.append(
                ft.Container(
                    content=ft.Text(f"{broker_total:.0f}", size=9,
                                    color=t_color, weight=ft.FontWeight.W_700,
                                    text_align=ft.TextAlign.CENTER),
                    width=60, height=28,
                    bgcolor="#1e2a3a",
                    border_radius=4,
                    alignment=ft.Alignment(0, 0),
                    border=ft.border.all(1, "#3a4f6e"),
                )
            )
            rows.append(ft.Row(row_cells, spacing=2))

        # Linha de total geral
        gt_color = "#10B981" if grand_total > 0 else "#EF4444" if grand_total < 0 else "#9CA3AF"
        total_cells = [ft.Text("TOTAL", size=9, color="#FFFFFF",
                               weight=ft.FontWeight.W_700, width=70)]
        for day_str in active_days:
            day_sum = sum(monthly.get(bk, {}).get(day_str, 0.0) for bk in ["iq_option", "bullex", "casatrader"])
            dc = "#10B981" if day_sum > 0 else "#EF4444" if day_sum < 0 else "#555555"
            total_cells.append(
                ft.Container(
                    content=ft.Text(f"{day_sum:.0f}" if day_sum != 0 else "-", size=9,
                                    color=dc, weight=ft.FontWeight.W_600,
                                    text_align=ft.TextAlign.CENTER),
                    width=55, height=28,
                    bgcolor="#1e2a3a",
                    border_radius=4,
                    alignment=ft.Alignment(0, 0),
                    border=ft.border.all(1, "#3a4f6e"),
                )
            )
        total_cells.append(
            ft.Container(
                content=ft.Text(f"{grand_total:.0f}", size=10,
                                color=gt_color, weight=ft.FontWeight.W_700,
                                text_align=ft.TextAlign.CENTER),
                width=60, height=28,
                bgcolor="#253040",
                border_radius=4,
                alignment=ft.Alignment(0, 0),
                border=ft.border.all(1, "#4a6080"),
            )
        )
        rows.append(ft.Divider(color="#3f4654", height=1))
        rows.append(ft.Row(total_cells, spacing=2))

        return ft.Container(
            content=ft.Column(rows, spacing=4, scroll=ft.ScrollMode.AUTO),
            padding=ft.padding.only(top=6, bottom=6),
        )

    report_bar_ct = ft.Container(content=_build_report_bar_chart(), padding=6)
    report_line_ct = ft.Container(content=_build_report_line_chart(), padding=6)

    report_legend = ft.Row(
        controls=[
            ft.Row([ft.Icon(ft.Icons.CIRCLE, size=8, color="#FF9800"), ft.Text("IQ Option", size=10, color="#9CA3AF")], spacing=4),
            ft.Row([ft.Icon(ft.Icons.CIRCLE, size=8, color="#10B981"), ft.Text("Bullex", size=10, color="#9CA3AF")], spacing=4),
            ft.Row([ft.Icon(ft.Icons.CIRCLE, size=8, color="#5B8DEF"), ft.Text("CasaTrader", size=10, color="#9CA3AF")], spacing=4),
        ],
        spacing=14, alignment=ft.MainAxisAlignment.CENTER,
    )

    report_panel = ft.Container(
        content=ft.Column([
            report_filter_row,
            ft.Text("Ganho Acumulado no Mês", size=13, weight=ft.FontWeight.W_600, color="#FFFFFF"),
            report_bar_ct,
            ft.Divider(color="#3f4654", height=1),
            ft.Text("Ganhos Diários — Mês Atual", size=13, weight=ft.FontWeight.W_600, color="#FFFFFF"),
            report_legend,
            report_line_ct,
        ], spacing=6),
        padding=ft.padding.only(left=18, right=18, top=6, bottom=6),
        visible=False,
        animate=ft.Animation(300, ft.AnimationCurve.EASE_IN_OUT),
    )

    report_visible_state = {"value": False}

    def toggle_report(e=None):
        report_visible_state["value"] = not report_visible_state["value"]
        if report_visible_state["value"]:
            _update_report_filter_btns()
            report_bar_ct.content = _build_report_bar_chart()
            report_line_ct.content = _build_report_line_chart()
            # Adicionar dentro do chat_list para rolar junto
            if report_panel not in chat_list.controls:
                chat_list.controls.append(report_panel)
            report_panel.visible = True
        else:
            report_panel.visible = False
            if report_panel in chat_list.controls:
                chat_list.controls.remove(report_panel)
        try:
            chat_list.update()
        except Exception:
            page.update()

    toggle_report_ref["fn"] = toggle_report

    def _refresh_report_charts():
        """Atualiza os gráficos do relatório automaticamente se estiver visível"""
        if report_visible_state["value"]:
            try:
                report_bar_ct.content = _build_report_bar_chart()
                report_line_ct.content = _build_report_line_chart()
                report_bar_ct.update()
                report_line_ct.update()
            except Exception:
                pass

    # Container do gráfico (para poder atualizar)
    accuracy_chart_container = ft.Container()

    # Função para atualizar o gráfico
    def update_accuracy_chart():
        """Atualiza o gráfico de acurácia com novos dados"""
        try:
            # Verificar se o container está na página antes de atualizar
            if accuracy_chart_container is None:
                return
            if not hasattr(accuracy_chart_container, 'page') or accuracy_chart_container.page is None:
                return
            
            new_chart = create_accuracy_chart()
            accuracy_chart_container.content = new_chart
            
            if isinstance(new_chart, ft.Container) and isinstance(new_chart.data, dict):
                ring_ctrl = new_chart.data.get("ring")
                text_ctrl = new_chart.data.get("percent")
                acc_val = float(new_chart.data.get("accuracy", 0.0))
                if ring_ctrl and text_ctrl and acc_val > 0:
                    animate_accuracy(acc_val, ring_ctrl, text_ctrl)
            
            # Forçar atualização do container e da página com verificação
            try:
                if accuracy_chart_container.page is not None:
                    accuracy_chart_container.update()
                    page.update()
                    logger.info("[GRÁFICO] ✅ Gráfico atualizado com sucesso")
            except Exception:
                pass  # Ignora se não conseguir atualizar (página fechada)
        except Exception as e:
            # Só loga se não for erro de página não adicionada
            if "must be added to the page" not in str(e):
                logger.error(f"[GRÁFICO] ❌ Erro ao atualizar: {e}")

    def animate_accuracy(target_percent: float, ring_ctrl: ft.ProgressRing, text_ctrl: ft.Text):
        steps = 20
        duration = 0.5
        step_time = duration / steps
        target_value = max(0.0, min(1.0, target_percent / 100.0))

        def _run():
            for i in range(steps + 1):
                val = target_value * (i / steps)
                percent = target_percent * (i / steps)
                ui(setattr, ring_ctrl, "value", val)
                ui(setattr, text_ctrl, "value", f"{percent:.1f}%" if target_percent > 0 else "0%")
                time.sleep(step_time)

        threading.Thread(target=_run, daemon=True).start()

    # Gráfico de Acurácia com dados DO DIA ATUAL
    def create_accuracy_chart():
        """Cria visualização de acurácia da IA baseado nas operações DO DIA ATUAL (separado DEMO/REAL)"""
        # Verificar se mudou o dia (reseta estatísticas se necessário)
        check_reset_daily_stats()

        # USAR DADOS DO DIA ATUAL PARA CONTA ATUAL (DEMO ou REAL)
        wins = _get_total_wins()
        losses = _get_total_losses()
        total = wins + losses
        accuracy = 0.0
        acct = account_type["value"]

        logger.info(f"[GRÁFICO] Dados do DIA ({daily_stats['date']}) {acct}: Wins={wins}, Losses={losses}, Total={total}")

        # Se não tiver dados, mostrar mensagem apropriada
        if total == 0:
            accuracy = 0.0
            logger.info("[GRÁFICO] Nenhuma operação hoje - mostrando zeros")
        else:
            accuracy = (wins / total) * 100
            logger.info(f"[GRÁFICO] Acurácia do dia: {accuracy:.1f}%")

        # Calcular porcentagens
        wins_percent = (wins / total * 100) if total > 0 else 0
        losses_percent = (losses / total * 100) if total > 0 else 0

        # Determinar cor baseada em wins vs losses (cores mais suaves/leves)
        if total == 0:
            text_color = COLORS["muted"]
            status_text = t["accuracy_no_data"]
        elif wins > losses:
            # Mais wins = verde claro suave (opacity reduzida)
            text_color = "#6EE7B7"  # Verde mais claro/suave
            status_text = t["accuracy_status"]
        else:
            # Mais losses (ou empate) = vermelho claro suave
            text_color = "#FCA5A5"  # Vermelho mais claro/suave
            status_text = t["accuracy_status"]

        ring = ft.ProgressRing(
            value=0.0 if total > 0 else 0,
            width=120,
            height=120,
            stroke_width=11,
            color="#10B981",
            bgcolor="#2A3140",
        )

        percent_text = ft.Text(
            f"0.0%" if total > 0 else "0%",
            size=32,
            weight=ft.FontWeight.BOLD,
            color=text_color,
        )

        chart_container = ft.Container(
            content=ft.Column(
                [
                    ft.Container(height=8),
                    ft.Text(t["accuracy_title"], size=12, weight=ft.FontWeight.W_600, color=COLORS["text"]),
                    ft.Container(height=8),

                    # Gráfico circular: ProgressRing verde sobre fundo vermelho
                    ft.Container(
                        content=ft.Stack(
                            [
                                # Fundo escuro
                                ft.Container(
                                    width=120,
                                    height=120,
                                    border_radius=60,
                                    bgcolor="#0E1522",
                                ),
                                # ProgressRing com fundo VERMELHO (losses) e progresso VERDE (wins)
                                ring,
                                # Texto central
                                ft.Container(
                                    content=ft.Column(
                                        [
                                            percent_text,
                                            ft.Text(
                                                "Acurácia" if total > 0 else status_text,
                                                size=10,
                                                color=COLORS["muted"],
                                            ),
                                        ],
                                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                        alignment=ft.MainAxisAlignment.CENTER,
                                        spacing=0,
                                    ),
                                    width=120,
                                    height=120,
                                    alignment=ft.alignment.Alignment(0, 0),
                                ),
                            ],
                        ),
                        alignment=ft.alignment.Alignment(0, 0),
                    ),

                    ft.Container(height=10),

                    # Estatísticas detalhadas
                    ft.Row(
                        [
                            ft.Container(
                                content=ft.Column(
                                    [
                                        ft.Row(
                                            [
                                                ft.Container(
                                                    width=7,
                                                    height=7,
                                                    bgcolor="#10B981",
                                                    border_radius=3,
                                                ),
                                                ft.Text(t["wins_label"], size=9, color=COLORS["muted"]),
                                            ],
                                            spacing=4,
                                        ),
                                        ft.Text(str(wins), size=15, weight=ft.FontWeight.BOLD, color=COLORS["text"]),
                                        ft.Text(f"{wins_percent:.1f}%", size=8, color=COLORS["muted2"]),
                                    ],
                                    spacing=2,
                                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                ),
                                expand=True,
                                padding=ft.padding.all(9),
                                bgcolor=GLASS_BG_SOFT,
                                border=ft.border.all(1, GLASS_BORDER),
                                border_radius=11,
                            ),
                            ft.Container(
                                content=ft.Column(
                                    [
                                        ft.Row(
                                            [
                                                ft.Container(
                                                    width=7,
                                                    height=7,
                                                    bgcolor="#EF4444",
                                                    border_radius=3,
                                                ),
                                                ft.Text(t["losses_label"], size=9, color=COLORS["muted"]),
                                            ],
                                            spacing=4,
                                        ),
                                        ft.Text(str(losses), size=15, weight=ft.FontWeight.BOLD, color=COLORS["text"]),
                                        ft.Text(f"{losses_percent:.1f}%", size=8, color=COLORS["muted2"]),
                                    ],
                                    spacing=2,
                                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                ),
                                expand=True,
                                padding=ft.padding.all(9),
                                bgcolor=GLASS_BG_SOFT,
                                border=ft.border.all(1, GLASS_BORDER),
                                border_radius=11,
                            ),
                        ],
                        spacing=9,
                        alignment=ft.MainAxisAlignment.CENTER,
                    ),

                    ft.Container(height=8),
                    ft.Text(
                        t["accuracy_total"].format(total=total) if total > 0 else t["accuracy_empty"],
                        size=8,
                        color=COLORS["muted2"],
                        weight=ft.FontWeight.W_400
                    ),
                    ft.Text(
                        t["accuracy_hint"] if total > 0 else t["accuracy_hint_empty"],
                        size=8,
                        color=COLORS["muted2"],
                        italic=True
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=5,
            ),
            padding=ft.padding.all(14),
            bgcolor=GLASS_BG_SOFT,
            border=ft.border.all(1, GLASS_BORDER),
            border_radius=18,
        )

        chart_container.data = {"ring": ring, "percent": percent_text, "accuracy": accuracy}
        return chart_container

    sidebar_ref["container"] = None

    # =========================
    # Main chat area
    # =========================
    chat_container = ft.Container(
        content=ft.Column(
            [
                topbar,
                ft.Container(
                    content=broker_cards_panel,
                    padding=ft.padding.only(left=18, right=18, top=4),
                ),
                ft.Container(
                    content=ai_phase_card,
                    padding=ft.padding.only(left=18, right=18, top=4),
                ),
                ft.Container(
                    content=ft.Column([chat_list], expand=True, spacing=12),
                    expand=True,
                ),
                ft.Container(
                    content=typing_row,
                    padding=ft.padding.only(left=18, right=18, bottom=6),
                ),
                ft.Container(
                    content=input_box,
                    padding=ft.padding.only(left=18, right=18, bottom=18, top=10),
                ),
            ],
            spacing=0,
            expand=True,
        ),
        expand=True,
    )

    # =========================
    # Root layout (somente chat)
    # =========================
    root = ft.Container(
        content=chat_container,
        expand=True,
    )

    background = ft.Container(
        content=root,
        expand=True,
        gradient=ft.LinearGradient(
            begin=ft.Alignment(-1, -1),
            end=ft.Alignment(1, 1),
            colors=["#15181C", "#1A1D22"]
        )
    )

    page.views.clear()
    page.views.append(ft.View(route="/chat", bgcolor=ft.Colors.TRANSPARENT, padding=0, spacing=0, controls=[background]))
    page.update()

    # Carregar estado de meta batida do dia POR CORRETORA
    load_meta_lockout_broker()
    
    # Verificar metas batidas por corretora e mostrar mensagens específicas
    def show_broker_goals_on_load():
        time.sleep(1.0)  # Aguarda a tela carregar
        # Separar metas batidas por conta (DEMO e REAL)
        metas_demo = []
        metas_real = []
        for bk in ["iq_option", "bullex", "casatrader"]:
            nome = {"iq_option": "IQ Option", "bullex": "Bullex", "casatrader": "CasaTrader"}.get(bk, bk)
            bk_data = meta_batida_broker.get(bk, {})
            if isinstance(bk_data, dict):
                if bk_data.get("DEMO", False):
                    metas_demo.append(nome)
                if bk_data.get("REAL", False):
                    metas_real.append(nome)
        
        msgs = []
        if metas_demo:
            if len(metas_demo) == 3:
                msgs.append("**DEMO: Todas as metas alcançadas!** 🎉")
            else:
                msgs.append(f"**DEMO: Meta {', '.join(metas_demo)} batida!** 🎉")
        if metas_real:
            if len(metas_real) == 3:
                msgs.append("**REAL: Todas as metas alcançadas!** 🎉")
            else:
                msgs.append(f"**REAL: Meta {', '.join(metas_real)} batida!** 🎉")
        
        if msgs:
            msg = "\n".join(msgs)
            # Informar o que ainda pode operar
            acct = account_type["value"]
            not_batida = [n for n in ["IQ Option", "Bullex", "CasaTrader"] 
                          if n not in (metas_demo if acct == "DEMO" else metas_real)]
            if not_batida:
                msg += f"\n\nVocê ainda pode operar {acct}: {', '.join(not_batida)}"
            ui(add_ai_message, msg)
    
    # Só mostrar se alguma meta foi batida (DEMO ou REAL)
    has_any_meta = False
    for bk in ["iq_option", "bullex", "casatrader"]:
        bk_data = meta_batida_broker.get(bk, {})
        if isinstance(bk_data, dict) and (bk_data.get("DEMO", False) or bk_data.get("REAL", False)):
            has_any_meta = True
            break
    if has_any_meta:
        threading.Thread(target=show_broker_goals_on_load, daemon=True).start()

    # ===================== TIMER AUTOMÁTICO - ATUALIZA INDICADOR DA IA =====================
    def _ai_phase_auto_refresh():
        """Atualiza o indicador de fase da IA a cada 30 segundos"""
        import time as _time
        while True:
            _time.sleep(30)
            try:
                ui(_update_ai_phase_ui)
            except Exception:
                pass
    threading.Thread(target=_ai_phase_auto_refresh, daemon=True).start()

    # Welcome
    refresh_sidebar()
    update_send_button()
    update_status_header()

    # Inicializar o gráfico de acurácia (sidebar removido)
    # update_accuracy_chart()

    page.update()
    # Abreviar email para mostrar apenas a parte antes do @
    email_abreviado = email.split('@')[0] if '@' in email else email
    ui(add_ai_message, t["welcome_message"].format(email=email_abreviado))


if __name__ == "__main__":
    ft.app(target=lambda page: chat_screen(page, "teste@email.com", "123"))
