# -*- coding: utf-8 -*-
"""
Assistente de IA para o Chat do WS Trader
Integração com OpenAI GPT para processar comandos e responder perguntas
"""
import logging
import json
import os
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ChatAIAssistant:
    """Assistente de IA para processar mensagens do usuário"""

    def __init__(self):
        """Inicializa o assistente"""
        self.knowledge_base = self._load_knowledge_base()
        self.conversation_history = []

        # Se você tiver a API Key do OpenAI, descomente:
        # import openai
        # self.openai_api_key = os.getenv("OPENAI_API_KEY")
        # if self.openai_api_key:
        #     openai.api_key = self.openai_api_key
        #     self.use_openai = True
        # else:
        #     self.use_openai = False
        #     logger.warning("OpenAI API Key não configurada, usando respostas pré-definidas")

        # Por enquanto, vamos usar lógica baseada em regras
        self.use_openai = False

    def _load_knowledge_base(self) -> Dict:
        """Carrega base de conhecimento do arquivo JSON"""
        try:
            kb_path = os.path.join(os.path.dirname(__file__), 'knowledge_base.json')
            if os.path.exists(kb_path):
                with open(kb_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Knowledge base não encontrada em {kb_path}, usando base padrão")
                return self._get_default_knowledge_base()
        except Exception as e:
            logger.error(f"Erro ao carregar knowledge base: {e}")
            return self._get_default_knowledge_base()

    def _get_default_knowledge_base(self) -> Dict:
        """Retorna base de conhecimento padrão"""
        return {
            "system_info": {
                "name": "WS Trader",
                "description": "Plataforma inteligente de trading com IA",
                "features": [
                    "Operações automatizadas com IA",
                    "Suporte para IQ Option e Bullex",
                    "Contas Demo e Real",
                    "Relatórios detalhados em HTML",
                    "Histórico completo em JSON",
                    "Análise de mercado em tempo real"
                ]
            },
            "brokers": {
                "IQ Option": {
                    "name": "IQ Option",
                    "description": "Corretora de opções binárias e digitais",
                    "account_types": ["DEMO", "REAL"],
                    "min_stake": 1.0
                },
                "Bullex": {
                    "name": "Bullex",
                    "description": "Corretora de opções digitais",
                    "account_types": ["DEMO", "REAL"],
                    "min_stake": 1.0
                }
            },
            "commands": {
                "execute": {
                    "patterns": [
                        "executar",
                        "iniciar",
                        "começar",
                        "rodar",
                        "ligar"
                    ],
                    "description": "Inicia operação do bot"
                },
                "stop": {
                    "patterns": [
                        "parar",
                        "desligar",
                        "cancelar",
                        "interromper"
                    ],
                    "description": "Para operação do bot"
                },
                "results": {
                    "patterns": [
                        "resultados",
                        "estatísticas",
                        "desempenho",
                        "lucro",
                        "win rate"
                    ],
                    "description": "Mostra resultados das operações"
                },
                "help": {
                    "patterns": [
                        "ajuda",
                        "como funciona",
                        "o que é",
                        "explique",
                        "dúvida"
                    ],
                    "description": "Fornece ajuda e explicações"
                },
                "report": {
                    "patterns": [
                        "relatório",
                        "relatorio",
                        "gerar relatório",
                        "exportar"
                    ],
                    "description": "Gera relatórios"
                }
            }
        }

    def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa mensagem do usuário e retorna resposta

        Args:
            message: Mensagem do usuário
            context: Contexto atual (broker, account, bot_running, etc)

        Returns:
            Dict com 'message' e opcionalmente 'suggested_action'
        """
        message_lower = message.lower().strip()

        # Adiciona ao histórico
        self.conversation_history.append({
            'role': 'user',
            'content': message
        })

        # Se OpenAI estiver disponível, usa GPT
        if self.use_openai:
            return self._process_with_openai(message, context)

        # Caso contrário, usa lógica baseada em regras
        return self._process_with_rules(message_lower, context)

    def _process_with_rules(self, message: str, context: Dict) -> Dict[str, Any]:
        """Processa mensagem usando regras predefinidas"""

        # Detecta intenção
        intent = self._detect_intent(message)

        # ===== COMANDO: EXECUTAR BOT =====
        if intent == "execute":
            return self._handle_execute_command(message, context)

        # ===== COMANDO: PARAR BOT =====
        elif intent == "stop":
            return self._handle_stop_command(context)

        # ===== COMANDO: RESULTADOS =====
        elif intent == "results":
            return self._handle_results_command(context)

        # ===== COMANDO: RELATÓRIO =====
        elif intent == "report":
            return self._handle_report_command(context)

        # ===== COMANDO: AJUDA =====
        elif intent == "help":
            return self._handle_help_command(message)

        # ===== MENSAGEM GENÉRICA =====
        else:
            return self._handle_generic_message(message)

    def _detect_intent(self, message: str) -> str:
        """Detecta intenção do usuário baseado em padrões"""
        commands = self.knowledge_base.get('commands', {})

        for intent, data in commands.items():
            patterns = data.get('patterns', [])
            for pattern in patterns:
                if pattern in message:
                    return intent

        return "generic"

    def _handle_execute_command(self, message: str, context: Dict) -> Dict[str, Any]:
        """Trata comando de execução do bot"""

        if context.get('bot_running'):
            return {
                'message': "⚠️ O bot já está em execução. Você precisa pará-lo antes de iniciar uma nova operação.\n\nDeseja parar o bot atual?"
            }

        if not context.get('subscription_active'):
            return {
                'message': "❌ Sua assinatura não está ativa. Por favor, renove sua assinatura para continuar usando o WS Trader."
            }

        # Detecta corretora
        broker = self._extract_broker(message)
        if not broker:
            broker = context.get('broker', 'IQ Option')

        # Detecta tipo de conta
        account = self._extract_account_type(message)
        if not account:
            account = context.get('account', 'DEMO')

        # Aviso para conta REAL
        warning = ""
        if account == "REAL":
            warning = "\n\n⚠️ **ATENÇÃO**: Você está prestes a operar em conta REAL. Dinheiro real será usado!"

        return {
            'message': f"""🚀 **Plano de Execução**

📍 **Corretora:** {broker}
💼 **Tipo de Conta:** {account}{warning}

🤖 **O bot irá:**
1. Conectar à corretora
2. Analisar o mercado em tempo real
3. Executar operações baseadas na IA
4. Gerenciar risco automaticamente

**Você confirma esta operação?**
""",
            'suggested_action': 'execute_bot',
            'broker': broker,
            'account': account
        }

    def _handle_stop_command(self, context: Dict) -> Dict[str, Any]:
        """Trata comando para parar o bot"""

        if not context.get('bot_running'):
            return {
                'message': "ℹ️ O bot não está em execução no momento."
            }

        return {
            'message': "🛑 O bot será parado após a operação atual ser finalizada.\n\nAguarde alguns segundos...",
            'suggested_action': 'stop_bot'
        }

    def _handle_results_command(self, context: Dict) -> Dict[str, Any]:
        """Trata comando para mostrar resultados"""

        # Aqui você buscaria os resultados reais
        # Por enquanto, vamos retornar uma mensagem padrão

        return {
            'message': """📊 **Resumo dos seus Resultados**

✅ **Vitórias:** 0
❌ **Derrotas:** 0
📈 **Win Rate:** 0%
💰 **Lucro/Prejuízo:** R$ 0.00

📝 Você ainda não tem operações registradas.

Deseja ver o relatório completo ou exportar os dados?
""",
            'suggested_action': 'show_results'
        }

    def _handle_report_command(self, context: Dict) -> Dict[str, Any]:
        """Trata comando para gerar relatório"""

        return {
            'message': """📄 **Geração de Relatório**

Posso gerar relatórios em dois formatos:

📊 **HTML Interativo**: Visualização completa com gráficos e estatísticas
💾 **JSON**: Dados brutos para análise ou backup

Qual formato você prefere?
""",
            'suggested_action': 'show_results'
        }

    def _handle_help_command(self, message: str) -> Dict[str, Any]:
        """Trata comandos de ajuda"""

        # Detecta sobre o que o usuário quer ajuda
        if any(word in message for word in ['iq option', 'corretora', 'broker']):
            return {
                'message': """📚 **Sobre as Corretoras**

**IQ Option**
• Corretora de opções binárias e digitais
• Suporte para contas Demo e Real
• Depósito mínimo: $10
• Stake mínimo: $1

**Bullex**
• Corretora de opções digitais
• Suporte para contas Demo e Real
• Depósito mínimo: $10
• Stake mínimo: $1

Para executar em uma corretora específica, diga:
• "executar iq option"
• "executar bullex"
"""
            }

        elif any(word in message for word in ['conta', 'demo', 'real']):
            return {
                'message': """📚 **Tipos de Conta**

**Conta DEMO**
✅ Dinheiro virtual para praticar
✅ Sem risco financeiro
✅ Ideal para testar estratégias
✅ Mesmas condições de mercado

**Conta REAL**
⚠️ Dinheiro real
⚠️ Risco de perda de capital
✅ Lucros reais
✅ Requer gestão de risco

Para escolher o tipo de conta:
• "executar em conta demo"
• "executar em conta real"
"""
            }

        else:
            return {
                'message': f"""❓ **Como Funciona o WS Trader**

O WS Trader é uma plataforma inteligente que usa IA para operar no mercado automaticamente.

**Principais Recursos:**
{chr(10).join('• ' + feature for feature in self.knowledge_base['system_info']['features'])}

**Comandos Principais:**
• "executar [corretora] em conta [tipo]" - Inicia bot
• "parar" - Para o bot
• "mostrar resultados" - Exibe estatísticas
• "gerar relatório" - Cria relatório HTML
• "ajuda" - Mostra esta mensagem

**Precisa de mais ajuda?**
Digite sua dúvida específica ou entre em contato com o suporte!
"""
            }

    def _handle_generic_message(self, message: str) -> Dict[str, Any]:
        """Trata mensagens genéricas"""

        # Tenta identificar se é uma pergunta
        if any(word in message for word in ['?', 'como', 'qual', 'quando', 'onde', 'por que', 'o que']):
            return {
                'message': """🤔 Desculpe, não entendi completamente sua pergunta.

Posso ajudar com:
• Executar operações
• Mostrar resultados
• Gerar relatórios
• Explicar como funciona o sistema

Tente reformular sua pergunta ou escolha uma das opções acima!
"""
            }

        # Mensagem genérica
        return {
            'message': """💬 Olá! Estou aqui para ajudar.

Use os atalhos rápidos abaixo ou digite comandos como:
• "executar iq option"
• "mostrar meus resultados"
• "como funciona"

O que posso fazer por você?
"""
        }

    def _extract_broker(self, message: str) -> str:
        """Extrai nome da corretora da mensagem"""
        message_lower = message.lower()

        if 'iq option' in message_lower or 'iqoption' in message_lower or 'iq' in message_lower:
            return "IQ Option"
        elif 'bullex' in message_lower:
            return "Bullex"

        return None

    def _extract_account_type(self, message: str) -> str:
        """Extrai tipo de conta da mensagem"""
        message_lower = message.lower()

        if 'real' in message_lower:
            return "REAL"
        elif 'demo' in message_lower or 'treino' in message_lower or 'prática' in message_lower or 'pratica' in message_lower:
            return "DEMO"

        return None

    def _process_with_openai(self, message: str, context: Dict) -> Dict[str, Any]:
        """
        Processa mensagem usando OpenAI GPT
        (Implementar quando tiver API Key)
        """
        try:
            import openai

            # Monta prompt com contexto
            system_prompt = f"""Você é um assistente de IA do WS Trader, uma plataforma de trading automatizado.

Informações do sistema:
{json.dumps(self.knowledge_base['system_info'], indent=2, ensure_ascii=False)}

Corretoras disponíveis:
{json.dumps(self.knowledge_base['brokers'], indent=2, ensure_ascii=False)}

Contexto atual do usuário:
- Corretora selecionada: {context.get('broker')}
- Tipo de conta: {context.get('account')}
- Bot em execução: {context.get('bot_running')}
- Assinatura ativa: {context.get('subscription_active')}

Sua missão:
1. Ajudar o usuário a operar de forma segura
2. Sempre pedir confirmação antes de executar operações
3. Alertar sobre riscos em contas REAL
4. Fornecer informações claras e objetivas
5. Ser amigável e prestativo

Ao sugerir execução, retorne JSON no formato:
{{"message": "sua mensagem", "suggested_action": "execute_bot", "broker": "IQ Option", "account": "DEMO"}}
"""

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *self.conversation_history[-10:],  # Últimas 10 mensagens
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=500
            )

            ai_response = response.choices[0].message.content

            # Tenta parsear como JSON
            try:
                return json.loads(ai_response)
            except:
                return {'message': ai_response}

        except Exception as e:
            logger.error(f"Erro ao processar com OpenAI: {e}")
            # Fallback para regras
            return self._process_with_rules(message.lower(), context)
