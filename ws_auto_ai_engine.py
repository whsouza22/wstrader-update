# -*- coding: utf-8 -*-
"""
WS Auto AI Engine - Wrapper simples para integração com UI Flet
Usa o código original WS_AUTO_AI.py com todas as funcionalidades
"""

import time
from typing import Optional, Callable, Dict, Any, Tuple
from datetime import datetime
import os

# Importa do código original
from WS_AUTO_AI import (
    TF_M1, EXP_FIXA, DECIDIR_ANTES_FECHAR_SEC,
    PERCENT_BANCA, META_LUCRO_PERCENT, STOP_LOSS_PERCENT, USE_DYNAMIC_STAKE,
    VALOR_MINIMO, STAKE_FIXA,
    IA_ON, AI_STATS_FILE, AI_MIN_SAMPLES, AI_CONF_MIN, AI_MIN_PROB,
    safe_call, wait_until_minus,
    obter_top_ativos_otc, escolher_melhor_setup,
    enviar_ordem, wait_result,
    verificar_meta_atingida, calcular_stake_dinamico,
    ai_predict, ai_update,
    _safe_load_json, _safe_save_json,
    cooldown
)

# ===================== MULTI-BROKER SUPPORT =====================
# Importa todas as APIs disponíveis dinamicamente
from iqoptionapi.stable_api import IQ_Option

try:
    from bullexapi.stable_api import Bullex
except ImportError:
    Bullex = None

try:
    from casatraderapi.stable_api import Casa_Trader
except ImportError:
    Casa_Trader = None


def _get_broker_api(broker_type: str):
    """Retorna a classe da API correta baseado no tipo de corretora"""
    broker_type = broker_type.lower().strip().replace(" ", "_")
    if broker_type == "bullex":
        if Bullex is None:
            raise ImportError("bullexapi não está instalada")
        return Bullex, "Bullex"
    elif broker_type in ("casatrader", "casa_trader"):
        if Casa_Trader is None:
            raise ImportError("casatraderapi não está instalada")
        return Casa_Trader, "CasaTrader"
    else:
        return IQ_Option, "IQ Option"

# Importa analisador de loss
try:
    from loss_analyzer import get_loss_analyzer
    LOSS_ANALYZER_ENABLED = True
except ImportError:
    LOSS_ANALYZER_ENABLED = False


class TradingConfig:
    """Configuração mínima"""
    def __init__(self):
        self.EMAIL = ""
        self.SENHA = ""
        self.CONTA = "PRACTICE"
        self.BROKER_TYPE = "iq_option"  # "iq_option", "bullex", "casatrader"


class TradingEngine:
    """Engine de trading integrado com UI Flet"""

    def __init__(self, config: TradingConfig,
                 logger_callback: Optional[Callable] = None,
                 stats_callback: Optional[Callable] = None,
                 operation_callback: Optional[Callable] = None,
                 result_callback: Optional[Callable] = None):

        self.config = config
        self.logger_callback = logger_callback
        self.stats_callback = stats_callback
        self.operation_callback = operation_callback
        self.result_callback = result_callback

        # Determina a API correta baseado no broker selecionado
        self.BrokerAPIClass, self.broker_name = _get_broker_api(config.BROKER_TYPE)
        self.iq = None
        self.running = False

        # Estatísticas (compatível com UI)
        self.saldo_inicial = 0.0
        self.total_trades = 0  # Nome usado pela UI
        self.total_ops = 0     # Alias
        self.wins = 0
        self.losses = 0

        # IA
        self.stats_ai = _safe_load_json(AI_STATS_FILE) if IA_ON else {"meta": {"total": 0}, "arms": {}}
        
        # Loss Analyzer
        self.loss_analyzer = None
        if LOSS_ANALYZER_ENABLED:
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
            self.loss_analyzer = get_loss_analyzer(backend_url)
            self._log("🔍 Loss Analyzer ativado")

    def _log(self, msg: str):
        """Log com callback para UI (sem duplicação)"""
        if self.logger_callback:
            self.logger_callback(msg)

    def _update_stats(self):
        """Atualiza estatísticas na UI"""
        if self.stats_callback:
            try:
                saldo_atual = float(self.iq.get_balance())
            except Exception:
                saldo_atual = self.saldo_inicial

            lucro = saldo_atual - self.saldo_inicial
            win_rate = (self.wins / self.total_ops * 100) if self.total_ops > 0 else 0.0

            self.stats_callback({
                "wins": self.wins,
                "losses": self.losses,
                "win_rate": win_rate,
                "profit": lucro,
                "saldo": saldo_atual
            })

    def conectar(self) -> bool:
        """Conecta à corretora selecionada"""
        try:
            self._log(f"🔌 Conectando à {self.broker_name}...")

            # Configura variável de ambiente para WS_AUTO_AI usar o broker correto
            broker_env = self.config.BROKER_TYPE.lower().strip().replace(" ", "_")
            os.environ["BROKER_TYPE"] = broker_env

            self.iq = self.BrokerAPIClass(self.config.EMAIL, self.config.SENHA)
            check, reason = self.iq.connect()

            if not check:
                self._log(f"❌ Falha na conexão: {reason}")
                return False

            self._log("✅ Conectado com sucesso!")
            self.iq.change_balance(self.config.CONTA)

            self.saldo_inicial = float(self.iq.get_balance())
            self._log(f"💰 Saldo inicial: R$ {self.saldo_inicial:.2f}")

            if USE_DYNAMIC_STAKE:
                self._log(f"📊 Gestão: {PERCENT_BANCA:.1f}% da banca por operação")
            else:
                self._log(f"📊 Gestão: Stake fixo de R$ {STAKE_FIXA:.2f}")

            self._log(f"🎯 Meta: {META_LUCRO_PERCENT:.1f}% de lucro")
            self._log(f"🛑 Stop: {STOP_LOSS_PERCENT:.1f}% de perda")

            self.running = True  # IA está rodando
            self._update_stats()
            return True

        except Exception as e:
            self._log(f"❌ Erro ao conectar: {e}")
            return False

    def loop_principal(self, stop_flag=None):
        """Executa UM ciclo do loop (compatível com UI)"""
        if stop_flag and stop_flag.is_set():
            return

        try:
            # Verifica meta/stop
            saldo_atual = float(self.iq.get_balance())
            deve_parar, lucro_percent = verificar_meta_atingida(self.saldo_inicial, saldo_atual)

            if deve_parar:
                lucro_abs = saldo_atual - self.saldo_inicial
                if lucro_percent >= META_LUCRO_PERCENT:
                    self._log(f"🎯 META ATINGIDA! Lucro: R$ {lucro_abs:.2f} ({lucro_percent:.2f}%)")
                else:
                    self._log(f"🛑 STOP LOSS! Perda: R$ {lucro_abs:.2f} ({lucro_percent:.2f}%)")
                self.running = False
                if stop_flag:
                    stop_flag.set()
                return

            # Obtém ativos
            ativos = obter_top_ativos_otc(self.iq)
            if not ativos:
                self._log("⚠️ Sem ativos disponíveis. Aguardando 10s...")
                time.sleep(10)
                return

            # Aguarda próximo candle
            wait_until_minus(TF_M1, DECIDIR_ANTES_FECHAR_SEC)
            t_antes_analise = time.time()

            # Escolhe melhor setup (CÓDIGO ORIGINAL)
            best_trade, best_any = escolher_melhor_setup(self.iq, ativos)

            # GUARD: Verifica se não passou do tempo (entrada atrasada)
            # Se a análise demorou demais e o candle já fechou + novo abriu,
            # a entrada seria no início do candle seguinte (fora do timing).
            tempo_analise = time.time() - t_antes_analise
            seg_restantes = TF_M1 - (time.time() % TF_M1)
            if seg_restantes > DECIDIR_ANTES_FECHAR_SEC + 5:
                # Já estamos no início do PRÓXIMO candle — entrada seria atrasada
                self._log(f"⏰ Análise demorou {tempo_analise:.1f}s, candle já virou. Pulando entrada.")
                return

            if not best_trade:
                if best_any:
                    sc, a, setup, _ = best_any
                    reasons = ", ".join(str(r) for r in setup.get("reasons", []))
                    self._log(f"⏸️ Melhor: {a} | score={sc:.2f} | {reasons} (abaixo threshold)")
                else:
                    self._log("⏸️ Nenhum setup encontrado")
                return

            sc, ativo, setup, atr_val = best_trade
            direcao = setup["dir"]
            reasons = ", ".join(str(r) for r in setup.get("reasons", []))

            # Filtro IA (CÓDIGO ORIGINAL)
            if IA_ON:
                pred = ai_predict(ativo, setup, self.stats_ai)
                prob = pred["prob"]
                conf = pred["conf"]
                n_arm = pred["n_arm"]

                if n_arm >= AI_MIN_SAMPLES and conf >= AI_CONF_MIN:
                    if prob < AI_MIN_PROB:
                        self._log(f"🤖 IA bloqueou: {ativo} | prob={prob:.2f} < {AI_MIN_PROB:.2f}")
                        return

                self._log(f"🤖 IA: prob={prob:.2f} conf={conf:.2f} n={n_arm}")

            # Calcula stake
            stake = calcular_stake_dinamico(self.iq, STAKE_FIXA)

            self._log(f"📊 SINAL: {ativo} | {direcao} | {reasons}")
            self._log(f"💰 Operando: {ativo} | {direcao} | R$ {stake:.2f} | Exp: {EXP_FIXA}min")

            # Envia ordem (CÓDIGO ORIGINAL)
            ordem = enviar_ordem(self.iq, ativo, direcao, stake)
            if not ordem:
                self._log("❌ Falha ao enviar ordem")
                return

            op_type, order_id = ordem
            self._log(f"✅ Ordem executada: {order_id} ({op_type})")

            self.total_ops += 1
            self.total_trades += 1
            cooldown[ativo] = time.time()

            # Callback de operação
            if self.operation_callback:
                payout = 85
                try:
                    payout = int(self.iq.get_payout_turbo(ativo) if op_type == "turbo" else 85)
                except Exception:
                    pass

                self.operation_callback({
                    "order_id": order_id,
                    "asset": ativo,
                    "direction": direcao,
                    "stake": stake,
                    "payout": payout,
                    "timestamp": datetime.now()
                })

            # Aguarda resultado com POLLING IMEDIATO (CÓDIGO ORIGINAL)
            self._aguardar_resultado(order_id, op_type, stake, ativo, direcao, setup)

        except Exception as e:
            self._log(f"❌ Erro no ciclo: {e}")
            import traceback
            self._log(traceback.format_exc())

    def _aguardar_resultado(self, order_id: int, op_type: str, stake: float,
                           ativo: str, direcao: str, setup: Dict[str, Any]):
        """Aguarda resultado com polling imediato"""
        tempo_expiracao = EXP_FIXA * 60

        # Aguarda expiração
        self._log(f"⏳ Aguardando {EXP_FIXA}min para resultado...")

        for i in range(tempo_expiracao):
            if i % 30 == 0 and i > 0:
                faltam = tempo_expiracao - i
                self._log(f"⏳ Aguardando... Faltam {faltam}s")
            time.sleep(1)

        # POLLING IMEDIATO (código original)
        self._log("🔍 Verificando resultado...")
        resultado_valor = wait_result(self.iq, op_type, order_id)

        # Processa resultado
        if resultado_valor > 0:
            lucro = resultado_valor - stake
            self._log(f"✅ WIN! Lucro: R$ {lucro:.2f}")
            self.wins += 1

            if self.result_callback:
                self.result_callback(order_id, "win", lucro)

            # Atualiza IA
            if IA_ON:
                ai_update(ativo, setup, lucro, self.stats_ai)
                _safe_save_json(AI_STATS_FILE, self.stats_ai)

        else:
            self._log(f"❌ LOSS! Perda: R$ {stake:.2f}")
            self.losses += 1

            if self.result_callback:
                self.result_callback(order_id, "loss", -stake)

            # 🔍 ANÁLISE DE LOSS
            if self.loss_analyzer:
                try:
                    self._log("🔍 Iniciando análise de loss...")
                    import threading
                    # Executa em thread separada para não bloquear
                    analysis_thread = threading.Thread(
                        target=self.loss_analyzer.analyze_loss,
                        args=(self.iq, order_id, ativo, direcao, stake, setup),
                        daemon=True
                    )
                    analysis_thread.start()
                except Exception as e:
                    self._log(f"⚠️ Erro ao iniciar análise de loss: {e}")

            # Atualiza IA
            if IA_ON:
                ai_update(ativo, setup, -stake, self.stats_ai)
                _safe_save_json(AI_STATS_FILE, self.stats_ai)

        # Atualiza estatísticas
        self._update_stats()

    def stop(self):
        """Para a IA"""
        self.running = False
        self._log("🛑 Parando IA...")
