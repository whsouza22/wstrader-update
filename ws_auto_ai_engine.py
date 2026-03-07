# -*- coding: utf-8 -*-
"""
WS Auto AI Engine - Wrapper para integração com UI Flet (trading_bot.py)
Importa do engine unificado WS_AUTO_AI_BULLEX.py
"""

import time
from typing import Optional, Callable, Dict, Any, Tuple
from datetime import datetime
import os

# ── Importa do engine UNIFICADO ──────────────────────────────
from WS_AUTO_AI_BULLEX import (
    TF_M1, EXP_FIXA, DECIDIR_ANTES_FECHAR_SEC,
    PERCENT_BANCA, META_LUCRO_PERCENT, STOP_LOSS_PERCENT, USE_DYNAMIC_STAKE,
    VALOR_MINIMO, STAKE_FIXA,
    IA_ON, AI_STATS_FILE, AI_MIN_SAMPLES, AI_CONF_MIN, AI_MIN_PROB,
    HORARIO_INICIO, HORARIO_FIM,
    safe_call, wait_until_minus,
    obter_top_ativos_otc, escolher_melhor_setup,
    enviar_ordem, wait_result,
    verificar_meta_atingida, calcular_stake_dinamico,
    ai_predict, ai_update,
    _safe_load_json, _safe_save_json,
    cooldown,
    _log_live_trade,
)

# ===================== MULTI-BROKER SUPPORT =====================
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


# Loss Analyzer desabilitado (módulo não disponível)
LOSS_ANALYZER_ENABLED = False


class TradingConfig:
    """Configuração mínima"""
    def __init__(self):
        self.EMAIL = ""
        self.SENHA = ""
        self.CONTA = "PRACTICE"
        self.BROKER_TYPE = "iq_option"


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

        self.BrokerAPIClass, self.broker_name = _get_broker_api(config.BROKER_TYPE)
        self.iq = None
        self.running = False

        self.saldo_inicial = 0.0
        self.total_trades = 0
        self.total_ops = 0
        self.wins = 0
        self.losses = 0

        # IA
        self.stats_ai = _safe_load_json(AI_STATS_FILE) if IA_ON else {"meta": {"total": 0}, "arms": {}}

        # Loss Analyzer (desabilitado — módulo não disponível)
        self.loss_analyzer = None

    def _log(self, msg: str):
        if self.logger_callback:
            self.logger_callback(msg)

    def _update_stats(self):
        if self.stats_callback:
            try:
                saldo_atual = float(self.iq.get_balance())
            except Exception:
                saldo_atual = self.saldo_inicial

            lucro = saldo_atual - self.saldo_inicial
            self.stats_callback({
                "wins": self.wins,
                "losses": self.losses,
                "win_rate": (self.wins / self.total_ops * 100) if self.total_ops > 0 else 0.0,
                "profit": lucro,
                "saldo": saldo_atual,
            })

    def conectar(self) -> bool:
        try:
            # Guarda final: só libera REAL se produto for PRO
            _pro_prod = "prod_S4t8FQuUptWQ6R"
            _demo_prod = "prod_U3CRqZJMVigJAK"
            _stripe_pid = os.environ.get("STRIPE_PRODUCT_ID", "")
            if _stripe_pid != _pro_prod:
                self.config.CONTA = "PRACTICE"
                _plan = "DEMO" if _stripe_pid == _demo_prod else "DESCONHECIDO"
                self._log(f"🔒 Plano {_plan} — forçando PRACTICE")

            self._log(f"🔌 Conectando à {self.broker_name}...")
            os.environ["BROKER_TYPE"] = self.config.BROKER_TYPE.lower().strip().replace(" ", "_")

            self.iq = self.BrokerAPIClass(self.config.EMAIL, self.config.SENHA)
            check, reason = self.iq.connect()
            if not check:
                reason_str = str(reason) if reason else ""
                reason_lower = reason_str.lower()
                if any(kw in reason_lower for kw in ["invalid", "credentials", "password", "unauthorized", "403", "incorrect", "wrong"]):
                    self._log(f"❌ SENHA_INCORRETA: Email ou senha incorretos para {self.broker_name}. Verifique suas credenciais.")
                elif "2fa" in reason_lower:
                    self._log(f"⚠️ 2FA_REQUIRED: {self.broker_name} requer verificação em 2 etapas.")
                else:
                    self._log(f"❌ Falha na conexão: {reason}")
                return False

            self._log("✅ Conectado com sucesso!")
            self.iq.change_balance(self.config.CONTA)
            self.saldo_inicial = float(self.iq.get_balance())
            self._log(f"💰 Saldo inicial: R$ {self.saldo_inicial:.2f}")
            self._log(f"🎯 Meta: {META_LUCRO_PERCENT:.1f}% | 🛑 Stop: {STOP_LOSS_PERCENT:.1f}%")
            self._log(f"🕒 Horário: {HORARIO_INICIO}h às {HORARIO_FIM}h | Estratégia: H&S Cabeça e Ombros")

            # Atualizar ACTIVES dinamicamente
            try:
                self.iq.update_ACTIVES_OPCODE()
                self._log("✅ ACTIVES atualizados")
            except Exception:
                pass

            self.running = True
            self._update_stats()
            return True
        except Exception as e:
            self._log(f"❌ Erro ao conectar: {e}")
            return False

    def loop_principal(self, stop_flag=None):
        if stop_flag and stop_flag.is_set():
            return
        try:
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

            # Verificar horário de operação (7h-18h)
            hora_atual = datetime.now().hour
            if hora_atual < HORARIO_INICIO or hora_atual >= HORARIO_FIM:
                self._log(f"⏰ Fora do horário ({hora_atual}h). Operando: {HORARIO_INICIO}h-{HORARIO_FIM}h")
                time.sleep(30)
                return

            ativos = obter_top_ativos_otc(self.iq)
            if not ativos:
                self._log("⚠️ Sem ativos disponíveis. Aguardando 10s...")
                time.sleep(10)
                return

            wait_until_minus(TF_M1, DECIDIR_ANTES_FECHAR_SEC)
            t_antes = time.time()
            best_trade, best_any = escolher_melhor_setup(self.iq, ativos)

            seg_restantes = TF_M1 - (time.time() % TF_M1)
            if seg_restantes > DECIDIR_ANTES_FECHAR_SEC + 5:
                self._log(f"⏰ Análise atrasada ({time.time()-t_antes:.1f}s). Pulando.")
                return

            if not best_trade:
                if best_any:
                    sc, a, setup, _ = best_any
                    self._log(f"⏸️ {a} | confianca insuficiente")
                else:
                    self._log("⏸️ Nenhum setup encontrado")
                return

            sc, ativo, setup, atr_val = best_trade
            direcao = setup["dir"]

            if IA_ON:
                pred = ai_predict(ativo, setup, self.stats_ai)
                if pred["n_arm"] >= AI_MIN_SAMPLES and pred["conf"] >= AI_CONF_MIN:
                    if pred["prob"] < AI_MIN_PROB:
                        self._log(f"🤖 IA bloqueou: {ativo} | sinal nao passou nos filtros")
                        return
                self._log(f"🤖 IA analisou {ativo} | confianca {'alta' if pred['prob'] >= 0.70 else 'media' if pred['prob'] >= 0.50 else 'baixa'}")

            stake = calcular_stake_dinamico(self.iq, STAKE_FIXA)
            self._log(f"📊 SINAL: {ativo} | {direcao} | R$ {stake:.2f}")

            ordem = enviar_ordem(self.iq, ativo, direcao, stake)
            if not ordem:
                self._log("❌ Falha ao enviar ordem")
                return

            op_type, order_id = ordem
            self._log(f"✅ Ordem: {order_id} ({op_type})")
            self.total_ops += 1
            self.total_trades += 1
            cooldown[ativo] = time.time()

            # ── Log para dashboard: registra entrada ──
            _ia_prob = pred.get("prob", 0.5) if IA_ON and 'pred' in dir() else 0.5
            _log_live_trade(ativo, direcao, None, None, stake,
                            confidence=_ia_prob * 100, status="entry")

            if self.operation_callback:
                payout = 85
                try:
                    payout = int(self.iq.get_payout_turbo(ativo) if op_type == "turbo" else 85)
                except Exception:
                    pass
                self.operation_callback({
                    "order_id": order_id, "asset": ativo, "direction": direcao,
                    "stake": stake, "payout": payout, "timestamp": datetime.now(),
                })

            self._aguardar_resultado(order_id, op_type, stake, ativo, direcao, setup)
        except Exception as e:
            self._log(f"❌ Erro no ciclo: {e}")
            import traceback
            self._log(traceback.format_exc())

    def _aguardar_resultado(self, order_id, op_type, stake, ativo, direcao, setup):
        tempo_exp = EXP_FIXA * 60
        self._log(f"⏳ Aguardando {EXP_FIXA}min...")
        for i in range(tempo_exp):
            if i % 30 == 0 and i > 0:
                self._log(f"⏳ Faltam {tempo_exp - i}s")
            time.sleep(1)

        self._log("🔍 Verificando resultado...")
        resultado_valor = wait_result(self.iq, op_type, order_id)

        if resultado_valor > 0:
            lucro = resultado_valor - stake
            self._log(f"✅ WIN! +R$ {lucro:.2f}")
            self.wins += 1
            # ── Log para dashboard: WIN ──
            _log_live_trade(ativo, direcao, lucro, None, stake, status="win")
            if self.result_callback:
                self.result_callback(order_id, "win", lucro)
            if IA_ON:
                ai_update(ativo, setup, lucro, self.stats_ai)
                _safe_save_json(AI_STATS_FILE, self.stats_ai)
        else:
            self._log(f"❌ LOSS! -R$ {stake:.2f}")
            self.losses += 1
            # ── Log para dashboard: LOSS ──
            _log_live_trade(ativo, direcao, -stake, None, stake, status="loss")
            if self.result_callback:
                self.result_callback(order_id, "loss", -stake)
            if self.loss_analyzer:
                try:
                    import threading
                    threading.Thread(
                        target=self.loss_analyzer.analyze_loss,
                        args=(self.iq, order_id, ativo, direcao, stake, setup),
                        daemon=True,
                    ).start()
                except Exception:
                    pass
            if IA_ON:
                ai_update(ativo, setup, -stake, self.stats_ai)
                _safe_save_json(AI_STATS_FILE, self.stats_ai)

        self._update_stats()

    def stop(self):
        self.running = False
        self._log("🛑 Parando IA...")
