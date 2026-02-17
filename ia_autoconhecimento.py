# -*- coding: utf-8 -*-
"""
IA AUTO-CONHECIMENTO — Sistema de Aprendizado Inteligente

A IA aprende automaticamente:
- ONDE ENTRAR: Setups que deram WIN
- ONDE NAO ENTRAR: Setups que deram LOSS

Funcionalidades:
1. Salva TODOS os detalhes de cada operacao
2. Analisa padroes de LOSS vs WIN
3. Bloqueia automaticamente setups ruins
4. Reforça setups vencedores
5. Aprende com contexto de mercado (tendencia, volatilidade, hora, etc.)
"""

import os
import json
import logging
import tempfile
from datetime import datetime, date
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

def _atomic_write_json(path: str, data: dict):
    dir_name = os.path.dirname(os.path.abspath(path)) or "."
    base_name = os.path.basename(path)
    fd, tmp_path = tempfile.mkstemp(prefix=base_name + ".tmp.", dir=dir_name, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

# Arquivo de memoria da IA
MEMORIA_FILE = os.getenv("WS_IA_MEMORIA", "ws_ia_memoria.json")
ANALISE_FILE = os.getenv("WS_IA_ANALISE", "ws_ia_analise.json")

# Configuracoes de aprendizado
MIN_TRADES_PARA_APRENDER = int(os.getenv("WS_IA_MIN_TRADES", "3"))  # Minimo de trades para considerar padrao
WINRATE_MINIMO = float(os.getenv("WS_IA_WINRATE_MIN", "0.40"))  # Se winrate < 40%, bloqueia
LOSING_STREAK_MAX = int(os.getenv("WS_IA_LOSS_STREAK_MAX", "2"))  # Bloqueia apos N LOSS seguidos
CONFIANCA_MINIMA = float(os.getenv("WS_IA_CONF_MIN", "0.55"))  # Confianca minima para entrar
PENALTY_MINUTES = int(os.getenv("WS_IA_PENALTY_MIN", "30"))  # Bloqueio temporario por loss
PENALTY_MULT = float(os.getenv("WS_IA_PENALTY_MULT", "1.0"))


class IAAutoConhecimento:
    """
    Sistema de Auto-Conhecimento para Trading

    A IA aprende com cada operacao e ajusta automaticamente
    onde deve e onde NAO deve entrar.
    """

    def __init__(self):
        self.memoria = self._carregar_memoria()
        self.analise = self._carregar_analise()
        self.sessao_atual = {
            "data": date.today().isoformat(),
            "trades": [],
            "wins": 0,
            "losses": 0,
            "lucro": 0.0
        }
        logger.info(f"[IA-AUTO] Memoria carregada: {len(self.memoria.get('trades', []))} trades historicos")

    def _carregar_memoria(self) -> dict:
        """Carrega memoria do arquivo"""
        try:
            if os.path.exists(MEMORIA_FILE):
                with open(MEMORIA_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"[IA-AUTO] Erro ao carregar memoria: {e}")
            try:
                if os.path.exists(MEMORIA_FILE):
                    backup = MEMORIA_FILE + ".bad"
                    with open(MEMORIA_FILE, 'r', encoding='utf-8') as src:
                        raw = src.read()
                    with open(backup, 'w', encoding='utf-8') as dst:
                        dst.write(raw)
                    logger.error(f"[IA-AUTO] Memoria corrompida movida para: {backup}")
            except Exception:
                pass

        return {
            "trades": [],
            "setups_bloqueados": [],
            "setups_favoritos": [],
            "ativos_bloqueados": [],
            "penalty": {},
            "estatisticas": {},
            "ultima_atualizacao": None
        }

    def _carregar_analise(self) -> dict:
        """Carrega analise do arquivo"""
        try:
            if os.path.exists(ANALISE_FILE):
                with open(ANALISE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"[IA-AUTO] Erro ao carregar analise: {e}")
            try:
                if os.path.exists(ANALISE_FILE):
                    backup = ANALISE_FILE + ".bad"
                    with open(ANALISE_FILE, 'r', encoding='utf-8') as src:
                        raw = src.read()
                    with open(backup, 'w', encoding='utf-8') as dst:
                        dst.write(raw)
                    logger.error(f"[IA-AUTO] Analise corrompida movida para: {backup}")
            except Exception:
                pass

        return {
            "padroes_loss": {},
            "padroes_win": {},
            "contextos_ruins": [],
            "contextos_bons": [],
            "hora_ruim": [],
            "hora_boa": []
        }

    def _salvar_memoria(self):
        """Salva memoria no arquivo"""
        try:
            self.memoria["ultima_atualizacao"] = datetime.now().isoformat()
            _atomic_write_json(MEMORIA_FILE, self.memoria)
        except Exception as e:
            logger.error(f"[IA-AUTO] Erro ao salvar memoria: {e}")

    def _salvar_analise(self):
        """Salva analise no arquivo"""
        try:
            _atomic_write_json(ANALISE_FILE, self.analise)
        except Exception as e:
            logger.error(f"[IA-AUTO] Erro ao salvar analise: {e}")

    def registrar_trade(self, trade_info: dict):
        """
        Registra um trade na memoria.

        trade_info deve conter:
        - ativo: str
        - direcao: "CALL" ou "PUT"
        - resultado: "WIN", "LOSS" ou "EMPATE"
        - lucro: float
        - contexto: dict com detalhes do mercado
        """
        trade = {
            "timestamp": datetime.now().isoformat(),
            "data": date.today().isoformat(),
            "hora": datetime.now().strftime("%H:%M"),
            "ativo": trade_info.get("ativo", ""),
            "direcao": trade_info.get("direcao", ""),
            "resultado": trade_info.get("resultado", ""),
            "lucro": trade_info.get("lucro", 0.0),
            "contexto": trade_info.get("contexto", {}),
            "setup_key": self._gerar_setup_key(trade_info)
        }

        # Adiciona na memoria
        self.memoria["trades"].append(trade)

        # Limita a 1000 trades mais recentes
        if len(self.memoria["trades"]) > 1000:
            self.memoria["trades"] = self.memoria["trades"][-1000:]

        # Atualiza sessao atual
        self.sessao_atual["trades"].append(trade)
        if trade["resultado"] == "WIN":
            self.sessao_atual["wins"] += 1
        elif trade["resultado"] == "LOSS":
            self.sessao_atual["losses"] += 1
        self.sessao_atual["lucro"] += trade["lucro"]

        # Atualiza estatisticas
        self._atualizar_estatisticas(trade)

        # Analisa e aprende
        self._aprender_com_trade(trade)

        # Salva
        self._salvar_memoria()
        self._salvar_analise()

        logger.info(f"[IA-AUTO] Trade registrado: {trade['ativo']} {trade['direcao']} = {trade['resultado']}")

    def _gerar_setup_key(self, trade_info: dict) -> str:
        """Gera uma chave unica para o setup"""
        ativo = trade_info.get("ativo", "")
        direcao = trade_info.get("direcao", "")
        contexto = trade_info.get("contexto", {})

        tendencia = contexto.get("tendencia", "lateral")
        volatilidade = contexto.get("volatilidade", "normal")
        pernada_a = contexto.get("pernada_a", 0)
        pernada_b = contexto.get("pernada_b", 0)
        retraction = contexto.get("retraction", 0)

        # Categoriza retracao
        if retraction < 0.30:
            ret_cat = "rasa"
        elif retraction < 0.50:
            ret_cat = "media"
        elif retraction < 0.70:
            ret_cat = "profunda"
        else:
            ret_cat = "muito_profunda"

        return f"{ativo}_{direcao}_{tendencia}_{volatilidade}_A{pernada_a}_B{pernada_b}_{ret_cat}"

    def _atualizar_estatisticas(self, trade: dict):
        """Atualiza estatisticas do setup"""
        setup_key = trade["setup_key"]

        if setup_key not in self.memoria["estatisticas"]:
            self.memoria["estatisticas"][setup_key] = {
                "total": 0,
                "wins": 0,
                "losses": 0,
                "empates": 0,
                "lucro_total": 0.0,
                "ultima_sequencia": [],  # Ultimos 5 resultados
                "win_rate": 0.0
            }

        stats = self.memoria["estatisticas"][setup_key]
        stats["total"] += 1
        stats["lucro_total"] += trade["lucro"]

        if trade["resultado"] == "WIN":
            stats["wins"] += 1
        elif trade["resultado"] == "LOSS":
            stats["losses"] += 1
        else:
            stats["empates"] += 1

        # Atualiza sequencia
        stats["ultima_sequencia"].append(1 if trade["resultado"] == "WIN" else 0)
        if len(stats["ultima_sequencia"]) > 5:
            stats["ultima_sequencia"] = stats["ultima_sequencia"][-5:]

        # Calcula win rate
        if stats["wins"] + stats["losses"] > 0:
            stats["win_rate"] = stats["wins"] / (stats["wins"] + stats["losses"])

    def _aprender_com_trade(self, trade: dict):
        """Aprende com o resultado do trade"""
        setup_key = trade["setup_key"]
        stats = self.memoria["estatisticas"].get(setup_key, {})

        # Verifica se deve bloquear
        if self._deve_bloquear_setup(setup_key, stats):
            if setup_key not in self.memoria["setups_bloqueados"]:
                self.memoria["setups_bloqueados"].append(setup_key)
                logger.info(f"[IA-AUTO] BLOQUEADO: {setup_key} (WinRate muito baixo)")

        # Verifica se deve favoritar
        elif self._deve_favoritar_setup(setup_key, stats):
            if setup_key not in self.memoria["setups_favoritos"]:
                self.memoria["setups_favoritos"].append(setup_key)
                logger.info(f"[IA-AUTO] FAVORITO: {setup_key} (WinRate alto)")

        # Verifica sequencia de LOSS
        if self._detectar_losing_streak(stats):
            if setup_key not in self.memoria["setups_bloqueados"]:
                self.memoria["setups_bloqueados"].append(setup_key)
                logger.info(f"[IA-AUTO] BLOQUEADO: {setup_key} (Losing streak)")

        # Penaliza por LOSS / alivia por WIN
        if trade["resultado"] == "LOSS":
            self._penalizar_setup(setup_key)
        elif trade["resultado"] == "WIN":
            self._aliviar_penalidade(setup_key)

        # Analisa contexto
        self._analisar_contexto(trade)

    def _penalizar_setup(self, setup_key: str):
        penalty = self.memoria.get("penalty", {})
        item = penalty.get(setup_key, {"count": 0, "blocked_until": 0})
        item["count"] = int(item.get("count", 0)) + 1
        bloqueio_min = max(1, int(PENALTY_MINUTES * max(1.0, float(PENALTY_MULT)) * item["count"]))
        item["blocked_until"] = int(datetime.now().timestamp()) + (bloqueio_min * 60)
        penalty[setup_key] = item
        self.memoria["penalty"] = penalty

    def _aliviar_penalidade(self, setup_key: str):
        penalty = self.memoria.get("penalty", {})
        item = penalty.get(setup_key)
        if not item:
            return
        item["count"] = max(0, int(item.get("count", 0)) - 1)
        if item["count"] <= 0:
            penalty.pop(setup_key, None)
        else:
            penalty[setup_key] = item
        self.memoria["penalty"] = penalty

    def _deve_bloquear_setup(self, setup_key: str, stats: dict) -> bool:
        """Verifica se deve bloquear o setup"""
        total = stats.get("total", 0)
        win_rate = stats.get("win_rate", 0.5)

        # Precisa de minimo de trades
        if total < MIN_TRADES_PARA_APRENDER:
            return False

        # Bloqueia se win rate muito baixo
        if win_rate < WINRATE_MINIMO:
            return True

        return False

    def _deve_favoritar_setup(self, setup_key: str, stats: dict) -> bool:
        """Verifica se deve favoritar o setup"""
        total = stats.get("total", 0)
        win_rate = stats.get("win_rate", 0.5)

        # Precisa de minimo de trades
        if total < MIN_TRADES_PARA_APRENDER:
            return False

        # Favorita se win rate alto (>= 70%)
        if win_rate >= 0.70:
            return True

        return False

    def _detectar_losing_streak(self, stats: dict) -> bool:
        """Detecta sequencia de LOSS"""
        sequencia = stats.get("ultima_sequencia", [])

        if len(sequencia) < LOSING_STREAK_MAX:
            return False

        # Verifica se ultimos N foram LOSS
        ultimos = sequencia[-LOSING_STREAK_MAX:]
        return all(r == 0 for r in ultimos)

    def _analisar_contexto(self, trade: dict):
        """Analisa o contexto do trade para aprender padroes"""
        contexto = trade.get("contexto", {})
        resultado = trade["resultado"]
        hora = trade["hora"]

        # Cria fingerprint do contexto
        fingerprint = {
            "tendencia": contexto.get("tendencia", "lateral"),
            "volatilidade": contexto.get("volatilidade", "normal"),
            "retraction": round(contexto.get("retraction", 0), 1),
            "pernada_a": contexto.get("pernada_a", 0),
            "pernada_b": contexto.get("pernada_b", 0),
            "hora": hora[:2]  # Apenas a hora (ex: "18")
        }

        fingerprint_str = json.dumps(fingerprint, sort_keys=True)

        if resultado == "LOSS":
            if fingerprint_str not in self.analise["padroes_loss"]:
                self.analise["padroes_loss"][fingerprint_str] = 0
            self.analise["padroes_loss"][fingerprint_str] += 1

            # Adiciona hora ruim
            if hora[:2] not in self.analise["hora_ruim"]:
                hora_count = sum(1 for t in self.memoria["trades"][-50:]
                               if t["hora"][:2] == hora[:2] and t["resultado"] == "LOSS")
                if hora_count >= 3:
                    self.analise["hora_ruim"].append(hora[:2])

        elif resultado == "WIN":
            if fingerprint_str not in self.analise["padroes_win"]:
                self.analise["padroes_win"][fingerprint_str] = 0
            self.analise["padroes_win"][fingerprint_str] += 1

    def pode_entrar(self, ativo: str, direcao: str, contexto: dict) -> dict:
        """
        Verifica se a IA permite entrar neste trade.

        Retorna:
        - pode: bool
        - confianca: float (0-1)
        - motivo: str
        """
        # Gera setup key
        trade_info = {"ativo": ativo, "direcao": direcao, "contexto": contexto}
        setup_key = self._gerar_setup_key(trade_info)

        # 0. Penalidade ativa
        penalty = self.memoria.get("penalty", {}).get(setup_key)
        if penalty and int(penalty.get("blocked_until", 0)) > int(datetime.now().timestamp()):
            return {
                "pode": False,
                "confianca": 0.0,
                "motivo": f"PENALIZADO: {setup_key} em cooldown"
            }

        # 0.5 Filtros de motivos ruins (penaliza entrada)
        motivos = contexto.get("motivos", []) or []
        motivos_txt = " ".join([str(m) for m in motivos]).lower()
        motivos_ruins = [
            "longe_fundo",
            "fundo_distante",
            "longe_topo",
            "topo_distante",
            "vela_esticada",
            "tendencia_alta_forte",
            "tendencia_baixa_forte"
        ]
        if any(m in motivos_txt for m in motivos_ruins):
            return {
                "pode": False,
                "confianca": 0.0,
                "motivo": "FILTRO_IA: motivo_ruim"
            }

        # 1. Verifica se setup esta bloqueado
        if setup_key in self.memoria["setups_bloqueados"]:
            return {
                "pode": False,
                "confianca": 0.0,
                "motivo": f"BLOQUEADO: {setup_key} tem historico ruim"
            }

        # 2. Verifica se ativo esta bloqueado
        if ativo in self.memoria.get("ativos_bloqueados", []):
            return {
                "pode": False,
                "confianca": 0.0,
                "motivo": f"ATIVO BLOQUEADO: {ativo}"
            }

        # 3. Verifica hora ruim
        hora_atual = datetime.now().strftime("%H")
        if hora_atual in self.analise.get("hora_ruim", []):
            return {
                "pode": False,
                "confianca": 0.0,
                "motivo": f"HORA RUIM: {hora_atual}h tem historico de LOSS"
            }

        # 4. Calcula confianca baseada no historico
        stats = self.memoria["estatisticas"].get(setup_key, {})
        total = stats.get("total", 0)
        win_rate = stats.get("win_rate", 0.5)

        if total >= MIN_TRADES_PARA_APRENDER:
            confianca = win_rate
        else:
            # Sem historico suficiente, usa confianca neutra
            confianca = 0.55

        # 5. Bonus se for setup favorito
        if setup_key in self.memoria.get("setups_favoritos", []):
            confianca = min(1.0, confianca + 0.10)

        # 6. Analisa contexto atual vs padroes de LOSS
        fingerprint = {
            "tendencia": contexto.get("tendencia", "lateral"),
            "volatilidade": contexto.get("volatilidade", "normal"),
            "retraction": round(contexto.get("retraction", 0), 1),
            "pernada_a": contexto.get("pernada_a", 0),
            "pernada_b": contexto.get("pernada_b", 0),
            "hora": hora_atual
        }
        fingerprint_str = json.dumps(fingerprint, sort_keys=True)

        loss_count = self.analise["padroes_loss"].get(fingerprint_str, 0)
        win_count = self.analise["padroes_win"].get(fingerprint_str, 0)

        if loss_count > win_count and loss_count >= 3:
            return {
                "pode": False,
                "confianca": 0.0,
                "motivo": f"CONTEXTO RUIM: Similar a {loss_count} LOSSes anteriores"
            }

        # 7. Decisao final
        if confianca >= CONFIANCA_MINIMA:
            return {
                "pode": True,
                "confianca": confianca,
                "motivo": f"OK: Confianca {confianca*100:.0f}% (historico: {total} trades, WR: {win_rate*100:.0f}%)"
            }
        else:
            return {
                "pode": False,
                "confianca": confianca,
                "motivo": f"BAIXA CONFIANCA: {confianca*100:.0f}% < {CONFIANCA_MINIMA*100:.0f}%"
            }

    def get_estatisticas(self) -> dict:
        """Retorna estatisticas gerais"""
        trades = self.memoria.get("trades", [])

        total = len(trades)
        wins = sum(1 for t in trades if t["resultado"] == "WIN")
        losses = sum(1 for t in trades if t["resultado"] == "LOSS")

        return {
            "total_trades": total,
            "total_wins": wins,
            "total_losses": losses,
            "win_rate": wins / total if total > 0 else 0,
            "setups_bloqueados": len(self.memoria.get("setups_bloqueados", [])),
            "setups_favoritos": len(self.memoria.get("setups_favoritos", [])),
            "sessao": self.sessao_atual
        }

    def desbloquear_setup(self, setup_key: str):
        """Desbloqueia um setup manualmente"""
        if setup_key in self.memoria["setups_bloqueados"]:
            self.memoria["setups_bloqueados"].remove(setup_key)
            self._salvar_memoria()
            logger.info(f"[IA-AUTO] DESBLOQUEADO: {setup_key}")

    def resetar_dia(self):
        """Reseta estatisticas do dia"""
        self.sessao_atual = {
            "data": date.today().isoformat(),
            "trades": [],
            "wins": 0,
            "losses": 0,
            "lucro": 0.0
        }
        logger.info("[IA-AUTO] Sessao do dia resetada")

    def listar_bloqueados(self) -> List[str]:
        """Lista todos os setups bloqueados"""
        return self.memoria.get("setups_bloqueados", [])

    def listar_favoritos(self) -> List[str]:
        """Lista todos os setups favoritos"""
        return self.memoria.get("setups_favoritos", [])


# Instancia global
ia_autoconhecimento = IAAutoConhecimento()


# Funcoes de conveniencia
def registrar_trade(ativo: str, direcao: str, resultado: str, lucro: float, contexto: dict):
    """Registra um trade no sistema de auto-conhecimento"""
    ia_autoconhecimento.registrar_trade({
        "ativo": ativo,
        "direcao": direcao,
        "resultado": resultado,
        "lucro": lucro,
        "contexto": contexto
    })


def pode_entrar(ativo: str, direcao: str, contexto: dict) -> dict:
    """Verifica se pode entrar no trade"""
    return ia_autoconhecimento.pode_entrar(ativo, direcao, contexto)


def get_estatisticas() -> dict:
    """Retorna estatisticas"""
    return ia_autoconhecimento.get_estatisticas()


if __name__ == "__main__":
    # Teste
    print("Sistema de Auto-Conhecimento da IA")
    stats = get_estatisticas()
    print(f"Total trades: {stats['total_trades']}")
    print(f"Win Rate: {stats['win_rate']*100:.1f}%")
    print(f"Setups bloqueados: {stats['setups_bloqueados']}")
    print(f"Setups favoritos: {stats['setups_favoritos']}")
