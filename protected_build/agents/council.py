# -*- coding: utf-8 -*-
"""
CONSELHO DE AGENTES — Orquestrador IMUTÁVEL
Carrega todos os agentes, executa a votação e registra decisões.
Os agentes NUNCA mudam. O conselho NUNCA muda.
Só o histórico (JSON) evolui com o tempo.

Inclui agente GPT externo que analisa 40 velas via ChatGPT
com 4 sub-analistas (tendência, estrutura, momentum, risco).
"""
import os
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from agents.base_agent import BaseAgent
from agents.agent_trend_m1 import TrendM1Agent
from agents.agent_trend_m5 import TrendM5Agent
from agents.agent_zone_strength import ZoneStrengthAgent
from agents.agent_rejection import RejectionAgent
from agents.agent_momentum import MomentumAgent
from agents.agent_structure import StructureAgent
from agents.agent_confluence import ConfluenceAgent
from agents.agent_brain_score import BrainScoreAgent
from agents.agent_gpt_structure import gpt_analyze_structure

log = logging.getLogger("WS_AI")


# ══════════════════════════════════════════════
# CONSTANTES IMUTÁVEIS
# ══════════════════════════════════════════════
MIN_VOTES = 5            # Mínimo de votos SIM para aprovar (de 8)
LEARN_MIN_SAMPLES = 20   # Amostras mínimas para aprender combo
COMBO_WR_BLOCK = 0.40    # WR abaixo disso = combo bloqueado
MAX_HISTORY = 500        # Máximo de decisões no JSON

# Ordem fixa dos agentes (NUNCA muda — define a combo_key)
AGENT_ORDER = [
    "tendencia_m1", "tendencia_m5", "zona_forte", "rejeicao",
    "momentum", "estrutura", "confluencia", "brain_score",
]


class AgentCouncil:
    """
    Conselho de 8 agentes independentes.
    
    Cada agente analisa UM aspecto do trade e vota SIM/NÃO.
    O conselho conta os votos e decide se entra ou não.
    
    Além disso, mantém um histórico de decisões com resultados
    (WIN/LOSS) para aprender quais combinações de votos funcionam.
    """
    
    def __init__(self, history_file: str):
        """
        Args:
            history_file: Caminho completo do JSON de histórico
                          (ex: ~/.wstrader/ws_agent_decisions_iq_option.json)
        """
        self.history_file = history_file
        
        # Instanciar os 8 agentes (IMUTÁVEIS)
        self.agents: List[BaseAgent] = [
            TrendM1Agent(),
            TrendM5Agent(),
            ZoneStrengthAgent(),
            RejectionAgent(),
            MomentumAgent(),
            StructureAgent(),
            ConfluenceAgent(),
            BrainScoreAgent(),
        ]
        
        # Histórico em memória
        self._decisions: List[Dict] = []
        self._combo_stats: Dict[str, Dict] = {}
        
        # Carregar histórico do disco
        self._load_history()
    
    # ══════════════════════════════════════════
    # VOTAÇÃO
    # ══════════════════════════════════════════
    
    def validate(self, setup: Dict[str, Any], brain_score: float,
                 atr_val: float = 0.0,
                 df: Optional[pd.DataFrame] = None,
                 direction: str = "",
                 ativo: str = "") -> Tuple[bool, Dict[str, bool], Dict[str, str], str]:
        """
        Executa a votação de todos os agentes + agente GPT.
        
        Args:
            setup: dict com indicadores do ativo
            brain_score: score do brain autônomo
            atr_val: ATR atual
            df: DataFrame M1 com OHLC (para agente GPT — últimas 40 velas)
            direction: "CALL" ou "PUT" (para agente GPT)
            ativo: nome do ativo (para agente GPT)
        
        Returns:
            approved: bool — entrada aprovada?
            votes: Dict[name, bool] — voto de cada agente
            reasons: Dict[name, str] — razão de cada agente
            summary: str — resumo legível
        """
        votes: Dict[str, bool] = {}
        reasons: Dict[str, str] = {}
        
        for agent in self.agents:
            vote, reason = agent.analyze(setup, brain_score, atr_val)
            votes[agent.name] = vote
            reasons[agent.name] = reason
        
        yes_count = sum(1 for v in votes.values() if v)
        
        # Consultar histórico de combos
        combo_key = self._combo_key(votes)
        combo_wr = None
        combo_block = False
        
        if combo_key in self._combo_stats:
            cs = self._combo_stats[combo_key]
            cs_total = cs.get("total", 0)
            cs_wins = cs.get("wins", 0)
            if cs_total >= LEARN_MIN_SAMPLES:
                combo_wr = cs_wins / max(cs_total, 1)
                if combo_wr < COMBO_WR_BLOCK:
                    combo_block = True
        
        # ══ AGENTE GPT: análise de estrutura com 40 velas ══
        gpt_ok = True
        gpt_reason = ""
        if df is not None and direction and yes_count >= MIN_VOTES and not combo_block:
            # Só chamar GPT se os agentes locais já aprovaram
            # (evita gastar tokens em sinais que já seriam bloqueados)
            gpt_ok, gpt_reason = gpt_analyze_structure(df, direction, setup, ativo=ativo)
            votes["gpt_estrutura"] = gpt_ok
            reasons["gpt_estrutura"] = gpt_reason
            if not gpt_ok:
                log.info(f"[GPT-BLOCK] 🚫 {ativo} {direction} | {gpt_reason}")
        
        # Decisão final
        approved = (yes_count >= MIN_VOTES) and not combo_block and gpt_ok
        
        # Montar resumo
        yes_names = [k for k, v in votes.items() if v]
        no_names = [k for k, v in votes.items() if not v]
        parts = [f"{yes_count}/8 a favor ({','.join(yes_names)})"]
        if no_names:
            parts.append(f"contra: {','.join(no_names)}")
        if combo_wr is not None:
            parts.append(f"combo_WR={combo_wr*100:.0f}%")
        if combo_block:
            parts.append("COMBO_HISTORICO_RUIM")
        if gpt_reason:
            parts.append(f"GPT={'OK' if gpt_ok else 'BLOCK'}")
        summary = " | ".join(parts)
        
        return approved, votes, reasons, summary
    
    # ══════════════════════════════════════════
    # REGISTRO DE DECISÕES
    # ══════════════════════════════════════════
    
    def record_decision(self, ativo: str, direction: str, setup: Dict[str, Any],
                        votes: Dict[str, bool], reasons: Dict[str, str],
                        approved: bool, brain_score: float):
        """Registra uma nova decisão (antes do resultado)."""
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ativo": ativo,
            "direction": direction,
            "approved": approved,
            "votes": votes,
            "reasons": reasons,
            "combo_key": self._combo_key(votes),
            "yes_count": sum(1 for v in votes.values() if v),
            "brain_score": round(brain_score, 1),
            "confluence_points": int(setup.get("confluence_points", 0)),
            "zone_strength": round(float(setup.get("zone_strength", 0)), 2),
            "trend_dir": str(setup.get("trend_dir", "?")),
            "m5_trend_dir": str(setup.get("m5_trend_dir", "?")),
            "sr_touches": int(setup.get("sr_touches", 0)),
            "result": None,
            "profit": None,
        }
        self._decisions.append(record)
        
        if len(self._decisions) > MAX_HISTORY + 100:
            self._decisions[:] = self._decisions[-MAX_HISTORY:]
        
        self._save_history()
    
    def record_result(self, ativo: str, votes: Dict[str, bool], result: float):
        """Atualiza o último registro com o resultado (WIN/LOSS/DRAW)."""
        combo_key = self._combo_key(votes)
        win = result > 0
        
        # Atualizar combo_stats
        if combo_key not in self._combo_stats:
            self._combo_stats[combo_key] = {"wins": 0, "total": 0, "losses": 0}
        self._combo_stats[combo_key]["total"] += 1
        if win:
            self._combo_stats[combo_key]["wins"] += 1
        else:
            self._combo_stats[combo_key]["losses"] += 1
        
        # Atualizar último registro pendente deste ativo
        for rec in reversed(self._decisions):
            if rec.get("ativo") == ativo and rec.get("result") is None:
                rec["result"] = "WIN" if win else ("LOSS" if result < 0 else "DRAW")
                rec["profit"] = round(result, 2)
                break
        
        self._save_history()
    
    # ══════════════════════════════════════════
    # HELPER PRIVADOS
    # ══════════════════════════════════════════
    
    def _combo_key(self, votes: Dict[str, bool]) -> str:
        """Gera chave binária para a combinação de votos (ex: '11010110')."""
        return "".join("1" if votes.get(k, False) else "0" for k in AGENT_ORDER)
    
    def _load_history(self):
        """Carrega histórico do JSON."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._decisions = data.get("decisions", [])
                self._combo_stats = data.get("combo_stats", {})
        except Exception:
            self._decisions = []
            self._combo_stats = {}
    
    def _save_history(self):
        """Salva histórico no JSON."""
        try:
            data = {
                "decisions": self._decisions[-MAX_HISTORY:],
                "combo_stats": self._combo_stats,
            }
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
        except Exception:
            pass
    
    def get_stats_summary(self) -> str:
        """Retorna resumo das estatísticas de combos para debug."""
        total_decisions = len(self._decisions)
        total_combos = len(self._combo_stats)
        wins = sum(1 for d in self._decisions if d.get("result") == "WIN")
        losses = sum(1 for d in self._decisions if d.get("result") == "LOSS")
        wr = wins / max(wins + losses, 1) * 100
        return (f"Agentes: {total_decisions} decisões, {total_combos} combos, "
                f"WR={wr:.0f}% ({wins}W/{losses}L)")
