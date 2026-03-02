# -*- coding: utf-8 -*-
"""
AGENTE 6: ESTRUTURA DE MERCADO
Verifica se não há Lower Highs / Higher Lows perigosos.
NUNCA MUDA.
"""
from typing import Any, Dict, Tuple
from agents.base_agent import BaseAgent


class StructureAgent(BaseAgent):
    name = "estrutura"
    description = "Estrutura de mercado sem sinais de perigo (LH/HL)"
    
    def analyze(self, setup: Dict[str, Any], brain_score: float = 0.0,
                atr_val: float = 0.0) -> Tuple[bool, str]:
        danger = bool(setup.get("structure_danger", False))
        reason_detail = str(setup.get("structure_reason", ""))
        lh = int(setup.get("structure_lh_count", 0))
        hl = int(setup.get("structure_hl_count", 0))
        
        if not danger:
            return True, "estrutura_ok"
        return False, f"estrutura_perigosa(LH={lh},HL={hl},{reason_detail})"
