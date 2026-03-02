# -*- coding: utf-8 -*-
"""
AGENTE 4: REJEIÇÃO / CONFIRMAÇÃO
Verifica se o candle mostrou rejeição na zona (wick, hammer, etc).
NUNCA MUDA.
"""
from typing import Any, Dict, Tuple
from agents.base_agent import BaseAgent


class RejectionAgent(BaseAgent):
    name = "rejeicao"
    description = "Candle mostrou rejeição clara na zona"
    
    def analyze(self, setup: Dict[str, Any], brain_score: float = 0.0,
                atr_val: float = 0.0) -> Tuple[bool, str]:
        confirmed = bool(setup.get("bounce_confirmed", False))
        has_wick = bool(setup.get("has_wick_rejection", False))
        
        if confirmed:
            return True, "rejeicao_confirmada"
        if has_wick:
            return True, "wick_rejeicao"
        return False, "sem_rejeicao"
