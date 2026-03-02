# -*- coding: utf-8 -*-
"""
AGENTE 8: BRAIN SCORE
Verifica se o modelo ML estimou P(win) >= 55%.
NUNCA MUDA.
"""
from typing import Any, Dict, Tuple
from agents.base_agent import BaseAgent


class BrainScoreAgent(BaseAgent):
    name = "brain_score"
    description = "Brain ML estima P(win) >= 55%"
    
    def analyze(self, setup: Dict[str, Any], brain_score: float = 0.0,
                atr_val: float = 0.0) -> Tuple[bool, str]:
        if brain_score >= 55.0:
            return True, f"brain={brain_score:.1f}%"
        return False, f"brain_baixo={brain_score:.1f}%"
