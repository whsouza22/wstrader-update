# -*- coding: utf-8 -*-
"""
AGENTE 5: MOMENTUM
Verifica se o momentum não está forte contra o trade.
NUNCA MUDA.
"""
from typing import Any, Dict, Tuple
from agents.base_agent import BaseAgent


class MomentumAgent(BaseAgent):
    name = "momentum"
    description = "Momentum não está forte contra o trade"
    
    def analyze(self, setup: Dict[str, Any], brain_score: float = 0.0,
                atr_val: float = 0.0) -> Tuple[bool, str]:
        momentum_ok = bool(setup.get("momentum_ok", True))
        momentum_contra = int(setup.get("momentum_contra", 0))
        
        ok = momentum_ok or momentum_contra <= 1
        
        if ok:
            return True, f"momentum_ok(contra={momentum_contra})"
        return False, f"momentum_forte_contra({momentum_contra})"
