# -*- coding: utf-8 -*-
"""
AGENTE 1: TENDÊNCIA M1
Verifica se a tendência no timeframe M1 está a favor do trade.
NUNCA MUDA.
"""
from typing import Any, Dict, Tuple
from agents.base_agent import BaseAgent


class TrendM1Agent(BaseAgent):
    name = "tendencia_m1"
    description = "Tendência M1 a favor do trade"
    
    def analyze(self, setup: Dict[str, Any], brain_score: float = 0.0,
                atr_val: float = 0.0) -> Tuple[bool, str]:
        trend_ok = bool(setup.get("trend_ok", True))
        trend_dir = str(setup.get("trend_dir", "LATERAL"))
        
        if trend_ok:
            return True, f"tendencia_ok({trend_dir})"
        return False, f"tendencia_contra({trend_dir})"
