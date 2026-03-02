# -*- coding: utf-8 -*-
"""
AGENTE 2: TENDÊNCIA M5
Verifica se o timeframe maior (M5) confirma a direção do trade.
NUNCA MUDA.
"""
from typing import Any, Dict, Tuple
from agents.base_agent import BaseAgent


class TrendM5Agent(BaseAgent):
    name = "tendencia_m5"
    description = "Tendência M5 não está contra o trade"
    
    def analyze(self, setup: Dict[str, Any], brain_score: float = 0.0,
                atr_val: float = 0.0) -> Tuple[bool, str]:
        m5_contra = bool(setup.get("m5_trend_contra", False))
        m5_dir = str(setup.get("m5_trend_dir", "?"))
        
        if not m5_contra:
            return True, f"m5_ok({m5_dir})"
        return False, f"m5_contra({m5_dir})"
