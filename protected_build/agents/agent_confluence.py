# -*- coding: utf-8 -*-
"""
AGENTE 7: CONFLUÊNCIA
Verifica se há pelo menos 4 fatores independentes a favor.
NUNCA MUDA.
"""
from typing import Any, Dict, Tuple
from agents.base_agent import BaseAgent


class ConfluenceAgent(BaseAgent):
    name = "confluencia"
    description = "Mínimo de 4 fatores de confluência a favor"
    
    def analyze(self, setup: Dict[str, Any], brain_score: float = 0.0,
                atr_val: float = 0.0) -> Tuple[bool, str]:
        conf_pts = int(setup.get("confluence_points", 0))
        
        if conf_pts >= 4:
            return True, f"confluencia={conf_pts}"
        return False, f"confluencia_baixa={conf_pts}"
