# -*- coding: utf-8 -*-
"""
AGENTE 3: ZONA S/R FORTE
Verifica se a zona de suporte/resistência é forte o suficiente.
NUNCA MUDA.
"""
from typing import Any, Dict, Tuple
from agents.base_agent import BaseAgent


class ZoneStrengthAgent(BaseAgent):
    name = "zona_forte"
    description = "Zona S/R com toques suficientes e qualidade"
    
    def analyze(self, setup: Dict[str, Any], brain_score: float = 0.0,
                atr_val: float = 0.0) -> Tuple[bool, str]:
        touches = int(setup.get("sr_touches", 0))
        zone_str = float(setup.get("zone_strength", 0))
        clean_bounces = int(setup.get("zone_clean_bounces", 0))
        
        forte = (touches >= 3 and zone_str >= 0.50) or clean_bounces >= 1
        
        if forte:
            return True, f"zona({touches}t,{zone_str:.0%},bounce={clean_bounces})"
        return False, f"zona_fraca({touches}t,{zone_str:.0%})"
