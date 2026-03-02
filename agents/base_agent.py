# -*- coding: utf-8 -*-
"""
AGENTE BASE — Interface que todos os agentes implementam.
NUNCA MUDA. É a fundação do sistema de agentes.
"""
from typing import Any, Dict, Tuple


class BaseAgent:
    """
    Interface base para todos os agentes validadores.
    
    Cada agente:
      - Tem um NOME único (usado como chave nos votos)
      - Tem uma DESCRIÇÃO do que analisa  
      - Implementa analyze() que retorna (voto: bool, razão: str)
      - É IMUTÁVEL — a lógica de cada agente NUNCA muda após criação
    """
    
    name: str = "base"
    description: str = "Agente base"
    
    def analyze(self, setup: Dict[str, Any], brain_score: float = 0.0,
                atr_val: float = 0.0) -> Tuple[bool, str]:
        """
        Analisa o setup e retorna:
          - vote (bool): True=SIM (pode entrar), False=NÃO (não entrar)
          - reason (str): Explicação curta da decisão
        """
        raise NotImplementedError("Cada agente deve implementar analyze()")
