# -*- coding: utf-8 -*-
"""
Auto Optimizer - Sistema de otimização automática baseado em análises de loss
Lê análises do Firebase e aplica ajustes automáticos no bot
"""

import json
import logging
import requests
from typing import Dict, Any, List, Optional
import os

logger = logging.getLogger(__name__)


class AutoOptimizer:
    """Otimizador automático baseado em análises de loss"""
    
    def __init__(self, backend_url: str = "http://localhost:8000", 
                 config_file: str = "auto_config.json"):
        self.backend_url = backend_url
        self.config_file = config_file
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Carrega configuração atual"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Erro ao carregar config: {e}")
        
        # Config padrão
        return {
            "filters": {
                "MIN_TREND_ALIGNMENT": 0.5,
                "MIN_VOLATILITY_RATIO": 0.7,
                "SR_MIN_DISTANCE_PERCENT": 0.15,
                "MIN_BODY_RATIO": 0.6,
                "MIN_ALIGNMENT_RATIO": 0.5,
                "HIGH_VOLATILITY_STAKE_REDUCTION": 0.7
            },
            "blacklist_assets": [],
            "optimization_history": []
        }
    
    def save_config(self):
        """Salva configuração"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Configuração salva em {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao salvar config: {e}")
            return False
    
    def get_recommendations(self) -> Optional[Dict[str, Any]]:
        """Busca recomendações do Firebase"""
        try:
            endpoint = f"{self.backend_url}/api/loss/recommendations"
            response = requests.get(endpoint, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return data
            
            logger.error(f"Erro ao buscar recomendações: {response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"Erro ao conectar com backend: {e}")
            return None
    
    def apply_recommendation(self, rec: Dict[str, Any]) -> bool:
        """Aplica uma recomendação específica"""
        try:
            config_suggestion = rec.get("config_suggestion", "")
            
            # Parse da sugestão (formato: "PARAM = valor")
            if " = " in config_suggestion:
                parts = config_suggestion.split(" = ")
                param_name = parts[0].strip()
                param_value = parts[1].strip()
                
                # Remove aspas se for string
                if param_value.startswith("[") and param_value.endswith("]"):
                    # É uma lista
                    param_value = json.loads(param_value.replace("'", '"'))
                elif param_value.replace(".", "").replace("-", "").isdigit():
                    # É um número
                    param_value = float(param_value) if "." in param_value else int(param_value)
                
                # Aplica no config
                if param_name == "BLACKLIST_ASSETS":
                    self.config["blacklist_assets"] = param_value
                else:
                    self.config["filters"][param_name] = param_value
                
                logger.info(f"✅ Aplicado: {param_name} = {param_value}")
                
                # Registra no histórico
                self.config["optimization_history"].append({
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                    "recommendation": rec.get("recommendation"),
                    "config": f"{param_name} = {param_value}",
                    "priority": rec.get("priority")
                })
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erro ao aplicar recomendação: {e}")
            return False
    
    def auto_optimize(self, apply_high_priority_only: bool = False) -> Dict[str, Any]:
        """
        Executa otimização automática
        
        Args:
            apply_high_priority_only: Se True, aplica apenas recomendações HIGH
        """
        logger.info("🔧 Iniciando otimização automática...")
        
        # 1. Busca recomendações
        recommendations_data = self.get_recommendations()
        if not recommendations_data:
            return {
                "success": False,
                "message": "Não foi possível buscar recomendações"
            }
        
        recommendations = recommendations_data.get("recommendations", [])
        if not recommendations:
            return {
                "success": True,
                "message": "Nenhuma recomendação disponível",
                "applied": 0
            }
        
        logger.info(f"📊 {len(recommendations)} recomendações encontradas")
        
        # 2. Aplica recomendações
        applied = 0
        skipped = 0
        
        for rec in recommendations:
            priority = rec.get("priority", "LOW")
            
            # Filtro por prioridade
            if apply_high_priority_only and priority != "HIGH":
                skipped += 1
                continue
            
            logger.info(f"\n[{priority}] {rec.get('category')}")
            logger.info(f"   Issue: {rec.get('issue')}")
            logger.info(f"   Rec: {rec.get('recommendation')}")
            
            if self.apply_recommendation(rec):
                applied += 1
        
        # 3. Salva configuração
        if applied > 0:
            self.save_config()
        
        result = {
            "success": True,
            "message": f"Otimização concluída: {applied} ajustes aplicados, {skipped} ignorados",
            "applied": applied,
            "skipped": skipped,
            "total_recommendations": len(recommendations),
            "based_on_losses": recommendations_data.get("based_on_losses", 0)
        }
        
        logger.info(f"\n✅ {result['message']}")
        return result
    
    def get_current_filters(self) -> Dict[str, Any]:
        """Retorna os filtros atualmente configurados"""
        return self.config.get("filters", {})
    
    def get_blacklist(self) -> List[str]:
        """Retorna lista de ativos bloqueados"""
        return self.config.get("blacklist_assets", [])
    
    def manual_adjust(self, param: str, value: Any) -> bool:
        """Ajuste manual de um parâmetro"""
        try:
            if param == "BLACKLIST_ASSETS":
                self.config["blacklist_assets"] = value if isinstance(value, list) else [value]
            else:
                self.config["filters"][param] = value
            
            self.save_config()
            logger.info(f"✅ Ajuste manual: {param} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro no ajuste manual: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """Reseta configuração para padrões"""
        self.config = {
            "filters": {
                "MIN_TREND_ALIGNMENT": 0.5,
                "MIN_VOLATILITY_RATIO": 0.7,
                "SR_MIN_DISTANCE_PERCENT": 0.15,
                "MIN_BODY_RATIO": 0.6,
                "MIN_ALIGNMENT_RATIO": 0.5,
                "HIGH_VOLATILITY_STAKE_REDUCTION": 0.7
            },
            "blacklist_assets": [],
            "optimization_history": []
        }
        self.save_config()
        logger.info("✅ Configuração resetada para padrões")
        return True
    
    def show_optimization_history(self) -> List[Dict[str, Any]]:
        """Mostra histórico de otimizações"""
        return self.config.get("optimization_history", [])


def run_auto_optimization(backend_url: str = "http://localhost:8000", 
                         high_priority_only: bool = False):
    """
    Função helper para executar otimização rápida
    """
    optimizer = AutoOptimizer(backend_url)
    result = optimizer.auto_optimize(apply_high_priority_only=high_priority_only)
    
    print("\n" + "="*60)
    print("🔧 OTIMIZAÇÃO AUTOMÁTICA DO BOT")
    print("="*60)
    print(f"\n{result['message']}")
    print(f"Baseado em {result.get('based_on_losses', 0)} losses analisados")
    
    if result["applied"] > 0:
        print("\n📋 NOVOS FILTROS APLICADOS:")
        filters = optimizer.get_current_filters()
        for key, value in filters.items():
            print(f"   {key}: {value}")
        
        blacklist = optimizer.get_blacklist()
        if blacklist:
            print(f"\n🚫 BLACKLIST: {', '.join(blacklist)}")
    
    print("\n" + "="*60)
    return result


# CLI para testes
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    backend = os.getenv("BACKEND_URL", "http://localhost:8000")
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        optimizer = AutoOptimizer(backend)
        
        if command == "optimize":
            run_auto_optimization(backend)
        
        elif command == "optimize-high":
            run_auto_optimization(backend, high_priority_only=True)
        
        elif command == "show":
            print("\n📋 CONFIGURAÇÃO ATUAL:")
            print(json.dumps(optimizer.config, indent=2, ensure_ascii=False))
        
        elif command == "history":
            history = optimizer.show_optimization_history()
            print(f"\n📜 HISTÓRICO DE OTIMIZAÇÕES ({len(history)}):")
            for i, entry in enumerate(history, 1):
                print(f"\n{i}. [{entry['priority']}] {entry['timestamp']}")
                print(f"   {entry['recommendation']}")
                print(f"   Config: {entry['config']}")
        
        elif command == "reset":
            optimizer.reset_to_defaults()
            print("✅ Configuração resetada!")
        
        else:
            print("Comandos disponíveis:")
            print("  optimize       - Otimiza baseado em todas recomendações")
            print("  optimize-high  - Aplica apenas recomendações HIGH priority")
            print("  show          - Mostra configuração atual")
            print("  history       - Mostra histórico de otimizações")
            print("  reset         - Reseta para configuração padrão")
    else:
        # Executa otimização padrão
        run_auto_optimization(backend)
