# -*- coding: utf-8 -*-
"""
AI Learning Module - Sistema de Aprendizado Inteligente
Aprende com losses do Firebase e aplica penalizações nos setups
"""

import json
import logging
import requests
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

# ===================== CONFIGURAÇÃO =====================
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
LEARNING_CACHE_FILE = os.getenv("WS_LEARNING_CACHE", "ws_ai_learning_cache.json")
LEARNING_REFRESH_SEC = int(os.getenv("WS_LEARNING_REFRESH", "120"))  # Atualiza a cada 2 min (mais rapido)
MIN_LOSSES_TO_LEARN = int(os.getenv("WS_MIN_LOSSES_LEARN", "1"))  # RIGOROSO: aprende com 1 loss apenas

# Pesos de penalizacao RIGOROSOS para 80%+ win rate
PENALTY_WEIGHTS = {
    "contra_tendencia": 0.25,      # RIGOROSO: 25% - nunca operar contra tendencia
    "consolidacao": 0.30,          # RIGOROSO: 30% - evitar mercado lateral
    "sr_forte": 0.20,              # RIGOROSO: 20% - cuidado com S/R
    "entrada_fraca": 0.18,         # RIGOROSO: 18% - exige entrada forte
    "desalinhamento": 0.20,        # RIGOROSO: 20% - velas precisam alinhar
    "alta_volatilidade": 0.12,     # RIGOROSO: 12% - cuidado com volatilidade
    "momentum_errado": 0.18,       # RIGOROSO: 18% - momentum deve confirmar
}

# Criterios de bloqueio RIGOROSOS para 80%+ win rate
LEARNED_FILTERS = {
    "MIN_TREND_ALIGNMENT": 0.55,      # RIGOROSO: 55% alinhamento com tendencia
    "MIN_VOLATILITY_RATIO": 0.85,     # RIGOROSO: 85% volatilidade adequada
    "SR_MIN_DISTANCE_PERCENT": 0.002, # RIGOROSO: 0.2% distancia de S/R
    "MIN_BODY_RATIO": 0.60,           # RIGOROSO: 60% corpo da vela
    "MIN_ALIGNMENT_RATIO": 0.60,      # RIGOROSO: 60% alinhamento de velas
    "MAX_CONSOLIDATION_STD": 0.0008,  # RIGOROSO: desvio menor = mais sensivel
}


class AILearning:
    """Sistema de aprendizado inteligente baseado em losses"""
    
    def __init__(self, backend_url: str = None):
        self.backend_url = backend_url or BACKEND_URL
        self.cache = {"last_update": 0, "statistics": {}, "losses": [], "penalties": {}}
        self.learned_filters = LEARNED_FILTERS.copy()
        self._load_cache()
    
    def _load_cache(self):
        """Carrega cache de aprendizado do disco"""
        try:
            if os.path.exists(LEARNING_CACHE_FILE):
                with open(LEARNING_CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.cache = data.get("cache", self.cache)
                    self.learned_filters = data.get("learned_filters", self.learned_filters)
                    logger.info(f"📚 Cache de aprendizado carregado: {len(self.cache.get('losses', []))} losses")
        except Exception as e:
            logger.warning(f"Erro ao carregar cache: {e}")
    
    def _save_cache(self):
        """Salva cache de aprendizado no disco"""
        try:
            data = {
                "cache": self.cache,
                "learned_filters": self.learned_filters,
                "updated_at": datetime.now().isoformat()
            }
            with open(LEARNING_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Erro ao salvar cache: {e}")
    
    def _fetch_losses_from_firebase(self, limit: int = 100) -> List[Dict]:
        """Busca losses recentes do Firebase"""
        try:
            endpoint = f"{self.backend_url}/api/loss/list?limit={limit}"
            response = requests.get(endpoint, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return data.get("analyses", [])
            return []
        except Exception as e:
            logger.warning(f"Erro ao buscar losses do Firebase: {e}")
            return []
    
    def _fetch_statistics(self) -> Dict:
        """Busca estatísticas agregadas do Firebase"""
        try:
            endpoint = f"{self.backend_url}/api/loss/statistics"
            response = requests.get(endpoint, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return data.get("statistics", {})
            return {}
        except Exception as e:
            logger.warning(f"Erro ao buscar estatísticas: {e}")
            return {}
    
    def _fetch_recommendations(self) -> List[Dict]:
        """Busca recomendações do Firebase"""
        try:
            endpoint = f"{self.backend_url}/api/loss/recommendations"
            response = requests.get(endpoint, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return data.get("recommendations", [])
            return []
        except Exception as e:
            logger.warning(f"Erro ao buscar recomendações: {e}")
            return []
    
    def refresh_learning(self, force: bool = False) -> bool:
        """Atualiza aprendizado a partir do Firebase"""
        now = time.time()
        
        # Verifica se precisa atualizar
        if not force and (now - self.cache.get("last_update", 0)) < LEARNING_REFRESH_SEC:
            return False
        
        logger.info("🔄 Atualizando aprendizado do Firebase...")
        
        # Busca dados do Firebase
        losses = self._fetch_losses_from_firebase()
        statistics = self._fetch_statistics()
        recommendations = self._fetch_recommendations()
        
        if not losses and not statistics:
            logger.warning("Não foi possível obter dados do Firebase")
            return False
        
        # Atualiza cache
        self.cache["losses"] = losses
        self.cache["statistics"] = statistics
        self.cache["recommendations"] = recommendations
        self.cache["last_update"] = now
        
        # Calcula penalidades por ativo e padrão
        self._calculate_penalties()
        
        # Ajusta filtros automaticamente
        self._auto_adjust_filters()
        
        # Salva cache
        self._save_cache()
        
        logger.info(f"✅ Aprendizado atualizado: {len(losses)} losses analisados")
        return True
    
    def _calculate_penalties(self):
        """Calcula penalidades baseadas nos losses"""
        penalties = {
            "assets": {},      # Penalidades por ativo
            "patterns": {},    # Penalidades por padrão
            "contexts": {},    # Penalidades por contexto
        }
        
        losses = self.cache.get("losses", [])
        
        for loss in losses:
            asset = loss.get("asset", "")
            direction = loss.get("direction", "")
            market_context = loss.get("market_context", {})
            entry_quality = loss.get("entry_quality", {})
            ai_analysis = loss.get("ai_analysis", "").lower()
            
            # Penalidade por ativo
            if asset not in penalties["assets"]:
                penalties["assets"][asset] = {"count": 0, "total_stake": 0, "penalty": 0.0}
            penalties["assets"][asset]["count"] += 1
            penalties["assets"][asset]["total_stake"] += loss.get("stake", 0)
            
            # Penalidade por padrões detectados na análise
            pattern_key = f"{asset}_{direction}"
            if pattern_key not in penalties["patterns"]:
                penalties["patterns"][pattern_key] = {"count": 0, "problems": [], "penalty": 0.0}
            
            penalties["patterns"][pattern_key]["count"] += 1
            
            # Identifica problemas específicos
            problems = []
            if "contra tendência" in ai_analysis:
                problems.append("contra_tendencia")
            if "consolidação" in ai_analysis:
                problems.append("consolidacao")
            if "resistência" in ai_analysis or "suporte" in ai_analysis:
                problems.append("sr_forte")
            if "entrada fraca" in ai_analysis:
                problems.append("entrada_fraca")
            if "desalinhadas" in ai_analysis:
                problems.append("desalinhamento")
            if "alta volatilidade" in ai_analysis:
                problems.append("alta_volatilidade")
            if market_context.get("is_consolidating"):
                problems.append("consolidacao")
            if entry_quality.get("momentum_direction") == "wrong":
                problems.append("momentum_errado")
            if entry_quality.get("alignment_ratio", 1.0) < 0.4:
                problems.append("desalinhamento")
            
            penalties["patterns"][pattern_key]["problems"].extend(problems)
            
            # Penalidade por contexto de mercado
            ctx_key = f"{market_context.get('trend', 'neutral')}_{market_context.get('volatility', 'medium')}"
            if ctx_key not in penalties["contexts"]:
                penalties["contexts"][ctx_key] = {"count": 0, "penalty": 0.0}
            penalties["contexts"][ctx_key]["count"] += 1
        
        # Calcula penalidade final por ativo (baseado em frequência)
        total_losses = len(losses) if losses else 1
        for asset, data in penalties["assets"].items():
            # SÓ aplica penalidade se tiver pelo menos 3 losses nesse ativo
            if data["count"] < 3:
                data["penalty"] = 0.0
                continue
            
            freq = data["count"] / total_losses
            # Penaliza mais se o ativo representa >30% dos losses (ajustado)
            if freq > 0.40 and data["count"] >= 5:
                data["penalty"] = 0.20  # Penaliza 20%
            elif freq > 0.30 and data["count"] >= 4:
                data["penalty"] = 0.15
            elif freq > 0.20 and data["count"] >= 3:
                data["penalty"] = 0.10
        
        # Calcula penalidade por padrão (AJUSTADO - menos agressivo)
        for pattern, data in penalties["patterns"].items():
            # Exige pelo menos 4 losses com mesmo padrão antes de penalizar
            if data["count"] >= max(4, MIN_LOSSES_TO_LEARN):
                # Soma penalidades dos problemas identificados
                total_penalty = 0.0
                for prob in set(data["problems"]):  # unique problems
                    total_penalty += PENALTY_WEIGHTS.get(prob, 0.03)
                # Escala a penalidade com a quantidade de losses (mais losses = mais certeza)
                scale_factor = min(1.0, data["count"] / 10.0)  # Escala até 10 losses
                data["penalty"] = min(0.35, total_penalty * scale_factor)  # Máximo 35%
        
        # Calcula penalidade por contexto
        for ctx, data in penalties["contexts"].items():
            if data["count"] >= MIN_LOSSES_TO_LEARN:
                freq = data["count"] / total_losses
                data["penalty"] = min(0.30, freq * 0.5)  # Máximo 30%
        
        self.cache["penalties"] = penalties
        
        # Log resumo
        if penalties["assets"]:
            worst_assets = sorted(penalties["assets"].items(), 
                                key=lambda x: x[1]["count"], reverse=True)[:3]
            logger.info(f"📊 Ativos problemáticos: {[(a, d['count']) for a, d in worst_assets]}")
    
    def _auto_adjust_filters(self):
        """Ajusta filtros automaticamente baseado nos problemas mais comuns"""
        stats = self.cache.get("statistics", {})
        top_problems = stats.get("top_problems", [])
        
        for problem_data in top_problems:
            problem = problem_data.get("problem", "")
            count = problem_data.get("count", 0)
            
            # Se um problema é muito frequente, aperta o filtro
            if count >= MIN_LOSSES_TO_LEARN:
                if problem == "consolidacao":
                    # Aumenta a sensibilidade à consolidação
                    self.learned_filters["MAX_CONSOLIDATION_STD"] = max(
                        0.0005, 
                        self.learned_filters["MAX_CONSOLIDATION_STD"] * 0.9
                    )
                    logger.info(f"⚙️ Filtro ajustado: MAX_CONSOLIDATION_STD = {self.learned_filters['MAX_CONSOLIDATION_STD']:.6f}")
                
                elif problem == "desalinhamento":
                    # Exige mais alinhamento
                    self.learned_filters["MIN_ALIGNMENT_RATIO"] = min(
                        0.70,
                        self.learned_filters["MIN_ALIGNMENT_RATIO"] + 0.05
                    )
                    logger.info(f"⚙️ Filtro ajustado: MIN_ALIGNMENT_RATIO = {self.learned_filters['MIN_ALIGNMENT_RATIO']:.2f}")
                
                elif problem == "entrada_fraca":
                    # Exige corpo mais forte
                    self.learned_filters["MIN_BODY_RATIO"] = min(
                        0.70,
                        self.learned_filters["MIN_BODY_RATIO"] + 0.05
                    )
                    logger.info(f"⚙️ Filtro ajustado: MIN_BODY_RATIO = {self.learned_filters['MIN_BODY_RATIO']:.2f}")
                
                elif problem == "sr_forte":
                    # Exige mais distância de S/R
                    self.learned_filters["SR_MIN_DISTANCE_PERCENT"] = min(
                        0.005,
                        self.learned_filters["SR_MIN_DISTANCE_PERCENT"] * 1.2
                    )
                    logger.info(f"⚙️ Filtro ajustado: SR_MIN_DISTANCE_PERCENT = {self.learned_filters['SR_MIN_DISTANCE_PERCENT']:.4f}")
    
    def get_penalty_for_setup(self, ativo: str, direction: str, 
                              market_context: Dict = None, 
                              entry_quality: Dict = None) -> Tuple[float, List[str]]:
        """
        Retorna penalidade e motivos para um setup específico
        Quanto maior a penalidade, menos confiável o setup
        """
        # Atualiza aprendizado se necessário
        self.refresh_learning()
        
        total_penalty = 0.0
        reasons = []
        
        penalties = self.cache.get("penalties", {})
        
        # 1. Penalidade por ativo
        asset_penalty = penalties.get("assets", {}).get(ativo, {}).get("penalty", 0)
        if asset_penalty > 0:
            total_penalty += asset_penalty
            reasons.append(f"ativo_ruim({asset_penalty:.0%})")
        
        # 2. Penalidade por padrão (ativo + direção)
        pattern_key = f"{ativo}_{direction}"
        pattern_data = penalties.get("patterns", {}).get(pattern_key, {})
        pattern_penalty = pattern_data.get("penalty", 0)
        if pattern_penalty > 0:
            total_penalty += pattern_penalty
            problems = list(set(pattern_data.get("problems", [])))[:3]
            reasons.append(f"padrao_ruim({pattern_penalty:.0%}:{','.join(problems)})")
        
        # 3. Penalidade por contexto de mercado atual
        if market_context:
            trend = market_context.get("trend", "neutral")
            volatility = market_context.get("volatility", "medium")
            ctx_key = f"{trend}_{volatility}"
            ctx_penalty = penalties.get("contexts", {}).get(ctx_key, {}).get("penalty", 0)
            if ctx_penalty > 0:
                total_penalty += ctx_penalty
                reasons.append(f"contexto_ruim({ctx_penalty:.0%})")
            
            # Verifica consolidação aprendida
            if market_context.get("is_consolidating"):
                total_penalty += PENALTY_WEIGHTS["consolidacao"]
                reasons.append("aprendido:consolidacao")
            
            # Verifica proximidade de S/R
            near_sr = market_context.get("near_resistance") or market_context.get("near_support")
            if near_sr:
                if direction == "CALL" and market_context.get("near_resistance"):
                    total_penalty += PENALTY_WEIGHTS["sr_forte"]
                    reasons.append("aprendido:call_em_resistencia")
                elif direction == "PUT" and market_context.get("near_support"):
                    total_penalty += PENALTY_WEIGHTS["sr_forte"]
                    reasons.append("aprendido:put_em_suporte")
        
        # 4. Penalidade por qualidade de entrada
        if entry_quality:
            alignment = entry_quality.get("alignment_ratio", 1.0)
            if alignment < self.learned_filters["MIN_ALIGNMENT_RATIO"]:
                total_penalty += PENALTY_WEIGHTS["desalinhamento"]
                reasons.append(f"aprendido:alinhamento_baixo({alignment:.0%})")
            
            if entry_quality.get("momentum_direction") == "wrong":
                total_penalty += PENALTY_WEIGHTS["momentum_errado"]
                reasons.append("aprendido:momentum_errado")
            
            if entry_quality.get("entry_quality") == "weak":
                total_penalty += PENALTY_WEIGHTS["entrada_fraca"]
                reasons.append("aprendido:entrada_fraca")
        
        # RIGOROSO: Penalidade maxima de 45% para bloquear setups ruins
        # Com poucos dados ainda aplica penalidade moderada
        max_penalty = 0.45
        total_losses = len(self.cache.get("losses", []))
        if total_losses < 5:
            max_penalty = 0.35  # Com poucos dados, penalidade de 35%

        total_penalty = min(max_penalty, total_penalty)
        
        return total_penalty, reasons
    
    def should_block_trade(self, ativo: str, direction: str, 
                          score: float, conf: float,
                          market_context: Dict = None,
                          entry_quality: Dict = None) -> Tuple[bool, str]:
        """
        Decide se deve bloquear uma operação baseado no aprendizado
        Retorna (should_block, reason)
        """
        penalty, reasons = self.get_penalty_for_setup(
            ativo, direction, market_context, entry_quality
        )
        
        # Ajusta score e confiança baseado na penalidade
        adjusted_score = score * (1 - penalty)
        adjusted_conf = conf * (1 - penalty)
        
        # RIGOROSO: Criterios de bloqueio para 80%+ win rate
        MIN_ADJUSTED_SCORE = 0.35  # Score ajustado minimo de 35%
        MIN_ADJUSTED_CONF = 0.40   # Confianca ajustada minima de 40%
        MAX_PENALTY_HARD = 0.50    # Bloqueia se penalidade > 50%

        # Mesmo com poucos dados, aplica bloqueio rigoroso
        total_losses = len(self.cache.get("losses", []))
        if total_losses < 5:
            # Com poucos dados, ainda bloqueia se penalty > 60%
            MAX_PENALTY_HARD = 0.60
        
        if penalty >= MAX_PENALTY_HARD:
            return True, f"APRENDIDO_BLOCK: penalty={penalty:.0%} | {'; '.join(reasons)}"
        
        if adjusted_score < MIN_ADJUSTED_SCORE:
            return True, f"APRENDIDO_LOW_SCORE: score={score:.2f} -> {adjusted_score:.2f} após penalty={penalty:.0%}"
        
        if adjusted_conf < MIN_ADJUSTED_CONF:
            return True, f"APRENDIDO_LOW_CONF: conf={conf:.2f} -> {adjusted_conf:.2f} após penalty={penalty:.0%}"
        
        return False, ""
    
    def get_adjusted_probability(self, ativo: str, direction: str, 
                                 base_prob: float,
                                 market_context: Dict = None,
                                 entry_quality: Dict = None) -> float:
        """Retorna probabilidade ajustada baseada no aprendizado"""
        penalty, _ = self.get_penalty_for_setup(
            ativo, direction, market_context, entry_quality
        )
        
        # Reduz probabilidade baseado na penalidade
        adjusted_prob = base_prob * (1 - penalty)
        
        return max(0.10, adjusted_prob)  # Mínimo 10%
    
    def get_learning_summary(self) -> str:
        """Retorna resumo do aprendizado atual"""
        stats = self.cache.get("statistics", {})
        penalties = self.cache.get("penalties", {})
        
        summary = "\n📚 RESUMO DO APRENDIZADO:\n"
        summary += "=" * 40 + "\n"
        
        total_losses = stats.get("total_losses", 0)
        summary += f"📊 Total de losses analisados: {total_losses}\n"
        summary += f"💰 Valor total perdido: ${stats.get('total_stake_lost', 0):.2f}\n"
        
        # Top problemas
        top_problems = stats.get("top_problems", [])
        if top_problems:
            summary += "\n🔴 Problemas mais comuns:\n"
            for p in top_problems[:5]:
                summary += f"   - {p['problem']}: {p['count']} vezes\n"
        
        # Ativos problemáticos
        assets_penalties = penalties.get("assets", {})
        if assets_penalties:
            worst = sorted(assets_penalties.items(), 
                          key=lambda x: x[1]["count"], reverse=True)[:5]
            summary += "\n⚠️ Ativos com mais losses:\n"
            for a, d in worst:
                summary += f"   - {a}: {d['count']} losses (penalty {d['penalty']:.0%})\n"
        
        # Filtros aprendidos
        summary += "\n⚙️ Filtros ajustados:\n"
        for k, v in self.learned_filters.items():
            summary += f"   - {k}: {v:.4f}\n"
        
        return summary


# Instância global
_global_learning = None

def get_ai_learning(backend_url: str = None) -> AILearning:
    """Retorna instância global do sistema de aprendizado"""
    global _global_learning
    if _global_learning is None:
        _global_learning = AILearning(backend_url)
    return _global_learning
