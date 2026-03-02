# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  IA AUTONOMOUS BRAIN — Sistema de Aprendizado Autônomo
═══════════════════════════════════════════════════════════════════════

Inspirado em robôs humanóides que aprendem sozinhos (como os robôs chineses):
A IA evolui SOZINHA a cada trade, sem intervenção humana.

CAPACIDADES:
  1. AUTO-CALIBRAÇÃO DE PARÂMETROS
     - Ajusta threshold, learning rate, penalidades automaticamente
     - Se WR está baixo → fica mais seletiva. Se alto → mais agressiva
     - Aprende o ritmo ideal para cada condição de mercado

  2. MEMÓRIA CONTEXTUAL
     - Lembra quais horários ganham mais (sessão Londres, NY, Ásia)
     - Lembra quais ativos são mais lucrativos
     - Lembra quais combinações de features ganharam/perderam

  3. AUTO-DISCOVERY DE PADRÕES
     - Descobre NOVOS combos de features que funcionam
     - Remove combos que param de funcionar
     - Evolui padrões sem programação humana

  4. META-LEARNING
     - Aprende COMO aprender melhor
     - Ajusta learning rate baseado na estabilidade
     - Sabe quando confiar mais/menos no modelo

  5. ADAPTAÇÃO EM TEMPO REAL
     - Detecta mudança de regime do mercado
     - Ajusta thresholds automaticamente
     - Expiração inteligente auto-otimizada

Autor: WS Trader AI — Sistema autônomo gerado por IA
"""

import time
import json
import os
import math
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

log = logging.getLogger("WS_AUTONOMOUS")

# ════════════════════════════════════════════════════════════════
# ARQUIVO DE ESTADO PERSISTENTE — SEPARADO POR BROKER
# Cada corretora tem ativos OTC com comportamento diferente,
# então a memória da IA deve ser isolada por broker.
# ════════════════════════════════════════════════════════════════
_broker_id = os.getenv("BROKER_TYPE", "bullex").strip().lower().replace("iq_option", "iq")
AUTONOMOUS_STATE_FILE = os.path.join(
    os.path.expanduser("~"), ".wstrader", f"autonomous_brain_state_{_broker_id}.json"
)

# ════════════════════════════════════════════════════════════════
# 1. MEMÓRIA CONTEXTUAL — Aprende com cada contexto
# ════════════════════════════════════════════════════════════════

class ContextMemory:
    """
    Memória que aprende padrões por contexto:
    - Horário (sessão de mercado)
    - Ativo específico
    - Dia da semana
    - Condições combinadas
    
    Usa EMA (Exponential Moving Average) para dar mais peso
    aos resultados recentes — como memória humana.
    """

    def __init__(self, ema_alpha: float = 0.15):
        self.alpha = ema_alpha
        # {context_key: {"wr": float, "count": int, "streak": int, "last_5": []}}
        self.contexts: Dict[str, Dict] = {}

    def _get_or_create(self, key: str) -> Dict:
        if key not in self.contexts:
            self.contexts[key] = {
                "wr": 0.50,         # WR começa neutro
                "count": 0,
                "streak": 0,        # Streak atual (+ = wins, - = losses)
                "best_streak": 0,
                "worst_streak": 0,
                "last_results": [],  # Últimos 20 resultados
                "total_wins": 0,
                "total_trades": 0,
            }
        return self.contexts[key]

    def update(self, key: str, win: bool):
        """Atualiza estatísticas de um contexto."""
        ctx = self._get_or_create(key)
        ctx["count"] += 1
        ctx["total_trades"] += 1
        if win:
            ctx["total_wins"] += 1

        # EMA do WR
        val = 1.0 if win else 0.0
        ctx["wr"] = ctx["wr"] * (1 - self.alpha) + val * self.alpha

        # Streak tracking
        if win:
            ctx["streak"] = max(0, ctx["streak"]) + 1
            ctx["best_streak"] = max(ctx["best_streak"], ctx["streak"])
        else:
            ctx["streak"] = min(0, ctx["streak"]) - 1
            ctx["worst_streak"] = min(ctx["worst_streak"], ctx["streak"])

        # Janela deslizante dos últimos 20
        ctx["last_results"].append(1 if win else 0)
        if len(ctx["last_results"]) > 20:
            ctx["last_results"] = ctx["last_results"][-20:]

    def get_wr(self, key: str) -> float:
        """Retorna WR de um contexto (0.5 se desconhecido)."""
        ctx = self.contexts.get(key)
        if ctx is None or ctx["count"] < 3:
            return 0.50
        return ctx["wr"]

    def get_confidence(self, key: str) -> float:
        """Retorna confiança baseada no número de trades (0 a 1)."""
        ctx = self.contexts.get(key)
        if ctx is None:
            return 0.0
        n = ctx["count"]
        return min(1.0, n / 30.0)  # Confiança máxima após 30 trades

    def get_recent_wr(self, key: str) -> float:
        """WR dos últimos N trades de um contexto."""
        ctx = self.contexts.get(key)
        if ctx is None:
            return 0.50
        results = ctx.get("last_results", [])
        if len(results) < 3:
            return 0.50
        return sum(results) / len(results)

    def is_hot(self, key: str) -> bool:
        """Está em boa sequência?"""
        ctx = self.contexts.get(key)
        if ctx is None:
            return False
        return ctx.get("streak", 0) >= 3

    def is_cold(self, key: str) -> bool:
        """Está em má sequência?"""
        ctx = self.contexts.get(key)
        if ctx is None:
            return False
        return ctx.get("streak", 0) <= -3

    def get_top_contexts(self, n: int = 10) -> List[Tuple[str, float, int]]:
        """Retorna os N melhores contextos (key, wr, count)."""
        items = []
        for key, ctx in self.contexts.items():
            if ctx["count"] >= 5:
                items.append((key, ctx["wr"], ctx["count"]))
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:n]

    def get_worst_contexts(self, n: int = 10) -> List[Tuple[str, float, int]]:
        """Retorna os N piores contextos."""
        items = []
        for key, ctx in self.contexts.items():
            if ctx["count"] >= 5:
                items.append((key, ctx["wr"], ctx["count"]))
        items.sort(key=lambda x: x[1])
        return items[:n]

    def get_state(self) -> Dict:
        return {"contexts": self.contexts, "alpha": self.alpha}

    def load_state(self, d: Dict):
        self.contexts = d.get("contexts", {})
        self.alpha = d.get("alpha", self.alpha)


# ════════════════════════════════════════════════════════════════
# 2. AUTO-DISCOVERY — Descobre padrões novos automaticamente
# ════════════════════════════════════════════════════════════════

class PatternDiscovery:
    """
    Descobre automaticamente combinações de features que ganham ou perdem.
    Funciona como um cientista: testa hipóteses e valida com dados.
    
    Exemplo de padrão descoberto:
      "candle=doji+approach=exhaustion+hour=10" → WR 72% (n=15)
    """

    def __init__(self, min_samples: int = 8, ema_alpha: float = 0.12):
        self.min_samples = min_samples
        self.alpha = ema_alpha
        # {combo_key: {"wr": float, "count": int, "discovered_at": timestamp}}
        self.combos: Dict[str, Dict] = {}
        self.max_combos = 500  # Limite de memória

    def record(self, features: Dict[str, Any], win: bool):
        """
        Registra resultado para TODAS as combinações de 2 e 3 features.
        Descobre quais combos têm edge positivo ou negativo.
        """
        # Extrair features categóricas relevantes
        keys_of_interest = [
            "candle", "structure", "zone_fresh", "approach",
            "vol_ctx", "dist_cat", "rejection", "pre_pattern",
            "momentum", "bodies"
        ]
        feat_pairs = []
        for k in keys_of_interest:
            v = features.get(k)
            if v is not None:
                feat_pairs.append(f"{k}={v}")

        # Adicionar contexto temporal
        hour = datetime.now().hour
        session = self._get_session(hour)
        feat_pairs.append(f"session={session}")

        # Combinações de 2 features
        combos_to_track = []
        for i in range(len(feat_pairs)):
            for j in range(i + 1, len(feat_pairs)):
                key = f"{feat_pairs[i]}+{feat_pairs[j]}"
                combos_to_track.append(key)

        # Combinações de 3 features (as mais frequentes apenas)
        if len(feat_pairs) >= 3:
            # Selecionar apenas as 3 features mais importantes
            top_feats = feat_pairs[:6]  # Primeiras 6 (candle, struct, fresh, approach, vol, dist)
            for i in range(len(top_feats)):
                for j in range(i + 1, len(top_feats)):
                    for k in range(j + 1, len(top_feats)):
                        key = f"{top_feats[i]}+{top_feats[j]}+{top_feats[k]}"
                        combos_to_track.append(key)

        val = 1.0 if win else 0.0
        for key in combos_to_track:
            if key not in self.combos:
                if len(self.combos) >= self.max_combos:
                    self._prune()
                self.combos[key] = {
                    "wr": 0.50,
                    "count": 0,
                    "discovered_at": time.time(),
                }
            combo = self.combos[key]
            combo["count"] += 1
            combo["wr"] = combo["wr"] * (1 - self.alpha) + val * self.alpha

    def _get_session(self, hour: int) -> str:
        """Identifica sessão de mercado pelo horário."""
        if 3 <= hour < 9:
            return "asia"
        elif 9 <= hour < 13:
            return "london"
        elif 13 <= hour < 18:
            return "ny"
        else:
            return "off_hours"

    def _prune(self):
        """Remove combos com poucas amostras e muito antigos."""
        items = sorted(
            self.combos.items(),
            key=lambda x: x[1]["count"]
        )
        # Remove os 20% com menos amostras
        cut = max(1, len(items) // 5)
        for key, _ in items[:cut]:
            del self.combos[key]

    def get_combo_edge(self, features: Dict[str, Any]) -> float:
        """
        Retorna ajuste de probabilidade baseado nos padrões descobertos.
        Positivo = padrão bom descoberto, Negativo = padrão ruim.
        """
        keys_of_interest = [
            "candle", "structure", "zone_fresh", "approach",
            "vol_ctx", "dist_cat", "rejection", "pre_pattern",
            "momentum", "bodies"
        ]
        feat_pairs = []
        for k in keys_of_interest:
            v = features.get(k)
            if v is not None:
                feat_pairs.append(f"{k}={v}")

        hour = datetime.now().hour
        session = self._get_session(hour)
        feat_pairs.append(f"session={session}")

        total_adj = 0.0
        n_combos = 0

        # Verificar combos de 2
        for i in range(len(feat_pairs)):
            for j in range(i + 1, len(feat_pairs)):
                key = f"{feat_pairs[i]}+{feat_pairs[j]}"
                combo = self.combos.get(key)
                if combo and combo["count"] >= self.min_samples:
                    edge = combo["wr"] - 0.50  # Desvio em relação ao neutro
                    confidence = min(1.0, combo["count"] / 30.0)
                    total_adj += edge * confidence * 0.03  # Contribuição sutil
                    n_combos += 1

        return round(total_adj, 4)

    def get_discovered_patterns(self, min_count: int = 10) -> List[Tuple[str, float, int]]:
        """Retorna padrões descobertos com edge significativo."""
        patterns = []
        for key, combo in self.combos.items():
            if combo["count"] >= min_count:
                edge = combo["wr"] - 0.50
                if abs(edge) >= 0.05:  # Edge mínimo de 5%
                    patterns.append((key, combo["wr"], combo["count"]))
        patterns.sort(key=lambda x: abs(x[1] - 0.50), reverse=True)
        return patterns

    def get_state(self) -> Dict:
        return {"combos": self.combos}

    def load_state(self, d: Dict):
        self.combos = d.get("combos", {})


# ════════════════════════════════════════════════════════════════
# 3. META-LEARNING — A IA aprende COMO aprender
# ════════════════════════════════════════════════════════════════

class MetaLearner:
    """
    Ajusta os hiperparâmetros do aprendizado baseado na performance.
    
    Se o modelo está estável → learning rate baixo (não perturbar)
    Se o modelo está errando muito → learning rate alto (adaptar rápido)
    Se está em sequência boa → não mexer
    Se está em sequência ruim → aumentar adaptação
    """

    def __init__(self):
        self.base_lr = 0.02
        self.base_threshold = 0.48
        self.performance_window: List[int] = []  # Últimos 50 resultados
        self.threshold_history: List[float] = []
        self.adaptation_speed = 1.0  # Multiplicador de velocidade
        self.stability_score = 0.5   # 0 = instável, 1 = estável
        self.total_adjustments = 0

    def record(self, win: bool):
        """Registra resultado e recalcula meta-parâmetros."""
        self.performance_window.append(1 if win else 0)
        if len(self.performance_window) > 50:
            self.performance_window = self.performance_window[-50:]

        self._recalculate()

    def _recalculate(self):
        """Recalcula meta-parâmetros baseado na janela de performance."""
        if len(self.performance_window) < 10:
            return

        recent_wr = sum(self.performance_window[-10:]) / 10
        overall_wr = sum(self.performance_window) / len(self.performance_window)

        # Estabilidade: quanto os últimos 10 diferem do geral
        diff = abs(recent_wr - overall_wr)
        self.stability_score = max(0.0, min(1.0, 1.0 - diff * 3))

        # Velocidade de adaptação
        if self.stability_score > 0.7:
            # Modelo estável → adaptar devagar (não estragar o que funciona)
            self.adaptation_speed = 0.7
        elif self.stability_score < 0.3:
            # Modelo instável → adaptar rápido
            self.adaptation_speed = 1.5
        else:
            self.adaptation_speed = 1.0

        self.total_adjustments += 1

    def get_learning_rate(self) -> float:
        """Learning rate dinâmico baseado na estabilidade."""
        return self.base_lr * self.adaptation_speed

    def get_threshold_adjustment(self) -> float:
        """
        Ajuste do threshold baseado na performance recente.
        Se perdendo muito → sobe threshold (mais seletivo)
        Se ganhando muito → pode relaxar um pouco
        """
        if len(self.performance_window) < 15:
            return 0.0

        recent_wr = sum(self.performance_window[-15:]) / 15

        if recent_wr < 0.40:
            return +0.05   # Muito loss → bem mais seletivo
        elif recent_wr < 0.48:
            return +0.02   # Abaixo do breakeven → levemente mais seletivo
        elif recent_wr > 0.65:
            return -0.02   # Muito bom → pode relaxar levemente
        elif recent_wr > 0.58:
            return -0.01   # Bom → relaxa mínimo
        return 0.0

    def get_status(self) -> Dict:
        n = len(self.performance_window)
        wr = sum(self.performance_window) / max(1, n) * 100
        return {
            "stability": round(self.stability_score, 2),
            "adaptation_speed": round(self.adaptation_speed, 2),
            "current_lr": round(self.get_learning_rate(), 4),
            "threshold_adj": round(self.get_threshold_adjustment(), 3),
            "window_size": n,
            "window_wr": round(wr, 1),
        }

    def get_state(self) -> Dict:
        return {
            "performance_window": self.performance_window,
            "stability_score": self.stability_score,
            "adaptation_speed": self.adaptation_speed,
            "total_adjustments": self.total_adjustments,
        }

    def load_state(self, d: Dict):
        self.performance_window = d.get("performance_window", [])
        self.stability_score = d.get("stability_score", 0.5)
        self.adaptation_speed = d.get("adaptation_speed", 1.0)
        self.total_adjustments = d.get("total_adjustments", 0)
        self._recalculate()


# ════════════════════════════════════════════════════════════════
# 4. EXPIRATION OPTIMIZER — Auto-otimiza tempo de expiração
# ════════════════════════════════════════════════════════════════

class ExpirationOptimizer:
    """
    Aprende qual tempo de expiração funciona melhor para cada contexto.
    Similar ao Brain, mas focado APENAS na expiração.
    """

    def __init__(self):
        # {context_key: {exp_minutes: {"wr": float, "count": int}}}
        self.data: Dict[str, Dict[int, Dict]] = {}

    def record(self, context_key: str, exp_min: int, win: bool):
        """Registra resultado de uma expiração em um contexto."""
        if context_key not in self.data:
            self.data[context_key] = {}
        if exp_min not in self.data[context_key]:
            self.data[context_key][exp_min] = {"wr": 0.50, "count": 0}

        entry = self.data[context_key][exp_min]
        entry["count"] += 1
        val = 1.0 if win else 0.0
        alpha = 0.15
        entry["wr"] = entry["wr"] * (1 - alpha) + val * alpha

    def get_best_expiration(self, context_key: str, default: int = 3) -> int:
        """Retorna a melhor expiração para um contexto (mínimo 2 min)."""
        ctx_data = self.data.get(context_key, {})
        if not ctx_data:
            return max(2, default)

        best_exp = default
        best_wr = 0.0
        for exp_min, entry in ctx_data.items():
            if entry["count"] >= 8 and entry["wr"] > best_wr:
                best_wr = entry["wr"]
                best_exp = exp_min

        return max(2, best_exp)

    def get_state(self) -> Dict:
        return {"data": self.data}

    def load_state(self, d: Dict):
        self.data = d.get("data", {})


# ════════════════════════════════════════════════════════════════
# 5. AUTONOMOUS BRAIN — O cérebro principal que orquestra tudo
# ════════════════════════════════════════════════════════════════

class AutonomousBrain:
    """
    Cérebro autônomo que integra todos os subsistemas.
    
    FUNCIONA COMO UM ROBÔ QUE APRENDE SOZINHO:
    1. A cada trade → atualiza memória contextual
    2. A cada trade → descobre novos padrões
    3. A cada trade → meta-learning ajusta velocidade
    4. A cada decisão → combina todos os sinais inteligentemente
    
    NÃO substitui o ConfluenceBrain (ML model) — COMPLEMENTA.
    O ConfluenceBrain dá P(win). O AutonomousBrain refina com contexto.
    """

    def __init__(self):
        self.context_memory = ContextMemory()
        self.pattern_discovery = PatternDiscovery()
        self.meta_learner = MetaLearner()
        self.exp_optimizer = ExpirationOptimizer()

        # Estatísticas globais
        self.total_trades = 0
        self.total_wins = 0
        self.session_trades = 0
        self.session_wins = 0
        self.session_start = time.time()

        # Insights descobertos (log)
        self.insights: List[str] = []

        # Carregar estado anterior
        self._load_state()

        wr = self.total_wins / max(1, self.total_trades) * 100
        log.info(
            f"[AUTONOMOUS] Cérebro autônomo iniciado | "
            f"trades={self.total_trades} WR={wr:.1f}% | "
            f"padrões conhecidos={len(self.pattern_discovery.combos)} | "
            f"contextos={len(self.context_memory.contexts)} | "
            f"estabilidade={self.meta_learner.stability_score:.2f}"
        )

    # ─────────────────────────────────────────────────
    # DECISÃO REFINADA — Ajuste de probabilidade
    # ─────────────────────────────────────────────────
    def refine_probability(self, base_prob: float, features: Dict,
                           ativo: str, setup: Dict) -> Tuple[float, str]:
        """
        Recebe P(win) do ConfluenceBrain e REFINA com contexto autônomo.
        
        Retorna:
            (adjusted_prob, detail_string)
        """
        adjustments = []
        total_adj = 0.0

        # 1. CONTEXTO DO ATIVO — Esse ativo está bom ou ruim?
        ativo_wr = self.context_memory.get_wr(f"ativo:{ativo}")
        ativo_conf = self.context_memory.get_confidence(f"ativo:{ativo}")
        if ativo_conf >= 0.3:
            ativo_edge = (ativo_wr - 0.50) * ativo_conf * 0.05
            total_adj += ativo_edge
            if abs(ativo_edge) >= 0.005:
                adjustments.append(f"ativo({ativo_wr:.0%})")

        # 2. CONTEXTO DO HORÁRIO — Essa sessão está boa?
        hour = datetime.now().hour
        session = self._get_session(hour)
        session_wr = self.context_memory.get_wr(f"session:{session}")
        session_conf = self.context_memory.get_confidence(f"session:{session}")
        if session_conf >= 0.3:
            session_edge = (session_wr - 0.50) * session_conf * 0.04
            total_adj += session_edge
            if abs(session_edge) >= 0.005:
                adjustments.append(f"session({session}:{session_wr:.0%})")

        # 3. CONTEXTO DIA DA SEMANA
        weekday = datetime.now().strftime("%A")
        day_wr = self.context_memory.get_wr(f"day:{weekday}")
        day_conf = self.context_memory.get_confidence(f"day:{weekday}")
        if day_conf >= 0.3:
            day_edge = (day_wr - 0.50) * day_conf * 0.03
            total_adj += day_edge
            if abs(day_edge) >= 0.005:
                adjustments.append(f"dia({weekday[:3]}:{day_wr:.0%})")

        # 4. PADRÕES DESCOBERTOS — Combos automáticos
        combo_edge = self.pattern_discovery.get_combo_edge(features)
        total_adj += combo_edge
        if abs(combo_edge) >= 0.005:
            adjustments.append(f"combos({combo_edge:+.1%})")

        # 5. META-THRESHOLD — Ajuste adaptativo
        meta_adj = self.meta_learner.get_threshold_adjustment()
        # Nota: meta_adj ajusta o THRESHOLD, não a probabilidade
        # Mas se está em cold streak severo, penaliza diretamente
        if meta_adj >= 0.04:
            total_adj -= 0.02  # Penalidade extra quando muito seletivo
            adjustments.append("meta_cold")

        # 6. SEQUÊNCIA DO ATIVO — Momentum detection
        if self.context_memory.is_hot(f"ativo:{ativo}"):
            total_adj += 0.01
            adjustments.append("hot_asset")
        elif self.context_memory.is_cold(f"ativo:{ativo}"):
            total_adj -= 0.02
            adjustments.append("cold_asset")

        # Limitar ajuste total
        total_adj = max(-0.10, min(0.10, total_adj))
        adjusted = max(0.0, min(1.0, base_prob + total_adj))

        detail = " | ".join(adjustments) if adjustments else "sem_ajuste"
        return adjusted, f"[AUTO:{total_adj:+.1%} {detail}]"

    # ─────────────────────────────────────────────────
    # EXPIRAÇÃO AUTÔNOMA — Aprende o melhor tempo
    # ─────────────────────────────────────────────────
    def get_smart_expiration(self, features: Dict, ativo: str,
                             setup: Dict, atr_val: float,
                             caller_heuristic: int = 0) -> int:
        """
        Calcula expiração inteligente AUTÔNOMA.
        Combina lógica heurística + aprendizado por contexto.
        """
        # Contexto baseado no ativo + sessão
        hour = datetime.now().hour
        session = self._get_session(hour)
        context_key = f"{ativo}:{session}"

        # Expiração aprendida
        learned_exp = self.exp_optimizer.get_best_expiration(context_key, default=0)

        # Usar heurística do caller (mais completa) se disponível
        heuristic_exp = caller_heuristic if caller_heuristic > 0 else self._heuristic_expiration(setup, atr_val)

        # Se já tem dados suficientes, combinar
        ctx_data = self.exp_optimizer.data.get(context_key, {})
        total_samples = sum(e.get("count", 0) for e in ctx_data.values())

        if total_samples >= 20 and learned_exp >= 2:
            # 50% peso aprendido, 50% heurístico (equilibrado)
            final = round(learned_exp * 0.5 + heuristic_exp * 0.5)
        else:
            final = heuristic_exp

        return max(2, min(5, final))

    def _heuristic_expiration(self, setup: Dict, atr_val: float) -> int:
        """Heurística base para expiração (sem aprendizado)."""
        base = 3.0
        dist_atr = float(setup.get("zone_distance_atr", 0.3))
        inside_zone = bool(setup.get("inside_zone", False))
        confirmed = bool(setup.get("confirmed_candle", False))
        score = float(setup.get("score", 0.5))
        momentum = abs(float(setup.get("momentum_total_move_atr", 0)))

        if inside_zone:
            base -= 0.5  # Dentro da zona → menos tempo (mas não exagerar)
        if dist_atr <= 0.30:
            base -= 0.5
        elif dist_atr >= 0.80:
            base += 1.0
        if confirmed:
            base -= 0.5
        if score >= 0.70:
            base -= 0.5
        elif score <= 0.45:
            base += 0.5
        if momentum >= 1.5:
            base -= 0.5
        elif momentum <= 0.3:
            base += 0.5

        return max(1, min(5, round(base)))

    def _get_session(self, hour: int) -> str:
        if 3 <= hour < 9:
            return "asia"
        elif 9 <= hour < 13:
            return "london"
        elif 13 <= hour < 18:
            return "ny"
        else:
            return "off_hours"

    # ─────────────────────────────────────────────────
    # APRENDER — Chamado após CADA trade
    # ─────────────────────────────────────────────────
    def learn(self, features: Dict, ativo: str, setup: Dict,
              win: bool, exp_min: int = 3):
        """
        MÉTODO PRINCIPAL: A IA aprende com o resultado do trade.
        
        Atualiza TODOS os subsistemas:
        1. Memória contextual (ativo, sessão, dia)
        2. Descoberta de padrões (combos de features)
        3. Meta-learner (velocidade de adaptação)
        4. Otimizador de expiração
        """
        self.total_trades += 1
        self.session_trades += 1
        if win:
            self.total_wins += 1
            self.session_wins += 1

        hour = datetime.now().hour
        session = self._get_session(hour)
        weekday = datetime.now().strftime("%A")

        # 1. Atualizar memória contextual
        self.context_memory.update(f"ativo:{ativo}", win)
        self.context_memory.update(f"session:{session}", win)
        self.context_memory.update(f"day:{weekday}", win)
        self.context_memory.update(f"hour:{hour}", win)

        # Context composto: ativo + sessão
        self.context_memory.update(f"{ativo}:{session}", win)

        # Context com direção
        direction = setup.get("dir", "CALL")
        self.context_memory.update(f"dir:{direction}", win)
        self.context_memory.update(f"{ativo}:{direction}", win)

        # 2. Atualizar descoberta de padrões
        if features:
            self.pattern_discovery.record(features, win)

        # 3. Meta-learning
        self.meta_learner.record(win)

        # 4. Expiração
        context_key = f"{ativo}:{session}"
        self.exp_optimizer.record(context_key, exp_min, win)

        # 5. Gerar insights periódicos
        if self.total_trades % 20 == 0:
            self._generate_insights()

        # 6. Log do aprendizado
        wr = self.total_wins / max(1, self.total_trades) * 100
        session_wr = self.session_wins / max(1, self.session_trades) * 100
        meta = self.meta_learner.get_status()
        log.info(
            f"[AUTONOMOUS] {'✅' if win else '❌'} {ativo} | "
            f"total={self.total_trades} WR={wr:.1f}% | "
            f"sessão={self.session_trades} WR={session_wr:.1f}% | "
            f"estab={meta['stability']:.2f} vel={meta['adaptation_speed']:.1f}x"
        )

        # 7. Salvar estado
        self._save_state()

    # ─────────────────────────────────────────────────
    # INSIGHTS AUTOMÁTICOS
    # ─────────────────────────────────────────────────
    def _generate_insights(self):
        """Gera insights baseado nos dados acumulados."""
        insights = []

        # Melhores contextos
        top = self.context_memory.get_top_contexts(5)
        for key, wr, count in top:
            if wr >= 0.60 and count >= 10:
                insights.append(f"🔥 {key} está com WR={wr:.0%} (n={count})")

        # Piores contextos
        worst = self.context_memory.get_worst_contexts(5)
        for key, wr, count in worst:
            if wr <= 0.40 and count >= 10:
                insights.append(f"⚠️ {key} está com WR={wr:.0%} (n={count})")

        # Padrões descobertos
        patterns = self.pattern_discovery.get_discovered_patterns(min_count=10)
        for key, wr, count in patterns[:3]:
            if wr >= 0.60:
                insights.append(f"🧠 Padrão BOM descoberto: {key} WR={wr:.0%} (n={count})")
            elif wr <= 0.40:
                insights.append(f"🧠 Padrão RUIM descoberto: {key} WR={wr:.0%} (n={count})")

        # Meta-learning status
        meta = self.meta_learner.get_status()
        if meta["stability"] < 0.3:
            insights.append(f"📊 Mercado INSTÁVEL — adaptação rápida ativa ({meta['adaptation_speed']:.1f}x)")
        elif meta["stability"] > 0.7:
            insights.append(f"📊 Mercado ESTÁVEL — modo conservador ({meta['adaptation_speed']:.1f}x)")

        for insight in insights:
            log.info(f"[INSIGHT] {insight}")

        self.insights = insights

    # ─────────────────────────────────────────────────
    # RELATÓRIO COMPLETO
    # ─────────────────────────────────────────────────
    def get_report(self) -> str:
        """Relatório completo do cérebro autônomo."""
        lines = []
        lines.append("=" * 60)
        lines.append("  IA AUTONOMOUS BRAIN — Relatório")
        lines.append("=" * 60)

        wr = self.total_wins / max(1, self.total_trades) * 100
        session_wr = self.session_wins / max(1, self.session_trades) * 100
        lines.append(f"\n  Total: {self.total_trades} trades | WR={wr:.1f}%")
        lines.append(f"  Sessão: {self.session_trades} trades | WR={session_wr:.1f}%")

        meta = self.meta_learner.get_status()
        lines.append(f"\n  Meta-Learning:")
        lines.append(f"    Estabilidade: {meta['stability']:.2f}")
        lines.append(f"    Velocidade: {meta['adaptation_speed']:.1f}x")
        lines.append(f"    LR dinâmico: {meta['current_lr']:.4f}")
        lines.append(f"    Threshold adj: {meta['threshold_adj']:+.3f}")

        lines.append(f"\n  Padrões descobertos: {len(self.pattern_discovery.combos)}")
        good_patterns = [p for p in self.pattern_discovery.get_discovered_patterns()
                         if p[1] >= 0.55]
        bad_patterns = [p for p in self.pattern_discovery.get_discovered_patterns()
                        if p[1] <= 0.45]
        lines.append(f"    Bons (WR>55%): {len(good_patterns)}")
        lines.append(f"    Ruins (WR<45%): {len(bad_patterns)}")

        top_ctx = self.context_memory.get_top_contexts(5)
        if top_ctx:
            lines.append("\n  Melhores contextos:")
            for key, wr_ctx, count in top_ctx:
                lines.append(f"    ✅ {key}: WR={wr_ctx:.0%} (n={count})")

        worst_ctx = self.context_memory.get_worst_contexts(3)
        if worst_ctx:
            lines.append("\n  Piores contextos:")
            for key, wr_ctx, count in worst_ctx:
                lines.append(f"    ❌ {key}: WR={wr_ctx:.0%} (n={count})")

        if self.insights:
            lines.append("\n  Último insight:")
            for ins in self.insights[-3:]:
                lines.append(f"    {ins}")

        lines.append("")
        return "\n".join(lines)

    def get_status(self) -> Dict:
        """Retorna status resumido como dict."""
        wr = self.total_wins / max(1, self.total_trades) * 100
        meta = self.meta_learner.get_status()
        return {
            "total_trades": self.total_trades,
            "total_wins": self.total_wins,
            "win_rate": round(wr, 1),
            "session_trades": self.session_trades,
            "session_wr": round(self.session_wins / max(1, self.session_trades) * 100, 1),
            "stability": meta["stability"],
            "adaptation_speed": meta["adaptation_speed"],
            "patterns_discovered": len(self.pattern_discovery.combos),
            "contexts_learned": len(self.context_memory.contexts),
        }

    # ─────────────────────────────────────────────────
    # PERSISTÊNCIA
    # ─────────────────────────────────────────────────
    def _save_state(self):
        try:
            state_dir = os.path.dirname(AUTONOMOUS_STATE_FILE)
            os.makedirs(state_dir, exist_ok=True)
            state = {
                "version": 1,
                "total_trades": self.total_trades,
                "total_wins": self.total_wins,
                "context_memory": self.context_memory.get_state(),
                "pattern_discovery": self.pattern_discovery.get_state(),
                "meta_learner": self.meta_learner.get_state(),
                "exp_optimizer": self.exp_optimizer.get_state(),
                "saved_at": time.time(),
            }
            with open(AUTONOMOUS_STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log.debug(f"[AUTONOMOUS] Erro ao salvar: {e}")

    def _load_state(self) -> bool:
        try:
            if not os.path.exists(AUTONOMOUS_STATE_FILE):
                return False
            with open(AUTONOMOUS_STATE_FILE, "r") as f:
                state = json.load(f)
            if state.get("version") != 1:
                return False

            self.total_trades = int(state.get("total_trades", 0))
            self.total_wins = int(state.get("total_wins", 0))
            self.context_memory.load_state(state.get("context_memory", {}))
            self.pattern_discovery.load_state(state.get("pattern_discovery", {}))
            self.meta_learner.load_state(state.get("meta_learner", {}))
            self.exp_optimizer.load_state(state.get("exp_optimizer", {}))
            return True
        except Exception as e:
            log.debug(f"[AUTONOMOUS] Erro ao carregar: {e}")
            return False


# ════════════════════════════════════════════════════════════════
# SINGLETON
# ════════════════════════════════════════════════════════════════

_autonomous_instance: Optional[AutonomousBrain] = None


def get_autonomous_brain() -> AutonomousBrain:
    """Retorna instância singleton do AutonomousBrain."""
    global _autonomous_instance
    if _autonomous_instance is None:
        _autonomous_instance = AutonomousBrain()
    return _autonomous_instance
