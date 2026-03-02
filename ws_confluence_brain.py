# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  WS CONFLUENCE BRAIN v2 — REAL Machine Learning Trading AI
═══════════════════════════════════════════════════════════════════════

DIFERENTE da v1 (tabelona de pontos hardcoded), esta versão usa:

1. MODELO ML REAL (Online Logistic Regression)
   - 42 features one-hot + 18 interações + 4 numéricas = 64 features
   - Aprende pesos a partir dos DADOS, não de regras manuais
   - Dá P(win) REAL calibrado (não score arbitrário)

2. ONLINE LEARNING
   - Atualiza pesos após CADA trade via SGD
   - lr=0.02 normal, lr=0.04 após LOSS (aprende mais com erros)
   - L2 regularization previne overfitting

3. WARM START
   - Pré-treinado com dados de 10.668 setups reais
   - 22 combos estatisticamente comprovados + 500 amostras aleatórias
   - Modelo funciona desde o primeiro trade

4. REGIME DETECTION
   - Monitora WR dos últimos 30 trades
   - Regime "cold" → threshold sobe (mais seletivo)
   - Regime "hot" → threshold pode relaxar levemente

5. FEATURE IMPORTANCE DINÂMICA
   - Rastreia WR por feature via EMA (Exponential Moving Average)
   - Mostra relatório de quais features estão performando
   - Alerta quando uma feature está com WR < 45%

COMO O MODELO PENSA:
  classify_features(df, setup, atr) → 11 features categóricas + 4 numéricas
  encode(features)                 → vetor de 64 dimensões
  model.predict_proba(X)           → P(win) entre 0 e 1
  + ajuste RT (posição real-time)  → P(win) ajustado
  → ENTER se P >= threshold, WAIT se próximo, SKIP se baixo

Autor: Sistema gerado por ML treinado em dados reais de mercado OTC.
"""

import time
import json
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from collections import defaultdict

log = logging.getLogger("WS_BRAIN")

# ── IA Autônoma (auto-learning) ──
try:
    from ia_autonomous_brain import get_autonomous_brain as _get_auto_brain
    _AUTONOMOUS_AVAILABLE = True
except Exception:
    _AUTONOMOUS_AVAILABLE = False

# ── Padrões de Estrutura (13 padrões vencedores do estudo empírico) ──
try:
    from ws_structure_patterns import detect_structure_pattern, pattern_summary, get_hour_bonus
    _STRUCTURE_PATTERNS_AVAILABLE = True
except Exception:
    _STRUCTURE_PATTERNS_AVAILABLE = False

# ── Mapeamento Estrutural (zigzag + regiões) ──
try:
    from ws_structure_map import detect_structure_touch, structure_map_summary, check_alignment
    _STRUCTURE_MAP_AVAILABLE = True
except Exception:
    _STRUCTURE_MAP_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
# 1. CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════

BRAIN_VERSION = 5  # v5: drift limit 15%->6%, calibration floor, warm start preservado

# Probabilidade mínima para entrar
# Com payout ~85%, breakeven = ~54%. Threshold 55% garante edge real.
# MODO ESTRITO: só entra com certeza alta (55%+)
import os as _os
_IA_MODE = _os.getenv("WS_IA_MODE", "learning").strip().lower()
BASE_PROB_THRESHOLD = 0.55  # SEMPRE 55%+ — sem atalhos

# Acima disso = confiança ALTA
HIGH_CONFIDENCE_PROB = 0.65

# Learning rates
LEARNING_RATE = 0.015             # Normal (online, por trade)
LEARNING_RATE_AFTER_LOSS = 0.025  # Levemente maior após LOSS
L2_REGULARIZATION = 0.0005        # Menos regularização = pesos maiores = probabilidades mais extremas
WARM_START_LR = 0.10              # Mais agressivo para warm start (fundação)
WARM_START_EPOCHS = 8             # Mais epochs = pesos mais definidos

# Regime detection
REGIME_WINDOW = 30
REGIME_COLD_WR = 0.45
REGIME_HOT_WR = 0.62

# Warm start — máximo de amostras por combo (evita dominar o modelo)
WARM_MAX_SAMPLES_PER_COMBO = 100

# Arquivo de estado — SEPARADO POR BROKER
# Cada corretora tem ativos OTC com comportamento diferente,
# então o modelo ML deve ser isolado por broker.
_broker_id = os.getenv("BROKER_TYPE", "bullex").strip().lower().replace("iq_option", "iq")
BRAIN_STATE_FILE = os.path.join(
    os.path.expanduser("~"), ".wstrader", f"confluence_brain_state_{_broker_id}.json"
)

# ═══════════════════════════════════════════════════════════════
# 2. CODIFICAÇÃO DE FEATURES
# ═══════════════════════════════════════════════════════════════

# Schema: cada feature categórica → possíveis valores
FEATURE_SCHEMA = {
    "candle":        ["doji", "pin_bar", "engulfing", "strong_aligned", "strong_contra", "neutral"],
    "structure":     ["favor", "contra", "range"],
    "zone_fresh":    ["very_fresh", "fresh", "used"],
    "approach":      ["exhaustion", "impulse", "moderate", "gradual"],
    "vol_ctx":       ["low", "normal", "high"],
    "dist_cat":      ["tight", "close", "loose"],
    "rejection":     ["strong", "moderate", "weak"],
    "pre_pattern":   ["3_contra", "1_contra", "mixed"],
    "momentum":      ["strong_move", "moderate_move", "low_move"],
    "bodies":        ["big", "normal", "small"],
    "is_sweep":      [True, False],
    "channel":       ["at_edge", "mid", "none"],
    "wick_extreme":  ["strong_rejection", "moderate_rejection", "no_rejection"],
    "sr_cluster":    ["mega_zone", "double_zone", "single"],
}

# Pré-computar mapeamento de índice para one-hot encoding
_FEATURE_INDEX: Dict[tuple, int] = {}
_idx = 0
for _feat_name, _values in FEATURE_SCHEMA.items():
    for _val in _values:
        _FEATURE_INDEX[(_feat_name, _val)] = _idx
        _idx += 1
N_ONEHOT = _idx  # 36

# Features de interação — combos provados (e anti-combos) do estudo
INTERACTION_FEATURES = [
    # Positivos (edge > base 51.2%)
    {"candle": "doji", "approach": "exhaustion"},                              # 72.5%
    {"pre_pattern": "3_contra", "rejection": "strong", "structure": "favor"},  # 64.9%
    {"zone_fresh": "fresh", "vol_ctx": "low", "rejection": "strong"},          # 60.9%
    {"approach": "exhaustion", "bodies": "small"},                              # 60.7%
    {"candle": "doji", "vol_ctx": "low"},                                       # 60.3%
    {"zone_fresh": "fresh", "momentum": "low_move"},                            # 59.4%
    {"candle": "strong_contra", "vol_ctx": "low"},                              # 58.8%
    {"structure": "contra", "approach": "exhaustion"},                           # 58.7%
    {"candle": "doji", "pre_pattern": "3_contra"},                              # 58.7%
    {"structure": "favor", "vol_ctx": "low"},                                   # 58.5%
    {"zone_fresh": "very_fresh", "candle": "doji"},                             # 57.1%
    {"dist_cat": "close", "candle": "strong_contra"},                           # 57.0%
    {"structure": "contra", "bodies": "small"},                                 # 55.9%
    {"structure": "contra", "pre_pattern": "3_contra"},                         # 55.2%
    # Negativos — anti-padrões que DESTROEM edge
    {"approach": "exhaustion", "candle": "pin_bar"},                            # 42.1% !!
    {"approach": "exhaustion", "rejection": "strong"},                          # 40.0% !!
    {"approach": "gradual", "rejection": "strong"},                             # 45.7%
    {"is_sweep": True, "zone_fresh": "very_fresh"},                             # 48.4%
    # Canal + Wick: padrões de borda de canal com rejeição forte
    {"channel": "at_edge", "wick_extreme": "strong_rejection"},                 # estimado 62%
    {"channel": "at_edge", "rejection": "strong"},                              # estimado 60%
    {"channel": "at_edge", "wick_extreme": "strong_rejection", "structure": "favor"},  # estimado 65%
]
N_INTERACTIONS = len(INTERACTION_FEATURES)  # 21

# Features numéricas: body_ratio, wick_ratio, dist_atr, consec_before
N_NUMERIC = 4

# TOTAL
N_TOTAL_FEATURES = N_ONEHOT + N_INTERACTIONS + N_NUMERIC  # 42 + 21 + 4 = 67


# ═══════════════════════════════════════════════════════════════
# 3. ONLINE LOGISTIC REGRESSION — Modelo ML de verdade
# ═══════════════════════════════════════════════════════════════

class OnlineLogisticRegression:
    """
    Regressão logística com SGD online.
    Recebe vetor de 58 features → retorna P(win) calibrado.
    Atualiza pesos com cada novo exemplo (online learning).
    """

    def __init__(self, n_features: int, lr: float = LEARNING_RATE,
                 l2: float = L2_REGULARIZATION):
        self.n_features = n_features
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0
        self.lr = lr
        self.l2 = l2
        self.n_updates = 0

    def _sigmoid(self, z: float) -> float:
        """Sigmoid numericamente estável."""
        z = np.clip(z, -15, 15)
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-z))
        else:
            ez = np.exp(z)
            return float(ez / (1.0 + ez))

    def predict_proba(self, X: np.ndarray) -> float:
        """Retorna P(y=1|X) — probabilidade de WIN."""
        z = float(np.dot(self.w, X) + self.b)
        return self._sigmoid(z)

    def partial_fit(self, X: np.ndarray, y: int, lr_override: float = None):
        """
        Atualiza pesos com UM exemplo (SGD + L2).
        y=1 → WIN, y=0 → LOSS
        """
        lr = lr_override if lr_override is not None else self.lr
        p = self.predict_proba(X)
        error = y - p  # Gradiente da log-loss

        # SGD update com regularização L2
        self.w += lr * (error * X - self.l2 * self.w)
        self.b += lr * error
        self.n_updates += 1

    def get_state(self) -> dict:
        return {
            "w": self.w.tolist(),
            "b": float(self.b),
            "n_updates": self.n_updates,
        }

    def load_state(self, d: dict):
        w = d.get("w", [])
        if len(w) == self.n_features:
            self.w = np.array(w, dtype=np.float64)
            self.b = float(d.get("b", 0.0))
            self.n_updates = int(d.get("n_updates", 0))
        else:
            log.warning(f"[BRAIN] Weights incompatíveis ({len(w)} vs {self.n_features}), resetando")


# ═══════════════════════════════════════════════════════════════
# 4. TRACKER DE WIN RATE POR FEATURE
# ═══════════════════════════════════════════════════════════════

class FeatureWRTracker:
    """
    Rastreia WR empírico por valor de feature usando EMA.
    Usado para relatório e diagnóstico (não para predição direta).
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.ema_wr: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, features: Dict, win: bool):
        result = 1.0 if win else 0.0
        for feat_name in FEATURE_SCHEMA:
            val = features.get(feat_name)
            if val is None:
                continue
            key = f"{feat_name}_{val}"

            if key not in self.ema_wr:
                self.ema_wr[key] = 0.512
                self.counts[key] = 0

            self.counts[key] += 1
            n = self.counts[key]
            effective_alpha = max(self.alpha, 1.0 / n)
            self.ema_wr[key] = (1 - effective_alpha) * self.ema_wr[key] + effective_alpha * result

    def get_alerts(self, features: Dict) -> List[str]:
        """Retorna alertas para features com WR muito baixo."""
        alerts = []
        for feat_name in FEATURE_SCHEMA:
            val = features.get(feat_name)
            if val is None:
                continue
            key = f"{feat_name}_{val}"
            count = self.counts.get(key, 0)
            if count >= 8:
                wr = self.ema_wr.get(key, 0.512)
                if wr < 0.42:
                    alerts.append(f"{key}={wr:.0%}({count})")
        return alerts

    def get_state(self) -> dict:
        return {"ema_wr": dict(self.ema_wr), "counts": dict(self.counts)}

    def load_state(self, d: dict):
        self.ema_wr = d.get("ema_wr", {})
        self.counts = d.get("counts", {})


# ═══════════════════════════════════════════════════════════════
# 5. DETECTOR DE REGIME
# ═══════════════════════════════════════════════════════════════

class RegimeDetector:
    """
    Monitora performance recente para detectar mudança de regime.
    Ajusta o threshold de entrada quando o mercado muda.
    """

    def __init__(self, window: int = REGIME_WINDOW):
        self.window = window
        self.results: List[bool] = []

    def update(self, win: bool):
        self.results.append(win)
        if len(self.results) > self.window * 2:
            self.results = self.results[-self.window:]

    def get_recent_wr(self) -> float:
        if not self.results:
            return 0.512
        recent = self.results[-self.window:]
        return sum(recent) / len(recent)

    def get_regime(self) -> str:
        if len(self.results) < 8:
            return "unknown"
        wr = self.get_recent_wr()
        if wr >= REGIME_HOT_WR:
            return "hot"
        elif wr <= REGIME_COLD_WR:
            return "cold"
        return "normal"

    def get_threshold_adjustment(self) -> float:
        """
        Ajuste ao threshold base.
        +positivo = mais seletivo (mercado frio)
        -negativo = pode ser menos seletivo (mercado bom)
        """
        regime = self.get_regime()
        if regime == "cold":
            return +0.03   # 58% em vez de 55% (mais seletivo em mercado frio)
        elif regime == "hot":
            return -0.02   # 53% em mercado quente (ainda acima de breakeven)
        return 0.0

    def get_state(self) -> dict:
        return {"results": self.results[-self.window:]}

    def load_state(self, d: dict):
        self.results = d.get("results", [])


# ═══════════════════════════════════════════════════════════════
# 6. DADOS DE WARM START
# ═══════════════════════════════════════════════════════════════

# Resultados do estudo: (features_do_combo, WR_observado, n_amostras)
STUDY_COMBOS = [
    # Tier 1: Edge forte (WR >= 58%)
    ({"candle": "doji", "approach": "exhaustion"}, 0.725, 40),
    ({"pre_pattern": "3_contra", "rejection": "strong", "structure": "favor"}, 0.649, 37),
    ({"zone_fresh": "fresh", "vol_ctx": "low", "rejection": "strong"}, 0.609, 23),
    ({"approach": "exhaustion", "bodies": "small"}, 0.607, 28),
    ({"candle": "doji", "vol_ctx": "low"}, 0.603, 63),
    ({"zone_fresh": "fresh", "momentum": "low_move"}, 0.594, 101),
    ({"candle": "strong_contra", "vol_ctx": "low"}, 0.588, 68),
    ({"structure": "contra", "approach": "exhaustion"}, 0.587, 126),
    ({"candle": "doji", "pre_pattern": "3_contra"}, 0.587, 143),
    ({"structure": "favor", "vol_ctx": "low"}, 0.585, 130),
    # Tier 2: Edge moderado (54-58%)
    ({"zone_fresh": "very_fresh", "candle": "doji"}, 0.571, 240),
    ({"dist_cat": "close", "candle": "strong_contra"}, 0.570, 649),
    ({"pre_pattern": "1_contra", "bodies": "small"}, 0.569, 188),
    ({"candle": "engulfing", "bodies": "small"}, 0.561, 212),
    ({"structure": "contra", "bodies": "small"}, 0.559, 170),
    ({"structure": "contra", "pre_pattern": "3_contra"}, 0.552, 464),
    ({"zone_fresh": "very_fresh", "is_sweep": False}, 0.550, 1196),
    ({"structure": "contra", "is_sweep": False}, 0.547, 1822),
    # Anti-padrões (WR < 50%)
    ({"approach": "exhaustion", "candle": "pin_bar"}, 0.421, 38),
    ({"approach": "exhaustion", "rejection": "strong"}, 0.400, 30),
    ({"approach": "gradual", "rejection": "strong"}, 0.457, 490),
    ({"is_sweep": True, "zone_fresh": "very_fresh"}, 0.484, 1105),
    # Canal + Wick (estimativas iniciais — o modelo aprende online)
    ({"channel": "at_edge", "wick_extreme": "strong_rejection"}, 0.62, 50),
    ({"channel": "at_edge", "rejection": "strong"}, 0.60, 50),
    ({"channel": "at_edge", "wick_extreme": "strong_rejection", "structure": "favor"}, 0.65, 30),
    ({"channel": "mid", "wick_extreme": "no_rejection"}, 0.49, 100),
]

# Taxas base para gerar features aleatórias
FEATURE_BASE_RATES = {
    "candle": [("doji", 0.15), ("pin_bar", 0.10), ("engulfing", 0.15),
               ("strong_aligned", 0.15), ("strong_contra", 0.15), ("neutral", 0.30)],
    "structure": [("favor", 0.35), ("contra", 0.35), ("range", 0.30)],
    "zone_fresh": [("very_fresh", 0.40), ("fresh", 0.30), ("used", 0.30)],
    "approach": [("exhaustion", 0.12), ("impulse", 0.20), ("moderate", 0.25), ("gradual", 0.43)],
    "vol_ctx": [("low", 0.20), ("normal", 0.55), ("high", 0.25)],
    "dist_cat": [("tight", 0.30), ("close", 0.35), ("loose", 0.35)],
    "rejection": [("strong", 0.25), ("moderate", 0.40), ("weak", 0.35)],
    "pre_pattern": [("3_contra", 0.15), ("1_contra", 0.50), ("mixed", 0.35)],
    "momentum": [("strong_move", 0.30), ("moderate_move", 0.35), ("low_move", 0.35)],
    "bodies": [("big", 0.20), ("normal", 0.50), ("small", 0.30)],
    "is_sweep": [(True, 0.25), (False, 0.75)],
    "channel": [("at_edge", 0.20), ("mid", 0.30), ("none", 0.50)],
    "wick_extreme": [("strong_rejection", 0.12), ("moderate_rejection", 0.25), ("no_rejection", 0.63)],
    "sr_cluster": [("mega_zone", 0.15), ("double_zone", 0.35), ("single", 0.50)],
}


def _random_feature_value(feat_name: str, rng: np.random.RandomState):
    """Amostra um valor aleatório para uma feature baseado nas taxas base."""
    options = FEATURE_BASE_RATES[feat_name]
    values, probs = zip(*options)
    probs = np.array(probs, dtype=np.float64)
    probs /= probs.sum()
    return values[rng.choice(len(values), p=probs)]


def generate_warm_start_data(rng: np.random.RandomState = None) -> List[Tuple[Dict, int]]:
    """
    Gera dados sintéticos de treinamento a partir dos resultados do estudo.
    Retorna lista de (features_dict, label) onde label = 0 ou 1.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    data = []

    for combo_features, wr, n_samples in STUDY_COMBOS:
        actual_n = min(n_samples, WARM_MAX_SAMPLES_PER_COMBO)

        for _ in range(actual_n):
            # Gerar features aleatórias de base
            features = {}
            for feat_name in FEATURE_SCHEMA:
                features[feat_name] = _random_feature_value(feat_name, rng)

            # Sobrescrever com features do combo
            for k, v in combo_features.items():
                features[k] = v

            # Label baseado na WR observada
            label = 1 if rng.random() < wr else 0
            data.append((features, label))

    # Dados de baseline: features aleatórias com WR base (51.2%)
    for _ in range(500):
        features = {}
        for feat_name in FEATURE_SCHEMA:
            features[feat_name] = _random_feature_value(feat_name, rng)
        label = 1 if rng.random() < 0.512 else 0
        data.append((features, label))

    rng.shuffle(data)
    return data


# ═══════════════════════════════════════════════════════════════
# 7. CONFLUENCE BRAIN — Classe principal
# ═══════════════════════════════════════════════════════════════

class ConfluenceBrain:
    """
    Cérebro ML que PENSA antes de cada trade.

    Fluxo:
    1. classify_features() → extrair 13 features do contexto
    2. encode() → vetor numérico de 64 dimensões
    3. model.predict_proba() → P(win) calibrado
    4. Ajustar com real-time data → P(win) ajustado
    5. Comparar com threshold dinâmico → decisão
    6. Após resultado: model.partial_fit() → aprender
    """

    def __init__(self):
        self.model = OnlineLogisticRegression(N_TOTAL_FEATURES)
        self.wr_tracker = FeatureWRTracker()
        self.regime = RegimeDetector()
        self.total_trades = 0
        self.total_wins = 0
        self._last_loss_ts = 0.0

        # Contadores de treinamento (backtest) — separados dos trades ao vivo
        self.training_samples = 0
        self.training_wins = 0

        # Âncora: pesos do warm start — referência para limitar drift total
        self._anchor_w = None
        self._anchor_b = None

        # Tentar carregar estado anterior
        loaded = self._load_state()
        if not loaded or self.model.n_updates == 0:
            self._warm_start()

        # Salvar âncora após warm start (ou carregar estado bom)
        if self._anchor_w is None:
            self._anchor_w = self.model.w.copy()
            self._anchor_b = self.model.b

    # ─────────────────────────────────────────────────
    # ENCODE — Vetor numérico de features
    # ─────────────────────────────────────────────────
    def encode(self, features: Dict) -> np.ndarray:
        """
        Codifica features categóricas + numéricas em vetor de 64 dimensões.

        [0..41]   → one-hot das 13 features categóricas
        [42..59]  → features de interação (combos conhecidos)
        [60..63]  → features numéricas normalizadas
        """
        X = np.zeros(N_TOTAL_FEATURES, dtype=np.float64)

        # ── One-hot ──
        for feat_name in FEATURE_SCHEMA:
            val = features.get(feat_name)
            if val is not None:
                idx = _FEATURE_INDEX.get((feat_name, val))
                if idx is not None:
                    X[idx] = 1.0

        # ── Interações ──
        for i, combo in enumerate(INTERACTION_FEATURES):
            match = True
            for k, v in combo.items():
                if features.get(k) != v:
                    match = False
                    break
            if match:
                X[N_ONEHOT + i] = 1.0

        # ── Numéricas (normalizadas para ~[-1, +1]) ──
        # Defaults são valores "neutros" que codificam para 0.0
        body_ratio = float(features.get("body_ratio", 0.5))
        wick_ratio = float(features.get("wick_ratio", 0.5))
        dist_atr = float(features.get("dist_atr", 1.0))
        consec = float(features.get("consec_before", 3))

        X[N_ONEHOT + N_INTERACTIONS + 0] = (body_ratio - 0.5) * 2.0
        X[N_ONEHOT + N_INTERACTIONS + 1] = (wick_ratio - 0.5) * 2.0
        X[N_ONEHOT + N_INTERACTIONS + 2] = np.clip(dist_atr, 0.0, 2.0) - 1.0
        X[N_ONEHOT + N_INTERACTIONS + 3] = (np.clip(consec, 0.0, 6.0) - 3.0) / 3.0

        return X

    # ─────────────────────────────────────────────────
    # PREDICT — Probabilidade de WIN
    # ─────────────────────────────────────────────────
    def predict(self, features: Dict) -> float:
        """Retorna P(win) entre 0.0 e 1.0.
        
        Combina modelo ML + regras duras de anti-padrões.
        O modelo ML aprende padrões positivos (o que funciona).
        As regras duras bloqueiam anti-padrões comprovados (garantia).
        """
        X = self.encode(features)
        prob = self.model.predict_proba(X)

        # ── HARD ANTI-PATTERN RULES ──
        # Estes anti-padrões são COMPROVADOS por 10.668 amostras.
        # Logistic regression é linear e pode não aprender bem as interações
        # negativas. Regras duras garantem proteção.
        candle = features.get("candle")
        approach = features.get("approach")
        rejection = features.get("rejection")
        is_sweep = features.get("is_sweep")
        vol_ctx = features.get("vol_ctx")

        # Exaustão + Pin bar = 42.1% (n=38) — PIOR combo conhecido
        if approach == "exhaustion" and candle == "pin_bar":
            prob = min(prob, 0.42)

        # Exaustão + Reject forte = 40.0% (n=30) — SEGUNDO pior
        if approach == "exhaustion" and rejection == "strong":
            prob = min(prob, 0.40)

        # Gradual + Reject forte = 45.7% (n=490) — grande amostra!
        if approach == "gradual" and rejection == "strong":
            prob = min(prob, 0.46)

        # Sweep em geral = 49.7% — sem edge
        if is_sweep:
            prob = min(prob, 0.50)

        # Volume alto = 49.4% — zonas menos respeitadas
        if vol_ctx == "high":
            prob *= 0.95  # Penalizar 5% (menos agressivo)

        return prob

    # ─────────────────────────────────────────────────
    # CLASSIFY — Extrair 11 features do contexto
    # ─────────────────────────────────────────────────
    def classify_features(self, df: pd.DataFrame, setup: Dict,
                          atr_val: float) -> Dict[str, Any]:
        """
        Extrai TODAS as features do contexto atual.
        Usa dados do DataFrame (últimas velas) + setup da estratégia.
        """
        n = len(df)
        if n < 20:
            return {}

        o = df["open"].values.astype(float)
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        c = df["close"].values.astype(float)

        i = n - 1  # última vela (a do toque)
        body = c[i] - o[i]
        body_abs = abs(body)
        candle_range = h[i] - l[i]
        if candle_range < 1e-9:
            candle_range = atr_val * 0.01

        upper_wick = h[i] - max(o[i], c[i])
        lower_wick = min(o[i], c[i]) - l[i]
        body_ratio = body_abs / candle_range

        direction = setup.get("dir", "CALL")
        zone_price = float(setup.get("zone_price", c[i]))
        zone_high = float(setup.get("zone_high", zone_price + atr_val * 0.2))
        zone_low = float(setup.get("zone_low", zone_price - atr_val * 0.2))

        # Wick de rejeição (na direção certa)
        if direction == "CALL":
            rejection_wick = lower_wick
            aligned_body = body > 0
        else:
            rejection_wick = upper_wick
            aligned_body = body < 0

        wick_ratio = rejection_wick / candle_range if candle_range > 0 else 0

        features = {}

        # ── 1. CANDLE TYPE ──
        if wick_ratio > 0.55:
            features["candle"] = "pin_bar"
        elif body_ratio < 0.20:
            features["candle"] = "doji"
        elif i > 0:
            prev_body = abs(c[i - 1] - o[i - 1])
            if body_abs > prev_body * 1.2 and aligned_body:
                features["candle"] = "engulfing"
            elif body_ratio > 0.55 and aligned_body:
                features["candle"] = "strong_aligned"
            elif body_ratio > 0.55 and not aligned_body:
                features["candle"] = "strong_contra"
            else:
                features["candle"] = "neutral"
        else:
            features["candle"] = "neutral"

        # ── 2. STRUCTURE ──
        trend_dir = setup.get("trend_dir", "neutro")
        if direction == "CALL":
            if trend_dir in ("alta", "up"):
                features["structure"] = "favor"
            elif trend_dir in ("baixa", "down"):
                features["structure"] = "contra"
            else:
                features["structure"] = "range"
        else:
            if trend_dir in ("baixa", "down"):
                features["structure"] = "favor"
            elif trend_dir in ("alta", "up"):
                features["structure"] = "contra"
            else:
                features["structure"] = "range"

        if features["structure"] == "range" and n >= 30:
            features["structure"] = self._calc_structure_from_swings(h, l, c, direction)

        # ── 3. ZONE FRESHNESS ──
        # A estratégia seleciona zonas com S/R forte (muitos toques no backtest).
        # Por isso zonas aprovadas NATURALMENTE têm muitos toques.
        # Thresholds alinhados com a realidade: zona com 15+ toques = "used".
        prior_touches = 0
        lookback = min(200, n - 1)
        for j in range(max(0, i - lookback), i):
            if l[j] <= zone_high and h[j] >= zone_low:
                prior_touches += 1

        if prior_touches <= 5:
            features["zone_fresh"] = "very_fresh"
        elif prior_touches <= 15:
            features["zone_fresh"] = "fresh"
        else:
            features["zone_fresh"] = "used"

        # ── 4. APPROACH ──
        consec = 0
        for j in range(i - 1, max(i - 8, 0), -1):
            if direction == "CALL" and c[j] < o[j]:
                consec += 1
            elif direction == "PUT" and c[j] > o[j]:
                consec += 1
            else:
                break

        prev_body_atr = abs(c[i - 1] - o[i - 1]) / atr_val if i > 0 and atr_val > 0 else 0

        # Detectar se bodies estão diminuindo (exaustão real) ou persistentes
        bodies_fading = False
        if consec >= 3 and i >= consec + 1:
            recent_bodies = [abs(c[i - 1 - k] - o[i - 1 - k]) for k in range(min(consec, 5))]
            if len(recent_bodies) >= 3:
                first_half = sum(recent_bodies[len(recent_bodies)//2:])
                second_half = sum(recent_bodies[:len(recent_bodies)//2])
                if first_half > 0 and second_half < first_half * 0.6:
                    bodies_fading = True  # bodies DIMINUINDO = exaustão real

        if consec >= 4 and bodies_fading:
            features["approach"] = "exhaustion"  # Exaustão REAL: muitas velas + bodies encolhendo
        elif consec >= 4 and not bodies_fading:
            features["approach"] = "impulse"     # 4+ velas mas bodies fortes = impulso ainda ativo
        elif consec >= 3 and prev_body_atr >= 0.5:
            features["approach"] = "impulse"     # 3 velas com body forte = impulso
        elif prev_body_atr >= 0.6:
            features["approach"] = "impulse"
        elif consec <= 1 and prev_body_atr < 0.3:
            features["approach"] = "moderate"
        else:
            features["approach"] = "gradual"

        features["consec_before"] = consec
        features["bodies_fading"] = bodies_fading

        # ── 5. VOLATILITY CONTEXT ──
        if n >= 25:
            ranges_hist = h[max(0, i - 20):i] - l[max(0, i - 20):i]
            vol_std = float(np.std(ranges_hist))
            recent_ranges = h[max(0, i - 5):i] - l[max(0, i - 5):i]
            recent_vol = float(np.std(recent_ranges)) if len(recent_ranges) > 2 else vol_std

            if recent_vol < vol_std * 0.7:
                features["vol_ctx"] = "low"
            elif recent_vol > vol_std * 1.3:
                features["vol_ctx"] = "high"
            else:
                features["vol_ctx"] = "normal"
        else:
            features["vol_ctx"] = "normal"

        # ── 6. DISTANCE CATEGORY ──
        dist_atr = float(setup.get("zone_distance_atr",
                                    abs(c[i] - zone_price) / max(atr_val, 1e-9)))
        if dist_atr < 0.15:
            features["dist_cat"] = "tight"
        elif dist_atr < 0.30:
            features["dist_cat"] = "close"
        else:
            features["dist_cat"] = "loose"

        # ── 7. REJECTION STRENGTH ──
        if wick_ratio > 0.60:
            features["rejection"] = "strong"
        elif wick_ratio > 0.35:
            features["rejection"] = "moderate"
        else:
            features["rejection"] = "weak"

        # ── 8. PRE-PATTERN ──
        if i >= 3:
            bodies_3 = [c[i - k] - o[i - k] for k in range(1, 4)]
            if direction == "CALL":
                contra_count = sum(1 for b in bodies_3 if b < 0)
            else:
                contra_count = sum(1 for b in bodies_3 if b > 0)

            if contra_count >= 3:
                features["pre_pattern"] = "3_contra"
            elif contra_count >= 1:
                features["pre_pattern"] = "1_contra"
            else:
                features["pre_pattern"] = "mixed"
        else:
            features["pre_pattern"] = "mixed"

        # ── 9. MOMENTUM ──
        if i >= 5:
            move_5 = abs(c[i] - c[i - 5])
            move_atr = move_5 / max(atr_val, 1e-9)
            if move_atr > 1.5:
                features["momentum"] = "strong_move"
            elif move_atr > 0.8:
                features["momentum"] = "moderate_move"
            else:
                features["momentum"] = "low_move"
        else:
            features["momentum"] = "low_move"

        # ── 10. BODIES TYPE ──
        if i >= 3:
            avg_body = np.mean([abs(c[i - k] - o[i - k]) for k in range(1, 4)])
            bodies_atr = avg_body / max(atr_val, 1e-9)
            if bodies_atr > 0.45:
                features["bodies"] = "big"
            elif bodies_atr < 0.20:
                features["bodies"] = "small"
            else:
                features["bodies"] = "normal"
        else:
            features["bodies"] = "normal"

        # ── 11. IS SWEEP ──
        if direction == "CALL":
            features["is_sweep"] = bool(l[i] < zone_low and c[i] > zone_price)
        else:
            features["is_sweep"] = bool(h[i] > zone_high and c[i] < zone_price)

        # ── 12. CHANNEL CONTEXT ──
        ch_type = str(setup.get("channel_type", "none"))
        if ch_type in ("ascending", "descending", "lateral"):
            at_upper = bool(setup.get("channel_at_upper", False))
            at_lower = bool(setup.get("channel_at_lower", False))
            if at_upper or at_lower:
                features["channel"] = "at_edge"
            else:
                features["channel"] = "mid"
        else:
            features["channel"] = "none"

        # ── 13. WICK REJECTION AT EXTREME ──
        has_wick = bool(setup.get("has_wick_rejection", False))
        wick_at_zone = bool(setup.get("wick_at_zone", False))
        wick_at_ch = bool(setup.get("wick_at_channel", False))
        if has_wick and (wick_at_zone or wick_at_ch):
            features["wick_extreme"] = "strong_rejection"
        elif has_wick:
            features["wick_extreme"] = "moderate_rejection"
        else:
            features["wick_extreme"] = "no_rejection"

        # ── 14. CLUSTER S/R — zonas agrupadas (múltiplas zonas próximas) ──
        nearby_cnt = int(setup.get("nearby_zones_count", 0))
        if nearby_cnt >= 2:
            features["sr_cluster"] = "mega_zone"   # 3+ zonas = mega zona
        elif nearby_cnt == 1:
            features["sr_cluster"] = "double_zone"  # 2 zonas = zona dupla
        else:
            features["sr_cluster"] = "single"        # zona isolada

        # Extras numéricos para o modelo ML
        features["body_ratio"] = round(body_ratio, 3)
        features["wick_ratio"] = round(wick_ratio, 3)
        features["dist_atr"] = round(dist_atr, 3)
        features["atr_val"] = atr_val

        return features

    def _calc_structure_from_swings(self, h, l, c, direction) -> str:
        """Calcula estrutura HH/HL/LH/LL das últimas 50 velas."""
        n = len(h)
        look = min(50, n)
        seg_h = h[n - look:]
        seg_l = l[n - look:]

        sH, sL = [], []
        for j in range(3, look - 3):
            if seg_h[j] == max(seg_h[j - 3:j + 4]):
                sH.append(float(seg_h[j]))
            if seg_l[j] == min(seg_l[j - 3:j + 4]):
                sL.append(float(seg_l[j]))

        if len(sH) < 2 or len(sL) < 2:
            return "range"

        hh = sH[-1] > sH[-2]
        hl = sL[-1] > sL[-2]

        if hh and hl:
            trend = "up"
        elif not hh and not hl:
            trend = "down"
        else:
            return "range"

        if direction == "CALL":
            return "favor" if trend == "up" else "contra"
        else:
            return "favor" if trend == "down" else "contra"

    # ─────────────────────────────────────────────────
    # THINK — Decidir com base no modelo ML
    # ─────────────────────────────────────────────────
    def think(self, df: pd.DataFrame, setup: Dict,
              atr_val: float) -> Tuple[str, float, str, Dict]:
        """
        Cérebro ML — classifica features e consulta o modelo.

        Retorna:
            (decision, prob_pct, reasoning, features)
            decision: "ENTER_NOW" / "WAIT_NEXT" / "SKIP"
            prob_pct: P(win) * 100 (para logging/ranking)
            reasoning: texto explicativo
            features: dict com features classificadas
        """
        features = self.classify_features(df, setup, atr_val)
        if not features:
            return ("SKIP", 0.0, "Dados insuficientes (<20 candles)", {})

        prob = self.predict(features)
        threshold = self.get_dynamic_threshold()

        # Alertas do tracker (features com WR muito baixo)
        alerts = self.wr_tracker.get_alerts(features)

        # Resumo das features
        feat_summary = (
            f"candle={features.get('candle', '?')}, "
            f"struct={features.get('structure', '?')}, "
            f"fresh={features.get('zone_fresh', '?')}, "
            f"approach={features.get('approach', '?')}, "
            f"vol={features.get('vol_ctx', '?')}, "
            f"sweep={'S' if features.get('is_sweep') else 'N'}, "
            f"cluster={features.get('sr_cluster', 'single')}"
        )

        # Reasoning
        parts = [f"[{feat_summary}]"]
        regime = self.regime.get_regime()
        if regime != "normal" and regime != "unknown":
            parts.append(f"regime={regime}")
        if alerts:
            parts.append(f"ALERTA:{','.join(alerts)}")

        # Decisão
        prob_pct = round(prob * 100, 1)

        if prob >= threshold:
            confidence = "ALTA" if prob >= HIGH_CONFIDENCE_PROB else "MÉDIA"
            parts.append(f"P={prob:.1%}>={threshold:.0%} -> ENTRAR ({confidence})")
            decision = "ENTER_NOW"
        elif prob >= threshold - 0.02:
            # Muito perto do threshold (53-55%) → esperar próxima vela confirmar
            confidence = "MARGINAL"
            parts.append(f"P={prob:.1%} perto de {threshold:.0%} -> WAIT_NEXT")
            decision = "WAIT_NEXT"
            features["_brain_marginal"] = True
        else:
            parts.append(f"P={prob:.1%}<{threshold-0.02:.0%} -> SKIP")
            decision = "SKIP"

        reasoning = " | ".join(parts)
        return (decision, prob_pct, reasoning, features)

    def think_with_rt(self, bx, ativo: str, df: pd.DataFrame,
                      setup: Dict, atr_val: float,
                      wait_count: int = 0,
                      get_candles_fn=None) -> Tuple[str, float, str, Dict]:
        """
        Análise completa com verificação real-time do preço.

        1. Predição base do modelo ML
        2. Leitura do preço real-time
        3. Ajuste de probabilidade baseado na posição
        4. Decisão final com probabilidade ajustada
        """
        # ── Predição base ──
        decision, prob_pct, reasoning, features = self.think(df, setup, atr_val)

        if decision == "SKIP":
            return (decision, prob_pct, reasoning, features)

        if wait_count >= 3:
            return ("SKIP", prob_pct,
                    f"Esperou {wait_count} candles | {reasoning}", features)

        prob = prob_pct / 100.0  # Converter de volta para [0, 1]

        direction = setup.get("dir", "CALL")
        zone_price = float(setup.get("zone_price", 0))
        zone_high = float(setup.get("zone_high", zone_price + atr_val * 0.2))
        zone_low = float(setup.get("zone_low", zone_price - atr_val * 0.2))

        if zone_price == 0:
            return (decision, prob_pct, reasoning, features)

        # ── Leitura real-time ──
        rt_adj = 0.0
        rt_detail = ""

        try:
            df_rt = get_candles_fn(bx, ativo, 60, 5, time.time()) if get_candles_fn else None
            if df_rt is not None and len(df_rt) >= 2:
                forming = df_rt.iloc[-1]
                current_price = float(forming["close"])
                forming_body = current_price - float(forming["open"])

                margin = atr_val * 0.30
                inside_zone = (zone_low - margin) <= current_price <= (zone_high + margin)
                dist_atr = abs(current_price - zone_price) / max(atr_val, 1e-9)
                far_from_zone = dist_atr > 0.60

                if direction == "CALL":
                    forming_favor = forming_body > 0
                    broke_through = current_price < zone_low - margin
                else:
                    forming_favor = forming_body < 0
                    broke_through = current_price > zone_high + margin

                if broke_through and far_from_zone:
                    rt_adj = -0.20
                    rt_detail = f"rompeu ({dist_atr:.2f}ATR)"
                elif far_from_zone:
                    rt_adj = -0.12
                    rt_detail = f"longe ({dist_atr:.2f}ATR)"
                elif inside_zone and forming_favor:
                    rt_adj = +0.03
                    rt_detail = f"zona+favor ({dist_atr:.2f}ATR)"
                elif inside_zone:
                    rt_adj = 0.0
                    rt_detail = f"na zona ({dist_atr:.2f}ATR)"
                elif forming_favor:
                    rt_adj = +0.01
                    rt_detail = f"perto+favor ({dist_atr:.2f}ATR)"
                else:
                    rt_adj = -0.05
                    rt_detail = f"afastando ({dist_atr:.2f}ATR)"
            else:
                rt_detail = "sem RT"
        except Exception as e:
            log.debug(f"[BRAIN] RT error: {e}")
            rt_detail = "erro RT"

        # ── Override de momentum extremo (menos agressivo para evitar dupla penalidade) ──
        momentum_total = abs(float(setup.get("momentum_total_move_atr", 0)))
        momentum_contra = int(setup.get("momentum_contra", 0))

        if momentum_total >= 3.0:
            rt_adj = min(rt_adj, -0.15)
            rt_detail += f" | FALLING KNIFE ({momentum_total:.1f}ATR)"
        elif momentum_total >= 2.0:
            rt_adj = min(rt_adj, -0.08)
            rt_detail += f" | mom_forte ({momentum_total:.1f}ATR)"
        elif momentum_contra >= 4:
            rt_adj = min(rt_adj, -0.05)
            rt_detail += f" | {momentum_contra}v contra"

        # ── Penalty por APPROACH IMPULSE (velas fortes chegando na zona) ──
        approach_type = features.get("approach", "moderate") if features else "moderate"
        if approach_type == "impulse" and momentum_total >= 1.5:
            rt_adj = min(rt_adj, -0.05)
            rt_detail += " | impulse_approach"
        # ── Penalty por continuation flag (pausa após move forte) ──
        continuation = bool(setup.get("continuation_flag", False))
        if continuation:
            rt_adj = min(rt_adj, -0.07)
            rt_detail += " | continuation_flag"

        # ── Penalty por ESTRUTURA CONTRA ──
        # Se o preço está NA zona e vela confirmou direção, NÃO penalizar por
        # struct_contra — trade S/R é contra-tendência por natureza.
        struct_type = features.get("structure", "range") if features else "range"
        zone_dist_atr = float(setup.get("zone_distance_atr", 1.0))
        candle_type = features.get("candle", "neutral") if features else "neutral"
        candle_confirms = candle_type in ("pin_bar", "engulfing", "doji", "strong_aligned")
        inside_zone_setup = zone_dist_atr < 0.30

        if struct_type == "contra" and not (inside_zone_setup and candle_confirms):
            rt_adj = min(rt_adj, -0.05)
            rt_detail += " | struct_contra"

        # ── Bônus por CLUSTER S/R (zonas agrupadas = mega zona forte) ──
        sr_cluster = features.get("sr_cluster", "single") if features else "single"
        if sr_cluster == "mega_zone":
            rt_adj += 0.06
            rt_detail += " | mega_zone(+6%)"
        elif sr_cluster == "double_zone":
            rt_adj += 0.04
            rt_detail += " | double_zone(+4%)"

        # ── Bônus por PADRÃO DE ESTRUTURA (13 padrões vencedores do estudo) ──
        if _STRUCTURE_PATTERNS_AVAILABLE and df is not None and len(df) >= 50:
            try:
                direction = setup.get("dir", "CALL")
                sp_result = detect_structure_pattern(df, atr_val, direction=direction)
                if sp_result.get("match"):
                    sp_bonus = sp_result["bonus_pct"]
                    rt_adj += sp_bonus
                    hr_tag = "HR_OK" if sp_result["hour_match"] else "hr_diff"
                    rt_detail += (f" | PATTERN({sp_result['pattern_dir']} "
                                  f"WR={sp_result['pattern_wr']:.0%} {hr_tag} "
                                  f"+{sp_bonus:.0%})")
                    if features:
                        features["struct_pattern"] = sp_result["pattern_name"]
                        features["struct_pattern_wr"] = sp_result["pattern_wr"]
                        features["struct_pattern_bonus"] = sp_bonus
                    log.info(f"[BRAIN] {pattern_summary(sp_result)}")
                else:
                    if features:
                        features["struct_pattern"] = "none"
            except Exception as e:
                log.debug(f"[BRAIN] Structure pattern error: {e}")

        # ── Bônus/penalidade por HORA DO DIA ──
        if _STRUCTURE_PATTERNS_AVAILABLE:
            try:
                _utc_h = None
                if df is not None and len(df) > 0:
                    _ts = df.index[-1]
                    _utc_h = int(_ts.hour) if hasattr(_ts, "hour") else None
                h_bonus = get_hour_bonus(_utc_h)
                if h_bonus != 0:
                    rt_adj += h_bonus
                    rt_detail += f" | hour({_utc_h:02d}h {h_bonus:+.0%})"
            except Exception:
                pass

        # ── Bônus/penalidade por MAPA ESTRUTURAL (zigzag + regiões) ──
        if _STRUCTURE_MAP_AVAILABLE and df is not None and len(df) >= 30:
            try:
                map_result = detect_structure_touch(df, atr_val)
                if map_result.get("touch"):
                    # Verificar alinhamento com direção do setup
                    setup_dir = setup.get("dir", "CALL")
                    alignment = check_alignment(map_result, setup_dir)
                    
                    if alignment["aligned"]:
                        map_bonus = alignment["bonus"]
                        rt_adj += map_bonus
                        rt_detail += (f" | MAP_ALINHADO({map_result['direction']} "
                                      f"conf={map_result['confidence']:.0%} "
                                      f"+{map_bonus:.0%})")
                        if features:
                            features["struct_map"] = "aligned"
                            features["struct_map_conf"] = map_result["confidence"]
                    else:
                        map_pen = alignment["bonus"]  # negativo
                        rt_adj += map_pen
                        rt_detail += (f" | MAP_CONTRA({map_result['direction']} "
                                      f"vs {setup_dir} {map_pen:+.0%})")
                        if features:
                            features["struct_map"] = "contra"
                    
                    log.info(f"[BRAIN] {structure_map_summary(map_result)}")
                else:
                    if features:
                        features["struct_map"] = "sem_toque"
            except Exception as e:
                log.debug(f"[BRAIN] Structure map error: {e}")

        # ── Probabilidade ajustada ──
        adjusted_prob = max(0.0, min(1.0, prob + rt_adj))

        # ── IA AUTÔNOMA — Refina probabilidade com aprendizado contextual ──
        auto_detail = ""
        if _AUTONOMOUS_AVAILABLE:
            try:
                auto_brain = _get_auto_brain()
                adjusted_prob, auto_detail = auto_brain.refine_probability(
                    adjusted_prob, features or {}, ativo, setup
                )
            except Exception as e:
                log.debug(f"[BRAIN] Autonomous refine error: {e}")
                auto_detail = ""

        threshold = self.get_dynamic_threshold()

        # ── Meta-learning threshold adjustment ──
        if _AUTONOMOUS_AVAILABLE:
            try:
                auto_brain = _get_auto_brain()
                meta_adj = auto_brain.meta_learner.get_threshold_adjustment()
                threshold = max(0.35, min(0.65, threshold + meta_adj))
            except Exception:
                pass

        adj_str = f"{rt_adj:+.0%}" if rt_adj != 0 else "0"
        full_reason = (
            f"P={prob:.1%}->{adjusted_prob:.1%} [RT:{adj_str} {rt_detail}]"
            f" {auto_detail} | {reasoning}"
        )
        adj_pct = round(adjusted_prob * 100, 1)

        if adjusted_prob >= threshold:
            confidence = "ALTA" if adjusted_prob >= HIGH_CONFIDENCE_PROB else "MEDIA"
            # Limpar flag marginal do think() — RT ajustou para acima do threshold
            if features:
                features.pop("_brain_marginal", None)
            return ("ENTER_NOW", adj_pct,
                    f"{confidence} | {full_reason}", features)
        elif adjusted_prob >= threshold - 0.02:
            # Muito perto do threshold (53-55%) → esperar próxima vela confirmar
            confidence = "MARGINAL"
            if features:
                features["_brain_marginal"] = True
            return ("WAIT_NEXT", adj_pct,
                    f"{confidence} (marginal→wait_next) | {full_reason}", features)
        else:
            return ("SKIP", adj_pct,
                    f"P<threshold ({threshold:.0%}) | {full_reason}", features)

    # ─────────────────────────────────────────────────
    # THRESHOLD DINÂMICO
    # ─────────────────────────────────────────────────
    def get_dynamic_threshold(self) -> float:
        """Retorna threshold como probabilidade (0-1).
        Threshold sobe com experiência: IA iniciante explora, IA experiente é seletiva."""
        base = BASE_PROB_THRESHOLD
        adj = self.regime.get_threshold_adjustment()
        
        # Experiência: modelo precisa explorar no início
        # Com poucos trades online, threshold BAIXO para aprender
        # Após 50+ trades, threshold volta ao normal
        if self.total_trades < 10:
            exp_adj = -0.03  # Exploração moderada (primeiros trades)
        elif self.total_trades < 30:
            exp_adj = -0.02  # Ainda aprendendo
        elif self.total_trades < 50:
            exp_adj = -0.01  # Quase maduro
        else:
            exp_adj = 0.0    # Modelo maduro
        
        # Floor nunca abaixo de 50% — moeda jogada NÃO é trade
        _floor = 0.50
        return max(_floor, base + adj + exp_adj)

    # ─────────────────────────────────────────────────
    # APRENDER — Atualizar modelo após cada trade
    # ─────────────────────────────────────────────────
    def record_result(self, features: Dict, win: bool,
                      ativo: str = "", setup: Dict = None, exp_min: int = 3):
        """
        Registra resultado e ATUALIZA o modelo ML + IA autônoma.
        Este é o online learning real — o modelo melhora a cada trade.
        """
        if not features:
            return

        self.total_trades += 1
        if win:
            self.total_wins += 1

        # ── Atualizar modelo ML via SGD ──
        X = self.encode(features)
        y = 1 if win else 0

        # ── Meta-learning: usar LR dinâmico do cérebro autônomo ──
        lr = LEARNING_RATE_AFTER_LOSS if not win else LEARNING_RATE
        if _AUTONOMOUS_AVAILABLE:
            try:
                auto_brain = _get_auto_brain()
                meta_lr = auto_brain.meta_learner.get_learning_rate()
                # Combinar: se loss, multiplicar; se win, usar meta_lr
                if not win:
                    lr = max(lr, meta_lr * 1.5)
                else:
                    lr = meta_lr
            except Exception:
                pass
        self.model.partial_fit(X, y, lr_override=lr)

        # ── Atualizar tracker e regime ──
        self.wr_tracker.update(features, win)
        self.regime.update(win)

        if not win:
            self._last_loss_ts = time.time()

        # ── IA AUTÔNOMA: aprendizado contextual ──
        if _AUTONOMOUS_AVAILABLE:
            try:
                auto_brain = _get_auto_brain()
                auto_brain.learn(
                    features=features,
                    ativo=ativo or "unknown",
                    setup=setup or {},
                    win=win,
                    exp_min=exp_min,
                )
            except Exception as e:
                log.debug(f"[BRAIN] Autonomous learn error: {e}")

        # ── Log ──
        wr = self.total_wins / max(1, self.total_trades) * 100
        level = self._get_experience_level()
        log.info(
            f"[BRAIN] {'WIN' if win else 'LOSS'} registrado | "
            f"nivel={level} | "
            f"online={self.total_trades} trades WR={wr:.1f}% | "
            f"treinamento={self.training_samples} amostras | "
            f"regime={self.regime.get_regime()} | "
            f"model.updates={self.model.n_updates}"
        )

        # ── Salvar ──
        self._save_state()

    def record_loss_feedback(self, features: Dict, setup: Dict):
        """
        Feedback adicional após LOSS.
        NÃO atualiza o modelo de novo (já foi feito em record_result).
        Apenas analisa e emite alertas.
        """
        alerts = self.wr_tracker.get_alerts(features)
        if alerts:
            for alert in alerts:
                log.warning(f"[BRAIN] ⚠️ Feature em ALERTA: {alert}")

        # Verificar se anti-combo foi ativado
        for i, combo in enumerate(INTERACTION_FEATURES):
            if i >= 14:  # anti-combos são índices 14-17
                match = all(features.get(k) == v for k, v in combo.items())
                if match:
                    log.warning(f"[BRAIN] Anti-combo detectado no LOSS: {combo}")

    # ─────────────────────────────────────────────────
    # WARM START — Pré-treinar com dados do estudo
    # ─────────────────────────────────────────────────
    def _warm_start(self):
        """
        Pré-treina o modelo com dados sintéticos baseados no estudo real.
        Faz múltiplos epochs sobre ~2000 amostras.
        """
        rng = np.random.RandomState(42)
        data = generate_warm_start_data(rng)

        for epoch in range(WARM_START_EPOCHS):
            rng.shuffle(data)
            lr = WARM_START_LR / (1.0 + epoch * 0.3)
            for features, label in data:
                X = self.encode(features)
                self.model.partial_fit(X, label, lr_override=lr)

        # Verificar calibração com setups conhecidos
        test_good = {"candle": "doji", "approach": "exhaustion",
                     "structure": "contra", "zone_fresh": "fresh",
                     "vol_ctx": "low", "bodies": "small",
                     "dist_cat": "close", "rejection": "weak",
                     "pre_pattern": "3_contra", "momentum": "low_move",
                     "is_sweep": False}
        test_bad = {"candle": "pin_bar", "approach": "exhaustion",
                    "structure": "range", "zone_fresh": "used",
                    "vol_ctx": "high", "dist_cat": "loose",
                    "rejection": "strong", "pre_pattern": "mixed",
                    "momentum": "moderate_move", "bodies": "normal",
                    "is_sweep": True}
        test_mid = {"candle": "neutral", "approach": "moderate",
                    "structure": "range", "zone_fresh": "fresh",
                    "vol_ctx": "normal", "dist_cat": "close",
                    "rejection": "moderate", "pre_pattern": "1_contra",
                    "momentum": "low_move", "bodies": "normal",
                    "is_sweep": False}

        p_good = self.predict(test_good)
        p_bad = self.predict(test_bad)
        p_mid = self.predict(test_mid)

        log.info(
            f"[BRAIN] Warm start OK: {len(data)} amostras x {WARM_START_EPOCHS} epochs "
            f"({self.model.n_updates} updates) | "
            f"Calibracao: bom={p_good:.1%} medio={p_mid:.1%} ruim={p_bad:.1%}"
        )

    # ─────────────────────────────────────────────────
    # TREINAMENTO COM DADOS REAIS DE BACKTEST
    # ─────────────────────────────────────────────────
    def train_from_backtest(self, samples):
        """
        FINE-TUNE o modelo com dados REAIS extraídos de backtest histórico.

        samples: lista de (features_dict, label)
            features_dict = resultado de classify_features()
            label = 1 (WIN) ou 0 (LOSS)

        IMPORTANTE: Usa LR BAIXO para AJUSTAR o modelo, não sobrescrever.
        O warm start (10.668 amostras do estudo) é a fundação.
        Backtest é apenas fine-tuning com dados recentes do mercado.
        """
        if not samples:
            return

        n = len(samples)
        rng = np.random.RandomState(int(time.time()) % (2**31))

        # Fine-tuning: mais epochs para aprendizado real
        # 8 epochs = mesmo que warm start. Modelo aprende de verdade com dados recentes.
        # Antes era 2 — insuficiente para o modelo absorver padrões do mercado real.
        epochs = min(8, max(3, 600 // max(n, 1)))
        base_lr = 0.008  # LR levemente maior para 8 epochs (antes 0.005 com 2 epochs)

        for epoch in range(epochs):
            indices = rng.permutation(n)
            lr = base_lr / (1.0 + epoch * 0.4)
            for idx in indices:
                feats, label = samples[idx]
                X = self.encode(feats)
                self.model.partial_fit(X, label, lr_override=lr)

        # PROTEÇÃO: limitar drift dos pesos em relação à ÂNCORA (warm start)
        # Com 8 epochs, precisa de mais margem para aprendizado real.
        # 10% permite adaptação real ao mercado sem destruir fundação.
        max_total_drift = 0.10
        if self._anchor_w is not None:
            total_delta = self.model.w - self._anchor_w
            anchor_norm = np.linalg.norm(self._anchor_w) + 1e-9
            drift_ratio = np.linalg.norm(total_delta) / anchor_norm
            if drift_ratio > max_total_drift:
                scale = max_total_drift / drift_ratio
                self.model.w = self._anchor_w + total_delta * scale
                self.model.b = self._anchor_b + (self.model.b - self._anchor_b) * scale
                log.info(f"[BRAIN] Drift total limitado: {drift_ratio:.1%} -> {max_total_drift:.0%} (protegendo fundação)")

            # CALIBRAÇÃO FLOOR: mesmo dentro do drift permitido, garantir que
            # bom combo NÃO caia abaixo de 58%. Se cair, reverter parcialmente.
            _floor_good = {"candle": "pin_bar", "structure": "favor",
                          "zone_fresh": "very_fresh", "approach": "moderate",
                          "vol_ctx": "low", "dist_cat": "close",
                          "rejection": "moderate", "pre_pattern": "3_contra",
                          "momentum": "low_move", "bodies": "small",
                          "is_sweep": False, "channel": "none",
                          "wick_extreme": "none"}
            p_check = self.predict(_floor_good)
            if p_check < 0.58:
                # Blend parcialmente de volta ao anchor até p_good >= 0.60
                for blend in [0.3, 0.5, 0.7, 0.9]:
                    self.model.w = self._anchor_w + (self.model.w - self._anchor_w) * (1 - blend)
                    self.model.b = self._anchor_b + (self.model.b - self._anchor_b) * (1 - blend)
                    p_check = self.predict(_floor_good)
                    if p_check >= 0.60:
                        break
                log.info(f"[BRAIN] Calibration floor ativado: bom combo ajustado para {p_check:.1%}")

        # Estatísticas do mercado real
        wins = sum(1 for _, lbl in samples if lbl == 1)
        wr = wins / n * 100 if n > 0 else 0

        # Atualizar contadores de treinamento (experiência do modelo)
        self.training_samples += n
        self.training_wins += wins

        # Calibração pós-treinamento
        test_good = {"candle": "pin_bar", "structure": "favor",
                     "zone_fresh": "very_fresh", "approach": "moderate",
                     "vol_ctx": "low", "dist_cat": "close",
                     "rejection": "moderate", "pre_pattern": "3_contra",
                     "momentum": "low_move", "bodies": "small",
                     "is_sweep": False, "channel": "none",
                     "wick_extreme": "none"}
        test_bad = {"candle": "strong_contra", "structure": "contra",
                    "zone_fresh": "used", "approach": "exhaustion",
                    "vol_ctx": "high", "dist_cat": "loose",
                    "rejection": "strong", "pre_pattern": "mixed",
                    "momentum": "high_move", "bodies": "big",
                    "is_sweep": True, "channel": "none",
                    "wick_extreme": "none"}
        p_good = self.predict(test_good)
        p_bad = self.predict(test_bad)

        log.info(
            f"[BRAIN] 🎯 Treinamento REAL: {n} amostras x {epochs} epochs "
            f"({n * epochs} updates) | WR mercado={wr:.1f}% | "
            f"Calibracao: bom={p_good:.1%} ruim={p_bad:.1%} | "
            f"Total updates={self.model.n_updates} | "
            f"Treinamento acumulado={self.training_samples} amostras | "
            f"Nivel={self._get_experience_level()}"
        )

        self._save_state()

    # ─────────────────────────────────────────────────
    # PERSISTÊNCIA
    # ─────────────────────────────────────────────────
    def _save_state(self):
        try:
            state_dir = os.path.dirname(BRAIN_STATE_FILE)
            os.makedirs(state_dir, exist_ok=True)

            state = {
                "version": BRAIN_VERSION,
                "total_trades": self.total_trades,
                "total_wins": self.total_wins,
                "training_samples": self.training_samples,
                "training_wins": self.training_wins,
                "model": self.model.get_state(),
                "wr_tracker": self.wr_tracker.get_state(),
                "regime": self.regime.get_state(),
                "anchor_w": self._anchor_w.tolist() if self._anchor_w is not None else None,
                "anchor_b": float(self._anchor_b) if self._anchor_b is not None else None,
                "saved_at": time.time(),
            }
            with open(BRAIN_STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log.debug(f"[BRAIN] Erro ao salvar: {e}")

    def _load_state(self) -> bool:
        """Carrega estado anterior. Retorna True se carregou com sucesso."""
        try:
            if not os.path.exists(BRAIN_STATE_FILE):
                return False

            with open(BRAIN_STATE_FILE, "r") as f:
                state = json.load(f)

            # Verificar versão
            if state.get("version") != BRAIN_VERSION:
                log.info("[BRAIN] Versao diferente, re-treinando do zero")
                return False

            self.total_trades = int(state.get("total_trades", 0))
            self.total_wins = int(state.get("total_wins", 0))
            self.training_samples = int(state.get("training_samples", 0))
            self.training_wins = int(state.get("training_wins", 0))
            self.model.load_state(state.get("model", {}))
            self.wr_tracker.load_state(state.get("wr_tracker", {}))
            self.regime.load_state(state.get("regime", {}))

            # Restaurar âncora (pesos fundação do warm start)
            anchor_w = state.get("anchor_w")
            anchor_b = state.get("anchor_b")
            if anchor_w is not None and len(anchor_w) == self.model.n_features:
                self._anchor_w = np.array(anchor_w, dtype=np.float64)
                self._anchor_b = float(anchor_b) if anchor_b is not None else 0.0

            total_exp = self.training_samples + self.total_trades
            level = self._get_experience_level()
            training_wr = self.training_wins / max(1, self.training_samples) * 100

            log.info(
                f"[BRAIN] Estado carregado: "
                f"nivel={level} | "
                f"treinamento={self.training_samples} amostras (WR={training_wr:.1f}%) | "
                f"online={self.total_trades} trades (WR={self.total_wins / max(1, self.total_trades) * 100:.1f}%) | "
                f"experiencia_total={total_exp} | "
                f"model.updates={self.model.n_updates} | "
                f"regime={self.regime.get_regime()}"
            )
            return True
        except Exception as e:
            log.debug(f"[BRAIN] Erro ao carregar: {e}")
            return False

    # ─────────────────────────────────────────────────
    # RELATÓRIOS
    # ─────────────────────────────────────────────────
    def _get_experience_level(self) -> str:
        """
        Nível de experiência da IA baseado em TODO treinamento.
        Combina amostras de backtest + trades ao vivo.
        4 níveis: Iniciante → Intermediária → Avançada → Expert
        """
        total = self.training_samples + self.total_trades
        updates = self.model.n_updates
        if total >= 2000 or updates >= 40000:
            return "EXPERT"
        elif total >= 500 or updates >= 15000:
            return "AVANCADA"
        elif total >= 100 or updates >= 5000:
            return "INTERMEDIARIA"
        else:
            return "INICIANTE"

    def _is_model_trained(self) -> bool:
        """
        Verifica se o modelo já foi treinado com dados suficientes.
        Retorna True se o modelo tem experiência significativa
        (backtest + warm start + online).
        """
        return self.training_samples >= 30 or self.model.n_updates >= 3000

    def get_status(self) -> dict:
        """Retorna status como dict (para o engine)."""
        live_wr = self.total_wins / max(1, self.total_trades) * 100
        training_wr = self.training_wins / max(1, self.training_samples) * 100
        total_exp = self.training_samples + self.total_trades
        level = self._get_experience_level()
        return {
            "total_trades": self.total_trades,
            "win_rate": round(live_wr, 1),
            "training_samples": self.training_samples,
            "training_wr": round(training_wr, 1),
            "total_experience": total_exp,
            "level": level,
            "model_trained": self._is_model_trained(),
            "threshold": round(self.get_dynamic_threshold() * 100, 1),
            "regime": self.regime.get_regime(),
            "model_updates": self.model.n_updates,
        }

    def get_feature_report(self) -> str:
        """Relatório detalhado de performance por feature."""
        lines = ["[BRAIN] Performance por Feature (EMA WR):"]
        sorted_feats = sorted(
            self.wr_tracker.ema_wr.items(),
            key=lambda x: self.wr_tracker.counts.get(x[0], 0),
            reverse=True,
        )
        for key, wr in sorted_feats[:25]:
            count = self.wr_tracker.counts.get(key, 0)
            if count >= 3:
                marker = "V" if wr >= 0.55 else ("!" if wr < 0.45 else " ")
                lines.append(f"  {marker} {key:30s}: {wr * 100:5.1f}% (n={count})")
        return "\n".join(lines)

    def get_top_weights(self, top_n: int = 15) -> str:
        """Mostra os pesos mais importantes do modelo (feature importance)."""
        lines = ["[BRAIN] Top Feature Importance (pesos do modelo):"]

        weight_labels = []
        for (feat_name, val), idx in _FEATURE_INDEX.items():
            w = self.model.w[idx]
            label = f"{feat_name}={val}"
            weight_labels.append((label, w))

        for i, combo in enumerate(INTERACTION_FEATURES):
            w = self.model.w[N_ONEHOT + i]
            label = "COMBO:" + "+".join(f"{k}={v}" for k, v in combo.items())
            weight_labels.append((label, w))

        num_names = ["body_ratio", "wick_ratio", "dist_atr", "consec_before"]
        for j, name in enumerate(num_names):
            w = self.model.w[N_ONEHOT + N_INTERACTIONS + j]
            weight_labels.append((f"NUM:{name}", w))

        weight_labels.sort(key=lambda x: abs(x[1]), reverse=True)

        for label, w in weight_labels[:top_n]:
            marker = "+" if w > 0.1 else ("-" if w < -0.1 else " ")
            lines.append(f"  {marker} {label:55s}: {w:+.4f}")

        lines.append(f"  Bias: {self.model.b:+.4f}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════

_brain_instance: Optional[ConfluenceBrain] = None


def get_brain() -> ConfluenceBrain:
    """Retorna instância singleton do ConfluenceBrain."""
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = ConfluenceBrain()
    return _brain_instance


# ═══════════════════════════════════════════════════════════════
# TESTE DE VALIDAÇÃO
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s")

    brain = ConfluenceBrain()

    print("\n" + "=" * 60)
    print("TESTE DE CALIBRACAO DO MODELO ML")
    print("=" * 60)

    # Teste 1: Setup IDEAL (doji + exaustão + muitos positivos)
    t1 = {"candle": "doji", "approach": "exhaustion",
          "structure": "contra", "zone_fresh": "fresh",
          "vol_ctx": "low", "bodies": "small",
          "dist_cat": "close", "rejection": "weak",
          "pre_pattern": "3_contra", "momentum": "low_move",
          "is_sweep": False,
          "body_ratio": 0.15, "wick_ratio": 0.1,
          "dist_atr": 0.2, "consec_before": 5}
    p1 = brain.predict(t1)

    # Teste 2: Anti-pattern (exaustão + pin bar + tudo negativo)
    t2 = {"candle": "pin_bar", "approach": "exhaustion",
          "structure": "range", "zone_fresh": "used",
          "vol_ctx": "high", "dist_cat": "loose",
          "rejection": "strong", "pre_pattern": "mixed",
          "momentum": "moderate_move", "bodies": "normal",
          "is_sweep": True,
          "body_ratio": 0.7, "wick_ratio": 0.6,
          "dist_atr": 0.8, "consec_before": 5}
    p2 = brain.predict(t2)

    # Teste 3: Setup MÉDIO (neutro)
    t3 = {"candle": "neutral", "approach": "moderate",
          "structure": "range", "zone_fresh": "fresh",
          "vol_ctx": "normal", "dist_cat": "close",
          "rejection": "moderate", "pre_pattern": "1_contra",
          "momentum": "low_move", "bodies": "normal",
          "is_sweep": False,
          "body_ratio": 0.4, "wick_ratio": 0.3,
          "dist_atr": 0.25, "consec_before": 1}
    p3 = brain.predict(t3)

    # Teste 4: Setup BOM mas sem o melhor combo
    t4 = {"candle": "strong_contra", "approach": "impulse",
          "structure": "contra", "zone_fresh": "very_fresh",
          "vol_ctx": "low", "bodies": "small",
          "dist_cat": "close", "rejection": "moderate",
          "pre_pattern": "1_contra", "momentum": "strong_move",
          "is_sweep": False,
          "body_ratio": 0.6, "wick_ratio": 0.2,
          "dist_atr": 0.15, "consec_before": 2}
    p4 = brain.predict(t4)

    threshold = brain.get_dynamic_threshold()

    print(f"\n  Threshold atual: {threshold:.1%}")
    print(f"\n  Setup IDEAL  (doji+exaustao+tudo bom):  P(win) = {p1:.1%}  {'ENTER' if p1 >= threshold else 'SKIP'}")
    print(f"  Anti-pattern (pin+exaustao+tudo ruim):  P(win) = {p2:.1%}  {'ENTER' if p2 >= threshold else 'SKIP'}")
    print(f"  Setup MEDIO  (neutro):                  P(win) = {p3:.1%}  {'ENTER' if p3 >= threshold else 'SKIP'}")
    print(f"  Setup BOM    (forte sem combo top):     P(win) = {p4:.1%}  {'ENTER' if p4 >= threshold else 'SKIP'}")

    # Verificar que o modelo discrimina corretamente
    ok = p1 > p4 > p3 > p2
    print(f"\n  Ordem correta (ideal > bom > medio > anti): {'SIM' if ok else 'NAO'}")

    if p1 < 0.60:
        print("  AVISO: Setup ideal com P < 60% — warm start pode precisar de mais epochs")
    if p2 > 0.50:
        print("  AVISO: Anti-pattern com P > 50% — modelo precisa aprender mais sobre anti-combos")

    # Simular online learning
    print("\n" + "=" * 60)
    print("TESTE DE ONLINE LEARNING (50 trades simulados)")
    print("=" * 60)

    import random
    random.seed(123)

    for trade_i in range(50):
        rng_np = np.random.RandomState(trade_i)
        feat = {}
        for fn in FEATURE_SCHEMA:
            feat[fn] = _random_feature_value(fn, rng_np)
        feat["body_ratio"] = random.uniform(0.1, 0.8)
        feat["wick_ratio"] = random.uniform(0.05, 0.6)
        feat["dist_atr"] = random.uniform(0.05, 0.7)
        feat["consec_before"] = random.randint(0, 5)

        has_edge = feat.get("candle") == "doji" or feat.get("approach") == "exhaustion"
        wr = 0.55 if has_edge else 0.48
        win = random.random() < wr

        brain.record_result(feat, win)

    status = brain.get_status()
    print(f"\n  Apos 50 trades: WR={status['win_rate']:.1f}% | "
          f"threshold={status['threshold']:.1f}% | "
          f"regime={status['regime']} | "
          f"model.updates={status['model_updates']}")

    # Re-testar calibração
    p1_after = brain.predict(t1)
    p2_after = brain.predict(t2)
    print(f"\n  Setup ideal ANTES: {p1:.1%} -> DEPOIS: {p1_after:.1%}")
    print(f"  Anti-pattern ANTES: {p2:.1%} -> DEPOIS: {p2_after:.1%}")

    print("\n" + brain.get_top_weights())
    print("\n" + brain.get_feature_report())
    print("\nTeste completo.")
