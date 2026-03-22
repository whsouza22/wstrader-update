"""
ws_context_filter.py — Filtro de contexto baseado em tabela pré-computada do backtest.

Substitui GPT + RAG manual. Decisão 100% local, <1ms, offline.

Fluxo:
  1. Bot detecta padrão → extrai geometria
  2. context_lookup(ativo, dir, hour, depth_ratio, symmetry) → decision
  3. Decision: BLOCK / raise_threshold / pass

Thresholds dinâmicos:
  WR_backtest < 80%  → BLOQUEIA (geometria historicamente ruim)
  WR 80-85%          → NN ≥ 85% (mais exigente)
  WR 85-90%          → NN ≥ 75% (padrão)
  WR ≥ 90%           → NN ≥ 68% (confiável)

Fallback cascade:
  L1 exact (n≥5) → L2 ativo+hora (n≥10) → L3 ativo+dir (n≥20) → L4 global
"""

import logging
import os
import pickle
from typing import Optional, Tuple

log = logging.getLogger("ws_context_filter")

# ── Paths ──
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_PKL_PATH = os.path.join(_MODELS_DIR, "context_table.pkl")

# ── Global table (loaded once) ──
_table: Optional[dict] = None


# ── Bracket helpers (same as seed) ──
def _hour_bracket(h: int) -> str:
    base = (h // 4) * 4
    return f"{base:02d}-{base+4:02d}"


def _depth_bracket(dr: float) -> str:
    if dr < 2.0:
        return "raso"
    if dr < 4.0:
        return "medio"
    return "fundo"


def _sym_bracket(s: float) -> str:
    if s < 0.5:
        return "assimetrico"
    return "simetrico"


# ── Load ──
def _ensure_loaded():
    global _table
    if _table is not None:
        return
    if not os.path.exists(_PKL_PATH):
        log.warning("context_table.pkl não encontrado — filtro desativado")
        _table = {}
        return
    try:
        with open(_PKL_PATH, "rb") as f:
            _table = pickle.load(f)
        log.info(f"Context table carregada: {len(_table)} buckets")
    except Exception as e:
        log.warning(f"Erro ao carregar context_table.pkl: {e}")
        _table = {}


# ── Lookup with fallback cascade ──
def context_lookup(
    ativo: str,
    direcao: str,
    hour: int,
    depth_ratio: float,
    symmetry: float,
) -> dict:
    """
    Consulta a tabela de contexto com fallback cascade.

    Returns:
        {
            "wr": float,           # Win rate do bucket (0-100)
            "n": int,              # Quantidade de trades no bucket
            "level": int,          # 1=exact, 2=hora, 3=ativo, 4=global
            "action": str,         # "block" | "raise" | "pass" | "boost"
            "nn_threshold": float, # Threshold NN recomendado (0-100)
            "reason": str,         # Razão legível
        }
    """
    _ensure_loaded()

    direcao_lower = direcao.lower()
    hora = _hour_bracket(hour)
    depth = _depth_bracket(depth_ratio)
    sym = _sym_bracket(symmetry)

    # Cascade: L1 → L2 → L3 → L4
    candidates = [
        (1, (ativo, direcao_lower, hora, depth, sym), 5),
        (2, (ativo, direcao_lower, hora, "_", "_"), 10),
        (3, (ativo, direcao_lower, "_", "_", "_"), 20),
        (4, ("_ALL_", direcao_lower, "_", "_", "_"), 1),
    ]

    for level, key, min_n in candidates:
        entry = _table.get(key)
        if entry and entry["total"] >= min_n:
            wr = entry["wr"]
            n = entry["total"]
            action, threshold, reason = _decide(wr, n, level, key)
            return {
                "wr": wr,
                "n": n,
                "level": level,
                "action": action,
                "nn_threshold": threshold,
                "reason": reason,
            }

    # Fallback total — sem dados
    return {
        "wr": 50.0,
        "n": 0,
        "level": 0,
        "action": "pass",
        "nn_threshold": 75.0,
        "reason": "Sem dados no context table",
    }


def _decide(
    wr: float, n: int, level: int, key: tuple
) -> Tuple[str, float, str]:
    """Decide ação com base no WR do bucket."""
    level_names = {1: "exact", 2: "hora", 3: "ativo", 4: "global"}
    lname = level_names.get(level, "?")

    if wr < 80.0:
        return (
            "block",
            100.0,
            f"WR={wr:.1f}% (n={n}, L{level}:{lname}) — geometria historicamente ruim",
        )
    if wr < 85.0:
        return (
            "raise",
            85.0,
            f"WR={wr:.1f}% (n={n}, L{level}:{lname}) — exige NN≥85%",
        )
    if wr < 90.0:
        return (
            "pass",
            75.0,
            f"WR={wr:.1f}% (n={n}, L{level}:{lname}) — padrão OK",
        )
    # WR >= 90%
    return (
        "boost",
        68.0,
        f"WR={wr:.1f}% (n={n}, L{level}:{lname}) — geometria forte",
    )


def format_context_log(result: dict) -> str:
    """Formata resultado para log do bot."""
    action_icons = {
        "block": "⛔",
        "raise": "⚠️",
        "pass": "📊",
        "boost": "🚀",
    }
    icon = action_icons.get(result["action"], "❓")
    return (
        f"{icon} CTX L{result['level']}: "
        f"WR={result['wr']:.1f}% n={result['n']} → "
        f"{result['action'].upper()} (NN≥{result['nn_threshold']:.0f}%) | "
        f"{result['reason']}"
    )
