# -*- coding: utf-8 -*-
"""
RAG Trade Memory — Memória de Trades para IA Generativa
════════════════════════════════════════════════════════
Armazena resultados de trades (WIN/LOSS) com contexto completo.
Quando o GPT guard é chamado, busca trades SIMILARES no histórico
e injeta no prompt para que o GPT aprenda com experiências passadas.

Similaridade:
  - Mesmo ativo
  - Mesma direção (CALL/PUT)
  - RS price próximo (±2 ATR)
  - Condições de mercado similares (consolidação, volatilidade)

Armazenamento: JSON por ativo em pasta ws_trade_memory/
"""

import os
import json
import time
import logging
import threading

log = logging.getLogger("WS_BULLEX")

# ── Diretório de memória ──
_MEMORY_DIR = os.path.join(
    os.getenv("APPDATA", os.path.expanduser("~")),
    "WsTrader", "trade_memory"
)
os.makedirs(_MEMORY_DIR, exist_ok=True)

_MAX_PER_ASSET = 200  # máximo de trades por ativo (mais dados = RAG melhor)
_MAX_RETRIEVE = 8     # máximo de trades similares no prompt
_lock = threading.Lock()


def _asset_file(ativo: str) -> str:
    safe = ativo.replace("/", "_").replace("\\", "_")
    return os.path.join(_MEMORY_DIR, f"{safe}.json")


def save_trade(
    ativo: str,
    direcao: str,
    result: str,         # "win" | "loss" | "tie"
    profit: float,
    entry_price: float,
    rs_price: float,
    neckline: float,
    atr_val: float,
    nn_score: float,
    gpt_confidence: int,
    gpt_exp: int,
    consolidacao: bool,
    pat_type: str,       # "DOUBLE_TOP" | "DOUBLE_BOTTOM"
    depth: float = 0.0,
    symmetry: float = 0.0,
    span: int = 0,
    depth_ratio: float = 0.0,
    hour: int = -1,
):
    """Salva trade na memória RAG após resultado."""
    record = {
        "ts": time.time(),
        "ativo": ativo,
        "dir": direcao,
        "result": result,
        "profit": round(profit, 2),
        "entry_price": round(entry_price, 6),
        "rs_price": round(rs_price, 6),
        "neckline": round(neckline, 6),
        "atr": round(atr_val, 6),
        "nn_score": round(nn_score, 3),
        "gpt_conf": gpt_confidence,
        "gpt_exp": gpt_exp,
        "consolidacao": consolidacao,
        "pat_type": pat_type,
        "depth": round(depth, 6),
        "symmetry": round(symmetry, 3),
        "span": span,
        "depth_ratio": round(depth_ratio, 3),
        "hour": hour,
    }

    fpath = _asset_file(ativo)
    with _lock:
        trades = []
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    trades = json.load(f)
            except (json.JSONDecodeError, Exception):
                trades = []

        trades.append(record)
        if len(trades) > _MAX_PER_ASSET:
            trades = trades[-_MAX_PER_ASSET:]

        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(trades, f, ensure_ascii=False, indent=1)

    log.info(f"  💾 RAG: Memória salva {ativo} {direcao} {result} (total={len(trades)})")


def retrieve_similar(
    ativo: str,
    direcao: str,
    rs_price: float,
    atr_val: float,
    consolidacao: bool,
    depth_ratio: float = 0.0,
    symmetry: float = 0.0,
    span: int = 0,
    hour: int = -1,
) -> list[dict]:
    """Busca trades similares na memória para contexto RAG.

    Critérios de similaridade:
    1. Mesmo ativo (obrigatório)
    2. Mesma direção (obrigatório)
    3. RS price próximo (±2 ATR)
    4. Condições similares (consolidação, geometria, horário)

    Retorna lista ordenada por relevância (mais recentes primeiro).
    """
    fpath = _asset_file(ativo)
    if not os.path.exists(fpath):
        return []

    with _lock:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                trades = json.load(f)
        except (json.JSONDecodeError, Exception):
            return []

    similar = []
    for t in trades:
        # Filtro obrigatório: mesma direção
        if t.get("dir") != direcao:
            continue

        # Similaridade: RS price próximo (±2 ATR)
        t_rs = t.get("rs_price", 0)
        dist = abs(t_rs - rs_price)
        t_atr = t.get("atr", atr_val)
        threshold = max(atr_val, t_atr) * 2
        if dist > threshold:
            continue

        # Score de relevância (mais recente + mesma condição = melhor)
        score = 0.0
        age_hours = (time.time() - t.get("ts", 0)) / 3600
        if age_hours < 6:
            score += 4      # trade muito recente
        elif age_hours < 24:
            score += 3      # últimas 24h
        elif age_hours < 72:
            score += 2      # últimos 3 dias
        else:
            score += 1

        if t.get("consolidacao") == consolidacao:
            score += 2      # mesma condição de mercado

        # RS muito próximo = mais relevante
        if dist < atr_val * 0.5:
            score += 2      # mesmo nível de preço

        # Geometria similar (depth_ratio, symmetry, span)
        t_dr = t.get("depth_ratio", 0)
        if depth_ratio > 0 and t_dr > 0 and abs(t_dr - depth_ratio) < 0.5:
            score += 1.5    # profundidade similar
        t_sym = t.get("symmetry", 0)
        if symmetry > 0 and t_sym > 0 and abs(t_sym - symmetry) < 0.2:
            score += 1      # simetria similar
        t_span = t.get("span", 0)
        if span > 0 and t_span > 0 and abs(t_span - span) <= 8:
            score += 1      # span similar

        # Mesmo horário (± 1 hora)
        t_hour = t.get("hour", -1)
        if hour >= 0 and t_hour >= 0:
            h_diff = min(abs(t_hour - hour), 24 - abs(t_hour - hour))
            if h_diff <= 1:
                score += 1.5  # horário muito similar

        similar.append({**t, "_score": score})

    # Ordenar: maior score primeiro, depois mais recente
    similar.sort(key=lambda x: (-x["_score"], -x.get("ts", 0)))
    return similar[:_MAX_RETRIEVE]


def format_memory_for_prompt(similar_trades: list[dict]) -> str:
    """Formata trades similares como texto para injetar no prompt GPT."""
    if not similar_trades:
        return ""

    wins = sum(1 for t in similar_trades if t["result"] == "win")
    losses = sum(1 for t in similar_trades if t["result"] == "loss")

    lines = [
        "═══ MEMÓRIA RAG: TRADES SIMILARES ANTERIORES ═══",
        f"Encontrados {len(similar_trades)} trades similares ({wins}W / {losses}L):",
    ]

    for i, t in enumerate(similar_trades, 1):
        result_emoji = "✅" if t["result"] == "win" else "❌" if t["result"] == "loss" else "⚪"
        age_h = (time.time() - t.get("ts", 0)) / 3600
        if age_h < 1:
            age_str = f"{int(age_h * 60)}min atrás"
        elif age_h < 24:
            age_str = f"{int(age_h)}h atrás"
        else:
            age_str = f"{int(age_h / 24)}d atrás"

        consol = "SIM" if t.get("consolidacao") else "NÃO"
        dr_str = f"Depth={t.get('depth_ratio', 0):.1f}ATR" if t.get("depth_ratio") else ""
        sym_str = f"Sym={t.get('symmetry', 0):.0%}" if t.get("symmetry") else ""
        span_str = f"Span={t.get('span', 0)}" if t.get("span") else ""
        geo_parts = [p for p in [dr_str, sym_str, span_str] if p]
        geo_str = f" | {' '.join(geo_parts)}" if geo_parts else ""
        hour_str = f" H={t.get('hour')}h" if t.get("hour", -1) >= 0 else ""
        lines.append(
            f"  {i}. {result_emoji} {t['result'].upper()} | RS={t.get('rs_price', 0):.5f} | "
            f"NN={t.get('nn_score', 0):.0%} | GPT_conf={t.get('gpt_conf', 0)}% | "
            f"EXP={t.get('gpt_exp', 2)}min | Consol={consol}{geo_str}{hour_str} | {age_str}"
        )

    # Adicionar insight
    if losses > wins:
        lines.append(
            f"⚠️ ATENÇÃO: {losses} LOSSES de {len(similar_trades)} trades neste nível — considere REJECT ou CONFIDENCE baixa."
        )
    elif wins > losses:
        lines.append(
            f"✅ HISTÓRICO FAVORÁVEL: {wins} WINS de {len(similar_trades)} trades neste nível."
        )

    # WR do nível (dado estatístico concreto)
    total = wins + losses
    if total >= 3:
        wr = wins / total * 100
        lines.append(f"📊 WR NESTE NÍVEL: {wr:.0f}% ({wins}W / {losses}L de {total} trades)")
        if wr < 40:
            lines.append("🔴 NÍVEL PERIGOSO: WR < 40% — forte indicação de REJECT")
        elif wr >= 70:
            lines.append("🟢 NÍVEL CONSISTENTE: WR ≥ 70% — confiança alta para APPROVE")

    return "\n".join(lines)


def rag_should_block(similar_trades: list[dict], min_sample: int = 3) -> dict:
    """Verifica se a memória RAG recomenda BLOQUEAR a entrada.

    Se temos 3+ trades similares e WR < 35% → bloqueia automaticamente.
    Não precisa do GPT — é uma decisão puramente estatística.

    Returns:
        {"block": bool, "reason": str, "wr": float, "sample": int}
    """
    if not similar_trades or len(similar_trades) < min_sample:
        return {"block": False, "reason": "amostra insuficiente", "wr": 0, "sample": len(similar_trades or [])}

    wins = sum(1 for t in similar_trades if t.get("result") == "win")
    losses = sum(1 for t in similar_trades if t.get("result") == "loss")
    total = wins + losses
    if total < min_sample:
        return {"block": False, "reason": "poucos resultados", "wr": 0, "sample": total}

    wr = wins / total * 100
    if wr < 35:
        return {
            "block": True,
            "reason": f"RAG: WR={wr:.0f}% ({wins}W/{losses}L) neste nível — BLOQUEADO",
            "wr": round(wr, 1),
            "sample": total,
        }
    return {"block": False, "reason": f"RAG ok: WR={wr:.0f}%", "wr": round(wr, 1), "sample": total}


def get_memory_stats(ativo: str = None) -> dict:
    """Retorna estatísticas da memória (para debug/log)."""
    if ativo:
        fpath = _asset_file(ativo)
        if not os.path.exists(fpath):
            return {"total": 0, "wins": 0, "losses": 0}
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                trades = json.load(f)
            wins = sum(1 for t in trades if t.get("result") == "win")
            losses = sum(1 for t in trades if t.get("result") == "loss")
            return {"total": len(trades), "wins": wins, "losses": losses}
        except Exception:
            return {"total": 0, "wins": 0, "losses": 0}

    # Stats globais
    total, wins, losses = 0, 0, 0
    if os.path.exists(_MEMORY_DIR):
        for fname in os.listdir(_MEMORY_DIR):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(_MEMORY_DIR, fname), "r", encoding="utf-8") as f:
                        trades = json.load(f)
                    total += len(trades)
                    wins += sum(1 for t in trades if t.get("result") == "win")
                    losses += sum(1 for t in trades if t.get("result") == "loss")
                except Exception:
                    pass
    return {"total": total, "wins": wins, "losses": losses}
