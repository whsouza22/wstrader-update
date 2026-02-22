# -*- coding: utf-8 -*-
"""
AI LOSS MEMORY — IA Generativa que analisa cada LOSS e salva o motivo.

🧠 OBJETIVO: Quando uma operação dá LOSS, a IA analisa TODO o contexto
   (setup, indicadores, mercado, tendência, zona S/R, candle) e gera
   uma explicação em linguagem natural do POR QUÊ perdeu.

📁 Salva em ws_loss_memory.json para análise posterior.
   Cada entrada contém:
   - timestamp, ativo, direção, pnl
   - contexto completo do setup
   - DIAGNÓSTICO: explicação gerada pela IA do motivo do LOSS
   - SUGESTÃO: o que poderia ser melhorado no código
   - PADRÃO: classificação do tipo de erro

💡 COMO USAR DEPOIS:
   - Abrir ws_loss_memory.json e ler os diagnósticos
   - Identificar padrões recorrentes (ex: "contra tendência" aparece 15x)
   - Ajustar o código baseado nos padrões mais frequentes
"""

import json
import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

log = logging.getLogger("WS_AUTO_AI")

# Arquivo onde salva a memória de LOSSes
LOSS_MEMORY_FILE = os.path.join(os.path.dirname(__file__), "ws_loss_memory.json")

# Máximo de registros (evitar arquivo gigante)
MAX_LOSS_RECORDS = 500


def _load_memory() -> List[Dict]:
    """Carrega memória de LOSSes do arquivo JSON."""
    try:
        if os.path.exists(LOSS_MEMORY_FILE):
            with open(LOSS_MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
    except Exception as e:
        log.warning(f"[LOSS_MEMORY] Erro ao carregar: {e}")
    return []


def _save_memory(records: List[Dict]):
    """Salva memória de LOSSes no arquivo JSON."""
    try:
        # Limitar tamanho
        if len(records) > MAX_LOSS_RECORDS:
            records = records[-MAX_LOSS_RECORDS:]
        with open(LOSS_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"[LOSS_MEMORY] Erro ao salvar: {e}")


# ══════════════════════════════════════════════════════════════
# MOTOR DE DIAGNÓSTICO — Analisa o contexto e gera explicação
# ══════════════════════════════════════════════════════════════

def _diagnose_momentum(setup: Dict) -> Optional[str]:
    """Analisa se o momentum estava contra."""
    reasons = setup.get("reasons", [])
    momentum_contra = any("CONTRA" in str(r) and "momentum" in str(r).lower() for r in reasons)
    breakout_risk = setup.get("breakout_risk", "low")
    
    if momentum_contra or breakout_risk in ("high", "critical"):
        return "MOMENTUM_CONTRA: Preço chegava na zona com força (velas de corpo cheio na direção oposta). A zona provavelmente foi rompida."
    return None


def _diagnose_trend(setup: Dict) -> Optional[str]:
    """Analisa se entrou contra a tendência."""
    is_counter = setup.get("is_counter_trend", False)
    is_counter_macro = setup.get("is_counter_macro", False)
    macro_str = setup.get("macro_trend_strength", 0)
    trend_dir = setup.get("macro_trend_dir", "neutral")
    direction = setup.get("dir", "")
    
    if is_counter_macro and macro_str > 0.40:
        return (f"CONTRA_TENDENCIA_FORTE: Entrou {direction} contra macro tendência "
                f"'{trend_dir}' com força {macro_str:.2f}. "
                f"Tendência forte raramente reverte em zona S/R simples.")
    if is_counter_macro and macro_str > 0.20:
        return (f"CONTRA_TENDENCIA: Entrou {direction} contra macro tendência "
                f"'{trend_dir}' (força {macro_str:.2f}).")
    if is_counter:
        return (f"CONTRA_MICRO_TREND: Entrou {direction} contra tendência de curto prazo. "
                f"Preço em movimento sem dar sinais claros de reversão.")
    return None


def _diagnose_rejection(setup: Dict) -> Optional[str]:
    """Analisa se tinha candle de rejeição."""
    candle_pattern = setup.get("candle_pattern", "none")
    candle_strength = setup.get("candle_strength", 0)
    has_rej = candle_pattern not in ("none", "doji_tiny", "")
    
    if not has_rej:
        return ("SEM_REJEICAO: Entrou sem candle de rejeição na zona S/R. "
                "Sem confirmação de que compradores/vendedores estão defendendo a zona.")
    if candle_strength < 0.40:
        return (f"REJEICAO_FRACA: Candle '{candle_pattern}' com qualidade baixa ({candle_strength:.2f}). "
                f"Rejeição existia mas era fraca demais para confirmar reversão.")
    return None


def _diagnose_zone(setup: Dict) -> Optional[str]:
    """Analisa qualidade da zona S/R."""
    touches = setup.get("sr_touches", 0)
    proximity = setup.get("sr_proximity", 0)
    reasons = setup.get("reasons", [])
    
    if touches <= 3:
        return (f"ZONA_FRACA: Zona S/R com apenas {touches} toques. "
                f"Zonas com 3 toques são o mínimo — pouca confiabilidade.")
    if proximity > 0.35:
        return (f"ZONA_LONGE: Preço estava a {proximity:.2f} ATR da zona. "
                f"Quanto mais longe da zona, menor a probabilidade de bounce.")
    
    is_old = any("zona_antiga" in str(r) for r in reasons)
    if is_old:
        return "ZONA_ANTIGA: Zona S/R formada há mais de 120 velas. Zonas antigas perdem força."
    return None


def _diagnose_market_quality(setup: Dict) -> Optional[str]:
    """Analisa qualidade do mercado."""
    mkt = setup.get("market_quality", 0.5)
    context = setup.get("context", "neutro")
    
    if mkt < 0.40:
        return (f"MERCADO_RUIM: Market quality muito baixo ({mkt:.2f}). "
                f"Contexto '{context}' indica mercado desfavorável para operar.")
    if mkt < 0.55:
        return (f"MERCADO_NEUTRO: Market quality medíocre ({mkt:.2f}). "
                f"Mercado sem convicção clara — sinais mistos.")
    return None


def _diagnose_score(setup: Dict) -> Optional[str]:
    """Analisa se o score era baixo demais."""
    score = setup.get("score", 0)
    
    if score < 0.52:
        return (f"SCORE_BAIXO: Score do sinal foi apenas {score:.2f} — "
                f"muito próximo do mínimo. Sinais fracos tem baixa taxa de acerto.")
    return None


def _diagnose_candle_features(setup: Dict) -> Optional[str]:
    """Analisa features do candle."""
    body_ratio = setup.get("candle_body_ratio", 0.5)
    body_strength = setup.get("candle_body_strength", 0)
    ret1 = setup.get("candle_ret1", 0)
    ret3 = setup.get("candle_ret3", 0)
    direction = setup.get("dir", "")
    
    # Candle de corpo grande na direção errada
    if direction == "CALL" and ret1 < -0.002:
        return (f"CANDLE_BEARISH: Último candle tinha retorno negativo ({ret1*100:.3f}%). "
                f"Entrou CALL com candle caindo.")
    if direction == "PUT" and ret1 > 0.002:
        return (f"CANDLE_BULLISH: Último candle tinha retorno positivo ({ret1*100:.3f}%). "
                f"Entrou PUT com candle subindo.")
    
    if body_ratio < 0.25:
        return (f"CANDLE_INDECISO: Candle com corpo muito pequeno ({body_ratio:.2f}). "
                f"Doji/indecisão — mercado sem direção clara.")
    return None


def _diagnose_confluence(setup: Dict) -> Optional[str]:
    """Analisa confluência."""
    conf = setup.get("confluence_count", 1)
    
    if conf <= 1:
        return ("CONFLUENCIA_BAIXA: Apenas 1 confluência (S/R sozinho). "
                "Sem rejeição, sem tendência a favor, sem zona forte — sinal fraco.")
    return None


def _classify_pattern(diagnostics: List[str]) -> str:
    """Classifica o padrão de erro baseado nos diagnósticos."""
    text = " ".join(diagnostics).upper()
    
    if "CONTRA_TENDENCIA_FORTE" in text:
        return "contra_tendencia_forte"
    if "CONTRA_TENDENCIA" in text or "CONTRA_MICRO" in text:
        return "contra_tendencia"
    if "MOMENTUM_CONTRA" in text:
        return "breakout_rompimento"
    if "SEM_REJEICAO" in text and "ZONA_FRACA" in text:
        return "sinal_fraco_sem_confirmacao"
    if "SEM_REJEICAO" in text:
        return "sem_confirmacao_candle"
    if "MERCADO_RUIM" in text:
        return "mercado_desfavoravel"
    if "SCORE_BAIXO" in text:
        return "score_insuficiente"
    if "ZONA_FRACA" in text or "ZONA_LONGE" in text:
        return "zona_sr_fraca"
    if "CONFLUENCIA_BAIXA" in text:
        return "pouca_confluencia"
    if "CANDLE_BEARISH" in text or "CANDLE_BULLISH" in text:
        return "candle_contra_direcao"
    return "indefinido"


def _generate_suggestion(pattern: str, diagnostics: List[str]) -> str:
    """Gera sugestão de melhoria baseada no padrão."""
    suggestions = {
        "contra_tendencia_forte": (
            "SUGESTÃO: Aumentar penalidade para trades contra macro tendência forte (>0.40). "
            "Considerar BLOQUEAR entrada quando macro_trend_strength > 0.50 e é contra."
        ),
        "contra_tendencia": (
            "SUGESTÃO: Aumentar SCORE_MACRO_AGAINST ou adicionar filtro que exige "
            "rejeição forte quando contra tendência."
        ),
        "breakout_rompimento": (
            "SUGESTÃO: Implementar verificação de momentum antes de entrar. "
            "Se 3+ velas de corpo cheio indo para a zona, aguardar confirmação de bounce."
        ),
        "sinal_fraco_sem_confirmacao": (
            "SUGESTÃO: Exigir pelo menos 2 confluências (rejeição + zona forte) "
            "para entrar. Score mínimo deveria ser mais alto."
        ),
        "sem_confirmacao_candle": (
            "SUGESTÃO: Exigir candle de rejeição (hammer/engulfing) para confirmar "
            "que a zona está sendo defendida. Sem rejeição = sem entrada."
        ),
        "mercado_desfavoravel": (
            "SUGESTÃO: Aumentar threshold de market_quality mínimo. "
            "Não operar quando market_quality < 0.50."
        ),
        "score_insuficiente": (
            "SUGESTÃO: Aumentar score mínimo para entrada (atualmente 0.48). "
            "Scores próximos do mínimo têm taxa de acerto baixa."
        ),
        "zona_sr_fraca": (
            "SUGESTÃO: Exigir mínimo de 4 toques para zona válida, "
            "ou reduzir SR_PROXIMITY_ATR para operar mais perto da zona."
        ),
        "pouca_confluencia": (
            "SUGESTÃO: Exigir mínimo de 2 confluências (S/R + pelo menos um de: "
            "rejeição, tendência a favor, zona forte)."
        ),
        "candle_contra_direcao": (
            "SUGESTÃO: Verificar direção do último candle antes de entrar. "
            "Se candle é bearish → não entrar CALL. Se bullish → não entrar PUT."
        ),
        "indefinido": (
            "SUGESTÃO: Caso atípico. Revisar contexto completo manualmente. "
            "Pode ser volatilidade OTC ou manipulação de mercado."
        ),
    }
    return suggestions.get(pattern, suggestions["indefinido"])


# ══════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL: Analisar LOSS e salvar diagnóstico
# ══════════════════════════════════════════════════════════════

def analyze_and_save_loss(
    ativo: str,
    direction: str,
    pnl: float,
    setup: Dict[str, Any],
    ai_stats: Optional[Dict] = None,
    extra_info: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Analisa um LOSS, gera diagnóstico inteligente e salva no JSON.
    
    Args:
        ativo: nome do ativo (ex: "EURUSD-OTC")
        direction: "CALL" ou "PUT"
        pnl: resultado financeiro (negativo)
        setup: dict completo do setup/sinal
        ai_stats: estatísticas da IA (opcional)
        extra_info: informações extras como saldo, broker, etc (opcional)
    
    Returns:
        Dict com o diagnóstico completo
    """
    now = datetime.now()
    
    # ── Rodar todos os diagnósticos ──
    diagnostics = []
    
    checks = [
        _diagnose_momentum,
        _diagnose_trend,
        _diagnose_rejection,
        _diagnose_zone,
        _diagnose_market_quality,
        _diagnose_score,
        _diagnose_candle_features,
        _diagnose_confluence,
    ]
    
    for check_fn in checks:
        result = check_fn(setup)
        if result:
            diagnostics.append(result)
    
    # Se nenhum diagnóstico encontrado
    if not diagnostics:
        diagnostics.append(
            "INDEFINIDO: Nenhum problema claro identificado. "
            "Pode ser volatilidade aleatória do mercado OTC ou manipulação."
        )
    
    # ── Classificar padrão e gerar sugestão ──
    pattern = _classify_pattern(diagnostics)
    suggestion = _generate_suggestion(pattern, diagnostics)
    
    # ── Montar registro completo ──
    record = {
        "timestamp": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "ativo": ativo,
        "direction": direction,
        "pnl": round(pnl, 2),
        # Contexto do setup
        "score": round(setup.get("score", 0), 4),
        "market_quality": round(setup.get("market_quality", 0), 4),
        "sr_touches": setup.get("sr_touches", 0),
        "sr_proximity_atr": round(setup.get("sr_proximity", 0), 4),
        "candle_pattern": setup.get("candle_pattern", "none"),
        "candle_strength": round(setup.get("candle_strength", 0), 4),
        "confluence_count": setup.get("confluence_count", 1),
        "breakout_risk": setup.get("breakout_risk", "low"),
        "is_counter_trend": setup.get("is_counter_trend", False),
        "is_counter_macro": setup.get("is_counter_macro", False),
        "macro_trend_dir": setup.get("macro_trend_dir", "neutral"),
        "macro_trend_strength": round(setup.get("macro_trend_strength", 0), 4),
        "trend_dir": setup.get("trend_dir_detected", "neutral"),
        "trend_strength": round(setup.get("trend_strength_detected", 0), 4),
        "setup_type": setup.get("setup_type", "?"),
        "reasons": setup.get("reasons", []),
        # DIAGNÓSTICO DA IA
        "diagnostico": diagnostics,
        "padrao_erro": pattern,
        "sugestao": suggestion,
        # Info extra
        "broker": (extra_info or {}).get("broker", "?"),
        "account_type": (extra_info or {}).get("account_type", "?"),
    }
    
    # ── Salvar no JSON ──
    memory = _load_memory()
    memory.append(record)
    _save_memory(memory)
    
    # ── Log resumido ──
    log.info("=" * 60)
    log.info(f"🧠 [LOSS MEMORY] Diagnóstico para {ativo} {direction}:")
    for d in diagnostics:
        tag = d.split(":")[0] if ":" in d else "INFO"
        log.info(f"   📋 {tag}")
    log.info(f"   🏷️ Padrão: {pattern}")
    log.info(f"   💡 {suggestion[:80]}...")
    log.info(f"   📁 Salvo em ws_loss_memory.json ({len(memory)} registros)")
    log.info("=" * 60)
    
    return record


# ══════════════════════════════════════════════════════════════
# RELATÓRIO: Resumo dos padrões mais frequentes
# ══════════════════════════════════════════════════════════════

def get_loss_summary() -> Dict[str, Any]:
    """
    Gera relatório resumido dos LOSSes para identificar padrões.
    
    Returns:
        {
            "total_losses": int,
            "patterns": {"contra_tendencia": 15, "breakout_rompimento": 8, ...},
            "top_ativos_loss": {"EURUSD-OTC": 12, ...},
            "avg_score_loss": 0.54,
            "avg_mkt_quality_loss": 0.48,
            "suggestions": ["Padrão mais comum: contra_tendencia (15x) — ...", ...]
        }
    """
    memory = _load_memory()
    if not memory:
        return {"total_losses": 0, "patterns": {}, "top_ativos_loss": {},
                "avg_score_loss": 0, "avg_mkt_quality_loss": 0, "suggestions": []}
    
    # Contar padrões
    patterns = {}
    ativos = {}
    scores = []
    mkt_qualities = []
    
    for r in memory:
        p = r.get("padrao_erro", "indefinido")
        patterns[p] = patterns.get(p, 0) + 1
        
        a = r.get("ativo", "?")
        ativos[a] = ativos.get(a, 0) + 1
        
        scores.append(r.get("score", 0))
        mkt_qualities.append(r.get("market_quality", 0))
    
    # Ordenar por frequência
    patterns_sorted = dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True))
    ativos_sorted = dict(sorted(ativos.items(), key=lambda x: x[1], reverse=True))
    
    # Gerar sugestões baseadas nos padrões mais frequentes
    suggestions = []
    for pattern, count in list(patterns_sorted.items())[:3]:
        pct = count / len(memory) * 100
        sug = _generate_suggestion(pattern, [])
        suggestions.append(f"Padrão '{pattern}' ({count}x, {pct:.0f}%): {sug}")
    
    return {
        "total_losses": len(memory),
        "patterns": patterns_sorted,
        "top_ativos_loss": dict(list(ativos_sorted.items())[:10]),
        "avg_score_loss": round(sum(scores) / len(scores), 4) if scores else 0,
        "avg_mkt_quality_loss": round(sum(mkt_qualities) / len(mkt_qualities), 4) if mkt_qualities else 0,
        "suggestions": suggestions,
    }


def print_loss_report():
    """Imprime relatório de LOSSes no console."""
    summary = get_loss_summary()
    
    if summary["total_losses"] == 0:
        print("Nenhum LOSS registrado ainda.")
        return
    
    print("\n" + "=" * 70)
    print(f"📊 RELATÓRIO DE LOSSES — {summary['total_losses']} operações perdidas")
    print("=" * 70)
    
    print(f"\n📈 Score médio nos LOSSes: {summary['avg_score_loss']:.4f}")
    print(f"📈 Market Quality médio: {summary['avg_mkt_quality_loss']:.4f}")
    
    print(f"\n🏷️ PADRÕES DE ERRO:")
    for pattern, count in summary["patterns"].items():
        pct = count / summary["total_losses"] * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"   {bar} {pattern}: {count}x ({pct:.0f}%)")
    
    print(f"\n📉 TOP ATIVOS COM MAIS LOSSES:")
    for ativo, count in list(summary["top_ativos_loss"].items())[:5]:
        print(f"   • {ativo}: {count}x")
    
    print(f"\n💡 SUGESTÕES PRIORITÁRIAS:")
    for i, sug in enumerate(summary["suggestions"], 1):
        print(f"   {i}. {sug}")
    
    print("=" * 70 + "\n")
