"""
AI Claude Calibrator - Usa Claude Opus para analisar resultados do backtest
e recalibrar parâmetros automaticamente.

Fluxo:
1. Após cada backtest, envia os resultados para Claude
2. Claude analisa padrões de WIN vs LOSS
3. Retorna sugestões de novos thresholds
4. Aplica automaticamente os ajustes

Requer: pip install anthropic
Configurar: WS_CLAUDE_API_KEY no .env ou variável de ambiente
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple

log = logging.getLogger("WS_AUTO_AI")

# ===================== CONFIGURAÇÃO =====================
try:
    from config_keys import CLAUDE_API_KEY_2 as _KEY
    CLAUDE_API_KEY = _KEY
except ImportError:
    CLAUDE_API_KEY = os.getenv("WS_CLAUDE_API_KEY", "")
CLAUDE_MODEL = os.getenv("WS_CLAUDE_MODEL", "claude-sonnet-4-20250514")  # Modelo padrão (custo-benefício)
CLAUDE_MAX_TOKENS = int(os.getenv("WS_CLAUDE_MAX_TOKENS", "2000"))
CLAUDE_TIMEOUT = int(os.getenv("WS_CLAUDE_TIMEOUT", "30"))  # Timeout em segundos
CLAUDE_CALIBRATE_ON = os.getenv("WS_CLAUDE_CALIBRATE", "1").strip() == "1"  # Ativado por padrão

# Limites de segurança para parâmetros (Claude não pode ir além destes)
PARAM_LIMITS = {
    "GATE_MIN_SCORE": (0.45, 0.85),
    "GATE_SOFT_SCORE": (0.35, 0.75),
    "GATE_CONTEXT_VERY_BAD": (0.30, 0.60),
    "MIN_ENTRY_EFF": (0.30, 0.70),
    "MIN_EFF_A": (0.30, 0.65),
    "ENS_MIN_CTX_RUIM": (0.55, 0.75),
    "ENS_MIN_CTX_MED": (0.50, 0.70),
    "ENS_MIN_CTX_BOM": (0.45, 0.65),
    "MIN_CONFLUENCE": (1, 4),
    "MAX_WICK_AGAINST": (0.50, 0.80),
    "RETR_MIN": (0.05, 0.25),
    "RETR_MAX": (0.65, 0.95),
}

# Arquivo para salvar histórico de calibrações
CALIBRATION_HISTORY_FILE = os.getenv("WS_CALIB_HISTORY", "ws_claude_calibrations.json")

# ===================== CLIENTE ANTHROPIC =====================
_claude_client = None
CLAUDE_AVAILABLE = False

try:
    import anthropic
    if CLAUDE_API_KEY and len(CLAUDE_API_KEY) > 10:
        _claude_client = anthropic.Anthropic(
            api_key=CLAUDE_API_KEY,
            timeout=float(CLAUDE_TIMEOUT)
        )
        CLAUDE_AVAILABLE = True
        log.info(f"[CLAUDE] ✅ API configurada | Modelo: {CLAUDE_MODEL}")
    else:
        log.info("[CLAUDE] ⚠️ API key não configurada. Definir WS_CLAUDE_API_KEY para ativar.")
except ImportError:
    log.info("[CLAUDE] ⚠️ Biblioteca 'anthropic' não instalada. Instalar com: pip install anthropic")
except Exception as e:
    log.warning(f"[CLAUDE] Erro ao inicializar: {e}")


def _build_backtest_prompt(backtest_result: Dict[str, Any],
                            current_params: Dict[str, Any],
                            trade_history: List[Dict[str, Any]],
                            loss_patterns: List[Dict[str, Any]]) -> str:
    """
    Constrói o prompt para Claude analisar os resultados do backtest.
    """
    total_sinais = backtest_result.get("sinais", 0)
    wins = backtest_result.get("wins", 0)
    losses = backtest_result.get("losses", 0)
    taxa = backtest_result.get("taxa_acerto", 0.0)
    filtros_ativo = backtest_result.get("filtros_por_ativo", {})
    
    # Formatar trades recentes
    trades_str = ""
    if trade_history:
        for t in trade_history[-20:]:  # Últimos 20 trades
            result = "WIN" if t.get("profit", 0) > 0 else "LOSS"
            trades_str += (
                f"  {t.get('ativo','?')} {t.get('dir','?')} | {result} ${t.get('profit',0):.2f} | "
                f"score={t.get('score',0):.2f} ctx={t.get('ctx',0):.2f} "
                f"entry_conf={t.get('entry_conf',0):.2f} effA={t.get('effA',0):.2f} "
                f"lt={t.get('lt_confluence',0):.2f} sr={t.get('sr_proximity',0):.2f} "
                f"ens={t.get('ensemble',0):.2f}\n"
            )
    
    # Formatar padrões de LOSS
    losses_str = ""
    if loss_patterns:
        for lp in loss_patterns[-10:]:
            losses_str += (
                f"  {lp.get('ativo','?')} {lp.get('dir','?')} | "
                f"score={lp.get('score',0):.2f} ctx={lp.get('ctx',0):.2f} "
                f"entry_conf={lp.get('entry_conf',0):.2f} effA={lp.get('effA',0):.2f} "
                f"lt={lp.get('lt_confluence',0):.2f} sr={lp.get('sr_proximity',0):.2f} "
                f"ens={lp.get('ensemble',0):.2f} | "
                f"razão: {lp.get('reason','?')}\n"
            )
    
    # Formatar filtros por ativo
    filtros_str = ""
    for ativo, filtro in filtros_ativo.items():
        filtros_str += f"  {ativo}: taxa={filtro.get('taxa',0)*100:.0f}% ctx≥{filtro.get('min_ctx',0):.2f} score≥{filtro.get('min_score',0):.2f}\n"
    
    # Formatar parâmetros atuais
    params_str = json.dumps(current_params, indent=2, ensure_ascii=False)
    
    # Limites de segurança
    limits_str = json.dumps(PARAM_LIMITS, indent=2)
    
    prompt = f"""Você é um especialista em trading de opções binárias OTC (M1 - 1 minuto).
Analise os resultados do backtest e trades recentes para recalibrar os parâmetros do sistema.

## RESULTADO DO BACKTEST ATUAL
- Sinais analisados: {total_sinais}
- WINs: {wins} | LOSSes: {losses}
- Taxa de acerto: {taxa*100:.1f}%

## FILTROS POR ATIVO
{filtros_str if filtros_str else "  Nenhum dado disponível"}

## TRADES RECENTES (últimos 20)
{trades_str if trades_str else "  Nenhum trade registrado"}

## PADRÕES DE LOSS IDENTIFICADOS
{losses_str if losses_str else "  Nenhum LOSS registrado"}

## PARÂMETROS ATUAIS
{params_str}

## LIMITES DE SEGURANÇA (não ultrapassar)
{limits_str}

## REGRAS DE ANÁLISE
1. Se taxa < 50%, os filtros estão MUITO relaxados - aperte
2. Se taxa > 65%, pode relaxar levemente para pegar mais oportunidades
3. Se WINs têm score/ctx/effA muito maiores que LOSSes, use esses valores como referência
4. Se LOSSes acontecem com ctx<0.40, aumente GATE_CONTEXT_VERY_BAD
5. Se LOSSes acontecem com entry_conf<0.50, aumente os thresholds de entrada
6. Se LOSSes acontecem sem trendline (lt=0), mantenha MIN_CONFLUENCE alto
7. Se LOSSes acontecem sem S/R (sr=0), aumente exigência de ensemble
8. Priorize EVITAR LOSS sobre fazer mais trades
9. Ajustes devem ser INCREMENTAIS (max ±0.05 por parâmetro por vez)
10. Se taxa está entre 50-65%, faça ajustes MÍNIMOS

## RESPOSTA OBRIGATÓRIA (JSON puro, sem markdown)
Retorne APENAS um JSON com esta estrutura exata:
{{
    "analysis": "Breve análise do mercado (1-2 frases)",
    "main_problem": "Principal causa dos LOSSes",
    "adjustments": {{
        "PARAM_NAME": new_value,
        ...
    }},
    "confidence": 0.0 a 1.0,
    "reasoning": "Por que cada ajuste foi feito"
}}

IMPORTANTE: Retorne APENAS o JSON, sem texto extra, sem ```json, sem explicações fora do JSON."""

    return prompt


def calibrate_with_claude(backtest_result: Dict[str, Any],
                          current_params: Dict[str, Any],
                          trade_history: Optional[List[Dict[str, Any]]] = None,
                          loss_patterns: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """
    Envia resultados do backtest para Claude e recebe sugestões de calibração.
    
    Args:
        backtest_result: Resultado do backtest (sinais, wins, losses, taxa)
        current_params: Parâmetros atuais do sistema
        trade_history: Histórico de trades recentes
        loss_patterns: Padrões de LOSS identificados
    
    Returns:
        Dict com ajustes sugeridos ou None se falhar
    """
    if not CLAUDE_AVAILABLE or not CLAUDE_CALIBRATE_ON:
        return None
    
    if not _claude_client:
        return None
    
    trade_history = trade_history or []
    loss_patterns = loss_patterns or []
    
    prompt = _build_backtest_prompt(backtest_result, current_params, trade_history, loss_patterns)
    
    try:
        log.info("[CLAUDE] 🧠 Enviando resultados para análise...")
        start_time = time.time()
        
        message = _claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        elapsed = time.time() - start_time
        response_text = message.content[0].text.strip()
        
        # Parse do JSON
        # Remove possíveis wrappers de markdown
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])
        
        result = json.loads(response_text)
        
        log.info(f"[CLAUDE] ✅ Análise recebida em {elapsed:.1f}s | Confiança: {result.get('confidence', 0):.0%}")
        log.info(f"[CLAUDE] 📊 Análise: {result.get('analysis', '?')}")
        log.info(f"[CLAUDE] 🔍 Problema: {result.get('main_problem', '?')}")
        
        # Validar e limitar ajustes
        adjustments = result.get("adjustments", {})
        validated = _validate_adjustments(adjustments, current_params)
        result["adjustments"] = validated
        result["elapsed_seconds"] = elapsed
        result["model"] = CLAUDE_MODEL
        result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        
        # Salvar no histórico
        _save_calibration_history(result, backtest_result)
        
        return result
        
    except json.JSONDecodeError as e:
        log.warning(f"[CLAUDE] ⚠️ Resposta inválida (não é JSON): {e}")
        log.debug(f"[CLAUDE] Resposta raw: {response_text[:500]}")
        return None
    except anthropic.APITimeoutError:
        log.warning(f"[CLAUDE] ⚠️ Timeout ({CLAUDE_TIMEOUT}s) - Claude demorou demais")
        return None
    except anthropic.APIError as e:
        log.warning(f"[CLAUDE] ⚠️ Erro API: {e}")
        return None
    except Exception as e:
        log.warning(f"[CLAUDE] ⚠️ Erro inesperado: {e}")
        return None


def _validate_adjustments(adjustments: Dict[str, Any], 
                          current_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida os ajustes sugeridos pela Claude:
    1. Verifica se parâmetro existe
    2. Aplica limites de segurança
    3. Limita mudança incremental (max ±0.05 floats, ±1 ints)
    """
    validated = {}
    
    for param, new_value in adjustments.items():
        # Só aceita parâmetros conhecidos
        if param not in PARAM_LIMITS:
            log.warning(f"[CLAUDE] ⚠️ Parâmetro desconhecido ignorado: {param}")
            continue
        
        limits = PARAM_LIMITS[param]
        min_val, max_val = limits
        
        # Pegar valor atual
        current = current_params.get(param)
        if current is None:
            log.warning(f"[CLAUDE] ⚠️ Parâmetro {param} não tem valor atual, usando sugestão direta")
            validated[param] = max(min_val, min(max_val, new_value))
            continue
        
        # Limitar mudança incremental
        if isinstance(min_val, int):
            # Parâmetro inteiro
            new_value = int(new_value)
            max_change = 1
            delta = max(-max_change, min(max_change, new_value - current))
            final_value = current + delta
        else:
            # Parâmetro float
            new_value = float(new_value)
            max_change = 0.05
            delta = max(-max_change, min(max_change, new_value - current))
            final_value = round(current + delta, 3)
        
        # Aplicar limites absolutos
        final_value = max(min_val, min(max_val, final_value))
        
        if final_value != current:
            validated[param] = final_value
            direction = "↑" if final_value > current else "↓"
            log.info(f"[CLAUDE] 🔧 {param}: {current} → {final_value} ({direction}{abs(final_value - current):.3f})")
        else:
            log.info(f"[CLAUDE] ⏸️ {param}: mantém {current} (sem mudança necessária)")
    
    return validated


def apply_calibration(adjustments: Dict[str, Any], confidence: float = 0.0) -> Dict[str, Tuple[Any, Any]]:
    """
    Aplica os ajustes validados aos parâmetros globais do WS_AUTO_AI.
    Só aplica se confiança >= 0.50.
    
    Returns:
        Dict com parâmetros alterados: {param: (old_value, new_value)}
    """
    import WS_AUTO_AI as ws
    
    if confidence < 0.50:
        log.info(f"[CLAUDE] ⚠️ Confiança baixa ({confidence:.0%}) - ajustes NÃO aplicados (mínimo 50%)")
        return {}
    
    applied = {}
    
    for param, new_value in adjustments.items():
        old_value = getattr(ws, param, None)
        if old_value is not None and old_value != new_value:
            setattr(ws, param, new_value)
            applied[param] = (old_value, new_value)
            log.info(f"[CLAUDE] ✅ APLICADO: {param} = {old_value} → {new_value}")
    
    if applied:
        log.info(f"[CLAUDE] 🎯 {len(applied)} parâmetros recalibrados por Claude")
    else:
        log.info("[CLAUDE] ℹ️ Nenhum ajuste necessário")
    
    return applied


def _save_calibration_history(result: Dict[str, Any], backtest_result: Dict[str, Any]):
    """Salva histórico de calibrações para análise futura."""
    try:
        history = []
        if os.path.exists(CALIBRATION_HISTORY_FILE):
            with open(CALIBRATION_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        
        entry = {
            "timestamp": result.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S")),
            "model": result.get("model", CLAUDE_MODEL),
            "confidence": result.get("confidence", 0),
            "analysis": result.get("analysis", ""),
            "main_problem": result.get("main_problem", ""),
            "adjustments": result.get("adjustments", {}),
            "backtest_taxa": backtest_result.get("taxa_acerto", 0),
            "backtest_sinais": backtest_result.get("sinais", 0),
            "elapsed": result.get("elapsed_seconds", 0)
        }
        
        history.append(entry)
        
        # Manter últimas 100 calibrações
        history = history[-100:]
        
        with open(CALIBRATION_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        log.warning(f"[CLAUDE] Erro ao salvar histórico: {e}")


def get_current_params() -> Dict[str, Any]:
    """
    Lê os parâmetros atuais do WS_AUTO_AI para enviar ao Claude.
    """
    try:
        import WS_AUTO_AI as ws
        return {
            "GATE_MIN_SCORE": getattr(ws, "GATE_MIN_SCORE", 0.60),
            "GATE_SOFT_SCORE": getattr(ws, "GATE_SOFT_SCORE", 0.52),
            "GATE_CONTEXT_VERY_BAD": getattr(ws, "GATE_CONTEXT_VERY_BAD", 0.40),
            "MIN_ENTRY_EFF": getattr(ws, "MIN_ENTRY_EFF", 0.50),
            "MIN_EFF_A": getattr(ws, "MIN_EFF_A", 0.45),
            "ENS_MIN_CTX_RUIM": getattr(ws, "ENS_MIN_CTX_RUIM", 0.65),
            "ENS_MIN_CTX_MED": getattr(ws, "ENS_MIN_CTX_MED", 0.60),
            "ENS_MIN_CTX_BOM": getattr(ws, "ENS_MIN_CTX_BOM", 0.55),
            "MIN_CONFLUENCE": getattr(ws, "MIN_CONFLUENCE", 2),
            "MAX_WICK_AGAINST": getattr(ws, "MAX_WICK_AGAINST", 0.75),
            "RETR_MIN": getattr(ws, "RETR_MIN", 0.05),
            "RETR_MAX": getattr(ws, "RETR_MAX", 0.95),
        }
    except Exception:
        return {}


def get_trade_history_for_claude() -> List[Dict[str, Any]]:
    """
    Lê o histórico de trades recentes para enviar ao Claude.
    """
    try:
        import WS_AUTO_AI as ws
        lgbm_data = getattr(ws, "lgbm_data", [])
        trades = []
        for d in lgbm_data[-30:]:  # Últimos 30
            trades.append({
                "ativo": d.get("ativo", "?"),
                "dir": d.get("dir", "?"),
                "profit": 1.0 if d.get("win", False) else -1.0,
                "score": d.get("score", 0),
                "ctx": d.get("ctx", 0),
                "entry_conf": d.get("entry_conf", 0),
                "effA": d.get("effA", 0),
                "lt_confluence": d.get("lt_conf", 0),
                "sr_proximity": d.get("sr_prox", 0),
                "ensemble": d.get("ensemble", 0),
            })
        return trades
    except Exception:
        return []


def get_loss_patterns_for_claude() -> List[Dict[str, Any]]:
    """
    Lê padrões de LOSS recentes para enviar ao Claude.
    """
    try:
        loss_file = "loss_history.json"
        if not os.path.exists(loss_file):
            return []
        
        with open(loss_file, "r", encoding="utf-8") as f:
            loss_data = json.load(f)
        
        if isinstance(loss_data, list):
            return loss_data[-10:]
        elif isinstance(loss_data, dict):
            return loss_data.get("losses", loss_data.get("history", []))[-10:]
        return []
    except Exception:
        return []


def calibrate_after_backtest(backtest_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Função principal: chama após backtest para calibrar com Claude.
    
    Uso:
        from ai_claude_calibrator import calibrate_after_backtest
        result = calibrate_after_backtest(backtest_result)
    """
    if not CLAUDE_AVAILABLE or not CLAUDE_CALIBRATE_ON:
        return None
    
    # Coletar dados
    current_params = get_current_params()
    trade_history = get_trade_history_for_claude()
    loss_patterns = get_loss_patterns_for_claude()
    
    # Enviar para Claude
    result = calibrate_with_claude(
        backtest_result=backtest_result,
        current_params=current_params,
        trade_history=trade_history,
        loss_patterns=loss_patterns
    )
    
    if result and result.get("adjustments"):
        confidence = result.get("confidence", 0)
        adjustments = result["adjustments"]
        
        # Aplicar ajustes (só se confiança >= 50%)
        applied = apply_calibration(adjustments, confidence)
        result["applied"] = applied
        
        return result
    
    return result
