# -*- coding: utf-8 -*-
"""
AGENTE GPT — Análise de Estrutura de Mercado via ChatGPT
=========================================================
Envia as últimas 40 velas (OHLC) para o ChatGPT com múltiplos
"sub-agentes" especializados. Cada sub-agente analisa um aspecto:

  1. Analista de Tendência    — direção dominante das 40 velas
  2. Analista de Estrutura    — HH/HL (alta) ou LH/LL (baixa)?
  3. Analista de Momentum     — força está aumentando ou esgotando?
  4. Analista de Risco        — há sinais de armadilha / fakeout?

O GPT responde com SIM ou NÃO + motivo em 1 linha.
Se NÃO → a entrada é BLOQUEADA.

Timeout: 8 segundos (não atrasa o bot significativamente).
Fallback: se GPT falhar, retorna SIM (não bloqueia — fail-open).
"""
import os
import json
import time
import logging
import threading
from typing import Any, Dict, Optional, Tuple

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

log = logging.getLogger("WS_AI")

# ══════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════
GPT_TIMEOUT = 8          # segundos máx de espera
GPT_MODEL = "gpt-4o-mini"  # modelo rápido e barato
GPT_MAX_TOKENS = 200     # resposta curta
GPT_CANDLES = 40         # últimas N velas para enviar
GPT_COOLDOWN = 30        # segundos entre chamadas GPT (evitar rate limit)

# Cache simples: não chamar GPT repetidamente para o mesmo ativo
_last_call: Dict[str, float] = {}
_last_result: Dict[str, Tuple[bool, str]] = {}


def _get_api_key() -> Optional[str]:
    """Tenta obter a chave da OpenAI de várias fontes."""
    # 1. Variável de ambiente
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    # 2. config_keys.py
    try:
        from config_keys import OPENAI_API_KEY
        if OPENAI_API_KEY and len(OPENAI_API_KEY) > 20:
            return OPENAI_API_KEY
    except Exception:
        pass
    return None


def _format_candles_for_gpt(df, direction: str, setup: Dict[str, Any]) -> str:
    """Formata as últimas 40 velas em texto compacto para o prompt."""
    tail = df.tail(GPT_CANDLES)
    lines = []
    for i, (idx, row) in enumerate(tail.iterrows()):
        o = round(float(row["open"]), 5)
        h = round(float(row["high"]), 5)
        l = round(float(row["low"]), 5)
        c = round(float(row["close"]), 5)
        body = "▲" if c > o else ("▼" if c < o else "=")
        lines.append(f"{i+1}. O={o} H={h} L={l} C={c} {body}")

    candles_text = "\n".join(lines)

    # Contexto do setup
    trend_dir = setup.get("trend_dir", "?")
    m5_dir = setup.get("m5_trend_dir", "?")
    zone_type = setup.get("zone_type", "?")
    sr_touches = setup.get("sr_touches", 0)
    zone_strength = setup.get("zone_strength", 0)

    return (
        f"Ativo: binário M1\n"
        f"Direção pretendida: {direction}\n"
        f"Tendência M1: {trend_dir}\n"
        f"Tendência M5: {m5_dir}\n"
        f"Zona: {zone_type} ({sr_touches} toques, força {zone_strength:.0%})\n"
        f"\nÚltimas {len(tail)} velas M1 (1-min OHLC):\n{candles_text}"
    )


def _build_prompt(candles_info: str, direction: str) -> str:
    """Monta o system prompt com os 4 sub-agentes."""
    return f"""Você é um comitê de 4 analistas de mercado especializados em opções binárias M1.
Cada analista examina as 40 velas abaixo e vota se a entrada {direction} é segura.

ANALISTAS:
1. TENDÊNCIA: A tendência macro das 40 velas favorece {direction}? Procure Higher Highs/Higher Lows para CALL ou Lower Highs/Lower Lows para PUT.
2. ESTRUTURA: A estrutura de preço está organizada (não caótica)? Existe padrão claro de topos/fundos?
3. MOMENTUM: A força do movimento recente (últimas 5-8 velas) está a favor de {direction}? Ou está esgotando?
4. RISCO: Existe sinal de armadilha? Spike em velas recentes? Reversão violenta iminente?

DADOS:
{candles_info}

REGRAS:
- Se 3 ou 4 analistas votam NÃO → responda NÃO
- Se 3 ou 4 analistas votam SIM → responda SIM
- Seja CONSERVADOR: na dúvida, vote NÃO

Responda EXATAMENTE neste formato JSON:
{{"decision": "SIM" ou "NAO", "votes": {{"tendencia": "SIM/NAO", "estrutura": "SIM/NAO", "momentum": "SIM/NAO", "risco": "SIM/NAO"}}, "reason": "motivo em 1 linha"}}"""


def gpt_analyze_structure(
    df,
    direction: str,
    setup: Dict[str, Any],
    ativo: str = "",
) -> Tuple[bool, str]:
    """
    Chama o ChatGPT para analisar estrutura de mercado.
    
    Args:
        df: DataFrame M1 com OHLC (precisa ter >= 40 linhas)
        direction: "CALL" ou "PUT"
        setup: dict do setup atual (trend_dir, m5_trend_dir etc.)
        ativo: nome do ativo (para logging e cache)
    
    Returns:
        (approved, reason): True se GPT aprova, False se bloqueia
    """
    # ── Failsafe: sem httpx ou sem API key → fail-open ──
    if not _HAS_HTTPX:
        return True, "httpx não instalado — GPT skip"
    
    api_key = _get_api_key()
    if not api_key:
        return True, "sem API key — GPT skip"
    
    # ── Cooldown: não chamar GPT repetidamente para o mesmo ativo ──
    now = time.time()
    cache_key = f"{ativo}_{direction}"
    if cache_key in _last_call and (now - _last_call[cache_key]) < GPT_COOLDOWN:
        if cache_key in _last_result:
            return _last_result[cache_key]
        return True, "GPT cooldown — usando cache"
    
    # ── Verificar se temos velas suficientes ──
    if df is None or len(df) < GPT_CANDLES:
        return True, f"poucas velas ({len(df) if df is not None else 0}) — GPT skip"
    
    # ── Formatar dados e prompt ──
    candles_info = _format_candles_for_gpt(df, direction, setup)
    prompt = _build_prompt(candles_info, direction)
    
    # ── Chamar GPT ──
    try:
        _last_call[cache_key] = now
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GPT_MODEL,
            "messages": [
                {"role": "system", "content": "Você é um analista de mercado binário. Responda APENAS em JSON."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": GPT_MAX_TOKENS,
            "temperature": 0.1,  # deterministico
        }
        
        with httpx.Client(timeout=GPT_TIMEOUT) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
        
        if resp.status_code != 200:
            log.warning(f"[GPT] HTTP {resp.status_code} — fail-open")
            return True, f"GPT HTTP {resp.status_code} — skip"
        
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        
        # ── Parse da resposta ──
        # Limpar possíveis markdown wrappers
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        result = json.loads(content)
        decision = str(result.get("decision", "SIM")).upper().strip()
        reason = str(result.get("reason", "sem motivo"))
        votes = result.get("votes", {})
        
        # Contar votos
        yes_count = sum(1 for v in votes.values() if str(v).upper().strip() == "SIM")
        no_count = sum(1 for v in votes.values() if str(v).upper().strip() in ("NAO", "NÃO", "NO"))
        
        approved = decision == "SIM"
        
        vote_detail = " | ".join(f"{k}={v}" for k, v in votes.items())
        full_reason = f"GPT {yes_count}/4 SIM — {vote_detail} — {reason}"
        
        log.info(f"[GPT-AGENTES] {'✅' if approved else '🚫'} {ativo} {direction} | {full_reason}")
        
        # Cache
        _last_result[cache_key] = (approved, full_reason)
        
        return approved, full_reason
        
    except json.JSONDecodeError as e:
        log.warning(f"[GPT] JSON inválido na resposta — fail-open: {e}")
        return True, "GPT resposta inválida — skip"
    except httpx.TimeoutException:
        log.warning(f"[GPT] Timeout {GPT_TIMEOUT}s — fail-open")
        return True, f"GPT timeout {GPT_TIMEOUT}s — skip"
    except Exception as e:
        log.warning(f"[GPT] Erro: {e} — fail-open")
        return True, f"GPT erro — skip"
