# -*- coding: utf-8 -*-
"""
IA 4 — Guard Generativa (GPT-4o-mini / Claude Haiku)
═══════════════════════════════════════════════════════
Última camada de validação antes da entrada.
Recebe 100 velas + geometria + scores NN → GPT analisa e decide.

FLUXO:
  1. IA 1-3 (XGBoost + LightGBM + MLP) votam WIN com score ≥ threshold
  2. IA 4 (esta) recebe:
     - 100 velas recentes (OHLC compacto)
     - Geometria: head, shoulders, valleys, neckline, target
     - Scores das 3 NNs + features importantes
     - Direção (CALL/PUT) + ativo
  3. GPT analisa o contexto completo e responde APPROVE ou REJECT
  4. Se REJECT → bloqueia entrada (evita loss)

CACHE:
  - Hash do padrão → resultado → evita re-queries na mesma vela
  - TTL: 60s (padrão muda a cada candle)

FALLBACK:
  - Se API falhar → APPROVE (não bloqueia por falha técnica)
  - Timeout: 8s (tempo suficiente antes da virada :00)
"""

import os
import json
import time
import hashlib
import logging
import threading

log = logging.getLogger("WS_BULLEX")

# ── API Keys ──
try:
    from config_keys import OPENAI_API_KEY
except ImportError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Config ──
_GPT_MODEL = os.getenv("WS_GPT_MODEL", "gpt-4o-mini")
_GPT_TIMEOUT = int(os.getenv("WS_GPT_TIMEOUT", "8"))
_GPT_ENABLED = os.getenv("WS_GPT_GUARD", "1").strip() == "1"
_CACHE_TTL = 60  # segundos

# ── Cache thread-safe ──
_cache = {}
_cache_lock = threading.Lock()

# ── Cliente OpenAI (lazy init) ──
_client = None


def _get_client():
    """Inicializa cliente OpenAI sob demanda."""
    global _client
    if _client is None:
        try:
            from openai import OpenAI
            _client = OpenAI(
                api_key=OPENAI_API_KEY,
                timeout=float(_GPT_TIMEOUT),
                max_retries=1,
            )
        except Exception as e:
            log.warning(f"  ⚠️ IA4: OpenAI client init falhou: {e}")
    return _client


# ══════════════════════════════════════════════════════════════
# SISTEMA DE PROMPTS — RAG (conhecimento de trading embutido)
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Você é um analista técnico ESPECIALISTA em padrões Double Top / Double Bottom (DT) em opções binárias M1.

## SUA FUNÇÃO
Analisar as 100 velas + geometria do padrão e decidir: **APPROVE** ou **REJECT**.
Você é a ÚLTIMA camada de proteção. 3 IAs estatísticas já aprovaram. Sua missão é REJEITAR setups perigosos que as IAs não conseguem ver.

## CONHECIMENTO: PADRÃO DOUBLE TOUCH (DT)
- **Double Top (PUT)**: Preço toca RESISTÊNCIA 2x formando dois picos, depois cai.
  - Left Shoulder (LS) = 1º toque no topo
  - Head = ponto mais alto (entre LS e RS)
  - Right Shoulder (RS) = 2º toque no topo (confirmação)
  - Valley = ponto mais baixo entre os toques
  - Neckline = nível de suporte (objetivo mínimo)
  - ENTRADA: na rejeição do RS (PUT = queda esperada)

- **Double Bottom (CALL)**: Preço toca SUPORTE 2x formando dois fundos, depois sobe.
  - Mesma estrutura, invertida.
  - ENTRADA: na rejeição do RS (CALL = subida esperada)

## CRITÉRIOS DE REJEIÇÃO (REJECT se qualquer um for verdadeiro):

### 1. TENDÊNCIA FORTE CONTRA
- Se as últimas 20 velas mostram tendência FORTE contra a direção da operação
- Ex: PUT mas preço subindo forte com velas de corpo grande → REJECT
- Ex: CALL mas preço caindo forte → REJECT

### 2. MOMENTUM EXPLOSIVO CONTRA
- Verificar as últimas 5-10 velas: se têm corpos grandes e consecutivos contra o trade → REJECT
- Vela atual é a MAIOR dos últimos 20 candles E contra o trade → REJECT

### 3. PREÇO MUITO ESTICADO
- Se o preço já percorreu >60% do caminho RS→Neckline → entrada TARDE demais → REJECT

### 4. PADRÃO DEFORMADO
- Simetria muito ruim (um lado 3x maior que o outro) → REJECT
- Profundidade muito rasa (<0.3 ATR) → padrão fraco → REJECT
- Head não é o ponto mais extremo → formação inválida → REJECT

### 5. MERCADO CAÓTICO (CHOPPY)
- Últimas 30 velas com alternância excessiva de cores (>80%) → mercado sem rumo → REJECT
- Wicks enormes em ambas direções → mercado indeciso → REJECT

### 6. ROMPIMENTO JÁ OCORREU
- Se preço já rompeu o neckline → o trade JÁ FOI → REJECT
- Se preço está do lado ERRADO do RS → padrão invalidado → REJECT

## CRITÉRIOS DE APROVAÇÃO (APPROVE):
- Rejeição clara no RS (wick de rejeição visível)
- Momentum desacelerando na direção contra (padrão de exaustão)
- Mercado em range ou lateralizado (contexto ideal para DT)
- Velas recentes diminuindo de tamanho (perda de força)
- Padrão simétrico com boa profundidade

## FORMATO DA RESPOSTA (OBRIGATÓRIO — siga EXATAMENTE):
```
DECISION: APPROVE
CONFIDENCE: 85
REASON: Rejeição clara no RS com wick, mercado lateralizado, momentum desacelerando
```
ou
```
DECISION: REJECT
CONFIDENCE: 90
REASON: Tendência forte de alta contra PUT, últimas 15 velas com corpos grandes ascendentes
```

REGRAS:
- DECISION deve ser APPROVE ou REJECT (nada mais)
- CONFIDENCE é um número de 0 a 100
- REASON é uma frase curta (máximo 100 palavras)
- NÃO use markdown, NÃO use ```
- Responda APENAS as 3 linhas acima"""


def _format_candles(H, L, C, O, n, last_n=100):
    """Formata últimas N velas como texto compacto para o GPT."""
    start = max(0, n - last_n)
    lines = []
    for i in range(start, n):
        idx = i - start
        o, h, l, c = O[i], H[i], L[i], C[i]
        body = c - o
        direction = "▲" if body > 0 else "▼" if body < 0 else "─"
        lines.append(f"{idx:3d}|O={o:.5f} H={h:.5f} L={l:.5f} C={c:.5f} {direction}")
    return "\n".join(lines)


def _format_pattern(pat_data, atr_val, direcao):
    """Formata geometria do padrão para o prompt."""
    ls = pat_data.get("left_shoulder", {})
    head = pat_data.get("head", {})
    rs = pat_data.get("right_shoulder", {})
    v1 = pat_data.get("valley1", {})
    v2 = pat_data.get("valley2", {})

    neck = pat_data.get("neckline", 0)
    target = pat_data.get("target", 0)
    depth = pat_data.get("depth", 0)

    iL = ls.get("idx", 0)
    iH = head.get("idx", 0)
    iR = rs.get("idx", 0)
    span = iR - iL
    d_left = iH - iL
    d_right = iR - iH
    symmetry = min(d_left, d_right) / max(d_left, d_right) if max(d_left, d_right) > 0 else 0
    depth_ratio = depth / atr_val if atr_val > 0 else 0

    return (
        f"DIREÇÃO: {direcao}\n"
        f"TIPO: {'Double Top (resistência)' if direcao == 'PUT' else 'Double Bottom (suporte)'}\n"
        f"Left Shoulder:  preço={ls.get('price', 0):.5f} idx={iL}\n"
        f"Head:           preço={head.get('price', 0):.5f} idx={iH}\n"
        f"Right Shoulder: preço={rs.get('price', 0):.5f} idx={iR}\n"
        f"Valley 1:       preço={v1.get('price', 0):.5f}\n"
        f"Valley 2:       preço={v2.get('price', 0):.5f}\n"
        f"Neckline:       {neck:.5f}\n"
        f"Target:         {target:.5f}\n"
        f"ATR:            {atr_val:.5f}\n"
        f"Span:           {span} velas\n"
        f"Simetria:       {symmetry:.2f} (1.0 = perfeita)\n"
        f"Profundidade:   {depth_ratio:.2f} ATR\n"
        f"D_left:         {d_left} velas (LS→Head)\n"
        f"D_right:        {d_right} velas (Head→RS)\n"
    )


def _format_nn_scores(nn_pred):
    """Formata scores das 3 NNs para o prompt."""
    if nn_pred is None:
        return "NN: sem dados"
    p1 = nn_pred.get("p1", 0)
    p2 = nn_pred.get("p2", 0)
    p3 = nn_pred.get("p3")
    score = nn_pred.get("nn_score", nn_pred.get("prob_win", 0))
    votes = nn_pred.get("votes_win", 0)
    total = nn_pred.get("total_voters", 3)
    p3_str = f"IA3(MLP)={p3:.0%}" if p3 is not None else "IA3=N/A"
    return (
        f"IA1(XGBoost)={p1:.0%} | IA2(LightGBM)={p2:.0%} | {p3_str}\n"
        f"Score final={score:.0%} | Votos WIN={votes}/{total}"
    )


def _cache_key(ativo, direcao, pat_data):
    """Gera hash único para cache do padrão."""
    rs_price = pat_data.get("right_shoulder", {}).get("price", 0)
    head_price = pat_data.get("head", {}).get("price", 0)
    key = f"{ativo}_{direcao}_{head_price:.6f}_{rs_price:.6f}"
    return hashlib.md5(key.encode()).hexdigest()


def _parse_response(text):
    """Extrai DECISION, CONFIDENCE e REASON da resposta do GPT."""
    result = {"approved": True, "confidence": 50, "reason": "parse_error"}
    if not text:
        return result

    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.upper().startswith("DECISION:"):
            val = line.split(":", 1)[1].strip().upper()
            result["approved"] = val == "APPROVE"
        elif line.upper().startswith("CONFIDENCE:"):
            try:
                result["confidence"] = int(line.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.upper().startswith("REASON:"):
            result["reason"] = line.split(":", 1)[1].strip()

    return result


# ══════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL — chamada pelo bot
# ══════════════════════════════════════════════════════════════

def gpt_guard_check(
    ativo: str,
    direcao: str,
    pat_data: dict,
    H, L, C, O, n: int,
    atr_val: float,
    nn_pred: dict = None,
    cur_price: float = 0,
) -> dict:
    """
    IA 4 — Guard Generativa.

    Envia 100 velas + geometria + scores NN → GPT analisa e decide.

    Retorna:
        {
            "approved": bool,
            "confidence": int (0-100),
            "reason": str,
            "source": "gpt" | "cache" | "fallback",
            "latency_ms": int,
        }
    """
    t0 = time.time()

    # ── Desabilitado? → APPROVE ──
    if not _GPT_ENABLED:
        return {
            "approved": True, "confidence": 0,
            "reason": "GPT guard desabilitado",
            "source": "disabled", "latency_ms": 0,
        }

    # ── Sem API key? → APPROVE ──
    if not OPENAI_API_KEY:
        return {
            "approved": True, "confidence": 0,
            "reason": "API key não configurada",
            "source": "no_key", "latency_ms": 0,
        }

    # ── Cache hit? ──
    ck = _cache_key(ativo, direcao, pat_data)
    with _cache_lock:
        cached = _cache.get(ck)
        if cached and (time.time() - cached["ts"]) < _CACHE_TTL:
            cached["result"]["source"] = "cache"
            cached["result"]["latency_ms"] = 0
            return cached["result"]

    # ── Montar prompt ──
    candles_text = _format_candles(H, L, C, O, n, last_n=100)
    pattern_text = _format_pattern(pat_data, atr_val, direcao)
    nn_text = _format_nn_scores(nn_pred)

    user_prompt = (
        f"ATIVO: {ativo}\n"
        f"PREÇO ATUAL: {cur_price:.5f}\n\n"
        f"═══ PADRÃO DETECTADO ═══\n{pattern_text}\n"
        f"═══ SCORES DAS 3 IAs ═══\n{nn_text}\n\n"
        f"═══ ÚLTIMAS 100 VELAS (M1) ═══\n{candles_text}\n\n"
        f"Analise o contexto completo e decida: APPROVE ou REJECT?"
    )

    # ── Chamar GPT ──
    try:
        client = _get_client()
        if client is None:
            raise RuntimeError("Cliente OpenAI indisponível")

        response = client.chat.completions.create(
            model=_GPT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=150,
        )

        reply = response.choices[0].message.content
        result = _parse_response(reply)
        result["source"] = "gpt"
        result["latency_ms"] = int((time.time() - t0) * 1000)

        # ── Salvar no cache ──
        with _cache_lock:
            _cache[ck] = {"ts": time.time(), "result": result}
            # Limpar cache antigo (máx 50 entradas)
            if len(_cache) > 50:
                oldest = sorted(_cache, key=lambda k: _cache[k]["ts"])
                for old_key in oldest[:20]:
                    del _cache[old_key]

        return result

    except Exception as e:
        latency = int((time.time() - t0) * 1000)
        log.warning(f"  ⚠️ IA4 GPT falhou ({latency}ms): {e}")
        return {
            "approved": True, "confidence": 0,
            "reason": f"API error: {type(e).__name__}",
            "source": "fallback", "latency_ms": latency,
        }
