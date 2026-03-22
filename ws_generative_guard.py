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
# SISTEMA DE PROMPTS (conhecimento de trading embutido)
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Você é um guard de proteção para trades Double Touch (DT) em opções binárias M1.
3 IAs estatísticas já APROVARAM este trade. Você SÓ rejeita se houver PERIGO CLARO.

## REGRA #1 — ENTENDA O PADRÃO:
- Double Top (PUT): preço SOBE até resistência (RS) 2x → rejeita → CAI. A subida até RS é PARTE do padrão, NÃO é contra.
- Double Bottom (CALL): preço DESCE até suporte (RS) 2x → rejeita → SOBE. A descida até RS é PARTE do padrão, NÃO é contra.
- O preço se mover em direção ao RS é FAVORÁVEL, NÃO é contra o trade.

## REGRA #2 — USE OS FATOS COMPUTADOS:
O prompt inclui FATOS PRÉ-COMPUTADOS (preço vs neckline, preço vs RS, etc).
USE ESSES FATOS. NÃO tente recalcular a partir das velas. Os fatos são 100% corretos.

## REGRA #3 — CONSOLIDAÇÃO (VELAS PEQUENAS):
- Se o fato "CONSOLIDACAO" = SIM, as últimas velas têm corpos muito pequenos (mercado lateralizando).
- Consolidação perto do RS é PERIGOSO: preço pode demorar para reagir ou romper lateralmente.
- Se CONSOLIDACAO=SIM E corpo médio < 20% do ATR → REJECT (preço sem força para mover no tempo do trade).
- Se CONSOLIDACAO=SIM mas corpo médio entre 20-40% do ATR → APPROVE com CONFIDENCE baixa (60-70).

## CRITÉRIOS DE REJEIÇÃO (SÓ rejeite se um destes for CLARO):
1. BREAKOUT DO RS: Preço FECHOU ALÉM do RS com corpo grande (>1.5 ATR) → rompeu o nível → REJECT
2. MOMENTUM PÓS-BREAKOUT: Preço rompeu RS E continua se afastando com 3+ velas grandes → REJECT
3. PREÇO JÁ ROMPEU NECKLINE: Veja o fato "ROMPEU_NECKLINE". Se SIM → REJECT
4. MERCADO CAÓTICO: >80% das últimas 20 velas alternam cor → sem direção → REJECT
5. CONSOLIDAÇÃO FORTE: CONSOLIDACAO=SIM E corpo médio < 20% ATR → preço parado, sem momentum → REJECT

## CRITÉRIOS DE APROVAÇÃO:
- Preço perto do RS (fato "DIST_RS_PCT" < 50%) → APPROVE
- Score NN > 90% → as 3 IAs estão confiantes → APPROVE
- Preço NÃO rompeu neckline → trade ainda válido → APPROVE

## REGRA #4 — MEMÓRIA RAG (TRADES SIMILARES):
Se o prompt incluir "MEMÓRIA RAG: TRADES SIMILARES", USE essas informações:
- Se WR < 40% naquele nível → REJECT (padrão estatisticamente perdedor ali)
- Se WR ≥ 70% com 3+ trades → APPROVE com confiança elevada
- Se histórico mostra que consolidação = LOSS naquele ativo → REJECT se consolidação presente
- A memória é de trades REAIS anteriores, não simulações. É a evidência mais forte.

## REGRA #5 — TEMPO DE EXPIRAÇÃO (EXP):
Escolha o melhor tempo de expiração com base nas condições do mercado:
- EXP=1: momentum ALTO, velas grandes, ATR alto → preço move rápido.
- EXP=2: condição NORMAL, volatilidade média → padrão.
- EXP=4: CONSOLIDACAO=SIM ou volatilidade BAIXA → preço demora para reagir.

## RESPOSTA (EXATAMENTE 4 linhas):
DECISION: APPROVE
CONFIDENCE: 85
EXP_MINUTES: 2
REASON: frase curta explicando

REGRAS: DECISION=APPROVE ou REJECT. CONFIDENCE=0-100. EXP_MINUTES=1,2 ou 4. REASON=máx 50 palavras. Sem markdown."""


def _format_candles(H, L, C, O, n, last_n=30):
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
    result = {"approved": True, "confidence": 50, "reason": "parse_error", "exp_minutes": 2}
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
        elif line.upper().startswith("EXP_MINUTES:"):
            try:
                _exp = int(line.split(":", 1)[1].strip())
                if _exp in (1, 2, 4):
                    result["exp_minutes"] = _exp
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
    rag_context: str = "",
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
    candles_text = _format_candles(H, L, C, O, n, last_n=30)
    pattern_text = _format_pattern(pat_data, atr_val, direcao)
    nn_text = _format_nn_scores(nn_pred)

    # ── Fatos pré-computados (GPT NÃO precisa recalcular) ──
    _rs_price = pat_data.get("right_shoulder", {}).get("price", 0)
    _neckline = pat_data.get("neckline", 0)
    _dist_rs = abs(cur_price - _rs_price) if _rs_price > 0 else 0
    _range_rs_neck = abs(_neckline - _rs_price) if _neckline > 0 and _rs_price > 0 else 1
    _dist_rs_pct = (_dist_rs / _range_rs_neck * 100) if _range_rs_neck > 0 else 0
    _dist_rs_atr = (_dist_rs / atr_val) if atr_val > 0 else 0

    # Verificar rompimento do neckline
    if direcao == "CALL":
        _rompeu_neck = "SIM" if cur_price >= _neckline else "NÃO"
        _rompeu_rs = "SIM" if cur_price < _rs_price - atr_val * 1.5 else "NÃO"
        _pos_vs_neck = "ABAIXO" if cur_price < _neckline else "ACIMA"
        _pos_vs_rs = "ACIMA" if cur_price > _rs_price else "ABAIXO"
    else:
        _rompeu_neck = "SIM" if cur_price <= _neckline else "NÃO"
        _rompeu_rs = "SIM" if cur_price > _rs_price + atr_val * 1.5 else "NÃO"
        _pos_vs_neck = "ACIMA" if cur_price > _neckline else "ABAIXO"
        _pos_vs_rs = "ABAIXO" if cur_price < _rs_price else "ACIMA"

    facts_text = (
        f"═══ FATOS PRÉ-COMPUTADOS (100% corretos, USE ESTES) ═══\n"
        f"PREÇO_ATUAL: {cur_price:.5f}\n"
        f"RS: {_rs_price:.5f}\n"
        f"NECKLINE: {_neckline:.5f}\n"
        f"PREÇO_VS_RS: {_pos_vs_rs} do RS (dist={_dist_rs:.5f}, {_dist_rs_pct:.0f}% do range)\n"
        f"PREÇO_VS_NECKLINE: {_pos_vs_neck} do Neckline\n"
        f"ROMPEU_NECKLINE: {_rompeu_neck}\n"
        f"ROMPEU_RS_BREAKOUT: {_rompeu_rs}\n"
        f"DIST_RS_ATR: {_dist_rs_atr:.2f} ATR\n"
        f"DIST_RS_PCT: {_dist_rs_pct:.0f}%\n"
    )

    # ── Detecção de consolidação (velas pequenas) ──
    _last_5_start = max(0, n - 5)
    _bodies = [abs(C[i] - O[i]) for i in range(_last_5_start, n)]
    _avg_body = sum(_bodies) / len(_bodies) if _bodies else 0
    _body_vs_atr = (_avg_body / atr_val * 100) if atr_val > 0 else 100
    _consolidacao = "SIM" if _body_vs_atr < 40 else "NÃO"
    facts_text += (
        f"CONSOLIDACAO: {_consolidacao} (corpo médio últimas 5 velas = {_body_vs_atr:.0f}% do ATR)\n"
    )

    # ── Memória RAG (trades similares do passado) ──
    rag_section = ""
    if rag_context:
        rag_section = f"\n{rag_context}\n"

    user_prompt = (
        f"ATIVO: {ativo}\n\n"
        f"{facts_text}\n"
        f"═══ PADRÃO DETECTADO ═══\n{pattern_text}\n"
        f"═══ SCORES DAS 3 IAs ═══\n{nn_text}\n\n"
        f"{rag_section}"
        f"═══ ÚLTIMAS 30 VELAS (M1) ═══\n{candles_text}\n\n"
        f"Analise OS FATOS acima e decida: APPROVE ou REJECT?"
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
