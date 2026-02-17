"""
AI Pattern Analyzer - Sistema Híbrido com Claude
Analisa padrões de candlestick e chart patterns usando IA
"""

import os
import anthropic
import json
from datetime import datetime

# API Key do Claude
try:
    from config_keys import CLAUDE_API_KEY_1 as _KEY
    CLAUDE_API_KEY = _KEY
except ImportError:
    CLAUDE_API_KEY = os.getenv("WS_CLAUDE_API_KEY", "")

# Cliente Anthropic
client = None

def init_claude():
    """Inicializa o cliente Claude"""
    global client
    try:
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        return True
    except Exception as e:
        print(f"[❌ CLAUDE] Erro ao inicializar: {e}")
        return False

def format_candles_for_analysis(df, num_candles=20):
    """
    Formata os últimos candles para enviar ao Claude
    """
    if df is None or len(df) < num_candles:
        return None
    
    candles_data = []
    df_recent = df.tail(num_candles)
    
    for i, (idx, row) in enumerate(df_recent.iterrows()):
        candle = {
            "num": i + 1,
            "open": round(row['open'], 5),
            "high": round(row['high'], 5),
            "low": round(row['low'], 5),
            "close": round(row['close'], 5),
            "tipo": "ALTA" if row['close'] > row['open'] else "BAIXA" if row['close'] < row['open'] else "DOJI"
        }
        
        # Calcular tamanho do corpo e sombras
        body = abs(row['close'] - row['open'])
        upper_shadow = row['high'] - max(row['open'], row['close'])
        lower_shadow = min(row['open'], row['close']) - row['low']
        total_range = row['high'] - row['low']
        
        if total_range > 0:
            candle["corpo_pct"] = round((body / total_range) * 100, 1)
            candle["sombra_sup_pct"] = round((upper_shadow / total_range) * 100, 1)
            candle["sombra_inf_pct"] = round((lower_shadow / total_range) * 100, 1)
        
        candles_data.append(candle)
    
    return candles_data

def analyze_with_claude(df, ativo, direcao_sinal, contexto_mercado=None, timeout=10):
    """
    Envia dados para Claude analisar padrões
    
    Args:
        df: DataFrame com candles
        ativo: Nome do ativo
        direcao_sinal: "CALL" ou "PUT" - direção do sinal inicial
        contexto_mercado: Dict com contexto (tendência, S/R, etc)
        timeout: Timeout em segundos
    
    Returns:
        Dict com análise do Claude
    """
    global client
    
    if client is None:
        if not init_claude():
            return {"aprovado": True, "confianca": 50, "motivo": "Claude offline - usando filtro básico"}
    
    # Formatar candles
    candles = format_candles_for_analysis(df, 20)
    if candles is None:
        return {"aprovado": True, "confianca": 50, "motivo": "Dados insuficientes"}
    
    # Construir contexto
    contexto_str = ""
    if contexto_mercado:
        contexto_str = f"""
CONTEXTO DO MERCADO (análise prévia):
- Tipo de mercado: {contexto_mercado.get('market_type', 'N/A')}
- Melhor setup: {contexto_mercado.get('best_setup', 'N/A')}
- Direção recomendada: {contexto_mercado.get('recommended_direction', 'N/A')}
- Confiança prévia: {contexto_mercado.get('confidence', 'N/A')}%
"""
    
    # Prompt para Claude
    prompt = f"""Você é um trader profissional especialista em opções binárias analisando o ativo {ativo}.

DADOS DOS ÚLTIMOS 20 CANDLES (M1):
{json.dumps(candles, indent=2)}

SINAL DETECTADO: {direcao_sinal}
{contexto_str}

TAREFA: Analise os padrões de candlestick e chart patterns para validar ou rejeitar o sinal {direcao_sinal}.

PADRÕES DE CANDLESTICK A VERIFICAR:
- Engolfo de alta/baixa (média confiabilidade)
- Martelo/Martelo Invertido (compra após queda)
- Estrela Cadente/Enforcado (venda após alta)
- Doji (indecisão - cuidado!)
- 3 Soldados Brancos/3 Corvos Pretos (alta confiabilidade)
- Morning Star/Evening Star (alta confiabilidade)
- Harami de alta/baixa (baixa confiabilidade)
- Piercing Line/Nuvem Negra (média confiabilidade)

CHART PATTERNS A VERIFICAR:
- Double Top/Bottom (reversão)
- Head & Shoulders (reversão forte)
- Triângulos (continuação/reversão)
- Flags/Pennants (continuação)
- Wedges (reversão)

REGRAS IMPORTANTES:
1. Se o último candle é DOJI = CUIDADO, mercado indeciso
2. Se há 3+ candles na mesma direção = possível exaustão
3. Se corpo do último candle > 70% = candle de força
4. Se sombras longas = rejeição de preço
5. Padrão de reversão SÓ funciona após tendência clara

RESPONDA APENAS em JSON válido:
{{
    "aprovado": true/false,
    "confianca": 0-100,
    "padrao_detectado": "nome do padrão principal ou NENHUM",
    "tipo_padrao": "REVERSAO/CONTINUACAO/INDEFINIDO",
    "motivo": "explicação curta de 1 linha",
    "alerta": "aviso importante ou null"
}}"""

    try:
        # Chamar Claude
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            timeout=timeout,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extrair resposta
        response_text = message.content[0].text
        
        # Tentar parsear JSON
        # Encontrar JSON na resposta
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)
            
            # Garantir campos obrigatórios
            result.setdefault("aprovado", True)
            result.setdefault("confianca", 50)
            result.setdefault("padrao_detectado", "NENHUM")
            result.setdefault("tipo_padrao", "INDEFINIDO")
            result.setdefault("motivo", "Análise concluída")
            result.setdefault("alerta", None)
            
            return result
        else:
            return {
                "aprovado": True,
                "confianca": 50,
                "padrao_detectado": "NENHUM",
                "tipo_padrao": "INDEFINIDO",
                "motivo": "Resposta não estruturada do Claude",
                "alerta": None
            }
            
    except anthropic.APITimeoutError:
        print(f"[⏱️ CLAUDE] Timeout - usando filtro básico")
        return {
            "aprovado": True,
            "confianca": 50,
            "padrao_detectado": "TIMEOUT",
            "motivo": "Timeout na API - filtro básico aplicado"
        }
    except anthropic.APIError as e:
        print(f"[❌ CLAUDE] Erro API: {e}")
        return {
            "aprovado": True,
            "confianca": 50,
            "padrao_detectado": "ERRO",
            "motivo": f"Erro API: {str(e)[:50]}"
        }
    except json.JSONDecodeError:
        print(f"[⚠️ CLAUDE] Resposta não é JSON válido")
        return {
            "aprovado": True,
            "confianca": 50,
            "padrao_detectado": "PARSE_ERROR",
            "motivo": "Erro ao interpretar resposta"
        }
    except Exception as e:
        print(f"[❌ CLAUDE] Erro inesperado: {e}")
        return {
            "aprovado": True,
            "confianca": 50,
            "padrao_detectado": "ERRO",
            "motivo": f"Erro: {str(e)[:50]}"
        }


def quick_pattern_check(df, direcao):
    """
    Verificação rápida de padrões (sem API) - fallback
    Retorna padrões óbvios detectados localmente
    """
    if df is None or len(df) < 5:
        return None
    
    patterns = []
    
    # Últimos candles
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    
    last_body = abs(last['close'] - last['open'])
    last_range = last['high'] - last['low']
    last_upper = last['high'] - max(last['open'], last['close'])
    last_lower = min(last['open'], last['close']) - last['low']
    
    is_last_bullish = last['close'] > last['open']
    is_last_bearish = last['close'] < last['open']
    
    prev_body = abs(prev['close'] - prev['open'])
    is_prev_bullish = prev['close'] > prev['open']
    is_prev_bearish = prev['close'] < prev['open']
    
    # 1. DOJI (indecisão)
    if last_range > 0 and (last_body / last_range) < 0.1:
        patterns.append({"nome": "DOJI", "tipo": "INDEFINIDO", "alerta": "Mercado indeciso!"})
    
    # 2. ENGOLFO
    if is_last_bullish and is_prev_bearish:
        if last['close'] > prev['open'] and last['open'] < prev['close']:
            patterns.append({"nome": "ENGOLFO_ALTA", "tipo": "REVERSAO", "favorece": "CALL"})
    
    if is_last_bearish and is_prev_bullish:
        if last['close'] < prev['open'] and last['open'] > prev['close']:
            patterns.append({"nome": "ENGOLFO_BAIXA", "tipo": "REVERSAO", "favorece": "PUT"})
    
    # 3. MARTELO (após queda)
    if last_range > 0:
        if (last_lower / last_range) > 0.6 and (last_upper / last_range) < 0.1:
            # Verificar se houve queda antes
            if df.iloc[-5:-1]['close'].mean() > last['low']:
                patterns.append({"nome": "MARTELO", "tipo": "REVERSAO", "favorece": "CALL"})
    
    # 4. ESTRELA CADENTE (após alta)
    if last_range > 0:
        if (last_upper / last_range) > 0.6 and (last_lower / last_range) < 0.1:
            # Verificar se houve alta antes
            if df.iloc[-5:-1]['close'].mean() < last['high']:
                patterns.append({"nome": "ESTRELA_CADENTE", "tipo": "REVERSAO", "favorece": "PUT"})
    
    # 5. 3 CANDLES NA MESMA DIREÇÃO (possível exaustão)
    last_3 = df.tail(3)
    all_bullish = all(last_3['close'] > last_3['open'])
    all_bearish = all(last_3['close'] < last_3['open'])
    
    if all_bullish:
        patterns.append({"nome": "3_ALTAS_SEGUIDAS", "tipo": "CONTINUACAO", "alerta": "Possível exaustão de alta"})
    if all_bearish:
        patterns.append({"nome": "3_BAIXAS_SEGUIDAS", "tipo": "CONTINUACAO", "alerta": "Possível exaustão de baixa"})
    
    # 6. CANDLE DE FORÇA
    avg_body = df.tail(10)['close'].sub(df.tail(10)['open']).abs().mean()
    if last_body > avg_body * 2:
        tipo = "FORCA_ALTA" if is_last_bullish else "FORCA_BAIXA"
        patterns.append({"nome": tipo, "tipo": "CONTINUACAO", "favorece": "CALL" if is_last_bullish else "PUT"})
    
    return patterns if patterns else None


# Configuração para habilitar/desabilitar Claude
CLAUDE_ENABLED = True
CLAUDE_MIN_SCORE = 65  # Score mínimo para chamar Claude (economia de API)

def should_use_claude(score):
    """Decide se deve usar Claude baseado no score"""
    return CLAUDE_ENABLED and score >= CLAUDE_MIN_SCORE


# Teste
if __name__ == "__main__":
    print("Testando conexão com Claude...")
    if init_claude():
        print("✅ Claude inicializado com sucesso!")
        
        # Teste simples
        try:
            test_result = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=50,
                messages=[{"role": "user", "content": "Diga apenas: OK"}]
            )
            print(f"✅ Resposta: {test_result.content[0].text}")
            print("🎉 Claude funcionando perfeitamente!")
        except Exception as e:
            error_msg = str(e)
            if "credit balance" in error_msg.lower():
                print("⚠️ Sua conta Claude precisa de créditos!")
                print("   👉 Acesse: https://console.anthropic.com/settings/billing")
                print("   👉 Adicione créditos para usar a análise híbrida")
                print("   📌 Enquanto isso, o sistema usará análise local de padrões")
            else:
                print(f"❌ Erro: {e}")
    else:
        print("❌ Falha ao inicializar Claude")
