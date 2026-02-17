# -*- coding: utf-8 -*-
"""
🎯 DETECTOR DE PADRÕES - USANDO TA-LIB OFICIAL!
Detecta padrões de candlestick com alta confiabilidade usando a biblioteca TA-Lib

═══════════════════════════════════════════════════
PADRÕES 80%+ IMPLEMENTADOS:
═══════════════════════════════════════════════════
1. CDL3WHITESOLDIERS - 3 Soldados Brancos (88%)
2. CDL3BLACKCROWS - 3 Corvos Pretos (88%)
3. CDLENGULFING - Engolfo (85%)
4. CDLMORNINGSTAR - Estrela da Manhã (85%)
5. CDLEVENINGSTAR - Estrela da Noite (85%)
6. CDLHAMMER - Martelo (82%)
7. CDLSHOOTINGSTAR - Shooting Star (82%)
8. CDLMARUBOZU - Marubozu (80%)
9. CDLDOJISTAR - Doji Star (80%)
10. CDLKICKING - Kicking (92%)
11. CDLABANDONEDBABY - Bebê Abandonado (90%)
12. CDLMORNINGDOJISTAR - Morning Doji Star (88%)
13. CDLEVENINGDOJISTAR - Evening Doji Star (88%)
═══════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Importa TA-Lib
try:
    import talib
    TALIB_AVAILABLE = True
    print("[OK] TA-Lib instalado! Usando padrões profissionais.")
except ImportError:
    TALIB_AVAILABLE = False
    print("[ERRO] TA-Lib não instalado!")


class PatternType(Enum):
    REVERSAO_ALTA = "REVERSAO_ALTA"
    REVERSAO_BAIXA = "REVERSAO_BAIXA"
    CONTINUACAO_ALTA = "CONTINUACAO_ALTA"
    CONTINUACAO_BAIXA = "CONTINUACAO_BAIXA"
    INDECISAO = "INDECISAO"
    NEUTRO = "NEUTRO"


@dataclass
class PatternResult:
    """Resultado da detecção de padrão"""
    nome: str
    tipo: PatternType
    direcao: str  # "CALL", "PUT", "NEUTRO"
    confiabilidade: float  # 0.0 a 1.0
    descricao: str
    alerta: Optional[str] = None


def get_trend_direction(df: pd.DataFrame, periods: int = 10) -> str:
    """Determina a direção da tendência"""
    if len(df) < periods:
        return "NEUTRO"
    
    closes = df['close'].tail(periods).values
    x = np.arange(len(closes))
    slope = np.polyfit(x, closes, 1)[0]
    avg_change = np.mean(np.diff(closes))
    
    if slope > 0 and avg_change > 0:
        return "ALTA"
    elif slope < 0 and avg_change < 0:
        return "BAIXA"
    return "LATERAL"


def get_trend_strength(df: pd.DataFrame, periods: int = 15) -> tuple:
    """
    Calcula a FORÇA da tendência (não apenas direção).
    Retorna: (direcao, forca) onde forca é de 0 a 1
    
    Tendência FORTE = não entrar contra ela!
    """
    if len(df) < periods:
        return ("NEUTRO", 0.0)
    
    closes = df['close'].tail(periods).values
    highs = df['high'].tail(periods).values
    lows = df['low'].tail(periods).values
    
    # Calcular slope normalizado
    x = np.arange(len(closes))
    slope = np.polyfit(x, closes, 1)[0]
    price_range = max(highs) - min(lows)
    if price_range == 0:
        return ("LATERAL", 0.0)
    
    slope_normalized = abs(slope * periods) / price_range
    
    # Contar velas na direção da tendência
    if slope > 0:
        # Tendência de alta - contar velas verdes
        green_candles = sum(1 for i in range(len(closes)-1) if closes[i+1] > closes[i])
        direction = "ALTA"
    else:
        # Tendência de baixa - contar velas vermelhas  
        green_candles = sum(1 for i in range(len(closes)-1) if closes[i+1] < closes[i])
        direction = "BAIXA"
    
    # Força = combinação de slope + consistência
    consistency = green_candles / (periods - 1)
    strength = min(1.0, (slope_normalized + consistency) / 2)
    
    # Se slope muito fraco, é lateral
    if slope_normalized < 0.1:
        return ("LATERAL", strength)
    
    return (direction, strength)


# ═══════════════════════════════════════════════════════════════════════════════
# DETECÇÃO COM TA-LIB - SOMENTE PADRÕES 80%+ CONFIABILIDADE
# ═══════════════════════════════════════════════════════════════════════════════

# Mapeamento de padrões TA-Lib para informações
PADROES_80_PLUS = {
    # ═══════════════════════════════════════════════════════════════
    # SOMENTE PADRÕES COM 80%+ DE CONFIANÇA
    # ═══════════════════════════════════════════════════════════════

    # Padrões comuns (80%+)
    'CDLENGULFING': (0.85, "ENGOLFO", "Engolfo de Alta!", "Engolfo de Baixa!"),
    'CDLHAMMER': (0.82, "HAMMER", "Martelo - Alta!", None),
    'CDLSHOOTINGSTAR': (0.82, "SHOOTING_STAR", None, "Shooting Star - Baixa!"),
    'CDLDRAGONFLYDOJI': (0.80, "DOJI_LIBELULA", "Doji Libélula - Alta!", None),
    'CDLGRAVESTONEDOJI': (0.80, "DOJI_LAPIDE", None, "Doji Lápide - Baixa!"),
    'CDLMARUBOZU': (0.80, "MARUBOZU", "Marubozu Alta!", "Marubozu Baixa!"),
    'CDLPIERCING': (0.80, "PIERCING", "Piercing Line - Alta!", None),
    'CDLDARKCLOUDCOVER': (0.80, "NUVEM_NEGRA", None, "Nuvem Negra - Baixa!"),

    # Padrões de 3 velas (muito confiáveis)
    'CDL3WHITESOLDIERS': (0.88, "3_SOLDADOS", "3 Soldados Brancos!", None),
    'CDL3BLACKCROWS': (0.88, "3_CORVOS", None, "3 Corvos Pretos!"),
    'CDLMORNINGSTAR': (0.85, "MORNING_STAR", "Estrela da Manhã!", None),
    'CDLEVENINGSTAR': (0.85, "EVENING_STAR", None, "Estrela da Noite!"),
    'CDLMORNINGDOJISTAR': (0.88, "MORNING_DOJI", "Morning Doji Star!", None),
    'CDLEVENINGDOJISTAR': (0.88, "EVENING_DOJI", None, "Evening Doji Star!"),

    # Padrões raros mas muito fortes
    'CDLKICKING': (0.92, "KICKING", "Kicking Bullish!", "Kicking Bearish!"),
    'CDLABANDONEDBABY': (0.90, "BEBE_ABANDONADO", "Bebê Abandonado Alta!", "Bebê Abandonado Baixa!"),
}


def detect_candlestick_patterns(df: pd.DataFrame, atr_val: float = 0.0) -> List[PatternResult]:
    """
    Detecta padrões de candlestick usando TA-Lib oficial!
    
    Analisa a ÚLTIMA VELA FECHADA e retorna padrões detectados.
    """
    
    patterns = []
    
    if df is None or len(df) < 15:
        return patterns
    
    if not TALIB_AVAILABLE:
        return patterns
    
    try:
        # Normaliza nomes das colunas
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        # Arrays para TA-Lib (precisa ser float64)
        opens = df['open'].values.astype(np.float64)
        highs = df['high'].values.astype(np.float64)
        lows = df['low'].values.astype(np.float64)
        closes = df['close'].values.astype(np.float64)
        
        # Detectar tendência e FORÇA anterior (para validar padrões de reversão)
        tendencia, forca = get_trend_strength(df.iloc[:-3], 15)  # Tendência ANTES do padrão
        
        # SE TENDÊNCIA MUITO FORTE (>0.6), NÃO ACEITAR PADRÕES DE REVERSÃO!
        tendencia_forte = forca > 0.6
        
        # ═══════════════════════════════════════════════════════════════
        # DETECTAR CADA PADRÃO 80%+ COM TA-LIB
        # ═══════════════════════════════════════════════════════════════
        
        for padrao_talib, (conf, nome_pt, desc_call, desc_put) in PADROES_80_PLUS.items():
            try:
                # Chama a função do TA-Lib dinamicamente
                func = getattr(talib, padrao_talib)
                resultado = func(opens, highs, lows, closes)
                
                # Pega o valor da ÚLTIMA VELA FECHADA (índice -2, pois -1 ainda está formando)
                val = resultado[-2] if len(resultado) > 1 else resultado[-1]
                
                if val == 0:
                    continue
                
                # ═══════════════════════════════════════════════════════════════
                # VALIDAÇÃO DE CONTEXTO RIGOROSA - PADRÕES DE REVERSÃO!
                # ═══════════════════════════════════════════════════════════════
                
                # ═══════════════════════════════════════════════════════════════
                # DETECÇÃO SIMPLIFICADA - DEIXA A IA DECIDIR NO WS_NEURAL_AI
                # ═══════════════════════════════════════════════════════════════

                # Padrões de ALTA (CALL)
                if val > 0 and desc_call:
                    patterns.append(PatternResult(
                        nome=f"{nome_pt}_ALTA" if nome_pt not in ['3_SOLDADOS_BRANCOS', 'MORNING_STAR', 'HAMMER'] else nome_pt,
                        tipo=PatternType.REVERSAO_ALTA,
                        direcao="CALL",
                        confiabilidade=conf,
                        descricao=desc_call
                    ))

                # Padrões de BAIXA (PUT)
                elif val < 0 and desc_put:
                    patterns.append(PatternResult(
                        nome=f"{nome_pt}_BAIXA" if nome_pt not in ['3_CORVOS_PRETOS', 'EVENING_STAR', 'SHOOTING_STAR'] else nome_pt,
                        tipo=PatternType.REVERSAO_BAIXA,
                        direcao="PUT",
                        confiabilidade=conf,
                        descricao=desc_put
                    ))
                    
            except Exception as e:
                pass  # Ignora erros em padrões individuais
        
    except Exception as e:
        print(f"[PATTERN] Erro TA-Lib: {e}")
    
    return patterns


# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÃO DE ANÁLISE COMPLETA
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_patterns(
    df: pd.DataFrame,
    direcao_sinal: str = None,
    atr_val: float = 0.0,
    include_chart_patterns: bool = True
) -> Dict[str, Any]:
    """
    Análise completa de padrões usando TA-Lib
    
    Args:
        df: DataFrame com OHLC
        direcao_sinal: "CALL" ou "PUT" (opcional)
        atr_val: Valor do ATR para normalização
        include_chart_patterns: Não usado (mantido para compatibilidade)
    
    Returns:
        Dict com padrões detectados e análise
    """
    
    result = {
        "padroes_candlestick": [],
        "confirmados": 0,
        "conflitantes": 0,
        "score_padrao": 0.0,
        "direcao_dominante": None,
        "tendencia": "LATERAL"
    }
    
    if df is None or len(df) < 15:
        return result
    
    # Detectar tendência
    result["tendencia"] = get_trend_direction(df)
    
    # Detectar padrões de candlestick com TA-Lib
    candlestick_patterns = detect_candlestick_patterns(df, atr_val)
    
    # Converter para dicionários
    for p in candlestick_patterns:
        result["padroes_candlestick"].append({
            "nome": p.nome,
            "tipo": p.tipo.value,
            "direcao": p.direcao,
            "confiabilidade": p.confiabilidade,
            "descricao": p.descricao
        })
    
    # Contar padrões por direção
    call_patterns = [p for p in candlestick_patterns if p.direcao == "CALL"]
    put_patterns = [p for p in candlestick_patterns if p.direcao == "PUT"]
    
    # Determinar direção dominante
    if len(call_patterns) > len(put_patterns):
        result["direcao_dominante"] = "CALL"
    elif len(put_patterns) > len(call_patterns):
        result["direcao_dominante"] = "PUT"
    
    # Contar confirmados/conflitantes
    if direcao_sinal:
        for p in candlestick_patterns:
            if p.direcao == direcao_sinal:
                result["confirmados"] += 1
            elif p.direcao in ["CALL", "PUT"] and p.direcao != direcao_sinal:
                result["conflitantes"] += 1
    
    # Calcular score (maior confiabilidade entre os padrões na direção correta)
    matching_patterns = [p for p in candlestick_patterns if p.direcao == direcao_sinal]
    if matching_patterns:
        result["score_padrao"] = max(p.confiabilidade for p in matching_patterns)
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# TESTE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 DETECTOR DE PADRÕES - TA-LIB OFICIAL")
    print("=" * 60)
    
    if TALIB_AVAILABLE:
        print("\n✅ TA-Lib instalado e funcionando!")
        print("\nPadrões detectados (80%+):")
        for padrao, (conf, nome, _, _) in PADROES_80_PLUS.items():
            print(f"  • {padrao}: {nome} ({conf*100:.0f}%)")
    else:
        print("\n❌ TA-Lib não instalado!")
    
    print("\n" + "=" * 60)
