# -*- coding: utf-8 -*-
"""
Loss Analyzer - Agente de Análise de Perdas
Captura as últimas 100 velas quando há loss, analisa com IA e grava no Firebase
"""

import json
import logging
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import time
from iqoptionapi.stable_api import IQ_Option

logger = logging.getLogger(__name__)


class LossAnalyzer:
    """Analisador inteligente de losses"""
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
    
    def _ensure_connected(self, iq: IQ_Option) -> None:
        """Garante conexão ativa com a IQ Option"""
        try:
            if hasattr(iq, "check_connect") and iq.check_connect():
                return
        except Exception:
            pass

        try:
            if hasattr(iq, "connect"):
                iq.connect()
                for _ in range(6):
                    if hasattr(iq, "check_connect") and iq.check_connect():
                        return
                    time.sleep(1)
        except Exception:
            return
        
    def get_candles(self, iq: IQ_Option, ativo: str, timeframe: int = 60, n: int = 100) -> Optional[pd.DataFrame]:
        """Captura as últimas N velas com retry"""
        max_retries = 3
        retry_delay = 2  # segundos
        
        for attempt in range(max_retries):
            try:
                # Aguarda um pouco antes de tentar (API pode estar processando resultado)
                if attempt > 0:
                    time.sleep(retry_delay)
                    logger.info(f"Tentativa {attempt + 1}/{max_retries} de capturar velas...")
                
                # Garante conexão ativa antes de buscar velas
                self._ensure_connected(iq)
                
                candles = iq.get_candles(ativo, timeframe, n, time.time())
                if not candles or isinstance(candles, int):
                    if attempt < max_retries - 1:
                        continue
                    return None
                    
                df = pd.DataFrame(candles)
                if df.empty:
                    if attempt < max_retries - 1:
                        continue
                    return None
                    
                df = self._normalize_candles_df(df)
                
                logger.info(f"✅ {len(df)} velas capturadas com sucesso")
                return df
                
            except Exception as e:
                msg = str(e).lower()
                if "need reconnect" in msg or "disconnected" in msg or "connection" in msg:
                    self._ensure_connected(iq)
                logger.error(f"Erro ao capturar velas (tentativa {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None
        
        return None

    def _normalize_candles_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Garante colunas padronizadas para análise (compatível com low/high ou min/max)."""
        # Normaliza nomes de colunas de preço
        if "min" not in df.columns and "low" in df.columns:
            df["min"] = df["low"]
        if "max" not in df.columns and "high" in df.columns:
            df["max"] = df["high"]
        if "low" not in df.columns and "min" in df.columns:
            df["low"] = df["min"]
        if "high" not in df.columns and "max" in df.columns:
            df["high"] = df["max"]

        # Colunas básicas obrigatórias
        for col in ("open", "close", "min", "max"):
            if col not in df.columns:
                df[col] = pd.NA

        # Adiciona colunas úteis se não existirem
        if "body" not in df.columns:
            df["body"] = (df["close"] - df["open"]).abs()
        if "upper_wick" not in df.columns:
            df["upper_wick"] = df["max"] - df[["close", "open"]].max(axis=1)
        if "lower_wick" not in df.columns:
            df["lower_wick"] = df[["close", "open"]].min(axis=1) - df["min"]
        if "range" not in df.columns:
            df["range"] = df["max"] - df["min"]
        if "is_green" not in df.columns:
            df["is_green"] = df["close"] > df["open"]

        return df
    
    def analyze_market_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisa o contexto de mercado"""
        try:
            df = self._normalize_candles_df(df)
            # Tendência geral (últimas 20 velas)
            recent = df.tail(20)
            green_count = recent['is_green'].sum()
            red_count = len(recent) - green_count
            
            # Momentum
            price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
            
            # Volatilidade (ATR simplificado)
            df['tr'] = df[['max', 'close']].max(axis=1) - df[['min', 'close']].min(axis=1)
            atr = df['tr'].tail(14).mean()
            
            # Volume relativo (variação de tamanho das velas)
            avg_body = df['body'].mean()
            recent_body = recent['body'].mean()
            volume_ratio = recent_body / avg_body if avg_body > 0 else 1.0
            
            # Detectar consolidação
            is_consolidating = df['close'].tail(20).std() / df['close'].mean() < 0.001
            
            # Suporte e resistência recentes
            resistance = df['max'].tail(50).max()
            support = df['min'].tail(50).min()
            current_price = df['close'].iloc[-1]
            
            near_resistance = abs(current_price - resistance) / current_price < 0.001
            near_support = abs(current_price - support) / current_price < 0.001
            
            return {
                "trend": "bullish" if green_count > red_count else "bearish" if red_count > green_count else "neutral",
                "green_candles": int(green_count),
                "red_candles": int(red_count),
                "price_change_percent": float(price_change),
                "atr": float(atr),
                "volatility": "high" if atr > df['tr'].mean() * 1.5 else "low",
                "volume_ratio": float(volume_ratio),
                "is_consolidating": bool(is_consolidating),
                "near_resistance": bool(near_resistance),
                "near_support": bool(near_support),
                "current_price": float(current_price),
                "resistance": float(resistance),
                "support": float(support)
            }
        except Exception as e:
            logger.error(f"Erro na análise de contexto: {e}")
            return {}
    
    def analyze_entry_quality(self, df: pd.DataFrame, direction: str, entry_index: int = -1) -> Dict[str, Any]:
        """Analisa a qualidade da entrada"""
        try:
            df = self._normalize_candles_df(df)
            entry_candle = df.iloc[entry_index]
            prev_candles = df.iloc[entry_index-5:entry_index]
            
            # Força do sinal de entrada
            entry_body_ratio = entry_candle['body'] / entry_candle['range'] if entry_candle['range'] > 0 else 0
            
            # Alinhamento das velas anteriores
            if direction.upper() == "CALL":
                aligned = prev_candles['is_green'].sum()
            else:
                aligned = (~prev_candles['is_green']).sum()
            
            alignment_ratio = aligned / len(prev_candles)
            
            # Momentum na direção
            price_momentum = prev_candles['close'].iloc[-1] - prev_candles['close'].iloc[0]
            momentum_direction = "correct" if (direction.upper() == "CALL" and price_momentum > 0) or (direction.upper() == "PUT" and price_momentum < 0) else "wrong"
            
            return {
                "entry_body_ratio": float(entry_body_ratio),
                "entry_quality": "strong" if entry_body_ratio > 0.7 else "weak",
                "alignment_ratio": float(alignment_ratio),
                "momentum_direction": momentum_direction,
                "prev_candles_aligned": int(aligned),
                "prev_candles_total": len(prev_candles)
            }
        except Exception as e:
            logger.error(f"Erro na análise de entrada: {e}")
            return {}
    
    def generate_ai_analysis(self, market_context: Dict, entry_quality: Dict, 
                           ativo: str, direction: str, stake: float) -> str:
        """Gera análise textual usando IA (GPT-like)"""
        
        # Análise estruturada
        problems = []
        recommendations = []
        
        # Problema 1: Tendência contrária
        if market_context.get("trend") == "bullish" and direction.upper() == "PUT":
            problems.append("Operação contra tendência: mercado está bullish mas operou PUT")
            recommendations.append("Evitar PUT quando tendência recente é claramente bullish (>60% velas verdes)")
        elif market_context.get("trend") == "bearish" and direction.upper() == "CALL":
            problems.append("Operação contra tendência: mercado está bearish mas operou CALL")
            recommendations.append("Evitar CALL quando tendência recente é claramente bearish (>60% velas vermelhas)")
        
        # Problema 2: Consolidação
        if market_context.get("is_consolidating"):
            problems.append("Mercado em consolidação: movimentos laterais com baixa volatilidade")
            recommendations.append("Evitar operar durante consolidação - aguardar rompimento claro")
        
        # Problema 3: Resistência/Suporte
        if market_context.get("near_resistance") and direction.upper() == "CALL":
            problems.append("Operação CALL próxima de resistência forte")
            recommendations.append("Evitar CALL próximo de resistência - maior probabilidade de rejeição")
        elif market_context.get("near_support") and direction.upper() == "PUT":
            problems.append("Operação PUT próxima de suporte forte")
            recommendations.append("Evitar PUT próximo de suporte - maior probabilidade de bounce")
        
        # Problema 4: Entrada fraca
        if entry_quality.get("entry_quality") == "weak":
            problems.append("Vela de entrada fraca: corpo pequeno com muito wick")
            recommendations.append("Aguardar velas com corpo forte (>70% do range) para entrada")
        
        # Problema 5: Desalinhamento
        if entry_quality.get("alignment_ratio", 0) < 0.6:
            problems.append("Velas anteriores desalinhadas com direção da operação")
            recommendations.append("Aguardar pelo menos 3 de 5 velas alinhadas antes de operar")
        
        # Problema 6: Momentum errado
        if entry_quality.get("momentum_direction") == "wrong":
            problems.append("Momentum contrário à direção da operação")
            recommendations.append("Confirmar momentum favorável antes da entrada")
        
        # Problema 7: Volatilidade
        if market_context.get("volatility") == "high":
            problems.append("Alta volatilidade: movimentos imprevisíveis")
            recommendations.append("Reduzir stake ou evitar operar em períodos de alta volatilidade")
        
        # Monta análise final
        analysis = f"""
📊 ANÁLISE DE LOSS - {ativo}
{'='*50}

💰 Stake: ${stake:.2f}
📈 Direção: {direction.upper()}
📉 Resultado: LOSS

🔍 PROBLEMAS IDENTIFICADOS:
"""
        
        if problems:
            for i, prob in enumerate(problems, 1):
                analysis += f"\n{i}. {prob}"
        else:
            analysis += "\n- Nenhum problema óbvio detectado (loss por variação natural do mercado)"
        
        analysis += f"""

💡 RECOMENDAÇÕES:
"""
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                analysis += f"\n{i}. {rec}"
        else:
            analysis += "\n- Loss dentro da normalidade - manter estratégia atual"
        
        analysis += f"""

📊 CONTEXTO DE MERCADO:
- Tendência: {market_context.get('trend', 'N/A')}
- Velas verdes/vermelhas: {market_context.get('green_candles', 0)}/{market_context.get('red_candles', 0)}
- Volatilidade: {market_context.get('volatility', 'N/A')}
- Consolidação: {'Sim' if market_context.get('is_consolidating') else 'Não'}
- ATR: {market_context.get('atr', 0):.5f}

🎯 QUALIDADE DA ENTRADA:
- Força da vela: {entry_quality.get('entry_quality', 'N/A')}
- Alinhamento: {entry_quality.get('alignment_ratio', 0):.1%}
- Momentum: {entry_quality.get('momentum_direction', 'N/A')}
"""
        
        return analysis
    
    def save_to_firebase(self, analysis_data: Dict[str, Any]) -> bool:
        """Salva análise no Firebase"""
        try:
            endpoint = f"{self.backend_url}/api/loss/analyze"
            response = requests.post(endpoint, json=analysis_data, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"✅ Análise salva no Firebase: {analysis_data['order_id']}")
                return True
            else:
                logger.error(f"❌ Erro ao salvar no Firebase: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erro ao comunicar com backend: {e}")
            return False
    
    def analyze_loss(self, iq: IQ_Option, order_id: int, ativo: str, 
                    direction: str, stake: float, setup: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Analisa um loss completo"""
        try:
            logger.info(f"🔍 Iniciando análise de loss: {ativo} | {direction} | ${stake:.2f}")
            
            # 1. Captura velas
            df = self.get_candles(iq, ativo, timeframe=60, n=100)
            if df is None or df.empty:
                logger.error("❌ Não foi possível capturar velas")
                return None
            
            logger.info(f"✅ Capturadas {len(df)} velas")
            
            # 2. Análise de contexto
            market_context = self.analyze_market_context(df)
            
            # 3. Análise da entrada
            entry_quality = self.analyze_entry_quality(df, direction)
            
            # 4. Gera análise com IA
            ai_analysis = self.generate_ai_analysis(
                market_context, entry_quality, ativo, direction, stake
            )
            
            # 5. Prepara dados para salvar
            analysis_data = {
                "order_id": str(order_id),
                "timestamp": datetime.now().isoformat(),
                "asset": ativo,
                "direction": direction.upper(),
                "stake": float(stake),
                "market_context": market_context,
                "entry_quality": entry_quality,
                "ai_analysis": ai_analysis,
                "setup": setup if setup else {},
                "candles_data": {
                    "count": len(df),
                    "last_10_closes": df['close'].tail(10).tolist(),
                    "last_10_opens": df['open'].tail(10).tolist()
                }
            }
            
            # 6. Salva no Firebase
            self.save_to_firebase(analysis_data)
            
            # 7. Log da análise
            logger.info("\n" + ai_analysis)
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"❌ Erro na análise de loss: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


# Instância global para fácil acesso
_global_analyzer = None

def get_loss_analyzer(backend_url: str = "http://localhost:8000") -> LossAnalyzer:
    """Retorna instância global do analisador"""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = LossAnalyzer(backend_url)
    return _global_analyzer
