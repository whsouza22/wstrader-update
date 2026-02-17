# -*- coding: utf-8 -*-
"""
WS_NEURAL_AI — IA com Rede Neural Profissional + Padrões de Velas 2026
✅ TensorFlow/Keras para rede neural otimizada
✅ 13+ padrões de candlestick profissionais
✅ Análise inteligente de 30 velas
✅ Sistema de aprendizado adaptativo
✅ Envio de LOSS para Firebase para análise
✅ Análise de LOSS com OpenAI GPT
"""

import os
import time
import math
import json
import logging
import requests
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from iqoptionapi.stable_api import IQ_Option

# OpenAI para análise inteligente de LOSS
try:
    from openai import OpenAI
    try:
        from config_keys import OPENAI_API_KEY
    except ImportError:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=15.0)
    OPENAI_AVAILABLE = True
except Exception:
    openai_client = None
    OPENAI_AVAILABLE = False

# TA-Lib patterns (padrões profissionais)
try:
    import talib
    from pattern_detector import detect_candlestick_patterns, TALIB_AVAILABLE, PADROES_80_PLUS
except Exception:
    talib = None
    detect_candlestick_patterns = None
    TALIB_AVAILABLE = False
    PADROES_80_PLUS = None

# Smart Memory - Sistema de bloqueio de combinações ruins
try:
    from smart_memory import SmartMemory
    SMART_MEMORY_AVAILABLE = True
except Exception:
    SmartMemory = None
    SMART_MEMORY_AVAILABLE = False

# Backend URL para enviar LOSS
BACKEND_URL = os.getenv("WS_BACKEND_URL", "http://127.0.0.1:8000")


def detect_talib_pattern_simple(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Detecta padrão TA-Lib na última vela fechada (sem filtros rígidos)."""
    if not TALIB_AVAILABLE or talib is None or PADROES_80_PLUS is None:
        return None
    if df is None or len(df) < 3:
        return None

    try:
        data = df.copy()
        data.columns = [c.lower() for c in data.columns]
        opens = data["open"].values.astype(np.float64)
        highs = data["high"].values.astype(np.float64)
        lows = data["low"].values.astype(np.float64)
        closes = data["close"].values.astype(np.float64)

        candidates = []
        for padrao_talib, (conf, nome_pt, desc_call, desc_put) in PADROES_80_PLUS.items():
            func = getattr(talib, padrao_talib, None)
            if func is None:
                continue
            res = func(opens, highs, lows, closes)
            val = res[-2] if len(res) > 1 else res[-1]
            if val == 0:
                continue
            if val > 0 and desc_call:
                candidates.append({
                    "direction": "CALL",
                    "score": float(conf),
                    "name": f"{nome_pt}_ALTA",
                    "value": float(val),
                    "weight": float(conf * abs(val))
                })
            elif val < 0 and desc_put:
                candidates.append({
                    "direction": "PUT",
                    "score": float(conf),
                    "name": f"{nome_pt}_BAIXA",
                    "value": float(val),
                    "weight": float(conf * abs(val))
                })

        if not candidates:
            return None

        return max(candidates, key=lambda x: x["weight"])
    except Exception:
        return None

# TensorFlow/Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silencia warnings do TF
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ===================== CONFIG =====================
EMAIL = os.getenv("IQ_EMAIL", "") or "wstrader@wstrader.onmicrosoft.com"
SENHA = os.getenv("IQ_PASS", "") or "P152030@w"
CONTA = os.getenv("IQ_CONTA", "PRACTICE")

TF_M1 = 60
DECIDIR_ANTES_FECHAR_SEC = int(os.getenv("WS_DECIDIR_ANTES_FECHAR", "10"))  # 10 segundos antes para analisar
N_M1 = int(os.getenv("WS_N_M1", "340"))

PAYOUT_MINIMO = int(os.getenv("WS_PAYOUT_MIN", "80"))
NUM_ATIVOS = int(os.getenv("WS_NUM_ATIVOS", "12"))
PAYOUT_REFRESH_SEC = int(os.getenv("WS_PAYOUT_REFRESH", "60"))  # Atualiza ativos a cada 60s para mais variedade

EXP_FIXA = int(os.getenv("WS_EXP_MIN", "1"))
ORDER_MODE = os.getenv("WS_ORDER_MODE", "turbo").strip().lower()  # turbo | digital | auto
VALOR_MINIMO = float(os.getenv("WS_VALOR_MINIMO", "3"))
STAKE_FIXA = float(os.getenv("WS_STAKE", "100"))

# ===================== GESTÃO DE BANCA =====================
PERCENT_BANCA = float(os.getenv("WS_PERCENT_BANCA", "1.0"))
META_LUCRO_PERCENT = float(os.getenv("WS_META_LUCRO", "7.0"))
STOP_LOSS_PERCENT = float(os.getenv("WS_STOP_LOSS", "5.0"))
USE_DYNAMIC_STAKE = (os.getenv("WS_DYNAMIC_STAKE", "1").strip() == "1")

COOLDOWN_ATIVO = int(os.getenv("WS_COOLDOWN_ATIVO", "30"))

# ===================== IA NEURAL =====================
NEURAL_ON = (os.getenv("WS_NEURAL_ON", "1").strip() == "1")
NEURAL_FILE = os.getenv("WS_NEURAL_FILE", "ws_neural_weights.json")
NEURAL_MIN_CONFIDENCE = float(os.getenv("WS_NEURAL_MIN_CONF", "0.52"))  # Reduzido para fase de aprendizado
NEURAL_USE_MIN_CONF = (os.getenv("WS_NEURAL_USE_MIN_CONF", "1").strip() == "1")
TALIB_FALLBACK_SIMPLE = (os.getenv("WS_TALIB_FALLBACK_SIMPLE", "0").strip() == "1")
NEURAL_LEARNING_RATE = float(os.getenv("WS_NEURAL_LR", "0.01"))

# ===================== FILTROS =====================
MIN_SCORE = float(os.getenv("WS_MIN_SCORE", "0.65"))  # Score mínimo para operar
MIN_PATTERN_STRENGTH = float(os.getenv("WS_MIN_PATTERN", "0.65"))  # Força mínima do padrão

# ===================== LOG =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [WS_NEURAL] %(message)s"
)
log = logging.getLogger("WS_NEURAL")

class C:
    G = "\033[92m"
    R = "\033[91m"
    Y = "\033[93m"
    B = "\033[94m"
    M = "\033[95m"
    C = "\033[96m"  # Cyan para neural
    Z = "\033[0m"

def paint(s: str, color: str) -> str:
    return f"{color}{s}{C.Z}"

_cache_ativos: List[str] = []
_cache_ativos_ts: float = 0.0
cooldown: Dict[str, float] = {}

# ===================== SMART MEMORY - BLOQUEIO DE COMBINACOES RUINS =====================
smart_memory = None
if SMART_MEMORY_AVAILABLE and SmartMemory is not None:
    smart_memory = SmartMemory()
    log.info(paint("[MEMORY] ✅ Smart Memory ativado - bloqueio de combinacoes ruins", C.G))

# ===================== REDE NEURAL PROFISSIONAL MELHORADA (KERAS) =====================
class TradingNeuralNetwork:
    """
    Rede Neural Profissional COMPLETA usando TensorFlow/Keras

    MELHORIAS:
    - Input: 440 features (7 por vela x 60 velas + 20 globais)
    - Arquitetura maior: 128 -> 64 -> 32 neuronios
    - Regularizacao L2 para evitar overfitting
    - Dropout aumentado para generalizacao
    
    APRENDE:
    - Padrões de velas (corpo, sombras, direção)
    - Topos e Fundos (estrutura de mercado)
    - Tendência (MAs, RSI)
    - Momentum e Volatilidade
    - WIN e LOSS

    Output: 2 neuronios (prob_call, prob_put) com softmax
    """

    def __init__(self, input_size: int = 460):  # MELHORADO: 460 features (incluindo S/R)
        self.input_size = input_size
        self.model = None
        self.build_model()

    def build_model(self):
        """Constroi a arquitetura da rede neural MELHORADA"""
        try:
            from tensorflow.keras.regularizers import l2
            from tensorflow.keras.initializers import GlorotUniform
            regularizer = l2(0.01)
            # Inicializador com seed para reprodutibilidade
            initializer = GlorotUniform(seed=42)
        except ImportError:
            regularizer = None
            initializer = 'glorot_uniform'

        self.model = Sequential([
            Input(shape=(self.input_size,)),

            # Camada 1 - Extracao de features (maior para mais features)
            Dense(128, activation='relu', kernel_regularizer=regularizer, kernel_initializer=initializer),
            BatchNormalization(),
            Dropout(0.4),  # AUMENTADO: mais dropout para generalizar

            # Camada 2 - Processamento
            Dense(64, activation='relu', kernel_regularizer=regularizer, kernel_initializer=initializer),
            BatchNormalization(),
            Dropout(0.3),

            # Camada 3 - Decisao
            Dense(32, activation='relu', kernel_initializer=initializer),
            BatchNormalization(),
            Dropout(0.2),

            # Output layer - Inicializa com zeros para começar 50/50
            Dense(2, activation='softmax', kernel_initializer='zeros', bias_initializer='zeros')
        ])

        # Compilar modelo com learning rate menor para estabilidade
        self.model.compile(
            optimizer=Adam(learning_rate=NEURAL_LEARNING_RATE * 0.5),  # LR menor
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def check_bias(self, test_samples: int = 10) -> Dict[str, Any]:
        """
        Verifica se a rede está viciada (sempre prevendo mesma direção)
        Retorna estatísticas de bias
        """
        calls = 0
        puts = 0
        
        # Gera amostras aleatórias para testar
        for _ in range(test_samples):
            random_input = np.random.randn(1, self.input_size).astype(np.float32)
            pred = self.model.predict(random_input, verbose=0)
            if pred[0, 0] > pred[0, 1]:
                calls += 1
            else:
                puts += 1
        
        total = calls + puts
        call_ratio = calls / total if total > 0 else 0.5
        put_ratio = puts / total if total > 0 else 0.5
        
        # Se mais de 80% para um lado, está viciada
        is_biased = call_ratio > 0.8 or put_ratio > 0.8
        
        return {
            "calls": calls,
            "puts": puts,
            "call_ratio": call_ratio,
            "put_ratio": put_ratio,
            "is_biased": is_biased,
            "bias_direction": "CALL" if call_ratio > put_ratio else "PUT"
        }
    
    def reset_output_layer(self):
        """
        Reseta apenas a camada de output para 50/50
        Mantém o conhecimento das camadas intermediárias
        """
        # Pega a última camada (output)
        output_layer = self.model.layers[-1]
        
        # Reseta para zeros (50/50 com softmax)
        weights = output_layer.get_weights()
        new_weights = [np.zeros_like(w) for w in weights]
        output_layer.set_weights(new_weights)
        
        log.info(paint("[NEURAL] 🔄 Output layer resetada para 50/50", C.Y))
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predição"""
        if X.shape[0] != 1:
            X = X.reshape(1, -1)
        
        predictions = self.model.predict(X, verbose=0)
        
        prob_call = float(predictions[0, 0])
        prob_put = float(predictions[0, 1])
        
        # Quando neural está incerta (50/50), randomiza a direção
        # Isso evita viés para um lado durante fase de aprendizado
        if abs(prob_call - prob_put) < 0.02:  # Diferença menor que 2%
            import random
            predicted = random.choice(["CALL", "PUT"])
            confidence = 0.50  # Indicar incerteza
        else:
            predicted = "CALL" if prob_call > prob_put else "PUT"
            confidence = max(prob_call, prob_put)
        
        return {
            "direction": predicted,
            "confidence": float(confidence),
            "prob_call": float(prob_call),
            "prob_put": float(prob_put)
        }
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, verbose: int = 0):
        """Treina a rede com os dados"""
        if X.shape[0] != 1:
            X = X.reshape(1, -1)
        if y.shape[0] != 1:
            y = y.reshape(1, -1)
        
        self.model.fit(X, y, epochs=epochs, verbose=verbose)
    
    def save(self, path: str):
        """Salva o modelo"""
        try:
            # Remove extensão .json e adiciona .h5 (formato Keras)
            model_path = path.replace('.json', '.h5')
            self.model.save(model_path)
        except Exception as e:
            log.warning(f"Erro ao salvar modelo: {e}")
    
    def load(self, path: str) -> bool:
        """Carrega o modelo e recompila para permitir treino"""
        try:
            model_path = path.replace('.json', '.h5')
            if not os.path.exists(model_path):
                return False
            
            self.model = load_model(model_path)
            expected = self.input_size
            loaded = self.model.input_shape[-1]
            if loaded != expected:
                log.warning(f"Modelo incompatível: esperado {expected}, carregado {loaded}. Recriando modelo.")
                self.build_model()
                return False
            
            # IMPORTANTE: Recompilar modelo após carregar para permitir treino
            # O optimizer precisa ser recriado para as novas variáveis
            self.model.compile(
                optimizer=Adam(learning_rate=NEURAL_LEARNING_RATE * 0.5),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # VERIFICAR VIÉS após carregar
            bias_info = self.check_bias(test_samples=20)
            log.info(f"[NEURAL] Verificação de viés: CALL={bias_info['call_ratio']*100:.0f}% PUT={bias_info['put_ratio']*100:.0f}%")
            
            if bias_info['is_biased']:
                log.warning(f"[NEURAL] ⚠️ Modelo VICIADO em {bias_info['bias_direction']}! Resetando output layer...")
                self.reset_output_layer()
                log.info("[NEURAL] ✅ Modelo corrigido para 50/50")
            
            return True
        except Exception as e:
            log.warning(f"Erro ao carregar modelo: {e}")
            return False

# ===================== PRE-TREINO COM DADOS HISTÓRICOS =====================
# Arquivo para salvar histórico de FEATURES REAIS de cada operação
FEATURES_HISTORY_FILE = "ws_neural_features_history.json"
MAX_FEATURES_HISTORY = 500  # Máximo de operações salvas

def load_historical_stats(json_path: str = "ws_ai_stats_m1.json") -> Optional[Dict]:
    """Carrega estatísticas históricas do JSON para pre-treino"""
    try:
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                log.info(f"[PRETRAIN] 📊 Carregado {json_path} com {len(data.get('arms', {}))} padrões")
                return data
        else:
            log.info(f"[PRETRAIN] ⚠️ Arquivo {json_path} não encontrado")
            return None
    except Exception as e:
        log.warning(f"[PRETRAIN] Erro ao carregar JSON: {e}")
        return None


def save_operation_features(
    features: np.ndarray,
    direction: str,
    is_win: bool,
    ativo: str = "",
    pattern: str = ""
) -> bool:
    """
    Salva as FEATURES REAIS de cada operação para pre-treino futuro.
    Isso permite que a neural 'lembre' das operações anteriores quando reiniciar.
    """
    try:
        # Carrega histórico existente
        history = []
        if os.path.exists(FEATURES_HISTORY_FILE):
            try:
                with open(FEATURES_HISTORY_FILE, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                history = []
        
        # Cria registro da operação
        operation = {
            "timestamp": datetime.now().isoformat(),
            "ativo": ativo,
            "direction": direction,
            "is_win": is_win,
            "pattern": pattern,
            "features": features.flatten().tolist()  # Converte numpy para lista
        }
        
        history.append(operation)
        
        # Limita tamanho do histórico (mantém mais recentes)
        if len(history) > MAX_FEATURES_HISTORY:
            history = history[-MAX_FEATURES_HISTORY:]
        
        # Salva
        with open(FEATURES_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        log.info(f"[FEATURES] 💾 Salvo: {direction} {'WIN' if is_win else 'LOSS'} | Total: {len(history)} ops")
        return True
        
    except Exception as e:
        log.warning(f"[FEATURES] Erro ao salvar: {e}")
        return False


def load_features_history() -> list:
    """Carrega histórico de features reais para pre-treino"""
    try:
        if os.path.exists(FEATURES_HISTORY_FILE):
            with open(FEATURES_HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
                log.info(f"[FEATURES] 📂 Carregado histórico com {len(history)} operações")
                return history
        return []
    except Exception as e:
        log.warning(f"[FEATURES] Erro ao carregar histórico: {e}")
        return []


def pretrain_neural_with_historical(neural: TradingNeuralNetwork, stats_path: str = "ws_ai_stats_m1.json"):
    """
    Pre-treina a rede neural usando APENAS WINS!
    A IA só aprende com padrões que FUNCIONARAM.
    """
    # 1) PRIMEIRO: Carrega features reais do histórico
    features_history = load_features_history()
    total_trained = 0
    
    if features_history:
        # IMPORTANTE: Filtra APENAS os WINs!
        wins = [op for op in features_history if op.get('is_win', False)]
        losses = [op for op in features_history if not op.get('is_win', True)]
        
        log.info(f"[PRETRAIN] 📊 Histórico: {len(wins)} WINs / {len(losses)} LOSSes")
        log.info(f"[PRETRAIN] 🧠 Treinando APENAS com {len(wins)} WINs (LOSSes ignorados)...")
        
        if not wins:
            log.info("[PRETRAIN] ⚠️ Nenhum WIN no histórico - Neural começa do zero")
        else:
            # Treina APENAS com WINs
            for op in wins:
                try:
                    features = np.array(op['features'], dtype=np.float32).reshape(1, -1)
                    
                    # Verifica se features tem tamanho correto
                    if features.shape[1] != neural.input_size:
                        continue
                    
                    direction = op.get('direction', 'CALL')
                    
                    # WIN: Reforça a direção com 90% certeza
                    if direction == "CALL":
                        y = np.array([[0.90, 0.10]])  # Reforça CALL
                    else:  # PUT
                        y = np.array([[0.10, 0.90]])  # Reforça PUT
                    
                    # Treina com 1 epoch
                    neural.train(features, y, epochs=1, verbose=0)
                    total_trained += 1
                    
                except Exception as e:
                    continue
            
            if total_trained > 0:
                log.info(f"[PRETRAIN] ✅ Pre-treino: {total_trained} WINs aprendidos!")
    else:
        log.info("[PRETRAIN] ⚠️ Sem histórico de features reais - Neural começa do zero")
    
    # 2) DEPOIS: Carrega estatísticas do JSON para ajuste fino (opcional)
    data = load_historical_stats(stats_path)
    if data and 'arms' in data:
        arms = data['arms']
        patterns_trained = 0
        
        # Ajuste fino baseado em win_rate por padrão (SÓ SE WIN_RATE > 50%)
        for pattern_key, stats in arms.items():
            n = stats.get('n', 0)
            if n < 3:  # Ignora padrões com poucas operações
                continue
            
            try:
                parts = pattern_key.split('|')
                if len(parts) < 2:
                    continue
                
                direction = parts[0]
                a = stats.get('a', 1)
                b = stats.get('b', 1)
                win_rate = a / (a + b) if (a + b) > 0 else 0.5
                
                # SÓ treina padrões com win_rate > 50% (padrões que funcionam!)
                if win_rate <= 0.50:
                    continue
                
                direction = parts[0]
                a = stats.get('a', 1)
                b = stats.get('b', 1)
                win_rate = a / (a + b) if (a + b) > 0 else 0.5
                
                # Só ajusta se win_rate muito diferente de 50%
                if abs(win_rate - 0.5) < 0.1:
                    continue
                
                # Gera features sintéticas para ajuste fino
                np.random.seed(hash(pattern_key) % (2**32))
                features = np.random.randn(1, neural.input_size).astype(np.float32)
                
                if direction == "CALL":
                    prob_call = min(0.90, max(0.10, win_rate))
                    y = np.array([[prob_call, 1 - prob_call]])
                else:
                    prob_put = min(0.90, max(0.10, win_rate))
                    y = np.array([[1 - prob_put, prob_put]])
                
                neural.train(features, y, epochs=1, verbose=0)
                patterns_trained += 1
                
            except:
                continue
        
        if patterns_trained > 0:
            log.info(f"[PRETRAIN] 📈 Ajuste fino: {patterns_trained} padrões")
    
    return total_trained


def update_json_stats(direction: str, pattern_name: str, is_win: bool, stats_path: str = "ws_ai_stats_m1.json"):
    """
    Atualiza o JSON de estatísticas após cada operação.
    Isso permite que a neural aprenda com os resultados históricos.
    """
    try:
        # Carrega dados existentes ou cria novo
        if os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"meta": {"total": 0, "global_wins": 0, "global_losses": 0}, "arms": {}}
        
        # Atualiza meta
        data["meta"]["total"] = data["meta"].get("total", 0) + 1
        if is_win:
            data["meta"]["global_wins"] = data["meta"].get("global_wins", 0) + 1
        else:
            data["meta"]["global_losses"] = data["meta"].get("global_losses", 0) + 1
        
        # Cria chave do padrão simplificada
        # Formato: DIRECTION|PATTERN_NAME
        pattern_key = f"{direction}|sc15|pb0|re0|A0|eff0|fl0|dst0|{pattern_name[:8]}" if pattern_name else f"{direction}|sc15|pb0|re0|A0|eff0|fl0|dst0"
        
        # Inicializa ou atualiza arm
        if pattern_key not in data["arms"]:
            data["arms"][pattern_key] = {"a": 1.0, "b": 1.0, "n": 0}
        
        arm = data["arms"][pattern_key]
        
        # Atualiza Beta distribution (Thompson Sampling)
        if is_win:
            arm["a"] = arm.get("a", 1.0) + 1.0
        else:
            arm["b"] = arm.get("b", 1.0) + 1.0
        arm["n"] = arm.get("n", 0) + 1
        
        # Salva JSON
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        win_rate = arm["a"] / (arm["a"] + arm["b"]) * 100
        log.info(f"[STATS] 📊 {pattern_key}: WR={win_rate:.1f}% ({arm['n']} ops)")
        
    except Exception as e:
        log.warning(f"[STATS] Erro ao atualizar JSON: {e}")


def send_loss_to_firebase(
    ativo: str,
    direction: str,
    stake: float,
    pattern_name: str = "",
    trend: str = "",
    neural_confidence: float = 0.0,
    pattern_score: float = 0.0,
    motivo: str = ""
) -> bool:
    """
    Envia dados de LOSS para o Firebase para análise futura pela IA.
    Usa OpenAI GPT para gerar análise inteligente do motivo.
    """
    try:
        # Gera order_id único baseado em timestamp
        order_id = f"loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{ativo}"
        
        # Analisa motivo do loss COM OPENAI
        ai_analysis = _analyze_loss_with_openai(
            ativo=ativo,
            direction=direction,
            pattern=pattern_name,
            trend=trend,
            neural_conf=neural_confidence,
            pattern_score=pattern_score
        )
        
        motivo_completo = ai_analysis.get("motivo", "Análise não disponível")
        sugestao = ai_analysis.get("sugestao", "Revisar contexto")
        
        # ai_analysis como string (formato esperado pelo backend)
        ai_analysis_str = f"MOTIVO: {motivo_completo} | SUGESTÃO: {sugestao}"
        
        analysis_data = {
            "order_id": order_id,
            "timestamp": datetime.now().isoformat(),
            "asset": ativo,
            "direction": direction.upper(),
            "stake": float(stake),
            "market_context": {
                "trend": trend if trend else "unknown",
                "pattern": pattern_name if pattern_name else "none"
            },
            "entry_quality": {
                "pattern": pattern_name,
                "pattern_score": pattern_score,
                "neural_confidence": neural_confidence,
                "trend": trend
            },
            "ai_analysis": ai_analysis_str,
            "setup": {
                "pattern_name": pattern_name,
                "pattern_score": pattern_score,
                "neural_confidence": neural_confidence,
                "motivo": motivo_completo,
                "sugestao": sugestao
            },
            "candles_data": {}
        }
        
        endpoint = f"{BACKEND_URL}/api/loss/analyze"
        response = requests.post(endpoint, json=analysis_data, timeout=10)
        
        if response.status_code == 200:
            log.info(paint(f"[FIREBASE] ✅ Loss enviado | {ativo} {direction} | {motivo_completo}", C.B))
            return True
        else:
            log.warning(f"[FIREBASE] Erro ao enviar loss: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        log.warning("[FIREBASE] Backend não disponível - loss não enviado")
        return False
    except Exception as e:
        log.warning(f"[FIREBASE] Erro ao enviar loss: {e}")
        return False


def _analyze_loss_with_openai(
    ativo: str,
    direction: str,
    pattern: str,
    trend: str,
    neural_conf: float,
    pattern_score: float
) -> Dict[str, str]:
    """
    Usa OpenAI GPT para analisar o motivo do LOSS de forma inteligente.
    Retorna motivo e sugestão gerados pela IA generativa.
    """
    if not OPENAI_AVAILABLE or openai_client is None:
        # Fallback para análise manual se OpenAI não disponível
        return {
            "motivo": _analyze_loss_reason_manual(direction, pattern, trend, neural_conf, pattern_score),
            "sugestao": _generate_loss_suggestion_manual(direction, pattern, trend, neural_conf)
        }
    
    try:
        prompt = f"""Analise este LOSS de operação de opções binárias e identifique o MOTIVO da perda:

DADOS DA OPERAÇÃO:
- Ativo: {ativo}
- Direção: {direction}
- Padrão detectado: {pattern if pattern else 'Nenhum'}
- Força do padrão: {pattern_score*100:.0f}%
- Confiança da neural: {neural_conf*100:.0f}%
- Tendência: {trend if trend else 'Não identificada'}

REGRAS DE ANÁLISE:
1. Se neural_confidence < 55%, a IA estava incerta
2. Se pattern_score < 70%, o padrão era fraco
3. Se direção CALL e tendência BAIXA = operou contra tendência
4. Se direção PUT e tendência ALTA = operou contra tendência

Responda em formato JSON:
{{"motivo": "motivo resumido em 1 frase", "sugestao": "sugestão para evitar no futuro em 1 frase"}}"""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um analista de trading profissional. Analise o LOSS e identifique o motivo de forma direta e objetiva. Responda APENAS em JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Tenta parsear JSON
        try:
            # Remove possíveis marcadores de código
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            result = json.loads(result_text)
            log.info(f"[OPENAI] 🧠 Análise de LOSS: {result.get('motivo', 'N/A')}")
            return result
        except json.JSONDecodeError:
            # Se não conseguiu parsear, usa o texto como motivo
            return {
                "motivo": result_text[:100],
                "sugestao": "Revisar contexto antes de operar"
            }
            
    except Exception as e:
        log.warning(f"[OPENAI] Erro na análise: {e}")
        # Fallback para análise manual
        return {
            "motivo": _analyze_loss_reason_manual(direction, pattern, trend, neural_conf, pattern_score),
            "sugestao": _generate_loss_suggestion_manual(direction, pattern, trend, neural_conf)
        }


def _analyze_loss_reason_manual(direction: str, pattern: str, trend: str, neural_conf: float, pattern_score: float) -> str:
    """Análise manual de fallback (quando OpenAI não disponível)"""
    reasons = []
    
    # Verifica se operou contra tendência
    if trend:
        if (direction == "CALL" and "BAIXA" in trend.upper()) or (direction == "PUT" and "ALTA" in trend.upper()):
            reasons.append("Operou CONTRA a tendência")
    
    # Verifica confiança neural baixa
    if neural_conf < 0.55:
        reasons.append(f"Neural incerta ({neural_conf*100:.0f}%)")
    
    # Verifica padrão fraco
    if pattern_score < 0.70:
        reasons.append(f"Padrão fraco ({pattern_score*100:.0f}%)")
    
    # Se não tem padrão
    if not pattern or pattern == "Nenhum":
        reasons.append("Sem padrão de candlestick definido")
    
    if not reasons:
        reasons.append("Movimento de mercado imprevisível")
    
    return " | ".join(reasons)


def _generate_loss_suggestion_manual(direction: str, pattern: str, trend: str, neural_conf: float) -> str:
    """Sugestão manual de fallback (quando OpenAI não disponível)"""
    suggestions = []
    
    if neural_conf < 0.55:
        suggestions.append("Aguardar neural com confiança > 55%")
    
    if trend and "CONTRA" in trend.upper():
        suggestions.append("Evitar operar contra a tendência dominante")
    
    if not pattern:
        suggestions.append("Aguardar padrão de candlestick forte")
    
    if not suggestions:
        suggestions.append("Revisar contexto de mercado antes de entrar")
    
    return " | ".join(suggestions)


# Instância global da rede neural
neural_net = TradingNeuralNetwork()

# ===================== TEMPO =====================
def seconds_to_next(tf: int) -> float:
    now = time.time()
    return tf - (now % tf)

def wait_until_minus(tf: int, seconds_before: int):
    while True:
        s = seconds_to_next(tf)
        if s <= seconds_before:
            return
        time.sleep(min(0.25, max(0.04, s - seconds_before)))

def wait_for_next_open(tf: int):
    s = seconds_to_next(tf)
    time.sleep(s + 0.12)

def end_ts_closed(tf: int) -> float:
    now = time.time()
    return now - (now % tf) - 1

# ===================== PATCH WEBSOCKET =====================
def patch_websocket_on_close():
    try:
        from iqoptionapi.ws.client import WebsocketClient
        if getattr(WebsocketClient, "__WS_PATCHED__", False):
            return
        old = WebsocketClient.on_close

        def on_close_compat(self, *args, **kwargs):
            try:
                return old(self)
            except TypeError:
                try:
                    return old(self, *args, **kwargs)
                except Exception:
                    return None
            except Exception:
                return None

        WebsocketClient.on_close = on_close_compat
        WebsocketClient.__WS_PATCHED__ = True
    except Exception:
        pass

# ===================== IQ OPTION =====================
def conectar_iq() -> IQ_Option:
    if not EMAIL or not SENHA:
        raise RuntimeError("Defina IQ_EMAIL e IQ_PASS")
    patch_websocket_on_close()
    log.info("Conectando à IQ Option...")
    iq = IQ_Option(EMAIL, SENHA)
    iq.connect()

    for _ in range(12):
        if iq.check_connect():
            break
        time.sleep(1.5)

    if not iq.check_connect():
        raise RuntimeError("Falha na conexão")

    iq.change_balance(CONTA)
    try:
        log.info(f"Conectado | Saldo: {iq.get_balance():.2f} | Conta: {CONTA}")
    except Exception:
        log.info(f"Conectado | Conta: {CONTA}")

    return iq

# Flag global para indicar reconexão recente (0 = primeira conexão, não esperar)
last_reconnect_time = 0
is_first_connection = True  # Primeira conexão não deve esperar

def ensure_connected(iq: Optional[IQ_Option]) -> IQ_Option:
    global last_reconnect_time, is_first_connection
    
    if iq is None:
        # Primeira conexão - não marca como reconexão
        if is_first_connection:
            is_first_connection = False
            return conectar_iq()
        # Reconexão após queda
        last_reconnect_time = time.time()
        log.warning("🔄 Reconectando após queda de conexão...")
        return conectar_iq()
    
    try:
        if iq.check_connect():
            return iq
    except Exception:
        pass
    
    # Conexão perdida - reconectar
    last_reconnect_time = time.time()
    log.warning("🔄 Conexão perdida! Reconectando...")
    return conectar_iq()

def safe_call(iq: IQ_Option, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        if ("10054" in msg) or ("forçado" in msg) or ("goodbye" in msg):
            ensure_connected(iq)
            return fn(*args, **kwargs)
        raise

# ===================== CANDLES =====================
def get_candles_df(iq: IQ_Option, ativo: str, timeframe: int, n: int, end_ts: Optional[float] = None) -> Optional[pd.DataFrame]:
    try:
        if end_ts is None:
            end_ts = time.time()

        candles = safe_call(iq, iq.get_candles, ativo, timeframe, n, end_ts)
        if not candles or isinstance(candles, int):
            return None

        df = pd.DataFrame(candles)
        if "from" in df.columns:
            df.rename(columns={"from": "time"}, inplace=True)
        if "min" in df.columns:
            df.rename(columns={"min": "low"}, inplace=True)
        if "max" in df.columns:
            df.rename(columns={"max": "high"}, inplace=True)

        if "time" not in df.columns:
            return None

        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)

        needed = ["open", "high", "low", "close"]
        for col in needed:
            if col not in df.columns:
                return None

        df = df[needed].dropna().sort_index()

        if len(df) < 60:
            return None
        return df
    except Exception:
        return None

# ===================== ATIVOS =====================
def obter_top_ativos_otc(iq: IQ_Option) -> List[str]:
    global _cache_ativos, _cache_ativos_ts
    now = time.time()
    if _cache_ativos and (now - _cache_ativos_ts) < PAYOUT_REFRESH_SEC:
        return _cache_ativos

    try:
        dados = safe_call(iq, iq.get_all_open_time)
        turbo = dados.get("turbo", {})
    except Exception:
        return []

    abertos = [a for a, info in turbo.items() if info.get("open", False)]
    abertos_otc = [a for a in abertos if "-OTC" in a.upper()]
    if not abertos_otc:
        abertos_otc = abertos

    filtrados = []
    for a in abertos_otc:
        try:
            payout = safe_call(iq, iq.get_digital_payout, a)
            payout = int(payout) if payout is not None else 0
        except Exception:
            payout = 0
        if payout >= PAYOUT_MINIMO:
            filtrados.append((a, payout))

    filtrados.sort(key=lambda x: x[1], reverse=True)
    top = [a for a, _ in filtrados[:NUM_ATIVOS]]
    _cache_ativos = top
    _cache_ativos_ts = now
    log.info(f"TOP ativos: {top}")
    return top

# ===================== PADRÕES DE VELAS (13+) =====================
def identify_candle_pattern(row: pd.Series, prev_row: Optional[pd.Series] = None) -> Dict[str, Any]:
    """Identifica padrões clássicos de candlestick"""
    o = float(row["open"])
    h = float(row["high"])
    l = float(row["low"])
    c = float(row["close"])
    
    body = abs(c - o)
    total_range = h - l
    
    if total_range < 1e-9:
        return {"pattern": "none", "signal": "NEUTRAL", "strength": 0.0}
    
    body_ratio = body / total_range
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    upper_shadow_ratio = upper_shadow / total_range
    lower_shadow_ratio = lower_shadow / total_range
    
    # DOJI
    if body_ratio < 0.10:
        return {"pattern": "doji", "signal": "NEUTRAL", "strength": 0.3}
    
    # HAMMER (alta)
    if body_ratio < 0.30 and lower_shadow_ratio > 0.60 and upper_shadow_ratio < 0.10:
        if c > o:
            return {"pattern": "hammer_bull", "signal": "CALL", "strength": 0.80}
        else:
            return {"pattern": "hammer_bear", "signal": "CALL", "strength": 0.70}
    
    # SHOOTING STAR (baixa)
    if body_ratio < 0.30 and upper_shadow_ratio > 0.60 and lower_shadow_ratio < 0.10:
        if c < o:
            return {"pattern": "shooting_star_bear", "signal": "PUT", "strength": 0.80}
        else:
            return {"pattern": "shooting_star_bull", "signal": "PUT", "strength": 0.70}
    
    # MARUBOZU (força)
    if body_ratio > 0.85:
        if c > o:
            return {"pattern": "marubozu_bull", "signal": "CALL", "strength": 0.85}
        else:
            return {"pattern": "marubozu_bear", "signal": "PUT", "strength": 0.85}
    
    # SPINNING TOP
    if body_ratio < 0.30 and upper_shadow_ratio > 0.30 and lower_shadow_ratio > 0.30:
        return {"pattern": "spinning_top", "signal": "NEUTRAL", "strength": 0.2}
    
    # Padrões de 2 velas
    if prev_row is not None:
        prev_o = float(prev_row["open"])
        prev_c = float(prev_row["close"])
        prev_body = abs(prev_c - prev_o)
        
        # ENGULFING BULLISH
        if prev_c < prev_o and c > o:
            if c > prev_o and o < prev_c and body > prev_body * 0.90:
                return {"pattern": "engulfing_bull", "signal": "CALL", "strength": 0.95}
        
        # ENGULFING BEARISH
        if prev_c > prev_o and c < o:
            if c < prev_o and o > prev_c and body > prev_body * 0.90:
                return {"pattern": "engulfing_bear", "signal": "PUT", "strength": 0.95}
        
        # PIERCING LINE
        if prev_c < prev_o and c > o:
            if o < prev_c and c > (prev_o + prev_c) / 2 and c < prev_o:
                return {"pattern": "piercing_line", "signal": "CALL", "strength": 0.75}
        
        # DARK CLOUD COVER
        if prev_c > prev_o and c < o:
            if o > prev_c and c < (prev_o + prev_c) / 2 and c > prev_o:
                return {"pattern": "dark_cloud", "signal": "PUT", "strength": 0.75}
    
    # Vela comum
    if c > o:
        return {"pattern": "bullish_common", "signal": "CALL", "strength": 0.45}
    elif c < o:
        return {"pattern": "bearish_common", "signal": "PUT", "strength": 0.45}
    else:
        return {"pattern": "none", "signal": "NEUTRAL", "strength": 0.0}

def analyze_patterns_sequence(df: pd.DataFrame, lookback: int = 5) -> Dict[str, Any]:
    """Analisa sequência de padrões"""
    if len(df) < lookback:
        return {"score": 0.0, "direction": "NEUTRAL", "patterns": []}
    
    recent = df.tail(lookback)
    patterns = []
    call_strength = 0.0
    put_strength = 0.0
    
    for i in range(len(recent)):
        prev_row = recent.iloc[i-1] if i > 0 else None
        curr_row = recent.iloc[i]
        
        pattern_data = identify_candle_pattern(curr_row, prev_row)
        patterns.append(pattern_data)
        
        signal = pattern_data["signal"]
        strength = pattern_data["strength"]
        
        # Peso maior para padrões recentes
        weight = 1.0 + (i / lookback) * 1.0
        
        if signal == "CALL":
            call_strength += strength * weight
        elif signal == "PUT":
            put_strength += strength * weight
    
    total = call_strength + put_strength
    if total < 0.1:
        return {"score": 0.0, "direction": "NEUTRAL", "patterns": patterns}
    
    if call_strength > put_strength:
        return {
            "score": float(call_strength / total),
            "direction": "CALL",
            "patterns": patterns,
            "call_strength": float(call_strength),
            "put_strength": float(put_strength)
        }
    else:
        return {
            "score": float(put_strength / total),
            "direction": "PUT",
            "patterns": patterns,
            "call_strength": float(call_strength),
            "put_strength": float(put_strength)
        }

# ===================== EXTRACAO DE FEATURES COMPLETA PARA NEURAL =====================
def extract_neural_features(df: pd.DataFrame, lookback: int = 60) -> Optional[np.ndarray]:
    """
    Extrai features COMPLETAS para a rede neural aprender TUDO
    
    FEATURES POR VELA (7 x 60 = 420):
    1. price_change - Variação de preço
    2. body_ratio - Proporção do corpo
    3. upper_ratio - Sombra superior (rejeição de alta)
    4. lower_ratio - Sombra inferior (rejeição de baixa)
    5. range_norm - Volatilidade
    6. direction - Direção (+1 alta, -1 baixa)
    7. momentum - Momentum 3 velas
    
    FEATURES GLOBAIS (+20):
    8-9. Tendência (alta/baixa)
    10-11. Topos/Fundos recentes
    12-15. Médias móveis
    16-20. RSI e volatilidade
    
    FEATURES DE SUPORTE/RESISTÊNCIA (+20):
    21-25. Níveis de suporte (5 níveis)
    26-30. Níveis de resistência (5 níveis)
    31-35. Distância do preço aos níveis
    36-40. Força dos níveis (toques)
    
    TOTAL: 460 features
    """
    if len(df) < lookback:
        return None

    recent = df.tail(lookback)
    features = []

    # Pre-calcula valores
    closes = recent["close"].values
    highs = recent["high"].values
    lows = recent["low"].values

    # ===== FEATURES POR VELA (7 x 60 = 420) =====
    for i in range(len(recent)):
        row = recent.iloc[i]
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])

        total_range = max(h - l, 1e-9)
        body = abs(c - o)
        upper = h - max(o, c)
        lower = min(o, c) - l

        price_change = (c - o) / max(o, 1e-9)
        body_ratio = body / total_range
        upper_ratio = upper / total_range
        lower_ratio = lower / total_range
        range_norm = total_range / max(o, 1e-9)
        direction = 1.0 if c > o else -1.0

        if i >= 2:
            momentum = (closes[i] - closes[i-2]) / max(closes[i-2], 1e-9)
        else:
            momentum = 0.0

        features.extend([
            price_change,
            body_ratio,
            upper_ratio,
            lower_ratio,
            range_norm,
            direction,
            momentum
        ])

    # ===== FEATURES GLOBAIS (+20) =====
    
    # 1. Tendência baseada em topos/fundos
    swing_highs = []
    swing_lows = []
    for i in range(2, len(recent) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append(lows[i])
    
    # Tendência: topos e fundos ascendentes = ALTA, descendentes = BAIXA
    trend_up = 0.0
    trend_down = 0.0
    if len(swing_highs) >= 2:
        if swing_highs[-1] > swing_highs[-2]:
            trend_up = 1.0
        else:
            trend_down = 1.0
    if len(swing_lows) >= 2:
        if swing_lows[-1] > swing_lows[-2]:
            trend_up += 0.5
        else:
            trend_down += 0.5
    
    features.extend([trend_up, trend_down])  # +2
    
    # 2. Quantidade de topos/fundos (normalizado)
    features.extend([
        min(len(swing_highs) / 5.0, 1.0),
        min(len(swing_lows) / 5.0, 1.0)
    ])  # +2
    
    # 3. Posição do preço atual em relação aos swings
    current_close = closes[-1]
    if swing_highs:
        dist_to_high = (max(swing_highs) - current_close) / max(current_close, 1e-9)
    else:
        dist_to_high = 0.0
    if swing_lows:
        dist_to_low = (current_close - min(swing_lows)) / max(current_close, 1e-9)
    else:
        dist_to_low = 0.0
    
    features.extend([dist_to_high, dist_to_low])  # +2
    
    # 4. Médias móveis simples (5, 10, 20 períodos)
    ma5 = np.mean(closes[-5:]) if len(closes) >= 5 else closes[-1]
    ma10 = np.mean(closes[-10:]) if len(closes) >= 10 else closes[-1]
    ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
    
    # Preço acima/abaixo das médias
    features.extend([
        1.0 if current_close > ma5 else -1.0,
        1.0 if current_close > ma10 else -1.0,
        1.0 if current_close > ma20 else -1.0,
        1.0 if ma5 > ma10 else -1.0,  # Cruzamento MAs
        1.0 if ma10 > ma20 else -1.0
    ])  # +5
    
    # 5. RSI simplificado (14 períodos)
    gains = []
    losses = []
    for i in range(1, min(15, len(closes))):
        diff = closes[-i] - closes[-i-1] if i+1 <= len(closes) else 0
        if diff > 0:
            gains.append(diff)
        else:
            losses.append(abs(diff))
    
    avg_gain = np.mean(gains) if gains else 0.001
    avg_loss = np.mean(losses) if losses else 0.001
    rs = avg_gain / max(avg_loss, 0.001)
    rsi = 100 - (100 / (1 + rs))
    rsi_norm = (rsi - 50) / 50  # Normaliza para -1 a +1
    
    features.append(rsi_norm)  # +1
    
    # 6. Volatilidade recente vs histórica
    vol_recent = np.std(closes[-5:]) if len(closes) >= 5 else 0
    vol_hist = np.std(closes[-20:]) if len(closes) >= 20 else vol_recent
    vol_ratio = vol_recent / max(vol_hist, 0.001) - 1.0  # 0 = normal, >0 = alta vol
    
    features.append(np.clip(vol_ratio, -1, 1))  # +1
    
    # 7. Força do último movimento (últimas 3 velas)
    last_move = (closes[-1] - closes[-4]) / max(closes[-4], 1e-9) if len(closes) >= 4 else 0
    features.append(np.clip(last_move * 10, -1, 1))  # +1
    
    # 8. Padrão das últimas 3 velas (sequência de direções)
    dir1 = 1.0 if closes[-1] > closes[-2] else -1.0 if len(closes) >= 2 else 0
    dir2 = 1.0 if closes[-2] > closes[-3] else -1.0 if len(closes) >= 3 else 0
    dir3 = 1.0 if closes[-3] > closes[-4] else -1.0 if len(closes) >= 4 else 0
    
    features.extend([dir1, dir2, dir3])  # +3
    
    # ===== FEATURES DE SUPORTE E RESISTÊNCIA (+20) =====
    
    # Calcula níveis de suporte e resistência baseado em pivôs
    price_range = max(highs) - min(lows)
    current_price = closes[-1]
    
    # Encontra níveis de resistência (topos significativos)
    resistance_levels = []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            resistance_levels.append(highs[i])
    
    # Encontra níveis de suporte (fundos significativos)
    support_levels = []
    for i in range(2, len(lows) - 2):
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            support_levels.append(lows[i])
    
    # Adiciona máximas/mínimas recentes como níveis importantes
    resistance_levels.append(max(highs[-10:]))  # Máxima das últimas 10 velas
    resistance_levels.append(max(highs[-20:]))  # Máxima das últimas 20 velas
    support_levels.append(min(lows[-10:]))      # Mínima das últimas 10 velas
    support_levels.append(min(lows[-20:]))      # Mínima das últimas 20 velas
    
    # Ordena e filtra níveis únicos
    resistance_levels = sorted(list(set(resistance_levels)), reverse=True)[:5]  # Top 5 resistências
    support_levels = sorted(list(set(support_levels)))[:5]  # Top 5 suportes
    
    # Preenche com zeros se não tiver níveis suficientes
    while len(resistance_levels) < 5:
        resistance_levels.append(current_price * 1.01)  # 1% acima
    while len(support_levels) < 5:
        support_levels.append(current_price * 0.99)  # 1% abaixo
    
    # Feature 1-5: Distância normalizada do preço às resistências
    for res_level in resistance_levels[:5]:
        dist = (res_level - current_price) / max(price_range, 1e-9)
        features.append(np.clip(dist, -1, 1))
    
    # Feature 6-10: Distância normalizada do preço aos suportes
    for sup_level in support_levels[:5]:
        dist = (current_price - sup_level) / max(price_range, 1e-9)
        features.append(np.clip(dist, -1, 1))
    
    # Feature 11: Preço está mais perto de suporte ou resistência?
    nearest_resistance = min(resistance_levels) if resistance_levels else current_price * 1.01
    nearest_support = max(support_levels) if support_levels else current_price * 0.99
    dist_to_resistance = nearest_resistance - current_price
    dist_to_support = current_price - nearest_support
    
    if dist_to_resistance + dist_to_support > 0:
        proximity_ratio = (dist_to_support - dist_to_resistance) / (dist_to_resistance + dist_to_support)
    else:
        proximity_ratio = 0.0
    features.append(np.clip(proximity_ratio, -1, 1))  # +1 = perto de resistência, -1 = perto de suporte
    
    # Feature 12: Quantidade de toques em resistência (força do nível)
    resistance_touches = 0
    for h in highs:
        for res in resistance_levels[:3]:
            if abs(h - res) / max(res, 1e-9) < 0.002:  # 0.2% de tolerância
                resistance_touches += 1
    features.append(min(resistance_touches / 10.0, 1.0))
    
    # Feature 13: Quantidade de toques em suporte (força do nível)
    support_touches = 0
    for l in lows:
        for sup in support_levels[:3]:
            if abs(l - sup) / max(sup, 1e-9) < 0.002:  # 0.2% de tolerância
                support_touches += 1
    features.append(min(support_touches / 10.0, 1.0))
    
    # Feature 14: Preço rompeu resistência recentemente?
    broke_resistance = 0.0
    for res in resistance_levels[:3]:
        if current_price > res and closes[-5] < res:  # Rompeu nas últimas 5 velas
            broke_resistance = 1.0
            break
    features.append(broke_resistance)
    
    # Feature 15: Preço rompeu suporte recentemente?
    broke_support = 0.0
    for sup in support_levels[:3]:
        if current_price < sup and closes[-5] > sup:  # Rompeu nas últimas 5 velas
            broke_support = 1.0
            break
    features.append(broke_support)
    
    # Feature 16: Preço testando resistência agora?
    testing_resistance = 0.0
    for res in resistance_levels[:3]:
        if abs(current_price - res) / max(res, 1e-9) < 0.003:  # 0.3% de tolerância
            testing_resistance = 1.0
            break
    features.append(testing_resistance)
    
    # Feature 17: Preço testando suporte agora?
    testing_support = 0.0
    for sup in support_levels[:3]:
        if abs(current_price - sup) / max(sup, 1e-9) < 0.003:  # 0.3% de tolerância
            testing_support = 1.0
            break
    features.append(testing_support)
    
    # Feature 18: Zona de range (preço no meio entre suporte e resistência)
    if nearest_resistance > nearest_support:
        zone_position = (current_price - nearest_support) / (nearest_resistance - nearest_support)
    else:
        zone_position = 0.5
    features.append(np.clip(zone_position * 2 - 1, -1, 1))  # -1 = suporte, 0 = meio, +1 = resistência
    
    # Feature 19: Largura da zona S/R (volatilidade estrutural)
    zone_width = (nearest_resistance - nearest_support) / max(current_price, 1e-9)
    features.append(np.clip(zone_width * 20, 0, 1))  # Normalizado
    
    # Feature 20: Tendência dos níveis de S/R
    # Se resistências estão subindo = mercado em alta
    # Se suportes estão descendo = mercado em baixa
    if len(resistance_levels) >= 2:
        res_trend = 1.0 if resistance_levels[0] > resistance_levels[1] else -1.0
    else:
        res_trend = 0.0
    if len(support_levels) >= 2:
        sup_trend = 1.0 if support_levels[0] > support_levels[1] else -1.0
    else:
        sup_trend = 0.0
    features.append((res_trend + sup_trend) / 2.0)

    # ===== FEATURES DE MANIPULAÇÃO OTC (SPIKE DETECTION) =====
    # Estas features ajudam a neural a aprender padrões de manipulação

    opens = recent["open"].values

    # Feature 21-25: Spike ratio nas últimas 5 velas
    # Spike = sombra grande vs corpo pequeno (manipulação deixa rastro)
    for i in range(-5, 0):
        if i >= -len(recent):
            o = opens[i]
            h = highs[i]
            l = lows[i]
            c = closes[i]
            body = abs(c - o)
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            total_range = max(h - l, 1e-9)
            # Spike ratio: quanto maior, mais manipulação
            spike_ratio = (upper_wick + lower_wick) / max(body, 1e-9)
            features.append(np.clip(spike_ratio / 5.0, 0, 1))  # Normalizado
        else:
            features.append(0.0)

    # Feature 26-30: False breakout nas últimas 5 velas
    # False breakout = preço vai em uma direção mas fecha no oposto
    for i in range(-5, 0):
        if i >= -len(recent):
            o = opens[i]
            h = highs[i]
            l = lows[i]
            c = closes[i]
            # Spike para cima mas fechou em baixa = false breakout bullish
            if h - o > 0 and c < o:
                false_break = (h - o) / max(abs(c - o), 1e-9)
                features.append(np.clip(false_break / 3.0, -1, 1))
            # Spike para baixo mas fechou em alta = false breakout bearish
            elif o - l > 0 and c > o:
                false_break = (o - l) / max(abs(c - o), 1e-9)
                features.append(np.clip(-false_break / 3.0, -1, 1))
            else:
                features.append(0.0)
        else:
            features.append(0.0)

    # Feature 31: Contagem de spikes recentes (últimas 10 velas)
    spike_count = 0
    for i in range(-10, 0):
        if i >= -len(recent):
            body = abs(closes[i] - opens[i])
            total_range = highs[i] - lows[i]
            if total_range > 0 and body / total_range < 0.3:  # Corpo < 30% do range = spike
                spike_count += 1
    features.append(spike_count / 10.0)

    # Feature 32: Direção dominante dos spikes (manipulação tem padrão)
    spike_up = 0
    spike_down = 0
    for i in range(-10, 0):
        if i >= -len(recent):
            upper_wick = highs[i] - max(opens[i], closes[i])
            lower_wick = min(opens[i], closes[i]) - lows[i]
            if upper_wick > lower_wick * 1.5:
                spike_up += 1
            elif lower_wick > upper_wick * 1.5:
                spike_down += 1
    features.append((spike_up - spike_down) / 10.0)  # +1 = spikes para cima, -1 = para baixo

    # Feature 33: Volatilidade recente (manipulação aumenta volatilidade)
    recent_ranges = [highs[i] - lows[i] for i in range(-5, 0) if i >= -len(recent)]
    older_ranges = [highs[i] - lows[i] for i in range(-15, -5) if i >= -len(recent)]
    if recent_ranges and older_ranges:
        vol_ratio = np.mean(recent_ranges) / max(np.mean(older_ranges), 1e-9)
        features.append(np.clip(vol_ratio - 1, -1, 1))  # 0 = normal, >0 = vol aumentando
    else:
        features.append(0.0)

    # Feature 34: Consistência direcional (manipulação quebra consistência)
    direction_changes = 0
    for i in range(-9, 0):
        if i >= -len(recent) and i-1 >= -len(recent):
            curr_dir = 1 if closes[i] > opens[i] else -1
            prev_dir = 1 if closes[i-1] > opens[i-1] else -1
            if curr_dir != prev_dir:
                direction_changes += 1
    features.append(direction_changes / 9.0)  # 1 = muitas mudanças (lateralização/manipulação)

    # Feature 35: Hora do dia (manipulação varia por horário) - simplificado
    try:
        hour = datetime.now().hour
        # Normaliza hora: 0-23 -> -1 a +1
        hour_norm = (hour - 12) / 12.0
        features.append(hour_norm)
    except:
        features.append(0.0)

    # Padding final para manter 460 features
    while len(features) < 460:
        features.append(0.0)

    # Converte para numpy (1, 460)
    features = np.array(features[:460], dtype=np.float32).reshape(1, -1)

    # Normalização
    features = np.clip(features, -3, 3) / 3.0

    return features


# ===================== ANALISE DE TOPOS E FUNDOS (ESTRUTURA DE MERCADO) =====================
def analyze_market_structure(df: pd.DataFrame, lookback: int = 30) -> dict:
    """
    Analisa a estrutura de mercado detectando TOPOS e FUNDOS

    Retorna:
    - trend: "ALTA", "BAIXA", "LATERAL"
    - swing_highs: Lista de topos recentes
    - swing_lows: Lista de fundos recentes
    - hh_hl: True se Higher Highs + Higher Lows (tendencia de alta)
    - lh_ll: True se Lower Highs + Lower Lows (tendencia de baixa)
    - last_swing: "HIGH" ou "LOW" (ultimo pivo)
    - strength: Forca da tendencia (0-1)
    """
    result = {
        "trend": "LATERAL",
        "swing_highs": [],
        "swing_lows": [],
        "hh_hl": False,
        "lh_ll": False,
        "last_swing": None,
        "strength": 0.0,
        "suggestion": "NEUTRAL"
    }

    if len(df) < lookback:
        return result

    recent = df.tail(lookback)
    highs = recent["high"].values
    lows = recent["low"].values
    closes = recent["close"].values

    # Detecta Swing Highs (topos) - ponto mais alto que os 2 vizinhos
    swing_highs = []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append((i, highs[i]))

    # Detecta Swing Lows (fundos) - ponto mais baixo que os 2 vizinhos
    swing_lows = []
    for i in range(2, len(lows) - 2):
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append((i, lows[i]))

    result["swing_highs"] = swing_highs
    result["swing_lows"] = swing_lows

    # Precisa de pelo menos 2 topos e 2 fundos para determinar tendencia
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        # Ultimos 2 topos
        last_high = swing_highs[-1][1]
        prev_high = swing_highs[-2][1]

        # Ultimos 2 fundos
        last_low = swing_lows[-1][1]
        prev_low = swing_lows[-2][1]

        # Higher Highs + Higher Lows = TENDENCIA DE ALTA
        if last_high > prev_high and last_low > prev_low:
            result["hh_hl"] = True
            result["trend"] = "ALTA"
            result["suggestion"] = "CALL"
            result["strength"] = min(1.0, (last_high - prev_high) / prev_high * 100 + 0.5)

        # Lower Highs + Lower Lows = TENDENCIA DE BAIXA
        elif last_high < prev_high and last_low < prev_low:
            result["lh_ll"] = True
            result["trend"] = "BAIXA"
            result["suggestion"] = "PUT"
            result["strength"] = min(1.0, (prev_low - last_low) / prev_low * 100 + 0.5)

        # Misto = LATERAL
        else:
            result["trend"] = "LATERAL"
            result["suggestion"] = "NEUTRAL"
            result["strength"] = 0.3

    # Determina qual foi o ultimo swing
    if swing_highs and swing_lows:
        if swing_highs[-1][0] > swing_lows[-1][0]:
            result["last_swing"] = "HIGH"
        else:
            result["last_swing"] = "LOW"

    return result


# ===================== IA BAYESIANA PARA TRADING =====================
class BayesianTrader:
    """
    IA Bayesiana para Trading de Opções Binárias

    Usa o Teorema de Bayes para calcular probabilidade de WIN:
    P(WIN|contexto) = P(contexto|WIN) * P(WIN) / P(contexto)

    Aprende com cada operação e atualiza as probabilidades.
    """

    def __init__(self, prior_win_rate: float = 0.50):
        self.prior_win_rate = prior_win_rate

        # Contadores Bayesianos por contexto
        # Formato: {contexto_key: {"wins": N, "losses": N, "alpha": a, "beta": b}}
        self.contexts = {}

        # Arquivo de persistência
        self.save_file = "ws_bayesian_memory.json"
        self.load()

    def _context_key(self, trend: str, volatility: str, pernada: str, direction: str) -> str:
        """Gera chave única para o contexto"""
        return f"{trend}_{volatility}_{pernada}_{direction}"

    def get_posterior(self, trend: str, volatility: str, pernada: str, direction: str) -> dict:
        """
        Calcula probabilidade posterior de WIN usando distribuição Beta.

        P(WIN) = alpha / (alpha + beta)

        Onde:
        - alpha = número de wins + 1 (prior)
        - beta = número de losses + 1 (prior)
        """
        key = self._context_key(trend, volatility, pernada, direction)

        if key not in self.contexts:
            # Sem dados históricos, usa prior
            return {
                "prob_win": self.prior_win_rate,
                "confidence": 0.0,  # Sem dados = sem confiança
                "n_samples": 0,
                "alpha": 1.0,
                "beta": 1.0
            }

        ctx = self.contexts[key]
        alpha = ctx.get("alpha", 1.0)
        beta = ctx.get("beta", 1.0)
        n = ctx.get("wins", 0) + ctx.get("losses", 0)

        # Probabilidade posterior (média da Beta)
        prob_win = alpha / (alpha + beta)

        # Confiança baseada no número de amostras
        # Quanto mais amostras, maior a confiança
        confidence = min(1.0, n / 20.0)  # Máximo após 20 operações

        return {
            "prob_win": prob_win,
            "confidence": confidence,
            "n_samples": n,
            "alpha": alpha,
            "beta": beta
        }

    def update(self, trend: str, volatility: str, pernada: str, direction: str, is_win: bool):
        """
        Atualiza a distribuição Beta após resultado.

        WIN: alpha += 1
        LOSS: beta += 1
        """
        key = self._context_key(trend, volatility, pernada, direction)

        if key not in self.contexts:
            self.contexts[key] = {
                "wins": 0,
                "losses": 0,
                "alpha": 1.0,  # Prior uniforme
                "beta": 1.0
            }

        if is_win:
            self.contexts[key]["wins"] += 1
            self.contexts[key]["alpha"] += 1.0
        else:
            self.contexts[key]["losses"] += 1
            self.contexts[key]["beta"] += 1.0

        # Salva após cada atualização
        self.save()

    def should_enter(self, trend: str, volatility: str, pernada: str, direction: str, min_prob: float = 0.55) -> dict:
        """
        Decide se deve entrar baseado na probabilidade Bayesiana.

        Retorna:
        - enter: True/False
        - prob_win: Probabilidade de WIN
        - confidence: Confiança na estimativa
        """
        posterior = self.get_posterior(trend, volatility, pernada, direction)

        # Precisa de probabilidade mínima E confiança mínima
        min_confidence = 0.3  # Pelo menos 6 operações

        enter = (posterior["prob_win"] >= min_prob and posterior["confidence"] >= min_confidence)

        # Se sem dados suficientes, permite entrada exploratória
        if posterior["n_samples"] < 3:
            enter = True  # Fase de exploração

        return {
            "enter": enter,
            "prob_win": posterior["prob_win"],
            "confidence": posterior["confidence"],
            "n_samples": posterior["n_samples"]
        }

    def save(self):
        """Salva estado para arquivo"""
        try:
            with open(self.save_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "contexts": self.contexts,
                    "updated_at": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            log.warning(f"[BAYES] Erro ao salvar: {e}")

    def load(self):
        """Carrega estado do arquivo"""
        try:
            if os.path.exists(self.save_file):
                with open(self.save_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.contexts = data.get("contexts", {})
                    log.info(f"[BAYES] ✅ Carregado {len(self.contexts)} contextos Bayesianos")
        except Exception as e:
            log.warning(f"[BAYES] Erro ao carregar: {e}")
            self.contexts = {}

    def get_stats(self) -> dict:
        """Retorna estatísticas gerais"""
        total_wins = sum(c.get("wins", 0) for c in self.contexts.values())
        total_losses = sum(c.get("losses", 0) for c in self.contexts.values())
        total = total_wins + total_losses

        return {
            "total_contexts": len(self.contexts),
            "total_wins": total_wins,
            "total_losses": total_losses,
            "global_win_rate": total_wins / total if total > 0 else 0.50
        }


# Instância global do trader Bayesiano
bayesian_trader = BayesianTrader()
log.info(paint("[BAYES] ✅ IA Bayesiana inicializada", C.G))


# ===================== ESTRATÉGIA PERNADA A/B (CORRIGIDA) =====================
def detect_pernada_ab(df: pd.DataFrame, min_impulso: int = 3, max_correcao: int = 3) -> dict:
    """
    ESTRATÉGIA PERNADA A/B INTELIGENTE

    REGRAS RÍGIDAS:
    1. Pernada A (Impulso): 3+ velas na mesma direção
    2. Pernada B (Correção): 1-3 velas NA DIREÇÃO OPOSTA (OBRIGATÓRIO!)
    3. Retração Fibonacci: OBRIGATÓRIO estar entre 30%-70%
    4. NÃO ENTRAR NO TOPO: Preço atual deve estar ABAIXO do topo (CALL) ou ACIMA do fundo (PUT)
    5. Vela de retomada: Última vela deve ser na direção do impulso

    SEM CORREÇÃO REAL = SEM ENTRADA!
    """
    result = {
        "signal": None,
        "pernada_a": 0,
        "pernada_b": 0,
        "impulso_dir": None,
        "correcao_ok": False,
        "retomada": False,
        "fib_zone": False,
        "confidence": 0.0,
        "reason": None,
        "retraction": 0.0,
        "no_topo": False  # Flag: está no topo (perigoso)?
    }

    if len(df) < 25:
        result["reason"] = "dados_insuficientes"
        return result

    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    current_price = closes[-1]

    # Calcula direção de cada vela
    directions = []
    for i in range(len(closes)):
        body = closes[i] - opens[i]
        avg_range = np.mean([highs[j] - lows[j] for j in range(max(0, i-5), i+1)])

        if abs(body) < avg_range * 0.1:
            directions.append(0)
        elif body > 0:
            directions.append(1)
        else:
            directions.append(-1)

    # ===== DETECTA PERNADA A (IMPULSO) =====
    impulso_start = -1
    impulso_end = -1
    impulso_dir = 0

    for i in range(len(directions) - 5, max(0, len(directions) - 25), -1):
        count = 0
        direction = directions[i]

        if direction == 0:
            continue

        for j in range(i, max(0, i - 12), -1):
            if directions[j] == direction:
                count += 1
            elif directions[j] == 0:
                continue
            else:
                break

        if count >= min_impulso:
            impulso_start = i - count + 1
            impulso_end = i
            impulso_dir = direction
            break

    if impulso_dir == 0:
        result["reason"] = "sem_impulso"
        return result

    result["pernada_a"] = abs(impulso_end - impulso_start) + 1
    result["impulso_dir"] = "CALL" if impulso_dir == 1 else "PUT"

    # ===== DETECTA PERNADA B (CORREÇÃO) - MAIS RIGOROSO =====
    correcao_count = 0
    correcao_velas = []

    for i in range(impulso_end + 1, len(directions) - 1):
        if directions[i] == -impulso_dir:
            correcao_count += 1
            correcao_velas.append(i)
        elif directions[i] == 0:
            correcao_count += 0.3  # Doji conta menos
        else:
            break

    result["pernada_b"] = int(correcao_count)

    # REGRA RÍGIDA: Precisa de pelo menos 1 vela de correção REAL
    if correcao_count < 1:
        result["reason"] = "sem_correcao_real"
        return result

    if correcao_count > max_correcao + 1:
        result["reason"] = f"correcao_muito_grande_{correcao_count:.0f}"
        return result

    # ===== VERIFICA FIBONACCI - OBRIGATÓRIO =====
    if impulso_dir == 1:  # Impulso de alta
        impulso_low = min(lows[impulso_start:impulso_end+1])
        impulso_high = max(highs[impulso_start:impulso_end+1])

        if impulso_end + 1 < len(lows):
            correcao_low = min(lows[impulso_end+1:])
        else:
            correcao_low = lows[-1]

        fib_range = impulso_high - impulso_low
        if fib_range > 0:
            retraction = (impulso_high - correcao_low) / fib_range
            result["retraction"] = retraction

            # Correção na zona de Fibonacci (30% - 70%)
            if 0.25 <= retraction <= 0.75:
                result["fib_zone"] = True

        # VERIFICA SE ESTÁ NO TOPO (perigoso!)
        dist_from_top = (impulso_high - current_price) / fib_range if fib_range > 0 else 0
        if dist_from_top < 0.15:  # Menos de 15% do topo = TOPO!
            result["no_topo"] = True
            result["reason"] = "muito_perto_do_topo"

    else:  # Impulso de baixa
        impulso_high = max(highs[impulso_start:impulso_end+1])
        impulso_low = min(lows[impulso_start:impulso_end+1])

        if impulso_end + 1 < len(highs):
            correcao_high = max(highs[impulso_end+1:])
        else:
            correcao_high = highs[-1]

        fib_range = impulso_high - impulso_low
        if fib_range > 0:
            retraction = (correcao_high - impulso_low) / fib_range
            result["retraction"] = retraction

            if 0.25 <= retraction <= 0.75:
                result["fib_zone"] = True

        # VERIFICA SE ESTÁ NO FUNDO (perigoso!)
        dist_from_bottom = (current_price - impulso_low) / fib_range if fib_range > 0 else 0
        if dist_from_bottom < 0.15:
            result["no_topo"] = True
            result["reason"] = "muito_perto_do_fundo"

    # REGRA RÍGIDA: Fibonacci obrigatório
    if not result["fib_zone"]:
        result["reason"] = f"fora_zona_fib_{result['retraction']*100:.0f}%"
        return result

    # REGRA RÍGIDA: Não entrar no topo/fundo
    if result["no_topo"]:
        return result

    result["correcao_ok"] = True

    # ===== DETECTA RETOMADA =====
    last_dir = directions[-1]
    penult_dir = directions[-2] if len(directions) > 1 else 0

    # Retomada: última vela na direção do impulso
    if last_dir == impulso_dir:
        # Verifica se penúltima foi correção ou doji
        if penult_dir == -impulso_dir or penult_dir == 0:
            result["retomada"] = True

    if not result["retomada"]:
        result["reason"] = "aguardando_retomada"
        return result

    # ===== GERA SINAL =====
    result["signal"] = "CALL" if impulso_dir == 1 else "PUT"

    # Calcula confiança
    confidence = 0.55  # Base mais alta

    # Bônus por tamanho do impulso
    if result["pernada_a"] >= 4:
        confidence += 0.08
    if result["pernada_a"] >= 5:
        confidence += 0.05

    # Bônus por correção ideal (2 velas = ideal)
    if 1.5 <= correcao_count <= 2.5:
        confidence += 0.10

    # Bônus por zona Fibonacci ideal (38%-62%)
    retraction = result.get("retraction", 0)
    if 0.35 <= retraction <= 0.65:
        confidence += 0.10

    # Bônus por retração ideal (50%)
    if 0.45 <= retraction <= 0.55:
        confidence += 0.07

    result["confidence"] = min(0.90, confidence)
    result["reason"] = f"AB_{result['pernada_a']}_{result['pernada_b']}_fib{retraction*100:.0f}%"

    return result


# ===================== ESTRATÉGIA SIMPLES: S/R + LTA/LTB =====================
def detect_sr_touch(df: pd.DataFrame, lookback: int = 60) -> dict:
    """
    ESTRATÉGIA SIMPLES: Detecta toques em Suporte/Resistência + Tendência

    CALL: Preço toca SUPORTE + Tendência ALTA (LTA)
    PUT:  Preço toca RESISTÊNCIA + Tendência BAIXA (LTB)

    Retorna:
    - signal: "CALL", "PUT" ou None
    - reason: motivo do sinal
    - sr_level: nível de S/R tocado
    - confidence: força do sinal (0-1)
    """
    result = {
        "signal": None,
        "reason": None,
        "sr_level": None,
        "confidence": 0.0,
        "support_levels": [],
        "resistance_levels": []
    }

    if len(df) < lookback:
        return result

    recent = df.tail(lookback)
    highs = recent["high"].values
    lows = recent["low"].values
    closes = recent["close"].values
    current_price = closes[-1]
    current_low = lows[-1]
    current_high = highs[-1]

    # ===== 1) DETECTA NÍVEIS DE SUPORTE/RESISTÊNCIA =====
    # Suporte: mínimas que foram testadas várias vezes
    support_levels = []
    for i in range(2, len(lows) - 2):
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            support_levels.append(lows[i])

    # Adiciona mínimas recentes
    support_levels.append(min(lows[-10:]))
    support_levels.append(min(lows[-20:]))
    support_levels = sorted(list(set(support_levels)))[:5]

    # Resistência: máximas que foram testadas várias vezes
    resistance_levels = []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            resistance_levels.append(highs[i])

    # Adiciona máximas recentes
    resistance_levels.append(max(highs[-10:]))
    resistance_levels.append(max(highs[-20:]))
    resistance_levels = sorted(list(set(resistance_levels)), reverse=True)[:5]

    result["support_levels"] = support_levels
    result["resistance_levels"] = resistance_levels

    # ===== 2) DETECTA TENDÊNCIA (LTA/LTB) =====
    # LTA: Fundos ascendentes (Higher Lows)
    # LTB: Topos descendentes (Lower Highs)

    swing_lows = []
    swing_highs = []

    for i in range(2, len(lows) - 2):
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            swing_lows.append(lows[i])
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            swing_highs.append(highs[i])

    trend = "LATERAL"
    if len(swing_lows) >= 2 and len(swing_highs) >= 2:
        # Verifica Higher Lows (LTA)
        lows_ascending = swing_lows[-1] > swing_lows[-2] if len(swing_lows) >= 2 else False
        highs_ascending = swing_highs[-1] > swing_highs[-2] if len(swing_highs) >= 2 else False

        # Verifica Lower Highs (LTB)
        lows_descending = swing_lows[-1] < swing_lows[-2] if len(swing_lows) >= 2 else False
        highs_descending = swing_highs[-1] < swing_highs[-2] if len(swing_highs) >= 2 else False

        if lows_ascending and highs_ascending:
            trend = "ALTA"  # LTA
        elif lows_descending and highs_descending:
            trend = "BAIXA"  # LTB

    # ===== 3) VERIFICA TOQUE EM S/R =====
    TOUCH_TOLERANCE = 0.001  # 0.1% de tolerância para considerar "toque"

    touched_support = None
    touched_resistance = None

    # Verifica se preço tocou suporte (mínima da vela atual perto do suporte)
    for sup in support_levels:
        if abs(current_low - sup) / max(sup, 1e-9) < TOUCH_TOLERANCE:
            touched_support = sup
            break

    # Verifica se preço tocou resistência (máxima da vela atual perto da resistência)
    for res in resistance_levels:
        if abs(current_high - res) / max(res, 1e-9) < TOUCH_TOLERANCE:
            touched_resistance = res
            break

    # ===== 4) GERA SINAL =====
    # CALL: Tocou suporte + Tendência de ALTA
    if touched_support and trend == "ALTA" and current_price > touched_support:
        result["signal"] = "CALL"
        result["reason"] = f"Toque em SUPORTE {touched_support:.5f} + LTA"
        result["sr_level"] = touched_support
        result["confidence"] = 0.80

    # PUT: Tocou resistência + Tendência de BAIXA
    elif touched_resistance and trend == "BAIXA" and current_price < touched_resistance:
        result["signal"] = "PUT"
        result["reason"] = f"Toque em RESISTÊNCIA {touched_resistance:.5f} + LTB"
        result["sr_level"] = touched_resistance
        result["confidence"] = 0.80

    return result


# ===================== DECISAO DE ENTRADA: PERNADA A/B INTELIGENTE =====================
def should_enter_trade(df: pd.DataFrame, neural_network) -> dict:
    """
    ESTRATÉGIA PERNADA A/B INTELIGENTE

    REGRAS RÍGIDAS:
    1. SÓ ENTRA com Pernada A/B VÁLIDA (com correção real!)
    2. Fibonacci OBRIGATÓRIO (30%-70%)
    3. NÃO entra no TOPO/FUNDO
    4. Bayesiano e Neural são SECUNDÁRIOS

    SEM CORREÇÃO = SEM ENTRADA!
    """
    ativo = df.attrs.get("ativo", "ATIVO") if hasattr(df, "attrs") else "ATIVO"

    # Verifica dados mínimos
    if len(df) < 30:
        log.info(paint(f"[{ativo}] Dados insuficientes ({len(df)}/30 velas)", C.Y))
        return {"enter": False, "direction": None, "reason": "dados_insuficientes"}

    # ===== 1) EXTRAI CONTEXTO DO MERCADO =====
    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values

    # Últimas 10 velas
    last_10_bodies = [closes[i] - opens[i] for i in range(-10, 0)]

    # Métricas do contexto
    bullish_count = sum(1 for b in last_10_bodies if b > 0)
    bearish_count = sum(1 for b in last_10_bodies if b < 0)

    # Volatilidade
    ranges = [highs[i] - lows[i] for i in range(-10, 0)]
    avg_range = np.mean(ranges)
    old_avg_range = np.mean([highs[i] - lows[i] for i in range(-30, -10)])
    volatility = "ALTA" if avg_range > old_avg_range * 1.3 else "NORMAL"

    # Tendência simples
    if bullish_count >= 7:
        trend = "FORTE_ALTA"
    elif bullish_count >= 5:
        trend = "ALTA"
    elif bearish_count >= 7:
        trend = "FORTE_BAIXA"
    elif bearish_count >= 5:
        trend = "BAIXA"
    else:
        trend = "LATERAL"

    log.info(paint(f"[{ativo}] 📊 Tendência: {trend} | Bullish: {bullish_count}/10 | Vol: {volatility}", C.B))

    # ===== 2) DETECTA PERNADA A/B (PRINCIPAL!) =====
    pernada = detect_pernada_ab(df, min_impulso=3, max_correcao=4)

    pernada_signal = pernada.get("signal")
    pernada_confidence = pernada.get("confidence", 0)
    pernada_reason = pernada.get("reason", "")
    retraction = pernada.get("retraction", 0)

    if pernada_signal:
        log.info(paint(f"[{ativo}] ✅ PERNADA A/B VÁLIDA: {pernada_signal}", C.G if pernada_signal == "CALL" else C.R))
        log.info(paint(f"  • Impulso: {pernada.get('pernada_a', 0)} velas | Correção: {pernada.get('pernada_b', 0)} velas", C.B))
        log.info(paint(f"  • Retração Fib: {retraction*100:.0f}% | Zona válida: {pernada.get('fib_zone', False)}", C.B))
        log.info(paint(f"  • Confiança: {pernada_confidence*100:.0f}%", C.B))
    else:
        # SEM PERNADA VÁLIDA = SEM ENTRADA!
        log.info(paint(f"[{ativo}] ⏸️ SEM setup válido: {pernada_reason}", C.Y))
        return {"enter": False, "direction": None, "reason": pernada_reason, "pernada": pernada}

    # ===== 3) CONSULTA IA BAYESIANA (SECUNDÁRIO) =====
    pernada_type = f"AB_{pernada.get('pernada_a', 0)}_{pernada.get('pernada_b', 0)}"

    bayesian_result = bayesian_trader.should_enter(
        trend=trend,
        volatility=volatility,
        pernada=pernada_type,
        direction=pernada_signal,
        min_prob=0.50  # Mínimo mais baixo, Pernada A/B é o principal
    )

    log.info(paint(f"[{ativo}] 🎲 BAYESIANO: P(WIN)={bayesian_result['prob_win']*100:.0f}% | N={bayesian_result['n_samples']}", C.C))

    # ===== 4) CONSULTA REDE NEURAL (SECUNDÁRIO) =====
    neural_result = None
    neural_confidence = 0.50

    if neural_network and NEURAL_ON:
        try:
            # Extrai features para neural (versão simplificada)
            features = extract_features_for_neural(df)

            if features is not None:
                neural_result = neural_network.predict(features)
                neural_confidence = neural_result.get("confidence", 0.50)

                log.info(paint(f"[{ativo}] 🧠 NEURAL: {neural_result.get('direction', '?')} | Conf={neural_confidence*100:.0f}%", C.M))
        except Exception as e:
            log.warning(f"[{ativo}] Erro neural: {e}")

    # ===== 5) DECISÃO FINAL: SISTEMA DE VOTAÇÃO =====
    votes = {"CALL": 0, "PUT": 0}
    total_confidence = 0
    num_signals = 0

    # Voto da Pernada A/B (peso 3 - mais importante)
    if pernada_signal and pernada_confidence >= 0.60:
        votes[pernada_signal] += 3
        total_confidence += pernada_confidence * 3
        num_signals += 3
        log.info(paint(f"  🗳️ Pernada A/B vota: {pernada_signal} (peso 3)", C.B))

    # Voto Bayesiano (peso 2)
    if bayesian_result and bayesian_result["prob_win"] >= 0.55 and direction_for_bayes:
        votes[direction_for_bayes] += 2
        total_confidence += bayesian_result["prob_win"] * 2
        num_signals += 2
        log.info(paint(f"  🗳️ Bayesiano vota: {direction_for_bayes} (peso 2)", C.B))

    # Voto Neural (peso 1)
    if neural_result and neural_confidence >= 0.55:
        neural_dir = neural_result.get("direction")
        if neural_dir in votes:
            votes[neural_dir] += 1
            total_confidence += neural_confidence
            num_signals += 1
            log.info(paint(f"  🗳️ Neural vota: {neural_dir} (peso 1)", C.B))

    # ===== 6) DECISÃO SIMPLIFICADA =====
    # Pernada A/B é o filtro PRINCIPAL - se chegou aqui, já passou nos filtros rígidos
    final_direction = pernada_signal  # Usa direção da Pernada A/B
    vote_strength = votes.get(pernada_signal, 0)

    # Calcula confiança média
    avg_confidence = total_confidence / num_signals if num_signals > 0 else pernada_confidence

    # ===== 7) FILTROS FINAIS (SIMPLES) =====

    # Verifica Smart Memory (bloqueio de combinações perdedoras)
    if smart_memory:
        pattern_name = f"AB_{pernada.get('pernada_a', 0)}_{pernada.get('pernada_b', 0)}"
        is_blocked = smart_memory.is_blocked(ativo, final_direction, pattern_name)
        if is_blocked:
            log.info(paint(f"[{ativo}] 🚫 BLOQUEADO pelo Smart Memory", C.R))
            return {"enter": False, "direction": None, "reason": "smart_memory_block"}

    # Volatilidade extrema = cuidado
    if volatility == "ALTA" and pernada_confidence < 0.60:
        log.info(paint(f"[{ativo}] ⚠️ Volatilidade ALTA + confiança baixa - AGUARDAR", C.Y))
        return {"enter": False, "direction": None, "reason": "volatilidade_alta"}

    # ===== 8) APROVADO - ENTRAR =====
    color = C.G if final_direction == "CALL" else C.R

    log.info(paint("=" * 60, color))
    log.info(paint(f"[{ativo}] ✅ ENTRAR {final_direction}", color))
    log.info(paint(f"  📈 Pernada A: {pernada.get('pernada_a', 0)} velas | Pernada B: {pernada.get('pernada_b', 0)} velas", color))
    log.info(paint(f"  📊 Retração Fib: {pernada.get('retraction', 0)*100:.0f}%", color))
    log.info(paint(f"  🎯 Confiança: {pernada_confidence*100:.0f}%", color))
    log.info(paint(f"  🎲 Bayesian P(WIN): {bayesian_result['prob_win']*100:.0f}%", color))
    log.info(paint("=" * 60, color))

    # Contexto para aprendizado
    context = {
        "trend": trend,
        "volatility": volatility,
        "pernada_type": pernada_type,
        "retraction": pernada.get("retraction", 0),
        "pernada_a": pernada.get("pernada_a", 0),
        "pernada_b": pernada.get("pernada_b", 0)
    }

    return {
        "enter": True,
        "direction": final_direction,
        "score": pernada_confidence,
        "confidence": pernada_confidence,
        "reason": f"PERNADA_AB_{final_direction}_fib{pernada.get('retraction', 0)*100:.0f}%",
        "context": context,
        "pernada": pernada,
        "bayesian": bayesian_result,
        "neural": neural_result,
        "pattern": {"direction": final_direction, "score": avg_confidence, "name": pernada_type}
    }


def extract_features_for_neural(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Extrai features para a rede neural (versão simplificada)"""
    try:
        if len(df) < 60:
            return None

        closes = df["close"].values[-60:]
        opens = df["open"].values[-60:]
        highs = df["high"].values[-60:]
        lows = df["low"].values[-60:]

        features = []

        # 7 features por vela x 60 velas = 420 features
        for i in range(60):
            body = closes[i] - opens[i]
            upper_wick = highs[i] - max(closes[i], opens[i])
            lower_wick = min(closes[i], opens[i]) - lows[i]
            candle_range = highs[i] - lows[i]

            # Normaliza
            if candle_range > 0:
                body_ratio = body / candle_range
                upper_ratio = upper_wick / candle_range
                lower_ratio = lower_wick / candle_range
            else:
                body_ratio = 0
                upper_ratio = 0
                lower_ratio = 0

            # Direção binária
            direction = 1 if body > 0 else -1 if body < 0 else 0

            # RSI local (janela de 14)
            if i >= 14:
                window = closes[i-14:i]
                changes = np.diff(window)
                gains = np.mean(np.where(changes > 0, changes, 0))
                losses_val = np.mean(np.where(changes < 0, -changes, 0))
                rs = gains / max(losses_val, 1e-9)
                rsi = 100 - (100 / (1 + rs))
                rsi_norm = (rsi - 50) / 50  # -1 a 1
            else:
                rsi_norm = 0

            # Posição relativa (onde está o preço no range recente)
            if i >= 20:
                recent_high = max(highs[i-20:i])
                recent_low = min(lows[i-20:i])
                if recent_high > recent_low:
                    position = (closes[i] - recent_low) / (recent_high - recent_low)
                else:
                    position = 0.5
            else:
                position = 0.5

            features.extend([
                body_ratio,
                upper_ratio,
                lower_ratio,
                direction,
                rsi_norm,
                position,
                0  # Placeholder para volume
            ])

        # 40 features globais adicionais
        # Tendência geral
        ma20 = np.mean(closes[-20:])
        ma50 = np.mean(closes[-50:]) if len(closes) >= 50 else ma20
        trend_strength = (ma20 - ma50) / max(ma50, 1e-9) * 100

        # Volatilidade
        atr = np.mean([highs[i] - lows[i] for i in range(-20, 0)])
        volatility = atr / max(closes[-1], 1e-9) * 100

        # Momentum
        momentum = (closes[-1] - closes[-10]) / max(closes[-10], 1e-9) * 100

        # RSI global
        changes_global = np.diff(closes[-15:])
        gains_g = np.mean(np.where(changes_global > 0, changes_global, 0))
        losses_g = np.mean(np.where(changes_global < 0, -changes_global, 0))
        rs_g = gains_g / max(losses_g, 1e-9)
        rsi_global = 100 - (100 / (1 + rs_g))

        global_features = [
            trend_strength / 10,  # Normaliza
            volatility,
            momentum / 10,
            (rsi_global - 50) / 50,
        ]

        # Padding para completar 460 features (420 + 40)
        global_features.extend([0] * (40 - len(global_features)))

        features.extend(global_features)

        return np.array(features, dtype=np.float32).reshape(1, -1)

    except Exception as e:
        log.warning(f"Erro ao extrair features: {e}")
        return None


def save_trade_to_memory(direction: str, context: dict, is_win: bool):
    """
    Salva operação na memória para aprendizado.
    Atualiza tanto a memória simples quanto a IA Bayesiana.
    """
    global trade_memory

    if 'trade_memory' not in globals():
        trade_memory = {"wins": [], "losses": []}

    # Extrai informações do contexto
    trend = context.get("trend", "LATERAL")
    volatility = context.get("volatility", "NORMAL")
    pernada_type = context.get("pernada_type", "AB_0_0")

    entry = {
        "direction": direction,
        "trend": trend,
        "volatility": volatility,
        "pernada_type": pernada_type,
        "timestamp": time.time()
    }

    if is_win:
        trade_memory["wins"].append(entry)
        trade_memory["wins"] = trade_memory["wins"][-100:]
    else:
        trade_memory["losses"].append(entry)
        trade_memory["losses"] = trade_memory["losses"][-100:]

    # Salva em arquivo
    try:
        with open("ws_trade_memory.json", "w") as f:
            json.dump(trade_memory, f, indent=2)
    except:
        pass

    # ===== ATUALIZA IA BAYESIANA =====
    bayesian_trader.update(
        trend=trend,
        volatility=volatility,
        pernada=pernada_type,
        direction=direction,
        is_win=is_win
    )

    # Log da atualização
    posterior = bayesian_trader.get_posterior(trend, volatility, pernada_type, direction)
    log.info(paint(
        f"[BAYES] Atualizado: {trend}_{pernada_type}_{direction} = P(WIN)={posterior['prob_win']*100:.0f}% (N={posterior['n_samples']})",
        C.G if is_win else C.R
    ))



# ===================== GESTÃO =====================
def calcular_stake_dinamico(iq: IQ_Option, base_stake: float) -> float:
    if not USE_DYNAMIC_STAKE:
        return float(max(VALOR_MINIMO, base_stake))
    try:
        saldo = float(iq.get_balance())
        stake = (saldo * PERCENT_BANCA) / 100.0
        return float(max(VALOR_MINIMO, stake))
    except Exception:
        return float(max(VALOR_MINIMO, base_stake))

def verificar_meta_atingida(saldo_inicial: float, saldo_atual: float) -> Tuple[bool, float]:
    lucro = saldo_atual - saldo_inicial
    lucro_percent = (lucro / saldo_inicial) * 100.0
    if lucro_percent >= META_LUCRO_PERCENT:
        return True, lucro_percent
    if lucro_percent <= -STOP_LOSS_PERCENT:
        return True, lucro_percent
    return False, lucro_percent

# ===================== ORDEM =====================
def enviar_ordem(iq: IQ_Option, ativo: str, direcao: str, stake: float) -> Optional[Tuple[str, int]]:
    d = "call" if direcao == "CALL" else "put"
    valor = float(max(VALOR_MINIMO, stake))

    if ORDER_MODE in ("turbo", "auto"):
        try:
            ok, op_id = safe_call(iq, iq.buy, valor, ativo, d, int(EXP_FIXA))
            if ok and op_id:
                return ("turbo", int(op_id))
            if ORDER_MODE == "turbo":
                return None
        except Exception:
            if ORDER_MODE == "turbo":
                return None

    if ORDER_MODE in ("digital", "auto"):
        try:
            ok, op_id = safe_call(iq, iq.buy_digital_spot, ativo, valor, d, int(EXP_FIXA))
            if ok and op_id:
                return ("digital", int(op_id))
        except Exception:
            pass

    return None

def wait_result(iq: IQ_Option, op_type: str, op_id: int) -> float:
    while True:
        try:
            if op_type == "turbo":
                ok, res = safe_call(iq, iq.check_win_v4, op_id)
                if ok:
                    return float(res)
            else:
                res = safe_call(iq, iq.check_win_digital_v2, op_id)
                if isinstance(res, (int, float)):
                    return float(res)
        except Exception as e:
            log.error(f"Erro ao checar resultado: {e}")
        
        time.sleep(1)


# ===================== MAIN =====================
def main():
    iq: Optional[IQ_Option] = None
    iq = ensure_connected(iq)

    log.info(paint("=" * 60, C.C))
    log.info(paint("🧠 WS_NEURAL_AI - PERNADA A/B + BAYESIANO + TENSORFLOW", C.C))
    log.info(paint("=" * 60, C.C))
    log.info(paint("✅ Estratégia PERNADA A/B (Impulso + Correção)", C.G))
    log.info(paint("✅ IA Bayesiana (Probabilidade Posterior)", C.G))
    log.info(paint("✅ Rede Neural TensorFlow/Keras (3 camadas)", C.G))
    log.info(paint("✅ Sistema de Votação (2/3 confirmações)", C.M))
    log.info(paint("✅ Aprendizado em tempo real", C.M))

    # Mostra estatísticas da IA Bayesiana
    bayes_stats = bayesian_trader.get_stats()
    log.info(paint(f"🎲 Bayesiano: {bayes_stats['total_contexts']} contextos | WR Global: {bayes_stats['global_win_rate']*100:.0f}%", C.C))

    if NEURAL_ON:
        model_path = NEURAL_FILE.replace('.json', '.h5')
        loaded = neural_net.load(NEURAL_FILE)
        
        if loaded:
            log.info(paint(f"🧠 Modelo carregado de {model_path}", C.C))
        else:
            log.info(paint("🧠 Novo modelo inicializado (sem pesos salvos)", C.Y))
        
        # PRE-TREINO com FEATURES REAIS de operações anteriores
        # Isso faz a IA "LEMBRAR" do que aprendeu nas sessões anteriores!
        log.info(paint("=" * 50, C.B))
        log.info(paint("🔄 CARREGANDO MEMÓRIA DA IA...", C.C))
        
        pretrained = pretrain_neural_with_historical(neural_net, "ws_ai_stats_m1.json")
        
        if pretrained > 0:
            log.info(paint(f"✅ IA LEMBROU {pretrained} operações reais!", C.G))
            log.info(paint("💡 A IA NÃO está burra - ela carregou aprendizado anterior", C.G))
            # Salva modelo com pre-treino incorporado
            neural_net.save(NEURAL_FILE)
        else:
            log.info(paint("⚠️ IA sem memória anterior - Iniciando do ZERO", C.Y))
            log.info(paint("💡 Cada operação vai criar memória para próximas sessões", C.Y))
        
        log.info(paint("=" * 50, C.B))

    try:
        saldo_inicial = float(iq.get_balance())
        log.info(paint(f"💰 SALDO: {saldo_inicial:.2f} | META: {META_LUCRO_PERCENT:.1f}%", C.G))
    except Exception:
        saldo_inicial = 10000.0

    total = 0
    wins = 0

    while True:
        iq = ensure_connected(iq)

        try:
            saldo_atual = float(iq.get_balance())
            deve_parar, lucro_percent = verificar_meta_atingida(saldo_inicial, saldo_atual)
            if deve_parar:
                lucro_abs = saldo_atual - saldo_inicial
                if lucro_percent >= META_LUCRO_PERCENT:
                    log.info(paint(f"🎯 META! Lucro: {lucro_abs:.2f} ({lucro_percent:.2f}%)", C.G))
                else:
                    log.info(paint(f"🛑 STOP! Perda: {lucro_abs:.2f} ({lucro_percent:.2f}%)", C.R))
                
                # Salva modelo antes de sair
                if NEURAL_ON:
                    neural_net.save(NEURAL_FILE)
                    model_path = NEURAL_FILE.replace('.json', '.h5')
                    log.info(paint(f"💾 Modelo salvo em {model_path}", C.C))
                break
        except Exception:
            pass

        # PROTEÇÃO: Após reconexão, aguarda 60 segundos antes de entrar
        if last_reconnect_time > 0 and (time.time() - last_reconnect_time) < 60:
            log.info(paint("⏳ Aguardando estabilização após reconexão (60s)...", C.Y))
            time.sleep(10)
            continue

        ativos = obter_top_ativos_otc(iq)
        if not ativos:
            time.sleep(10)
            continue

        # Analisa faltando poucos segundos para fechar a vela
        wait_until_minus(TF_M1, DECIDIR_ANTES_FECHAR_SEC)

        best_setup = None
        best_score = 0.0

        for ativo in ativos:
            if ativo in cooldown and (time.time() - cooldown[ativo]) < COOLDOWN_ATIVO:
                continue

            df = get_candles_df(iq, ativo, TF_M1, 100, end_ts=end_ts_closed(TF_M1))
            if df is None:
                continue

            # Passa o nome do ativo para o dataframe (para logs)
            df.attrs["ativo"] = ativo

            # Analise com Neural + Padroes (SEMPRE mostra logs)
            decision = should_enter_trade(df, neural_net)

            if decision["enter"]:
                setup_score = decision.get("score", 0.5)
                if setup_score > best_score:
                    best_score = setup_score
                    best_setup = (ativo, decision)
                    
                    # OTIMIZAÇÃO: Se score bom (>=0.60), para de procurar e entra!
                    if setup_score >= 0.60:
                        log.info(paint(f"🎯 Setup forte encontrado! Parando análise...", C.G))
                        break

        if best_setup is None:
            log.info(paint("[SKIP] Nenhum setup aprovado neste ciclo", C.Y))
            # IMPORTANTE: Espera o PRÓXIMO MINUTO para analisar novamente
            # NÃO fica re-analisando na mesma vela!
            wait_for_next_open(TF_M1)
            continue

        ativo, decision = best_setup
        direction = decision["direction"]
        score = decision.get("score", 0.5)
        confidence = decision.get("confidence", 0.5)
        
        neural_info = decision.get("neural", {})
        pattern_info = decision.get("pattern", {})
        
        log.info(paint(
            f"[SINAL] {ativo} -> {direction} | score={score:.2f} | conf={confidence:.2f}",
            C.G if direction == "CALL" else C.R
        ))
        log.info(paint(
            f"  🧠 Neural: {neural_info.get('direction', 'N/A')} ({neural_info.get('confidence', 0):.2f})",
            C.C
        ))
        value_tag = f" v={pattern_info.get('value')}" if pattern_info.get("value") is not None else ""
        log.info(paint(
            f"  📊 Pattern: {pattern_info.get('direction', 'N/A')} ({pattern_info.get('score', 0):.2f}) {pattern_info.get('name', '')}{value_tag}",
            C.B
        ))

        # IMPORTANTE: Espera a VIRADA DA VELA para entrar (segundo 00:00)
        # Entrada EXATA na abertura da nova vela
        seconds_left = seconds_to_next(TF_M1)

        if seconds_left > 0.1:
            log.info(paint(f"⏱️ Aguardando virada da vela... ({seconds_left:.1f}s)", C.Y))
            # Espera até EXATAMENTE a virada (sem delay extra)
            time.sleep(max(0, seconds_left - 0.05))  # -0.05s para compensar latência

        # Entra IMEDIATAMENTE na virada (sem delay)
        log.info(paint(f"🎯 ENTRANDO NA VIRADA! (segundo 00)", C.G))

        stake = calcular_stake_dinamico(iq, STAKE_FIXA)
        
        op = enviar_ordem(iq, ativo, direction, stake)
        if not op:
            log.error(paint(f"[{ativo}] ❌ Falha ao enviar ordem", C.R))
            cooldown[ativo] = time.time()
            continue

        op_type, op_id = op
        log.info(paint(f"[{ativo}] ✅ ORDEM {direction} ({op_type}) stake={stake:.2f}", C.G))

        res = wait_result(iq, op_type, op_id)
        
        # Pega contexto da decisão para salvar na memória
        trade_context = decision.get("context", {})

        total += 1
        if res > 0:
            wins += 1
            log.info(paint(f"[{ativo}] ✅ WIN +{res:.2f}$", C.G))
            
            # SALVA WIN NA MEMÓRIA (para aprendizado futuro)
            save_trade_to_memory(direction, trade_context, is_win=True)
            log.info(paint(f"[MEMORY] ✅ WIN salvo: {trade_context.get('trend', '')} + {direction}", C.G))
            
            # Treina neural com WIN
            if NEURAL_ON:
                features = extract_neural_features(df, lookback=60)
                if features is not None:
                    y = np.array([[1.0, 0.0]]) if direction == "CALL" else np.array([[0.0, 1.0]])
                    neural_net.train(features, y, epochs=5)

        elif res < 0:
            log.info(paint(f"[{ativo}] ❌ LOSS {res:.2f}$", C.R))
            
            # SALVA LOSS NA MEMÓRIA (para evitar contextos similares)
            save_trade_to_memory(direction, trade_context, is_win=False)
            log.info(paint(f"[MEMORY] ⚠️ LOSS salvo: {trade_context.get('trend', '')} + {direction} - Contexto será evitado!", C.Y))
        
        # Salva modelo periodicamente
        if NEURAL_ON and total % 10 == 0:
            neural_net.save(NEURAL_FILE)
            log.info(paint(f"💾 Checkpoint: modelo salvo ({total} trades)", C.B))

        acc = (wins / max(1, total)) * 100.0
        log.info(paint(f"📊 TRADES={total} WINS={wins} ACC={acc:.2f}%\n", C.B))

        cooldown[ativo] = time.time()

if __name__ == "__main__":
    main()
