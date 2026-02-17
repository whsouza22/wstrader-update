# -*- coding: utf-8 -*-
"""
WS_NEURAL_BRAIN — IA DE VERDADE com Rede Neural + Sistema de Pensamento

🧠 ARQUITETURA:
├── LSTM: Análise temporal de padrões (30 candles)
├── Dense Network: Análise de features do setup
├── Ensemble: Combina múltiplos modelos para consenso
├── Pensamento Multi-Etapa: Analisa antes de decidir
└── Aprendizado Contínuo: Melhora com cada trade

✅ DIFERENÇAS DA IA SIMPLES:
- Rede Neural REAL (não apenas Bayes)
- Memory Replay para aprender padrões
- Confiança calculada por consenso
- Sistema de "pensamento" que explica decisões
- Threshold dinâmico baseado em histórico

Requisitos:
pip install iqoptionapi pandas numpy tensorflow scikit-learn
"""

# ===================== SILENCIA TENSORFLOW E WARNINGS =====================
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suprime INFO, WARNING, ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desativa mensagens oneDNN
warnings.filterwarnings('ignore')  # Silencia todos os warnings do Python/Keras

import time
import math
import json
import logging
import pickle
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

# ===================== NEURAL IMPORTS =====================
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')  # Silencia logs do TensorFlow
    # Silencia logs absl do TensorFlow
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.optimizers import Adam
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
except Exception:
    NEURAL_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
except Exception:
    SKLEARN_AVAILABLE = False

from iqoptionapi.stable_api import IQ_Option

# ===================== CONFIG =====================
EMAIL = os.getenv("IQ_EMAIL", "") or "wstrader@wstrader.onmicrosoft.com"
SENHA = os.getenv("IQ_PASS", "") or "P152030@w"
CONTA = os.getenv("IQ_CONTA", "REAL")

# ===================== TIMEFRAME M5 =====================
TF_M5 = 300  # 5 minutos = 300 segundos
DECIDIR_ANTES_FECHAR_SEC = int(os.getenv("WS_DECIDIR_ANTES_FECHAR", "30"))  # 30s antes do fechamento
N_CANDLES = int(os.getenv("WS_N_CANDLES", "200"))  # 200 candles M5 = ~16 horas

PAYOUT_MINIMO = int(os.getenv("WS_PAYOUT_MIN", "80"))
NUM_ATIVOS = int(os.getenv("WS_NUM_ATIVOS", "12"))
PAYOUT_REFRESH_SEC = int(os.getenv("WS_PAYOUT_REFRESH", "180"))

EXP_FIXA = int(os.getenv("WS_EXP_MIN", "5"))
VALOR_MINIMO = float(os.getenv("WS_VALOR_MINIMO", "3"))
STAKE_FIXA = float(os.getenv("WS_STAKE", "5"))

# ===================== GESTÃO DE BANCA =====================
PERCENT_BANCA = float(os.getenv("WS_PERCENT_BANCA", "1.0"))
META_LUCRO_PERCENT = float(os.getenv("WS_META_LUCRO", "1.5"))
STOP_LOSS_PERCENT = float(os.getenv("WS_STOP_LOSS", "3.0"))
USE_DYNAMIC_STAKE = (os.getenv("WS_DYNAMIC_STAKE", "1").strip() == "1")

COOLDOWN_ATIVO = int(os.getenv("WS_COOLDOWN_ATIVO", "20"))

# ===================== NEURAL BRAIN CONFIG =====================
NEURAL_ENABLED = (os.getenv("WS_NEURAL_ON", "1").strip() == "1")
NEURAL_MODEL_PATH = os.getenv("WS_NEURAL_MODEL", "ws_neural_brain.h5")
NEURAL_SCALER_PATH = os.getenv("WS_NEURAL_SCALER", "ws_neural_scaler.pkl")
NEURAL_MEMORY_PATH = os.getenv("WS_NEURAL_MEMORY", "ws_neural_memory.json")
NEURAL_STATS_PATH = os.getenv("WS_NEURAL_STATS", "ws_neural_stats.json")

# Configuração da Rede Neural
NEURAL_LOOKBACK = 30  # Candles para análise temporal (LSTM)
NEURAL_FEATURES = 52  # Features do setup
NEURAL_MIN_CONFIDENCE = float(os.getenv("WS_NEURAL_MIN_CONF", "0.65"))  # 65% confiança mínima
NEURAL_CONSENSUS_MIN = float(os.getenv("WS_NEURAL_CONSENSUS", "0.60"))  # 60% consenso ensemble
NEURAL_MIN_SAMPLES_TRAIN = int(os.getenv("WS_NEURAL_MIN_TRAIN", "50"))  # Mínimo para treinar
NEURAL_RETRAIN_EVERY = int(os.getenv("WS_NEURAL_RETRAIN", "25"))  # Retreina a cada N trades

# Memória de experiência
MEMORY_MAX_SIZE = 2000  # Máximo de experiências guardadas
MEMORY_BATCH_SIZE = 32  # Batch para treinamento

# Pensamento Multi-Etapa
THOUGHT_STAGES = 5  # Etapas de análise antes de decidir

# ===================== PERNADA B CONFIG =====================
IMPULSO_MIN_ATR = float(os.getenv("WS_IMPULSO_MIN_ATR", "0.50"))
IMPULSO_JANELA_MIN = int(os.getenv("WS_IMP_JMIN", "3"))
IMPULSO_JANELA_MAX = int(os.getenv("WS_IMP_JMAX", "15"))

PULLBACK_MIN = int(os.getenv("WS_PB_MIN", "1"))
PULLBACK_MAX = int(os.getenv("WS_PB_MAX", "6"))

RETR_MIN = float(os.getenv("WS_RETR_MIN", "0.10"))
RETR_MAX = float(os.getenv("WS_RETR_MAX", "0.85"))

BREAK_MARGIN_ATR = float(os.getenv("WS_BREAK_MARGIN_ATR", "0.01"))
MAX_BREAK_DISTANCE_ATR = float(os.getenv("WS_MAX_BREAK_DIST_ATR", "0.40"))

MIN_EFF_A = float(os.getenv("WS_MIN_EFF_A", "0.40"))
CHOP_LOOKBACK = int(os.getenv("WS_CHOP_LB", "28"))
MAX_COLOR_FLIPS_FRAC = float(os.getenv("WS_MAX_FLIPS", "0.80"))
MIN_NET_GROSS_EFF = float(os.getenv("WS_MIN_NETGROSS", "0.10"))

COMP_LOOKBACK = int(os.getenv("WS_COMP_LB", "18"))
MIN_RANGE_ATR = float(os.getenv("WS_MIN_RANGE_ATR", "0.50"))

LATE_LOOKBACK = int(os.getenv("WS_LATE_LB", "18"))
MAX_LATE_EXT_ATR = float(os.getenv("WS_MAX_LATE_EXT_ATR", "12.0"))

MIN_BODY_FRAC_BREAK = float(os.getenv("WS_MIN_BODY_FRAC", "0.10"))
MAX_WICK_AGAINST = float(os.getenv("WS_MAX_WICK_AGAINST", "0.75"))

SPIKE_RANGE_ATR = float(os.getenv("WS_SPIKE_RANGE_ATR", "1.35"))
SPIKE_WICK_FRAC = float(os.getenv("WS_SPIKE_WICK_FRAC", "0.62"))
SPIKE_COOLDOWN_MIN = int(os.getenv("WS_SPIKE_COOLDOWN_MIN", "6"))

SR_LOOKBACK = int(os.getenv("WS_SR_LOOKBACK", "220"))
SR_CLUSTER_ATR = float(os.getenv("WS_SR_CLUSTER_ATR", "0.45"))
SR_MIN_TOUCHES_STRONG = int(os.getenv("WS_SR_MIN_TOUCHES", "3"))
SR_TOP_LEVELS = int(os.getenv("WS_SR_TOP_LEVELS", "6"))
SR_CHECK_NEAR = int(os.getenv("WS_SR_CHECK_NEAR", "2"))
SR_BLOCK_DIST_ATR = float(os.getenv("WS_SR_BLOCK_ATR", "0.65"))

# ===================== LOG =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [WS_NEURAL] %(message)s"
)
log = logging.getLogger("WS_NEURAL")

class C:
    G = "\033[92m"  # Verde
    R = "\033[91m"  # Vermelho
    Y = "\033[93m"  # Amarelo
    B = "\033[94m"  # Azul
    M = "\033[95m"  # Magenta
    C = "\033[96m"  # Cyan
    W = "\033[97m"  # Branco
    Z = "\033[0m"   # Reset

def paint(s: str, color: str) -> str:
    return f"{color}{s}{C.Z}"

def dir_color(direction: str) -> str:
    return C.G if direction == "CALL" else (C.R if direction == "PUT" else C.Y)

_cache_ativos: List[str] = []
_cache_ativos_ts: float = 0.0

cooldown: Dict[str, float] = {}
cooldown_spike: Dict[str, float] = {}


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         🧠 NEURAL BRAIN (IA REAL)                          ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class NeuralBrain:
    """
    🧠 CÉREBRO NEURAL - IA de verdade com:
    - LSTM para padrões temporais
    - Dense Network para features
    - Ensemble de modelos (consenso)
    - Aprendizado contínuo
    - Sistema de pensamento multi-etapa
    """
    
    def __init__(self):
        self.lstm_model = None
        self.dense_model = None
        self.ensemble_rf = None
        self.ensemble_gb = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.temporal_scaler = MinMaxScaler() if SKLEARN_AVAILABLE else None
        
        # Memória de experiências (replay buffer)
        self.memory = deque(maxlen=MEMORY_MAX_SIZE)
        
        # Estatísticas de performance
        self.stats = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "neural_accuracy": 0.0,
            "ensemble_accuracy": 0.0,
            "last_retrain": None,
            "model_version": 0,
            "confidence_history": [],
            "thought_history": []
        }
        
        # Carregar modelos existentes
        self._load_models()
        self._load_memory()
        self._load_stats()
        
        log.info(paint("🧠 Neural Brain inicializado", C.M))
        if self.lstm_model:
            log.info(paint(f"   ├── LSTM Model: CARREGADO (v{self.stats['model_version']})", C.C))
        else:
            log.info(paint("   ├── LSTM Model: AGUARDANDO PRÉ-TREINO", C.Y))
        log.info(paint(f"   ├── Memória: {len(self.memory)}/{MEMORY_MAX_SIZE} experiências", C.C))
        log.info(paint(f"   └── Stats: {self.stats['total_trades']} trades, {self.stats['wins']} wins", C.C))
        
        # Flag para indicar se precisa de pré-treino
        self.needs_pretrain = (
            self.lstm_model is None and 
            self.dense_model is None and 
            len(self.memory) < NEURAL_MIN_SAMPLES_TRAIN
        )
    
    def pretrain_with_history(self, iq: 'IQ_Option', ativos: List[str], n_candles: int = 900):
        """
        🎓 PRÉ-TREINAMENTO COM DADOS HISTÓRICOS
        
        Usa candles históricos para simular trades e treinar a IA
        ANTES de começar a operar com dinheiro real.
        
        Args:
            iq: Conexão com IQ Option
            ativos: Lista de ativos para treinar
            n_candles: Número de candles para analisar (default: 900)
        """
        if not self.needs_pretrain:
            log.info(paint("✅ IA já treinada, pulando pré-treino", C.G))
            return
        
        log.info(paint("═" * 60, C.M))
        log.info(paint("🎓 INICIANDO PRÉ-TREINAMENTO COM HISTÓRICO", C.M))
        log.info(paint(f"   ├── Ativos: {len(ativos)}", C.C))
        log.info(paint(f"   ├── Candles por ativo: {n_candles}", C.C))
        log.info(paint(f"   └── Objetivo: Coletar experiências para treinar IA", C.C))
        log.info(paint("═" * 60, C.M))
        
        total_experiences = 0
        total_wins = 0
        total_losses = 0
        
        for ativo in ativos[:6]:  # Limita a 6 ativos para não demorar muito
            log.info(paint(f"   📊 Analisando histórico de {ativo}...", C.B))
            
            try:
                # Busca dados históricos
                df = self._get_historical_data(iq, ativo, n_candles)
                if df is None or len(df) < 300:
                    log.warning(paint(f"      ⚠️ Dados insuficientes para {ativo}", C.Y))
                    continue
                
                # Simula trades no histórico
                experiences = self._simulate_trades_on_history(df, ativo)
                
                wins = sum(1 for e in experiences if e.get('win', False))
                losses = len(experiences) - wins
                
                total_experiences += len(experiences)
                total_wins += wins
                total_losses += losses
                
                # Adiciona à memória
                for exp in experiences:
                    self.memory.append(exp)
                
                winrate = (wins / max(1, len(experiences))) * 100
                log.info(paint(f"      ✅ {ativo}: {len(experiences)} trades simulados | WR: {winrate:.1f}%", C.G))
                
            except Exception as e:
                log.warning(paint(f"      ❌ Erro em {ativo}: {e}", C.R))
                continue
        
        log.info(paint("═" * 60, C.M))
        log.info(paint(f"📊 RESUMO DO PRÉ-TREINO:", C.M))
        log.info(paint(f"   ├── Total experiências: {total_experiences}", C.C))
        log.info(paint(f"   ├── Wins simulados: {total_wins}", C.G))
        log.info(paint(f"   ├── Losses simulados: {total_losses}", C.R))
        if total_experiences > 0:
            wr = (total_wins / total_experiences) * 100
            log.info(paint(f"   └── Winrate histórico: {wr:.1f}%", C.C))
        log.info(paint("═" * 60, C.M))
        
        # Salva memória
        self._save_memory()
        
        # Treina modelos se tiver dados suficientes
        if len(self.memory) >= NEURAL_MIN_SAMPLES_TRAIN:
            log.info(paint("🎓 Dados suficientes! Treinando modelos neurais...", C.M))
            self.train_models()
            self.needs_pretrain = False
        else:
            log.info(paint(f"⚠️ Ainda faltam dados: {len(self.memory)}/{NEURAL_MIN_SAMPLES_TRAIN}", C.Y))
    
    def _get_historical_data(self, iq: 'IQ_Option', ativo: str, n_candles: int) -> Optional[pd.DataFrame]:
        """Busca dados históricos de um ativo em M5."""
        try:
            end_ts = time.time() - TF_M5  # Garante candle M5 fechado
            candles = iq.get_candles(ativo, TF_M5, n_candles, end_ts)
            
            if not candles or len(candles) < 100:
                return None
            
            df = pd.DataFrame(candles)
            
            # Normaliza colunas
            if "from" in df.columns:
                df.rename(columns={"from": "time"}, inplace=True)
            if "min" in df.columns:
                df.rename(columns={"min": "low"}, inplace=True)
            if "max" in df.columns:
                df.rename(columns={"max": "high"}, inplace=True)
            
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            df = df[["open", "high", "low", "close"]].dropna().sort_index()
            
            return df
            
        except Exception as e:
            log.warning(f"Erro ao buscar histórico de {ativo}: {e}")
            return None
    
    def _simulate_trades_on_history(self, df: pd.DataFrame, ativo: str) -> List[Dict]:
        """
        Simula trades no histórico e verifica resultados reais.
        
        Para cada ponto do histórico onde detectamos um setup:
        1. Registra o setup
        2. Verifica o candle SEGUINTE para ver se seria WIN ou LOSS
        """
        from datetime import datetime
        experiences = []
        
        # Precisa de pelo menos 300 candles
        if len(df) < 300:
            return experiences
        
        # Percorre o histórico (começa do candle 250 para ter contexto)
        for i in range(250, len(df) - 2):  # -2 para ter o candle de resultado
            try:
                # Pega subset até o ponto atual (como se fosse em tempo real)
                df_subset = df.iloc[:i+1].copy()
                
                # Calcula ATR
                atr_val = self._calc_atr(df_subset)
                if atr_val < 1e-9:
                    continue
                
                # Tenta detectar setup (versão simplificada)
                setup = self._detect_simple_setup(df_subset, atr_val)
                
                if not setup.get('trade', False):
                    continue
                
                direction = setup.get('dir', 'NEUTRAL')
                if direction == 'NEUTRAL':
                    continue
                
                # ═══ SIMULA O RESULTADO ═══
                # Pega o candle SEGUINTE (onde a ordem seria executada)
                entry_candle = df.iloc[i + 1]
                entry_price = float(entry_candle['open'])
                
                # Verifica resultado no fechamento
                close_price = float(entry_candle['close'])
                
                if direction == 'CALL':
                    win = close_price > entry_price
                else:  # PUT
                    win = close_price < entry_price
                
                # Cria experiência
                temporal_features = self.extract_temporal_features(df_subset)
                setup_features = self.extract_setup_features(setup, df_subset, atr_val)
                
                experience = {
                    "timestamp": datetime.now().isoformat(),
                    "ativo": ativo,
                    "simulated": True,
                    "temporal_features": temporal_features.tolist() if temporal_features is not None else None,
                    "setup_features": setup_features.tolist(),
                    "setup": {k: v for k, v in setup.items() if k != 'reasons'},
                    "direction": direction,
                    "result": 1.0 if win else -1.0,
                    "win": win,
                    "entry_price": entry_price,
                    "close_price": close_price
                }
                
                experiences.append(experience)
                
                # Limita a ~100 experiências por ativo para não sobrecarregar
                if len(experiences) >= 100:
                    break
                    
            except Exception:
                continue
        
        return experiences
    
    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calcula ATR."""
        if len(df) < period + 2:
            return 0.0
        
        sub = df.tail(period + 2)
        h = sub["high"].to_numpy(float)
        l = sub["low"].to_numpy(float)
        c = sub["close"].to_numpy(float)
        
        tr = np.maximum(
            h[1:] - l[1:],
            np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1]))
        )
        return float(np.mean(tr[-period:]))
    
    def _detect_simple_setup(self, df: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
        """
        Detecção simplificada de setup para o pré-treino.
        Usa lógica similar ao pernada_b mas mais leve.
        """
        if len(df) < 50:
            return {"trade": False}
        
        # Análise das últimas 20 velas
        recent = df.tail(20)
        closes = recent["close"].to_numpy(float)
        opens = recent["open"].to_numpy(float)
        highs = recent["high"].to_numpy(float)
        lows = recent["low"].to_numpy(float)
        
        # Detecta direção predominante
        bullish = sum(1 for i in range(len(recent)) if closes[i] > opens[i])
        bearish = len(recent) - bullish
        
        # Momentum recente
        momentum = (closes[-1] - closes[0]) / max(atr_val, 1e-9)
        
        # Última vela
        last = recent.iloc[-1]
        body = abs(last['close'] - last['open'])
        rng = last['high'] - last['low']
        body_ratio = body / max(rng, 1e-9)
        
        # Não entra se vela muito fraca
        if body_ratio < 0.15:
            return {"trade": False}
        
        # Direção baseada no momentum
        if momentum > 0.5 and bullish >= 12:
            direction = "CALL"
            setup_score = min(0.95, 0.50 + momentum * 0.1 + (bullish - 10) * 0.02)
        elif momentum < -0.5 and bearish >= 12:
            direction = "PUT"
            setup_score = min(0.95, 0.50 + abs(momentum) * 0.1 + (bearish - 10) * 0.02)
        else:
            return {"trade": False}
        
        # Calcula features simplificadas
        market_quality = abs(bullish - bearish) / 20.0
        
        return {
            "trade": True,
            "dir": direction,
            "score": setup_score,
            "market_quality": market_quality,
            "entry_confidence": setup_score * 0.9,
            "momentum": momentum,
            "body_ratio": body_ratio,
            "bullish_ratio": bullish / 20.0,
            "pb_len": 2,
            "retr": 0.4,
            "A_atr": abs(momentum),
            "effA": 0.6,
            "flips": 0.4,
            "comp": 1.0,
            "late": 2.0,
            "distBreak": 0.2,
            "confluence_bonus": 0.0,
            "trend_strength": abs(momentum) * 0.5,
            "entry_alignment": (bullish if direction == "CALL" else bearish) / 20.0,
            "risk_atr": 0.5,
            "lt_confluence": 0.0,
            "has_lt": False
        }
    
    def _build_lstm_model(self) -> keras.Model:
        """Constrói modelo LSTM para análise temporal de candles."""
        if not NEURAL_AVAILABLE:
            return None
        
        model = models.Sequential([
            # LSTM layers para padrões temporais
            layers.LSTM(64, return_sequences=True, input_shape=(NEURAL_LOOKBACK, 5)),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            
            # Dense layers para decisão
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Probabilidade de WIN
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_dense_model(self) -> keras.Model:
        """Constrói modelo Dense para análise de features do setup."""
        if not NEURAL_AVAILABLE:
            return None
        
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(NEURAL_FEATURES,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _load_models(self):
        """Carrega modelos salvos se existirem."""
        try:
            if os.path.exists(NEURAL_MODEL_PATH) and NEURAL_AVAILABLE:
                self.lstm_model = keras.models.load_model(NEURAL_MODEL_PATH)
                log.info(paint(f"✅ LSTM carregado: {NEURAL_MODEL_PATH}", C.G))
            
            dense_path = NEURAL_MODEL_PATH.replace('.h5', '_dense.h5')
            if os.path.exists(dense_path) and NEURAL_AVAILABLE:
                self.dense_model = keras.models.load_model(dense_path)
                log.info(paint(f"✅ Dense carregado: {dense_path}", C.G))
            
            if os.path.exists(NEURAL_SCALER_PATH) and SKLEARN_AVAILABLE:
                with open(NEURAL_SCALER_PATH, 'rb') as f:
                    scalers = pickle.load(f)
                    self.scaler = scalers.get('feature_scaler', StandardScaler())
                    self.temporal_scaler = scalers.get('temporal_scaler', MinMaxScaler())
                log.info(paint(f"✅ Scalers carregados: {NEURAL_SCALER_PATH}", C.G))
            
            # Ensemble models
            ensemble_path = NEURAL_MODEL_PATH.replace('.h5', '_ensemble.pkl')
            if os.path.exists(ensemble_path) and SKLEARN_AVAILABLE:
                with open(ensemble_path, 'rb') as f:
                    ensemble = pickle.load(f)
                    self.ensemble_rf = ensemble.get('rf')
                    self.ensemble_gb = ensemble.get('gb')
                log.info(paint(f"✅ Ensemble carregado: {ensemble_path}", C.G))
                
        except Exception as e:
            log.warning(paint(f"⚠️ Erro ao carregar modelos: {e}", C.Y))
    
    def _save_models(self):
        """Salva todos os modelos."""
        try:
            # Silencia warnings durante o save
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                if self.lstm_model and NEURAL_AVAILABLE:
                    self.lstm_model.save(NEURAL_MODEL_PATH)
                
                if self.dense_model and NEURAL_AVAILABLE:
                    dense_path = NEURAL_MODEL_PATH.replace('.h5', '_dense.h5')
                    self.dense_model.save(dense_path)
            
            if self.scaler and SKLEARN_AVAILABLE:
                with open(NEURAL_SCALER_PATH, 'wb') as f:
                    pickle.dump({
                        'feature_scaler': self.scaler,
                        'temporal_scaler': self.temporal_scaler
                    }, f)
            
            if self.ensemble_rf and SKLEARN_AVAILABLE:
                ensemble_path = NEURAL_MODEL_PATH.replace('.h5', '_ensemble.pkl')
                with open(ensemble_path, 'wb') as f:
                    pickle.dump({
                        'rf': self.ensemble_rf,
                        'gb': self.ensemble_gb
                    }, f)
                    
            log.info(paint("💾 Modelos salvos", C.G))
        except Exception as e:
            log.warning(paint(f"⚠️ Erro ao salvar modelos: {e}", C.Y))
    
    def _load_memory(self):
        """Carrega memória de experiências."""
        try:
            if os.path.exists(NEURAL_MEMORY_PATH):
                with open(NEURAL_MEMORY_PATH, 'r') as f:
                    data = json.load(f)
                    for exp in data.get('experiences', [])[-MEMORY_MAX_SIZE:]:
                        self.memory.append(exp)
        except Exception as e:
            log.warning(paint(f"⚠️ Erro ao carregar memória: {e}", C.Y))
    
    def _save_memory(self):
        """Salva memória de experiências."""
        try:
            with open(NEURAL_MEMORY_PATH, 'w') as f:
                json.dump({
                    'experiences': list(self.memory)[-MEMORY_MAX_SIZE:],
                    'saved_at': datetime.now().isoformat()
                }, f)
        except Exception:
            pass
    
    def _load_stats(self):
        """Carrega estatísticas."""
        try:
            if os.path.exists(NEURAL_STATS_PATH):
                with open(NEURAL_STATS_PATH, 'r') as f:
                    saved = json.load(f)
                    self.stats.update(saved)
        except Exception:
            pass
    
    def _save_stats(self):
        """Salva estatísticas."""
        try:
            with open(NEURAL_STATS_PATH, 'w') as f:
                json.dump(self.stats, f, indent=2, default=str)
        except Exception:
            pass
    
    def extract_temporal_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extrai features temporais das últimas N candles para o LSTM.
        Retorna: (NEURAL_LOOKBACK, 5) -> [open, high, low, close, volume_proxy]
        """
        if len(df) < NEURAL_LOOKBACK:
            return None
        
        sub = df.tail(NEURAL_LOOKBACK)
        
        # Normaliza OHLC pelo primeiro valor do período
        base_price = float(sub['open'].iloc[0])
        
        temporal = np.zeros((NEURAL_LOOKBACK, 5))
        
        for i, (_, row) in enumerate(sub.iterrows()):
            temporal[i, 0] = (row['open'] - base_price) / max(base_price, 1e-9) * 100  # % change
            temporal[i, 1] = (row['high'] - base_price) / max(base_price, 1e-9) * 100
            temporal[i, 2] = (row['low'] - base_price) / max(base_price, 1e-9) * 100
            temporal[i, 3] = (row['close'] - base_price) / max(base_price, 1e-9) * 100
            temporal[i, 4] = (row['high'] - row['low'])  # Range como proxy de volume
        
        # Normaliza com MinMaxScaler
        if self.temporal_scaler:
            try:
                temporal = self.temporal_scaler.fit_transform(temporal)
            except:
                pass
        
        return temporal
    
    def extract_setup_features(self, setup: Dict[str, Any], df: pd.DataFrame, atr_val: float) -> np.ndarray:
        """
        Extrai 52 features do setup para o modelo Dense.
        """
        features = np.zeros(NEURAL_FEATURES)
        
        # 1-5: Features do setup básico
        features[0] = float(setup.get('score', 0.0))
        features[1] = float(setup.get('pb_len', 0)) / 6.0  # Normalizado
        features[2] = float(setup.get('retr', 0.0))
        features[3] = float(setup.get('A_atr', 0.0)) / 5.0
        features[4] = float(setup.get('effA', 0.0))
        
        # 6-10: Features de contexto
        features[5] = float(setup.get('flips', 0.0))
        features[6] = float(setup.get('comp', 0.0)) / 3.0
        features[7] = float(setup.get('late', 0.0)) / 10.0
        features[8] = float(setup.get('distBreak', 0.0))
        features[9] = float(setup.get('market_quality', 0.0))
        
        # 11-15: Features de entrada
        features[10] = float(setup.get('entry_confidence', 0.0))
        features[11] = float(setup.get('entry_momentum', 0.0))
        features[12] = float(setup.get('entry_alignment', 0.0))
        features[13] = float(setup.get('risk_atr', 0.0)) / 2.0
        features[14] = float(setup.get('lt_confluence', 0.0))
        
        # 16-20: Features de tendência
        features[15] = float(setup.get('trend_strength', 0.0))
        features[16] = 1.0 if setup.get('has_lt', False) else 0.0
        features[17] = float(setup.get('confluence_bonus', 0.0))
        features[18] = 1.0 if setup.get('dir') == 'CALL' else 0.0
        features[19] = 1.0 if setup.get('dir') == 'PUT' else 0.0
        
        # 21-35: Análise estatística dos candles
        if df is not None and len(df) >= 20:
            recent = df.tail(20)
            closes = recent['close'].to_numpy(float)
            highs = recent['high'].to_numpy(float)
            lows = recent['low'].to_numpy(float)
            
            # Momentum
            features[20] = (closes[-1] - closes[0]) / max(atr_val, 1e-9) / 20.0
            features[21] = (closes[-1] - closes[-5]) / max(atr_val, 1e-9) / 5.0
            features[22] = (closes[-1] - closes[-10]) / max(atr_val, 1e-9) / 10.0
            
            # Volatilidade
            features[23] = np.std(closes) / max(atr_val, 1e-9)
            features[24] = np.mean(highs - lows) / max(atr_val, 1e-9)
            
            # Tendência
            bullish = sum(1 for i in range(len(recent)) if closes[i] > recent['open'].iloc[i])
            features[25] = bullish / len(recent)
            
            # Higher highs / Lower lows
            hh = sum(1 for i in range(5, len(recent)) if highs[i] > max(highs[i-5:i]))
            ll = sum(1 for i in range(5, len(recent)) if lows[i] < min(lows[i-5:i]))
            features[26] = hh / max(1, len(recent) - 5)
            features[27] = ll / max(1, len(recent) - 5)
            
            # RSI (calculado manualmente)
            diffs = np.diff(closes)
            gains = np.where(diffs > 0, diffs, 0)
            losses = np.where(diffs < 0, -diffs, 0)
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
            rs = avg_gain / max(avg_loss, 1e-9)
            features[28] = 100 - (100 / (1 + rs)) if rs > 0 else 50
            features[28] /= 100  # Normaliza 0-1
            
            # Candle patterns
            last = recent.iloc[-1]
            body = abs(last['close'] - last['open'])
            rng = last['high'] - last['low']
            features[29] = body / max(rng, 1e-9)  # Body ratio
            features[30] = (last['high'] - max(last['open'], last['close'])) / max(rng, 1e-9)  # Upper wick
            features[31] = (min(last['open'], last['close']) - last['low']) / max(rng, 1e-9)  # Lower wick
            
            # Moving averages
            ma5 = np.mean(closes[-5:])
            ma10 = np.mean(closes[-10:])
            ma20 = np.mean(closes)
            features[32] = 1.0 if closes[-1] > ma5 else 0.0
            features[33] = 1.0 if closes[-1] > ma10 else 0.0
            features[34] = 1.0 if closes[-1] > ma20 else 0.0
        
        # 36-45: Features de S/R
        features[35] = float(setup.get('near_support', 0.0))
        features[36] = float(setup.get('near_resistance', 0.0))
        features[37] = float(setup.get('sr_strength', 0.0))
        
        # 46-52: Features adicionais de contexto
        features[38] = float(self.stats.get('neural_accuracy', 0.5))
        features[39] = float(self.stats.get('ensemble_accuracy', 0.5))
        features[40] = min(1.0, len(self.memory) / MEMORY_MAX_SIZE)
        features[41] = atr_val / 0.001 if atr_val < 0.01 else atr_val  # ATR normalizado
        
        # Hora do dia (pode influenciar volatilidade)
        hour = datetime.now().hour
        features[42] = hour / 24.0
        features[43] = 1.0 if 9 <= hour <= 17 else 0.0  # Horário de mercado
        
        # Day of week
        dow = datetime.now().weekday()
        features[44] = dow / 6.0
        features[45] = 1.0 if dow < 5 else 0.0  # Dia útil
        
        # Performance recente
        recent_trades = list(self.memory)[-20:] if len(self.memory) >= 20 else list(self.memory)
        if recent_trades:
            recent_wins = sum(1 for t in recent_trades if t.get('result', 0) > 0)
            features[46] = recent_wins / len(recent_trades)
        
        # Features de padrão específico
        features[47] = float(setup.get('impulso_score', 0.0))
        features[48] = float(setup.get('retr_score', 0.0))
        features[49] = float(setup.get('chop_penalty', 0.0))
        features[50] = float(setup.get('entry_score', 0.0))
        features[51] = float(setup.get('risk_penalty', 0.0))
        
        return features
    
    def _predict_lstm(self, temporal_features: np.ndarray) -> float:
        """Predição do modelo LSTM."""
        if self.lstm_model is None or temporal_features is None:
            return 0.5
        
        try:
            X = temporal_features.reshape(1, NEURAL_LOOKBACK, 5)
            pred = float(self.lstm_model.predict(X, verbose=0)[0][0])
            return np.clip(pred, 0.0, 1.0)
        except Exception:
            return 0.5
    
    def _predict_dense(self, setup_features: np.ndarray) -> float:
        """Predição do modelo Dense."""
        if self.dense_model is None:
            return 0.5
        
        try:
            X = setup_features.reshape(1, -1)
            if self.scaler:
                try:
                    X = self.scaler.transform(X)
                except:
                    pass
            pred = float(self.dense_model.predict(X, verbose=0)[0][0])
            return np.clip(pred, 0.0, 1.0)
        except Exception:
            return 0.5
    
    def _predict_ensemble_rf(self, setup_features: np.ndarray) -> float:
        """Predição do Random Forest."""
        if self.ensemble_rf is None:
            return 0.5
        
        try:
            X = setup_features.reshape(1, -1)
            pred = float(self.ensemble_rf.predict_proba(X)[0][1])
            return np.clip(pred, 0.0, 1.0)
        except Exception:
            return 0.5
    
    def _predict_ensemble_gb(self, setup_features: np.ndarray) -> float:
        """Predição do Gradient Boosting."""
        if self.ensemble_gb is None:
            return 0.5
        
        try:
            X = setup_features.reshape(1, -1)
            pred = float(self.ensemble_gb.predict_proba(X)[0][1])
            return np.clip(pred, 0.0, 1.0)
        except Exception:
            return 0.5
    
    def think(self, setup: Dict[str, Any], df: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
        """
        🧠 SISTEMA DE PENSAMENTO MULTI-ETAPA
        
        Analisa o setup em múltiplas etapas antes de decidir.
        Retorna análise completa com explicação das decisões.
        """
        thought_process = {
            "stages": [],
            "conclusion": None,
            "confidence": 0.0,
            "should_trade": False,
            "reasoning": []
        }
        
        # ═══════════════════════════════════════════════════════════════
        # ETAPA 1: ANÁLISE TEMPORAL (LSTM)
        # ═══════════════════════════════════════════════════════════════
        temporal_features = self.extract_temporal_features(df)
        lstm_prob = self._predict_lstm(temporal_features)
        
        stage1 = {
            "name": "ANÁLISE TEMPORAL (LSTM)",
            "description": "Analisa padrões nos últimos 30 candles",
            "probability": lstm_prob,
            "verdict": "POSITIVO" if lstm_prob > 0.55 else ("NEUTRO" if lstm_prob > 0.45 else "NEGATIVO")
        }
        thought_process["stages"].append(stage1)
        
        if lstm_prob > 0.60:
            thought_process["reasoning"].append(f"📈 Padrão temporal favorável ({lstm_prob:.0%})")
        elif lstm_prob < 0.40:
            thought_process["reasoning"].append(f"📉 Padrão temporal desfavorável ({lstm_prob:.0%})")
        
        # ═══════════════════════════════════════════════════════════════
        # ETAPA 2: ANÁLISE DE FEATURES (Dense Network)
        # ═══════════════════════════════════════════════════════════════
        setup_features = self.extract_setup_features(setup, df, atr_val)
        dense_prob = self._predict_dense(setup_features)
        
        stage2 = {
            "name": "ANÁLISE DE SETUP (Dense)",
            "description": "Analisa as 52 features do setup",
            "probability": dense_prob,
            "verdict": "POSITIVO" if dense_prob > 0.55 else ("NEUTRO" if dense_prob > 0.45 else "NEGATIVO")
        }
        thought_process["stages"].append(stage2)
        
        if dense_prob > 0.60:
            thought_process["reasoning"].append(f"✅ Setup de qualidade ({dense_prob:.0%})")
        elif dense_prob < 0.40:
            thought_process["reasoning"].append(f"❌ Setup fraco ({dense_prob:.0%})")
        
        # ═══════════════════════════════════════════════════════════════
        # ETAPA 3: ENSEMBLE (Random Forest + Gradient Boosting)
        # ═══════════════════════════════════════════════════════════════
        rf_prob = self._predict_ensemble_rf(setup_features)
        gb_prob = self._predict_ensemble_gb(setup_features)
        ensemble_prob = (rf_prob + gb_prob) / 2.0 if (rf_prob != 0.5 or gb_prob != 0.5) else 0.5
        
        stage3 = {
            "name": "ENSEMBLE (RF + GB)",
            "description": "Consenso entre Random Forest e Gradient Boosting",
            "probability": ensemble_prob,
            "rf_prob": rf_prob,
            "gb_prob": gb_prob,
            "verdict": "POSITIVO" if ensemble_prob > 0.55 else ("NEUTRO" if ensemble_prob > 0.45 else "NEGATIVO")
        }
        thought_process["stages"].append(stage3)
        
        # Verifica consenso
        consensus = abs(rf_prob - gb_prob) < 0.15
        if consensus and ensemble_prob > 0.55:
            thought_process["reasoning"].append(f"🤝 Modelos em consenso ({ensemble_prob:.0%})")
        elif not consensus:
            thought_process["reasoning"].append(f"⚠️ Modelos divergentes (RF={rf_prob:.0%} vs GB={gb_prob:.0%})")
        
        # ═══════════════════════════════════════════════════════════════
        # ETAPA 4: ANÁLISE DE CONTEXTO (regras baseadas em conhecimento)
        # ═══════════════════════════════════════════════════════════════
        context_score = 0.5
        context_reasons = []
        
        # Score do setup
        setup_score = float(setup.get('score', 0.0))
        if setup_score > 0.70:
            context_score += 0.15
            context_reasons.append(f"Score alto: {setup_score:.2f}")
        elif setup_score < 0.50:
            context_score -= 0.10
            context_reasons.append(f"Score baixo: {setup_score:.2f}")
        
        # Qualidade do mercado
        market_qual = float(setup.get('market_quality', 0.0))
        if market_qual > 0.60:
            context_score += 0.10
            context_reasons.append(f"Mercado limpo: {market_qual:.2f}")
        elif market_qual < 0.30:
            context_score -= 0.15
            context_reasons.append(f"Mercado sujo: {market_qual:.2f}")
        
        # Confiança da entrada
        entry_conf = float(setup.get('entry_confidence', 0.0))
        if entry_conf > 0.65:
            context_score += 0.12
            context_reasons.append(f"Entrada confiante: {entry_conf:.2f}")
        elif entry_conf < 0.45:
            context_score -= 0.10
            context_reasons.append(f"Entrada fraca: {entry_conf:.2f}")
        
        # Linha de tendência
        if setup.get('has_lt', False) and float(setup.get('lt_confluence', 0)) > 0.6:
            context_score += 0.10
            context_reasons.append("Confluência com LT")
        
        context_score = np.clip(context_score, 0.0, 1.0)
        
        stage4 = {
            "name": "ANÁLISE DE CONTEXTO",
            "description": "Regras baseadas em conhecimento de trading",
            "probability": context_score,
            "factors": context_reasons,
            "verdict": "POSITIVO" if context_score > 0.55 else ("NEUTRO" if context_score > 0.45 else "NEGATIVO")
        }
        thought_process["stages"].append(stage4)
        
        for reason in context_reasons:
            thought_process["reasoning"].append(f"📊 {reason}")
        
        # ═══════════════════════════════════════════════════════════════
        # ETAPA 5: DECISÃO FINAL (Combinação ponderada)
        # ═══════════════════════════════════════════════════════════════
        
        # Pesos para cada componente
        weights = {
            'lstm': 0.25,
            'dense': 0.30,
            'ensemble': 0.25,
            'context': 0.20
        }
        
        # Ajusta pesos baseado em quantos modelos temos
        if self.lstm_model is None:
            weights['lstm'] = 0
            weights['dense'] += 0.10
            weights['ensemble'] += 0.10
            weights['context'] += 0.05
        
        if self.dense_model is None:
            weights['dense'] = 0
            weights['context'] += 0.30
        
        if self.ensemble_rf is None:
            weights['ensemble'] = 0
            weights['context'] += 0.25
        
        # Normaliza pesos
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calcula probabilidade final ponderada
        final_prob = (
            weights['lstm'] * lstm_prob +
            weights['dense'] * dense_prob +
            weights['ensemble'] * ensemble_prob +
            weights['context'] * context_score
        )
        
        # Calcula confiança baseada na concordância
        probs = [p for p in [lstm_prob, dense_prob, ensemble_prob, context_score] 
                 if p != 0.5]  # Ignora modelos não treinados
        
        if len(probs) >= 2:
            # Confiança = 1 - variância normalizada dos modelos
            confidence = 1.0 - (np.std(probs) * 2)
            confidence = np.clip(confidence, 0.3, 0.95)
        else:
            confidence = 0.5
        
        # Decisão final
        should_trade = (
            final_prob >= NEURAL_MIN_CONFIDENCE and
            confidence >= NEURAL_CONSENSUS_MIN and
            setup_score >= 0.50
        )
        
        # Se não tem modelos treinados, usa apenas regras RELAXADAS para coletar dados
        is_learning_phase = all(m is None for m in [self.lstm_model, self.dense_model, self.ensemble_rf])
        
        if is_learning_phase:
            # FASE DE APRENDIZADO: Ser mais agressivo para coletar dados
            # Score alto do setup é o mais importante
            if setup_score >= 0.75:
                # Score muito alto: sempre entra para aprender
                should_trade = True
                confidence = min(0.80, setup_score)
                thought_process["reasoning"].append("🎓 MODO APRENDIZADO: Score muito alto, entrando para coletar dados")
            elif setup_score >= 0.60:
                # Score bom: entra se contexto não for muito ruim
                should_trade = (market_qual >= 0.30 or entry_conf >= 0.40)
                confidence = min(0.70, setup_score)
                thought_process["reasoning"].append("🎓 MODO APRENDIZADO: Score bom, verificando contexto mínimo")
            elif setup_score >= 0.50:
                # Score médio: precisa de contexto razoável
                should_trade = (
                    context_score >= 0.50 and
                    (market_qual >= 0.35 or entry_conf >= 0.45)
                )
                confidence = context_score
                thought_process["reasoning"].append("🎓 MODO APRENDIZADO: Score médio, verificando contexto")
            else:
                should_trade = False
                confidence = 0.4
            
            final_prob = setup_score
            thought_process["reasoning"].append("⚠️ Usando apenas regras (IA ainda não treinada)")
        
        stage5 = {
            "name": "DECISÃO FINAL",
            "description": "Combinação ponderada de todos os modelos",
            "probability": final_prob,
            "confidence": confidence,
            "should_trade": should_trade,
            "weights_used": weights,
            "verdict": "ENTRAR" if should_trade else "NÃO ENTRAR"
        }
        thought_process["stages"].append(stage5)
        
        # Conclusão
        if should_trade:
            thought_process["reasoning"].append(
                f"✅ DECISÃO: ENTRAR ({final_prob:.0%} prob, {confidence:.0%} conf)"
            )
        else:
            reasons = []
            if final_prob < NEURAL_MIN_CONFIDENCE:
                reasons.append(f"prob baixa ({final_prob:.0%} < {NEURAL_MIN_CONFIDENCE:.0%})")
            if confidence < NEURAL_CONSENSUS_MIN:
                reasons.append(f"conf baixa ({confidence:.0%} < {NEURAL_CONSENSUS_MIN:.0%})")
            if setup_score < 0.50:
                reasons.append(f"score baixo ({setup_score:.2f})")
            
            thought_process["reasoning"].append(
                f"❌ DECISÃO: NÃO ENTRAR ({', '.join(reasons)})"
            )
        
        thought_process["conclusion"] = {
            "final_probability": float(final_prob),
            "confidence": float(confidence),
            "should_trade": should_trade,
            "direction": setup.get('dir', 'NEUTRAL')
        }
        thought_process["confidence"] = confidence
        thought_process["should_trade"] = should_trade
        
        return thought_process
    
    def add_experience(self, setup: Dict[str, Any], df: pd.DataFrame, 
                       atr_val: float, result: float, thought: Dict[str, Any]):
        """
        Adiciona experiência à memória para aprendizado.
        """
        temporal_features = self.extract_temporal_features(df)
        setup_features = self.extract_setup_features(setup, df, atr_val)
        
        experience = {
            "timestamp": datetime.now().isoformat(),
            "temporal_features": temporal_features.tolist() if temporal_features is not None else None,
            "setup_features": setup_features.tolist(),
            "setup": {k: v for k, v in setup.items() if k != 'reasons'},  # Sem reasons (muito grande)
            "direction": setup.get('dir'),
            "result": float(result),
            "win": result > 0,
            "thought_confidence": thought.get('confidence', 0.5),
            "thought_probability": thought.get('conclusion', {}).get('final_probability', 0.5)
        }
        
        self.memory.append(experience)
        
        # Atualiza stats
        self.stats["total_trades"] += 1
        if result > 0:
            self.stats["wins"] += 1
        else:
            self.stats["losses"] += 1
        
        # Salva memória e stats
        self._save_memory()
        self._save_stats()
        
        # Verifica se deve retreinar
        if self.stats["total_trades"] % NEURAL_RETRAIN_EVERY == 0:
            if len(self.memory) >= NEURAL_MIN_SAMPLES_TRAIN:
                log.info(paint("🔄 Iniciando retreinamento da IA...", C.M))
                self.train_models()
    
    def train_models(self):
        """
        Treina todos os modelos com as experiências acumuladas.
        """
        if len(self.memory) < NEURAL_MIN_SAMPLES_TRAIN:
            log.warning(paint(f"⚠️ Experiências insuficientes: {len(self.memory)}/{NEURAL_MIN_SAMPLES_TRAIN}", C.Y))
            return
        
        log.info(paint(f"🎓 Treinando modelos com {len(self.memory)} experiências...", C.M))
        
        # Prepara dados
        X_temporal = []
        X_features = []
        y = []
        
        for exp in self.memory:
            if exp.get('temporal_features') is not None:
                X_temporal.append(exp['temporal_features'])
            X_features.append(exp['setup_features'])
            y.append(1 if exp.get('win', False) else 0)
        
        X_features = np.array(X_features)
        y = np.array(y)
        
        # Split treino/validação
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_features, y, test_size=0.2, random_state=42, stratify=y
            )
        except:
            X_train, X_val, y_train, y_val = train_test_split(
                X_features, y, test_size=0.2, random_state=42
            )
        
        # ═══════════════════════════════════════════════════════════════
        # TREINA DENSE MODEL
        # ═══════════════════════════════════════════════════════════════
        if NEURAL_AVAILABLE and len(X_train) >= 30:
            try:
                # Fit scaler
                self.scaler = StandardScaler()
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)
                
                # Build e treina
                self.dense_model = self._build_dense_model()
                
                early_stop = callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True
                )
                
                self.dense_model.fit(
                    X_train_scaled, y_train,
                    validation_data=(X_val_scaled, y_val),
                    epochs=100,
                    batch_size=min(32, len(X_train)),
                    callbacks=[early_stop],
                    verbose=0
                )
                
                # Avalia
                loss, acc = self.dense_model.evaluate(X_val_scaled, y_val, verbose=0)
                log.info(paint(f"   Dense Model: acc={acc:.2%}, loss={loss:.4f}", C.G))
                self.stats["neural_accuracy"] = float(acc)
                
            except Exception as e:
                log.warning(paint(f"⚠️ Erro ao treinar Dense: {e}", C.Y))
        
        # ═══════════════════════════════════════════════════════════════
        # TREINA LSTM MODEL (se houver dados temporais suficientes)
        # ═══════════════════════════════════════════════════════════════
        if NEURAL_AVAILABLE and len(X_temporal) >= 30:
            try:
                X_temp = np.array(X_temporal)
                y_temp = np.array([self.memory[i]['win'] for i in range(len(X_temporal))])
                
                X_train_t, X_val_t, y_train_t, y_val_t = train_test_split(
                    X_temp, y_temp, test_size=0.2, random_state=42
                )
                
                self.lstm_model = self._build_lstm_model()
                
                early_stop = callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True
                )
                
                self.lstm_model.fit(
                    X_train_t, y_train_t,
                    validation_data=(X_val_t, y_val_t),
                    epochs=50,
                    batch_size=min(16, len(X_train_t)),
                    callbacks=[early_stop],
                    verbose=0
                )
                
                loss, acc = self.lstm_model.evaluate(X_val_t, y_val_t, verbose=0)
                log.info(paint(f"   LSTM Model: acc={acc:.2%}, loss={loss:.4f}", C.G))
                
            except Exception as e:
                log.warning(paint(f"⚠️ Erro ao treinar LSTM: {e}", C.Y))
        
        # ═══════════════════════════════════════════════════════════════
        # TREINA ENSEMBLE (Random Forest + Gradient Boosting)
        # ═══════════════════════════════════════════════════════════════
        if SKLEARN_AVAILABLE and len(X_train) >= 20:
            try:
                # Random Forest
                self.ensemble_rf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
                self.ensemble_rf.fit(X_train, y_train)
                rf_acc = self.ensemble_rf.score(X_val, y_val)
                log.info(paint(f"   Random Forest: acc={rf_acc:.2%}", C.G))
                
                # Gradient Boosting
                self.ensemble_gb = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
                self.ensemble_gb.fit(X_train, y_train)
                gb_acc = self.ensemble_gb.score(X_val, y_val)
                log.info(paint(f"   Gradient Boosting: acc={gb_acc:.2%}", C.G))
                
                self.stats["ensemble_accuracy"] = float((rf_acc + gb_acc) / 2)
                
            except Exception as e:
                log.warning(paint(f"⚠️ Erro ao treinar Ensemble: {e}", C.Y))
        
        # Atualiza versão e salva
        self.stats["model_version"] += 1
        self.stats["last_retrain"] = datetime.now().isoformat()
        
        self._save_models()
        self._save_stats()
        
        log.info(paint(f"✅ Treinamento concluído! Versão: {self.stats['model_version']}", C.G))
    
    def print_thought_process(self, thought: Dict[str, Any], ativo: str, direction: str):
        """
        Imprime o processo de pensamento de forma legível.
        """
        log.info(paint("═" * 60, C.M))
        log.info(paint(f"🧠 PENSAMENTO NEURAL - {ativo} {direction}", C.M))
        log.info(paint("═" * 60, C.M))
        
        for i, stage in enumerate(thought.get('stages', []), 1):
            prob = stage.get('probability', 0.5)
            verdict = stage.get('verdict', '?')
            
            color = C.G if verdict == "POSITIVO" or verdict == "ENTRAR" else (
                    C.R if verdict == "NEGATIVO" or verdict == "NÃO ENTRAR" else C.Y)
            
            log.info(paint(f"  [{i}] {stage['name']}: {prob:.0%} -> {verdict}", color))
        
        log.info(paint("-" * 60, C.M))
        
        for reason in thought.get('reasoning', []):
            log.info(paint(f"  {reason}", C.C))
        
        log.info(paint("═" * 60, C.M))


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                           FUNÇÕES AUXILIARES                                ║
# ╚════════════════════════════════════════════════════════════════════════════╝

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
        log.info("Patch aplicado: WebsocketClient.on_close compatível.")
    except Exception as e:
        log.warning(f"Patch websocket falhou: {e}")

# ===================== IQ OPTION =====================
def conectar_iq() -> IQ_Option:
    if not EMAIL or not SENHA:
        raise RuntimeError("Defina IQ_EMAIL e IQ_PASS nas variáveis de ambiente.")
    patch_websocket_on_close()
    log.info("Conectando à IQ Option...")
    iq = IQ_Option(EMAIL, SENHA)
    iq.connect()

    for _ in range(12):
        if iq.check_connect():
            break
        time.sleep(1.5)

    if not iq.check_connect():
        raise RuntimeError("Falha na conexão com a IQ Option.")

    iq.change_balance(CONTA)
    try:
        log.info(f"Conectado | Saldo: {iq.get_balance():.2f} | Conta: {CONTA}")
    except Exception:
        log.info(f"Conectado | Conta: {CONTA}")

    return iq

def ensure_connected(iq: Optional[IQ_Option]) -> IQ_Option:
    if iq is None:
        return conectar_iq()
    try:
        if iq.check_connect():
            return iq
    except Exception:
        pass

    log.warning(paint("Conexão caiu. Tentando reconectar...", C.Y))
    try:
        iq.connect()
        for _ in range(12):
            if iq.check_connect():
                iq.change_balance(CONTA)
                log.info("Reconectado.")
                return iq
            time.sleep(1.5)
    except Exception:
        pass

    return conectar_iq()

def safe_call(iq: IQ_Option, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        if ("10054" in msg) or ("forçado o cancelamento" in msg) or ("goodbye" in msg) or ("10053" in msg):
            log.error(paint(f"Erro de conexão: {e}", C.R))
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
        if "from" in df.columns and "time" not in df.columns:
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

        need_min = max(220, SR_LOOKBACK + 20)
        if len(df) < need_min:
            return None
        return df
    except Exception:
        return None

# ===================== ATIVOS / PAYOUT =====================
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

# ===================== HELPERS =====================
def atr(df: pd.DataFrame, period: int = 14) -> float:
    sub = df.tail(period + 2)
    h = sub["high"].to_numpy(float)
    l = sub["low"].to_numpy(float)
    c = sub["close"].to_numpy(float)
    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1]))
    )
    return float(np.mean(tr[-period:]))

def candle_dir(row: pd.Series) -> int:
    o = float(row["open"]); c = float(row["close"])
    return 1 if c > o else (-1 if c < o else 0)

def candle_range(row: pd.Series) -> float:
    return float(row["high"]) - float(row["low"])

def wick_fractions(row: pd.Series) -> Dict[str, float]:
    o = float(row["open"]); c = float(row["close"])
    h = float(row["high"]); l = float(row["low"])
    rng = max(h - l, 1e-9)
    upper = h - max(o, c)
    lower = min(o, c) - l
    body  = abs(c - o)
    return {"rng": rng, "upper_frac": upper / rng, "lower_frac": lower / rng, "body_frac": body / rng}

def is_spike_wicky(row: pd.Series, atr_val: float) -> bool:
    w = wick_fractions(row)
    if candle_range(row) < SPIKE_RANGE_ATR * atr_val:
        return False
    return (w["upper_frac"] >= SPIKE_WICK_FRAC) or (w["lower_frac"] >= SPIKE_WICK_FRAC)

def leg_efficiency(df_leg: pd.DataFrame) -> float:
    if len(df_leg) < 3:
        return 0.0
    closes = df_leg["close"].to_numpy(float)
    net = abs(closes[-1] - closes[0])
    gross = np.sum(np.abs(np.diff(closes))) + 1e-9
    return float(net / gross)

def chop_stats(df_m1: pd.DataFrame, lookback: int) -> Tuple[float, float]:
    sub = df_m1.tail(lookback)
    if len(sub) < 6:
        return 1.0, 0.0
    dirs = [candle_dir(sub.iloc[i]) for i in range(len(sub))]
    dirs2 = [d for d in dirs if d != 0]
    if len(dirs2) < 6:
        return 1.0, 0.0
    flips = 0
    for i in range(1, len(dirs2)):
        if dirs2[i] != dirs2[i-1]:
            flips += 1
    flips_frac = flips / max(1, (len(dirs2) - 1))
    eff = leg_efficiency(sub)
    return float(flips_frac), float(eff)

def compression_ratio(df_m1: pd.DataFrame, atr_val: float, lookback: int) -> float:
    sub = df_m1.tail(lookback)
    if len(sub) < 6:
        return 0.0
    rng = float(sub["high"].max() - sub["low"].min())
    return float(rng / max(atr_val, 1e-9))

def late_extension_atr(df_m1: pd.DataFrame, atr_val: float, lookback: int) -> float:
    sub = df_m1.tail(lookback)
    if len(sub) < 6:
        return 0.0
    o0 = float(sub["open"].iloc[0])
    cN = float(sub["close"].iloc[-1])
    return float(abs(cN - o0) / max(atr_val, 1e-9))

# ===================== S/R FORTE =====================
def _cluster_levels(levels: List[float], tol: float) -> List[Tuple[float, int]]:
    if not levels:
        return []
    levels = sorted(levels)
    clusters = []
    cur = [levels[0]]
    for x in levels[1:]:
        if abs(x - cur[-1]) <= tol:
            cur.append(x)
        else:
            clusters.append((float(np.mean(cur)), len(cur)))
            cur = [x]
    clusters.append((float(np.mean(cur)), len(cur)))
    clusters.sort(key=lambda t: t[1], reverse=True)
    return clusters

def strong_sr_levels_last200(df_m1: pd.DataFrame, atr_val: float) -> Tuple[List[Tuple[float,int]], List[Tuple[float,int]]]:
    sub = df_m1.tail(SR_LOOKBACK)
    h = sub["high"].to_numpy(float)
    l = sub["low"].to_numpy(float)

    highs: List[float] = []
    lows: List[float] = []

    k = 2
    for i in range(k, len(sub) - k):
        if h[i] == np.max(h[i-k:i+k+1]):
            highs.append(float(h[i]))
        if l[i] == np.min(l[i-k:i+k+1]):
            lows.append(float(l[i]))

    tol_price = max(atr_val * SR_CLUSTER_ATR, 1e-9)

    res = _cluster_levels(highs, tol_price)
    sup = _cluster_levels(lows, tol_price)

    res = [(lvl, n) for (lvl, n) in res if n >= SR_MIN_TOUCHES_STRONG]
    sup = [(lvl, n) for (lvl, n) in sup if n >= SR_MIN_TOUCHES_STRONG]
    return res, sup

def pick_top_levels(levels: List[Tuple[float,int]], top_n: int) -> List[Tuple[float,int]]:
    return sorted(levels, key=lambda t: t[1], reverse=True)[:top_n]

def nearest_k(levels: List[Tuple[float,int]], price: float, k: int) -> List[Tuple[float,int,float]]:
    arr = [(lvl, touches, abs(lvl - price)) for (lvl, touches) in levels]
    arr.sort(key=lambda x: x[2])
    return arr[:k]

def sr_block_directional_multi(df_m1: pd.DataFrame, atr_val: float, direction: str) -> Optional[str]:
    res, sup = strong_sr_levels_last200(df_m1, atr_val)
    price = float(df_m1["close"].iloc[-1])
    atr_safe = max(atr_val, 1e-9)

    res = pick_top_levels(res, SR_TOP_LEVELS)
    sup = pick_top_levels(sup, SR_TOP_LEVELS)

    if direction == "CALL" and res:
        above = [(lvl,t) for (lvl,t) in res if lvl >= price]
        if not above:
            return None
        cand = nearest_k(above, price, SR_CHECK_NEAR)
        for lvl, touches, dist_abs in cand:
            dist_atr = dist_abs / atr_safe
            zone = min(SR_BLOCK_DIST_ATR + 0.10 * max(0, touches - SR_MIN_TOUCHES_STRONG), 1.30)
            if dist_atr <= zone:
                return f"bloqueado_RES(nivel={lvl:.6f},toques={touches},dist={dist_atr:.2f}ATR<=zona{zone:.2f})"
        return None

    if direction == "PUT" and sup:
        below = [(lvl,t) for (lvl,t) in sup if lvl <= price]
        if not below:
            return None
        cand = nearest_k(below, price, SR_CHECK_NEAR)
        for lvl, touches, dist_abs in cand:
            dist_atr = dist_abs / atr_safe
            zone = min(SR_BLOCK_DIST_ATR + 0.10 * max(0, touches - SR_MIN_TOUCHES_STRONG), 1.30)
            if dist_atr <= zone:
                return f"bloqueado_SUP(nivel={lvl:.6f},toques={touches},dist={dist_atr:.2f}ATR<=zona{zone:.2f})"
        return None

    return None

def sr_pingpong_zone(df_m1: pd.DataFrame, atr_val: float) -> Optional[str]:
    res, sup = strong_sr_levels_last200(df_m1, atr_val)
    if not res or not sup:
        return None
    price = float(df_m1["close"].iloc[-1])
    atr_safe = max(atr_val, 1e-9)

    res_near = sorted([(lvl,t) for (lvl,t) in res], key=lambda x: abs(x[0]-price))[:2]
    sup_near = sorted([(lvl,t) for (lvl,t) in sup], key=lambda x: abs(x[0]-price))[:2]

    above = [(lvl,t) for (lvl,t) in res_near if lvl >= price]
    below = [(lvl,t) for (lvl,t) in sup_near if lvl <= price]
    if not above or not below:
        return None

    r_lvl, r_t = min(above, key=lambda x: abs(x[0]-price))
    s_lvl, s_t = min(below, key=lambda x: abs(x[0]-price))

    corridor_atr = abs(r_lvl - s_lvl) / atr_safe
    if corridor_atr <= 0.85:
        return f"pingpong(corredor={corridor_atr:.2f}ATR sup={s_lvl:.6f} res={r_lvl:.6f})"
    return None

# ===================== LINHA DE TENDÊNCIA =====================
def detect_trendline(df: pd.DataFrame, lookback: int, direction: str) -> Optional[Tuple[float, float]]:
    if len(df) < lookback:
        return None

    sub = df.tail(lookback)

    if direction == "CALL":
        pivots = []
        lows = sub["low"].to_numpy(float)

        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                pivots.append((i, lows[i]))

        if len(pivots) < 2:
            return None

        if pivots[-1][1] <= pivots[0][1]:
            return None

        x = np.array([p[0] for p in pivots])
        y = np.array([p[1] for p in pivots])
        slope, intercept = np.polyfit(x, y, 1)

        if slope <= 0:
            return None

        return (float(slope), float(intercept))

    else:  # PUT
        pivots = []
        highs = sub["high"].to_numpy(float)

        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                pivots.append((i, highs[i]))

        if len(pivots) < 2:
            return None

        if pivots[-1][1] >= pivots[0][1]:
            return None

        x = np.array([p[0] for p in pivots])
        y = np.array([p[1] for p in pivots])
        slope, intercept = np.polyfit(x, y, 1)

        if slope >= 0:
            return None

        return (float(slope), float(intercept))

def check_trendline_confluence(df: pd.DataFrame, pb_high: float, pb_low: float,
                                direction: str, atr_val: float) -> Dict[str, Any]:
    trendline = detect_trendline(df.tail(50), 50, direction)

    if trendline is None:
        return {"has_trendline": False, "confluence": 0.0, "distance": 999.0}

    slope, intercept = trendline
    x_pb = len(df.tail(50)) - 1
    lt_value = slope * x_pb + intercept

    if direction == "CALL":
        distance = abs(pb_low - lt_value) / max(atr_val, 1e-9)

        if distance < 0.3:
            return {"has_trendline": True, "confluence": 1.0, "distance": distance, "lt_value": lt_value}
        elif distance < 0.6:
            return {"has_trendline": True, "confluence": 0.6, "distance": distance, "lt_value": lt_value}
        else:
            return {"has_trendline": True, "confluence": 0.2, "distance": distance, "lt_value": lt_value}

    else:  # PUT
        distance = abs(pb_high - lt_value) / max(atr_val, 1e-9)

        if distance < 0.3:
            return {"has_trendline": True, "confluence": 1.0, "distance": distance, "lt_value": lt_value}
        elif distance < 0.6:
            return {"has_trendline": True, "confluence": 0.6, "distance": distance, "lt_value": lt_value}
        else:
            return {"has_trendline": True, "confluence": 0.2, "distance": distance, "lt_value": lt_value}

# ===================== ANÁLISE DE CONTEXTO =====================
def analyze_market_context(df_m1: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    if len(df_m1) < 50:
        return {"quality": 0.0, "context": "insuficiente"}

    recent = df_m1.tail(20)
    closes = recent["close"].to_numpy(float)
    highs = recent["high"].to_numpy(float)
    lows = recent["low"].to_numpy(float)

    bullish = sum(1 for i in range(len(recent)) if closes[i] > recent["open"].iloc[i])
    bearish = sum(1 for i in range(len(recent)) if closes[i] < recent["open"].iloc[i])
    directional_bias = abs(bullish - bearish) / len(recent)

    ranges = [highs[i] - lows[i] for i in range(len(recent))]
    avg_range = np.mean(ranges)
    std_range = np.std(ranges)
    volatility_consistency = 1.0 - min(1.0, std_range / max(avg_range, 1e-9))

    hh_count = sum(1 for i in range(5, len(recent)) if highs[i] > max(highs[i-5:i]))
    ll_count = sum(1 for i in range(5, len(recent)) if lows[i] < min(lows[i-5:i]))
    structure_quality = max(hh_count, ll_count) / max(1, len(recent) - 5)

    last_10 = closes[-10:]
    price_momentum = abs(last_10[-1] - last_10[0]) / (max(atr_val, 1e-9) * 10)
    price_momentum = min(1.0, price_momentum)

    body_ratios = []
    for i in range(len(recent)):
        o = recent["open"].iloc[i]
        c = recent["close"].iloc[i]
        h = recent["high"].iloc[i]
        l = recent["low"].iloc[i]
        body = abs(c - o)
        total = h - l
        if total > 1e-9:
            body_ratios.append(body / total)
    avg_body_ratio = np.mean(body_ratios) if body_ratios else 0.0

    quality = (
        directional_bias * 0.25 +
        volatility_consistency * 0.20 +
        structure_quality * 0.25 +
        price_momentum * 0.15 +
        avg_body_ratio * 0.15
    )

    return {
        "quality": float(quality),
        "directional_bias": float(directional_bias),
        "volatility_consistency": float(volatility_consistency),
        "structure_quality": float(structure_quality),
        "price_momentum": float(price_momentum),
        "body_ratio": float(avg_body_ratio),
        "context": "excelente" if quality > 0.70 else ("bom" if quality > 0.55 else ("mediano" if quality > 0.40 else "ruim"))
    }

# ===================== VALIDAÇÃO DE ENTRADA =====================
def validate_entry_quality(df_m1: pd.DataFrame, atr_val: float, direction: str, entry_price: float, pb_high: float, pb_low: float) -> Dict[str, Any]:
    if len(df_m1) < 5:
        return {"valid": False, "confidence": 0.0, "reason": "dados_insuficientes"}

    last_candle = df_m1.iloc[-1]
    open_price = float(last_candle["open"])
    close_price = float(last_candle["close"])
    high_price = float(last_candle["high"])
    low_price = float(last_candle["low"])

    candle_range = high_price - low_price
    body = abs(close_price - open_price)
    body_ratio = body / max(candle_range, 1e-9)

    if body_ratio < 0.25:
        return {"valid": False, "confidence": 0.0, "reason": f"candle_fraco(body={body_ratio:.2f})"}

    if direction == "CALL":
        stop_loss = pb_low - (0.15 * atr_val)
        risk = entry_price - stop_loss
        target_1 = entry_price + (risk * 1.5)

        recent_highs = df_m1.tail(20)["high"].to_numpy(float)
        max_recent = float(np.max(recent_highs))

        if target_1 > max_recent * 1.005:
            confidence = 0.75
        else:
            confidence = 0.55

    else:  # PUT
        stop_loss = pb_high + (0.15 * atr_val)
        risk = stop_loss - entry_price
        target_1 = entry_price - (risk * 1.5)

        recent_lows = df_m1.tail(20)["low"].to_numpy(float)
        min_recent = float(np.min(recent_lows))

        if target_1 < min_recent * 0.995:
            confidence = 0.75
        else:
            confidence = 0.55

    risk_atr = risk / max(atr_val, 1e-9)

    if risk_atr < 0.2 or risk_atr > 2.0:
        return {"valid": False, "confidence": 0.0, "reason": f"risco_inadequado({risk_atr:.2f}ATR)"}

    last_3_closes = df_m1.tail(3)["close"].to_numpy(float)
    momentum = abs(last_3_closes[-1] - last_3_closes[0]) / max(atr_val, 1e-9)

    if momentum < 0.10:
        confidence *= 0.80

    last_3 = df_m1.tail(3)
    aligned = 0
    for _, row in last_3.iterrows():
        c = float(row["close"])
        o = float(row["open"])
        if (direction == "CALL" and c > o) or (direction == "PUT" and c < o):
            aligned += 1

    alignment_ratio = aligned / 3.0
    if alignment_ratio >= 0.67:
        confidence *= 1.15
    elif alignment_ratio < 0.34:
        confidence *= 0.80

    confidence = min(1.0, max(0.0, confidence))

    if confidence < 0.40:
        return {"valid": False, "confidence": confidence, "reason": f"confianca_baixa({confidence:.2f})"}

    return {
        "valid": True,
        "confidence": float(confidence),
        "risk_atr": float(risk_atr),
        "risk_reward": 1.5,
        "momentum": float(momentum),
        "body_ratio": float(body_ratio),
        "alignment": float(alignment_ratio),
        "reason": "entrada_validada"
    }

# ===================== VALIDAÇÃO DE CONTINUAÇÃO DE TENDÊNCIA =====================
def validate_trend_continuation(df_m1: pd.DataFrame, impulso_dir: str, pb_end_idx: int) -> Dict[str, Any]:
    if len(df_m1) < pb_end_idx + 30:
        return {"valid": True, "reason": "dados_insuficientes", "strength": 0.5}

    pre_impulse = df_m1.iloc[max(0, pb_end_idx - 25):pb_end_idx - 5]

    if len(pre_impulse) < 10:
        return {"valid": True, "reason": "contexto_curto", "strength": 0.5}

    closes = pre_impulse["close"].to_numpy(float)
    price_change = closes[-1] - closes[0]
    price_change_pct = abs(price_change) / max(closes[0], 1e-9)

    if impulso_dir == "PUT":
        if price_change > 0:
            if price_change_pct > 0.015:
                return {"valid": False, "reason": "contra_tendencia_forte_alta", "strength": 0.2}
            return {"valid": True, "reason": "contra_tendencia_fraca", "strength": 0.4}
        return {"valid": True, "reason": "continuacao_queda", "strength": min(1.0, price_change_pct * 50)}

    else:  # CALL
        if price_change < 0:
            if price_change_pct > 0.015:
                return {"valid": False, "reason": "contra_tendencia_forte_baixa", "strength": 0.2}
            return {"valid": True, "reason": "contra_tendencia_fraca", "strength": 0.4}
        return {"valid": True, "reason": "continuacao_alta", "strength": min(1.0, price_change_pct * 50)}

# ===================== GESTÃO DE BANCA =====================
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


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                              PERNADA B                                      ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def pernada_b(df_m1: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    """
    Detecta setup Pernada B (impulso + pullback + breakout).
    Retorna setup com todas as features para a IA analisar.
    """
    if len(df_m1) < 240:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["poucas_velas"]}

    context = analyze_market_context(df_m1, atr_val)
    market_quality = float(context.get("quality", 0.0))

    flips_frac, eff_zone = chop_stats(df_m1, CHOP_LOOKBACK)
    comp = compression_ratio(df_m1, atr_val, COMP_LOOKBACK)
    late_ext = late_extension_atr(df_m1, atr_val, LATE_LOOKBACK)

    decision = df_m1.iloc[-1]
    q = wick_fractions(decision)

    if q["body_frac"] < MIN_BODY_FRAC_BREAK:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0,
                "reasons": [f"gatilho_fraco(body={q['body_frac']:.2f})"]}

    ping = sr_pingpong_zone(df_m1, atr_val)
    if ping:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": [ping]}

    best = None

    for pb_len in range(PULLBACK_MIN, PULLBACK_MAX + 1):
        pb = df_m1.iloc[-(pb_len + 1):-1]
        if len(pb) != pb_len:
            continue

        for w in range(IMPULSO_JANELA_MIN, IMPULSO_JANELA_MAX + 1):
            imp = df_m1.iloc[-(pb_len + 1 + w):-(pb_len + 1)]
            if len(imp) != w:
                continue

            top = float(imp["high"].max())
            bot = float(imp["low"].min())
            size_A = top - bot

            if size_A < IMPULSO_MIN_ATR * atr_val:
                continue

            start = float(imp["open"].iloc[0])
            end = float(imp["close"].iloc[-1])
            move = end - start

            dir_impulso_A = "PUT" if move < 0 else ("CALL" if move > 0 else "NEUTRAL")
            if dir_impulso_A == "NEUTRAL":
                continue

            eff_A = leg_efficiency(imp)
            if eff_A < MIN_EFF_A:
                continue

            contra = 0
            for _, r in pb.iterrows():
                d = candle_dir(r)
                if dir_impulso_A == "PUT" and d == 1:
                    contra += 1
                if dir_impulso_A == "CALL" and d == -1:
                    contra += 1

            if contra < max(1, int(math.ceil(pb_len * 0.50))):
                continue

            impulse_start_idx = len(df_m1) - (pb_len + 1 + w)
            trend_validation = validate_trend_continuation(df_m1, dir_impulso_A, impulse_start_idx)
            trend_strength = float(trend_validation.get("strength", 0.0)) if trend_validation.get("valid", False) else 0.0

            if dir_impulso_A == "PUT":
                pb_high = float(pb["high"].max())
                retr = (pb_high - bot) / max(size_A, 1e-9)
            else:
                pb_low = float(pb["low"].min())
                retr = (top - pb_low) / max(size_A, 1e-9)

            if retr < RETR_MIN or retr > RETR_MAX:
                continue

            c1 = float(decision["close"])
            dir_entrada = dir_impulso_A

            blk_sr = sr_block_directional_multi(df_m1, atr_val, dir_entrada)
            if blk_sr:
                continue

            pb_high = float(pb["high"].max())
            pb_low = float(pb["low"].min())

            if dir_entrada == "CALL":
                if not (c1 > pb_low + BREAK_MARGIN_ATR * atr_val):
                    continue
                if q["upper_frac"] > MAX_WICK_AGAINST:
                    continue

                dist = (c1 - pb_low) / max(atr_val, 1e-9)
                if dist > MAX_BREAK_DISTANCE_ATR:
                    continue

                entry_validation = validate_entry_quality(df_m1, atr_val, "CALL", c1, pb_high, pb_low)
                if not entry_validation.get("valid", False):
                    continue

                entry_confidence = float(entry_validation.get("confidence", 0.0))
                risk_atr = float(entry_validation.get("risk_atr", 0.0))
                entry_momentum = float(entry_validation.get("momentum", 0.0))
                entry_alignment = float(entry_validation.get("alignment", 0.0))

                lt_conf = check_trendline_confluence(df_m1, pb_high, pb_low, "CALL", atr_val)
                lt_confluence = float(lt_conf.get("confluence", 0.0))
                has_lt = lt_conf.get("has_trendline", False)

                # Score calculation
                score = 0.35
                impulso_score = min(0.12, (size_A / atr_val - IMPULSO_MIN_ATR) * 0.06)
                score += impulso_score

                eff_score = min(0.15, max(0, (eff_A - MIN_EFF_A) * 0.35))
                score += eff_score

                if 0.30 <= retr <= 0.50:
                    retr_score = 0.10
                elif 0.25 <= retr <= 0.60:
                    retr_score = 0.05
                else:
                    retr_score = max(-0.05, -(abs(retr - 0.40) * 0.15))
                score += retr_score

                if 2 <= pb_len <= 4:
                    pb_score = 0.05
                elif pb_len == 1 or pb_len == 5:
                    pb_score = 0.02
                else:
                    pb_score = 0.0
                score += pb_score

                if flips_frac > 0.60:
                    chop_penalty = min(0.15, (flips_frac - 0.60) * 0.50)
                    score -= chop_penalty
                else:
                    chop_penalty = 0

                if market_quality > 0.60:
                    ctx_score = 0.18
                elif market_quality > 0.45:
                    ctx_score = 0.10
                elif market_quality > 0.30:
                    ctx_score = 0.03
                else:
                    ctx_score = -0.08
                score += ctx_score

                trend_score = min(0.08, trend_strength * 0.12)
                score += trend_score

                if entry_confidence > 0.65:
                    entry_score = 0.25
                elif entry_confidence > 0.55:
                    entry_score = 0.15
                elif entry_confidence > 0.48:
                    entry_score = 0.08
                else:
                    entry_score = -0.05
                score += entry_score

                momentum_score = min(0.10, entry_momentum * 0.08)
                score += momentum_score

                if entry_alignment >= 0.67:
                    align_score = 0.08
                elif entry_alignment >= 0.34:
                    align_score = 0.03
                else:
                    align_score = -0.03
                score += align_score

                if has_lt and lt_confluence > 0.8:
                    score += 0.25
                elif has_lt and lt_confluence > 0.5:
                    score += 0.15
                elif has_lt and lt_confluence > 0.2:
                    score += 0.05

                if risk_atr > 1.3:
                    risk_penalty = 0.10
                elif risk_atr > 1.0:
                    risk_penalty = 0.05
                elif risk_atr < 0.30:
                    risk_penalty = 0.05
                else:
                    risk_penalty = 0.0
                score -= risk_penalty

                perfect_count = 0
                if market_quality > 0.60: perfect_count += 1
                if eff_A > 0.70: perfect_count += 1
                if 0.30 <= retr <= 0.50: perfect_count += 1
                if entry_confidence > 0.65: perfect_count += 1
                if entry_alignment >= 0.67: perfect_count += 1
                if lt_confluence > 0.8: perfect_count += 1

                if perfect_count >= 5:
                    confluence_bonus = 0.20
                elif perfect_count >= 4:
                    confluence_bonus = 0.12
                elif perfect_count >= 3:
                    confluence_bonus = 0.05
                else:
                    confluence_bonus = 0.0

                score += confluence_bonus
                score = float(max(0.0, min(0.95, score)))

                if score < 0.48:
                    continue

                setup = {
                    "trade": True, "dir": "CALL", "score": score,
                    "pb_len": pb_len, "retr": float(retr),
                    "A_atr": float(size_A / max(atr_val, 1e-9)),
                    "effA": float(eff_A),
                    "flips": float(flips_frac),
                    "comp": float(comp),
                    "late": float(late_ext),
                    "distBreak": float(dist),
                    "market_quality": float(market_quality),
                    "context": str(context.get("context", "?")),
                    "confluence_bonus": float(confluence_bonus),
                    "trend_strength": float(trend_strength),
                    "trend_reason": str(trend_validation.get("reason", "?")),
                    "entry_confidence": float(entry_confidence),
                    "entry_momentum": float(entry_momentum),
                    "entry_alignment": float(entry_alignment),
                    "risk_atr": float(risk_atr),
                    "lt_confluence": float(lt_confluence),
                    "has_lt": has_lt,
                    # Extra features for neural
                    "impulso_score": float(impulso_score),
                    "retr_score": float(retr_score),
                    "chop_penalty": float(chop_penalty) if flips_frac > 0.60 else 0.0,
                    "entry_score": float(entry_score),
                    "risk_penalty": float(risk_penalty),
                    "reasons": [
                        "pernadaB_CALL",
                        f"A={size_A/atr_val:.2f}ATR", f"retr={retr:.2f}", f"pb={pb_len}",
                        f"effA={eff_A:.2f}",
                        f"ctx={context.get('context','?')}({market_quality:.2f})",
                        f"entry_conf={entry_confidence:.2f}",
                        f"⭐LTA={lt_confluence:.2f}" if has_lt else "sem_LTA"
                    ]
                }

            else:  # PUT
                if not (c1 < pb_high - BREAK_MARGIN_ATR * atr_val):
                    continue
                if q["lower_frac"] > MAX_WICK_AGAINST:
                    continue

                dist = (pb_high - c1) / max(atr_val, 1e-9)
                if dist > MAX_BREAK_DISTANCE_ATR:
                    continue

                entry_validation = validate_entry_quality(df_m1, atr_val, "PUT", c1, pb_high, pb_low)
                if not entry_validation.get("valid", False):
                    continue

                entry_confidence = float(entry_validation.get("confidence", 0.0))
                risk_atr = float(entry_validation.get("risk_atr", 0.0))
                entry_momentum = float(entry_validation.get("momentum", 0.0))
                entry_alignment = float(entry_validation.get("alignment", 0.0))

                lt_conf = check_trendline_confluence(df_m1, pb_high, pb_low, "PUT", atr_val)
                lt_confluence = float(lt_conf.get("confluence", 0.0))
                has_lt = lt_conf.get("has_trendline", False)

                # Score calculation (same as CALL)
                score = 0.35
                impulso_score = min(0.12, (size_A / atr_val - IMPULSO_MIN_ATR) * 0.06)
                score += impulso_score

                eff_score = min(0.15, max(0, (eff_A - MIN_EFF_A) * 0.35))
                score += eff_score

                if 0.30 <= retr <= 0.50:
                    retr_score = 0.10
                elif 0.25 <= retr <= 0.60:
                    retr_score = 0.05
                else:
                    retr_score = max(-0.05, -(abs(retr - 0.40) * 0.15))
                score += retr_score

                if 2 <= pb_len <= 4:
                    pb_score = 0.05
                elif pb_len == 1 or pb_len == 5:
                    pb_score = 0.02
                else:
                    pb_score = 0.0
                score += pb_score

                if flips_frac > 0.60:
                    chop_penalty = min(0.15, (flips_frac - 0.60) * 0.50)
                    score -= chop_penalty
                else:
                    chop_penalty = 0

                if market_quality > 0.60:
                    ctx_score = 0.18
                elif market_quality > 0.45:
                    ctx_score = 0.10
                elif market_quality > 0.30:
                    ctx_score = 0.03
                else:
                    ctx_score = -0.08
                score += ctx_score

                trend_score = min(0.08, trend_strength * 0.12)
                score += trend_score

                if entry_confidence > 0.65:
                    entry_score = 0.25
                elif entry_confidence > 0.55:
                    entry_score = 0.15
                elif entry_confidence > 0.48:
                    entry_score = 0.08
                else:
                    entry_score = -0.05
                score += entry_score

                momentum_score = min(0.10, entry_momentum * 0.08)
                score += momentum_score

                if entry_alignment >= 0.67:
                    align_score = 0.08
                elif entry_alignment >= 0.34:
                    align_score = 0.03
                else:
                    align_score = -0.03
                score += align_score

                if has_lt and lt_confluence > 0.8:
                    score += 0.25
                elif has_lt and lt_confluence > 0.5:
                    score += 0.15
                elif has_lt and lt_confluence > 0.2:
                    score += 0.05

                if risk_atr > 1.3:
                    risk_penalty = 0.10
                elif risk_atr > 1.0:
                    risk_penalty = 0.05
                elif risk_atr < 0.30:
                    risk_penalty = 0.05
                else:
                    risk_penalty = 0.0
                score -= risk_penalty

                perfect_count = 0
                if market_quality > 0.60: perfect_count += 1
                if eff_A > 0.70: perfect_count += 1
                if 0.30 <= retr <= 0.50: perfect_count += 1
                if entry_confidence > 0.65: perfect_count += 1
                if entry_alignment >= 0.67: perfect_count += 1
                if lt_confluence > 0.8: perfect_count += 1

                if perfect_count >= 5:
                    confluence_bonus = 0.20
                elif perfect_count >= 4:
                    confluence_bonus = 0.12
                elif perfect_count >= 3:
                    confluence_bonus = 0.05
                else:
                    confluence_bonus = 0.0

                score += confluence_bonus
                score = float(max(0.0, min(0.95, score)))

                if score < 0.48:
                    continue

                setup = {
                    "trade": True, "dir": "PUT", "score": score,
                    "pb_len": pb_len, "retr": float(retr),
                    "A_atr": float(size_A / max(atr_val, 1e-9)),
                    "effA": float(eff_A),
                    "flips": float(flips_frac),
                    "comp": float(comp),
                    "late": float(late_ext),
                    "distBreak": float(dist),
                    "market_quality": float(market_quality),
                    "context": str(context.get("context", "?")),
                    "confluence_bonus": float(confluence_bonus),
                    "trend_strength": float(trend_strength),
                    "trend_reason": str(trend_validation.get("reason", "?")),
                    "entry_confidence": float(entry_confidence),
                    "entry_momentum": float(entry_momentum),
                    "entry_alignment": float(entry_alignment),
                    "risk_atr": float(risk_atr),
                    "lt_confluence": float(lt_confluence),
                    "has_lt": has_lt,
                    "impulso_score": float(impulso_score),
                    "retr_score": float(retr_score),
                    "chop_penalty": float(chop_penalty) if flips_frac > 0.60 else 0.0,
                    "entry_score": float(entry_score),
                    "risk_penalty": float(risk_penalty),
                    "reasons": [
                        "pernadaB_PUT",
                        f"A={size_A/atr_val:.2f}ATR", f"retr={retr:.2f}", f"pb={pb_len}",
                        f"effA={eff_A:.2f}",
                        f"ctx={context.get('context','?')}({market_quality:.2f})",
                        f"entry_conf={entry_confidence:.2f}",
                        f"⭐LTB={lt_confluence:.2f}" if has_lt else "sem_LTB"
                    ]
                }

            if best is None or setup["score"] > best["score"]:
                best = setup

    if best is None:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": ["sem_pernadaB_valida"]}

    block_final = sr_block_directional_multi(df_m1, atr_val, best["dir"])
    if block_final:
        return {"trade": False, "dir": "NEUTRAL", "score": 0.0, "reasons": [block_final]}

    return best


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                              ESCOLHER SETUP                                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def escolher_melhor_setup(iq: IQ_Option, ativos: List[str], brain: NeuralBrain):
    """
    Escolhe o melhor setup usando a IA Neural Brain.
    """
    best_trade = None
    best_any = None

    for a in ativos:
        if a in cooldown and (time.time() - cooldown[a]) < COOLDOWN_ATIVO:
            continue
        if a in cooldown_spike and (time.time() - cooldown_spike[a]) < (SPIKE_COOLDOWN_MIN * 60):
            continue

        df = get_candles_df(iq, a, TF_M5, N_CANDLES, end_ts=end_ts_closed(TF_M5))
        if df is None:
            continue

        atr_val = atr(df, 14)
        last_closed = df.iloc[-1]

        if is_spike_wicky(last_closed, atr_val):
            cooldown_spike[a] = time.time()
            continue

        setup = pernada_b(df, atr_val)

        sc_any = float(setup.get("score", 0.0))
        cand_any = (sc_any, a, setup, float(atr_val), df)
        if best_any is None or cand_any[0] > best_any[0]:
            best_any = cand_any

        if setup.get("trade"):
            # Usa Neural Brain para decidir
            thought = brain.think(setup, df, atr_val)
            
            if thought["should_trade"]:
                cand_trade = (
                    float(setup["score"]), 
                    a, 
                    setup, 
                    float(atr_val), 
                    df, 
                    thought
                )
                if best_trade is None or cand_trade[0] > best_trade[0]:
                    best_trade = cand_trade

    return best_trade, best_any


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                                  ORDEM                                      ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def enviar_ordem(iq: IQ_Option, ativo: str, direcao: str, stake: float) -> Optional[Tuple[str, int]]:
    d = "call" if direcao == "CALL" else "put"
    valor = float(max(VALOR_MINIMO, stake))

    # TURBO
    try:
        ok, op_id = safe_call(iq, iq.buy, valor, ativo, d, int(EXP_FIXA))
        if ok and op_id:
            return ("turbo", int(op_id))
        log.warning(paint(f"[ORDEM-FAIL] TURBO ok={ok} op_id={op_id}", C.Y))
    except Exception as e:
        log.warning(paint(f"[ORDEM-EXC] TURBO {e}", C.Y))

    # DIGITAL
    try:
        ok, op_id = safe_call(iq, iq.buy_digital_spot, ativo, valor, d, int(EXP_FIXA))
        if ok and op_id:
            return ("digital", int(op_id))
        log.warning(paint(f"[ORDEM-FAIL] DIGITAL ok={ok} op_id={op_id}", C.Y))
    except Exception as e:
        log.warning(paint(f"[ORDEM-EXC] DIGITAL {e}", C.Y))

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
        except Exception:
            ensure_connected(iq)
        time.sleep(0.25)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                                   MAIN                                      ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def main():
    """
    Loop principal com Neural Brain.
    """
    iq: Optional[IQ_Option] = None
    iq = ensure_connected(iq)

    # Inicializa Neural Brain
    brain = NeuralBrain()
    
    log.info(paint("═" * 60, C.M))
    log.info(paint("🧠 WS NEURAL BRAIN — IA DE VERDADE", C.M))
    log.info(paint("═" * 60, C.M))
    log.info(paint("├── LSTM: Análise temporal de padrões", C.C))
    log.info(paint("├── Dense: Análise de 52 features", C.C))
    log.info(paint("├── Ensemble: RF + GB para consenso", C.C))
    log.info(paint("├── Pensamento: 5 etapas de análise", C.C))
    log.info(paint(f"├── Confiança mínima: {NEURAL_MIN_CONFIDENCE:.0%}", C.C))
    log.info(paint(f"└── Consenso mínimo: {NEURAL_CONSENSUS_MIN:.0%}", C.C))
    log.info(paint("═" * 60, C.M))

    # ═══════════════════════════════════════════════════════════
    # 🎓 PRÉ-TREINAMENTO COM HISTÓRICO
    # ═══════════════════════════════════════════════════════════
    if brain.needs_pretrain:
        log.info(paint("🎓 IA precisa de pré-treino! Coletando dados históricos...", C.M))
        ativos_treino = obter_top_ativos_otc(iq)
        if ativos_treino:
            brain.pretrain_with_history(iq, ativos_treino, n_candles=900)
        else:
            log.warning(paint("⚠️ Sem ativos disponíveis para pré-treino", C.Y))
    else:
        log.info(paint("✅ IA já treinada! Pronta para operar.", C.G))
    # ═══════════════════════════════════════════════════════════

    # Gestão de banca
    try:
        saldo_inicial = float(iq.get_balance())
        log.info(paint(f"💰 SALDO INICIAL: {saldo_inicial:.2f} | META: {META_LUCRO_PERCENT:.1f}%", C.G))
    except Exception:
        saldo_inicial = 1000.0

    total = 0
    wins = 0

    while True:
        iq = ensure_connected(iq)

        # Verificar meta/stop
        try:
            saldo_atual = float(iq.get_balance())
            deve_parar, lucro_percent = verificar_meta_atingida(saldo_inicial, saldo_atual)
            if deve_parar:
                lucro_abs = saldo_atual - saldo_inicial
                if lucro_percent >= META_LUCRO_PERCENT:
                    log.info(paint(f"🎯 META ATINGIDA! Lucro: {lucro_abs:.2f} ({lucro_percent:.2f}%)", C.G))
                else:
                    log.info(paint(f"🛑 STOP LOSS! Perda: {lucro_abs:.2f} ({lucro_percent:.2f}%)", C.R))
                break
        except Exception as e:
            log.warning(f"Erro ao verificar meta: {e}")

        ativos = obter_top_ativos_otc(iq)
        if not ativos:
            log.warning("Sem ativos com payout mínimo. Tentando em 10s...")
            time.sleep(10)
            continue

        wait_until_minus(TF_M5, DECIDIR_ANTES_FECHAR_SEC)

        best_trade, best_any = escolher_melhor_setup(iq, ativos, brain)

        if not best_trade:
            if best_any:
                sc, at, st, _av, _df = best_any
                log.info(paint(
                    f"[NEURAL-SKIP] {at} | score={sc:.2f} | Brain não aprovou",
                    C.Y
                ))
                cooldown[at] = time.time()
            else:
                log.info(paint("[SKIP] nenhum ativo analisável", C.Y))

            wait_for_next_open(TF_M5)
            continue

        score, ativo, setup, atr_val, df, thought = best_trade
        score = float(score)
        final_dir = str(setup["dir"])

        # Mostra pensamento da IA
        brain.print_thought_process(thought, ativo, final_dir)

        log.info(paint(
            f"[🧠 NEURAL] {ativo} -> {final_dir} | score={score:.2f} | conf={thought['confidence']:.0%}",
            dir_color(final_dir)
        ))

        wait_for_next_open(TF_M5)

        stake = calcular_stake_dinamico(iq, STAKE_FIXA)
        log.info(paint(f"[{ativo}] 💵 Stake: {stake:.2f}", C.B))

        op = enviar_ordem(iq, ativo, final_dir, stake)

        if not op:
            log.error(paint(f"[{ativo}] ❌ Ordem falhou", C.R))
            cooldown[ativo] = time.time()
            continue

        op_type, op_id = op
        log.info(paint(
            f"[{ativo}] ✅ ORDEM ENVIADA {final_dir} exp={EXP_FIXA}m ({op_type}) | stake={stake:.2f}",
            dir_color(final_dir)
        ))

        res = wait_result(iq, op_type, op_id)

        total += 1
        if res > 0:
            wins += 1
            log.info(paint(f"[{ativo}] ✅ WIN {res:.2f}$", C.G))
        elif res < 0:
            log.info(paint(f"[{ativo}] ❌ LOSS {res:.2f}$", C.R))
        else:
            log.info(paint(f"[{ativo}] ⚪ EMPATE {res:.2f}$", C.B))

        # Adiciona experiência ao Neural Brain
        brain.add_experience(setup, df, atr_val, res, thought)

        acc = (wins / max(1, total)) * 100.0

        try:
            saldo_atual = float(iq.get_balance())
            lucro_atual = saldo_atual - saldo_inicial
            lucro_percent_atual = (lucro_atual / saldo_inicial) * 100.0

            if lucro_percent_atual >= 0:
                log.info(paint(f"📊 TRADES: {total} | WINS: {wins} | ACC: {acc:.2f}%", C.G))
                log.info(paint(f"💰 SALDO: {saldo_atual:.2f} | LUCRO: +{lucro_atual:.2f} ({lucro_percent_atual:.2f}%)\n", C.G))
            else:
                log.info(paint(f"📊 TRADES: {total} | WINS: {wins} | ACC: {acc:.2f}%", C.Y))
                log.info(paint(f"💰 SALDO: {saldo_atual:.2f} | PERDA: {lucro_atual:.2f} ({lucro_percent_atual:.2f}%)\n", C.Y))
        except Exception:
            log.info(f"📊 TRADES: {total} | WINS: {wins} | ACC: {acc:.2f}%\n")

        cooldown[ativo] = time.time()


if __name__ == "__main__":
    main()
