"""
Neural Model - CNN 1D com 3 Classes

Arquitetura: CNN 1D para deteccao de padroes locais em series temporais.
Lookback: 100 velas (1h40min de contexto)
Classes: CALL, PUT, NO_TRADE

Treinamento: Online (aprende operando)
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from datetime import datetime

# ========== SUPRIMIR MENSAGENS DO TENSORFLOW ==========
# Deve ser feito ANTES de importar o TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desativa mensagens oneDNN

# Tenta importar TensorFlow
try:
    import tensorflow as tf
    # Suprimir logs do TensorFlow
    tf.get_logger().setLevel('ERROR')
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    # Nao mostrar mensagem para usuario

# ============================================================================
#                         CONFIGURACOES
# ============================================================================

# Arquitetura
LOOKBACK = 100          # 100 velas de historico
N_FEATURES = 7          # 7 features por vela
N_CLASSES = 3           # CALL, PUT, NO_TRADE

# Classes
CLASS_CALL = 0
CLASS_PUT = 1
CLASS_NO_TRADE = 2

CLASS_NAMES = {
    CLASS_CALL: "CALL",
    CLASS_PUT: "PUT",
    CLASS_NO_TRADE: "NO_TRADE"
}

# Treinamento
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2
MIN_SAMPLES_TO_TRAIN = 50   # Minimo de amostras para treinar
RETRAIN_EVERY = 20          # Retreina a cada N novos trades

# Pesos de classe (NO_TRADE sera maioria)
CLASS_WEIGHTS = {
    CLASS_CALL: 1.0,
    CLASS_PUT: 1.0,
    CLASS_NO_TRADE: 0.5     # Menos peso para NO_TRADE
}

# Persistencia
MODEL_FILE = "cnn_trading_model.keras"
TRAINING_DATA_FILE = "cnn_training_data.json"

# ============================================================================
#                         CLASSE TRADING CNN
# ============================================================================

class TradingCNN:
    """
    Modelo CNN 1D para previsao de direcao de trading.

    Entrada: 100 velas x 7 features
    Saida: softmax [P_CALL, P_PUT, P_NO_TRADE]
    """

    def __init__(
        self,
        lookback: int = LOOKBACK,
        n_features: int = N_FEATURES,
        model_file: str = MODEL_FILE,
        data_file: str = TRAINING_DATA_FILE
    ):
        self.lookback = lookback
        self.n_features = n_features
        self.model_file = model_file
        self.data_file = data_file

        self.model = None
        self.training_data: List[Dict] = []
        self.trades_since_train = 0

        # Carrega modelo e dados existentes
        self._load_model()
        self._load_training_data()

    def _build_model(self) -> Optional[Model]:
        """Constroi a arquitetura CNN 1D."""
        if not TF_AVAILABLE:
            return None

        inputs = layers.Input(shape=(self.lookback, self.n_features))

        # Primeira camada Conv1D
        x = layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.2)(x)

        # Segunda camada Conv1D
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.2)(x)

        # Terceira camada Conv1D
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.3)(x)

        # Flatten e Dense
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        # Saida: 3 classes com softmax
        outputs = layers.Dense(N_CLASSES, activation='softmax')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _load_model(self):
        """Carrega modelo salvo ou cria novo."""
        if not TF_AVAILABLE:
            return

        if os.path.exists(self.model_file):
            try:
                self.model = tf.keras.models.load_model(self.model_file)
                print(f"[CNN] Modelo carregado de {self.model_file}")
            except Exception as e:
                print(f"[CNN] Erro ao carregar modelo: {e}")
                self.model = self._build_model()
        else:
            self.model = self._build_model()
            print("[CNN] Novo modelo criado")

    def _save_model(self):
        """Salva modelo no disco."""
        if self.model is None:
            return

        try:
            self.model.save(self.model_file)
            print(f"[CNN] Modelo salvo em {self.model_file}")
        except Exception as e:
            print(f"[CNN] Erro ao salvar modelo: {e}")

    def _load_training_data(self):
        """Carrega dados de treinamento salvos."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    self.training_data = json.load(f)
                print(f"[CNN] {len(self.training_data)} amostras de treinamento carregadas")
            except Exception as e:
                print(f"[CNN] Erro ao carregar dados: {e}")
                self.training_data = []
        else:
            self.training_data = []

    def _save_training_data(self):
        """Salva dados de treinamento no disco."""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.training_data, f)
        except Exception as e:
            print(f"[CNN] Erro ao salvar dados de treinamento: {e}")

    def extract_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Extrai features das ultimas N velas.

        Features (7 por vela):
        1. body_size - Tamanho do corpo normalizado
        2. upper_wick - Pavio superior normalizado
        3. lower_wick - Pavio inferior normalizado
        4. direction - 1 = alta, 0 = baixa
        5. momentum - Close - Open normalizado
        6. range - High - Low normalizado
        7. position - Posicao do close no range (0-1)
        """
        if len(df) < self.lookback:
            return None

        recent = df.tail(self.lookback).copy()

        # Calcula ATR para normalizacao
        atr = self._calculate_atr(df)
        if atr <= 0:
            atr = 0.0001  # Fallback

        features = []

        for i in range(len(recent)):
            row = recent.iloc[i]

            open_price = row['open']
            high = row['high']
            low = row['low']
            close = row['close']

            candle_range = high - low
            if candle_range <= 0:
                candle_range = atr

            body = abs(close - open_price)
            body_top = max(close, open_price)
            body_bottom = min(close, open_price)
            upper_wick = high - body_top
            lower_wick = body_bottom - low

            # Normaliza pelo ATR
            body_size = body / atr
            upper_wick_norm = upper_wick / atr
            lower_wick_norm = lower_wick / atr
            momentum = (close - open_price) / atr
            range_norm = candle_range / atr

            # Direcao e posicao
            direction = 1.0 if close > open_price else 0.0
            position = (close - low) / candle_range if candle_range > 0 else 0.5

            features.append([
                body_size,
                upper_wick_norm,
                lower_wick_norm,
                direction,
                momentum,
                range_norm,
                position
            ])

        return np.array(features, dtype=np.float32)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calcula ATR (Average True Range)."""
        if len(df) < period:
            return 0.0001

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr = np.maximum(
            high[-period:] - low[-period:],
            np.maximum(
                np.abs(high[-period:] - np.roll(close, 1)[-period:]),
                np.abs(low[-period:] - np.roll(close, 1)[-period:])
            )
        )
        return float(np.mean(tr[1:]))

    def predict(self, df_m1: pd.DataFrame, m5_direction: str = "NEUTRAL", m5_strength: float = 0.0) -> Dict:
        """
        Faz predicao sincronizada usando M1 + contexto M5.

        Args:
            df_m1: DataFrame com candles M1
            m5_direction: Direcao do M5 ("BULLISH", "BEARISH", "NEUTRAL")
            m5_strength: Forca da tendencia M5 (0.0-1.0)

        Returns:
            Dict com:
            - class: "CALL", "PUT", "NO_TRADE"
            - probability: 0.0-1.0 (probabilidade da classe escolhida)
            - raw_probs: [p_call, p_put, p_notrade]
            - confidence: diferenca entre top 2 probabilidades
            - m5_aligned: se a predicao esta alinhada com M5
        """
        # Resultado padrao (NO_TRADE com baixa confianca)
        default_result = {
            "class": "NO_TRADE",
            "probability": 0.33,
            "raw_probs": [0.33, 0.33, 0.34],
            "confidence": 0.0,
            "m5_aligned": False,
            "m5_direction": m5_direction
        }

        # REGRA 1: Se M5 é NEUTRAL, nao tem direcao clara
        if m5_direction == "NEUTRAL":
            default_result["reason"] = "M5_NEUTRAL"
            return default_result

        if self.model is None:
            # Sem modelo treinado, usa M5 como guia
            if m5_direction == "BULLISH":
                return {
                    "class": "CALL",
                    "probability": 0.5 + (m5_strength * 0.2),
                    "raw_probs": [0.5, 0.3, 0.2],
                    "confidence": 0.2,
                    "m5_aligned": True,
                    "m5_direction": m5_direction,
                    "reason": "M5_GUIA"
                }
            elif m5_direction == "BEARISH":
                return {
                    "class": "PUT",
                    "probability": 0.5 + (m5_strength * 0.2),
                    "raw_probs": [0.3, 0.5, 0.2],
                    "confidence": 0.2,
                    "m5_aligned": True,
                    "m5_direction": m5_direction,
                    "reason": "M5_GUIA"
                }
            return default_result

        features = self.extract_features(df_m1)
        if features is None:
            return default_result

        # Adiciona dimensao de batch
        X = np.expand_dims(features, axis=0)

        try:
            # Predicao da CNN baseada em M1
            probs = self.model.predict(X, verbose=0)[0]

            # Ajusta probabilidades baseado no M5 (sincronizacao)
            # Se M5 é BULLISH, aumenta prob de CALL e diminui PUT
            # Se M5 é BEARISH, aumenta prob de PUT e diminui CALL
            m5_boost = m5_strength * 0.15  # Bonus de ate 15%

            if m5_direction == "BULLISH":
                probs[CLASS_CALL] += m5_boost
                probs[CLASS_PUT] -= m5_boost * 0.5
            elif m5_direction == "BEARISH":
                probs[CLASS_PUT] += m5_boost
                probs[CLASS_CALL] -= m5_boost * 0.5

            # Normaliza para somar 1
            probs = np.clip(probs, 0, 1)
            probs = probs / probs.sum()

            # Classe com maior probabilidade
            predicted_class = int(np.argmax(probs))
            probability = float(probs[predicted_class])

            # Confianca = diferenca entre top 2
            sorted_probs = np.sort(probs)[::-1]
            confidence = sorted_probs[0] - sorted_probs[1]

            # Verifica alinhamento com M5
            cnn_class = CLASS_NAMES[predicted_class]
            m5_aligned = (
                (cnn_class == "CALL" and m5_direction == "BULLISH") or
                (cnn_class == "PUT" and m5_direction == "BEARISH") or
                (cnn_class == "NO_TRADE")
            )

            # REGRA 2: Se CNN discorda do M5, retorna NO_TRADE
            if not m5_aligned and cnn_class != "NO_TRADE":
                return {
                    "class": "NO_TRADE",
                    "probability": 0.4,
                    "raw_probs": probs.tolist(),
                    "confidence": 0.0,
                    "m5_aligned": False,
                    "m5_direction": m5_direction,
                    "cnn_original": cnn_class,
                    "reason": f"CONFLITO_M1_M5(CNN={cnn_class},M5={m5_direction})"
                }

            return {
                "class": cnn_class,
                "probability": probability,
                "raw_probs": probs.tolist(),
                "confidence": float(confidence),
                "m5_aligned": m5_aligned,
                "m5_direction": m5_direction,
                "reason": "SINCRONIZADO"
            }
        except Exception as e:
            print(f"[CNN] Erro na predicao: {e}")
            return default_result

    def add_training_sample(
        self,
        df: pd.DataFrame,
        direction: str,
        win: bool
    ):
        """
        Adiciona amostra de treinamento.

        Args:
            df: DataFrame com candles no momento da entrada
            direction: "CALL" ou "PUT"
            win: True se ganhou, False se perdeu
        """
        features = self.extract_features(df)
        if features is None:
            return

        # Determina label
        if win:
            # Se ganhou, a direcao estava correta
            label = CLASS_CALL if direction == "CALL" else CLASS_PUT
        else:
            # Se perdeu, adiciona como NO_TRADE (nao deveria ter entrado)
            label = CLASS_NO_TRADE

        # Salva amostra
        sample = {
            "features": features.tolist(),
            "label": label,
            "direction": direction,
            "win": win,
            "timestamp": datetime.now().isoformat()
        }
        self.training_data.append(sample)
        self.trades_since_train += 1

        # Salva dados
        self._save_training_data()

        # Verifica se deve retreinar
        if self.trades_since_train >= RETRAIN_EVERY:
            self.retrain()

    def retrain(self):
        """Retreina o modelo com todos os dados acumulados."""
        if not TF_AVAILABLE or self.model is None:
            return

        if len(self.training_data) < MIN_SAMPLES_TO_TRAIN:
            print(f"[CNN] Poucos dados para treinar ({len(self.training_data)}/{MIN_SAMPLES_TO_TRAIN})")
            return

        print(f"[CNN] Retreinando com {len(self.training_data)} amostras...")

        # Prepara dados
        X = np.array([s['features'] for s in self.training_data], dtype=np.float32)
        y_labels = np.array([s['label'] for s in self.training_data], dtype=np.int32)

        # One-hot encoding
        y = np.zeros((len(y_labels), N_CLASSES), dtype=np.float32)
        for i, label in enumerate(y_labels):
            y[i, label] = 1.0

        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )

        # Treina
        try:
            self.model.fit(
                X, y,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=VALIDATION_SPLIT,
                class_weight=CLASS_WEIGHTS,
                callbacks=[early_stop],
                verbose=0
            )
            print("[CNN] Treinamento concluido")

            # Salva modelo
            self._save_model()
            self.trades_since_train = 0

        except Exception as e:
            print(f"[CNN] Erro no treinamento: {e}")

    def get_stats(self) -> Dict:
        """Retorna estatisticas do modelo."""
        n_samples = len(self.training_data)
        n_wins = sum(1 for s in self.training_data if s.get('win', False))
        n_losses = n_samples - n_wins

        class_counts = {
            "CALL": sum(1 for s in self.training_data if s.get('label') == CLASS_CALL),
            "PUT": sum(1 for s in self.training_data if s.get('label') == CLASS_PUT),
            "NO_TRADE": sum(1 for s in self.training_data if s.get('label') == CLASS_NO_TRADE)
        }

        return {
            "total_samples": n_samples,
            "wins": n_wins,
            "losses": n_losses,
            "win_rate": n_wins / n_samples if n_samples > 0 else 0,
            "class_counts": class_counts,
            "trades_since_train": self.trades_since_train,
            "model_loaded": self.model is not None
        }


# ============================================================================
#                         FUNCOES AUXILIARES
# ============================================================================

def create_trading_cnn(**kwargs) -> TradingCNN:
    """Factory function para criar TradingCNN."""
    return TradingCNN(**kwargs)


# ============================================================================
#                         TESTE LOCAL
# ============================================================================

if __name__ == "__main__":
    # Teste basico
    np.random.seed(42)

    # Simula candles
    n = 150
    close = 1.0 + np.cumsum(np.random.randn(n) * 0.001)
    open_prices = close - np.random.randn(n) * 0.0005
    high = np.maximum(close, open_prices) + np.abs(np.random.randn(n) * 0.0003)
    low = np.minimum(close, open_prices) - np.abs(np.random.randn(n) * 0.0003)

    df_test = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close
    })

    # Testa o modelo
    cnn = TradingCNN()

    print("\n=== Predicao ===")
    result = cnn.predict(df_test)
    print(f"Classe: {result['class']}")
    print(f"Probabilidade: {result['probability']:.3f}")
    print(f"Raw probs: {[f'{p:.3f}' for p in result['raw_probs']]}")
    print(f"Confianca: {result['confidence']:.3f}")

    print("\n=== Estatisticas ===")
    stats = cnn.get_stats()
    for k, v in stats.items():
        print(f"{k}: {v}")
