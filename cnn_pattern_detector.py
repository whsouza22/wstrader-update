# -*- coding: utf-8 -*-
"""
CNN Pattern Detector — Detector de Padrões de Velas LEVE (sem TensorFlow)

Implementa detecção de padrões usando convolução 1D com numpy puro.
Mesmo conceito do neural_model.py (7 features × N candles), mas sem TensorFlow.

Arquitetura:
  Input(50, 7) → Conv1D(8 filtros, kernel=5) → ReLU → AvgPool(5) → Dense(3) → Softmax
  Total: ~500 parâmetros (vs ~100k do TF original)

Features por vela (mesmas do neural_model.py):
  1. body_size      — Tamanho do corpo normalizado por ATR
  2. upper_wick     — Pavio superior normalizado
  3. lower_wick     — Pavio inferior normalizado
  4. direction      — 1.0 = alta, 0.0 = baixa
  5. momentum       — (close - open) / ATR
  6. range          — (high - low) / ATR
  7. position       — Posição do close no range (0-1)

Classes:
  0 = CALL, 1 = PUT, 2 = NO_TRADE

Online learning com SGD simples.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any
from datetime import datetime

# ===================== CONFIGURAÇÃO =====================
LOOKBACK = 50           # Candles de lookback (50 vs 100 do TF - mais leve)
N_FEATURES = 7          # Features por candle
N_FILTERS = 8           # Filtros convolucionais
KERNEL_SIZE = 5         # Tamanho do kernel de convolução
POOL_SIZE = 5           # Tamanho do pooling
N_CLASSES = 3           # CALL, PUT, NO_TRADE

CLASS_CALL = 0
CLASS_PUT = 1
CLASS_NO_TRADE = 2
CLASS_NAMES = ["CALL", "PUT", "NO_TRADE"]

# Training
MIN_SAMPLES = 30        # Mínimo de amostras para prever com confiança
RETRAIN_EVERY = 15      # Retreinar a cada N trades
LEARNING_RATE = 0.01    # Taxa de aprendizado
BATCH_SIZE = 16         # Mini-batch size
MAX_EPOCHS = 30         # Épocas de treino
DECAY_FACTOR = 0.995    # Decaimento das amostras antigas

# Persistência
DATA_FILE = "cnn_training_data.json"
WEIGHTS_FILE = "cnn_weights.json"


# ===================== FUNÇÕES AUXILIARES =====================

def _relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)

def _relu_deriv(x: np.ndarray) -> np.ndarray:
    """Derivada do ReLU."""
    return (x > 0).astype(np.float32)

def _softmax(x: np.ndarray) -> np.ndarray:
    """Softmax estável numericamente."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-8)

def _cross_entropy_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    """Cross-entropy loss."""
    eps = 1e-7
    return -np.mean(np.sum(labels * np.log(probs + eps), axis=-1))

def _conv1d_forward(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Convolução 1D manual com numpy.
    
    x:       (seq_len, n_features)
    weights: (kernel_size, n_features, n_filters)
    bias:    (n_filters,)
    
    Returns: (seq_len - kernel_size + 1, n_filters)
    """
    seq_len = x.shape[0]
    k_size = weights.shape[0]
    n_filters = weights.shape[2]
    out_len = seq_len - k_size + 1
    
    output = np.zeros((out_len, n_filters), dtype=np.float32)
    for i in range(out_len):
        window = x[i:i + k_size]  # (kernel_size, n_features)
        # Dot product com cada filtro
        for f in range(n_filters):
            output[i, f] = np.sum(window * weights[:, :, f]) + bias[f]
    
    return output

def _avg_pool1d(x: np.ndarray, pool_size: int) -> np.ndarray:
    """Average pooling 1D."""
    seq_len = x.shape[0]
    n_feat = x.shape[1]
    out_len = seq_len // pool_size
    
    if out_len == 0:
        return np.mean(x, axis=0, keepdims=True)
    
    output = np.zeros((out_len, n_feat), dtype=np.float32)
    for i in range(out_len):
        start = i * pool_size
        end = start + pool_size
        output[i] = np.mean(x[start:end], axis=0)
    
    return output


# ===================== CLASSE PRINCIPAL =====================

class LightCNN:
    """
    CNN 1D leve usando numpy puro — substitui TensorFlow para produção.
    
    Arquitetura:
      Conv1D(7→8, kernel=5) → ReLU → AvgPool(5) → Flatten → Dense(3) → Softmax
    """
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.lookback = LOOKBACK
        self.n_features = N_FEATURES
        self.n_filters = N_FILTERS
        self.kernel_size = KERNEL_SIZE
        self.pool_size = POOL_SIZE
        
        # Inicializa pesos (Xavier initialization)
        self._init_weights()
        
        # Dados de treinamento
        self.training_data: List[Dict] = []
        self.trades_since_train = 0
        self.total_trained = 0
        
        # Estatísticas
        self.stats = {
            "predictions": 0,
            "correct": 0,
            "total_trained": 0,
            "last_train_time": None,
            "last_loss": None
        }
        
        # Carrega dados persistidos
        self._load_weights()
        self._load_training_data()
    
    def _init_weights(self):
        """Inicializa pesos com Xavier initialization."""
        np.random.seed(int(time.time()) % 10000)
        
        # Conv1D: (kernel_size, n_features, n_filters)
        fan_in = self.kernel_size * self.n_features
        fan_out = self.n_filters
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        
        self.conv_w = np.random.uniform(-limit, limit, 
                                         (self.kernel_size, self.n_features, self.n_filters)).astype(np.float32)
        self.conv_b = np.zeros(self.n_filters, dtype=np.float32)
        
        # Após conv + pool: calcular tamanho do flatten
        conv_out_len = self.lookback - self.kernel_size + 1  # 50-5+1 = 46
        pool_out_len = conv_out_len // self.pool_size         # 46//5 = 9
        self.flat_size = pool_out_len * self.n_filters         # 9*8 = 72
        
        # Dense: flat_size → N_CLASSES
        fan_in_d = self.flat_size
        fan_out_d = N_CLASSES
        limit_d = np.sqrt(6.0 / (fan_in_d + fan_out_d))
        
        self.dense_w = np.random.uniform(-limit_d, limit_d, 
                                          (self.flat_size, N_CLASSES)).astype(np.float32)
        self.dense_b = np.zeros(N_CLASSES, dtype=np.float32)
    
    def _forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward pass completo.
        
        x: (lookback, n_features)
        Returns: dict com outputs intermediários (para backprop)
        """
        # Conv1D
        conv_out = _conv1d_forward(x, self.conv_w, self.conv_b)  # (46, 8)
        
        # ReLU
        relu_out = _relu(conv_out)  # (46, 8)
        
        # AvgPool
        pool_out = _avg_pool1d(relu_out, self.pool_size)  # (9, 8)
        
        # Flatten
        flat = pool_out.flatten()  # (72,)
        
        # Ajuste se flat_size diferente (bordas)
        if len(flat) < self.flat_size:
            flat = np.pad(flat, (0, self.flat_size - len(flat)))
        elif len(flat) > self.flat_size:
            flat = flat[:self.flat_size]
        
        # Dense
        logits = flat @ self.dense_w + self.dense_b  # (3,)
        
        # Softmax
        probs = _softmax(logits)
        
        return {
            "x": x,
            "conv_out": conv_out,
            "relu_out": relu_out,
            "pool_out": pool_out,
            "flat": flat,
            "logits": logits,
            "probs": probs
        }
    
    def _backward(self, cache: Dict[str, np.ndarray], label: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Backward pass (backpropagation simplificada).
        
        cache: outputs do forward pass
        label: one-hot encoded label (3,)
        
        Returns: gradientes dos pesos
        """
        probs = cache["probs"]
        flat = cache["flat"]
        relu_out = cache["relu_out"]
        conv_out = cache["conv_out"]
        x = cache["x"]
        
        # ===== Dense layer =====
        # dL/d_logits = probs - label (derivada cross-entropy + softmax)
        d_logits = probs - label  # (3,)
        
        # dL/d_dense_w = flat.T @ d_logits
        d_dense_w = np.outer(flat, d_logits)  # (72, 3)
        d_dense_b = d_logits.copy()  # (3,)
        
        # dL/d_flat = d_logits @ dense_w.T
        d_flat = d_logits @ self.dense_w.T  # (72,)
        
        # ===== Unflatten → pool output shape =====
        pool_out = cache["pool_out"]
        d_pool = d_flat[:pool_out.size].reshape(pool_out.shape)  # (9, 8)
        
        # ===== AvgPool backward =====
        # Distribui gradiente igualmente
        d_relu = np.zeros_like(relu_out)  # (46, 8)
        pool_out_len = relu_out.shape[0] // self.pool_size
        for i in range(pool_out_len):
            start = i * self.pool_size
            end = start + self.pool_size
            d_relu[start:end] = d_pool[i] / self.pool_size
        
        # ===== ReLU backward =====
        d_conv = d_relu * _relu_deriv(conv_out)  # (46, 8)
        
        # ===== Conv1D backward =====
        d_conv_w = np.zeros_like(self.conv_w)  # (5, 7, 8)
        d_conv_b = np.sum(d_conv, axis=0)  # (8,)
        
        out_len = d_conv.shape[0]
        for i in range(out_len):
            window = x[i:i + self.kernel_size]  # (5, 7)
            for f in range(self.n_filters):
                d_conv_w[:, :, f] += window * d_conv[i, f]
        
        return {
            "d_conv_w": d_conv_w,
            "d_conv_b": d_conv_b,
            "d_dense_w": d_dense_w,
            "d_dense_b": d_dense_b
        }
    
    def _update_weights(self, grads: Dict[str, np.ndarray], lr: float):
        """Atualiza pesos com SGD."""
        self.conv_w -= lr * np.clip(grads["d_conv_w"], -1.0, 1.0)
        self.conv_b -= lr * np.clip(grads["d_conv_b"], -1.0, 1.0)
        self.dense_w -= lr * np.clip(grads["d_dense_w"], -1.0, 1.0)
        self.dense_b -= lr * np.clip(grads["d_dense_b"], -1.0, 1.0)
    
    # ===================== FEATURE EXTRACTION =====================
    
    def extract_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Extrai 7 features das últimas N velas (mesmo padrão do neural_model.py).
        
        Returns: np.array shape (lookback, 7) ou None
        """
        if len(df) < self.lookback + 14:  # +14 para ATR
            return None
        
        recent = df.tail(self.lookback).copy()
        
        # ATR para normalização
        atr = self._calculate_atr(df)
        if atr <= 0:
            atr = 0.0001
        
        features = []
        for i in range(len(recent)):
            row = recent.iloc[i]
            
            open_p = float(row['open'])
            high = float(row['high'])
            low = float(row['low'])
            close = float(row['close'])
            
            candle_range = high - low
            if candle_range <= 0:
                candle_range = atr
            
            body = abs(close - open_p)
            body_top = max(close, open_p)
            body_bottom = min(close, open_p)
            upper_wick = high - body_top
            lower_wick = body_bottom - low
            
            # Normaliza pelo ATR
            body_size = body / atr
            upper_wick_norm = upper_wick / atr
            lower_wick_norm = lower_wick / atr
            momentum = (close - open_p) / atr
            range_norm = candle_range / atr
            
            # Direção e posição
            direction = 1.0 if close > open_p else 0.0
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
    
    # ===================== PREDIÇÃO =====================
    
    def predict(self, df: pd.DataFrame, direction_hint: str = "") -> Dict[str, Any]:
        """
        Faz predição usando CNN leve.
        
        Args:
            df: DataFrame com candles M1 (open, high, low, close)
            direction_hint: "CALL" ou "PUT" — direção sugerida pelo setup
        
        Returns:
            {
                "class": "CALL" | "PUT" | "NO_TRADE",
                "probability": float (0-1),
                "raw_probs": [p_call, p_put, p_notrade],
                "confidence": float (diferença entre top 2),
                "n_samples": int,
                "reliable": bool
            }
        """
        default_result = {
            "class": "NO_TRADE",
            "probability": 0.33,
            "raw_probs": [0.33, 0.33, 0.34],
            "confidence": 0.0,
            "n_samples": len(self.training_data),
            "reliable": False
        }
        
        features = self.extract_features(df)
        if features is None:
            default_result["reason"] = "insufficient_candles"
            return default_result
        
        # Se poucos dados de treinamento, retorna padrão (sem confiança)
        if len(self.training_data) < MIN_SAMPLES:
            # Usa heurística simples baseada nas features
            heuristic = self._heuristic_predict(features, direction_hint)
            heuristic["n_samples"] = len(self.training_data)
            heuristic["reliable"] = False
            heuristic["reason"] = f"warmup({len(self.training_data)}/{MIN_SAMPLES})"
            return heuristic
        
        try:
            cache = self._forward(features)
            probs = cache["probs"]
            
            # Classe predita
            predicted_class = int(np.argmax(probs))
            probability = float(probs[predicted_class])
            
            # Confiança = diferença entre top 2
            sorted_probs = np.sort(probs)[::-1]
            confidence = float(sorted_probs[0] - sorted_probs[1])
            
            cnn_class = CLASS_NAMES[predicted_class]
            
            self.stats["predictions"] += 1
            
            return {
                "class": cnn_class,
                "probability": probability,
                "raw_probs": probs.tolist(),
                "confidence": confidence,
                "n_samples": len(self.training_data),
                "reliable": True,
                "reason": "cnn_trained"
            }
        
        except Exception as e:
            default_result["reason"] = f"error:{str(e)[:50]}"
            return default_result
    
    def _heuristic_predict(self, features: np.ndarray, direction_hint: str = "") -> Dict[str, Any]:
        """
        Predição heurística quando CNN ainda não tem dados suficientes.
        Usa estatísticas das features para estimar direção.
        """
        # Últimas 10 candles
        recent = features[-10:]
        
        # Momentum médio
        avg_momentum = float(np.mean(recent[:, 4]))  # col 4 = momentum
        
        # Direção dominante (% de candles de alta)
        pct_bullish = float(np.mean(recent[:, 3]))  # col 3 = direction (1=alta, 0=baixa)
        
        # Posição média do close no range
        avg_position = float(np.mean(recent[:, 6]))  # col 6 = position
        
        # Tendência do momentum (acelerando ou desacelerando)
        mom_trend = float(np.mean(features[-5:, 4]) - np.mean(features[-15:-5, 4])) if len(features) >= 15 else 0.0
        
        # Score composto
        bull_score = (avg_momentum * 0.3 + (pct_bullish - 0.5) * 0.3 + 
                      (avg_position - 0.5) * 0.2 + mom_trend * 0.2)
        
        if bull_score > 0.05:
            pred_class = "CALL"
            prob = min(0.65, 0.50 + abs(bull_score))
        elif bull_score < -0.05:
            pred_class = "PUT"
            prob = min(0.65, 0.50 + abs(bull_score))
        else:
            pred_class = "NO_TRADE"
            prob = 0.50
        
        # Se tem hint e concorda, boost
        if direction_hint == pred_class:
            prob = min(0.70, prob + 0.05)
        
        # Probabilities
        if pred_class == "CALL":
            raw_probs = [prob, (1 - prob) * 0.6, (1 - prob) * 0.4]
        elif pred_class == "PUT":
            raw_probs = [(1 - prob) * 0.6, prob, (1 - prob) * 0.4]
        else:
            raw_probs = [0.30, 0.30, 0.40]
        
        return {
            "class": pred_class,
            "probability": prob,
            "raw_probs": raw_probs,
            "confidence": abs(bull_score)
        }
    
    # ===================== TREINAMENTO ONLINE =====================
    
    def add_sample(self, df: pd.DataFrame, direction: str, win: bool):
        """
        Adiciona amostra de treinamento após resultado de trade.
        
        Args:
            df: DataFrame com candles no momento da entrada (precisa ter ≥ lookback+14)
            direction: "CALL" ou "PUT"
            win: True se ganhou, False se perdeu
        """
        features = self.extract_features(df)
        if features is None:
            return
        
        # Label
        if win:
            label = CLASS_CALL if direction == "CALL" else CLASS_PUT
        else:
            label = CLASS_NO_TRADE  # Não deveria ter entrado
        
        sample = {
            "features": features.tolist(),
            "label": int(label),
            "direction": direction,
            "win": win,
            "timestamp": datetime.now().isoformat()
        }
        self.training_data.append(sample)
        self.trades_since_train += 1
        
        # Salva dados
        self._save_training_data()
        
        # Retreina se necessário
        if self.trades_since_train >= RETRAIN_EVERY and len(self.training_data) >= MIN_SAMPLES:
            self.retrain()
    
    def retrain(self):
        """Retreina o modelo com todos os dados acumulados."""
        n = len(self.training_data)
        if n < MIN_SAMPLES:
            return
        
        print(f"[LightCNN] Retreinando com {n} amostras...")
        
        # Prepara dados
        X = np.array([s['features'] for s in self.training_data], dtype=np.float32)
        y_labels = np.array([s['label'] for s in self.training_data], dtype=np.int32)
        
        # One-hot labels
        Y = np.zeros((n, N_CLASSES), dtype=np.float32)
        for i, lbl in enumerate(y_labels):
            Y[i, lbl] = 1.0
        
        # Pesos por amostra (amostras mais recentes pesam mais)
        sample_weights = np.array([DECAY_FACTOR ** (n - 1 - i) for i in range(n)], dtype=np.float32)
        sample_weights /= sample_weights.sum()
        
        # Class weights (balancear classes)
        class_counts = np.bincount(y_labels, minlength=N_CLASSES).astype(np.float32) + 1.0
        class_weights = n / (N_CLASSES * class_counts)
        
        best_loss = float('inf')
        patience = 5
        no_improve = 0
        
        lr = LEARNING_RATE
        
        for epoch in range(MAX_EPOCHS):
            # Shuffle
            indices = np.random.permutation(n)
            total_loss = 0.0
            n_batches = 0
            
            for start in range(0, n, BATCH_SIZE):
                end = min(start + BATCH_SIZE, n)
                batch_idx = indices[start:end]
                
                # Acumula gradientes do batch
                batch_grads = None
                batch_loss = 0.0
                
                for idx in batch_idx:
                    x_i = X[idx]
                    y_i = Y[idx]
                    w_i = sample_weights[idx] * class_weights[y_labels[idx]]
                    
                    # Forward
                    cache = self._forward(x_i)
                    
                    # Loss
                    batch_loss += _cross_entropy_loss(
                        cache["probs"].reshape(1, -1), 
                        y_i.reshape(1, -1)
                    ) * w_i
                    
                    # Backward
                    grads = self._backward(cache, y_i)
                    
                    # Escala por peso
                    for k in grads:
                        grads[k] = grads[k] * w_i
                    
                    if batch_grads is None:
                        batch_grads = grads
                    else:
                        for k in grads:
                            batch_grads[k] += grads[k]
                
                # Média dos gradientes
                bs = end - start
                for k in batch_grads:
                    batch_grads[k] /= bs
                
                # Update
                self._update_weights(batch_grads, lr)
                
                total_loss += batch_loss / bs
                n_batches += 1
            
            avg_loss = total_loss / max(n_batches, 1)
            
            # Early stopping
            if avg_loss < best_loss - 0.001:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                break
            
            # Reduz learning rate
            lr *= 0.95
        
        self.trades_since_train = 0
        self.total_trained += 1
        self.stats["total_trained"] = self.total_trained
        self.stats["last_train_time"] = datetime.now().isoformat()
        self.stats["last_loss"] = float(best_loss)
        
        # Salva pesos
        self._save_weights()
        
        print(f"[LightCNN] Treinamento concluído | loss={best_loss:.4f} | amostras={n}")
    
    # ===================== PERSISTÊNCIA =====================
    
    def _save_weights(self):
        """Salva pesos do modelo em JSON."""
        try:
            path = os.path.join(self.data_dir, WEIGHTS_FILE)
            data = {
                "conv_w": self.conv_w.tolist(),
                "conv_b": self.conv_b.tolist(),
                "dense_w": self.dense_w.tolist(),
                "dense_b": self.dense_b.tolist(),
                "stats": self.stats,
                "total_trained": self.total_trained,
                "saved_at": datetime.now().isoformat()
            }
            with open(path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[LightCNN] Erro ao salvar pesos: {e}")
    
    def _load_weights(self):
        """Carrega pesos do modelo de JSON."""
        try:
            path = os.path.join(self.data_dir, WEIGHTS_FILE)
            if not os.path.exists(path):
                return
            
            with open(path, 'r') as f:
                data = json.load(f)
            
            self.conv_w = np.array(data["conv_w"], dtype=np.float32)
            self.conv_b = np.array(data["conv_b"], dtype=np.float32)
            self.dense_w = np.array(data["dense_w"], dtype=np.float32)
            self.dense_b = np.array(data["dense_b"], dtype=np.float32)
            self.stats = data.get("stats", self.stats)
            self.total_trained = data.get("total_trained", 0)
            
            print(f"[LightCNN] Pesos carregados | treinos={self.total_trained}")
        except Exception as e:
            print(f"[LightCNN] Erro ao carregar pesos (reiniciando): {e}")
            self._init_weights()
    
    def _save_training_data(self):
        """Salva dados de treinamento."""
        try:
            path = os.path.join(self.data_dir, DATA_FILE)
            
            # Mantém só as últimas 500 amostras para não crescer demais
            to_save = self.training_data[-500:]
            
            with open(path, 'w') as f:
                json.dump(to_save, f)
        except Exception as e:
            print(f"[LightCNN] Erro ao salvar dados: {e}")
    
    def _load_training_data(self):
        """Carrega dados de treinamento."""
        try:
            path = os.path.join(self.data_dir, DATA_FILE)
            if not os.path.exists(path):
                return
            
            with open(path, 'r') as f:
                self.training_data = json.load(f)
            
            n = len(self.training_data)
            if n > 0:
                wins = sum(1 for s in self.training_data if s.get("win", False))
                print(f"[LightCNN] {n} amostras carregadas | wins={wins} ({100*wins/n:.0f}%)")
        except Exception as e:
            print(f"[LightCNN] Erro ao carregar dados: {e}")
            self.training_data = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do modelo."""
        n = len(self.training_data)
        wins = sum(1 for s in self.training_data if s.get("win", False))
        
        class_counts = {
            "CALL": sum(1 for s in self.training_data if s.get("label") == CLASS_CALL),
            "PUT": sum(1 for s in self.training_data if s.get("label") == CLASS_PUT),
            "NO_TRADE": sum(1 for s in self.training_data if s.get("label") == CLASS_NO_TRADE)
        }
        
        return {
            "total_samples": n,
            "wins": wins,
            "losses": n - wins,
            "win_rate": wins / n if n > 0 else 0,
            "class_counts": class_counts,
            "trades_since_train": self.trades_since_train,
            "total_trained": self.total_trained,
            "reliable": n >= MIN_SAMPLES,
            "predictions": self.stats.get("predictions", 0),
            "last_loss": self.stats.get("last_loss")
        }


# ===================== FACTORY =====================

_instance: Optional[LightCNN] = None

def get_cnn(data_dir: str = ".") -> LightCNN:
    """Singleton factory — retorna instância global do LightCNN."""
    global _instance
    if _instance is None:
        _instance = LightCNN(data_dir=data_dir)
    return _instance


# ===================== TESTE LOCAL =====================

if __name__ == "__main__":
    np.random.seed(42)
    
    # Simula 150 candles
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
    
    cnn = LightCNN(data_dir=".")
    
    print("=== Predição (sem treinamento) ===")
    result = cnn.predict(df_test, direction_hint="CALL")
    print(f"Classe: {result['class']}")
    print(f"Probabilidade: {result['probability']:.3f}")
    print(f"Confiança: {result['confidence']:.3f}")
    print(f"Confiável: {result['reliable']}")
    print(f"Razão: {result.get('reason', '?')}")
    
    # Simula treinamento
    print("\n=== Simulando 40 trades para treino ===")
    for i in range(40):
        # Gera candles aleatórios
        start = np.random.randint(0, 80)
        df_slice = df_test.iloc[start:start + 70].copy().reset_index(drop=True)
        direction = "CALL" if np.random.random() > 0.5 else "PUT"
        win = np.random.random() > 0.45  # 55% win rate
        cnn.add_sample(df_slice, direction, win)
    
    print("\n=== Predição (após treinamento) ===")
    result2 = cnn.predict(df_test, direction_hint="CALL")
    print(f"Classe: {result2['class']}")
    print(f"Probabilidade: {result2['probability']:.3f}")
    print(f"Confiança: {result2['confidence']:.3f}")
    print(f"Confiável: {result2['reliable']}")
    print(f"Razão: {result2.get('reason', '?')}")
    
    print("\n=== Estatísticas ===")
    stats = cnn.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Limpa arquivos de teste
    import os
    for f in [DATA_FILE, WEIGHTS_FILE]:
        if os.path.exists(f):
            os.remove(f)
    
    print("\n✅ Teste concluído!")
