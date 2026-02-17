# -*- coding: utf-8 -*-
"""
WS_HYBRID_AI — Sistema Híbrido Avançado 2026
✅ IA Neural (TensorFlow/Keras) - Rede profunda para padrões
✅ IA Bayesiana (Bayes + UCB) - Aprende com resultados em tempo real
✅ Estratégia Pernada B - Impulso + Pullback + Rompimento
✅ Detecção LTA/LTB - Linhas de tendência
✅ Filtro S/R Forte - Suporte/Resistência
✅ Sistema de Memória - Bloqueia contextos perdedores
"""

import os
import sys
import time
import math
import json
import logging
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# Verifica TA-Lib
try:
    import talib
    TALIB_OK = True
    print("[OK] TA-Lib instalado!")
except ImportError:
    TALIB_OK = False
    print("[AVISO] TA-Lib não instalado.")

# TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_OK = True
    print("[OK] TensorFlow instalado!")
except ImportError:
    TENSORFLOW_OK = False
    print("[AVISO] TensorFlow não instalado.")

from iqoptionapi.stable_api import IQ_Option

# ===================== CONFIG =====================
EMAIL = os.getenv("IQ_EMAIL", "") or "wstrader@wstrader.onmicrosoft.com"
SENHA = os.getenv("IQ_PASS", "") or "P152030@w"
CONTA = os.getenv("IQ_CONTA", "PRACTICE")

TF_M1 = 60
DECIDIR_ANTES_FECHAR_SEC = int(os.getenv("WS_DECIDIR_ANTES_FECHAR", "10"))
N_M1 = int(os.getenv("WS_N_M1", "340"))

PAYOUT_MINIMO = int(os.getenv("WS_PAYOUT_MIN", "80"))
NUM_ATIVOS = int(os.getenv("WS_NUM_ATIVOS", "12"))
PAYOUT_REFRESH_SEC = int(os.getenv("WS_PAYOUT_REFRESH", "180"))

EXP_FIXA = int(os.getenv("WS_EXP_MIN", "1"))
VALOR_MINIMO = float(os.getenv("WS_VALOR_MINIMO", "2"))
STAKE_FIXA = float(os.getenv("WS_STAKE", "5"))

# GESTÃO DE BANCA
PERCENT_BANCA = float(os.getenv("WS_PERCENT_BANCA", "1.0"))
META_LUCRO_PERCENT = float(os.getenv("WS_META_LUCRO", "7.0"))
STOP_LOSS_PERCENT = float(os.getenv("WS_STOP_LOSS", "5.0"))
USE_DYNAMIC_STAKE = (os.getenv("WS_DYNAMIC_STAKE", "1").strip() == "1")

COOLDOWN_ATIVO = int(os.getenv("WS_COOLDOWN_ATIVO", "60"))

# IA NEURAL
NEURAL_ON = (os.getenv("WS_NEURAL_ON", "1").strip() == "1")
NEURAL_FILE = os.getenv("WS_NEURAL_FILE", "ws_hybrid_neural.h5")

# IA BAYESIANA
BAYES_ON = (os.getenv("WS_BAYES_ON", "1").strip() == "1")
BAYES_FILE = os.getenv("WS_BAYES_FILE", "ws_hybrid_bayes.json")
BAYES_MIN_SAMPLES = int(os.getenv("WS_BAYES_MIN_SAMPLES", "10"))
BAYES_MIN_PROB = float(os.getenv("WS_BAYES_MIN_PROB", "0.45"))
BAYES_MIN_WINRATE = float(os.getenv("WS_BAYES_MIN_WINRATE", "0.45"))

# MEMÓRIA DE MERCADO
MEMORY_FILE = "ws_hybrid_memory.json"

# PERNADA B
IMPULSO_MIN_ATR = float(os.getenv("WS_IMPULSO_MIN_ATR", "0.60"))
IMPULSO_JANELA_MIN = int(os.getenv("WS_IMP_JMIN", "3"))
IMPULSO_JANELA_MAX = int(os.getenv("WS_IMP_JMAX", "12"))
PULLBACK_MIN = int(os.getenv("WS_PB_MIN", "1"))
PULLBACK_MAX = int(os.getenv("WS_PB_MAX", "5"))
RETR_MIN = float(os.getenv("WS_RETR_MIN", "0.15"))
RETR_MAX = float(os.getenv("WS_RETR_MAX", "0.75"))
MIN_EFF_A = float(os.getenv("WS_MIN_EFF_A", "0.45"))

# S/R
SR_LOOKBACK = int(os.getenv("WS_SR_LOOKBACK", "200"))
SR_CLUSTER_ATR = float(os.getenv("WS_SR_CLUSTER_ATR", "0.5"))
SR_MIN_TOUCHES = int(os.getenv("WS_SR_MIN_TOUCHES", "3"))
SR_BLOCK_DIST_ATR = float(os.getenv("WS_SR_BLOCK_ATR", "0.5"))

# SCORE MÍNIMO
MIN_SCORE = float(os.getenv("WS_MIN_SCORE", "0.55"))

# ===================== LOG =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [WS_HYBRID] %(message)s"
)
log = logging.getLogger("WS_HYBRID")

class C:
    G = "\033[92m"
    R = "\033[91m"
    Y = "\033[93m"
    B = "\033[94m"
    C = "\033[96m"
    M = "\033[95m"
    Z = "\033[0m"

def paint(s: str, color: str) -> str:
    return f"{color}{s}{C.Z}"

# ===================== GLOBALS =====================
_cache_ativos: List[str] = []
_cache_ativos_ts: float = 0.0
cooldown: Dict[str, float] = {}
trade_memory: Dict[str, List] = {"wins": [], "losses": []}
bayes_stats: Dict[str, Any] = {"meta": {"total": 0}, "arms": {}, "patterns": {}}
is_first_connection = True
last_reconnect_time = 0

# ===================== REDE NEURAL TENSORFLOW =====================
class NeuralNetwork:
    def __init__(self, input_dim: int = 25):
        self.input_dim = input_dim
        self.model = None
        self.build_model()
    
    def build_model(self):
        if not TENSORFLOW_OK:
            return
        
        self.model = Sequential([
            Input(shape=(self.input_dim,)),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        log.info(paint("[NEURAL] Modelo criado: 25->128->64->32->1", C.C))
    
    def predict(self, features: np.ndarray) -> float:
        if self.model is None or not TENSORFLOW_OK:
            return 0.5
        try:
            features = np.array(features).reshape(1, -1)
            pred = self.model.predict(features, verbose=0)[0][0]
            return float(pred)
        except:
            return 0.5
    
    def train_on_result(self, features: np.ndarray, is_win: bool, epochs: int = 3):
        if self.model is None or not TENSORFLOW_OK:
            return
        try:
            features = np.array(features).reshape(1, -1)
            label = np.array([[1.0 if is_win else 0.0]])
            self.model.fit(features, label, epochs=epochs, verbose=0)
            log.info(paint(f"[NEURAL] Treinado com {'WIN' if is_win else 'LOSS'}", C.G if is_win else C.Y))
        except Exception as e:
            log.warning(f"[NEURAL] Erro ao treinar: {e}")
    
    def save(self, path: str):
        if self.model is None:
            return
        try:
            self.model.save(path)
            log.info(paint(f"[NEURAL] Modelo salvo: {path}", C.C))
        except Exception as e:
            log.warning(f"[NEURAL] Erro ao salvar: {e}")
    
    def load(self, path: str) -> bool:
        if not TENSORFLOW_OK:
            return False
        try:
            if os.path.exists(path):
                self.model = keras.models.load_model(path)
                log.info(paint(f"[NEURAL] Modelo carregado: {path}", C.G))
                return True
        except Exception as e:
            log.warning(f"[NEURAL] Erro ao carregar: {e}")
        return False

neural_net = NeuralNetwork()

# ===================== IA BAYESIANA =====================
def bayes_load() -> Dict[str, Any]:
    global bayes_stats
    try:
        if os.path.exists(BAYES_FILE):
            with open(BAYES_FILE, "r", encoding="utf-8") as f:
                bayes_stats = json.load(f)
                total = bayes_stats.get('meta', {}).get('total', 0)
                patterns = len(bayes_stats.get('patterns', {}))
                log.info(paint(f"[BAYES] Carregado: {total} operações | {patterns} padrões", C.C))
    except Exception as e:
        log.warning(f"[BAYES] Erro ao carregar: {e}")
    return bayes_stats

def bayes_save():
    try:
        with open(BAYES_FILE, "w", encoding="utf-8") as f:
            json.dump(bayes_stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"[BAYES] Erro ao salvar: {e}")

def bayes_make_key(setup: Dict[str, Any]) -> str:
    """Cria chave única para o padrão"""
    d = str(setup.get("dir", "NEUTRAL"))
    trend = str(setup.get("trend", "?"))
    pb_len = int(setup.get("pb_len", 0))
    retr_bucket = int(float(setup.get("retr", 0.5)) * 10)
    eff_bucket = int(float(setup.get("effA", 0.5)) * 10)
    
    return f"{d}|{trend}|pb{pb_len}|re{retr_bucket}|eff{eff_bucket}"

def bayes_predict(setup: Dict[str, Any]) -> Dict[str, float]:
    """Previsão Bayesiana com UCB"""
    key = bayes_make_key(setup)
    arms = bayes_stats.setdefault("arms", {})
    meta = bayes_stats.setdefault("meta", {"total": 0})
    
    total = int(meta.get("total", 0))
    arm = arms.get(key)
    
    # Prior baseado no score
    prior = 0.50 + (float(setup.get("score", 0.5)) - 0.50) * 0.30
    
    if arm is None:
        return {"prob": prior, "bayes": prior, "n_arm": 0, "key": key}
    
    a = float(arm.get("a", 1.0))
    b = float(arm.get("b", 1.0))
    n = int(arm.get("n", 0))
    
    bayes_mean = a / (a + b)
    
    # UCB bonus
    if n > 0:
        bonus = math.sqrt(2.0 * math.log(max(2, total + 1)) / max(1, n))
        ucb = min(1.0, max(0.0, bayes_mean + bonus))
    else:
        ucb = 1.0
    
    # Mistura prior + bayes
    w = min(1.0, n / (n + 15.0))
    prob = (1.0 - w) * prior + w * bayes_mean
    
    return {"prob": float(prob), "bayes": float(bayes_mean), "ucb": float(ucb), "n_arm": n, "key": key}

def bayes_update(setup: Dict[str, Any], pnl: float):
    """Atualiza estatísticas Bayesianas"""
    if pnl == 0:
        return
    
    key = bayes_make_key(setup)
    arms = bayes_stats.setdefault("arms", {})
    meta = bayes_stats.setdefault("meta", {"total": 0})
    patterns = bayes_stats.setdefault("patterns", {})
    
    arm = arms.get(key)
    if arm is None:
        prior = 0.50
        arms[key] = {"a": 2.0 * prior, "b": 2.0 * (1.0 - prior), "n": 0}
        arm = arms[key]
    
    a = float(arm.get("a", 1.0))
    b = float(arm.get("b", 1.0))
    n = int(arm.get("n", 0))
    
    if pnl > 0:
        a += 1.0
    else:
        b += 1.0
    
    n += 1
    arm["a"], arm["b"], arm["n"] = a, b, n
    meta["total"] = int(meta.get("total", 0)) + 1
    
    # Padrão tracking
    pattern = patterns.get(key)
    if pattern is None:
        patterns[key] = {"trades": 0, "wins": 0, "losses": 0}
        pattern = patterns[key]
    
    pattern["trades"] += 1
    if pnl > 0:
        pattern["wins"] += 1
    else:
        pattern["losses"] += 1
    
    log.info(paint(f"[BAYES] Atualizado: {key} -> {pattern['wins']}W/{pattern['losses']}L", C.C))

def bayes_should_block(setup: Dict[str, Any]) -> Tuple[bool, str]:
    """Verifica se deve bloquear padrão baseado no histórico"""
    key = bayes_make_key(setup)
    patterns = bayes_stats.get("patterns", {})
    pattern = patterns.get(key)
    
    if pattern is None or pattern["trades"] < BAYES_MIN_SAMPLES:
        trades = pattern["trades"] if pattern else 0
        return False, f"learning({trades}/{BAYES_MIN_SAMPLES})"
    
    winrate = pattern["wins"] / max(1, pattern["trades"])
    
    if winrate < BAYES_MIN_WINRATE:
        return True, f"blocked_wr={winrate:.0%}({pattern['wins']}W/{pattern['losses']}L)"
    
    return False, f"approved_wr={winrate:.0%}"

# ===================== MEMÓRIA DE MERCADO =====================
def memory_load():
    global trade_memory
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                trade_memory = json.load(f)
                wins = len(trade_memory.get('wins', []))
                losses = len(trade_memory.get('losses', []))
                log.info(paint(f"[MEMORY] Carregado: {wins} WINs | {losses} LOSSes", C.C))
    except:
        trade_memory = {"wins": [], "losses": []}

def memory_save():
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(trade_memory, f, ensure_ascii=False, indent=2)
    except:
        pass

def memory_add(context: Dict[str, Any], is_win: bool):
    """Salva contexto na memória"""
    entry = {
        "direction": context.get("direction", ""),
        "trend": context.get("trend", ""),
        "volatility": context.get("volatility", ""),
        "rsi_zone": context.get("rsi_zone", ""),
        "timestamp": time.time()
    }
    
    if is_win:
        trade_memory["wins"].append(entry)
        trade_memory["wins"] = trade_memory["wins"][-100:]
        log.info(paint(f"[MEMORY] ✅ WIN salvo: {entry['trend']} + {entry['direction']}", C.G))
    else:
        trade_memory["losses"].append(entry)
        trade_memory["losses"] = trade_memory["losses"][-100:]
        log.info(paint(f"[MEMORY] ⚠️ LOSS salvo: {entry['trend']} + {entry['direction']}", C.Y))
    
    memory_save()

def memory_check_context(trend: str, direction: str, rsi_zone: str) -> Tuple[int, int]:
    """Conta contextos similares em WINs e LOSSes"""
    similar_wins = 0
    similar_losses = 0
    
    for win in trade_memory.get("wins", [])[-50:]:
        if (win.get("trend") == trend and 
            win.get("direction") == direction and
            win.get("rsi_zone") == rsi_zone):
            similar_wins += 1
    
    for loss in trade_memory.get("losses", [])[-50:]:
        if (loss.get("trend") == trend and 
            loss.get("direction") == direction and
            loss.get("rsi_zone") == rsi_zone):
            similar_losses += 1
    
    return similar_wins, similar_losses

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
    time.sleep(s + 0.1)

def end_ts_closed(tf: int) -> float:
    now = time.time()
    return now - (now % tf) - 1

# ===================== IQ OPTION =====================
def patch_websocket():
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
                except:
                    return None
            except:
                return None
        WebsocketClient.on_close = on_close_compat
        WebsocketClient.__WS_PATCHED__ = True
    except:
        pass

def conectar_iq() -> IQ_Option:
    if not EMAIL or not SENHA:
        raise RuntimeError("Defina IQ_EMAIL e IQ_PASS")
    patch_websocket()
    log.info("Conectando à IQ Option...")
    iq = IQ_Option(EMAIL, SENHA)
    iq.connect()
    
    for _ in range(15):
        if iq.check_connect():
            break
        time.sleep(1.5)
    
    if not iq.check_connect():
        raise RuntimeError("Falha na conexão")
    
    iq.change_balance(CONTA)
    try:
        log.info(paint(f"Conectado | Saldo: {iq.get_balance():.2f} | Conta: {CONTA}", C.G))
    except:
        log.info(f"Conectado | Conta: {CONTA}")
    
    return iq

def ensure_connected(iq: Optional[IQ_Option]) -> IQ_Option:
    global is_first_connection, last_reconnect_time
    
    if iq is None:
        if is_first_connection:
            is_first_connection = False
            return conectar_iq()
        last_reconnect_time = time.time()
        log.warning(paint("🔄 Reconectando após queda...", C.Y))
        return conectar_iq()
    
    try:
        if iq.check_connect():
            return iq
    except:
        pass
    
    last_reconnect_time = time.time()
    log.warning(paint("🔄 Conexão perdida! Reconectando...", C.Y))
    return conectar_iq()

def safe_call(iq: IQ_Option, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        if "10054" in msg or "forçado" in msg or "goodbye" in msg:
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
        
        if len(df) < max(50, SR_LOOKBACK + 20):
            return None
        
        return df
    except:
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
    except:
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
        except:
            payout = 0
        if payout >= PAYOUT_MINIMO:
            filtrados.append((a, payout))
    
    filtrados.sort(key=lambda x: x[1], reverse=True)
    top = [a for a, _ in filtrados[:NUM_ATIVOS]]
    _cache_ativos = top
    _cache_ativos_ts = now
    log.info(f"TOP ativos: {top}")
    return top

# ===================== INDICADORES =====================
def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    sub = df.tail(period + 2)
    h = sub["high"].to_numpy(float)
    l = sub["low"].to_numpy(float)
    c = sub["close"].to_numpy(float)
    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1]))
    )
    return float(np.mean(tr[-period:]))

def calc_rsi(df: pd.DataFrame, period: int = 14) -> float:
    closes = df["close"].tail(period + 1).to_numpy(float)
    changes = np.diff(closes)
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 1e-9
    rs = avg_gain / max(avg_loss, 1e-9)
    return 100 - (100 / (1 + rs))

def calc_trend(df: pd.DataFrame, lookback: int = 10) -> str:
    """Calcula tendência baseada nas últimas velas"""
    sub = df.tail(lookback)
    closes = sub["close"].to_numpy(float)
    opens = sub["open"].to_numpy(float)
    
    bullish = sum(1 for i in range(len(sub)) if closes[i] > opens[i])
    bearish = lookback - bullish
    
    # Também verifica inclinação geral
    price_start = closes[0]
    price_end = closes[-1]
    change_pct = (price_end - price_start) / max(price_start, 1e-9) * 100
    
    if bullish >= 7 or change_pct > 0.3:
        return "FORTE_ALTA"
    elif bullish >= 5:
        return "ALTA"
    elif bearish >= 7 or change_pct < -0.3:
        return "FORTE_BAIXA"
    elif bearish >= 5:
        return "BAIXA"
    else:
        return "LATERAL"

def leg_efficiency(df_leg: pd.DataFrame) -> float:
    if len(df_leg) < 2:
        return 0.0
    closes = df_leg["close"].to_numpy(float)
    net = abs(closes[-1] - closes[0])
    gross = np.sum(np.abs(np.diff(closes))) + 1e-9
    return float(net / gross)

def chop_stats(df: pd.DataFrame, lookback: int = 20) -> Tuple[float, float]:
    """Retorna (flips_frac, efficiency) para detectar mercado lateral"""
    sub = df.tail(lookback)
    if len(sub) < 6:
        return 1.0, 0.0
    
    opens = sub["open"].to_numpy(float)
    closes = sub["close"].to_numpy(float)
    
    dirs = [1 if closes[i] > opens[i] else -1 for i in range(len(sub))]
    
    flips = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i-1])
    flips_frac = flips / max(1, len(dirs) - 1)
    
    eff = leg_efficiency(sub)
    
    return float(flips_frac), float(eff)

# ===================== S/R (SUPORTE/RESISTÊNCIA) =====================
def cluster_levels(levels: List[float], tol: float) -> List[Tuple[float, int]]:
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

def find_sr_levels(df: pd.DataFrame, atr_val: float) -> Tuple[List[Tuple[float,int]], List[Tuple[float,int]]]:
    """Encontra níveis de suporte e resistência"""
    sub = df.tail(SR_LOOKBACK)
    h = sub["high"].to_numpy(float)
    l = sub["low"].to_numpy(float)
    
    highs = []
    lows = []
    
    k = 2
    for i in range(k, len(sub) - k):
        if h[i] == np.max(h[i-k:i+k+1]):
            highs.append(float(h[i]))
        if l[i] == np.min(l[i-k:i+k+1]):
            lows.append(float(l[i]))
    
    tol = max(atr_val * SR_CLUSTER_ATR, 1e-9)
    
    res = cluster_levels(highs, tol)
    sup = cluster_levels(lows, tol)
    
    res = [(lvl, n) for (lvl, n) in res if n >= SR_MIN_TOUCHES]
    sup = [(lvl, n) for (lvl, n) in sup if n >= SR_MIN_TOUCHES]
    
    return res, sup

def check_sr_block(df: pd.DataFrame, atr_val: float, direction: str) -> Optional[str]:
    """Verifica se deve bloquear por S/R"""
    res, sup = find_sr_levels(df, atr_val)
    price = float(df["close"].iloc[-1])
    atr_safe = max(atr_val, 1e-9)
    
    if direction == "CALL" and res:
        above = [(lvl, t) for (lvl, t) in res if lvl > price]
        for lvl, touches in sorted(above, key=lambda x: x[0])[:3]:
            dist = (lvl - price) / atr_safe
            if dist <= SR_BLOCK_DIST_ATR:
                return f"RES_PERTO(lvl={lvl:.5f},dist={dist:.2f}ATR)"
    
    if direction == "PUT" and sup:
        below = [(lvl, t) for (lvl, t) in sup if lvl < price]
        for lvl, touches in sorted(below, key=lambda x: -x[0])[:3]:
            dist = (price - lvl) / atr_safe
            if dist <= SR_BLOCK_DIST_ATR:
                return f"SUP_PERTO(lvl={lvl:.5f},dist={dist:.2f}ATR)"
    
    return None

def check_sr_pingpong(df: pd.DataFrame, atr_val: float) -> Optional[str]:
    """Detecta mercado em corredor (ping-pong)"""
    res, sup = find_sr_levels(df, atr_val)
    if not res or not sup:
        return None
    
    price = float(df["close"].iloc[-1])
    atr_safe = max(atr_val, 1e-9)
    
    above = [(lvl, t) for (lvl, t) in res if lvl >= price]
    below = [(lvl, t) for (lvl, t) in sup if lvl <= price]
    
    if not above or not below:
        return None
    
    r_lvl = min(above, key=lambda x: x[0])[0]
    s_lvl = max(below, key=lambda x: x[0])[0]
    
    corridor = abs(r_lvl - s_lvl) / atr_safe
    
    if corridor <= 1.0:
        return f"pingpong(corredor={corridor:.2f}ATR)"
    
    return None

# ===================== LINHA DE TENDÊNCIA =====================
def detect_trendline(df: pd.DataFrame, lookback: int, direction: str) -> Optional[Tuple[float, float]]:
    """Detecta LTA (alta) ou LTB (baixa)"""
    if len(df) < lookback:
        return None
    
    sub = df.tail(lookback)
    
    if direction == "CALL":
        lows = sub["low"].to_numpy(float)
        pivots = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                pivots.append((i, lows[i]))
        
        if len(pivots) < 2 or pivots[-1][1] <= pivots[0][1]:
            return None
        
        x = np.array([p[0] for p in pivots])
        y = np.array([p[1] for p in pivots])
        slope, intercept = np.polyfit(x, y, 1)
        
        if slope <= 0:
            return None
        return (float(slope), float(intercept))
    
    else:
        highs = sub["high"].to_numpy(float)
        pivots = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                pivots.append((i, highs[i]))
        
        if len(pivots) < 2 or pivots[-1][1] >= pivots[0][1]:
            return None
        
        x = np.array([p[0] for p in pivots])
        y = np.array([p[1] for p in pivots])
        slope, intercept = np.polyfit(x, y, 1)
        
        if slope >= 0:
            return None
        return (float(slope), float(intercept))

def check_trendline_confluence(df: pd.DataFrame, pb_extreme: float, direction: str, atr_val: float) -> float:
    """Verifica confluência com linha de tendência (0.0 a 1.0)"""
    trendline = detect_trendline(df.tail(50), 50, direction)
    
    if trendline is None:
        return 0.0
    
    slope, intercept = trendline
    x_pb = len(df.tail(50)) - 1
    lt_value = slope * x_pb + intercept
    
    distance = abs(pb_extreme - lt_value) / max(atr_val, 1e-9)
    
    if distance < 0.3:
        return 1.0
    elif distance < 0.6:
        return 0.6
    elif distance < 1.0:
        return 0.3
    else:
        return 0.0

# ===================== FEATURES PARA NEURAL =====================
def extract_features(df: pd.DataFrame, direction: str, setup: Dict) -> np.ndarray:
    """Extrai features para a rede neural"""
    closes = df["close"].tail(20).to_numpy(float)
    opens = df["open"].tail(20).to_numpy(float)
    highs = df["high"].tail(20).to_numpy(float)
    lows = df["low"].tail(20).to_numpy(float)
    
    price_mean = np.mean(closes)
    price_std = np.std(closes) + 1e-9
    
    features = []
    
    # 1. Direção
    features.append(1.0 if direction == "CALL" else 0.0)
    
    # 2-6. Retornos
    returns = np.diff(closes[-6:]) / price_std
    features.extend(returns.tolist())
    
    # 7-11. Ranges
    ranges = (highs[-5:] - lows[-5:]) / price_std
    features.extend(ranges.tolist())
    
    # 12. RSI
    rsi = calc_rsi(df, 14)
    features.append((rsi - 50) / 50)
    
    # 13. ATR
    atr_val = calc_atr(df, 14)
    features.append(atr_val / price_std)
    
    # 14-17. Setup
    features.append(float(setup.get("score", 0.5)))
    features.append(float(setup.get("effA", 0.5)))
    features.append(float(setup.get("retr", 0.5)))
    features.append(float(setup.get("pb_len", 2)) / 5.0)
    
    # 18. LT confluence
    features.append(float(setup.get("lt_confluence", 0.0)))
    
    # 19. Bullish ratio
    bullish_last5 = sum(1 for i in range(-5, 0) if closes[i] > opens[i]) / 5.0
    features.append(bullish_last5)
    
    # 20. Volatilidade
    vol_recent = np.std(closes[-10:]) / price_std
    features.append(vol_recent)
    
    # 21. Momentum
    momentum = (closes[-1] - np.mean(closes[-10:])) / price_std
    features.append(momentum)
    
    # 22-25. Chop stats e padding
    flips, eff = chop_stats(df, 20)
    features.append(flips)
    features.append(eff)
    
    while len(features) < 25:
        features.append(0.0)
    
    return np.array(features[:25], dtype=np.float32)

# ===================== PERNADA B =====================
def analyze_pernada_b(df: pd.DataFrame, atr_val: float) -> Dict[str, Any]:
    """Analisa setup de Pernada B completo"""
    if len(df) < 100:
        return {"trade": False, "reasons": ["dados_insuficientes"]}
    
    # Contexto
    trend = calc_trend(df, 10)
    rsi = calc_rsi(df, 14)
    rsi_zone = "OVERSOLD" if rsi < 30 else ("OVERBOUGHT" if rsi > 70 else "NEUTRO")
    
    # Volatilidade
    ranges = [df["high"].iloc[i] - df["low"].iloc[i] for i in range(-10, 0)]
    avg_range = np.mean(ranges)
    old_ranges = [df["high"].iloc[i] - df["low"].iloc[i] for i in range(-30, -10)]
    volatility = "ALTA" if avg_range > np.mean(old_ranges) * 1.3 else "NORMAL"
    
    # Choppiness
    flips, eff = chop_stats(df, 20)
    if flips > 0.70 and eff < 0.15:
        return {"trade": False, "reasons": [f"lateral_chop(flips={flips:.2f},eff={eff:.2f})"]}
    
    # Ping-pong
    ping = check_sr_pingpong(df, atr_val)
    if ping:
        return {"trade": False, "reasons": [ping]}
    
    decision = df.iloc[-1]
    best = None
    
    for pb_len in range(PULLBACK_MIN, PULLBACK_MAX + 1):
        pb = df.iloc[-(pb_len + 1):-1]
        if len(pb) != pb_len:
            continue
        
        for w in range(IMPULSO_JANELA_MIN, IMPULSO_JANELA_MAX + 1):
            imp = df.iloc[-(pb_len + 1 + w):-(pb_len + 1)]
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
            
            dir_A = "PUT" if move < 0 else ("CALL" if move > 0 else "NEUTRAL")
            if dir_A == "NEUTRAL":
                continue
            
            eff_A = leg_efficiency(imp)
            if eff_A < MIN_EFF_A:
                continue
            
            # Velas contra no pullback
            contra = 0
            for _, r in pb.iterrows():
                o = float(r["open"])
                c = float(r["close"])
                if dir_A == "PUT" and c > o:
                    contra += 1
                if dir_A == "CALL" and c < o:
                    contra += 1
            
            if contra < max(1, int(pb_len * 0.5)):
                continue
            
            pb_high = float(pb["high"].max())
            pb_low = float(pb["low"].min())
            
            if dir_A == "PUT":
                retr = (pb_high - bot) / max(size_A, 1e-9)
            else:
                retr = (top - pb_low) / max(size_A, 1e-9)
            
            if retr < RETR_MIN or retr > RETR_MAX:
                continue
            
            dir_entrada = dir_A
            
            sr_block = check_sr_block(df, atr_val, dir_entrada)
            if sr_block:
                continue
            
            c1 = float(decision["close"])
            
            if dir_entrada == "CALL":
                if c1 <= pb_low:
                    continue
                dist = (c1 - pb_low) / max(atr_val, 1e-9)
                pb_extreme = pb_low
            else:
                if c1 >= pb_high:
                    continue
                dist = (pb_high - c1) / max(atr_val, 1e-9)
                pb_extreme = pb_high
            
            if dist > 0.5:
                continue
            
            lt_conf = check_trendline_confluence(df, pb_extreme, dir_entrada, atr_val)
            
            # ===== SCORE =====
            score = 0.40
            
            # Impulso
            score += min(0.10, (size_A / atr_val - IMPULSO_MIN_ATR) * 0.05)
            
            # Eficiência
            score += min(0.12, max(0, (eff_A - MIN_EFF_A) * 0.30))
            
            # Retração ideal
            if 0.30 <= retr <= 0.50:
                score += 0.08
            elif 0.25 <= retr <= 0.60:
                score += 0.04
            
            # Pullback
            if 2 <= pb_len <= 4:
                score += 0.05
            
            # LT
            score += lt_conf * 0.15
            
            # Tendência
            if (dir_entrada == "CALL" and trend in ["ALTA", "FORTE_ALTA"]) or \
               (dir_entrada == "PUT" and trend in ["BAIXA", "FORTE_BAIXA"]):
                score += 0.10
            elif trend == "LATERAL":
                score -= 0.05
            
            # Volatilidade
            if volatility == "ALTA":
                score -= 0.05
            
            # Chop penalty
            if flips > 0.55:
                score -= 0.05
            
            score = float(max(0.0, min(0.95, score)))
            
            setup = {
                "trade": True,
                "dir": dir_entrada,
                "score": score,
                "pb_len": pb_len,
                "retr": float(retr),
                "effA": float(eff_A),
                "A_atr": float(size_A / atr_val),
                "distBreak": float(dist),
                "lt_confluence": float(lt_conf),
                "trend": trend,
                "rsi": float(rsi),
                "rsi_zone": rsi_zone,
                "volatility": volatility,
                "flips": float(flips),
                "eff_zone": float(eff),
                "reasons": [
                    f"pernadaB_{dir_entrada}",
                    f"A={size_A/atr_val:.2f}ATR",
                    f"retr={retr:.2f}",
                    f"pb={pb_len}",
                    f"eff={eff_A:.2f}",
                    f"trend={trend}",
                    f"LT={lt_conf:.2f}"
                ]
            }
            
            if best is None or setup["score"] > best["score"]:
                best = setup
    
    if best is None:
        return {"trade": False, "reasons": ["sem_setup_valido"]}
    
    return best

# ===================== DECISÃO PRINCIPAL =====================
def should_enter_trade(df: pd.DataFrame, ativo: str) -> Dict[str, Any]:
    """
    Sistema híbrido de decisão:
    1. Pernada B (setup técnico)
    2. Memória de Mercado (contextos)
    3. IA Neural (TensorFlow)
    4. IA Bayesiana (histórico de padrões)
    """
    if len(df) < 100:
        return {"enter": False, "reason": "dados_insuficientes"}
    
    atr_val = calc_atr(df, 14)
    
    # 1. PERNADA B
    setup = analyze_pernada_b(df, atr_val)
    
    if not setup.get("trade", False):
        reasons = setup.get('reasons', [])
        if reasons and 'sem_setup_valido' not in str(reasons):
            log.info(paint(f"[{ativo}] ⏸️ {', '.join(reasons)}", C.Y))
        return {"enter": False, "reason": "sem_setup"}
    
    direction = setup["dir"]
    score = setup["score"]
    trend = setup["trend"]
    rsi_zone = setup["rsi_zone"]
    
    log.info(paint(f"[{ativo}] 📊 Setup: {direction} | Score: {score:.2f} | Trend: {trend} | RSI: {setup['rsi']:.0f}", C.B))
    log.info(paint(f"[{ativo}]   {', '.join(setup.get('reasons', []))}", C.B))
    
    # 2. MEMÓRIA DE MERCADO
    sim_wins, sim_losses = memory_check_context(trend, direction, rsi_zone)
    total_similar = sim_wins + sim_losses
    
    if total_similar >= 3:
        memory_wr = sim_wins / total_similar
        log.info(paint(f"[{ativo}] 🧠 Memória: {sim_wins}W / {sim_losses}L = {memory_wr*100:.0f}%", 
                       C.G if memory_wr >= 0.55 else C.R))
        
        if sim_losses >= 3 and sim_wins < sim_losses:
            log.info(paint(f"[{ativo}] 🚫 BLOQUEADO por memória (contexto perdedor)", C.R))
            return {"enter": False, "reason": "memoria_bloqueou"}
        
        score = score * 0.6 + memory_wr * 0.4
    
    # 3. IA NEURAL
    if NEURAL_ON and neural_net.model is not None:
        features = extract_features(df, direction, setup)
        neural_pred = neural_net.predict(features)
        
        log.info(paint(f"[{ativo}] 🤖 Neural: {neural_pred:.2f}", 
                       C.G if neural_pred >= 0.55 else C.R))
        
        score = score * 0.7 + neural_pred * 0.3
        setup["neural_pred"] = float(neural_pred)
        setup["features"] = features.tolist()
    
    # 4. IA BAYESIANA
    if BAYES_ON:
        bayes_pred = bayes_predict(setup)
        
        log.info(paint(f"[{ativo}] 📈 Bayes: prob={bayes_pred['prob']:.2f} (n={bayes_pred['n_arm']})", 
                       C.G if bayes_pred['prob'] >= 0.50 else C.Y))
        
        should_block, block_reason = bayes_should_block(setup)
        if should_block:
            log.info(paint(f"[{ativo}] 🚫 BLOQUEADO por Bayes: {block_reason}", C.R))
            return {"enter": False, "reason": f"bayes_bloqueou:{block_reason}"}
        
        score = score * 0.8 + bayes_pred['prob'] * 0.2
        setup["bayes_prob"] = float(bayes_pred['prob'])
        setup["bayes_n"] = int(bayes_pred['n_arm'])
    
    setup["score"] = float(score)
    
    # 5. DECISÃO FINAL
    if score < MIN_SCORE:
        log.info(paint(f"[{ativo}] ⏸️ Score final baixo ({score:.2f} < {MIN_SCORE})", C.Y))
        return {"enter": False, "reason": "score_baixo"}
    
    # APROVADO!
    color = C.G if direction == "CALL" else C.R
    log.info(paint("=" * 50, color))
    log.info(paint(f"[{ativo}] ✅ ENTRAR {direction} | Score: {score:.2f}", color))
    log.info(paint("=" * 50, color))
    
    return {
        "enter": True,
        "direction": direction,
        "score": score,
        "setup": setup,
        "context": {
            "trend": trend,
            "rsi_zone": rsi_zone,
            "volatility": setup.get("volatility", "NORMAL"),
            "direction": direction
        }
    }

# ===================== GESTÃO =====================
def calcular_stake(iq: IQ_Option, base: float) -> float:
    if not USE_DYNAMIC_STAKE:
        return float(max(VALOR_MINIMO, base))
    try:
        saldo = float(iq.get_balance())
        stake = (saldo * PERCENT_BANCA) / 100.0
        return float(max(VALOR_MINIMO, stake))
    except:
        return float(max(VALOR_MINIMO, base))

def verificar_meta(saldo_inicial: float, saldo_atual: float) -> Tuple[bool, float]:
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
    
    try:
        ok, op_id = safe_call(iq, iq.buy, valor, ativo, d, int(EXP_FIXA))
        if ok and op_id:
            log.info(paint(f"[{ativo}] ✅ ORDEM {direcao} (turbo) stake={valor:.2f}", C.G if direcao == "CALL" else C.R))
            return ("turbo", int(op_id))
    except Exception as e:
        log.warning(f"Turbo falhou: {e}")
    
    try:
        ok, op_id = safe_call(iq, iq.buy_digital_spot, ativo, valor, d, int(EXP_FIXA))
        if ok and op_id:
            log.info(paint(f"[{ativo}] ✅ ORDEM {direcao} (digital) stake={valor:.2f}", C.G if direcao == "CALL" else C.R))
            return ("digital", int(op_id))
    except Exception as e:
        log.warning(f"Digital falhou: {e}")
    
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
        except:
            pass
        time.sleep(0.5)

# ===================== MAIN =====================
def main():
    global last_reconnect_time
    
    iq: Optional[IQ_Option] = None
    iq = ensure_connected(iq)
    
    log.info(paint("=" * 60, C.C))
    log.info(paint("🧠 WS_HYBRID_AI - Sistema Híbrido Avançado 2026", C.C))
    log.info(paint("=" * 60, C.C))
    log.info(paint("✅ Estratégia Pernada B (Impulso + Pullback + Rompimento)", C.G))
    log.info(paint("✅ Detecção LTA/LTB (Linhas de Tendência)", C.G))
    log.info(paint("✅ Filtro S/R (Suporte/Resistência)", C.G))
    log.info(paint("✅ IA Neural (TensorFlow) - Aprende padrões profundos", C.G))
    log.info(paint("✅ IA Bayesiana (Bayes + UCB) - Aprende com resultados", C.G))
    log.info(paint("✅ Memória de Mercado - Bloqueia contextos perdedores", C.G))
    log.info(paint("=" * 60, C.C))
    
    # Carrega dados
    if NEURAL_ON:
        neural_net.load(NEURAL_FILE)
    if BAYES_ON:
        bayes_load()
    memory_load()
    
    try:
        saldo_inicial = float(iq.get_balance())
        log.info(paint(f"💰 SALDO INICIAL: {saldo_inicial:.2f} | META: {META_LUCRO_PERCENT:.1f}% | STOP: {STOP_LOSS_PERCENT:.1f}%", C.G))
    except:
        saldo_inicial = 10000.0
    
    total = 0
    wins = 0
    
    while True:
        iq = ensure_connected(iq)
        
        # Verifica meta
        try:
            saldo_atual = float(iq.get_balance())
            deve_parar, lucro_percent = verificar_meta(saldo_inicial, saldo_atual)
            if deve_parar:
                lucro_abs = saldo_atual - saldo_inicial
                if lucro_percent >= META_LUCRO_PERCENT:
                    log.info(paint(f"🎯 META ATINGIDA! +{lucro_abs:.2f} ({lucro_percent:.2f}%)", C.G))
                else:
                    log.info(paint(f"🛑 STOP LOSS! {lucro_abs:.2f} ({lucro_percent:.2f}%)", C.R))
                
                if NEURAL_ON:
                    neural_net.save(NEURAL_FILE)
                if BAYES_ON:
                    bayes_save()
                break
        except:
            pass
        
        # Proteção pós-reconexão
        if last_reconnect_time > 0 and (time.time() - last_reconnect_time) < 60:
            remaining = 60 - (time.time() - last_reconnect_time)
            log.info(paint(f"⏳ Aguardando estabilização após reconexão ({remaining:.0f}s)...", C.Y))
            time.sleep(10)
            continue
        
        ativos = obter_top_ativos_otc(iq)
        if not ativos:
            log.warning("Sem ativos disponíveis...")
            time.sleep(10)
            continue
        
        # Espera fim da vela
        wait_until_minus(TF_M1, DECIDIR_ANTES_FECHAR_SEC)
        
        best_setup = None
        best_score = 0.0
        best_ativo = None
        
        for ativo in ativos:
            if ativo in cooldown and (time.time() - cooldown[ativo]) < COOLDOWN_ATIVO:
                continue
            
            df = get_candles_df(iq, ativo, TF_M1, N_M1, end_ts=end_ts_closed(TF_M1))
            if df is None:
                continue
            
            decision = should_enter_trade(df, ativo)
            
            if decision.get("enter", False):
                if decision["score"] > best_score:
                    best_score = decision["score"]
                    best_setup = decision
                    best_ativo = ativo
                    
                    if best_score >= 0.70:
                        log.info(paint(f"🎯 Setup muito forte! Score: {best_score:.2f}", C.G))
                        break
        
        if best_setup is None:
            log.info(paint("[SKIP] Nenhum setup aprovado neste ciclo", C.Y))
            wait_for_next_open(TF_M1)
            continue
        
        ativo = best_ativo
        direction = best_setup["direction"]
        setup = best_setup.get("setup", {})
        context = best_setup.get("context", {})
        
        # Aguarda virada da vela
        wait_for_next_open(TF_M1)
        
        stake = calcular_stake(iq, STAKE_FIXA)
        op = enviar_ordem(iq, ativo, direction, stake)
        
        if not op:
            log.error(paint(f"[{ativo}] ❌ Falha ao enviar ordem", C.R))
            cooldown[ativo] = time.time()
            continue
        
        op_type, op_id = op
        res = wait_result(iq, op_type, op_id)
        
        total += 1
        is_win = res > 0
        
        if is_win:
            wins += 1
            log.info(paint(f"[{ativo}] ✅ WIN +{res:.2f}$", C.G))
        elif res < 0:
            log.info(paint(f"[{ativo}] ❌ LOSS {res:.2f}$", C.R))
        else:
            log.info(paint(f"[{ativo}] ⚪ EMPATE", C.B))
        
        # Atualiza IAs
        if NEURAL_ON and "features" in setup:
            features = np.array(setup["features"])
            neural_net.train_on_result(features, is_win)
        
        if BAYES_ON:
            bayes_update(setup, res)
            bayes_save()
        
        # Salva na memória
        memory_add(context, is_win)
        
        # Checkpoint
        if total % 10 == 0:
            if NEURAL_ON:
                neural_net.save(NEURAL_FILE)
            log.info(paint(f"💾 Checkpoint: modelo salvo ({total} trades)", C.C))
        
        acc = (wins / max(1, total)) * 100.0
        
        # Status
        try:
            saldo_atual = float(iq.get_balance())
            lucro = saldo_atual - saldo_inicial
            lucro_pct = (lucro / saldo_inicial) * 100
            log.info(paint(f"📊 TRADES={total} WINS={wins} ACC={acc:.1f}% | SALDO={saldo_atual:.2f} ({lucro:+.2f} / {lucro_pct:+.1f}%)\n", 
                           C.G if lucro > 0 else C.R))
        except:
            log.info(paint(f"📊 TRADES={total} WINS={wins} ACC={acc:.1f}%\n", C.B))
        
        cooldown[ativo] = time.time()

if __name__ == "__main__":
    main()
