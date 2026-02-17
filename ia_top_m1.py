"""
IA M1 com indicador TOP + confirmacao + treinamento online.
- Usa candles fechados (remove vela em formacao)
- Arma sinal no fechamento e executa na abertura da proxima vela
- Treina IA com features amarradas ao id da operacao

Requisitos:
  pip install iqoptionapi pandas numpy scikit-learn
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import threading
import queue
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures
import logging

from iqoptionapi.stable_api import IQ_Option
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =========================
# CONFIG
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

TIMEFRAME = 60
NUM_VELAS = 300
PERCENTUAL_ENTRADA = 0.02
MIN_BET = 3
PAYOUT_MINIMO = 80
MAX_RECONEXOES = 3
EXPIRACAO_FIXA = 1

# Analisa no fechamento e executa na abertura da proxima
CALCULO_ANTECIPACAO = 5
ANALYSIS_INTERVAL = 60.0

# IA
PROBABILITY_THRESHOLD = 0.70
INITIAL_TRADES = 5
INITIAL_THRESHOLD = 0.55

# =========================
# ESTADO GLOBAL
# =========================
model = LogisticRegression()
scaler = StandardScaler()
X_train, y_train = [], []
is_model_trained = False
trade_count = 0

active_orders = 0
pending_signal = {"asset": None, "side": None, "armed_ts": 0.0, "expires_in_sec": 80}
features_by_trade = {}
last_analysis_time_by_asset = {}

# =========================
# IQ Helpers
# =========================
def conectar_iq_option(email, senha):
    logging.info("Conectando a IQ Option...")
    iq = IQ_Option(email, senha)
    iq.connect()

    for tentativa in range(MAX_RECONEXOES):
        if iq.check_connect():
            logging.info("Conectado")
            break
        logging.info(f"Tentativa {tentativa + 1}/{MAX_RECONEXOES}...")
        time.sleep(5)
    else:
        logging.error("Falha na conexao")
        return None

    iq.change_balance("PRACTICE")
    try:
        logging.info(f"Saldo: {iq.get_balance()}")
    except Exception:
        pass
    return iq


def get_candles(iq, ativo, timeframe, num_candles, max_attempts=5):
    now = time.time()
    for attempt in range(max_attempts):
        try:
            candles = iq.get_candles(ativo, timeframe, num_candles, now)
            if candles is None or candles == [] or isinstance(candles, int):
                logging.warning(f"[{ativo}] Falha get_candles (tentativa {attempt + 1}/{max_attempts})")
                time.sleep(1.5)
                continue

            df = pd.DataFrame(candles)
            df.rename(columns={"from": "time", "max": "high", "min": "low"}, inplace=True)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("UTC")
            df.set_index("time", inplace=True)
            return df[["open", "high", "low", "close"]].copy()
        except Exception as e:
            logging.error(f"[{ativo}] Excecao get_candles: {e}")
            time.sleep(1.5)

    return None


def get_closed_candles(iq, ativo, timeframe, num_candles=NUM_VELAS + 3):
    df = get_candles(iq, ativo, timeframe, num_candles=num_candles)
    if df is None or len(df) < 50:
        return None

    now = datetime.utcnow().replace(tzinfo=None)
    last_time = df.index[-1].to_pydatetime().replace(tzinfo=None)
    if (now - last_time).total_seconds() < timeframe:
        df = df.iloc[:-1]

    return df


def wait_until(seconds_before_close=CALCULO_ANTECIPACAO):
    while True:
        agora = datetime.utcnow()
        fim_candle = agora - timedelta(seconds=agora.second % TIMEFRAME, microseconds=agora.microsecond) + timedelta(seconds=TIMEFRAME)
        t = (fim_candle - agora).total_seconds()
        if t > seconds_before_close:
            time.sleep(max(0.05, t - seconds_before_close))
            continue
        return fim_candle


def wait_open_next_candle(fim_candle):
    while True:
        agora = datetime.utcnow()
        t = (fim_candle - agora).total_seconds()
        if t > 0.02:
            time.sleep(max(0.01, t - 0.02))
            continue
        return

# =========================
# IA
# =========================
def update_model():
    global model, scaler, X_train, y_train, is_model_trained

    if len(X_train) < 20:
        return

    unique = np.unique(y_train)
    if len(unique) < 2:
        return

    X_scaled = scaler.fit_transform(X_train)
    model.fit(X_scaled, y_train)
    is_model_trained = True


def extract_features(df):
    current_price = df["close"].iloc[-1]
    current_range = float(df["high"].iloc[-1] - df["low"].iloc[-1])

    roll_hi = df["high"].rolling(window=20).max()
    roll_lo = df["low"].rolling(window=20).min()
    avg_range = float((roll_hi - roll_lo).iloc[-1]) if not np.isnan((roll_hi - roll_lo).iloc[-1]) else 0.01
    if avg_range <= 0:
        avg_range = 0.01
    relative_strength = current_range / avg_range

    sma_fast = df["close"].rolling(window=1).mean()
    sma_slow = df["close"].rolling(window=34).mean()
    buffer1 = sma_fast - sma_slow
    buffer2 = buffer1.rolling(window=5).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=False
    ).iloc[-1]

    smaa = df["close"].rolling(window=20).mean().iloc[-1]
    stdev = df["close"].rolling(window=20).std().iloc[-1]
    stdev = 0.0001 if np.isnan(stdev) or stdev == 0 else float(stdev)
    upper_band = float(smaa + (stdev * 2.5))
    lower_band = float(smaa - (stdev * 2.5))
    emaa = float(df["close"].ewm(span=100, adjust=False).mean().iloc[-1])

    above_upper = 1 if current_price > upper_band else 0
    below_lower = 1 if current_price < lower_band else 0
    ema_trend = 1 if current_price > emaa else 0

    return [
        float(relative_strength),
        float(buffer1.iloc[-1] - buffer2),
        int(above_upper),
        int(below_lower),
        int(ema_trend),
    ]


def predict_win_probability(features):
    X = np.array(features, dtype=float).reshape(1, -1)
    if not is_model_trained:
        return 0.50
    Xs = scaler.transform(X)
    return float(model.predict_proba(Xs)[:, 1][0])

# =========================
# TOP com confirmacao
# =========================
def top_signal_confirmed(df):
    if len(df) < 120:
        return False, False, False, False, {}

    smaa = df["close"].rolling(window=20).mean().iloc[-1]
    stdev = df["close"].rolling(window=20).std().iloc[-1]
    stdev = 0.0001 if np.isnan(stdev) or stdev == 0 else float(stdev)
    upper_band = float(smaa + (stdev * 2.5))
    lower_band = float(smaa - (stdev * 2.5))

    current_price = float(df["close"].iloc[-1])
    sobrecompra = current_price > upper_band
    sobrevenda = current_price < lower_band

    sma_fast = df["close"].rolling(window=1).mean()
    sma_slow = df["close"].rolling(window=34).mean()
    buffer1 = sma_fast - sma_slow
    buffer2 = buffer1.rolling(window=5).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=False
    )

    cross_up = (buffer1.iloc[-2] > buffer2.iloc[-2]) and (buffer1.iloc[-3] < buffer2.iloc[-3])
    cross_dn = (buffer1.iloc[-2] < buffer2.iloc[-2]) and (buffer1.iloc[-3] > buffer2.iloc[-3])

    persist_up = (buffer1.iloc[-1] > buffer2.iloc[-1])
    persist_dn = (buffer1.iloc[-1] < buffer2.iloc[-1])

    bull_confirm = df["close"].iloc[-1] > df["open"].iloc[-1]
    bear_confirm = df["close"].iloc[-1] < df["open"].iloc[-1]

    buy_confirmed = bool(cross_up and persist_up and bull_confirm and (not sobrecompra))
    sell_confirmed = bool(cross_dn and persist_dn and bear_confirm and (not sobrevenda))

    dbg = {
        "b1_last": float(buffer1.iloc[-1]),
        "b2_last": float(buffer2.iloc[-1]),
        "upper": upper_band,
        "lower": lower_band,
        "price": current_price,
        "cross_up": bool(cross_up),
        "cross_dn": bool(cross_dn),
        "persist_up": bool(persist_up),
        "persist_dn": bool(persist_dn),
        "bull_confirm": bool(bull_confirm),
        "bear_confirm": bool(bear_confirm),
    }
    return buy_confirmed, sell_confirmed, sobrecompra, sobrevenda, dbg


def current_threshold():
    return INITIAL_THRESHOLD if trade_count < INITIAL_TRADES else PROBABILITY_THRESHOLD


def arm_signal(asset, side):
    pending_signal["asset"] = asset
    pending_signal["side"] = side
    pending_signal["armed_ts"] = datetime.utcnow().timestamp()


def consume_signal_if_valid(asset):
    if pending_signal["asset"] != asset:
        return None

    age = datetime.utcnow().timestamp() - float(pending_signal["armed_ts"])
    if age > float(pending_signal["expires_in_sec"]):
        pending_signal["asset"] = None
        pending_signal["side"] = None
        return None

    side = pending_signal["side"]
    pending_signal["asset"] = None
    pending_signal["side"] = None
    return side

# =========================
# Execucao de trade
# =========================
def execute_trade(iq, ativo, side, valor, expiracao):
    global active_orders
    try:
        if not iq.check_connect():
            iq.connect()

        status, id_op = iq.buy(valor, ativo, side, expiracao)
        if not status:
            logging.error(f"[{ativo}] Falha buy() (status={status}, id={id_op})")
            return None, None, None

        active_orders += 1
        result_queue = queue.Queue()

        def check_result():
            global active_orders
            while True:
                time.sleep(0.1)
                ok, resultado = iq.check_win_v4(id_op)
                if ok:
                    result_queue.put(resultado)
                    active_orders -= 1
                    break

        t = threading.Thread(target=check_result, daemon=True)
        t.start()
        t.join(timeout=expiracao * 60 + 12)

        if result_queue.empty():
            logging.error(f"[{ativo}] Timeout check_win_v4 id={id_op}")
            active_orders -= 1
            return None, None, id_op

        resultado = float(result_queue.get())
        success = resultado > 0
        return success, resultado, id_op

    except Exception as e:
        logging.error(f"[{ativo}] Excecao execute_trade: {e}")
        return None, None, None


def train_from_trade(id_op, resultado):
    global trade_count
    feat = features_by_trade.pop(id_op, None)
    if feat is None:
        return
    if resultado == 0:
        return

    X_train.append(feat)
    y_train.append(1 if resultado > 0 else 0)
    update_model()
    trade_count += 1

# =========================
# Selecao de ativos
# =========================
def obter_top_40_ativos_turbo(iq):
    try:
        dados = iq.get_all_open_time()
        turbo = dados.get("turbo", {})
    except Exception as e:
        logging.error(f"Falha get_all_open_time: {e}")
        return []

    abertos = [a for a, info in turbo.items() if info.get("open")]
    if not abertos:
        return []

    filtrados = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(iq.get_digital_payout, a): a for a in abertos}
        for fut in concurrent.futures.as_completed(futures):
            a = futures[fut]
            try:
                payout = float(fut.result())
            except Exception:
                payout = 0.0
            if payout == 0:
                payout = PAYOUT_MINIMO
            if payout >= PAYOUT_MINIMO:
                filtrados.append((a, payout))

    filtrados.sort(key=lambda x: x[1], reverse=True)
    return [a for a, _ in filtrados][:40]

# =========================
# Loop por ativo
# =========================
def analyze_and_arm(iq, ativo):
    now_ts = datetime.utcnow().timestamp()
    last_ts = last_analysis_time_by_asset.get(ativo, 0.0)
    if (now_ts - last_ts) < ANALYSIS_INTERVAL:
        return
    last_analysis_time_by_asset[ativo] = now_ts

    df = get_closed_candles(iq, ativo, TIMEFRAME, num_candles=NUM_VELAS + 3)
    if df is None or len(df) < NUM_VELAS:
        logging.info(f"[{ativo}] DF insuficiente")
        return

    buy_ok, sell_ok, _sc, _sv, dbg = top_signal_confirmed(df)

    features = extract_features(df)
    prob = predict_win_probability(features)
    thr = current_threshold()

    logging.info(
        f"[{ativo}] TOP b1={dbg.get('b1_last'):.6f} b2={dbg.get('b2_last'):.6f} "
        f"cross_up={dbg.get('cross_up')} cross_dn={dbg.get('cross_dn')} "
        f"prob={prob:.2f} thr={thr:.2f}"
    )

    if buy_ok and prob >= thr:
        arm_signal(ativo, "call")
        logging.info(f"[{ativo}] SINAL ARMADO: CALL (prob={prob:.2f})")
        return

    if sell_ok and prob >= thr:
        arm_signal(ativo, "put")
        logging.info(f"[{ativo}] SINAL ARMADO: PUT (prob={prob:.2f})")
        return


def execute_if_armed(iq, ativo):
    side = consume_signal_if_valid(ativo)
    if side is None:
        return

    df = get_closed_candles(iq, ativo, TIMEFRAME, num_candles=NUM_VELAS + 3)
    if df is None or len(df) < NUM_VELAS:
        return

    buy_ok, sell_ok, _sc, _sv, _dbg = top_signal_confirmed(df)
    if side == "call" and not buy_ok:
        logging.info(f"[{ativo}] CALL armado, mas perdeu validacao. Cancelado.")
        return
    if side == "put" and not sell_ok:
        logging.info(f"[{ativo}] PUT armado, mas perdeu validacao. Cancelado.")
        return

    features = extract_features(df)
    prob = predict_win_probability(features)
    thr = current_threshold()
    if prob < thr:
        logging.info(f"[{ativo}] Sinal armado, mas prob caiu ({prob:.2f} < {thr:.2f}). Cancelado.")
        return

    saldo = float(iq.get_balance())
    valor = max(saldo * PERCENTUAL_ENTRADA, MIN_BET)

    logging.info(f"[{ativo}] EXECUTANDO {side.upper()} valor={valor:.2f} exp={EXPIRACAO_FIXA}")

    success, resultado, id_op = execute_trade(iq, ativo, side, valor, EXPIRACAO_FIXA)
    if id_op is not None:
        features_by_trade[id_op] = features

    if success is None:
        return

    if resultado > 0:
        logging.info(f"[{ativo}] WIN {resultado:.2f}")
    elif resultado == 0:
        logging.info(f"[{ativo}] EMPATE {resultado:.2f}")
    else:
        logging.info(f"[{ativo}] LOSS {resultado:.2f}")

    if id_op is not None:
        train_from_trade(id_op, resultado)


def run_asset_cycle(iq, ativos):
    idx = 0
    while True:
        if not ativos:
            logging.info("Sem ativos. Recarregando...")
            ativos = obter_top_40_ativos_turbo(iq)
            time.sleep(2)
            continue

        ativo = ativos[idx % len(ativos)]
        idx += 1

        fim_candle = wait_until(seconds_before_close=CALCULO_ANTECIPACAO)
        analyze_and_arm(iq, ativo)
        wait_open_next_candle(fim_candle)
        execute_if_armed(iq, ativo)

        if idx % max(1, len(ativos)) == 0:
            logging.info("Ciclo completo. Atualizando ativos...")
            ativos = obter_top_40_ativos_turbo(iq)
            time.sleep(1)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    EMAIL = os.getenv("IQ_EMAIL", "")
    SENHA = os.getenv("IQ_SENHA", "")

    if not EMAIL or not SENHA:
        raise RuntimeError("Defina IQ_EMAIL e IQ_SENHA nas variaveis de ambiente.")

    iq = conectar_iq_option(EMAIL, SENHA)
    if not iq:
        raise SystemExit("Falha na conexao com a IQ Option.")

    ativos = obter_top_40_ativos_turbo(iq)
    logging.info(f"Ativos monitorados: {ativos[:10]}... (total={len(ativos)})")

    run_asset_cycle(iq, ativos)
