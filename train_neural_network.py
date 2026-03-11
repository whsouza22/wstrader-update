# -*- coding: utf-8 -*-
"""
🧠 TREINO DAS 3 REDES NEURAIS — POR ATIVO (Double Top / Double Bottom)
========================================================================
Treina 1 modelo separado por ativo (3 NNs cada):
  IA1 (XGBoost) + IA2 (LightGBM) + IA3 (MLP 128→64→32)

40 features por padrão: geometria (f0-f25) + contexto/regime (f26-f39).
A IA aprende sozinha o que importa — sem filtros hardcoded.

Cada ativo recebe SOMENTE seus próprios padrões DT — sem misturar.
Padrões detectados via CSVs históricos (backtest) — NUNCA trades online.

Modelo salvo: ~/.wstrader/reversal_tf_{ATIVO}.pkl (1 arquivo por ativo)

Uso:
    python train_neural_network.py
    python train_neural_network.py --assets NZDJPY-OTC,GBPAUD-OTC
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ws_reversal_ai import ReversalAI, DT_FEATURE_NAMES, MIN_SAMPLES_ML, AI3_MIN_SAMPLES

# Importar funções de detecção do bot principal
from WS_AUTO_AI_BULLEX import (
    detect_pivots,
    detect_all_hs,
    detect_double_touch,
    backtest_pattern,
    _extract_geometry,
    EXP_FIXA,
)
from ws_adaptive_brain import extract_features

# ══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════
_BASE = os.path.dirname(os.path.abspath(__file__))
CSV_DIRS = [
    os.path.join(_BASE, "candles_5000"),
    os.path.join(_BASE, "candles_deep"),
]

# Ativos padrão para treinar (os 4 melhores da análise)
DEFAULT_ASSETS = ["NZDJPY-OTC", "GBPAUD-OTC", "USDCAD-OTC", "EURNZD-OTC"]


def print_header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def print_status(msg):
    print(f"  {msg}")


def load_csv(path):
    """Carrega CSV de velas e retorna DataFrame formatado."""
    try:
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        needed = ["open", "high", "low", "close"]
        for col in needed:
            if col not in df.columns:
                return None
        df = df[needed].dropna().sort_index()
        return df if len(df) >= 100 else None
    except Exception as e:
        print(f"  ⚠️ Erro lendo {path}: {e}")
        return None


def train_single_asset(ativo, csv_paths):
    """Treina 3 NNs para UM ativo usando SOMENTE dados desse ativo."""
    # Combinar CSVs
    frames = []
    for p in csv_paths:
        df_part = load_csv(p)
        if df_part is not None:
            frames.append(df_part)

    if not frames:
        return None

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep='first')]

    H = df["high"].values
    L = df["low"].values
    C = df["close"].values
    O = df["open"].values
    n = len(H)

    if n < 100:
        return None

    # ATR
    atr_vals = [float(H[k] - L[k]) for k in range(max(0, n - 14), n)]
    atr = float(np.mean(atr_vals)) if atr_vals else 0.001
    if atr <= 0:
        return None

    # Detectar pivots e DTs — SOMENTE double_touch (igual ao live)
    ph, pl = detect_pivots(H, L, window=5)
    all_dt = detect_double_touch(H, L, C, O, ph, pl, atr, n,
                                 max_candles_ago=9999, training=True)

    if not all_dt:
        return None

    # Criar ReversalAI LIMPA para este ativo (sem dados de outros)
    ai = ReversalAI(ativo)
    ai._ai1 = None
    ai._ai2 = None
    ai._ai3 = None
    ai._ai1_ready = False
    ai._ai2_ready = False
    ai._ai3_ready = False
    ai._train_data = []

    # Stats para arm_wr (feature f11) — SOMENTE deste ativo
    hs_stats = {"meta": {"total": 0, "wins": 0}, "arms": {}}

    # FASE 1: Coletar features — primeiro pass para arm_wr stats
    for pat in all_dt:
        bt = backtest_pattern(pat, C, O, H, L, n)
        if bt is None or bt["result"] not in ("win", "loss"):
            continue
        result = 1 if bt["result"] == "win" else 0
        pat_type = pat.get("type", "HS")
        mode = pat.get("mode", "classic")
        arm = f"{ativo}_{pat_type}_{mode}"
        if arm not in hs_stats["arms"]:
            hs_stats["arms"][arm] = {"wins": 0, "total": 0}
        hs_stats["arms"][arm]["total"] += 1
        if result:
            hs_stats["arms"][arm]["wins"] += 1

    # FASE 2: Extrair features e alimentar IA
    _w, _l, _added = 0, 0, 0
    for pat in all_dt:
        bt = backtest_pattern(pat, C, O, H, L, n)
        if bt is None or bt["result"] not in ("win", "loss"):
            continue

        result = 1 if bt["result"] == "win" else 0

        rs_idx = pat["right_shoulder"]["idx"]
        win_start = max(0, rs_idx - 110)
        win_end = min(n, rs_idx + 1)
        H_win = H[win_start:win_end]
        L_win = L[win_start:win_end]
        C_win = C[win_start:win_end]
        O_win = O[win_start:win_end]
        n_win = len(H_win)

        atr_local_vals = [float(H_win[k] - L_win[k]) for k in range(max(0, n_win - 14), n_win)]
        atr_local = float(np.mean(atr_local_vals)) if atr_local_vals else atr

        pat_copy = dict(pat)
        pat_copy["candles_ago"] = max(0, n_win - 1 - (rs_idx - win_start))

        feats = extract_features(pat_copy, H_win, L_win, C_win, O_win, n_win,
                                 atr_local, hs_stats, ativo)
        if feats is not None:
            ok = ai.feed_dt_features(feats, result)
            if ok:
                _added += 1
                if result == 1:
                    _w += 1
                else:
                    _l += 1

    if _added < MIN_SAMPLES_ML:
        print_status(f"  ⚠️ {ativo}: poucos padrões ({_added} < {MIN_SAMPLES_ML}) — pulando")
        return None

    # FASE 3: Treinar
    ok = ai.train_all(force=True)
    if not ok:
        print_status(f"  ❌ {ativo}: treino falhou")
        return None

    wr = _w / _added * 100 if _added > 0 else 0
    return {
        "ativo": ativo,
        "ai": ai,
        "patterns": _added,
        "wins": _w,
        "losses": _l,
        "wr": wr,
        "candles": n,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Treina 3 NNs POR ATIVO com padrões DT dos CSVs históricos")
    parser.add_argument("--assets", type=str, default=None,
                        help="Ativos separados por vírgula (ex: NZDJPY-OTC,GBPAUD-OTC)")
    args = parser.parse_args()

    target_assets = args.assets.split(",") if args.assets else None

    print_header("🧠 TREINO POR ATIVO — 3 NNs separadas por ativo")

    # Coletar CSVs
    csv_map = {}
    for csv_dir in CSV_DIRS:
        if not os.path.isdir(csv_dir):
            continue
        for f in sorted(os.listdir(csv_dir)):
            if f.endswith(".csv"):
                ativo = f.replace(".csv", "")
                path = os.path.join(csv_dir, f)
                if ativo not in csv_map:
                    csv_map[ativo] = [path]
                else:
                    csv_map[ativo].append(path)

    if not csv_map:
        print(f"  ❌ Nenhum CSV encontrado")
        sys.exit(1)

    # Se nenhum ativo especificado, treinar TODOS com CSV disponível
    if target_assets is None:
        target_assets = sorted(csv_map.keys())

    print_status(f"🎯 Ativos: {len(target_assets)} ({', '.join(target_assets[:5])}{'...' if len(target_assets) > 5 else ''})")
    print()

    # Verificar quais ativos têm CSV
    missing = [a for a in target_assets if a not in csv_map]
    if missing:
        print_status(f"⚠️ Sem CSV para: {', '.join(missing)}")

    # ══════════════════════════════════════════════════════════
    # TREINAR CADA ATIVO SEPARADAMENTE
    # ══════════════════════════════════════════════════════════
    t0 = time.time()
    results = []

    for ativo in target_assets:
        if ativo not in csv_map:
            continue

        print_header(f"🏋️ Treinando: {ativo}")
        t_asset = time.time()

        result = train_single_asset(ativo, csv_map[ativo])

        if result is not None:
            ai = result["ai"]
            elapsed_a = time.time() - t_asset
            print_status(f"  ✅ {ativo}: {result['patterns']} padrões "
                         f"({result['wins']}W/{result['losses']}L = {result['wr']:.1f}%)")
            print_status(f"     IA1={ai._ai1_val:.1%} | IA2={ai._ai2_val:.1%}"
                         + (f" | IA3={ai._ai3_val:.1%}" if ai._ai3_ready else ""))
            print_status(f"     Velas: {result['candles']:,} | Tempo: {elapsed_a:.1f}s")

            # Verificar modelo salvo
            model_path = os.path.join(os.path.expanduser("~"), ".wstrader",
                                      f"reversal_tf_{ativo}.pkl")
            if os.path.exists(model_path):
                size_kb = os.path.getsize(model_path) / 1024
                print_status(f"     💾 Salvo: {model_path} ({size_kb:.0f} KB)")

            results.append(result)
        else:
            print_status(f"  ❌ {ativo}: falhou ou poucos dados")

    # ══════════════════════════════════════════════════════════
    # RESUMO FINAL
    # ══════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print_header("📊 RESUMO DO TREINO POR ATIVO")

    if not results:
        print(f"\n  ❌ Nenhum ativo treinado com sucesso")
        sys.exit(1)

    total_patterns = 0
    for r in results:
        ai = r["ai"]
        ia3_str = f" IA3={ai._ai3_val:.1%}" if ai._ai3_ready else ""
        print_status(f"  {r['ativo']:18s} | {r['patterns']:5d} padrões | "
                     f"WR={r['wr']:5.1f}% | IA1={ai._ai1_val:.1%} IA2={ai._ai2_val:.1%}{ia3_str}")
        total_patterns += r["patterns"]

    print()
    print_status(f"📊 Total: {total_patterns} padrões em {len(results)} ativos")
    print_status(f"⏱️ Tempo total: {elapsed:.1f}s")
    print_status(f"💾 Modelos em: ~/.wstrader/reversal_tf_{{ATIVO}}.pkl")

    print_header("✅ TREINO POR ATIVO CONCLUÍDO")
    print()
    print("  Cada ativo agora tem seu próprio modelo NN independente.")
    print("  O bot carrega automaticamente o modelo do ativo que está operando.")
    print()


if __name__ == "__main__":
    main()
