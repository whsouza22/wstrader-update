# -*- coding: utf-8 -*-
"""
🔄 RETRAIN & PUSH — Automatiza retreino semanal + upload para GitHub
=====================================================================
Fluxo completo em 1 comando:
  1. Treina todos os 37 modelos NN com CSVs atualizados
  2. Copia modelos para pasta models/ do repositório
  3. Faz git add + commit + push automaticamente

Os usuários recebem os modelos novos automaticamente (download do GitHub).

Uso:
    python retrain_and_push.py              # treina tudo e faz push
    python retrain_and_push.py --no-push    # treina mas NÃO faz push (revisar antes)
    python retrain_and_push.py --assets NZDJPY-OTC,GBPAUD-OTC  # só alguns ativos
"""

import os
import sys
import glob
import shutil
import subprocess
import time
import argparse
from datetime import datetime


def print_header(msg):
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def print_status(msg):
    print(f"    {msg}")


def run_cmd(cmd, cwd=None, check=True):
    """Executa comando e retorna stdout."""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if check and result.returncode != 0:
        print(f"  ❌ Erro ao executar: {cmd}")
        print(f"     {result.stderr[:500]}")
        return None
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description="Retrain NN models and push to GitHub")
    parser.add_argument("--no-push", action="store_true", help="Treinar sem fazer git push")
    parser.add_argument("--assets", type=str, default=None,
                        help="Ativos específicos (separados por vírgula)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    user_models_dir = os.path.join(os.path.expanduser("~"), ".wstrader")

    os.makedirs(models_dir, exist_ok=True)

    print_header("🔄 RETRAIN & PUSH — Retreino Semanal Automatizado")
    print_status(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_status(f"📂 Repo: {script_dir}")
    print_status(f"📂 Modelos: {user_models_dir}")

    # ══════════════════════════════════════════════════════════
    # FASE 1: TREINAR MODELOS
    # ══════════════════════════════════════════════════════════
    print_header("🧠 FASE 1: Treinando modelos NN")

    train_cmd = [sys.executable, os.path.join(script_dir, "train_neural_network.py")]
    if args.assets:
        train_cmd += ["--assets", args.assets]

    t0 = time.time()
    result = subprocess.run(
        train_cmd, cwd=script_dir,
        text=True, encoding="utf-8", errors="replace"
    )

    if result.returncode != 0:
        print(f"\n  ❌ Treino falhou (exit code {result.returncode})")
        sys.exit(1)

    train_time = time.time() - t0
    print_status(f"⏱️ Treino concluído em {train_time:.0f}s")

    # ══════════════════════════════════════════════════════════
    # FASE 2: COPIAR MODELOS PARA PASTA DO GIT
    # ══════════════════════════════════════════════════════════
    print_header("📦 FASE 2: Copiando modelos para models/")

    src_pattern = os.path.join(user_models_dir, "reversal_tf_*-OTC.pkl")
    src_files = sorted(glob.glob(src_pattern))

    if not src_files:
        print(f"  ❌ Nenhum modelo encontrado em {user_models_dir}")
        sys.exit(1)

    copied = 0
    total_size = 0
    for src in src_files:
        fname = os.path.basename(src)
        dst = os.path.join(models_dir, fname)
        shutil.copy2(src, dst)
        size_kb = os.path.getsize(dst) / 1024
        total_size += size_kb
        copied += 1

    print_status(f"✅ {copied} modelos copiados ({total_size / 1024:.1f} MB total)")

    # ══════════════════════════════════════════════════════════
    # FASE 3: GIT ADD + COMMIT + PUSH
    # ══════════════════════════════════════════════════════════
    if args.no_push:
        print_header("⏸️ FASE 3: --no-push ativo, pulando git push")
        print_status("Modelos atualizados localmente. Faça git push manualmente.")
        print_status(f"  cd \"{script_dir}\"")
        print_status(f"  git add models/ ws_adaptive_brain.py ws_reversal_ai.py")
        print_status(f"  git commit -m \"retrain: modelos NN atualizados\"")
        print_status(f"  git push origin main")
    else:
        print_header("🚀 FASE 3: Git push para GitHub")

        # Verificar se tem mudanças
        status = run_cmd("git status --porcelain models/", cwd=script_dir)
        if not status:
            print_status("ℹ️ Nenhuma mudança nos modelos — nada a commitar")
        else:
            changed_count = len([l for l in status.splitlines() if l.strip()])
            print_status(f"📝 {changed_count} arquivos modificados")

            # Stage
            run_cmd("git add models/", cwd=script_dir)

            # Also stage core files if modified
            for f in ["ws_adaptive_brain.py", "ws_reversal_ai.py", "WS_AUTO_AI_BULLEX.py"]:
                fpath = os.path.join(script_dir, f)
                st = run_cmd(f"git status --porcelain \"{f}\"", cwd=script_dir, check=False)
                if st and st.strip():
                    run_cmd(f"git add -f \"{f}\"", cwd=script_dir)

            # Commit
            date_str = datetime.now().strftime("%Y-%m-%d")
            commit_msg = f"retrain: {copied} modelos NN atualizados ({date_str})"
            run_cmd(f'git commit -m "{commit_msg}"', cwd=script_dir)
            print_status(f"✅ Commit: {commit_msg}")

            # Push
            push_result = run_cmd("git push origin main", cwd=script_dir)
            if push_result is not None:
                print_status("✅ Push concluído — usuários receberão modelos novos automaticamente")
            else:
                print_status("❌ Push falhou — verifique conexão/credenciais e tente git push manualmente")

    # ══════════════════════════════════════════════════════════
    # RESUMO FINAL
    # ══════════════════════════════════════════════════════════
    print_header("✅ RETRAIN & PUSH CONCLUÍDO")
    print_status(f"🧠 {copied} modelos retreinados")
    print_status(f"⏱️ Tempo total: {time.time() - t0:.0f}s")
    if not args.no_push:
        print_status("🌐 Usuários receberão os modelos novos ao iniciar o bot")
    print()


if __name__ == "__main__":
    main()
