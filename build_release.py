# -*- coding: utf-8 -*-
"""
🚀 BUILD & RELEASE — Automatiza todo o processo de build e release
===================================================================
Fluxo completo em 1 comando:
  1. Atualiza versão (version.json, installer.nsi, version_info.txt, WS Trader.spec)
  2. Criptografa os .py com PyArmor → protected_build/
  3. Compila com PyInstaller → dist/WsTrader.exe
  4. Gera instalador NSIS → WsTrader_Setup_X.X.exe
  5. Faz commit + push + cria release no GitHub

Uso:
    python build_release.py --version 5.6
    python build_release.py --version 5.6 --no-push       # sem push
    python build_release.py --version 5.6 --skip-pyarmor   # pula pyarmor (já protegido)
    python build_release.py --version 5.6 --skip-build     # só atualiza versão + NSIS
"""

import os
import sys
import json
import glob
import shutil
import subprocess
import argparse
import time
import re
from datetime import datetime

# ══════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROTECTED_DIR = os.path.join(PROJECT_DIR, "protected_build")
DIST_DIR = os.path.join(PROJECT_DIR, "dist")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
IMG_DIR = os.path.join(PROJECT_DIR, "Img")

MAKENSIS = r"C:\Program Files (x86)\NSIS\makensis.exe"

# Arquivos que devem ser protegidos com PyArmor
PYARMOR_FILES = [
    "TelaPrincipal.py",
    "WS_AUTO_AI_BULLEX.py",
    "ws_reversal_ai.py",
    "ws_adaptive_brain.py",
    "ws_generative_guard.py",
    "config_keys.py",
    "Login_Screen.py",
    "trading_bot.py",
    "dashboard_hs_ia.py",
    "operations_manager.py",
    "chat_screen_new.py",
    "tutorial_screen.py",
    "license_manager.py",
    "backend_server.py",
    "train_neural_network.py",
]

GITHUB_REPO = "whsouza22/wstrader-update"


def print_header(msg):
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def print_status(msg):
    print(f"    {msg}")


def run(cmd, cwd=None, check=True, shell=True):
    """Executa comando e retorna (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, shell=shell, cwd=cwd or PROJECT_DIR,
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if check and result.returncode != 0:
        print(f"  ❌ ERRO: {cmd}")
        if result.stderr:
            print(f"     {result.stderr[:800]}")
        return None
    return result.stdout.strip()


def update_version(version):
    """Atualiza a versão em todos os arquivos necessários."""
    print_header(f"📝 FASE 1: Atualizando versão para {version}")

    v_parts = version.split(".")
    v_major = int(v_parts[0])
    v_minor = int(v_parts[1]) if len(v_parts) > 1 else 0
    v_patch = int(v_parts[2]) if len(v_parts) > 2 else 0
    v_tuple = f"{v_major}, {v_minor}, {v_patch}, 0"
    v_dotted = f"{v_major}.{v_minor}.{v_patch}.0"

    # 1. version.json
    vj_path = os.path.join(PROJECT_DIR, "version.json")
    vj = {
        "version": version,
        "changelog": f"v{version} - NN v7 24 features, sem guards, IA decide tudo",
        "download_url": f"https://github.com/{GITHUB_REPO}/releases/download/v{version}/WsTrader_Setup_{version}.exe"
    }
    with open(vj_path, "w", encoding="utf-8") as f:
        json.dump(vj, f, ensure_ascii=False, indent=2)
    print_status(f"✅ version.json → {version}")

    # 2. version_info.txt
    vi_path = os.path.join(PROJECT_DIR, "version_info.txt")
    vi_content = f"""VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({v_tuple}),
    prodvers=({v_tuple}),
    mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(u'040904B0', [
        StringStruct(u'CompanyName', u'WS Trader Team'),
        StringStruct(u'FileDescription', u'WS Trader AI - Assistente Inteligente de Trading'),
        StringStruct(u'FileVersion', u'{version}'),
        StringStruct(u'InternalName', u'WsTrader'),
        StringStruct(u'LegalCopyright', u'© 2026 WS Trader Team'),
        StringStruct(u'OriginalFilename', u'WsTrader.exe'),
        StringStruct(u'ProductName', u'WS Trader AI'),
        StringStruct(u'ProductVersion', u'{version}')
      ])
    ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""
    with open(vi_path, "w", encoding="utf-8") as f:
        f.write(vi_content)
    print_status(f"✅ version_info.txt → {v_tuple}")

    # 3. installer.nsi — atualizar versão
    nsi_path = os.path.join(PROJECT_DIR, "installer.nsi")
    with open(nsi_path, "r", encoding="utf-8") as f:
        nsi = f.read()

    # OutFile
    nsi = re.sub(
        r'OutFile\s+"WsTrader_Setup_[\d.]+\.exe"',
        f'OutFile "WsTrader_Setup_{version}.exe"',
        nsi
    )
    # VIProductVersion
    nsi = re.sub(
        r'VIProductVersion\s+"[\d.]+"',
        f'VIProductVersion "{v_dotted}"',
        nsi
    )
    # FileVersion and ProductVersion in VIAddVersionKey
    nsi = re.sub(
        r'("FileVersion"\s+")([\d.]+)(")',
        f'\\g<1>{version}\\3',
        nsi
    )
    nsi = re.sub(
        r'("ProductVersion"\s+")([\d.]+)(")',
        f'\\g<1>{version}\\3',
        nsi
    )
    # DisplayVersion in registry
    nsi = re.sub(
        r'("DisplayVersion"\s+")([\d.]+)(")',
        f'\\g<1>{version}\\3',
        nsi
    )
    # WriteRegStr Version
    nsi = re.sub(
        r'("Version"\s+")([\d.]+)(")',
        f'\\g<1>{version}\\3',
        nsi
    )

    with open(nsi_path, "w", encoding="utf-8") as f:
        f.write(nsi)
    print_status(f"✅ installer.nsi → {version}")

    return True


def run_pyarmor():
    """Criptografa os arquivos Python com PyArmor."""
    print_header("🔒 FASE 2: PyArmor — Criptografando código")

    os.makedirs(PROTECTED_DIR, exist_ok=True)

    # Limpar arquivos .py antigos (manter pyarmor_runtime_009928/)
    for f in glob.glob(os.path.join(PROTECTED_DIR, "*.py")):
        os.remove(f)

    # Verificar que todos os arquivos fonte existem
    missing = []
    for f in PYARMOR_FILES:
        if not os.path.exists(os.path.join(PROJECT_DIR, f)):
            missing.append(f)
    if missing:
        print(f"  ❌ Arquivos fonte não encontrados: {', '.join(missing)}")
        return False

    # Executar PyArmor gen para todos os arquivos
    file_args = " ".join(f'"{f}"' for f in PYARMOR_FILES)
    cmd = f'pyarmor gen --output "{PROTECTED_DIR}" {file_args}'

    print_status(f"Protegendo {len(PYARMOR_FILES)} arquivos...")
    t0 = time.time()

    result = subprocess.run(
        cmd, shell=True, cwd=PROJECT_DIR,
        text=True, encoding="utf-8", errors="replace"
    )

    if result.returncode != 0:
        print(f"  ❌ PyArmor falhou (exit code {result.returncode})")
        return False

    elapsed = time.time() - t0

    # Verificar que os arquivos foram gerados
    generated = [f for f in PYARMOR_FILES if os.path.exists(os.path.join(PROTECTED_DIR, f))]
    print_status(f"✅ {len(generated)}/{len(PYARMOR_FILES)} arquivos protegidos ({elapsed:.1f}s)")

    if len(generated) < len(PYARMOR_FILES):
        failed = [f for f in PYARMOR_FILES if f not in generated]
        print_status(f"⚠️ Falharam: {', '.join(failed)}")
        return False

    return True


def run_pyinstaller():
    """Compila o executável com PyInstaller."""
    print_header("📦 FASE 3: PyInstaller — Compilando executável")

    spec_file = os.path.join(PROJECT_DIR, "WS Trader.spec")
    if not os.path.exists(spec_file):
        print(f"  ❌ Spec file não encontrado: {spec_file}")
        return False

    # Limpar build anterior
    build_dir = os.path.join(PROJECT_DIR, "build")
    if os.path.exists(build_dir):
        print_status("Limpando build anterior...")
        shutil.rmtree(build_dir, ignore_errors=True)

    cmd = f'pyinstaller "{spec_file}" --noconfirm --clean'
    print_status("Compilando WsTrader.exe (isto pode demorar)...")
    t0 = time.time()

    result = subprocess.run(
        cmd, shell=True, cwd=PROJECT_DIR,
        text=True, encoding="utf-8", errors="replace"
    )

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ❌ PyInstaller falhou (exit code {result.returncode})")
        if result.stderr:
            # Mostrar últimas linhas do erro
            lines = result.stderr.strip().split("\n")
            for line in lines[-10:]:
                print(f"     {line}")
        return False

    # Verificar que o EXE foi gerado
    exe_path = os.path.join(DIST_DIR, "WsTrader.exe")
    if not os.path.exists(exe_path):
        print(f"  ❌ WsTrader.exe não foi gerado em {DIST_DIR}")
        return False

    size_mb = os.path.getsize(exe_path) / (1024 * 1024)
    print_status(f"✅ WsTrader.exe gerado ({size_mb:.0f} MB) em {elapsed:.0f}s")

    return True


def run_nsis(version):
    """Gera o instalador NSIS."""
    print_header("🛠️ FASE 4: NSIS — Gerando instalador")

    if not os.path.exists(MAKENSIS):
        print(f"  ❌ makensis não encontrado em: {MAKENSIS}")
        print(f"     Instale o NSIS: https://nsis.sourceforge.io/")
        return False, None

    nsi_path = os.path.join(PROJECT_DIR, "installer.nsi")
    if not os.path.exists(nsi_path):
        print(f"  ❌ installer.nsi não encontrado")
        return False, None

    # Verificar que WsTrader.exe existe
    exe_path = os.path.join(DIST_DIR, "WsTrader.exe")
    if not os.path.exists(exe_path):
        print(f"  ❌ dist/WsTrader.exe não existe — execute a fase de build primeiro")
        return False, None

    cmd = f'"{MAKENSIS}" "{nsi_path}"'
    print_status("Gerando instalador...")
    t0 = time.time()

    result = subprocess.run(
        cmd, shell=True, cwd=PROJECT_DIR,
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ❌ NSIS falhou (exit code {result.returncode})")
        if result.stderr:
            print(f"     {result.stderr[:500]}")
        return False, None

    installer_name = f"WsTrader_Setup_{version}.exe"
    installer_path = os.path.join(PROJECT_DIR, installer_name)

    if not os.path.exists(installer_path):
        print(f"  ❌ Instalador não gerado: {installer_name}")
        return False, None

    size_mb = os.path.getsize(installer_path) / (1024 * 1024)
    print_status(f"✅ {installer_name} gerado ({size_mb:.0f} MB) em {elapsed:.0f}s")

    return True, installer_path


def git_push_and_release(version, installer_path):
    """Faz commit, push e cria release no GitHub."""
    print_header("🚀 FASE 5: Git Push + GitHub Release")

    # Stage version files
    run("git add version.json version_info.txt installer.nsi", check=False)
    run("git add -f ws_adaptive_brain.py ws_reversal_ai.py WS_AUTO_AI_BULLEX.py", check=False)
    run("git add models/", check=False)

    # Check if there are changes
    status = run("git status --porcelain", check=False)
    if status:
        commit_msg = f"release: v{version} — NN v7 24 features, sem guards, IA decide tudo"
        run(f'git commit -m "{commit_msg}"', check=False)
        print_status(f"✅ Commit: {commit_msg}")

        push_result = run("git push origin main", check=False)
        if push_result is not None:
            print_status("✅ Push concluído")
        else:
            print_status("❌ Push falhou — tente manualmente: git push origin main")
            return False
    else:
        print_status("ℹ️ Nenhuma mudança para commitar")

    # Criar release no GitHub usando gh CLI (se disponível)
    gh_available = run("gh --version", check=False)
    if gh_available and installer_path and os.path.exists(installer_path):
        print_status("Criando release no GitHub...")
        tag = f"v{version}"

        # Criar tag
        run(f'git tag -f {tag}', check=False)
        run(f'git push origin {tag} --force', check=False)

        # Criar release com o instalador
        installer_name = os.path.basename(installer_path)
        release_cmd = (
            f'gh release create {tag} "{installer_path}" '
            f'--title "WS Trader AI {tag}" '
            f'--notes "v{version} - NN v7 com 24 features. Sem guards rigidos, IA decide tudo. '
            f'37 modelos por ativo retreinados." '
            f'--latest'
        )
        rel_result = run(release_cmd, check=False)
        if rel_result is not None:
            print_status(f"✅ Release {tag} criada com {installer_name}")
        else:
            # Tentar fazer upload para release existente
            run(f'gh release upload {tag} "{installer_path}" --clobber', check=False)
            print_status(f"✅ Upload para release {tag}")
    else:
        if not gh_available:
            print_status("⚠️ GitHub CLI (gh) não encontrado — release manual necessária")
            print_status(f"   Instale: winget install GitHub.cli")
            print_status(f"   Ou faça upload manual do instalador em:")
            print_status(f"   https://github.com/{GITHUB_REPO}/releases/new?tag=v{version}")
        if installer_path:
            print_status(f"📦 Instalador pronto: {installer_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Build & Release WS Trader AI")
    parser.add_argument("--version", type=str, required=True,
                        help="Versão (ex: 5.6)")
    parser.add_argument("--no-push", action="store_true",
                        help="Não fazer git push/release")
    parser.add_argument("--skip-pyarmor", action="store_true",
                        help="Pular proteção PyArmor (usar protected_build/ existente)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Pular PyInstaller (usar dist/WsTrader.exe existente)")
    parser.add_argument("--skip-nsis", action="store_true",
                        help="Pular geração do instalador NSIS")
    args = parser.parse_args()

    version = args.version
    t_total = time.time()

    print_header(f"🚀 BUILD & RELEASE — WS Trader AI v{version}")
    print_status(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_status(f"📂 Projeto: {PROJECT_DIR}")
    print()

    # ── FASE 1: Atualizar versão ──
    if not update_version(version):
        sys.exit(1)

    # ── FASE 2: PyArmor ──
    if not args.skip_pyarmor:
        if not run_pyarmor():
            sys.exit(1)
    else:
        print_header("⏭️ FASE 2: PyArmor — PULADO (--skip-pyarmor)")

    # ── FASE 3: PyInstaller ──
    if not args.skip_build:
        if not run_pyinstaller():
            sys.exit(1)
    else:
        print_header("⏭️ FASE 3: PyInstaller — PULADO (--skip-build)")

    # ── FASE 4: NSIS ──
    installer_path = None
    if not args.skip_nsis:
        ok, installer_path = run_nsis(version)
        if not ok:
            sys.exit(1)
    else:
        print_header("⏭️ FASE 4: NSIS — PULADO (--skip-nsis)")

    # ── FASE 5: Git Push + Release ──
    if not args.no_push:
        git_push_and_release(version, installer_path)
    else:
        print_header("⏭️ FASE 5: Git Push — PULADO (--no-push)")

    # ── RESUMO ──
    elapsed = time.time() - t_total
    print_header(f"✅ BUILD & RELEASE v{version} CONCLUÍDO")
    print_status(f"⏱️ Tempo total: {elapsed:.0f}s")
    if installer_path and os.path.exists(installer_path):
        size_mb = os.path.getsize(installer_path) / (1024 * 1024)
        print_status(f"📦 Instalador: {os.path.basename(installer_path)} ({size_mb:.0f} MB)")
    print_status(f"🌐 Usuários receberão a atualização automaticamente")
    print()


if __name__ == "__main__":
    main()
