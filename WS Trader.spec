# -*- mode: python ; coding: utf-8 -*-
import os
import importlib
from PyInstaller.utils.hooks import collect_all

PROJECT_ROOT = os.getcwd()
SOURCE_FILES = [
    'TelaPrincipal.py',
    'WS_AUTO_AI_BULLEX.py',
    'ws_reversal_ai.py',
    'ws_adaptive_brain.py',
    'ws_generative_guard.py',
    'config_keys.py',
    'Login_Screen.py',
    'trading_bot.py',
    'dashboard_hs_ia.py',
    'operations_manager.py',
    'chat_screen_new.py',
    'tutorial_screen.py',
    'license_manager.py',
    'backend_server.py',
    'train_neural_network.py',
    'ws_continuation_ai.py',
    'train_continuation_ml.py',
]

# Localizar flet_desktop automaticamente
flet_desktop_dir = os.path.dirname(importlib.import_module('flet_desktop').__file__)
flet_desktop_app = os.path.join(flet_desktop_dir, 'app')

# Localizar flet controls (icons.json etc)
flet_dir = os.path.dirname(importlib.import_module('flet').__file__)
flet_controls = os.path.join(flet_dir, 'controls')

# Coletar TODOS os submodules + dados do stripe e certifi
stripe_datas, stripe_binaries, stripe_hiddenimports = collect_all('stripe')
certifi_datas, certifi_binaries, certifi_hiddenimports = collect_all('certifi')
xgboost_datas, xgboost_binaries, xgboost_hiddenimports = collect_all('xgboost')

# === PYARMOR PROTECTION ===
# Os arquivos .py ofuscados estão em protected_build/
# O runtime PyArmor (.pyd) precisa ser incluído como binary
protected_dir = os.path.join(os.getcwd(), 'protected_build')
pyarmor_runtime_dir = os.path.join(protected_dir, 'pyarmor_runtime_009928')


def _assert_protected_build_fresh():
    if os.getenv('WS_SKIP_PROTECTED_FRESHNESS_CHECK', '0').strip() == '1':
        return

    missing = []
    stale = []
    for name in SOURCE_FILES:
        src = os.path.join(PROJECT_ROOT, name)
        prot = os.path.join(protected_dir, name)
        if not os.path.exists(prot):
            missing.append(name)
            continue
        if os.path.exists(src) and os.path.getmtime(prot) < os.path.getmtime(src):
            stale.append(name)

    runtime_pyd = os.path.join(pyarmor_runtime_dir, 'pyarmor_runtime.pyd')
    if not os.path.exists(runtime_pyd):
        missing.append('pyarmor_runtime_009928/pyarmor_runtime.pyd')

    if missing or stale:
        problems = []
        if missing:
            problems.append('faltando: ' + ', '.join(missing))
        if stale:
            problems.append('desatualizado: ' + ', '.join(stale))
        raise RuntimeError(
            'protected_build não está pronto para compilar (' + '; '.join(problems) + '). '
            'Rode tools/build_release.py sem --skip-pyarmor ou regenere o PyArmor antes do PyInstaller.'
        )


_assert_protected_build_fresh()

import sys as _sys
_python_home = os.path.dirname(_sys.executable)
# Em venvs, o python3xx.dll fica na instalação base
_python_base = os.path.dirname(os.path.dirname(_sys.executable))  # venv -> base
if not os.path.exists(os.path.join(_python_home, 'python313.dll')):
    # Tentar base do Python (fora do venv)
    for _candidate in [
        os.path.join(_python_base, 'python313.dll'),
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Python', 'Python313', 'python313.dll'),
        os.path.join(os.environ.get('PROGRAMFILES', ''), 'Python313', 'python313.dll'),
    ]:
        if os.path.exists(_candidate):
            _python_home = os.path.dirname(_candidate)
            break

_python_dlls = []
for _dll_name in ['python313.dll', 'python3.dll']:
    _dll_path = os.path.join(_python_home, _dll_name)
    if os.path.exists(_dll_path):
        _python_dlls.append((_dll_path, '.'))

a = Analysis(
    [os.path.join(protected_dir, 'TelaPrincipal.py')],
    pathex=[protected_dir, os.getcwd()],
    binaries=[
        # PyArmor runtime DLL - essencial para código protegido
        (os.path.join(pyarmor_runtime_dir, 'pyarmor_runtime.pyd'), 'pyarmor_runtime_009928'),
    ] + _python_dlls + xgboost_binaries,
    datas=[
        ('Img', 'Img'),
        ('backend', 'backend'),
        ('models', 'models'),
        ('models_entry_guard', 'models_entry_guard'),
        ('candles_100k', 'candles_100k'),
        ('trade_decisions.html', '.'),
        ('ws_ai_base_training.json', '.'),
        ('version_info.txt', '.'),
        # === Broker APIs (pacotes locais completos) ===
        ('iqoptionapi', 'iqoptionapi'),
        ('bullexapi', 'bullexapi'),
        ('casatraderapi', 'casatraderapi'),
        # PyArmor runtime package completo
        (pyarmor_runtime_dir, 'pyarmor_runtime_009928'),
        # Arquivos protegidos como datas (para importlib.import_module no frozen)
        (os.path.join(protected_dir, 'config_keys.py'), '.'),
        (os.path.join(protected_dir, 'WS_AUTO_AI_BULLEX.py'), '.'),
        (os.path.join(protected_dir, 'ws_generative_guard.py'), '.'),

        (os.path.join(protected_dir, 'operations_manager.py'), '.'),
        (os.path.join(protected_dir, 'chat_screen_new.py'), '.'),
        (os.path.join(protected_dir, 'trading_bot.py'), '.'),
        (os.path.join(protected_dir, 'Login_Screen.py'), '.'),
        (os.path.join(protected_dir, 'tutorial_screen.py'), '.'),
        (os.path.join(protected_dir, 'license_manager.py'), '.'),
        (os.path.join(protected_dir, 'backend_server.py'), '.'),

        (os.path.join(protected_dir, 'ws_reversal_ai.py'), '.'),
        # numpy_pickle_compat — helper de compatibilidade (não protegido)
        ('numpy_pickle_compat.py', '.'),

        (os.path.join(protected_dir, 'dashboard_hs_ia.py'), '.'),
        (os.path.join(protected_dir, 'ws_adaptive_brain.py'), '.'),
        (os.path.join(protected_dir, 'ws_continuation_ai.py'), '.'),
        (os.path.join(protected_dir, 'train_continuation_ml.py'), '.'),
        (flet_desktop_app, os.path.join('flet_desktop', 'app')),
        (flet_controls, os.path.join('flet', 'controls')),
    ] + stripe_datas + certifi_datas + xgboost_datas,
    hiddenimports=[
        # === PyArmor Runtime ===
        'pyarmor_runtime_009928',
        # === Frameworks / libs ===
        'flet', 'flet_desktop', 'websocket', 'numpy', 'pandas', 'scipy', 'aiohttp',
        'openai', 'anthropic', 'dotenv', 'requests',
        'certifi', 'charset_normalizer', 'urllib3', 'idna',
        'pickle', 'ctypes', 'atexit', 'importlib', 'lightgbm',
        'xgboost', 'xgboost.core', 'xgboost.sklearn',
        'sklearn', 'sklearn.neural_network', 'sklearn.preprocessing',
        'sklearn.ensemble', 'sklearn.utils', 'sklearn.utils._bunch',
        'psutil',
        'tkinter',
        'sqlalchemy', 'sqlalchemy.ext.declarative', 'sqlalchemy.orm',
        'uvicorn', 'fastapi', 'pydantic',
        'fastapi.middleware', 'fastapi.middleware.cors', 'fastapi.staticfiles',
        'starlette.middleware', 'starlette.middleware.cors', 'starlette.staticfiles',
        'stripe',
        # === Broker APIs (todos submodules) ===
        'iqoptionapi', 'iqoptionapi.stable_api', 'iqoptionapi.api',
        'iqoptionapi.constants', 'iqoptionapi.country_id', 'iqoptionapi.expiration',
        'iqoptionapi.global_value', 'iqoptionapi.version_control',
        'bullexapi', 'bullexapi.stable_api', 'bullexapi.api',
        'bullexapi.constants', 'bullexapi.country_id', 'bullexapi.expiration',
        'bullexapi.global_value', 'bullexapi.version_control',
        'casatraderapi', 'casatraderapi.stable_api', 'casatraderapi.api',
        'casatraderapi.constants', 'casatraderapi.country_id', 'casatraderapi.expiration',
        'casatraderapi.global_value', 'casatraderapi.version_control',
        # === Motor IA / Estratégia ===
        'WS_AUTO_AI_BULLEX', 'ws_generative_guard',

        'operations_manager', 'config_keys',
        'operations_manager',
        'ws_reversal_ai',
        'numpy_pickle_compat',
        'dashboard_hs_ia', 'ws_adaptive_brain',
        'ws_continuation_ai', 'train_continuation_ml',
        # === Backend / Licença ===
        'backend_server', 'license_manager',
        # === UI / Telas ===
        'chat_screen_new', 'Login_Screen', 'trading_bot', 'tutorial_screen',
        # === Submodules extras broker APIs ===
        'iqoptionapi.ws', 'iqoptionapi.ws.client',
        'iqoptionapi.http', 'iqoptionapi.http.login', 'iqoptionapi.http.auth',
        'iqoptionapi.http.billing', 'iqoptionapi.http.resource',
        'iqoptionapi.http.appinit', 'iqoptionapi.http.getprofile',
        'iqoptionapi.http.changebalance', 'iqoptionapi.http.buyback',
        'bullexapi.ws', 'bullexapi.ws.client',
        'bullexapi.http', 'bullexapi.http.login', 'bullexapi.http.auth',
        'bullexapi.http.billing', 'bullexapi.http.resource',
        'bullexapi.http.appinit', 'bullexapi.http.getprofile',
        'bullexapi.http.changebalance', 'bullexapi.http.buyback',
        'casatraderapi.ws', 'casatraderapi.ws.client',
        'casatraderapi.http', 'casatraderapi.http.login', 'casatraderapi.http.auth',
        'casatraderapi.http.billing', 'casatraderapi.http.resource',
        'casatraderapi.http.appinit', 'casatraderapi.http.getprofile',
        'casatraderapi.http.changebalance', 'casatraderapi.http.buyback',
    ] + stripe_hiddenimports + certifi_hiddenimports + xgboost_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook_ssl.py'],
    excludes=['tensorflow', 'keras', 'tensorboard', 'tf_keras', 'h5py',
              'torch', 'torchvision', 'torchaudio', 'transformers',
              'IPython', 'notebook', 'jupyter', 'jupyterlab',
              'google_auth_oauthlib', 'googleapiclient', 'google.auth',
              'google.oauth2', 'google_auth_httplib2',
              'firebase_admin', 'google.cloud',
              'llvmlite', 'numba'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='WsTrader',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['Img\\app_icon.ico'],
    version='version_info.txt',
)
