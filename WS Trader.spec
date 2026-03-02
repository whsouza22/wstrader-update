# -*- mode: python ; coding: utf-8 -*-
import os
import importlib
from PyInstaller.utils.hooks import collect_all

# Localizar flet_desktop automaticamente
flet_desktop_dir = os.path.dirname(importlib.import_module('flet_desktop').__file__)
flet_desktop_app = os.path.join(flet_desktop_dir, 'app')

# Localizar flet controls (icons.json etc)
flet_dir = os.path.dirname(importlib.import_module('flet').__file__)
flet_controls = os.path.join(flet_dir, 'controls')

# Coletar TODOS os submodules + dados do stripe e certifi
stripe_datas, stripe_binaries, stripe_hiddenimports = collect_all('stripe')
certifi_datas, certifi_binaries, certifi_hiddenimports = collect_all('certifi')

# === PYARMOR PROTECTION ===
# Os arquivos .py ofuscados estão em protected_build/
# O runtime PyArmor (.pyd) precisa ser incluído como binary
protected_dir = os.path.join(os.getcwd(), 'protected_build')
pyarmor_runtime_dir = os.path.join(protected_dir, 'pyarmor_runtime_009928')

a = Analysis(
    [os.path.join(protected_dir, 'TelaPrincipal.py')],
    pathex=[protected_dir, os.getcwd()],
    binaries=[
        # PyArmor runtime DLL - essencial para código protegido
        (os.path.join(pyarmor_runtime_dir, 'pyarmor_runtime.pyd'), 'pyarmor_runtime_009928'),
    ],
    datas=[
        ('Img', 'Img'),
        ('backend', 'backend'),
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
        (os.path.join(protected_dir, 'ws_auto_ai_engine.py'), '.'),
        (os.path.join(protected_dir, 'simple_sr_strategy.py'), '.'),
        (os.path.join(protected_dir, 'ws_confluence_brain.py'), '.'),
        (os.path.join(protected_dir, 'final_entry_guard.py'), '.'),
        (os.path.join(protected_dir, 'ws_data_manager.py'), '.'),
        (os.path.join(protected_dir, 'operations_manager.py'), '.'),
        (os.path.join(protected_dir, 'chat_screen_new.py'), '.'),
        (os.path.join(protected_dir, 'trading_bot.py'), '.'),
        (os.path.join(protected_dir, 'Login_Screen.py'), '.'),
        (os.path.join(protected_dir, 'tutorial_screen.py'), '.'),
        (os.path.join(protected_dir, 'license_manager.py'), '.'),
        (os.path.join(protected_dir, 'license_activation_screen.py'), '.'),
        (os.path.join(protected_dir, 'backend_server.py'), '.'),
        (os.path.join(protected_dir, 'backend_license_endpoint.py'), '.'),
        (os.path.join(protected_dir, 'ia_autonomous_brain.py'), '.'),
        (os.path.join(protected_dir, 'ws_candle_color_ai.py'), '.'),
        (os.path.join(protected_dir, 'ws_reversal_ai.py'), '.'),
        (os.path.join(protected_dir, 'ws_structure_map.py'), '.'),
        (os.path.join(protected_dir, 'ws_structure_patterns.py'), '.'),
        (os.path.join(protected_dir, 'dashboard_hs_ia.py'), '.'),
        (flet_desktop_app, os.path.join('flet_desktop', 'app')),
        (flet_controls, os.path.join('flet', 'controls')),
    ] + stripe_datas + certifi_datas,
    hiddenimports=[
        # === PyArmor Runtime ===
        'pyarmor_runtime_009928',
        # === Frameworks / libs ===
        'flet', 'flet_desktop', 'websocket', 'numpy', 'pandas', 'scipy', 'aiohttp',
        'openai', 'anthropic', 'dotenv', 'requests',
        'certifi', 'charset_normalizer', 'urllib3', 'idna',
        'pickle', 'ctypes', 'atexit', 'importlib', 'lightgbm',
        'psutil',
        'tkinter',
        'sqlalchemy', 'sqlalchemy.ext.declarative', 'sqlalchemy.orm',
        'uvicorn', 'fastapi', 'pydantic',
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
        'WS_AUTO_AI_BULLEX', 'ws_auto_ai_engine',
        'simple_sr_strategy', 'ws_confluence_brain', 'final_entry_guard',
        'ws_data_manager', 'config_keys',
        'operations_manager',
        'ia_autonomous_brain', 'ws_candle_color_ai',
        'ws_reversal_ai', 'ws_structure_map', 'ws_structure_patterns',
        'dashboard_hs_ia',
        # === Backend / Licença ===
        'backend_server', 'backend_license_endpoint', 'license_manager',
        'license_activation_screen',
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
    ] + stripe_hiddenimports + certifi_hiddenimports,
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
