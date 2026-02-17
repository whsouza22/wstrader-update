# -*- mode: python ; coding: utf-8 -*-
import os
import importlib

# Localizar flet_desktop automaticamente
flet_desktop_dir = os.path.dirname(importlib.import_module('flet_desktop').__file__)
flet_desktop_app = os.path.join(flet_desktop_dir, 'app')

# Localizar flet controls (icons.json etc)
flet_dir = os.path.dirname(importlib.import_module('flet').__file__)
flet_controls = os.path.join(flet_dir, 'controls')

a = Analysis(
    ['TelaPrincipal.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('Img', 'Img'),
        ('backend', 'backend'),
        ('version_info.txt', '.'),
        ('config_keys.py', '.'),
        (flet_desktop_app, os.path.join('flet_desktop', 'app')),
        (flet_controls, os.path.join('flet', 'controls')),
    ],
    hiddenimports=[
        'flet', 'flet_desktop', 'websocket', 'numpy', 'pandas', 'scipy', 'aiohttp',
        'openai', 'anthropic', 'dotenv', 'requests',
        'iqoptionapi', 'iqoptionapi.stable_api',
        'bullexapi', 'bullexapi.stable_api',
        'casatraderapi', 'casatraderapi.stable_api',
        'WS_AUTO_AI', 'WS_AUTO_AI_BULLEX', 'WS_AUTO_AI_OPTIMIZED',
        'IA_Cod_CasaTrader', 'IA_Cod_IQ', 'IA_Cod_Bullex',
        'WS_NEURAL_AI', 'WS_HYBRID_AI', 'WS_NEURAL_BRAIN',
        'dom_forex_strategy', 'ai_claude_calibrator', 'ai_pattern_analyzer', 'ai_learning',
        'ai_auto_fixer', 'ia_autoconhecimento', 'cnn_pattern_detector',
        'ia_m1_otc_signal', 'ia_top_m1',
        'config_keys', 'auto_updater',
        'ws_auto_ai_engine', 'loss_analyzer', 'loss_analyzer_ai',
        'pattern_detector', 'smart_memory', 'regime_filter', 'risk_control',
        'neural_model', 'multi_agent_consensus', 'trend_structure_ai',
        'logistic_online',
        'operations_manager', 'backend_server', 'backend_license_endpoint', 'license_manager',
        'license_activation_screen',
        'chat_screen_new', 'Login_Screen', 'trading_bot', 'tutorial_screen',
        'chat_ai_assistant', 'auto_tuner', 'auto_optimizer',
        'pickle', 'ctypes', 'atexit', 'importlib', 'lightgbm',
        'uvicorn', 'fastapi', 'pydantic',
        'firebase_admin', 'firebase_admin.credentials', 'firebase_admin.firestore',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tensorflow', 'keras', 'tensorboard', 'tf_keras', 'h5py',
              'torch', 'torchvision', 'torchaudio', 'transformers',
              'IPython', 'notebook', 'jupyter', 'jupyterlab',
              'google_auth_oauthlib', 'googleapiclient', 'google.auth',
              'google.oauth2', 'google_auth_httplib2',
              'llvmlite', 'numba'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
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
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='WsTrader',
)
