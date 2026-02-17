@echo off
title WS Trader - Firebase Backend
echo ================================================================
echo                 WS TRADER - FIREBASE BACKEND
echo ================================================================
echo.
echo Iniciando servidor Firebase para gravacao de losses...
echo.

cd /d "%~dp0"

:: Verificar se credentials.json existe
if not exist "backend\credentials.json" (
    echo [ERRO] Arquivo credentials.json nao encontrado em backend/
    echo Por favor, coloque seu arquivo de credenciais do Firebase
    pause
    exit /b 1
)

echo [OK] Credenciais encontradas em: backend\credentials.json
echo.
echo Servidor rodando em: http://localhost:8000
echo.
echo NAO FECHE ESTA JANELA enquanto usar o WS Trader!
echo.
echo ================================================================
echo.

:: Mudar para pasta backend e iniciar servidor
cd backend
python -c "import uvicorn; uvicorn.run('main_firebase:app', host='0.0.0.0', port=8000, reload=False)"

pause
