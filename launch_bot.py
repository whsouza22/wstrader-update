"""Script para lançar o bot com credenciais carregadas do .env"""
import os, sys
from pathlib import Path

# Carregar .env
env_file = Path.home() / ".wstrader" / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()

# Detectar broker automaticamente pelas variáveis presentes no .env
if os.environ.get("IQ_EMAIL"):
    email = os.environ["IQ_EMAIL"]
    senha = os.environ.get("IQ_PASSWORD", "") or os.environ.get("IQ_PASS", "")
    os.environ["IQ_EMAIL"] = email
    os.environ["IQ_PASS"] = senha
    os.environ["IQ_PASSWORD"] = senha
    os.environ["BROKER_TYPE"] = "iq_option"
    os.environ["IQ_CONTA"] = os.environ.get("IQ_CONTA", "PRACTICE")
    broker = "iq_option"
elif os.environ.get("BULLUX_EMAIL") or os.environ.get("BULLEX_EMAIL"):
    email = os.environ.get("BULLUX_EMAIL", "") or os.environ.get("BULLEX_EMAIL", "")
    senha = os.environ.get("BULLUX_PASS", "") or os.environ.get("BULLEX_PASS", "")
    os.environ["BULLUX_EMAIL"] = email
    os.environ["BULLUX_PASS"] = senha
    os.environ["BROKER_TYPE"] = "bullex"
    os.environ["BULLUX_CONTA"] = os.environ.get("BULLUX_CONTA", "PRACTICE")
    broker = "bullex"
elif os.environ.get("CASATRADER_EMAIL"):
    email = os.environ["CASATRADER_EMAIL"]
    senha = os.environ.get("CASATRADER_PASS", "")
    os.environ["BROKER_TYPE"] = "casatrader"
    os.environ["CASATRADER_CONTA"] = os.environ.get("CASATRADER_CONTA", "PRACTICE")
    broker = "casatrader"
else:
    print("ERRO: Nenhuma credencial de broker encontrada no .env")
    sys.exit(1)

print(f"Launching bot: email={email[:5]}... broker={broker} conta=PRACTICE")

# Importar e rodar
from WS_AUTO_AI_BULLEX import main
main()
