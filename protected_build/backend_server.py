"""
Servidor Backend Embutido - Roda automaticamente com o app
"""
import sys
import os
from pathlib import Path
import threading
import logging
import socket

# === FIX CRÍTICO para PyInstaller com console=False ===
# Quando console=False, sys.stdout e sys.stderr são None
# Isso causa crash silencioso em qualquer lib que tente print/log para stdout
# (uvicorn, fastapi, stripe, etc.)
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

# Configurar logging para arquivo — robusto para modo empacotado (console=False)
log_file = Path.home() / "wstrader_backend.log"

# Criar logger específico (não depende de basicConfig)
logger = logging.getLogger("wstrader_backend")
logger.setLevel(logging.INFO)
logger.handlers.clear()  # Limpar handlers anteriores

# FileHandler: sempre funciona
_fh = logging.FileHandler(str(log_file), mode='w', encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(_fh)

# StreamHandler: apenas se stdout existe (console=False -> sys.stdout=None)
if sys.stdout is not None:
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(_sh)


def _is_port_in_use(port: int) -> bool:
    """Verifica se a porta já está em uso."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True


def _kill_process_on_port(port: int):
    """Mata o processo que está ocupando a porta (Windows)."""
    try:
        import subprocess
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
        )
        for line in result.stdout.splitlines():
            if f"127.0.0.1:{port}" in line and "LISTENING" in line:
                parts = line.strip().split()
                pid = int(parts[-1])
                if pid > 0:
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid)],
                        capture_output=True, timeout=5,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
                    )
                    logger.info(f"Processo PID {pid} na porta {port} finalizado")
                    import time
                    time.sleep(0.5)
                    return True
    except Exception as ex:
        logger.warning(f"Não foi possível liberar porta {port}: {ex}")
    return False


def _check_existing_backend() -> bool:
    """Verifica se já existe um backend saudável rodando na porta 8000."""
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as resp:
            if resp.status == 200:
                logger.info("Backend já está rodando e saudável — reutilizando")
                return True
    except Exception:
        pass
    return False

def _run_server():
    """Função para rodar o uvicorn em thread separada"""
    try:
        logger.info("=== Iniciando servidor backend ===")
        # Forçar flush dos handlers para garantir que logs são escritos
        for h in logger.handlers:
            h.flush()

        # Detectar se está rodando empacotado (PyInstaller)
        is_frozen = getattr(sys, 'frozen', False)
        logger.info(f"Rodando empacotado: {is_frozen}")

        if is_frozen:
            # Rodando como executável
            base_path = Path(sys._MEIPASS)
            logger.info(f"Usando sys._MEIPASS: {base_path}")
            
            # Corrigir SSL para Stripe no modo empacotado
            # Tentar certifi primeiro, depois stripe ca-certificates como fallback
            certifi_path = base_path / "certifi" / "cacert.pem"
            stripe_ca_path = base_path / "stripe" / "data" / "ca-certificates.crt"
            
            ssl_cert = None
            if certifi_path.exists():
                ssl_cert = str(certifi_path)
                logger.info(f"Usando certifi cacert.pem: {certifi_path}")
            elif stripe_ca_path.exists():
                ssl_cert = str(stripe_ca_path)
                logger.info(f"Usando stripe ca-certificates.crt: {stripe_ca_path}")
            else:
                logger.warning(f"NENHUM certificado SSL encontrado em {base_path}")
                logger.warning(f"  certifi: {certifi_path} -> {certifi_path.exists()}")
                logger.warning(f"  stripe:  {stripe_ca_path} -> {stripe_ca_path.exists()}")
            
            if ssl_cert:
                os.environ["SSL_CERT_FILE"] = ssl_cert
                os.environ["REQUESTS_CA_BUNDLE"] = ssl_cert
                logger.info(f"SSL_CERT_FILE configurado: {ssl_cert}")
            
            # Verificar se pacote stripe está acessível
            try:
                import stripe as _stripe_test
                logger.info(f"stripe importado OK - path: {_stripe_test.__file__}")
            except Exception as stripe_err:
                logger.error(f"ERRO ao importar stripe: {stripe_err}", exc_info=True)
        else:
            # Rodando como script Python
            base_path = Path(__file__).parent
            logger.info(f"Usando __file__: {base_path}")
        
        # Flush após setup SSL
        for h in logger.handlers:
            h.flush()

        # Adicionar backend ao path
        backend_path = base_path / "backend"
        logger.info(f"Backend path: {backend_path}")
        logger.info(f"Backend existe: {backend_path.exists()}")

        if backend_path.exists():
            logger.info(f"Conteúdo de backend/: {list(backend_path.iterdir())[:10]}")

        sys.path.insert(0, str(backend_path))

        # Setar STRIPE_SECRET_KEY para o endpoint /check_subscription
        # Chave fica APENAS no config_keys.py (criptografado no executável)
        if not os.environ.get("STRIPE_SECRET_KEY"):
            try:
                parent_path = str(base_path)
                if parent_path not in sys.path:
                    sys.path.insert(0, parent_path)
                from config_keys import STRIPE_SECRET_KEY
                os.environ["STRIPE_SECRET_KEY"] = STRIPE_SECRET_KEY
                logger.info("STRIPE_SECRET_KEY configurada via config_keys")
            except Exception as ex:
                logger.warning(f"STRIPE_SECRET_KEY não encontrada: {ex}")

        # Importar main_stripe do backend
        logger.info("Importando main_stripe...")
        try:
            from main_stripe import app
            logger.info("main_stripe importado com sucesso!")
        except Exception as import_err:
            logger.error(f"ERRO ao importar main_stripe: {import_err}", exc_info=True)
            for h in logger.handlers:
                h.flush()
            return

        # Importar uvicorn
        logger.info("Importando uvicorn...")
        import uvicorn
        logger.info("uvicorn importado com sucesso!")
        
        # Flush antes de iniciar uvicorn
        for h in logger.handlers:
            h.flush()

        logger.info("Iniciando servidor uvicorn em 127.0.0.1:8000...")

        # Verificar se a porta está livre, se não, tentar liberar
        if _is_port_in_use(8000):
            logger.warning("Porta 8000 já está em uso!")
            if _check_existing_backend():
                logger.info("Backend anterior ainda responde — nada a fazer")
                return
            logger.info("Tentando liberar porta 8000...")
            _kill_process_on_port(8000)
            import time
            time.sleep(1)
            if _is_port_in_use(8000):
                logger.error("Não foi possível liberar porta 8000")
                return

        logger.info("Chamando uvicorn agora...")
        for h in logger.handlers:
            h.flush()

        try:
            # Usar uvicorn.run() diretamente — stdout/stderr já estão
            # redirecionados para devnull no topo do módulo (fix console=False)
            uvicorn.run(
                app,
                host="127.0.0.1",
                port=8000,
                log_level="info",
                access_log=True
            )
        except Exception as uv_err:
            logger.error(f"ERRO uvicorn.run(): {uv_err}", exc_info=True)
            for h in logger.handlers:
                h.flush()
    except Exception as e:
        logger.error(f"ERRO ao rodar servidor: {e}", exc_info=True)
        # Garantir que o erro é gravado no log mesmo se o processo morrer
        for h in logger.handlers:
            h.flush()


def _run_server_safe():
    """Wrapper de segurança: qualquer exceção não capturada vai pro log file."""
    try:
        _run_server()
    except Exception as e:
        # Last resort: escrever direto no arquivo se logging falhar
        try:
            import traceback
            crash_file = Path.home() / "wstrader_backend_crash.log"
            with open(str(crash_file), "w", encoding="utf-8") as f:
                f.write(f"CRASH: {e}\n\n{traceback.format_exc()}\n")
        except Exception:
            pass


def start_backend_server():
    """Inicia o servidor backend em segundo plano usando thread"""
    try:
        # Se já tem um backend saudável rodando, não inicia outro
        if _check_existing_backend():
            logger.info("Backend já ativo — pulando inicialização")
            return None

        logger.info("Criando thread do servidor...")
        server_thread = threading.Thread(target=_run_server_safe, daemon=True)
        server_thread.start()
        logger.info("Thread do servidor iniciada!")
        logger.info(f"Log do backend: {log_file}")
        return server_thread

    except Exception as e:
        logger.error(f"Erro ao iniciar servidor backend: {e}", exc_info=True)
        return None
