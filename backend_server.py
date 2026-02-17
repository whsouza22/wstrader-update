"""
Servidor Backend Embutido - Roda automaticamente com o app
"""
import sys
import os
from pathlib import Path
import threading
import logging

# Configurar logging para arquivo
log_file = Path.home() / "wstrader_backend.log"

# Configurar handlers baseado no modo de execução
handlers = [logging.FileHandler(str(log_file), mode='w', encoding='utf-8')]

# Sempre adicionar StreamHandler (para mostrar logs no console)
handlers.append(logging.StreamHandler(sys.stdout))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger(__name__)

def _run_server():
    """Função para rodar o uvicorn em thread separada"""
    try:
        logger.info("=== Iniciando servidor backend ===")

        # Detectar se está rodando empacotado (PyInstaller)
        is_frozen = getattr(sys, 'frozen', False)
        logger.info(f"Rodando empacotado: {is_frozen}")

        if is_frozen:
            # Rodando como executável
            base_path = Path(sys._MEIPASS)
            logger.info(f"Usando sys._MEIPASS: {base_path}")
        else:
            # Rodando como script Python
            base_path = Path(__file__).parent
            logger.info(f"Usando __file__: {base_path}")

        # Adicionar backend ao path
        backend_path = base_path / "backend"
        logger.info(f"Backend path: {backend_path}")
        logger.info(f"Backend existe: {backend_path.exists()}")

        if backend_path.exists():
            logger.info(f"Conteúdo de backend/: {list(backend_path.iterdir())[:10]}")

        sys.path.insert(0, str(backend_path))

        # Definir caminho do credentials.json
        credentials_path = backend_path / "credentials.json"
        logger.info(f"Credentials path: {credentials_path}")
        logger.info(f"Credentials existe: {credentials_path.exists()}")

        if credentials_path.exists():
            logger.info(f"Tamanho do arquivo: {credentials_path.stat().st_size} bytes")

        # Setar variável de ambiente para o backend encontrar
        os.environ["FIREBASE_CREDENTIALS_PATH"] = str(credentials_path)

        # Importar main_firebase do backend
        logger.info("Importando main_firebase...")
        from main_firebase import app
        logger.info("main_firebase importado com sucesso!")

        # Importar uvicorn
        logger.info("Importando uvicorn...")
        import uvicorn
        logger.info("uvicorn importado com sucesso!")

        logger.info("Iniciando servidor uvicorn em 127.0.0.1:8000...")
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"ERRO ao rodar servidor: {e}", exc_info=True)

def start_backend_server():
    """Inicia o servidor backend em segundo plano usando thread"""
    try:
        logger.info("Criando thread do servidor...")
        server_thread = threading.Thread(target=_run_server, daemon=True)
        server_thread.start()
        logger.info("Thread do servidor iniciada!")
        logger.info(f"Log do backend: {log_file}")
        return server_thread

    except Exception as e:
        logger.error(f"Erro ao iniciar servidor backend: {e}", exc_info=True)
        return None
