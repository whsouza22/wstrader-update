"""
Auto-Updater para WS Trader AI
Realiza atualização silenciosa sem necessidade de reinstalação manual.
"""

import os
import sys
import time
import logging
import requests
import subprocess
import tempfile
from typing import Tuple, Optional, Callable

logger = logging.getLogger(__name__)

class AutoUpdater:
    """Gerenciador de auto-atualização silenciosa"""

    def __init__(self, current_version: str, version_url: str):
        self.current_version = current_version
        self.version_url = version_url
        self.download_progress_callback: Optional[Callable[[float, str], None]] = None

    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Define callback para progresso do download (progress: 0-1, message: str)"""
        self.download_progress_callback = callback

    def _report_progress(self, progress: float, message: str):
        """Reporta progresso se callback definido"""
        if self.download_progress_callback:
            try:
                self.download_progress_callback(progress, message)
            except Exception:
                pass

    def _parse_version(self, v: str) -> tuple:
        """Converte string de versão para tupla comparável"""
        import re
        parts = re.findall(r"\d+", v or "")
        return tuple(int(p) for p in parts) if parts else (0,)

    def check_for_update(self) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """
        Verifica se há atualização disponível.
        Retorna: (has_update, new_version, changelog, download_url)
        """
        try:
            self._report_progress(0.0, "Verificando atualizacoes...")
            response = requests.get(self.version_url, timeout=15)
            response.raise_for_status()

            data = response.json()
            latest_version = data.get("version", "")
            changelog = data.get("changelog", "")
            download_url = data.get("download_link") or data.get("installer_url", "")

            # Limpa URL
            if download_url:
                download_url = download_url.replace("\n", "").replace(" ", "").strip()
                if download_url.startswith("https//"):
                    download_url = download_url.replace("https//", "https://", 1)

            if not latest_version or not download_url:
                logger.warning("Dados de versao incompletos no servidor")
                return False, None, None, None

            has_update = self._parse_version(latest_version) > self._parse_version(self.current_version)

            if has_update:
                logger.info(f"Nova versao disponivel: {latest_version} (atual: {self.current_version})")
            else:
                logger.info(f"Versao {self.current_version} esta atualizada")

            return has_update, latest_version, changelog, download_url

        except Exception as e:
            logger.error(f"Erro ao verificar atualizacao: {e}")
            return False, None, None, None

    def download_update(self, download_url: str, target_path: Optional[str] = None) -> Optional[str]:
        """
        Baixa o instalador de atualização.
        Retorna: caminho do arquivo baixado ou None se falhou
        """
        try:
            if not target_path:
                # Usa pasta temporária do usuário
                temp_dir = os.path.join(tempfile.gettempdir(), "wstrader_update")
                os.makedirs(temp_dir, exist_ok=True)
                target_path = os.path.join(temp_dir, "WsTrader_Update.exe")

            self._report_progress(0.0, "Iniciando download...")
            logger.info(f"Baixando atualizacao de: {download_url}")

            with requests.get(download_url, stream=True, timeout=300) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                downloaded = 0
                chunk_size = 8192

                with open(target_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            if total_size > 0:
                                progress = downloaded / total_size
                                mb_downloaded = downloaded / (1024 * 1024)
                                mb_total = total_size / (1024 * 1024)
                                self._report_progress(
                                    progress,
                                    f"Baixando... {mb_downloaded:.1f}/{mb_total:.1f} MB ({progress*100:.0f}%)"
                                )

            logger.info(f"Download concluido: {target_path}")
            self._report_progress(1.0, "Download concluido!")
            return target_path

        except Exception as e:
            logger.error(f"Erro no download: {e}")
            self._report_progress(0.0, f"Erro no download: {str(e)[:50]}")
            return None

    def apply_update(self, installer_path: str, restart_app: bool = True) -> bool:
        """
        Aplica a atualização executando o instalador em modo silencioso.
        O app atual será fechado e o novo será iniciado automaticamente.
        """
        try:
            if not os.path.exists(installer_path):
                logger.error(f"Instalador nao encontrado: {installer_path}")
                return False

            self._report_progress(1.0, "Aplicando atualizacao...")
            logger.info(f"Executando instalador silencioso: {installer_path}")

            # Cria script batch para executar após o app fechar
            batch_path = os.path.join(tempfile.gettempdir(), "wstrader_update.bat")

            # Obtém o caminho do executável atual
            if getattr(sys, 'frozen', False):
                app_path = sys.executable
            else:
                app_path = os.path.abspath(sys.argv[0])

            # Obtém diretório de instalação do registro ou usa padrão
            install_dir = os.environ.get('PROGRAMFILES', 'C:\\Program Files') + '\\WsTrader'
            app_exe = os.path.join(install_dir, 'WsTrader.exe')

            batch_content = f'''@echo off
title WS Trader - Atualizando...
echo.
echo ========================================
echo    WS Trader AI - Atualizacao
echo ========================================
echo.
echo Aguardando aplicativo fechar...
timeout /t 2 /nobreak >nul

set WAIT_COUNT=0
:wait_loop
tasklist /FI "IMAGENAME eq WsTrader.exe" 2>NUL | find /I /N "WsTrader.exe">NUL
if "%ERRORLEVEL%"=="0" (
    set /a WAIT_COUNT+=1
    if %WAIT_COUNT% GTR 30 (
        echo Timeout aguardando aplicativo fechar
        echo Forçando encerramento...
        taskkill /F /IM WsTrader.exe >nul 2>&1
        timeout /t 2 /nobreak >nul
        goto install
    )
    timeout /t 1 /nobreak >nul
    goto wait_loop
)

:install
echo Instalando atualizacao...
echo Diretorio: {install_dir}
"{installer_path}" /S

echo Aguardando instalacao concluir...
timeout /t 8 /nobreak >nul

echo Verificando se arquivos foram atualizados...
if exist "{app_exe}" (
    echo Instalacao concluida com sucesso
) else (
    echo AVISO: Executavel nao encontrado apos instalacao
)
'''

            if restart_app:
                batch_content += f'''
echo Verificando instalacao...
if exist "{app_exe}" (
    echo Iniciando WS Trader AI...
    timeout /t 2 /nobreak >nul
    start "" "{app_exe}"
    timeout /t 2 /nobreak >nul
) else (
    echo ERRO: Executavel nao encontrado em {app_exe}
    echo Tentando caminho alternativo...
    if exist "{app_path}" (
        start "" "{app_path}"
    ) else (
        echo Nenhum executavel encontrado
        pause
    )
)
'''

            batch_content += '''
del "%~f0" & exit
'''

            with open(batch_path, 'w', encoding='cp1252') as f:
                f.write(batch_content)

            logger.info(f"Script de atualizacao criado: {batch_path}")

            # Executa o batch em segundo plano
            subprocess.Popen(
                f'cmd /c start /min "" "{batch_path}"',
                shell=True,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )

            logger.info("Script de atualizacao iniciado, fechando aplicativo...")
            return True

        except Exception as e:
            logger.error(f"Erro ao aplicar atualizacao: {e}")
            return False

    def perform_auto_update(self) -> Tuple[bool, str]:
        """
        Executa o processo completo de auto-atualização.
        Retorna: (success, message)
        """
        # Verifica atualização
        has_update, new_version, changelog, download_url = self.check_for_update()

        if not has_update:
            return False, "Nenhuma atualizacao disponivel"

        if not download_url:
            return False, "URL de download nao disponivel"

        # Baixa o instalador
        installer_path = self.download_update(download_url)

        if not installer_path:
            return False, "Falha no download da atualizacao"

        # Aplica a atualização
        if self.apply_update(installer_path):
            return True, f"Atualizando para versao {new_version}..."
        else:
            return False, "Falha ao aplicar atualizacao"


def get_updater(current_version: str, version_url: str) -> AutoUpdater:
    """Factory function para criar instância do AutoUpdater"""
    return AutoUpdater(current_version, version_url)
