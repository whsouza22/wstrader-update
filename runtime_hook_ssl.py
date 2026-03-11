# ===== PyInstaller Runtime Hook: SSL pre-initialization =====
# Este hook roda ANTES de qualquer módulo PyArmor-protegido.
#
# PROBLEMA: requests/adapters.py linha 80 faz (no nível do módulo):
#   _preloaded_ssl_context = create_urllib3_context()
# Mas só trata ImportError, não PermissionError.
# PyArmor intercepta operações de arquivo e redireciona para
# \\?\Volume{...}\virtual_file.log (read-only) → PermissionError.
#
# SOLUÇÃO: Importar requests COMPLETAMENTE aqui (antes do PyArmor carregar).
# Quando TelaPrincipal.py (PyArmor-protegido) fizer "import requests",
# o módulo já está em sys.modules → sem re-execução de código de módulo.

import os
import sys

# ===== FIX: Flet Desktop client path in frozen mode =====
# No PyInstaller OneFile, flet_desktop.get_package_bin_dir() usa __file__
# que pode não resolver para _MEIPASS corretamente.
# Forçamos o path correto E setamos FLET_VIEW_PATH como fallback.
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    _flet_app_dir = os.path.join(sys._MEIPASS, 'flet_desktop', 'app')
    _flet_exe_dir = os.path.join(_flet_app_dir, 'flet')
    if os.path.isdir(_flet_exe_dir):
        os.environ['FLET_VIEW_PATH'] = _flet_exe_dir
    try:
        import flet_desktop
        def _frozen_bin_dir():
            return _flet_app_dir
        flet_desktop.get_package_bin_dir = _frozen_bin_dir
    except Exception:
        pass

# 1) Configurar certificados SSL via certifi
try:
    import certifi
    _ca = certifi.where()
    os.environ.setdefault('SSL_CERT_FILE', _ca)
    os.environ.setdefault('REQUESTS_CA_BUNDLE', _ca)
    os.environ.setdefault('CURL_CA_BUNDLE', _ca)
except Exception:
    pass

# 2) Remover SSLKEYLOGFILE se existir (previne escrita em virtual_file.log)
os.environ.pop('SSLKEYLOGFILE', None)

# 3) Pré-criar SSL context (antes do PyArmor)
try:
    import ssl
    ssl.create_default_context()
except Exception:
    pass

# 4) Monkey-patch create_urllib3_context para capturar PermissionError
try:
    import urllib3.util.ssl_ as _u3ssl
    _orig_create_urllib3_context = _u3ssl.create_urllib3_context

    def _safe_create_urllib3_context(*args, **kwargs):
        try:
            return _orig_create_urllib3_context(*args, **kwargs)
        except PermissionError:
            # Fallback: criar contexto básico sem operação de arquivo
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.check_hostname = True
            ctx.verify_mode = ssl.CERT_REQUIRED
            try:
                import certifi as _cf
                ctx.load_verify_locations(_cf.where())
            except Exception:
                pass
            return ctx

    _u3ssl.create_urllib3_context = _safe_create_urllib3_context
except Exception:
    pass

# 5) Importar requests COMPLETAMENTE (dispara todo o module-level code
#    ANTES do PyArmor ativar o virtual filesystem)
try:
    import requests  # noqa: F401
except Exception:
    pass
