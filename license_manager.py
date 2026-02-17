"""
Sistema de Licenciamento WS Trader - VERSÃO STRIPE ONLY
Validação apenas via Stripe - sem chaves manuais
Usuários FREE têm vencimento até 2050 no Stripe
"""
import requests
import logging

logger = logging.getLogger(__name__)

# URL do backend que verifica assinatura no Stripe
STRIPE_CHECK_URL = "https://api-wstrader.onrender.com/check_subscription"


def check_stripe_subscription(email: str) -> tuple:
    """
    Verifica se o email tem assinatura ativa no Stripe.

    Args:
        email: Email do usuário

    Returns:
        tuple: (is_valid, license_type, error_message)
        - is_valid: True se tem assinatura válida
        - license_type: "FREE" ou "PRO"
        - error_message: Mensagem de erro (se houver)
    """
    try:
        logger.info(f"🔑 Verificando assinatura Stripe para: {email}")

        response = requests.post(
            STRIPE_CHECK_URL,
            json={"email": email.strip().lower()},
            timeout=15
        )

        logger.info(f"📥 Resposta Stripe: HTTP {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            if data.get("valid") or data.get("is_active"):
                license_type = data.get("plan_type", "PRO").upper()
                if license_type not in ["FREE", "PRO", "PREMIUM"]:
                    license_type = "PRO"

                logger.info(f"✅ Assinatura válida: {license_type}")
                return True, license_type, None
            else:
                error = data.get("message", "Nenhuma assinatura ativa encontrada")
                logger.warning(f"❌ {error}")
                return False, None, error
        else:
            error_msg = f"Erro ao verificar assinatura (HTTP {response.status_code})"
            logger.error(error_msg)
            return False, None, error_msg

    except requests.exceptions.Timeout:
        logger.error("❌ Timeout ao conectar no Stripe")
        return False, None, "Servidor não respondeu (timeout)"

    except requests.exceptions.ConnectionError:
        logger.error("❌ Erro de conexão com servidor")
        return False, None, "Não foi possível conectar ao servidor"

    except Exception as e:
        logger.error(f"❌ Erro inesperado: {str(e)}")
        return False, None, f"Erro: {str(e)}"


# Alias para compatibilidade
def validate_license(license_key: str = None, email: str = None) -> tuple:
    """
    Wrapper para compatibilidade - agora usa apenas Stripe.
    A chave é ignorada, apenas o email é usado.
    """
    if not email:
        return False, "Email é obrigatório", None

    is_valid, license_type, error = check_stripe_subscription(email)

    if is_valid:
        user_data = {"license_type": license_type, "email": email}
        return True, None, user_data
    else:
        return False, error, None


def check_online_connection():
    """Verifica se consegue conectar ao servidor Stripe"""
    try:
        response = requests.get("https://api-wstrader.onrender.com/health", timeout=5)
        return response.status_code == 200
    except:
        return False
