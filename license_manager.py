"""
Sistema de Licenciamento WS Trader - VERSÃO STRIPE DIRETO
Validação direto na API Stripe usando a chave local (config_keys.py)
Sem dependência de backend externo (Render)
"""
import logging
import stripe

logger = logging.getLogger(__name__)

# ── Chave Stripe direto do config_keys (embarcada no executável) ──
try:
    from config_keys import STRIPE_SECRET_KEY
    stripe.api_key = STRIPE_SECRET_KEY
    logger.info("🔑 Stripe API key carregada via config_keys")
except ImportError:
    logger.warning("⚠️ config_keys não encontrado — Stripe desabilitado")

# ── IDs de produto Stripe ──
PRO_PRODUCT_ID  = "prod_S4t8FQuUptWQ6R"
DEMO_PRODUCT_ID = "prod_U3CRqZJMVigJAK"


def check_stripe_subscription(email: str) -> tuple:
    """
    Verifica assinatura ativa direto na API Stripe (sem backend externo).

    Returns:
        tuple: (is_valid, license_type, error_message)
    """
    try:
        clean_email = email.strip().lower()
        logger.info(f"🔑 Verificando assinatura Stripe para: {clean_email}")

        # 1) Buscar cliente pelo email
        customers = stripe.Customer.list(email=clean_email, limit=1).data
        if not customers:
            logger.warning("❌ Cliente não encontrado no Stripe")
            return False, None, "Cliente não encontrado no Stripe."

        customer_id = customers[0].id

        # 2) Buscar assinaturas do cliente
        subscriptions = stripe.Subscription.list(customer=customer_id, limit=10).data

        # 3) Procurar assinatura ativa ou em trial
        active_sub = next(
            (s for s in subscriptions if s.status in ("active", "trialing")),
            None
        )

        if not active_sub:
            logger.warning("❌ Nenhuma assinatura ativa encontrada")
            return False, None, "Nenhuma assinatura ativa encontrada."

        # 4) Extrair product_id
        product_id = ""
        try:
            items = active_sub.get("items", {}).get("data", [])
            if items:
                product_id = items[0].get("price", {}).get("product", "")
        except Exception:
            pass

        # 5) Determinar plan_type
        if product_id == PRO_PRODUCT_ID:
            plan_type = "PRO"
        elif product_id == DEMO_PRODUCT_ID:
            plan_type = "DEMO"
        else:
            plan_type = "DEMO"  # Qualquer outro produto = DEMO

        logger.info(f"✅ Assinatura válida — plano: {plan_type}  product: {product_id}")
        return True, plan_type, None

    except stripe.error.AuthenticationError:
        logger.error("❌ Chave Stripe inválida")
        return False, None, "Chave Stripe inválida"

    except stripe.error.APIConnectionError:
        logger.error("❌ Sem conexão com Stripe API")
        return False, None, "Sem conexão com a internet"

    except stripe.error.StripeError as e:
        logger.error(f"❌ Erro Stripe: {e}")
        return False, None, f"Erro Stripe: {str(e)}"

    except Exception as e:
        logger.error(f"❌ Erro inesperado: {e}")
        return False, None, f"Erro: {str(e)}"


# Alias para compatibilidade
def validate_license(license_key: str = None, email: str = None) -> tuple:
    """
    Wrapper para compatibilidade — usa Stripe direto.
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


# Funções-stub para compatibilidade de imports
def get_hardware_id():
    """Stub — não usado no modelo Stripe"""
    return "N/A"


def get_machine_info():
    """Stub — não usado no modelo Stripe"""
    return {}


def check_online_connection():
    """Verifica se consegue conectar ao Stripe"""
    try:
        stripe.Customer.list(limit=1)
        return True
    except Exception:
        return False
