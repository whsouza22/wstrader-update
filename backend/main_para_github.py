"""
API Backend WS Trader - Versão Simplificada
Endpoints: Stripe check + Free License validation
"""
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional
import os
import logging
import stripe
from dotenv import load_dotenv

# Imports do sistema de licenças
try:
    from database import get_db, init_db, FreeLicense, LicenseActivation
    from free_license_endpoint import router as free_license_router
    LICENSE_SYSTEM_ENABLED = True
except ImportError as e:
    logger.warning(f"⚠️  Sistema de licenças não disponível: {e}")
    LICENSE_SYSTEM_ENABLED = False
    free_license_router = None

load_dotenv()
logger = logging.getLogger("wstrader")
logging.basicConfig(level=logging.INFO)

# ===================== APP =====================
app = FastAPI(title="WS Trader API", version="1.0.0")

# Incluir router de licenças gratuitas (se disponível)
if LICENSE_SYSTEM_ENABLED and free_license_router:
    app.include_router(free_license_router)
    logger.info("✅ Sistema de licenças gratuitas habilitado")


# ===================== HELPERS =====================
def _mask(key: str) -> str:
    if not key:
        return ""
    k = key.strip()
    if len(k) <= 10:
        return "*" * len(k)
    return f"{k[:6]}...{k[-4:]}"


def get_stripe_key_from_env() -> str:
    key = os.getenv("STRIPE_SECRET_KEY", "").strip()
    return key


def validate_stripe_key(key: str) -> dict:
    """
    Valida se a key realmente autentica no Stripe.
    Retorna infos úteis sem vazar segredo.
    """
    if not key or key.strip() == "":
        raise ValueError("STRIPE_SECRET_KEY não encontrada no ambiente.")

    stripe.api_key = key.strip()

    try:
        # Chamada leve para testar autenticação
        acct = stripe.Account.retrieve()
        return {
            "ok": True,
            "account_id": acct.get("id"),
            "country": acct.get("country"),
            "charges_enabled": acct.get("charges_enabled"),
            "payouts_enabled": acct.get("payouts_enabled"),
            "key_masked": _mask(key),
        }
    except stripe.error.AuthenticationError as e:
        return {
            "ok": False,
            "error": "Chave Stripe inválida (falha de autenticação).",
            "detail": str(e),
            "key_masked": _mask(key),
        }
    except stripe.error.StripeError as e:
        return {
            "ok": False,
            "error": "Erro Stripe ao validar chave.",
            "detail": str(e),
            "key_masked": _mask(key),
        }


# ===================== STARTUP =====================
@app.on_event("startup")
def startup_check():
    # Inicializar banco de dados (se disponível)
    if LICENSE_SYSTEM_ENABLED:
        try:
            init_db()
            logger.info("✅ Database initialized")
        except Exception as e:
            logger.warning(f"⚠️  Erro ao inicializar database: {e}")

    # Verificar Stripe key
    key = get_stripe_key_from_env()
    if not key:
        logger.warning("⚠️  STRIPE_SECRET_KEY não configurada no ambiente.")
        return

    result = validate_stripe_key(key)
    if not result.get("ok"):
        logger.warning(f"⚠️  STRIPE_SECRET_KEY configurada, mas inválida. Detalhe: {result.get('detail')}")
    else:
        logger.info(f"✅ Stripe autenticado. Conta: {result.get('account_id')} | Key: {result.get('key_masked')}")


# ===================== ROOT ENDPOINT =====================
@app.get("/")
def root():
    return {
        "message": "🚀 API WS Trader Online",
        "version": "1.0.0",
        "endpoints": {
            "stripe_status": "/stripe/status",
            "stripe_validate_key": "/stripe/validate_key",
            "check_subscription": "/check_subscription",
            "free_license_validate": "/api/license/validate_free",
            "free_license_check": "/api/license/check/{key}",
            "docs": "/docs"
        }
    }


# ===================== STRIPE ENDPOINTS =====================
@app.get("/stripe/status")
def stripe_status():
    """
    Útil pra confirmar se no servidor (terminal/render) está tudo ok.
    """
    key = get_stripe_key_from_env()
    if not key:
        return {"ok": False, "configured": False, "message": "STRIPE_SECRET_KEY não está no ambiente."}

    result = validate_stripe_key(key)
    return {
        "ok": bool(result.get("ok")),
        "configured": True,
        "stripe": result,
    }


class StripeKeyCheck(BaseModel):
    stripe_secret_key: str


@app.post("/stripe/validate_key")
def stripe_validate_key(body: StripeKeyCheck):
    """
    Endpoint para testar uma chave digitada.
    ⚠️ Evite usar isso em produção aberto ao público.
    Ideal: proteger com senha/token/admin.
    """
    result = validate_stripe_key(body.stripe_secret_key)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


class SubscriptionCheck(BaseModel):
    email: str


@app.post("/check_subscription")
def check_subscription(data: SubscriptionCheck):
    """
    Checa se usuário tem assinatura ativa no Stripe
    """
    key = get_stripe_key_from_env()
    if not key:
        raise HTTPException(status_code=500, detail="❌ Stripe não configurado no backend.")

    stripe.api_key = key

    try:
        customers = stripe.Customer.list(email=data.email).data
        if not customers:
            return {"status": "inactive", "message": "❌ Cliente não encontrado no Stripe."}

        customer_id = customers[0].id
        subscriptions = stripe.Subscription.list(customer=customer_id).data

        active_subscription = next(
            (sub for sub in subscriptions if sub.status in ["active", "trialing"]),
            None
        )

        if active_subscription:
            status = "trial" if active_subscription.status == "trialing" else "active"
            trial_end = active_subscription.trial_end
            return {
                "status": status,
                "trial_end": trial_end,
                "message": "✅ Assinatura ativa." if status == "active" else f"🕒 Período de teste até {trial_end}"
            }

        return {"status": "inactive", "message": "❌ Assinatura inativa ou não encontrada."}

    except stripe.error.StripeError as e:
        raise HTTPException(status_code=500, detail=f"💥 Erro Stripe: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"💥 Erro interno no servidor: {str(e)}")


# ===================== HEALTH CHECK =====================
@app.get("/health")
def health_check():
    """Health check detalhado"""
    try:
        # Testar se tem Stripe configurado
        key = get_stripe_key_from_env()
        stripe_ok = bool(key)
    except:
        stripe_ok = False

    return {
        "status": "online",
        "stripe_configured": stripe_ok,
        "version": "1.0.0"
    }
