"""
API Backend Simplificada - WS Trader
Endpoints: Stripe check + Free License validation
"""
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
from datetime import datetime
import stripe
import os
from dotenv import load_dotenv

# Importar database e modelos
from database import get_db, init_db, FreeLicense, LicenseActivation

# Importar router de licenças gratuitas
from free_license_endpoint import router as free_license_router

# Carregar variáveis de ambiente
load_dotenv()

# Configurar Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")

if not stripe.api_key or stripe.api_key.strip() == "":
    print("⚠️  WARNING: Stripe API key não configurada!")

# Criar app
app = FastAPI(
    title="WS Trader API",
    description="API para validação de Stripe e Licenças Gratuitas",
    version="1.0.0"
)

# Adicionar CORS (permitir requisições do app)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especifique os domínios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir router de licenças gratuitas
app.include_router(free_license_router)


# ===================== STARTUP =====================
@app.on_event("startup")
async def startup():
    """Inicializa banco de dados"""
    init_db()
    print("✅ Database initialized")


# ===================== MODELS =====================
class SubscriptionCheck(BaseModel):
    """Request para checar assinatura no Stripe"""
    email: str


# ===================== STRIPE ENDPOINTS =====================

@app.post("/check_subscription")
def check_subscription(data: SubscriptionCheck):
    """
    Endpoint original: checa se usuário tem assinatura ativa no Stripe
    """
    try:
        # Procura cliente pelo email
        customers = stripe.Customer.list(email=data.email).data

        if not customers:
            return {
                "status": "inactive",
                "message": "❌ Cliente não encontrado no Stripe."
            }

        customer_id = customers[0].id
        subscriptions = stripe.Subscription.list(customer=customer_id).data

        # Procura assinatura ativa ou em trial
        active_subscription = next(
            (sub for sub in subscriptions if sub.status in ["active", "trialing"]),
            None
        )

        if active_subscription:
            status = "trial" if active_subscription.status == "trialing" else "active"
            trial_end = active_subscription.trial_end

            # Extrair product_id da assinatura
            product_id = ""
            try:
                items = active_subscription.get("items", {}).get("data", [])
                if items:
                    product_id = items[0].get("price", {}).get("product", "")
            except Exception:
                pass

            # Determinar plan_type baseado no product_id
            PRO_PRODUCT_ID = "prod_S4t8FQuUptWQ6R"
            DEMO_PRODUCT_ID = "prod_U3CRqZJMVigJAK"
            if product_id == PRO_PRODUCT_ID:
                plan_type = "PRO"
            elif product_id == DEMO_PRODUCT_ID:
                plan_type = "DEMO"
            else:
                plan_type = "DEMO"  # Qualquer outro produto = DEMO

            return {
                "status": status,
                "trial_end": trial_end,
                "product_id": product_id,
                "plan_type": plan_type,
                "valid": True,
                "is_active": True,
                "message": "✅ Assinatura ativa." if status == "active" else f"🕒 Período de teste até {trial_end}"
            }
        else:
            return {
                "status": "inactive",
                "message": "❌ Assinatura inativa ou não encontrada."
            }

    except stripe.error.StripeError as e:
        raise HTTPException(status_code=500, detail=f"💥 Erro Stripe: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"💥 Erro interno: {str(e)}")


# ===================== ROOT ENDPOINT =====================

@app.get("/")
def root():
    """
    Health check - verifica se API está online
    """
    return {
        "message": "🚀 API WS Trader Online",
        "version": "1.0.0",
        "endpoints": {
            "stripe": "/check_subscription",
            "free_license_validate": "/api/license/validate_free",
            "free_license_check": "/api/license/check/{license_key}",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    """
    Health check detalhado
    """
    return {
        "status": "online",
        "timestamp": datetime.utcnow().isoformat(),
        "stripe_configured": bool(stripe.api_key and stripe.api_key.strip()),
        "database": "connected"
    }


# ===================== ADMIN ENDPOINTS (OPCIONAL) =====================

@app.get("/admin/licenses/list")
def list_free_licenses(db: Session = Depends(get_db)):
    """
    Lista todas as licenças gratuitas (para admin)
    """
    licenses = db.query(FreeLicense).all()

    return {
        "total": len(licenses),
        "licenses": [
            {
                "license_key": lic.license_key,
                "user_email": lic.user_email,
                "is_active": lic.is_active,
                "activations": f"{lic.current_activations}/{lic.max_activations}",
                "created_at": lic.created_at.isoformat()
            }
            for lic in licenses
        ]
    }


@app.post("/admin/licenses/create")
def create_free_license(
    license_key: str,
    user_email: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Cria uma nova licença gratuita (para admin)
    """
    # Verificar se já existe
    existing = db.query(FreeLicense).filter(
        FreeLicense.license_key == license_key.upper()
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="License key already exists")

    # Criar nova licença
    new_license = FreeLicense(
        license_key=license_key.upper(),
        user_email=user_email,
        max_activations=1,
        current_activations=0,
        is_active=True,
        expires_at=None
    )

    db.add(new_license)
    db.commit()
    db.refresh(new_license)

    return {
        "message": "✅ License created successfully",
        "license_key": new_license.license_key,
        "max_activations": new_license.max_activations
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
