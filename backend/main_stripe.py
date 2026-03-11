"""
API WS Trader - Backend local (Stripe + Dashboard)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import sys
from pathlib import Path

# Stripe
try:
    import stripe
    STRIPE_ENABLED = True
except ImportError:
    STRIPE_ENABLED = False

# ===================== APP =====================
app = FastAPI(title="WS Trader API", version="2.0.0")

# CORS — permite o dashboard HTML acessar a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Dashboard API router ───
try:
    from dashboard_api import router as dashboard_router
    app.include_router(dashboard_router)
except ImportError:
    pass

# ─── Servir arquivos estáticos (img/, dashboard.html) ───
def _get_project_root():
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else Path(sys.executable).parent
    return Path(__file__).parent.parent

_root = _get_project_root()
_img_dir = _root / "img"
if _img_dir.exists():
    app.mount("/img", StaticFiles(directory=str(_img_dir)), name="img")

# ===================== ENDPOINTS =====================
@app.get("/")
def root():
    return {
        "message": "API WS Trader Online",
        "version": "2.0.0",
        "stripe": STRIPE_ENABLED,
        "endpoints": {
            "health": "/health",
            "check_subscription": "/check_subscription",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "online",
        "stripe": STRIPE_ENABLED,
    }

# ===================== STRIPE CHECK SUBSCRIPTION =====================
class SubscriptionCheckRequest(BaseModel):
    email: str

@app.post("/check_subscription")
def check_subscription(data: SubscriptionCheckRequest):
    """Verifica assinatura Stripe e retorna product_id"""
    print(f"[check_subscription] Email recebido: '{data.email}'")
    print(f"[check_subscription] STRIPE_ENABLED: {STRIPE_ENABLED}")
    
    if not STRIPE_ENABLED:
        raise HTTPException(status_code=503, detail="Stripe nao disponivel")

    stripe_key = os.getenv("STRIPE_SECRET_KEY", "")
    print(f"[check_subscription] Stripe key presente: {bool(stripe_key)}, len: {len(stripe_key)}")
    if not stripe_key:
        raise HTTPException(status_code=503, detail="STRIPE_SECRET_KEY nao configurada")

    stripe.api_key = stripe_key

    try:
        customers = stripe.Customer.list(email=data.email).data
        if not customers:
            return {"status": "inactive", "message": "Cliente nao encontrado no Stripe."}

        customer_id = customers[0].id
        subscriptions = stripe.Subscription.list(customer=customer_id).data

        active_subscription = next(
            (sub for sub in subscriptions if sub.status in ["active", "trialing"]),
            None
        )

        if active_subscription:
            status = "trial" if active_subscription.status == "trialing" else "active"
            trial_end = active_subscription.trial_end

            product_id = ""
            try:
                items = active_subscription["items"]["data"]
                if items:
                    product_id = items[0]["price"]["product"]
            except Exception:
                pass

            PRO_PRODUCT_ID = "prod_S4t8FQuUptWQ6R"
            DEMO_PRODUCT_ID = "prod_U3CRqZJMVigJAK"
            PREMIUM_PRODUCT_ID = "prod_U4ZxrEEApDg2Hb"
            if product_id == PREMIUM_PRODUCT_ID:
                plan_type = "PREMIUM"
            elif product_id == PRO_PRODUCT_ID:
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
                "message": "Assinatura ativa." if status == "active" else f"Periodo de teste ate {trial_end}"
            }
        else:
            return {"status": "inactive", "message": "Assinatura inativa ou nao encontrada."}

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[ERRO check_subscription] {type(e).__name__}: {e}")
        print(f"[TRACEBACK] {error_detail}")
        raise HTTPException(status_code=500, detail=f"Erro Stripe: {str(e)}\n{error_detail}")
