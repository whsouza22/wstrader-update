"""
API Backend para sistema de licenças Wstrader
Endpoints: auth, license, bot control, stripe webhook
"""
from fastapi import FastAPI, Depends, HTTPException, status, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import hashlib
import stripe
import os
from dotenv import load_dotenv

from database import get_db, init_db, User, Subscription, Device, Session as DBSession, Heartbeat, AuditLog, BrokerLink

# Try to import security functions (may not exist in simple setup)
try:
    from security import (
        hash_password,
        verify_password,
        create_access_token,
        create_refresh_token,
        verify_token,
        is_trading_hours,
        get_next_trading_window,
        validate_device_id,
        rate_limiter
    )
    SECURITY_ENABLED = True
except ImportError:
    SECURITY_ENABLED = False
    print("⚠️  Security module not found - some endpoints will be disabled")

from free_license_endpoint import router as free_license_router

# Pydantic models
class SubscriptionCheckRequest(BaseModel):
    email: str

load_dotenv()

# ===================== CONFIG =====================
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

# ===================== APP =====================
app = FastAPI(
    title="Wstrader License API",
    description="Sistema de licenciamento e autenticação para Wstrader Bot",
    version="1.0.0"
)

# CORS (ajuste para produção)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção: liste origens específicas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(free_license_router)


# ===================== STARTUP =====================
@app.on_event("startup")
async def startup():
    """Inicializa banco de dados"""
    init_db()
    print("✅ Database initialized")


# ===================== HELPERS =====================

def get_current_user(
    authorization: str = Header(...),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency: extrai usuário do token JWT Bearer
    """
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme"
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header"
        )

    payload = verify_token(token, expected_type="access")
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    return user


def log_audit(
    db: Session,
    event_type: str,
    user_id: Optional[int] = None,
    ip_address: Optional[str] = None,
    device_id: Optional[str] = None,
    details: Optional[str] = None
):
    """Registra evento de auditoria"""
    log = AuditLog(
        user_id=user_id,
        event_type=event_type,
        ip_address=ip_address,
        device_id=device_id,
        details=details
    )
    db.add(log)
    db.commit()


# ===================== AUTH ENDPOINTS =====================

@app.post("/auth/register")
async def register(
    email: str,
    password: str,
    broker_email: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Registra novo usuário (sem pagamento ainda = trial)
    """
    # Rate limit por IP
    client_ip = request.client.host
    allowed, remaining = rate_limiter.check_rate_limit(f"register:{client_ip}", max_attempts=3, window_seconds=3600)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many registration attempts. Try again later."
        )

    # Verifica se email já existe
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Cria usuário
    user = User(
        email=email,
        password_hash=hash_password(password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Cria subscription trial (7 dias)
    subscription = Subscription(
        user_id=user.id,
        status="trial",
        trial_ends_at=datetime.utcnow() + timedelta(days=7)
    )
    db.add(subscription)

    # Cria broker link (não verificado)
    broker_link = BrokerLink(
        user_id=user.id,
        broker_email=broker_email,
        verified=False
    )
    db.add(broker_link)

    db.commit()

    log_audit(db, "register", user_id=user.id, ip_address=client_ip)

    return {
        "message": "User registered successfully",
        "user_id": user.id,
        "email": user.email,
        "status": "trial",
        "trial_ends_at": subscription.trial_ends_at.isoformat()
    }


@app.post("/auth/login")
async def login(
    email: str,
    password: str,
    device_id: str,
    device_name: Optional[str] = None,
    request: Request = None,
    db: Session = Depends(get_db)
):
    """
    Login: valida credenciais, device, licença e retorna tokens
    """
    client_ip = request.client.host if request else None

    # Rate limit por IP
    allowed, remaining = rate_limiter.check_rate_limit(f"login:{client_ip}", max_attempts=5, window_seconds=300)
    if not allowed:
        log_audit(db, "failed_login_rate_limit", ip_address=client_ip, details=email)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Try again later."
        )

    # Valida device_id
    if not validate_device_id(device_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid device_id format (must be UUID v4)"
        )

    # Busca usuário
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        log_audit(db, "failed_login", ip_address=client_ip, device_id=device_id, details=email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Verifica subscription
    subscription = db.query(Subscription).filter(Subscription.user_id == user.id).first()
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No active subscription"
        )

    # Verifica status da licença
    now = datetime.utcnow()
    license_active = False

    if subscription.status == "active":
        # Verifica se não expirou
        if subscription.current_period_end and subscription.current_period_end > now:
            license_active = True
    elif subscription.status == "trial":
        # Verifica se trial não expirou
        if subscription.trial_ends_at and subscription.trial_ends_at > now:
            license_active = True

    if not license_active:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"License expired or inactive (status: {subscription.status})"
        )

    # Verifica device binding (1 device por usuário)
    existing_device = db.query(Device).filter(Device.user_id == user.id, Device.is_active == True).first()

    if existing_device:
        # Já tem device vinculado
        if existing_device.device_id != device_id:
            # Tentando logar de outro device
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"License already bound to another device ({existing_device.device_name or 'Unknown'}). Contact support to change device."
            )
        else:
            # Mesmo device, atualiza last_seen
            existing_device.last_seen = now
            db.commit()
    else:
        # Primeiro login, vincula device
        new_device = Device(
            user_id=user.id,
            device_id=device_id,
            device_name=device_name or "Unknown",
            is_active=True
        )
        db.add(new_device)
        db.commit()
        log_audit(db, "bind_device", user_id=user.id, device_id=device_id, ip_address=client_ip)

    # Gera tokens
    token_data = {
        "user_id": user.id,
        "email": user.email,
        "device_id": device_id
    }

    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    # Salva refresh token
    refresh_token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
    session = DBSession(
        user_id=user.id,
        device_id=device_id,
        refresh_token_hash=refresh_token_hash,
        expires_at=datetime.utcnow() + timedelta(days=30)
    )
    db.add(session)
    db.commit()

    log_audit(db, "login_success", user_id=user.id, device_id=device_id, ip_address=client_ip)

    # Reset rate limit após login bem-sucedido
    rate_limiter.reset(f"login:{client_ip}")

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "license_status": subscription.status,
        "device_bound": True,
        "user": {
            "id": user.id,
            "email": user.email
        }
    }


@app.post("/auth/refresh")
async def refresh_token_endpoint(
    refresh_token: str,
    db: Session = Depends(get_db)
):
    """
    Renova access token usando refresh token
    """
    payload = verify_token(refresh_token, expected_type="refresh")
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )

    # Verifica se refresh token existe e não foi revogado
    refresh_token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
    session = db.query(DBSession).filter(
        DBSession.refresh_token_hash == refresh_token_hash,
        DBSession.revoked == False,
        DBSession.expires_at > datetime.utcnow()
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token revoked or expired"
        )

    # Gera novo access token
    token_data = {
        "user_id": payload["user_id"],
        "email": payload["email"],
        "device_id": payload["device_id"]
    }

    new_access_token = create_access_token(token_data)

    return {
        "access_token": new_access_token,
        "token_type": "bearer"
    }


# ===================== LICENSE ENDPOINTS =====================

@app.get("/license/status")
async def get_license_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retorna status completo da licença do usuário
    """
    subscription = db.query(Subscription).filter(Subscription.user_id == current_user.id).first()
    device = db.query(Device).filter(Device.user_id == current_user.id, Device.is_active == True).first()

    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No subscription found"
        )

    # Verifica trading hours
    trading_window = get_next_trading_window()

    return {
        "user_id": current_user.id,
        "email": current_user.email,
        "license": {
            "status": subscription.status,
            "trial_ends_at": subscription.trial_ends_at.isoformat() if subscription.trial_ends_at else None,
            "current_period_end": subscription.current_period_end.isoformat() if subscription.current_period_end else None
        },
        "device": {
            "device_id": device.device_id if device else None,
            "device_name": device.device_name if device else None,
            "first_seen": device.first_seen.isoformat() if device else None,
            "last_seen": device.last_seen.isoformat() if device else None
        } if device else None,
        "trading_hours": {
            "allowed": trading_window["in_window"],
            "current_time": trading_window["current_time"].isoformat(),
            "next_start": trading_window["next_start"].isoformat(),
            "next_end": trading_window["next_end"].isoformat()
        }
    }


# ===================== BOT CONTROL ENDPOINTS =====================

@app.post("/bot/heartbeat")
async def bot_heartbeat(
    device_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Heartbeat: bot chama a cada 30s para confirmar que está autorizado a operar

    Retorna:
        - ok_to_trade: bool (se pode continuar operando)
        - reason: str (motivo se não puder)
    """
    # Verifica se device corresponde
    device = db.query(Device).filter(
        Device.user_id == current_user.id,
        Device.device_id == device_id,
        Device.is_active == True
    ).first()

    if not device:
        return {
            "ok_to_trade": False,
            "reason": "device_not_authorized"
        }

    # Atualiza last_seen
    device.last_seen = datetime.utcnow()

    # Verifica subscription
    subscription = db.query(Subscription).filter(Subscription.user_id == current_user.id).first()
    if not subscription or subscription.status not in ["active", "trial"]:
        return {
            "ok_to_trade": False,
            "reason": f"license_inactive (status: {subscription.status if subscription else 'none'})"
        }

    # Verifica se trial ou subscription expirou
    now = datetime.utcnow()
    if subscription.status == "trial" and subscription.trial_ends_at and subscription.trial_ends_at < now:
        subscription.status = "expired"
        db.commit()
        return {
            "ok_to_trade": False,
            "reason": "trial_expired"
        }

    if subscription.status == "active" and subscription.current_period_end and subscription.current_period_end < now:
        subscription.status = "expired"
        db.commit()
        return {
            "ok_to_trade": False,
            "reason": "subscription_expired"
        }

    # Verifica horário de operação
    in_trading_hours = is_trading_hours()

    # Registra heartbeat
    heartbeat = Heartbeat(
        user_id=current_user.id,
        device_id=device_id,
        is_trading_hours=in_trading_hours
    )
    db.add(heartbeat)
    db.commit()

    if not in_trading_hours:
        trading_window = get_next_trading_window()
        return {
            "ok_to_trade": False,
            "reason": "outside_trading_hours",
            "trading_hours": {
                "next_start": trading_window["next_start"].isoformat(),
                "next_end": trading_window["next_end"].isoformat()
            }
        }

    return {
        "ok_to_trade": True,
        "reason": "authorized",
        "license_status": subscription.status
    }


# ===================== STRIPE WEBHOOK =====================

@app.post("/stripe/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """
    Webhook do Stripe para atualizar status de assinaturas automaticamente
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Processa eventos
    event_type = event["type"]
    data = event["data"]["object"]

    if event_type == "checkout.session.completed":
        # Pagamento inicial bem-sucedido
        customer_email = data.get("customer_email")
        stripe_customer_id = data.get("customer")
        stripe_subscription_id = data.get("subscription")

        user = db.query(User).filter(User.email == customer_email).first()
        if user:
            subscription = db.query(Subscription).filter(Subscription.user_id == user.id).first()
            if subscription:
                subscription.status = "active"
                subscription.stripe_customer_id = stripe_customer_id
                subscription.stripe_subscription_id = stripe_subscription_id
                db.commit()

    elif event_type == "customer.subscription.updated":
        # Assinatura atualizada
        stripe_subscription_id = data.get("id")
        status = data.get("status")
        current_period_start = datetime.fromtimestamp(data.get("current_period_start"))
        current_period_end = datetime.fromtimestamp(data.get("current_period_end"))

        subscription = db.query(Subscription).filter(
            Subscription.stripe_subscription_id == stripe_subscription_id
        ).first()

        if subscription:
            subscription.status = "active" if status == "active" else status
            subscription.current_period_start = current_period_start
            subscription.current_period_end = current_period_end
            db.commit()

    elif event_type == "customer.subscription.deleted":
        # Assinatura cancelada
        stripe_subscription_id = data.get("id")

        subscription = db.query(Subscription).filter(
            Subscription.stripe_subscription_id == stripe_subscription_id
        ).first()

        if subscription:
            subscription.status = "canceled"
            db.commit()

            # Revoga todas as sessões do usuário
            db.query(DBSession).filter(
                DBSession.user_id == subscription.user_id
            ).update({"revoked": True})
            db.commit()

    elif event_type == "invoice.payment_failed":
        # Falha no pagamento
        stripe_subscription_id = data.get("subscription")

        subscription = db.query(Subscription).filter(
            Subscription.stripe_subscription_id == stripe_subscription_id
        ).first()

        if subscription:
            subscription.status = "past_due"
            db.commit()

    return {"status": "success"}


# ===================== STRIPE CHECK SUBSCRIPTION =====================

@app.post("/check_subscription")
async def check_subscription(data: SubscriptionCheckRequest):
    """
    Checa se usuário tem assinatura ativa no Stripe
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

            return {
                "status": status,
                "trial_end": trial_end,
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


# ===================== HEALTH CHECK =====================

@app.get("/")
async def root():
    """Health check"""
    return {
        "service": "Wstrader License API",
        "status": "online",
        "version": "1.0.0",
        "endpoints": {
            "stripe_check": "/check_subscription",
            "free_license_validate": "/api/license/validate_free",
            "free_license_check": "/api/license/check/{key}",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check com verificação de banco"""
    try:
        # Testa conexão com banco
        db.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {
        "status": "online",
        "database": db_status,
        "timestamp": datetime.utcnow().isoformat()
    }
