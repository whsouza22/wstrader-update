"""
Database setup e models para o sistema de licenças
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Float, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./wstrader_licenses.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ===================== MODELS =====================

class User(Base):
    """Usuário do sistema"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    telegram_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relacionamentos
    subscription = relationship("Subscription", back_populates="user", uselist=False)
    devices = relationship("Device", back_populates="user")
    sessions = relationship("Session", back_populates="user")
    heartbeats = relationship("Heartbeat", back_populates="user")
    broker_link = relationship("BrokerLink", back_populates="user", uselist=False)


class Subscription(Base):
    """Assinatura/Licença do usuário"""
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)

    # Status: trial, active, past_due, canceled, expired
    status = Column(String, default="trial", index=True)

    stripe_customer_id = Column(String, nullable=True)
    stripe_subscription_id = Column(String, nullable=True)

    trial_ends_at = Column(DateTime, nullable=True)
    current_period_start = Column(DateTime, nullable=True)
    current_period_end = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relacionamento
    user = relationship("User", back_populates="subscription")


class Device(Base):
    """Dispositivos autorizados (1 por usuário)"""
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    device_id = Column(String, unique=True, index=True, nullable=False)  # UUID gerado no app
    device_name = Column(String, nullable=True)  # Nome/hostname do PC

    is_active = Column(Boolean, default=True)

    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relacionamento
    user = relationship("User", back_populates="devices")


class BrokerLink(Base):
    """Link entre email do usuário e email da corretora"""
    __tablename__ = "broker_links"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)

    broker_email = Column(String, nullable=False)
    verified = Column(Boolean, default=False)
    verified_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relacionamento
    user = relationship("User", back_populates="broker_link")


class Session(Base):
    """Sessões ativas (refresh tokens)"""
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    device_id = Column(String, ForeignKey("devices.device_id"), nullable=False)

    refresh_token_hash = Column(String, unique=True, nullable=False)

    revoked = Column(Boolean, default=False)
    expires_at = Column(DateTime, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relacionamento
    user = relationship("User", back_populates="sessions")


class Heartbeat(Base):
    """Registro de heartbeats (para detectar desconexão)"""
    __tablename__ = "heartbeats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    device_id = Column(String, nullable=False)

    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    is_trading_hours = Column(Boolean, default=False)  # Se estava no horário permitido

    # Relacionamento
    user = relationship("User", back_populates="heartbeats")


class AuditLog(Base):
    """Logs de auditoria (segurança)"""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)

    event_type = Column(String, index=True)  # login, bind_device, start_bot, stop_bot, failed_login
    ip_address = Column(String, nullable=True)
    device_id = Column(String, nullable=True)

    details = Column(String, nullable=True)  # JSON string

    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class FreeLicense(Base):
    """Licença gratuita - cada chave pode ser usada em 1 computador"""
    __tablename__ = "free_licenses"

    license_key = Column(String, primary_key=True, index=True)
    user_email = Column(String, nullable=True)
    max_activations = Column(Integer, default=1)
    current_activations = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)


class LicenseActivation(Base):
    """Registro de ativação de licença (vinculada ao hardware)"""
    __tablename__ = "license_activations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    license_key = Column(String, index=True, nullable=False)
    hwid = Column(String, index=True, unique=True, nullable=False)
    machine_info = Column(JSON, nullable=True)
    activated_at = Column(DateTime, default=datetime.utcnow)
    last_validated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)


# Criar todas as tabelas
def init_db():
    """Inicializa o banco de dados"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency para FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
