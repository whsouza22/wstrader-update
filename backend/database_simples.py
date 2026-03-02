"""
Database setup simplificado - apenas para licenças gratuitas
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# URL do banco (SQLite local ou PostgreSQL do Render)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./wstrader_licenses.db")

# Fix para PostgreSQL no Render (troca postgresql:// por postgresql+psycopg2://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ===================== MODELS =====================

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


# ===================== FUNCTIONS =====================

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
