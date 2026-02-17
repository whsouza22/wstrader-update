"""
API SIMPLES - Lê chaves do .env e verifica se já foram usadas
"""
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# ===================== DATABASE =====================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./licenses.db")

# Fix para PostgreSQL no Render
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ===================== MODEL =====================
class LicenseUsage(Base):
    """Registra quais chaves já foram usadas"""
    __tablename__ = "license_usage"

    license_key = Column(String, primary_key=True, index=True)
    used_at = Column(DateTime, default=datetime.utcnow)
    used_by_email = Column(String, nullable=True)

# Criar tabelas
Base.metadata.create_all(bind=engine)

# ===================== APP =====================
app = FastAPI(title="WS Trader API - Simples", version="1.0.0")

class LicenseCheckRequest(BaseModel):
    license_key: str
    email: str = None

# ===================== FUNÇÕES =====================
def get_valid_keys():
    """Lê as chaves válidas das variáveis de ambiente"""
    keys = []
    for i in range(1, 6):
        key = os.getenv(f"FreeLicense.license_key{i}", "").strip().upper()
        if key:
            keys.append(key)
    return keys

# ===================== ENDPOINTS =====================
@app.get("/")
def root():
    valid_keys = get_valid_keys()
    return {
        "message": "🚀 API WS Trader Online",
        "version": "1.0.0",
        "total_licenses": len(valid_keys),
        "endpoints": {
            "check_license": "POST /api/license/check",
            "docs": "/docs"
        }
    }

@app.post("/api/license/check")
def check_license(request: LicenseCheckRequest):
    """
    Verifica se a chave:
    1. Está na lista de chaves válidas (variáveis de ambiente)
    2. Ainda não foi usada (não está no banco)
    """
    db = SessionLocal()
    license_key = request.license_key.strip().upper()

    try:
        # 1. Verificar se a chave está nas variáveis de ambiente
        valid_keys = get_valid_keys()

        if license_key not in valid_keys:
            return {
                "valid": False,
                "message": "❌ Chave inválida"
            }

        # 2. Verificar se já foi usada
        usage = db.query(LicenseUsage).filter(
            LicenseUsage.license_key == license_key
        ).first()

        if usage:
            return {
                "valid": False,
                "message": "❌ Esta chave já foi utilizada"
            }

        # 3. Marcar como usada
        new_usage = LicenseUsage(
            license_key=license_key,
            used_at=datetime.utcnow(),
            used_by_email=request.email
        )
        db.add(new_usage)
        db.commit()

        return {
            "valid": True,
            "message": "✅ Licença ativada com sucesso!"
        }

    finally:
        db.close()

@app.get("/health")
def health():
    """Health check"""
    valid_keys = get_valid_keys()
    return {
        "status": "online",
        "licenses_configured": len(valid_keys)
    }

@app.get("/admin/status")
def admin_status():
    """Ver status das licenças (apenas para admin)"""
    db = SessionLocal()
    try:
        valid_keys = get_valid_keys()
        used_licenses = db.query(LicenseUsage).all()

        return {
            "total_licenses": len(valid_keys),
            "used_licenses": len(used_licenses),
            "available_licenses": len(valid_keys) - len(used_licenses),
            "used_keys": [
                {
                    "key": f"{lic.license_key[:8]}...{lic.license_key[-4:]}",
                    "used_at": lic.used_at.isoformat(),
                    "email": lic.used_by_email
                }
                for lic in used_licenses
            ]
        }
    finally:
        db.close()
