"""
CÓDIGO DO SERVIDOR - Adicione isto ao seu servidor Render

Este código deve ser adicionado ao seu backend existente que já valida Stripe.
Ele gerencia as chaves gratuitas limitadas a 5 ativações.

CONFIGURAÇÃO:
1. Adicione este código ao seu servidor Render
2. Configure um banco de dados (SQLite ou PostgreSQL)
3. Gere chaves gratuitas e distribua para os 5 usuários
4. O sistema bloqueia automaticamente após 5 ativações diferentes
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import hashlib
from sqlalchemy import Column, String, Integer, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

# ==================== MODELS ====================

Base = declarative_base()


class FreeLicense(Base):
    """Modelo de licença gratuita no banco de dados"""
    __tablename__ = "free_licenses"

    license_key = Column(String, primary_key=True, index=True)
    user_email = Column(String, nullable=True)  # Email do usuário (opcional)
    max_activations = Column(Integer, default=5)  # Máximo de ativações
    current_activations = Column(Integer, default=0)  # Ativações atuais
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # Null = sem expiração


class LicenseActivation(Base):
    """Registro de cada ativação de licença"""
    __tablename__ = "license_activations"

    id = Column(Integer, primary_key=True, index=True)
    license_key = Column(String, index=True)
    hwid = Column(String, index=True, unique=True)  # Hardware ID único
    machine_info = Column(JSON, nullable=True)  # Info da máquina
    activated_at = Column(DateTime, default=datetime.utcnow)
    last_validated_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)


# ==================== SCHEMAS ====================

class LicenseValidationRequest(BaseModel):
    """Requisição de validação de licença"""
    license_key: str
    hwid: str
    machine_info: Optional[Dict[str, Any]] = None
    app_version: Optional[str] = None
    platform: Optional[str] = None
    timestamp: Optional[str] = None


class LicenseValidationResponse(BaseModel):
    """Resposta de validação"""
    valid: bool
    error: Optional[str] = None
    reason: Optional[str] = None
    user_email: Optional[str] = None
    activation_number: Optional[int] = None
    max_activations: Optional[int] = None
    activated_at: Optional[str] = None
    expires_at: Optional[str] = None


# ==================== ROUTER ====================

router = APIRouter(prefix="/api/license", tags=["License"])


@router.post("/validate_free", response_model=LicenseValidationResponse)
def validate_free_license(
    request: LicenseValidationRequest,
    db: Session = None  # Injetar dependência do DB
):
    """
    Valida uma chave de licença gratuita.

    Regras:
    1. Máximo de 5 ativações (HWIDs) diferentes
    2. Um HWID só pode ativar uma vez
    3. Licenças podem ser desativadas manualmente
    4. Validações atualizam last_validated_at
    """

    license_key = request.license_key.strip().upper()
    hwid = request.hwid.strip()

    # 1. Verificar se a licença existe
    license_obj = db.query(FreeLicense).filter(
        FreeLicense.license_key == license_key
    ).first()

    if not license_obj:
        return LicenseValidationResponse(
            valid=False,
            error="Chave de licença não encontrada",
            reason="A chave fornecida não existe no sistema"
        )

    # 2. Verificar se a licença está ativa
    if not license_obj.is_active:
        return LicenseValidationResponse(
            valid=False,
            error="Licença desativada",
            reason="Esta licença foi desativada pelo administrador"
        )

    # 3. Verificar expiração (se houver)
    if license_obj.expires_at and datetime.utcnow() > license_obj.expires_at:
        return LicenseValidationResponse(
            valid=False,
            error="Licença expirada",
            reason=f"Esta licença expirou em {license_obj.expires_at.isoformat()}"
        )

    # 4. Verificar se este HWID já está ativado
    existing_activation = db.query(LicenseActivation).filter(
        LicenseActivation.license_key == license_key,
        LicenseActivation.hwid == hwid,
        LicenseActivation.is_active == True
    ).first()

    if existing_activation:
        # HWID já ativado - atualizar last_validated_at e permitir
        existing_activation.last_validated_at = datetime.utcnow()
        db.commit()

        activation_number = db.query(LicenseActivation).filter(
            LicenseActivation.license_key == license_key,
            LicenseActivation.is_active == True
        ).count()

        return LicenseValidationResponse(
            valid=True,
            user_email=license_obj.user_email,
            activation_number=activation_number,
            max_activations=license_obj.max_activations,
            activated_at=existing_activation.activated_at.isoformat(),
            expires_at=license_obj.expires_at.isoformat() if license_obj.expires_at else None
        )

    # 5. Nova ativação - verificar limite
    current_active_count = db.query(LicenseActivation).filter(
        LicenseActivation.license_key == license_key,
        LicenseActivation.is_active == True
    ).count()

    if current_active_count >= license_obj.max_activations:
        return LicenseValidationResponse(
            valid=False,
            error="Limite de ativações atingido",
            reason=f"Esta licença já está ativa em {license_obj.max_activations} computadores diferentes"
        )

    # 6. Verificar se este HWID não está usando outra licença gratuita
    hwid_in_other_license = db.query(LicenseActivation).filter(
        LicenseActivation.hwid == hwid,
        LicenseActivation.license_key != license_key,
        LicenseActivation.is_active == True
    ).first()

    if hwid_in_other_license:
        return LicenseValidationResponse(
            valid=False,
            error="Hardware já vinculado a outra licença",
            reason="Este computador já está usando outra chave de licença gratuita"
        )

    # 7. Criar nova ativação
    new_activation = LicenseActivation(
        license_key=license_key,
        hwid=hwid,
        machine_info=request.machine_info,
        activated_at=datetime.utcnow(),
        last_validated_at=datetime.utcnow(),
        is_active=True
    )

    db.add(new_activation)

    # Atualizar contador de ativações
    license_obj.current_activations = current_active_count + 1

    db.commit()

    return LicenseValidationResponse(
        valid=True,
        user_email=license_obj.user_email,
        activation_number=license_obj.current_activations,
        max_activations=license_obj.max_activations,
        activated_at=new_activation.activated_at.isoformat(),
        expires_at=license_obj.expires_at.isoformat() if license_obj.expires_at else None
    )


@router.get("/health")
def health_check():
    """Health check do serviço de licenças"""
    return {"status": "ok", "service": "license_validation"}


# ==================== FUNÇÕES AUXILIARES ====================

def generate_free_license_key(prefix="FREE"):
    """
    Gera uma chave de licença gratuita única.
    Formato: FREE-XXXXX-XXXXX-XXXXX
    """
    import secrets
    import string

    def random_segment():
        return ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(5))

    return f"{prefix}-{random_segment()}-{random_segment()}-{random_segment()}"


def create_free_license(db: Session, user_email: Optional[str] = None, max_activations: int = 5):
    """
    Cria uma nova licença gratuita no banco.

    Args:
        db: Sessão do banco de dados
        user_email: Email do usuário (opcional)
        max_activations: Número máximo de ativações (padrão: 5)

    Returns:
        str: Chave de licença gerada
    """
    license_key = generate_free_license_key()

    # Verificar se já existe (improvável, mas por segurança)
    while db.query(FreeLicense).filter(FreeLicense.license_key == license_key).first():
        license_key = generate_free_license_key()

    new_license = FreeLicense(
        license_key=license_key,
        user_email=user_email,
        max_activations=max_activations,
        current_activations=0,
        is_active=True,
        created_at=datetime.utcnow(),
        expires_at=None  # Sem expiração para licenças gratuitas
    )

    db.add(new_license)
    db.commit()

    return license_key


def deactivate_license(db: Session, license_key: str):
    """Desativa uma licença (impede novas ativações)"""
    license_obj = db.query(FreeLicense).filter(
        FreeLicense.license_key == license_key
    ).first()

    if license_obj:
        license_obj.is_active = False
        db.commit()
        return True
    return False


def revoke_hwid_activation(db: Session, hwid: str):
    """Remove a ativação de um hardware específico"""
    activation = db.query(LicenseActivation).filter(
        LicenseActivation.hwid == hwid,
        LicenseActivation.is_active == True
    ).first()

    if activation:
        activation.is_active = False

        # Atualizar contador da licença
        license_obj = db.query(FreeLicense).filter(
            FreeLicense.license_key == activation.license_key
        ).first()

        if license_obj:
            active_count = db.query(LicenseActivation).filter(
                LicenseActivation.license_key == activation.license_key,
                LicenseActivation.is_active == True
            ).count()
            license_obj.current_activations = active_count

        db.commit()
        return True
    return False


# ==================== SCRIPT DE GERAÇÃO DE LICENÇAS ====================

if __name__ == "__main__":
    """
    Script para gerar 5 chaves gratuitas.
    Execute isto uma vez para criar as chaves.
    """
    print("=" * 60)
    print("GERADOR DE CHAVES GRATUITAS - WS TRADER")
    print("=" * 60)
    print("\nGerando 5 chaves de licença gratuita...\n")

    # Simulação (sem DB - apenas gera as chaves)
    for i in range(5):
        key = generate_free_license_key()
        print(f"{i+1}. {key}")

    print("\n" + "=" * 60)
    print("INSTRUÇÕES:")
    print("1. Adicione estas chaves no banco de dados do seu servidor")
    print("2. Distribua UMA chave para cada usuário gratuito")
    print("3. Cada chave pode ser ativada em até 5 computadores diferentes")
    print("4. Após 5 ativações, a chave fica bloqueada")
    print("=" * 60)
