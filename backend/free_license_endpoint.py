"""
Endpoint para validação de licenças gratuitas
Adicione este código ao seu main.py
"""
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, Dict, Any
from database import get_db, FreeLicense, LicenseActivation

router = APIRouter(prefix="/api/license", tags=["Free License"])


class LicenseValidationRequest(BaseModel):
    """Request para validação de licença"""
    license_key: str
    hwid: str
    machine_info: Optional[Dict[str, Any]] = None


class LicenseValidationResponse(BaseModel):
    """Response da validação"""
    valid: bool
    message: str
    user_data: Optional[Dict[str, Any]] = None


@router.post("/validate_free", response_model=LicenseValidationResponse)
def validate_free_license(
    request: LicenseValidationRequest,
    db: Session = Depends(get_db)
):
    """
    Valida uma chave de licença gratuita.
    Cada chave pode ser usada em apenas 1 computador (1 HWID).
    """
    license_key = request.license_key.strip().upper()
    hwid = request.hwid.strip()

    # 1. Verificar se a chave existe
    license_obj = db.query(FreeLicense).filter(
        FreeLicense.license_key == license_key
    ).first()

    if not license_obj:
        return LicenseValidationResponse(
            valid=False,
            message="Chave de licença não encontrada",
            user_data=None
        )

    # 2. Verificar se a licença está ativa
    if not license_obj.is_active:
        return LicenseValidationResponse(
            valid=False,
            message="Esta licença foi desativada",
            user_data=None
        )

    # 3. Verificar se expirou
    if license_obj.expires_at and license_obj.expires_at < datetime.utcnow():
        return LicenseValidationResponse(
            valid=False,
            message="Esta licença expirou",
            user_data=None
        )

    # 4. Verificar se este HWID já está ativado com esta chave
    existing_activation = db.query(LicenseActivation).filter(
        LicenseActivation.license_key == license_key,
        LicenseActivation.hwid == hwid,
        LicenseActivation.is_active == True
    ).first()

    if existing_activation:
        # HWID já ativado - apenas atualizar última validação
        existing_activation.last_validated_at = datetime.utcnow()
        db.commit()

        return LicenseValidationResponse(
            valid=True,
            message="Licença válida",
            user_data={
                "license_key": license_key,
                "user_email": license_obj.user_email,
                "activation_number": license_obj.current_activations,
                "max_activations": license_obj.max_activations,
                "expires_at": license_obj.expires_at.isoformat() if license_obj.expires_at else None
            }
        )

    # 5. Verificar se este HWID já está usando OUTRA chave
    hwid_used_elsewhere = db.query(LicenseActivation).filter(
        LicenseActivation.hwid == hwid,
        LicenseActivation.is_active == True
    ).first()

    if hwid_used_elsewhere:
        return LicenseValidationResponse(
            valid=False,
            message="Este computador já está vinculado a outra licença",
            user_data=None
        )

    # 6. Verificar se atingiu o limite de ativações (1 por chave)
    if license_obj.current_activations >= license_obj.max_activations:
        return LicenseValidationResponse(
            valid=False,
            message="Esta chave já foi ativada em outro computador",
            user_data=None
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

    # 8. Incrementar contador de ativações
    license_obj.current_activations += 1
    db.commit()

    return LicenseValidationResponse(
        valid=True,
        message="Licença ativada com sucesso",
        user_data={
            "license_key": license_key,
            "user_email": license_obj.user_email,
            "activation_number": license_obj.current_activations,
            "max_activations": license_obj.max_activations,
            "expires_at": license_obj.expires_at.isoformat() if license_obj.expires_at else None
        }
    )


@router.get("/check/{license_key}")
def check_license_status(
    license_key: str,
    db: Session = Depends(get_db)
):
    """
    Verifica o status de uma licença (para admin)
    """
    license_key = license_key.strip().upper()

    license_obj = db.query(FreeLicense).filter(
        FreeLicense.license_key == license_key
    ).first()

    if not license_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Licença não encontrada"
        )

    activations = db.query(LicenseActivation).filter(
        LicenseActivation.license_key == license_key,
        LicenseActivation.is_active == True
    ).all()

    return {
        "license_key": license_obj.license_key,
        "user_email": license_obj.user_email,
        "is_active": license_obj.is_active,
        "current_activations": license_obj.current_activations,
        "max_activations": license_obj.max_activations,
        "created_at": license_obj.created_at.isoformat(),
        "expires_at": license_obj.expires_at.isoformat() if license_obj.expires_at else None,
        "activations": [
            {
                "hwid": act.hwid[:16] + "...",
                "machine_info": act.machine_info,
                "activated_at": act.activated_at.isoformat(),
                "last_validated_at": act.last_validated_at.isoformat()
            }
            for act in activations
        ]
    }
