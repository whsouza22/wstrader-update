"""
Modelos de banco de dados para licenças gratuitas
Adicione estas tabelas ao seu banco de dados existente
"""
from sqlalchemy import Column, String, Integer, DateTime, Boolean, JSON
from datetime import datetime
from database import Base


class FreeLicense(Base):
    """Licença gratuita - cada chave pode ser usada em 1 computador"""
    __tablename__ = "free_licenses"

    license_key = Column(String, primary_key=True, index=True)
    user_email = Column(String, nullable=True)  # Email opcional do usuário
    max_activations = Column(Integer, default=1)  # 1 computador por chave
    current_activations = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)


class LicenseActivation(Base):
    """Registro de ativação de licença (vinculada ao hardware)"""
    __tablename__ = "license_activations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    license_key = Column(String, index=True, nullable=False)
    hwid = Column(String, index=True, unique=True, nullable=False)  # Hardware ID único
    machine_info = Column(JSON, nullable=True)  # Informações da máquina
    activated_at = Column(DateTime, default=datetime.utcnow)
    last_validated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
