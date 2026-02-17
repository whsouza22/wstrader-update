"""
Módulo de segurança: JWT, hashing, device validation, horário
"""
from datetime import datetime, timedelta, time
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
from dotenv import load_dotenv

load_dotenv()

# ===================== CONFIG =====================
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_THIS_SECRET_KEY_IN_PRODUCTION")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))

ALLOWED_START_HOUR = int(os.getenv("ALLOWED_START_HOUR", "3"))
ALLOWED_END_HOUR = int(os.getenv("ALLOWED_END_HOUR", "15"))

# ===================== PASSWORD HASHING =====================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash de senha usando bcrypt"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifica senha contra hash"""
    return pwd_context.verify(plain_password, hashed_password)


# ===================== JWT TOKENS =====================

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Cria access token JWT (curto, 15 min)
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """
    Cria refresh token JWT (longo, 30 dias)
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str, expected_type: str = "access") -> Optional[Dict[str, Any]]:
    """
    Verifica e decodifica token JWT

    Args:
        token: Token JWT
        expected_type: "access" ou "refresh"

    Returns:
        Payload do token ou None se inválido
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

        # Verifica tipo de token
        token_type = payload.get("type")
        if token_type != expected_type:
            return None

        return payload

    except JWTError:
        return None


# ===================== HORÁRIO DE OPERAÇÃO =====================

def is_trading_hours(now: Optional[datetime] = None) -> bool:
    """
    Verifica se está no horário permitido para operar (03:00 - 15:00)

    Args:
        now: Datetime para verificar (default: agora)

    Returns:
        True se está no horário permitido
    """
    if now is None:
        now = datetime.now()

    current_hour = now.hour

    # Permite operar das ALLOWED_START_HOUR até ALLOWED_END_HOUR (exclusive)
    # Ex: 3 <= hora < 15 (03:00 até 14:59)
    return ALLOWED_START_HOUR <= current_hour < ALLOWED_END_HOUR


def get_next_trading_window() -> Dict[str, Any]:
    """
    Retorna informações sobre a próxima janela de operação

    Returns:
        Dict com: in_window (bool), next_start (datetime), next_end (datetime)
    """
    now = datetime.now()
    current_hour = now.hour

    in_window = is_trading_hours(now)

    if in_window:
        # Está na janela, próximo fim é hoje
        next_end = now.replace(hour=ALLOWED_END_HOUR, minute=0, second=0, microsecond=0)
        next_start = now  # já está dentro
    else:
        # Fora da janela
        if current_hour < ALLOWED_START_HOUR:
            # Antes do início de hoje
            next_start = now.replace(hour=ALLOWED_START_HOUR, minute=0, second=0, microsecond=0)
        else:
            # Depois do fim de hoje, próximo início é amanhã
            next_start = (now + timedelta(days=1)).replace(
                hour=ALLOWED_START_HOUR, minute=0, second=0, microsecond=0
            )

        next_end = next_start.replace(hour=ALLOWED_END_HOUR, minute=0, second=0, microsecond=0)

    return {
        "in_window": in_window,
        "next_start": next_start,
        "next_end": next_end,
        "current_time": now
    }


# ===================== DEVICE VALIDATION =====================

def validate_device_id(device_id: str) -> bool:
    """
    Valida formato do device_id (deve ser UUID v4)

    Args:
        device_id: ID do dispositivo

    Returns:
        True se válido
    """
    if not device_id or len(device_id) < 32:
        return False

    # Aceita UUID com ou sem hífens
    import re
    uuid_pattern = r'^[0-9a-f]{8}-?[0-9a-f]{4}-?4[0-9a-f]{3}-?[89ab][0-9a-f]{3}-?[0-9a-f]{12}$'
    return bool(re.match(uuid_pattern, device_id.lower()))


# ===================== RATE LIMITING =====================

class RateLimiter:
    """
    Rate limiter simples em memória (para produção use Redis)
    """
    def __init__(self):
        self._attempts: Dict[str, list] = {}

    def check_rate_limit(
        self,
        key: str,
        max_attempts: int = 5,
        window_seconds: int = 300
    ) -> tuple[bool, int]:
        """
        Verifica se ultrapassou rate limit

        Args:
            key: Identificador (IP, email, etc)
            max_attempts: Máximo de tentativas
            window_seconds: Janela de tempo em segundos

        Returns:
            (permitido: bool, tentativas_restantes: int)
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=window_seconds)

        # Limpa tentativas antigas
        if key in self._attempts:
            self._attempts[key] = [
                ts for ts in self._attempts[key] if ts > cutoff
            ]
        else:
            self._attempts[key] = []

        # Verifica limite
        current_attempts = len(self._attempts[key])

        if current_attempts >= max_attempts:
            return False, 0

        # Registra tentativa
        self._attempts[key].append(now)

        remaining = max_attempts - (current_attempts + 1)
        return True, remaining

    def reset(self, key: str):
        """Remove rate limit para uma chave"""
        if key in self._attempts:
            del self._attempts[key]


# Instância global (em produção, use Redis)
rate_limiter = RateLimiter()
