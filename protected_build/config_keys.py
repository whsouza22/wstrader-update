"""
Arquivo de chaves de API - Carrega de variáveis de ambiente
Este arquivo é incluído no executável pelo PyInstaller
"""
import os

# Claude / Anthropic
CLAUDE_API_KEY_1 = os.getenv("CLAUDE_API_KEY_1", "")
CLAUDE_API_KEY_2 = os.getenv("CLAUDE_API_KEY_2", "")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Stripe
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
