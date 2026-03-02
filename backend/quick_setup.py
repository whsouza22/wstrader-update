"""
Script rápido para configurar licença no Render Shell
Cole este código diretamente no Python do Render Shell
"""

# Passo 1: Criar tabelas
print("Criando tabelas...")
from database import Base, engine, SessionLocal, FreeLicense, LicenseActivation
Base.metadata.create_all(bind=engine)
print("✅ Tabelas criadas!")

# Passo 2: Adicionar chave
print("\nAdicionando chave...")
db = SessionLocal()

license_key = "0E8D31699C0DCB497DD95A678D41A187"

existing = db.query(FreeLicense).filter(FreeLicense.license_key == license_key).first()

if existing:
    print(f"⚠️  Chave já existe! Ativações: {existing.current_activations}/{existing.max_activations}")
else:
    new_license = FreeLicense(
        license_key=license_key,
        user_email=None,
        max_activations=1,
        current_activations=0,
        is_active=True,
        expires_at=None
    )
    db.add(new_license)
    db.commit()
    print(f"✅ Chave {license_key} adicionada!")

db.close()

print("\n🎉 Configuração completa!")
print(f"Endpoint: https://api-wstrader.onrender.com/api/license/validate_free")
