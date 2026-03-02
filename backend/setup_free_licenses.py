"""
Script para criar tabelas de licenças gratuitas e gerar 5 chaves
Execute este script UMA VEZ para configurar o sistema
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import engine, SessionLocal, Base
from free_license_models import FreeLicense, LicenseActivation
import secrets
import string


def generate_license_key():
    """Gera uma chave de licença única (formato: 32 caracteres hexadecimais)"""
    return secrets.token_hex(16)  # 32 caracteres


def create_tables():
    """Cria as tabelas no banco de dados"""
    print("Criando tabelas de licenças gratuitas...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tabelas criadas com sucesso!")


def generate_free_licenses(count=5):
    """Gera chaves de licença gratuitas"""
    db = SessionLocal()

    try:
        print(f"\nGerando {count} chaves de licença gratuitas...")
        print("=" * 60)

        generated_keys = []

        for i in range(count):
            # Gerar chave única
            while True:
                license_key = generate_license_key().upper()
                # Verificar se já existe
                existing = db.query(FreeLicense).filter(
                    FreeLicense.license_key == license_key
                ).first()
                if not existing:
                    break

            # Criar licença
            new_license = FreeLicense(
                license_key=license_key,
                user_email=None,  # Será preenchido depois ou deixado vazio
                max_activations=1,  # 1 computador por chave
                current_activations=0,
                is_active=True,
                expires_at=None  # Sem expiração
            )

            db.add(new_license)
            generated_keys.append(license_key)

            print(f"Chave {i+1}: {license_key}")

        db.commit()
        print("=" * 60)
        print(f"✅ {count} chaves criadas com sucesso!")
        print("\nDistribua estas chaves para seus 5 usuários.")
        print("Cada chave pode ser usada em apenas 1 computador.\n")

        return generated_keys

    except Exception as e:
        db.rollback()
        print(f"❌ Erro ao gerar chaves: {e}")
        return []
    finally:
        db.close()


def insert_specific_key(license_key: str):
    """Insere uma chave específica no banco"""
    db = SessionLocal()

    try:
        # Verificar se já existe
        existing = db.query(FreeLicense).filter(
            FreeLicense.license_key == license_key
        ).first()

        if existing:
            print(f"⚠️  Chave {license_key} já existe no banco!")
            return False

        # Criar licença
        new_license = FreeLicense(
            license_key=license_key.upper(),
            user_email=None,
            max_activations=1,
            current_activations=0,
            is_active=True,
            expires_at=None
        )

        db.add(new_license)
        db.commit()
        print(f"✅ Chave {license_key} adicionada com sucesso!")
        return True

    except Exception as e:
        db.rollback()
        print(f"❌ Erro ao adicionar chave: {e}")
        return False
    finally:
        db.close()


def list_all_licenses():
    """Lista todas as licenças no banco"""
    db = SessionLocal()

    try:
        licenses = db.query(FreeLicense).all()

        if not licenses:
            print("Nenhuma licença encontrada no banco.")
            return

        print(f"\n📋 Total de licenças: {len(licenses)}")
        print("=" * 80)

        for lic in licenses:
            print(f"Chave: {lic.license_key}")
            print(f"  Email: {lic.user_email or 'N/A'}")
            print(f"  Ativações: {lic.current_activations}/{lic.max_activations}")
            print(f"  Ativa: {'Sim' if lic.is_active else 'Não'}")
            print(f"  Criada em: {lic.created_at}")
            print("-" * 80)

    except Exception as e:
        print(f"❌ Erro ao listar licenças: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  SETUP DE LICENÇAS GRATUITAS - WS TRADER")
    print("=" * 60)

    # Menu
    print("\nO que deseja fazer?")
    print("1. Criar tabelas no banco de dados")
    print("2. Gerar 5 chaves automáticas")
    print("3. Adicionar chave específica")
    print("4. Listar todas as licenças")
    print("5. Fazer tudo (criar tabelas + gerar chaves)")

    choice = input("\nEscolha (1-5): ").strip()

    if choice == "1":
        create_tables()

    elif choice == "2":
        generate_free_licenses(5)

    elif choice == "3":
        license_key = input("Digite a chave de licença: ").strip()
        insert_specific_key(license_key)

    elif choice == "4":
        list_all_licenses()

    elif choice == "5":
        create_tables()
        print()
        generate_free_licenses(5)

    else:
        print("Opção inválida!")
