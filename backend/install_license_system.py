"""
Script para instalar o sistema de licenças no Render
Execute este script no Render Shell após fazer deploy dos arquivos
"""
import sys

def install():
    print("=" * 70)
    print("  INSTALAÇÃO DO SISTEMA DE LICENÇAS - WS TRADER")
    print("=" * 70)

    try:
        # 1. Importar modelos
        print("\n[1/3] Importando modelos do banco de dados...")
        from database import Base, engine, SessionLocal, FreeLicense, LicenseActivation
        print("✅ Modelos importados com sucesso!")

        # 2. Criar tabelas
        print("\n[2/3] Criando tabelas no banco de dados...")
        Base.metadata.create_all(bind=engine)
        print("✅ Tabelas 'free_licenses' e 'license_activations' criadas!")

        # 3. Adicionar chave
        print("\n[3/3] Adicionando chave de licença...")
        db = SessionLocal()

        license_key = "0E8D31699C0DCB497DD95A678D41A187"

        # Verificar se já existe
        existing = db.query(FreeLicense).filter(
            FreeLicense.license_key == license_key
        ).first()

        if existing:
            print(f"⚠️  Chave {license_key} já existe no banco!")
            print(f"   Status: {'Ativa' if existing.is_active else 'Inativa'}")
            print(f"   Ativações: {existing.current_activations}/{existing.max_activations}")
        else:
            # Criar nova licença
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
            print(f"✅ Chave {license_key} adicionada com sucesso!")

        db.close()

        # Resumo final
        print("\n" + "=" * 70)
        print("  ✅ INSTALAÇÃO CONCLUÍDA COM SUCESSO!")
        print("=" * 70)
        print("\nChave de licença:")
        print(f"  {license_key}")
        print("\nEndpoint disponível em:")
        print("  https://api-wstrader.onrender.com/api/license/validate_free")
        print("\nPróximo passo:")
        print("  Teste o endpoint no app WS Trader!")
        print("=" * 70)

        return True

    except ImportError as e:
        print(f"\n❌ Erro ao importar módulos: {e}")
        print("\n⚠️  Certifique-se de que você fez deploy dos arquivos:")
        print("   - free_license_endpoint.py")
        print("   - Modificações em main.py")
        print("   - Modificações em database.py")
        return False

    except Exception as e:
        print(f"\n❌ Erro durante instalação: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = install()
    sys.exit(0 if success else 1)
