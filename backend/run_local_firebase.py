"""
Servidor LOCAL para testar Firebase
Execute: python run_local_firebase.py
"""
import os
import uvicorn

# Configurar Firebase Credentials
print("=" * 70)
print("🔥 SERVIDOR LOCAL FIREBASE")
print("=" * 70)
print()

# Pedir caminho do arquivo de credenciais
cred_path = input("Digite o caminho do arquivo JSON do Firebase: ").strip()

if not os.path.exists(cred_path):
    print(f"❌ Arquivo não encontrado: {cred_path}")
    exit(1)

# Ler o arquivo e colocar em variável de ambiente
with open(cred_path, 'r') as f:
    cred_json = f.read()

os.environ['FIREBASE_CREDENTIALS'] = cred_json

print("✅ Credenciais carregadas!")
print()
print("🚀 Iniciando servidor em http://localhost:8000")
print()
print("Endpoints disponíveis:")
print("  - GET  http://localhost:8000/")
print("  - POST http://localhost:8000/api/license/check")
print("  - GET  http://localhost:8000/admin/status")
print("  - GET  http://localhost:8000/docs")
print()
print("Pressione Ctrl+C para parar o servidor")
print("=" * 70)
print()

# Importar e rodar o app
from main_firebase import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
