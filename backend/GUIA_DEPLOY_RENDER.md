# 🚀 Guia Completo - Deploy no Render

## 📋 Situação Atual
- ✅ Código do endpoint de licenças está pronto localmente
- ✅ URL configurada: `https://api-wstrader.onrender.com`
- ❌ Endpoint ainda não existe no servidor (erro 404)
- ❌ Chave não está no banco de dados do servidor

---

## 🔧 O que precisa ser feito:

### Opção 1: Deploy via Git (RECOMENDADO)

#### Passo 1: Encontrar o Repositório Git
O código do servidor Render deve estar em um repositório Git (GitHub, GitLab, etc.)

1. Acesse: https://dashboard.render.com
2. Clique no serviço `api-wstrader`
3. Na aba "Settings", procure por "Repository"
4. Você verá o link do repositório Git

#### Passo 2: Clonar o Repositório
```bash
git clone [URL_DO_REPOSITORIO]
cd [NOME_DO_REPOSITORIO]
```

#### Passo 3: Adicionar os Arquivos Novos
Copie estes arquivos para a pasta do repositório:
- `backend/free_license_endpoint.py`
- `backend/free_license_models.py` (não precisa mais, usamos database.py)
- `backend/setup_free_licenses.py`
- `backend/INSTRUCOES_SETUP_LICENCAS.md`

E modifique estes arquivos:
- `backend/main.py` (já modificado)
- `backend/database.py` (já modificado)

#### Passo 4: Commit e Push
```bash
git add .
git commit -m "Add free license system with validation endpoint"
git push origin main
```

O Render vai fazer deploy automático! ✅

---

### Opção 2: Upload Manual via Render Dashboard

Se você não tem acesso ao Git, pode fazer upload manual:

#### Passo 1: Acessar o Render Shell
1. Acesse: https://dashboard.render.com
2. Clique no serviço `api-wstrader`
3. Clique em "Shell" (no menu lateral)

#### Passo 2: Editar o main.py
No shell, execute:
```bash
nano main.py
```

Adicione estas linhas no arquivo:
```python
# Após os imports existentes, adicione:
from free_license_endpoint import router as free_license_router

# Após app.add_middleware(...), adicione:
app.include_router(free_license_router)
```

Salve: `Ctrl+O`, Enter, `Ctrl+X`

#### Passo 3: Criar free_license_endpoint.py
```bash
nano free_license_endpoint.py
```

Cole o conteúdo do arquivo `backend/free_license_endpoint.py` que está na sua máquina.

Salve: `Ctrl+O`, Enter, `Ctrl+X`

#### Passo 4: Editar database.py
```bash
nano database.py
```

Adicione no início:
```python
from sqlalchemy import JSON  # Adicionar na linha de imports
```

No final do arquivo, antes de `def init_db()`, adicione:
```python
class FreeLicense(Base):
    """Licença gratuita"""
    __tablename__ = "free_licenses"

    license_key = Column(String, primary_key=True, index=True)
    user_email = Column(String, nullable=True)
    max_activations = Column(Integer, default=1)
    current_activations = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)


class LicenseActivation(Base):
    """Ativação de licença"""
    __tablename__ = "license_activations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    license_key = Column(String, index=True, nullable=False)
    hwid = Column(String, index=True, unique=True, nullable=False)
    machine_info = Column(JSON, nullable=True)
    activated_at = Column(DateTime, default=datetime.utcnow)
    last_validated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
```

Salve: `Ctrl+O`, Enter, `Ctrl+X`

#### Passo 5: Reiniciar o Serviço
No Render Dashboard, clique em "Manual Deploy" → "Deploy latest commit"

---

## 📊 Após o Deploy

### 1. Criar as Tabelas
No Render Shell:
```bash
python
```

Então execute este código Python:
```python
from database import Base, engine, SessionLocal, FreeLicense, LicenseActivation

# Criar tabelas
Base.metadata.create_all(bind=engine)
print("✅ Tabelas criadas!")
```

Saia: `exit()`

### 2. Adicionar a Chave
No Render Shell:
```bash
python
```

Então execute:
```python
from database import SessionLocal, FreeLicense
from datetime import datetime

db = SessionLocal()

# Sua chave
license_key = "0E8D31699C0DCB497DD95A678D41A187"

# Criar licença
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
```

Saia: `exit()`

### 3. Verificar
Teste no navegador:
```
https://api-wstrader.onrender.com/api/license/check/0E8D31699C0DCB497DD95A678D41A187
```

Deve retornar informações da licença! ✅

---

## 🧪 Testar no App

Depois que tudo estiver configurado:

1. Execute o app: `python TelaPrincipal.py`
2. Faça login com email e senha
3. Digite a chave: `0e8d31699c0dcb497dd95a678d41a187`
4. Deve funcionar! 🎉

---

## ❓ Problemas?

### "404 Not Found" ao validar
- ✅ Verifique se fez deploy do código
- ✅ Verifique se o serviço reiniciou
- ✅ Teste o endpoint no navegador primeiro

### "Chave não encontrada"
- ✅ Verifique se criou as tabelas
- ✅ Verifique se adicionou a chave no banco
- ✅ Use `/api/license/check/[CHAVE]` para verificar

### Servidor não inicia
- ✅ Verifique logs do Render
- ✅ Pode ter erro de sintaxe no código
- ✅ Verifique se todos os imports estão corretos

---

## 📞 Suporte

Se precisar de ajuda:
1. Verifique os logs do Render
2. Teste os endpoints manualmente no navegador
3. Use o Render Shell para debug

✅ **Boa sorte com o deploy!** 🚀
