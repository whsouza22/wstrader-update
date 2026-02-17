# 🔐 Instruções para Configurar Licenças Gratuitas

## 📋 Resumo
Sistema que permite distribuir 5 chaves gratuitas, cada uma válida para **1 computador apenas**.

---

## 🚀 Passo a Passo

### 1️⃣ Adicionar o Router no main.py

Abra o arquivo `backend/main.py` e adicione estas linhas:

```python
# No início do arquivo, adicione o import:
from free_license_endpoint import router as free_license_router

# Depois da criação do app FastAPI, adicione:
app.include_router(free_license_router)
```

**Exemplo completo:**
```python
from fastapi import FastAPI
# ... outros imports ...
from free_license_endpoint import router as free_license_router

app = FastAPI(
    title="Wstrader License API",
    description="Sistema de licenciamento e autenticação para Wstrader Bot",
    version="1.0.0"
)

# Adicionar router de licenças gratuitas
app.include_router(free_license_router)

# ... resto do código ...
```

---

### 2️⃣ Criar as Tabelas e Gerar Chaves

No terminal, navegue até a pasta `backend` e execute:

```bash
cd backend
python setup_free_licenses.py
```

Escolha a opção **5** (Fazer tudo) para:
- Criar as tabelas `free_licenses` e `license_activations`
- Gerar 5 chaves automáticas

**Você verá algo assim:**
```
Chave 1: 0E8D31699C0DCB497DD95A678D41A187
Chave 2: 1F9E42788D1ECB598EE06B789E52B298
Chave 3: 2G0F53899E2FDC699FF17C890F63C309
Chave 4: 3H1G64900F3GED700GG28D901G74D410
Chave 5: 4I2H75011G4HFE811HH39E012H85E521
```

**⚠️ IMPORTANTE:** Copie e guarde estas 5 chaves! Você vai distribuí-las para seus usuários.

---

### 3️⃣ Adicionar Sua Chave Específica (OPCIONAL)

Se você já tem uma chave específica que quer usar (como `0e8d31699c0dcb497dd95a678d41a187`), execute:

```bash
python setup_free_licenses.py
```

Escolha opção **3** e cole sua chave quando solicitado.

---

### 4️⃣ Fazer Deploy no Render

1. **Commit dos arquivos novos:**
   ```bash
   git add backend/free_license_models.py
   git add backend/free_license_endpoint.py
   git add backend/setup_free_licenses.py
   git commit -m "Add free license system"
   git push
   ```

2. **No Render Dashboard:**
   - Acesse: https://dashboard.render.com/env-group/evg-d5h5pnq4d50c738rs0k0
   - Aguarde o deploy automático

3. **Criar as tabelas no servidor:**
   - Vá em "Shell" no Render
   - Execute: `python backend/setup_free_licenses.py`
   - Escolha opção **1** (criar tabelas) ou **5** (criar tudo)

---

### 5️⃣ Obter a URL do Servidor

Sua URL do Render deve ser algo como:
```
https://wstrader-backend-xxxx.onrender.com
```

O endpoint completo será:
```
https://wstrader-backend-xxxx.onrender.com/api/license/validate_free
```

---

### 6️⃣ Atualizar o Cliente (license_manager.py)

No arquivo `license_manager.py` (linha 19), substitua:

```python
LICENSE_SERVER_URL = os.getenv("LICENSE_SERVER_URL", "https://seu-servidor.onrender.com/api/license/validate_free")
```

Por:

```python
LICENSE_SERVER_URL = os.getenv("LICENSE_SERVER_URL", "https://wstrader-backend-xxxx.onrender.com/api/license/validate_free")
```

**Substitua `xxxx` pela URL real do seu servidor!**

---

## 🧪 Testar o Sistema

### Teste Local (antes de compilar):

1. Execute o app:
   ```bash
   python TelaPrincipal.py
   ```

2. Faça login com email e senha

3. Digite uma das 5 chaves geradas

4. Se tudo estiver correto, você verá:
   ```
   ✅ Licença ativada com sucesso!
   Ativação 1 de 1
   ```

---

## 📊 Gerenciar Licenças

### Ver status de uma chave:

Acesse no navegador:
```
https://wstrader-backend-xxxx.onrender.com/api/license/check/0E8D31699C0DCB497DD95A678D41A187
```

### Ver todas as licenças:

```bash
cd backend
python setup_free_licenses.py
# Escolha opção 4
```

---

## 🔒 Como Funciona

1. **Cada chave = 1 computador**
   - Você distribui 5 chaves diferentes
   - Cada usuário usa sua chave em 1 computador apenas

2. **Vinculação ao Hardware**
   - A chave é vinculada ao Hardware ID (HWID) do computador
   - Se o usuário tentar usar em outro PC, será bloqueado

3. **Validação Online**
   - Toda vez que o usuário faz login, valida com o servidor
   - Impossível usar offline por muito tempo

---

## ❓ Problemas Comuns

### Erro: "Chave de licença não encontrada"
- ✅ Verifique se a URL está correta no `license_manager.py`
- ✅ Verifique se as tabelas foram criadas no servidor Render
- ✅ Verifique se a chave foi adicionada ao banco

### Erro: "Esta chave já foi ativada"
- ✅ Normal se a chave já foi usada em outro computador
- ✅ Cada chave só funciona em 1 PC

### Erro: "Sem conexão com servidor"
- ✅ Verifique sua conexão com internet
- ✅ Verifique se o servidor Render está online

---

## 📝 Distribuir para Usuários

Envie para cada usuário:

```
Olá! Aqui está sua chave de licença gratuita do WS Trader:

Chave: 0E8D31699C0DCB497DD95A678D41A187

Instruções:
1. Baixe e instale o WS Trader
2. Faça login com seu email e senha da IQ Option
3. Digite esta chave quando solicitado
4. Pronto! Sua licença está ativada

IMPORTANTE:
- Esta chave funciona apenas no seu computador
- Não compartilhe com outras pessoas
- Se precisar trocar de computador, entre em contato

Dúvidas? Responda este email.
```

---

✅ **Sistema configurado! Suas 5 licenças gratuitas estão prontas!** 🎉
