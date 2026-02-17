# 🚀 Quick Start - Sistema de Análise de Loss

## Instalação Rápida

### 1. Dependências
```bash
pip install pandas requests firebase-admin
```

### 2. Verificar Backend
O backend Firebase deve estar rodando:
```bash
cd backend
python main_firebase.py
```

### 3. Testar Sistema

#### Opção A - Usar com o Bot (automático)
O sistema já está integrado! Basta rodar o bot normalmente:
```python
python TelaPrincipal.py
```

Quando houver loss, a análise é automática!

#### Opção B - Testar Manualmente
```bash
# Ver estatísticas
python -c "import requests; print(requests.get('http://localhost:8000/api/loss/statistics').json())"

# Executar otimização
python auto_optimizer.py optimize

# Exemplos interativos
python loss_analysis_examples.py
```

## 🔥 Uso Básico

### 1. O bot está operando e teve um LOSS
✅ Automático - análise é feita e salva no Firebase

### 2. Após alguns losses (5-10), ver estatísticas:
```bash
curl http://localhost:8000/api/loss/statistics
```

### 3. Obter recomendações:
```bash
curl http://localhost:8000/api/loss/recommendations
```

### 4. Aplicar otimizações:
```bash
python auto_optimizer.py optimize
```

### 5. Reiniciar bot com novos filtros
✅ Bot automaticamente usa as otimizações!

## 📊 Endpoints Disponíveis

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/loss/analyze` | POST | Salvar análise de loss |
| `/api/loss/list` | GET | Listar análises |
| `/api/loss/statistics` | GET | Estatísticas agregadas |
| `/api/loss/recommendations` | GET | Recomendações de ajustes |

## 🎯 Fluxo Automático

```
Bot opera → LOSS → Loss Analyzer → Firebase → Recomendações → Auto Optimizer → Bot otimizado
```

## ⚙️ Configurações

Arquivo gerado automaticamente: `auto_config.json`

Para ajustar manualmente:
```python
from auto_optimizer import AutoOptimizer

optimizer = AutoOptimizer()
optimizer.manual_adjust("MIN_TREND_ALIGNMENT", 0.7)
```

## 📝 Logs

Os logs de análise aparecem no console do bot:
```
🔍 Iniciando análise de loss...
✅ Capturadas 100 velas
📊 ANÁLISE DE LOSS - EURUSD-OTC
...
```

## 🆘 Troubleshooting

**Erro: Firebase não configurado**
- Verifique se o backend está rodando
- Confirme credentials.json no backend/

**Análises não aparecem**
- Aguarde alguns losses (5-10 mínimo)
- Verifique `/api/loss/list`

**Otimizações não aplicam**
- Verifique se auto_config.json foi criado
- Execute: `python auto_optimizer.py show`

## 📚 Mais Informações

- README completo: `LOSS_ANALYSIS_README.md`
- Exemplos práticos: `python loss_analysis_examples.py`
- Documentação API: http://localhost:8000/docs

---

**Pronto para usar!** 🎉
