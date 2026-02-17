# 🔍 Sistema de Análise Automática de Loss

## Visão Geral

Sistema inteligente que automaticamente:
1. **Detecta** quando ocorre um LOSS
2. **Captura** as últimas 100 velas do mercado
3. **Analisa** com IA o motivo do loss
4. **Grava** análise detalhada no Firebase
5. **Gera** recomendações automáticas para ajustar o bot
6. **Aplica** os ajustes necessários para evitar futuros losses

## 📁 Arquivos Criados

### 1. `loss_analyzer.py`
Módulo principal de análise de loss.

**Principais funcionalidades:**
- Captura 100 velas quando há loss
- Análise de contexto de mercado (tendência, volatilidade, S/R)
- Análise de qualidade da entrada
- Geração de relatório com IA
- Integração com Firebase

**Classe:** `LossAnalyzer`

### 2. `auto_optimizer.py`
Sistema de otimização automática baseado nas análises.

**Principais funcionalidades:**
- Busca recomendações do Firebase
- Aplica ajustes automaticamente
- Mantém histórico de otimizações
- CLI para testes e ajustes manuais

**Classe:** `AutoOptimizer`

### 3. Endpoints no `backend/main_firebase.py`

#### `POST /api/loss/analyze`
Salva uma análise de loss no Firebase.

**Request:**
```json
{
  "order_id": "123456",
  "timestamp": "2026-01-28T10:30:00",
  "asset": "EURUSD-OTC",
  "direction": "CALL",
  "stake": 10.0,
  "market_context": {...},
  "entry_quality": {...},
  "ai_analysis": "..."
}
```

#### `GET /api/loss/list?limit=50&asset=EURUSD-OTC`
Lista análises de loss (com filtros opcionais).

#### `GET /api/loss/statistics`
Retorna estatísticas agregadas dos losses.

**Response:**
```json
{
  "success": true,
  "statistics": {
    "total_losses": 45,
    "total_stake_lost": 450.00,
    "avg_stake": 10.00,
    "direction_distribution": {"CALL": 25, "PUT": 20},
    "top_assets_with_loss": [...],
    "top_problems": [...]
  }
}
```

#### `GET /api/loss/recommendations`
Gera recomendações automáticas baseadas nas análises.

**Response:**
```json
{
  "success": true,
  "total_recommendations": 5,
  "recommendations": [
    {
      "priority": "HIGH",
      "category": "Filtro de Tendência",
      "issue": "15 losses por operar contra tendência",
      "recommendation": "Adicionar filtro: bloquear operações contra tendência...",
      "config_suggestion": "MIN_TREND_ALIGNMENT = 0.4"
    }
  ],
  "based_on_losses": 45
}
```

## 🚀 Como Usar

### 1. Ativação Automática

O sistema já está integrado no `ws_auto_ai_engine.py`. Quando há um loss, automaticamente:
- Captura e analisa as velas
- Grava no Firebase
- Log detalhado da análise

### 2. Visualizar Análises

```python
import requests

# Listar últimas análises
response = requests.get("http://localhost:8000/api/loss/list?limit=10")
data = response.json()

for analysis in data["analyses"]:
    print(f"\n{analysis['asset']} - {analysis['direction']}")
    print(analysis['ai_analysis'])
```

### 3. Obter Estatísticas

```python
response = requests.get("http://localhost:8000/api/loss/statistics")
stats = response.json()["statistics"]

print(f"Total de losses: {stats['total_losses']}")
print(f"Problemas principais:")
for problem in stats['top_problems']:
    print(f"  - {problem['problem']}: {problem['count']} vezes")
```

### 4. Otimização Automática

#### Via CLI:
```bash
# Otimizar com todas recomendações
python auto_optimizer.py optimize

# Apenas recomendações HIGH priority
python auto_optimizer.py optimize-high

# Ver configuração atual
python auto_optimizer.py show

# Ver histórico
python auto_optimizer.py history

# Resetar para padrão
python auto_optimizer.py reset
```

#### Via Python:
```python
from auto_optimizer import AutoOptimizer

optimizer = AutoOptimizer("http://localhost:8000")

# Otimização automática
result = optimizer.auto_optimize()
print(f"Aplicados {result['applied']} ajustes")

# Ver filtros atuais
filters = optimizer.get_current_filters()
print(filters)

# Ajuste manual
optimizer.manual_adjust("MIN_TREND_ALIGNMENT", 0.6)
```

### 5. Integração com Bot

O bot automaticamente usa os filtros otimizados:

```python
from auto_optimizer import AutoOptimizer

# Carrega otimizações
optimizer = AutoOptimizer()
filters = optimizer.get_current_filters()
blacklist = optimizer.get_blacklist()

# Aplica nos filtros do bot
MIN_TREND_ALIGNMENT = filters.get("MIN_TREND_ALIGNMENT", 0.5)
BLACKLIST_ASSETS = blacklist
```

## 📊 Análises Geradas

### Contexto de Mercado
- Tendência (bullish/bearish/neutral)
- Contagem de velas verdes/vermelhas
- Volatilidade (ATR)
- Consolidação
- Proximidade de S/R

### Qualidade da Entrada
- Força da vela de entrada
- Alinhamento das velas anteriores
- Momentum na direção

### Problemas Identificados
1. **Contra tendência** - Operou contra a tendência dominante
2. **Consolidação** - Mercado lateral sem direção clara
3. **S/R forte** - Próximo de resistência/suporte
4. **Entrada fraca** - Vela com corpo pequeno
5. **Desalinhamento** - Velas anteriores não alinhadas
6. **Alta volatilidade** - Movimentos imprevisíveis

### Recomendações Geradas
- Ajustes nos filtros de entrada
- Blacklist de ativos problemáticos
- Ajustes na gestão de risco
- Melhorias na detecção de S/R

## 🔧 Configurações Ajustáveis

Arquivo: `auto_config.json`

```json
{
  "filters": {
    "MIN_TREND_ALIGNMENT": 0.5,
    "MIN_VOLATILITY_RATIO": 0.7,
    "SR_MIN_DISTANCE_PERCENT": 0.15,
    "MIN_BODY_RATIO": 0.6,
    "MIN_ALIGNMENT_RATIO": 0.5,
    "HIGH_VOLATILITY_STAKE_REDUCTION": 0.7
  },
  "blacklist_assets": [],
  "optimization_history": []
}
```

### Parâmetros:

- **MIN_TREND_ALIGNMENT** (0-1): Mínimo de alinhamento com tendência
- **MIN_VOLATILITY_RATIO** (0-2): Mínimo de volatilidade para operar
- **SR_MIN_DISTANCE_PERCENT** (%): Distância mínima de S/R
- **MIN_BODY_RATIO** (0-1): Mínimo de corpo forte na vela
- **MIN_ALIGNMENT_RATIO** (0-1): Mínimo de velas alinhadas
- **HIGH_VOLATILITY_STAKE_REDUCTION** (0-1): Redução de stake em alta vol.

## 📈 Fluxo Completo

```
1. BOT OPERA → 2. RESULTADO = LOSS
                     ↓
3. Loss Analyzer captura 100 velas
                     ↓
4. Análise de contexto + entrada
                     ↓
5. Gera análise com IA
                     ↓
6. Salva no Firebase (coleção: loss_analyses)
                     ↓
7. Sistema lê análises e gera recomendações
                     ↓
8. Auto Optimizer aplica ajustes
                     ↓
9. Bot usa novos filtros → Menos losses!
```

## 🎯 Benefícios

✅ **Aprendizado contínuo** - Sistema aprende com cada loss
✅ **Ajustes automáticos** - Não precisa ajustar manualmente
✅ **Análise detalhada** - Entende exatamente o motivo do loss
✅ **Histórico completo** - Todas análises salvas no Firebase
✅ **Melhoria constante** - Bot fica melhor a cada operação

## ⚙️ Requisitos

```bash
pip install pandas requests firebase-admin
```

## 🔐 Firebase

Certifique-se que o Firebase está configurado:
- Coleção `loss_analyses` será criada automaticamente
- Permissões de leitura/escrita configuradas

## 📝 Logs

O sistema gera logs detalhados:

```
🔍 Iniciando análise de loss: EURUSD-OTC | CALL | $10.00
✅ Capturadas 100 velas

📊 ANÁLISE DE LOSS - EURUSD-OTC
==================================================

💰 Stake: $10.00
📈 Direção: CALL
📉 Resultado: LOSS

🔍 PROBLEMAS IDENTIFICADOS:
1. Operação contra tendência: mercado está bearish mas operou CALL
2. Velas anteriores desalinhadas com direção da operação

💡 RECOMENDAÇÕES:
1. Evitar CALL quando tendência recente é claramente bearish (>60% velas vermelhas)
2. Aguardar pelo menos 3 de 5 velas alinhadas antes de operar

...
```

## 🆘 Troubleshooting

### Loss não está sendo analisado
- Verifique se o backend está rodando
- Confirme que o Firebase está configurado
- Veja os logs para erros

### Recomendações não aparecem
- Execute pelo menos 5-10 operações com loss
- Verifique conexão com Firebase
- Use: `GET /api/loss/statistics` para debug

### Otimizações não estão sendo aplicadas
- Verifique se `auto_config.json` foi criado
- Execute `python auto_optimizer.py show` para ver config
- Reinicie o bot após otimizações

## 🔄 Atualizações Futuras

- [ ] Dashboard web para visualizar análises
- [ ] Machine Learning para detectar padrões
- [ ] Alertas em tempo real
- [ ] Testes A/B de estratégias
- [ ] Otimização multi-objetivo

## 📞 Suporte

Para dúvidas ou sugestões, consulte a documentação completa ou entre em contato.

---

**Desenvolvido para WS Trader** 🚀
