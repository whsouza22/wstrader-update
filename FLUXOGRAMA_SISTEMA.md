# 🎯 Fluxograma do Sistema de Análise de Loss

## Visão Geral do Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                     WS TRADER BOT                               │
│                  (trading_bot.py)                               │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│              TRADING ENGINE                                      │
│           (ws_auto_ai_engine.py)                                │
│                                                                  │
│  • Conecta à IQ Option                                          │
│  • Executa operações                                            │
│  • Aguarda resultado                                            │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ↓
              ┌──────┴──────┐
              │             │
         WIN? │             │ LOSS?
              │             │
         ┌────┘             └────┐
         │                       │
         ↓                       ↓
    ┌────────┐         ┌──────────────────┐
    │  WIN   │         │  LOSS DETECTED   │
    │        │         │                  │
    │ Update │         │ Trigger Analysis │
    │  Stats │         └────────┬─────────┘
    └────────┘                  │
                                ↓
                    ┌──────────────────────────┐
                    │   LOSS ANALYZER          │
                    │  (loss_analyzer.py)      │
                    │                          │
                    │  1. Captura 100 velas    │
                    │  2. Analisa mercado      │
                    │  3. Analisa entrada      │
                    │  4. Gera análise IA      │
                    └──────────┬───────────────┘
                               │
                               ↓
                    ┌──────────────────────────┐
                    │   FIREBASE               │
                    │  (main_firebase.py)      │
                    │                          │
                    │  POST /api/loss/analyze  │
                    │                          │
                    │  Coleção: loss_analyses  │
                    └──────────┬───────────────┘
                               │
                               ↓
                    ┌──────────────────────────┐
                    │   DADOS SALVOS           │
                    │                          │
                    │  • order_id              │
                    │  • asset                 │
                    │  • direction             │
                    │  • market_context        │
                    │  • entry_quality         │
                    │  • ai_analysis           │
                    └──────────┬───────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            │                                     │
            ↓                                     ↓
┌──────────────────────┐              ┌──────────────────────┐
│  STATISTICS          │              │  RECOMMENDATIONS     │
│                      │              │                      │
│  GET /api/loss/      │              │  GET /api/loss/      │
│      statistics      │              │      recommendations │
│                      │              │                      │
│  • Total losses      │              │  • Filtros           │
│  • Top problems      │              │  • Blacklist         │
│  • Top assets        │              │  • Ajustes           │
└──────────────────────┘              └──────────┬───────────┘
                                                 │
                                                 ↓
                                      ┌──────────────────────┐
                                      │  AUTO OPTIMIZER      │
                                      │ (auto_optimizer.py)  │
                                      │                      │
                                      │  1. Lê recomendações │
                                      │  2. Aplica filtros   │
                                      │  3. Salva config     │
                                      └──────────┬───────────┘
                                                 │
                                                 ↓
                                      ┌──────────────────────┐
                                      │  auto_config.json    │
                                      │                      │
                                      │  • Novos filtros     │
                                      │  • Blacklist         │
                                      │  • Histórico         │
                                      └──────────┬───────────┘
                                                 │
                                                 ↓
                                      ┌──────────────────────┐
                                      │  BOT REINICIADO      │
                                      │                      │
                                      │  Usa novos filtros   │
                                      │  Menos losses! 📈    │
                                      └──────────────────────┘
```

---

## Fluxo Detalhado da Análise

### 1️⃣ Captura de Velas
```
┌──────────────────────────────────────┐
│  get_candles()                       │
│  ────────────────                    │
│  • Conecta IQ Option API             │
│  • Busca últimas 100 velas           │
│  • Timeframe: M1 (60 segundos)       │
│  • Calcula: body, wicks, range       │
└──────────────────────────────────────┘
```

### 2️⃣ Análise de Contexto
```
┌──────────────────────────────────────┐
│  analyze_market_context()            │
│  ──────────────────────              │
│                                      │
│  📊 Tendência                        │
│  • Conta velas verdes/vermelhas      │
│  • bullish / bearish / neutral       │
│                                      │
│  📈 Volatilidade                     │
│  • Calcula ATR (Average True Range)  │
│  • Compara com média histórica       │
│                                      │
│  🔄 Consolidação                     │
│  • Detecta movimentos laterais       │
│  • Baixa variação de preço           │
│                                      │
│  🎯 Suporte/Resistência              │
│  • Identifica máximos/mínimos        │
│  • Verifica proximidade              │
└──────────────────────────────────────┘
```

### 3️⃣ Análise de Entrada
```
┌──────────────────────────────────────┐
│  analyze_entry_quality()             │
│  ─────────────────────               │
│                                      │
│  💪 Força da Vela                    │
│  • Body ratio (corpo/range)          │
│  • strong: >70% | weak: <50%         │
│                                      │
│  🎯 Alinhamento                      │
│  • Últimas 5 velas                   │
│  • Quantas na mesma direção?         │
│                                      │
│  🚀 Momentum                         │
│  • Preço subindo ou descendo?        │
│  • Favorável à direção?              │
└──────────────────────────────────────┘
```

### 4️⃣ Geração de Análise IA
```
┌──────────────────────────────────────┐
│  generate_ai_analysis()              │
│  ────────────────────                │
│                                      │
│  🔍 Identifica Problemas:            │
│  ✓ Contra tendência                  │
│  ✓ Consolidação                      │
│  ✓ Próximo de S/R                    │
│  ✓ Entrada fraca                     │
│  ✓ Desalinhamento                    │
│  ✓ Alta volatilidade                 │
│                                      │
│  💡 Gera Recomendações:              │
│  ✓ Ajustes nos filtros               │
│  ✓ Blacklist de ativos               │
│  ✓ Gestão de risco                   │
│                                      │
│  📝 Relatório Completo               │
└──────────────────────────────────────┘
```

---

## Exemplo de Análise Gerada

```
📊 ANÁLISE DE LOSS - EURUSD-OTC
==================================================

💰 Stake: $10.00
📈 Direção: CALL
📉 Resultado: LOSS

🔍 PROBLEMAS IDENTIFICADOS:

1. Operação contra tendência: mercado está bearish mas operou CALL
   → Últimas 20 velas: 5 verdes, 15 vermelhas (75% bearish)

2. Velas anteriores desalinhadas com direção da operação
   → Apenas 1 de 5 velas alinhadas (20%)

3. Entrada fraca: vela com corpo pequeno
   → Body ratio: 0.35 (ideal: >0.70)

💡 RECOMENDAÇÕES:

1. Adicionar filtro: MIN_TREND_ALIGNMENT = 0.4
   → Bloquear operações quando <40% das velas estão alinhadas

2. Adicionar filtro: MIN_BODY_RATIO = 0.7
   → Aguardar velas com corpo forte

3. Adicionar filtro: MIN_ALIGNMENT_RATIO = 0.6
   → Exigir 3 de 5 velas alinhadas

📊 CONTEXTO DE MERCADO:
- Tendência: bearish
- Velas verdes/vermelhas: 5/15
- Volatilidade: low
- ATR: 0.00015
- Consolidação: Não

🎯 QUALIDADE DA ENTRADA:
- Força da vela: weak
- Body ratio: 0.35
- Alinhamento: 20.0%
- Momentum: wrong
```

---

## Estrutura de Dados no Firebase

### Documento em `loss_analyses`:
```json
{
  "order_id": "123456",
  "timestamp": "2026-01-28T10:30:00",
  "asset": "EURUSD-OTC",
  "direction": "CALL",
  "stake": 10.0,
  
  "market_context": {
    "trend": "bearish",
    "green_candles": 5,
    "red_candles": 15,
    "price_change_percent": -0.5,
    "atr": 0.00015,
    "volatility": "low",
    "is_consolidating": false,
    "near_resistance": false,
    "near_support": false
  },
  
  "entry_quality": {
    "entry_body_ratio": 0.35,
    "entry_quality": "weak",
    "alignment_ratio": 0.2,
    "momentum_direction": "wrong"
  },
  
  "ai_analysis": "📊 ANÁLISE DE LOSS...",
  
  "candles_data": {
    "count": 100,
    "last_10_closes": [1.1000, 1.0999, ...],
    "last_10_opens": [1.1001, 1.1000, ...]
  }
}
```

---

## Estatísticas Agregadas

### Response de `/api/loss/statistics`:
```json
{
  "success": true,
  "statistics": {
    "total_losses": 45,
    "total_stake_lost": 450.00,
    "avg_stake": 10.00,
    
    "direction_distribution": {
      "CALL": 25,
      "PUT": 20
    },
    
    "top_assets_with_loss": [
      {"asset": "EURUSD-OTC", "count": 15},
      {"asset": "GBPUSD-OTC", "count": 12}
    ],
    
    "top_problems": [
      {"problem": "contra_tendencia", "count": 18},
      {"problem": "entrada_fraca", "count": 12},
      {"problem": "desalinhamento", "count": 10}
    ]
  }
}
```

---

## Recomendações Automáticas

### Response de `/api/loss/recommendations`:
```json
{
  "success": true,
  "total_recommendations": 5,
  "based_on_losses": 45,
  
  "recommendations": [
    {
      "priority": "HIGH",
      "category": "Filtro de Tendência",
      "issue": "18 losses por operar contra tendência",
      "recommendation": "Adicionar filtro: bloquear operações contra tendência",
      "config_suggestion": "MIN_TREND_ALIGNMENT = 0.4"
    },
    {
      "priority": "HIGH",
      "category": "Qualidade de Entrada",
      "issue": "12 losses com vela de entrada fraca",
      "recommendation": "Exigir corpo forte nas velas",
      "config_suggestion": "MIN_BODY_RATIO = 0.7"
    },
    {
      "priority": "MEDIUM",
      "category": "Blacklist de Ativos",
      "issue": "EURUSD-OTC tem 15 losses (33%)",
      "recommendation": "Adicionar à blacklist temporária",
      "config_suggestion": "BLACKLIST_ASSETS = ['EURUSD-OTC']"
    }
  ]
}
```

---

## Aplicação de Otimizações

### Arquivo `auto_config.json`:
```json
{
  "filters": {
    "MIN_TREND_ALIGNMENT": 0.4,        // ← APLICADO
    "MIN_VOLATILITY_RATIO": 0.7,
    "SR_MIN_DISTANCE_PERCENT": 0.15,
    "MIN_BODY_RATIO": 0.7,             // ← APLICADO
    "MIN_ALIGNMENT_RATIO": 0.6,        // ← APLICADO
    "HIGH_VOLATILITY_STAKE_REDUCTION": 0.7
  },
  
  "blacklist_assets": [
    "EURUSD-OTC"                       // ← APLICADO
  ],
  
  "optimization_history": [
    {
      "timestamp": "2026-01-28T11:00:00",
      "recommendation": "Bloquear operações contra tendência",
      "config": "MIN_TREND_ALIGNMENT = 0.4",
      "priority": "HIGH"
    }
  ]
}
```

---

## Ciclo de Melhoria Contínua

```
        ┌─────────────────────────────────────┐
        │    Bot opera com configuração       │
        │         inicial/atual               │
        └───────────────┬─────────────────────┘
                        │
                        ↓
        ┌─────────────────────────────────────┐
        │  Resultados das operações           │
        │  Wins: ██████████ 70%               │
        │  Losses: ████ 30%                   │
        └───────────────┬─────────────────────┘
                        │
                        ↓
        ┌─────────────────────────────────────┐
        │  Análise de cada loss               │
        │  Identifica padrões e problemas     │
        └───────────────┬─────────────────────┘
                        │
                        ↓
        ┌─────────────────────────────────────┐
        │  Geração de recomendações           │
        │  baseadas em dados reais            │
        └───────────────┬─────────────────────┘
                        │
                        ↓
        ┌─────────────────────────────────────┐
        │  Aplicação automática de ajustes    │
        │  nos filtros e configurações        │
        └───────────────┬─────────────────────┘
                        │
                        ↓
        ┌─────────────────────────────────────┐
        │  Bot opera com nova configuração    │
        │  Wins: ████████████████ 85%         │
        │  Losses: ██ 15%                     │
        └───────────────┬─────────────────────┘
                        │
                        └─────────────┐
                                      │
                        ┌─────────────┘
                        ↓
                  (ciclo continua)
```

---

## Comandos Rápidos

### Ver Estatísticas:
```bash
curl http://localhost:8000/api/loss/statistics | python -m json.tool
```

### Listar Últimos Losses:
```bash
curl http://localhost:8000/api/loss/list?limit=10 | python -m json.tool
```

### Obter Recomendações:
```bash
curl http://localhost:8000/api/loss/recommendations | python -m json.tool
```

### Aplicar Otimizações:
```bash
python auto_optimizer.py optimize
```

### Ver Configuração:
```bash
python auto_optimizer.py show
```

---

## 🎯 Conclusão

Este sistema transforma cada loss em uma oportunidade de aprendizado, criando um ciclo de melhoria contínua que torna o bot cada vez mais eficiente.

**Resultado:** Menos losses, mais wins, melhor performance! 📈

---

**WS Trader - Sistema de Análise Inteligente de Loss** 🚀
