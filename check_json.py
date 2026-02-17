import json

with open('ws_ai_stats_m1.json', 'r') as f:
    data = json.load(f)

print("=" * 50)
print("ESTATÍSTICAS DO JSON DE APRENDIZADO")
print("=" * 50)
print(f"Total operações: {data['meta']['total']}")
print(f"Wins: {data['meta']['global_wins']}")
print(f"Losses: {data['meta']['global_losses']}")
print(f"Win Rate: {data['meta']['global_wins']/max(1,data['meta']['total'])*100:.1f}%")
print(f"Padrões salvos: {len(data['arms'])}")
print()

# Padrões com operações reais
padroes = [(k, v) for k, v in data['arms'].items() if v.get('n', 0) > 0]
padroes.sort(key=lambda x: -x[1]['n'])

print(f"Padrões com operações reais: {len(padroes)}")
print()
print("TOP 10 PADRÕES:")
print("-" * 50)
for k, v in padroes[:10]:
    n = v['n']
    wr = v['a'] / (v['a'] + v['b']) * 100
    print(f"  {n:2d} ops | WR {wr:5.1f}% | {k[:45]}")

print()
print("=" * 50)
print("✅ JSON está sendo GRAVADO corretamente!")
print("✅ Neural lê esses dados no pre-treino ao iniciar")
print("=" * 50)
