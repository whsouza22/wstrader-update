import json

with open('ws_ai_stats_m1.json', 'r') as f:
    data = json.load(f)

# Analisar setups com LOSS (b > a)
losses = []
wins = []

for setup, values in data.get('arms', {}).items():
    a = values.get('a', 1)
    b = values.get('b', 1)
    n = values.get('n', 0)
    
    if b > a:  # Loss
        losses.append({
            'setup': setup,
            'a': a,
            'b': b,
            'n': n,
            'ratio': b/a
        })
    elif a > b and n > 0:  # Win
        wins.append({
            'setup': setup,
            'a': a,
            'b': b,
            'n': n,
            'ratio': a/b
        })

# Ordenar
losses.sort(key=lambda x: x['ratio'], reverse=True)
wins.sort(key=lambda x: x['ratio'], reverse=True)

print('='*70)
print('ANALISE DE LOSSES - SETUPS QUE PERDERAM')
print('='*70)
print(f'Total de setups com LOSS: {len(losses)}')
print(f'Total de setups com WIN: {len(wins)}')
print()

print('TOP 10 PIORES SETUPS (maior penalidade de loss):')
print('-'*70)
for i, loss in enumerate(losses[:10], 1):
    parts = loss['setup'].split('|')
    direcao = parts[0]
    
    # Decodificar o setup
    # Formato: CALL|sc10|pb3|re6|A8|eff0|fl5|dst7
    score = parts[1].replace('sc', 'Score=') if len(parts) > 1 else '?'
    pb = parts[2].replace('pb', 'Pullback=') if len(parts) > 2 else '?'
    re_pos = parts[3].replace('re', 'RangePos=') if len(parts) > 3 else '?'
    ativo_code = parts[4] if len(parts) > 4 else '?'
    eff = parts[5].replace('eff', 'Efficiency=') if len(parts) > 5 else '?'
    
    print(f'{i}. {direcao} | {score} | {pb} | {re_pos}')
    print(f'   Ativo: {ativo_code} | {eff}')
    print(f'   Ratio loss: {loss["ratio"]:.2f}x | Trades: {loss["n"]}')
    print()

print('='*70)
print('TOP 5 MELHORES SETUPS (maior win rate):')
print('-'*70)
for i, win in enumerate(wins[:5], 1):
    parts = win['setup'].split('|')
    direcao = parts[0]
    score = parts[1].replace('sc', 'Score=') if len(parts) > 1 else '?'
    pb = parts[2].replace('pb', 'Pullback=') if len(parts) > 2 else '?'
    
    print(f'{i}. {direcao} | {score} | {pb}')
    print(f'   Ratio win: {win["ratio"]:.2f}x | Trades: {win["n"]}')
    print()

# Análise de padrões nos losses
print('='*70)
print('PADROES NOS LOSSES:')
print('-'*70)

# Contar direções
call_losses = sum(1 for l in losses if l['setup'].startswith('CALL'))
put_losses = sum(1 for l in losses if l['setup'].startswith('PUT'))
print(f'CALL losses: {call_losses}')
print(f'PUT losses: {put_losses}')
print()

# Contar scores baixos
low_score_losses = sum(1 for l in losses if 'sc1' in l['setup'] or 'sc2' in l['setup'] or 'sc3' in l['setup'] or 'sc4' in l['setup'] or 'sc5' in l['setup'] or 'sc6' in l['setup'] or 'sc7' in l['setup'] or 'sc8' in l['setup'])
high_score_losses = sum(1 for l in losses if 'sc1' not in l['setup'] and 'sc2' not in l['setup'] and 'sc3' not in l['setup'] and 'sc4' not in l['setup'] and 'sc5' not in l['setup'] and 'sc6' not in l['setup'] and 'sc7' not in l['setup'] and 'sc8' not in l['setup'])
print(f'Losses com Score baixo (<=8): {low_score_losses}')
print(f'Losses com Score alto (>8): {high_score_losses}')
print()

# Contar efficiency
eff0_losses = sum(1 for l in losses if 'eff0' in l['setup'])
eff_high_losses = sum(1 for l in losses if 'eff5' in l['setup'] or 'eff6' in l['setup'])
print(f'Losses com Efficiency=0 (baixa): {eff0_losses}')
print(f'Losses com Efficiency>=5 (alta): {eff_high_losses}')
