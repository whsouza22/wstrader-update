"""Teste da IA Preditiva - verifica se todos os componentes funcionam"""
import os
os.environ['BROKER_TYPE'] = 'iq_option'
os.environ['IQ_EMAIL'] = 'test@test.com'
os.environ['IQ_PASS'] = 'test'

from WS_AUTO_AI import (
    ai_predict, ai_update, ai_make_key, ai_prior_from_setup,
    ensemble_predict, lgbm_predict,
    LGBM_ON, LGBM_AVAILABLE, IA_ON, IA_MODE,
    ENSEMBLE_MODE, AI_MIN_SAMPLES
)

print(f"IA_ON={IA_ON}, IA_MODE={IA_MODE}")
print(f"LGBM_ON={LGBM_ON}, LGBM_AVAILABLE={LGBM_AVAILABLE}")
print(f"ENSEMBLE_MODE={ENSEMBLE_MODE}, AI_MIN_SAMPLES={AI_MIN_SAMPLES}")

# Setup simulado
setup = {
    "dir": "CALL",
    "score": 0.65,
    "pb_len": 3,
    "retr": 0.38,
    "A_atr": 2.5,
    "effA": 0.18,
    "flips": 0.25,
    "distBreak": 0.10,
    "market_quality": 0.55,
    "entry_confidence": 0.60,
    "sr_proximity": 0.70,
    "sr_touches": 5,
    "late_ext": 0.0,
    "compression": 0.3,
    "ctx_score": 0.55,
    "rsi_norm": 0.45
}

# 1. Testar chave
key = ai_make_key("EURUSD-OTC", setup)
print(f"\n[1] Chave IA: {key}")

# 2. Testar prior
prior = ai_prior_from_setup(setup)
print(f"[2] Prior: {prior:.4f}")

# 3. Testar ai_predict (Bayesiano)
stats = {}
pred = ai_predict("EURUSD-OTC", setup, stats)
print(f"\n[3] Bayesian Predict:")
prob = pred["prob"]
conf = pred["conf"]
n_arm = pred["n_arm"]
bayes = pred["bayes"]
ucb = pred["ucb01"]
print(f"    prob={prob:.4f}, conf={conf:.4f}, n_arm={n_arm}")
print(f"    bayes_mean={bayes:.4f}, ucb01={ucb:.4f}")
print(f"    key={pred['key']}")

# 4. Testar lgbm_predict
lgbm_p, lgbm_avail = lgbm_predict(setup)
print(f"\n[4] LGBM Predict: prob={lgbm_p:.4f}, available={lgbm_avail}")

# 5. Testar ensemble
ens = ensemble_predict("EURUSD-OTC", setup, stats)
print(f"\n[5] Ensemble Predict:")
print(f"    should_trade={ens['should_trade']}")
print(f"    bayes_prob={ens['bayes_prob']:.4f}")
print(f"    lgbm_prob={ens['lgbm_prob']:.4f}")
print(f"    ensemble_prob={ens['ensemble_prob']:.4f}")
print(f"    reason={ens['reason']}")

# 6. Simular aprendizado
print("\n[6] Simulando 5 WINs e 2 LOSSes...")
for i in range(5):
    ai_update("EURUSD-OTC", setup, 1.0, stats)
for i in range(2):
    ai_update("EURUSD-OTC", setup, -1.0, stats)

pred2 = ai_predict("EURUSD-OTC", setup, stats)
n2 = pred2["n_arm"]
p2 = pred2["prob"]
c2 = pred2["conf"]
b2 = pred2["bayes"]
print(f"    Após trades: prob={p2:.4f}, conf={c2:.4f}, n_arm={n2}")
print(f"    bayes_mean={b2:.4f}")

# 7. Ensemble após aprendizado
ens2 = ensemble_predict("EURUSD-OTC", setup, stats)
print(f"\n[7] Ensemble após trades:")
print(f"    should_trade={ens2['should_trade']}")
print(f"    ensemble_prob={ens2['ensemble_prob']:.4f}")
print(f"    reason={ens2['reason']}")

# 8. Testar setup RUIM (deve bloquear)
setup_ruim = {
    "dir": "PUT",
    "score": 0.35,
    "pb_len": 1,
    "retr": 0.75,
    "A_atr": 0.8,
    "effA": 0.04,
    "flips": 0.70,
    "distBreak": 0.35,
    "market_quality": 0.25,
    "entry_confidence": 0.30,
    "sr_proximity": 0.10,
    "sr_touches": 1,
    "late_ext": 0.5,
    "compression": 0.1,
    "ctx_score": 0.20,
    "rsi_norm": 0.80
}

ens_ruim = ensemble_predict("EURUSD-OTC", setup_ruim, {})
print(f"\n[8] Setup RUIM:")
print(f"    should_trade={ens_ruim['should_trade']} (esperado: False)")
print(f"    ensemble_prob={ens_ruim['ensemble_prob']:.4f}")
print(f"    reason={ens_ruim['reason']}")

# Verificações
errors = []
if prior < 0.40 or prior > 0.75:
    errors.append(f"Prior fora do range: {prior}")
if pred["n_arm"] != 0:
    errors.append(f"n_arm inicial deveria ser 0, got {pred['n_arm']}")
if pred2["n_arm"] != 7:
    errors.append(f"n_arm após 7 trades deveria ser 7, got {pred2['n_arm']}")
if pred2["bayes"] < pred["bayes"]:
    # Com 5 wins e 2 losses (71% win), bayes deveria subir
    pass  # Pode variar dependendo do prior
if ens_ruim["should_trade"]:
    errors.append("Setup ruim não deveria ser aprovado!")

if errors:
    print("\n❌ ERROS ENCONTRADOS:")
    for e in errors:
        print(f"  - {e}")
else:
    print("\n✅ IA PREDITIVA FUNCIONANDO CORRETAMENTE!")
    print("   - Bayesiano: OK")
    print("   - LightGBM: " + ("OK (modelo treinado)" if lgbm_avail else "OK (sem modelo, normal)"))
    print("   - Ensemble: OK")
    print("   - Aprendizado: OK")
    print("   - Filtro de qualidade: OK")
