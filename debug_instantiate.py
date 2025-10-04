#!/usr/bin/env python3
"""
Debug final : comprendre pourquoi instantiate() ne fonctionne pas
"""

from pyannote.audio import Pipeline
import yaml
import torch
import os

print("="*70)
print("🔍 DEBUG INSTANTIATE")
print("="*70)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Charger le pipeline
pipeline = Pipeline.from_pretrained(
    'pyannote/speaker-diarization-community-1',
    use_auth_token=HF_TOKEN
)

print(f"\n1️⃣ État INITIAL:")
print(f"   - Type: {type(pipeline).__name__}")
print(f"   - instantiated: {pipeline.instantiated}")
print(f"   - parameters: {pipeline.parameters}")

# Charger params du config
with open('/home/lycaos/.cache/torch/pyannote/models--pyannote--speaker-diarization-community-1/snapshots/3533c8cf8e369892e6b79ff1bf80f7b0286a54ee/config.yaml') as f:
    config = yaml.safe_load(f)

params = config.get('params', {})
print(f"\n2️⃣ Paramètres du CONFIG:")
for key, value in params.items():
    print(f"   - {key}: {value}")

# Essayer instantiate
print(f"\n3️⃣ Appel instantiate(params)...")
try:
    result = pipeline.instantiate(params)
    print(f"   ✅ Retour: {result}")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    import traceback
    traceback.print_exc()

print(f"\n4️⃣ État APRÈS instantiate:")
print(f"   - instantiated: {pipeline.instantiated}")
print(f"   - parameters: {pipeline.parameters}")

# Vérifier les sous-pipelines
print(f"\n5️⃣ Sous-pipelines:")
for attr in ['segmentation_model', 'klustering', 'embedding']:
    if hasattr(pipeline, attr):
        obj = getattr(pipeline, attr)
        print(f"   - {attr}: {type(obj).__name__}")
        if hasattr(obj, 'instantiated'):
            print(f"     * instantiated: {obj.instantiated}")

# Essayer d'appliquer le pipeline
print(f"\n6️⃣ Test APPLICATION sur audio...")
try:
    result = pipeline('tests/data/physicsworks.wav')
    print(f"   ✅ SUCCÈS!")
    print(f"   - Type résultat: {type(result)}")
    print(f"   - Speakers: {list(result.labels())}")
except Exception as e:
    print(f"   ❌ ERREUR: {e}")
    
    # Debug plus profond
    print(f"\n7️⃣ Recherche méthode alternative...")
    print(f"   Méthodes utiles: {[m for m in dir(pipeline) if not m.startswith('_') and 'apply' in m.lower() or 'call' in m.lower() or 'run' in m.lower()]}")

print("\n" + "="*70)
print("🏁 FIN DEBUG")
print("="*70)
