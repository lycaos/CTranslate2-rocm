#!/usr/bin/env python3
"""
Script de débogage pour comprendre pourquoi PyAnnote charge $model/segmentation
"""

import sys
import yaml
import functools

# ===== INTERCEPTION DES APPELS =====

# 1. Intercepter yaml.safe_load pour voir ce qui est chargé
_original_yaml_load = yaml.safe_load

@functools.wraps(_original_yaml_load)
def debug_yaml_load(stream):
    result = _original_yaml_load(stream)
    if isinstance(result, dict) and 'pipeline' in result:
        print(f"\n{'='*70}")
        print("🔍 YAML CHARGÉ:")
        print(f"{'='*70}")
        print(yaml.dump(result, default_flow_style=False))
        print(f"{'='*70}\n")
    return result

yaml.safe_load = debug_yaml_load

# 2. Intercepter hf_hub_download pour voir quels fichiers sont téléchargés
from huggingface_hub import hf_hub_download as _original_hf_hub_download

def debug_hf_hub_download(repo_id, filename, **kwargs):
    result = _original_hf_hub_download(repo_id, filename, **kwargs)
    if filename == 'config.yaml':
        print(f"\n{'='*70}")
        print(f"📥 HF_HUB_DOWNLOAD: {repo_id}/{filename}")
        print(f"📂 Chemin: {result}")
        print(f"{'='*70}")
        with open(result, 'r') as f:
            content = f.read()
            print("📝 Contenu du fichier:")
            print(content)
        print(f"{'='*70}\n")
    return result

import huggingface_hub
huggingface_hub.hf_hub_download = debug_hf_hub_download

# 3. Intercepter SpeakerDiarization.__init__
from pyannote.audio.pipelines import SpeakerDiarization

_original_init = SpeakerDiarization.__init__

@functools.wraps(_original_init)
def debug_init(self, plda=None, **kwargs):
    print(f"\n{'='*70}")
    print("🎯 SpeakerDiarization.__init__ APPELÉ")
    print(f"{'='*70}")
    print(f"📋 Arguments reçus:")
    if plda is not None:
        print(f"  - plda: {plda}")
    for key, value in kwargs.items():
        print(f"  - {key}: {value}")
    print(f"{'='*70}\n")
    
    if plda is not None:
        print("⚠️  Argument 'plda' ignoré (non supporté par PyAnnote 3.4.0)")
    
    return _original_init(self, **kwargs)

SpeakerDiarization.__init__ = debug_init

# ===== TEST DE CHARGEMENT =====

print("\n" + "="*70)
print("🚀 DÉBUT DU DÉBOGAGE")
print("="*70)

from pyannote.audio import Pipeline
import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")

print("\n📥 Tentative de chargement du pipeline...")
try:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        use_auth_token=HF_TOKEN
    )
    print("\n✅ SUCCÈS! Pipeline chargé")
except Exception as e:
    print(f"\n❌ ERREUR: {e}")
    print("\n📋 Traceback complet:")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("🏁 FIN DU DÉBOGAGE")
print("="*70)
