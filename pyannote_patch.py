#!/usr/bin/env python3
"""
Patch pour PyAnnote 3.4.0 - Supprime l'argument 'plda' non supporté
"""

from pyannote.audio import Pipeline
from pyannote.audio.pipelines import SpeakerDiarization
import functools

# Sauvegarder l'init original
_original_init = SpeakerDiarization.__init__

@functools.wraps(_original_init)
def patched_init(self, **kwargs):
    """Version patchée de __init__ qui ignore l'argument plda"""
    # Supprimer l'argument plda s'il existe
    if 'plda' in kwargs:
        print(f"⚠️  Argument 'plda' supprimé (non supporté par PyAnnote 3.4.0)")
        del kwargs['plda']
    
    # Appeler l'init original
    return _original_init(self, **kwargs)

# Appliquer le patch
SpeakerDiarization.__init__ = patched_init

print("✅ Patch PyAnnote appliqué: argument 'plda' sera ignoré")
