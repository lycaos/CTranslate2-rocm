#!/usr/bin/env python3
"""
Test réel de faster-whisper avec CTranslate2
"""

import time
from faster_whisper import WhisperModel

def test_faster_whisper():
    print("="*70)
    print("TEST FASTER-WHISPER EN CONDITIONS RÉELLES")
    print("="*70)
    
    # Tester avec le modèle tiny d'abord (plus rapide)
    print("\n1. Chargement du modèle 'tiny'...")
    start = time.time()
    model = WhisperModel("tiny", device="cuda", compute_type="float16")
    load_time = time.time() - start
    print(f"   ✅ Modèle chargé en {load_time:.2f}s")
    
    # Transcription
    print("\n2. Transcription de 'tests/data/physicsworks.wav'...")
    start = time.time()
    segments, info = model.transcribe("tests/data/physicsworks.wav", language="en")
    transcribe_time = time.time() - start
    
    print(f"   ✅ Transcription terminée en {transcribe_time:.2f}s")
    print(f"   Langue détectée: {info.language} (probabilité: {info.language_probability:.2f})")
    
    # Afficher les segments
    print("\n3. Résultats:")
    print("-" * 70)
    full_text = ""
    for segment in segments:
        print(f"   [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        full_text += segment.text
    print("-" * 70)
    
    print(f"\n4. Texte complet: {full_text.strip()}")
    
    # Vérifier que c'est cohérent
    if "physics" in full_text.lower():
        print("\n✅ TEST RÉUSSI - La transcription contient 'physics'")
        return True
    else:
        print("\n⚠️  ATTENTION - La transcription ne contient pas 'physics'")
        return False

if __name__ == "__main__":
    try:
        success = test_faster_whisper()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
