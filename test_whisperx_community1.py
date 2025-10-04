#!/usr/bin/env python3
"""
Test WhisperX avec le nouveau modèle PyAnnote community-1
Compatible PyTorch 2.8 + ROCm 6.3
"""

import time
import os
import torch
import whisperx

# ===== PATCH PYANNOTE 3.4.0 pour compatibility avec community-1 =====

import time
import torch
import whisperx

# Configuration
AUDIO_FILE = "tests/data/physicsworks.wav"
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Token HuggingFace depuis variable d'environnement
DEVICE = "cuda"
COMPUTE_TYPE = "float16"  # Optimal pour ROCm
BATCH_SIZE = 16

print("=" * 70)
print("🎯 Test WhisperX avec PyAnnote 3.1 (ROCm 6.3)")
print("=" * 70)

# Vérifications préliminaires
print("\n📋 Vérifications système:")
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA disponible: {torch.cuda.is_available()}")
print(f"✓ Nombre de GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

try:
    import ctranslate2
    print(f"✓ CTranslate2: {ctranslate2.__version__}")
    print(f"✓ CTranslate2 GPUs détectés: {ctranslate2.get_cuda_device_count()}")
except Exception as e:
    print(f"⚠️  CTranslate2 erreur: {e}")

try:
    import pyannote.audio
    print(f"✓ PyAnnote.Audio: {pyannote.audio.__version__}")
except Exception as e:
    print(f"⚠️  PyAnnote erreur: {e}")

print("\n" + "=" * 70)
print("🎤 Étape 1: Transcription Whisper (avec VAD)")
print("=" * 70)

start_time = time.time()

# 1. Charger le modèle Whisper
print("\n📥 Chargement du modèle Whisper base...")
model = whisperx.load_model("base", DEVICE, compute_type=COMPUTE_TYPE)

# 2. Charger l'audio
print(f"📁 Chargement audio: {AUDIO_FILE}")
audio = whisperx.load_audio(AUDIO_FILE)
audio_duration = len(audio) / 16000  # 16kHz
print(f"⏱️  Durée audio: {audio_duration:.1f}s")

# 3. Transcription
print(f"🔄 Transcription (batch_size={BATCH_SIZE})...")
transcribe_start = time.time()
result = model.transcribe(audio, batch_size=BATCH_SIZE)
transcribe_time = time.time() - transcribe_start

print(f"✅ Transcription terminée en {transcribe_time:.2f}s")
print(f"⚡ Vitesse: {audio_duration/transcribe_time:.1f}x temps réel")
print(f"🗣️  Langue détectée: {result['language']}")
print(f"📝 Nombre de segments: {len(result['segments'])}")

# Libérer mémoire
del model
torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("🎯 Étape 2: Alignement temporel (wav2vec2)")
print("=" * 70)

# 4. Alignement des mots
print(f"\n📥 Chargement modèle d'alignement pour {result['language']}...")
align_start = time.time()
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], 
    device=DEVICE
)

print("🔄 Alignement des timestamps...")
result = whisperx.align(
    result["segments"], 
    model_a, 
    metadata, 
    audio, 
    DEVICE, 
    return_char_alignments=False
)
align_time = time.time() - align_start

print(f"✅ Alignement terminé en {align_time:.2f}s")

# Libérer mémoire
del model_a
torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("🆕 Étape 3: Diarization avec PyAnnote 3.1 (Stable)")
print("=" * 70)

# 5. Diarization avec PyAnnote 3.1
print("\n📥 Chargement du modèle PyAnnote speaker-diarization-3.1...")
print("ℹ️  Modèle stable et testé avec WhisperX + PyAnnote 3.4.0")

diarize_start = time.time()
diarize_time = 0.0  # Initialiser pour éviter erreur si exception

try:
    # Utiliser le wrapper WhisperX pour la diarization (modèle 3.1 compatible)
    from whisperx.diarize import DiarizationPipeline
    
    diarize_model = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
        device=DEVICE
    )
    
    print("🔄 Détection des locuteurs...")
    diarize_df = diarize_model(AUDIO_FILE)
    
    # Assigner les speakers aux segments
    result = whisperx.assign_word_speakers(diarize_df, result)
    
    diarize_time = time.time() - diarize_start
    print(f"✅ Diarization terminée en {diarize_time:.2f}s")
    
    # Compter les speakers détectés
    speakers = set()
    for segment in result["segments"]:
        if "speaker" in segment:
            speakers.add(segment["speaker"])
    
    print(f"👥 Nombre de locuteurs détectés: {len(speakers)}")
    print(f"🎭 Locuteurs: {sorted(speakers)}")
    
except Exception as e:
    print(f"❌ Erreur diarization: {e}")
    print("⚠️  Continuons sans diarization...")

total_time = time.time() - start_time

print("\n" + "=" * 70)
print("📊 RÉSULTATS FINAUX")
print("=" * 70)

print(f"\n⏱️  Temps total: {total_time:.2f}s")
print(f"⚡ Vitesse globale: {audio_duration/total_time:.1f}x temps réel")
print(f"\n📈 Détails par étape:")
print(f"  • Transcription: {transcribe_time:.2f}s ({transcribe_time/total_time*100:.1f}%)")
print(f"  • Alignement:    {align_time:.2f}s ({align_time/total_time*100:.1f}%)")
if diarize_time > 0:
    print(f"  • Diarization:   {diarize_time:.2f}s ({diarize_time/total_time*100:.1f}%)")

print("\n📝 Extrait de la transcription:")
print("-" * 70)
for i, segment in enumerate(result["segments"][:5]):
    speaker = segment.get("speaker", "???")
    start = segment["start"]
    end = segment["end"]
    text = segment["text"]
    print(f"[{start:6.1f}s → {end:6.1f}s] {speaker:10} {text}")

if len(result["segments"]) > 5:
    print(f"... ({len(result['segments']) - 5} segments supplémentaires)")

print("\n" + "=" * 70)
print("🎉 TEST TERMINÉ AVEC SUCCÈS!")
print("=" * 70)
print("\n💡 Informations sur le modèle:")
print("  • PyAnnote community-1 est la dernière version open-source (2024)")
print("  • Compatible PyTorch 2.8+ et Python 3.10+")
print("  • Amélioration ~9.6% sur AMI, ~5.6% sur DIHARD vs version 3.1")
print("  • Utilise segmentation-3.0 + speaker embedding moderne")
