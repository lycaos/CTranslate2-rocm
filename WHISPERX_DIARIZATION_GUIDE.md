# WhisperX avec Diarization - Guide Complet ROCm 6.3

## ✅ Installation et Configuration

### Environnement Testé
- **ROCm** : 6.3.0
- **PyTorch** : 2.8.0+rocm6.3
- **Python** : 3.10.18
- **WhisperX** : 3.4.3
- **PyAnnote** : 3.4.0
- **CTranslate2** : 4.4.0 (compilé avec ROCm 6.3)

### Installation

```bash
# 1. Environnement conda
conda create -n whisperxrocm python=3.10 -y
conda activate whisperxrocm

# 2. PyTorch ROCm 6.3
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

# 3. WhisperX
pip install whisperx

# 4. PyAnnote
pip install pyannote.audio==3.4.0

# 5. CTranslate2 (depuis wheel compilé)
pip install dist/ctranslate2-4.4.0-cp310-cp310-linux_x86_64.whl
```

## 🎯 Utilisation

### Script de Test Complet

```python
#!/usr/bin/env python3
import whisperx
import torch

# Configuration
AUDIO_FILE = "votre_audio.wav"
HF_TOKEN = "votre_token_huggingface"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

# 1. Transcription
model = whisperx.load_model("base", DEVICE, compute_type=COMPUTE_TYPE)
audio = whisperx.load_audio(AUDIO_FILE)
result = model.transcribe(audio, batch_size=16)

# 2. Alignement
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], 
    device=DEVICE
)
result = whisperx.align(
    result["segments"], 
    model_a, 
    metadata, 
    audio, 
    DEVICE
)

# 3. Diarization (détection speakers)
from pyannote.audio import Pipeline

diarize_model = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",  # Modèle stable
    use_auth_token=HF_TOKEN
)
diarize_model.to(torch.device(DEVICE))

diarize_segments = diarize_model(AUDIO_FILE)
result = whisperx.assign_word_speakers(diarize_segments, result)

# 4. Résultat
for segment in result["segments"]:
    speaker = segment.get("speaker", "UNKNOWN")
    start = segment["start"]
    end = segment["end"]
    text = segment["text"]
    print(f"[{start:.1f}s → {end:.1f}s] {speaker}: {text}")
```

## 📊 Performances

Test sur audio de 203.3 secondes (AMD Radeon gfx906) :

| Étape | Temps | Vitesse |
|-------|-------|---------|
| Transcription | 2.5s | **82x temps réel** |
| Alignement | 5.0s | 40x temps réel |
| Diarization | 7.5s | 27x temps réel |
| **Total** | **15.0s** | **13.5x temps réel** |

## 🔧 Configuration HuggingFace

### Accepter les Conditions

1. Créer un compte sur [HuggingFace](https://huggingface.co/)
2. Générer un token : [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Accepter les conditions :
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

### Téléchargement Local (Optionnel)

```bash
# Télécharger les modèles en local
huggingface-cli download pyannote/speaker-diarization-3.1 \
    --token YOUR_TOKEN \
    --cache-dir ~/.cache/huggingface/hub
```

## ⚠️ Notes Importantes

### Cache PyAnnote

PyAnnote utilise **deux caches différents** :
- `~/.cache/huggingface/hub/` (cache HuggingFace standard)
- `~/.cache/torch/pyannote/` (cache PyTorch - **utilisé en priorité**)

### Modèles PyAnnote

| Modèle | Recommandé | Notes |
|--------|------------|-------|
| `speaker-diarization-3.1` | ✅ **OUI** | Stable, testé, compatible WhisperX |
| `speaker-diarization-community-1` | ❌ Non | Nécessite PyAnnote 4.x, problèmes d'instantiation |

### GPU ROCm

WhisperX détecte automatiquement les GPUs AMD :

```python
import torch
print(f"GPUs disponibles: {torch.cuda.device_count()}")
print(f"GPU actuel: {torch.cuda.get_device_name(0)}")
```

## 🐛 Dépannage

### Erreur: "Could not download model"

**Solution** : Vérifier que vous avez accepté les conditions sur HuggingFace

### Erreur: "instantiate() required"

**Solution** : Utiliser `pyannote/speaker-diarization-3.1` au lieu de `community-1`

### Performances lentes

**Optimisations** :
1. Utiliser `compute_type="float16"` (défaut)
2. Augmenter `batch_size` (16 recommandé)
3. Vérifier que PyTorch utilise le GPU :
   ```python
   print(torch.cuda.is_available())  # Doit afficher True
   ```

### Cache PyAnnote corrompu

```bash
# Nettoyer les caches
rm -rf ~/.cache/torch/pyannote/
rm -rf ~/.cache/huggingface/hub/models--pyannote*

# Retélécharger
huggingface-cli download pyannote/speaker-diarization-3.1 --token YOUR_TOKEN
```

## 📚 Ressources

- [CTranslate2 ROCm 6.3 Update](README_ROCM6.3_UPDATE.md)
- [Investigation PyAnnote](PYANNOTE_DEBUG_INVESTIGATION.md)
- [Tests de validation](TESTS_VALIDATION.md)
- [WhisperX Documentation](https://github.com/m-bain/whisperX)
- [PyAnnote Documentation](https://github.com/pyannote/pyannote-audio)

## 🎉 Résultat Final

WhisperX avec diarization fonctionne parfaitement sur ROCm 6.3 :
- ✅ Transcription ultra-rapide (82x temps réel)
- ✅ Alignement précis des mots
- ✅ Détection automatique des speakers
- ✅ Performance globale 13.5x temps réel

**Test complet** : `python test_whisperx_community1.py`
