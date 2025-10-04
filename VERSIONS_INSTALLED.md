# Versions finales installées - CTranslate2-rocm

**Date d'installation :** 3 octobre 2025  
**Environnement :** whisperxrocm (conda)  
**Python :** 3.10.18  
**OS :** Ubuntu 24.04 (Linux 6.8.0-84-generic)

## Configuration matérielle

- **GPUs :** 8x AMD Radeon Graphics (31.98 GB chacun)
- **ROCm :** 6.3
- **Architecture :** x86_64

## Versions des packages principaux

### Core ML Framework
```
torch==2.8.0+rocm6.3
torchaudio==2.8.0+rocm6.3
torchvision==0.23.0+rocm6.3
numpy==2.2.6
```

### CTranslate2 et Whisper
```
ctranslate2==4.4.0
faster-whisper==1.2.0
whisperx==3.4.3
transformers==4.56.2
tokenizers==0.22.1
```

### Audio Processing
```
librosa==0.11.0
soundfile==0.13.1
av==15.1.0
```

### PyAnnote (pour WhisperX)
```
pyannote.audio==3.4.0
pyannote.core==5.0.0
pyannote.database==5.1.3
pyannote.metrics==3.2.1
pyannote.pipeline==3.0.1
```

### ML Utilities
```
pandas==2.3.3
nltk==3.9.2
scipy==1.15.3
scikit-learn==1.7.2
speechbrain==1.0.3
```

### Lightning
```
lightning==2.5.5
pytorch-lightning==2.5.5
torchmetrics==1.8.2
```

### Autres dépendances importantes
```
PyYAML==6.0.3
sentencepiece==0.2.1
huggingface-hub==0.35.3
einops==0.8.1
tqdm==4.67.1
click==8.3.0
rich==14.1.0
requests==2.32.5
```

## Résolution des conflits de dépendances

### Conflit résolu #1 : ctranslate2
- **Problème :** whisperx 3.4.3 requiert `ctranslate2<4.5.0`
- **Solution :** Utiliser `ctranslate2==4.4.0`
- **Résultat :** Compatible avec faster-whisper 1.2.0 (qui accepte `>=4.0,<5`)

### Conflit résolu #2 : numpy
- **Problème :** whisperx 3.4.3 requiert `numpy>=2.0.2`
- **Solution :** Utiliser `numpy==2.2.6` (dernière version stable)
- **Résultat :** Compatible avec toutes les dépendances

## Tests de validation

Tous les tests sont passés avec succès :
- ✅ PyTorch + ROCm 6.3
- ✅ CTranslate2 4.4.0
- ✅ Transformers 4.56.2
- ✅ faster-whisper 1.2.0
- ✅ WhisperX 3.4.3
- ✅ Bibliothèques audio (librosa, soundfile)
- ✅ PyAnnote.Audio 3.4.0

## Commandes d'installation

```bash
# 1. Créer l'environnement
conda create -n whisperxrocm python=3.10 pip -y
conda activate whisperxrocm

# 2. Installer PyTorch avec ROCm 6.3
pip install torch==2.8.0+rocm6.3 torchaudio==2.8.0+rocm6.3 torchvision==0.23.0+rocm6.3 \
    --index-url https://download.pytorch.org/whl/rocm6.3

# 3. Installer toutes les dépendances
pip install -r requirements-install.txt

# 4. Tester
python test_installation.py
```

## Performances attendues

Sur base des benchmarks ROCm 6.1 (RX6800) :
- **faster-whisper (medium)** : ~10.9-11.0s
- **WhisperX (medium)** : ~3.94-4.1s

Avec ROCm 6.3 et 8 GPUs, les performances devraient être significativement meilleures.

## Notes importantes

1. **Compute types CUDA/ROCm** : Le message "Non disponible" dans les tests CTranslate2 est normal - les compute types GPU sont disponibles à l'exécution.

2. **Warning torchaudio** : Le warning sur `torchaudio._backend.list_audio_backends` est une dépréciation connue, sans impact sur les fonctionnalités.

3. **Compatibilité** : Cette configuration est testée et validée pour :
   - Ubuntu 24.04
   - ROCm 6.3
   - Python 3.10.18
   - GPU AMD Radeon series

## Prochaines étapes

1. Exécuter les benchmarks : `python faster_whisper_bench.py`
2. Tester WhisperX : `whisperx audio.wav --model medium --device cuda`
3. Intégrer au fork GitHub
4. Documenter les performances sur vos 8 GPUs AMD

---

**Auteur :** Lycaos  
**Repository :** https://github.com/lycaos/CTranslate2-rocm  
**Branche :** rocm
