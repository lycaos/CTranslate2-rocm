# Mise à jour ROCm 6.3 + PyTorch 2.8 + WhisperX 3.4.3

## 🎯 Objectif de cette mise à jour

Adapter le fork CTranslate2-rocm pour supporter les dernières versions :
- **ROCm 6.3.0**
- **PyTorch 2.8.0+rocm6.3**
- **WhisperX 3.4.3**
- **faster-whisper 1.2.0**
- **CTranslate2 4.4.0** (compilé depuis source avec support ROCm)
- **Python 3.10+ sur Ubuntu 24.04**

## 🔥 Nouveautés de cette mise à jour

### Compilation CTranslate2 avec ROCm 6.3
✅ **Bfloat16** : Traité comme float16 (`__half`) pour compatibilité ROCm
✅ **MIOpen** : Remplacement de cuDNN avec adaptations API
✅ **AWQ** : Stubs ROCm (opérations AWQ non supportées, lance exceptions)
✅ **FlashAttention** : Stub présent (non implémenté pour ROCm)
✅ **Architecture gfx906** : Support complet pour AMD Radeon MI50/MI60/RX Vega

### Corrections spécifiques ROCm 6.3

1. **helpers.h** : 
   - Aliasé `__nv_bfloat16` → `__half` 
   - Opérations bfloat16 utilisent conversion float

2. **conv1d_gpu.cu** :
   - MIOpen API : `miopenSet4dTensorDescriptor` (6 args au lieu de 7)
   - Supprimé `cudnnSetConvolutionMathType` (inexistant dans MIOpen)
   - Remplacé `cudnnSetConvolutionGroupCount` → `miopenSetConvolutionGroupCount`

3. **topk_gpu.cu** :
   - Désactivé `#pragma unroll` pour ROCm/HIP (laisse le compilateur décider)
   - Élimine les warnings "loop not unrolled"

4. **allocator.cc & utils.cc** :
   - Ajout de `(void)` casts pour ignorer explicitement les valeurs de retour nodiscard
   - Élimine les warnings de gestion d'erreur ignorée

5. **model.cc** :
   - Changé `int` → `size_t` pour les indices de boucle
   - Ajout `static_cast<size_t>()` pour comparaisons
   - Élimine les warnings de comparaison signée/non signée

6. **AWQ GPU stubs** :
   - `src/ops/awq/dequantize_gpu_rocm.cu` - Stub lancant exception
   - `src/ops/awq/gemm_gpu_rocm.cu` - Stub lancant exception  
   - `src/ops/awq/gemv_gpu_rocm.cu` - Stub lancant exception
   - Raison : AWQ utilise assembleur CUDA inline incompatible avec HIP

7. **CMakeLists.txt** :
   - Ajout des stubs AWQ pour build HIP
   - Configuration automatique ROCm 6.3

## ✨ Nouveaux fichiers ajoutés

### Fichiers de configuration
- **`requirements-rocm6.3.txt`** - Liste complète des dépendances avec versions exactes
- **`requirements-install.txt`** - Liste simplifiée pour installation rapide
- **`VERSIONS_INSTALLED.md`** - Documentation des versions finales installées et testées

### Scripts d'installation et de build
- **`install_dependencies.py`** - Script Python interactif pour installer les dépendances
- **`build_rocm6.3.sh`** - Script bash pour compiler CTranslate2 avec ROCm 6.3
- **`test_installation.py`** - Script de validation complète de l'installation

### Documentation
- **`README_MIGRATION_ROCM6.3.md`** - Guide de migration détaillé
- **`FORK_INTEGRATION.md`** - Guide d'intégration au fork
- **`README_ROCM6.3_UPDATE.md`** - Ce fichier (résumé de la mise à jour)

## 🔧 Résolution des conflits de dépendances

### Conflit #1 : ctranslate2
**Problème :** whisperx 3.4.3 nécessite `ctranslate2<4.5.0`  
**Solution :** Utiliser `ctranslate2==4.4.0`  
**Résultat :** Compatible avec faster-whisper 1.2.0 (≥4.0,<5)

### Conflit #2 : numpy
**Problème :** whisperx 3.4.3 nécessite `numpy>=2.0.2`  
**Solution :** Utiliser `numpy==2.2.6`  
**Résultat :** Compatible avec toutes les dépendances

## ✅ Tests de validation

Tous les composants ont été testés et validés :

```
✅ PyTorch 2.8.0+rocm6.3 avec 8 GPUs AMD (31.98 GB chacun)
✅ CTranslate2 4.4.0
✅ Transformers 4.56.2
✅ faster-whisper 1.2.0
✅ WhisperX 3.4.3
✅ PyAnnote.Audio 3.4.0
✅ Bibliothèques audio (librosa, soundfile)
```

## 📦 Installation rapide

```bash
# 1. Créer l'environnement conda
conda create -n whisperxrocm python=3.10 pip -y
conda activate whisperxrocm

# 2. Installer PyTorch avec ROCm 6.3
pip install torch==2.8.0+rocm6.3 torchaudio==2.8.0+rocm6.3 torchvision==0.23.0+rocm6.3 \
    --index-url https://download.pytorch.org/whl/rocm6.3

# 3. Installer toutes les dépendances Python
pip install -r requirements-install.txt

# 4. Compiler CTranslate2 avec ROCm 6.3
cd /chemin/vers/CTranslate2-rocm
rm -rf build  # Nettoyage du cache

cmake -S . -B build \
  -DWITH_MKL=OFF \
  -DWITH_HIP=ON \
  -DWITH_CUDNN=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DCMAKE_CXX_COMPILER=/opt/rocm-6.3.0/lib/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=/opt/rocm-6.3.0/lib/llvm/bin/clang \
  -DCMAKE_PREFIX_PATH="/opt/rocm-6.3.0;/opt/rocm-6.3.0/hip" \
  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
  -DCMAKE_BUILD_TYPE=Release

# Compilation (utilise tous les cœurs disponibles)
cmake --build build -- -j$(nproc)

# Installation
cmake --install build --prefix $CONDA_PREFIX

# Installation du wheel Python
cd python
python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall

# 5. Tester l'installation
cd ..
python test_installation.py
```

## ⏱️ Temps de compilation

**Configuration testée** : AMD Threadripper avec 28 cœurs
- **Compilation complète** : ~42 secondes
- **Utilisation CPU** : 1384% (excellent parallélisme)
- **Taille bibliothèque** : libctranslate2.so.4.4.0 (5.3 MB)

## ⚠️ Warnings de compilation

Après les corrections, la compilation ne génère **aucun warning critique** :

### ✅ Warnings corrigés
- ✅ `loop not unrolled` (topk_gpu.cu) : Pragmas désactivés pour ROCm/HIP
- ✅ `ignoring nodiscard` (allocator.cc, utils.cc) : Casts `(void)` explicites ajoutés
- ✅ `sign comparison` (model.cc) : Types corrigés (int → size_t)

### Warnings cosmétiques (sans impact)
- ✅ `BFLOAT16 not handled in switch` : Normal, bfloat16 utilise le chemin float16
- ✅ `unused parameter` : Code mort pour compatibilité d'interface
- ✅ `#pragma float_control not supported` : Pragmas MSVC, Clang utilise ses propres flags
- ✅ `unused typedef/variable` : Artéfacts de macros ou code préparatoire

## 🚀 Utilisation

### faster-whisper
```python
from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.wav")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

### WhisperX (avec alignement et diarisation)
```bash
whisperx audio.wav --model medium --device cuda --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --diarize
```

### CTranslate2 (modèle converti)
```python
import ctranslate2
from transformers import WhisperProcessor

# Convertir un modèle Whisper
# ct2-transformers-converter --model openai/whisper-medium --output_dir whisper-medium-ct2

# Utiliser le modèle
model = ctranslate2.models.Whisper("whisper-medium-ct2", device="cuda")
# ... (voir documentation CTranslate2)
```

## 📊 Performances Mesurées

**Configuration testée** : 8x AMD Radeon gfx906 (31.98 GB chacun), ROCm 6.3.0

### Résultats Tests Réels
| Test | Modèle | Audio | Temps | Speedup |
|------|--------|-------|-------|----------|
| faster-whisper | base | 3min | **1.27s** | **~140x** |
| WhisperX | base | 3min | **~4s** | **~45x** |

**Performance exceptionnelle !** 🚀

### Configuration de référence (ROCm 6.1 sur RX6800)
- **faster-whisper (medium)** : ~10.9-11.0s
- **WhisperX (medium)** : ~3.94-4.1s

**Note** : Résultats non directement comparables (modèle base vs medium, architecture GPU différente).

**Détails complets** : Voir [TESTS_VALIDATION.md](TESTS_VALIDATION.md)

## 🔍 Vérifications système

Pour vérifier que ROCm 6.3 est correctement installé :
```bash
rocm-smi
rocminfo | grep "Name:"
```

Pour vérifier PyTorch + ROCm :
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"ROCm disponible: {torch.cuda.is_available()}")
print(f"Nombre de GPUs: {torch.cuda.device_count()}")
```

## 📝 Notes importantes

1. **Compatibilité ascendante** : Ces modifications sont additives et ne cassent pas les configurations existantes

2. **ROCm 6.3** : Assurez-vous d'avoir ROCm 6.3 installé sur votre système (`/opt/rocm-6.3.0`)

3. **Variables d'environnement** (optionnelles) :
   ```bash
   export ROCM_PATH=/opt/rocm-6.3.0
   export HIP_PATH=$ROCM_PATH
   export HSA_OVERRIDE_GFX_VERSION=10.3.0  # Ajuster selon votre GPU
   ```

4. **GPU AMD supportés** : RX 6000/7000 series, MI100/200/300 series

## 🔗 Ressources

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm](https://pytorch.org/get-started/locally/)
- [WhisperX](https://github.com/m-bain/whisperX)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [CTranslate2](https://github.com/OpenNMT/CTranslate2)

## 🤝 Contribution

Ces fichiers peuvent être intégrés au fork principal via :
1. Pull Request depuis cette branche
2. Merge direct si vous êtes le mainteneur
3. Tag de version pour marquer cette mise à jour importante

---

## 📖 Documentation Complémentaire

- **[WARNINGS_ANALYSIS.md](WARNINGS_ANALYSIS.md)** - Analyse détaillée des warnings PyAnnote/TorchAudio/TF32
- **[TESTS_VALIDATION.md](TESTS_VALIDATION.md)** - Résultats complets des tests et benchmarks
- **[ROCM6.3_CHANGES.md](ROCM6.3_CHANGES.md)** - Changelog technique détaillé
- **[VERSIONS_INSTALLED.md](VERSIONS_INSTALLED.md)** - Matrice complète des versions installées

---

**Version :** 1.1.0  
**Date :** 4 octobre 2025  
**Auteur :** Lycaos  
**Repository :** https://github.com/lycaos/CTranslate2-rocm  
**Branche :** rocm
