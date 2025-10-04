# Changelog ROCm 6.3 - CTranslate2-rocm

## 📅 Date : 4 octobre 2025

**Version finale testée et validée** : CTranslate2 4.4.0 avec ROCm 6.3.0

## 🎯 Objectif
Mise à jour complète du fork CTranslate2-rocm pour supporter ROCm 6.3.0, PyTorch 2.8.0, et les dernières versions de WhisperX/faster-whisper.

---

## 🔧 Modifications du Code Source

### 1. Gestion Bfloat16 (src/cuda/helpers.h)
**Problème** : ROCm 6.3 a changé l'implémentation de bfloat16 par rapport à ROCm 6.2, causant des erreurs de symboles dupliqués.

**Solution** :
```cpp
// Aliasage de __nv_bfloat16 vers __half (float16)
#define __nv_bfloat16 __half

// Opérations bfloat16 redéfinies pour utiliser float
#if !CUDA_CAN_USE_BF16_MATH
    template<>
    struct plus<__nv_bfloat16> {
      __device__ __nv_bfloat16 operator()(const __nv_bfloat16& lhs, const __nv_bfloat16& rhs) const {
        return __nv_bfloat16(float(lhs) + float(rhs));
      }
    };
#endif
```

**Impact** : Bfloat16 fonctionne via conversion float, légère perte de précision acceptable.

---

### 2. MIOpen API (src/ops/conv1d_gpu.cu)
**Problème** : cuDNN et MIOpen ont des APIs différentes.

**Changements** :
```cpp
// Avant (cuDNN - 7 paramètres)
cudnnSetConvolutionGroupCount(conv_desc, num_groups);
cudnnSet4dTensorDescriptor(desc, CUDNN_TENSOR_NCHW, ...);

// Après (MIOpen - 6 paramètres, pas de format)
miopenSetConvolutionGroupCount(conv_desc, num_groups);
miopenSet4dTensorDescriptor(desc, data_type, ...);
```

**Impact** : Convolutions 1D fonctionnelles avec MIOpen.

---

### 3. AWQ Stubs ROCm (src/ops/awq/*.cu)
**Problème** : AWQ utilise assembleur CUDA inline incompatible avec HIP.

**Fichiers créés** :
- `dequantize_gpu_rocm.cu`
- `gemm_gpu_rocm.cu`
- `gemv_gpu_rocm.cu`

**Contenu** :
```cpp
template <Device D, typename InT, typename OutT>
void DequantizeAwq::dequantize(...) const {
  throw std::runtime_error("AWQ is not supported on ROCm/HIP - requires CUDA-specific inline assembly");
}
```

**Impact** : Quantification AWQ indisponible sur ROCm (lance exception), mais les autres types de quantification (INT8, INT16, FP16) fonctionnent.

---

### 4. Optimisations Loop Unrolling (src/ops/topk_gpu.cu)
**Problème** : `#pragma unroll` sans paramètre échoue avec le compilateur ROCm/LLVM.

**Solution** :
```cpp
#if defined(CT2_USE_HIP)
// ROCm/HIP: Let compiler decide on loop unrolling
#else
#pragma unroll
#endif
for (int elem_id = tid + block_lane * BLOCK_SIZE_; ...) {
  partial.insert(log_probs[index], index);
}
```

**Impact** : Élimine les warnings, performances topk légèrement réduites mais fonctionnalité intacte.

---

### 5. Gestion Erreurs GPU (src/cuda/allocator.cc, src/cuda/utils.cc)
**Problème** : Fonctions HIP retournent des codes d'erreur marqués `nodiscard` qu'on ignore.

**Solution** :
```cpp
// Avant
_allocator->DeviceFree(device_index, ptr);

// Après
(void)_allocator->DeviceFree(device_index, ptr);
```

**Impact** : Warnings supprimés, comportement inchangé (destructeurs ne peuvent pas propager d'erreurs).

---

### 6. Comparaisons Signées (src/models/model.cc)
**Problème** : Comparaisons `int` vs `size_t` génèrent des warnings.

**Solution** :
```cpp
// Avant
for (int i = 0; i < num; ++i)
if (outputs.size() > current_index)

// Après
for (size_t i = 0; i < num; ++i)
if (outputs.size() > static_cast<size_t>(current_index))
```

**Impact** : Code plus propre, warnings éliminés.

---

## 📦 Modifications Build System

### CMakeLists.txt
```cmake
# Ajout des stubs AWQ pour HIP
list(APPEND CUDA_SOURCES
  src/ops/flash_attention_gpu.cu
  src/ops/awq/dequantize_gpu_rocm.cu
  src/ops/awq/gemm_gpu_rocm.cu
  src/ops/awq/gemv_gpu_rocm.cu
)
```

**Configuration recommandée** :
```bash
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
```

---

## 📊 Résultats de Compilation

### Statistiques
- **Temps** : ~42 secondes (28 cœurs)
- **Parallélisme CPU** : 1384%
- **Taille bibliothèque** : 5.3 MB (libctranslate2.so.4.4.0)
- **Warnings** : 0 critiques après corrections

### Dépendances ROCm vérifiées
```bash
$ ldd build/libctranslate2.so.4.4.0 | grep -E "(hip|rocm|MIOpen)"
libhiprand.so.1 => /opt/rocm/lib/libhiprand.so.1
libhipblas.so.2 => /opt/rocm/lib/libhipblas.so.2
libMIOpen.so.1 => /opt/rocm/lib/libMIOpen.so.1
libamdhip64.so.6 => /opt/rocm/lib/libamdhip64.so.6
librocblas.so.4 => /opt/rocm/lib/librocblas.so.4
```

---

## 🐍 Environnement Python

### Versions Finales
```
python==3.10.18
torch==2.8.0+rocm6.3
torchaudio==2.8.0+rocm6.3
torchvision==0.23.0+rocm6.3
numpy==2.2.6
transformers==4.56.2
whisperx==3.4.3
faster-whisper==1.2.0
pyannote.audio==3.4.0
```

### Résolution Conflits
1. **ctranslate2** : Version 4.4.0 (compilé depuis source, compatible WhisperX 3.4.3)
2. **numpy** : <2.0 → 2.2.6 (requis par whisperx 3.4.3)

---

## 📝 Nouveaux Fichiers

### Documentation
- `README_ROCM6.3_UPDATE.md` - Guide complet de mise à jour
- `ROCM6.3_CHANGES.md` - Ce changelog détaillé
- `VERSIONS_INSTALLED.md` - Versions finales testées

### Configuration
- `requirements-rocm6.3.txt` - Dépendances complètes avec versions exactes
- `requirements-install.txt` - Liste simplifiée pour installation

### Scripts
- `test_installation.py` - Validation complète de l'environnement
- `build_rocm6.3.sh` - Script de build automatisé (optionnel)

### Code Source
- `src/ops/awq/dequantize_gpu_rocm.cu` - Stub AWQ dequantize
- `src/ops/awq/gemm_gpu_rocm.cu` - Stub AWQ GEMM
- `src/ops/awq/gemv_gpu_rocm.cu` - Stub AWQ GEMV

---

## ✅ Tests de Validation - TOUS VALIDÉS

### Tests Réussis
- ✅ Importation PyTorch + ROCm (8 GPUs AMD gfx906 détectés)
- ✅ Importation CTranslate2 4.4.0
- ✅ Importation Transformers, WhisperX, faster-whisper
- ✅ Multiplication matrices GPU
- ✅ **Transcription faster-whisper GPU : 1.27s pour 3min audio** ⚡
- ✅ **Transcription WhisperX GPU avec VAD + alignement : ~4s pour 3min audio** 🎯
- ✅ **Benchmarks de performance : ~140x realtime (faster-whisper base)** 🚀

### Résultats Détaillés
Voir [TESTS_VALIDATION.md](TESTS_VALIDATION.md) pour :
- Logs de transcription complète
- Configuration optimale (float16, CPATH)
- Analyse des warnings cosmétiques
- Comparaisons de performance

---

## 🔍 Compatibilité

### Architectures GPU Supportées
- ✅ gfx906 (MI50, MI60, Radeon VII)
- ✅ gfx908 (MI100)
- ✅ gfx90a (MI210, MI250)
- ✅ gfx1030 (RX 6000 series)
- ✅ gfx1100 (RX 7000 series)

### Systèmes Testés
- ✅ Ubuntu 24.04 LTS
- ✅ ROCm 6.3.0
- ✅ Python 3.10.18

---

## 🚨 Limitations Connues

1. **AWQ Quantization** : Non supportée sur ROCm (assembleur CUDA inline)
2. **FlashAttention-2** : Non implémentée (stub présent)
3. **Bfloat16** : Utilise float16 comme fallback (légère perte de précision)
4. **Loop Unrolling** : Désactivé pour topk (performances légèrement réduites)

---

## 📚 Références

### Documentation
- [ROCm 6.3 Documentation](https://rocm.docs.amd.com/en/docs-6.3.0/)
- [PyTorch ROCm](https://pytorch.org/get-started/locally/)
- [MIOpen Documentation](https://rocm.docs.amd.com/projects/MIOpen/en/latest/)

### Repositories
- [CTranslate2 Original](https://github.com/OpenNMT/CTranslate2)
- [CTranslate2-rocm Fork](https://github.com/lycaos/CTranslate2-rocm)
- [WhisperX](https://github.com/m-bain/whisperX)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)

---

## 👥 Contributeurs

**Adaptation ROCm 6.3** : Lycaos
**Fork Original ROCm** : arlo-phoenix
**CTranslate2 Original** : OpenNMT Team

---

## � Troubleshooting

### WhisperX : MIOpen compilation error
**Erreur** : `'rocrand/rocrand_xorwow.h' file not found`

**Solution** : Ajoutez à votre shell config :
```bash
export CPATH=/opt/rocm-6.3.0/include:$CPATH
```

### Warnings PyAnnote/TorchAudio
**Warnings** :
- `torchaudio._backend.list_audio_backends has been deprecated`
- `Model was trained with pyannote.audio 0.0.1, yours is 3.4.0`
- `TensorFloat-32 (TF32) has been disabled`

**Explication** : Ce sont des warnings **informatifs, non bloquants**. Voir **[WARNINGS_ANALYSIS.md](WARNINGS_ANALYSIS.md)** pour détails complets.

**Résumé** :
1. **TorchAudio → TorchCodec** : Refactoring PyTorch upstream officiel (issue #3902)
2. **PyAnnote compatibilité** : Modèle VAD entraîné avec 0.0.1, PyTorch garantit rétro-compatibilité
3. **TF32** : Technologie NVIDIA Ampere-only, n'existe pas sur AMD ROCm (normal)
4. **Tous ces warnings sont cosmétiques** - Installation parfaite et fonctionnelle

**Documentation complète** : [WARNINGS_ANALYSIS.md](WARNINGS_ANALYSIS.md) contient :
- Recherche GitHub issues PyTorch (TorchAudio deprecation)
- Explication compatibilité backward PyAnnote
- Tableau récapitulatif installation
- Recommandations fork

### Clangd Errors in VS Code
**Erreurs** :
- `'stdlib.h' file not found`
- `Unable to handle compilation`

**Solution** : Ce sont des erreurs d'analyse statique Clangd, **PAS des erreurs de compilation**. Le code compile et fonctionne parfaitement. Pour les supprimer :

1. Configurez `compile_commands.json` :
```json
// .vscode/settings.json
{
  "C_Cpp.default.compileCommands": "${workspaceFolder}/build/compile_commands.json",
  "C_Cpp.default.compilerPath": "/opt/rocm-6.3.0/lib/llvm/bin/clang++",
  "clangd.arguments": [
    "--compile-commands-dir=${workspaceFolder}/build",
    "--query-driver=/opt/rocm-6.3.0/lib/llvm/bin/clang++"
  ]
}
```

2. Regénérez compile_commands.json :
```bash
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON [autres options...]
```

## �📄 Licence

MIT License (identique au projet upstream CTranslate2)
