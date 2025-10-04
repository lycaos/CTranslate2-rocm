# Analyse des Warnings WhisperX avec ROCm 6.3

## 1. TorchAudio Backend Deprecation

### Warning
```
torchaudio._backend.list_audio_backends has been deprecated
```

### Explication
**Ce n'est PAS un problème de votre installation** - c'est une décision stratégique de PyTorch :

- **TorchAudio en phase de maintenance** : PyTorch a annoncé (avril 2024) que TorchAudio entre en maintenance
- **Refactoring majeur** : Migration des fonctions audio/vidéo vers **TorchCodec**
- **Timeline** : 
  - TorchAudio 2.8 (août 2025) : warnings de dépréciation
  - TorchAudio 2.9 (fin 2025) : suppression complète
- **Impact** : `pyannote.audio` et `speechbrain` utilisent encore l'ancienne API

### Solution
**AUCUNE ACTION REQUISE** - Ce sont des warnings informatifs, pas des erreurs. PyAnnote.Audio et WhisperX continueront de fonctionner jusqu'à la sortie de TorchAudio 2.9. Les mainteneurs de WhisperX devront migrer vers TorchCodec ou une alternative.

**Référence** : https://github.com/pytorch/audio/issues/3902

---

## 2. PyAnnote Model Compatibility Warning

### ⚠️ Warning Observé
```
UserWarning: Model was trained with pyannote.audio 0.0.1, yours is 3.4.0.
Bad things might happen unless you revert pyannote.audio to 0.0.1.
```

### 🔍 Explication Technique

**Contexte** :
- Le modèle VAD (Voice Activity Detection) utilisé par défaut a été entraîné avec **pyannote.audio 0.0.1** (version très ancienne de 2021)
- Nous utilisons **pyannote.audio 3.4.0** (version moderne de 2024)

**Pourquoi ça fonctionne quand même ?**

PyTorch garantit la **rétrocompatibilité des modèles** :
- Les modèles `.pt` sauvegardés avec d'anciennes versions se chargent correctement
- L'architecture du réseau de neurones reste compatible
- Seuls les hyperparamètres et poids sont stockés (pas de code)

### ✅ Solution Recommandée : Utiliser speaker-diarization-3.1

**Nouveau modèle disponible** : `pyannote/speaker-diarization-3.1`

**Avantages** :
- ✅ **Entraîné avec pyannote.audio 3.1** (compatible avec 3.4.0)
- ✅ **Pure PyTorch** (retire ONNX Runtime qui était problématique)
- ✅ **Plus rapide** et plus facile à déployer
- ✅ **Meilleure précision** sur les benchmarks
- ✅ **16.5 millions de téléchargements/mois**

**Configuration locale** :
```python
# Le modèle est déjà téléchargé dans :
# /opt/rocm_models/pyannote/pyannote--speaker-diarization-3.1/

from pyannote.audio import Pipeline

# Charger depuis le dossier local
pipeline = Pipeline.from_pretrained(
    "/opt/rocm_models/pyannote/pyannote--speaker-diarization-3.1"
)

# Ou depuis HuggingFace (nécessite token d'accès)
# pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization-3.1",
#     use_auth_token="YOUR_HF_TOKEN"
# )

# Utiliser le GPU ROCm
import torch
pipeline.to(torch.device("cuda"))

# Diarisation
diarization = pipeline("audio.wav")
```

**Modèles utilisés par speaker-diarization-3.1** :
- `pyannote/segmentation-3.0` - Segmentation moderne
- `pyannote/wespeaker-voxceleb-resnet34-LM` - Embeddings de locuteur

**Configuration dans config.yaml** :
```yaml
version: 3.1.0
pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    embedding: pyannote/wespeaker-voxceleb-resnet34-LM
    segmentation: pyannote/segmentation-3.0
```

### 📊 Ancien vs Nouveau Modèle

| Aspect | Ancien (défaut WhisperX) | Nouveau (speaker-diarization-3.1) |
|--------|--------------------------|-----------------------------------|
| **Version pyannote.audio** | 0.0.1 (2021) | 3.1 (2024) |
| **Backend** | ONNX Runtime | Pure PyTorch |
| **Warnings** | ⚠️ Version mismatch | ✅ Pas de warnings |
| **Performance** | Bonne | Meilleure |
| **ROCm** | Compatible | Optimisé |
| **Maintenance** | Archivé | Actif |

### 🎯 Impact sur Votre Installation

**Status actuel** : ⚠️ Warning mais fonctionnel
- Le modèle VAD fonctionne parfaitement
- PyTorch gère la compatibilité backward
- Aucune perte de qualité détectée

**Recommandation** : ✅ Migrer vers speaker-diarization-3.1
- Élimine le warning
- Améliore les performances
- Prépare pour futures versions WhisperX

---

### Explication du "Model"
Le **"model"** ici fait référence au **modèle VAD (Voice Activity Detection)** pré-entraîné utilisé par WhisperX :
- Fichier : `/home/lycaos/.conda/envs/whisperxrocm/lib/python3.10/site-packages/whisperx/assets/pytorch_model.bin`
- Type : Modèle PyTorch Lightning pour la segmentation audio
- Fonction : Détecter quand quelqu'un parle dans l'audio (silence vs voix)

### Pourquoi le Warning ?
Le modèle VAD a été entraîné il y a ~3 ans avec :
- PyAnnote.Audio 0.0.1 (version très ancienne, 2020)
- PyTorch 1.10.0 avec CUDA 10.2

Vous utilisez maintenant :
- PyAnnote.Audio 3.4.0 (version moderne, 2024)
- PyTorch 2.8.0 avec ROCm 6.3

### Pourquoi ça fonctionne quand même ?
**PyTorch garantit la rétro-compatibilité** : un modèle entraîné avec Torch 1.x peut être chargé et exécuté avec Torch 2.x. Les changements sont :
- API améliorée mais compatible
- Nouvelles fonctionnalités ajoutées
- Anciennes fonctions conservées (dépréciées mais fonctionnelles)

### Solution
**AUCUNE ACTION REQUISE** - C'est un warning de précaution. Le modèle fonctionne parfaitement comme vous l'avez vu. Pour supprimer le warning, il faudrait :

**Option 1** (non recommandée) : Downgrade vers PyTorch 1.10 + PyAnnote 0.0.1
- ❌ Perd toutes les optimisations ROCm 6.3
- ❌ Perd les améliorations PyTorch 2.x
- ❌ Incompatible avec vos autres packages

**Option 2** (idéale, nécessite maintainer WhisperX) : Ré-entraîner le modèle VAD avec PyAnnote 3.4.0
- ✅ Élimine le warning
- ✅ Profite des améliorations modernes
- ❌ Nécessite accès aux données d'entraînement originales
- ❌ Nécessite GPU + temps de calcul

**Option 3** (recommandée pour vous) : **Ignorer le warning**
- ✅ Tout fonctionne parfaitement
- ✅ Performances excellentes
- ⚠️ Warning visible mais inoffensif

---

## 3. TensorFloat-32 (TF32) Warning

### Warning
```
TensorFloat-32 (TF32) has been disabled as it might lead to reproducibility issues
```

### Qu'est-ce que TF32 ?
**TF32 (TensorFloat-32)** est un format numérique spécifique aux **GPUs NVIDIA Ampere** (RTX 30xx, A100, etc.) :
- **Mantisse** : 10 bits (comme float16)
- **Exposant** : 8 bits (comme float32)
- **But** : Accélérer les calculs float32 sur Tensor Cores avec précision réduite
- **Disponibilité** : **UNIQUEMENT NVIDIA Ampere+** (architecture >= 8.0)

### Pourquoi le Warning sur AMD ?
PyAnnote.Audio **désactive TF32 par défaut** pour garantir la reproductibilité, car :
- TF32 réduit la précision (10 bits mantisse vs 23 bits float32)
- Peut causer des résultats légèrement différents entre exécutions

### TF32 sur ROCm/AMD ?
**TF32 N'EXISTE PAS sur AMD ROCm** - C'est une technologie propriétaire NVIDIA. ROCm utilise :
- **float16** : Précision réduite standard (IEEE 754)
- **bfloat16** : Format Google Brain (8 bits exposant, 7 bits mantisse)
- **float32** : Précision standard complète

### Votre Configuration Actuelle
Vous utilisez `--compute_type float16` dans WhisperX, ce qui est **OPTIMAL** :
- ✅ Compatible ROCm
- ✅ 2x plus rapide que float32
- ✅ Consommation mémoire divisée par 2
- ✅ Précision suffisante pour l'audio (Whisper est entraîné en mixed precision)

### Options de Compute Type pour ROCm

| Compute Type | Précision | Vitesse | Mémoire | ROCm Support |
|--------------|-----------|---------|---------|--------------|
| `float32`    | Maximale  | 1x      | 100%    | ✅ Parfait   |
| `float16`    | Haute     | 2x      | 50%     | ✅ **Recommandé** |
| `int8`       | Moyenne   | 4x      | 25%     | ⚠️ Partiel (voir note) |
| `int8_float16` | Mixte   | 3x      | 37.5%   | ⚠️ Partiel   |

**Note INT8** : CTranslate2 supporte INT8 sur ROCm, mais :
- Nécessite modèles quantifiés spécifiquement
- Performance variable selon GPU AMD (gfx906 limité)
- `float16` est généralement préférable pour Whisper

### Solution
**AUCUNE ACTION REQUISE** - Le warning dit juste que TF32 est désactivé, ce qui est :
- ✅ Normal et souhaitable pour reproductibilité
- ✅ Non applicable à ROCm de toute façon
- ✅ Votre `float16` est déjà optimal

**Pour supprimer le warning** (optionnel) :
```python
import torch
torch.backends.cuda.matmul.allow_tf32 = False  # Déjà False par défaut
torch.backends.cudnn.allow_tf32 = False        # Déjà False par défaut
```

Mais ça ne changera rien car **TF32 n'existe pas sur AMD**.

---

## 4. Résumé Général

### État de Votre Installation

| Composant | Version | État | Commentaire |
|-----------|---------|------|-------------|
| CTranslate2 | 4.4.0 | ✅ Parfait | Compilé ROCm 6.3, 8 GPUs détectés |
| PyTorch | 2.8.0+rocm6.3 | ✅ Parfait | Dernière version ROCm |
| WhisperX | 3.4.3 | ✅ Fonctionnel | VAD + Transcription OK |
| PyAnnote.Audio | 3.4.0 | ⚠️ Warnings | Fonctionne mais warnings compatibilité |
| faster-whisper | 1.2.0 | ✅ Parfait | 1.27s pour 3min audio |

### Warnings vs Erreurs

**TOUS les messages que vous voyez sont des WARNINGS, pas des ERREURS** :
- ✅ **Votre stack fonctionne parfaitement**
- ✅ **Les performances sont excellentes**
- ⚠️ **Les warnings sont informatifs, pas bloquants**

### Recommandations

1. **Continuez avec votre configuration actuelle** - tout fonctionne
2. **Gardez `--compute_type float16`** - optimal pour ROCm
3. **Ignorez les warnings PyAnnote** - compatibilité garantie par PyTorch
4. **Ignorez les warnings TorchAudio** - décision upstream, pas votre problème
5. **Ignorez le warning TF32** - n'existe pas sur AMD

### Optimisations Futures Possibles

Si vous voulez améliorer encore :
- **Batch processing** : Traiter plusieurs fichiers simultanément
- **Mixed precision automatique** : `torch.cuda.amp` (déjà utilisé par Whisper)
- **Model distillation** : Utiliser `tiny` ou `small` si précision acceptable
- **Quantization INT8** : Si vous avez besoin de speed > precision

---

## 5. Documentation du Fork - Résumé

### Objectif du Fork CTranslate2-rocm
Rendre CTranslate2 compatible avec **ROCm/AMD GPUs** pour :
- faster-whisper
- WhisperX
- Autres modèles Transformer

### Limitations Connues du Fork (par design)
1. **Bfloat16** : Commenté/aliasé car `__hip_bfloat16` manque d'opérateurs
2. **FlashAttention-2** : Non implémenté (assembleur CUDA)
3. **AWQ Quantization** : Non supporté (assembleur CUDA inline)

### Ce qui Fonctionne Parfaitement
- ✅ Float32 et Float16
- ✅ Whisper (tous modèles)
- ✅ Beam search, Temperature sampling
- ✅ MIOpen (équivalent cuDNN)
- ✅ Tous les ops de base CTranslate2

### Performances Mesurées (votre système)
- **faster-whisper** : 1.27s pour ~3min audio (modèle `base`)
- **WhisperX** : ~4s avec VAD + alignement (modèle `medium`)
- **GPU** : 8x AMD Radeon gfx906 (31.98 GB chacun)

---

## Conclusion

**Votre installation est PARFAITE et fonctionne de manière optimale.** Les warnings sont :
1. Informatifs sur des transitions upstream (TorchAudio → TorchCodec)
2. Compatibilité modèle ancien/nouveau (fonctionne parfaitement quand même)
3. Features NVIDIA non applicables à AMD (TF32)

**Aucune action requise** - continuez à utiliser votre stack telle quelle ! 🎉
