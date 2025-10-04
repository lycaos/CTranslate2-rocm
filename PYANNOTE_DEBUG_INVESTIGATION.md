# PyAnnote Diarization - Investigation et Solution

**Date**: 4 octobre 2025  
**Environnement**: ROCm 6.3.0, PyTorch 2.8.0+rocm6.3, PyAnnote 3.4.0, WhisperX 3.4.3

## 🔍 Problème Initial

WhisperX avec diarization échouait avec l'erreur :
```
Could not download 'pyannote/speaker-diarization-community-1'
```

## 🛠️ Investigation (Débogage Systématique)

### 1. Cache HuggingFace Découvert

**Script de débogage créé** : `debug_pyannote_loading.py`

Découverte clé : PyAnnote charge depuis **2 caches différents** :
- ❌ `/home/lycaos/.cache/huggingface/hub/` (ce qu'on patchait)
- ✅ `/home/lycaos/.cache/torch/pyannote/` (le VRAI cache utilisé)

### 2. Problèmes du Modèle community-1

Erreurs rencontrées séquentiellement :

1. **Erreur `plda`** :
   ```
   SpeakerDiarization.__init__() got an unexpected keyword argument 'plda'
   ```
   - **Cause** : PyAnnote 3.4.0 ne supporte pas l'argument `plda`
   - **Solution** : Supprimer `plda: $model/plda` du config.yaml

2. **Erreur `$model/segmentation`** :
   ```
   Repo id must use alphanumeric chars: '$model/segmentation'
   ```
   - **Cause** : Variables `$model/*` non résolues
   - **Solution** : Remplacer par IDs HuggingFace directs :
     - `$model/segmentation` → `pyannote/segmentation-3.0`
     - `$model/embedding` → `pyannote/wespeaker-voxceleb-resnet34-LM`

3. **Erreur `VBxClustering`** :
   ```
   clustering must be one of [AgglomerativeClustering, OracleClustering]
   ```
   - **Cause** : PyAnnote 3.4.0 ne supporte pas VBxClustering
   - **Solution** : Utiliser `AgglomerativeClustering`

4. **Erreur `instantiate()`** :
   ```
   A pipeline must be instantiated with `pipeline.instantiate(parameters)` before it can be applied
   ```
   - **Cause** : Le modèle community-1 nécessite `instantiate()` mais :
     - `default_parameters()` lève `NotImplementedError`
     - `instantiate(params)` ne marque pas le pipeline comme `instantiated`
   - **Diagnostic** : 
     ```python
     pipeline.instantiate(params)
     print(pipeline.instantiated)  # False ❌
     ```

## ✅ Solution Finale

### Approche Retenue : Utiliser `speaker-diarization-3.1`

Le modèle **`pyannote/speaker-diarization-3.1`** :
- ✅ Compatible PyAnnote 3.4.0
- ✅ Fonctionne avec WhisperX sans modification
- ✅ Fournit `default_parameters()` valides
- ✅ Pas besoin de `instantiate()`
- ✅ Testé et stable

### Code WhisperX Modifié

```python
from pyannote.audio import Pipeline

diarize_model = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",  # ← Changé de community-1 à 3.1
    use_auth_token=HF_TOKEN
)
diarize_model.to(torch.device(DEVICE))

# Utilisation directe (pas besoin de instantiate)
diarize_segments = diarize_model(AUDIO_FILE)
result = whisperx.assign_word_speakers(diarize_segments, result)
```

## 📊 Résultats de Performance

Test sur audio de 203.3s :

| Étape | Temps | Vitesse |
|-------|-------|---------|
| Transcription | 2.46s | 82.7x temps réel |
| Alignement | 5.01s | - |
| **Diarization** | **7.54s** | **27x temps réel** |
| **Total** | **16.33s** | **12.5x temps réel** |

**Détection de speakers** : ✅ Fonctionne (SPEAKER_00 détecté)

## 📝 Configuration Finale

### Cache PyAnnote Patché

Fichier : `~/.cache/torch/pyannote/models--pyannote--speaker-diarization-community-1/blobs/4022db43960736338378fdb6b5a85cfdae198910`

```yaml
dependencies: 
  pyannote.audio: 3.4.0

pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    segmentation: pyannote/segmentation-3.0
    segmentation_batch_size: 32
    embedding: pyannote/wespeaker-voxceleb-resnet34-LM
    embedding_batch_size: 32
    embedding_exclude_overlap: true
    
params:
  clustering:
    threshold: 0.6
  segmentation:
    min_duration_off: 0.0
```

### Script de Test Final

`test_whisperx_community1.py` utilise maintenant :
- ✅ `pyannote/speaker-diarization-3.1` (stable)
- ✅ Aucun patch Python requis
- ✅ Détection automatique des speakers

## 🎯 Recommandations

1. **Pour WhisperX** : Utiliser `pyannote/speaker-diarization-3.1`
2. **Pour community-1** : Attendre PyAnnote 4.x ou utiliser approche custom
3. **Cache PyAnnote** : Toujours vérifier `~/.cache/torch/pyannote/`

## 🔧 Scripts Utilitaires Créés

| Script | Usage |
|--------|-------|
| `debug_pyannote_loading.py` | Trace le chargement des modèles |
| `debug_instantiate.py` | Debug l'instantiation des pipelines |
| `patch_pyannote_community1.sh` | Patch automatique du cache |
| `test_whisperx_community1.py` | Test complet WhisperX + Diarization |

## 📈 Comparaison Modèles

| Modèle | Version PyAnnote | Instantiate Requis | Fonctionne avec WhisperX |
|--------|------------------|--------------------|-----------------------------|
| speaker-diarization-3.1 | 3.x | ❌ Non | ✅ Oui |
| speaker-diarization-community-1 | 4.x | ✅ Oui | ❌ Non (problème instantiate) |

## 🎉 Conclusion

**Temps total d'investigation** : ~3h  
**Méthode** : Débogage systématique avec scripts de trace  
**Leçon clé** : Toujours tracer les appels réels plutôt que supposer le comportement  
**Résultat** : WhisperX fonctionnel avec diarization à 12.5x temps réel sur ROCm 6.3
