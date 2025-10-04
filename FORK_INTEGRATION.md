# Guide d'adaptation du fork CTranslate2-rocm pour les dernières versions

## Objectif
Adapter le fork CTranslate2-rocm pour être compatible avec :
- **ROCm 6.3**
- **PyTorch** (dernière version compatible ROCm 6.3)
- **WhisperX 3.4.3**
- **faster-whisper 1.2.0**
- **ctranslate2 4.4.0** (contrainte: whisperx < 4.5.0)
- **Python 3.10+**
- **Ubuntu 24.04**

## Modifications apportées au fork

### 1. Fichiers de configuration créés

- **`requirements-rocm6.3.txt`** : Liste complète des dépendances avec versions exactes
- **`requirements-install.txt`** : Liste simplifiée pour installation rapide
- **`build_rocm6.3.sh`** : Script de compilation automatisé pour ROCm 6.3
- **`install_dependencies.py`** : Script d'installation Python interactif
- **`test_installation.py`** : Script de test de l'installation
- **`README_MIGRATION_ROCM6.3.md`** : Guide de migration complet

### 2. Versions des packages clés

```
ctranslate2==4.4.0          # Compatible whisperx<4.5.0
faster-whisper==1.2.0
whisperx==3.4.3
transformers==4.56.2
pyannote.audio==3.4.0
PyYAML==6.0.3
```

### 3. Contraintes de compatibilité identifiées

**Conflit résolu** :
- `whisperx 3.4.3` requiert `ctranslate2<4.5.0`
- `faster-whisper 1.2.0` requiert `ctranslate2>=4.0,<5`
- **Solution** : Utiliser `ctranslate2==4.4.0`

### 4. Structure du workflow d'installation

```bash
# 1. Activer l'environnement conda
conda activate whisperxrocm

# 2. Installer PyTorch avec ROCm 6.3
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/rocm6.3

# 3. Installer toutes les dépendances
pip install -r requirements-install.txt

# 4. Compiler CTranslate2 (optionnel si on utilise le package précompilé)
bash build_rocm6.3.sh

# 5. Tester l'installation
python test_installation.py
```

## Intégration dans le fork GitHub

### Option 1 : Branche de développement (Recommandé)

Créer une branche dédiée pour ces modifications :

```bash
# Créer et passer sur une nouvelle branche
git checkout -b rocm6.3-pytorch-latest

# Ajouter les nouveaux fichiers
git add requirements-rocm6.3.txt requirements-install.txt
git add build_rocm6.3.sh install_dependencies.py test_installation.py
git add README_MIGRATION_ROCM6.3.md FORK_INTEGRATION.md

# Commit
git commit -m "feat: Add support for ROCm 6.3 + latest PyTorch/WhisperX

- Add requirements files for exact version compatibility
- Add automated build script for ROCm 6.3
- Add installation and testing scripts
- Add comprehensive migration guide
- Resolve ctranslate2 version conflict (use 4.4.0)
- Support whisperx 3.4.3 and faster-whisper 1.2.0"

# Pousser vers GitHub
git push origin rocm6.3-pytorch-latest
```

### Option 2 : Mise à jour directe de la branche principale

Si vous voulez mettre à jour directement la branche `rocm` :

```bash
# S'assurer d'être sur la bonne branche
git checkout rocm

# Ajouter et commiter
git add .
git commit -m "feat: Update to ROCm 6.3 with latest compatible versions"
git push origin rocm
```

## Modifications du CMakeLists.txt existant

Le `CMakeLists.txt` existant devrait déjà avoir le support ROCm via `WITH_HIP=ON`. 
Vérifiez que ces options sont bien présentes :

```cmake
option(WITH_HIP "Compile with HIP backend" OFF)
```

Pour compiler avec ROCm 6.3, utilisez :

```bash
cmake -DWITH_HIP=ON -DWITH_CUDA=OFF ...
```

## Tests recommandés après intégration

1. **Test de base** :
   ```bash
   python test_installation.py
   ```

2. **Test faster-whisper** :
   ```bash
   python faster_whisper_bench.py
   ```

3. **Test whisperX** :
   ```bash
   python whisperx_bench.py
   # ou
   whisperx tests/data/physicsworks.wav --model medium --device cuda
   ```

## Documentation à créer/mettre à jour

1. **README principal** : Ajouter une section "Installation pour ROCm 6.3"
2. **CHANGELOG** : Documenter les changements de versions
3. **Issues GitHub** : Créer une issue pour tracker les tests de compatibilité
4. **Wiki** : Créer une page sur les versions compatibles

## Compatibilité ascendante

Ces modifications sont **rétrocompatibles** :
- Les anciens scripts continuent de fonctionner
- Les nouveaux fichiers sont additifs (pas de suppression)
- Le CMakeLists.txt original n'est pas modifié
- Les utilisateurs de ROCm 6.1/6.2 peuvent continuer à utiliser les anciennes instructions

## Prochaines étapes suggérées

1. ✅ Créer les fichiers de configuration
2. ✅ Résoudre les conflits de dépendances
3. ⬜ Tester l'installation complète
4. ⬜ Créer une Pull Request ou mettre à jour la branche
5. ⬜ Mettre à jour la documentation principale
6. ⬜ Créer des GitHub Actions pour CI/CD (optionnel)
7. ⬜ Tester sur différentes cartes AMD (RX 6000/7000 series)

## Notes importantes

- **ctranslate2 4.4.0** est la dernière version compatible avec whisperx < 4.5.0
- Si whisperx met à jour sa dépendance, on pourra passer à ctranslate2 4.6.0+
- ROCm 6.3 est bien supporté par PyTorch
- Les performances devraient être similaires ou meilleures qu'avec ROCm 6.1

## Ressources

- [ROCm 6.3 Release Notes](https://rocm.docs.amd.com/)
- [PyTorch ROCm Support](https://pytorch.org/get-started/locally/)
- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
