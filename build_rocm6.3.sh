#!/bin/bash
# Script de compilation CTranslate2 avec support ROCm 6.3
# Pour Ubuntu 24.04, Python 3.10+

set -e  # Arrêter en cas d'erreur

echo "======================================"
echo "CTranslate2-rocm Build Script"
echo "ROCm 6.3 + PyTorch 2.8"
echo "======================================"

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Vérifier que nous sommes dans le bon environnement conda
if [[ "$CONDA_DEFAULT_ENV" != "whisperxrocm" ]]; then
    echo -e "${RED}Erreur: L'environnement conda 'whisperxrocm' n'est pas activé${NC}"
    echo "Exécutez: conda activate whisperxrocm"
    exit 1
fi

# Configuration
ROCM_VERSION=${ROCM_VERSION:-6.3.0}
ROCM_PATH=${ROCM_PATH:-/opt/rocm-$ROCM_VERSION}
BUILD_TYPE=${BUILD_TYPE:-Release}
NUM_JOBS=${NUM_JOBS:-$(nproc)}

echo -e "${GREEN}Configuration:${NC}"
echo "  ROCm Path: $ROCM_PATH"
echo "  Build Type: $BUILD_TYPE"
echo "  Parallel Jobs: $NUM_JOBS"
echo "  Install Prefix: $CONDA_PREFIX"
echo ""

# Vérifier que ROCm est installé
if [ ! -d "$ROCM_PATH" ]; then
    echo -e "${RED}Erreur: ROCm n'est pas installé dans $ROCM_PATH${NC}"
    echo "Installez ROCm 6.3 ou définissez ROCM_PATH"
    exit 1
fi

# Définir les variables d'environnement ROCm
export ROCM_PATH=$ROCM_PATH
export HIP_PATH=$ROCM_PATH
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# Détecter la version GFX du GPU
echo -e "${YELLOW}Détection du GPU AMD...${NC}"
if command -v rocminfo &> /dev/null; then
    GPU_INFO=$(rocminfo | grep "Name:" | head -1)
    echo "  GPU détecté: $GPU_INFO"
    
    # Extraire le gfx version
    GFX_VERSION=$(rocminfo | grep "gfx" | head -1 | awk '{print $2}')
    if [ -n "$GFX_VERSION" ]; then
        echo "  GFX Version: $GFX_VERSION"
        # Convertir gfx1030 -> 10.3.0
        export HSA_OVERRIDE_GFX_VERSION=$(echo $GFX_VERSION | sed 's/gfx//' | sed 's/\(.\)\(.\)\(.\)/\1.\2.\3/')
        echo "  HSA_OVERRIDE_GFX_VERSION: $HSA_OVERRIDE_GFX_VERSION"
    fi
else
    echo -e "${YELLOW}  rocminfo non trouvé, impossible de détecter le GPU${NC}"
fi
echo ""

# Nettoyer le répertoire de build précédent si demandé
if [ "$1" == "clean" ]; then
    echo -e "${YELLOW}Nettoyage du répertoire build...${NC}"
    rm -rf build
    echo -e "${GREEN}✓ Nettoyage terminé${NC}"
    echo ""
fi

# Créer le répertoire de build
mkdir -p build
cd build

# Configurer CMake
echo -e "${YELLOW}Configuration CMake...${NC}"
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DWITH_HIP=ON \
      -DWITH_CUDA=OFF \
      -DWITH_MKL=OFF \
      -DWITH_DNNL=OFF \
      -DWITH_OPENBLAS=ON \
      -DBUILD_CLI=ON \
      -DBUILD_TESTS=ON \
      -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
      -DCMAKE_PREFIX_PATH=$ROCM_PATH \
      -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/hipcc \
      -DCMAKE_C_COMPILER=$ROCM_PATH/bin/hipcc \
      ..

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Échec de la configuration CMake${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Configuration CMake réussie${NC}"
echo ""

# Compiler
echo -e "${YELLOW}Compilation en cours avec $NUM_JOBS threads...${NC}"
make -j$NUM_JOBS

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Échec de la compilation${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Compilation réussie${NC}"
echo ""

# Installer
echo -e "${YELLOW}Installation dans $CONDA_PREFIX...${NC}"
make install

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Échec de l'installation${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Installation réussie${NC}"
echo ""

# Retour au répertoire racine
cd ..

# Compiler le wrapper Python
echo -e "${YELLOW}Compilation du wrapper Python...${NC}"
cd python

# Installer les dépendances de build
pip install -r install_requirements.txt -q

# Définir la variable d'environnement pour trouver la bibliothèque
export CTRANSLATE2_ROOT=$CONDA_PREFIX

# Compiler le wheel
python setup.py bdist_wheel

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Échec de la compilation du wrapper Python${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Wrapper Python compilé${NC}"
echo ""

# Installer le wheel
echo -e "${YELLOW}Installation du package Python...${NC}"
pip install dist/*.whl --force-reinstall

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Échec de l'installation du package Python${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Package Python installé${NC}"
echo ""

cd ..

# Vérification de l'installation
echo -e "${YELLOW}Vérification de l'installation...${NC}"
python -c "import ctranslate2; print(f'CTranslate2 version: {ctranslate2.__version__}')"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Impossible d'importer ctranslate2${NC}"
    exit 1
fi

python -c "import ctranslate2; print(f'Compute types supportés: {ctranslate2.get_supported_compute_types(\"cuda\")}')"

echo ""
echo -e "${GREEN}======================================"
echo "✓ Build terminé avec succès!"
echo "======================================${NC}"
echo ""
echo "Prochaines étapes:"
echo "  1. Tester avec: python -c 'import ctranslate2; print(ctranslate2.__version__)'"
echo "  2. Installer faster-whisper: pip install faster-whisper>=1.0.3"
echo "  3. Installer whisperx: pip install whisperx --no-deps"
echo ""
