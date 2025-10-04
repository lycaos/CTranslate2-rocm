#!/usr/bin/env python3
"""
Script d'installation des dépendances pour CTranslate2-rocm
Compatible avec ROCm 6.3 + PyTorch 2.8
"""

import subprocess
import sys
import os

def run_command(cmd, description, check=True):
    """Exécute une commande et affiche le statut"""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    print(f"Commande: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, check=check)
    
    if result.returncode == 0:
        print(f"✅ {description} - SUCCÈS")
    else:
        print(f"❌ {description} - ÉCHEC")
        if check:
            sys.exit(1)
    
    return result.returncode == 0

def check_conda_env():
    """Vérifie que l'environnement conda whisperxrocm est activé"""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if conda_env != 'whisperxrocm':
        print("❌ Erreur: L'environnement conda 'whisperxrocm' n'est pas activé")
        print("Exécutez: conda activate whisperxrocm")
        sys.exit(1)
    print(f"✅ Environnement conda actif: {conda_env}")

def check_python_version():
    """Vérifie la version de Python"""
    version = sys.version_info
    print(f"🐍 Version Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 10:
        print("❌ Python 3.10+ requis")
        sys.exit(1)
    print("✅ Version Python compatible")

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║   Installation des dépendances CTranslate2-rocm             ║
║   ROCm 6.3 + PyTorch 2.8 + WhisperX                         ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Vérifications préliminaires
    check_conda_env()
    check_python_version()
    
    # Étape 1: Installation de PyTorch avec ROCm 6.3
    pytorch_installed = run_command(
        "pip install torch==2.8.0+rocm6.3 torchaudio==2.8.0+rocm6.3 torchvision==0.23.0+rocm6.3 "
        "--index-url https://download.pytorch.org/whl/rocm6.3",
        "Installation de PyTorch 2.8.0 avec ROCm 6.3",
        check=False
    )
    
    if not pytorch_installed:
        print("\n⚠️  PyTorch n'a pas pu être installé depuis le dépôt ROCm 6.3")
        print("Tentative avec la dernière version disponible...")
        run_command(
            "pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/rocm6.2",
            "Installation de PyTorch (version alternative)",
            check=True
        )
    
    # Étape 2: Vérification de PyTorch
    run_command(
        'python -c "import torch; '
        'print(f\'PyTorch: {torch.__version__}\'); '
        'print(f\'ROCm disponible: {torch.cuda.is_available()}\'); '
        'print(f\'Devices: {torch.cuda.device_count()}\')"',
        "Vérification de PyTorch + ROCm"
    )
    
    # Étape 3: Installation des dépendances de base
    run_command(
        "pip install numpy pyyaml setuptools",
        "Installation des dépendances de base"
    )
    
    # Étape 4: Installation de Transformers et tokenizers
    run_command(
        "pip install transformers>=4.40.0 tokenizers>=0.15.0 sentencepiece huggingface-hub",
        "Installation de Transformers et dépendances"
    )
    
    # Étape 5: Installation des bibliothèques audio
    run_command(
        "pip install librosa soundfile ffmpeg-python",
        "Installation des bibliothèques audio"
    )
    
    # Étape 6: Installation d'OpenAI Whisper
    run_command(
        "pip install openai-whisper",
        "Installation d'OpenAI Whisper"
    )
    
    # Étape 7: Installation de faster-whisper
    run_command(
        "pip install faster-whisper>=1.0.3",
        "Installation de faster-whisper"
    )
    
    # Étape 8: Installation des dépendances de WhisperX
    run_command(
        "pip install pyannote.audio>=3.1.1 nltk pandas tqdm scipy",
        "Installation des dépendances WhisperX"
    )
    
    # Étape 9: Installation de WhisperX (sans dépendances pour éviter les conflits)
    run_command(
        "pip install whisperx --no-deps",
        "Installation de WhisperX (sans dépendances)"
    )
    
    # Étape 10: Installation des dépendances de build pour CTranslate2
    run_command(
        "pip install pybind11 cmake",
        "Installation des dépendances de build"
    )
    
    # Étape 11: Installation des outils de développement
    run_command(
        "pip install pytest pytest-cov",
        "Installation des outils de développement"
    )
    
    # Résumé final
    print(f"\n{'='*60}")
    print("📊 RÉSUMÉ DE L'INSTALLATION")
    print(f"{'='*60}\n")
    
    # Vérifier les packages installés
    packages_to_check = [
        ('torch', 'PyTorch'),
        ('ctranslate2', 'CTranslate2'),
        ('transformers', 'Transformers'),
        ('faster_whisper', 'faster-whisper'),
        ('whisperx', 'WhisperX'),
    ]
    
    for module_name, display_name in packages_to_check:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'version inconnue')
            print(f"✅ {display_name:20} : {version}")
        except ImportError:
            if module_name == 'ctranslate2':
                print(f"⚠️  {display_name:20} : Non installé (compilez avec build_rocm6.3.sh)")
            else:
                print(f"❌ {display_name:20} : Non installé")
    
    print(f"\n{'='*60}")
    print("🎉 Installation des dépendances terminée!")
    print(f"{'='*60}\n")
    
    print("Prochaines étapes:")
    print("  1. Compiler CTranslate2 avec: bash build_rocm6.3.sh")
    print("  2. Tester avec: python test_installation.py")
    print("  3. Benchmarker avec: python faster_whisper_bench.py")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Installation interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Erreur inattendue: {e}")
        sys.exit(1)
