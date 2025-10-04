#!/usr/bin/env python3
"""
Script de test de l'installation CTranslate2-rocm
Vérifie PyTorch, ROCm, CTranslate2, faster-whisper et whisperX
"""

import sys

def print_section(title):
    """Affiche un titre de section"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def test_pytorch():
    """Test PyTorch et ROCm"""
    print_section("Test PyTorch + ROCm")
    
    try:
        import torch
        print(f"✅ PyTorch importé avec succès")
        print(f"   Version: {torch.__version__}")
        print(f"   ROCm disponible: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   Nombre de GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"      Mémoire totale: {props.total_memory / 1024**3:.2f} GB")
            
            # Test simple
            print("\n   Test de calcul GPU...")
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            z = torch.matmul(x, y)
            print(f"   ✅ Multiplication de matrices GPU réussie")
            
        else:
            print("   ⚠️  ROCm non disponible - PyTorch utilisera le CPU")
            
    except ImportError as e:
        print(f"❌ Erreur lors de l'import de PyTorch: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur lors du test PyTorch: {e}")
        return False
    
    return True

def test_ctranslate2():
    """Test CTranslate2"""
    print_section("Test CTranslate2")
    
    try:
        import ctranslate2
        print(f"✅ CTranslate2 importé avec succès")
        print(f"   Version: {ctranslate2.__version__}")
        
        # Tester les compute types supportés
        try:
            cpu_types = ctranslate2.get_supported_compute_types("cpu")
            print(f"   Compute types CPU: {cpu_types}")
        except:
            print(f"   Compute types CPU: Non disponible")
        
        try:
            cuda_types = ctranslate2.get_supported_compute_types("cuda")
            print(f"   Compute types CUDA/ROCm: {cuda_types}")
        except:
            print(f"   Compute types CUDA/ROCm: Non disponible")
            
    except ImportError as e:
        print(f"❌ CTranslate2 n'est pas installé")
        print(f"   Compilez avec: bash build_rocm6.3.sh")
        return False
    except Exception as e:
        print(f"❌ Erreur lors du test CTranslate2: {e}")
        return False
    
    return True

def test_transformers():
    """Test Transformers"""
    print_section("Test Transformers")
    
    try:
        import transformers
        print(f"✅ Transformers importé avec succès")
        print(f"   Version: {transformers.__version__}")
        
        # Tester l'import de modules spécifiques
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        print(f"   ✅ Modules Whisper disponibles")
        
    except ImportError as e:
        print(f"❌ Erreur lors de l'import de Transformers: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur lors du test Transformers: {e}")
        return False
    
    return True

def test_audio_libraries():
    """Test des bibliothèques audio"""
    print_section("Test des bibliothèques audio")
    
    libraries = [
        ('librosa', 'Librosa'),
        ('soundfile', 'SoundFile'),
    ]
    
    all_ok = True
    for module_name, display_name in libraries:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'version inconnue')
            print(f"✅ {display_name:15} : {version}")
        except ImportError:
            print(f"❌ {display_name:15} : Non installé")
            all_ok = False
    
    return all_ok

def test_faster_whisper():
    """Test faster-whisper"""
    print_section("Test faster-whisper")
    
    try:
        import faster_whisper
        print(f"✅ faster-whisper importé avec succès")
        print(f"   Version: {faster_whisper.__version__}")
        
        # Tester l'import de la classe principale
        from faster_whisper import WhisperModel
        print(f"   ✅ WhisperModel disponible")
        
        # Test de disponibilité des compute types
        print(f"   Compute types disponibles:")
        for compute_type in ['int8', 'float16', 'float32']:
            print(f"      - {compute_type}")
        
    except ImportError as e:
        print(f"❌ faster-whisper n'est pas installé")
        print(f"   Installez avec: pip install faster-whisper>=1.0.3")
        return False
    except Exception as e:
        print(f"❌ Erreur lors du test faster-whisper: {e}")
        return False
    
    return True

def test_whisperx():
    """Test WhisperX"""
    print_section("Test WhisperX")
    
    try:
        import whisperx
        print(f"✅ WhisperX importé avec succès")
        
        # Tester l'import de fonctions principales
        from whisperx import load_model
        print(f"   ✅ load_model disponible")
        
        # Vérifier les dépendances WhisperX
        dependencies = [
            ('pyannote.audio', 'PyAnnote.Audio'),
            ('nltk', 'NLTK'),
            ('pandas', 'Pandas'),
        ]
        
        print(f"\n   Dépendances WhisperX:")
        for module_name, display_name in dependencies:
            try:
                __import__(module_name)
                print(f"      ✅ {display_name}")
            except ImportError:
                print(f"      ⚠️  {display_name} manquant")
        
    except ImportError as e:
        print(f"⚠️  WhisperX n'est pas installé ou incomplet")
        print(f"   Installez avec: pip install whisperx --no-deps")
        return False
    except Exception as e:
        print(f"❌ Erreur lors du test WhisperX: {e}")
        return False
    
    return True

def test_python_environment():
    """Affiche les informations sur l'environnement Python"""
    print_section("Environnement Python")
    
    import platform
    import os
    
    print(f"Version Python: {sys.version}")
    print(f"Plateforme: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Non détecté')
    print(f"Environnement Conda: {conda_env}")
    
    rocm_path = os.environ.get('ROCM_PATH', 'Non défini')
    print(f"ROCM_PATH: {rocm_path}")
    
    hip_path = os.environ.get('HIP_PATH', 'Non défini')
    print(f"HIP_PATH: {hip_path}")
    
    return True

def main():
    """Fonction principale"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║        Test d'installation CTranslate2-rocm                      ║
║        ROCm 6.3 + PyTorch 2.8 + WhisperX                        ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    results = []
    
    # Exécuter tous les tests
    results.append(("Environnement Python", test_python_environment()))
    results.append(("PyTorch + ROCm", test_pytorch()))
    results.append(("CTranslate2", test_ctranslate2()))
    results.append(("Transformers", test_transformers()))
    results.append(("Bibliothèques audio", test_audio_libraries()))
    results.append(("faster-whisper", test_faster_whisper()))
    results.append(("WhisperX", test_whisperx()))
    
    # Afficher le résumé
    print_section("RÉSUMÉ DES TESTS")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30} : {status}")
        if not result:
            all_passed = False
    
    print(f"\n{'='*70}\n")
    
    if all_passed:
        print("🎉 Tous les tests sont passés avec succès!")
        print("\nVous pouvez maintenant:")
        print("  1. Exécuter des benchmarks: python faster_whisper_bench.py")
        print("  2. Tester WhisperX: python whisperx_bench.py")
        print("  3. Utiliser l'API CTranslate2 dans vos projets\n")
        return 0
    else:
        print("⚠️  Certains tests ont échoué.")
        print("\nActions recommandées:")
        
        # Recommandations spécifiques
        if not results[1][1]:  # PyTorch
            print("  - Réinstallez PyTorch: pip install torch==2.8.0+rocm6.3 --index-url https://download.pytorch.org/whl/rocm6.3")
        if not results[2][1]:  # CTranslate2
            print("  - Compilez CTranslate2: bash build_rocm6.3.sh")
        if not results[5][1]:  # faster-whisper
            print("  - Installez faster-whisper: pip install faster-whisper>=1.0.3")
        if not results[6][1]:  # WhisperX
            print("  - Installez WhisperX: pip install whisperx --no-deps")
        
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
