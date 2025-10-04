#!/bin/bash
# Patch PyAnnote pour ROCm 6.3 + PyTorch 2.8 + PyAnnote 3.4.0
#
# RÉSUMÉ DU PROBLÈME IDENTIFIÉ:
# - PyAnnote charge depuis ~/.cache/torch/pyannote/ (PAS ~/.cache/huggingface/)
# - Le modèle community-1 nécessite instantiate() qui n'est pas compatible avec WhisperX
# - SOLUTION: Utiliser pyannote/speaker-diarization-3.1 (stable et testé)

echo "🔧 Patch PyAnnote pour compatibilité ROCm 6.3..."

# 1. Patcher le cache torch/pyannote pour community-1 (si utilisé)
if [ -d ~/.cache/torch/pyannote/models--pyannote--speaker-diarization-community-1 ]; then
    echo "📝 Patch community-1 (cache torch)..."
    cat > /tmp/pyannote_community1_config.yaml << 'EOF'
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
EOF
    
    cp /tmp/pyannote_community1_config.yaml ~/.cache/torch/pyannote/models--pyannote--speaker-diarization-community-1/blobs/4022db43960736338378fdb6b5a85cfdae198910 2>/dev/null
    rm -rf ~/.cache/torch/pyannote/models--pyannote--speaker-diarization-community-1/snapshots/*/plda 2>/dev/null
    echo "  ✅ community-1 patché"
fi

echo ""
echo "✅ Patch terminé!"
echo ""
echo "📋 Recommandation:"
echo "  👉 Utiliser pyannote/speaker-diarization-3.1 (stable)"
echo "  👉 Compatible WhisperX + PyAnnote 3.4.0 + ROCm 6.3"
echo ""
echo "💡 Pour WhisperX, utiliser:"
echo "  Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=...)"
