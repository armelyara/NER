#!/usr/bin/env python3
"""
Test du support GPU sur Mac M2 Pro - VERSION CORRIGÉE (sans erreurs)
"""

import torch
import sys
import time

print("=" * 60)
print("TEST ACCÉLÉRATION GPU - MAC M2 PRO")
print("=" * 60)

# Vérifier disponibilité MPS
print("\n[1] Détection du GPU...")
if torch.backends.mps.is_available():
    print("✓ MPS (Metal Performance Shaders) disponible")
    print("  → GPU M2 Pro détecté et utilisable!")
    device = torch.device("mps")
elif torch.cuda.is_available():
    print("✓ CUDA disponible")
    device = torch.device("cuda")
else:
    print("⚠️  Aucun GPU détecté")
    print("  → Utilisation CPU uniquement")
    device = torch.device("cpu")

print(f"\nDevice actif: {device}")
print(f"PyTorch version: {torch.__version__}")

# Vérifier si MPS est vraiment utilisable
if device.type == "mps":
    try:
        x = torch.ones(1, device=device)
        print("✓ Test MPS basique: OK")
    except Exception as e:
        print(f"❌ MPS détecté mais non fonctionnel: {e}")
        device = torch.device("cpu")
        print("  → Basculement sur CPU")

# ============================================
# TEST DE PERFORMANCE
# ============================================
print("\n" + "=" * 60)
print("TEST DE PERFORMANCE - MULTIPLICATION DE MATRICES")
print("=" * 60)

size = 4096
print(f"\nMultiplication de matrices {size}x{size}...")

# Test CPU
print("\n[2] Test CPU...")
x_cpu = torch.randn(size, size)
y_cpu = torch.randn(size, size)

start = time.time()
z_cpu = torch.matmul(x_cpu, y_cpu)
cpu_time = time.time() - start
print(f"  ⏱️  Temps CPU: {cpu_time:.3f} secondes")

# Test GPU (si disponible)
gpu_time = None  # ← CORRECTION: Initialiser gpu_time
z_gpu = None  # ← CORRECTION: Initialiser z_gpu

if device != torch.device("cpu"):
    print(f"\n[3] Test GPU ({device.type.upper()})...")
    x_gpu = x_cpu.to(device)
    y_gpu = y_cpu.to(device)

    # Warmup (important pour MPS)
    for _ in range(3):
        _ = torch.matmul(x_gpu, y_gpu)

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark réel
    start = time.time()
    for _ in range(5):
        z_gpu = torch.matmul(x_gpu, y_gpu)

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    gpu_time = (time.time() - start) / 5

    print(f"  ⏱️  Temps GPU: {gpu_time:.3f} secondes")
    print(f"  🚀 Accélération: {cpu_time/gpu_time:.1f}x plus rapide!")

    # Validation résultats (seulement si z_gpu existe)
    if z_gpu is not None:  # ← CORRECTION: Vérifier avant utilisation
        z_gpu_cpu = z_gpu.cpu()
        max_diff = torch.max(torch.abs(z_cpu - z_gpu_cpu)).item()
        print(f"  ✓ Différence max CPU vs GPU: {max_diff:.2e} (doit être < 1e-3)")
else:
    print("\n[3] Test GPU: Ignoré (pas de GPU)")

# ============================================
# TEST TRANSFORMERS
# ============================================
print("\n" + "=" * 60)
print("TEST TRANSFORMERS (BERT)")
print("=" * 60)

try:
    from transformers import AutoTokenizer, AutoModel
    import warnings

    warnings.filterwarnings("ignore")

    print("\n[4] Chargement BERT-base...")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Déplacer sur GPU
    model = model.to(device)
    model.eval()

    print(f"  ✓ Modèle chargé sur: {device}")

    # Test inférence
    print("\n[5] Test d'inférence...")
    text = "This is a test sentence to verify GPU acceleration works correctly."
    inputs = tokenizer(text, return_tensors="pt")

    # Déplacer inputs sur GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inférence avec timing
    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    inference_time = time.time() - start

    print(f"  ✓ Inférence réussie")
    print(f"  ⏱️  Temps: {inference_time*1000:.1f} ms")
    print(f"  📊 Output shape: {outputs.last_hidden_state.shape}")

    # Benchmark sur plusieurs phrases
    print("\n[6] Benchmark sur batch...")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Neural networks are powerful machine learning models.",
        "Natural language processing is a fascinating field.",
    ] * 10  # 30 phrases

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    batch_time = time.time() - start

    print(f"  ✓ Traité {len(texts)} phrases")
    print(f"  ⏱️  Temps total: {batch_time:.3f} sec")
    print(f"  📈 Throughput: {len(texts)/batch_time:.1f} phrases/sec")

except ImportError:
    print("\n❌ Transformers non installé")
    print("   Installer avec: pip install transformers")
except Exception as e:
    print(f"\n❌ Erreur lors du test Transformers: {e}")

# ============================================
# TEST FLAIR (OPTIONNEL)
# ============================================
print("\n" + "=" * 60)
print("TEST FLAIR (OPTIONNEL)")
print("=" * 60)

try:
    import flair
    from flair.data import Sentence
    from flair.models import SequenceTagger

    print("\n[7] Configuration Flair pour GPU...")
    if device.type == "mps":
        flair.device = device

    print(f"  ✓ Flair device: {flair.device}")

    print("\n[8] Chargement modèle NER (peut prendre 1-2 min)...")
    tagger = SequenceTagger.load("ner-fast")

    print(f"  ✓ Modèle chargé")

    # Test inférence
    print("\n[9] Test d'inférence Flair...")
    sentence = Sentence("George Washington went to Washington.")

    start = time.time()
    tagger.predict(sentence)
    flair_time = time.time() - start

    print(f"  ✓ Inférence réussie")
    print(f"  ⏱️  Temps: {flair_time*1000:.1f} ms")
    print(f"  🏷️  Entités détectées:")
    for entity in sentence.get_spans("ner"):
        print(f"     - {entity.text}: {entity.tag}")

except ImportError:
    print("\n⚠️  Flair non installé")
    print("   Installer avec: pip install flair")
except Exception as e:
    print(f"\n⚠️  Test Flair ignoré: {e}")

# ============================================
# RÉSUMÉ
# ============================================
print("\n" + "=" * 60)
print("RÉSUMÉ")
print("=" * 60)

print(f"\n📱 Device détecté: {device}")
print(f"🔧 PyTorch version: {torch.__version__}")

if device.type == "mps":
    print(f"\n✅ VOTRE MAC M2 PRO EST PRÊT!")
    print(f"   → Le GPU sera utilisé automatiquement")
    if gpu_time is not None:
        print(f"   → Accélération mesurée: {cpu_time/gpu_time:.1f}x")
    print(f"   → Ollama utilisera le Neural Engine")
elif device.type == "cuda":
    print(f"\n✅ GPU NVIDIA DÉTECTÉ")
    print(f"   → Accélération CUDA active")
elif device.type == "cpu":
    print(f"\n⚠️  MODE CPU UNIQUEMENT")
    print(f"   → Les modèles fonctionneront mais seront plus lents")

print("\n" + "=" * 60)
print("Vous pouvez maintenant lancer les expériences NER")
print("=" * 60 + "\n")
