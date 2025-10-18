#!/bin/bash
# Installation complète avec vérifications pour Mac M2 Pro

set -e  # Arrêter en cas d'erreur

echo "======================================================"
echo "  INSTALLATION NER - MAC M2 PRO"
echo "======================================================"
echo ""

# Vérifier architecture
echo "[1/10] Vérification de l'architecture..."
if [[ $(uname -m) != "arm64" ]]; then
    echo "❌ Ce script nécessite Apple Silicon (M1/M2/M3)"
    exit 1
fi
echo "  ✓ Architecture: arm64 (Apple Silicon)"

# Vérifier version macOS
echo ""
echo "[2/10] Vérification macOS..."
macos_version=$(sw_vers -productVersion)
echo "  ✓ macOS version: $macos_version"

if [[ "$macos_version" < "12.3" ]]; then
    echo "  ⚠️  MPS nécessite macOS 12.3+. Mise à jour recommandée."
fi

# Vérifier Python
echo ""
echo "[3/10] Vérification Python..."
if ! command -v python3 &> /dev/null; then
    echo "Python3 non trouvé. Installez avec: brew install python3"
    exit 1
fi
python_version=$(python3 --version)
echo "  ✓ $python_version"

# Créer environnement virtuel
#echo ""
####echo "  ✓ Environnement 'ner_env' créé et activé"

# Upgrade pip
echo ""
echo "[5/10] Mise à jour de pip..."
pip install --upgrade pip --quiet
echo "  ✓ pip mis à jour"

# Installer PyTorch avec MPS
echo ""
echo "[6/10] Installation PyTorch (avec support MPS)..."
pip install torch torchvision torchaudio --quiet
echo "  ✓ PyTorch installé"

# Vérifier MPS
echo ""
echo "[7/10] Vérification support MPS..."
python3 -c "import torch; print('  ✓ MPS disponible' if torch.backends.mps.is_available() else '  ⚠️  MPS non disponible')"

# Installer Spacy
echo ""
echo "[8/10] Installation Spacy + modèles..."
pip install spacy --quiet
python3 -m spacy download en_core_web_sm --quiet
python3 -m spacy download en_core_web_md --quiet
python3 -m spacy download en_core_web_lg --quiet
echo "  ✓ Spacy et 3 modèles installés"

# Installer Flair
echo ""
echo "[9/10] Installation Flair..."
pip install flair --quiet
echo "  ✓ Flair installé"

# Installer Transformers
echo ""
echo "[10/10] Installation Transformers..."
pip install transformers accelerate --quiet
echo "  ✓ Transformers installé"

# Installer utilitaires
echo ""
echo "[11/10] Installation utilitaires..."
pip install numpy pandas scikit-learn matplotlib seaborn tqdm --quiet
echo "  ✓ Utilitaires installés"

# Installer Ollama
echo ""
echo "[12/10] Vérification Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "  ⏳ Installation Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "  ✓ Ollama installé"
else
    echo "  ✓ Ollama déjà installé"
fi

# Télécharger modèles LLM
echo ""
echo "[13/10] Téléchargement des modèles LLM..."
echo "  (Cela peut prendre 5-10 minutes selon votre connexion)"

echo "  → llama3.2:3b (1.6 GB)..."
ollama pull llama3.2:3b 2>&1 | grep -E "success|error" || true

echo "  → mistral:7b (4.1 GB)..."
ollama pull mistral:7b 2>&1 | grep -E "success|error" || true

echo "  → llama3.1:8b (4.7 GB)..."
ollama pull llama3.1:8b 2>&1 | grep -E "success|error" || true

echo "  ✓ Modèles LLM téléchargés"

# Résumé
echo ""
echo "======================================================"
echo "  ✅ INSTALLATION TERMINÉE!"
echo "======================================================"
echo ""
echo "📦 Packages installés:"
echo "  - PyTorch (avec MPS)"
echo "  - Spacy (sm, md, lg)"
echo "  - Flair"
echo "  - Transformers"
echo "  - Ollama + 3 modèles LLM"
#echo ""
#echo "🧪 Lancer le test:"
#echo "  source ner_env/bin/activate"
#echo "  python3 test_mps.py"
#echo ""
echo "🚀 Lancer les expériences:"
echo "  ./run_all_tests.sh"
echo ""
echo "======================================================"