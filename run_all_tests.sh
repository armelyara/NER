#!/bin/bash
# run_all_experiments.sh - Expériences complètes NER sur NRB
# Temps estimé: 1h30 sur Mac M2 Pro

set -e  # Arrêter en cas d'erreur

DATA_DIR="./data"
OUTPUT_DIR="./results"
mkdir -p $OUTPUT_DIR

# Couleurs pour terminal
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================================"
echo "  EXPÉRIENCES NER - CORPUS NRB"
echo "  Mac M2 Pro avec accélération GPU"
echo "======================================================"

# Vérifier données
if [ ! -f "$DATA_DIR/en.nrb.txt" ]; then
    echo "❌ Fichier en.nrb.txt non trouvé dans $DATA_DIR"
    exit 1
fi

total_lines=$(wc -l < "$DATA_DIR/en.nrb.txt")
echo "📊 Corpus: $total_lines phrases"
echo ""

# Timer global
start_time=$(date +%s)

# ======================================
# PARTIE 1: MODÈLES SPACY (rapides)
# ======================================
echo -e "${BLUE}========== MODÈLES SPACY ==========${NC}"

echo -e "${GREEN}[1/9]${NC} Spacy small..."
time cat "$DATA_DIR/en.nrb.txt" | python3 nrb.py \
    --method="spacy" \
    --model="en_core_web_sm" \
    > "$OUTPUT_DIR/en-nrb-spacy-sm.out"
echo "  ✓ Résultats: en-nrb-spacy-sm.out"

echo -e "${GREEN}[2/9]${NC} Spacy medium..."
time cat "$DATA_DIR/en.nrb.txt" | python3 nrb.py \
    --method="spacy" \
    --model="en_core_web_md" \
    > "$OUTPUT_DIR/en-nrb-spacy-md.out"
echo "  ✓ Résultats: en-nrb-spacy-md.out"

echo -e "${GREEN}[3/9]${NC} Spacy large..."
time cat "$DATA_DIR/en.nrb.txt" | python3 nrb.py \
    --method="spacy" \
    --model="en_core_web_lg" \
    > "$OUTPUT_DIR/en-nrb-spacy-lg.out"
echo "  ✓ Résultats: en-nrb-spacy-lg.out"

# ======================================
# PARTIE 2: MODÈLES TRANSFORMERS
# ======================================
echo ""
echo -e "${BLUE}========== MODÈLES TRANSFORMERS ==========${NC}"

echo -e "${GREEN}[4/9]${NC} BERT-base-NER (avec GPU M2)..."
time cat "$DATA_DIR/en.nrb.txt" | python3 nrb.py \
    --method="transformers" \
    --model="dslim/bert-base-NER" \
    > "$OUTPUT_DIR/en-nrb-bert-base.out"
echo "  ✓ Résultats: en-nrb-bert-base.out"

echo -e "${GREEN}[5/9]${NC} Flair NER-large (avec GPU M2)..."
time cat "$DATA_DIR/en.nrb.txt" | python3 nrb.py \
    --method="flair" \
    --model="ner-large" \
    > "$OUTPUT_DIR/en-nrb-flair-large.out"
echo "  ✓ Résultats: en-nrb-flair-large.out"

# ======================================
# PARTIE 3: GRANDS MODÈLES DE LANGUE
# ======================================
echo ""
echo -e "${BLUE}========== GRANDS MODÈLES DE LANGUE ==========${NC}"
echo -e "${YELLOW}⚠️  Cette partie est lente (~45 min). Vous pouvez:${NC}"
echo -e "${YELLOW}   - Laisser tourner et aller prendre un café${NC}"
echo -e "${YELLOW}   - Lancer en arrière-plan: nohup ./run_all_experiments.sh &${NC}"
echo -e "${YELLOW}   - Ou appuyer Ctrl+C pour tester seulement les 5 premiers modèles${NC}"
echo ""
read -t 10 -p "Continuer avec les LLMs? (Enter=oui, Ctrl+C=non) " || echo ""

echo -e "${GREEN}[6/9]${NC} Llama 3.2 (zero-shot)..."
time cat "$DATA_DIR/en.nrb.txt" | python3 nrb.py \
    --method="ollama" \
    --model="llama3.2:3b" \
    > "$OUTPUT_DIR/en-nrb-llama32-zero.out"
echo "  ✓ Résultats: en-nrb-llama32-zero.out"

echo -e "${GREEN}[7/9]${NC} Llama 3.2 (few-shot)..."
time cat "$DATA_DIR/en.nrb.txt" | python3 nrb.py \
    --method="ollama_fewshot" \
    --model="llama3.2:3b" \
    > "$OUTPUT_DIR/en-nrb-llama32-few.out"
echo "  ✓ Résultats: en-nrb-llama32-few.out"

echo -e "${GREEN}[8/9]${NC} Mistral 7B..."
time cat "$DATA_DIR/en.nrb.txt" | python3 nrb.py \
    --method="ollama" \
    --model="mistral:7b" \
    > "$OUTPUT_DIR/en-nrb-mistral-zero.out"
echo "  ✓ Résultats: en-nrb-mistral-zero.out"

echo -e "${GREEN}[9/9]${NC} Llama 3.1 8B..."
time cat "$DATA_DIR/en.nrb.txt" | python3 nrb.py \
    --method="ollama" \
    --model="llama3.1:8b" \
    > "$OUTPUT_DIR/en-nrb-llama31-zero.out"
echo "  ✓ Résultats: en-nrb-llama31-zero.out"

# ======================================
# PARTIE 4: CORPUS WTS (optionnel)
# ======================================
if [ -f "$DATA_DIR/en.wts.txt" ]; then
    echo ""
    echo -e "${BLUE}========== CORPUS WTS (WITNESS) ==========${NC}"
    
    echo "Test Spacy-lg sur WTS..."
    cat "$DATA_DIR/en.wts.txt" | python3 nrb.py \
        --method="spacy" \
        --model="en_core_web_lg" \
        > "$OUTPUT_DIR/en-wts-spacy-lg.out"
    
    echo "Test BERT sur WTS..."
    cat "$DATA_DIR/en.wts.txt" | python3 nrb.py \
        --method="transformers" \
        --model="dslim/bert-base-NER" \
        > "$OUTPUT_DIR/en-wts-bert-base.out"
    
    echo "  ✓ Tests WTS terminés"
fi

# ======================================
# PARTIE 5: ÉVALUATION
# ======================================
echo ""
echo -e "${BLUE}========== ÉVALUATION DES RÉSULTATS ==========${NC}"

for out_file in "$OUTPUT_DIR"/en-nrb-*.out; do
    if [ -f "$out_file" ]; then
        base_name=$(basename "$out_file" .out)
        echo "Évaluation: $base_name..."
        python3 evaluate.py "$DATA_DIR/en.nrb.txt" "$out_file" > "${out_file%.out}.metrics"
    fi
done

echo "  ✓ Évaluation terminée"

# ======================================
# PARTIE 6: ANALYSE & VISUALISATION
# ======================================
echo ""
echo -e "${BLUE}========== ANALYSE & VISUALISATION ==========${NC}"

# Tableaux comparatifs
if [ -f "generate_comparison_table.py" ]; then
    echo "Génération des tableaux comparatifs..."
    python3 generate_comparison_table.py
    echo "  ✓ Tableaux générés: results/comparison_table.md"
fi

# Graphiques
if [ -f "generate_plots.py" ]; then
    echo "Génération des graphiques..."
    python3 generate_plots.py "$DATA_DIR/en.nrb.txt" "$OUTPUT_DIR/en-nrb-bert-base.out"
    echo "  ✓ Graphiques générés"
fi

# Analyse du biais
if [ -f "analyze_bias.py" ]; then
    echo "Analyse du biais sur BERT..."
    python3 analyze_bias.py "$DATA_DIR/en.nrb.txt" "$OUTPUT_DIR/en-nrb-bert-base.out" > "$OUTPUT_DIR/bias_analysis.txt"
    echo "  ✓ Analyse sauvegardée: results/bias_analysis.txt"
fi

# Timer global
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

# ======================================
# RÉSUMÉ FINAL
# ======================================
echo ""
echo "======================================================"
echo "  ✅ TOUTES LES EXPÉRIENCES TERMINÉES!"
echo "======================================================"
echo ""
echo "⏱️  Temps total: ${minutes}m ${seconds}s"
echo ""
echo "📁 Résultats générés:"
ls -1 "$OUTPUT_DIR"/*.out | wc -l | xargs echo "  - Fichiers .out:"
ls -1 "$OUTPUT_DIR"/*.metrics 2>/dev/null | wc -l | xargs echo "  - Fichiers .metrics:"
echo ""
echo "📊 Voir les performances:"
echo "  cat results/comparison_table.md"
echo ""
echo "📈 Voir les graphiques:"
echo "  open results/f1_comparison.png  # ou xdg-open sur Linux"
echo ""
echo "📝 Top 3 modèles par F1-score:"
grep "F1-score" "$OUTPUT_DIR"/*.metrics | sort -t: -k2 -nr | head -3
echo ""
echo "🎯 Prochaine étape: Écrire le rapport!"
echo "======================================================"