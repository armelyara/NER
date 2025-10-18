#!/usr/bin/env python3
"""
Génère des graphiques pour le rapport
Usage: python3 generate_plots.py
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re


def plot_f1_comparison(results_dir='./results'):
    """Graphique en barres des F1-scores"""
    
    models = []
    f1_scores = []
    
    for metrics_file in sorted(Path(results_dir).glob('en-nrb-*.metrics')):
        with open(metrics_file, 'r') as f:
            content = f.read()
        
        match = re.search(r'F1-score:\s+([\d.]+)', content)
        if match:
            f1 = float(match.group(1))
        else:
            print(f"Warning: no F1-score found in {metrics_file}, skipping")
            continue

        model_name = metrics_file.stem.replace('en-nrb-', '').replace('-', ' ').title()
        
        models.append(model_name)
        f1_scores.append(f1)
    
    # Créer graphique
    plt.figure(figsize=(12, 6))
    colors = ['#3498db' if 'Spacy' in m else '#e74c3c' if 'Llama' in m or 'Mistral' in m else '#2ecc71' for m in models]
    bars = plt.bar(range(len(models)), f1_scores, color=colors, alpha=0.8)
    
    plt.xlabel('Modèle NER', fontsize=12, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
    plt.title('Comparaison des Performances sur NRB (English)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.ylim(0.5, 0.85)
    plt.grid(axis='y', alpha=0.3)
    
    # Ajouter valeurs sur barres
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/f1_comparison.png', dpi=300)
    print(f"✓ Graphique sauvegardé: {results_dir}/f1_comparison.png")
    plt.close()


def plot_confusion_matrix(gold_file, pred_file, output_file):
    """Génère une heatmap de matrice de confusion"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Charger données
    gold = []
    with open(gold_file, 'r') as f:
        for line in f:
            match = re.search(r'_\{([^\}]+)\}', line)
            if match:
                gold.append(match.group(1))
    
    pred = []
    with open(pred_file, 'r') as f:
        for line in f:
            p = line.strip()
            pred.append(p if p != "None" else "NONE")
    
    # Matrice de confusion
    labels = ['PERSON', 'ORGANIZATION', 'LOCATION', 'NONE']
    cm = confusion_matrix([g if g in labels else "NONE" for g in gold],
                          [p if p in labels else "NONE" for p in pred],
                          labels=labels)
    
    # Normaliser par ligne (rappel par classe)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Créer heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Proportion'})
    plt.ylabel('Type Gold', fontsize=12, fontweight='bold')
    plt.xlabel('Type Prédit', fontsize=12, fontweight='bold')
    plt.title('Matrice de Confusion Normalisée', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


if __name__ == "__main__":
    import sys
    
    plot_f1_comparison()
    
    if len(sys.argv) >= 3:
        # python3 generate_plots.py data/en.nrb.txt results/en-nrb-bert-base.out
        plot_confusion_matrix(sys.argv[1], sys.argv[2], 'results/confusion_matrix.png')
