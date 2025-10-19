#!/usr/bin/env python3
"""Analyse du biais de nom sur le corpus NRB"""

import re
from collections import defaultdict

# Charger gold labels et entités
gold_file = "data/en.nrb.txt"
pred_file = "results/en-nrb-bert-base.out"

entities = {}
with open(gold_file, 'r') as f:
    for line in f:
        match = re.search(r'\[([^\]]+)\]_\{([^\}]+)\}', line)
        if match:
            entity = match.group(1)
            etype = match.group(2)
            if entity not in entities:
                entities[entity] = set()
            entities[entity].add(etype)

# Entités ambiguës
ambiguous = {e: types for e, types in entities.items() if len(types) > 1}

# Charger prédictions
gold_labels = []
with open(gold_file, 'r') as f:
    for line in f:
        match = re.search(r'_\{([^\}]+)\}', line)
        if match:
            gold_labels.append(match.group(1))

pred_labels = []
with open(pred_file, 'r') as f:
    for line in f:
        line = line.strip()
        pred_labels.append(line if line and line != "None" else None)

# Analyser erreurs
min_len = min(len(gold_labels), len(pred_labels))
gold_labels = gold_labels[:min_len]
pred_labels = pred_labels[:min_len]

confusion = defaultdict(int)
for g, p in zip(gold_labels, pred_labels):
    if g != p:
        confusion[(g, p)] += 1

# Générer rapport
with open('rapport_data/analyse_biais.md', 'w') as f:
    f.write("# Analyse du Biais de Nom\n\n")
    f.write(f"## Statistiques Générales\n\n")
    f.write(f"- Entités uniques: {len(entities)}\n")
    f.write(f"- Entités ambiguës: {len(ambiguous)}\n")
    f.write(f"- Taux d'ambiguïté: {len(ambiguous)/len(entities)*100:.1f}%\n\n")
    
    f.write(f"## Top 10 Entités Ambiguës\n\n")
    f.write("| Entité | Types observés |\n")
    f.write("|--------|----------------|\n")
    for entity, types in sorted(ambiguous.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        f.write(f"| {entity} | {', '.join(sorted(types))} |\n")
    
    f.write(f"\n## Top 5 Erreurs Fréquentes (BERT)\n\n")
    f.write("| Gold | Prédiction | Fréquence |\n")
    f.write("|------|------------|----------|\n")
    for (gold, pred), count in sorted(confusion.items(), key=lambda x: x[1], reverse=True)[:5]:
        f.write(f"| {gold} | {pred or 'None'} | {count} |\n")

print("Analyse du biais terminée!")
