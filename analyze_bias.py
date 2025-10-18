#!/usr/bin/env python3
"""
Analyse approfondie du biais de nom
"""

import re
from collections import Counter, defaultdict


def analyze_entity_ambiguity(nrb_file):
    """
    Identifie les entités qui apparaissent avec différents types
    (preuve du biais de nom)
    """
    entity_types = defaultdict(set)

    with open(nrb_file, "r") as f:
        for line in f:
            match = re.search(r"\[([^\]]+)\]_\{([^\}]+)\}", line)
            if match:
                entity = match.group(1)
                etype = match.group(2)
                entity_types[entity].add(etype)

    # Entités ambiguës
    ambiguous = {e: types for e, types in entity_types.items() if len(types) > 1}

    print(f"Entités ambiguës trouvées: {len(ambiguous)}")
    print("\nExemples d'entités avec plusieurs types:")
    for entity, types in list(ambiguous.items())[:10]:
        print(f"  {entity}: {', '.join(types)}")

    return ambiguous


def analyze_error_patterns(gold_file, pred_file):
    """Analyse les patterns d'erreurs"""
    errors = defaultdict(list)

    gold_labels = []
    with open(gold_file, "r") as f:
        for line in f:
            match = re.search(r"\[([^\]]+)\]_\{([^\}]+)\}", line)
            if match:
                entity = match.group(1)
                gold_type = match.group(2)
                sentence = re.sub(r"\[([^\]]+)\]_\{[^\}]+\}", r"\1", line).strip()
                gold_labels.append((entity, gold_type, sentence))

    with open(pred_file, "r") as f:
        preds = [line.strip() for line in f]

    for (entity, gold, sentence), pred in zip(gold_labels, preds):
        if gold != pred:
            errors[(gold, pred)].append((entity, sentence))

    print("\n=== Patterns d'erreurs ===")
    for (gold, pred), examples in sorted(
        errors.items(), key=lambda x: len(x[1]), reverse=True
    ):
        print(f"\n{gold} → {pred}: {len(examples)} erreurs")
        for entity, sent in examples[:3]:
            print(f"  - {entity}: {sent[:80]}...")


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 analyze_bias.py en.nrb.txt [prediction_file]")
        sys.exit(1)

    print("=== Analyse du biais de nom ===\n")
    ambiguous = analyze_entity_ambiguity(sys.argv[1])

    if len(sys.argv) == 3:
        analyze_error_patterns(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
