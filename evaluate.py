#!/usr/bin/env python3
"""
Script pour évaluer les prédictions par rapport au gold standard
Usage: python3 evaluate.py gold_file prediction_file
"""

import sys
import re
from collections import defaultdict


def load_gold_labels(filename):
    """Charge les labels gold depuis le fichier NRB"""
    labels = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"_\{([^\}]+)\}", line)
            if match:
                labels.append(match.group(1))
    return labels


def load_predictions(filename):
    """Charge les prédictions"""
    predictions = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            pred = line.strip()
            predictions.append(pred if pred != "None" else None)
    return predictions


def calculate_metrics(gold, pred):
    """Calcule précision, rappel, F1"""
    assert len(gold) == len(pred), "Longueurs différentes!"

    correct = 0
    total = len(gold)

    # Métriques par classe
    class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for g, p in zip(gold, pred):
        if g == p:
            correct += 1
            if p is not None:
                class_stats[p]["tp"] += 1
        else:
            if p is not None:
                class_stats[p]["fp"] += 1
            if g is not None:
                class_stats[g]["fn"] += 1

    accuracy = correct / total

    # Calcul micro-averaged F1
    total_tp = sum(s["tp"] for s in class_stats.values())
    total_fp = sum(s["fp"] for s in class_stats.values())
    total_fn = sum(s["fn"] for s in class_stats.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_stats": dict(class_stats),
    }


def print_confusion_matrix(gold, pred):
    """Affiche la matrice de confusion"""
    from collections import Counter

    classes = sorted(set(gold + pred) - {None})
    matrix = defaultdict(Counter)

    for g, p in zip(gold, pred):
        matrix[g][p] += 1

    print("\nMatrice de confusion:")
    print(f"{'':15s}", end="")
    for c in classes + [None]:
        print(f"{str(c):15s}", end="")
    print()

    for g in classes + [None]:
        print(f"{str(g):15s}", end="")
        for p in classes + [None]:
            print(f"{matrix[g][p]:15d}", end="")
        print()


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 evaluate.py gold_file prediction_file")
        sys.exit(1)

    gold = load_gold_labels(sys.argv[1])
    pred = load_predictions(sys.argv[2])

    metrics = calculate_metrics(gold, pred)

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")

    print("\nMétriques par classe:")
    for cls, stats in metrics["class_stats"].items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{cls:15s} - P: {prec:.4f}, R: {rec:.4f}, F1: {f1:.4f}")

    print_confusion_matrix(gold, pred)


if __name__ == "__main__":
    main()
