#!/usr/bin/env python3
"""
Génère un tableau LaTeX comparatif de tous les résultats
Usage: python3 generate_comparison_table.py
"""

import os
import re
from pathlib import Path


def parse_metrics_file(filepath):
    """Parse un fichier .metrics et extrait les métriques"""
    with open(filepath, "r") as f:
        content = f.read()

    metrics = {}
    metrics["accuracy"] = float(re.search(r"Accuracy:\s+([\d.]+)", content).group(1))
    metrics["precision"] = float(re.search(r"Precision:\s+([\d.]+)", content).group(1))
    metrics["recall"] = float(re.search(r"Recall:\s+([\d.]+)", content).group(1))
    metrics["f1"] = float(re.search(r"F1-score:\s+([\d.]+)", content).group(1))

    return metrics


def generate_latex_table(results_dir="./results"):
    """Génère un tableau LaTeX avec tous les résultats"""

    results = []

    for metrics_file in Path(results_dir).glob("*.metrics"):
        filename = metrics_file.stem
        parts = filename.split("-")

        lang = parts[0]
        corpus = parts[1]
        method = "-".join(parts[2:])

        metrics = parse_metrics_file(metrics_file)

        results.append({"lang": lang, "corpus": corpus, "method": method, **metrics})

    # Trier par F1 décroissant
    results.sort(key=lambda x: x["f1"], reverse=True)

    # Générer LaTeX
    latex = r"""
\begin{table}[h]
\centering
\caption{Résultats comparatifs sur le corpus NRB}
\begin{tabular}{llcccc}
\toprule
Langue & Modèle & Acc. & Prec. & Rappel & F1 \\
\midrule
"""

    for r in results:
        if r["corpus"] == "nrb":
            latex += f"{r['lang']} & {r['method']} & {r['accuracy']:.4f} & "
            latex += f"{r['precision']:.4f} & {r['recall']:.4f} & "
            latex += f"\\textbf{{{r['f1']:.4f}}} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    # Aussi générer version Markdown
    markdown = "| Langue | Modèle | Accuracy | Precision | Recall | F1-Score |\n"
    markdown += "|--------|--------|----------|-----------|--------|----------|\n"

    for r in results:
        if r["corpus"] == "nrb":
            markdown += f"| {r['lang']} | {r['method']} | {r['accuracy']:.4f} | "
            markdown += (
                f"{r['precision']:.4f} | {r['recall']:.4f} | **{r['f1']:.4f}** |\n"
            )

    # Sauvegarder
    with open(f"{results_dir}/comparison_table.tex", "w") as f:
        f.write(latex)

    with open(f"{results_dir}/comparison_table.md", "w") as f:
        f.write(markdown)

    print("✓ Tableaux générés:")
    print(f"  - {results_dir}/comparison_table.tex")
    print(f"  - {results_dir}/comparison_table.md")

    return results


if __name__ == "__main__":
    results = generate_latex_table()

    print("\nTop 3 modèles:")
    for i, r in enumerate(results[:3], 1):
        print(f"  {i}. {r['method']}: F1={r['f1']:.4f}")
