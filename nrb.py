#!/usr/bin/env python3
"""
Programme d'évaluation de systèmes NER sur le corpus NRB
OPTIMISÉ POUR MAC M2 PRO avec accélération GPU (MPS)

Auteurs: Rui Cong Su, Armel Yara 
Cours: IFT6285 - TALN
Date: Octobre 2025

Usage:
    cat en.nrb.txt | python3 nrb.py --method="spacy" --model="en_core_web_lg"
    cat en.nrb.txt | python3 nrb.py --method="transformers" --model="dslim/bert-base-NER"
    cat en.nrb.txt | python3 nrb.py --method="ollama" --model="llama3.2:3b"
    cat en.nrb.txt | python3 nrb.py --method="ollama_fewshot" --model="mistral:7b"
"""

import sys
import re
import argparse
from typing import Optional, Tuple, Any
import warnings

warnings.filterwarnings("ignore")

# ========================================
# DÉTECTION AUTOMATIQUE DU DEVICE (GPU/CPU)
# ========================================


def get_optimal_device():
    """
    Détecte automatiquement le meilleur device disponible
    - MPS pour Mac M1/M2/M3 (Apple Silicon)
    - CUDA pour GPU NVIDIA
    - CPU sinon
    """
    try:
        import torch

        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("✓ Utilisation GPU M2 Pro (MPS)", file=sys.stderr)
            return device
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("✓ Utilisation GPU CUDA", file=sys.stderr)
            return device
        else:
            device = torch.device("cpu")
            print("⚠️  Utilisation CPU uniquement", file=sys.stderr)
            return device
    except ImportError:
        print("⚠️  PyTorch non installé, utilisation CPU", file=sys.stderr)
        return None


# Détecter le device au démarrage (variable globale)
DEVICE = get_optimal_device()


# ========================================
# EXTRACTION DE L'ENTITÉ CIBLE
# ========================================


def extract_target_entity(
    line: str,
) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[int], Optional[str]]:
    """
    Extrait l'entité cible et son type gold de la ligne NRB

    Args:
        line: Ligne du format "[entity]_{TYPE} reste de la phrase"

    Returns:
        (sentence, target_text, target_start, target_end, gold_type)
    """
    # Pattern: [entity]_{TYPE}
    pattern = r"\[([^\]]+)\]_\{([^\}]+)\}"
    match = re.search(pattern, line)

    if not match:
        return None, None, None, None, None

    target_text = match.group(1)
    gold_type = match.group(2)

    # Reconstruire la phrase sans annotations
    sentence = re.sub(pattern, target_text, line).strip()

    # Trouver position de l'entité dans la phrase nettoyée
    target_start = sentence.find(target_text)
    target_end = target_start + len(target_text)

    return sentence, target_text, target_start, target_end, gold_type


# ========================================
# CHARGEMENT DES MODÈLES
# ========================================


def load_model(method: str, model: str) -> Any:
    """Charge le modèle NER selon la méthode spécifiée avec support GPU M2"""

    if method == "spacy":
        import spacy

        try:
            nlp = spacy.load(model)
            print(f"✓ Modèle Spacy chargé: {model} (CPU)", file=sys.stderr)
            return nlp
        except OSError:
            print(f"❌ Erreur: Modèle {model} non trouvé.", file=sys.stderr)
            print(
                f"   Installez avec: python -m spacy download {model}", file=sys.stderr
            )
            sys.exit(1)

    elif method == "flair":
        from flair.models import SequenceTagger
        import flair

        # Configurer Flair pour utiliser MPS si disponible
        if DEVICE is not None and DEVICE.type == "mps":
            flair.device = DEVICE
            print(f"⏳ Chargement Flair sur GPU M2 (MPS)...", file=sys.stderr)
        else:
            print(f"⏳ Chargement Flair sur CPU...", file=sys.stderr)

        tagger = SequenceTagger.load(model)
        print(f"✓ Modèle Flair chargé: {model} ({flair.device})", file=sys.stderr)
        return tagger

    elif method == "transformers":
        from transformers import pipeline as hf_pipeline

        # Configurer device pour Transformers
        if DEVICE is not None:
            if DEVICE.type == "mps":
                device_id = 0
                print(
                    f"⏳ Chargement Transformers sur GPU M2 (MPS)...", file=sys.stderr
                )
            elif DEVICE.type == "cuda":
                device_id = 0
                print(f"⏳ Chargement Transformers sur GPU CUDA...", file=sys.stderr)
            else:
                device_id = -1
                print(f"⏳ Chargement Transformers sur CPU...", file=sys.stderr)
        else:
            device_id = -1
            print(f"⏳ Chargement Transformers sur CPU...", file=sys.stderr)

        # Créer le pipeline NER
        ner_pipeline = hf_pipeline(
            task="token-classification",
            model=model,
            aggregation_strategy="simple",
            device=device_id,
        )  # type: ignore

        device_str = (
            "GPU M2 (MPS)"
            if device_id == 0 and DEVICE and DEVICE.type == "mps"
            else (
                "GPU CUDA"
                if device_id == 0 and DEVICE and DEVICE.type == "cuda"
                else "CPU"
            )
        )
        print(f"✓ Modèle Transformers chargé: {model} ({device_str})", file=sys.stderr)
        return ner_pipeline

    elif method in ["ollama", "ollama_fewshot"]:
        try:
            import ollama

            # Vérifier que le modèle est disponible (avec gestion d'erreur robuste)
            try:
                models_response = ollama.list()

                # Gérer différents formats de réponse
                if isinstance(models_response, dict):
                    models_list = models_response.get("models", [])
                else:
                    models_list = models_response

                # Extraire les noms de modèles de manière robuste
                available = []
                for m in models_list:
                    if isinstance(m, dict):
                        # Essayer différentes clés possibles
                        model_name = m.get("name") or m.get("model") or m.get("id")
                        if model_name:
                            available.append(model_name)
                    elif isinstance(m, str):
                        available.append(m)

                # Vérification flexible (avec ou sans tag de version)
                model_found = False
                for avail_model in available:
                    if model in avail_model or avail_model in model:
                        model_found = True
                        break

                if not model_found and len(available) > 0:
                    print(f"⚠️  Modèle {model} non trouvé.", file=sys.stderr)
                    print(f"   Modèles disponibles:", file=sys.stderr)
                    for m in available[:10]:  # Limiter à 10 pour lisibilité
                        print(f"     - {m}", file=sys.stderr)
                    print(f"   Téléchargez avec: ollama pull {model}", file=sys.stderr)
                    sys.exit(1)
                elif len(available) == 0:
                    print(f"⚠️  Aucun modèle Ollama détecté.", file=sys.stderr)
                    print(
                        f"   Assurez-vous qu'Ollama est démarré: ollama serve",
                        file=sys.stderr,
                    )
                    print(f"   Puis téléchargez: ollama pull {model}", file=sys.stderr)

            except Exception as e:
                print(
                    f"⚠️  Impossible de lister les modèles Ollama: {e}", file=sys.stderr
                )
                print(
                    f"   Vérifiez qu'Ollama est démarré: ollama serve", file=sys.stderr
                )
                print(f"   Tentative de continuer avec {model}...", file=sys.stderr)

            # Ollama utilise automatiquement le Neural Engine sur M2
            print(
                f"✓ Ollama prêt avec modèle: {model} (Neural Engine M2)",
                file=sys.stderr,
            )
            return model

        except ImportError:
            print(
                f"❌ Ollama non installé. Visitez: https://ollama.com/download",
                file=sys.stderr,
            )
            sys.exit(1)
    elif method in ["openai","openai_fewshot"]:
        return model

    else:
        print(f"❌ Méthode {method} non supportée", file=sys.stderr)
        print(
            f"   Méthodes disponibles: spacy, flair, transformers, ollama, ollama_fewshot, openai, openai_fewshot",
            file=sys.stderr,
        )
        sys.exit(1)


# ========================================
# PRÉDICTIONS - MODÈLES DÉDIÉS NER
# ========================================


def predict_spacy(
    nlp: Any, sentence: str, target_start: int, target_end: int
) -> Optional[str]:
    """Prédiction avec Spacy"""
    doc = nlp(sentence)

    for ent in doc.ents:
        if (
            (ent.start_char <= target_start < ent.end_char)
            or (ent.start_char < target_end <= ent.end_char)
            or (target_start <= ent.start_char and ent.end_char <= target_end)
        ):
            label_map = {
                "PERSON": "PERSON",
                "ORG": "ORGANIZATION",
                "GPE": "LOCATION",
                "LOC": "LOCATION",
                "FAC": "LOCATION",
                "NORP": None,
                "EVENT": None,
                "WORK_OF_ART": None,
            }
            return label_map.get(ent.label_, None)

    return None


def predict_flair(
    tagger: Any, sentence: str, target_start: int, target_end: int
) -> Optional[str]:
    """Prédiction avec Flair"""
    from flair.data import Sentence as FlairSentence

    flair_sentence = FlairSentence(sentence)
    tagger.predict(flair_sentence)

    for entity in flair_sentence.get_spans("ner"):
        if (
            (entity.start_position <= target_start < entity.end_position)
            or (entity.start_position < target_end <= entity.end_position)
            or (
                target_start <= entity.start_position
                and entity.end_position <= target_end
            )
        ):
            label_map = {
                "PER": "PERSON",
                "ORG": "ORGANIZATION",
                "LOC": "LOCATION",
                "MISC": None,
            }
            return label_map.get(entity.tag, None)

    return None


def predict_transformers(
    ner_pipeline: Any, sentence: str, target_start: int, target_end: int
) -> Optional[str]:
    """Prédiction avec Transformers (Hugging Face)"""
    results = ner_pipeline(sentence)

    for entity in results:
        ent_start = entity["start"]
        ent_end = entity["end"]

        if (
            (ent_start <= target_start < ent_end)
            or (ent_start < target_end <= ent_end)
            or (target_start <= ent_start and ent_end <= target_end)
        ):
            label = entity["entity_group"]
            label_map = {
                "PER": "PERSON",
                "ORG": "ORGANIZATION",
                "LOC": "LOCATION",
                "MISC": None,
            }
            return label_map.get(label, None)

    return None


# ========================================
# PRÉDICTIONS - GRANDS MODÈLES DE LANGUE
# ========================================
SYSTEM_PROMPT = """You are a Named Entity Recognition system. 
Given the sentence and the target entity, classify the target as exactly one of: PERSON, ORGANIZATION, or LOCATION.

Rules:
- Output ONLY one of these three words: PERSON, ORGANIZATION, LOCATION
- Do not explain, do not add punctuation
- Base your answer on the context of the sentence"""

USER_PROMPT_ZERO_SHOT = """Sentence: %s
Target entity: %s

Classification:
"""

PROMPT_ZERO_SHOT = """You are a Named Entity Recognition system. 
Given the sentence and the target entity, classify the target as exactly one of: PERSON, ORGANIZATION, or LOCATION.

Rules:
- Output ONLY one of these three words: PERSON, ORGANIZATION, LOCATION
- Do not explain, do not add punctuation
- Base your answer on the context of the sentence

Sentence: %s
Target entity: %s

Classification:"""

PROMPT_FEW_SHOT = """You are a Named Entity Recognition system. Classify entities based on context.

Example 1:
Sentence: "Throughout the 20th century, Coutts opened more branches."
Target: Coutts
Answer: ORGANIZATION

Example 2:
Sentence: "Dahmer is an unincorporated community located in Pendleton County, West Virginia."
Target: Dahmer
Answer: LOCATION

Example 3:
Sentence: "Obama served two terms as president of the United States."
Target: Obama
Answer: PERSON

Example 4:
Sentence: "Masamune's father Date Terumune entered Miyamori Castle shortly after Masamune entered Obama."
Target: Obama
Answer: LOCATION

Now classify this:
Sentence: %s
Target: %s
Answer:"""


def predict_ollama_zero(
    model_name: str, sentence: str, target_text: str
) -> Optional[str]:
    """Prédiction avec Ollama en zero-shot"""
    import ollama

    prompt = PROMPT_ZERO_SHOT % (sentence, target_text)

    try:
        response = ollama.generate(model=model_name, prompt=prompt)
        prediction = response["response"].strip().upper()

        if "PERSON" in prediction:
            return "PERSON"
        elif "ORGANIZATION" in prediction:
            return "ORGANIZATION"
        elif "LOCATION" in prediction:
            return "LOCATION"
        else:
            return None
    except Exception as e:
        print(f"❌ Erreur Ollama: {e}", file=sys.stderr)
        return None


def predict_ollama_fewshot(
    model_name: str, sentence: str, target_text: str
) -> Optional[str]:
    """Prédiction avec Ollama en few-shot learning"""
    import ollama

    prompt = PROMPT_FEW_SHOT % (sentence,target_text)

    try:
        response = ollama.generate(model=model_name, prompt=prompt)
        prediction = response["response"].strip().upper()

        if "PERSON" in prediction:
            return "PERSON"
        elif "ORGANIZATION" in prediction:
            return "ORGANIZATION"
        elif "LOCATION" in prediction:
            return "LOCATION"
        else:
            return None
    except Exception as e:
        print(f"❌ Erreur Ollama: {e}", file=sys.stderr)
        return None
    
def predict_openai_zero(
    model_name: str, sentence: str, target_text: str
) -> Optional[str]:
    """Prédiction avec OpenAI en zero-shot"""
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_KEY"])

    user_prompt = USER_PROMPT_ZERO_SHOT % (sentence,target_text)

    try:
        response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                },
            {
                "role": "user",
                "content": user_prompt}
            ]
        )
        output = response.choices[0].message.content
        #print(output)
        prediction = output.strip().upper()

        if "PERSON" in prediction:
            return "PERSON"
        elif "ORGANIZATION" in prediction:
            return "ORGANIZATION"
        elif "LOCATION" in prediction:
            return "LOCATION"
        else:
            return None
    except Exception as e:
        print(f"❌ Erreur OpenAI: {e}", file=sys.stderr)
        return None
    
def predict_openai_fewshot(
    model_name: str, sentence: str, target_text: str
) -> Optional[str]:
    """Prédiction avec OpenAI en few-shot"""
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPEN_AI_KEY"])

    user_prompt = PROMPT_FEW_SHOT % (sentence,target_text)

    try:
        response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                },
            {
                "role": "user",
                "content": user_prompt}
            ]
        )
        output = response.choices[0].message.content
        prediction = output.strip().upper()

        if "PERSON" in prediction:
            return "PERSON"
        elif "ORGANIZATION" in prediction:
            return "ORGANIZATION"
        elif "LOCATION" in prediction:
            return "LOCATION"
        else:
            return None
    except Exception as e:
        print(f"❌ Erreur OpenAI: {e}", file=sys.stderr)
        return None


# ========================================
# FONCTION PRINCIPALE
# ========================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évaluation NER sur corpus NRB - Optimisé Mac M2 Pro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  cat en.nrb.txt | python3 nrb.py --method="spacy" --model="en_core_web_lg"
  cat en.nrb.txt | python3 nrb.py --method="transformers" --model="dslim/bert-base-NER"
  cat en.nrb.txt | python3 nrb.py --method="flair" --model="ner-large"
  cat en.nrb.txt | python3 nrb.py --method="ollama" --model="llama3.2:3b"
  cat en.nrb.txt | python3 nrb.py --method="ollama_fewshot" --model="mistral:7b"
        """,
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["spacy", "flair", "transformers", "ollama", "ollama_fewshot", "openai", "openai_fewshot"],
        help="Méthode NER à utiliser",
    )
    parser.add_argument("--model", required=True, help="Nom du modèle spécifique")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher les détails de prédiction",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=None,
        help="Optionnel: Limite le nombre de lignes à lire",
    )
    args = parser.parse_args()

    # Afficher info device
    if DEVICE is not None:
        device_name = (
            "GPU M2 Pro (MPS)"
            if DEVICE.type == "mps"
            else "GPU CUDA" if DEVICE.type == "cuda" else "CPU"
        )
        print(f"🖥️  Device: {device_name}", file=sys.stderr)

    # Charger le modèle
    print(f"", file=sys.stderr)
    model = load_model(args.method, args.model)
    print(f"", file=sys.stderr)

    # Statistiques
    total = 0
    processed = 0
    errors = 0

    # Traiter chaque ligne
    for line_num, line in enumerate(sys.stdin, 1):
        line = line.strip()
        if not line:
            continue

        total += 1
        sentence, target_text, target_start, target_end, gold_type = (
            extract_target_entity(line)
        )

        if sentence is None:
            print("None")
            errors += 1
            continue

        try:
            if args.method == "spacy":
                prediction = predict_spacy(model, sentence, target_start, target_end)
            elif args.method == "flair":
                prediction = predict_flair(model, sentence, target_start, target_end)
            elif args.method == "transformers":
                prediction = predict_transformers(
                    model, sentence, target_start, target_end
                )
            elif args.method == "ollama":
                prediction = predict_ollama_zero(args.model, sentence, target_text)
            elif args.method == "ollama_fewshot":
                prediction = predict_ollama_fewshot(args.model, sentence, target_text)
            elif args.method == "openai":
                prediction = predict_openai_zero(args.model, sentence, target_text)
            elif args.method == "openai_fewshot":
                prediction = predict_openai_fewshot(args.model, sentence, target_text)
            else:
                prediction = None

            processed += 1

            if args.verbose:
                match = "✓" if prediction == gold_type else "✗"
                pred_str = prediction if prediction else "None"
                print(
                    f"[{line_num:4d}] {match} Gold: {gold_type:15s} Pred: {pred_str:15s} Entity: {target_text}",
                    file=sys.stderr,
                )

        except Exception as e:
            prediction = None
            errors += 1
            if args.verbose:
                print(f"[{line_num}] ❌ {e}", file=sys.stderr)

        print(prediction if prediction else "None")

        if args.lines and processed >= args.lines:
                break

    if args.verbose:
        print(f"\n{'='*60}", file=sys.stderr)
        print(
            f"  Total: {total} | Succès: {processed} | Erreurs: {errors}",
            file=sys.stderr,
        )
        print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()
