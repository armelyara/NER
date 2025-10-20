import os
from collections import Counter, defaultdict
import re

entities = {}
entity_list = []

def read_files(truth_file, output_files):
    """
    Read all files and return their contents as lists of lines
    """
    truth = []
    with open(truth_file, 'r', encoding="utf-8") as f:
        for line in f:
            match = re.search(r'\[([^\]]+)\]_\{([^\}]+)\}', line)
            if match:
                entity = match.group(1)
                etype = match.group(2)
                entity_list.append(entity)
                truth.append(etype)
                if entity not in entities:
                    entities[entity] = set()
                entities[entity].add(etype)
    
    # Read all output files
    outputs = []
    for output_file in output_files:
        with open(output_file, 'r') as f:
            outputs.append([line.strip() for line in f.readlines()])
    
    # Verify all files have the same number of lines
    num_lines = len(truth)
    for i, output in enumerate(outputs):
        if len(output) != num_lines:
            raise ValueError(f"Output file {output_files[i]} has {len(output)} lines, but ground truth has {num_lines}")
    
    return truth, outputs

def find_unanimous_mistakes(truth, outputs):
    """
    Find lines where ALL outputs made the same mistake
    Returns list of (line_number, truth_label, common_mistake)
    """
    num_outputs = len(outputs)
    num_lines = len(truth)
    unanimous_mistakes = []
    
    for line_num in range(num_lines):
        truth_label = truth[line_num]
        
        # Check if all outputs are wrong
        all_wrong = True
        predictions = []
        
        for output in outputs:
            pred = output[line_num]
            predictions.append(pred)
            if pred == truth_label:
                all_wrong = False
                break
        
        # If all outputs are wrong, check if they all made the same mistake
        if all_wrong:
            # Count frequency of each wrong prediction
            pred_counter = Counter(predictions)
            most_common_pred, count = pred_counter.most_common(1)[0]
            
            # Check if all outputs made the same prediction
            if count == num_outputs:
                unanimous_mistakes.append((line_num + 1, truth_label, most_common_pred))
    
    return unanimous_mistakes

def analyze_errors(truth, outputs):
    """
    Analyze errors and agreements between outputs and ground truth
    """
    num_outputs = len(outputs)
    num_lines = len(truth)
    
    # Track errors per line
    line_errors = []
    
    for line_num in range(num_lines):
        truth_label = truth[line_num]
        error_count = 0
        
        # Count how many outputs are wrong for this line
        for output in outputs:
            if output[line_num] != truth_label:
                error_count += 1
        
        line_errors.append((line_num + 1, error_count))  # +1 for 1-based line numbers
    
    return line_errors

def get_most_common_mistakes(line_errors, num_outputs, top_n=20):
    """
    Get lines with the most common mistakes, sorted by error frequency
    """
    # Convert to error rate and filter lines with at least one error
    mistakes_with_rates = [(line_num, error_count/num_outputs) 
                          for line_num, error_count in line_errors 
                          if error_count > 0]
    
    # Sort by error rate descending, then by line number
    sorted_mistakes = sorted(mistakes_with_rates, 
                           key=lambda x: (-x[1], x[0]))
    
    return sorted_mistakes[:top_n]

def get_high_agreement_lines(line_errors, num_outputs, agreement_threshold=0.9):
    """
    Get lines where most outputs agree with ground truth
    """
    high_agreement = []
    
    for line_num, error_count in line_errors:
        agreement_rate = (num_outputs - error_count) / num_outputs
        if agreement_rate >= agreement_threshold:
            high_agreement.append((line_num, agreement_rate))
    
    # Sort by agreement rate descending, then by line number
    return sorted(high_agreement, key=lambda x: (-x[1], x[0]))

def analyze_error_patterns(truth, outputs, mistake_lines):
    """
    Analyze patterns in the most common mistakes
    """
    error_patterns = defaultdict(Counter)
    
    for line_num, _ in mistake_lines:
        idx = line_num - 1  # Convert to 0-based index
        truth_label = truth[idx]
        
        # Count what outputs predicted instead of the truth
        for output in outputs:
            pred = output[idx]
            if pred != truth_label:
                error_patterns[truth_label][pred] += 1
    
    return error_patterns

def generate_bias_report(truth_file, output_files, top_mistakes=20, agreement_threshold=0.9):
    """
    Generate a comprehensive bias report
    """
    print("=== NER System Bias Analysis Report ===\n")
    
    # Read files
    truth, outputs = read_files(truth_file, output_files)
    num_outputs = len(outputs)
    
    print(f"Analyzing {num_outputs} output files against ground truth")
    print(f"Total lines: {len(truth)}\n")
    
    # Find unanimous mistakes
    unanimous_mistakes = find_unanimous_mistakes(truth, outputs)
    
    print(f"--- UNANIMOUS MISTAKES (all {num_outputs} outputs made the SAME error) ---")
    print(f"Found {len(unanimous_mistakes)} lines with unanimous mistakes\n")
    
    if unanimous_mistakes:
        print("Line# | Entity          | Ground Truth | Common Mistake")
        print("-" * 45)
        for line_num, truth_label, common_mistake in unanimous_mistakes:
            print(f"{line_num:5d} | {entity_list[line_num-1]:15s} | {truth_label:12} | {common_mistake}")
        
        # Analyze patterns in unanimous mistakes
        mistake_patterns = Counter()
        for _, truth_label, common_mistake in unanimous_mistakes:
            mistake_patterns[(truth_label, common_mistake)] += 1
        
        print(f"\n--- PATTERNS IN UNANIMOUS MISTAKES ---")
        print("Truth → Mistake | Count")
        print("-" * 30)
        for (truth_label, mistake_label), count in mistake_patterns.most_common():
            print(f"{truth_label:12} → {mistake_label:10} | {count:3d}")
    else:
        print("No lines found where all outputs made the same mistake.")

    # Analyze errors
    line_errors = analyze_errors(truth, outputs)
    
    # Calculate overall statistics
    total_errors = sum(error_count for _, error_count in line_errors)
    total_predictions = num_outputs * len(truth)
    overall_accuracy = (total_predictions - total_errors) / total_predictions
    
    print(f"Overall Agreement Rate: {overall_accuracy:.3f} ({total_predictions - total_errors}/{total_predictions})")
    
    # Get most common mistakes
    common_mistakes = get_most_common_mistakes(line_errors, num_outputs, top_mistakes)
    
    print(f"\n--- TOP {len(common_mistakes)} MOST COMMON MISTAKES ---")
    print("Line# | Entity          | Error Rate | Consensus Level")
    print("-" * 50)
    
    for line_num, error_rate in common_mistakes:
        consensus = 1.0 - error_rate
        print(f"{line_num:5d} | {entity_list[line_num-1]:15s} | {error_rate:10.2%} | {consensus:15.2%}")
    
    # Get high agreement lines
    high_agreement = get_high_agreement_lines(line_errors, num_outputs, agreement_threshold)
    
    print(f"\n--- LINES WITH HIGH AGREEMENT (≥{agreement_threshold:.0%}) ---")
    print(f"Found {len(high_agreement)} lines with high agreement")
    
    # Show top high-agreement lines
    if high_agreement:
        print("\nTop high-agreement lines:")
        print("Line# | Entity          | Agreement Rate")
        print("-" * 30)
        for line_num, agreement_rate in high_agreement[:20]:
            print(f"{line_num:5d} | {entity_list[line_num-1]:15s} | {agreement_rate:14.2%}")
    
    # Analyze error patterns for top mistake lines
    if common_mistakes:
        error_patterns = analyze_error_patterns(truth, outputs, common_mistakes[:10])
        
        print(f"\n--- ERROR PATTERNS FOR TOP MISTAKE LINES ---")
        for truth_label, patterns in error_patterns.items():
            print(f"\nWhen truth is '{truth_label}':")
            total_errors_for_label = sum(patterns.values())
            for pred_label, count in patterns.most_common(5):
                percentage = count / total_errors_for_label
                print(f"  → Mislabeled as '{pred_label}': {count} times ({percentage:.1%})")
    
    # Additional statistics
    error_distribution = Counter(error_count for _, error_count in line_errors)
    
    print(f"\n--- ERROR DISTRIBUTION ---")
    print("Errors per Line | Count of Lines")
    print("-" * 30)
    for error_count in sorted(error_distribution.keys()):
        count = error_distribution[error_count]
        percentage = count / len(truth)
        print(f"{error_count:14d} | {count:13d} ({percentage:.1%})")
    
    return {
        'common_mistakes': common_mistakes,
        'high_agreement_lines': high_agreement,
        'overall_accuracy': overall_accuracy,
        'error_distribution': error_distribution
    }

# Example usage
if __name__ == "__main__":
    # Define your file paths
    ground_truth_file = "./data/en.nrb.txt"
    output_files = [
        "./results/en-nrb-bert-base.out",
        "./results/en-nrb-flair-large.out",
        "./results/en-nrb-llama31-zero.out",
        "./results/en-nrb-llama32-few.out",
        "./results/en-nrb-llama32-zero.out",
        "./results/en-nrb-mistral-zero.out",
        "./results/en-nrb-openai-zero-gpt-5-nano.out",
        "./results/en-nrb-spacy-lg.out",
        "./results/en-nrb-spacy-md.out",
        "./results/en-nrb-spacy-sm.out",
        ]
    
    # Generate the report
    report = generate_bias_report(
        ground_truth_file, 
        output_files,
        top_mistakes=20,           # Show top 20 most common mistakes
        agreement_threshold=0.9    # Consider ≥90% agreement as "high agreement"
    )