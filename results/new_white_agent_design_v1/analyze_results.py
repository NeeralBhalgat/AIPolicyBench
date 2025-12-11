#!/usr/bin/env python3
"""
AIPolicyBench Results Analysis Script
Generates comprehensive analysis of model evaluation results.
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Results directory
RESULTS_DIR = Path(__file__).parent

def parse_statistics_file(stats_path):
    """Parse a statistics.txt file and extract metrics."""
    with open(stats_path, 'r') as f:
        content = f.read()
    
    metrics = {}
    
    # Extract model name
    match = re.search(r'White Agent Model: (.+)', content)
    metrics['model'] = match.group(1) if match else 'Unknown'
    
    # Extract timestamp
    match = re.search(r'Timestamp: (\d+_\d+)', content)
    metrics['timestamp'] = match.group(1) if match else 'Unknown'
    
    # Extract total queries
    match = re.search(r'Total Queries: (\d+)', content)
    metrics['total_queries'] = int(match.group(1)) if match else 0
    
    # Extract evaluated count
    match = re.search(r'Evaluated \(excl\. timeout\): (\d+)', content)
    metrics['evaluated'] = int(match.group(1)) if match else 0
    
    # Extract timeout
    match = re.search(r'Timeout: (\d+)', content)
    metrics['timeout'] = int(match.group(1)) if match else 0
    
    # Extract correct
    match = re.search(r'Correct: (\d+) \(([\d.]+)%\)', content)
    if match:
        metrics['correct'] = int(match.group(1))
        metrics['correct_pct'] = float(match.group(2))
    
    # Extract miss
    match = re.search(r'Miss: (\d+) \(([\d.]+)%\)', content)
    if match:
        metrics['miss'] = int(match.group(1))
        metrics['miss_pct'] = float(match.group(2))
    
    # Extract hallucination
    match = re.search(r'Hallucination: (\d+) \(([\d.]+)%\)', content)
    if match:
        metrics['hallucination'] = int(match.group(1))
        metrics['hallucination_pct'] = float(match.group(2))
    
    # Extract factuality rate
    match = re.search(r'Factuality Rate: ([\d.]+)%', content)
    metrics['factuality_rate'] = float(match.group(1)) if match else 0.0
    
    return metrics


def load_all_results():
    """Load results from all model directories."""
    results = []
    
    for subdir in sorted(RESULTS_DIR.iterdir()):
        if subdir.is_dir() and (subdir / 'statistics.txt').exists():
            stats_path = subdir / 'statistics.txt'
            metrics = parse_statistics_file(stats_path)
            metrics['directory'] = subdir.name
            
            # Parse model name and mode
            parts = subdir.name.rsplit('_', 1)
            if len(parts) == 2:
                metrics['model_name'] = parts[0]
                metrics['mode'] = parts[1].upper()
            else:
                metrics['model_name'] = subdir.name
                metrics['mode'] = 'Unknown'
            
            results.append(metrics)
    
    return results


def generate_comparison_table(results):
    """Generate a formatted comparison table."""
    # Sort by model name then mode
    results_sorted = sorted(results, key=lambda x: (x['model_name'], x['mode']))
    
    lines = []
    lines.append("=" * 120)
    lines.append("MODEL COMPARISON TABLE")
    lines.append("=" * 120)
    lines.append("")
    
    # Header
    header = f"{'Model':<45} {'Mode':<8} {'Correct':>10} {'Miss':>10} {'Halluc.':>10} {'Timeout':>8} {'Factuality':>12}"
    lines.append(header)
    lines.append("-" * 120)
    
    for r in results_sorted:
        model_short = r['model_name'].replace('deepseek-', '').replace('openai-', '').replace('mistralai-', '')
        row = f"{model_short:<45} {r['mode']:<8} {r['correct_pct']:>9.2f}% {r['miss_pct']:>9.2f}% {r['hallucination_pct']:>9.2f}% {r['timeout']:>8} {r['factuality_rate']:>11.2f}%"
        lines.append(row)
    
    lines.append("-" * 120)
    lines.append("")
    
    return "\n".join(lines)


def generate_model_analysis(results):
    """Generate detailed analysis for each model."""
    lines = []
    lines.append("=" * 120)
    lines.append("DETAILED MODEL ANALYSIS")
    lines.append("=" * 120)
    lines.append("")
    
    # Group by base model
    models = defaultdict(list)
    for r in results:
        base_model = r['model_name'].split('-')[0]  # Get provider name
        models[base_model].append(r)
    
    for provider, model_results in sorted(models.items()):
        lines.append(f"## {provider.upper()} Models")
        lines.append("-" * 60)
        
        for r in sorted(model_results, key=lambda x: x['mode']):
            lines.append(f"\n### {r['model_name']} ({r['mode']})")
            lines.append(f"   - Correct Answers: {r['correct']}/{r['evaluated']} ({r['correct_pct']:.2f}%)")
            lines.append(f"   - Missed Answers:  {r['miss']}/{r['evaluated']} ({r['miss_pct']:.2f}%)")
            lines.append(f"   - Hallucinations:  {r['hallucination']}/{r['evaluated']} ({r['hallucination_pct']:.2f}%)")
            lines.append(f"   - Timeouts:        {r['timeout']}/{r['total_queries']} ({r['timeout']/r['total_queries']*100:.2f}%)")
            lines.append(f"   - Factuality Rate: {r['factuality_rate']:.2f}%")
        
        lines.append("")
    
    return "\n".join(lines)


def generate_rag_vs_direct_analysis(results):
    """Compare RAG vs Direct modes for each model."""
    lines = []
    lines.append("=" * 120)
    lines.append("RAG vs DIRECT MODE COMPARISON")
    lines.append("=" * 120)
    lines.append("")
    
    # Group by model name
    models = defaultdict(dict)
    for r in results:
        models[r['model_name']][r['mode']] = r
    
    for model_name, modes in sorted(models.items()):
        if 'RAG' in modes and 'DIRECT' in modes:
            rag = modes['RAG']
            direct = modes['DIRECT']
            
            lines.append(f"## {model_name}")
            lines.append("-" * 80)
            lines.append("")
            lines.append(f"{'Metric':<25} {'RAG':>15} {'DIRECT':>15} {'Difference':>15}")
            lines.append("-" * 80)
            
            # Correct rate
            diff = rag['correct_pct'] - direct['correct_pct']
            sign = "+" if diff > 0 else ""
            lines.append(f"{'Correct Rate':<25} {rag['correct_pct']:>14.2f}% {direct['correct_pct']:>14.2f}% {sign}{diff:>14.2f}%")
            
            # Miss rate
            diff = rag['miss_pct'] - direct['miss_pct']
            sign = "+" if diff > 0 else ""
            lines.append(f"{'Miss Rate':<25} {rag['miss_pct']:>14.2f}% {direct['miss_pct']:>14.2f}% {sign}{diff:>14.2f}%")
            
            # Hallucination rate
            diff = rag['hallucination_pct'] - direct['hallucination_pct']
            sign = "+" if diff > 0 else ""
            lines.append(f"{'Hallucination Rate':<25} {rag['hallucination_pct']:>14.2f}% {direct['hallucination_pct']:>14.2f}% {sign}{diff:>14.2f}%")
            
            # Factuality
            diff = rag['factuality_rate'] - direct['factuality_rate']
            sign = "+" if diff > 0 else ""
            lines.append(f"{'Factuality Rate':<25} {rag['factuality_rate']:>14.2f}% {direct['factuality_rate']:>14.2f}% {sign}{diff:>14.2f}%")
            
            lines.append("")
            
            # Analysis
            if rag['correct_pct'] > direct['correct_pct']:
                lines.append(f"   → RAG improves correct answer rate by {rag['correct_pct'] - direct['correct_pct']:.2f}%")
            else:
                lines.append(f"   → DIRECT has higher correct answer rate by {direct['correct_pct'] - rag['correct_pct']:.2f}%")
            
            if rag['hallucination_pct'] > direct['hallucination_pct']:
                lines.append(f"   → RAG increases hallucination rate by {rag['hallucination_pct'] - direct['hallucination_pct']:.2f}%")
            else:
                lines.append(f"   → RAG reduces hallucination rate by {direct['hallucination_pct'] - rag['hallucination_pct']:.2f}%")
            
            if rag['factuality_rate'] > direct['factuality_rate']:
                lines.append(f"   → RAG improves factuality by {rag['factuality_rate'] - direct['factuality_rate']:.2f}%")
            else:
                lines.append(f"   → DIRECT has better factuality by {direct['factuality_rate'] - rag['factuality_rate']:.2f}%")
            
            lines.append("")
    
    return "\n".join(lines)


def generate_ranking_analysis(results):
    """Generate rankings for different metrics."""
    lines = []
    lines.append("=" * 120)
    lines.append("MODEL RANKINGS")
    lines.append("=" * 120)
    lines.append("")
    
    # Ranking by Correct Rate
    lines.append("### Ranking by Correct Answer Rate (Higher is Better)")
    lines.append("-" * 60)
    sorted_by_correct = sorted(results, key=lambda x: x['correct_pct'], reverse=True)
    for i, r in enumerate(sorted_by_correct, 1):
        lines.append(f"  {i}. {r['model_name']} ({r['mode']}): {r['correct_pct']:.2f}%")
    lines.append("")
    
    # Ranking by Factuality Rate
    lines.append("### Ranking by Factuality Rate (Higher is Better)")
    lines.append("-" * 60)
    sorted_by_factuality = sorted(results, key=lambda x: x['factuality_rate'], reverse=True)
    for i, r in enumerate(sorted_by_factuality, 1):
        lines.append(f"  {i}. {r['model_name']} ({r['mode']}): {r['factuality_rate']:.2f}%")
    lines.append("")
    
    # Ranking by Hallucination Rate
    lines.append("### Ranking by Hallucination Rate (Lower is Better)")
    lines.append("-" * 60)
    sorted_by_hallucination = sorted(results, key=lambda x: x['hallucination_pct'])
    for i, r in enumerate(sorted_by_hallucination, 1):
        lines.append(f"  {i}. {r['model_name']} ({r['mode']}): {r['hallucination_pct']:.2f}%")
    lines.append("")
    
    return "\n".join(lines)


def generate_summary_statistics(results):
    """Generate overall summary statistics."""
    lines = []
    lines.append("=" * 120)
    lines.append("SUMMARY STATISTICS")
    lines.append("=" * 120)
    lines.append("")
    
    # Aggregate stats
    total_queries = sum(r['total_queries'] for r in results)
    total_correct = sum(r['correct'] for r in results)
    total_miss = sum(r['miss'] for r in results)
    total_hallucination = sum(r['hallucination'] for r in results)
    total_timeout = sum(r['timeout'] for r in results)
    
    avg_correct = sum(r['correct_pct'] for r in results) / len(results)
    avg_miss = sum(r['miss_pct'] for r in results) / len(results)
    avg_hallucination = sum(r['hallucination_pct'] for r in results) / len(results)
    avg_factuality = sum(r['factuality_rate'] for r in results) / len(results)
    
    lines.append(f"Total Models Evaluated: {len(results)}")
    lines.append(f"Total Queries Processed: {total_queries}")
    lines.append("")
    lines.append("### Aggregate Results")
    lines.append(f"   - Total Correct:       {total_correct}")
    lines.append(f"   - Total Miss:          {total_miss}")
    lines.append(f"   - Total Hallucination: {total_hallucination}")
    lines.append(f"   - Total Timeout:       {total_timeout}")
    lines.append("")
    lines.append("### Average Rates Across All Models")
    lines.append(f"   - Average Correct Rate:       {avg_correct:.2f}%")
    lines.append(f"   - Average Miss Rate:          {avg_miss:.2f}%")
    lines.append(f"   - Average Hallucination Rate: {avg_hallucination:.2f}%")
    lines.append(f"   - Average Factuality Rate:    {avg_factuality:.2f}%")
    lines.append("")
    
    # Best performers
    best_correct = max(results, key=lambda x: x['correct_pct'])
    best_factuality = max(results, key=lambda x: x['factuality_rate'])
    lowest_hallucination = min(results, key=lambda x: x['hallucination_pct'])
    
    lines.append("### Best Performers")
    lines.append(f"   - Highest Correct Rate:      {best_correct['model_name']} ({best_correct['mode']}) at {best_correct['correct_pct']:.2f}%")
    lines.append(f"   - Highest Factuality:        {best_factuality['model_name']} ({best_factuality['mode']}) at {best_factuality['factuality_rate']:.2f}%")
    lines.append(f"   - Lowest Hallucination:      {lowest_hallucination['model_name']} ({lowest_hallucination['mode']}) at {lowest_hallucination['hallucination_pct']:.2f}%")
    lines.append("")
    
    return "\n".join(lines)


def generate_key_insights(results):
    """Generate key insights and observations."""
    lines = []
    lines.append("=" * 120)
    lines.append("KEY INSIGHTS & OBSERVATIONS")
    lines.append("=" * 120)
    lines.append("")
    
    # Separate RAG and DIRECT results
    rag_results = [r for r in results if r['mode'] == 'RAG']
    direct_results = [r for r in results if r['mode'] == 'DIRECT']
    
    if rag_results and direct_results:
        avg_rag_correct = sum(r['correct_pct'] for r in rag_results) / len(rag_results)
        avg_direct_correct = sum(r['correct_pct'] for r in direct_results) / len(direct_results)
        
        avg_rag_halluc = sum(r['hallucination_pct'] for r in rag_results) / len(rag_results)
        avg_direct_halluc = sum(r['hallucination_pct'] for r in direct_results) / len(direct_results)
        
        avg_rag_factuality = sum(r['factuality_rate'] for r in rag_results) / len(rag_results)
        avg_direct_factuality = sum(r['factuality_rate'] for r in direct_results) / len(direct_results)
        
        lines.append("### 1. RAG vs Direct Mode Overall Impact")
        lines.append("")
        lines.append(f"   Average Correct Rate:       RAG={avg_rag_correct:.2f}%  vs  DIRECT={avg_direct_correct:.2f}%")
        lines.append(f"   Average Hallucination Rate: RAG={avg_rag_halluc:.2f}%  vs  DIRECT={avg_direct_halluc:.2f}%")
        lines.append(f"   Average Factuality Rate:    RAG={avg_rag_factuality:.2f}%  vs  DIRECT={avg_direct_factuality:.2f}%")
        lines.append("")
        
        if avg_rag_correct > avg_direct_correct:
            lines.append(f"   → RAG mode generally IMPROVES correct answer rate (+{avg_rag_correct - avg_direct_correct:.2f}%)")
        else:
            lines.append(f"   → DIRECT mode generally has BETTER correct answer rate (+{avg_direct_correct - avg_rag_correct:.2f}%)")
        
        if avg_rag_halluc > avg_direct_halluc:
            lines.append(f"   → RAG mode INCREASES hallucination rate (+{avg_rag_halluc - avg_direct_halluc:.2f}%)")
        else:
            lines.append(f"   → RAG mode DECREASES hallucination rate (-{avg_direct_halluc - avg_rag_halluc:.2f}%)")
        
        if avg_rag_factuality > avg_direct_factuality:
            lines.append(f"   → RAG mode generally IMPROVES factuality (+{avg_rag_factuality - avg_direct_factuality:.2f}%)")
        else:
            lines.append(f"   → DIRECT mode generally has BETTER factuality (+{avg_direct_factuality - avg_rag_factuality:.2f}%)")
        
        lines.append("")
    
    # Model-specific insights
    lines.append("### 2. Model-Specific Observations")
    lines.append("")
    
    for r in sorted(results, key=lambda x: (x['model_name'], x['mode'])):
        provider = r['model_name'].split('-')[0]
        
        # High hallucination warning
        if r['hallucination_pct'] > 25:
            lines.append(f"   ⚠️  {r['model_name']} ({r['mode']}): High hallucination rate ({r['hallucination_pct']:.2f}%)")
        
        # Low factuality warning
        if r['factuality_rate'] < 50:
            lines.append(f"   ⚠️  {r['model_name']} ({r['mode']}): Low factuality rate ({r['factuality_rate']:.2f}%)")
        
        # Good performers
        if r['correct_pct'] > 20 and r['factuality_rate'] > 50:
            lines.append(f"   ✓  {r['model_name']} ({r['mode']}): Good balance (Correct: {r['correct_pct']:.2f}%, Factuality: {r['factuality_rate']:.2f}%)")
    
    lines.append("")
    
    # Trade-off analysis
    lines.append("### 3. Accuracy vs Factuality Trade-off")
    lines.append("")
    lines.append("   The results reveal a trade-off between providing correct answers and avoiding hallucinations:")
    lines.append("")
    
    # Find models that demonstrate this trade-off
    for r in results:
        if r['correct_pct'] > 15 and r['hallucination_pct'] > 20:
            lines.append(f"   - {r['model_name']} ({r['mode']}): Higher correct rate ({r['correct_pct']:.2f}%) but also higher hallucination ({r['hallucination_pct']:.2f}%)")
        elif r['correct_pct'] < 15 and r['hallucination_pct'] < 15:
            lines.append(f"   - {r['model_name']} ({r['mode']}): Conservative approach - lower correct ({r['correct_pct']:.2f}%) but lower hallucination ({r['hallucination_pct']:.2f}%)")
    
    lines.append("")
    
    return "\n".join(lines)


def export_to_csv(results, output_path):
    """Export results to CSV format."""
    import csv
    
    fieldnames = ['model_name', 'mode', 'total_queries', 'evaluated', 'correct', 'correct_pct', 
                  'miss', 'miss_pct', 'hallucination', 'hallucination_pct', 'timeout', 'factuality_rate']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda x: (x['model_name'], x['mode'])):
            row = {k: r.get(k, '') for k in fieldnames}
            writer.writerow(row)
    
    print(f"CSV exported to: {output_path}")


def try_generate_charts(results, output_dir):
    """Try to generate visualization charts if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Prepare data
        models = sorted(set(r['model_name'] for r in results))
        model_short_names = []
        for m in models:
            short = m.replace('deepseek-', '').replace('openai-', '').replace('mistralai-', '')
            model_short_names.append(short)
        
        # Figure 1: Correct Rate Comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Subplot 1: Correct Rate by Model and Mode
        ax = axes[0, 0]
        x = np.arange(len(models))
        width = 0.35
        
        rag_correct = []
        direct_correct = []
        for m in models:
            rag = next((r for r in results if r['model_name'] == m and r['mode'] == 'RAG'), None)
            direct = next((r for r in results if r['model_name'] == m and r['mode'] == 'DIRECT'), None)
            rag_correct.append(rag['correct_pct'] if rag else 0)
            direct_correct.append(direct['correct_pct'] if direct else 0)
        
        bars1 = ax.bar(x - width/2, rag_correct, width, label='RAG', color='#2ecc71')
        bars2 = ax.bar(x + width/2, direct_correct, width, label='DIRECT', color='#3498db')
        ax.set_ylabel('Correct Rate (%)')
        ax.set_title('Correct Answer Rate by Model')
        ax.set_xticks(x)
        ax.set_xticklabels(model_short_names, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Subplot 2: Hallucination Rate
        ax = axes[0, 1]
        rag_halluc = []
        direct_halluc = []
        for m in models:
            rag = next((r for r in results if r['model_name'] == m and r['mode'] == 'RAG'), None)
            direct = next((r for r in results if r['model_name'] == m and r['mode'] == 'DIRECT'), None)
            rag_halluc.append(rag['hallucination_pct'] if rag else 0)
            direct_halluc.append(direct['hallucination_pct'] if direct else 0)
        
        bars1 = ax.bar(x - width/2, rag_halluc, width, label='RAG', color='#e74c3c')
        bars2 = ax.bar(x + width/2, direct_halluc, width, label='DIRECT', color='#f39c12')
        ax.set_ylabel('Hallucination Rate (%)')
        ax.set_title('Hallucination Rate by Model')
        ax.set_xticks(x)
        ax.set_xticklabels(model_short_names, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Subplot 3: Factuality Rate
        ax = axes[1, 0]
        rag_fact = []
        direct_fact = []
        for m in models:
            rag = next((r for r in results if r['model_name'] == m and r['mode'] == 'RAG'), None)
            direct = next((r for r in results if r['model_name'] == m and r['mode'] == 'DIRECT'), None)
            rag_fact.append(rag['factuality_rate'] if rag else 0)
            direct_fact.append(direct['factuality_rate'] if direct else 0)
        
        bars1 = ax.bar(x - width/2, rag_fact, width, label='RAG', color='#9b59b6')
        bars2 = ax.bar(x + width/2, direct_fact, width, label='DIRECT', color='#1abc9c')
        ax.set_ylabel('Factuality Rate (%)')
        ax.set_title('Factuality Rate by Model')
        ax.set_xticks(x)
        ax.set_xticklabels(model_short_names, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Subplot 4: Overall Distribution (Stacked Bar)
        ax = axes[1, 1]
        labels = [f"{r['model_name'].split('-')[-1][:8]}\n({r['mode'][:3]})" for r in sorted(results, key=lambda x: (x['model_name'], x['mode']))]
        correct_vals = [r['correct_pct'] for r in sorted(results, key=lambda x: (x['model_name'], x['mode']))]
        miss_vals = [r['miss_pct'] for r in sorted(results, key=lambda x: (x['model_name'], x['mode']))]
        halluc_vals = [r['hallucination_pct'] for r in sorted(results, key=lambda x: (x['model_name'], x['mode']))]
        
        x = np.arange(len(labels))
        ax.bar(x, correct_vals, label='Correct', color='#2ecc71')
        ax.bar(x, miss_vals, bottom=correct_vals, label='Miss', color='#95a5a6')
        ax.bar(x, halluc_vals, bottom=np.array(correct_vals)+np.array(miss_vals), label='Hallucination', color='#e74c3c')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Result Distribution by Model')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        chart_path = output_dir / 'analysis_charts.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Charts saved to: {chart_path}")
        return True
        
    except ImportError:
        print("matplotlib not available - skipping chart generation")
        return False


def main():
    """Main analysis function."""
    print("=" * 80)
    print("AIPolicyBench Results Analysis")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # Load all results
    results = load_all_results()
    print(f"Found {len(results)} model configurations to analyze")
    print()
    
    # Generate report sections
    report_sections = []
    
    # Header
    header = f"""
################################################################################
#                     AIPolicyBench Results Analysis Report                     #
#                                                                              #
#                  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                           #
################################################################################

"""
    report_sections.append(header)
    
    # Add all sections
    report_sections.append(generate_summary_statistics(results))
    report_sections.append(generate_comparison_table(results))
    report_sections.append(generate_ranking_analysis(results))
    report_sections.append(generate_rag_vs_direct_analysis(results))
    report_sections.append(generate_model_analysis(results))
    report_sections.append(generate_key_insights(results))
    
    # Combine report
    full_report = "\n".join(report_sections)
    
    # Save report
    report_path = RESULTS_DIR / 'analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write(full_report)
    print(f"Report saved to: {report_path}")
    
    # Export CSV
    csv_path = RESULTS_DIR / 'analysis_data.csv'
    export_to_csv(results, csv_path)
    
    # Try to generate charts
    try_generate_charts(results, RESULTS_DIR)
    
    # Print report to console
    print()
    print(full_report)
    
    return results


if __name__ == "__main__":
    main()

