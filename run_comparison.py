"""Optimized Comparison Script for Probabilistic FIN experiments."""

import subprocess
import json
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def run_experiment(exp_name, use_probabilistic=True, use_uq_weighting=True,
                   lr=1e-4, local_epochs=3, uq_alpha=0.5, uq_temperature=0.1,
                   pretrain_fin_epochs=10, multimodal_boost=3.0):
    """Run a single experiment with specified parameters."""
    import shutil

    cmd = [
        'python', 'train_federated.py',
        '--num_rounds', '20',
        '--local_epochs', str(local_epochs),
        '--batch_size', '32',
        '--lr', str(lr),
        '--eval_every', '1',
        '--pretrain_fin_epochs', str(pretrain_fin_epochs),
        '--uq_temperature', str(uq_temperature),
        '--multimodal_boost', str(multimodal_boost),
        '--save_dir', f'checkpoints_{exp_name}'
    ]

    if use_probabilistic:
        cmd.append('--use_probabilistic')
    if use_uq_weighting:
        cmd.append('--use_uq_weighting')
        cmd.extend(['--uq_alpha', str(uq_alpha)])

    print(f"\n{'='*60}")
    print(f"Running experiment: {exp_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Experiment {exp_name} failed!")
        return None

    os.makedirs('paper_results', exist_ok=True)
    src_csv = os.path.join(f'checkpoints_{exp_name}', 'training_results.csv')
    dst_csv = os.path.join('paper_results', f'{exp_name}.csv')
    if os.path.exists(src_csv):
        shutil.copy(src_csv, dst_csv)
        print(f"Copied results to {dst_csv}")

    results_path = os.path.join(f'checkpoints_{exp_name}', 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)

    return None


def save_results_to_csv(all_results: dict, output_dir: str = "paper_results"):
    """Save all results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    method_names = {
        'probabilistic_fin_uq_optimized': 'PCFI_Ours',
        'deterministic_fin_standard': 'FIN_Baseline',
        'probabilistic_fin_standard': 'PFIN_No_UQ',
        'probabilistic_fin_moderate_uq': 'PFIN_Moderate_UQ'
    }

    for exp_name, results in all_results.items():
        if results is None:
            continue

        csv_path = os.path.join(output_dir, f"{exp_name}_results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'test_auc', 'test_f1'])
            for r in results:
                writer.writerow([r['round'], r['auc'], r['f1']])
        print(f"Saved: {csv_path}")

    combined_auc_path = os.path.join(output_dir, "combined_auc_comparison.csv")
    with open(combined_auc_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['round']
        for exp_name in all_results.keys():
            if all_results[exp_name] is not None:
                header.append(method_names.get(exp_name, exp_name))
        writer.writerow(header)

        max_rounds = max(len(r) for r in all_results.values() if r is not None)
        for i in range(max_rounds):
            row = [i + 1]
            for exp_name, results in all_results.items():
                if results is not None and i < len(results):
                    row.append(f"{results[i]['auc']:.4f}")
                else:
                    row.append('')
            writer.writerow(row)
    print(f"Saved: {combined_auc_path}")

    combined_f1_path = os.path.join(output_dir, "combined_f1_comparison.csv")
    with open(combined_f1_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['round']
        for exp_name in all_results.keys():
            if all_results[exp_name] is not None:
                header.append(method_names.get(exp_name, exp_name))
        writer.writerow(header)

        max_rounds = max(len(r) for r in all_results.values() if r is not None)
        for i in range(max_rounds):
            row = [i + 1]
            for exp_name, results in all_results.items():
                if results is not None and i < len(results):
                    row.append(f"{results[i]['f1']:.4f}")
                else:
                    row.append('')
            writer.writerow(row)
    print(f"Saved: {combined_f1_path}")

    summary_path = os.path.join(output_dir, "final_summary.csv")
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'final_auc', 'best_auc', 'final_f1', 'best_f1', 'improvement_over_baseline_auc'])

        baseline_auc = 0.0
        if all_results.get('deterministic_fin_standard'):
            baseline_auc = all_results['deterministic_fin_standard'][-1]['auc']

        for exp_name, results in all_results.items():
            if results is not None and len(results) > 0:
                final_auc = results[-1]['auc']
                best_auc = max(r['auc'] for r in results)
                final_f1 = results[-1]['f1']
                best_f1 = max(r['f1'] for r in results)
                improvement = ((final_auc - baseline_auc) / baseline_auc * 100) if baseline_auc > 0 else 0

                writer.writerow([
                    method_names.get(exp_name, exp_name),
                    f"{final_auc:.4f}", f"{best_auc:.4f}",
                    f"{final_f1:.4f}", f"{best_f1:.4f}",
                    f"{improvement:+.2f}%"
                ])
    print(f"Saved: {summary_path}")

    return output_dir


def plot_comparison(results_dict, output_path="paper_results/comparison_plot.png"):
    """Plot comparison of different methods."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        'probabilistic_fin_uq_optimized': '#2ecc71',
        'deterministic_fin_standard': '#e74c3c',
        'probabilistic_fin_standard': '#3498db',
        'probabilistic_fin_moderate_uq': '#f39c12'
    }

    labels = {
        'probabilistic_fin_uq_optimized': 'PCFI (Ours)',
        'deterministic_fin_standard': 'FIN (Baseline)',
        'probabilistic_fin_standard': 'Prob FIN + FedAvg',
        'probabilistic_fin_moderate_uq': 'Prob FIN + Moderate UQ'
    }

    linewidths = {
        'probabilistic_fin_uq_optimized': 2.5,
        'deterministic_fin_standard': 2.0,
        'probabilistic_fin_standard': 1.5,
        'probabilistic_fin_moderate_uq': 1.5
    }

    for exp_name, results in results_dict.items():
        if results is None:
            continue

        rounds = [r['round'] for r in results]
        aucs = [r['auc'] for r in results]
        f1s = [r['f1'] for r in results]

        ax1.plot(rounds, aucs, marker='o', markersize=4,
                color=colors.get(exp_name, 'gray'),
                linewidth=linewidths.get(exp_name, 1.5),
                label=labels.get(exp_name, exp_name))
        ax2.plot(rounds, f1s, marker='s', markersize=4,
                color=colors.get(exp_name, 'gray'),
                linewidth=linewidths.get(exp_name, 1.5),
                label=labels.get(exp_name, exp_name))

    ax1.set_xlabel('Federated Round', fontsize=12)
    ax1.set_ylabel('Test AUC', fontsize=12)
    ax1.set_title('Test AUC over Federated Rounds', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.3, 0.9])

    ax2.set_xlabel('Federated Round', fontsize=12)
    ax2.set_ylabel('Test F1', fontsize=12)
    ax2.set_title('Test F1 over Federated Rounds', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to {output_path}")


def print_final_summary(all_results: dict):
    """Print final summary of all experiments."""
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)

    labels = {
        'probabilistic_fin_uq_optimized': 'PCFI (Ours)',
        'deterministic_fin_standard': 'FIN (Baseline)',
        'probabilistic_fin_standard': 'Prob FIN + FedAvg',
        'probabilistic_fin_moderate_uq': 'Prob FIN + Moderate UQ'
    }

    baseline_final_auc = 0.0
    if all_results.get('deterministic_fin_standard'):
        baseline_results = all_results['deterministic_fin_standard']
        if baseline_results:
            baseline_final_auc = baseline_results[-1]['auc']

    print(f"\n{'Method':<25} {'Final AUC':<12} {'Best AUC':<12} {'Improvement':<12}")
    print("-" * 65)

    for exp_name, results in all_results.items():
        if results is not None and len(results) > 0:
            final_auc = results[-1]['auc']
            best_auc = max(r['auc'] for r in results)

            if baseline_final_auc > 0:
                improvement = (final_auc - baseline_final_auc) / baseline_final_auc * 100
                imp_str = f"{improvement:+.2f}%"
            else:
                imp_str = "N/A"

            print(f"{labels.get(exp_name, exp_name):<25} {final_auc:<12.4f} {best_auc:<12.4f} {imp_str:<12}")

    print("="*70)


def main():
    """Run all optimized experiments."""
    print("\n" + "="*70)
    print("FAIR COMPARISON: PCFI vs FIN BASELINE")
    print("Started at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("="*70)

    BASE_CONFIG = {
        'lr': 1e-4,
        'local_epochs': 4,
        'pretrain_fin_epochs': 20,
        'multimodal_boost': 3.0,
    }

    experiments = {
        'probabilistic_fin_uq_optimized': {
            **BASE_CONFIG,
            'use_probabilistic': True,
            'use_uq_weighting': True,
            'uq_alpha': 0.6,
            'uq_temperature': 0.2,
            'description': 'PCFI (Ours) - Probabilistic FIN + UQ-weighted'
        },
        'deterministic_fin_standard': {
            **BASE_CONFIG,
            'use_probabilistic': False,
            'use_uq_weighting': False,
            'uq_alpha': 0.5,
            'uq_temperature': 0.1,
            'description': 'FIN (Baseline) - Deterministic + Standard FedAvg'
        },
        'probabilistic_fin_standard': {
            **BASE_CONFIG,
            'use_probabilistic': True,
            'use_uq_weighting': False,
            'uq_alpha': 0.5,
            'uq_temperature': 0.1,
            'description': 'Prob FIN + Standard FedAvg (Ablation)'
        },
        'deterministic_fin_uq': {
            **BASE_CONFIG,
            'use_probabilistic': False,
            'use_uq_weighting': True,
            'uq_alpha': 0.6,
            'uq_temperature': 0.2,
            'description': 'FIN + UQ-Weighted (Ablation)'
        }
    }

    all_results = {}

    for exp_name, config in experiments.items():
        print(f"\n>>> {config['description']}")

        results = run_experiment(
            exp_name,
            use_probabilistic=config['use_probabilistic'],
            use_uq_weighting=config['use_uq_weighting'],
            lr=config['lr'],
            local_epochs=config['local_epochs'],
            uq_alpha=config['uq_alpha'],
            uq_temperature=config['uq_temperature'],
            pretrain_fin_epochs=config['pretrain_fin_epochs'],
            multimodal_boost=config['multimodal_boost']
        )

        all_results[exp_name] = results

    output_dir = save_results_to_csv(all_results)
    plot_comparison(all_results, os.path.join(output_dir, "comparison_plot.png"))
    print_final_summary(all_results)

    with open(os.path.join(output_dir, "all_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to: {output_dir}/")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
