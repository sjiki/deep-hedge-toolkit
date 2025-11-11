"""Generate performance reports for VIX ETP agents."""

import json
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np


def generate_report(
    results_dir: str = 'outputs',
    output_file: str = 'vix_etp_report.png'
):
    """
    Generate visual report from training/evaluation results.
    
    Args:
        results_dir: Directory containing results JSON files
        output_file: Output file for report plots
    """
    results_path = Path(results_dir)
    
    # Load results
    walkforward_file = results_path / 'rolling_walkforward.json'
    
    if not walkforward_file.exists():
        print(f"Warning: {walkforward_file} not found. Skipping walkforward plots.")
        return
    
    with open(walkforward_file, 'r') as f:
        wf_results = json.load(f)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('VIX ETP Agent Performance Report', fontsize=16, fontweight='bold')
    
    fold_results = wf_results.get('fold_results', [])
    
    if not fold_results:
        print("No fold results found")
        return
    
    # Plot 1: Sharpe Ratio by Fold
    ax = axes[0, 0]
    folds = [f['fold'] for f in fold_results]
    sharpes = [f['sharpe_ratio'] for f in fold_results]
    
    ax.bar(folds, sharpes, color='steelblue', alpha=0.7)
    ax.axhline(y=np.mean(sharpes), color='red', linestyle='--', label=f'Mean: {np.mean(sharpes):.2f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio by Fold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Returns Distribution
    ax = axes[0, 1]
    returns = [f['total_return'] for f in fold_results]
    
    ax.hist(returns, bins=15, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.3f}')
    ax.set_xlabel('Total Return')
    ax.set_ylabel('Frequency')
    ax.set_title('Returns Distribution Across Folds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Risk-Return Scatter
    ax = axes[1, 0]
    vols = [f['volatility'] for f in fold_results]
    
    ax.scatter(vols, returns, s=100, alpha=0.6, c=sharpes, cmap='RdYlGn', edgecolors='black')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Total Return')
    ax.set_title('Risk-Return Profile')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=min(sharpes), vmax=max(sharpes)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Sharpe Ratio')
    
    # Plot 4: Metrics Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Overall metrics
    overall = wf_results.get('overall_metrics', {})
    avg_metrics = wf_results.get('average_metrics', {})
    
    summary_text = "Overall Performance:\n\n"
    
    metrics_to_show = [
        ('Sharpe Ratio', 'sharpe_ratio'),
        ('Sortino Ratio', 'sortino_ratio'),
        ('Calmar Ratio', 'calmar_ratio'),
        ('Max Drawdown', 'max_drawdown'),
        ('Hit Ratio', 'hit_ratio'),
        ('ES95', 'expected_shortfall_95')
    ]
    
    for label, key in metrics_to_show:
        if key in overall:
            value = overall[key]
            avg_value = avg_metrics.get(key, value)
            summary_text += f"{label}:\n"
            summary_text += f"  Overall: {value:.4f}\n"
            summary_text += f"  Avg Fold: {avg_value:.4f}\n\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save figure
    output_path = results_path / output_file
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Report saved to {output_path}")
    
    plt.close()


def print_summary(results_dir: str = 'outputs'):
    """
    Print text summary of results.
    
    Args:
        results_dir: Directory containing results JSON files
    """
    results_path = Path(results_dir)
    
    print("="*60)
    print("VIX ETP AGENT PERFORMANCE SUMMARY")
    print("="*60)
    
    # Walk-forward results
    wf_file = results_path / 'rolling_walkforward.json'
    if wf_file.exists():
        with open(wf_file, 'r') as f:
            wf_results = json.load(f)
        
        print("\nRolling Walk-Forward Validation:")
        print(f"  Number of folds: {wf_results['n_folds']}")
        
        overall = wf_results.get('overall_metrics', {})
        if overall:
            print("\n  Overall Metrics:")
            print(f"    Sharpe Ratio:     {overall.get('sharpe_ratio', 0):.4f}")
            print(f"    Sortino Ratio:    {overall.get('sortino_ratio', 0):.4f}")
            print(f"    Calmar Ratio:     {overall.get('calmar_ratio', 0):.4f}")
            print(f"    Max Drawdown:     {overall.get('max_drawdown', 0):.4f}")
            print(f"    Total Return:     {overall.get('total_return', 0):.4f}")
            print(f"    Hit Ratio:        {overall.get('hit_ratio', 0):.4f}")
    
    # Training metrics
    for agent_type in ['ppo', 'sac']:
        metrics_file = results_path / f'training_metrics_{agent_type}.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            print(f"\n{agent_type.upper()} Training:")
            final = metrics.get('final_metrics', {})
            print(f"  Total timesteps:   {metrics.get('total_timesteps', 0)}")
            print(f"  Final avg return:  {final.get('avg_return', 0):.4f}")
            print(f"  Final avg Sharpe:  {final.get('avg_sharpe', 0):.4f}")
    
    # Optuna results
    optuna_file = results_path / 'optuna_best_params.json'
    if optuna_file.exists():
        with open(optuna_file, 'r') as f:
            optuna_results = json.load(f)
        
        print("\nOptuna Hyperparameter Tuning:")
        print(f"  Best Sharpe:       {optuna_results.get('best_value', 0):.4f}")
        print(f"  Total trials:      {optuna_results.get('n_trials', 0)}")
        print(f"\n  Best Parameters:")
        for param, value in optuna_results.get('best_params', {}).items():
            print(f"    {param}: {value}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate VIX ETP agent report')
    parser.add_argument('--results-dir', type=str, default='outputs',
                        help='Directory containing results')
    parser.add_argument('--output', type=str, default='vix_etp_report.png',
                        help='Output file for plots')
    parser.add_argument('--text-only', action='store_true',
                        help='Print text summary only (no plots)')
    
    args = parser.parse_args()
    
    if args.text_only:
        print_summary(args.results_dir)
    else:
        generate_report(args.results_dir, args.output)
        print_summary(args.results_dir)
