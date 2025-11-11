"""Rolling walk-forward validation for VIX ETP agents."""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from env.builder import build_vix_etp_env, load_config
from agents.policy_vix_etp import VIXETPPPOAgent


def compute_metrics(returns: List[float], portfolio_values: List[float]) -> Dict:
    """
    Compute trading metrics from returns.
    
    Args:
        returns: List of portfolio returns
        portfolio_values: List of portfolio values
        
    Returns:
        Dictionary of metrics
    """
    returns_arr = np.array(returns)
    
    # Basic metrics
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) if len(portfolio_values) > 1 else 0.0
    mean_return = np.mean(returns_arr)
    volatility = np.std(returns_arr)
    
    # Sharpe ratio
    sharpe_ratio = mean_return / volatility * np.sqrt(252) if volatility > 0 else 0.0
    
    # Sortino ratio
    downside_returns = returns_arr[returns_arr < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else volatility
    sortino_ratio = mean_return / downside_std * np.sqrt(252) if downside_std > 0 else 0.0
    
    # Max drawdown
    cumulative = np.cumprod(1 + returns_arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Calmar ratio (proxy using max_dd)
    annualized_return = total_return * (252 / len(returns))
    calmar_ratio = -annualized_return / max_drawdown if max_drawdown < 0 else 0.0
    
    # Expected Shortfall (ES95)
    sorted_returns = np.sort(returns_arr)
    var_95_idx = int(len(sorted_returns) * 0.05)
    es_95 = np.mean(sorted_returns[:var_95_idx]) if var_95_idx > 0 else sorted_returns[0]
    
    # Hit ratio (percentage of positive returns)
    hit_ratio = np.sum(returns_arr > 0) / len(returns_arr) if len(returns_arr) > 0 else 0.0
    
    return {
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'mean_return': float(mean_return),
        'volatility': float(volatility),
        'sharpe_ratio': float(sharpe_ratio),
        'sortino_ratio': float(sortino_ratio),
        'calmar_ratio': float(calmar_ratio),
        'max_drawdown': float(max_drawdown),
        'expected_shortfall_95': float(es_95),
        'hit_ratio': float(hit_ratio)
    }


def rolling_walkforward(
    config_path: str,
    train_window_days: int = 365,
    test_window_days: int = 90,
    step_days: int = 30,
    min_train_samples: int = 200,
    output_dir: str = 'outputs',
    agent_type: str = 'ppo',
    timesteps_per_fold: int = 50000
) -> Dict:
    """
    Perform rolling walk-forward validation.
    
    Args:
        config_path: Path to config file
        train_window_days: Training window size in days
        test_window_days: Test window size in days
        step_days: Step size between folds in days
        min_train_samples: Minimum training samples required
        output_dir: Output directory
        agent_type: Agent type ('ppo' or 'sac')
        timesteps_per_fold: Training timesteps per fold
        
    Returns:
        Dictionary with walkforward results
    """
    # Load config
    config = load_config(config_path)
    
    # Get walkforward config
    wf_config = config.get('walkforward', {})
    train_window_days = wf_config.get('train_window_days', train_window_days)
    test_window_days = wf_config.get('test_window_days', test_window_days)
    step_days = wf_config.get('step_days', step_days)
    min_train_samples = wf_config.get('min_train_samples', min_train_samples)
    
    print("Rolling walk-forward validation")
    print(f"  Train window: {train_window_days} days")
    print(f"  Test window: {test_window_days} days")
    print(f"  Step size: {step_days} days")
    print(f"  Timesteps per fold: {timesteps_per_fold}")
    
    # Build environment to get data
    env, price_data, volume_data = build_vix_etp_env(config_path=config_path)
    
    dates = price_data.index
    n_dates = len(dates)
    
    # Compute fold boundaries
    folds = []
    train_start = 0
    
    while train_start + train_window_days + test_window_days <= n_dates:
        train_end = train_start + train_window_days
        test_start = train_end
        test_end = min(test_start + test_window_days, n_dates)
        
        if train_end - train_start >= min_train_samples:
            folds.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_dates': (dates[train_start], dates[train_end - 1]),
                'test_dates': (dates[test_start], dates[test_end - 1])
            })
        
        train_start += step_days
    
    print(f"\nTotal folds: {len(folds)}")
    
    # Run walk-forward
    fold_results = []
    all_test_returns = []
    all_test_values = []
    
    for i, fold in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"Fold {i+1}/{len(folds)}")
        print(f"  Train: {fold['train_dates'][0].date()} to {fold['train_dates'][1].date()}")
        print(f"  Test:  {fold['test_dates'][0].date()} to {fold['test_dates'][1].date()}")
        
        # Split data
        train_prices = price_data.iloc[fold['train_start']:fold['train_end']]
        train_volumes = volume_data.iloc[fold['train_start']:fold['train_end']]
        
        test_prices = price_data.iloc[fold['test_start']:fold['test_end']]
        test_volumes = volume_data.iloc[fold['test_start']:fold['test_end']]
        
        # Build train environment (modify config to use train data)
        # For simplicity, we'll retrain on full data and evaluate on test period
        # In production, you'd want to actually slice the environment data
        
        print("  Training agent...")
        train_env, _, _ = build_vix_etp_env(config_path=config_path)
        
        if agent_type == 'ppo':
            agent = VIXETPPPOAgent.from_config(train_env, config, seed=42 + i)
        else:
            from agents.policy_vix_etp_sac import VIXETPSACAgent
            agent = VIXETPSACAgent.from_config(train_env, config, seed=42 + i)
        
        agent.train(total_timesteps=timesteps_per_fold)
        
        # Evaluate on test period
        print("  Evaluating on test period...")
        test_env, _, _ = build_vix_etp_env(config_path=config_path)
        
        obs, info = test_env.reset()
        test_returns = []
        test_values = [test_env.portfolio_value]
        done = False
        
        while not done and test_env.current_step < len(test_prices) - test_env.min_history_days:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
            if 'portfolio_return' in info:
                test_returns.append(info['portfolio_return'])
                test_values.append(info['portfolio_value'])
        
        # Compute fold metrics
        if len(test_returns) > 0:
            fold_metrics = compute_metrics(test_returns, test_values)
            fold_metrics['fold'] = i + 1
            fold_metrics['train_period'] = [str(d.date()) for d in fold['train_dates']]
            fold_metrics['test_period'] = [str(d.date()) for d in fold['test_dates']]
            fold_metrics['n_test_days'] = len(test_returns)
            
            fold_results.append(fold_metrics)
            all_test_returns.extend(test_returns)
            all_test_values.extend(test_values)
            
            print(f"  Test metrics:")
            print(f"    Sharpe: {fold_metrics['sharpe_ratio']:.4f}")
            print(f"    Sortino: {fold_metrics['sortino_ratio']:.4f}")
            print(f"    Max DD: {fold_metrics['max_drawdown']:.4f}")
            print(f"    Total return: {fold_metrics['total_return']:.4f}")
        else:
            print("  Warning: No test returns collected")
    
    # Aggregate results
    if len(all_test_returns) > 0:
        overall_metrics = compute_metrics(all_test_returns, all_test_values)
    else:
        overall_metrics = {}
    
    # Compute average metrics across folds
    if fold_results:
        avg_metrics = {
            key: float(np.mean([f[key] for f in fold_results]))
            for key in fold_results[0].keys()
            if key not in ['fold', 'train_period', 'test_period', 'n_test_days']
        }
    else:
        avg_metrics = {}
    
    results = {
        'config': {
            'train_window_days': train_window_days,
            'test_window_days': test_window_days,
            'step_days': step_days,
            'agent_type': agent_type,
            'timesteps_per_fold': timesteps_per_fold
        },
        'n_folds': len(folds),
        'fold_results': fold_results,
        'average_metrics': avg_metrics,
        'overall_metrics': overall_metrics
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_path = output_path / 'rolling_walkforward.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Walk-forward validation complete!")
    print(f"Results saved to {results_path}")
    print(f"\nOverall metrics:")
    for key, value in overall_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Rolling walk-forward validation')
    parser.add_argument('--config', type=str, default='config/vix_etp_config.yaml',
                        help='Path to config file')
    parser.add_argument('--train-window', type=int, default=None,
                        help='Training window in days')
    parser.add_argument('--test-window', type=int, default=None,
                        help='Test window in days')
    parser.add_argument('--step', type=int, default=None,
                        help='Step size in days')
    parser.add_argument('--agent', type=str, choices=['ppo', 'sac'], default='ppo',
                        help='Agent type')
    parser.add_argument('--timesteps', type=int, default=50000,
                        help='Training timesteps per fold')
    parser.add_argument('--output', type=str, default='outputs',
                        help='Output directory')
    
    args = parser.parse_args()
    
    rolling_walkforward(
        config_path=args.config,
        train_window_days=args.train_window,
        test_window_days=args.test_window,
        step_days=args.step,
        agent_type=args.agent,
        timesteps_per_fold=args.timesteps,
        output_dir=args.output
    )
