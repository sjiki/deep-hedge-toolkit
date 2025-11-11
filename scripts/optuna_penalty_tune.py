"""Optuna hyperparameter tuning for VIX ETP agents."""

import os
import json
from pathlib import Path
from typing import Dict, Optional
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np

from env.builder import build_vix_etp_env, load_config
from agents.policy_vix_etp import VIXETPPPOAgent


def objective(
    trial: optuna.Trial,
    config: Dict,
    config_path: str,
    n_timesteps: int = 50000
) -> float:
    """
    Optuna objective function for hyperparameter tuning.
    
    Args:
        trial: Optuna trial object
        config: Base configuration dictionary
        config_path: Path to config file
        n_timesteps: Number of timesteps for training
        
    Returns:
        Mean Sharpe ratio
    """
    # Sample hyperparameters
    optuna_config = config.get('optuna', {})
    tune_params = optuna_config.get('tune_params', {})
    
    # Agent hyperparameters
    learning_rate = trial.suggest_float(
        'learning_rate',
        tune_params.get('learning_rate', [1e-4, 1e-3])[0],
        tune_params.get('learning_rate', [1e-4, 1e-3])[1],
        log=True
    )
    
    clip_range = trial.suggest_float(
        'clip_range',
        tune_params.get('clip_range', [0.1, 0.3])[0],
        tune_params.get('clip_range', [0.1, 0.3])[1]
    )
    
    ent_coef = trial.suggest_float(
        'ent_coef',
        tune_params.get('ent_coef', [0.0, 0.1])[0],
        tune_params.get('ent_coef', [0.0, 0.1])[1]
    )
    
    gamma = trial.suggest_float(
        'gamma',
        tune_params.get('gamma', [0.95, 0.99])[0],
        tune_params.get('gamma', [0.99, 0.999])[1]
    )
    
    # Environment hyperparameters
    max_turnover_daily = trial.suggest_float(
        'max_turnover_daily',
        tune_params.get('max_turnover_daily', [0.1, 0.3])[0],
        tune_params.get('max_turnover_daily', [0.1, 0.3])[1]
    )
    
    spike_cooldown_days = trial.suggest_int(
        'spike_cooldown_days',
        int(tune_params.get('spike_cooldown_days', [3, 10])[0]),
        int(tune_params.get('spike_cooldown_days', [3, 10])[1])
    )
    
    # Update config with trial parameters
    trial_config = config.copy()
    trial_config['agents']['ppo']['learning_rate'] = learning_rate
    trial_config['agents']['ppo']['clip_range'] = clip_range
    trial_config['agents']['ppo']['ent_coef'] = ent_coef
    trial_config['agents']['ppo']['gamma'] = gamma
    trial_config['environment']['constraints']['max_turnover_daily'] = max_turnover_daily
    trial_config['environment']['risk']['spike_cooldown_days'] = spike_cooldown_days
    
    # Build environment with trial parameters
    try:
        env, _, _ = build_vix_etp_env(config=trial_config)
    except Exception as e:
        print(f"Failed to build environment: {e}")
        return -np.inf
    
    # Create agent
    agent = VIXETPPPOAgent(
        env=env,
        learning_rate=learning_rate,
        clip_range=clip_range,
        ent_coef=ent_coef,
        gamma=gamma,
        verbose=0
    )
    
    # Train
    try:
        agent.train(total_timesteps=n_timesteps)
    except Exception as e:
        print(f"Training failed: {e}")
        return -np.inf
    
    # Evaluate
    n_eval_episodes = 5
    episode_sharpes = []
    
    for _ in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        sharpe = info.get('sharpe_ratio', -np.inf)
        episode_sharpes.append(sharpe)
        
        # Report intermediate value for pruning
        trial.report(sharpe, len(episode_sharpes))
        
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    mean_sharpe = np.mean(episode_sharpes)
    
    return mean_sharpe


def tune_hyperparameters(
    config_path: str,
    n_trials: int = 50,
    n_timesteps: int = 50000,
    output_dir: str = 'outputs',
    timeout: Optional[int] = None,
    n_jobs: int = 1
) -> Dict:
    """
    Run Optuna hyperparameter tuning.
    
    Args:
        config_path: Path to config file
        n_trials: Number of trials
        n_timesteps: Training timesteps per trial
        output_dir: Output directory
        timeout: Timeout in seconds (optional)
        n_jobs: Number of parallel jobs
        
    Returns:
        Dictionary with best parameters and study results
    """
    # Load config
    config = load_config(config_path)
    
    # Get Optuna config
    optuna_config = config.get('optuna', {})
    n_trials = optuna_config.get('n_trials', n_trials)
    timeout = optuna_config.get('timeout_seconds', timeout)
    n_jobs = optuna_config.get('n_jobs', n_jobs)
    
    # Create study
    print("Starting Optuna hyperparameter tuning...")
    print(f"  Trials: {n_trials}")
    print(f"  Timesteps per trial: {n_timesteps}")
    print(f"  Timeout: {timeout}s" if timeout else "  No timeout")
    print(f"  Parallel jobs: {n_jobs}")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, config, config_path, n_timesteps),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=True
    )
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\nOptimization complete!")
    print(f"Best Sharpe ratio: {best_value:.4f}")
    print(f"Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'best_params': best_params,
        'best_value': float(best_value),
        'n_trials': len(study.trials),
        'trials': [
            {
                'number': trial.number,
                'value': trial.value if trial.value is not None else None,
                'params': trial.params,
                'state': trial.state.name
            }
            for trial in study.trials
        ]
    }
    
    results_path = output_path / 'optuna_best_params.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Tune VIX ETP agent hyperparameters')
    parser.add_argument('--config', type=str, default='config/vix_etp_config.yaml',
                        help='Path to config file')
    parser.add_argument('--trials', type=int, default=None,
                        help='Number of trials (default: from config)')
    parser.add_argument('--timesteps', type=int, default=50000,
                        help='Training timesteps per trial')
    parser.add_argument('--output', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds')
    parser.add_argument('--jobs', type=int, default=None,
                        help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    tune_hyperparameters(
        config_path=args.config,
        n_trials=args.trials,
        n_timesteps=args.timesteps,
        output_dir=args.output,
        timeout=args.timeout,
        n_jobs=args.jobs
    )
