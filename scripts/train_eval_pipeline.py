"""Training and evaluation pipeline for VIX ETP agents."""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd

from env.builder import build_vix_etp_env, load_config
from agents.policy_vix_etp import VIXETPPPOAgent, TradingMetricsCallback
from agents.policy_vix_etp_sac import VIXETPSACAgent, SACTradingMetricsCallback


def split_data(
    price_data: pd.DataFrame,
    volume_data: pd.DataFrame,
    train_split: float = 0.7,
    val_split: float = 0.15
) -> Tuple:
    """
    Split data into train, validation, and test sets.
    
    Args:
        price_data: Price DataFrame
        volume_data: Volume DataFrame
        train_split: Training data fraction
        val_split: Validation data fraction
        
    Returns:
        Tuple of (train_prices, train_volumes, val_prices, val_volumes, test_prices, test_volumes)
    """
    n = len(price_data)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    
    train_prices = price_data.iloc[:train_end]
    train_volumes = volume_data.iloc[:train_end]
    
    val_prices = price_data.iloc[train_end:val_end]
    val_volumes = volume_data.iloc[train_end:val_end]
    
    test_prices = price_data.iloc[val_end:]
    test_volumes = volume_data.iloc[val_end:]
    
    return train_prices, train_volumes, val_prices, val_volumes, test_prices, test_volumes


def train_agent(
    config_path: str,
    agent_type: str = 'ppo',
    total_timesteps: int = 100000,
    output_dir: str = 'outputs',
    seed: Optional[int] = None
) -> Dict:
    """
    Train a VIX ETP agent.
    
    Args:
        config_path: Path to config YAML file
        agent_type: Type of agent ('ppo' or 'sac')
        total_timesteps: Total training timesteps
        output_dir: Output directory for models and logs
        seed: Random seed
        
    Returns:
        Dictionary with training metrics
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = load_config(config_path)
    
    if seed is None:
        seed = config.get('training', {}).get('seed', 42)
    
    # Override total_timesteps from config if not provided
    if total_timesteps == 100000:  # Default value
        if agent_type == 'ppo':
            total_timesteps = config.get('agents', {}).get('ppo', {}).get('total_timesteps', 100000)
        else:
            total_timesteps = config.get('agents', {}).get('sac', {}).get('total_timesteps', 100000)
    
    # Build environment
    print("Building training environment...")
    train_env, price_data, volume_data = build_vix_etp_env(config_path=config_path)
    
    # Split data for validation
    training_config = config.get('training', {})
    train_split = training_config.get('train_split', 0.7)
    val_split = training_config.get('val_split', 0.15)
    
    # Create training agent
    print(f"\nCreating {agent_type.upper()} agent...")
    
    if agent_type == 'ppo':
        agent = VIXETPPPOAgent.from_config(train_env, config, seed=seed)
        callback = TradingMetricsCallback(verbose=1)
    elif agent_type == 'sac':
        agent = VIXETPSACAgent.from_config(train_env, config, seed=seed)
        callback = SACTradingMetricsCallback(verbose=1)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Train
    print(f"\nTraining for {total_timesteps} timesteps...")
    agent.train(
        total_timesteps=total_timesteps,
        callback=callback
    )
    
    # Save model
    model_path = output_path / f"vix_etp_{agent_type}_model.zip"
    agent.save(str(model_path))
    print(f"\nModel saved to {model_path}")
    
    # Get metrics
    metrics = callback.get_metrics()
    
    # Save metrics
    metrics_data = {
        'agent_type': agent_type,
        'total_timesteps': total_timesteps,
        'seed': seed,
        'final_metrics': {
            'avg_return': float(np.mean(metrics['episode_returns'][-10:])) if metrics['episode_returns'] else 0.0,
            'avg_sharpe': float(np.mean(metrics['episode_sharpes'][-10:])) if metrics['episode_sharpes'] else 0.0,
            'n_episodes': len(metrics['episode_returns'])
        }
    }
    
    metrics_path = output_path / f"training_metrics_{agent_type}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"Training metrics saved to {metrics_path}")
    print(f"\nFinal metrics:")
    print(f"  Average return (last 10 ep): {metrics_data['final_metrics']['avg_return']:.4f}")
    print(f"  Average Sharpe (last 10 ep): {metrics_data['final_metrics']['avg_sharpe']:.4f}")
    print(f"  Total episodes: {metrics_data['final_metrics']['n_episodes']}")
    
    return metrics_data


def evaluate_agent(
    config_path: str,
    model_path: str,
    agent_type: str = 'ppo',
    n_episodes: int = 10,
    output_dir: str = 'outputs'
) -> Dict:
    """
    Evaluate a trained agent.
    
    Args:
        config_path: Path to config YAML file
        model_path: Path to trained model
        agent_type: Type of agent ('ppo' or 'sac')
        n_episodes: Number of evaluation episodes
        output_dir: Output directory for results
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Build environment
    print("Building evaluation environment...")
    eval_env, price_data, volume_data = build_vix_etp_env(config_path=config_path)
    
    # Load agent
    print(f"Loading {agent_type.upper()} agent from {model_path}...")
    if agent_type == 'ppo':
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
    elif agent_type == 'sac':
        from stable_baselines3 import SAC
        model = SAC.load(model_path)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Evaluate
    print(f"\nEvaluating for {n_episodes} episodes...")
    
    episode_returns = []
    episode_sharpes = []
    episode_values = []
    
    for ep in range(n_episodes):
        obs, info = eval_env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        
        # Record metrics
        episode_returns.append(info['total_return'])
        episode_sharpes.append(info.get('sharpe_ratio', 0.0))
        episode_values.append(info['portfolio_value'])
        
        print(f"  Episode {ep+1}/{n_episodes}: Return={info['total_return']:.4f}, Sharpe={info.get('sharpe_ratio', 0.0):.4f}")
    
    # Compute statistics
    eval_metrics = {
        'agent_type': agent_type,
        'n_episodes': n_episodes,
        'mean_return': float(np.mean(episode_returns)),
        'std_return': float(np.std(episode_returns)),
        'mean_sharpe': float(np.mean(episode_sharpes)),
        'std_sharpe': float(np.std(episode_sharpes)),
        'mean_final_value': float(np.mean(episode_values)),
        'episode_returns': [float(x) for x in episode_returns],
        'episode_sharpes': [float(x) for x in episode_sharpes]
    }
    
    # Save metrics
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metrics_path = output_path / f"eval_metrics_{agent_type}.json"
    with open(metrics_path, 'w') as f:
        json.dump(eval_metrics, f, indent=2)
    
    print(f"\nEvaluation results saved to {metrics_path}")
    print(f"\nEvaluation summary:")
    print(f"  Mean return: {eval_metrics['mean_return']:.4f} ± {eval_metrics['std_return']:.4f}")
    print(f"  Mean Sharpe: {eval_metrics['mean_sharpe']:.4f} ± {eval_metrics['std_sharpe']:.4f}")
    
    return eval_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or evaluate VIX ETP agent')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True,
                        help='Mode: train or eval')
    parser.add_argument('--config', type=str, default='config/vix_etp_config.yaml',
                        help='Path to config file')
    parser.add_argument('--agent', type=str, choices=['ppo', 'sac'], default='ppo',
                        help='Agent type: ppo or sac')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total training timesteps')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model for evaluation')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--output', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_agent(
            config_path=args.config,
            agent_type=args.agent,
            total_timesteps=args.timesteps,
            output_dir=args.output,
            seed=args.seed
        )
    else:  # eval
        if args.model is None:
            args.model = f"{args.output}/vix_etp_{args.agent}_model.zip"
        
        evaluate_agent(
            config_path=args.config,
            model_path=args.model,
            agent_type=args.agent,
            n_episodes=args.episodes,
            output_dir=args.output
        )
