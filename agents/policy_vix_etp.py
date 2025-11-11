"""PPO agent wrapper for VIX ETP trading."""

from typing import Optional, Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pandas as pd


class VIXETPPPOAgent:
    """PPO agent wrapper for VIX ETP trading environment."""
    
    def __init__(
        self,
        env,
        policy: str = 'MlpPolicy',
        learning_rate: float = 0.0003,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        seed: Optional[int] = None,
        device: str = 'auto',
        verbose: int = 1
    ):
        """
        Initialize PPO agent.
        
        Args:
            env: VIX ETP environment
            policy: Policy network architecture
            learning_rate: Learning rate
            n_steps: Steps per rollout
            batch_size: Batch size for training
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clipping parameter
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm
            seed: Random seed
            device: Device to use ('cpu', 'cuda', or 'auto')
            verbose: Verbosity level
        """
        # Wrap environment in DummyVecEnv for stable-baselines3
        if not isinstance(env, DummyVecEnv):
            env = DummyVecEnv([lambda: env])
        
        self.model = PPO(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            seed=seed,
            device=device,
            verbose=verbose
        )
        
    def train(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        eval_env=None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5
    ):
        """
        Train the agent.
        
        Args:
            total_timesteps: Total training timesteps
            callback: Custom callback
            eval_env: Evaluation environment
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
        """
        callbacks = []
        
        if callback:
            callbacks.append(callback)
        
        if eval_env is not None:
            # Wrap eval env
            if not isinstance(eval_env, DummyVecEnv):
                eval_env = DummyVecEnv([lambda: eval_env])
            
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path='./outputs/best_model',
                log_path='./outputs/eval_logs',
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        final_callback = callbacks[0] if len(callbacks) == 1 else callbacks if callbacks else None
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=final_callback
        )
    
    def predict(self, observation, deterministic: bool = True):
        """
        Predict action given observation.
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            action, state
        """
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path: str):
        """Save model to file."""
        self.model.save(path)
    
    def load(self, path: str):
        """Load model from file."""
        self.model = PPO.load(path)
    
    @classmethod
    def from_config(cls, env, config: Dict[str, Any], seed: Optional[int] = None):
        """
        Create agent from configuration dictionary.
        
        Args:
            env: VIX ETP environment
            config: Configuration dictionary
            seed: Random seed
            
        Returns:
            VIXETPPPOAgent instance
        """
        ppo_config = config.get('agents', {}).get('ppo', {})
        
        return cls(
            env=env,
            policy=ppo_config.get('policy', 'MlpPolicy'),
            learning_rate=ppo_config.get('learning_rate', 0.0003),
            n_steps=ppo_config.get('n_steps', 2048),
            batch_size=ppo_config.get('batch_size', 64),
            n_epochs=ppo_config.get('n_epochs', 10),
            gamma=ppo_config.get('gamma', 0.99),
            gae_lambda=ppo_config.get('gae_lambda', 0.95),
            clip_range=ppo_config.get('clip_range', 0.2),
            ent_coef=ppo_config.get('ent_coef', 0.01),
            vf_coef=ppo_config.get('vf_coef', 0.5),
            max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
            seed=seed or config.get('training', {}).get('seed', 42),
            verbose=1
        )


class TradingMetricsCallback(BaseCallback):
    """Custom callback for logging trading metrics during training."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_returns = []
        self.episode_sharpes = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """Called at each step."""
        # Check if episode finished
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                # Episode finished
                ep_info = info['episode']
                self.episode_lengths.append(ep_info['l'])
                
                # Get final info
                if 'total_return' in info:
                    self.episode_returns.append(info['total_return'])
                if 'sharpe_ratio' in info:
                    self.episode_sharpes.append(info['sharpe_ratio'])
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout."""
        if len(self.episode_returns) > 0:
            avg_return = np.mean(self.episode_returns[-10:])
            self.logger.record('trading/avg_return_10ep', avg_return)
        
        if len(self.episode_sharpes) > 0:
            avg_sharpe = np.mean(self.episode_sharpes[-10:])
            self.logger.record('trading/avg_sharpe_10ep', avg_sharpe)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return {
            'episode_returns': self.episode_returns,
            'episode_sharpes': self.episode_sharpes,
            'episode_lengths': self.episode_lengths
        }
