"""SAC agent wrapper for VIX ETP trading."""

from typing import Optional, Dict, Any
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np


class VIXETPSACAgent:
    """SAC agent wrapper for VIX ETP trading environment."""
    
    def __init__(
        self,
        env,
        policy: str = 'MlpPolicy',
        learning_rate: float = 0.0003,
        buffer_size: int = 100000,
        learning_starts: int = 1000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        ent_coef: str = 'auto',
        seed: Optional[int] = None,
        device: str = 'auto',
        verbose: int = 1
    ):
        """
        Initialize SAC agent.
        
        Args:
            env: VIX ETP environment
            policy: Policy network architecture
            learning_rate: Learning rate
            buffer_size: Size of replay buffer
            learning_starts: Steps before learning starts
            batch_size: Batch size for training
            tau: Soft update coefficient
            gamma: Discount factor
            train_freq: Update frequency
            gradient_steps: Gradient steps per update
            ent_coef: Entropy coefficient ('auto' for automatic tuning)
            seed: Random seed
            device: Device to use ('cpu', 'cuda', or 'auto')
            verbose: Verbosity level
        """
        # Wrap environment in DummyVecEnv for stable-baselines3
        if not isinstance(env, DummyVecEnv):
            env = DummyVecEnv([lambda: env])
        
        self.model = SAC(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            ent_coef=ent_coef,
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
        self.model = SAC.load(path)
    
    @classmethod
    def from_config(cls, env, config: Dict[str, Any], seed: Optional[int] = None):
        """
        Create agent from configuration dictionary.
        
        Args:
            env: VIX ETP environment
            config: Configuration dictionary
            seed: Random seed
            
        Returns:
            VIXETPSACAgent instance
        """
        sac_config = config.get('agents', {}).get('sac', {})
        
        return cls(
            env=env,
            policy=sac_config.get('policy', 'MlpPolicy'),
            learning_rate=sac_config.get('learning_rate', 0.0003),
            buffer_size=sac_config.get('buffer_size', 100000),
            learning_starts=sac_config.get('learning_starts', 1000),
            batch_size=sac_config.get('batch_size', 256),
            tau=sac_config.get('tau', 0.005),
            gamma=sac_config.get('gamma', 0.99),
            train_freq=sac_config.get('train_freq', 1),
            gradient_steps=sac_config.get('gradient_steps', 1),
            ent_coef=sac_config.get('ent_coef', 'auto'),
            seed=seed or config.get('training', {}).get('seed', 42),
            verbose=1
        )


class SACTradingMetricsCallback(BaseCallback):
    """Custom callback for logging trading metrics during SAC training."""
    
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
