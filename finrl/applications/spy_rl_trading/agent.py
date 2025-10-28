"""PPO Agent Wrapper for SPY Trading.

This module provides a wrapper around Stable-Baselines3 PPO for SPY trading,
with TensorBoard integration and model persistence.

Example:
    >>> from finrl.applications.spy_rl_trading.agent import PPOAgent
    >>> from finrl.applications.spy_rl_trading.environment import SPYTradingEnv
    >>> from finrl.config import SPY_PPO_PARAMS
    >>>
    >>> agent = PPOAgent(env=env, config=SPY_PPO_PARAMS)
    >>> trained_model = agent.train(total_timesteps=100_000)
    >>> agent.save("trained_models/spy_ppo")
"""

from __future__ import annotations

import os
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv


class TensorboardCallback(BaseCallback):
    """Custom TensorBoard callback for logging training metrics.

    Logs episode statistics to TensorBoard during training.
    """

    def __init__(self, verbose: int = 0):
        """Initialize callback.

        Args:
            verbose: Verbosity level (0: no output, 1: info, 2: debug)
        """
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """Called after each step.

        Returns:
            True to continue training, False to stop
        """
        # Log episode statistics if available
        if len(self.model.ep_info_buffer) > 0:
            episode_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            episode_lengths = [ep_info["l"] for ep_info in self.model.ep_info_buffer]

            self.logger.record("rollout/ep_rew_mean", sum(episode_rewards) / len(episode_rewards))
            self.logger.record("rollout/ep_len_mean", sum(episode_lengths) / len(episode_lengths))

        return True


class PPOAgent:
    """PPO Agent wrapper for SPY trading.

    Wraps Stable-Baselines3 PPO with SPY-specific configuration and utilities.
    """

    def __init__(self, env: Any, config: dict | None = None):
        """Initialize PPO agent.

        Args:
            env: Trading environment (SPYTradingEnv or DummyVecEnv)
            config: PPO hyperparameters (default: from finrl.config.SPY_PPO_PARAMS)
        """
        self.env = env if isinstance(env, DummyVecEnv) else DummyVecEnv([lambda: env])
        self.config = config or {}
        self.model = None

    def create_model(
        self,
        tensorboard_log: str = "./tensorboard_logs/spy_ppo",
        verbose: int = 1,
    ) -> PPO:
        """Create PPO model with configuration.

        Args:
            tensorboard_log: Directory for TensorBoard logs
            verbose: Verbosity level (0: no output, 1: info)

        Returns:
            Initialized PPO model
        """
        # Ensure tensorboard log directory exists
        os.makedirs(tensorboard_log, exist_ok=True)

        # Create PPO model
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            **self.config,
        )

        return self.model

    def train(
        self,
        total_timesteps: int = 100_000,
        tb_log_name: str = "spy_ppo_run",
        callback: BaseCallback | None = None,
    ) -> PPO:
        """Train PPO agent.

        Args:
            total_timesteps: Total training timesteps
            tb_log_name: TensorBoard log name (subdirectory)
            callback: Custom callback (default: TensorboardCallback)

        Returns:
            Trained PPO model

        Example:
            >>> agent = PPOAgent(env=env, config=SPY_PPO_PARAMS)
            >>> trained_model = agent.train(total_timesteps=100_000)
        """
        if self.model is None:
            self.create_model()

        # Use default TensorBoard callback if none provided
        if callback is None:
            callback = TensorboardCallback(verbose=0)

        # Train model
        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=callback,
        )

        return self.model

    def save(self, path: str):
        """Save trained model to disk.

        Args:
            path: Path to save model (without .zip extension)

        Example:
            >>> agent.save("trained_models/spy_ppo_discrete")
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model
        self.model.save(path)
        print(f"Model saved to {path}.zip")

    def load(self, path: str):
        """Load trained model from disk.

        Args:
            path: Path to load model (without .zip extension)

        Example:
            >>> agent = PPOAgent(env=env)
            >>> agent.load("trained_models/spy_ppo_discrete")
        """
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}.zip")

    def predict(self, observation, deterministic: bool = True):
        """Predict action for given observation.

        Args:
            observation: Current state observation
            deterministic: Use deterministic policy (default: True for evaluation)

        Returns:
            Tuple of (action, state) where state is for recurrent policies (None for PPO)

        Example:
            >>> obs, info = env.reset()
            >>> action, _ = agent.predict(obs)
        """
        if self.model is None:
            raise ValueError("No model to predict with. Load or train a model first.")

        return self.model.predict(observation, deterministic=deterministic)

    def evaluate(
        self,
        env: Any,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> tuple[float, float]:
        """Evaluate agent on environment.

        Args:
            env: Evaluation environment
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic policy

        Returns:
            Tuple of (mean_reward, std_reward)

        Example:
            >>> mean_reward, std_reward = agent.evaluate(test_env, n_episodes=10)
            >>> print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        """
        if self.model is None:
            raise ValueError("No model to evaluate. Load or train a model first.")

        from stable_baselines3.common.evaluation import evaluate_policy

        mean_reward, std_reward = evaluate_policy(
            self.model,
            env if isinstance(env, DummyVecEnv) else DummyVecEnv([lambda: env]),
            n_eval_episodes=n_episodes,
            deterministic=deterministic,
        )

        return mean_reward, std_reward


def create_spy_agent(
    env: Any,
    config: dict | None = None,
    tensorboard_log: str = "./tensorboard_logs/spy_ppo",
) -> PPOAgent:
    """Factory function to create SPY PPO agent.

    Args:
        env: Trading environment
        config: PPO hyperparameters
        tensorboard_log: TensorBoard log directory

    Returns:
        Initialized PPOAgent

    Example:
        >>> from finrl.config import SPY_PPO_PARAMS
        >>> agent = create_spy_agent(env, config=SPY_PPO_PARAMS)
    """
    agent = PPOAgent(env=env, config=config)
    agent.create_model(tensorboard_log=tensorboard_log)
    return agent
