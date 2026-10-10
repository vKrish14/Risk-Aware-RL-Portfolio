"""PPO model construction, persistence, and deterministic reload tests."""
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.agents.ppo_agent import create_ppo
from src.environment.portfolio_env import PortfolioEnv


def test_ppo_save_reload_preserves_deterministic_inference(tmp_path) -> None:
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    features = pd.DataFrame({"x": np.linspace(-1, 1, len(dates))}, index=dates)
    returns = pd.DataFrame({"SPY": np.zeros(len(dates)), "CASH": np.zeros(len(dates))}, index=dates)
    env = PortfolioEnv(features, returns)
    config = {"policy": "MlpPolicy", "learning_rate": .0003, "gamma": .99, "gae_lambda": .95, "clip_range": .2, "batch_size": 8, "n_steps": 16, "ent_coef": .01, "vf_coef": .5, "device": "cpu"}
    model = create_ppo(env, config, seed=42)
    model.learn(total_timesteps=16)
    obs, _ = env.reset(seed=42)
    expected, _ = model.predict(obs, deterministic=True)
    path = tmp_path / "ppo"
    model.save(path)
    loaded = PPO.load(path, env=env, device="cpu")
    actual, _ = loaded.predict(obs, deterministic=True)
    np.testing.assert_allclose(expected, actual)
