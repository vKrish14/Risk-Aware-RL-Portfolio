"""Stable-Baselines3 PPO construction and deterministic policy inference."""
from __future__ import annotations

from typing import Any

from stable_baselines3 import PPO


def create_ppo(env, config: dict[str, Any], *, seed: int = 42) -> PPO:
    """Construct PPO from a YAML-compatible hyperparameter mapping."""
    options = {key: config[key] for key in (
        "learning_rate", "gamma", "gae_lambda", "clip_range", "batch_size",
        "n_steps", "ent_coef", "vf_coef",
    ) if key in config}
    return PPO(
        policy=config.get("policy", "MlpPolicy"),
        env=env,
        seed=seed,
        device=config.get("device", "cpu"),
        verbose=1,
        **options,
    )
