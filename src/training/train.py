"""Locally train PPO with chronological validation-only checkpoint selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.agents.ppo_agent import create_ppo
from src.data.cleaner import align_common_dates
from src.environment.portfolio_env import PortfolioEnv
from src.features.pipeline import build_feature_splits
from src.utils.seed import set_seed


def _returns_for(prices: pd.DataFrame, annual_cash_return: float, day_count: int) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None)
    returns["CASH"] = (1 + annual_cash_return) ** (1 / day_count) - 1
    return returns.dropna(how="any")


def prepare_data(root: Path):
    config = yaml.safe_load((root / "configs/config.yaml").read_text(encoding="utf-8"))
    # Use all raw split prices to give validation/test causal rolling indicators their prior history.
    frames = [pd.read_csv(root / f"data/processed/{name}_prices.csv", index_col="date", parse_dates=["date"]) for name in ("train", "validation", "test")]
    full_prices = pd.concat(frames).sort_index()
    full_prices, _ = align_common_dates(full_prices)
    windows = config["project"]
    feature_splits, scaler = build_feature_splits(
        full_prices,
        train_end=windows["train"]["end"],
        validation_end=windows["validation"]["end"],
        test_start=windows["test"]["start"],
        metadata_path=root / "models/metadata/preprocessor.json",
    )
    returns = _returns_for(
        full_prices,
        config["portfolio"]["synthetic_cash_annual_return"],
        config["portfolio"]["cash_day_count"],
    )
    return config, feature_splits, returns, scaler


def train(root: str | Path = ".", *, timesteps: int | None = None, reward_variant: str = "risk_aware") -> Path:
    """Train PPO and save final/best models with configuration provenance."""
    project = Path(root).resolve()
    config, splits, returns, scaler = prepare_data(project)
    ppo = yaml.safe_load((project / "configs/ppo_config.yaml").read_text(encoding="utf-8"))
    seed = int(ppo.get("seed", config["project"].get("seed", 42)))
    set_seed(seed)
    env_config = config["portfolio"]
    make_env = lambda name: PortfolioEnv(
        splits[name], returns.loc[splits[name].index],
        initial_capital=config["project"]["initial_capital"],
        transaction_cost_rate=env_config["transaction_cost_rate"],
        reward_variant=reward_variant,
    )
    training_env, validation_env = make_env("train"), Monitor(make_env("validation"))
    model = create_ppo(training_env, ppo, seed=seed)
    checkpoint_dir = project / "models/checkpoints" / reward_variant
    log_dir = project / "results/reports/training" / reward_variant
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    steps = int(timesteps if timesteps is not None else ppo["total_timesteps"])
    callback = EvalCallback(
        validation_env,
        best_model_save_path=str(checkpoint_dir / "best"),
        log_path=str(log_dir),
        eval_freq=max(int(ppo["n_steps"]), min(10_000, steps // 2 if steps >= 2 else 1)),
        n_eval_episodes=1,
        deterministic=True,
        render=False,
    )
    model.learn(total_timesteps=steps, callback=callback, progress_bar=False)
    final_path = checkpoint_dir / "final_model"
    model.save(str(final_path))
    metadata = {
        "algorithm": "Stable-Baselines3 PPO",
        "policy": ppo.get("policy", "MlpPolicy"),
        "seed": seed,
        "requested_timesteps": steps,
        "actual_timesteps": model.num_timesteps,
        "reward_variant": reward_variant,
        "reward_parameters": {
            "transaction_cost_rate": env_config["transaction_cost_rate"],
            "transaction_cost_penalty": 1.0,
            "volatility_penalty": 0.1,
            "drawdown_penalty": 0.1,
            "reward_scale": 1.0,
        },
        "data_sha256": json.loads((project / "data/processed/data_manifest.json").read_text(encoding="utf-8"))["data_sha256"],
        "selected_using": "validation mean reward",
        "train_first": splits["train"].index.min().date().isoformat(),
        "train_last": splits["train"].index.max().date().isoformat(),
        "validation_first": splits["validation"].index.min().date().isoformat(),
        "validation_last": splits["validation"].index.max().date().isoformat(),
        "test_used_for_selection": False,
        "ppo_config": ppo,
        "preprocessing": scaler.to_dict(),
    }
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    training_env.close()
    validation_env.close()
    return final_path.with_suffix(".zip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--reward", choices=("return_only", "return_net_cost", "risk_aware", "all"), default="risk_aware")
    args = parser.parse_args()
    variants = ("return_only", "return_net_cost", "risk_aware") if args.reward == "all" else (args.reward,)
    for variant in variants:
        print(train(args.root, timesteps=args.timesteps, reward_variant=variant))


if __name__ == "__main__":
    main()
