"""Daily long-only portfolio environment with explicit rebalance accounting."""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.environment.action_space import logits_to_weights
from src.environment.rewards import calculate_reward
from src.environment.transaction_costs import half_l1_turnover, transaction_cost


class PortfolioEnv(gym.Env):
    """Portfolio MDP using features at t and asset returns from t to t+1.

    The action targets post-rebalance weights. One-way turnover is half the
    L1 change from current drifted holdings. Costs are deducted from NAV at
    rebalance; target weights then earn the next observed daily returns and
    drift naturally. Exhausting the supplied price history is truncation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: pd.DataFrame,
        asset_returns: pd.DataFrame,
        *,
        initial_capital: float = 100_000.0,
        transaction_cost_rate: float = 0.001,
        cash_column: str = "CASH",
        reward_variant: str = "risk_aware",
        transaction_cost_penalty: float = 1.0,
        volatility_penalty: float = 0.1,
        drawdown_penalty: float = 0.1,
        reward_scale: float = 1.0,
        volatility_window: int = 20,
    ) -> None:
        super().__init__()
        if len(features) < 2 or features.empty or asset_returns.empty:
            raise ValueError("Environment requires at least two feature dates and returns.")
        if not features.index.equals(asset_returns.index):
            raise ValueError("Feature and asset-return indexes must match exactly.")
        if features.index.has_duplicates or not features.index.is_monotonic_increasing:
            raise ValueError("Environment dates must be unique and sorted.")
        if not np.isfinite(features.to_numpy()).all() or not np.isfinite(asset_returns.to_numpy()).all():
            raise ValueError("Environment inputs must be finite.")
        if initial_capital <= 0 or not 0 <= transaction_cost_rate < 1:
            raise ValueError("Capital must be positive and cost rate in [0, 1).")
        if cash_column not in asset_returns:
            raise ValueError(f"Asset returns must include a {cash_column} column.")
        if (asset_returns.to_numpy() < -1).any():
            raise ValueError("Asset returns cannot be less than -100%.")
        self.features = features.astype(float)
        self.asset_returns = asset_returns.astype(float)
        self.asset_names = list(asset_returns.columns)
        self.initial_capital = float(initial_capital)
        self.transaction_cost_rate = float(transaction_cost_rate)
        self.reward_variant = reward_variant
        self.reward_params = {
            "transaction_cost_penalty": transaction_cost_penalty,
            "volatility_penalty": volatility_penalty,
            "drawdown_penalty": drawdown_penalty,
            "scale": reward_scale,
        }
        self.volatility_window = volatility_window
        self._cash_index = self.asset_names.index(cash_column)
        self._feature_values = self.features.to_numpy(dtype=np.float32)
        self._returns = self.asset_returns.to_numpy(dtype=np.float64)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.asset_names),), dtype=np.float32)
        self.include_cvar_state = reward_variant == "cvar_aware"
        observation_size = self._feature_values.shape[1] + len(self.asset_names) + 4 + int(self.include_cvar_state)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_size,), dtype=np.float32)
        self._index = 0
        self._nav = self.initial_capital
        self._peak = self.initial_capital
        self._weights = np.zeros(len(self.asset_names), dtype=float)
        self._weights[self._cash_index] = 1.0
        self._previous_return = 0.0
        self._return_history: deque[float] = deque(maxlen=max(volatility_window, 60))
        self._last_info: dict[str, object] = {}

    def _observation(self) -> np.ndarray:
        vol = float(np.std(list(self._return_history)[-self.volatility_window :], ddof=0)) if self._return_history else 0.0
        drawdown = max(0.0, 1.0 - self._nav / self._peak)
        cvar95_loss = self._cvar95_loss()
        portfolio_state = [self._nav / self.initial_capital, self._previous_return, drawdown, vol]
        if self.include_cvar_state:
            portfolio_state.append(cvar95_loss)
        obs = np.concatenate((
            self._feature_values[self._index],
            self._weights.astype(np.float32),
            np.asarray(portfolio_state, dtype=np.float32),
        )).astype(np.float32)
        return obs

    def _cvar95_loss(self) -> float:
        """Estimate recent 95% conditional loss from at most 60 past returns."""
        if not self._return_history:
            return 0.0
        values = np.asarray(self._return_history, dtype=float)
        cutoff = float(np.quantile(values, 0.05))
        return max(0.0, -float(values[values <= cutoff].mean()))

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Reset at the first feature row and allocate all initial capital to cash."""
        super().reset(seed=seed)
        self._index = 0
        self._nav = self.initial_capital
        self._peak = self.initial_capital
        self._weights = np.zeros(len(self.asset_names), dtype=float)
        self._weights[self._cash_index] = 1.0
        self._previous_return = 0.0
        self._return_history.clear()
        self._last_info = {"date": self.features.index[0].date().isoformat(), "portfolio_value": self._nav, "weights": self._weights.copy()}
        return self._observation(), dict(self._last_info)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Rebalance using current state and apply the next row's returns."""
        if self._index >= len(self.features) - 1:
            raise RuntimeError("step() called after the episode ended; call reset().")
        target = logits_to_weights(np.clip(np.asarray(action, dtype=float), -1.0, 1.0))
        if target.size != len(self.asset_names):
            raise ValueError(f"Action needs {len(self.asset_names)} logits.")
        turnover = half_l1_turnover(self._weights, target)
        cost_fraction = transaction_cost(turnover, self.transaction_cost_rate)
        next_returns = self._returns[self._index + 1]
        gross_return = float(np.dot(target, next_returns))
        net_growth = (1.0 - cost_fraction) * (1.0 + gross_return)
        net_return = net_growth - 1.0
        old_nav = self._nav
        self._nav *= net_growth
        self._peak = max(self._peak, self._nav)
        self._weights = target * (1.0 + next_returns) / (1.0 + gross_return)
        self._previous_return = net_return
        self._return_history.append(net_return)
        self._index += 1
        drawdown = max(0.0, 1.0 - self._nav / self._peak)
        volatility = float(np.std(list(self._return_history)[-self.volatility_window :], ddof=0))
        reward = calculate_reward(
            self.reward_variant,
            gross_return=gross_return,
            net_return=net_return,
            cost_fraction=cost_fraction,
            volatility=volatility,
            drawdown=drawdown,
            cvar95_loss=self._cvar95_loss(),
            **self.reward_params,
        )
        terminated = self._nav <= 0 or not np.isfinite(self._nav)
        truncated = self._index >= len(self.features) - 1 and not terminated
        info = {
            "date": self.features.index[self._index].date().isoformat(),
            "portfolio_value": float(self._nav),
            "weights": self._weights.copy(),
            "target_weights": target.copy(),
            "gross_return": gross_return,
            "net_return": net_return,
            "turnover": turnover,
            "transaction_cost_fraction": cost_fraction,
            "transaction_cost_amount": old_nav * cost_fraction,
            "drawdown": drawdown,
            "volatility": volatility,
            "cvar95_loss": self._cvar95_loss(),
        }
        self._last_info = info
        return self._observation(), float(reward), bool(terminated), bool(truncated), info
