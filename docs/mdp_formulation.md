# MDP formulation

## Observation

At date t the policy receives all causal market features calculated through t, standardized by training-only means and scales; the current six portfolio weights; NAV relative to initial capital; prior net return; current drawdown; and trailing 20-session portfolio-return volatility. It does not receive the next-period returns.

## Action

The PPO action is six bounded logits. Stable softmax maps them to nonnegative risky-asset and cash target weights that sum to one. The bounded action range is [-1, 1].

## Transition/accounting convention

At t, one-way turnover is `0.5 * sum(abs(target_weights - current_drifted_weights))`. Cost fraction is `0.001 * turnover`, deducted from pre-rebalance NAV. The target weights then earn the observed close-to-close return from t to t+1. NAV transition is `NAV[t+1] = NAV[t] * (1-cost_fraction) * (1 + target dot asset_returns[t+1])`. Holdings drift after returns: `w[i,t+1] = target[i] * (1+r[i,t+1]) / (1+portfolio_return[t+1])`. This simplified fractional-share convention assumes the cost is paid from NAV and target allocations apply to the post-rebalance portfolio; it does not model lots, spreads, taxes, or market impact.

History exhaustion is a Gymnasium truncation. Bankruptcy is termination. Each call to step observes current-date features, trades at that date, and uses the next row's return.

## Episode and reproducibility

An episode spans one supplied chronological feature panel. The default seed is 42. Initial capital is $100,000 allocated fully to synthetic cash. Cash return and transaction costs are applied by the same functions in training and independent evaluation.
