# Reward design

The environment supports three explicit reward variants:

- `return_only`: gross next-period portfolio return.
- `return_net_cost`: portfolio return after cost is deducted from NAV.
- `risk_aware`: gross return minus `lambda_cost * cost_fraction`, `lambda_vol * trailing_volatility`, and `lambda_dd * current_drawdown`.

Reward scaling is explicit in `calculate_reward`. Default penalty coefficients are 1.0, 0.1, and 0.1, respectively, and scale is 1.0. In the risk-aware reward, cost is subtracted once as an explicit penalty; portfolio NAV also reflects the same real cost. The reward uses gross return as its return term, so the cost penalty is not applied twice inside that reward. In the net-cost variant, the reward is the net return and has no second cost penalty. These are learning objectives, not forecasts of investable performance.
