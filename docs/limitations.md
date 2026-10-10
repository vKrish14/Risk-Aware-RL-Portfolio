# Limitations

- Yahoo Finance is an external source; availability, historical corrections, and licensing/access terms can change. Reproducibility depends on retaining the local cached input and its manifest/hash.
- Adjusted prices are used for split/dividend-adjusted close return calculations. Cash is synthetic, and execution is simplified to daily fractional-share target weights.
- The cost model includes proportional turnover costs only. It omits bid-ask spreads, slippage, taxes, liquidity, market impact, and order failures.
- Technical indicators are descriptive transformations of past prices and may not add predictive information. Standardization is trained on the training period only, but market regimes can change.
- PPO is stochastic during training, sensitive to random seeds and reward specification, and can overfit historical paths. A single seed and one test window do not establish robust investment performance.
- Minimum-variance and momentum baselines are simple reference rules; their lookback choices are not claimed optimal.
- The service and dashboard are local research demonstrations, not trading software. Outputs are hypothetical and are not investment advice.
- Report actual benchmark/model results, including underperformance, before drawing conclusions.
