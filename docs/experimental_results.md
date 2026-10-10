# Observed experiment results

## Protocol

The original three reward variants and the added tail-risk variant were trained locally with SB3 PPO on CPU, seed 42, with 100,000 requested timesteps each. SB3 completed 100,352 timesteps per run because it collects 2,048-step rollouts. The train observations span 2015-03-31 through 2021-12-31 after the 60-session feature warm-up; validation spans 2022-01-03 through 2023-12-29. Best checkpoints were selected only using deterministic validation reward. The held-out test observations cover 2024-01-02 through 2026-10-09; their realized return series starts on 2024-01-03 because the first row is the decision date. All amounts and metrics below come from the saved local Yahoo Finance adjusted-price data and generated test report.

The experimental tail-risk policy adds rolling 60-day 95% CVaR (the mean loss in the worst 5% of recent daily portfolio returns) to its state and penalizes values above a 2% daily tail-loss budget. This is a project-specific experiment, not a claim of a new RL algorithm. Its validation reward peaked at 10,000 timesteps (-5.74) and declined to -11.21 at 100,000; the validation-selected checkpoint was used for the test. This pattern suggests training instability/overfitting and argues against treating the experiment as a successful improvement.

The transaction-cost model charges 0.1% × half-L1 target-weight change. All strategies use the same test rows, initial $100,000, synthetic 2% cash return, and 2% annual risk-free rate for ratios. One seed and one test window are insufficient to establish general performance.

## Held-out results

| Strategy | Total return | Annualized return | Annualized volatility | Sharpe | Maximum drawdown | Daily CVaR 95% | Turnover sum | Transaction costs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Experimental CVaR-aware PPO | 39.33% | 12.78% | 9.11% | 1.15 | -9.18% | 1.336% | 17.35 | $2,066.50 |
| Risk-aware PPO | 41.71% | 13.47% | 9.30% | 1.19 | -8.64% | 1.343% | 20.39 | $2,448.34 |
| PPO return net of cost | 34.31% | 11.29% | 9.89% | 0.93 | -11.35% | 1.417% | 75.27 | $8,998.04 |
| PPO return only | 32.53% | 10.75% | 9.61% | 0.90 | -10.78% | 1.380% | 77.92 | $9,203.61 |
| Equal weight | 44.10% | 14.16% | 9.28% | 1.26 | -8.66% | 1.345% | 2.71 | $316.02 |
| Buy and hold | 47.03% | 15.00% | 10.32% | 1.21 | -8.83% | 1.560% | 0.83 | $83.33 |
| Momentum | 58.22% | 18.10% | 11.60% | 1.32 | -10.75% | 1.691% | 52.89 | $6,752.66 |
| Minimum variance | 4.03% | 1.44% | 0.58% | -0.94 | -1.47% | 0.036% | 2.25 | $223.14 |

## Interpretation

The experimental CVaR-aware policy had slightly lower measured daily 95% CVaR than the standard risk-aware PPO (1.336% vs 1.343%) and lower turnover/costs, but it also had lower return and Sharpe and a worse maximum drawdown. The small CVaR difference is not evidence of a reliable risk improvement. Standard risk-aware PPO did not beat equal weight, buy-and-hold, or momentum on Sharpe or total return in this test. This run therefore does not support a claim that PPO achieved superior risk-adjusted performance. The return-only PPO variants traded frequently and paid materially higher modeled costs. The minimum-variance portfolio had low volatility and low return; at a 2% risk-free assumption its Sharpe was negative.

These observations are not investment advice or a forecast. They depend on the chosen assets, dates, single seed, synthetic cash, close-price convention, and simplified cost model. Do not tune against this test period. Result artifacts are `results/backtests/test_report.json`, `results/tables/test_metrics.csv`, and figures under `results/figures/`.
