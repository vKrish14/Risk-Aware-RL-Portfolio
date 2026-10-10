# Evaluation

`python -m src.evaluation.run_evaluation` generates the held-out report after training settings are frozen. It compares each available PPO reward variant against equal weight, buy-and-hold, trailing momentum, and rolling minimum variance, using the same eligible return rows, dates, initial capital, turnover definition, cost, and annualization.

Daily returns are annualized using 252 observations. Annualized return is geometric; annualized volatility uses sample standard deviation. Sharpe and Sortino subtract the configured 2% annual risk-free rate converted to a daily rate. Maximum drawdown includes initial capital as the first peak; Calmar divides geometric annualized return by absolute maximum drawdown. Undefined metrics are reported as JSON null. Turnover is the sum of half-L1 target-weight changes. Cumulative transaction cost is the dollar cost paid at each rebalance.

Outputs include `results/tables/test_metrics.csv`, `results/backtests/test_report.json`, equity and drawdown curves, rolling Sharpe, risk-return, and allocation plots. These are generated artifacts ignored by Git. Do not select models or tune hyperparameters based on the test report.
