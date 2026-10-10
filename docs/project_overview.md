# Project overview

## Research objective

Can a locally trained, risk-aware PPO policy allocate across SPY, QQQ, GLD, IEF, VNQ, and synthetic cash with competitive risk-adjusted results after transaction costs? This is an educational research system, not live trading or investment advice.

## Fixed experiment setup

Daily, adjusted close prices come from Yahoo Finance through yfinance. Project windows use inclusive boundaries: train 2015-01-01–2021-12-31; validation 2022-01-01–2023-12-31; test 2024-01-01 through the latest completed observation obtained. yfinance's `end` is exclusive; with no configured inclusive end, the downloader uses New York market time to exclude a still-forming session and include today's completed session after a close buffer. The observed panel is stored in ignored local cache/processed paths with a manifest and SHA-256 data hash.

All aligned dates contain all five risky-asset prices. Missing rows are removed only when any configured asset is missing, and the exact dates and per-asset missing counts are recorded; no future-price filling is used. Adjusted close is fetched with `auto_adjust=True` and the returned `Close` column is used once.

Initial capital is $100,000. Weights are long-only, unlevered, sum to one across five risky assets and cash. The daily transaction cost is 0.1% times one-way turnover (half the L1 target-weight change). Cash earns a configurable 2% annual return converted to a 252-session daily rate. The annual risk-free rate for Sharpe/Sortino is configured separately (2%). Seed is 42.

See `mdp_formulation.md`, `reward_design.md`, `training.md`, `evaluation.md`, `architecture.md`, and `limitations.md` for methodological detail.

Project-authored code is distributed under the MIT License in `LICENSE`. No external source code was copied; third-party references are documented separately.
