# Risk-Aware RL Portfolio Management

Educational research code for locally trained PPO allocation across SPY, QQQ, GLD, IEF, VNQ, and synthetic cash. The project includes an adjusted-price data pipeline, causal features, a Gymnasium portfolio environment, configurable rewards, PPO training, conventional benchmarks, cost-aware held-out backtesting, FastAPI, and Streamlit. It is not a live-trading system and does not provide investment advice.

## Setup (Windows PowerShell)

Use Python 3.11–3.13. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\verify_environment.py
```

If activation is blocked in this PowerShell process, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate. `deactivate` exits the environment.

## Data and experiments

```powershell
# Download if not cached; writes data/processed/*.csv and provenance manifest
python -m src.data.pipeline
# Force a fresh provider request
python -m src.data.pipeline --refresh
# Train one reward variant or all three (100,000 requested steps each by default)
python -m src.training.train --timesteps 100000 --reward risk_aware
python -m src.training.train --reward all
# Generate held-out test comparisons after freezing methodology
python -m src.evaluation.run_evaluation
```

Project date boundaries are inclusive. yfinance's `end` parameter is exclusive; with a dynamic latest-date setting, the downloader excludes the current session before the close and advances the bound after the close buffer. Prices use yfinance `Close` with `auto_adjust=True`; the same data is not adjusted again. See [docs/project_overview.md](docs/project_overview.md) and [docs/evaluation.md](docs/evaluation.md).

## API and dashboard

Start the backend and dashboard in separate activated terminals from the project root:

```powershell
uvicorn api.main:app --reload
streamlit run streamlit/app.py
```

API OpenAPI docs: `http://127.0.0.1:8000/docs`. The dashboard calls the API at `http://127.0.0.1:8000`; set `PORTFOLIO_API_URL` to override. Set `PORTFOLIO_MODEL_PATH` to load a different model artifact. If the model is unavailable, health reports `model_ready: false`, and prediction endpoints return a useful error.

## Tests

```powershell
python -m pytest -q
```

## Observed results

The saved 2024–2026 held-out experiment did not show PPO outperforming all baselines: risk-aware PPO Sharpe was 1.19 versus 1.26 equal weight, 1.21 buy-and-hold, and 1.32 momentum. See [the full result table and limitations](docs/experimental_results.md), the machine-readable [test report](results/backtests/test_report.json), and generated plots in `results/figures/`. These are actual results from the local cached data, not a performance promise.

![Held-out test equity curves](results/figures/equity_curves.png)

The four notebooks under `notebooks/` can be opened in JupyterLab (install it with `python -m pip install jupyterlab` if needed). They inspect cached data, causal features, a seeded environment episode, and generated evaluation output.

## Configuration and conventions

- Assets: SPY, QQQ, GLD, IEF, VNQ, CASH. Initial capital: $100,000; seed: 42.
- Long-only weights, no leverage, daily rebalance decisions.
- Cost: 0.1% × half-L1 change in target weights (one-way turnover).
- Synthetic cash return: configurable 2% annual, converted with 252 sessions.
- PPO starts with MlpPolicy, learning rate 0.0003, gamma 0.99, GAE lambda 0.95, clip 0.2, batch 64, 2048 rollout steps, entropy coefficient 0.01, value coefficient 0.5, CPU.
- Baselines and the agent share the same eligible dates and cost-aware backtester. Test data is held out from checkpoint selection.

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for project scope and [docs/reference_repositories.md](docs/reference_repositories.md) for reference-source inspection and code-reuse records.

Project-authored code is under the [MIT License](LICENSE). No third-party implementation code was copied.
