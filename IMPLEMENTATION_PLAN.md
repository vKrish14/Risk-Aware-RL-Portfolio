# Implementation plan and phase gates

This repository follows the user-approved nine-phase plan for the educational Risk-Aware RL Portfolio Management project.

1. **Project setup and technical specification** — Python environment, dependencies, configuration, setup documentation, and reference inspection. Stop for review after acceptance checks.
2. **Market-data pipeline** — yfinance acquisition, cache, cleaning, alignment, provenance, and chronological splits.
3. **Feature engineering** — causal features, train-only preprocessing, finite matrices, and leakage checks.
4. **Custom Gymnasium environment** — portfolio accounting, action mapping, costs, rewards, and contract checks.
5. **Local PPO training** — configurable training, validation checkpoints, logs, metadata, and reloadable inference.
6. **Backtesting and benchmarks** — cost-consistent strategies and held-out evaluation.
7. **FastAPI backend** — service layer, schemas, health, prediction, simulation, and backtest routes.
8. **Streamlit dashboard** — five pages backed by the API and saved evaluation outputs.
9. **Testing, documentation, and GitHub finalization** — comprehensive verification, final docs, attribution and repository review.

Complete each phase's acceptance checks before building on it. The user's current instruction authorizes continuing through all phases without asking for intermediate approval. Do not claim data downloads, test results, or model performance unless actually observed.
