# Architecture

- `src/data`: adjusted yfinance download, local cache, validation, common-date alignment, and date splits.
- `src/features`: causal rolling market features and a serializable scaler fitted on training rows only.
- `src/environment`: softmax allocation mapping, explicit transaction cost, reward variants, and Gymnasium portfolio accounting.
- `src/agents`, `src/training`: SB3 PPO model creation, deterministic validation callbacks, and checkpoint metadata.
- `src/baselines`, `src/evaluation`: baseline schedules and independent shared-cost backtests/metrics.
- `api`: FastAPI schemas/routes and a service that loads model/preprocessing artifacts once.
- `streamlit`: five presentation pages that call the FastAPI service.
- `configs`: versioned experiment settings. Generated data, models, and results remain local and ignored by Git.

The dashboard has no direct market download or model training path. API business logic is held in `ModelService` and shared evaluation modules rather than route handlers.
