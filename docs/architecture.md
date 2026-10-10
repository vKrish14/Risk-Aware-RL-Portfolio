# Architecture

- `src/data`: adjusted yfinance download, local cache, validation, common-date alignment, and date splits.
- `src/features`: causal rolling market features and a serializable scaler fitted on training rows only.
- `src/environment`: softmax allocation mapping, explicit transaction cost, reward variants, and Gymnasium portfolio accounting.
- `src/agents`, `src/training`: SB3 PPO model creation, deterministic validation callbacks, and checkpoint metadata.
- `src/baselines`, `src/evaluation`: baseline schedules and independent shared-cost backtests/metrics.
- `api`: FastAPI schemas/routes and a service that loads model/preprocessing artifacts once.
- `streamlit`: strategy comparison, PPO state inputs, model analysis, and a user-return what-if simulator backed by FastAPI.
- `configs`: versioned experiment settings. Generated data, models, and results remain local and ignored by Git.

The dashboard has no direct market download or model training path. Historical strategy comparisons read the saved test report, so those views work without a model checkpoint. The what-if simulator accepts uploaded daily returns and chosen target weights. PPO inference still requires the saved model and preprocessing metadata. API business logic is held in `ModelService` and shared evaluation modules rather than route handlers.
