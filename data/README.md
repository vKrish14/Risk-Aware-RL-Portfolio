# Generated data

Raw provider downloads, processed datasets, and caches are generated locally and are not committed. Run `python -m src.data.pipeline` from the repository root to download or reuse cached adjusted prices and produce chronological split CSVs plus a provenance manifest. Run `python -m src.data.pipeline --refresh` to request fresh prices from Yahoo Finance. See `docs/project_overview.md` for date, adjustment, alignment, and cost conventions.
