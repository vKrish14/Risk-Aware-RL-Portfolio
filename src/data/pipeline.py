"""Configured end-to-end price acquisition, alignment, splitting, and provenance."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yaml

from src.data.cleaner import align_common_dates
from src.data.downloader import load_or_download
from src.data.splitter import DateWindow, split_chronologically


def run_data_pipeline(root: str | Path = ".", *, force_download: bool = False) -> dict[str, object]:
    """Build reproducible split files from project YAML settings."""
    project = Path(root).resolve()
    config = yaml.safe_load((project / "configs/config.yaml").read_text(encoding="utf-8"))
    assets = yaml.safe_load((project / "configs/assets.yaml").read_text(encoding="utf-8"))
    settings = config["project"]
    inclusive_end = settings.get("end_date_inclusive")
    exclusive_end = (pd.Timestamp(inclusive_end).date() + timedelta(days=1)).isoformat() if inclusive_end else None
    result = load_or_download(
        assets["risky_assets"], settings["start_date"], exclusive_end,
        project / "data/cache", force_download=force_download,
    )
    aligned, report = align_common_dates(result.prices)
    ranges = settings["train"], settings["validation"], settings["test"]
    splits = split_chronologically(aligned, *(DateWindow(item["start"], item.get("end")) for item in ranges))
    processed = project / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(processed / f"{name}_prices.csv", index_label="date", float_format="%.12g")
    metadata = {
        **result.metadata,
        "pipeline_generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_yfinance_version": __import__("yfinance").__version__,
        "cache_hit": result.from_cache,
        "split_windows_inclusive": {name: window for name, window in zip(("train", "validation", "test"), ranges)},
        "alignment": {
            "input_rows": report.input_rows,
            "output_rows": report.output_rows,
            "dropped_dates": list(report.dropped_dates),
            "missing_by_asset": report.missing_by_asset,
        },
        "splits": {name: {"rows": len(frame), "first": frame.index.min().date().isoformat(), "last": frame.index.max().date().isoformat()} for name, frame in splits.items()},
    }
    (processed / "data_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"prices": aligned, "splits": splits, "metadata": metadata}


if __name__ == "__main__":
    outcome = run_data_pipeline(force_download="--refresh" in __import__("sys").argv)
    print(json.dumps(outcome["metadata"], indent=2))
