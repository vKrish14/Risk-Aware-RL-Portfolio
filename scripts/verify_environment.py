"""Verify configured dependencies and YAML files in the active environment."""
from __future__ import annotations

import importlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("numpy", "pandas", "yfinance", "gymnasium", "stable_baselines3", "torch", "fastapi", "streamlit", "pydantic")
CONFIGS = ("config.yaml", "assets.yaml", "ppo_config.yaml")


def main() -> int:
    """Print environment checks and return a nonzero status on failure."""
    failures: list[str] = []
    for name in MODULES:
        try:
            module = importlib.import_module(name)
            print(f"OK {name} {getattr(module, '__version__', 'version unknown')}")
        except Exception as exc:  # report optional binary/runtime import issues clearly
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    for name in CONFIGS:
        path = ROOT / "configs" / name
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("expected a YAML mapping at document root")
            print(f"OK configs/{name}")
        except Exception as exc:
            failures.append(f"configs/{name}: {type(exc).__name__}: {exc}")
    if failures:
        print("FAIL " + "\nFAIL ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
