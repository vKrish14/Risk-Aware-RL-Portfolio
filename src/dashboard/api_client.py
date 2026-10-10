"""Small HTTP client for the local FastAPI service."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

BASE_URL = os.getenv("PORTFOLIO_API_URL", "http://127.0.0.1:8000")
_API_PROCESS: subprocess.Popen | None = None
_API_START_LOCK = threading.Lock()


def get_json(path: str) -> dict:
    return _request("GET", path)


def post_json(path: str, payload: dict) -> dict:
    return _request("POST", path, json=payload, timeout=30)


def _request(method: str, path: str, **kwargs) -> dict:
    """Return decoded API JSON or raise an actionable message for the UI."""
    timeout = kwargs.pop("timeout", 10)
    try:
        response = httpx.request(method, f"{BASE_URL}{path}", timeout=timeout, **kwargs)
    except httpx.RequestError as exc:
        if _is_loopback_api():
            _ensure_loopback_api()
            try:
                response = httpx.request(method, f"{BASE_URL}{path}", timeout=timeout, **kwargs)
            except httpx.RequestError as retry_exc:
                raise RuntimeError(
                    f"Cannot reach the portfolio API at {BASE_URL}: {retry_exc}"
                ) from retry_exc
        else:
            raise RuntimeError(f"Cannot reach the portfolio API at {BASE_URL}: {exc}") from exc
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API returned HTTP {response.status_code}: {detail}") from exc
    return response.json()


def _is_loopback_api() -> bool:
    """Return whether the configured API URL points to this machine."""
    parsed = urlsplit(BASE_URL)
    return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}


def _ensure_loopback_api() -> None:
    """Start the local FastAPI sidecar if the Streamlit host has not started it."""
    global _API_PROCESS
    parsed = urlsplit(BASE_URL)
    port = parsed.port or 8000
    with _API_START_LOCK:
        if _API_PROCESS is None or _API_PROCESS.poll() is not None:
            project_root = Path(__file__).resolve().parents[2]
            child_env = os.environ.copy()
            python_path = child_env.get("PYTHONPATH", "")
            child_env["PYTHONPATH"] = os.pathsep.join(
                part for part in (str(project_root), python_path) if part
            )
            _API_PROCESS = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "api.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--log-level",
                    "warning",
                ],
                cwd=project_root,
                env=child_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        deadline = time.monotonic() + 60
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if _API_PROCESS.poll() is not None:
                raise RuntimeError(
                    "The local portfolio API exited before it became ready. "
                    "Check that model and report artifacts are included."
                )
            try:
                response = httpx.get(f"{BASE_URL}/health", timeout=1)
                response.raise_for_status()
                return
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                last_error = exc
                time.sleep(0.25)
        raise RuntimeError(
            f"The local portfolio API did not become ready at {BASE_URL} within 60 seconds: "
            f"{last_error}"
        ) from last_error
