"""Small HTTP client for the local FastAPI service."""
from __future__ import annotations

import os

import httpx

BASE_URL = os.getenv("PORTFOLIO_API_URL", "http://127.0.0.1:8000")


def get_json(path: str) -> dict:
    response = httpx.get(f"{BASE_URL}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


def post_json(path: str, payload: dict) -> dict:
    response = httpx.post(f"{BASE_URL}{path}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()
