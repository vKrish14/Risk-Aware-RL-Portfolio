"""FastAPI schema, health, and offline endpoint tests."""
from fastapi.testclient import TestClient

from api.main import app


def test_openapi_and_health_are_available_without_model() -> None:
    previous = app.state.service
    try:
        app.state.service = None
        client = TestClient(app)
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/health").json()["model_ready"] is False
        response = client.post("/portfolio/predict", json={"as_of_date": "2024-01-01", "features": {}})
        assert response.status_code == 503
    finally:
        app.state.service = previous


def test_simulation_endpoint_validates_and_returns_metrics() -> None:
    client = TestClient(app)
    response = client.post("/portfolio/simulate", json={
        "initial_capital": 1000,
        "transaction_cost_rate": .001,
        "steps": [
            {"date": "2024-01-01", "returns": {"SPY": 0, "CASH": 0}, "target_weights": {"SPY": 1, "CASH": 0}},
            {"date": "2024-01-02", "returns": {"SPY": .01, "CASH": .0001}, "target_weights": {"SPY": 1, "CASH": 0}},
        ],
    })
    assert response.status_code == 200
    assert response.json()["metrics"]["final_value"] > 1000


def test_prediction_endpoint_returns_valid_weights_from_service() -> None:
    class ReadyService:
        model_path = "test-model.zip"

        @staticmethod
        def predict(_payload):
            return {"SPY": .2, "QQQ": .2, "GLD": .2, "IEF": .2, "VNQ": .1, "CASH": .1}

    old = app.state.service
    try:
        app.state.service = ReadyService()
        response = TestClient(app).post("/portfolio/predict", json={"as_of_date": "2024-01-01", "features": {"x": 1}})
        assert response.status_code == 200
        weights = response.json()["weights"]
        assert all(value >= 0 for value in weights.values())
        assert sum(weights.values()) == pytest.approx(1)
    finally:
        app.state.service = old


def test_invalid_backtest_dates_receive_validation_error() -> None:
    response = TestClient(app).post("/backtest/run", json={"start_date": "2024-03-01", "end_date": "2024-01-01"})
    assert response.status_code == 422


def test_market_search_and_quote_endpoints(monkeypatch) -> None:
    import api.routes.market as market
    monkeypatch.setattr(market, "search_symbols", lambda query: [{"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", "kind": "Equity"}])
    monkeypatch.setattr(market, "get_quote", lambda symbol: {"symbol": symbol, "price": 100.0, "retrieved_at_utc": "2026-01-01T00:00:00+00:00"})
    client = TestClient(app)
    result = client.get("/market/search", params={"q": "Apple"})
    assert result.status_code == 200
    assert result.json()["results"][0]["symbol"] == "AAPL"
    quote = client.get("/market/quote", params={"symbol": "AAPL"})
    assert quote.status_code == 200
    assert quote.json()["price"] == 100.0


import pytest
