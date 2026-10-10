"""FastAPI app for educational portfolio-policy inference and backtesting."""
from __future__ import annotations

from fastapi import FastAPI

from api.model_service import ModelService
from api.routes.backtest import router as backtest_router
from api.routes.health import router as health_router
from api.routes.market import router as market_router
from api.routes.portfolio import router as portfolio_router
from src.evaluation.saved_report import compare_saved_strategies, load_saved_report, report_availability

app = FastAPI(title="Risk-Aware RL Portfolio API", version="0.1.0")
try:
    app.state.service = ModelService()
    app.state.model_error = None
except Exception as exc:
    app.state.service = None
    app.state.model_error = str(exc)

app.include_router(health_router)
app.include_router(portfolio_router)
app.include_router(backtest_router)
app.include_router(market_router)


@app.get("/portfolio/metrics")
def portfolio_metrics():
    try:
        report = load_saved_report()
        period = report_availability(report)
        return compare_saved_strategies(period["minimum"], period["maximum"], report=report)["risk_aware"]
    except (FileNotFoundError, KeyError, ValueError) as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"Saved test metrics unavailable: {exc}") from exc
