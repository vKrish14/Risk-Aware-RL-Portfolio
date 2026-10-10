"""FastAPI app for educational portfolio-policy inference and backtesting."""
from __future__ import annotations

from fastapi import FastAPI

from api.model_service import ModelService
from api.routes.backtest import router as backtest_router
from api.routes.health import router as health_router
from api.routes.portfolio import router as portfolio_router

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


@app.get("/portfolio/metrics")
def portfolio_metrics():
    if app.state.service is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"Model unavailable: {app.state.model_error}")
    result = app.state.service.run_saved_test()
    return result["metrics"]
