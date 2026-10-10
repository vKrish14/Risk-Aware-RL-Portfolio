"""Saved model backtest endpoints."""
from fastapi import APIRouter, HTTPException, Request

from api.schemas import BacktestRunRequest
from src.evaluation.metrics import calculate_metrics

router = APIRouter(prefix="/backtest")


@router.get("/availability")
def availability(request: Request):
    service = request.app.state.service
    if service is None:
        raise HTTPException(status_code=503, detail="Trained model is unavailable.")
    try:
        return service.test_availability()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/run")
def run_backtest(payload: BacktestRunRequest, request: Request):
    service = request.app.state.service
    if service is None:
        raise HTTPException(status_code=503, detail="Trained model is unavailable.")
    try:
        result = service.run_saved_test()
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    equity = result["equity"].loc[str(payload.start_date):str(payload.end_date)]
    if equity.empty:
        raise HTTPException(status_code=422, detail="Requested range has no available saved test observations.")
    prior = result["equity"].loc[result["equity"].index < equity.index.min()]
    previous = prior.iloc[-1] if not prior.empty else service.config["project"]["initial_capital"]
    net_returns = equity.pct_change()
    net_returns.iloc[0] = equity.iloc[0] / previous - 1
    turnover = result["turnover"].reindex(equity.index)
    costs = result["costs"].reindex(equity.index)
    metrics = calculate_metrics(
        net_returns, turnover=turnover, transaction_cost_amount=costs,
        initial_capital=float(previous), risk_free_annual=service.config["portfolio"]["risk_free_annual"],
    )
    weights = result["weights"].loc[equity.index]
    try:
        comparison = service.saved_comparison(payload.start_date, payload.end_date)
    except FileNotFoundError:
        comparison = {}
    return {
        "metrics": metrics,
        "comparison": comparison,
        "equity": {d.strftime("%Y-%m-%d"): float(v) for d, v in equity.items()},
        "weights": {d.strftime("%Y-%m-%d"): row.to_dict() for d, row in weights.iterrows()},
    }
