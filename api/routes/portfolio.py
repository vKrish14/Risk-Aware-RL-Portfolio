"""Portfolio inference and cost-aware user-supplied simulation endpoints."""
from fastapi import APIRouter, HTTPException, Request

from api.schemas import PortfolioPredictRequest, PortfolioPredictResponse, PortfolioSimulateRequest
from src.evaluation.backtester import backtest_targets
import pandas as pd

router = APIRouter(prefix="/portfolio")


@router.post("/predict", response_model=PortfolioPredictResponse)
def predict(payload: PortfolioPredictRequest, request: Request):
    service = request.app.state.service
    if service is None:
        raise HTTPException(status_code=503, detail="Trained model is unavailable.")
    try:
        weights = service.predict(payload)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PortfolioPredictResponse(as_of_date=payload.as_of_date, weights=weights)


@router.post("/simulate")
def simulate(payload: PortfolioSimulateRequest):
    returns = pd.DataFrame([step.returns for step in payload.steps], index=[step.date for step in payload.steps])
    targets = pd.DataFrame([[step.target_weights.get(name, float("nan")) for name in returns.columns] for step in payload.steps], index=returns.index, columns=returns.columns)
    try:
        result = backtest_targets(
            returns,
            targets,
            initial_capital=payload.initial_capital,
            transaction_cost_rate=payload.transaction_cost_rate,
            risk_free_annual=payload.risk_free_annual,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    equity = result["equity"]
    weights = result["weights"]
    return {
        "metrics": result["metrics"],
        "equity": {date.strftime("%Y-%m-%d"): float(value) for date, value in equity.items()},
        "weights": {
            date.strftime("%Y-%m-%d"): {asset: float(value) for asset, value in row.items()}
            for date, row in weights.iterrows()
        },
    }
