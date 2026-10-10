"""Interactive access to the saved, held-out multi-strategy comparison."""
from fastapi import APIRouter, HTTPException

from api.schemas import BacktestRunRequest
from src.evaluation.saved_report import (
    compare_saved_strategies,
    load_saved_report,
    report_availability,
    saved_strategy_series,
)


router = APIRouter(prefix="/backtest")


@router.get("/availability")
def availability():
    try:
        return report_availability(load_saved_report())
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/run")
def run_backtest(payload: BacktestRunRequest):
    """Slice any saved strategy and compare all strategies over identical dates."""
    try:
        report = load_saved_report()
        result = saved_strategy_series(payload.strategy, payload.start_date, payload.end_date, report=report)
        comparison = compare_saved_strategies(payload.start_date, payload.end_date, report=report)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {**result, "comparison": comparison}
