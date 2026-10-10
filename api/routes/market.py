"""Search public market symbols and return educational quote snapshots."""
from fastapi import APIRouter, HTTPException, Query

from src.data.stock_quotes import get_quote, search_symbols

router = APIRouter(prefix="/market", tags=["market data"])


@router.get("/search")
def market_search(q: str = Query(min_length=1, max_length=80)):
    try:
        return {"results": search_symbols(q)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Yahoo Finance search unavailable: {exc}") from exc


@router.get("/quote")
def market_quote(symbol: str = Query(min_length=1, max_length=20)):
    try:
        return get_quote(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Yahoo Finance quote unavailable: {exc}") from exc
