"""Service health endpoint."""
from fastapi import APIRouter, Request
from src.evaluation.saved_report import DEFAULT_REPORT

router = APIRouter()


@router.get("/health")
def health(request: Request):
    service = getattr(request.app.state, "service", None)
    return {
        "status": "ok",
        "model_ready": service is not None,
        "model": str(service.model_path) if service else None,
        "saved_results_ready": DEFAULT_REPORT.is_file(),
    }
