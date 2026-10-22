"""Email inference route."""

from fastapi import APIRouter

from backend.app.schemas.inference import AnalyzeRequest, AnalyzeResponse
from backend.app.services.classifier_service import classifier_service


router = APIRouter(prefix="/api/v1", tags=["inference"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_email(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze an email using locally configured model artifacts."""

    return classifier_service.analyze(request)
