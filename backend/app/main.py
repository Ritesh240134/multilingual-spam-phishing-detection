"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.inference import router as inference_router


app = FastAPI(
    title="Multilingual Spam and Phishing Email Detection API",
    version="0.1.0",
    description="Backend API foundation for multilingual email threat detection.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Return the service health status."""
    return {"status": "ok", "message": "API is running"}


app.include_router(inference_router)
