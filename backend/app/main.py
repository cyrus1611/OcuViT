"""
OcuViT — FastAPI application entry point.

Loads the trained Vision Transformer model once at startup and exposes
prediction and health endpoints.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes.prediction import router as prediction_router
from app.services.inference import model_service


# ---------------------------------------------------------------------------
# Lifespan — load model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ViT model before the first request, release on shutdown."""
    print("[Startup] Loading OcuViT model …")
    model_service.load()
    print("[Startup] Model ready.")
    yield
    print("[Shutdown] Cleaning up …")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="OcuViT — Ophthalmic Disease Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the Vite dev server
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static files — serve generated attention maps
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _BACKEND_DIR / "results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/results", StaticFiles(directory=str(_RESULTS_DIR)), name="results")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
app.include_router(prediction_router, prefix="/api")


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model_service.loaded,
    }
