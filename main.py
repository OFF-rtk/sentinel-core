"""
Sentinel Orchestrator API

FastAPI application exposing:
- POST /stream/keyboard → 204 (no body)
- POST /stream/mouse → 204 (no body)
- POST /evaluate → JSON response

All endpoints implement rate limiting via Redis.
"""

from contextlib import asynccontextmanager
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from core.orchestrator import (
    SentinelOrchestrator,
    ReplayAttackError,
)
from core.schemas.inputs import (
    KeyboardStreamPayload,
    MouseStreamPayload,
    EvaluatePayload,
)
from core.schemas.outputs import EvaluateResponse
from persistence.session_repository import SessionRepository


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Application State
# =============================================================================

class AppState:
    """Application state container."""
    orchestrator: Optional[SentinelOrchestrator] = None
    repo: Optional[SessionRepository] = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Sentinel Orchestrator API...")
    state.repo = SessionRepository()
    state.orchestrator = SentinelOrchestrator(repo=state.repo)
    logger.info("Sentinel Orchestrator ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Sentinel Orchestrator API...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Sentinel Orchestrator",
    description="Behavioral biometrics security orchestrator",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


# =============================================================================
# Stream Endpoints (HTTP 204)
# =============================================================================

@app.post("/stream/keyboard", status_code=status.HTTP_204_NO_CONTENT)
async def stream_keyboard(payload: KeyboardStreamPayload):
    """
    Ingest keyboard biometric stream.
    
    - Validates batch_id for anti-replay
    - Updates behavioral state
    - Never returns security decisions
    """
    # Rate limiting
    if not state.repo.check_stream_rate_limit(payload.session_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded (max 20 batches/sec)"
        )
    
    try:
        state.orchestrator.process_keyboard_stream(payload)
    except ReplayAttackError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Keyboard stream error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error processing keyboard stream"
        )
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/stream/mouse", status_code=status.HTTP_204_NO_CONTENT)
async def stream_mouse(payload: MouseStreamPayload):
    """
    Ingest mouse biometric stream.
    
    - Validates batch_id for anti-replay
    - Updates behavioral state
    - Never returns security decisions
    """
    # Rate limiting
    if not state.repo.check_stream_rate_limit(payload.session_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded (max 20 batches/sec)"
        )
    
    try:
        state.orchestrator.process_mouse_stream(payload)
    except ReplayAttackError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Mouse stream error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error processing mouse stream"
        )
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# =============================================================================
# Evaluate Endpoint (JSON Response)
# =============================================================================

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(payload: EvaluatePayload):
    """
    Evaluate session risk and produce security decision.
    
    - Supports idempotency via eval_id
    - Returns ALLOW, CHALLENGE, or BLOCK
    - Implements weighted MAX fusion
    """
    # Rate limiting (10/sec for evaluate)
    if not state.repo.check_eval_rate_limit(payload.session_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded (max 10 evaluations/sec)"
        )
    
    try:
        result = state.orchestrator.evaluate(payload)
        return result
    except Exception as e:
        logger.error(f"Evaluate error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during evaluation"
        )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
