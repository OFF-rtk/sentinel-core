"""
Sentinel Core Schemas

Public exports for input and output Pydantic models.
"""

# Input schemas - Async biometric payloads
from core.schemas.inputs import (
    KeyboardEvent,
    KeyEventType,
    KeystrokePayload,
    MouseEvent,
    MouseEventType,
    MousePayload,
)

# Input schemas - Sync evaluation request
from core.schemas.inputs import (
    BusinessContext,
    ClientNetworkContext,
    EvaluationRequest,
    UserSessionContext,
)

# Input schemas - New orchestrator API
from core.schemas.inputs import (
    KeyboardStreamPayload,
    MouseStreamPayload,
    RequestContext,
    EvaluatePayload,
)

# Output schemas
from core.schemas.outputs import (
    ActionContext,
    ActorContext,
    ClientFingerprint,
    GeoLocation,
    NetworkContext,
    SecurityEnforcement,
    SentinelAnalysis,
    SentinelDecision,
    EvaluateResponse,
)

__all__ = [
    # Input - Events
    "KeyEventType",
    "MouseEventType",
    "KeyboardEvent",
    "MouseEvent",
    # Input - Async Payloads (Legacy)
    "KeystrokePayload",
    "MousePayload",
    # Input - Sync Evaluation (Legacy)
    "UserSessionContext",
    "BusinessContext",
    "ClientNetworkContext",
    "EvaluationRequest",
    # Input - New Orchestrator API
    "KeyboardStreamPayload",
    "MouseStreamPayload",
    "RequestContext",
    "EvaluatePayload",
    # Output
    "SentinelDecision",
    "ActorContext",
    "ActionContext",
    "GeoLocation",
    "ClientFingerprint",
    "NetworkContext",
    "SentinelAnalysis",
    "SecurityEnforcement",
    "EvaluateResponse",
]
