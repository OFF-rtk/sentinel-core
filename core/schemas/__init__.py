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
    SentinelResponse,
)

__all__ = [
    # Input - Events
    "KeyEventType",
    "MouseEventType",
    "KeyboardEvent",
    "MouseEvent",
    # Input - Async Payloads
    "KeystrokePayload",
    "MousePayload",
    # Input - Sync Evaluation
    "UserSessionContext",
    "BusinessContext",
    "ClientNetworkContext",
    "EvaluationRequest",
    # Output
    "SentinelDecision",
    "ActorContext",
    "ActionContext",
    "GeoLocation",
    "ClientFingerprint",
    "NetworkContext",
    "SentinelAnalysis",
    "SecurityEnforcement",
    "SentinelResponse",
]
