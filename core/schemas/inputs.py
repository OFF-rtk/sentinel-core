"""
Sentinel Core Input Schemas - Decoupled Ingestion Pattern

This module defines Pydantic V2 models for:
- Async biometric stream payloads (KeystrokePayload, MousePayload)
- Sync evaluation request (EvaluationRequest)
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class KeyEventType(str, Enum):
    """Keyboard event type for dwell/flight time calculation."""
    DOWN = "DOWN"
    UP = "UP"


class MouseEventType(str, Enum):
    """Mouse event type for movement/click tracking."""
    MOVE = "MOVE"
    CLICK = "CLICK"


# =============================================================================
# Biometric Event Models
# =============================================================================

class KeyboardEvent(BaseModel):
    """Single keyboard event captured by the client wrapper."""
    key: str = Field(..., description="Key code or character pressed")
    event_type: KeyEventType = Field(..., description="DOWN or UP event")
    timestamp: float = Field(..., description="Event timestamp in milliseconds")


class MouseEvent(BaseModel):
    """Single mouse event captured by the client wrapper."""
    x: int = Field(..., description="X coordinate on screen")
    y: int = Field(..., description="Y coordinate on screen")
    event_type: MouseEventType = Field(..., description="MOVE or CLICK event")
    timestamp: float = Field(..., description="Event timestamp in milliseconds")


# =============================================================================
# Async Biometric Stream Payloads
# =============================================================================

class KeystrokePayload(BaseModel):
    """
    Async payload for keystroke stream ingestion.
    Sent periodically by the client during a session.
    """
    session_id: str = Field(..., description="Active session identifier")
    sequence_id: int = Field(..., description="Sequence number for ordering batches")
    events: List[KeyboardEvent] = Field(..., description="Batch of keyboard events")


class MousePayload(BaseModel):
    """
    Async payload for mouse movement/click stream ingestion.
    Sent periodically by the client during a session.
    """
    session_id: str = Field(..., description="Active session identifier")
    sequence_id: int = Field(..., description="Sequence number for ordering batches")
    events: List[MouseEvent] = Field(..., description="Batch of mouse events")


# =============================================================================
# Sync Evaluation Request Context Models
# =============================================================================

class UserSessionContext(BaseModel):
    """
    User and session context for evaluation.
    Maps to 'actor' and 'security_enforcement' in output.
    """
    user_id: str = Field(..., description="Unique user identifier")
    session_id: str = Field(..., description="Active session identifier")
    role: str = Field(..., description="User role (e.g., analyst, admin)")
    session_start_time: datetime = Field(
        ..., 
        description="Session start time for calculating session_age_seconds"
    )
    mfa_status: str = Field(..., description="MFA verification status")


class BusinessContext(BaseModel):
    """
    Business action context for evaluation.
    Maps to 'action_context' in output.
    """
    service: str = Field(..., description="Service name (e.g., card_service)")
    action_type: str = Field(..., description="Action being performed (e.g., card_activation)")
    resource_target: str = Field(..., description="Target resource identifier")
    transaction_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Transaction details (amount, currency, recipient, etc.)"
    )


class ClientFingerprint(BaseModel):
    """Client device and TLS fingerprint data for input."""
    device_id: str = Field(..., description="Unique device identifier")
    ja3_hash: Optional[str] = Field(None, description="TLS fingerprint hash")


class ClientNetworkContext(BaseModel):
    """
    Network context captured by API gateway.
    Provides data for 'network_context' in output.
    """
    ip_address: str = Field(..., description="Client IP address")
    user_agent: str = Field(..., description="Raw user agent string")
    ja3_hash: Optional[str] = Field(
        None, 
        description="TLS fingerprint hash from gateway (deprecated, use client_fingerprint)"
    )
    client_fingerprint: Optional[ClientFingerprint] = Field(
        None,
        description="Client device fingerprint with device_id"
    )


# =============================================================================
# Sync Evaluation Request (Root Model)
# =============================================================================

class EvaluationRequest(BaseModel):
    """
    Synchronous evaluation request payload.
    Submitted when a high-risk action requires real-time risk assessment.
    """
    user_session: UserSessionContext = Field(..., description="User and session context")
    business_context: BusinessContext = Field(..., description="Business action context")
    network_context: ClientNetworkContext = Field(..., description="Network/client context")
