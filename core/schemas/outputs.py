"""
Sentinel Core Output Schemas

This module defines Pydantic V2 models that strictly enforce
the SentinelResponse JSON contract for risk assessment outputs.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums
# =============================================================================

class SentinelDecision(str, Enum):
    """Risk assessment decision."""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    CHALLENGE = "CHALLENGE"


# =============================================================================
# Actor Context
# =============================================================================

class ActorContext(BaseModel):
    """User and session information for the evaluated action."""
    role: str = Field(..., description="User role (e.g., analyst, admin)")
    user_id: str = Field(..., description="Unique user identifier")
    session_id: str = Field(..., description="Active session identifier")
    session_age_seconds: int = Field(
        ..., 
        ge=0,
        description="Seconds since session start"
    )


# =============================================================================
# Action Context
# =============================================================================

class ActionContext(BaseModel):
    """Details of the action being evaluated."""
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Transaction details (amount, currency, recipient, etc.)"
    )
    service: str = Field(..., description="Service name")
    action_type: str = Field(..., description="Action type being performed")
    resource_target: str = Field(..., description="Target resource identifier")


# =============================================================================
# Network Context
# =============================================================================

class GeoLocation(BaseModel):
    """IP geolocation data."""
    asn: str = Field(..., description="Autonomous System Number and name")
    city: str = Field(..., description="City name")
    country: str = Field(..., description="Country code (ISO 3166-1 alpha-2)")


class ClientFingerprint(BaseModel):
    """Client device and TLS fingerprint data."""
    ja3_hash: str = Field(..., description="TLS fingerprint hash")
    device_id: str = Field(..., description="Unique device identifier")
    user_agent_raw: str = Field(..., description="Raw user agent string")


class NetworkContext(BaseModel):
    """Complete network context for the request."""
    ip_address: str = Field(..., description="Client IP address")
    geo_location: GeoLocation = Field(..., description="IP geolocation data")
    ip_reputation: str = Field(..., description="IP reputation category")
    client_fingerprint: ClientFingerprint = Field(..., description="Client fingerprint data")


# =============================================================================
# Sentinel Analysis
# =============================================================================

class SentinelAnalysis(BaseModel):
    """Risk analysis output from the Sentinel engine."""
    decision: SentinelDecision = Field(..., description="ALLOW or BLOCK decision")
    risk_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Risk score from 0.0 (safe) to 1.0 (high risk)"
    )
    engine_version: str = Field(..., description="Sentinel engine version")
    anomaly_vectors: List[str] = Field(
        default_factory=list,
        description="List of detected anomaly types"
    )


# =============================================================================
# Security Enforcement
# =============================================================================

class SecurityEnforcement(BaseModel):
    """Security enforcement actions and policy information."""
    mfa_status: str = Field(..., description="MFA verification status")
    policy_applied: str = Field(..., description="Applied security policy identifier")


# =============================================================================
# Sentinel Response (Root Model)
# =============================================================================

# SentinelResponse removed (unused legacy output contract)


# =============================================================================
# Simplified Evaluate Response (Per New Spec)
# =============================================================================

class EvaluateResponse(BaseModel):
    """
    Simplified response for /evaluate endpoint.
    
    Returns only the essential decision information:
    - decision: ALLOW, CHALLENGE, or BLOCK
    - risk: Final fused risk score (0.0 to 1.0)
    - mode: Current session mode (NORMAL or CHALLENGE)
    """
    decision: SentinelDecision = Field(..., description="Security decision")
    risk: float = Field(..., ge=0.0, le=1.0, description="Final risk score")
    mode: str = Field(..., description="Session mode (NORMAL or CHALLENGE)")

