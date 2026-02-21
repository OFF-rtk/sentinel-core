"""
Sentinel Navigator Policy Engine

Pure business logic for risk assessment decisions.
This module is STATELESS and DETERMINISTIC.

No ML. No external calls. Just rules.
"""

from typing import Dict, List

from core.schemas.outputs import SentinelAnalysis, SentinelDecision


# =============================================================================
# Engine Configuration
# =============================================================================

_ENGINE_VERSION = "2.0.0"


class NavigatorPolicyEngine:
    """
    Stateless, deterministic policy engine for risk decisions.
    
    Decision Logic:
        BLOCK: risk_score >= 0.85
        CHALLENGE: risk_score >= 0.50
        ALLOW: otherwise
    
    Anomaly Vectors:
        - "impossible_travel" if velocity > MAX_VELOCITY
        - "infra_mismatch" if device_ip_mismatch == 1.0
        - "policy_violation" if policy_violation == 1.0
    
    Risk Score:
        max(velocity_risk, infra_risk, policy_risk, device_risk)
    """
    
    # Thresholds (STRICT)
    MAX_VELOCITY: float = 500.0  # mph
    BLOCK_THRESHOLD: float = 0.85
    CHALLENGE_THRESHOLD: float = 0.50
    
    def evaluate(self, metrics: Dict[str, float]) -> SentinelAnalysis:
        """
        Evaluate context metrics and produce risk decision.
        
        Args:
            metrics: Dictionary of context metrics including:
                - geo_velocity_mph: Travel speed between locations
                - device_ip_mismatch: 1.0 if desktop + VPN/hosting
                - policy_violation: 1.0 if role violates access
                - is_new_device: 1.0 if device is unknown
                - ip_reputation: 0.0-1.0 reputation score
                - simultaneous_sessions: Active session count
                - time_since_last_seen: Seconds since last activity
        
        Returns:
            SentinelAnalysis with decision, risk_score, and anomaly_vectors.
        """
        anomaly_vectors: List[str] = []
        
        # Extract metrics with safe defaults
        geo_velocity = float(metrics.get("geo_velocity_mph", 0.0))
        device_ip_mismatch = float(metrics.get("device_ip_mismatch", 0.0))
        policy_violation = float(metrics.get("policy_violation", 0.0))
        is_new_device = float(metrics.get("is_new_device", 0.0))
        
        # =================================================================
        # Anomaly Vector Detection
        # =================================================================
        
        # Impossible travel detection
        if geo_velocity > self.MAX_VELOCITY:
            anomaly_vectors.append("impossible_travel")
        
        # Infrastructure mismatch detection
        if device_ip_mismatch == 1.0:
            anomaly_vectors.append("infra_mismatch")
        
        # Policy violation detection
        if policy_violation == 1.0:
            anomaly_vectors.append("policy_violation")
        
        # Unknown user-agent detection (bot/script/automation)
        is_unknown_ua = float(metrics.get("is_unknown_user_agent", 0.0))
        if is_unknown_ua == 1.0:
            anomaly_vectors.append("unknown_user_agent")
        
        # =================================================================
        # Calculate Risk Score
        # =================================================================
        
        # Velocity risk: normalize to 0-1 (500 mph = 1.0)
        velocity_risk = min(geo_velocity / self.MAX_VELOCITY, 1.0)
        
        # Infrastructure risk: direct value
        infra_risk = device_ip_mismatch
        
        # Policy risk: direct value
        policy_risk = policy_violation
        
        # Device risk: direct value
        device_risk = is_new_device * 0.5  # New device is 50% risk factor
        
        # Note: unknown_user_agent only emits an anomaly vector for audit context.
        # It does NOT inflate the risk score — could just be a niche browser.
        
        # Final risk score: maximum of all risk components
        risk_score = max(
            velocity_risk,
            infra_risk,
            policy_risk,
            device_risk
        )
        
        # Clamp to [0.0, 1.0]
        risk_score = min(max(risk_score, 0.0), 1.0)
        
        # =================================================================
        # Decision Logic (STRICT THRESHOLDS)
        # =================================================================
        
        if risk_score >= self.BLOCK_THRESHOLD:
            decision = SentinelDecision.BLOCK
        elif risk_score >= self.CHALLENGE_THRESHOLD:
            decision = SentinelDecision.CHALLENGE
        else:
            decision = SentinelDecision.ALLOW
        
        return SentinelAnalysis(
            decision=decision,
            risk_score=risk_score,
            engine_version=_ENGINE_VERSION,
            anomaly_vectors=anomaly_vectors
        )


# =============================================================================
# Session-Level Strike System (MANDATORY)
# =============================================================================

class MouseSessionTracker:
    """
    Tracks bot/human strokes within a session using a strike system.
    
    Rules:
        - Bot stroke → strikes += 1
        - Human stroke → strikes = max(0, strikes - 1)
        - Session flagged if strikes >= 3
    """
    
    STRIKE_THRESHOLD: int = 3
    
    def __init__(self) -> None:
        """Initialize with zero strikes."""
        self.strikes: int = 0
        self.flagged: bool = False
    
    def record_bot_stroke(self) -> None:
        """Record a bot-like stroke, incrementing strikes."""
        self.strikes += 1
        self._update_flag()
    
    def record_human_stroke(self) -> None:
        """Record a human-like stroke, decrementing strikes (min 0)."""
        self.strikes = max(0, self.strikes - 1)
        self._update_flag()
    
    def _update_flag(self) -> None:
        """Update flagged status based on strike count."""
        if self.strikes >= self.STRIKE_THRESHOLD:
            self.flagged = True
    
    def is_flagged(self) -> bool:
        """Check if session is flagged as suspicious."""
        return self.flagged
    
    def get_strikes(self) -> int:
        """Get current strike count."""
        return self.strikes
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.strikes = 0
        self.flagged = False
