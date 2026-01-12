"""
Sentinel Navigator Policy Engine

Rule-based decision engine that combines context metrics and ML scores
to produce final risk assessment decisions with aggregated anomaly vectors.
"""

from typing import Any, Dict, List, Tuple

from core.schemas.outputs import SentinelAnalysis, SentinelDecision


# Engine version for audit trail
_ENGINE_VERSION = "1.1.0"


class NavigatorPolicyEngine:
    """
    Rule-based policy engine for risk decisions with vector aggregation.
    
    Evaluates context metrics and ML anomaly scores to produce
    a final ALLOW/CHALLENGE/BLOCK decision with risk scoring.
    Aggregates anomaly vectors from context, keyboard, and mouse.
    
    Decision Logic:
        BLOCK: Impossible travel (geo_velocity > 500 mph) OR policy violation
        CHALLENGE: High ML score (> 0.8) OR new device
        ALLOW: Otherwise
    """
    
    # Thresholds
    _GEO_VELOCITY_THRESHOLD: float = 500.0  # mph
    _ML_SCORE_THRESHOLD: float = 0.8
    
    def evaluate(
        self,
        context_metrics: Dict[str, Any],
        biometric_data: Dict[str, Tuple[float, List[str]]]
    ) -> SentinelAnalysis:
        """
        Evaluate context metrics and biometric data to produce risk decision.
        
        Args:
            context_metrics: Dictionary of context processor metrics including:
                - geo_velocity_mph: Travel speed between locations
                - policy_violation_flag: 1.0 if policy violated, 0.0 otherwise
                - is_new_device: 1.0 if device is new, 0.0 if known
                - (other metrics from ContextProcessor)
            biometric_data: Dictionary with keys 'keyboard' and 'mouse' containing
                tuples of (anomaly_score, anomaly_vectors):
                - anomaly_score: float from 0.0 to 1.0
                - anomaly_vectors: List of feature drift tags (e.g., "dwell_mean_ms_high")
        
        Returns:
            SentinelAnalysis object with decision, risk_score, and aggregated anomaly_vectors.
        """
        anomaly_vectors: List[str] = []
        decision: SentinelDecision = SentinelDecision.ALLOW
        
        # Extract context metrics with defaults
        geo_velocity = float(context_metrics.get("geo_velocity_mph", 0.0))
        policy_violation = float(context_metrics.get("policy_violation_flag", 0.0))
        is_new_device = float(context_metrics.get("is_new_device", 0.0))
        
        # Extract biometric scores and vectors
        keyboard_score, keyboard_vectors = biometric_data.get("keyboard", (0.0, []))
        mouse_score, mouse_vectors = biometric_data.get("mouse", (0.0, []))
        
        # =================================================================
        # BLOCK Conditions (highest priority)
        # =================================================================
        
        # Check for impossible travel
        if geo_velocity > self._GEO_VELOCITY_THRESHOLD:
            anomaly_vectors.append("impossible_travel")
            decision = SentinelDecision.BLOCK
        
        # Check for policy violation
        if policy_violation == 1.0:
            anomaly_vectors.append("policy_violation")
            decision = SentinelDecision.BLOCK
        
        # =================================================================
        # CHALLENGE Conditions (if not already BLOCK)
        # =================================================================
        
        if decision != SentinelDecision.BLOCK:
            # Check keyboard anomaly
            if keyboard_score > self._ML_SCORE_THRESHOLD:
                anomaly_vectors.append("keystroke_anomaly")
                decision = SentinelDecision.CHALLENGE
            
            # Check mouse anomaly
            if mouse_score > self._ML_SCORE_THRESHOLD:
                anomaly_vectors.append("mouse_anomaly")
                decision = SentinelDecision.CHALLENGE
            
            # Check new device
            if is_new_device == 1.0:
                anomaly_vectors.append("new_device")
                decision = SentinelDecision.CHALLENGE
        
        # =================================================================
        # Aggregate Biometric Vectors
        # =================================================================
        
        # Add keyboard feature attribution vectors
        anomaly_vectors.extend(keyboard_vectors)
        
        # Add mouse feature attribution vectors
        anomaly_vectors.extend(mouse_vectors)
        
        # =================================================================
        # Calculate Risk Score
        # =================================================================
        
        # Risk score is the maximum of all input signals
        risk_score = max(
            geo_velocity / 1000.0,  # Normalize velocity (1000 mph = 1.0)
            policy_violation,
            is_new_device,
            keyboard_score,
            mouse_score
        )
        # Clamp to [0.0, 1.0]
        risk_score = min(max(risk_score, 0.0), 1.0)
        
        return SentinelAnalysis(
            decision=decision,
            risk_score=risk_score,
            engine_version=_ENGINE_VERSION,
            anomaly_vectors=anomaly_vectors
        )
