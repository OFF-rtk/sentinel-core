"""
Navigator Policy Engine Unit Tests

Tests for NavigatorPolicyEngine with deterministic risk decisions.
Validates BLOCK/CHALLENGE/ALLOW thresholds, anomaly vector detection,
and risk score calculation.
"""

import pytest

from core.models.navigator import NavigatorPolicyEngine, MouseSessionTracker
from core.schemas.outputs import SentinelDecision


# =============================================================================
# Metric Generators (Inline)
# =============================================================================

def generate_normal_metrics():
    """Normal user: low risk, should ALLOW."""
    return {
        "geo_velocity_mph": 30.0,        # Normal driving speed
        "device_ip_mismatch": 0.0,
        "policy_violation": 0.0,
        "is_new_device": 0.0,
        "ip_reputation": 0.8,
        "simultaneous_sessions": 1,
        "time_since_last_seen": 300.0,
    }


def generate_impossible_travel_metrics():
    """Impossible travel: teleportation speed, should BLOCK."""
    return {
        "geo_velocity_mph": 1000.0,      # > 500 mph = impossible
        "device_ip_mismatch": 0.0,
        "policy_violation": 0.0,
        "is_new_device": 0.0,
    }


def generate_infra_mismatch_metrics():
    """Infrastructure mismatch: desktop + VPN/hosting IP, should flag."""
    return {
        "geo_velocity_mph": 0.0,
        "device_ip_mismatch": 1.0,       # Desktop + VPN detected
        "policy_violation": 0.0,
        "is_new_device": 0.0,
    }


def generate_policy_violation_metrics():
    """Policy violation: role doesn't have access, should BLOCK."""
    return {
        "geo_velocity_mph": 0.0,
        "device_ip_mismatch": 0.0,
        "policy_violation": 1.0,         # Access policy violated
        "is_new_device": 0.0,
    }


def generate_new_device_metrics():
    """New device: moderate risk, should CHALLENGE."""
    return {
        "geo_velocity_mph": 0.0,
        "device_ip_mismatch": 0.0,
        "policy_violation": 0.0,
        "is_new_device": 1.0,            # Unknown device
    }


# =============================================================================
# Decision Threshold Tests
# =============================================================================

class TestDecisionThresholds:
    """Test BLOCK/CHALLENGE/ALLOW threshold logic."""
    
    def test_allow_low_risk(self, navigator_engine):
        """Risk score < 0.50 should result in ALLOW."""
        metrics = generate_normal_metrics()
        
        result = navigator_engine.evaluate(metrics)
        
        assert result.decision == SentinelDecision.ALLOW
        assert result.risk_score < 0.50
    
    def test_challenge_medium_risk(self, navigator_engine):
        """Risk score 0.50-0.84 should result in CHALLENGE."""
        metrics = generate_new_device_metrics()  # is_new_device → 0.5 risk
        
        result = navigator_engine.evaluate(metrics)
        
        assert result.decision == SentinelDecision.CHALLENGE
        assert 0.50 <= result.risk_score < 0.85
    
    def test_block_high_risk(self, navigator_engine):
        """Risk score >= 0.85 should result in BLOCK."""
        metrics = generate_impossible_travel_metrics()  # 1000 mph → 1.0 risk
        
        result = navigator_engine.evaluate(metrics)
        
        assert result.decision == SentinelDecision.BLOCK
        assert result.risk_score >= 0.85
    
    def test_block_policy_violation(self, navigator_engine):
        """Policy violation should result in BLOCK."""
        metrics = generate_policy_violation_metrics()
        
        result = navigator_engine.evaluate(metrics)
        
        assert result.decision == SentinelDecision.BLOCK
        assert result.risk_score == 1.0  # Direct: policy_violation = 1.0


# =============================================================================
# Anomaly Vector Tests
# =============================================================================

class TestAnomalyVectors:
    """Test anomaly vector detection."""
    
    def test_impossible_travel_vector(self, navigator_engine):
        """Velocity > 500 mph should add 'impossible_travel' vector."""
        metrics = generate_impossible_travel_metrics()
        
        result = navigator_engine.evaluate(metrics)
        
        assert "impossible_travel" in result.anomaly_vectors
    
    def test_infra_mismatch_vector(self, navigator_engine):
        """device_ip_mismatch = 1.0 should add 'infra_mismatch' vector."""
        metrics = generate_infra_mismatch_metrics()
        
        result = navigator_engine.evaluate(metrics)
        
        assert "infra_mismatch" in result.anomaly_vectors
    
    def test_policy_violation_vector(self, navigator_engine):
        """policy_violation = 1.0 should add 'policy_violation' vector."""
        metrics = generate_policy_violation_metrics()
        
        result = navigator_engine.evaluate(metrics)
        
        assert "policy_violation" in result.anomaly_vectors
    
    def test_no_vectors_for_normal(self, navigator_engine):
        """Normal metrics should produce no anomaly vectors."""
        metrics = generate_normal_metrics()
        
        result = navigator_engine.evaluate(metrics)
        
        assert len(result.anomaly_vectors) == 0


# =============================================================================
# Risk Score Calculation Tests
# =============================================================================

class TestRiskScoreCalculation:
    """Test risk score calculation logic."""
    
    def test_velocity_risk_normalization(self, navigator_engine):
        """Velocity risk should be normalized: velocity/500."""
        metrics = generate_normal_metrics()
        metrics["geo_velocity_mph"] = 250.0  # 250/500 = 0.5
        
        result = navigator_engine.evaluate(metrics)
        
        # Risk should be 0.5 (velocity dominates)
        assert 0.45 <= result.risk_score <= 0.55
    
    def test_velocity_risk_capped(self, navigator_engine):
        """Velocity risk should cap at 1.0 even for extreme speeds."""
        metrics = generate_normal_metrics()
        metrics["geo_velocity_mph"] = 10000.0  # Way over 500
        
        result = navigator_engine.evaluate(metrics)
        
        assert result.risk_score == 1.0
    
    def test_new_device_half_risk(self, navigator_engine):
        """New device should contribute 0.5 risk factor."""
        metrics = generate_normal_metrics()
        metrics["is_new_device"] = 1.0
        
        result = navigator_engine.evaluate(metrics)
        
        # Risk should be 0.5 from new device alone
        assert result.risk_score == 0.5
    
    def test_max_of_all_risks(self, navigator_engine):
        """Final risk should be max of all components."""
        metrics = {
            "geo_velocity_mph": 100.0,       # 100/500 = 0.2
            "device_ip_mismatch": 0.3,       # 0.3
            "policy_violation": 0.0,         # 0.0
            "is_new_device": 1.0,            # 0.5
        }
        
        result = navigator_engine.evaluate(metrics)
        
        # Max should be 0.5 (from new device)
        assert result.risk_score == 0.5


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_metrics(self, navigator_engine):
        """Empty metrics should result in ALLOW with 0.0 risk."""
        result = navigator_engine.evaluate({})
        
        assert result.decision == SentinelDecision.ALLOW
        assert result.risk_score == 0.0
        assert len(result.anomaly_vectors) == 0
    
    def test_boundary_block_threshold(self, navigator_engine):
        """Risk exactly at 0.85 should BLOCK."""
        metrics = generate_normal_metrics()
        metrics["geo_velocity_mph"] = 425.0  # 425/500 = 0.85
        
        result = navigator_engine.evaluate(metrics)
        
        assert result.decision == SentinelDecision.BLOCK
    
    def test_boundary_challenge_threshold(self, navigator_engine):
        """Risk exactly at 0.50 should CHALLENGE."""
        metrics = generate_normal_metrics()
        metrics["geo_velocity_mph"] = 250.0  # 250/500 = 0.50
        
        result = navigator_engine.evaluate(metrics)
        
        assert result.decision == SentinelDecision.CHALLENGE
    
    def test_negative_velocity(self, navigator_engine):
        """Negative velocity should be treated as 0."""
        metrics = generate_normal_metrics()
        metrics["geo_velocity_mph"] = -100.0
        
        result = navigator_engine.evaluate(metrics)
        
        # Negative normalized to 0, so risk is 0
        assert result.risk_score >= 0.0


# =============================================================================
# Session Tracker Tests (Navigator module version)
# =============================================================================

class TestMouseSessionTrackerNavigator:
    """Test MouseSessionTracker from navigator module."""
    
    def test_bot_stroke_increases_strikes(self):
        """Bot strokes should increase strike count."""
        tracker = MouseSessionTracker()
        tracker.record_bot_stroke()
        tracker.record_bot_stroke()
        
        assert tracker.get_strikes() == 2
        assert not tracker.is_flagged()
    
    def test_human_stroke_decreases_strikes(self):
        """Human strokes should decrease strike count."""
        tracker = MouseSessionTracker()
        tracker.record_bot_stroke()
        tracker.record_bot_stroke()
        assert tracker.get_strikes() == 2
        
        tracker.record_human_stroke()
        assert tracker.get_strikes() == 1
    
    def test_strikes_minimum_zero(self):
        """Strikes should never go below 0."""
        tracker = MouseSessionTracker()
        tracker.record_human_stroke()
        tracker.record_human_stroke()
        
        assert tracker.get_strikes() == 0
    
    def test_three_strikes_flags_session(self):
        """3 strikes should flag the session."""
        tracker = MouseSessionTracker()
        tracker.record_bot_stroke()
        tracker.record_bot_stroke()
        tracker.record_bot_stroke()
        
        assert tracker.is_flagged()
        assert tracker.get_strikes() == 3
    
    def test_reset_clears_tracking(self):
        """Reset should clear all tracking state."""
        tracker = MouseSessionTracker()
        tracker.record_bot_stroke()
        tracker.record_bot_stroke()
        tracker.record_bot_stroke()
        assert tracker.is_flagged()
        
        tracker.reset()
        
        assert not tracker.is_flagged()
        assert tracker.get_strikes() == 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestPolicyIntegration:
    """Test engine in realistic scenarios."""
    
    def test_traveling_employee_scenario(self, navigator_engine):
        """Employee traveling at normal speed should be ALLOWED."""
        metrics = {
            "geo_velocity_mph": 65.0,        # Highway speed
            "device_ip_mismatch": 0.0,
            "policy_violation": 0.0,
            "is_new_device": 0.0,
        }
        
        result = navigator_engine.evaluate(metrics)
        
        assert result.decision == SentinelDecision.ALLOW
        assert len(result.anomaly_vectors) == 0
    
    def test_vpn_user_scenario(self, navigator_engine):
        """VPN user should be flagged but might not be blocked."""
        metrics = {
            "geo_velocity_mph": 0.0,
            "device_ip_mismatch": 1.0,       # VPN detected
            "policy_violation": 0.0,
            "is_new_device": 0.0,
        }
        
        result = navigator_engine.evaluate(metrics)
        
        assert "infra_mismatch" in result.anomaly_vectors
        # VPN alone is risk=1.0 (infra_mismatch), should BLOCK
        assert result.decision == SentinelDecision.BLOCK
    
    def test_new_device_login_scenario(self, navigator_engine):
        """New device login should trigger CHALLENGE."""
        metrics = {
            "geo_velocity_mph": 0.0,
            "device_ip_mismatch": 0.0,
            "policy_violation": 0.0,
            "is_new_device": 1.0,            # First time seeing this device
        }
        
        result = navigator_engine.evaluate(metrics)
        
        assert result.decision == SentinelDecision.CHALLENGE
