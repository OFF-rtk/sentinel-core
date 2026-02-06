"""
Mouse Model Unit Tests

Tests for PhysicsMouseModel with tiered bot detection (deterministic, no ML).
Validates tier 1 hard fails, tier 2 additive scoring, tier 3 decision,
and session-level strike tracking.
"""

import pytest

from core.models.mouse import PhysicsMouseModel, MouseSessionTracker
from core.processors.mouse import MouseProcessor
from core.schemas.inputs import MouseEvent, MouseEventType


# =============================================================================
# Bot Feature Generators (Inline)
# =============================================================================

def generate_bot_teleport_features():
    """Bot feature set: impossible teleport speed (Tier 1 hard fail)."""
    return {
        "velocity_mean": 5.0,
        "velocity_std": 1.0,
        "velocity_max": 15.0,        # > 9.0 px/ms = teleport
        "path_distance": 200.0,
        "linearity_error": 5.0,
        "time_diff_std": 10.0,
        "segment_count": 20,
    }


def generate_bot_perfect_line_features():
    """Bot feature set: impossibly straight line (Tier 1 hard fail)."""
    return {
        "velocity_mean": 2.0,
        "velocity_std": 0.5,
        "velocity_max": 3.0,
        "path_distance": 350.0,       # > 300px
        "linearity_error": 0.1,       # < 0.2px = impossible
        "time_diff_std": 5.0,
        "segment_count": 25,
    }


def generate_bot_suspicious_features():
    """Bot feature set: multiple tier 2 flags summing to >= 0.7."""
    return {
        "velocity_mean": 2.0,
        "velocity_std": 0.005,        # < 0.01 = low jitter (+0.25)
        "velocity_max": 3.0,
        "path_distance": 200.0,       # > 150px
        "linearity_error": 0.3,       # < 0.5px = excessive linearity (+0.25)
        "time_diff_std": 0.01,        # < 0.02ms = regular timing (+0.35)
        "segment_count": 25,          # >= 20 segments for timing check
    }


def generate_human_features():
    """Human-like feature set: natural variation, no red flags."""
    return {
        "velocity_mean": 1.5,
        "velocity_std": 0.8,          # Natural jitter
        "velocity_max": 4.0,          # Normal peak
        "path_distance": 150.0,
        "linearity_error": 5.0,       # Natural curve
        "time_diff_std": 15.0,        # Normal timing variation
        "segment_count": 20,
    }


# =============================================================================
# Tier 1 Hard Fail Tests
# =============================================================================

class TestTier1HardFail:
    """Test Tier 1 physics violations cause immediate bot detection."""
    
    def test_teleport_speed_detection(self, mouse_model):
        """Velocity > 9.0 px/ms should immediately return score 1.0."""
        features = generate_bot_teleport_features()
        
        score, vectors = mouse_model.score_one(features)
        
        assert score == 1.0, f"Teleport speed should be score=1.0, got {score}"
        assert "teleport_speed" in vectors, f"Expected 'teleport_speed' in vectors, got {vectors}"
    
    def test_inhuman_linearity_detection(self, mouse_model):
        """Path > 300px with linearity_error < 0.2px should fail."""
        features = generate_bot_perfect_line_features()
        
        score, vectors = mouse_model.score_one(features)
        
        assert score == 1.0, f"Inhuman linearity should be score=1.0, got {score}"
        assert "inhuman_linearity" in vectors, f"Expected 'inhuman_linearity' in vectors, got {vectors}"
    
    def test_boundary_values_teleport(self, mouse_model):
        """Test boundary: velocity_max exactly at 9.0 should pass."""
        features = generate_human_features()
        features["velocity_max"] = 9.0  # Exactly at boundary
        
        score, vectors = mouse_model.score_one(features)
        
        # Should pass (not >9.0)
        assert score == 0.0, f"velocity_max=9.0 should pass, got score={score}"


# =============================================================================
# Tier 2 Additive Scoring Tests
# =============================================================================

class TestTier2Additive:
    """Test Tier 2 suspicious patterns accumulate risk."""
    
    def test_multiple_tier2_flags_trigger_bot(self, mouse_model):
        """Multiple tier 2 flags summing to >= 0.7 should trigger bot."""
        features = generate_bot_suspicious_features()
        
        score, vectors = mouse_model.score_one(features)
        
        assert score == 1.0, f"Multiple tier 2 flags should be bot, got score={score}"
        # Should have multiple vectors
        assert len(vectors) >= 2, f"Expected multiple vectors, got {vectors}"
    
    def test_single_tier2_flag_passes(self, mouse_model):
        """Single tier 2 flag should not exceed threshold."""
        features = generate_human_features()
        features["velocity_std"] = 0.005  # Only one flag: low jitter (+0.25)
        
        score, vectors = mouse_model.score_one(features)
        
        # Risk = 0.25 < 0.7, should pass
        assert score == 0.0, f"Single tier 2 flag should pass, got score={score}"
    
    def test_timing_check_requires_enough_segments(self, mouse_model):
        """Timing check should only apply when segment_count >= 20."""
        features = generate_human_features()
        features["time_diff_std"] = 0.01  # Would trigger if checked
        features["segment_count"] = 15    # Not enough segments
        
        score, vectors = mouse_model.score_one(features)
        
        # Should pass because segment count too low
        assert "overly_regular_timing" not in vectors


# =============================================================================
# Human Data Tests
# =============================================================================

class TestHumanData:
    """Test that real human data produces low/zero scores."""
    
    def test_human_strokes_score_zero(self, mouse_model, human_mouse_features):
        """Human stroke features should mostly score 0.0."""
        if len(human_mouse_features) < 5:
            pytest.skip("Not enough human stroke features")
        
        scores = []
        for features in human_mouse_features:
            score, _ = mouse_model.score_one(features)
            scores.append(score)
        
        # Calculate pass rate
        pass_count = sum(1 for s in scores if s == 0.0)
        total = len(scores)
        pass_rate = pass_count / total
        
        print(f"\n✅ Human data: {pass_count}/{total} strokes passed ({pass_rate*100:.1f}%)")
        
        # At least 80% should pass
        assert pass_rate >= 0.8, f"Pass rate too low: {pass_rate*100:.1f}%"
    
    def test_human_data_no_tier1_fails(self, mouse_model, human_mouse_features):
        """Human data should never trigger tier 1 hard fails."""
        if len(human_mouse_features) < 5:
            pytest.skip("Not enough human stroke features")
        
        tier1_fails = []
        
        for features in human_mouse_features:
            score, vectors = mouse_model.score_one(features)
            if "teleport_speed" in vectors or "inhuman_linearity" in vectors:
                tier1_fails.append((features, vectors))
        
        assert len(tier1_fails) == 0, \
            f"Human data triggered {len(tier1_fails)} tier 1 fails: {tier1_fails[0][1] if tier1_fails else 'N/A'}"


# =============================================================================
# Session Tracker Tests
# =============================================================================

class TestSessionTracker:
    """Test MouseSessionTracker strike accumulation."""
    
    def test_bot_strokes_increase_strikes(self, mouse_session_tracker):
        """Bot strokes should increase strike count."""
        # Record 2 bot strokes
        mouse_session_tracker.record_stroke(1.0, ["teleport_speed"])
        mouse_session_tracker.record_stroke(1.0, ["low_velocity_jitter"])
        
        assert mouse_session_tracker.strikes == 2
        assert not mouse_session_tracker.is_flagged  # Not at threshold yet
    
    def test_human_strokes_decay_strikes(self, mouse_session_tracker):
        """Human strokes should decay strike count."""
        # Add 2 strikes
        mouse_session_tracker.record_stroke(1.0, ["test"])
        mouse_session_tracker.record_stroke(1.0, ["test"])
        assert mouse_session_tracker.strikes == 2
        
        # Decay with human strokes
        mouse_session_tracker.record_stroke(0.0, [])
        assert mouse_session_tracker.strikes == 1
        
        mouse_session_tracker.record_stroke(0.0, [])
        assert mouse_session_tracker.strikes == 0
    
    def test_three_strikes_flags_session(self, mouse_session_tracker):
        """3 strikes should flag the session as bot."""
        mouse_session_tracker.record_stroke(1.0, ["test1"])
        mouse_session_tracker.record_stroke(1.0, ["test2"])
        
        # Third strike
        is_flagged, strikes = mouse_session_tracker.record_stroke(1.0, ["test3"])
        
        assert is_flagged is True
        assert mouse_session_tracker.is_flagged is True
        assert strikes == 3
    
    def test_session_stays_flagged(self, mouse_session_tracker):
        """Once flagged, session should stay flagged even with human strokes."""
        # Flag the session
        for _ in range(3):
            mouse_session_tracker.record_stroke(1.0, ["test"])
        
        assert mouse_session_tracker.is_flagged
        
        # Human strokes should decay strikes but not unflag
        mouse_session_tracker.record_stroke(0.0, [])
        mouse_session_tracker.record_stroke(0.0, [])
        
        assert mouse_session_tracker.is_flagged  # Still flagged
    
    def test_reset_clears_session(self, mouse_session_tracker):
        """Reset should clear all session state."""
        # Flag the session
        for _ in range(3):
            mouse_session_tracker.record_stroke(1.0, ["test"])
        
        mouse_session_tracker.reset()
        
        assert mouse_session_tracker.strikes == 0
        assert not mouse_session_tracker.is_flagged
        assert mouse_session_tracker.stats["total_strokes"] == 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Test model + session tracker integration."""
    
    def test_end_to_end_human_session(self, mouse_model, mouse_session_tracker, human_mouse_features):
        """Human session should not get flagged."""
        if len(human_mouse_features) < 10:
            pytest.skip("Not enough human stroke features")
        
        for features in human_mouse_features[:50]:
            score, vectors = mouse_model.score_one(features)
            mouse_session_tracker.record_stroke(score, vectors)
        
        stats = mouse_session_tracker.stats
        print(f"\n📊 Session stats: {stats}")
        
        # Human session should not be flagged
        assert not mouse_session_tracker.is_flagged, \
            f"Human session incorrectly flagged: {stats}"
    
    def test_end_to_end_bot_session(self, mouse_model, mouse_session_tracker):
        """Bot session should get flagged after 3 bot strokes."""
        # Feed 3 teleport strokes
        for _ in range(3):
            features = generate_bot_teleport_features()
            score, vectors = mouse_model.score_one(features)
            mouse_session_tracker.record_stroke(score, vectors)
        
        assert mouse_session_tracker.is_flagged, "Bot session should be flagged"
        assert mouse_session_tracker.strikes >= 3
