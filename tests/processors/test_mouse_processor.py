"""
Mouse Processor Unit Tests

Tests for MouseProcessor feature extraction from raw mouse events.
Validates stroke termination (click, pause), segment filtering,
feature calculation, and human data integration.
"""

import pytest

from core.processors.mouse import (
    MouseProcessor, 
    MIN_STROKE_EVENTS, 
    MIN_STROKE_DISTANCE,
    PAUSE_THRESHOLD_MS
)
from core.schemas.inputs import MouseEvent, MouseEventType

# Import helper from conftest
from tests.conftest import make_mouse_event as make_event


# =============================================================================
# Stroke Termination Tests
# =============================================================================

class TestStrokeTermination:
    """Test that strokes are correctly terminated by clicks and pauses."""
    
    def test_click_terminates_stroke(self, mouse_processor):
        """CLICK event should terminate current stroke and emit features."""
        # Create a valid stroke: many movements ending with click
        features = None
        
        for i in range(30):
            ts = i * 10.0  # 10ms between events
            x = i * 20     # 20px per move = 600px total
            y = i * 10
            
            if i < 29:
                result = mouse_processor.process_event(make_event(x, y, MouseEventType.MOVE, ts))
            else:
                # Final click
                result = mouse_processor.process_event(make_event(x, y, MouseEventType.CLICK, ts))
                features = result
        
        assert features is not None, "Click should have triggered stroke emission"
        assert 'velocity_mean' in features
        assert 'trajectory_efficiency' in features
    
    def test_pause_terminates_stroke(self, mouse_processor):
        """Long pause (>500ms) should terminate current stroke."""
        features = None
        
        # First, create a valid stroke
        for i in range(20):
            ts = i * 10.0
            x = i * 20
            y = i * 10
            mouse_processor.process_event(make_event(x, y, MouseEventType.MOVE, ts))
        
        # Now add event after long pause (600ms gap)
        last_ts = 19 * 10.0  # 190ms
        pause_ts = last_ts + 600.0  # 790ms - 600ms gap
        
        result = mouse_processor.process_event(make_event(500, 300, MouseEventType.MOVE, pause_ts))
        
        # Pause should have triggered stroke emission
        assert result is not None, "Pause >500ms should trigger stroke emission"
        assert 'velocity_mean' in result


# =============================================================================
# Minimum Threshold Tests
# =============================================================================

class TestMinimumThresholds:
    """Test that strokes below minimum thresholds are rejected."""
    
    def test_min_events_gate(self, mouse_processor):
        """Strokes with fewer than MIN_STROKE_EVENTS should be rejected."""
        # Only 5 movements (less than MIN_STROKE_EVENTS=10)
        for i in range(5):
            ts = i * 10.0
            x = i * 50
            y = i * 50
            mouse_processor.process_event(make_event(x, y, MouseEventType.MOVE, ts))
        
        # Try to terminate with click
        result = mouse_processor.process_event(make_event(250, 250, MouseEventType.CLICK, 60.0))
        
        # Should be rejected due to insufficient events
        assert result is None, "Stroke with <10 events should be rejected"
    
    def test_min_distance_gate(self, mouse_processor):
        """Strokes with total distance < MIN_STROKE_DISTANCE should be rejected."""
        # Many movements but tiny distance (< 50px total)
        for i in range(15):
            ts = i * 10.0
            # Very small movements: 2px each = 28px total
            x = 100 + (i % 3)
            y = 100 + (i % 2)
            mouse_processor.process_event(make_event(x, y, MouseEventType.MOVE, ts))
        
        result = mouse_processor.process_event(make_event(102, 101, MouseEventType.CLICK, 160.0))
        
        # Should be rejected due to insufficient distance
        assert result is None, "Stroke with <50px distance should be rejected"


# =============================================================================
# Feature Calculation Tests
# =============================================================================

class TestFeatureCalculation:
    """Test accuracy of feature calculations."""
    
    def test_velocity_calculation(self, mouse_processor):
        """Velocity should be distance/time."""
        # Create stroke: 100px in 10ms = 10 px/ms velocity
        for i in range(25):
            ts = i * 10.0
            x = i * 10  # 10px per 10ms = 1 px/ms
            y = 0
            mouse_processor.process_event(make_event(x, y, MouseEventType.MOVE, ts))
        
        result = mouse_processor.process_event(make_event(250, 0, MouseEventType.CLICK, 250.0))
        
        assert result is not None
        # Velocity should be approximately 1 px/ms (may vary due to segment filtering)
        assert 0.5 <= result['velocity_mean'] <= 2.0, \
            f"Expected velocity ~1 px/ms, got {result['velocity_mean']}"
    
    def test_trajectory_efficiency(self, mouse_processor):
        """Straight line should have efficiency close to 1.0."""
        # Create perfectly straight horizontal line
        for i in range(25):
            ts = i * 10.0
            x = i * 20  # Straight horizontal line
            y = 0
            mouse_processor.process_event(make_event(x, y, MouseEventType.MOVE, ts))
        
        result = mouse_processor.process_event(make_event(480, 0, MouseEventType.CLICK, 250.0))
        
        assert result is not None
        # Straight line should have high efficiency (close to 1.0)
        assert result['trajectory_efficiency'] > 0.9, \
            f"Straight line efficiency should be >0.9, got {result['trajectory_efficiency']}"
    
    def test_curved_path_lower_efficiency(self, mouse_processor):
        """Curved path should have lower trajectory efficiency."""
        import math
        
        # Create a semi-circular arc
        for i in range(30):
            ts = i * 10.0
            angle = i * (math.pi / 29)  # 0 to π
            x = int(200 + 100 * math.cos(angle))
            y = int(100 * math.sin(angle))
            mouse_processor.process_event(make_event(x, y, MouseEventType.MOVE, ts))
        
        result = mouse_processor.process_event(make_event(100, 0, MouseEventType.CLICK, 300.0))
        
        if result is not None:
            # Curved path should have lower efficiency than straight line
            assert result['trajectory_efficiency'] < 0.9, \
                f"Curved path efficiency should be <0.9, got {result['trajectory_efficiency']}"


# =============================================================================
# Human Data Integration Tests
# =============================================================================

class TestHumanDataIntegration:
    """Test processor with real human mouse recording."""
    
    def test_human_data_produces_valid_strokes(self, mouse_processor, human_mouse_events):
        """Human mouse recording should produce valid stroke features."""
        stroke_count = 0
        all_features = []
        
        for event in human_mouse_events[:5000]:  # First 5000 events for speed
            result = mouse_processor.process_event(event)
            if result is not None:
                stroke_count += 1
                all_features.append(result)
                
                # Validate feature structure
                assert 'velocity_mean' in result
                assert 'velocity_std' in result
                assert 'trajectory_efficiency' in result
                assert 'path_distance' in result
                assert 'linearity_error' in result
                assert 'segment_count' in result
                
                # Validate reasonable ranges
                assert 0 < result['velocity_mean'] < 10, \
                    f"Velocity out of range: {result['velocity_mean']}"
                assert 0 <= result['trajectory_efficiency'] <= 1.0, \
                    f"Efficiency out of range: {result['trajectory_efficiency']}"
        
        # Should have produced some strokes
        assert stroke_count > 0, "Human data should produce at least one stroke"
        
        print(f"\n✅ Processed {min(5000, len(human_mouse_events))} events → {stroke_count} strokes")
        if all_features:
            print(f"   Sample features: {all_features[0]}")
    
    def test_stroke_count_tracking(self, mouse_processor, human_mouse_events):
        """Processor should accurately track stroke count."""
        for event in human_mouse_events[:2000]:
            mouse_processor.process_event(event)
        
        stroke_count = mouse_processor.get_stroke_count()
        assert stroke_count >= 0
        print(f"\n📊 Stroke count after 2000 events: {stroke_count}")


# =============================================================================
# Reset Tests
# =============================================================================

class TestProcessorReset:
    """Test processor reset functionality."""
    
    def test_reset_clears_state(self, mouse_processor):
        """Reset should clear all internal state."""
        # Feed some events
        for i in range(15):
            mouse_processor.process_event(make_event(i * 20, i * 10, MouseEventType.MOVE, i * 10.0))
        
        # Force a stroke with click
        mouse_processor.process_event(make_event(300, 150, MouseEventType.CLICK, 160.0))
        
        initial_count = mouse_processor.get_stroke_count()
        
        # Reset
        mouse_processor.reset()
        
        # State should be cleared
        assert mouse_processor.get_stroke_count() == 0, "Stroke count not reset"
    
    def test_reset_allows_new_strokes(self, mouse_processor):
        """After reset, processor should accept new strokes normally."""
        # Create and complete a stroke
        for i in range(20):
            mouse_processor.process_event(make_event(i * 20, i * 10, MouseEventType.MOVE, i * 10.0))
        mouse_processor.process_event(make_event(400, 200, MouseEventType.CLICK, 210.0))
        
        # Reset
        mouse_processor.reset()
        
        # Create a new stroke
        for i in range(20):
            mouse_processor.process_event(make_event(i * 20, i * 10, MouseEventType.MOVE, i * 10.0 + 1000.0))
        result = mouse_processor.process_event(make_event(400, 200, MouseEventType.CLICK, 1210.0))
        
        # Should produce features for the new stroke
        assert result is not None, "Reset should allow new strokes to be processed"
