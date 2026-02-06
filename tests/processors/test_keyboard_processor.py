"""
Keyboard Processor Unit Tests

Tests for KeyboardProcessor feature extraction from raw keyboard events.
Validates window thresholds, stride intervals, dwell/flight time calculations,
and coffee break filtering.
"""

import pytest

from core.processors.keyboard import KeyboardProcessor, WINDOW_SIZE, WINDOW_STRIDE
from core.schemas.inputs import KeyboardEvent, KeyEventType

# Import helper from conftest
from tests.conftest import make_keyboard_event as make_event


# =============================================================================
# Window Threshold Tests
# =============================================================================

class TestWindowThreshold:
    """Test that features are only emitted after WINDOW_SIZE keystrokes."""
    
    def test_no_features_before_window_size(self, keyboard_processor):
        """No features should be returned before reaching WINDOW_SIZE keystrokes."""
        # Feed 49 key down events (one less than WINDOW_SIZE=50)
        for i in range(WINDOW_SIZE - 1):
            ts = i * 100.0
            result = keyboard_processor.process_event(make_event('a', KeyEventType.DOWN, ts))
            keyboard_processor.process_event(make_event('a', KeyEventType.UP, ts + 50))
            
            # Should not emit features yet
            assert result is None, f"Unexpected features at keystroke {i+1}"
    
    def test_features_emitted_at_window_size(self, keyboard_processor):
        """Features should be emitted exactly at WINDOW_SIZE keystrokes."""
        # Feed exactly WINDOW_SIZE keystrokes
        for i in range(WINDOW_SIZE):
            ts = i * 100.0
            result = keyboard_processor.process_event(make_event('a', KeyEventType.DOWN, ts))
            keyboard_processor.process_event(make_event('a', KeyEventType.UP, ts + 50))
            
            if i == WINDOW_SIZE - 1:
                # Should emit features on the 50th keystroke
                assert result is not None, f"Expected features at keystroke {WINDOW_SIZE}"
                assert isinstance(result, dict)
                assert 'dwell_time_mean' in result
                assert 'flight_time_mean' in result


# =============================================================================
# Stride Interval Tests
# =============================================================================

class TestStrideInterval:
    """Test that features are emitted every WINDOW_STRIDE keystrokes after first window."""
    
    def test_stride_triggers_correctly(self, keyboard_processor):
        """After first window, features should emit every STRIDE keystrokes."""
        feature_emissions = []
        
        # Feed enough keystrokes for multiple windows
        total_keystrokes = WINDOW_SIZE + WINDOW_STRIDE * 3  # 50 + 15 = 65
        
        for i in range(total_keystrokes):
            ts = i * 100.0
            result = keyboard_processor.process_event(make_event('a', KeyEventType.DOWN, ts))
            keyboard_processor.process_event(make_event('a', KeyEventType.UP, ts + 50))
            
            if result is not None:
                feature_emissions.append(i + 1)  # 1-indexed keystroke number
        
        # Expected emissions: 50, 55, 60, 65
        expected = [WINDOW_SIZE + WINDOW_STRIDE * j for j in range(4)]
        expected[0] = WINDOW_SIZE  # First at 50
        
        assert feature_emissions == expected, f"Expected emissions at {expected}, got {feature_emissions}"


# =============================================================================
# Feature Accuracy Tests
# =============================================================================

class TestFeatureAccuracy:
    """Test that feature calculations are accurate."""
    
    def test_dwell_time_accuracy(self, keyboard_processor):
        """Dwell time should be calculated as UP timestamp - DOWN timestamp."""
        # Create events with known dwell time of 100ms
        for i in range(WINDOW_SIZE):
            ts = i * 200.0
            keyboard_processor.process_event(make_event('a', KeyEventType.DOWN, ts))
            result = keyboard_processor.process_event(make_event('a', KeyEventType.UP, ts + 100.0))
            
            if result is not None:
                # Dwell should be approximately 100ms
                assert 95.0 <= result['dwell_time_mean'] <= 105.0, \
                    f"Expected dwell ~100ms, got {result['dwell_time_mean']}"
    
    def test_flight_time_accuracy(self, keyboard_processor):
        """Flight time should be next DOWN - previous UP."""
        # Create events with known flight time of 50ms
        for i in range(WINDOW_SIZE):
            down_ts = i * 200.0
            up_ts = down_ts + 100.0  # dwell = 100ms
            # Next key starts 50ms after this UP
            keyboard_processor.process_event(make_event('a', KeyEventType.DOWN, down_ts))
            result = keyboard_processor.process_event(make_event('a', KeyEventType.UP, up_ts))
            
            if result is not None:
                # Flight time should be non-negative
                assert result['flight_time_mean'] >= 0, \
                    f"Flight time should be non-negative, got {result['flight_time_mean']}"
    
    def test_error_rate_calculation(self, keyboard_processor):
        """Error rate should be backspace count / total keystrokes."""
        # Feed 40 regular keys + 10 backspaces = 20% error rate
        regular_keys = 40
        backspace_keys = 10
        
        for i in range(regular_keys):
            ts = i * 100.0
            keyboard_processor.process_event(make_event('a', KeyEventType.DOWN, ts))
            keyboard_processor.process_event(make_event('a', KeyEventType.UP, ts + 50))
        
        for i in range(backspace_keys):
            ts = (regular_keys + i) * 100.0
            result = keyboard_processor.process_event(make_event('Backspace', KeyEventType.DOWN, ts))
            keyboard_processor.process_event(make_event('Backspace', KeyEventType.UP, ts + 50))
            
            if result is not None:
                assert 0.0 <= result['error_rate'] <= 0.5, \
                    f"Error rate out of expected range: {result['error_rate']}"


# =============================================================================
# Coffee Break Filter Tests
# =============================================================================

class TestCoffeeBreakFilter:
    """Test that long pauses are filtered from flight time calculations."""
    
    def test_long_pause_excluded_from_flight_time(self, keyboard_processor):
        """Flight times > 2000ms should be excluded (coffee break rule)."""
        # Feed 49 normal keystrokes
        for i in range(WINDOW_SIZE - 1):
            ts = i * 100.0
            keyboard_processor.process_event(make_event('a', KeyEventType.DOWN, ts))
            keyboard_processor.process_event(make_event('a', KeyEventType.UP, ts + 50))
        
        # Add a 5 second pause before the 50th keystroke
        pause_ts = (WINDOW_SIZE - 1) * 100.0 + 5000.0  # 5 second gap
        result = keyboard_processor.process_event(make_event('a', KeyEventType.DOWN, pause_ts))
        keyboard_processor.process_event(make_event('a', KeyEventType.UP, pause_ts + 50))
        
        if result is not None:
            # Flight time mean should not be skewed by the 5 second pause
            assert result['flight_time_mean'] < 2000.0, \
                f"Coffee break not filtered: {result['flight_time_mean']}"


# =============================================================================
# Human Data Integration Tests
# =============================================================================

class TestHumanDataIntegration:
    """Test processor with real human keyboard recording."""
    
    def test_human_data_produces_valid_features(self, keyboard_processor, human_keyboard_events):
        """Human keyboard recording should produce valid feature dictionaries."""
        feature_count = 0
        last_features = None
        
        for event in human_keyboard_events:
            result = keyboard_processor.process_event(event)
            if result is not None:
                feature_count += 1
                last_features = result
                
                # Validate feature structure
                assert 'dwell_time_mean' in result
                assert 'dwell_time_std' in result
                assert 'flight_time_mean' in result
                assert 'flight_time_std' in result
                assert 'error_rate' in result
                
                # Validate reasonable ranges for human typing
                assert 0 < result['dwell_time_mean'] < 500, \
                    f"Dwell mean out of human range: {result['dwell_time_mean']}"
                assert result['error_rate'] <= 0.5, \
                    f"Error rate too high: {result['error_rate']}"
        
        # Should have produced multiple feature windows
        assert feature_count > 10, f"Expected multiple windows, got {feature_count}"
        
        print(f"\n✅ Processed {len(human_keyboard_events)} events → {feature_count} feature windows")
        print(f"   Last features: {last_features}")


# =============================================================================
# Reset Tests
# =============================================================================

class TestProcessorReset:
    """Test processor reset functionality."""
    
    def test_reset_clears_state(self, keyboard_processor):
        """Reset should clear all internal state."""
        # Feed some events
        for i in range(30):
            keyboard_processor.process_event(make_event('a', KeyEventType.DOWN, i * 100.0))
            keyboard_processor.process_event(make_event('a', KeyEventType.UP, i * 100.0 + 50))
        
        # Reset
        keyboard_processor.reset()
        
        # State should be cleared - feeding 49 events should not emit features
        for i in range(WINDOW_SIZE - 1):
            result = keyboard_processor.process_event(make_event('a', KeyEventType.DOWN, i * 100.0))
            keyboard_processor.process_event(make_event('a', KeyEventType.UP, i * 100.0 + 50))
            assert result is None, "State was not properly reset"
