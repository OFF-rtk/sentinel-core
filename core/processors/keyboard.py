"""
Sentinel Keyboard Processor

Stateful feature engineering for keyboard biometrics.
Extracts dwell time, flight time, and error rate from raw key events.
Implements a sliding window with configurable stride for continuous streaming.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.schemas.inputs import KeyboardEvent, KeyEventType


# =============================================================================
# Constants
# =============================================================================

# Maximum flight time before considered a "pause" (coffee break rule)
MAX_FLIGHT_TIME_MS = 2000.0

# Window size for feature calculation
WINDOW_SIZE = 50

# Stride for sliding window (every N keystrokes, emit a new feature vector)
WINDOW_STRIDE = 5

# Debug flag
DEBUG = True


# =============================================================================
# Internal Data Structures
# =============================================================================

@dataclass
class KeyPress:
    """Paired key press with down and up timestamps."""
    key: str
    press_time: float  # DOWN timestamp
    release_time: float  # UP timestamp
    
    @property
    def dwell_time(self) -> float:
        """Time key was held down (ms)."""
        return self.release_time - self.press_time


# =============================================================================
# Keyboard Processor
# =============================================================================

class KeyboardProcessor:
    """
    Processes raw keyboard events to extract behavioral features.
    
    Implements a sliding window with stride for continuous streaming:
    - Window 1: keystrokes 0-49
    - Window 2: keystrokes 5-54  (stride=5)
    - Window 3: keystrokes 10-59
    - etc.
    
    Features extracted:
    - dwell_time_mean: Average time keys are held down (ms)
    - dwell_time_std: Standard deviation of dwell times
    - flight_time_mean: Average time between key releases and next key press (ms)
    - flight_time_std: Standard deviation of flight times
    - error_rate: Ratio of backspace/delete presses to total keypresses
    """
    
    # Keys that indicate typing errors
    ERROR_KEYS = {"Backspace", "Delete", "backspace", "delete"}
    
    def __init__(self) -> None:
        """Initialize the processor with empty state."""
        self._pending_downs: Dict[str, List[float]] = defaultdict(list)
        self._key_presses: List[KeyPress] = []
        self._all_events: List[KeyboardEvent] = []
        self._keystroke_count: int = 0
        self._last_window_start: int = 0
    
    def process_event(self, event: KeyboardEvent) -> Optional[Dict[str, float]]:
        """
        Process a single keyboard event and return features if window is ready.
        
        This is the main streaming API. Call this for every keyup/keydown event.
        Returns a feature dict every WINDOW_STRIDE keystrokes after reaching
        WINDOW_SIZE keystrokes.
        
        Args:
            event: Single KeyboardEvent (DOWN or UP)
            
        Returns:
            Feature dictionary if window is ready, None otherwise
        """
        self._all_events.append(event)
        
        # Pair DOWN/UP events
        if event.event_type == KeyEventType.DOWN:
            self._pending_downs[event.key].append(event.timestamp)
            self._keystroke_count += 1
            
            if DEBUG:
                print(f"[PROCESSOR] Keystroke #{self._keystroke_count}: {event.key} DOWN")
            
        elif event.event_type == KeyEventType.UP:
            if self._pending_downs[event.key]:
                press_time = self._pending_downs[event.key].pop(0)
                kp = KeyPress(
                    key=event.key,
                    press_time=press_time,
                    release_time=event.timestamp
                )
                self._key_presses.append(kp)
                
                if DEBUG:
                    print(f"[PROCESSOR] Paired {event.key}: dwell={kp.dwell_time:.1f}ms")
        
        # Check if we should emit a feature vector
        # Need at least WINDOW_SIZE keystrokes
        if self._keystroke_count < WINDOW_SIZE:
            return None
        
        # Emit on first window, then every STRIDE keystrokes
        should_emit = (
            self._keystroke_count == WINDOW_SIZE or
            (self._keystroke_count - WINDOW_SIZE) % WINDOW_STRIDE == 0
        )
        
        if should_emit and event.event_type == KeyEventType.DOWN:
            if DEBUG:
                window_num = 1 + (self._keystroke_count - WINDOW_SIZE) // WINDOW_STRIDE
                print(f"[PROCESSOR] 📊 Emitting window #{window_num} at keystroke {self._keystroke_count}")
            
            return self._extract_features_from_window()
        
        return None
    
    def _extract_features_from_window(self) -> Dict[str, float]:
        """Extract features from the last WINDOW_SIZE keypresses."""
        # Sort keypresses by time
        sorted_presses = sorted(self._key_presses, key=lambda kp: kp.press_time)
        
        # Take the last WINDOW_SIZE keypresses
        window_presses = sorted_presses[-WINDOW_SIZE:]
        
        if len(window_presses) < 2:
            return self._empty_features()
        
        # Extract dwell times
        dwell_times = [kp.dwell_time for kp in window_presses if kp.dwell_time >= 0]
        
        # Extract flight times
        flight_times = self._extract_flight_times(window_presses)
        
        # Calculate error rate from recent events
        recent_events = self._all_events[-(WINDOW_SIZE * 2):]  # Approximate
        error_rate = self._calculate_error_rate(recent_events)
        
        features = {
            "dwell_time_mean": self._mean(dwell_times),
            "dwell_time_std": self._std(dwell_times),
            "flight_time_mean": self._mean(flight_times),
            "flight_time_std": self._std(flight_times),
            "error_rate": error_rate,
        }
        
        if DEBUG:
            print(f"[PROCESSOR] Features extracted:")
            for k, v in features.items():
                print(f"[PROCESSOR]   {k}: {v:.4f}")
        
        return features
    
    # extract_features removed (legacy batch API)
    # _pair_events removed (legacy helper)
    
    def _extract_flight_times(self, key_presses: List[KeyPress]) -> List[float]:
        """
        Calculate flight times between consecutive key presses.
        Applies the "coffee break" rule to filter pauses.
        """
        flight_times: List[float] = []
        filtered_count = 0
        
        for i in range(len(key_presses) - 1):
            current = key_presses[i]
            next_press = key_presses[i + 1]
            
            # Flight = Next key DOWN - Current key UP
            flight = next_press.press_time - current.release_time
            
            # Coffee break rule: skip long pauses
            if flight <= MAX_FLIGHT_TIME_MS:
                flight_times.append(flight)
            else:
                filtered_count += 1
        
        if DEBUG and filtered_count > 0:
            print(f"[PROCESSOR] _extract_flight_times: Filtered out {filtered_count} pauses > {MAX_FLIGHT_TIME_MS}ms")
        
        return flight_times
    
    def _calculate_error_rate(self, events: List[KeyboardEvent]) -> float:
        """Calculate ratio of error correction keys to total keypresses."""
        down_events = [e for e in events if e.event_type == KeyEventType.DOWN]
        
        if not down_events:
            return 0.0
        
        error_count = sum(1 for e in down_events if e.key in self.ERROR_KEYS)
        return error_count / len(down_events)
    
    def _mean(self, values: List[float]) -> float:
        """Calculate mean of values."""
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation of values."""
        if len(values) < 2:
            return 0.0
        
        mean = self._mean(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature set when no valid data."""
        return {
            "dwell_time_mean": 0.0,
            "dwell_time_std": 0.0,
            "flight_time_mean": 0.0,
            "flight_time_std": 0.0,
            "error_rate": 0.0,
        }
    
    def reset(self) -> None:
        """Reset processor state for a new session."""
        self._pending_downs.clear()
        self._key_presses.clear()
        self._all_events.clear()
        self._keystroke_count = 0
        self._last_window_start = 0
