"""
Sentinel Keyboard Processor

Stateless feature engineering for keyboard biometrics.
Extracts dwell time, flight time, and error rate from raw key events.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from core.schemas.inputs import KeyboardEvent, KeyEventType


# =============================================================================
# Constants
# =============================================================================

# Maximum flight time before considered a "pause" (coffee break rule)
MAX_FLIGHT_TIME_MS = 2000.0

# Window size for feature calculation
WINDOW_SIZE = 50


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
    
    Features extracted:
    - dwell_time_mean: Average time keys are held down (ms)
    - dwell_time_std: Standard deviation of dwell times
    - flight_time_mean: Average time between key releases and next key press (ms)
    - flight_time_std: Standard deviation of flight times
    - error_rate: Ratio of backspace/delete presses to total keypresses
    """
    
    # Keys that indicate typing errors
    ERROR_KEYS = {"Backspace", "Delete", "backspace", "delete"}
    
    def extract_features(self, events: List[KeyboardEvent]) -> Dict[str, float]:
        """
        Extract keyboard behavioral features from raw events.
        
        Args:
            events: List of KeyboardEvent objects (unsorted, unpaired)
            
        Returns:
            Dictionary of feature names to values
        """
        if not events:
            return self._empty_features()
        
        # Step 1: Pair DOWN/UP events for same key
        key_presses = self._pair_events(events)
        
        if not key_presses:
            return self._empty_features()
        
        # Step 2: Sort by press_time
        key_presses.sort(key=lambda kp: kp.press_time)
        
        # Step 3: Extract dwell times
        dwell_times = [kp.dwell_time for kp in key_presses if kp.dwell_time > 0]
        
        # Step 4: Extract flight times with "coffee break" filtering
        flight_times = self._extract_flight_times(key_presses)
        
        # Step 5: Apply window (last N intervals)
        dwell_times = dwell_times[-WINDOW_SIZE:]
        flight_times = flight_times[-WINDOW_SIZE:]
        
        # Step 6: Calculate error rate
        error_rate = self._calculate_error_rate(events)
        
        # Step 7: Compute statistics
        return {
            "dwell_time_mean": self._mean(dwell_times),
            "dwell_time_std": self._std(dwell_times),
            "flight_time_mean": self._mean(flight_times),
            "flight_time_std": self._std(flight_times),
            "error_rate": error_rate,
        }
    
    def _pair_events(self, events: List[KeyboardEvent]) -> List[KeyPress]:
        """
        Match DOWN/UP events for the same key.
        Discards incomplete pairs.
        """
        # Track pending DOWN events per key
        pending_downs: Dict[str, List[float]] = defaultdict(list)
        key_presses: List[KeyPress] = []
        
        # Sort by timestamp first to ensure proper ordering
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        for event in sorted_events:
            if event.event_type == KeyEventType.DOWN:
                pending_downs[event.key].append(event.timestamp)
            elif event.event_type == KeyEventType.UP:
                if pending_downs[event.key]:
                    # Match with earliest pending DOWN for this key
                    press_time = pending_downs[event.key].pop(0)
                    key_presses.append(KeyPress(
                        key=event.key,
                        press_time=press_time,
                        release_time=event.timestamp
                    ))
        
        return key_presses
    
    def _extract_flight_times(self, key_presses: List[KeyPress]) -> List[float]:
        """
        Calculate flight times between consecutive key presses.
        Applies the "coffee break" rule to filter pauses.
        """
        flight_times: List[float] = []
        
        for i in range(len(key_presses) - 1):
            current = key_presses[i]
            next_press = key_presses[i + 1]
            
            # Flight = Next key DOWN - Current key UP
            flight = next_press.press_time - current.release_time
            
            # Coffee break rule: skip long pauses
            if 0 < flight <= MAX_FLIGHT_TIME_MS:
                flight_times.append(flight)
        
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
