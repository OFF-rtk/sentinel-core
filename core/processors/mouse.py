"""
Sentinel Mouse Processor

Stateless feature engineering for mouse movement biometrics.
Extracts velocity, angle, and trajectory efficiency from raw mouse events.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from core.schemas.inputs import MouseEvent


# =============================================================================
# Constants
# =============================================================================

# Minimum distance to avoid jitter (sensor noise)
MIN_DISTANCE_PIXELS = 3.0

# Minimum time difference to avoid lag spikes
MIN_TIME_DIFF_MS = 5.0

# Maximum time difference before considered idle/new gesture
MAX_TIME_DIFF_MS = 1000.0

# Window size for feature calculation
WINDOW_SIZE = 50


# =============================================================================
# Internal Data Structures
# =============================================================================

@dataclass
class MouseSegment:
    """A valid movement segment between two points."""
    distance: float  # pixels
    time_diff: float  # ms
    velocity: float  # pixels/ms
    angle: float  # radians
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]


# =============================================================================
# Mouse Processor
# =============================================================================

class MouseProcessor:
    """
    Processes raw mouse events to extract behavioral features.
    
    Features extracted:
    - velocity_mean: Average movement speed (pixels/ms)
    - velocity_std: Standard deviation of velocity
    - angle_mean: Average movement angle (radians)
    - angle_std: Standard deviation of angles
    - trajectory_efficiency: Ratio of net distance to path distance
    """
    
    def extract_features(self, events: List[MouseEvent]) -> Dict[str, float]:
        """
        Extract mouse movement behavioral features from raw events.
        
        Args:
            events: List of MouseEvent objects (may contain duplicates, unsorted)
            
        Returns:
            Dictionary of feature names to values
        """
        if len(events) < 2:
            return self._empty_features()
        
        # Step 1: Sort by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Step 2: Deduplicate events with same timestamp
        deduped_events = self._deduplicate(sorted_events)
        
        if len(deduped_events) < 2:
            return self._empty_features()
        
        # Step 3-4: Extract valid segments with noise filtering
        segments = self._extract_segments(deduped_events)
        
        if not segments:
            return self._empty_features()
        
        # Step 5: Apply window (last N segments)
        segments = segments[-WINDOW_SIZE:]
        
        # Step 6: Calculate features
        velocities = [seg.velocity for seg in segments]
        angles = [seg.angle for seg in segments]
        
        # Trajectory efficiency
        trajectory_efficiency = self._calculate_trajectory_efficiency(segments)
        
        return {
            "velocity_mean": self._mean(velocities),
            "velocity_std": self._std(velocities),
            "angle_mean": self._mean(angles),
            "angle_std": self._std(angles),
            "trajectory_efficiency": trajectory_efficiency,
        }
    
    def _deduplicate(self, events: List[MouseEvent]) -> List[MouseEvent]:
        """Remove events with duplicate timestamps."""
        seen_timestamps = set()
        unique_events = []
        
        for event in events:
            if event.timestamp not in seen_timestamps:
                seen_timestamps.add(event.timestamp)
                unique_events.append(event)
        
        return unique_events
    
    def _extract_segments(self, events: List[MouseEvent]) -> List[MouseSegment]:
        """
        Extract valid movement segments between event pairs.
        Applies jitter, lag, and idle filtering.
        """
        segments: List[MouseSegment] = []
        
        for i in range(len(events) - 1):
            p1 = events[i]
            p2 = events[i + 1]
            
            # Calculate distance
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            distance = math.sqrt(dx * dx + dy * dy)
            
            # Calculate time difference
            time_diff = p2.timestamp - p1.timestamp
            
            # Filter: Jitter (too short distance)
            if distance < MIN_DISTANCE_PIXELS:
                continue
            
            # Filter: Lag spike (too short time)
            if time_diff < MIN_TIME_DIFF_MS:
                continue
            
            # Filter: Idle/parking (too long time - new gesture)
            if time_diff > MAX_TIME_DIFF_MS:
                continue
            
            # Calculate velocity (pixels/ms)
            velocity = distance / time_diff
            
            # Calculate angle (radians)
            angle = math.atan2(dy, dx)
            
            segments.append(MouseSegment(
                distance=distance,
                time_diff=time_diff,
                velocity=velocity,
                angle=angle,
                start_point=(p1.x, p1.y),
                end_point=(p2.x, p2.y)
            ))
        
        return segments
    
    def _calculate_trajectory_efficiency(self, segments: List[MouseSegment]) -> float:
        """
        Calculate trajectory efficiency.
        Efficiency = Net Distance / Path Distance
        """
        if not segments:
            return 0.0
        
        # Path distance: sum of all segment distances
        path_distance = sum(seg.distance for seg in segments)
        
        if path_distance == 0:
            return 0.0
        
        # Net distance: straight line from first point to last point
        first_point = segments[0].start_point
        last_point = segments[-1].end_point
        
        dx = last_point[0] - first_point[0]
        dy = last_point[1] - first_point[1]
        net_distance = math.sqrt(dx * dx + dy * dy)
        
        return net_distance / path_distance
    
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
            "velocity_mean": 0.0,
            "velocity_std": 0.0,
            "angle_mean": 0.0,
            "angle_std": 0.0,
            "trajectory_efficiency": 0.0,
        }
