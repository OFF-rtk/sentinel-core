"""
Mouse Movement Processor - Sentinel Physics v2

This module implements the Action-Based Segmentation strategy for mouse behavioral
biometrics. Instead of fixed time windows, it captures "Strokes" - intentional
movement sequences that end with a Click, Drag, or Pause.

Architecture:
- Stateless transformation of raw events into behavioral feature vectors
- Biomechanical filters for data sanitization
- Physics-ready features for deterministic liveness detection
- Circular statistics for angular measurements

Features extracted:
- velocity_mean, velocity_std, velocity_max (p95)
- angle_mean, angle_std
- curvature_mean, curvature_std
- trajectory_efficiency
- path_distance, linearity_error, time_diff_std
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from core.schemas.inputs import MouseEvent, MouseEventType

# Debug flag
DEBUG = False

# =============================================================================
# STROKE VALIDATION THRESHOLDS
# =============================================================================

# Minimum events required to form a valid stroke
MIN_STROKE_EVENTS = 10

# Minimum distance (pixels) for a valid stroke
MIN_STROKE_DISTANCE = 50.0

# Pause threshold (ms) - movement stops trigger stroke flush
PAUSE_THRESHOLD_MS = 500.0

# Segment-level filters (biomechanical + corruption detection)
MIN_SEGMENT_DISTANCE = 3.0      # px - minimum distance (sub-pixel noise filter)
MIN_SEGMENT_TIME_MS = 4.0       # ms - human motor resolution (~250 Hz)
MAX_SEGMENT_TIME_MS = 2000.0    # ms - beyond this is a pause, not a segment
MAX_VELOCITY_PX_PER_MS = 8.0    # px/ms - biomechanical ceiling (extreme flicks ≤6, generous)


@dataclass
class Segment:
    """A validated movement segment between two points."""
    distance: float
    time_diff: float
    velocity: float
    angle: float
    start_point: tuple
    end_point: tuple


class MouseProcessor:
    """
    Action-Based Mouse Movement Processor.
    
    Converts raw mouse events into behavioral feature vectors using
    "Stroke" segmentation - capturing intentional movement sequences
    that end with Clicks, Drags, or Pauses.
    
    Features extracted per stroke:
    - velocity_mean: Average speed (px/ms)
    - velocity_std: Speed variation
    - angle_mean: Circular mean of movement direction
    - angle_std: Circular std of movement direction
    - curvature_mean: Average change in angle per pixel
    - curvature_std: Curvature variation
    - trajectory_efficiency: Net distance / Path distance
    """
    
    def __init__(self) -> None:
        """Initialize the processor with empty buffers."""
        self._event_buffer: List[MouseEvent] = []
        self._segment_buffer: List[Segment] = []
        self._last_event: Optional[MouseEvent] = None
        self._stroke_count: int = 0
        
        if DEBUG:
            print("[MOUSE PROCESSOR] Initialized with action-based segmentation")
    
    def process_event(self, event: MouseEvent) -> Optional[Dict[str, float]]:
        """
        Process a single mouse event and return features if a stroke completes.
        
        Stroke terminators:
        - CLICK: User found target and clicked
        - DRAG: User performing precise operation (future)
        - PAUSE: >500ms since last movement
        
        Args:
            event: Single MouseEvent (MOVE or CLICK)
            
        Returns:
            Feature dictionary if stroke completes, None otherwise
        """
        features = None
        
        # Check for PAUSE trigger (time since last event)
        if self._last_event is not None:
            time_gap = event.timestamp - self._last_event.timestamp
            if time_gap > PAUSE_THRESHOLD_MS and len(self._segment_buffer) > 0:
                # Pause detected - flush current stroke
                if DEBUG:
                    print(f"[MOUSE PROCESSOR] PAUSE detected ({time_gap:.0f}ms)")
                features = self._flush_stroke("PAUSE")
        
        # Check for CLICK trigger
        if event.event_type == MouseEventType.CLICK:
            # Try to add final segment before flushing
            if self._last_event is not None:
                segment = self._try_create_segment(self._last_event, event)
                if segment is not None:
                    self._segment_buffer.append(segment)
            
            if len(self._segment_buffer) > 0:
                if DEBUG:
                    print(f"[MOUSE PROCESSOR] CLICK detected")
                features = self._flush_stroke("CLICK")
            
            self._last_event = event
            return features
        
        # MOVE event - add to buffer
        if self._last_event is not None:
            segment = self._try_create_segment(self._last_event, event)
            if segment is not None:
                self._segment_buffer.append(segment)
        
        self._event_buffer.append(event)
        self._last_event = event
        
        return features
    
    def _flush_stroke(self, trigger: str) -> Optional[Dict[str, float]]:
        """
        Validate and extract features from the current stroke buffer.
        
        Args:
            trigger: What triggered the flush ("CLICK", "PAUSE", "DRAG")
            
        Returns:
            Feature dictionary if valid stroke, None otherwise
        """
        segments = self._segment_buffer
        
        # Validate stroke
        if len(segments) < MIN_STROKE_EVENTS:
            if DEBUG:
                print(f"[MOUSE PROCESSOR] Stroke rejected: only {len(segments)} segments (need {MIN_STROKE_EVENTS})")
            self._clear_buffers()
            return None
        
        # Calculate total path distance
        path_distance = sum(seg.distance for seg in segments)
        if path_distance < MIN_STROKE_DISTANCE:
            if DEBUG:
                print(f"[MOUSE PROCESSOR] Stroke rejected: path={path_distance:.1f}px (need {MIN_STROKE_DISTANCE})")
            self._clear_buffers()
            return None
        
        # Extract features
        features = self._extract_features(segments)
        
        self._stroke_count += 1
        if DEBUG:
            print(f"[MOUSE PROCESSOR] ✅ Stroke #{self._stroke_count} emitted ({trigger}, {len(segments)} segments)")
            for k, v in features.items():
                print(f"[MOUSE PROCESSOR]   {k}: {v:.4f}")
        
        self._clear_buffers()
        return features
    
    def _clear_buffers(self) -> None:
        """Clear all buffers for the next stroke."""
        self._event_buffer.clear()
        self._segment_buffer.clear()
    
    def _try_create_segment(self, p1: MouseEvent, p2: MouseEvent) -> Optional[Segment]:
        """
        Create a segment between two points.
        
        Filters for corrupted data:
        - Minimum distance (sub-pixel noise)
        - Minimum/maximum time (hardware noise, gaps)
        - Maximum velocity (teleports/corrupted timestamps)
        
        Returns:
            Segment if valid, None if filtered
        """
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        distance = math.sqrt(dx * dx + dy * dy)
        time_diff = p2.timestamp - p1.timestamp
        
        # Filter: minimum distance (noise reduction)
        if distance < MIN_SEGMENT_DISTANCE:
            return None
        
        # Filter: corrupted timestamps
        if time_diff < MIN_SEGMENT_TIME_MS or time_diff > MAX_SEGMENT_TIME_MS:
            return None
        
        velocity = distance / time_diff
        
        # Filter: velocity limit (corrupted/teleport)
        if velocity > MAX_VELOCITY_PX_PER_MS:
            return None
        
        angle = math.atan2(dy, dx)
        
        return Segment(
            distance=distance,
            time_diff=time_diff,
            velocity=velocity,
            angle=angle,
            start_point=(p1.x, p1.y),
            end_point=(p2.x, p2.y)
        )
    
    def _extract_features(self, segments: List[Segment]) -> Dict[str, float]:
        """
        Extract behavioral features from a validated stroke.
        
        Features:
        - velocity_mean, velocity_std, velocity_max (p95)
        - angle_mean, angle_std (circular)
        - curvature_mean, curvature_std
        - trajectory_efficiency
        - time_diff_std (temporal consistency - bots have flat dt)
        """
        velocities = [seg.velocity for seg in segments]
        angles = [seg.angle for seg in segments]
        time_diffs = [seg.time_diff for seg in segments]
        
        # Calculate curvatures (change in angle per pixel)
        curvatures = []
        for i in range(1, len(segments)):
            angle_diff = self._angle_diff(segments[i].angle, segments[i-1].angle)
            distance = segments[i].distance
            if distance > 0:
                curvature = abs(angle_diff) / distance
                curvatures.append(curvature)
        
        # Trajectory efficiency
        path_distance = sum(seg.distance for seg in segments)
        start = segments[0].start_point
        end = segments[-1].end_point
        net_distance = math.hypot(end[0] - start[0], end[1] - start[1])
        efficiency = min(1.0, net_distance / path_distance) if path_distance > 0 else 0.0
        
        # Peak velocity - use p95 to ignore single-segment spikes (fix #3)
        # Humans spike briefly; bots sustain
        velocities_sorted = sorted(velocities)
        p95_idx = int(len(velocities_sorted) * 0.95)
        velocity_max = velocities_sorted[min(p95_idx, len(velocities_sorted) - 1)]
        
        # Temporal consistency - std of time intervals (fix #4)
        # Bots often have flat dt; humans have noisy timing
        time_diff_std = self._std(time_diffs)
        
        # Linearity error (perpendicular distance from ideal straight line)
        linearity_error = self._calculate_linearity_error(segments)
        
        return {
            "velocity_mean": self._mean(velocities),
            "velocity_std": self._std(velocities),
            "velocity_max": velocity_max,
            "angle_mean": self._circular_mean(angles),
            "angle_std": self._circular_std(angles),
            "curvature_mean": self._mean(curvatures) if curvatures else 0.0,
            "curvature_std": self._std(curvatures) if curvatures else 0.0,
            "trajectory_efficiency": efficiency,
            "path_distance": path_distance,
            "linearity_error": linearity_error,
            "time_diff_std": time_diff_std,
            "segment_count": len(segments),
        }
    
    def _calculate_linearity_error(self, segments: List[Segment]) -> float:
        """
        Calculate the mean perpendicular distance of all intermediate points
        from the ideal straight line connecting start to end.
        
        Uses the cross product method for point-to-line distance:
        distance = |cross(end - start, point - start)| / |end - start|
        
        Returns:
            Mean perpendicular distance in pixels (linearity error).
            Returns 0.0 if fewer than 3 points (no intermediate points).
        """
        if len(segments) < 2:
            return 0.0
        
        # Collect all points from segments
        points: List[tuple] = [segments[0].start_point]
        for seg in segments:
            points.append(seg.end_point)
        
        if len(points) < 3:
            return 0.0  # Need at least start, middle, end
        
        # Start and end points define the ideal line
        start = np.array(points[0], dtype=np.float64)
        end = np.array(points[-1], dtype=np.float64)
        
        line_vec = end - start
        line_length = np.linalg.norm(line_vec)
        
        if line_length < 1e-9:
            return 0.0  # Degenerate case: start == end
        
        # Calculate perpendicular distances for intermediate points
        distances: List[float] = []
        for point in points[1:-1]:  # Exclude start and end
            p = np.array(point, dtype=np.float64)
            # Cross product in 2D: (end - start) × (point - start)
            # Result is a scalar representing the signed area
            cross = abs(line_vec[0] * (p[1] - start[1]) - line_vec[1] * (p[0] - start[0]))
            distance = cross / line_length
            distances.append(distance)
        
        if not distances:
            return 0.0
        
        return sum(distances) / len(distances)
    
    # =========================================================================
    # MATH UTILITIES
    # =========================================================================
    
    def _mean(self, values: List[float]) -> float:
        """Calculate arithmetic mean."""
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = self._mean(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def _circular_mean(self, angles: List[float]) -> float:
        """
        Calculate circular mean of angles using vector averaging.
        
        Converts angles to unit vectors, averages, then converts back.
        This correctly handles wraparound (e.g., -π and π are neighbors).
        """
        if not angles:
            return 0.0
        sin_sum = sum(math.sin(a) for a in angles)
        cos_sum = sum(math.cos(a) for a in angles)
        return math.atan2(sin_sum, cos_sum)
    
    def _circular_std(self, angles: List[float]) -> float:
        """
        Calculate circular standard deviation using resultant vector length.
        
        R = length of mean resultant vector (0 = chaos, 1 = aligned)
        σ = sqrt(-2 * ln(R))
        """
        if len(angles) < 2:
            return 0.0
        
        sin_sum = sum(math.sin(a) for a in angles)
        cos_sum = sum(math.cos(a) for a in angles)
        R = math.sqrt(sin_sum**2 + cos_sum**2) / len(angles)
        
        # Clamp R to valid range
        R = min(1.0, max(0.0, R))
        
        if R < 1e-9:
            return math.pi  # Maximum dispersion
        if R >= 0.999999:
            return 0.0  # Perfect alignment
        
        return math.sqrt(-2 * math.log(R))
    
    def _angle_diff(self, a1: float, a2: float) -> float:
        """Calculate signed angle difference, handling wraparound."""
        diff = a1 - a2
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff
    
    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    def reset(self) -> None:
        """Reset processor state for a new session."""
        self._event_buffer.clear()
        self._segment_buffer.clear()
        self._last_event = None
        self._stroke_count = 0
        
        if DEBUG:
            print("[MOUSE PROCESSOR] State reset")
    
    def get_stroke_count(self) -> int:
        """Return the number of strokes processed."""
        return self._stroke_count
