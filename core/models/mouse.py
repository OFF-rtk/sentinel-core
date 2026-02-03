"""
Sentinel Physics v2 - Mouse Liveness Detection

Deterministic physics-based detection of bots and automation scripts.
Zero ML, zero learning, zero drift.

Architecture:
    Raw Mouse Events -> MouseProcessor -> PhysicsMouseModel -> (score, reasons)
    
Tiered Detection:
    Tier 1 (HARD FAIL): Impossible physics - immediate 1.0
    Tier 2 (ADDITIVE): Suspicious patterns - accumulate risk
    Tier 3 (DECISION): risk >= 0.7 -> bot
    
Session-Level:
    Strike system: 3 bot strokes = session flagged
    Decay: human strokes reduce strike count
"""

from typing import Dict, List, Tuple

# Debug flag
DEBUG = False


class PhysicsMouseModel:
    """
    Physics-Based Mouse Liveness Detection Model.
    
    A deterministic, Zero Trust approach with tiered scoring:
    
    Tier 1 - Non-Negotiable Physics (HARD FAIL):
        - Teleport speed: velocity_max > 9.0 px/ms
        - Impossible motor rate: time_diff < 4ms repeatedly (handled by processor)
        - Inhuman linearity: path > 300px AND linearity_error < 0.2px
    
    Tier 2 - Suspicious Regularity (ADDITIVE):
        - Overly regular timing: segment_count >= 20 AND time_diff_std < 0.02ms (+0.35)
        - Low velocity jitter: velocity_std < 0.01 (+0.25)
        - Excessive linearity: path > 150px AND linearity_error < 0.5px (+0.25)
    
    Tier 3 - Decision:
        - risk >= 0.7 -> bot (1.0)
        - else -> human (0.0)
    
    This model is stateless and requires no training.
    """
    
    # ==========================================================================
    # TIER 1: HARD FAIL THRESHOLDS (Impossible for humans)
    # ==========================================================================
    TELEPORT_SPEED: float = 9.0           # px/ms - literally impossible
    INHUMAN_PATH_MIN: float = 300.0       # px - long stroke requirement
    INHUMAN_LINEARITY_MAX: float = 0.2    # px - impossibly straight
    
    # ==========================================================================
    # TIER 2: SUSPICIOUS THRESHOLDS (Unlikely for humans)
    # ==========================================================================
    MIN_TIMING_SEGMENTS: int = 20         # segments needed for timing check
    SUSPICIOUS_TIME_STD: float = 0.02     # ms - OS-quantized floor
    SUSPICIOUS_VELOCITY_STD: float = 0.01 # minimum natural jitter
    SUSPICIOUS_PATH_MIN: float = 150.0    # px - linearity check requirement
    SUSPICIOUS_LINEARITY_MAX: float = 0.5 # px - too straight
    
    # Risk weights for Tier 2
    WEIGHT_TIMING: float = 0.35
    WEIGHT_JITTER: float = 0.25
    WEIGHT_LINEARITY: float = 0.25
    
    # Decision threshold
    RISK_THRESHOLD: float = 0.7
    
    def __init__(self) -> None:
        """Initialize the physics model (stateless - no configuration needed)."""
        if DEBUG:
            print("[PHYSICS MODEL] Initialized with Tiered Zero Trust scoring")
            print(f"[PHYSICS MODEL] Tier 1 (Hard Fail):")
            print(f"[PHYSICS MODEL]   TELEPORT_SPEED={self.TELEPORT_SPEED} px/ms")
            print(f"[PHYSICS MODEL]   INHUMAN_LINEARITY: path>{self.INHUMAN_PATH_MIN}px AND error<{self.INHUMAN_LINEARITY_MAX}px")
            print(f"[PHYSICS MODEL] Tier 2 (Additive): threshold={self.RISK_THRESHOLD}")
    
    def score_one(self, features: Dict[str, float]) -> Tuple[float, List[str]]:
        """
        Score a single feature vector using tiered physics detection.
        
        Args:
            features: Dictionary of behavioral features from MouseProcessor.
                      Required: velocity_max, velocity_std, linearity_error, 
                               path_distance, time_diff_std, segment_count
        
        Returns:
            Tuple of (risk_score, anomaly_vectors):
                - risk_score: 0.0 (Human) or 1.0 (Bot)
                - anomaly_vectors: List of violation descriptions
        """
        # Extract features
        velocity_max = features.get("velocity_max", 0.0)
        velocity_std = features.get("velocity_std", 1.0)
        path_distance = features.get("path_distance", 0.0)
        time_diff_std = features.get("time_diff_std", 10.0)
        segment_count = features.get("segment_count", 0)
        linearity_error = features.get("linearity_error", 1.0)
        
        # ======================================================================
        # TIER 1: NON-NEGOTIABLE PHYSICS (HARD FAIL)
        # These are literally impossible for humans. Immediate bot.
        # ======================================================================
        
        # Teleport speed - no human can sustain this
        if velocity_max > self.TELEPORT_SPEED:
            if DEBUG:
                print(f"[PHYSICS MODEL] 🚨 TIER 1 FAIL: teleport_speed ({velocity_max:.2f} > {self.TELEPORT_SPEED})")
            return (1.0, ["teleport_speed"])
        
        # Inhuman linearity on long strokes - humans cannot draw perfect lines
        if path_distance > self.INHUMAN_PATH_MIN and linearity_error < self.INHUMAN_LINEARITY_MAX:
            if DEBUG:
                print(f"[PHYSICS MODEL] 🚨 TIER 1 FAIL: inhuman_linearity ({linearity_error:.3f}px on {path_distance:.0f}px stroke)")
            return (1.0, ["inhuman_linearity"])
        
        # ======================================================================
        # TIER 2: SUSPICIOUS REGULARITY (ADDITIVE)
        # These are possible but unlikely for humans. Bots hit multiple.
        # ======================================================================
        
        risk = 0.0
        reasons: List[str] = []
        
        # Overly regular timing (only on strokes with enough samples)
        if segment_count >= self.MIN_TIMING_SEGMENTS and time_diff_std < self.SUSPICIOUS_TIME_STD:
            risk += self.WEIGHT_TIMING
            reasons.append("overly_regular_timing")
            if DEBUG:
                print(f"[PHYSICS MODEL] ⚠️ Tier 2: overly_regular_timing (+{self.WEIGHT_TIMING}) - dt_std={time_diff_std:.4f}ms")
        
        # Low velocity jitter (constant speed)
        if velocity_std < self.SUSPICIOUS_VELOCITY_STD:
            risk += self.WEIGHT_JITTER
            reasons.append("low_velocity_jitter")
            if DEBUG:
                print(f"[PHYSICS MODEL] ⚠️ Tier 2: low_velocity_jitter (+{self.WEIGHT_JITTER}) - v_std={velocity_std:.4f}")
        
        # Excessive linearity on long-ish strokes
        if path_distance > self.SUSPICIOUS_PATH_MIN and linearity_error < self.SUSPICIOUS_LINEARITY_MAX:
            risk += self.WEIGHT_LINEARITY
            reasons.append("excessive_linearity")
            if DEBUG:
                print(f"[PHYSICS MODEL] ⚠️ Tier 2: excessive_linearity (+{self.WEIGHT_LINEARITY}) - err={linearity_error:.3f}px")
        
        # ======================================================================
        # TIER 3: DECISION
        # ======================================================================
        
        if risk >= self.RISK_THRESHOLD:
            if DEBUG:
                print(f"[PHYSICS MODEL] 🚨 TIER 3 FAIL: risk={risk:.2f} >= {self.RISK_THRESHOLD} | reasons={reasons}")
            return (1.0, reasons)
        
        # All checks passed
        if DEBUG:
            if reasons:
                print(f"[PHYSICS MODEL] ✅ Human (risk={risk:.2f} < {self.RISK_THRESHOLD}) | partial flags: {reasons}")
            else:
                print(f"[PHYSICS MODEL] ✅ Human (clean)")
        return (0.0, [])


class MouseSessionTracker:
    """
    Session-level strike accumulator for mouse liveness detection.
    
    Tracks bot detections across a session to catch:
    - Low-and-slow evasion attempts
    - Bots that try to "look human" occasionally
    
    Strike System:
    - Bot stroke (score=1.0): strikes += 1
    - Human stroke (score=0.0): strikes = max(0, strikes - 1)
    - Session flagged when strikes >= STRIKE_THRESHOLD
    
    Still deterministic, no ML.
    """
    
    STRIKE_THRESHOLD: int = 3  # Number of strikes to flag session as bot
    
    def __init__(self) -> None:
        """Initialize a new session tracker."""
        self._strikes: int = 0
        self._total_strokes: int = 0
        self._bot_strokes: int = 0
        self._session_flagged: bool = False
        self._flag_reasons: List[str] = []
    
    @property
    def strikes(self) -> int:
        """Current strike count."""
        return self._strikes
    
    @property
    def is_flagged(self) -> bool:
        """Whether the session has been flagged as bot."""
        return self._session_flagged
    
    @property
    def stats(self) -> Dict[str, any]:
        """Get session statistics."""
        return {
            "strikes": self._strikes,
            "total_strokes": self._total_strokes,
            "bot_strokes": self._bot_strokes,
            "is_flagged": self._session_flagged,
            "flag_reasons": self._flag_reasons,
        }
    
    def record_stroke(self, score: float, reasons: List[str]) -> Tuple[bool, int]:
        """
        Record a stroke result and update strike count.
        
        Args:
            score: Risk score from PhysicsMouseModel (0.0 or 1.0)
            reasons: List of anomaly vectors from PhysicsMouseModel
        
        Returns:
            Tuple of (session_is_bot, current_strikes)
        """
        self._total_strokes += 1
        
        if score >= 1.0:
            # Bot detected - add strike
            self._strikes += 1
            self._bot_strokes += 1
            self._flag_reasons.extend(reasons)
            
            if DEBUG:
                print(f"[SESSION] ⚠️ Strike {self._strikes}/{self.STRIKE_THRESHOLD} - {reasons}")
            
            # Check if threshold reached
            if self._strikes >= self.STRIKE_THRESHOLD and not self._session_flagged:
                self._session_flagged = True
                if DEBUG:
                    print(f"[SESSION] 🚨 SESSION FLAGGED AS BOT after {self._total_strokes} strokes")
        else:
            # Human behavior - decay strikes
            self._strikes = max(0, self._strikes - 1)
            if DEBUG and self._strikes > 0:
                print(f"[SESSION] ✅ Human stroke - strikes decayed to {self._strikes}")
        
        return (self._session_flagged, self._strikes)
    
    def reset(self) -> None:
        """Reset the session tracker."""
        self._strikes = 0
        self._total_strokes = 0
        self._bot_strokes = 0
        self._session_flagged = False
        self._flag_reasons = []
        if DEBUG:
            print("[SESSION] Reset")
