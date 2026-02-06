"""
Sentinel Keyboard Anomaly Model

River-based online learning anomaly detection for keystroke dynamics.
Uses Half-Space Trees for unsupervised anomaly scoring with Z-Score
feature attribution for explainability.
"""

from typing import Dict, List, Tuple

from river.anomaly import HalfSpaceTrees
from river.base import Transformer
from river.stats import Var


# Debug flag - set to True for verbose output during development
DEBUG = False


class RawMinMaxScaler(Transformer):
    """
    Custom Min-Max scaler with hardcoded bounds for keystroke features.
    
    Clips input values to predefined bounds and scales to [0, 1] range.
    This ensures consistent normalization regardless of training data,
    using research-backed bounds for keystroke dynamics.
    
    Bounds (matching KeyboardProcessor output):
        - dwell_time_mean: [0, 500] ms
        - dwell_time_std: [0, 150] ms
        - flight_time_mean: [-100, 1200] ms
        - flight_time_std: [0, 400] ms
        - error_rate: [0.0, 0.3]
    """
    
    # Hardcoded bounds: (min, max) for each feature
    # Names match KeyboardProcessor output: dwell_time_mean, flight_time_mean, etc.
    # Wider bounds = less sensitive to normal human variation
    _BOUNDS: Dict[str, Tuple[float, float]] = {
        "dwell_time_mean": (0.0, 500.0),       # Typical: 60-200ms, slow: up to 400ms
        "dwell_time_std": (0.0, 150.0),        # Skilled: 20-40ms, normal: 40-80ms
        "flight_time_mean": (-100.0, 1200.0),  # Rollover: -100ms, slow: up to 1200ms
        "flight_time_std": (0.0, 400.0),       # Typical: 50-200ms, free-text: up to 400ms
        "error_rate": (0.0, 0.3),              # Normal: 2-8%, stressed: up to 30%
    }
    
    def learn_one(self, x: Dict[str, float]) -> "RawMinMaxScaler":
        """No-op: bounds are fixed, no learning required."""
        return self
    
    def transform_one(self, x: Dict[str, float]) -> Dict[str, float]:
        """
        Clip and scale features to [0, 1] using predefined bounds.
        
        Args:
            x: Dictionary of raw feature values.
        
        Returns:
            Dictionary of scaled feature values in [0, 1] range.
        """
        scaled: Dict[str, float] = {}
        
        if DEBUG:
            print(f"\n[MODEL SCALER] transform_one called with {len(x)} features")
        
        for feature_name, value in x.items():
            if feature_name in self._BOUNDS:
                min_val, max_val = self._BOUNDS[feature_name]
                original_value = value
                
                # Clip to bounds
                clipped = max(min_val, min(max_val, value))
                was_clipped = (clipped != value)
                
                # Min-max scale to [0, 1]
                if max_val - min_val == 0:
                    scaled[feature_name] = 0.0
                else:
                    scaled[feature_name] = (clipped - min_val) / (max_val - min_val)
                
                if DEBUG:
                    clip_status = "⚠️ CLIPPED" if was_clipped else "✓"
                    print(f"[MODEL CLIPPAGE] {feature_name}: raw={original_value:.4f} -> clipped={clipped:.4f} -> scaled={scaled[feature_name]:.4f} [{min_val}, {max_val}] {clip_status}")
            else:
                # Unknown feature: pass through unchanged
                scaled[feature_name] = value
                if DEBUG:
                    print(f"[MODEL CLIPPAGE] {feature_name}: UNKNOWN FEATURE, passed through: {value:.4f}")
        
        return scaled


class KeyboardAnomalyModel:
    """
    Anomaly detection model for keyboard biometrics with explainability.
    
    Uses RawMinMaxScaler -> HalfSpaceTrees for online learning anomaly 
    detection on keystroke features. The scaler uses research-backed 
    hardcoded bounds to ensure consistent normalization.
    
    **Streaming Percentile-based Scoring**: Uses River's P² algorithm for
    streaming quantile estimation. Multiple quantile checkpoints (50th, 75th,
    90th, 95th, 99th) are maintained and used to interpolate where a new
    score falls in the learned distribution. This is memory-efficient and
    doesn't require storing full history.
    
    Attributes:
        _scaler: RawMinMaxScaler with fixed bounds for keystroke features.
        _detector: HalfSpaceTrees anomaly detector.
        _feature_stats: Running variance trackers per feature for Z-Score.
        _quantile_estimators: Dict of quantile -> Quantile estimator.
    """
    
    # Thresholds
    _ANOMALY_THRESHOLD: float = 0.6
    _ZSCORE_THRESHOLD: float = 2.0
    
    # HST configuration
    _HST_WINDOW_SIZE: int = 50  # HST returns 0.0 during cold start until window is filled
    
    # Quantile checkpoints for percentile interpolation
    _QUANTILE_CHECKPOINTS: List[float] = [0.5, 0.75, 0.9, 0.95, 0.99]
    _MIN_SAMPLES_FOR_PERCENTILE: int = 20  # Minimum samples after cold start
    
    def __init__(self) -> None:
        """Initialize the keyboard anomaly detection pipeline."""
        from river.stats import Quantile
        
        self._scaler: RawMinMaxScaler = RawMinMaxScaler()
        self._detector: HalfSpaceTrees = HalfSpaceTrees(
            n_trees=100,
            height=6,
            window_size=50,
            seed=42
        )
        self._feature_stats: Dict[str, Var] = {}
        self._learn_count: int = 0
        self._score_count: int = 0
        
        # Streaming quantile estimators for percentile calculation (P² algorithm)
        self._quantile_estimators: Dict[float, Quantile] = {
            q: Quantile(q) for q in self._QUANTILE_CHECKPOINTS
        }
        
        if DEBUG:
            print(f"[MODEL] KeyboardAnomalyModel initialized")
            print(f"[MODEL]   n_trees=100, height=6, window_size=50")
            print(f"[MODEL]   ANOMALY_THRESHOLD={self._ANOMALY_THRESHOLD}")
            print(f"[MODEL]   Streaming quantile checkpoints: {self._QUANTILE_CHECKPOINTS}")
    
    def score_one(self, features: Dict[str, float]) -> Tuple[float, List[str]]:
        """
        Score a single observation for anomaly with feature attribution.
        
        Uses percentile-based scoring: the raw HST score is converted to a
        percentile rank based on scores seen during learning. This makes 
        anomaly detection relative to the user's baseline.
        
        Args:
            features: Dictionary of keystroke features (e.g., dwell_time, 
                      flight_time, std_dwell, std_flight, error_rate)
        
        Returns:
            Tuple of (anomaly_score, anomaly_vectors):
                - anomaly_score: Percentile rank [0.0, 1.0] where higher = more anomalous
                - anomaly_vectors: List of feature attribution tags (e.g., 
                  "dwell_time_high", "flight_time_low")
        """
        self._score_count += 1
        
        if DEBUG:
            print(f"\n[MODEL SCORE] ========== score_one #{self._score_count} ==========")
            print(f"[MODEL SCORE] Input features:")
            for k, v in features.items():
                print(f"[MODEL SCORE]   {k}: {v:.4f}")
        
        # Scale features using hardcoded bounds
        scaled_features = self._scaler.transform_one(features)
        
        if DEBUG:
            print(f"[MODEL SCORE] Scaled features for detector:")
            for k, v in scaled_features.items():
                print(f"[MODEL SCORE]   {k}: {v:.4f}")
        
        raw_score = self._detector.score_one(scaled_features)
        
        if DEBUG:
            print(f"[MODEL SCORE] ⚡ HalfSpaceTrees raw score: {raw_score:.6f}")
            print(f"[MODEL SCORE]   Model has seen {self._learn_count} training samples")
        
        # Convert raw score to percentile-based risk score
        risk_score = self._compute_percentile_risk(raw_score)
        
        if DEBUG:
            print(f"[MODEL SCORE] 📊 Percentile risk score: {risk_score:.4f}")
        
        vectors: List[str] = []
        
        # Only compute attribution if percentile score indicates anomaly
        if risk_score > self._ANOMALY_THRESHOLD:
            if DEBUG:
                print(f"[MODEL SCORE] Percentile {risk_score:.4f} > threshold {self._ANOMALY_THRESHOLD}, computing attribution...")
            
            for feature_name, value in features.items():
                stat = self._feature_stats.get(feature_name)
                if stat is None:
                    if DEBUG:
                        print(f"[MODEL SCORE]   {feature_name}: no stats yet, skipping")
                    continue
                
                # Avoid division by zero
                sigma = stat.get() ** 0.5  # Var.get() returns variance, so sqrt for std
                if sigma <= 0:
                    if DEBUG:
                        print(f"[MODEL SCORE]   {feature_name}: sigma=0, skipping")
                    continue
                
                z_score = (value - stat.mean.get()) / sigma
                
                if DEBUG:
                    print(f"[MODEL SCORE]   {feature_name}: z_score={z_score:.4f} (mean={stat.mean.get():.4f}, std={sigma:.4f})")
                
                if z_score > self._ZSCORE_THRESHOLD:
                    vectors.append(f"{feature_name}_high")
                    if DEBUG:
                        print(f"[MODEL SCORE]     -> Added {feature_name}_high")
                elif z_score < -self._ZSCORE_THRESHOLD:
                    vectors.append(f"{feature_name}_low")
                    if DEBUG:
                        print(f"[MODEL SCORE]     -> Added {feature_name}_low")
        else:
            if DEBUG:
                print(f"[MODEL SCORE] Percentile {risk_score:.4f} <= threshold {self._ANOMALY_THRESHOLD}, no attribution")
        
        if DEBUG:
            print(f"[MODEL SCORE] Final result: raw={raw_score:.6f}, percentile={risk_score:.4f}, vectors={vectors}")
        
        return (risk_score, vectors)
    
    def _compute_percentile_risk(self, raw_score: float) -> float:
        """
        Convert raw HST score to percentile-based risk using streaming quantiles.
        
        Uses River's P² algorithm quantile estimators. Interpolates between
        quantile checkpoints to estimate where the score falls in the distribution.
        
        Args:
            raw_score: Raw anomaly score from HalfSpaceTrees
            
        Returns:
            Percentile rank in [0.0, 1.0], higher = more anomalous
        """
        # Need enough samples AFTER HST cold start for valid percentile
        min_required = self._HST_WINDOW_SIZE + self._MIN_SAMPLES_FOR_PERCENTILE
        if self._learn_count < min_required:
            # Cold start: not enough data, use raw score directly
            if DEBUG:
                print(f"[MODEL PERCENTILE] Cold start ({self._learn_count} < {min_required}), using raw score")
            return raw_score
        
        # Get quantile values from estimators
        quantile_values = []
        for q in self._QUANTILE_CHECKPOINTS:
            val = self._quantile_estimators[q].get()
            quantile_values.append((q, val))
        
        if DEBUG:
            print(f"[MODEL PERCENTILE] Quantile checkpoints: {[(q, f'{v:.4f}') for q, v in quantile_values]}")
            print(f"[MODEL PERCENTILE] Raw score to rank: {raw_score:.4f}")
        
        # Find where raw_score falls in the quantile distribution
        # If score is below the lowest quantile checkpoint, extrapolate
        if raw_score <= quantile_values[0][1]:
            # Below 50th percentile - linear interpolation to 0
            q_val = quantile_values[0][1]
            if q_val > 0:
                percentile = (raw_score / q_val) * quantile_values[0][0]
            else:
                percentile = 0.0
        # If score is above the highest quantile checkpoint, extrapolate
        elif raw_score >= quantile_values[-1][1]:
            # Above 99th percentile - cap at 1.0
            percentile = 1.0
        else:
            # Interpolate between checkpoints
            for i in range(len(quantile_values) - 1):
                q1, v1 = quantile_values[i]
                q2, v2 = quantile_values[i + 1]
                if v1 <= raw_score <= v2:
                    # Linear interpolation between q1 and q2
                    if v2 - v1 > 0:
                        t = (raw_score - v1) / (v2 - v1)
                        percentile = q1 + t * (q2 - q1)
                    else:
                        percentile = q1
                    break
            else:
                # Fallback (shouldn't happen)
                percentile = raw_score
        
        if DEBUG:
            print(f"[MODEL PERCENTILE] Estimated percentile: {percentile:.4f}")
        
        return percentile
    

    def learn_one(self, features: Dict[str, float]) -> None:
        """
        Update the model with a single observation.
        
        Also updates streaming quantile estimators for percentile calculation.
        
        Args:
            features: Dictionary of keystroke features to learn from.
        """
        self._learn_count += 1
        
        if DEBUG:
            print(f"\n[MODEL LEARN] ========== learn_one #{self._learn_count} ==========")
            print(f"[MODEL LEARN] Input features:")
            for k, v in features.items():
                print(f"[MODEL LEARN]   {k}: {v:.4f}")
        
        # Scale features using hardcoded bounds
        scaled_features = self._scaler.transform_one(features)
        
        if DEBUG:
            print(f"[MODEL LEARN] Scaled features for detector:")
            for k, v in scaled_features.items():
                print(f"[MODEL LEARN]   {k}: {v:.4f}")
        
        # Score BEFORE learning (to track what "normal" looks like)
        raw_score = self._detector.score_one(scaled_features)
        
        # Update detector with this sample
        self._detector.learn_one(scaled_features)
        
        if DEBUG:
            print(f"[MODEL LEARN] ✓ HalfSpaceTrees updated (total samples: {self._learn_count})")
            print(f"[MODEL LEARN] Sample score before learn: {raw_score:.6f}")
        
        # Update streaming quantile estimators (P² algorithm)
        # Skip cold start period where HST returns 0.0 (pollutes distribution)
        if self._learn_count > self._HST_WINDOW_SIZE:
            for q, estimator in self._quantile_estimators.items():
                estimator.update(raw_score)
            
            if DEBUG:
                quantile_vals = {q: self._quantile_estimators[q].get() for q in self._QUANTILE_CHECKPOINTS}
                print(f"[MODEL LEARN] Quantile estimates: {quantile_vals}")
        elif DEBUG:
            print(f"[MODEL LEARN] Cold start ({self._learn_count}/{self._HST_WINDOW_SIZE}), skipping quantile update")
        
        # Update feature statistics
        for feature_name, value in features.items():
            if feature_name not in self._feature_stats:
                self._feature_stats[feature_name] = Var()
            self._feature_stats[feature_name].update(value)
        
        if DEBUG:
            print(f"[MODEL LEARN] Feature stats updated:")
            for k, stat in self._feature_stats.items():
                print(f"[MODEL LEARN]   {k}: mean={stat.mean.get():.4f}, var={stat.get():.4f}")


