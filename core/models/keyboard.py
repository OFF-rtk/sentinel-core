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


# Debug flag
DEBUG = True


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
    Provides Z-Score based feature attribution for anomaly vectors.
    
    Attributes:
        _scaler: RawMinMaxScaler with fixed bounds for keystroke features.
        _detector: HalfSpaceTrees anomaly detector.
        _feature_stats: Running variance trackers per feature for Z-Score.
    """
    
    # Thresholds
    _ANOMALY_THRESHOLD: float = 0.6
    _ZSCORE_THRESHOLD: float = 2.0
    
    def __init__(self) -> None:
        """Initialize the keyboard anomaly detection pipeline."""
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
        
        if DEBUG:
            print(f"[MODEL] KeyboardAnomalyModel initialized")
            print(f"[MODEL]   n_trees=100, height=6, window_size=80")
            print(f"[MODEL]   ANOMALY_THRESHOLD={self._ANOMALY_THRESHOLD}")
    
    def score_one(self, features: Dict[str, float]) -> Tuple[float, List[str]]:
        """
        Score a single observation for anomaly with feature attribution.
        
        Args:
            features: Dictionary of keystroke features (e.g., dwell_time, 
                      flight_time, std_dwell, std_flight, error_rate)
        
        Returns:
            Tuple of (anomaly_score, anomaly_vectors):
                - anomaly_score: float between 0.0 (normal) and 1.0 (anomalous)
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
        
        risk_score = self._detector.score_one(scaled_features)
        
        if DEBUG:
            print(f"[MODEL SCORE] ⚡ HalfSpaceTrees raw score: {risk_score:.6f}")
            print(f"[MODEL SCORE]   Model has seen {self._learn_count} training samples")
        
        vectors: List[str] = []
        
        # Only compute attribution if score indicates anomaly
        if risk_score > self._ANOMALY_THRESHOLD:
            if DEBUG:
                print(f"[MODEL SCORE] Score {risk_score:.4f} > threshold {self._ANOMALY_THRESHOLD}, computing attribution...")
            
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
                print(f"[MODEL SCORE] Score {risk_score:.4f} <= threshold {self._ANOMALY_THRESHOLD}, no attribution")
        
        if DEBUG:
            print(f"[MODEL SCORE] Final result: score={risk_score:.6f}, vectors={vectors}")
        
        return (risk_score, vectors)
    
    def learn_one(self, features: Dict[str, float]) -> None:
        """
        Update the model with a single observation.
        
        Args:
            features: Dictionary of keystroke features to learn from.
        """
        self._learn_count += 1
        
        if DEBUG:
            print(f"\n[MODEL LEARN] ========== learn_one #{self._learn_count} ==========")
            print(f"[MODEL LEARN] Input features:")
            for k, v in features.items():
                print(f"[MODEL LEARN]   {k}: {v:.4f}")
        
        # Scale features using hardcoded bounds and update detector
        scaled_features = self._scaler.transform_one(features)
        
        if DEBUG:
            print(f"[MODEL LEARN] Scaled features for detector:")
            for k, v in scaled_features.items():
                print(f"[MODEL LEARN]   {k}: {v:.4f}")
        
        self._detector.learn_one(scaled_features)
        
        if DEBUG:
            print(f"[MODEL LEARN] ✓ HalfSpaceTrees updated (total samples: {self._learn_count})")
        
        # Update feature statistics
        for feature_name, value in features.items():
            if feature_name not in self._feature_stats:
                self._feature_stats[feature_name] = Var()
            self._feature_stats[feature_name].update(value)
        
        if DEBUG:
            print(f"[MODEL LEARN] Feature stats updated:")
            for k, stat in self._feature_stats.items():
                print(f"[MODEL LEARN]   {k}: mean={stat.mean.get():.4f}, var={stat.get():.4f}")
