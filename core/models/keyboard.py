"""
Sentinel Keyboard Anomaly Model

River-based online learning anomaly detection for keystroke dynamics.
Uses Half-Space Trees for unsupervised anomaly scoring with Z-Score
feature attribution for explainability.
"""

from typing import Dict, List, Tuple

from river.anomaly import HalfSpaceTrees
from river.compose import Pipeline
from river.preprocessing import MinMaxScaler
from river.stats import Var


class KeyboardAnomalyModel:
    """
    Anomaly detection model for keyboard biometrics with explainability.
    
    Uses a pipeline of MinMaxScaler -> HalfSpaceTrees for
    online learning anomaly detection on keystroke features.
    Provides Z-Score based feature attribution for anomaly vectors.
    
    Attributes:
        _pipeline: River pipeline containing scaler and anomaly detector.
        _feature_stats: Running variance trackers per feature for Z-Score.
    """
    
    # Thresholds
    _ANOMALY_THRESHOLD: float = 0.5
    _ZSCORE_THRESHOLD: float = 2.0
    
    def __init__(self) -> None:
        """Initialize the keyboard anomaly detection pipeline."""
        self._pipeline: Pipeline = Pipeline(
            MinMaxScaler(),
            HalfSpaceTrees(
                n_trees=10,
                height=8,
                window_size=250,
                seed=42
            )
        )
        self._feature_stats: Dict[str, Var] = {}
    
    def score_one(self, features: Dict[str, float]) -> Tuple[float, List[str]]:
        """
        Score a single observation for anomaly with feature attribution.
        
        Args:
            features: Dictionary of keystroke features (e.g., dwell_mean_ms, 
                      flight_mean_ms, wpm, etc.)
        
        Returns:
            Tuple of (anomaly_score, anomaly_vectors):
                - anomaly_score: float between 0.0 (normal) and 1.0 (anomalous)
                - anomaly_vectors: List of feature attribution tags (e.g., 
                  "dwell_mean_ms_high", "wpm_low")
        """
        risk_score = self._pipeline.score_one(features)
        vectors: List[str] = []
        
        # Only compute attribution if score indicates anomaly
        if risk_score > self._ANOMALY_THRESHOLD:
            for feature_name, value in features.items():
                stat = self._feature_stats.get(feature_name)
                if stat is None:
                    continue
                
                # Avoid division by zero
                sigma = stat.get() ** 0.5  # Var.get() returns variance, so sqrt for std
                if sigma <= 0:
                    continue
                
                z_score = (value - stat.mean.get()) / sigma
                
                if z_score > self._ZSCORE_THRESHOLD:
                    vectors.append(f"{feature_name}_high")
                elif z_score < -self._ZSCORE_THRESHOLD:
                    vectors.append(f"{feature_name}_low")
        
        return (risk_score, vectors)
    
    def learn_one(self, features: Dict[str, float]) -> None:
        """
        Update the model with a single observation.
        
        Args:
            features: Dictionary of keystroke features to learn from.
        """
        # Update pipeline
        self._pipeline.learn_one(features)
        
        # Update feature statistics
        for feature_name, value in features.items():
            if feature_name not in self._feature_stats:
                self._feature_stats[feature_name] = Var()
            self._feature_stats[feature_name].update(value)
