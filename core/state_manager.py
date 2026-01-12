"""
Sentinel State Manager

This module acts as the thread-safe, in-memory "Hot Storage" for the Sentinel Core.
It persists the User History snapshots required to calculate the 8 "Golden Metrics" 
in the Context Processor.

Usage:
    manager = StateManager()
    history = manager.get_snapshot("usr_123")
    # history is passed to ContextProcessor.derive_context_metrics(request, history)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Set, Tuple


@dataclass
class UserSnapshot:
    """
    Represents the cached state of a user in hot storage.
    
    This dataclass holds all historical data needed by the ContextProcessor
    to derive context metrics for fraud detection.
    """
    
    user_id: str
    """Unique identifier for the user."""
    
    last_seen_timestamp: float
    """Unix timestamp of the last successful action."""
    
    last_geo_coords: Tuple[float, float] = (0.0, 0.0)
    """(latitude, longitude) of the last action."""
    
    known_device_hashes: Set[str] = field(default_factory=set)
    """A set of previously seen JA3/Device hashes."""
    
    avg_transaction_amount: float = 0.0
    """The running average of transaction values."""
    
    transaction_count: int = 0
    """The total count of transactions (used to calculate the running average)."""
    
    usual_hours: List[int] = field(default_factory=list)
    """A list of integers (0-23) representing typical active hours."""
    
    risk_score_history: List[float] = field(default_factory=list)
    """A list storing the last 10 risk scores."""


class StateManager:
    """
    Thread-safe singleton for managing in-memory user state.
    
    This class acts as the "Hot Storage" layer, mocking a production Redis
    instance for user profile caching.
    
    Attributes:
        _instance: Singleton instance reference.
        _lock: Class-level lock for thread-safe singleton creation.
    """
    
    _instance: StateManager | None = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls) -> StateManager:
        """Ensure only one instance of StateManager exists (Singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """
        Initialize the StateManager with pre-loaded mock data.
        
        Uses a flag to prevent re-initialization on subsequent calls.
        """
        # Prevent re-initialization
        if getattr(self, '_initialized', False):
            return
        
        self._store: Dict[str, UserSnapshot] = {}
        self._store_lock: threading.Lock = threading.Lock()
        
        # Pre-load Mock Profile for testing
        self._preload_mock_data()
        
        self._initialized = True
    
    def _preload_mock_data(self) -> None:
        """
        Pre-load mock user profile for out-of-the-box testing.
        
        Mock Data:
            - User: "usr_77252"
            - Last Seen: Current time - 3600 seconds (1 hour ago)
            - Last Geo: (37.7749, -122.4194) - San Francisco, CA
            - Known Device: {"dev_ab4e80f2cbe04656"}
            - Avg Amount: 500.0
            - Usual Hours: [9, 10, 11, 14, 15, 16]
            - Risk History: [0.1, 0.2, 0.05]
        """
        mock_user = UserSnapshot(
            user_id="usr_77252",
            last_seen_timestamp=time.time() - 3600,
            last_geo_coords=(37.7749, -122.4194),
            known_device_hashes={"dev_ab4e80f2cbe04656"},
            avg_transaction_amount=500.0,
            transaction_count=10,  # Assume 10 historical transactions
            usual_hours=[9, 10, 11, 14, 15, 16],
            risk_score_history=[0.1, 0.2, 0.05]
        )
        self._store["usr_77252"] = mock_user
    
    def get_snapshot(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve the user's snapshot as a dictionary.
        
        Args:
            user_id: Unique identifier for the user.
            
        Returns:
            Dictionary containing user snapshot data with keys:
                - user_id
                - last_seen_timestamp
                - last_geo_coords
                - known_device_hashes
                - avg_transaction_amount
                - transaction_count
                - usual_hours
                - risk_score_history
                
        Note:
            If the user_id does not exist (cold start), a new default
            UserSnapshot is created, saved, and returned.
        """
        with self._store_lock:
            if user_id not in self._store:
                # Cold start: create a new default snapshot
                new_snapshot = UserSnapshot(
                    user_id=user_id,
                    last_seen_timestamp=time.time()
                )
                self._store[user_id] = new_snapshot
            
            snapshot = self._store[user_id]
            
            # Convert to dictionary for ContextProcessor compatibility
            return asdict(snapshot)
    
    def update_snapshot(self, user_id: str, updates: Dict[str, Any]) -> None:
        """
        Update the in-memory state for a user.
        
        Args:
            user_id: Unique identifier for the user.
            updates: Dictionary containing fields to update:
                - derived_geo (Tuple[float, float]): New geo coordinates
                - current_device_hash (str): Device hash to add
                - transaction_amount (float): Transaction value (> 0)
                - risk_score (float): Risk score to append
                
        Logic:
            - Always updates last_seen_timestamp to current time
            - If derived_geo provided, updates last_geo_coords
            - If current_device_hash provided, adds to known_device_hashes
            - If transaction_amount > 0, updates avg using cumulative moving average:
                NewAvg = (OldAvg × Count + NewVal) / (Count + 1)
            - If risk_score provided, appends to risk_score_history (max 10 items)
        """
        with self._store_lock:
            # Ensure snapshot exists (cold start handling)
            if user_id not in self._store:
                self._store[user_id] = UserSnapshot(
                    user_id=user_id,
                    last_seen_timestamp=time.time()
                )
            
            snapshot = self._store[user_id]
            
            # Always update last_seen_timestamp
            snapshot.last_seen_timestamp = time.time()
            
            # Update geo coordinates if provided
            if 'derived_geo' in updates and updates['derived_geo'] is not None:
                snapshot.last_geo_coords = updates['derived_geo']
            
            # Add device hash if provided
            if 'current_device_hash' in updates and updates['current_device_hash']:
                snapshot.known_device_hashes.add(updates['current_device_hash'])
            
            # Update transaction average using cumulative moving average formula
            if 'transaction_amount' in updates:
                amount = updates['transaction_amount']
                if amount > 0:
                    old_avg = snapshot.avg_transaction_amount
                    count = snapshot.transaction_count
                    
                    # NewAvg = (OldAvg × Count + NewVal) / (Count + 1)
                    new_avg = (old_avg * count + amount) / (count + 1)
                    
                    snapshot.avg_transaction_amount = new_avg
                    snapshot.transaction_count = count + 1
            
            # Append risk score, keeping only last 10
            if 'risk_score' in updates and updates['risk_score'] is not None:
                snapshot.risk_score_history.append(updates['risk_score'])
                # Keep only the last 10 items
                if len(snapshot.risk_score_history) > 10:
                    snapshot.risk_score_history = snapshot.risk_score_history[-10:]
    
    def reset(self) -> None:
        """
        Reset the store and re-initialize with mock data.
        
        This is primarily useful for testing purposes.
        """
        with self._store_lock:
            self._store.clear()
            self._preload_mock_data()
    
    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance.
        
        This is primarily useful for testing purposes to get a fresh instance.
        """
        with cls._lock:
            cls._instance = None