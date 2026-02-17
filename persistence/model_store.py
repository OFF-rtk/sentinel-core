"""
Sentinel Model Store

Supabase-based persistence for behavioral models.
Uses the user_behavior_models table.

Supports two model types:
- keyboard_hst: HST model for session anomaly detection (per-user persistent)
- keyboard_identity: Identity model for continuity detection
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import pickle
import threading
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from supabase import create_client, Client


logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Model types stored in Supabase."""
    HST = "keyboard_hst"
    IDENTITY = "keyboard_identity"


@dataclass
class StoredModel:
    """Model data from Supabase."""
    model: Any
    feature_window_count: int
    model_version: int
    user_id: str
    model_type: ModelType


class ModelStore:
    """
    Supabase-based model persistence with optimistic locking.
    
    Supports:
    - HST model (per-user persistent anomaly detection)
    - Identity model (behavioral continuity)
    - Checksum verification
    - Retry loop for concurrent updates
    """
    
    TABLE_NAME = "user_behavior_models"
    MAX_RETRIES = 3
    
    # Per-user locks to serialize learn_with_retry calls
    _learn_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
    _lock_guard = threading.Lock()  # Protects _learn_locks dict itself
    
    def __init__(self) -> None:
        """Initialize Supabase client from environment."""
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        
        if not url or not key:
            logger.warning("Supabase credentials not configured, model store disabled")
            self.client: Optional[Client] = None
        else:
            self.client = create_client(url, key)
            logger.info("ModelStore initialized with Supabase")
    
    def load_model(
        self,
        user_id: str,
        model_type: ModelType = ModelType.IDENTITY
    ) -> Optional[StoredModel]:
        """Load model from Supabase with checksum verification."""
        if self.client is None:
            return None
        
        try:
            response = self.client.table(self.TABLE_NAME).select(
                "model_blob, feature_window_count, model_version, checksum"
            ).eq("user_id", user_id).eq("model_type", model_type.value).execute()
            
            if not response.data:
                logger.debug(f"No {model_type.value} model for user {user_id}")
                return None
            
            row = response.data[0]
            blob = row["model_blob"]
            stored_checksum = row.get("checksum")
            
            if isinstance(blob, str):
                # Validate base64 integrity before decoding
                if len(blob) % 4 != 0:
                    logger.error(
                        f"Corrupted blob for {user_id}/{model_type.value}: "
                        f"base64 length {len(blob)} not divisible by 4. "
                        f"Model will be rebuilt from scratch on next ALLOW."
                    )
                    return None
                blob = base64.b64decode(blob)
            
            if stored_checksum:
                computed = hashlib.sha256(blob).hexdigest()
                if computed != stored_checksum:
                    logger.error(f"Checksum mismatch for {user_id}/{model_type.value}")
                    return None
            
            model = pickle.loads(blob)
            
            return StoredModel(
                model=model,
                feature_window_count=row["feature_window_count"],
                model_version=row["model_version"],
                user_id=user_id,
                model_type=model_type
            )
            
        except Exception as e:
            logger.error(f"Failed to load {model_type.value} for {user_id}: {e}")
            return None
    
    def save_model(
        self,
        user_id: str,
        model: Any,
        feature_window_count: int,
        model_type: ModelType = ModelType.IDENTITY,
        expected_version: Optional[int] = None
    ) -> bool:
        """Save model with optimistic locking."""
        if self.client is None:
            return False
        
        try:
            blob = pickle.dumps(model)
            checksum = hashlib.sha256(blob).hexdigest()
            encoded_blob = base64.b64encode(blob).decode("utf-8")
            
            # Sanity check: base64 must always be divisible by 4
            if len(encoded_blob) % 4 != 0:
                logger.error(
                    f"Base64 encoding produced invalid length {len(encoded_blob)} "
                    f"for {user_id}/{model_type.value}. Aborting save."
                )
                return False
            
            record = {
                "user_id": user_id,
                "model_type": model_type.value,
                "model_blob": encoded_blob,
                "feature_window_count": feature_window_count,
                "checksum": checksum,
                "last_trained_at": "now()",
                "updated_at": "now()"
            }
            
            if expected_version is not None:
                record["model_version"] = expected_version + 1
                response = self.client.table(self.TABLE_NAME).update(record).eq(
                    "user_id", user_id
                ).eq("model_type", model_type.value).eq(
                    "model_version", expected_version
                ).execute()
                
                if not response.data:
                    logger.warning(f"Version conflict {user_id}/{model_type.value}")
                    return False
            else:
                record["model_version"] = 1
                record["created_at"] = "now()"
                self.client.table(self.TABLE_NAME).upsert(
                    record,
                    on_conflict="user_id,model_type"
                ).execute()
            
            logger.debug(f"Saved {model_type.value} for {user_id}, windows={feature_window_count}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save {model_type.value} for {user_id}: {e}")
            return False
    
    def _get_learn_lock(self, user_id: str, model_type: ModelType) -> threading.Lock:
        """Get or create a per-user per-model lock."""
        lock_key = f"{user_id}:{model_type.value}"
        with self._lock_guard:
            return self._learn_locks[lock_key]
    
    def learn_with_retry(
        self,
        user_id: str,
        model_type: ModelType,
        learn_fn: Callable[[Any], None],
        create_model_fn: Callable[[], Any],
        window_increment: int = 1
    ) -> bool:
        """
        Load-train-save with retry loop for concurrent safety.
        
        Uses a per-user per-model threading lock to serialize access.
        This prevents concurrent load-train-save cycles from producing
        corrupted blobs due to partial reads during writes.
        
        Args:
            user_id: User identifier
            model_type: HST or IDENTITY
            learn_fn: Function to update model (receives model object)
            create_model_fn: Factory for new models if none exists
            window_increment: Number of windows being added
        
        Returns:
            True if learning succeeded
        """
        lock = self._get_learn_lock(user_id, model_type)
        
        # Non-blocking acquire: if another thread is already learning
        # for this user+model, skip this cycle entirely instead of
        # queueing up (the next stream batch will pick it up).
        acquired = lock.acquire(blocking=False)
        if not acquired:
            logger.debug(
                f"Skipping {model_type.value} learning for {user_id}: "
                f"another thread is already learning"
            )
            return False
        
        try:
            for attempt in range(self.MAX_RETRIES):
                stored = self.load_model(user_id, model_type)
                
                if stored is None:
                    # Create new model
                    model = create_model_fn()
                    learn_fn(model)
                    if self.save_model(user_id, model, window_increment, model_type, None):
                        return True
                else:
                    # Update existing
                    learn_fn(stored.model)
                    new_count = stored.feature_window_count + window_increment
                    if self.save_model(
                        user_id,
                        stored.model,
                        new_count,
                        model_type,
                        stored.model_version
                    ):
                        return True
                
                logger.debug(f"Retry {attempt + 1} for {model_type.value}/{user_id}")
            
            logger.warning(f"Failed to learn after {self.MAX_RETRIES} retries")
            return False
        finally:
            lock.release()
    
    def get_sample_count(
        self,
        user_id: str,
        model_type: ModelType = ModelType.IDENTITY
    ) -> int:
        """Get feature_window_count without loading full model."""
        if self.client is None:
            return 0
        
        try:
            response = self.client.table(self.TABLE_NAME).select(
                "feature_window_count"
            ).eq("user_id", user_id).eq("model_type", model_type.value).execute()
            
            if response.data:
                return response.data[0]["feature_window_count"]
            return 0
            
        except Exception as e:
            logger.warning(f"Failed to get sample count: {e}")
            return 0
