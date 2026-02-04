"""
Sentinel Session Repository

Redis-based session state management with atomic operations.
Implements all security, consistency, and reliability guarantees.

Key Schemas:
    SESSION:{session_id}         → Session state JSON
    KEYBOARD_STATE:{session_id}  → Keyboard buffer JSON
    MOUSE_STATE:{session_id}     → Mouse buffer JSON
    RATE:{session_id}:{second}   → Rate limit counter
    EVAL_DEDUP:{eval_id}         → Idempotency marker
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable

from redis.exceptions import RedisError, WatchError
from .connection import get_redis_client


logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class SessionState:
    """Session state stored in Redis."""
    mode: str = "NORMAL"
    strikes: float = 0.0
    last_activity_ts: float = 0.0      # Any activity (stream/eval)
    last_verified_ts: float = 0.0       # Evaluate only (for trust decay)
    last_keyboard_batch_id: int = 0
    last_mouse_batch_id: int = 0
    learning_suspended_until: float = 0.0
    last_clean_activity_ts: Optional[float] = None
    consecutive_allows: int = 0
    last_strike_decay_ts: float = 0.0
    challenge_entered_ts: float = 0.0
    last_decision: Optional[str] = None
    last_risk: Optional[float] = None
    last_eval_ts: Optional[float] = None
    last_eval_id: Optional[str] = None
    
    # Trust system fields
    trust_score: float = 0.0
    last_context_change_ts: float = 0.0
    
    # Keyboard confidence fields
    keyboard_window_count: int = 0
    keyboard_first_window_ts: float = 0.0
    
    # Identity cold-start
    identity_ready: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SessionState:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class KeyboardState:
    """Keyboard behavioral state stored in Redis."""
    completed_windows: List[Dict[str, Any]] = field(default_factory=list)
    pending_events: List[Dict[str, Any]] = field(default_factory=list)
    last_score: float = 0.0
    last_event_ts: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KeyboardState:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MouseState:
    """Mouse behavioral state stored in Redis."""
    completed_strokes: List[Dict[str, Any]] = field(default_factory=list)
    pending_events: List[Dict[str, Any]] = field(default_factory=list)
    last_score: float = 0.0
    last_event_ts: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MouseState:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Repository
# =============================================================================

class SessionRepository:
    """
    Redis-based session state repository with atomic operations.
    
    Implements:
    - Multi-key atomicity via WATCH/MULTI/EXEC
    - Sliding TTL (30 minutes)
    - Rate limiting via Redis counters
    - Eval idempotency via deduplication keys
    """
    
    SESSION_TTL: int = 1800  # 30 minutes
    MAX_RETRIES: int = 5
    BATCH_TOLERANCE: int = 3
    STREAM_RATE_LIMIT: int = 20  # per second
    EVAL_RATE_LIMIT: int = 10   # per second
    MAX_PENDING_EVENTS: int = 50
    MAX_COMPLETED_ITEMS: int = 20
    
    def __init__(self) -> None:
        self.client = get_redis_client()
    
    # -------------------------------------------------------------------------
    # Key Builders
    # -------------------------------------------------------------------------
    
    def _session_key(self, session_id: str) -> str:
        return f"SESSION:{session_id}"
    
    def _keyboard_key(self, session_id: str) -> str:
        return f"KEYBOARD_STATE:{session_id}"
    
    def _mouse_key(self, session_id: str) -> str:
        return f"MOUSE_STATE:{session_id}"
    
    def _rate_key(self, session_id: str, prefix: str = "RATE") -> str:
        return f"{prefix}:{session_id}:{int(time.time())}"
    
    def _eval_dedup_key(self, eval_id: str) -> str:
        return f"EVAL_DEDUP:{eval_id}"
    
    # -------------------------------------------------------------------------
    # Session Operations
    # -------------------------------------------------------------------------
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session state, returns None if expired/missing."""
        try:
            data = self.client.get(self._session_key(session_id))
            if data is None:
                return None
            return SessionState.from_dict(json.loads(data))
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    def get_or_create_session(self, session_id: str) -> SessionState:
        """Get existing session or create new with defaults."""
        session = self.get_session(session_id)
        if session is None:
            session = SessionState(
                last_activity_ts=time.time() * 1000.0,
                last_strike_decay_ts=time.time() * 1000.0
            )
            self._save_session(session_id, session)
        return session
    
    def _save_session(self, session_id: str, session: SessionState) -> None:
        """Save session with TTL."""
        try:
            self.client.setex(
                self._session_key(session_id),
                self.SESSION_TTL,
                json.dumps(session.to_dict())
            )
        except RedisError as e:
            logger.error(f"Failed to save session {session_id}: {e}")
    
    def refresh_session_ttl(self, session_id: str) -> None:
        """Refresh session TTL (called on /evaluate)."""
        try:
            self.client.expire(self._session_key(session_id), self.SESSION_TTL)
        except RedisError as e:
            logger.warning(f"Failed to refresh TTL {session_id}: {e}")
    
    # -------------------------------------------------------------------------
    # Keyboard State Operations
    # -------------------------------------------------------------------------
    
    def get_keyboard_state(self, session_id: str) -> KeyboardState:
        """Get keyboard state or defaults."""
        try:
            data = self.client.get(self._keyboard_key(session_id))
            if data is None:
                return KeyboardState()
            return KeyboardState.from_dict(json.loads(data))
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to get keyboard state {session_id}: {e}")
            return KeyboardState()
    
    def reset_keyboard_state(self, session_id: str) -> None:
        """Reset keyboard buffer (on batch gap)."""
        try:
            self.client.delete(self._keyboard_key(session_id))
        except RedisError as e:
            logger.warning(f"Failed to reset keyboard state {session_id}: {e}")
    
    # -------------------------------------------------------------------------
    # Mouse State Operations
    # -------------------------------------------------------------------------
    
    def get_mouse_state(self, session_id: str) -> MouseState:
        """Get mouse state or defaults."""
        try:
            data = self.client.get(self._mouse_key(session_id))
            if data is None:
                return MouseState()
            return MouseState.from_dict(json.loads(data))
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to get mouse state {session_id}: {e}")
            return MouseState()
    
    def reset_mouse_state(self, session_id: str) -> None:
        """Reset mouse buffer (on batch gap)."""
        try:
            self.client.delete(self._mouse_key(session_id))
        except RedisError as e:
            logger.warning(f"Failed to reset mouse state {session_id}: {e}")
    
    # -------------------------------------------------------------------------
    # Atomic Multi-Key Operations
    # -------------------------------------------------------------------------
    
    def update_keyboard_stream_atomic(
        self,
        session_id: str,
        session_updates: Callable[[SessionState], SessionState],
        keyboard_state: KeyboardState
    ) -> bool:
        """
        Atomically update SESSION + KEYBOARD_STATE.
        
        Uses WATCH/MULTI/EXEC with retry on conflict.
        """
        session_key = self._session_key(session_id)
        keyboard_key = self._keyboard_key(session_id)
        
        for attempt in range(self.MAX_RETRIES):
            try:
                pipe = self.client.pipeline(True)
                pipe.watch(session_key, keyboard_key)
                
                # Load current session
                raw = self.client.get(session_key)
                if raw is None:
                    session = SessionState(
                        last_activity_ts=time.time() * 1000.0,
                        last_strike_decay_ts=time.time() * 1000.0
                    )
                else:
                    session = SessionState.from_dict(json.loads(raw))
                
                # Apply updates
                session = session_updates(session)
                
                # Cap buffers
                keyboard_state.completed_windows = keyboard_state.completed_windows[-self.MAX_COMPLETED_ITEMS:]
                keyboard_state.pending_events = keyboard_state.pending_events[-self.MAX_PENDING_EVENTS:]
                
                # Execute atomic write
                pipe.multi()
                pipe.setex(session_key, self.SESSION_TTL, json.dumps(session.to_dict()))
                pipe.setex(keyboard_key, self.SESSION_TTL, json.dumps(keyboard_state.to_dict()))
                pipe.execute()
                
                return True
                
            except WatchError:
                logger.debug(f"Watch conflict on keyboard update, attempt {attempt + 1}")
                continue
            except RedisError as e:
                logger.error(f"Redis error on keyboard update: {e}")
                return False
        
        logger.warning(f"Max retries exceeded for keyboard update {session_id}")
        return False
    
    def update_mouse_stream_atomic(
        self,
        session_id: str,
        session_updates: Callable[[SessionState], SessionState],
        mouse_state: MouseState
    ) -> bool:
        """
        Atomically update SESSION + MOUSE_STATE.
        """
        session_key = self._session_key(session_id)
        mouse_key = self._mouse_key(session_id)
        
        for attempt in range(self.MAX_RETRIES):
            try:
                pipe = self.client.pipeline(True)
                pipe.watch(session_key, mouse_key)
                
                raw = self.client.get(session_key)
                if raw is None:
                    session = SessionState(
                        last_activity_ts=time.time() * 1000.0,
                        last_strike_decay_ts=time.time() * 1000.0
                    )
                else:
                    session = SessionState.from_dict(json.loads(raw))
                
                session = session_updates(session)
                
                mouse_state.completed_strokes = mouse_state.completed_strokes[-self.MAX_COMPLETED_ITEMS:]
                mouse_state.pending_events = mouse_state.pending_events[-self.MAX_PENDING_EVENTS:]
                
                pipe.multi()
                pipe.setex(session_key, self.SESSION_TTL, json.dumps(session.to_dict()))
                pipe.setex(mouse_key, self.SESSION_TTL, json.dumps(mouse_state.to_dict()))
                pipe.execute()
                
                return True
                
            except WatchError:
                logger.debug(f"Watch conflict on mouse update, attempt {attempt + 1}")
                continue
            except RedisError as e:
                logger.error(f"Redis error on mouse update: {e}")
                return False
        
        logger.warning(f"Max retries exceeded for mouse update {session_id}")
        return False
    
    def update_session_atomic(
        self,
        session_id: str,
        update_fn: Callable[[SessionState], SessionState]
    ) -> Optional[SessionState]:
        """
        Atomically update session state only (for /evaluate).
        """
        session_key = self._session_key(session_id)
        
        for attempt in range(self.MAX_RETRIES):
            try:
                pipe = self.client.pipeline(True)
                pipe.watch(session_key)
                
                raw = self.client.get(session_key)
                if raw is None:
                    session = SessionState(
                        last_activity_ts=time.time() * 1000.0,
                        last_strike_decay_ts=time.time() * 1000.0
                    )
                else:
                    session = SessionState.from_dict(json.loads(raw))
                
                session = update_fn(session)
                
                pipe.multi()
                pipe.setex(session_key, self.SESSION_TTL, json.dumps(session.to_dict()))
                pipe.execute()
                
                return session
                
            except WatchError:
                logger.debug(f"Watch conflict on session update, attempt {attempt + 1}")
                continue
            except RedisError as e:
                logger.error(f"Redis error on session update: {e}")
                return None
        
        logger.warning(f"Max retries exceeded for session update {session_id}")
        return None
    
    # -------------------------------------------------------------------------
    # Rate Limiting
    # -------------------------------------------------------------------------
    
    def check_stream_rate_limit(self, session_id: str) -> bool:
        """Check rate limit for stream endpoints (20/sec)."""
        return self._check_rate_limit(session_id, "STREAM_RATE", self.STREAM_RATE_LIMIT)
    
    def check_eval_rate_limit(self, session_id: str) -> bool:
        """Check rate limit for evaluate endpoint (10/sec)."""
        return self._check_rate_limit(session_id, "EVAL_RATE", self.EVAL_RATE_LIMIT)
    
    def _check_rate_limit(self, session_id: str, prefix: str, limit: int) -> bool:
        """Redis-based sliding window rate limit."""
        key = f"{prefix}:{session_id}:{int(time.time())}"
        try:
            count = self.client.incr(key)
            if count == 1:
                self.client.expire(key, 2)  # Auto-cleanup
            return count <= limit
        except RedisError as e:
            logger.warning(f"Rate limit check failed: {e}")
            return True  # Fail open
    
    # -------------------------------------------------------------------------
    # Idempotency
    # -------------------------------------------------------------------------
    
    def is_eval_processed(self, eval_id: str) -> bool:
        """Check if eval_id was already processed."""
        if not eval_id:
            return False
        try:
            return self.client.exists(self._eval_dedup_key(eval_id)) > 0
        except RedisError as e:
            logger.warning(f"Idempotency check failed: {e}")
            return False
    
    def mark_eval_processed(self, eval_id: str) -> None:
        """Mark eval_id as processed with 60s TTL."""
        if not eval_id:
            return
        try:
            self.client.setex(self._eval_dedup_key(eval_id), 60, "1")
        except RedisError as e:
            logger.warning(f"Failed to mark eval processed: {e}")
    
    def get_cached_eval_response(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get last evaluation response for idempotent replay."""
        session = self.get_session(session_id)
        if session and session.last_decision:
            return {
                "decision": session.last_decision,
                "risk": session.last_risk,
                "mode": session.mode
            }
        return None
