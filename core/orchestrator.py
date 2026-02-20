"""
Sentinel Orchestrator (v3.1)

Stateless orchestrator with identity continuity and trust system.

Detection Layers:
    Physics → HST → Identity → Trust → Decision

Bug Fixes in v3.1:
- HST model now persistent (per-user in Supabase)
- Separate last_verified_ts for trust decay
- Identity learning with retry loop
- Trust reset on BLOCK decision
- Immature identity guard (CHALLENGE if risk >= 0.98)
- Geometric mean for keyboard confidence
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from core.models import KeyboardAnomalyModel, PhysicsMouseModel, NavigatorPolicyEngine
from core.processors import KeyboardProcessor, MouseProcessor, NavigatorContextProcessor
from core.schemas.inputs import (
    KeyboardStreamPayload,
    MouseStreamPayload,
    EvaluatePayload,
    KeyboardEvent,
    MouseEvent,
)
from core.schemas.outputs import EvaluateResponse, SentinelDecision
from persistence.session_repository import (
    SessionRepository,
    SessionState,
    KeyboardState,
    MouseState,
)
from persistence.model_store import ModelStore, StoredModel, ModelType


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

RISK_EPSILON = 0.001

# Base weights by mode
WEIGHTS = {
    "NORMAL": {"keyboard": 0.70, "mouse": 0.90, "navigator": 1.00, "identity": 0.65},
    "CHALLENGE": {"keyboard": 0.85, "mouse": 1.00, "navigator": 1.00, "identity": 0.85},
}

# Base thresholds by mode
THRESHOLDS = {
    "NORMAL": {"allow": 0.50, "challenge": 0.85},
    "CHALLENGE": {"allow": 0.40, "challenge": 0.75},
}

# Trusted session
TRUSTED_THRESHOLDS = {"allow": 0.60, "challenge": 0.92}
TRUSTED_THRESHOLD = 0.75
TRUSTED_IDENTITY_MULTIPLIER = 0.6
TRUSTED_KEYBOARD_MULTIPLIER = 0.8
TRUSTED_HYSTERESIS_ALLOWS = 3
TRUSTED_HYSTERESIS_TIME = 10.0

# Timing
LEARNING_SUSPENSION_DURATION = 30.0
SUSPENSION_RECOVERY_WINDOW = 60.0
STRIKE_DECAY_INTERVAL = 10.0
MAX_STRIKE_DECAY_INTERVALS = 6
CHALLENGE_HYSTERESIS_TIME = 20.0
CHALLENGE_HYSTERESIS_ALLOWS = 5
DECAY_TIME_CONSTANT = 45.0

# Trust system
TRUST_INACTIVITY_HALFLIFE = 300.0
TRUST_UPDATE_COEFFICIENT = 0.12

# Identity system
IDENTITY_MODEL_SAMPLES_REQUIRED = 150
IDENTITY_MIN_WINDOWS = 3
IDENTITY_MAX_WINDOWS = 5
IDENTITY_CONTRADICTION_THRESHOLD = 0.95
IDENTITY_CONTRADICTION_CONFIDENCE = 0.6
IDENTITY_IMMATURE_GUARD_THRESHOLD = 0.98  # Soft guard when confidence < 0.6

# Keyboard confidence
KEYBOARD_TIME_MATURITY = 20.0
KEYBOARD_COUNT_MATURITY = 15

# Context stability
CONTEXT_STABILITY_DELAY = 30.0


# =============================================================================
# Exceptions
# =============================================================================

class ReplayAttackError(Exception):
    """Raised when a replay attack is detected."""
    pass


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    pass


# =============================================================================
# Orchestrator
# =============================================================================

class SentinelOrchestrator:
    """
    Stateless orchestrator with HST/identity persistence.
    
    HST model is persistent per-user (not in-memory).
    """
    
    def __init__(
        self,
        repo: Optional[SessionRepository] = None,
        model_store: Optional[ModelStore] = None
    ) -> None:
        """Initialize orchestrator."""
        self.repo = repo or SessionRepository()
        self.model_store = model_store or ModelStore()
        
        # Physics model (truly stateless)
        self.mouse_model = PhysicsMouseModel()
        
        # Identity Context & Policy Engine
        self.context_processor = NavigatorContextProcessor()
        self.policy_engine = NavigatorPolicyEngine()
        
        logger.info("SentinelOrchestrator v3.1 initialized (Navigator Integrated)")
    
    # -------------------------------------------------------------------------
    # Keyboard Stream Processing
    # -------------------------------------------------------------------------
    
    def process_keyboard_stream(self, payload: KeyboardStreamPayload) -> None:
        """Process keyboard stream with persistent HST."""
        session = self.repo.get_or_create_session(payload.session_id)
        keyboard_state = self.repo.get_keyboard_state(payload.session_id)
        
        # Validate batch_id (high-water-mark — tolerates out-of-order proxy delivery)
        if payload.batch_id <= session.last_keyboard_batch_id:
            raise ReplayAttackError(
                f"Duplicate/old batch: received {payload.batch_id}, "
                f"last accepted was {session.last_keyboard_batch_id}"
            )
        
        gap = payload.batch_id - session.last_keyboard_batch_id
        if gap > 10:
            logger.warning(
                f"Large keyboard batch gap: last={session.last_keyboard_batch_id}, "
                f"got={payload.batch_id} (gap={gap})"
            )
            keyboard_state = KeyboardState()
            session.strikes += 0.5
            session.keyboard_window_count = 0
            session.keyboard_first_window_ts = 0.0
        elif gap > 1:
            logger.info(
                f"Out-of-order keyboard batch accepted: expected {session.last_keyboard_batch_id + 1}, "
                f"got {payload.batch_id} (gap={gap})"
            )
        
        # Reconstruct processor from pending events, capturing any windows
        # that trigger during replay (e.g. when cumulative count hits 50)
        processor = KeyboardProcessor()
        replay_window_idx = -1  # Track last window position in pending
        features_list = []
        
        for idx, event_dict in enumerate(keyboard_state.pending_events):
            event = KeyboardEvent(**event_dict)
            features = processor.process_event(event)
            if features is not None:
                features_list.append((features, event_dict.get("timestamp", 0)))
                replay_window_idx = idx
        
        # If replay produced windows, trim pending to only keep events after
        # the last window (those events haven't been consumed yet)
        if replay_window_idx >= 0:
            keyboard_state.pending_events = keyboard_state.pending_events[replay_window_idx + 1:]
            logger.info(f"Replay produced {len(features_list)} window(s), "
                        f"{len(keyboard_state.pending_events)} events remain pending")
        
        # Load HST model for scoring (Uses payload.user_id)
        stored_hst = self.model_store.load_model(payload.user_id, ModelType.HST)
        if stored_hst:
            hst_model = stored_hst.model
        else:
            hst_model = KeyboardAnomalyModel()
        
        # Process new events
        new_pending = list(keyboard_state.pending_events)  # Start from trimmed pending
        last_event_ts = keyboard_state.last_event_ts
        now = time.time() * 1000.0  # Milliseconds
        
        for event in payload.events:
            last_event_ts = max(last_event_ts, event.timestamp)
            features = processor.process_event(event)
            new_pending.append(event.model_dump())
            
            if features is not None:
                features_list.append((features, event.timestamp))
        
        # Score each feature window
        for features, event_ts in features_list:
            score, vectors = hst_model.score_one(features)
            
            # Apply decay (frozen during suspension)
            prev_score = keyboard_state.last_score
            if now < session.learning_suspended_until:
                decayed_score = prev_score
            else:
                decayed_score = self._apply_decay(
                    prev_score,
                    keyboard_state.last_event_ts,
                    event_ts
                )
            
            keyboard_state.last_score = max(decayed_score, score)
            keyboard_state.completed_windows.append({
                "features": features,
                "score": score,
                "event_ts": event_ts,
            })
            new_pending = []
            
            # Track first window timestamp
            if session.keyboard_first_window_ts == 0.0:
                session.keyboard_first_window_ts = now
            session.keyboard_window_count += 1
        
        keyboard_state.pending_events = new_pending
        keyboard_state.last_event_ts = last_event_ts
        
        def update_session(s: SessionState) -> SessionState:
            s.last_keyboard_batch_id = payload.batch_id
            s.last_activity_ts = time.time() * 1000.0  # Stream activity
            s.keyboard_window_count = session.keyboard_window_count
            s.keyboard_first_window_ts = session.keyboard_first_window_ts
            return s
        
        self.repo.update_keyboard_stream_atomic(
            payload.session_id,
            update_session,
            keyboard_state
        )
    
    # -------------------------------------------------------------------------
    # Mouse Stream Processing
    # -------------------------------------------------------------------------
    
    def process_mouse_stream(self, payload: MouseStreamPayload) -> None:
        """Process mouse stream."""
        session = self.repo.get_or_create_session(payload.session_id)
        mouse_state = self.repo.get_mouse_state(payload.session_id)
        
        # Validate batch_id (high-water-mark — tolerates out-of-order proxy delivery)
        if payload.batch_id <= session.last_mouse_batch_id:
            raise ReplayAttackError(
                f"Duplicate/old mouse batch: received {payload.batch_id}, "
                f"last accepted was {session.last_mouse_batch_id}"
            )
        
        gap = payload.batch_id - session.last_mouse_batch_id
        if gap > 10:
            logger.warning(
                f"Large mouse batch gap: last={session.last_mouse_batch_id}, "
                f"got={payload.batch_id} (gap={gap})"
            )
            mouse_state = MouseState()
            session.strikes += 0.5
        elif gap > 1:
            logger.info(
                f"Out-of-order mouse batch accepted: expected {session.last_mouse_batch_id + 1}, "
                f"got {payload.batch_id} (gap={gap})"
            )
        
        processor = MouseProcessor()
        for event_dict in mouse_state.pending_events:
            event = MouseEvent(**event_dict)
            processor.process_event(event)
        
        new_pending = []
        last_event_ts = mouse_state.last_event_ts
        
        for event in payload.events:
            last_event_ts = max(last_event_ts, event.timestamp)
            features = processor.process_event(event)
            new_pending.append(event.model_dump())
            
            if features is not None:
                score, vectors = self.mouse_model.score_one(features)
                mouse_state.last_score = max(mouse_state.last_score, score)
                mouse_state.completed_strokes.append({
                    "features": features,
                    "score": score,
                    "event_ts": last_event_ts,
                })
                new_pending = []
        
        mouse_state.pending_events = new_pending
        mouse_state.last_event_ts = last_event_ts
        
        def update_session(s: SessionState) -> SessionState:
            s.last_mouse_batch_id = payload.batch_id
            s.last_activity_ts = time.time() * 1000.0  # Stream activity
            return s
        
        self.repo.update_mouse_stream_atomic(
            payload.session_id,
            update_session,
            mouse_state
        )
    
    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------
    
    def evaluate(self, payload: EvaluatePayload) -> EvaluateResponse:
        """Evaluate session risk with full fusion pipeline."""
        # Idempotency check
        if payload.eval_id and self.repo.is_eval_processed(payload.eval_id):
            cached = self.repo.get_cached_eval_response(payload.session_id)
            if cached:
                return EvaluateResponse(**cached)
        
        session = self.repo.get_session(payload.session_id)
        if session is None:
            logger.warning(f"[EVAL DEBUG] Session {payload.session_id} NOT FOUND in Redis. Returning CHALLENGE 0.5 early.")
            return EvaluateResponse(
                decision=SentinelDecision.CHALLENGE,
                risk=0.5,
                mode="NORMAL"
            )
        logger.info(f"[EVAL DEBUG] Session found: mode={session.mode}, trust={session.trust_score:.3f}, strikes={session.strikes}, consecutive_allows={session.consecutive_allows}")
        
        now = time.time() * 1000.0  # Milliseconds
        self.repo.refresh_session_ttl(payload.session_id)
        
        keyboard_state = self.repo.get_keyboard_state(payload.session_id)
        mouse_state = self.repo.get_mouse_state(payload.session_id)
        
        keyboard_risk_raw = keyboard_state.last_score
        mouse_risk = mouse_state.last_score
        logger.info(f"[EVAL DEBUG] Raw scores: keyboard={keyboard_risk_raw:.4f}, mouse={mouse_risk:.4f}, kb_windows={session.keyboard_window_count}")
        
        # ===== Navigator Risk =====
        # 1. Build EvaluationRequest
        from datetime import datetime, timezone
        from core.schemas.inputs import (
            EvaluationRequest, 
            UserSessionContext, 
            ClientNetworkContext,
            ClientFingerprint
        )
        
        # Convert ms timestamp to datetime
        start_dt = datetime.fromtimestamp(payload.session_start_time / 1000.0, tz=timezone.utc)
        
        eval_request = EvaluationRequest(
            user_session=UserSessionContext(
                user_id=payload.request_context.user_id,
                session_id=payload.session_id,
                role=payload.role,
                session_start_time=start_dt,
                mfa_status=payload.mfa_status
            ),
            business_context=payload.business_context,
            network_context=ClientNetworkContext(
                ip_address=payload.request_context.ip_address,
                user_agent=payload.request_context.user_agent,
                client_fingerprint=payload.client_fingerprint
            )
        )
        
        # 2. Get Metrics
        nav_metrics = self.context_processor.process(eval_request)
        
        # 3. Assess Decision
        nav_analysis = self.policy_engine.evaluate(nav_metrics)
        navigator_risk = nav_analysis.risk_score
        logger.info(f"[EVAL DEBUG] Navigator: risk={navigator_risk:.4f}, decision={nav_analysis.decision}")
        
        # ===== TOFU: Trust On First Use =====
        user_id = payload.request_context.user_id
        trusted_context = self.context_processor.repo.get_trusted_context(user_id)
        
        logger.info(f"[EVAL DEBUG] TOFU check: trusted_context={'EXISTS' if trusted_context else 'NONE'}, navigator_risk={navigator_risk:.4f}")
        if trusted_context is None and navigator_risk == 0.5:
            logger.info(f"TOFU: First session for {user_id}. Trusting explicitly.")
            navigator_risk = 0.0
        
        # Update context change tracking
        if navigator_risk >= 0.5:
            session.last_context_change_ts = now
        
        self._update_learning_suspension(session, navigator_risk)
        self._apply_strike_decay(session)
        
        # ===== Keyboard confidence (geometric mean) =====
        keyboard_risk = self._apply_keyboard_confidence(
            keyboard_risk_raw, session, now
        )
        logger.info(f"[EVAL DEBUG] After confidence: keyboard_risk={keyboard_risk:.4f} (raw was {keyboard_risk_raw:.4f}), first_window_ts={session.keyboard_first_window_ts}, window_count={session.keyboard_window_count}")
        
        # ===== Trust inactivity decay (uses last_verified_ts) =====
        self._apply_trust_inactivity_decay(session, now)
        
        # ===== Trusted session status =====
        is_trusted = session.trust_score >= TRUSTED_THRESHOLD
        logger.info(f"[EVAL DEBUG] Trust: score={session.trust_score:.4f}, is_trusted={is_trusted}, threshold={TRUSTED_THRESHOLD}")
        
        # ===== Identity risk =====
        identity_risk, identity_confidence, cold_start_identity, stored_model = \
            self._compute_identity_risk(payload, keyboard_state, session)
        logger.info(f"[EVAL DEBUG] Identity: risk={identity_risk:.4f}, confidence={identity_confidence:.4f}, cold_start={cold_start_identity}, model={'EXISTS' if stored_model else 'NONE'}")
        
        # Force BLOCK if strikes >= 3
        if session.strikes >= 3:
            session.trust_score = 0  # Reset trust on block
            return self._finalize_evaluate(
                payload, session, SentinelDecision.BLOCK, 1.0,
                keyboard_state, navigator_risk, identity_risk,
                identity_confidence, cold_start_identity, stored_model,
                nav_metrics=nav_metrics
            )
        
        # ===== Hard block priority =====
        
        # Physics violation
        if mouse_risk >= 1.0 - RISK_EPSILON:
            session.trust_score = 0
            return self._finalize_evaluate(
                payload, session, SentinelDecision.BLOCK, 1.0,
                keyboard_state, navigator_risk, identity_risk,
                identity_confidence, cold_start_identity, stored_model,
                nav_metrics=nav_metrics
            )
        
        # Navigator hard block (Engine Decision)
        if nav_analysis.decision == SentinelDecision.BLOCK:
            session.trust_score = 0
            # If Navigator says BLOCK, we respect it fully
            return self._finalize_evaluate(
                payload, session, SentinelDecision.BLOCK, 1.0,
                keyboard_state, navigator_risk, identity_risk,
                identity_confidence, cold_start_identity, stored_model,
                skip_strike_update=False,
                nav_metrics=nav_metrics
            )
        
        # Identity contradiction (mature model)
        if (identity_confidence >= IDENTITY_CONTRADICTION_CONFIDENCE and
                identity_risk >= IDENTITY_CONTRADICTION_THRESHOLD):
            session.trust_score = 0
            return self._finalize_evaluate(
                payload, session, SentinelDecision.BLOCK, 1.0,
                keyboard_state, navigator_risk, identity_risk,
                identity_confidence, cold_start_identity, stored_model,
                nav_metrics=nav_metrics
            )
        
        # Immature identity guard (soft CHALLENGE)
        if (session.identity_ready and 
                identity_confidence < IDENTITY_CONTRADICTION_CONFIDENCE and
                identity_risk >= IDENTITY_IMMATURE_GUARD_THRESHOLD):
            return self._finalize_evaluate(
                payload, session, SentinelDecision.CHALLENGE, identity_risk,
                keyboard_state, navigator_risk, identity_risk,
                identity_confidence, cold_start_identity, stored_model,
                nav_metrics=nav_metrics
            )
        
        # ===== Trusted session modifiers =====
        weights = dict(WEIGHTS[session.mode])
        thresholds = dict(THRESHOLDS[session.mode])
        hysteresis_allows = CHALLENGE_HYSTERESIS_ALLOWS
        hysteresis_time = CHALLENGE_HYSTERESIS_TIME
        
        if is_trusted:
            thresholds = dict(TRUSTED_THRESHOLDS)
            keyboard_risk *= TRUSTED_KEYBOARD_MULTIPLIER
            weights["identity"] *= TRUSTED_IDENTITY_MULTIPLIER
            hysteresis_allows = TRUSTED_HYSTERESIS_ALLOWS
            hysteresis_time = TRUSTED_HYSTERESIS_TIME
        
        # ===== Identity weight scaling (asymmetric) =====
        effective_identity_risk = identity_risk * identity_confidence
        weights["identity"] *= math.sqrt(identity_confidence)
        
        # ===== Weighted MAX fusion =====
        kb_weighted = keyboard_risk * weights["keyboard"]
        ms_weighted = mouse_risk * weights["mouse"]
        nv_weighted = navigator_risk * weights["navigator"]
        id_weighted = effective_identity_risk * weights["identity"]
        
        logger.info(f"[EVAL DEBUG] Weighted risks: kb={kb_weighted:.4f}, mouse={ms_weighted:.4f}, nav={nv_weighted:.4f}, identity={id_weighted:.4f}")
        logger.info(f"[EVAL DEBUG] Weights: {weights} | Thresholds: {thresholds} | Mode: {session.mode}")
        
        final_risk = max(kb_weighted, ms_weighted, nv_weighted, id_weighted)
        final_risk = max(0.0, min(1.0, final_risk))
        
        # ===== Decision =====
        if final_risk < thresholds["allow"]:
            decision = SentinelDecision.ALLOW
        elif final_risk < thresholds["challenge"]:
            decision = SentinelDecision.CHALLENGE
        else:
            decision = SentinelDecision.BLOCK
        
        logger.info(f"[EVAL DEBUG] ★ FINAL: risk={final_risk:.4f}, decision={decision}, allow_threshold={thresholds['allow']}, challenge_threshold={thresholds['challenge']}")
        
        # ===== Cold Start Override =====
        # If HST model is immature (< 50 windows learned), force CHALLENGE
        # on every action that has no fresh typing data. After the user types
        # the challenge text, completed_windows will be non-empty on retry,
        # letting that specific action through as ALLOW.
        stored_hst = self.model_store.load_model(user_id, ModelType.HST)
        hst_cold_start = (stored_hst is None or stored_hst.feature_window_count < 50)
        
        if hst_cold_start and decision == SentinelDecision.ALLOW:
            if not keyboard_state.completed_windows:
                # No fresh typing data for this action → force CHALLENGE
                decision = SentinelDecision.CHALLENGE
                logger.info(f"[COLD START] HST immature (windows={stored_hst.feature_window_count if stored_hst else 0}), forcing CHALLENGE — no fresh windows")
            else:
                # Has windows from challenge typing → ALLOW so HST can learn
                logger.info(f"[COLD START] Allowing: {len(keyboard_state.completed_windows)} fresh windows available, mode={session.mode}")
        
        return self._finalize_evaluate(
            payload, session, decision, final_risk,
            keyboard_state, navigator_risk, identity_risk,
            identity_confidence, cold_start_identity, stored_model,
            hysteresis_allows=hysteresis_allows,
            hysteresis_time=hysteresis_time,
            nav_metrics=nav_metrics
        )
    
    # -------------------------------------------------------------------------
    # Confidence & Identity Helpers
    # -------------------------------------------------------------------------
    
    def _apply_keyboard_confidence(
        self,
        raw_risk: float,
        session: SessionState,
        now: float
    ) -> float:
        """Apply keyboard confidence using geometric mean."""
        # Time-based confidence
        if session.keyboard_first_window_ts > 0:
            duration = now - session.keyboard_first_window_ts
            # KEYBOARD_TIME_MATURITY is typically small (e.g., 20s), we should check if context 
            # implies seconds or ms. Constants in orchestrator are 20.0 etc.
            # If everything is MS now, we need to multiply constants by 1000
            conf_time = min(1.0, duration / (KEYBOARD_TIME_MATURITY * 1000.0))
        else:
            conf_time = 0.0
        
        # Count-based confidence
        conf_count = min(1.0, session.keyboard_window_count / KEYBOARD_COUNT_MATURITY)
        
        # Geometric mean (both matter, neither fully disables)
        if conf_time > 0 and conf_count > 0:
            confidence = math.sqrt(conf_time * conf_count)
        else:
            confidence = 0.0
        
        return raw_risk * confidence
    
    def _apply_trust_inactivity_decay(
        self,
        session: SessionState,
        now: float
    ) -> None:
        """Apply trust decay based on time since last EVALUATE (not stream)."""
        if session.last_verified_ts > 0:
            idle_time = now - session.last_verified_ts
            # TRUST_INACTIVITY_HALFLIFE stored as seconds (300.0). Convert to ms.
            decay = math.exp(-idle_time / (TRUST_INACTIVITY_HALFLIFE * 1000.0))
            session.trust_score *= decay
    
    def _compute_identity_risk(
        self,
        payload: EvaluatePayload,
        keyboard_state: KeyboardState,
        session: SessionState
    ) -> Tuple[float, float, bool, Optional[StoredModel]]:
        """Compute identity risk from stored model."""
        user_id = payload.request_context.user_id
        
        windows = keyboard_state.completed_windows[-IDENTITY_MAX_WINDOWS:]
        if len(windows) < IDENTITY_MIN_WINDOWS:
            return 0.0, 0.0, True, None
        
        stored_model = self.model_store.load_model(user_id, ModelType.IDENTITY)
        if stored_model is None:
            return 0.0, 0.0, True, None
        
        identity_confidence = min(
            1.0,
            stored_model.feature_window_count / IDENTITY_MODEL_SAMPLES_REQUIRED
        )
        
        if identity_confidence == 0:
            return 0.0, 0.0, True, stored_model
        
        scores = []
        for window in windows:
            try:
                score, _ = stored_model.model.score_one(window["features"])
                scores.append(score)
            except Exception as e:
                logger.warning(f"Identity scoring failed: {e}")
        
        if not scores:
            return 0.0, 0.0, True, stored_model
        
        identity_risk = sum(scores) / len(scores)
        identity_risk = min(1.0, max(0.0, identity_risk))
        
        return identity_risk, identity_confidence, False, stored_model
    
    # -------------------------------------------------------------------------
    # Core Helpers
    # -------------------------------------------------------------------------
    
    def _apply_decay(
        self,
        prev_score: float,
        prev_event_ts: float,
        current_event_ts: float
    ) -> float:
        """Apply event-timestamp decay."""
        if prev_event_ts <= 0:
            return prev_score
        
        delta_ms = current_event_ts - prev_event_ts
        # DECAY_TIME_CONSTANT is 45.0 (seconds). We need consistent units.
        # If delta_ms is ms, and constant is seconds, we need to convert constant to ms.
        # exp(-t/T). t and T must match in units.
        # Original code: delta_seconds = delta_ms / 1000.0
        # return prev_score * math.exp(-delta_seconds / DECAY_TIME_CONSTANT)
        # This is correct if DECAY_TIME_CONSTANT is seconds.
        # We can keep it or use pure ms: math.exp(-delta_ms / (DECAY_TIME_CONSTANT * 1000.0))
        # Let's preserve the existing logic which was correct:
        
        delta_seconds = delta_ms / 1000.0
        
        if delta_seconds <= 0:
            return prev_score
        
        return prev_score * math.exp(-delta_seconds / DECAY_TIME_CONSTANT)
    
    def _compute_navigator_risk(self, payload: EvaluatePayload) -> float:
        """Deprecated: Now handled in evaluate() directly via NavigatorPolicyEngine."""
        return 0.1
    
    def _update_learning_suspension(
        self,
        session: SessionState,
        navigator_risk: float
    ) -> None:
        """Update learning suspension state."""
        now = time.time() * 1000.0
        
        # Constants in seconds -> convert to ms
        suspension_dur_ms = LEARNING_SUSPENSION_DURATION * 1000.0
        recovery_win_ms = SUSPENSION_RECOVERY_WINDOW * 1000.0
        
        if navigator_risk >= 0.85:
            session.learning_suspended_until = now + suspension_dur_ms
            session.last_clean_activity_ts = None
        elif navigator_risk < 0.5:
            if session.last_clean_activity_ts is None:
                session.last_clean_activity_ts = now
            elif now - session.last_clean_activity_ts >= recovery_win_ms:
                session.learning_suspended_until = 0
                session.last_clean_activity_ts = now
        else:
            session.last_clean_activity_ts = None
    
    def _apply_strike_decay(self, session: SessionState) -> None:
        """Apply time-based strike decay."""
        now = time.time() * 1000.0
        time_since = now - session.last_strike_decay_ts
        
        decay_interval_ms = STRIKE_DECAY_INTERVAL * 1000.0
        
        intervals = min(int(time_since / decay_interval_ms), MAX_STRIKE_DECAY_INTERVALS)
        
        if intervals > 0:
            session.strikes = max(0, session.strikes - (0.5 * intervals))
            session.last_strike_decay_ts = now
    
    def _update_strikes(
        self,
        session: SessionState,
        decision: SentinelDecision
    ) -> None:
        """Update strikes based on decision."""
        if decision == SentinelDecision.BLOCK:
            session.strikes += 2
            session.consecutive_allows = 0
            session.trust_score = 0  # Always reset trust on block
        elif decision == SentinelDecision.CHALLENGE:
            session.strikes += 1
            session.consecutive_allows = 0
        else:
            session.consecutive_allows += 1
    
    def _update_trust(
        self,
        session: SessionState,
        final_risk: float,
        identity_risk: float
    ) -> None:
        """Update trust using stabilizer formula."""
        if identity_risk >= 0.9:
            session.trust_score = 0
            return
        
        trust_delta = TRUST_UPDATE_COEFFICIENT * (0.5 - final_risk)
        session.trust_score = max(0.0, min(1.0, session.trust_score + trust_delta))
    
    def _update_mode(
        self,
        session: SessionState,
        decision: SentinelDecision,
        hysteresis_allows: int,
        hysteresis_time: float
    ) -> None:
        """Update session mode with hysteresis."""
        now = time.time() * 1000.0
        
        if decision == SentinelDecision.CHALLENGE and session.mode == "NORMAL":
            session.mode = "CHALLENGE"
            session.challenge_entered_ts = now
            session.consecutive_allows = 0
        
        elif session.mode == "CHALLENGE":
            time_in_challenge = now - session.challenge_entered_ts
            # hysteresis_time is in seconds (10.0 or 20.0). Convert to ms.
            if (session.consecutive_allows >= hysteresis_allows and
                    time_in_challenge >= (hysteresis_time * 1000.0)):
                session.mode = "NORMAL"
                session.consecutive_allows = 0
    
    def _should_learn_identity(
        self,
        session: SessionState,
        navigator_risk: float,
        cold_start_identity: bool,
        now: float
    ) -> bool:
        """Check if identity learning is allowed."""
        if session.mode != "NORMAL":
            return False
        if now < session.learning_suspended_until:
            return False
        if navigator_risk >= 0.5:
            return False
        if session.trust_score < 0.65:
            return False
        if session.consecutive_allows < 5:
            return False
        # cold_start_identity guard removed — BUG-002 fix
        # The identity model can never be created if we block learning
        # when no model exists. Other guards are sufficient.
        
        # CONTEXT_STABILITY_DELAY is seconds (30.0) -> ms
        if now - session.last_context_change_ts < (CONTEXT_STABILITY_DELAY * 1000.0):
            return False
        
        return True
    
    def _finalize_evaluate(
        self,
        payload: EvaluatePayload,
        session: SessionState,
        decision: SentinelDecision,
        risk: float,
        keyboard_state: KeyboardState,
        navigator_risk: float,
        identity_risk: float,
        identity_confidence: float,
        cold_start_identity: bool,
        stored_model: Optional[StoredModel],
        skip_strike_update: bool = False,
        hysteresis_allows: int = CHALLENGE_HYSTERESIS_ALLOWS,
        hysteresis_time: float = CHALLENGE_HYSTERESIS_TIME,
        nav_metrics: Optional[Dict] = None
    ) -> EvaluateResponse:
        """Finalize evaluation and persist state."""
        now = time.time() * 1000.0
        user_id = payload.request_context.user_id
        
        # Update strikes
        if not skip_strike_update:
            self._update_strikes(session, decision)
        
        # Update mode
        self._update_mode(session, decision, hysteresis_allows, hysteresis_time)
        
        # Update trust (stabilizer formula)
        self._update_trust(session, risk, identity_risk)
        
        # ===== HST Learning (persistent, per-user) =====
        # Cold start: learn in any mode to bootstrap quickly
        # Post cold start: only learn in NORMAL mode (anti-poisoning)
        hst_for_learning = self.model_store.load_model(user_id, ModelType.HST)
        hst_is_cold = (hst_for_learning is None or hst_for_learning.feature_window_count < 50)
        hst_mode_ok = (session.mode == "NORMAL" or hst_is_cold)
        
        if (decision == SentinelDecision.ALLOW and
                hst_mode_ok and
                now >= session.learning_suspended_until):
            if keyboard_state.completed_windows:
                windows_to_learn = keyboard_state.completed_windows  # Learn ALL windows (BUG-003 fix)
                
                def learn_hst(model: Any) -> None:
                    for window in windows_to_learn:
                        model.learn_one(window["features"])
                
                self.model_store.learn_with_retry(
                    user_id,
                    ModelType.HST,
                    learn_hst,
                    lambda: KeyboardAnomalyModel(),
                    len(windows_to_learn)
                )
                logger.info(f"[HST] Learned {len(windows_to_learn)} windows (cold_start={hst_is_cold}, mode={session.mode})")
                
                # Clear consumed windows so cold start forces fresh CHALLENGE next time
                keyboard_state.completed_windows = []
                keyboard_state.last_score = 0.0
                self.repo._save_keyboard_state(payload.session_id, keyboard_state)
        
        # ===== HST Cold Start Bootstrap (learn on CHALLENGE too) =====
        elif (decision == SentinelDecision.CHALLENGE and
                keyboard_state.completed_windows):
            stored_hst_check = self.model_store.load_model(user_id, ModelType.HST)
            if stored_hst_check is None or stored_hst_check.feature_window_count < 50:
                windows_to_learn = keyboard_state.completed_windows
                
                def learn_hst_bootstrap(model: Any) -> None:
                    for window in windows_to_learn:
                        model.learn_one(window["features"])
                
                self.model_store.learn_with_retry(
                    user_id,
                    ModelType.HST,
                    learn_hst_bootstrap,
                    lambda: KeyboardAnomalyModel(),
                    len(windows_to_learn)
                )
                logger.info(f"[COLD START] Bootstrap learning: {len(windows_to_learn)} windows on CHALLENGE")
        
        # ===== Identity Learning (with retry) =====
        if (decision == SentinelDecision.ALLOW and
                self._should_learn_identity(session, navigator_risk, cold_start_identity, now)):
            if keyboard_state.completed_windows:
                windows_to_learn = keyboard_state.completed_windows[-IDENTITY_MAX_WINDOWS:]
                
                def learn_identity(model: Any) -> None:
                    for window in windows_to_learn:
                        model.learn_one(window["features"])
                
                self.model_store.learn_with_retry(
                    user_id,
                    ModelType.IDENTITY,
                    learn_identity,
                    lambda: KeyboardAnomalyModel(),
                    len(windows_to_learn)
                )
        
        # ===== Persist Trusted Context (TOFU Write-Behind) =====
        if (decision == SentinelDecision.ALLOW and
                session.mode == "NORMAL" and
                payload.client_fingerprint):
            geo_data = nav_metrics.get("current_geo_data") if nav_metrics else None
            self.context_processor.repo.save_trusted_context(
                user_id=user_id,
                device_id=payload.client_fingerprint.device_id,
                ip=payload.request_context.ip_address,
                geo=geo_data
            )
        
        # Mark identity ready
        if decision == SentinelDecision.ALLOW and not session.identity_ready:
            session.identity_ready = True
        
        # Persist audit fields
        session.last_decision = decision.value
        session.last_risk = risk
        session.last_eval_ts = now
        session.last_verified_ts = now  # For trust decay
        session.last_eval_id = payload.eval_id
        session.last_activity_ts = now
        
        self.repo.update_session_atomic(
            payload.session_id,
            lambda s: session
        )
        
        if payload.eval_id:
            self.repo.mark_eval_processed(payload.eval_id)
        
        # ===== Provisional Ban (First Responder) =====
        # On BLOCK, set a 5-minute provisional ban in Redis.
        # Stops spam while the Sentinel Auditor reviews the audit log.
        # NX = won't overwrite a longer auditor-confirmed ban.
        if decision == SentinelDecision.BLOCK:
            ban_key = f"blacklist:{user_id}"
            try:
                was_set = self.repo.client.set(
                    ban_key, "provisional_sentinel_block", ex=300, nx=True
                )
                if was_set:
                    logger.info(f"Provisional ban set for {user_id} (300s)")
                else:
                    logger.info(f"Ban already exists for {user_id}, skipping provisional")
            except Exception as e:
                logger.error(f"Failed to set provisional ban for {user_id}: {e}")
        
        return EvaluateResponse(
            decision=decision,
            risk=risk,
            mode=session.mode
        )
