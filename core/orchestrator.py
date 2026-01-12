"""
Sentinel Orchestrator

Central controller that coordinates Processors, Models, and State.
Handles async biometric ingestion and sync transaction evaluation.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Union

from core.models import KeyboardAnomalyModel, MouseAnomalyModel, NavigatorPolicyEngine
from core.processors import ContextProcessor, KeyboardProcessor, MouseProcessor
from core.schemas.inputs import EvaluationRequest, KeystrokePayload, MousePayload
from core.schemas.outputs import (
    ActionContext,
    ActorContext,
    ClientFingerprint,
    GeoLocation,
    NetworkContext,
    SecurityEnforcement,
    SentinelAnalysis,
    SentinelResponse,
)
from core.state_manager import StateManager


# Configure module logger
logger = logging.getLogger(__name__)


class SentinelOrchestrator:
    """
    Central orchestrator for the Sentinel fraud detection system.
    
    This class coordinates:
    - Feature processors (keyboard, mouse, context)
    - Anomaly detection models (keyboard, mouse)
    - Policy engine (navigator)
    - State management (user snapshots)
    
    Attributes:
        SCORE_DECAY_RATE: Exponential decay rate for biometric scores.
        LEARNING_THRESHOLD: Maximum risk score for safe model training.
    """
    
    # Configuration constants
    SCORE_DECAY_RATE: float = 0.01
    LEARNING_THRESHOLD: float = 0.4
    
    def __init__(self) -> None:
        """
        Initialize the orchestrator with all components.
        
        Instantiates:
        - StateManager (singleton)
        - All processors (keyboard, mouse, context)
        - All models (keyboard, mouse, navigator)
        """
        # State management
        self.state_manager = StateManager()
        
        # Feature processors
        self.keyboard_processor = KeyboardProcessor()
        self.mouse_processor = MouseProcessor()
        self.context_processor = ContextProcessor()
        
        # Anomaly models
        self.keyboard_model = KeyboardAnomalyModel()
        self.mouse_model = MouseAnomalyModel()
        
        # Policy engine
        self.navigator = NavigatorPolicyEngine()
        
        logger.info("SentinelOrchestrator initialized successfully")
    
    def _apply_decay(self, score: float, last_ts: float) -> float:
        """
        Apply exponential decay to a score based on elapsed time.
        
        Args:
            score: Original risk score (0.0 to 1.0).
            last_ts: Unix timestamp when score was recorded.
            
        Returns:
            Decayed score: score * exp(-DECAY_RATE * (now - last_ts))
        """
        elapsed = time.time() - last_ts
        decay_factor = math.exp(-self.SCORE_DECAY_RATE * elapsed)
        return score * decay_factor
    
    def process_biometric_stream(
        self, 
        payload: Union[KeystrokePayload, MousePayload]
    ) -> None:
        """
        Async ingestion of biometric stream data.
        
        Processes incoming biometric events, scores them for anomalies,
        applies gated learning, and persists results to state.
        
        Args:
            payload: Either KeystrokePayload or MousePayload from client.
            
        Logic:
            1. Extract features via appropriate processor
            2. Score via appropriate model (unpacks tuple)
            3. Gated learning: only train if risk < LEARNING_THRESHOLD
            4. Persist to state manager with structured schema
        """
        if isinstance(payload, KeystrokePayload):
            self._process_keyboard_stream(payload)
        elif isinstance(payload, MousePayload):
            self._process_mouse_stream(payload)
        else:
            logger.warning(f"Unknown payload type: {type(payload)}")
    
    def _process_keyboard_stream(self, payload: KeystrokePayload) -> None:
        """Process keyboard biometric stream."""
        # Extract features
        features = self.keyboard_processor.extract_features(payload.events)
        
        # Score (unpack tuple!)
        risk, vectors = self.keyboard_model.score_one(features)
        
        # Gated learning
        if risk < self.LEARNING_THRESHOLD:
            self.keyboard_model.learn_one(features)
            logger.debug(f"Keyboard model trained (risk={risk:.4f})")
        else:
            logger.info(f"Skipping keyboard training (High Risk: {risk:.4f})")
        
        # Persist to state
        # Note: Using session_id as a proxy for user identification
        # In production, this would map to actual user_id
        self.state_manager.update_snapshot(
            user_id=payload.session_id,
            updates={
                "latest_keyboard_entry": {
                    "score": risk,
                    "vectors": vectors,
                    "timestamp": time.time(),
                }
            }
        )
    
    def _process_mouse_stream(self, payload: MousePayload) -> None:
        """Process mouse biometric stream."""
        # Extract features
        features = self.mouse_processor.extract_features(payload.events)
        
        # Score (unpack tuple!)
        risk, vectors = self.mouse_model.score_one(features)
        
        # Gated learning
        if risk < self.LEARNING_THRESHOLD:
            self.mouse_model.learn_one(features)
            logger.debug(f"Mouse model trained (risk={risk:.4f})")
        else:
            logger.info(f"Skipping mouse training (High Risk: {risk:.4f})")
        
        # Persist to state
        self.state_manager.update_snapshot(
            user_id=payload.session_id,
            updates={
                "latest_mouse_entry": {
                    "score": risk,
                    "vectors": vectors,
                    "timestamp": time.time(),
                }
            }
        )
    
    def evaluate_transaction(self, request: EvaluationRequest) -> SentinelResponse:
        """
        Synchronous transaction evaluation.
        
        Evaluates a transaction request for fraud risk by combining
        context metrics with decayed biometric scores.
        
        Args:
            request: EvaluationRequest containing user, business, and network context.
            
        Returns:
            SentinelResponse with complete risk assessment.
            
        Logic:
            1. Fetch user snapshot from StateManager
            2. Derive context metrics via ContextProcessor
            3. Retrieve & decay biometric entries (keyboard/mouse)
            4. Call NavigatorPolicyEngine.evaluate()
            5. Persist final risk score
            6. Build and return SentinelResponse
        """
        user_id = request.user_session.user_id
        
        # Step 1: Fetch state snapshot
        snapshot = self.state_manager.get_snapshot(user_id)
        
        # Step 2: Derive context metrics
        ctx_metrics = self.context_processor.derive_context_metrics(request, snapshot)
        
        # Step 3: Retrieve and decay biometric entries
        kb_decayed, kb_vectors = self._get_decayed_biometric(
            snapshot.get("latest_keyboard_entry")
        )
        ms_decayed, ms_vectors = self._get_decayed_biometric(
            snapshot.get("latest_mouse_entry")
        )
        
        # Step 4: Call navigator policy engine
        biometric_data: Dict[str, Tuple[float, List[str]]] = {
            "keyboard": (kb_decayed, kb_vectors),
            "mouse": (ms_decayed, ms_vectors),
        }
        analysis: SentinelAnalysis = self.navigator.evaluate(ctx_metrics, biometric_data)
        
        # Step 5: Persist final risk score
        self.state_manager.update_snapshot(
            user_id=user_id,
            updates={
                "risk_score": analysis.risk_score,
                "derived_geo": self._get_geo_coords(request.network_context.ip_address),
                "current_device_hash": request.network_context.ja3_hash,
                "transaction_amount": request.business_context.transaction_details.get(
                    "amount", 0.0
                ),
            }
        )
        
        # Step 6: Build response
        return self._build_response(request, analysis)
    
    def _get_decayed_biometric(
        self, 
        entry: Dict[str, Any] | None
    ) -> Tuple[float, List[str]]:
        """
        Get decayed biometric score and vectors from state entry.
        
        Args:
            entry: State entry dict with 'score', 'vectors', 'timestamp' or None.
            
        Returns:
            Tuple of (decayed_score, vectors). Returns (0.0, []) if entry is None.
        """
        if entry is None:
            return (0.0, [])
        
        score = entry.get("score", 0.0)
        timestamp = entry.get("timestamp", time.time())
        vectors = entry.get("vectors", [])
        
        decayed_score = self._apply_decay(score, timestamp)
        return (decayed_score, vectors)
    
    def _get_geo_coords(self, ip_address: str) -> Tuple[float, float] | None:
        """Get geo coordinates for an IP address."""
        # Use context processor's geo lookup
        geo_data = self.context_processor.get_geo_location(ip_address)
        # The context processor doesn't expose coords directly,
        # so we return None here. In production, this would be enhanced.
        return None
    
    def _build_response(
        self, 
        request: EvaluationRequest, 
        analysis: SentinelAnalysis
    ) -> SentinelResponse:
        """
        Build the complete SentinelResponse from request and analysis.
        
        Args:
            request: Original evaluation request.
            analysis: Risk analysis from navigator.
            
        Returns:
            Complete SentinelResponse with all required fields.
        """
        # Calculate session age
        now = datetime.now(timezone.utc)
        session_age = int((now - request.user_session.session_start_time).total_seconds())
        
        # Get geo location and IP reputation
        ip_address = request.network_context.ip_address
        geo_location_data = self.context_processor.get_geo_location(ip_address)
        ip_reputation = self.context_processor.get_ip_reputation(ip_address)
        
        return SentinelResponse(
            event_id=str(uuid.uuid4()),
            timestamp=now,
            correlation_id=request.user_session.session_id,
            environment="production",
            actor=ActorContext(
                user_id=request.user_session.user_id,
                session_id=request.user_session.session_id,
                role=request.user_session.role,
                session_age_seconds=session_age,
            ),
            action_context=ActionContext(
                service=request.business_context.service,
                action_type=request.business_context.action_type,
                resource_target=request.business_context.resource_target,
                details=request.business_context.transaction_details,
            ),
            network_context=NetworkContext(
                ip_address=ip_address,
                geo_location=GeoLocation(
                    asn=geo_location_data.get("asn", "AS0 Unknown"),
                    city=geo_location_data.get("city", "Unknown"),
                    country=geo_location_data.get("country", "XX"),
                ),
                ip_reputation=ip_reputation,
                client_fingerprint=ClientFingerprint(
                    ja3_hash=request.network_context.ja3_hash or "unknown",
                    device_id=request.network_context.ja3_hash or "unknown",
                    user_agent_raw=request.network_context.user_agent,
                ),
            ),
            sentinel_analysis=analysis,
            security_enforcement=SecurityEnforcement(
                mfa_status=request.user_session.mfa_status,
                policy_applied="sentinel_default_v1",
            ),
        )
