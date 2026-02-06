"""
Pydantic Schema Validation Tests

Tests for input and output schemas to ensure proper validation,
serialization, and type enforcement.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from core.schemas.inputs import (
    KeyEventType,
    MouseEventType,
    KeyboardEvent,
    MouseEvent,
    KeystrokePayload,
    MousePayload,
    UserSessionContext,
    BusinessContext,
    ClientFingerprint,
    ClientNetworkContext,
    EvaluationRequest,
    KeyboardStreamPayload,
    MouseStreamPayload,
    RequestContext,
    EvaluatePayload,
)

from core.schemas.outputs import (
    SentinelDecision,
    ActorContext,
    ActionContext,
    GeoLocation,
    ClientFingerprint as OutputClientFingerprint,
    NetworkContext,
    SentinelAnalysis,
    SecurityEnforcement,
    EvaluateResponse,
)


# =============================================================================
# Input Schema Tests - Enums
# =============================================================================

class TestEnums:
    """Test enum definitions."""
    
    def test_key_event_type_values(self):
        """KeyEventType should have DOWN and UP values."""
        assert KeyEventType.DOWN == "DOWN"
        assert KeyEventType.UP == "UP"
    
    def test_mouse_event_type_values(self):
        """MouseEventType should have MOVE and CLICK values."""
        assert MouseEventType.MOVE == "MOVE"
        assert MouseEventType.CLICK == "CLICK"
    
    def test_sentinel_decision_values(self):
        """SentinelDecision should have ALLOW, BLOCK, CHALLENGE."""
        assert SentinelDecision.ALLOW == "ALLOW"
        assert SentinelDecision.BLOCK == "BLOCK"
        assert SentinelDecision.CHALLENGE == "CHALLENGE"


# =============================================================================
# Input Schema Tests - Events
# =============================================================================

class TestEventSchemas:
    """Test event schema validation."""
    
    def test_keyboard_event_valid(self):
        """Valid KeyboardEvent should parse correctly."""
        event = KeyboardEvent(
            key="a",
            event_type=KeyEventType.DOWN,
            timestamp=1234567890.0
        )
        assert event.key == "a"
        assert event.event_type == KeyEventType.DOWN
        assert event.timestamp == 1234567890.0
    
    def test_keyboard_event_missing_required(self):
        """Missing required fields should raise ValidationError."""
        with pytest.raises(ValidationError):
            KeyboardEvent(key="a", event_type=KeyEventType.DOWN)
    
    def test_mouse_event_valid(self):
        """Valid MouseEvent should parse correctly."""
        event = MouseEvent(
            x=100,
            y=200,
            event_type=MouseEventType.MOVE,
            timestamp=1234567890.0
        )
        assert event.x == 100
        assert event.y == 200
        assert event.event_type == MouseEventType.MOVE
    
    def test_mouse_event_missing_required(self):
        """Missing required fields should raise ValidationError."""
        with pytest.raises(ValidationError):
            MouseEvent(x=100, y=200, event_type=MouseEventType.CLICK)


# =============================================================================
# Input Schema Tests - Payloads
# =============================================================================

class TestPayloadSchemas:
    """Test payload schema validation."""
    
    def test_keystroke_payload_valid(self):
        """Valid KeystrokePayload should parse correctly."""
        payload = KeystrokePayload(
            session_id="sess-123",
            user_id="user-456",
            sequence_id=1,
            events=[
                KeyboardEvent(key="a", event_type=KeyEventType.DOWN, timestamp=100.0),
                KeyboardEvent(key="a", event_type=KeyEventType.UP, timestamp=150.0),
            ]
        )
        assert payload.session_id == "sess-123"
        assert len(payload.events) == 2
    
    def test_mouse_payload_valid(self):
        """Valid MousePayload should parse correctly."""
        payload = MousePayload(
            session_id="sess-123",
            user_id="user-456",
            sequence_id=1,
            events=[
                MouseEvent(x=0, y=0, event_type=MouseEventType.MOVE, timestamp=100.0),
                MouseEvent(x=100, y=100, event_type=MouseEventType.CLICK, timestamp=200.0),
            ]
        )
        assert len(payload.events) == 2
    
    def test_keyboard_stream_payload_batch_id_validation(self):
        """batch_id must be >= 1."""
        with pytest.raises(ValidationError):
            KeyboardStreamPayload(
                session_id="sess-123",
                user_id="user-456",
                batch_id=0,  # Invalid: must be >= 1
                events=[]
            )
    
    def test_keyboard_stream_payload_valid(self):
        """Valid KeyboardStreamPayload should parse correctly."""
        payload = KeyboardStreamPayload(
            session_id="sess-123",
            user_id="user-456",
            batch_id=1,
            events=[]
        )
        assert payload.batch_id == 1


# =============================================================================
# Input Schema Tests - Evaluation Request
# =============================================================================

class TestEvaluationRequestSchema:
    """Test EvaluationRequest schema validation."""
    
    def test_evaluation_request_valid(self):
        """Valid EvaluationRequest should parse correctly."""
        request = EvaluationRequest(
            user_session=UserSessionContext(
                user_id="user-123",
                session_id="sess-456",
                role="analyst",
                session_start_time=datetime.now(),
                mfa_status="verified"
            ),
            business_context=BusinessContext(
                service="card_service",
                action_type="card_activation",
                resource_target="card-789"
            ),
            network_context=ClientNetworkContext(
                ip_address="192.168.1.1",
                user_agent="Mozilla/5.0"
            )
        )
        assert request.user_session.user_id == "user-123"
        assert request.business_context.service == "card_service"
    
    def test_evaluate_payload_valid(self):
        """Valid EvaluatePayload should parse correctly."""
        payload = EvaluatePayload(
            session_id="sess-123",
            request_context=RequestContext(
                ip_address="10.0.0.1",
                user_agent="TestAgent",
                endpoint="/api/evaluate",
                method="POST",
                user_id="user-456"
            ),
            business_context=BusinessContext(
                service="card_service",
                action_type="activate",
                resource_target="card-789"
            ),
            role="admin",
            mfa_status="verified",
            session_start_time=1234567890.0
        )
        assert payload.session_id == "sess-123"
        assert payload.role == "admin"


# =============================================================================
# Output Schema Tests
# =============================================================================

class TestOutputSchemas:
    """Test output schema validation."""
    
    def test_sentinel_analysis_valid(self):
        """Valid SentinelAnalysis should parse correctly."""
        analysis = SentinelAnalysis(
            decision=SentinelDecision.ALLOW,
            risk_score=0.25,
            engine_version="2.0.0",
            anomaly_vectors=[]
        )
        assert analysis.decision == SentinelDecision.ALLOW
        assert analysis.risk_score == 0.25
    
    def test_sentinel_analysis_risk_bounds(self):
        """risk_score must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            SentinelAnalysis(
                decision=SentinelDecision.ALLOW,
                risk_score=1.5,  # Invalid: > 1.0
                engine_version="2.0.0"
            )
        
        with pytest.raises(ValidationError):
            SentinelAnalysis(
                decision=SentinelDecision.ALLOW,
                risk_score=-0.1,  # Invalid: < 0.0
                engine_version="2.0.0"
            )
    
    def test_evaluate_response_valid(self):
        """Valid EvaluateResponse should parse correctly."""
        response = EvaluateResponse(
            decision=SentinelDecision.CHALLENGE,
            risk=0.65,
            mode="CHALLENGE"
        )
        assert response.decision == SentinelDecision.CHALLENGE
        assert response.risk == 0.65
        assert response.mode == "CHALLENGE"
    
    def test_actor_context_session_age_non_negative(self):
        """session_age_seconds must be >= 0."""
        with pytest.raises(ValidationError):
            ActorContext(
                role="user",
                user_id="user-123",
                session_id="sess-456",
                session_age_seconds=-1  # Invalid: negative
            )
    
    def test_network_context_valid(self):
        """Valid NetworkContext should parse correctly."""
        context = NetworkContext(
            ip_address="192.168.1.1",
            geo_location=GeoLocation(
                asn="AS12345 Example ISP",
                city="New York",
                country="US"
            ),
            ip_reputation="residential",
            client_fingerprint=OutputClientFingerprint(
                ja3_hash="abc123",
                device_id="dev-789",
                user_agent_raw="Mozilla/5.0"
            )
        )
        assert context.ip_address == "192.168.1.1"
        assert context.geo_location.country == "US"


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerialization:
    """Test JSON serialization and deserialization."""
    
    def test_keyboard_event_to_dict(self):
        """KeyboardEvent should serialize to dict correctly."""
        event = KeyboardEvent(
            key="Enter",
            event_type=KeyEventType.DOWN,
            timestamp=100.0
        )
        data = event.model_dump()
        
        assert data["key"] == "Enter"
        assert data["event_type"] == "DOWN"
        assert data["timestamp"] == 100.0
    
    def test_sentinel_analysis_to_json(self):
        """SentinelAnalysis should serialize to JSON correctly."""
        analysis = SentinelAnalysis(
            decision=SentinelDecision.BLOCK,
            risk_score=0.95,
            engine_version="2.0.0",
            anomaly_vectors=["impossible_travel", "infra_mismatch"]
        )
        json_str = analysis.model_dump_json()
        
        assert '"decision":"BLOCK"' in json_str
        assert '"risk_score":0.95' in json_str
        assert '"impossible_travel"' in json_str
    
    def test_evaluate_response_from_dict(self):
        """EvaluateResponse should deserialize from dict."""
        data = {
            "decision": "ALLOW",
            "risk": 0.1,
            "mode": "NORMAL"
        }
        response = EvaluateResponse(**data)
        
        assert response.decision == SentinelDecision.ALLOW
        assert response.risk == 0.1
        assert response.mode == "NORMAL"
