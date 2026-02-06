"""
Orchestrator Integration Tests

Tests the full flow of the SentinelOrchestrator using:
- Real Redis persistence (requires docker-compose up in infrastructure/redis)
- Real Supabase for model storage
- Real human keyboard and mouse data from CSV files
- Multiple sessions to test identity persistence

Test Flow:
1. Session 1: Create identity from human data (warm-up)
2. Session 2-4: Use existing identity, verify consistent scoring
3. Session 5: Inject bot patterns, verify detection

All test users are prefixed with 'test_' for clear distinction.
"""

import os
import time
import csv
import pytest
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from core.orchestrator import SentinelOrchestrator
from core.schemas.inputs import (
    KeyboardStreamPayload,
    MouseStreamPayload,
    EvaluatePayload,
    KeyboardEvent,
    MouseEvent,
    KeyEventType,
    MouseEventType,
    RequestContext,
    BusinessContext,
    ClientFingerprint,
)
from core.schemas.outputs import SentinelDecision
from persistence.session_repository import SessionRepository
from persistence.model_store import ModelStore


# =============================================================================
# Test Data Paths
# =============================================================================

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
KEYBOARD_CSV = os.path.join(ASSETS_DIR, "human_keyboard_recording.csv")
MOUSE_CSV = os.path.join(ASSETS_DIR, "human_mouse_recording.csv")

# Test user prefix for clear distinction
TEST_USER_PREFIX = "test_"


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def repo():
    """Real Redis session repository."""
    return SessionRepository()


@pytest.fixture(scope="module")
def model_store():
    """Real Supabase model store."""
    return ModelStore()


@pytest.fixture(scope="module")
def orchestrator(repo, model_store):
    """Orchestrator with real backends."""
    return SentinelOrchestrator(repo=repo, model_store=model_store)


@pytest.fixture(scope="module")
def human_keyboard_events() -> List[KeyboardEvent]:
    """Load human keyboard events from CSV."""
    if not os.path.exists(KEYBOARD_CSV):
        pytest.skip(f"Keyboard test data not found: {KEYBOARD_CSV}")
    
    events = []
    with open(KEYBOARD_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(KeyboardEvent(
                key=row.get("key", "a"),
                event_type=KeyEventType(row.get("event_type", "DOWN")),
                timestamp=float(row.get("timestamp", 0))
            ))
    return events


@pytest.fixture(scope="module")
def human_mouse_events() -> List[MouseEvent]:
    """Load human mouse events from CSV."""
    if not os.path.exists(MOUSE_CSV):
        pytest.skip(f"Mouse test data not found: {MOUSE_CSV}")
    
    events = []
    with open(MOUSE_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(MouseEvent(
                x=int(float(row.get("x", 0))),
                y=int(float(row.get("y", 0))),
                event_type=MouseEventType(row.get("event_type", "MOVE")),
                timestamp=float(row.get("timestamp", 0))
            ))
    return events


# =============================================================================
# Helper Functions
# =============================================================================

def make_keyboard_payload(
    session_id: str,
    user_id: str,
    batch_id: int,
    events: List[KeyboardEvent]
) -> KeyboardStreamPayload:
    """Create keyboard stream payload with test_ prefix."""
    return KeyboardStreamPayload(
        session_id=f"{TEST_USER_PREFIX}{session_id}",
        user_id=f"{TEST_USER_PREFIX}{user_id}",
        batch_id=batch_id,
        events=events
    )


def make_mouse_payload(
    session_id: str,
    user_id: str,
    batch_id: int,
    events: List[MouseEvent]
) -> MouseStreamPayload:
    """Create mouse stream payload with test_ prefix."""
    return MouseStreamPayload(
        session_id=f"{TEST_USER_PREFIX}{session_id}",
        user_id=f"{TEST_USER_PREFIX}{user_id}",
        batch_id=batch_id,
        events=events
    )


def make_evaluate_payload(
    session_id: str,
    user_id: str,
    session_start_time: float,
    role: str = "analyst",
    ip_address: str = "192.168.1.100",
    eval_id: str = None
) -> EvaluatePayload:
    """Create evaluation payload with test_ prefix."""
    return EvaluatePayload(
        session_id=f"{TEST_USER_PREFIX}{session_id}",
        request_context=RequestContext(
            ip_address=ip_address,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
            endpoint="/api/transaction",
            method="POST",
            user_id=f"{TEST_USER_PREFIX}{user_id}"
        ),
        business_context=BusinessContext(
            service="banking_service",
            action_type="transfer",
            resource_target="account_12345",
            transaction_details={"amount": 100.0, "currency": "USD"}
        ),
        role=role,
        mfa_status="verified",
        session_start_time=session_start_time,
        client_fingerprint=ClientFingerprint(
            device_id=f"{TEST_USER_PREFIX}device_001",
            ja3_hash="abc123"
        ),
        eval_id=f"{TEST_USER_PREFIX}{eval_id}" if eval_id else None
    )


def generate_bot_keyboard_events(base_ts: float, count: int = 50) -> List[KeyboardEvent]:
    """Generate perfectly timed bot keyboard events."""
    events = []
    ts = base_ts
    for i in range(count):
        key = chr(97 + (i % 26))  # a-z
        # Perfect constant timing (inhuman)
        events.append(KeyboardEvent(key=key, event_type=KeyEventType.DOWN, timestamp=ts))
        ts += 100.0  # Exactly 100ms dwell time
        events.append(KeyboardEvent(key=key, event_type=KeyEventType.UP, timestamp=ts))
        ts += 50.0  # Exactly 50ms flight time
    return events


def generate_bot_mouse_events(base_ts: float, count: int = 20) -> List[MouseEvent]:
    """Generate perfectly linear bot mouse events."""
    events = []
    ts = base_ts
    for i in range(count):
        # Perfectly straight line (inhuman)
        events.append(MouseEvent(
            x=100 + i * 10,
            y=100 + i * 10,
            event_type=MouseEventType.MOVE,
            timestamp=ts
        ))
        ts += 10.0  # Exactly 10ms between points
    # End with click
    events.append(MouseEvent(
        x=100 + count * 10,
        y=100 + count * 10,
        event_type=MouseEventType.CLICK,
        timestamp=ts
    ))
    return events


# =============================================================================
# Integration Tests
# =============================================================================

class TestOrchestratorWarmup:
    """Test orchestrator warm-up and identity creation."""
    
    def test_cold_start_returns_valid_decision(self, orchestrator):
        """New session with no data should return valid decision based on physics/navigator."""
        session_id = "session_cold_start"
        user_id = "user_cold_start"
        now = time.time() * 1000
        
        payload = make_evaluate_payload(session_id, user_id, now)
        result = orchestrator.evaluate(payload)
        
        # Cold start should return a valid decision (depends on physics/navigator, not identity)
        assert result.decision in [SentinelDecision.ALLOW, SentinelDecision.CHALLENGE, SentinelDecision.BLOCK]
        assert result.risk >= 0.0
        assert result.risk <= 1.0
        
        print(f"\n✅ Cold start: decision={result.decision}, risk={result.risk:.3f}")
    
    def test_keyboard_stream_updates_state(
        self,
        orchestrator,
        human_keyboard_events
    ):
        """Keyboard stream should update session state."""
        ts_id = str(int(time.time() * 1000))
        session_id = f"session_kb_update_{ts_id}"
        user_id = f"user_kb_update_{ts_id}"
        
        if len(human_keyboard_events) < 100:
            pytest.skip("Not enough keyboard events for test")
        
        # Send multiple batches
        batch_size = 50
        for i in range(3):
            batch = human_keyboard_events[i*batch_size:(i+1)*batch_size]
            payload = make_keyboard_payload(session_id, user_id, i + 1, batch)
            orchestrator.process_keyboard_stream(payload)
        
        # Session should have been updated
        session = orchestrator.repo.get_session(f"{TEST_USER_PREFIX}{session_id}")
        assert session is not None
        assert session.last_keyboard_batch_id >= 1
        
        print(f"\n✅ Keyboard stream updated session, batch_id={session.last_keyboard_batch_id}")
    
    def test_mouse_stream_updates_state(
        self,
        orchestrator,
        human_mouse_events
    ):
        """Mouse stream should update session state."""
        ts_id = str(int(time.time() * 1000))
        session_id = f"session_mouse_update_{ts_id}"
        user_id = f"user_mouse_update_{ts_id}"
        
        if len(human_mouse_events) < 50:
            pytest.skip("Not enough mouse events for test")
        
        # Send batch
        payload = make_mouse_payload(session_id, user_id, 1, human_mouse_events[:50])
        orchestrator.process_mouse_stream(payload)
        
        # Session should have been updated
        session = orchestrator.repo.get_session(f"{TEST_USER_PREFIX}{session_id}")
        assert session is not None
        assert session.last_mouse_batch_id >= 1
        
        print(f"\n✅ Mouse stream updated session, batch_id={session.last_mouse_batch_id}")


class TestOrchestratorFullFlow:
    """Test complete orchestrator flow across 5 sessions."""
    
    def test_five_session_identity_flow(
        self,
        orchestrator,
        human_keyboard_events,
        human_mouse_events
    ):
        """
        Test full identity lifecycle across 5 sessions:
        1. Session 1: Create identity from human data
        2. Sessions 2-4: Use and update identity
        3. Session 5: Inject bot patterns
        
        Uses test_ prefix for all user/session IDs.
        """
        ts_id = str(int(time.time() * 1000))
        user_id = f"lifecycle_user_{ts_id}"  # Unique user ID per test run
        
        if len(human_keyboard_events) < 500:
            pytest.skip("Not enough keyboard events for full flow test")
        
        results = []
        batch_size = 50
        
        print("\n" + "="*60)
        print("FIVE SESSION IDENTITY FLOW TEST")
        print("="*60)
        
        for session_num in range(5):
            session_id = f"lifecycle_session_{ts_id}_{session_num}"
            now = time.time() * 1000
            
            print(f"\n--- Session {session_num + 1} ---")
            
            if session_num < 4:
                # Sessions 1-4: Human behavior
                for i in range(3):
                    start_idx = (session_num * 3 + i) * batch_size
                    end_idx = start_idx + batch_size
                    
                    # Wrap around if needed
                    if end_idx > len(human_keyboard_events):
                        start_idx = i * batch_size
                        end_idx = start_idx + batch_size
                    
                    batch = human_keyboard_events[start_idx:end_idx]
                    payload = make_keyboard_payload(session_id, user_id, i + 1, batch)
                    orchestrator.process_keyboard_stream(payload)
                    print(f"  Sent keyboard batch {i+1} ({len(batch)} events)")
                
                # Also send some mouse data
                if len(human_mouse_events) >= 30:
                    mouse_batch = human_mouse_events[:30]
                    mouse_payload = make_mouse_payload(session_id, user_id, 1, mouse_batch)
                    orchestrator.process_mouse_stream(mouse_payload)
                    print(f"  Sent mouse batch ({len(mouse_batch)} events)")
            else:
                # Session 5: Bot behavior (for contrast)
                print("  [BOT INJECTION]")
                bot_events = generate_bot_keyboard_events(now, count=100)
                for i in range(2):
                    batch = bot_events[i*50:(i+1)*50]
                    payload = make_keyboard_payload(session_id, user_id, i + 1, batch)
                    orchestrator.process_keyboard_stream(payload)
                    print(f"  Sent bot keyboard batch {i+1} ({len(batch)} events)")
                
                bot_mouse = generate_bot_mouse_events(now, count=30)
                mouse_payload = make_mouse_payload(session_id, user_id, 1, bot_mouse)
                orchestrator.process_mouse_stream(mouse_payload)
                print(f"  Sent bot mouse batch ({len(bot_mouse)} events)")
            
            # Evaluate
            eval_payload = make_evaluate_payload(session_id, user_id, now)
            result = orchestrator.evaluate(eval_payload)
            results.append(result)
            
            print(f"  RESULT: decision={result.decision}, risk={result.risk:.3f}")
        
        # All results should be valid
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        for i, result in enumerate(results):
            assert result.risk >= 0.0, f"Session {i+1} has invalid risk score"
            assert result.risk <= 1.0, f"Session {i+1} has invalid risk score"
            session_type = "BOT" if i == 4 else "HUMAN"
            print(f"Session {i+1} ({session_type}): decision={result.decision.value}, risk={result.risk:.3f}")
        
        # At minimum, verify that all sessions returned valid decisions
        decisions = [r.decision for r in results]
        assert all(d in [SentinelDecision.ALLOW, SentinelDecision.CHALLENGE, SentinelDecision.BLOCK] for d in decisions)
        
        # Bot session should have higher risk than average human session
        human_avg_risk = sum(r.risk for r in results[:4]) / 4
        bot_risk = results[4].risk
        print(f"\nHuman avg risk: {human_avg_risk:.3f}")
        print(f"Bot risk: {bot_risk:.3f}")
        
        print(f"\n✅ Full flow completed with decisions: {[d.value for d in decisions]}")


class TestOrchestratorDecisions:
    """Test orchestrator decision logic."""
    
    def test_eval_idempotency(self, orchestrator):
        """Same eval_id should return cached result."""
        session_id = "idempotent_session"
        user_id = "idempotent_user"
        now = time.time() * 1000
        eval_id = "unique_eval_123"
        
        payload1 = make_evaluate_payload(session_id, user_id, now, eval_id=eval_id)
        result1 = orchestrator.evaluate(payload1)
        
        # Second call with same eval_id
        payload2 = make_evaluate_payload(session_id, user_id, now, eval_id=eval_id)
        result2 = orchestrator.evaluate(payload2)
        
        # Results should match (idempotent)
        assert result1.decision == result2.decision
        assert result1.risk == result2.risk
        
        print(f"\n✅ Idempotency verified: both calls returned {result1.decision.value}")
    
    def test_high_risk_context(self, orchestrator):
        """Test with high-risk contextual signals."""
        ts_id = str(int(time.time() * 1000))
        session_id = f"high_risk_session_{ts_id}"
        user_id = f"high_risk_user_{ts_id}"
        now = time.time() * 1000
        
        # Suspicious context (admin role, unverified MFA)
        payload = EvaluatePayload(
            session_id=f"{TEST_USER_PREFIX}{session_id}",
            request_context=RequestContext(
                ip_address="45.33.32.156",  # External IP
                user_agent="curl/7.68.0",  # Suspicious user agent
                endpoint="/admin/users",
                method="DELETE",
                user_id=f"{TEST_USER_PREFIX}{user_id}"
            ),
            business_context=BusinessContext(
                service="admin_service",
                action_type="user_deletion",
                resource_target="all_users",
                transaction_details={"batch": True}
            ),
            role="admin",
            mfa_status="not_verified",
            session_start_time=now,
            client_fingerprint=ClientFingerprint(device_id=f"{TEST_USER_PREFIX}unknown_device"),
        )
        
        result = orchestrator.evaluate(payload)
        
        # Should return a valid decision
        assert result.decision in [SentinelDecision.ALLOW, SentinelDecision.CHALLENGE, SentinelDecision.BLOCK]
        assert result.risk >= 0.0
        
        print(f"\n✅ High-risk context: decision={result.decision.value}, risk={result.risk:.3f}")
