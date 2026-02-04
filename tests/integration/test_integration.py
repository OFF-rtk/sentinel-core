"""
Sentinel Engine Integration Tests - Full API Data Flow

This script validates the end-to-end flow from API input to Orchestrator output.
It uses FastAPI TestClient to simulate real HTTP traffic.

Test Phases:
1. Warm-up: Feed 300+ sliding windows via /stream endpoints.
2. Attack: Switch to bot-like patterns and verify anomaly detection via /evaluate.
3. Decay: Verify time-based score decay via /evaluate.

Usage:
    pytest tests/integration/test_integration.py
"""

import json
import math
import os
import sys
import time
import random
from typing import List, Tuple, Dict, Any
from datetime import datetime, timezone

from fastapi.testclient import TestClient

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from main import app
from core.schemas.inputs import (
    KeyboardStreamPayload,
    MouseStreamPayload,
    EvaluatePayload,
    KeyboardEvent,
    KeyEventType,
    MouseEvent,
    MouseEventType,
    RequestContext,
    BusinessContext,
    ClientFingerprint
)
from core.schemas.outputs import SentinelDecision

# Initialize TestClient
client = TestClient(app)

# =============================================================================
# Configuration
# =============================================================================

random.seed(42)
MODEL_WINDOW_SIZE = 250

WARMUP_TEXT = """
The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.
How vexingly quick daft zebras jump! The five boxing wizards jump quickly at dawn.
Sphinx of black quartz, judge my vow. Two driven jocks help fax my big quiz.
The job requires extra pluck and zeal from every young wage earner. Quick zephyrs blow,
vexing daft Jim. Crazy Frederick bought many very exquisite opal jewels.
""".replace("\n", " ").strip()

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# Generators
# =============================================================================

def generate_natural_typing(
    text: str,
    mode: str = "normal",
    start_timestamp: float = None
) -> List[KeyboardEvent]:
    events = []
    current_ts = start_timestamp or (time.time() * 1000.0)
    
    for char in text:
        if mode == "normal":
            dwell_time = max(50.0, random.gauss(100.0, 15.0))
            flight_time = max(30.0, random.gauss(120.0, 30.0))
        else:
            dwell_time = 10.0
            flight_time = 0.0
        
        events.append(KeyboardEvent(key=char, event_type=KeyEventType.DOWN, timestamp=current_ts))
        current_ts += dwell_time
        events.append(KeyboardEvent(key=char, event_type=KeyEventType.UP, timestamp=current_ts))
        current_ts += flight_time
    
    return events

def generate_mouse_movements(
    n_points: int,
    pattern: str = "normal",
    start_timestamp: float = None,
    start_x: int = 100,
    start_y: int = 100
) -> List[MouseEvent]:
    events = []
    current_ts = start_timestamp or (time.time() * 1000.0)
    x, y = start_x, start_y
    
    for i in range(n_points):
        if pattern == "normal":
            angle = random.gauss(0.0, 0.5)
            speed = random.gauss(50.0, 20.0)
            dx = int(speed * math.cos(angle + i * 0.1))
            dy = int(speed * math.sin(angle + i * 0.1))
            time_delta = max(10.0, random.gauss(50.0, 20.0))
        else:
            dx, dy = 10, 0
            time_delta = 5.0
        
        x = max(0, min(1920, x + dx))
        y = max(0, min(1080, y + dy))
        current_ts += time_delta
        
        events.append(MouseEvent(x=x, y=y, event_type=MouseEventType.MOVE, timestamp=current_ts))
    
    return events

# =============================================================================
# Helper Functions
# =============================================================================

# Helper Functions
# =============================================================================

def send_streams(
    client: TestClient,
    session_id: str, 
    user_id: str, 
    kb_events: List[KeyboardEvent], 
    ms_events: List[MouseEvent],
    window_start: int = 0
) -> None:
    """Send batched streams to API."""
    # Chunk into small batches to simulate streaming
    BATCH_SIZE = 50
    kb_chunk = kb_events[window_start : window_start + BATCH_SIZE]
    ms_chunk = ms_events[window_start : window_start + BATCH_SIZE]
    
    # We use a simple sequential logic here for the test invocation count
    # In a real app, client manages sequence_id.
    # We will use window_start as a proxy for specific calls
    seq_id = (window_start // BATCH_SIZE) + 1
    
    if kb_chunk:
        payload = KeyboardStreamPayload(
            session_id=session_id,
            user_id=user_id,
            sequence_id=seq_id,
            batch_id=seq_id,
            events=kb_chunk
        )
        resp = client.post("/stream/keyboard", json=payload.model_dump(mode='json'))
        assert resp.status_code == 204, f"Keyboard stream failed: {resp.text}"

    if ms_chunk:
        payload = MouseStreamPayload(
            session_id=session_id,
            user_id=user_id,
            sequence_id=seq_id,
            batch_id=seq_id,
            events=ms_chunk
        )
        resp = client.post("/stream/mouse", json=payload.model_dump(mode='json'))
        assert resp.status_code == 204, f"Mouse stream failed: {resp.text}"

def call_evaluate(
    client: TestClient,
    session_id: str,
    user_id: str,
    ip: str = "127.0.0.1",
    ua: str = "Mozilla/Test",
    fingerprint: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Call evaluate endpoint."""
    fp_model = None
    if fingerprint:
        fp_model = ClientFingerprint(**fingerprint)

    payload = EvaluatePayload(
        session_id=session_id,
        request_context=RequestContext(
            ip_address=ip,
            user_agent=ua,
            endpoint="/login",
            method="POST",
            user_id=user_id
        ),
        business_context=BusinessContext(
            service="banking",
            action_type="transfer",
            resource_target="account_123",
            transaction_details={"amount": 500}
        ),
        role="analyst",
        mfa_status="verified",
        session_start_time=time.time() * 1000.0,
        client_fingerprint=fp_model
    )
    
    resp = client.post("/evaluate", json=payload.model_dump(mode='json'))
    assert resp.status_code == 200, f"Evaluate failed: {resp.text}"
    return resp.json()

# =============================================================================
# Test Phases
# =============================================================================

def test_full_flow():
    session_id = f"sess_{int(time.time())}"
    user_id = "test_user_integration"
    
    print(f"\n🚀 Starting Integration Test [Session: {session_id}]")
    
    # Use context manager to trigger lifespan events (DB connection, etc.)
    with TestClient(app) as client:
    
        # --- PHASE 1: WARM-UP ---
        print("\n📋 Phase 1: Warm-up (Training Normal Patterns)")
        
        # Generate large pool
        full_text = WARMUP_TEXT * 5
        kb_pool = generate_natural_typing(full_text, mode="normal")
        ms_pool = generate_mouse_movements(len(kb_pool) + 100, pattern="normal")
        
        print(f"Generated {len(kb_pool)} keyboard events and {len(ms_pool)} mouse events.")
        
        # Stream in chunks (simulating sliding window inputs)
        # We send enough data to fill the models
        chunk_size = 50
        steps = min(len(kb_pool), len(ms_pool)) // chunk_size
        
        for i in range(steps):
            start_idx = i * chunk_size
            send_streams(client, session_id, user_id, kb_pool, ms_pool, start_idx)
            
            # Periodically evaluate
            if i % 5 == 0:
                eval_resp = call_evaluate(client, session_id, user_id)
                print(f"   [Step {i}] Risk: {eval_resp['risk']:.4f} | Decision: {eval_resp['decision']}")
        
        final_warmup = call_evaluate(client, session_id, user_id)
        print(f"✅ Warm-up Complete. Risk: {final_warmup['risk']:.4f}")
        
        # Ideally risk should be low after training on normal data
        # (Though HST needs ~250 samples to mature, steps per 50 items = many events)
        assert final_warmup['risk'] < 0.6, "Risk should be low after warm-up"


        # --- PHASE 2: ATTACK ---
        print("\n📋 Phase 2: Bot Attack")
        
        # Generate bot data
        bot_text = "evil_bot_payload" * 5
        bot_kb = generate_natural_typing(bot_text, mode="bot")
        bot_ms = generate_mouse_movements(len(bot_kb), pattern="bot")
        
        # Send attack stream (use higher sequence IDs)
        # Send 3 rapid batches
        for i in range(3):
            current_batch_id = steps + i + 1
            
            pl_kb = KeyboardStreamPayload(
                session_id=session_id, user_id=user_id, sequence_id=current_batch_id, 
                batch_id=current_batch_id, events=bot_kb[:50]
            )
            client.post("/stream/keyboard", json=pl_kb.model_dump(mode='json'))
            
            pl_ms = MouseStreamPayload(
                session_id=session_id, user_id=user_id, sequence_id=current_batch_id, 
                batch_id=current_batch_id, events=bot_ms[:50]
            )
            client.post("/stream/mouse", json=pl_ms.model_dump(mode='json'))

        # Evaluate Attack
        # We add suspicious context too (New Device)
        fp = {"ja3_hash": "sus_hash", "device_id": "unknown_device", "user_agent_raw": "Python/Requests"}
        
        attack_resp = call_evaluate(client, session_id, user_id, fingerprint=fp)
        print(f"   Attack Risk: {attack_resp['risk']:.4f} | Decision: {attack_resp['decision']}")
        
        # Assert risk is high OR decision is block/challenge
        assert attack_resp['risk'] > 0.6 or attack_resp['decision'] in ["CHALLENGE", "BLOCK"], \
            "Attack should trigger high risk"

        # --- PHASE 3: DECAY ---
        print("\n📋 Phase 3: Decay Test (Waiting 2s)")
        time.sleep(2.1)
        
        # Evaluate again with CLEAN context
        decay_resp = call_evaluate(client, session_id, user_id, ip="192.168.1.5", ua="Mozilla/Clean")
        print(f"   Decay Risk: {decay_resp['risk']:.4f}")
        
        # Check that risk dropped
        assert decay_resp['risk'] < attack_resp['risk'], "Risk should decay over time"
        
        print("\n✅ Integration Test Passed!")

if __name__ == "__main__":
    test_full_flow()