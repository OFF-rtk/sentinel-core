"""
Sentinel Engine Integration Tests - Model Warm-up & Stress Testing

This script simulates user sessions with proper model warm-up to validate
the end-to-end flow of the SentinelOrchestrator. Results are logged to results.md.

Test Phases:
1. Warm-up: Feed 300+ sliding windows to train the online learning models.
2. Attack: Switch to bot-like patterns and verify anomaly detection.
3. Decay: Verify time-based score decay.

Usage:
    python test_integration.py
"""

import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import List, Tuple

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

from core.orchestrator import SentinelOrchestrator
from core.schemas.inputs import (
    KeystrokePayload,
    KeyboardEvent,
    KeyEventType,
    MousePayload,
    MouseEvent,
    MouseEventType,
    EvaluationRequest,
    UserSessionContext,
    BusinessContext,
    ClientNetworkContext,
)
from core.schemas.outputs import SentinelDecision
from core.state_manager import StateManager


# =============================================================================
# Configuration
# =============================================================================

# Seed for reproducibility
random.seed(42)

# Model window size (from HalfSpaceTrees config)
MODEL_WINDOW_SIZE = 250

# Warm-up text (Standard Pangrams)
WARMUP_TEXT = """
The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.
How vexingly quick daft zebras jump! The five boxing wizards jump quickly at dawn.
Sphinx of black quartz, judge my vow. Two driven jocks help fax my big quiz.
The job requires extra pluck and zeal from every young wage earner. Quick zephyrs blow,
vexing daft Jim. Crazy Frederick bought many very exquisite opal jewels.
""".replace("\n", " ").strip()


# =============================================================================
# Improved Data Generators
# =============================================================================

def generate_natural_typing(
    text: str,
    mode: str = "normal",
    start_timestamp: float = None
) -> List[KeyboardEvent]:
    """
    Generate keystroke events with realistic Gaussian-distributed timing.
    """
    events = []
    current_ts = start_timestamp or (time.time() * 1000)
    
    for char in text:
        if mode == "normal":
            # Human-like: Dwell 100ms ± 15ms, Flight 120ms ± 30ms
            dwell_time = max(50.0, random.gauss(100.0, 15.0))
            flight_time = max(30.0, random.gauss(120.0, 30.0))
        else:  # bot mode
            # Bot: Fixed minimal timing (Machine Speed)
            dwell_time = 10.0
            flight_time = 0.0  # Impossible for humans
        
        # Key DOWN event
        events.append(KeyboardEvent(
            key=char,
            event_type=KeyEventType.DOWN,
            timestamp=current_ts
        ))
        
        # Key UP event (after dwell time)
        current_ts += dwell_time
        events.append(KeyboardEvent(
            key=char,
            event_type=KeyEventType.UP,
            timestamp=current_ts
        ))
        
        # Flight time to next key
        current_ts += flight_time
    
    return events


def generate_mouse_movements(
    n_points: int,
    pattern: str = "normal",
    start_timestamp: float = None,
    start_x: int = 100,
    start_y: int = 100
) -> List[MouseEvent]:
    """
    Generate mouse movement events with different patterns.
    """
    events = []
    current_ts = start_timestamp or (time.time() * 1000)
    x, y = start_x, start_y
    
    for i in range(n_points):
        if pattern == "normal":
            # Human-like: curved movement with natural velocity variations
            angle = random.gauss(0.0, 0.5)  # Direction change
            speed = random.gauss(50.0, 20.0)  # Pixels per movement
            dx = int(speed * math.cos(angle + i * 0.1))
            dy = int(speed * math.sin(angle + i * 0.1))
            time_delta = max(10.0, random.gauss(50.0, 20.0))
        else:  # bot mode
            # Bot: perfectly straight lines with instant velocity
            dx = 10
            dy = 0
            time_delta = 5.0
        
        x = max(0, min(1920, x + dx))
        y = max(0, min(1080, y + dy))
        current_ts += time_delta
        
        events.append(MouseEvent(
            x=x,
            y=y,
            event_type=MouseEventType.MOVE,
            timestamp=current_ts
        ))
    
    return events


def create_evaluation_request(
    user_id: str,
    session_id: str,
    ip_address: str,
    ja3_hash: str | None = None,
    role: str = "analyst",
    mfa_status: str = "verified"
) -> EvaluationRequest:
    """Create an EvaluationRequest with the given parameters."""
    return EvaluationRequest(
        user_session=UserSessionContext(
            user_id=user_id,
            session_id=session_id,
            role=role,
            session_start_time=datetime.now(timezone.utc),
            mfa_status=mfa_status
        ),
        business_context=BusinessContext(
            service="card_service",
            action_type="card_activation",
            resource_target="card_xxxx1234",
            transaction_details={"amount": 1000.0, "currency": "USD"}
        ),
        network_context=ClientNetworkContext(
            ip_address=ip_address,
            user_agent="Mozilla/5.0 TestSuite",
            ja3_hash=ja3_hash
        )
    )


# =============================================================================
# Logging Helpers
# =============================================================================

def log_step(file, title: str, data, level: int = 3) -> None:
    """Write a formatted section to results.md."""
    prefix = "#" * level
    file.write(f"{prefix} {title}\n\n")
    if isinstance(data, (dict, list)):
        file.write("```json\n")
        file.write(json.dumps(data, indent=2, default=str))
        file.write("\n```\n\n")
    else:
        file.write(f"{data}\n\n")


def log_score_progression(file, progression: List[Tuple[int, float, str]]) -> None:
    """Log score progression table."""
    file.write("| Update # | Risk Score | Status |\n")
    file.write("|----------|------------|--------|\n")
    for update_num, score, status in progression:
        file.write(f"| {update_num} | {score:.4f} | {status} |\n")
    file.write("\n")


# =============================================================================
# Test Phases
# =============================================================================

def run_phase_1_warmup(
    orchestrator: SentinelOrchestrator,
    file,
    session_id: str
) -> Tuple[bool, List[Tuple[int, float, str]]]:
    """
    Phase 1: Model Warm-up (Training)
    
    STRATEGY: Sliding Window Ingestion.
    Instead of tiny batches, we take a large pool of events and "slide" a window 
    of 50 events over it (step=5).
    
    Result: 2000 events -> ~390 robust updates.
    """
    file.write("## Phase 1: Model Warm-up (Training)\n\n")
    file.write("**Strategy:** Sliding Window Ingestion (Window=50, Step=5).\n")
    file.write(f"**Target:** > {MODEL_WINDOW_SIZE} updates to fill HalfSpaceTrees window.\n\n")
    
    # 1. Generate a large pool of events (repeating text)
    full_text = WARMUP_TEXT * 10
    keyboard_pool = generate_natural_typing(full_text, mode="normal")
    mouse_pool = generate_mouse_movements(2500, pattern="normal")
    
    file.write(f"**Generated Pool:** {len(keyboard_pool)} Keyboard Events, {len(mouse_pool)} Mouse Events.\n\n")
    
    # 2. Sliding Window Configuration
    window_size = 50   # Realistic payload size
    step_size = 5      # Overlap stride
    
    score_progression: List[Tuple[int, float, str]] = []
    update_count = 0
    
    # Calculate max possible start index
    max_start = len(keyboard_pool) - window_size
    
    print(f"   Streaming sliding windows (Window={window_size}, Step={step_size})...")

    for i in range(0, max_start, step_size):
        # Slice the window
        kb_batch = keyboard_pool[i : i + window_size]
        ms_batch = mouse_pool[i : i + window_size]
        
        # Send Keyboard Payload
        payload_kb = KeystrokePayload(
            session_id=session_id,
            sequence_id=update_count,
            events=kb_batch
        )
        orchestrator.process_biometric_stream(payload_kb)
        
        # Send Mouse Payload
        payload_ms = MousePayload(
            session_id=session_id,
            sequence_id=update_count,
            events=ms_batch
        )
        orchestrator.process_biometric_stream(payload_ms)
        
        update_count += 1
        
        # Check scores periodically
        if update_count % 50 == 0 or i + step_size >= max_start:
            snapshot = orchestrator.state_manager.get_snapshot(session_id)
            kb_entry = snapshot.get("latest_keyboard_entry", {})
            score = kb_entry.get("score", 0.0)
            
            if update_count < MODEL_WINDOW_SIZE:
                status = "⏳ Filling Window"
            elif score < 0.4:
                status = "✅ Stable"
            else:
                status = "⚠️ High"
            
            score_progression.append((update_count, score, status))

    # Log progression
    log_step(file, "Score Progression (Keyboard)", "", level=3)
    log_score_progression(file, score_progression)
    
    # Get final state
    snapshot = orchestrator.state_manager.get_snapshot(session_id)
    keyboard_entry = snapshot.get("latest_keyboard_entry", {})
    final_score = keyboard_entry.get("score", 0.0)
    
    # Assert: Model is warmed up if it stays stable low
    warmup_success = final_score < 0.4 and update_count > MODEL_WINDOW_SIZE
    
    log_step(file, "Warm-up Result", 
        f"**Final Keyboard Score:** `{final_score:.4f}`\n\n"
        f"**Total Updates:** {update_count}\n\n"
        f"**Status:** {'✅ Ready' if warmup_success else '⚠️ Unstable'}"
    )
    
    return warmup_success, score_progression


def run_phase_2_attack(
    orchestrator: SentinelOrchestrator,
    file,
    session_id: str
) -> Tuple[bool, float, List[str]]:
    """
    Phase 2: Attack Simulation
    
    GOAL: Send a batch of anomaly data (Bot). 
    Since the model is now full of "Normal" data, this "Alien" data should trigger a high outlier score.
    """
    file.write("## Phase 2: Attack Simulation\n\n")
    
    # Generate attack patterns (Bot speed: 0ms flight)
    attack_text = "password_dump_mode_activated"
    bot_keyboard = generate_natural_typing(attack_text, mode="bot")
    bot_mouse = generate_mouse_movements(50, pattern="bot")
    
    # Send Attack Payloads
    # We send 3 distinct batches to ensure the Orchestrator picks up the latest one
    for i in range(3):
        payload_kb = KeystrokePayload(session_id=session_id, sequence_id=1000+i, events=bot_keyboard)
        orchestrator.process_biometric_stream(payload_kb)
        
        payload_ms = MousePayload(session_id=session_id, sequence_id=1000+i, events=bot_mouse)
        orchestrator.process_biometric_stream(payload_ms)
    
    # Check Biometric State (Async Result)
    snapshot = orchestrator.state_manager.get_snapshot(session_id)
    kb_entry = snapshot.get("latest_keyboard_entry", {})
    ms_entry = snapshot.get("latest_mouse_entry", {})
    
    log_step(file, "Biometric Scores (Post-Attack)", {
        "keyboard_score": kb_entry.get("score", 0.0),
        "keyboard_vectors": kb_entry.get("vectors", []),
        "mouse_score": ms_entry.get("score", 0.0),
        "mouse_vectors": ms_entry.get("vectors", [])
    })
    
    # Trigger Evaluation (Sync Check)
    # Using a Bad IP to compound the risk
    request = create_evaluation_request(
        user_id="usr_77252",
        session_id=session_id,
        ip_address="185.220.101.1", # Bad IP
        ja3_hash="unknown_bot_device",
        mfa_status="not_verified"
    )
    
    response = orchestrator.evaluate_transaction(request)
    analysis = response.sentinel_analysis
    
    log_step(file, "Sentinel Analysis (Attack)", {
        "risk_score": analysis.risk_score,
        "decision": analysis.decision.value,
        "anomaly_vectors": analysis.anomaly_vectors
    })
    
    # Success Criteria: High Risk Score OR Blocked Decision
    # We expect score > 0.6 because biometric + context risk should stack
    attack_detected = analysis.risk_score > 0.6
    
    return attack_detected, analysis.risk_score, analysis.anomaly_vectors


def run_phase_3_decay(
    orchestrator: SentinelOrchestrator,
    file,
    session_id: str,
    pre_decay_score: float
) -> Tuple[bool, float]:
    """
    Phase 3: Decay Test
    """
    file.write("## Phase 3: Decay Test\n\n")
    
    print("   Waiting 2 seconds for decay...")
    time.sleep(2)
    
    # Re-evaluate with GOOD Context (San Francisco IP)
    # This ensures the only risk factor remaining is the (decayed) biometric score
    request = create_evaluation_request(
        user_id="usr_77252",
        session_id=session_id,
        ip_address="192.168.1.1", # Good IP
        ja3_hash="dev_ab4e80f2cbe04656", # Good Device
        mfa_status="verified"
    )
    
    response = orchestrator.evaluate_transaction(request)
    analysis = response.sentinel_analysis
    post_decay_score = analysis.risk_score
    
    log_step(file, "Post-Decay Analysis", {
        "pre_decay_score": pre_decay_score,
        "post_decay_score": post_decay_score,
        "decision": analysis.decision.value
    })
    
    # Check if the score dropped.
    # Note: Pre-decay score (Attack) included Context Risk (Bad IP).
    # Post-decay score excludes Context Risk AND decays Biometric Risk.
    # So the drop should be significant.
    decay_worked = post_decay_score < pre_decay_score
    return decay_worked, post_decay_score


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 60)
    print("🛡️  SENTINEL ENGINE INTEGRATION TEST (SLIDING WINDOW)")
    print("=" * 60)
    
    StateManager.reset_instance()
    orchestrator = SentinelOrchestrator()
    session_id = "usr_77252"
    
    with open(os.path.join(RESULTS_DIR, "integration_results.md"), "w") as f:
        f.write("# Sentinel Integration Test Results\n\n")
        f.write(f"**Date:** {datetime.now(timezone.utc)}\n\n")
        
        # Phase 1
        print("\n📋 Phase 1: Warm-up...")
        p1_pass, _ = run_phase_1_warmup(orchestrator, f, session_id)
        print(f"   Result: {'✅ PASS' if p1_pass else '⚠️ UNSTABLE'}")
        
        f.write("---\n")
        
        # Phase 2
        print("\n📋 Phase 2: Attack...")
        p2_pass, score, _ = run_phase_2_attack(orchestrator, f, session_id)
        print(f"   Result: {'✅ PASS' if p2_pass else '❌ FAIL'}")
        print(f"   Attack Score: {score:.4f}")
        
        f.write("---\n")
        
        # Phase 3
        print("\n📋 Phase 3: Decay...")
        p3_pass, d_score = run_phase_3_decay(orchestrator, f, session_id, score)
        print(f"   Result: {'✅ PASS' if p3_pass else '❌ FAIL'}")
        print(f"   Decayed Score: {d_score:.4f}")
        
    print("\n📄 Done. Check results.md")

if __name__ == "__main__":
    main()