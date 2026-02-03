#!/usr/bin/env python3
"""
Sentinel Physics Pipeline Test Suite

Comprehensive end-to-end tests for the Physics Engine:
- MouseProcessor feature extraction
- PhysicsMouseModel scoring

Debug mode enabled: prints all features and scores.

Usage:
    python test_physics_pipeline.py
"""

import csv
import math
import os
import random
import sys
import unittest
from typing import List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

from core.schemas.inputs import MouseEvent, MouseEventType
from core.processors.mouse import MouseProcessor
from core.models.mouse import PhysicsMouseModel


# =============================================================================
# Configuration
# =============================================================================

DEBUG = True
HUMAN_CSV_PATH = os.path.join(ASSETS_DIR, "mouse_recording.csv")


# =============================================================================
# Helper Functions
# =============================================================================

def load_mouse_csv(filepath: str) -> List[MouseEvent]:
    """Load mouse events from CSV file."""
    events: List[MouseEvent] = []
    
    if not os.path.exists(filepath):
        print(f"⚠️  Warning: CSV file not found: {filepath}")
        return events
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_type = MouseEventType.CLICK if row['event_type'] == 'CLICK' else MouseEventType.MOVE
            events.append(MouseEvent(
                x=int(float(row['x'])),
                y=int(float(row['y'])),
                event_type=event_type,
                timestamp=float(row['timestamp'])
            ))
    
    return events


def inject_terminal_click(events: List[MouseEvent]) -> List[MouseEvent]:
    """Ensure the event stream ends with a click to flush the stroke."""
    if not events:
        return events
    
    result = list(events)
    if result[-1].event_type != MouseEventType.CLICK:
        result.append(MouseEvent(
            x=result[-1].x,
            y=result[-1].y,
            event_type=MouseEventType.CLICK,
            timestamp=result[-1].timestamp + 1
        ))
    return result


def run_pipeline(
    events: List[MouseEvent],
    expected_score: float,
    expected_vectors: List[str],
    test_name: str
) -> bool:
    """
    Run the full Processor -> Model pipeline and verify results.
    
    Returns True if all strokes match expectations, False otherwise.
    """
    processor = MouseProcessor()
    model = PhysicsMouseModel()
    
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    print(f"Total events: {len(events)}")
    print(f"Expected: score={expected_score}, vectors={expected_vectors}")
    print("-" * 60)
    
    stroke_count = 0
    all_passed = True
    
    for event in events:
        features = processor.process_event(event)
        
        if features is not None:
            stroke_count += 1
            score, vectors = model.score_one(features)
            
            if DEBUG:
                print(f"\n📊 Stroke #{stroke_count} Features:")
                for k, v in sorted(features.items()):
                    # Highlight key physics features
                    marker = ""
                    if k == "velocity_max" and v > 5.0:
                        marker = " ⚠️ EXCEEDS LIMIT"
                    elif k == "velocity_std" and v < 0.001:
                        marker = " ⚠️ TOO CONSTANT"
                    elif k == "linearity_error" and v < 0.5:
                        marker = " ⚠️ TOO LINEAR"
                    print(f"    {k}: {v:.6f}{marker}")
                
                print(f"\n🎯 Model Output: score={score:.2f}, vectors={vectors}")
            
            # Verify expectations
            score_match = abs(score - expected_score) < 0.01
            vectors_match = set(vectors) == set(expected_vectors)
            
            if score_match and vectors_match:
                print(f"✅ Stroke #{stroke_count} PASSED")
            else:
                print(f"❌ Stroke #{stroke_count} FAILED")
                if not score_match:
                    print(f"   Expected score: {expected_score}, Got: {score}")
                if not vectors_match:
                    print(f"   Expected vectors: {expected_vectors}, Got: {vectors}")
                all_passed = False
    
    print("-" * 60)
    print(f"Total strokes processed: {stroke_count}")
    
    if stroke_count == 0:
        print("⚠️  WARNING: No strokes were extracted from events!")
        print("   Check if events meet minimum stroke requirements (>=10 segments, >=50px)")
        return False
    
    return all_passed


# =============================================================================
# Bot Event Generators
# =============================================================================

def generate_speed_bot_events() -> List[MouseEvent]:
    """
    Generate events that trigger TIER 2 accumulation (multiple flags).
    
    Strategy:
    - Constant timing (time_diff_std ~ 0) -> +0.35
    - Constant velocity via exact integer steps -> +0.25
    - Near-linear path (mostly straight) -> +0.25
    - Total: 0.85 >= 0.7 threshold
    """
    events: List[MouseEvent] = []
    t = 1000.0
    
    # EXACTLY constant timing and EXACTLY constant distance
    segment_time = 10.0  # ms (constant)
    step_x = 8  # pixels (constant) - results in constant velocity
    step_y = 0  # mostly straight line
    
    x, y = 100, 100
    
    for i in range(30):
        events.append(MouseEvent(
            x=x, y=y,
            event_type=MouseEventType.MOVE,
            timestamp=t
        ))
        t += segment_time
        x += step_x
        y += step_y
    
    # Terminal click
    events.append(MouseEvent(
        x=x, y=y,
        event_type=MouseEventType.CLICK,
        timestamp=t
    ))
    
    return events


def generate_linear_bot_events() -> List[MouseEvent]:
    """
    Generate events that trigger TIER 1: inhuman_linearity.
    
    Strategy:
    - path_distance > 300px
    - linearity_error < 0.2px (nearly perfect line)
    - Jittered timing to avoid Tier 2 timing flags
    """
    events: List[MouseEvent] = []
    t = 1000.0
    
    # Move exactly along y=x, 400+ pixels (>300px requirement)
    num_points = 35
    for i in range(num_points):
        x = 100 + i * 12  # 12px per step = 408px total
        y = 100 + i * 12  # Perfectly linear: y = x -> linearity_error = 0
        
        # Heavy timing jitter to avoid Tier 2 timing flags
        time_step = 40 + random.uniform(-20, 20)  # 20-60ms per step
        
        events.append(MouseEvent(
            x=x, y=y,
            event_type=MouseEventType.MOVE,
            timestamp=t
        ))
        t += time_step
    
    # Terminal click
    events.append(MouseEvent(
        x=events[-1].x, y=events[-1].y,
        event_type=MouseEventType.CLICK,
        timestamp=t
    ))
    
    return events


def generate_constant_bot_events() -> List[MouseEvent]:
    """
    Generate events that trigger TIER 2 accumulation (multiple flags).
    
    Strategy:
    - Constant timing (time_diff_std ~ 0) -> +0.35
    - Constant velocity (velocity_std ~ 0) -> +0.25
    - Excessive linearity (mostly straight) -> +0.25
    - Total: 0.85 >= 0.7 threshold
    """
    events: List[MouseEvent] = []
    t = 1000.0
    
    # EXACTLY constant timing
    segment_time = 20.0  # ms
    
    x, y = 100, 100
    
    for i in range(25):
        events.append(MouseEvent(
            x=x, y=y,
            event_type=MouseEventType.MOVE,
            timestamp=t
        ))
        t += segment_time  # EXACTLY 20ms every time
        
        # Move exactly 10 pixels each time (constant distance -> constant velocity)
        x += 10
        y += 0  # Straight line for excessive_linearity
    
    # Terminal click
    events.append(MouseEvent(
        x=x, y=y,
        event_type=MouseEventType.CLICK,
        timestamp=t
    ))
    
    return events


# =============================================================================
# Test Cases
# =============================================================================

class TestPhysicsPipeline(unittest.TestCase):
    """Test suite for the Sentinel Physics Engine pipeline."""
    
    def test_01_human_baseline(self):
        """Test that real human data passes all physics checks."""
        events = load_mouse_csv(HUMAN_CSV_PATH)
        
        if not events:
            self.skipTest(f"Human recording not found: {HUMAN_CSV_PATH}")
        
        events = inject_terminal_click(events)
        
        # Run pipeline - expect all strokes to pass
        processor = MouseProcessor()
        model = PhysicsMouseModel()
        
        print(f"\n{'='*60}")
        print("TEST: Human Baseline")
        print(f"{'='*60}")
        print(f"Loaded {len(events)} events from {HUMAN_CSV_PATH}")
        
        stroke_count = 0
        violations = []
        
        for event in events:
            features = processor.process_event(event)
            
            if features is not None:
                stroke_count += 1
                score, vectors = model.score_one(features)
                
                if DEBUG:
                    print(f"\n📊 Stroke #{stroke_count}: score={score:.2f}, vectors={vectors}")
                    print(f"    velocity_max={features.get('velocity_max', 0):.4f}")
                    print(f"    velocity_std={features.get('velocity_std', 0):.6f}")
                    print(f"    linearity_error={features.get('linearity_error', 0):.4f}")
                
                if score > 0.0 or vectors:
                    violations.append((stroke_count, score, vectors))
        
        print("-" * 60)
        print(f"Total strokes: {stroke_count}")
        print(f"Violations: {len(violations)}")
        
        if violations:
            print("\n⚠️  False Positives Detected:")
            for stroke_num, score, vectors in violations:
                print(f"   Stroke #{stroke_num}: score={score}, vectors={vectors}")
        
        # Assert no false positives
        self.assertEqual(len(violations), 0, 
            f"Human data triggered {len(violations)} false positive(s)")
    
    def test_02_speed_bot(self):
        """Test that Tier 2 accumulation detects multi-flag bots."""
        events = generate_speed_bot_events()
        
        # This bot triggers multiple Tier 2 flags
        # We just check that score=1.0 and at least one vector is present
        processor = MouseProcessor()
        model = PhysicsMouseModel()
        
        print(f"\n{'='*60}")
        print("TEST: Speed Bot (Tier 2 Accumulation)")
        print(f"{'='*60}")
        
        for event in events:
            features = processor.process_event(event)
            if features:
                score, vectors = model.score_one(features)
                print(f"Score: {score}, Vectors: {vectors}")
                self.assertEqual(score, 1.0, "Bot should be detected")
                self.assertTrue(len(vectors) > 0, "Should have at least one reason")
    
    def test_03_linear_bot(self):
        """Test that TIER 1 inhuman_linearity is detected."""
        events = generate_linear_bot_events()
        
        result = run_pipeline(
            events=events,
            expected_score=1.0,
            expected_vectors=["inhuman_linearity"],
            test_name="Linear Bot (Tier 1: Inhuman Linearity)"
        )
        
        self.assertTrue(result, "Linear bot was not correctly detected")
    
    def test_04_constant_bot(self):
        """Test that Tier 2 accumulation detects constant bots."""
        events = generate_constant_bot_events()
        
        # This bot triggers multiple Tier 2 flags (timing + jitter + linearity)
        processor = MouseProcessor()
        model = PhysicsMouseModel()
        
        print(f"\n{'='*60}")
        print("TEST: Constant Bot (Tier 2 Accumulation)")
        print(f"{'='*60}")
        
        for event in events:
            features = processor.process_event(event)
            if features:
                score, vectors = model.score_one(features)
                print(f"Score: {score}, Vectors: {vectors}")
                self.assertEqual(score, 1.0, "Bot should be detected")
                self.assertTrue(len(vectors) >= 2, "Should have multiple Tier 2 flags")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SENTINEL PHYSICS PIPELINE TEST SUITE")
    print("=" * 60)
    print(f"Debug mode: {DEBUG}")
    print(f"Human CSV: {HUMAN_CSV_PATH}")
    
    # Run with verbose output
    unittest.main(verbosity=2)
