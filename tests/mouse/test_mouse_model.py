#!/usr/bin/env python3
"""
Sentinel Mouse Model Test Suite v2 (Hybrid)

Tests the action-based mouse behavioral anomaly detection pipeline.
Implements 'Training Noise' to prevent overfitting on efficient human behavior.

Tests:
1. Human Baseline Replay - Your real data (Target Score: < 0.5)
2. Straight-Line Bot - Physics violation (Target Score: 1.0)
3. Jittered Bot - Noisy lines (Target Score: > 0.85)
4. Bézier Curve Bot - Smooth bot arcs (Target Score: > 0.85)
5. Perturbed Human - You + 10% speed/jitter (Target Score: < 0.6)
"""

import copy
import csv
import math
import random
import sys
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

from core.processors.mouse import MouseProcessor
from core.models.mouse import MouseAnomalyModel
from core.schemas.inputs import MouseEvent, MouseEventType


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CSV = os.path.join(ASSETS_DIR, "heavy_training_data.csv")
COLD_START_STROKES = 2500       # Warmup period
SCORE_LOG_INTERVAL = 50         # Log every N strokes
ANOMALY_THRESHOLD = 0.85        # Score > 0.85 is an anomaly


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TestResult:
    test_name: str
    total_strokes: int
    scores: List[Tuple[int, float, List[str]]]
    anomaly_count: int
    mean_score: float
    max_score: float
    passed: bool
    notes: str = ""


# =============================================================================
# Utilities
# =============================================================================

def load_mouse_csv(filepath: str) -> List[MouseEvent]:
    """Load mouse events from CSV file."""
    events = []
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found {filepath}")
        sys.exit(1)
        
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

def inject_clicks(events: List[MouseEvent], click_interval_ms: float = 2000.0) -> List[MouseEvent]:
    """Inject synthetic clicks to break long movement streams into strokes."""
    if not events: return events
    result = []
    last_click = events[0].timestamp
    
    for event in events:
        result.append(event)
        if event.timestamp - last_click >= click_interval_ms:
            # FIX: Use Keyword Arguments
            result.append(MouseEvent(
                x=event.x, 
                y=event.y, 
                event_type=MouseEventType.CLICK, 
                timestamp=event.timestamp
            ))
            last_click = event.timestamp
            
    # Ensure final click
    if result[-1].event_type != MouseEventType.CLICK:
        # FIX: Use Keyword Arguments
        result.append(MouseEvent(
            x=result[-1].x, 
            y=result[-1].y, 
            event_type=MouseEventType.CLICK, 
            timestamp=result[-1].timestamp + 1
        ))
    return result

def perturb_events(events: List[MouseEvent], time_scale=1.0, spatial_noise=0.0, drop_rate=0.0) -> List[MouseEvent]:
    """Apply speed/jitter variance to human data."""
    perturbed = []
    base_time = events[0].timestamp if events else 0
    
    for event in events:
        if random.random() < drop_rate: continue
        
        new_time = base_time + (event.timestamp - base_time) * time_scale
        new_x = event.x + int(random.gauss(0, spatial_noise))
        new_y = event.y + int(random.gauss(0, spatial_noise))
        
        # FIX: Use Keyword Arguments
        perturbed.append(MouseEvent(
            x=new_x, 
            y=new_y, 
            event_type=event.event_type, 
            timestamp=new_time
        ))
    return perturbed


# =============================================================================
# Bot Generators
# =============================================================================

def generate_straight_line_bot(num_strokes=150) -> List[MouseEvent]:
    """Generates physically impossible straight lines (Curvature=0)."""
    events = []
    t = 1000.0
    for _ in range(num_strokes):
        sx, sy = random.randint(100, 1800), random.randint(100, 1000)
        angle = random.uniform(0, 6.28)
        dist = random.uniform(200, 500)
        
        # 20 points, perfect line, constant speed
        for i in range(21):
            factor = i / 20
            px = int(sx + dist*math.cos(angle)*factor)
            py = int(sy + dist*math.sin(angle)*factor)
            
            # FIX: Use Keyword Arguments
            events.append(MouseEvent(
                x=px, y=py, 
                event_type=MouseEventType.MOVE, 
                timestamp=t
            ))
            t += 10 # 10ms per step
            
        ex = int(sx + dist*math.cos(angle))
        ey = int(sy + dist*math.sin(angle))
        
        # FIX: Use Keyword Arguments
        events.append(MouseEvent(
            x=ex, y=ey, 
            event_type=MouseEventType.CLICK, 
            timestamp=t
        ))
        t += 500
    return events

def generate_jittered_bot(num_strokes=150, noise=5.0) -> List[MouseEvent]:
    """Generates straight lines with random noise."""
    events = []
    t = 1000.0
    for _ in range(num_strokes):
        sx, sy = random.randint(100, 1800), random.randint(100, 1000)
        angle = random.uniform(0, 6.28)
        dist = random.uniform(200, 500)
        
        for i in range(21):
            factor = i / 20
            px = int(sx + dist*math.cos(angle)*factor + random.gauss(0, noise))
            py = int(sy + dist*math.sin(angle)*factor + random.gauss(0, noise))
            
            # FIX: Use Keyword Arguments
            events.append(MouseEvent(
                x=px, y=py, 
                event_type=MouseEventType.MOVE, 
                timestamp=t
            ))
            t += 10 + random.gauss(0, 1)
            
        ex = int(sx + dist*math.cos(angle))
        ey = int(sy + dist*math.sin(angle))
        
        # FIX: Use Keyword Arguments
        events.append(MouseEvent(
            x=ex, y=ey, 
            event_type=MouseEventType.CLICK, 
            timestamp=t
        ))
        t += 500
    return events

def generate_bezier_bot(num_strokes=150) -> List[MouseEvent]:
    """Generates smooth quadratic curves."""
    events = []
    t = 1000.0
    for _ in range(num_strokes):
        sx, sy = random.randint(100, 1800), random.randint(100, 1000)
        ex, ey = random.randint(100, 1800), random.randint(100, 1000)
        cx, cy = (sx+ex)/2 + random.uniform(-100,100), (sy+ey)/2 + random.uniform(-100,100)
        
        for i in range(31):
            step = i / 30
            # Quadratic Bezier Formula
            px = int((1-step)**2 * sx + 2*(1-step)*step * cx + step**2 * ex)
            py = int((1-step)**2 * sy + 2*(1-step)*step * cy + step**2 * ey)
            
            # FIX: Use Keyword Arguments
            events.append(MouseEvent(
                x=px, y=py, 
                event_type=MouseEventType.MOVE, 
                timestamp=t
            ))
            t += 8 + random.gauss(0, 0.5)
            
        # FIX: Use Keyword Arguments
        events.append(MouseEvent(
            x=ex, y=ey, 
            event_type=MouseEventType.CLICK, 
            timestamp=t
        ))
        t += 500
    return events


# =============================================================================
# CORE TEST LOGIC
# =============================================================================

def run_test(test_name: str, events: List[MouseEvent], model: MouseAnomalyModel, 
             cold_start: int, train: bool, expect_high: bool) -> TestResult:
    
    processor = MouseProcessor()
    scores = []
    stroke_num = 0
    anomaly_count = 0
    all_scores = []
    
    print(f"\n🧪 {test_name}")
    print(f"   Events: {len(events)}")
    
    for event in events:
        features = processor.process_event(event)
        if features is None:
            continue
            
        stroke_num += 1
        
        # --- TRAINING PHASE (WITH NOISE INJECTION) ---
        if stroke_num <= cold_start:
            # 1. Learn the REAL data point (Precise)
            model.learn_one(features)
            
            # 2. Learn JITTERED versions (Fuzzy)
            # This forces the model to accept "Adjacent" behavior as normal.
            
            # Jitter 1: Speed Variation (+/- 20%)
            j1 = features.copy()
            j1['velocity_mean'] *= random.uniform(0.80, 1.20)
            j1['velocity_std'] *= random.uniform(0.80, 1.20)
            model.learn_one(j1)
            
            # Jitter 2: Path Variation (Add artificial tremor)
            j2 = features.copy()
            j2['curvature_std'] += random.uniform(0.001, 0.01) # Add shake
            j2['angle_std'] += random.uniform(0.02, 0.1)       # Add wobble
            model.learn_one(j2)
            
            if stroke_num % 500 == 0:
                print(f"   Training... {stroke_num}/{cold_start}")
            continue
            
        # --- TESTING PHASE ---
        score, vectors = model.score_one(features)
        all_scores.append(score)
        
        is_anomaly = score > ANOMALY_THRESHOLD
        if is_anomaly: anomaly_count += 1
        
        scores.append((stroke_num, score, vectors))
        
        if train:
            model.learn_one(features)
            
        if (stroke_num - cold_start) % SCORE_LOG_INTERVAL == 0:
            status = "🚨" if is_anomaly else "✅"
            print(f"   📊 Stroke {stroke_num} | Score: {score:.3f} {status}")

    # Metrics
    scored_strokes = len(all_scores)
    rate = anomaly_count / scored_strokes if scored_strokes > 0 else 0
    mean_score = sum(all_scores) / scored_strokes if scored_strokes > 0 else 0
    max_score = max(all_scores) if all_scores else 0
    
    # Pass/Fail Logic
    if expect_high:
        passed = rate > 0.5  # Bots should be caught >50% of the time
        notes = f"Expected HIGH anomalies. Got {rate:.1%}."
    else:
        passed = rate < 0.15 # Humans should trigger <15% false positives
        notes = f"Expected LOW anomalies. Got {rate:.1%}."
        
    status_str = "✅ PASSED" if passed else "❌ FAILED"
    print(f"\n{status_str}")
    print(f"  Anomaly Rate: {rate:.1%} ({anomaly_count}/{scored_strokes})")
    print(f"  Mean Score:   {mean_score:.3f}")
    
    return TestResult(test_name, total_strokes=stroke_num, scores=scores, anomaly_count=anomaly_count, 
                      mean_score=mean_score, max_score=max_score, passed=passed, notes=notes)


# =============================================================================
# MAIN
# =============================================================================

def write_md(results: List[TestResult]):
    with open(os.path.join(RESULTS_DIR, "mouse_results.md"), "w") as f:
        f.write(f"# Test Results ({datetime.now().strftime('%H:%M:%S')})\n\n")
        f.write("| Test | Rate | Mean | Status |\n|---|---|---|---|\n")
        for r in results:
            status = "✅" if r.passed else "❌"
            f.write(f"| {r.test_name} | {r.anomaly_count/max(1, len(r.scores)):.1%} | {r.mean_score:.3f} | {status} |\n")

def main():
    csv_path = DEFAULT_CSV
    if len(sys.argv) > 1 and sys.argv[1] != "--csv": csv_path = sys.argv[1]
    elif len(sys.argv) > 2: csv_path = sys.argv[2]
    
    if not os.path.exists(csv_path):
        print("❌ CSV not found. Please provide 'heavy_training_data.csv'")
        return

    print("="*40 + "\nSENTINEL MOUSE MODEL v2 TEST\n" + "="*40)
    
    # Load and prep data
    raw = load_mouse_csv(csv_path)
    human_data = inject_clicks(raw, 1500)
    
    results = []
    
    # 1. Base Human
    model = MouseAnomalyModel()
    results.append(run_test("Human Baseline", human_data, model, COLD_START_STROKES, True, False))
    
    frozen = copy.deepcopy(model)
    
    # 2. Straight Bot (Physics Check)
    bot1 = generate_straight_line_bot(150)
    results.append(run_test("Straight Bot", bot1, copy.deepcopy(frozen), 0, False, True))
    
    # 3. Jitter Bot
    bot2 = generate_jittered_bot(150, 5.0)
    results.append(run_test("Jitter Bot", bot2, copy.deepcopy(frozen), 0, False, True))
    
    # 4. Bezier Bot
    bot3 = generate_bezier_bot(150)
    results.append(run_test("Bezier Bot", bot3, copy.deepcopy(frozen), 0, False, True))
    
    # 5. Perturbed Human (Overfitting Check)
    # 10% faster, 3px noise, 5% drop
    human_noisy = perturb_events(raw, 1.1, 3.0, 0.05)
    human_noisy = inject_clicks(human_noisy, 1500)
    results.append(run_test("Perturbed Human", human_noisy, copy.deepcopy(frozen), 0, False, False))
    
    write_md(results)
    print("\nCheck results.md for details.")

if __name__ == "__main__":
    main()