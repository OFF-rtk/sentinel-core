"""
Sentinel Keystroke Model Test (Linux/Wayland Native)

Records real-time keystrokes, trains KeyboardAnomalyModel, then validates
detection with simulated bot behavior. All feature extraction and anomaly
scoring uses KeyboardProcessor and KeyboardAnomalyModel - no explicit logic.

Usage:
    sudo ./venv/bin/python test_keystroke_model.py

Output:
    results.md - Detailed test results with human and bot scores
"""

import sys
import os
from datetime import datetime
from typing import List, Tuple, Optional

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

try:
    import evdev
    from evdev import ecodes
except ImportError:
    print("❌ Error: 'evdev' not found. Run: pip install evdev")
    sys.exit(1)

# Import Sentinel Core modules
from core.processors.keyboard import KeyboardProcessor, WINDOW_SIZE
from core.models.keyboard import KeyboardAnomalyModel
from core.schemas.inputs import KeyboardEvent, KeyEventType


# =============================================================================
# Constants
# =============================================================================

COLD_START_WINDOWS = 50        # Train-only period (first 50 feature windows)
SCORE_LOG_INTERVAL = 10        # Log every N scoring windows to results.md


# =============================================================================
# Key Mapping (for evdev -> KeyboardEvent conversion)
# =============================================================================

def find_keyboard_device():
    """Auto-detect the first keyboard device."""
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    for device in devices:
        if "keyboard" in device.name.lower():
            return device
    return None


def evdev_to_key_name(event_code: int) -> str:
    """Convert evdev key code to processor-compatible key name."""
    if event_code == ecodes.KEY_BACKSPACE:
        return "Backspace"
    elif event_code == ecodes.KEY_DELETE:
        return "Delete"
    elif event_code == ecodes.KEY_SPACE:
        return " "
    elif event_code == ecodes.KEY_ENTER:
        return "Enter"
    
    raw_name = evdev.ecodes.KEY.get(event_code, "UNKNOWN")
    if isinstance(raw_name, str) and raw_name.startswith("KEY_"):
        return raw_name[4:]
    return str(raw_name)


# =============================================================================
# Results Writer
# =============================================================================

class ResultsWriter:
    """Writes test results to results.md with proper formatting."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.human_scores: List[Tuple[int, float, List[str]]] = []
        self.bot_scores: List[Tuple[int, float, List[str]]] = []
        self.total_keystrokes = 0
        self.total_windows = 0
        self.start_time = datetime.now()
    
    def add_human_score(self, window_num: int, score: float, vectors: List[str]):
        """Record a human typing score."""
        self.human_scores.append((window_num, score, vectors))
    
    def add_bot_score(self, test_num: int, score: float, vectors: List[str]):
        """Record a bot simulation score."""
        self.bot_scores.append((test_num, score, vectors))
    
    def save(self):
        """Write all results to markdown file."""
        with open(self.filepath, "w") as f:
            f.write("# Keystroke Model Test Results\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Keystrokes Recorded**: {self.total_keystrokes}\n\n")
            f.write(f"**Total Feature Windows**: {self.total_windows}\n\n")
            f.write(f"**Cold Start Period**: First {COLD_START_WINDOWS} windows (train-only)\n\n")
            f.write(f"**Window Config**: Size={WINDOW_SIZE}, Stride=5\n\n")
            f.write("---\n\n")
            
            # Human Scores Section
            f.write("## 👤 Human Typing Scores (Training Phase)\n\n")
            f.write(f"Scores recorded every {SCORE_LOG_INTERVAL} scoring windows after cold start.\n\n")
            f.write("| Window # | Score | Status | Anomaly Vectors |\n")
            f.write("|----------|-------|--------|------------------|\n")
            
            for window_num, score, vectors in self.human_scores:
                status = self._score_status(score)
                vectors_str = ", ".join(vectors) if vectors else "—"
                f.write(f"| {window_num} | {score:.3f} | {status} | {vectors_str} |\n")
            
            if not self.human_scores:
                f.write("| — | — | No scores recorded | — |\n")
            
            f.write("\n---\n\n")
            
            # Bot Simulation Section
            f.write("## 🤖 Bot Simulation Scores (Post-Training)\n\n")
            f.write("Extreme bot behavior features scored against trained model.\n\n")
            f.write("| Test # | Score | Status | Anomaly Vectors |\n")
            f.write("|--------|-------|--------|------------------|\n")
            
            for test_num, score, vectors in self.bot_scores:
                status = self._score_status(score)
                vectors_str = ", ".join(vectors) if vectors else "—"
                f.write(f"| {test_num} | {score:.3f} | {status} | {vectors_str} |\n")
            
            f.write("\n---\n\n")
            
            # Summary
            f.write("## 📊 Summary\n\n")
            
            if self.human_scores:
                human_avg = sum(s[1] for s in self.human_scores) / len(self.human_scores)
                human_first = self.human_scores[0][1] if self.human_scores else 0
                human_last = self.human_scores[-1][1] if self.human_scores else 0
                f.write(f"- **Human Avg Score**: {human_avg:.3f}\n")
                f.write(f"- **Human First Score**: {human_first:.3f}\n")
                f.write(f"- **Human Last Score**: {human_last:.3f}\n")
                f.write(f"- **Score Improvement**: {human_first - human_last:.3f} (lower is better)\n")
            
            if self.bot_scores:
                bot_avg = sum(s[1] for s in self.bot_scores) / len(self.bot_scores)
                bot_detections = sum(1 for s in self.bot_scores if s[1] > 0.5)
                f.write(f"- **Bot Avg Score**: {bot_avg:.3f}\n")
                f.write(f"- **Bot Detections (>0.5)**: {bot_detections}/{len(self.bot_scores)}\n")
            
            f.write("\n### Legend\n\n")
            f.write("- ✅ Normal: Score < 0.3\n")
            f.write("- 🟡 Medium: Score 0.3-0.5\n")
            f.write("- ⚠️ High: Score 0.5-0.7\n")
            f.write("- 🚨 Anomaly: Score > 0.7\n")
    
    def _score_status(self, score: float) -> str:
        if score > 0.7:
            return "🚨 Anomaly"
        elif score > 0.5:
            return "⚠️ High"
        elif score > 0.3:
            return "🟡 Medium"
        return "✅ Normal"


# =============================================================================
# Bot Simulator
# =============================================================================

def generate_bot_features() -> List[dict]:
    """
    Generate 10 extreme bot behavior feature vectors.
    These represent inhuman typing precision that should trigger anomalies.
    """
    import random
    random.seed(42)
    
    bot_vectors = []
    for i in range(10):
        # Inhuman precision: very fast, very consistent
        bot_vectors.append({
            "dwell_time_mean": random.uniform(1.0, 5.0),      # 1-5ms (humans: 60-200ms)
            "dwell_time_std": random.uniform(0.1, 0.5),       # Near-zero variation
            "flight_time_mean": random.uniform(10.0, 30.0),   # 10-30ms (humans: 100-600ms)
            "flight_time_std": random.uniform(0.5, 2.0),      # Robotic consistency
            "error_rate": 0.0,                                  # Perfect typing
        })
    return bot_vectors


# =============================================================================
# Main Test Loop
# =============================================================================

def main():
    print("=" * 60)
    print("🧪 SENTINEL KEYSTROKE MODEL TEST")
    print("=" * 60)
    
    # 1. Setup Device
    device = find_keyboard_device()
    if not device:
        print("❌ No keyboard detected! Are you running with sudo?")
        sys.exit(1)
    
    print(f"✅ Connected to: {device.name}")
    print("-" * 60)
    print("Test Protocol:")
    print(f"  • Window Size: {WINDOW_SIZE} keystrokes")
    print(f"  • Window Stride: 5 (new window every 5 keystrokes)")
    print(f"  • Cold Start: First {COLD_START_WINDOWS} windows (train-only)")
    print(f"  • Logging: Every {SCORE_LOG_INTERVAL} scoring windows")
    print("  • Bot Test: 10 simulations after ESC")
    print("-" * 60)
    print("🔴 RECORDING STARTED... Type naturally! Press ESC to finish.\n")
    
    # Initialize components
    processor = KeyboardProcessor()
    model = KeyboardAnomalyModel()
    results = ResultsWriter(os.path.join(RESULTS_DIR, "keyboard_results.md"))
    
    keystroke_count = 0
    window_count = 0
    scoring_window_count = 0
    
    # 2. Capture Loop - Stream events directly to processor
    try:
        device.grab()
        for event in device.read_loop():
            if event.type == ecodes.EV_KEY:
                # Skip key hold (autorepeat)
                if event.value == 2:
                    continue
                
                # ESC to finish
                if event.code == ecodes.KEY_ESC:
                    break
                
                # Skip modifier keys
                if event.code in [ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT,
                                  ecodes.KEY_LEFTCTRL, ecodes.KEY_RIGHTCTRL,
                                  ecodes.KEY_LEFTALT, ecodes.KEY_RIGHTALT]:
                    continue
                
                # Convert to KeyboardEvent
                ts = event.timestamp() * 1000.0
                etype = KeyEventType.DOWN if event.value == 1 else KeyEventType.UP
                key_name = evdev_to_key_name(event.code)
                
                keyboard_event = KeyboardEvent(key=key_name, event_type=etype, timestamp=ts)
                
                # Count keystrokes (DOWN events only)
                if event.value == 1:
                    keystroke_count += 1
                
                # Stream event to processor - it handles windowing internally
                features = processor.process_event(keyboard_event)
                
                # If processor emitted a feature vector, process it
                if features is not None:
                    window_count += 1
                    
                    if window_count <= COLD_START_WINDOWS:
                        # Cold start: train only
                        model.learn_one(features)
                        print(f"\r🔄 Cold Start: Window {window_count}/{COLD_START_WINDOWS} | Keystrokes: {keystroke_count}", end="")
                    else:
                        # After cold start: score then train
                        score, vectors = model.score_one(features)
                        model.learn_one(features)
                        
                        scoring_window_count += 1
                        
                        # Log every N scoring windows
                        if scoring_window_count % SCORE_LOG_INTERVAL == 0:
                            results.add_human_score(scoring_window_count, score, vectors)
                        
                        print(f"\r📊 Keystrokes: {keystroke_count} | Windows: {window_count} | Score: {score:.3f} | Logged: {len(results.human_scores)}", end="")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted.")
    finally:
        try:
            device.ungrab()
            print("\n\n🔓 Keyboard released.")
        except:
            pass
    
    results.total_keystrokes = keystroke_count
    results.total_windows = window_count
    print(f"\n✅ RECORDING STOPPED.")
    print(f"   Keystrokes: {keystroke_count}")
    print(f"   Windows: {window_count}")
    print(f"   Scored: {scoring_window_count}")
    
    if window_count < COLD_START_WINDOWS:
        print(f"❌ Not enough windows for cold start ({window_count}/{COLD_START_WINDOWS})")
        keystrokes_needed = (COLD_START_WINDOWS - window_count) * 5 + WINDOW_SIZE
        print(f"   Need ~{keystrokes_needed} more keystrokes next time!")
        results.save()
        print(f"\n📄 Partial results saved to: {os.path.abspath('results.md')}")
        return
    
    # 3. Bot Simulation
    print("\n🤖 Running Bot Simulation (10 tests)...")
    bot_features = generate_bot_features()
    
    for i, features in enumerate(bot_features, 1):
        score, vectors = model.score_one(features)
        results.add_bot_score(i, score, vectors)
        print(f"  Bot Test {i}: Score={score:.3f}, Vectors={vectors}")
    
    # 4. Save Results
    results.save()
    print(f"\n✅ Results saved to: {os.path.abspath('results.md')}")
    print("   Open this file to review the test results!")


if __name__ == "__main__":
    main()
