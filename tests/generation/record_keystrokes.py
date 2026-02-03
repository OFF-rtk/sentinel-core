"""
Sentinel Keystroke Recorder & Visualizer (Linux/Wayland Native)

1. Records real-time keystrokes (Bypassing Wayland via evdev).
2. Provides enhanced visual feedback (Backspace handling, etc.).
3. Extracts raw features using KeyboardProcessor.
4. Scales them using River's StandardScaler (Z-Score).
5. SAVES the visualization to 'sentinel_analysis.png'.

Usage:
    sudo ./venv/bin/python record_keystrokes.py
"""

import sys
import os
from typing import List

try:
    import evdev
    from evdev import ecodes
except ImportError:
    print("❌ Error: 'evdev' not found. Run: pip install evdev")
    sys.exit(1)

try:
    import matplotlib

    # Force a non-interactive backend that works without a display server
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from river.preprocessing import StandardScaler
except ImportError:
    print("❌ Error: 'matplotlib', 'numpy', or 'river' not found.")
    print("Run: pip install matplotlib numpy river")
    sys.exit(1)

# Import Sentinel Core modules
from core.processors.keyboard import KeyboardProcessor
from core.schemas.inputs import KeyboardEvent, KeyEventType

# --- VISUAL MAPPING HELPERS ---
CHAR_MAP = {
    ecodes.KEY_SPACE: " ",
    ecodes.KEY_ENTER: "\n",
    ecodes.KEY_TAB: "\t",
    ecodes.KEY_DOT: ".",
    ecodes.KEY_COMMA: ",",
    ecodes.KEY_SLASH: "/",
    ecodes.KEY_SEMICOLON: ";",
    ecodes.KEY_APOSTROPHE: "'",
    ecodes.KEY_LEFTBRACE: "[",
    ecodes.KEY_RIGHTBRACE: "]",
    ecodes.KEY_BACKSLASH: "\\",
    ecodes.KEY_MINUS: "-",
    ecodes.KEY_EQUAL: "=",
    ecodes.KEY_GRAVE: "`",
    ecodes.KEY_1: "1",
    ecodes.KEY_2: "2",
    ecodes.KEY_3: "3",
    ecodes.KEY_4: "4",
    ecodes.KEY_5: "5",
    ecodes.KEY_6: "6",
    ecodes.KEY_7: "7",
    ecodes.KEY_8: "8",
    ecodes.KEY_9: "9",
    ecodes.KEY_0: "0",
}

SHIFT_MAP = {
    "1": "!",
    "2": "@",
    "3": "#",
    "4": "$",
    "5": "%",
    "6": "^",
    "7": "&",
    "8": "*",
    "9": "(",
    "0": ")",
    "-": "_",
    "=": "+",
    "[": "{",
    "]": "}",
    "\\": "|",
    ";": ":",
    "'": '"',
    ",": "<",
    ".": ">",
    "/": "?",
}


def find_keyboard_device():
    """Auto-detect the first keyboard device."""
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    for device in devices:
        if "keyboard" in device.name.lower():
            return device
    return None


def main():
    print("=" * 60)
    print("📊 SENTINEL RECORDER & VISUALIZER")
    print("=" * 60)

    # 1. Setup Device
    device = find_keyboard_device()
    if not device:
        print("❌ No keyboard detected! Are you running with sudo?")
        sys.exit(1)

    print(f"✅ Connected to: {device.name}")
    print("-" * 60)
    print("Instructions:")
    print("1. Type a long paragraph (at least 3-4 sentences).")
    print("2. The more you type, the better the histograms will look.")
    print("3. Press 'ESC' to finish and generate plots.")
    print("-" * 60)
    print("🔴 RECORDING STARTED... (Type now)\n")

    raw_events: List[KeyboardEvent] = []
    shift_pressed = False

    # 2. Capture Loop
    try:
        device.grab()
        for event in device.read_loop():
            if event.type == ecodes.EV_KEY:
                # Value 2 is KeyHold (autorepeat), ignore it for processing
                if event.value == 2:
                    continue

                if event.code == ecodes.KEY_ESC:
                    break

                # --- Shift Tracking ---
                if event.code in [ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT]:
                    shift_pressed = event.value == 1

                ts = event.timestamp() * 1000.0
                etype = KeyEventType.DOWN if event.value == 1 else KeyEventType.UP

                # --- 1. Internal Processor Mapping ---
                raw_key_name = evdev.ecodes.KEY.get(event.code, "UNKNOWN")
                if isinstance(raw_key_name, str) and raw_key_name.startswith("KEY_"):
                    processor_key = raw_key_name[4:]
                else:
                    processor_key = str(raw_key_name)

                # Special overrides
                if event.code == ecodes.KEY_BACKSPACE:
                    processor_key = "Backspace"
                elif event.code == ecodes.KEY_DELETE:
                    processor_key = "Delete"
                elif event.code == ecodes.KEY_SPACE:
                    processor_key = " "
                elif event.code == ecodes.KEY_ENTER:
                    processor_key = "Enter"

                raw_events.append(
                    KeyboardEvent(key=processor_key, event_type=etype, timestamp=ts)
                )

                # --- 2. Enhanced Visual Feedback ---
                if event.value == 1:  # On Down
                    visual_char = ""

                    if event.code == ecodes.KEY_BACKSPACE:
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                        continue

                    elif event.code == ecodes.KEY_ENTER:
                        visual_char = "\n"

                    elif event.code in CHAR_MAP:
                        visual_char = CHAR_MAP[event.code]

                    elif len(processor_key) == 1 and processor_key.isalpha():
                        visual_char = processor_key.lower()

                    # Apply Shift
                    if shift_pressed:
                        if visual_char in SHIFT_MAP:
                            visual_char = SHIFT_MAP[visual_char]
                        else:
                            visual_char = visual_char.upper()

                    if visual_char:
                        sys.stdout.write(visual_char)
                        sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted.")
    finally:
        try:
            device.ungrab()
            print("\n\n🔓 Keyboard released.")
        except:
            pass

    print(f"✅ RECORDING STOPPED. Captured {len(raw_events)} events.")
    if len(raw_events) < 10:
        print("❌ Not enough data to plot.")
        return

    # 3. Processing & Scaling
    print("\n🔄 Processing and Scaling Data...")
    processor = KeyboardProcessor()

    # Extract Raw Lists (using internal methods to verify logic)
    key_presses = processor._pair_events(raw_events)
    key_presses.sort(key=lambda kp: kp.press_time)

    raw_dwells = [kp.dwell_time for kp in key_presses if kp.dwell_time >= 0]
    raw_flights = processor._extract_flight_times(key_presses)

    # Scale Data
    scaler_dwell = StandardScaler()
    scaler_flight = StandardScaler()

    scaled_dwells = []
    scaled_flights = []

    for d in raw_dwells:
        scaler_dwell.learn_one({"x": d})
        scaled_dwells.append(scaler_dwell.transform_one({"x": d})["x"])

    for f in raw_flights:
        scaler_flight.learn_one({"x": f})
        scaled_flights.append(scaler_flight.transform_one({"x": f})["x"])

    # 4. Visualization (Save to File)
    print("📈 Generating Plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Sentinel Biometric Analysis ({len(raw_dwells)} Keystrokes)", fontsize=16
    )

    # Dwell Time
    axes[0, 0].hist(raw_dwells, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    axes[0, 0].set_title(
        f"Original Dwell (ms)\nMean: {np.mean(raw_dwells):.1f} | Std: {np.std(raw_dwells):.1f}"
    )

    axes[1, 0].hist(scaled_dwells, bins=30, color="green", edgecolor="black", alpha=0.7)
    axes[1, 0].set_title(
        f"Scaled Dwell (Z-Score)\nMean: {np.mean(scaled_dwells):.2f} | Std: {np.std(scaled_dwells):.2f}"
    )
    axes[1, 0].axvline(0, color="red", linestyle="--", alpha=0.5)

    # Flight Time
    axes[0, 1].hist(raw_flights, bins=30, color="salmon", edgecolor="black", alpha=0.7)
    axes[0, 1].set_title(
        f"Original Flight (ms)\nMean: {np.mean(raw_flights):.1f} | Std: {np.std(raw_flights):.1f}"
    )
    # Highlight Rollover
    min_flight = min(raw_flights) if raw_flights else 0
    if min_flight < 0:
        axes[0, 1].axvspan(min_flight, 0, color="red", alpha=0.1, label="Rollover")
        axes[0, 1].legend()

    axes[1, 1].hist(
        scaled_flights, bins=30, color="purple", edgecolor="black", alpha=0.7
    )
    axes[1, 1].set_title(
        f"Scaled Flight (Z-Score)\nMean: {np.mean(scaled_flights):.2f} | Std: {np.std(scaled_flights):.2f}"
    )
    axes[1, 1].axvline(0, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()

    # Save instead of show
    output_filename = "sentinel_analysis.png"
    plt.savefig(output_filename)
    print(f"\n✅ Analysis saved to: {os.path.abspath(output_filename)}")
    print("   Open this file to view the histograms.")


if __name__ == "__main__":
    main()
