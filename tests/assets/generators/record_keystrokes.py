#!/usr/bin/env python3
"""
Sentinel Keystroke Recorder - CSV Export for Testing

Records keystrokes using evdev and saves to CSV format for test assets.
Output CSV format: key,event_type,timestamp

Usage:
    sudo python record_keystrokes.py                     # Default output
    sudo python record_keystrokes.py my_recording.csv   # Custom output
"""

import csv
import signal
import sys
import time
from pathlib import Path

try:
    import evdev
    from evdev import ecodes
except ImportError:
    print("❌ Error: 'evdev' not found. Run: pip install evdev")
    sys.exit(1)


# =============================================================================
# Key Mapping
# =============================================================================

CHAR_MAP = {
    ecodes.KEY_SPACE: " ",
    ecodes.KEY_ENTER: "Enter",
    ecodes.KEY_TAB: "Tab",
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
    ecodes.KEY_BACKSPACE: "Backspace",
    ecodes.KEY_DELETE: "Delete",
}


def find_keyboard_device():
    """Auto-detect the first keyboard device."""
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    for device in devices:
        if "keyboard" in device.name.lower():
            return device
    return None


# =============================================================================
# Recorder Class
# =============================================================================

class KeystrokeRecorder:
    """Records keystrokes to CSV with kernel timestamps."""
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.running = True
        self.event_count = 0
        self.start_time = time.time()
        self.csv_file = None
        self.csv_writer = None
    
    def start(self):
        """Start recording keystrokes."""
        device = find_keyboard_device()
        if not device:
            print("❌ No keyboard detected! Are you running with sudo?")
            sys.exit(1)
        
        print("=" * 60)
        print("⌨️  SENTINEL KEYSTROKE RECORDER")
        print("=" * 60)
        print(f"Device: {device.name}")
        print(f"Output: {self.output_path.absolute()}")
        print("-" * 60)
        print("Instructions:")
        print("1. Type the paragraph shown below naturally.")
        print("2. Press ESC when finished to save and exit.")
        print("-" * 60)
        print()
        print("📝 PARAGRAPH TO TYPE:")
        print()
        print('"""')
        print("The quick brown fox jumps over the lazy dog near the riverbank.")
        print("Security systems must balance usability with protection against threats.")
        print("Behavioral biometrics analyze how users interact with their devices.")
        print("Keystroke dynamics measure the unique rhythm of each person typing.")
        print("Machine learning models can detect anomalies in typing patterns.")
        print("When suspicious activity is detected the system may challenge users.")
        print("This approach provides continuous authentication without interruption.")
        print("The Half Space Trees algorithm learns incrementally from streaming data.")
        print("Each observation updates the model making it adaptive to user behavior.")
        print("Testing ensures our detection system accurately identifies both humans and bots.")
        print('"""')
        print()
        print("-" * 60)
        print("🔴 RECORDING STARTED... (Type now, ESC to finish)")
        print()
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        # Open CSV file
        self.csv_file = open(self.output_path, 'w', newline='', buffering=1)
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['key', 'event_type', 'timestamp'])
        
        try:
            device.grab()
            self._record_loop(device)
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            try:
                device.ungrab()
            except:
                pass
            self._finalize()
    
    def _record_loop(self, device):
        """Main event loop."""
        import select
        
        shift_pressed = False
        
        while self.running:
            r, w, x = select.select([device.fd], [], [], 0.5)
            
            if not r:
                continue
            
            for event in device.read():
                if not self.running:
                    return
                
                if event.type != ecodes.EV_KEY:
                    continue
                
                # Ignore key hold (autorepeat)
                if event.value == 2:
                    continue
                
                # ESC to exit
                if event.code == ecodes.KEY_ESC:
                    self.running = False
                    return
                
                # Track shift state
                if event.code in [ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT]:
                    shift_pressed = event.value == 1
                    continue
                
                # Get timestamp in milliseconds
                timestamp = event.timestamp() * 1000.0
                event_type = "DOWN" if event.value == 1 else "UP"
                
                # Map key code to string
                if event.code in CHAR_MAP:
                    key = CHAR_MAP[event.code]
                else:
                    raw_key_name = evdev.ecodes.KEY.get(event.code, "UNKNOWN")
                    if isinstance(raw_key_name, str) and raw_key_name.startswith("KEY_"):
                        key = raw_key_name[4:]
                    else:
                        key = str(raw_key_name)
                
                # Apply shift for letters
                if len(key) == 1 and key.isalpha():
                    key = key.upper() if shift_pressed else key.lower()
                
                # Write to CSV
                self.csv_writer.writerow([key, event_type, f"{timestamp:.3f}"])
                self.event_count += 1
                
                # Visual feedback on key down
                if event.value == 1:
                    if key == "Enter":
                        print()
                    elif key == "Backspace":
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                    elif len(key) == 1:
                        sys.stdout.write(key)
                        sys.stdout.flush()
    
    def _handle_shutdown(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\n⏹️  Stopping...")
        self.running = False
    
    def _finalize(self):
        """Close file and print summary."""
        if self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()
        
        elapsed = time.time() - self.start_time
        print("\n" + "=" * 60)
        print("✅ RECORDING COMPLETE")
        print("=" * 60)
        print(f"Total events: {self.event_count:,}")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Saved to: {self.output_path.absolute()}")
        
        # Show preview
        if self.output_path.exists() and self.event_count > 0:
            with open(self.output_path) as f:
                lines = f.readlines()[:11]
            print("\nPreview:")
            for line in lines:
                print(f"  {line.rstrip()}")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    # Default output path: generators/ -> assets/ (parent directory)
    default_output = Path(__file__).parent.parent / "human_keyboard_recording.csv"
    output_file = sys.argv[1] if len(sys.argv) > 1 else str(default_output)
    
    recorder = KeystrokeRecorder(output_file)
    recorder.start()


if __name__ == "__main__":
    main()
