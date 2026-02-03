#!/usr/bin/env python3
"""
Sentinel Mouse Movement Recorder (Linux/evdev)

Records all mouse/touchpad movements and clicks to CSV using kernel timestamps.
Runs in background until Ctrl+C.

Usage:
    sudo python record_mouse.py                    # Default: mouse_recording.csv
    sudo python record_mouse.py my_session.csv    # Custom output file

Output CSV format:
    timestamp,x,y,event_type
    1712345678123.456,512,384,MOVE
    1712345678150.789,520,390,CLICK
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
# Device Detection
# =============================================================================

def find_mouse_device():
    """Auto-detect the best mouse/touchpad device."""
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    
    # Look for touchpad with ABS_X/ABS_Y (most laptops)
    for device in devices:
        caps = device.capabilities()
        name_lower = device.name.lower()
        
        # Prefer actual touchpad device
        if 'touchpad' in name_lower:
            return device, "touchpad"
    
    # Fallback: any device with ABS_X and ABS_Y
    for device in devices:
        caps = device.capabilities()
        has_abs = ecodes.EV_ABS in caps
        has_key = ecodes.EV_KEY in caps
        
        if has_abs and has_key:
            abs_caps = [code for code, _ in caps.get(ecodes.EV_ABS, [])]
            if ecodes.ABS_X in abs_caps and ecodes.ABS_Y in abs_caps:
                return device, "touchpad"
    
    # Fallback: traditional mouse with REL_X/REL_Y
    for device in devices:
        caps = device.capabilities()
        has_rel = ecodes.EV_REL in caps
        
        if has_rel:
            rel_caps = caps.get(ecodes.EV_REL, [])
            if ecodes.REL_X in rel_caps and ecodes.REL_Y in rel_caps:
                return device, "mouse"
    
    return None, None


# =============================================================================
# Main Recorder
# =============================================================================

class MouseRecorder:
    """Records mouse events to CSV with kernel timestamps."""
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.running = True
        self.event_count = 0
        self.start_time = time.time()
        
        # Current cursor position
        self.x = 0
        self.y = 0
        self.last_x = None
        self.last_y = None
        
        # Pending state
        self.has_pending = False
        
        # Device type
        self.device_type = "unknown"
        
        # CSV writer
        self.csv_file = None
        self.csv_writer = None
    
    def start(self):
        """Start recording mouse events."""
        result = find_mouse_device()
        if result[0] is None:
            print("❌ No mouse/touchpad detected!")
            print("   Make sure you're running with: sudo")
            sys.exit(1)
        
        device, self.device_type = result
        
        print("=" * 60)
        print("🖱️  SENTINEL MOUSE RECORDER")
        print("=" * 60)
        print(f"Device: {device.name}")
        print(f"Path: {device.path}")
        print(f"Type: {self.device_type}")
        print(f"Output: {self.output_path.absolute()}")
        print("-" * 60)
        print("Recording... Use laptop normally. Press Ctrl+C to stop.")
        print("-" * 60)
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        # Open CSV file
        self.csv_file = open(self.output_path, 'w', newline='', buffering=1)
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'x', 'y', 'event_type'])
        
        try:
            self._record_loop(device)
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            self._finalize()
    
    def _record_loop(self, device):
        """Main event loop with non-blocking reads for clean shutdown."""
        import select
        
        while self.running:
            # Use select with 0.5s timeout so we can check self.running
            r, w, x = select.select([device.fd], [], [], 0.5)
            
            if not r:
                # Timeout, no events - loop back to check self.running
                continue
            
            # Read all available events
            for event in device.read():
                if not self.running:
                    return
                
                # Kernel timestamp in milliseconds
                timestamp = event.timestamp() * 1000.0
                
                if event.type == ecodes.EV_REL:
                    # Relative mouse movement (external mouse)
                    if event.code == ecodes.REL_X:
                        self.x += event.value
                        self.has_pending = True
                    elif event.code == ecodes.REL_Y:
                        self.y += event.value
                        self.has_pending = True
                        
                elif event.type == ecodes.EV_ABS:
                    # Absolute position (touchpad)
                    if event.code == ecodes.ABS_X:
                        self.x = event.value
                        self.has_pending = True
                    elif event.code == ecodes.ABS_Y:
                        self.y = event.value
                        self.has_pending = True
                        
                elif event.type == ecodes.EV_KEY:
                    # Mouse button / touchpad tap
                    if event.code in [ecodes.BTN_LEFT, ecodes.BTN_RIGHT, ecodes.BTN_MIDDLE,
                                      ecodes.BTN_TOUCH]:
                        if event.value == 1:  # Button down
                            self._write_event(timestamp, self.x, self.y, 'CLICK')
                            
                elif event.type == ecodes.EV_SYN and event.code == ecodes.SYN_REPORT:
                    # Sync event - emit pending movement
                    if self.has_pending:
                        # Only emit if position actually changed
                        if self.last_x != self.x or self.last_y != self.y:
                            self._write_event(timestamp, self.x, self.y, 'MOVE')
                            self.last_x = self.x
                            self.last_y = self.y
                        
                        self.has_pending = False
    
    def _write_event(self, timestamp: float, x: int, y: int, event_type: str):
        """Write a single event to CSV."""
        self.csv_writer.writerow([f"{timestamp:.3f}", x, y, event_type])
        self.event_count += 1
        
        # Progress update every 500 events
        if self.event_count % 500 == 0:
            elapsed = time.time() - self.start_time
            rate = self.event_count / elapsed if elapsed > 0 else 0
            print(f"\r📊 Events: {self.event_count:,} | Rate: {rate:.0f}/s | Elapsed: {elapsed:.0f}s", end="", flush=True)
    
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
        if self.output_path.exists():
            with open(self.output_path) as f:
                lines = f.readlines()[:10]
            print("\nPreview:")
            for line in lines:
                print(f"  {line.rstrip()}")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    output_file = sys.argv[1] if len(sys.argv) > 1 else "mouse_recording.csv"
    recorder = MouseRecorder(output_file)
    recorder.start()


if __name__ == "__main__":
    main()
