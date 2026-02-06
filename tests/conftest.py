"""
Sentinel-ML Test Suite - Shared Fixtures

This module provides common fixtures used across all test modules.
Import these fixtures by simply including them as test function parameters.
"""

import csv
import pytest
from pathlib import Path

from core.processors.keyboard import KeyboardProcessor
from core.processors.mouse import MouseProcessor
from core.models.keyboard import KeyboardAnomalyModel
from core.models.mouse import PhysicsMouseModel, MouseSessionTracker
from core.models.navigator import NavigatorPolicyEngine
from core.schemas.inputs import KeyboardEvent, KeyEventType, MouseEvent, MouseEventType


# =============================================================================
# Path Constants
# =============================================================================

ASSETS_DIR = Path(__file__).parent / "assets"
HUMAN_KEYBOARD_CSV = ASSETS_DIR / "human_keyboard_recording.csv"
HUMAN_MOUSE_CSV = ASSETS_DIR / "human_mouse_recording.csv"


# =============================================================================
# Processor Fixtures
# =============================================================================

@pytest.fixture
def keyboard_processor():
    """Fresh KeyboardProcessor for each test."""
    return KeyboardProcessor()


@pytest.fixture
def mouse_processor():
    """Fresh MouseProcessor for each test."""
    return MouseProcessor()


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def keyboard_model():
    """Fresh KeyboardAnomalyModel for each test."""
    return KeyboardAnomalyModel()


@pytest.fixture
def mouse_model():
    """Fresh PhysicsMouseModel for each test."""
    return PhysicsMouseModel()


@pytest.fixture
def mouse_session_tracker():
    """Fresh MouseSessionTracker for each test."""
    return MouseSessionTracker()


@pytest.fixture
def navigator_engine():
    """Fresh NavigatorPolicyEngine for each test."""
    return NavigatorPolicyEngine()


# =============================================================================
# Human Data Fixtures
# =============================================================================

@pytest.fixture
def human_keyboard_events():
    """Load human keyboard recording from CSV."""
    if not HUMAN_KEYBOARD_CSV.exists():
        pytest.skip(f"Human keyboard recording not found: {HUMAN_KEYBOARD_CSV}")
    
    events = []
    with open(HUMAN_KEYBOARD_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_type = KeyEventType.DOWN if row['event_type'] == 'DOWN' else KeyEventType.UP
            events.append(KeyboardEvent(
                key=row['key'],
                event_type=event_type,
                timestamp=float(row['timestamp'])
            ))
    
    return events


@pytest.fixture
def human_mouse_events():
    """Load human mouse recording from CSV."""
    if not HUMAN_MOUSE_CSV.exists():
        pytest.skip(f"Human mouse recording not found: {HUMAN_MOUSE_CSV}")
    
    events = []
    with open(HUMAN_MOUSE_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_type = MouseEventType.CLICK if row['event_type'] == 'CLICK' else MouseEventType.MOVE
            events.append(MouseEvent(
                x=int(row['x']),
                y=int(row['y']),
                event_type=event_type,
                timestamp=float(row['timestamp'])
            ))
    
    return events


# =============================================================================
# Feature Extraction Fixtures
# =============================================================================

@pytest.fixture
def human_keyboard_features(keyboard_processor, human_keyboard_events):
    """Extract keyboard features from human recording."""
    features_list = []
    
    for event in human_keyboard_events:
        result = keyboard_processor.process_event(event)
        if result is not None:
            features_list.append(result)
    
    return features_list


@pytest.fixture
def human_mouse_features(mouse_processor, human_mouse_events):
    """Extract mouse stroke features from human recording."""
    features_list = []
    
    for event in human_mouse_events[:5000]:  # Limit for speed
        result = mouse_processor.process_event(event)
        if result is not None:
            features_list.append(result)
    
    return features_list


# =============================================================================
# Bot Pattern Generators
# =============================================================================

@pytest.fixture
def bot_keyboard_constant():
    """Bot keyboard features: perfect timing, zero variance."""
    return {
        "dwell_time_mean": 100.0,
        "dwell_time_std": 0.5,
        "flight_time_mean": 50.0,
        "flight_time_std": 0.5,
        "error_rate": 0.0
    }


@pytest.fixture
def bot_keyboard_fast():
    """Bot keyboard features: impossibly fast typing."""
    return {
        "dwell_time_mean": 1.0,
        "dwell_time_std": 0.1,
        "flight_time_mean": 1.0,
        "flight_time_std": 0.1,
        "error_rate": 0.0
    }


@pytest.fixture
def bot_mouse_teleport():
    """Bot mouse features: teleport speed (Tier 1 fail)."""
    return {
        "velocity_mean": 5.0,
        "velocity_std": 1.0,
        "velocity_max": 15.0,
        "path_distance": 200.0,
        "linearity_error": 5.0,
        "time_diff_std": 10.0,
        "segment_count": 20,
    }


@pytest.fixture
def bot_mouse_perfect_line():
    """Bot mouse features: impossibly straight line (Tier 1 fail)."""
    return {
        "velocity_mean": 2.0,
        "velocity_std": 0.5,
        "velocity_max": 3.0,
        "path_distance": 350.0,
        "linearity_error": 0.1,
        "time_diff_std": 5.0,
        "segment_count": 25,
    }


# =============================================================================
# Helper Functions
# =============================================================================

def make_keyboard_event(key: str, event_type: KeyEventType, timestamp: float) -> KeyboardEvent:
    """Helper to create KeyboardEvent."""
    return KeyboardEvent(key=key, event_type=event_type, timestamp=timestamp)


def make_mouse_event(x: int, y: int, event_type: MouseEventType, timestamp: float) -> MouseEvent:
    """Helper to create MouseEvent."""
    return MouseEvent(x=x, y=y, event_type=event_type, timestamp=timestamp)
