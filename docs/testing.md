# Testing Philosophy & Standards

This document defines how Sentinel-ML is tested, what guarantees those tests provide,
and the invariants that every contributor must maintain.

> **This file answers:** "How do I know the system is correctly verified?"

---

## Test Philosophy

Sentinel-ML uses a **three-tier testing strategy**:

### 1. Unit Tests (Fast, Isolated)

- **Scope**: Individual functions, classes, and methods
- **Dependencies**: None external (no Redis, no Supabase)
- **Data**: Synthetic fixtures or recorded human data from CSVs
- **Speed**: < 100ms per test
- **Purpose**: Verify correctness of business logic in isolation

```python
def test_dwell_time_calculation(processor):
    down = KeyboardEvent(key="a", event_type=KeyEventType.DOWN, timestamp=1000)
    up = KeyboardEvent(key="a", event_type=KeyEventType.UP, timestamp=1150)
    processor.process_event(down)
    processor.process_event(up)
    assert processor.last_dwell_time == 150.0
```

### 2. Integration Tests (Real Dependencies)

- **Scope**: Multi-component flows (API → Orchestrator → Persistence)
- **Dependencies**: Real Redis, real Supabase
- **Data**: Unique session IDs with timestamps for isolation
- **Speed**: < 30s total for the integration suite
- **Purpose**: Verify the system works end-to-end with real infrastructure

```python
def test_keyboard_then_evaluate(client):
    # Stream keyboard events via API
    response = client.post("/stream/keyboard", json=keyboard_payload)
    assert response.status_code == 204
    
    # Then evaluate via API
    response = client.post("/evaluate", json=eval_payload)
    assert response.status_code == 200
    assert response.json()["decision"] in ["ALLOW", "CHALLENGE", "BLOCK"]
```

### 3. Property-Based Tests (Invariant Verification)

- **Scope**: Algorithmic invariants that must always hold
- **Method**: Test with boundary values, random inputs, and edge cases
- **Purpose**: Catch edge cases humans wouldn't think to test

Examples:
- Risk scores are always in [0.0, 1.0]
- Dwell times are always positive
- Physics violations always trigger BLOCK

---

## Source → Test Mapping Rules

Every production source file has a corresponding test file following this pattern:

```
core/<module>/<file>.py → tests/<module>/test_<file>.py
```

### Mapping Table

| Source File | Test File |
|-------------|-----------|
| `core/processors/keyboard.py` | `tests/processors/test_keyboard_processor.py` |
| `core/processors/mouse.py` | `tests/processors/test_mouse_processor.py` |
| `core/models/keyboard.py` | `tests/models/test_keyboard_model.py` |
| `core/models/mouse.py` | `tests/models/test_mouse_model.py` |
| `core/models/navigator.py` | `tests/models/test_navigator_model.py` |
| `core/schemas/inputs.py` | `tests/schemas/test_schemas.py` |
| `core/schemas/outputs.py` | `tests/schemas/test_schemas.py` |
| `core/orchestrator.py` | `tests/test_orchestrator.py` |
| `main.py` | `tests/test_api.py` |

### Naming Conventions

- Test files: `test_<source_name>.py`
- Test classes: `TestFeatureName` or `TestClassName`
- Test methods: `test_<what_is_being_tested>`

---

## What MUST Be Tested

### Always Test

| Category | Examples | Why |
|----------|----------|-----|
| **Public APIs** | Every endpoint in `main.py` | Contract guarantee |
| **Decision Logic** | ALLOW/CHALLENGE/BLOCK thresholds | Security-critical paths |
| **Physics Rules** | Teleport detection, linearity checks | Bot detection accuracy |
| **Boundary Conditions** | Min/max values, empty inputs | Prevent edge case bugs |
| **Error Paths** | Invalid payloads, rate limiting | Graceful degradation |
| **Idempotency** | `eval_id` deduplication | Correctness under retry |

### Test Data Requirements

| Type | Requirement |
|------|-------------|
| **Human Behavior** | Use real recorded CSVs (`human_keyboard_recording.csv`, `human_mouse_recording.csv`) |
| **Bot Patterns** | Generate synthetic data with known anomalies (constant timing, perfect lines) |
| **Cold Start** | Test with zero prior data (no identity, no model) |
| **Session Isolation** | Use `test_` prefix and timestamp suffixes for unique session IDs |

---

## What Should NOT Be Tested

| Don't Test | Reason |
|------------|--------|
| Third-party library internals | e.g., Pydantic's serialization, River's HST internals |
| Private methods (`_method`) | Implementation details that may change |
| Logging output content | Side effects, not behavior |
| Exact floating-point values | Use `pytest.approx()` for numerical comparisons |
| Supabase/Redis configuration | Infrastructure, not application logic |

---

## Test Invariants

These properties **must always hold** and are verified by tests:

### Risk Scores

```python
assert 0.0 <= result.risk <= 1.0  # Always bounded
```

### Decision Thresholds

```python
if result.risk >= 0.8:
    assert result.decision == SentinelDecision.BLOCK
elif result.risk >= 0.5:
    assert result.decision == SentinelDecision.CHALLENGE
else:
    assert result.decision == SentinelDecision.ALLOW
```

### Physics Violations

```python
# Teleportation (>3000 px/s) → BLOCK
if velocity > 3000:
    assert result.decision == SentinelDecision.BLOCK

# Perfect linearity (<0.5px error) → Bot suspected
if linearity_error < 0.5:
    assert result.risk > 0.5
```

### Timing Invariants

```python
assert dwell_time >= 0  # No negative durations
assert flight_time >= 0
```

### Idempotency

```python
result1 = orchestrator.evaluate(payload_with_eval_id)
result2 = orchestrator.evaluate(payload_with_eval_id)
assert result1.decision == result2.decision
assert result1.risk == result2.risk
```

---

## CI Enforcement

### Required Checks

| Check | Command | Threshold |
|-------|---------|-----------|
| All tests pass | `pytest tests/` | 100% |
| Coverage (models) | `pytest --cov=core/models` | ≥ 90% |
| Coverage (processors) | `pytest --cov=core/processors` | ≥ 90% |
| Coverage (schemas) | `pytest --cov=core/schemas` | 100% |

### Pre-Commit Workflow

```bash
# Run before every commit
pytest tests/ -v -x --tb=short

# Full coverage check before merge
pytest tests/ --cov=core --cov-report=term-missing
```

### Coverage Goals

| Module | Target | Current |
|--------|--------|---------|
| `core/processors/` | 90%+ | ✅ |
| `core/models/` | 90%+ | ✅ |
| `core/schemas/` | 100% | ✅ |
| `core/orchestrator.py` | 80%+ | ✅ |
| `main.py` (API) | 80%+ | ✅ |

---

## Test Data Assets

### Human Behavior Recordings

Located in `tests/assets/`:

| File | Events | Duration | Use Case |
|------|--------|----------|----------|
| `human_keyboard_recording.csv` | ~1,500 | ~109s | Train HST, verify low anomaly scores |
| `human_mouse_recording.csv` | ~500 | ~30s | Verify physics, extract stroke features |

### Generator Scripts

Located in `tests/assets/generators/`:

| Script | Purpose |
|--------|---------|
| `record_keystrokes.py` | Record real human typing (run locally) |
| `record_mouse.py` | Record real human mouse movements |
| `bot_generators.py` | Generate synthetic bot patterns for testing |

### Bot Pattern Generators (Inline)

```python
def generate_bot_keyboard_events(base_ts, count=50):
    """Perfectly timed keystrokes (0 variance)."""
    events = []
    for i in range(count):
        events.append(KeyboardEvent(key=chr(97+i%26), event_type=DOWN, timestamp=base_ts + i*150))
        events.append(KeyboardEvent(key=chr(97+i%26), event_type=UP, timestamp=base_ts + i*150 + 100))
    return events
```

---

## Adding New Tests

### Checklist

1. ☐ Create test file matching source: `core/foo/bar.py` → `tests/foo/test_bar.py`
2. ☐ Import shared fixtures from `conftest.py`
3. ☐ Cover all public methods of the class/module
4. ☐ Include both success and failure cases
5. ☐ Use real human data for human behavior tests
6. ☐ Use synthetic data with known properties for bot tests
7. ☐ Verify invariants hold (score bounds, decision thresholds)
8. ☐ Run `pytest tests/ -v` before committing

### Test Class Template

```python
"""
Tests for core/module/feature.py
"""
import pytest
from core.module.feature import FeatureClass


class TestFeatureClass:
    """Tests for FeatureClass functionality."""
    
    def test_basic_operation(self, fixture):
        """Verify basic functionality works."""
        result = fixture.do_something()
        assert result is not None
    
    def test_edge_case_empty_input(self, fixture):
        """Empty input should return safe default."""
        result = fixture.do_something([])
        assert result == expected_default
    
    def test_boundary_max_value(self, fixture):
        """Maximum values are handled correctly."""
        result = fixture.do_something(MAX_VALUE)
        assert 0.0 <= result <= 1.0


class TestFeatureClassBotDetection:
    """Bot detection tests for FeatureClass."""
    
    def test_detects_constant_timing(self, fixture):
        """Zero-variance timing triggers high score."""
        bot_data = generate_constant_timing()
        score = fixture.compute_score(bot_data)
        assert score > 0.8
```

---

## Summary

| Question | Answer |
|----------|--------|
| How do I run tests? | `pytest tests/` |
| Where's the test for X? | `tests/<module>/test_<X>.py` |
| What must be tested? | Public APIs, decisions, physics, boundaries |
| What shouldn't be tested? | Library internals, private methods, logging |
| How is quality enforced? | 90%+ coverage, all tests pass in CI |
| What invariants must hold? | Score bounds, decision thresholds, idempotency |
