# Testing Sentinel

Sentinel is a security product, so "it works on my machine" is not enough. We rely on a tiered testing strategy.

## 1. Unit Tests (Fast)
**Location**: `tests/keyboard/`, `tests/mouse/`, `tests/navigator/`
*   **Goal**: Verify individual components in isolation.
*   **Key Tests**:
    *   `test_features.py`: Ensure mathematical vectors (velocity, jerk) are calculated correctly.
    *   `test_streams.py`: Ensure stream payloads parse correctly or raise 400.

## 2. Integration Tests (Slower)
**Location**: `tests/integration/`
*   **Goal**: Ensure the Orchestrator, Redis, and Models talk to each other.
*   **Key Tests**:
    *   `test_orchestrator_integration.py`: Feeds 100 events, expects a Trust Score update in Redis.

## 3. The "Bot" Test Suite
**Location**: `tests/integration/test_bot_scenarios.py`
We do not just test for success; we test for failure. This suite runs simulated attacks against the system.

### Scenarios
1.  **The Teleporter**: Mouse moves from (0,0) to (1000,1000) in 1ms.
    *   *Expectation*: `PhysicsModel` triggers. Risk = 1.0. BLOCK.
2.  **The Replay**: Send the exact same batch of events twice.
    *   *Expectation*: Orchestrator rejects Batch 2 with `400 Bad Request` or `429`.
3.  **The Perfect Liner**: Mouse moves in a mathematically perfect line.
    *   *Expectation*: `LinearityScore` is 1.0. Risk increases.

## 4. Synthetic Data Generation
We use a custom `Generator` class to create mock streams.
*   `generate_human_stream()`: Adds Perlin noise to mouse paths and Gaussian jitter to keystrokes.
*   `generate_bot_stream()`: Creates linear paths and fixed timings.

Run the generator:
```bash
python scripts/generate_traffic.py --type=human --count=1000
```
