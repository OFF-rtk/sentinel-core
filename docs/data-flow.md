# Data Flow Architecture

This document maps the runtime path of a data packet from the user's hand to the final decision code.

## 1. Telemetry Ingestion
**File**: `main.py` (FastAPI)

Data enters via `POST /stream/mouse` or `POST /stream/keyboard`.
*   **Validation**: Pydantic schemas (`KeyboardStreamPayload`) validate types and ranges.
*   **Buffering**: Data is not processed immediately; it is pushed to the Orchestrator.

## 2. Feature Extraction
**File**: `core/processors/`

Raw events (x, y, timestamp) are converted into statistical vectors.
*   **Keyboard**: Inter-key latency, flight time, dwell time.
*   **Mouse**: Velocity, acceleration, jerk, angular velocity, curvature.

## 3. Model Scoring
**File**: `core/models/`

The feature vector is passed to multiple models in parallel.
*   **Physics Check**: `PhysicsMouseModel.predict()` checks for hard limits.
*   **Anomaly detection**: Online Anomaly Models (River Half-Space Trees) scores the vector against the generic "Human" cluster.
*   **Identity verification**: If the user has a profile, their specific `River` model scores the vector.

## 4. Orchestration
**File**: `core/orchestrator.py`

The `Orchestrator` class manages the lifecycle.
*   It retrieves the current `SessionState` from Redis.
*   It combines the model scores using the Fusion Logic.
*   It updates the `SessionState` with the new Trust Score.

## 5. Persistence
**File**: `persistence/model_store.py`

*   **Hot State**: The updated `SessionState` is written back to Redis (TTL ~30m).
*   **Cold State**: If the session ends or hits a learning checkpoint, the updated Model parameters are serialized and saved to Supabase.

## 6. Decision Output (Conceptual)

When `POST /evaluate` is called:
1.  Read current Trust Score from Redis.
2.  Compare against Thresholds (defined in config).
    *   Score > 0.8: `ALLOW` (Mode: NORMAL)
    *   Score < 0.5: `CHALLENGE` (Mode: HIGH_ALERT)
    *   Score < 0.1: `BLOCK`
3.  Return JSON response.
