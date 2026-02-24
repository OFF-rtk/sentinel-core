# Data Flow Architecture

This document maps the runtime path of a data packet from the user's hand to the final decision code.

## 1. Telemetry Ingestion
**File**: `main.py` (FastAPI)

Data enters via `POST /stream/mouse` or `POST /stream/keyboard`.
*   **Validation**: Pydantic schemas (`KeyboardStreamPayload`, `MouseStreamPayload`) validate types and ranges.
*   **Anti-Replay**: `batch_id` must exceed the session's high-water mark. Out-of-order delivery within a ≤10 gap is tolerated; gaps >10 trigger a state reset and a +0.5 strike.
*   **Buffering**: Data is pushed to the Orchestrator which processes it against Redis-backed state.

## 2. Feature Extraction
**File**: `core/processors/`

Raw events (x, y, timestamp) are converted into statistical vectors.
*   **Keyboard**: Dwell time (key down → up), flight time (key up → next key down), mean/std/min/max per feature window.
*   **Mouse**: Velocity, acceleration, jerk, angular velocity, curvature across stroke segments.

## 3. Model Scoring
**File**: `core/models/`

The feature vector is passed to multiple models:
*   **Physics Check**: `PhysicsMouseModel.score_one()` checks for hard velocity, linearity, and timing limits using tiered detection (Tier 1 hard fails, Tier 2 additive risk, Tier 3 threshold decision).
*   **Mouse Teleportation**: The Orchestrator counts MOVE events between CLICKs. Fewer than 3 MOVEs before a click indicates cursor teleportation (physically impossible with a real mouse). The ratio of teleported clicks to total clicks becomes a risk signal.
*   **HST Anomaly**: River Half-Space Trees model (`KeyboardAnomalyModel`) scores the keyboard feature vector against the learned "normal" typing cluster. Persistent per-user in Supabase.
*   **Identity Verification**: If the user has ≥150 feature windows in their identity model, their personal HST scores the latest 3–5 windows and averages the result.

## 4. Orchestration
**File**: `core/orchestrator.py`

The `SentinelOrchestrator` class manages the lifecycle:
*   Retrieves the current `SessionState` from Redis (optimistic locking via WATCH/MULTI/EXEC).
*   Computes `NavigatorPolicyEngine` risk (unknown user agent, impossible travel, device mismatch).
*   Applies keyboard confidence gating (geometric mean of time and count maturity).
*   Evaluates hard-block priority chain: strikes ≥ 3 → physics violation → navigator BLOCK → identity contradiction → immature identity guard.
*   Fuses remaining signals using Weighted SUM (keyboard + mouse + navigator + identity, clamped to [0.0, 1.0]).
*   Updates trust score using the linear delta stabilizer.

## 5. Learning
**Files**: `persistence/model_store.py`, `core/orchestrator.py`

*   **HST Cold Start**: When <50 feature windows, the model forces CHALLENGE on every action, learns from the typed text (both ALLOW and CHALLENGE decisions), then clears windows so the next action triggers another CHALLENGE.
*   **HST Post Cold Start**: Only learns during NORMAL mode ALLOW decisions when learning is not suspended.
*   **Identity**: Only learns when `mode == NORMAL`, `trust ≥ 0.65`, `navigator_risk < 0.50`, `consecutive_allows ≥ 5`, and context has been stable for 30 seconds. Uses retry loop for concurrent safety.
*   **Time-Delayed Commitment**: Model updates persist immediately via `learn_with_retry()` (optimistic locking). If a session ends in BLOCK, trust is reset but learned data stays — the model is robust because the learning gate prevents bad data from ever reaching it.

## 6. Decision Output

When `POST /evaluate` is called:
1.  Retrieve session state (keyboard score, mouse score, trust score) from Redis.
2.  Compute mouse teleportation ratio from click/move counters.
3.  Compute navigator risk via `NavigatorPolicyEngine.evaluate()`.
4.  Apply TOFU (Trust On First Use) — first session gets navigator risk zeroed.
5.  Apply keyboard confidence gating (geometric mean).
6.  Check hard-block priority chain.
7.  Apply Weighted SUM Fusion across all four components.
8.  Compare **final risk** against mode-specific thresholds:

    | Mode      | ALLOW         | CHALLENGE           | BLOCK        |
    |-----------|---------------|---------------------|--------------|
    | NORMAL    | risk < 0.50   | 0.50 ≤ risk < 0.85  | risk ≥ 0.85  |
    | CHALLENGE | risk < 0.40   | 0.40 ≤ risk < 0.75  | risk ≥ 0.75  |
    | TRUSTED   | risk < 0.60   | 0.60 ≤ risk < 0.92  | risk ≥ 0.92  |

9.  Apply HST cold start override (force CHALLENGE if <50 windows and no fresh typing data).
10. Update trust score, strikes, and mode.
11. Trigger selective learning (HST and identity) if conditions are met.
12. Write structured audit log to Supabase `audit_logs` table.
13. On BLOCK: set provisional 5-minute ban in Redis.
14. Return JSON: `{ "decision": "ALLOW|CHALLENGE|BLOCK", "risk": float, "mode": "NORMAL|CHALLENGE", "anomaly_vectors": [...] }`
