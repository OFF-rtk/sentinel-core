# Learning Safety & Anti-Poisoning

Online learning (updating models in real-time) is powerful but dangerous. If an attacker can feed data into the model, they can "teach" it that their attack is normal behavior. Sentinel prevents this through **Gated Selective Learning**.

## The Poisoning Problem

In a naive online system, every new data point updates the model.
*   *Attack Scenario*: A bot starts typing exactly like a human (High score). Then, over 10,000 events, it slowly shifts its timing by 0.1ms per event until it is typing like a machine. If the model updates on every event, it will "drift" with the attacker.

## Sentinel's Solution: The Trust Gate

Sentinel enforces a strict rule: **"Only learn from the best."**
*Disclaimer: Selective Learning prioritizes security over recall and may slow personalization for edge users.*

### 1. HST Learning (Anomaly Baseline)
The HST (Half-Space Trees) model learns what "normal" typing looks like. It has a **cold-start-aware mode gate**:

```
Cold start (HST < 50 windows): learn in ANY mode → fast bootstrap
Post cold start:               learn only in NORMAL mode → anti-poisoning
```

During cold start, the system forces a CHALLENGE on each action to collect typing data. It learns from the completed windows on *both* ALLOW and CHALLENGE decisions, then clears the windows to force the next CHALLENGE. This continues until the model reaches 50 feature windows.

After cold start, HST only learns on ALLOW decisions in NORMAL mode when learning suspension is not active.

### 2. Identity Learning (Per-User Profile)
We only trigger an identity model update (`model.learn_one()`) if the session meets strict criteria:

```
ShouldLearn = (Mode == "NORMAL") ∧
              (LearningNotSuspended) ∧
              (NavigatorRisk < 0.5) ∧
              (TrustScore ≥ 0.65) ∧
              (ConsecutiveAllows ≥ 5) ∧
              (ContextStable for 30s)
```

If the Trust Score drops below 0.65, learning is immediately disabled for that user. This creates a "ratchet" effect: you can lose trust easily, but you can only define "normal" when you are beyond suspicion.

### 3. Concurrent Safety
Model updates use **optimistic locking with retry** (`learn_with_retry`). The pattern:
1. Load model from Supabase with its current `feature_window_count` and version.
2. Apply `learn_one()` for each feature window.
3. Save back with an incremented count, using a conditional update that fails if another worker changed the model concurrently.
4. On conflict, retry from step 1 (re-loads the updated model and re-applies learning).

This prevents lost updates when multiple API workers process the same user's events simultaneously.

### 4. Learning Suspension
When navigator risk spikes (≥ 0.85), learning is suspended for 30 seconds. It only resumes after 60 seconds of clean activity (navigator risk < 0.5). This prevents context-switch attacks where an attacker triggers a device change and immediately tries to feed data.

## Protection Against Specific Attacks

### Replay Attacks
Since replay attacks are detected by the Orchestrator (via `batch_id` high-water-mark and jitter checks), they result in an immediate `BLOCK`. The learning gate closes because BLOCK decisions never trigger learning.

### Slow-Roll Poisoning
If an attacker tries to drift slowly:
1.  They start high (Learning is ON).
2.  They drift slightly.
3.  The Anomaly Score increases slightly.
4.  The Trust Score drops slightly (e.g., 0.9 -> 0.8).
5.  **Gate Closes**. Learning stops at trust < 0.65.
6.  The attacker continues to drift.
7.  Since the model stopped updating at step 5, the attacker's further drift now looks radically different from the frozen model state.
8.  Trust crashes. Block.
