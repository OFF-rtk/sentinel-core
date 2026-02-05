# Learning Safety & Anti-Poisoning

Online learning (updating models in real-time) is powerful but dangerous. If an attacker can feed data into the model, they can "teach" it that their attack is normal behavior. Sentinel prevents this through **Gated Selective Learning**.

## The Poisoning Problem

In a naive online system, every new data point updates the model.
*   *Attack Scenario*: A bot starts typing exactly like a human (High score). Then, over 10,000 events, it slowly shifts its timing by 0.1ms per event until it is typing like a machine. If the model updates on every event, it will "drift" with the attacker.

## Sentinel's Solution: The Trust Gate

Sentinel enforces a strict rule: **"Only learn from the best."**
*Disclaimer: Selective Learning prioritizes security over recall and may slow personalization for edge users.*

### 1. The Gating Logic
We only trigger a model update (`model.learn_one()`) if the session meets strict criteria:

```
ShouldLearn = (Mode == "NORMAL") ∧
              (TrustScore ≥ 0.65) ∧
              (NavigatorRisk < 0.5) ∧
              (ConsecutiveAllows ≥ 5) ∧
              (NotInColdStart) ∧
              (ContextStable for 30s)
```

If the Trust Score drops below 0.65, learning is immediately disabled for that user. This creates a "ratchet" effect: you can lose trust easily, but you can only define "normal" when you are beyond suspicion.

### 2. Time-Delayed Commitment
We do not persist model updates to the database immediately. Updates are kept in memory (Redis) during the session.
*   **Commit**: Only when the session ends gracefully with a High Trust score.
*   **Discard**: If the session ends in a BLOCK or CHALLENGE, the in-memory updates are discarded. The model reverts to its state before the attack began.

## Protection Against Specific Attacks

### Replay Attacks
Since replay attacks are detected by the Orchestrator (via `batch_id` sequencing and jitter checks), they result in an immediate `BLOCK`. Because of the Time-Delayed Commitment, the model never sees these events as "valid" training data.

### Slow-Roll Poisoning
If an attacker tries to drift slowly:
1.  They start high (Learning is ON).
2.  They drift slightly.
3.  The Anomaly Score increases slightly.
4.  The Trust Score drops slightly (e.g., 0.9 -> 0.8).
5.  **Gate Closes**. Learning stops.
6.  The attacker continues to drift.
7.  Since the model stopped updating at step 5, the attacker's further drift now looks radically different from the frozen model state.
8.  Trust crashes. Block.
