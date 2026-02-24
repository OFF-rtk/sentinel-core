# The Sentinel Decision Engine 

The heart of Sentinel is the Decision Engine, which converts raw signals into a unified trust score.

## Risk Philosophy: Time-Variant Trust

Sentinel does not believe trust is static. Just because you logged in successfully 10 minutes ago does not mean you are still the same person. Trust decays over time unless reinforced by positive behavior.

## Trust Phases

> **Note**: Trust Score ranges from 0.0 (no trust) to 1.0 (fully trusted). Lower scores indicate higher risk.

### 1. Cold Start
*   **State**: `UNKNOWN`
*   **Duration**: Until 50 keyboard feature windows are collected (`KEYBOARD_COUNT_MATURITY`) and at least 20 seconds have elapsed (`KEYBOARD_TIME_MATURITY`). Keyboard confidence is computed as `√(time_confidence × count_confidence)` (geometric mean — both time and count must contribute).
*   **Logic**: High scrutiny. Physics violations trigger immediate block. Identity models are disabled until 150 samples are collected. HST model forces CHALLENGE on every action when <50 feature windows are learned, collecting typing data during each challenge.

### 2. Trust Formation
*   **State**: `VERIFYING`
*   **Duration**: Until trust score ≥ 0.75 (`TRUSTED_THRESHOLD`).
*   **Logic**: The system aggregates positive evidence. Consistent, human-like behavior gently pushes the score up.

### 3. Mature Session
*   **State**: `TRUSTED`
*   **Duration**: Remainder of session.
*   **Logic**: The user has "proven" themselves. The system relaxes — keyboard weight drops to 0.8× and identity weight to 0.6×, and thresholds widen. Structural anomalies (e.g., changing typing patterns entirely) cause a "Trust Crash," resetting the phase.

## Logic & Gating

### Trust Stabilizer
To prevent jittery scores (e.g., 0.8 -> 0.4 -> 0.9 in seconds), Sentinel uses a **Linear Delta Update** formula for the trust score:

```
trust_delta = 0.12 × (0.5 - final_risk)
trust_score = clamp(trust_score + trust_delta, 0.0, 1.0)
```

This pushes trust up when risk is low (< 0.5) and down when risk is high (> 0.5). If identity risk exceeds 0.9, trust is immediately reset to zero.

### Gating Rules
Not all signals are equal.
*   **Physics Gate**: If a mouse movement is physically impossible (speed, linearity), risk is set to 1.0 immediately, overriding all ML models.
*   **Teleportation Gate**: If a click arrives with fewer than 3 preceding MOVE events, it counts as teleportation. The ratio of teleported clicks to total clicks is used as a risk signal. Human hand micro-tremor always produces 3+ move events before a click.
*   **Identity Gate**: We do not penalize a user for "not looking like themselves" until we have at least 150 feature windows for them (`IDENTITY_MODEL_SAMPLES_REQUIRED`). Below that, an immature identity guard issues CHALLENGE if risk ≥ 0.98.

## Weighted SUM Fusion Formula

The final risk score uses a **Weighted SUM Fusion** approach. Each component's risk is multiplied by its weight, and the results are summed and clamped to [0.0, 1.0]. This means multiple suspicious signals compound — a bot that passes one detector still accumulates risk from others.

```
final_risk = clamp(
    keyboard_risk × W_kb +
    mouse_risk × W_ms +
    navigator_risk × W_nav +
    identity_risk × confidence × W_id,
    0.0, 1.0
)
```

| Component | Weight (NORMAL) | Weight (CHALLENGE) |
|-----------|-----------------|-------------------|
| Keyboard  | 0.70            | 0.85              |
| Mouse     | 0.90            | 1.00              |
| Navigator | 1.00            | 1.00              |
| Identity  | 0.65            | 0.85              |

*Note: Mouse risk is `max(physics_score, teleportation_ratio)`. Identity weight is further scaled by `√confidence`. In TRUSTED mode, keyboard weight is ×0.8 and identity weight is ×0.6.*

## Logical Decision Flowchart

```mermaid
graph TD
    Start["Event Batch"] --> Physics{"Physics Check"}

    Physics -- Fail --> Block["BLOCK"]
    Physics -- Pass --> Teleport{"Teleportation Check"}

    Teleport --> Features["Feature Extraction"]

    Features --> Anomaly["Anomaly Model (HST)"]
    Features --> Identity["Identity Model"]

    Anomaly --> Fusion["Weighted SUM Fusion"]
    Identity --> Fusion

    Fusion --> TrustUpdate["Update Trust Score"]
    TrustUpdate --> Threshold{"Low Risk?"}

    Threshold -- Yes --> Allow["ALLOW"]
    Threshold -- No --> Challenge["CHALLENGE"]
```

## Risk Fusion & Decision Lifecycle

```mermaid
flowchart TD
    Start[Evaluate Request] --> Collect[Collect Latest Risks]
    
    Collect --> TeleportCheck[Mouse Teleportation Ratio]
    TeleportCheck --> KeyboardGate[Apply Keyboard Confidence Gating]
    KeyboardGate --> TrustAdjust[Apply Trust Modifiers]
    
    TrustAdjust --> IdentityRisk[Compute Identity Risk]
    IdentityRisk --> Fusion[Weighted SUM Fusion]
    
    Fusion --> RiskScore[Final Risk Score]
    
    RiskScore --> DecisionGate{Decision Thresholds}
    
    DecisionGate -- Low Risk --> Allow[ALLOW]
    DecisionGate -- Medium Risk --> Challenge[CHALLENGE]
    DecisionGate -- High Risk --> Block[BLOCK]
    
    Allow --> AllowPost["Increase Trust<br/>Optional Learning<br/>Audit Log"]
    Challenge --> ChallengePost["Increase Strictness<br/>No Learning<br/>Audit Log"]
    Block --> BlockPost["Increase Strikes<br/>Reset Trust<br/>Provisional Ban<br/>Audit Log"]
```

## Override & Priority Rules

```mermaid 
flowchart TD
    Start[Evaluate Request] --> StrikeCheck{"Strikes >= 3?"}
    
    StrikeCheck -- Yes --> BlockStrikes["BLOCK<br/>Strike Limit"]
    StrikeCheck -- No --> PhysicsCheck{"Mouse Risk >= 1.0?"}
    
    PhysicsCheck -- Yes --> BlockPhysics["BLOCK<br/>Non-Human Physics"]
    PhysicsCheck -- No --> NavCheck{"Navigator Decision == BLOCK?"}
    
    NavCheck -- Yes --> BlockNav["BLOCK<br/>Environment Violation"]
    NavCheck -- No --> IdentityCheck{"Identity Risk >= 0.95<br/>AND Confidence >= 0.6?"}
    
    IdentityCheck -- Yes --> BlockIdentity["BLOCK<br/>Identity Contradiction"]
    IdentityCheck -- No --> ImmatureGuard{"Identity Risk >= 0.98<br/>AND Confidence < 0.6?"}
    
    ImmatureGuard -- Yes --> ChallengeImmature["CHALLENGE<br/>Immature Identity Guard"]
    ImmatureGuard -- No --> SoftFusion[Proceed to Weighted SUM Fusion]
    
    SoftFusion --> MaxRisk[Compute Final Risk]
    
    MaxRisk --> DecisionGate{Final Risk}
    
    DecisionGate -- Low --> Allow[ALLOW]
    DecisionGate -- Medium --> Challenge[CHALLENGE]
    DecisionGate -- High --> BlockSoft[BLOCK]
```

**Thresholds by Mode:**
| Mode      | ALLOW         | CHALLENGE        | BLOCK        |
|-----------|---------------|------------------|--------------|
| NORMAL    | risk < 0.50   | 0.50 ≤ risk < 0.85 | risk ≥ 0.85 |
| CHALLENGE | risk < 0.40   | 0.40 ≤ risk < 0.75 | risk ≥ 0.75 |
| TRUSTED   | risk < 0.60   | 0.60 ≤ risk < 0.92 | risk ≥ 0.92 |
