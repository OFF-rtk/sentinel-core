# Threat Model

This document outlines the specific threats Sentinel is designed to detect, as well as its limitations and non-goals.

## Threat Categories

### 1. Automated Bots (The "Script Kiddie")
*   **Attack**: Simple scripts using Selenium, Playwright, or PyAutoGUI to perform actions.
*   **Characteristics**: Perfect straight-line mouse movements, instant keystrokes (0ms dwell time), fixed timing intervals, cursor teleportation (click events with no preceding mouse movement).
*   **Detection**:
    *   `PhysicsMouseModel`: Tiered biomechanical checks (Tier 1 hard fails for impossible speed/linearity, Tier 2 additive risk for suspicious regularity).
    *   **Teleportation detection**: The Orchestrator counts MOVE events between CLICKs. Fewer than 3 MOVEs before a click indicates the cursor teleported to the target — physically impossible with a real hand. The ratio of teleported clicks to total clicks becomes a standalone risk signal.
    *   `KeyboardAnomalyModel`: HST detects uniformly-timed keystrokes (zero jitter) as anomalous against the learned "normal human" cluster.

### 2. Replay Attacks
*   **Attack**: Recording a real human's session and replaying it later to bypass behavioral checks.
*   **Characteristics**: Valid "human" data, but identical to a previous session.
*   **Detection**:
    *   **Batch Sequencing**: The API enforces strict sequential `batch_id`s (high-water mark). Duplicate or old batch IDs are rejected. Gaps >10 trigger a state reset and strike.
    *   **Jitter Analysis**: Exact repetition of floating-point timestamps is statistically impossible in nature.
    *   **Idempotency**: `eval_id` on `/evaluate` prevents duplicate decision processing.

### 3. Mimicry / Generative Bots
*   **Attack**: Advanced AI (e.g., generative adversarial networks) treating mouse generation as a pathfinding problem to "look human."
*   **Characteristics**: Smooth curves, variable timing.
*   **Detection**: `IdentityModel`. While they look "human," they do not look like *the specific user*. They lack the user's unique motor cortex quirks. Requires 150+ feature windows for confident detection.

### 4. Model Poisoning
*   **Attack**: An attacker slowly changes their behavior over weeks to "train" the model to accept malicious patterns.
*   **Detection**: **Gated Selective Learning**. Sentinel only learns from sessions that meet strict criteria (NORMAL mode, trust ≥ 0.65, consecutive_allows ≥ 5, context stable for 30s). If an attacker acts suspiciously, the trust drops and the learning gate closes. The model freezes at its pre-attack state, making further drift detectable.

### 5. Environment Manipulation
*   **Attack**: Using unknown user agents, VPNs, or switching devices to avoid fingerprint matching.
*   **Detection**: `NavigatorPolicyEngine` detects unknown user agents, new device IDs, and IP-based anomalies. TOFU (Trust On First Use) pins the first session's context and flags deviations in subsequent sessions.

## Enforcement Pipeline

When Sentinel issues a BLOCK:
1.  **Provisional ban**: ML sets `blacklist:{user_id}` in Redis with a 5-minute TTL.
2.  **Audit log**: The evaluation result is written to Supabase, triggering a webhook to the Auditor.
3.  **Agent review**: The Auditor's LLM pipeline (Triage → Intel → Judge → CISO) reviews the event against security policies.
4.  **Confirmed ban**: If the agents confirm BLOCK, the Enforcer escalates to a 1-hour or 24-hour ban (based on strike count), overwriting the provisional TTL.
5.  **Pardon**: If the agents override to ALLOW, the Enforcer deletes the blacklist key immediately.

## Non-Goals

*   **Malware Detection**: Sentinel cannot detect if the user's machine is infected with malware that is not interacting with the input stream.
*   **Pixel-Perfect Botting**: If a bot creates a hardware-level USB signal that perfectly replicates a specific human's physical hand motion, Sentinel may not distinguish it.

## Assumptions

*   **Secure Transport**: We assume TLS between Client and Orchestrator.
*   **Client Integrity**: We assume the client-side JavaScript is not fully reverse-engineered to suppress all events.

## Failure Modes

*   **False Positive (Type I)**: A legitimate user is CHALLENGED.
    *   *Mitigation*: The system biases toward Challenges over Blocks. Users type a verification phrase to proceed — no CAPTCHA needed.
    *   *Note*: Sentinel intentionally biases toward false positives over false negatives during high-risk actions.
*   **False Negative (Type II)**: An attacker is ALLOWED.
    *   *Mitigation*: Defense in depth. Sentinel is one layer; all BLOCK decisions go through the Auditor's agent pipeline for secondary review.
