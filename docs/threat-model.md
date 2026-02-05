# Threat Model

This document outlines the specific threats Sentinel is designed to detect, as well as its limitations and non-goals.

## Threat Categories

### 1. Automated Bots (The "Script Kiddie")
*   **Attack**: Simple scripts using Selenium or PyAutoGUI to perform actions.
*   **Characteristics**: Perfect straight-line mouse movements, instant keystrokes (0ms dwell time), fixed timing intervals.
*   **Detection**: `PhysicsModel`. These attacks fail hard constraints on human biomechanics.

### 2. Replay Attacks
*   **Attack**: Recording a real human's session and replaying it later to bypass behavioral checks.
*   **Characteristics**: Valid "human" data, but identical to a previous session.
*   **Detection**:
    *   **Batch Sequencing**: The API enforces strict sequential `batch_id`s.
    *   **Jitter Analysis**: Exact repetition of floating-point timestamps is statistically impossible in nature.

### 3. Mimicry / Generative Bots
*   **Attack**: Advanced AI (e.g., generative adversarial networks) treating mouse generation as a pathfinding problem to "look human."
*   **Characteristics**: Smooth curves, variable timing.
*   **Detection**: `IdentityModel`. While they look "human," they do not look like *the specific user*. They lack the user's unique motor cortex quirks.

### 4. Model Poisoning
*   **Attack**: An attacker slowly changes their behavior over weeks to "train" the model to accept malicious patterns.
*   **Detection**: `Selective Learning`. Sentinel only learns from sessions that end in a "High Trust" state. If an attacker acts weirdly to shift the model, the session trust drops, and learning is disabled for that session.

## Non-Goals

*   **Malware Detection**: Sentinel cannot detect if the user's machine is infected with valid malware that is not interacting with the input stream.
*   **Pixel-Perfect Botting**: If a bot creates a hardware-level USB signal that perfectly replicates a specific human's physical hand motion (recorded from hardware), Sentinel may not distinguish it.

## Assumptions

*   **Secure Transport**: We assume TLS between Client and Orchestrator.
*   **Client Integrity**: We assume the client-side JavaScript is not fully reverse-engineered to suppress all events. (Though heartbeat checks help here).

## Failure Modes

*   **False Positive (Type I)**: A legitimate user is CHALLENGED.
    *   *Mitigation*: The system is tuned to prefer Challenges over Blocks. A user can solve a CAPTCHA to proceed.
    *   *Note*: Sentinel intentionally biases toward false positives over false negatives during high-risk actions.
*   **False Negative (Type II)**: An attacker is ALLOWED.
    *   *Mitigation*: Defense in depth. Sentinel is one layer; meaningful transactions should still require 2FA if risk is marginal.
