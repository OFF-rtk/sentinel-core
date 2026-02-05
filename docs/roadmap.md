# Roadmap & Future Work

Sentinel is currently in **v2.0 (Beta)**. This document outlines the path to v3.0 and production readiness.

## Known Limitations 🚧

*   **Mobile Support**: The current Physics Model is tuned for Desktop Mouse/Trackpad. Touch events are treated as "jumps" and often flag false positives.
*   **Session Handoff**: Cross-device sessions (switching from Laptop to Phone) are treated as two separate sessions with separate trust scores.
*   **Cold Start Latency**: New users require ~50 events before the Identity Model kicks in.

## Short-Term Improvements (Q2)

- [ ] **Touch Event Support**: Add `TouchStreamPayload` and specific gesture acceleration models.
- [ ] **WebSocket API**: Replace HTTP Polling (`POST /stream`) with a bidirectional WebSocket for lower latency challenge-response.
- [ ] **Admin Dashboard**: A UI to visualize user session plots in real-time (The "God View").

## Long-Term Research (Q3-Q4)

- [ ] **Transformer Models**: Replace Isolation Forests with a light Transformer (Attention mechanism) to understand *sequences* of actions (e.g., "User types password, then grabs mouse").
- [ ] **Federated Learning**: Train a global "Generic Human" model on client devices without ever sending raw coordinates to the server (Privacy Preserving).

## Production Hardening

Before deploying to a bank:
1.  **Rate Limiting**: Move from internal counters to Redis-backed generic rate limiters.
2.  **Secret Rotation**: Integrate with Vault/AWS Secrets Manager for key rotation.
3.  **Audit**: Third-party security audit of the `Selective Learning` logic to prove it cannot be poisoned.

## Research Disclaimer
Some roadmap items may not be suitable for production without regulatory review.
