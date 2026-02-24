# Roadmap & Future Work

Sentinel is currently in **v2.1.0**. This document outlines what's been completed and what's next.

## Completed ✅

- [x] **Redis-backed Rate Limiting**: Sliding window rate limiter using Upstash Redis (shared instance).
- [x] **Strike Escalation System**: Strikes 1–2 → 1h ban, Strike 3+ → 24h ban, 7-day strike window.
- [x] **Provisional → Confirmed Ban Pipeline**: ML sets 5-min provisional bans; Auditor confirms or pardons.
- [x] **Admin Dashboard**: SOC dashboard with 3D threat globe, live event feed, and agent investigation replay.
- [x] **Audit Logging**: Every `/evaluate` call produces a structured audit log entry in Supabase.
- [x] **GeoIP Resolution**: IP → city/country via GeoLite2-City for audit logs and dashboard.
- [x] **Bot Attack Demo**: Playwright-based bot script for reproducing detection in live environments.

## Known Limitations 🚧

*   **Mobile Support**: The Physics Model is tuned for Desktop Mouse/Trackpad. Touch events may flag false positives.
*   **Session Handoff**: Cross-device sessions are treated as separate sessions with separate trust scores.
*   **Cold Start Latency**: New users require ~50 events before the Identity Model kicks in.

## Short-Term Improvements

- [ ] **Touch Event Support**: Add `TouchStreamPayload` and gesture acceleration models.
- [ ] **WebSocket Streaming**: Replace HTTP polling (`POST /stream`) with bidirectional WebSocket.
- [ ] **Mobile Sidebar**: Hamburger toggle for narrow screens on the treasury platform.

## Long-Term Research

- [ ] **Transformer Models**: Replace Half-Space Trees with a light Transformer for sequence-aware behavior analysis.
- [ ] **Federated Learning**: Train a global "Generic Human" model on client devices without sending raw data to the server.

## Production Hardening

Before deploying to a regulated environment:
1.  **Secret Rotation**: Integrate with Vault/AWS Secrets Manager for key rotation.
2.  **Audit**: Third-party security audit of the Selective Learning logic.

## Research Disclaimer
Some roadmap items may not be suitable for production without regulatory review.
