# Storage & State Management

Sentinel separates "Hot State" (Session) from "Cold State" (Long-term Profile). Both Sentinel ML and the Sentinel Auditor share a single **Upstash Redis** instance for ban enforcement.

## 1. Hot State: Redis (Upstash) ⚡

Redis is used for all high-frequency, low-latency operations during an active session.

### What's stored?
*   **Session Metadata**: `session_id` -> `user_id`, `start_time`
*   **Current Trust Score**: Floating point value (0.0 - 1.0).
*   **Temporal Windows**: Recent event lists for calculating velocity/jitter.
*   **Batch Counters**: Last processed `batch_id` (for replay protection).
*   **Blacklist State**: `blacklist:{user_id}` — ban reason string with TTL-based expiry.
*   **Strike Counters**: `global_strikes:{user_id}` — integer counter with 7-day TTL.

### Keyspace Design

| Key Pattern | Owner | TTL | Purpose |
|-------------|-------|-----|---------|
| `session:{uuid}:state` | Sentinel ML | 30 min idle | Session state hash |
| `blacklist:{user_id}` | ML (provisional) / Auditor (confirmed) | 5 min / 1h / 24h | Ban enforcement |
| `global_strikes:{user_id}` | Auditor | 7 days | Strike escalation counter |
| `auditor:rate_limit:{user_id}` | Auditor | 60s | Sliding window rate limit |

### Ban Lifecycle
1. **Provisional ban** (Sentinel ML): On a BLOCK decision, ML sets `blacklist:{user_id}` with a 5-minute TTL and value `provisional_ban|{reason}`.
2. **Audit log fires**: The evaluate result is written to Supabase `audit_logs`, which triggers a webhook to the Auditor.
3. **Confirmed ban** (Auditor Enforcer): If the agent pipeline confirms BLOCK, the Enforcer calls `SETEX blacklist:{user_id}` with a 1-hour or 24-hour TTL (based on strike count), *overwriting* the provisional key.
4. **Pardon** (Auditor Enforcer): If the agent pipeline overrides to ALLOW, the Enforcer calls `DEL blacklist:{user_id}` to immediately unban.

### Persistence
Upstash Redis is a managed serverless instance with built-in durability. No local AOF/RDB configuration required.

## 2. Cold State: Supabase / Postgres 🧊

Supabase is the source of truth for long-term identity models and audit logs.

### What's stored?
*   **User Profiles**: `user_id` -> encrypted model binaries (pickled River objects) in `user_models` table.
*   **Audit Logs**: Every `/evaluate` call is logged (structured JSON payload) for compliance and auditor consumption.

### The Storage Loop
1.  **Load**: On first request of a session, Sentinel fetches the user's Model Binary from Supabase.
2.  **Cache**: It deserializes the model and stores it in the Redis session state.
3.  **Update**: As the session progresses, the model updates in memory.
4.  **Save**: On session termination (or periodically), the updated model is serialized and written back to Supabase.

## Stateless API Design

The API nodes themselves hold **no state**.
*   Request 1 can go to Server A.
*   Request 2 can go to Server B.
*   Both servers read the same `session:{uuid}` from the shared Redis instance.

This eliminates the need for "Sticky Sessions" at the load balancer level.
