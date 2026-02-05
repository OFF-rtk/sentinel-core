# Storage & State Management

Sentinel separates "Hot State" (Session) from "Cold State" (Long-term Profile).

## 1. Hot State: Redis ⚡

Redis is used for all high-frequency, low-latency operations during an active session.

### What's stored?
*   **Session Metadata**: `session_id` -> `user_id`, `start_time`
*   **Current Trust Score**: Floating point value (0.0 - 1.0).
*   **Temporal Windows**: Recent event lists for calculating velocity/jitter.
*   **Batch Counters**: Last processed `batch_id` (for replay protection).

### Keyspace Design
*   `session:{uuid}:state`: Hash map of session variables.
*   `session:{uuid}:expiry`: TTL key (defaults to 30 minutes idle).
*   *Note*: Redis keys are versioned to allow forward-compatible state migrations.

### Persistence Guarantee
Redis is configured with **AOF (Append Only File)** every second. In a crash, we lose at most 1 second of behavioral data, which is acceptable for a biometric system (users just generate more data).

## 2. Cold State: Supabase / Postgres 🧊

Supabase is the source of truth for long-term identity models.

### What's stored?
*   **User Profiles**: `user_id` -> encrypted model binaries (pickled River objects).
*   **Audit Logs**: Every `BLOCK` decision and every `evaluate` call is logged for compliance.

### The Storage Loop
1.  **Load**: On first request of a session, Sentinel fetches the user's Model Binary from Supabase.
2.  **Cache**: It deserializes the model and stores it in Redis (or local LRU cache if using sticky routing).
3.  **Update**: As the session progresses, the model updates in memory.
4.  **Save**: On session termination, the updated model is serialized and written back to Supabase.

## Stateless API Design

The API nodes themselves hold **no state**.
*   Request 1 can go to Server A.
*   Request 2 can go to Server B.
*   Both servers read the same `session:{uuid}` from the shared Redis cluster.

This eliminates the need for "Sticky Sessions" at the load balancer level.
