# Sentinel API Reference

**Base URL**: `http://<sentinel-host>:8000`
**Version**: 2.1.0

## 1. Biometric Streaming

Used for high-volume, asynchronous ingestion of behavioral data. These endpoints buffer data for the Orchestrator and never return a decision.

### 1.1 Keyboard Stream
`POST /stream/keyboard`

**Payload**:
```json
{
  "session_id": "uuid",
  "user_id": "string",
  "batch_id": 1,
  "events": [
    { "key": "a", "event_type": "DOWN", "timestamp": 167888.12 },
    { "key": "a", "event_type": "UP", "timestamp": 167938.12 }
  ]
}
```

### 1.2 Mouse Stream
`POST /stream/mouse`

**Payload**:
```json
{
  "session_id": "uuid",
  "user_id": "string",
  "batch_id": 1,
  "events": [
    { "x": 100, "y": 200, "event_type": "MOVE", "timestamp": 167899.22 },
    { "x": 150, "y": 250, "event_type": "CLICK", "timestamp": 167999.22 }
  ]
}
```

**Responses**:
*   `204 No Content`: Batch accepted.
*   `400 Bad Request`: Invalid schema or non-sequential `batch_id`.

> **Note**: `batch_id` must be strictly sequential per session for anti-replay protection.

---

## 2. Risk Evaluation

Used synchronously at decision points (login, payment approval, admin actions).

### 2.1 Evaluate Risk
`POST /evaluate`

**Payload**:
```json
{
  "session_id": "uuid",
  "request_context": {
    "ip_address": "203.0.113.45",
    "user_agent": "Mozilla/5.0 ...",
    "endpoint": "/api/payments/approve",
    "method": "POST",
    "user_id": "user-uuid"
  },
  "business_context": {
    "service": "vault_treasury",
    "action_type": "approve_payment",
    "resource_target": "payment-uuid",
    "transaction_details": { "amount": 5000, "currency": "USD" }
  },
  "role": "treasury_admin",
  "mfa_status": "verified",
  "session_start_time": 1708500000000,
  "client_fingerprint": { "device_id": "fp-uuid" },
  "eval_id": "eval-uuid"
}
```

**Response**:
```json
{
  "decision": "ALLOW | CHALLENGE | BLOCK",
  "risk": 0.15,
  "mode": "NORMAL | CHALLENGE",
  "anomaly_vectors": ["keystroke_anomaly_0.85_confidence_0.40"],
  "ban_expires_in_seconds": 0
}
```

**Side Effects**:
- On `BLOCK`: Sets a provisional 5-minute ban in Redis (`blacklist:{user_id}`).
- On every call: Writes a structured audit log to Supabase `audit_logs` table (triggers auditor webhook).

## Error Handling

*   **401 Unauthorized**: Missing API key (if configured).
*   **429 Too Many Requests**: Throttling active.
*   **503 Service Unavailable**: Redis connection lost — fails safe to CHALLENGE.

For full OpenAPI specifications, visit `/docs` on the running API instance.
