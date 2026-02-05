# Sentinel API Reference

**Base URL**: `http://<sentinel-host>:8000`
**Version**: 2.0.0

## 1. Biometric Streaming

Used for high-volume, asynchronous ingestion of behavioral data.

### 1.1 Keyboard Stream
`POST /stream/keyboard`

**Payload**:
```json
{
  "session_id": "uuid",
  "user_id": "string",
  "batch_id": 1,
  "events": [
    { "key": "a", "event_type": "DOWN", "timestamp": 167888.12 }
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
    { "x": 100, "y": 200, "event_type": "MOVE", "timestamp": 167899.22 }
  ]
}
```

**Responses**:
*   `204 No Content`: BATCH ACCEPTED. (Standard success).
*   `400 Bad Request`: Invalid schema or non-sequential batch ID.

> **Note**: Streaming endpoints never block user actions and return no decision.

---

## 2. Risk Evaluation

Used synchronously at decision points (Login, Checkout).

### 2.1 Evaluate Risk
`POST /evaluate`

**Payload**:
```json
{
  "session_id": "uuid",
  "user_id": "string",
  "context": { "action": "login", "ip": "1.2.3.4" }
}
```

**Response**:
```json
{
  "decision": "ALLOW | CHALLENGE | BLOCK",
  "risk": 0.15,
  "mode": "NORMAL"
}
```

## Error Handling

*   **401 Unauthorized**: Missing API key (if configured).
*   **429 Too Many Requests**: Throttling active (defaults to 100 req/sec per IP). Rate limits exist to prevent replay amplification attacks.
*   **503 Service Unavailable**: Redis connection lost.

For full OpenAPI specifications, visit `/docs` on the running API instance.
