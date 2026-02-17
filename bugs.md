# Sentinel-ML Known Bugs

## BUG-001: HST Model Blob Corruption (Race Condition)

**Status:** Open
**Severity:** High — causes permanent CHALLENGE fallback for affected users
**Discovered:** 2026-02-17

### Symptoms

- `/evaluate` always returns `risk=0.5`, `decision=CHALLENGE` for a specific user
- Sentinel-ml logs repeat on every request:
  ```
  Failed to load keyboard_hst for <user_id>: Invalid base64-encoded string:
  number of data characters (945001) cannot be 1 more than a multiple of 4
  ```
- The orchestrator can't load the HST model → falls back to default 0.5 risk

### Root Cause

**Race condition in `persistence/model_store.py` during concurrent `learn_with_retry` calls.**

The `/stream/keyboard` endpoint fires every ~2-3 seconds. Each call triggers `learn_with_retry()` for the `keyboard_hst` model, which does:

1. `load_model()` → reads `model_blob` from Supabase
2. `learn_fn(model)` → trains on new window
3. `save_model()` → upserts the updated blob back

When two stream batches arrive nearly simultaneously:

```
Request A: load(v1) → train → save(v2)   ← succeeds
Request B: load(v1) → train → save(v2)   ← version conflict, retries
         → retry: load(v2)               ← catches partial write of v2's blob
         → base64 decode fails (blob is 1 char off)
         → save corrupted blob as v3      ← CORRUPTED PERMANENTLY
```

The `model_blob` column stores ~945KB of base64 text. A concurrent read during a large column write can return a partially-flushed value. The resulting blob has a character count not divisible by 4, making it invalid base64.

### Immediate Fix (Manual)

Delete the corrupted row from Supabase:
```sql
DELETE FROM user_behavior_models
WHERE user_id = '<affected_user_id>'
  AND model_type = 'keyboard_hst';
```

The system rebuilds the model from scratch on the next session.

### Permanent Fix (Code Change)

Apply these changes to `persistence/model_store.py`:

#### 1. Validate blob on save (prevent writing bad data)

In `save_model()`, after `base64.b64encode(blob).decode("utf-8")`, add:

```python
encoded = base64.b64encode(blob).decode("utf-8")

# Sanity check: base64 must be divisible by 4
assert len(encoded) % 4 == 0, f"Invalid base64 length: {len(encoded)}"

record = {
    ...
    "model_blob": encoded,
    ...
}
```

#### 2. Validate blob on load (recover from corruption instead of permanent failure)

In `load_model()`, before `base64.b64decode(blob)`, add:

```python
if isinstance(blob, str):
    if len(blob) % 4 != 0:
        logger.error(
            f"Corrupted blob for {user_id}/{model_type.value}: "
            f"length {len(blob)} not divisible by 4. Deleting row."
        )
        # Auto-heal: delete corrupted row so it gets rebuilt
        self.client.table(self.TABLE_NAME).delete().eq(
            "user_id", user_id
        ).eq("model_type", model_type.value).execute()
        return None
    blob = base64.b64decode(blob)
```

#### 3. Add a debounce/lock to prevent concurrent HST learning

In the orchestrator or stream handler, ensure only one `learn_with_retry` for `keyboard_hst` runs per user at a time. Options:

- **Redis lock:** `SET hst_lock:{user_id} 1 EX 5 NX` before learning
- **Debounce:** Only persist HST every N windows instead of every stream batch
- **Queue:** Buffer learn calls and process sequentially per user

### Files Affected

- `persistence/model_store.py` — save/load validation + auto-heal
- `core/orchestrator.py` — debounce HST persistence
