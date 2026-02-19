# Sentinel-ML Known Bugs

## BUG-001: HST Model Blob Corruption (Race Condition)

**Status:** ✅ Fixed
**Severity:** High — causes permanent CHALLENGE fallback for affected users
**Discovered:** 2026-02-17
**Fixed:** 2026-02-17

### Fix Applied

Three-part fix in `persistence/model_store.py`:

1. **Per-user threading lock** (`_get_learn_lock`) — `learn_with_retry` uses a non-blocking `threading.Lock` per `(user_id, model_type)` pair. If another thread is already learning for that user, the call is skipped (next stream batch will pick it up). This prevents concurrent load-train-save cycles from racing.

2. **Base64 validation on save** — After encoding, verifies `len(encoded_blob) % 4 == 0` before writing. Aborts save if invalid (safety net).

3. **Base64 integrity check on load** — Before `base64.b64decode`, validates string length divisibility by 4. If corrupted, returns `None` with a clear error log instead of crashing.
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

---

## BUG-002: Identity Learning Catch-22 (Cold Start Deadlock)

**Status:** 🔴 Open
**Severity:** Critical — identity model is never created for any user
**Discovered:** 2026-02-18

### Symptoms

- No `keyboard_identity` rows ever appear in the `user_behavior_models` Supabase table
- Identity risk and identity confidence are always 0.0
- `_compute_identity_risk` always returns `cold_start_identity=True`

### Root Cause

**Chicken-and-egg in `_should_learn_identity` (orchestrator.py, line ~760).**

```python
def _should_learn_identity(self, session, navigator_risk, cold_start_identity, now):
    ...
    if cold_start_identity:   # ← THIS IS THE BUG
        return False
    ...
```

`cold_start_identity` is set to `True` by `_compute_identity_risk` when no stored identity model exists. Since identity learning is the only way to CREATE the model, and it's gated behind the model already existing, the model can never be created.

### Fix

Remove the `cold_start_identity` guard from `_should_learn_identity`. The other guards (trust ≥ 0.65, consecutive_allows ≥ 5, NORMAL mode, navigator_risk < 0.5, context stability) already provide sufficient safety gating.

### Files Affected

- `core/orchestrator.py` — `_should_learn_identity` method

---

## BUG-003: Keyboard HST Score Always 0.0

**Status:** 🔴 Open
**Severity:** High — keyboard biometrics provide zero risk signal
**Discovered:** 2026-02-18

### Symptoms

- Keyboard risk score is always 0.0 in evaluate responses
- HST model rows may exist in Supabase but have very low `feature_window_count`
- Risk fusion relies entirely on mouse (physics) and navigator (rules)

### Root Cause (Three Compounding Factors)

**Factor 1: HST cold start period requires 50 `learn_one()` calls.**

The `HalfSpaceTrees` detector has `window_size=50`, meaning it returns 0.0 for the first 50 training samples. Learning only happens in `_finalize_evaluate` on ALLOW decisions.

**Factor 2: Learning cap discards most training data.**

`IDENTITY_MAX_WINDOWS = 5` limits learning to the last 5 completed windows per evaluate call:
```python
windows_to_learn = keyboard_state.completed_windows[-IDENTITY_MAX_WINDOWS:]
```
If a session has 20 completed windows, 15 are discarded. This means 10+ ALLOWs are needed to exit cold start.

**Factor 3: Keyboard confidence multiplier suppresses early scores.**

Even if HST produced a small non-zero score, `_apply_keyboard_confidence` multiplies it by `√(conf_time × conf_count)` which is 0.0 early in a session, crushing any score to zero.

### Fix

1. Remove or increase `IDENTITY_MAX_WINDOWS` cap for HST learning — learn all available windows
2. Force one CHALLENGE per action during HST cold start — generates training data from typed challenge text
3. After the forced CHALLENGE, ALLOW the action — HST learns from completed windows
4. Filter out high-anomaly windows (score > 95th percentile) during learning to prevent training on suspicious data

### Files Affected

- `core/orchestrator.py` — learning logic, cold start CHALLENGE, window filtering
