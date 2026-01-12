# Sentinel Integration Test Results

**Date:** 2026-01-12 13:10:15.907060+00:00

## Phase 1: Model Warm-up (Training)

**Strategy:** Sliding Window Ingestion (Window=50, Step=5).
**Target:** > 250 updates to fill HalfSpaceTrees window.

**Generated Pool:** 8120 Keyboard Events, 2500 Mouse Events.

### Score Progression (Keyboard)



| Update # | Risk Score | Status |
|----------|------------|--------|
| 50 | 0.0000 | ⏳ Filling Window |
| 100 | 0.0000 | ⏳ Filling Window |
| 150 | 0.0000 | ⏳ Filling Window |
| 200 | 0.0000 | ⏳ Filling Window |
| 250 | 0.0000 | ✅ Stable |
| 300 | 0.0000 | ✅ Stable |
| 350 | 0.0000 | ✅ Stable |
| 400 | 0.0000 | ✅ Stable |
| 450 | 0.0000 | ✅ Stable |
| 500 | 0.0000 | ✅ Stable |
| 550 | 0.0000 | ✅ Stable |
| 600 | 0.0000 | ✅ Stable |
| 650 | 0.0000 | ✅ Stable |
| 700 | 0.0000 | ✅ Stable |
| 750 | 0.0000 | ✅ Stable |
| 800 | 0.0000 | ✅ Stable |
| 850 | 0.0000 | ✅ Stable |
| 900 | 0.0000 | ✅ Stable |
| 950 | 0.0000 | ✅ Stable |
| 1000 | 0.0000 | ✅ Stable |
| 1050 | 0.0000 | ✅ Stable |
| 1100 | 0.0000 | ✅ Stable |
| 1150 | 0.0000 | ✅ Stable |
| 1200 | 0.0000 | ✅ Stable |
| 1250 | 0.0000 | ✅ Stable |
| 1300 | 0.0000 | ✅ Stable |
| 1350 | 0.0000 | ✅ Stable |
| 1400 | 0.0000 | ✅ Stable |
| 1450 | 0.0000 | ✅ Stable |
| 1500 | 0.0000 | ✅ Stable |
| 1550 | 0.0000 | ✅ Stable |
| 1600 | 0.0000 | ✅ Stable |
| 1614 | 0.0000 | ✅ Stable |

### Warm-up Result

**Final Keyboard Score:** `0.0000`

**Total Updates:** 1614

**Status:** ✅ Ready

---
## Phase 2: Attack Simulation

### Biometric Scores (Post-Attack)

```json
{
  "keyboard_score": 0.0,
  "keyboard_vectors": [],
  "mouse_score": 0.0,
  "mouse_vectors": []
}
```

### Sentinel Analysis (Attack)

```json
{
  "risk_score": 1.0,
  "decision": "BLOCK",
  "anomaly_vectors": [
    "impossible_travel"
  ]
}
```

---
## Phase 3: Decay Test

### Post-Decay Analysis

```json
{
  "pre_decay_score": 1.0,
  "post_decay_score": 0.0,
  "decision": "ALLOW"
}
```

