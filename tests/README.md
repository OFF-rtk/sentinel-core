# Tests

> **I'm in the tests folder — what do I do?**

## Quick Reference

```bash
# Run all tests
pytest

# Run a specific module
pytest tests/models/

# Run a specific file
pytest tests/models/test_keyboard_model.py

# Run with output
pytest -v -s

# Stop on first failure
pytest -x
```

## Naming Convention

```
test_<source_file>.py
```

Example: `core/models/keyboard.py` → `tests/models/test_keyboard_model.py`

## Directory Structure

```
tests/
├── assets/           # Human recordings & generators
├── models/           # Model tests
├── processors/       # Processor tests
├── schemas/          # Schema validation tests
├── conftest.py       # Shared fixtures
├── test_api.py       # API endpoint tests (integration)
└── test_orchestrator.py  # Orchestrator tests (integration)
```

## Test Categories

| Marker | Description | Requires |
|--------|-------------|----------|
| `unit` | Fast, isolated tests | Nothing |
| `integration` | Full system tests | Redis + Supabase |

## Integration Test Prerequisites

```bash
# Start Redis
cd infrastructure/redis && docker-compose up -d

# Verify .env has REDIS_* and SUPABASE_* variables
```

---

📖 **For test philosophy, mapping rules, and invariants, see [docs/testing.md](../docs/testing.md)**
