# Sentinel — Real-Time Behavioral Security Engine

Sentinel is a **real-time behavioral security engine** that detects bots and account takeovers by continuously verifying *how* a user interacts—not just which credentials they present.

Unlike CAPTCHAs and one-shot risk checks, Sentinel performs **continuous authentication** using a hybrid decision engine that combines deterministic physics with online machine learning. Trust is earned gradually, adapts over time, and makes account takeover increasingly expensive—without harming legitimate users.

---

## What Sentinel Is
- Continuous behavioral authentication engine
- Bot and account takeover (ATO) detection without CAPTCHAs
- Trust-based, time-aware risk orchestration

## What Sentinel Is Not
- A replacement for passwords or MFA
- Cross-device user tracking
- Surveillance or biometric identification

---

## 🏗 System Architecture

**Pattern:** Stateless API + Stateful Storage  
**Goal:** Horizontal scalability without sticky sessions

Sentinel scales horizontally without sticky sessions while preserving strict consistency guarantees using Redis transactions.

### Core Components
- **Client (Next.js):** Captures high-frequency telemetry (mouse, keyboard) via non-blocking beacons
- **Orchestrator (FastAPI):** Ingests streams, fuses signals, computes dynamic risk
- **State Store (Redis):** Ephemeral session context (<5ms latency)
- **Auditor (RAG Agent):** Reviews flagged sessions against security policy

> Redis transactions preserve consistency across stateless workers.

---

## 📁 Directory Structure

```text
sentinel-ml/
├── core/               # Detection & decision logic
│   ├── models/         # Physics + online ML models
│   ├── processors/     # Feature extraction
│   └── orchestrator.py # Risk fusion
├── persistence/        # State & model storage
├── infrastructure/     # Redis + Docker config
├── main.py             # FastAPI entrypoint
└── tests/              # Unit & integration tests
```

## 🛡 Threat Model

**Mitigates**
- Automated bots (mouse teleportation, scripted input)
- Replay attacks (batch monotonicity)
- Session hijacking (identity continuity mismatch)
- Low-and-slow mimicry attacks
- Model poisoning during active attacks

**Explicitly Out of Scope**
- Cross-device identification
- MFA or password replacement
- Surveillance or biometric tracking

📘 Full threat analysis: [docs/threat-model.md](docs/threat-model.md)

## 🧠 Decision Engine (High-Level)
Sentinel uses a time-variant risk orchestration model.
Users are evaluated differently at second 1 vs minute 10.

**Key ideas**
- Zero-trust cold start
- Liveness before behavior
- Gradual trust stabilization
- Identity continuity checks
- Weighted MAX risk fusion

📘 Full lifecycle & flowcharts: [docs/decision-engine.md](docs/decision-engine.md)

## 🛠 Tech Stack
- **Language**: Python 3.11
- **Web**: FastAPI (async)
- **ML**: River (online learning), NumPy (physics)
- **Storage**: Redis (hot state), Supabase (long-term)
- **DevOps**: Docker, GitHub Actions

## 🚀 Quick Start
⚠️ *Sentinel is a research-grade prototype, not a production authentication replacement.*

### Prerequisites
- Docker & Docker Compose
- Python 3.11+

### Environment
Create a `.env` file in the project root:

```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=secret
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
```

### Launch Infrastructure
```bash
cd infrastructure/redis
docker-compose up -d
cd ../..
```

### Start the Engine
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
API available at: `http://localhost:8000`

## 📡 API Overview
- `/stream/*` — High-frequency async telemetry ingestion
- `/evaluate` — Synchronous risk decision for sensitive actions

📘 Full API reference: [docs/api.md](docs/api.md)

## 🧪 Testing
We use pytest for unit and integration testing.

```bash
pytest
```
📘 Testing strategy: [docs/testing.md](docs/testing.md)

## 📚 Documentation
For comprehensive documentation, including deep dives into the architecture, decision engine, and API contracts, please refer to the **[Documentation Hub](docs/README.md)**.

## 📜 License
Distributed under the MIT License. See LICENSE for details.
