"""
API Endpoint Tests

Tests for FastAPI endpoints using TestClient with real persistence backends.
Requires:
- Redis running (docker-compose up in infrastructure/redis)
- Supabase connection

All test users and sessions use unique timestamps for isolation.
"""

import os
import time
import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

# Load environment variables
load_dotenv()

from main import app


def unique_id(prefix: str) -> str:
    """Generate unique ID with timestamp for test isolation."""
    return f"test_{prefix}_{int(time.time() * 1000)}"


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def client():
    """TestClient for FastAPI app with lifespan context."""
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        
        print(f"\n✅ Health check passed: {data}")


# =============================================================================
# Keyboard Stream Tests
# =============================================================================

class TestKeyboardStreamEndpoint:
    """Test keyboard stream endpoint."""
    
    def test_valid_keyboard_stream_returns_204(self, client):
        """Valid keyboard stream should return 204 No Content."""
        now = time.time() * 1000
        session_id = unique_id("kb_valid")
        payload = {
            "session_id": session_id,
            "user_id": unique_id("user_kb"),
            "batch_id": 1,
            "events": [
                {"key": "a", "event_type": "DOWN", "timestamp": now},
                {"key": "a", "event_type": "UP", "timestamp": now + 100},
                {"key": "b", "event_type": "DOWN", "timestamp": now + 150},
                {"key": "b", "event_type": "UP", "timestamp": now + 250},
            ]
        }
        
        response = client.post("/stream/keyboard", json=payload)
        assert response.status_code == 204
        print(f"\n✅ Keyboard stream accepted")
    
    def test_invalid_batch_id_returns_422(self, client):
        """Invalid batch_id (< 1) should return 422."""
        now = time.time() * 1000
        payload = {
            "session_id": unique_id("kb_invalid"),
            "user_id": unique_id("user"),
            "batch_id": 0,  # Invalid: must be >= 1
            "events": [
                {"key": "a", "event_type": "DOWN", "timestamp": now},
            ]
        }
        
        response = client.post("/stream/keyboard", json=payload)
        assert response.status_code == 422
        print(f"\n✅ Invalid batch_id correctly rejected")
    
    def test_missing_required_field_returns_422(self, client):
        """Missing required field should return 422."""
        payload = {
            "session_id": unique_id("kb_missing"),
            # Missing user_id
            "batch_id": 1,
            "events": []
        }
        
        response = client.post("/stream/keyboard", json=payload)
        assert response.status_code == 422
        print(f"\n✅ Missing field correctly rejected")
    
    def test_invalid_event_type_returns_422(self, client):
        """Invalid event_type should return 422."""
        now = time.time() * 1000
        payload = {
            "session_id": unique_id("kb_invalid_type"),
            "user_id": unique_id("user"),
            "batch_id": 1,
            "events": [
                {"key": "a", "event_type": "INVALID", "timestamp": now},
            ]
        }
        
        response = client.post("/stream/keyboard", json=payload)
        assert response.status_code == 422
        print(f"\n✅ Invalid event_type correctly rejected")


# =============================================================================
# Mouse Stream Tests
# =============================================================================

class TestMouseStreamEndpoint:
    """Test mouse stream endpoint."""
    
    def test_valid_mouse_stream_returns_204(self, client):
        """Valid mouse stream should return 204 No Content."""
        now = time.time() * 1000
        payload = {
            "session_id": unique_id("mouse_valid"),
            "user_id": unique_id("user_mouse"),
            "batch_id": 1,
            "events": [
                {"x": 100, "y": 100, "event_type": "MOVE", "timestamp": now},
                {"x": 110, "y": 105, "event_type": "MOVE", "timestamp": now + 10},
                {"x": 120, "y": 110, "event_type": "MOVE", "timestamp": now + 20},
                {"x": 130, "y": 115, "event_type": "CLICK", "timestamp": now + 30},
            ]
        }
        
        response = client.post("/stream/mouse", json=payload)
        assert response.status_code == 204
        print(f"\n✅ Mouse stream accepted")
    
    def test_invalid_event_type_returns_422(self, client):
        """Invalid mouse event_type should return 422."""
        now = time.time() * 1000
        payload = {
            "session_id": unique_id("mouse_invalid"),
            "user_id": unique_id("user"),
            "batch_id": 1,
            "events": [
                {"x": 100, "y": 100, "event_type": "SCROLL", "timestamp": now},
            ]
        }
        
        response = client.post("/stream/mouse", json=payload)
        assert response.status_code == 422
        print(f"\n✅ Invalid mouse event_type correctly rejected")


# =============================================================================
# Evaluate Endpoint Tests
# =============================================================================

class TestEvaluateEndpoint:
    """Test evaluate endpoint."""
    
    def test_valid_evaluate_returns_json(self, client):
        """Valid evaluate request should return JSON response."""
        now = time.time() * 1000
        session_id = unique_id("eval_valid")
        user_id = unique_id("user_eval")
        
        payload = {
            "session_id": session_id,
            "request_context": {
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0",
                "endpoint": "/api/transfer",
                "method": "POST",
                "user_id": user_id
            },
            "business_context": {
                "service": "banking",
                "action_type": "transfer",
                "resource_target": "account_123"
            },
            "role": "analyst",
            "mfa_status": "verified",
            "session_start_time": now
        }
        
        response = client.post("/evaluate", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "decision" in data
        assert "risk" in data
        assert data["decision"] in ["ALLOW", "CHALLENGE", "BLOCK"]
        assert 0.0 <= data["risk"] <= 1.0
        
        print(f"\n✅ Evaluate returned: decision={data['decision']}, risk={data['risk']:.3f}")
    
    def test_evaluate_with_fingerprint(self, client):
        """Evaluate with client fingerprint should work."""
        now = time.time() * 1000
        session_id = unique_id("eval_fp")
        user_id = unique_id("user_fp")
        
        payload = {
            "session_id": session_id,
            "request_context": {
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0",
                "endpoint": "/api/transfer",
                "method": "POST",
                "user_id": user_id
            },
            "business_context": {
                "service": "banking",
                "action_type": "transfer",
                "resource_target": "account_123"
            },
            "role": "analyst",
            "mfa_status": "verified",
            "session_start_time": now,
            "client_fingerprint": {
                "device_id": unique_id("device"),
                "ja3_hash": "abc123"
            }
        }
        
        response = client.post("/evaluate", json=payload)
        assert response.status_code == 200
        print(f"\n✅ Evaluate with fingerprint accepted")
    
    def test_evaluate_with_eval_id(self, client):
        """Evaluate with eval_id for idempotency should work."""
        now = time.time() * 1000
        session_id = unique_id("eval_idem")
        user_id = unique_id("user_idem")
        eval_id = unique_id("eval")
        
        payload = {
            "session_id": session_id,
            "request_context": {
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0",
                "endpoint": "/api/transfer",
                "method": "POST",
                "user_id": user_id
            },
            "business_context": {
                "service": "banking",
                "action_type": "transfer",
                "resource_target": "account_123"
            },
            "role": "analyst",
            "mfa_status": "verified",
            "session_start_time": now,
            "eval_id": eval_id
        }
        
        # First request
        response1 = client.post("/evaluate", json=payload)
        assert response1.status_code == 200
        data1 = response1.json()
        
        # Second request with same eval_id (should be idempotent)
        response2 = client.post("/evaluate", json=payload)
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Results should match
        assert data1["decision"] == data2["decision"]
        assert data1["risk"] == data2["risk"]
        
        print(f"\n✅ Idempotency verified: both calls returned {data1['decision']}")
    
    def test_missing_required_field_returns_422(self, client):
        """Missing required field in evaluate should return 422."""
        payload = {
            "session_id": unique_id("eval_missing"),
            # Missing request_context
            "business_context": {
                "service": "banking",
                "action_type": "transfer",
                "resource_target": "account_123"
            },
            "role": "analyst",
            "mfa_status": "verified",
            "session_start_time": 1000000
        }
        
        response = client.post("/evaluate", json=payload)
        assert response.status_code == 422
        print(f"\n✅ Missing request_context correctly rejected")


# =============================================================================
# End-to-End Flow Tests
# =============================================================================

class TestEndToEndFlow:
    """Test complete API flow: stream → evaluate."""
    
    def test_keyboard_then_evaluate(self, client):
        """Stream keyboard data, then evaluate."""
        session_id = unique_id("e2e_kb")
        user_id = unique_id("user_e2e_kb")
        now = time.time() * 1000
        
        # Stream keyboard events
        keyboard_payload = {
            "session_id": session_id,
            "user_id": user_id,
            "batch_id": 1,
            "events": [
                {"key": "h", "event_type": "DOWN", "timestamp": now},
                {"key": "h", "event_type": "UP", "timestamp": now + 100},
                {"key": "e", "event_type": "DOWN", "timestamp": now + 150},
                {"key": "e", "event_type": "UP", "timestamp": now + 250},
                {"key": "l", "event_type": "DOWN", "timestamp": now + 300},
                {"key": "l", "event_type": "UP", "timestamp": now + 400},
            ]
        }
        
        response = client.post("/stream/keyboard", json=keyboard_payload)
        assert response.status_code == 204
        
        # Now evaluate
        eval_payload = {
            "session_id": session_id,
            "request_context": {
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0",
                "endpoint": "/api/action",
                "method": "POST",
                "user_id": user_id
            },
            "business_context": {
                "service": "test_service",
                "action_type": "test_action",
                "resource_target": "test_resource"
            },
            "role": "user",
            "mfa_status": "verified",
            "session_start_time": now
        }
        
        response = client.post("/evaluate", json=eval_payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["decision"] in ["ALLOW", "CHALLENGE", "BLOCK"]
        
        print(f"\n✅ E2E keyboard flow: decision={data['decision']}, risk={data['risk']:.3f}")
    
    def test_mouse_then_evaluate(self, client):
        """Stream mouse data, then evaluate."""
        session_id = unique_id("e2e_mouse")
        user_id = unique_id("user_e2e_mouse")
        now = time.time() * 1000
        
        # Stream mouse events
        mouse_payload = {
            "session_id": session_id,
            "user_id": user_id,
            "batch_id": 1,
            "events": [
                {"x": 100, "y": 100, "event_type": "MOVE", "timestamp": now},
                {"x": 120, "y": 110, "event_type": "MOVE", "timestamp": now + 20},
                {"x": 140, "y": 120, "event_type": "MOVE", "timestamp": now + 40},
                {"x": 160, "y": 130, "event_type": "MOVE", "timestamp": now + 60},
                {"x": 180, "y": 140, "event_type": "CLICK", "timestamp": now + 80},
            ]
        }
        
        response = client.post("/stream/mouse", json=mouse_payload)
        assert response.status_code == 204
        
        # Now evaluate
        eval_payload = {
            "session_id": session_id,
            "request_context": {
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0",
                "endpoint": "/api/action",
                "method": "POST",
                "user_id": user_id
            },
            "business_context": {
                "service": "test_service",
                "action_type": "test_action",
                "resource_target": "test_resource"
            },
            "role": "user",
            "mfa_status": "verified",
            "session_start_time": now
        }
        
        response = client.post("/evaluate", json=eval_payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["decision"] in ["ALLOW", "CHALLENGE", "BLOCK"]
        
        print(f"\n✅ E2E mouse flow: decision={data['decision']}, risk={data['risk']:.3f}")
