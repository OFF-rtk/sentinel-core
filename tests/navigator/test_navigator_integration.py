"""
Sentinel Navigator Integration Test Suite

Comprehensive tests for the Persistence Layer, Context Processor, and Policy Engine.
Tests mimic real client behavior and verify end-to-end flows.

Test Scenarios:
    A. Infrastructure Health - Redis and GeoIP connectivity
    B. Happy Path - New user baseline behavior
    C. Impossible Travel - Superman attack detection
    D. Bot Detection - Infrastructure mismatch
    E. Session Persistence - Device memory verification

Usage:
    pytest test_navigator_integration.py -v -s
    pytest test_navigator_integration.py -v -s -k "impossible_travel"
"""

import os
import time
import pytest
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from unittest.mock import patch, MagicMock

from core.schemas.inputs import (
    EvaluationRequest,
    UserSessionContext,
    ClientNetworkContext,
    BusinessContext,
    ClientFingerprint,
)
from core.schemas.outputs import SentinelDecision
from core.processors.context import NavigatorContextProcessor
from core.models.navigator import NavigatorPolicyEngine, MouseSessionTracker
from persistence.repository import SentinelStateRepository


# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def log_header(test_name: str):
    """Print a formatted test header."""
    print(f"\n{'='*70}")
    print(f"🧪 TEST: {test_name}")
    print(f"{'='*70}")


def log_section(title: str):
    """Print a section divider."""
    print(f"\n{'─'*50}")
    print(f"📋 {title}")
    print(f"{'─'*50}")


def log_metric(name: str, value, expected=None, check=None):
    """Log a metric with optional expected value and check result."""
    status = ""
    if check is not None:
        status = "✅" if check else "❌"
    if expected is not None:
        print(f"  {status} {name}: {value} (expected: {expected})")
    else:
        print(f"  {status} {name}: {value}")


def log_result(passed: bool, message: str):
    """Log the test result."""
    emoji = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n📊 Result: {emoji} - {message}")


# =============================================================================
# Test Data Helper
# =============================================================================

def generate_request(
    user_id: str = "test_user_1",
    ip_address: str = "8.8.8.8",
    device_id: str = "device_abc123",
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    session_id: Optional[str] = None,
    role: str = "analyst",
    resource_target: str = "card_xxxx1234",
    mfa_status: str = "verified",
) -> EvaluationRequest:
    """
    Rapidly build Pydantic EvaluationRequest objects with sensible defaults.
    """
    if session_id is None:
        session_id = f"{user_id}_session"
    
    return EvaluationRequest(
        user_session=UserSessionContext(
            user_id=user_id,
            session_id=session_id,
            role=role,
            session_start_time=datetime.now(timezone.utc),
            mfa_status=mfa_status,
        ),
        business_context=BusinessContext(
            service="card_service",
            action_type="card_activation",
            resource_target=resource_target,
            transaction_details={"amount": 100.0, "currency": "USD"},
        ),
        network_context=ClientNetworkContext(
            ip_address=ip_address,
            user_agent=user_agent,
            client_fingerprint=ClientFingerprint(device_id=device_id),
        ),
    )


# =============================================================================
# Scenario A: Infrastructure Health Tests
# =============================================================================

class TestInfrastructureHealth:
    """Tests for Redis and GeoIP connectivity."""
    
    @pytest.mark.integration
    def test_redis_connection(self, redis_client):
        """Verify Docker Redis container is reachable."""
        log_header("Redis Connection Test")
        
        log_section("Testing Redis Ping")
        result = redis_client.ping()
        log_metric("ping()", result, expected=True, check=result is True)
        assert result is True, "Redis ping should return True"
        
        log_section("Testing Read/Write")
        redis_client.set("test_key", "test_value")
        value = redis_client.get("test_key")
        log_metric("SET test_key", "test_value")
        log_metric("GET test_key", value, expected="test_value", check=value == "test_value")
        assert value == "test_value", "Should be able to read/write to Redis"
        
        log_result(True, "Redis connection is healthy")
    
    @pytest.mark.integration
    def test_geoip_loading(self, geoip_available):
        """Verify the Context Processor can load the GeoIP database."""
        log_header("GeoIP Database Loading Test")
        
        log_section("Checking GeoIP Database File")
        log_metric("assets/GeoLite2-City.mmdb exists", geoip_available)
        
        if not geoip_available:
            pytest.skip("GeoIP database not available at assets/GeoLite2-City.mmdb")
        
        log_section("Initializing NavigatorContextProcessor")
        processor = NavigatorContextProcessor()
        geoip_loaded = processor.geoip is not None
        log_metric("geoip reader loaded", geoip_loaded, expected=True, check=geoip_loaded)
        
        assert processor.geoip is not None, "GeoIP reader should be loaded"
        log_result(True, "GeoIP database loaded successfully")
    
    @pytest.mark.integration
    def test_repository_initialization(self, clean_redis):
        """Verify SentinelStateRepository can connect to Redis."""
        log_header("Repository Initialization Test")
        
        log_section("Creating SentinelStateRepository")
        repo = SentinelStateRepository()
        print(f"  Repository created: {type(repo).__name__}")
        
        log_section("Fetching Context for Non-Existent User")
        context = repo.get_user_context("nonexistent_user_xyz")
        
        print(f"  Returned context keys: {list(context.keys())}")
        log_metric("known_device_hashes", context.get("known_device_hashes"), expected=[], check=context.get("known_device_hashes") == [])
        log_metric("last_geo_coords", context.get("last_geo_coords"), expected=None, check=context.get("last_geo_coords") is None)
        log_metric("last_seen_timestamp", context.get("last_seen_timestamp"), expected=None, check=context.get("last_seen_timestamp") is None)
        log_metric("active_session_count", context.get("active_session_count"), expected=0, check=context.get("active_session_count") == 0)
        
        assert "known_device_hashes" in context
        assert "last_geo_coords" in context
        assert "last_seen_timestamp" in context
        assert "active_session_count" in context
        
        log_result(True, "Repository returns proper defaults for new users")


# =============================================================================
# Scenario B: Happy Path (Baseline)
# =============================================================================

class TestHappyPath:
    """Tests for normal user behavior - the baseline scenario."""
    
    @pytest.mark.integration
    def test_new_user_login(self, context_processor, policy_engine, clean_redis):
        """Test a brand new user logging in for the first time."""
        log_header("New User Login Test")
        
        user_id = "user_1"
        device_id = "device_new_user_1"
        
        log_section("Request Parameters")
        print(f"  user_id: {user_id}")
        print(f"  device_id: {device_id}")
        print(f"  ip_address: 8.8.8.8 (Google DNS)")
        
        request = generate_request(
            user_id=user_id,
            ip_address="8.8.8.8",
            device_id=device_id,
        )
        
        log_section("Processing Request → Metrics")
        metrics = context_processor.process(request)
        for key, value in metrics.items():
            log_metric(key, value)
        
        log_section("Evaluating Metrics → Decision")
        analysis = policy_engine.evaluate(metrics)
        log_metric("decision", analysis.decision.value)
        log_metric("risk_score", f"{analysis.risk_score:.4f}")
        log_metric("anomaly_vectors", analysis.anomaly_vectors)
        
        is_valid_decision = analysis.decision in [SentinelDecision.ALLOW, SentinelDecision.CHALLENGE]
        log_metric("decision valid (ALLOW/CHALLENGE)", is_valid_decision, check=is_valid_decision)
        
        assert is_valid_decision, f"New user should be ALLOW or CHALLENGE, got {analysis.decision}"
        
        is_new_device_check = metrics["is_new_device"] == 1.0
        log_metric("is_new_device == 1.0", is_new_device_check, check=is_new_device_check)
        assert is_new_device_check, "First login should mark device as new"
        
        log_section("Simulating Post-Login State Update")
        repo = SentinelStateRepository()
        repo.update_user_state(user_id, {
            "device_id": device_id,
            "coords": (37.7749, -122.4194),
            "ip": "8.8.8.8",
            "active_session_count": 1,
        })
        print(f"  Updated state for {user_id}")
        
        log_section("Verifying Redis State")
        session_key = f"SESSION:{user_id}"
        session_exists = clean_redis.exists(session_key)
        log_metric(f"EXISTS {session_key}", bool(session_exists), expected=True, check=session_exists)
        assert session_exists, f"Session key should exist: {session_key}"
        
        devices_key = f"PROFILE:{user_id}:devices"
        devices = clean_redis.smembers(devices_key)
        device_registered = device_id in devices
        log_metric(f"SMEMBERS {devices_key}", list(devices))
        log_metric(f"device_id in devices", device_registered, expected=True, check=device_registered)
        assert device_registered, f"Device should be registered: {devices}"
        
        log_result(True, "New user login processed correctly, state persisted")
    
    @pytest.mark.integration
    def test_known_user_normal_behavior(self, context_processor, policy_engine, state_repository, clean_redis):
        """Test a known user with established history."""
        log_header("Known User Normal Behavior Test")
        
        user_id = "user_known"
        device_id = "device_known"
        
        log_section("Pre-seeding User Data in Redis")
        state_repository.update_user_state(user_id, {
            "device_id": device_id,
            "coords": (37.7749, -122.4194),
            "ip": "8.8.8.8",
            "active_session_count": 1,
        })
        print(f"  Created history for {user_id}")
        print(f"  Registered device: {device_id}")
        print(f"  Location: San Francisco (37.7749, -122.4194)")
        
        time.sleep(0.1)
        
        log_section("Request Parameters")
        print(f"  user_id: {user_id}")
        print(f"  device_id: {device_id} (SAME as registered)")
        print(f"  ip_address: 8.8.8.8 (SAME location)")
        
        request = generate_request(
            user_id=user_id,
            ip_address="8.8.8.8",
            device_id=device_id,
        )
        
        log_section("Processing Request → Metrics")
        metrics = context_processor.process(request)
        for key, value in metrics.items():
            log_metric(key, value)
        
        log_section("Evaluating Metrics → Decision")
        analysis = policy_engine.evaluate(metrics)
        log_metric("decision", analysis.decision.value, expected="ALLOW", check=analysis.decision == SentinelDecision.ALLOW)
        log_metric("risk_score", f"{analysis.risk_score:.4f}", check=analysis.risk_score < 0.5)
        log_metric("anomaly_vectors", analysis.anomaly_vectors)
        
        assert analysis.decision == SentinelDecision.ALLOW, "Known user should be ALLOW"
        assert analysis.risk_score < 0.5, f"Risk should be low, got {analysis.risk_score}"
        
        is_known_device = metrics["is_new_device"] == 0.0
        log_metric("is_new_device == 0.0 (recognized)", is_known_device, check=is_known_device)
        assert is_known_device, "Device should be recognized"
        
        log_result(True, "Known user with normal behavior gets ALLOW with low risk")


# =============================================================================
# Scenario C: Impossible Travel (Superman Attack)
# =============================================================================

class TestImpossibleTravel:
    """Tests for impossible travel detection - the Superman attack."""
    
    @pytest.mark.integration
    def test_impossible_travel_block(self, state_repository, clean_redis, mock_geoip):
        """Test detection of physically impossible travel."""
        log_header("Impossible Travel Attack Test")
        
        user_id = "user_2"
        device_id = "device_user_2"
        
        ip_responses = {
            "67.161.0.1": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "city_name": "New York",
                "country_iso": "US",
            },
            "212.58.244.1": {
                "latitude": 51.5074,
                "longitude": -0.1278,
                "city_name": "London",
                "country_iso": "GB",
            },
        }
        
        with mock_geoip(ip_responses):
            from core.processors.context import NavigatorContextProcessor
            from core.models.navigator import NavigatorPolicyEngine
            
            processor = NavigatorContextProcessor()
            engine = NavigatorPolicyEngine()
            
            log_section("Step 1: Login from New York")
            print(f"  IP: 67.161.0.1")
            print(f"  Location: New York (40.7128, -74.0060)")
            
            request_ny = generate_request(
                user_id=user_id,
                ip_address="67.161.0.1",
                device_id=device_id,
            )
            
            metrics_ny = processor.process(request_ny)
            analysis_ny = engine.evaluate(metrics_ny)
            
            log_metric("geo_velocity_mph", f"{metrics_ny['geo_velocity_mph']:.2f}")
            log_metric("decision", analysis_ny.decision.value)
            
            first_login_ok = analysis_ny.decision in [SentinelDecision.ALLOW, SentinelDecision.CHALLENGE]
            log_metric("first login ALLOW/CHALLENGE", first_login_ok, check=first_login_ok)
            
            assert first_login_ok, f"First login should be ALLOW/CHALLENGE, got {analysis_ny.decision}"
            
            log_section("Updating User State with NY Location")
            state_repository.update_user_state(user_id, {
                "device_id": device_id,
                "coords": (40.7128, -74.0060),
                "ip": "67.161.0.1",
                "active_session_count": 1,
            })
            print(f"  Saved coords: (40.7128, -74.0060)")
            
            time.sleep(0.1)
            
            log_section("Step 2: Login from London (5 min later)")
            print(f"  IP: 212.58.244.1")
            print(f"  Location: London (51.5074, -0.1278)")
            print(f"  Distance NYC→London: ~3,459 miles")
            print(f"  If 5 min: velocity = 41,508 mph (IMPOSSIBLE!)")
            
            log_section("Testing with Forced High Velocity")
            print("  (Simulating 5-minute travel time)")
            
            high_velocity_metrics = {
                "geo_velocity_mph": 1000.0,
                "device_ip_mismatch": 0.0,
                "policy_violation": 0.0,
                "is_new_device": 0.0,
            }
            
            for key, value in high_velocity_metrics.items():
                log_metric(key, value)
            
            forced_analysis = engine.evaluate(high_velocity_metrics)
            
            log_section("Policy Engine Evaluation")
            log_metric("decision", forced_analysis.decision.value, expected="BLOCK", check=forced_analysis.decision == SentinelDecision.BLOCK)
            log_metric("risk_score", f"{forced_analysis.risk_score:.4f}")
            log_metric("anomaly_vectors", forced_analysis.anomaly_vectors)
            
            has_impossible_travel = "impossible_travel" in forced_analysis.anomaly_vectors
            log_metric("'impossible_travel' in vectors", has_impossible_travel, check=has_impossible_travel)
            
            assert forced_analysis.decision == SentinelDecision.BLOCK, "Impossible travel should result in BLOCK"
            assert has_impossible_travel, "Should contain 'impossible_travel' vector"
            
            log_result(True, "Impossible travel correctly triggers BLOCK with 'impossible_travel' vector")
    
    @pytest.mark.integration
    def test_impossible_travel_velocity_calculation(self, policy_engine):
        """Test that velocity > 500 mph triggers impossible_travel."""
        log_header("Velocity Threshold Test")
        
        log_section("Test Case 1: Velocity at exactly 500 mph")
        metrics_threshold = {
            "geo_velocity_mph": 500.0,
            "device_ip_mismatch": 0.0,
            "policy_violation": 0.0,
            "is_new_device": 0.0,
        }
        
        log_metric("geo_velocity_mph", 500.0)
        analysis = policy_engine.evaluate(metrics_threshold)
        
        no_trigger_at_500 = "impossible_travel" not in analysis.anomaly_vectors
        log_metric("'impossible_travel' triggered", not no_trigger_at_500, expected=False, check=no_trigger_at_500)
        log_metric("decision", analysis.decision.value)
        
        assert no_trigger_at_500, "Velocity at exactly 500 should not trigger impossible_travel"
        
        log_section("Test Case 2: Velocity at 500.1 mph (just over)")
        metrics_over = {
            "geo_velocity_mph": 500.1,
            "device_ip_mismatch": 0.0,
            "policy_violation": 0.0,
            "is_new_device": 0.0,
        }
        
        log_metric("geo_velocity_mph", 500.1)
        analysis_over = policy_engine.evaluate(metrics_over)
        
        triggers_at_500_1 = "impossible_travel" in analysis_over.anomaly_vectors
        log_metric("'impossible_travel' triggered", triggers_at_500_1, expected=True, check=triggers_at_500_1)
        log_metric("decision", analysis_over.decision.value)
        
        assert triggers_at_500_1, "Velocity over 500 should trigger impossible_travel"
        
        log_result(True, "Velocity threshold (>500 mph) correctly enforced")


# =============================================================================
# Scenario D: Bot Detection (Infrastructure Mismatch)
# =============================================================================

class TestBotDetection:
    """Tests for bot/infrastructure mismatch detection."""
    
    @pytest.mark.integration
    def test_bot_infrastructure_detection(self, state_repository, clean_redis, mock_geoip):
        """Test detection of desktop browser on hosting/VPN infrastructure."""
        log_header("Bot Infrastructure Detection Test")
        
        user_id = "user_3"
        device_id = "device_user_3"
        
        ip_responses = {
            "104.16.0.1": {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "city_name": "San Francisco",
                "country_iso": "US",
            },
        }
        
        log_section("Attack Scenario")
        print("  Attacker uses: Desktop Chrome browser")
        print("  Attacker IP: 104.16.0.1 (Hosting/Cloud Provider)")
        print("  Expected: Desktop + Hosting ASN = BOT")
        
        with mock_geoip(ip_responses):
            from core.processors.context import NavigatorContextProcessor
            from core.models.navigator import NavigatorPolicyEngine
            
            with patch.object(
                NavigatorContextProcessor,
                '_classify_asn',
                return_value='hosting'
            ):
                processor = NavigatorContextProcessor()
                engine = NavigatorPolicyEngine()
                
                desktop_ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                
                log_section("Request Parameters")
                print(f"  user_id: {user_id}")
                print(f"  ip_address: 104.16.0.1 (HOSTING)")
                print(f"  user_agent: Windows Chrome (DESKTOP)")
                
                request = generate_request(
                    user_id=user_id,
                    ip_address="104.16.0.1",
                    device_id=device_id,
                    user_agent=desktop_ua,
                )
                
                log_section("Processing Request → Metrics")
                metrics = processor.process(request)
                for key, value in metrics.items():
                    log_metric(key, value)
                
                mismatch_detected = metrics["device_ip_mismatch"] == 1.0
                log_metric("device_ip_mismatch == 1.0", mismatch_detected, check=mismatch_detected)
                
                assert mismatch_detected, f"Desktop + hosting IP should trigger mismatch, got {metrics['device_ip_mismatch']}"
                
                log_section("Policy Engine Evaluation")
                analysis = engine.evaluate(metrics)
                
                log_metric("decision", analysis.decision.value, expected="BLOCK", check=analysis.decision == SentinelDecision.BLOCK)
                log_metric("risk_score", f"{analysis.risk_score:.4f}")
                log_metric("anomaly_vectors", analysis.anomaly_vectors)
                
                has_infra_mismatch = "infra_mismatch" in analysis.anomaly_vectors
                log_metric("'infra_mismatch' in vectors", has_infra_mismatch, check=has_infra_mismatch)
                
                assert has_infra_mismatch, f"Should contain 'infra_mismatch', got {analysis.anomaly_vectors}"
                assert analysis.decision == SentinelDecision.BLOCK, f"Infrastructure mismatch should BLOCK, got {analysis.decision}"
                
                log_result(True, "Desktop on hosting IP correctly detected as bot → BLOCK")
    
    @pytest.mark.integration
    def test_mobile_on_vpn_not_mismatch(self, policy_engine):
        """Test that mobile browser on VPN does NOT trigger mismatch."""
        log_header("Mobile on VPN Test (Should NOT Trigger)")
        
        log_section("Scenario")
        print("  User: Mobile phone user on legitimate VPN")
        print("  Expected: No mismatch (mobile VPN is common)")
        
        metrics_mobile = {
            "geo_velocity_mph": 0.0,
            "device_ip_mismatch": 0.0,
            "policy_violation": 0.0,
            "is_new_device": 0.0,
        }
        
        log_section("Metrics (device_ip_mismatch = 0.0)")
        for key, value in metrics_mobile.items():
            log_metric(key, value)
        
        analysis = policy_engine.evaluate(metrics_mobile)
        
        log_section("Policy Engine Evaluation")
        is_allow = analysis.decision == SentinelDecision.ALLOW
        log_metric("decision", analysis.decision.value, expected="ALLOW", check=is_allow)
        
        no_infra_mismatch = "infra_mismatch" not in analysis.anomaly_vectors
        log_metric("'infra_mismatch' absent", no_infra_mismatch, check=no_infra_mismatch)
        
        assert is_allow, "Mobile on VPN should be ALLOW"
        assert no_infra_mismatch
        
        log_result(True, "Mobile on VPN correctly allowed (no false positive)")


# =============================================================================
# Scenario E: Persistence & Session Continuity
# =============================================================================

class TestSessionPersistence:
    """Tests for Redis persistence and session state management."""
    
    @pytest.mark.integration
    def test_session_continuity(self, state_repository, clean_redis):
        """Test that devices are remembered across sessions."""
        log_header("Session Continuity Test")
        
        user_id = "user_4"
        device_a = "device_A_test"
        
        log_section("Step 1: Check Device is Initially Unknown")
        context_before = state_repository.get_user_context(user_id)
        device_unknown = device_a not in context_before["known_device_hashes"]
        log_metric("known_device_hashes", context_before["known_device_hashes"])
        log_metric(f"'{device_a}' unknown", device_unknown, check=device_unknown)
        
        assert device_unknown, "Device should not exist before first login"
        
        log_section("Step 2: First Login - Register Device")
        state_repository.update_user_state(user_id, {
            "device_id": device_a,
            "coords": (37.7749, -122.4194),
            "ip": "8.8.8.8",
            "active_session_count": 1,
        })
        print(f"  Registered device: {device_a}")
        
        log_section("Step 3: Verify Device is Now Known")
        context_after = state_repository.get_user_context(user_id)
        device_known = device_a in context_after["known_device_hashes"]
        log_metric("known_device_hashes", context_after["known_device_hashes"])
        log_metric(f"'{device_a}' known", device_known, check=device_known)
        
        assert device_known, "Device should be registered after first login"
        
        log_section("Step 4: Second Login - Test is_new_device Metric")
        from core.processors.context import NavigatorContextProcessor
        processor = NavigatorContextProcessor()
        
        request = generate_request(
            user_id=user_id,
            ip_address="8.8.8.8",
            device_id=device_a,
        )
        
        metrics = processor.process(request)
        device_recognized = metrics["is_new_device"] == 0.0
        log_metric("is_new_device", metrics["is_new_device"], expected=0.0, check=device_recognized)
        
        assert device_recognized, f"Device should be recognized on second login, got {metrics['is_new_device']}"
        
        log_result(True, "Device correctly remembered between sessions")
    
    @pytest.mark.integration
    def test_device_cap_enforcement(self, state_repository, clean_redis):
        """Test that known devices are capped at MAX_KNOWN_DEVICES (20)."""
        log_header("Device Cap Enforcement Test")
        
        user_id = "user_device_cap"
        
        log_section("Adding 25 Devices (Exceeds 20 Limit)")
        for i in range(25):
            state_repository.update_user_state(user_id, {
                "device_id": f"device_{i}",
                "coords": (37.0, -122.0),
            })
        print(f"  Added devices: device_0 through device_24")
        
        log_section("Checking Device Count")
        devices_key = f"PROFILE:{user_id}:devices"
        device_count = clean_redis.scard(devices_key)
        
        is_capped = device_count <= 20
        log_metric("device count", device_count, expected="≤ 20", check=is_capped)
        log_metric("MAX_KNOWN_DEVICES", 20)
        
        assert is_capped, f"Device count should be capped at 20, got {device_count}"
        
        log_result(True, f"Device count correctly capped at {device_count}")
    
    @pytest.mark.integration
    def test_session_ttl_refresh(self, state_repository, clean_redis):
        """Test that session TTL is refreshed on update."""
        log_header("Session TTL Refresh Test")
        
        user_id = "user_ttl_test"
        session_key = f"SESSION:{user_id}"
        
        log_section("First Update - Create Session")
        state_repository.update_user_state(user_id, {
            "device_id": "device_ttl",
            "coords": (37.0, -122.0),
            "ip": "8.8.8.8",
            "active_session_count": 1,
        })
        
        ttl_1 = clean_redis.ttl(session_key)
        log_metric("TTL after first update", f"{ttl_1} seconds")
        log_metric("TTL > 0", ttl_1 > 0, check=ttl_1 > 0)
        log_metric("TTL <= 86400", ttl_1 <= 86400, check=ttl_1 <= 86400)
        
        assert ttl_1 > 0, "Session should have TTL"
        assert ttl_1 <= 86400, f"TTL should be <= 86400, got {ttl_1}"
        
        log_section("Waiting 0.5 Seconds...")
        time.sleep(0.5)
        
        log_section("Second Update - Should Refresh TTL")
        state_repository.update_user_state(user_id, {
            "device_id": "device_ttl",
            "coords": (37.0, -122.0),
            "ip": "8.8.8.8",
            "active_session_count": 2,
        })
        
        ttl_2 = clean_redis.ttl(session_key)
        log_metric("TTL after second update", f"{ttl_2} seconds")
        log_metric("TTL refreshed (> 0)", ttl_2 > 0, check=ttl_2 > 0)
        
        assert ttl_2 > 0, "TTL should still be positive after refresh"
        
        log_result(True, "Session TTL correctly refreshed on update")


# =============================================================================
# Additional Tests: Strike System
# =============================================================================

class TestMouseSessionTracker:
    """Tests for the session-level strike system."""
    
    def test_strike_increment(self):
        """Test that bot strokes increment strikes."""
        log_header("Strike Increment Test")
        
        tracker = MouseSessionTracker()
        
        log_section("Initial State")
        log_metric("strikes", tracker.get_strikes(), expected=0, check=tracker.get_strikes() == 0)
        
        log_section("Recording Bot Strokes")
        tracker.record_bot_stroke()
        log_metric("after 1 bot stroke", tracker.get_strikes(), expected=1, check=tracker.get_strikes() == 1)
        
        tracker.record_bot_stroke()
        log_metric("after 2 bot strokes", tracker.get_strikes(), expected=2, check=tracker.get_strikes() == 2)
        
        assert tracker.get_strikes() == 2
        log_result(True, "Bot strokes correctly increment strikes")
    
    def test_strike_decrement(self):
        """Test that human strokes decrement strikes (min 0)."""
        log_header("Strike Decrement Test")
        
        tracker = MouseSessionTracker()
        
        log_section("Setup: Add 2 Bot Strokes")
        tracker.record_bot_stroke()
        tracker.record_bot_stroke()
        log_metric("strikes", tracker.get_strikes(), expected=2)
        
        log_section("Recording Human Strokes")
        tracker.record_human_stroke()
        log_metric("after 1 human stroke", tracker.get_strikes(), expected=1, check=tracker.get_strikes() == 1)
        
        tracker.record_human_stroke()
        tracker.record_human_stroke()  # Extra to test min(0)
        log_metric("after 3 human strokes (min 0)", tracker.get_strikes(), expected=0, check=tracker.get_strikes() == 0)
        
        assert tracker.get_strikes() == 0
        log_result(True, "Human strokes correctly decrement strikes (min 0)")
    
    def test_flag_threshold(self):
        """Test that session is flagged at 3 strikes."""
        log_header("Flag Threshold Test")
        
        tracker = MouseSessionTracker()
        
        log_section("Before Threshold")
        log_metric("flagged (0 strikes)", tracker.is_flagged(), expected=False, check=not tracker.is_flagged())
        
        tracker.record_bot_stroke()
        tracker.record_bot_stroke()
        log_metric("flagged (2 strikes)", tracker.is_flagged(), expected=False, check=not tracker.is_flagged())
        
        log_section("At Threshold (3 strikes)")
        tracker.record_bot_stroke()
        log_metric("strikes", tracker.get_strikes(), expected=3)
        log_metric("flagged", tracker.is_flagged(), expected=True, check=tracker.is_flagged())
        
        assert tracker.is_flagged()
        log_result(True, "Session correctly flagged at 3 strikes")
    
    def test_flag_persistence(self):
        """Test that flag persists even if strikes decrease."""
        log_header("Flag Persistence Test")
        
        tracker = MouseSessionTracker()
        
        log_section("Get Flagged (3 Bot Strokes)")
        for _ in range(3):
            tracker.record_bot_stroke()
        log_metric("flagged", tracker.is_flagged(), check=tracker.is_flagged())
        
        log_section("Attempt Recovery (Human Strokes)")
        tracker.record_human_stroke()
        tracker.record_human_stroke()
        log_metric("strikes after human strokes", tracker.get_strikes())
        log_metric("still flagged", tracker.is_flagged(), expected=True, check=tracker.is_flagged())
        
        assert tracker.is_flagged()
        log_result(True, "Flag persists after strike reduction (once flagged, stays flagged)")
    
    def test_reset(self):
        """Test tracker reset."""
        log_header("Tracker Reset Test")
        
        tracker = MouseSessionTracker()
        
        log_section("Before Reset")
        for _ in range(5):
            tracker.record_bot_stroke()
        log_metric("strikes", tracker.get_strikes())
        log_metric("flagged", tracker.is_flagged())
        
        log_section("After Reset")
        tracker.reset()
        log_metric("strikes", tracker.get_strikes(), expected=0, check=tracker.get_strikes() == 0)
        log_metric("flagged", tracker.is_flagged(), expected=False, check=not tracker.is_flagged())
        
        assert tracker.get_strikes() == 0
        assert not tracker.is_flagged()
        log_result(True, "Reset clears strikes and flag")


# =============================================================================
# Policy Engine Edge Cases
# =============================================================================

class TestPolicyEngineEdgeCases:
    """Tests for NavigatorPolicyEngine edge cases and thresholds."""
    
    def test_block_threshold(self, policy_engine):
        """Test that risk >= 0.85 triggers BLOCK."""
        log_header("Block Threshold Test (≥ 0.85)")
        
        log_section("Metrics (velocity = 425 mph → risk = 0.85)")
        metrics = {
            "geo_velocity_mph": 425.0,
            "device_ip_mismatch": 0.0,
            "policy_violation": 0.0,
            "is_new_device": 0.0,
        }
        
        for key, value in metrics.items():
            log_metric(key, value)
        
        print(f"\n  Calculation: 425 / 500 = 0.85")
        
        analysis = policy_engine.evaluate(metrics)
        
        log_section("Evaluation")
        log_metric("risk_score", f"{analysis.risk_score:.4f}")
        log_metric("decision", analysis.decision.value, expected="BLOCK", check=analysis.decision == SentinelDecision.BLOCK)
        
        assert analysis.decision == SentinelDecision.BLOCK
        log_result(True, "Risk ≥ 0.85 correctly triggers BLOCK")
    
    def test_challenge_threshold(self, policy_engine):
        """Test that 0.50 <= risk < 0.85 triggers CHALLENGE."""
        log_header("Challenge Threshold Test (0.50 ≤ risk < 0.85)")
        
        log_section("Metrics (velocity = 250 mph → risk = 0.50)")
        metrics = {
            "geo_velocity_mph": 250.0,
            "device_ip_mismatch": 0.0,
            "policy_violation": 0.0,
            "is_new_device": 0.0,
        }
        
        for key, value in metrics.items():
            log_metric(key, value)
        
        print(f"\n  Calculation: 250 / 500 = 0.50")
        
        analysis = policy_engine.evaluate(metrics)
        
        log_section("Evaluation")
        log_metric("risk_score", f"{analysis.risk_score:.4f}")
        log_metric("decision", analysis.decision.value, expected="CHALLENGE", check=analysis.decision == SentinelDecision.CHALLENGE)
        
        assert analysis.decision == SentinelDecision.CHALLENGE
        log_result(True, "Risk in [0.50, 0.85) correctly triggers CHALLENGE")
    
    def test_allow_threshold(self, policy_engine):
        """Test that risk < 0.50 triggers ALLOW."""
        log_header("Allow Threshold Test (< 0.50)")
        
        log_section("Metrics (velocity = 200 mph → risk = 0.40)")
        metrics = {
            "geo_velocity_mph": 200.0,
            "device_ip_mismatch": 0.0,
            "policy_violation": 0.0,
            "is_new_device": 0.0,
        }
        
        for key, value in metrics.items():
            log_metric(key, value)
        
        print(f"\n  Calculation: 200 / 500 = 0.40")
        
        analysis = policy_engine.evaluate(metrics)
        
        log_section("Evaluation")
        log_metric("risk_score", f"{analysis.risk_score:.4f}")
        log_metric("decision", analysis.decision.value, expected="ALLOW", check=analysis.decision == SentinelDecision.ALLOW)
        
        assert analysis.decision == SentinelDecision.ALLOW
        log_result(True, "Risk < 0.50 correctly triggers ALLOW")
    
    def test_policy_violation_block(self, policy_engine):
        """Test that policy_violation = 1.0 triggers BLOCK."""
        log_header("Policy Violation Test")
        
        log_section("Scenario: Intern Accessing Production")
        metrics = {
            "geo_velocity_mph": 0.0,
            "device_ip_mismatch": 0.0,
            "policy_violation": 1.0,
            "is_new_device": 0.0,
        }
        
        for key, value in metrics.items():
            log_metric(key, value)
        
        analysis = policy_engine.evaluate(metrics)
        
        log_section("Evaluation")
        log_metric("decision", analysis.decision.value, expected="BLOCK", check=analysis.decision == SentinelDecision.BLOCK)
        log_metric("anomaly_vectors", analysis.anomaly_vectors)
        
        has_violation = "policy_violation" in analysis.anomaly_vectors
        log_metric("'policy_violation' in vectors", has_violation, check=has_violation)
        
        assert analysis.decision == SentinelDecision.BLOCK
        assert has_violation
        log_result(True, "Policy violation correctly triggers BLOCK")
    
    def test_risk_score_clamping(self, policy_engine):
        """Test that risk score is clamped to [0.0, 1.0]."""
        log_header("Risk Score Clamping Test")
        
        log_section("Extreme Velocity (10,000 mph = 20x limit)")
        metrics = {
            "geo_velocity_mph": 10000.0,
            "device_ip_mismatch": 0.0,
            "policy_violation": 0.0,
            "is_new_device": 0.0,
        }
        
        log_metric("geo_velocity_mph", 10000.0)
        print(f"\n  Raw: 10000 / 500 = 20.0")
        print(f"  Expected: Clamped to 1.0")
        
        analysis = policy_engine.evaluate(metrics)
        
        log_section("Evaluation")
        clamped_correctly = 0.0 <= analysis.risk_score <= 1.0
        log_metric("risk_score", f"{analysis.risk_score:.4f}", expected="≤ 1.0", check=clamped_correctly)
        
        assert analysis.risk_score <= 1.0, "Risk should be clamped to 1.0"
        assert analysis.risk_score >= 0.0, "Risk should be non-negative"
        log_result(True, "Risk score correctly clamped to [0.0, 1.0]")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
