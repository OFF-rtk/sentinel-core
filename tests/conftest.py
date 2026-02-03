"""
Sentinel Test Suite - Shared Pytest Fixtures

This conftest.py provides fixtures for all test categories including:
- Redis connection and cleanup for Navigator tests
- GeoIP mocking utilities
- Processor and engine instances

Usage:
    pytest tests/ -v -s
"""

import os
import pytest
from contextlib import contextmanager
from typing import Dict, Optional
from unittest.mock import patch, MagicMock

# =============================================================================
# Path Helpers
# =============================================================================

def get_project_root() -> str:
    """Get the absolute path to the project root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_tests_dir() -> str:
    """Get the absolute path to the tests directory."""
    return os.path.dirname(os.path.abspath(__file__))

def get_assets_dir() -> str:
    """Get the absolute path to the tests/assets directory."""
    return os.path.join(get_tests_dir(), "assets")

def get_results_dir() -> str:
    """Get the absolute path to the tests/results directory."""
    return os.path.join(get_tests_dir(), "results")


# =============================================================================
# Redis Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def redis_client():
    """
    Session-scoped Redis client for integration tests.
    
    Requires Docker Redis to be running:
        cd infrastructure/redis && docker compose up -d
    """
    import redis
    
    host = os.environ.get("REDIS_HOST", "localhost")
    port = int(os.environ.get("REDIS_PORT", "6379"))
    password = os.environ.get("REDIS_PASSWORD", "PASS")
    
    try:
        client = redis.Redis(
            host=host,
            port=port,
            password=password,
            decode_responses=True,
        )
        client.ping()
        yield client
        client.close()
    except redis.ConnectionError:
        pytest.skip(f"Redis not available at {host}:{port}")


@pytest.fixture
def clean_redis(redis_client):
    """
    Function-scoped fixture that provides a clean Redis state.
    Flushes the database after each test for isolation.
    """
    yield redis_client
    redis_client.flushdb()


# =============================================================================
# GeoIP Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def geoip_available() -> bool:
    """Check if GeoIP database is available."""
    geoip_path = os.path.join(get_project_root(), "assets", "GeoLite2-City.mmdb")
    return os.path.exists(geoip_path)


@pytest.fixture
def mock_geoip():
    """
    Fixture that returns a context manager for mocking GeoIP responses.
    
    Usage:
        def test_example(mock_geoip):
            ip_responses = {
                "8.8.8.8": {"latitude": 37.7749, "longitude": -122.4194, ...}
            }
            with mock_geoip(ip_responses):
                # Your test code here
                pass
    """
    @contextmanager
    def _mock_geoip(ip_responses: Dict[str, dict]):
        """
        Create a mock GeoIP reader that returns specified responses.
        
        Args:
            ip_responses: Dict mapping IP addresses to response dicts with:
                - latitude: float
                - longitude: float
                - city_name: str (optional)
                - country_iso: str (optional)
        """
        def create_mock_response(ip: str):
            if ip not in ip_responses:
                raise Exception(f"IP {ip} not in mock database")
            
            data = ip_responses[ip]
            
            mock_response = MagicMock()
            mock_response.location.latitude = data.get("latitude", 0.0)
            mock_response.location.longitude = data.get("longitude", 0.0)
            
            if "city_name" in data:
                mock_response.city.name = data["city_name"]
            else:
                mock_response.city.name = "MockCity"
            
            if "country_iso" in data:
                mock_response.country.iso_code = data["country_iso"]
            else:
                mock_response.country.iso_code = "US"
            
            return mock_response
        
        mock_reader = MagicMock()
        mock_reader.city.side_effect = create_mock_response
        
        with patch("geoip2.database.Reader") as MockReader:
            MockReader.return_value = mock_reader
            yield mock_reader
    
    return _mock_geoip


# =============================================================================
# Processor & Engine Fixtures
# =============================================================================

@pytest.fixture
def context_processor(geoip_available):
    """Create a NavigatorContextProcessor instance."""
    from core.processors.context import NavigatorContextProcessor
    return NavigatorContextProcessor()


@pytest.fixture
def policy_engine():
    """Create a NavigatorPolicyEngine instance."""
    from core.models.navigator import NavigatorPolicyEngine
    return NavigatorPolicyEngine()


@pytest.fixture
def state_repository(clean_redis):
    """Create a SentinelStateRepository instance with clean Redis."""
    from persistence.repository import SentinelStateRepository
    return SentinelStateRepository()
