"""
Sentinel Navigator Context Processor

Enriches raw requests into numeric risk metrics only.
No decisions. No blocking. Pure metric derivation.

Uses GeoIP2 for location lookup and behavioral analysis.
"""

import logging
import math
import time
from typing import Any, Dict, Optional, Tuple

import geoip2.database
from user_agents import parse as parse_user_agent

from core.schemas.inputs import EvaluationRequest
from persistence.repository import SentinelStateRepository


logger = logging.getLogger(__name__)


# =============================================================================
# ASN Reputation Scoring (MANDATORY)
# =============================================================================

ASN_REPUTATION: Dict[str, float] = {
    "residential": 0.0,
    "mobile": 0.1,
    "unknown": 0.3,
    "hosting": 0.9,
    "vpn": 0.9,
}


# =============================================================================
# Context Processor
# =============================================================================

class NavigatorContextProcessor:
    """
    Enriches raw requests into numeric risk metrics.
    
    This processor is responsible for:
    1. GeoIP lookup and velocity calculation
    2. Device identity verification
    3. Session state analysis
    4. Policy violation detection
    
    All outputs are numeric metrics - no decisions are made here.
    """
    
    def __init__(self) -> None:
        """Initialize processor with repository and GeoIP reader."""
        self.repo = SentinelStateRepository()
        
        # GeoIP - Fail open if database is unavailable
        try:
            self.geoip = geoip2.database.Reader("assets/GeoLite2-City.mmdb")
        except Exception as e:
            logger.warning(f"GeoIP database unavailable, using defaults: {e}")
            self.geoip = None
    
    def process(self, request: EvaluationRequest) -> Dict[str, Any]:
        """
        Process evaluation request and derive all context metrics.
        
        Args:
            request: The incoming evaluation request with user/network context
            
        Returns:
            Dictionary of metric names to values:
            - geo_velocity_mph: Travel speed between last and current location
            - time_since_last_seen: Seconds since last activity
            - device_ip_mismatch: 1.0 if Desktop UA + VPN/hosting ASN
            - is_new_device: 1.0 if device_id is unknown
            - simultaneous_sessions: Count of active sessions
            - policy_violation: 1.0 if role violates resource access
            - ip_reputation: Risk score based on IP type
            - current_geo_data: Raw geo dict {city, country, coords, asn_type}
        """
        user_id = request.user_session.user_id
        current_time = time.time()
        
        # Get user history from repository
        history = self.repo.get_user_context(user_id)
        
        # Resolve current IP location
        current_ip = request.network_context.ip_address
        current_geo = self._resolve_ip(current_ip)
        
        # Get device_id from client fingerprint if available
        # Note: This expects the EvaluationRequest to have client_fingerprint
        device_id = self._get_device_id(request)
        
        # Calculate all metrics
        metrics: Dict[str, Any] = {
            "geo_velocity_mph": self._calc_geo_velocity(
                history.get("last_geo_coords"),
                current_geo.get("coords"),
                history.get("last_seen_timestamp"),
                current_time
            ),
            "time_since_last_seen": self._calc_time_since_last_seen(
                history.get("last_seen_timestamp"),
                current_time
            ),
            "device_ip_mismatch": self._calc_device_ip_mismatch(
                request.network_context.user_agent,
                current_geo.get("asn_type", "unknown")
            ),
            "is_new_device": self._calc_is_new_device(
                device_id,
                history.get("known_device_hashes", [])
            ),
            "simultaneous_sessions": float(history.get("active_session_count", 0)),
            "policy_violation": self._calc_policy_violation(
                request.user_session.role,
                request.business_context.resource_target
            ),
            "ip_reputation": ASN_REPUTATION.get(
                current_geo.get("asn_type", "unknown"),
                0.3
            ),
            "is_unknown_user_agent": self._calc_is_unknown_user_agent(
                request.network_context.user_agent
            ),
            # Raw geo data for persistence (used by orchestrator/TOFU)
            "current_geo_data": {
                "city": current_geo.get("city", "Unknown"),
                "country": current_geo.get("country", "XX"),
                "coords": current_geo.get("coords"),
            },
        }
        
        return metrics
    
    def _get_device_id(self, request: EvaluationRequest) -> Optional[str]:
        """Extract device_id from request, if available."""
        # The spec says device identity = request.client_fingerprint.device_id
        # We need to check if client_fingerprint is available in network_context
        if hasattr(request.network_context, 'client_fingerprint'):
            fingerprint = request.network_context.client_fingerprint
            if hasattr(fingerprint, 'device_id'):
                return fingerprint.device_id
        
        # Fallback: generate from user agent if no device_id
        return None
    
    def _resolve_ip(self, ip_address: str) -> Dict[str, Any]:
        """
        Resolve IP address to location data.
        
        Args:
            ip_address: IPv4 or IPv6 address
            
        Returns:
            Dict with coords, asn_type, city, country
            Returns neutral defaults for private IPs or on GeoIP failure
        """
        # Check for private/reserved IP ranges
        if self._is_private_ip(ip_address):
            return {
                "coords": None,
                "asn_type": "unknown",
                "city": "Unknown",
                "country": "XX",
            }
        
        # If GeoIP is unavailable, return defaults
        if self.geoip is None:
            return {
                "coords": None,
                "asn_type": "unknown",
                "city": "Unknown",
                "country": "XX",
            }
        
        try:
            response = self.geoip.city(ip_address)
            
            coords = None
            if response.location.latitude and response.location.longitude:
                coords = (response.location.latitude, response.location.longitude)
            
            # Determine ASN type (simplified classification)
            asn_type = self._classify_asn(response)
            
            return {
                "coords": coords,
                "asn_type": asn_type,
                "city": response.city.name or "Unknown",
                "country": response.country.iso_code or "XX",
            }
        except Exception as e:
            logger.debug(f"GeoIP lookup failed for {ip_address}: {e}")
            return {
                "coords": None,
                "asn_type": "unknown",
                "city": "Unknown",
                "country": "XX",
            }
    
    def _is_private_ip(self, ip_address: str) -> bool:
        """Check if IP is a private/reserved address."""
        # Simple check for common private ranges
        private_prefixes = (
            "10.",
            "172.16.", "172.17.", "172.18.", "172.19.",
            "172.20.", "172.21.", "172.22.", "172.23.",
            "172.24.", "172.25.", "172.26.", "172.27.",
            "172.28.", "172.29.", "172.30.", "172.31.",
            "192.168.",
            "127.",
            "0.",
            "::1",
            "fe80:",
        )
        return ip_address.startswith(private_prefixes)
    
    def _classify_asn(self, geoip_response: Any) -> str:
        """
        Classify ASN type from GeoIP response.
        
        Returns one of: residential, mobile, hosting, vpn, unknown
        """
        # In production, this would use additional ASN databases
        # For now, return unknown and let external enrichment handle it
        return "unknown"
    
    def _haversine(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate great-circle distance between two points.
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in miles
        """
        # Convert to radians
        lat1_r = math.radians(lat1)
        lon1_r = math.radians(lon1)
        lat2_r = math.radians(lat2)
        lon2_r = math.radians(lon2)
        
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        
        a = (
            math.sin(dlat / 2) ** 2 +
            math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in miles
        r = 3956
        
        return r * c
    
    def _calc_geo_velocity(
        self,
        last_coords: Optional[Tuple[float, float]],
        current_coords: Optional[Tuple[float, float]],
        last_timestamp: Optional[float],
        current_timestamp: float
    ) -> float:
        """
        Calculate geographic velocity in mph.
        
        CRITICAL FIX:
        - Returns 0.0 if no history (last_coords is None)
        - Returns 0.0 if time delta < 1.0 second (prevents division issues)
        - Otherwise returns miles / hours
        """
        # No history - no velocity
        if last_coords is None:
            return 0.0
        
        # No current location - no velocity
        if current_coords is None:
            return 0.0
        
        # No timestamp history - no velocity
        if last_timestamp is None:
            return 0.0
        
        # Calculate time since last seen in seconds
        time_since_last_seen = current_timestamp - last_timestamp
        
        # Guard against very small time deltas (< 1 second)
        if time_since_last_seen < 1.0:
            return 0.0
        
        # Calculate distance in miles
        miles = self._haversine(
            last_coords[0], last_coords[1],
            current_coords[0], current_coords[1]
        )
        
        # Convert time to hours
        hours = time_since_last_seen / 3600.0
        
        return miles / hours
    
    def _calc_time_since_last_seen(
        self,
        last_timestamp: Optional[float],
        current_timestamp: float
    ) -> float:
        """Calculate seconds since last activity."""
        if last_timestamp is None:
            return 0.0
        
        return current_timestamp - last_timestamp
    
    def _calc_device_ip_mismatch(
        self,
        user_agent: str,
        asn_type: str
    ) -> float:
        """
        Detect desktop user agent connecting from hosting/VPN IP.
        
        Returns 1.0 if Desktop UA AND ASN in {"hosting", "vpn"}, else 0.0
        """
        # Parse user agent
        ua = parse_user_agent(user_agent)
        
        # Check if desktop
        is_desktop = ua.is_pc
        
        # Check if suspicious ASN
        is_suspicious_asn = asn_type in {"hosting", "vpn"}
        
        if is_desktop and is_suspicious_asn:
            return 1.0
        
        return 0.0
    
    def _calc_is_new_device(
        self,
        device_id: Optional[str],
        known_devices: list
    ) -> float:
        """
        Check if device is new (unknown device_id).
        
        Returns 1.0 if new device, 0.0 if known or indeterminate.
        """
        if device_id is None:
            # No device_id means fingerprinting is not available.
            # Treat as neutral (0.0) to avoid permanent CHALLENGE loop
            # for clients that don't send client_fingerprint.
            return 0.0
        
        if device_id not in known_devices:
            return 1.0
        
        return 0.0
    
    def _calc_policy_violation(
        self,
        role: str,
        resource_target: str
    ) -> float:
        """
        Check for role ↔ resource policy violations.
        
        Returns 1.0 if violation detected, 0.0 otherwise.
        """
        role_lower = role.lower()
        resource_lower = resource_target.lower()
        
        # Interns cannot access production resources
        if role_lower == "intern" and "prod" in resource_lower:
            return 1.0
        
        # Viewers cannot access admin resources
        if role_lower == "viewer" and "admin" in resource_lower:
            return 1.0
        
        # Analyst cannot access secrets
        if role_lower == "analyst" and "secret" in resource_lower:
            return 1.0
        
        return 0.0

    def _calc_is_unknown_user_agent(self, user_agent: str) -> float:
        """
        Detect non-browser user agents (bots, scripts, automation tools).
        
        Returns 1.0 if the UA doesn't parse to a known browser family.
        Bot UAs like 'BotAttackDemo/1.0' or 'python-requests' return
        browser.family='Other' from the ua parser.
        """
        ua = parse_user_agent(user_agent)
        
        # ua-parser marks bots and unknown UAs
        if ua.is_bot:
            return 1.0
        
        # 'Other' family means the parser couldn't identify a real browser
        if ua.browser.family == "Other":
            return 1.0
        
        return 0.0
