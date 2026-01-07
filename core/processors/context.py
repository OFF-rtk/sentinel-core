"""
Sentinel Context Processor

Derives contextual risk metrics from evaluation request and user history.
Uses GeoIP lookup and behavioral analysis to generate the 7 "Golden Metrics".
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple

from core.schemas.inputs import EvaluationRequest


# =============================================================================
# Mock GeoIP Database
# =============================================================================

# Mock IP to location mapping for demonstration
# In production, this would use MaxMind GeoIP2 or similar service
_IP_DB: Dict[str, Dict[str, Any]] = {
    "107.208.116.189": {
        "city": "West Billfort",
        "country": "CA",
        "asn": "AS58229 Allen, Espinoza and Campbell",
        "coords": (49.2827, -123.1207),  # Vancouver area
        "reputation": "corporate",
    },
    "192.168.1.1": {
        "city": "Local",
        "country": "US",
        "asn": "AS0 Private Network",
        "coords": (37.7749, -122.4194),  # San Francisco
        "reputation": "residential",
    },
    "185.220.101.1": {
        "city": "Unknown",
        "country": "DE",
        "asn": "AS12345 VPN Provider",
        "coords": (52.5200, 13.4050),  # Berlin
        "reputation": "vpn",
    },
}

# Default for unknown IPs
_DEFAULT_GEO = {
    "city": "Unknown",
    "country": "XX",
    "asn": "AS0 Unknown",
    "coords": (0.0, 0.0),
    "reputation": "unknown",
}


# =============================================================================
# Helper Functions
# =============================================================================

def _haversine(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    
    Args:
        coord1: (latitude, longitude) of first point
        coord2: (latitude, longitude) of second point
        
    Returns:
        Distance in miles
    """
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth's radius in miles
    r = 3956
    
    return r * c


def _lookup_ip(ip_address: str) -> Dict[str, Any]:
    """
    Look up IP address in mock database.
    
    Args:
        ip_address: IPv4/IPv6 address string
        
    Returns:
        GeoIP data dictionary
    """
    return _IP_DB.get(ip_address, _DEFAULT_GEO)


# =============================================================================
# IP Reputation Scoring
# =============================================================================

_REPUTATION_SCORES: Dict[str, float] = {
    "residential": 0.0,
    "corporate": 0.1,
    "mobile": 0.2,
    "hosting": 0.5,
    "proxy": 0.7,
    "vpn": 0.8,
    "tor": 1.0,
    "unknown": 0.5,
}


# =============================================================================
# Context Processor
# =============================================================================

class ContextProcessor:
    """
    Derives contextual risk metrics from evaluation request and user history.
    
    Metrics derived:
    1. geo_velocity_mph: Travel speed between last and current location
    2. time_since_last_seen: Seconds since last activity
    3. is_new_device: Whether JA3 hash is unknown
    4. transaction_magnitude: Ratio of current to average transaction
    5. ip_reputation_score: Risk score based on IP type
    6. time_anomaly_score: Whether action is outside usual hours
    7. policy_violation_flag: Whether role violates resource access
    """
    
    def derive_context_metrics(
        self, 
        request: EvaluationRequest, 
        history: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Derive all context metrics from request and user history.
        
        Args:
            request: Current evaluation request
            history: User's historical data snapshot containing:
                - last_geo_coords: (lat, lon) of last action
                - last_seen_timestamp: Unix timestamp of last activity
                - known_device_hashes: Set of known JA3 hashes
                - avg_transaction_amount: Average transaction value
                - usual_hours: List of typical activity hours (0-23)
                - role: User's RBAC role
                
        Returns:
            Dictionary of metric names to values
        """
        current_ip = request.network_context.ip_address
        current_geo = _lookup_ip(current_ip)
        current_time = time.time()
        
        return {
            "geo_velocity_mph": self._calc_geo_velocity(
                history.get("last_geo_coords"),
                current_geo.get("coords", (0.0, 0.0)),
                history.get("last_seen_timestamp"),
                current_time
            ),
            "time_since_last_seen": self._calc_time_since_last_seen(
                history.get("last_seen_timestamp"),
                current_time
            ),
            "is_new_device": self._check_new_device(
                request.network_context.ja3_hash,
                history.get("known_device_hashes", set())
            ),
            "transaction_magnitude": self._calc_transaction_magnitude(
                request.business_context.transaction_details,
                history.get("avg_transaction_amount", 0.0)
            ),
            "ip_reputation_score": self._get_ip_reputation_score(current_ip),
            "time_anomaly_score": self._calc_time_anomaly(
                current_time,
                history.get("usual_hours", [])
            ),
            "policy_violation_flag": self._check_policy_violation(
                request.user_session.role,
                request.business_context.resource_target
            ),
        }
    
    def get_geo_location(self, ip_address: str) -> Dict[str, str]:
        """
        Get GeoLocation data for an IP address.
        
        Returns dict with: asn, city, country
        """
        geo = _lookup_ip(ip_address)
        return {
            "asn": geo.get("asn", "AS0 Unknown"),
            "city": geo.get("city", "Unknown"),
            "country": geo.get("country", "XX"),
        }
    
    def get_ip_reputation(self, ip_address: str) -> str:
        """Get IP reputation category."""
        geo = _lookup_ip(ip_address)
        return geo.get("reputation", "unknown")
    
    def _calc_geo_velocity(
        self,
        last_coords: Optional[Tuple[float, float]],
        current_coords: Tuple[float, float],
        last_timestamp: Optional[float],
        current_timestamp: float
    ) -> float:
        """
        Calculate geographic velocity (mph) between locations.
        
        Returns 0.0 if insufficient history or same location.
        """
        if last_coords is None or last_timestamp is None:
            return 0.0
        
        distance_miles = _haversine(last_coords, current_coords)
        time_hours = (current_timestamp - last_timestamp) / 3600.0
        
        if time_hours <= 0:
            return 0.0
        
        return distance_miles / time_hours
    
    def _calc_time_since_last_seen(
        self,
        last_timestamp: Optional[float],
        current_timestamp: float
    ) -> float:
        """Calculate seconds since last activity."""
        if last_timestamp is None:
            return float("inf")
        
        return current_timestamp - last_timestamp
    
    def _check_new_device(
        self,
        ja3_hash: Optional[str],
        known_hashes: set
    ) -> float:
        """
        Check if device is new (unknown JA3 hash).
        
        Returns 1.0 if new, 0.0 if known.
        """
        if ja3_hash is None:
            return 0.5  # Uncertain without JA3
        
        return 0.0 if ja3_hash in known_hashes else 1.0
    
    def _calc_transaction_magnitude(
        self,
        transaction_details: Dict[str, Any],
        avg_amount: float
    ) -> float:
        """
        Calculate transaction magnitude ratio.
        
        Returns current_amount / average_amount.
        Returns 1.0 if no average available.
        """
        current_amount = transaction_details.get("amount", 0.0)
        
        if avg_amount <= 0:
            return 1.0
        
        return current_amount / avg_amount
    
    def _get_ip_reputation_score(self, ip_address: str) -> float:
        """
        Get IP reputation score (0.0 = safe, 1.0 = risky).
        """
        geo = _lookup_ip(ip_address)
        reputation = geo.get("reputation", "unknown")
        return _REPUTATION_SCORES.get(reputation, 0.5)
    
    def _calc_time_anomaly(
        self,
        current_timestamp: float,
        usual_hours: List[int]
    ) -> float:
        """
        Calculate time anomaly score.
        
        Returns 1.0 if current hour is outside usual hours, 0.0 otherwise.
        Returns 0.0 if no usual hours data available.
        """
        if not usual_hours:
            return 0.0
        
        # Get current hour in 24-hour format
        current_hour = int(time.strftime("%H", time.localtime(current_timestamp)))
        
        return 0.0 if current_hour in usual_hours else 1.0
    
    def _check_policy_violation(
        self,
        role: str,
        resource_target: str
    ) -> float:
        """
        Check for policy violations based on role and resource.
        
        Returns 1.0 if violation detected, 0.0 otherwise.
        
        Example violation: intern accessing production resources.
        """
        role_lower = role.lower()
        resource_lower = resource_target.lower()
        
        # Interns cannot access production resources
        if role_lower == "intern" and "prod" in resource_lower:
            return 1.0
        
        # Viewers cannot access admin resources
        if role_lower == "viewer" and "admin" in resource_lower:
            return 1.0
        
        return 0.0
