"""
Sentinel State Repository

Data Access Object for User Context and Session State.
Uses Redis for persistence with proper key schemas and atomic operations.

Key Schemas:
    PROFILE:{user_id}           # Redis HASH (home_country, etc.)
    PROFILE:{user_id}:devices   # Redis SET (device_ids)
    SESSION:{user_id}           # Redis STRING (JSON with TTL)
"""

import json
import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

from redis.exceptions import RedisError
from supabase import create_client, Client
from .connection import get_redis_client


logger = logging.getLogger(__name__)


class SentinelStateRepository:
    """
    Data Access Object for User Context and Session State.
    
    Uses Redis Pipelines for network efficiency and atomic operations.
    Implements fail-open pattern for resilience.
    """
    
    # Maximum number of known devices per user (prevents unbounded growth)
    MAX_KNOWN_DEVICES: int = 20
    
    # Session TTL in seconds (24 hours)
    SESSION_TTL: int = 86400
    
    def __init__(self) -> None:
        """Initialize repository with Redis and Supabase clients."""
        self.client = get_redis_client()
        
        # Supabase for persistent user_context (TOFU)
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if url and key:
            self.supabase: Optional[Client] = create_client(url, key)
        else:
            logger.warning("Supabase credentials not configured, TOFU persistence disabled")
            self.supabase = None
    
    # -------------------------------------------------------------------------
    # Trusted Context (TOFU — Read-Through / Write-Behind)
    # -------------------------------------------------------------------------
    
    def get_trusted_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Read-through: Redis first, then Supabase fallback.
        
        Returns None if no history exists (signals TOFU condition).
        Returns dict with known_devices, last_ip, last_geo_data if found.
        """
        # 1. Try Redis first
        devices_key = self._devices_key(user_id)
        try:
            devices = self.client.smembers(devices_key)
            if devices:
                device_list = [
                    d.decode("utf-8") if isinstance(d, bytes) else d
                    for d in devices
                ]
                return {"known_devices": device_list, "source": "redis"}
        except RedisError as e:
            logger.warning(f"Redis read failed for trusted context {user_id}: {e}")
        
        # 2. Fallback to Supabase
        if self.supabase is None:
            return None
        
        try:
            response = self.supabase.table("user_context").select(
                "known_devices, last_ip, last_geo_data"
            ).eq("user_id", user_id).execute()
            
            if not response.data:
                return None  # No history → TOFU condition
            
            row = response.data[0]
            known_devices = row.get("known_devices") or []
            
            if not known_devices:
                return None  # Empty devices array → still TOFU
            
            # 3. Backfill Redis (cache warming)
            try:
                if known_devices:
                    self.client.sadd(devices_key, *known_devices)
                    logger.info(f"Cache warmed {len(known_devices)} devices for {user_id}")
            except RedisError as e:
                logger.warning(f"Redis backfill failed for {user_id}: {e}")
            
            return {
                "known_devices": known_devices,
                "last_ip": row.get("last_ip"),
                "last_geo_data": row.get("last_geo_data"),
                "source": "supabase",
            }
            
        except Exception as e:
            logger.error(f"Supabase read failed for trusted context {user_id}: {e}")
            return None
    
    def save_trusted_context(
        self,
        user_id: str,
        device_id: str,
        ip: str,
        geo: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Write-behind: persist allowed environment to Supabase + Redis.
        
        Called after an ALLOW decision to persist the trusted context.
        """
        # 1. Supabase upsert
        if self.supabase is not None:
            try:
                # Read current row to append device
                response = self.supabase.table("user_context").select(
                    "known_devices"
                ).eq("user_id", user_id).execute()
                
                if response.data:
                    existing_devices = response.data[0].get("known_devices") or []
                    if device_id not in existing_devices:
                        existing_devices.append(device_id)
                    
                    self.supabase.table("user_context").update({
                        "known_devices": existing_devices,
                        "last_ip": ip,
                        "last_geo_data": geo or {},
                        "updated_at": "now()",
                    }).eq("user_id", user_id).execute()
                else:
                    self.supabase.table("user_context").insert({
                        "user_id": user_id,
                        "known_devices": [device_id] if device_id else [],
                        "last_ip": ip,
                        "last_geo_data": geo or {},
                    }).execute()
                
                logger.info(f"Persisted trusted context for {user_id}")
                
            except Exception as e:
                logger.error(f"Supabase write failed for trusted context {user_id}: {e}")
        
        # 2. Redis write (immediate cache update)
        if device_id:
            devices_key = self._devices_key(user_id)
            try:
                self.client.sadd(devices_key, device_id)
                self._cap_known_devices(user_id)
            except RedisError as e:
                logger.warning(f"Redis device write failed for {user_id}: {e}")
    
    def _profile_key(self, user_id: str) -> str:
        """Get profile hash key."""
        return f"PROFILE:{user_id}"
    
    def _devices_key(self, user_id: str) -> str:
        """Get devices set key."""
        return f"PROFILE:{user_id}:devices"
    
    def _session_key(self, user_id: str) -> str:
        """Get session string key."""
        return f"SESSION:{user_id}"
    
    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieves full user context (Profile + Devices + Session) in a single round-trip.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Dictionary with:
                - known_device_hashes: List[str] - Known device IDs
                - last_geo_coords: Tuple[float, float] | None - Last location
                - last_seen_timestamp: float | None - Last activity time
                - active_session_count: int - Active session count
        """
        profile_key = self._profile_key(user_id)
        devices_key = self._devices_key(user_id)
        session_key = self._session_key(user_id)
        
        try:
            # Pipeline: Batch commands to reduce latency (single RTT)
            pipe = self.client.pipeline()
            pipe.hgetall(profile_key)          # Profile hash
            pipe.smembers(devices_key)         # Device set
            pipe.get(session_key)              # Session JSON
            results = pipe.execute()
            
            profile_data: Dict[bytes, bytes] = results[0]
            device_set: set = results[1]
            session_json: Optional[bytes] = results[2]
            
            # Build history object with defaults
            history: Dict[str, Any] = {
                "known_device_hashes": [],
                "last_geo_coords": None,
                "last_seen_timestamp": None,
                "active_session_count": 0,
            }
            
            # Parse Profile (Static Data)
            if profile_data:
                # Profile fields can be accessed here if needed
                # Currently only home_country is in spec, not used by context processor
                pass
            
            # Parse Device Set
            if device_set:
                # Convert bytes to strings
                history["known_device_hashes"] = [
                    d.decode("utf-8") if isinstance(d, bytes) else d
                    for d in device_set
                ]
            
            # Parse Session (Dynamic Data) with JSON safety
            if session_json:
                try:
                    session_str = (
                        session_json.decode("utf-8")
                        if isinstance(session_json, bytes)
                        else session_json
                    )
                    session = json.loads(session_str)
                    
                    # Extract last_coords as tuple
                    last_coords = session.get("last_coords")
                    if last_coords and isinstance(last_coords, list) and len(last_coords) == 2:
                        history["last_geo_coords"] = tuple(last_coords)
                    
                    history["last_seen_timestamp"] = session.get("last_seen_timestamp")
                    history["active_session_count"] = session.get("active_session_count", 0)
                    
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    # JSON corruption → fail open with defaults
                    logger.warning(f"Session JSON corrupted for user {user_id}: {e}")
            
            return history
            
        except RedisError as e:
            logger.error(f"Redis read failed for user {user_id}: {e}")
            # Fail Open: Return empty history rather than crashing request
            return {
                "known_device_hashes": [],
                "last_geo_coords": None,
                "last_seen_timestamp": None,
                "active_session_count": 0,
            }
    
    def update_user_state(self, user_id: str, updates: Dict[str, Any]) -> None:
        """
        Updates session state and device profile atomically.
        
        Args:
            user_id: Unique user identifier
            updates: Dictionary with optional keys:
                - device_id: str - Device ID to add to known devices
                - coords: Tuple[float, float] - Current coordinates
                - ip: str - Current IP address
                - active_session_count: int - Current session count
        
        CRITICAL RULES:
            1. Device identity uses device_id (NOT ja3_hash)
            2. Devices stored in Redis SET (atomic SADD)
            3. Known devices capped at MAX_KNOWN_DEVICES
            4. Session TTL always refreshed on update
        """
        devices_key = self._devices_key(user_id)
        session_key = self._session_key(user_id)
        
        try:
            pipe = self.client.pipeline()
            
            # 1. Add Device (if provided)
            device_id = updates.get("device_id")
            if device_id:
                # Atomic device storage using SET
                pipe.sadd(devices_key, device_id)
            
            # 2. Update Session (Fast-moving data)
            coords = updates.get("coords")
            if coords:
                session_payload = {
                    "last_ip": updates.get("ip"),
                    "last_coords": list(coords) if coords else None,
                    "last_seen_timestamp": time.time(),
                    "active_session_count": updates.get("active_session_count", 1),
                }
                # Serialize and Set with Expiry (always refresh TTL)
                pipe.setex(session_key, self.SESSION_TTL, json.dumps(session_payload))
            
            # Execute pipeline
            pipe.execute()
            
            # 3. Cap Known Devices (MANDATORY) - Done after pipeline for atomicity
            if device_id:
                self._cap_known_devices(user_id)
            
        except RedisError as e:
            logger.error(f"Redis write failed for user {user_id}: {e}")
            # Write failures are logged but don't crash the request (fail-open)
    
    def _cap_known_devices(self, user_id: str) -> None:
        """
        Ensure known devices doesn't exceed MAX_KNOWN_DEVICES.
        
        Removes oldest device if count exceeds limit.
        Uses SPOP for removal (random, but deterministic enough).
        """
        devices_key = self._devices_key(user_id)
        
        try:
            # Check current count
            count = self.client.scard(devices_key)
            
            # Remove excess devices
            while count > self.MAX_KNOWN_DEVICES:
                # SPOP removes and returns a random member
                self.client.spop(devices_key)
                count -= 1
                
        except RedisError as e:
            logger.warning(f"Failed to cap devices for user {user_id}: {e}")
    
    def refresh_session_ttl(self, user_id: str) -> None:
        """
        Refresh session TTL without updating data.
        
        Useful for keeping sessions alive during active use.
        """
        session_key = self._session_key(user_id)
        
        try:
            self.client.expire(session_key, self.SESSION_TTL)
        except RedisError as e:
            logger.warning(f"Failed to refresh TTL for user {user_id}: {e}")
    
    def get_known_devices(self, user_id: str) -> List[str]:
        """
        Get list of known device IDs for a user.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            List of device ID strings
        """
        devices_key = self._devices_key(user_id)
        
        try:
            devices = self.client.smembers(devices_key)
            return [
                d.decode("utf-8") if isinstance(d, bytes) else d
                for d in devices
            ]
        except RedisError as e:
            logger.error(f"Failed to get devices for user {user_id}: {e}")
            return []
    
    def add_known_device(self, user_id: str, device_id: str) -> bool:
        """
        Add a device to the known devices set.
        
        Args:
            user_id: Unique user identifier
            device_id: Device identifier to add
            
        Returns:
            True if device was added, False if already existed
        """
        devices_key = self._devices_key(user_id)
        
        try:
            result = self.client.sadd(devices_key, device_id)
            self._cap_known_devices(user_id)
            return result > 0
        except RedisError as e:
            logger.error(f"Failed to add device for user {user_id}: {e}")
            return False
    
    def set_home_country(self, user_id: str, country: str) -> None:
        """
        Set user's home country in profile.
        
        Args:
            user_id: Unique user identifier
            country: ISO country code
        """
        profile_key = self._profile_key(user_id)
        
        try:
            self.client.hset(profile_key, "home_country", country)
        except RedisError as e:
            logger.error(f"Failed to set home country for user {user_id}: {e}")
    
    def get_home_country(self, user_id: str) -> Optional[str]:
        """
        Get user's home country from profile.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            ISO country code or None
        """
        profile_key = self._profile_key(user_id)
        
        try:
            result = self.client.hget(profile_key, "home_country")
            if result:
                return result.decode("utf-8") if isinstance(result, bytes) else result
            return None
        except RedisError as e:
            logger.error(f"Failed to get home country for user {user_id}: {e}")
            return None