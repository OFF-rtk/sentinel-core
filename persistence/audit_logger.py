"""
Sentinel Audit Logger

Fire-and-forget audit log writer that inserts structured
audit log entries into the Supabase `audit_logs` table
after every /evaluate call.

Schema:
    audit_logs (
        event_id TEXT PRIMARY KEY,
        payload  JSONB,
        created_at TIMESTAMPTZ DEFAULT now()
    )
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import geoip2.database
from supabase import create_client, Client

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Builds and inserts structured audit log payloads into Supabase.
    
    All writes are best-effort — errors are logged but never raised
    to avoid disrupting the evaluate pipeline.
    """

    ENGINE_VERSION = "v2.1.0"

    def __init__(self) -> None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            logger.warning("Supabase credentials missing — audit logging disabled")
            self._client: Optional[Client] = None
            return
        self._client = create_client(url, key)

        # GeoIP reader — same database used by the context processor
        try:
            self.geoip = geoip2.database.Reader("assets/GeoLite2-City.mmdb")
        except Exception as e:
            logger.warning(f"GeoIP database unavailable for audit logger: {e}")
            self.geoip = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        payload: Any,
        result: Any,
    ) -> None:
        """
        Build and insert an audit log entry.

        Args:
            payload:  The EvaluatePayload from the request.
            result:   The EvaluateResponse returned to the caller.
        """
        if self._client is None:
            return

        try:
            entry = self._build_entry(payload, result)
            self._client.table("audit_logs").insert({
                "event_id": entry["event_id"],
                "payload": entry,
            }).execute()
            logger.debug(f"Audit log inserted: {entry['event_id']}")
        except Exception as e:
            logger.error(f"Audit log insertion failed: {e}")

    # ------------------------------------------------------------------
    # GeoIP Lookup
    # ------------------------------------------------------------------

    def _resolve_ip(self, ip_address: str) -> Dict[str, Any]:
        """
        Resolve IP address to geo data using GeoLite2.
        Returns { country, city, asn, lat, lng } — all best-effort.
        """
        if self.geoip is None:
            return {"country": "unknown", "city": "unknown", "asn": "unknown", "lat": None, "lng": None}

        # Private/reserved IPs can't be resolved
        if ip_address.startswith((
            "10.", "172.16.", "172.17.", "172.18.", "172.19.",
            "172.20.", "172.21.", "172.22.", "172.23.",
            "172.24.", "172.25.", "172.26.", "172.27.",
            "172.28.", "172.29.", "172.30.", "172.31.",
            "192.168.", "127.", "0.", "::1", "fe80:",
        )):
            return {"country": "private", "city": "private", "asn": "private", "lat": None, "lng": None}

        try:
            response = self.geoip.city(ip_address)
            lat = response.location.latitude if response.location.latitude else None
            lng = response.location.longitude if response.location.longitude else None
            return {
                "country": response.country.iso_code or "unknown",
                "city": response.city.name or "unknown",
                "asn": f"AS{response.traits.autonomous_system_number}" if hasattr(response.traits, 'autonomous_system_number') and response.traits.autonomous_system_number else "unknown",
                "lat": lat,
                "lng": lng,
            }
        except Exception as e:
            logger.debug(f"GeoIP lookup failed for {ip_address}: {e}")
            return {"country": "unknown", "city": "unknown", "asn": "unknown", "lat": None, "lng": None}

    # ------------------------------------------------------------------
    # Payload Builder
    # ------------------------------------------------------------------

    def _build_entry(
        self,
        payload: Any,
        result: Any,
    ) -> Dict[str, Any]:
        """Assemble the full audit log payload."""

        now = datetime.now(timezone.utc)
        event_id = f"evt_{uuid.uuid4()}"
        correlation_id = (
            f"corr_{payload.eval_id}" if payload.eval_id
            else f"corr_{uuid.uuid4().hex[:12]}"
        )

        # Session age in seconds
        session_age = 0
        if payload.session_start_time:
            session_age = int(
                (now.timestamp() * 1000 - payload.session_start_time) / 1000
            )
            session_age = max(session_age, 0)

        # Geo lookup directly from IP
        geo = self._resolve_ip(payload.request_context.ip_address)

        # Client fingerprint (device_id + user_agent only, no ja3_hash)
        fingerprint = {
            "device_id": "unknown",
            "user_agent": payload.request_context.user_agent,
        }
        if payload.client_fingerprint:
            fingerprint["device_id"] = payload.client_fingerprint.device_id

        # Decision string
        decision_str = (
            result.decision.value
            if hasattr(result.decision, "value")
            else str(result.decision)
        )

        # Anomaly vectors — pull from the result if present
        anomaly_vectors: List[str] = []
        if hasattr(result, "anomaly_vectors") and result.anomaly_vectors:
            anomaly_vectors = result.anomaly_vectors

        return {
            # Metadata
            "event_id": event_id,
            "correlation_id": correlation_id,
            "timestamp": now.isoformat(),
            "environment": os.getenv("SENTINEL_ENV", "production"),

            # Actor
            "actor": {
                "user_id": payload.request_context.user_id,
                "role": payload.role,
                "session_id": payload.session_id,
                "session_age_seconds": session_age,
            },

            # Network
            "network_context": {
                "ip_address": payload.request_context.ip_address,
                "geo_location": geo,
                "client_fingerprint": fingerprint,
            },

            # Action
            "action_context": {
                "service": payload.business_context.service,
                "action_type": payload.business_context.action_type,
                "resource_target": payload.business_context.resource_target,
                "details": payload.business_context.transaction_details,
            },

            # Sentinel Analysis
            "sentinel_analysis": {
                "engine_version": self.ENGINE_VERSION,
                "risk_score": result.risk,
                "decision": decision_str,
                "anomaly_vectors": anomaly_vectors,
            },

            # Security Enforcement
            "security_enforcement": {
                "mfa_status": payload.mfa_status,
                "policy_applied": "POLICY_SENTINEL_ML",
            },
        }
