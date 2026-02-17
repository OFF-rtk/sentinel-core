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
from typing import Any, Dict, Optional

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        payload: Any,
        result: Any,
        nav_metrics: Optional[Dict] = None,
    ) -> None:
        """
        Build and insert an audit log entry.

        Args:
            payload:  The EvaluatePayload from the request.
            result:   The EvaluateResponse returned to the caller.
            nav_metrics: Optional navigator metrics dict from orchestrator.
        """
        if self._client is None:
            return

        try:
            entry = self._build_entry(payload, result, nav_metrics)
            self._client.table("audit_logs").insert({
                "event_id": entry["event_id"],
                "payload": entry,
            }).execute()
            logger.debug(f"Audit log inserted: {entry['event_id']}")
        except Exception as e:
            logger.error(f"Audit log insertion failed: {e}")

    # ------------------------------------------------------------------
    # Payload Builder
    # ------------------------------------------------------------------

    def _build_entry(
        self,
        payload: Any,
        result: Any,
        nav_metrics: Optional[Dict],
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

        # Geo / IP reputation from navigator metrics
        geo_data = {}
        ip_reputation = "unknown"
        if nav_metrics:
            geo_data = nav_metrics.get("current_geo_data", {}) or {}
            ip_reputation = nav_metrics.get("ip_reputation", "unknown") or "unknown"

        # Client fingerprint
        fingerprint = {}
        if payload.client_fingerprint:
            fingerprint = {
                "device_id": payload.client_fingerprint.device_id,
                "ja3_hash": payload.client_fingerprint.ja3_hash or "unknown",
                "user_agent_raw": payload.request_context.user_agent,
            }
        else:
            fingerprint = {
                "device_id": "unknown",
                "ja3_hash": "unknown",
                "user_agent_raw": payload.request_context.user_agent,
            }

        # Decision string
        decision_str = (
            result.decision.value
            if hasattr(result.decision, "value")
            else str(result.decision)
        )

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
                "ip_reputation": ip_reputation,
                "geo_location": {
                    "country": geo_data.get("country", "unknown"),
                    "city": geo_data.get("city", "unknown"),
                    "asn": geo_data.get("asn", "unknown"),
                },
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
                "anomaly_vectors": [],  # populated if tracked by orchestrator
            },

            # Security Enforcement
            "security_enforcement": {
                "mfa_status": payload.mfa_status,
                "policy_applied": "POLICY_SENTINEL_ML",
            },
        }
