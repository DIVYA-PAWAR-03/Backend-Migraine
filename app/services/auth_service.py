"""
Authentication helpers for lightweight token generation and validation.
"""

import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta
from typing import Dict, Optional

from ..config import settings


class AuthService:
    """Stateless auth token service using HMAC-signed payloads."""

    def _encode_payload(self, payload: Dict) -> str:
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

    def _decode_payload(self, value: str) -> Dict:
        padding = "=" * (-len(value) % 4)
        decoded = base64.urlsafe_b64decode((value + padding).encode("utf-8"))
        return json.loads(decoded.decode("utf-8"))

    def _sign(self, payload_b64: str) -> str:
        digest = hmac.new(
            settings.AUTH_SECRET.encode("utf-8"),
            payload_b64.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")

    def create_access_token(self, user: Dict) -> str:
        expires_at = datetime.utcnow() + timedelta(hours=settings.AUTH_TOKEN_EXPIRY_HOURS)
        payload = {
            "user_id": str(user.get("_id") or user.get("id")),
            "email": user.get("email"),
            "full_name": user.get("full_name", "Patient"),
            "exp": int(expires_at.timestamp()),
            "iat": int(datetime.utcnow().timestamp()),
        }
        payload_b64 = self._encode_payload(payload)
        signature = self._sign(payload_b64)
        return f"{payload_b64}.{signature}"

    def verify_access_token(self, token: str) -> Optional[Dict]:
        try:
            parts = token.split(".")
            if len(parts) != 2:
                return None

            payload_b64, provided_signature = parts
            expected_signature = self._sign(payload_b64)
            if not hmac.compare_digest(provided_signature, expected_signature):
                return None

            payload = self._decode_payload(payload_b64)
            if int(payload.get("exp", 0)) < int(datetime.utcnow().timestamp()):
                return None

            return payload
        except Exception:
            return None


# Singleton
auth_service = AuthService()
