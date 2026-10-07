"""Thin HTTP client for the Synthetic Scheduling API.

Responsibilities:
- Call the provided scheduling API
- Send validated HTTP requests
- Parse JSON responses
- Surface API/network failures consistently

This module does NOT:
- detect user intent
- manage conversation state
- decide whether a patient is verified
- decide whether booking is allowed
- enforce confirmation workflow

Those responsibilities belong in the orchestrator.
"""

import json
import logging
import os
import time
from typing import Any, Mapping, Optional
from urllib.parse import quote

import requests

logger = logging.getLogger("scheduling.api")


DEFAULT_BASE_URL = "http://localhost:4010"
DEFAULT_TIMEOUT_SECONDS = 10


class SchedulingAPIError(Exception):
    """Raised when the scheduling API returns an error or cannot be reached."""

    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self.body = body

        code = body.get("code") if isinstance(body, dict) else None
        message = body.get("message") if isinstance(body, dict) else None

        self.code = code
        self.message = message if isinstance(message, str) else None

        detail = self.message or self.code or str(status_code)

        super().__init__(
            f"Scheduling API error {status_code}: {detail}"
        )


class SchedulingClient:
    """Client for the supplied synthetic scheduling API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        session: Optional[requests.Session] = None,
    ):
        configured = (
            base_url
            or os.environ.get("SCHEDULING_API_BASE_URL")
            or DEFAULT_BASE_URL
        )

        self.base_url = configured.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def lookup_providers(
        self,
        specialty: Optional[str] = None,
        location: Optional[str] = None,
        mock_scenario: Optional[str] = None,
    ) -> dict:
        """GET /providers."""

        return self._request(
            "GET",
            "/providers",
            params=_omit_none(
                {
                    "specialty": specialty,
                    "location": location,
                }
            ),
            mock_scenario=mock_scenario,
        )

    def search_patients(
        self,
        phone: str,
        dob: str,
        mock_scenario: Optional[str] = None,
    ) -> dict:
        """GET /patients/search."""

        return self._request(
            "GET",
            "/patients/search",
            params={
                "phone": phone,
                "dob": dob,
            },
            mock_scenario=mock_scenario,
        )

    def lookup_availability(
        self,
        patient_id: str,
        specialty: str,
        location: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        mock_scenario: Optional[str] = None,
    ) -> dict:
        """GET /availability."""

        return self._request(
            "GET",
            "/availability",
            params=_omit_none(
                {
                    "patientId": patient_id,
                    "specialty": specialty,
                    "location": location,
                    "startDate": start_date,
                    "endDate": end_date,
                }
            ),
            mock_scenario=mock_scenario,
        )

    def book_appointment(
        self,
        patient_id: str,
        slot_id: str,
        confirmed: bool,
        mock_scenario: Optional[str] = None,
    ) -> dict:
        """POST /appointments."""

        return self._request(
            "POST",
            "/appointments",
            json={
                "patientId": patient_id,
                "slotId": slot_id,
                "confirmed": confirmed,
            },
            mock_scenario=mock_scenario,
        )

    def create_handoff(
        self,
        reason: str,
        summary: str,
        patient_id: Optional[str] = None,
        mock_scenario: Optional[str] = None,
    ) -> dict:
        """POST /handoffs."""

        return self._request(
            "POST",
            "/handoffs",
            json=_omit_none(
                {
                    "patientId": patient_id,
                    "reason": reason,
                    "summary": summary,
                }
            ),
            mock_scenario=mock_scenario,
        )

    def lookup_appointments(
        self,
        patient_id: str,
        mock_scenario: Optional[str] = None,
    ) -> dict:
        """GET /patients/{patientId}/appointments."""

        encoded_patient_id = quote(patient_id, safe="")

        return self._request(
            "GET",
            f"/patients/{encoded_patient_id}/appointments",
            mock_scenario=mock_scenario,
        )

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
        json: Optional[Mapping[str, Any]] = None,
        mock_scenario: Optional[str] = None,
    ) -> dict:
        """Send one request to the scheduling API."""

        headers = {
            "Accept": "application/json",
        }

        # Used only for deterministic failure testing supplied by the assignment.
        if mock_scenario:
            headers["X-Mock-Scenario"] = mock_scenario

        started = time.perf_counter()
        status = 0
        outcome = "error"
        try:
            try:
                response = self.session.request(
                    method=method,
                    url=self.base_url + path,
                    params=params,
                    json=json,
                    timeout=self.timeout,
                    headers=headers,
                )
            except requests.Timeout as exc:
                outcome = "timeout"
                raise SchedulingAPIError(
                    0,
                    {
                        "code": "timeout",
                        "message": "Scheduling API request timed out.",
                    },
                ) from exc
            except requests.RequestException as exc:
                outcome = "network_error"
                raise SchedulingAPIError(
                    0,
                    {
                        "code": "network_error",
                        "message": "Unable to reach the scheduling API.",
                    },
                ) from exc

            status = response.status_code
            body = _parse_json(response)

            # Treat any HTTP 2xx response as success.
            if not 200 <= response.status_code < 300:
                outcome = _error_code(body) or "http_error"
                raise SchedulingAPIError(
                    response.status_code,
                    body,
                )

            if not isinstance(body, dict):
                outcome = "invalid_response"
                raise SchedulingAPIError(
                    response.status_code,
                    {
                        "code": "invalid_response",
                        "message": "Scheduling API returned an unexpected response.",
                    },
                )

            outcome = "ok"
            return body
        finally:
            _log_api(method, path, status, outcome, started)


def _log_api(method, path, status, outcome, started):
    """Log the call shape and result, never the request or patient payload."""
    logger.info(json.dumps({
        "api_operation": "{} {}".format(method, _public_path(path)),
        "status": status,
        "outcome": outcome,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }, separators=(",", ":")))


def _public_path(path):
    parts = [part for part in path.split("/") if part]
    if len(parts) == 3 and parts[0] == "patients" and parts[2] == "appointments":
        return "/patients/{patientId}/appointments"
    return path


def _error_code(body):
    if isinstance(body, dict) and isinstance(body.get("code"), str):
        return body["code"]
    return None


def _configure_logging():
    parent = logging.getLogger("scheduling")
    if parent.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    parent.addHandler(handler)
    parent.setLevel(logging.INFO)
    parent.propagate = False


_configure_logging()


def _omit_none(fields: Mapping[str, Any]) -> dict:
    """Remove fields whose value is None."""

    return {
        key: value
        for key, value in fields.items()
        if value is not None
    }


def _parse_json(response: requests.Response) -> Any:
    """Safely parse an HTTP response body."""

    if not response.content:
        return {}

    try:
        return response.json()

    except ValueError:
        return {
            "message": response.text
        }


_client: Optional[SchedulingClient] = None


def get_client() -> SchedulingClient:
    """Return a shared SchedulingClient instance."""

    global _client

    if _client is None:
        _client = SchedulingClient()

    return _client


# Convenience functions used by the orchestrator.


def lookup_providers(
    specialty: Optional[str] = None,
    location: Optional[str] = None,
    mock_scenario: Optional[str] = None,
) -> dict:
    return get_client().lookup_providers(
        specialty=specialty,
        location=location,
        mock_scenario=mock_scenario,
    )


def search_patients(
    phone: str,
    dob: str,
    mock_scenario: Optional[str] = None,
) -> dict:
    return get_client().search_patients(
        phone=phone,
        dob=dob,
        mock_scenario=mock_scenario,
    )


def lookup_availability(
    patient_id: str,
    specialty: str,
    location: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    mock_scenario: Optional[str] = None,
) -> dict:
    return get_client().lookup_availability(
        patient_id=patient_id,
        specialty=specialty,
        location=location,
        start_date=start_date,
        end_date=end_date,
        mock_scenario=mock_scenario,
    )


def book_appointment(
    patient_id: str,
    slot_id: str,
    confirmed: bool,
    mock_scenario: Optional[str] = None,
) -> dict:
    return get_client().book_appointment(
        patient_id=patient_id,
        slot_id=slot_id,
        confirmed=confirmed,
        mock_scenario=mock_scenario,
    )


def create_handoff(
    reason: str,
    summary: str,
    patient_id: Optional[str] = None,
    mock_scenario: Optional[str] = None,
) -> dict:
    return get_client().create_handoff(
        reason=reason,
        summary=summary,
        patient_id=patient_id,
        mock_scenario=mock_scenario,
    )


def lookup_appointments(
    patient_id: str,
    mock_scenario: Optional[str] = None,
) -> dict:
    return get_client().lookup_appointments(
        patient_id=patient_id,
        mock_scenario=mock_scenario,
    )