"""Polite Stack Exchange API v2.3 client.

Implements the documented etiquette: sequential requests with a minimum interval, honouring the
``backoff`` field, retrying transient failures with exponential backoff, and stopping cleanly
when the daily quota is exhausted.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests

log = logging.getLogger(__name__)

RETRYABLE_HTTP = {429, 500, 502, 503, 504}
THROTTLE_SECONDS_RE = re.compile(r"available in (\d+) seconds")


class ApiError(RuntimeError):
    def __init__(self, message: str, error_id: int | None = None, error_name: str | None = None):
        super().__init__(message)
        self.error_id = error_id
        self.error_name = error_name


class QuotaExhausted(ApiError):
    """Raised when the daily quota is used up or the API asks us to wait too long."""


@dataclass
class ApiResponse:
    method: str
    params: dict[str, Any]  # request params with the API key removed
    payload: dict[str, Any]

    @property
    def items(self) -> list[dict[str, Any]]:
        return self.payload.get("items", [])

    @property
    def has_more(self) -> bool:
        return bool(self.payload.get("has_more", False))

    @property
    def quota_remaining(self) -> int | None:
        return self.payload.get("quota_remaining")

    @property
    def backoff(self) -> int | None:
        return self.payload.get("backoff")


class StackExchangeClient:
    def __init__(
        self,
        site: str,
        api_filter: str,
        key: str | None = None,
        api_base: str = "https://api.stackexchange.com/2.3",
        min_interval: float = 0.5,
        max_retries: int = 5,
        timeout: float = 30.0,
        max_wait_seconds: float = 600.0,
        session: requests.Session | Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.site = site
        self.api_filter = api_filter
        self.key = key
        self.api_base = api_base.rstrip("/")
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_wait_seconds = max_wait_seconds
        self.session = session or requests.Session()
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None
        self._not_before: dict[str, float] = {}  # per-method backoff deadlines
        self.requests_made = 0
        self.quota_remaining: int | None = None

    def _wait_turn(self, method: str) -> None:
        now = self._clock()
        wait = 0.0
        if self._last_request_at is not None:
            wait = max(wait, self._last_request_at + self.min_interval - now)
        wait = max(wait, self._not_before.get(method, 0.0) - now)
        if wait > 0:
            self._sleep(wait)

    def get(self, method: str, params: dict[str, Any] | None = None) -> ApiResponse:
        public_params = {"site": self.site, "filter": self.api_filter, **(params or {})}
        request_params = dict(public_params)
        if self.key:
            request_params["key"] = self.key
        url = f"{self.api_base}/{method.lstrip('/')}"

        for attempt in range(self.max_retries + 1):
            self._wait_turn(method)
            self._last_request_at = self._clock()
            try:
                resp = self.session.get(url, params=request_params, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                self._retry_or_raise(attempt, f"network error: {exc}")
                continue
            self.requests_made += 1

            try:
                payload = resp.json()
            except ValueError:
                payload = {}

            if "error_id" in payload:
                self._handle_api_error(payload, attempt)
                continue
            if resp.status_code in RETRYABLE_HTTP:
                self._retry_or_raise(attempt, f"HTTP {resp.status_code}")
                continue
            if resp.status_code >= 400:
                raise ApiError(f"HTTP {resp.status_code} for {method}")

            self.quota_remaining = payload.get("quota_remaining", self.quota_remaining)
            if payload.get("backoff"):
                log.info("API requested backoff of %ss for %s", payload["backoff"], method)
                self._not_before[method] = self._clock() + float(payload["backoff"])
            return ApiResponse(method=method, params=public_params, payload=payload)

        raise ApiError(f"exhausted retries for {method}")  # pragma: no cover - loop always returns

    def _retry_or_raise(self, attempt: int, reason: str, wait: float | None = None) -> None:
        if attempt >= self.max_retries:
            raise ApiError(f"giving up after {attempt + 1} attempts: {reason}")
        delay = wait if wait is not None else min(2.0**attempt, 60.0)
        log.warning("retrying in %.1fs (%s)", delay, reason)
        self._sleep(delay)

    def _handle_api_error(self, payload: dict[str, Any], attempt: int) -> None:
        error_id = payload.get("error_id")
        name = payload.get("error_name")
        message = payload.get("error_message", "")
        if name == "throttle_violation" or error_id == 502:
            match = THROTTLE_SECONDS_RE.search(message)
            seconds = float(match.group(1)) if match else None
            if seconds is not None and seconds > self.max_wait_seconds:
                raise QuotaExhausted(message, error_id, name)
            self._retry_or_raise(attempt, f"throttled: {message}", wait=seconds)
            return
        if error_id in (500, 503) or name in ("internal_error", "temporarily_unavailable"):
            self._retry_or_raise(attempt, f"{name}: {message}")
            return
        raise ApiError(f"{name}: {message}", error_id, name)
