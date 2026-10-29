import pytest
import requests

from ml.data.stackexchange_client import ApiError, QuotaExhausted, StackExchangeClient
from tests.conftest import FakeClock, FakeResponse, FakeSession


def make_client(session, clock=None, **kw):
    clock = clock or FakeClock()
    return StackExchangeClient(
        site="apple",
        api_filter="F",
        key=kw.pop("key", None),
        session=session,
        sleep=clock.sleep,
        clock=clock,
        **kw,
    ), clock


def ok(items=None, **extra):
    return FakeResponse({"items": items or [], "has_more": False, "quota_remaining": 100, **extra})


def test_passes_site_filter_and_strips_key_from_recorded_params():
    session = FakeSession([ok([{"question_id": 1}])])
    client, _ = make_client(session, key="SECRET")
    resp = client.get("questions", {"page": 1})
    assert session.calls[0]["params"]["key"] == "SECRET"
    assert session.calls[0]["params"]["site"] == "apple"
    assert "key" not in resp.params
    assert resp.items == [{"question_id": 1}]
    assert client.quota_remaining == 100


def test_retries_transient_http_and_network_errors_with_exponential_backoff():
    session = FakeSession(
        [
            FakeResponse(None, 503),
            requests.ConnectionError("boom"),
            ok(),
        ]
    )
    client, clock = make_client(session, min_interval=0)
    client.get("questions")
    assert len(session.calls) == 3
    assert clock.sleeps == [1.0, 2.0]


def test_gives_up_after_max_retries():
    session = FakeSession([FakeResponse(None, 500)] * 3)
    client, _ = make_client(session, max_retries=2, min_interval=0)
    with pytest.raises(ApiError):
        client.get("questions")


def test_honours_backoff_field_before_next_request_to_same_method():
    session = FakeSession([ok(backoff=10), ok()])
    client, clock = make_client(session, min_interval=0)
    client.get("questions")
    client.get("questions")
    assert clock.sleeps == [10.0]


def test_min_interval_spacing():
    session = FakeSession([ok(), ok()])
    client, clock = make_client(session, min_interval=0.5)
    client.get("questions")
    client.get("questions")
    assert clock.sleeps == [0.5]


def test_short_throttle_is_waited_out_long_throttle_raises_quota_exhausted():
    short = FakeResponse(
        {
            "error_id": 502,
            "error_name": "throttle_violation",
            "error_message": "too many requests from this IP, more requests available in 5 seconds",
        },
        400,
    )
    long = FakeResponse(
        {
            "error_id": 502,
            "error_name": "throttle_violation",
            "error_message": "too many requests from this IP, more requests available in 80000 seconds",
        },
        400,
    )
    client, clock = make_client(FakeSession([short, ok()]), min_interval=0)
    client.get("questions")
    assert clock.sleeps == [5.0]

    client, _ = make_client(FakeSession([long]), min_interval=0)
    with pytest.raises(QuotaExhausted):
        client.get("questions")


def test_non_retryable_api_error_raises():
    bad = FakeResponse({"error_id": 400, "error_name": "bad_parameter", "error_message": "site"}, 400)
    client, _ = make_client(FakeSession([bad]))
    with pytest.raises(ApiError, match="bad_parameter"):
        client.get("questions")
