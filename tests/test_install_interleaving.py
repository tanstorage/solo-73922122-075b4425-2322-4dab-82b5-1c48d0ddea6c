"""Install-interleaving / stale-request regression tests.

These pin the race fix on the real device ASGI app:

* a request that passed its version check, then paused, must not overwrite a
  newer version activated while it waited (the original downgrade bug);
* same version + different digest is rejected even when the requests
  interleave around the activation delay;
* same version + same digest stays idempotent when interleaved;
* every receipt matches what actually happened to that request;
* /status stays queryable while an install is paused in its delay.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib

import httpx
import pytest
import pytest_asyncio

from tests.conftest import _lifespan


def _payload(version: int, firmware: bytes) -> dict:
    return {
        "model": "gateway-gx100",
        "version": version,
        "sha256": hashlib.sha256(firmware).hexdigest(),
        "firmware_b64": base64.b64encode(firmware).decode(),
    }


@pytest_asyncio.fixture
async def device_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DEVICE_DB_PATH", str(tmp_path / "device.db"))
    monkeypatch.setenv("DEVICE_FACTORY_VERSION", "1")
    monkeypatch.setenv("DEVICE_ACTIVATION_DELAY", "0")
    import device_sim.app as device_app

    importlib.reload(device_app)

    async with _lifespan(device_app.app):
        transport = httpx.ASGITransport(app=device_app.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://device"
        ) as client:
            yield client, device_app


async def _set_delay(client, seconds: float, one_shot: bool) -> None:
    r = await client.post(
        "/admin/activation-delay",
        json={"seconds": seconds, "one_shot": one_shot},
    )
    assert r.status_code == 200
    assert r.json()["activation_delay_seconds"] == seconds


@pytest.mark.asyncio
async def test_late_stale_request_cannot_downgrade(device_client):
    """The exact reported bug: pause after check, retry, then v3, then resume."""
    client, _ = device_client
    await _set_delay(client, 1.0, one_shot=True)

    fw2 = b"firmware-v2"
    fw3 = b"firmware-v3"

    # First v2 request passes the version check and then pauses (one-shot).
    first = asyncio.create_task(client.post("/install", json=_payload(2, fw2)))
    await asyncio.sleep(0.3)

    # /status must stay available while an install is waiting.
    status = (await client.get("/status")).json()
    assert status["version"] == 1

    # API timeout -> resend v2 while the first request is still paused.
    resend = await client.post("/install", json=_payload(2, fw2))
    assert resend.status_code == 200
    assert resend.json()["status"] == "installed"
    assert (await client.get("/status")).json()["version"] == 2

    # v3 is installed on top of v2 before the stale request resumes.
    r3 = await client.post("/install", json=_payload(3, fw3))
    assert r3.json()["status"] == "installed"

    # The first (now stale) v2 request resumes: it must be refused, not applied.
    late = await first
    assert late.status_code == 409
    body = late.json()["detail"]
    assert body["status"] == "rejected"
    assert "stale request refused" in body["reason"]

    final = (await client.get("/status")).json()
    assert final["version"] == 3
    assert final["sha256"] == hashlib.sha256(fw3).hexdigest()

    # The journal proves the late request was classified against the fresh
    # state and never activated.
    journal = (await client.get("/admin/install-journal")).json()["entries"]
    events = [e["event"] for e in journal]
    assert "activation_delay" in events
    assert events.count("installed") == 2  # v2 retry, then v3
    assert "rejected_stale_after_wait" in events


@pytest.mark.asyncio
async def test_interleaved_same_version_different_digest_rejected(device_client):
    """Two same-version requests with different firmware interleave."""
    client, _ = device_client

    fw_a = b"firmware-v2-A"
    fw_b = b"firmware-v2-B-different"

    # A passes its check and pauses; B (same version, other digest) then runs.
    await _set_delay(client, 1.0, one_shot=True)
    first = asyncio.create_task(client.post("/install", json=_payload(2, fw_a)))
    await asyncio.sleep(0.3)

    # At decision point 1 the device is still v1, so B is also "allowed" and
    # runs without a delay (one-shot was consumed by A). B activates v2 first.
    rb = await client.post("/install", json=_payload(2, fw_b))
    assert rb.status_code == 200 and rb.json()["status"] == "installed"

    # A resumes: same version, different digest -> rejected, no overwrite.
    ra = await first
    assert ra.status_code == 409
    assert ra.json()["detail"]["status"] == "rejected"
    assert "different digest" in ra.json()["detail"]["reason"]

    final = (await client.get("/status")).json()
    assert final["version"] == 2
    assert final["sha256"] == hashlib.sha256(fw_b).hexdigest()

    journal = (await client.get("/admin/install-journal")).json()["entries"]
    assert any(e["event"] == "rejected_digest_conflict_after_wait" for e in journal)


@pytest.mark.asyncio
async def test_interleaved_same_version_same_digest_is_idempotent(device_client):
    """A delayed request for the exact (version, digest) already installed."""
    client, _ = device_client
    fw2 = b"firmware-v2-idempotent"

    await _set_delay(client, 1.0, one_shot=True)
    first = asyncio.create_task(client.post("/install", json=_payload(2, fw2)))
    await asyncio.sleep(0.3)

    # The retry installs v2 first.
    retry = await client.post("/install", json=_payload(2, fw2))
    assert retry.json()["status"] == "installed"

    # The delayed request resumes: identical target -> idempotent, no overwrite.
    late = await first
    assert late.status_code == 200
    assert late.json()["status"] == "already_installed"
    assert late.json()["sha256"] == hashlib.sha256(fw2).hexdigest()

    assert (await client.get("/status")).json()["version"] == 2
    journal = (await client.get("/admin/install-journal")).json()["entries"]
    assert any(e["event"] == "already_installed_after_wait" for e in journal)


@pytest.mark.asyncio
async def test_stale_receipt_reports_this_requests_actual_outcome(device_client):
    """Each request gets a receipt for its own result, never a borrowed one."""
    client, _ = device_client
    fw2 = b"firmware-v2-receipt"
    fw3 = b"firmware-v3-receipt"

    await _set_delay(client, 1.0, one_shot=True)
    first = asyncio.create_task(
        client.post("/install", json=_payload(2, fw2), headers={"X-Request-ID": "req-v2-late"})
    )
    await asyncio.sleep(0.3)

    retry = await client.post(
        "/install", json=_payload(2, fw2), headers={"X-Request-ID": "req-v2-retry"}
    )
    assert retry.json()["request_id"] == "req-v2-retry"

    await client.post("/install", json=_payload(3, fw3))

    late = await first
    # The late v2 request's own receipt is a rejection, even though the retry
    # of the same package earlier succeeded.
    assert late.status_code == 409
    assert late.json()["detail"]["status"] == "rejected"


@pytest.mark.asyncio
async def test_activation_delay_can_be_disarmed_and_reset_clears_it(device_client):
    client, device_app = device_client
    await _set_delay(client, 0.5, one_shot=False)

    # Persistent delay applies to every install.
    r = await client.post("/install", json=_payload(2, b"fw-two"))
    assert r.status_code == 200

    # Disarm via zero seconds; subsequent installs are immediate.
    await _set_delay(client, 0.0, one_shot=False)
    loop = asyncio.get_running_loop()
    start = loop.time()
    r = await client.post("/install", json=_payload(3, b"fw-three"))
    elapsed = loop.time() - start
    assert r.status_code == 200 and elapsed < 0.2

    # Reset restores factory state and clears fault injection.
    r = await client.post("/admin/reset")
    assert r.json()["version"] == 1
    assert (await client.get("/admin/install-journal")).json()["count"] == 0
