"""End-to-end install race over real HTTP sockets.

Unlike the in-process ASGI transport (which does not enforce httpx timeouts),
these tests run the real device simulator and the real installation API on
real uvicorn servers and drive them with real *asynchronous* httpx clients, so
a short read timeout genuinely fires and the API's
timeout -> VERIFYING -> retry path is exercised together with the device's
stale-request protection. All client calls are async on purpose: the two
servers share this event loop, so a blocking call would freeze them too.
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib
import socket

import httpx
import pytest
import pytest_asyncio
import uvicorn

from app.config import Settings
from tests.conftest import make_package


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _serve(app, port: int) -> tuple[uvicorn.Server, asyncio.Task]:
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.05)
    return server, task


async def _wait_ready(client: httpx.AsyncClient, url: str) -> None:
    for _ in range(100):
        with contextlib.suppress(httpx.HTTPError):
            if (await client.get(url)).status_code == 200:
                return
        await asyncio.sleep(0.05)
    raise AssertionError(f"server at {url} never became ready")


@pytest_asyncio.fixture
async def live_stack(tmp_path, monkeypatch, key_pair):
    """Real API + real device, each on its own uvicorn server over sockets."""
    priv_path, pub_path = key_pair
    device_db = tmp_path / "device.db"
    api_db = tmp_path / "api.db"

    monkeypatch.setenv("DEVICE_DB_PATH", str(device_db))
    monkeypatch.setenv("DEVICE_FACTORY_VERSION", "1")
    monkeypatch.setenv("DEVICE_ACTIVATION_DELAY", "0")
    import device_sim.app as device_app

    importlib.reload(device_app)

    device_port = _free_port()
    api_port = _free_port()
    device_url = f"http://127.0.0.1:{device_port}"
    api_url = f"http://127.0.0.1:{api_port}"

    settings = Settings(
        database_path=api_db,
        trusted_public_key_path=pub_path,
        device_base_url=device_url,
        device_model="gateway-gx100",
        max_firmware_bytes=8 * 1024 * 1024,
        device_timeout_seconds=0.6,  # shorter than the activation delay
        reconcile_interval_seconds=3600.0,
    )

    from app.main import create_app

    api_app = create_app(settings, start_reconcile_loop=False)

    dev_server, dev_task = await _serve(device_app.app, device_port)
    api_server, api_task = await _serve(api_app, api_port)
    async with httpx.AsyncClient(timeout=5.0) as client:
        await _wait_ready(client, f"{device_url}/status")
        await _wait_ready(client, f"{api_url}/health")

    try:
        yield {
            "api": api_url,
            "device": device_url,
            "priv": priv_path,
            "settings": settings,
        }
    finally:
        dev_server.should_exit = True
        api_server.should_exit = True
        await asyncio.gather(dev_task, api_task, return_exceptions=True)


async def _submit(client: httpx.AsyncClient, base_url: str, pkg) -> int:
    firmware, manifest_bytes, sig, _ = pkg
    r = await client.post(
        f"{base_url}/installations",
        files={
            "firmware": ("f.bin", firmware, "application/octet-stream"),
            "manifest": ("m.json", manifest_bytes, "application/json"),
        },
        data={"signature": sig},
    )
    assert r.status_code == 202, r.text
    return r.json()["task_id"]


async def _get_task(client: httpx.AsyncClient, base_url: str, task_id: int) -> dict:
    return (await client.get(f"{base_url}/installations/{task_id}")).json()


async def _wait_state(
    client: httpx.AsyncClient, base_url: str, task_id: int, expected: str
) -> dict:
    for _ in range(100):
        view = await _get_task(client, base_url, task_id)
        if view["state"] == expected:
            return view
        await asyncio.sleep(0.1)
    raise AssertionError(f"task {task_id} never reached {expected}: {view}")


async def _wait_terminal(
    client: httpx.AsyncClient, base_url: str, task_id: int
) -> dict:
    for _ in range(100):
        view = await _get_task(client, base_url, task_id)
        if view["state"] in ("SUCCESS", "REJECTED"):
            return view
        await asyncio.sleep(0.1)
    raise AssertionError(f"task {task_id} stuck: {view}")


@pytest.mark.asyncio
async def test_timeout_retry_then_v3_late_request_keeps_v3(live_stack):
    api = live_stack["api"]
    device = live_stack["device"]
    priv = live_stack["priv"]

    async with httpx.AsyncClient(timeout=8.0) as client:
        # Arm the device: the FIRST request that passes its version check
        # pauses after the check (longer than the API timeout), then resumes.
        r = await client.post(
            f"{device}/admin/activation-delay",
            json={"seconds": 4.0, "one_shot": True},
        )
        assert r.status_code == 200

        # --- 1) first v2 install: API times out while the device holds it ---
        pkg2 = make_package(priv, version=2, size=256)
        fw2, mb2, sig2, sha2 = pkg2
        t2 = await _submit(client, api, pkg2)

        # The task lands in VERIFYING (no receipt) and the slot stays locked.
        view = await _wait_state(client, api, t2, "VERIFYING")
        assert "pending verification" in (view["device_response"] or "")

        # A newer v3 upload is refused while verification is open.
        fw3, mb3, sig3, sha3 = make_package(priv, version=3, size=256)
        conflict = await client.post(
            f"{api}/installations",
            files={
                "firmware": ("f.bin", fw3, "application/octet-stream"),
                "manifest": ("m.json", mb3, "application/json"),
            },
            data={"signature": sig3},
        )
        assert conflict.status_code == 409
        assert conflict.json()["detail"]["open_task_id"] == t2

        # Status queries remain available while the device request is paused.
        assert (await client.get(f"{api}/installations/{t2}")).status_code == 200
        assert (await client.get(f"{device}/status")).status_code == 200

        # Keep the first request paused across the retry.
        await asyncio.sleep(1.5)
        assert (await client.get(f"{device}/status")).json()["version"] == 1

        # --- 2) reconcile resends the ORIGINAL stored package; the retry is
        # not delayed (one-shot) and activates v2 while v1 is still current.
        await client.post(f"{api}/installations/reconcile")
        view = await _wait_terminal(client, api, t2)
        assert view["state"] == "SUCCESS"
        assert (await client.get(f"{device}/status")).json()["version"] == 2

        # --- 3) v3 is now installed successfully, freeing the slot.
        t3 = await _submit(client, api, (fw3, mb3, sig3, sha3))
        view = await _wait_terminal(client, api, t3)
        assert view["state"] == "SUCCESS"
        assert (await client.get(f"{device}/status")).json()["version"] == 3

        # --- 4) the first (stale) v2 request finally resumes on the device.
        await asyncio.sleep(1.5)
        final = (await client.get(f"{device}/status")).json()
        assert final["version"] == 3
        assert final["sha256"] == sha3  # v3 digest intact: never downgraded

        journal = (await client.get(f"{device}/admin/install-journal")).json()
        events = [e["event"] for e in journal["entries"]]
        assert "rejected_stale_after_wait" in events

        # Task results match the device's final reality.
        assert (await _get_task(client, api, t2))["state"] == "SUCCESS"
        assert (await _get_task(client, api, t3))["state"] == "SUCCESS"


@pytest.mark.asyncio
async def test_late_request_that_never_activated_leaves_tasks_consistent(live_stack):
    """The stale request's own receipt is a rejection; no task is wrongly failed."""
    api = live_stack["api"]
    device = live_stack["device"]
    priv = live_stack["priv"]

    async with httpx.AsyncClient(timeout=8.0) as client:
        await client.post(
            f"{device}/admin/activation-delay",
            json={"seconds": 4.0, "one_shot": True},
        )
        pkg2 = make_package(priv, version=2, size=128)
        t2 = await _submit(client, api, pkg2)
        await _wait_state(client, api, t2, "VERIFYING")
        await asyncio.sleep(1.5)

        # Reconcile completes v2 via the (undelayed) retry.
        await client.post(f"{api}/installations/reconcile")
        await _wait_terminal(client, api, t2)

        # v4 jumps ahead while the stale v2 is still paused on the device.
        pkg4 = make_package(priv, version=4, size=128)
        _, _, _, sha4 = pkg4
        t4 = await _submit(client, api, pkg4)
        await _wait_terminal(client, api, t4)

        await asyncio.sleep(1.5)  # stale v2 resumes and is refused
        final = (await client.get(f"{device}/status")).json()
        assert final["version"] == 4 and final["sha256"] == sha4

        events = [
            e["event"]
            for e in (await client.get(f"{device}/admin/install-journal")).json()[
                "entries"
            ]
        ]
        assert events.count("installed") == 2  # v2 retry, then v4
        assert "rejected_stale_after_wait" in events
