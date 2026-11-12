"""Simulated target gateway device.

The device enforces its own acceptance rules:

* only a higher version is installed (anti-downgrade);
* same version + same digest is idempotent -> the previous result is returned;
* same version + different digest is rejected;
* activation (firmware/version/digest switch) is persisted atomically.

Interleaving / stale-request protection
---------------------------------------

An ``/install`` request performs *two* observations of the durable device
state: once before the (optionally delayed) activation and once inside the
activation transaction.  The activation itself is a guarded, conditional
``UPDATE ... WHERE version < ?`` executed while holding an in-process install
lock, so a request that was correct when it arrived can never overwrite a
newer version that another request activated in the meantime ("late stale
request" / timeout-retry interleaving).  Every decision is made against the
state observed by *this* request, and the receipt always reflects what
actually happened to this request:

* the guarded update activates the new version        -> ``installed`` (200)
* current version is already (version, same digest)   -> ``already_installed`` (200)
* same version, different digest                      -> ``rejected`` (409)
* current version is higher (stale/late request)      -> ``rejected`` (409)
* digest mismatch / bad model / bad encoding          -> ``rejected`` (400)

The process runs uvicorn with a single worker, so the in-process lock
serializes every install against every other install and every admin reset.
Status queries never take the lock and stay available while an install waits.

Fault injection for demo / recovery testing:

* POST /admin/activation-delay    -> pause N seconds AFTER the version check
  passes but BEFORE the guarded activation (``one_shot`` arms a single
  request); this reproduces an API timeout/retry interleaving;
* POST /admin/lose-next-receipt  -> the next /install that actually activates
  new firmware will commit the switch locally but never send a receipt
  (connection dropped), exactly like an activated-but-receipt-lost event;
* GET  /admin/install-journal    -> ordered per-request install decisions;
* POST /admin/reset              -> restore the factory state.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

DEVICE_MODEL = os.environ.get("DEVICE_MODEL", "gateway-gx100")
DB_PATH = Path(os.environ.get("DEVICE_DB_PATH", "/data/device.db"))
FACTORY_VERSION = int(os.environ.get("DEVICE_FACTORY_VERSION", "1"))
FACTORY_DIGEST = os.environ.get(
    "DEVICE_FACTORY_DIGEST",
    "0000000000000000000000000000000000000000000000000000000000000001",
)
# Default post-check activation delay (seconds). 0 keeps the device snappy;
# the Compose demo overrides this so a timeout/retry interleave is visible.
DEFAULT_ACTIVATION_DELAY = float(os.environ.get("DEVICE_ACTIVATION_DELAY", "0"))

# How many per-request decisions the in-memory journal keeps.
JOURNAL_MAXLEN = 200


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = await _connect()
    await conn.close()
    yield


app = FastAPI(title="Gateway Device Simulator", version="1.1.0", lifespan=lifespan)

# In-process fault-injection flags. They only govern receipt delivery and the
# post-version-check pause; the firmware switch itself is always durable first.
_lose_next_receipt = False

# Serializes the whole decision/activation section of /install (and admin
# reset) within this single-worker process. Reads outside the critical
# section, and /status, never block on it.
_install_lock = asyncio.Lock()

# Post-version-check activation pause. ``one_shot`` applies the pause only to
# the next install request that reaches the activation phase, then disarms.
_activation_delay_seconds = DEFAULT_ACTIVATION_DELAY
_activation_delay_one_shot = False

# Bounded, ordered record of every install decision (not persisted: it is a
# demo/test observability aid for interleaving scenarios).
_install_journal: deque[dict] = deque(maxlen=JOURNAL_MAXLEN)


class InstallRequest(BaseModel):
    model: str
    version: int = Field(..., gt=0)
    sha256: str = Field(..., min_length=64, max_length=64)
    firmware_b64: str


class LoseReceiptRequest(BaseModel):
    enabled: bool = True


class ActivationDelayRequest(BaseModel):
    seconds: float = Field(..., ge=0, le=60)
    one_shot: bool = False


def _journal(request_id: str, event: str, **fields) -> None:
    entry = {"ts": time.time(), "request_id": request_id, "event": event}
    entry.update(fields)
    _install_journal.append(entry)


async def _connect() -> aiosqlite.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(DB_PATH)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS device_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            model TEXT NOT NULL,
            version INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            installed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        )
        """
    )
    await conn.execute(
        "INSERT OR IGNORE INTO device_state (id, model, version, sha256) "
        "VALUES (1, ?, ?, ?)",
        (DEVICE_MODEL, FACTORY_VERSION, FACTORY_DIGEST),
    )
    await conn.commit()
    return conn


async def _state(conn: aiosqlite.Connection) -> aiosqlite.Row:
    async with conn.execute(
        "SELECT model, version, sha256, installed_at FROM device_state WHERE id = 1"
    ) as cur:
        return await cur.fetchone()


def _rejected(status_code: int, reason: str) -> HTTPException:
    """A rejection whose body is a real receipt for this request."""
    return HTTPException(
        status_code=status_code,
        detail={"status": "rejected", "reason": reason},
    )


@app.get("/status")
async def status():
    # Never takes the install lock: task/state queries remain available while
    # an installation is paused in its activation delay.
    conn = await _connect()
    try:
        row = await _state(conn)
        return {
            "model": row["model"],
            "version": row["version"],
            "sha256": row["sha256"],
            "installed_at": row["installed_at"],
            "installed": True,
        }
    finally:
        await conn.close()


@app.post("/install")
async def install(
    req: InstallRequest,
    x_request_id: str | None = Header(default=None),
    x_lose_receipt: bool | None = Header(default=None),
):
    global _lose_next_receipt, _activation_delay_seconds, _activation_delay_one_shot

    request_id = x_request_id or uuid.uuid4().hex[:12]

    if req.model != DEVICE_MODEL:
        raise HTTPException(status_code=400, detail=f"unknown model {req.model!r}")

    try:
        firmware = base64.b64decode(req.firmware_b64, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="invalid firmware encoding") from exc

    digest = hashlib.sha256(firmware).hexdigest()
    if digest != req.sha256.lower():
        raise _rejected(400, "firmware sha256 mismatch")

    # Decision point 1 runs under the lock; the lock is released during the
    # (optional) activation pause so a timeout retry and a later install can
    # interleave. Decision point 2 re-acquires the lock and makes the final
    # call against the fresh, durable state via a guarded conditional UPDATE.
    async with _install_lock:
        conn = await _connect()
        try:
            row = await _state(conn)
            current_version = row["version"]
            current_digest = row["sha256"]

            # ---- decision point 1: against the state observed right now ----
            if req.version < current_version:
                _journal(
                    request_id,
                    "rejected_downgrade",
                    requested=req.version,
                    current=current_version,
                )
                raise _rejected(
                    409,
                    f"downgrade refused: current v{current_version}, "
                    f"requested v{req.version}",
                )

            if req.version == current_version:
                if req.sha256.lower() == current_digest:
                    # Idempotent repeat: return the original result unchanged.
                    _journal(
                        request_id,
                        "already_installed",
                        version=current_version,
                        digest=current_digest,
                    )
                    return {
                        "status": "already_installed",
                        "request_id": request_id,
                        "model": DEVICE_MODEL,
                        "version": current_version,
                        "sha256": current_digest,
                    }
                _journal(
                    request_id,
                    "rejected_digest_conflict",
                    version=req.version,
                    requested_digest=digest,
                    current_digest=current_digest,
                )
                raise _rejected(
                    409,
                    f"same version v{req.version} with different digest "
                    "is forbidden",
                )
        finally:
            await conn.close()

    # req.version > current_version at decision point 1: this request is
    # allowed to activate. Apply the controllable post-version-check pause
    # BEFORE the durable switch, WITHOUT holding the lock, so a timeout/retry
    # interleave is observable.
    delay = _activation_delay_seconds
    one_shot = _activation_delay_one_shot
    if delay > 0:
        _journal(
            request_id,
            "activation_delay",
            requested=req.version,
            delay_seconds=delay,
            one_shot=one_shot,
        )
        if one_shot:
            # Only this single request pauses; later requests (the retry, or
            # v3) proceed without waiting.
            _activation_delay_seconds = 0.0
            _activation_delay_one_shot = False
        await _sleep_interruptible(delay)

    # Decision point 2: re-acquire the lock and re-check inside the
    # transaction. Another request may have activated a newer version while
    # this one was paused. The guarded UPDATE makes the check and the switch
    # atomic, so a stale request can never downgrade.
    async with _install_lock:
        conn = await _connect()
        try:
            cur = await conn.execute(
                "UPDATE device_state "
                "SET model = ?, version = ?, sha256 = ?, "
                "installed_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') "
                "WHERE id = 1 AND version < ?",
                (DEVICE_MODEL, req.version, digest, req.version),
            )
            await conn.commit()
            activated = cur.rowcount == 1

            if not activated:
                # The guarded update changed nothing: a newer (or equal)
                # version is already active. Classify against the fresh state
                # so this request's receipt matches reality.
                row = await _state(conn)
                fresh_version = row["version"]
                fresh_digest = row["sha256"]
                if fresh_version == req.version and fresh_digest == digest:
                    _journal(
                        request_id,
                        "already_installed_after_wait",
                        version=fresh_version,
                    )
                    return {
                        "status": "already_installed",
                        "request_id": request_id,
                        "model": DEVICE_MODEL,
                        "version": fresh_version,
                        "sha256": fresh_digest,
                    }
                if fresh_version == req.version:
                    reason = (
                        f"same version v{req.version} with different digest "
                        "is forbidden"
                    )
                    event = "rejected_digest_conflict_after_wait"
                else:
                    reason = (
                        f"stale request refused: device already at "
                        f"v{fresh_version}, requested v{req.version}"
                    )
                    event = "rejected_stale_after_wait"
                _journal(
                    request_id,
                    event,
                    requested=req.version,
                    current=fresh_version,
                    current_digest=fresh_digest,
                )
                raise _rejected(409, reason)

            _journal(request_id, "installed", version=req.version, digest=digest)
        finally:
            await conn.close()

    # Durable activation committed. Now handle receipt delivery (outside the
    # lock so a dropped connection never blocks other installs).
    should_lose = _lose_next_receipt or bool(x_lose_receipt)
    if should_lose:
        _lose_next_receipt = False
        # Simulate a dropped connection AFTER durable activation. Raising here
        # means the API receives no receipt while the switch already happened.
        raise HTTPException(
            status_code=502,
            detail={
                "status": "receipt_lost",
                "request_id": request_id,
                "note": "firmware was activated but the receipt connection dropped",
            },
        )

    return {
        "status": "installed",
        "request_id": request_id,
        "model": DEVICE_MODEL,
        "version": req.version,
        "sha256": digest,
    }


async def _sleep_interruptible(seconds: float) -> None:
    """Pause while staying responsive to process shutdown (CancelledError)."""
    await asyncio.sleep(seconds)


@app.get("/admin/install-journal")
async def install_journal(limit: int = JOURNAL_MAXLEN):
    """Ordered per-request install decisions (newest last), for demos/tests."""
    limit = max(1, min(limit, JOURNAL_MAXLEN))
    entries = list(_install_journal)[-limit:]
    return {"count": len(entries), "entries": entries}


@app.post("/admin/activation-delay")
async def activation_delay(req: ActivationDelayRequest):
    """Configure the post-version-check activation pause.

    * ``seconds`` > 0 pauses each install that passes the version check and
      reaches activation;
    * ``one_shot=true`` applies the pause to the next such request only and
      then disarms automatically (used for the timeout/retry demo).
    """
    global _activation_delay_seconds, _activation_delay_one_shot
    _activation_delay_seconds = float(req.seconds)
    _activation_delay_one_shot = bool(req.one_shot)
    return {
        "activation_delay_seconds": _activation_delay_seconds,
        "one_shot": _activation_delay_one_shot,
    }


@app.post("/admin/lose-next-receipt")
async def lose_next_receipt(req: LoseReceiptRequest):
    global _lose_next_receipt
    _lose_next_receipt = req.enabled
    return {"lose_next_receipt": _lose_next_receipt}


@app.post("/admin/reset")
async def reset():
    global _lose_next_receipt, _activation_delay_seconds, _activation_delay_one_shot
    async with _install_lock:
        _lose_next_receipt = False
        _activation_delay_seconds = DEFAULT_ACTIVATION_DELAY
        _activation_delay_one_shot = False
        conn = await _connect()
        try:
            await conn.execute(
                "UPDATE device_state SET model = ?, version = ?, sha256 = ?, "
                "installed_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = 1",
                (DEVICE_MODEL, FACTORY_VERSION, FACTORY_DIGEST),
            )
            await conn.commit()
            row = await _state(conn)
            _install_journal.clear()
            return {
                "status": "reset",
                "model": row["model"],
                "version": row["version"],
                "sha256": row["sha256"],
            }
        finally:
            await conn.close()
