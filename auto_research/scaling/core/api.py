"""Conservative operational API limits; never change scientific request settings."""

import hashlib
import sqlite3
import time
import uuid
from contextlib import closing

from .common import DataError, read, root


def limits():
    path = root() / "config/operations.json"
    values = {
        "generation_concurrency": 8,
        "requests_per_minute_per_key": 30,
        "tokens_per_minute_per_key": 1500000,
        "minimum_interval_seconds": 2.0,
        **(read(path).get("api_limits", {}) if path.exists() else {}),
    }
    if not (
        1 <= values["generation_concurrency"] <= 32
        and 1 <= values["requests_per_minute_per_key"] <= 90
        and 1 <= values["tokens_per_minute_per_key"] <= 1800000
        and values["minimum_interval_seconds"] >= 0.68
    ):
        raise DataError("Invalid operational API limits")
    return values


def reserve_quota(key, tokens):
    """Count every attempt, including retries, with atomic per-key reservations."""
    path = root() / "state/api_quota.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(key.encode()).hexdigest()
    request_id = uuid.uuid4().hex
    while True:
        policy = limits()
        if tokens > policy["tokens_per_minute_per_key"]:
            raise DataError("Request exceeds operational token reservation limit")
        with closing(sqlite3.connect(path, timeout=30)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS calls "
                "(id TEXT PRIMARY KEY,k TEXT,t REAL,n INTEGER)"
            )
            db.execute("BEGIN IMMEDIATE")
            now = time.time()
            db.execute("DELETE FROM calls WHERE t < ?", (now - 61,))
            count, total, last = db.execute(
                "SELECT COUNT(*),COALESCE(SUM(n),0),COALESCE(MAX(t),0) "
                "FROM calls WHERE k=? AND t>?",
                (fingerprint, now - 60),
            ).fetchone()
            if (
                count < policy["requests_per_minute_per_key"]
                and total + tokens <= policy["tokens_per_minute_per_key"]
                and now - last >= policy["minimum_interval_seconds"]
            ):
                db.execute(
                    "INSERT INTO calls VALUES (?,?,?,?)",
                    (request_id, fingerprint, now, tokens),
                )
                return path, request_id
        time.sleep(0.7)
