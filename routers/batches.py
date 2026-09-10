"""Zep Cloud Batch API (/api/v2/batches*) for zep-cloud 3.25 SDK clients.

Implements the wire contract MiroFish's GraphBuilderService drives:
draft batch -> add graph_episode items -> process -> poll status/progress ->
list items for episode uuids. State lives in SQLite; ingestion reuses the
same resilient bulk pipeline as POST /graph-batch, and per-item episode
uuids are tracked in the shared in-memory episode tracker so
GET /graph/episodes/{uuid} keeps reporting `processed` for them.
"""
import asyncio
import json
import logging
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Request
from graphiti_core.utils.bulk_utils import RawEpisode

from database import DB_PATH, get_db
from deps import get_graphiti, verify_api_key
from engine.data_ingestion import normalize_episode_body, normalize_episode_type
from models.batch import BatchAddItemsRequest, BatchCreateRequest
from models.graph import GraphAddBatchRequest
from ontology_registry import get_ontology
from routers.graph import (
    _add_episode_bulk_resilient,
    _episode_status,
    _get_processing_sem,
)

router = APIRouter(prefix="/api/v2", tags=["batches"], dependencies=[Depends(verify_api_key)])
logger = logging.getLogger(__name__)

_MAX_ITEM_CHARS = 10_000
_INGEST_SUBGROUP_SIZE = 10


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _get_batch_row(db: aiosqlite.Connection, batch_id: str) -> aiosqlite.Row:
    cur = await db.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,))
    row = await cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"batch {batch_id} not found")
    return row


async def _item_status_counts(db: aiosqlite.Connection, batch_id: str) -> dict[str, int]:
    cur = await db.execute(
        "SELECT status, COUNT(*) AS c FROM batch_items WHERE batch_id = ? GROUP BY status",
        (batch_id,),
    )
    return {row["status"]: row["c"] for row in await cur.fetchall()}


async def _batch_summary(db: aiosqlite.Connection, row: aiosqlite.Row) -> dict[str, Any]:
    counts = await _item_status_counts(db, row["batch_id"])
    total = sum(counts.values())
    succeeded = counts.get("succeeded", 0)
    failed = counts.get("failed", 0)
    skipped = counts.get("skipped", 0)
    done = succeeded + failed + skipped
    progress = {
        "total_items": total,
        "succeeded_items": succeeded,
        "failed_items": failed,
        "skipped_items": skipped,
        "processing_items": counts.get("processing", 0),
        "queued_items": counts.get("queued", 0) + counts.get("pending", 0),
        "canceled_items": 0,
        "percent_complete": round(done * 100.0 / total, 2) if total else 0.0,
    }
    return {
        "batch_id": row["batch_id"],
        "status": row["status"],
        "item_count": total,
        "metadata": json.loads(row["metadata"] or "{}"),
        "progress": progress,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "processed_at": row["processed_at"],
        "completed_at": row["completed_at"],
    }


def _item_detail(row: aiosqlite.Row) -> dict[str, Any]:
    episode_uuid = row["episode_uuid"]
    return {
        "item_id": row["item_id"],
        "sequence_index": row["sequence_index"],
        "status": row["status"],
        "kind": row["kind"],
        "graph_id": row["graph_id"],
        "episode_uuid": episode_uuid,
        "source_uuid": episode_uuid,
        "metadata": json.loads(row["metadata"] or "{}"),
        "error": json.loads(row["error"]) if row["error"] else None,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


async def _set_batch_status(
    db: aiosqlite.Connection,
    batch_id: str,
    status: str,
    *,
    completed: bool = False,
    processed: bool = False,
) -> None:
    now = _now_iso()
    await db.execute(
        "UPDATE batches SET status = ?, updated_at = ?, "
        "processed_at = COALESCE(processed_at, ?), completed_at = COALESCE(completed_at, ?) "
        "WHERE batch_id = ?",
        (status, now, now if processed else None, now if completed else None, batch_id),
    )
    await db.commit()


# ── POST /batches (batch.create) ──────────────────────────────────────────────

@router.post("/batches")
async def create_batch(body: BatchCreateRequest | None = None, db=Depends(get_db)):
    batch_id = str(_uuid.uuid4())
    now = _now_iso()
    metadata = (body.metadata if body and body.metadata else {}) or {}
    await db.execute(
        "INSERT INTO batches (batch_id, status, metadata, created_at, updated_at) "
        "VALUES (?, 'draft', ?, ?, ?)",
        (batch_id, json.dumps(metadata), now, now),
    )
    await db.commit()
    return await _batch_summary(db, await _get_batch_row(db, batch_id))


# ── POST /batches/{batch_id}/items (batch.add) ────────────────────────────────

@router.post("/batches/{batch_id}/items")
async def add_items(batch_id: str, body: BatchAddItemsRequest, db=Depends(get_db)):
    row = await _get_batch_row(db, batch_id)
    if row["status"] != "draft":
        raise HTTPException(
            status_code=409,
            detail=f"batch {batch_id} is {row['status']}; items can only be added to a draft",
        )
    if not body.items:
        raise HTTPException(status_code=400, detail="items must not be empty")

    for item in body.items:
        if item.type != "graph_episode" or item.data_type != "text":
            raise HTTPException(
                status_code=400,
                detail=f"unsupported batch item type={item.type}/data_type={item.data_type}; "
                "only graph_episode text items are supported",
            )
        if not item.graph_id:
            raise HTTPException(status_code=400, detail="graph_episode items require graph_id")
        if not item.data.strip():
            raise HTTPException(status_code=400, detail="graph_episode items require data")
        if len(item.data) > _MAX_ITEM_CHARS:
            raise HTTPException(
                status_code=400,
                detail=f"batch item exceeds {_MAX_ITEM_CHARS} characters",
            )

    cur = await db.execute(
        "SELECT COUNT(*) FROM batch_items WHERE batch_id = ?", (batch_id,)
    )
    base_index = (await cur.fetchone())[0]

    now = _now_iso()
    details = []
    for offset, item in enumerate(body.items):
        item_id = str(_uuid.uuid4())
        sequence_index = base_index + offset
        episode_uuid = str(_uuid.uuid4())
        _episode_status[episode_uuid] = False
        await db.execute(
            "INSERT INTO batch_items (item_id, batch_id, sequence_index, status, kind, "
            "data, data_type, graph_id, source_description, metadata, episode_uuid, "
            "created_at, updated_at) VALUES (?, ?, ?, 'pending', 'graph_episode', ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item_id,
                batch_id,
                sequence_index,
                item.data,
                item.data_type,
                item.graph_id,
                item.source_description,
                json.dumps(item.metadata or {}),
                episode_uuid,
                now,
                now,
            ),
        )
        details.append(
            {
                "item_id": item_id,
                "sequence_index": sequence_index,
                "status": "pending",
                "kind": "graph_episode",
                "graph_id": item.graph_id,
                "episode_uuid": episode_uuid,
                "source_uuid": episode_uuid,
                "created_at": now,
                "updated_at": now,
            }
        )
    await db.commit()
    return details


# ── POST /batches/{batch_id}/process (batch.process) ──────────────────────────

@router.post("/batches/{batch_id}/process")
async def process_batch(batch_id: str, request: Request, db=Depends(get_db)):
    row = await _get_batch_row(db, batch_id)
    cur = await db.execute("SELECT COUNT(*) FROM batch_items WHERE batch_id = ?", (batch_id,))
    if (await cur.fetchone())[0] == 0:
        raise HTTPException(status_code=400, detail=f"batch {batch_id} has no items")

    if row["status"] == "draft":
        await _set_batch_status(db, batch_id, "queued", processed=True)
        graphiti = get_graphiti(request)
        asyncio.create_task(_run_batch(graphiti, batch_id))
    # already queued/processing/terminal -> idempotent: report current state

    return await _batch_summary(db, await _get_batch_row(db, batch_id))


# ── GET /batches/{batch_id} (batch.get) ───────────────────────────────────────

@router.get("/batches/{batch_id}")
async def get_batch(batch_id: str, db=Depends(get_db)):
    return await _batch_summary(db, await _get_batch_row(db, batch_id))


# ── GET /batches (batch.list) ─────────────────────────────────────────────────

@router.get("/batches")
async def list_batches(
    db=Depends(get_db),
    limit: int = 100,
    cursor: int | None = None,
    status: str | None = None,
):
    offset = max(0, cursor or 0)
    limit = max(1, min(limit, 500))
    if status:
        cur = await db.execute(
            "SELECT * FROM batches WHERE status = ? ORDER BY created_at, batch_id "
            "LIMIT ? OFFSET ?",
            (status, limit + 1, offset),
        )
    else:
        cur = await db.execute(
            "SELECT * FROM batches ORDER BY created_at, batch_id LIMIT ? OFFSET ?",
            (limit + 1, offset),
        )
    rows = await cur.fetchall()
    has_more = len(rows) > limit
    batches = [await _batch_summary(db, row) for row in rows[:limit]]
    return {"batches": batches, "next_cursor": (offset + limit) if has_more else None}


# ── GET /batches/{batch_id}/items (batch.list_items) ──────────────────────────

@router.get("/batches/{batch_id}/items")
async def list_batch_items(
    batch_id: str,
    db=Depends(get_db),
    limit: int = 100,
    cursor: int | None = None,
):
    await _get_batch_row(db, batch_id)
    offset = max(0, cursor or 0)
    limit = max(1, min(limit, 500))
    cur = await db.execute(
        "SELECT * FROM batch_items WHERE batch_id = ? ORDER BY sequence_index "
        "LIMIT ? OFFSET ?",
        (batch_id, limit + 1, offset),
    )
    rows = await cur.fetchall()
    has_more = len(rows) > limit
    items = [_item_detail(row) for row in rows[:limit]]
    return {"items": items, "next_cursor": (offset + limit) if has_more else None}


# ── background processing ─────────────────────────────────────────────────────

async def _count_group_episodes(driver, graph_id: str) -> int:
    res = await driver.execute_query(
        "MATCH (e:Episodic) WHERE e.group_id = $gid RETURN count(e) AS c",
        gid=graph_id,
    )
    records = getattr(res, "records", None) or []
    if not records:
        return 0
    record = records[0]
    try:
        return int(record["c"] or 0)
    except (KeyError, IndexError, TypeError):
        return int(getattr(record, "c", 0) or 0)


async def _run_batch(graphiti, batch_id: str) -> None:
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            await _set_batch_status(db, batch_id, "processing")
            cur = await db.execute(
                "SELECT * FROM batch_items WHERE batch_id = ? ORDER BY sequence_index",
                (batch_id,),
            )
            items = await cur.fetchall()
            groups: dict[str, list[aiosqlite.Row]] = {}
            for item in items:
                groups.setdefault(item["graph_id"], []).append(item)

            for graph_id, group in groups.items():
                # Ingest in sub-groups: one giant bulk holds the whole batch
                # hostage to a single slow stage, and item status only flips
                # when a group finishes. 10-wide slices keep progress visible
                # and bound the blast radius of any failure.
                for start in range(0, len(group), _INGEST_SUBGROUP_SIZE):
                    await _ingest_group(
                        db, graphiti, batch_id, graph_id, group[start : start + _INGEST_SUBGROUP_SIZE]
                    )

            counts = await _item_status_counts(db, batch_id)
            total = sum(counts.values())
            failed = counts.get("failed", 0)
            if failed == 0:
                final = "succeeded"
            elif failed == total:
                final = "failed"
            else:
                final = "partial"
            await _set_batch_status(db, batch_id, final, completed=True)
            logger.info("batch %s finished as %s", batch_id, final)
    except Exception:
        logger.exception("batch %s processing crashed", batch_id)
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                await _set_batch_status(db, batch_id, "failed", completed=True)
        except Exception:
            logger.exception("batch %s failure-status write failed", batch_id)


async def _ingest_group(
    db: aiosqlite.Connection,
    graphiti,
    batch_id: str,
    graph_id: str,
    group: list[aiosqlite.Row],
) -> None:
    now = _now_iso()
    for item in group:
        await db.execute(
            "UPDATE batch_items SET status = 'processing', updated_at = ? WHERE item_id = ?",
            (now, item["item_id"]),
        )
    await db.commit()

    ontology = get_ontology(graph_id=graph_id, user_id=None)
    raw_episodes = [
        RawEpisode(
            name=f"ep_{graph_id}_{item['sequence_index']}",
            content=normalize_episode_body(item["data"], "text"),
            source_description=item["source_description"] or "batch",
            source=normalize_episode_type("text"),
            reference_time=datetime.now(timezone.utc),
        )
        for item in group
    ]

    before = await _count_group_episodes(graphiti.driver, graph_id)
    error_message: str | None = None
    bulk_errors: list[str] = []
    try:
        async with _get_processing_sem():
            await _add_episode_bulk_resilient(
                graphiti,
                GraphAddBatchRequest(graph_id=graph_id, episodes=[]),
                raw_episodes,
                ontology,
                errors=bulk_errors,
            )
    except Exception as exc:
        logger.error(
            "batch %s ingestion failed for graph %s: %s",
            batch_id,
            graph_id,
            exc,
            exc_info=True,
        )
        error_message = f"{type(exc).__name__}: {exc}"

    landed = (await _count_group_episodes(graphiti.driver, graph_id)) - before
    if error_message is None and landed == 0:
        # Backstop: nothing landed at all (e.g. Neo4j unreachable). Note that
        # episodic nodes land even when later stages fail — per-episode errors
        # from the pipeline are the primary signal, this only catches total loss.
        error_message = (
            "ingestion produced no graph episodes; check LLM/embedder availability"
        )
    elif error_message is None and landed < len(group):
        logger.warning(
            "batch %s graph %s: %s/%s episodes landed",
            batch_id,
            graph_id,
            landed,
            len(group),
        )

    now = _now_iso()
    for item in group:
        episode_name = f"ep_{graph_id}_{item['sequence_index']}"
        item_error = error_message
        if item_error is None:
            # Pipeline errors are recorded as "<episode_name>: <message>".
            matching = [e for e in bulk_errors if e.startswith(f"{episode_name}:")]
            if matching:
                item_error = matching[0].split(": ", 1)[-1] if ": " in matching[0] else matching[0]
        await db.execute(
            "UPDATE batch_items SET status = ?, error = ?, updated_at = ? WHERE item_id = ?",
            (
                "failed" if item_error else "succeeded",
                json.dumps({"message": item_error}) if item_error else None,
                now,
                item["item_id"],
            ),
        )
        _episode_status[item["episode_uuid"]] = True
    await db.commit()
