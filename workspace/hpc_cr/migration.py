"""可恢复的检查点迁移: 冻结清单 → 临时保留 → 逐项传输/复用 →
全量校验 → 一次性发布可恢复代际。

任务状态机::

    PLANNED ──首次 transfer──▶ TRANSFERRING ──finalize 全部校验通过──▶ PUBLISHED
      │                            │
      └──────── cancel ──▶ CANCELLED ◀──────── cancel ────────┘
      │                            │
      └──────────── 终态失败 ──────▶ FAILED (清单变化/摘要错误/校验不过)

每个对象 (检查点分片或同代际输出) 有独立状态:
``PLANNED / TRANSFERRED / REUSED / FAILED``。传输中断、重复请求和服务
重启后继续同一任务: TERMINAL 状态的对象跳过, 未完成对象可重传。

关键不变量:
  * 导出端在创建任务时 *冻结* 对象清单 (含逐项 sha256 摘要、大小和清单
    摘要), 并对源载荷逐个从 CAS 重读重算哈希;
  * 涉及的 blob 在任务终态前被临时保留 (migration hold), GC 计划不会把
    它们选为候选; 取消/失败/发布即释放保留, 不影响正常空间回收;
  * 接收端按摘要复用已有对象 (REUSED), 缺失对象才实际传输;
  * finalize 对 *全部* 对象在目标存储重读重算哈希, 任一缺失/不符/清单
    变化都不产生可恢复代际 (任务 FAILED, 保留释放);
  * 全部校验通过后原子写入发布目录 (published catalog), 代际才一次发布;
  * 任务在发布前撤销/校验终止时, 对 *本任务写入接收端* 的 blob 做一次
    终态遗留清扫: 只删除不再被任何人引用的本次独占内容; 已有同摘要内容
    (REUSED)、其他进行中搬运涉及的对象、已发布恢复版本引用的对象一律
    保留。清扫结果随 ``migration_orphan_cleaned`` 事件持久化, 服务重启、
    重复撤销与回收并发时不会重复统计, 也不会误删仍可用的内容。

本模块只提供无锁的纯函数/视图; 锁与事件编排由 JobManager 负责
(加锁顺序: reference → migration → gc → 各 Job.lock)。
"""

from __future__ import annotations

import uuid
from typing import Any

from .checkpoints import COMPLETE, shard_id
from .errors import Conflict, NotFound
from .security import canonical_json, sha256_hex

# 任务状态
PLANNED = "PLANNED"
TRANSFERRING = "TRANSFERRING"
PUBLISHED = "PUBLISHED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"

TERMINAL_STATES = frozenset({PUBLISHED, FAILED, CANCELLED})
ACTIVE_STATES = frozenset({PLANNED, TRANSFERRING})

# 对象状态
ITEM_PLANNED = "PLANNED"
ITEM_TRANSFERRED = "TRANSFERRED"
ITEM_REUSED = "REUSED"
ITEM_FAILED = "ITEM_FAILED"
ITEM_DONE = frozenset({ITEM_TRANSFERRED, ITEM_REUSED})

# 失败原因 (稳定 reason 码, 同时用于 API、页面和事件)
REASON_MANIFEST_CHANGED = "manifest_changed"
REASON_SOURCE_DIGEST = "source_digest_mismatch"
REASON_DEST_DIGEST = "destination_digest_mismatch"
REASON_INCOMPLETE = "incomplete_objects"
REASON_DEST_WRITE = "destination_write_failed"
REASON_CANCELLED = "cancelled_by_user"

# 终态遗留清扫: 保留原因 / 恢复原因 (稳定 reason 码, API/页面/事件共用)
KEEP_PUBLISHED = "published_generation"        # 已发布恢复版本仍引用
KEEP_ACTIVE_MIGRATION = "active_migration"     # 其他进行中的搬运仍需要
KEEP_DESTINATION_JOB = "destination_job_reference"  # 绑定该存储的本地作业引用
KEEP_DELETE_FAILED = "delete_failed"           # 物理删除失败 (权限/IO)
KEEP_DEST_UNAVAILABLE = "destination_unavailable"   # 接收端存储不可用
RECOVER_ALREADY_DELETED = "already_deleted_after_crash"  # 崩溃前已物理删除

KIND_SHARD = "checkpoint_shard"
KIND_OUTPUT = "output"

MANIFEST_VERSION = 1


def new_migration_id() -> str:
    return f"mig-{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------- manifest
def _manifest_material(job: Job, generation: int) -> dict[str, Any]:
    """从作业当前状态构造规范化清单材料 (冻结与变更检测共用同一算法)。

    COMPLETE 检查点自身 (分片集合/进程集合) 不可变; 同代际输出只增不减,
    因此清单变化只能来自新输出提交。
    """
    cp = job.cpm.get(generation)
    shards = [
        cp.shards[shard_id(r, i)]
        for r in range(cp.world_size)
        for i in range(cp.shards_per_rank)
    ]
    outputs = [o for o in job.outputs.list(generation)]

    objects: list[dict[str, Any]] = []
    for sh in shards:
        objects.append({
            "object_id": shard_id(sh.rank, sh.index),
            "kind": KIND_SHARD,
            "blob": sh.blob,
            "size": sh.size,
            "rank": sh.rank,
            "index": sh.index,
            "node": sh.node,
            "input_pos": sh.input_pos,
        })
    for out in sorted(outputs, key=lambda o: o.output_id):
        objects.append({
            "object_id": out.output_id,
            "kind": KIND_OUTPUT,
            "blob": out.blob,
            "size": out.size,
            "rank": out.rank,
            "output_generation": out.generation,
            "output_ts": out.ts,
        })
    objects.sort(key=lambda o: (o["kind"], o["object_id"]))
    return {
        "manifest_version": MANIFEST_VERSION,
        "source_job_id": job.job_id,
        "checkpoint": {
            "job_id": job.job_id,
            "epoch": cp.epoch,
            "generation": cp.generation,
            "world_size": cp.world_size,
            "shards_per_rank": cp.shards_per_rank,
            "processes": [[r, n] for r, n in cp.processes],
            "completed_ts": cp.completed_ts,
            "checkpoint_manifest_digest":
                (cp.manifest or {}).get("manifest_digest"),
        },
        "objects": objects,
    }


def manifest_digest_of(material: dict[str, Any]) -> str:
    return sha256_hex(canonical_json(material))


def recompute_manifest_digest(manifest: dict[str, Any]) -> str:
    """对已冻结/已发布清单重算摘要。

    摘要只覆盖清单核心材料 (manifest_version/来源/检查点/对象);
    total_items/total_bytes 是派生展示字段, 不参与摘要。
    """
    skip = {"manifest_digest", "total_items", "total_bytes"}
    return manifest_digest_of(
        {k: v for k, v in manifest.items() if k not in skip})


def verify_manifest(manifest: dict[str, Any]) -> bool:
    return recompute_manifest_digest(manifest) == manifest.get("manifest_digest")


def freeze_manifest(job: Job, generation: int) -> dict[str, Any]:
    """冻结导出清单。调用方必须持有 job.lock 与全局引用锁。

    COMPLETE 是前置条件, 但仍逐个从源 CAS 重读重算哈希 —— bit-rot/缺失
    在导出前暴露, 绝不冻结一份引用坏载荷的清单。源校验失败抛 Conflict,
    调用方不得创建任务、不得加保留。
    """
    try:
        cp = job.cpm.get(generation)
    except NotFound as exc:
        raise Conflict(
            f"migration requires a COMPLETE checkpoint at gen{generation}"
        ) from exc
    if cp.state != COMPLETE:
        raise Conflict(
            f"only COMPLETE checkpoints can be migrated; gen{generation} "
            f"is {cp.state}"
        )
    material = _manifest_material(job, generation)
    bad: list[str] = []
    for obj in material["objects"]:
        if not job.store.verify(obj["blob"]):
            bad.append(obj["object_id"])
    if bad:
        raise Conflict(
            "export manifest freeze refused: source payloads missing or "
            f"corrupt: {bad[:10]}"
        )
    manifest = dict(material)
    manifest["manifest_digest"] = manifest_digest_of(material)
    manifest["total_items"] = len(material["objects"])
    manifest["total_bytes"] = sum(o["size"] for o in material["objects"])
    return manifest


def current_manifest_digest(job: Job, generation: int) -> str:
    """重新计算当前清单摘要 (用于冻结后检测清单变化)。"""
    return manifest_digest_of(_manifest_material(job, generation))


# -------------------------------------------------------------- task view
def make_task(
    *,
    migration_id: str,
    job: Job,
    generation: int,
    destination_id: str,
    manifest: dict[str, Any],
    watermark_seq: int,
    now: float,
) -> dict[str, Any]:
    objects = [
        {
            "object_id": o["object_id"],
            "kind": o["kind"],
            "blob": o["blob"],
            "size": o["size"],
            "status": ITEM_PLANNED,
            "attempts": 0,
            "last_error": None,
            "completed_ts": None,
        }
        for o in manifest["objects"]
    ]
    return {
        "migration_id": migration_id,
        "source_job_id": job.job_id,
        "source_name": job.name,
        "source_generation": generation,
        "destination_id": destination_id,
        "state": PLANNED,
        "created_ts": now,
        "started_ts": None,
        "finished_ts": None,
        "published_ts": None,
        "watermark_seq": watermark_seq,
        "completion_seq": None,
        "manifest_digest": manifest["manifest_digest"],
        "manifest": manifest,
        "objects": objects,
        "total_items": manifest["total_items"],
        "total_bytes": manifest["total_bytes"],
        "transferred_items": 0,
        "transferred_bytes": 0,
        "reused_items": 0,
        "reused_bytes": 0,
        "failed_items": 0,
        # 本任务物理写入接收端的 blob 摘要集合 (transferred/repaired);
        # REUSED 对象是接收端已有内容, 绝不进入清扫候选。
        "writes": set(),
        # 终态遗留清扫结果 (取消/失败时执行一次, 随事件持久化)。
        "cleanup": None,
        "error": None,
        "reason": None,
        "publication": None,
    }


def recount(task: dict[str, Any]) -> None:
    """从对象明细重算聚合计数 (事件重放后以此为准)。"""
    task["transferred_items"] = sum(
        1 for o in task["objects"] if o["status"] == ITEM_TRANSFERRED)
    task["reused_items"] = sum(
        1 for o in task["objects"] if o["status"] == ITEM_REUSED)
    task["failed_items"] = sum(
        1 for o in task["objects"] if o["status"] == ITEM_FAILED)
    task["transferred_bytes"] = sum(
        o["size"] for o in task["objects"]
        if o["status"] == ITEM_TRANSFERRED)
    task["reused_bytes"] = sum(
        o["size"] for o in task["objects"] if o["status"] == ITEM_REUSED)


def cleanup_public(cleanup: dict[str, Any] | None) -> dict[str, Any] | None:
    """终态遗留清扫结果的稳定 JSON 视图 (None 表示尚未清扫/不需要)。"""
    if cleanup is None:
        return None
    view = dict(cleanup)
    view["kept"] = [dict(k) for k in cleanup.get("kept", [])]
    view["recovery_reasons"] = dict(cleanup.get("recovery_reasons", {}))
    return view


def task_public(task: dict[str, Any]) -> dict[str, Any]:
    """稳定 JSON 视图 (清单/逐项进度/聚合计数/最终状态与原因/遗留清扫)。"""
    return {
        "migration_id": task["migration_id"],
        "source_job_id": task["source_job_id"],
        "source_name": task.get("source_name"),
        "source_generation": task["source_generation"],
        "destination_id": task["destination_id"],
        "state": task["state"],
        "created_ts": task.get("created_ts"),
        "started_ts": task.get("started_ts"),
        "finished_ts": task.get("finished_ts"),
        "published_ts": task.get("published_ts"),
        "watermark_seq": task.get("watermark_seq"),
        "completion_seq": task.get("completion_seq"),
        "manifest_digest": task["manifest_digest"],
        "total_items": task["total_items"],
        "total_bytes": task["total_bytes"],
        "transferred_items": task["transferred_items"],
        "transferred_bytes": task["transferred_bytes"],
        "reused_items": task["reused_items"],
        "reused_bytes": task["reused_bytes"],
        "failed_items": task["failed_items"],
        "done_items": task["transferred_items"] + task["reused_items"],
        "written_blobs": sorted(task.get("writes") or ()),
        "cleanup": cleanup_public(task.get("cleanup")),
        "error": task.get("error"),
        "reason": task.get("reason"),
        "publication": task.get("publication"),
        "manifest": task["manifest"],
        "objects": [dict(o) for o in task["objects"]],
    }


def publication_from_task(task: dict[str, Any], *, ts: float) -> dict[str, Any]:
    """全部校验通过后一次性写入发布目录的 *可恢复代际* 记录。"""
    return {
        "migration_id": task["migration_id"],
        "source_job_id": task["source_job_id"],
        "source_name": task.get("source_name"),
        "source_generation": task["source_generation"],
        "destination_id": task["destination_id"],
        "manifest_digest": task["manifest_digest"],
        "published_ts": ts,
        "total_items": task["total_items"],
        "total_bytes": task["total_bytes"],
        "transferred_items": task["transferred_items"],
        "reused_items": task["reused_items"],
        # 接收端 adopt 据此重建 COMPLETE 检查点/输出, 不需要源集群在线。
        "manifest": task["manifest"],
    }


def publication_key(record: dict[str, Any]) -> tuple[str, str, int, str]:
    return (record["destination_id"], record["source_job_id"],
            record["source_generation"], record["manifest_digest"])
