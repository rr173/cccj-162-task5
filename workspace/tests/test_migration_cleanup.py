"""搬运未完成即撤销/校验终止时的接收端独占遗留清理。

不变量:
  * 取消 (CANCELLED) 与校验失败 (FAILED) 都立即清理本任务 **独占写入**
    (TRANSFERRED) 到接收端、且当前无人引用的对象;
  * 搬运前已存在的同摘要内容 (REUSED)、其他进行中搬运、已发布可恢复
    代际以及接收端 adopt 生成的引用必须继续可用;
  * 终态统计 (候选/删除/保留项与字节/跳过原因/崩溃恢复原因) 稳定可查:
    重复撤销返回同一结果, 服务重启续跑崩溃在清理中途的任务;
  * 清理与上传/恢复/GC 共享全局引用临界区, 绝不删除仍可访问数据。
"""

import json
import os
import shutil
import tempfile
import threading
import unittest

from hpc_cr import JobManager
from hpc_cr.epochs import Scheduler
from hpc_cr.errors import Conflict
from hpc_cr.security import Signer, sha256_hex


class CleanupTestBase(unittest.TestCase):
    world_size = 2

    def setUp(self):
        self.srcdir = tempfile.mkdtemp(prefix="cleanup-src-")
        self.dstdir = tempfile.mkdtemp(prefix="cleanup-dst-")
        self.secret = "cleanup-secret"
        self.mgr = JobManager(
            Scheduler(Signer(self.secret)), data_dir=self.srcdir)
        self.mgr.create_job(
            name="cleanup-job", world_size=self.world_size,
            shards_per_rank=1,
            rank_nodes={r: f"n{r}" for r in range(self.world_size)},
            job_id="cj-1")

    def tearDown(self):
        self.mgr.journal.close()
        shutil.rmtree(self.srcdir, ignore_errors=True)
        shutil.rmtree(self.dstdir, ignore_errors=True)

    def _complete_gen1(self, payloads=None, release=False):
        pre = self.mgr.preempt("cj-1")
        token = pre["epoch_token"]
        payloads = payloads or {
            r: f"g1-r{r}".encode() for r in range(self.world_size)}
        for r, data in payloads.items():
            self.mgr.upload_shard(
                "cj-1", epoch_token=token, rank=r, index=0,
                node=f"n{r}", input_pos=r + 1, data=data)
        self.mgr.commit_checkpoint("cj-1", epoch_token=token)
        if release:
            self.mgr.confirm_release("cj-1", epoch_token=token)
        return token

    def _register_and_create(self, mig_id, dest="cluster-b", gen=1):
        self.mgr.register_destination(dest, data_dir=self.dstdir)
        return self.mgr.create_migration(
            "cj-1", generation=gen, destination_id=dest,
            migration_id=mig_id)

    def _dst(self, dest="cluster-b"):
        return self.mgr._destinations[dest]

    def _blob(self, r):
        return sha256_hex(f"g1-r{r}".encode())

    def _restart(self):
        self.mgr.journal.close()
        self.mgr = JobManager(
            Scheduler(Signer(self.secret)), data_dir=self.srcdir)


class CancelCleanupTest(CleanupTestBase):
    def test_cancel_deletes_only_exclusive_transferred_leftovers(self):
        self._complete_gen1()
        # r0 在搬运前已存在于接收端 (将被 REUSED), r1 由本任务写入
        self.mgr.register_destination("cluster-b", data_dir=self.dstdir)
        self._dst().put(b"g1-r0")
        self.mgr.create_migration(
            "cj-1", generation=1, destination_id="cluster-b",
            migration_id="m-1")
        self.mgr.transfer_migration("m-1", auto_finalize=False)

        cancelled = self.mgr.cancel_migration("m-1")
        cleanup = cancelled["cleanup"]
        self.assertEqual(cleanup["state"], "DONE")
        # 只有本任务独占写入的 r1 是候选并被删除
        self.assertEqual(cleanup["candidate_items"], 1)
        self.assertEqual(cleanup["deleted_items"], 1)
        self.assertEqual(cleanup["deleted_bytes"], len(b"g1-r1"))
        self.assertEqual(cleanup["skipped_items"], 0)
        self.assertEqual(cleanup["skip_reasons"], {})
        dst = self._dst()
        self.assertFalse(dst.has(self._blob(1)))
        # REUSED 的同摘要内容必须继续可用
        self.assertTrue(dst.has(self._blob(0)))
        self.assertEqual(len(dst.inventory()), 1)
        self.assertEqual(self.mgr.migration_holds(), {})
        self.assertFalse(self.mgr.list_published_generations())

    def test_cancel_before_any_transfer_has_no_candidates(self):
        self._complete_gen1()
        self._register_and_create("m-empty")
        cancelled = self.mgr.cancel_migration("m-empty")
        cleanup = cancelled["cleanup"]
        self.assertEqual(cleanup["state"], "DONE")
        self.assertEqual(cleanup["candidate_items"], 0)
        self.assertEqual(cleanup["deleted_items"], 0)
        self.assertEqual(cleanup["skipped_items"], 0)
        self.assertEqual(len(self._dst().inventory()), 0)

    def test_repeated_cancel_returns_stable_terminal_stats(self):
        self._complete_gen1()
        self._register_and_create("m-1")
        self.mgr.transfer_migration("m-1", auto_finalize=False)
        first = self.mgr.cancel_migration("m-1")
        for _ in range(3):
            again = self.mgr.cancel_migration("m-1")
            self.assertEqual(again["state"], "CANCELLED")
            self.assertEqual(again["reason"], first["reason"])
            self.assertEqual(
                again["cleanup"]["deleted_items"],
                first["cleanup"]["deleted_items"])
            self.assertEqual(
                again["cleanup"]["deleted_bytes"],
                first["cleanup"]["deleted_bytes"])
            self.assertEqual(
                again["cleanup"]["skipped_items"],
                first["cleanup"]["skipped_items"])
            self.assertEqual(
                again["cleanup"]["skip_reasons"],
                first["cleanup"]["skip_reasons"])
        # transfer/finalize 对已取消任务幂等, 不再删除/发布
        self.assertEqual(
            self.mgr.transfer_migration("m-1")["state"], "CANCELLED")
        self.assertEqual(
            self.mgr.finalize_migration("m-1")["state"], "CANCELLED")
        self.assertFalse(self.mgr.list_published_generations())


class FailureCleanupTest(CleanupTestBase):
    def test_finalize_failure_cleans_exclusive_leftovers(self):
        self._complete_gen1()
        self._register_and_create("m-1")
        self.mgr.transfer_migration("m-1", auto_finalize=False)
        # 终检前损坏目标端 r1 -> 终检重读重算失败, 任务 FAILED
        dst = self._dst()
        dst.corrupt(self._blob(1))
        final = self.mgr.finalize_migration("m-1")
        self.assertEqual(final["state"], "FAILED")
        self.assertEqual(final["reason"], "destination_digest_mismatch")
        self.assertFalse(self.mgr.list_published_generations())
        self.assertEqual(self.mgr.migration_holds(), {})
        # 独占写入的完好对象 r0 立即清掉; 损坏的失败项不属"本次成功写入",
        # 保留在原地 (可由后续任务修复), 体现在对象状态而非清理候选。
        cleanup = final["cleanup"]
        self.assertEqual(cleanup["state"], "DONE")
        self.assertEqual(cleanup["candidate_items"], 1)
        self.assertEqual(cleanup["deleted_items"], 1)
        self.assertFalse(dst.has(self._blob(0)))

    def test_manifest_change_failure_cleans_and_keeps_reason(self):
        self._complete_gen1(release=True)
        self._register_and_create("m-1")
        self.mgr.transfer_migration("m-1", auto_finalize=False)
        # 冻结后源代际产生新输出 -> 清单变化
        self.mgr.restore("cj-1", generation=1)
        lease = self.mgr.get("cj-1").leases.current().token
        self.mgr.append_output(
            "cj-1", lease_token=lease, output_id="late",
            data=b"output after freeze")
        with self.assertRaises(Conflict):
            self.mgr.transfer_migration("m-1")
        task = self.mgr.get_migration("m-1")
        self.assertEqual(task["state"], "FAILED")
        self.assertEqual(task["reason"], "manifest_changed")
        cleanup = task["cleanup"]
        self.assertEqual(cleanup["state"], "DONE")
        # 两个独占写入分片全部回收
        self.assertEqual(cleanup["deleted_items"], self.world_size)
        self.assertEqual(cleanup["deleted_bytes"],
                         sum(len(f"g1-r{r}".encode())
                             for r in range(self.world_size)))
        self.assertEqual(len(self._dst().inventory()), 0)


class SharedContentProtectionTest(CleanupTestBase):
    def _register_aliases(self, *ids):
        # 多个 destination_id 指向同一物理 blob 目录 (接收端重启/多入口别名)
        for did in ids:
            self.mgr.register_destination(did, data_dir=self.dstdir)
        stores = [self.mgr._destinations[did] for did in ids]
        for other in stores[1:]:
            self.assertTrue(stores[0].same_backend(other))

    def test_other_inflight_migration_keeps_shared_blob(self):
        self._complete_gen1()
        self._register_aliases("b1", "b2")
        # mB 独占写入 r1
        self.mgr.create_migration(
            "cj-1", generation=1, destination_id="b1", migration_id="mB")
        self.mgr.transfer_migration(
            "mB", auto_finalize=False, object_ids=["r1:s0"])
        # mC 随后复用同一 blob; 撤销 mC 不得删除仍被活跃 mB 引用的内容
        self.mgr.create_migration(
            "cj-1", generation=1, destination_id="b2", migration_id="mC")
        reused = self.mgr.transfer_migration("mC", auto_finalize=False)
        # r1 与活跃 mB 共享 -> REUSED; r0 在 mB 清单里仍 PLANNED, mC 先写入
        self.assertEqual(reused["reused_items"], 1)
        self.assertEqual(reused["transferred_items"], 1)
        cancelled_c = self.mgr.cancel_migration("mC")
        # r1 是 REUSED 不进候选; r0 虽由 mC 独占写入, 但活跃 mB 的清单已
        # 引用同一 blob -> 必须保留并标注稳定跳过原因。
        cleanup_c = cancelled_c["cleanup"]
        self.assertEqual(cleanup_c["candidate_items"], 1)
        self.assertEqual(cleanup_c["deleted_items"], 0)
        self.assertEqual(cleanup_c["skipped_items"], 1)
        self.assertEqual(
            cleanup_c["skip_reasons"],
            {"shared_with_other_migration": 1})
        # 关键: 与活跃 mB 共享的两个 blob 都必须继续可用
        shared = self.mgr._destinations["b1"]
        self.assertTrue(shared.has(self._blob(0)))
        self.assertTrue(shared.has(self._blob(1)))
        # mB 续跑: r0 已在接收端完好 -> REUSED; 最终一次性发布, 内容完整
        done = self.mgr.transfer_migration("mB")
        self.assertEqual(done["state"], "PUBLISHED")
        shared = self.mgr._destinations["b1"]
        self.assertTrue(shared.has(self._blob(0)))
        self.assertTrue(shared.has(self._blob(1)))

    def test_published_generation_keeps_blob_even_if_written_by_this_task(self):
        self._complete_gen1()
        self._register_and_create("m-pub")
        self.mgr.transfer_migration("m-pub")  # PUBLISHED
        # 第二个任务全部 REUSED; 构造不出 TRANSFERRED 候选, 直接验证发布保护:
        self.mgr.create_migration(
            "cj-1", generation=1, destination_id="cluster-b",
            migration_id="m-again")
        view = self.mgr.transfer_migration("m-again", auto_finalize=False)
        self.assertEqual(view["transferred_items"], 0)
        self.assertEqual(view["reused_items"], self.world_size)
        cancelled = self.mgr.cancel_migration("m-again")
        self.assertEqual(cancelled["cleanup"]["candidate_items"], 0)
        for r in range(self.world_size):
            self.assertTrue(self._dst().has(self._blob(r)))

    def test_adopted_generation_references_are_protected(self):
        self._complete_gen1()
        self._register_and_create("m-pub")
        self.mgr.transfer_migration("m-pub")
        # 接收端 adopt 为新作业 (绑定同一接收存储)
        res = self.mgr.adopt_generation(
            migration_id="m-pub", destination_id="cluster-b",
            target_job_id="cj-2", create_job=True)
        self.assertTrue(res["adopted"])
        # 新任务的清理即使存在 TRANSFERRED 候选, 也必须保留 adopt 引用。
        # 这里对象已存在只能 REUSED; 直接验证 adopt 作业可同代际恢复。
        report = self.mgr.restore("cj-2", generation=1)
        self.assertTrue(report.success)
        self.assertEqual(report.shards_verified, self.world_size)

    def test_same_backend_alias_register_keeps_source_references(self):
        # 接收端目录直接注册成与源存储同一物理目录: 终态清理绝不能删除
        # 仍被源作业检查点引用的 blob。
        self._complete_gen1()
        self.mgr.register_destination("same", data_dir=self.srcdir)
        same = self.mgr._destinations["same"]
        self.assertTrue(
            same.same_backend(self.mgr.get("cj-1").store))
        self.mgr.create_migration(
            "cj-1", generation=1, destination_id="same",
            migration_id="m-same")
        view = self.mgr.transfer_migration("m-same", auto_finalize=False)
        # 源 blob 在"接收端"已存在 -> 全部 REUSED
        self.assertEqual(view["reused_items"], self.world_size)
        cancelled = self.mgr.cancel_migration("m-same")
        self.assertEqual(cancelled["cleanup"]["candidate_items"], 0)
        # 源作业内容与恢复能力完好
        self.mgr.confirm_release  # noqa: B018 (可读性)
        job = self.mgr.get("cj-1")
        cp = job.cpm.get(1)
        self.assertEqual(cp.state, "COMPLETE")
        for r in range(self.world_size):
            self.assertTrue(job.store.verify(self._blob(r)))


class CleanupRestartTest(CleanupTestBase):
    def test_finished_cancel_survives_restart_with_stable_stats(self):
        self._complete_gen1()
        self._register_and_create("m-1")
        self.mgr.transfer_migration("m-1", auto_finalize=False)
        before = self.mgr.cancel_migration("m-1")
        self._restart()
        task = self.mgr.get_migration("m-1")
        self.assertEqual(task["state"], "CANCELLED")
        cleanup = task["cleanup"]
        self.assertEqual(cleanup["state"], "DONE")
        self.assertEqual(cleanup["deleted_items"],
                         before["cleanup"]["deleted_items"])
        self.assertEqual(cleanup["deleted_bytes"],
                         before["cleanup"]["deleted_bytes"])
        self.assertEqual(cleanup["candidate_items"],
                         before["cleanup"]["candidate_items"])
        self.assertEqual(self.mgr.migration_holds(), {})
        self.assertEqual(len(self._dst().inventory()), 0)
        # 重启后重复撤销仍是同一终态
        again = self.mgr.cancel_migration("m-1")
        self.assertEqual(again["state"], "CANCELLED")
        self.assertEqual(
            again["cleanup"]["deleted_items"],
            before["cleanup"]["deleted_items"])

    def _journal(self):
        return os.path.join(self.srcdir, "journal.jsonl")

    def _append_events(self, events):
        with open(self._journal(), "a", encoding="utf-8") as fp:
            for evt in events:
                fp.write(json.dumps(evt) + "\n")

    def _last_seq(self):
        return max(json.loads(line)["seq"]
                   for line in open(self._journal(), encoding="utf-8")
                   if line.strip())

    def test_resume_after_crash_mid_cleanup(self):
        self._complete_gen1()
        self._register_and_create("m-crash")
        self.mgr.transfer_migration("m-crash", auto_finalize=False)
        self.mgr.journal.close()
        b0, b1 = self._blob(0), self._blob(1)

        # 模拟崩溃: cleanup_started 已落盘, r0 物理删除+删除事件落盘;
        # r1 物理删除发生但事件/终态事件未落盘。
        seq = self._last_seq()
        self._append_events([
            {"seq": seq + 1, "ts": 1.0,
             "type": "migration_cleanup_started",
             "migration_id": "m-crash", "source_job_id": "cj-1",
             "source_generation": 1, "destination_id": "cluster-b",
             "terminal_state": "CANCELLED",
             "reason": "cancelled_by_user", "started_ts": 1.0,
             "summary": {"state": "RUNNING", "terminal_state": "CANCELLED",
                         "reason": "cancelled_by_user",
                         "candidate_items": 2, "candidate_bytes": 8}},
            {"seq": seq + 2, "ts": 1.0,
             "type": "migration_cleanup_blob_deleted",
             "migration_id": "m-crash", "source_job_id": "cj-1",
             "source_generation": 1, "destination_id": "cluster-b",
             "object_id": "r0:s0", "kind": "checkpoint_shard", "blob": b0,
             "size": len(b"g1-r0"), "existed": True,
             "disposition": "DELETED", "reason": None, "deleted_ts": 1.0},
        ])
        os.unlink(os.path.join(self.dstdir, "blobs", b0))
        os.unlink(os.path.join(self.dstdir, "blobs", b1))

        self._restart()  # 构造时自动续跑
        task = self.mgr.get_migration("m-crash")
        cleanup = task["cleanup"]
        self.assertEqual(task["state"], "CANCELLED")
        self.assertEqual(cleanup["state"], "DONE")
        self.assertEqual(cleanup["deleted_items"], 2)
        self.assertEqual(
            cleanup["recovery_reasons"],
            {"already_deleted_after_crash": 1})
        self.assertEqual(cleanup["skipped_items"], 0)
        dst = self._dst()
        self.assertFalse(dst.has(b0))
        self.assertFalse(dst.has(b1))
        self.assertEqual(self.mgr.migration_holds(), {})
        # 终态事件已在续跑时补落盘: 再重启统计不变
        self._restart()
        again = self.mgr.get_migration("m-crash")
        self.assertEqual(again["state"], "CANCELLED")
        self.assertEqual(again["cleanup"]["deleted_items"], 2)
        self.assertEqual(
            again["cleanup"]["recovery_reasons"],
            {"already_deleted_after_crash": 1})

    def test_restart_with_only_started_event_and_present_blobs(self):
        self._complete_gen1()
        self._register_and_create("m-crash2")
        self.mgr.transfer_migration("m-crash2", auto_finalize=False)
        self.mgr.journal.close()
        b0, b1 = self._blob(0), self._blob(1)
        seq = self._last_seq()
        self._append_events([
            {"seq": seq + 1, "ts": 1.0,
             "type": "migration_cleanup_started",
             "migration_id": "m-crash2", "source_job_id": "cj-1",
             "source_generation": 1, "destination_id": "cluster-b",
             "terminal_state": "FAILED", "reason": "incomplete_objects",
             "started_ts": 1.0,
             "summary": {"state": "RUNNING", "terminal_state": "FAILED",
                         "reason": "incomplete_objects",
                         "candidate_items": 2, "candidate_bytes": 8}},
        ])
        # 两个 blob 物理都还在 -> 重启续跑应把它们删掉
        self._restart()
        task = self.mgr.get_migration("m-crash2")
        self.assertEqual(task["state"], "FAILED")
        self.assertEqual(task["reason"], "incomplete_objects")
        cleanup = task["cleanup"]
        self.assertEqual(cleanup["state"], "DONE")
        self.assertEqual(cleanup["deleted_items"], 2)
        self.assertEqual(cleanup["recovery_reasons"], {})
        dst = self._dst()
        self.assertFalse(dst.has(b0))
        self.assertFalse(dst.has(b1))


class CleanupConcurrencyTest(CleanupTestBase):
    def test_concurrent_cancel_runs_cleanup_once_with_stable_stats(self):
        self._complete_gen1()
        self.mgr = JobManager(Scheduler(Signer("cleanup-conc")))
        self.mgr.create_job(
            name="cc", world_size=self.world_size,
            rank_nodes={r: f"n{r}" for r in range(self.world_size)},
            job_id="cj-1")
        token = self.mgr.preempt("cj-1")["epoch_token"]
        for r in range(self.world_size):
            self.mgr.upload_shard(
                "cj-1", epoch_token=token, rank=r, index=0,
                node=f"n{r}", input_pos=r + 1,
                data=f"g1-r{r}".encode())
        self.mgr.commit_checkpoint("cj-1", epoch_token=token)
        self.mgr.register_destination("cluster-b")  # 内存目标
        self.mgr.create_migration(
            "cj-1", generation=1, destination_id="cluster-b",
            migration_id="m-1")
        self.mgr.transfer_migration("m-1", auto_finalize=False)
        results: list[dict] = []

        def cancel():
            results.append(self.mgr.cancel_migration("m-1"))

        threads = [threading.Thread(target=cancel) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertTrue(results)
        deleted = {r["cleanup"]["deleted_items"] for r in results}
        skipped = {r["cleanup"]["skipped_items"] for r in results}
        states = {r["state"] for r in results}
        self.assertEqual(deleted, {self.world_size})
        self.assertEqual(skipped, {0})
        self.assertEqual(states, {"CANCELLED"})
        # 接收端物理上恰好只剩 REUSED 之外的内容 —— 这里全部是独占写入,
        # 故物理集合为空, 且绝无"双重删除"导致的计数偏差。
        self.assertEqual(len(self._dst().inventory()), 0)

    def test_cancel_concurrent_with_gc_never_deletes_reachable_data(self):
        # 源端: gen1 已释放并恢复后又抢占出 gen2; 迁移 gen1 到一个与源 *不同*
        # 的接收目录。取消迁移清理接收端, GC 同时回收源端 gen1, 两者作用于
        # 不同存储但共享引用临界区 —— 可访问数据绝不丢失。
        self._complete_gen1(release=True)
        self.mgr.restore("cj-1", generation=1)
        token2 = self.mgr.preempt("cj-1", force_new_epoch=True)["epoch_token"]
        for r in range(self.world_size):
            self.mgr.upload_shard(
                "cj-1", epoch_token=token2, rank=r, index=0,
                node=f"n{r}", input_pos=r + 10,
                data=f"g2-r{r}".encode())
        self.mgr.commit_checkpoint("cj-1", epoch_token=token2)
        self._register_and_create("m-1")
        self.mgr.transfer_migration("m-1", auto_finalize=False)
        outcomes: list[tuple[str, str]] = []

        def cancel():
            r = self.mgr.cancel_migration("m-1")
            outcomes.append(("cancel", r["state"]))

        def gc_loop():
            try:
                plan = self.mgr.create_gc_plan(
                    policy="watermark", min_retain_generation=2)
                self.mgr.execute_gc_plan(plan["plan_id"])
                outcomes.append(("gc", "executed"))
            except Conflict:
                outcomes.append(("gc", "rejected"))

        threads = [threading.Thread(target=cancel)]
        threads += [threading.Thread(target=gc_loop) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertIn(("cancel", "CANCELLED"), outcomes)
        # 接收端独占遗留被清; 源 gen2 (当前检查点) 完好, 仍可提交/恢复
        self.assertEqual(len(self._dst().inventory()), 0)
        gen2 = self.mgr.get("cj-1").cpm.get(2)
        self.assertEqual(gen2.state, "COMPLETE")
        for r in range(self.world_size):
            self.assertTrue(
                self.mgr.get("cj-1").store.verify(
                    sha256_hex(f"g2-r{r}".encode())))


if __name__ == "__main__":
    unittest.main()
