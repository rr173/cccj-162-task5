"""迁移终态遗留清扫: 撤销/校验终止时只删除本任务独占的接收端遗留。

覆盖: 已有同摘要内容保留、其他进行中搬运保留、已发布恢复版本保留、
服务重启/重复撤销/崩溃窗口恢复、与 GC 并发、HTTP API 与页面展示。
"""

import json
import os
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from hpc_cr import JobManager
from hpc_cr import jobs as jobs_mod
from hpc_cr.api import build_server
from hpc_cr.epochs import Scheduler
from hpc_cr.errors import Conflict
from hpc_cr.security import Signer, sha256_hex


class CleanupTestBase(unittest.TestCase):
    world_size = 2

    def setUp(self):
        self.srcdir = tempfile.mkdtemp(prefix="clean-src-")
        self.dstdir = tempfile.mkdtemp(prefix="clean-dst-")
        self.secret = "cleanup-secret"
        self.mgr = JobManager(
            Scheduler(Signer(self.secret)), data_dir=self.srcdir)
        self.mgr.create_job(
            name="clean-job", world_size=self.world_size,
            shards_per_rank=1,
            rank_nodes={r: f"n{r}" for r in range(self.world_size)},
            job_id="cj-1")

    def tearDown(self):
        self.mgr.journal.close()
        shutil.rmtree(self.srcdir, ignore_errors=True)
        shutil.rmtree(self.dstdir, ignore_errors=True)

    def _complete(self, gen_payloads, release=False):
        """完成一代检查点; gen_payloads: {rank: bytes}。"""
        pre = self.mgr.preempt("cj-1", force_new_epoch=True)
        token = pre["epoch_token"]
        for r, data in gen_payloads.items():
            self.mgr.upload_shard(
                "cj-1", epoch_token=token, rank=r, index=0,
                node=f"n{r}", input_pos=r + 1, data=data)
        self.mgr.commit_checkpoint("cj-1", epoch_token=token)
        if release:
            self.mgr.confirm_release("cj-1", epoch_token=token)
        return token

    def _gen1(self, payloads=None, release=False):
        payloads = payloads or {
            0: b"shard-A", 1: b"shard-B"}
        return self._complete(payloads, release=release)

    def _register(self, dest="cluster-b"):
        self.mgr.register_destination(dest, data_dir=self.dstdir)
        return self.mgr._destinations[dest]

    def _create(self, generation, mig_id, dest="cluster-b"):
        return self.mgr.create_migration(
            "cj-1", generation=generation, destination_id=dest,
            migration_id=mig_id)

    def _dst_blob(self, data: bytes) -> str:
        return sha256_hex(data)


class OrphanSweepTest(CleanupTestBase):
    def test_cancel_sweeps_exclusive_leftovers(self):
        self._gen1()
        dst = self._register()
        self._create(1, "m-1")
        self.mgr.transfer_migration(
            "m-1", auto_finalize=False, object_ids=["r0:s0"])
        self.assertTrue(dst.has(self._dst_blob(b"shard-A")))

        cancelled = self.mgr.cancel_migration("m-1")
        self.assertEqual(cancelled["state"], "CANCELLED")
        cleanup = cancelled["cleanup"]
        self.assertIsNotNone(cleanup)
        self.assertEqual(cleanup["trigger"], "cancelled")
        self.assertFalse(cleanup["crash_recovery"])
        self.assertEqual(cleanup["written_items"], 1)
        self.assertEqual(cleanup["deleted_items"], 1)
        self.assertEqual(cleanup["deleted_bytes"], len(b"shard-A"))
        self.assertEqual(cleanup["kept_items"], 0)
        self.assertEqual(cleanup["recovered_items"], 0)
        # 本任务写入的遗留已物理删除
        self.assertFalse(dst.has(self._dst_blob(b"shard-A")))
        self.assertEqual(dst.blob_count(), 0)
        # 终态统计稳定可查询
        view = self.mgr.get_migration("m-1")
        self.assertEqual(view["cleanup"], cleanup)
        self.assertEqual(view["written_blobs"], [self._dst_blob(b"shard-A")])

    def test_cancel_without_transfer_is_noop_sweep(self):
        self._gen1()
        dst = self._register()
        self._create(1, "m-1")
        cancelled = self.mgr.cancel_migration("m-1")
        cleanup = cancelled["cleanup"]
        self.assertEqual(cleanup["written_items"], 0)
        self.assertEqual(cleanup["deleted_items"], 0)
        self.assertEqual(dst.blob_count(), 0)

    def test_repeated_cancel_is_idempotent_and_stable(self):
        self._gen1()
        self._register()
        self._create(1, "m-1")
        self.mgr.transfer_migration("m-1", auto_finalize=False)
        first = self.mgr.cancel_migration("m-1")
        events_before = len(self.mgr.journal.all())
        again = self.mgr.cancel_migration("m-1")
        third = self.mgr.cancel_migration("m-1", reason="another_reason")
        # 重复撤销: 不新增任何事件, 终态统计逐字节一致 (原因不轮换)
        self.assertEqual(len(self.mgr.journal.all()), events_before)
        self.assertEqual(again["cleanup"], first["cleanup"])
        self.assertEqual(third["cleanup"], first["cleanup"])
        self.assertEqual(third["reason"], first["reason"])
        clean_events = [
            e for e in self.mgr.journal.all()
            if e["type"] == "migration_orphan_cleaned"]
        self.assertEqual(len(clean_events), 1)

    def test_preexisting_same_digest_content_is_never_a_candidate(self):
        self._gen1()
        dst = self._register()
        # 接收端已有同摘要内容 (例如其他集群早前迁入的共享分片)
        dst.put(b"shard-A")
        self._create(1, "m-1")
        result = self.mgr.transfer_migration("m-1", auto_finalize=False)
        self.assertEqual(result["reused_items"], 1)
        self.assertEqual(result["transferred_items"], 1)
        cancelled = self.mgr.cancel_migration("m-1")
        cleanup = cancelled["cleanup"]
        # 只有本任务真正写入的 shard-B 进入候选并被删除;
        # 已有同摘要内容 shard-A 连候选都不是, 必须继续可用
        self.assertEqual(cleanup["written_items"], 1)
        self.assertEqual(cleanup["deleted_items"], 1)
        self.assertEqual(cleanup["kept_items"], 0)
        self.assertTrue(dst.verify(self._dst_blob(b"shard-A")))
        self.assertFalse(dst.has(self._dst_blob(b"shard-B")))

    def test_failed_migration_sweeps_leftovers(self):
        self._gen1(release=True)
        dst = self._register()
        self._create(1, "m-1")
        self.mgr.transfer_migration("m-1", auto_finalize=False)
        # 冻结后源代际产生新输出 -> 清单变化 -> 校验终止 (FAILED)
        self.mgr.restore("cj-1", generation=1)
        lease = self.mgr.get("cj-1").leases.current().token
        self.mgr.append_output(
            "cj-1", lease_token=lease, output_id="late", data=b"late-output")
        with self.assertRaises(Conflict):
            self.mgr.transfer_migration("m-1")
        task = self.mgr.get_migration("m-1")
        self.assertEqual(task["state"], "FAILED")
        self.assertEqual(task["reason"], "manifest_changed")
        cleanup = task["cleanup"]
        self.assertEqual(cleanup["trigger"], "failed")
        self.assertEqual(cleanup["written_items"], self.world_size)
        self.assertEqual(cleanup["deleted_items"], self.world_size)
        self.assertEqual(dst.blob_count(), 0)

    def test_failed_finalize_sweeps_and_records_reason(self):
        self._gen1()
        dst = self._register()
        self._create(1, "m-1")
        self.mgr.transfer_migration("m-1", auto_finalize=False)
        # 目标端 bit-rot -> 终检摘要错误 -> FAILED
        dst.corrupt(self._dst_blob(b"shard-A"))
        final = self.mgr.finalize_migration("m-1")
        self.assertEqual(final["state"], "FAILED")
        self.assertEqual(final["reason"], "destination_digest_mismatch")
        cleanup = final["cleanup"]
        # 两个 blob 都是本任务写入; 损坏的 shard-A 也在写入集合中,
        # 一并按独占遗留删除 (损坏内容本来也不可用)
        self.assertEqual(cleanup["deleted_items"], self.world_size)
        self.assertEqual(dst.blob_count(), 0)


class ProtectionTest(CleanupTestBase):
    def _two_generations_shared_shard(self):
        """gen1=[A,B], gen2=[A,C]: rank0 内容跨代际相同 (同摘要)。"""
        self._gen1({0: b"shard-A", 1: b"shard-B"}, release=True)
        self.mgr.restore("cj-1", generation=1)
        self._complete({0: b"shard-A", 1: b"shard-C"})

    def test_other_active_migration_content_is_kept(self):
        self._two_generations_shared_shard()
        dst = self._register()
        self._create(1, "m-1")
        self._create(2, "m-2")
        # m-1 全部传完 (写入 A、B), m-2 尚未开始
        self.mgr.transfer_migration("m-1", auto_finalize=False)
        cancelled = self.mgr.cancel_migration("m-1")
        cleanup = cancelled["cleanup"]
        kept = {k["blob"]: k["reason"] for k in cleanup["kept"]}
        # A 是进行中搬运 m-2 的对象 -> 保留; B 与 m-2 无关 -> 独占遗留删除
        self.assertEqual(kept, {self._dst_blob(b"shard-A"): "active_migration"})
        self.assertEqual(cleanup["deleted_items"], 1)
        self.assertTrue(dst.has(self._dst_blob(b"shard-A")))
        self.assertFalse(dst.has(self._dst_blob(b"shard-B")))
        # 进行中的搬运继续可用: m-2 复用 A 并成功发布
        done = self.mgr.transfer_migration("m-2")
        self.assertEqual(done["state"], "PUBLISHED")
        self.assertEqual(done["reused_items"], 1)
        self.assertTrue(dst.verify(self._dst_blob(b"shard-A")))
        self.assertTrue(dst.verify(self._dst_blob(b"shard-C")))

    def test_published_generation_content_is_kept(self):
        self._two_generations_shared_shard()
        dst = self._register()
        # m-2 先只传 A (写入集合={A}); m-1 随后复用 A 并发布 gen1
        self._create(2, "m-2")
        self.mgr.transfer_migration(
            "m-2", auto_finalize=False, object_ids=["r0:s0"])
        self._create(1, "m-1")
        published = self.mgr.transfer_migration("m-1")
        self.assertEqual(published["state"], "PUBLISHED")
        self.assertEqual(published["reused_items"], 1)
        # m-2 校验终止前撤销: A 已被已发布恢复版本引用 -> 保留
        cancelled = self.mgr.cancel_migration("m-2")
        cleanup = cancelled["cleanup"]
        kept = {k["blob"]: k["reason"] for k in cleanup["kept"]}
        self.assertEqual(
            kept, {self._dst_blob(b"shard-A"): "published_generation"})
        self.assertEqual(cleanup["deleted_items"], 0)
        self.assertTrue(dst.verify(self._dst_blob(b"shard-A")))
        # 已发布代际仍可 adopt + 同代际恢复
        recv_dir = tempfile.mkdtemp(prefix="clean-recv-")
        recv = JobManager(Scheduler(Signer("recv-secret")),
                          data_dir=recv_dir)
        try:
            recv.register_destination("cluster-b", data_dir=self.dstdir)
            pub = self.mgr.list_published_generations("cluster-b")[0]
            res = recv.adopt_generation(
                manifest=pub["manifest"], destination_id="cluster-b")
            self.assertTrue(res["adopted"])
            report = recv.restore("cj-1", generation=1)
            self.assertTrue(report.success)
        finally:
            recv.journal.close()
            shutil.rmtree(recv_dir, ignore_errors=True)

    def test_shared_write_is_not_exclusive_and_survives(self):
        """双方共享的内容不属于任何一方的独占遗留, 终态后仍可用。"""
        self._two_generations_shared_shard()
        dst = self._register()
        self._create(1, "m-1")
        self._create(2, "m-2")
        # m-1 写入 A、B; m-2 复用 A、写入 C
        self.mgr.transfer_migration("m-1", auto_finalize=False)
        self.mgr.transfer_migration("m-2", auto_finalize=False)
        # 两个任务都撤销: A 不是任何一方的独占遗留 (m-1 删时 m-2 活跃,
        # m-2 根本没写过 A), 按规格保留且继续可用
        c1 = self.mgr.cancel_migration("m-1")["cleanup"]
        c2 = self.mgr.cancel_migration("m-2")["cleanup"]
        self.assertEqual(c1["kept_items"], 1)   # A 因 m-2 活跃保留
        self.assertEqual(c1["deleted_items"], 1)  # B 独占删除
        self.assertEqual(c2["written_items"], 1)  # m-2 只写过 C
        self.assertEqual(c2["deleted_items"], 1)  # C 独占删除
        self.assertTrue(dst.verify(self._dst_blob(b"shard-A")))
        self.assertFalse(dst.has(self._dst_blob(b"shard-B")))
        self.assertFalse(dst.has(self._dst_blob(b"shard-C")))
        # 后续新任务仍可复用 A
        self._create(1, "m-3")
        done = self.mgr.transfer_migration("m-3")
        self.assertEqual(done["state"], "PUBLISHED")
        self.assertEqual(done["reused_items"], 1)


class RestartRecoveryTest(CleanupTestBase):
    def _reopen(self):
        self.mgr.journal.close()
        self.mgr = JobManager(
            Scheduler(Signer(self.secret)), data_dir=self.srcdir)

    def test_restart_keeps_cleanup_results_queryable(self):
        self._gen1()
        dst = self._register()
        self._create(1, "m-1")
        self.mgr.transfer_migration("m-1", auto_finalize=False)
        before = self.mgr.cancel_migration("m-1")["cleanup"]
        self._reopen()
        try:
            view = self.mgr.get_migration("m-1")
            self.assertEqual(view["state"], "CANCELLED")
            self.assertEqual(view["cleanup"], before)
            # 重启后重复撤销: 同一份统计, 不产生新事件
            again = self.mgr.cancel_migration("m-1")
            self.assertEqual(again["cleanup"], before)
            clean_events = [
                e for e in self.mgr.journal.all()
                if e["type"] == "migration_orphan_cleaned"]
            self.assertEqual(len(clean_events), 1)
            dst2 = self.mgr._destinations["cluster-b"]
            self.assertEqual(dst2.blob_count(), 0)
        finally:
            pass

    def _truncate_last_event(self, expected_type):
        """模拟崩溃: 丢掉最后一条 (未对外可见的) 事件; 不重建管理器。"""
        self.mgr.journal.close()
        path = os.path.join(self.srcdir, "journal.jsonl")
        with open(path, "r", encoding="utf-8") as fp:
            lines = [ln for ln in fp.read().splitlines() if ln.strip()]
        last = json.loads(lines[-1])
        assert last["type"] == expected_type, last["type"]
        with open(path, "w", encoding="utf-8") as fp:
            fp.write("\n".join(lines[:-1]) + "\n")

    def test_crash_before_cleanup_event_recovers_on_restart(self):
        self._gen1()
        dst = self._register()
        self._create(1, "m-1")
        self.mgr.transfer_migration("m-1", auto_finalize=False)
        self.mgr.cancel_migration("m-1")
        # 崩溃窗口: 终态事件已落盘, 清扫事件与物理删除全部丢失
        self._truncate_last_event("migration_orphan_cleaned")
        dst.put(b"shard-A")
        dst.put(b"shard-B")
        self._reopen()
        # 重启补扫: 同一规则重新执行, 统计带 crash_recovery 标记
        view = self.mgr.get_migration("m-1")
        cleanup = view["cleanup"]
        self.assertIsNotNone(cleanup)
        self.assertTrue(cleanup["crash_recovery"])
        self.assertEqual(cleanup["deleted_items"], self.world_size)
        self.assertEqual(cleanup["recovered_items"], 0)
        self.assertEqual(
            self.mgr._destinations["cluster-b"].blob_count(), 0)
        # 补扫结果同样只落盘一次
        clean_events = [
            e for e in self.mgr.journal.all()
            if e["type"] == "migration_orphan_cleaned"]
        self.assertEqual(len(clean_events), 1)

    def test_crash_mid_sweep_counts_already_deleted(self):
        self._gen1()
        dst = self._register()
        self._create(1, "m-1")
        self.mgr.transfer_migration("m-1", auto_finalize=False)
        self.mgr.cancel_migration("m-1")
        # 崩溃窗口: 清扫事件丢失; 物理删除只完成一半 (A 已删, B 还在)
        self._truncate_last_event("migration_orphan_cleaned")
        dst.put(b"shard-B")
        self._reopen()
        view = self.mgr.get_migration("m-1")
        cleanup = view["cleanup"]
        self.assertTrue(cleanup["crash_recovery"])
        self.assertEqual(cleanup["deleted_items"], 1)   # B 本次删除
        self.assertEqual(cleanup["recovered_items"], 1)  # A 崩溃前已删
        self.assertEqual(
            cleanup["recovery_reasons"], {"already_deleted_after_crash": 1})
        self.assertEqual(
            self.mgr._destinations["cluster-b"].blob_count(), 0)
        # 再次重启: 统计保持稳定, 不会重复补扫
        self._reopen()
        self.assertEqual(
            self.mgr.get_migration("m-1")["cleanup"], cleanup)


class GcConcurrencyTest(CleanupTestBase):
    def _released_gen1_and_gen2(self):
        self._gen1(release=True)
        self.mgr.restore("cj-1", generation=1)
        self._complete({0: b"gen2-A", 1: b"gen2-B"})

    def test_cleanup_event_does_not_advance_gc_watermark(self):
        self.assertNotIn(
            jobs_mod.MIG_ORPHAN_CLEANED, jobs_mod._REFERENCE_EVENTS)
        self._released_gen1_and_gen2()
        self._register()
        self._create(1, "m-1")
        self.mgr.transfer_migration(
            "m-1", auto_finalize=False, object_ids=["r0:s0"])
        self.mgr.cancel_migration("m-1")
        # 计划在取消+清扫之后生成; 重复撤销不产生事件, 计划保持可执行
        plan = self.mgr.create_gc_plan(
            policy="watermark", min_retain_generation=2)
        self.mgr.cancel_migration("m-1")
        self.mgr.cancel_migration("m-1")
        result = self.mgr.execute_gc_plan(plan["plan_id"])
        self.assertEqual(result["state"], "COMPLETED")
        self.assertEqual(result["deleted_items"], self.world_size)

    def test_concurrent_cancel_gc_and_transfer_are_safe(self):
        self._released_gen1_and_gen2()
        dst = self._register()
        self._create(1, "m-1")
        errors, outcomes = [], []

        def cancel():
            try:
                outcomes.append(
                    ("cancel", self.mgr.cancel_migration("m-1")["state"]))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def transfer():
            try:
                outcomes.append(
                    ("transfer",
                     self.mgr.transfer_migration("m-1")["state"]))
            except Conflict:
                outcomes.append(("transfer", "conflict"))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def gc_loop():
            try:
                plan = self.mgr.create_gc_plan(
                    policy="watermark", min_retain_generation=2)
                self.mgr.execute_gc_plan(plan["plan_id"])
                outcomes.append(("gc", "executed"))
            except Conflict:
                outcomes.append(("gc", "rejected"))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = ([threading.Thread(target=cancel) for _ in range(4)]
                   + [threading.Thread(target=transfer) for _ in range(2)]
                   + [threading.Thread(target=gc_loop) for _ in range(4)])
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertIn(("cancel", "CANCELLED"), outcomes)
        # 恰好一次取消事件与一次清扫事件
        for etype in ("migration_cancelled", "migration_orphan_cleaned"):
            self.assertEqual(
                sum(1 for e in self.mgr.journal.all()
                    if e["type"] == etype), 1)
        task = self.mgr.get_migration("m-1")
        self.assertEqual(task["state"], "CANCELLED")
        cleanup = task["cleanup"]
        self.assertIsNotNone(cleanup)
        # 不变量: 被删的一定是本次独占; 保留的必须仍然可用
        for k in cleanup["kept"]:
            self.assertTrue(dst.has(k["blob"]))
        for blob in task["written_blobs"]:
            if not any(k["blob"] == blob for k in cleanup["kept"]):
                self.assertFalse(dst.has(blob))
        self.assertEqual(
            cleanup["deleted_items"] + cleanup["kept_items"]
            + cleanup["recovered_items"],
            cleanup["written_items"])
        # 源端 gen2 数据绝不受接收端清扫影响
        for data in (b"gen2-A", b"gen2-B"):
            self.assertTrue(self.mgr.store.verify(sha256_hex(data)))


class CleanupHttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dstdir = tempfile.mkdtemp(prefix="clean-http-dst-")
        cls.server = build_server(
            JobManager(Scheduler(Signer("clean-http-secret"))),
            host="127.0.0.1", port=0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        shutil.rmtree(cls.dstdir, ignore_errors=True)

    def _req(self, method, path, body=None):
        url = f"http://127.0.0.1:{self.port}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_cleanup_visible_via_api_and_page(self):
        import base64
        mgr = self.server.manager
        mgr.create_job(name="http-clean", world_size=1,
                       rank_nodes={0: "n0"}, job_id="clean-http")
        pre = mgr.preempt("clean-http")
        mgr.upload_shard(
            "clean-http", epoch_token=pre["epoch_token"], rank=0, index=0,
            node="n0", input_pos=1, data=b"http-orphan-shard")
        mgr.commit_checkpoint("clean-http", epoch_token=pre["epoch_token"])
        st, _ = self._req("POST", "/api/migration-destinations", {
            "destination_id": "cluster-b", "data_dir": self.dstdir})
        self.assertEqual(st, 201)
        st, mig = self._req("POST", "/api/migrations", {
            "job_id": "clean-http", "generation": 1,
            "destination_id": "cluster-b", "migration_id": "clean-1"})
        self.assertEqual(st, 201)
        st, running = self._req(
            "POST", "/api/migrations/clean-1",
            {"action": "transfer", "auto_finalize": False})
        self.assertEqual(st, 200)
        self.assertEqual(running["transferred_items"], 1)

        st, cancelled = self._req(
            "POST", "/api/migrations/clean-1", {"action": "cancel"})
        self.assertEqual(st, 200)
        cleanup = cancelled["cleanup"]
        self.assertEqual(cleanup["deleted_items"], 1)
        self.assertEqual(cleanup["written_items"], 1)
        # 重复撤销 (HTTP): 同一份终态统计
        st, again = self._req(
            "POST", "/api/migrations/clean-1", {"action": "cancel"})
        self.assertEqual(st, 200)
        self.assertEqual(again["cleanup"], cleanup)

        st, view = self._req("GET", "/api/migrations/clean-1")
        self.assertEqual(st, 200)
        self.assertEqual(view["cleanup"], cleanup)
        st, page = self._req("GET", "/api/jobs/clean-http/progress.json")
        self.assertEqual(st, 200)
        self.assertEqual(page["migrations"][0]["cleanup"], cleanup)

        with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/ui/jobs/clean-http",
                timeout=5) as resp:
            html = resp.read().decode()
        self.assertIn("遗留清扫", html)
        self.assertIn("删除独占遗留", html)


if __name__ == "__main__":
    unittest.main()
