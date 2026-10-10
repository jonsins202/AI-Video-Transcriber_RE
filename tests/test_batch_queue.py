"""批量队列单测：批处理入队/去重/上限、requeue/cancel 状态机、合并文档、串行 worker 顺序。"""
import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import main  # noqa: E402

URL_A = "https://example.com/a"
URL_B = "https://example.com/b"
URL_C = "https://example.com/c"


def _queued_task(url, **extra):
    """构造与 /api/process-batch 创建一致的 queued 任务。"""
    t = {
        "status": "processing",
        "progress": 0,
        "message": "已加入队列，等待处理...",
        "script": None,
        "summary": None,
        "error": None,
        "url": url,
        "batch_id": "batch-test",
        "queue_status": "queued",
        "settings": {
            "summary_language": "zh",
            "oral_cleanup": False,
            "spk_num": "",
            "api_key": "",
            "model_base_url": "",
            "model_id": "",
        },
    }
    t.update(extra)
    return t


class _BatchQueueTestCase(unittest.TestCase):
    """公共夹具：临时 TASKS_FILE/TEMP_DIR + 全新 tasks 字典 + 无操作广播。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        patchers = [
            patch.object(main, "TASKS_FILE", self.tmp_dir / "tasks.json"),
            patch.object(main, "TEMP_DIR", self.tmp_dir),
            patch.object(main, "tasks", {}),
            patch.object(main, "broadcast_task_update", self._noop_broadcast),
            patch.object(main, "_task_queue", asyncio.Queue()),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)
        main.processing_urls.clear()
        main.active_tasks.clear()
        main.sse_connections.clear()

    async def _noop_broadcast(self, task_id, task):
        pass

    def _client(self):
        from fastapi.testclient import TestClient
        return TestClient(main.app)


class BatchEndpointTests(_BatchQueueTestCase):
    """POST /api/process-batch：去重、上限、跳过、任务形状。"""

    def setUp(self):
        super().setUp()
        # 阻止 startup 钩子启动真实 worker 消费队列（测试直接驱动/断言队列内容）
        self._worker_patch = patch.object(main, "_queue_worker", self._noop_worker)
        self._worker_patch.start()
        self.addCleanup(self._worker_patch.stop)

    async def _noop_worker(self):
        return None

    def test_batch_dedup_and_skip_empty(self):
        with self._client() as client:
            resp = client.post("/api/process-batch", json={
                "items": [
                    {"url": URL_A},
                    {"url": "   "},          # 空白 → 丢弃
                    {"url": ""},             # 空 → 丢弃
                    {"url": URL_A},          # 批内重复 → skipped
                    {"url": URL_B},
                ]
            })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["tasks"]), 2)
        self.assertEqual(len(data["skipped"]), 1)
        self.assertEqual(data["skipped"], [{"url": URL_A, "reason": "批内重复链接"}])
        # 任务形状：与 /api/process-video 一致 + batch_id/queue_status/settings
        t = main.tasks[data["tasks"][0]["task_id"]]
        self.assertEqual(t["status"], "processing")
        self.assertEqual(t["progress"], 0)
        self.assertIsNone(t["script"])
        self.assertIsNone(t["summary"])
        self.assertIsNone(t["error"])
        self.assertEqual(t["url"], URL_A)
        self.assertEqual(t["batch_id"], data["batch_id"])
        self.assertEqual(t["queue_status"], "queued")
        self.assertEqual(t["settings"]["summary_language"], "zh")
        self.assertEqual(t["settings"]["oral_cleanup"], False)
        self.assertEqual(t["settings"]["spk_num"], "")
        # 未占用 processing_urls、未 spawn 任务
        self.assertNotIn(URL_A, main.processing_urls)
        self.assertNotIn(URL_B, main.processing_urls)
        self.assertEqual(main.active_tasks, {})
        # 两个任务都已入队
        self.assertEqual(main._task_queue.qsize(), 2)

    def test_batch_cap_11_returns_400(self):
        with self._client() as client:
            resp = client.post("/api/process-batch", json={
                "items": [{"url": f"https://example.com/{i}"} for i in range(11)]
            })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("最多支持 10 个", resp.json()["detail"])
        self.assertEqual(main.tasks, {})  # 未创建任何任务

    def test_batch_all_empty_returns_400(self):
        with self._client() as client:
            resp = client.post("/api/process-batch", json={
                "items": [{"url": "  "}, {"url": ""}]
            })
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(main.tasks, {})

    def test_batch_skips_url_already_processing(self):
        main.processing_urls.add(URL_A)
        with self._client() as client:
            resp = client.post("/api/process-batch", json={
                "items": [{"url": URL_A}, {"url": URL_B}]
            })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual([t["url"] for t in data["tasks"]], [URL_B])
        self.assertEqual(data["skipped"], [{"url": URL_A, "reason": "该链接正在处理中"}])

    def test_batch_settings_snapshot(self):
        with self._client() as client:
            resp = client.post("/api/process-batch", json={
                "items": [{
                    "url": URL_A,
                    "summary_language": "en",
                    "oral_cleanup": True,
                    "spk_num": "2",
                    "api_key": "sk-test",
                    "model_base_url": "http://localhost:9999/v1",
                    "model_id": "gpt-test",
                }]
            })
        self.assertEqual(resp.status_code, 200)
        t = main.tasks[resp.json()["tasks"][0]["task_id"]]
        self.assertEqual(t["settings"], {
            "summary_language": "en",
            "oral_cleanup": True,
            "spk_num": "2",
            "api_key": "sk-test",
            "model_base_url": "http://localhost:9999/v1",
            "model_id": "gpt-test",
        })


class RequeueCancelTests(_BatchQueueTestCase):
    """requeue/cancel 状态机：允许/拒绝规则 + 400 分支。"""

    def setUp(self):
        super().setUp()
        self._worker_patch = patch.object(main, "_queue_worker", self._noop_worker)
        self._worker_patch.start()
        self.addCleanup(self._worker_patch.stop)

    async def _noop_worker(self):
        return None

    def test_requeue_error_task(self):
        tid = "req-1"
        main.tasks[tid] = _queued_task(URL_A, queue_status="done", status="error", error="boom")
        with self._client() as client:
            resp = client.post(f"/api/task/{tid}/requeue")
        self.assertEqual(resp.status_code, 200)
        t = resp.json()
        self.assertEqual(t["queue_status"], "queued")
        self.assertEqual(t["status"], "processing")
        self.assertEqual(t["progress"], 0)
        self.assertEqual(t["message"], "已重新入队，等待处理...")
        self.assertIsNone(t["error"])
        self.assertEqual(main._task_queue.qsize(), 1)  # 已入队

    def test_requeue_interrupted_task(self):
        tid = "req-2"
        main.tasks[tid] = _queued_task(
            URL_A, queue_status="interrupted", message="服务重启，任务中断，请在任务列表中重新入队"
        )
        with self._client() as client:
            resp = client.post(f"/api/task/{tid}/requeue")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["queue_status"], "queued")

    def test_requeue_rejects_completed(self):
        tid = "req-3"
        main.tasks[tid] = _queued_task(URL_A, queue_status="done", status="completed", script="x")
        with self._client() as client:
            resp = client.post(f"/api/task/{tid}/requeue")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("不可重新入队", resp.json()["detail"])
        self.assertEqual(main._task_queue.qsize(), 0)  # 未入队

    def test_requeue_rejects_queued(self):
        tid = "req-4"
        with self._client() as client:
            main.tasks[tid] = _queued_task(URL_A)  # queued（startup 之后创建，避免被中断扫描改写）
            resp = client.post(f"/api/task/{tid}/requeue")
        self.assertEqual(resp.status_code, 400)

    def test_requeue_missing_404(self):
        with self._client() as client:
            resp = client.post("/api/task/ghost/requeue")
        self.assertEqual(resp.status_code, 404)

    def test_cancel_queued_task(self):
        tid = "can-1"
        with self._client() as client:
            main.tasks[tid] = _queued_task(URL_A)  # startup 之后创建，保持 queued
            resp = client.post(f"/api/task/{tid}/cancel")
        self.assertEqual(resp.status_code, 200)
        t = resp.json()
        self.assertEqual(t["queue_status"], "cancelled")
        self.assertEqual(t["status"], "cancelled")
        self.assertEqual(t["message"], "已取消")
        self.assertIsNone(t["error"])
        self.assertEqual(main._task_queue.qsize(), 0)  # 未入队

    def test_cancel_rejects_running(self):
        tid = "can-2"
        main.tasks[tid] = _queued_task(URL_A, queue_status="running")
        with self._client() as client:
            resp = client.post(f"/api/task/{tid}/cancel")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("不可取消", resp.json()["detail"])

    def test_cancel_rejects_done(self):
        tid = "can-3"
        main.tasks[tid] = _queued_task(URL_A, queue_status="done", status="completed", script="x")
        with self._client() as client:
            resp = client.post(f"/api/task/{tid}/cancel")
        self.assertEqual(resp.status_code, 400)

    def test_cancel_missing_404(self):
        with self._client() as client:
            resp = client.post("/api/task/ghost/cancel")
        self.assertEqual(resp.status_code, 404)

    def test_startup_marks_queued_as_interrupted(self):
        """startup 钩子：启动时刻不存在"还在跑"的任务——queued 与 running 残留
        （崩溃中断）一律 → interrupted（可重新入队）。"""
        main.tasks["q1"] = _queued_task(URL_A)
        main.tasks["q2"] = _queued_task(URL_B, queue_status="running")
        with self._client() as client:  # 进入上下文即触发 startup
            pass
        self.assertEqual(main.tasks["q1"]["queue_status"], "interrupted")
        self.assertIn("服务重启", main.tasks["q1"]["message"])
        self.assertEqual(main.tasks["q2"]["queue_status"], "interrupted")  # running 残留同样是死任务


class MergeTests(_BatchQueueTestCase):
    """POST /api/batch/merge：合并文档结构、失败任务列出、错误分支。"""

    def setUp(self):
        super().setUp()
        self._worker_patch = patch.object(main, "_queue_worker", self._noop_worker)
        self._worker_patch.start()
        self.addCleanup(self._worker_patch.stop)

    async def _noop_worker(self):
        return None

    def test_merge_builds_document(self):
        main.tasks["m1"] = _queued_task(
            URL_A, queue_status="done", status="completed",
            video_title="视频甲",
            script="# 视频甲\n\n第一段正文\n\nsource: https://example.com/a\n",
            summary="摘要甲\n\nsource: https://example.com/a\n",
            completed_at="2026-09-07T10:00:00", total_duration=12.5,
        )
        main.tasks["m2"] = _queued_task(
            URL_B, queue_status="done", status="completed",
            video_title="视频乙",
            script="# 视频乙\n\n第二段正文\n\nsource: https://example.com/b\n",
            summary="摘要乙\n\nsource: https://example.com/b\n",
            completed_at="2026-09-07T10:05:00", total_duration=8.0,
        )
        main.tasks["m3"] = _queued_task(URL_C, queue_status="done", status="error", error="x")
        with self._client() as client:
            resp = client.post("/api/batch/merge", json={"task_ids": ["m1", "m2", "m3"]})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["merged"], 2)
        self.assertEqual(data["failed"], [{"task_id": "m3", "url": URL_C, "reason": "任务未完成"}])
        self.assertTrue(data["filename"].startswith("merged_transcripts_"))
        self.assertTrue(data["filename"].endswith(".md"))
        self.assertEqual(data["download_url"], f"/api/download/{data['filename']}")

        content = (self.tmp_dir / data["filename"]).read_text(encoding="utf-8")
        self.assertIn("# 批量转录合并文档 ·", content)
        self.assertIn("共 3 个任务：成功 2 · 未收录 1", content)
        self.assertIn("## 目录", content)
        self.assertIn("1. 视频甲", content)
        self.assertIn("2. 视频乙", content)
        self.assertIn("## 1. 视频甲", content)
        self.assertIn("> 来源：https://example.com/a · 总耗时 12.5s · 完成于 2026-09-07T10:00:00", content)
        self.assertIn("### 转录文本", content)
        self.assertIn("第一段正文", content)
        self.assertNotIn("# 视频甲\n\n第一段正文", content)  # 一级标题被剥离
        self.assertIn("### 智能摘要", content)
        self.assertIn("摘要甲", content)
        self.assertIn("第二段正文", content)
        self.assertIn("摘要乙", content)
        self.assertIn("> 未收录：m3（https://example.com/c）：任务未完成", content)

    def test_merge_completed_without_script_failed(self):
        main.tasks["m1"] = _queued_task(URL_A, queue_status="done", status="completed", script=None)
        with self._client() as client:
            resp = client.post("/api/batch/merge", json={"task_ids": ["m1"]})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["detail"], "没有可合并的已完成任务")

    def test_merge_no_completed_returns_400(self):
        main.tasks["x1"] = _queued_task(URL_A, queue_status="done", status="error")
        with self._client() as client:
            resp = client.post("/api/batch/merge", json={"task_ids": ["x1"]})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["detail"], "没有可合并的已完成任务")

    def test_merge_missing_task_404(self):
        with self._client() as client:
            resp = client.post("/api/batch/merge", json={"task_ids": ["ghost"]})
        self.assertEqual(resp.status_code, 404)

    def test_merge_empty_list_400(self):
        with self._client() as client:
            resp = client.post("/api/batch/merge", json={"task_ids": []})
        self.assertEqual(resp.status_code, 400)


class ListTasksTests(_BatchQueueTestCase):
    """GET /api/tasks：轻量字段、插入顺序、旧任务默认值。"""

    def setUp(self):
        super().setUp()
        self._worker_patch = patch.object(main, "_queue_worker", self._noop_worker)
        self._worker_patch.start()
        self.addCleanup(self._worker_patch.stop)

    async def _noop_worker(self):
        return None

    def test_list_tasks_lightweight(self):
        with self._client() as client:
            main.tasks["l1"] = _queued_task(
                URL_A, queue_status="done", status="completed",
                video_title="标题", script="正文", summary="摘要",
                cache_hit="raw", completed_at="2026-09-07T10:00:00",
                total_duration=5.0, timeline={"transcribe": 3.0},
            )
            main.tasks["l2"] = _queued_task(URL_B)  # queued（startup 之后创建）
            resp = client.get("/api/tasks")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual([t["id"] for t in data["tasks"]], ["l1", "l2"])  # 插入顺序
        l1 = data["tasks"][0]
        self.assertEqual(l1["queue_status"], "done")
        self.assertEqual(l1["status"], "completed")
        self.assertEqual(l1["video_title"], "标题")
        self.assertEqual(l1["cache_hit"], "raw")
        self.assertEqual(l1["completed_at"], "2026-09-07T10:00:00")
        self.assertEqual(l1["total_duration"], 5.0)
        self.assertEqual(l1["timeline"], {"transcribe": 3.0})
        self.assertTrue(l1["has_script"])
        self.assertTrue(l1["has_summary"])
        # 绝不携带正文/文件路径
        self.assertNotIn("script", l1)
        self.assertNotIn("summary", l1)
        self.assertNotIn("script_path", l1)
        self.assertNotIn("summary_path", l1)
        l2 = data["tasks"][1]
        self.assertEqual(l2["queue_status"], "queued")
        self.assertFalse(l2["has_script"])
        self.assertFalse(l2["has_summary"])
        self.assertEqual(l2["summary_language"], "zh")  # 从 settings 兜底

    def test_list_tasks_old_task_defaults(self):
        main.tasks["old1"] = {"status": "completed", "url": URL_A, "script": "x"}
        main.tasks["old2"] = {"status": "processing", "url": URL_B}
        with self._client() as client:
            resp = client.get("/api/tasks")
        data = resp.json()
        by_id = {t["id"]: t for t in data["tasks"]}
        self.assertEqual(by_id["old1"]["queue_status"], "done")    # completed 无 queue_status → done
        self.assertEqual(by_id["old2"]["queue_status"], "interrupted")  # processing 僵尸 → startup 标记中断
        self.assertEqual(by_id["old1"]["batch_id"], "")
        self.assertEqual(by_id["old1"]["timeline"], {})
        self.assertEqual(by_id["old1"]["summary_language"], "")
        self.assertTrue(by_id["old1"]["has_script"])
        self.assertFalse(by_id["old1"]["has_summary"])


class QueueWorkerTests(unittest.IsolatedAsyncioTestCase):
    """串行 worker：FIFO 顺序、queued→running→done 迁移、跳过取消/已删任务、settings 透传。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        patchers = [
            patch.object(main, "TASKS_FILE", self.tmp_dir / "tasks.json"),
            patch.object(main, "TEMP_DIR", self.tmp_dir),
            patch.object(main, "tasks", {}),
            patch.object(main, "broadcast_task_update", self._noop_broadcast),
            patch.object(main, "_task_queue", asyncio.Queue()),
            patch.object(main, "process_video_task", self._recorder),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)
        main.processing_urls.clear()
        main.active_tasks.clear()
        main.sse_connections.clear()
        self.recorded = []

    async def _noop_broadcast(self, task_id, task):
        pass

    async def _recorder(self, task_id, url, summary_language="zh", api_key="",
                        model_base_url="", model_id="", oral_cleanup=False, spk_num=None):
        # 模拟真实 process_video_task 的收尾：释放 URL 占用 + 置 completed
        main.processing_urls.discard(url)
        main.tasks[task_id]["status"] = "completed"
        self.recorded.append({
            "task_id": task_id,
            "url": url,
            "summary_language": summary_language,
            "oral_cleanup": oral_cleanup,
            "spk_num": spk_num,
            "queue_status_at_call": main.tasks[task_id].get("queue_status"),
        })
        await asyncio.sleep(0.01)

    async def _run_worker_until_done(self, task_ids, worker):
        """轮询等待所有任务 queue_status == done，超时抛错。"""
        for _ in range(300):
            if all(main.tasks[t].get("queue_status") == "done" for t in task_ids):
                return
            await asyncio.sleep(0.02)
        self.fail(f"worker 未在超时内完成: {task_ids}")

    async def _stop_worker(self, worker):
        worker.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await worker

    async def test_worker_processes_in_fifo_order(self):
        urls = [URL_A, URL_B, URL_C]
        for i, url in enumerate(urls):
            tid = f"w{i}"
            main.tasks[tid] = _queued_task(url)
            await main._task_queue.put(tid)

        worker = asyncio.create_task(main._queue_worker())
        try:
            await self._run_worker_until_done(["w0", "w1", "w2"], worker)
            # FIFO 顺序
            self.assertEqual([r["task_id"] for r in self.recorded], ["w0", "w1", "w2"])
            self.assertEqual([r["url"] for r in self.recorded], urls)
            # 调用时已是 running
            for r in self.recorded:
                self.assertEqual(r["queue_status_at_call"], "running")
            # 终态 done
            for i in range(3):
                self.assertEqual(main.tasks[f"w{i}"]["queue_status"], "done")
                self.assertEqual(main.tasks[f"w{i}"]["status"], "completed")
            # URL 占用已释放
            self.assertEqual(main.processing_urls, set())
        finally:
            await self._stop_worker(worker)

    async def test_worker_skips_cancelled_and_missing_tasks(self):
        main.tasks["skip1"] = _queued_task(URL_A, queue_status="cancelled")
        main.tasks["skip2"] = _queued_task(URL_B)
        await main._task_queue.put("skip1")
        await main._task_queue.put("ghost")  # 已删除的任务
        await main._task_queue.put("skip2")

        worker = asyncio.create_task(main._queue_worker())
        try:
            await self._run_worker_until_done(["skip2"], worker)
            self.assertEqual([r["task_id"] for r in self.recorded], ["skip2"])
            self.assertEqual(main.tasks["skip1"]["queue_status"], "cancelled")  # 未被改动
        finally:
            await self._stop_worker(worker)

    async def test_worker_passes_settings_to_pipeline(self):
        tid = "w-settings"
        main.tasks[tid] = _queued_task(
            URL_A,
            settings={
                "summary_language": "en",
                "oral_cleanup": True,
                "spk_num": "2",
                "api_key": "sk-test",
                "model_base_url": "http://localhost:9999/v1",
                "model_id": "gpt-test",
            },
        )
        await main._task_queue.put(tid)

        worker = asyncio.create_task(main._queue_worker())
        try:
            await self._run_worker_until_done([tid], worker)
            self.assertEqual(len(self.recorded), 1)
            r = self.recorded[0]
            self.assertEqual(r["summary_language"], "en")
            self.assertTrue(r["oral_cleanup"])
            self.assertEqual(r["spk_num"], 2)  # _parse_spk_num("2") → 2
        finally:
            await self._stop_worker(worker)

    async def test_worker_marks_interrupted_when_pipeline_aborts(self):
        """process_video_task 异常中断（未置 completed/error）→ interrupted，可重新入队。"""
        tid = "w-abort"
        main.tasks[tid] = _queued_task(URL_A)
        aborted = False

        async def aborting_recorder(task_id, url, *args, **kwargs):
            nonlocal aborted
            main.processing_urls.discard(url)
            if not aborted:
                aborted = True
                raise RuntimeError("模拟管线崩溃")
            main.tasks[task_id]["status"] = "completed"

        with patch.object(main, "process_video_task", aborting_recorder):
            await main._task_queue.put(tid)
            worker = asyncio.create_task(main._queue_worker())
            try:
                for _ in range(300):
                    if main.tasks[tid].get("queue_status") in ("done", "interrupted"):
                        break
                    await asyncio.sleep(0.02)
                self.assertEqual(main.tasks[tid]["queue_status"], "interrupted")
                # worker 未死：还能处理下一个任务
                main.tasks["w2"] = _queued_task(URL_B)
                await main._task_queue.put("w2")
                await self._run_worker_until_done(["w2"], worker)
                self.assertEqual(main.tasks["w2"]["queue_status"], "done")
            finally:
                await self._stop_worker(worker)


if __name__ == "__main__":
    unittest.main(verbosity=2)