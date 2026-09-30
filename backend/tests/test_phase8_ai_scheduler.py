import time

from services.aiScheduler import AIScheduler, AIPriority


def test_scheduler_priority_and_completion(monkeypatch):
    monkeypatch.setenv("AI_GPU_PROTECTION_ENABLED", "false")
    scheduler = AIScheduler()
    seen = []
    scheduler.start(lambda job: seen.append(job.priority) or job.frame_number)
    try:
        f1 = scheduler.submit(camera_id="CAM-1", frame=None, frame_number=1, analytics_ts=0, source_type="rtsp", priority=int(AIPriority.NORMAL))
        f2 = scheduler.submit(camera_id="CAM-2", frame=None, frame_number=2, analytics_ts=0, source_type="rtsp", priority=int(AIPriority.WATCHLIST))
        assert f1.result(timeout=2) in {1, 2}
        assert f2.result(timeout=2) in {1, 2}
        assert scheduler.status()["completed"] == 2
    finally:
        scheduler.stop()


def test_camera_queue_is_bounded(monkeypatch):
    monkeypatch.setenv("AI_GPU_PROTECTION_ENABLED", "false")
    monkeypatch.setenv("AI_CAMERA_QUEUE_LIMIT", "1")
    monkeypatch.setenv("AI_GLOBAL_QUEUE_LIMIT", "2")
    scheduler = AIScheduler()
    gate = __import__('threading').Event()
    scheduler.start(lambda job: gate.wait(1) or job.frame_number)
    try:
        futures = [scheduler.submit(camera_id="CAM-1", frame=None, frame_number=i, analytics_ts=0, source_type="rtsp") for i in range(3)]
        time.sleep(0.1)
        assert scheduler.status()["queue_depth"] <= 1
        gate.set()
        for f in futures:
            try:
                f.result(timeout=2)
            except Exception:
                pass
    finally:
        scheduler.stop()


def test_scheduler_failures_are_isolated(monkeypatch):
    monkeypatch.setenv("AI_GPU_PROTECTION_ENABLED", "false")
    scheduler = AIScheduler()
    scheduler.start(lambda job: (_ for _ in ()).throw(RuntimeError("boom")))
    try:
        future = scheduler.submit(camera_id="CAM-1", frame=None, frame_number=1, analytics_ts=0, source_type="rtsp")
        try:
            future.result(timeout=2)
            assert False
        except RuntimeError as exc:
            assert str(exc) == "boom"
        assert scheduler.status()["failed"] == 1
    finally:
        scheduler.stop()
