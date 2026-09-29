import sys
import tempfile
import time
import unittest
from pathlib import Path

from webapp.service import JobManager


def wait_for_job(manager, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(0.05)
    raise AssertionError("Job did not finish in time")


class JobRetryTests(unittest.TestCase):
    def test_failed_training_retries_only_incomplete_stage_after_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "training_runs.txt"
            ready = root / "evaluation_ready.txt"
            manager = JobManager(store_dir=root / "jobs")
            steps = [
                ("training", [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).open('a').write('x')"]),
                ("evaluation", [sys.executable, "-c", f"from pathlib import Path; import sys; sys.exit(0 if Path({str(ready)!r}).exists() else 9)"]),
            ]
            first = manager.start("training:continue", steps, kind="training")
            failed = wait_for_job(manager, first["id"])
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["retry_stage"], "evaluation")
            self.assertTrue(failed["can_retry"])
            self.assertEqual(marker.read_text(), "x")

            ready.write_text("ready")
            restored = JobManager(store_dir=root / "jobs")
            second = restored.retry(first["id"])
            succeeded = wait_for_job(restored, second["id"])
            self.assertEqual(succeeded["status"], "succeeded")
            self.assertEqual(succeeded["retry_of"], first["id"])
            self.assertEqual(marker.read_text(), "x")


if __name__ == "__main__":
    unittest.main()
