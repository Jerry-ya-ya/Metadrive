import os
import re
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from cli import discover_tools
from model_metadata import load_model_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FOLDERS = ("models", "checkpoints", "models_backup", "model_backup")
INVALID_TEST_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
ACTIVE_STATUSES = {"queued", "running"}


def project_path(value, *, add_zip=False):
    path = Path(value)
    if add_zip and path.suffix.lower() != ".zip":
        path = Path(f"{path}.zip")
    candidate = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError("Paths must stay inside the project directory.") from error
    return resolved


def relative_path(path):
    return str(Path(path).resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


def validate_test_name(value):
    name = (value or "").strip()
    if not name:
        return datetime.now().strftime("web_%Y%m%d_%H%M%S")
    if name in {".", ".."} or INVALID_TEST_NAME.search(name) or name.endswith((" ", ".")):
        raise ValueError("Invalid test name.")
    return name


def serialize_tool(tool):
    return {
        "package": tool.package,
        "module": tool.module,
        "description": tool.description,
        "arguments": [
            {
                "names": list(argument.names),
                "display_name": argument.display_name,
                "action": argument.action,
                "required": argument.required,
                "help": argument.help_text,
                "default": argument.default_text,
                "positional": argument.positional,
            }
            for argument in tool.arguments
        ],
    }


def tool_inventory():
    groups = {}
    for tool in discover_tools():
        groups.setdefault(tool.package, []).append(serialize_tool(tool))
    return groups


def training_defaults():
    from config import METADRIVE_CONFIG

    return {"map": str(METADRIVE_CONFIG["map"])}


def scan_models():
    models = []
    for folder_name in MODEL_FOLDERS:
        folder = PROJECT_ROOT / folder_name
        if not folder.is_dir():
            continue
        for path in folder.rglob("*.zip"):
            try:
                try:
                    metadata = load_model_metadata(path)
                except (OSError, ValueError):
                    metadata = {}
                models.append(
                    {
                        "path": relative_path(path),
                        "size": path.stat().st_size,
                        "modified": path.stat().st_mtime,
                        "map": metadata.get("map"),
                    }
                )
            except OSError:
                continue
    return sorted(models, key=lambda item: item["modified"], reverse=True)


def command_for_tool(module, arguments):
    available = {tool.module: tool for tool in discover_tools()}
    tool = available.get(module)
    if tool is None:
        raise ValueError(f"Unknown tool: {module}")

    command = [sys.executable, "-m", module]
    for argument in tool.arguments:
        key = argument.display_name
        value = arguments.get(key)
        if value in (None, "", False):
            if argument.required or argument.positional:
                raise ValueError(f"Missing required argument: {key}")
            continue

        if argument.action in {"store_true", "store_false"}:
            if bool(value):
                command.append(key)
            continue

        if argument.positional:
            command.append(str(value))
        else:
            command.extend([key, str(value)])
    return tool, command


def training_steps(payload):
    mode = payload["mode"]
    model_value = payload["model_path"]
    timesteps = int(payload["timesteps"])
    learning_rate = float(payload["learning_rate"])
    map_name = str(payload.get("map") or "").strip()
    if mode not in {"new", "continue"}:
        raise ValueError("Training mode must be new or continue.")
    if timesteps < 1 or learning_rate <= 0:
        raise ValueError("Timesteps and learning rate must be positive.")

    model_path = project_path(model_value, add_zip=mode == "continue")
    if mode == "continue" and not model_path.is_file():
        raise ValueError(f"Model not found: {relative_path(model_path)}")
    if mode == "new":
        model_output = project_path(model_value)
        model_output.parent.mkdir(parents=True, exist_ok=True)
        train_command = [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--timesteps",
            str(timesteps),
            "--learning-rate",
            str(learning_rate),
            "--model-path",
            str(model_output),
            "--skip-post-test",
        ]
        saved_model = project_path(model_value, add_zip=True)
    else:
        train_command = [
            sys.executable,
            str(PROJECT_ROOT / "continue_train.py"),
            "--timesteps",
            str(timesteps),
            "--learning-rate",
            str(learning_rate),
            "--model-path",
            str(model_path),
        ]
        saved_model = model_path

    if map_name:
        train_command.extend(["--map", map_name])

    steps = [("training", train_command)]
    summary = None
    post_test = payload.get("post_test") or {}
    if post_test.get("enabled", True):
        test_name = validate_test_name(post_test.get("test_name"))
        output_dir = project_path(f"model_backup/{test_name}")
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"Test output already exists: {relative_path(output_dir)}")
        output_dir.mkdir(parents=True, exist_ok=True)

        evaluation_path = output_dir / "evaluation.txt"
        recording_path = output_dir / "recording.txt"
        video_path = output_dir / "first_person.mp4"
        episodes = int(post_test.get("episodes", 5))
        max_steps = int(post_test.get("max_steps", 1000))
        record_steps = int(post_test.get("record_steps", 1000))
        seed = int(post_test.get("seed", 0))
        fps = int(post_test.get("fps", 30))
        screen_size = int(post_test.get("screen_size", 672))
        if min(episodes, max_steps, record_steps, fps, screen_size) < 1 or seed < 0:
            raise ValueError("Post-training values are outside the allowed range.")

        steps.extend(
            [
                (
                    "evaluation",
                    [
                        sys.executable,
                        "-m",
                        "analyze.model_evaluate",
                        "--model-path",
                        str(saved_model),
                        "--episodes",
                        str(episodes),
                        "--max-steps",
                        str(max_steps),
                        "--output",
                        str(evaluation_path),
                    ],
                ),
                (
                    "recording",
                    [
                        sys.executable,
                        "-m",
                        "record.1st_person",
                        "--model-path",
                        str(saved_model),
                        "--steps",
                        str(record_steps),
                        "--seed",
                        str(seed),
                        "--fps",
                        str(fps),
                        "--screen-size",
                        str(screen_size),
                        "--output",
                        str(video_path),
                        "--report-output",
                        str(recording_path),
                    ],
                ),
            ]
        )
        summary = {
            "path": output_dir / "run_summary.txt",
            "lines": [
                "MetaDrive Web Test Run",
                "======================",
                f"Test name         : {test_name}",
                f"Model             : {relative_path(saved_model)}",
                f"Map               : {map_name or 'stored model setting'}",
                f"Evaluation report : {relative_path(evaluation_path)}",
                f"Recording report  : {relative_path(recording_path)}",
                f"First-person video: {relative_path(video_path)}",
                "Status            : completed",
            ],
        }
    return steps, summary


@dataclass
class Job:
    id: str
    name: str
    status: str = "queued"
    stage: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    logs: list[str] = field(default_factory=list)
    process: subprocess.Popen | None = field(default=None, repr=False)
    cancel_requested: bool = False

    def public(self):
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "stage": self.stage,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "return_code": self.return_code,
            "logs": list(self.logs),
            "cancel_requested": self.cancel_requested,
        }


class JobManager:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.RLock()

    def list(self):
        with self.lock:
            return [job.public() for job in reversed(list(self.jobs.values()))]

    def get(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            return job.public() if job else None

    def start(self, name, steps, summary=None):
        with self.lock:
            if any(job.status in ACTIVE_STATUSES for job in self.jobs.values()):
                raise RuntimeError("Another job is already running.")
            job = Job(id=uuid.uuid4().hex[:12], name=name)
            self.jobs[job.id] = job

        thread = threading.Thread(
            target=self._run,
            args=(job, steps, summary),
            daemon=True,
            name=f"job-{job.id}",
        )
        thread.start()
        return job.public()

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            if job.status not in ACTIVE_STATUSES:
                return job.public()
            job.cancel_requested = True
            process = job.process
        if process and process.poll() is None:
            process.terminate()
        return job.public()

    def _append(self, job, line):
        with self.lock:
            job.logs.append(line.rstrip())
            if len(job.logs) > 5000:
                del job.logs[:1000]

    def _run(self, job, steps, summary):
        with self.lock:
            job.status = "running"
            job.started_at = datetime.now().isoformat(timespec="seconds")

        try:
            for stage, command in steps:
                with self.lock:
                    if job.cancel_requested:
                        raise InterruptedError
                    job.stage = stage
                self._append(job, f"$ {subprocess.list2cmdline(command)}")
                environment = os.environ.copy()
                environment["PYTHONUNBUFFERED"] = "1"
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=environment,
                )
                with self.lock:
                    job.process = process
                if process.stdout:
                    for line in process.stdout:
                        self._append(job, line)
                return_code = process.wait()
                with self.lock:
                    job.process = None
                    job.return_code = return_code
                    cancelled = job.cancel_requested
                if cancelled:
                    raise InterruptedError
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, command)

            if summary:
                summary["path"].write_text("\n".join(summary["lines"]) + "\n", encoding="utf-8")
            with self.lock:
                job.status = "succeeded"
                job.stage = "completed"
        except InterruptedError:
            with self.lock:
                job.status = "cancelled"
                job.stage = "cancelled"
            self._append(job, "Job cancelled.")
        except Exception as error:
            with self.lock:
                job.status = "failed"
                job.stage = "failed"
            self._append(job, f"ERROR: {error}")
        finally:
            with self.lock:
                job.process = None
                job.finished_at = datetime.now().isoformat(timespec="seconds")


job_manager = JobManager()
