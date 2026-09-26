import argparse
import ast
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
TOOL_PACKAGES = ("analyze", "env_check", "record", "statistic")
DEFAULT_MODEL_PATH = "models/ppo_metadrive"


@dataclass(frozen=True)
class ToolArgument:
    names: tuple[str, ...]
    action: str | None = None
    required: bool = False
    help_text: str = ""
    default_text: str | None = None
    positional: bool = False

    @property
    def display_name(self):
        return self.names[-1]


@dataclass(frozen=True)
class Tool:
    package: str
    module: str
    path: Path
    description: str
    arguments: tuple[ToolArgument, ...]


@dataclass(frozen=True)
class PostTestConfig:
    test_name: str
    episodes: int
    max_steps: int
    record_steps: int
    seed: int
    fps: int
    screen_size: int


def _literal_value(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _format_default(node):
    value = _literal_value(node)
    if value is not None:
        return str(value)
    try:
        return ast.unparse(node)
    except AttributeError:
        return "script default"


def _parse_tool_arguments(tree):
    arguments = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue

        names = tuple(
            value
            for argument in node.args
            if isinstance((value := _literal_value(argument)), str)
        )
        if not names:
            continue

        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        action = _literal_value(keywords["action"]) if "action" in keywords else None
        required = bool(_literal_value(keywords["required"])) if "required" in keywords else False
        help_text = str(_literal_value(keywords["help"]) or "") if "help" in keywords else ""
        default_text = _format_default(keywords["default"]) if "default" in keywords else None
        positional = not any(name.startswith("-") for name in names)

        arguments.append(
            ToolArgument(
                names=names,
                action=action,
                required=required,
                help_text=help_text,
                default_text=default_text,
                positional=positional,
            )
        )
    return tuple(arguments)


def discover_tools():
    tools = []
    for package in TOOL_PACKAGES:
        package_dir = PROJECT_ROOT / package
        if not package_dir.is_dir():
            continue

        for path in sorted(package_dir.rglob("*.py")):
            if path.name == "__init__.py":
                continue
            if "__pycache__" in path.parts:
                continue
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
            except (OSError, SyntaxError, UnicodeError) as error:
                print(f"Warning: skipped {path}: {error}", file=sys.stderr)
                continue

            docstring = ast.get_docstring(tree)
            description = (
                docstring.strip().splitlines()[0]
                if docstring
                else path.stem.replace("_", " ").replace("1st", "first").title()
            )
            tools.append(
                Tool(
                    package=package,
                    module=".".join(path.relative_to(PROJECT_ROOT).with_suffix("").parts),
                    path=path,
                    description=description,
                    arguments=_parse_tool_arguments(tree),
                )
            )
    return tools


def grouped_tools(tools):
    return {
        package: [tool for tool in tools if tool.package == package]
        for package in TOOL_PACKAGES
        if any(tool.package == package for tool in tools)
    }


def prompt_text(label, default=None, required=False):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{label}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return str(default)
        if not required:
            return ""
        print("This value is required.")


def prompt_int(label, default, minimum=1):
    while True:
        value = prompt_text(label, default)
        try:
            result = int(value)
        except ValueError:
            print("Enter an integer.")
            continue
        if result < minimum:
            print(f"Enter a value of at least {minimum}.")
            continue
        return result


def prompt_float(label, default, minimum=0.0):
    while True:
        value = prompt_text(label, default)
        try:
            result = float(value)
        except ValueError:
            print("Enter a number, for example 0.0003 or 3e-4.")
            continue
        if result <= minimum:
            print(f"Enter a value greater than {minimum}.")
            continue
        return result


def prompt_yes_no(label, default=True):
    marker = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{label} [{marker}]: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Enter y or n.")


def choose_index(title, labels, allow_back=True, zero_label="Back"):
    print(f"\n{title}")
    for index, label in enumerate(labels, start=1):
        print(f"  {index}. {label}")
    if allow_back:
        print(f"  0. {zero_label}")

    while True:
        answer = input("Select: ").strip()
        if allow_back and answer == "0":
            return None
        try:
            index = int(answer) - 1
        except ValueError:
            index = -1
        if 0 <= index < len(labels):
            return index
        print("Invalid selection.")


def display_command(command):
    return subprocess.list2cmdline([str(part) for part in command])


def execute(command, *, confirm=True):
    print(f"\nCommand:\n{display_command(command)}")
    if confirm and not prompt_yes_no("Run this command?", default=True):
        print("Cancelled.")
        return False
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        return True
    except subprocess.CalledProcessError as error:
        print(f"Command failed with exit code {error.returncode}.", file=sys.stderr)
        return False


def choose_model():
    candidates = []
    for folder_name in ("models", "checkpoints", "models_backup", "model_backup"):
        folder = PROJECT_ROOT / folder_name
        if folder.is_dir():
            candidates.extend(folder.rglob("*.zip"))
    candidates = sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)

    labels = [str(path.relative_to(PROJECT_ROOT)) for path in candidates]
    labels.append("Enter another model path")
    index = choose_index("Choose a model", labels)
    if index is None:
        return None
    if index == len(candidates):
        return prompt_text("Model path", required=True)
    return str(candidates[index].relative_to(PROJECT_ROOT))


def saved_model_path(path_value):
    path = Path(path_value)
    return str(path if path.suffix.lower() == ".zip" else Path(f"{path}.zip"))


def prompt_post_test_config():
    if not prompt_yes_no("Run model evaluation and recording after training?", default=True):
        return None

    test_name = prompt_text("Post-training test name (blank = timestamp)")
    if not test_name:
        test_name = datetime.now().strftime("training_%Y%m%d_%H%M%S")

    return PostTestConfig(
        test_name=test_name,
        episodes=prompt_int("Evaluation episodes", 5),
        max_steps=prompt_int("Maximum evaluation steps", 1000),
        record_steps=prompt_int("Maximum recording steps", 1000),
        seed=prompt_int("Recording seed", 0, minimum=0),
        fps=prompt_int("Recording FPS", 30),
        screen_size=prompt_int("Recording screen size", 672),
    )


def run_post_training_tests(model_path, config):
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    test_runner = PROJECT_ROOT / "run_model_tests.ps1"
    if powershell is None:
        print("Post-training tests failed: PowerShell was not found.", file=sys.stderr)
        return False
    if not test_runner.is_file():
        print(f"Post-training tests failed: missing {test_runner}.", file=sys.stderr)
        return False

    command = [
        powershell,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(test_runner),
        "-TestName",
        config.test_name,
        "-ModelPath",
        saved_model_path(model_path),
        "-Episodes",
        str(config.episodes),
        "-MaxSteps",
        str(config.max_steps),
        "-RecordSteps",
        str(config.record_steps),
        "-Seed",
        str(config.seed),
        "-Fps",
        str(config.fps),
        "-ScreenSize",
        str(config.screen_size),
        "-PythonExecutable",
        sys.executable,
    ]
    print(f"\nTraining completed. Starting model tests: {config.test_name}")
    return execute(command, confirm=False)


def run_new_training():
    print("\nNew training")
    timesteps = prompt_int("Training timesteps", 50_000)
    learning_rate = prompt_float("Learning rate", "3e-4")
    model_path = prompt_text("Output model path", DEFAULT_MODEL_PATH)
    post_test = prompt_post_test_config()

    command = [
        sys.executable,
        str(PROJECT_ROOT / "train.py"),
        "--timesteps",
        str(timesteps),
        "--learning-rate",
        str(learning_rate),
        "--model-path",
        model_path,
        "--skip-post-test",
    ]
    if execute(command) and post_test is not None:
        run_post_training_tests(model_path, post_test)


def run_continued_training():
    print("\nContinue training")
    model_path = choose_model()
    if model_path is None:
        return
    timesteps = prompt_int("Additional timesteps", 25_000)
    learning_rate = prompt_float("Learning rate", "1e-4")
    post_test = prompt_post_test_config()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "continue_train.py"),
        "--model-path",
        model_path,
        "--timesteps",
        str(timesteps),
        "--learning-rate",
        str(learning_rate),
    ]
    if execute(command) and post_test is not None:
        run_post_training_tests(model_path, post_test)


def build_tool_command(tool):
    command = [sys.executable, "-m", tool.module]
    if not tool.arguments:
        return command

    print("\nLeave an optional value blank to use the script default.")
    for argument in tool.arguments:
        details = []
        if argument.default_text is not None:
            details.append(f"default={argument.default_text}")
        if argument.help_text:
            details.append(argument.help_text)
        label = argument.display_name
        if details:
            label += f" ({'; '.join(details)})"

        if argument.action in {"store_true", "store_false"}:
            enabled = prompt_yes_no(label, default=False)
            if enabled:
                command.append(argument.display_name)
            continue

        value = prompt_text(label, required=argument.required or argument.positional)
        if not value:
            continue
        if argument.positional:
            command.extend(shlex.split(value, posix=False))
        else:
            command.extend([argument.display_name, value])
    return command


def run_tool_menu(tools):
    groups = grouped_tools(tools)
    package_names = list(groups)
    package_labels = [f"{name} ({len(groups[name])} tools)" for name in package_names]
    package_index = choose_index("Tool packages", package_labels)
    if package_index is None:
        return

    package_tools = groups[package_names[package_index]]
    tool_labels = [
        f"{tool.module.removeprefix(package_names[package_index] + '.')} - {tool.description}"
        for tool in package_tools
    ]
    tool_index = choose_index(f"{package_names[package_index]} tools", tool_labels)
    if tool_index is None:
        return

    tool = package_tools[tool_index]
    execute(build_tool_command(tool))


def print_tool_inventory(tools):
    for package, package_tools in grouped_tools(tools).items():
        print(f"{package}:")
        for tool in package_tools:
            print(f"  {tool.module:<36} {tool.description}")


def interactive_main(tools):
    while True:
        choice = choose_index(
            "MetaDrive PPO CLI",
            [
                "Start new training",
                "Continue training from an existing model",
                "Open tool packages",
            ],
            allow_back=True,
            zero_label="Exit",
        )
        if choice is None:
            return
        if choice == 0:
            run_new_training()
        elif choice == 1:
            run_continued_training()
        else:
            run_tool_menu(tools)


def parse_args():
    parser = argparse.ArgumentParser(description="MetaDrive PPO command-line interface")
    parser.add_argument("--list-tools", action="store_true")
    parser.add_argument("--run-tool", metavar="MODULE")
    parser.add_argument("tool_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    tools = discover_tools()

    if args.list_tools:
        print_tool_inventory(tools)
        return

    if args.run_tool:
        available = {tool.module: tool for tool in tools}
        if args.run_tool not in available:
            choices = ", ".join(sorted(available))
            raise SystemExit(f"Unknown tool: {args.run_tool}\nAvailable tools: {choices}")
        forwarded_args = args.tool_args[1:] if args.tool_args[:1] == ["--"] else args.tool_args
        subprocess.run(
            [sys.executable, "-m", args.run_tool, *forwarded_args],
            cwd=PROJECT_ROOT,
            check=True,
        )
        return

    if args.tool_args:
        raise SystemExit("Tool arguments require --run-tool.")

    interactive_main(tools)


if __name__ == "__main__":
    main()
