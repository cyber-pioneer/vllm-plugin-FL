#!/usr/bin/env python3
"""Start a profiled vLLM server from the shared model registry."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS = TOOL_DIR / "models.json"
DEFAULT_RUN_ROOT = Path("/vllm-workspace/graph_operator_profile_runs")
CAPTURE_SIZES = [1, 2, 4, 8, 16, 32, 64]


def load_model(path: Path, model_key: str) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if model_key not in registry:
        choices = ", ".join(sorted(registry))
        raise ValueError(f"unknown model {model_key!r}; choose one of: {choices}")
    model = registry[model_key]
    required = {"model_path", "served_model_name", "tensor_parallel_size"}
    missing = sorted(required - set(model))
    if missing:
        raise ValueError(f"model {model_key!r} is missing: {', '.join(missing)}")
    server_args = model.get("server_args", [])
    if not isinstance(server_args, list) or not all(
        isinstance(item, str) for item in server_args
    ):
        raise ValueError(f"model {model_key!r} server_args must be a string list")
    return model


def archive_existing(run_dir: Path, run_root: Path) -> None:
    if not run_dir.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = run_root / "archive" / f"{run_dir.name}_{timestamp}_{os.getpid()}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), destination)


def build_command(
    model: dict[str, Any], execution_mode: str, profile_dir: Path
) -> list[str]:
    profiler_config = {
        "profiler": "torch",
        "torch_profiler_dir": str(profile_dir),
        "torch_profiler_record_shapes": True,
        "torch_profiler_with_stack": False,
        "torch_profiler_dump_cuda_time_total": False,
        "torch_profiler_with_memory": False,
        "ignore_frontend": True,
    }
    compilation_config = {
        "cudagraph_capture_sizes": CAPTURE_SIZES,
        "cudagraph_num_of_warmups": 0,
    }
    command = [
        "vllm",
        "serve",
        str(model["model_path"]),
        "--served-model-name",
        str(model["served_model_name"]),
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--tensor-parallel-size",
        str(model["tensor_parallel_size"]),
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "64",
        "--no-enable-prefix-caching",
        "--trust-remote-code",
        *model.get("server_args", []),
    ]
    if execution_mode == "eager":
        command.append("--enforce-eager")
    command.extend(
        [
            "--compilation-config",
            json.dumps(compilation_config, separators=(",", ":")),
            "--profiler-config",
            json.dumps(profiler_config, separators=(",", ":")),
        ]
    )
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_key")
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model = load_model(args.models, args.model_key)
    execution_mode = os.environ.get("PROFILE_EXECUTION_MODE", "graph")
    if execution_mode not in {"graph", "eager"}:
        raise ValueError(
            f"unsupported PROFILE_EXECUTION_MODE: {execution_mode}; "
            "expected graph or eager"
        )

    run_root = Path(os.environ.get("PROFILE_RUN_ROOT", str(DEFAULT_RUN_ROOT)))
    run_suffix = os.environ.get("PROFILE_RUN_SUFFIX", "")
    run_dir = run_root / f"{args.model_key}{run_suffix}"
    profile_dir = run_dir / "profile"
    command = build_command(model, execution_mode, profile_dir)
    environment = os.environ.copy()
    environment.setdefault("VLLM_PLUGINS", "fl")
    environment["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model_key": args.model_key,
                    "execution_mode": execution_mode,
                    "run_dir": str(run_dir),
                    "vllm_plugins": environment["VLLM_PLUGINS"],
                    "command": command,
                },
                indent=2,
            )
        )
        return

    archive_existing(run_dir, run_root)
    profile_dir.mkdir(parents=True)
    log_path = run_dir / "serve.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"model_key={args.model_key}\n")
        log.write(f"execution_mode={execution_mode}\n")
        log.write(f"vllm_plugins={environment['VLLM_PLUGINS']!r}\n")
        log.write(f"command={shlex.join(command)}\n")
        log.flush()
        os.dup2(log.fileno(), sys.stdout.fileno())
        os.dup2(log.fileno(), sys.stderr.fileno())
        os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
