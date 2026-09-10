#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

SCENARIOS = ("plugin_graph", "plugin_eager", "native_graph", "native_eager")
PAIRWISE = (
    ("plugin_vs_native_graph", "plugin graph vs native graph"),
    ("plugin_vs_native_eager", "plugin eager vs native eager"),
    ("plugin_graph_vs_eager", "plugin graph vs plugin eager"),
    ("native_graph_vs_eager", "native graph vs native eager"),
)
KIND_ORDER = (
    "aten",
    "custom",
    "fused_communication_compute",
    "runtime_operator",
    "torch_compile",
    "triton_compiled",
    "unattributed",
    "unattributed_nvjet",
    "communication",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a four-scenario runtime profile report."
    )
    parser.add_argument("--model-title", required=True)
    parser.add_argument("--tp-size", required=True, type=int)
    parser.add_argument("--plugin-graph", required=True, type=Path)
    parser.add_argument("--plugin-eager", required=True, type=Path)
    parser.add_argument("--native-graph", required=True, type=Path)
    parser.add_argument("--native-eager", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def display_kind(kind):
    return kind.replace("_", " ")


def load_run(path):
    results = path / "results"
    summary = read_json(results / "summary.json")
    metrics = read_json(path / "profiled_metrics.json")
    operator_rows = read_csv(results / "operator_list.csv")
    time_rows = read_csv(results / "kernel_time.csv")
    shape_rows = read_csv(results / "kernel_shape_dtype.csv")
    trace_paths = [Path(item) for item in summary["runtime_trace_files"]]
    trace_bytes = sum(item.stat().st_size for item in trace_paths)

    known_shapes = {
        (row["operator_name"], row["input_shapes"], row["input_dtypes"])
        for row in shape_rows
        if row["operator_name"] != "null"
        and row["input_shapes"] != "null"
        and row["input_dtypes"] != "null"
    }
    operator_ids = {
        row["operator_id"]
        for row in operator_rows
        if row["operator_id"] and row["operator_id"] != "null"
    }
    kinds = Counter(row["operator_kind"] for row in operator_rows)
    kernels = defaultdict(lambda: {"calls": 0, "time_us": 0.0})
    for row in time_rows:
        item = kernels[row["kernel_name"]]
        item["calls"] += int(row["kernel_call_count"])
        item["time_us"] += float(row["kernel_time_us"])

    conservation = summary["conservation"]
    failed_checks = sorted(key for key, value in conservation.items() if not value)
    event_status = summary["kernel_mapping_event_count_by_status"]
    time_status = summary["kernel_mapping_time_us_by_status"]
    matched_events = event_status.get("operator_shape_matched", 0)
    matched_time = time_status.get("operator_shape_matched", 0.0)
    total_events = summary["kernel_event_count"]
    total_time = summary["kernel_time_total_us"]

    return {
        "path": path,
        "summary": summary,
        "metrics": metrics,
        "operator_rows": operator_rows,
        "known_shapes": known_shapes,
        "operator_ids": operator_ids,
        "kinds": kinds,
        "kernels": dict(kernels),
        "trace_mb": trace_bytes / 1_000_000,
        "event_mapping_pct": matched_events * 100.0 / total_events,
        "time_mapping_pct": matched_time * 100.0 / total_time,
        "failed_checks": failed_checks,
    }


def top_kernels(run, limit=10):
    total = run["summary"]["kernel_time_total_us"]
    rows = []
    for name, item in run["kernels"].items():
        rows.append(
            (item["time_us"] * 100.0 / total, item["time_us"], item["calls"], name)
        )
    return sorted(rows, reverse=True)[:limit]


def format_name(value, limit=100):
    value = value.replace("|", "\\|")
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def pct_change(left, right):
    return (right - left) * 100.0 / left if left else 0.0


def main():
    args = parse_args()
    run_paths = {
        "plugin_graph": args.plugin_graph,
        "plugin_eager": args.plugin_eager,
        "native_graph": args.native_graph,
        "native_eager": args.native_eager,
    }
    runs = {name: load_run(path) for name, path in run_paths.items()}
    for name, run in runs.items():
        if run["failed_checks"]:
            raise ValueError(f"{name} failed checks: {run['failed_checks']}")
        metrics = run["metrics"]
        expected = (64, 4096, 256, 64)
        actual = (
            metrics["concurrency"],
            metrics["input_tokens_per_request"],
            metrics["output_tokens_per_request"],
            metrics["request_count"],
        )
        if actual != expected:
            raise ValueError(f"{name} workload mismatch: {actual} != {expected}")

    comparisons = {
        directory: read_json(args.output_dir / directory / "comparison_summary.json")
        for directory, _ in PAIRWISE
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {args.model_title} 4096/256 four-scenario comparison",
        "",
        "## Scope",
        "",
        f"All four runs used tensor parallel size {args.tp_size}, 64 concurrent requests, 4096 input ",
        "tokens per request, and 256 output tokens per request. Each run completed one ",
        "64-request warmup batch before `/start_profile`; only the following 64-request ",
        "batch is present in the runtime trace. Extraction uses rank 0 only. CUDA Graph ",
        "construction and capture are not profiled.",
        "",
        "The native runs used an explicit empty `VLLM_PLUGINS` value. Their API-server ",
        "process environments and logs were checked during execution. Plugin runs loaded ",
        "and activated `fl`; their per-run FlagGems oplists were timestamp-validated and ",
        "moved into the corresponding `results` directories.",
        "",
        "## Scenario summary",
        "",
        "| Scenario | Profiled batch (s) | Kernel events | Kernel types | Kernel duration (us) | CPU op types | Known shape/dtype variants | Operator-kernel relations | Numbered IDs | Event mapping | Time mapping | Rank-0 trace (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in SCENARIOS:
        run = runs[name]
        summary = run["summary"]
        lines.append(
            f"| {name} | {run['metrics']['batch_wall_time_seconds']:.3f} | "
            f"{summary['kernel_event_count']:,} | {summary['unique_kernel_names']:,} | "
            f"{summary['kernel_time_total_us']:,.3f} | {summary['unique_cpu_operator_names']:,} | "
            f"{len(run['known_shapes']):,} | {len(run['operator_rows']):,} | "
            f"{len(run['operator_ids']):,} | {run['event_mapping_pct']:.3f}% | "
            f"{run['time_mapping_pct']:.3f}% | {run['trace_mb']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Profiled batch time includes profiler overhead and is not an unprofiled throughput ",
            "benchmark. Kernel duration is the sum of rank-0 runtime kernel durations, not ",
            "end-to-end latency. Every conservation and classification check in all four ",
            "`summary.json` files is `true`.",
            "",
            "## Relative timing",
            "",
            "| Comparison | Profiled batch change | Kernel-duration change |",
            "|---|---:|---:|",
        ]
    )
    timing_pairs = (
        ("plugin graph vs native graph", "plugin_graph", "native_graph"),
        ("plugin eager vs native eager", "plugin_eager", "native_eager"),
        ("plugin graph vs plugin eager", "plugin_graph", "plugin_eager"),
        ("native graph vs native eager", "native_graph", "native_eager"),
    )
    for label, left, right in timing_pairs:
        left_run = runs[left]
        right_run = runs[right]
        batch_delta = pct_change(
            left_run["metrics"]["batch_wall_time_seconds"],
            right_run["metrics"]["batch_wall_time_seconds"],
        )
        kernel_delta = pct_change(
            left_run["summary"]["kernel_time_total_us"],
            right_run["summary"]["kernel_time_total_us"],
        )
        lines.append(f"| {label} | {batch_delta:+.3f}% | {kernel_delta:+.3f}% |")

    kinds = [
        kind for kind in KIND_ORDER if any(run["kinds"][kind] for run in runs.values())
    ]
    lines.extend(
        [
            "",
            "Changes are right relative to left. The two columns use different denominators and ",
            "must not be interpreted as equivalent latency measurements.",
            "",
            "## Operator classification",
            "",
            "| Scenario | " + " | ".join(display_kind(kind) for kind in kinds) + " |",
            "|---|" + "---:|" * len(kinds),
        ]
    )
    for name in SCENARIOS:
        lines.append(
            f"| {name} | "
            + " | ".join(str(runs[name]["kinds"][kind]) for kind in kinds)
            + " |"
        )

    lines.extend(
        [
            "",
            "Counts are unique operator-kernel relations from `operator_list.csv`, not runtime ",
            "call counts. ATen relations are listed first. Pure communication relations are ",
            "numbered and listed last. Missing attribution remains present as `operator_name=null`.",
            "",
            "## Type and shape differences",
            "",
            "Each cell is `left / right / intersection / left-only / right-only`.",
            "",
            "| Comparison | Kernel types | Raw attributed operator labels | Shape/dtype variants | Complete CPU op types |",
            "|---|---|---|---|---|",
        ]
    )
    for directory, label in PAIRWISE:
        item = comparisons[directory]
        fields = (
            item["kernel_types"],
            item["known_operator_types"],
            item["known_shape_dtype_variants"],
            item["cpu_operator_types"],
        )
        cells = []
        for value in fields:
            cells.append(
                f"{value['left']} / {value['right']} / {value['intersection']} / "
                f"{value['left_only']} / {value['right_only']}"
            )
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "Graph replay preserves physical kernel accounting but does not replay most original ",
            "CPU operator events. This is why graph runs have lower runtime logical shape ",
            "attribution than eager runs. Complete unions are retained in the pairwise CSV files.",
            "",
            "## Top kernel time shares",
            "",
        ]
    )
    for name in SCENARIOS:
        lines.extend(
            [
                f"### {name}",
                "",
                "| Kernel | Calls | Kernel time (us) | Share |",
                "|---|---:|---:|---:|",
            ]
        )
        for share, time_us, calls, kernel_name in top_kernels(runs[name]):
            lines.append(
                f"| {format_name(kernel_name)} | {calls:,} | {time_us:,.3f} | {share:.3f}% |"
            )
        lines.append("")

    lines.extend(["## Result locations", ""])
    for name in SCENARIOS:
        lines.append(f"- `{runs[name]['path']}/`")
    lines.extend(
        [
            "",
            "Pairwise comparisons are under this report directory in `plugin_vs_native_graph`, ",
            "`plugin_vs_native_eager`, `plugin_graph_vs_eager`, and ",
            "`native_graph_vs_eager`.",
            "",
        ]
    )
    output = args.output_dir / "FOUR_SCENARIO_ANALYSIS.md"
    output.write_text("\n".join(line.rstrip() for line in lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
