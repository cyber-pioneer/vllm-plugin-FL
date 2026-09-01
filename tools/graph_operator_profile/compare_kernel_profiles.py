#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from extract_operator_shapes import iter_events


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two extracted runtime kernel profiles."
    )
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--scan-cpu-operators",
        action="store_true",
        help="Scan raw runtime traces and compare complete CPU operator name sets.",
    )
    return parser.parse_args()


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalized_metadata(value):
    return value if value and value != "null" else "null"


def load_run(run_dir, scan_cpu_operators):
    result_dir = run_dir / "results"
    summary = read_json(result_dir / "summary.json")
    metrics = read_json(run_dir / "profiled_metrics.json")
    kernels = {}
    for row in read_csv(result_dir / "kernel_summary.csv"):
        kernels[row["kernel_name"]] = {
            "call_count": int(row["total_call_count"]),
            "time_us": float(row["total_time_us"]),
        }

    operators = defaultdict(
        lambda: {
            "kernel_names": set(),
            "shape_variants": set(),
            "event_count": 0,
            "time_us": 0.0,
        }
    )
    shapes = defaultdict(
        lambda: {
            "kernel_names": set(),
            "mapping_statuses": set(),
            "event_count": 0,
            "time_us": 0.0,
        }
    )
    for row in read_csv(result_dir / "kernel_details_report.csv"):
        operator_name = normalized_metadata(row["operator_name"])
        input_shapes = normalized_metadata(row["input_shapes"])
        input_dtypes = normalized_metadata(row["input_dtypes"])
        event_count = int(row["kernel_event_count"])
        time_us = float(row["kernel_time_us"])

        operator = operators[operator_name]
        operator["kernel_names"].add(row["kernel_name"])
        operator["shape_variants"].add((input_shapes, input_dtypes))
        operator["event_count"] += event_count
        operator["time_us"] += time_us

        shape = shapes[(operator_name, input_shapes, input_dtypes)]
        shape["kernel_names"].add(row["kernel_name"])
        shape["mapping_statuses"].add(row["mapping_status"])
        shape["event_count"] += event_count
        shape["time_us"] += time_us

    cpu_operator_names = set()
    if scan_cpu_operators:
        for trace_name in summary["runtime_trace_files"]:
            for event in iter_events(Path(trace_name)):
                if event.get("cat") == "cpu_op":
                    cpu_operator_names.add(str(event.get("name", "")))
        if len(cpu_operator_names) != summary["unique_cpu_operator_names"]:
            raise ValueError("CPU operator type count does not match summary.json")

    return {
        "run_dir": str(run_dir),
        "summary": summary,
        "metrics": metrics,
        "kernels": kernels,
        "operators": operators,
        "shapes": shapes,
        "kernel_time_us": sum(item["time_us"] for item in kernels.values()),
        "cpu_operator_names": cpu_operator_names,
    }


def presence(key, left, right):
    if key in left and key in right:
        return "both"
    if key in left:
        return "left_only"
    return "right_only"


def percent(value, denominator):
    return value * 100.0 / denominator if denominator else 0.0


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def kernel_rows(left, right):
    rows = []
    for name in sorted(set(left["kernels"]) | set(right["kernels"])):
        ldata = left["kernels"].get(name, {"call_count": 0, "time_us": 0.0})
        rdata = right["kernels"].get(name, {"call_count": 0, "time_us": 0.0})
        lpct = percent(ldata["time_us"], left["kernel_time_us"])
        rpct = percent(rdata["time_us"], right["kernel_time_us"])
        rows.append(
            {
                "kernel_name": name,
                "presence": presence(name, left["kernels"], right["kernels"]),
                "left_call_count": ldata["call_count"],
                "right_call_count": rdata["call_count"],
                "left_time_us": round(ldata["time_us"], 3),
                "right_time_us": round(rdata["time_us"], 3),
                "left_percent": round(lpct, 6),
                "right_percent": round(rpct, 6),
                "percent_point_delta": round(rpct - lpct, 6),
            }
        )
    return rows


def operator_rows(left, right):
    rows = []
    for name in sorted(set(left["operators"]) | set(right["operators"])):
        ldata = left["operators"].get(name)
        rdata = right["operators"].get(name)
        lpct = percent(ldata["time_us"], left["kernel_time_us"]) if ldata else 0.0
        rpct = percent(rdata["time_us"], right["kernel_time_us"]) if rdata else 0.0
        rows.append(
            {
                "operator_name": name,
                "presence": presence(name, left["operators"], right["operators"]),
                "left_kernel_type_count": len(ldata["kernel_names"]) if ldata else 0,
                "right_kernel_type_count": len(rdata["kernel_names"]) if rdata else 0,
                "left_shape_dtype_variant_count": (
                    len(ldata["shape_variants"]) if ldata else 0
                ),
                "right_shape_dtype_variant_count": (
                    len(rdata["shape_variants"]) if rdata else 0
                ),
                "left_event_count": ldata["event_count"] if ldata else 0,
                "right_event_count": rdata["event_count"] if rdata else 0,
                "left_time_us": round(ldata["time_us"], 3) if ldata else 0.0,
                "right_time_us": round(rdata["time_us"], 3) if rdata else 0.0,
                "left_percent": round(lpct, 6),
                "right_percent": round(rpct, 6),
                "percent_point_delta": round(rpct - lpct, 6),
            }
        )
    return rows


def shape_rows(left, right):
    rows = []
    for key in sorted(set(left["shapes"]) | set(right["shapes"])):
        ldata = left["shapes"].get(key)
        rdata = right["shapes"].get(key)
        lpct = percent(ldata["time_us"], left["kernel_time_us"]) if ldata else 0.0
        rpct = percent(rdata["time_us"], right["kernel_time_us"]) if rdata else 0.0
        rows.append(
            {
                "operator_name": key[0],
                "input_shapes": key[1],
                "input_dtypes": key[2],
                "presence": presence(key, left["shapes"], right["shapes"]),
                "left_mapping_statuses": (
                    json.dumps(sorted(ldata["mapping_statuses"])) if ldata else "[]"
                ),
                "right_mapping_statuses": (
                    json.dumps(sorted(rdata["mapping_statuses"])) if rdata else "[]"
                ),
                "left_kernel_type_count": len(ldata["kernel_names"]) if ldata else 0,
                "right_kernel_type_count": len(rdata["kernel_names"]) if rdata else 0,
                "left_event_count": ldata["event_count"] if ldata else 0,
                "right_event_count": rdata["event_count"] if rdata else 0,
                "left_time_us": round(ldata["time_us"], 3) if ldata else 0.0,
                "right_time_us": round(rdata["time_us"], 3) if rdata else 0.0,
                "left_percent": round(lpct, 6),
                "right_percent": round(rpct, 6),
                "percent_point_delta": round(rpct - lpct, 6),
            }
        )
    return rows


def set_counts(left_set, right_set):
    return {
        "left": len(left_set),
        "right": len(right_set),
        "intersection": len(left_set & right_set),
        "left_only": len(left_set - right_set),
        "right_only": len(right_set - left_set),
    }


def known_operator_set(run):
    return set(run["operators"]) - {"null"}


def known_shape_set(run):
    return {
        key
        for key in run["shapes"]
        if key[0] != "null" and key[1] != "null" and key[2] != "null"
    }


def build_summary(left, right, left_label, right_label):
    left_kernels = set(left["kernels"])
    right_kernels = set(right["kernels"])
    left_operators = known_operator_set(left)
    right_operators = known_operator_set(right)
    left_shapes = known_shape_set(left)
    right_shapes = known_shape_set(right)
    return {
        "labels": {"left": left_label, "right": right_label},
        "run_directories": {
            "left": left["run_dir"],
            "right": right["run_dir"],
        },
        "workload": {
            "left": left["metrics"],
            "right": right["metrics"],
        },
        "cpu_operator_types": (
            set_counts(left["cpu_operator_names"], right["cpu_operator_names"])
            if left["cpu_operator_names"] or right["cpu_operator_names"]
            else None
        ),
        "kernel_types": set_counts(left_kernels, right_kernels),
        "known_operator_types": set_counts(left_operators, right_operators),
        "known_shape_dtype_variants": set_counts(left_shapes, right_shapes),
        "kernel_events": {
            "left": left["summary"]["kernel_event_count"],
            "right": right["summary"]["kernel_event_count"],
        },
        "kernel_time_us": {
            "left": left["kernel_time_us"],
            "right": right["kernel_time_us"],
        },
        "operator_shape_mapping": {
            "left_event_counts": left["summary"][
                "kernel_mapping_event_count_by_status"
            ],
            "right_event_counts": right["summary"][
                "kernel_mapping_event_count_by_status"
            ],
            "left_time_us": left["summary"]["kernel_mapping_time_us_by_status"],
            "right_time_us": right["summary"]["kernel_mapping_time_us_by_status"],
        },
    }


def format_top(rows, key_name, limit=15):
    ordered = sorted(
        rows,
        key=lambda row: max(row["left_percent"], row["right_percent"]),
        reverse=True,
    )[:limit]
    result = [
        f"| {key_name} | Left % | Right % | Delta pp |",
        "|---|---:|---:|---:|",
    ]
    for row in ordered:
        name = str(row[key_name]).replace("|", "\\|")
        result.append(
            f"| {name} | {row['left_percent']:.6f} | "
            f"{row['right_percent']:.6f} | {row['percent_point_delta']:+.6f} |"
        )
    return result


def write_markdown(path, summary, kernels, operators):
    labels = summary["labels"]
    kernel_types = summary["kernel_types"]
    operator_types = summary["known_operator_types"]
    shape_types = summary["known_shape_dtype_variants"]
    workload = summary["workload"]
    lines = [
        "# Runtime profile comparison",
        "",
        f"- Left: `{labels['left']}`",
        f"- Right: `{labels['right']}`",
        "- Percentages use total rank-0 runtime kernel duration as denominator.",
        "- Known operator and shape counts exclude the explicit `null` metadata bucket.",
        "- Missing logical metadata is retained in all CSV files and in mapping coverage.",
        "",
        "## Scope summary",
        "",
        "| Metric | Left | Right | Intersection | Left only | Right only |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if summary["cpu_operator_types"] is not None:
        cpu_types = summary["cpu_operator_types"]
        lines.append(
            f"| CPU operator types | {cpu_types['left']} | {cpu_types['right']} | "
            f"{cpu_types['intersection']} | {cpu_types['left_only']} | "
            f"{cpu_types['right_only']} |"
        )
    lines.extend(
        [
            (
                f"| Kernel types | {kernel_types['left']} | {kernel_types['right']} | "
                f"{kernel_types['intersection']} | {kernel_types['left_only']} | "
                f"{kernel_types['right_only']} |"
            ),
            (
                f"| Known operator types | {operator_types['left']} | "
                f"{operator_types['right']} | {operator_types['intersection']} | "
                f"{operator_types['left_only']} | {operator_types['right_only']} |"
            ),
            (
                f"| Known shape/dtype variants | {shape_types['left']} | "
                f"{shape_types['right']} | {shape_types['intersection']} | "
                f"{shape_types['left_only']} | {shape_types['right_only']} |"
            ),
            "",
            "## Runtime totals",
            "",
            "| Metric | Left | Right | Right / Left |",
            "|---|---:|---:|---:|",
        ]
    )
    totals = [
        (
            "Output tokens",
            workload["left"]["total_output_tokens"],
            workload["right"]["total_output_tokens"],
        ),
        (
            "Batch wall time (s)",
            workload["left"]["batch_wall_time_seconds"],
            workload["right"]["batch_wall_time_seconds"],
        ),
        (
            "Kernel events",
            summary["kernel_events"]["left"],
            summary["kernel_events"]["right"],
        ),
        (
            "Kernel duration (us)",
            summary["kernel_time_us"]["left"],
            summary["kernel_time_us"]["right"],
        ),
    ]
    for name, left_value, right_value in totals:
        ratio = right_value / left_value if left_value else 0.0
        lines.append(f"| {name} | {left_value:.6f} | {right_value:.6f} | {ratio:.6f} |")
    lines.extend(["", "## Largest kernel shares", ""])
    lines.extend(format_top(kernels, "kernel_name"))
    lines.extend(["", "## Largest attributed operator shares", ""])
    known_operators = [row for row in operators if row["operator_name"] != "null"]
    lines.extend(format_top(known_operators, "operator_name"))
    lines.extend(
        [
            "",
            "See `kernel_comparison.csv`, `operator_comparison.csv`, and "
            "`shape_dtype_comparison.csv` for the complete union.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    left = load_run(args.left, args.scan_cpu_operators)
    right = load_run(args.right, args.scan_cpu_operators)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    kernels = kernel_rows(left, right)
    operators = operator_rows(left, right)
    shapes = shape_rows(left, right)
    summary = build_summary(left, right, args.left_label, args.right_label)

    if args.scan_cpu_operators:
        cpu_rows = [
            {
                "operator_name": name,
                "presence": presence(
                    name, left["cpu_operator_names"], right["cpu_operator_names"]
                ),
            }
            for name in sorted(left["cpu_operator_names"] | right["cpu_operator_names"])
        ]
        write_csv(
            args.output_dir / "cpu_operator_type_comparison.csv",
            ["operator_name", "presence"],
            cpu_rows,
        )
    write_csv(
        args.output_dir / "kernel_comparison.csv",
        list(kernels[0]),
        kernels,
    )
    write_csv(
        args.output_dir / "operator_comparison.csv",
        list(operators[0]),
        operators,
    )
    write_csv(
        args.output_dir / "shape_dtype_comparison.csv",
        list(shapes[0]),
        shapes,
    )
    with (args.output_dir / "comparison_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_markdown(
        args.output_dir / "comparison.md",
        summary,
        kernels,
        operators,
    )


if __name__ == "__main__":
    main()
