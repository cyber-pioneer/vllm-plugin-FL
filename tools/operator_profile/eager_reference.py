"""Attach explicitly supplied eager references without changing runtime evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

OPERATOR_REFERENCE_FIELDS = [
    "eager_operator_name",
    "eager_api_candidates",
    "eager_shape_dtype_count",
    "mapping_source",
]
DETAIL_REFERENCE_FIELDS = [
    "eager_operator_name",
    "eager_input_shapes",
    "eager_input_dtypes",
    "eager_candidates",
    "mapping_source",
]
INDUCTOR_PREFIXES = ("triton_poi_", "triton_red_", "triton_per_", "triton_for_")


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def load_eager_reference(
    results: Path, rank: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Read only direct eager observations, never chained reference columns."""
    summary_path = results / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scope = summary.get("scope", {})
    if scope.get("rank") != rank or scope.get("phase") != "runtime_only":
        raise ValueError("eager reference must use the same rank and runtime scope")
    if scope.get("graph_capture_included", False):
        raise ValueError("eager reference must not include graph capture data")
    failed = [
        key
        for key, value in summary.get("conservation", {}).items()
        if isinstance(value, bool) and not value
    ]
    if failed:
        raise ValueError(f"eager reference conservation failed: {failed}")

    path = results / "kernel_shape_dtype.csv"
    content = path.read_bytes()
    candidates: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            kernel = row["kernel_name"]
            operator = row["operator_name"]
            if kernel.startswith(INDUCTOR_PREFIXES) or operator in {"", "null"}:
                continue
            if not row["mapping_status"].startswith("operator_"):
                continue
            candidate = {
                "operator_name": operator,
                "input_shapes": json.loads(row["input_shapes"]),
                "input_dtypes": json.loads(row["input_dtypes"]),
            }
            candidates[kernel][encode(candidate)] = candidate
    index = {
        kernel: [values[key] for key in sorted(values)]
        for kernel, values in candidates.items()
    }
    return index, {
        "source": "eager-refer",
        "results_directory": str(results.resolve()),
        "kernel_shape_dtype_sha256": hashlib.sha256(content).hexdigest(),
        "environment_validation": "caller_supplied_same_environment",
        "rank": rank,
    }


def select_candidates(
    index: dict[str, list[dict[str, Any]]], kernel: str, operator: str
) -> list[dict[str, Any]]:
    if kernel.startswith(INDUCTOR_PREFIXES):
        return []
    candidates = index.get(kernel, [])
    # A known API must not borrow inputs from another API using the same kernel.
    if "::" in operator:
        return [row for row in candidates if row["operator_name"] == operator]
    return candidates


def unique_value(candidates: list[dict[str, Any]], field: str) -> Any:
    values = {
        encode(row[field]): row[field] for row in candidates if row[field] is not None
    }
    return next(iter(values.values())) if len(values) == 1 else None


def attach_eager_reference(
    operator_rows: list[dict[str, Any]],
    detail_rows: list[dict[str, Any]],
    index: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Keep graph observations and counters intact; references are extra columns."""
    needs_reference = set()
    referenced_kernels = set()
    referenced_events = 0
    enriched_details = []
    for row in detail_rows:
        missing = row["operator_name"] == "null" or any(
            row[field] == "null" for field in ("input_shapes", "input_dtypes")
        )
        if missing:
            needs_reference.add(row["kernel_name"])
        candidates = (
            select_candidates(index, row["kernel_name"], row["operator_name"])
            if missing
            else []
        )
        if candidates:
            referenced_kernels.add(row["kernel_name"])
            referenced_events += int(row["kernel_event_count"])
        enriched_details.append(
            {
                **row,
                "eager_operator_name": (
                    unique_value(candidates, "operator_name") or "null"
                ),
                "eager_input_shapes": encode(
                    unique_value(candidates, "input_shapes")
                    if row["input_shapes"] == "null"
                    else None
                ),
                "eager_input_dtypes": encode(
                    unique_value(candidates, "input_dtypes")
                    if row["input_dtypes"] == "null"
                    else None
                ),
                "eager_candidates": encode(candidates or None),
                "mapping_source": "eager-refer" if candidates else "null",
            }
        )

    enriched_operators = []
    for row in operator_rows:
        candidates = (
            select_candidates(index, row["kernel_name"], row["operator_name"])
            if row["kernel_name"] in needs_reference
            else []
        )
        apis = sorted({candidate["operator_name"] for candidate in candidates})
        shape_dtype_pairs = {
            encode([candidate["input_shapes"], candidate["input_dtypes"]])
            for candidate in candidates
            if candidate["input_shapes"] is not None
            and candidate["input_dtypes"] is not None
        }
        enriched_operators.append(
            {
                **row,
                "eager_operator_name": apis[0] if len(apis) == 1 else "null",
                "eager_api_candidates": encode(apis or None),
                "eager_shape_dtype_count": len(shape_dtype_pairs),
                "mapping_source": "eager-refer" if candidates else "null",
            }
        )
    return (
        enriched_operators,
        enriched_details,
        {
            "referenced_kernel_count": len(referenced_kernels),
            "events_with_reference_candidates": referenced_events,
            "references_are_runtime_observations": False,
        },
    )
