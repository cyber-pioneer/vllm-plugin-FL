"""Auditable rules for FlagOS operator coverage classification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TRITON_OPERATOR_KINDS = frozenset({"torch_compile", "triton_compiled"})

# The module component in ``flag_gems.ops.<module>.<callable>`` normally maps
# directly to an ATen API. These aliases cover names where it does not.
FLAGGEMS_ATEN_ALIASES = {
    "layernorm": "native_layer_norm",
    "true_divide": "div",
}

# Explicit fused implementation to baseline API mappings. A rule is usable
# only when its evidence token is present in the captured enable-op list.
FLAGGEMS_FUSED_API_RULES = {
    "flag_gems.fused.fused_moe.": frozenset({"vllm::moe_forward_shared"}),
    "flag_gems.fused.moe_align_block_size.": frozenset(
        {"_moe_C::moe_align_block_size", "moe_align_block_size"}
    ),
    "flag_gems.fused.topk_softmax.": frozenset({"_moe_C::topk_softmax"}),
}

# Kernels with an implementation source owned by FlagGems. These rules cover
# lower-level ATen APIs that may not have their own enable-log entry because a
# higher-level FlagGems API launched them.
FLAGGEMS_KERNEL_PREFIX_RULES = {
    "aten::argmax": ("argmax_kernel_inner",),
    "aten::fill_": ("fill_scalar_kernel",),
    "aten::masked_fill_": ("masked_fill_kernel",),
    "aten::mm": ("mm_kernel_general_host_tma",),
}


@dataclass(frozen=True)
class CoverageDecision:
    covered: bool
    flagos_type: str
    evidence: str


@dataclass(frozen=True)
class FlagOSEvidence:
    aten_apis: frozenset[str]
    fused_apis: frozenset[str]
    lines: tuple[str, ...]


def read_flagos_evidence(path: Path | None) -> FlagOSEvidence:
    """Parse the runtime FlagGems enable list without inferring from names."""
    if path is None:
        return FlagOSEvidence(frozenset(), frozenset(), ())
    lines = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    )
    aten_apis: set[str] = set()
    for line in lines:
        match = re.search(r"flag_gems\.ops\.([^.]+)\.", line)
        if not match:
            continue
        module_name = match.group(1)
        api_name = FLAGGEMS_ATEN_ALIASES.get(module_name, module_name)
        aten_apis.add(f"aten::{api_name}")

    fused_apis: set[str] = set()
    for token, api_names in FLAGGEMS_FUSED_API_RULES.items():
        if any(token in line for line in lines):
            fused_apis.update(api_names)
    return FlagOSEvidence(frozenset(aten_apis), frozenset(fused_apis), lines)


def is_triton_operator(
    operator_names: set[str], operator_kinds: set[str], kernel_names: set[str]
) -> bool:
    return bool(operator_kinds & TRITON_OPERATOR_KINDS) or any(
        name.startswith("triton_") for name in operator_names | kernel_names
    )


def classify_coverage(
    operator_names: set[str],
    operator_kinds: set[str],
    kernel_names: set[str],
    plugin_kernel_names: set[str],
    evidence: FlagOSEvidence,
) -> CoverageDecision:
    """Classify one normalized baseline operator.

    The project-specific policy counts every observed Triton operator in the
    numerator. Other operators require runtime evidence. Communication is not
    covered unless a future FlagCX evidence rule is added here.
    """
    has_void_kernel = any("void" in name for name in kernel_names)
    if has_void_kernel and "communication" not in operator_kinds:
        return CoverageDecision(
            False,
            "none",
            "non-communication operator contains a void kernel name",
        )
    if is_triton_operator(operator_names, operator_kinds, kernel_names):
        return CoverageDecision(
            True,
            "triton",
            "policy: all observed Triton operators count as FlagOS coverage",
        )
    matched_aten = sorted(operator_names & set(evidence.aten_apis))
    non_native_plugin_kernels = sorted(
        kernel_name
        for kernel_name in plugin_kernel_names
        if not kernel_name.startswith(
            (
                "void at::",
                "void gemv",
                "nvjet_tst_",
            )
        )
    )
    if matched_aten and non_native_plugin_kernels:
        return CoverageDecision(
            True,
            "flaggems",
            "flaggems_enable_oplist and non-native plugin kernel: "
            + ",".join(matched_aten),
        )
    matched_kernel_rules = sorted(
        operator_name
        for operator_name in operator_names
        if any(
            kernel_name.startswith(prefix)
            for prefix in FLAGGEMS_KERNEL_PREFIX_RULES.get(operator_name, ())
            for kernel_name in plugin_kernel_names
        )
    )
    if matched_kernel_rules:
        return CoverageDecision(
            True,
            "flaggems",
            "known FlagGems kernel implementation: " + ",".join(matched_kernel_rules),
        )
    matched_fused = sorted(operator_names & set(evidence.fused_apis))
    if matched_fused:
        return CoverageDecision(
            True,
            "flaggems",
            "flaggems fused mapping: " + ",".join(matched_fused),
        )
    if "communication" in operator_kinds:
        return CoverageDecision(
            False,
            "none",
            "communication is in the denominator; no FlagCX evidence",
        )
    return CoverageDecision(False, "none", "no auditable FlagOS evidence")
