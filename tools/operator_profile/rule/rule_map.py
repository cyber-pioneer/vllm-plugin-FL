"""Generic runtime-kernel to operator identity rules."""

from __future__ import annotations

import re

OperatorIdentity = tuple[str, ...]

_COLLECTIVE_MARKERS = (
    "all_gather",
    "all_reduce",
    "allgather",
    "allreduce",
    "cross_device_reduce",
    "reduce_scatter",
    "reducescatter",
    "all_to_all",
    "alltoall",
)
_COMMUNICATION_BACKEND_MARKERS = (
    "flagcx",
    "nccl",
    "rccl",
    "symm_mem",
)
_DISTRIBUTED_NAMESPACE_MARKERS = (
    "c10d::",
    "collective",
    "distributed::",
    "processgroup",
)
_COMPUTE_MARKERS = (
    "fused",
    "fusion",
    "gemm",
    "matmul",
    "norm",
)
_STAGED_KERNEL = re.compile(
    r"^(?P<family>.+?)(?:_stage|::stage)_?\d+(?:_[A-Za-z0-9]+)*$",
    re.IGNORECASE,
)
_MOE_ALIGN_STAGE_KERNEL = re.compile(
    r"^moe_align_block_size_stage_?\d+(?:_[A-Za-z0-9]+)*$",
    re.IGNORECASE,
)


def _combined_name(operator_name: str, kernel_name: str) -> str:
    return f"{operator_name} {kernel_name}".lower()


def _contains_any(value: str, markers: tuple[str, ...]) -> bool:
    return any(marker in value for marker in markers)


def is_fused_communication_compute(operator_name: str, kernel_name: str) -> bool:
    """Return whether a collective also performs non-communication compute."""
    combined = _combined_name(operator_name, kernel_name)
    return _contains_any(combined, _COLLECTIVE_MARKERS) and _contains_any(
        combined, _COMPUTE_MARKERS
    )


def is_pure_communication(operator_name: str, kernel_name: str) -> bool:
    """Classify communication from generic collective/backend name markers."""
    if is_fused_communication_compute(operator_name, kernel_name):
        return False
    combined = _combined_name(operator_name, kernel_name)
    is_collective = _contains_any(
        combined, _COLLECTIVE_MARKERS + _COMMUNICATION_BACKEND_MARKERS
    )
    is_distributed_broadcast = "broadcast" in combined and _contains_any(
        combined,
        _DISTRIBUTED_NAMESPACE_MARKERS + _COMMUNICATION_BACKEND_MARKERS,
    )
    return is_collective or is_distributed_broadcast


def kernel_callable_identity_name(kernel_name: str) -> str:
    """Return a callable identity, preserving semantic template types."""
    name = kernel_name.strip()
    if name.startswith("void "):
        name = name[5:].lstrip()
    if name.endswith(")"):
        depth = 0
        for index in range(len(name) - 1, -1, -1):
            character = name[index]
            if character == ")":
                depth += 1
            elif character == "(":
                depth -= 1
                if depth == 0:
                    name = name[:index].rstrip()
                    break

    template_start = name.find("<")
    if template_start >= 0:
        prefix = re.sub(r"_rank_\d+$", "", name[:template_start])
        return prefix + name[template_start:]
    return re.sub(r"_rank_\d+$", "", name)


def staged_kernel_family_name(kernel_name: str) -> str | None:
    """Return the common callable for a numbered multi-stage kernel family."""
    callable_name = kernel_callable_identity_name(kernel_name)
    if _MOE_ALIGN_STAGE_KERNEL.fullmatch(callable_name):
        return None
    match = _STAGED_KERNEL.fullmatch(callable_name)
    return match.group("family") if match else None


def _operator_kind(source_operator: str, kernel_name: str) -> str:
    if source_operator.startswith("aten::"):
        return "aten"
    if source_operator.startswith("triton_") or kernel_name.startswith("triton_"):
        return "triton_compiled"
    if source_operator == "null":
        return "unattributed"
    if "::" in source_operator:
        return "custom"
    return "runtime_operator"


def operator_descriptor(
    source_operator: str, kernel_name: str
) -> tuple[str, str, OperatorIdentity]:
    """Map an operator/kernel pair without model- or kernel-specific tables."""
    callable_name = kernel_callable_identity_name(kernel_name)
    if callable_name.startswith("nvjet_tst_"):
        return (
            "aten::mm",
            "aten",
            ("operator", "aten", "aten::mm"),
        )
    if _MOE_ALIGN_STAGE_KERNEL.fullmatch(callable_name):
        return (
            callable_name,
            "custom",
            ("moe_align_block_size_stage", callable_name),
        )
    if is_fused_communication_compute(source_operator, kernel_name):
        return (
            source_operator,
            "fused_communication_compute",
            ("fused_communication_compute", callable_name),
        )
    if is_pure_communication(source_operator, kernel_name):
        return (
            source_operator,
            "communication",
            ("communication_kernel", callable_name),
        )

    staged_family = staged_kernel_family_name(kernel_name)
    if staged_family is not None:
        return (
            source_operator,
            _operator_kind(source_operator, kernel_name),
            ("staged_kernel_family", staged_family),
        )

    operator_kind = _operator_kind(source_operator, kernel_name)
    if operator_kind in {"aten", "runtime_operator"}:
        identity = ("operator", operator_kind, source_operator)
    elif operator_kind == "triton_compiled":
        identity = ("compile_kernel", kernel_name)
    else:
        identity = ("kernel_callable", callable_name)
    return source_operator, operator_kind, identity
