"""Special cases used to map runtime kernels to stable operator identities."""

from __future__ import annotations

import re

OperatorIdentity = tuple[str, ...]

VOCAB_MASK_COMPILE_FUNCTION = (
    "vllm.model_executor.layers.vocab_parallel_embedding."
    "get_masked_input_and_mask"
)
VOCAB_MASK_TRITON_KERNELS = frozenset(
    {
        "triton_poi_fused___and_____or___add_bitwise_not_ge_lt_mul_sub_0",
        "triton_poi_fused___and_____or___add_ge_lt_mul_sub_0",
        "triton_poi_fused___and_____or___bitwise_not_ge_lt_1",
        "triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0",
        "triton_poi_fused_add_bitwise_and_bitwise_or_ge_lt_mul_sub_0",
        "triton_poi_fused_bitwise_and_bitwise_not_bitwise_or_ge_lt_1",
    }
)

MOE_ALIGN_BLOCK_SIZE_KERNEL = re.compile(
    r"^moe_align_block_size_stage\d+(?:_[A-Za-z0-9]+)*$"
)
MOE_ALIGN_BLOCK_SIZE_OPERATOR = "moe_align_block_size"

PURE_COMMUNICATION_OPERATORS = frozenset(
    {
        "_C_custom_ar::all_reduce",
        "symm_mem::one_shot_all_reduce",
        "symm_mem::one_shot_all_reduce_",
        "symm_mem::two_shot_all_reduce_",
        "symm_mem::two_shot_all_reduce_out",
        "symm_mem::multimem_all_reduce_",
        "symm_mem::multimem_one_shot_all_reduce",
    }
)
FUSED_COMMUNICATION_COMPUTE_OPERATORS = frozenset(
    {"vllm::flashinfer_trtllm_fused_allreduce_norm"}
)


def is_fused_communication_compute(operator_name: str, kernel_name: str) -> bool:
    if operator_name in FUSED_COMMUNICATION_COMPUTE_OPERATORS:
        return True
    lowered = kernel_name.lower()
    return any(
        token in lowered
        for token in (
            "allreduce_fusion_kernel",
            "fused_all_gather_matmul",
            "fused_all_gather_scaled_matmul",
            "fused_matmul_reduce_scatter",
            "fused_scaled_matmul_reduce_scatter",
        )
    )


def is_pure_communication(operator_name: str, kernel_name: str) -> bool:
    if operator_name in PURE_COMMUNICATION_OPERATORS:
        return True
    if kernel_name.startswith("ncclDevKernel_"):
        return True
    lowered = kernel_name.lower()
    return any(
        token in lowered
        for token in (
            "cross_device_reduce_",
            "one_shot_all_reduce_kernel",
            "two_shot_all_reduce_kernel",
            "multimem_all_reduce_kernel",
        )
    )


def kernel_callable_identity_name(kernel_name: str) -> str:
    """Return a demangled kernel's namespace-qualified callable name."""
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

    normalized: list[str] = []
    template_depth = 0
    for character in name:
        if character == "<":
            template_depth += 1
        elif character == ">" and template_depth:
            template_depth -= 1
        elif template_depth == 0:
            normalized.append(character)
    callable_name = "".join(normalized) if template_depth == 0 else name
    # Rank suffixes are launch specializations, not different operations.
    return re.sub(r"_rank_\d+$", "", callable_name)


def operator_descriptor(
    source_operator: str, kernel_name: str
) -> tuple[str, str, OperatorIdentity | None]:
    """Apply explicit mapping and grouping rules to one operator/kernel pair."""
    if MOE_ALIGN_BLOCK_SIZE_KERNEL.fullmatch(kernel_name):
        return (
            MOE_ALIGN_BLOCK_SIZE_OPERATOR,
            "custom",
            ("custom_group", MOE_ALIGN_BLOCK_SIZE_OPERATOR),
        )
    if is_fused_communication_compute(source_operator, kernel_name):
        return (
            source_operator,
            "fused_communication_compute",
            ("fused_communication_compute", kernel_name),
        )
    if is_pure_communication(source_operator, kernel_name):
        return (
            source_operator,
            "communication",
            (
                "communication_kernel",
                source_operator,
                kernel_callable_identity_name(kernel_name),
            ),
        )
    if kernel_name in VOCAB_MASK_TRITON_KERNELS:
        return (
            VOCAB_MASK_COMPILE_FUNCTION,
            "torch_compile",
            ("compile_kernel", kernel_name),
        )
    if source_operator == "null":
        if kernel_name.startswith("nvjet_tst_"):
            return "null", "unattributed_nvjet", ("unattributed_nvjet",)
        return (
            "null",
            "unattributed",
            ("kernel_callable", kernel_callable_identity_name(kernel_name)),
        )
    if source_operator.startswith("aten::"):
        operator_kind = "aten"
    elif source_operator.startswith("triton_"):
        operator_kind = "triton_compiled"
    elif "::" in source_operator:
        operator_kind = "custom"
    else:
        operator_kind = "runtime_operator"
    if operator_kind in {"aten", "runtime_operator"}:
        identity = ("operator", operator_kind, source_operator)
    elif operator_kind == "triton_compiled":
        identity = ("compile_kernel", kernel_name)
    else:
        identity = ("kernel_callable", kernel_callable_identity_name(kernel_name))
    return source_operator, operator_kind, identity
