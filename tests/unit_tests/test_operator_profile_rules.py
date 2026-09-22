from tools.operator_profile.extract_operator_shapes import (
    format_operator_time_percent,
    operator_list_rows,
)
from tools.operator_profile.rule.rule_coverage import (
    FlagOSEvidence,
    classify_coverage,
)
from tools.operator_profile.rule.rule_map import (
    kernel_callable_identity_name,
    operator_descriptor,
    staged_kernel_family_name,
)


def test_operator_time_percent_uses_hundredth_percent_resolution():
    assert format_operator_time_percent(123, 10_000) == "1.23"
    assert format_operator_time_percent(1, 100_000) == "<0.01"
    assert format_operator_time_percent(1, 0) == "<0.01"


def test_kernel_callable_identity_ignores_launch_specialization():
    kernel = "void backend::kernel_rank_3<float, 128>(float*, int)"

    assert kernel_callable_identity_name(kernel) == "backend::kernel"


def test_triton_kernel_is_classified_without_an_operator_mapping():
    kernel = "triton_poi_fused_add_mul_0"

    name, kind, identity = operator_descriptor("null", kernel)

    assert name == "null"
    assert kind == "triton_compiled"
    assert identity == ("compile_kernel", kernel)


def test_operator_list_merges_attributed_and_unattributed_kernel_rows():
    kernel = "void extension::kernel<float>(float*)"
    rows = operator_list_rows(
        [
            {
                "operator_name": "extension::operator",
                "kernel_name": kernel,
                "kernel_time_us": 2.0,
            },
            {
                "operator_name": "null",
                "kernel_name": kernel,
                "kernel_time_us": 3.0,
            },
        ]
    )

    assert rows == [
        {
            "operator_id": 1,
            "operator_name": "extension::operator",
            "operator_kind": "custom",
            "kernel_name": kernel,
            "kernel_time_percent(%)": "100.00",
        }
    ]


def test_nvjet_kernels_share_the_aten_mm_operator():
    attributed = operator_descriptor("aten::mm", "nvjet_tst_128x64_TNT")
    unattributed = operator_descriptor("null", "nvjet_tst_64x8_TNN")

    assert attributed == (
        "aten::mm",
        "aten",
        ("operator", "aten", "aten::mm"),
    )
    assert unattributed == attributed


def test_moe_align_stages_are_distinct_custom_operators():
    first = operator_descriptor(
        "vllm::moe_forward_shared", "moe_align_block_size_stage1"
    )
    second = operator_descriptor("null", "moe_align_block_size_stage2_vec")

    assert first == (
        "moe_align_block_size_stage1",
        "custom",
        ("moe_align_block_size_stage", "moe_align_block_size_stage1"),
    )
    assert second == (
        "moe_align_block_size_stage2_vec",
        "custom",
        ("moe_align_block_size_stage", "moe_align_block_size_stage2_vec"),
    )
    assert first[2] != second[2]
    assert staged_kernel_family_name("moe_align_block_size_stage1") is None
    assert staged_kernel_family_name("moe_align_block_size_stage2_vec") is None


def test_numbered_stages_share_a_generic_kernel_family():
    first = "backend_pipeline_stage1"
    second = "backend_pipeline_stage2_vec"

    assert staged_kernel_family_name(first) == "backend_pipeline"
    assert staged_kernel_family_name(second) == "backend_pipeline"
    assert operator_descriptor("extension::pipeline", first)[2] == (
        "staged_kernel_family",
        "backend_pipeline",
    )
    assert operator_descriptor("extension::pipeline", second)[2] == (
        "staged_kernel_family",
        "backend_pipeline",
    )


def test_collective_and_fused_collective_are_distinguished_by_semantics():
    pure = operator_descriptor(
        "extension::all_reduce",
        "void backend::all_reduce_kernel(float*)",
    )
    fused = operator_descriptor(
        "extension::fused_allreduce_norm",
        "void backend::allreduce_norm_kernel(float*)",
    )

    assert pure[1] == "communication"
    assert fused[1] == "fused_communication_compute"


def test_elementwise_broadcast_is_not_classified_as_communication():
    descriptor = operator_descriptor("aten::mul", "mul_broadcast_2d_kernel")

    assert descriptor[1] == "aten"


def test_aten_coverage_uses_flaggems_enable_oplist():
    evidence = FlagOSEvidence(
        aten_apis=frozenset({"aten::add"}),
        fused_apis=frozenset(),
        lines=(),
    )

    decision = classify_coverage(
        operator_names={"aten::add"},
        operator_kinds={"aten"},
        kernel_names={"void at::native::add_kernel(float*)"},
        evidence=evidence,
    )

    assert decision.covered is True
    assert decision.flagos_type == "flaggems"
    assert decision.evidence.startswith("flaggems_enable_oplist")


def test_void_communication_kernel_uses_communication_policy():
    decision = classify_coverage(
        operator_names={"c10d::all_reduce"},
        operator_kinds={"communication"},
        kernel_names={"void nccl::all_reduce_kernel(float*)"},
        evidence=FlagOSEvidence(frozenset(), frozenset(), ()),
    )

    assert decision.covered is False
    assert decision.flagos_type == "none"
    assert decision.evidence == (
        "communication is in the denominator; no FlagCX evidence"
    )
