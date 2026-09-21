from tools.operator_profile.extract_operator_shapes import format_operator_time_percent
from tools.operator_profile.rule.rule_map import (
    kernel_callable_identity_name,
    operator_descriptor,
    staged_kernel_family_name,
)


def test_operator_time_percent_uses_hundredth_percent_resolution():
    assert format_operator_time_percent(123, 10_000) == "1.23"
    assert format_operator_time_percent(1, 100_000) == "<0.01%"
    assert format_operator_time_percent(1, 0) == "<0.01%"


def test_kernel_callable_identity_ignores_launch_specialization():
    kernel = "void backend::kernel_rank_3<float, 128>(float*, int)"

    assert kernel_callable_identity_name(kernel) == "backend::kernel"


def test_triton_kernel_is_classified_without_an_operator_mapping():
    kernel = "triton_poi_fused_add_mul_0"

    name, kind, identity = operator_descriptor("null", kernel)

    assert name == "null"
    assert kind == "triton_compiled"
    assert identity == ("compile_kernel", kernel)


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
