# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace
from unittest.mock import patch

from vllm_fl.platform import PlatformFL


def test_nvidia_uses_cuda_ir_op_priority():
    expected = object()
    config = SimpleNamespace()

    with (
        patch.object(PlatformFL, "vendor_name", "nvidia"),
        patch(
            "vllm.platforms.cuda.CudaPlatform.get_default_ir_op_priority",
            return_value=expected,
        ) as get_priority,
    ):
        actual = PlatformFL.get_default_ir_op_priority(config)

    assert actual is expected
    get_priority.assert_called_once_with(config)
