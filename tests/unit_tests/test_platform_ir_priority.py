# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from vllm_fl.platform import PlatformFL


def test_nvidia_uses_cuda_ir_op_priority():
    expected = object()
    config = SimpleNamespace()
    cuda_module = ModuleType("vllm.platforms.cuda")
    cuda_platform = type("CudaPlatform", (), {})
    cuda_platform.get_default_ir_op_priority = classmethod(
        lambda cls, actual_config: expected
    )
    cuda_module.CudaPlatform = cuda_platform

    with (
        patch.object(PlatformFL, "vendor_name", "nvidia"),
        patch.dict(sys.modules, {"vllm.platforms.cuda": cuda_module}),
    ):
        actual = PlatformFL.get_default_ir_op_priority(config)

    assert actual is expected
