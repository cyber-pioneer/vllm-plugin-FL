# Copyright (c) 2025 BAAI. All rights reserved.

import logging
from typing import Optional, List

from vllm.model_executor.custom_op import CustomOp
from .layernorm import *  # noqa F403 F401
from .activation import *  # noqa F403 F401
from .rotary_embedding import *  # noqa F403 F401
from .fused_moe import *  # noqa F403 F401

logger = logging.getLogger(__name__)

# Mapping from OOT operator name (op_name) to its class
OOT_OP_CLASSES = {
    "silu_and_mul": SiluAndMulFL,  # noqa F405
    "rms_norm": RMSNormFL,  # noqa F405
    "rotary_embedding": RotaryEmbeddingFL,  # noqa F405
    "fused_moe": FusedMoEFL,  # noqa F405
    "unquantized_fused_moe_method": UnquantizedFusedMoEMethodFL,  # noqa F405
}


def register_oot_ops(whitelist: Optional[List[str]] = None) -> None:
    """
    Register OOT (out-of-tree) custom operators.

    Args:
        whitelist: If provided, only register operators in this list.
                   If None, check VLLM_FL_OOT_WHITELIST env var.
                   If neither is set, register all operators.
    """
    from vllm_fl.utils import get_oot_whitelist, is_oot_enabled

    # Check if OOT registration is enabled
    if not is_oot_enabled():
        return

    # Determine which operators to register
    env_whitelist = get_oot_whitelist()
    if env_whitelist is not None:
        ops_to_register = env_whitelist
    elif whitelist is not None:
        ops_to_register = whitelist
    else:
        ops_to_register = list(OOT_OP_CLASSES.keys())

    for op_name in ops_to_register:
        if op_name in OOT_OP_CLASSES:
            logger.info(f"Registering oot op: {op_name}")
            CustomOp.register_oot(
                _decorated_op_cls=OOT_OP_CLASSES[op_name], name=op_name
            )
        else:
            logger.warning(f"OOT op '{op_name}' not found in OOT_OP_CLASSES, skipping.")
