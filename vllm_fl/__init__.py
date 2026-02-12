# Copyright (c) 2025 BAAI. All rights reserved.


import os
import logging
from vllm_fl.utils import get_op_config as _get_op_config


logger = logging.getLogger(__name__)


def _patch_transformers_compat():
    """Patch transformers to provide ALLOWED_LAYER_TYPES if missing.

    vLLM 0.13.0 imports ``ALLOWED_LAYER_TYPES`` from
    ``transformers.configuration_utils`` but newer transformers versions
    renamed it to ``ALLOWED_MLP_LAYER_TYPES``.  We add the alias so that
    vLLM's config module can be imported successfully.
    """
    import transformers.configuration_utils as _tu
    if not hasattr(_tu, "ALLOWED_LAYER_TYPES"):
        _tu.ALLOWED_LAYER_TYPES = getattr(_tu, "ALLOWED_MLP_LAYER_TYPES", ())


def _patch_is_deepseek_mla():
    """Patch ``ModelConfig.is_deepseek_mla`` to recognise ``glm_moe_dsa``."""
    from vllm.config.model import ModelConfig

    _orig = ModelConfig.is_deepseek_mla.fget

    @property  # type: ignore[misc]
    def _patched(self):
        if (
            hasattr(self.hf_text_config, "model_type")
            and self.hf_text_config.model_type == "glm_moe_dsa"
            and getattr(self.hf_text_config, "kv_lora_rank", None) is not None
        ):
            return True
        return _orig(self)

    ModelConfig.is_deepseek_mla = _patched


def register():
    """Register the FL platform."""

    _patch_transformers_compat()

    multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if multiproc_method is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    _get_op_config()
    return "vllm_fl.platform.PlatformFL"


def register_model():
    """Register the FL model."""
    from vllm import ModelRegistry

    _patch_is_deepseek_mla()

    try:
        from vllm_fl.models.qwen3_next import Qwen3NextForCausalLM  # noqa: F401

        ModelRegistry.register_model(
            "Qwen3NextForCausalLM", "vllm_fl.models.qwen3_next:Qwen3NextForCausalLM"
        )
    except ImportError:
        logger.info(
            "From vllm_fl.models.qwen3_next cannot import Qwen3NextForCausalLM, skipped"
        )
    except Exception as e:
        logger.error(f"Register model error: {str(e)}")

    ModelRegistry.register_model(
        "MiniCPMO",
        "vllm_fl.models.minicpmo:MiniCPMO")

    # Register Kimi-K2.5 model
    try:
        ModelRegistry.register_model(
            "KimiK25ForConditionalGeneration",
            "vllm_fl.models.kimi_k25:KimiK25ForConditionalGeneration"
        )
    except Exception as e:
        logger.error(f"Register KimiK25 model error: {str(e)}")

    # Register GLM-5 (GlmMoeDsa) model
    try:
        ModelRegistry.register_model(
            "GlmMoeDsaForCausalLM",
            "vllm_fl.models.glm_moe_dsa:GlmMoeDsaForCausalLM"
        )
    except Exception as e:
        logger.error(f"Register GlmMoeDsa model error: {str(e)}")
