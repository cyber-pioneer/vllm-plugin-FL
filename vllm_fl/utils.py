# Copyright (c) 2025 BAAI. All rights reserved.

import json
import os
from typing import Optional, Tuple

import flag_gems
from flag_gems.runtime.backend.device import DeviceDetector
from flag_gems.runtime import backend

_OP_CONFIG: Optional[dict[str, str]] = None


def use_flaggems(default: bool = True) -> bool:
    if os.environ.get("VLLM_FL_PREFER_ENABLED", "True").lower() not in ("true", "1"):
        return False
    prefer_backend = os.environ.get("VLLM_FL_PREFER", "").strip()
    if prefer_backend and prefer_backend.lower() != "flagos":
        return False
    value = os.environ.get("USE_FLAGGEMS", None)
    if value is None:
        return default
    return value.lower() in ("true", "1")


def use_flaggems_op(op_name: str, default: bool = True) -> bool:
    if not use_flaggems(default=default):
        return False
    whitelist_str = os.environ.get("VLLM_FL_FLAGOS_WHITELIST", "")
    blacklist_str = os.environ.get("VLLM_FL_FLAGOS_BLACKLIST", "")
    if not whitelist_str and not blacklist_str:
        return True
    whitelist = {op.strip() for op in whitelist_str.split(",") if op.strip()}
    blacklist = {op.strip() for op in blacklist_str.split(",") if op.strip()}
    if op_name in whitelist and op_name in blacklist:
        raise ValueError(
            "VLLM_FL_FLAGOS_WHITELIST and VLLM_FL_FLAGOS_BLACKLIST both contain "
            f"{op_name!r}. Please remove the conflict."
        )
    if op_name in blacklist:
        return False
    if not whitelist:
        return True
    return op_name in whitelist


def _load_op_config_from_env() -> None:
    global _OP_CONFIG
    config_path = os.environ.get("VLLM_FL_OP_CONFIG", None)
    if config_path is None or not config_path.strip():
        _OP_CONFIG = None
        return
    if not os.path.isfile(config_path):
        raise ValueError(f"VLLM_FL_OP_CONFIG file not found: {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            parsed = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid VLLM_FL_OP_CONFIG JSON file.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("VLLM_FL_OP_CONFIG must be a JSON object.")
    normalized: dict[str, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("VLLM_FL_OP_CONFIG must map strings to strings.")
        normalized[key] = value
    _OP_CONFIG = normalized


def get_op_config() -> Optional[dict[str, str]]:
    return _OP_CONFIG


_load_op_config_from_env()


class DeviceInfo:
    def __init__(self):
        self.device = DeviceDetector()
        self.supported_device = ["nvidia", "ascend"]
        backend.set_torch_backend_device_fn(self.device.vendor_name)

    @property
    def dispatch_key(self):
        return self.device.dispatch_key

    @property
    def vendor_name(self):
        return self.device.vendor_name

    @property
    def device_type(self):
        return self.device.name

    @property
    def torch_device_fn(self):
        # torch_device_fn is like 'torch.cuda' object
        return backend.gen_torch_device_object()

    @property
    def torch_backend_device(self):
        # torch_backend_device is like 'torch.backend.cuda' object
        return backend.get_torch_backend_device_fn()

    def get_supported_device(self):
        if self.vendor_name in self.supported_device:
            raise NotImplementedError(f"{self.vendor_name} is not support now!")
        return True


def get_flag_gems_whitelist_blacklist() -> Tuple[
    Optional[list[str]], Optional[list[str]]
]:
    """
    Get FlagGems operator whitelist and blacklist from environment variables.

    Reads VLLM_FL_FLAGOS_WHITELIST and VLLM_FL_FLAGOS_BLACKLIST environment variables,
    parses comma-separated operator names, and returns them as lists.

    Note: VLLM_FL_FLAGOS_WHITELIST and VLLM_FL_FLAGOS_BLACKLIST cannot be set simultaneously.
    If both are set, a ValueError will be raised.

    Returns:
        Tuple[Optional[list[str]], Optional[list[str]]]:
            A tuple of (whitelist, blacklist). Each is None if not set,
            or a list of operator names (stripped of whitespace) if set.

    Raises:
        ValueError: If both VLLM_FL_FLAGOS_WHITELIST and VLLM_FL_FLAGOS_BLACKLIST
                    are set simultaneously.

    Example:
        >>> # Set whitelist only:
        >>> # export VLLM_FL_FLAGOS_WHITELIST="silu_and_mul,rms_norm"
        >>> whitelist, blacklist = get_flag_gems_whitelist_blacklist()
        >>> # whitelist: ["silu_and_mul", "rms_norm"]
        >>> # blacklist: None

        >>> # Set blacklist only:
        >>> # export VLLM_FL_FLAGOS_BLACKLIST="index,index_put_"
        >>> whitelist, blacklist = get_flag_gems_whitelist_blacklist()
        >>> # whitelist: None
        >>> # blacklist: ["index", "index_put_"]
    """
    whitelist_str = os.environ.get("VLLM_FL_FLAGOS_WHITELIST", "")
    blacklist_str = os.environ.get("VLLM_FL_FLAGOS_BLACKLIST", "")

    # Check if both are set
    if whitelist_str and blacklist_str:
        raise ValueError(
            "Cannot set both VLLM_FL_FLAGOS_WHITELIST and VLLM_FL_FLAGOS_BLACKLIST "
            "simultaneously. Please set only one of them."
        )

    whitelist = None
    blacklist = None

    if whitelist_str:
        whitelist = [op.strip() for op in whitelist_str.split(",") if op.strip()]

    if blacklist_str:
        blacklist = [op.strip() for op in blacklist_str.split(",") if op.strip()]

    return whitelist, blacklist


def get_flaggems_all_ops() -> list[str]:
    """
    Get all FlagGems operator names from flag_gems._FULL_CONFIG.
    """
    try:
        pass
    except Exception:
        return []
    ops = flag_gems.all_registered_ops()

    return ops


# OOT operator names as registered in custom_ops.py (op_name lowercase)
OOT_OP_NAMES = [
    "silu_and_mul",
    "rms_norm",
    "rotary_embedding",
    "fused_moe",
    "unquantized_fused_moe_method",
]


def get_oot_whitelist() -> Optional[list[str]]:
    """
    Get OOT operator whitelist from VLLM_FL_OOT_WHITELIST environment variable.

    If set, only the specified OOT operators will be registered.
    Comma-separated list of OOT operator names (e.g., "silu_and_mul,rms_norm").

    Returns:
        List of OOT operator names to register, or None if not set (register all).
    """
    whitelist_str = os.environ.get("VLLM_FL_OOT_WHITELIST", "")
    if not whitelist_str:
        return None
    return [op.strip() for op in whitelist_str.split(",") if op.strip()]


def is_oot_enabled() -> bool:
    """
    Check if OOT registration is enabled.

    Controlled by VLLM_FL_OOT_ENABLED environment variable.
    Default is True (enabled).

    Returns:
        True if OOT registration is enabled, False otherwise.
    """
    if os.environ.get("VLLM_FL_PREFER_ENABLED", "True").lower() not in ("true", "1"):
        return False
    enabled_str = os.environ.get("VLLM_FL_OOT_ENABLED", "1")
    return enabled_str.lower() in ("1", "true")


if __name__ == "__main__":
    device = DeviceInfo()
