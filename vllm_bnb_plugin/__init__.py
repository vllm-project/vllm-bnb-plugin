# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.model_loader import register_model_loader

_REGISTERED = False


def _patch_engine_args() -> None:
    """Patch EngineArgs to set load_format='bitsandbytes' when BNB
    quantization is used, matching upstream vLLM behavior.

    Handles two cases:
    1. Inflight quantization: user explicitly sets quantization='bitsandbytes'
    2. Pre-quantized models: quantization resolved from model config.json
    """
    from vllm.engine.arg_utils import EngineArgs

    _orig_create_engine_config = EngineArgs.create_engine_config

    def _patched_create_engine_config(self, *args, **kwargs):
        result = _orig_create_engine_config(self, *args, **kwargs)
        if result.model_config.quantization == "bitsandbytes":
            self.quantization = "bitsandbytes"
            self.load_format = "bitsandbytes"
            # Rebuild load_config with the correct load_format
            result.load_config.load_format = "bitsandbytes"
        return result

    EngineArgs.create_engine_config = _patched_create_engine_config  # type: ignore[assignment]


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    from .bitsandbytes import BitsAndBytesConfig
    from .bitsandbytes_loader import BitsAndBytesModelLoader

    register_quantization_config("bitsandbytes")(BitsAndBytesConfig)
    register_model_loader("bitsandbytes")(BitsAndBytesModelLoader)
    _patch_engine_args()
    _REGISTERED = True


__all__ = ["register"]
