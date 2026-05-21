# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.model_loader import register_model_loader

_REGISTERED = False


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    from .bitsandbytes import BitsAndBytesConfig
    from .bitsandbytes_loader import BitsAndBytesModelLoader

    register_quantization_config("bitsandbytes")(BitsAndBytesConfig)
    register_model_loader("bitsandbytes")(BitsAndBytesModelLoader)
    _REGISTERED = True


__all__ = ["register"]
