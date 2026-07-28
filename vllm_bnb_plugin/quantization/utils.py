# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from packaging import version
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _get_min_bitsandbytes_version() -> str:
    return "0.49.2" if current_platform.is_rocm() else "0.48.1"


def _check_bitsandbytes_version() -> None:
    min_version = _get_min_bitsandbytes_version()
    try:
        import bitsandbytes

        if version.parse(bitsandbytes.__version__) < version.parse(min_version):
            raise ImportError(
                "bitsandbytes version is wrong. Please "
                f"install bitsandbytes>={min_version}."
            )
    except ImportError as err:
        raise ImportError(
            f"Please install bitsandbytes>={min_version} via "
            f"`pip install bitsandbytes>={min_version}` to use "
            "bitsandbytes quantizer."
        ) from err


def _adjust_bitsandbytes_4bit_shard(
    param: torch.nn.Parameter,
    shard_offsets: dict[str, tuple[int, int]],
    loaded_shard_id: str,
) -> tuple[int, int]:
    total, _ = shard_offsets["total"]
    orig_offset, orig_size = shard_offsets[loaded_shard_id]
    quantized_total = param.data.shape[0]
    quantized_offset = orig_offset * quantized_total // total
    quantized_size = orig_size * quantized_total // total
    return quantized_size, quantized_offset


def is_layer_skipped_bnb(prefix: str, llm_int8_skip_modules: list[str]) -> bool:
    components = prefix.split(".")
    substr_check = any(
        module_name in components for module_name in llm_int8_skip_modules
    )
    set_components = set(".".join(components[: i + 1]) for i in range(len(components)))
    set_llm_int8_skip_modules = set(llm_int8_skip_modules)
    prefix_check = len(set_llm_int8_skip_modules & set_components) != 0

    return substr_check or prefix_check


def calculate_quant_ratio(dtype: torch.dtype) -> int:
    if dtype.is_floating_point:
        return torch.finfo(dtype).bits // torch.iinfo(torch.uint8).bits
    return torch.iinfo(dtype).bits // torch.iinfo(torch.uint8).bits
