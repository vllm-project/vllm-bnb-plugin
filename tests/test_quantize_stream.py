# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_bnb_plugin.quantization.utils import bnb_quantize_stream

# Large enough that the `absmax - offset` subtraction feeding bitsandbytes'
# nested quantization is still in flight when its kernel runs.
SHAPE = (7168, 4096)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_double_quant_is_correct_on_a_non_default_stream() -> None:
    """bitsandbytes quantizes on the default stream regardless of the current
    one, so vLLM's dedicated stream would otherwise corrupt the statistics."""
    from bitsandbytes.functional import dequantize_4bit, quantize_4bit

    weight = torch.randn(SHAPE, dtype=torch.bfloat16, device="cuda") * 0.02

    with torch.cuda.stream(torch.cuda.Stream()):
        loaded_weight = weight * 1.0
        with bnb_quantize_stream():
            qweight, quant_state = quantize_4bit(
                loaded_weight, compress_statistics=True, quant_type="nf4"
            )
        dequantized = dequantize_4bit(qweight, quant_state)

    error = (dequantized.float() - weight.float()).abs().mean()
    # nf4 round-trip lands around 0.09; corrupted statistics are orders of
    # magnitude worse.
    assert error / weight.float().abs().mean() < 0.2
