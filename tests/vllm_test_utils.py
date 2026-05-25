# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Simplified local test utilities (no vllm repo test imports)."""

import pytest
from vllm.platforms import current_platform


def multi_gpu_test(*, num_gpus: int):
    """Mark test to require multiple GPUs, skip if not enough available."""

    def wrapper(f):
        f = pytest.mark.distributed(num_gpus=num_gpus)(f)
        f = pytest.mark.skipif(
            current_platform.device_count() < num_gpus,
            reason=f"Need at least {num_gpus} GPUs to run the test.",
        )(f)
        return f

    return wrapper
