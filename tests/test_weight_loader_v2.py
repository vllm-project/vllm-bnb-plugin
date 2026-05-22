# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import pytest

from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor import parameter as parameter_module

from vllm_bnb_plugin.bitsandbytes import (
    BitsAndBytes4bitParameter,
    BitsAndBytes8bitParameter,
    BitsAndBytesConfig,
)


@pytest.fixture(autouse=True)
def _mock_single_rank_tp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_world_size", lambda: 1)


def _assert_uses_weight_loader_v2(weight, layer) -> None:
    assert weight.weight_loader.__self__ is layer
    assert weight.weight_loader.__func__ is layer.weight_loader_v2.__func__


def test_bitsandbytes_4bit_row_parallel_uses_weight_loader_v2() -> None:
    layer = RowParallelLinear(
        input_size=4,
        output_size=8,
        bias=False,
        params_dtype=torch.float16,
        quant_config=BitsAndBytesConfig(),
        disable_tp=True,
    )

    assert isinstance(layer.weight, BitsAndBytes4bitParameter)
    _assert_uses_weight_loader_v2(layer.weight, layer)

    loaded_weight = torch.arange(16, dtype=torch.uint8).reshape(16, 1)
    layer.weight.weight_loader(layer.weight, loaded_weight)

    assert torch.equal(layer.weight.data, loaded_weight)


def test_bitsandbytes_4bit_merged_column_loading_v2() -> None:
    layer = MergedColumnParallelLinear(
        input_size=4,
        output_sizes=[4, 4],
        bias=False,
        params_dtype=torch.float16,
        quant_config=BitsAndBytesConfig(),
        disable_tp=True,
    )

    assert isinstance(layer.weight, BitsAndBytes4bitParameter)
    _assert_uses_weight_loader_v2(layer.weight, layer)

    shard_0 = torch.arange(8, dtype=torch.uint8).reshape(8, 1)
    shard_1 = torch.arange(8, 16, dtype=torch.uint8).reshape(8, 1)

    layer.weight.weight_loader(layer.weight, shard_0, 0)
    layer.weight.weight_loader(layer.weight, shard_1, 1)

    expected = torch.cat([shard_0, shard_1], dim=0)
    assert torch.equal(layer.weight.data, expected)


def test_bitsandbytes_4bit_qkv_loading_v2() -> None:
    layer = QKVParallelLinear(
        hidden_size=8,
        head_size=2,
        total_num_heads=2,
        total_num_kv_heads=1,
        bias=False,
        params_dtype=torch.float16,
        quant_config=BitsAndBytesConfig(),
        disable_tp=True,
    )

    assert isinstance(layer.weight, BitsAndBytes4bitParameter)
    _assert_uses_weight_loader_v2(layer.weight, layer)

    q = torch.arange(16, dtype=torch.uint8).reshape(16, 1)
    k = torch.arange(16, 24, dtype=torch.uint8).reshape(8, 1)
    v = torch.arange(24, 32, dtype=torch.uint8).reshape(8, 1)

    layer.weight.weight_loader(layer.weight, q, "q")
    layer.weight.weight_loader(layer.weight, k, "k")
    layer.weight.weight_loader(layer.weight, v, "v")

    expected = torch.cat([q, k, v], dim=0)
    assert torch.equal(layer.weight.data, expected)


def test_bitsandbytes_8bit_merged_column_loading_v2() -> None:
    layer = MergedColumnParallelLinear(
        input_size=4,
        output_sizes=[4, 4],
        bias=False,
        params_dtype=torch.float16,
        quant_config=BitsAndBytesConfig(load_in_8bit=True, load_in_4bit=False),
        disable_tp=True,
    )

    assert isinstance(layer.weight, BitsAndBytes8bitParameter)
    _assert_uses_weight_loader_v2(layer.weight, layer)

    shard_0 = torch.arange(16, dtype=torch.int8).reshape(4, 4)
    shard_1 = torch.arange(16, 32, dtype=torch.int8).reshape(4, 4)

    layer.weight.weight_loader(layer.weight, shard_0, 0)
    layer.weight.weight_loader(layer.weight, shard_1, 1)

    expected = torch.cat([shard_0, shard_1], dim=0)
    assert torch.equal(layer.weight.data, expected)
