# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.config.load import LoadConfig
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.model_loader import get_model_loader

from vllm_bnb_plugin import _enforce_eager_for_8bit, register
from vllm_bnb_plugin.bitsandbytes import BitsAndBytesConfig
from vllm_bnb_plugin.bitsandbytes_loader import BitsAndBytesModelLoader


def test_register_bitsandbytes_plugin() -> None:
    register()

    assert get_quantization_config("bitsandbytes") is BitsAndBytesConfig
    assert isinstance(
        get_model_loader(LoadConfig(load_format="bitsandbytes")),
        BitsAndBytesModelLoader,
    )


def test_bitsandbytes_plugin_forces_eager_for_8bit() -> None:
    dummy_model_config = SimpleNamespace(
        model_arch_config=SimpleNamespace(quantization_config={"load_in_8bit": True}),
        enforce_eager=False,
    )

    _enforce_eager_for_8bit(dummy_model_config)

    assert dummy_model_config.enforce_eager is True
