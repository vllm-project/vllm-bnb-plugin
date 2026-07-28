# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from contextlib import nullcontext, suppress
from typing import TYPE_CHECKING, Any, TypeVar, cast

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    BatchFeature,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from vllm import LLM, SamplingParams
from vllm.config.model import ConvertOption, RunnerOption, _get_and_verify_dtype
from vllm.connections import global_http_connection
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.torch_utils import set_default_torch_num_threads

from tests.models.utils import (
    TokensTextLogprobs,
    TokensTextLogprobsPromptLogprobs,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature, dict)


# ---------------------------------------------------------------------------
# example_prompts fixture
# ---------------------------------------------------------------------------
_TEST_DIR = os.path.dirname(__file__)
_TEST_PROMPTS = [os.path.join(_TEST_DIR, "prompts", "example.txt")]


@pytest.fixture
def example_prompts() -> list[str]:
    prompts: list[str] = []
    for filename in _TEST_PROMPTS:
        with open(filename) as f:
            prompts.extend(f.readlines())
    return prompts


# ---------------------------------------------------------------------------
# Autouse fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def init_test_http_connection():
    global_http_connection.reuse_client = False


@pytest.fixture()
def should_do_global_cleanup_after_test(request) -> bool:
    return not request.node.get_closest_marker("skip_global_cleanup")


@pytest.fixture(autouse=True)
def cleanup_fixture(should_do_global_cleanup_after_test: bool):
    yield
    if should_do_global_cleanup_after_test:
        cleanup_dist_env_and_memory()


@pytest.fixture(autouse=True)
def dynamo_reset():
    yield
    torch._dynamo.reset()


# ---------------------------------------------------------------------------
# HfRunner
# ---------------------------------------------------------------------------
def _wrap_device(x: _T, device: str) -> _T:
    if x is None or isinstance(x, (bool,)):
        return x
    if isinstance(x, dict):
        return {k: _wrap_device(v, device) for k, v in x.items()}
    if hasattr(x, "device") and x.device.type == device:
        return x
    return x.to(device)


class HfRunner:
    def __init__(
        self,
        model_name: str,
        dtype: str = "auto",
        *,
        revision: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        trust_remote_code: bool = True,
        is_sentence_transformer: bool = False,
        skip_tokenizer_init: bool = False,
        auto_cls: type[_BaseAutoModelClass] = AutoModelForCausalLM,
        tokenizer_name: str | None = None,
        processor: Any | None = None,
        default_torch_num_threads: int | None = None,
    ) -> None:
        init_ctx = (
            nullcontext()
            if default_torch_num_threads is None
            else set_default_torch_num_threads(default_torch_num_threads)
        )
        with init_ctx:
            model_name = maybe_model_redirect(model_name)
            self.model_name = model_name

            self.config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )
            if self.config.__module__.startswith("vllm.transformers_utils.configs"):
                from transformers.models.auto.configuration_auto import (
                    CONFIG_MAPPING,
                )

                del CONFIG_MAPPING._extra_content[self.config.model_type]
                self.config = AutoConfig.from_pretrained(
                    model_name, trust_remote_code=trust_remote_code
                )

            from vllm.platforms import current_platform

            self.device = (
                "cpu" if current_platform.is_cpu() else current_platform.device_type
            )
            self.dtype = dtype = _get_and_verify_dtype(
                model_name,
                self.config,
                dtype=dtype,
                is_pooling_model=is_sentence_transformer,
                config_format="hf",
            )

            model_kwargs = model_kwargs if model_kwargs is not None else {}
            model_kwargs.setdefault("dtype", dtype)

            if is_sentence_transformer:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(
                    model_name,
                    revision=revision,
                    device=self.device,
                    model_kwargs=model_kwargs,
                    trust_remote_code=trust_remote_code,
                )
            else:
                model = cast(
                    nn.Module,
                    auto_cls.from_pretrained(
                        model_name,
                        revision=revision,
                        trust_remote_code=trust_remote_code,
                        **model_kwargs,
                    ),
                )
                if getattr(model, "quantization_method", None) is None and any(
                    p.dtype != self.dtype for p in model.parameters()
                ):
                    model = model.to(dtype=self.dtype)
                if (
                    getattr(model, "quantization_method", None) is None
                    and len({p.device for p in model.parameters()}) < 2
                ):
                    model = model.to(device=self.device)
                self.model = model

            if not skip_tokenizer_init:
                self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
                    AutoTokenizer.from_pretrained(
                        tokenizer_name or model_name,
                        trust_remote_code=trust_remote_code,
                    )
                )

            if processor is not None:
                self.processor = processor
            else:
                from transformers import AutoProcessor

                self.processor = AutoProcessor.from_pretrained(
                    model_name, trust_remote_code=trust_remote_code
                )
            if skip_tokenizer_init:
                if self.processor is None:
                    raise ValueError(
                        "skip_tokenizer_init=True requires processor initialization."
                    )
                self.tokenizer = self.processor.tokenizer

    def _get_inputs(
        self,
        prompts: list[str] | list[list[int]],
    ) -> list[BatchFeature | BatchEncoding | dict[str, torch.Tensor]]:
        all_inputs: list[BatchFeature | BatchEncoding | dict[str, torch.Tensor]] = []
        for prompt in prompts:
            if isinstance(prompt, str):
                inputs = self.processor(text=prompt, return_tensors="pt")
                if isinstance(inputs, BatchFeature):
                    inputs = inputs.to(dtype=self.dtype)
                all_inputs.append(inputs)
            else:
                all_inputs.append(
                    {
                        "input_ids": torch.tensor(prompt, dtype=torch.long).unsqueeze(
                            0
                        ),
                    }
                )
        return all_inputs

    def generate_greedy(
        self,
        prompts: list[str] | list[list[int]],
        max_tokens: int,
        **kwargs: Any,
    ) -> list[tuple[list[int], str]]:
        all_inputs = self._get_inputs(prompts)
        outputs: list[tuple[list[int], str]] = []
        for inputs in all_inputs:
            output_ids: torch.Tensor = self.model.generate(
                **_wrap_device(inputs, self.device),
                use_cache=True,
                do_sample=False,
                max_new_tokens=max_tokens,
                **kwargs,
            )
            output_str = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outputs.append((output_ids[0].tolist(), output_str[0]))
        return outputs

    def generate_greedy_logprobs_limit(
        self,
        prompts: list[str],
        max_tokens: int,
        num_logprobs: int | None,
        **kwargs: Any,
    ) -> list[TokensTextLogprobs]:
        all_inputs = self._get_inputs(prompts)
        all_logprobs: list[list[dict[int, float]]] = []
        all_output_ids: list[list[int]] = []
        all_output_strs: list[str] = []
        for inputs in all_inputs:
            output = self.model.generate(
                **_wrap_device(inputs, self.device),
                use_cache=True,
                do_sample=False,
                max_new_tokens=max_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs,
            )
            hidden_states = (
                getattr(output, "hidden_states", None) or output.decoder_hidden_states
            )
            # Compute logprobs from hidden states
            output_embeddings = self.model.get_output_embeddings()
            seq_logprobs_lst: list[dict[int, float]] = []
            for tok_idx, hidden_state in enumerate(hidden_states):
                last_hidden = hidden_state[-1][0]
                logits = torch.matmul(
                    last_hidden.to(
                        device=output_embeddings.weight.device,
                        dtype=output_embeddings.weight.dtype,
                    ),
                    output_embeddings.weight.t(),
                )
                if getattr(output_embeddings, "bias", None) is not None:
                    logits += output_embeddings.bias.unsqueeze(0)
                tok_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                if tok_idx == 0:
                    tok_logprobs = tok_logprobs[-1, :].reshape(1, -1)
                topk = tok_logprobs.topk(num_logprobs)
                tok_logprobs_dct = {}
                for token_id, logprob in zip(topk.indices[0], topk.values[0]):
                    tok_logprobs_dct[token_id.item()] = logprob.item()
                seq_logprobs_lst.append(tok_logprobs_dct)

            all_logprobs.append(seq_logprobs_lst)
            seq_ids = output.sequences[0]
            output_len = len(seq_logprobs_lst)
            output_ids = seq_ids[-output_len:]
            all_output_ids.append(output_ids.tolist())
            all_output_strs.append(self.tokenizer.decode(output_ids))
        return [
            (ids, s, lp)
            for ids, s, lp in zip(all_output_ids, all_output_strs, all_logprobs)
        ]

    def encode(self, prompts: list[str], *args, **kwargs) -> list[list[torch.Tensor]]:
        return self.model.encode(prompts, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup_dist_env_and_memory()


@pytest.fixture(scope="session")
def hf_runner():
    return HfRunner


# ---------------------------------------------------------------------------
# VllmRunner
# ---------------------------------------------------------------------------
class VllmRunner:
    def __init__(
        self,
        model_name: str,
        runner: RunnerOption = "auto",
        convert: ConvertOption = "auto",
        tokenizer_name: str | None = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = True,
        seed: int = 0,
        max_model_len: int | None = 1024,
        dtype: str = "auto",
        disable_log_stats: bool = True,
        tensor_parallel_size: int = 1,
        block_size: int = (16 if not torch.xpu.is_available() else 64),
        enable_chunked_prefill: bool | None = False,
        enforce_eager: bool | None = False,
        default_torch_num_threads: int | None = None,
        **kwargs,
    ) -> None:
        init_ctx = (
            nullcontext()
            if default_torch_num_threads is None
            else set_default_torch_num_threads(default_torch_num_threads)
        )
        if not kwargs.get("compilation_config"):
            kwargs["compilation_config"] = {"cudagraph_capture_sizes": [4]}
        with init_ctx:
            self.llm = LLM(
                model=model_name,
                runner=runner,
                convert=convert,
                tokenizer=tokenizer_name,
                tokenizer_mode=tokenizer_mode,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                seed=seed,
                enforce_eager=enforce_eager,
                disable_log_stats=disable_log_stats,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                block_size=block_size,
                enable_chunked_prefill=enable_chunked_prefill,
                **kwargs,
            )

    def _get_inputs(
        self,
        prompts: (
            list[str] | list[torch.Tensor] | list[list[int]] | list[dict[str, Any]]
        ),
    ) -> list[dict[str, Any]]:
        inputs = list[dict[str, Any]]()
        for prompt in prompts:
            if isinstance(prompt, dict):
                inputs.append(prompt.copy())
            else:
                prompt_dict = dict[str, Any]()
                if isinstance(prompt, str):
                    prompt_dict["prompt"] = prompt
                elif isinstance(prompt, list):
                    prompt_dict["prompt_token_ids"] = prompt
                else:
                    prompt_dict["prompt_embeds"] = prompt
                inputs.append(prompt_dict)
        return inputs

    def generate_greedy(
        self,
        prompts: list[str] | list[torch.Tensor] | list[list[int]],
        max_tokens: int,
        **kwargs: Any,
    ) -> list[tuple[list[int], str]]:
        inputs = self._get_inputs(prompts)
        greedy_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        req_outputs = self.llm.generate(inputs, sampling_params=greedy_params, **kwargs)
        outputs: list[tuple[list[int], str]] = []
        for req_output in req_outputs:
            for sample in req_output.outputs:
                output_ids = list(req_output.prompt_token_ids) + list(sample.token_ids)
                output_str = (req_output.prompt or "") + sample.text
                outputs.append((output_ids, output_str))
        return outputs

    def generate_greedy_logprobs(
        self,
        prompts: list[str],
        max_tokens: int,
        num_logprobs: int | None,
        num_prompt_logprobs: int | None = None,
        stop_token_ids: list[int] | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> list[TokensTextLogprobs] | list[TokensTextLogprobsPromptLogprobs]:
        inputs = self._get_inputs(prompts)
        greedy_logprobs_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=num_logprobs,
            prompt_logprobs=num_prompt_logprobs,
            stop_token_ids=stop_token_ids,
            stop=stop,
        )
        req_outputs = self.llm.generate(
            inputs, sampling_params=greedy_logprobs_params, **kwargs
        )
        # Extract logprobs from request outputs
        outputs: list[TokensTextLogprobsPromptLogprobs] = []
        for req_output in req_outputs:
            assert len(req_output.outputs) > 0
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = list(sample.token_ids)
                output_logprobs = sample.logprobs
            outputs.append(
                (
                    output_ids,
                    output_str,
                    output_logprobs,
                    req_output.prompt_logprobs,
                )
            )
        return (
            [x[0:-1] for x in outputs]
            if greedy_logprobs_params.prompt_logprobs is None
            else outputs
        )

    def embed(
        self,
        prompts: list[str],
        *args,
        **kwargs,
    ) -> list[list[float]]:
        inputs = self._get_inputs(prompts)
        req_outputs = self.llm.embed(inputs, *args, **kwargs)
        return [req_output.outputs.embedding for req_output in req_outputs]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with suppress(Exception):
            self.llm.llm_engine.engine_core.shutdown()
        del self.llm
        cleanup_dist_env_and_memory()


@pytest.fixture(scope="session")
def vllm_runner():
    return VllmRunner


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "distributed(num_gpus): mark a test that requires multiple GPUs.",
    )
