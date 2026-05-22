# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import os
import sys
from pathlib import Path


def _resolve_vllm_repo_root() -> Path:
    candidates: list[Path] = []

    if repo_override := os.environ.get("VLLM_TEST_REPO"):
        candidates.append(Path(repo_override).expanduser().resolve())

    candidates.append(Path(__file__).resolve().parents[2] / "vllm-remove-bnb")

    for candidate in candidates:
        if (candidate / "tests" / "conftest.py").is_file():
            return candidate

    raise RuntimeError(
        "Could not find the vLLM repository needed for plugin end-to-end tests. "
        "Set VLLM_TEST_REPO to a checkout containing tests/conftest.py."
    )


VLLM_REPO_ROOT = _resolve_vllm_repo_root()
if str(VLLM_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_REPO_ROOT))

_tests_spec = importlib.util.spec_from_file_location(
    "tests",
    VLLM_REPO_ROOT / "tests" / "__init__.py",
    submodule_search_locations=[str(VLLM_REPO_ROOT / "tests")],
)
if _tests_spec is None or _tests_spec.loader is None:
    raise RuntimeError("Failed to load the upstream vLLM tests package.")

_tests_module = importlib.util.module_from_spec(_tests_spec)
sys.modules["tests"] = _tests_module
_tests_spec.loader.exec_module(_tests_module)

_spec = importlib.util.spec_from_file_location(
    "_vllm_repo_tests_conftest",
    VLLM_REPO_ROOT / "tests" / "conftest.py",
)
if _spec is None or _spec.loader is None:
    raise RuntimeError("Failed to load vLLM test fixtures from tests/conftest.py.")

_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

for _name in dir(_module):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_module, _name)


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "distributed(num_gpus): mark a test that requires multiple GPUs.",
    )
