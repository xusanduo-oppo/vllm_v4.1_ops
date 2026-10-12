# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import torch

from .allclose_default import get_default_atol, get_default_rtol
from .utils import opcheck
from vllm.model_executor.layers.activation import (
    FastGELU,
    FatreluAndMul,
    GeluAndMul,
    MulAndSilu,
    NewGELU,
    QuickGELU,
    SiluAndMul,
    SiluAndMulWithClamp,
    SwigluOAIAndMul,
    SwigluStepAndMul,
    swiglustep_and_mul_triton,
)
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048]  # Arbitrary values for testing
D = [512, 13824]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]




SWIGLU_LIMITS = [3.0, 7.0, 15.0]


@pytest.mark.parametrize("swiglu_limit", SWIGLU_LIMITS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_with_clamp(
    swiglu_limit: float,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    """SiluAndMulWithClamp: cuda kernel must match native reference."""
    set_random_seed(seed)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        torch.set_default_device(device)
        # Use large values to ensure clamping is exercised.
        x = torch.randn(num_tokens, 2 * d, dtype=dtype) * swiglu_limit * 2

        layer = SiluAndMulWithClamp(swiglu_limit, compile_native=False)
        out = layer(x)
        ref_out = layer.forward_native(x)

        rtol = {
            torch.float16: 2e-3,
            torch.bfloat16: 2e-2,
            torch.float: 1.3e-6,
        }
        torch.testing.assert_close(
            out, ref_out, atol=get_default_atol(out), rtol=rtol[out.dtype]
        )

        # Verify clamping is actually being applied: the clamped output should
        # differ from the unclamped SiluAndMul output when inputs are large.
        unclamped_out = SiluAndMul.forward_native(x)
        assert not torch.equal(ref_out.float(), unclamped_out.float()), (
            "Input was not large enough to exercise the clamp; increase scale"
        )

        # Verify gate clamping semantics with a controlled scalar case.
        # gate=large_val is clamped to limit first, then silu(limit) * 1.0.
        x_gate = torch.tensor(
            [[swiglu_limit * 20.0, 1.0]], dtype=torch.float32, device=device
        )
        out_gate = SiluAndMulWithClamp(swiglu_limit, compile_native=False)(x_gate)
        expected_gate = torch.nn.functional.silu(
            torch.tensor(swiglu_limit, dtype=torch.float32)
        ).item()
        torch.testing.assert_close(
            out_gate,
            torch.tensor([[expected_gate]], dtype=torch.float32, device=device),
            atol=1e-3,
            rtol=1e-3,
        )

        # Verify up clamping semantics: up >> limit gets clamped to limit.
        x_up = torch.tensor(
            [[1.0, swiglu_limit * 20.0]], dtype=torch.float32, device=device
        )
        out_up = SiluAndMulWithClamp(swiglu_limit, compile_native=False)(x_up)
        silu_1 = torch.nn.functional.silu(torch.tensor(1.0)).item()
        torch.testing.assert_close(
            out_up,
            torch.tensor([[silu_1 * swiglu_limit]], dtype=torch.float32, device=device),
            atol=1e-3,
            rtol=1e-3,
        )

        # opcheck
        out_buf = torch.empty(x.shape[:-1] + (d,), dtype=dtype, device=device)
        opcheck(torch.ops._C.silu_and_mul_with_clamp, (out_buf, x, swiglu_limit))


if __name__ == "__main__":
    pass
    # pytest -q /root/Code/script/V4/ut/moe/test_activation.py --log-cli-level=DEBUG -v -s --junit-xml=/root/Code/script/V4/ut/moe/test_activation.xml