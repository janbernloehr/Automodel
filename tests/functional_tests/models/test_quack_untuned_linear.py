# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QuACK default-GEMM parity and no-autotuning guard for issue #4353.

Run on one Hopper or Blackwell GPU with the CUDA extra installed:
    python -m pytest tests/functional_tests/models/test_quack_untuned_linear.py -q
"""

from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.common.utils import initialize_linear_module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QuACK GEMMs require CUDA")
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("in_features,out_features", [(128, 256), (256, 128), (128, 1024)])
def test_untuned_quack_forward_backward_without_autotuning(bias, recompute, in_features, out_features):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("QuACK GEMMs require Hopper or newer")
    autotuner = pytest.importorskip("quack.autotuner")
    torch.manual_seed(42)
    module = initialize_linear_module(
        "quack_untuned", in_features, out_features, bias=bias, device="cuda", dtype=torch.bfloat16
    )
    reference = nn.Linear(in_features, out_features, bias=bias, device="cuda", dtype=torch.float32)
    reference.load_state_dict(module.state_dict())
    x = torch.randn(2, 32, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().float().requires_grad_(True)
    expected = reference(x_ref)
    grad = torch.randn_like(expected, dtype=torch.bfloat16)
    expected.backward(grad.float())

    # A warm cache must not hide accidental re-entry into the tuned path.
    with patch.object(autotuner.Autotuner, "__call__", side_effect=AssertionError("unexpected GEMM autotuning")):
        output = checkpoint(module, x, use_reentrant=False) if recompute else module(x)
        output.backward(grad)
        torch.cuda.synchronize()

    pairs = [(output, expected), (x.grad, x_ref.grad), (module.weight.grad, reference.weight.grad)]
    if bias:
        pairs.append((module.bias.grad, reference.bias.grad))
    for actual, ref in pairs:
        assert actual is not None
        assert torch.isfinite(actual).all()
        # BF16 rounding permits ~0.4% relative error; 1% normalized L2 also
        # accommodates accumulation order without hiding layout/gradient errors.
        relative_error = (actual.float() - ref).norm() / ref.norm().clamp_min(1e-8)
        assert relative_error < 0.01
