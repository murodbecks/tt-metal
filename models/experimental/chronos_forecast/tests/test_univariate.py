# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Univariate fast path (tt/univariate) against the reference Chronos-2 model.

Needs the amazon/chronos-2 checkpoint in models/experimental/chronos_forecast/weights/chronos-2
(skipped otherwise). Gate: max per-row quantile NRMSE <= 0.04 against the FP32 reference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from models.experimental.chronos_forecast.tests.mesh_params import MULTI_CHIP_MESHES
from models.experimental.chronos_forecast.tests.perf.test_paper_forward import CKPT, _checkpoint_ready

CONTEXT = 512
NUM_OUTPUT_PATCHES = 4  # 64 steps
MAX_NRMSE = 0.04
DEVICE_PARAMS = [{"trace_region_size": 32 * 1024 * 1024, "num_command_queues": 2}]

pytestmark = pytest.mark.skipif(not _checkpoint_ready(), reason=f"amazon/chronos-2 checkpoint not in {CKPT}")


def _series(batch, length=CONTEXT, seed=0):
    """Distinct seasonal series with trend and noise, different scales per row."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(length, dtype=torch.float32)
    period = 12 + 36 * torch.rand(batch, 1, generator=g)
    level = 10 ** (4 * torch.rand(batch, 1, generator=g) - 1)
    trend = 0.002 * torch.randn(batch, 1, generator=g) * t
    noise = 0.1 * torch.randn(batch, length, generator=g)
    return level * (1 + 0.5 * torch.sin(2 * math.pi * t / period) + trend + noise)


def _max_row_nrmse(actual, expected):
    a, e = actual.double().flatten(1), expected.double().flatten(1)
    error = (a - e).pow(2).mean(1).sqrt() / e.pow(2).mean(1).sqrt().clamp_min(1e-12)
    return float(error.max())


@pytest.fixture(scope="module")
def reference():
    from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model

    model = Chronos2Model.from_pretrained(str(CKPT)).eval()
    model.config._attn_implementation = "eager"
    return model


def _reference_forecast(reference, context, context_mask=None):
    with torch.no_grad():
        out = reference(context=context, context_mask=context_mask, num_output_patches=NUM_OUTPUT_PATCHES)
    return out.quantile_preds.float()


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("precision", ["bf16", "bfp8_b"])
@pytest.mark.parametrize("batch", [1, 4, 64, 256])
def test_univariate_matches_reference(mesh_device, reference, precision, batch):
    from models.experimental.chronos_forecast.tt.univariate import TtChronosUnivariate

    model = TtChronosUnivariate.from_torch_model(mesh_device, reference, precision)
    context = _series(batch)
    expected = _reference_forecast(reference, context)
    actual = model(context, num_output_patches=NUM_OUTPUT_PATCHES)
    assert actual.shape == expected.shape
    nrmse = _max_row_nrmse(actual, expected)
    print(f"univariate {precision} b{batch}: max row NRMSE {nrmse:.4f}")
    assert nrmse <= MAX_NRMSE
    # traced replays are bit-identical
    assert torch.equal(model(context, num_output_patches=NUM_OUTPUT_PATCHES), actual)


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_univariate_missing_values_and_row_independence(mesh_device, reference):
    from models.experimental.chronos_forecast.tt.univariate import TtChronosUnivariate

    model = TtChronosUnivariate.from_torch_model(mesh_device, reference, "bf16")
    context = _series(64, seed=1)
    context[:8, -100:] = float("nan")  # missing tail
    context[8:16, :300] = float("nan")  # short history
    context[16:24] = 7.0  # constant series
    expected = _reference_forecast(reference, context)
    actual = model(context, num_output_patches=NUM_OUTPUT_PATCHES)
    assert _max_row_nrmse(actual, expected) <= MAX_NRMSE
    # a row's forecast does not depend on the other rows in the batch
    single = model(context[5:6], num_output_patches=NUM_OUTPUT_PATCHES)
    assert _max_row_nrmse(single, actual[5:6]) <= 0.01


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MULTI_CHIP_MESHES, indirect=True)
@pytest.mark.parametrize("precision", ["bf16", "bfp8_b"])
def test_univariate_data_parallel(mesh_device, reference, precision):
    """1xN mesh: weights replicated, rows sharded; uneven batches are padded per card."""
    from models.experimental.chronos_forecast.tt.univariate import TtChronosUnivariate

    model = TtChronosUnivariate.from_torch_model(mesh_device, reference, precision)
    for batch in (mesh_device.get_num_devices() + 2, 1024):
        context = _series(batch, seed=batch)
        expected = _reference_forecast(reference, context)
        actual = model(context, num_output_patches=NUM_OUTPUT_PATCHES)
        nrmse = _max_row_nrmse(actual, expected)
        print(f"univariate {precision} mesh {mesh_device.get_num_devices()} b{batch}: max row NRMSE {nrmse:.4f}")
        assert np.isfinite(nrmse) and nrmse <= MAX_NRMSE
