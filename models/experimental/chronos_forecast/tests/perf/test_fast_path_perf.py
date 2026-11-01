# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end timing of TtChronos.forward on its fast path: host arrays in, host forecasts out.

Context 512, horizon 64 (4 output patches). Prints steady-state latency and series/s per batch;
asserts only shapes, like the paper-shape perf tests.
"""

from __future__ import annotations

import statistics
import time

import pytest

from models.experimental.chronos_forecast.tests.mesh_params import MULTI_CHIP_MESHES
from models.experimental.chronos_forecast.tests.perf.test_paper_forward import CKPT, _checkpoint_ready
from models.experimental.chronos_forecast.tests.test_fast_path import (
    DEVICE_PARAMS,
    NUM_OUTPUT_PATCHES,
    PRECISIONS,
    _series,
)

WARMUP = 3
MIN_SECONDS = 5.0
MIN_ITERS = 10

pytestmark = pytest.mark.skipif(not _checkpoint_ready(), reason=f"amazon/chronos-2 checkpoint not in {CKPT}")


def _time_batches(mesh_device, precision_name, batches):
    from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model
    from models.experimental.chronos_forecast.tt.model import TtChronos
    from models.experimental.chronos_forecast.tt.program_configs import TtChronosPrecision

    precision = TtChronosPrecision.performance() if precision_name == "performance" else TtChronosPrecision()
    model = TtChronos.from_torch_model(mesh_device, Chronos2Model.from_pretrained(str(CKPT)), precision)
    rows = []
    for batch in batches:
        context = _series(batch, seed=batch)
        assert model.uses_fast_path(context, num_output_patches=NUM_OUTPUT_PATCHES)
        for _ in range(WARMUP):
            out = model.forward(context, num_output_patches=NUM_OUTPUT_PATCHES)
        times, start = [], time.perf_counter()
        while len(times) < MIN_ITERS or time.perf_counter() - start < MIN_SECONDS:
            t0 = time.perf_counter()
            out = model.forward(context, num_output_patches=NUM_OUTPUT_PATCHES)
            times.append(time.perf_counter() - t0)
        assert out.shape == (batch, 21, NUM_OUTPUT_PATCHES * 16)
        p50 = statistics.median(times)
        rows.append((batch, p50, sorted(times)[int(0.99 * (len(times) - 1))], batch / p50))
    print(f"\n[FAST PATH PERF] {precision_name}, {mesh_device.get_num_devices()} chip(s), context 512, horizon 64")
    for batch, p50, p99, rate in rows:
        print(f"  b{batch:<5d} p50 {p50 * 1e3:8.2f} ms  p99 {p99 * 1e3:8.2f} ms  {rate:9.0f} series/s")
    return rows


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("precision_name", PRECISIONS)
def test_fast_path_perf(mesh_device, precision_name):
    _time_batches(mesh_device, precision_name, (1, 4, 64, 256, 1024))


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MULTI_CHIP_MESHES, indirect=True)
@pytest.mark.parametrize("precision_name", PRECISIONS)
def test_fast_path_perf_data_parallel(mesh_device, precision_name):
    _time_batches(mesh_device, precision_name, (64, 1024, 4096))
