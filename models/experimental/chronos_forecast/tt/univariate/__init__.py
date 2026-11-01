# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Univariate fast path for Chronos-2 (large-batch throughput, 1xN data parallel)."""

from models.experimental.chronos_forecast.tt.univariate.model import TtChronosUnivariate
from models.experimental.chronos_forecast.tt.univariate.ops import DEVICE_OPTIONS, PRECISION_PRESETS

__all__ = ["DEVICE_OPTIONS", "PRECISION_PRESETS", "TtChronosUnivariate"]
