# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared binary_backward device operation (issue #56601).

Exercises MUL_BW as the first op routed through the shared device op. Covers
dtypes x shapes vs torch, mixed operand dtypes (input vs other vs grad_output),
memory configs, preallocated outputs, program-cache keying, and validation
rejections that flip the caller back onto the composite path.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


_TORCH_OF = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
}


def _pt_and_tt(shape, low, high, device, dtype, memory_config, seed=213919):
    torch.manual_seed(seed)
    torch_dtype = _TORCH_OF.get(dtype, torch.bfloat16)
    pt = torch.rand(shape, dtype=torch_dtype) * (high - low) + low
    tt = ttnn.from_torch(
        pt.float() if torch_dtype != torch.float32 else pt,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=memory_config,
    )
    # bfloat8_b round-trips through the tile packer, so read back the exact
    # value the device sees for the golden comparison.
    return ttnn.to_torch(tt).float(), tt


def _torch_mul_bw(grad, a, b):
    # d(a*b)/da = grad*b, d(a*b)/db = grad*a
    return grad * b, grad * a


# ---------------------------------------------------------------------------
# forward correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),
        (1, 1, 320, 384),
        (1, 3, 320, 384),
        (4, 8, 512, 512),
    ],
)
# Accuracy bar per dtype. The bit-addressable floats (bfloat16, float32) are checked
# ELEMENTWISE: PCC is a correlation over the whole tile, so it can mask a localized per-element
# error, and these dtypes can resolve one. The block float types keep PCC -- they carry one
# shared exponent per 16 values, so an individually small element legitimately flushes to zero,
# and an elementwise bound there would be measuring the storage format rather than the kernel.
#
# Bars sit above measurement with headroom, not at a dtype epsilon. mul_bw is a pure multiply
# with no SFPU op, so error is dominated by the pack-back rounding of grad*other and grad*input
# to the operand dtype -- roughly one ulp of that dtype, steady across shape.
@pytest.mark.parametrize(
    "dtype, rtol, expected_pcc",
    [
        (ttnn.bfloat16, 8e-3, None),
        (ttnn.float32, 1e-3, None),
        (ttnn.bfloat8_b, None, 0.99),
        (ttnn.bfloat4_b, None, 0.93),
    ],
)
@pytest.mark.parametrize(
    "memory_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    ids=["dram", "l1"],
)
def test_mul_bw_correctness(shape, dtype, rtol, expected_pcc, memory_config, device):
    if shape == (4, 8, 512, 512) and dtype == ttnn.float32 and memory_config == ttnn.L1_MEMORY_CONFIG:
        pytest.skip("fp32 (4,8,512,512) x 5 buffers exceeds L1 budget")
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, dtype, memory_config, seed=213919)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, dtype, memory_config, seed=213920)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, dtype, memory_config, seed=213921)

    grad_a_pt, grad_b_pt = _torch_mul_bw(g_pt, a_pt, b_pt)

    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=memory_config)
    grad_a_tt = ttnn.to_torch(out[0]).float()
    grad_b_tt = ttnn.to_torch(out[1]).float()

    assert out[0].dtype == a_tt.dtype, f"input_grad dtype {out[0].dtype} != input dtype {a_tt.dtype}"
    assert out[1].dtype == b_tt.dtype, f"other_grad dtype {out[1].dtype} != other dtype {b_tt.dtype}"

    if expected_pcc is not None:
        assert_with_pcc(grad_a_pt, grad_a_tt, expected_pcc)
        assert_with_pcc(grad_b_pt, grad_b_tt, expected_pcc)
        return

    torch.testing.assert_close(grad_a_tt, grad_a_pt, rtol=rtol, atol=1e-4)
    torch.testing.assert_close(grad_b_tt, grad_b_pt, rtol=rtol, atol=1e-4)


# ---------------------------------------------------------------------------
# mixed operand dtypes — the exact class the tanh_bw factory bug bit (#56061)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "grad_dtype,a_dtype,b_dtype",
    [
        (ttnn.float32, ttnn.bfloat16, ttnn.bfloat16),
        (ttnn.bfloat16, ttnn.float32, ttnn.bfloat16),
        (ttnn.bfloat16, ttnn.bfloat16, ttnn.float32),
    ],
)
def test_mul_bw_mixed_operand_dtypes(grad_dtype, a_dtype, b_dtype, device):
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, a_dtype, mc, seed=213919)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, b_dtype, mc, seed=213920)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, grad_dtype, mc, seed=213921)

    grad_a_pt, grad_b_pt = _torch_mul_bw(g_pt, a_pt, b_pt)

    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=mc)
    grad_a_tt = ttnn.to_torch(out[0]).float()
    grad_b_tt = ttnn.to_torch(out[1]).float()

    assert_with_pcc(grad_a_pt, grad_a_tt, 0.999)
    assert_with_pcc(grad_b_pt, grad_b_tt, 0.999)


# ---------------------------------------------------------------------------
# preallocated outputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preallocate",
    ["both", "input_only", "other_only"],
)
def test_mul_bw_preallocated(preallocate, device):
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=213919)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, seed=213920)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=213921)

    input_grad = None
    other_grad = None
    if preallocate in ("both", "input_only"):
        input_grad = ttnn.empty_like(a_tt)
    if preallocate in ("both", "other_only"):
        other_grad = ttnn.empty_like(b_tt)

    grad_a_pt, grad_b_pt = _torch_mul_bw(g_pt, a_pt, b_pt)

    out = ttnn.mul_bw(
        g_tt,
        a_tt,
        b_tt,
        are_required_outputs=[True, True],
        memory_config=mc,
        input_grad=input_grad,
        other_grad=other_grad,
    )
    assert_with_pcc(grad_a_pt, ttnn.to_torch(out[0]).float(), 0.999)
    assert_with_pcc(grad_b_pt, ttnn.to_torch(out[1]).float(), 0.999)


# ---------------------------------------------------------------------------
# partial mask stays on the composite path (device op contract requires both)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mask", [[True, False], [False, True]])
def test_mul_bw_partial_mask_routes_to_composite(mask, device):
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=213919)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, seed=213920)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=213921)

    out = ttnn.mul_bw(g_tt, a_tt, b_tt, are_required_outputs=mask, memory_config=mc)

    grad_a_pt, grad_b_pt = _torch_mul_bw(g_pt, a_pt, b_pt)
    if mask[0]:
        assert_with_pcc(grad_a_pt, ttnn.to_torch(out[0]).float(), 0.999)
    if mask[1]:
        assert_with_pcc(grad_b_pt, ttnn.to_torch(out[1]).float(), 0.999)


# ---------------------------------------------------------------------------
# program-cache keying: one entry per dtype-set for the device-op path
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# broadcasting operands: TILE padding equalises padded_shape when logical_shape
# still differs (e.g. (1,1,32,128) vs (1,1,1,128) both pad to (1,1,32,128)).
# A routing gate that only compared padded_shape would leak this through, and
# the device op walks tile-for-tile with no broadcast, so it would multiply
# grad against padding bytes and silently return wrong-answer. The property
# this test pins: mul_bw never returns silent wrong-answer for broadcasting
# operands — either the composite handles it, or it raises. The device op's
# own logical_shape TT_FATAL guarantees that if the routing gate regresses,
# the failure will be loud and traceable to binary_backward_device_operation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "grad_shape,input_shape,other_shape",
    [
        ((1, 1, 32, 128), (1, 1, 32, 128), (1, 1, 1, 128)),  # other broadcasts row-wise
        ((1, 1, 32, 128), (1, 1, 1, 128), (1, 1, 32, 128)),  # input broadcasts row-wise
    ],
)
def test_mul_bw_broadcast_never_silently_wrong(grad_shape, input_shape, other_shape, device):
    mc = ttnn.DRAM_MEMORY_CONFIG
    _, a_tt = _pt_and_tt(input_shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=1)
    _, b_tt = _pt_and_tt(other_shape, -5.0, 5.0, device, ttnn.bfloat16, mc, seed=2)
    _, g_tt = _pt_and_tt(grad_shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)

    try:
        ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=mc)
    except RuntimeError as exc:
        # If the routing gate at binary_backward.cpp:841 regresses, this fires
        # in binary_backward_device_operation with the MUL_BW-specific message.
        assert "MUL_BW operation requires" not in str(
            exc
        ), f"routing gate leaked a broadcasting call into the device op:\n{exc}"


def test_mul_bw_program_cache_keying(device):
    shape = (1, 1, 320, 384)
    mc = ttnn.DRAM_MEMORY_CONFIG

    def _run(dtype):
        _, a = _pt_and_tt(shape, -1.0, 1.0, device, dtype, mc, seed=1)
        _, b = _pt_and_tt(shape, -5.0, 5.0, device, dtype, mc, seed=2)
        _, g = _pt_and_tt(shape, -3.0, 3.0, device, dtype, mc, seed=3)
        return ttnn.mul_bw(g, a, b, memory_config=mc)

    start = device.num_program_cache_entries()
    _run(ttnn.bfloat16)
    _run(ttnn.bfloat16)  # second call must not add an entry
    after_bf16 = device.num_program_cache_entries()
    added_bf16 = after_bf16 - start
    assert added_bf16 == 1, (
        f"expected 1 program cache entry for two identical bfloat16 mul_bw calls (second a hit), got {added_bf16}; "
        "more means the hash separates runs it should share, fewer means it collides distinct programs"
    )
    _run(ttnn.float32)
    added_f32 = device.num_program_cache_entries() - after_bf16
    assert added_f32 == 1, (
        f"expected 1 additional entry when switching bfloat16 -> float32 mul_bw, got {added_f32}; "
        "0 means the hash collides dtypes, >1 means the second float32 program was not cached"
    )


# ---------------------------------------------------------------------------
# Rejection paths. The composite mul_bw predates the device op and accepts a
# wider surface than the fused kernel can serve, so the routing gate and the
# device op's own TT_FATALs together decide what the shared path never
# silently accepts. These tests pin those decisions.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_role", ["grad", "input", "other"])
def test_mul_bw_rejects_int_operands(bad_role, device):
    # Any int/uint operand must be rejected per-role with a message naming the
    # offending role and dtype, regardless of what the other two carry.
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    _, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=1)
    _, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, seed=2)
    _, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)

    int_tensor = ttnn.zeros(shape, dtype=ttnn.int32, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mc)
    grad, input_, other = (
        (int_tensor, a_tt, b_tt)
        if bad_role == "grad"
        else (g_tt, int_tensor, b_tt)
        if bad_role == "input"
        else (g_tt, a_tt, int_tensor)
    )

    with pytest.raises(RuntimeError, match="only supports floating-point dtypes"):
        ttnn.mul_bw(grad, input_, other, memory_config=mc)


def test_mul_bw_rejects_preallocated_output_dtype_mismatch(device):
    # A preallocated grad tensor whose dtype differs from the corresponding
    # operand is rejected before the kernel binds the wrong CB format.
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    _, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=1)
    _, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, seed=2)
    _, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)

    wrong_dtype = ttnn.zeros(shape, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mc)
    correct = ttnn.empty_like(b_tt)

    with pytest.raises(RuntimeError, match="dtype to match its operand"):
        ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=mc, input_grad=wrong_dtype, other_grad=correct)


def test_mul_bw_sharded_stays_on_composite(device):
    # Sharded operands, or a sharded output_mem_config, must fall back to the
    # composite path rather than being rejected: the composite is built from
    # unary and binary ops that support sharding, the fused device op does not.
    # This is the same class of regression sigmoid_bw hit in #56061 (b59f52d).
    shape = (1, 1, 32, 32)
    sharded_mc = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, sharded_mc, seed=1)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, sharded_mc, seed=2)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, sharded_mc, seed=3)

    # If the routing gate regresses and lets sharded operands reach the device
    # op, the per-operand sharded rejection fires with the MUL_BW-keyed message;
    # the composite accepts them and returns a matching-precision result.
    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=sharded_mc)
    grad_a_pt, grad_b_pt = _torch_mul_bw(g_pt, a_pt, b_pt)
    assert_with_pcc(grad_a_pt, ttnn.to_torch(out[0]).float(), 0.999)
    assert_with_pcc(grad_b_pt, ttnn.to_torch(out[1]).float(), 0.999)
