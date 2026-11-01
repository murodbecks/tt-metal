# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Fused split-heads + RoPE for the fast path's large batches: q, k, v [B, H, S_pad, 64] from one
qkv [B, S_pad, 3 * H * 64] read, with HF rotate_half RoPE applied to q and k (d_kv = 64, two tiles,
so rotate_half is a swap of the two tiles)."""

from __future__ import annotations

import math
import os

import ttnn

TILE = 32
SPLIT_ROPE_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
SPLIT_ROPE_KERNELS = ("reader_split_rope.cpp", "writer_split_rope.cpp", "compute_split_rope.cpp")


def _tile_cb(index, num_pages, cores):
    page = 2 * TILE * TILE  # bf16 tile
    return ttnn.CBDescriptor(
        total_size=num_pages * page,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=page)],
    )


def split_rope(
    qkv, cos, sin, *, num_heads, kernel_dir, compute_kernel_config=None, memory_config=ttnn.DRAM_MEMORY_CONFIG
):
    """q, k, v [B, H, S_pad, 64] bf16 from qkv [B, S_pad, 3 * H * 64] bf16 (tile layout), with HF RoPE
    applied to q and k. cos / sin: [1, 1, S_pad, 64] bf16. A work unit is (batch, tile row, head);
    units are split evenly over the compute grid. The cos / sin tiles of a tile row are shared by its
    heads, so each core reads them once per run of consecutive units on the same tile row and the
    compute kernel keeps them in its circular buffer until the row changes (reuse within a core, not
    a NoC multicast)."""
    d_kv = 2 * TILE
    width = tuple(qkv.padded_shape)[-1]
    if width != 3 * num_heads * d_kv or qkv.dtype != ttnn.bfloat16:
        raise ValueError(f"split_rope: needs bf16 qkv of width 3 * {num_heads} * {d_kv}, got {width} {qkv.dtype}")
    wt = width // TILE
    rows = math.prod(qkv.padded_shape) // (TILE * width)
    s_pad = int(tuple(cos.padded_shape)[-2])
    rt = s_pad // TILE
    batch = rows // rt
    device = qkv.device()
    shape = ttnn.Shape([batch, num_heads, s_pad, d_kv])
    q, k, v = (
        ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config) for _ in range(3)
    )
    _, cores, group_1, _, n1, n2 = ttnn.split_work_to_cores(
        device.compute_with_storage_grid_size(), rows * num_heads, True
    )
    g1 = group_1.num_cores()
    work, start = [], 0
    for i, core in enumerate(ttnn.corerange_to_cores(cores, row_wise=True)):
        n = n1 if i < g1 else n2
        work.append(((core.x, core.y), n, start))
        start += n
    xargs = list(ttnn.TensorAccessorArgs(qkv).get_compile_time_args())
    targs = list(ttnn.TensorAccessorArgs(cos).get_compile_time_args())
    oargs = list(ttnn.TensorAccessorArgs(q).get_compile_time_args())
    xa, ca, sa = qkv.buffer_address(), cos.buffer_address(), sin.buffer_address()
    qa, ka, va = q.buffer_address(), k.buffer_address(), v.buffer_address()
    ck = compute_kernel_config
    fid = ck.math_fidelity if ck is not None else ttnn.MathFidelity.HiFi4
    acc = ck.fp32_dest_acc_en if ck is not None else True

    def kern(name, ct, rt_args, config):
        return ttnn.KernelDescriptor(
            kernel_source=os.path.join(kernel_dir, name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=rt_args,
            config=config,
        )

    kernels = [
        kern(
            "reader_split_rope.cpp",
            [rt, num_heads, wt, len(xargs), *xargs, *targs],
            [(c, [xa, ca, sa, n, s]) for c, n, s in work],
            ttnn.ReaderConfigDescriptor(),
        ),
        kern(
            "writer_split_rope.cpp",
            [rt, num_heads, *oargs],
            [(c, [qa, ka, va, n, s]) for c, n, s in work],
            ttnn.WriterConfigDescriptor(),
        ),
        kern(
            "compute_split_rope.cpp",
            [rt, num_heads],
            [(c, [n, s]) for c, n, s in work],
            ttnn.ComputeConfigDescriptor(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=acc),
        ),
    ]
    # c0: q/k tile pairs, c1: cos/sin tiles (c0 c1 s0 s1), c2: v pass-through, c16: rotated out, c24: cross terms
    cbs = [
        _tile_cb(0, 4, cores),
        _tile_cb(1, 8, cores),
        _tile_cb(2, 4, cores),
        _tile_cb(16, 4, cores),
        _tile_cb(24, 2, cores),
    ]
    ttnn.generic_op([qkv, cos, sin, q, k, v], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs))
    return q, k, v
