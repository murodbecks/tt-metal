// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

// RoPE for d_kv = 64 (two tiles), so HF rotate_half is a swap of the two tiles:
//   out0 = x0 * c0 - x1 * s0,   out1 = x1 * c1 + x0 * s1.
// The cross products go through cb_tmp; the direct products stay in DST and the cross term is
// subtracted / added with dest reuse. Runs twice per unit (q, then k); v bypasses compute.
// The cos/sin tiles stay in cb_cs for every consecutive unit of the same tile row (the reader pushes
// them once per row run), so they are waited on at the first unit of a run and popped after its last.
//
// Compile-time args: Rt (tile rows per batch), H (heads)
// Runtime args: num_units, start_unit

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t num_units = get_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(1);

    constexpr uint32_t Rt = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);

    constexpr uint32_t x_cb = tt::CBIndex::c_0;
    constexpr uint32_t cs_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    constexpr uint32_t tmp_cb = tt::CBIndex::c_24;

    CircularBuffer cb_x(x_cb);
    CircularBuffer cb_cs(cs_cb);
    CircularBuffer cb_out(out_cb);
    CircularBuffer cb_tmp(tmp_cb);

    compute_kernel_hw_startup(x_cb, cs_cb, out_cb);

    const uint32_t end_unit = start_unit + num_units;
    for (uint32_t u = start_unit; u < end_unit; ++u) {
        const uint32_t r = u / H % Rt;
        if (u == start_unit || (u - 1) / H % Rt != r) {
            cb_cs.wait_front(4);
        }
        for (uint32_t part = 0; part < 2; ++part) {
            cb_x.wait_front(2);

            mul_init(x_cb, cs_cb);
            cb_tmp.reserve_back(2);
            tile_regs_acquire();
            mul_tiles(x_cb, cs_cb, 1, 2, 0);  // x1 * s0
            mul_tiles(x_cb, cs_cb, 0, 3, 1);  // x0 * s1
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, tmp_cb);
            pack_tile(1, tmp_cb);
            tile_regs_release();
            cb_tmp.push_back(2);
            cb_tmp.wait_front(2);

            cb_out.reserve_back(2);
            mul_init(x_cb, cs_cb);
            tile_regs_acquire();
            mul_tiles(x_cb, cs_cb, 0, 0, 0);  // x0 * c0
            mul_tiles(x_cb, cs_cb, 1, 1, 1);  // x1 * c1
            sub_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(tmp_cb);
            sub_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(tmp_cb, 0, 0);
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(tmp_cb);
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(tmp_cb, 1, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out_cb);
            pack_tile(1, out_cb);
            tile_regs_release();
            cb_out.push_back(2);

            cb_tmp.pop_front(2);
            cb_x.pop_front(2);
        }
        if (u + 1 == end_unit || u / H % Rt != (u + 1) / H % Rt) {
            cb_cs.pop_front(4);
        }
    }
}
