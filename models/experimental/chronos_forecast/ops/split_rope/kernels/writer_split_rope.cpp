// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for the fused split-heads + RoPE op: rotated q and k tiles from cb_out, v tiles from cb_v, into
// [B, H, S_pad, 64] head-major outputs (two tiles per tile row).
//
// Compile-time args: Rt, H, TensorAccessorArgs(q) (k and v share the layout)
// Runtime args: q_addr, k_addr, v_addr, num_units, start_unit

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_units = get_arg_val<uint32_t>(3);
    const uint32_t start_unit = get_arg_val<uint32_t>(4);

    constexpr uint32_t Rt = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr auto o_args = TensorAccessorArgs<2>();

    constexpr uint32_t v_cb_id = tt::CBIndex::c_2;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_16;

    Noc noc;
    CircularBuffer cb_v(v_cb_id);
    CircularBuffer cb_out(out_cb_id);
    const uint32_t tile_bytes = get_tile_size(out_cb_id);
    const auto q = TensorAccessor(o_args, q_addr);
    const auto k = TensorAccessor(o_args, k_addr);
    const auto v = TensorAccessor(o_args, v_addr);

    for (uint32_t u = start_unit; u < start_unit + num_units; ++u) {
        const uint32_t h = u % H;
        const uint32_t br = u / H;
        const uint32_t b = br / Rt;
        const uint32_t r = br % Rt;
        const uint32_t page = ((b * H + h) * Rt + r) * 2;

        cb_out.wait_front(2);
        for (uint32_t j = 0; j < 2; ++j) {
            noc.async_write(cb_out, q, tile_bytes, {.offset_bytes = j * tile_bytes}, {.page_id = page + j});
        }
        noc.async_writes_flushed();
        cb_out.pop_front(2);

        cb_out.wait_front(2);
        for (uint32_t j = 0; j < 2; ++j) {
            noc.async_write(cb_out, k, tile_bytes, {.offset_bytes = j * tile_bytes}, {.page_id = page + j});
        }
        noc.async_writes_flushed();
        cb_out.pop_front(2);

        cb_v.wait_front(2);
        for (uint32_t j = 0; j < 2; ++j) {
            noc.async_write(cb_v, v, tile_bytes, {.offset_bytes = j * tile_bytes}, {.page_id = page + j});
        }
        noc.async_writes_flushed();
        cb_v.pop_front(2);
    }
    noc.async_write_barrier();
}
