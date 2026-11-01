// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for the fused split-heads + RoPE op. A work unit is (batch, tile row, head), head fastest. The
// cos/sin tiles of a tile row (c0, c1, s0, s1) are shared by every head of that row, so they are read once
// per run of consecutive units with the same tile row (the compute kernel keeps them until the row changes),
// instead of once per unit. Per unit: the two q tiles and the two k tiles of the head into cb_x (one pair at
// a time) and the two v tiles into cb_v.
//
// Compile-time args: Rt (tile rows per batch), H (heads), Wt (qkv width in tiles), XLEN,
//                    TensorAccessorArgs(qkv), TensorAccessorArgs(cos/sin)
// Runtime args: qkv_addr, cos_addr, sin_addr, num_units, start_unit

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t qkv_addr = get_arg_val<uint32_t>(0);
    const uint32_t cos_addr = get_arg_val<uint32_t>(1);
    const uint32_t sin_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_units = get_arg_val<uint32_t>(3);
    const uint32_t start_unit = get_arg_val<uint32_t>(4);

    constexpr uint32_t Rt = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t XLEN = get_compile_time_arg_val(3);
    constexpr auto x_args = TensorAccessorArgs<4>();
    constexpr auto t_args = TensorAccessorArgs<4 + XLEN>();

    constexpr uint32_t x_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t cs_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t v_cb_id = tt::CBIndex::c_2;

    Noc noc;
    CircularBuffer cb_x(x_cb_id);
    CircularBuffer cb_cs(cs_cb_id);
    CircularBuffer cb_v(v_cb_id);
    const uint32_t tile_bytes = get_tile_size(x_cb_id);
    const auto x = TensorAccessor(x_args, qkv_addr);
    const auto c = TensorAccessor(t_args, cos_addr);
    const auto s = TensorAccessor(t_args, sin_addr);

    for (uint32_t u = start_unit; u < start_unit + num_units; ++u) {
        const uint32_t h = u % H;
        const uint32_t br = u / H;
        const uint32_t r = br % Rt;
        const uint32_t base = br * Wt + 2 * h;

        // new tile row on this core (first unit, or the previous unit had another tile row)
        if (u == start_unit || (u - 1) / H % Rt != r) {
            cb_cs.reserve_back(4);
            noc.async_read(c, cb_cs, tile_bytes, {.page_id = 2 * r}, {.offset_bytes = 0});
            noc.async_read(c, cb_cs, tile_bytes, {.page_id = 2 * r + 1}, {.offset_bytes = tile_bytes});
            noc.async_read(s, cb_cs, tile_bytes, {.page_id = 2 * r}, {.offset_bytes = 2 * tile_bytes});
            noc.async_read(s, cb_cs, tile_bytes, {.page_id = 2 * r + 1}, {.offset_bytes = 3 * tile_bytes});
            noc.async_read_barrier();
            cb_cs.push_back(4);
        }

        for (uint32_t part = 0; part < 2; ++part) {
            cb_x.reserve_back(2);
            for (uint32_t j = 0; j < 2; ++j) {
                noc.async_read(
                    x, cb_x, tile_bytes, {.page_id = base + part * 2 * H + j}, {.offset_bytes = j * tile_bytes});
            }
            noc.async_read_barrier();
            cb_x.push_back(2);
        }

        cb_v.reserve_back(2);
        for (uint32_t j = 0; j < 2; ++j) {
            noc.async_read(x, cb_v, tile_bytes, {.page_id = base + 4 * H + j}, {.offset_bytes = j * tile_bytes});
        }
        noc.async_read_barrier();
        cb_v.push_back(2);
    }
}
