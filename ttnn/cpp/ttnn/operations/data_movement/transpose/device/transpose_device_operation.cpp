// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_device_operation.hpp"
#include "transpose_utils.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/per_core_allocation/memory_config.hpp>
#include <tt-metalium/experimental/range_lockstep_allocation/memory_config.hpp>
#include <tt-metalium/hal.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
using ttnn::operations::data_movement::transpose::adjust_shard_spec_to_shape;
using ttnn::operations::data_movement::transpose::generate_transpose_shard_spec;
using ttnn::operations::data_movement::transpose::is_native_transpose_sharding;

namespace ttnn::prim {

namespace {

// Output logical+padded shapes per transpose dim. HC TILE: dim[1] = logical H (slot 1 isn't
// tile-padded), dim[2] = round_up(logical C, TILE_HEIGHT). Shared by
// derive_effective_output_memory_config and compute_output_specs.
struct TransposedShapes {
    ttnn::Shape logical;
    ttnn::Shape padded;
};

TransposedShapes transposed_shapes(const Tensor& input_tensor, TransposeOpDim dim) {
    auto output_shape = input_tensor.logical_shape();
    auto output_padded_shape = input_tensor.padded_shape();
    switch (dim) {
        case TransposeOpDim::CN:
            std::swap(output_shape[0], output_shape[1]);
            std::swap(output_padded_shape[0], output_padded_shape[1]);
            break;
        case TransposeOpDim::HC:
            if (input_tensor.layout() == Layout::ROW_MAJOR) {
                std::swap(output_shape[1], output_shape[2]);
                std::swap(output_padded_shape[1], output_padded_shape[2]);
            } else {
                const uint32_t C = output_shape[1];
                const uint32_t C_p = tt::round_up(C, input_tensor.tensor_spec().tile().get_height());
                const uint32_t H = output_shape[2];
                output_shape[1] = H;
                output_shape[2] = C;
                output_padded_shape[1] = H;
                output_padded_shape[2] = C_p;
            }
            break;
        case TransposeOpDim::WH:
            std::swap(output_shape[2], output_shape[3]);
            std::swap(output_padded_shape[2], output_padded_shape[3]);
            break;
        default: TT_THROW("Unsupported transpose dim"); break;
    }
    return {output_shape, output_padded_shape};
}

// Reindex an ND shard shape for the given transpose dim, mirroring the same index swap
// transposed_shapes() applies to the padded shape. HC on TILE is excluded: its padded-shape
// contract is asymmetric (dim[1] = logical H, dim[2] = round_up(logical C, TILE_HEIGHT)) rather
// than a plain swap, so a raw shard-extent swap wouldn't be sound there; callers must fall back to
// fresh synthesis (see nd_shard_spec_from_legacy) for that case.
//
// NdShardSpec::shard_shape is trailing-aligned to the *tensor's* rank, not a standalone shape —
// see squeeze_shape_ranks(): shard_shape[-1] is always the tensor's last dim, shard_shape[-N] its
// N-th-from-last dim, for shard_shape of rank N. Any leading tensor dims not covered by
// shard_shape simply aren't sharded (full tensor extent along that axis). tensor_rank is needed to
// translate the transpose's absolute tensor-axis pair into shard_shape indices correctly.
std::optional<NdShardSpec> adjust_nd_shard_spec_for_transpose(
    const NdShardSpec& nd_shard_spec, TransposeOpDim dim, Layout layout, int tensor_rank) {
    auto shard_shape = nd_shard_spec.shard_shape;
    const int shard_rank = static_cast<int>(shard_shape.rank());
    const int offset = tensor_rank - shard_rank;  // shard_shape[i] <=> tensor axis (i + offset)

    // Map an absolute tensor axis to its shard_shape index, or nullopt if that axis isn't
    // represented in shard_shape at all (i.e. not sharded along that axis).
    auto to_shard_index = [&](int tensor_axis) -> std::optional<int> {
        int idx = tensor_axis - offset;
        if (idx < 0 || idx >= shard_rank) {
            return std::nullopt;
        }
        return idx;
    };

    std::optional<int> a;
    std::optional<int> b;
    switch (dim) {
        case TransposeOpDim::CN:
            if (tensor_rank < 2) {
                return std::nullopt;
            }
            a = to_shard_index(0);
            b = to_shard_index(1);
            break;
        case TransposeOpDim::WH:
            if (tensor_rank < 2) {
                return std::nullopt;
            }
            a = to_shard_index(tensor_rank - 2);
            b = to_shard_index(tensor_rank - 1);
            break;
        case TransposeOpDim::HC:
            if (layout != Layout::ROW_MAJOR || tensor_rank < 3) {
                return std::nullopt;
            }
            a = to_shard_index(1);
            b = to_shard_index(2);
            break;
        default: return std::nullopt;
    }
    if (a.has_value() != b.has_value()) {
        // Exactly one of the two swapped axes is represented in shard_shape: expressing the
        // result would require renormalizing shard_shape to the full tensor rank first (padding
        // the missing axis back in). Bail out so the caller falls back to fresh synthesis instead
        // of guessing.
        return std::nullopt;
    }
    if (a.has_value()) {
        std::swap(shard_shape[*a], shard_shape[*b]);
    }
    // Else: neither swapped axis is represented in shard_shape, so swapping their roles doesn't
    // change shard_shape at all.
    NdShardSpec adjusted = nd_shard_spec;
    adjusted.shard_shape = std::move(shard_shape);
    return adjusted;
}

// Wrap a freshly-synthesized legacy 2D ShardSpec into an equivalent NdShardSpec, matching the same
// 2-element flatten TensorSpec::populate_nd_shard_spec_from_legacy() uses internally. Used to
// re-derive ND-sharding provenance from fresh geometry rather than reusing a stale nd_shard_spec
// that may not describe the transposed shape.
//
// original_strategy carries the input's shard_distribution_strategy through: inferring it solely
// from memory_layout defaults everything but BLOCK_SHARDED to ROUND_ROBIN_1D, silently changing the
// shard-to-bank mapping for e.g. CONTIGUOUS_1D or GRID_2D ND-only inputs that have no legacy
// equivalent to infer from.
NdShardSpec nd_shard_spec_from_legacy(
    const ShardSpec& shard_spec, TensorMemoryLayout memory_layout, ShardDistributionStrategy original_strategy) {
    NdShardSpec nd_shard_spec{
        .shard_shape = ttnn::Shape({shard_spec.shape[0], shard_spec.shape[1]}),
        .grid = shard_spec.grid,
        .orientation = shard_spec.orientation,
        .shard_distribution_strategy = original_strategy,
    };
    if (original_strategy == ShardDistributionStrategy::ROUND_ROBIN_1D &&
        memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        nd_shard_spec.shard_distribution_strategy = ShardDistributionStrategy::GRID_2D;
    }
    return nd_shard_spec;
}

// MemoryConfig's public constructors always reset experimental per-core-allocation /
// range-lockstep-allocation state to defaults, so any hand-reconstruction of a MemoryConfig (to
// swap buffer_type/shard geometry while otherwise preserving it) must re-apply them explicitly.
void copy_experimental_allocation_flags(const MemoryConfig& source, MemoryConfig& dest) {
    experimental::per_core_allocation::set_per_core_allocation(
        dest, experimental::per_core_allocation::is_per_core_allocation(source));
    experimental::range_lockstep_allocation::set_range_lockstep_allocation(
        dest, experimental::range_lockstep_allocation::is_range_lockstep_allocation(source));
}

// Synthesize a shard_spec when the user asks for sharded output without one, so downstream
// (select_program_factory, compute_output_specs) sees a fully-specified config. Falls back to a
// fresh full-grid spec when input shard can't be scaled exactly.
MemoryConfig derive_effective_output_memory_config(
    const TransposeDeviceOperation::operation_attributes_t& operation_attributes,
    const TransposeDeviceOperation::tensor_args_t& tensor_args) {
    auto output_mem_config = operation_attributes.output_mem_config;
    if (!output_mem_config.is_sharded() || output_mem_config.shard_spec().has_value()) {
        return output_mem_config;
    }
    const auto& input_tensor = tensor_args.input;
    const auto output_padded_shape = transposed_shapes(input_tensor, operation_attributes.dim).padded;

    // ND provenance: MemoryConfig only exposes a public constructor for ND-only configs
    // (buffer_type, nd_shard_spec) — there's no public way to attach a legacy shard_spec
    // alongside it, so TensorSpec's own population logic derives the legacy shard_spec (if any)
    // downstream. Prefer reindexing the existing ND shard geometry for the specific transpose dim
    // (keeps the same shard grid/placement, just reordered) over the legacy 2D scaling below,
    // which doesn't model ND shard shapes at all.
    //
    // output_mem_config_is_explicit (set by transpose.cpp) distinguishes a caller-supplied output
    // config — already expressed in the output coordinate frame, must be honored verbatim — from
    // one defaulted/mirrored from the input, which still needs reindexing to the output frame.
    // Both input and a caller-supplied output can independently carry ND provenance, so the flag
    // (not the input's provenance) is what actually determines whether reindexing is correct here.
    const bool output_has_nd_spec =
        output_mem_config.created_with_nd_shard_spec() && output_mem_config.nd_shard_spec().has_value();
    if (output_has_nd_spec && operation_attributes.output_mem_config_is_explicit) {
        return output_mem_config;
    }
    const bool preserve_nd_provenance = output_has_nd_spec;
    if (preserve_nd_provenance) {
        if (auto adjusted_nd = adjust_nd_shard_spec_for_transpose(
                *output_mem_config.nd_shard_spec(),
                operation_attributes.dim,
                input_tensor.layout(),
                static_cast<int>(input_tensor.padded_shape().rank()))) {
            auto adjusted_config = MemoryConfig(output_mem_config.buffer_type(), std::move(adjusted_nd));
            copy_experimental_allocation_flags(output_mem_config, adjusted_config);
            return adjusted_config;
        }
        // Fall through (e.g. TILE HC): can't reindex losslessly, so synthesize fresh legacy shard
        // geometry below and re-wrap it as an NdShardSpec instead of dropping provenance outright.
    }
    // output_mem_config.memory_layout() always reports ND_SHARDED for an ND-only config, losing
    // the actual HEIGHT/WIDTH/BLOCK distribution style the input was created with — which
    // generate_transpose_shard_spec()/nd_shard_spec_from_legacy() need to synthesize a matching
    // geometry instead of silently defaulting to BLOCK_SHARDED. The input's own memory_config()
    // already carries that legacy-equivalent layout (derived at tensor-construction time by
    // TensorSpec::populate_sharding_specs()) when the ND shape was flattenable, so prefer it here.
    const TensorMemoryLayout synthesis_memory_layout =
        preserve_nd_provenance ? input_tensor.memory_config().memory_layout() : output_mem_config.memory_layout();

    // adjust_shard_spec_to_shape preserves the sharding style — only reuse input geometry when
    // requested output layout matches input's; otherwise generate_transpose_shard_spec builds
    // fresh. Skipped when preserving ND provenance: this branch requires a legacy shard_spec on
    // the (ND-only) output_mem_config, which it doesn't have.
    if (!preserve_nd_provenance && input_tensor.is_sharded() && input_tensor.shard_spec().has_value() &&
        input_tensor.memory_config().memory_layout() == output_mem_config.memory_layout()) {
        auto adjusted = adjust_shard_spec_to_shape(
            input_tensor.shard_spec().value(), input_tensor.padded_shape(), output_padded_shape);
        if (adjusted.has_value()) {
            // TILE sharded factories need tile-aligned shards; adjust_shard_spec_to_shape may now
            // produce sub-tile when transpose legitimately shrinks a dim → fall through to
            // generate_transpose_shard_spec rather than feeding an unusable spec.
            const bool tile_layout = input_tensor.layout() == Layout::TILE;
            const bool tile_aligned = adjusted->shape[0] % tt::constants::TILE_HEIGHT == 0 &&
                                      adjusted->shape[1] % tt::constants::TILE_WIDTH == 0;
            if (!tile_layout || tile_aligned) {
                auto adjusted_config = MemoryConfig(
                    output_mem_config.memory_layout(), output_mem_config.buffer_type(), std::move(adjusted));
                copy_experimental_allocation_flags(output_mem_config, adjusted_config);
                return adjusted_config;
            }
        }
    }
    auto shard_spec = generate_transpose_shard_spec(input_tensor, output_padded_shape, synthesis_memory_layout);
    if (preserve_nd_provenance) {
        const auto original_strategy = output_mem_config.nd_shard_spec()->shard_distribution_strategy;
        auto rewrapped_config = MemoryConfig(
            output_mem_config.buffer_type(),
            nd_shard_spec_from_legacy(shard_spec, synthesis_memory_layout, original_strategy));
        copy_experimental_allocation_flags(output_mem_config, rewrapped_config);
        return rewrapped_config;
    }
    auto synthesized_config =
        MemoryConfig(output_mem_config.memory_layout(), output_mem_config.buffer_type(), shard_spec);
    copy_experimental_allocation_flags(output_mem_config, synthesized_config);
    return synthesized_config;
}

}  // namespace

TransposeDeviceOperation::program_factory_t TransposeDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto output_memory_config = derive_effective_output_memory_config(operation_attributes, tensor_args);
    const auto& dim = operation_attributes.dim;
    bool is_row_major = input_tensor.layout() == Layout::ROW_MAJOR;

    // !native → fall through to interleaved factories (TensorAccessorArgs handles sharded buffers
    // transparently via NOC).
    bool native = is_native_transpose_sharding(input_tensor.tensor_spec(), output_memory_config);

    // shard_spec.shape is in padded terms; comparisons below use padded dims.
    const auto& input_padded_shape = input_tensor.padded_shape();
    const auto output_padded_shape = transposed_shapes(input_tensor, dim).padded;
    uint32_t N = input_padded_shape[0], C = input_padded_shape[1];
    uint32_t output_width = output_padded_shape[-1];
    uint32_t output_height = output_padded_shape[-2];

    bool input_height_sharded = native && input_tensor.is_sharded() && input_tensor.shard_spec().has_value() &&
                                input_tensor.shard_spec()->shape[1] == input_padded_shape[-1];
    bool input_width_and_height_fully_in_shard =
        input_height_sharded && input_tensor.shard_spec()->shape[0] % input_padded_shape[-2] == 0;
    bool output_height_sharded = native && output_memory_config.is_sharded() &&
                                 output_memory_config.shard_spec().has_value() &&
                                 output_memory_config.shard_spec()->shape[1] == output_width;
    bool output_width_sharded = native && output_memory_config.is_sharded() &&
                                output_memory_config.shard_spec().has_value() &&
                                output_memory_config.shard_spec()->shape[0] == output_height;
    bool output_width_and_height_fully_in_shard =
        output_height_sharded && output_memory_config.shard_spec()->shape[0] % output_height == 0;
    // Second disjunct is TILE-only; the RM branch below dispatches on use_sharded_wh_rm, not use_sharded_wh.
    bool use_sharded_wh =
        native && ((input_width_and_height_fully_in_shard && output_width_and_height_fully_in_shard) ||
                   (N == 1 && C == 1 && input_height_sharded && output_width_sharded));
    // RM factory hardcodes num_hw_blocks_per_core = shard_height/H, so RM demands the fully-in-shard subset.
    bool use_sharded_wh_rm = native && input_width_and_height_fully_in_shard && output_width_and_height_fully_in_shard;
    bool use_sharded_hc = native && input_height_sharded && output_height_sharded && is_row_major;

    auto parallelization_strategy = get_parallelization_strategy(operation_attributes, tensor_args);

    switch (parallelization_strategy) {
        case TransposeOpParallelizationStrategy::MULTI_CORE_WH:
            if (is_row_major) {
                if (use_sharded_wh_rm) {
                    return TransposeWHShardedRMProgramFactory{};
                }
            } else if (use_sharded_wh) {
                return TransposeWHShardedProgramFactory{};
            }
            return TransposeWHProgramFactory{};

        case TransposeOpParallelizationStrategy::MULTI_CORE_HC:
            if (use_sharded_hc) {
                return TransposeHCShardedProgramFactory{};
            }
            if (is_row_major) {
                return TransposeHCRMProgramFactory{};
            }
            return TransposeHCTiledInterleavedProgramFactory{};

        case TransposeOpParallelizationStrategy::MULTI_CORE_CN: return TransposeCNProgramFactory{};

        default: TT_THROW("Unsupported parallelization strategy");
    }
}

TransposeOpParallelizationStrategy TransposeDeviceOperation::get_parallelization_strategy(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    switch (operation_attributes.dim) {
        case TransposeOpDim::WH: return TransposeOpParallelizationStrategy::MULTI_CORE_WH;
        case TransposeOpDim::HC: return TransposeOpParallelizationStrategy::MULTI_CORE_HC;
        case TransposeOpDim::CN: return TransposeOpParallelizationStrategy::MULTI_CORE_CN;
        default: TT_THROW("Unsupported transpose dim for parallelization strategy");
    }
}

void TransposeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& dim = operation_attributes.dim;
    const float pad_value = operation_attributes.pad_value;

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to transpose need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to transpose need to be allocated in buffers on device!");
    TT_FATAL(
        !(dim != TransposeOpDim::HC && pad_value != 0.0f),
        "Non-zero padding {} is not supported for any transpose other than HC.",
        pad_value);
    TT_FATAL(
        dim == TransposeOpDim::HC || dim == TransposeOpDim::WH || dim == TransposeOpDim::CN,
        "Transpose HC, WH, CN are the only supported transpose operations. Transpose {} is not supported.",
        static_cast<int>(dim));

    const auto& shape = input_tensor.padded_shape();
    bool row_major = input_tensor.layout() == Layout::ROW_MAJOR;
    uint32_t W = shape[3], H = shape[2];

    if (!row_major) {
        TT_FATAL(
            W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0,
            "Tiled tensor H {} W {} must be a multiple of TILE HEIGHT {} and TILE WIDTH",
            H,
            W,
            TILE_HEIGHT,
            TILE_WIDTH);
        TT_FATAL(
            input_tensor.physical_volume() % TILE_HW == 0,
            "Tiled tensor volume {} must be a multiple of TILE HEIGHT * TILE WIDTH",
            input_tensor.physical_volume(),
            TILE_HW);
    }
}

tt::tt_metal::TensorSpec TransposeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto output_mem_config = derive_effective_output_memory_config(operation_attributes, tensor_args);
    const auto [output_shape, output_padded_shape] = transposed_shapes(input_tensor, operation_attributes.dim);

    return tt::tt_metal::TensorSpec(
        output_shape,
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            PageConfig(input_tensor.layout()),
            output_mem_config,
            output_shape,
            output_padded_shape));
}

Tensor TransposeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> TransposeDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args, const Tensor& output) {
    const auto& input_tensor = tensor_args.input;
    int ideal_dev_clock_cycles = ttnn::operations::data_movement::common_tm_bw_model(input_tensor, output);
    tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> result({input_tensor}, {output}, ideal_dev_clock_cycles);
    return result;
}

}  // namespace ttnn::prim

namespace ttnn::prim {
ttnn::Tensor transpose(
    const Tensor& input_tensor,
    ttnn::prim::TransposeOpDim dim,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    float pad_value,
    bool output_mem_config_is_explicit) {
    using OperationType = ttnn::prim::TransposeDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .dim = dim,
            .output_mem_config = output_mem_config,
            .pad_value = pad_value,
            .output_mem_config_is_explicit = output_mem_config_is_explicit,
        },
        TransposeInputs{
            .input = input_tensor,
        });
}
}  // namespace ttnn::prim
