// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation_types.hpp"

namespace ttnn::prim {

struct SparseMatmulMultiCoreReuseMcast1DProgramFactory {
    // Descriptor-based factory: no shared_variables_t, no cached_program_t, and no
    // override_runtime_arguments. The only values that vary between two dispatches sharing a program
    // hash are the in0/in1/sparsity(-or-indices)/output buffer addresses, and those are declared as
    // Buffer* bindings via KernelDescriptor::emplace_runtime_args in create_descriptor. The framework
    // therefore patches them in place on a cache hit; everything else is derived from the hashed
    // tensor specs and operation attributes, so a hit guarantees it is already correct.
    //
    // Do NOT add a cached_program_t/cached_mesh_workload_t typedef here: ProgramDescriptorFactoryConcept
    // excludes by typedef presence, so a stray alias would silently demote this off the descriptor path.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ttnn::prim::SparseMatmulParams& operation_attributes,
        const ttnn::prim::SparseMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::prim
