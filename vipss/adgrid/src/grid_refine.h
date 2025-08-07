//
//  mesh_refine.h
//  adaptive_mesh_refinement
//
//  Created by Yiwen Ju on 8/4/24.
//

#pragma once

#include <SmallVector.h>
#include "implicit_functions/implicit_functions.h"

#include "adaptive_grid_gen.h"
#include "io.h"
#include "refine_crit.h"
#include "tet_quality.h"

using namespace mtet;

/// First, hash four tet vertices into a `uint64_t`
/// Since the tetid isn't const during the process, mount the boolean using vertexids of 4 corners.
struct TetHash
{
    using is_avalanching = void;
    using is_transparent = void;
    auto operator()(std::span<VertexId, 4> const& x) const noexcept -> uint64_t {
        ankerl::unordered_dense::hash<uint64_t> hash_fn;
        return ankerl::unordered_dense::detail::wyhash::hash(hash_fn(value_of(x[0])) + hash_fn(value_of(x[1])) + hash_fn(value_of(x[2])) + hash_fn(value_of(x[3])));
    }
};

/// Determine if a tet's hash is equal to another by comparing their vertices.
/// Two tet's vertex ids should be identical as each tet has a unique map to its tet vertices according to `mTet`.
struct TetEqual
{
    using is_transparent = void;
    bool operator()(std::span<VertexId, 4> const& lhs, std::span<VertexId, 4> const& rhs)
        const noexcept
    {
        return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2] && lhs[3] == rhs[3];
    }
};

/// The main function for adaptively refine an initial grid based on a set of input implicit functions. The result forms an adaptive background grid for the given implicit complexes (check paper for details: https://dl.acm.org/doi/10.1145/3658215)
///
/// @param[in] mode         The modality of the implicit complex, including Implicit Arrangement(IA), Contructive Solid Geometry(CSG), Material Interface(MI).
/// @param[in] curve_network            Whether we're only computing the curve network of the complex.
/// @param[in] threshold            The user-defined error threshold. The smaller the value, the more refined the output grid.
/// @param[in] alpha            An experimental variable. This helps to improve the quality of the tet that contains a part of the iso-surface. In the paper, the default is set to be infinity. If it's set to be 1, it will generate tet with better quality (>= 0.64 in radius ratio, where radius ratio = 1 is the perfect tet) in our experiments. However, there is no theoretical gaurantee.
/// @param[in] max_elements         The maximum number of grid elements to cap the algorithm.
/// @param[in] funcNum          The number of functions that consist the complex.
/// @param[in] func         The lambda function that computes implicit functions' values and gradients given a 3D coordinate and the number of functions.
/// The input std::span<const Scalar, 3> is the 3D coordinate, size_t is the number of functions. The output is a vector of Eigen::RowVector4d. The vector size is the function number. Each eigen vector represents the value at 0th index and gradients at {1, 2, 3} index.
/// @param[in] csg_func         The lambda function that represents the CSG operations. Given a vector of function intervals, this function returns the final interval(std::array<double, 2>) after the CSG tree traversing and a vector of active functions' indices(llvm_vecsmall::SmallVector<int, 20>).
/// @param[out] grid            The final adaptive grid.
/// @param[out] metric_list         The tet metrics, see `io.h` for the detail.
/// @param[out] profileTimer            The timer's profile, see `timer.h` for detail.
///
///@return          Whether this function successfully proceeds.
bool gridRefine(
                const int mode,
                const bool curve_network,
                const double threshold,
                const double alpha,
                const int max_elements,
                const size_t funcNum,
                const std::function<llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20>(std::span<const Scalar, 3>, size_t)> func,
                const std::function<std::pair<std::array<double, 2>, llvm_vecsmall::SmallVector<int, 20>>(llvm_vecsmall::SmallVector<std::array<double, 2>, 20>)> csg_func,
                mtet::MTetMesh &grid,
                tet_metric &metric_list,
                std::array<double, timer_amount> profileTimer
                );
