#pragma once
#include <vector>

namespace marching3D {
/**
 * @brief Given a 4D simplicial mesh and a scalar at each vertex,
 * compute the zero-hypersurface as a list of 4D zero-crossing points and a list of 3D tets.
 * Note that the resulting tets may not have consistent orientation.
 * @param vertices vertices of the 4D simplicial mesh
 * @param simplices simplices of the 4D simplicial mesh
 * @param values scalar at each vertex
 * @param output_vertices output vertices of the zero-hypersurface
 * @param output_tets output tets of the zero-hypersurface
 */
template <typename Float>
void Marching4Simplex(const std::vector<std::array<Float, 4> >& vertices,
                      const std::vector<std::array<size_t, 5> >& simplices, const std::vector<Float>& values,
                      std::vector<std::array<Float, 4> >& output_vertices,
                      std::vector<std::array<size_t, 4> >& output_tets);

// extern template void Marching4Simplex<double>(const std::vector<std::array<double, 4> >& vertices,
//                                               const std::vector<std::array<size_t, 5> >& simplices,
//                                               const std::vector<double>& values,
//                                               std::vector<std::array<double, 4> >& output_vertices,
//                                               std::vector<std::array<size_t, 4> >& output_tets);

// extern template void Marching4Simplex<float>(const std::vector<std::array<float, 4> >& vertices,
//                                              const std::vector<std::array<size_t, 5> >& simplices,
//                                              const std::vector<float>& values,
//                                              std::vector<std::array<float, 4> >& output_vertices,
//                                              std::vector<std::array<size_t, 4> >& output_tets);

/**
 * @brief Given a 4D simplicial mesh and a scalar at each vertex,
 * compute the zero-hypersurface as a list of 4D zero-crossing points, a list of 3D tets, and a list of 3D prisms.
 * Note that the resulting tets/prisms may not have consistent orientation.
 * @param vertices vertices of the 4D simplicial mesh
 * @param simplices simplices of the 4D simplicial mesh
 * @param values scalar at each vertex
 * @param output_vertices output vertices of the zero-hypersurface
 * @param output_tets output tets of the zero-hypersurface
 * @param output_prisms output prisms of the zero-hypersurface
 */
template <typename Float>
void Marching4Simplex(const std::vector<std::array<Float, 4> >& vertices,
                      const std::vector<std::array<size_t, 5> >& simplices, const std::vector<Float>& values,
                      std::vector<std::array<Float, 4> >& output_vertices,
                      std::vector<std::array<size_t, 4> >& output_tets,
                      std::vector<std::array<size_t, 6> >& output_prisms);

// extern template void Marching4Simplex<double>(const std::vector<std::array<double, 4> >& vertices,
//                                               const std::vector<std::array<size_t, 5> >& simplices,
//                                               const std::vector<double>& values,
//                                               std::vector<std::array<double, 4> >& output_vertices,
//                                               std::vector<std::array<size_t, 4> >& output_tets,
//                                               std::vector<std::array<size_t, 6> >& output_prisms);

// extern template void Marching4Simplex<float>(const std::vector<std::array<float, 4> >& vertices,
//                                              const std::vector<std::array<size_t, 5> >& simplices,
//                                              const std::vector<float>& values,
//                                              std::vector<std::array<float, 4> >& output_vertices,
//                                              std::vector<std::array<size_t, 4> >& output_tets,
//                                              std::vector<std::array<size_t, 6> >& output_prisms);

/**
 * @brief [For sweeping along an open trajectory] Given a 4D simplicial mesh, a scalar at each vertex,
 * and two lists of 3D tets (upper and lower boundary of the 4D simplicial mesh),
 * compute the zero-hypersurface as well as the upper level sets in upper_tets and lower level sets in lower_tets.
 * Note that the resulting tets may not have consistent orientation.
 * @param vertices vertices of the 4D simplicial mesh
 * @param simplices simplices of the 4D simplicial mesh
 * @param upper_tets upper boundary of the 4D simplicial mesh
 * @param lower_tets lower boundary of the 4D simplicial mesh
 * @param values scalar at each vertex
 * @param output_vertices output vertices of the zero-hypersurface and upper/lower level sets
 * @param output_tets output tets of the zero-hypersurface and upper/lower level sets
 */
template <typename Float>
void Marching4SimplexUL(const std::vector<std::array<Float, 4> >& vertices,
                        const std::vector<std::array<size_t, 5> >& simplices,
                        const std::vector<std::array<size_t, 4> >& upper_tets,
                        const std::vector<std::array<size_t, 4> >& lower_tets,
                        const std::vector<Float>& values,
                        std::vector<std::array<Float, 4> >& output_vertices,
                        std::vector<std::array<size_t, 4> >& output_tets);

// extern template void Marching4SimplexUL<double>(const std::vector<std::array<double, 4> >& vertices,
//                                                 const std::vector<std::array<size_t, 5> >& simplices,
//                                                 const std::vector<std::array<size_t, 4> >& upper_tets,
//                                                 const std::vector<std::array<size_t, 4> >& lower_tets,
//                                                 const std::vector<double>& values,
//                                                 std::vector<std::array<double, 4> >& output_vertices,
//                                                 std::vector<std::array<size_t, 4> >& output_tets);

// extern template void Marching4SimplexUL<float>(const std::vector<std::array<float, 4> >& vertices,
//                                                const std::vector<std::array<size_t, 5> >& simplices,
//                                                const std::vector<std::array<size_t, 4> >& upper_tets,
//                                                const std::vector<std::array<size_t, 4> >& lower_tets,
//                                                const std::vector<float>& values,
//                                                std::vector<std::array<float, 4> >& output_vertices,
//                                                std::vector<std::array<size_t, 4> >& output_tets);

/**
 * @brief [For sweeping along an open trajectory] Given a 4D simplicial mesh, a scalar at each vertex,
 * and two lists of 3D tets (upper and lower boundary of the 4D simplicial mesh),
 * compute the zero-hypersurface as well as the upper level sets in upper_tets and lower level sets in lower_tets.
 * Note that the resulting tets/prisms may not have consistent orientation.
 * @param vertices vertices of the 4D simplicial mesh
 * @param simplices simplices of the 4D simplicial mesh
 * @param upper_tets upper boundary of the 4D simplicial mesh
 * @param lower_tets lower boundary of the 4D simplicial mesh
 * @param values scalar at each vertex
 * @param output_vertices output vertices of the zero-hypersurface and upper/lower level sets
 * @param output_tets output tets of the zero-hypersurface and upper/lower level sets
 * @param output_prisms output prisms of the zero-hypersurface and upper/lower level sets
 */
template <typename Float>
void Marching4SimplexUL(const std::vector<std::array<Float, 4> >& vertices,
                        const std::vector<std::array<size_t, 5> >& simplices,
                        const std::vector<std::array<size_t, 4> >& upper_tets,
                        const std::vector<std::array<size_t, 4> >& lower_tets,
                        const std::vector<Float>& values,
                        std::vector<std::array<Float, 4> >& output_vertices,
                        std::vector<std::array<size_t, 4> >& output_tets,
                        std::vector<std::array<size_t, 6> >& output_prisms);

// extern template void Marching4SimplexUL<double>(const std::vector<std::array<double, 4> >& vertices,
//                                                 const std::vector<std::array<size_t, 5> >& simplices,
//                                                 const std::vector<std::array<size_t, 4> >& upper_tets,
//                                                 const std::vector<std::array<size_t, 4> >& lower_tets,
//                                                 const std::vector<double>& values,
//                                                 std::vector<std::array<double, 4> >& output_vertices,
//                                                 std::vector<std::array<size_t, 4> >& output_tets,
//                                                 std::vector<std::array<size_t, 6> >& output_prisms);

// extern template void Marching4SimplexUL<float>(const std::vector<std::array<float, 4> >& vertices,
//                                                const std::vector<std::array<size_t, 5> >& simplices,
//                                                const std::vector<std::array<size_t, 4> >& upper_tets,
//                                                const std::vector<std::array<size_t, 4> >& lower_tets,
//                                                const std::vector<float>& values,
//                                                std::vector<std::array<float, 4> >& output_vertices,
//                                                std::vector<std::array<size_t, 4> >& output_tets,
//                                                std::vector<std::array<size_t, 6> >& output_prisms);

/**
 * @brief [For sweeping along a closed trajectory] Given a 4D simplicial mesh, a scalar value at each vertex,
 * and a list of vertex index pairs indicating equivalent vertices
 * (e.g., a vertex on the bottom cap is paired with the spatially identical vertex on the top cap),
 * the function computes the zero-level hypersurface.
 * It ensures that any zero-crossing vertex lying on an edge between two vertices, {v1, v2},
 * has the same index as a vertex lying on another edge, {u1, u2},
 * if the pairs {v1, u1} and {v2, u2} are specified in the input list.
 * Note that the resulting tets  may not have consistent orientations.
 * @param vertices vertices of the 4D simplicial mesh
 * @param simplices simplices of the 4D simplicial mesh
 * @param vertex_pairs pairs of equivalent vertex indices
 * @param values scalar at each vertex
 * @param output_vertices output vertices of the zero-hypersurface
 * @param output_tets output tets of the zero-hypersurface
 */
template <typename Float>
void Marching4SimplexCyclic(const std::vector<std::array<Float, 4> >& vertices,
                            const std::vector<std::array<size_t, 5> >& simplices,
                            const std::vector<std::array<size_t, 2> >& vertex_pairs,
                            const std::vector<Float>& values,
                            std::vector<std::array<Float, 4> >& output_vertices,
                            std::vector<std::array<size_t, 4> >& output_tets);

// extern template void Marching4SimplexCyclic<double>(const std::vector<std::array<double, 4> >& vertices,
//                                                     const std::vector<std::array<size_t, 5> >& simplices,
//                                                     const std::vector<std::array<size_t, 2> >& vertex_pairs,
//                                                     const std::vector<double>& values,
//                                                     std::vector<std::array<double, 4> >& output_vertices,
//                                                     std::vector<std::array<size_t, 4> >& output_tets);

// extern template void Marching4SimplexCyclic<float>(const std::vector<std::array<float, 4> >& vertices,
//                                                    const std::vector<std::array<size_t, 5> >& simplices,
//                                                    const std::vector<std::array<size_t, 2> >& vertex_pairs,
//                                                    const std::vector<float>& values,
//                                                    std::vector<std::array<float, 4> >& output_vertices,
//                                                    std::vector<std::array<size_t, 4> >& output_tets);

/**
 * @brief [For sweeping along a closed trajectory] Given a 4D simplicial mesh, a scalar value at each vertex,
 * and a list of vertex index pairs indicating equivalent vertices
 * (e.g., a vertex on the bottom cap is paired with the spatially identical vertex on the top cap),
 * the function computes the zero-level hypersurface.
 * It ensures that any zero-crossing vertex lying on an edge between two vertices, {v1, v2},
 * has the same index as a vertex lying on another edge, {u1, u2},
 * if the pairs {v1, u1} and {v2, u2} are specified in the input list.
 * Note that the resulting tets/prisms  may not have consistent orientations.
 * @param vertices vertices of the 4D simplicial mesh
 * @param simplices simplices of the 4D simplicial mesh
 * @param vertex_pairs pairs of equivalent vertex indices
 * @param values scalar at each vertex
 * @param output_vertices output vertices of the zero-hypersurface
 * @param output_tets output tets of the zero-hypersurface
 * @param output_prisms output prisms of the zero-hypersurface
 */
template <typename Float>
void Marching4SimplexCyclic(const std::vector<std::array<Float, 4> >& vertices,
                            const std::vector<std::array<size_t, 5> >& simplices,
                            const std::vector<std::array<size_t, 2> >& vertex_pairs,
                            const std::vector<Float>& values,
                            std::vector<std::array<Float, 4> >& output_vertices,
                            std::vector<std::array<size_t, 4> >& output_tets,
                            std::vector<std::array<size_t, 6> >& output_prisms);

// extern template void Marching4SimplexCyclic<double>(const std::vector<std::array<double, 4> >& vertices,
//                                                     const std::vector<std::array<size_t, 5> >& simplices,
//                                                     const std::vector<std::array<size_t, 2> >& vertex_pairs,
//                                                     const std::vector<double>& values,
//                                                     std::vector<std::array<double, 4> >& output_vertices,
//                                                     std::vector<std::array<size_t, 4> >& output_tets,
//                                                     std::vector<std::array<size_t, 6> >& output_prisms);

// extern template void Marching4SimplexCyclic<float>(const std::vector<std::array<float, 4> >& vertices,
//                                                    const std::vector<std::array<size_t, 5> >& simplices,
//                                                    const std::vector<std::array<size_t, 2> >& vertex_pairs,
//                                                    const std::vector<float>& values,
//                                                    std::vector<std::array<float, 4> >& output_vertices,
//                                                    std::vector<std::array<size_t, 4> >& output_tets,
//                                                    std::vector<std::array<size_t, 6> >& output_prisms);

/**
 * @brief Given a list of vertices, tetrahedrons, prisms, scalar values, and gradients (optional) at each vertex,
 * compute the zero-surface as a list of 3D zero-crossing points and a list of 2D triangular faces.
 * The resulting triangles are orientated towards the positive space.
 * @param vertices 4D vertices of the tet-prism mesh
 * @param tets tets of the tet-prism mesh
 * @param prisms prisms of the tet-prism mesh
 * @param values scalar values at each vertex
 * @param gradients gradients at each vertex. Only used when its size is equal to vertices.size().
 * @param output_vertices output vertices of the zero-surface
 * @param output_triangles output triangles of the zero-surface
 * @param prism_insert_cycle_center triangulate each prism cycle with >3 vertices by inserting the centroid.
 */
template <typename Float>
void MarchingTetPrism(const std::vector<std::array<Float, 4> >& vertices,
                      const std::vector<std::array<size_t, 4> >& tets,
                      const std::vector<std::array<size_t, 6> >& prisms,
                      const std::vector<Float>& values,
                      const std::vector<std::array<Float, 4> >& gradients,
                      std::vector<std::array<Float, 3> >& output_vertices,
                      std::vector<std::array<size_t, 3> >& output_triangles,
                      bool prism_insert_cycle_center = false);

// extern template void MarchingTetPrism<double>(const std::vector<std::array<double, 4> >& vertices,
//                                               const std::vector<std::array<size_t, 4> >& tets,
//                                               const std::vector<std::array<size_t, 6> >& prisms,
//                                               const std::vector<double>& values,
//                                               const std::vector<std::array<double, 4> >& gradients,
//                                               std::vector<std::array<double, 3> >& output_vertices,
//                                               std::vector<std::array<size_t, 3> >& output_triangles,
//                                               bool prism_insert_cycle_center);

// extern template void MarchingTetPrism<float>(const std::vector<std::array<float, 4> >& vertices,
//                                              const std::vector<std::array<size_t, 4> >& tets,
//                                              const std::vector<std::array<size_t, 6> >& prisms,
//                                              const std::vector<float>& values,
//                                              const std::vector<std::array<float, 4> >& gradients,
//                                              std::vector<std::array<float, 3> >& output_vertices,
//                                              std::vector<std::array<size_t, 3> >& output_triangles,
//                                              bool prism_insert_cycle_center);

/**
 * @brief Given a list of vertices, tetrahedrons,  scalar values, and gradients (optional) at each vertex,
 * compute the zero-surface as a list of 3D zero-crossing points and a list of 2D triangular faces.
 * The resulting triangles are orientated towards the positive space.
 * @param vertices 3D vertices of the tet-prism mesh
 * @param tets tets of the tet-prism mesh
 * @param values scalar values at each vertex
 * @param gradients gradients at each vertex. Only used when its size is equal to vertices.size().
 * @param output_vertices output vertices of the zero-surface
 * @param output_triangles output triangles of the zero-surface
 */
template <typename Float>
void MarchingTet3D(const std::vector<std::array<Float, 3> >& vertices,
                   const std::vector<std::array<size_t, 4> >& tets,
                   const std::vector<Float>& values,
                   const std::vector<std::array<Float, 3> >& gradients,
                   std::vector<std::array<Float, 3> >& output_vertices,
                   std::vector<std::array<size_t, 3> >& output_triangles);

// extern template void MarchingTet3D<double>(const std::vector<std::array<double, 3> >& vertices,
//                                            const std::vector<std::array<size_t, 4> >& tets,
//                                            const std::vector<double>& values,
//                                            const std::vector<std::array<double, 3> >& gradients,
//                                            std::vector<std::array<double, 3> >& output_vertices,
//                                            std::vector<std::array<size_t, 3> >& output_triangles);

// extern template void MarchingTet3D<float>(const std::vector<std::array<float, 3> >& vertices,
//                                           const std::vector<std::array<size_t, 4> >& tets,
//                                           const std::vector<float>& values,
//                                           const std::vector<std::array<float, 3> >& gradients,
//                                           std::vector<std::array<float, 3> >& output_vertices,
//                                           std::vector<std::array<size_t, 3> >& output_triangles);
} // namespace marching4D