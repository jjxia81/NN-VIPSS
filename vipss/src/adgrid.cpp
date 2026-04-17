#include "adgrid.h"
#include <iostream>

//#define Check_Flip_Tets
#include <mtet/mtet.h>
#include <mtet/io.h>
#include <ankerl/unordered_dense.h>
#include <span>
#include <queue>
#include <optional>
#include <SmallVector.h>

#include <implicit_functions/implicit_functions.h>
#include <adgrid/subdivide_multi.h>
#include <CLI/CLI.hpp>
#include <adgrid/tet_quality.h>
#include <adgrid/timer.h>
#include <adgrid/grid_mesh.h>
#include <nlohmann/json.hpp>
#include "marching3D.h"

using json = nlohmann::json;
using namespace mtet;

// =============================================================================
// Type aliases used across helpers
// =============================================================================
using FuncGradList  = llvm_vecsmall::SmallVector<std::array<double, 4>, 20>;
using VertexFuncMap = ankerl::unordered_dense::map<uint64_t, FuncGradList>;
using TetActiveMap  = ankerl::unordered_dense::map<uint64_t, bool>;
using EdgeQueue     = std::vector<std::pair<mtet::Scalar, mtet::EdgeId>>;

// =============================================================================
// Small utility helpers
// =============================================================================

// Hash for identifying a tet by its 4 corner vertex ids (tet ids are not
// stable across refinement, but vertex ids are).
static uint64_t vertexHash(std::span<VertexId, 4>& x)
{
    ankerl::unordered_dense::hash<uint64_t> hash_fn;
    return hash_fn(value_of(x[0])) + hash_fn(value_of(x[1]))
         + hash_fn(value_of(x[2])) + hash_fn(value_of(x[3]));
}

// Compute squared length of an edge.
static inline mtet::Scalar edge_length_sq(std::span<const Scalar, 3> p0,
                                          std::span<const Scalar, 3> p1)
{
    const double dx = p0[0] - p1[0];
    const double dy = p0[1] - p1[1];
    const double dz = p0[2] - p1[2];
    return dx * dx + dy * dy + dz * dz;
}

// Find the longest edge of a given tet and return (length^2, edge_id).
static std::pair<mtet::Scalar, mtet::EdgeId>
find_longest_edge_in_tet(mtet::MTetMesh& mesh, mtet::TetId tid)
{
    mtet::EdgeId longest_edge;
    mtet::Scalar longest_edge_length = 0;
    mesh.foreach_edge_in_tet(tid,
        [&](mtet::EdgeId eid, mtet::VertexId v0, mtet::VertexId v1) {
            auto l = edge_length_sq(mesh.get_vertex(v0), mesh.get_vertex(v1));
            if (l > longest_edge_length) {
                longest_edge_length = l;
                longest_edge = eid;
            }
        });
    return {longest_edge_length, longest_edge};
}

// =============================================================================
// Public mesh <-> JSON helpers (unchanged from original)
// =============================================================================

bool save_mesh_json(const std::string& filename, const mtet::MTetMesh mesh)
{
    std::vector<std::array<double, 3>> vertices((int)mesh.get_num_vertices());
    std::vector<std::array<size_t, 4>> tets((int)mesh.get_num_tets());
    using IndexMap = ankerl::unordered_dense::map<uint64_t, size_t>;
    IndexMap vertex_tag_map;
    vertex_tag_map.reserve(mesh.get_num_vertices());

    int counter = 0;
    mesh.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3> data) {
        size_t vertex_tag = vertex_tag_map.size() + 1;
        vertex_tag_map[value_of(vid)] = vertex_tag;
        vertices[counter] = {data[0], data[1], data[2]};
        counter++;
    });
    counter = 0;
    mesh.seq_foreach_tet([&](TetId, std::span<const VertexId, 4> data) {
        tets[counter] = {
            vertex_tag_map[value_of(data[0])] - 1,
            vertex_tag_map[value_of(data[1])] - 1,
            vertex_tag_map[value_of(data[2])] - 1,
            vertex_tag_map[value_of(data[3])] - 1};
        counter++;
    });
    if (std::filesystem::exists(filename.c_str())) {
        std::filesystem::remove(filename.c_str());
    }
    std::ofstream fout(filename.c_str(), std::ios::app);
    json jOut;
    jOut.push_back(json(vertices));
    jOut.push_back(json(tets));
    fout << jOut.dump(4, ' ', true, json::error_handler_t::replace) << std::endl;
    fout.close();
    return true;
}

bool get_mesh_data(const mtet::MTetMesh& mesh,
                   std::vector<std::array<double, 3>>& vertices,
                   std::vector<std::array<size_t, 4>>& tets)
{
    vertices.resize((int)mesh.get_num_vertices());
    tets.resize((int)mesh.get_num_tets());
    using IndexMap = ankerl::unordered_dense::map<uint64_t, size_t>;
    IndexMap vertex_tag_map;
    vertex_tag_map.reserve(mesh.get_num_vertices());

    int counter = 0;
    mesh.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3> data) {
        size_t vertex_tag = vertex_tag_map.size() + 1;
        vertex_tag_map[value_of(vid)] = vertex_tag;
        vertices[counter] = {data[0], data[1], data[2]};
        counter++;
    });
    counter = 0;
    mesh.seq_foreach_tet([&](TetId, std::span<const VertexId, 4> data) {
        tets[counter] = {
            vertex_tag_map[value_of(data[0])] - 1,
            vertex_tag_map[value_of(data[1])] - 1,
            vertex_tag_map[value_of(data[2])] - 1,
            vertex_tag_map[value_of(data[3])] - 1};
        counter++;
    });
    return true;
}

bool save_function_json(const std::string& filename,
                       const mtet::MTetMesh mesh,
                       VertexFuncMap vertex_func_grad_map,
                       const size_t funcNum)
{
    std::vector<std::vector<double>> values(funcNum);
    for (size_t f = 0; f < funcNum; f++) {
        values[f].reserve(((int)mesh.get_num_vertices()));
    }
    mesh.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3>) {
        FuncGradList func_gradList(funcNum);
        func_gradList = vertex_func_grad_map[value_of(vid)];
        for (size_t f = 0; f < funcNum; f++) {
            values[f].push_back(func_gradList[f][0]);
        }
    });
    if (std::filesystem::exists(filename.c_str())) {
        std::filesystem::remove(filename.c_str());
    }
    std::ofstream fout(filename.c_str(), std::ios::app);
    json jOut;
    for (size_t f = 0; f < funcNum; f++) {
        json jFunc;
        jFunc["type"] = "customized";
        jFunc["value"] = values[f];
        jOut.push_back(jFunc);
    }
    fout << jOut.dump(4, ' ', true, json::error_handler_t::replace) << std::endl;
    fout.close();
    return true;
}

bool get_function_val_and_gradients(
    const mtet::MTetMesh& mesh,
    VertexFuncMap& vertex_func_grad_map,
    std::vector<double>& values,
    std::vector<std::array<double, 3>>& gradients)
{
    values.resize((int)mesh.get_num_vertices());
    gradients.resize((int)mesh.get_num_vertices());
    std::cout << "mesh vertices size " << (int)mesh.get_num_vertices() << std::endl;
    std::cout << " vertex_func_grad_map size " << vertex_func_grad_map.size() << std::endl;

    int counter = 0;
    mesh.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3>) {
        FuncGradList func_gradList(1);
        size_t id = value_of(vid);
        func_gradList = vertex_func_grad_map[id];
        size_t funcId = 0;
        values[counter]       = func_gradList[funcId][0];
        gradients[counter][0] = func_gradList[funcId][1];
        gradients[counter][1] = func_gradList[funcId][2];
        gradients[counter][2] = func_gradList[funcId][3];
        counter++;
    });
    return true;
}

// =============================================================================
// Stage 1: bounding-box expansion
// =============================================================================
struct ExpandedBBox {
    std::array<double, 3> min;
    std::array<double, 3> max;
};

static ExpandedBBox compute_expanded_bbox(const std::array<double, 3>& bbox_min,
                                          const std::array<double, 3>& bbox_max)
{
    const double dx = bbox_max[0] - bbox_min[0];
    const double dy = bbox_max[1] - bbox_min[1];
    const double dz = bbox_max[2] - bbox_min[2];
    const double max_len = std::max({dx, dy, dz});

    constexpr double base_scale = 0.2;
    constexpr double max_scale  = 5.0;
    const double ex = base_scale * std::max(1.0, std::min(max_scale, max_len / dx / 2.0));
    const double ey = base_scale * std::max(1.0, std::min(max_scale, max_len / dy / 2.0));
    const double ez = base_scale * std::max(1.0, std::min(max_scale, max_len / dz / 2.0));

    ExpandedBBox out;
    out.min = {bbox_min[0] - ex * dx, bbox_min[1] - ey * dy, bbox_min[2] - ez * dz};
    out.max = {bbox_max[0] + ex * dx, bbox_max[1] + ey * dy, bbox_max[2] + ez * dz};

    std::cout << " adgrid min bbox :  " << out.min[0] << " " << out.min[1] << " " << out.min[2] << std::endl;
    std::cout << " adgrid max bbox :  " << out.max[0] << " " << out.max[1] << " " << out.max[2] << std::endl;
    return out;
}

// =============================================================================
// Stage 2: precompute multiple_indices (active-function combinations)
// =============================================================================
static void precompute_multiple_indices(size_t funcNum)
{
    multiple_indices.resize(funcNum);
    for (int funcIter = 0; funcIter < (int)funcNum; funcIter++) {
        multiple_indices[funcIter].resize(3);
        const int activeNum = funcIter + 1;
        const int pairNum   = activeNum * (activeNum - 1) / 2;
        const int triNum    = activeNum * (activeNum - 1) * (activeNum - 2) / 6;
        const int quadNum   = activeNum * (activeNum - 1) * (activeNum - 2) * (activeNum - 3) / 24;

        llvm_vecsmall::SmallVector<std::array<int, 4>, 100> pair(pairNum);
        llvm_vecsmall::SmallVector<std::array<int, 4>, 100> triple(triNum);
        llvm_vecsmall::SmallVector<std::array<int, 4>, 100> quad(quadNum);
        int pairIt = 0, triIt = 0, quadIt = 0;

        for (int i = 0; i < activeNum - 1; i++) {
            for (int j = i + 1; j < activeNum; j++) {
                pair[pairIt++] = {i, j, 0, 0};
                if (j < activeNum - 1) {
                    for (int k = j + 1; k < activeNum; k++) {
                        triple[triIt++] = {i, j, k, 0};
                        if (GLOBAL_METHOD == MI && k < activeNum - 1) {
                            for (int m = k + 1; m < activeNum; m++) {
                                quad[quadIt++] = {i, j, k, m};
                            }
                        }
                    }
                }
            }
        }
        if (GLOBAL_METHOD == MI) {
            multiple_indices[funcIter] = {pair, triple, quad};
        } else {
            multiple_indices[funcIter] = {pair, triple};
        }
    }
}

// =============================================================================
// Stage 3: initial evaluation of all vertex functions / gradients
// =============================================================================
static void evaluate_vertex_functions(
    const mtet::MTetMesh& mesh,
    const std::vector<std::shared_ptr<ImplicitFunction<double>>>& functions,
    VertexFuncMap& vertex_func_grad_map)
{
    const size_t funcNum = functions.size();
    mesh.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3> data) {
        FuncGradList func_gradList(funcNum);
        for (size_t f = 0; f < funcNum; f++) {
            std::array<double, 4> fg;
            fg[0] = functions[f]->evaluate_gradient(
                data[0], data[1], data[2], fg[1], fg[2], fg[3]);
            func_gradList[f] = fg;
        }
        vertex_func_grad_map[value_of(vid)] = func_gradList;
    });
}

// =============================================================================
// Stage 4: evaluate / cache functions at the 4 corners of a tet
// =============================================================================
static void gather_tet_corners(
    mtet::MTetMesh& mesh,
    mtet::TetId tid,
    const std::vector<std::shared_ptr<ImplicitFunction<double>>>& functions,
    VertexFuncMap& vertex_func_grad_map,
    std::array<std::array<double, 3>, 4>& pts,
    llvm_vecsmall::SmallVector<std::array<double, 4>, 20>& vals,
    llvm_vecsmall::SmallVector<std::array<std::array<double, 3>, 4>, 20>& grads)
{
    const size_t funcNum = functions.size();
    std::span<VertexId, 4> vs = mesh.get_tet(tid);

    for (int i = 0; i < 4; ++i) {
        auto vid    = vs[i];
        auto coords = mesh.get_vertex(vid);
        pts[i][0] = coords[0];
        pts[i][1] = coords[1];
        pts[i][2] = coords[2];

        FuncGradList func_gradList(funcNum);
        if (!vertex_func_grad_map.contains(value_of(vid))) {
            for (size_t f = 0; f < funcNum; f++) {
                std::array<double, 4> fg;
                fg[0] = functions[f]->evaluate_gradient(
                    coords[0], coords[1], coords[2], fg[1], fg[2], fg[3]);
                func_gradList[f] = fg;
            }
            vertex_func_grad_map[value_of(vid)] = func_gradList;
        } else {
            func_gradList = vertex_func_grad_map[value_of(vid)];
        }

        for (size_t f = 0; f < funcNum; f++) {
            vals[f][i]     = func_gradList[f][0];
            grads[f][i][0] = func_gradList[f][1];
            grads[f][i][1] = func_gradList[f][2];
            grads[f][i][2] = func_gradList[f][3];
        }
    }
}

// =============================================================================
// Stage 5: process a single tet — decide whether it needs subdivision and, if
// so, push its longest edge onto the queue. Returns true if an edge was pushed.
// =============================================================================
struct PushEdgeContext {
    bool bfs;
    bool dfs;
    int& search_counter;
};

static bool push_longest_edge_for_tet(
    mtet::MTetMesh& mesh,
    mtet::TetId tid,
    const std::vector<std::shared_ptr<ImplicitFunction<double>>>& functions,
    VertexFuncMap& vertex_func_grad_map,
    TetActiveMap& vertex_active_map,
    EdgeQueue& Q,
    double threshold,
    PushEdgeContext& ctx)
{
    const size_t funcNum = functions.size();
    std::span<VertexId, 4> vs = mesh.get_tet(tid);

    std::array<std::array<double, 3>, 4> pts;
    llvm_vecsmall::SmallVector<std::array<double, 4>, 20> vals(funcNum);
    llvm_vecsmall::SmallVector<std::array<std::array<double, 3>, 4>, 20> grads(funcNum);

    {
        Timer eval_timer(evaluation,
            [&](auto r) { profileTimer = combine_timer(profileTimer, r); });
        gather_tet_corners(mesh, tid, functions, vertex_func_grad_map,
                           pts, vals, grads);
        eval_timer.Stop();
    }

    bool isActive  = false;
    bool subResult = false;
    {
        Timer sub_timer(subdivision,
            [&](auto r) { profileTimer = combine_timer(profileTimer, r); });
        if (GLOBAL_METHOD != MI) {
            subResult = subTet(pts, vals, grads, threshold, isActive);
        } else {
            subResult = subMI(pts, vals, grads, threshold, isActive);
        }
        sub_timer.Stop();
    }
    vertex_active_map[vertexHash(vs)] = isActive;

    Timer eval_timer(evaluation,
        [&](auto r) { profileTimer = combine_timer(profileTimer, r); });
    if (subResult) {
        auto [longest_edge_length, longest_edge] = find_longest_edge_in_tet(mesh, tid);
        if (ctx.bfs) {
            Q.emplace_back(ctx.search_counter, longest_edge);
            ctx.search_counter--;
        } else if (ctx.dfs) {
            Q.emplace_back(ctx.search_counter, longest_edge);
            ctx.search_counter++;
        } else {
            Q.emplace_back(longest_edge_length, longest_edge);
        }
        eval_timer.Stop();
        return true;
    }
    eval_timer.Stop();
    return false;
}

// =============================================================================
// Stage 6: main adaptive-subdivision loop
// =============================================================================
struct SubdivideOptions {
    double threshold;
    double alpha;
    int    max_elements;
    bool   bfs;
    bool   dfs;
};

static void run_adaptive_subdivision(
    mtet::MTetMesh& mesh,
    const std::vector<std::shared_ptr<ImplicitFunction<double>>>& functions,
    VertexFuncMap& vertex_func_grad_map,
    TetActiveMap& vertex_active_map,
    const SubdivideOptions& opts)
{
    auto comp = [](const std::pair<mtet::Scalar, mtet::EdgeId>& e0,
                   const std::pair<mtet::Scalar, mtet::EdgeId>& e1) {
        return e0.first < e1.first;
    };
    EdgeQueue Q;

    int search_counter = 0;
    PushEdgeContext ctx{opts.bfs, opts.dfs, search_counter};

    // Seed the queue with every initial tet.
    mesh.seq_foreach_tet([&](mtet::TetId tid,
                             [[maybe_unused]] std::span<const mtet::VertexId, 4>) {
        push_longest_edge_for_tet(mesh, tid, functions, vertex_func_grad_map,
                                  vertex_active_map, Q, opts.threshold, ctx);
    });
    std::make_heap(Q.begin(), Q.end(), comp);

    // Keep splitting the longest edge until queue drains or we hit the cap.
    while (!Q.empty()) {
        std::pop_heap(Q.begin(), Q.end(), comp);
        auto [edge_length, eid] = Q.back();
        if (!mesh.has_edge(eid)) {
            Q.pop_back();
            continue;
        }
        // Alpha-based early re-queue of active neighbors.
        const mtet::Scalar comp_edge_length = opts.alpha * edge_length;
        bool addedActive = false;
        mesh.foreach_tet_around_edge(eid, [&](mtet::TetId tid) {
            std::span<VertexId, 4> vs = mesh.get_tet(tid);
            if (vertex_active_map.contains(vertexHash(vs))
                && vertex_active_map[vertexHash(vs)]) {
                auto [len, le] = find_longest_edge_in_tet(mesh, tid);
                if (len > comp_edge_length) {
                    Q.emplace_back(len, le);
                    addedActive = true;
                }
            }
        });
        if (addedActive) {
            std::push_heap(Q.begin(), Q.end(), comp);
            continue;
        }
        Q.pop_back();

        // Split the edge.
        Timer split_timer(splitting,
            [&](auto r) { profileTimer = combine_timer(profileTimer, r); });
        auto [vid, eid0, eid1] = mesh.split_edge(eid);
        split_timer.Stop();

        if ((int)mesh.get_num_tets() > opts.max_elements) break;

        // Re-queue the newly-formed tets around both halves of the split edge.
        auto process_around = [&](mtet::EdgeId e) {
            mesh.foreach_tet_around_edge(e, [&](mtet::TetId tid) {
                if (push_longest_edge_for_tet(mesh, tid, functions,
                                              vertex_func_grad_map,
                                              vertex_active_map, Q,
                                              opts.threshold, ctx)) {
                    std::push_heap(Q.begin(), Q.end(), comp);
                }
            });
        };
        process_around(eid0);
        process_around(eid1);
    }
}

// =============================================================================
// Stage 7: mesh-quality statistics over the final tet mesh
// =============================================================================
struct MeshStats {
    double                   min_rratio_all    = 1.0;
    double                   min_rratio_active = 1.0;
    double                   active_tet_count  = 0.0;
    std::vector<mtet::TetId> active_tet_ids;
};

static MeshStats compute_mesh_stats(mtet::MTetMesh& mesh,
                                    TetActiveMap& vertex_active_map)
{
    MeshStats stats;
    mesh.seq_foreach_tet([&](mtet::TetId tid, std::span<const VertexId, 4>) {
        std::span<VertexId, 4> vs = mesh.get_tet(tid);
        std::array<std::valarray<double>, 4> vallPoints;
        for (int i = 0; i < 4; i++) vallPoints[i] = {0.0, 0.0, 0.0};
        for (int i = 0; i < 4; i++) {
            std::span<Scalar, 3> coords = mesh.get_vertex(vs[i]);
            vallPoints[i][0] = coords[0];
            vallPoints[i][1] = coords[1];
            vallPoints[i][2] = coords[2];
        }
        const double ratio = tet_radius_ratio(vallPoints);
        if (ratio < stats.min_rratio_all) stats.min_rratio_all = ratio;

        if (vertex_active_map.contains(vertexHash(vs))
            && vertex_active_map[vertexHash(vs)]) {
            stats.active_tet_count++;
            stats.active_tet_ids.push_back(tid);
            if (ratio < stats.min_rratio_active) stats.min_rratio_active = ratio;
        }
    });
    return stats;
}

// =============================================================================
// Stage 8: optional re-evaluation of function values/grads directly on the
// output grid (kept as a helper but not invoked by default).
// =============================================================================
static void evaluate_values_on_vertices(
    const std::vector<std::array<double, 3>>& vertices,
    const std::vector<std::shared_ptr<ImplicitFunction<double>>>& functions,
    std::vector<double>& values,
    std::vector<std::array<double, 3>>& gradients)
{
    values.resize(vertices.size());
    gradients.resize(vertices.size());
    size_t v_count = 0;
    for (const auto& pt : vertices) {
        std::array<double, 3> g;
        values[v_count] = functions[0]->evaluate_gradient(
            pt[0], pt[1], pt[2], g[0], g[1], g[2]);
        gradients[v_count] = g;
        v_count++;
    }
}

// =============================================================================
// Public entry point — now a short, high-level orchestrator.
// =============================================================================
void GenerateAdaptiveGridOut(
    const std::array<size_t, 3>& resolution,
    const std::array<double, 3>& bbox_min,
    const std::array<double, 3>& bbox_max,
    // const std::string& /*outdir*/,
    // const std::string& /*filename*/,
    std::vector<std::shared_ptr<ImplicitFunction<double>>>& functions,
    double in_threshold,
    std::vector<std::array<double, 3>>& output_vertices,
    std::vector<std::array<size_t, 3>>& output_triangles,
    bool refine_grid /*= true*/)
{
    std::cout << "start to call  GenerateAdaptiveGridOut" << std::endl;
    std::cout << "bbox min " << bbox_min[0] << " " << bbox_min[1] << " " << bbox_min[2] << std::endl;
    std::cout << "bbox max " << bbox_max[0] << " " << bbox_max[1] << " " << bbox_max[2] << std::endl;
    std::cout << "refine_grid: " << (refine_grid ? "true" : "false") << std::endl;
 
    // 1. Expand bbox and build initial tet mesh.
    const ExpandedBBox ebb = compute_expanded_bbox(bbox_min, bbox_max);
    // const std::array<size_t, 3> init_resolution = {3, 3, 3};
    mtet::MTetMesh mesh = generate_tet_mesh(
        resolution, ebb.min, ebb.max, grid_mesh::TET5);
    std::cout << " finish init tet mesh" << std::endl;
 
    // Cached per-vertex function/gradient values. Populated either by the
    // refinement pipeline or (when refinement is off) just before marching.
    VertexFuncMap vertex_func_grad_map;
    vertex_func_grad_map.reserve(mesh.get_num_vertices());
    // refine_grid = false;
    if (refine_grid) {
        // 2. Bundle subdivision options.
        const size_t funcNum = functions.size();
        SubdivideOptions opts;
        opts.threshold    = in_threshold;
        opts.alpha        = std::numeric_limits<double>::infinity();
        opts.max_elements = std::numeric_limits<int>::max();
        opts.bfs          = false;
        opts.dfs          = false;
 
        // 3. Precompute active-function combinatorial indices.
        precompute_multiple_indices(funcNum);
 
        // 4. Evaluate functions at all initial vertices.
        evaluate_vertex_functions(mesh, functions, vertex_func_grad_map);
 
        TetActiveMap vertex_active_map;
        vertex_active_map.reserve(mesh.get_num_tets());
 
        // 5. Run the adaptive subdivision loop (timed end-to-end).
        {
            Timer timer(total_time,
                [&](auto r) { profileTimer = combine_timer(profileTimer, r); });
            run_adaptive_subdivision(mesh, functions, vertex_func_grad_map,
                                     vertex_active_map, opts);
            timer.Stop();
        }
        // 6. Dump profile timings.
        for (int i = 0; i < (int)profileTimer.size(); i++) {
            std::cout << time_label[i] << ": " << profileTimer[i] << std::endl;
        }
        // 7. Collect mesh quality statistics.
        MeshStats stats = compute_mesh_stats(mesh, vertex_active_map);
        (void)stats;
    } else {
        // Refinement disabled — still need values at every initial vertex so
        // that marching tets has something to interpolate against.
        evaluate_vertex_functions(mesh, functions, vertex_func_grad_map);
    }
 
    // 8. Extract data and run marching tets to produce the surface mesh.
    std::vector<std::array<double, 3>> vertices;
    std::vector<std::array<size_t, 4>> tets;
    get_mesh_data(mesh, vertices, tets);
    std::cout << "vertices size " << vertices.size() << std::endl;
    std::cout << "tets size "     << tets.size()     << std::endl;
 
    std::vector<double> values;
    std::vector<std::array<double, 3>> gradients;
    get_function_val_and_gradients(mesh, vertex_func_grad_map, values, gradients);
  
    // std::cout << " val size "       << values.size()    << std::endl;
    // std::cout << " gradients size " << gradients.size() << std::endl;
    // std::cout << " vertices size "  << vertices.size()  << std::endl;
 
    marching3D::MarchingTet3D(vertices, tets, values, gradients,
                              output_vertices, output_triangles);
}