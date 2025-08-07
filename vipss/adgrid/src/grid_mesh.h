//
// Created by Charles Du on 1/15/24.
//
#pragma once

#include <vector>
#include <string>
#include <cassert>
#include <mtet/mtet.h>
#include <nlohmann/json.hpp>

namespace grid_mesh {

    enum GridStyle {
        TET5, // 5 tetrahedrons per grid cell
        TET6  // 6 tetrahedrons per grid cell
    };

    mtet::MTetMesh generate_tet_mesh(const std::array<size_t, 3> &resolution,
                                     const std::array<double, 3> &bbox_min,
                                     const std::array<double, 3> &bbox_max,
                                     GridStyle style = TET5) {
        assert(resolution[0] > 0 && resolution[1] > 0 && resolution[2] > 0);
        const size_t N0 = resolution[0] + 1;
        const size_t N1 = resolution[1] + 1;
        const size_t N2 = resolution[2] + 1;
        std::vector<std::array<double, 3>> pts(N0 * N1 * N2);
        auto compute_coordinate = [&](double t, size_t i) {
            return t * (bbox_max[i] - bbox_min[i]) + bbox_min[i];
        };
        // vertices
        for (size_t i = 0; i < N0; i++) {
            double x = compute_coordinate(double(i) / double(N0 - 1), 0);
            for (size_t j = 0; j < N1; j++) {
                double y = compute_coordinate(double(j) / double(N1 - 1), 1);
                for (size_t k = 0; k < N2; k++) {
                    double z = compute_coordinate(double(k) / double(N2 - 1), 2);

                    size_t idx = i * N1 * N2 + j * N2 + k;
                    pts[idx] = {x, y, z};
                }
            }
        }
        // tets
        std::vector<std::array<size_t, 4>> tets;
        size_t num_tet_per_cell = 0;
        if (style == TET5) {
            num_tet_per_cell = 5;
        } else if (style == TET6) {
            num_tet_per_cell = 6;
        } else {
            throw std::runtime_error("unknown grid style!");
        }
        tets.resize(resolution[0] * resolution[1] * resolution[2] * num_tet_per_cell);
        for (size_t i = 0; i < resolution[0]; i++) {
            for (size_t j = 0; j < resolution[1]; j++) {
                for (size_t k = 0; k < resolution[2]; k++) {
                    size_t idx = (i * resolution[1] * resolution[2] + j * resolution[2] + k) * num_tet_per_cell;
                    size_t v0 = i * N1 * N2 + j * N2 + k;
                    size_t v1 = (i + 1) * N1 * N2 + j * N2 + k;
                    size_t v2 = (i + 1) * N1 * N2 + (j + 1) * N2 + k;
                    size_t v3 = i * N1 * N2 + (j + 1) * N2 + k;
                    size_t v4 = i * N1 * N2 + j * N2 + k + 1;
                    size_t v5 = (i + 1) * N1 * N2 + j * N2 + k + 1;
                    size_t v6 = (i + 1) * N1 * N2 + (j + 1) * N2 + k + 1;
                    size_t v7 = i * N1 * N2 + (j + 1) * N2 + k + 1;
                    switch (style) {
                        case TET5:
                            if ((i + j + k) % 2 == 0) {
                                tets[idx] = {v4, v6, v1, v3};
                                tets[idx + 1] = {v6, v3, v4, v7};
                                tets[idx + 2] = {v1, v3, v0, v4};
                                tets[idx + 3] = {v3, v1, v2, v6};
                                tets[idx + 4] = {v4, v1, v6, v5};
                            } else {
                                tets[idx] = {v7, v0, v2, v5};
                                tets[idx + 1] = {v2, v3, v0, v7};
                                tets[idx + 2] = {v5, v7, v0, v4};
                                tets[idx + 3] = {v7, v2, v6, v5};
                                tets[idx + 4] = {v0, v1, v2, v5};
                            }
                            break;
                        case TET6:
                            //{{0, 4, 6, 7}, {6, 0, 5, 4}, {1, 0, 5, 6}, {1, 2, 0, 6}, {0, 6, 2, 3}, {6, 3, 0, 7}}
                            tets[idx] = {v0, v4, v6, v7};
                            tets[idx + 1] = {v6, v0, v5, v4};
                            tets[idx + 2] = {v1, v0, v5, v6};
                            tets[idx + 3] = {v1, v2, v0, v6};
                            tets[idx + 4] = {v0, v6, v2, v3};
                            tets[idx + 5] = {v6, v3, v0, v7};
                            break;
                    }
                }
            }
        }
        // build mesh
        mtet::MTetMesh mesh;
        std::vector<mtet::VertexId> vertex_ids;
        vertex_ids.reserve(pts.size());
        for (auto &v: pts) {
            vertex_ids.push_back(mesh.add_vertex(v[0], v[1], v[2]));
        }
        for (auto &t: tets) {
            mesh.add_tet(vertex_ids[t[0]], vertex_ids[t[1]], vertex_ids[t[2]], vertex_ids[t[3]]);
        }
        mesh.initialize_connectivity();
        return mesh;
    }

    mtet::MTetMesh generate_from_kuhn_mesh(const std::array<size_t, 3> &resolution,
                                     const std::array<double, 3> &bbox_min,
                                     const std::array<double, 3> &bbox_max,
                                     GridStyle style = TET6) {
        assert(resolution[0] > 0);
        std::vector<std::array<double, 3>> pts(8);
        auto compute_coordinate = [&](double t, size_t i) {
            return t * (bbox_max[i] - bbox_min[i]) + bbox_min[i];
        };
        // vertices
        for (size_t i = 0; i < 2; i++) {
            double x = compute_coordinate(double(i) / double(2 - 1), 0);
            for (size_t j = 0; j < 2; j++) {
                double y = compute_coordinate(double(j) / double(2 - 1), 1);
                for (size_t k = 0; k < 2; k++) {
                    double z = compute_coordinate(double(k) / double(2 - 1), 2);

                    size_t idx = i * 4 + j * 2 + k;
                    pts[idx] = {x, y, z};
                }
            }
        }
        // tets
        std::array<std::array<size_t, 4>, 6> tets;
        tets[0] = {0, 1, 7, 3};
        tets[1] = {7, 0, 5, 1};
        tets[2] = {4, 0, 5, 7};
        tets[3] = {4, 6, 0, 7};
        tets[4] = {0, 7, 6, 2};
        tets[5] = {7, 2, 0, 3};

        // build mesh
        mtet::MTetMesh grid;
        std::vector<mtet::VertexId> vertex_ids;
        vertex_ids.reserve(pts.size());
        for (auto &v: pts) {
            vertex_ids.push_back(grid.add_vertex(v[0], v[1], v[2]));
        }
        for (auto &t: tets) {
            grid.add_tet(vertex_ids[t[0]], vertex_ids[t[1]], vertex_ids[t[2]], vertex_ids[t[3]]);
        }
        std::array<double, 3> bound_box = {bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1], bbox_max[2] - bbox_min[2]};
        double longest_edge = *std::max_element(bound_box.begin(), bound_box.end()) * resolution[0];
        
        auto comp = [](std::pair<mtet::Scalar, mtet::EdgeId> e0,
                       std::pair<mtet::Scalar, mtet::EdgeId> e1)
        { return e0.first < e1.first; };
        std::vector<std::pair<mtet::Scalar, mtet::EdgeId>> Q;
        auto push_longest_edge = [&](mtet::TetId tid, mtet::Scalar longest_bound)
        {
            std::span<mtet::VertexId, 4> vs = grid.get_tet(tid);
            mtet::EdgeId longest_edge;
            mtet::Scalar longest_edge_length = 0;
            grid.foreach_edge_in_tet(tid, [&](mtet::EdgeId eid, mtet::VertexId v0, mtet::VertexId v1)
                                     {
                auto p0 = grid.get_vertex(v0);
                auto p1 = grid.get_vertex(v1);
                mtet::Scalar l = (p0[0] - p1[0]) * (p0[0] - p1[0]) + (p0[1] - p1[1]) * (p0[1] - p1[1]) +
                (p0[2] - p1[2]) * (p0[2] - p1[2]);
                if (l > longest_edge_length) {
                    longest_edge_length = l;
                    longest_edge = eid;
                } });
            if (longest_edge_length > longest_bound){
                Q.emplace_back(longest_edge_length, longest_edge);
                return true;
            }
            return false;
        };
        grid.seq_foreach_tet([&](mtet::TetId tid, [[maybe_unused]] std::span<const mtet::VertexId, 4> vs)
                             { push_longest_edge(tid, longest_edge); });
        std::make_heap(Q.begin(), Q.end(), comp);
        while (!Q.empty())
        {
            std::pop_heap(Q.begin(), Q.end(), comp);
            auto [edge_length, eid] = Q.back();
            if (!grid.has_edge(eid)){
                Q.pop_back();
                continue;
            }
            Q.pop_back();
            auto [vid, eid0, eid1] = grid.split_edge(eid);
            grid.foreach_tet_around_edge(eid0, [&](mtet::TetId tid)
                                         {
                if (push_longest_edge(tid, longest_edge)) {
                    std::push_heap(Q.begin(), Q.end(), comp);
                } });
            grid.foreach_tet_around_edge(eid1, [&](mtet::TetId tid)
                                         {
                if (push_longest_edge(tid, longest_edge)) {
                    std::push_heap(Q.begin(), Q.end(), comp);
                } });
        }
        return grid;
    }

    // load tet mesh from json file
    mtet::MTetMesh load_tet_mesh(const std::string &filename) {
        using json = nlohmann::json;
        std::ifstream fin(filename.c_str());
        if (!fin) {
            throw std::runtime_error("tet mesh file not exist!");
        }
        json data;
        fin >> data;
        fin.close();
        // if the tet grid is specified by resolution and bounding box
        if (data.contains("resolution")) {
            size_t num_resolution = data["resolution"].size();
            assert(num_resolution <= 3 && num_resolution > 0);
            size_t res = data["resolution"][0].get<size_t>();
            std::array<size_t, 3> resolution = {res, res, res};
            for (size_t i = 0; i < num_resolution; i++) {
                resolution[i] = data["resolution"][i].get<size_t>();
            }
            assert(data.contains("bbox_min"));
            assert(data["bbox_min"].size() == 3);
            std::array<double, 3> bbox_min{0, 0, 0};
            for (size_t i = 0; i < 3; i++) {
                bbox_min[i] = data["bbox_min"][i].get<double>();
            }
            assert(data.contains("bbox_max"));
            assert(data["bbox_max"].size() == 3);
            std::array<double, 3> bbox_max{1, 1, 1};
            for (size_t i = 0; i < 3; i++) {
                bbox_max[i] = data["bbox_max"][i].get<double>();
            }
            GridStyle style = TET6;
            if (data.contains("style")) {
                auto style_str = data["style"].get<std::string>();
                if (style_str == "TET5") {
                    style = TET5;
                } else if (style_str == "TET6") {
                    style = TET6;
                } else {
                    throw std::runtime_error("unknown grid style!");
                }
            }
            return generate_tet_mesh(resolution, bbox_min, bbox_max, style);
        }
        // vertices
        std::vector<std::array<double, 3>> pts;
        pts.resize(data[0].size());
        for (size_t j = 0; j < pts.size(); j++) {
            for (size_t k = 0; k < 3; k++) {
                pts[j][k] = data[0][j][k].get<double>();
            }
        }
        // tets
        std::vector<std::array<size_t, 4>> tets;
        tets.resize(data[1].size());
        for (size_t j = 0; j < tets.size(); j++) {
            for (size_t k = 0; k < 4; k++) {
                tets[j][k] = data[1][j][k].get<size_t>();
            }
        }
        // build mesh
        mtet::MTetMesh mesh;
        std::vector<mtet::VertexId> vertex_ids;
        vertex_ids.reserve(pts.size());
        for (auto &v: pts) {
            vertex_ids.push_back(mesh.add_vertex(v[0], v[1], v[2]));
        }
        for (auto &t: tets) {
            mesh.add_tet(vertex_ids[t[0]], vertex_ids[t[1]], vertex_ids[t[2]], vertex_ids[t[3]]);
        }
        return mesh;
    }


} // namespace grid_mesh


