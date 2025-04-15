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

bool save_mesh_json(const std::string& filename,
                    const mtet::MTetMesh mesh)
{
    vector<array<double, 3>> vertices((int)mesh.get_num_vertices());
    vector<array<size_t, 4>> tets((int)mesh.get_num_tets());
    using IndexMap = ankerl::unordered_dense::map<uint64_t, size_t>;
    IndexMap vertex_tag_map;
    vertex_tag_map.reserve(mesh.get_num_vertices());
    int counter = 0;
    mesh.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3> data){
        size_t vertex_tag = vertex_tag_map.size() + 1;
        vertex_tag_map[value_of(vid)] = vertex_tag;
        vertices[counter] = {data[0], data[1], data[2]};
        counter ++;
    });
    counter = 0;
    mesh.seq_foreach_tet([&](TetId, std::span<const VertexId, 4> data) {
        tets[counter] = {vertex_tag_map[value_of(data[0])] - 1, vertex_tag_map[value_of(data[1])] - 1, vertex_tag_map[value_of(data[2])] - 1, vertex_tag_map[value_of(data[3])] - 1};
        counter ++;
    });
    if (std::filesystem::exists(filename.c_str())){
        std::filesystem::remove(filename.c_str());
    }
    using json = nlohmann::json;
    std::ofstream fout(filename.c_str(),std::ios::app);
    json jOut;
    jOut.push_back(json(vertices));
    jOut.push_back(json(tets));
    fout << jOut.dump(4, ' ', true, json::error_handler_t::replace) << std::endl;
    fout.close();
    return true;
}

bool get_mesh_data(const mtet::MTetMesh& mesh, 
                    vector<array<double, 3>>& vertices, 
                    vector<array<size_t, 4>>& tets)
{
    vertices.resize((int)mesh.get_num_vertices());
    tets.resize((int)mesh.get_num_tets());
    using IndexMap = ankerl::unordered_dense::map<uint64_t, size_t>;
    IndexMap vertex_tag_map;
    vertex_tag_map.reserve(mesh.get_num_vertices());
    int counter = 0;
    mesh.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3> data){
        size_t vertex_tag = vertex_tag_map.size() + 1;
        vertex_tag_map[value_of(vid)] = vertex_tag;
        vertices[counter] = {data[0], data[1], data[2]};
        counter ++;
    });
    counter = 0;
    mesh.seq_foreach_tet([&](TetId, std::span<const VertexId, 4> data) {
        tets[counter] = {vertex_tag_map[value_of(data[0])] - 1, vertex_tag_map[value_of(data[1])] - 1, vertex_tag_map[value_of(data[2])] - 1, vertex_tag_map[value_of(data[3])] - 1};
        counter ++;
    });
    return true;
}

bool save_function_json(const std::string& filename,
                        const mtet::MTetMesh mesh,
                        ankerl::unordered_dense::map<uint64_t, llvm_vecsmall::SmallVector<std::array<double, 4>, 20>> vertex_func_grad_map,
                        const size_t funcNum)
{
    vector<vector<double>> values(funcNum);
    for (size_t funcIter = 0; funcIter <  funcNum; funcIter++){
        values[funcIter].reserve(((int)mesh.get_num_vertices()));
    }
    mesh.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3> data){
        llvm_vecsmall::SmallVector<std::array<double, 4>, 20> func_gradList(funcNum);
        func_gradList = vertex_func_grad_map[value_of(vid)];
        for (size_t funcIter = 0; funcIter < funcNum; funcIter++){
            // cout << data[0] << " " << data[1] << " " << data[2] << ": " << func_gradList[funcIter][0] << ", " << func_gradList[funcIter][1] << ", " << func_gradList[funcIter][2] << ", " << func_gradList[funcIter][3] << endl;
            values[funcIter].push_back(func_gradList[funcIter][0]);
        }
    });
    if (std::filesystem::exists(filename.c_str())){
        std::filesystem::remove(filename.c_str());
    }
    using json = nlohmann::json;
    std::ofstream fout(filename.c_str(),std::ios::app);
    json jOut;
    for (size_t funcIter = 0; funcIter <  funcNum; funcIter++){
        json jFunc;
        jFunc["type"] = "customized";
        jFunc["value"] = values[funcIter];
        jOut.push_back(jFunc);
    }
    fout << jOut.dump(4, ' ', true, json::error_handler_t::replace) << std::endl;
    fout.close();
    return true;
}

bool get_function_val_and_gradients(const mtet::MTetMesh& mesh,
                        ankerl::unordered_dense::map<uint64_t, llvm_vecsmall::SmallVector<std::array<double, 4>, 20>>& vertex_func_grad_map,
                        vector<double>& values,
                        vector<std::array<double, 3>>& gradients)
{
    values.resize((int)mesh.get_num_vertices());
    gradients.resize((int)mesh.get_num_vertices());
    std::cout << "mesh vertices size " << (int)mesh.get_num_vertices() << std::endl;
    std::cout << " vertex_func_grad_map size " << vertex_func_grad_map.size() << std::endl;
    int counter = 0;
    mesh.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3> data){
        llvm_vecsmall::SmallVector<std::array<double, 4>, 20> func_gradList(1);
        size_t id = value_of(vid); 
        func_gradList = vertex_func_grad_map[id];
        size_t funcId = 0;
        // std::cout << func_gradList[funcId][0] << std::endl;
        // std::cout << " v id " << id << std::endl;
        values[counter] = func_gradList[funcId][0];
        gradients[counter][0] = func_gradList[funcId][1];
        gradients[counter][1] = func_gradList[funcId][2];
        gradients[counter][2] = func_gradList[funcId][3];
        counter ++;    
    });
    
    return true;
}

//hash for mounting a boolean that represents the activeness to a tet
//since the tetid isn't const during the process, mount the boolean using vertexids of 4 corners.
uint64_t vertexHash(std::span<VertexId, 4>& x)
{
    ankerl::unordered_dense::hash<uint64_t> hash_fn;
    return hash_fn(value_of(x[0])) + hash_fn(value_of(x[1])) + hash_fn(value_of(x[2])) + hash_fn(value_of(x[3]));
}

void GenerateAdaptiveGridOut(const std::array<size_t, 3>& resolution, 
                             const std::array<double, 3>& bbox_min, 
                             const std::array<double, 3>& bbox_max,
                             const std::string& outdir,
                             const std::string& filename,
                             std::vector<shared_ptr<ImplicitFunction<double>>>& functions,
                             double in_threshold,
                             std::vector<std::array<double, 3> >& output_vertices,
                            std::vector<std::array<size_t, 3> >& output_triangles)
{
    struct
    {
        string mesh_file;
        string function_file;
        double threshold = 0.001;
        double alpha = std::numeric_limits<double>::infinity();
        int max_elements = -1;
        double smallest_edge_length = 0;
        string method = "IA";
        string csg_file;
        bool bfs = false;
        bool dfs = false;
        bool curve_network = false;
        //bool analysis_mode = false;
    } args;
    
    std::cout << "start to call  GenerateAdaptiveGridOut" << std::endl;
    std::cout << "bbox min " << bbox_min[0] << " " << bbox_min[1] << " " << bbox_min[2] << std::endl;
    std::cout << "bbox max " << bbox_max[0] << " " << bbox_max[1] << " " << bbox_max[2] << std::endl;

    double expand_scale = 0.2;
    double dx = bbox_max[0] - bbox_min[0];
    double dy = bbox_max[1] - bbox_min[1];
    double dz = bbox_max[2] - bbox_min[2];
    double max_len = std::max(dx, std::max(dy, dz));
    
    double max_scale = 5.0;
    double expand_scale_x = 0.2 * std::max(1.0, std::min(max_scale, max_len / dx / 2.0));
    double expand_scale_y = 0.2 * std::max(1.0, std::min(max_scale, max_len / dy / 2.0));
    double expand_scale_z = 0.2 * std::max(1.0, std::min(max_scale, max_len / dz / 2.0));

    // double expand_scale_x = expand_scale;
    // double expand_scale_y = expand_scale;
    // double expand_scale_z = expand_scale;

    // double expand_scale_x = 0.2 ;
    // double expand_scale_y = 0.2 ;
    // double expand_scale_z = 0.2 ;

    double cx = (bbox_max[0] + bbox_min[0]) / 2.0;
    double cy = (bbox_max[1] + bbox_min[1]) / 2.0;
    double cz = (bbox_max[2] + bbox_min[2]) / 2.0;

    std::array<double, 3> expand_bbox_min = {bbox_min[0] - expand_scale_x * dx, 
                                            bbox_min[1]  - expand_scale_y * dy,
                                            bbox_min[2]  - expand_scale_z * dz };
    std::array<double, 3> expand_bbox_max = {bbox_max[0] + expand_scale_x * dx, 
                                            bbox_max[1]  + expand_scale_y * dy,
                                            bbox_max[2]  + expand_scale_z * dz};

    // std::array<double, 3> new_bbox_min;
    // double cx = (bbox_max[0] + bbox_min[0]) / 2.0;
    // double cy = (bbox_max[1] + bbox_min[1]) / 2.0;
    // double cz = (bbox_max[2] + bbox_min[2]) / 2.0;
    // double expand_len = max_len / 2.0 * (1 + expand_scale);
    // std::array<double, 3> expand_bbox_min = {cx - expand_len, cy - expand_len, cz - expand_len};
    // std::array<double, 3> expand_bbox_max = {cx + expand_len, cy + expand_len, cz + expand_len};

    std::cout << " adgrid min bbox :  " << expand_bbox_min[0] << " " << expand_bbox_min[1] << " " << expand_bbox_min[2] << std::endl;
    std::cout << " adgrid max bbox :  " << expand_bbox_max[0] << " " << expand_bbox_max[1] << " " << expand_bbox_max[2] << std::endl;
    
    double new_dx = expand_bbox_max[0] - expand_bbox_min[0];
    double new_dy = expand_bbox_max[1] - expand_bbox_min[1];
    double new_dz = expand_bbox_max[2] - expand_bbox_min[2];
    double d_step = std::max(std::max(new_dx, new_dy), new_dz) / 10.0;
    int x_dim = max(int(new_dx / d_step + 0.5), 1);
    int y_dim = max(int(new_dy / d_step + 0.5), 1);
    int z_dim = max(int(new_dz / d_step + 0.5), 1);
    
    // double voxel_len = max_len / double(vol_dim);
    // int dimx = std::max(int(dx / voxel_len), 1);
    // int dimy = std::max(int(dy / voxel_len), 1);
    // int dimz = std::max(int(dz / voxel_len), 1);

    std::array<size_t, 3> new_resolution = {3, 3, 3};
    // std::array<size_t, 3> new_resolution = {x_dim, y_dim, z_dim};

    mtet::MTetMesh mesh = generate_tet_mesh(new_resolution, expand_bbox_min, expand_bbox_max, grid_mesh::TET5);
    // mtet::MTetMesh mesh;
    // json j;
    // j["resolution"] = {1};
    // j["bbox_min"] = expand_bbox_min;
    // j["bbox_max"] = expand_bbox_max;
    // std::string temp_out = "temp.json";
    // std::ofstream fout(temp_out.c_str(),std::ios::out);
    // fout << j.dump(4) << std::endl;
    // fout.close();
    // // if (args.mesh_file.find(".json") != std::string::npos){
    // mesh = grid_mesh::load_tet_mesh(temp_out);
    // mtet::save_mesh("init.msh", mesh);
    // mesh = mtet::load_mesh("init.msh");

    std::cout << " finish init tet mesh" << std::endl;
    // } else {
    //     mesh = mtet::load_mesh(args.mesh_file);
    // }


    // Read implicit function
    // vector<shared_ptr<HRBFDistanceFunction>> functions;
    // load_functions(args.function_file, functions);
    // std::shared_ptr<HRBFDistanceFunction> hrbf_func = std::make_shared<HRBFDistanceFunction>();
    // functions.push_back(hrbf_func);
    size_t funcNum = functions.size();
    // Read options
    if (args.max_elements < 0)
    {
        args.max_elements = numeric_limits<int>::max();
    }
    double threshold = in_threshold;
    double alpha = args.alpha;
    double smallest_edge_length = args.smallest_edge_length;
    
   
    
    //precomputing active multiples' indices:
    multiple_indices.resize(funcNum);
    for (int funcIter = 0; funcIter < funcNum; funcIter++){
        multiple_indices[funcIter].resize(3);
        int activeNum = funcIter + 1;
        int pairNum = activeNum * (activeNum-1)/2, triNum = activeNum * (activeNum-1) * (activeNum - 2)/ 6;
        int quadNum = activeNum * (activeNum - 1) * (activeNum - 2) * (activeNum - 3)/ 24;
        llvm_vecsmall::SmallVector<array<int, 4>,100> pair(pairNum);
        llvm_vecsmall::SmallVector<array<int, 4>, 100> triple(triNum);
        llvm_vecsmall::SmallVector<array<int, 4>, 100> quad(quadNum);
        int pairIt = 0, triIt = 0, quadIt = 0;
        for (int i = 0; i < activeNum - 1; i++){
            for (int j = i + 1; j < activeNum; j++){
                pair[pairIt] = {i, j, 0, 0};
                pairIt ++;
                if (j < activeNum - 1){
                    for (int k = j + 1; k < activeNum; k++){
                        triple[triIt] = {i, j, k, 0};
                        triIt ++;
                        if (GLOBAL_METHOD == MI){
                            if (k < activeNum - 1){
                                for (int m = k + 1; m < activeNum; m++){
                                    quad[quadIt] = {i, j, k, m};
                                    quadIt++;
                                }
                            }
                        }
                    }
                }
            }
        }
        if (GLOBAL_METHOD == MI){
            multiple_indices[funcIter] = {pair, triple, quad};
        }else{
            multiple_indices[funcIter] = {pair, triple};
        }
    }

    // std::cout << " finish init tet mesh  000 " << std::endl;
    int search_counter;
    if (args.bfs || args.dfs){
        search_counter = 0;
    }
    // initialize vertex map: vertex index -> {{f_i, gx, gy, gz} | for all f_i in the function}
    using IndexMap = ankerl::unordered_dense::map<uint64_t, llvm_vecsmall::SmallVector<std::array<double, 4>, 20>>;
    IndexMap vertex_func_grad_map;
    vertex_func_grad_map.reserve(mesh.get_num_vertices());

    // std::cout << " finish init tet mesh  000 00" << std::endl;
    
    //initialize activeness map: four vertexids (v0, v1, v2, v3) -> hash(v0, v1, v2, v3) -> active boolean
    using activeMap = ankerl::unordered_dense::map<uint64_t, bool>;
    activeMap vertex_active_map;
    vertex_active_map.reserve(mesh.get_num_tets());

    // std::cout << " finish init tet mesh  000 000" << std::endl;
    
    mesh.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3> data)
                            {
        llvm_vecsmall::SmallVector<std::array<double, 4>, 20> func_gradList(funcNum);
        for(size_t funcIter = 0; funcIter < funcNum; funcIter++){
            auto &func = functions[funcIter];
            array<double, 4> func_grad;
            func_grad[0] = func->evaluate_gradient(data[0], data[1], data[2], func_grad[1], func_grad[2], func_grad[3]);
            func_gradList[funcIter] = func_grad;
        }
        vertex_func_grad_map[value_of(vid)] = func_gradList;});
    auto comp = [](std::pair<mtet::Scalar, mtet::EdgeId> e0,
                   std::pair<mtet::Scalar, mtet::EdgeId> e1)
    { return e0.first < e1.first; };
    std::vector<std::pair<mtet::Scalar, mtet::EdgeId>> Q;
    // std::cout << " finish init tet mesh  000 11" << std::endl;
    std::array<std::array<double, 3>, 4> pts;
    llvm_vecsmall::SmallVector<std::array<double, 4>, 20> vals(funcNum);
    llvm_vecsmall::SmallVector<std::array<std::array<double, 3>,4>, 20> grads(funcNum);
    double activeTet = 0;
    auto push_longest_edge = [&](mtet::TetId tid)
    {
        std::span<VertexId, 4> vs = mesh.get_tet(tid);
        {
            Timer eval_timer(evaluation, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
            for (int i = 0; i < 4; ++i)
            {
                auto vid = vs[i];
                auto coords = mesh.get_vertex(vid);
                pts[i][0] = coords[0];
                pts[i][1] = coords[1];
                pts[i][2] = coords[2];
                llvm_vecsmall::SmallVector<std::array<double, 4>, 20> func_gradList(funcNum);
                std::array<double, 4> func_grad;
                if (!vertex_func_grad_map.contains(value_of(vid))) {
                    for(size_t funcIter = 0; funcIter < funcNum; funcIter++){
                        auto &func = functions[funcIter];
                        array<double, 4> func_grad;
                        func_grad[0] = func->evaluate_gradient(coords[0], coords[1], coords[2], func_grad[1], func_grad[2],
                                                               func_grad[3]);
                        func_gradList[funcIter] = func_grad;
                    }
                    vertex_func_grad_map[value_of(vid)] = func_gradList;
                }
                else {
                    func_gradList = vertex_func_grad_map[value_of(vid)];
                }
                for(size_t funcIter = 0; funcIter < funcNum; funcIter++){
                    vals[funcIter][i] = func_gradList[funcIter][0];
                    grads[funcIter][i][0] = func_gradList[funcIter][1];
                    grads[funcIter][i][1] = func_gradList[funcIter][2];
                    grads[funcIter][i][2] = func_gradList[funcIter][3];
                }
            }
            eval_timer.Stop();
        }
        bool isActive = 0;
        bool subResult;
        {
            Timer sub_timer(subdivision, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
            if (GLOBAL_METHOD != MI){
                subResult = subTet(pts, vals, grads, threshold, isActive);
            }else{
                subResult = subMI(pts, vals, grads, threshold, isActive);
            }
            sub_timer.Stop();
        }
        vertex_active_map[vertexHash(vs)] = isActive;
        Timer eval_timer(evaluation, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        if (subResult)
        {
            mtet::EdgeId longest_edge;
            mtet::Scalar longest_edge_length = 0;
            mesh.foreach_edge_in_tet(tid, [&](mtet::EdgeId eid, mtet::VertexId v0, mtet::VertexId v1)
                                     {
                auto p0 = mesh.get_vertex(v0);
                auto p1 = mesh.get_vertex(v1);
                mtet::Scalar l = (p0[0] - p1[0]) * (p0[0] - p1[0]) + (p0[1] - p1[1]) * (p0[1] - p1[1]) +
                (p0[2] - p1[2]) * (p0[2] - p1[2]);
                if (l > longest_edge_length) {
                    longest_edge_length = l;
                    longest_edge = eid;
                } });
            if (args.bfs){
                Q.emplace_back(search_counter, longest_edge);
                search_counter--;
            }else if(args.dfs){
                Q.emplace_back(search_counter, longest_edge);
                search_counter++;
            }
            else{
                Q.emplace_back(longest_edge_length, longest_edge);
            }
            eval_timer.Stop();
            return true;
        }
        eval_timer.Stop();
        return false;
    };
    
    
    {
        Timer timer(total_time, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        
        // Initialize priority queue.
        mesh.seq_foreach_tet([&](mtet::TetId tid, [[maybe_unused]] std::span<const mtet::VertexId, 4> vs)
                             { push_longest_edge(tid); });
        std::make_heap(Q.begin(), Q.end(), comp);
        
        // Keep splitting the longest edge
        while (!Q.empty())
        {
            std::pop_heap(Q.begin(), Q.end(), comp);
            auto [edge_length, eid] = Q.back();
            if (!mesh.has_edge(eid)){
                Q.pop_back();
                continue;
            }
            //implement alpha value:
            mtet::Scalar comp_edge_length = alpha * edge_length;
            bool addedActive = false;
            mesh.foreach_tet_around_edge(eid,[&](mtet::TetId tid){
                std::span<VertexId, 4> vs = mesh.get_tet(tid);
                if(vertex_active_map.contains(vertexHash(vs))){
                    if (vertex_active_map[vertexHash(vs)]){
                        mtet::EdgeId longest_edge;
                        mtet::Scalar longest_edge_length = 0;
                        mesh.foreach_edge_in_tet(tid, [&](mtet::EdgeId eid_active, mtet::VertexId v0, mtet::VertexId v1)
                                                 {
                            auto p0 = mesh.get_vertex(v0);
                            auto p1 = mesh.get_vertex(v1);
                            mtet::Scalar l = (p0[0] - p1[0]) * (p0[0] - p1[0]) + (p0[1] - p1[1]) * (p0[1] - p1[1]) +
                            (p0[2] - p1[2]) * (p0[2] - p1[2]);
                            if (l > longest_edge_length) {
                                longest_edge_length = l;
                                longest_edge = eid_active;
                            }
                        });
                        if (longest_edge_length > comp_edge_length) {
                            Q.emplace_back(longest_edge_length, longest_edge);
                            addedActive = true;
                        }
                    }
                }
            });
            if(addedActive){
                std::push_heap(Q.begin(), Q.end(), comp);
                continue;
            }
            Q.pop_back();
            std::array<VertexId, 2> vs_old = mesh.get_edge_vertices(eid);
            Timer split_timer(splitting, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
            auto [vid, eid0, eid1] = mesh.split_edge(eid);
            split_timer.Stop();
            //std::cout << "Number of elements: " << mesh.get_num_tets() << std::endl;
            if (mesh.get_num_tets() > args.max_elements) {
                break;
            }
            mesh.foreach_tet_around_edge(eid0, [&](mtet::TetId tid)
                                         {
                if (push_longest_edge(tid)) {
                    std::push_heap(Q.begin(), Q.end(), comp);
                } });
            mesh.foreach_tet_around_edge(eid1, [&](mtet::TetId tid)
                                         {
                if (push_longest_edge(tid)) {
                    std::push_heap(Q.begin(), Q.end(), comp);
                } });
                
#ifdef Check_Flip_Tets
            std::array<VertexId, 2> vs_new = mesh.get_edge_vertices(eid0);
            mesh.foreach_tet_around_edge(eid0, [&](mtet::TetId tid)
                                         {
                std::span<VertexId, 4> vs = mesh.get_tet(tid);
                std::vector<VertexId> parent(4);
                parent[0] = vs_old[0]; parent[1] = vs_old[1];
                int parentIndex = 2;
                for (auto vIter : vs){
                    if (vIter !=  vs_new[0] && vIter != vs_new[1]){
                        parent[parentIndex] = vIter;
                        parentIndex ++;
                    }
                }
                std::span<VertexId, 4> spanVec(parent);
                if (!vertex_active_map[vertexHash(spanVec)] && vertex_active_map[vertexHash(vs)]){
                    {
                        using json = nlohmann::json;
                        std::string filePath = "flip_tets.json";
                        //                        if (std::filesystem::exists(filePath)) {
                        //                            std::filesystem::remove(filePath);
                        //                        }
                        std::ofstream fout(filePath,std::ios::app);
                        json jOut;
                        std::array<std::array<double, 3>, 4> pts;
                        llvm_vecsmall::SmallVector<std::array<double, 4>, 20> vals(funcNum);
                        llvm_vecsmall::SmallVector<std::array<std::array<double, 3>,4>, 20> grads(funcNum);
                        for (size_t i = 0; i < 4; ++i) {
                            auto coords = mesh.get_vertex(vs[i]);
                            pts[i][0] = coords[0];
                            pts[i][1] = coords[1];
                            pts[i][2] = coords[2];
                            auto func_gradList = vertex_func_grad_map[value_of(vs[i])];
                            for(size_t funcIter = 0; funcIter < funcNum; funcIter++){
                                vals[funcIter][i] = func_gradList[funcIter][0];
                                grads[funcIter][i][0] = func_gradList[funcIter][1];
                                grads[funcIter][i][1] = func_gradList[funcIter][2];
                                grads[funcIter][i][2] = func_gradList[funcIter][3];
                            }
                        }
                        jOut["vertices: "] = pts;
                        jOut["value: "] = vals;
                        jOut["gradient: "] = grads;
                        //
                        fout << jOut << std::endl;
                        fout.close();
                    }GenerateAdaptiveGridOutparent.json";
                        //                        if (std::filesystem::exists(filePath)) {
                        //                            std::filesystem::remove(filePath);
                        //                        }
                        std::ofstream fout(filePath,std::ios::app);
                        json jOut;
                        std::array<std::array<double, 3>, 4> pts;
                        llvm_vecsmall::SmallVector<std::array<double, 4>, 20> vals(funcNum);
                        llvm_vecsmall::SmallVector<std::array<std::array<double, 3>,4>, 20> grads(funcNum);
                        for (size_t i = 0; i < 4; ++i) {
                            auto coords = mesh.get_vertex(spanVec[i]);
                            pts[i][0] = coords[0];
                            pts[i][1] = coords[1];
                            pts[i][2] = coords[2];
                            auto func_gradList = vertex_func_grad_map[value_of(spanVec[i])];
                            for(size_t funcIter = 0; funcIter < funcNum; funcIter++){
                                vals[funcIter][i] = func_gradList[funcIter][0];
                                grads[funcIter][i][0] = func_gradList[funcIter][1];
                                grads[funcIter][i][1] = func_gradList[funcIter][2];
                                grads[funcIter][i][2] = func_gradList[funcIter][3];
                            }
                        }
                        jOut["vertices: "] = pts;
                        jOut["value: "] = vals;
                        jOut["gradient: "] = grads;
                        //
                        fout << jOut << std::endl;
                        fout.close();
                    }
                }
            });
            vs_new = mesh.get_edge_vertices(eid1);
            mesh.foreach_tet_around_edge(eid1, [&](mtet::TetId tid)
                                         {
                std::span<VertexId, 4> vs = mesh.get_tet(tid);
                std::vector<VertexId> parent(4);
                parent[0] = vs_old[0]; parent[1] = vs_old[1];
                int parentIndex = 2;
                for (auto vIter : vs){
                    if (vIter !=  vs_new[0] && vIter != vs_new[1]){
                        parent[parentIndex] = vIter;
                        parentIndex ++;
                    }
                }
                std::span<VertexId, 4> spanVec(parent);
                if (!vertex_active_map[vertexHash(spanVec)] && vertex_active_map[vertexHash(vs)]){
                    {
                        using json = nlohmann::json;
                        std::string filePath = "flip_tets.json";
                        //                        if (std::filesystem::exists(filePath)) {
                        //                            std::filesystem::remove(filePath);
                        //                        }
                        std::ofstream fout(filePath,std::ios::app);
                        json jOut;
                        std::array<std::array<double, 3>, 4> pts;
                        llvm_vecsmall::SmallVector<std::array<double, 4>, 20> vals(funcNum);
                        llvm_vecsmall::SmallVector<std::array<std::array<double, 3>,4>, 20> grads(funcNum);
                        for (size_t i = 0; i < 4; ++i) {
                            auto coords = mesh.get_vertex(vs[i]);
                            pts[i][0] = coords[0];
                            pts[i][1] = coords[1];
                            pts[i][2] = coords[2];
                            auto func_gradList = vertex_func_grad_map[value_of(vs[i])];
                            for(size_t funcIter = 0; funcIter < funcNum; funcIter++){
                                vals[funcIter][i] = func_gradList[funcIter][0];
                                grads[funcIter][i][0] = func_gradList[funcIter][1];
                                grads[funcIter][i][1] = func_gradList[funcIter][2];
                                grads[funcIter][i][2] = func_gradList[funcIter][3];
                            }
                        }
                        jOut["vertices: "] = pts;
                        jOut["value: "] = vals;
                        jOut["gradient: "] = grads;
                        //
                        fout << jOut << std::endl;
                        fout.close();
                    }
                    {
                        using json = nlohmann::json;
                        std::string filePath = "flip_tets_parent.json";
                        //                        if (std::filesystem::exists(filePath)) {
                        //                            std::filesystem::remove(filePath);
                        //                        }
                        std::ofstream fout(filePath,std::ios::app);
                        json jOut;
                        std::array<std::array<double, 3>, 4> pts;
                        llvm_vecsmall::SmallVector<std::array<double, 4>, 20> vals(funcNum);
                        llvm_vecsmall::SmallVector<std::array<std::array<double, 3>,4>, 20> grads(funcNum);
                        for (size_t i = 0; i < 4; ++i) {
                            auto coords = mesh.get_vertex(spanVec[i]);
                            pts[i][0] = coords[0];
                            pts[i][1] = coords[1];
                            pts[i][2] = coords[2];
                            auto func_gradList = vertex_func_grad_map[value_of(spanVec[i])];
                            for(size_t funcIter = 0; funcIter < funcNum; funcIter++){
                                vals[funcIter][i] = func_gradList[funcIter][0];
                                grads[funcIter][i][0] = func_gradList[funcIter][1];
                                grads[funcIter][i][1] = func_gradList[funcIter][2];
                                grads[funcIter][i][2] = func_gradList[funcIter][3];
                            }
                        }
                        jOut["vertices: "] = pts;
                        jOut["value: "] = vals;
                        jOut["gradient: "] = grads;
                        //
                        fout << jOut << std::endl;
                        fout.close();
                    }
                }
            });
#endif
        }
        timer.Stop();
    }

    //profiled time(see details in time.h) and profiled number of calls to zero
    for (int i = 0; i < profileTimer.size(); i++){
        timeProfileName time_type = static_cast<timeProfileName>(i);
        std::cout << time_label[i] << ": " << profileTimer[i] << std::endl;
    }
    //    std::cout << profileTimer[0] << " "<< profileTimer[1] << " "<< profileTimer[2] << " "<< profileTimer[3] << " "<< profileTimer[4] << " "<< profileTimer[5] << " "<< profileTimer[6] << " "<< profileTimer[7] << " "<< profileTimer[8] << " "<< profileTimer[9] << " "<< sub_call_two << " "<< sub_call_three << std::endl;
    //std::cout << "sub two func calls: " << sub_call_two << std::endl;
    //std::cout << "sub three func calls: " << sub_call_three << std::endl;
    double min_rratio_all = 1;
    double min_rratio_active = 1;
    std::vector<mtet::TetId> activeTetId;
    mesh.seq_foreach_tet([&](mtet::TetId tid, std::span<const VertexId, 4> data) {
        std::span<VertexId, 4> vs = mesh.get_tet(tid);
        std::array<valarray<double>,4> vallPoints;
        for (int i = 0; i < 4; i++){
            vallPoints[i] = {0.0,0.0,0.0};
        }
        for (int i = 0; i < 4; i++){
            VertexId vid = vs[i];
            std::span<Scalar, 3> coords = mesh.get_vertex(vid);
            vallPoints[i][0] = coords[0];
            vallPoints[i][1] = coords[1];
            vallPoints[i][2] = coords[2];
        }
        double ratio = tet_radius_ratio(vallPoints);
        if (ratio < min_rratio_all){
            min_rratio_all = ratio;
            
        }
        if(vertex_active_map.contains(vertexHash(vs))){
            if (vertex_active_map[vertexHash(vs)]){
                activeTet++;
                activeTetId.push_back(tid);
                if (ratio < min_rratio_active){
                    min_rratio_active = ratio;
                }
            }
        }
    });

    std::string outfile = outdir + "/" + filename;
    // save timing records
    std::string str_th =  std::to_string(in_threshold);

    str_th.erase(str_th.find_last_not_of('0') + 1, std::string::npos);
    
    // save_timings(outfile + "_timings_" + str_th + ".json",time_label, profileTimer);
    // save statistics
    // save_metrics(outfile + "_stats_"  + str_th +".json", tet_metric_labels, {(double)mesh.get_num_tets(), activeTet, min_rratio_all, min_rratio_active, (double)sub_call_two, (double) sub_call_three});
    // save the mesh output for isosurfacing tool
    // save_mesh_json(outfile + "_mesh_" + str_th + ".json", mesh);
    // save the mesh output for isosurfacing tool
    // save_function_json(outfile + "_function_value_" + str_th + ".json", mesh, vertex_func_grad_map, funcNum);
    vector<array<double, 3>> vertices;
    vector<array<size_t, 4>> tets;
    
    // std::string input_tet_path = "/home/jjxia/Documents/prejects/VIPSS_LOCAL/output/kitten/kitten_2000_tet_tet_data.txt";
    // ReadTetFromFile(vertices, tets, input_tet_path);
    get_mesh_data(mesh, vertices, tets);

    std::cout << "vertices size " << vertices.size() << std::endl;
    std::cout << "tets size " << tets.size() << std::endl;

    vector<double> values;
    vector<std::array<double, 3>> gradients;
    get_function_val_and_gradients(mesh, vertex_func_grad_map, values, gradients);
    // values.resize(vertices.size());
    // gradients.resize(vertices.size());
    // size_t v_count = 0;
    // for(const auto& pt : vertices)
    // {
    //     // std::array<double, 3> gradient;       
    //     // double gradient[3];
    //     // double dist_val = LocalVipss::NNDistGradient({pt[0], pt[1], pt[2]}, gradient);
    //     double dist_val = LocalVipss::NNDistFunction({pt[0], pt[1], pt[2]});
    //     std::array<double, 3> gradient   = LocalVipss::NumericGradientFunction({pt[0], pt[1], pt[2]});
    //     values[v_count] = dist_val;
    //     gradients[v_count] = {gradient[0], gradient[1], gradient[2]};
    //     v_count ++;
    // }
    
    std::cout << " val size " <<  values.size() << std::endl;
    std::cout << " gradients size " <<  gradients.size() << std::endl;
    std::cout << " vertices size " <<  vertices.size() << std::endl;

    marching3D::MarchingTet3D( vertices, tets, values, gradients, output_vertices, output_triangles);

    // std::string tet_edge_path = outfile + "_tet_edges.ply";
    // SaveTetMeshToPly(vertices, tets, values, tet_edge_path);
    
    // std::string tet_path = outfile + "_tet_data.txt";
    // SaveTetToFile(vertices, tets, tet_path);

    // //write mesh and active tets
    // std::cout << "save tet mesh path : " << outfile + "_tet_mesh_" + str_th + ".msh" << std::endl;
    // mtet::save_mesh(outfile + "_tet_mesh_" + str_th + ".msh", mesh);
    // mtet::save_mesh(outfile + "_active_tets_" + str_th + ".msh", mesh, std::span<mtet::TetId>(activeTetId));
    
}
