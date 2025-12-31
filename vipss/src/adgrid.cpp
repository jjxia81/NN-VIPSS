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
// #include <subdivide_multi.h>
#include <CLI/CLI.hpp>
#include <tet_quality.h>
#include <nlohmann/json.hpp>

#include "timer.h"
#include "csg.h"
#include "grid_mesh.h"
#include "grid_refine.h"
#include "marching3D.h"

using json = nlohmann::json;
using namespace mtet;


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
    // uint64_t max_vid = 0;

    std::cout << " ----- mesh.get_num_vertices() " << mesh.get_num_vertices() << std::endl;

    mesh.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3> data){
        size_t vertex_tag = vertex_tag_map.size() + 1;
        // max_vid = max_vid > value_of(vid) ? value_of(vid) : value_of(vid);
        vertex_tag_map[value_of(vid)] = vertex_tag;
        vertices[counter] = {data[0], data[1], data[2]};
        counter ++;
    });
    std::unordered_map<std::string, std::array<double,8>> temp_map;
    auto& tet_edge_mid_sample_map = GetTetEdgeMidSampleMap();
    for(const auto& ele : tet_edge_mid_sample_map)
    {
        const auto& key = ele.first;
        const auto& val = ele.second;
        size_t pos = key.find('_');
        uint64_t a = std::stoull(key.substr(0, pos));
        uint64_t b = std::stoull(key.substr(pos + 1));
        size_t new_a_id = vertex_tag_map[a] - 1; 
        size_t new_b_id = vertex_tag_map[b] - 1; 
        std::string new_key = new_a_id < new_b_id ? 
        std::to_string(new_a_id) + "_" + std::to_string(new_b_id) :
        std::to_string(new_b_id) + "_" + std::to_string(new_a_id);
        temp_map[new_key] = val;
    }
    tet_edge_mid_sample_map.clear();
    tet_edge_mid_sample_map = temp_map;

    // std::cout << " ----- max_vid " << max_vid << std::endl;
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
                        IndexMap& vertex_func_grad_map,
                        vector<double>& values,
                        vector<std::array<double, 3>>& gradients)
{
    values.resize((int)mesh.get_num_vertices());
    gradients.resize((int)mesh.get_num_vertices());
    std::cout << "mesh vertices size " << (int)mesh.get_num_vertices() << std::endl;
    std::cout << " vertex_func_grad_map size " << vertex_func_grad_map.size() << std::endl;
    int counter = 0;
    mesh.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3> data){
        // llvm_vecsmall::SmallVector<std::array<double, 4>, 20> func_gradList(1);
        size_t id = value_of(vid); 
        auto  func_gradList = vertex_func_grad_map[value_of(vid)];
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


bool get_function_curvature_data(const mtet::MTetMesh& mesh,
                        IndexMapRidge& vertex_func_grad_map, 
                        vector<CurvatureData<double>>& tet_pt_curvature_data)
{
    tet_pt_curvature_data.resize((int)mesh.get_num_vertices());
    std::cout << "mesh vertices size " << (int)mesh.get_num_vertices() << std::endl;
    std::cout << " vertex_func_grad_map size " << vertex_func_grad_map.size() << std::endl;
    int counter = 0;
    mesh.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3> data){
        // llvm_vecsmall::SmallVector<std::array<double, 4>, 20> func_gradList(1);
        size_t id = value_of(vid); 
        auto  func_gradList = vertex_func_grad_map[value_of(vid)];
        size_t funcId = 0;
        // std::cout << func_gradList[funcId][0] << std::endl;
        // std::cout << " v id " << id << std::endl;
        // values[counter] = func_gradList[funcId][0];
        // gradients[counter][0] = func_gradList[funcId][1];
        // gradients[counter][1] = func_gradList[funcId][2];
        // gradients[counter][2] = func_gradList[funcId][3];
        tet_pt_curvature_data[counter] = func_gradList[funcId];
        counter ++;    
    });
    
    return true;
}

bool get_function_val_and_gradients_from_funcs(const vector<array<double, 3>> vertices,
                        shared_ptr<ImplicitFunction<double>> im_func,
                        vector<double>& values,
                        vector<std::array<double, 3>>& gradients)
{
    values.resize((int)vertices.size());
    gradients.resize((int)vertices.size());
    // std::cout << "mesh vertices size " << (int)mesh.get_num_vertices() << std::endl;
    // std::cout << " vertex_func_grad_map size " << vertex_func_grad_map.size() << std::endl;
    int counter = 0;
    for(size_t i = 0; i < vertices.size(); ++i){
        // llvm_vecsmall::SmallVector<std::array<double, 4>, 20> func_gradList(1);
        // auto  func_gradList = vertex_func_grad_map[value_of(vid)];
        // std::cout << func_gradList[funcId][0] << std::endl;
        // std::cout << " v id " << id << std::endl;
        const auto& pt = vertices[i];
        double gx, gy, gz = 0;
        values[i] = im_func->evaluate_gradient(pt[0], pt[1], pt[2], gx, gy,gz);
        gradients[i][0] = gx;
        gradients[i][1] = gy;
        gradients[i][2] = gz;
    };
    
    return true;
}

//hash for mounting a boolean that represents the activeness to a tet
//since the tetid isn't const during the process, mount the boolean using vertexids of 4 corners.
uint64_t vertexHash(std::span<VertexId, 4>& x)
{
    ankerl::unordered_dense::hash<uint64_t> hash_fn;
    return hash_fn(value_of(x[0])) + hash_fn(value_of(x[1])) + hash_fn(value_of(x[2])) + hash_fn(value_of(x[3]));
}

struct ADargs
{
    std::string grid_file;
    std::string function_file;
    double threshold;
    double alpha = std::numeric_limits<double>::infinity();
    // double alpha = 0.01;
    int max_elements = -1;
    double smallest_edge_length = 0;
    std::string method = "IA";
    std::string csg_file;
    bool bfs = false;
    bool dfs = false;
    bool curve_network = false;
    bool discretize_later = false;
    //bool analysis_mode = false;
} ;

void GenerateAdaptiveGridOut(const std::array<size_t, 3>& resolution, 
                             const std::array<double, 3>& bbox_min, 
                             const std::array<double, 3>& bbox_max,
                             const int crest_type,
                             const double tet_size_limit,
                             const std::string& outdir,
                             const std::string& filename,
                             std::vector<shared_ptr<ImplicitFunction<double>>>& functions,
                             double in_threshold,
                             std::vector<std::array<double, 3> >& output_vertices,
                            std::vector<std::array<size_t, 3> >& output_triangles)

// int main(int argc, const char *argv[])
{
    ADargs args;
    
    double expand_scale = 0.2;
    double dx = bbox_max[0] - bbox_min[0];
    double dy = bbox_max[1] - bbox_min[1];
    double dz = bbox_max[2] - bbox_min[2];
    double max_len = std::max(dx, std::max(dy, dz));
    
    double max_scale = 5.0;
    double expand_scale_x = expand_scale * std::max(1.0, std::min(max_scale, max_len / dx / 2.0));
    double expand_scale_y = expand_scale * std::max(1.0, std::min(max_scale, max_len / dy / 2.0));
    double expand_scale_z = expand_scale * std::max(1.0, std::min(max_scale, max_len / dz / 2.0));

    double cx = (bbox_max[0] + bbox_min[0]) / 2.0;
    double cy = (bbox_max[1] + bbox_min[1]) / 2.0;
    double cz = (bbox_max[2] + bbox_min[2]) / 2.0;

    std::array<double, 3> expand_bbox_min = {bbox_min[0] - expand_scale_x * dx, 
                                            bbox_min[1]  - expand_scale_y * dy,
                                            bbox_min[2]  - expand_scale_z * dz };
    std::array<double, 3> expand_bbox_max = {bbox_max[0] + expand_scale_x * dx, 
                                            bbox_max[1]  + expand_scale_y * dy,
                                            bbox_max[2]  + expand_scale_z * dz};
    
    // std::array<size_t, 3> new_resolution = {3, 3, 3};
    // mtet::MTetMesh grid_mesh = generate_tet_mesh(new_resolution, expand_bbox_min, expand_bbox_max, grid_mesh::TET5);
    size_t volume_dim = 4;
    std::array<size_t, 3> new_resolution = {volume_dim, volume_dim, volume_dim};
    // std::array<size_t, 3> new_resolution = {52, 52, 52};
    //Mesh Bounding Box min -1.291880 -0.607902 0.410953
    //Mesh Bounding Box max 0.772905 1.236420 1.989450
    //Mesh Bounding Box min -0.762772 -0.220353 0.525951
    //Mesh Bounding Box max 0.488482 0.977300 1.989450
    // expand_bbox_min = { -0.762772, -0.220353, 0.525951};
    // expand_bbox_max = { 0.488482,  0.977300,  1.989450};

    expand_bbox_min = { -1.5, -1.5, -1.5};
    expand_bbox_max = { 1.5,  1.5,  1.5};
    mtet::MTetMesh grid_mesh = generate_tet_mesh(new_resolution, expand_bbox_min, expand_bbox_max, grid_mesh::TET5);


// if(1)
// {
    args.threshold = in_threshold;
    int max_elements = args.max_elements;
    if (max_elements < 0)
    {
        max_elements = std::numeric_limits<int>::max();
    }
    std::string function_file = args.function_file;
    double threshold = args.threshold;
    int mode = IA;
    llvm_vecsmall::SmallVector<csg_unit, 20> csg_tree = {};
    // /// Read implicit function
    // std::vector<std::unique_ptr<ImplicitFunction<double>>> functions;
    // load_functions(function_file, functions);
    const size_t funcNum = functions.size();
    
    /// the lambda function for function evaluations
    ///  @param[in] data            The 3D coordinate
    ///  @param[in] funcNum         The number of functions
    ///
    ///  @return        A vector of `Eigen::RowVector4d`.The vector size is the function number. Each eigen vector represents the value at 0th index and gradients at {1, 2, 3} index.
    auto implicit_func = [&](std::span<const Scalar, 3> data, size_t funcNum){
        llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20> vertex_eval(funcNum);
        for(size_t funcIter = 0; funcIter < funcNum; funcIter++){
            auto &func = functions[funcIter];
            Eigen::Vector4d eval;
            eval[0] = func->evaluate_gradient(data[0], data[1], data[2], eval[1], eval[2], eval[3]);
            vertex_eval[funcIter] = eval;
        }
        return vertex_eval;
    };

    auto implicit_func_ridge = [&](std::span<const Scalar, 3> data, size_t funcNum){
        llvm_vecsmall::SmallVector<CurvatureData<double>, 20> vertex_eval(funcNum);
        for(size_t funcIter = 0; funcIter < funcNum; funcIter++){
            std::shared_ptr<Hermite_RBF<double>> func = std::dynamic_pointer_cast<Hermite_RBF<double>>(functions[funcIter]);
            // std::shared_ptr<ImplicitFunctionRidge<double>> func = std::dynamic_pointer_cast<ImplicitFunctionRidge<double>>(functions[funcIter]);
            // Vec7D eval;
            // eval[0] = (static_cast<std::shared_ptr<Hermite_RBF<double>>>(func))->evaluate_gradient(data[0], data[1], data[2], eval[1], eval[2], eval[3]);
            CurvatureData<double> curv_data;
            // std::cout << " start  EvaluateCurvatureData ...... " << std::endl;
            // std::cout << " data : " << data[0] << " " << data[1] << " " << data[2] << std::endl;
            // std::cout << " control points size " << func->control_points_.size() << std::endl;;
            func->EvaluateCurvatureData(data[0], data[1], data[2], curv_data);
            // std::cout << " finish  EvaluateCurvatureData ...... " << std::endl;
            // eval << curv_data.e2_, curv_data.t2_, curv_data.e2_d2_;
            // std::cout << " finish  assign eval ...... " << std::endl;
            vertex_eval[funcIter] = curv_data;
        }
        return vertex_eval;
    };
    
    // std::shared_ptr<Hermite_RBF<double>> funcTemp = 
    // std::dynamic_pointer_cast<Hermite_RBF<double>>(functions[0]);
    // double gx, gy, gz;
    // funcTemp->evaluate_gradient(0.0010000, 0.583785, -0.313607, gx, gy, gz );
    // std::cout << " test pt gradient : " << std::endl;
    // std::cout << gx << " " << gy<< " " << gz<< std::endl;
    // Hermite_RBF<double>::Mat33 hessian;
    // funcTemp->EvaluateHessian(0.0010000, 0.583785, -0.313607,  hessian);
    // std::cout << " test hessian mat : " << std::endl;
    // std::cout << hessian << std::endl;

    /// the lambda function for csg tree iteration/evaluation.
    /// @param[in] funcInt          Given an input of value range std::array<double, 2> for an arbitrary number of functions
    /// @return   A value range of this CSG operation in a form of `std::array<double, 2>` and a list of active function in a form of    `llvm_vecsmall::SmallVector<int, 20>>`
    ///
    auto csg_func = [&](llvm_vecsmall::SmallVector<std::array<double, 2>, 20> funcInt){
        if (args.csg_file == ""){
            throw std::runtime_error("ERROR: no csg file provided");
            std::pair<std::array<double, 2>, llvm_vecsmall::SmallVector<int, 20>> null_csg = {{},{}};
            return null_csg;
        }else{
            return iterTree(csg_tree, 1, funcInt);
        }
    };
    printf("start gridRefineRidges ...... \n");
    //perform main grid refinement algorithm:
    tet_metric metric_list;
    //an array of 10 timings: {total time getting the multiple indices, total time,time spent on single function, time spent on double functions, time spent on triple functions time spent on double functions' zero crossing test, time spent on three functions' zero crossing test, total subdivision time, total evaluation time,total splitting time}
    std::array<double, timer_amount> profileTimer = {0,0,0,0,0,0,0,0,0,0};
    double max_tet_edge_len = max_len * tet_size_limit;
    // int crest_type = 0;
    auto time_start = std::chrono::high_resolution_clock::now();
    if(crest_type >= 0)
    {
        if (!gridRefineRidges(mode, crest_type, max_tet_edge_len, args.curve_network, args.threshold, args.alpha, max_elements, funcNum, implicit_func_ridge, csg_func, grid_mesh, metric_list, profileTimer))
        {
            throw std::runtime_error("ERROR: unsuccessful grid refinement");
        }
    } else {
        if (!gridRefine(mode, args.curve_network, args.threshold, args.alpha, max_elements, funcNum, implicit_func, csg_func, grid_mesh, metric_list, profileTimer))
        {
            throw std::runtime_error("ERROR: unsuccessful grid refinement");
        }
    }
    auto time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = time_end - time_start;
    std::cout << "-------------- adgrid Elapsed time: " << elapsed.count() << " ms\n";

    // save timing records
    save_timings("timings.json",time_label, profileTimer);
    //profiled time(see details in time.h) and profiled number of calls to zero
    for (int i = 0; i < profileTimer.size(); i++){
        timeProfileName time_type = static_cast<timeProfileName>(i);
        std::cout << time_label[i] << ": " << profileTimer[i] << std::endl;
    }
    // save tet metrics
    save_metrics("stats.json", tet_metric_labels, metric_list);
    
    if (args.discretize_later){
        /// save the grid output for discretization tool
        save_mesh_json("grid.json", grid_mesh);
        /// save the grid output for isosurfacing tool
        save_function_json("function_value.json", grid_mesh, metric_list.vertex_func_grad_map, funcNum);
        /// write grid and active tets
        // mtet::save_mesh("tet_grid.msh", grid_mesh);
        // mtet::save_mesh("active_tets.msh", grid_mesh, std::span<mtet::TetId>(metric_list.activeTetId));
    }
    mtet::save_mesh(outdir + "tet_grid.msh", grid_mesh);
    mtet::save_mesh(outdir + "active_tets.msh", grid_mesh, std::span<mtet::TetId>(metric_list.activeTetId));

    // std::cout << " val size " <<  values.size() << std::endl;
    // std::cout << " gradients size " <<  gradients.size() << std::endl;
    // std::cout << " vertices size " <<  vertices.size() << std::endl;
    // volume_dim = 32;
    // new_resolution = {volume_dim, volume_dim, volume_dim};
    // grid_mesh = generate_tet_mesh(new_resolution, expand_bbox_min, expand_bbox_max, grid_mesh::TET5);
    vector<array<double, 3>> vertices;
    vector<array<size_t, 4>> tets;
    get_mesh_data(grid_mesh, vertices, tets);
    std::cout << "vertices size " << vertices.size() << std::endl;
    std::cout << "tets size " << tets.size() << std::endl;
    vector<double> values(vertices.size());
    vector<double> e1_values(vertices.size());
    vector<double> e2_values(vertices.size());
    vector<double> k1_values(vertices.size());
    vector<double> k2_values(vertices.size());
    std::vector<PrincipleCurvature> tet_curvatures; 
    vector<std::array<double, 3>> gradients(vertices.size());
    vector<std::array<double, 3>> t1_vectors(vertices.size());
    int counter = 0;
    constexpr std::array<std::array<size_t, 2>, 6> tet_edges3D = {
        {
            {{0, 1}}, {{0, 2}}, {{0, 3}}, {{1, 2}}, {{1, 3}}, {{2, 3}}
        }
    };
    // std::shared_ptr<Hermite_RBF<double>> rbf_func = std::dynamic_pointer_cast<Hermite_RBF<double>>(functions[0]);

if(crest_type != -1)
{ 
    // std::span<mtet::TetId>(metric_list.activeTetId)
    std::unordered_set<uint64_t> tet_pt_visited;
    std::vector<std::array<double,3>> active_tet_pts;
    std::vector<std::vector<size_t>> active_tet_edges;
    std::vector<double> active_pts;
    std::vector<double> active_normals; 
    for (auto tet_id : metric_list.activeTetId) {
        auto v_data = grid_mesh.get_tet(tet_id);
        size_t active_tet_pt_size = active_tet_pts.size();
        for(const auto& cur_e : tet_edges3D)
        {
            active_tet_edges.push_back({cur_e[0] + active_tet_pt_size, cur_e[1] + active_tet_pt_size});
        }
        for (int i = 0; i < 4; ++i)
        {
            auto vid = value_of(v_data[i]);
            auto coords = grid_mesh.get_vertex(v_data[i]);
            active_tet_pts.push_back({coords[0], coords[1], coords[2]});
            if(tet_pt_visited.find(vid) != tet_pt_visited.end()) continue;
            tet_pt_visited.insert(vid);
            auto curv_data = metric_list.vertex_func_grad_map_ridge[vid];
            
            active_pts.push_back(coords[0]);
            active_pts.push_back(coords[1]);
            active_pts.push_back(coords[2]);
            active_normals.push_back(curv_data[0].t1_[0]);
            active_normals.push_back(curv_data[0].t1_[1]);
            active_normals.push_back(curv_data[0].t1_[2]);
        }
    }   
    std::string pt_k_dir = outdir + "tet_active_pts_k1_dir.xyz";
    writeXYZnormal(pt_k_dir, active_pts, active_normals);
    std::string tet_edges_save_path = outdir + "tet_active_edges.obj";
    VIPSSRidges::SaveRidgesToObj(tet_edges_save_path, 
        active_tet_pts, active_tet_edges, 1.0, {0, 0, 0});
    vector<CurvatureData<double>> tet_pt_curvature_data;
    get_function_curvature_data(grid_mesh, metric_list.vertex_func_grad_map_ridge, tet_pt_curvature_data);

    // auto gaussian_func = std::dynamic_pointer_cast<ImplicitFunctionRidge<double>>(functions[0]);

    // for(auto pt : gaussian_func->control_points_)
    // {
    //     CurvatureData<double> curv_data;
    //     gaussian_func->EvaluateCurvatureData(pt[0], pt[1], pt[2], curv_data);
    //     std::cout << " input pt val : " <<  curv_data.f_val_ << std::endl;
    // }
    
    std::cout << " finish get  get_function_curvature_data " << tet_pt_curvature_data.size() << std::endl;
    grid_mesh.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3> data){
        // llvm_vecsmall::SmallVector<std::array<double, 4>, 20> func_gradList(1);
        
            // size_t id = value_of(vid); 
            // std::cout << " id 000 " << id << std::endl;
            CurvatureData<double> curv_data = tet_pt_curvature_data[counter];
            // CurvatureData<double> curv_data;
            // gaussian_func->EvaluateCurvatureData(data[0], data[1], data[2], curv_data);
            // rbf_func->EvaluateCurvatureData(data[0], data[1], data[2], curv_data);
            e1_values[counter] = curv_data.e1_;
            // e1_values[counter] = curv_data.f_gradient_.norm();
            e2_values[counter] = curv_data.e2_;
            k1_values[counter] = curv_data.k1_;
            k2_values[counter] = curv_data.k2_;
            values[counter] = curv_data.f_val_;
            t1_vectors[counter] = {curv_data.t1_[0], curv_data.t1_[1], curv_data.t1_[2]};
            gradients[counter] = {curv_data.f_gradient_[0], curv_data.f_gradient_[1], curv_data.f_gradient_[2]};
            PrincipleCurvature pt_curv;
            pt_curv.emax_ = curv_data.e1_;
            pt_curv.emin_ = curv_data.e2_;
            pt_curv.tmax_ = {curv_data.t1_[0], curv_data.t1_[1], curv_data.t1_[2]};
            pt_curv.tmin_ = {curv_data.t2_[0], curv_data.t2_[1], curv_data.t2_[2]};
            pt_curv.kmax_ = curv_data.k1_;
            pt_curv.kmin_ = curv_data.k2_;
            // pt_curv.emax_ = curv_data.f_gradient_.norm();
            // pt_curv.tmax_ = {curv_data.f_gradient_[0], curv_data.f_gradient_[1], curv_data.f_gradient_[2]};
            // pt_curv.emin_ = curv_data.e2_;
            // pt_curv.emax_ = curv_data.f_gradient_.dot(curv_data.t1_);
            // pt_curv.emax_ = curv_data.f_gradient_.norm();
            // pt_curv.kmax_ = 100; 
            // pt_curv.kmin_ = 0;
            
            pt_curv.de1_  = {curv_data.e1_d1_[0], curv_data.e1_d1_[1], curv_data.e1_d1_[2]};
            pt_curv.de2_  = {curv_data.e2_d2_[0], curv_data.e2_d2_[1], curv_data.e2_d2_[2]};
            pt_curv.emax_prime_ = curv_data.e1_prime_;
            pt_curv.emin_prime_ = curv_data.e2_prime_;
            tet_curvatures.push_back(pt_curv);
            // values[id] = func->evaluate_gradient(data[0], data[1], data[2], gradients[id][0], gradients[id][1], gradients[id][2]);           
        counter ++;   
         
    });
}

    // CurvatureData<double> curv_data;
    // std::vector<std::array<double,3>> test_pts(5); 
    // test_pts[0] = {0.041489, -1.607394, 0.108060};
    // test_pts[1] = {-1.279732, -0.814506, -1.195449};
    // test_pts[2] = {1.743549, 0.824388, 1.669528};
    // test_pts[3] = {1.580017, 0.868936, 1.935822 }; 
    // for(int i = 0; i < 4; ++i)
    // {
    //     rbf_func->EvaluateCurvatureData(test_pts[i][0], test_pts[i][1], test_pts[i][2], curv_data);
    //     std::cout << "test pt id "  << i <<  ",  k1 : " << curv_data.k1_ << ", k2 : " <<  curv_data.k2_ << " , e1: " << curv_data.e1_ 
    // << ", e2 : " << curv_data.e2_ 
    // << ", d e1 : " << curv_data.e1_d1_[0] << " " << curv_data.e1_d1_[1] << " "<< curv_data.e1_d1_[2] 
    // << ", d e2 : " << curv_data.e2_d2_[0] << " " << curv_data.e2_d2_[1] << " "<< curv_data.e2_d2_[2] << std::endl; 

    // Hermite_RBF<double>::hrbf_ptr_->EvaluateCurvatureData(test_pts[i][0], test_pts[i][1], test_pts[i][2], curv_data);
    //     std::cout << "test pt id2 "  << i <<  ",  k1 : " << curv_data.k1_ << ", k2 : " <<  curv_data.k2_ << " , e1: " << curv_data.e1_ 
    // << ", e2 : " << curv_data.e2_ 
    // << ", d e1 : " << curv_data.e1_d1_[0] << " " << curv_data.e1_d1_[1] << " "<< curv_data.e1_d1_[2] 
    // << ", d e2 : " << curv_data.e2_d2_[0] << " " << curv_data.e2_d2_[1] << " "<< curv_data.e2_d2_[2] << std::endl; 


    // }
    // std::string planck_sample_path = "/home/jjxia/Documents/prejects/NN-VIPSS/data/planck_sample_pts3.xyz";
    // std::vector<double> sample_pts;
    // readXYZ(planck_sample_path, sample_pts);
    // for(size_t i = 0; i < sample_pts.size()/3; ++i)
    // {
    //     CurvatureData<double> curv_data;
    //     rbf_func->EvaluateCurvatureData(sample_pts[3*i], sample_pts[3*i + 1], sample_pts[3*i + 2], curv_data);
    //     std::cout << "k2 " << curv_data.k2_ << " , e2 " << curv_data.e2_ << " de2 " << curv_data.e2_d2_ << std::endl;
    // }

    
    
    // get_function_val_and_gradients_from_funcs(vertices, functions[0], values, gradients);
    // marching3D::MarchingTet3D( vertices, tets, values, gradients, output_vertices, output_triangles);
    // marching3D::MarchingTet3DEdges( vertices, tets, values, gradients, output_vertices, output_triangles);
    if(crest_type != -1)
    {
    std::string unoriented_tets_path = outdir + "unoriented_tets_all_" + filename+ ".obj";
    std::string unoriented_tets_unsplit_path = outdir + "unoriented_tets_unsplit_" + filename +".obj";
    std::string unoriented_tets_split_path = outdir + "unoriented_tets_split" + filename +".obj";
    OutTetObj::unoriented_tets.SaveTetDataToObj(unoriented_tets_path);
    OutTetObj::unoriented_tets_split.SaveTetDataToObj(unoriented_tets_split_path);
    OutTetObj::unoriented_tets_unsplit.SaveTetDataToObj(unoriented_tets_unsplit_path);

    std::string threshold_tets_path = outdir + "threshold_tets_" + filename +".obj";
    OutTetObj::threshold_tets.SaveTetDataToObj(threshold_tets_path);

    
    std::string limit_orientable_tets_path = outdir + "limit_orientable_tets_" + filename +".obj";
    std::string rv_fail_tets_path = outdir + "rv_fail_tets_" + filename +".obj";
    std::string eprime_fail_tets_path = outdir + "eprime_fail_tets" +".obj";
    
    
    std::string boundary1_tets_path = outdir + "boundary1_unorientable_tets"  +".obj";
    std::string boundary2_tets_path = outdir + "boundary2_RV_tets"  +".obj";
    std::string boundary3_tets_path = outdir + "boundary3_eprime_tets"  +".obj";
    OutTetObj::boundary1_unorientable_tets .SaveTetDataToObj(boundary1_tets_path);
    OutTetObj::boundary2_RV_tets.SaveTetDataToObj(boundary2_tets_path);
    OutTetObj::boundary3_eprime_tets.SaveTetDataToObj(boundary3_tets_path);
    OutTetObj::limit_orientable_tets.SaveTetDataToObj(limit_orientable_tets_path);
    OutTetObj::rv_failed_tets.SaveTetDataToObj(rv_fail_tets_path);
    OutTetObj::eprime_failed_tets.SaveTetDataToObj(eprime_fail_tets_path);
        
    }

    if(crest_type == -1)
    {
        get_function_val_and_gradients(grid_mesh, metric_list.vertex_func_grad_map, values, gradients);
        marching3D::MarchingTet3D(vertices, tets, values, gradients, output_vertices, output_triangles);

    } else {

        // marching3D::MarchingTet3D(vertices, tets, values, gradients, output_vertices, output_triangles);
        std::cout << " start MarchingTet3DCrestMesh ...... "<< std::endl;
        marching3D::MarchingTet3DCrestMesh(vertices, tets, tet_curvatures, outdir); 
        std::cout << " finish MarchingTet3DCrestMesh ...... "<< std::endl;
    }

    if(crest_type == 0)
    {
        std::string tet_vals_path = outdir + filename + "_tet_vals.ply";
        SaveTetMeshToPly(vertices, tets, values, tet_vals_path);
        std::string tet_grads_path = outdir + filename + "_tet_grads.xyz";
        writeXYZnormal(tet_grads_path, vertices, gradients);
        std::string tet_e1_path = outdir + filename + "_tet_e1.ply";
        SaveTetMeshToPly(vertices, tets, e1_values, tet_e1_path);
        std::string tet_k1_path = outdir + filename + "_tet_k1.ply";
        SaveTetMeshToPly(vertices, tets, k1_values, tet_k1_path);
        std::string tet_t1_path = outdir + filename + "_tet_t1.xyz";
        writeXYZnormal(tet_t1_path, vertices, t1_vectors);
    } else {
        std::string tet_e2_path = outdir + filename + "_tet_e2.ply";
        SaveTetMeshToPly(vertices, tets, e2_values, tet_e2_path);
        std::string tet_k2_path = outdir + filename + "_tet_k2.ply";  
        SaveTetMeshToPly(vertices, tets, k2_values, tet_k2_path);
    }
    
    // return 0;
}

