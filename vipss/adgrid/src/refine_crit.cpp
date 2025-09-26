//
//  subdivide_multi.cpp
//  adaptive_mesh_refinement
//
//  Created by Yiwen Ju on 6/20/24.
//
#include <iostream>
#include "implicit_functions/implicit_functions.h"
#include "refine_crit.h"

///Stores the 2d and 3d origin for the convex hull check happened in zero-crossing criteria.
std::array<double, 2> query_2d = {0.0, 0.0}; // X, Y
std::array<double, 3> query_3d = {0.0, 0.0, 0.0}; // X, Y, Z

OutTetObj OutTetObj::unoriented_tets;
OutTetObj OutTetObj::unoriented_tets_split;
OutTetObj OutTetObj::unoriented_tets_unsplit;
// OutTetObj OutTetObj::unoriented_tets_unsplit;
OutTetObj OutTetObj::limit_unorientable_tets; 
OutTetObj OutTetObj::limit_orientable_tets;
OutTetObj OutTetObj::rv_failed_tets;
OutTetObj OutTetObj::threshold_tets;

std::unordered_map<std::string, std::array<std::array<double,4>,2>> tet_edge_mid_ridge_curv_map;
std::unordered_map<std::string, std::array<double,4>> tet_face_mid_ridge_curv_map;  
std::unordered_map<std::string, std::array<std::array<double,4>,2>> tet_edge_mid_valle_curv_map;
std::unordered_map<std::string, std::array<double,4>> tet_face_mid_valle_curv_map; 

// std::unordered_map<std::string, std::array<double,4>> tet_edge_mid_ridge_curv_map_simple;
// std::unordered_map<std::string, std::array<double,4>> tet_edge_mid_valle_curv_map_simple;

std::unordered_map<std::string, std::array<double,8>> tet_edge_mid_curv_map;

std::unordered_map<std::string, std::array<double,8>>& GetTetEdgeMidSampleMap()
{
    return tet_edge_mid_curv_map;
}

inline std::string generate_tet_edge_key(uint64_t a, uint64_t b)
{
    if(a <= b)
    {
        return std::to_string(a) + "_" + std::to_string(b);
    }
    return std::to_string(b) + "_" + std::to_string(a);
}

inline std::string generate_tet_edge_key( std::vector<uint64_t>& pids)
{
    std::sort(pids.begin(), pids.end());
    std::string key = std::to_string(pids[0]);
    for(int i = 1; i < pids.size(); ++i)
    {
        key += ("_" + std::to_string(pids[i]));
    }
    return key;
    
}

constexpr std::array<std::array<int, 2>, 6> CRIT_tet_edges3D = {
        {
            {{0, 1}}, {{0, 2}}, {{0, 3}}, {{1, 2}}, {{1, 3}}, {{2, 3}}
        }
    };


constexpr std::array<std::array<int, 3>, 6> CRIT_tet_faces3D = {
        {
            {{0, 1, 2}}, {{0, 2, 3}}, {{0, 3, 1}}, {{1, 3 , 2}}
        }
    };

constexpr std::array<int,4> CRIT_tet_vt_ids = {3, 1, 2, 0};
// OutTetObj unoriented_tets ;
// OutTetObj unoriented_tets_split;
// OutTetObj unoriented_tets_unsplit;

void OutTetObj::SaveTetDataToObj(const std::string& out_path)
{
    std::ofstream objFile(out_path);
    if (!objFile) {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return;
    }
    // Write vertices
    for (const auto& point : pts_) {
        double px = point[0] ;
        double py = point[1] ;
        double pz = point[2] ;
        objFile << "v " << px << " " << py << " " << pz << "\n";
    }
    // Write edges as line elements (OBJ uses 'l' for lines)
    // for (const auto& edge : edges_) {
    //     objFile << "l " << edge[0] + 1 << " " << edge[1] + 1 << "\n";  // OBJ uses 1-based indexing
    // }
    for (const auto& face : faces_) {
        objFile << "f " << face[0] + 1 << " " << face[1] + 1 << " " << face[2] + 1 << "\n";  // OBJ uses 1-based indexing
    }
    objFile.close();
    std::cout << "OBJ file saved successfully: " << out_path << std::endl;
};

void OutTetObj::AddNewTet(const Eigen::Matrix<double, 4, 3> &pts)
{
    size_t current_size = this->pts_.size();
    for(int i = 0; i < 4; ++i)
    {
        pts_.push_back({pts(i,0), pts(i,1), pts(i,2)});
    }
    
    // for(const auto &edge : CRIT_tet_edges3D)
    // {
    //     // edges_.push_back({edge[0] + int(current_size), edge[1] + int(current_size)});
    //     edges_.push_back({edge[0] + int(current_size), edge[1] + int(current_size)});
    // }

    for(const auto &face: CRIT_tet_faces3D)
    {
        faces_.push_back({face[0] + int(current_size), face[1] + int(current_size), face[2] + int(current_size)});
    }
}


bool TetCrestTypeTest(const Eigen::RowVector4d& kmax_vals, const Eigen::RowVector4d& kmin_vals, const int crest_type)
{
    if(crest_type == 0)
    {
        for(int pid = 0; pid < 4; ++ pid)
        {
            if(abs(kmax_vals[pid]) > abs(kmin_vals[pid]) && kmax_vals[pid] > 0 )
            {
                return true;
            }
        }
    } else {

        for(int pid = 0; pid < 4; ++ pid)
        {
            // std::cout << " start kk compare ..." << pid << std::endl;
            if(abs(kmax_vals[pid]) < abs(kmin_vals[pid]) && kmin_vals[pid] < 0)
            {
                return true;
            }
        }
    }
    
    return false;
}

bool TetCrestTypeTest2(const Eigen::RowVector4d& kmax_vals, const Eigen::RowVector4d& kmin_vals, const int crest_type, int& count)
{
    count = 0;
    if(crest_type == 0)
    {
        for(int pid = 0; pid < 4; ++ pid)
        {
            if(abs(kmax_vals[pid]) > abs(kmin_vals[pid]) && kmax_vals[pid] > 0)
            {
                count ++;
            }
        }
    } else {

        for(int pid = 0; pid < 4; ++ pid)
        {
            // std::cout << " start kk compare ..." << pid << std::endl;
            if(abs(kmax_vals[pid]) < abs(kmin_vals[pid]) && kmax_vals[pid] < 0 )
            {
                count ++;
            }
        }
    }
    if(count > 0)
    {
        return true;
    }
    return false;
}

inline bool TetPerEdgeZeroCrossingTest( const Eigen::Matrix<double, 4, 3>& k_dirs,
        const Eigen::RowVector4d& e_vals)
{
    for(int e_id = 0; e_id < 6; ++e_id)
    {
        auto edge = CRIT_tet_edges3D[e_id];
        Eigen::Vector3d k1_dir_a = k_dirs.row(edge[0]);
        Eigen::Vector3d k1_dir_b = k_dirs.row(edge[1]);
        double e_a = e_vals[edge[0]];
        double e_b = e_vals[edge[1]];
        if(k1_dir_a.dot(k1_dir_b) < 0) e_b *= -1;
    
        if(e_a * e_b <= 0) return true; 
    }
    return false;
}



/// Below are the local functions servicing `critIA` , `critCSG`, and `critMI`

/// returns a `bool` value that `true` represents positive and `false` represents negative of the input value `x`.
inline bool get_sign(double x) {
    return x > 0;
}

/// Construct the values of one function at the bezier control points within a tet.
///
/// @param[in] vals         The function values at four tet vertices
/// @param[in] grads            The total derivative of the functions in x, y, z direction at four tet vertices.
/// @param[in] vec          sampled vectors using four tet vertices. Given tet vertices as p0, p1, p2, p3. These 6 vectors are p1 - p0, p2 - p0, p3 - p0, p2 - p1, p3 - p1, p3 - p2.
///
/// @return         The eigen vector of 20 bezier values.
Eigen::Vector<double, 20> bezierConstruct(const Eigen::RowVector4d vals,
                                                const Eigen::Matrix<double, 4, 3> grads,
                                                const Eigen::Matrix<double, 3, 6> vec)
{
    Eigen::RowVector3d v0s, v1s, v2s, v3s;
    v0s = grads.row(0) * vec(Eigen::all, {0, 1, 2}) / 3;
    v0s.array() += vals(0);
    v1s =  grads.row(1) * vec(Eigen::all, {3, 4, 0}) / 3;
    v1s = v1s.asDiagonal() * Eigen::Vector3d({1, 1, -1});
    v1s.array() += vals(1);
    v2s = grads.row(2) * vec(Eigen::all, {5, 1, 3}) / 3;
    v2s = v2s.asDiagonal() * Eigen::Vector3d({1, -1, -1});
    v2s.array() += vals(2);
    v3s = grads.row(3) * vec(Eigen::all, {2, 4, 5}) / 3;
    v3s *= -1;
    v3s.array() += vals(3);
    
    double vMid0 = (9 * (v1s(0) + v1s(1) + v2s(0) + v2s(2) + v3s(1) + v3s(2)) / 6 - vals(1) - vals(2) - vals(3))/ 6;
    double vMid1 =(9 * (v0s[1] + v0s[2] + v2s[0] + v2s[1] + v3s[0] + v3s[2]) / 6 - vals(0) - vals(2) - vals(3))/ 6;
    double vMid2 =(9 * (v0s[0] + v0s[2] + v1s[1] + v1s[2] + v3s[0] + v3s[1]) / 6 - vals(0) - vals(1) - vals(3))/ 6;
    double vMid3 =(9 * (v0s[0] + v0s[1] + v1s[0] + v1s[2] + v2s[1] + v2s[2]) / 6 - vals(0) - vals(1) - vals(2))/ 6;
    Eigen::RowVector<double, 20> valList;
    valList << vals, v0s, v1s, v2s, v3s, vMid0, vMid1, vMid2, vMid3;
    return valList;
}
inline bool CheckTetOrientable(Eigen::RowVector4d& vals,
                        Eigen::Matrix<double, 4, 3>& k_dirs)
{
    std::vector<Eigen::Vector3d> dirs(4);
    dirs[0] = k_dirs.row(0);
    for(int i = 1; i < 4; ++i)
    {
        dirs[i] = k_dirs.row(i);
        if(dirs[0].dot(dirs[i]) < 0)
        {
            vals[i] *= -1;
            // grads.row(i) *= -1;
            dirs[i] *= -1;
            k_dirs.row(i) *= -1.0;
        }
    }
    bool is_k_dir_consistent = true;
    if(dirs[1].dot(dirs[2]) < 0 || dirs[1].dot(dirs[3]) < 0 || dirs[2].dot(dirs[3]) < 0)
    {
        // std::cout << " not a consistent oriented tet !" << std::endl;
        is_k_dir_consistent = false;
    } 
    return is_k_dir_consistent;
}

/// Construct the values of one function at the bezier control points within a tet.
///
/// @param[in] vals         The function values at four tet vertices
/// @param[in] grads            The total derivative of the functions in x, y, z direction at four tet vertices.
/// @param[in] vec          sampled vectors using four tet vertices. Given tet vertices as p0, p1, p2, p3. These 6 vectors are p1 - p0, p2 - p0, p3 - p0, p2 - p1, p3 - p1, p3 - p2.
///
/// @return         The eigen vector of 20 bezier values.
Eigen::Vector<double, 20> bezierConstructRidge( Eigen::RowVector4d& vals,
                                                Eigen::Matrix<double, 4, 3>& k_dirs,
                                                Eigen::Matrix<double, 4, 3>& grads,
                                                const Eigen::Matrix<double, 3, 6>& vec,
                                                bool& is_k_dir_consistent)
{
    std::vector<Eigen::Vector3d> dirs(4);
    dirs[0] = k_dirs.row(0);
    for(int i = 1; i < 4; ++i)
    {
        dirs[i] = k_dirs.row(i);
        if(dirs[0].dot(dirs[i]) < 0)
        {
            vals[i] *= -1;
            grads.row(i) *= -1;
            dirs[i] *= -1;
        }
    }
    is_k_dir_consistent = true;
    if(dirs[1].dot(dirs[2]) < 0 || dirs[1].dot(dirs[3]) < 0 || dirs[2].dot(dirs[3]) < 0)
    {
        // std::cout << " not a consistent oriented tet !" << std::endl;
        is_k_dir_consistent = false;
    } 

    Eigen::RowVector3d v0s, v1s, v2s, v3s;
    v0s = grads.row(0) * vec(Eigen::all, {0, 1, 2}) / 3;
    v0s.array() += vals(0);
    v1s =  grads.row(1) * vec(Eigen::all, {3, 4, 0}) / 3;
    v1s = v1s.asDiagonal() * Eigen::Vector3d({1, 1, -1});
    v1s.array() += vals(1);
    v2s = grads.row(2) * vec(Eigen::all, {5, 1, 3}) / 3;
    v2s = v2s.asDiagonal() * Eigen::Vector3d({1, -1, -1});
    v2s.array() += vals(2);
    v3s = grads.row(3) * vec(Eigen::all, {2, 4, 5}) / 3;
    v3s *= -1;
    v3s.array() += vals(3);
    
    double vMid0 =(9 * (v1s(0) + v1s(1) + v2s(0) + v2s(2) + v3s(1) + v3s(2)) / 6 - vals(1) - vals(2) - vals(3))/ 6;
    double vMid1 =(9 * (v0s[1] + v0s[2] + v2s[0] + v2s[1] + v3s[0] + v3s[2]) / 6 - vals(0) - vals(2) - vals(3))/ 6;
    double vMid2 =(9 * (v0s[0] + v0s[2] + v1s[1] + v1s[2] + v3s[0] + v3s[1]) / 6 - vals(0) - vals(1) - vals(3))/ 6;
    double vMid3 =(9 * (v0s[0] + v0s[1] + v1s[0] + v1s[2] + v2s[1] + v2s[2]) / 6 - vals(0) - vals(1) - vals(2))/ 6;
    Eigen::RowVector<double, 20> valList;
    valList << vals, v0s, v1s, v2s, v3s, vMid0, vMid1, vMid2, vMid3;
    return valList;
}

Eigen::Vector<double, 20> bezierSampleVals(const Eigen::Matrix<double, 4, 3> &pts, 
                                         const Eigen::RowVector4d& vals, 
                                         const Eigen::Matrix<double, 4, 3>& k_dirs,
                                         const int crest_type)
{
    // std::ofstream out("output_tet_pts.xyz");
    Eigen::RowVector<double, 20> valList;
    valList.head(4) = vals;
    // for(int i = 0; i < 4; ++i)
    // {
    //     valList[i] = vals[i];
    // }

    const Eigen::Matrix<double, 16, 4> linear_coeff {{2, 1, 0, 0}, {2, 0, 1, 0}, {2, 0, 0, 1}, {0, 2, 1, 0},{0, 2, 0, 1}, {1, 2, 0, 0}, {0, 0, 2, 1}, {1, 0, 2, 0},{0, 1, 2, 0}, {1, 0, 0, 2}, {0, 1, 0, 2}, {0, 0, 1, 2},{0, 1, 1, 1}, {1, 0, 1, 1}, {1, 1, 0, 1}, {1, 1, 1, 0}};
    Eigen::Vector<double, 16> linear_val_x = (linear_coeff * pts.col(0)) / 3;
    Eigen::Vector<double, 16> linear_val_y = (linear_coeff * pts.col(1)) / 3;
    Eigen::Vector<double, 16> linear_val_z = (linear_coeff * pts.col(2)) / 3;
    // std::cout << " start to calculate vallist for edge mid  points...... " << std::endl; 
    #pragma omp parallel for 
    for(int i = 0; i < 16; ++i)
    {
        CurvatureData<double> cur_data;
        Hermite_RBF<double>::hrbf_ptr_->EvaluateCurvatureData(linear_val_x[i], linear_val_y[i], linear_val_z[i],  cur_data);
        if(crest_type == 0)
        {
            valList[4 + i] = cur_data.e1_;
            const auto& kdir = cur_data.t1_;
            if(k_dirs.row(0).dot(kdir) < 0)
            {
                valList[4 + i] *= -1.0;
            }
        } else {
            valList[4 + i] = cur_data.e2_;
            const auto& kdir = cur_data.t2_;
            if(k_dirs.row(0).dot(kdir) < 0)
            {
                valList[4 + i] *= -1.0;
            }
        }
    }
    // valList << vals, emid_valList, mid_vals;  
    // out.close();
    // std::cout << " finished assigned ." << std::endl;
    return valList;
}

std::vector<std::array<double,3>> CalEdgeInterPts(const std::array<double,3>& pa, const std::array<double,3>& pb)
{
    double dx = pb[0] - pa[0];
    double dy = pb[1] - pa[1];
    double dz = pb[2] - pa[2];

    double int_a_x = pa[0] + dx /3;
    double int_a_y = pa[1] + dy /3;
    double int_a_z = pa[2] + dz /3;

    double int_b_x = pa[0] + dx /3 * 2;
    double int_b_y = pa[1] + dy /3 * 2;
    double int_b_z = pa[2] + dz /3 * 2;

    return {{int_a_x, int_a_y, int_a_z}, {int_b_x, int_b_y, int_b_z}};
} 


double  bezierSampleValsWithHash(
                                        const mtet::TetId tid,
                                        mtet::MTetMesh &grid_mesh,
                                        const Eigen::Matrix<double, 4, 3> &pts, 
                                        const Eigen::RowVector4d& vals, 
                                        const Eigen::Matrix<double, 4, 3>& k_dirs,
                                        const int crest_type)
{
    auto vs = grid_mesh.get_tet(tid);
    std::array<uint64_t,4> tet_pids;
    std::array<std::array<double,3>, 4> tet_pts_coords;
    for (int i = 0; i < 4; ++i)
    {
        auto vid = vs[i];
        tet_pids[i] = vid.value_of();
        auto coords = grid_mesh.get_vertex(vid);
        tet_pts_coords[i] = {coords[0], coords[1], coords[2]};
    }
    Eigen::RowVector<double, 16> sample_vals;
    Eigen::RowVector<double, 16> linear_vals;
    Eigen::Vector3d k_dir_main = k_dirs.row(0);
    auto& tet_edge_curv_val_map = crest_type == 0 ? tet_edge_mid_ridge_curv_map : tet_edge_mid_valle_curv_map;

    for(int eid = 0; eid < CRIT_tet_edges3D.size(); ++eid)
    {   
        const auto& edge = CRIT_tet_edges3D[eid];
        uint64_t pa_id = tet_pids[edge[0]];
        uint64_t pb_id = tet_pids[edge[1]];
        
        std::array<double,3> pa = tet_pts_coords[edge[0]];
        std::array<double,3> pb = tet_pts_coords[edge[1]];
        double val_a = vals[edge[0]];
        double val_b = vals[edge[1]];
        if(pa_id > pb_id)
        {
            pa_id = pb_id;
            pb_id = tet_pids[edge[0]]; 
            pa = pb;
            pb = tet_pts_coords[edge[0]];
            val_a = val_b;
            val_b = vals[edge[0]];
        }
        linear_vals[2*eid]     = val_a + (val_b - val_a) / 3;
        linear_vals[2*eid + 1] = val_a + (val_b - val_a) / 3 * 2;
        auto e_key = generate_tet_edge_key(pa_id, pb_id);
        std::array<std::array<double,4>,2> edge_int_cur_vals;
        if(tet_edge_curv_val_map.find(e_key) == tet_edge_curv_val_map.end())
        {
            auto inter_pts = CalEdgeInterPts(pa, pb);
            for(int pid = 0; pid < 2; ++pid)
            {
                CurvatureData<double> cur_data;
                const auto& pt = inter_pts[pid];
                Hermite_RBF<double>::hrbf_ptr_->EvaluateCurvatureData(pt[0], pt[1], pt[2], cur_data);
                std::array<double,4> curv_vals;
                if(crest_type == 0)
                {
                    if(cur_data.t1_.dot(k_dir_main) < 0)
                    {
                        cur_data.t1_ *= -1;
                        cur_data.e1_ *= -1;
                    }
                    curv_vals = {cur_data.e1_, cur_data.t1_[0], cur_data.t1_[1], cur_data.t1_[2]};
                    edge_int_cur_vals[pid] = curv_vals;
                } else {
                    if(cur_data.t2_.dot(k_dir_main) < 0)
                    {
                        cur_data.t2_ *= -1;
                        cur_data.e2_ *= -1;
                    }
                    curv_vals = {cur_data.e2_, cur_data.t2_[0], cur_data.t2_[1], cur_data.t2_[2]};
                    edge_int_cur_vals[pid] = curv_vals;
                } 
            }
            tet_edge_curv_val_map[e_key] = edge_int_cur_vals;
        }  else {
            edge_int_cur_vals = tet_edge_curv_val_map[e_key];
            for(int pid = 0; pid < 2; ++pid)
            {
                Eigen::Vector3d cur_k_dir = {edge_int_cur_vals[pid][1], edge_int_cur_vals[pid][2], edge_int_cur_vals[pid][3]};
                if(cur_k_dir.dot(k_dir_main) < 0)
                {
                    edge_int_cur_vals[pid][0] *= -1;
                    edge_int_cur_vals[pid][1] *= -1;
                    edge_int_cur_vals[pid][2] *= -1;
                    edge_int_cur_vals[pid][3] *= -1;
                }
            }
        }
        sample_vals[2*eid]     = edge_int_cur_vals[0][0];
        sample_vals[2*eid + 1] = edge_int_cur_vals[1][0];
    }

    auto& tet_face_curv_val_map = crest_type == 0 ? tet_face_mid_ridge_curv_map : tet_face_mid_valle_curv_map;
    for(int fid = 0; fid < 4; ++fid)
    {   
        const auto& face = CRIT_tet_faces3D[fid];
        uint64_t pa_id = tet_pids[face[0]];
        uint64_t pb_id = tet_pids[face[1]];
        uint64_t pc_id = tet_pids[face[2]];
        linear_vals[12 + fid] = (vals[face[0]] + vals[face[1]] + vals[face[2]]) / 3;
        double f_mid_pt_x =  (tet_pts_coords[face[0]][0] +  tet_pts_coords[face[1]][0] + tet_pts_coords[face[2]][0]) / 3;
        double f_mid_pt_y =  (tet_pts_coords[face[0]][1] +  tet_pts_coords[face[1]][1] + tet_pts_coords[face[2]][1]) / 3;
        double f_mid_pt_z =  (tet_pts_coords[face[0]][2] +  tet_pts_coords[face[1]][2] + tet_pts_coords[face[2]][2]) / 3;

        std::vector<uint64_t> face_pids = {pa_id, pb_id, pc_id};
        std::string f_token = generate_tet_edge_key(face_pids);
        if(tet_face_curv_val_map.find(f_token) == tet_face_curv_val_map.end())
        {
            CurvatureData<double> cur_data;
            Hermite_RBF<double>::hrbf_ptr_->EvaluateCurvatureData(f_mid_pt_x, f_mid_pt_y, f_mid_pt_z, cur_data);
            double cur_e_val = crest_type == 0? cur_data.e1_ : cur_data.e2_;
            auto cur_k_dir = crest_type == 0? cur_data.t1_ : cur_data.t2_;

            sample_vals[12 + fid] = k_dir_main.dot(cur_k_dir) < 0 ? -cur_e_val : cur_e_val;
            tet_face_curv_val_map[f_token] = {cur_e_val, cur_k_dir[0], cur_k_dir[1], cur_k_dir[2]};

        } else {
            auto vals = tet_face_curv_val_map[f_token];
            Eigen::RowVector3d cur_k_dir = {vals[1], vals[2], vals[3]};
            sample_vals[12 + fid] = k_dir_main.dot(cur_k_dir) < 0 ? -vals[0] : vals[0];
        }
    }

    Eigen::RowVector<double, 16> diff = sample_vals - linear_vals;
    double max_diff = std::max(diff.maxCoeff(), - diff.minCoeff()); 
    // std::cout << " val list 1 : " << sample_vals << std::endl;
    return max_diff;
}


double  bezierSampleValsWithHashSimple(
                                        const mtet::TetId tid,
                                        mtet::MTetMesh &grid_mesh,
                                        const Eigen::Matrix<double, 4, 3> &pts, 
                                        const Eigen::RowVector4d& vals, 
                                        const Eigen::Matrix<double, 4, 3>& k_dirs,
                                        const int crest_type)
{
    auto vs = grid_mesh.get_tet(tid);
    std::array<uint64_t,4> tet_pids;
    std::array<std::array<double,3>, 4> tet_pts_coords;
    for (int i = 0; i < 4; ++i)
    {
        auto vid = vs[i];
        tet_pids[i] = vid.value_of();
        auto coords = grid_mesh.get_vertex(vid);
        tet_pts_coords[i] = {coords[0], coords[1], coords[2]};
    }
    Eigen::RowVector<double, 6> sample_vals;
    Eigen::RowVector<double, 6> linear_vals;
    Eigen::Vector3d k_dir_main = k_dirs.row(0);
    // auto& tet_edge_curv_val_map = crest_type == 0 ? tet_edge_mid_ridge_curv_map_simple : tet_edge_mid_valle_curv_map_simple;
        
    
    int e_num = int(CRIT_tet_edges3D.size());
    std::vector<std::string> edge_keys(6);
    std::vector<std::array<double,8>> edge_mid_e_t_vals(6);
    #pragma omp parallel for
    for(int eid = 0; eid < e_num; ++eid)
    {   
        const auto& edge = CRIT_tet_edges3D[eid];
        uint64_t pa_id = tet_pids[edge[0]];
        uint64_t pb_id = tet_pids[edge[1]];
        
        std::array<double,3> pa = tet_pts_coords[edge[0]];
        std::array<double,3> pb = tet_pts_coords[edge[1]];
        double val_a = vals[edge[0]];
        double val_b = vals[edge[1]];
        if(pa_id > pb_id)
        {
            pa_id = pb_id;
            pb_id = tet_pids[edge[0]]; 
            pa = pb;
            pb = tet_pts_coords[edge[0]];
            val_a = val_b;
            val_b = vals[edge[0]];
        }
        linear_vals[eid] = (val_a + val_b) / 2;
        auto e_key = generate_tet_edge_key(pa_id, pb_id);
        edge_keys[eid] = e_key;
        if(tet_edge_mid_curv_map.find(e_key) == tet_edge_mid_curv_map.end())
        {
            double inter_pt_x = (pa[0] + pb[0])/2;
            double inter_pt_y = (pa[1] + pb[1])/2;
            double inter_pt_z = (pa[2] + pb[2])/2;
            CurvatureData<double> cur_data;
            Hermite_RBF<double>::hrbf_ptr_->EvaluateCurvatureData(inter_pt_x, inter_pt_y, inter_pt_z, cur_data);
            
            if(crest_type == 0)
            {
                if(cur_data.t1_.dot(k_dir_main) < 0)
                {
                    cur_data.t1_ *= -1;
                    cur_data.e1_ *= -1;
                }
            } else {
                if(cur_data.t2_.dot(k_dir_main) < 0)
                {
                    cur_data.t2_ *= -1;
                    cur_data.e2_ *= -1;
                }
            } 
            edge_mid_e_t_vals[eid] = {cur_data.e1_, cur_data.t1_[0], 
                                            cur_data.t1_[1], cur_data.t1_[2],
                                            cur_data.e2_, cur_data.t2_[0], 
                                            cur_data.t2_[1], cur_data.t2_[2]};
            sample_vals[eid] = crest_type == 0 ?  edge_mid_e_t_vals[eid][0] : edge_mid_e_t_vals[eid][4];
        }  else {
            edge_mid_e_t_vals[eid] = tet_edge_mid_curv_map[e_key];
            Eigen::Vector3d cur_kmax_dir = {edge_mid_e_t_vals[eid][1], edge_mid_e_t_vals[eid][2], edge_mid_e_t_vals[eid][3]};
            Eigen::Vector3d cur_kmin_dir = {edge_mid_e_t_vals[eid][5], edge_mid_e_t_vals[eid][6], edge_mid_e_t_vals[eid][7]};
            auto cur_k_dir = crest_type == 0? cur_kmax_dir : cur_kmin_dir;
            sample_vals[eid] = crest_type == 0?  edge_mid_e_t_vals[eid][0] : edge_mid_e_t_vals[eid][4];
            if(cur_k_dir.dot(k_dir_main) < 0)
            {
                sample_vals[eid] *= -1;
            }
        }
    }
    for(int eid =0; eid < e_num; ++eid)
    {
        if(tet_edge_mid_curv_map.find(edge_keys[eid]) == tet_edge_mid_curv_map.end())
        tet_edge_mid_curv_map[edge_keys[eid]] = edge_mid_e_t_vals[eid];
    }
    Eigen::RowVector<double, 6> diff = sample_vals - linear_vals;
    double max_diff = std::max(diff.maxCoeff(), - diff.minCoeff()); 
    return max_diff;
}

bool TetDirConsistenceThreshold(const Eigen::Matrix<double, 4, 3>& k_dirs, double threshold)
{
    Eigen::Vector3d k_dir0 = k_dirs.row(0);
    Eigen::Vector3d k_dir1 = k_dirs.row(1);
    Eigen::Vector3d k_dir2 = k_dirs.row(2);
    Eigen::Vector3d k_dir3 = k_dirs.row(3);

    k_dir0 = k_dir0 / sqrt(k_dir0.dot(k_dir0));
    k_dir1 = k_dir1 / sqrt(k_dir1.dot(k_dir1));
    k_dir2 = k_dir2 / sqrt(k_dir2.dot(k_dir2));
    k_dir3 = k_dir3 / sqrt(k_dir3.dot(k_dir3));

    Eigen::Vector3d mean_dir = k_dir0 + k_dir1 + k_dir2 + k_dir3;
    mean_dir = mean_dir / sqrt(mean_dir.dot(mean_dir));

    double d0 = mean_dir.dot(k_dir0);
    double d1 = mean_dir.dot(k_dir1);
    double d2 = mean_dir.dot(k_dir2);
    double d3 = mean_dir.dot(k_dir3);

    double mean_diff = (d0 + d1 + d2 + d3) / 4.0;  

    // Eigen::RowVector3d mean_dir = k_dirs.colwise().mean();
    // double len = sqrt(mean_dir.dot(mean_dir));
    // mean_dir = mean_dir / len;
    // Eigen::VectorXd result = k_dirs * mean_dir.transpose();
    // return result.mean() > threshold;
    return mean_diff >= threshold;
}


/// Construct the value differences between linear interpolations and bezier approximations at 16 bezier control points (excluding control points at tet vertices)
/// @param[in] valList          The eigen vector of 20 bezier values.
///
/// @return         The value differences at 16 control points.
inline Eigen::Vector<double, 16> bezierDiff(const Eigen::Vector<double,20>& valList)
{
    /// Constant coefficient to obtain linear interpolated values at each bezier control points
    const Eigen::Matrix<double, 16, 4> linear_coeff {{2, 1, 0, 0}, {2, 0, 1, 0}, {2, 0, 0, 1}, {0, 2, 1, 0},{0, 2, 0, 1}, {1, 2, 0, 0}, {0, 0, 2, 1}, {1, 0, 2, 0},{0, 1, 2, 0}, {1, 0, 0, 2}, {0, 1, 0, 2}, {0, 0, 1, 2},{0, 1, 1, 1}, {1, 0, 1, 1}, {1, 1, 0, 1}, {1, 1, 1, 0}};
    Eigen::Vector<double, 16> linear_val = (linear_coeff * valList.head(4)) / 3;
    return valList.tail(16) - linear_val;
}

/// The check whether the two functions' intersection curve lies in the tet.
/// Transforms values at 20 bezier control points for two functions into the correct format that `convex_hull_membership` library can use.
/// @param[in] valList          The eigen vector of 20 bezier values.
///
/// @return A array that convexhull memship library can use.
std::array<double, 40> parse_convex_points2d(const Eigen::Matrix<double, 2, 20> &valList) {
    std::array<double, 40> transposed;
    Eigen::MatrixXd::Map(transposed.data(), 2, 20) = valList;
    return transposed;
}

/// The check whether the three functions' intersection point lies in the tet.
/// Transforms values at 20 bezier control points for three functions into the correct format that `convex_hull_membership` library can use.
/// @param[in] valList          The eigen vector of 20 bezier values.
///
/// @return A array that convexhull memship library can use.
std::array<double, 60> parse_convex_points3d(const Eigen::Matrix<double, 3, 20>& valList) {
    std::array<double, 60> transposed;
    Eigen::MatrixXd::Map(transposed.data(), 3, 20) = valList;
    return transposed;
}

/// Given two functions, here is the check whether the two functions' intersection curve can be well approximated by linear interpolation.
/// @param[in] grad         The linear interpolations' gradients of these two functions within the tet.
/// @param[in] diff_matrix          The difference between linear interpolations and bezier approximations at 16 bezier control points (excluding control points at tet vertices) for these two functions
/// @param[in] sqD          The squared determinant to offset the un-normalized gradients
/// @param[in] threshold            The user-defined error threshold
///
/// @return         Whether the tet passes the check for these two functions.
bool two_func_check (Eigen::Matrix<double, 2, 3> grad,
                     const Eigen::Matrix<double, 16, 2> diff_matrix,
                     const double sqD,
                     const double threshold)
{
    Eigen::Matrix2d w;
    w << grad.row(0).squaredNorm(), grad.row(0).dot(grad.row(1)),
    grad.row(0).dot(grad.row(1)), grad.row(1).squaredNorm();
    double E = w.determinant();
    Eigen::Matrix<double, 2, 3> H = Eigen::Matrix2d({{w(1, 1), -w(1, 0)}, {-w(0, 1), w(0,0)}}) * grad;
    
    //find the largest max error (max squared gamma: the LHS of the equation) among all 16 bezier control points
    Eigen::Matrix<double, 16, 3> unNormDis = diff_matrix * H;
    Eigen::Vector<double, 16> dotProducts = sqD * unNormDis.cwiseProduct(unNormDis).rowwise().sum();
    return (dotProducts.maxCoeff() > threshold*threshold * E * E);
}

/// Given two functions, here is the check whether the three functions' intersection curve can be well approximated by linear interpolation.
/// @param[in] grad         The linear interpolations' gradients of these three functions within the tet.
/// @param[in] diff_matrix          The difference between linear interpolations and bezier approximations at 16 bezier control points (excluding control points at tet vertices) for these two functions
/// @param[in] sqD          The squared determinant to offset the un-normalized gradients
/// @param[in] threshold            The user-defined error threshold
///
/// @return         Whether the tet passes the check for these three functions.
bool three_func_check (Eigen::Matrix<double, 3, 3> grad,
                     const Eigen::Matrix<double, 16, 3> diff_matrix,
                     const double sqD,
                     const double threshold)
{
    double E = grad.determinant();
    Eigen::Matrix<double, 3, 3> H;
    H << grad.row(1).cross(grad.row(2)),
    grad.row(2).cross(grad.row(0)),
    grad.row(0).cross(grad.row(1));
    Eigen::Matrix<double, 16, 3> unNormDis_eigen = diff_matrix * H;
    Eigen::Vector<double, 16> dotProducts = sqD * unNormDis_eigen.cwiseProduct(unNormDis_eigen).rowwise().sum();
    //double maxGammaSq = dotProducts.maxCoeff();
    return (dotProducts.maxCoeff() > threshold*threshold * E * E);
}

bool critIA(
            const Eigen::Matrix<double, 4, 3> &pts,
            const std::array<llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20>,4>& tet_info,
            const size_t funcNum,
            const double threshold,
            const bool curve_network,
            bool& active,
            int &sub_call_two,
            int &sub_call_three)
{
    Eigen::Matrix<double, Eigen::Dynamic, 20> valList (funcNum, 20);
    Eigen::Matrix<double, Eigen::Dynamic, 16> diffList(funcNum, 16);
    llvm_vecsmall::SmallVector<bool, 20> activeTF(funcNum);
    Eigen::Matrix<double, 20, 3> gradList;
    Eigen::Vector3d eigenVec1 = pts.row(1) - pts.row(0), eigenVec2 = pts.row(2) - pts.row(0), eigenVec3 = pts.row(3) - pts.row(0), eigenVec4 = pts.row(2) - pts.row(1), eigenVec5 = pts.row(3) - pts.row(1), eigenVec6 = pts.row(3) - pts.row(2);
    Eigen::Matrix<double, 3, 6> vec;
    vec << eigenVec1, eigenVec2, eigenVec3, eigenVec4, eigenVec5, eigenVec6;
    double D = vec.leftCols(3).determinant();
    double sqD = D*D;
    Eigen::Matrix3d crossMatrix;
    crossMatrix << eigenVec2.cross(eigenVec3), eigenVec3.cross(eigenVec1), eigenVec1.cross(eigenVec2);
    
    int activeNum = 0;
    //single function linearity check:
    for (int funcIter = 0; funcIter < funcNum; funcIter++){
        //Timer single_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        //storing bezier and linear info for later linearity comparison
        Eigen::Matrix4d func_info;
        func_info << tet_info[0][funcIter], tet_info[1][funcIter], tet_info[2][funcIter], tet_info[3][funcIter];
        Eigen::RowVector4d vals = func_info.col(0);
        Eigen::Matrix<double, 4, 3> grads_eigen = func_info.rightCols(3);
        valList.row(funcIter) = bezierConstruct(vals, grads_eigen, vec);
        //Timer single_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        activeTF[funcIter] = get_sign(valList.row(funcIter).maxCoeff()) != get_sign(valList.row(funcIter).minCoeff());
        //single_timer.Stop();
        if (activeTF[funcIter]){
            if (!active){
                active = true;
            }
            activeNum++;
            Eigen::Vector3d unNormF = Eigen::RowVector3d(vals(1)-vals(0), vals(2)-vals(0), vals(3)-vals(0)) * crossMatrix.transpose();
            gradList.row(funcIter) = unNormF;
            diffList.row(funcIter) = bezierDiff(valList.row(funcIter));
            double error = std::max(diffList.row(funcIter).maxCoeff(), -diffList.row(funcIter).minCoeff());
            //Timer single2_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
            double lhs = error * error * sqD;
            double rhs;
            if (!curve_network){
                rhs = threshold * threshold * gradList.row(funcIter).squaredNorm();
            }else{
                rhs = std::numeric_limits<double>::infinity() * gradList.row(funcIter).squaredNorm();
            }
            if (lhs > rhs) {
                //single2_timer.Stop();
                return true;
            }
            //single2_timer.Stop();
        }
    }
    //Timer single_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
    if(activeNum < 2){
        //single_timer.Stop();
        return false;
    }
    llvm_vecsmall::SmallVector<int, 20> activeFunc(activeNum);
    int activeFuncIter = 0;
    for (int funcIter = 0; funcIter < funcNum; funcIter++){
        if (activeTF[funcIter]){
            activeFunc[activeFuncIter] = funcIter;
            activeFuncIter++;
        }
    }
    llvm_vecsmall::SmallVector<llvm_vecsmall::SmallVector<bool, 20>, 20> zeroXResult(funcNum, llvm_vecsmall::SmallVector<bool, 20>(funcNum));
    //single_timer.Stop();
    const int pairNum = activeNum * (activeNum-1)/2, triNum = activeNum * (activeNum-1) * (activeNum - 2)/ 6;
    
    // 2-function checks
    int activeDouble_count = 0;
    {
        //Timer timer(twoFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        bool zeroX;
        
        for (int i = 0; i < activeNum - 1; i++){
            for (int j = i + 1; j < activeNum; j++){
                std::array<int, 2> pairIndices = {activeFunc[i], activeFunc[j]};
                std::array<double, 40> nPoints = parse_convex_points2d(valList(pairIndices, Eigen::all));
                //Timer sub_timer(sub_twoFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
                zeroX = convex_hull_membership::contains<2, double>(nPoints, query_2d);
                //sub_timer.Stop();
                
                if (zeroX){
                    activeDouble_count++;
                    sub_call_two ++;
                    zeroXResult[pairIndices[0]][pairIndices[1]] = true;
                    zeroXResult[pairIndices[1]][pairIndices[0]] = true;
                    Eigen::Matrix<double, 2, 3> grad = gradList(pairIndices, Eigen::all);
                    Eigen::Matrix<double, 16, 2> diff_matrix = diffList(pairIndices, Eigen::all).transpose();
                    // two function linearity test:
                    if (two_func_check (grad, diff_matrix, sqD, threshold)){
                        //timer.Stop();
                        return true;
                    }
                }
            }
        }
        //timer.Stop();
    }
    if(activeDouble_count < 3)
        return false;
    // 3-function checks
    {
        //Timer timer(threeFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        bool zeroX;
        for (int i = 0; i < activeNum - 2; i++){
            for (int j = i + 1; j < activeNum - 1; j++){
                for (int k = j + 1; k < activeNum; k++){
                    std::array<int, 3> tripleIndices = {activeFunc[i], activeFunc[j], activeFunc[k]};
                    if(!(zeroXResult[tripleIndices[0]][tripleIndices[1]]&&zeroXResult[tripleIndices[0]][tripleIndices[2]]&&zeroXResult[tripleIndices[1]][tripleIndices[2]]))
                        continue;
                    std::array<double, 60> nPoints = parse_convex_points3d(valList(tripleIndices, Eigen::all));
                    sub_call_three ++;
                    //Timer sub_timer(sub_threeFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
                    zeroX = convex_hull_membership::contains<3, double>(nPoints, query_3d);
                    //sub_timer.Stop();
                    
                    if (zeroX){
                        Eigen::Matrix<double, 3, 3> grad = gradList(tripleIndices, Eigen::all);
                        Eigen::Matrix<double, 16, 3> diff_matrix = diffList(tripleIndices, Eigen::all).transpose();
                        if (three_func_check (grad, diff_matrix, sqD, threshold)){
                            //timer.Stop();
                            return true;
                        }
                    }
                }
            }
        }
        //timer.Stop();
    }
    return false;
}

bool critIARidge(
            const Eigen::Matrix<double, 4, 3> &pts,
            const std::array<llvm_vecsmall::SmallVector<CurvatureData<double>, 20>,4>& tet_info,
            const mtet::TetId tid,
            mtet::MTetMesh &grid_mesh,
            const size_t funcNum,
            const double threshold,
            const bool curve_network,
            const int crest_type,
            bool& active,
            bool& orientable,
            int &sub_call_two,
            int &sub_call_three )
{
    Eigen::Matrix<double, Eigen::Dynamic, 20> valList (funcNum, 20);
    Eigen::Matrix<double, Eigen::Dynamic, 16> diffList(funcNum, 16);
    llvm_vecsmall::SmallVector<bool, 20> activeTF(funcNum);
    Eigen::Matrix<double, 20, 3> gradList;
    Eigen::Vector3d eigenVec1 = pts.row(1) - pts.row(0), eigenVec2 = pts.row(2) - pts.row(0), eigenVec3 = pts.row(3) - pts.row(0), eigenVec4 = pts.row(2) - pts.row(1), eigenVec5 = pts.row(3) - pts.row(1), eigenVec6 = pts.row(3) - pts.row(2);
    Eigen::Matrix<double, 3, 6> vec;
    vec << eigenVec1, eigenVec2, eigenVec3, eigenVec4, eigenVec5, eigenVec6;
    double D = vec.leftCols(3).determinant();
    double sqD = D*D;
    Eigen::Matrix3d crossMatrix;
    crossMatrix << eigenVec2.cross(eigenVec3), eigenVec3.cross(eigenVec1), eigenVec1.cross(eigenVec2);
    
    int activeNum = 0;
    //single function linearity check:
    for (int funcIter = 0; funcIter < funcNum; funcIter++){
        //Timer single_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        //storing bezier and linear info for later linearity comparison
        // Eigen::Matrix<double, 4, 7> func_info;
        // func_info << tet_info[0][funcIter].transpose(), tet_info[1][funcIter].transpose(), 
                    //  tet_info[2][funcIter].transpose(), tet_info[3][funcIter].transpose();
        // std::cout << " finish assign func_info ..." << std::endl;
        // Eigen::RowVector4d vals = func_info.col(0);
        Eigen::RowVector4d f_vals;
        Eigen::Matrix<double, 4, 3> f_grads;
        Eigen::RowVector4d emax_vals;
        Eigen::RowVector4d emin_vals;
        Eigen::RowVector4d kmax_vals;
        Eigen::RowVector4d kmin_vals;
        Eigen::Matrix<double, 4, 3> grads_eigen_de1;
        Eigen::Matrix<double, 4, 3> grads_eigen_de2;
        Eigen::Matrix<double, 4, 3> k1_dirs;
        Eigen::Matrix<double, 4, 3> k2_dirs;
        // std::cout << " start assign curvature data info ..." << std::endl;
        for(int i = 0; i < 4; ++i)
        {
            f_vals[i] = tet_info[i][funcIter].f_val_;
            emax_vals[i] = tet_info[i][funcIter].e1_;
            emin_vals[i] = tet_info[i][funcIter].e2_;
            kmax_vals[i] = tet_info[i][funcIter].k1_;
            kmin_vals[i] = tet_info[i][funcIter].k2_;
            k1_dirs.row(i) =  tet_info[i][funcIter].t1_;
            k2_dirs.row(i) =  tet_info[i][funcIter].t2_;
            grads_eigen_de1.row(i) =  tet_info[i][funcIter].e1_d1_;
            grads_eigen_de2.row(i) =  tet_info[i][funcIter].e2_d2_;
            f_grads.row(i) =  tet_info[i][funcIter].f_gradient_;
        }
        // auto bezier_f_vals = bezierConstruct(f_vals, f_grads, vec);
        // bool zero_cross_iso = get_sign(bezier_f_vals.maxCoeff()) != get_sign(bezier_f_vals.minCoeff());
        // // std::cout << " zero cross iso res : " << zero_cross_iso << "  " << bezier_f_vals.maxCoeff() << " " 
        // //             << bezier_f_vals.minCoeff() << std::endl;
        // if(!zero_cross_iso)
        // {
        //     return false;
        // }
        // std::cout << " finish assign curvature data info ..." << std::endl;
        // 0 stands for rigdes, 1 for valleys
        // bool is_target_crest_type = TetCrestTypeTest(kmax_vals, kmin_vals, crest_type);
        auto& k_dirs = crest_type == 0 ? k1_dirs : k2_dirs;
        auto& e_vals = crest_type == 0 ? emax_vals : emin_vals;
        
        int rv_tet_pt_count = 0;
        bool is_target_crest_type = TetCrestTypeTest2(kmax_vals, kmin_vals, crest_type, rv_tet_pt_count);
        if(!is_target_crest_type)
        {
            OutTetObj::rv_failed_tets.AddNewTet(pts);
            return false;
        } 
        if(rv_tet_pt_count < 4)
        {
            active = TetPerEdgeZeroCrossingTest(k_dirs, e_vals);
            return active;
        }
        orientable = CheckTetOrientable(e_vals, k_dirs);

        if(orientable) 
        {
            active = e_vals.maxCoeff() * e_vals.minCoeff() < 0;
        } else {
            OutTetObj::unoriented_tets.AddNewTet(pts);
            // active = TetPerEdgeZeroCrossingTest(k_dirs, e_vals);
            active = true;
            return true;
        }
        if(!active) 
        {
            return false;
        }
        // valList.row(funcIter) = bezierSampleVals(pts, e_vals, k_dirs, crest_type);
        // std::cout << " val list 0 : " << valList.row(funcIter) << std::endl;
        // double dot_threshold = 0.996;
        // if(active )
        // {
        // return !TetDirConsistenceThreshold(k_dirs, dot_threshold);
        // } 
     
        // return true;
        // Eigen::Matrix<double, 4, 3> grads_eigen = func_info.rightCols(3);
        // Eigen::Matrix<double, 4, 3> k1_dirs = func_info.middleCols(1, 3);
        // std::cout << " start bezierConstructRidge ..." << std::endl; 
        // valList.row(funcIter) = bezierConstructRidge(vals, k1_dirs, grads_eigen, vec);
        // std::cout << " finsih bezierConstructRidge ..." << std::endl; 
        //Timer single_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        // activeTF[funcIter] = get_sign(valList.row(funcIter).maxCoeff()) != get_sign(valList.row(funcIter).minCoeff());
        // if(get_sign(vals.maxCoeff()) != get_sign(vals.minCoeff()))
        // {
        //     activeTF[funcIter] = true;
        // }
        //single_timer.Stop();
        // if (activeTF[funcIter])
        {
            // if (!active){
            //     active = true;
            // }
            activeNum++;
            Eigen::Vector3d unNormF = Eigen::RowVector3d(e_vals(1)-e_vals(0), e_vals(2)-e_vals(0), e_vals(3)-e_vals(0)) * crossMatrix.transpose();
            gradList.row(funcIter) = unNormF;
            // diffList.row(funcIter) = bezierDiff(valList.row(funcIter));
            // double error = std::max(diffList.row(funcIter).maxCoeff(), -diffList.row(funcIter).minCoeff());
            // std::cout << "error 0  : " << error << std::endl;
            // double error = bezierSampleValsWithHash(tid, grid_mesh, pts, e_vals, k_dirs, crest_type);
            double error = bezierSampleValsWithHashSimple(tid, grid_mesh, pts, e_vals, k_dirs, crest_type);
            // std::cout << "error 1  : " << error << " threshold " << threshold << std::endl;
            //Timer single2_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
            double lhs = error * error * sqD;
            double rhs;
            if (!curve_network){
                rhs = threshold * threshold * gradList.row(funcIter).squaredNorm();
            }else{
                rhs = std::numeric_limits<double>::infinity() * gradList.row(funcIter).squaredNorm();
            }
            if (lhs > rhs) {
                //single2_timer.Stop();
                // if(!is_tet_kdir_consistent) OutTetObj::unoriented_tets_split.AddNewTet(pts);
                return true;
            } else {
                // if(!is_tet_kdir_consistent) OutTetObj::unoriented_tets_unsplit.AddNewTet(pts);
                OutTetObj::threshold_tets.AddNewTet(pts);
                return false;
            }
            //single2_timer.Stop();
        }
    }
    //Timer single_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
    if(activeNum < 2){
        //single_timer.Stop();
        return false;
    }
    llvm_vecsmall::SmallVector<int, 20> activeFunc(activeNum);
    int activeFuncIter = 0;
    for (int funcIter = 0; funcIter < funcNum; funcIter++){
        if (activeTF[funcIter]){
            activeFunc[activeFuncIter] = funcIter;
            activeFuncIter++;
        }
    }
    llvm_vecsmall::SmallVector<llvm_vecsmall::SmallVector<bool, 20>, 20> zeroXResult(funcNum, llvm_vecsmall::SmallVector<bool, 20>(funcNum));
    //single_timer.Stop();
    const int pairNum = activeNum * (activeNum-1)/2, triNum = activeNum * (activeNum-1) * (activeNum - 2)/ 6;
    
    // 2-function checks
    int activeDouble_count = 0;
    {
        //Timer timer(twoFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        bool zeroX;
        
        for (int i = 0; i < activeNum - 1; i++){
            for (int j = i + 1; j < activeNum; j++){
                std::array<int, 2> pairIndices = {activeFunc[i], activeFunc[j]};
                std::array<double, 40> nPoints = parse_convex_points2d(valList(pairIndices, Eigen::all));
                //Timer sub_timer(sub_twoFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
                zeroX = convex_hull_membership::contains<2, double>(nPoints, query_2d);
                //sub_timer.Stop();
                
                if (zeroX){
                    activeDouble_count++;
                    sub_call_two ++;
                    zeroXResult[pairIndices[0]][pairIndices[1]] = true;
                    zeroXResult[pairIndices[1]][pairIndices[0]] = true;
                    Eigen::Matrix<double, 2, 3> grad = gradList(pairIndices, Eigen::all);
                    Eigen::Matrix<double, 16, 2> diff_matrix = diffList(pairIndices, Eigen::all).transpose();
                    // two function linearity test:
                    if (two_func_check (grad, diff_matrix, sqD, threshold)){
                        //timer.Stop();
                        return true;
                    }
                }
            }
        }
        //timer.Stop();
    }
    if(activeDouble_count < 3)
        return false;
    // 3-function checks
    {
        //Timer timer(threeFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        bool zeroX;
        for (int i = 0; i < activeNum - 2; i++){
            for (int j = i + 1; j < activeNum - 1; j++){
                for (int k = j + 1; k < activeNum; k++){
                    std::array<int, 3> tripleIndices = {activeFunc[i], activeFunc[j], activeFunc[k]};
                    if(!(zeroXResult[tripleIndices[0]][tripleIndices[1]]&&zeroXResult[tripleIndices[0]][tripleIndices[2]]&&zeroXResult[tripleIndices[1]][tripleIndices[2]]))
                        continue;
                    std::array<double, 60> nPoints = parse_convex_points3d(valList(tripleIndices, Eigen::all));
                    sub_call_three ++;
                    //Timer sub_timer(sub_threeFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
                    zeroX = convex_hull_membership::contains<3, double>(nPoints, query_3d);
                    //sub_timer.Stop();
                    
                    if (zeroX){
                        Eigen::Matrix<double, 3, 3> grad = gradList(tripleIndices, Eigen::all);
                        Eigen::Matrix<double, 16, 3> diff_matrix = diffList(tripleIndices, Eigen::all).transpose();
                        if (three_func_check (grad, diff_matrix, sqD, threshold)){
                            //timer.Stop();
                            return true;
                        }
                    }
                }
            }
        }
        //timer.Stop();
    }
    return false;
}

bool critCSG(
             const Eigen::Matrix<double, 4, 3> &pts,
             const std::array<llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20>,4>& tet_info,
             const size_t funcNum,
             const std::function<std::pair<std::array<double, 2>, llvm_vecsmall::SmallVector<int, 20>>(llvm_vecsmall::SmallVector<std::array<double, 2>, 20>)> csg_func,
             const double threshold,
             const bool curve_network,
             bool& active,
             int &sub_call_two,
             int &sub_call_three)
{
    Eigen::Matrix<double, Eigen::Dynamic, 20> valList (funcNum, 20);
    Eigen::Matrix<double, Eigen::Dynamic, 16> diffList(funcNum, 16);
    llvm_vecsmall::SmallVector<bool, 20> activeTF(funcNum);
    llvm_vecsmall::SmallVector<std::array<double , 2>, 20> funcInt(funcNum);
    Eigen::Matrix<double, 20, 3> gradList;
    Eigen::Vector3d eigenVec1 = pts.row(1) - pts.row(0), eigenVec2 = pts.row(2) - pts.row(0), eigenVec3 = pts.row(3) - pts.row(0), eigenVec4 = pts.row(2) - pts.row(1), eigenVec5 = pts.row(3) - pts.row(1), eigenVec6 = pts.row(3) - pts.row(2);
    Eigen::Matrix<double, 3, 6> vec;
    vec << eigenVec1, eigenVec2, eigenVec3, eigenVec4, eigenVec5, eigenVec6;
    double D = vec.leftCols(3).determinant();
    double sqD = D*D;
    Eigen::Matrix3d crossMatrix;
    crossMatrix << eigenVec2.cross(eigenVec3), eigenVec3.cross(eigenVec1), eigenVec1.cross(eigenVec2);
    
    int activeNum = 0;
    //single function linearity check:
    for (int funcIter = 0; funcIter < funcNum; funcIter++){
        //Timer single_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        Eigen::Matrix4d func_info;
        func_info << tet_info[0][funcIter], tet_info[1][funcIter], tet_info[2][funcIter], tet_info[3][funcIter];
        Eigen::RowVector4d vals = func_info.col(0);
        Eigen::Matrix<double, 4, 3> grads_eigen = func_info.rightCols(3);
        valList.row(funcIter) = bezierConstruct(vals, grads_eigen, vec);
        funcInt[funcIter] = {valList.row(funcIter).minCoeff(), valList.row(funcIter).maxCoeff()};
    }
    
        std::pair<std::array<double, 2>, llvm_vecsmall::SmallVector<int, 20>> csgResult = csg_func(funcInt);
        if(csgResult.first[0] * csgResult.first[1] > 0){
            return false;
        }else{
            for (size_t funcIter = 0; funcIter < funcNum; funcIter++){
                //Timer single_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
                activeTF[funcIter] = !csgResult.second[funcIter];
                //single_timer.Stop();
                if (activeTF[funcIter]){
                    if (!active){
                        active = true;
                    }
                    activeNum++;
                    double v0 = tet_info[0][funcIter][0], v1 = tet_info[1][funcIter][0], v2 = tet_info[2][funcIter][0], v3 = tet_info[3][funcIter][0];
                    Eigen::Vector3d unNormF = Eigen::RowVector3d(v1-v0, v2-v0, v3-v0) * crossMatrix.transpose();
                    gradList.row(funcIter) = unNormF;
                    diffList.row(funcIter) = bezierDiff(valList.row(funcIter));
                    double error = std::max(diffList.row(funcIter).maxCoeff(), -diffList.row(funcIter).minCoeff());
                    //Timer single2_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
                    double lhs = error * error * sqD;
                    double rhs;
                    if (!curve_network){
                        rhs = threshold * threshold * gradList.row(funcIter).squaredNorm();
                    }else{
                        rhs = std::numeric_limits<double>::infinity() * gradList.row(funcIter).squaredNorm();
                    }
                    if (lhs > rhs) {
                        //single2_timer.Stop();
                        return true;
                    }
                    //single2_timer.Stop();
                }
            }
        }
    //Timer single_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
    if(activeNum < 2){
        //single_timer.Stop();
        return false;
    }
    llvm_vecsmall::SmallVector<int, 20> activeFunc(activeNum);
    int activeFuncIter = 0;
    for (int funcIter = 0; funcIter < funcNum; funcIter++){
        if (activeTF[funcIter]){
            activeFunc[activeFuncIter] = funcIter;
            activeFuncIter++;
        }
    }
    llvm_vecsmall::SmallVector<llvm_vecsmall::SmallVector<bool, 20>, 20> zeroXResult(funcNum, llvm_vecsmall::SmallVector<bool, 20>(funcNum));
    //single_timer.Stop();
    const int pairNum = activeNum * (activeNum-1)/2, triNum = activeNum * (activeNum-1) * (activeNum - 2)/ 6;
    
    // 2-function checks
    int activeDouble_count = 0;
    {
        //Timer timer(twoFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        bool zeroX;
        for (int i = 0; i < activeNum - 1; i++){
            for (int j = i + 1; j < activeNum; j++){
                std::array<int, 2> pairIndices = {activeFunc[i], activeFunc[j]};
                std::array<double, 40> nPoints = parse_convex_points2d(valList(pairIndices, Eigen::all));
                //Timer sub_timer(sub_twoFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
                zeroX = convex_hull_membership::contains<2, double>(nPoints, query_2d);
                //sub_timer.Stop();
                
                if (zeroX){
                    activeDouble_count++;
                    sub_call_two ++;
                    zeroXResult[pairIndices[0]][pairIndices[1]] = true;
                    zeroXResult[pairIndices[1]][pairIndices[0]] = true;
                    Eigen::Matrix<double, 2, 3> grad = gradList(pairIndices, Eigen::all);
                    Eigen::Matrix<double, 16, 2> diff_matrix = diffList(pairIndices, Eigen::all).transpose();
                    // two function linearity test:
                    if (two_func_check (grad, diff_matrix, sqD, threshold)){
                        //timer.Stop();
                        return true;
                    }
                }
            }
        }
        //timer.Stop();
    }
    if(activeDouble_count < 3)
        return false;
    // 3-function checks
    {
        //Timer timer(threeFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        bool zeroX;
        for (int i = 0; i < activeNum - 2; i++){
            for (int j = i + 1; j < activeNum - 1; j++){
                for (int k = j + 1; k < activeNum; k++){
                    std::array<int, 3> tripleIndices = {activeFunc[i], activeFunc[j], activeFunc[k]};
                    if(!(zeroXResult[tripleIndices[0]][tripleIndices[1]]&&zeroXResult[tripleIndices[0]][tripleIndices[2]]&&zeroXResult[tripleIndices[1]][tripleIndices[2]]))
                        continue;
                    std::array<double, 60> nPoints = parse_convex_points3d(valList(tripleIndices, Eigen::all));
                    sub_call_three ++;
                    //Timer sub_timer(sub_threeFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
                    zeroX = convex_hull_membership::contains<3, double>(nPoints, query_3d);
                    //sub_timer.Stop();
                    
                    if (zeroX){
                        Eigen::Matrix<double, 3, 3> grad = gradList(tripleIndices, Eigen::all);
                        Eigen::Matrix<double, 16, 3> diff_matrix = diffList(tripleIndices, Eigen::all).transpose();
                        if (three_func_check (grad, diff_matrix, sqD, threshold)){
                            //timer.Stop();
                            return true;
                        }
                    }
                }
            }
        }
        //timer.Stop();
    }
    return false;
}
bool critMI(
            const Eigen::Matrix<double, 4, 3> &pts,
            const std::array<llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20>,4>& tet_info,
            const size_t funcNum,
            const double threshold,
            const bool curve_network,
            bool& active,
            int &sub_call_two,
            int &sub_call_three)
{
    Eigen::Vector3d eigenVec1 = pts.row(1) - pts.row(0), eigenVec2 = pts.row(2) - pts.row(0), eigenVec3 = pts.row(3) - pts.row(0), eigenVec4 = pts.row(2) - pts.row(1), eigenVec5 = pts.row(3) - pts.row(1), eigenVec6 = pts.row(3) - pts.row(2);
    Eigen::Matrix<double, 3, 6> vec;
    vec << eigenVec1, eigenVec2, eigenVec3, eigenVec4, eigenVec5, eigenVec6;
    double D = vec.leftCols(3).determinant();
    double sqD = D*D;
    Eigen::Matrix3d crossMatrix_eigen;
    crossMatrix_eigen << eigenVec2.cross(eigenVec3), eigenVec3.cross(eigenVec1), eigenVec1.cross(eigenVec2);
    Eigen::Matrix<double, 20, 3> gradList_eigen;
    Eigen::Matrix<double, Eigen::Dynamic, 20> valList (funcNum, 20);
    Eigen::Matrix<double, Eigen::Dynamic, 16> diffList(funcNum, 16);
    
    llvm_vecsmall::SmallVector<bool, 20> activeList(funcNum);
    llvm_vecsmall::SmallVector<std::array<double , 2>, 20> funcInt(funcNum);
    double maxLow = -1 * std::numeric_limits<double>::infinity();
    llvm_vecsmall::SmallVector<llvm_vecsmall::SmallVector<bool, 20>, 20> activePair(funcNum, llvm_vecsmall::SmallVector<bool, 20>(funcNum, false));
    //single function linearity check:
    for (int funcIter = 0; funcIter < funcNum; funcIter++){
        Eigen::Matrix4d func_info;
        func_info << tet_info[0][funcIter], tet_info[1][funcIter], tet_info[2][funcIter], tet_info[3][funcIter];
        Eigen::RowVector4d vals = func_info.col(0);
        Eigen::Matrix<double, 4, 3> grads_eigen = func_info.rightCols(3);
        valList.row(funcIter) = bezierConstruct(vals, grads_eigen, vec);
        funcInt[funcIter] = {valList.row(funcIter).minCoeff(), valList.row(funcIter).maxCoeff()};
        if (maxLow < funcInt[funcIter][0]){
            maxLow = funcInt[funcIter][0];
        }
    }
    llvm_vecsmall::SmallVector<int, 20> activeFunc;
    for (int funcIter = 0; funcIter < funcNum; funcIter++){
        if(funcInt[funcIter][1] > maxLow){
            activeFunc.push_back(funcIter);
        }
    }
    size_t activeNum = activeFunc.size();
    if(activeNum < 2)
        return false;

    //Timer get_func_timer(getActiveMuti, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
    const size_t pairNum = activeNum * (activeNum-1)/2, triNum = activeNum * (activeNum-1) * (activeNum - 2)/ 6, quadNum = activeNum * (activeNum - 1) * (activeNum - 2) * (activeNum - 3)/ 24;
//    get_func_timer.Stop();
    
    for (int i = 0; i < activeNum - 1; i++){
        for (int j = i + 1; j < activeNum; j++){
            //Timer single_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
            int funcIndex1 = activeFunc[i];
            int funcIndex2 = activeFunc[j];
            Eigen::Vector<double, 20> diff_at_point;
            diff_at_point = valList.row(funcIndex2) - valList.row(funcIndex1);
            bool activeTF = get_sign(diff_at_point.maxCoeff()) == get_sign(diff_at_point.minCoeff()) ? false : true;
            //single_timer.Stop();
            if (activeTF){
                if (!active){
                    active = true;
                }
                activePair[funcIndex1][funcIndex2] = true;
                activePair[funcIndex2][funcIndex1] = true;
                if (!activeList[funcIndex1]){
                    activeList[funcIndex1] = true;
                    double v0 = valList(funcIndex1, 0), v1 = valList(funcIndex1, 1), v2 = valList(funcIndex1, 2), v3 = valList(funcIndex1, 3);
                    Eigen::Vector3d unNormF_eigen = Eigen::RowVector3d(v1-v0, v2-v0, v3-v0) * crossMatrix_eigen.transpose();
                    gradList_eigen.row(funcIndex1) = unNormF_eigen;
                    
                    diffList.row(funcIndex1) = bezierDiff(valList.row(funcIndex1));
                }
                if (!activeList[funcIndex2]){
                    activeList[funcIndex2] = true;
                    double v0 = valList(funcIndex2, 0), v1 = valList(funcIndex2, 1), v2 = valList(funcIndex2, 2), v3 = valList(funcIndex2, 3);
                    Eigen::Vector3d unNormF_eigen = Eigen::RowVector3d(v1-v0, v2-v0, v3-v0) * crossMatrix_eigen.transpose();
                    gradList_eigen.row(funcIndex2) = unNormF_eigen;
                    diffList.row(funcIndex2) = bezierDiff(valList.row(funcIndex2));
                    
                }
                Eigen::Vector<double, 16> diff_twofunc;
                diff_twofunc = diffList.row(funcIndex1) - diffList.row(funcIndex2);
                //Timer single2_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
                double error = std::max(diff_twofunc.maxCoeff(), -diff_twofunc.minCoeff());
                Eigen::Vector3d grad_eigen;
                grad_eigen = gradList_eigen.row(funcIndex1) - gradList_eigen.row(funcIndex2);
                double lhs = error * error * sqD;
                double rhs;
                if (!curve_network){
                    rhs = threshold * threshold * grad_eigen.squaredNorm();
                }else{
                    rhs = std::numeric_limits<double>::infinity() * grad_eigen.squaredNorm();
                }
                if (lhs > rhs) {
                    //single2_timer.Stop();
                    return true;
                }
                //single2_timer.Stop();
            }
        }
    }
    
    // 2-function checks
    int activeTriple_count = 0;
    {
        //Timer timer(twoFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        bool zeroX;
        for (int i = 0; i < activeNum - 2; i++){
            for (int j = i + 1; j < activeNum - 1; j++){
                for (int k = j + 1; k < activeNum; k++){
                    int funcIndex1 = activeFunc[i];
                    int funcIndex2 = activeFunc[j];
                    int funcIndex3 = activeFunc[k];
                    if(!(activePair[funcIndex1][funcIndex2]&&activePair[funcIndex1][funcIndex3]&&activePair[funcIndex2][funcIndex3]))
                        continue;
                    Eigen::Matrix<double,2, 20> diff_mi(2, 20);
                    diff_mi.row(0) = valList.row(funcIndex1) - valList.row(funcIndex2);
                    diff_mi.row(1) =  valList.row(funcIndex2) - valList.row(funcIndex3);
                    std::array<double, 40> nPoints = parse_convex_points2d(diff_mi);
                    //Timer sub_timer(sub_twoFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
                    zeroX = convex_hull_membership::contains<2, double>(nPoints, query_2d);
                    //sub_timer.Stop();
                    if (zeroX){
                        sub_call_two ++;
                        activeTriple_count++;
                        Eigen::Matrix<double, 2, 3> grad(2, 3);
                        grad.row(0) = gradList_eigen.row(funcIndex1) - gradList_eigen.row(funcIndex2);
                        grad.row(1) = gradList_eigen.row(funcIndex2) - gradList_eigen.row(funcIndex3);
                        Eigen::Matrix<double, 2, 16> diff_matrix(2, 16);
                        diff_matrix.row(0) = diffList.row(funcIndex1) - diffList.row(funcIndex2);
                        diff_matrix.row(1) = diffList.row(funcIndex2) - diffList.row(funcIndex3);
                        if (two_func_check (grad, diff_matrix.transpose(), sqD, threshold)){
                            //timer.Stop();
                            return true;
                        }
                    }
                }
            }
        }
        //timer.Stop();
    }
    if(activeTriple_count < 4)
        return false;
    {
        //Timer timer(threeFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        bool zeroX;
        for (int i = 0; i < activeNum - 3; i++){
            for (int j = i + 1; j < activeNum - 2; j++){
                for (int k = j + 1; k < activeNum - 1; k++){
                    for (int m = k + 1; m < activeNum; m++){
                        int funcIndex1 = activeFunc[i];
                        int funcIndex2 = activeFunc[j];
                        int funcIndex3 = activeFunc[k];
                        int funcIndex4 = activeFunc[m];
                        if(!(activePair[funcIndex1][funcIndex2]&&activePair[funcIndex1][funcIndex3]&&activePair[funcIndex1][funcIndex4]&&activePair[funcIndex2][funcIndex3]&&activePair[funcIndex2][funcIndex4]&&activePair[funcIndex3][funcIndex4]))
                            continue;
                        
                        Eigen::Matrix<double,3, 20> diff_mi(3, 20);
                        diff_mi.row(0) = valList.row(funcIndex1) - valList.row(funcIndex2);
                        diff_mi.row(1) =  valList.row(funcIndex2) - valList.row(funcIndex3);
                        diff_mi.row(2) =  valList.row(funcIndex3) - valList.row(funcIndex4);
                        std::array<double, 60> nPoints = parse_convex_points3d(diff_mi);
                        //Timer sub_timer(sub_twoFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
                        zeroX = convex_hull_membership::contains<3, double>(nPoints, query_3d);
                        //sub_timer.Stop();
                        
                        if (zeroX){
                            sub_call_three ++;
                            Eigen::Matrix<double, 3, 3> grad(3, 3);
                            grad.row(0) = gradList_eigen.row(funcIndex1) - gradList_eigen.row(funcIndex2);
                            grad.row(1) = gradList_eigen.row(funcIndex2) - gradList_eigen.row(funcIndex3);
                            grad.row(2) = gradList_eigen.row(funcIndex3) - gradList_eigen.row(funcIndex4);
                            Eigen::Matrix<double, 3, 16> diff_matrix(3, 16);
                            diff_matrix.row(0) = diffList.row(funcIndex1) - diffList.row(funcIndex2);
                            diff_matrix.row(1) = diffList.row(funcIndex2) - diffList.row(funcIndex3);
                            diff_matrix.row(2) = diffList.row(funcIndex3) - diffList.row(funcIndex4);
                            if (three_func_check (grad, diff_matrix.transpose(), sqD, threshold)){
                                //timer.Stop();
                                return true;
                            }
                        }
                    }
                }
            }
        }
        //timer.Stop();
    }
    return false;
}


