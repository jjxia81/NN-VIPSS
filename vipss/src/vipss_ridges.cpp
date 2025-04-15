
#include "voronoi_gen.h"
#include "local_vipss.hpp"
#include "vipss_ridges.h"
#include <unordered_set>
#include <stack>
#include <queue>

double PointDist(const std::array<double, 3>& p1, const std::array<double, 3>& p2)
{
    double dx = p1[0] - p2[0];
    double dy = p1[1] - p2[1];
    double dz = p1[2] - p2[2];
    return sqrt(dx * dx + dy * dy + dz * dz);

}

bool VIPSSRidges::LoadMeshPly(const std::string & mesh_path)
{
    mesh_points_.clear();
    mesh_faces_.clear();
    std::vector<std::vector<size_t>> faces;
    readPlyMesh(mesh_path, mesh_points_, faces);
    mesh_faces_.resize(faces.size());
    int f_count = 0;
    for(const auto& face : faces)
    {
        mesh_faces_[f_count] = {face[0], face[1], face[2]};
        f_count ++;
    }
    
    
    for(auto& p : mesh_points_)
    {
        p[0] = (p[0] - ori_center_[0]) / scale_; 
        p[1] = (p[1] - ori_center_[1]) / scale_; 
        p[2] = (p[2] - ori_center_[2]) / scale_; 
    }
    return true;
}

void VIPSSRidges::SetDataCenterAndScale(const Point& center, const double scale)
{
    ori_center_ = center;
    scale_ = scale;
}

bool VIPSSRidges::ProcessFaces()
{
 
}

bool VIPSSRidges::CalculateCreaseValues()
{
    size_t ptn = mesh_points_.size();
    crease_values_.resize(ptn);
    for(int i = 0; i < ptn; ++i)
    {
        const auto& eig_vec = point_eig_vecs_[i];
        const auto& gradient = point_graidents_[i];
        crease_values_[i] = arma::dot(eig_vec, gradient);
         
    }
    std::string crease_vals_path = "crease_vals.txt";
    WriteVectorValsToCSV(crease_vals_path, crease_values_);
    return true;
}

bool VIPSSRidges::CalMeshPointsGradientAndEigenVecs(LocalVipss* local_vipss)
{
    local_vipss_ = local_vipss;
    size_t ptn = mesh_points_.size();
    point_eig_vecs_.resize(ptn);
    point_graidents_.resize(ptn);
    std::vector<double> points;
    std::vector<double> eig_vecs;
    std::vector<double> gredients;
    for(int i = 0; i < ptn; ++i)
    {
        double pt[3] = {mesh_points_[i][0], mesh_points_[i][1], mesh_points_[i][2]};
        arma::mat hessian = local_vipss->NNHRBFHessianOMP(pt) * (-1.0);
        arma::vec eigval;
        arma::mat eigvec;
        // arma::eig_sym(eigval, eigvec, hessian);
        arma::mat H_sym = 0.5 * (hessian + hessian.t());
        arma::eig_sym(eigval, eigvec, H_sym);
        eigval = arma::abs(eigval);
        size_t max_id = eigval.index_max();
        Vec max_vec = eigvec.col(max_id);
        // arma::uvec indices = arma::sort_index(eigval, "descend");
        // Vec max_vec = eigvec.col(indices(0));
        point_eig_vecs_[i] = arma::normalise(max_vec);
        double g[3];
        double f_val = local_vipss->NatureNeighborGradientOMP(pt, g);
        Vec gradient = {g[0], g[1], g[2]};
        point_graidents_[i] = arma::normalise(gradient);
    }

    return true;
}




// Function to compute the Hessian numerically
arma::mat VIPSSRidges::computeHessian(std::shared_ptr<RBF_Core> rfb_ptr, Point x, double h) {
    int n = x.size();
    arma::mat H = arma::zeros(3, 3);
    double px = x[0];
    double py = x[1];
    double pz = x[2];
    // Compute Hessian matrix elements using central differences
    for (int j = 0; j < 3; ++j) { // i corresponds to gx, gy, gz
        double grad_minus[3];
        double grad_plus[3];
        // Compute gradient at (x+h, y, z), (x-h, y, z) for j=0 (x), etc.
        if (j == 0) { // Perturb x
            rfb_ptr->evaluate_gradient(px + h, py, pz, grad_minus[0], grad_minus[1], grad_minus[2]);
            rfb_ptr->evaluate_gradient(px - h, py, pz, grad_plus[0], grad_plus[1], grad_plus[2]);
        } else if (j == 1) { // Perturb y
            rfb_ptr->evaluate_gradient(px, py + h, pz, grad_minus[0], grad_minus[1], grad_minus[2]);
            rfb_ptr->evaluate_gradient(px, py - h, pz, grad_plus[0], grad_plus[1], grad_plus[2]);
        } else { // Perturb z
            rfb_ptr->evaluate_gradient(px, py, pz + h, grad_minus[0], grad_minus[1], grad_minus[2]);
            rfb_ptr->evaluate_gradient(px, py, pz - h, grad_plus[0], grad_plus[1], grad_plus[2]);
        }
        for (int i = 0; i < 3; ++i) { // j corresponds to x, y, z
            // Compute second derivative d(g_i)/d(variable_j) using central difference
            H(i,j) = (grad_plus[i] - grad_minus[i]) / (2 * h);
        }
    }
    return H;
}


bool VIPSSRidges::CalMeshPointsGradientAndEigenVecs(std::shared_ptr<RBF_Core> rfb_ptr)
{
    hrfb_ptr_ = rfb_ptr;
    size_t ptn = mesh_points_.size();
    point_eig_vecs_.resize(ptn);
    point_eig_vals_.resize(ptn);
    point_graidents_.resize(ptn);
    std::vector<double> points;
    std::vector<double> eig_vecs;
    std::vector<double> gredients;
    for(int i = 0; i < ptn; ++i)
    {
        double pt[3] = {mesh_points_[i][0], mesh_points_[i][1], mesh_points_[i][2]};

        double g[3];
        double f_val = rfb_ptr->evaluate_gradient(pt[0], pt[1], pt[2], g[0], g[1], g[2]) ;
        Vec gradient = {g[0], g[1], g[2]};
        point_graidents_[i] = arma::normalise(gradient);
        arma::mat hessian = arma::zeros(3,3);
        rfb_ptr->EvaluateHessian(pt[0], pt[1], pt[2], hessian);
        // std::cout << " id " << i << " hessian : " << hessian << std::endl;
        // hessian = computeHessian(rfb_ptr, mesh_points_[i], 0.03);
        // std::cout << "hessian : " << hessian << std::endl;
        // hessian *= -1.0;
        // std::cout << " =--------------" << std::endl;
        arma::vec eigval;
        arma::mat eigvec;
        arma::mat H_sym = 0.5 * (hessian + hessian.t());
        arma::eig_sym(eigval, eigvec, H_sym);
        // TransformEclips(eigval, eigvec, mesh_points_[i]);
        arma::vec eigval_abs = arma::abs(eigval);
        // arma::uvec sort_ids = arma::sort_index(eigval);
        // size_t max_id = arma::index_max(arma::abs(eigval));
        size_t max_id = arma::index_max(eigval_abs);
        // point_eig_vals_[i] = eigval_abs[max_id];
        // size_t max_id = sort_ids[2];
        Vec max_vec = eigvec.col(max_id);
        // if((arma::dot(max_vec, gradient)) < 0)
        // {
        //     max_vec *= -1.0;
        // }
        // point_eig_vecs_[i] = arma::normalise(max_vec);
        point_eig_vecs_[i] = arma::normalise(max_vec) * eigval_abs[max_id];
        eig_vecs.push_back(point_eig_vecs_[i][0]);
        eig_vecs.push_back(point_eig_vecs_[i][1]);
        eig_vecs.push_back(point_eig_vecs_[i][2]);
    }
    return true;
}

string CalEdgeToken(int a, int b)
{
    if (a < b)
    {
        return std::to_string(a) +"_" + std::to_string(b); 
    } 
    return std::to_string(b) +"_" + std::to_string(a); 
}

void VIPSSRidges::GetEdges()
{
    size_t face_num = mesh_faces_.size();
    size_t eid = 0;
    for(int i = 0; i < face_num; ++i)
    {
        const auto& pids = mesh_faces_[i];
        // std::cout << " face v ids :   " << pids[0] << " " << pids[1] << " " << pids[2] << std::endl;
        string token_ab = CalEdgeToken(pids[0], pids[1]);
        if(edge_id_map_.find(token_ab) == edge_id_map_.end())
        {
            edge_id_map_[token_ab] = eid;
            eid ++;
            edges_.push_back({pids[0], pids[1]});
        }
        string token_bc = CalEdgeToken(pids[1], pids[2]);
        if(edge_id_map_.find(token_bc) == edge_id_map_.end())
        {
            edge_id_map_[token_bc] = eid;
            eid ++;
            edges_.push_back({pids[1], pids[2]});
        }
        string token_ca = CalEdgeToken(pids[2], pids[0]);
        if(edge_id_map_.find(token_ca) == edge_id_map_.end())
        {
            edge_id_map_[token_ca] = eid;
            eid ++;
            edges_.push_back({pids[2], pids[0]});
        }
    }
    std::cout << " total edge processed : " << eid << std::endl;
}

// VIPSSRidges::Point VIPSSRidges::IterpolateEdgesPt(const Point& pa, const Point& pb, double va, double vb, double inter_val)
// {
//     // double t = ()
// }

bool VIPSSRidges::CalculateEdgeCreasePoints()
{
    size_t e_num = edges_.size();
    edge_signs_.resize(e_num, -1);
    edge_int_pts_.clear();
    // edge_point_types_.clear();
    edge_eig_vals_.clear();
    int interp_p_id = 0;
    for(int i = 0; i < e_num; ++i)
    {
        const auto& cur_e = edges_[i];
        // arma::vec eig_va = arma::normalise(point_eig_vecs_[cur_e[0]]) * point_eig_vals_[cur_e[0]];  
        // arma::vec eig_vb = arma::normalise(point_eig_vecs_[cur_e[1]]) * point_eig_vals_[cur_e[1]];
        arma::vec eig_va = point_eig_vecs_[cur_e[0]] ;  
        arma::vec eig_vb = point_eig_vecs_[cur_e[1]] ;
        if(arma::dot(eig_va, eig_vb) < 0)
        {
            eig_vb *= -1.0;
        }  
        double crease_val_a = arma::dot(eig_va, point_graidents_[cur_e[0]]);
        double crease_val_b = arma::dot(eig_vb, point_graidents_[cur_e[1]]);
        
        double dot_val = crease_val_a * crease_val_b;
        // std::cout << " edge pid " << cur_e[0] << " " << cur_e[1] << "  dot val " << dot_val << std::endl;
        if(dot_val < 0)
        {
            const auto& pa = mesh_points_[cur_e[0]]; 
            const auto& pb = mesh_points_[cur_e[1]];
            // double va = crease_values_[cur_e[0]];
            // double vb = crease_values_[cur_e[1]];
            double len = abs(crease_val_a) + abs(crease_val_b);
            if(len > 1e-20)
            {
                double t = (- crease_val_a) / (crease_val_b - crease_val_a);
                double px = pa[0] + t * (pb[0] - pa[0]);
                double py = pa[1] + t * (pb[1] - pa[1]);
                double pz = pa[2] + t * (pb[2] - pa[2]);
                
                // std::cout << " interp p id " << interp_p_id << std::endl;
                if(hrfb_ptr_)
                {
                    arma::mat hessian = arma::zeros(3, 3);
                    hrfb_ptr_->EvaluateHessian(px, py, pz, hessian);
                    // hessian = computeHessian(hrfb_ptr_, )
                    arma::vec eigval;
                    arma::mat eigvec;
                    arma::mat H_sym = 0.5 * (hessian + hessian.t());
                    arma::eig_sym(eigval, eigvec, H_sym);
                    arma::vec eigval_abs = arma::abs(eigval);
                    arma::uvec val_ids =  arma::sort_index(eigval_abs);
                    size_t max_id = arma::index_max(eigval_abs);
                    
                    // double ratio = abs(eigval_abs[max_id]/arma::accu(eigval_abs));
                    double ratio = abs(eigval_abs[val_ids[2]]/eigval_abs[val_ids[1]]);
                    

                    if(eigval_abs[max_id] > 0.2)
                    {
                        Point cur_p{px, py, pz};
                        // std::cout << "eigen val size " << eigval.size() << std::endl;
                        TransformEclips(eigval, eigvec, cur_p);

                        edge_int_pts_.push_back({px, py, pz});
                        edge_signs_[i] = interp_p_id;
                        interp_p_id ++; 
                        edge_eig_vals_.push_back(eigval[max_id]);
                        edge_eig_abs_vals_.push_back(eigval_abs[max_id]);
                        edge_eig_val_ratios_.push_back(ratio);
                        double r_threshold = 1.6;
                        if(eigval[max_id] > 0)
                        {
                            edge_pt_color_.push_back({255, 0, 0});
                        } else {
                            edge_pt_color_.push_back({0, 0, 255});
                        }
                    }
                    // double ratio = eigval_abs[max_id];
                    
                }
            }
        }
    }
    return true;
}

bool VIPSSRidges::CalculateEdgeCreasePoint(const size_t pa, const size_t pb)
{
    return false;
}   


bool VIPSSRidges::CalculateFaceCreaseEdge(int f_id)
{
    const auto& cur_f = mesh_faces_[f_id];
    string token_ab = CalEdgeToken(cur_f[0], cur_f[1]);
    string token_bc = CalEdgeToken(cur_f[1], cur_f[2]);
    string token_ca = CalEdgeToken(cur_f[2], cur_f[0]);
    int sum_sign = 0; 
    std::vector<size_t> new_e_ids; 
    // if(edge_id_map_.find(token_ab) != edge_id_map_.end())
    {
        size_t e_ab_id = edge_id_map_[token_ab];
        if(edge_signs_[e_ab_id] >= 0)
        {
            sum_sign ++;
            new_e_ids.push_back(edge_signs_[e_ab_id]);
        }
    }
    // if(edge_id_map_.find(token_bc) != edge_id_map_.end())
    {
        size_t e_bc_id = edge_id_map_[token_bc];
        if(edge_signs_[e_bc_id] >= 0)
        {
            sum_sign ++;
            new_e_ids.push_back(edge_signs_[e_bc_id]);
        }
    }
    // if(edge_id_map_.find(token_ca) != edge_id_map_.end())
    {
        size_t e_ca_id = edge_id_map_[token_ca];
        if(edge_signs_[e_ca_id] >= 0)
        {
            sum_sign ++;
            new_e_ids.push_back(edge_signs_[e_ca_id]);
        }
    }
    if(sum_sign == 2)
    {
        ridge_edges_.push_back({new_e_ids[0], new_e_ids[1]});
    }
    if(sum_sign == 3)
    {
        const auto& pa = edge_int_pts_[new_e_ids[0]];
        const auto& pb = edge_int_pts_[new_e_ids[1]];
        const auto& pc = edge_int_pts_[new_e_ids[2]];
        double px = (pa[0] + pb[0] + pc[0])/3.0;
        double py = (pa[1] + pb[1] + pc[1])/3.0;
        double pz = (pa[2] + pb[2] + pc[2])/3.0;
        size_t new_pid = edge_int_pts_.size(); 
        edge_int_pts_.push_back({px, py, pz});
        ridge_edges_.push_back({new_e_ids[0], new_pid});
        ridge_edges_.push_back({new_e_ids[1], new_pid});
        ridge_edges_.push_back({new_e_ids[2], new_pid});
        double ratio = (edge_eig_val_ratios_[new_e_ids[0]] 
                        + edge_eig_val_ratios_[new_e_ids[1]] 
                        + edge_eig_val_ratios_[new_e_ids[2]]) / 3.0;
        edge_eig_val_ratios_.push_back(ratio);   
        
        double abs_val = (edge_eig_abs_vals_[new_e_ids[0]] 
                        + edge_eig_abs_vals_[new_e_ids[1]]
                        + edge_eig_abs_vals_[new_e_ids[2]]) / 3.0;
        edge_eig_abs_vals_.push_back(abs_val);  
    }
    return true;
}


void VIPSSRidges::BuildClusterMST()
{
    std::set<std::string> visited_edge_ids;
    std::vector<C_Edege> tree_edges;
    auto cmp = [](const C_Edege& left, const C_Edege& right) { return left.score_ > right.score_; };
    std::priority_queue<C_Edege, std::vector<C_Edege>, decltype(cmp)> edge_priority_queue(cmp);

    C_Edege st_e(0,0);
    edge_priority_queue.push(st_e);
    std::unordered_set<size_t> visited_vids;
    while(!edge_priority_queue.empty())
    {
        C_Edege cur_e = edge_priority_queue.top();
        edge_priority_queue.pop();

        if(visited_vids.find(cur_e.c_b_ ) != visited_vids.end()) continue;
        visited_vids.insert(cur_e.c_b_);
        if(cur_e.c_a_ != cur_e.c_b_)
        {
            tree_edges.push_back(cur_e);
        }
        size_t cur_pid = cur_e.c_b_ ;
        const auto& p_n_ids = pt_adj_vec_[cur_pid];
        for(const auto& n_id : p_n_ids)
        {
            if(visited_vids.find(n_id) != visited_vids.end()) continue;
            if(n_id == cur_pid) continue;
            C_Edege edge(cur_pid, n_id);
            double nor_diff =1.0 - arma::dot(point_graidents_[cur_pid], point_graidents_[n_id]);

            double dist = PointDist(mesh_points_[n_id], mesh_points_[cur_pid]);
            edge.score_ = dist * nor_diff; 
            edge_priority_queue.push(edge);
        }
    }
    size_t p_num = mesh_points_.size();
    cluster_MST_mat_.resize(p_num, p_num);
    std::vector<TripletInt> edge_eles(tree_edges.size() *2);
    auto e_iter = edge_eles.begin();
    for(const auto& edge: tree_edges)
    {
        size_t i = edge.c_a_;
        size_t j = edge.c_b_;
        *(e_iter ++) = TripletInt(i,j,1);
        *(e_iter ++) = TripletInt(j,i,1);
    }
    cluster_MST_mat_.setFromTriplets(edge_eles.begin(), edge_eles.end());
    // cluster_scores_mat_.colw;
}


void VIPSSRidges::FlipEigenVectorByMST()
{
    // size_t c_num = cluster_cores_mat_.n_cols;
    size_t p_num = mesh_points_.size();
    std::queue<size_t> cluster_queued_ids;
    cluster_queued_ids.push(0);
    std::set<size_t> visited_cluster_ids;
    std::set<size_t> flipped_cluster_ids;
    flipped_cluster_ids.insert(0);
    while(!cluster_queued_ids.empty())
    {
        size_t cur_cid = cluster_queued_ids.front();
        cluster_queued_ids.pop();
        if(visited_cluster_ids.find(cur_cid) != visited_cluster_ids.end()) continue;
        for(SpiMat::InnerIterator iter(cluster_MST_mat_, cur_cid); iter ; ++iter)
        {   
            size_t n_cid = iter.row();
            if(flipped_cluster_ids.find(n_cid) != flipped_cluster_ids.end()) continue;
            flipped_cluster_ids.insert(n_cid);
            double dot_val = arma::dot(point_eig_vecs_[cur_cid], point_eig_vecs_[n_cid]);
            if(dot_val < 0)
            {
                point_eig_vecs_[n_cid] *= -1.0;
            }
            cluster_queued_ids.push(n_cid);
        }
    }
}

bool VIPSSRidges::CalculateRidgeEdgesFromMesh()
{
    if(point_graidents_.empty())
    {
        std::cout << "error:  No mesh grident!! " << std::endl;
        return false;
    }
    if(point_eig_vecs_.empty())
    {
        std::cout << "error:  No mesh point eigen vectors!! " << std::endl;
        return false;
    }
    BuildPtAdjInfo();
    // FlipEigenVector();
    BuildClusterMST();
    FlipEigenVectorByMST();

    std::string out_eigvec_path = out_dir_ + "/" + file_name_ + "_eig_vec_l" + std::to_string(user_lambda_) + ".xyz";
    SavePointsNormalToXYZ(out_eigvec_path, mesh_points_, point_eig_vecs_);
    std::string out_gradient_path = out_dir_ + "/" + file_name_ + "_gradient_l" + std::to_string(user_lambda_)+ ".xyz";
    SavePointsNormalToXYZ(out_gradient_path, mesh_points_, point_graidents_);

    std::cout << "start to CalculateCreaseValues !! " << std::endl;
    CalculateCreaseValues();
    std::cout << "start to CalculateEdgeCreasePoints  !! " << std::endl;
    GetEdges();
    CalculateEdgeCreasePoints();
    // std::string out_path =  "int_pts.obj";
    // SaveRidgesToObj(out_path);
    size_t face_num = mesh_faces_.size();
    std::cout << "start to CalculateFaceCreaseEdge  !! " << std::endl;
    for(int i = 0; i < face_num; ++i)
    {
        CalculateFaceCreaseEdge(i);
    }
    std::cout << "finished  CalculateFaceCreaseEdge  !! " << std::endl;
    return true;
}

void VIPSSRidges::BuildPtAdjInfo()
{
    size_t f_num = mesh_faces_.size();
    size_t p_num = mesh_points_.size();
    pt_adj_vec_.resize(p_num);
    for(int i = 0; i < int(f_num); ++i)
    {
        const auto& cur_f = mesh_faces_[i];
        for(const auto pid : cur_f)
        {
            for(const auto new_pid : cur_f)
            {
                pt_adj_vec_[pid].insert(new_pid);
            }
        }
    }
}

void VIPSSRidges::FlipEigenVector()
{
    std::vector<int> pt_visited;
    size_t p_num = mesh_points_.size();
    pt_visited.resize(p_num);
    std::stack<size_t> candid_pids;
    candid_pids.push(0);
    pt_visited[0] = 1;
    while(!candid_pids.empty())
    {
        size_t cur_pid = candid_pids.top();
        candid_pids.pop();
        const auto& neis = pt_adj_vec_[cur_pid];
        const Vec& cur_eigvec = point_eig_vecs_[cur_pid];
        for(const auto n_id : neis)
        {
            if(pt_visited[n_id]) continue;
            Vec& n_eigvec = point_eig_vecs_[n_id];
            if(arma::dot(cur_eigvec, n_eigvec) < 0)
            {
                n_eigvec *= -1.0;
            }
            candid_pids.push(n_id);
            pt_visited[n_id] = 1;
        }
    }
}

void VIPSSRidges::CalEdgePointQuality(LocalVipss* local_vipss)
{
    edge_points_quality_.resize(edge_int_pts_.size());
    for(int i = 0; i < edge_int_pts_.size(); ++i)
    {
        const auto& pt = edge_int_pts_[i]; 
        double cur_pt[3] = {pt[0], pt[1], pt[2]};
        arma::mat hessian = local_vipss->NNHRBFHessianOMP(cur_pt) * (-1.0);
        // std::cout << hessian << std::endl;
        arma::vec eigval;
        arma::eig_sym(eigval, hessian);
        arma::uvec indices = arma::sort_index(eigval, "descend");
        // if(eigval[indices[0]] > abs(eigval[indices[2]]))
        if(eigval[indices[0]] > 0)
        {
            edge_points_quality_[i] = 1.0;
        } else {
            edge_points_quality_[i] = 0.0;
        }
    }
}

void VIPSSRidges::SaveRidgesToObj(const std::string& out_path)
{
    std::ofstream objFile(out_path);
    if (!objFile) {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return;
    }
    // Write vertices
    for (const auto& point : edge_int_pts_) {
        double px = point[0] * scale_ + ori_center_[0];
        double py = point[1] * scale_ + ori_center_[1];
        double pz = point[2] * scale_ + ori_center_[2];
        objFile << "v " << px << " " << py << " " << pz << "\n";
    }
    // Write edges as line elements (OBJ uses 'l' for lines)
    for (const auto& edge : ridge_edges_) {
        objFile << "l " << edge[0] + 1 << " " << edge[1] + 1 << "\n";  // OBJ uses 1-based indexing
    }
    objFile.close();
    std::cout << "OBJ file saved successfully: " << out_path << std::endl;
}


void VIPSSRidges::SaveRidgesWithColorToPLY(const std::string& filename) {
    //  const std::vector<Point>& points, const std::vector<Edge>& edges
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << edge_int_pts_.size() + ridge_edges_.size()<< "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "element face " << ridge_edges_.size() << "\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";

    // std::cout << "edge_point_types_ size " << edge_point_types_.size() << std::endl;
    // Write vertex data
    for (int i =0; i < edge_int_pts_.size(); ++i) {
        const auto& point = edge_int_pts_[i];
        double px = point[0] * scale_ + ori_center_[0];
        double py = point[1] * scale_ + ori_center_[1];
        double pz = point[2] * scale_ + ori_center_[2];
        file << px << " " << py << " " << pz << " " ;
        const auto& color = edge_pt_color_[i];
        file << color[0] << " " << color[1] << " " << color[2] << " "  << "\n";
    }
    // Write edge data
    size_t pid = edge_int_pts_.size();
    std::vector<std::array<size_t,3>> faces;
    for (const auto& e : ridge_edges_) {
        
        faces.push_back({e[0], e[1], pid});
        pid ++;
        const auto& p0 = edge_int_pts_[e[0]];
        const auto& p1 = edge_int_pts_[e[1]];
        double px = (p0[0] + p1[0])/ 2.0; 
        double py = (p0[1] + p1[1])/ 2.0; 
        double pz = (p0[2] + p1[2])/ 2.0; 
        file << px << " " << py << " " << pz << " " ;
        double e_v0 = edge_eig_vals_[e[0]];
        double e_v1 = edge_eig_vals_[e[1]];
        std::string color_str = "0 0 0";

        const auto& color0 = edge_pt_color_[e[0]];
        const auto& color1 = edge_pt_color_[e[1]];
        if(abs(e_v0) >= abs(e_v1))
        {
            file << color0[0] << " " << color0[1] << " " << color0[2] << " " << "\n";
        } else {
            file << color1[0] << " " << color1[1] << " " << color1[2] <<" "  << "\n";
        }
    }

    for(const auto& face : faces)
    {
        file << "3 " << face[0] << " " << face[1] << " " << face[2] << std::endl;
    }

    file.close();
    std::cout << "PLY file saved: " << filename << std::endl;
}



void VIPSSRidges::SaveRidgesWithQualityToPLY(const std::string& filename, const std::vector<double>& qualtity) {
    //  const std::vector<Point>& points, const std::vector<Edge>& edges
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << edge_int_pts_.size() + ridge_edges_.size()<< "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property float quality\n";
    file << "element face " << ridge_edges_.size() << "\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";
    // Write vertex data
    for (int i =0; i < edge_int_pts_.size(); ++i) {
        const auto& point = edge_int_pts_[i];
        double px = point[0] * scale_ + ori_center_[0];
        double py = point[1] * scale_ + ori_center_[1];
        double pz = point[2] * scale_ + ori_center_[2];
        file << px << " " << py << " " << pz << " " << qualtity[i] << "\n";
    }
    // Write edge data
    size_t pid = edge_int_pts_.size();
    std::vector<std::array<size_t,3>> faces;
    for (const auto& e : ridge_edges_) {
        
        faces.push_back({e[0], e[1], pid});
        pid ++;
        const auto& p0 = edge_int_pts_[e[0]];
        const auto& p1 = edge_int_pts_[e[1]];
        double px = (p0[0] + p1[0])/ 2.0; 
        double py = (p0[1] + p1[1])/ 2.0; 
        double pz = (p0[2] + p1[2])/ 2.0; 
        file << px << " " << py << " " << pz << " " ;
        double e_v0 = edge_eig_vals_[e[0]];
        double e_v1 = edge_eig_vals_[e[1]];


        const auto& color0 = edge_pt_color_[e[0]];
        const auto& color1 = edge_pt_color_[e[1]];
        if(abs(e_v0) >= abs(e_v1))
        {
            file << qualtity[e[0]]<< "\n";
        } else {
            file << qualtity[e[1]] << "\n";
        }
    }

    for(const auto& face : faces)
    {
        file << "3 " << face[0] << " " << face[1] << " " << face[2] << std::endl;
    }

    file.close();
    std::cout << "PLY file saved: " << filename << std::endl;
}


void VIPSSRidges::SaveMeshWithPointQuality(const std::string& mesh_path)
{
    std::vector<Point> out_points;
    for(const auto& p : mesh_points_)
    {
        double px = p[0] * scale_ + ori_center_[0];
        double py = p[1] * scale_ + ori_center_[1];
        double pz = p[2] * scale_ + ori_center_[2];
        out_points.push_back({px, py, pz});
    }
    SaveMeshWithQualityToPly(mesh_path, out_points, crease_values_, mesh_faces_);
}

void VIPSSRidges::SaveEigBallsMesh(const std::string& mesh_path)
{
    std::vector<Point> out_points;
    for(const auto& p : eig_ball_pts_)
    {
        double px = p[0] * scale_ + ori_center_[0];
        double py = p[1] * scale_ + ori_center_[1];
        double pz = p[2] * scale_ + ori_center_[2];
        out_points.push_back({px, py, pz});
    }
    // SaveMeshToPly(mesh_path, out_points, eig_ball_faces_);
    SaveMeshToPly(mesh_path, out_points, eig_ball_pts_quality_, eig_ball_faces_);
}



void VIPSSRidges::SavePointsNormalToXYZ(const std::string& out_path, 
                                const std::vector<Point>& points,
                                const std::vector<Vec>& normals)
{
    //  const std::vector<Point>& points, const std::vector<Edge>& edges
    std::ofstream file(out_path);
    if (!file) {
        std::cerr << "Error: Unable to open file " << out_path << std::endl;
        return;
    }
    // Write vertex data
    for (int i =0; i < points.size(); ++i) {
        const auto& point = points[i];
        double px = point[0] * scale_ + ori_center_[0];
        double py = point[1] * scale_ + ori_center_[1];
        double pz = point[2] * scale_ + ori_center_[2];
        const auto& vec = normals[i];
        file << px << " " << py << " " << pz  << " " << vec[0] << " " << vec[1] << " " << vec[2]<< "\n";
    }
    file.close();
    std::cout << "PLY file saved: " << out_path << std::endl;
}

void VIPSSRidges::TransformEclips(const arma::vec& eigvals, const arma::mat& eigen_vectors, const Point& cur_pt)
{
    // arma::mat rotation = eigen_vectors.t();
    // arma::mat rotation = eigen_vectors.t();
    // rotation.col(2) = arma::normalise(arma::cross(rotation.col(0), rotation.col(1)));
    // rotation = arma::inv(rotation);
    // Orthonormalize eigenvectors using QR decomposition
    // std::cout << " eigen vector mat " << eigen_vectors << std::endl;
    arma::mat Q, R;
    arma::qr(Q, R, eigen_vectors);

    // Ensure determinant is +1 for a proper rotation matrix
    if (det(Q) < 0) {
        Q.col(2) *= -1;  // Flip the last column
    }
    // std::cout << " Q vector mat " << Q << std::endl;
    // arma::mat rotation = arma::inv(Q); 
    arma::mat rotation = Q;

    // std::cout << " transform mat " << rotation << std::endl;
    // std::cout << " xy dot " << arma::dot(rotation.col(0), rotation.col(1)) << std::endl;
    // std::cout << " yz dot " << arma::dot(rotation.col(1), rotation.col(2)) << std::endl;
    // std::cout << " zx dot " << arma::dot(rotation.col(2), rotation.col(0)) << std::endl;
    // std::vector<Point> final_pts;
    arma::vec center{cur_pt[0], cur_pt[1], cur_pt[2]};
    double g_scale = 0.002;
    // arma::vec scale = arma::abs(eigvals) / arma::max(arma::abs(eigvals)) * g_scale;
    arma::vec scale = arma::abs(eigvals) / 20.0 * g_scale;
    arma::mat scale_mat = arma::diagmat(scale);
    double ratio = arma::max(arma::abs(eigvals)) / arma::accu(arma::abs(eigvals));
    // std::cout << " scale mat " << scale_mat << std::endl;
    size_t pt_size = eig_ball_pts_.size(); 
    for(const auto& pt: ball_pts_)
    {
        arma::vec in_pt{pt[0], pt[1], pt[2]};
        arma::vec out_pt =  (rotation * (scale_mat * in_pt)) + center;
        eig_ball_pts_.push_back({out_pt[0], out_pt[1], out_pt[2]});
        eig_ball_pts_quality_.push_back(ratio);
    }
    for(const auto& face:  ball_faces_)
    {
        std::vector<size_t> new_face;
        for(const auto pid : face)
        {
            new_face.push_back(pid + pt_size);
        }
        eig_ball_faces_.push_back(new_face);
    }
    // return final_pts;
}

