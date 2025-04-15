#pragma once
#include "rbfcore.h"
#include <memory>

class RBF_API{
    public:
        RBF_API(){};
        ~RBF_API(){};
        void Set_RBF_PARA();
        void run_vipss(std::vector<double> &Vs);
        void run_vipss(std::vector<double> &Vs, size_t key_ptn);
        void run_vipss_for_incremental(std::vector<double> &Vs, size_t key_ptn);
        void run_vipss(std::vector<double> &Vs, std::vector<double> &Vn);
        void run_vipss(std::vector<double> &Vs, std::vector<double> &Vn, const std::vector<double>& s_vals);
        void build_unit_vipss(std::vector<double> &Vs); 
        void build_unit_vipss(std::vector<double> &Vs, size_t key_npt); 
        void build_cluster_hrbf(std::vector<double> &Vs, std::vector<double> &Vn, 
                                 const std::vector<double>& s_vals, std::shared_ptr<RBF_Core> rbf_ptr);

        void build_cluster_hrbf_surface(std::shared_ptr<RBF_Core> rbf_ptr, const std::string& mesh_path);

        
    inline std::vector<double> GetClusterNormalsFromIds
                            (const std::vector<int>& pt_ids, const std::vector<double>& all_normals) 
    {
        std::vector<double> normals(pt_ids.size() * 3);
        for(int i = 0; i < pt_ids.size(); ++i) 
        {
            // auto &pt = points_[id];
            auto id = pt_ids[i];
            normals[3*i] = (all_normals[3 * id]);
            normals[3*i + 1] = (all_normals[3 * id + 1]);
            normals[3*i + 2] = (all_normals[3 * id + 2]);
        }
        return normals;
    }

    

    public:

        static RBF_Paras para_;
        bool is_surfacing_ = false;
        bool is_outputtime_ = false;
        double user_lambda_ = 0.0; 
        double unit_lambda_ = 0.0; 
        int n_voxel_line_ = 100;
        std::vector<double> normals_;
        // size_t key_ptn_ = 0;
        RBF_Core rbf_core_;
        std::string outpath_ = "./vipss";
        std::string filename_ = "";
        double u_v_time = 0;
        std::vector<size_t> p_ids_;
        // std::vector<double> dist_vals_; 
        arma::vec auxi_dist_vec_; 
        std::vector<double> key_opt_normals_;
        std::vector<double> pre_out_normals_;
        bool use_input_normal_ = false;
        double incre_vipss_beta_ = 1.0;

        std::vector<std::shared_ptr<RBF_Core>> local_rbf_vec;

};

void InitNormalPartialVipss(std::vector<double> &Vs, size_t key_ptn, std::shared_ptr<RBF_Core> rfb_ptr, double lambda);

// double HRBF_Dist(const double* in_pt, const arma::vec& a, const arma::vec& b, 
//                 const std::vector<double>& cluster_pts);

double HRBF_Dist_Alone(const double* in_pt, const arma::vec& a, const arma::vec& b, 
                const std::vector<int>& cluster_pids, 
                const std::vector<double*>& all_pts);

double VIPSS_HRBF_Dist_Alone(const double* in_pt, const arma::vec& a, const arma::vec& b, 
                                const std::vector<std::array<double,3>>& all_pts);

void BuildGlobalHRBFVipss(std::vector<double> &Vs, std::shared_ptr<RBF_Core> rfb_ptr, double lambda);
