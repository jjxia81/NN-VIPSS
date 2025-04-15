#pragma once
#include "rbfcore.h"
#include "voronoi_gen.h"
#include "local_vipss.hpp"
#include "Solver.h"
#include "vipss_ridges.h"

enum AXI_PlANE {
        XYZ,
        XZY,
        YZX
    };

enum HRBF_SURFACE_TYPE{
    GLOBAL_HRBF,
    LOCAL_HRBF_NN
};


// void WriteStatsLog(const std::string& path, const VP_STATS& vp_stats);


class VIPSSUnit {

    public:
        VIPSSUnit(){};
        ~VIPSSUnit(){};

        // void GetLocalVipssClusters();
        void InitPtNormalWithLocalVipss();
        // void BuildVipssUnitMatrixP();
        void OptUnitVipssNormalSimple();
        void OptUnitVipssNormalDirectSimple();
        void OptUnitVipssNormalDirect();
        void OptUnitVipssNormal();
        void ReconSurface();
        void ReconSurfaceHRBF(std::shared_ptr<RBF_Core> HRBF_ptr);
        void GenerateAdaptiveGrid();
        void Run();
        void RunRidges();
        void RunRidgesGHRBF();
        void test_partial_vipss();
        void BuildTetCentersMap();
        void BuildLocalHRBFPerNode();
        void CalEnergyWithGtNormal();

        void SolveOptimizaiton();
        void BuildNNHRBFFunctions();
        void CompareMeshDiff( std::shared_ptr<RBF_Core> rbf_func);
        void GenerateGridPts();
    private:
        void AdaptiveGridHRBF(std::shared_ptr<RBF_Core> rbf_func, 
                    std::vector<std::array<double, 3> >& output_vertices,
                    std::vector<std::array<size_t, 3> >& output_triangles);

    public:
        std::string file_name_;
        std::string data_dir_; 
        std::string out_dir_;
        std::string out_normal_path_;
        std::string out_surface_path_;
        std::string out_debug_path_;
        std::string input_data_path_;
        std::string input_data_ext_;
        std::string ridge_mesh_path_;
        std::string ridge_edges_save_path_;

        bool open_debug_ = false;
        bool is_surfacing_ = true;
        bool hard_constraints_ = true;
 
        //VoronoiGen voro_gen_;
        LocalVipss local_vipss_;
        RBF_API rbf_api_;
        // arma::sp_imat adjacent_mat_;
        arma::vec sg_;
        VIPSSRidges vipss_ridges_;
        
        // std::vector<arma::mat> local_M_vec_;
        // std::vector<arma::mat> local_J_vec_;
        // std::vector<arma::sp_mat> unit_matrix_vec_;
        // double user_lambda_ = 0;
        double user_lambda_ = 0;
        // arma::sp_mat Final_H_;
        size_t npt_; 
        Solution_Struct solver_;
        // std::vector<double> initnormals_;
        std::vector<double> newnormals_;
        std::vector<double> s_func_vals_;
        const double M_PI_ = 3.14159265358979323846;
        bool use_hrbf_surface_ = false;
        int volume_dim_ = 100;
        size_t countopt_ = 0;
        std::vector<arma::sp_mat> temp_Hs_;
        size_t constraint_count_ = 0; 
        AXI_PlANE axi_plane_ = AXI_PlANE::XYZ;

        bool init_with_cluster_merge_ = false;
        double merge_angle_ = 40;
        static int opt_func_count_g;
        static double opt_func_time_g;
        

        std::vector<std::shared_ptr<RBF_Core*>> node_rbf_vec;
        std::vector<double>finalMesh_v_;
        std::vector<uint>finalMesh_fv_;
        std::vector<std::array<double,3>> grid_pts_;
        std::vector<double> grid_pts_dist_vals_; 

        HRBF_SURFACE_TYPE hrbf_type_;

        double opt_tor_ = 1e-7;
        int max_opt_iter_ = 3000; 
        double adgrid_threshold_ = 0.001;
        bool use_adgrid_ = true;
        double distfunc_threshold_ = 0.3;
        bool make_nn_const_neighbor_num_ = true; 
        bool use_global_hrbf_ = false;
        bool use_efficient_memory_ = false;
        int soft_constraint_level_ = 0;
        double user_alpha_ = 50.0;
        bool use_input_normal_ = false;
        bool only_use_nn_hrbf_surface_ = false;
        double iso_offset_val_ = 0;
        
};