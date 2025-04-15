#pragma once
#include <unordered_set>
#include "rbfcore.h"
#include "voronoi_gen.h"
#include "sp_mat.h"



class C_Edege{
    public:
        C_Edege(size_t c_a, size_t c_b)
        {
            c_a_ = c_a;
            c_b_ = c_b; 
            // if(c_a > c_b)
            // {
            //     c_a_ = c_b;
            //     c_b_ = c_a;
            // }
            e_id_ = std::to_string(c_a_) + "_" + std::to_string(c_b_);
        };
        ~C_Edege () {};

    public:
        size_t c_a_;
        size_t c_b_;
        double score_;
        std::string e_id_;
};

struct SP_BBOX{
    double min_corner[3];
    double max_corner[3];
};

// inline std::vector<double> GetClusterSvalsFromIds(const std::vector<size_t>& pt_ids, 
//                         const std::vector<double>& all_svals);

// inline std::vector<double> GetClusterNormalsFromIds
//                             (const std::vector<size_t>& pt_ids, const std::vector<double>& all_normals); 
         
class LocalVipss {

    typedef tetgenmesh::point P3tr;
    // typedef Eigen::SparseMatrix<double> SpMat;
    // typedef Eigen::SparseMatrix<int> SpiMat;
    typedef Eigen::Triplet<double> Triplet;
    

    public:
        LocalVipss() {};
        ~ LocalVipss() {};

        void Init(const std::string & path, const std::string& ext);
        void Init(const std::vector<double>& in_pts);

        void BuildHRBFPerNode();
        // inline std::vector<double> GetClusterNormalsFromIds
        //                     (const std::vector<size_t>& pt_ids, const std::vector<double>& all_normals) const;

        // inline std::vector<double> GetClusterSvalsFromIds(const std::vector<size_t>& pt_ids, 
        //                 const std::vector<double>& all_svals) const; 
        inline std::vector<size_t> GetClusterCoreIds(size_t cluster_id) const;
        void InitNormalWithVipss();
        inline double CalculateClusterPairScore(size_t c_a, size_t c_b, bool& flip) const;
        // void BuildClusterAdjacentMat();
        void CalculateClusterNeiScores(bool is_init = false);

        void OuputPtN(const std::string& out_path, bool orient_normal = false);
        std::vector<double> GetInitPts() const;
        std::vector<double> GetInitNormals() const;

        inline bool IsFlipNormal(const arma::mat& a_normals, const arma::mat& b_normals) const;
        inline bool FlipClusterNormal(size_t c_a, size_t c_b) const;
        inline bool FlipClusterNormalSimple(size_t c_a, size_t c_b) const;
        inline bool FlipClusterNormalSimple2(size_t c_a, size_t c_b) const;
        inline bool FlipClusterNormal2(size_t c_a, size_t c_b) const;
        void BuildClusterMST(bool use_distance_weight);
        void FlipClusterNormalsByMST();
        void WriteVipssTimeLog();
        void SaveCluster();
        void SaveClusterPts(const std::string& path,
                            const std::vector<P3tr>& key_pts, 
                            const std::vector<P3tr>& nei_pts);
        void InitNormals();
        void InitNormalsWithSample();
        void GroupPtsWithVolume();
        void SamplePtsWithClusterAveScores();
        void SamplePtsWithOctree(const std::vector<double>& pts, int depth= 7);

    public:
        void BuidClusterCoresPtIds();

        inline double CalculateScores(const std::vector<arma::vec3>& a_normals, const std::vector<arma::vec3>& b_normals) const;
        inline double CalculateScores(const arma::mat& a_normals, const arma::mat& b_normals) const;

        inline void AddClusterHMatrix(const std::vector<int>& p_ids, const arma::mat& J_m,size_t npt);
        inline void AddClusterHMatrix(const std::vector<int>& p_ids, const arma::mat& J_m, size_t npt, std::vector<Triplet>& ele_vect );
        inline void AddClusterHMatrix(const std::vector<int>& p_ids, const arma::mat& J_m, size_t npt, 
                                    std::vector<Triplet>::iterator& ele_iter );
        inline void AddHalfClusterHMatrix(const std::vector<int>& p_ids, const arma::mat& J_m, size_t npt, 
                                    std::vector<Triplet>::iterator& ele_iter, std::vector<Triplet>::iterator& diagonal_ele_iter);
        void BuildMatrixH();
        void BuildMatrixHSparse();
        void BuildMatrixHBatches();
        void BuildMatrixHMemoryOpt();
        void SetBestNormalsWithHmat();
        // double NodeDistanceFunction(const tetgenmesh::point nn_pt, const tetgenmesh::point cur_pt) const;
        double NatureNeighborDistanceFunctionOMP(const tetgenmesh::point cur_pt) const;
        double NatureNeighborGradientOMP(const tetgenmesh::point cur_pt,double* gradient) const;
        arma::mat NNHRBFHessianOMP(const tetgenmesh::point cur_pt) const;
        static double NNDistFunction(const R3Pt &in_pt);  
        static std::array<double,3> NumericGradientFunction(const R3Pt &in_pt); 

        // double NNDistFunction(const double* in_pt);
        static double NNDistGradient(const R3Pt &in_pt, double* gradient); 
        void SetThis();      
        void VisualFuncValues(double (*function)(const R3Pt &in_pt), const VoroPlane& plane,
                              const std::string& dist_fuc_color_path);
        
        // void GroupClustersWithDegree();
        // void SaveGroupPtsWithColor(const std::string& path);
        // void PtPCA(std::vector<double>& pts);
        // void OptimizeAdjacentMat();

        // void InitAdjacentData();
        // void MergeHRBFClustersWithMap();
        // void BuildHRBFPerCluster();

        void ClearPartialMemory();
        void TestInsertPt();
        void TestVoronoiPts();
    
    public:
        static int DistCallNum;
        static int InitWithPartialVipss;
        static double DistCallTime;
        bool is_group_cluster_ = false;
        static VoronoiGen voro_gen_;
        RBF_API vipss_api_;

        std::array<double,3> in_pt_center_;
        double in_pt_scale_;

        // std::vector<std::unordered_set<size_t>> cluster_adjacent_ids_;
        // std::vector<std::unordered_set<size_t>> cluster_pt_ids_;
        // std::vector<std::vector<size_t> > cluster_core_pt_ids_vec_; 
        // arma::uvec cluster_core_pt_nums_;
        // arma::vec cluster_volume_vals_;
        // std::vector<size_t> cluster_id_map_;
        // std::vector<size_t> valid_cluster_dist_map_;
        // SpiMat cluster_adjacent_emat_;
        // SpiMat cluster_pt_emat_;
        // SpiMat cluster_cores_emat_;

        // arma::sp_umat adjacent_mat_;
        // arma::sp_umat cluster_cores_mat_;
        // arma::sp_umat cluster_adjacent_mat_;
        // arma::sp_umat cluster_adjacent_mat_opt_;
        // arma::sp_umat cluster_MST_mat_;
        // arma::sp_mat cluster_scores_mat_;
        // arma::sp_umat cluster_adjacent_pt_mat_;
        // arma::sp_mat cluster_normal_x_;
        // arma::sp_mat cluster_normal_y_;
        // arma::sp_mat cluster_normal_z_;
        SpMat cluster_scores_mat_;
        SpMat cluster_normal_x_;
        SpMat cluster_normal_y_;
        SpMat cluster_normal_z_;
        SpiMat cluster_MST_mat_;
        // arma::vec cluster_scores_ave_;

        // arma::sp_mat final_H_; 
        // arma::uvec cluster_valid_sign_vec_;

        static std::vector<tetgenmesh::point> points_; 
        // std::vector<double> normals_;
        std::vector<double> s_vals_;
        std::vector<double> s_vals_score_;
        std::vector<double> s_vals_dist_;
        // std::vector<std::vector<size_t>> cluster_core_pt_ids_;
        // static std::vector<std::vector<size_t>> cluster_all_pt_ids_;
        std::vector<tetgenmesh::point> cluster_centers_;
        size_t pt_num_;
        // std::vector<std::pair<int, double>> cluster_id_scores_;
        // std::set<size_t> update_score_cluster_ids_;
        std::vector<double> input_normals_;
        std::vector<double> out_pts_;
        std::vector<double> out_normals_;
        std::vector<double> out_normals_dist_;
        std::vector<std::pair<size_t, double>> vipss_time_stats_;
        std::vector<std::vector<std::pair<size_t, double>>> cluster_ptn_vipss_time_stats_;

        // std::vector<arma::sp_mat> cluster_pt_mat_vec_; 
        // std::vector<arma::mat> cluster_J_mat_vec_;
        // std::vector<arma::sp_mat> temp_H_vec_;
        const size_t temp_H_max_num_ = 2048;
        SpMat final_h_eigen_;
        
        std::set<P3tr> sample_cluster_pts_;
        static std::vector<std::shared_ptr<RBF_Core>> node_rbf_vec_;
        std::vector<double>finalMesh_v_;
        std::vector<uint>finalMesh_fv_;
        //arma::vec nn_dist_vec_;
        //arma::vec nn_volume_vec_;
        // SimOctree2::Point newbbox;
        // SimOctree::SimpleOctree octree_;
        std::vector<double> octree_leaf_pts_;
        std::vector<double> octree_split_leaf_pts_;
        // std::vector<double> origin_in_pts_;
        
    public:
        int max_group_iter_ = 12;
        int max_group_pt_num_ = 16;
        bool flip_normal_ = false;
        double user_lambda_ = 0.0;
        double unit_lambda_ = 0.0;
        const double M_PI2  = 2*acos(0.0);
        const double Anlge_PI_Rate = 180 / M_PI2;
        size_t max_iter_ = 30;
        bool use_hrbf_surface_ = false;
        int volume_dim_ = 100;
        int top_k_ = 5;
        double angle_threshold_= 45;
        size_t merged_cluster_size_ = 1;
        std::string out_dir_ = "./";
        std::string filename_ = "local_vipss"; 
        double build_j_time_total_ = 0;
        double build_h_time_total_ = 0;
        double build_h_time_total2_ = 0;
        std::vector<tetgenmesh::point> insert_pts_;

        static double pass_time_sum_;
        static double search_nn_time_sum_;
        double dist_time_sum_ = 0;
        size_t dist_call_num_ = 0;
        size_t in_cluster_surface_pt_count = 0;
        static int ave_voxel_nn_pt_num_; 
        static bool use_rbf_base_; 
        static bool use_octree_sample_;
        // bool use_octree_sample_ = true; 
        int octree_sample_depth_ = 7;
        double tet_gen_triangulation_time_;
        double tet_build_adj_mat_time_;
        double dummy_sign_ = 0;
        int min_batch_size_ = 10000;
        bool input_has_normal_ = true;
};