#pragma once
#include "tetgen.h"
#include <string>
#include <vector>
#include "rbf_api.h"
#include <unordered_map>
#include <unordered_set>
#include <set>
#include "pico_tree.h"
// #include <eigen3/Eigen/Dense>

typedef std::set<tetgenmesh::point> P_Set; 
typedef std::unordered_map<tetgenmesh::point, arma::vec3> PtNCluster;


// struct PtCluster{
//     P_Set key_pts;
//     P_Set group_pts;
//     std::set<PtCluster*> adjacent_clusters;
//     PtNCluster pt_normal_map;
//     bool merged = false;
//     PtCluster* merged_cluster;

//     void UpdateAdjacentClusters(); 
// };

struct VoroPlane{
    double px, py, pz;
    double nx, ny, nz;
    double intPx, intPy, intPz;
    VoroPlane(){};

    VoroPlane(double in_px, double in_py, double in_pz, double in_nx, double in_ny, double in_nz)
    {
        px = in_px; py = in_py; pz = in_pz;
        nx = in_nx; ny = in_ny; nz = in_nz; 
    };

    VoroPlane(double* p0, double* p1, double* p2)
    {
        arma::vec3 e01;
        arma::vec3 e02;
        e01[0] = p1[0] - p0[0];
        e01[1] = p1[1] - p0[1];
        e01[2] = p1[2] - p0[2];
        e02[0] = p2[0] - p0[0];
        e02[1] = p2[1] - p0[1];
        e02[2] = p2[2] - p0[2];
        px = (p0[0] + p1[0] + p2[0]) / 3.0;
        py = (p0[1] + p1[1] + p2[1]) / 3.0;
        pz = (p0[2] + p1[2] + p2[2]) / 3.0;

        arma::vec3 pn = arma::cross(e01, e02);
        pn = arma::normalise(pn);
        nx = pn[0];
        ny = pn[1];
        nz = pn[2];

        double f_step = 0.002;
        px += nx * f_step;
        py += ny * f_step;
        pz += nz * f_step;
    }

    void SavePlane(const std::string& outpath);
};






inline double PtDistance(double* p1, double* p2)
{
  return sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) +
              (p2[1] - p1[1]) * (p2[1] - p1[1]) +
              (p2[2] - p1[2]) * (p2[2] - p1[2]));
}

class VoronoiGen{

    enum EdgeSign {
        IN_UNION,
        INTERSECT,
        OUT_UNION,
        IN_VALID
    };

    inline double PtDistance(double* p1, double* p2)
    {
        return sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) +
                    (p2[1] - p1[1]) * (p2[1] - p1[1]) +
                    (p2[2] - p1[2]) * (p2[2] - p1[2]));
    }

    public:

        VoronoiGen() {};
        ~VoronoiGen() {};
        void loadData(const std::string& path );
        void loadData(const std::vector<double>& in_pts);
        void InitMesh();
        void Tetrahedralize();
        void BuildAdjecentMat();
        void BuildAdjecentMatKNN();
        void GetVertexStar(tetgenmesh::point& p_st, std::set<tetgenmesh::point>& candid_pts, int level);
        void BuildPtIdMap();
        void GenerateVoroData();
        void InsertPt(tetgenmesh::point pt);

        void GetVoroCellEdgeList(tetgenmesh::point nei_pt, std::vector<tetgenio::voroedge*>& edge_list);
        void GetVoroCellPtAndEdgeIdList(tetgenmesh::point nei_pt, std::set<int>& vcell_pt_list, std::set<int>& vcell_edge_list);
        void CalVoroCellPtSign(const VoroPlane &plane, const std::set<int>& vcell_pt_list, std::vector<int>& positive_pids);
        // void CalVoroEdgeIntersection(tetgenmesh::point nei_pt, const VoroPlane& plane, std::vector<double>& intersect_pts);
        void CalVoroEdgeIntersection(const std::set<int>& cell_eids, std::vector<double>& intersect_pts);
        void GetVoroCellPtList(std::vector<tetgenio::voroedge*>& edge_list, std::vector<double>& pt_list);
        std::vector<double> GetPolygonSplitPts(tetgenmesh::point cur_pt, tetgenmesh::point nei_pt);
        bool CalPlaneEdgeIntersection(tetgenio::voroedge* edge, VoroPlane &plane);
        void OutputVoronisMesh();
        void InsertBoundryPts();
        void InsertSphereBoundryPts();
        void InsertSphereBoundryPts42();
        void GetVoronoiNeiPts(tetgenmesh::point pt, std::vector<tetgenmesh::point>& candid_pts);
        void BuildTetMeshTetCenterMap();
        void BuildPicoTree();
        tetgenmesh::tetrahedron* GetClosetTet(double x, double y, double z);
        double CalTruncatedCellVolumePassOMP(tetgenmesh::point in_pt, tetgenmesh::point nei_pt, int thread_id = 0);
        double CalTruncatedCellVolumeGradientOMP(tetgenmesh::point in_pt, tetgenmesh::point nei_pt, 
                                                    arma::vec3& gradient, int thread_id);
        void OutputTetMesh(const std::string& tetmesh_path);
        void SetInsertBoundaryPtsToUnused();
        void SetInsertBoundaryPtsToUsed();
        void InsertPts(const std::vector< std::array<double,3>>& insert_pts);

        bool InsideTet(tetgenmesh::point search_pt, tetgenmesh::point p0, tetgenmesh::point p1, tetgenmesh::point p2, tetgenmesh::point p3);
                
    public:
        size_t pt_num_;
        tetgenio tetIO_;
        tetgenio tetInsertIO_;
        tetgenmesh tetMesh_;
        tetgenbehavior tet_beha_;
        tetgenmesh::arraypool *tetlist_, *ptlist_;
        tetgenmesh::arraypool *tetlist2_, *ptlist2_;
        tetgenmesh::arraypool *tetlist3_, *ptlist3_;
        std::vector<double> tet_center_pts_;
        std::unordered_map<int, tetgenmesh::tetrahedron*> tc_pt_tet_map_;
        std::vector<tetgenmesh::point> points_;
        // std::unordered_set<tetgenmesh::point> insert_boundary_pts_;
        std::vector<tetgenmesh::point> insert_boundary_pts_;
        std::unordered_map<tetgenmesh::point, double> dummy_pt_dist_vals_map_;

        // std::unordered_map<int, double> vcell_pt_sign_values_;
        std::vector<double> vpt_sign_vals_;
        std::vector<int> edge_insect_symbols_;
        std::vector<double> edge_insect_pts_;
        std::vector<arma::vec3> vcell_face_centers_; 
        std::vector<arma::vec3> vcell_face_normals_; 
        double truncated_cell_center_[3];
        double truncated_face_center_[3];
        std::vector<tetgenmesh::point> convex_hull_pts_; 

        // static const int MAX_ELEMENT_ = 1000;
        // static const int MAX_THREAD_NUM_ = 32;
        // double* truncated_tets_omp_[MAX_THREAD_NUM_][MAX_ELEMENT_*2];
        // double truncated_centers_omp_[MAX_THREAD_NUM_][3];
        // double intersect_pts_omp_[MAX_THREAD_NUM_][MAX_ELEMENT_];
        // int face_tet_count_omp_[MAX_THREAD_NUM_][MAX_ELEMENT_];
        // double* truncated_tets_[MAX_THREAD_NUM_][MAX_ELEMENT_*4];

        RBF_API vipss_api_;
        tetgenio voronoi_data_;
        std::string filename_;
        std::string out_dir_;
        std::set<tetgenmesh::point> candidate_pts_;
        arma::sp_umat pt_adjecent_mat_;

        static std::unordered_map<tetgenmesh::point, int> point_id_map_;
        // std::unordered_map<tetgenmesh::point, P_Set> point_cluster_pts_map_;
        static std::vector<std::vector<int>> cluster_init_pids_;
        // static std::vector<std::vector<double>> cluster_init_pts_;
        static arma::ivec cluster_size_vec_;
        static arma::ivec cluster_accum_size_vec_;
        // Eigen::SparseMatrix<uint8_t>  
        // std::vector<double> vertices_;
        // std::vector<double> normals_;6532
        // std::vector<uint8_t> colors_;
        // double min_angle_ = 30.0;
        PicoTree pTree_;
        std::vector<std::vector<int>> vorocell_pids_;
        std::vector<std::vector<int>> vorocell_eids_;

        std::array<double, 3> bbox_min_;
        std::array<double, 3> bbox_max_;
        size_t average_neighbor_num_ = 0;
        std::array<double,3> dummy_sign_pt_;
        std::vector<std::array<double,3>> singular_pts_; 

        
    private:
        clock_t ts_[6];
        clock_t tv_[2];


};