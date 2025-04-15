#pragma once 
#include <vector>
#include <string>
#include <array>
#include "local_vipss.hpp"
#include <unordered_map>
#include <unordered_set>



class VIPSSRidges{
    // typedef double* Point; 
    enum PointType {
        NonePt, 
        RidgePt,
        ValleyPt 
    };
    typedef std::array<double,3> Point;
    typedef std::array<size_t,3> TriFace;
    typedef arma::vec3 Vec; 
    struct Edge{
        int a;
        int b;
        Edge(){};
        Edge(int a_, int b_){
            if ( a < b)
            {
                a_ = a;
                b_ = b;
            } else {
                a_ = b;
                b_ = a;
            }
        };
    };
    // typedef std::array<double,3> Point;
    public:
    bool LoadMeshPly(const std::string & mesh_path);
    bool ProcessFaces();
    bool CalculateCreaseValues();
    bool CalculateEdgeCreasePoints();
    bool CalculateEdgeCreasePoint(const size_t pa, const size_t pb);
    bool CalculateFaceCreaseEdge(int f_id);
    bool CalculateRidgeEdgesFromMesh();

    bool CalMeshPointsGradientAndEigenVecs(std::shared_ptr<RBF_Core> rfb_ptr);

    bool CalMeshPointsGradientAndEigenVecs(LocalVipss* local_vipss);
    void GetEdges();
    void SaveRidgesToObj(const std::string& out_path);
    void SaveRidgesWithColorToObj(const std::string& objFile, const std::string& mtlFile);
    void CalEdgePointQuality(LocalVipss* local_vipss);
    void SaveRidgesWithQualityToPLY(const std::string& filename, const std::vector<double>& qualtity); 
    void SaveRidgesWithColorToPLY(const std::string& filename);
    void SetDataCenterAndScale(const Point& center, const double scale); 
    // Point IterpolateEdgesPt(const Point& pa, const Point& pb, double va, double vb, double inter_val = 0);
    void BuildPtAdjInfo();
    void FlipEigenVector();
    void SavePointsNormalToXYZ(const std::string& out_path, 
                                const std::vector<Point>& points,
                                const std::vector<Vec>& normals);
    void BuildClusterMST();
    void FlipEigenVectorByMST();
    void SaveMeshWithPointQuality(const std::string& mesh_path);
    void SaveEigBallsMesh(const std::string& mesh_path);
    arma::mat computeHessian(std::shared_ptr<RBF_Core> rfb_ptr, Point x, double h = 1e-5);
    void TransformEclips(const arma::vec& eigvals, const arma::mat& eigen_vectors, const Point& cur_pt); 

    public:
    LocalVipss* local_vipss_ = nullptr;
    std::shared_ptr<RBF_Core> hrfb_ptr_; 
    std::vector<Point> mesh_points_;
    std::vector<TriFace> mesh_faces_;
    std::vector<double> ridge_points_;
    // std::vector<Edge> edges_;
    std::vector<std::vector<size_t>> edges_;
    std::unordered_map<string, size_t> edge_id_map_;
    // std::unordered_map<size_t, Point> edge_int_points;
    std::vector<int> edge_signs_;
    std::vector<Point> edge_int_pts_;
    std::vector<Vec> point_graidents_;
    std::vector<Vec> point_eig_vecs_;
    std::vector<Vec> point_eig_vals_;
    std::vector<double> crease_values_;
    std::vector<double> edge_points_quality_;
    std::vector<std::vector<size_t>> ridge_edges_;
    std::vector<std::unordered_set<size_t>> pt_adj_vec_;
    std::vector<Point> eig_ball_pts_;
    std::vector<double> eig_ball_pts_quality_;
    std::vector<std::vector<size_t>> eig_ball_faces_;

    std::vector<Point> ball_pts_;
    std::vector<std::vector<size_t>> ball_faces_;
    // std::vector<PointType> edge_point_types_;
    std::vector<double> edge_eig_vals_;
    std::vector<double> edge_eig_val_ratios_;
    std::vector<double> edge_eig_abs_vals_;
    std::vector<std::array<int,3>> edge_pt_color_;


    SpiMat cluster_MST_mat_;

    Point ori_center_ = {0, 0, 0};
    double scale_ = 1.0;
    
    std::string out_dir_ = "./";
    std::string file_name_ = "";
    double user_lambda_ = 0.0;

};