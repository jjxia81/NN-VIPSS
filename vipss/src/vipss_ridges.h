#pragma once 
#include <vector>
#include <string>
#include <array>
#include "local_vipss.hpp"
#include <unordered_map>
#include <unordered_set>


string CalEdgeToken(int a, int b);


struct PEdge{
        int a_;
        int b_;
        PEdge(){};
        PEdge(int a, int b){
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

struct PrincipleCurvature{
        double k1_, k2_ = 0;
        double kmax_, kmin_ = 0;
        arma::vec3 t1_, t2_;
        arma::vec3 tmax_, tmin_;
        double e1_, e2_ = 0;
        double emax_, emin_ = 0;
        double demax_, demin_ = 0;
        double gaussain_e = 0;
        arma::vec3 pt_;
        arma::vec3 de1_;
        arma::vec3 de2_;
        PrincipleCurvature(){}
        PrincipleCurvature(double k1, double k2, const arma::vec3& t1, const arma::vec3& t2)
        : k1_(k1), k2_(k2), t1_(t1), t2_(t2)  {
            if(k1 > k2)
            {
                kmax_ = k1;
                kmin_ = k2;
                tmax_ = t1;
                tmin_ = t2;
            } else {
                kmax_ = k2;
                kmin_ = k1;
                tmax_ = t2;
                tmin_ = t1;
            }
        }
        void UpdateMax(){
            // k_max_ = abs(k1_) > abs(k2_) ? k1_ : k2_;
            // e_max_ = abs(k1_) > abs(k2_) ? e1_ : e2_;
            // t_max_ = abs(k1_) > abs(k2_) ? t1_ : t2_;
        }
    };

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
    

    
    // typedef std::array<double,3> Point;
    public:
    bool LoadMeshPly(const std::string & mesh_path);
    bool ProcessFaces();
    bool CalculateCreaseValues();
    bool CalculateEdgeCreasePoints();
    bool CalculateEdgeCreasePoint(const size_t pa, const size_t pb);
    bool CalculateFaceCreaseEdge(int f_id);
    bool CalculateRidgeEdgesFromMesh();
    bool CalculateRidgeEdgesFromMesh2();

    bool CalMeshPointsGradientAndEigenVecs(std::shared_ptr<RBF_Core> rfb_ptr);

    bool CalMeshPointsGradientAndEigenVecs(LocalVipss* local_vipss);
    void GetEdges();
    void SaveRidgesToObj(const std::string& out_path);
    static void SaveRidgesToObj(const std::string& out_path, 
       const std::vector<Point>& edge_int_pts, 
       const std::vector<std::vector<size_t>>& ridge_edges, double scale, Point ori_center);
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

    Vec ComputeGradient(std::shared_ptr<RBF_Core> rfb_ptr, const Point& p, double h = 1e-6);
    arma::mat ComputeHessian(std::shared_ptr<RBF_Core> rfb_ptr, Point x, double h = 1e-6);
    void TransformEclips(const arma::vec& eigvals, const arma::mat& eigen_vectors, const Point& cur_pt); 
    void ComputePrincleCurvature(const Point& pt, std::shared_ptr<RBF_Core> rfb_ptr, 
                double& k1, double& k2, Vec& t1, Vec& t2);
    // double ComputeCurvatureDerivative(const Point& pt,const Vec& normal, 
    //         std::shared_ptr<RBF_Core> rfb_ptr, const Vec& t1, double h = 1e-6);
    static double ComputeCurvatureDerivative(const Point& pt, const Vec& normal, 
                                                const arma::mat& Hessian,
                                                const arma::cube& third_derivs,  
                                                std::shared_ptr<RBF_Core> rfb_ptr, 
                                                const Vec& t1);
                                                
    static double ComputeCurvatureSecondDerivative(const Point& pt, const Vec& gradient, 
                                                const arma::mat& Hessian,
                                                const arma::cube& third_derivs, 
                                                const std::vector<arma::cube>& fourth_derivs,
                                                std::shared_ptr<RBF_Core> rfb_ptr, 
                                                const Vec& t1,
                                                const double k1, 
                                                const double e1);

    static PrincipleCurvature ComputePrincipalCurvaturesMonga(const Vec& gradient, const arma::mat& hessian);
    static void ComputeThirdDerivatives(const Point& pt, std::shared_ptr<RBF_Core> rfb_ptr, arma::cube& third_derivs, double h = 1e-6); 
    static void ComputeFourthDerivatives(const Point& pt, std::shared_ptr<RBF_Core> rfb_ptr, 
                                std::vector<arma::cube>& third_derivs, double h = 1e-6); 
    
    // bool CalMeshPointsGradient(std::shared_ptr<RBF_Core> rfb_ptr);
    static bool CalMeshPointsGradient(std::shared_ptr<RBF_Core> rfb_ptr, 
                                        const std::vector<Point> &points,
                                        std::vector<Vec> &gradients);

    static bool CalMeshPointsCurvature(std::shared_ptr<RBF_Core> rfb_ptr,
                                        const std::vector<Point>& points,
                                        const std::vector<Vec>& gradients,
                                        std::vector<PrincipleCurvature>& pt_curvatures);
    static void CalSinglePointCurvatureData(const Point& pt, PrincipleCurvature& curvature);
 
    
    // bool CalMeshPointsCurvature(std::shared_ptr<RBF_Core> rfb_ptr);
    // bool CalMeshPointsCurvatureDerivatives(std::shared_ptr<RBF_Core> rfb_ptr);
    bool CalculateEdgeRidgeValleyPoints();
    bool CalculateCrestPoints(const Point& pa, const Point& pb, 
                        const PrincipleCurvature& ca, const PrincipleCurvature& cb,
                    int& riges_sign, int& valley_sign, int& gaussian_sign);
    static bool CalculateCrestPointsSingle(const Point& pa, const Point& pb, 
                        const PrincipleCurvature& ca, const PrincipleCurvature& cb,
                        int& edge_emax_sign, Point& inter_pa, double& inter_cur_a,
                        int& edge_emin_sign, Point& inter_pb, double& inter_cur_b);

    static bool CalculateCrestPointsSingleQuadratic(const Point& pa, const Point& pb, 
                        const PrincipleCurvature& ca, const PrincipleCurvature& cb,
                        const PrincipleCurvature& c_mid,
                        int& edge_emax_sign, Point& inter_pa, double& inter_cur_a,
                        int& edge_emin_sign, Point& inter_pb, double& inter_cur_b);
    // static bool CalculateCrestPoints2(const Point& pa, const Point& pb, 
    //                     const PrincipleCurvature& ca, const PrincipleCurvature& cb,
    //                 int& inter_sign, Point& inter_pa);
    static bool CalculateCrestPointsSingleWithGrad(const Point& pa, const Point& pb, 
                        const PrincipleCurvature& ca, const PrincipleCurvature& cb,
                        int& edge_emax_sign, Point& inter_pa, double& inter_cur_a,
                        int& edge_emin_sign, Point& inter_pb, double& inter_cur_b);

    static bool CalculateRidegeEdges(const TriFace& cur_f, 
                        std::unordered_map<string, size_t>& edge_id_map,
                        const std::vector<int>& edge_signs,
                        std::vector<Point>& edge_int_pts, 
                        std::vector<double>& edge_pts_curvatures,
                        std::vector<std::vector<size_t>>& out_ridge_edges);

    void SaveMeshCurvaturesVisualResults(const std::string& out_dir);

    void ProjectMeshPtsToSurface(std::vector<Point>& mesh_points, std::shared_ptr<RBF_Core> hrfb_ptr);

    static void ExtractLevelSetCurvesOnMesh(const std::vector<Point>& mesh_points, 
                const std::vector<std::vector<size_t>>& mesh_faces, 
                const std::vector<double> &pt_vals,
                const string& curve_path, double level_val = 0);


    public:
    LocalVipss* local_vipss_ = nullptr;
    static std::shared_ptr<RBF_Core> hrfb_ptr_; 
    std::vector<Point> mesh_points_;
    std::vector<TriFace> mesh_faces_;
    std::vector<double> ridge_points_;
    std::vector<PrincipleCurvature> mesh_pt_curvatures_;


    std::vector<double> edge_ridge_pts_curvature_;
    std::vector<double> edge_valley_pts_curvature_;
    std::vector<double> edge_gaussian_curvature_;
    // std::vector<Edge> edges_;
    std::vector<std::vector<size_t>> edges_;
    std::unordered_map<string, size_t> edge_id_map_;
    // std::unordered_map<size_t, Point> edge_int_points;
    std::vector<int> edge_signs_;
    std::vector<int> edge_ridge_signs_;
    std::vector<int> edge_valley_signs_;
    std::vector<int> edge_gaussian_signs_;
    std::vector<Point> edge_int_pts_;
    std::vector<Point> edge_ridge_pts_;
    std::vector<Point> edge_valley_pts_;
    std::vector<Point> edge_gaussian_pts_;
    std::vector<std::vector<size_t>> out_ridge_edges_;
    std::vector<std::vector<size_t>> out_valley_edges_;
    std::vector<std::vector<size_t>> out_gaussian_edges_;


    int interp_ridge_p_id_ = 0;
    int interp_valley_p_id_ = 0;
    int interp_gaussian_p_id_ = 0;

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

    static std::shared_ptr<RBF_Core> g_hrfb_ptr;


    SpiMat cluster_MST_mat_;

    static Point ori_center_;
    static double scale_;
    
    static std::string out_dir_;
    std::string file_name_ = "";
    double user_lambda_ = 0.0;
    static std::vector<std::string> edge_curv_values_string; 
    static std::vector<std::vector<PrincipleCurvature>> edge_sample_curv_dataset; 

};

