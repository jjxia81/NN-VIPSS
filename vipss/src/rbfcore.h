#ifndef RBFCORE_H
#define RBFCORE_H




#include <iostream>
#include <vector>
#include "Solver.h"
#include "ImplicitedSurfacing.h"
//#include "eigen3/Eigen/Dense"
#include <armadillo>
#include <unordered_map>
// using namespace std;
typedef unsigned int uint;

enum RBF_INPUT{
    ON,
    ONandNORMAL,
    ALL,
    INandOUT,
};


enum RBF_METHOD{
    Variational,
    Variational_P,
    LS,
    LSinterp,
    Interp,
    RayleighQuotients,
    RayleighQuotients_P,
    RayleighQuotients_I,
    Hermite,
    Hermite_UnitNorm,
    Hermite_UnitNormal,
    Hermite_Tangent_UnitNorm,
    Hermite_Tangent_UnitNormal,
    HandCraft,
    Hermite_InitializationTest
};

enum RBF_InitMethod{
    GT_NORMAL,
    GlobalEigen,
    GlobalEigenWithMST,
    GlobalEigenWithGT,
    LocalEigen,
    IterativeEigen,
    ClusterEigen,
    Lamnbda_Search,
    GlobalMRF,
    Voronoi_Covariance,
    CNN,
    PCA,
    RBF_Init_EMPTY
};

enum RBF_Kernal{
    XCube,
    ThinSpline,
    XLinear,
    Gaussian,
};

class RBF_Paras{
public:
    RBF_METHOD Method = RBF_METHOD::Hermite_UnitNormal;
    RBF_Kernal Kernal = XCube;
    RBF_InitMethod InitMethod = Lamnbda_Search;
    bool isusesparse = false;
    int polyDeg = 1;
    double sigma = 0.9;
    double user_lamnbda = 0;
    double rangevalue = 0.001;
    double sparse_para = 1e-3;
    double Hermite_weight_smoothness = 0;
    double Hermite_ls_weight = 0;
    double Hermite_designcurve_weight = 0;
    double ClusterCut_percentage;
    double ClusterCut_LocalMax_percentage;
    int ClusterVisualMethod;
    double wDir,wOrt,wFlip,handcraft_sigma;
    RBF_Paras(RBF_METHOD Method, RBF_Kernal Kernal, int polyDeg, double sigma, double user_lamnbda,double rangevalue):\
        Method(Method),Kernal(Kernal),InitMethod(RBF_Init_EMPTY),polyDeg(polyDeg),sigma(sigma),user_lamnbda(user_lamnbda),rangevalue(rangevalue),Hermite_weight_smoothness(0){}
    RBF_Paras(){}
};



class RBF_Core{

public:

    size_t npt;
    size_t key_npt;
    int polyDeg = 1;
    int bsize;

    bool isinv = true;
    bool isnewformula = true;
    double User_Lamnbda = 0;
    bool open_debug_log = false;
    double user_beta = 1.0;
    double sigma = 0.9;
    double inv_sigma_squarex2 = 1;

    RBF_Kernal kernal = XCube;
    RBF_METHOD curMethod = RBF_METHOD::Hermite_UnitNormal;
    RBF_InitMethod curInitMethod = Lamnbda_Search;

    double rangevalue =0.001;
    double maxvalue = 10000;

    std::vector<double>pts;
    std::vector<double>normals;
    std::vector<double>tangents;
    std::vector<uint>edges;
    std::vector<int>labels;
    static int DistFuncCallNum;
    static double DistFuncCallTime;

public:

    std::vector<double>initnormals;
    std::vector<double>initnormals_uninorm;

public:
    std::vector<double>newnormals;
    std::vector<double>out_normals_;

public:
    std::vector<double>coff_cos;
    std::vector<double>coff_sin;


public:
    std::vector<double>finalMesh_v;
    std::vector<uint>finalMesh_fv;

public:

    arma::mat M;
    arma::mat N;

    arma::vec a;
    arma::vec b;

    arma::vec kern_;
    arma::vec kb_;

    arma::mat Minv;
    arma::mat P;
    arma::mat K;
    arma::mat bprey;
    arma::mat saveK;
    arma::mat saveK_finalH;
    arma::mat finalH;

    arma::mat RQ;

    arma::mat bigM;
    arma::mat bigMinv;
    arma::mat Ninv;
    arma::mat K00;
    arma::mat K01;
    arma::mat K11;
    arma::mat dI;


    bool isuse_sparse = false;
    double sparse_para = 1e-3;

public:
    std::unordered_map<int, std::string>mp_RBF_INITMETHOD;
    std::unordered_map<int, std::string>mp_RBF_METHOD;
    std::unordered_map<int, std::string>mp_RBF_Kernal;

public:

    double (*Kernal_Function)(const double x);

    double (*Kernal_Function_2p)(const double *p1, const double *p2);

    double (*P_Function_2p)(const double *p1, const double *p2);

public:

    bool isNewApprox;
    bool isHermite = true;
    double Hermite_weight_smoothness;
    double Hermite_ls_weight_inject, User_Lamnbda_inject;
    double Hermite_designcurve_weight;
    

    double ls_coef = 0;
    void (*Kernal_Gradient_Function_2p)(const double *p1, const double *p2, double *G);
    void (*Kernal_Hessian_Function_2p)(const double *p1, const double *p2, double *H);

private:
    std::vector<double>local_eigenBe, local_eigenEd, eigenBe, eigenEd, gtBe, gtEd;

private:
    std::vector<std::vector<double> >lamnbdaGlobal_Be,lamnbdaGlobal_Ed;
    std::vector<double>lamnbda_list_sa;
private:
    std::unordered_map<int,std::unordered_map<int, std::vector<double> > >mp_RBF_InitNormal;
    std::unordered_map<int,std::unordered_map<int, std::vector<double> > >mp_RBF_OptNormal;

public:
    RBF_Core();
    ~RBF_Core(){};
    RBF_Core(RBF_Kernal kernal);
    void Init(RBF_Kernal kernal);
    void EstimateNormals();

    std::vector<double> EstimateNormals(const std::vector<double>& pts);

public:
    double Dist_Function(const double x, const double y, const double z);
    double evaluate_gradient(double x, double y, double z, double &gx, double &gy, double &gz) ;
    void EvaluateHessian(double x, double y, double z, arma::mat& hessian) ;

    inline double Dist_Function(const double *p);

public:
    static double Dist_Function(const R3Pt &in_pt);
    //static FT Dist_Function(const Point_3 in_pt);
    int n_evacalls;
public:
    void SetThis();
public:
    void SetSigma(double x);

public:


    void NormalRecification(double maxlen, std::vector<double> &nors);



public:
    void Set_HermiteRBF(std::vector<double>&pts);
    void Set_HermiteRBF(const std::vector<double*>&pts);
    void BuildUnitVipssMat(std::vector<double>&pts);
    int Solve_HermiteRBF(std::vector<double>&vn);

public:
    void Set_Hermite_PredictNormal(std::vector<double>&pts);

public:

    int Solve_Hermite_PredictNormal_UnitNorm();


    int Lamnbda_Search_GlobalEigen();


public:
    int Solve_Hermite_PredictNormal_UnitNormal();

public:

    void Print_LamnbdaSearchTest(std::string fname);



public:


    void Set_RBFCoef(arma::vec &y);
    void Set_RBFCoefWithInitNormal();
    void Set_RBFCoefWithOptNormalAndSval(const std::vector<double>& Vn, 
                                                const std::vector<double>& s_vals );
    void Solve_RBFCoefWithOptNormalAndSval(const std::vector<double>& Vn, 
                                                const std::vector<double>& s_vals );

    void Set_Actual_Hermite_LSCoef(double hermite_ls);
    void Set_HermiteApprox_Lamnda(double hermite_ls);
    void Set_Actual_User_LSCoef(double user_ls);
    void Set_User_Lamnda_ToMatrix(double user_ls);

    void Set_SparsePara(double spa);


public:

    bool Write_Hermite_NormalPrediction(std::string fname,int mode);
    bool Write_Hermite_MST(std::string fname);
    void WriteSeletivePLY(std::string fname, std::vector<double>&allnormals, std::vector<int>&pickInd);

    void SetInitnormal_Uninorm();
    void ClearSavedIterBatchInit();
    void SaveIterBatchInit(std::vector<double>&allnormals, std::vector<double>&allnormals_uninorm, std::vector<int>&pickInd);

    void Write_Surface(std::string fname);

public:

    int Opt_Hermite_PredictNormal_UnitNormal();
    void OptUnitVipssNormalDirect();

public:

    int ThreeStep(std::vector<double> &pts, std::vector<int> &labels, std::vector<double> &normals, std::vector<double> &tangents, std::vector<uint> &edges, RBF_Paras para);
    int AllStep(std::vector<double> &pts, std::vector<int> &labels, std::vector<double> &normals, std::vector<double> &tangents, std::vector<uint> &edges, RBF_Paras para);

    int InjectData(std::vector<double> &pts, std::vector<int> &labels, std::vector<double> &normals, std::vector<double> &tangents, std::vector<uint> &edges, RBF_Paras para);

    int InjectData(std::vector<double> &pts, RBF_Paras para);

    void BuildK(RBF_Paras para);
    void BuildK(double lambda);

    void InitNormal(RBF_Paras para);
    void InitNormal();

    void OptNormal(int method);

    void Surfacing(int method, int n_voxels_1d);

    //void BuildCoherentGraph();

    void BatchInitEnergyTest(std::vector<double> &pts, std::vector<int> &labels, std::vector<double> &normals, std::vector<double> &tangents, std::vector<uint> &edges, RBF_Paras para);


public:
    std::vector<double>* ExportPts();
    std::vector<double>* ExportPtsNormal(int normal_type);


    std::vector<double>* ExportInitNormal(int kmethod, RBF_InitMethod init_type);
    std::vector<double>* ExportOptNormal(int kmethod, RBF_InitMethod init_type);



public:

    Solution_Struct sol;

    std::vector<int>record_partition;
    std::vector<std::string>record_partition_name;

    std::vector<size_t>npoints;

    std::vector<std::string>record_initmethod;
    std::vector<std::string>record_method;
    std::vector<std::string>record_kernal;
    std::vector<double>record_initenergy;
    std::vector<double>record_energy;
    std::vector<double>record_time;

    double setup_time, init_time, solve_time, callfunc_time,invM_time, setK_time, surf_time;
    std::vector<double>setup_timev, init_timev, solve_timev, callfunc_timev,invM_timev,setK_timev;

    double bigM_inv_time;

    void Record();
    void Record(RBF_METHOD method, RBF_Kernal kernal, Solution_Struct &rsol, double time);
    void AddPartition(std::string pname);
    void Print_Record();
    void Print_TimerRecord(std::string fname);
    void Clear_TimerRecord();
    void Print_Record_Init();

    void Print_TimerRecord_Single(std::string fname);

};














#endif // RBFCORE_H
