#include "rbf_api.h"
#include <chrono>
#include "kernel.h"

typedef std::chrono::high_resolution_clock Clock;


RBF_Paras RBF_API::para_;

void RBF_API::Set_RBF_PARA(){

    // std::cout << " Start to set rbf para ..... " << endl;
    RBF_Paras &para= para_;
    RBF_InitMethod initmethod = Lamnbda_Search;
    RBF_Kernal Kernal = XCube;
    int polyDeg = 1;
    double sigma = 0.9;
    double rangevalue = 0.001;

    para.Kernal = Kernal;para.polyDeg = polyDeg;
    para.sigma = sigma;para.rangevalue = rangevalue;
    para.Hermite_weight_smoothness = 0.0;
    para.Hermite_ls_weight = 0;
    para.Hermite_designcurve_weight = 0.0;
    para.Method = RBF_METHOD::Hermite_UnitNormal;
    // para.Method = RBF_METHOD::FastHermite;

    para.InitMethod = initmethod;
    para.user_lamnbda = 0;
    para.isusesparse = false;
    // return para;
}


void RBF_API::run_vipss(std::vector<double> &Vs)
{
    // vector<double>Vs;
    // std::cout << "start to set rbf para "<< std::endl;
    // RBF_Paras para = this->Set_RBF_PARA();
    // std::cout << Vs.size() << endl;
    para_.user_lamnbda = user_lambda_;
    // std::cout << "start inject data "<< std::endl;
    RBF_Core rbf_core_;
    rbf_core_.InjectData(Vs, para_);
    // std::cout << "finish inject data "<< std::endl;
    rbf_core_.BuildK(para_);
    // std::cout << "finish build K "<< std::endl;
    rbf_core_.InitNormal(para_);
    rbf_core_.OptNormal(0);
    
    
    if(is_surfacing_){
        rbf_core_.Write_Hermite_NormalPrediction(outpath_ + "_normal", 1);
        rbf_core_.Surfacing(0,n_voxel_line_);
        rbf_core_.Write_Surface(outpath_ +"_surface");
    }
    if(is_outputtime_){
        rbf_core_.Print_TimerRecord_Single(outpath_ +"_time.txt");
    }
    // printf("newnormals size : %zu -----------\n", rbf_core_.newnormals.size());
    normals_ = rbf_core_.newnormals;
    // rbf_core_.clear();
}

void RBF_API::run_vipss_for_incremental(std::vector<double> &Vs, size_t key_ptn)
{
    auto t0 = Clock::now();
    para_.user_lamnbda = user_lambda_;
    // RBF_Core rbf_core_;
    rbf_core_.user_beta = incre_vipss_beta_;
    rbf_core_.key_npt = key_ptn;
    rbf_core_.InjectData(Vs, para_);


    rbf_core_.BuildK(para_);
    // std::cout << "finish build K "<< std::endl;
    double build_time = (std::chrono::nanoseconds(Clock::now() - t0).count()/1e9);
    printf("vipss build %f \n", build_time);
    if(key_ptn == 1)
    {
        rbf_core_.Solve_Hermite_PredictNormal_UnitNorm();
        rbf_core_.Set_RBFCoefWithInitNormal();
    } else {
        if(!use_input_normal_)
        {
            rbf_core_.InitNormal(para_);
        }
        // rbf_core_.OptNormal(0);
        rbf_core_.OptUnitVipssNormalDirect();
    }
    auto t1 = Clock::now();

    double pre_time = (std::chrono::nanoseconds(t1 - t0).count()/1e9);
    printf("vipss build and opt time %f \n", pre_time);
    
    std::string init_path = outpath_ + std::to_string(key_ptn) + "normal_init.xyz";

    std::vector<double> key_vs;
    for(size_t i = 0; i < key_ptn *3; ++i)
    {
        key_vs.push_back(Vs[i]);
    }
    // writeXYZnormal(init_path, key_vs, rbf_core_.initnormals);
    std::string opt_path = outpath_ + std::to_string(key_ptn) + "normal_opt.xyz";
    writeXYZnormal(opt_path, key_vs, rbf_core_.newnormals);

    std::string all_opt_path = outpath_ + std::to_string(key_ptn) + "normal_all.xyz";

    rbf_core_.EstimateNormals();

    pre_out_normals_ = rbf_core_.out_normals_;
    key_opt_normals_ = rbf_core_.newnormals;

    // writeXYZnormal(all_opt_path, Vs, rbf_core_.out_normals_);

    
    
    if(is_surfacing_){
        // rbf_core_.Write_Hermite_NormalPrediction(outpath_ + std::to_string(key_ptn) + "_normal", 1);
        rbf_core_.Surfacing(0,n_voxel_line_);
        rbf_core_.Write_Surface(outpath_ + std::to_string(key_ptn) + "_surface");
    }
    if(is_outputtime_){
        rbf_core_.Print_TimerRecord_Single(outpath_ +"_time.txt");
    }
    // printf("newnormals size : %zu -----------\n", rbf_core_.newnormals.size());
    // printf("newnormals size : %zu -----------\n", rbf_core_.out_normals_.size());
    normals_ = rbf_core_.out_normals_;

    // p_ids_.clear();
    // dist_vals_.clear(); 
    size_t auxi_n = Vs.size()/3 - key_ptn; 
    if(auxi_n > 0)
    {
        auxi_dist_vec_.resize(auxi_n);
    }
    
    for(size_t i = 0; i < auxi_n; ++i)
    {
        size_t id = i + key_ptn;
        R3Pt new_pt(Vs[3*id], Vs[3*id +1], Vs[3*id + 2]);
        double dist_val = rbf_core_.Dist_Function(new_pt);
        // p_ids_.push_back(i);
        auxi_dist_vec_(i) = dist_val;
    }
}

void RBF_API::run_vipss(std::vector<double> &Vs, size_t key_ptn)
{
    auto t0 = Clock::now();
    para_.user_lamnbda = user_lambda_;
    // RBF_Core rbf_core_;
    rbf_core_.key_npt = key_ptn;
    rbf_core_.InjectData(Vs, para_);
    rbf_core_.BuildK(para_);
    // std::cout << "finish build K "<< std::endl;
    double build_time = (std::chrono::nanoseconds(Clock::now() - t0).count()/1e9);
    // printf("vipss build %f \n", build_time);
    if(key_ptn == 1)
    {
        rbf_core_.Solve_Hermite_PredictNormal_UnitNorm();
        rbf_core_.Set_RBFCoefWithInitNormal();
    } else {
        rbf_core_.InitNormal(para_);
        rbf_core_.OptNormal(0);
    }
    rbf_core_.EstimateNormals();
    normals_ = rbf_core_.out_normals_;
    auto t1 = Clock::now();
    double pre_time = (std::chrono::nanoseconds(t1 - t0).count()/1e9);
    // printf("vipss build and opt time %f \n", pre_time);
}


void RBF_API::run_vipss(std::vector<double> &Vs, std::vector<double> &Vn)
{
    para_.user_lamnbda = user_lambda_;
    std::cout << "start inject data "<< std::endl;
    RBF_Core rbf_core_;
    rbf_core_.User_Lamnbda = user_lambda_;

    printf("user lambda : %f \n", user_lambda_);
    // rbf_core_
    rbf_core_.key_npt = Vs.size()/3;
    rbf_core_.InjectData(Vs, para_);
    // std::cout << "finish inject data "<< std::endl;
    rbf_core_.BuildK(para_);
    // std::cout << "finish build K "<< std::endl;
    // rbf_core_.InitNormal(para_);
    rbf_core_.initnormals = Vn;
    rbf_core_.newnormals = Vn;
    // rbf_core_.OptNormal(0);
    rbf_core_.Set_RBFCoefWithInitNormal();



    if(is_surfacing_){
        // rbf_core_.Write_Hermite_NormalPrediction(outpath_ + "_normal", 1);
        rbf_core_.Surfacing(0,n_voxel_line_);
        rbf_core_.Write_Surface(outpath_ +"_surface");
    }
    if(is_outputtime_){
        rbf_core_.Print_TimerRecord_Single(outpath_ +"_time.txt");
    }
    normals_ = rbf_core_.newnormals;
}


void RBF_API::run_vipss(std::vector<double> &Vs, std::vector<double> &Vn, 
                                    const std::vector<double>& s_vals)
{
    auto t0 = Clock::now();
    para_.user_lamnbda = user_lambda_;
    std::cout << "start inject data "<< std::endl;
    RBF_Core rbf_core_;
    rbf_core_.User_Lamnbda = user_lambda_;
    // printf("user lambda : %f \n", user_lambda_);
    // rbf_core_
    rbf_core_.key_npt = Vs.size()/3;
    rbf_core_.npt = Vs.size()/3;
    rbf_core_.InjectData(Vs, para_);
    // std::cout << "finish inject data "<< std::endl;
    rbf_core_.Set_HermiteRBF(Vs);
    // std::cout << "finish build K "<< std::endl;
    // rbf_core_.InitNormal(para_);
    rbf_core_.initnormals = Vn;
    rbf_core_.newnormals = Vn;
    // rbf_core_.OptNormal(0);
    rbf_core_.Solve_RBFCoefWithOptNormalAndSval(Vn, s_vals);
    auto t1 = Clock::now();
    double t_time =  std::chrono::nanoseconds(t1 - t0).count()/1e9;
    printf("solve hrbf linear time: %f \n", t_time);

    if(is_surfacing_){
        // rbf_core_.Write_Hermite_NormalPrediction(outpath_ + "_normal", 1);
        RBF_Core::DistFuncCallNum = 0;
        RBF_Core::DistFuncCallTime = 0.0;
        std::cout << " n_voxel_line_ "<< n_voxel_line_ << std::endl;
        rbf_core_.Surfacing(0, n_voxel_line_);
        std::string save_path = outpath_  + filename_ +".ply";
        rbf_core_.Write_Surface( save_path);
        printf("number of call dist function : %d \n", RBF_Core::DistFuncCallNum);
        printf("time of call dist function : %f \n", RBF_Core::DistFuncCallTime);
    }
    if(is_outputtime_){
        rbf_core_.Print_TimerRecord_Single(outpath_ +"_time.txt");
    }
    normals_ = rbf_core_.newnormals;
}

void RBF_API::build_cluster_hrbf_surface(std::shared_ptr<RBF_Core> rbf_ptr, const std::string& mesh_path)
{

}

void RBF_API::build_cluster_hrbf(std::vector<double> &Vs, std::vector<double> &Vn, 
                                 const std::vector<double>& s_vals, std::shared_ptr<RBF_Core> rbf_ptr)
{
    auto t0 = Clock::now();
    // para_.user_lamnbda = user_lambda_;
    // std::cout << "start inject data "<< std::endl;
    // RBF_Core rbf_core_;
    if(rbf_ptr == NULL)
    {
        rbf_ptr = std::make_shared<RBF_Core>();
    }
    rbf_ptr->User_Lamnbda = user_lambda_;
    // printf("build hrbf user lambda : %f \n", user_lambda_);
    rbf_ptr->key_npt = Vn.size()/3;
    rbf_ptr->npt = Vs.size()/3;
    rbf_ptr->InjectData(Vs, para_);
    rbf_ptr->Set_HermiteRBF(Vs);

    // printf(" start to Solve_RBFCoefWithOptNormalAndSval\n");
    rbf_ptr->Solve_RBFCoefWithOptNormalAndSval(Vn, s_vals);
    auto t1 = Clock::now();
    double t_time =  std::chrono::nanoseconds(t1 - t0).count()/1e9;
    // printf("solve hrbf linear time: %f \n", t_time);
}

void RBF_API::build_unit_vipss(std::vector<double> &Vs)
{
    // auto t0 = Clock::now();

    para_.user_lamnbda = unit_lambda_;
    // std::cout << "start inject data "<< std::endl;
    rbf_core_.key_npt = Vs.size()/3;
    rbf_core_.InjectData(Vs, para_);
    // rbf_core_.npt = Vs.size()/3;
    auto t0 = Clock::now();
    rbf_core_.BuildUnitVipssMat(Vs);
    auto t1 = Clock::now();
    double t_time =  std::chrono::nanoseconds(t1 - t0).count()/1e9;
    u_v_time += t_time;
}

void RBF_API::build_unit_vipss(std::vector<double> &Vs, size_t key_npt)
{
    // auto t0 = Clock::now();

    para_.user_lamnbda = unit_lambda_;
    // std::cout << "start inject data "<< std::endl;
    rbf_core_.key_npt = key_npt;
    rbf_core_.InjectData(Vs, para_);
    // rbf_core_.npt = Vs.size()/3;
    auto t0 = Clock::now();
    rbf_core_.BuildUnitVipssMat(Vs);
    auto t1 = Clock::now();
    double t_time =  std::chrono::nanoseconds(t1 - t0).count()/1e9;
    u_v_time += t_time;
}


void InitNormalPartialVipss(std::vector<double> &Vs, size_t key_ptn, std::shared_ptr<RBF_Core> rfb_ptr, double lambda)
{
    auto t0 = Clock::now();
    // para_.user_lamnbda = lambda;
    // RBF_Core rbf_core_;
    rfb_ptr->key_npt = key_ptn;
    rfb_ptr->InjectData(Vs, RBF_API::para_);
    // rfb_ptr->User_Lamnbda = lambda;
    rfb_ptr->BuildK(lambda);
    double build_time = (std::chrono::nanoseconds(Clock::now() - t0).count()/1e9);
    if(key_ptn == 1)
    {
        rfb_ptr->Solve_Hermite_PredictNormal_UnitNorm();
        rfb_ptr->Set_RBFCoefWithInitNormal();
    } else {
        rfb_ptr->InitNormal();
        rfb_ptr->OptNormal(0);
    }
    rfb_ptr->EstimateNormals();
    
    auto t1 = Clock::now();
    double pre_time = (std::chrono::nanoseconds(t1 - t0).count()/1e9);
}

double HRBF_Dist_Alone(const double* in_pt, const arma::vec& a, const arma::vec& b, 
                const std::vector<int>& cluster_pids, 
                const std::vector<double*>& all_pts)
{
    int npt = cluster_pids.size();
    int key_npt = npt;
    double G[3];
    arma::vec kern(npt + 3 * key_npt);
    // std::cout << "  start dist pt size  " << npt << std::endl;
    // for(int i=0;i<npt;++i)
    // {
    //     std::cout << " p id : " << cluster_pids[i] << std::endl;
    //     auto cur_pt = all_pts[cluster_pids[i]];
    //     std::cout << cur_pt[0] << " " << cur_pt[1] << " " << cur_pt[2] << std::endl;
    // }

    for(int i=0;i<npt;++i) kern(i) = VIPSSKernel::XCube_Kernel_2p(all_pts[cluster_pids[i]], in_pt);
    // std::cout << "  XCube_Kernel_2p " << std::endl;
    for(int i=0;i<key_npt;++i){
        VIPSSKernel::XCube_Gradient_Kernel_2p(in_pt, all_pts[cluster_pids[i]],G);
        for(int j=0;j<3;++j)kern(npt+i+j*key_npt) = G[j];
    }
    // kern.save("kern.txt", arma::arma_ascii);
    // a.save("a.txt",  arma::arma_ascii);
    // std::cout << " a vec size " << a.size() << std::endl; 
    // std::cout << " kern vec size " << kern.size() << std::endl; 
    // std::cout << "  XCube_Gradient_Kernel_2p " << std::endl;
    // double loc_part = arma::dot(kern,a);
    double loc_part = 0;
    for(int i=0;i<npt * 4;++i)
    {
        loc_part += kern[i] * a[i];
    }

    // std::cout << "  loc_part " << loc_part << std::endl;
    arma::vec kb(4);
    for(int i=0;i<3;++i) {
        kb(i+1) = in_pt[i];
    }
    kb(0) = 1;
    double poly_part = 0;
    // double poly_part = arma::dot(kb,b);
    for(int i=0;i< 4;++i)
    {
        poly_part += kb[i] * b[i];
    }
    // std::cout << "  poly_part " << poly_part << std::endl;
    double re = loc_part + poly_part;
    return re;
}



double VIPSS_HRBF_Dist_Alone(const double* in_pt, const arma::vec& a, const arma::vec& b, 
                                const std::vector<std::array<double,3>>& all_pts)
{
    int npt = all_pts.size();
    int key_npt = npt;
    double G[3];
    arma::vec kern(npt + 3 * key_npt);
    // std::cout << "  start dist pt size  " << npt << std::endl;
    for(int i=0;i<npt;++i) kern(i) = VIPSSKernel::XCube_Kernel_2p(&(all_pts[i][0]), in_pt);
    // std::cout << "  XCube_Kernel_2p " << std::endl;
    for(int i=0;i<key_npt;++i){
        VIPSSKernel::XCube_Gradient_Kernel_2p(in_pt, &(all_pts[i][0]), G);
        for(int j=0;j<3;++j)kern(npt+i+j*key_npt) = G[j];
    }
    // kern.save("kern.txt", arma::arma_ascii);
    // a.save("a.txt",  arma::arma_ascii);
    // std::cout << " a vec size " << a.size() << std::endl; 
    // std::cout << " kern vec size " << kern.size() << std::endl; 
    // std::cout << "  XCube_Gradient_Kernel_2p " << std::endl;
    
    // double loc_part = arma::dot(kern,a);

    double loc_part = 0;
    for(int i=0;i<npt * 4;++i)
    {
        loc_part += kern[i] * a[i];
    }

    // std::cout << "  loc_part " << loc_part << std::endl;
    arma::vec kb(4);
    for(int i=0;i<3;++i) {
        kb(i+1) = in_pt[i];
    }
    kb(0) = 1;
    double poly_part = arma::dot(kb,b);
    // std::cout << "  poly_part " << poly_part << std::endl;
    double re = loc_part + poly_part;
    return re;
}



void BuildGlobalHRBFVipss(std::vector<double> &Vs, std::shared_ptr<RBF_Core> rfb_ptr, double lambda)
{
    auto t0 = Clock::now();
    // std::cout << " input pt size " << Vs.size()/3 << std::endl;
    // para_.user_lamnbda = lambda;
    // RBF_Core rbf_core_;
    // std::cout << " input key pt size  00 "  << std::endl;
    rfb_ptr->key_npt = Vs.size()/3;

    std::cout << " input key pt size " << rfb_ptr->key_npt  << std::endl;
    
    rfb_ptr->InjectData(Vs, RBF_API::para_);
    // rfb_ptr->User_Lamnbda = lambda;
    rfb_ptr->BuildK(lambda);

    std::cout << " finish build k " << std::endl;
    double build_time = (std::chrono::nanoseconds(Clock::now() - t0).count()/1e9);
    
    rfb_ptr->InitNormal();
    std::cout << " finish InitNormal " << std::endl;

    rfb_ptr->OptNormal(0);

    

    
    auto t1 = Clock::now();
    double pre_time = (std::chrono::nanoseconds(t1 - t0).count()/1e9);
}