#include <chrono>
#include "vipss_unit.hpp"
#include "stats.h"
#include "adgrid.h"
#include "SimpleOctree.h"
#include "test_timing.h"
#include "color.h"
#include <numeric>

typedef std::chrono::high_resolution_clock Clock;

VP_STATS G_VP_stats;
int VIPSSUnit::opt_func_count_g = 0;
double VIPSSUnit::opt_func_time_g = 0;
Eigen::VectorXd arma_x_opt_g;
Eigen::VectorXd res_vec_g;
Eigen::VectorXd scsc_vec;
bool is_alpha_initialized = false;
double soft_constraints_alpha = 10.0;

void VIPSSUnit::InitPtNormalWithLocalVipss()
{
    std::string& data_dir =  data_dir_;
    local_vipss_.filename_ = file_name_;
    // l_vp.filename_ = "planck";
    local_vipss_.out_dir_ = out_dir_;
    // std::cout << " out dir --------------- " << local_vipss_.out_dir_ << std::endl;
    std::string path = data_dir + local_vipss_.filename_ + "/" + local_vipss_.filename_ + ".ply";
    // local_vipss_.angle_threshold_ = 30;
    // local_vipss_.user_lambda_ = user_lambda_;
    local_vipss_.user_lambda_ = user_lambda_;
    // local_vipss_.max_iter_ = 30;
    local_vipss_.use_hrbf_surface_ = false;
    local_vipss_.angle_threshold_ = merge_angle_;
    auto t1 = Clock::now();
    // local_vipss_.volume_dim_ = 100;
    local_vipss_.Init(input_data_path_, input_data_ext_);
    local_vipss_.InitNormals();
    auto t2 = Clock::now();
    G_VP_stats.init_normal_total_time_ = std::chrono::nanoseconds(t2 - t1).count() / 1e9;
    G_VP_stats.tetgen_triangulation_time_ = local_vipss_.tet_gen_triangulation_time_;
    npt_ = local_vipss_.points_.size();

    auto t3 = Clock::now();
    local_vipss_.ClearPartialMemory();
    std::cout << "free init normal allocated memory !" << std::endl;
    if(!only_use_nn_hrbf_surface_)
    {

        local_vipss_.BuildMatrixHMemoryOpt();
        // local_vipss_.SetBestNormalsWithHmat();
        if(user_lambda_ < 1e-12)
        {
            local_vipss_.final_h_eigen_ = local_vipss_.final_h_eigen_.block(npt_, npt_, 3 * npt_, 3 * npt_);
        } 
    }
    auto t4 = Clock::now();
    G_VP_stats.build_H_total_time_ = std::chrono::nanoseconds(t4 - t3).count() / 1e9;
    // double construct_Hmat_time = std::chrono::nanoseconds(t4 - t3).count() / 1e9;
    // printf("unit vipss J mat init time : %f \n", local_vipss_.vipss_api_.u_v_time);
}

// double acc_tim1;
// static int countopt = 0;


double optfunc_unit_vipss_simple_eigen(const std::vector<double>&x, std::vector<double>&grad, void *fdata){

    auto t1 = Clock::now();
    VIPSSUnit *drbf = reinterpret_cast<VIPSSUnit*>(fdata);
    // size_t n = drbf->npt;
    size_t n = drbf->npt_;
    // printf("input point number : %llu \n", n);
    // Eigen::VectorXd arma_x(n*3);
    //(  sin(a)cos(b), sin(a)sin(b), cos(a)  )  a =>[0, pi], b => [-pi, pi];
    // std::vector<double>sina_cosa_sinb_cosb(n * 4);
    #pragma omp parallel for 
    for(int i=0;i<n;++i){
        size_t ind = i*4;
        scsc_vec[ind] = sin(x[i*2]);
        scsc_vec[ind+1] = cos(x[i*2]);
        scsc_vec[ind+2] = sin(x[i*2+1]);
        scsc_vec[ind+3] = cos(x[i*2+1]);
        if(drbf->axi_plane_ == AXI_PlANE::XYZ)
        {
            arma_x_opt_g(i) = scsc_vec[ind] * scsc_vec[ind + 3];
            arma_x_opt_g(i+n) = scsc_vec[ind] * scsc_vec[ind + 2];
            arma_x_opt_g(i+n*2) = scsc_vec[ind + 1];
        } else {
            arma_x_opt_g(i) = scsc_vec[ind] * scsc_vec[ind + 3];
            arma_x_opt_g(i+n) = scsc_vec[ind + 1];
            arma_x_opt_g(i+n*2) = scsc_vec[ind] * scsc_vec[ind + 2];
        }
    }
    // for(int i=0;i<n;++i){
    // }

    Eigen::VectorXd a2 = (arma_x_opt_g.transpose() * drbf->local_vipss_.final_h_eigen_).transpose();
    // if (!grad.empty()) 
    {
        grad.resize(n*2);
        #pragma omp parallel for 
        for(int i=0;i<n;++i){
            // auto p_scsc = sina_cosa_sinb_cosb.data()+i*4;
            size_t ind = i*4;
            
            if(drbf->axi_plane_ == AXI_PlANE::XYZ)
            {
                grad[i*2] = a2(i) * scsc_vec[ind + 1] * scsc_vec[ind + 3] + a2(i+n) * scsc_vec[ind + 1] * scsc_vec[ind + 2] - a2(i+n*2) * scsc_vec[ind];
                grad[i*2+1] = -a2(i) * scsc_vec[ind] * scsc_vec[ind + 2] + a2(i+n) * scsc_vec[ind] * scsc_vec[ind + 3];
            } else {
                grad[i*2] = a2(i) * scsc_vec[ind + 1] * scsc_vec[ind + 3] + a2(i+n*2) * scsc_vec[ind + 1] * scsc_vec[ind + 2] - a2(i+n) * scsc_vec[ind];
                grad[i*2+1] = -a2(i) * scsc_vec[ind] * scsc_vec[ind+ 2] + a2(i+n*2) * scsc_vec[ind] * scsc_vec[ind + 3];
            }
            
        }
    }
    double re =  arma_x_opt_g.dot( a2 );
    // printf("residual val : %f \n", re);
    // printf("Final_H_ non zero  : %d \n", drbf->Final_H_.n_nonzero);
    // countopt++;
    // acc_time+=(std::chrono::nanoseconds(Clock::now() - t1).count()/1e9);
    drbf->countopt_ ++;
    VIPSSUnit::opt_func_count_g ++;
    if(G_VP_stats.save_residuals_)
    G_VP_stats.residuals_.push_back(re);
    return re;
}

double optfunc_unit_vipss_direct_simple_eigen(const std::vector<double>&x, std::vector<double>&grad, void *fdata){

    auto t1 = Clock::now();
    VIPSSUnit *drbf = reinterpret_cast<VIPSSUnit*>(fdata);
    // size_t n = drbf->npt;
    int n = drbf->npt_;
    Eigen::VectorXd arma_x(n*3);
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        // auto p_scsc = sina_cosa_sinb_cosb.data()+i*4;
        arma_x(i)     = x[3*i];
        arma_x(i+n)   = x[3*i + 1];
        arma_x(i+n*2) = x[3*i + 2];
    }
    // std::cout << " grad size " <<grad.size() << std::endl;
    Eigen::VectorXd a2 = (arma_x.transpose() * drbf->local_vipss_.final_h_eigen_).transpose();
    // if (!grad.empty()) 
    {
        // std::cout << " grad size " <<grad.size() << std::endl;
        grad.resize(n*3);
        #pragma omp parallel for
        for(int i=0;i<n;++i){
            grad[i*3]   = a2(i) ;
            grad[i*3+1] = a2(n+i) ;
            grad[i*3+2] = a2(2*n+i) ;
        }
    }
    double re = arma_x.dot(a2);
    // double alpha = 10.0;
    
    if(!is_alpha_initialized)
    {
        // soft_constraints_alpha = 100.0;
        soft_constraints_alpha = drbf->user_alpha_;
        if(drbf->user_lambda_ > 1e-16)
        {
            soft_constraints_alpha *= drbf->user_lambda_;
        }
        is_alpha_initialized = true;
        std::cout << "set alpha : " << soft_constraints_alpha << std::endl;
    }
    
    double  k = 2.0;
    // std::cout << "alpha " << alpha << std::endl;
    
    Eigen::VectorXd res_g(n);
    #pragma omp parallel for
    for(int id =0; id < n; ++id)
    {
        double cur_re = x[3*id] * x[3*id] + x[3*id + 1] * x[3*id + 1] + x[3*id + 2] * x[3*id + 2] - 1;
        // if(!grad.empty()) 
        // std::cout << " cur id : " << id <<  " . cur res : " << cur_re << std::endl;
        {
            grad[3* id]     += soft_constraints_alpha * k * x[3*id] * cur_re; 
            grad[3* id + 1] += soft_constraints_alpha * k * x[3*id + 1] * cur_re; 
            grad[3* id + 2] += soft_constraints_alpha * k * x[3*id + 2] * cur_re; 
        }
        res_g[id] = soft_constraints_alpha * cur_re * cur_re;
    }
    re += res_g.sum();
    // std::cout << "re " << re << std::endl;
    // printf("res val : %f \n", re);
    VIPSSUnit::opt_func_count_g ++;
    if(G_VP_stats.save_residuals_)
    G_VP_stats.residuals_.push_back(re);
    return re;
}

double optfunc_unit_vipss_direct_eigen(const std::vector<double>& x, std::vector<double>& grad, void* fdata) {

    auto t1 = Clock::now();
    VIPSSUnit::opt_func_count_g++;
    VIPSSUnit* drbf = reinterpret_cast<VIPSSUnit*>(fdata);
    // size_t n = drbf->npt;
    size_t n = drbf->npt_;
    size_t u_size = 4;
    // Eigen::VectorXd para_x(n * u_size);
    // double s_scale = 0.01 / double(n);
    Eigen::VectorXd res_s = Eigen::VectorXd::Zero(n);
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        // auto p_scsc = sina_cosa_sinb_cosb.data()+i*4;
        arma_x_opt_g(i)         = x[u_size * i];
        arma_x_opt_g(i + n)     = x[u_size * i + 1];
        arma_x_opt_g(i + n * 2) = x[u_size * i + 2];
        arma_x_opt_g(i + n * 3) = x[u_size * i + 3];

        res_s[i] = x[u_size * i] * x[u_size * i];
        // para_x(i) =  s_scale;
        // para_x(i + n) =  1.0;
        // para_x(i + n * 2) =  1.0;
        // para_x(i + n * 3) =  1.0;
    }
    //Eigen::VectorXd a2 = (arma_x_opt_g.transpose() * drbf->local_vipss_.final_h_eigen_).transpose();
    int id;
    //auto x_t = arma_x_opt_g.transpose();
    // double alpha = 0.01;
    double k = 2.0;
    // const auto& final_h = drbf->local_vipss_.final_h_eigen_;
    Eigen::VectorXd a2 = (arma_x_opt_g.transpose() * drbf->local_vipss_.final_h_eigen_).transpose();
    // a2 = a2.cwiseProduct(para_x);
    res_vec_g = arma_x_opt_g.cwiseProduct(a2);



    if(!is_alpha_initialized)
    {
        soft_constraints_alpha = drbf->user_alpha_;
        if(drbf->user_lambda_ > 1e-16)
        {
            soft_constraints_alpha *= drbf->user_lambda_;
        }
        // soft_constraints_alpha = drbf->user_alpha_ * drbf->user_lambda_;
        // soft_constraints_alpha = 50.0 * drbf->user_lambda_;
        is_alpha_initialized = true;
        std::cout << "set alpha : " << soft_constraints_alpha << std::endl;
    }
    

    // std::cout << " residual " << res_vec_g.sum() << std::endl;
    
    if(grad.empty()) grad.resize(4 * n);
    #pragma omp parallel for shared(x, grad, arma_x_opt_g, res_vec_g) private(id)
    for (id = 0; id < n; ++id)
    {
        // res_vec_g[id] = 0;
        for (int j = 0; j < 4; ++j)
        {
            double val = a2[id + j *n];
            //  arma_x_opt_g .transpose() * final_h.col(id + j *n);
            // res_vec_g[id] += arma_x_opt_g[id + j *n] * val;
            grad[id * 4 + j] = val * k ;
        }
        grad[id * 4 ] += 2 * x[u_size * id];
        double cur_re = x[4 * id + 1] * x[4 * id + 1] + x[4 * id + 2] * x[4 * id + 2]
            + x[4 * id + 3] * x[4 * id + 3] - 1;
        grad[4 * id + 1] += soft_constraints_alpha * k * x[4 * id + 1] * cur_re;
        grad[4 * id + 2] += soft_constraints_alpha * k * x[4 * id + 2] * cur_re;
        grad[4 * id + 3] += soft_constraints_alpha * k * x[4 * id + 3] * cur_re;
        res_vec_g[id] += soft_constraints_alpha * cur_re * cur_re;
    }
    double re = res_s.sum()  + res_vec_g.sum();
    // double re = res_vec_g.sum();
    auto t2 = Clock::now();
    double opt_time = std::chrono::nanoseconds(t2 - t1).count()/ 1e9;
    VIPSSUnit::opt_func_time_g += opt_time;
    // printf("opt fun call time accu: %f \n", VIPSSUnit::opt_func_time_g);
    if(G_VP_stats.save_residuals_)
    G_VP_stats.residuals_.push_back(re);
    return re;
}

double optfunc_unit_vipss(const std::vector<double>&x, std::vector<double>&grad, void *fdata){

    auto t1 = Clock::now();
    VIPSSUnit *drbf = reinterpret_cast<VIPSSUnit*>(fdata);
    size_t n = drbf->npt_;
    Eigen::VectorXd arma_x(n*4);
    std::vector<double>sina_cosa_sinb_cosb(n * 4);
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        size_t ind = i*4;
        sina_cosa_sinb_cosb[ind]   = sin(x[i*3 + 1]);
        sina_cosa_sinb_cosb[ind+1] = cos(x[i*3 + 1]);
        sina_cosa_sinb_cosb[ind+2] = sin(x[i*3 + 2]);
        sina_cosa_sinb_cosb[ind+3] = cos(x[i*3 + 2]);
    }
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        auto p_scsc = sina_cosa_sinb_cosb.data()+i*4;
        arma_x(i) = x[i *3];
        arma_x(i + n) = p_scsc[0] * p_scsc[3];
        arma_x(i + n * 2) = p_scsc[0] * p_scsc[2];
        arma_x(i + n * 3) = p_scsc[1];
    }

    Eigen::VectorXd a2 =  (arma_x.transpose() * drbf ->local_vipss_.final_h_eigen_).transpose();
    if (!grad.empty()) {
        // grad.resize(n*3);
        // #pragma omp parallel for
        double k = 2.0;
        #pragma omp parallel for
        for(int i=0;i<n;++i){
            auto p_scsc = sina_cosa_sinb_cosb.data()+i*4;
            grad[i*3] = k * a2[i];
            grad[i*3 + 1] = k * (a2(n + i) * p_scsc[1] * p_scsc[3] + a2(i+ n * 2) * p_scsc[1] * p_scsc[2] - a2(i+n*3) * p_scsc[0]);
            grad[i*3 + 2] = k * (-a2(i + n) * p_scsc[0] * p_scsc[2] + a2(i+n* 2) * p_scsc[0] * p_scsc[3]);
        }
    }
    double re =  arma_x.dot(a2);
    // printf("residual val : %f \n", re);
    // printf("Final_H_ non zero  : %d \n", drbf->Final_H_.n_nonzero);
    // countopt++;
    // acc_time+=(std::chrono::nanoseconds(Clock::now() - t1).count()/1e9);
    drbf->countopt_ ++;
    VIPSSUnit::opt_func_count_g ++;

    if(G_VP_stats.save_residuals_)
    G_VP_stats.residuals_.push_back(re);
    return re;
}

void VIPSSUnit::OptUnitVipssNormalSimple(){

    printf("start to call solver ! \n");
    solver_.solveval.resize(npt_ * 2);

    for(size_t i=0;i<npt_;++i){
        double *veccc = local_vipss_.out_normals_.data()+i*3;
        {
            if(axi_plane_ == AXI_PlANE::XYZ)
            {
                solver_.solveval[i*2] = atan2(sqrt(veccc[0]*veccc[0]+veccc[1]*veccc[1]),veccc[2] );
                solver_.solveval[i*2 + 1] = atan2( veccc[1], veccc[0]);
            } else {
                solver_.solveval[i*2] = atan2(sqrt(veccc[0]*veccc[0]+veccc[2]*veccc[2]),veccc[1] );
                solver_.solveval[i*2 + 1] = atan2( veccc[2], veccc[0]);
            }
            // solver_.solveval[i*2] = atan2(sqrt(veccc[1]*veccc[1]+veccc[2]*veccc[2]),veccc[0] );
            // solver_.solveval[i*2 + 1] = atan2( veccc[2], veccc[1]);
        }
    }
    arma_x_opt_g.resize(npt_ * 3);
    res_vec_g.resize(npt_);
    scsc_vec.resize(npt_ * 4);
    // printf("finish init solver ! \n");
    if(1){ 
        std::vector<double>upper(npt_*2);
        std::vector<double>lower(npt_*2);
        for(int i=0;i<npt_;++i){
            upper[i*2] = 1 * M_PI_;
            upper[i*2 + 1] = 1 * M_PI_;

            lower[i*2] = -1 * M_PI_;
            lower[i*2 + 1] = -1 * M_PI_;
        }
        // countopt = 0;
        // acc_time = 0;
        //LocalIterativeSolver(sol,kk==0?normals:newnormals,300,1e-7);
        // printf("start the solver ! \n");
        Solver::nloptwrapper(lower,upper,optfunc_unit_vipss_simple_eigen,this,opt_tor_, max_opt_iter_,solver_);
        // callfunc_time = acc_time;
        // solve_time = sol.time;
        //for(int i=0;i<npt;++i)cout<< sol.solveval[i]<<' ';cout<<endl;
    }
    newnormals_.resize(npt_*3);
    s_func_vals_.resize(npt_, 0); 
    // printf("-----------newnormal size : %lu \n", newnormals_.size());
    arma::vec y(npt_ + 3 * npt_);
    for(size_t i=0;i<npt_;++i)y(i) = 0;
    for(size_t i=0;i<npt_;++i){
        double a = solver_.solveval[i*2], b = solver_.solveval[i*2+1];

        if(axi_plane_ == AXI_PlANE::XYZ)
        {
            newnormals_[i*3]   = sin(a) * cos(b);
            newnormals_[i*3+1] = sin(a) * sin(b);
            newnormals_[i*3+2] = cos(a);
        } else {
            newnormals_[i*3]   = sin(a) * cos(b);
            newnormals_[i*3+1] = cos(a);
            newnormals_[i*3+2] = sin(a) * sin(b);
        }
        
        // MyUtility::normalize(newnormals.data()+i*3);
    }
    // Set_RBFCoef(y);
    //sol.energy = arma::dot(a,M*a);
    // if(open_debug_log)
    // cout<<"Opt_Hermite_PredictNormal_UnitNormal"<<endl;
    return;
}

void VIPSSUnit::OptUnitVipssNormalDirectSimple(){

    printf("start to call simple soft solver ! \n");
    solver_.solveval.resize(npt_ * 3);
    #pragma omp parallel for
    for(int i=0; i<npt_;++i){
        double *veccc = local_vipss_.out_normals_.data()+i*3;
        {
            solver_.solveval[i*3] =veccc[0];
            solver_.solveval[i*3 + 1] = veccc[1];
            solver_.solveval[i*3 + 2] = veccc[2];
        }
    }
    std::vector<double>upper(npt_*3);
    std::vector<double>lower(npt_*3);
    #pragma omp parallel for
    for(int i=0;i<npt_;++i){
        upper[i*3] = 1;
        upper[i*3 + 1] = 1;
        upper[i*3 + 2] = 1;

        lower[i*3] = -1.0;
        lower[i*3 + 1] = -1.0;
        lower[i*3 + 2] = -1.0;
    }
    // printf("start to call simple soft solver optimizer ! \n");
    Solver::nloptwrapperDirect(lower,upper,optfunc_unit_vipss_direct_simple_eigen,
                this,opt_tor_, max_opt_iter_,solver_);
    // Solver::nloptwrapper(lower,upper,optfunc_unit_vipss_simple,this,1e-7,3000,solver_);
    
    newnormals_.resize(npt_*3);
    s_func_vals_.resize(npt_, 0);
    arma::vec y(npt_ + 3 * npt_);
    
    #pragma omp parallel for
    for(int i=0;i<npt_;++i){
        y(i) = 0;
        newnormals_[i*3]   = solver_.solveval[i*3];
        newnormals_[i*3+1] = solver_.solveval[i*3 + 1];
        newnormals_[i*3+2] = solver_.solveval[i*3 + 2];
        // MyUtility::normalize(newnormals.data()+i*3);
    }
    return;
}


void VIPSSUnit::OptUnitVipssNormalDirect(){

    printf("start to call soft solver ! \n");

    size_t u_size = 4;
    
    solver_.solveval.resize(npt_ * u_size);

    #pragma omp parallel for
    for(int i=0;i<npt_;++i){
        double *veccc = local_vipss_.out_normals_.data()+i*3;
        {
            solver_.solveval[i*u_size]     = 0;
            // solver_.solveval[i*u_size]     =  local_vipss_.s_vals_[i];
            // std::cout << " local_vipss_.s_vals_[i] " << i << " " << local_vipss_.s_vals_[i] << std::endl; 
            solver_.solveval[i*u_size + 1] = veccc[0];
            solver_.solveval[i*u_size + 2] = veccc[1];
            solver_.solveval[i*u_size + 3] = veccc[2];
        }
    }
    std::vector<double>upper(npt_ * u_size);
    std::vector<double>lower(npt_ * u_size);
    for(int i=0;i<npt_;++i){
        for(size_t j = 0; j < u_size; ++j)
        {
            upper[i*u_size + j] = 1;
            lower[i*u_size + j] = -1.0;
        }
    }
    arma_x_opt_g.resize(npt_ * 4);
    res_vec_g.resize(npt_);

    printf("start to call optimizer ! \n");
    Solver::nloptwrapperDirect(lower,upper,optfunc_unit_vipss_direct_eigen,
                this, opt_tor_, max_opt_iter_, solver_);
    
    // Solver::nloptwrapper(lower,upper,optfunc_unit_vipss_simple,this,1e-7,3000,solver_);
    newnormals_.resize(npt_*3);
    s_func_vals_.resize(npt_);
    // arma::vec y(npt_ + 3 * npt_);
    // for(size_t i=0;i<npt_;++i)y(i) = 0;
    for(size_t i=0;i<npt_;++i){
        s_func_vals_[i]    = solver_.solveval[i*u_size];
        newnormals_[i*3]   = solver_.solveval[i*u_size + 1];
        newnormals_[i*3+1] = solver_.solveval[i*u_size + 2];
        newnormals_[i*3+2] = solver_.solveval[i*u_size + 3];
        // MyUtility::normalize(newnormals.data()+i*3);
    }
    return;
}


void VIPSSUnit::OptUnitVipssNormal(){

    printf("start to call solver ! \n");
    solver_.solveval.resize(npt_ * 3);
    
    // std::string s_val_path = "./s_vals_init.txt";
    // WriteVectorValsToCSV(s_val_path, local_vipss_.s_vals_);


    for(int i=0;i<npt_;++i){
        double *veccc = local_vipss_.out_normals_.data()+i*3;
        {
            solver_.solveval[i*3] =  local_vipss_.s_vals_[i];
            // solver_.solveval[i*3] = 0;
            solver_.solveval[i*3 + 1] = atan2(sqrt(veccc[0]*veccc[0]+veccc[1]*veccc[1]),veccc[2] );
            solver_.solveval[i*3 + 2] = atan2( veccc[1], veccc[0] );
        }
    }
    arma_x_opt_g.resize(npt_ * 4);
    res_vec_g.resize(npt_);
    // printf("finish init solver ! \n");
    if(1){
        std::vector<double>upper(npt_*3);
        std::vector<double>lower(npt_*3);
        for(int i=0;i<npt_;++i){
            upper[i*3 ] = 1.0;
            upper[i*3 + 1] = 1 * M_PI_;
            upper[i*3 + 2] = 1 * M_PI_;

            lower[i*3] = -1.0;
            lower[i*3 + 1] = -1.0 * M_PI_;
            lower[i*3 + 2] = -1 * M_PI_;
        }
        // printf("start the solver ! \n");
        Solver::nloptwrapper(lower,upper,optfunc_unit_vipss,this,opt_tor_, max_opt_iter_,solver_);
    }
    newnormals_.resize(npt_*3);
    s_func_vals_.resize(npt_);
    for(size_t i=0;i<npt_;++i) s_func_vals_[i] = solver_.solveval[i*3];;
    for(size_t i=0;i<npt_;++i){
        double a = solver_.solveval[i*3 + 1], b = solver_.solveval[i*3+2];
        newnormals_[i*3]   = sin(a) * cos(b);
        newnormals_[i*3+1] = sin(a) * sin(b);
        newnormals_[i*3+2] = cos(a);
        // MyUtility::normalize(newnormals.data()+i*3);
    }
    // std::string s_val_opt_path = "./s_vals_opt.txt";
    // WriteVectorValsToCSV(s_val_opt_path, s_func_vals_);
    // Set_RBFCoef(y);
    //sol.energy = arma::dot(a,M*a);
    // if(open_debug_log)
    // cout<<"Opt_Hermite_PredictNormal_UnitNormal"<<endl;
    return;
}

void VIPSSUnit::BuildNNHRBFFunctions()
{
    auto t000 = Clock::now();
    std::vector<std::array<double,3>> octree_sample_pts;
    if(make_nn_const_neighbor_num_)
    {
        SimOctree::SimpleOctree octree;
        // std::cout << " start to init octree " << std::endl;
        // octree.InitOctTree(local_vipss_.origin_in_pts_, 5);
        std::cout << " insert octree center pts num : " << octree.octree_centers_.size() << std::endl; 
        octree_sample_pts = octree.octree_centers_;
    }

    local_vipss_.voro_gen_.GenerateVoroData();
    local_vipss_.voro_gen_.SetInsertBoundaryPtsToUnused();
    std::cout << " build voronoi success ! " << std::endl;
    local_vipss_.voro_gen_.BuildTetMeshTetCenterMap();
    local_vipss_.voro_gen_.BuildPicoTree();
    std::cout << " build pico tree success ! " << std::endl;
    // auto boundary_pts = local_vipss_.voro_gen_.insert_boundary_pts_;
    local_vipss_.voro_gen_.insert_boundary_pts_.clear();
    auto t001 = Clock::now();
    G_VP_stats.generate_voro_data_time_ = std::chrono::nanoseconds(t001 - t000).count() / 1e9;
    // G_VP_stats.generate_voro_data_time_ = generate_voro_data_time
    // auto t002 = Clock::now();
    // double generate_voro_data_time = std::chrono::nanoseconds(t002 - t001).count() / 1e9;
    local_vipss_.out_normals_ = newnormals_;
    local_vipss_.s_vals_ = s_func_vals_;
    local_vipss_.user_lambda_ = user_lambda_;

    std::cout << "--- input user lambda val " << user_lambda_ << std::endl;
    if(only_use_nn_hrbf_surface_)
    {
        std::vector<double> in_pts;
        auto& in_normals = local_vipss_.input_normals_;
        // readXYZnormal(input_data_path_, in_pts, in_normals);
        // std::cout << "--- input normal size " << in_normals.size() / 3 << std::endl;
        newnormals_ = in_normals;
        local_vipss_.out_normals_ = in_normals;
        s_func_vals_.resize(in_normals.size()/3, 0);

        // std::string vipss_s_val_path = "/home/jjxia/Documents/prejects/VIPSS/vipss/build/s_val_walrus.txt";
        // s_func_vals_ = ReadVectorFromFile(vipss_s_val_path);
        local_vipss_.s_vals_ = s_func_vals_;

        std::cout << "input func svals size : " << local_vipss_.s_vals_.size() << std::endl;
    }

    // for(int i = 0; i < 9 ; ++i)
    // {
    //     // std::cout << v_pts[i] * 2<< " " << local_vipss_.out_pts_[i] << std::endl;
    //     std::cout << "v_normals " << local_vipss_.out_normals_[i] << std::endl;
    // }
    
    // newnormals_.clear();
    // s_func_vals_.clear();
    local_vipss_.BuildHRBFPerNode();
    local_vipss_.SetThis(); 
    // local_vipss_.dummy_sign_ = LocalVipss::NNDistFunction(R3Pt(local_vipss_.voro_gen_.dummy_sign_pt_[0],
    //                                                            local_vipss_.voro_gen_.dummy_sign_pt_[1], 
    //                                                            local_vipss_.voro_gen_.dummy_sign_pt_[2]));    
    if(abs(local_vipss_.dummy_sign_ ) > 1e-18)
    {
        local_vipss_.dummy_sign_ = local_vipss_.dummy_sign_ / abs(local_vipss_.dummy_sign_);
    }
    std::cout << " ********** dummy pt sign val : " << local_vipss_.dummy_sign_ << std::endl;
                                                        

    if(make_nn_const_neighbor_num_)
    {
        std::vector<double> insert_pt_func_vals;
        std::vector<double> insert_pt_func_gradients;
        std::vector<std::array<double,3>> valid_pts;
        std::vector<double> dummy_dist_vals; 
        auto t001 = Clock::now();
        // std::string octree_sample_path = out_dir_ + file_name_ + "octree_sample.xyz";
        // std::ofstream octree_file(octree_sample_path);
    // if(0)
    // {
    //     for(auto pt : octree_sample_pts)
    //     {
    //         double gradient[3];
    //         double dist_val = local_vipss_.NatureNeighborGradientOMP(&pt[0], gradient);
    //         local_vipss_.s_vals_.push_back(dist_val);
    //         local_vipss_.normals_.push_back(-1.0 * gradient[0] );
    //         local_vipss_.normals_.push_back(-1.0 * gradient[1] );
    //         local_vipss_.normals_.push_back(-1.0 * gradient[2] );
    //         // octree_file << pt[0] << " " << pt[1] << " " << pt[2] << " ";
    //         // octree_file << gradient[0] << " " << gradient[1] << " " << gradient[2] << std::endl;
    //     }
    // }
        auto t0022 = Clock::now();
        G_VP_stats.octree_pt_gradient_cal_time_ = std::chrono::nanoseconds(t0022 - t001).count() / 1e9;
        std::cout << "evaluate octree sample time : " << G_VP_stats.octree_pt_gradient_cal_time_ << std::endl;
        G_VP_stats.octree_dummy_pt_num_ = octree_sample_pts.size();
        local_vipss_.voro_gen_.InsertPts(octree_sample_pts);
        local_vipss_.voro_gen_.BuildTetMeshTetCenterMap();
        local_vipss_.voro_gen_.BuildPicoTree();
        
        local_vipss_.voro_gen_.tetMesh_.generate_voronoi_cell(&(local_vipss_.voro_gen_.voronoi_data_));
        local_vipss_.voro_gen_.SetInsertBoundaryPtsToUnused();
        auto t0033 = Clock::now();
        double rebuild_pico_and_voro_time = std::chrono::nanoseconds(t0033 - t0022).count() / 1e9;
        std::cout << "rebuild_pico_and_voro_time : " << rebuild_pico_and_voro_time << std::endl;

    if(0)
    {
        const auto& insert_pts = local_vipss_.voro_gen_.insert_boundary_pts_;
        int input_pt_size = local_vipss_.points_.size();
        // std::vector<tetgenmesh::point> octree_insert_pts;
        for(auto pt : insert_pts)
        {
            int new_pid = local_vipss_.points_.size();
            local_vipss_.points_.push_back(pt);
            local_vipss_.voro_gen_.point_id_map_[pt] = new_pid;
        }
        auto all_valid_pt_size = local_vipss_.points_.size();
        std::cout << "octree_insert_pts size : " << insert_pts.size() << std::endl;
        int cluster_sum = 0;
        local_vipss_.voro_gen_.cluster_init_pids_.clear();
        local_vipss_.voro_gen_.cluster_init_pids_.resize(all_valid_pt_size);
        // local_vipss_.node_rbf_vec_.clear();
        local_vipss_.node_rbf_vec_.resize(all_valid_pt_size);

        auto t0041 = Clock::now();
        // VoronoiGen::cluster_init_pts_.resize(all_valid_pt_size);
        std::vector<std::vector<double>> cluster_sv_vecs(all_valid_pt_size); 
        std::vector<std::vector<double>> cluster_pt_vecs(all_valid_pt_size);
        std::vector<std::vector<double>> cluster_nl_vecs(all_valid_pt_size);

        int max_cluster_size = 0;
        for(int i =0; i < all_valid_pt_size; ++i)
        {
            auto cur_pt = local_vipss_.points_[i];
            std::set<tetgenmesh::point> candidate_pts;
            local_vipss_.voro_gen_.GetVertexStar(cur_pt, candidate_pts, 1);
            std::vector<int> cluster_pt_ids;
            // cluster_pt_ids.push_back(local_vipss_.voro_gen_.point_id_map_[cur_pt]);
            // std::vector<double> cluster_nl_vec;
            auto cur_pid = local_vipss_.voro_gen_.point_id_map_[cur_pt];
            std::vector<double> cluster_nl_vec;
            std::vector<double> cluster_sv_vec; 
            // std::vector<double> cluster_pt_vec;
            for(auto nn_pt : candidate_pts)
            {
                // if( nn_pt == cur_pt) continue;
                auto pid = local_vipss_.voro_gen_.point_id_map_[nn_pt];
                cluster_pt_ids.push_back(pid);
                // cluster_pt_vec.push_back(nn_pt[0]);
                // cluster_pt_vec.push_back(nn_pt[1]);
                // cluster_pt_vec.push_back(nn_pt[2]);
                // cluster_sv_vec.push_back(local_vipss_.s_vals_[pid]);
                // cluster_nl_vec.push_back(local_vipss_.normals_[3*pid]);
                // cluster_nl_vec.push_back(local_vipss_.normals_[3*pid + 1]);
                // cluster_nl_vec.push_back(local_vipss_.normals_[3*pid + 2]);
            }
            max_cluster_size = max_cluster_size > cluster_pt_ids.size() ? max_cluster_size : cluster_pt_ids.size();
            cluster_sum += cluster_pt_ids.size();
            local_vipss_.voro_gen_.cluster_init_pids_[i] = cluster_pt_ids;
            // VoronoiGen::cluster_init_pts_[i] = cluster_pt_vec;
            // cluster_sv_vecs[i] = cluster_sv_vec;
            // cluster_pt_vecs[i] = cluster_pt_vec; 
            // cluster_nl_vecs[i] = cluster_nl_vec;
        }
        std::cout << " max_cluster_size : " << max_cluster_size << std::endl;
        auto t0042 = Clock::now();
        double get_pt_neigbor_time = std::chrono::nanoseconds(t0042 - t0041).count() / 1e9;
        std::cout << "get_pt_neigbor_time : " << get_pt_neigbor_time << std::endl;

        // local_vipss_.s_vals_ = s_func_vals_;
        

        local_vipss_.BuildHRBFPerNode();
        local_vipss_.SetThis();
        
        auto t0043 = Clock::now();
        double rebuild_neigbor_time = std::chrono::nanoseconds(t0043 - t0042).count() / 1e9;
        std::cout << "rebuild_NN HRBF time : " << rebuild_neigbor_time << std::endl;

        int ave_cluster_size = int(double(cluster_sum) / double(local_vipss_.points_.size()));
        std::cout << " ------ ave cluster size " << ave_cluster_size << std::endl;
    }
    }
    // local_vipss_.out_normals_.clear();
    // local_vipss_.s_vals_.clear();
    auto t003 = Clock::now();
    // double build_nn_rbf_time  
    // G_VP_stats.build_nn_rbf_time_ = std::chrono::nanoseconds(t003 - t001).count() / 1e9;
    G_VP_stats.build_nn_rbf_time_ = std::chrono::nanoseconds(t003 - t000).count() / 1e9;
    G_VP_stats.average_cluster_size_ = local_vipss_.voro_gen_.average_neighbor_num_;
    G_VP_stats.pt_num_ = local_vipss_.points_.size();
    std::cout << " ------ build HRBF time all :  " << G_VP_stats.build_nn_rbf_time_ << std::endl;
    // local_vipss_.voro_gen_.SetInsertBoundaryPtsToUnused();
}

void VIPSSUnit::ReconSurface()
{
    printf(" start ReconSurface \n");
    if(LOCAL_HRBF_NN == hrbf_type_)
    {
        auto t000 = Clock::now();
        size_t n_voxels_1d = volume_dim_;
        std::cout << "pt size " << local_vipss_.out_pts_.size() << std::endl;
        Surfacer sf;
        int closet_id = 0;
        double closet_dist = std::numeric_limits<double>::max();
        for(int id = 0; id < s_func_vals_.size(); ++id)
        {
            if( closet_dist > abs(s_func_vals_[id]))
            {
                closet_dist = s_func_vals_[id];
                closet_id = id;
            }
        }
        sf.closet_id = closet_id;
        auto surf_time = sf.Surfacing_Implicit(local_vipss_.out_pts_, n_voxels_1d, false, LocalVipss::NNDistFunction);
        auto t004 = Clock::now();
        sf.WriteSurface(finalMesh_v_,finalMesh_fv_);
        auto t005 = Clock::now();
        double surface_file_save_time = std::chrono::nanoseconds(t005 - t004).count() / 1e9;
        double total_surface_time = std::chrono::nanoseconds(t004 - t000).count() / 1e9;
        writePLYFile_VF(out_surface_path_, finalMesh_v_, finalMesh_fv_);
        std::cout << "------- tet search time "<< tetgenmesh::tet_search_time_st << std::endl;
        std::cout << "------- voxel pt ave nn num "<< LocalVipss::ave_voxel_nn_pt_num_ / LocalVipss::DistCallNum << std::endl;
        printf("------- nn search time: %f \n", local_vipss_.search_nn_time_sum_);
        printf("------- cal nn coordinate and hrbf time: %f \n", local_vipss_.pass_time_sum_);
        printf(" ------ voxel dist func val evaluated count : %d  \n", LocalVipss::DistCallNum);
        printf(" ------ voxel dist func val evaluated time : %f \n", LocalVipss::DistCallTime);
        printf("------- total surface time: %f \n", total_surface_time);
        G_VP_stats.neighbor_search_time_ += local_vipss_.search_nn_time_sum_;
        G_VP_stats.cal_nn_coordinate_and_hbrf_time_ += local_vipss_.pass_time_sum_;
        G_VP_stats.voxel_cal_num += LocalVipss::DistCallNum;
        G_VP_stats.nn_evaluate_count_ = LocalVipss::DistCallNum;
        G_VP_stats.average_neighbor_num_ = double(LocalVipss::ave_voxel_nn_pt_num_)/ double(LocalVipss::DistCallNum);
        // G_VP_stats.surface_total_time_ += total_surface_time;
        G_VP_stats.surface_total_time_ = total_surface_time;

    } else {
        rbf_api_.user_lambda_ = user_lambda_;
        // rbf_api_.user_lambda_ = 0.001;
        rbf_api_.outpath_ = out_dir_ ;
        rbf_api_.filename_ =  file_name_;
        rbf_api_.is_surfacing_ = true;
        rbf_api_.n_voxel_line_ = (int)volume_dim_;
        std::cout << " surface volume dim : " << rbf_api_.n_voxel_line_ << std::endl;
        rbf_api_.run_vipss(local_vipss_.out_pts_, newnormals_, s_func_vals_);
    }
}

void VIPSSUnit::GenerateAdaptiveGrid()
{
    // std::cout << " test val " << test_val << std::endl;
    std::array<size_t,3> resolution = {3, 3, 3};
    std::vector<shared_ptr<ImplicitFunction<double>>> functions;
    // load_functions(args.function_file, functions);

    if(use_global_hrbf_)
    {
        std::cout << "surface function : global HRBF !" << std::endl;
        auto g_t0 = Clock::now();
        using Vec3 = Eigen::Matrix<double, 3, 1>;
        using Vec4 = Eigen::Matrix<double, 4, 1>;
        auto g_hrbf = std::make_shared<RBF_Core>();
        // if(only_use_nn_hrbf_surface_)
        // {
            std::vector<double> in_pts;
            std::vector<double> in_normals;
            readXYZnormal(input_data_path_, in_pts, in_normals);
            // std::cout << "--- input normal size " << in_normals.size() / 3 << std::endl;
            newnormals_ = in_normals;
            local_vipss_.out_normals_ = in_normals;
        // }
        // std::string vipss_pt_path = "/home/jjxia/Documents/prejects/VIPSS/data/noise_kitten/kitten_h004.001/input_normal_0.001.ply";
        // std::string vipss_s_val_path = "/home/jjxia/Documents/projects/VIPSS_LOCAL/data/scaled/kitten_h004_vals/kitten_h004_0.01.txt";
        // std::string vipss_s_val_path = "/home/jjxia/Documents/projects/VIPSS_LOCAL/data/scaled/kitten_h004_vals/kitten_h004_s_val_0.01.txt";
        // readPLYFile(vipss_pt_path, v_pts, v_normals);
        // std::cout << "s val data path " << vipss_s_val_path << std::endl;
        // s_func_vals_ = ReadVectorFromFile(vipss_s_val_path);
        // local_vipss_.vipss_api_.build_cluster_hrbf(v_pts, v_normals, s_vals, g_hrbf);

        std::cout << " pt size " << local_vipss_.out_pts_.size() /3 << std::endl;
        std::cout << " normal size " << local_vipss_.out_normals_.size() /3 << std::endl;
        std::cout << " s_func_vals_ size " << s_func_vals_.size() << std::endl;

        local_vipss_.vipss_api_.build_cluster_hrbf(in_pts, local_vipss_.out_normals_, s_func_vals_, g_hrbf);
        std::vector<std::array<double, 3> > output_vertices;
        std::vector<std::array<size_t, 3> > output_triangles;
        AdaptiveGridHRBF(g_hrbf, output_vertices, output_triangles);

    } else {
        std::shared_ptr<ImplicitFunction<double>> hrbf_func = std::make_shared<HRBFDistanceFunction>(iso_offset_val_);
        // hrbf_func->SetIsoOffset(iso_offset_val_);
        functions.push_back(hrbf_func);
        auto t000 = Clock::now();   
        std::vector<std::array<double, 3> > output_vertices;
        std::vector<std::array<size_t, 3> > output_triangles;
        GenerateAdaptiveGridOut(resolution, local_vipss_.voro_gen_.bbox_min_, 
                                local_vipss_.voro_gen_.bbox_max_, out_dir_,  
                                file_name_,  functions, adgrid_threshold_, output_vertices, output_triangles);
        auto t001 = Clock::now();

        for(auto& pt : output_vertices)
        {
            pt[0] = pt[0] *  local_vipss_.in_pt_scale_ + local_vipss_.in_pt_center_[0];
            pt[1] = pt[1] *  local_vipss_.in_pt_scale_ + local_vipss_.in_pt_center_[1];
            pt[2] = pt[2] *  local_vipss_.in_pt_scale_ + local_vipss_.in_pt_center_[2];
        }
        
        SaveMeshToPly(out_surface_path_, output_vertices, output_triangles);
        G_VP_stats.adgrid_gen_time_ = std::chrono::nanoseconds(t001 - t000).count() / 1e9;

        printf("adaptive grid generation time : %f ! \n", G_VP_stats.adgrid_gen_time_);
    }

    
}


void VIPSSUnit::AdaptiveGridHRBF(std::shared_ptr<RBF_Core> g_hrbf, 
                    std::vector<std::array<double, 3> >& output_vertices,
                    std::vector<std::array<size_t, 3> >& output_triangles)
{
    std::array<size_t,3> resolution = {3, 3, 3};
    std::vector<shared_ptr<ImplicitFunction<double>>> functions;
    using Vec3 = Eigen::Matrix<double, 3, 1>;
    using Vec4 = Eigen::Matrix<double, 4, 1>;

    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();

    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    double max_z = std::numeric_limits<double>::lowest();

    std::vector<Vec3> in_points(g_hrbf->pts.size()/3);
    for(int i = 0; i < g_hrbf->pts.size()/3; ++i)
    {
        in_points[i] =  {g_hrbf->pts[3*i], g_hrbf->pts[3*i +1], g_hrbf->pts[3*i +2]};
        min_x = min_x < g_hrbf->pts[3*i]     ? min_x : g_hrbf->pts[3*i];
        min_y = min_y < g_hrbf->pts[3*i + 1] ? min_y : g_hrbf->pts[3*i + 1];
        min_z = min_z < g_hrbf->pts[3*i + 2] ? min_z : g_hrbf->pts[3*i + 2];

        max_x = max_x > g_hrbf->pts[3*i]     ? max_x : g_hrbf->pts[3*i];
        max_y = max_y > g_hrbf->pts[3*i + 1] ? max_y : g_hrbf->pts[3*i + 1];
        max_z = max_z > g_hrbf->pts[3*i + 2] ? max_z : g_hrbf->pts[3*i + 2];
    }

    std::array<double,3> bbox_min = {min_x, min_y, min_z};
    std::array<double,3> bbox_max = {max_x, max_y, max_z};

    Eigen::VectorXd hrbf_a;
    hrbf_a.resize(g_hrbf->a.size());

    std::cout << " g_hrbf->a.size() " << g_hrbf->a.size() << std::endl;
    for(int i = 0; i < g_hrbf->a.size(); ++i)
    {
        hrbf_a[i] = g_hrbf->a[i];
    }
    Vec4 hrbf_b;
    for(int i =0; i < 4; ++i)
    {
        hrbf_b[i] = g_hrbf->b[i];
    }
    std::shared_ptr<ImplicitFunction<double>> hrbf_func = std::make_shared<Hermite_RBF<double>>(in_points, hrbf_a, hrbf_b, iso_offset_val_);
    functions.push_back(hrbf_func);
    auto g_t1 = Clock::now();
    auto t000 = Clock::now();

    
      
    GenerateAdaptiveGridOut(resolution, bbox_min, bbox_max, out_dir_,  
                            file_name_,  functions, adgrid_threshold_, output_vertices, output_triangles);
    
    auto t001 = Clock::now();
    G_VP_stats.adgrid_gen_time_ = std::chrono::nanoseconds(t001 - t000).count() / 1e9;

    printf("adaptive grid generation time : %f ! \n", G_VP_stats.adgrid_gen_time_);

}


void VIPSSUnit::SolveOptimizaiton()
{
    // auto tn0 = Clock::now();
    
    // auto tn1 = Clock::now();
    // double get_h_sub_block_time = std::chrono::nanoseconds(tn1 - tn0).count() / 1e9;
    // // printf("get H sub block time : %f ! \n", get_h_sub_block_time);
    // G_VP_stats.take_h_sub_block_time_ = get_h_sub_block_time;
    auto ts0 = Clock::now();
    Solver::open_log_ = true;
    std::cout << " user lambda : " << user_lambda_ << std::endl;
    std::cout << " user hard_constraints_ : " << hard_constraints_ << std::endl;
    if(user_lambda_ < 1e-12)
    {
        if (hard_constraints_)
        {
            OptUnitVipssNormalSimple();
        }
        else {
            OptUnitVipssNormalDirectSimple();
        }
    } else {
        if (hard_constraints_)
        {
            OptUnitVipssNormal();
        }
        else {
            OptUnitVipssNormalDirect();
        }
    }
    solver_.solvec_arma.clear();
    solver_.solveval.clear();
    arma_x_opt_g.resize(0);
    res_vec_g.resize(0);
    scsc_vec.resize(0);
    // arma_x_opt_g.data().squeeze();
    // res_vec_g.data().squeeze();
    // scsc_vec.data().squeeze();

    // Final_H_.clear();
    local_vipss_.final_h_eigen_.resize(0,0);
    local_vipss_.final_h_eigen_.data().squeeze();
    
    // printf("opt fun call time : %f \n", VIPSSUnit::opt_func_time_g);
#pragma omp parallel for
    for(int i = 0; i < newnormals_.size()/3; ++i)
    {
        double normal_len = sqrt(newnormals_[3*i] * newnormals_[3*i] 
        + newnormals_[3*i + 1] * newnormals_[3*i + 1] + newnormals_[3*i + 2] * newnormals_[3*i + 2]);

        newnormals_[3*i] /= normal_len;
        newnormals_[3*i + 1] /= normal_len;
        newnormals_[3*i + 2] /= normal_len;
    }

    auto ts1 = Clock::now();
    double solve_time = std::chrono::nanoseconds(ts1 - ts0).count() / 1e9;
    printf("opt solve time : %f ! \n", solve_time);
    printf("opt fun call count : %d \n", VIPSSUnit::opt_func_count_g);
    G_VP_stats.opt_solver_time_ = solve_time;
    G_VP_stats.opt_func_call_num_ = VIPSSUnit::opt_func_count_g;
}

void VIPSSUnit::Run()
{
    local_vipss_.out_dir_ = data_dir_ + "/" + file_name_ + "/";
    auto t00 = Clock::now();
    local_vipss_.voro_gen_.dense_input_ = is_dense_input_;
    if(MST_weight_type_ == 1)
    {
        local_vipss_.use_distance_weight_mst_ = true;
    }
    
    rbf_api_.Set_RBF_PARA();
    InitPtNormalWithLocalVipss();
    
    // std::string out_init_normal_path = out_dir_  + "/" + file_name_ + "_init_normal.ply";
    // writePLYFile_VN(out_init_normal_path, local_vipss_.out_pts_, local_vipss_.out_normals_);

    if(! only_use_nn_hrbf_surface_ )
    {
        SolveOptimizaiton();
    }
    
    if (is_surfacing_)
    {
        BuildNNHRBFFunctions();
    }
    
    // 
    if(LocalVipss::use_octree_sample_)
    {
        auto new_pts = local_vipss_.octree_leaf_pts_;
        auto ts00 = Clock::now();
        const auto&pts = local_vipss_.octree_split_leaf_pts_;
        // const auto&pts = local_vipss_.octree_leaf_pts_;
        std::cout << "split pts size : " << pts.size()/3 << std::endl;
        for(int i =0; i < pts.size()/3; ++i)
        {
            // printf("cur pt  : %f %f %f \n", pts[3*i], pts[3*i + 1], pts[3*i + 2] );
            R3Pt cur_pt(pts[3*i], pts[3*i + 1], pts[3*i + 2]);
            double cur_dist = LocalVipss::NNDistFunction(cur_pt);
            // printf("cur dist : %f \n", cur_dist );
            if(abs(cur_dist) > distfunc_threshold_)
            {
                new_pts.push_back(pts[3*i]);
                new_pts.push_back(pts[3*i + 1]);
                new_pts.push_back(pts[3*i + 2]);
            }
        }
        auto ts01 = Clock::now();
        double total_time = std::chrono::nanoseconds(ts01 - ts00).count()/1e9;
        printf("------- remaining pts dist function evaluation time : %f ! \n", total_time);
        // std::string octree_sample_path = out_dir_  + file_name_ +  "_octree_distSample.xyz";
        // writeXYZ(octree_sample_path, new_pts);
    }

    auto t01 = Clock::now();
    double total_time = std::chrono::nanoseconds(t01 - t00).count()/1e9;
    printf("total local vipss running time : %f ! \n", total_time);
    // std::string out_path  = local_vipss_.out_dir_ + local_vipss_.filename_  + "_opt";
    if(! only_use_nn_hrbf_surface_)
    {   
        // std::vector<tetgenmesh::point> hull_pts;
        // local_vipss_.voro_gen_.tetMesh_.outhullPts(&(local_vipss_.voro_gen_.tetIO_), hull_pts);
        std::vector<double> out_hull_pts;
        arma::vec3 hull_c= {0, 0, 0};   
        std::vector<arma::vec3> hull_normals;
        int normal_size = newnormals_.size();
        // std::cout << " newnormals_ size 00 " << normal_size << std::endl;
        for(auto pt: local_vipss_.voro_gen_.convex_hull_pts_)
        {
            hull_c[0] += pt[0];
            hull_c[1] += pt[1];
            hull_c[2] += pt[2];
            size_t pid = local_vipss_.voro_gen_.point_id_map_[pt];
            hull_normals.push_back({newnormals_[3* pid], newnormals_[3* pid + 1], newnormals_[3* pid+ 2]});
        }
        if(! local_vipss_.voro_gen_.convex_hull_pts_.empty())
        {
            hull_c = hull_c / double(local_vipss_.voro_gen_.convex_hull_pts_.size());
        }
        double sign_sum = 0;
        for(int i = 0; i < local_vipss_.voro_gen_.convex_hull_pts_.size(); ++i)
        {
            const auto& pt = local_vipss_.voro_gen_.convex_hull_pts_[i];
            arma::vec3 diff = {pt[0] - hull_c[0], pt[1] - hull_c[1], pt[2] - hull_c[2]};
            sign_sum += arma::dot(diff, hull_normals[i]); 
        }
        std::cout << " sign_sum value " << sign_sum << std::endl;
        auto out_normals  = newnormals_;
        std::cout << " newnormals_ size " <<(int) newnormals_.size() << std::endl;
        if(sign_sum < 0)
        {
            for( auto& ele : out_normals)
            {
                ele *= -1.0;
            }
        }
        std::cout << "recover data to origin scale !" << std::endl;
        // std::string hull_pt_path = "convex_hull_pts.xyz";
        // writeXYZ(hull_pt_path, out_hull_pts);
        std::vector<double> out_pts(local_vipss_.out_pts_.size(), 0);
        for(int i = 0; i < local_vipss_.out_pts_.size()/3; ++i)
        {
            out_pts[3*i] = local_vipss_.out_pts_[3*i] * local_vipss_.in_pt_scale_ + local_vipss_.in_pt_center_[0];
            out_pts[3*i + 1] = local_vipss_.out_pts_[3*i + 1] * local_vipss_.in_pt_scale_ + local_vipss_.in_pt_center_[1];
            out_pts[3*i + 2] = local_vipss_.out_pts_[3*i + 2] * local_vipss_.in_pt_scale_ + local_vipss_.in_pt_center_[2];
        }
        writePLYFile_VN(out_normal_path_, out_pts, out_normals);
        std::cout << " save estimated normal to file : " << out_normal_path_ << std::endl;
    }
    
    newnormals_.clear();
    // local_vipss_.out_pts_.clear();
    local_vipss_.voro_gen_.vcell_face_centers_.clear();
    // is_surfacing_ = false;
    if (is_surfacing_)
    {
    
        // std::string vipss_pt_path = "../../out/torus/torus_two_parts_out_normal.ply";
        if(use_adgrid_)
        {
            VoronoiGen::cluster_init_pids_.clear();
            GenerateAdaptiveGrid();
        } else {
            ReconSurface();
        }  
        local_vipss_.voro_gen_.SetInsertBoundaryPtsToUnused();
        // if(make_nn_const_neighbor_num_)
        // local_vipss_.voro_gen_.SetInsertBoundaryPtsToUnused();
    }
    // test_vipss_timing::test_local_vipss(input_data_path_);
    // test_vipss_timing::visual_distval_pt(input_data_path_, 200);
    // std::string out_csv_file = out_dir_ + file_name_ + "_time_stats_" + std::to_string(user_lambda_) + " .txt";
    std::string out_csv_file = out_dir_ + file_name_ + "_time_stats.txt";
    WriteStatsTimeCSV(out_csv_file, G_VP_stats);

    // std::string out_csv_re_file = out_dir_ + file_name_ + "_res.txt";
    // WriteVectorValsToCSV(out_csv_re_file, G_VP_stats.residuals_);
}

void VIPSSUnit::CalEnergyWithGtNormal()
{
    std::string norm_path = "";
    std::vector<double> vertices;
    std::vector<double> normals;
    readPLYFile(norm_path, vertices, normals);
    if(user_lambda_ <= 1e-12)
    {
        size_t n = vertices.size()/3;
        Eigen::VectorXd arma_x(n*3);
        for(int i=0;i<n;++i){
            // auto p_scsc = sina_cosa_sinb_cosb.data()+i*4;
            arma_x(i)     = normals[3*i];
            arma_x(i+n)   = normals[3*i + 1];
            arma_x(i+n*2) = normals[3*i + 2];
        }
        Eigen::VectorXd a2 = (arma_x.transpose() * local_vipss_.final_h_eigen_).transpose();
        double re = arma_x.dot(a2);
        std::cout << "final residual val : " << re << std::endl;
    } else {
        size_t n = vertices.size()/3;
        Eigen::VectorXd arma_x(n*4);
        for(int i=0;i<n;++i){
            // auto p_scsc = sina_cosa_sinb_cosb.data()+i*4;
            arma_x(i)       = s_func_vals_[i];
            arma_x(i + n)   = normals[3*i];
            arma_x(i + n*2) = normals[3*i + 1];
            arma_x(i + n*3) = normals[3*i + 2];
        }

        Eigen::VectorXd a2 = (arma_x.transpose() * local_vipss_.final_h_eigen_).transpose();
        double re = arma_x.dot(a2);
        std::cout << "final residual val : " << re << std::endl;
    }
}

void VIPSSUnit::CompareMeshDiff(std::shared_ptr<RBF_Core> rbf_func)
{
    std::string mesh_path = "../../out/test/kitten_h004_0.01_mesh_lv.obj";
    std::vector<double> vertices;
    std::vector<unsigned int> faces;
    std::vector<double> normals;
    readObjFile(mesh_path, vertices, faces, normals);
    double max_dist = 0.01;
    std::vector<std::array<double, 3>> vts;
    std::vector<std::array<double, 3>> colors;
    std::vector<std::vector<size_t>> out_faces;

    std::cout << " input vertices size : " << vertices.size()/3 << std::endl;
    for(int i = 0; i < vertices.size()/3; ++i)
    {
        double dist = rbf_func->Dist_Function(R3Pt(vertices[3*i], vertices[3*i + 1], vertices[3*i + 2]));
        double t = min(max_dist, abs(dist)) / max_dist;
        RGBColor color = ErrorColorBlend(t);
        vts.push_back({vertices[3*i], vertices[3*i + 1], vertices[3*i + 2]});
        colors.push_back({color.r, color.g, color.b});
    }
    for(int i = 0; i < faces.size()/3; ++i)
    {
        out_faces.push_back({faces[3*i], faces[3*i +1], faces[3*i + 2]});
    }

    std::string out_path = "../../out/test/kitten_h004_0.01_mesh_lv_color.obj";
    std::cout << "output mesh path : " << out_path << std::endl;
    writePlyMeshWithColor(out_path, vts, colors, out_faces);

}