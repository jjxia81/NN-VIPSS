#include "rbfcore.h"
#include "utility.h"
#include "Solver.h"
#include <armadillo>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <ctime>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <queue>
#include "readers.h"
//#include "mymesh/UnionFind.h"
//#include "mymesh/tinyply.h"

using namespace std;

typedef std::chrono::high_resolution_clock Clock;
double randomdouble() {return static_cast <double> (rand()) / static_cast <double> (RAND_MAX);}
double randomdouble(double be,double ed) {return be + randomdouble()*(ed-be);	}

void RBF_Core::NormalRecification(double maxlen, std::vector<double>&nors){


    double maxlen_r = -1;
    auto p_vn = nors.data();
    size_t  np = nors.size()/3;
    if(1){
        for(size_t i=0;i<np;++i){
            maxlen_r = max(maxlen_r,MyUtility::normVec(p_vn+i*3));
        }

        cout<<"maxlen_r: "<<maxlen_r<<endl;
        double ratio = maxlen / maxlen_r;
        for(auto &a:nors)a*=ratio;
    }else{
        for(size_t i=0;i<np;++i){
            MyUtility::normalize(p_vn+i*3);
        }

    }




}

bool RBF_Core::Write_Hermite_NormalPrediction(std::string fname, int mode){


//    std::vector<uchar>labelcolor(npt*4);
//    std::vector<uint>f2v;
//    uchar red[] = {255,0,0, 255};
//    uchar green[] = {0,255,0, 255};
//    uchar blue[] = {0,0,255, 255};
//    for(int i=0;i<labels.size();++i){
//        uchar *pcolor;
//        if(labels[i]==0)pcolor = green;
//        else if(labels[i]==-1)pcolor = blue;
//        else if(labels[i]==1)pcolor = red;
//        for(int j=0;j<4;++j)labelcolor[i*4+j] = pcolor[j];
//    }
    //fname += mp_RBF_METHOD[curMethod];

//    for(int i=0;i<npt;++i){
//        uchar *pcolor = green;
//        for(int j=0;j<4;++j)labelcolor[i*4+j] = pcolor[j];
//    }

    std::vector<double>nors;
    if(mode ==0)nors=initnormals;
    else if(mode == 1)nors=newnormals;
    else if(mode == 2)nors = initnormals_uninorm;
    NormalRecification(1.,nors);

    //for(int i=0;i<npt;++i)if(randomdouble()<0.5)MyUtility::negVec(nors.data()+i*3);
    //cout<<pts.size()<<' '<<f2v.size()<<' '<<nors.size()<<' '<<labelcolor.size()<<endl;
    //writePLYFile(fname,pts,f2v,nors,labelcolor);

//    writeObjFile_vn(fname,pts,nors);
    writePLYFile_VN(fname,pts,nors);

    return 1;
}

void RBF_Core::BuildUnitVipssMat(std::vector<double>&pts)
{
    key_npt = npt;
    // Init(XCube);
    Set_HermiteRBF(pts);
    size_t m_dim = npt + 3* key_npt;
    bigM.zeros(m_dim + 4, m_dim + 4);
    bigM.submat(0,0,m_dim-1,m_dim-1) = M;
    bigM.submat(0,m_dim,m_dim-1, m_dim + 3) = N;
    bigM.submat(m_dim,0,m_dim + 3, m_dim-1) = N.t();

    //for(int i=0;i<4;++i)bigM(i+(npt)*4,i+(npt)*4) = 1;

    auto t2 = Clock::now();
    bigMinv = inv(bigM);

    bigM_inv_time = std::chrono::nanoseconds(Clock::now() - t2).count()/1e9;

    // printf("big M size %llu, %f \n", bigMinv.n_rows, bigM_inv_time);

    if(open_debug_log)
    cout<<"bigMinv inv: "<<(std::chrono::nanoseconds(Clock::now() - t2).count()/1e9)<<endl;
    bigM.clear();
    Minv = bigMinv.submat(0,0,m_dim-1,m_dim-1);
    // Ninv = bigMinv.submat(0,m_dim,m_dim-1, m_dim + 3);
    bigMinv.clear();
    //K = Minv - Ninv *(N.t()*Minv);
    // K = Minv;
    // K00 = K.submat(0,0,npt-1,npt-1);
    // K01 = K.submat(0,npt,npt-1,m_dim-1);
    // K11 = K.submat( npt, npt, m_dim-1, m_dim-1 );
    M.clear();N.clear();
    
}


void RBF_Core::Set_HermiteRBF(std::vector<double>&pts){

    if(open_debug_log)
    cout<<"Set_HermiteRBF"<<endl;
    //for(auto a:pts)cout<<a<<' ';cout<<endl;
    isHermite = true;
    npt = pts.size() / 3;
    size_t m_dim = npt + 3 * key_npt;
    a.set_size(m_dim);
    M.set_size(m_dim,m_dim);
    double *p_pts = pts.data();
    
    for(int i=0;i<npt;++i){
        for(int j=i;j<npt;++j){
            M(i,j) = M(j,i) = Kernal_Function_2p(p_pts+i*3, p_pts+j*3);
        }
    }
    double G[3];
    for(int i=0;i<npt;++i){
        for(int j=0;j<key_npt;++j){
            Kernal_Gradient_Function_2p(p_pts+i*3, p_pts+j*3, G);
            for(int k=0;k<3;++k)M(i,npt+j+k*key_npt) = G[k];
            for(int k=0;k<3;++k)M(npt+j+k*key_npt,i) = G[k];

        }
    }
    double H[9];
    for(int i=0;i<key_npt;++i){
        for(int j=i;j<key_npt;++j){
            Kernal_Hessian_Function_2p(p_pts+i*3, p_pts+j*3, H);
            for(int k=0;k<3;++k)
                for(int l=0;l<3;++l)
                    M(npt+j+l*key_npt,npt+i+k*key_npt) = M(npt+i+k*key_npt,npt+j+l*key_npt) = -H[k*3+l];
        }
    }

    //cout<<std::setprecision(5)<<std::fixed<<M<<endl;

    bsize= 4;
    N.zeros(m_dim, 4);
    b.set_size(4);

    for(int i=0;i<npt;++i){
        N(i,0) = 1;
        for(int j=0;j<3;++j)N(i,j+1) = pts[i*3+j];
    }
    for(int i=0;i<key_npt;++i){
        for(int j=0;j<3;++j)N(npt+i+j*key_npt,j+1) = -1;
    }
}



void RBF_Core::Set_HermiteRBF(const std::vector<double*>&in_pts){

    if(open_debug_log)
    cout<<"Set_HermiteRBF"<<endl;
    //for(auto a:pts)cout<<a<<' ';cout<<endl;
    isHermite = true;
    npt = in_pts.size();
    size_t m_dim = npt + 3 * key_npt;
    a.set_size(m_dim);
    M.set_size(m_dim,m_dim);
    
    for(int i=0;i<npt;++i){
        for(int j=i;j<npt;++j){
            M(i,j) = M(j,i) = Kernal_Function_2p(in_pts[i], in_pts[j]);
        }
    }
    double G[3];
    for(int i=0;i<npt;++i){
        for(int j=0;j<key_npt;++j){
            Kernal_Gradient_Function_2p(in_pts[i], in_pts[j], G);
            for(int k=0;k<3;++k)M(i,npt+j+k*key_npt) = G[k];
            for(int k=0;k<3;++k)M(npt+j+k*key_npt,i) = G[k];

        }
    }
    double H[9];
    for(int i=0;i<key_npt;++i){
        for(int j=i;j<key_npt;++j){
            Kernal_Hessian_Function_2p(in_pts[i], in_pts[j], H);
            for(int k=0;k<3;++k)
                for(int l=0;l<3;++l)
                    M(npt+j+l*key_npt,npt+i+k*key_npt) = M(npt+i+k*key_npt,npt+j+l*key_npt) = -H[k*3+l];
        }
    }

    bsize= 4;
    N.zeros(m_dim, 4);
    b.set_size(4);

    for(int i=0;i<npt;++i){
        N(i,0) = 1;
        for(int j=0;j<3;++j)N(i,j+1) = in_pts[i][j];
    }
    for(int i=0;i<key_npt;++i){
        for(int j=0;j<3;++j)N(npt+i+j*key_npt,j+1) = -1;
    }

}


double Gaussian_2p(const double *p1, const double *p2, double sigma){

    return exp(-MyUtility::vecSquareDist(p1,p2)/(2*sigma*sigma));
}



void RBF_Core::Set_Actual_User_LSCoef(double user_ls){

    User_Lamnbda = User_Lamnbda_inject = user_ls > 0 ?  user_ls : 0;

}

void RBF_Core::Set_Actual_Hermite_LSCoef(double hermite_ls){

    ls_coef = Hermite_ls_weight_inject = hermite_ls > 0?hermite_ls:0;
}

void RBF_Core::Set_SparsePara(double spa){
    sparse_para = spa;
}

void RBF_Core::Set_User_Lamnda_ToMatrix(double user_ls){


    {
        Set_Actual_User_LSCoef(user_ls);
        auto t1 = Clock::now();
        if(open_debug_log)
        cout<<"setting K, HermiteApprox_Lamnda"<<endl;
        if(User_Lamnbda>0){
            arma::sp_mat eye;
            eye.eye(npt,npt);

            dI = inv(eye + User_Lamnbda*K00);
            saveK_finalH = K = K11 - (User_Lamnbda)*(K01.t()*dI*K01);

        }else saveK_finalH = K = K11;
        if(open_debug_log)
        cout<<"solved: "<<(std::chrono::nanoseconds(Clock::now() - t1).count()/1e9)<<endl;
    }

    finalH = saveK_finalH;

}

void RBF_Core::Set_HermiteApprox_Lamnda(double hermite_ls){


    {
        Set_Actual_Hermite_LSCoef(hermite_ls);
        auto t1 = Clock::now();
        if(open_debug_log)
        cout<<"setting K, HermiteApprox_Lamnda"<<endl;
        if(ls_coef>0){
            arma::sp_mat eye;
            eye.eye(npt,npt);

            if(ls_coef > 0){
                arma:: mat tmpdI = inv(eye + (ls_coef+User_Lamnbda)*K00);
                K = K11 - (ls_coef+User_Lamnbda)*(K01.t()*tmpdI*K01);
            }else{
                K = saveK_finalH;
            }
        }
        if(open_debug_log)
        cout<<"solved: "<<(std::chrono::nanoseconds(Clock::now() - t1).count()/1e9)<<endl;    
    }


}



void RBF_Core::Set_Hermite_PredictNormal(std::vector<double>&pts){

    Set_HermiteRBF(pts);
    auto t1 = Clock::now();
    // cout<<"setting K"<<endl;
    if(!isnewformula){
        arma::mat D = N.t()*Minv;
        K = Minv - D.t()*inv(D*N)*D;
        K = K.submat( npt, npt, npt*4-1, npt*4-1 );
        finalH = saveK_finalH = K;
    }else{
        // if(open_debug_log)
        // cout<<"using new formula"<<endl;
        // M = arma::symmatu(M);
        size_t m_dim = npt + 3* key_npt;
        bigM.zeros(m_dim + 4, m_dim + 4);
        bigM.submat(0,0,m_dim-1,m_dim-1) = M;
        bigM.submat(0,m_dim,m_dim-1, m_dim + 3) = N;
        bigM.submat(m_dim,0,m_dim + 3, m_dim-1) = N.t();

        //for(int i=0;i<4;++i)bigM(i+(npt)*4,i+(npt)*4) = 1;

        auto t2 = Clock::now();
        // cout << " start bigMinv inv " <<  endl;
        bigMinv = inv(bigM);
        // bigMinv = inv(bigM, arma::inv_opts::allow_approx);
        //   cout << " finish bigMinv inv " <<  endl;
        if(open_debug_log)
        cout<<"bigMinv: "<<(setK_time= std::chrono::nanoseconds(Clock::now() - t2).count()/1e9)<<endl;
		bigM.clear();
        Minv = bigMinv.submat(0,0,m_dim-1,m_dim-1);
        Ninv = bigMinv.submat(0,m_dim,m_dim-1, m_dim + 3);

        bigMinv.clear();
        //K = Minv - Ninv *(N.t()*Minv);
        K = Minv;
        K00 = K.submat(0,0,npt-1,npt-1);
        K01 = K.submat(0,npt,npt-1,m_dim-1);
        K11 = K.submat( npt, npt, m_dim-1, m_dim-1 );
        M.clear();N.clear();
        // cout<<"K11: "<<K11.n_cols<<endl;
        //Set_Hermite_DesignedCurve();
        Set_User_Lamnda_ToMatrix(User_Lamnbda_inject);
        
        // double cur_lambda = user_lambda_;
        

        // printf("build H \n");

        
        // finalH = K;
//		arma::vec eigval, ny;
//		arma::mat eigvec;
//		ny = eig_sym( eigval, eigvec, K);
//		cout<<ny<<endl;
        // cout<<"K: "<<K.n_cols<<endl;
    }




    //K = ( K.t() + K )/2;
    if(open_debug_log)
    cout<<"solve K total: "<<(setK_time= std::chrono::nanoseconds(Clock::now() - t1).count()/1e9)<<endl;
    return;

}



void RBF_Core::SetInitnormal_Uninorm(){

    initnormals_uninorm = initnormals;
    for(int i=0;i<key_npt;++i)MyUtility::normalize(initnormals_uninorm.data()+i*3);
    initnormals = initnormals_uninorm;

}

int RBF_Core::Solve_Hermite_PredictNormal_UnitNorm(){

    arma::vec eigval, ny;
    arma::mat eigvec;

    if(!isuse_sparse){
        K = symmatu(K);
        ny = eig_sym( eigval, eigvec, K);
    }else{
//		cout<<"use sparse eigen"<<endl;
//        int k = 4;
//        do{
//            ny = eigs_sym( eigval, eigvec, sp_K, k, "sa" );
//            k+=4;
//        }while(ny(0)==0);
    }

    if(open_debug_log)
    cout<<"eigval(0): "<<eigval(0)<<endl;

    int smalleig = 0;

    initnormals.resize(key_npt*3);
    arma::vec y(npt + 3 * key_npt);
    for(int i=0;i<npt;++i)y(i) = 0;
    
    for(int i=0;i<key_npt*3;++i)y(i+npt) = eigvec(i,smalleig);

    for(int i=0;i<key_npt;++i){
        initnormals[i*3]   = y(npt+i);
        initnormals[i*3+1] = y(npt+i+key_npt);
        initnormals[i*3+2] = y(npt+i+key_npt*2);
        //MyUtility::normalize(normals.data()+i*3);
    }

    SetInitnormal_Uninorm();
    if(open_debug_log)
    cout<<"Solve_Hermite_PredictNormal_UnitNorm finish"<<endl;
    return 1;
}



/***************************************************************************************************/
/***************************************************************************************************/
double acc_time;

static int countopt = 0;

double optfunc_Hermite(const std::vector<double>&x, std::vector<double>&grad, void *fdata){

    auto t1 = Clock::now();
    RBF_Core *drbf = reinterpret_cast<RBF_Core*>(fdata);
    // size_t n = drbf->npt;
    size_t n = drbf->key_npt;
    arma::vec arma_x(n*3);

    //(  sin(a)cos(b), sin(a)sin(b), cos(a)  )  a =>[0, pi], b => [-pi, pi];
    std::vector<double>sina_cosa_sinb_cosb(n * 4);
    for(size_t i=0;i<n;++i){
        size_t ind = i*4;
        sina_cosa_sinb_cosb[ind] = sin(x[i*2]);
        sina_cosa_sinb_cosb[ind+1] = cos(x[i*2]);
        sina_cosa_sinb_cosb[ind+2] = sin(x[i*2+1]);
        sina_cosa_sinb_cosb[ind+3] = cos(x[i*2+1]);
    }

    for(int i=0;i<n;++i){
        auto p_scsc = sina_cosa_sinb_cosb.data()+i*4;
        //        int ind = i*3;
        //        arma_x(ind) = p_scsc[0] * p_scsc[3];
        //        arma_x(ind+1) = p_scsc[0] * p_scsc[2];
        //        arma_x(ind+2) = p_scsc[1];
        arma_x(i) = p_scsc[0] * p_scsc[3];
        arma_x(i+n) = p_scsc[0] * p_scsc[2];
        arma_x(i+n*2) = p_scsc[1];
    }

    arma::vec a2;
    //if(drbf->isuse_sparse)a2 = drbf->sp_H * arma_x;
    //else
    a2 = drbf->finalH * arma_x;


    if (!grad.empty()) {

        grad.resize(n*2);

        for(size_t i=0;i<n;++i){
            auto p_scsc = sina_cosa_sinb_cosb.data()+i*4;

            //            int ind = i*3;
            //            grad[i*2] = a2(ind) * p_scsc[1] * p_scsc[3] + a2(ind+1) * p_scsc[1] * p_scsc[2] - a2(ind+2) * p_scsc[0];
            //            grad[i*2+1] = -a2(ind) * p_scsc[0] * p_scsc[2] + a2(ind+1) * p_scsc[0] * p_scsc[3];

            grad[i*2] = a2(i) * p_scsc[1] * p_scsc[3] + a2(i+n) * p_scsc[1] * p_scsc[2] - a2(i+n*2) * p_scsc[0];
            grad[i*2+1] = -a2(i) * p_scsc[0] * p_scsc[2] + a2(i+n) * p_scsc[0] * p_scsc[3];

        }
    }

    double re = arma::dot( arma_x, a2 );
    countopt++;

    acc_time+=(std::chrono::nanoseconds(Clock::now() - t1).count()/1e9);

    //cout<<countopt++<<' '<<re<<endl;
    return re;

}


double optfunc_unit_vipss_direct2(const std::vector<double>&x, std::vector<double>&grad, void *fdata){

    auto t1 = Clock::now();
    RBF_Core *drbf = reinterpret_cast<RBF_Core*>(fdata);
    // VIPSSUnit *drbf = reinterpret_cast<VIPSSUnit*>(fdata);
    // size_t n = drbf->npt;
    size_t n = drbf->npt;
    size_t kn = drbf->key_npt;
    size_t u_size = 4;
    arma::vec arma_x( n + 3 * kn);

    // printf("start to call opt ! \n");
    for(int i=0;i<n;++i){
        // auto p_scsc = sina_cosa_sinb_cosb.data()+i*4;
        
        if(i <  kn)
        {
            arma_x(i+n )   = x[n + 3*i + 0];
            arma_x(i+n + kn) = x[n + 3*i + 1];
            arma_x(i+n + kn * 2) = x[n + 3*i + 2];
            arma_x(i)     = x[i];
        } else {
            arma_x(i)     = drbf->user_beta * x[i];
            // arma_x(i+n )   = x[n + 3*i + 0];
        }
    }
    arma::vec a2;
    a2 = drbf->finalH * arma_x;

    // printf("start to call opt 11 ! \n");
    if (!grad.empty()) {
        // std::cout << " grad size " <<grad.size() << std::endl;
        grad.resize(n*u_size);
        for(size_t i=0;i<n;++i){
            grad[i]   = 2 * a2(i);
            if(i < kn)
            {
                grad[n + i*3+0] = 2 * a2(i + n);
                grad[n + i*3+1] = 2 * a2(i + n + kn);
                grad[n + i*3+2] = 2 * a2(i + n + kn*2);
            }
            
        }
    }
    double re = arma::dot(arma_x, a2);
    double alpha = 100.0;

    for(size_t id =0; id < kn; ++id)
    {
        double cur_re = x[n + 3*id + 0] * x[n + 3*id + 0] + x[n + 3*id + 1] * x[n + 3*id + 1] 
                        + x[n + 3*id + 2] * x[n + 3*id + 2] - 1;
        if(!grad.empty()) 
        {
            grad[n + 3*id + 0] += alpha * 2 * x[n + 3*id + 0] * cur_re; 
            grad[n + 3*id + 1] += alpha * 2 * x[n + 3*id + 1] * cur_re; 
            grad[n + 3*id + 2] += alpha * 2 * x[n + 3*id + 2] * cur_re; 
        }
        re += alpha* cur_re * cur_re;
    }
    // printf("res val : %f \n", re);
    return re;
}

void RBF_Core::OptUnitVipssNormalDirect(){

    // printf("start to call solver ! \n");

    size_t u_size = 4;
    sol.solveval.resize(npt + 3 * key_npt);

    arma::mat E(npt, npt);
    E.eye();
    arma::mat F(npt + key_npt*3, npt + key_npt*3);
    F(0, 0, arma::size(npt, npt)) = E;
    // printf("build H 00 \n");
    finalH = (F + Minv * User_Lamnbda);

    for(size_t i=0;i<npt;++i){
        sol.solveval[i]     = 0;
        if(i < key_npt)
        {
            double *veccc = initnormals.data()+i*3;
            sol.solveval[npt + i*3 + 0] = veccc[0];
            sol.solveval[npt + i*3 + 1] = veccc[1];
            sol.solveval[npt + i*3 + 2] = veccc[2];
        }
    }
    std::vector<double>upper(npt + 3 * key_npt);
    std::vector<double>lower(npt + 3 * key_npt);

    //  printf("start to call solver 11! \n");

    for(int i=0;i<npt;++i){
        upper[i] = 1;
        lower[i] = -1.0;

        if(i < key_npt)
        {
            for(size_t j = 0; j < 3; ++j)
            {
                upper[npt + i*3 + j] = 1;
                lower[npt + i*3 + j] = -1.0;
            }
        }
    }
    //  printf("start to call solver 22! \n");

    Solver::nloptwrapperDirect(lower,upper,optfunc_unit_vipss_direct2,
                this, 1e-6, 2000, sol);
    // Solver::nloptwrapper(lower,upper,optfunc_unit_vipss_simple,this,1e-7,3000,solver_);
    newnormals.resize(key_npt*3);
    // arma::vec y(npt_ + 3 * npt_);
    // for(size_t i=0;i<npt_;++i)y(i) = 0;
    arma::vec y(npt + 3 * key_npt);
    for(size_t i=0;i<npt;++i)y(i) = sol.solveval[i];

    for(size_t i=0;i<key_npt;++i){
        newnormals[i*3]   = sol.solveval[npt + i*3 + 0];
        newnormals[i*3+1] = sol.solveval[npt + i*3 + 1];
        newnormals[i*3+2] = sol.solveval[npt + i*3 + 2];
        MyUtility::normalize(newnormals.data()+i*3);

        y(npt+i) = newnormals[i*3];
        y(npt+i+key_npt) = newnormals[i*3+1];
        y(npt+i+key_npt*2) = newnormals[i*3+2];
        // 
    }
    // Set_RBFCoef(y);

    a = Minv*y;
    b = Ninv.t()*y;

    return;
}


int RBF_Core::Opt_Hermite_PredictNormal_UnitNormal(){

    
    sol.solveval.resize(key_npt * 2);

    for(size_t i=0;i<key_npt;++i){
        double *veccc = initnormals.data()+i*3;
        {
            //MyUtility::normalize(veccc);
            sol.solveval[i*2] = atan2(sqrt(veccc[0]*veccc[0]+veccc[1]*veccc[1]),veccc[2] );
            sol.solveval[i*2 + 1] = atan2( veccc[1], veccc[0]   );
        }

    }
    //cout<<"smallvec: "<<smallvec<<endl;

    if(1){
        std::vector<double>upper(key_npt*2);
        std::vector<double>lower(key_npt*2);
        for(int i=0;i<key_npt;++i){
            upper[i*2] = 2 * my_PI;
            upper[i*2 + 1] = 2 * my_PI;

            lower[i*2] = -2 * my_PI;
            lower[i*2 + 1] = -2 * my_PI;
        }

        countopt = 0;
        acc_time = 0;

        //LocalIterativeSolver(sol,kk==0?normals:newnormals,300,1e-7);
        Solver::nloptwrapper(lower,upper,optfunc_Hermite,this,1e-7,3000,sol);
        if(open_debug_log)
        cout<<"number of call: "<<countopt<<" t: "<<acc_time<<" ave: "<<acc_time/countopt<<endl;
        callfunc_time = acc_time;
        solve_time = sol.time;
        //for(int i=0;i<npt;++i)cout<< sol.solveval[i]<<' ';cout<<endl;

    }
    newnormals.resize(key_npt*3);
    arma::vec y(npt + 3 * key_npt);
    for(size_t i=0;i<npt;++i)y(i) = 0;
    for(size_t i=0;i<key_npt;++i){

        double a = sol.solveval[i*2], b = sol.solveval[i*2+1];
        newnormals[i*3]   = y(npt+i) = sin(a) * cos(b);
        newnormals[i*3+1] = y(npt+i+key_npt) = sin(a) * sin(b);
        newnormals[i*3+2] = y(npt+i+key_npt*2) = cos(a);
        MyUtility::normalize(newnormals.data()+i*3);
    }
    y *= -1.0;
    Set_RBFCoef(y);

    //sol.energy = arma::dot(a,M*a);
    if(open_debug_log)
    cout<<"Opt_Hermite_PredictNormal_UnitNormal"<<endl;
    return 1;
}

void RBF_Core::Set_RBFCoefWithInitNormal()
{
    
    // newnormals = initnormals;
    arma::vec y(npt + 3 * key_npt);
    for(size_t i=0;i<npt;++i)y(i) = 0;
    for(size_t i=0;i<key_npt;++i){
        y(npt+i) = initnormals[i*3];
        y(npt+i+key_npt) = initnormals[i*3+1];
        y(npt+i+key_npt*2) = initnormals[i*3+2];
    }
    Set_RBFCoef(y);
}


void RBF_Core::Set_RBFCoefWithOptNormalAndSval(const std::vector<double>& Vn, 
                                                const std::vector<double>& s_vals )
{
    
    newnormals = Vn;
    arma::vec y(npt + 3 * key_npt);
    for(size_t i=0;i<key_npt;++i){
        y(npt+i) = -newnormals[i*3];
        y(npt+i+key_npt) = -newnormals[i*3+1];
        y(npt+i+key_npt*2) = -newnormals[i*3+2];
    }
    if(User_Lamnbda>0)
    {
        for(size_t i = 0; i < npt; ++i)
        {
            y(i) = s_vals[i];
        }
    } else {
        for(size_t i=0;i<npt;++i)y(i) = 0;
    }

    y.subvec(0,npt-1);
    a = Minv*y;
    b = Ninv.t()*y;

    kern_.resize(npt + 3*key_npt); 
    kb_.resize(4);
    kb_[0] = 1;
}


void RBF_Core::Solve_RBFCoefWithOptNormalAndSval(const std::vector<double>& Vn, 
                                                const std::vector<double>& s_vals )
{
    
    newnormals = Vn;
    arma::vec y(npt + 3 * key_npt + 4);
    for(size_t i=0;i<key_npt;++i){
        y(npt+i) = Vn[i*3];
        y(npt+i+key_npt) = Vn[i*3+1];
        y(npt+i+key_npt*2) = Vn[i*3+2];
    }
    // if(User_Lamnbda>0)
    {
        for(size_t i = 0; i < npt; ++i)
        {
            y(i) = s_vals[i];
        }
    } 
    // else {
    //     for(size_t i=0;i<npt;++i)y(i) = 0;
    // }
    
    size_t m_dim = npt + 3* key_npt;
    bigM.zeros(m_dim + 4, m_dim + 4);
    bigM.submat(0,0,m_dim-1,m_dim-1) = M;
    bigM.submat(0,m_dim,m_dim-1, m_dim + 3) = N;
    bigM.submat(m_dim,0,m_dim + 3, m_dim-1) = N.t();  

    M.clear();
    N.clear();

    // size_t mat_mem_size = bigM.n_elem * sizeof(double) / (1024 * 1024 );
    // printf("------mat_mem_size %llu MB\n", mat_mem_size);
    // auto t0 = Clock::now();
    // arma::mat X2 = arma::solve(bigM, y, arma::solve_opts::fast);

    arma::mat X2 = arma::solve(bigM, y, arma::solve_opts::likely_sympd);

    // arma::mat X2 = arma::inv(bigM) * y;
    bigM.clear();
    newnormals.clear();

    // auto t1 = Clock::now();
    // double t_time =  std::chrono::nanoseconds(t1 - t0).count()/1e9;
    // printf("pure solve linear system size %llu and time: %f \n", m_dim, t_time);
    // X2 = X2 * y;
    a = X2(0, 0, arma::size(npt + 3 * key_npt, 1));
    b = X2(npt + 3 * key_npt, 0, arma::size(4, 1));

    kern_.resize(npt + 3*key_npt); 
    kb_.resize(4);
    kb_[0] = 1;
}


void RBF_Core::Set_RBFCoef(arma::vec &y){
    if(open_debug_log)
    cout<<"Set_RBFCoef"<<endl;
    if(curMethod==HandCraft){
        cout<<"HandCraft, not RBF"<<endl;
        return;
    }
    if(!isnewformula){
        b = bprey * y;
        a = Minv * (y - N*b);
    }else{
        // if(User_Lamnbda>0)y.subvec(0,npt-1) = -User_Lamnbda*dI*K01*y.subvec(npt,npt*4-1);
        if(User_Lamnbda>0)y.subvec(0,npt-1) = -User_Lamnbda*dI*K01*y.subvec(npt,npt + 3* key_npt-1);
        a = Minv*y;
        b = Ninv.t()*y;
        // a.save("a.txt", arma::arma_ascii);
    }
    kern_.resize(npt + 3*key_npt); 
    kb_.resize(4);
    kb_[0] = 1;
}



int RBF_Core::Lamnbda_Search_GlobalEigen(){

    std::vector<double>lamnbda_list({0, 0.001, 0.01, 0.1, 1});
    //std::vector<double>lamnbda_list({  0.5,0.6,0.7,0.8,0.9,1,1.1,1.5,2,3});
    //lamnbda_list.clear();
    //for(double i=1.5;i<2.5;i+=0.1)lamnbda_list.push_back(i);
    //std::vector<double>lamnbda_list({0});
    std::vector<double>initen_list(lamnbda_list.size());
    std::vector<double>finalen_list(lamnbda_list.size());
    std::vector<std::vector<double>>init_normallist;
    std::vector<std::vector<double>>opt_normallist;

    lamnbda_list_sa = lamnbda_list;
    for(int i=0;i<lamnbda_list.size();++i){

        Set_HermiteApprox_Lamnda(lamnbda_list[i]);

        if(curMethod==Hermite_UnitNormal){
            Solve_Hermite_PredictNormal_UnitNorm();
        }

        //Solve_Hermite_PredictNormal_UnitNorm();
        OptNormal(1);

        initen_list[i] = sol.init_energy;
        finalen_list[i] = sol.energy;

        init_normallist.emplace_back(initnormals);
        opt_normallist.emplace_back(newnormals);
    }

    lamnbdaGlobal_Be.emplace_back(initen_list);
    lamnbdaGlobal_Ed.emplace_back(finalen_list);

    cout<<std::setprecision(8);
    if(open_debug_log)
    {
        for(int i=0;i<initen_list.size();++i){
        cout<<lamnbda_list[i]<<": "<<initen_list[i]<<" -> "<<finalen_list[i]<<endl;
        }
    }
    

    int minind = min_element(finalen_list.begin(),finalen_list.end()) - finalen_list.begin();
    if(open_debug_log)
    {
        cout<<"min energy: "<<endl;
        cout<<lamnbda_list[minind]<<": "<<initen_list[minind]<<" -> "<<finalen_list[minind]<<endl;
    }
    


    initnormals = init_normallist[minind];
    SetInitnormal_Uninorm();
    newnormals = opt_normallist[minind];
	return 1;
}


void RBF_Core::Print_LamnbdaSearchTest(std::string fname){


    cout<<setprecision(7);
    cout<<"Print_LamnbdaSearchTest"<<endl;
    for(int i=0;i<lamnbda_list_sa.size();++i)cout<<lamnbda_list_sa[i]<<' ';cout<<endl;
    cout<<lamnbdaGlobal_Be.size()<<endl;
    for(int i=0;i<lamnbdaGlobal_Be.size();++i){
        for(int j=0;j<lamnbdaGlobal_Be[i].size();++j){
            cout<<lamnbdaGlobal_Be[i][j]<<"\t"<<lamnbdaGlobal_Ed[i][j]<<"\t";
        }
        cout<<gtBe[i]<<"\t"<<gtEd[i]<<endl;
    }

    ofstream fout(fname);
    fout<<setprecision(7);
    if(!fout.fail()){
        for(int i=0;i<lamnbda_list_sa.size();++i)fout<<lamnbda_list_sa[i]<<' ';fout<<endl;
        fout<<lamnbdaGlobal_Be.size()<<endl;
        for(int i=0;i<lamnbdaGlobal_Be.size();++i){
            for(int j=0;j<lamnbdaGlobal_Be[i].size();++j){
                fout<<lamnbdaGlobal_Be[i][j]<<"\t"<<lamnbdaGlobal_Ed[i][j]<<"\t";
            }
            fout<<gtBe[i]<<"\t"<<gtEd[i]<<endl;
        }
    }
    fout.close();

}


