
#include <math.h>
#include "kernel.h"

namespace VIPSSKernel
{
double sigma = 2.0;
double inv_sigma_squarex2 = 1/(2 * pow(sigma, 2));
double Gaussian_Kernel(const double x_square){

    return exp(-x_square*inv_sigma_squarex2);
}



inline double Vec_Dist(const double *p1, const double *p2)
{
    double dx = p1[0] - p2[0];
    double dy = p1[1] - p2[1];
    double dz = p1[2] - p2[2];
    double dist = sqrt(dx * dx + dy * dy + dz * dz);
    return dist ;
}

inline double Vec_Dot(const double *p1, const double *p2)
{
    return p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2];
}

inline double Vec_Len(const double *p1)
{
    double len = p1[0] * p1[0] + p1[1] * p1[1] + p1[2] * p1[2];
    return sqrt(len);
}

double Gaussian_Kernel_2p(const double *p1, const double *p2){
    
    double s_dist = Vec_Dist(p1, p2);
    return Gaussian_Kernel(s_dist);

}

double Gaussian_PKernel_Dirichlet_2p(const double *p1, const double *p2){

    double s_dist = Vec_Dist(p1, p2);
    return (6*sigma*sigma-s_dist)*sqrt(Gaussian_Kernel(s_dist));
}

double Gaussian_PKernel_Bending_2p(const double *p1, const double *p2){


    double d2 = Vec_Dist(p1, p2);
    double d4 = d2*d2;
    double sigma2 = sigma * sigma;
    double sigma4 = sigma2 * sigma2;
    return (60*sigma4-20*sigma2*d2+d4)*sqrt(Gaussian_Kernel(d2));
}


double XCube_Kernel(const double x){

    return pow(x,3);
}

double XCube_Kernel_2p(const double *p1, const double *p2){


    return XCube_Kernel(Vec_Dist(p1,p2));

}

void XCube_Gradient_Kernel_2p(const double *p1, const double *p2, double *G){


    double len_dist  = Vec_Dist(p1,p2);
    for(int i=0;i<3;++i)G[i] = 3*len_dist*(p1[i]-p2[i]);
    return;

}


double XCube_GradientDot_Kernel_2p(const double *p1, const double *p2, const double *p3){


    double G[3];
    XCube_Gradient_Kernel_2p(p1,p2,G);
    return Vec_Dot(p3,G);

}

void XCube_Hessian_Kernel_2p(const double *p1, const double *p2, double *H){


    double diff[3];
    for(int i=0;i<3;++i)diff[i] = p1[i] - p2[i];
    double len_dist  = Vec_Len(diff);

    if(len_dist<1e-8){
        for(int i=0;i<9;++i)H[i] = 0;
    }else{
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)
            if(i==j)H[i*3+j] = 3 * pow(diff[i],2) / len_dist + 3 * len_dist;
            else H[i*3+j] = 3 * diff[i] * diff[j] / len_dist;
    }
    return;
}

void XCube_HessianDot_Kernel_2p(const double *p1, const double *p2, const double *p3, std::vector<double>&dotout){


    double H[9];
    XCube_Gradient_Kernel_2p(p1,p2,H);
    dotout.resize(3);
    for(int i=0;i<3;++i){
        dotout[i] = 0;
        for(int j=0;j<3;++j){
            dotout[i] += H[i*3+j] * p3[j];
        }
    }
}


double fifthPower_Kernel(const double x){

    return pow(x,5);
}

double fifthPower_Kernel_2p(const double *p1, const double *p2){


    return fifthPower_Kernel(Vec_Dist(p1,p2));

}

void fifthPower_Gradient_Kernel_2p(const double *p1, const double *p2, double *G){


    double len_dist  = Vec_Dist(p1,p2);
    for(int i=0;i<3;++i)G[i] = 5*pow(len_dist,3)*(p1[i]-p2[i]);
    return;

}


double fifthPower_GradientDot_Kernel_2p(const double *p1, const double *p2, const double *p3){

    double G[3];
    fifthPower_Gradient_Kernel_2p(p1,p2,G);
    return Vec_Dot(p3,G);
}

void fifthPower_Hessian_Kernel_2p(const double *p1, const double *p2, double *H){

    double diff[3];
    for(int i=0;i<3;++i)diff[i] = p1[i] - p2[i];
    double len_dist  = Vec_Len(diff);

    if(len_dist<1e-8){
        for(int i=0;i<9;++i)H[i] = 0;
    }else{
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)
            if(i==j)H[i*3+j] = 15 * pow(diff[i],2) * len_dist + 5 * pow(len_dist,3);
            else H[i*3+j] = 15 * diff[i] * diff[j] * len_dist;
    }
    return;
}

void fifthPower_HessianDot_Kernel_2p(const double *p1, const double *p2, const double *p3, std::vector<double>&dotout){

    double H[9];
    fifthPower_Hessian_Kernel_2p(p1,p2,H);
    dotout.resize(3);
    for(int i=0;i<3;++i){
        dotout[i] = 0;
        for(int j=0;j<3;++j){
            dotout[i] += H[i*3+j] * p3[j];
        }
    }
}


arma::mat BuildHrbfMat(std::vector<double>&pts){
    int npt = int(pts.size()/3);
    int key_npt = npt;
    size_t m_dim = npt + 3 * key_npt;
    
    arma::mat M(m_dim,m_dim);
    double *p_pts = pts.data();
    for(int i=0;i<npt;++i){
        for(int j=i;j<npt;++j){
            M(i,j) = M(j,i) = XCube_Kernel_2p(p_pts+i*3, p_pts+j*3);
        }
    }
    double G[3];
    for(int i=0;i<npt;++i){
        for(int j=0;j<key_npt;++j){

            XCube_Gradient_Kernel_2p(p_pts+i*3, p_pts+j*3, G);
            for(int k=0;k<3;++k)M(i,npt+j+k*key_npt) = G[k];
            for(int k=0;k<3;++k)M(npt+j+k*key_npt,i) = G[k];
        }
    }
    double H[9];
    for(int i=0;i<key_npt;++i){
        for(int j=i;j<key_npt;++j){
            XCube_Hessian_Kernel_2p(p_pts+i*3, p_pts+j*3, H);
            for(int k=0;k<3;++k)
                for(int l=0;l<3;++l)
                    M(npt+j+l*key_npt,npt+i+k*key_npt) = M(npt+i+k*key_npt,npt+j+l*key_npt) = -H[k*3+l];
        }
    }
    arma::mat N(m_dim, 4);
    for(int i=0;i<npt;++i){
        N(i,0) = 1;
        for(int j=0;j<3;++j)N(i,j+1) = pts[i*3+j];
    }
    for(int i=0;i<key_npt;++i){
        for(int j=0;j<3;++j)N(npt+i+j*key_npt,j+1) = -1;
    }

    arma::mat bigM(m_dim + 4, m_dim + 4);
    bigM.submat(0,0,m_dim-1,m_dim-1) = M;
    bigM.submat(0,m_dim,m_dim-1, m_dim + 3) = N;
    bigM.submat(m_dim,0,m_dim + 3, m_dim-1) = N.t();

    arma::mat bigMinv = inv(bigM);
    arma::mat Minv = bigMinv.submat(0,0,m_dim-1,m_dim-1);
    return Minv;
}

arma::mat BuildHrbfMat(const std::vector<tetgenmesh::point>&pts, const std::vector<int>&pids, bool use_rbf_base)
{
    int npt = int(pids.size());
    int key_npt = npt;
    size_t m_dim = npt + 3 * key_npt;
    arma::mat bigM(m_dim + 4, m_dim + 4);

    for(int i=0;i<npt;++i){
        for(int j=i;j<npt;++j){
            bigM(i,j) = bigM(j,i) = XCube_Kernel_2p(pts[pids[i]], pts[pids[j]]);
        }
    }
    double G[3];
    for(int i=0;i<npt;++i){
        for(int j=0;j<key_npt;++j){

            XCube_Gradient_Kernel_2p(pts[pids[i]], pts[pids[j]], G);
            for(int k=0;k<3;++k)bigM(i,npt+j+k*key_npt) = G[k];
            for(int k=0;k<3;++k)bigM(npt+j+k*key_npt,i) = G[k];
        }
    }

    double H[9];
    for(int i=0;i<key_npt;++i){
        for(int j=i;j<key_npt;++j){
            XCube_Hessian_Kernel_2p(pts[pids[i]], pts[pids[j]], H);
            for(int k=0;k<3;++k)
                for(int l=0;l<3;++l)
                    bigM(npt+j+l*key_npt,npt+i+k*key_npt) = bigM(npt+i+k*key_npt,npt+j+l*key_npt) = -H[k*3+l];
        }
    }

    arma::mat N(m_dim, 4);
    for(int i=0;i<npt;++i){
        N(i,0) = 1;
        for(int j=0;j<3;++j)N(i,j+1) = pts[pids[i]][j];
    }
    for(int i=0;i<key_npt;++i){
        for(int j=0;j<3;++j)N(npt+i+j*key_npt,j+1) = -1;
    }

    bigM.submat(0,m_dim,m_dim-1, m_dim + 3) = N;
    bigM.submat(m_dim,0,m_dim + 3, m_dim-1) = N.t(); 
    // bool use_rbf_base = true;
    arma::mat Minv(m_dim, m_dim);
    if(use_rbf_base)
    {
        arma::vec invTaget1(m_dim + 4);
        invTaget1[0] = 1.0;
        arma::vec x = arma::solve(bigM, invTaget1, arma::solve_opts::fast); 
        arma::mat x_t = x.t();
        Minv.submat(0,0,0,m_dim-1) = x_t.submat(0,0,0,m_dim-1);
    

        arma::vec invTaget2(m_dim + 4);
        invTaget1[1] = 1.0;
        arma::mat x2 = arma::solve(bigM, invTaget2, arma::solve_opts::fast).t();
        Minv.submat(1,0,1,m_dim-1) = x2.submat(0,0,0,m_dim-1);

        arma::vec invTaget3(m_dim + 4);
        invTaget1[2] = 1.0;
        arma::mat x3 = arma::solve(bigM, invTaget3, arma::solve_opts::fast).t();
        Minv.submat(2,0,2,m_dim-1) = x3.submat(0,0,0,m_dim-1);

        arma::vec invTaget4(m_dim + 4);
        invTaget1[3] = 1.0;
        arma::mat x4 = arma::solve(bigM, invTaget2, arma::solve_opts::fast).t();
        Minv.submat(3,0,3,m_dim-1) = x4.submat(0,0,0,m_dim-1);

    } else {
        arma::mat bigMinv = inv(bigM);
        Minv = bigMinv.submat(0,0,m_dim-1,m_dim-1);
    }

    return Minv;
}


}