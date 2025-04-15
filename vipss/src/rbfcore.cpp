#include "rbfcore.h"
#include "utility.h"
#include "Solver.h"
#include <armadillo>
#include <fstream>
#include <limits>
#include <iomanip>
#include <ctime>
#include <chrono>
#include<algorithm>
#include "readers.h"
#include "kernel.h"


using namespace std;

double sigma = 2.0;
double inv_sigma_squarex2 = 1/(2 * pow(sigma, 2));
double Gaussian_Kernel(const double x_square){

    return exp(-x_square*inv_sigma_squarex2);
}

double Gaussian_Kernel_2p(const double *p1, const double *p2){

    return Gaussian_Kernel(MyUtility::vecSquareDist(p1,p2));

}

double Gaussian_PKernel_Dirichlet_2p(const double *p1, const double *p2){


    double d2 = MyUtility::vecSquareDist(p1,p2);
    return (6*sigma*sigma-d2)*sqrt(Gaussian_Kernel(d2));


}

double Gaussian_PKernel_Bending_2p(const double *p1, const double *p2){


    double d2 = MyUtility::vecSquareDist(p1,p2);
    double d4 = d2*d2;
    double sigma2 = sigma * sigma;
    double sigma4 = sigma2 * sigma2;
    return (60*sigma4-20*sigma2*d2+d4)*sqrt(Gaussian_Kernel(d2));


}


double XCube_Kernel(const double x){

    return pow(x,3);
}

double XCube_Kernel_2p(const double *p1, const double *p2){


    return XCube_Kernel(MyUtility::_VerticesDistance(p1,p2));

}

void XCube_Gradient_Kernel_2p(const double *p1, const double *p2, double *G){


    double len_dist  = MyUtility::_VerticesDistance(p1,p2);
    for(int i=0;i<3;++i)G[i] = 3*len_dist*(p1[i]-p2[i]);
    return;

}

double XCube_GradientDot_Kernel_2p(const double *p1, const double *p2, const double *p3){


    double G[3];
    XCube_Gradient_Kernel_2p(p1,p2,G);
    return MyUtility::dot(p3,G);

}

void XCube_Hessian_Kernel_2p(const double *p1, const double *p2, double *H){


    double diff[3];
    for(int i=0;i<3;++i)diff[i] = p1[i] - p2[i];
    double len_dist  = sqrt(MyUtility::len(diff));

    if(len_dist<1e-12){
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

RBF_Core::RBF_Core(){

  /*  Kernal_Function = Gaussian_Kernel; X
    Kernal_Function_2p = Gaussian_Kernel_2p;
    P_Function_2p = Gaussian_PKernel_Dirichlet_2p;*/

    //Kernal_Function = XCube_Kernel; 
    //Kernal_Function_2p = XCube_Kernel_2p;
    P_Function_2p = Gaussian_PKernel_Dirichlet_2p;

    Kernal_Function = XCube_Kernel;
    Kernal_Function_2p = XCube_Kernel_2p;
    Kernal_Gradient_Function_2p = XCube_Gradient_Kernel_2p;
    Kernal_Hessian_Function_2p = XCube_Hessian_Kernel_2p;

    isHermite = false;

    mp_RBF_INITMETHOD.insert(make_pair(GT_NORMAL,"GT_NORMAL"));
    mp_RBF_INITMETHOD.insert(make_pair(GlobalEigen,"GlobalEigen"));
    mp_RBF_INITMETHOD.insert(make_pair(GlobalEigenWithMST,"GlobalEigenWithMST"));
    mp_RBF_INITMETHOD.insert(make_pair(GlobalEigenWithGT,"GlobalEigenWithGT"));
    mp_RBF_INITMETHOD.insert(make_pair(LocalEigen,"LocalEigen"));
    mp_RBF_INITMETHOD.insert(make_pair(IterativeEigen,"IterativeEigen"));
    mp_RBF_INITMETHOD.insert(make_pair(ClusterEigen,"ClusterEigen"));


    mp_RBF_METHOD.insert(make_pair(Variational,"Variational"));
    mp_RBF_METHOD.insert(make_pair(Variational_P,"Variation_P"));
    mp_RBF_METHOD.insert(make_pair(LS,"LS"));
    mp_RBF_METHOD.insert(make_pair(LSinterp,"LSinterp"));
    mp_RBF_METHOD.insert(make_pair(Interp,"Interp"));
    mp_RBF_METHOD.insert(make_pair(RayleighQuotients,"Rayleigh"));
    mp_RBF_METHOD.insert(make_pair(RayleighQuotients_P,"Rayleigh_P"));
    mp_RBF_METHOD.insert(make_pair(RayleighQuotients_I,"Rayleigh_I"));
    mp_RBF_METHOD.insert(make_pair(Hermite,"Hermite"));
    mp_RBF_METHOD.insert(make_pair(Hermite_UnitNorm,"UnitNorm"));
    mp_RBF_METHOD.insert(make_pair(Hermite_UnitNormal,"UnitNormal"));
    mp_RBF_METHOD.insert(make_pair(Hermite_Tangent_UnitNorm,"T_UnitNorm"));
    mp_RBF_METHOD.insert(make_pair(Hermite_Tangent_UnitNormal,"T_UnitNormal"));

    mp_RBF_Kernal.insert(make_pair(XCube,"TriH"));
    mp_RBF_Kernal.insert(make_pair(ThinSpline,"ThinSpline"));
    mp_RBF_Kernal.insert(make_pair(XLinear,"XLinear"));
    mp_RBF_Kernal.insert(make_pair(Gaussian,"Gaussian"));

}
RBF_Core::RBF_Core(RBF_Kernal kernal){
    isHermite = false;
    Init(kernal);
}

void RBF_Core::Init(RBF_Kernal kernal){

    this->kernal = kernal;
    switch(kernal){
    case Gaussian:
        Kernal_Function = Gaussian_Kernel;
        Kernal_Function_2p = Gaussian_Kernel_2p;
        P_Function_2p = Gaussian_PKernel_Dirichlet_2p;
        break;

    case XCube:
        Kernal_Function = XCube_Kernel;
        Kernal_Function_2p = XCube_Kernel_2p;
        Kernal_Gradient_Function_2p = XCube_Gradient_Kernel_2p;
        Kernal_Hessian_Function_2p = XCube_Hessian_Kernel_2p;
        break;

    default:
        break;

    }

}

void RBF_Core::SetSigma(double x){
    sigma = x;
    inv_sigma_squarex2 = 1/(2 * pow(sigma, 2));
}

double RBF_Core::Dist_Function(const double x, const double y, const double z){

    double p[3] = {x, y, z};
	const double *p_pts = pts.data();
    double G[3];
    for(int i=0;i<npt;++i) kern_(i) = Kernal_Function_2p(p_pts+i*3, p);
    for(int i=0;i<key_npt;++i){
        Kernal_Gradient_Function_2p(p,p_pts+i*3,G);
        //for(int j=0;j<3;++j)kern(npt+i*3+j) = -G[j];
        for(int j=0;j<3;++j)kern_(npt+i+j*key_npt) = G[j];
    }
    double loc_part = dot(kern_,a);
        for(int i=0;i<3;++i) {
            kb_(i+1) = p[i];
        }

    double poly_part = arma::dot(kb_,b);
    double re = loc_part + poly_part;
    return re;

}



double RBF_Core::Dist_Function(const double *p){

    const double *p_pts = pts.data();
    double G[3];
    kern_.zeros();
    for(int i=0;i<npt;++i) kern_(i) = Kernal_Function_2p(p_pts+i*3, p);
    for(int i=0;i<key_npt;++i){
        Kernal_Gradient_Function_2p(p,p_pts+i*3,G);
        //for(int j=0;j<3;++j)kern(npt+i*3+j) = -G[j];
        for(int j=0;j<3;++j)kern_(npt+i+j*key_npt) = G[j];
    }
    double loc_part = dot(kern_,a);
        for(int i=0;i<3;++i) {
            kb_(i+1) = p[i];
        }

    double poly_part = arma::dot(kb_,b);
    // double poly_part = 0;
    double re = loc_part + poly_part;
    // std::cout << " dif re " << re << std::endl;
    return re;
}

double RBF_Core::evaluate_gradient(double x, double y, double z, double &gx, double &gy, double &gz)  
{
    // size_t num_pt = npt;

    double pt[3] = {x, y, z};
    gx = 0;
    gy = 0;
    gz = 0;

    const double *p_pts = pts.data();
    double G[3];
    kern_.zeros();
    for(int i=0;i<npt;++i) kern_(i) = Kernal_Function_2p(p_pts+i*3, pt);
    for(int i=0;i<key_npt;++i){
        Kernal_Gradient_Function_2p(pt,p_pts+i*3,G);
        gx += G[0] * a[i];
        gy += G[1] * a[i];
        gz += G[2] * a[i];
        //for(int j=0;j<3;++j)kern(npt+i*3+j) = -G[j];
        for(int j=0;j<3;++j)kern_(npt+i+j*key_npt) = G[j];
    }

    arma::mat33 Hess;
    for(int i=0;i<key_npt;++i){
        // Kernal_Hessian_Function_2p(pt, p_pts+i*3, H);
        arma::vec3 diff = {pt[0] - pts[3 *i],pt[1] - pts[3 *i + 1], pt[2] - pts[3 *i + 2]};
        double len =  arma::norm(diff);
        if (len > 1e-10) {
           Hess = diff * (diff.t() / len);
            Hess(0, 0) += len;
            Hess(1, 1) += len;
            Hess(2, 2) += len;
            Hess *= 3;
        }
        for(int j = 0; j < 3; ++j)
        {
            gx += Hess.col(j)[0] * a[npt + j * key_npt + i];
            gy += Hess.col(j)[1] * a[npt + j * key_npt + i];
            gz += Hess.col(j)[2] * a[npt + j * key_npt + i];
        }
    }

    gx += b[1];
    gy += b[2];
    gz += b[3];

    double loc_part = dot(kern_,a);
    for(int i=0;i<3;++i) {
            kb_(i+1) = pt[i];
        }
    double poly_part = arma::dot(kb_,b);

    double re = loc_part + poly_part;
    return re;
}

void RBF_Core::EvaluateHessian(double x, double y, double z, arma::mat& hessian) 
{
    // size_t num_pt = npt;
    double pt[3] = {x, y, z};
    
    arma::mat33 Hess;
    for(int i=0;i<key_npt;++i){
        // Kernal_Hessian_Function_2p(pt, p_pts+i*3, H);
        arma::vec3 diff = {pt[0] - pts[3 *i],pt[1] - pts[3 *i + 1], pt[2] - pts[3 *i + 2]};
        double len =  sqrt(arma::dot(diff, diff));
        if (len > 1e-10) {
           Hess = diff * (diff.t() / len);
            Hess(0, 0) += len;
            Hess(1, 1) += len;
            Hess(2, 2) += len;
            Hess *= 3;
            hessian += a[i] * Hess;
            // std::cout << " hessian 0 " << hessian << std::endl;
        }
        double dx = diff(0);
        double dy = diff(1);
        double dz = diff(2);
        double l3 = len * len * len;
        if (len > 1e-10) {
            // cal H(Dx)
            Hess(0, 0) = 3 * dx / len -  dx * dx * dx / l3;
            Hess(0, 1) =  dy / len -  dx * dx * dy / l3;
            Hess(0, 2) =  dz / len -  dx * dx * dz / l3;

            Hess(1, 0) = dy / len -  dx * dx * dy / l3;
            Hess(1, 1) = dx / len -  dx * dy * dy / l3;
            Hess(1, 2) =  - dx * dz * dy / l3;

            Hess(2, 0) = dz / len - dx * dx * dz / l3;
            Hess(2, 1) =  -  dx * dz * dy / l3;
            Hess(2, 2) =  dx / len - dx * dz * dz / l3;
            Hess = a[npt  + i] * 3.0 * Hess; 
            hessian += Hess;
            // std::cout << " hessian 1 " << Hess << std::endl;

            // cal H(Dy)
            Hess(0, 0) = dy / len - dx * dx * dy / l3;
            Hess(0, 1) = dx / len - dx * dy * dy / l3;
            Hess(0, 2) =  - dx * dy * dz / l3;

            Hess(1, 0) = dx / len - dx * dy * dy / l3;
            Hess(1, 1) = 3* dy / len - dy * dy * dy / l3;
            Hess(1, 2) = dz / len - dy * dy * dz / l3;

            Hess(2, 0) = - dx * dy * dz / l3;
            Hess(2, 1) = dz / len - dy * dy * dz / l3;
            Hess(2, 2) = dy / len - dy * dz * dz / l3;
            hessian += a[npt + key_npt + i] * Hess * 3.0;

            // std::cout << " hessian 2 " << Hess << std::endl;

            // cal H(Dz)
            Hess(0, 0) = dz / len - dx * dx * dz / l3;
            Hess(0, 1) = - dx * dy * dz / l3;
            Hess(0, 2) = dx / len -  dx * dz * dz / l3;

            Hess(1, 0) =  -  dx * dy * dz / l3;
            Hess(1, 1) =  dz / len -  dy * dy * dz / l3;
            Hess(1, 2) =  dy / len -  dy * dz * dz / l3;

            Hess(2, 0) = dx / len - dx * dz * dz / l3;
            Hess(2, 1) = dy / len - dy * dz * dz / l3;
            Hess(2, 2) = 3 * dz / len - dz * dz * dz / l3;
            hessian += a[npt + key_npt * 2 + i] * 3.0 * Hess;
            // std::cout << " hessian 3 " << Hess << std::endl;
        }
    }

    return ;
}


int RBF_Core::DistFuncCallNum = 0;
double RBF_Core::DistFuncCallTime = 0.0;
static RBF_Core * s_hrbf;

typedef std::chrono::high_resolution_clock Clock;

double RBF_Core::Dist_Function(const R3Pt &in_pt){
    DistFuncCallNum ++;
   auto t0 = Clock::now();
   double dist = s_hrbf->Dist_Function(&(in_pt[0]));
   auto t1 = Clock::now();
   double call_time = std::chrono::nanoseconds(t1 - t0).count()/1e9;
   DistFuncCallTime += call_time;
   return dist;
}

//FT RBF_Core::Dist_Function(const Point_3 in_pt){

//    return s_hrbf->Dist_Function(&(in_pt.x()));
//}

void RBF_Core::SetThis(){

    s_hrbf = this;
}

void RBF_Core::Write_Surface(std::string fname){

    //writeObjFile(fname,finalMesh_v,finalMesh_fv);

    writePLYFile_VF(fname,finalMesh_v,finalMesh_fv);
}

/**********************************************************/


void RBF_Core::Record(RBF_METHOD method, RBF_Kernal kernal, Solution_Struct &rsol, double time){

    npoints.push_back(npt);

    record_initmethod.push_back(mp_RBF_INITMETHOD[curInitMethod]);
    record_method.push_back(mp_RBF_METHOD[method]);
    record_kernal.push_back(mp_RBF_Kernal[kernal]);
    record_initenergy.push_back(rsol.init_energy);
    record_energy.push_back(rsol.energy);
    record_time.push_back(time);


    setup_timev.push_back(setup_time);
    init_timev.push_back(init_time);
    solve_timev.push_back(solve_time);
    callfunc_timev.push_back(callfunc_time);
    invM_timev.push_back(invM_time);
    setK_timev.push_back(setK_time);

}

void RBF_Core::Record(){

    //cout<<"record"<<endl;
    npoints.push_back(npt);

    record_initmethod.push_back(mp_RBF_INITMETHOD[curInitMethod]);
    record_method.push_back(mp_RBF_METHOD[curMethod]);
    //record_kernal.push_back(mp_RBF_Kernal[kernal]);
    record_initenergy.push_back(sol.init_energy);
    record_energy.push_back(sol.energy);
    //cout<<"record"<<endl;


//    setup_timev.push_back(setup_time);
//    init_timev.push_back(init_time);
//    solve_timev.push_back(solve_time);
//    callfunc_timev.push_back(callfunc_time);
//    invM_timev.push_back(invM_time);
//    setK_timev.push_back(setK_time);
   // cout<<"record end"<<endl;
}

void RBF_Core::AddPartition(std::string pname){

    record_partition.push_back((int)record_method.size());
    record_partition_name.push_back(pname);
}


void RBF_Core::Print_Record(){

    cout<<"Method\t\t Kernal\t\t Energy\t\t Time"<<endl;
    cout<<std::setprecision(8)<<endl;
    if(record_partition.size()==0){
        for(int i=0;i<record_method.size();++i){
            cout<<record_method[i]<<"\t\t"<<record_kernal[i]<<"\t\t"<<record_energy[i]<<"\t\t"<<record_time[i]<<endl;
        }
        for(int i=0;i<setup_timev.size();++i){
            cout<<setup_timev[i]<<"\t\t"<<init_timev[i]<<"\t\t"<<solve_timev[i]<<"\t\t"<<callfunc_timev[i]<<"\t\t"<<invM_timev[i]<<"\t\t"<<setK_timev[i]<<endl;
        }
    }else{
        for(int j=0;j<record_partition.size();++j){
            cout<<record_partition_name[j]<<endl;
            for(int i=j==0?0:record_partition[j-1];i<record_partition[j];++i){
                cout<<record_method[i]<<"\t\t"<<record_kernal[i]<<"\t\t"<<record_energy[i]<<"\t\t"<<record_time[i]<<endl;
            }
            for(int i=j==0?0:record_partition[j-1];i<record_partition[j];++i){
                cout<<setup_timev[i]<<"\t\t"<<init_timev[i]<<"\t\t"<<solve_timev[i]<<"\t\t"<<callfunc_timev[i]<<endl;
            }
        }
    }

}


void RBF_Core::Print_TimerRecord(std::string fname){

    ofstream fout(fname);
    fout<<setprecision(5);
    if(!fout.fail()){
        for(int i=0;i<setup_timev.size();++i){
            fout<<npoints[i]<<'\t'<<setup_timev[i]<<"\t"<<init_timev[i]<<"\t"<<solve_timev[i]<<"\t"<<callfunc_timev[i]<<"\t"<<invM_timev[i]<<"\t"<<setK_timev[i]<<endl;
        }
    }
    fout.close();

}

void RBF_Core::Print_TimerRecord_Single(std::string fname){

    ofstream fout(fname);
    fout<<setprecision(5);
    if(!fout.fail()){
        fout<<"number of points: "<<npt<<endl
           <<"setup_time (Compute H): "<<setup_time<<" s"<<endl
          <<"init_time (Optimize g/Eigen): "<<init_time<<" s"<<endl
         <<"solve_time (Optimize g/LBFGS): "<<solve_time<<" s"<<endl
        <<"surfacing_time: "<<surf_time<<" s"<<endl;
    }
    fout.close();
}

void RBF_Core::Clear_TimerRecord(){
    npoints.clear();
    setup_timev.clear();
    init_timev.clear();
    solve_timev.clear();
    callfunc_timev.clear();
    invM_timev.clear();
    setK_timev.clear();
}

void RBF_Core::EstimateNormals()
{
    double delt = 0.00000001;
    out_normals_.resize(npt*3);
    for(size_t i = 0; i < npt; ++i)
    {
        double x = pts[i * 3];
        double y = pts[i * 3 + 1];
        double z = pts[i * 3 + 2];

        double curPxN[3] = {x - delt, y, z};
        double curPxO[3] = {x + delt, y, z};
        double dx = (this->Dist_Function(curPxO) - this->Dist_Function(curPxN))/ (2 * delt);

        double curPyN[3] = {x, y - delt, z};
        double curPyO[3] = {x, y + delt, z};
        double dy = (this->Dist_Function(curPyO) -this->Dist_Function(curPyN))/ (2 * delt);

        double curPzN[3] = {x, y, z - delt};
        double curPzO[3] = {x, y, z + delt};
        double dz = (this->Dist_Function(curPzO) -this->Dist_Function(curPzN))/ (2 * delt);
        double len = std::max(sqrt(dx * dx + dy * dy + dz * dz), 1e-8);

        out_normals_[3*i] = dx / len;
        out_normals_[3*i + 1] = dy / len ;
        out_normals_[3*i + 2] = dz / len ;
    }
}

std::vector<double> RBF_Core::EstimateNormals(const std::vector<double>& pts)
{
    double delt = 0.00000001;
    // out_normals_.resize(npt*3);
    std::vector<double> out_normals(pts.size());
    for(size_t i = 0; i < pts.size()/3; ++i)
    {
        double x = pts[i * 3];
        double y = pts[i * 3 + 1];
        double z = pts[i * 3 + 2];
        double curPxN[3] = {x - delt, y, z};
        double curPxO[3] = {x + delt, y, z};
        double dx = (this->Dist_Function(curPxO) - this->Dist_Function(curPxN))/ (2 * delt);

        double curPyN[3] = {x, y - delt, z};
        double curPyO[3] = {x, y + delt, z};
        double dy = (this->Dist_Function(curPyO) -this->Dist_Function(curPyN))/ (2 * delt);

        double curPzN[3] = {x, y, z - delt};
        double curPzO[3] = {x, y, z + delt};
        double dz = (this->Dist_Function(curPzO) -this->Dist_Function(curPzN))/ (2 * delt);
        double len = std::max(sqrt(dx * dx + dy * dy + dz * dz), 1e-8);

        // out_normals.push_back(dx / len);
        // out_normals.push_back(dy / len);
        // out_normals.push_back(dz / len);
        out_normals[3*i] = dx / len;
        out_normals[3*i + 1] = dy / len;
        out_normals[3*i + 2] = dz / len;
    }
    return out_normals;
}