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
// #include "ImplicitedSurfacing.h"
typedef std::chrono::high_resolution_clock Clock;

using namespace std;

void RBF_Core::BuildK(RBF_Paras para){
    isuse_sparse = para.isusesparse;
    sparse_para = para.sparse_para;
    Hermite_weight_smoothness = para.Hermite_weight_smoothness;
    Hermite_designcurve_weight = para.Hermite_designcurve_weight;
//    handcraft_sigma = para.handcraft_sigma;
//    wDir = para.wDir;
//    wOrt = para.wOrt;
//    wFlip = para.wFlip;
    curMethod = para.Method;

    Set_Actual_Hermite_LSCoef( para.Hermite_ls_weight );
    Set_Actual_User_LSCoef(  para.user_lamnbda  );
    isNewApprox = true;
    isnewformula = true;

    auto t1 = Clock::now();

    // cout << " start Set_Hermite_PredictNormal " <<  endl;
    switch(curMethod){

    case Hermite_UnitNormal:
        Set_Hermite_PredictNormal(pts);
        break;
    }
    // cout << " finish Set_Hermite_PredictNormal " <<  endl;
    auto t2 = Clock::now();
    if(open_debug_log)
    cout << "Build Time: " << (setup_time = std::chrono::nanoseconds(t2 - t1).count()/1e9) << endl<< endl;
}

void RBF_Core::BuildK(double lambda){


    // Set_Actual_Hermite_LSCoef( para.Hermite_ls_weight );
    Set_Actual_User_LSCoef(  lambda  );
    isNewApprox = true;
    isnewformula = true;
    Set_Hermite_PredictNormal(pts);
}

void RBF_Core::InitNormal(RBF_Paras para){

    auto t1 = Clock::now();
    curInitMethod = para.InitMethod;
    // cout<<"Init Method: "<<mp_RBF_INITMETHOD[curInitMethod]<<endl;
    switch(curInitMethod){

    case Lamnbda_Search:
        Lamnbda_Search_GlobalEigen();
        break;

    }
    auto t2 = Clock::now();
    if(open_debug_log)
    cout << "Init Time: " << (init_time = std::chrono::nanoseconds(t2 - t1).count()/1e9) << endl<< endl;

    mp_RBF_InitNormal[curMethod==HandCraft?0:1][curInitMethod] = initnormals;
}

void RBF_Core::InitNormal(){

    auto t1 = Clock::now();
    Lamnbda_Search_GlobalEigen();
    auto t2 = Clock::now();
    if(open_debug_log)
    cout << "Init Time: " << (init_time = std::chrono::nanoseconds(t2 - t1).count()/1e9) << endl<< endl;

    mp_RBF_InitNormal[curMethod==HandCraft?0:1][curInitMethod] = initnormals;
}

void RBF_Core::OptNormal(int method){

    if(open_debug_log)
    cout<<"OptNormal"<<endl;
    auto t1 = Clock::now();
  
    switch(curMethod){

    case Hermite_UnitNormal:
        Opt_Hermite_PredictNormal_UnitNormal();
        break;

    }
    auto t2 = Clock::now();
    if(open_debug_log)
    cout << "Opt Time: " << (solve_time = std::chrono::nanoseconds(t2 - t1).count()/1e9) << endl<< endl;
    if(method==0)mp_RBF_OptNormal[curMethod==HandCraft?0:1][curInitMethod] = newnormals;
}


void RBF_Core::Surfacing(int method, int n_voxels_1d){
    // n_evacalls = 0;
    // Surfacer sf;
    // SetThis();
    // std::cout << "n_voxels_1d " << n_voxels_1d << std::endl;
    // surf_time = sf.Surfacing_Implicit(pts, n_voxels_1d, false, RBF_Core::Dist_Function);
    // // surf_time = sf.Surfacing_Implicit(pts, n_voxels_1d, false, this->Dist_Function);

    // sf.WriteSurface(finalMesh_v,finalMesh_fv);
    // if(open_debug_log)
    // cout<<"n_evacalls: "<<n_evacalls<<" ave: "<<surf_time/n_evacalls<<endl;
}


int RBF_Core::InjectData(std::vector<double> &pts, RBF_Paras para)
{
    std::vector<int> labels;
    std::vector<double> normals,tangents;
    std::vector<uint> edges;

    InjectData(pts,labels,normals,tangents,edges,para);
    // cout << " finish InjectData " << endl;
    return 0;
}

int RBF_Core::InjectData(std::vector<double> &pts, std::vector<int> &labels, std::vector<double> &normals, std::vector<double> &tangents, std::vector<uint> &edges, RBF_Paras para){

    isuse_sparse = para.isusesparse;
    sparse_para = para.sparse_para;
    //isuse_sparse = false;
    this->pts = pts;
    this->labels = labels;
    this->normals = normals;
    this->tangents = tangents;
    this->edges = edges;
    npt = this->pts.size()/3;
    curMethod = para.Method;
    curInitMethod = para.InitMethod;

    polyDeg = para.polyDeg;
    User_Lamnbda = para.user_lamnbda;
    rangevalue = para.rangevalue;
    maxvalue = 10000;
    if(open_debug_log)
    {
        cout<<"number of points: "<<pts.size()/3<<endl;
        cout<<"normals: "<<this->normals.size()<<endl;
    }
    
    sol.Statue = 1;
    Init(para.Kernal);
    
    // cout << " start to set sigma " << endl;
    SetSigma(para.sigma);
    // cout << " finish set sigma " << endl;
    SetThis();
    // cout << " finish set this " << endl;
	return 1;
}

int RBF_Core::ThreeStep(std::vector<double>&pts, std::vector<int>&labels, std::vector<double>&normals, std::vector<double>&tangents,  std::vector<uint>&edges, RBF_Paras para){

    InjectData(pts, labels, normals, tangents, edges,  para);
    BuildK(para);
    InitNormal(para);
    OptNormal(0);
	
	return 1;
}


int RBF_Core::AllStep(std::vector<double> &pts, std::vector<int> &labels, std::vector<double> &normals, std::vector<double> &tangents, std::vector<uint> &edges, RBF_Paras para){

    InjectData(pts, labels, normals, tangents, edges,  para);
    BuildK(para);
    InitNormal(para);
    OptNormal(0);
    Surfacing(0,100);
	return 1;

}

void RBF_Core::BatchInitEnergyTest(std::vector<double> &pts, std::vector<int> &labels, std::vector<double> &normals, std::vector<double> &tangents, std::vector<uint> &edges, RBF_Paras para){

    InjectData(pts, labels, normals, tangents, edges,  para);
    BuildK(para);
    para.ClusterVisualMethod = 0;//RBF_Init_EMPTY
    for(int i=0;i<RBF_Init_EMPTY;++i){
        para.InitMethod = RBF_InitMethod(i);
        InitNormal(para);
        OptNormal(0);
        Record();
    }
    Print_Record_Init();
}


std::vector<double>* RBF_Core::ExportPts(){

    return &pts;



}

std::vector<double>* RBF_Core::ExportPtsNormal(int normal_type){

    if(normal_type==0)return &normals;
    else if(normal_type==1)return &initnormals;
    else if(normal_type==2)return &initnormals_uninorm;
    else if(normal_type==3)return &newnormals;

	return NULL;
}



std::vector<double>* RBF_Core::ExportInitNormal(int kmethod, RBF_InitMethod init_type){

    if(mp_RBF_InitNormal[kmethod].find(init_type)!=mp_RBF_InitNormal[kmethod].end())return &(mp_RBF_InitNormal[kmethod][init_type]);
    else return NULL;
}

std::vector<double>* RBF_Core::ExportOptNormal(int kmethod, RBF_InitMethod init_type){

    if(mp_RBF_OptNormal[kmethod].find(init_type)!=mp_RBF_OptNormal[kmethod].end())return &(mp_RBF_OptNormal[kmethod][init_type]);
    else return NULL;
}



void RBF_Core::Print_Record_Init(){

    cout<<"InitMethod"<<std::string(30-std::string("InitMethod").size(),' ')<<"InitEn\t\t FinalEn"<<endl;
    cout<<std::setprecision(8);
    {
        for(int i=0;i<record_initmethod.size();++i){
            cout<<record_initmethod[i]<<std::string(30-record_initmethod[i].size(),' ')<<record_initenergy[i]<<"\t\t"<<record_energy[i]<<endl;
        }
    }

}
