#pragma once 
#include <math.h>
#include <vector>
#include <armadillo>
#include "tetgen.h"

namespace VIPSSKernel
{
double Gaussian_Kernel(const double x_square);
inline double Vec_Squared_Dist(const double *p1, const double *p2);

inline double Vec_Dot(const double *p1, const double *p2);
inline double Vec_Len(const double *p1);
double Gaussian_Kernel_2p(const double *p1, const double *p2);

double Gaussian_PKernel_Dirichlet_2p(const double *p1, const double *p2);

double Gaussian_PKernel_Bending_2p(const double *p1, const double *p2);


double XCube_Kernel(const double x);

double XCube_Kernel_2p(const double *p1, const double *p2);

void XCube_Gradient_Kernel_2p(const double *p1, const double *p2, double *G);



double XCube_GradientDot_Kernel_2p(const double *p1, const double *p2, const double *p3);

void XCube_Hessian_Kernel_2p(const double *p1, const double *p2, double *H);

void XCube_HessianDot_Kernel_2p(const double *p1, const double *p2, const double *p3, std::vector<double>&dotout);



arma::mat BuildHrbfMat(std::vector<double>&pts);

arma::mat BuildHrbfMat(const std::vector<tetgenmesh::point>&pts, const std::vector<int>&pids,  bool use_rbf_base = false);

arma::mat BuildHrbfMatHalf(const std::vector<tetgenmesh::point>&pts, const std::vector<int>&pids,  bool use_rbf_base = false);

// bool use_rbf_base = false;

}