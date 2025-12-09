#pragma once
#include <iostream>
#include "ImplicitFunction.h"
#include <Eigen/Core>

// template<typename Scalar> 
// class CurvatureData {
//     public:
//         using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
//         CurvatureData(){};
//         CurvatureData(Scalar k1, Scalar k2, Vec3 t1, Vec3 t2) :
//         k1_(k1), k2_(k2), t1_(t1), t2_(t2){};
//     public:
//         Scalar f_val_;
//         Vec3 f_gradient_;
//         Scalar k1_;
//         Scalar k2_;
//         Vec3 t1_; 
//         Vec3 t2_;
//         Scalar e1_;
//         Scalar e2_;
//         Vec3 e1_d1_; 
//         Vec3 e2_d2_;
//         Scalar e1_prime_ = 0;
//         Scalar e2_prime_ = 0;
// };


template<typename Scalar>
class ImplicitFunctionRidge : public ImplicitFunction<Scalar> {
public:
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
    using VecX = Eigen::Matrix<Scalar, -1, 1>;
    using Mat33 = Eigen::Matrix <Scalar, 3,3>; 
    using Tensor3D = std::array<Mat33, 3>;
    using Tensor4D = std::array<std::array<Mat33, 3>, 3>;
    
    // constant 0 function
    ImplicitFunctionRidge() {}

    ImplicitFunctionRidge(const std::vector<Vec3> &control_points)
            : control_points_(control_points) {}
    
    ImplicitFunctionRidge(const std::vector<Vec3> &control_points, Scalar offset)
            : control_points_(control_points), offset_value_(offset) {}

    Scalar kernel_function_gaussian(const Vec3 &p1, const Vec3 &p2) const
    {
        Scalar sq_dist = PtSQDist(p1, p2);
        Scalar dist = std::exp(- sq_dist / (2 * sigma_val_ * sigma_val_));
        return dist;
    }

    Scalar PtSQDist(const Vec3 &p1, const Vec3 &p2) const
    {
        Vec3 v12 = p2 - p1;
        Scalar Sq_dist = v12[0] * v12[0] + v12[1] * v12[1] + v12[2] * v12[2];
        return Sq_dist;
    }

    Scalar evaluate(Scalar x, Scalar y, Scalar z) const override {
        // std::cout << "start implicit HRBF evaluate val " << std::endl;
        size_t num_pt = control_points_.size();
        int dim = 3;
        Vec3 p(x, y, z);
        VecX kern(num_pt);
        VecX weights(num_pt);
        #pragma omp parallel for 
        for (int i = 0; i < num_pt; ++i) {
            weights(i) = kernel_function_gaussian(p, control_points_[i]);
            // kern(i) = weights(i) * sqrt(PtSQDist(p, control_points_[i]));
            kern(i) = weights(i) * PtSQDist(p, control_points_[i]);
        }
        Scalar dist_sum = kern.sum();
        Scalar weigts_sum = weights.sum();
        Scalar dist = - dist_sum / weigts_sum;
        return dist;
    }

    // Scalar evaluate(Scalar x, Scalar y, Scalar z) const override {
    //     // std::cout << "start implicit HRBF evaluate val " << std::endl;
    //     size_t num_pt = control_points_.size();
    //     int dim = 3;
    //     Vec3 plane_n = {0.1, 0.1, 1.0 };
        
    //     Scalar diff = x * x + y * y + z * z - 1 ;
    //     // Scalar diff =  z *z  ;
    //     // Scalar proj = 0.1 * x + 0.2 *y + z;
    //     Scalar sigma = 0.1;
    //     Scalar dist = std::exp(- diff * diff / (2 * sigma * sigma));
    //     // Scalar dist = - proj * proj;
    //     return  dist ;
    // }

    Scalar evaluate_gradient(Scalar x, Scalar y, Scalar z, Scalar &gx, Scalar &gy, Scalar &gz) const override {
        size_t num_pt = control_points_.size();
        int dim = 3;
        Scalar dist_px = evaluate(x + h, y, z);
        Scalar dist_nx = evaluate(x - h, y, z);
        gx = (dist_px - dist_nx) / (2 * h);

        Scalar dist_py = evaluate(x, y + h, z);
        Scalar dist_ny = evaluate(x, y - h, z);
        gy = (dist_py - dist_ny) / (2 * h);

        Scalar dist_pz = evaluate(x, y, z + h);
        Scalar dist_nz = evaluate(x, y, z - h);
        gz = (dist_pz - dist_nz) / (2 * h);

        // std::cout << "implicit HRBF evaluate val : " << loc_part + poly_part - offset_value_ << std::endl;
        return evaluate(x, y, z);
    }


    void EvaluateHessian(Scalar x, Scalar y, Scalar z, Mat33& hessian) const
    {
        size_t num_pt = control_points_.size();
        Vec3 p(x, y, z);
        hessian = Mat33::Zero();
        Scalar dx1, dy1, dz1;
        Scalar dx2, dy2, dz2;
        // d/dx of gradient components
        evaluate_gradient(x + h, y, z, dx1, dy1, dz1);
        evaluate_gradient(x - h, y, z, dx2, dy2, dz2);
        hessian(0,0) = (dx1 - dx2) / (2*h);   // f_xx
        hessian(1,0) = (dy1 - dy2) / (2*h);   // f_yx
        hessian(2,0) = (dz1 - dz2) / (2*h);   // f_zx
        // d/dy of gradient components
        evaluate_gradient(x, y + h, z, dx1, dy1, dz1);
        evaluate_gradient(x, y - h, z, dx2, dy2, dz2);
        hessian(0,1) = (dx1 - dx2) / (2*h);   // f_xy
        hessian(1,1) = (dy1 - dy2) / (2*h);   // f_yy
        hessian(2,1) = (dz1 - dz2) / (2*h);   // f_zy
        // d/dz of gradient components
        evaluate_gradient(x, y, z + h, dx1, dy1, dz1);
        evaluate_gradient(x, y, z - h, dx2, dy2, dz2);
        hessian(0,2) = (dx1 - dx2) / (2*h);   // f_xz
        hessian(1,2) = (dy1 - dy2) / (2*h);   // f_yz
        hessian(2,2) = (dz1 - dz2) / (2*h);   // f_zz

        return ;
    }

    // Compute principal curvatures and directions using the method from Monga et al.
    CurvatureData<Scalar> ComputePrincipalCurvaturesMonga(const Vec3& gradient, const Mat33& hessian) const
    {
        const auto& g = gradient;
        const auto& H = hessian;
        Scalar g_norm = g.norm();

        // Compute rotation matrix P to align g with the first basis vector
        Scalar g1 = g(0), g2 = g(1), g3 = g(2);
        Scalar gamma = std::sqrt(g1 * g1 + g2 * g2);
        Scalar delta = g_norm;
        // arma::mat P(3, 3);
        Vec3 h = {g2 / gamma, -g1 / gamma, 0};
        Vec3 f = {g1 * g3 / (gamma * delta), g2 * g3 / (gamma * delta), -gamma / delta};
        // Compute terms for principal curvatures
        Scalar hHh = (h.transpose() * H * h);
        Scalar fHf = (f.transpose() * H * f);
        Scalar hHf = (h.transpose() * H * f);

        // Compute principal curvatures
        Scalar discriminant = std::sqrt((hHh - fHf) * (hHh - fHf) + 4 * hHf * hHf);
        Scalar k1 = (hHh + fHf + discriminant) / (2 * g_norm); // Larger curvature (in magnitude)
        Scalar k2 = (hHh + fHf - discriminant) / (2 * g_norm); // Smaller curvature

        Scalar factor1 = (g_norm * k1 - hHh) / hHf;
        Scalar factor2 = (g_norm * k2 - hHh) / hHf;
        Vec3 t1 = h + f * factor1;
        Vec3 t2 = h + f * factor2;
        t1 = t1.normalized();
        t2 = t2.normalized();
        // t2 = arma::normalise(arma::cross(t1, g));
        CurvatureData<Scalar> p_curvature(k1, k2, t1, t2);
        return p_curvature;
    }

    void EvaluateCurvatureData(double x, double y, double z, CurvatureData<Scalar>& cur_data) const
    {
        // std::cout << " start EvaluateCurvatureData " << std::endl;

        // std::cout << "input pt : " << x << " " <<y << " " << z << std::endl;
        Vec3 gradient;
        // std::cout << " start to evaluate gradient " << std::endl;
        double f_val_ = evaluate_gradient(x, y, z, gradient[0], gradient[1], gradient[2]);
        // std::cout << " cur_data.f_val_ " << cur_data.f_val_ << std::endl;
        // std::cout << " Hrbf : finish evaluate_gradient ..." << std::endl;
        Mat33 hessian = Mat33::Zero();
        EvaluateHessian(x, y, z, hessian);
          Eigen::SelfAdjointEigenSolver<Mat33> solver(hessian);
        Vec3 eigenValues = solver.eigenvalues();
        Mat33 eigenVectors = solver.eigenvectors();
        cur_data.k1_ = eigenValues[0];
        cur_data.k2_ = eigenValues[2];
        cur_data.t1_ = eigenVectors.col(0);
        cur_data.t2_ = eigenVectors.col(2);
        cur_data.e1_  = gradient.dot(cur_data.t1_);
        cur_data.e2_  = gradient.dot(cur_data.t2_);
        cur_data.f_gradient_ = gradient;
        cur_data.f_val_ = f_val_;
        // std::cout << " Hrbf : finish ComputeCurvatureDerivative2 ..." << std::endl;
    } 


    // Scalar ComputeCurvatureDerivative(const std::array<Scalar,3>& pt, 
    //                                     const Vec3& gradient, 
    //                                     const Mat33& Hessian,
    //                                     const Tensor3D& third_derivs,  
    //                                     const Vec3& t1) const
    // {
    //     Scalar g_norm = gradient.norm();
    //     Vec3 normalized_t = t1.normalized(); 
    //     Vec3 third_deriv_terms;
    //     for (int i = 0; i < 3; ++i) {
    //         third_deriv_terms(i) = normalized_t.transpose()
    //         * third_derivs[i] * normalized_t; // t1^T H_{x,y,z} t1
    //     }
    //     Scalar tHt = normalized_t.transpose() * Hessian * normalized_t;
    //     Scalar tHg = normalized_t.transpose() * Hessian * gradient;

    //     // Compute derivative: (||g||^2 * t1^T [t1^T H_x t1, t1^T H_y t1, t1^T H_z t1] - (t1^T H t1)(t1^T H g)) / ||g||^3
    //     Scalar numerator = g_norm * g_norm * (normalized_t.transpose().dot(third_deriv_terms)) - tHt * tHg;
    //     Scalar denominator = g_norm * g_norm * g_norm;
    //     return numerator / denominator;
    // }

    
    // Scalar ComputeCurvatureDerivative2(const std::array<Scalar,3>& pt, 
    //                                     const Vec3& t1, Vec3& e_gradient) const
    // {
    //     // Scalar h = 1e-8;
    //     Scalar h = 1e-6;
    //     std::vector<std::array<Scalar,3>> pts(6);
    //     pts[0] = {pt[0]+ h, pt[1], pt[2]};
    //     pts[1] = {pt[0]- h, pt[1], pt[2]};
    //     pts[2] = {pt[0], pt[1] +h, pt[2]};
    //     pts[3] = {pt[0], pt[1] -h, pt[2]};
    //     pts[4] = {pt[0], pt[1], pt[2] +h};
    //     pts[5] = {pt[0], pt[1], pt[2] -h};
    //     std::vector<Scalar> e_vals(6);
    //     #pragma omp parallel for
    //     for(int i = 0; i < 6; ++i)
    //     {
    //         Mat33 Hessian_xp = Mat33::Zero(); 
    //         std::vector<Mat33> third_derivs_xp;
    //         const std::array<Scalar,3>& pt_xp = pts[i];
    //         Vec3 g_xp = Vec3::Zero();
    //         evaluate_gradient(pt_xp[0], pt_xp[1], pt_xp[2], g_xp[0], g_xp[1], g_xp[2]);
    //         EvaluateHessian(pt_xp[0], pt_xp[1], pt_xp[2], Hessian_xp);
    //         ComputeThirdDerivatives(pt_xp[0], pt_xp[1], pt_xp[2], third_derivs_xp);
    //         e_vals[i] = ComputeCurvatureDerivative(pt_xp, g_xp, Hessian_xp, third_derivs_xp, t1); 
    //     }
    //     e_gradient = Vec3::Zero();
    //     for(int i =0; i < 3; ++i)
    //     {
    //         e_gradient[i] = (e_vals[2*i] - e_vals[2*i +1]) /( 2 * h);
    //     } 
    //     return 0;
    // }
public:
    // static Hermite_RBF<Scalar>* hrbf_ptr_; 
    // static CurvatureData<Scalar> EvaluateCurDataStatic(double px, double py, double pz)
    // {
    //     CurvatureData<Scalar> cur_data;
    //     hrbf_ptr_->EvaluateCurvatureData(px,  py,  pz,  cur_data);
    //     return cur_data;
    // }
public:
    Scalar offset_value_ = 0;
    Scalar h = 1e-6;
    std::vector<Vec3> control_points_;
    // // using KernelFunction = Scalar(*)(const Vec3&, const Vec3 &);
    // // using KernelGradient = Vec3(*)(const Vec3&, const Vec3 &);
    // // using KernelHessian = Eigen::Matrix<Scalar, 3, 3>(*)(const Vec3&, const Vec3 &);
    double sigma_val_ = 0.2;
    static Tensor3D ConstTensor3D;  //= {Mat33::Zero(), Mat33::Zero(), Mat33::Zero()};
    static Tensor4D ConstTensor4D;  //= {ConstTensor3D, ConstTensor3D, ConstTensor3D};

};

// template<typename Scalar> ImplicitFunctionRidge<Scalar>::sigma_val_ =  Scalar (0.01);


// template<typename Scalar>
// Hermite_RBF<Scalar>* Hermite_RBF<Scalar>::hrbf_ptr_ = nullptr; 

// template<typename Scalar> Hermite_RBF<Scalar>::KernelFunction Hermite_RBF<Scalar>::kernel_function_ 
//             = &Hermite_RBF<Scalar>::kernel_function_r5;

// template<typename Scalar> Hermite_RBF<Scalar>::KernelGradient Hermite_RBF<Scalar>::kernel_gradient_ 
//             = &Hermite_RBF<Scalar>::kernel_gradient_r5;

// template<typename Scalar> Hermite_RBF<Scalar>::KernelHessian Hermite_RBF<Scalar>::kernel_hessian_ 
//             = &Hermite_RBF<Scalar>::kernel_Hessian_r5;

// template<typename Scalar> Hermite_RBF<Scalar>::KernelHessian Hermite_RBF<Scalar>::kernel_dx_hessian_ 
//             = &Hermite_RBF<Scalar>::kernel_dx_Hessian_r5;

// template<typename Scalar> Hermite_RBF<Scalar>::KernelHessian Hermite_RBF<Scalar>::kernel_dy_hessian_ 
//             = &Hermite_RBF<Scalar>::kernel_dy_Hessian_r5;

// template<typename Scalar> Hermite_RBF<Scalar>::KernelHessian Hermite_RBF<Scalar>::kernel_dz_hessian_ 
//             = &Hermite_RBF<Scalar>::kernel_dz_Hessian_r5;

template<typename Scalar> ImplicitFunctionRidge<Scalar>::Tensor3D ImplicitFunctionRidge<Scalar>::ConstTensor3D = 
    {ImplicitFunctionRidge<Scalar>::Mat33::Zero(), ImplicitFunctionRidge<Scalar>::Mat33::Zero(), ImplicitFunctionRidge<Scalar>::Mat33::Zero()};
template<typename Scalar> ImplicitFunctionRidge<Scalar>::Tensor4D ImplicitFunctionRidge<Scalar>::ConstTensor4D = 
    {ImplicitFunctionRidge<Scalar>::ConstTensor3D, ImplicitFunctionRidge<Scalar>::ConstTensor3D, ImplicitFunctionRidge<Scalar>::ConstTensor3D};