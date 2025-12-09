#pragma once
#include <iostream>
#include "ImplicitFunction.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

template<typename Scalar> 
class CurvatureData {
    public:
        using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
        CurvatureData(){};
        CurvatureData(Scalar k1, Scalar k2, Vec3 t1, Vec3 t2) :
        k1_(k1), k2_(k2), t1_(t1), t2_(t2){};
    public:
        Scalar f_val_;
        Vec3 f_gradient_;
        Scalar k1_;
        Scalar k2_;
        Vec3 t1_; 
        Vec3 t2_;
        Scalar e1_;
        Scalar e2_;
        Vec3 e1_d1_; 
        Vec3 e2_d2_;
        Scalar e1_prime_ = 0;
        Scalar e2_prime_ = 0;
};


enum HRBFKernelType{ CubicKernelRBF, FifthPowerRBF};

template<typename Scalar>
class Hermite_RBF : public ImplicitFunction<Scalar> {
public:
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
    using VecX = Eigen::Matrix<Scalar, -1, 1>;
    using Mat33 = Eigen::Matrix <Scalar, 3,3>; 
    using Tensor3D = std::array<Mat33, 3>;
    using Tensor4D = std::array<std::array<Mat33, 3>, 3>;
    
    // constant 0 function
    Hermite_RBF() : coeff_a_(), coeff_b_(0, 0, 0, 0) {
        hrbf_ptr_ = this;
        InitKernel();
    }

    Hermite_RBF(const std::vector<Vec3> &control_points, const VecX &coeff_a,
                const Vec4 &coeff_b)
            : coeff_a_(coeff_a), coeff_b_(coeff_b), control_points_(control_points) {
                hrbf_ptr_ = this;
                InitKernel();
            }
    
    Hermite_RBF(const std::vector<Vec3> &control_points, const VecX &coeff_a,
                const Vec4 &coeff_b, Scalar offset)
            : coeff_a_(coeff_a), coeff_b_(coeff_b), control_points_(control_points), offset_value_(offset) {
                hrbf_ptr_ = this;
                InitKernel();
            }

    void InitKernel()
    {
        if(kernel_type_ == HRBFKernelType::CubicKernelRBF)
        {
            kernel_function_ = kernel_function_r3;
            kernel_gradient_ = kernel_gradient_r3;
            kernel_hessian_ = kernel_Hessian_r3;
            kernel_dx_hessian_ = kernel_dx_Hessian_r3;
            kernel_dy_hessian_ = kernel_dy_Hessian_r3;
            kernel_dz_hessian_ = kernel_dz_Hessian_r3;
        }
        if(kernel_type_ == HRBFKernelType::FifthPowerRBF)
        {
            kernel_function_ = kernel_function_r5;
            kernel_gradient_ = kernel_gradient_r5;
            kernel_hessian_ = kernel_Hessian_r5;
            kernel_dx_hessian_ = kernel_dx_Hessian_r5;
            kernel_dy_hessian_ = kernel_dy_Hessian_r5;
            kernel_dz_hessian_ = kernel_dz_Hessian_r5;
        }
    }

    Scalar evaluate(Scalar x, Scalar y, Scalar z) const override {
        // std::cout << "start implicit HRBF evaluate val " << std::endl;
        size_t num_pt = control_points_.size();
        int dim = 3;
        Vec3 p(x, y, z);

        VecX kern(num_pt * (dim + 1));
        for (size_t i = 0; i < num_pt; ++i) {
            kern(i) = kernel_function_(p, control_points_[i]);
        }
        Vec3 G;
        for (size_t i = 0; i < num_pt; ++i) {
            G = kernel_gradient_(p, control_points_[i]);
            for (int j = 0; j < dim; ++j) {
                kern(num_pt + i + j * num_pt) = G(j);
            }
        }
        Scalar loc_part = kern.dot(coeff_a_);

        Vec4 kb(1, p(0), p(1), p(2));
        Scalar poly_part = kb.dot(coeff_b_);

        // std::cout << "implicit HRBF evaluate val : " << loc_part + poly_part - offset_value_ << std::endl;

        return loc_part + poly_part - offset_value_;
    }

    Scalar evaluate_gradient(Scalar x, Scalar y, Scalar z, Scalar &gx, Scalar &gy, Scalar &gz) const override {
        size_t num_pt = control_points_.size();
        int dim = 3;
        Vec3 p(x, y, z);

        // std::cout << "start implicit HRBF evaluate val " << " pt num " <<  num_pt << std::endl;

        VecX kern(num_pt * (dim + 1));
        for (size_t i = 0; i < num_pt; ++i) {
            // std::cout << " pt index " << i << " " << control_points_[i] << std::endl;
            kern(i) = kernel_function_(p, control_points_[i]);
        }
        
        // std::cout << "start implicit HRBF evaluate val  1 " << std::endl;

        Vec3 G;
        Vec3 grad;
        grad.setZero();
        // sum(ai * fi)
        for (size_t i = 0; i < num_pt; ++i) {
            // std::cout << " pt index " << i << " " << control_points_[i] << std::endl;
            G = kernel_gradient_(p, control_points_[i]);
            grad += G * coeff_a_[i];
            for (int j = 0; j < dim; ++j) {
                kern(num_pt + i + j * num_pt) = G(j);
            }
        }
        // std::cout << "start implicit HRBF evaluate val  000 " << std::endl;
        // sum(hi * bi)
        Eigen::Matrix<Scalar, 3, 3> H;
        for (size_t i = 0; i < num_pt; ++i) {
            H = kernel_hessian_(p, control_points_[i]);
            grad += H.col(0) * coeff_a_[num_pt + i];
            grad += H.col(1) * coeff_a_[2 * num_pt + i];
            grad += H.col(2) * coeff_a_[3 * num_pt + i];
        }
        // c
        grad(0) += coeff_b_(1);
        grad(1) += coeff_b_(2);
        grad(2) += coeff_b_(3);

        gx = grad(0);
        gy = grad(1);
        gz = grad(2);
        // compute function value
        Scalar loc_part = kern.dot(coeff_a_);
        Vec4 kb(1, p(0), p(1), p(2));
        Scalar poly_part = kb.dot(coeff_b_);

        // std::cout << "implicit HRBF evaluate val : " << loc_part + poly_part - offset_value_ << std::endl;
        return loc_part + poly_part - offset_value_;
    }

    
    void EvaluateHessian(Scalar x, Scalar y, Scalar z, Mat33& hessian) const
    {
        size_t num_pt = control_points_.size();
        Vec3 p(x, y, z);
        hessian = Mat33::Zero();
        for(int i=0;i<num_pt;++i){
            // cal H(fx)
            hessian += coeff_a_[i] * kernel_hessian_(p, control_points_[i]);
            // cal H(Dx)
            hessian += coeff_a_[num_pt + i]    * kernel_dx_hessian_(p, control_points_[i]);
            // cal H(Dy)
            hessian += coeff_a_[num_pt *2 + i] * kernel_dy_hessian_(p, control_points_[i]);
            // cal H(Dz)
            hessian += coeff_a_[num_pt *3 + i] * kernel_dz_hessian_(p, control_points_[i]);
        }
        return ;
    }

    // void ComputeThirdDerivatives(const double x, const double y, const double z, Tensor3D& third_derivs) const
    // {
    //     size_t num_pt = control_points_.size();
    //     Vec3 p(x, y, z);
    //     Mat33 hessian_dx = Mat33::Zero();
    //     Mat33 hessian_dy = Mat33::Zero();
    //     Mat33 hessian_dz = Mat33::Zero();
    //     for(int i=0;i<num_pt;++i){
    //         hessian_dx += coeff_a_[i] * kernel_dx_hessian_(p, control_points_[i]);
    //         hessian_dy += coeff_a_[i] * kernel_dy_hessian_(p, control_points_[i]);
    //         hessian_dz += coeff_a_[i] * kernel_dz_hessian_(p, control_points_[i]);
    //         Mat33 dxyz_hdx = Mat33::Zero();
    //         Mat33 dxyz_hdy = Mat33::Zero();
    //         Mat33 dxyz_hdz = Mat33::Zero();
    //         compute_third_deriv_r5(p, control_points_[i], coeff_a_[i + num_pt], coeff_a_[i + num_pt * 2], 
    //             coeff_a_[i + num_pt* 3], dxyz_hdx, dxyz_hdy, dxyz_hdz);
    //         hessian_dx += dxyz_hdx;
    //         hessian_dy += dxyz_hdy;
    //         hessian_dz += dxyz_hdz;
    //     }
    //     // third_derivs.resize(3);
    //     third_derivs[0] = hessian_dx; // H_x = dH/dx
    //     third_derivs[1] = hessian_dy; // H_y = dH/dy
    //     third_derivs[2] = hessian_dz; // H_z = dH/dz
    // }

    void ComputeThirdDerivatives(const double x, const double y, const double z, Tensor3D& third_derivs) const
    {
        // Scalar h = 1e-8;
        Scalar h = 1e-6;
        Mat33 hessian_xp = Mat33::Zero(); 
        EvaluateHessian(x + h, y, z, hessian_xp);
        Mat33 hessian_xn = Mat33::Zero();
        EvaluateHessian(x - h, y, z, hessian_xn);
        Mat33 hessian_yp = Mat33::Zero();
        EvaluateHessian(x, y + h, z, hessian_yp);
        Mat33 hessian_yn = Mat33::Zero();
        EvaluateHessian(x, y - h, z, hessian_yn);
        Mat33 hessian_zp = Mat33::Zero();
        EvaluateHessian(x, y, z + h, hessian_zp);
        Mat33 hessian_zn = Mat33::Zero();
        EvaluateHessian(x, y, z - h, hessian_zn);
    
        // third_derivs.resize(3);
        third_derivs[0] = (hessian_xp - hessian_xn) / (2 * h); // H_x = dH/dx
        third_derivs[1] = (hessian_yp - hessian_yn) / (2 * h); // H_y = dH/dy
        third_derivs[2] = (hessian_zp - hessian_zn) / (2 * h); // H_z = dH/dz
    }

    
    void ComputeFourthDerivatives(double x, double y, double z, Tensor4D&  fourth_derivs) const
    {
        Scalar h = 1e-6;
        Tensor3D third_derivs_xp = ConstTensor3D;
        ComputeThirdDerivatives(x + h, y, z, third_derivs_xp);
        Tensor3D third_derivs_xn = ConstTensor3D;
        ComputeThirdDerivatives(x - h, y, z, third_derivs_xn);

        Tensor3D third_derivs_yp = ConstTensor3D;
        ComputeThirdDerivatives(x, y + h, z, third_derivs_yp);
        Tensor3D third_derivs_yn = ConstTensor3D;
        ComputeThirdDerivatives(x, y - h, z, third_derivs_yn);

        Tensor3D third_derivs_zp = ConstTensor3D;
        ComputeThirdDerivatives(x, y, z + h, third_derivs_zp);
        Tensor3D third_derivs_zn = ConstTensor3D;
        ComputeThirdDerivatives(x, y, z - h, third_derivs_zn);

        for(int i = 0; i < 3; ++i)
        {
            Mat33 fourth_deriv_x = (third_derivs_xp[i] - third_derivs_xn[i]) / (2 * h); 
            Mat33 fourth_deriv_y = (third_derivs_yp[i] - third_derivs_yn[i]) / (2 * h); 
            Mat33 fourth_deriv_z = (third_derivs_zp[i] - third_derivs_zn[i]) / (2 * h); 
            fourth_derivs[0][i] = fourth_deriv_x;
            fourth_derivs[1][i] = fourth_deriv_y;
            fourth_derivs[2][i] = fourth_deriv_z;
        }
    }

    // void ComputeFourthDerivatives(double x, double y, double z, Tensor4D& fourth_derivs) const
    // {
    //     Vec3 p(x, y, z);
    //     size_t num_pt = control_points_.size();
    //     for(int i=0;i<num_pt;++i){
    //         auto tensor_f4d  = kernel_foruth_derivative_r5(p, control_points_[i]);
    //         auto tensor_dx4d = dx_foruth_derivative_r5(p, control_points_[i]);     
    //         auto tensor_dy4d = dy_foruth_derivative_r5(p, control_points_[i]);    
    //         auto tensor_dz4d = dz_foruth_derivative_r5(p, control_points_[i]);    
    //         for( int n = 0; n < 3; ++n)
    //         {
    //             for(int m = 0; m < 3; ++m)
    //             {
    //                 fourth_derivs[n][m] += (coeff_a_[i] * tensor_f4d[n][m] 
    //                 + coeff_a_[i + num_pt] * tensor_dx4d[n][m]
    //                 + coeff_a_[i + num_pt * 2] * tensor_dy4d[n][m]
    //                 + coeff_a_[i + num_pt * 3] * tensor_dz4d[n][m] );
    //             }
    //         }
    //     }
    // }

    
    Scalar ComputeCurvatureSecondDerivative(const Vec3& gradient, 
                                            const Mat33& Hessian,
                                            const Tensor3D& third_derivs, 
                                            const Tensor4D& fourth_derivs,
                                            const Vec3& t1,
                                            const Scalar k1, 
                                            const Scalar e1) const
    {

        Scalar g_norm = sqrt(gradient[0] * gradient[0] + gradient[1]* gradient[1] + gradient[2] * gradient[2]);
        // auto normalized_t = arma::normalise(t1); 
        Scalar sum1 = 0;
        for(int i = 0; i < 3; ++i)
        {
            for(int j = 0; j < 3; ++j)
            {
                for(int l = 0; l < 3; ++l)
                {
                    for(int m = 0; m < 3; ++m)
                    {
                        // sum1 += fourth_derivs[i](j,l,m) * t1[i] * t1[j] * t1[l] * t1[m];
                        sum1 += fourth_derivs[i][j](l,m) * t1[i] * t1[j] * t1[l] * t1[m];
                    }
                }
            }
        }

        Scalar sum2 = 0;
        for(int i = 0; i < 3; ++i)
        {
            for(int j = 0; j < 3; ++j)
            {
                for(int l = 0; l < 3; ++l)
                {
                    sum2 += third_derivs[i](j, l) * t1[i] * t1[j] * gradient[l] * 6 * k1;
                }
            }
        }

        Scalar sum3 = 0;
        for(int i = 0; i < 3; ++i)
        {
            for(int j = 0; j < 3; ++j)
            {
                sum3 += (Hessian(i,j) * t1[i] * gradient[j] * 4 * e1 
                    + Hessian(i,j) * gradient[i] * gradient[j] * 3 * k1 * k1);  
            }
        }

        Scalar result = (sum1 + sum2 + sum3) / g_norm - 3 * k1 * k1 *k1;
        return result;
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
        Vec3 gradient;
        double f_val_ = evaluate_gradient(x, y, z, gradient[0], gradient[1], gradient[2]);
        // std::cout << " cur_data.f_val_ " << cur_data.f_val_ << std::endl;
        // std::cout << " Hrbf : finish evaluate_gradient ..." << std::endl;
        Mat33 hessian = Mat33::Zero();
        EvaluateHessian(x, y, z, hessian);

        if(0)
    {
        // std::cout << " Hrbf : finish EvaluateHessian ..." << std::endl;
        Tensor3D third_derivs = ConstTensor3D;
        ComputeThirdDerivatives(x, y, z, third_derivs);

        // Tensor4D fourth_derivs = ConstTensor4D;
        // ComputeFourthDerivatives(x, y, z, fourth_derivs);
        // std::cout << " Hrbf : finish ComputeFourthDerivatives ..." << std::endl;
        // ComputeThirdDerivatives
        cur_data = ComputePrincipalCurvaturesMonga(gradient, hessian);
        // std::cout << " Hrbf : finish ComputePrincipalCurvaturesMonga ..." << std::endl;
        cur_data.e1_ = ComputeCurvatureDerivative({x,y,z}, gradient, hessian, third_derivs, cur_data.t1_);
        cur_data.e2_ = ComputeCurvatureDerivative({x,y,z}, gradient, hessian, third_derivs, cur_data.t2_);
    }
        Eigen::SelfAdjointEigenSolver<Mat33> solver(hessian);
        Vec3 eigenValues = solver.eigenvalues();
        // std::cout << "eigen vals " << eigenValues << std::endl;
        Mat33 eigenVectors = solver.eigenvectors();
        // double maxEval = evals(2);               // largest eigenvalue
        cur_data.k1_ = eigenValues[0];
        cur_data.k2_ = eigenValues[2];
        cur_data.t1_ = eigenVectors.col(0);
        cur_data.t2_ = eigenVectors.col(2);
        cur_data.e1_  = gradient.dot(cur_data.t1_);
        cur_data.e2_  = gradient.dot(cur_data.t2_);
        cur_data.e1_prime_ = eigenVectors.col(0).transpose() * hessian * eigenVectors.col(0);
        cur_data.e2_prime_ = eigenVectors.col(2).transpose() * hessian * eigenVectors.col(2);
        // cur_data.e1_prime_ = ComputeCurvatureSecondDerivative(gradient, hessian, third_derivs,  
        //                                  fourth_derivs, cur_data.t1_, cur_data.k1_, cur_data.e1_);
        // cur_data.e2_prime_ = ComputeCurvatureSecondDerivative(gradient, hessian, third_derivs,  
        //                                  fourth_derivs, cur_data.t2_, cur_data.k2_, cur_data.e2_);    
        cur_data.f_gradient_ = gradient;
        cur_data.f_val_ = f_val_;
        // std::cout << " Hrbf : finish ComputeCurvatureDerivative2 ..." << std::endl;
    } 


    Scalar ComputeCurvatureDerivative(const std::array<Scalar,3>& pt, 
                                        const Vec3& gradient, 
                                        const Mat33& Hessian,
                                        const Tensor3D& third_derivs,  
                                        const Vec3& t1) const
    {
        Scalar g_norm = gradient.norm();
        Vec3 normalized_t = t1.normalized(); 
        Vec3 third_deriv_terms;
        for (int i = 0; i < 3; ++i) {
            third_deriv_terms(i) = normalized_t.transpose()
            * third_derivs[i] * normalized_t; // t1^T H_{x,y,z} t1
        }
        
        Scalar tHt = normalized_t.transpose() * Hessian * normalized_t;
        Scalar tHg = normalized_t.transpose() * Hessian * gradient;

        // Compute derivative: (||g||^2 * t1^T [t1^T H_x t1, t1^T H_y t1, t1^T H_z t1] - (t1^T H t1)(t1^T H g)) / ||g||^3
        Scalar numerator = g_norm * g_norm * (normalized_t.transpose().dot(third_deriv_terms)) - tHt * tHg;
        Scalar denominator = g_norm * g_norm * g_norm;
        return numerator / denominator;
    }

    
    Scalar ComputeCurvatureDerivative2(const std::array<Scalar,3>& pt, 
                                        const Vec3& t1, Vec3& e_gradient) const
    {
        // Scalar h = 1e-8;
        Scalar h = 1e-6;
        std::vector<std::array<Scalar,3>> pts(6);
        pts[0] = {pt[0]+ h, pt[1], pt[2]};
        pts[1] = {pt[0]- h, pt[1], pt[2]};
        pts[2] = {pt[0], pt[1] +h, pt[2]};
        pts[3] = {pt[0], pt[1] -h, pt[2]};
        pts[4] = {pt[0], pt[1], pt[2] +h};
        pts[5] = {pt[0], pt[1], pt[2] -h};
        std::vector<Scalar> e_vals(6);
        #pragma omp parallel for
        for(int i = 0; i < 6; ++i)
        {
            Mat33 Hessian_xp = Mat33::Zero(); 
            std::vector<Mat33> third_derivs_xp;
            const std::array<Scalar,3>& pt_xp = pts[i];
            Vec3 g_xp = Vec3::Zero();
            evaluate_gradient(pt_xp[0], pt_xp[1], pt_xp[2], g_xp[0], g_xp[1], g_xp[2]);
            EvaluateHessian(pt_xp[0], pt_xp[1], pt_xp[2], Hessian_xp);
            ComputeThirdDerivatives(pt_xp[0], pt_xp[1], pt_xp[2], third_derivs_xp);
            e_vals[i] = ComputeCurvatureDerivative(pt_xp, g_xp, Hessian_xp, third_derivs_xp, t1); 
        }
        e_gradient = Vec3::Zero();
        for(int i =0; i < 3; ++i)
        {
            e_gradient[i] = (e_vals[2*i] - e_vals[2*i +1]) /( 2 * h);
        } 
        return 0;
    }
public:
    static Hermite_RBF<Scalar>* hrbf_ptr_; 
    // static CurvatureData<Scalar> EvaluateCurDataStatic(double px, double py, double pz)
    // {
    //     CurvatureData<Scalar> cur_data;
    //     hrbf_ptr_->EvaluateCurvatureData(px,  py,  pz,  cur_data);
    //     return cur_data;
    // }

public:
    VecX coeff_a_;
    Vec4 coeff_b_;
    Scalar offset_value_ = 0;
    std::vector<Vec3> control_points_;
    HRBFKernelType kernel_type_ = HRBFKernelType::FifthPowerRBF;
    // HRBFKernelType kernel_type_ = HRBFKernelType::CubicKernelRBF;
    using KernelFunction = Scalar(*)(const Vec3&, const Vec3 &);
    using KernelGradient = Vec3(*)(const Vec3&, const Vec3 &);
    using KernelHessian = Eigen::Matrix<Scalar, 3, 3>(*)(const Vec3&, const Vec3 &);
    
    static KernelFunction kernel_function_;
    static KernelGradient kernel_gradient_;
    static KernelHessian kernel_hessian_;
    static KernelHessian kernel_dx_hessian_;
    static KernelHessian kernel_dy_hessian_;
    static KernelHessian kernel_dz_hessian_;

    static Tensor3D ConstTensor3D;  //= {Mat33::Zero(), Mat33::Zero(), Mat33::Zero()};
    static Tensor4D ConstTensor4D;  //= {ConstTensor3D, ConstTensor3D, ConstTensor3D};


    // |p1-p2|^3
    static Scalar kernel_function_r3(const Vec3 &p1, const Vec3 &p2) {
        return pow((p1 - p2).norm(), 3);
    }

    // 3 |p1-p2| (p1-p2)
    static Vec3 kernel_gradient_r3(const Vec3 &p1, const Vec3 &p2) {
        return 3 * (p1 - p2).norm() * (p1 - p2);
    }

    // 3 [ |p1-p2|I + (p1-p2)*(p1-p2)^T/|p1-p1| ]
    static Eigen::Matrix<Scalar, 3, 3> kernel_Hessian_r3(const Vec3 &p1, const Vec3 &p2) {
        Vec3 diff = p1 - p2;
        Scalar len = diff.norm();
        if (len < 1e-12) {
            return Eigen::Matrix<Scalar, 3, 3>::Zero();
        }
        Eigen::Matrix<Scalar, 3, 3> hess = diff * (diff.transpose() / len);
        hess(0, 0) += len;
        hess(1, 1) += len;
        hess(2, 2) += len;
        hess *= 3;
        return hess;
    }

    static Eigen::Matrix<Scalar, 3, 3> kernel_dx_Hessian_r3(const Vec3 &p1, const Vec3 &p2) {
        Vec3 diff = p1 - p2;
        Scalar len = diff.norm();
        if (len < 1e-12) {
            return Eigen::Matrix<Scalar, 3, 3>::Zero();
        }
        Eigen::Matrix<Scalar, 3, 3> Hess = Eigen::Matrix<Scalar, 3, 3>::Zero();
        Scalar l3 = len * len * len;
        Scalar dx = diff[0]; Scalar dy = diff[1]; Scalar dz = diff[2]; 
        Hess(0, 0) = 3 * dx / len -  dx * dx * dx / l3;
        Hess(0, 1) = dy / len -  dx * dx * dy / l3;
        Hess(0, 2) = dz / len -  dx * dx * dz / l3;

        Hess(1, 0) = dy / len -  dx * dx * dy / l3;
        Hess(1, 1) = dx / len -  dx * dy * dy / l3;
        Hess(1, 2) =  - dx * dz * dy / l3;

        Hess(2, 0) = dz / len - dx * dx * dz / l3;
        Hess(2, 1) = - dx * dz * dy / l3;
        Hess(2, 2) = dx / len - dx * dz * dz / l3;
        Hess *= 3.0; 
        return Hess;
    }

    static Eigen::Matrix<Scalar, 3, 3> kernel_dy_Hessian_r3(const Vec3 &p1, const Vec3 &p2) {
        Vec3 diff = p1 - p2;
        Scalar len = diff.norm();
        if (len < 1e-12) {
            return Eigen::Matrix<Scalar, 3, 3>::Zero();
        }
        Eigen::Matrix<Scalar, 3, 3> Hess = Eigen::Matrix<Scalar, 3, 3>::Zero();
        Scalar l3 = len * len * len;
        Scalar dx = diff[0]; Scalar dy = diff[1]; Scalar dz = diff[2]; 

        Hess(0, 0) = dy / len - dx * dx * dy / l3;
        Hess(0, 1) = dx / len - dx * dy * dy / l3;
        Hess(0, 2) = - dx * dy * dz / l3;

        Hess(1, 0) = dx / len - dx * dy * dy / l3;
        Hess(1, 1) = 3* dy / len - dy * dy * dy / l3;
        Hess(1, 2) = dz / len - dy * dy * dz / l3;

        Hess(2, 0) = - dx * dy * dz / l3;
        Hess(2, 1) = dz / len - dy * dy * dz / l3;
        Hess(2, 2) = dy / len - dy * dz * dz / l3;
        Hess *= 3.0; 
        return Hess;
    }

    static Eigen::Matrix<Scalar, 3, 3> kernel_dz_Hessian_r3(const Vec3 &p1, const Vec3 &p2) {
        Vec3 diff = p1 - p2;
        Scalar len = diff.norm();
        if (len < 1e-12) {
            return Eigen::Matrix<Scalar, 3, 3>::Zero();
        }
        Eigen::Matrix<Scalar, 3, 3> Hess = Eigen::Matrix<Scalar, 3, 3>::Zero();
        Scalar l3 = len * len * len;
        Scalar dx = diff[0]; Scalar dy = diff[1]; Scalar dz = diff[2]; 

        Hess(0, 0) = dz / len - dx * dx * dz / l3;
        Hess(0, 1) = - dx * dy * dz / l3;
        Hess(0, 2) = dx / len -  dx * dz * dz / l3;

        Hess(1, 0) =  -  dx * dy * dz / l3;
        Hess(1, 1) =  dz / len -  dy * dy * dz / l3;
        Hess(1, 2) =  dy / len -  dy * dz * dz / l3;

        Hess(2, 0) = dx / len - dx * dz * dz / l3;
        Hess(2, 1) = dy / len - dy * dz * dz / l3;
        Hess(2, 2) = 3 * dz / len - dz * dz * dz / l3;
        Hess *= 3.0; 
        return Hess;
    }


    static Scalar PtDist(const Vec3 &p1, const Vec3 &p2)
    {
        Vec3 v12 = p2 - p1;
        Scalar dist = sqrt(v12.dot(v12));
        return dist;
    }

    // |p1-p2|^5
    static Scalar kernel_function_r5(const Vec3 &p1, const Vec3 &p2) {
        Scalar r = PtDist(p1, p2);
        return pow(r, 5);
    }

    // 5 |p1-p2|^3 (p1-p2)
    static Vec3 kernel_gradient_r5(const Vec3 &p1, const Vec3 &p2) {
        Scalar r = PtDist(p1, p2);
        return 5 * pow(r,3) * (p1 - p2);
    }

    // 5 [ |p1-p2|^3I + 3(p1-p2)*(p1-p2)^T|p1-p1| ]
    static Eigen::Matrix<Scalar, 3, 3> kernel_Hessian_r5(const Vec3 &p1, const Vec3 &p2) {
        Vec3 diff = p1 - p2;
        Scalar len = diff.norm();
        Scalar l3 = pow(len,3);
        Eigen::Matrix<Scalar, 3, 3> hess =  diff * diff.transpose() * (15 * len);
        hess(0, 0) += 5 * l3;
        hess(1, 1) += 5 * l3;
        hess(2, 2) += 5 * l3;
        return hess;
    }

    static Eigen::Matrix<Scalar, 3, 3> kernel_dx_Hessian_r5(const Vec3 &p1, const Vec3 &p2) {
        Vec3 diff = p1 - p2;
        Scalar len = diff.norm();
        if (len < 1e-12) {
            return Eigen::Matrix<Scalar, 3, 3>::Zero();
        }
        Eigen::Matrix<Scalar, 3, 3> Hess = Eigen::Matrix<Scalar, 3, 3>::Zero();
        Scalar dx = diff[0]; Scalar dy = diff[1]; Scalar dz = diff[2]; 
        Hess(0, 0) = 15 * dx * dx * dx / len + 45 * dx * len;
        Hess(0, 1) = 15 * dx * dx * dy / len + 15 * dy * len;
        Hess(0, 2) = 15 * dx * dx * dz / len + 15 * dz * len;

        Hess(1, 0) = 15 * dx * dx * dy / len + 15 * dy * len;
        Hess(1, 1) = 15 * dx * dy * dy / len + 15 * dx * len;
        Hess(1, 2) = 15 * dx * dy * dz / len;

        Hess(2, 0) = 15 * dx * dx * dz / len + 15 * dz * len;
        Hess(2, 1) = 15 * dx * dy * dz / len;
        Hess(2, 2) = 15 * dx * dz * dz / len + 15 * dx * len;
        return Hess;
    }

    static Eigen::Matrix<Scalar, 3, 3> kernel_dy_Hessian_r5(const Vec3 &p1, const Vec3 &p2) {
        Vec3 diff = p1 - p2;
        Scalar len = diff.norm();
        if (len < 1e-12) {
            return Eigen::Matrix<Scalar, 3, 3>::Zero();
        }
        Eigen::Matrix<Scalar, 3, 3> Hess = Eigen::Matrix<Scalar, 3, 3>::Zero();
        Scalar dx = diff[0]; Scalar dy = diff[1]; Scalar dz = diff[2]; 

        Hess(0, 0) = 15 * dx * dx * dy / len + 15 * dy * len;
        Hess(0, 1) = 15 * dx * dy * dy / len + 15 * dx * len;
        Hess(0, 2) = 15 * dx * dy * dz / len;

        Hess(1, 0) = 15 * dx * dy * dy / len + 15 * dx * len;
        Hess(1, 1) = 15 * dy * dy * dy / len + 45 * dy * len;
        Hess(1, 2) = 15 * dy * dy * dz / len + 15 * dz * len;

        Hess(2, 0) = 15 * dx * dy * dz / len;
        Hess(2, 1) = 15 * dy * dy * dz / len + 15 * dz * len;
        Hess(2, 2) = 15 * dy * dz * dz / len + 15 * dy * len;
        return Hess;
    }

    static Eigen::Matrix<Scalar, 3, 3> kernel_dz_Hessian_r5(const Vec3 &p1, const Vec3 &p2) {
        Vec3 diff = p1 - p2;
        Scalar len = diff.norm();
        if (len < 1e-12) {
            return Eigen::Matrix<Scalar, 3, 3>::Zero();
        }
        Eigen::Matrix<Scalar, 3, 3> Hess = Eigen::Matrix<Scalar, 3, 3>::Zero();
        Scalar l3 = len * len * len;
        Scalar dx = diff[0]; Scalar dy = diff[1]; Scalar dz = diff[2]; 

        Hess(0, 0) = 15 * dx * dx * dz / len + 15 * dz * len;
        Hess(0, 1) = 15 * dx * dy * dz / len;
        Hess(0, 2) = 15 * dx * dz * dz / len + 15 * dx * len;

        Hess(1, 0) = 15 * dx * dy * dz / len;
        Hess(1, 1) = 15 * dy * dy * dz / len + 15 * dz * len;
        Hess(1, 2) = 15 * dy * dz * dz / len + 15 * dy * len;

        Hess(2, 0) = 15 * dx * dz * dz / len + 15 * dx * len;
        Hess(2, 1) = 15 * dy * dz * dz / len + 15 * dy * len;
        Hess(2, 2) = 15 * dz * dz * dz / len + 45 * dz * len;
        return Hess;
    }

    static void compute_third_deriv_r5(const Vec3 &p1, const Vec3 &p2, 
        const Scalar bx, const Scalar by, const Scalar bz,    
        Mat33& hessainDx, Mat33& hessainDy, Mat33& hessainDz ) {
        // Common term: (x-a)^2 + (y-b)^2 + (z-c)^2
        Vec3 diff = p1 - p2;
        Scalar dx = diff[0];
        Scalar dy = diff[1];
        Scalar dz = diff[2];
        Scalar len = std::sqrt(dx * dx + dy * dy + dz * dz);
        Scalar len_3 = std::pow(len, 3);

        // Initialize 3x3 matrix
        Mat33 dx_hessian_dx = Mat33::Zero();
        dx_hessian_dx(0, 0) = -((15 * std::pow(dx, 4)) / len_3) + (90 * dx * dx) / len + 45 * len;
        dx_hessian_dx(0, 1) = -((15 * std::pow(dx, 3) * dy) / len_3) + (45 * dx * dy) / len;
        dx_hessian_dx(0, 2) = -((15 * std::pow(dx, 3) * dz) / len_3) + (45 * dx * dz) / len;
        dx_hessian_dx(1, 0) = dx_hessian_dx(0, 1); 
        dx_hessian_dx(1, 1) = -((15 * dx * dx * dy * dy) / len_3) + (15 * dx * dx) / len + (15 * dy * dy) / len + 15 * len;
        dx_hessian_dx(1, 2) = -((15 * dx * dx * dy * dz) / len_3) + (15 * dy * dz) / len;
        dx_hessian_dx(2, 0) = dx_hessian_dx(0, 2); 
        dx_hessian_dx(2, 1) = dx_hessian_dx(1, 2); 
        dx_hessian_dx(2, 2) = -((15 * dx * dx * dz * dz) / len_3) + (15 * dx * dx) / len + (15 * dz * dz) / len + 15 * len;

        Mat33 dx_hessian_dy = Mat33::Zero();
        dx_hessian_dy(0, 0) = -((15 * std::pow(dx, 3) * dy) / len_3) + (45 * dx * dy) / len;
        dx_hessian_dy(0, 1) = -((15 * dx * dx * dy * dy) / len_3) + (15 * dx * dx) / len + (15 * dy * dy) / len + 15 * len;
        dx_hessian_dy(0, 2) = -((15 * dx * dx * dy * dz) / len_3) + (15 * dy * dz) / len;
        dx_hessian_dy(1, 0) = dx_hessian_dy(0, 1); 
        dx_hessian_dy(1, 1) = -((15 * dx * std::pow(dy, 3)) / len_3) + (45 * dx * dy) / len;
        dx_hessian_dy(1, 2) = -((15 * dx * std::pow(dy, 2) * dz) / len_3) + (15 * dx * dz) / len;
        dx_hessian_dy(2, 0) = dx_hessian_dy(0, 2); 
        dx_hessian_dy(2, 1) = dx_hessian_dy(1, 2); 
        dx_hessian_dy(2, 2) = -((15 * dx * dy * std::pow(dz, 2)) / len_3) + (15 * dx * dy) / len;

        Mat33 dx_hessian_dz = Mat33::Zero();
        dx_hessian_dz(0, 0) = -((15 * std::pow(dx, 3) * dz) / len_3) + (45 * dx * dz) / len;
        dx_hessian_dz(0, 1) = -((15 * dx * dx * dy * dz) / len_3) + (15 * dy * dz) / len;
        dx_hessian_dz(0, 2) = -((15 * dx * dx * std::pow(dz, 2)) / len_3) + (15 * dx * dx) / len + (15 * std::pow(dz, 2)) / len + 15 * len;
        dx_hessian_dz(1, 0) = dx_hessian_dz(0, 1); 
        dx_hessian_dz(1, 1) = -((15 * dx * std::pow(dy, 2) * dz) / len_3) + (15 * dx * dz) / len;
        dx_hessian_dz(1, 2) = -((15 * dx * dy * std::pow(dz, 2)) / len_3) + (15 * dx * dy) / len;
        dx_hessian_dz(2, 0) = dx_hessian_dz(0, 2); 
        dx_hessian_dz(2, 1) = dx_hessian_dz(1, 2); 
        dx_hessian_dz(2, 2) = -((15 * dx * std::pow(dz, 3)) / len_3) + (45 * dx * dz) / len;

        Mat33 dy_hessian_dx = dx_hessian_dy;
        Mat33 dy_hessian_dy = Mat33::Zero();
        dy_hessian_dy(0, 0) = -((15 * dx * dx * dy * dy) / len_3) + (15 * dx * dx) / len + (15 * dy * dy) / len + 15 * len;
        dy_hessian_dy(0, 1) = -((15 * dx * std::pow(dy, 3)) / len_3) + (45 * dx * dy) / len;
        dy_hessian_dy(0, 2) = -((15 * dx * std::pow(dy, 2) * dz) / len_3) + (15 * dx * dz) / len;
        dy_hessian_dy(1, 0) = dy_hessian_dy(0, 1); 
        dy_hessian_dy(1, 1) = -((15 * std::pow(dy, 4)) / len_3) + (90 * dy * dy) / len + 45 * len;
        dy_hessian_dy(1, 2) = -((15 * std::pow(dy, 3) * dz) / len_3) + (45 * dy * dz) / len;
        dy_hessian_dy(2, 0) = dy_hessian_dy(0, 2); 
        dy_hessian_dy(2, 1) = dy_hessian_dy(1, 2); 
        dy_hessian_dy(2, 2) = -((15 * dy * dy * dz * dz) / len_3) + (15 * dy * dy) / len + (15 * dz * dz) / len + 15 * len;
        Mat33 dy_hessian_dz = Mat33::Zero();
        dy_hessian_dz(0, 0) = -((15 * dx * dx * dy * dz) / len_3) + (15 * dy * dz) / len;
        dy_hessian_dz(0, 1) = -((15 * dx * std::pow(dy, 2) * dz) / len_3) + (15 * dx * dz) / len;
        dy_hessian_dz(0, 2) = -((15 * dx * dy * std::pow(dz, 2)) / len_3) + (15 * dx * dy) / len;
        dy_hessian_dz(1, 0) = dy_hessian_dz(0, 1); 
        dy_hessian_dz(1, 1) = -((15 * std::pow(dy, 3) * dz) / len_3) + (45 * dy * dz) / len;
        dy_hessian_dz(1, 2) = -((15 * std::pow(dy, 2) * std::pow(dz, 2)) / len_3) + (15 * dy * dy) / len + (15 * std::pow(dz, 2)) / len + 15 * len;
        dy_hessian_dz(2, 0) = dy_hessian_dz(0, 2); 
        dy_hessian_dz(2, 1) = dy_hessian_dz(1, 2); 
        dy_hessian_dz(2, 2) = -((15 * dy * std::pow(dz, 3)) / len_3) + (45 * dy * dz) / len;

        Mat33 dz_hessian_dx = dx_hessian_dz;
        Mat33 dz_hessian_dy = dy_hessian_dz;
        Mat33 dz_hessian_dz = Mat33::Zero();
        dz_hessian_dz(0, 0) = -((15 * dx * dx * dz * dz) / len_3) + (15 * dx * dx) / len + (15 * dz * dz) / len + 15 * len;
        dz_hessian_dz(0, 1) = -((15 * dx * dy * std::pow(dz, 2)) / len_3) + (15 * dx * dy) / len;
        dz_hessian_dz(0, 2) = -((15 * dx * std::pow(dz, 3)) / len_3) + (45 * dx * dz) / len;
        dz_hessian_dz(1, 0) = dz_hessian_dz(0, 1); 
        dz_hessian_dz(1, 1) = -((15 * dy * dy * dz * dz) / len_3) + (15 * dy * dy) / len + (15 * dz * dz) / len + 15 * len;
        dz_hessian_dz(1, 2) = -((15 * dy * std::pow(dz, 3)) / len_3) + (45 * dy * dz) / len;
        dz_hessian_dz(2, 0) = dz_hessian_dz(0, 2); 
        dz_hessian_dz(2, 1) = dz_hessian_dz(1, 2); 
        dz_hessian_dz(2, 2) = -((15 * std::pow(dz, 4)) / len_3) + (90 * dz * dz) / len + 45 * len;
        hessainDx = bx * dx_hessian_dx + by * dy_hessian_dx + bz * dz_hessian_dx;
        hessainDy = bx * dx_hessian_dy + by * dy_hessian_dy + bz * dz_hessian_dy;
        hessainDz = bx * dx_hessian_dz + by * dy_hessian_dz + bz * dz_hessian_dz;
        return ;
    }

    static Tensor4D kernel_foruth_derivative_r5(const Vec3 &p1, const Vec3 &p2) {
        Tensor4D tensor = ConstTensor4D;
        // Common denominator terms
        Vec3 diff = p1 - p2;
        Scalar dx = diff[0];
        Scalar dy = diff[1];
        Scalar dz = diff[2];
        double denom = dx * dx + dy * dy + dz * dz;
        double sqrt_denom = std::sqrt(denom);
        double denom_3_2 = std::pow(denom, 1.5);
        if(sqrt_denom < 1e-12)
        {

            return tensor;
        }
        // Precompute common factors
        Scalar dx2 = dx * dx;
        Scalar dy2 = dy * dy;
        Scalar dz2 = dz * dz;
        Scalar dx3 = dx2 * dx;
        Scalar dy3 = dy2 * dy;
        Scalar dz3 = dz2 * dz;
        Scalar dx4 = dx2 * dx2;
        Scalar dy4 = dy2 * dy2;
        Scalar dz4 = dz2 * dz2;

        // Populate the 3x3 matrices for each (i,j) pair
        // i=0
        tensor[0][0] << 
            (-15.0 * dx4 / denom_3_2) + (90.0 * dx2 / sqrt_denom) + (45.0 * sqrt_denom),
            (-15.0 * dx3 * dy / denom_3_2) + (45.0 * dx * dy / sqrt_denom),
            (-15.0 * dx3 * dz / denom_3_2) + (45.0 * dx * dz / sqrt_denom),
            (-15.0 * dx3 * dy / denom_3_2) + (45.0 * dx * dy / sqrt_denom),
            (-15.0 * dx2 * dy2 / denom_3_2) + (15.0 * dx2 / sqrt_denom) + (15.0 * dy2 / sqrt_denom) + (15.0 * sqrt_denom),
            (-15.0 * dx2 * dy * dz / denom_3_2) + (15.0 * dy * dz / sqrt_denom),
            (-15.0 * dx3 * dz / denom_3_2) + (45.0 * dx * dz / sqrt_denom),
            (-15.0 * dx2 * dy * dz / denom_3_2) + (15.0 * dy * dz / sqrt_denom),
            (-15.0 * dx2 * dz2 / denom_3_2) + (15.0 * dx2 / sqrt_denom) + (15.0 * dz2 / sqrt_denom) + (15.0 * sqrt_denom);

        tensor[0][1] << 
            (-15.0 * dx3 * dy / denom_3_2) + (45.0 * dx * dy / sqrt_denom),
            (-15.0 * dx2 * dy2 / denom_3_2) + (15.0 * dx2 / sqrt_denom) + (15.0 * dy2 / sqrt_denom) + (15.0 * sqrt_denom),
            (-15.0 * dx2 * dy * dz / denom_3_2) + (15.0 * dy * dz / sqrt_denom),
            (-15.0 * dx2 * dy2 / denom_3_2) + (15.0 * dx2 / sqrt_denom) + (15.0 * dy2 / sqrt_denom) + (15.0 * sqrt_denom),
            (-15.0 * dx * dy3 / denom_3_2) + (45.0 * dx * dy / sqrt_denom),
            (-15.0 * dx * dy2 * dz / denom_3_2) + (15.0 * dx * dz / sqrt_denom),
            (-15.0 * dx2 * dy * dz / denom_3_2) + (15.0 * dy * dz / sqrt_denom),
            (-15.0 * dx * dy2 * dz / denom_3_2) + (15.0 * dx * dz / sqrt_denom),
            (-15.0 * dx * dy * dz2 / denom_3_2) + (15.0 * dx * dy / sqrt_denom);

        tensor[0][2] << 
            (-15.0 * dx3 * dz / denom_3_2) + (45.0 * dx * dz / sqrt_denom),
            (-15.0 * dx2 * dy * dz / denom_3_2) + (15.0 * dy * dz / sqrt_denom),
            (-15.0 * dx2 * dz2 / denom_3_2) + (15.0 * dx2 / sqrt_denom) + (15.0 * dz2 / sqrt_denom) + (15.0 * sqrt_denom),
            (-15.0 * dx2 * dy * dz / denom_3_2) + (15.0 * dy * dz / sqrt_denom),
            (-15.0 * dx * dy2 * dz / denom_3_2) + (15.0 * dx * dz / sqrt_denom),
            (-15.0 * dx * dy * dz2 / denom_3_2) + (15.0 * dx * dy / sqrt_denom),
            (-15.0 * dx2 * dz2 / denom_3_2) + (15.0 * dx2 / sqrt_denom) + (15.0 * dz2 / sqrt_denom) + (15.0 * sqrt_denom),
            (-15.0 * dx * dy * dz2 / denom_3_2) + (15.0 * dx * dy / sqrt_denom),
            (-15.0 * dx * dz3 / denom_3_2) + (45.0 * dx * dz / sqrt_denom);

        // i=1
        tensor[1][0] = tensor[0][1]; // Symmetry: T_ijkl = T_jikl

        tensor[1][1] << 
            (-15.0 * dx2 * dy2 / denom_3_2) + (15.0 * dx2 / sqrt_denom) + (15.0 * dy2 / sqrt_denom) + (15.0 * sqrt_denom),
            (-15.0 * dx * dy3 / denom_3_2) + (45.0 * dx * dy / sqrt_denom),
            (-15.0 * dx * dy2 * dz / denom_3_2) + (15.0 * dx * dz / sqrt_denom),
            (-15.0 * dx * dy3 / denom_3_2) + (45.0 * dx * dy / sqrt_denom),
            (-15.0 * dy4 / denom_3_2) + (90.0 * dy2 / sqrt_denom) + (45.0 * sqrt_denom),
            (-15.0 * dy3 * dz / denom_3_2) + (45.0 * dy * dz / sqrt_denom),
            (-15.0 * dx * dy2 * dz / denom_3_2) + (15.0 * dx * dz / sqrt_denom),
            (-15.0 * dy3 * dz / denom_3_2) + (45.0 * dy * dz / sqrt_denom),
            (-15.0 * dy2 * dz2 / denom_3_2) + (15.0 * dy2 / sqrt_denom) + (15.0 * dz2 / sqrt_denom) + (15.0 * sqrt_denom);

        tensor[1][2] << 
            (-15.0 * dx2 * dy * dz / denom_3_2) + (15.0 * dy * dz / sqrt_denom),
            (-15.0 * dx * dy2 * dz / denom_3_2) + (15.0 * dx * dz / sqrt_denom),
            (-15.0 * dx * dy * dz2 / denom_3_2) + (15.0 * dx * dy / sqrt_denom),
            (-15.0 * dx * dy2 * dz / denom_3_2) + (15.0 * dx * dz / sqrt_denom),
            (-15.0 * dy3 * dz / denom_3_2) + (45.0 * dy * dz / sqrt_denom),
            (-15.0 * dy2 * dz2 / denom_3_2) + (15.0 * dy2 / sqrt_denom) + (15.0 * dz2 / sqrt_denom) + (15.0 * sqrt_denom),
            (-15.0 * dx * dy * dz2 / denom_3_2) + (15.0 * dx * dy / sqrt_denom),
            (-15.0 * dy2 * dz2 / denom_3_2) + (15.0 * dy2 / sqrt_denom) + (15.0 * dz2 / sqrt_denom) + (15.0 * sqrt_denom),
            (-15.0 * dy * dz3 / denom_3_2) + (45.0 * dy * dz / sqrt_denom);

        // i=2
        tensor[2][0] = tensor[0][2]; // Symmetry: T_ijkl = T_jikl

        tensor[2][1] = tensor[1][2]; // Symmetry: T_ijkl = T_jikl

        tensor[2][2] << 
            (-15.0 * dx2 * dz2 / denom_3_2) + (15.0 * dx2 / sqrt_denom) + (15.0 * dz2 / sqrt_denom) + (15.0 * sqrt_denom),
            (-15.0 * dx * dy * dz2 / denom_3_2) + (15.0 * dx * dy / sqrt_denom),
            (-15.0 * dx * dz3 / denom_3_2) + (45.0 * dx * dz / sqrt_denom),
            (-15.0 * dx * dy * dz2 / denom_3_2) + (15.0 * dx * dy / sqrt_denom),
            (-15.0 * dy2 * dz2 / denom_3_2) + (15.0 * dy2 / sqrt_denom) + (15.0 * dz2 / sqrt_denom) + (15.0 * sqrt_denom),
            (-15.0 * dy * dz3 / denom_3_2) + (45.0 * dy * dz / sqrt_denom),
            (-15.0 * dx * dz3 / denom_3_2) + (45.0 * dx * dz / sqrt_denom),
            (-15.0 * dy * dz3 / denom_3_2) + (45.0 * dy * dz / sqrt_denom),
            (-15.0 * dz4 / denom_3_2) + (90.0 * dz2 / sqrt_denom) + (45.0 * sqrt_denom);
        return tensor;
    }

    static Tensor4D dx_foruth_derivative_r5(const Vec3 &p1, const Vec3 &p2) {
        Tensor4D tensor = {};
        // Common denominator terms
        Vec3 diff = p1 - p2;
        Scalar dx = diff[0];
        Scalar dy = diff[1];
        Scalar dz = diff[2];
        double denom = dx * dx + dy * dy + dz * dz;
        double sqrt_denom = std::sqrt(denom);
        double denom_3_2 = std::pow(denom, 1.5);
        Scalar denom_5_2 = std::pow(denom, 2.5);
        if(sqrt_denom < 1e-12)
        {
            return tensor;
        }
        // Precompute common factors
        Scalar dx2 = dx * dx;
        Scalar dy2 = dy * dy;
        Scalar dz2 = dz * dz;
        Scalar dx3 = dx2 * dx;
        Scalar dy3 = dy2 * dy;
        Scalar dz3 = dz2 * dz;
        Scalar dx4 = dx2 * dx2;
        Scalar dy4 = dy2 * dy2;
        Scalar dz4 = dz2 * dz2;
        Scalar dx5 = dx4 * dx;
        Scalar dy5 = dy4 * dy;
        Scalar dz5 = dz4 * dz;

        // Populate the 3x3 matrices for each (i,j) pair
        // i=0
        tensor[0][0] << 
            (45.0 * dx5 / denom_5_2) - (150.0 * dx3 / denom_3_2) + (225.0 * dx / sqrt_denom),
            (45.0 * dx4 * dy / denom_5_2) - (90.0 * dx2 * dy / denom_3_2) + (45.0 * dy / sqrt_denom),
            (45.0 * dx4 * dz / denom_5_2) - (90.0 * dx2 * dz / denom_3_2) + (45.0 * dz / sqrt_denom),
            (45.0 * dx4 * dy / denom_5_2) - (90.0 * dx2 * dy / denom_3_2) + (45.0 * dy / sqrt_denom),
            (45.0 * dx3 * dy2 / denom_5_2) - (15.0 * dx3 / denom_3_2) - (45.0 * dx * dy2 / denom_3_2) + (45.0 * dx / sqrt_denom),
            (45.0 * dx3 * dy * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx4 * dz / denom_5_2) - (90.0 * dx2 * dz / denom_3_2) + (45.0 * dz / sqrt_denom),
            (45.0 * dx3 * dy * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx3 * dz2 / denom_5_2) - (15.0 * dx3 / denom_3_2) - (45.0 * dx * dz2 / denom_3_2) + (45.0 * dx / sqrt_denom);

        tensor[0][1] << 
            (45.0 * dx4 * dy / denom_5_2) - (90.0 * dx2 * dy / denom_3_2) + (45.0 * dy / sqrt_denom),
            (45.0 * dx3 * dy2 / denom_5_2) - (15.0 * dx3 / denom_3_2) - (45.0 * dx * dy2 / denom_3_2) + (45.0 * dx / sqrt_denom),
            (45.0 * dx3 * dy * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx3 * dy2 / denom_5_2) - (15.0 * dx3 / denom_3_2) - (45.0 * dx * dy2 / denom_3_2) + (45.0 * dx / sqrt_denom),
            (45.0 * dx2 * dy3 / denom_5_2) - (45.0 * dx2 * dy / denom_3_2) - (15.0 * dy3 / denom_3_2) + (45.0 * dy / sqrt_denom),
            (45.0 * dx2 * dy2 * dz / denom_5_2) - (15.0 * dx2 * dz / denom_3_2) - (15.0 * dy2 * dz / denom_3_2) + (15.0 * dz / sqrt_denom),
            (45.0 * dx3 * dy * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx2 * dy2 * dz / denom_5_2) - (15.0 * dx2 * dz / denom_3_2) - (15.0 * dy2 * dz / denom_3_2) + (15.0 * dz / sqrt_denom),
            (45.0 * dx2 * dy * dz2 / denom_5_2) - (15.0 * dx2 * dy / denom_3_2) - (15.0 * dy * dz2 / denom_3_2) + (15.0 * dy / sqrt_denom);

        tensor[0][2] << 
            (45.0 * dx4 * dz / denom_5_2) - (90.0 * dx2 * dz / denom_3_2) + (45.0 * dz / sqrt_denom),
            (45.0 * dx3 * dy * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx3 * dz2 / denom_5_2) - (15.0 * dx3 / denom_3_2) - (45.0 * dx * dz2 / denom_3_2) + (45.0 * dx / sqrt_denom),
            (45.0 * dx3 * dy * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx2 * dy2 * dz / denom_5_2) - (15.0 * dx2 * dz / denom_3_2) - (15.0 * dy2 * dz / denom_3_2) + (15.0 * dz / sqrt_denom),
            (45.0 * dx2 * dy * dz2 / denom_5_2) - (15.0 * dx2 * dy / denom_3_2) - (15.0 * dy * dz2 / denom_3_2) + (15.0 * dy / sqrt_denom),
            (45.0 * dx3 * dz2 / denom_5_2) - (15.0 * dx3 / denom_3_2) - (45.0 * dx * dz2 / denom_3_2) + (45.0 * dx / sqrt_denom),
            (45.0 * dx2 * dy * dz2 / denom_5_2) - (15.0 * dx2 * dy / denom_3_2) - (15.0 * dy * dz2 / denom_3_2) + (15.0 * dy / sqrt_denom),
            (45.0 * dx2 * dz3 / denom_5_2) - (45.0 * dx2 * dz / denom_3_2) - (15.0 * dz3 / denom_3_2) + (45.0 * dz / sqrt_denom);

        // i=1
        tensor[1][0] = tensor[0][1]; // Symmetry: T_ijkl = T_jikl

        tensor[1][1] << 
            (45.0 * dx3 * dy2 / denom_5_2) - (15.0 * dx3 / denom_3_2) - (45.0 * dx * dy2 / denom_3_2) + (45.0 * dx / sqrt_denom),
            (45.0 * dx2 * dy3 / denom_5_2) - (45.0 * dx2 * dy / denom_3_2) - (15.0 * dy3 / denom_3_2) + (45.0 * dy / sqrt_denom),
            (45.0 * dx2 * dy2 * dz / denom_5_2) - (15.0 * dx2 * dz / denom_3_2) - (15.0 * dy2 * dz / denom_3_2) + (15.0 * dz / sqrt_denom),
            (45.0 * dx2 * dy3 / denom_5_2) - (45.0 * dx2 * dy / denom_3_2) - (15.0 * dy3 / denom_3_2) + (45.0 * dy / sqrt_denom),
            (45.0 * dx * dy4 / denom_5_2) - (90.0 * dx * dy2 / denom_3_2) + (45.0 * dx / sqrt_denom),
            (45.0 * dx * dy3 * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx2 * dy2 * dz / denom_5_2) - (15.0 * dx2 * dz / denom_3_2) - (15.0 * dy2 * dz / denom_3_2) + (15.0 * dz / sqrt_denom),
            (45.0 * dx * dy3 * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx * dy2 * dz2 / denom_5_2) - (15.0 * dx * dy2 / denom_3_2) - (15.0 * dx * dz2 / denom_3_2) + (15.0 * dx / sqrt_denom);

        tensor[1][2] << 
            (45.0 * dx3 * dy * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx2 * dy2 * dz / denom_5_2) - (15.0 * dx2 * dz / denom_3_2) - (15.0 * dy2 * dz / denom_3_2) + (15.0 * dz / sqrt_denom),
            (45.0 * dx2 * dy * dz2 / denom_5_2) - (15.0 * dx2 * dy / denom_3_2) - (15.0 * dy * dz2 / denom_3_2) + (15.0 * dy / sqrt_denom),
            (45.0 * dx2 * dy2 * dz / denom_5_2) - (15.0 * dx2 * dz / denom_3_2) - (15.0 * dy2 * dz / denom_3_2) + (15.0 * dz / sqrt_denom),
            (45.0 * dx * dy3 * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx * dy2 * dz2 / denom_5_2) - (15.0 * dx * dy2 / denom_3_2) - (15.0 * dx * dz2 / denom_3_2) + (15.0 * dx / sqrt_denom),
            (45.0 * dx2 * dy * dz2 / denom_5_2) - (15.0 * dx2 * dy / denom_3_2) - (15.0 * dy * dz2 / denom_3_2) + (15.0 * dy / sqrt_denom),
            (45.0 * dx * dy2 * dz2 / denom_5_2) - (15.0 * dx * dy2 / denom_3_2) - (15.0 * dx * dz2 / denom_3_2) + (15.0 * dx / sqrt_denom),
            (45.0 * dx * dy * dz3 / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2);

        // i=2
        tensor[2][0] = tensor[0][2]; // Symmetry: T_ijkl = T_jikl

        tensor[2][1] = tensor[1][2]; // Symmetry: T_ijkl = T_jikl

        tensor[2][2] << 
            (45.0 * dx3 * dz2 / denom_5_2) - (15.0 * dx3 / denom_3_2) - (45.0 * dx * dz2 / denom_3_2) + (45.0 * dx / sqrt_denom),
            (45.0 * dx2 * dy * dz2 / denom_5_2) - (15.0 * dx2 * dy / denom_3_2) - (15.0 * dy * dz2 / denom_3_2) + (15.0 * dy / sqrt_denom),
            (45.0 * dx2 * dz3 / denom_5_2) - (45.0 * dx2 * dz / denom_3_2) - (15.0 * dz3 / denom_3_2) + (45.0 * dz / sqrt_denom),
            (45.0 * dx2 * dy * dz2 / denom_5_2) - (15.0 * dx2 * dy / denom_3_2) - (15.0 * dy * dz2 / denom_3_2) + (15.0 * dy / sqrt_denom),
            (45.0 * dx * dy2 * dz2 / denom_5_2) - (15.0 * dx * dy2 / denom_3_2) - (15.0 * dx * dz2 / denom_3_2) + (15.0 * dx / sqrt_denom),
            (45.0 * dx * dy * dz3 / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx2 * dz3 / denom_5_2) - (45.0 * dx2 * dz / denom_3_2) - (15.0 * dz3 / denom_3_2) + (45.0 * dz / sqrt_denom),
            (45.0 * dx * dy * dz3 / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx * dz4 / denom_5_2) - (90.0 * dx * dz2 / denom_3_2) + (45.0 * dx / sqrt_denom);

        return tensor;
    }

    
    static Tensor4D dy_foruth_derivative_r5(const Vec3 &p1, const Vec3 &p2) {
        Tensor4D tensor = ConstTensor4D;
        // Common denominator terms
        Vec3 diff = p1 - p2;
        Scalar dx = diff[0];
        Scalar dy = diff[1];
        Scalar dz = diff[2];
        double denom = dx * dx + dy * dy + dz * dz;
        double sqrt_denom = std::sqrt(denom);
        double denom_3_2 = std::pow(denom, 1.5);
        Scalar denom_5_2 = std::pow(denom, 2.5);
        if(sqrt_denom < 1e-12)
        {
            return tensor;
        }
        // Precompute common factors
        Scalar dx2 = dx * dx;
        Scalar dy2 = dy * dy;
        Scalar dz2 = dz * dz;
        Scalar dx3 = dx2 * dx;
        Scalar dy3 = dy2 * dy;
        Scalar dz3 = dz2 * dz;
        Scalar dx4 = dx2 * dx2;
        Scalar dy4 = dy2 * dy2;
        Scalar dz4 = dz2 * dz2;
        Scalar dx5 = dx4 * dx;
        Scalar dy5 = dy4 * dy;
        Scalar dz5 = dz4 * dz;

        // Populate the 3x3 matrices for each (i,j) pair
        // i=0
        tensor[0][0] << 
            (45.0 * dx4 * dy / denom_5_2) - (90.0 * dx2 * dy / denom_3_2) + (45.0 * dy / sqrt_denom),
            (45.0 * dx3 * dy2 / denom_5_2) - (15.0 * dx3 / denom_3_2) - (45.0 * dx * dy2 / denom_3_2) + (45.0 * dx / sqrt_denom),
            (45.0 * dx3 * dy * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx3 * dy2 / denom_5_2) - (15.0 * dx3 / denom_3_2) - (45.0 * dx * dy2 / denom_3_2) + (45.0 * dx / sqrt_denom),
            (45.0 * dx2 * dy3 / denom_5_2) - (45.0 * dx2 * dy / denom_3_2) - (15.0 * dy3 / denom_3_2) + (45.0 * dy / sqrt_denom),
            (45.0 * dx2 * dy2 * dz / denom_5_2) - (15.0 * dx2 * dz / denom_3_2) - (15.0 * dy2 * dz / denom_3_2) + (15.0 * dz / sqrt_denom),
            (45.0 * dx3 * dy * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx2 * dy2 * dz / denom_5_2) - (15.0 * dx2 * dz / denom_3_2) - (15.0 * dy2 * dz / denom_3_2) + (15.0 * dz / sqrt_denom),
            (45.0 * dx2 * dy * dz2 / denom_5_2) - (15.0 * dx2 * dy / denom_3_2) - (15.0 * dy * dz2 / denom_3_2) + (15.0 * dy / sqrt_denom);

        tensor[0][1] << 
            (45.0 * dx3 * dy2 / denom_5_2) - (15.0 * dx3 / denom_3_2) - (45.0 * dx * dy2 / denom_3_2) + (45.0 * dx / sqrt_denom),
            (45.0 * dx2 * dy3 / denom_5_2) - (45.0 * dx2 * dy / denom_3_2) - (15.0 * dy3 / denom_3_2) + (45.0 * dy / sqrt_denom),
            (45.0 * dx2 * dy2 * dz / denom_5_2) - (15.0 * dx2 * dz / denom_3_2) - (15.0 * dy2 * dz / denom_3_2) + (15.0 * dz / sqrt_denom),
            (45.0 * dx2 * dy3 / denom_5_2) - (45.0 * dx2 * dy / denom_3_2) - (15.0 * dy3 / denom_3_2) + (45.0 * dy / sqrt_denom),
            (45.0 * dx * dy4 / denom_5_2) - (90.0 * dx * dy2 / denom_3_2) + (45.0 * dx / sqrt_denom),
            (45.0 * dx * dy3 * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx2 * dy2 * dz / denom_5_2) - (15.0 * dx2 * dz / denom_3_2) - (15.0 * dy2 * dz / denom_3_2) + (15.0 * dz / sqrt_denom),
            (45.0 * dx * dy3 * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx * dy2 * dz2 / denom_5_2) - (15.0 * dx * dy2 / denom_3_2) - (15.0 * dx * dz2 / denom_3_2) + (15.0 * dx / sqrt_denom);

        tensor[0][2] << 
            (45.0 * dx3 * dy * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx2 * dy2 * dz / denom_5_2) - (15.0 * dx2 * dz / denom_3_2) - (15.0 * dy2 * dz / denom_3_2) + (15.0 * dz / sqrt_denom),
            (45.0 * dx2 * dy * dz2 / denom_5_2) - (15.0 * dx2 * dy / denom_3_2) - (15.0 * dy * dz2 / denom_3_2) + (15.0 * dy / sqrt_denom),
            (45.0 * dx2 * dy2 * dz / denom_5_2) - (15.0 * dx2 * dz / denom_3_2) - (15.0 * dy2 * dz / denom_3_2) + (15.0 * dz / sqrt_denom),
            (45.0 * dx * dy3 * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx * dy2 * dz2 / denom_5_2) - (15.0 * dx * dy2 / denom_3_2) - (15.0 * dx * dz2 / denom_3_2) + (15.0 * dx / sqrt_denom),
            (45.0 * dx2 * dy * dz2 / denom_5_2) - (15.0 * dx2 * dy / denom_3_2) - (15.0 * dy * dz2 / denom_3_2) + (15.0 * dy / sqrt_denom),
            (45.0 * dx * dy2 * dz2 / denom_5_2) - (15.0 * dx * dy2 / denom_3_2) - (15.0 * dx * dz2 / denom_3_2) + (15.0 * dx / sqrt_denom),
            (45.0 * dx * dy * dz3 / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2);

        // i=1
        tensor[1][0] = tensor[0][1]; // Symmetry: T_ijkl = T_jikl

        tensor[1][1] << 
            (45.0 * dx2 * dy3 / denom_5_2) - (45.0 * dx2 * dy / denom_3_2) - (15.0 * dy3 / denom_3_2) + (45.0 * dy / sqrt_denom),
            (45.0 * dx * dy4 / denom_5_2) - (90.0 * dx * dy2 / denom_3_2) + (45.0 * dx / sqrt_denom),
            (45.0 * dx * dy3 * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx * dy4 / denom_5_2) - (90.0 * dx * dy2 / denom_3_2) + (45.0 * dx / sqrt_denom),
            (45.0 * dy5 / denom_5_2) - (150.0 * dy3 / denom_3_2) + (225.0 * dy / sqrt_denom),
            (45.0 * dy4 * dz / denom_5_2) - (90.0 * dy2 * dz / denom_3_2) + (45.0 * dz / sqrt_denom),
            (45.0 * dx * dy3 * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dy4 * dz / denom_5_2) - (90.0 * dy2 * dz / denom_3_2) + (45.0 * dz / sqrt_denom),
            (45.0 * dy3 * dz2 / denom_5_2) - (15.0 * dy3 / denom_3_2) - (45.0 * dy * dz2 / denom_3_2) + (45.0 * dy / sqrt_denom);

        tensor[1][2] << 
            (45.0 * dx2 * dy2 * dz / denom_5_2) - (15.0 * dx2 * dz / denom_3_2) - (15.0 * dy2 * dz / denom_3_2) + (15.0 * dz / sqrt_denom),
            (45.0 * dx * dy3 * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx * dy2 * dz2 / denom_5_2) - (15.0 * dx * dy2 / denom_3_2) - (15.0 * dx * dz2 / denom_3_2) + (15.0 * dx / sqrt_denom),
            (45.0 * dx * dy3 * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dy4 * dz / denom_5_2) - (90.0 * dy2 * dz / denom_3_2) + (45.0 * dz / sqrt_denom),
            (45.0 * dy3 * dz2 / denom_5_2) - (15.0 * dy3 / denom_3_2) - (45.0 * dy * dz2 / denom_3_2) + (45.0 * dy / sqrt_denom),
            (45.0 * dx * dy2 * dz2 / denom_5_2) - (15.0 * dx * dy2 / denom_3_2) - (15.0 * dx * dz2 / denom_3_2) + (15.0 * dx / sqrt_denom),
            (45.0 * dy3 * dz2 / denom_5_2) - (15.0 * dy3 / denom_3_2) - (45.0 * dy * dz2 / denom_3_2) + (45.0 * dy / sqrt_denom),
            (45.0 * dy2 * dz3 / denom_5_2) - (45.0 * dy2 * dz / denom_3_2) - (15.0 * dz3 / denom_3_2) + (45.0 * dz / sqrt_denom);

        // i=2
        tensor[2][0] = tensor[0][2]; // Symmetry: T_ijkl = T_jikl

        tensor[2][1] = tensor[1][2]; // Symmetry: T_ijkl = T_jikl

        tensor[2][2] << 
            (45.0 * dx2 * dy * dz2 / denom_5_2) - (15.0 * dx2 * dy / denom_3_2) - (15.0 * dy * dz2 / denom_3_2) + (15.0 * dy / sqrt_denom),
            (45.0 * dx * dy2 * dz2 / denom_5_2) - (15.0 * dx * dy2 / denom_3_2) - (15.0 * dx * dz2 / denom_3_2) + (15.0 * dx / sqrt_denom),
            (45.0 * dx * dy * dz3 / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx * dy2 * dz2 / denom_5_2) - (15.0 * dx * dy2 / denom_3_2) - (15.0 * dx * dz2 / denom_3_2) + (15.0 * dx / sqrt_denom),
            (45.0 * dy3 * dz2 / denom_5_2) - (15.0 * dy3 / denom_3_2) - (45.0 * dy * dz2 / denom_3_2) + (45.0 * dy / sqrt_denom),
            (45.0 * dy2 * dz3 / denom_5_2) - (45.0 * dy2 * dz / denom_3_2) - (15.0 * dz3 / denom_3_2) + (45.0 * dz / sqrt_denom),
            (45.0 * dx * dy * dz3 / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dy2 * dz3 / denom_5_2) - (45.0 * dy2 * dz / denom_3_2) - (15.0 * dz3 / denom_3_2) + (45.0 * dz / sqrt_denom),
            (45.0 * dy * dz4 / denom_5_2) - (90.0 * dy * dz2 / denom_3_2) + (45.0 * dy / sqrt_denom);
        return tensor;
    }

    static Tensor4D dz_foruth_derivative_r5(const Vec3 &p1, const Vec3 &p2) {
        Tensor4D tensor = ConstTensor4D;
        // Common denominator terms
        Vec3 diff = p1 - p2;
        Scalar dx = diff[0];
        Scalar dy = diff[1];
        Scalar dz = diff[2];
        double denom = dx * dx + dy * dy + dz * dz;
        double sqrt_denom = std::sqrt(denom);
        double denom_3_2 = std::pow(denom, 1.5);
        Scalar denom_5_2 = std::pow(denom, 2.5);
        if(sqrt_denom < 1e-12)
        {
            return tensor;
        }
        // Precompute common factors
        Scalar dx2 = dx * dx;
        Scalar dy2 = dy * dy;
        Scalar dz2 = dz * dz;
        Scalar dx3 = dx2 * dx;
        Scalar dy3 = dy2 * dy;
        Scalar dz3 = dz2 * dz;
        Scalar dx4 = dx2 * dx2;
        Scalar dy4 = dy2 * dy2;
        Scalar dz4 = dz2 * dz2;
        Scalar dx5 = dx4 * dx;
        Scalar dy5 = dy4 * dy;
        Scalar dz5 = dz4 * dz;

        // Populate the 3x3 matrices for each (i,j) pair
        // i=0
        tensor[0][0] << 
            (45.0 * dx4 * dz / denom_5_2) - (90.0 * dx2 * dz / denom_3_2) + (45.0 * dz / sqrt_denom),
            (45.0 * dx3 * dy * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx3 * dz2 / denom_5_2) - (15.0 * dx3 / denom_3_2) - (45.0 * dx * dz2 / denom_3_2) + (45.0 * dx / sqrt_denom),
            (45.0 * dx3 * dy * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx2 * dy2 * dz / denom_5_2) - (15.0 * dx2 * dz / denom_3_2) - (15.0 * dy2 * dz / denom_3_2) + (15.0 * dz / sqrt_denom),
            (45.0 * dx2 * dy * dz2 / denom_5_2) - (15.0 * dx2 * dy / denom_3_2) - (15.0 * dy * dz2 / denom_3_2) + (15.0 * dy / sqrt_denom),
            (45.0 * dx3 * dz2 / denom_5_2) - (15.0 * dx3 / denom_3_2) - (45.0 * dx * dz2 / denom_3_2) + (45.0 * dx / sqrt_denom),
            (45.0 * dx2 * dy * dz2 / denom_5_2) - (15.0 * dx2 * dy / denom_3_2) - (15.0 * dy * dz2 / denom_3_2) + (15.0 * dy / sqrt_denom),
            (45.0 * dx2 * dz3 / denom_5_2) - (45.0 * dx2 * dz / denom_3_2) - (15.0 * dz3 / denom_3_2) + (45.0 * dz / sqrt_denom);

        tensor[0][1] << 
            (45.0 * dx3 * dy * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx2 * dy2 * dz / denom_5_2) - (15.0 * dx2 * dz / denom_3_2) - (15.0 * dy2 * dz / denom_3_2) + (15.0 * dz / sqrt_denom),
            (45.0 * dx2 * dy * dz2 / denom_5_2) - (15.0 * dx2 * dy / denom_3_2) - (15.0 * dy * dz2 / denom_3_2) + (15.0 * dy / sqrt_denom),
            (45.0 * dx2 * dy2 * dz / denom_5_2) - (15.0 * dx2 * dz / denom_3_2) - (15.0 * dy2 * dz / denom_3_2) + (15.0 * dz / sqrt_denom),
            (45.0 * dx * dy3 * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx * dy2 * dz2 / denom_5_2) - (15.0 * dx * dy2 / denom_3_2) - (15.0 * dx * dz2 / denom_3_2) + (15.0 * dx / sqrt_denom),
            (45.0 * dx2 * dy * dz2 / denom_5_2) - (15.0 * dx2 * dy / denom_3_2) - (15.0 * dy * dz2 / denom_3_2) + (15.0 * dy / sqrt_denom),
            (45.0 * dx * dy2 * dz2 / denom_5_2) - (15.0 * dx * dy2 / denom_3_2) - (15.0 * dx * dz2 / denom_3_2) + (15.0 * dx / sqrt_denom),
            (45.0 * dx * dy * dz3 / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2);

        tensor[0][2] << 
            (45.0 * dx3 * dz2 / denom_5_2) - (15.0 * dx3 / denom_3_2) - (45.0 * dx * dz2 / denom_3_2) + (45.0 * dx / sqrt_denom),
            (45.0 * dx2 * dy * dz2 / denom_5_2) - (15.0 * dx2 * dy / denom_3_2) - (15.0 * dy * dz2 / denom_3_2) + (15.0 * dy / sqrt_denom),
            (45.0 * dx2 * dz3 / denom_5_2) - (45.0 * dx2 * dz / denom_3_2) - (15.0 * dz3 / denom_3_2) + (45.0 * dz / sqrt_denom),
            (45.0 * dx2 * dy * dz2 / denom_5_2) - (15.0 * dx2 * dy / denom_3_2) - (15.0 * dy * dz2 / denom_3_2) + (15.0 * dy / sqrt_denom),
            (45.0 * dx * dy2 * dz2 / denom_5_2) - (15.0 * dx * dy2 / denom_3_2) - (15.0 * dx * dz2 / denom_3_2) + (15.0 * dx / sqrt_denom),
            (45.0 * dx * dy * dz3 / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx2 * dz3 / denom_5_2) - (45.0 * dx2 * dz / denom_3_2) - (15.0 * dz3 / denom_3_2) + (45.0 * dz / sqrt_denom),
            (45.0 * dx * dy * dz3 / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx * dz4 / denom_5_2) - (90.0 * dx * dz2 / denom_3_2) + (45.0 * dx / sqrt_denom);

        // i=1
        tensor[1][0] = tensor[0][1]; // Symmetry: T_ijkl = T_jikl

        tensor[1][1] << 
            (45.0 * dx2 * dy2 * dz / denom_5_2) - (15.0 * dx2 * dz / denom_3_2) - (15.0 * dy2 * dz / denom_3_2) + (15.0 * dz / sqrt_denom),
            (45.0 * dx * dy3 * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx * dy2 * dz2 / denom_5_2) - (15.0 * dx * dy2 / denom_3_2) - (15.0 * dx * dz2 / denom_3_2) + (15.0 * dx / sqrt_denom),
            (45.0 * dx * dy3 * dz / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dy4 * dz / denom_5_2) - (90.0 * dy2 * dz / denom_3_2) + (45.0 * dz / sqrt_denom),
            (45.0 * dy3 * dz2 / denom_5_2) - (15.0 * dy3 / denom_3_2) - (45.0 * dy * dz2 / denom_3_2) + (45.0 * dy / sqrt_denom),
            (45.0 * dx * dy2 * dz2 / denom_5_2) - (15.0 * dx * dy2 / denom_3_2) - (15.0 * dx * dz2 / denom_3_2) + (15.0 * dx / sqrt_denom),
            (45.0 * dy3 * dz2 / denom_5_2) - (15.0 * dy3 / denom_3_2) - (45.0 * dy * dz2 / denom_3_2) + (45.0 * dy / sqrt_denom),
            (45.0 * dy2 * dz3 / denom_5_2) - (45.0 * dy2 * dz / denom_3_2) - (15.0 * dz3 / denom_3_2) + (45.0 * dz / sqrt_denom);

        tensor[1][2] << 
            (45.0 * dx2 * dy * dz2 / denom_5_2) - (15.0 * dx2 * dy / denom_3_2) - (15.0 * dy * dz2 / denom_3_2) + (15.0 * dy / sqrt_denom),
            (45.0 * dx * dy2 * dz2 / denom_5_2) - (15.0 * dx * dy2 / denom_3_2) - (15.0 * dx * dz2 / denom_3_2) + (15.0 * dx / sqrt_denom),
            (45.0 * dx * dy * dz3 / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx * dy2 * dz2 / denom_5_2) - (15.0 * dx * dy2 / denom_3_2) - (15.0 * dx * dz2 / denom_3_2) + (15.0 * dx / sqrt_denom),
            (45.0 * dy3 * dz2 / denom_5_2) - (15.0 * dy3 / denom_3_2) - (45.0 * dy * dz2 / denom_3_2) + (45.0 * dy / sqrt_denom),
            (45.0 * dy2 * dz3 / denom_5_2) - (45.0 * dy2 * dz / denom_3_2) - (15.0 * dz3 / denom_3_2) + (45.0 * dz / sqrt_denom),
            (45.0 * dx * dy * dz3 / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dy2 * dz3 / denom_5_2) - (45.0 * dy2 * dz / denom_3_2) - (15.0 * dz3 / denom_3_2) + (45.0 * dz / sqrt_denom),
            (45.0 * dy * dz4 / denom_5_2) - (90.0 * dy * dz2 / denom_3_2) + (45.0 * dy / sqrt_denom);

        // i=2
        tensor[2][0] = tensor[0][2]; // Symmetry: T_ijkl = T_jikl

        tensor[2][1] = tensor[1][2]; // Symmetry: T_ijkl = T_jikl

        tensor[2][2] << 
            (45.0 * dx2 * dz3 / denom_5_2) - (45.0 * dx2 * dz / denom_3_2) - (15.0 * dz3 / denom_3_2) + (45.0 * dz / sqrt_denom),
            (45.0 * dx * dy * dz3 / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dx * dz4 / denom_5_2) - (90.0 * dx * dz2 / denom_3_2) + (45.0 * dx / sqrt_denom),
            (45.0 * dx * dy * dz3 / denom_5_2) - (45.0 * dx * dy * dz / denom_3_2),
            (45.0 * dy2 * dz3 / denom_5_2) - (45.0 * dy2 * dz / denom_3_2) - (15.0 * dz3 / denom_3_2) + (45.0 * dz / sqrt_denom),
            (45.0 * dy * dz4 / denom_5_2) - (90.0 * dy * dz2 / denom_3_2) + (45.0 * dy / sqrt_denom),
            (45.0 * dx * dz4 / denom_5_2) - (90.0 * dx * dz2 / denom_3_2) + (45.0 * dx / sqrt_denom),
            (45.0 * dy * dz4 / denom_5_2) - (90.0 * dy * dz2 / denom_3_2) + (45.0 * dy / sqrt_denom),
            (45.0 * dz5 / denom_5_2) - (150.0 * dz3 / denom_3_2) + (225.0 * dz / sqrt_denom);
        
        return tensor;
    }







};

template<typename Scalar>
Hermite_RBF<Scalar>* Hermite_RBF<Scalar>::hrbf_ptr_ = nullptr; 

template<typename Scalar> Hermite_RBF<Scalar>::KernelFunction Hermite_RBF<Scalar>::kernel_function_ 
            = &Hermite_RBF<Scalar>::kernel_function_r5;

template<typename Scalar> Hermite_RBF<Scalar>::KernelGradient Hermite_RBF<Scalar>::kernel_gradient_ 
            = &Hermite_RBF<Scalar>::kernel_gradient_r5;

template<typename Scalar> Hermite_RBF<Scalar>::KernelHessian Hermite_RBF<Scalar>::kernel_hessian_ 
            = &Hermite_RBF<Scalar>::kernel_Hessian_r5;

template<typename Scalar> Hermite_RBF<Scalar>::KernelHessian Hermite_RBF<Scalar>::kernel_dx_hessian_ 
            = &Hermite_RBF<Scalar>::kernel_dx_Hessian_r5;

template<typename Scalar> Hermite_RBF<Scalar>::KernelHessian Hermite_RBF<Scalar>::kernel_dy_hessian_ 
            = &Hermite_RBF<Scalar>::kernel_dy_Hessian_r5;

template<typename Scalar> Hermite_RBF<Scalar>::KernelHessian Hermite_RBF<Scalar>::kernel_dz_hessian_ 
            = &Hermite_RBF<Scalar>::kernel_dz_Hessian_r5;

template<typename Scalar> Hermite_RBF<Scalar>::Tensor3D Hermite_RBF<Scalar>::ConstTensor3D = 
    {Hermite_RBF<Scalar>::Mat33::Zero(), Hermite_RBF<Scalar>::Mat33::Zero(), Hermite_RBF<Scalar>::Mat33::Zero()};
template<typename Scalar> Hermite_RBF<Scalar>::Tensor4D Hermite_RBF<Scalar>::ConstTensor4D = 
    {Hermite_RBF<Scalar>::ConstTensor3D, Hermite_RBF<Scalar>::ConstTensor3D, Hermite_RBF<Scalar>::ConstTensor3D};