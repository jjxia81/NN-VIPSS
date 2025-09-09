#pragma once
#include <iostream>
#include "ImplicitFunction.h"
#include <Eigen/Core>

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
};


template<typename Scalar>
class Hermite_RBF : public ImplicitFunction<Scalar> {
public:
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
    using VecX = Eigen::Matrix<Scalar, -1, 1>;
    using Mat33 = Eigen::Matrix <Scalar, 3,3>; 
    

    // constant 0 function
    Hermite_RBF() : coeff_a_(), coeff_b_(0, 0, 0, 0) {
        hrbf_ptr_ = this;
    }

    Hermite_RBF(const std::vector<Vec3> &control_points, const VecX &coeff_a,
                const Vec4 &coeff_b)
            : coeff_a_(coeff_a), coeff_b_(coeff_b), control_points_(control_points) {
                hrbf_ptr_ = this;
            }
    
    Hermite_RBF(const std::vector<Vec3> &control_points, const VecX &coeff_a,
                const Vec4 &coeff_b, Scalar offset)
            : coeff_a_(coeff_a), coeff_b_(coeff_b), control_points_(control_points), offset_value_(offset) {
                hrbf_ptr_ = this;
            }

    Scalar evaluate(Scalar x, Scalar y, Scalar z) const override {
        size_t num_pt = control_points_.size();
        int dim = 3;
        Vec3 p(x, y, z);

        VecX kern(num_pt * (dim + 1));
        for (size_t i = 0; i < num_pt; ++i) {
            kern(i) = kernel_function(p, control_points_[i]);
        }
        Vec3 G;
        for (size_t i = 0; i < num_pt; ++i) {
            G = kernel_gradient(p, control_points_[i]);
            for (int j = 0; j < dim; ++j) {
                kern(num_pt + i + j * num_pt) = G(j);
            }
        }
        Scalar loc_part = kern.dot(coeff_a_);

        Vec4 kb(1, p(0), p(1), p(2));
        Scalar poly_part = kb.dot(coeff_b_);

        return loc_part + poly_part - offset_value_;
    }

    Scalar evaluate_gradient(Scalar x, Scalar y, Scalar z, Scalar &gx, Scalar &gy, Scalar &gz) const override {
        size_t num_pt = control_points_.size();
        int dim = 3;
        Vec3 p(x, y, z);

        VecX kern(num_pt * (dim + 1));
        for (size_t i = 0; i < num_pt; ++i) {
            kern(i) = kernel_function(p, control_points_[i]);
        }

        Vec3 G;
        Vec3 grad;
        grad.setZero();
        // sum(ai * fi)
        for (size_t i = 0; i < num_pt; ++i) {
            G = kernel_gradient(p, control_points_[i]);
            grad += G * coeff_a_[i];
            for (int j = 0; j < dim; ++j) {
                kern(num_pt + i + j * num_pt) = G(j);
            }
        }
        // sum(hi * bi)
        Eigen::Matrix<Scalar, 3, 3> H;
        for (size_t i = 0; i < num_pt; ++i) {
            H = kernel_Hessian(p, control_points_[i]);
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
        return loc_part + poly_part - offset_value_;
    }

    void EvaluateHessian(Scalar x, Scalar y, Scalar z, Mat33& hessian) const
    {
        // size_t num_pt = npt;
        double pt[3] = {x, y, z};
        size_t num_pt = control_points_.size();
        Mat33 Hess;
        for(int i=0;i<num_pt;++i){
            // Kernal_Hessian_Function_2p(pt, p_pts+i*3, H);
            Vec3 diff = {x - control_points_[i][0], y - control_points_[i][1], z - control_points_[i][2]};
            Scalar len =  diff.norm();
            if (len > Scalar(1e-16)) {
            Hess = diff * (diff.transpose() / len);
                Hess(0, 0) += len;
                Hess(1, 1) += len;
                Hess(2, 2) += len;
                Hess *= 3;
                hessian += coeff_a_[i] * Hess;
                // std::cout << " hessian 0 " << hessian << std::endl;
            }
            Scalar dx = diff(0);
            Scalar dy = diff(1);
            Scalar dz = diff(2);
            Scalar l3 = len * len * len;
            if (len > Scalar(1e-16)) {
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
                Hess = coeff_a_[num_pt  + i] * 3.0 * Hess; 
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
                hessian += coeff_a_[num_pt + num_pt + i] * Hess * 3.0;

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
                hessian += coeff_a_[num_pt + num_pt * 2 + i] * 3.0 * Hess;
                // std::cout << " hessian 3 " << Hess << std::endl;
            }
        }
        return ;
    }

    void ComputeThirdDerivatives(double x, double y, double z, std::vector<Mat33>& third_derivs) const
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
    
        third_derivs.resize(3);
        third_derivs[0] = (hessian_xp - hessian_xn) / (2 * h); // H_x = dH/dx
        third_derivs[1] = (hessian_yp - hessian_yn) / (2 * h); // H_y = dH/dy
        third_derivs[2] = (hessian_zp - hessian_zn) / (2 * h); // H_z = dH/dz
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
        // std::cout << " Hrbf : finish EvaluateHessian ..." << std::endl;
        std::vector<Mat33> third_derivs;
        ComputeThirdDerivatives(x, y, z, third_derivs);
        // std::cout << " Hrbf : finish ComputeThirdDerivatives ..." << std::endl;
        cur_data = ComputePrincipalCurvaturesMonga(gradient, hessian);
        // std::cout << " Hrbf : finish ComputePrincipalCurvaturesMonga ..." << std::endl;
        cur_data.e1_ = ComputeCurvatureDerivative({x,y,z}, gradient, hessian, third_derivs, cur_data.t1_);
        cur_data.e2_ = ComputeCurvatureDerivative({x,y,z}, gradient, hessian, third_derivs, cur_data.t2_);
        // std::cout << " Hrbf : finish ComputeCurvatureDerivative ..." << std::endl;
        ComputeCurvatureDerivative2({x,y,z}, cur_data.t1_, cur_data.e1_d1_);
        ComputeCurvatureDerivative2({x,y,z}, cur_data.t2_, cur_data.e2_d2_);
        cur_data.f_gradient_ = gradient;
        cur_data.f_val_ = f_val_;
        // std::cout << " Hrbf : finish ComputeCurvatureDerivative2 ..." << std::endl;
    } 


    Scalar ComputeCurvatureDerivative(const std::array<Scalar,3>& pt, 
                                        const Vec3& gradient, 
                                        const Mat33& Hessian,
                                        const std::vector<Mat33>& third_derivs,  
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

private:
    VecX coeff_a_;
    Vec4 coeff_b_;
    Scalar offset_value_ = 0;
    std::vector<Vec3> control_points_;

    // |p1-p2|^3
    static Scalar kernel_function(const Vec3 &p1, const Vec3 &p2) {
        return pow((p1 - p2).norm(), 3);
    }

    // 3 |p1-p2| (p1-p2)
    static Vec3 kernel_gradient(const Vec3 &p1, const Vec3 &p2) {
        return 3 * (p1 - p2).norm() * (p1 - p2);
    }

    // 3 [ |p1-p2|I + (p1-p2)*(p1-p2)^T/|p1-p1| ]
    static Eigen::Matrix<Scalar, 3, 3> kernel_Hessian(const Vec3 &p1, const Vec3 &p2) {
        Vec3 diff = p1 - p2;
        Scalar len = diff.norm();
        if (len < 1e-8) {
            return Eigen::Matrix<Scalar, 3, 3>::Zero();
        }

        Eigen::Matrix<Scalar, 3, 3> hess = diff * (diff.transpose() / len);
        hess(0, 0) += len;
        hess(1, 1) += len;
        hess(2, 2) += len;
        hess *= 3;

        return hess;
    }
};

template<typename Scalar>
Hermite_RBF<Scalar>* Hermite_RBF<Scalar>::hrbf_ptr_ = nullptr; 
