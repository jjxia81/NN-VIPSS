
#include "voronoi_gen.h"
#include "local_vipss.hpp"
#include "vipss_ridges.h"
#include <unordered_set>
#include <stack>
#include <queue>

string CalEdgeToken(int a, int b)
{
    if (a < b)
    {
        return std::to_string(a) +"_" + std::to_string(b); 
    } 
    return std::to_string(b) +"_" + std::to_string(a); 
}

std::shared_ptr<RBF_Core> VIPSSRidges::g_hrfb_ptr = std::make_shared<RBF_Core>();
std::shared_ptr<RBF_Core> VIPSSRidges::hrfb_ptr_ = std::make_shared<RBF_Core>();
std::string VIPSSRidges::out_dir_ = "./";
std::vector<std::string> VIPSSRidges::edge_curv_values_string; 
std::vector<std::vector<PrincipleCurvature>> VIPSSRidges::edge_sample_curv_dataset;
std::vector<arma::vec3> VIPSSRidges::search_pts_all;
std::vector<double> VIPSSRidges::search_pts_iso_vals;
VIPSSRidges::Point VIPSSRidges::ori_center_ = {0, 0, 0};
double VIPSSRidges::scale_ = 1.0;

std::unordered_map<string, std::vector<VIPSSRidges::Point>> VIPSSRidges::curves_pts_map;
std::unordered_map<string, std::vector<std::vector<size_t>>> VIPSSRidges::curves_edges_map;

double PointDist(const std::array<double, 3>& p1, const std::array<double, 3>& p2)
{
    double dx = p1[0] - p2[0];
    double dy = p1[1] - p2[1];
    double dz = p1[2] - p2[2];
    return sqrt(dx * dx + dy * dy + dz * dz);

}

bool VIPSSRidges::LoadMeshPly(const std::string & mesh_path)
{
    mesh_points_.clear();
    mesh_faces_.clear();
    std::vector<std::vector<size_t>> faces;
    readPlyMesh(mesh_path, mesh_points_, faces);
    mesh_faces_.resize(faces.size());
    int f_count = 0;
    for(const auto& face : faces)
    {
        mesh_faces_[f_count] = {face[0], face[1], face[2]};
        f_count ++;
    }
    for(auto& p : mesh_points_)
    {
        p[0] = (p[0] - ori_center_[0]) / scale_; 
        p[1] = (p[1] - ori_center_[1]) / scale_; 
        p[2] = (p[2] - ori_center_[2]) / scale_; 
    }
    return true;
}

void VIPSSRidges::SetDataCenterAndScale(const Point& center, const double scale)
{
    ori_center_ = center;
    scale_ = scale;
}

bool VIPSSRidges::ProcessFaces()
{
 
    return false;
}

bool VIPSSRidges::CalculateCreaseValues()
{
    size_t ptn = mesh_points_.size();
    crease_values_.resize(ptn);
    for(int i = 0; i < ptn; ++i)
    {
        const auto& eig_vec = point_eig_vecs_[i];
        const auto& gradient = point_graidents_[i];
        crease_values_[i] = arma::dot(eig_vec, gradient);
         
    }
    std::string crease_vals_path = "crease_vals.txt";
    WriteVectorValsToCSV(crease_vals_path, crease_values_);
    return true;
}

bool VIPSSRidges::CalMeshPointsGradientAndEigenVecs(LocalVipss* local_vipss)
{
    local_vipss_ = local_vipss;
    size_t ptn = mesh_points_.size();
    point_eig_vecs_.resize(ptn);
    point_graidents_.resize(ptn);
    std::vector<double> points;
    std::vector<double> eig_vecs;
    std::vector<double> gredients;
    for(int i = 0; i < ptn; ++i)
    {
        double pt[3] = {mesh_points_[i][0], mesh_points_[i][1], mesh_points_[i][2]};
        arma::mat hessian = local_vipss->NNHRBFHessianOMP(pt) * (-1.0);
        arma::vec eigval;
        arma::mat eigvec;
        // arma::eig_sym(eigval, eigvec, hessian);
        arma::mat H_sym = 0.5 * (hessian + hessian.t());
        arma::eig_sym(eigval, eigvec, H_sym);
        eigval = arma::abs(eigval);
        size_t max_id = eigval.index_max();
        Vec max_vec = eigvec.col(max_id);
        // arma::uvec indices = arma::sort_index(eigval, "descend");
        // Vec max_vec = eigvec.col(indices(0));
        point_eig_vecs_[i] = arma::normalise(max_vec);
        double g[3];
        double f_val = local_vipss->NatureNeighborGradientOMP(pt, g);
        Vec gradient = {g[0], g[1], g[2]};
        point_graidents_[i] = arma::normalise(gradient);
    }

    return true;
}


VIPSSRidges::Vec VIPSSRidges::ComputeGradient(std::shared_ptr<RBF_Core> rfb_ptr, const Point& p, double h) {
    Vec grad;
    for (int i = 0; i < 3; ++i) {
        R3Pt p_plus(p[0], p[1], p[2]),  p_minus(p[0], p[1], p[2]);
        p_plus[i] += h;
        p_minus[i] -= h;
        grad(i) = (rfb_ptr->Dist_Function(p_plus) - rfb_ptr->Dist_Function(p_minus)) / (2 * h);
    }
    return grad;
}



// Function to compute the Hessian numerically
arma::mat VIPSSRidges::ComputeHessian(std::shared_ptr<RBF_Core> rfb_ptr, Point x, double h) {
    int n = x.size();
    arma::mat H = arma::zeros(3, 3);
    double px = x[0];
    double py = x[1];
    double pz = x[2];
    // Compute Hessian matrix elements using central differences
    for (int j = 0; j < 3; ++j) { // i corresponds to gx, gy, gz
        double grad_minus[3];
        double grad_plus[3];
        // Compute gradient at (x+h, y, z), (x-h, y, z) for j=0 (x), etc.
        if (j == 0) { // Perturb x
            rfb_ptr->evaluate_gradient(px + h, py, pz, grad_minus[0], grad_minus[1], grad_minus[2]);
            rfb_ptr->evaluate_gradient(px - h, py, pz, grad_plus[0], grad_plus[1], grad_plus[2]);
        } else if (j == 1) { // Perturb y
            rfb_ptr->evaluate_gradient(px, py + h, pz, grad_minus[0], grad_minus[1], grad_minus[2]);
            rfb_ptr->evaluate_gradient(px, py - h, pz, grad_plus[0], grad_plus[1], grad_plus[2]);
        } else { // Perturb z
            rfb_ptr->evaluate_gradient(px, py, pz + h, grad_minus[0], grad_minus[1], grad_minus[2]);
            rfb_ptr->evaluate_gradient(px, py, pz - h, grad_plus[0], grad_plus[1], grad_plus[2]);
        }
        for (int i = 0; i < 3; ++i) { // j corresponds to x, y, z
            // Compute second derivative d(g_i)/d(variable_j) using central difference
            H(i,j) = (grad_plus[i] - grad_minus[i]) / (2 * h);
        }
    }
    return H;
}

void VIPSSRidges::ComputePrincleCurvature(const Point& pt, std::shared_ptr<RBF_Core> rfb_ptr, 
    double& k1, double& k2, Vec& t1, Vec& t2)
{
    Vec grad = ComputeGradient(rfb_ptr, pt);
    double grad_len = arma::norm(grad);
    Vec normal = arma::normalise(grad);
    arma::mat identity_matrix = arma::eye(3,3);
    // Adjugate of gradient (for Weingarten map)
    arma::mat adj_grad =  grad_len * identity_matrix - grad * grad.t() / grad_len;
    arma::mat hessian = ComputeHessian(rfb_ptr, pt, 1e-10);

    // Weingarten map: W = - (H * adj_grad) / |grad|^3
    arma::mat W = -hessian * adj_grad / std::pow(grad_len, 3);

    // Project W onto tangent plane
    arma::mat proj = identity_matrix - normal * normal.t();
    arma::mat W_tangent = proj * W * proj;

}


// Compute principal curvatures and directions using the method from Monga et al.
PrincipleCurvature VIPSSRidges::ComputePrincipalCurvaturesMonga(const Vec& gradient, const arma::mat& hessian) {
    const auto& g = gradient;
    const auto& H = hessian;
    double g_norm = arma::norm(g);

    // PrincipleCurvature p_curvature; 

    // if (g_norm < 1e-16) {    
    //     return p_curvature;
    // }

    // Compute rotation matrix P to align g with the first basis vector
    double g1 = g(0), g2 = g(1), g3 = g(2);
    double gamma = std::sqrt(g1 * g1 + g2 * g2);
    double delta = g_norm;

    arma::mat P(3, 3);
    // if (gamma < 1e-16) {
    //     // Handle case where g1 = g2 = 0 (g is along z-axis)
    //     P << 0 << 1 << 0 << arma::endr
    //       << 0 << 0 << 1 << arma::endr
    //       << 1 << 0 << 0 << arma::endr;
    // } else {
    //     P << g1 / delta << g2 / gamma << g1 * g3 / (gamma * delta) << arma::endr
    //       << g2 / delta << -g1 / gamma << g2 * g3 / (gamma * delta) << arma::endr
    //       << g3 / delta << 0 << -gamma / delta << arma::endr;
    // }
    
    // Extract h and f (columns 2 and 3 of P, spanning the tangent plane)
    arma::vec h = {g2 / gamma, -g1 / gamma, 0};


    arma::vec f = {g1 * g3 / (gamma * delta), g2 * g3 / (gamma * delta), -gamma / delta};
    // arma::vec f = P.col(2);

    // std::cout << " g : " << h << std::endl;

    // Compute terms for principal curvatures
    double hHh = as_scalar(h.t() * H * h);
    double fHf = as_scalar(f.t() * H * f);
    double hHf = as_scalar(h.t() * H * f);

    // Compute principal curvatures
    double discriminant = std::sqrt((hHh - fHf) * (hHh - fHf) + 4 * hHf * hHf);
    double k1 = (hHh + fHf + discriminant) / (2 * g_norm); // Larger curvature (in magnitude)
    double k2 = (hHh + fHf - discriminant) / (2 * g_norm); // Smaller curvature

    // Compute principal directions
    // if (std::abs(hHf) > 1e-16) {
        double factor1 = (g_norm * k1 - hHh) / hHf;
        double factor2 = (g_norm * k2 - hHh) / hHf;
        arma::vec t1 = h + f * factor1;
        arma::vec t2 = h + f * factor2;
        t2 = arma::normalise(arma::cross(t1, g));
        // p_curvature.t1_ = normalise( p_curvature.t1_);
        // p_curvature.t2_ = normalise( p_curvature.t2_);
        // double sign_val = arma::dot(arma::cross(p_curvature.t2_ , p_curvature.t1_), g) ;
        // if(sign_val < 0)
        // {
        //     p_curvature.t1_ *= -1;
        // } 
    // } else {
    //     // If hHf is zero, h and f are already principal directions
    //     p_curvature.t1_ = f;
    //     p_curvature.t2_ = h;
    // }
    PrincipleCurvature p_curvature(k1, k2, t1, t2);

    return p_curvature;
}

void VIPSSRidges::ComputeThirdDerivatives(const Point& pt, std::shared_ptr<RBF_Core> hrfb_ptr, arma::cube& third_derivs, double h) 
{
    // Compute Hessians at nearby points
    // Point pt_xp = {pt[0] + h, pt[1], pt[2]};
    // arma::mat H_px = ComputeHessian(rfb_ptr, pt_xp, h);
    // Point pt_xm = {pt[0] - h, pt[1], pt[2]};
    // arma::mat H_mx = ComputeHessian(rfb_ptr, pt_xm, h);
    arma::mat hessian_xp = arma::zeros(3, 3);
    hrfb_ptr->EvaluateHessian(pt[0] + h, pt[1], pt[2], hessian_xp);
    arma::mat hessian_xn = arma::zeros(3, 3);
    hrfb_ptr->EvaluateHessian(pt[0] - h, pt[1], pt[2], hessian_xn);

    // Point pt_yp = {pt[0], pt[1] + h, pt[2]};
    // arma::mat H_py = ComputeHessian(rfb_ptr, pt_yp, h);
    // Point pt_ym = {pt[0], pt[1] - h, pt[2]};
    // arma::mat H_my = ComputeHessian(rfb_ptr, pt_ym, h);

    // Point pt_zp = {pt[0], pt[1], pt[2] + h};
    // arma::mat H_pz = ComputeHessian(rfb_ptr, pt_yp, h);
    // Point pt_zm = {pt[0], pt[1], pt[2] + h};
    // arma::mat H_mz = ComputeHessian(rfb_ptr, pt_ym, h);

    arma::mat hessian_yp = arma::zeros(3, 3);
    hrfb_ptr->EvaluateHessian(pt[0], pt[1] + h, pt[2], hessian_yp);
    arma::mat hessian_yn = arma::zeros(3, 3);
    hrfb_ptr->EvaluateHessian(pt[0], pt[1] - h, pt[2], hessian_yn);

    arma::mat hessian_zp = arma::zeros(3, 3);
    hrfb_ptr->EvaluateHessian(pt[0], pt[1], pt[2] + h, hessian_zp);
    arma::mat hessian_zn = arma::zeros(3, 3);
    hrfb_ptr->EvaluateHessian(pt[0], pt[1], pt[2] - h, hessian_zn);
 
    // Initialize cube (3x3x3 for H_x, H_y, H_z)
    third_derivs = arma::cube(3, 3, 3);
    // Compute derivatives using central differences
    // third_derivs.slice(0) = (H_px - H_mx) / (2 * h); // H_x = dH/dx
    // third_derivs.slice(1) = (H_py - H_my) / (2 * h); // H_y = dH/dy
    // third_derivs.slice(2) = (H_pz - H_mz) / (2 * h); // H_z = dH/dz

    third_derivs.slice(0) = (hessian_xp - hessian_xn) / (2 * h); // H_x = dH/dx
    third_derivs.slice(1) = (hessian_yp - hessian_yn) / (2 * h); // H_y = dH/dy
    third_derivs.slice(2) = (hessian_zp - hessian_zn) / (2 * h); // H_z = dH/dz
}

void VIPSSRidges::ComputeFourthDerivatives(const Point& pt, std::shared_ptr<RBF_Core> hrfb_ptr, std::vector<arma::cube>& fourth_derivs, double h) 
{
    auto third_derivs_xp = arma::cube(3, 3, 3);
    ComputeThirdDerivatives({pt[0] + h, pt[1], pt[2]}, hrfb_ptr, third_derivs_xp);
    auto third_derivs_xn = arma::cube(3, 3, 3);
    ComputeThirdDerivatives({pt[0] - h, pt[1], pt[2]}, hrfb_ptr, third_derivs_xn);

    auto third_derivs_yp = arma::cube(3, 3, 3);
    ComputeThirdDerivatives({pt[0], pt[1] + h, pt[2]}, hrfb_ptr, third_derivs_yp);
    auto third_derivs_yn = arma::cube(3, 3, 3);
    ComputeThirdDerivatives({pt[0], pt[1] - h, pt[2]}, hrfb_ptr, third_derivs_yn);

    auto third_derivs_zp = arma::cube(3, 3, 3);
    ComputeThirdDerivatives({pt[0], pt[1], pt[2] + h}, hrfb_ptr, third_derivs_zp);
    auto third_derivs_zn = arma::cube(3, 3, 3);
    ComputeThirdDerivatives({pt[0], pt[1], pt[2] - h}, hrfb_ptr, third_derivs_zn);
 
    arma::cube fourth_derivs_x = (third_derivs_xp - third_derivs_xn) / (2 * h); 
    arma::cube fourth_derivs_y = (third_derivs_yp - third_derivs_yn) / (2 * h);
    arma::cube fourth_derivs_z = (third_derivs_zp - third_derivs_zn) / (2 * h);
    fourth_derivs = {fourth_derivs_x, fourth_derivs_y, fourth_derivs_z};
}

double VIPSSRidges::ComputeCurvatureDerivative(const Point& pt, const Vec& normal, 
                                                const arma::mat& Hessian,
                                                const arma::cube& third_derivs,  
                                                std::shared_ptr<RBF_Core> rfb_ptr, 
                                                const Vec& t1)
{

    double g_norm = norm(normal);
    // Compute t1^T H_x t1, t1^T H_y t1, t1^T H_z t1 from the combined cube
    auto normalized_t = arma::normalise(t1); 
    arma::vec third_deriv_terms(3);
    for (int i = 0; i < 3; ++i) {
        third_deriv_terms(i) = arma::as_scalar(normalized_t.t() 
        * arma::mat(third_derivs.slice(i)) * normalized_t); // t1^T H_{x,y,z} t1
    }
    
    // Compute t1^T H t1 and t1^T H g
    double tHt = arma::as_scalar(normalized_t.t() * Hessian * normalized_t);
    double tHg = arma::as_scalar(normalized_t.t() * Hessian * normal);
    // std::cout << "pt " << pt[0] << " " << pt[1] << " " << pt[2] << 
    // " " << H << std::endl;
    // Compute derivative: (||g||^2 * t1^T [t1^T H_x t1, t1^T H_y t1, t1^T H_z t1] - (t1^T H t1)(t1^T H g)) / ||g||^3
    double numerator = g_norm * g_norm * arma::as_scalar(normalized_t.t() * third_deriv_terms) - tHt * tHg;
    double denominator = g_norm * g_norm * g_norm;
    return numerator / denominator;
}


double VIPSSRidges::ComputeCurvatureSecondDerivative(const Point& pt, const Vec& gradient, 
                                                const arma::mat& Hessian,
                                                const arma::cube& third_derivs, 
                                                const std::vector<arma::cube>& fourth_derivs,
                                                std::shared_ptr<RBF_Core> rfb_ptr, 
                                                const Vec& t1,
                                                const double k1, 
                                                const double e1)
{

    double g_norm = norm(gradient);
    // Compute t1^T H_x t1, t1^T H_y t1, t1^T H_z t1 from the combined cube
    auto normalized_t = arma::normalise(t1); 
    double sum1 = 0;
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            for(int l = 0; l < 3; ++l)
            {
                for(int m = 0; m < 3; ++m)
                {
                    sum1 += fourth_derivs[i](j,l,m) * t1[i] * t1[j] * t1[l] * t1[m];
                }
            }
        }
    }

    double sum2 = 0;
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            for(int l = 0; l < 3; ++l)
            {
                sum2 += third_derivs(i, j, l) * t1[i] * t1[j] * gradient[l] * 6 * k1;
            }
        }
    }

    double sum3 = 0;
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            sum3 += (Hessian(i,j) * t1[i] * gradient[j] * 4 * e1 
                   + Hessian(i,j) * gradient[i] * gradient[j] * 3 * k1 * k1);  
        }
    }

    double result = (sum1 + sum2 + sum3) / g_norm - 3 * k1 * k1 *k1;

    return result;
}

void ProjectMeshPtsToSurface()
{

}

bool VIPSSRidges::CalMeshPointsGradient(std::shared_ptr<RBF_Core> rfb_ptr, 
                                        const std::vector<Point> &points,
                                        std::vector<Vec> &gradients)
{
    // hrfb_ptr_ = rfb_ptr;
    size_t ptn = points.size();
    gradients.resize(ptn);
    // points.resize(ptn);
    for(int i = 0; i < ptn; ++i)
    {
        double pt[3] = {points[i][0], points[i][1], points[i][2]};
        double g[3];
        // std::cout << " start evaluate_gradient .... " << std::endl;
        double f_val = rfb_ptr->evaluate_gradient(pt[0], pt[1], pt[2], g[0], g[1], g[2]);
        // std::cout << " func value :  " << f_val << std::endl;
        double glen = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]); 
        // point_graidents_[i] =  {g[0]/glen, g[1]/glen, g[2]/glen};
        gradients[i] =  {g[0], g[1], g[2]};

        // std::cout << "pt " << pt[0] << " " << pt[1] << " " << pt[2] << 
        // "  g: " << g[0] << " " << g[1] << " " << g[2] << std::endl;
    }
    // double pt[3] = {0., 0.587785, 0.323607};
    // double g[3];
    // rfb_ptr->evaluate_gradient(pt[0], pt[1], pt[2], g[0], g[1], g[2]);
    //   std::cout << "pt " << pt[0] << " " << pt[1] << " " << pt[2] << 
    //     "  g: " << g[0] << " " << g[1] << " " << g[2] << std::endl;

    return true;
}

bool VIPSSRidges::CalMeshPointsCurvature(std::shared_ptr<RBF_Core> hrfb_ptr,
                                        const std::vector<Point>& points,
                                        const std::vector<Vec>& gradients,
                                        std::vector<PrincipleCurvature>& pt_curvatures)
{
    size_t ptn = points.size();
    pt_curvatures.resize(ptn);
    
    for(int i = 0; i < ptn; ++i)
    {
        const auto& gradient = gradients[i];
        const auto& pt = points[i];
        // arma::mat hessian = ComputeHessian(rfb_ptr, pt, 1e-10);
        // std::cout << "hessian numeric : " << hessian1 << std::endl;
        arma::mat hessian = arma::zeros(3, 3);
        hrfb_ptr->EvaluateHessian(pt[0], pt[1], pt[2], hessian);
        // std::cout << "hessian analytic : " << hessian << std::endl;
        pt_curvatures[i] = ComputePrincipalCurvaturesMonga(gradient, hessian);
        arma::cube third_derivs;
        ComputeThirdDerivatives(pt, hrfb_ptr, third_derivs, 1e-8);

        // std::vector<arma::cube> fourth_derivs;
        // ComputeFourthDerivatives(pt, hrfb_ptr, fourth_derivs, 1e-8);

        // arma::mat hessian = arma::zeros(3,3);
        // hrfb_ptr->EvaluateHessian(pt[0], pt[1], pt[2], hessian);
        arma::vec eigval;
        arma::mat eigvec;
        arma::mat H_sym = 0.5 * (hessian + hessian.t());
        arma::eig_sym(eigval, eigvec, H_sym);

        arma::vec eigval_abs = arma::abs(eigval);
        size_t max_id = arma::index_max(eigval_abs);
        arma::vec max_vec = eigvec.col(max_id);

        // std::cout << "pt " << pt[0] << " " << pt[1] << " " << pt[2] 
        // << " third : " << third_derivs << std::endl;
        // std::cout << "pt " << pt[0] << " " << pt[1] << " " << pt[2] << " " << hessian << std::endl;
        pt_curvatures[i].emax_ = ComputeCurvatureDerivative(
            pt, gradient, hessian, third_derivs, hrfb_ptr, pt_curvatures[i].tmax_);
        pt_curvatures[i].emin_ = ComputeCurvatureDerivative(
            pt, gradient, hessian, third_derivs, hrfb_ptr, pt_curvatures[i].tmin_);

        pt_curvatures[i].tmax_ = max_vec;

        // if(0)
        // {
        //     pt_curvatures[i].demax_ = ComputeCurvatureSecondDerivative(pt, gradient, hessian, third_derivs, 
        //         fourth_derivs, hrfb_ptr, pt_curvatures[i].tmax_, pt_curvatures[i].kmax_, pt_curvatures[i].emax_);
        //     pt_curvatures[i].demin_ = ComputeCurvatureSecondDerivative(pt, gradient, hessian, third_derivs, 
        //         fourth_derivs, hrfb_ptr, pt_curvatures[i].tmin_, pt_curvatures[i].kmin_, pt_curvatures[i].emin_);
        // }
        
        // pt_curvatures[i].UpdateMax();
    }
    return true;
}

void VIPSSRidges::SaveMeshCurvaturesVisualResults(const std::string& out_dir)
{
    std::vector<double> k1_vecs;
    std::vector<double> k2_vecs;
    std::vector<double> e1_vecs;
    std::vector<double> e2_vecs;
    std::vector<double> e12_vecs;
    std::vector<double> de1_vecs;
    std::vector<double> de2_vecs;
    std::vector<Vec> d1_vecs;
    std::vector<Vec> d2_vecs;
    for(int i = 0; i < mesh_points_.size(); ++i)
    {
        k1_vecs.emplace_back(mesh_pt_curvatures_[i].kmax_);
        k2_vecs.emplace_back(mesh_pt_curvatures_[i].kmin_);
        d1_vecs.emplace_back(mesh_pt_curvatures_[i].tmax_);
        d2_vecs.emplace_back(mesh_pt_curvatures_[i].tmin_);
        e1_vecs.emplace_back(mesh_pt_curvatures_[i].emax_);
        e2_vecs.emplace_back(mesh_pt_curvatures_[i].emin_);
        // de1_vecs.emplace_back(mesh_pt_curvatures_[i].demax_);
        // de2_vecs.emplace_back(mesh_pt_curvatures_[i].demin_);
        double e12 = mesh_pt_curvatures_[i].emax_ * mesh_pt_curvatures_[i].emin_;
        e12_vecs.emplace_back(e12);
    }
    std::string k1_mesh_path = out_dir + "k1.ply";
    SaveMeshWithQualityToPly(k1_mesh_path, mesh_points_, k1_vecs, mesh_faces_);
    std::string k2_mesh_path = out_dir + "k2.ply";
    SaveMeshWithQualityToPly(k2_mesh_path, mesh_points_, k2_vecs, mesh_faces_);
    std::string e1_mesh_path = out_dir + "e1.ply";
    SaveMeshWithQualityToPly(e1_mesh_path, mesh_points_, e1_vecs, mesh_faces_);
    std::string e2_mesh_path = out_dir + "e2.ply";
    SaveMeshWithQualityToPly(e2_mesh_path, mesh_points_, e2_vecs, mesh_faces_);
    // std::string de1_mesh_path = out_dir + "de1.ply";
    // SaveMeshWithQualityToPly(de1_mesh_path, mesh_points_, de1_vecs, mesh_faces_);
    // std::string de2_mesh_path = out_dir + "de2.ply";
    // SaveMeshWithQualityToPly(de2_mesh_path, mesh_points_, de2_vecs, mesh_faces_);

    std::string e12_mesh_path = out_dir + "e12.ply";
    SaveMeshWithQualityToPly(e12_mesh_path, mesh_points_, e12_vecs, mesh_faces_);

    std::string ke_values_path = out_dir + "k1k2e1e2.csv";
    std::vector<std::vector<double>> ke_values = {k1_vecs, k2_vecs, e1_vecs, e2_vecs};
    WriteVectorValsToCSV(ke_values_path, ke_values);

    std::string e_ridge_pt_path = out_dir + "e_ridge_pts.ply";
    std::vector<TriFace> non_faces;
    SaveMeshWithQualityToPly(e_ridge_pt_path, edge_ridge_pts_, edge_ridge_pts_curvature_, non_faces);
    std::string e_valley_pt_path = out_dir + "e_valley_pts.ply";
    SaveMeshWithQualityToPly(e_valley_pt_path, edge_valley_pts_, edge_valley_pts_curvature_, non_faces);

    std::string e_gaussian_pt_path = out_dir + "e_gaussian_pts.ply";
    SaveMeshWithQualityToPly(e_gaussian_pt_path, edge_gaussian_pts_, edge_gaussian_curvature_, non_faces);



    std::string d1_mesh_path = out_dir + "d1.xyz";
    SavePointsNormalToXYZ(d1_mesh_path, mesh_points_, d1_vecs);
    std::string d2_mesh_path = out_dir + "d2.xyz";
    SavePointsNormalToXYZ(d2_mesh_path, mesh_points_, d2_vecs);

    std::string edge_vals_path = out_dir + "edge_ridge_extract_vals.txt";
    std::cout << "----- edge_curv_values_string size " << edge_curv_values_string.size() << std::endl;
    SaveStringValsToText(edge_vals_path, edge_curv_values_string);

    std::string edge_sample_vals_path = out_dir + "edge_ridge_sample_vals.txt";
    std::ofstream sample_file(edge_sample_vals_path);
    if( sample_file.is_open())
    {
        
        for(const auto& edge_curv_data : edge_sample_curv_dataset)
        {
            
            for(int sp_id = 0; sp_id < edge_curv_data.size(); ++sp_id)
            {
                const auto& pt_data = edge_curv_data[sp_id];
                sample_file << "{" << pt_data.pt_[0] << "," <<  pt_data.pt_[1] << "," << pt_data.pt_[2]  << ","
                << pt_data.tmax_[0] << "," << pt_data.tmax_[1] << "," << pt_data.tmax_[2] << "," 
                << pt_data.kmax_ << "," << pt_data.emax_  << "}"; 
                if(sp_id != edge_curv_data.size()-1)
                {
                    sample_file << ","; 
                }
            }
            sample_file<< std::endl;
        }
        
        sample_file.close();
    }
}

void VIPSSRidges::CalSinglePointCurvatureData(const Point& pt, PrincipleCurvature& curvature)
{
    std::vector<Point> points = {pt};
    std::vector<Vec> gradients;
    CalMeshPointsGradient(hrfb_ptr_, points, gradients);
    std::vector<PrincipleCurvature> pts_curvature;
    CalMeshPointsCurvature(hrfb_ptr_, points,gradients,pts_curvature);
    curvature = pts_curvature[0];
}

bool VIPSSRidges::CalMeshPointsGradientAndEigenVecs(std::shared_ptr<RBF_Core> rfb_ptr)
{
    hrfb_ptr_ = rfb_ptr;
    size_t ptn = mesh_points_.size();
    point_eig_vecs_.resize(ptn);
    point_eig_vals_.resize(ptn);
    point_graidents_.resize(ptn);
    std::vector<double> points;
    std::vector<double> eig_vecs;
    std::vector<double> gredients;
    for(int i = 0; i < ptn; ++i)
    {
        double pt[3] = {mesh_points_[i][0], mesh_points_[i][1], mesh_points_[i][2]};

        double g[3];
        double f_val = rfb_ptr->evaluate_gradient(pt[0], pt[1], pt[2], g[0], g[1], g[2]) ;
        Vec gradient = {g[0], g[1], g[2]};
        point_graidents_[i] = arma::normalise(gradient);
        arma::mat hessian = arma::zeros(3,3);
        rfb_ptr->EvaluateHessian(pt[0], pt[1], pt[2], hessian);
        arma::vec eigval;
        arma::mat eigvec;
        arma::mat H_sym = 0.5 * (hessian + hessian.t());
        arma::eig_sym(eigval, eigvec, H_sym);
        // TransformEclips(eigval, eigvec, mesh_points_[i]);
        arma::vec eigval_abs = arma::abs(eigval);
        // arma::uvec sort_ids = arma::sort_index(eigval);
        // size_t max_id = arma::index_max(arma::abs(eigval));
        size_t max_id = arma::index_max(eigval_abs);
        // point_eig_vals_[i] = eigval_abs[max_id];
        // size_t max_id = sort_ids[2];
        Vec max_vec = eigvec.col(max_id);
        point_eig_vecs_[i] = arma::normalise(max_vec) * eigval_abs[max_id];
        eig_vecs.push_back(point_eig_vecs_[i][0]);
        eig_vecs.push_back(point_eig_vecs_[i][1]);
        eig_vecs.push_back(point_eig_vecs_[i][2]);
    }
    return true;
}



void VIPSSRidges::GetEdges()
{
    size_t face_num = mesh_faces_.size();
    size_t eid = 0;
    for(int i = 0; i < face_num; ++i)
    {
        const auto& pids = mesh_faces_[i];
        // std::cout << " face v ids :   " << pids[0] << " " << pids[1] << " " << pids[2] << std::endl;
        string token_ab = CalEdgeToken(pids[0], pids[1]);
        if(edge_id_map_.find(token_ab) == edge_id_map_.end())
        {
            edge_id_map_[token_ab] = eid;
            eid ++;
            edges_.push_back({pids[0], pids[1]});
        }
        string token_bc = CalEdgeToken(pids[1], pids[2]);
        if(edge_id_map_.find(token_bc) == edge_id_map_.end())
        {
            edge_id_map_[token_bc] = eid;
            eid ++;
            edges_.push_back({pids[1], pids[2]});
        }
        string token_ca = CalEdgeToken(pids[2], pids[0]);
        if(edge_id_map_.find(token_ca) == edge_id_map_.end())
        {
            edge_id_map_[token_ca] = eid;
            eid ++;
            edges_.push_back({pids[2], pids[0]});
        }
    }
    std::cout << " total edge processed : " << eid << std::endl;
}

bool VIPSSRidges::CalculateCrestPoints(const Point& pa, const Point& pb, 
                        const PrincipleCurvature& ca, const PrincipleCurvature& cb,
                        int& riges_sign, int& valley_sign, int& gaussain_sign)
{
    double ka_max = ca.kmax_; double ka_min = ca.kmin_;
    double kb_max = cb.kmax_; double kb_min = cb.kmin_;

    double ea_max = ca.emax_; double ea_min = ca.emin_;
    double eb_max = cb.emax_; double eb_min = cb.emin_;

    double eg_a = ea_max * ea_min; 
    double eg_b = eb_max * eb_min;

    auto ta_max = ca.tmax_;  auto ta_min = ca.tmin_;
    auto tb_max = cb.tmax_;  auto tb_min = cb.tmin_;

    arma::vec pba = {pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};

    // cal max for ridges 
    if(arma::dot(ta_max, tb_max) < 0)
    {
        tb_max *= -1.0;
        eb_max *= -1.0;
    } 
    // condition 3 from paper "Ridge-Valley Lines on Meshes via Implicit Surface Fitting"
    if(ka_max > abs(ka_min) && kb_max > abs(kb_min) )
    // if(ka_max > 0 && kb_max > 0 )
    {
        if(ea_max * eb_max < 0)
        {
            // if(ea_max * (arma::dot(pba, ta_max)) > 0)
            {
                double abs_sum = abs(ea_max)  + abs(eb_max);
                double px = (abs(eb_max) * pa[0] + abs(ea_max) * pb[0])/abs_sum;
                double py = (abs(eb_max) * pa[1] + abs(ea_max) * pb[1])/abs_sum;
                double pz = (abs(eb_max) * pa[2] + abs(ea_max) * pb[2])/abs_sum;
                double int_curvature = (abs(eb_max) * ka_max+ abs(ea_max) * kb_max)/abs_sum;
                edge_ridge_pts_.push_back({px, py, pz});
                edge_ridge_pts_curvature_.push_back(int_curvature);
                riges_sign = interp_ridge_p_id_;
                interp_ridge_p_id_ ++;
            }
        } 
    }

    // cal min for valley 
    if(arma::dot(ta_min, tb_min) < 0)
    {
        tb_min *= -1.0;
        eb_min *= -1.0;
    } 
    // condition 3 from paper "Ridge-Valley Lines on Meshes via Implicit Surface Fitting"
    if(abs(ka_min) > abs(ka_max) && abs(kb_min) > abs(kb_max) )
    // if(ka_min < 0 && kb_min< 0 )
    {
        if(ea_min * eb_min < 0)
        {
            // if(ea_min * (arma::dot(pba, ta_min)) > 0)
            {
                // double t = (- eb_min) / (ea_min - eb_min);
                // double px = pa[0] + t * (pb[0] - pa[0]);
                // double py = pa[1] + t * (pb[1] - pa[1]);
                // double pz = pa[2] + t * (pb[2] - pa[2]);
                double abs_sum = abs(ea_min)  + abs(eb_min);
                double px = (abs(eb_min) * pa[0] + abs(ea_min) * pb[0])/abs_sum;
                double py = (abs(eb_min) * pa[1] + abs(ea_min) * pb[1])/abs_sum;
                double pz = (abs(eb_min) * pa[2] + abs(ea_min) * pb[2])/abs_sum;

                double int_curvature = (abs(eb_min) * ka_min+ abs(ea_min) * kb_min) 
                                /(abs(ea_min)  + abs(eb_min));
                edge_valley_pts_.push_back({px, py, pz});
                edge_valley_pts_curvature_.push_back(int_curvature);
                valley_sign = interp_valley_p_id_;
                interp_valley_p_id_ ++;
            }
        } 
    }

    if(eg_a * eg_b < 0)
    {
        {
            double abs_sum = abs(eg_a)  + abs(eg_b);
            double px = (abs(eg_b) * pa[0] + abs(eg_a) * pb[0])/abs_sum;
            double py = (abs(eg_b) * pa[1] + abs(eg_a) * pb[1])/abs_sum;
            double pz = (abs(eg_b) * pa[2] + abs(eg_a) * pb[2])/abs_sum;
            double int_curvature_max = (abs(eg_b) * ka_max+ abs(eg_a) * kb_max)/abs_sum;
            double int_curvature_min = (abs(eg_b) * ka_min+ abs(eg_a) * kb_min)/abs_sum;

            edge_gaussian_pts_.push_back({px, py, pz});
            edge_gaussian_curvature_.push_back(int_curvature_max * int_curvature_min);
            gaussain_sign = interp_gaussian_p_id_;
            interp_gaussian_p_id_ ++;
        }
    } 
    return true;
}

double get_cubic_root(double val1, double val2, double g1, double g2) {
  assert(val1 * val2 < 0);

  // make sure val1 < 0 and val2 > 0
  if (val1 > 0) {
    val1 = -val1;
    val2 = -val2;
    g1 = -g1;
    g2 = -g2;
  }

  // compute the cubic function f(x) = a*x^3 + b*x^2 + c*x + d
  const double a = g1 + g2 + 2 * (val1 - val2);
  const double b = 3 * (val2 - val1) - 2 * g1 - g2;
  const double c = g1;
  const double d = val1;

  // initial guess: the linear root
  double x = val1 / (val1 - val2);

  // root finding: combine Halley's method and bisect method
  // mostly a bisect method, but first find the next guess using Hally's method,
  // if the guess doesn't lie in the sign-changing interval, use the midpoint of that interval
  // terminate when the change in x is small
  double xlo = 0;
  double xhi = 1;
  constexpr double x_tol = 1e-4;
  while (true) {
    double f = d + x * (c + x * (b + x * a));
    if (f == 0) {
      break;
    }
    if (f < 0) {
      xlo = x;
    } else {
      xhi = x;
    }
    // f'(x) = 3*a*x^2 + 2*b*x + c
    // f''(x) = 6*a*x + 2*b
    double df = c + x * (2 * b + x * 3 * a);
    double ddf = 2 * b + x * 6 * a;
    double dx = 2 * f * df / (2 * df * df - f * ddf);
    double x_new = x - dx;
    if (x_new <= xlo || x_new >= xhi) {
      x_new = 0.5 * (xlo + xhi);
    }
    x = x_new;
    if (std::abs(dx) < x_tol) {
      break;
    }
  }

  return x;
}

void InterpolateCrestPoint(const std::array<double,3>& pa, const std::array<double,3>& pb, 
            const double e_a, const double e_b, std::array<double,3>& interp)
{
    double abs_sum = abs(e_a)  + abs(e_b);
    double px = (abs(e_b) * pa[0] + abs(e_a) * pb[0])/abs_sum;
    double py = (abs(e_b) * pa[1] + abs(e_a) * pb[1])/abs_sum;
    double pz = (abs(e_b) * pa[2] + abs(e_a) * pb[2])/abs_sum;
    interp = {px, py, pz};
}

// void InterpolateCrestPointQuadratic(const std::array<double,3>& pa, const std::array<double,3>& pb, 
//     const arma::vec& t_a,const arma::vec& t_b,
//     const double k_a, const double k_b, 
//     const double e_a, const double e_b, std::array<double,3>& interp)
// {
//     // std::cout << " curv data : " << k_a << " " << k_b << " . e : " << e_a << " " << e_b ; 
//     arma::vec ab = {pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};
//     auto ave_t = arma::normalise(t_a + t_b);
//     double x_b = arma::dot(ab, ave_t);
//     // arma::mat mat_A = arma::mat::zeros(4,3);
//     arma::mat mat_A = {{0, 1, 0}, {2 * x_b, 1, 0}, {0, 0, 1}, {x_b * x_b, x_b, 1}};
//     arma::vec4 vec_B = {e_a, e_b, k_a, k_b};
//     arma::vec quadratic_abc = arma::solve(mat_A, vec_B, arma::solve_opts::fast);
//     // std::cout << " x b : " << x_b << " quadratic inter t : " << - quadratic_abc[1] / quadratic_abc[0] * 0.5 << std::endl;
//     double x_t = abs( quadratic_abc[1] / quadratic_abc[0] * 0.5);
//     double inter_t = x_t / abs(x_b);
//     double px = pa[0] + inter_t * ab[0];
//     double py = pa[1] + inter_t * ab[1];
//     double pz = pa[2] + inter_t * ab[2];
//     interp = {px, py, pz};
// }

// p1(6t - 6t^2) + len m0 (1- 4t +3t^2) + len m1 (-2t + 3t^2) + p0(-6t + 6t^2); 
// 6(p0 - p1)t^2 + 3 len (m0 + m1)t^2 + 6(p1 - p0)t +  len(-4m0 -2m1)t + len m0
// a = 6(p0 - p1) + 3 len (m0 + m1)
// b = 6(p1 - p0) + len(-4m0 - 2m1)
// c = len m0
double SolveCubicHermite(const double k0, const double k1, 
                        const double e0, const double e1,
                        const double len)
{
    double a = 6.0 * (k0 - k1) + 3.0 * len * (k0 + k1);
    double b = 6.0 * (k1 - k0) + len *( -4* k0 - 2.0 * k1);
    double c = len * k0;
    double d = sqrt(b*b - 4 * a * c);
    double x1 =  (-b + d)/(2 * a);
    double x2 =  (-b - d)/(2 * a);
    if(x1 >= 0 && x1 <= 1)
    {
        return x1;
    }
    if(x2 >= 0 && x2 <= 1)
    {
        return x2;
    }
    return 0;
}

void InterpolateCrestPointQuadratic(const std::array<double,3>& pa, const std::array<double,3>& pb, 
    const arma::vec& t_a,const arma::vec& t_b,
    const double k_a, const double k_b, 
    const double e_a, const double e_b, 
    std::array<double,3>& interp)
{
    // std::cout << " curv data : " << k_a << " " << k_b << " . e : " << e_a << " " << e_b ; 
    arma::vec ab = {pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};
    auto ave_t = arma::normalise(t_a + t_b);
    // auto ave_t = arma::normalise(t_a);
    double len = abs(arma::dot(ab, ave_t));
    if(k_a <= k_b)
    {
        double xt = SolveCubicHermite(k_a, k_b, e_a, e_b, len);
        double px = pa[0] + xt * ab[0];
        double py = pa[1] + xt * ab[1];
        double pz = pa[2] + xt * ab[2];
        interp = {px, py, pz};
    } else {
        double xt = SolveCubicHermite(k_b, k_a, e_b, e_a, len);
        double px = pa[0] + (1 - xt) * ab[0];
        double py = pa[1] + (1 - xt) * ab[1];
        double pz = pa[2] + (1 - xt) * ab[2];
        interp = {px, py, pz};
    }
}

void get_cubic_root(const std::array<double, 3>& p1, const std::array<double, 3>& p2,
                    const double val1, const double val2, const std::array<double, 3>& grad1,
                    const std::array<double, 3>& grad2,
                    std::array<double, 3>& interp) {
    // require val1 and val2 to have different signs
    assert(val1 * val2 < 0);
    
    // directional derivative
    const std::array<double, 3> p1p2 = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
    // since the interval is scaled to (0,1), we don't normalize by dividing by the length of p1p2
    const double g1 = (grad1[0] * p1p2[0] + grad1[1] * p1p2[1] + grad1[2] * p1p2[2]);
    const double g2 = (grad2[0] * p1p2[0] + grad2[1] * p1p2[1] + grad2[2] * p1p2[2]);
    
    // compute the root
    const double x = get_cubic_root(val1, val2, g1, g2);
    interp[0] = p1[0] + x * p1p2[0];
    interp[1] = p1[1] + x * p1p2[1];
    interp[2] = p1[2] + x * p1p2[2];
}

bool VIPSSRidges::CalculateCrestPointsSingle(const Point& pa, const Point& pb, 
                        const PrincipleCurvature& ca, const PrincipleCurvature& cb,
                        int& edge_emax_sign, Point& inter_pa, double& inter_cur_a,
                        int& edge_emin_sign, Point& inter_pb, double& inter_cur_b)
{
    double ka_max = ca.kmax_; double ka_min = ca.kmin_;
    double kb_max = cb.kmax_; double kb_min = cb.kmin_;

    double ea_max = ca.emax_; double ea_min = ca.emin_;
    double eb_max = cb.emax_; double eb_min = cb.emin_;

    // double eg_a = ea_max * ea_min; 
    // double eg_b = eb_max * eb_min;

    auto ta_max = ca.tmax_;  auto ta_min = ca.tmin_;
    auto tb_max = cb.tmax_;  auto tb_min = cb.tmin_;

    arma::vec pab = {pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};

    // cal max for ridges 
    if(arma::dot(ta_max, tb_max) < 0)
    {
        tb_max *= -1.0;
        eb_max *= -1.0;
    } 
    // condition 3 from paper "Ridge-Valley Lines on Meshes via Implicit Surface Fitting"
    // if(ka_max > abs(ka_min) || kb_max > abs(kb_min) )
    if(ka_max + ka_min > 0 || kb_max + kb_min > 0 )
    // if( (ka_max < 0 && abs(ka_max) > abs(ka_min)) || (kb_max < 0 && abs(ka_max) > abs(ka_min) ) )
    // if( ka_max < 0  || kb_max < 0)
    {
        if(ea_max * eb_max < 0 && (ca.emax_prime_ <= 0 || cb.emax_prime_ <= 0))
        // if(ea_max * eb_max < 0 )
        {
            
            if(ea_max * (arma::dot(pab, ta_max)) > 0)
            {
                edge_emax_sign = 1;
                double abs_sum = abs(ea_max)  + abs(eb_max);
                
                // InterpolateCrestPointQuadratic(pa, pb, ta_max, tb_max, ka_max, kb_max, ea_max, eb_max, inter_pa);
                InterpolateCrestPoint(pa, pb, ea_max, eb_max, inter_pa);
                inter_cur_a = (abs(eb_max) * ka_max+ abs(ea_max) * kb_max)/abs_sum;
               
            }
        } 
    }

    // cal min for valley 
    if(arma::dot(ta_min, tb_min) < 0)
    {
        tb_min *= -1.0;
        eb_min *= -1.0;
    } 
    // condition 3 from paper "Ridge-Valley Lines on Meshes via Implicit Surface Fitting"
    // if(abs(ka_min) > abs(ka_max) || abs(kb_min) > abs(kb_max) )
    // if(ka_min > 0  || kb_min > 0)
    if(ka_max + ka_min < 0 || kb_max + kb_min < 0 )
    {
        if(ea_min * eb_min < 0  && (ca.emin_prime_ >= 0 || cb.emin_prime_ >= 0))
        // if(ea_min * eb_min < 0)
        {
            
            if(ea_min * (arma::dot(pab, ta_min)) < 0)
            {
                edge_emin_sign = 1;
                double abs_sum = abs(ea_min)  + abs(eb_min);
                inter_cur_b = (abs(eb_min) * ka_min+ abs(ea_min) * kb_min) 
                                /(abs(ea_min)  + abs(eb_min));
                // InterpolateCrestPointQuadratic(pa, pb, ta_min, tb_min, ka_min, kb_min, ea_min, eb_min, inter_pb);
                InterpolateCrestPoint(pa, pb, ea_min, eb_min, inter_pb);
            }
        } 
    }
    return true;
}

arma::mat33 QuadraticA = {{0, 0, 1}, {0.5* 0.5, 0.5, 1 }, {1, 1, 1}};
arma::mat33 QuadraticAInverse = arma::inv(QuadraticA);

double InterpolateQuadratic(double ea, double em, double eb)
{
  arma::vec3 B = {ea, em, eb};
//   arma::mat33 A = {{0, 0, 1}, {0.5* 0.5, 0.5, 1 }, {1, 1, 1}};
//   arma::vec3 X = arma::solve(A, B);
  arma::vec3 X = QuadraticAInverse * B;
  double delt = X[1] * X[1] - 4 * X[0] *X[2];
  // if(delt < 0) return 0;
  
  double xroot1 = (-X[1] + sqrt(delt))/(2 * X[0]);
  double xroot2 = (-X[1] - sqrt(delt))/(2 * X[0]);
  double root = xroot1 <= 1 && xroot1 >= 0 ? xroot1 : xroot2;

//   std::cout << " B " << B << " , X : " << X << " root : " << root <<  std::endl;
  return root;
} 


bool VIPSSRidges::CalculateCrestPointsSingleQuadratic(const Point& pa, const Point& pb, 
                        const PrincipleCurvature& ca, const PrincipleCurvature& cb,
                        const PrincipleCurvature& c_mid,
                        int& edge_emax_sign, Point& inter_pa, double& inter_cur_a,
                        int& edge_emin_sign, Point& inter_pb, double& inter_cur_b)
{
    double ka_max = ca.kmax_; double ka_min = ca.kmin_;
    double kb_max = cb.kmax_; double kb_min = cb.kmin_;

    double ea_max = ca.emax_; double ea_min = ca.emin_;
    double eb_max = cb.emax_; double eb_min = cb.emin_;

    double eg_a = ea_max * ea_min; 
    double eg_b = eb_max * eb_min;

    auto ta_max = ca.tmax_;  auto ta_min = ca.tmin_;
    auto tb_max = cb.tmax_;  auto tb_min = cb.tmin_;

    double emax_prime_a = ca.emax_prime_;

    arma::vec pab = {pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};

    // cal max for ridges 
    if(arma::dot(ta_max, tb_max) < 0)
    {
        tb_max *= -1.0;
        eb_max *= -1.0;
    } 
    // condition 3 from paper "Ridge-Valley Lines on Meshes via Implicit Surface Fitting"
    if(ka_max + ka_min > 0 || kb_max + kb_min > 0 )
    // if(ka_max < 0  || kb_max < 0)
    // if( (ka_max < 0 && abs(ka_max) > abs(ka_min)) || (kb_max < 0 && abs(ka_max) > abs(ka_min) ) )
    {
        if(ea_max * eb_max < 0 && (ca.emax_prime_ <= 0 && cb.emax_prime_ <= 0))
        // if(ea_max * eb_max < 0 )
        {
            
            if(ea_max * (arma::dot(pab, ta_max)) > 0)
            {
                edge_emax_sign = 1;
                double abs_sum = abs(ea_max)  + abs(eb_max);
                double e_mid = arma::dot(ta_max, c_mid.tmax_) < 0 ? - c_mid.emax_ : c_mid.emax_;
                double t = InterpolateQuadratic(ea_max, e_mid, eb_max);
                double interp_px = pa[0] + (pb[0] - pa[0]) * t;
                double interp_py = pa[1] + (pb[1] - pa[1]) * t;
                double interp_pz = pa[2] + (pb[2] - pa[2]) * t;
                inter_pa = {interp_px, interp_py, interp_pz};
                inter_cur_a = (abs(eb_max) * ka_max+ abs(ea_max) * kb_max)/abs_sum;
            }
        } 
    }

    // cal min for valley 
    if(arma::dot(ta_min, tb_min) < 0)
    {
        tb_min *= -1.0;
        eb_min *= -1.0;
    } 
    // condition 3 from paper "Ridge-Valley Lines on Meshes via Implicit Surface Fitting"
    // if(abs(ka_min) > abs(ka_max) || abs(kb_min) > abs(kb_max) )
    if(ka_min + ka_max < 0 || kb_min + kb_max < 0 )
    // if(ka_min > 0  || kb_min > 0)
    {
        if(ea_min * eb_min < 0 && (ca.emin_prime_ >= 0 || cb.emin_prime_ >= 0))
        // if(ea_min * eb_min < 0)
        {
            
            if(ea_min * (arma::dot(pab, ta_min)) < 0)
            {
                edge_emin_sign = 1;
                double abs_sum = abs(ea_min)  + abs(eb_min);
                inter_cur_b = (abs(eb_min) * ka_min+ abs(ea_min) * kb_min) 
                                /(abs(ea_min)  + abs(eb_min));
                // InterpolateCrestPointQuadratic(pa, pb, ta_min, tb_min, ka_min, kb_min, ea_min, eb_min, inter_pb);
                // InterpolateCrestPoint(pa, pb, ea_min, eb_min, inter_pb);
                double e_mid_min = arma::dot(ta_min, c_mid.tmin_) < 0 ? - c_mid.emin_ : c_mid.emin_;
                double t = InterpolateQuadratic(ea_min, e_mid_min, eb_min);
                double interp_px = pa[0] + (pb[0] - pa[0]) * t;
                double interp_py = pa[1] + (pb[1] - pa[1]) * t;
                double interp_pz = pa[2] + (pb[2] - pa[2]) * t;
                inter_pb = {interp_px, interp_py, interp_pz};
            }
        } 
    }
    return true;
}



bool VIPSSRidges::CalculateCrestPointsSingleWithGrad(const Point& pa, const Point& pb, 
                        const PrincipleCurvature& ca, const PrincipleCurvature& cb,
                        int& edge_emax_sign, Point& inter_pa, double& inter_cur_a,
                        int& edge_emin_sign, Point& inter_pb, double& inter_cur_b)
{
    double ka_max = ca.kmax_; double ka_min = ca.kmin_;
    double kb_max = cb.kmax_; double kb_min = cb.kmin_;

    double ea_max = ca.emax_; double ea_min = ca.emin_;
    double eb_max = cb.emax_; double eb_min = cb.emin_;

    double eg_a = ea_max * ea_min; 
    double eg_b = eb_max * eb_min;

    auto ta_max = ca.tmax_;  auto ta_min = ca.tmin_;
    auto tb_max = cb.tmax_;  auto tb_min = cb.tmin_;
    auto de1_a  = ca.de1_;   auto de2_a  = ca.de2_;
    auto de1_b  = cb.de1_;   auto de2_b  = cb.de2_;

    arma::vec pba = {pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};

    // cal max for ridges 
    if(arma::dot(ta_max, tb_max) < 0)
    {
        tb_max *= -1.0;
        eb_max *= -1.0;
        de1_b *= -1.0;
    } 
    // condition 3 from paper "Ridge-Valley Lines on Meshes via Implicit Surface Fitting"
    if(ka_max > abs(ka_min) || kb_max > abs(kb_min) )
    {
        if(ea_max * eb_max < 0)
        {
            edge_emax_sign = 1;
            // if(ea_max * (arma::dot(pba, ta_max)) > 0)
            {
                double abs_sum = abs(ea_max)  + abs(eb_max);
                // double px = (abs(eb_max) * pa[0] + abs(ea_max) * pb[0])/abs_sum;
                // double py = (abs(eb_max) * pa[1] + abs(ea_max) * pb[1])/abs_sum;
                // double pz = (abs(eb_max) * pa[2] + abs(ea_max) * pb[2])/abs_sum;
                // InterpolateCrestPoint(pa, pb, ea_max, eb_max, inter_pa);
                get_cubic_root(pa, pb, ea_max, eb_max, {de1_a[0],de1_a[1],de1_a[2]}, {de1_b[0], de1_b[1], de1_b[2]}, inter_pa);
                inter_cur_a = (abs(eb_max) * ka_max+ abs(ea_max) * kb_max)/abs_sum;
                Point new_pa = pa;
                Point new_pb = pb;
                // std::cout << " inter p coord : " << inter_pa[0] << " " << inter_pa[1] << " " << inter_pa[2] << std::endl;
                int loop_num = 0;
                for(int loop_id = 0; loop_id < loop_num; ++loop_id)
                {
                    PrincipleCurvature intera_curva;
                    CalSinglePointCurvatureData(inter_pa, intera_curva);
                    auto t_interp_max = intera_curva.tmax_;
                    auto e_interp_max = intera_curva.emax_;
                    if(arma::dot(ta_max, t_interp_max) < 0)
                    {
                        t_interp_max *= -1.0;
                        e_interp_max *= -1.0;

                    } 
                    Point new_interp_p;
                    if(ea_max * e_interp_max <= 0)
                    {
                        InterpolateCrestPoint(new_pa, inter_pa, ea_max, e_interp_max, new_interp_p);
                        // get_cubic_root(pa, pb, ea_max, eb_max, de1_a, de1_b, inter_pa);
                        eb_max = e_interp_max;
                        new_pb = new_interp_p; 
                    } else {
                        InterpolateCrestPoint(new_pb, inter_pa, eb_max, e_interp_max, new_interp_p);
                        ea_max = e_interp_max;
                        new_pa = new_interp_p;
                    }
                    inter_pa = new_interp_p;
                    // std::cout << " inter p coord " << loop_id << " : " << inter_pa[0] << " " << inter_pa[1] << " " << inter_pa[2] << std::endl;
                }
                
                // inter_pa = {px, py, pz};
                // edge_emax_sign = 1;
                // edge_ridge_pts_.push_back({px, py, pz});
                // edge_ridge_pts_curvature_.push_back(int_curvature);
                // riges_sign = interp_ridge_p_id_;
                // interp_ridge_p_id_ ++;
            }
        } 
    }

    // cal min for valley 
    if(arma::dot(ta_min, tb_min) < 0)
    {
        tb_min *= -1.0;
        eb_min *= -1.0;
        de2_b  *= -1.0;
    } 
    // condition 3 from paper "Ridge-Valley Lines on Meshes via Implicit Surface Fitting"
    if(abs(ka_min) > abs(ka_max) || abs(kb_min) > abs(kb_max) )
    {
        if(ea_min * eb_min < 0)
        {
            edge_emin_sign = 1;
            // if(ea_min * (arma::dot(pba, ta_min)) > 0)
            {
                // double t = (- eb_min) / (ea_min - eb_min);
                // double px = pa[0] + t * (pb[0] - pa[0]);
                // double py = pa[1] + t * (pb[1] - pa[1]);
                // double pz = pa[2] + t * (pb[2] - pa[2]);
                double abs_sum = abs(ea_min)  + abs(eb_min);
                // double px = (abs(eb_min) * pa[0] + abs(ea_min) * pb[0])/abs_sum;
                // double py = (abs(eb_min) * pa[1] + abs(ea_min) * pb[1])/abs_sum;
                // double pz = (abs(eb_min) * pa[2] + abs(ea_min) * pb[2])/abs_sum;
                inter_cur_b = (abs(eb_min) * ka_min+ abs(ea_min) * kb_min) 
                                /(abs(ea_min)  + abs(eb_min));
                // inter_pb = {px, py, pz};

                // InterpolateCrestPoint(pa, pb, ea_min, eb_min, inter_pb);
                get_cubic_root(pa, pb, ea_min, eb_min,{de2_a[0],de2_a[1],de2_a[2]}, {de2_b[0], de2_b[1], de2_b[2]}, inter_pb);
                Point new_pa = pa;
                Point new_pb = pb;
                int loop_num = 0;
                for(int loop_id = 0; loop_id < loop_num; ++loop_id)
                {
                    PrincipleCurvature intera_curva;
                    CalSinglePointCurvatureData(inter_pa, intera_curva);
                    auto t_interp_min = intera_curva.tmin_;
                    auto e_interp_min = intera_curva.emin_;
                    if(arma::dot(ta_min, t_interp_min) < 0)
                    {
                        t_interp_min *= -1.0;
                        e_interp_min *= -1.0;
                    } 
                    Point new_interp_p;
                    if(ea_min * e_interp_min <= 0)
                    {
                        InterpolateCrestPoint(new_pa, inter_pa, ea_min, e_interp_min, new_interp_p);
                        eb_min = e_interp_min;
                        new_pb = new_interp_p;
                    } else {
                        InterpolateCrestPoint(new_pb, inter_pa, eb_min, e_interp_min, new_interp_p);
                        ea_min = e_interp_min;
                        new_pa = new_interp_p;
                    }
                    inter_pa = new_interp_p;
                }
                // edge_valley_pts_.push_back({px, py, pz});
                // edge_valley_pts_curvature_.push_back(int_curvature);
                // valley_sign = interp_valley_p_id_;
                // interp_valley_p_id_ ++;
            }
        } 
    }
    return true;
}


bool VIPSSRidges::CalculateEdgeRidgeValleyPoints()
{
    size_t e_num = edges_.size();
    edge_ridge_signs_.resize(e_num, -1);
    edge_valley_signs_.resize(e_num, -1);
    edge_gaussian_signs_.resize(e_num, -1);
    edge_ridge_pts_.clear();
    edge_valley_pts_.clear();
    edge_gaussian_pts_.clear();

    edge_ridge_pts_curvature_.clear();
    edge_valley_pts_curvature_.clear();
    edge_gaussian_curvature_.clear();
    interp_ridge_p_id_ = 0;
    interp_valley_p_id_ = 0;
    interp_gaussian_p_id_ = 0;
    for(int i = 0; i < e_num; ++i)
    {
        const auto& cur_e = edges_[i];
        size_t pa_id = cur_e[0];
        size_t pb_id = cur_e[1];
        const auto& pa = mesh_points_[pa_id];
        const auto& pb = mesh_points_[pb_id];
        const auto& curvature_a = mesh_pt_curvatures_[pa_id]; 
        const auto& curvature_b = mesh_pt_curvatures_[pb_id]; 
        CalculateCrestPoints(pa, pb, curvature_a, curvature_b,
                        edge_ridge_signs_[i], 
                        edge_valley_signs_[i],
                        edge_gaussian_signs_[i]
                    );

    }
    return true;
}


bool VIPSSRidges::CalculateRidegeEdges(const TriFace& cur_f, 
                        std::unordered_map<string, size_t>& edge_id_map,
                        const std::vector<int>& edge_signs,
                        std::vector<Point>& edge_int_pts, 
                        std::vector<double>& edge_pts_curvatures, 
                        std::vector<std::vector<size_t>>& out_ridge_edges)
{
    // const auto& cur_f = mesh_faces_[f_id];
    string token_ab = CalEdgeToken(cur_f[0], cur_f[1]);
    string token_bc = CalEdgeToken(cur_f[1], cur_f[2]);
    string token_ca = CalEdgeToken(cur_f[2], cur_f[0]);
    int sum_sign = 0; 
    std::vector<size_t> new_e_ids; 
    // std::cout << "tri face : " << tet[v1]<<" " << tet[v2] <<  " " << tet[v3] << std::endl;
    size_t e_ab_id = edge_id_map[token_ab];

    // std::cout << "e_ab_id : "  << e_ab_id << std::endl;
    // std::cout << "edge_signs : "  << edge_signs.size() << std::endl;

    if(edge_signs[e_ab_id] >= 0)
    {
        sum_sign ++;
        new_e_ids.push_back(edge_signs[e_ab_id]);
    }
    // std::cout << "token_bc : "  << token_bc << std::endl;
    size_t e_bc_id = edge_id_map[token_bc];
    // std::cout << "e_bc_id : "  << e_bc_id << std::endl;
    if(edge_signs[e_bc_id] >= 0)
    {
        sum_sign ++;
        new_e_ids.push_back(edge_signs[e_bc_id]);
    }
    size_t e_ca_id = edge_id_map[token_ca];
    // std::cout << "e_ca_id : "  << e_ca_id << std::endl;
    if(edge_signs[e_ca_id] >= 0)
    {
        sum_sign ++;
        new_e_ids.push_back(edge_signs[e_ca_id]);
    }
    if(sum_sign == 2)
    {
        out_ridge_edges.push_back({new_e_ids[0], new_e_ids[1]});
    }
    if(sum_sign == 3)
    {
        const auto& pa = edge_int_pts[new_e_ids[0]];
        const auto& pb = edge_int_pts[new_e_ids[1]];
        const auto& pc = edge_int_pts[new_e_ids[2]];
        double px = (pa[0] + pb[0] + pc[0])/3.0;
        double py = (pa[1] + pb[1] + pc[1])/3.0;
        double pz = (pa[2] + pb[2] + pc[2])/3.0;
        size_t new_pid = edge_int_pts.size(); 
        edge_int_pts.push_back({px, py, pz});
        out_ridge_edges.push_back({new_e_ids[0], new_pid});
        out_ridge_edges.push_back({new_e_ids[1], new_pid});
        out_ridge_edges.push_back({new_e_ids[2], new_pid});
        double new_k = (edge_pts_curvatures[new_e_ids[0]] +
                        edge_pts_curvatures[new_e_ids[1]] +
                        edge_pts_curvatures[new_e_ids[2]])/ 3.0;
        edge_pts_curvatures.push_back(new_k);
        // edge_pt_color_.push_back({255, 0, 0});
    }
    return true;
}

// bool VIPSSRidges::CalculateRidegeEdges(int f_id)
// {
//     const auto& cur_f = mesh_faces_[f_id];
//     string token_ab = CalEdgeToken(cur_f[0], cur_f[1]);
//     string token_bc = CalEdgeToken(cur_f[1], cur_f[2]);
//     string token_ca = CalEdgeToken(cur_f[2], cur_f[0]);
//     int sum_sign = 0; 
//     std::vector<size_t> new_e_ids; 
    
//     size_t e_ab_id = edge_id_map_[token_ab];
//     if(edge_signs_[e_ab_id] >= 0)
//     {
//         sum_sign ++;
//         new_e_ids.push_back(edge_signs_[e_ab_id]);
//     }
//     size_t e_bc_id = edge_id_map_[token_bc];
//     if(edge_signs_[e_bc_id] >= 0)
//     {
//         sum_sign ++;
//         new_e_ids.push_back(edge_signs_[e_bc_id]);
//     }
//     size_t e_ca_id = edge_id_map_[token_ca];
//     if(edge_signs_[e_ca_id] >= 0)
//     {
//         sum_sign ++;
//         new_e_ids.push_back(edge_signs_[e_ca_id]);
//     }
//     if(sum_sign == 2)
//     {
//         ridge_edges_.push_back({new_e_ids[0], new_e_ids[1]});
//     }
//     if(sum_sign == 3)
//     {
//         const auto& pa = edge_int_pts_[new_e_ids[0]];
//         const auto& pb = edge_int_pts_[new_e_ids[1]];
//         const auto& pc = edge_int_pts_[new_e_ids[2]];
//         double px = (pa[0] + pb[0] + pc[0])/3.0;
//         double py = (pa[1] + pb[1] + pc[1])/3.0;
//         double pz = (pa[2] + pb[2] + pc[2])/3.0;
//         size_t new_pid = edge_int_pts_.size(); 
//         edge_int_pts_.push_back({px, py, pz});
//         ridge_edges_.push_back({new_e_ids[0], new_pid});
//         ridge_edges_.push_back({new_e_ids[1], new_pid});
//         ridge_edges_.push_back({new_e_ids[2], new_pid});
//         edge_pt_color_.push_back({255, 0, 0});
//     }
//     return true;
// }


// VIPSSRidges::Point VIPSSRidges::IterpolateEdgesPt(const Point& pa, const Point& pb, double va, double vb, double inter_val)
// {
//     // double t = ()
// }

void VIPSSRidges::ExtractLevelSetCurvesOnMesh(const std::vector<Point>& mesh_points, 
                const std::vector<std::vector<size_t>>& mesh_faces, 
                const std::vector<double> &pt_vals,
                const string& curve_path,
                const double level_val, 
                const std::array<double,3> color)
{
    std::vector<std::vector<size_t>> mesh_edges; 
    std::unordered_map<std::string, size_t> edge_token_id_map;
    for(const auto& face : mesh_faces)
    {
        auto token_ab = CalEdgeToken(face[0], face[1]);
        if(edge_token_id_map.find(token_ab) == edge_token_id_map.end())
        {
            edge_token_id_map[token_ab] = mesh_edges.size();
            mesh_edges.push_back({face[0], face[1]});
        }

        auto token_bc = CalEdgeToken(face[1], face[2]);
        if(edge_token_id_map.find(token_bc) == edge_token_id_map.end())
        {
            edge_token_id_map[token_bc] = mesh_edges.size();
            mesh_edges.push_back({face[1], face[2]});
        }

        auto token_ca = CalEdgeToken(face[2], face[0]);
        if(edge_token_id_map.find(token_ca) == edge_token_id_map.end())
        {
            edge_token_id_map[token_ca] = mesh_edges.size();
            mesh_edges.push_back({face[0], face[2]});
        }
    }

    std::vector<Point> inter_pts;
    std::vector<int> edge_signs(mesh_edges.size(), -1);
    // for(const auto& edge : mesh_edges)
    for(int e_id = 0; e_id < mesh_edges.size(); ++e_id)
    {
        const auto& edge = mesh_edges[e_id];
        size_t pa_id = edge[0];
        size_t pb_id = edge[1];

        const auto& pa = mesh_points[pa_id];
        const auto& pb = mesh_points[pb_id];

        const auto val_a = pt_vals[pa_id] - level_val;
        const auto val_b = pt_vals[pb_id] - level_val;

        
        if( val_a * val_b <= 0)
        {
            Point new_p;
            for(int i = 0; i < 3; ++i)
            {
                new_p[i] = (pa[i] * abs(val_b) + pb[i] * abs(val_a)) /(abs(val_b) + abs(val_a));
            }
            edge_signs[e_id] = inter_pts.size(); 
            inter_pts.push_back(new_p);
        }   
    }

    std::vector<std::vector<size_t>> curve_edges;


    for(const auto& face : mesh_faces)
    {
        auto token_ab = CalEdgeToken(face[0], face[1]);
        auto token_bc = CalEdgeToken(face[1], face[2]);
        auto token_ca = CalEdgeToken(face[2], face[0]);

        std::vector<string> tokens = {token_ab, token_bc, token_ca};
        std::vector<size_t> pids;
        for(const auto& token : tokens)
        {
            size_t e_id = edge_token_id_map[token];
            if(edge_signs[e_id] >= 0)
            {
                pids.push_back(edge_signs[e_id]);
            }
        }
        if(pids.size() == 2)
        {
            curve_edges.push_back({pids[0], pids[1]});
        } else if(pids.size() == 3)
        {
            Point new_center = {0, 0, 0};
            for(auto pid : pids)
            {
                new_center[0] += inter_pts[pid][0];
                new_center[1] += inter_pts[pid][1];
                new_center[2] += inter_pts[pid][2];
            }
            new_center[0] /= 3.0;
            new_center[1] /= 3.0;
            new_center[2] /= 3.0;
            size_t new_id = inter_pts.size();
            inter_pts.push_back(new_center);
            for(auto pid : pids)
            {
                curve_edges.push_back({new_id, pid});
            }
        }
    }
    // SaveRidgesToObj(curve_path, inter_pts, curve_edges, scale_, ori_center_);
    // std::array<double,3> color = {0, 0.5, 0};
    // if(level_val > 0)
    // {
    //     color[0] = std::min(1.0, level_val/0.2);
    // } else {
    //     color[2] = std::min(1.0, -level_val/0.2);
    // }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << level_val;
    std::string key = oss.str();
    curves_pts_map[key] = inter_pts;
    curves_edges_map[key] = curve_edges;

    SaveRidgesToObjWithColor(curve_path, inter_pts, curve_edges, color, scale_, ori_center_);

    VIPSSRidges::SaveRidgesWithColorToPLY(curve_path + "_color.ply", inter_pts, curve_edges, color);
}


size_t locateTriFaceEdge(const std::vector<std::array<double,3>>& mesh_points, 
                const std::vector<std::array<size_t, 2>> & all_mesh_edges,
                const std::vector<double> &pt_vals,
                const arma::vec3& pt, double& iso_val)
{
    size_t e_n = all_mesh_edges.size();
    size_t e_id = e_n;
    double step = 1e-12;
    
    for(size_t i = 0; i < e_n; ++i)
    {
        const auto& cur_edge = all_mesh_edges[i];
        const auto& pa = mesh_points[cur_edge[0]];
        const auto& pb = mesh_points[cur_edge[1]];
        double min_x = std::min(pa[0], pb[0]) - step;
        double min_y = std::min(pa[1], pb[1]) - step;
        double min_z = std::min(pa[2], pb[2]) - step;
        double max_x = std::max(pa[0], pb[0]) + step;
        double max_y = std::max(pa[1], pb[1]) + step;
        double max_z = std::max(pa[2], pb[2]) + step;

        if(pt[0] <= max_x && pt[0] >= min_x &&
           pt[1] <= max_y && pt[1] >= min_y &&
           pt[2] <= max_z && pt[2] >= min_z )
        {
            arma::vec3 v_ab = {pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};
            arma::vec3 v_pa = {pt[0] - pa[0], pt[1] - pa[1], pt[2] - pa[2]};
            arma::vec3 v_pb = {pt[0] - pb[0], pt[1] - pb[1], pt[2] - pb[2]};

            double len_ab = arma::norm(v_ab);
            double len_pa = arma::norm(v_pa);
            double len_pb = arma::norm(v_pb);
        
            double len_diff_thred = 1e-6;
            if(len_pa + len_pb - len_ab < len_diff_thred)
            {
                double iso_val_a = pt_vals[cur_edge[0]];
                double iso_val_b = pt_vals[cur_edge[1]];
                iso_val = iso_val_a + ( iso_val_b - iso_val_a) * len_pa / len_ab; 
                return i;
            }
        }
    }
    return e_id;
}

size_t locateTriFace(const std::vector<std::array<double,3>>& mesh_points, 
                const std::vector<std::vector<size_t>>& mesh_faces,
                const arma::vec3& pt)
{
    size_t f_n = mesh_faces.size();
    size_t f_id = f_n;
    for(size_t i = 0; i < f_n; ++i)
    {
        const auto& f = mesh_faces[i];
        const auto& pa = mesh_points[f[0]];
        const auto& pb = mesh_points[f[1]];
        const auto& pc = mesh_points[f[2]];
        double min_x = std::min(pa[0], std::min(pb[0], pc[0]));
        double min_y = std::min(pa[1], std::min(pb[1], pc[1]));
        double min_z = std::min(pa[2], std::min(pb[2], pc[2]));

        double max_x = std::max(pa[0], std::max(pb[0], pc[0]));
        double max_y = std::max(pa[1], std::max(pb[1], pc[1]));
        double max_z = std::max(pa[2], std::max(pb[2], pc[2]));
        if(pt[0] <= max_x && pt[0] >= min_x &&
            pt[1] <= max_y && pt[1] >= min_y &&
            pt[2] <= max_z && pt[2] >= min_z)
        {
            arma::vec3 v_ab =  {pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};
            arma::vec3 v_bc =  {pc[0] - pb[0], pc[1] - pb[1], pc[2] - pb[2]};
            arma::vec3 v_ca =  {pa[0] - pc[0], pa[1] - pc[1], pa[2] - pc[2]};
            arma::vec3 v_pa = {pt[0] - pa[0], pt[1] - pa[1], pt[2] - pa[2]};
            arma::vec3 v_pb = {pt[0] - pb[0], pt[1] - pb[1], pt[2] - pb[2]};
            arma::vec3 v_pc = {pt[0] - pc[0], pt[1] - pc[1], pt[2] - pc[2]};
            arma::vec3 cross_ap = arma::cross(v_pa, v_ab);
            arma::vec3 cross_bp = arma::cross(v_pb, v_bc);
            arma::vec3 cross_cp = arma::cross(v_pc, v_ca);
            // f_id = f_id == f_n?  i : f_id;
            if(arma::dot(cross_ap, cross_bp) >= 0 
            && arma::dot(cross_ap, cross_cp) >= 0
            && arma::dot(cross_bp, cross_cp) >= 0)
            {
                return i;
            }
            double len_ab = arma::norm(v_ab);
            double len_bc = arma::norm(v_bc);
            double len_ca = arma::norm(v_ca);
            double len_pa = arma::norm(v_pa);
            double len_pb = arma::norm(v_pb);
            double len_pc = arma::norm(v_pc);
        
            double len_diff_thred = 1e-6;
            if(len_pa + len_pb - len_ab < len_diff_thred)
            {
                return i;
            }
            if(len_pb + len_pc - len_bc < len_diff_thred)
            {
                return i;
            }
            if(len_pc + len_pa - len_ca < len_diff_thred)
            {
                return i;
            }
        }
    }
    return f_id;
}

bool CheckIntersectFace(const std::vector<std::array<double,3>>& mesh_points, 
                    const arma::vec3& face_interplane_normal,
                    const std::vector<size_t>& face,
                    const arma::vec3& pt)
{
    std::array<std::array<size_t,2>, 3> cur_edges = {{{0, 1}, {1, 2}, {2, 0}}};
    arma::vec3 pa = {mesh_points[face[0]][0],mesh_points[face[0]][1], mesh_points[face[0]][2]};
    arma::vec3 pb = {mesh_points[face[1]][0],mesh_points[face[1]][1], mesh_points[face[1]][2]};
    arma::vec3 pc = {mesh_points[face[2]][0],mesh_points[face[2]][1], mesh_points[face[2]][2]}; 
    arma::vec3 v_pa = pa - pt;
    arma::vec3 v_pb = pb - pt;
    arma::vec3 v_pc = pc - pt;
    std::vector<arma::vec3> f_pts = {pa, pb, pc};
    double proja = arma::dot(face_interplane_normal,v_pa);
    double projb = arma::dot(face_interplane_normal,v_pb);
    double projc = arma::dot(face_interplane_normal,v_pc);
    std::vector<double> pro_vals = {proja, projb, projc};
    if(proja < 0 && projb < 0 && projc < 0) return false;
    if(proja > 0 && projb > 0 && projc > 0) return false;
    return true;
}


bool IntersectTriFace(
                    const std::vector<std::array<double,3>>& mesh_points, 
                    const arma::vec3& face_interplane_normal,
                    const std::vector<size_t>& face,
                    const arma::vec3& pt, const arma::vec3& tri_isovals, 
                    TriFaceIntersectRes& intersect_res )
{
    
    // std::array<std::array<size_t,2>, 3> cur_edges = 
    //             {{face[0], face[1]}, {face[1], face[2]}, {face[2], face[0]}};
    bool& has_intersect_res   = intersect_res.has_intersect_res; 
    arma::vec3& iso_incre_pt = intersect_res.iso_incre_pt;
    double& iso_incre_val    = intersect_res.iso_incre_val;
    std::array<size_t,2>& incre_edge = intersect_res.incre_edge; 
    arma::vec3& iso_decre_pt  = intersect_res.iso_decre_pt; 
    double& iso_decre_val     = intersect_res.iso_decre_val; 
    std::array<size_t,2>& decre_edge = intersect_res.decre_edge; 
    std::array<std::array<size_t,2>, 3> cur_edges = {{{0, 1}, {1, 2}, {2, 0}}};
    arma::vec3 pa = {mesh_points[face[0]][0],mesh_points[face[0]][1], mesh_points[face[0]][2]};
    arma::vec3 pb = {mesh_points[face[1]][0],mesh_points[face[1]][1], mesh_points[face[1]][2]};
    arma::vec3 pc = {mesh_points[face[2]][0],mesh_points[face[2]][1], mesh_points[face[2]][2]}; 
    arma::vec3 v_pa = pa - pt;
    arma::vec3 v_pb = pb - pt;
    arma::vec3 v_pc = pc - pt;
    std::vector<arma::vec3> f_pts = {pa, pb, pc};
    double proja = arma::dot(face_interplane_normal,v_pa);
    double projb = arma::dot(face_interplane_normal,v_pb);
    double projc = arma::dot(face_interplane_normal,v_pc);

    // std::cout << std::fixed << std::setprecision(8)  << std::endl;
    // std::cout <<  " v_pt : " << std::fixed << std::setprecision(8)  << pt.t() << std::endl;
    // std::cout << " face_interplane_normal : " << std::fixed << std::setprecision(8) <<  face_interplane_normal.t() << std::endl;
    // std::cout <<  " v_pa : " <<std::fixed << std::setprecision(8) << v_pa.t() << std::endl;
    // std::cout <<  " v_pb : " <<std::fixed << std::setprecision(8) << v_pb.t() << std::endl;
    // std::cout << " v_pc : " <<std::fixed << std::setprecision(8) <<  v_pc.t() << std::endl;
    std::vector<double> pro_vals = {proja, projb, projc};
    iso_incre_val = arma::min(tri_isovals);
    iso_decre_val = arma::max(tri_isovals);
    double numerical_threshold = 1e-6;
    std::vector<size_t> intersect_edge;
    // std::cout << "---proj vals " << proja << " " << projb << " " << projc << std::endl;
    // to deal with numerical error cases where the intersect plane go through one edge of the triangle. 
    // if( (proja < 0 && projb < 0 && projc < 0) || (proja > 0 && projb > 0 && projc > 0) )
    // {
    //     for(int i =0; i < 3; ++i)
    //     {
    //         if(abs(pro_vals[i]) < numerical_threshold)
    //         {
    //             if(tri_isovals[i] >= iso_incre_val)
    //             {
    //                 iso_incre_val = tri_isovals[i];
    //                 iso_incre_pt  = f_pts[i];
    //                 intersect_edge.push_back(face[i]);
    //             }
    //             if(tri_isovals[i] <= iso_decre_val)
    //             {
    //                 iso_decre_val = tri_isovals[i];
    //                 iso_decre_pt  = f_pts[i];
    //                 intersect_edge.push_back(face[i]);
    //             }
    //         }
    //     }
    // } 
    // std::cout << " proj vals : " << proja << " " << projb << " " << projc << std::endl;
    // std::cout << " min max iso vals " << iso_decre_val << " " << iso_incre_val << std::endl; 
    for(const auto& edge : cur_edges)
    {
        if(pro_vals[edge[0]] * pro_vals[edge[1]] <= 0)
        {
            has_intersect_res = true;
            double t = std::abs(pro_vals[edge[0]]) /(std::abs(pro_vals[edge[0]]) + std::abs(pro_vals[edge[1]]));
            arma::vec3 newPt = f_pts[edge[0]] + t * (f_pts[edge[1]] - f_pts[edge[0]]);
            // std::cout << " inter pts : " << newPt[0] << " " << newPt[1] << " " << newPt[2] << std::endl;
            double new_val = tri_isovals[edge[0]] + t * (tri_isovals[edge[1]] - tri_isovals[edge[0]]);
            // std::cout << " iso vals " << new_val << std::endl;
            if(new_val >= iso_incre_val)
            {
                iso_incre_val = new_val;
                iso_incre_pt  = newPt;
                incre_edge = {face[edge[0]], face[edge[1]]};
            } 
            if(new_val <= iso_decre_val)
            {
                iso_decre_val = new_val;
                iso_decre_pt  = newPt;
                decre_edge = {face[edge[0]], face[edge[1]]};
            } 
        }
    }
    return true;
}
std::string GenerateEdgeKey(const size_t e_a, const size_t e_b)
{
    if(e_a > e_b)
    {
        return std::to_string(e_b) + "_" + std::to_string(e_a);
    }
    return std::to_string(e_a) + "_" + std::to_string(e_b);
}


void SearchNextPt(const std::vector<std::array<double,3>>& mesh_points, 
            const std::vector<std::vector<size_t>>& mesh_faces,
            const std::vector<arma::vec3>& face_interPlane_normals,
            const std::vector<arma::vec3>& face_gradients,
            const std::vector<double> &pt_vals,
            const std::vector<std::vector<size_t>>& pid_eids_map, 
            const std::vector<std::array<size_t, 2>>& all_mesh_edges,
            const std::vector<std::vector<size_t>>& edge_hash_face_id_map,
            const std::vector<std::vector<size_t>>& pid_fid_map,
            std::unordered_map<std::string, size_t>& edge_id_map,
            std::unordered_map<std::string, int>& face_edge_signs,
            std::unordered_map<std::string, int>& pt_face_grad_sign_map,
            TriMeshSearchRes& pre_search_res,
            bool search_iso_incre, 
            TriMeshSearchRes& search_res)
{
    // auto edge_id = locateTriFaceEdge(mesh_points, all_mesh_edges, pt); 
    // std::string e_key = GenerateEdgeKey(incre_edge[0], incre_edge[1]);
    const arma::vec3& seed_pt = pre_search_res.next_pt;
    size_t e_id = pre_search_res.edge_id;
    size_t p_id = pre_search_res.p_id;
    size_t e_n = all_mesh_edges.size();
    size_t p_n = mesh_points.size();
    size_t f_n = mesh_faces.size();
    if(pre_search_res.pt_type == NextPtType::EdgePt)
    {
        bool has_valid_fid  = false;
        size_t largest_f_id = f_n;
        double largest_grad_mag = 0;
       
        const auto& next_fids =  edge_hash_face_id_map[e_id];

        // std::cout << "--- next face ids : "<< next_fids[0] << " " << next_fids[1] << std::endl;
        if(search_iso_incre)
        {
            for(auto f_id : next_fids)
            {
                std::string ef_key = std::to_string(e_id) + "_" + std::to_string(f_id);
                // std::cout << " ef_key : "<< ef_key << " " << face_edge_signs[ef_key] << std::endl;
                if(face_edge_signs[ef_key] > 0)
                {
                    has_valid_fid = true;
                    const auto& grad = face_gradients[f_id];
                    double g_norm = arma::norm(grad);
                    if(g_norm > largest_grad_mag)
                    {
                        largest_grad_mag = g_norm;
                        largest_f_id = f_id;
                    }
                }
            }
            // std::cout << " largest_f_id : "<< largest_f_id << std::endl;
            TriFaceIntersectRes intersect_res;
            if(has_valid_fid)
            {
                const auto& max_face = mesh_faces[largest_f_id];
                arma::vec3 tri_isovals = {pt_vals[max_face[0]], pt_vals[max_face[1]], pt_vals[max_face[2]]};
                // std::cout << " pt iso vals : " << tri_isovals << std::endl;
                IntersectTriFace(mesh_points, face_interPlane_normals[largest_f_id], 
                        max_face, seed_pt, tri_isovals, intersect_res);
                
                // std::cout << "intersect_res: " << intersect_res.has_intersect_res << std::endl;        
                if(intersect_res.has_intersect_res)
                {
                    const auto& incre_edge =  intersect_res.incre_edge;
                    auto e_key = GenerateEdgeKey(incre_edge[0], incre_edge[1]);
                    search_res.edge_id = edge_id_map[e_key];
                    search_res.pt_type = NextPtType::EdgePt;
                    search_res.next_pt = intersect_res.iso_incre_pt;
                    search_res.iso_val = intersect_res.iso_incre_val;
                    return;
                }        
                
            } 
            if( !has_valid_fid || !intersect_res.has_intersect_res) 
            {
                const auto& cur_edge =  all_mesh_edges[e_id];
                size_t larger_val_pid = pt_vals[cur_edge[0]] > pt_vals[cur_edge[1]] ?
                                        cur_edge[0] : cur_edge[1];
                search_res.p_id = larger_val_pid;
                search_res.pt_type = NextPtType::MeshPt;
                const auto& cur_pt = mesh_points[larger_val_pid];
                search_res.next_pt = {cur_pt[0], cur_pt[1], cur_pt[2]};
                search_res.iso_val = pt_vals[larger_val_pid];

                // std::cout << " get next mesh pid : " << larger_val_pid << std::endl;
                return;
            }
        }
        else {
            for(auto f_id : next_fids)
            {
                std::string ef_key = std::to_string(e_id) + "_" + std::to_string(f_id);
                if(face_edge_signs[ef_key] < 0)
                {
                    has_valid_fid = true;
                    const auto& grad = face_gradients[f_id];
                    double g_norm = arma::norm(grad);
                    if(g_norm > largest_grad_mag)
                    {
                        largest_grad_mag = g_norm;
                        largest_f_id = f_id;
                    }
                }
            }
            TriFaceIntersectRes intersect_res;
            if(has_valid_fid)
            {
                const auto& max_face = mesh_faces[largest_f_id];
                arma::vec3 tri_isovals = {pt_vals[max_face[0]], pt_vals[max_face[1]], pt_vals[max_face[2]]};
                
                IntersectTriFace(mesh_points, face_interPlane_normals[largest_f_id], 
                        max_face, seed_pt, tri_isovals, intersect_res);

                // std::cout << "intersect_res: " << intersect_res.has_intersect_res << std::endl;   

                if(intersect_res.has_intersect_res)
                {
                    const auto& decre_edge =  intersect_res.decre_edge;
                    auto e_key = GenerateEdgeKey(decre_edge[0], decre_edge[1]);
                    search_res.edge_id = edge_id_map[e_key];
                    search_res.pt_type = NextPtType::EdgePt;
                    search_res.next_pt = intersect_res.iso_decre_pt;
                    search_res.iso_val = intersect_res.iso_decre_val;
                    return;
                }
            }
            if( !has_valid_fid || !intersect_res.has_intersect_res)  
            {
                if(edge_hash_face_id_map[e_id].size() < 2) return;
                const auto& cur_edge =  all_mesh_edges[e_id];
                size_t min_val_pid = pt_vals[cur_edge[0]] < pt_vals[cur_edge[1]] ?
                                        cur_edge[0] : cur_edge[1];
                search_res.p_id = min_val_pid;
                search_res.pt_type = NextPtType::MeshPt;
                const auto& cur_pt = mesh_points[min_val_pid];
                search_res.next_pt = {cur_pt[0], cur_pt[1], cur_pt[2]};
                search_res.iso_val = pt_vals[min_val_pid];
                return;
            }
        }
    } // end of valid eid 
    bool has_valid_face_id = false;
    TriFaceIntersectRes intersect_res;
    if(pre_search_res.pt_type == NextPtType::MeshPt)
    {
        const auto& next_fids = pid_fid_map[p_id];
        size_t largest_grad_fid;
        double max_grad_len = 0;
        for(auto f_id : next_fids)
        {
            std::string pf_key = std::to_string(p_id) + "_" + std::to_string(f_id);
            if(pt_face_grad_sign_map[pf_key] > 0 && search_iso_incre)
            {
                double grad_len = arma::norm(face_gradients[f_id]);
                if(grad_len > max_grad_len)
                {
                    max_grad_len = grad_len;
                    has_valid_face_id = true;
                    largest_grad_fid = f_id;
                }
            }
            if(pt_face_grad_sign_map[pf_key] < 0 && !search_iso_incre)
            {
                double grad_len = arma::norm(face_gradients[f_id]);
                if(grad_len > max_grad_len)
                {
                    max_grad_len = grad_len;
                    has_valid_face_id = true;
                    largest_grad_fid = f_id;
                }
            }
        }
        
        if(has_valid_face_id)
        {
            const auto& max_face = mesh_faces[largest_grad_fid];
            arma::vec3 tri_isovals = {pt_vals[max_face[0]], pt_vals[max_face[1]], pt_vals[max_face[2]]};
            IntersectTriFace(mesh_points, face_interPlane_normals[largest_grad_fid], 
                    max_face, seed_pt, tri_isovals, intersect_res);
            if(intersect_res.has_intersect_res)
            {
                if(search_iso_incre)
                {
                    const auto& incre_edge =  intersect_res.incre_edge;
                    auto e_key = GenerateEdgeKey(incre_edge[0], incre_edge[1]);
                    search_res.edge_id = edge_id_map[e_key];
                    search_res.pt_type = NextPtType::EdgePt;
                    search_res.next_pt = intersect_res.iso_incre_pt;
                    search_res.iso_val = intersect_res.iso_incre_val;
                } else {
                    const auto& decre_edge =  intersect_res.decre_edge;
                    auto e_key = GenerateEdgeKey(decre_edge[0], decre_edge[1]);
                    search_res.edge_id = edge_id_map[e_key];
                    search_res.pt_type = NextPtType::EdgePt;
                    search_res.next_pt = intersect_res.iso_decre_pt;
                    search_res.iso_val = intersect_res.iso_decre_val;
                }
            } else {
                // std::cout << " valid face has no intersection !" << std::endl;
            }

        }
    }

    if(pre_search_res.pt_type == NextPtType::MeshPt)
    {
        if( !has_valid_face_id || !intersect_res.has_intersect_res)
        {
            {
                // std::cout << " mesh pt has no valid face for iso incre search !" << std::endl;
                bool has_valid_max_grad_edge = false; 
                bool has_valid_min_grad_edge = false;
                double max_grad_len = 0;
                double min_grad_len = 0;
                size_t max_p_id;
                size_t max_e_id;
                size_t min_p_id;
                size_t min_e_id;
                
                const auto& neig_eids = pid_eids_map[p_id];
                // std::cout << " pt neighbor edge count " <<  neig_eids.size() << std::endl;
                for(const auto e_id : neig_eids)
                {
                    const auto& cur_e = all_mesh_edges[e_id];
                    size_t e_pid2 = cur_e[0] == p_id ? cur_e[1] : cur_e[0]; 
                    arma::vec3 e_pa = {mesh_points[p_id][0],mesh_points[p_id][1],mesh_points[p_id][2]};
                    arma::vec3 e_pb = {mesh_points[e_pid2][0],mesh_points[e_pid2][1],mesh_points[e_pid2][2]};
                    double e_len = arma::norm(e_pb - e_pa);
                    double iso_val_a = pt_vals[p_id];
                    double iso_val_b = pt_vals[e_pid2];
                    double cur_grad_len = (iso_val_b - iso_val_a) / e_len;
                    if(cur_grad_len > max_grad_len)
                    {
                        has_valid_max_grad_edge = true;
                        max_grad_len = cur_grad_len;
                        max_p_id = e_pid2;
                        max_e_id = e_id;
                    } 
                    if(cur_grad_len < min_grad_len)
                    {
                        has_valid_min_grad_edge = true;
                        min_grad_len = cur_grad_len;
                        min_p_id = e_pid2;
                        min_e_id = e_id;
                    }
                }
                if(search_iso_incre && has_valid_max_grad_edge)
                {
                    if(edge_hash_face_id_map[max_e_id].size() < 2) return;
                    arma::vec3 next_pt = {mesh_points[max_p_id][0],mesh_points[max_p_id][1],mesh_points[max_p_id][2]}; 
                    search_res.p_id = max_p_id;
                    search_res.pt_type = NextPtType::MeshPt;
                    search_res.next_pt = next_pt;
                    search_res.iso_val = pt_vals[max_p_id];
                }
                if(!search_iso_incre && has_valid_min_grad_edge)
                {
                    if(edge_hash_face_id_map[min_e_id].size() < 2) return;
                    arma::vec3 next_pt = {mesh_points[min_p_id][0],mesh_points[min_p_id][1],mesh_points[min_p_id][2]}; 
                    search_res.p_id = min_p_id;
                    search_res.pt_type = NextPtType::MeshPt;
                    search_res.next_pt = next_pt;
                    search_res.iso_val = pt_vals[min_p_id];
                }
            } 
        }
    }
    
}

void IterativeSearch(const std::vector<std::array<double,3>>& mesh_points, 
            const std::vector<std::vector<size_t>>& mesh_faces,
            const std::vector<arma::vec3>& face_interPlane_normals,
            const std::vector<arma::vec3>& face_gradients,
            const std::vector<double> &pt_vals,
            const std::vector<std::array<size_t, 2>>& all_mesh_edges,
            const std::vector<std::vector<size_t>>& pid_eids_map, 
            const std::vector<std::vector<size_t>>& pid_fids_map,
            const std::vector<std::vector<size_t>>& edge_hash_face_id_map,
            const std::vector<std::vector<size_t>>& pid_fid_map,
            std::unordered_map<std::string, size_t>& edge_id_map,
            std::unordered_map<std::string, int>& face_edge_signs,
            std::unordered_map<std::string, int>& pt_face_grad_sign_map,
            const arma::vec3& pt,
            bool search_iso_incre,
            double& consistence)
{
    
    size_t p_count = 0;
    // size_t cur_f_id =  f_id;
    arma::vec3 seed_pt = pt; 
    std::vector<double> inter_search_pts;
    std::vector<arma::vec3> inter_incre_pts;
    std::vector<arma::vec3> inter_decre_pts;
    inter_incre_pts.push_back(pt);
    inter_decre_pts.push_back(pt);
    
    double max_iso_val = 0.05;
    double min_iso_val = -max_iso_val;
    // arma::vec3 gradient;
    // VIPSSRidges::g_hrfb_ptr->evaluate_gradient(pt[0], pt[1], pt[2], gradient[0], gradient[1], gradient[2]);
    // const auto& tri_face = mesh_faces[cur_f_id];
    // const auto& pa = mesh_points[tri_face[0]];
    // const auto& pb = mesh_points[tri_face[1]];
    // const auto& pc = mesh_points[tri_face[2]];
    // arma::vec3 v_ab = {pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};
    // arma::vec3 v_ac = {pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]};
    // arma::vec3 f_normal = arma::cross(v_ab, v_ac);
    // arma::vec3 inter_plane_normal = arma::cross(f_normal, gradient);
    // std::unordered_set<size_t> visited_f_ids;
    // std::cout << " start f id " << cur_f_id << std::endl;
    // visited_f_ids.insert(cur_f_id);
    double cur_iso_val= 0;
    size_t cur_eid = locateTriFaceEdge(mesh_points, all_mesh_edges, pt_vals, seed_pt, cur_iso_val);
    // std::cout << " --------- cur e id " << cur_eid << "  ; all_mesh_edges size : " << all_mesh_edges.size() << std::endl;
    
    TriMeshSearchRes pre_search_res;
    pre_search_res.pt_type = NextPtType::EdgePt;
    pre_search_res.edge_id = cur_eid;
    pre_search_res.next_pt = seed_pt;
    if(search_iso_incre)
    {
        pre_search_res.iso_val = std::numeric_limits<double>::lowest(); 
    } else {
        pre_search_res.iso_val = std::numeric_limits<double>::max(); 
    }
    
    VIPSSRidges::search_pts_all.push_back(seed_pt);
    VIPSSRidges::search_pts_iso_vals.push_back(cur_iso_val);

    while(p_count < 10000)
    {
        // std::cout << " loop id " << p_count << std::endl;
        TriMeshSearchRes search_res;
        SearchNextPt(mesh_points,mesh_faces,face_interPlane_normals, face_gradients,
            pt_vals, pid_eids_map, all_mesh_edges,edge_hash_face_id_map,
            pid_fids_map,  edge_id_map, face_edge_signs, pt_face_grad_sign_map,
            pre_search_res, search_iso_incre, search_res);
        // std::cout << " pre_search_res iso_val : " <<  pre_search_res.iso_val << " new search_res.iso_val " << search_res.iso_val << std::endl;
        // std::cout << " next pt " << " next search vals " << search_res.iso_val  << " pt type :" << search_res.pt_type << std::endl;
        // std::cout << " ------- next pt : " << search_res.next_pt << std::endl;

        if(search_res.pt_type == NextPtType::None) break;

        VIPSSRidges::search_pts_all.push_back(search_res.next_pt);
        VIPSSRidges::search_pts_iso_vals.push_back(search_res.iso_val);
        if(search_iso_incre)
        {
            if(search_res.iso_val > pre_search_res.iso_val)
            {
                consistence = search_res.iso_val;
                if(consistence > max_iso_val) 
                {
                    consistence = max_iso_val;
                    break;
                }
            } 
            else {
                break;
            }

        } else {
            if(search_res.iso_val < pre_search_res.iso_val )
            {
                consistence = search_res.iso_val;
                if(consistence < min_iso_val) 
                {
                    consistence = min_iso_val;
                    break;
                }
            }
            else {
                break;
            }
        }
        
        // std::cout << " ------- next pt : " << search_res.next_pt << std::endl;
        pre_search_res = search_res;
        // std::cout << " ------- next pt : " << pre_search_res.next_pt << std::endl;
        p_count ++;
    }
    // if(search_iso_incre){
    //     // consistence = max_iso_val;
    //     consistence = consistence > 0 ? consistence : 0;
    // }  else {
    //     consistence = consistence < 0 ? consistence : 0;
    // }


    // while(p_count < 100)
    // {
    //     const auto& cur_f = mesh_faces[cur_f_id];
    //     // const arma::vec3& inter_plane_normal = face_interPlane_normals[cur_f_id];
    //     arma::vec3 tri_isovals = {pt_vals[cur_f[0]], pt_vals[cur_f[1]], pt_vals[cur_f[2]]};
    //     arma::vec3 iso_incre_pt;
    //     arma::vec3 iso_decre_pt;
    //     double iso_incre_val; 
    //     double iso_decre_val;
    //     std::array<size_t,2> incre_edge;
    //     std::array<size_t,2> decre_edge;
    //     IntersectTriFace(mesh_points, inter_plane_normal, 
    //             cur_f, seed_pt, tri_isovals,
    //             iso_incre_pt, iso_incre_val, incre_edge, 
    //             iso_decre_pt, iso_decre_val, decre_edge);
    //     if(search_iso_incre)
    //     {
    //         // inter_incre_pts.push_back(iso_incre_pt);
    //         VIPSSRidges::search_pts_all.push_back(iso_incre_pt);
    //         VIPSSRidges::search_pts_iso_vals.push_back(iso_incre_val);
    //         // consistence = iso_incre_val < max_iso_val ? iso_incre_val : max_iso_val;
    //         consistence = iso_incre_val;
    //         if(iso_incre_val >= max_iso_val) break;
    //         std::string e_key = GenerateEdgeKey(incre_edge[0], incre_edge[1]);
    //         size_t next_cre_edge_id = edge_id_map[e_key];
    //         const auto& next_fids =  edge_hash_face_id_map[next_cre_edge_id]; 
    //         seed_pt = iso_incre_pt;
    //         std::cout << " incre next fids " << next_fids[0] << " " << next_fids[1] << std::endl;
    //         if(next_fids.size() < 2) break;
    //         cur_f_id = next_fids[0] == cur_f_id ? next_fids[1] : next_fids[0];
    //         if(visited_f_ids.find(cur_f_id) == visited_f_ids.end())
    //         {
    //             visited_f_ids.insert(cur_f_id);
    //         } else {
    //             std::cout << "face id has already been visited " << std::endl;
    //             break;
    //         }
    //         std::string f_e_key = std::to_string(next_cre_edge_id) + "_" + std::to_string(cur_f_id);
    //         if(face_edge_signs[f_e_key] < 0) break;
    //     } else {
    //         // inter_decre_pts.push_back(iso_decre_pt);
    //         VIPSSRidges::search_pts_all.push_back(iso_decre_pt);
    //         VIPSSRidges::search_pts_iso_vals.push_back(iso_decre_val);
    //         consistence = iso_decre_val;
    //         if(iso_decre_val <= min_iso_val) break;
    //         std::string e_key = GenerateEdgeKey(decre_edge[0], decre_edge[1]);
    //         size_t next_cre_edge_id = edge_id_map[e_key];
    //         const auto& next_fids = edge_hash_face_id_map[next_cre_edge_id]; 
    //         if(next_fids.size() < 2) break;
    //         seed_pt = iso_decre_pt;
    //         cur_f_id = next_fids[0] == cur_f_id ? next_fids[1] : next_fids[0];
    //         std::string f_e_key = std::to_string(next_cre_edge_id) + "_" + std::to_string(cur_f_id);
    //         if(visited_f_ids.find(cur_f_id) == visited_f_ids.end())
    //         {
    //             visited_f_ids.insert(cur_f_id);
    //         } else {
    //             std::cout << "face id has already been visited " << std::endl;
    //             break;
    //         }
    //         if(face_edge_signs[f_e_key] > 0) break;

    //     }
    //     p_count++;
    // }  
    // consistence = consistence < max_iso_val ? consistence : max_iso_val;
    // consistence = consistence > min_iso_val ? consistence : min_iso_val;


    
    // if(search_iso_incre) 
    // {
    //     writeXYZ("search_incre_pts_test.xyz", inter_incre_pts);
    // } else {
    //     writeXYZ("search_decre_pts_test.xyz", inter_decre_pts);
    // }
    
}

void CalPtFaceGradSign(const std::vector<std::array<double,3>>& mesh_points, 
                const std::vector<std::vector<size_t>>& mesh_faces,
                const std::vector<arma::vec3>& face_gradients,
                const std::vector<arma::vec3>& face_centers,
                std::unordered_map<std::string, int> & pt_face_grad_sign_map)
{
    size_t f_n = mesh_faces.size();
    
    std::array<std::array<size_t,2>, 3> face_neighbors = {{{1,2}, {0,2}, {0,1}}};
    for(size_t i = 0; i < f_n; ++i)
    {
        const auto& face = mesh_faces[i];
        const auto& grad = face_gradients[i];
        const auto& center = face_centers[i];
        std::array<arma::vec3,3> f_pts;
        f_pts[0] = {mesh_points[face[0]][0], mesh_points[face[0]][1],mesh_points[face[0]][2]};
        f_pts[1] = {mesh_points[face[1]][0], mesh_points[face[1]][1],mesh_points[face[1]][2]};
        f_pts[2] = {mesh_points[face[2]][0], mesh_points[face[2]][1],mesh_points[face[2]][2]};
        
        for(int j = 0; j < 3; ++j)
        {
            size_t pid = face[j];
            const auto& cur_pt = f_pts[j];
            const auto& neig = face_neighbors[j];
            arma::vec3 d1 = f_pts[neig[0]] - cur_pt;
            arma::vec3 d2 = f_pts[neig[1]] - cur_pt;
            std::string key = std::to_string(pid) + "-" + std::to_string(i);
            pt_face_grad_sign_map[key] = 0;
            arma::vec3 sign1 = arma::cross(grad, d1); 
            arma::vec3 sign2 = arma::cross(grad, d2); 
            if(arma::dot(sign1, sign2) <= 0)
            {
                arma::vec3 dcenter = center - cur_pt;
                if(arma::dot(dcenter, grad) >= 0)
                {
                    pt_face_grad_sign_map[key] = 1;
                } else {
                    pt_face_grad_sign_map[key] = -1;
                }
            }
        }
    }
}

void CalEdgeFaceGradSign(const std::vector<std::array<double,3>>& mesh_points, 
                const std::vector<std::vector<size_t>>& mesh_faces,
                const std::vector<arma::vec3>& face_normals,
                std::vector<arma::vec3>& face_gradient,
                const std::vector<arma::vec3>& face_centers,
                std::unordered_map<std::string, size_t>& edge_id_map,
                std::unordered_map<std::string, int>& face_edge_signs,
                std::vector<std::vector<size_t>>& edge_hash_face_id_map)
{
    size_t f_n = mesh_faces.size();
    for(size_t i = 0; i < f_n; ++i)
    {
        const auto& f = mesh_faces[i];
        std::array<std::array<size_t,2>,3> f_edges = {{{f[0], f[1]}, {f[1],f[2]}, {f[2], f[0]}}};
        const auto& gradient = face_gradient[i];
        const auto& f_normal = face_normals[i];
        const auto& f_center = face_centers[i];
        for(const auto& edge : f_edges)
        {
            const auto& pa = mesh_points[edge[0]];
            const auto& pb = mesh_points[edge[1]];
            arma::vec3 v_ab = {pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};
            arma::vec3 e_vertical_dir = arma::cross(f_normal, v_ab);
            arma::vec3 mid_ab = {(pa[0] + pb[0])/2.0,(pa[1] + pb[1])/2.0,(pa[2] + pb[2])/2.0}; 
            arma::vec3 inward_dir = f_center - mid_ab;
            if(arma::dot(inward_dir,e_vertical_dir) < 0)
            {
                e_vertical_dir *= -1.0;
            }
            std::string e_key = GenerateEdgeKey(edge[0], edge[1]);
            size_t e_id = edge_id_map[e_key];
            edge_hash_face_id_map[e_id].push_back(i);
            std::string f_e_key = std::to_string(e_id) + "_" + std::to_string(i); 
            if(arma::dot(e_vertical_dir, gradient) <= 0)
            {
                face_edge_signs[f_e_key] = -1;
            } else {
                face_edge_signs[f_e_key] = 1;
            }
        }
    }
}

void VIPSSRidges::CalCurvesPtsConsistence(const std::vector<std::array<double,3>>& mesh_points, 
                const std::vector<std::vector<size_t>>& mesh_faces, 
                const std::vector<double> &pt_vals,
                const std::string& curve_key,
                const std::string& curve_path, 
                const double max_iso_val)
{
    size_t f_n = mesh_faces.size();
    size_t p_n = mesh_points.size();
    // std::vector<arma::vec3> mesh_pts_new(p_n);
    // for(size_t i = 0; i < p_n; ++i)
    // {
    //     mesh_pts_new[i] = {mesh_points[i][0], mesh_points[i][1], mesh_points[i][2]};
    // }
    std::vector<arma::vec3> face_gradients(f_n);  
    std::vector<arma::vec3> face_normals(f_n);  
    std::vector<arma::vec3> face_interPlane_normals(f_n);
    std::vector<std::array<size_t,2>> all_mesh_edges;
    std::vector<arma::vec3> edge_gradients;
    std::unordered_map<std::string, size_t> edge_id_map;
    std::vector<arma::vec3> face_centers(f_n);
    std::vector<std::vector<size_t> > pid_fid_map(p_n);
    std::vector<std::vector<size_t> > pid_eid_map(p_n);

    for(size_t i = 0; i < f_n; ++i)
    {
        // std::cout << " f id " << i << std::endl;
        const auto& f = mesh_faces[i];
        for(auto p_id : f)
        {
            pid_fid_map[p_id].push_back(i);
        }
        const auto& pa = mesh_points[f[0]];
        const auto& pb = mesh_points[f[1]];
        const auto& pc = mesh_points[f[2]];
        std::array<std::array<size_t, 2>,3> face_edges = 
                {{{f[0], f[1]}, {f[1], f[2]}, {f[2], f[0]}}};  
        // auto key_ab = GenerateEdgeKey(f[0], f[1]);
        // auto key_bc = GenerateEdgeKey(f[1], f[2]);
        // auto key_ca = GenerateEdgeKey(f[2], f[0]);
        // std::vector<std::string> f_keys = {key_ab, key_bc, key_ca};
        for(const auto& cur_edge : face_edges)
        {
            auto edge_key = GenerateEdgeKey(cur_edge[0], cur_edge[1]);
            if(edge_id_map.find(edge_key) == edge_id_map.end())
            {
                size_t ele_n = edge_id_map.size();
                edge_id_map[edge_key] = ele_n;
                all_mesh_edges.push_back(cur_edge);
                const auto& e_pa = mesh_points[cur_edge[0]];
                const auto& e_pb = mesh_points[cur_edge[1]];
                arma::vec3 d_ab  = {e_pb[0] - e_pa[0], e_pb[1] - e_pa[1], e_pb[2] - e_pa[2] };
                double d_iso = pt_vals[cur_edge[1]] - pt_vals[cur_edge[0]];
                arma::vec3 e_grad = arma::normalise(d_ab) * d_iso;
            }
        }

        face_centers[i] = {(pa[0] + pb[0] + pc[0])/3.0, 
                           (pa[1] + pb[1] + pc[1])/3.0,
                           (pa[2] + pb[2] + pc[2])/3.0};

        arma::vec3 v_ab =  {pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};
        arma::vec3 v_ac =  {pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]};
        arma::vec3 v_bc =  {pc[0] - pb[0], pc[1] - pb[1], pc[2] - pb[2]};
        // [v1, v2, v3] [v1T, v2T] [k1, k2] = B
        // arma::mat33 Amat = { {v_ab[0], v_ab[1], v_ab[2]},
        //                      {v_ac[0], v_ac[1], v_ac[2]},
        //                      {v_bc[0], v_bc[1], v_bc[2]}};

        arma::vec3 v_ab_unit =  arma::normalise(v_ab);
        arma::vec3 v_ac_unit =  arma::normalise(v_ac);

        arma::mat Amat = { {arma::dot(v_ab, v_ab_unit), arma::dot(v_ab, v_ac_unit)},
                             {arma::dot(v_ac, v_ab_unit), arma::dot(v_ac, v_ac_unit)},
                             {arma::dot(v_bc, v_ab_unit), arma::dot(v_bc, v_ac_unit)}};

        arma::vec3 Bvec = {pt_vals[f[1]] - pt_vals[f[0]], 
                           pt_vals[f[2]] - pt_vals[f[0]],
                           pt_vals[f[2]] - pt_vals[f[1]]};
        // if(i == 179655)
        // {
        //     std::cout << "pa " << pa[0] << " " << pa[1] << " " << pa[2] << std::endl;
        //     std::cout << "pb " << pb[0] << " " << pb[1] << " " << pb[2] << std::endl;
        //     std::cout << "pc " << pc[0] << " " << pc[1] << " " << pc[2] << std::endl;
         
        //     std::cout << " A mat " << Amat << std::endl;
        //     std::cout << " B vec " << Bvec << std::endl;
        // }
        // arma::vec2 k1k2 = arma::solve(Amat, Bvec, arma::solve_opts::fast);
        arma::vec2 k1k2 = arma::solve(Amat, Bvec);
        // if(i == 179655)
        // {
        //     std::cout << " k1k2 " << k1k2 << std::endl;
        // }

        face_gradients[i] = arma::normalise( k1k2[0] * v_ab_unit + k1k2[1] * v_ac_unit);
        arma::vec3 normal = arma::cross(v_ab, v_ac);
        face_normals[i] = arma::normalise(normal);
        face_interPlane_normals[i] = arma::normalise(arma::cross(face_gradients[i], face_normals[i]));
    }
    // writeXYZnormal("ridge_mesh_tri_gradients.xyz", face_centers, face_gradients);
    // writeXYZnormal("ridge_mesh_tri_face_normal.xyz", face_centers, face_normals);
    // writeXYZnormal("ridge_mesh_tri_plane_normal.xyz", face_centers, face_interPlane_normals); 

    for(size_t eid = 0; eid < all_mesh_edges.size(); ++eid)
    {
        const auto& cur_edge = all_mesh_edges[eid];
        pid_eid_map[cur_edge[0]].push_back(eid);
        pid_eid_map[cur_edge[1]].push_back(eid);
    }

    unordered_map<string, int> pt_face_grad_sign_map;
    CalPtFaceGradSign(mesh_points,mesh_faces,face_gradients,face_centers,pt_face_grad_sign_map);

    std::unordered_map<std::string, int> face_edge_signs;
    std::vector<std::vector<size_t>> edge_hash_face_id_map(edge_id_map.size()); 
    CalEdgeFaceGradSign(mesh_points,mesh_faces,face_normals, face_gradients,
            face_centers, edge_id_map, face_edge_signs,edge_hash_face_id_map);
    
    
    const auto& curve_pts = curves_pts_map[curve_key];
    const auto& curve_edges = curves_edges_map[curve_key];
    std::vector<double> cuv_pt_consistence(curve_pts.size(), 0);
    // double max_iso_val = 0.1;
    double min_iso_val = -max_iso_val;
    for(int i = 0; i < curve_pts.size(); ++i)
    {
        
        arma::vec3 seed_pt = {curve_pts[i][0], curve_pts[i][1], curve_pts[i][2]};
        // seed_pt = {-0.006917, 0.205088, 1.044140};
        // arma::vec3 bad_pt = {-0.533471, 0.346235, 0.983294};
        // arma::vec3 bad_pt = {-0.006917, 0.205088, 1.044140};
        // arma::vec3 bad_pt = {0.055146, 0.209869, 0.986487};
        // arma::vec3 bad_pt = {-0.36396362, 0.2436876, 1.1753531};
        // seed_pt = {-0.435075, 0.278983, 1.11872 };
        // seed_pt =  {-0.533471, 0.346235, 0.983294};
        // if(arma::norm(bad_pt - seed_pt) > 1e-7) continue;
        // std::cout << " ------------- bad seed pt " << seed_pt << std::endl;
        // std::cout << " ------------- bad seed pt " << seed_pt[0]  << " " << seed_pt[1] << " " << seed_pt[2]<< std::endl;
        // size_t f_id = locateTriFace(mesh_points, mesh_faces, seed_pt);
        // std::cout << "f id " << f_id << "  mesh face size : " << mesh_faces.size() << std::endl;
        // if(f_id >= mesh_faces.size()) continue;
        // const auto& f = mesh_faces[f_id];
        // if(i != 27 ) continue;
        
        // std::cout << " curve_pts size : " << curve_pts.size() << "  ; v id : " <<  i << std::endl; 
        // std::vector<double> test_pts;
        // test_pts.push_back(seed_pt[0]);test_pts.push_back(seed_pt[1]);test_pts.push_back(seed_pt[2]);
        // for(int k = 0; k < 3; ++k)
        // {
        //     test_pts.push_back(mesh_points[f[k]][0]);
        //     test_pts.push_back(mesh_points[f[k]][1]);
        //     test_pts.push_back(mesh_points[f[k]][2]);
        // }
        // writeXYZ("test_pts.xyz", test_pts);
        // if(f_id < f_n)
        {
            bool search_iso_incre = true;
            double max_consistence = 0;

            IterativeSearch( mesh_points, mesh_faces, face_interPlane_normals,
            face_gradients, pt_vals, all_mesh_edges, pid_eid_map, 
            pid_fid_map,edge_hash_face_id_map, pid_fid_map, edge_id_map,
            face_edge_signs, pt_face_grad_sign_map, seed_pt, search_iso_incre, 
            max_consistence);
            max_consistence = max_consistence > 0 ? max_consistence : 0;
            
            search_iso_incre = false;
            double min_consistence = 0;
            IterativeSearch( mesh_points, mesh_faces, face_interPlane_normals,
            face_gradients, pt_vals, all_mesh_edges, pid_eid_map, 
            pid_fid_map,edge_hash_face_id_map, pid_fid_map, edge_id_map,
            face_edge_signs, pt_face_grad_sign_map, seed_pt, 
            search_iso_incre, min_consistence);
            min_consistence = min_consistence < 0 ? min_consistence : 0;

            // std::cout << " max val : " << max_consistence << " , min val : " << min_consistence << std::endl;
            cuv_pt_consistence[i] = max_consistence + abs(min_consistence);
            
        }
        // break;
    }
    SavePointsWithQualityToPLY(curve_path, curve_pts, cuv_pt_consistence);
}


bool VIPSSRidges::CalculateEdgeCreasePoints()
{
    size_t e_num = edges_.size();
    edge_signs_.resize(e_num, -1);
    edge_int_pts_.clear();
    // edge_point_types_.clear();
    edge_eig_vals_.clear();
    int interp_p_id = 0;
    for(int i = 0; i < e_num; ++i)
    {
        const auto& cur_e = edges_[i];
        // arma::vec eig_va = arma::normalise(point_eig_vecs_[cur_e[0]]) * point_eig_vals_[cur_e[0]];  
        // arma::vec eig_vb = arma::normalise(point_eig_vecs_[cur_e[1]]) * point_eig_vals_[cur_e[1]];
        arma::vec eig_va = point_eig_vecs_[cur_e[0]] ;  
        arma::vec eig_vb = point_eig_vecs_[cur_e[1]] ;
        if(arma::dot(eig_va, eig_vb) < 0)
        {
            eig_vb *= -1.0;
        }  
        double crease_val_a = arma::dot(eig_va, point_graidents_[cur_e[0]]);
        double crease_val_b = arma::dot(eig_vb, point_graidents_[cur_e[1]]);
        
        double dot_val = crease_val_a * crease_val_b;
        // std::cout << " edge pid " << cur_e[0] << " " << cur_e[1] << "  dot val " << dot_val << std::endl;
        if(dot_val < 0)
        {
            const auto& pa = mesh_points_[cur_e[0]]; 
            const auto& pb = mesh_points_[cur_e[1]];
            // double va = crease_values_[cur_e[0]];
            // double vb = crease_values_[cur_e[1]];
            double len = abs(crease_val_a) + abs(crease_val_b);
            if(len > 1e-20)
            {
                double t = (- crease_val_a) / (crease_val_b - crease_val_a);
                double px = pa[0] + t * (pb[0] - pa[0]);
                double py = pa[1] + t * (pb[1] - pa[1]);
                double pz = pa[2] + t * (pb[2] - pa[2]);
                
                // std::cout << " interp p id " << interp_p_id << std::endl;
                if(hrfb_ptr_)
                {
                    arma::mat hessian = arma::zeros(3, 3);
                    hrfb_ptr_->EvaluateHessian(px, py, pz, hessian);
                    // hessian = computeHessian(hrfb_ptr_, )
                    arma::vec eigval;
                    arma::mat eigvec;
                    arma::mat H_sym = 0.5 * (hessian + hessian.t());
                    arma::eig_sym(eigval, eigvec, H_sym);
                    arma::vec eigval_abs = arma::abs(eigval);
                    arma::uvec val_ids =  arma::sort_index(eigval_abs);
                    size_t max_id = arma::index_max(eigval_abs);
                    
                    // double ratio = abs(eigval_abs[max_id]/arma::accu(eigval_abs));
                    double ratio = abs(eigval_abs[val_ids[2]]/eigval_abs[val_ids[1]]);
                    

                    if(eigval_abs[max_id] > 0.2)
                    {
                        Point cur_p{px, py, pz};
                        // std::cout << "eigen val size " << eigval.size() << std::endl;
                        TransformEclips(eigval, eigvec, cur_p);

                        edge_int_pts_.push_back({px, py, pz});
                        edge_signs_[i] = interp_p_id;
                        interp_p_id ++; 
                        edge_eig_vals_.push_back(eigval[max_id]);
                        edge_eig_abs_vals_.push_back(eigval_abs[max_id]);
                        edge_eig_val_ratios_.push_back(ratio);
                        double r_threshold = 1.6;
                        if(eigval[max_id] > 0)
                        {
                            edge_pt_color_.push_back({255, 0, 0});
                        } else {
                            edge_pt_color_.push_back({0, 0, 255});
                        }
                    }
                    // double ratio = eigval_abs[max_id];
                }
            }
        }
    }
    return true;
}

bool VIPSSRidges::CalculateFaceCreaseEdge(int f_id)
{
    const auto& cur_f = mesh_faces_[f_id];
    string token_ab = CalEdgeToken(cur_f[0], cur_f[1]);
    string token_bc = CalEdgeToken(cur_f[1], cur_f[2]);
    string token_ca = CalEdgeToken(cur_f[2], cur_f[0]);
    int sum_sign = 0; 
    std::vector<size_t> new_e_ids; 
    // if(edge_id_map_.find(token_ab) != edge_id_map_.end())
    {
        size_t e_ab_id = edge_id_map_[token_ab];
        if(edge_signs_[e_ab_id] >= 0)
        {
            sum_sign ++;
            new_e_ids.push_back(edge_signs_[e_ab_id]);
        }
    }
    // if(edge_id_map_.find(token_bc) != edge_id_map_.end())
    {
        size_t e_bc_id = edge_id_map_[token_bc];
        if(edge_signs_[e_bc_id] >= 0)
        {
            sum_sign ++;
            new_e_ids.push_back(edge_signs_[e_bc_id]);
        }
    }
    // if(edge_id_map_.find(token_ca) != edge_id_map_.end())
    {
        size_t e_ca_id = edge_id_map_[token_ca];
        if(edge_signs_[e_ca_id] >= 0)
        {
            sum_sign ++;
            new_e_ids.push_back(edge_signs_[e_ca_id]);
        }
    }
    if(sum_sign == 2)
    {
        ridge_edges_.push_back({new_e_ids[0], new_e_ids[1]});
    }
    if(sum_sign == 3)
    {
        const auto& pa = edge_int_pts_[new_e_ids[0]];
        const auto& pb = edge_int_pts_[new_e_ids[1]];
        const auto& pc = edge_int_pts_[new_e_ids[2]];
        double px = (pa[0] + pb[0] + pc[0])/3.0;
        double py = (pa[1] + pb[1] + pc[1])/3.0;
        double pz = (pa[2] + pb[2] + pc[2])/3.0;
        size_t new_pid = edge_int_pts_.size(); 
        edge_int_pts_.push_back({px, py, pz});
        ridge_edges_.push_back({new_e_ids[0], new_pid});
        ridge_edges_.push_back({new_e_ids[1], new_pid});
        ridge_edges_.push_back({new_e_ids[2], new_pid});
        double ratio = (edge_eig_val_ratios_[new_e_ids[0]] 
                        + edge_eig_val_ratios_[new_e_ids[1]] 
                        + edge_eig_val_ratios_[new_e_ids[2]]) / 3.0;
        edge_eig_val_ratios_.push_back(ratio);   
        
        double abs_val = (edge_eig_abs_vals_[new_e_ids[0]] 
                        + edge_eig_abs_vals_[new_e_ids[1]]
                        + edge_eig_abs_vals_[new_e_ids[2]]) / 3.0;
        edge_eig_abs_vals_.push_back(abs_val);  

        
    }
    return true;
}


void VIPSSRidges::BuildClusterMST()
{
    std::set<std::string> visited_edge_ids;
    std::vector<C_Edege> tree_edges;
    auto cmp = [](const C_Edege& left, const C_Edege& right) { return left.score_ > right.score_; };
    std::priority_queue<C_Edege, std::vector<C_Edege>, decltype(cmp)> edge_priority_queue(cmp);

    C_Edege st_e(0,0);
    edge_priority_queue.push(st_e);
    std::unordered_set<size_t> visited_vids;
    while(!edge_priority_queue.empty())
    {
        C_Edege cur_e = edge_priority_queue.top();
        edge_priority_queue.pop();

        if(visited_vids.find(cur_e.c_b_ ) != visited_vids.end()) continue;
        visited_vids.insert(cur_e.c_b_);
        if(cur_e.c_a_ != cur_e.c_b_)
        {
            tree_edges.push_back(cur_e);
        }
        size_t cur_pid = cur_e.c_b_ ;
        const auto& p_n_ids = pt_adj_vec_[cur_pid];
        for(const auto& n_id : p_n_ids)
        {
            if(visited_vids.find(n_id) != visited_vids.end()) continue;
            if(n_id == cur_pid) continue;
            C_Edege edge(cur_pid, n_id);
            double nor_diff =1.0 - arma::dot(point_graidents_[cur_pid], point_graidents_[n_id]);

            double dist = PointDist(mesh_points_[n_id], mesh_points_[cur_pid]);
            edge.score_ = dist * nor_diff; 
            edge_priority_queue.push(edge);
        }
    }
    size_t p_num = mesh_points_.size();
    cluster_MST_mat_.resize(p_num, p_num);
    std::vector<TripletInt> edge_eles(tree_edges.size() *2);
    auto e_iter = edge_eles.begin();
    for(const auto& edge: tree_edges)
    {
        size_t i = edge.c_a_;
        size_t j = edge.c_b_;
        *(e_iter ++) = TripletInt(i,j,1);
        *(e_iter ++) = TripletInt(j,i,1);
    }
    cluster_MST_mat_.setFromTriplets(edge_eles.begin(), edge_eles.end());
    // cluster_scores_mat_.colw;
}


void VIPSSRidges::FlipEigenVectorByMST()
{
    // size_t c_num = cluster_cores_mat_.n_cols;
    size_t p_num = mesh_points_.size();
    std::queue<size_t> cluster_queued_ids;
    cluster_queued_ids.push(0);
    std::set<size_t> visited_cluster_ids;
    std::set<size_t> flipped_cluster_ids;
    flipped_cluster_ids.insert(0);
    while(!cluster_queued_ids.empty())
    {
        size_t cur_cid = cluster_queued_ids.front();
        cluster_queued_ids.pop();
        if(visited_cluster_ids.find(cur_cid) != visited_cluster_ids.end()) continue;
        for(SpiMat::InnerIterator iter(cluster_MST_mat_, cur_cid); iter ; ++iter)
        {   
            size_t n_cid = iter.row();
            if(flipped_cluster_ids.find(n_cid) != flipped_cluster_ids.end()) continue;
            flipped_cluster_ids.insert(n_cid);
            double dot_val = arma::dot(point_eig_vecs_[cur_cid], point_eig_vecs_[n_cid]);
            if(dot_val < 0)
            {
                point_eig_vecs_[n_cid] *= -1.0;
            }
            cluster_queued_ids.push(n_cid);
        }
    }
}

bool VIPSSRidges::CalculateRidgeEdgesFromMesh()
{
    // if(point_graidents_.empty())
    // {
    //     std::cout << "error:  No mesh grident!! " << std::endl;
    //     return false;
    // }
    if(point_eig_vecs_.empty())
    {
        std::cout << "error:  No mesh point eigen vectors!! " << std::endl;
        return false;
    }
    BuildPtAdjInfo();
    // FlipEigenVector();
    BuildClusterMST();
    FlipEigenVectorByMST();

    std::string out_eigvec_path = out_dir_ + "/" + file_name_ + "_eig_vec_l" + std::to_string(user_lambda_) + ".xyz";
    SavePointsNormalToXYZ(out_eigvec_path, mesh_points_, point_eig_vecs_);
    std::string out_gradient_path = out_dir_ + "/" + file_name_ + "_gradient_l" + std::to_string(user_lambda_)+ ".xyz";
    SavePointsNormalToXYZ(out_gradient_path, mesh_points_, point_graidents_);

    std::cout << "start to CalculateCreaseValues !! " << std::endl;
    CalculateCreaseValues();
    std::cout << "start to CalculateEdgeCreasePoints  !! " << std::endl;
    GetEdges();
    CalculateEdgeCreasePoints();
    // std::string out_path =  "int_pts.obj";
    // SaveRidgesToObj(out_path);
    size_t face_num = mesh_faces_.size();
    std::cout << "start to CalculateFaceCreaseEdge  !! " << std::endl;
    for(int i = 0; i < face_num; ++i)
    {
        CalculateFaceCreaseEdge(i);
    }
    std::cout << "finished  CalculateFaceCreaseEdge  !! " << std::endl;
    return true;
}


bool VIPSSRidges::CalculateRidgeEdgesFromMesh2()
{
    // if(point_graidents_.empty())
    // {
    //     std::cout << "error:  No mesh grident!! " << std::endl;
    //     return false;
    // }
    // std::cout << "error:  No mesh grident!! " << std::endl;
    CalMeshPointsGradient(hrfb_ptr_, mesh_points_, point_graidents_);
    std::string out_gradient_path = out_dir_ + "/" + file_name_ + "_gradient_l" + std::to_string(user_lambda_)+ ".xyz";
    SavePointsNormalToXYZ(out_gradient_path, mesh_points_, point_graidents_);
    CalMeshPointsCurvature(hrfb_ptr_, mesh_points_, point_graidents_, mesh_pt_curvatures_);

    GetEdges();
    CalculateEdgeRidgeValleyPoints();
    
    // std::string out_path =  "int_pts.obj";
    // SaveRidgesToObj(out_path);
    size_t face_num = mesh_faces_.size();
    std::cout << "start to CalculateFaceCreaseEdge  !! " << std::endl;
    for(int i = 0; i < face_num; ++i)
    {
        CalculateRidegeEdges(mesh_faces_[i], edge_id_map_, edge_ridge_signs_, edge_ridge_pts_,edge_ridge_pts_curvature_, out_ridge_edges_);
        CalculateRidegeEdges(mesh_faces_[i], edge_id_map_, edge_valley_signs_, edge_valley_pts_,edge_valley_pts_curvature_, out_valley_edges_);
        CalculateRidegeEdges(mesh_faces_[i], edge_id_map_, edge_gaussian_signs_, edge_gaussian_pts_,edge_gaussian_curvature_, out_gaussian_edges_);

    }
    std::cout << "finished  CalculateFaceCreaseEdge  !! " << std::endl;
    return true;
}


void VIPSSRidges::BuildPtAdjInfo()
{
    size_t f_num = mesh_faces_.size();
    size_t p_num = mesh_points_.size();
    pt_adj_vec_.resize(p_num);
    for(int i = 0; i < int(f_num); ++i)
    {
        const auto& cur_f = mesh_faces_[i];
        for(const auto pid : cur_f)
        {
            for(const auto new_pid : cur_f)
            {
                pt_adj_vec_[pid].insert(new_pid);
            }
        }
    }
}

void VIPSSRidges::FlipEigenVector()
{
    std::vector<int> pt_visited;
    size_t p_num = mesh_points_.size();
    pt_visited.resize(p_num);
    std::stack<size_t> candid_pids;
    candid_pids.push(0);
    pt_visited[0] = 1;
    while(!candid_pids.empty())
    {
        size_t cur_pid = candid_pids.top();
        candid_pids.pop();
        const auto& neis = pt_adj_vec_[cur_pid];
        const Vec& cur_eigvec = point_eig_vecs_[cur_pid];
        for(const auto n_id : neis)
        {
            if(pt_visited[n_id]) continue;
            Vec& n_eigvec = point_eig_vecs_[n_id];
            if(arma::dot(cur_eigvec, n_eigvec) < 0)
            {
                n_eigvec *= -1.0;
            }
            candid_pids.push(n_id);
            pt_visited[n_id] = 1;
        }
    }
}

void VIPSSRidges::CalEdgePointQuality(LocalVipss* local_vipss)
{
    edge_points_quality_.resize(edge_int_pts_.size());
    for(int i = 0; i < edge_int_pts_.size(); ++i)
    {
        const auto& pt = edge_int_pts_[i]; 
        double cur_pt[3] = {pt[0], pt[1], pt[2]};
        arma::mat hessian = local_vipss->NNHRBFHessianOMP(cur_pt) * (-1.0);
        // std::cout << hessian << std::endl;
        arma::vec eigval;
        arma::eig_sym(eigval, hessian);
        arma::uvec indices = arma::sort_index(eigval, "descend");
        // if(eigval[indices[0]] > abs(eigval[indices[2]]))
        if(eigval[indices[0]] > 0)
        {
            edge_points_quality_[i] = 1.0;
        } else {
            edge_points_quality_[i] = 0.0;
        }
    }
}

void VIPSSRidges::SaveRidgesToObj(const std::string& out_path)
{
    std::ofstream objFile(out_path);
    if (!objFile) {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return;
    }
    // Write vertices
    for (const auto& point : edge_int_pts_) {
        double px = point[0] * scale_ + ori_center_[0];
        double py = point[1] * scale_ + ori_center_[1];
        double pz = point[2] * scale_ + ori_center_[2];
        objFile << "v " << px << " " << py << " " << pz << "\n";
    }
    // Write edges as line elements (OBJ uses 'l' for lines)
    for (const auto& edge : ridge_edges_) {
        objFile << "l " << edge[0] + 1 << " " << edge[1] + 1 << "\n";  // OBJ uses 1-based indexing
    }
    objFile.close();
    std::cout << "OBJ file saved successfully: " << out_path << std::endl;
}


void VIPSSRidges::SaveRidgesToObj(const std::string& out_path, 
       const std::vector<Point>& edge_int_pts, 
       const std::vector<std::vector<size_t>>& ridge_edges,
        double scale = 1.0, Point ori_center = {0, 0, 0})
{
    std::ofstream objFile(out_path);
    if (!objFile) {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return;
    }
    // Write vertices
    for (const auto& point : edge_int_pts) {
        double px = point[0] * scale + ori_center[0];
        double py = point[1] * scale + ori_center[1];
        double pz = point[2] * scale + ori_center[2];
        objFile << "v " << px << " " << py << " " << pz << "\n";
    }
    // Write edges as line elements (OBJ uses 'l' for lines)
    for (const auto& edge : ridge_edges) {
        objFile << "l " << edge[0] + 1 << " " << edge[1] + 1 << "\n";  // OBJ uses 1-based indexing
    }
    objFile.close();
    std::cout << "OBJ file saved successfully: " << out_path << std::endl;
}

void VIPSSRidges::SaveRidgesToObjWithColor(const std::string& out_path, 
       const std::vector<Point>& edge_int_pts, 
       const std::vector<std::vector<size_t>>& ridge_edges,
       const std::array<double,3>& color,
        double scale = 1.0, Point ori_center = {0, 0, 0})
{
    std::ofstream objFile(out_path);
    if (!objFile) {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return;
    }
    // Write vertices
    for (const auto& point : edge_int_pts) {
        double px = point[0] * scale + ori_center[0];
        double py = point[1] * scale + ori_center[1];
        double pz = point[2] * scale + ori_center[2];
        objFile << "v " << px << " " << py << " " << pz << " " 
                <<  color[0] << " " << color[1] << " " << color[2] <<  "\n";
    }
    // Write edges as line elements (OBJ uses 'l' for lines)
    for (const auto& edge : ridge_edges) {
        objFile << "l " << edge[0] + 1 << " " << edge[1] + 1 << "\n";  // OBJ uses 1-based indexing
    }
    objFile.close();
    std::cout << "OBJ file saved successfully: " << out_path << std::endl;
}



void VIPSSRidges::SaveRidgesWithColorToPLY(const std::string& filename) {
    //  const std::vector<Point>& points, const std::vector<Edge>& edges
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }
    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << edge_int_pts_.size() + ridge_edges_.size()<< "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "element face " << ridge_edges_.size() << "\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";

    // std::cout << "edge_point_types_ size " << edge_point_types_.size() << std::endl;
    // Write vertex data
    for (int i =0; i < edge_int_pts_.size(); ++i) {
        const auto& point = edge_int_pts_[i];
        double px = point[0] * scale_ + ori_center_[0];
        double py = point[1] * scale_ + ori_center_[1];
        double pz = point[2] * scale_ + ori_center_[2];
        file << px << " " << py << " " << pz << " " ;
        const auto& color = edge_pt_color_[i];
        file << color[0] << " " << color[1] << " " << color[2] << " "  << "\n";
    }
    // Write edge data
    size_t pid = edge_int_pts_.size();
    std::vector<std::array<size_t,3>> faces;
    for (const auto& e : ridge_edges_) {
        
        faces.push_back({e[0], e[1], pid});
        pid ++;
        const auto& p0 = edge_int_pts_[e[0]];
        const auto& p1 = edge_int_pts_[e[1]];
        double px = (p0[0] + p1[0])/ 2.0; 
        double py = (p0[1] + p1[1])/ 2.0; 
        double pz = (p0[2] + p1[2])/ 2.0; 
        file << px << " " << py << " " << pz << " " ;
        double e_v0 = edge_eig_vals_[e[0]];
        double e_v1 = edge_eig_vals_[e[1]];
        std::string color_str = "0 0 0";
        const auto& color0 = edge_pt_color_[e[0]];
        const auto& color1 = edge_pt_color_[e[1]];
        if(abs(e_v0) >= abs(e_v1))
        {
            file << color0[0] << " " << color0[1] << " " << color0[2] << " " << "\n";
        } else {
            file << color1[0] << " " << color1[1] << " " << color1[2] << " " << "\n";
        }
    }
    for(const auto& face : faces)
    {
        file << "3 " << face[0] << " " << face[1] << " " << face[2] << std::endl;
    }
    file.close();
    std::cout << "PLY file saved: " << filename << std::endl;
}

void VIPSSRidges::SaveRidgesWithColorToPLY(const std::string& filename,
    const std::vector<Point>& pts, 
    const std::vector<std::vector<size_t>>& edges, 
    const std::array<double, 3>& edge_color) {
    const std::array<int, 3> color = {int(edge_color[0] * 255),int(edge_color[1] * 255), int(edge_color[2] * 255) };
    //  const std::vector<Point>& points, const std::vector<Edge>& edges
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }
    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << pts.size() + edges.size()<< "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "element face " << edges.size() << "\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";

    // std::cout << "edge_point_types_ size " << edge_point_types_.size() << std::endl;
    // Write vertex data
    for (int i =0; i < pts.size(); ++i) {
        const auto& point = pts[i];
        double px = point[0] * scale_ + ori_center_[0];
        double py = point[1] * scale_ + ori_center_[1];
        double pz = point[2] * scale_ + ori_center_[2];
        file << px << " " << py << " " << pz << " " ;
        file << color[0] << " " << color[1] << " " << color[2] << " "  << "\n";
    }
    // Write edge data
    size_t pid = pts.size();
    std::vector<std::array<size_t,3>> faces;
    for (const auto& e : edges) {
        
        faces.push_back({e[0], e[1], pid});
        pid ++;
        const auto& p0 = pts[e[0]];
        const auto& p1 = pts[e[1]];
        double px = (p0[0] + p1[0])/ 2.0; 
        double py = (p0[1] + p1[1])/ 2.0; 
        double pz = (p0[2] + p1[2])/ 2.0; 
        file << px << " " << py << " " << pz << " " ;
        file << color[0] << " " << color[1] << " " << color[2] << " " << "\n";
       
    }
    for(const auto& face : faces)
    {
        file << "3 " << face[0] << " " << face[1] << " " << face[2] << std::endl;
    }
    file.close();
    std::cout << "PLY file saved: " << filename << std::endl;
}






void VIPSSRidges::SaveRidgesWithQualityToPLY(const std::string& filename, const std::vector<double>& qualtity) {
    //  const std::vector<Point>& points, const std::vector<Edge>& edges
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << edge_int_pts_.size() + ridge_edges_.size()<< "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property float quality\n";
    file << "element face " << ridge_edges_.size() << "\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";
    // Write vertex data
    for (int i =0; i < edge_int_pts_.size(); ++i) {
        const auto& point = edge_int_pts_[i];
        double px = point[0] * scale_ + ori_center_[0];
        double py = point[1] * scale_ + ori_center_[1];
        double pz = point[2] * scale_ + ori_center_[2];
        file << px << " " << py << " " << pz << " " << qualtity[i] << "\n";
    }
    // Write edge data
    size_t pid = edge_int_pts_.size();
    std::vector<std::array<size_t,3>> faces;
    for (const auto& e : ridge_edges_) {
        
        faces.push_back({e[0], e[1], pid});
        pid ++;
        const auto& p0 = edge_int_pts_[e[0]];
        const auto& p1 = edge_int_pts_[e[1]];
        double px = (p0[0] + p1[0])/ 2.0; 
        double py = (p0[1] + p1[1])/ 2.0; 
        double pz = (p0[2] + p1[2])/ 2.0; 
        file << px << " " << py << " " << pz << " " ;
        double e_v0 = edge_eig_vals_[e[0]];
        double e_v1 = edge_eig_vals_[e[1]];


        const auto& color0 = edge_pt_color_[e[0]];
        const auto& color1 = edge_pt_color_[e[1]];
        if(abs(e_v0) >= abs(e_v1))
        {
            file << qualtity[e[0]]<< "\n";
        } else {
            file << qualtity[e[1]] << "\n";
        }
    }

    for(const auto& face : faces)
    {
        file << "3 " << face[0] << " " << face[1] << " " << face[2] << std::endl;
    }

    file.close();
    std::cout << "PLY file saved: " << filename << std::endl;
}


void VIPSSRidges::SaveMeshWithPointQuality(const std::string& mesh_path)
{
    std::vector<Point> out_points;
    for(const auto& p : mesh_points_)
    {
        double px = p[0] * scale_ + ori_center_[0];
        double py = p[1] * scale_ + ori_center_[1];
        double pz = p[2] * scale_ + ori_center_[2];
        out_points.push_back({px, py, pz});
    }
    SaveMeshWithQualityToPly(mesh_path, out_points, crease_values_, mesh_faces_);
}

void VIPSSRidges::SaveEigBallsMesh(const std::string& mesh_path)
{
    std::vector<Point> out_points;
    for(const auto& p : eig_ball_pts_)
    {
        double px = p[0] * scale_ + ori_center_[0];
        double py = p[1] * scale_ + ori_center_[1];
        double pz = p[2] * scale_ + ori_center_[2];
        out_points.push_back({px, py, pz});
    }
    // SaveMeshToPly(mesh_path, out_points, eig_ball_faces_);
    SaveMeshToPly(mesh_path, out_points, eig_ball_pts_quality_, eig_ball_faces_);
}



void VIPSSRidges::SavePointsNormalToXYZ(const std::string& out_path, 
                                const std::vector<Point>& points,
                                const std::vector<Vec>& normals)
{
    //  const std::vector<Point>& points, const std::vector<Edge>& edges
    std::ofstream file(out_path);
    if (!file) {
        std::cerr << "Error: Unable to open file " << out_path << std::endl;
        return;
    }
    // Write vertex data
    for (int i =0; i < points.size(); ++i) {
        const auto& point = points[i];
        double px = point[0] * scale_ + ori_center_[0];
        double py = point[1] * scale_ + ori_center_[1];
        double pz = point[2] * scale_ + ori_center_[2];
        const auto& vec = normals[i];
        file << px << " " << py << " " << pz  << " " << vec[0] << " " << vec[1] << " " << vec[2]<< "\n";
    }
    file.close();
    std::cout << "PLY file saved: " << out_path << std::endl;
}

void VIPSSRidges::TransformEclips(const arma::vec& eigvals, const arma::mat& eigen_vectors, const Point& cur_pt)
{
    // arma::mat rotation = eigen_vectors.t();
    // arma::mat rotation = eigen_vectors.t();
    // rotation.col(2) = arma::normalise(arma::cross(rotation.col(0), rotation.col(1)));
    // rotation = arma::inv(rotation);
    // Orthonormalize eigenvectors using QR decomposition
    // std::cout << " eigen vector mat " << eigen_vectors << std::endl;
    arma::mat Q, R;
    arma::qr(Q, R, eigen_vectors);

    // Ensure determinant is +1 for a proper rotation matrix
    if (det(Q) < 0) {
        Q.col(2) *= -1;  // Flip the last column
    }
    // std::cout << " Q vector mat " << Q << std::endl;
    // arma::mat rotation = arma::inv(Q); 
    arma::mat rotation = Q;

    // std::cout << " transform mat " << rotation << std::endl;
    // std::cout << " xy dot " << arma::dot(rotation.col(0), rotation.col(1)) << std::endl;
    // std::cout << " yz dot " << arma::dot(rotation.col(1), rotation.col(2)) << std::endl;
    // std::cout << " zx dot " << arma::dot(rotation.col(2), rotation.col(0)) << std::endl;
    // std::vector<Point> final_pts;
    arma::vec center{cur_pt[0], cur_pt[1], cur_pt[2]};
    double g_scale = 0.002;
    // arma::vec scale = arma::abs(eigvals) / arma::max(arma::abs(eigvals)) * g_scale;
    arma::vec scale = arma::abs(eigvals) / 20.0 * g_scale;
    arma::mat scale_mat = arma::diagmat(scale);
    double ratio = arma::max(arma::abs(eigvals)) / arma::accu(arma::abs(eigvals));
    // std::cout << " scale mat " << scale_mat << std::endl;
    size_t pt_size = eig_ball_pts_.size(); 
    for(const auto& pt: ball_pts_)
    {
        arma::vec in_pt{pt[0], pt[1], pt[2]};
        arma::vec out_pt =  (rotation * (scale_mat * in_pt)) + center;
        eig_ball_pts_.push_back({out_pt[0], out_pt[1], out_pt[2]});
        eig_ball_pts_quality_.push_back(ratio);
    }
    for(const auto& face:  ball_faces_)
    {
        std::vector<size_t> new_face;
        for(const auto pid : face)
        {
            new_face.push_back(pid + pt_size);
        }
        eig_ball_faces_.push_back(new_face);
    }
    // return final_pts;
}

void VIPSSRidges::ProjectMeshPtsToSurface(std::vector<Point>& mesh_points, std::shared_ptr<RBF_Core> hrfb_ptr)
{
    const double threshold = 1e-15;
    const int max_iter = 10;
    // #pragma omp parallel for 
    for(int i = 0; i < mesh_points.size(); ++i)
    {
        
        arma::vec3 g;
        arma::vec3 new_pt = {mesh_points[i][0], mesh_points[i][1], mesh_points[i][2]} ; 
        
        double func_val = hrfb_ptr->Dist_Function(R3Pt(new_pt[0], new_pt[1], new_pt[2]));
        int id = 0;
        while(abs(func_val) > threshold && id < max_iter)
        {
            hrfb_ptr->evaluate_gradient(new_pt[0], new_pt[1], new_pt[2], g[0], g[1], g[2]);
            double g_norm = arma::dot(g, g);
            new_pt = new_pt - func_val/ g_norm * g;
            func_val = hrfb_ptr->Dist_Function(R3Pt(new_pt[0], new_pt[1], new_pt[2]));
            id ++;
        } 
        mesh_points[i][0] = new_pt[0];
        mesh_points[i][1] = new_pt[1];
        mesh_points[i][2] = new_pt[2];
    } 
}

 