#pragma once
#include <span>
#include <queue>
#include <optional>
#include "local_vipss.hpp"
#include <CLI/CLI.hpp>

// #include "adgrid/timer.h"
// #include "adgrid/csg.h"
// #include "adgrid/grid_mesh.h"
// #include "adgrid/grid_refine.h"
// #include "adgrid/external/implicit_functions/implicit_functions.h"
// #include "adgrid/external/implicit_functions/Hermite_RBF.h"

#include "implicit_functions/implicit_functions.h"

class HRBFDistanceFunction : public ImplicitFunction<double>
{
public:
    HRBFDistanceFunction()
    {
        
    }

    HRBFDistanceFunction(double offset): offset_(offset)
    {
        
    }

    double evaluate(double x, double y, double z) const override
    {
        // std::cout << "eval pt : " << x << " " << y << " " << z << std::endl;
        R3Pt newPt(x, y, z);
        return LocalVipss::NNDistFunction(newPt) - offset_;
    }

    double evaluate_gradient(double x, double y, double z, double &gx, double &gy, double &gz) const override
    {
        // std::cout << "pt : " << x << " " << y << " " << z << std::endl;
        R3Pt newPt(x, y, z);
        // double dist_val0 = LocalVipss::NNDistFunction(newPt) - offset_;
        double gradient[3];
        double dist_val = LocalVipss::NNDistGradient(newPt, gradient) - offset_;

        // std::cout << "dist_val0 : " << dist_val0 << "  dist_val : " << dist_val << std::endl;
        
        gx = gradient[0];
        gy = gradient[1];
        gz = gradient[2];

        // // Partial derivative with respect to x 
        // double h = 0.00001;
        // double fxa = LocalVipss::NNDistFunction({x + h, y, z});
        // double fxb = LocalVipss::NNDistFunction({x - h, y, z});
        // gx = (fxa - fxb) / (2.0 * h);
        
        // // Partial derivative with respect to y
        // double fya = LocalVipss::NNDistFunction({x, y + h, z});
        // double fyb = LocalVipss::NNDistFunction({x, y - h, z});
        // gy = (fya - fyb) / (2.0 * h);
        
        // // Partial derivative with respect to z
        // double fza = LocalVipss::NNDistFunction({x, y, z + h});
        // double fzb = LocalVipss::NNDistFunction({x, y, z - h});
        // gz = (fza - fzb) / (2.0 * h);

        // std::cout << "analytic dist_val : " << dist_val << std::endl;
        // double dist_val = LocalVipss::NNDistFunction(newPt);
        // // std::cout << "numerical dist_val : " << dist_val << std::endl;
        // double delt = 1e-12; 
        // R3Pt newPt_xp(x + delt, y, z);
        // double dist_xp = LocalVipss::NNDistFunction(newPt_xp);
        // R3Pt newPt_xn(x - delt, y, z);
        // double dist_xn = LocalVipss::NNDistFunction(newPt_xn);

        // R3Pt newPt_yp(x, y + delt, z);
        // double dist_yp = LocalVipss::NNDistFunction(newPt_yp);
        // R3Pt newPt_yn(x, y - delt, z);
        // double dist_yn = LocalVipss::NNDistFunction(newPt_yn);

        // R3Pt newPt_zp(x, y, z + delt);
        // double dist_zp = LocalVipss::NNDistFunction(newPt_zp);
        // R3Pt newPt_zn(x, y, z - delt);
        // double dist_zn = LocalVipss::NNDistFunction(newPt_zn);

        // gx = (dist_xp - dist_xn) / (2 *delt);
        // gy = (dist_yp - dist_yn) / (2 *delt);
        // gz = (dist_zp - dist_zn) / (2 *delt);

        // std::cout << "n grad : " << n_gx << " " << n_gy << " " << n_gz << std::endl;
        // std::cout << "analytic grad : " << gx << " " << gy << " " << gz << std::endl;
        return dist_val;
    }

    // double evaluate_gradient(double x, double y, double z, double &gx, double &gy, double &gz) const override
    // {
    //     double delt = 1e-6;
    //     // std::cout << "pt : " << x << " " << y << " " << z << std::endl;
    //     R3Pt newPt(x, y, z);
    //     double dist_val = LocalVipss::NNDistFunction(newPt);
    //     // std::cout << "dist_val : " << dist_val << std::endl;
    //     R3Pt newPt_x(x + delt, y, z);
    //     double dist_x = LocalVipss::NNDistFunction(newPt_x);
    //     R3Pt newPt_y(x, y + delt, z);
    //     double dist_y = LocalVipss::NNDistFunction(newPt_y);
    //     R3Pt newPt_z(x, y, z + delt);
    //     double dist_z = LocalVipss::NNDistFunction(newPt_z);

    //     gx = (dist_x - dist_val) / delt;
    //     gy = (dist_y - dist_val) / delt;
    //     gz = (dist_z - dist_val) / delt;

    //     // std::cout << "grad : " << gx << " " << gy << " " << gz << std::endl;
    //     return dist_val;
    // }

    void SetIsoOffset(double offset_val)
    {
        offset_ = offset_val;
    }
    

private:
    double radius_;
    double offset_ = 0.0;
};


void GenerateAdaptiveGridOut(const std::array<size_t, 3>& resolution, 
                            const std::array<double, 3>& bbox_min,
                            const std::array<double, 3>& bbox_max,
                            const std::string& outdir,
                            const std::string& fillname, 
                            std::vector<shared_ptr<ImplicitFunction<double>>>& functions,
                            double threshold,
                            std::vector<std::array<double, 3> >& output_vertices,
                            std::vector<std::array<size_t, 3> >& output_triangles);

