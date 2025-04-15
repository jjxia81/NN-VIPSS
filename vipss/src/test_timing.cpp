#include "test_timing.h"

namespace test_vipss_timing {


void cal_bbox(const std::vector<double>&pts, Point& min_corner, Point& max_corner)
{
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();

    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    double max_z = std::numeric_limits<double>::lowest();

    const auto& in_pts = pts;
    for(size_t i =0; i < in_pts.size()/3; ++i)
    {
        min_x = min_x < in_pts[3*i]     ? min_x : in_pts[3*i];
        min_y = min_y < in_pts[3*i + 1] ? min_y : in_pts[3*i + 1];
        min_z = min_z < in_pts[3*i + 2] ? min_z : in_pts[3*i + 2];

        max_x = max_x > in_pts[3*i]     ? max_x : in_pts[3*i];
        max_y = max_y > in_pts[3*i + 1] ? max_y : in_pts[3*i + 1];
        max_z = max_z > in_pts[3*i + 2] ? max_z : in_pts[3*i + 2];
    }
    min_corner = {min_x, min_y, min_z};
    max_corner = {max_x, max_y, max_z};
}

void generate_test_volume_pts(const std::vector<double>& inpts, std::vector<Point>& grid_pts, int volume_res )
{
    
    Point min_corner;
    Point max_corner;
    cal_bbox(inpts, min_corner, max_corner);
    double scale  = 1.2;
    Point center;
    center[0] = (max_corner[0] + min_corner[0]) / 2.0;
    center[1] = (max_corner[1] + min_corner[1]) / 2.0;
    center[2] = (max_corner[2] + min_corner[2]) / 2.0;
    double dx = max_corner[0] - min_corner[0];
    double dy = max_corner[1] - min_corner[1];
    double dz = max_corner[2] - min_corner[2];
    double sx = center[0] - dx/2.0 * scale;
    double sy = center[1] - dy/2.0 * scale;
    double sz = center[2] - dz/2.0 * scale;
    
    double len_x = dx * scale / double(volume_res);
    double len_y = dy * scale / double(volume_res);
    double len_z = dz * scale / double(volume_res);  

    for(int i = 0; i < volume_res; ++i)
    {
        double px = sx + (i + 0.5) * len_x;   
        for(int j = 0; j < volume_res; ++j)
        {
            double py = sy + (j + 0.5) * len_y;
            for(int k = 0; k < volume_res; ++k)
            {
                double pz = sz + (k + 0.5) * len_z;
                grid_pts.push_back(Point{px, py, pz});
            }
        }
    } 
}  

void generate_test_volume_pts_all(const std::string& data_path, std::vector<double>& inpts, 
                                std::vector<Point>& grid_pts, int volume_res)
{
    readXYZ(data_path, inpts);
    generate_test_volume_pts(inpts, grid_pts, volume_res);
}


void test_vipss_timing(const std::string& data_path )
{
    // std::string data_path = "";
    // std::vector<double> in_pts;
    // std::vector<Point> test_pts;
    // std::cout << "inpt size : " << in_pts.size()/3 << std::endl;
    // generate_test_volume_pts_all(data_path, in_pts, test_pts, 20);  
    // std::cout << "inpt size : " << in_pts.size()/3 << std::endl;
    // size_t npt = in_pts.size()/3;
    // arma::vec a = arma::randn(npt * 4);
    // arma::vec b = arma::randn(4); 
    // std::vector<Point> new_pts;
    // for(size_t i = 0; i < npt; ++i)
    // {
    //     Point curPt{in_pts[3*i], in_pts[3*i + 1], in_pts[3*i +2]};
    //     new_pts.push_back(curPt);
    // }
    // auto t0 = Clock::now();
    // for(const auto& pt : test_pts)
    // {
    //     VIPSS_HRBF_Dist_Alone(&(pt[0]), a, b, new_pts);
    // }
    // auto t1 = Clock::now();
    // double call_time = std::chrono::nanoseconds(t1 - t0).count()/1e9;
    // double ave_time = call_time / double(test_pts.size());
    // std::cout << " --------- vipss ave test pt time : " << ave_time << std::endl;
}

std::vector<Point> generate_grid_pts(const Point& min_corner, const Point& max_corner, double py_val = 0, int volume_res = 200)
{
    double scale  = 1.5;
    Point center;
    center[0] = (max_corner[0] + min_corner[0]) / 2.0;
    center[1] = (max_corner[1] + min_corner[1]) / 2.0;
    center[2] = (max_corner[2] + min_corner[2]) / 2.0;
    double dx = max_corner[0] - min_corner[0];
    double dy = max_corner[1] - min_corner[1];
    double dz = max_corner[2] - min_corner[2];
    double sx = center[0] - dx/2.0 * scale;
    double sy = center[1] - dy/2.0 * scale;
    double sz = center[2] - dz/2.0 * scale;
    
    double len_x = dx * scale / double(volume_res);
    double len_y = 0;
    double len_z = dz * scale / double(volume_res);  

    std::vector<Point> layer_grid_pts;
    for(int i =0; i < volume_res; ++i)
    {
        double px = sx + double(i + 0.5) * len_x;
        for(int j = 0; j < volume_res; ++j)
        {
            double pz = sz + double(j + 0.5) * len_z;
            layer_grid_pts.push_back(Point{px, py_val, pz});
        }
    }
    return layer_grid_pts;
}

void visual_grid_pts(const std::string& out_path, const std::vector<Point>& grid_pts, double func(const R3Pt& pt))
{
    std::ofstream layer_file(out_path);
    double max_dist = 0.1;
    for(const auto& pt : grid_pts)
    {
        double dist_val = func(R3Pt(pt[0], pt[1], pt[2]));
        // double new_dist = max()
        double r, g, b = 0;
        if(dist_val > 0)
        {
            r = min(max_dist, dist_val) / max_dist;
            g = (max_dist - min(max_dist, dist_val)) / max_dist;
        } else {
            b = max(- max_dist, dist_val) / max_dist;
            g = (max(- max_dist, dist_val) + max_dist) / max_dist;
        }
        // double r = dist_val 
        layer_file << "v " << pt[0] << " " << pt[1] << " " <<pt[2] ;
        layer_file << " " << r << " " << g << " " << b << std::endl;
        
    }
    layer_file.close();
}

void visual_distval_pt(const std::string& in_path, int volume_res )
{   
    std::vector<double> inpts;
    readXYZ(in_path, inpts);
    Point min_corner;
    Point max_corner;
    cal_bbox(inpts, min_corner, max_corner);     
    auto& rbf_vec = LocalVipss::node_rbf_vec_;
    int input_pt_size = inpts.size()/3;
    // for(int i =0; i < rbf_vec.size(); ++i)
    // {
    //     if(i < input_pt_size) continue;
    //     auto pt = LocalVipss::points_[i];
    //     auto grid_pts = generate_grid_pts(min_corner, max_corner, pt[1]);
    //     double max_dist = 0.1;
    //     std::string grid_out_path = "../../out/planck_layer_" + std::to_string(i) + ".obj";
 
    // }
    
    // std::ofstream box_file("planck_bbox.obj");
    // // box_file << "v " << min_corner[0] << " " << min_corner[1] << " " <<min_corner[2] << std::endl ;
    // box_file << "v " << min_corner[0] << " " << min_corner[1] << " " <<min_corner[2] << std::endl ;
    // box_file << "v " << max_corner[0] 
    //          << " " <<  max_corner[1]
    //          << " " <<  max_corner[2]  << std::endl ;
    // box_file.close();

    std::string planck_grid_path = "../../out/kitten_0.001_layer_grid.obj";
    auto grid_pts = generate_grid_pts(min_corner, max_corner, 0);
    visual_grid_pts(planck_grid_path, grid_pts, LocalVipss::NNDistFunction);

    // double max_dist = 0.1;
    // std::ofstream layer_file("planck_layer.obj");
    // for(const auto& pt : layer_grid_pts)
    // {
    //     double dist_val = LocalVipss::NNDistFunction(R3Pt(pt[0], pt[1], pt[2]));
    //     // double new_dist = max()
    //     double r, g, b = 0;
    //     if(dist_val > 0)
    //     {
    //         r = min(max_dist, dist_val) / max_dist;
    //         g = (max_dist - min(max_dist, dist_val)) / max_dist;
    //     } else {
    //         b = max(- max_dist, dist_val) / max_dist;
    //         g = (max(- max_dist, dist_val) + max_dist) / max_dist;
    //     }
    //     // double r = dist_val 
    //     layer_file << "v " << pt[0] << " " << pt[1] << " " <<pt[2] ;
    //     layer_file << " " << r << " " << g << " " << b << std::endl;
        
    // }

    
}


void test_local_vipss(const std::string& in_path)
{
    std::vector<double> inpts;
    readXYZ(in_path, inpts);
    std::vector<Point> test_pts;
    generate_test_volume_pts(inpts, test_pts, 50);
    // auto t0 = Clock::now();
    std::vector<double> dist_vals(test_pts.size());
    std::vector<double> time_vals(test_pts.size());
    // for(const auto& pt : test_pts)
    for(int i = 0; i < test_pts.size(); ++i)
    {
        auto t00 = Clock::now();
        auto pt = test_pts[i];
        dist_vals[i] = LocalVipss::NNDistFunction(R3Pt(pt[0], pt[1], pt[2]));
        // dist_vals.push_back(dist)
        auto t11 = Clock::now();
        double call_time = std::chrono::nanoseconds(t11 - t00).count()/1e9;
        time_vals[i] = call_time;
        // VIPSS_HRBF_Dist_Alone(&(pt[0]), a, b, new_pts);
    }
    // auto t1 = Clock::now();
    // double call_time = std::chrono::nanoseconds(t1 - t0).count()/1e9;
    // double ave_time = call_time / double(test_pts.size());
    // std::cout << " --------- local vipss  pt time : " << ave_time << std::endl;

    std::ofstream time_file("dist_time.csv");
    for(int i = 0; i < test_pts.size(); ++i)
    {
        time_file << dist_vals[i] << ", " << time_vals[i] << std::endl;
    }
    time_file.close();
    
}


}