#include "local_vipss.hpp"
#include <cmath>
#include <algorithm>
#include "readers.h"  
#include <chrono>
#include <queue>
#include <omp.h>
#include <random>
#include "stats.h"
#include "kernel.h"
#include "SimpleOctree.h"

typedef std::chrono::high_resolution_clock Clock;

double LocalVipss::search_nn_time_sum_ = 0;
double LocalVipss::pass_time_sum_ = 0;
int LocalVipss::ave_voxel_nn_pt_num_ = 0;
int LocalVipss::InitWithPartialVipss = 1;
bool LocalVipss::use_rbf_base_ = false;
bool LocalVipss::use_octree_sample_ = false;

std::vector<tetgenmesh::point> LocalVipss::points_;
// std::vector<std::vector<size_t>> LocalVipss::cluster_all_pt_ids_;


double calculateMemoryUsageG(const SpMat& matrix) {
    using Scalar = typename SpMat::Scalar;
    using Index = typename SpMat::Index;

    size_t valueSize = sizeof(Scalar) * matrix.nonZeros();
    size_t innerIndexSize = sizeof(Index) * matrix.nonZeros();
    size_t outerIndexSize = sizeof(Index) * (matrix.outerSize() + 1);

    return double(valueSize + innerIndexSize + outerIndexSize) / (1024.0 * 1024.0 * 1024.0);
}


void LocalVipss::TestInsertPt()
{
    // std::string in_pt_path = "/home/jjxia/Documents/projects/VIPSS_LOCAL/data/doghead_100/doghead_100.ply";
    std::string in_pt_path =  "../../data/dragon/dragon_sample1k_noise.ply";
    std::vector<double> vts;
    std::vector<double> vns;
    readPLYFile(in_pt_path, vts, vns);
    // std::vector<tetgenmesh::point> insert_pts;
    auto t0 = Clock::now();
    for(size_t i = 0; i < vts.size() / 3; ++i)
    {
        tetgenmesh::point pt = new double[3];
        pt[0] = vts[3 * i];
        pt[1] = vts[3 * i + 1];
        pt[2] = vts[3 * i + 2];
        // insert_pts_.push_back(pt);
        // auto v_gen = voro_gen_;
        // voro_gen_.InsertPt(pt);
        printf("insert id : %ld \n", i);
    }

    auto t1 = Clock::now();
    double insert_time = std::chrono::nanoseconds(t1 - t0).count()/1e9;
    printf(" point insert time total: %g \n", insert_time);

    // for(auto &pt : insert_pts_)
    // {
    //     voro_gen_.InsertPt(pt);
    // }
}


void LocalVipss::VisualFuncValues(double (*function)(const R3Pt &in_pt), const VoroPlane& in_plane,
                              const std::string& dist_fuc_color_path)
{
    
    arma::vec3 ax_1 = {1, 1, 0}; 
    ax_1[2] = -(in_plane.nx * ax_1[0] + in_plane.ny * ax_1[1]) / in_plane.nz;
    double len = sqrt(ax_1[0] * ax_1[0] + ax_1[1] * ax_1[1] + ax_1[2] * ax_1[2]);
    ax_1[0] /= len;
    ax_1[1] /= len;
    ax_1[2] /= len;
    arma::vec3 pv{in_plane.nx, in_plane.ny, in_plane.nz};
    arma::vec3 ax_2 = arma::cross(ax_1, pv);

    std::ofstream out_file;
    std::string out_path = "../out/bi_intersect_plane.obj";
    out_file.open(out_path);

    arma::vec3 ori{in_plane.px, in_plane.py, in_plane.pz};
    double step = 0.001;
    int dim = 200;
    std::vector<double> pts;
    std::vector<uint8_t> pts_co;
    //rbf_core_.isHermite = true;
    //cout << "is hermite " << rbf_core_.isHermite << endl;
    
    for (int i = -dim; i < dim + 1; ++i)
    {
        for (int j = -dim; j < dim + 1; ++j)
        {
            arma::vec3 new_pt = ori + i * step * ax_1 + j * step * ax_2;
            
            pts.push_back(new_pt[0]);
            pts.push_back(new_pt[1]);
            pts.push_back(new_pt[2]);
            R3Pt pt(new_pt[0], new_pt[1], new_pt[2]);
            double dist = function(pt);
            double scale = 0.01;
            dist = dist > -scale ? dist : -scale;
            dist = dist < scale ? dist : scale;
            dist = dist / scale;
            int co_val = abs(dist) * 255;
            uint8_t c_val = uint8_t(co_val);
            if (dist >= 0)
            {
                pts_co.push_back(c_val);
                pts_co.push_back(0);
                pts_co.push_back(0);
            }
            else {
                pts_co.push_back(0);
                pts_co.push_back(c_val);
                pts_co.push_back(0);
            }
        }
    }
    printf("pts size : %ld , pts co size : %ld \n", pts.size()/3, pts_co.size()/3);
    writePLYFile_CO(dist_fuc_color_path, pts, pts_co);
}

void LocalVipss::Init(const std::string & path, const std::string& ext)
{   
    std::vector<double> in_pts;
    // std::vector<double> in_normals;
    input_normals_.clear();
    if(ext == ".ply") 
    {
        readPLYFile(path, in_pts, input_normals_);
    } else {
        
        readXYZnormal(path, in_pts, input_normals_);
    }
    bool normalize_init_pts = true;

    if(normalize_init_pts)
    {
        double min_x = std::numeric_limits<double>::max();
        double min_y = std::numeric_limits<double>::max();
        double min_z = std::numeric_limits<double>::max();

        double max_x = std::numeric_limits<double>::lowest();
        double max_y = std::numeric_limits<double>::lowest();
        double max_z = std::numeric_limits<double>::lowest();

        for(size_t i =0; i < in_pts.size()/3; ++i)
        {
            min_x = min_x < in_pts[3*i]     ? min_x : in_pts[3*i];
            min_y = min_y < in_pts[3*i + 1] ? min_y : in_pts[3*i + 1];
            min_z = min_z < in_pts[3*i + 2] ? min_z : in_pts[3*i + 2];

            max_x = max_x > in_pts[3*i]     ? max_x : in_pts[3*i];
            max_y = max_y > in_pts[3*i + 1] ? max_y : in_pts[3*i + 1];
            max_z = max_z > in_pts[3*i + 2] ? max_z : in_pts[3*i + 2];
        }
        double cx = (min_x + max_x) / 2.0;
        double cy = (min_y + max_y) / 2.0;
        double cz = (min_z + max_z) / 2.0;
        
        double sx = (max_x - min_x) / 2.0 ;
        double sy = (max_y - min_y) / 2.0 ;
        double sz = (max_z - min_z) / 2.0 ;
        std::cout << "input data min box corner : " << min_x << " " << min_y << " " << min_z << std::endl;
        std::cout << "input data max box corner : " << max_x << " " << max_y << " " << max_z << std::endl;
        double scale = std::max(sx, std::max(sy, sz));
        std::cout << " data scale " << scale << std::endl;
        for(size_t i =0; i < in_pts.size()/3; ++i)
        {
            in_pts[3*i] = (in_pts[3*i] - cx) / scale;
            in_pts[3*i + 1] = (in_pts[3*i + 1] - cy) / scale;
            in_pts[3*i + 2] = (in_pts[3*i + 2] - cz) / scale;
        }
        in_pt_center_ = {cx, cy, cz};
        in_pt_scale_ = scale;
    }
    // printf("load data file : %s \n", path.c_str());
    printf("read point size : %lu \n", in_pts.size()/3);
    // origin_in_pts_ = in_pts;
    if(use_octree_sample_)
    {
        auto t0 = Clock::now();
        SamplePtsWithOctree(in_pts, octree_sample_depth_);
        Init(octree_leaf_pts_);
        auto ts1 = Clock::now();
        double sample_time = std::chrono::nanoseconds(ts1 - t0).count()/1e9;
        printf("Init octree sample time : %f s ! \n", sample_time);
    } else {
        Init(in_pts);
    }   
}

void LocalVipss::Init(const std::vector<double>& in_pts)
{   
    auto t0 = Clock::now();
    voro_gen_.out_dir_ = out_dir_;
    voro_gen_.loadData(in_pts);
    printf("start to init mesh \n");
    voro_gen_.InitMesh();
    auto t001 = Clock::now();
    tet_gen_triangulation_time_ = std::chrono::nanoseconds(t001 - t0).count()/1e9;
    printf("finish triangulation :%f s \n", tet_gen_triangulation_time_);
    voro_gen_.BuildAdjecentMat();
    // voro_gen_.BuildAdjecentMatKNN();
    auto t002 = Clock::now();
    tet_build_adj_mat_time_ = std::chrono::nanoseconds(t002 - t001).count()/1e9;

    printf("finish build adjecent mat \n");
    // adjacent_mat_ = voro_gen_.pt_adjecent_mat_;
    points_ = voro_gen_.points_;
    pt_num_ = points_.size();

    // cluster_cores_mat_.resize(pt_num_, pt_num_);
    // cluster_cores_mat_.eye();
    vipss_api_.Set_RBF_PARA();
    vipss_api_.is_surfacing_ = false;
    vipss_api_.outpath_ = out_dir_;
    vipss_api_.user_lambda_ = user_lambda_;
    vipss_api_.user_lambda_ = user_lambda_;
    vipss_api_.n_voxel_line_ = volume_dim_;

    cluster_normal_x_.resize(pt_num_, pt_num_);
    cluster_normal_y_.resize(pt_num_, pt_num_);
    cluster_normal_z_.resize(pt_num_, pt_num_);

    cluster_scores_mat_.resize(pt_num_, pt_num_);
    auto t1 = Clock::now(); 
    double init_time = std::chrono::nanoseconds(t1 - t0).count()/1e9;
    printf("build adj mat and initilization : %f s ! \n", init_time);
}

// inline std::vector<size_t> LocalVipss::GetClusterCoreIds(size_t cluster_id) const
// {
//     arma::sp_umat cluster_core_ids(cluster_cores_mat_.col(cluster_id));
//     arma::sp_umat::const_iterator start = cluster_core_ids.begin();
//     arma::sp_umat::const_iterator end = cluster_core_ids.end();
//     std::vector<size_t> core_ids;
//     for ( auto i = start; i != end; ++i )
//     {
//         core_ids.push_back(i.row());
//     }
//     return core_ids;
// }


// void LocalVipss::BuidClusterCoresPtIds()
// {
//     cluster_core_pt_ids_.clear();
//     size_t c_num = points_.size();
//     for(size_t i =0; i < c_num; ++i)
//     {
//         auto cur_cores = GetClusterCoreIds(i);
//         cluster_core_pt_ids_.push_back(cur_cores);
//     }
// }



inline arma::sp_mat BuildClusterPtIdMatrix(const std::vector<size_t>& p_ids, size_t npt)
{
    size_t unit_npt = p_ids.size();
    arma::sp_mat unit_cluster_mat(unit_npt*4, npt * 4);
    for(size_t id = 0; id < unit_npt; ++id)
    {
        size_t pid = p_ids[id];
        unit_cluster_mat(id, pid) = 1.0;
        unit_cluster_mat(id + unit_npt,     pid + npt ) = 1.0;
        unit_cluster_mat(id + unit_npt * 2, pid + npt * 2) = 1.0;
        unit_cluster_mat(id + unit_npt * 3, pid + npt * 3) = 1.0;
    }
    return unit_cluster_mat;
}

void LocalVipss::AddClusterHMatrix(const std::vector<int>& p_ids, const arma::mat& J_m, size_t npt)
{
    // size_t unit_npt = p_ids.size();
    // size_t unit_key_npt = unit_npt;
    // std::vector<Triplet> coefficients;
    // for(size_t i = 0; i < unit_npt; ++i)
    // {
    //     size_t pi = p_ids[i];
    //     if(user_lambda_ > 1e-12)
    //     {
    //         for(size_t j = 0; j < unit_npt; ++j)
    //         {
    //             h_ele_triplets_.push_back(std::move(Triplet(pi, p_ids[j], J_m(i, j))));
    //         }
    //         for(size_t j = 0; j < unit_key_npt; ++j)
    //         {
    //             h_ele_triplets_.push_back(std::move(Triplet(pi, p_ids[j] + npt, J_m(i, j + unit_npt))));
    //             h_ele_triplets_.push_back(std::move(Triplet(pi, p_ids[j] + npt*2, J_m(i, j + unit_npt + unit_key_npt)))); 
    //             h_ele_triplets_.push_back(std::move(Triplet(pi, p_ids[j] + npt*3, J_m(i, j + unit_npt + unit_key_npt* 2))));   
    //         }
    //         for(size_t j = 0; j < unit_key_npt; ++j)
    //         {
    //             h_ele_triplets_.push_back(std::move(Triplet(p_ids[j] + npt,   pi, J_m(i, j + unit_npt))));
    //             h_ele_triplets_.push_back(std::move(Triplet(p_ids[j] + npt*2, pi, J_m(i, j + unit_npt + unit_key_npt)))); 
    //             h_ele_triplets_.push_back(std::move(Triplet(p_ids[j] + npt*3, pi, J_m(i, j + unit_npt + unit_key_npt* 2))));   
    //         }
    //     }
        
    //     if(i < unit_key_npt)
    //     {
    //         for(size_t step = 0; step < 3; ++step)
    //         {
    //             size_t row_i = npt + npt * step + pi;
    //             size_t rowv_i = i + unit_npt + unit_key_npt * step;
    //             for(size_t j = 0; j < unit_key_npt; ++j)
    //             {
    //                 h_ele_triplets_.push_back(std::move(Triplet(row_i, p_ids[j] + npt,     J_m( rowv_i, j + unit_npt))));
    //                 h_ele_triplets_.push_back(std::move(Triplet(row_i, p_ids[j] + npt * 2, J_m( rowv_i, j + unit_npt + unit_key_npt))));
    //                 h_ele_triplets_.push_back(std::move(Triplet(row_i, p_ids[j] + npt * 3, J_m( rowv_i, j + unit_npt + unit_key_npt * 2))));
    //             }
    //         }
    //     }

        // for(size_t step = 0; step < 4; ++step)
        // {
        //     size_t row = pi + npt * step;
        //     size_t j_row = i + unit_npt* step;
        //     for(size_t j = 0; j < unit_npt; ++j)
        //     {
        //         size_t pj = p_ids[j];
        //         // h_row_map[pj]           += J_m(j_row, j);
        //         // h_row_map[npt + pj]     += J_m(j_row, j + unit_npt);
        //         // h_row_map[npt * 2 + pj] += J_m(j_row, j + unit_npt * 2);
        //         // h_row_map[npt * 3 + pj] += J_m(j_row, j + unit_npt * 3);

                    
        //         for(size_t k = 0; k < 4; ++k)
        //         {
        //             h_ele_triplets_.push_back(std::move(Triplet(row, pj + npt * k, J_m(j_row, j + unit_npt * k))));
        //         }    
        //     }
        // }
    // }
}


void LocalVipss::AddClusterHMatrix(const std::vector<int>& p_ids, const arma::mat& J_m, size_t npt, std::vector<Triplet>& ele_vect )
{
    size_t unit_npt = p_ids.size();
    size_t unit_key_npt = unit_npt;
    std::vector<Triplet> coefficients;
    for(size_t i = 0; i < unit_npt; ++i)
    {
        size_t pi = p_ids[i];
        if(user_lambda_ > 1e-12)
        {
            for(size_t j = 0; j < unit_npt; ++j)
            {
                ele_vect.push_back(std::move(Triplet(pi, p_ids[j], J_m(i, j))));
            }
            for(size_t j = 0; j < unit_key_npt; ++j)
            {
                ele_vect.push_back(std::move(Triplet(pi, p_ids[j] + npt, J_m(i, j + unit_npt))));
                ele_vect.push_back(std::move(Triplet(pi, p_ids[j] + npt*2, J_m(i, j + unit_npt + unit_key_npt)))); 
                ele_vect.push_back(std::move(Triplet(pi, p_ids[j] + npt*3, J_m(i, j + unit_npt + unit_key_npt* 2))));   
            }
            for(size_t j = 0; j < unit_key_npt; ++j)
            {
                ele_vect.push_back(std::move(Triplet(p_ids[j] + npt,   pi, J_m(i, j + unit_npt))));
                ele_vect.push_back(std::move(Triplet(p_ids[j] + npt*2, pi, J_m(i, j + unit_npt + unit_key_npt)))); 
                ele_vect.push_back(std::move(Triplet(p_ids[j] + npt*3, pi, J_m(i, j + unit_npt + unit_key_npt* 2))));   
            }
        }
        
        if(i < unit_key_npt)
        {
            for(size_t step = 0; step < 3; ++step)
            {
                size_t row_i = npt + npt * step + pi;
                size_t rowv_i = i + unit_npt + unit_key_npt * step;
                for(size_t j = 0; j < unit_key_npt; ++j)
                {
                    ele_vect.push_back(std::move(Triplet(row_i, p_ids[j] + npt,     J_m( rowv_i, j + unit_npt))));
                    ele_vect.push_back(std::move(Triplet(row_i, p_ids[j] + npt * 2, J_m( rowv_i, j + unit_npt + unit_key_npt))));
                    ele_vect.push_back(std::move(Triplet(row_i, p_ids[j] + npt * 3, J_m( rowv_i, j + unit_npt + unit_key_npt * 2))));
                }
            }
        }
    }
}


void LocalVipss::AddClusterHMatrix(const std::vector<int>& p_ids, const arma::mat& J_m, size_t npt, 
                                    std::vector<Triplet>::iterator& ele_iter )
{
    size_t unit_npt = p_ids.size();
    size_t unit_key_npt = unit_npt;
    // std::vector<Triplet> coefficients;
    size_t iter_num = use_rbf_base_? 1 : unit_npt;

    for(size_t i = 0; i < iter_num; ++i)
    {
        size_t pi = p_ids[i];
        if(user_lambda_ > 1e-12)
        {
            for(size_t j = 0; j < unit_npt; ++j)
            {
                *(ele_iter ++) =  Triplet(pi, p_ids[j], J_m(i, j));
            }
            for(size_t j = 0; j < unit_key_npt; ++j)
            {
                *(ele_iter ++) = Triplet(pi, p_ids[j] + npt, J_m(i, j + unit_npt));
                *(ele_iter ++) = Triplet(pi, p_ids[j] + npt*2, J_m(i, j + unit_npt + unit_key_npt)); 
                *(ele_iter ++) = Triplet(pi, p_ids[j] + npt*3, J_m(i, j + unit_npt + unit_key_npt* 2));   
            }
            for(size_t j = 0; j < unit_key_npt; ++j)
            {
                *(ele_iter ++) = Triplet(p_ids[j] + npt,   pi, J_m(i, j + unit_npt));
                *(ele_iter ++) = Triplet(p_ids[j] + npt*2, pi, J_m(i, j + unit_npt + unit_key_npt)); 
                *(ele_iter ++) = Triplet(p_ids[j] + npt*3, pi, J_m(i, j + unit_npt + unit_key_npt* 2));   
            }
        }
        
        if(i < unit_key_npt)
        {
            for(size_t step = 0; step < 3; ++step)
            {
                size_t row_i = npt + npt * step + pi;
                size_t rowv_i = i + unit_npt + unit_key_npt * step;
                for(size_t j = 0; j < unit_key_npt; ++j)
                {
                    *(ele_iter ++) = Triplet(row_i, p_ids[j] + npt,     J_m( rowv_i, j + unit_npt));
                    *(ele_iter ++) = Triplet(row_i, p_ids[j] + npt * 2, J_m( rowv_i, j + unit_npt + unit_key_npt));
                    *(ele_iter ++) = Triplet(row_i, p_ids[j] + npt * 3, J_m( rowv_i, j + unit_npt + unit_key_npt * 2));
                }
            }
        }
    }
}



void LocalVipss::AddHalfClusterHMatrix(const std::vector<int>& p_ids, const arma::mat& J_m, size_t npt, 
                                    std::vector<Triplet>::iterator& ele_iter,  std::vector<Triplet>::iterator& diagonal_ele_iter)
{
    size_t unit_npt = p_ids.size();
    size_t unit_key_npt = unit_npt;
    // std::vector<Triplet> coefficients;

    for(size_t i = 0; i < unit_npt; ++i)
    {
        size_t pi = p_ids[i];
        if(user_lambda_ > 1e-12)
        {
            *(diagonal_ele_iter ++) =  Triplet(pi, pi, J_m(i, i));
            for(size_t j = 0; j < unit_npt; ++j)
            {
                if(pi < p_ids[j])
                *(ele_iter ++) =  Triplet(pi, p_ids[j], J_m(i, j));
            }
            for(size_t j = 0; j < unit_key_npt; ++j)
            {
                *(ele_iter ++) = Triplet(pi, p_ids[j] + npt,   J_m(i, j + unit_npt));
                *(ele_iter ++) = Triplet(pi, p_ids[j] + npt*2, J_m(i, j + unit_npt*2)); 
                *(ele_iter ++) = Triplet(pi, p_ids[j] + npt*3, J_m(i, j + unit_npt*3));   
            }
            // for(size_t j = i; j < unit_key_npt; ++j)
            // {
            //     *(ele_iter ++) = Triplet(p_ids[j] + npt,   pi, J_m(i, j + unit_npt));
            //     *(ele_iter ++) = Triplet(p_ids[j] + npt*2, pi, J_m(i, j + unit_npt + unit_key_npt)); 
            //     *(ele_iter ++) = Triplet(p_ids[j] + npt*3, pi, J_m(i, j + unit_npt + unit_key_npt* 2));   
            // }
        }
        
        size_t row_0 = npt  + pi;
        size_t row_1 = npt * 2  + pi;
        size_t row_2 = npt * 3  + pi;

        size_t rowv_i0 = i + unit_npt;
        size_t rowv_i1 = i + unit_npt * 2;
        size_t rowv_i2 = i + unit_npt * 3;

        *(diagonal_ele_iter ++) =  Triplet(row_0, row_0, J_m(rowv_i0, rowv_i0));
        *(diagonal_ele_iter ++) =  Triplet(row_1, row_1, J_m(rowv_i1, rowv_i1));
        *(diagonal_ele_iter ++) =  Triplet(row_2, row_2, J_m(rowv_i2, rowv_i2));

        for(size_t step = 1; step <= 3; ++step)
        {
            size_t row_i = npt * step + pi;
            size_t rowv_i = i + unit_npt * step;
            for(size_t j = 0; j < unit_key_npt; ++j)
            {
                for(int k = 1; k <= 3; ++k)
                {
                    if(row_i < p_ids[j] + npt*k)
                    {
                        *(ele_iter ++) = Triplet(row_i, p_ids[j] + npt*k,     J_m( rowv_i, j + unit_npt*k));
                    }
                }
            }
        }
        
    }
}

void saveSparseMatrixToFile(const Eigen::SparseMatrix<double>& mat, const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    // Save the matrix in a triplet format: row, col, value
    for (int k = 0; k < mat.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
            // if(it.row() != it.col()) continue;
            file << it.row() << " " << it.col() << " " << it.value() << "\n";
        }
    }

    file.close();
    std::cout << "Matrix saved to " << filename << "\n";
}


void LocalVipss::BuildMatrixH()
{
    auto t00 = Clock::now(); 
    size_t npt = this->points_.size();
    final_h_eigen_.resize(4 * npt , 4 * npt);
    double add_ele_to_vector_time = 0;

    // int all_ele_num = arma::dot(VoronoiGen::cluster_size_vec_ , VoronoiGen::cluster_size_vec_) * 16;
    arma::ivec cluster_j_size_vec = VoronoiGen::cluster_size_vec_  % VoronoiGen::cluster_size_vec_ * 16; 
    int all_ele_num = arma::accu(cluster_j_size_vec);
    std::vector<Triplet> h_ele_triplets;
    h_ele_triplets.resize(all_ele_num);
    printf("average cluster J ele num : %d \n", int(arma::mean(cluster_j_size_vec)));

    arma::ivec acc_j_size_vec = arma::cumsum(cluster_j_size_vec);
    auto iter = h_ele_triplets.begin();
    auto t5 = Clock::now();
    // VIPSSKernel::use_rbf_base = LocalVipss::use_rbf_base_;
// #pragma omp parallel for shared(points_, VoronoiGen::cluster_init_pids_) 
    // double s_factor  = 1.0 / double(npt);
#pragma omp parallel for
    for(int i =0; i < npt; ++i)
    {
        const auto& cluster_pt_ids = VoronoiGen::cluster_init_pids_[i];
        auto Minv =  VIPSSKernel::BuildHrbfMat(points_, cluster_pt_ids, use_rbf_base_);
        size_t unit_npt = cluster_pt_ids.size();
        size_t j_ele_num = unit_npt * unit_npt * 16;
        auto cur_iter = iter + acc_j_size_vec[i];
        if(user_lambda_ > 1e-10)
        {
            arma::mat F(4 * unit_npt, 4 * unit_npt);
            arma::mat E(unit_npt, unit_npt);
            E.eye();
            F(0, 0, arma::size(unit_npt, unit_npt)) = E;
            double cur_lambda = user_lambda_;
            arma::mat K = (F + Minv * cur_lambda);
            // arma::mat K =  Minv * cur_lambda ;
            AddClusterHMatrix(cluster_pt_ids, K, npt, cur_iter);
        } else {
            AddClusterHMatrix(cluster_pt_ids, Minv, npt, cur_iter);
        }
    }

    auto t_h1 = Clock::now();
    final_h_eigen_.setFromTriplets(h_ele_triplets.begin(), h_ele_triplets.end());
    // std::string hmat_txt_path = "./origin_hmat.txt";
    // saveSparseMatrixToFile(final_h_eigen_, hmat_txt_path);

    auto t_h2 = Clock::now();
    double build_h_from_tris_time = std::chrono::nanoseconds(t_h2 - t_h1).count()/1e9;
    
    auto t11 = Clock::now();
    double build_H_time_total = std::chrono::nanoseconds(t11 - t00).count()/1e9;

    printf("--- build_H_time_total sum  time : %f \n", build_H_time_total);
}


void LocalVipss::BuildMatrixHSparse()
{
    auto t00 = Clock::now(); 
    int npt = this->points_.size();
    // arma::ivec cluster_j_size_vec = VoronoiGen::cluster_size_vec_  % VoronoiGen::cluster_size_vec_ * 16; 
    // size_t all_ele_num = arma::accu(cluster_j_size_vec);

    // std::cout << " all elements number : " << all_ele_num << std::endl;

    enum { IsRowMajor = SpMat::IsRowMajor };
    typedef typename SpMat::Scalar Scalar;
    typedef typename SpMat::StorageIndex StorageIndex;
    // std::cout << " rows : " << mat.rows()  << " cols : " << mat.cols() << std::endl;
//   SpMat trMat(mat.rows(),mat.cols());
    // final_h_eigen_ = SpMat( 4* npt, 4* npt);
    // auto mat_ptr = std::make_shared<SpMat>( 4* npt, 4* npt);
    // auto& trMat = *mat_ptr;
    // Eigen::SparseMatrix<Scalar,IsRowMajor?Eigen::ColMajor:Eigen::RowMajor, long> trMat( 4* npt, 4* npt);
    Eigen::SparseMatrix<Scalar,IsRowMajor?Eigen::ColMajor:Eigen::RowMajor, long> diagMat( 4* npt, 4* npt);

  
    double add_ele_to_vector_time = 0;
    auto t5 = Clock::now();

    typename SpMat::IndexVector wi(4*npt);
    typename SpMat::IndexVector di(4*npt);
    
    di.setZero();
    Eigen::internal::scalar_sum_op<double, double> dup_func;
    // for(auto it = in_tris.begin(); it != in_tris.end(); ++it)
    for(int pid =0; pid < npt; ++pid)
    {
        const auto& p_ids = VoronoiGen::cluster_init_pids_[pid];
        for(auto cur_i : p_ids)
        {
            for(int i =0; i < 4; ++i)
            {
                di(npt*i + cur_i) ++;
            }
        }
    }
    // std::cout << " all element size wi : " << wi.sum() << std::endl;
    // trMat.reserve(wi);
    diagMat.reserve(di);
    std::cout << " finish reserve trMat "  << std::endl;
    int batch_num = 4;
    int batch_size = std::max(npt / batch_num + 1, 10000);
    // batch_size = npt >= 100000 ? batch_size : npt;

    std::unordered_set<MyTriplet, PairHash> seen;

    for(int count = 0; count < npt; count += batch_size)
    {
        wi.setZero();
        int cur_batch_size = std::min(batch_size, npt - count);
        // #pragma omp parallel for
        for(int pid =count; pid < count + cur_batch_size; ++pid)
        {
            const auto& p_ids = VoronoiGen::cluster_init_pids_[pid];
            for(auto cur_i : p_ids)
            {
                for(auto cur_j : p_ids)
                {
                    if(cur_i < cur_j) 
                    {
                        // #pragma omp critical
                        {
                            wi(IsRowMajor ? cur_j: cur_i)++;
                        }
                    }
                    // #pragma omp critical
                    {
                        wi(IsRowMajor ? cur_j + npt     : cur_i) ++;
                        wi(IsRowMajor ? cur_j + npt * 2 : cur_i) ++;
                        wi(IsRowMajor ? cur_j + npt * 3 : cur_i) ++;
                    }
                    
                    for(int si = 1; si<=3; ++si)
                    {
                        for(int sj = 1; sj <=3; ++sj)
                        {
                            if(si * npt + cur_i < cur_j + sj* npt )
                            {
                                // #pragma omp critical
                                {
                                    wi(IsRowMajor ? cur_j + sj* npt : si * npt + cur_i)++;
                                }
                            }
                        }
                    }
                }
            }
        }
        Eigen::SparseMatrix<Scalar,IsRowMajor?Eigen::ColMajor:Eigen::RowMajor, long> trMat( 4* npt, 4* npt);

        trMat.reserve(wi);
        // std::cout << " all element size wi : " << wi.sum() << std::endl;

        #pragma omp parallel for
        for(int pid = count; pid < count + cur_batch_size; ++pid)
        {
            const auto& p_ids = VoronoiGen::cluster_init_pids_[pid];
            auto Minv =  VIPSSKernel::BuildHrbfMat(points_, p_ids, use_rbf_base_);
            size_t unit_npt = p_ids.size();
            if(user_lambda_ > 1e-16)
            {
                arma::mat F(4 * unit_npt, 4 * unit_npt);
                arma::mat E(unit_npt, unit_npt);
                E.eye();
                F(0, 0, arma::size(unit_npt, unit_npt)) = E;
                double cur_lambda = user_lambda_;
                Minv = (F + Minv * cur_lambda);
            } 
            // int top_count = 0;
            // std::cout << "unit_npt : " << unit_npt << std::endl;
            for(size_t i = 0; i < unit_npt; ++i)
            {
                size_t pi = p_ids[i];
            
                if(user_lambda_ > 1e-12)
                {
                    for(size_t j = 0; j < unit_npt; ++j)
                    {
                        // #pragma omp critical
                        if(pi < p_ids[j])
                        {
                            #pragma omp critical
                            {
                                trMat.insertBackUncompressed(pi,p_ids[j]) = Minv(i, j);
                            }
                            
                        } else if(pi == p_ids[j])
                        {
                            #pragma omp critical
                            {
                                diagMat.insertBackUncompressed(pi,p_ids[j]) = Minv(i, j);
                                
                                // Index p = m_outerIndex[outer] + m_innerNonZeros[outer]++;
                                // m_data.index(p) = convert_index(inner);
                                // return (m_data.value(p) = Scalar(0));
                            }
                        }
                    }
                    for(size_t j = 0; j < unit_npt; ++j)
                    {
                        for(int k = 1; k <= 3; ++k)
                        {
                            {
                                #pragma omp critical
                                {
                                    trMat.insertBackUncompressed(pi,p_ids[j] + npt*k) =  Minv(i, j + unit_npt*k);
                                }
                            }
                        }
                    }
                }
                
                for(size_t step = 1; step <= 3; ++step)
                {
                    size_t row_i = npt * step + pi;
                    size_t rowv_i = i + unit_npt * step;
                    for(size_t j = 0; j < unit_npt; ++j)
                    {
                        for(int k = 1; k <= 3; ++k)
                        {
                            size_t col_id = p_ids[j] + npt * k;
                            {
                                if(row_i< col_id)
                                {
                                    #pragma omp critical
                                    {
                                        trMat.insertBackUncompressed(row_i, col_id) =  Minv(rowv_i, j + unit_npt*k);
                                    }
                                } else if(row_i == col_id)
                                {
                                    #pragma omp critical
                                    {
                                        diagMat.insertBackUncompressed(row_i, col_id) =  Minv(rowv_i, j + unit_npt*k);
                                    }
                                }
                                
                            }
                        }
                    }
                }
            }
        }
        // double cur_h_memory = calculateMemoryUsageG(final_h_eigen_);
        // double cur_trMat_memory = calculateMemoryUsageG(trMat);
        // std::cout << "----- current h memory G: " << cur_h_memory << std::endl;
        // std::cout << "----- current tr Mat memory G: " << cur_trMat_memory << std::endl;
        // std::cout << "----- current possible max memory G: " << cur_h_memory + cur_trMat_memory << std::endl;

        trMat.collapseDuplicates(dup_func);
        trMat.makeCompressed();
        // std::cout << " makeCompressed nonzero count : " << trMat.nonZeros() << std::endl;
        if(count == 0)
        {
            final_h_eigen_ = trMat;
        } else {
            SpMat tempmat = trMat;
            // double cur_h_memory = calculateMemoryUsageG(final_h_eigen_);
            // double compressed_trMat_memory = calculateMemoryUsageG(trMat);
            // double compressed_tempmat_memory = calculateMemoryUsageG(tempmat);
            // std::cout << "----- compressed_trMat_memory G: " << compressed_trMat_memory << std::endl;
            // std::cout << "----- compressed_tempmat_memory G: " << compressed_tempmat_memory << std::endl;
            // std::cout << "----- 2 current possible max memory G: " 
            //           << cur_h_memory + compressed_trMat_memory + compressed_tempmat_memory << std::endl;
            trMat.resize(0,0);
            trMat.data().squeeze();
            final_h_eigen_ += tempmat;
            // final_h_eigen_.makeCompressed();
        }
        // std::cout << " final_h_eigen_ makeCompressed nonzero count : " << final_h_eigen_.nonZeros() << std::endl;

    }

    diagMat.collapseDuplicates(dup_func);
    diagMat.makeCompressed();
    // trMat.resize(0,0);
    // trMat.data().squeeze();
    SpMat temp_mat = final_h_eigen_.transpose();
    final_h_eigen_ += temp_mat;
    
    double cur_h_memory = calculateMemoryUsageG(final_h_eigen_);
    double temp_mat_memory = calculateMemoryUsageG(temp_mat);
    double diag_memory = calculateMemoryUsageG(diagMat);
    G_VP_stats.possible_max_memory = cur_h_memory + temp_mat_memory + diag_memory;


    // std::cout << "----- final h memory G: " << cur_h_memory << std::endl;
    // std::cout << "----- temp_mat memory G: " << temp_mat_memory << std::endl;
    std::cout << "----- current possible max memory G: " 
                << G_VP_stats.possible_max_memory << std::endl;

    // final_h_eigen_.makeCompressed();
    temp_mat.resize(0,0);
    temp_mat.data().squeeze();
    SpMat d_mat = diagMat;    
    diagMat.resize(0,0);
    diagMat.data().squeeze();
    final_h_eigen_ += d_mat;
    // double diag_memory = calculateMemoryUsageG(final_h_eigen_);
    G_VP_stats.max_hmat_memory = calculateMemoryUsageG(final_h_eigen_);
    
    auto t11 = Clock::now();
    double build_H_time_total = std::chrono::nanoseconds(t11 - t00).count()/1e9;
    printf("--- build_H_time_total sum  time : %f \n", build_H_time_total);
}


void LocalVipss::BuildMatrixHBatches()
{
    auto t00 = Clock::now(); 
    size_t npt = this->points_.size();
    final_h_eigen_.resize(4 * npt , 4 * npt);
    // int all_ele_num = arma::dot(VoronoiGen::cluster_size_vec_ , VoronoiGen::cluster_size_vec_) * 16;
    
    arma::ivec cluster_j_size_vec = VoronoiGen::cluster_size_vec_  % VoronoiGen::cluster_size_vec_ * 16; 
    size_t all_ele_num = arma::accu(VoronoiGen::cluster_size_vec_);
    VoronoiGen::cluster_size_vec_.clear();

    double base = double(1024 * 1024 * 1024);

    std::cout << " current cluster ids memory : " << double(sizeof(size_t)) * double(all_ele_num) / base << std::endl;
    // std::vector<Triplet> h_ele_triplets;
    // h_ele_triplets.resize(all_ele_num);
    // printf("average cluster J ele num : %d \n", int(arma::mean(cluster_j_size_vec)));
    // arma::ivec acc_j_size_vec = arma::cumsum(cluster_j_size_vec);
    //std::vector<std::vector<Triplet>> ele_vector(cluster_num);
    auto t5 = Clock::now();
    size_t batch_size = npt / 4 + 1;
    // size_t batch_size = npt;
    std::vector<Triplet> h_ele_triplets;
    for(int count = 0; count < npt; count += batch_size)
    {
        arma::ivec temp_vec = cluster_j_size_vec.subvec(count+1, std::min(npt-1, count+batch_size));
        // std::cout << "temp_vec size : " << temp_vec.size() << std::endl;
        // std::cout << "npt : " << npt << std::endl;
        // std::cout << "cluster_j_size_vec size : " << cluster_j_size_vec.size() << std::endl;
        int batch_ele_num = arma::accu(temp_vec);
        h_ele_triplets.resize(batch_ele_num);
        auto iter = h_ele_triplets.begin();
        size_t cur_batch_size = std::min(batch_size, npt - count);
        // std::cout << "cur count : " << count << std::endl;
        // std::cout << "cur batch_ele_num : " << batch_ele_num << std::endl;
        // std::cout << "cur batch_size : " << cur_batch_size << std::endl;
        arma::ivec acc_j_size_vec = arma::cumsum(temp_vec);
        temp_vec.clear();
        #pragma omp parallel for
        for(int i =0; i < cur_batch_size; ++i)
        {
            auto id = count + i;
            const auto& cluster_pt_ids = VoronoiGen::cluster_init_pids_[id];
            auto Minv =  VIPSSKernel::BuildHrbfMat(points_, cluster_pt_ids, use_rbf_base_);
            size_t unit_npt = cluster_pt_ids.size();
            int shift_size = 0;
            if( i > 0)  shift_size =  acc_j_size_vec[i-1];
            auto cur_iter = iter + shift_size;
            // std::cout << "cur unit_npt : " << unit_npt << std::endl;
            // std::cout << "cur i : " << i << std::endl;
            if(user_lambda_ > 1e-10)
            {
                arma::mat F(4 * unit_npt, 4 * unit_npt);
                arma::mat E(unit_npt, unit_npt);
                E.eye();
                F(0, 0, arma::size(unit_npt, unit_npt)) = E;
                double cur_lambda = user_lambda_;
                arma::mat K = (F + Minv * cur_lambda);
                AddClusterHMatrix(cluster_pt_ids, K, npt, cur_iter);
            } else {
                AddClusterHMatrix(cluster_pt_ids, Minv, npt, cur_iter);
            }
        }
        acc_j_size_vec.clear();
        SpMat temp_mat(4 * npt , 4 * npt);
        temp_mat.setFromTriplets(h_ele_triplets.begin(), h_ele_triplets.end());
        auto cur_temp_me_size = calculateMemoryUsageG(temp_mat);
        
        auto vec_size = sizeof(Triplet) * h_ele_triplets.size();
        auto cur_h_size = calculateMemoryUsageG(final_h_eigen_);

        std::cout << " current vector memory size : " << double(vec_size) / base << std::endl;
        std::cout << " current temp h memory size : " << double(cur_temp_me_size) / base << std::endl;
        std::cout << " current final h memory size : " << double(cur_h_size) / base << std::endl;
        std::cout << " current max  memory size total : " << double(cur_h_size + vec_size + cur_temp_me_size) / base << std::endl;
        
        h_ele_triplets.clear();
        temp_mat.makeCompressed();
        final_h_eigen_ += temp_mat; 
        final_h_eigen_.makeCompressed();
        auto temp_me_size = calculateMemoryUsageG(temp_mat);
        auto max_h_size = calculateMemoryUsageG(final_h_eigen_);
        auto max_size_total = double(temp_me_size + max_h_size) ;
        std::cout << " current  batch id : " <<  count << " , memory size(G bytes) : " << max_size_total << std::endl;
    }

    auto max_h_size = calculateMemoryUsageG(final_h_eigen_);
    auto max_size_total = double(max_h_size);
    std::cout << "final h total memory size : "  << max_size_total << std::endl;
    auto t11 = Clock::now();
    double build_H_time_total = std::chrono::nanoseconds(t11 - t00).count()/1e9;

    printf("--- build_H_time_total sum  time : %f \n", build_H_time_total);

}


void LocalVipss::BuildMatrixHMemoryOpt()
{
    auto t00 = Clock::now(); 
    size_t npt = this->points_.size();
    final_h_eigen_.resize(4 * npt , 4 * npt);
    std::cout << " cluster size : " << npt << std::endl;
    std::cout << " cluster size : " << VoronoiGen::cluster_size_vec_.size() << std::endl;
    arma::ivec cluster_j_size_vec = VoronoiGen::cluster_size_vec_  % VoronoiGen::cluster_size_vec_ * 16;    
    arma::ivec cluster_j_top_size_vec = (cluster_j_size_vec - VoronoiGen::cluster_size_vec_ * 4)/2;
    auto t5 = Clock::now();
    int min_batch_size = min_batch_size_;
    int batch_num = 16;
    int batch_size = std::max(int(npt/batch_num +1), min_batch_size);

    SpMat diag_h(4* npt , 4 * npt);
    std::vector<SpMat> diag_h_vec;
    std::vector<SpMat> top_h_vec;
    bool has_assigned_h= false;
    
    // int st_id = 0;
    // int cur_batch_size = npt;
    for(int st_id = 0; st_id < npt; st_id += batch_size)
    {
        // std::cout << " st id " << st_id << std::endl;
        int cur_batch_size = std::min(int(npt - st_id), batch_size);
        // if(0)
        // std::cout << " cur_batch_size " << cur_batch_size << std::endl;
        arma::ivec cur_j_top_size_vec = cluster_j_top_size_vec.subvec(st_id, st_id + cur_batch_size);
        cur_j_top_size_vec[0] = 0;
        arma::ivec cur_acc_j_size_vec = arma::cumsum(cur_j_top_size_vec);
        arma::ivec cur_j_diag_size_vec = VoronoiGen::cluster_size_vec_.subvec(st_id, st_id + cur_batch_size) * 4;
        cur_j_diag_size_vec[0] = 0;

        // std::cout << " cur_batch_size 01" << cur_batch_size << std::endl;
        size_t all_ele_num = arma::accu(cur_j_top_size_vec);
        size_t diagonal_ele_num = arma::accu(cur_j_diag_size_vec);
        cur_j_diag_size_vec =  arma::cumsum(cur_j_diag_size_vec);
        std::vector<Triplet> h_ele_triplets;
        std::vector<Triplet> h_diag_ele_triplets;
        // std::cout << " cur_batch_size 02" << cur_batch_size << std::endl;
        std::cout << " all_ele_num " << all_ele_num << std::endl;
        std::cout << " diagonal_ele_num " << diagonal_ele_num << std::endl;

        size_t memory_size = sizeof(Triplet) * all_ele_num / (1024 * 1024 * 1024);
        std::cout << " allocate memory : " <<  memory_size << std::endl;
        h_ele_triplets.resize(all_ele_num);
        h_diag_ele_triplets.resize(diagonal_ele_num);

        std::cout << " all_ele_num 0 " << all_ele_num << std::endl;
        std::cout << " diagonal_ele_num 1 " << diagonal_ele_num << std::endl;
        auto iter = h_ele_triplets.begin();
        auto diag_iter = h_diag_ele_triplets.begin();
        

        // arma::ivec acc_j_diag_size_vec = arma::cumsum(cluster_j_diag_size_vec);
        
        std::cout << " cur_batch_size 22 " << cur_batch_size << std::endl;
        #pragma omp parallel for
        for(int i = st_id; i < st_id + cur_batch_size; ++i)
        {
            // std::cout << " curr id " << i << std::endl;
            const auto& cluster_pt_ids = VoronoiGen::cluster_init_pids_[i];
            auto Minv =  VIPSSKernel::BuildHrbfMat(points_, cluster_pt_ids, use_rbf_base_);
            size_t unit_npt = cluster_pt_ids.size();
            // if(0)
            
            auto cur_iter = iter + cur_acc_j_size_vec[i - st_id];
            auto cur_diag_iter = diag_iter + cur_j_diag_size_vec[i - st_id];
            if(user_lambda_ > 1e-10)
            {
                arma::mat F(4 * unit_npt, 4 * unit_npt);
                arma::mat E(unit_npt, unit_npt);
                E.eye();
                F(0, 0, arma::size(unit_npt, unit_npt)) = E;
                double cur_lambda = user_lambda_;
                // arma::mat K = (F + Minv * cur_lambda);
                arma::mat K =  Minv * cur_lambda;
                AddHalfClusterHMatrix(cluster_pt_ids, K, npt, cur_iter, cur_diag_iter);
            } else {
                AddHalfClusterHMatrix(cluster_pt_ids, Minv, npt, cur_iter, cur_diag_iter);
            }
        }
        std::cout << " finish retrive current batch" << std::endl;
       
        SpMat temp_diag(4 *npt, 4* npt);
        temp_diag.setFromTriplets(h_diag_ele_triplets.begin(), h_diag_ele_triplets.end());
        h_diag_ele_triplets.clear();
        diag_h_vec.push_back(std::move(temp_diag));
        
        SpMat temp_top(4 *npt, 4* npt);
        temp_top.setFromTriplets(h_ele_triplets.begin(), h_ele_triplets.end());
        h_ele_triplets.clear();
        top_h_vec.push_back(std::move(temp_top));
    }
  
    if(top_h_vec.size() > 0)
    {
        addSparseMatrixParallel(top_h_vec, diag_h_vec);
        final_h_eigen_ = top_h_vec[0];
        top_h_vec.clear();
        diag_h = diag_h_vec[0];
        diag_h_vec.clear();
    }
    auto t_h1 = Clock::now();
    printf("Finish push ele to vector \n");
    
    SpMat final_h_eigen_t =  final_h_eigen_.transpose();
    final_h_eigen_ += final_h_eigen_t;
    final_h_eigen_t.resize(0,0);
    final_h_eigen_t.data().squeeze();
    final_h_eigen_ += diag_h;
    
    // std::string hmat_txt_path = "./me_hmat.txt";
    // saveSparseMatrixToFile(final_h_eigen_, hmat_txt_path);
    auto t11 = Clock::now();
    double build_H_time_total = std::chrono::nanoseconds(t11 - t00).count()/1e9;
    // printf("--- build vipss j total  time : %f \n", build_j_time_total_);
    // printf("--- add  J matrix to triplet vector time : %f \n", add_ele_to_vector_time);
    // printf("--- build final_h  from triplets vector time : %f \n", build_h_from_tris_time);
    printf("--- build_H_time_total sum  time : %f \n", build_H_time_total);
    // G_VP_stats.build_H_total_time_ += build_H_time_total;
    // G_VP_stats.cal_cluster_J_total_time_ += build_j_time_total_;
    // G_VP_stats.add_J_ele_to_triplet_vector_time_ += add_ele_to_vector_time;
    // G_VP_stats.build_eigen_final_h_from_tris_time_ += build_h_from_tris_time;

}

void LocalVipss::SetBestNormalsWithHmat()
{
    size_t npt = points_.size();
    Eigen::VectorXd sg_score_init(npt* 4);
    Eigen::VectorXd sg_dist_init(npt* 4);
    for(size_t i = 0; i < npt; ++i)
    {
        sg_score_init(i) =  s_vals_score_[i];
        sg_score_init(i + npt)     = out_normals_[3 * i + 0];
        sg_score_init(i + npt * 2) = out_normals_[3 * i + 1];
        sg_score_init(i + npt * 3) = out_normals_[3 * i + 2];
        
        sg_dist_init(i) = s_vals_dist_[i];
        sg_dist_init(i + npt)     = out_normals_dist_[3 * i + 0];
        sg_dist_init(i + npt * 2) = out_normals_dist_[3 * i + 1];
        sg_dist_init(i + npt * 3) = out_normals_dist_[3 * i + 2];

        // if(s_vals_score_[i] != s_vals_dist_[i])
        // {
        //     std::cout << " s vals :  " << s_vals_score_[i] <<" " <<  s_vals_dist_[i] << std::endl;
        // }
    }

    Eigen::VectorXd temp = (sg_score_init.transpose() * final_h_eigen_).transpose();
    double res_score = temp.dot(sg_score_init);

    temp = (sg_dist_init.transpose() * final_h_eigen_).transpose();
    double res_dist = temp.dot(sg_dist_init);
    std::cout << " init normal energy score : " << res_score << " energy dist : " << res_dist << std::endl;

    if(res_dist < res_score)
    {
        out_normals_ = out_normals_dist_;
    }
}

inline std::vector<double> GetClusterNormalsFromIds
                            (const std::vector<int>& pt_ids, const std::vector<double>& all_normals) 
{
    std::vector<double> normals(pt_ids.size() * 3);
    for(int i = 0; i < pt_ids.size(); ++i) 
    {
        // auto &pt = points_[id];
        auto id = pt_ids[i];
        normals[3*i] = (all_normals[3 * id]);
        normals[3*i + 1] = (all_normals[3 * id + 1]);
        normals[3*i + 2] = (all_normals[3 * id + 2]);
    }
    return normals;
}

inline std::vector<double> GetClusterSvalsFromIds(const std::vector<int>& pt_ids, 
                        const std::vector<double>& all_svals) 
{
    std::vector<double> svals;
    for(size_t id : pt_ids)
    {
        svals.push_back(all_svals[id]);
    }
    return svals;
}

std::vector<std::shared_ptr<RBF_Core>> LocalVipss::node_rbf_vec_;
VoronoiGen LocalVipss::voro_gen_;

void LocalVipss::BuildHRBFPerNode()
{
    auto t00 = Clock::now(); 
// if(0)
    size_t npt = this->points_.size();
    printf("cluster num : %lu \n", npt);
    size_t cluster_num = points_.size();
    double time_sum = 0;
    node_rbf_vec_.resize(cluster_num);
    bool use_partial_vipss = false;
    vipss_api_.user_lambda_ = user_lambda_;
    // printf("cluster num : %lu \n", cluster_num);
#pragma omp parallel for 
    for(int i =0; i < cluster_num; ++i)
    {
        // std::vector<double> cluster_pt_vec = VoronoiGen::cluster_init_pts_[i];
        const auto& cluster_pt_ids = VoronoiGen::cluster_init_pids_[i];
        std::vector<double> cluster_pt_vec(cluster_pt_ids.size() * 3);  
        for(int j = 0; j < cluster_pt_ids.size(); ++j)
        {
            cluster_pt_vec[3*j]     = points_[cluster_pt_ids[j]][0];
            cluster_pt_vec[3*j + 1] = points_[cluster_pt_ids[j]][1];
            cluster_pt_vec[3*j + 2] = points_[cluster_pt_ids[j]][2];
        }
        
        std::vector<double> cluster_nl_vec = GetClusterNormalsFromIds(cluster_pt_ids, out_normals_);
        auto cluster_sv_vec = GetClusterSvalsFromIds(cluster_pt_ids, s_vals_);

        // std::string cluster_pt_path = out_dir_ + "cluster_pts/" + std::to_string(i) + "_cluster_color"; 
        // // std::cout << " Cluster pts : " << cluster_pt_path << std::endl;
        // output_opt_pts_with_color(cluster_pt_vec, cluster_sv_vec, cluster_pt_path);
        // printf("cluster_sv_vec size : %ld \n", cluster_sv_vec.size());
        node_rbf_vec_[i] = std::make_shared<RBF_Core>();
        vipss_api_.build_cluster_hrbf(cluster_pt_vec, cluster_nl_vec, cluster_sv_vec, node_rbf_vec_[i]);
    }
    // s_vals_.clear();
    // out_normals_.clear();

    auto t11 = Clock::now();
    double build_HRBF_time_total = std::chrono::nanoseconds(t11 - t00).count()/1e9;
    printf("--- build_HRBF_time_total sum  time : %f \n", build_HRBF_time_total);
}



// double LocalVipss::NodeDistanceFunction(const tetgenmesh::point nn_pt, const tetgenmesh::point cur_pt) const
// {
//     if(voro_gen_.point_id_map_.find(nn_pt) != voro_gen_.point_id_map_.end())
//     {
//         size_t pid = voro_gen_.point_id_map_[nn_pt];
//         return node_rbf_vec_[pid]->Dist_Function(cur_pt);
//     } 
//     return 0;
// }

double LocalVipss::NatureNeighborDistanceFunctionOMP(const tetgenmesh::point cur_pt) const
{
    std::vector<tetgenmesh::point> nei_pts;
    auto tn0 = Clock::now();
    // printf("start to get nei pts \n");
    voro_gen_.GetVoronoiNeiPts(cur_pt, nei_pts);
    auto tn1 = Clock::now();
    double search_nn_time = std::chrono::nanoseconds(tn1 - tn0).count()/1e9;
    search_nn_time_sum_ += search_nn_time;
    // printf("nn num %ld \n", nei_pts.size());
    size_t nn_num = nei_pts.size();
    auto t0 = Clock::now();
    arma::vec nn_dist_vec_(nn_num);
    arma::vec nn_volume_vec_(nn_num);
    ave_voxel_nn_pt_num_ += nn_num;
    const std::vector<double*>& all_pts = points_;
    // int max_nn_num = 1000;
    
    arma::vec nn_pt_dist_vec_(nn_num);

    // bool over_max_nn_num = nn_num > max_nn_num;
    // if(over_max_nn_num)
    // {
    //     #pragma omp parallel for     
    //     for(int ni = 0; ni < nn_num; ++ni)
    //     {
    //         nn_pt_dist_vec_[ni] = PtDistance(cur_pt, nei_pts[ni]);
    //     }
    // }
    // arma::uvec dist_ids = arma::sort_index(nn_pt_dist_vec_);
    
#pragma omp parallel for 
// #pragma omp parallel for shared(nei_pts, cur_pt, nn_dist_vec_, nn_volume_vec_) private(i)    
    for(int i = 0; i < nn_num; ++i)
    {
        // if(i < max_nn_num)
        {
            // auto nn_pt = over_max_nn_num ? nei_pts[dist_ids[i]] : nei_pts[i];
            auto nn_pt = nei_pts[i];
            if(VoronoiGen::point_id_map_.find(nn_pt) != VoronoiGen::point_id_map_.end())
            {
                size_t pid = VoronoiGen::point_id_map_[nn_pt];
                // std::cout << " n id 1111  " << pid << std::endl;
                const arma::vec& a = node_rbf_vec_[pid]->a;
                const arma::vec& b = node_rbf_vec_[pid]->b;
                const auto& cluster_pids = VoronoiGen::cluster_init_pids_[pid];
                // std::cout << " n id 2222  " << pid << std::endl;
                // std::cout << " cluster_pids size  " << cluster_pids.size() << std::endl;
                // std::cout << " all_pts size  " << all_pts.size() << std::endl;
                // std::cout << " a size  " << a.size() << std::endl;
                // std::cout << " b size  " << b.size() << std::endl;
                // nn_dist_vec_[i] = HRBF_Dist_Alone(cur_pt,  a, b, cluster_pids, all_pts);
                nn_dist_vec_[i] = node_rbf_vec_[pid]->Dist_Function(cur_pt[0], cur_pt[1], cur_pt[2]);
                // std::cout << " cur pt dist :  " << nn_dist_vec_[i] << std::endl;
                int thread_id = omp_get_thread_num();
                // nn_volume_vec_[i] = 1.0;
                nn_volume_vec_[i] = voro_gen_.CalTruncatedCellVolumePassOMP(cur_pt, nn_pt, thread_id); 
            } 
        }
    }

    // std::cout << "nn_dist_vec_ " << nn_dist_vec_[0] << " " << nn_dist_vec_[1] << " " << nn_dist_vec_[2] << " "  << std::endl;

    auto t1 = Clock::now();
    double pass_time = std::chrono::nanoseconds(t1 - t0).count()/1e9;
    pass_time_sum_ += pass_time;

    double volume_sum = arma::accu(nn_volume_vec_);
    if(volume_sum > 1e-16)
    {
        return arma::dot(nn_dist_vec_, nn_volume_vec_) / volume_sum;
    } 
    // return dummy_sign_;;
    return 0;
}

double LocalVipss::NatureNeighborGradientOMP(const tetgenmesh::point cur_pt, double* gradient) const
{
    std::vector<tetgenmesh::point> nei_pts;
    auto tn0 = Clock::now();
    // printf("start to get nei pts \n");
    voro_gen_.GetVoronoiNeiPts(cur_pt, nei_pts);
    auto tn1 = Clock::now();
    double search_nn_time = std::chrono::nanoseconds(tn1 - tn0).count()/1e9;
    search_nn_time_sum_ += search_nn_time;
    // printf("nn num %ld \n", nei_pts.size());
    size_t nn_num = nei_pts.size();
    int i;  
    auto t0 = Clock::now();
    arma::vec nn_dist_vec_(nn_num);
    arma::vec nn_volume_vec_(nn_num);
    arma::vec nn_compare_volume_vec_(nn_num);
    arma::vec gx_vec_(nn_num);
    arma::vec gy_vec_(nn_num);
    arma::vec gz_vec_(nn_num);
    std::vector<arma::vec3> nn_vol_grads_(nn_num);
    
    ave_voxel_nn_pt_num_ += nn_num;
    const std::vector<double*>& all_pts = points_;

#pragma omp parallel for
//  shared(node_rbf_vec_, voro_gen_, VoronoiGen::point_id_map_, nei_pts, cur_pt, nn_dist_vec_, nn_volume_vec_) private(i)
    for( i = 0; i < nn_num; ++i)
    {
        auto nn_pt = nei_pts[i];
        if(VoronoiGen::point_id_map_.find(nn_pt) != VoronoiGen::point_id_map_.end())
        {
            size_t pid = VoronoiGen::point_id_map_[nn_pt];
            const arma::vec& a = node_rbf_vec_[pid]->a;
            const arma::vec& b = node_rbf_vec_[pid]->b;
            const auto& cluster_pids = VoronoiGen::cluster_init_pids_[pid];
            // double dist_val = HRBF_Dist_Alone(cur_pt,  a, b, cluster_pids, all_pts);
            // double dist_val = node_rbf_vec_[pid]->Dist_Function(cur_pt[0], cur_pt[1], cur_pt[2]);
            
            // nn_dist_vec_[i] = 1;
            double gx, gy, gz;
            nn_dist_vec_[i] = node_rbf_vec_[pid]->evaluate_gradient(cur_pt[0],cur_pt[1], cur_pt[2], gx, gy, gz);
            double pure_dist = node_rbf_vec_[pid]->Dist_Function(cur_pt[0], cur_pt[1], cur_pt[2]);

            
            gx_vec_[i] = gx;
            gy_vec_[i] = gy;
            gz_vec_[i] = gz;

            int thread_id = omp_get_thread_num();
            // nn_volume_vec_[i] = voro_gen_.CalTruncatedCellVolumePassOMP(cur_pt, nn_pt, thread_id); 
            // nn_compare_volume_vec_[i] = voro_gen_.CalTruncatedCellVolumePassOMP(cur_pt, nn_pt, thread_id);
            arma::vec3 vol_grad; 
            nn_volume_vec_[i] = voro_gen_.CalTruncatedCellVolumeGradientOMP(cur_pt, nn_pt, vol_grad, thread_id);
            nn_vol_grads_[i] = vol_grad;
            
        }
    }

    // std::cout << "  nn_volume_vec_[i] " <<  nn_volume_vec_[i] << "  , nn_compare_volume_vec_[i]" << nn_compare_volume_vec_[i] << std::endl;

    // std::cout << " gradient nn_dist_vec_ " << nn_dist_vec_[0] << " " << nn_dist_vec_[1] << " " << nn_dist_vec_[2] << " "  << std::endl;

    auto t1 = Clock::now();
    double pass_time = std::chrono::nanoseconds(t1 - t0).count()/1e9;
    pass_time_sum_ += pass_time;
    double volume_sum = arma::accu(nn_volume_vec_);
    arma::vec3 vol_grads_sum = {0, 0, 0};
    for(const auto& cur_grad : nn_vol_grads_)
    {
        vol_grads_sum += cur_grad;
    }

    gradient[0] = 0;
    gradient[1] = 0;
    gradient[2] = 0;

    if(volume_sum > 1e-20)
    {

        gradient[0] += arma::dot(nn_volume_vec_, gx_vec_) / volume_sum;
        gradient[1] += arma::dot(nn_volume_vec_, gy_vec_) / volume_sum;
        gradient[2] += arma::dot(nn_volume_vec_, gz_vec_) / volume_sum;

        arma::vec3 partial_grad = {0, 0, 0};
        for(int vi = 0; vi < nn_num; ++vi)
        {
            partial_grad += nn_dist_vec_[vi] / (volume_sum * volume_sum) * ( volume_sum * nn_vol_grads_[vi] - nn_volume_vec_[vi] * vol_grads_sum);
        }
        gradient[0] += partial_grad[0];
        gradient[1] += partial_grad[1];
        gradient[2] += partial_grad[2];

        return arma::dot(nn_dist_vec_, nn_volume_vec_) / volume_sum;
    }
    return 0;
}

arma::mat LocalVipss::NNHRBFHessianOMP(const tetgenmesh::point cur_pt) const
{
    std::vector<tetgenmesh::point> nei_pts;
    auto tn0 = Clock::now();
    // printf("start to get nei pts \n");
    voro_gen_.GetVoronoiNeiPts(cur_pt, nei_pts);
    auto tn1 = Clock::now();
    double search_nn_time = std::chrono::nanoseconds(tn1 - tn0).count()/1e9;
    search_nn_time_sum_ += search_nn_time;
    // printf("nn num %ld \n", nei_pts.size());
    size_t nn_num = nei_pts.size();
    int i;  
    auto t0 = Clock::now();
    arma::vec nn_dist_vec_(nn_num);
    arma::vec nn_volume_vec_(nn_num);
    arma::vec gx_vec_(nn_num);
    arma::vec gy_vec_(nn_num);
    arma::vec gz_vec_(nn_num);
    std::vector<arma::vec3> nn_vol_grads_(nn_num);
    
    ave_voxel_nn_pt_num_ += nn_num;
    const std::vector<double*>& all_pts = points_;
    std::vector<arma::mat> hessian_mats(nn_num);

#pragma omp parallel for
//  shared(node_rbf_vec_, voro_gen_, VoronoiGen::point_id_map_, nei_pts, cur_pt, nn_dist_vec_, nn_volume_vec_) private(i)
    for( i = 0; i < nn_num; ++i)
    {
        auto nn_pt = nei_pts[i];
        hessian_mats[i] = arma::zeros(3, 3); 
        if(VoronoiGen::point_id_map_.find(nn_pt) != VoronoiGen::point_id_map_.end())
        {
            size_t pid = VoronoiGen::point_id_map_[nn_pt];
            // const arma::vec& a = node_rbf_vec_[pid]->a;
            // const arma::vec& b = node_rbf_vec_[pid]->b;
            node_rbf_vec_[pid]->EvaluateHessian(cur_pt[0],cur_pt[1], cur_pt[2], hessian_mats[i]);
            // const std::vector<size_t>& cluster_pids = VoronoiGen::cluster_init_pids_[pid];
            // nn_dist_vec_[i] = HRBF_Dist_Alone(cur_pt,  a, b, cluster_pids, all_pts);
            // nn_dist_vec_[i] = 1;
            int thread_id = omp_get_thread_num();
            nn_volume_vec_[i] = voro_gen_.CalTruncatedCellVolumePassOMP(cur_pt, nn_pt, thread_id);
        }
    }

    auto t1 = Clock::now();
    double pass_time = std::chrono::nanoseconds(t1 - t0).count()/1e9;
    pass_time_sum_ += pass_time;
    double volume_sum = arma::accu(nn_volume_vec_);
    arma::mat hessian_mat = arma::zeros(3,3);
    for( i = 0; i < nn_num; ++i)
    {
        hessian_mat += hessian_mats[i] * nn_volume_vec_[i];
    }
    if(volume_sum > 1e-20)
    {
        hessian_mat /= volume_sum;
    }
    return hessian_mat;
}


void LocalVipss::InitNormalWithVipss()
{
    // size_t cluster_num = cluster_cores_mat_.n_cols;
    // printf("cluster num : %zu \n", cluster_num);
    vipss_time_stats_.clear();
    double vipss_sum = 0;
    auto t_init = Clock::now();
    size_t npt = this->points_.size();
    size_t ele_num = arma::accu(VoronoiGen::cluster_size_vec_);
    std::vector<Triplet> cluster_normals_xele(ele_num);
    std::vector<Triplet> cluster_normals_yele(ele_num);
    std::vector<Triplet> cluster_normals_zele(ele_num);
    cluster_normal_x_.resize(npt, npt);
    cluster_normal_y_.resize(npt, npt);
    cluster_normal_z_.resize(npt, npt);
    s_vals_.resize(npt, 0.0);
    G_VP_stats.ave_cluster_size = double(ele_num) / double(npt);
    
    double std_dev = 0;
    for(int i = 0; i < npt; ++i)
    {
        double delt_val = VoronoiGen::cluster_size_vec_[i+1] - G_VP_stats.ave_cluster_size;
        std_dev += delt_val * delt_val;
    }
    G_VP_stats.cluster_std_dev = sqrt(std_dev/double(npt));
    G_VP_stats.max_cluster_size = (int) arma::max(VoronoiGen::cluster_size_vec_);
    std::cout << " ave cluter pt number :  " << G_VP_stats.ave_cluster_size << std::endl;
    std::cout << " stand dev :  " << G_VP_stats.cluster_std_dev << std::endl;
    std::cout << " max_cluster_size  :  " << G_VP_stats.max_cluster_size << std::endl;

    
#pragma omp parallel for 
    for(int i =0; i < int(npt); ++i)
    {
        // auto vts = VoronoiGen::cluster_init_pts_[i];
        const auto& p_ids = VoronoiGen::cluster_init_pids_[i];
        std::vector<double> vts(p_ids.size() * 3, 0);
        for(int j = 0; j < p_ids.size(); ++j)
        {
            vts[3*j]     = points_[p_ids[j]][0];
            vts[3*j + 1] = points_[p_ids[j]][1];
            vts[3*j + 2] = points_[p_ids[j]][2];
        }
        // cluster_all_pt_ids_[i] = p_ids;
        // auto t1 = Clock::now();
        size_t unit_npt = p_ids.size(); 

        double cur_lambda = user_lambda_;
        if(InitWithPartialVipss)
        {
            cur_lambda = user_lambda_ / double(unit_npt);
        }
        // double cur_lambda = user_lambda_ / double(unit_npt);
        // if(!node_rbf_vec_[i])  
        
        // node_rbf_vec_[i] = std::make_shared<RBF_Core>();
        size_t key_ptn = 1;
        if(InitWithPartialVipss == 0)
        {
            key_ptn = unit_npt;
        }
        std::shared_ptr rbf_temp_ptr = std::make_shared<RBF_Core>();
        InitNormalPartialVipss(vts, key_ptn, rbf_temp_ptr, cur_lambda);

        s_vals_[i] = rbf_temp_ptr->Dist_Function(vts[0], vts[1], vts[2]);
        // auto t2 = Clock::now();
        // double vipss_time = std::chrono::nanoseconds(t2 - t1).count()/1e9;
        auto iterx = cluster_normals_xele.begin() + VoronoiGen::cluster_accum_size_vec_[i];
        auto itery = cluster_normals_yele.begin() + VoronoiGen::cluster_accum_size_vec_[i];
        auto iterz = cluster_normals_zele.begin() + VoronoiGen::cluster_accum_size_vec_[i];
        
        for(size_t p_id = 0; p_id < p_ids.size(); ++p_id)
        {
            size_t v_id = p_ids[p_id];
            *(iterx + p_id) = Triplet(v_id, i, rbf_temp_ptr->out_normals_[3* p_id]);
            *(itery + p_id) = Triplet(v_id, i, rbf_temp_ptr->out_normals_[3* p_id + 1]);
            *(iterz + p_id) = Triplet(v_id, i, rbf_temp_ptr->out_normals_[3* p_id + 2]);
        }
        // printf("Init vipss cluster id  : %d, cluster pt num : %d \n", i, int(p_ids.size()));
    }

    cluster_normal_x_.setFromTriplets(cluster_normals_xele.begin(), cluster_normals_xele.end());
    cluster_normal_y_.setFromTriplets(cluster_normals_yele.begin(), cluster_normals_yele.end());
    cluster_normal_z_.setFromTriplets(cluster_normals_zele.begin(), cluster_normals_zele.end());

    // auto t_init2 = Clock::now();
    // double all_init_time = std::chrono::nanoseconds(t_init2 - t_init).count()/1e9;
    // printf("all init time : %f \n", all_init_time);
    // printf("all init vipss time : %f \n", vipss_sum);
    // cluster_ptn_vipss_time_stats_.push_back(vipss_time_stats_);
}

static LocalVipss* local_vipss_ptr;
int LocalVipss::DistCallNum = 0;
double LocalVipss::DistCallTime = 0.0;

double LocalVipss::NNDistFunction(const R3Pt &in_pt)
{
    DistCallNum ++;
    auto t0 = Clock::now();
    double new_pt[3] = {in_pt[0], in_pt[1], in_pt[2]};
    // double dist = local_vipss_ptr->NatureNeighborDistanceFunction(&(new_pt[0]));
    double dist = local_vipss_ptr->NatureNeighborDistanceFunctionOMP(&(new_pt[0]));
    // std::cout << " dist " << dist << std::endl;
    auto t1 = Clock::now();
    double dist_time = std::chrono::nanoseconds(t1 - t0).count()/1e9;
    DistCallTime += dist_time;
    return dist;
}

std::array<double,3> LocalVipss::NumericGradientFunction(const R3Pt &in_pt)
{
    double x = in_pt[0];
    double y = in_pt[1];
    double z = in_pt[2];
    double h = 0.00001;
    double fxa = LocalVipss::NNDistFunction({x + h, y, z});
    double fxb = LocalVipss::NNDistFunction({x - h, y, z});
    double gx = (fxa - fxb) / (2.0 * h);
    
    // Partial derivative with respect to y
    double fya = LocalVipss::NNDistFunction({x, y + h, z});
    double fyb = LocalVipss::NNDistFunction({x, y - h, z});
    double gy = (fya - fyb) / (2.0 * h);
    
    // Partial derivative with respect to z
    double fza = LocalVipss::NNDistFunction({x, y, z + h});
    double fzb = LocalVipss::NNDistFunction({x, y, z - h});
    double gz = (fza - fzb) / (2.0 * h);
    return {gx, gy, gz};
}

// double LocalVipss::NNDistFunction(const double* in_pt)
// {
//     DistCallNum ++;
//     auto t0 = Clock::now();
//     // double new_pt[3] = {in_pt[0], in_pt[1], in_pt[2]};
//     // double dist = local_vipss_ptr->NatureNeighborDistanceFunction(&(new_pt[0]));
//     double dist = local_vipss_ptr->NatureNeighborDistanceFunctionOMP((const tetgenmesh::point)in_pt);
//     // std::cout << " dist " << dist << std::endl;
//     auto t1 = Clock::now();
//     double dist_time = std::chrono::nanoseconds(t1 - t0).count()/1e9;
//     DistCallTime += dist_time;
//     return dist;
// }


double LocalVipss::NNDistGradient(const R3Pt &in_pt, double* gradient)
{
    DistCallNum ++;
    auto t0 = Clock::now();
    double new_pt[3] = {in_pt[0], in_pt[1], in_pt[2]};
    // double dist = local_vipss_ptr->NatureNeighborDistanceFunction(&(new_pt[0]));
    double dist = local_vipss_ptr->NatureNeighborGradientOMP(&(new_pt[0]), gradient);
    // std::cout << " dist " << dist << std::endl;
    auto t1 = Clock::now();
    double dist_time = std::chrono::nanoseconds(t1 - t0).count()/1e9;
    DistCallTime += dist_time;
    return dist;
}

void LocalVipss::SetThis()
{
    local_vipss_ptr = this;
}


inline double LocalVipss::CalculateScores(const arma::mat& a_normals, const arma::mat& b_normals) const
{
    arma::mat dot_mat = a_normals % b_normals;
    arma::colvec dot_sum = arma::sum(dot_mat, 1);
    double min_project_p = dot_sum.min();
    double min_project_n = -1.0 * dot_sum.max();

    min_project_p = min_project_n > min_project_p ? min_project_n : min_project_p;

    // flip_normal_ = false;
    // if(min_project_n > min_project_p)
    // {
    //     // flip_normal_ = true;
    //     min_project_p = min_project_n;
    // }
    min_project_p = std::min(1.0, std::max(-1.0, min_project_p));
    double angle = acos (min_project_p) * Anlge_PI_Rate ;
    return angle;
}

inline bool LocalVipss::IsFlipNormal(const arma::mat& a_normals, const arma::mat& b_normals) const
{
    arma::mat dot_mat = a_normals % b_normals;
    arma::colvec dot_sum = arma::sum(dot_mat, 1);
    double min_project_p = dot_sum.min();
    double min_project_n = -1.0 * dot_sum.max();
    return min_project_n > min_project_p? true : false;

    // flip_normal_ = false;
    // if(min_project_n > min_project_p)
    // {
    //     flip_normal_ = true;
    //     min_project_p = min_project_n;
    // }
    // min_project_p = std::min(1.0, std::max(-1.0, min_project_p));
    // double angle = acos (min_project_p) * 180.0 / M_PI2 ;
    // return angle;
}

inline double LocalVipss::CalculateScores(const std::vector<arma::vec3>& a_normals, const std::vector<arma::vec3>& b_normals) const
{

    double min_project_p = 1.0;
    for(size_t i = 0; i < a_normals.size(); ++i)
    {
        double ab_proj = arma::dot(a_normals[i], b_normals[i]);
        min_project_p = min_project_p < ab_proj ? min_project_p : ab_proj;
    }
    
    double min_project_n = 1.0;
    for(size_t i = 0; i < a_normals.size(); ++i)
    {
        double ab_proj = arma::dot(a_normals[i], - 1.0 * b_normals[i]);
        min_project_n = min_project_n < ab_proj ? min_project_n : ab_proj;
    }
    // min_project_n > min_project_p 
    // flip_normal_ = false;
    // if(min_project_n > min_project_p)
    // {
    //     flip_normal_ = true;
    //     min_project_p = min_project_n;
    // }
    min_project_p = min_project_n > min_project_p ? min_project_n : min_project_p;
    min_project_p = std::max(-1.0, min_project_p);

    double angle = acos (min_project_p) * Anlge_PI_Rate ;
    return angle;
}

inline bool LocalVipss::FlipClusterNormalSimple(size_t c_a, size_t c_b) const
{
    arma::vec3 normal_maa;
    arma::vec3 normal_mba;
    {
        normal_maa(0) =  cluster_normal_x_.coeff(c_a, c_a);
        normal_maa(1) =  cluster_normal_y_.coeff(c_a, c_a);
        normal_maa(2) =  cluster_normal_z_.coeff(c_a, c_a);
        normal_mba(0) =  cluster_normal_x_.coeff(c_a, c_b);
        normal_mba(1) =  cluster_normal_y_.coeff(c_a, c_b);
        normal_mba(2) =  cluster_normal_z_.coeff(c_a, c_b);
    }
    double proj1 = arma::dot( normal_maa , normal_mba); 
    if(proj1 < 0) return true;
    return false;
}



inline bool LocalVipss::FlipClusterNormalSimple2(size_t c_a, size_t c_b) const
{
    const auto& pa = points_[c_a];
    const auto& pb = points_[c_b];
    arma::vec3 vector_ab = {pa[0] - pb[0], pa[1] - pb[1],pa[2] - pb[2]};
    vector_ab = arma::normalise(vector_ab);
    arma::vec3 normal_maa;
    arma::vec3 normal_mbb;
    {
        normal_maa(0) =  cluster_normal_x_.coeff(c_a, c_a);
        normal_maa(1) =  cluster_normal_y_.coeff(c_a, c_a);
        normal_maa(2) =  cluster_normal_z_.coeff(c_a, c_a);
        normal_mbb(0) =  cluster_normal_x_.coeff(c_b, c_b);
        normal_mbb(1) =  cluster_normal_y_.coeff(c_b, c_b);
        normal_mbb(2) =  cluster_normal_z_.coeff(c_b, c_b);
    }
    arma::vec3 ave_normal =  arma::normalise(normal_maa + normal_mbb);
    double proj = abs(arma::dot(ave_normal, vector_ab));

    if(proj > 0.7) return true;
    return false;
}

inline bool LocalVipss::FlipClusterNormal2(size_t c_a, size_t c_b) const
{
    const auto& cluster_a = voro_gen_.cluster_init_pids_[c_a];
    const auto& cluster_b = voro_gen_.cluster_init_pids_[c_b];
    std::unordered_set<int> cluster_a_set(cluster_a.begin(), cluster_a.end());
    double dot_sign_sum = 0;
    // for (auto pid: cluster_b)
    for (auto pid: {c_a, c_b})
    {
        // if(cluster_a_set.find(pid) != cluster_a_set.end()) 
        {
            arma::vec3 normal_ma_p;
            arma::vec3 normal_mb_p;
            {
                normal_ma_p(0) =  cluster_normal_x_.coeff(pid, c_a);
                normal_ma_p(1) =  cluster_normal_y_.coeff(pid, c_a);
                normal_ma_p(2) =  cluster_normal_z_.coeff(pid, c_a);
                normal_mb_p(0) =  cluster_normal_x_.coeff(pid, c_b);
                normal_mb_p(1) =  cluster_normal_y_.coeff(pid, c_b);
                normal_mb_p(2) =  cluster_normal_z_.coeff(pid, c_b);
            }
            double proj = arma::dot( normal_ma_p , normal_mb_p); 
            dot_sign_sum +=  proj;
        }
    }
    if(dot_sign_sum < 0) return true;
    return false;
}

inline bool LocalVipss::FlipClusterNormal(size_t c_a, size_t c_b) const
{
    // const auto& core_ids_a = cluster_core_pt_ids_[c_a];
    // const auto& core_ids_b = cluster_core_pt_ids_[c_b];
    const auto& core_ids_a = {c_a};
    const auto& core_ids_b = {c_b};
    std::vector<size_t> valid_ids;
    for(auto id : core_ids_a)
    {
        if(cluster_normal_x_.coeff(id, c_b) != 0)
        {
            valid_ids.push_back(id); 
        }
    }
    for(auto id : core_ids_b)
    {
        if(cluster_normal_x_.coeff(id, c_a) != 0)
        {
            valid_ids.push_back(id); 
        }
    }
    size_t p_size = valid_ids.size();
    std::cout << "p size " << p_size << std::endl;
    arma::mat normal_ma(p_size, 3);
    arma::mat normal_mb(p_size, 3);
    for(size_t i = 0; i < valid_ids.size(); ++i)
    {
        size_t id =  valid_ids[i];
        normal_ma(i, 0) =  cluster_normal_x_.coeff(id, c_a);
        normal_ma(i, 1) =  cluster_normal_y_.coeff(id, c_a);
        normal_ma(i, 2) =  cluster_normal_z_.coeff(id, c_a);

        normal_mb(i, 0) =  cluster_normal_x_.coeff(id, c_b);
        normal_mb(i, 1) =  cluster_normal_y_.coeff(id, c_b);
        normal_mb(i, 2) =  cluster_normal_z_.coeff(id, c_b);
    }
    return IsFlipNormal(normal_ma, normal_mb);
}

inline double LocalVipss::CalculateClusterPairScore(size_t c_a, size_t c_b, bool& flip) const
{
    // const auto& core_ids_a = cluster_core_pt_ids_[c_a];
    // const auto& core_ids_b = cluster_core_pt_ids_[c_b];
    const auto& core_ids_a = {c_a};
    const auto& core_ids_b = {c_b};
    std::vector<size_t> valid_ids;
    for(auto id : core_ids_a)
    {
        if(cluster_normal_x_.coeff(id, c_b) != 0)
        {
            valid_ids.push_back(id); 
        }
    }
    for(auto id : core_ids_b)
    {
        if(cluster_normal_x_.coeff(id, c_a) != 0)
        {
            valid_ids.push_back(id); 
        }
    }
    size_t p_size = valid_ids.size();
    arma::mat normal_ma(p_size, 3);
    arma::mat normal_mb(p_size, 3);

    for(size_t i = 0; i < valid_ids.size(); ++i)
    {
        size_t id =  valid_ids[i];
        normal_ma(i, 0) =  cluster_normal_x_.coeff(id, c_a);
        normal_ma(i, 1) =  cluster_normal_y_.coeff(id, c_a);
        normal_ma(i, 2) =  cluster_normal_z_.coeff(id, c_a);

        normal_mb(i, 0) =  cluster_normal_x_.coeff(id, c_b);
        normal_mb(i, 1) =  cluster_normal_y_.coeff(id, c_b);
        normal_mb(i, 2) =  cluster_normal_z_.coeff(id, c_b);
    }

    arma::mat dot_mat = normal_ma % normal_mb;
    arma::colvec dot_sum = arma::sum(dot_mat, 1);
    double min_project_p = dot_sum.min();
    double min_project_n = -1.0 * dot_sum.max();

    // min_project_p = min_project_n > min_project_p ? min_project_n : min_project_p;

    // flip_normal_ = false;
    if(min_project_n > min_project_p)
    {
        flip = true;
        min_project_p = min_project_n;
    }
    min_project_p = std::min(1.0, std::max(-1.0, min_project_p));
    double angle = acos (min_project_p) * Anlge_PI_Rate ;
    return angle;
}


void LocalVipss::CalculateClusterNeiScores(bool is_init)
{
    size_t c_num = points_.size();
    // cluster_scores_vec_.resize(c_num);
    int score_ele_num = arma::accu(VoronoiGen::cluster_size_vec_) - c_num;
    std::vector<Triplet> score_eles(score_ele_num);

#pragma omp parallel for
    for(int i = 0; i < c_num; ++i)
    {
        // InitSingleClusterNeiScores(i);
        const auto& nei_pt_ids = VoronoiGen::cluster_init_pids_[i];
        arma::mat cur_i_mat(2, 3);
        arma::mat cur_n_mat(2, 3);
        cur_i_mat(0, 0) = cluster_normal_x_.coeff(i, i);
        cur_i_mat(0, 1) = cluster_normal_y_.coeff(i, i);
        cur_i_mat(0, 2) = cluster_normal_z_.coeff(i, i);
        auto ele_iter =  score_eles.begin() + VoronoiGen::cluster_accum_size_vec_[i] - i; 
        // for(auto n_pt : nei_pts)
        for(auto n_pid : nei_pt_ids)
        {
            if(n_pid == i) continue;
            cur_i_mat(1, 0) = cluster_normal_x_.coeff(n_pid, i);
            cur_i_mat(1, 1) = cluster_normal_y_.coeff(n_pid, i);
            cur_i_mat(1, 2) = cluster_normal_z_.coeff(n_pid, i);

            cur_n_mat(0, 0) = cluster_normal_x_.coeff(i, n_pid);
            cur_n_mat(0, 1) = cluster_normal_y_.coeff(i, n_pid);
            cur_n_mat(0, 2) = cluster_normal_z_.coeff(i, n_pid);

            cur_n_mat(1, 0) = cluster_normal_x_.coeff(n_pid, n_pid);
            cur_n_mat(1, 1) = cluster_normal_y_.coeff(n_pid, n_pid);
            cur_n_mat(1, 2) = cluster_normal_z_.coeff(n_pid, n_pid);

            arma::mat dot_res = cur_i_mat % cur_n_mat;
            arma::vec dot_sum = arma::sum(dot_res, 1); 
            double s1 = dot_sum.min();
            double s2 = -dot_sum.max();
            if(s2 > s1) 
            {
                s1 = s2;
            }
            double score = std::min(1.0, std::max(-1.0, s1));
            score = acos(score) * Anlge_PI_Rate ;
            *(ele_iter ++) = Triplet(n_pid, i, score);
            // printf("cluster id : %d \n", i);   
        // #pragma omp critical
            // cluster_scores_mat_(n_pid, i) = score;
        // #pragma omp critical
            // cluster_scores_mat_(i, n_pid) = score;
        }
    }
    cluster_scores_mat_.setFromTriplets(score_eles.begin(), score_eles.end());
}

// void LocalVipss::InitAdjacentData()
// {
//     const auto& adjacent_mat_ = voro_gen_.pt_adjecent_mat_;
//     int cols = adjacent_mat_.n_cols;
//     // cluster_adjacent_ids_.resize(cols);
//     cluster_pt_ids_.resize(cols);
//     std::cout << " start to get col vec by id "<< std::endl;
//     for(size_t i = 0; i < cols; ++i)
//     {
//         arma::sp_umat col_vec(adjacent_mat_.col(i));
//         // std::cout << " get col vec by id "<< std::endl;
//         arma::sp_umat::const_iterator start  = col_vec.begin();
//         arma::sp_umat::const_iterator end    = col_vec.end();
//         for(auto iter = start; iter != end; ++iter)
//         {
//             // cluster_adjacent_ids_[i].insert(size_t(iter.row()));
//             cluster_pt_ids_[i].insert(size_t(iter.row()));
//         }
//         // std::cout << " get col vec by id 1"<< std::endl;
//         cluster_pt_ids_[i].insert(i);
//     }
//     // cluster_core_pt_nums_.ones(cols); 
// }

// void LocalVipss::SaveGroupPtsWithColor(const std::string& path)
// {
//     size_t group_size = cluster_cores_mat_.n_cols;
//     std::ofstream pt_file;
//     pt_file.open(path);
//     size_t pt_count = 0;
//     std::cout << " -----save cluster group size :  " << arma::accu(cluster_valid_sign_vec_) << std::endl;
//     for(size_t i = 0; i < group_size; ++i)
//     {
//         if(!cluster_valid_sign_vec_[i]) continue;
//         int base = 1000000;
//         double r = ((double) (rand() % base) / double(base)); r = std::min(sqrt(r) , 1.0);
//         double g = ((double) (rand() % base) / double(base)); g = std::max(sqrt(g) , 0.0);
//         double b = ((double) (rand() % base) / double(base)); b = std::max(sqrt(b) , 0.0);
//         // const arma::sp_umat cluster_col(cluster_cores_mat_.col(i));
//         // arma::sp_umat::const_iterator start = cluster_col.begin();
//         // arma::sp_umat::const_iterator end = cluster_col.end();
//         const auto& cur_pids = cluster_core_pt_ids_vec_[i];
//         // printf("cur cluster core pt num : %ld \n", cur_pids.size());
//         for(auto pid : cur_pids)
//         {
//             auto cur_pt = points_[pid];
//             pt_file << "v " << cur_pt[0] << " " << cur_pt[1] << " " << cur_pt[2] ;
//             pt_file << " " << r << " " << g << " " << b << std::endl;
//         }
//     }
//     pt_file.close();
// }

void LocalVipss::BuildClusterMST(bool use_distance_weight = false)
{
    std::set<std::string> visited_edge_ids;
    std::vector<C_Edege> tree_edges;
    auto cmp = [](const C_Edege& left, const C_Edege& right) { return left.score_ > right.score_; };
    std::priority_queue<C_Edege, std::vector<C_Edege>, decltype(cmp)> edge_priority_queue(cmp);
    C_Edege st_e(0,0);
    edge_priority_queue.push(st_e);
    std::set<size_t> visited_vids;
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
        const arma::sp_umat& adj_row =  voro_gen_.pt_adjecent_mat_.col(cur_pid);
        // printf("row %d contains no zero num : %d \n", cur_pid, adj_row.n_nonzero);
        const arma::sp_umat::const_iterator start = adj_row.begin();
        const arma::sp_umat::const_iterator end = adj_row.end();
        for(auto iter = start; iter != end; ++iter)
        {
            size_t n_id = iter.row();
            if(visited_vids.find(n_id) != visited_vids.end()) continue;
            if(n_id == cur_pid) continue;
            C_Edege edge(cur_pid, n_id);
            edge.score_ = cluster_scores_mat_.coeff(n_id, cur_pid);
            // if(use_distance_weight)
            {
                double dist = PtDistance(points_[n_id], points_[cur_pid]);
                edge.score_ = sqrt(dist) * edge.score_;
            }
            edge_priority_queue.push(edge);
        }
    }
    size_t c_num = points_.size();
    cluster_MST_mat_.resize(c_num, c_num);
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

void LocalVipss::FlipClusterNormalsByMST()
{
    // size_t c_num = cluster_cores_mat_.n_cols;
    size_t c_num = points_.size();
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
            if(FlipClusterNormalSimple(cur_cid, n_cid))
            // if(FlipClusterNormal2(cur_cid, n_cid))
            {
                cluster_normal_x_.col(n_cid) *= -1.0;
                cluster_normal_y_.col(n_cid) *= -1.0;
                cluster_normal_z_.col(n_cid) *= -1.0;
                s_vals_[n_cid] *= -1.0;
            }
            cluster_queued_ids.push(n_cid);
        }
    }
}

void LocalVipss::FlipClusterNormalsFromConvexHull()
{
    std::vector<double> out_hull_pts;
    arma::vec3 hull_c= {0, 0, 0};   
    std::vector<arma::vec3> hull_normals;
    std::unordered_set<size_t> visited_ids;
    std::unordered_set<size_t> flipped_ids;
    for(auto pt : voro_gen_.convex_hull_pts_)
    {
        hull_c[0] += pt[0];
        hull_c[1] += pt[1];
        hull_c[2] += pt[2];
    }
    if(!voro_gen_.convex_hull_pts_.empty())
    {
        hull_c = hull_c / voro_gen_.convex_hull_pts_.size();
    }

    auto cmp = [](const C_Edege& left, const C_Edege& right) 
    {
        // if (left.depth == right.depth)
        //     return left.score_ > right.score_; 
        // return  left.depth > right.depth; 
        return left.score_ > right.score_;
    };
    std::priority_queue<C_Edege, std::vector<C_Edege>, decltype(cmp)> edge_priority_queue(cmp);
    std::queue<size_t> candid_pids;
    for(auto pt : voro_gen_.convex_hull_pts_)
    {
        size_t pid = voro_gen_.point_id_map_[pt];
        double nx =  cluster_normal_x_.coeff(pid, pid);
        double ny =  cluster_normal_y_.coeff(pid, pid);
        double nz =  cluster_normal_z_.coeff(pid, pid);
        double dx = pt[0] - hull_c[0];
        double dy = pt[1] - hull_c[1];
        double dz = pt[2] - hull_c[2];
        double dot_val = nx * dx + ny * dy + nz * dz;
        if(dot_val < 0)
        {
            cluster_normal_x_.col(pid) *= -1.0;
            cluster_normal_y_.col(pid) *= -1.0;
            cluster_normal_z_.col(pid) *= -1.0;
        }
        // flipped_ids.insert(pid);
        visited_ids.insert(pid); 
      
        const arma::sp_umat& adj_row =  voro_gen_.pt_adjecent_mat_.col(pid);
        const arma::sp_umat::const_iterator start = adj_row.begin();
        const arma::sp_umat::const_iterator end = adj_row.end();
        for(auto iter = start; iter != end; ++iter)
        {
            size_t n_id = iter.row();
            if(visited_ids.find(n_id) != visited_ids.end()) continue;
            if(n_id == pid) continue;
            C_Edege edge(pid, n_id);
            double dist = PtDistance(points_[n_id], points_[pid]);
            edge.score_ = cluster_scores_mat_.coeff(n_id, pid) ;
            edge_priority_queue.push(edge);
        }
    }
    while(!edge_priority_queue.empty())
    {
        C_Edege cur_e = edge_priority_queue.top();
        edge_priority_queue.pop();
        if(visited_ids.find(cur_e.c_b_ ) != visited_ids.end()) continue;
        visited_ids.insert(cur_e.c_b_);
        if(FlipClusterNormalSimple(cur_e.c_a_, cur_e.c_b_))
        {
            cluster_normal_x_.col(cur_e.c_b_) *= -1.0;
            cluster_normal_y_.col(cur_e.c_b_) *= -1.0;
            cluster_normal_z_.col(cur_e.c_b_) *= -1.0;
            s_vals_[cur_e.c_b_] *= -1.0;
        }
        const arma::sp_umat& adj_row =  voro_gen_.pt_adjecent_mat_.col(cur_e.c_b_);
        const arma::sp_umat::const_iterator start = adj_row.begin();
        const arma::sp_umat::const_iterator end = adj_row.end();
        for(auto iter = start; iter != end; ++iter)
        {
            size_t n_id = iter.row();
            if(visited_ids.find(n_id) != visited_ids.end()) continue;
            if(n_id == cur_e.c_b_) continue;
            C_Edege edge(cur_e.c_b_, n_id);
            double dist = PtDistance(points_[n_id], points_[cur_e.c_b_]);
            edge.score_ = cluster_scores_mat_.coeff(n_id, cur_e.c_b_) ;
            edge.depth = cur_e.depth + 1;
            edge_priority_queue.push(edge);
        }
    }
}


std::vector<double> LocalVipss::GetInitPts() const
{
    size_t npt = points_.size();
    std::vector<double> pts;
    pts.resize(npt*3);
    for(size_t i = 0; i < npt; ++i)
    {
        const auto& pt = points_[i];
        pts[i *3]      = pt[0];
        pts[i *3 + 1]  = pt[1];
        pts[i *3 + 2]  = pt[2];
    }
    return pts;
}

std::vector<double> LocalVipss::GetInitNormals() const
{

    size_t npt = points_.size();
    std::vector<double> normals;
    normals.resize(npt*3);
    for(size_t i = 0; i < npt; ++i)
    {
        normals[i *3]     = cluster_normal_x_.coeff(i, i);
        normals[i *3 + 1] = cluster_normal_y_.coeff(i, i);
        normals[i *3 + 2] = cluster_normal_z_.coeff(i, i);
    }
    return normals;
}

void LocalVipss::OuputPtN(const std::string& out_path, bool orient_normal)
{
    out_pts_.clear();
    out_normals_.clear();
    std::vector<double>& pts = out_pts_;
    std::vector<double>& normals = out_normals_;

    size_t npt = points_.size();
    normals.resize(npt*3);

    for(size_t i = 0; i < npt; ++i)
    {
        auto& pt = points_[i];
        pts.push_back(pt[0]);
        pts.push_back(pt[1]);
        pts.push_back(pt[2]);
    }

    for(size_t i = 0; i < npt; ++i)
    {
        // const arma::sp_umat new_row(cluster_cores_mat_.col(i));
        // const arma::sp_umat::const_iterator start = new_row.begin();
        // const arma::sp_umat::const_iterator end = new_row.end();
        // for(auto iter = start; iter != end; ++iter)
        {
            // size_t p_id = iter.row();     
            normals[i *3]     = cluster_normal_x_.coeff(i, i);
            normals[i *3 + 1] = cluster_normal_y_.coeff(i, i);
            normals[i *3 + 2] = cluster_normal_z_.coeff(i, i);
        }
    }
    
    // for(size_t i = 0; i < cluster_cores_mat_.n_cols; ++i)
    // {
    //     // printf("cluster id : %d \n", i);
    //     // const arma::sp_ucolvec cur_row = cluster_cores_mat_.col(i);
    //     const arma::sp_umat new_row(cluster_cores_mat_.col(i));
    //     const arma::sp_umat::const_iterator start = new_row.begin();
    //     const arma::sp_umat::const_iterator end = new_row.end();
    //     for(auto iter = start; iter != end; ++iter)
    //     {
    //         size_t p_id = iter.row();
            
    //         // printf("p_id : %d \n", p_id);
    //         // printf("non zero count : %d \n", new_row.n_nonzero);
    //         auto pt = points_[p_id];
    //         pts.push_back(pt[0]);
    //         pts.push_back(pt[1]);
    //         pts.push_back(pt[2]);
    //         normals.push_back(cluster_normal_x_(p_id, i));
    //         normals.push_back(cluster_normal_y_(p_id, i));
    //         normals.push_back(cluster_normal_z_(p_id, i));
    //     }
    // }
    // printf("out pt size : %zu \n", pts.size() / 3);

    // if(orient_normal)
    // {
    //     ORIENT::OrientPointNormals(pts, normals);
    // }
    
    // writePLYFile_VN(out_path, pts, normals);
}

void LocalVipss::WriteVipssTimeLog()
{
    std::ofstream myfile;
    std::string csv_path = out_dir_ + "_vipss_time.csv";
    myfile.open(csv_path);
    
    double time_sum = 0.0;
    size_t count = 0;
    for(auto& v_times: cluster_ptn_vipss_time_stats_)
    {
        for(auto& ele : v_times)
        {
            time_sum += ele.second;
            myfile << std::to_string(count) << ",";
            myfile << std::to_string(ele.first) << ",";
            myfile << std::to_string(ele.second) << std::endl;
        }
        count ++;
    }
    printf("Vipss total time : %f \n", time_sum);
}

// void LocalVipss::SaveCluster()
// {
//     std::vector<std::vector<P3tr>> key_pts_vec;
//     for(size_t i = 0; i < cluster_cores_mat_.n_cols; ++i)
//     {
//         const arma::sp_umat& cur_row(cluster_cores_mat_.col(i));
//         if(cur_row.n_nonzero < 3) continue;
        
//         const arma::sp_umat::const_iterator start = cur_row.begin();
//         const arma::sp_umat::const_iterator end = cur_row.end();
//         std::set<P3tr> key_pts;
//         for(auto iter = start; iter != end; ++iter)
//         {
//             key_pts.insert(points_[iter.row()]);
//         } 
//         key_pts_vec.push_back(std::vector<P3tr>(key_pts.begin(), key_pts.end()) );
//     }
//     std::sort(key_pts_vec.begin(), key_pts_vec.end(), [](const auto& a, const auto& b)
//     {
//         return a.size() > b.size();
//     });
//     std::vector<std::vector<P3tr>> top_k_key_pts_vec;
//     for(size_t i = 0; i < top_k_ && i < key_pts_vec.size(); ++i)
//     {
//         top_k_key_pts_vec.push_back(key_pts_vec[i]);
//     }
//     std::string out_path = out_dir_ + filename_ + "_cluster_core_pts";
//     // SaveClusterCorePts(out_path, top_k_key_pts_vec);

// }



// void LocalVipss::SaveClusterPts(const std::string& path,
//                             const std::vector<P3tr>& key_pts, 
//                             const std::vector<P3tr>& nei_pts)
// {
//     std::vector<double> pts;
//     std::vector<uint8_t> colors;
//     size_t key_num = key_pts.size();
//     size_t nei_num = nei_pts.size();
//     pts.resize(3*(key_num + nei_num));
//     colors.resize(3*(key_num + nei_num));
//     for(size_t i = 0; i < key_num + nei_num; ++i)
//     {
//         if(i < key_num)
//         {
//             pts[3*i] = key_pts[i][0];
//             pts[3*i + 1] = key_pts[i][1];
//             pts[3*i + 2] = key_pts[i][2];
//             colors[3*i] = 255;
//             colors[3*i + 1] = 0;
//             colors[3*i + 2] = 0;
//         } else {
//             size_t id = i - key_num;
//             pts[3*id] = nei_pts[id][0];
//             pts[3*id + 1] = nei_pts[id][1];
//             pts[3*id + 2] = nei_pts[id][2];
//             colors[3*id] = 0;
//             colors[3*id + 1] = 0;
//             colors[3*id + 2] = 255;
//         } 
//     }
//     writePLYFile_CO(path, pts, colors);
// }

void LocalVipss::SamplePtsWithOctree(const std::vector<double>& pts, int depth)
{
    std::vector<double> sampled_pts;
    size_t c_size = points_.size();
    SimOctree::SimpleOctree octree;
    std::cout << " start to init octree " << std::endl;
    octree.InitOctTree(pts, depth);

    std::cout << " start to split octree leaf nodes " << std::endl;
    octree.SplitLeafNode(2);

    octree_leaf_pts_ = octree.GetLeafMapPts();
    octree_split_leaf_pts_ = octree.GetSplitLeafMapPts();

    std::string octree_sample_path = out_dir_  + filename_ +  "_octree_sample.xyz";
    std::cout << "save octree sample result to file : " << octree_sample_path << std::endl;
    writeXYZ(octree_sample_path, octree_leaf_pts_);

    std::string octree_split_sample_path = out_dir_  + filename_ +  "_octree_sample_split.xyz";
    std::cout << "save octree sample result to file : " << octree_sample_path << std::endl;
    writeXYZ(octree_split_sample_path, octree_split_leaf_pts_);
}

void LocalVipss::SamplePtsWithClusterAveScores()
{

    // std::vector<double> sampled_pts;
    // size_t c_size = points_.size();
    // std::vector<SimOctree::Point> new_pts;
    // for(size_t i = 0; i < c_size; ++i)
    // {
    //     SimOctree::Point newP = {points_[i][0], points_[i][1], points_[i][2]};
    //     new_pts.push_back(newP);
    // }
    // SimOctree::SimpleOctree octree;
    // octree.InitOctTree(new_pts, 6);
    // auto leaf_pts = octree.GetLeafMapPts();
    // // std::cout << " octree sample leaf pt number : " << octree.leaf_pts_.size() << std::endl;
    // for(int i = 0; i < leaf_pts.size(); ++i)
    // {
    //     sampled_pts.push_back(leaf_pts[i][0]);
    //     sampled_pts.push_back(leaf_pts[i][1]);
    //     sampled_pts.push_back(leaf_pts[i][2]);
    // }
    // // std::string octree_sample_path = out_dir_  + filename_ +  "_octree_sample.xyz";
    // // std::cout << "save octree sample result to file : " << octree_sample_path << std::endl;
    // // writeXYZ(octree_sample_path, sampled_pts);

    // PicoTree newPicoTree;
    // newPicoTree.Init(sampled_pts);

    // cluster_scores_ave_.resize(c_size);
    // for(int i = 0; i < int(c_size); ++i)
    // {
    //     int cur_nozero_count = cluster_scores_mat_.col(i).nonZeros();
    //     double score_sum = cluster_scores_mat_.col(i).sum();
    //     if( cur_nozero_count > 0)
    //     {
    //         cluster_scores_ave_[i] = score_sum / cur_nozero_count;
    //     } else {
    //         cluster_scores_ave_[i] = 0;
    //     }
    // }
    // arma::uvec res = arma::sort_index(cluster_scores_ave_, "descend");
    
    // int sample_num = int(c_size * 0.15);
    // int count = 0;
    // for(int i = 0; i < c_size; ++i)
    // {
    //     if(newPicoTree.NearestPtDist(points_[res[i]][0], points_[res[i]][1], points_[res[i]][2]) > 1e-8)
    //     {
    //         sampled_pts.push_back(points_[res[i]][0]);
    //         sampled_pts.push_back(points_[res[i]][1]);
    //         sampled_pts.push_back(points_[res[i]][2]);
    //         count ++;
    //         if(count >= sample_num) break;
    //     }
    // }

    // std::string sample_path = out_dir_ + filename_ +  "_feature_sample.xyz";
    // std::cout << "save feature sample result to file : " << sample_path << std::endl;
    // writeXYZ(sample_path, sampled_pts);    
}



void LocalVipss::InitNormals()
{
    double total_time = 0;
    auto t0 = Clock::now();
    // BuidClusterCoresPtIds();
    auto t1 = Clock::now();
    double build_mat_time = std::chrono::nanoseconds(t1 - t0).count()/1e9;
    printf("finish init adj mat and core pt ids time : %f ! \n", build_mat_time);

    InitNormalWithVipss();
    auto t12 = Clock::now();
    double normal_estimate_time = std::chrono::nanoseconds(t12 - t1).count()/1e9;
    printf("finish init cluster normals time : %f ! \n", normal_estimate_time);

    total_time += build_mat_time + normal_estimate_time;
    G_VP_stats.init_cluster_normal_time_ += (build_mat_time + normal_estimate_time);
    double sum_score_time = 0;

// if(0)
{
    auto t2 = Clock::now();
    CalculateClusterNeiScores(true);
    auto t3 = Clock::now();
    double scores_time = std::chrono::nanoseconds(t3 - t2).count()/1e9;
    printf("finish init cluster neigh scores time : %f ! \n", scores_time);
    // if(feature_preserve_sample_)
    // {
    //     SamplePtsWithClusterAveScores();
    // }
    //
    // CalculateClusterScores();
    auto t34 = Clock::now();
    double cluster_scores_time = std::chrono::nanoseconds(t34 - t3).count()/1e9;
    printf("finish calculate cluster scores time : %f ! \n", cluster_scores_time);
    total_time += scores_time;
    total_time += cluster_scores_time;
    sum_score_time += scores_time;
    sum_score_time += cluster_scores_time;
    G_VP_stats.cal_cluster_neigbor_scores_time_ += (scores_time + cluster_scores_time);

    size_t iter_num = 1;
    bool use_distance_weight = true;
    auto ti00 = Clock::now();
    BuildClusterMST();
    
    FlipClusterNormalsByMST();
    s_vals_score_ = s_vals_;
    auto ti11 = Clock::now();
    double MST_time = std::chrono::nanoseconds(ti11 - ti00).count()/1e9;
    G_VP_stats.build_normal_MST_time_ += MST_time;
    out_pts_ = GetInitPts();
    out_normals_ = GetInitNormals();
    
    // BuildClusterMST(use_distance_weight);
    // FlipClusterNormalsByMST();
    s_vals_dist_ = s_vals_;
    // out_normals_dist_ = GetInitNormals();
    out_normals_dist_ = out_normals_;

    // FlipClusterNormalsFromConvexHull();
    // out_normals_ = GetInitNormals();
    // out_normals_dist_ = GetInitNormals();

    auto finat_t = Clock::now();
    double init_total_time = std::chrono::nanoseconds(finat_t - t0).count()/1e9;
    printf("Normal initializtion with local vipss total time used : %f \n", init_total_time);
}
    // G_VP_stats.init_normal_total_time_ += init_total_time;
}

void LocalVipss::ClearPartialMemory()
{
    // adjacent_mat_.clear();
    cluster_MST_mat_.resize(0,0);
    cluster_MST_mat_.data().squeeze();
     cluster_scores_mat_.resize(0,0);
    cluster_scores_mat_.data().squeeze();
    cluster_normal_x_.resize(0,0);
    cluster_normal_x_.data().squeeze();
    cluster_normal_y_.resize(0,0);
    cluster_normal_y_.data().squeeze();
    cluster_normal_z_.resize(0,0);
    cluster_normal_z_.data().squeeze();
    voro_gen_.pt_adjecent_mat_.clear();
    voro_gen_.points_.clear();
    VoronoiGen::cluster_accum_size_vec_.clear();
    // voro_gen_.tetMesh_
    // VoronoiGen::cluster_accum_size_vec_.clear();

    // cluster_adjacent_ids_.clear();
}