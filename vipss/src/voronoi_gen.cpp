#include "voronoi_gen.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include "readers.h"
#include <chrono>
#include <math.h>
#include <cmath>
// #include <libqhull/libqhull.h>
// #include <libqhullcpp/Qhull.h>
// #include <libqhullcpp/RboxPoints.h>
// #include <libqhull/qhull_a.h>
#include "sample.h"
#include "omp.h"

typedef std::chrono::high_resolution_clock Clock;

REAL cps = (REAL) CLOCKS_PER_SEC;
// double M_PI  = 2*acos(0.0);

const int MAX_ELEMENT_ = 8000;
const int MAX_THREAD_NUM_ = 32;
double* truncated_tets_omp_[MAX_THREAD_NUM_][MAX_ELEMENT_*2];
// double truncated_centers_omp_[MAX_THREAD_NUM_][3];
double intersect_pts_omp_[MAX_THREAD_NUM_][MAX_ELEMENT_];
int face_tet_count_omp_[MAX_THREAD_NUM_][MAX_ELEMENT_];

void VoronoiGen::loadData(const std::string& path )
{

    tetIO_.load_node((char*)path.c_str());
    // tetIO_.load_ply((char*)path.c_str());
    printf("load data contain %d points! \n", tetIO_.numberofpoints);
    // tetIO_.pointlist;
}

void VoronoiGen::loadData(const std::vector<double>& in_pts)
{
    tetIO_.load_node((double*)in_pts.data(), in_pts.size()/3);
}

void VoronoiGen::InsertSphereBoundryPts42()
{
    std::vector<double> sphere_pts = {0.000000,1.241958,0.767572, 0.000000,1.241958,-0.767572,0.000000,-1.241958,0.767572,
                                    0.000000,-1.241958,-0.767572, 1.241958,0.767572,0.000000,1.241958,-0.767572,0.000000,
                                    -1.241958,0.767572,0.000000, -1.241958,-0.767572,0.000000, 0.767572,0.000000,1.241958, 
                                    -0.767572,0.000000,1.241958, 0.767572,0.000000,-1.241958, -0.767572,0.000000,-1.241958,
                                    0.000000,1.463526,0.000000, 0.731763,1.184017,0.452254, 0.731763,1.184017,-0.452254,
                                    -0.731763,1.184017,-0.452254, -0.731763,1.184017,0.452254, 0.000000,-1.463526,0.000000,
                                    0.731763,-1.184017,-0.452254, 0.731763,-1.184017,0.452254, -0.731763,-1.184017,0.452254,
                                    -0.731763,-1.184017,-0.452254, 1.463526,0.000000,0.000000, 1.184017,-0.452254,-0.731763,
                                    1.184017,0.452254,-0.731763, 1.184017,0.452254,0.731763, 1.184017,-0.452254,0.731763,
                                        -1.463526,0.000000,0.000000, -1.184017,-0.452254,0.731763, -1.184017,0.452254,0.731763,
                                        -1.184017,0.452254,-0.731763, -1.184017,-0.452254,-0.731763, 0.000000,0.000000,1.463526,
                                        -0.452254,-0.731763,1.184017, 0.452254,-0.731763,1.184017, 0.452254,0.731763,1.184017,
                                        -0.452254,0.731763,1.184017, 0.000000,0.000000,-1.463526, -0.452254,0.731763,-1.184017,
                                        0.452254,0.731763,-1.184017, 0.452254,-0.731763,-1.184017, -0.452254,-0.731763,-1.184017};
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();

    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    double max_z = std::numeric_limits<double>::lowest();

    auto& in_pts = tetIO_.pointlist;
    for(size_t i =0; i < tetIO_.numberofpoints; ++i)
    {
        min_x = min_x < in_pts[3*i]     ? min_x : in_pts[3*i];
        min_y = min_y < in_pts[3*i + 1] ? min_y : in_pts[3*i + 1];
        min_z = min_z < in_pts[3*i + 2] ? min_z : in_pts[3*i + 2];

        max_x = max_x > in_pts[3*i]     ? max_x : in_pts[3*i];
        max_y = max_y > in_pts[3*i + 1] ? max_y : in_pts[3*i + 1];
        max_z = max_z > in_pts[3*i + 2] ? max_z : in_pts[3*i + 2];
    }

    bbox_min_ = {min_x, min_y, min_z};
    bbox_max_ = {max_x, max_y, max_z};

    double cx = (min_x + max_x) / 2.0;
    double cy = (min_y + max_y) / 2.0;
    double cz = (min_z + max_z) / 2.0;
    double scale = 1.5;

    double x_len = (max_x - min_x);
    double y_len = (max_y - min_y);
    double z_len = (max_z - min_z);
    double max_len = std::max(x_len, std::max(y_len, z_len));

    double max_scale = 5.0;
    double expand_scale_x = 0.2 * std::max(1.0, std::min(max_scale, max_len / x_len));
    double expand_scale_y = 0.2 * std::max(1.0, std::min(max_scale, max_len / y_len));
    double expand_scale_z = 0.2 * std::max(1.0, std::min(max_scale, max_len / z_len));

    double dx = (max_x - min_x) / 2.0 * std::max(scale, expand_scale_x);
    double dy = (max_y - min_y) / 2.0 * std::max(scale, expand_scale_y);
    double dz = (max_z - min_z) / 2.0 * std::max(scale, expand_scale_z);
    // double radius = std::max(dx, std::max(dy, dz)); 
    double radius = sqrt(dx *dx + dy * dy + dz * dz);
    
    int pt_num = sphere_pts.size() /3;
    for(size_t i = 0; i < pt_num; ++i)
    {
        tetgenmesh::point newpt;
        auto& temp_mesh = tetMesh_;
        temp_mesh.makepoint(&newpt, tetgenmesh::VOLVERTEX);
        double cur_dist = sqrt(sphere_pts[i * 3] * sphere_pts[i * 3] 
                                + sphere_pts[i * 3 + 1] * sphere_pts[i * 3 + 1] 
                                + sphere_pts[i * 3 + 2] * sphere_pts[i * 3 + 2]);
        newpt[0] = sphere_pts[i * 3]     / cur_dist * radius;
        newpt[1] = sphere_pts[i * 3 + 1] / cur_dist * radius;
        newpt[2] = sphere_pts[i * 3 + 2] / cur_dist * radius;
        tetgenmesh::insertvertexflags ivf;
        ivf.bowywat = 1; // Use Bowyer-Watson algorithm
        ivf.lawson = 0;
        tetgenmesh::triface searchtet;
        searchtet.tet = NULL;
        ivf.iloc = (int) tetgenmesh::UNKNOWN;
        auto t0 = Clock::now();
        tetgenmesh::face *splitsh = NULL;
        tetgenmesh::face *splitseg = NULL;
        // temp_mesh.setpointtype(newpt, tetgenmesh::UNUSEDVERTEX);
        if(temp_mesh.insertpoint(newpt, &searchtet, splitsh, splitseg,  &ivf))
        {
            // printf(" %ld insertion succeeded ! \n", i);
        }
        insert_boundary_pts_.push_back(newpt);
    }
}

void VoronoiGen::InsertSphereBoundryPts()
{
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();

    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    double max_z = std::numeric_limits<double>::lowest();

    auto& in_pts = tetIO_.pointlist;
    for(size_t i =0; i < tetIO_.numberofpoints; ++i)
    {
        min_x = min_x < in_pts[3*i]     ? min_x : in_pts[3*i];
        min_y = min_y < in_pts[3*i + 1] ? min_y : in_pts[3*i + 1];
        min_z = min_z < in_pts[3*i + 2] ? min_z : in_pts[3*i + 2];

        max_x = max_x > in_pts[3*i]     ? max_x : in_pts[3*i];
        max_y = max_y > in_pts[3*i + 1] ? max_y : in_pts[3*i + 1];
        max_z = max_z > in_pts[3*i + 2] ? max_z : in_pts[3*i + 2];
    }

    bbox_min_ = {min_x, min_y, min_z};
    bbox_max_ = {max_x, max_y, max_z};

    double cx = (min_x + max_x) / 2.0;
    double cy = (min_y + max_y) / 2.0;
    double cz = (min_z + max_z) / 2.0;
    
    double scale = 1.5;
    double dx = (max_x - min_x) / 2.0 * scale;
    double dy = (max_y - min_y) / 2.0 * scale;
    double dz = (max_z - min_z) / 2.0 * scale;
    // double radius = std::max(dx, std::max(dy, dz)); 
    double radius = sqrt(dx *dx + dy * dy + dz * dz);
    int pt_num = 168;
    auto sphere_pts = CreateSpherePoints(cx, cy, cz, radius, pt_num);
    pt_num = sphere_pts.size()/3;
    
    // printf("sphere pt size : %ld \n",sphere_pts.size()/3 );
    if(0)
    {
        std::string out_sphere_path = out_dir_ + "sphere_boundry.xyz";
        writeXYZ(out_sphere_path, sphere_pts);
        printf("successfully write sphere boundary pts to file : %s\n", out_sphere_path.c_str());
    }
    
    // printf(out_sphere_path);
    // std::vector<tetgenmesh::point> boundary_pts;

    for(size_t i = 0; i < pt_num; ++i)
    {
        tetgenmesh::point newpt;
        auto& temp_mesh = tetMesh_;
        temp_mesh.makepoint(&newpt, tetgenmesh::VOLVERTEX);
        newpt[0] = sphere_pts[i * 3];
        newpt[1] = sphere_pts[i * 3 + 1];
        newpt[2] = sphere_pts[i * 3 + 2];
        tetgenmesh::insertvertexflags ivf;
        ivf.bowywat = 1; // Use Bowyer-Watson algorithm
        ivf.lawson = 0;
        tetgenmesh::triface searchtet;
        searchtet.tet = NULL;
        ivf.iloc = (int) tetgenmesh::UNKNOWN;
        auto t0 = Clock::now();
        tetgenmesh::face *splitsh = NULL;
        tetgenmesh::face *splitseg = NULL;
        // temp_mesh.setpointtype(newpt, tetgenmesh::UNUSEDVERTEX);
        if(temp_mesh.insertpoint(newpt, &searchtet, splitsh, splitseg,  &ivf))
        {
            // printf(" %ld insertion succeeded ! \n", i);
        }
        insert_boundary_pts_.push_back(newpt);
    }

}

void VoronoiGen::InsertBoundryPts()
{
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();

    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    double max_z = std::numeric_limits<double>::lowest();

    auto& in_pts = tetIO_.pointlist;
    for(size_t i =0; i < tetIO_.numberofpoints; ++i)
    {
        min_x = min_x < in_pts[3*i]     ? min_x : in_pts[3*i];
        min_y = min_y < in_pts[3*i + 1] ? min_y : in_pts[3*i + 1];
        min_z = min_z < in_pts[3*i + 2] ? min_z : in_pts[3*i + 2];

        max_x = max_x > in_pts[3*i]     ? max_x : in_pts[3*i];
        max_y = max_y > in_pts[3*i + 1] ? max_y : in_pts[3*i + 1];
        max_z = max_z > in_pts[3*i + 2] ? max_z : in_pts[3*i + 2];
    }

    bbox_min_ = {min_x, min_y, min_z};
    bbox_max_ = {max_x, max_y, max_z};

    dummy_sign_pt_[0] = (max_x + min_x) / 2.0 + (max_x - min_x) / 2.0 * (1 + 0.2);
    dummy_sign_pt_[1] = (max_y + min_y) / 2.0 + (max_y - min_y) / 2.0 * (1 + 0.2);
    dummy_sign_pt_[2] = (max_z + min_z) / 2.0 + (max_z - min_z) / 2.0 * (1 + 0.2);

    // bbox final_scale = scale + 1 
    double scale = 0.5;
    double dx = (max_x - min_x) / 2.0 * scale;
    double dy = (max_y - min_y) / 2.0 * scale;
    double dz = (max_z - min_z) / 2.0 * scale;

    min_x -= dx; min_y -= dy; min_z -= dz;
    max_x += dx; max_y += dy; max_z += dz;

    //     5-------6
    //    /|      /|
    //   4-|-----7 |
    //   | 1-----|-2
    //   |/      |/
    //   0-------3 
    const int pt_num = 14;
    double box_pts[pt_num][3];
    box_pts[0][0] = min_x; box_pts[0][1] = min_y; box_pts[0][2] = min_z;
    box_pts[1][0] = min_x; box_pts[1][1] = max_y; box_pts[1][2] = min_z;
    box_pts[2][0] = max_x; box_pts[2][1] = max_y; box_pts[2][2] = min_z;
    box_pts[3][0] = max_x; box_pts[3][1] = min_y; box_pts[3][2] = min_z;

    box_pts[4][0] = min_x; box_pts[4][1] = min_y; box_pts[4][2] = max_z;
    box_pts[5][0] = min_x; box_pts[5][1] = max_y; box_pts[5][2] = max_z;
    box_pts[6][0] = max_x; box_pts[6][1] = max_y; box_pts[6][2] = max_z;
    box_pts[7][0] = max_x; box_pts[7][1] = min_y; box_pts[7][2] = max_z;

    // face centers 
    // f0 : 0 1 2 3
    // f1 : 0 3 7 4
    // f2 : 3 2 6 7
    // f3 : 2 1 5 6
    // f4 : 0 1 5 4
    // f5 : 4 5 6 7
if(pt_num > 8)
{
    box_pts[8][0] = (box_pts[0][0] + box_pts[1][0] + box_pts[2][0] + box_pts[3][0])/4.0;
    box_pts[8][1] = (box_pts[0][1] + box_pts[1][1] + box_pts[2][1] + box_pts[3][1])/4.0;
    box_pts[8][2] = (box_pts[0][2] + box_pts[1][2] + box_pts[2][2] + box_pts[3][2])/4.0;

    box_pts[9][0] = (box_pts[0][0] + box_pts[3][0] + box_pts[7][0] + box_pts[4][0])/4.0;
    box_pts[9][1] = (box_pts[0][1] + box_pts[3][1] + box_pts[7][1] + box_pts[4][1])/4.0;
    box_pts[9][2] = (box_pts[0][2] + box_pts[3][2] + box_pts[7][2] + box_pts[4][2])/4.0;

    box_pts[10][0] = (box_pts[3][0] + box_pts[2][0] + box_pts[6][0] + box_pts[7][0])/4.0;
    box_pts[10][1] = (box_pts[3][1] + box_pts[2][1] + box_pts[6][1] + box_pts[7][1])/4.0;
    box_pts[10][2] = (box_pts[3][2] + box_pts[2][2] + box_pts[6][2] + box_pts[7][2])/4.0;

    box_pts[11][0] = (box_pts[2][0] + box_pts[1][0] + box_pts[5][0] + box_pts[6][0])/4.0;
    box_pts[11][1] = (box_pts[2][1] + box_pts[1][1] + box_pts[5][1] + box_pts[6][1])/4.0;
    box_pts[11][2] = (box_pts[2][2] + box_pts[1][2] + box_pts[5][2] + box_pts[6][2])/4.0;
    
    box_pts[12][0] = (box_pts[0][0] + box_pts[1][0] + box_pts[5][0] + box_pts[4][0])/4.0;
    box_pts[12][1] = (box_pts[0][1] + box_pts[1][1] + box_pts[5][1] + box_pts[4][1])/4.0;
    box_pts[12][2] = (box_pts[0][2] + box_pts[1][2] + box_pts[5][2] + box_pts[4][2])/4.0;

    box_pts[13][0] = (box_pts[4][0] + box_pts[5][0] + box_pts[7][0] + box_pts[6][0])/4.0;
    box_pts[13][1] = (box_pts[4][1] + box_pts[5][1] + box_pts[7][1] + box_pts[6][1])/4.0;
    box_pts[13][2] = (box_pts[4][2] + box_pts[5][2] + box_pts[7][2] + box_pts[6][2])/4.0;
}

    for(size_t i = 0; i < pt_num; ++i)
    {
        tetgenmesh::point newpt;
        auto& temp_mesh = tetMesh_;
        temp_mesh.makepoint(&newpt, tetgenmesh::VOLVERTEX);
        newpt[0] = box_pts[i][0];
        newpt[1] = box_pts[i][1];
        newpt[2] = box_pts[i][2];
        tetgenmesh::insertvertexflags ivf;
        ivf.bowywat = 1; // Use Bowyer-Watson algorithm
        ivf.lawson = 0;
        tetgenmesh::triface searchtet;
        searchtet.tet = NULL;
        ivf.iloc = (int) tetgenmesh::UNKNOWN;
        auto t0 = Clock::now();
        tetgenmesh::face *splitsh = NULL;
        tetgenmesh::face *splitseg = NULL;
        // temp_mesh.setpointtype(newpt, tetgenmesh::UNUSEDVERTEX);
        if(temp_mesh.insertpoint(newpt, &searchtet, splitsh, splitseg,  &ivf))
        {
            
        }
        insert_boundary_pts_.push_back(newpt);
    }
}

void VoronoiGen::InsertPts(const std::vector<std::array<double,3>>& insert_pts)
{
    size_t pt_num = insert_pts.size();
    for(size_t i = 0; i < pt_num; ++i)
    {
        tetgenmesh::point newpt;
        auto& temp_mesh = tetMesh_;
        temp_mesh.makepoint(&newpt, tetgenmesh::VOLVERTEX);
        newpt[0] = insert_pts[i][0];
        newpt[1] = insert_pts[i][1];
        newpt[2] = insert_pts[i][2];
        tetgenmesh::insertvertexflags ivf;
        ivf.bowywat = 1; // Use Bowyer-Watson algorithm
        ivf.lawson = 0;
        tetgenmesh::triface searchtet;
        searchtet.tet = NULL;
        ivf.iloc = (int) tetgenmesh::UNKNOWN;
        auto t0 = Clock::now();
        tetgenmesh::face *splitsh = NULL;
        tetgenmesh::face *splitseg = NULL;
        // temp_mesh.setpointtype(newpt, tetgenmesh::UNUSEDVERTEX);
        if(temp_mesh.insertpoint(newpt, &searchtet, splitsh, splitseg,  &ivf))
        {
            
        }
        insert_boundary_pts_.push_back(newpt);
    }
}

void VoronoiGen::SetInsertBoundaryPtsToUnused()
{
    for(auto pt : insert_boundary_pts_)
    {
        tetMesh_.setpointtype(pt, tetgenmesh::UNUSEDVERTEX);
    }
    // insert_boundary_pts_.clear();
}

void VoronoiGen::SetInsertBoundaryPtsToUsed()
{
    for(auto pt : insert_boundary_pts_)
    {
        tetMesh_.setpointtype(pt, tetgenmesh::VOLVERTEX);
    }
    // insert_boundary_pts_.clear();
}

void VoronoiGen::Tetrahedralize()
{
    tetgenmesh& m = tetMesh_;
    tetgenbehavior *b = &tet_beha_;
    tetgenio *in = &tetIO_;

    clock_t tv[13], ts[6]; // Timing informations (defined in time.h)
    // printf("start to Tetrahedralize \n");
    tv[0] = clock();
    b->voroout = 0;
    // b->refine = 1;
    // b->plc = 1;
    m.b = b;
    m.in = in;
    // m.addin = addin;
    // addin != NULL
    m.initializepools();
    m.transfernodes();
    tv[1] = clock();
    m.incrementaldelaunay(ts[0]);

    m.outhullPts(m.in, convex_hull_pts_);

    // InsertSphereBoundryPts42();

    BuildPtIdMap();
    InsertSphereBoundryPts42();
    // printf("finsh BuildPtIdMap \n");
    // InsertBoundryPts();
    // GenerateVoroData();
    // m.facetverticeslist
    // m.meshsurface();
    // m.recoverboundary(ts[0]);
    // m.reconstructmesh();
    // m.outvoronoi(&voronoi_data_);
    // voronoi_data_.vpointlist
    // m.generate_voronoi_cell(&voronoi_data_);
    // printf("voronoi pt num : %d \n", voronoi_data_.numberofvpoints);
    // printf("voronoi edge num : %d \n", voronoi_data_.numberofvedges);
    // printf("voronoi facet num : %d \n", voronoi_data_.numberofvfacets);
    // printf("voronoi cell num : %d \n", voronoi_data_.numberofvcells);

    // printf("finsh generate_voronoi_cell \n");

    
    // OutputVoronisMesh();
    
    // printf("finsh to Tetrahedralize \n");
}

void VoronoiGen::GenerateVoroData()
{
    // InsertBoundryPts();
// 
    auto t0 = Clock::now();
    // InsertSphereBoundryPts();
    // InsertSphereBoundryPts42();
    // InsertBoundryPts();
    

    // printf("finsh InsertBoundryPts \n");
    tetMesh_.generate_voronoi_cell(&voronoi_data_);
    
    if(0)
    {
        OutputVoronisMesh();
        std::string out_mesh_path = out_dir_ + "tet_mesh.obj";
        OutputTetMesh(out_mesh_path);
    }
    // printf("finsh to Tetrahedralize \n");
}

void VoronoiGen::GetVoroCellEdgeList(tetgenmesh::point nei_pt, std::vector<tetgenio::voroedge*>& edge_list)
{
    // printf("pt size 0000 : %ld \n", points_.size());
    auto vc_id   = point_id_map_[nei_pt];
    const auto v_cell  = voronoi_data_.vcelllist[vc_id];
    // printf("pt size 000000 : %ld \n", points_.size());
    auto vf_list = voronoi_data_.vfacetlist;
    auto ve_list = voronoi_data_.vedgelist;
    // printf("pt size 0001 : %ld \n", points_.size());
    if(v_cell == NULL) return;
    int f_num = v_cell[0];
    // printf("f_num : %ld \n", f_num);
    std::set<tetgenio::voroedge*> cell_edges;
    
    for(size_t i = 0; i < f_num; ++i)
    {
        size_t f_id = v_cell[i + 1];
        // printf("v_cell f id 0002 : %ld \n", f_id);
        auto facet = vf_list[f_id];
        // printf("pt size 0002 : %ld \n", points_.size());
        size_t e_num = facet.elist[0];
        for(size_t j = 0; j < e_num; ++j)
        {
            auto& ve = ve_list[facet.elist[1 + j]];
            cell_edges.insert(&ve);
        }
    }
    edge_list = std::vector<tetgenio::voroedge*>(cell_edges.begin(), cell_edges.end());
}

void VoronoiGen::GetVoroCellPtList(std::vector<tetgenio::voroedge*>& edge_list, std::vector<double>& pt_list)
{
    auto& vplist = voronoi_data_.vpointlist;
    std::set<int> pidset;
    for(auto edge : edge_list)
    {
        // auto pt1 = vplist[edge->v1];
        pidset.insert(edge->v1);
        pidset.insert(edge->v2);
    }
    for(auto pid : pidset)
    {
        pt_list.push_back(vplist[3 * pid]);
        pt_list.push_back(vplist[3 * pid + 1]);
        pt_list.push_back(vplist[3 * pid + 2]);
    }
}

void VoronoiGen::GetVoroCellPtAndEdgeIdList(tetgenmesh::point nei_pt, std::set<int>& vcell_pt_list, std::set<int>& vcell_edge_list)
{
    const auto vc_id   = point_id_map_[nei_pt];
    const auto v_cell  = voronoi_data_.vcelllist[vc_id];
    const auto vf_list = voronoi_data_.vfacetlist;
    const auto ve_list = voronoi_data_.vedgelist;
    // printf("pt size 0001 : %ld \n", points_.size());
    if(v_cell == NULL) return;
    int f_num = v_cell[0];
    std::set<int> p_ids;
    for(size_t i = 0; i < f_num; ++i)
    {
        size_t f_id = v_cell[i + 1];
        auto facet = vf_list[f_id];
        size_t e_num = facet.elist[0];
        arma::vec3 f_center;
        for(size_t j = 0; j < e_num; ++j)
        {
            int e_id = facet.elist[1 + j];
            vcell_edge_list.insert(e_id);
            const auto& ve = ve_list[e_id];
            vcell_pt_list.insert(ve.v1);
            vcell_pt_list.insert(ve.v2);
        }
    }
}

void VoronoiGen::CalVoroEdgeIntersection(const std::set<int>& cell_eids, std::vector<double>& intersect_pts)
{

    auto ve_list = voronoi_data_.vedgelist;
    auto vp_list = voronoi_data_.vpointlist;
    // printf("pt size 0001 : %ld \n", points_.size());
    // printf("start to cal facet center 000\n");
    // printf("vpt_sign_vals_ size %ld \n", vpt_sign_vals_.size());

    for(auto e_id : cell_eids)
    {
        auto& ve = ve_list[e_id];
        double t1 = vpt_sign_vals_[ve.v1];
        double t2 = vpt_sign_vals_[ve.v2];
        if(t1 * t2 <= 0)
        {
            double sum = abs(t1) + abs(t2);
            if(sum > 0) 
            {
                t1 = abs(t1) / sum;
                t2 = abs(t2) / sum;
            }
            double p1_x = vp_list[3 * ve.v1]; 
            double p1_y = vp_list[3 * ve.v1 + 1]; 
            double p1_z = vp_list[3 * ve.v1 + 2];
            double p2_x = vp_list[3 * ve.v2]; 
            double p2_y = vp_list[3 * ve.v2 + 1]; 
            double p2_z = vp_list[3 * ve.v2 + 2];
            double ix = t2 * p1_x + t1 * p2_x;
            double iy = t2 * p1_y + t1 * p2_y;
            double iz = t2 * p1_z + t1 * p2_z;
            intersect_pts.push_back(ix);
            intersect_pts.push_back(iy);
            intersect_pts.push_back(iz);
        }
    }
}

void VoronoiGen::CalVoroCellPtSign(const VoroPlane &plane, 
                                   const std::set<int>& vcell_pt_list, 
                                   std::vector<int>& positive_pids)
{
    // printf("vpt_sign_vals_ size : %ld \n", vpt_sign_vals_.size());
    // vcell_pt_sign_values_.clear();
    auto vp_list = voronoi_data_.vpointlist;
    for(auto p_id : vcell_pt_list)
    {
        double dx = vp_list[3 * p_id]     - plane.px;
        double dy = vp_list[3 * p_id + 1] - plane.py;
        double dz = vp_list[3 * p_id + 2] - plane.pz;
        double proj = dx * plane.nx + dy * plane.ny + dz * plane.nz;
        vpt_sign_vals_[p_id] = proj; 
        if(proj > 0)
        {
            positive_pids.push_back(p_id);
        }   
    } 
}


bool VoronoiGen::CalPlaneEdgeIntersection(tetgenio::voroedge* edge, VoroPlane &plane)
{
    auto vp_list = voronoi_data_.vpointlist;
    auto ax = vp_list[3 * edge->v1];
    auto ay = vp_list[3 * edge->v1 + 1];
    auto az = vp_list[3 * edge->v1 + 2];

    auto bx = vp_list[3 * edge->v2];
    auto by = vp_list[3 * edge->v2 + 1];
    auto bz = vp_list[3 * edge->v2 + 2];

    double ab_x = bx - ax;
    double ab_y = by - ay;
    double ab_z = bz - az;

    double ap_x =  ax - plane.px;
    double ap_y =  ay - plane.py;
    double ap_z =  az - plane.pz;

    double ap_dot = ap_x * plane.nx + ap_y * plane.ny + ap_z * plane.nz;
    double abp = ab_x * plane.nx  + ab_y * plane.ny + ab_z * plane.nz;
    if(abs(abp) < 1e-16) return false;
    double t = - ap_dot / abp;
    if(t <0 || t > 1.0) return false;

    plane.intPx = ax + ab_x * t;
    plane.intPy = ay + ab_y * t;
    plane.intPz = az + ab_z * t;
    return true;
}

std::vector<double> VoronoiGen::GetPolygonSplitPts(tetgenmesh::point cur_pt, tetgenmesh::point nei_pt)
{
    std::vector<tetgenio::voroedge*> vc_edges;
    GetVoroCellEdgeList(nei_pt, vc_edges);

    std::vector<double> vc_pts;
    GetVoroCellPtList(vc_edges, vc_pts);

    double mid_x = (cur_pt[0] + nei_pt[0])/2;
    double mid_y = (cur_pt[1] + nei_pt[1])/2;
    double mid_z = (cur_pt[2] + nei_pt[2])/2;

    double p_nx = cur_pt[0] - nei_pt[0];
    double p_ny = cur_pt[1] - nei_pt[1];
    double p_nz = cur_pt[2] - nei_pt[2];
    double n_len = sqrt(p_nx * p_nx + p_ny * p_ny + p_nz * p_nz);
    // n_len = 1e-14 > n_len ? 1e-14 : n_len; 
    p_nx /= n_len;
    p_ny /= n_len;
    p_nz /= n_len;

    std::vector<double> split_pts;
    for(size_t i = 0; i < vc_pts.size()/3; ++i)
    {
        double vp_nx = vc_pts[3 * i]     - mid_x;
        double vp_ny = vc_pts[3 * i + 1] - mid_y;
        double vp_nz = vc_pts[3 * i + 2] - mid_z;
        double sign = vp_nx * p_nx + vp_ny * p_ny + vp_nz * p_nz;
        if(sign > 0)
        {
            split_pts.push_back(vc_pts[3 * i]);
            split_pts.push_back(vc_pts[3 * i + 1]);
            split_pts.push_back(vc_pts[3 * i + 2]);
        }
    }
    VoroPlane v_plane(mid_x, mid_y, mid_z, p_nx, p_ny, p_nz);
    
    for(auto edge : vc_edges)
    {
        if(CalPlaneEdgeIntersection(edge, v_plane))
        {
            // double dx = v_plane.intPx - mid_x;
            // double dy = v_plane.intPy - mid_y;
            // double dz = v_plane.intPz - mid_z;
            // double res = v_plane.nx * dx + v_plane.ny * dy + v_plane.nz * dz;
            // printf(" res -------------- %f \n", res);
            split_pts.push_back(v_plane.intPx);
            split_pts.push_back(v_plane.intPy);
            split_pts.push_back(v_plane.intPz);
            // myfile_split << "v " << v_plane.intPx << " " << v_plane.intPy << " " << v_plane.intPz << std::endl;
            // v_id++;
        }
    }
    return split_pts;
}

inline double CalPtLen(double ax, double ay, double az, double bx, double by, double bz)
{
    double dx = ax - bx;
    double dy = ay - by;
    double dz = az - bz;
    double len = sqrt(dx * dx + dy * dy + dz * dz);
    return len;
}

double CalTetrahedronVolume(double *pa, double* pb, double* pc, double* pd)
{
    double len_ab = CalPtLen(pa[0], pa[1], pa[2], pb[0], pb[1], pb[2]);
    double len_bc = CalPtLen(pb[0], pb[1], pb[2], pc[0], pc[1], pc[2]);
    double len_ac = CalPtLen(pa[0], pa[1], pa[2], pc[0], pc[1], pc[2]);

    double s = (len_ab + len_ac + len_bc) / 2;
    double area = sqrt(s * (s - len_ab) *(s - len_ac) *(s- len_bc));

    double ab_x = pa[0] - pb[0];
    double ab_y = pa[1] - pb[1];
    double ab_z = pa[2] - pb[2];

    double bc_x = pb[0] - pc[0];
    double bc_y = pb[1] - pc[1];
    double bc_z = pb[2] - pc[2];

    double nx = ab_y * bc_z - ab_z * bc_y;
    double ny = ab_z * bc_x - ab_x * bc_z;
    double nz = ab_x * bc_y - ab_y * bc_x;

    double n_len = sqrt(nx * nx + ny * ny + nz * nz);

    if(n_len < 1e-16) return 0;
    nx /= n_len;
    ny /= n_len;
    nz /= n_len;

    double ad_x = pd[0] - pa[0];
    double ad_y = pd[1] - pa[1];
    double ad_z = pd[2] - pa[2];
    double proj = nx * ad_x + ny * ad_y + nz * ad_z;
    double volume = abs(proj) * area / 3.0;
    return volume;
}


inline double CalTetrahedronVolumeDet(const double* pa, const double* pb, const double* pc, const double* pd)
{
    double a00 = pa[0] - pd[0];
    double a10 = pb[0] - pd[0];
    double a20 = pc[0] - pd[0];

    double a01 = pa[1] - pd[1];
    double a11 = pb[1] - pd[1];
    double a21 = pc[1] - pd[1];

    double a02 = pa[2] - pd[2];
    double a12 = pb[2] - pd[2];
    double a22 = pc[2] - pd[2];

    double det1 = a00 * a11 * a22 + a01 * a12 * a20 + a02 * a10 * a21;
    double det2 = a02 * a11 * a20 + a01 * a10 * a22 + a00 * a12 * a21;
    double det = abs(det2 - det1) / 6.0;
    return det;
}

inline double CalSignedTetrahedronVolumeDet(const double* pa, const double* pb, const double* pc, const double* pd)
{
    double a00 = pa[0] - pd[0];
    double a10 = pb[0] - pd[0];
    double a20 = pc[0] - pd[0];

    double a01 = pa[1] - pd[1];
    double a11 = pb[1] - pd[1];
    double a21 = pc[1] - pd[1];

    double a02 = pa[2] - pd[2];
    double a12 = pb[2] - pd[2];
    double a22 = pc[2] - pd[2];

    double det1 = a00 * a11 * a22 + a01 * a12 * a20 + a02 * a10 * a21;
    double det2 = a02 * a11 * a20 + a01 * a10 * a22 + a00 * a12 * a21;
    double det = (det2 - det1) / 6.0;
    return det;
}


void VoroPlane::SavePlane(const std::string& outpath)
{
    arma::vec3 ax_1 = {1, 1, 0}; 
    ax_1[2] = -(nx * ax_1[0] + ny * ax_1[1]) / nz;
    double len = sqrt(ax_1[0] * ax_1[0] + ax_1[1] * ax_1[1] + ax_1[2] * ax_1[2]);
    ax_1[0] /= len;
    ax_1[1] /= len;
    ax_1[2] /= len;
    arma::vec3 pv{nx, ny, nz};
    arma::vec3 ax_2 = arma::cross(ax_1, pv);
    std::ofstream out_file;
    out_file.open(outpath);
    arma::vec3 ori{px, py, pz};
    arma::vec3 pa = ori + ax_1 * 2;
    arma::vec3 pb = ori + ax_2 * 2;
    arma::vec3 pc = ori - ax_1 * 2;
    arma::vec3 pd = ori - ax_2 * 2;

    out_file << "v " << pa[0] << " " << pa[1] << " " << pa[2] << std::endl;
    out_file << "v " << pb[0] << " " << pb[1] << " " << pb[2] << std::endl;
    out_file << "v " << pc[0] << " " << pc[1] << " " << pc[2] << std::endl;
    out_file << "v " << pd[0] << " " << pd[1] << " " << pd[2] << std::endl;

    out_file << "f 1 2 3" << std::endl;
    out_file << "f 1 4 3" << std::endl;
    out_file.close();
}

double VoronoiGen::CalTruncatedCellVolumePassOMP(tetgenmesh::point in_pt, tetgenmesh::point nei_pt, int thread_id)
{
    double plane_mid_x = (in_pt[0] + nei_pt[0])/2.0;
    double plane_mid_y = (in_pt[1] + nei_pt[1])/2.0;
    double plane_mid_z = (in_pt[2] + nei_pt[2])/2.0;
    double p_nx  = (in_pt[0] - nei_pt[0]);
    double p_ny  = (in_pt[1] - nei_pt[1]);
    double p_nz  = (in_pt[2] - nei_pt[2]);
    double pn_len = sqrt(p_nx * p_nx + p_ny * p_ny + p_nz * p_nz);
    if(pn_len > 0) 
    {
        p_nx /= pn_len;
        p_ny /= pn_len;
        p_nz /= pn_len;
    }
    // if(point_id_map_.find(nei_pt) == point_id_map_.end())
    // return 0;
    const auto vc_id   = point_id_map_[nei_pt];
    // if(vc_id >= voronoi_data_.numberofvcells) return 0;
    const auto v_cell  = voronoi_data_.vcelllist[vc_id];
    const auto vf_list = voronoi_data_.vfacetlist;
    const auto ve_list = voronoi_data_.vedgelist;
    const auto vp_list = voronoi_data_.vpointlist;
    // printf("pt size 0001 : %ld \n", points_.size());
    if(v_cell == NULL) return 0;
    const int f_num = v_cell[0];

    int cur_opm_id = thread_id;
    double* intersect_pts = intersect_pts_omp_[cur_opm_id];
    // double* trunc_center = truncated_centers_omp_[cur_opm_id];
    double** trunc_tets = truncated_tets_omp_[cur_opm_id];
    int* face_tet_count = face_tet_count_omp_[cur_opm_id];

    int intersect_total_count = 0;
    int tet_total_count = 0;
    for(size_t i = 0; i < f_num; ++i)
    {
        size_t f_id = v_cell[i + 1];
        const auto& facet = vf_list[f_id];
        size_t e_num = facet.elist[0];
        int tet_count = 0;
        int intersect_count = 0; 
        for(size_t j = 0; j < e_num; ++j)
        {
            // printf("edge id : %ld \n", j);
            size_t e_id = facet.elist[1 + j];
            const auto& ve = ve_list[e_id];
            if(ve.v1 != -1 && ve.v2 != -1)
            {
                double dx = vp_list[3 * ve.v1]     - plane_mid_x;
                double dy = vp_list[3 * ve.v1 + 1] - plane_mid_y;
                double dz = vp_list[3 * ve.v1 + 2] - plane_mid_z;
                double proj1 = dx * p_nx + dy * p_ny + dz * p_nz;

                dx = vp_list[3 * ve.v2]     - plane_mid_x;
                dy = vp_list[3 * ve.v2 + 1] - plane_mid_y;
                dz = vp_list[3 * ve.v2 + 2] - plane_mid_z;
                double proj2 = dx * p_nx + dy * p_ny + dz * p_nz;
                if(proj1 * proj2 < 0 )
                {
                    double r1 = abs(proj1) /(abs(proj1) + abs(proj2));
                    double r2 = abs(proj2) /(abs(proj1) + abs(proj2));
                    double inter_x = r2 * vp_list[3 * ve.v1]     + r1 * vp_list[3 * ve.v2];
                    double inter_y = r2 * vp_list[3 * ve.v1 + 1] + r1 * vp_list[3 * ve.v2 + 1];
                    double inter_z = r2 * vp_list[3 * ve.v1 + 2] + r1 * vp_list[3 * ve.v2 + 2];
                    intersect_pts[3 * (intersect_total_count + intersect_count)]     = inter_x;
                    intersect_pts[3 * (intersect_total_count + intersect_count) + 1] = inter_y;
                    intersect_pts[3 * (intersect_total_count + intersect_count) + 2] = inter_z;
                    int posi_id = proj1 > 0 ? ve.v1 : ve.v2;
                    trunc_tets[(tet_total_count + tet_count)*2] = &vp_list[3 * posi_id];
                    trunc_tets[(tet_count + tet_total_count)*2 + 1] = &intersect_pts[3 * (intersect_total_count + intersect_count)];
                    intersect_count++;
                    tet_count ++;
                } 
                else if(proj1 >= 0 && proj2 >= 0)
                {
                    trunc_tets[(tet_total_count + tet_count)*2] = &vp_list[3 * ve.v2];
                    trunc_tets[(tet_total_count + tet_count)*2 + 1] = &vp_list[3 * ve.v1];
                    tet_count ++;
                } 
            } 
        }
        // printf("intersect_count num : %d \n", intersect_count);
        if(intersect_count > 0)
        {
            trunc_tets[(tet_total_count + tet_count)*2] = &intersect_pts[3 * (intersect_total_count)];
            trunc_tets[(tet_total_count + tet_count)*2 + 1] = &intersect_pts[3 * intersect_total_count + 3];
            
            intersect_total_count += 2;
            tet_count ++;
        }
        // printf("tet_count num : %d \n", tet_count);
        face_tet_count[i] = tet_count;
        tet_total_count += tet_count;
    }
    double cell_volume = 0;
    if(intersect_total_count > 0)
    {
        const double* trunc_pt = intersect_pts; 
        int cur_tet_id = 0;
        for(size_t i = 0; i < f_num; ++i)
        {
            const int f_tet_num = face_tet_count[i];
            const auto f_pt = trunc_tets[cur_tet_id*2];
            // printf("cur face f_tet_num : %d \n", f_tet_num);
            for(int ti = 0; ti < f_tet_num; ++ti )
            {
                double volume = CalTetrahedronVolumeDet(trunc_tets[cur_tet_id*2], 
                                                        trunc_tets[cur_tet_id*2 + 1], 
                                                        f_pt,
                                                        trunc_pt);
                cell_volume += volume;
                cur_tet_id ++;
            }
        }
        return cell_volume;
    }
    return 0;
}

double VoronoiGen::CalTruncatedCellVolumeGradientOMP(tetgenmesh::point in_pt, tetgenmesh::point nei_pt, 
                                                        arma::vec3& gradient, int thread_id)
{
    double plane_mid_x = (in_pt[0] + nei_pt[0])/2.0;
    double plane_mid_y = (in_pt[1] + nei_pt[1])/2.0;
    double plane_mid_z = (in_pt[2] + nei_pt[2])/2.0;
    double p_nx  = (in_pt[0] - nei_pt[0]);
    double p_ny  = (in_pt[1] - nei_pt[1]);
    double p_nz  = (in_pt[2] - nei_pt[2]);
    double pn_len = sqrt(p_nx * p_nx + p_ny * p_ny + p_nz * p_nz);
    if(pn_len > 0) 
    {
        p_nx /= pn_len;
        p_ny /= pn_len;
        p_nz /= pn_len;
    }
    // if(point_id_map_.find(nei_pt) == point_id_map_.end())
    // return 0;
    const auto vc_id   = point_id_map_[nei_pt];
    // if(vc_id >= voronoi_data_.numberofvcells) return 0;
    const auto v_cell  = voronoi_data_.vcelllist[vc_id];
    const auto vf_list = voronoi_data_.vfacetlist;
    const auto ve_list = voronoi_data_.vedgelist;
    const auto vp_list = voronoi_data_.vpointlist;
    // printf("pt size 0001 : %ld \n", points_.size());
    if(v_cell == NULL) return 0;
    const int f_num = v_cell[0];

    int cur_opm_id = thread_id;
    double* intersect_pts = intersect_pts_omp_[cur_opm_id];
    // double* trunc_center = truncated_centers_omp_[cur_opm_id];
    double** trunc_tets = truncated_tets_omp_[cur_opm_id];
    int* face_tet_count = face_tet_count_omp_[cur_opm_id];

    int intersect_total_count = 0;
    int tet_total_count = 0;
    for(size_t i = 0; i < f_num; ++i)
    {
        size_t f_id = v_cell[i + 1];
        const auto& facet = vf_list[f_id];
        size_t e_num = facet.elist[0];
        int tet_count = 0;
        int intersect_count = 0; 
        for(size_t j = 0; j < e_num; ++j)
        {
            // printf("edge id : %ld \n", j);
            size_t e_id = facet.elist[1 + j];
            const auto& ve = ve_list[e_id];
            if(ve.v1 != -1 && ve.v2 != -1)
            {
                double dx = vp_list[3 * ve.v1]     - plane_mid_x;
                double dy = vp_list[3 * ve.v1 + 1] - plane_mid_y;
                double dz = vp_list[3 * ve.v1 + 2] - plane_mid_z;
                double proj1 = dx * p_nx + dy * p_ny + dz * p_nz;

                dx = vp_list[3 * ve.v2]     - plane_mid_x;
                dy = vp_list[3 * ve.v2 + 1] - plane_mid_y;
                dz = vp_list[3 * ve.v2 + 2] - plane_mid_z;
                double proj2 = dx * p_nx + dy * p_ny + dz * p_nz;
                if(proj1 * proj2 < 0 )
                {
                    double r1 = abs(proj1) /(abs(proj1) + abs(proj2));
                    double r2 = abs(proj2) /(abs(proj1) + abs(proj2));
                    double inter_x = r2 * vp_list[3 * ve.v1]     + r1 * vp_list[3 * ve.v2];
                    double inter_y = r2 * vp_list[3 * ve.v1 + 1] + r1 * vp_list[3 * ve.v2 + 1];
                    double inter_z = r2 * vp_list[3 * ve.v1 + 2] + r1 * vp_list[3 * ve.v2 + 2];
                    intersect_pts[3 * (intersect_total_count + intersect_count)]     = inter_x;
                    intersect_pts[3 * (intersect_total_count + intersect_count) + 1] = inter_y;
                    intersect_pts[3 * (intersect_total_count + intersect_count) + 2] = inter_z;
                    int posi_id = proj1 > 0 ? ve.v1 : ve.v2;
                    trunc_tets[(tet_total_count + tet_count)*2] = &vp_list[3 * posi_id];
                    trunc_tets[(tet_count + tet_total_count)*2 + 1] = &intersect_pts[3 * (intersect_total_count + intersect_count)];
                    intersect_count++;
                    tet_count ++;
                } 
                else if(proj1 >= 0 && proj2 >= 0)
                {
                    trunc_tets[(tet_total_count + tet_count)*2] = &vp_list[3 * ve.v2];
                    trunc_tets[(tet_total_count + tet_count)*2 + 1] = &vp_list[3 * ve.v1];
                    tet_count ++;
                } 
            } 
        }
        // printf("intersect_count num : %d \n", intersect_count);
        if(intersect_count > 0)
        {
            trunc_tets[(tet_total_count + tet_count)*2] = &intersect_pts[3 * (intersect_total_count)];
            trunc_tets[(tet_total_count + tet_count)*2 + 1] = &intersect_pts[3 * intersect_total_count + 3];
            intersect_total_count += 2;
            tet_count ++;
        }
        // printf("tet_count num : %d \n", tet_count);
        face_tet_count[i] = tet_count;
        tet_total_count += tet_count;
    }
    double cell_volume = 0;
    if(intersect_total_count > 0)
    {
        std::array<double,3> truncated_face_center;
        for(int ii = 0; ii < intersect_total_count; ++ii)
        {
            truncated_face_center[0] += intersect_pts[3 * ii];
            truncated_face_center[1] += intersect_pts[3 * ii + 1];
            truncated_face_center[2] += intersect_pts[3 * ii + 2];
        }
        truncated_face_center[0] /= double(intersect_total_count);
        truncated_face_center[1] /= double(intersect_total_count);
        truncated_face_center[2] /= double(intersect_total_count);

        // std::vector<std::array<double, 3>> triangle_centers(intersect_total_count/2);
        std::vector<double> triangle_areas(intersect_total_count/2);
        arma::vec3 f_centroid; 
        double f_area = 0.0;
        for(int ii = 0; ii < intersect_total_count/2; ++ii)
        {
            arma::vec3 tri_center;
            tri_center[0] = (intersect_pts[3 * (2 * ii)]     + intersect_pts[3 * (2 * ii + 1)]     + truncated_face_center[0])/3;
            tri_center[1] = (intersect_pts[3 * (2 * ii) + 1] + intersect_pts[3 * (2 * ii + 1) + 1] + truncated_face_center[1])/3;
            tri_center[2] = (intersect_pts[3 * (2 * ii) + 2] + intersect_pts[3 * (2 * ii + 1) + 2] + truncated_face_center[2])/3;
            // arma::mat triMat(2, 3);
            // triMat[0,0] = intersect_pts[3 * (2 * ii)]     - truncated_face_center[0];
            // triMat[0,1] = intersect_pts[3 * (2 * ii) + 1] - truncated_face_center[1];
            // triMat[0,2] = intersect_pts[3 * (2 * ii) + 2] - truncated_face_center[2];

            // triMat[1,0] = intersect_pts[3 * (2 * ii + 1)]     - truncated_face_center[0];
            // triMat[1,1] = intersect_pts[3 * (2 * ii + 1) + 1] - truncated_face_center[1];
            // triMat[1,2] = intersect_pts[3 * (2 * ii + 1) + 2] - truncated_face_center[2];
            // double area = sqrt(abs(arma::det(triMat * triMat.t()))) * 0.5;

            arma::vec3 v1;
            v1[0] = intersect_pts[3 * (2 * ii)]     - truncated_face_center[0];
            v1[1] = intersect_pts[3 * (2 * ii) + 1] - truncated_face_center[1];
            v1[2] = intersect_pts[3 * (2 * ii) + 2] - truncated_face_center[2];
            arma::vec3 v2;
            v2[0] = intersect_pts[3 * (2 * ii + 1)]     - truncated_face_center[0];
            v2[1] = intersect_pts[3 * (2 * ii + 1) + 1] - truncated_face_center[1];
            v2[2] = intersect_pts[3 * (2 * ii + 1) + 2] - truncated_face_center[2];
            arma::vec3 v3 = arma::cross(v1, v2);
            double area = sqrt(arma::dot(v3, v3)) * 0.5;

            f_area += area;
            f_centroid += tri_center * area;
        }
        // f_centroid /= f_area;
        arma::vec3 curPt = {in_pt[0], in_pt[1], in_pt[2]};
        double pt_dist = PtDistance(in_pt, nei_pt) + 1e-16;
        // gradient = f_area / pt_dist * (f_centroid - curPt);
        
        gradient = f_centroid / pt_dist  -  curPt * f_area / pt_dist;

        const double* trunc_pt = intersect_pts; 
        int cur_tet_id = 0;
        for(size_t i = 0; i < f_num; ++i)
        {
            const int f_tet_num = face_tet_count[i];
            const auto f_pt = trunc_tets[cur_tet_id*2];
            // printf("cur face f_tet_num : %d \n", f_tet_num);
            for(int ti = 0; ti < f_tet_num; ++ti )
            {
                double volume = CalTetrahedronVolumeDet(trunc_tets[cur_tet_id*2], 
                                                        trunc_tets[cur_tet_id*2 + 1], 
                                                        f_pt,
                                                        trunc_pt);
                cell_volume += volume;
                cur_tet_id ++;
            }
        }
        return cell_volume;
    }
    return 0;
}

void VoronoiGen::OutputTetMesh(const std::string& tetmesh_path)
{
    std::ofstream mesh_file;
    mesh_file.open(tetmesh_path);
    // for(auto pt : points_)
    // {
    //     mesh_file << "v " << pt[0] << " " << pt[1] << " " << pt[2] << std::endl; 
    // }

    tetMesh_.points->traversalinit();
    tetgenmesh::point ptloop;
    ptloop = tetMesh_.pointtraverse();
    while (ptloop != (tetgenmesh::point) NULL) {
        mesh_file << "v " << ptloop[0] << " " << ptloop[1] << " " << ptloop[2] << std::endl; 
        ptloop = tetMesh_.pointtraverse();
    }

    tetgenmesh::triface tetloop;
    tetMesh_.tetrahedrons->traversalinit();
    tetloop.tet = tetMesh_.alltetrahedrontraverse();
    size_t t_count = 0;
    while(tetloop.tet != (tetgenmesh::tetrahedron*) NULL)
    {
        auto pts = (tetgenmesh::point *) tetloop.tet;
        if(pts[4] == (tetgenmesh::point) NULL || pts[5] == (tetgenmesh::point) NULL ||
          pts[6] == (tetgenmesh::point) NULL || pts[7] == (tetgenmesh::point) NULL) 
        {
            tetloop.tet = tetMesh_.alltetrahedrontraverse();
            continue;
        }
      
        int idx0 = tetMesh_.pointmark(pts[4]); 
        int idx1 = tetMesh_.pointmark(pts[5]); 
        int idx2 = tetMesh_.pointmark(pts[6]); 
        int idx3 = tetMesh_.pointmark(pts[7]); 

        if(idx0 == -1 || idx1 == -1 || idx2 == -1 || idx3 == -1)
        {
            tetloop.tet = tetMesh_.alltetrahedrontraverse();
            continue;
        }

        mesh_file << "l " << idx0 + 1 << " " << idx1 + 1 << std::endl;
        mesh_file << "l " << idx0 + 1 << " " << idx2 + 1 << std::endl;
        mesh_file << "l " << idx0 + 1 << " " << idx3 + 1 << std::endl;
        mesh_file << "l " << idx1 + 1 << " " << idx2 + 1 << std::endl;
        mesh_file << "l " << idx1 + 1 << " " << idx3 + 1 << std::endl;
        mesh_file << "l " << idx2 + 1 << " " << idx3 + 1 << std::endl;
        tetloop.tet = tetMesh_.alltetrahedrontraverse();
    }

    // tetMesh_.subsegs->traversalinit();
    // tetgenmesh::face segloop;
    // segloop.sh = tetMesh_.shellfacetraverse(tetMesh_.subsegs);
    // while (segloop.sh != (tetgenmesh::shellface *) NULL) {
    //     auto torg = tetMesh_.sorg(segloop);
    //     auto tdest = tetMesh_.sdest(segloop);
    //     int idx0 = tetMesh_.pointmark(torg); 
    //     int idx1 = tetMesh_.pointmark(tdest);
    //     mesh_file << "l " << idx0 + 1 << " " << idx1 + 1 << std::endl; 
    //     segloop.sh = tetMesh_.shellfacetraverse(tetMesh_.subsegs);
    // }
    mesh_file.close();
}


void VoronoiGen::OutputVoronisMesh()
{
    std::string out_path = out_dir_ + "voronoi_cell.obj";
    std::string out_path2 = out_dir_ + "voronoi_split_pts.obj";
    std::cout << " voronoi out put size : " << out_path << std::endl;
    // std::cout << " voronoi tet size : " << tetMesh_.tetrahedrons->items << std::endl;
    std::ofstream myfile;
    myfile.open(out_path);
    // ofstream myfile_split;
    // myfile_split.open(out_path2);
    size_t vp_num = voronoi_data_.numberofvpoints;
    // std::cout << " voronoi point size : " << vp_num << std::endl;
    auto& vplist = voronoi_data_.vpointlist;
    for(size_t i = 0; i < vp_num; ++i)
    {
        myfile << "v " << vplist[3*i] << " " << vplist[3*i + 1] << " " << vplist[3*i + 2] << std::endl;
    }
    size_t e_num = voronoi_data_.numberofvedges;
    auto& v_edges = voronoi_data_.vedgelist;
    // int incre_pt_count = 0;
    // for(size_t i =0; i < e_num; ++i)
    // {
    //     if( v_edges[i].v2 == -1)
    //     {
    //         double nx =  vplist[3*v_edges[i].v1]     + v_edges[i].vnormal[0] * 10;
    //         double ny =  vplist[3*v_edges[i].v1 + 1] + v_edges[i].vnormal[1] * 10;
    //         double nz =  vplist[3*v_edges[i].v1 + 2] + v_edges[i].vnormal[2] * 10;
    //         myfile << "v " << nx << " " << ny << " " << nz << std::endl;
    //         v_edges[i].v2 = vp_num + incre_pt_count;
    //         incre_pt_count ++;
    //     }
    // }

    for(size_t i =0; i < e_num; ++i)
    {
        if(v_edges[i].v1 == -1 || v_edges[i].v2 == -1) continue;
        myfile << "l " << v_edges[i].v1 + 1 << " " << v_edges[i].v2 + 1 << std::endl;
    }
    myfile.close();

    // size_t cell_num = voronoi_data_.numberofvcells;
    // size_t face_num = voronoi_data_.numberofvfacets;
    // auto v_faces = voronoi_data_.vfacetlist;
    // auto v_cells = voronoi_data_.vcelllist;
    // printf("pt size : %ld \n", points_.size());
    // size_t f_vcount = 0;
    // for(size_t i = 0; i < face_num; ++i)
    // {
    //     f_vcount += v_faces[i].elist[0];
    // }
    // f_vcount += face_num * 3;
    // printf("voro face memory size : %ld \n", f_vcount * 4);

    // size_t c_fcount = 0;
    // for(size_t i = 0; i < cell_num; ++i)
    // {
    //     if(v_cells[i])
    //     {
    //         c_fcount += v_cells[i][0];
    //     }
    // }
    // c_fcount += cell_num ;
    // printf("voro cell memory size : %ld \n", c_fcount * 4);

    // tetMesh_.points->traversalinit();
    // tetgenmesh::point ploop = tetMesh_.pointtraverse();
    // while(ploop != (tetgenmesh::point)NULL)
    // {
    //     // printf("pt size 00 : %ld \n", points_.size());
    //     const auto p_type = tetMesh_.pointtype(ploop);
    //     if ((p_type == tetgenmesh::UNUSEDVERTEX)) 
    //     {
    //         ploop = tetMesh_.pointtraverse();
    //         continue;
    //     } 
    //     std::vector<tetgenio::voroedge*> edge_list;
    //     GetVoroCellEdgeList(ploop, edge_list);
    //     std::vector<double> pt_list;
    //     GetVoroCellPtList(edge_list, pt_list);

    //     VoroPlane v_plane;
    //     v_plane.nx = 0.5;
    //     v_plane.ny = 0.5;
    //     v_plane.nz = 0.5;
    //     v_plane.px = ploop[0] + 0.05;
    //     v_plane.py = ploop[1] + 0.05;
    //     v_plane.pz = ploop[2] + 0.05;
    //     double cur_pt[3];

    //     cur_pt[0] = v_plane.px;
    //     cur_pt[1] = v_plane.py;
    //     cur_pt[2] = v_plane.pz;
    //     auto split_pts = GetPolygonSplitPts(cur_pt, ploop);

    //     // CalSplitPolyhedronVolume(split_pts);

    //     // int v_id = split_pts.size()/3;
    //     // for(auto edge : edge_list)
    //     // {
    //     //     if(CalPlaneEdgeIntersection(edge, v_plane))
    //     //     {
    //     //         split_pts.push_back(v_plane.intPx);
    //     //         split_pts.push_back(v_plane.intPy);
    //     //         split_pts.push_back(v_plane.intPz);
    //     //         // myfile_split << "v " << v_plane.intPx << " " << v_plane.intPy << " " << v_plane.intPz << std::endl;
    //     //         // v_id++;
    //     //     }
    //     // }

    //     // for(size_t p_id = 0; p_id < split_pts.size()/3; ++p_id)
    //     // {
    //     //     myfile_split << "v " << split_pts[3*p_id]<< " " 
    //     //     << split_pts[3*p_id + 1] << " " <<  split_pts[3*p_id + 2] << std::endl;
    //     // }

    //     // myfile_split << "f ";
    //     // for(size_t v_i = 0; v_i < v_id; ++v_i)
    //     // {
    //     //     myfile_split << v_i + 1 << " ";
    //     // }
    //     // myfile_split <<std::endl;
    //     // myfile_split.close();
      
    //     // printf("pt size 02 : %ld \n", points_.size());
    //     // for(auto edge : edge_list)
    //     // {
    //     //     myfile << "l " << edge->v1 + 1 << " " << edge->v2 + 1 << std::endl;
    //     // }
    //     // printf("pt size : %ld \n", points_.size());
    //     ploop = tetMesh_.pointtraverse();
    //     // break;
    // }

    // for(auto& n_pt : points_)
    // {
    //     std::vector<tetgenio::voroedge*> edge_list;
    //     GetVoroCellEdgeList(n_pt, edge_list);
    //     for(auto edge : edge_list)
    //     {
    //         myfile << "l " << edge->v1 + 1 << " " << edge->v2 + 1 << std::endl;
    //     }
    //     // break;
    // }
    
    // for(voronoi_data_.vpointlist)

}


void VoronoiGen::GetVertexStar(tetgenmesh::point &p_st, 
            std::set<tetgenmesh::point>& candid_pts, int level = 1)
{
    tetlist_->restart();
    ptlist_->restart();
    tetMesh_.getvertexstar(1, p_st, tetlist_,  ptlist_, NULL);
    candid_pts.insert(p_st);
    for (int i = 0; i < ptlist_->objects; i++) {
        auto pt = * (tetgenmesh::point *) ptlist_-> lookup(i);
        if ((tetMesh_.pointtype(pt) == tetgenmesh::verttype::UNUSEDVERTEX) ||
        (tetMesh_.pointtype(pt) == tetgenmesh::verttype::DUPLICATEDVERTEX) ||
        (tetMesh_.pointtype(pt) == tetgenmesh::verttype::NREGULARVERTEX) ||
        (tetMesh_.pointtype(pt) == tetgenmesh::verttype::DEADVERTEX) ) 
        {
            continue;
        }
        if(point_id_map_.find(pt) == point_id_map_.end()) continue;
        // if(abs(pt[0]) <= 1e-3 && abs(pt[1]) < 1e-3 && abs(pt[2]) < 1e-3)
        // {
        //     // printf("zero pt : %f %f %f \n", pt[0], pt[1], pt[2]);
        //     continue;
        // }
        if(pt != (tetgenmesh::point)(NULL)) candid_pts.insert(pt);
    }
}
void GenerateTetTriSamplePts( tetgenmesh::point cp,  tetgenmesh::point p0,  
                              tetgenmesh::point p1,  tetgenmesh::point p2, 
                              std::vector<double>& pts)
{

    double f0_cx = ((p0[0] + p1[0] + p2[0]) + cp[0])/4.0;
    double f0_cy = ((p0[1] + p1[1] + p2[1]) + cp[1])/4.0;
    double f0_cz = ((p0[2] + p1[2] + p2[2]) + cp[2])/4.0;
    pts.push_back(f0_cx);
    pts.push_back(f0_cy);
    pts.push_back(f0_cz);

    // double p0x = (f0_cx + 4.0 * p0[0]) / 5.0;
    // double p0y = (f0_cy + 4.0 * p0[1]) / 5.0;
    // double p0z = (f0_cz + 4.0 * p0[2]) / 5.0;
    // results.push_back(p0x);
    // results.push_back(p0y);
    // results.push_back(p0z);

    // double p1x = (f0_cx + 4.0 * p1[0]) / 5.0;
    // double p1y = (f0_cy + 4.0 * p1[1]) / 5.0;
    // double p1z = (f0_cz + 4.0 * p1[2]) / 5.0;
    // results.push_back(p1x);
    // results.push_back(p1y);
    // results.push_back(p1z);

    // double p2x = (f0_cx + 4.0 * p2[0]) / 5.0;
    // double p2y = (f0_cy + 4.0 * p2[1]) / 5.0;
    // double p2z = (f0_cz + 4.0 * p2[2]) / 5.0;
    // results.push_back(p2x);
    // results.push_back(p2y);
    // results.push_back(p2z);

}

void VoronoiGen::BuildTetMeshTetCenterMap()
{
    tetMesh_.tetrahedrons->traversalinit();
    tetgenmesh::triface tetface;
    tetface.tet = tetMesh_.alltetrahedrontraverse();
    size_t tet_count = 0;
    tetgenmesh::point torg, tdest, tapex, toppo;
    tet_center_pts_.clear();
    tc_pt_tet_map_.clear();
    while(tetface.tet != (tetgenmesh::tetrahedron*)(NULL))
    {
        torg  = tetMesh_.org(tetface);
        tdest = tetMesh_.dest(tetface);
        tapex = tetMesh_.apex(tetface);
        toppo = tetMesh_.oppo(tetface);
        // if(tetMesh_.ishulltet(tetface))
        {
            double cx = (torg[0] + tdest[0] + tapex[0] + toppo[0])/ 4.0; 
            double cy = (torg[1] + tdest[1] + tapex[1] + toppo[1])/ 4.0; 
            double cz = (torg[2] + tdest[2] + tapex[2] + toppo[2])/ 4.0; 
         
            tet_center_pts_.push_back(cx);
            tet_center_pts_.push_back(cy);
            tet_center_pts_.push_back(cz);
            tc_pt_tet_map_[tet_count] = tetface.tet;
            tet_count++;
           
            // tet_center_pts_.push_back((cx + 9*torg[0])/10.0);
            // tet_center_pts_.push_back((cy + 9*torg[1])/10.0);
            // tet_center_pts_.push_back((cz + 9*torg[2])/10.0);
            // tc_pt_tet_map_[tet_count] = tetface.tet;
            // tet_count++;
            // tet_center_pts_.push_back((cx + 9*tdest[0])/10.0);
            // tet_center_pts_.push_back((cy + 9*tdest[1])/10.0);
            // tet_center_pts_.push_back((cz + 9*tdest[2])/10.0);
            // tc_pt_tet_map_[tet_count] = tetface.tet;
            // tet_count++;
            // tet_center_pts_.push_back((cx + 9*tapex[0])/10.0);
            // tet_center_pts_.push_back((cy + 9*tapex[1])/10.0);
            // tet_center_pts_.push_back((cz + 9*tapex[2])/10.0);
            // tc_pt_tet_map_[tet_count] = tetface.tet;
            // tet_count++;
            // tet_center_pts_.push_back((cx + 9*toppo[0])/10.0);
            // tet_center_pts_.push_back((cy + 9*toppo[1])/10.0);
            // tet_center_pts_.push_back((cz + 9*toppo[2])/10.0);
            // tc_pt_tet_map_[tet_count] = tetface.tet;
            // tet_count++;
            
            // auto p0 = torg;
            // auto p1 = tdest;
            // auto p2 = tapex;
            // auto p3 = toppo;
            // double cp[3] = {cx, cy, cz};
            // std::vector<double> new_pts;
            // GenerateTetTriSamplePts(cp, p0, p1, p2, new_pts);
            // GenerateTetTriSamplePts(cp, p0, p1, p3, new_pts);
            // GenerateTetTriSamplePts(cp, p0, p2, p3, new_pts);
            // GenerateTetTriSamplePts(cp, p3, p1, p2, new_pts);
            
            // for(size_t i =0; i < new_pts.size()/3; ++i)
            // {
            //     tet_center_pts_.push_back(new_pts[3*i]);
            //     tet_center_pts_.push_back(new_pts[3*i + 1]);
            //     tet_center_pts_.push_back(new_pts[3*i + 2]);
            //     tc_pt_tet_map_[tet_count] = tetface.tet;
            //     tet_count++;
            // }
               
        }
        tetface.tet = tetMesh_.alltetrahedrontraverse();
    }
    // std::string tet_center_path = out_dir_ + filename_ + "/" + "tet_centers";
    // writePLYFile(tet_center_path, tet_center_pts_);
}




tetgenmesh::tetrahedron* VoronoiGen::GetClosetTet(double x, double y, double z)
{
    size_t tet_id = pTree_.SearchNearestPt(x, y, z);
    // std::cout << " nearest tet id : " << tet_id << std::endl;
    // size_t tet_id = 0;
    // double dx = tet_center_pts_[tet_id * 3]     - x;
    // double dy = tet_center_pts_[tet_id * 3 + 1] - y;
    // double dz = tet_center_pts_[tet_id * 3 + 2] - z;
    // std::cout << "mini dist : " << sqrt(dx * dx + dy * dy + dz * dz) << std::endl;
    return tc_pt_tet_map_[tet_id];
}

void VoronoiGen::GetVoronoiNeiPts(tetgenmesh::point pt, std::vector<tetgenmesh::point>& candid_pts)
{
    // tetlist_->restart();
    
    tetgenmesh::triface searchtet;
    // auto t001 = Clock::now();
    // auto closet_tet = GetClosetTet(pt[0], pt[1], pt[2]);
    size_t tet_id = pTree_.SearchNearestPt(pt[0], pt[1], pt[2]);
    const auto& closet_tet = tc_pt_tet_map_[tet_id];

    // auto t002 = Clock::now();
    // tetgenmesh::tet_search_time_st += std::chrono::nanoseconds(t002 - t001).count()/ (1e9);

    tetMesh_.decode(*closet_tet, searchtet);
    // searchtet.tet = NULL;
    // searchtet.tet = GetClosetTet(pt[0], pt[1], pt[2]);
    // printf(" GetInsertPointBoundry .......... \n");
    // ptlist_->restart();
    // tetMesh_.GetInsertPointBoundry(pt, &search_tet, ptlist_);
    tetgenmesh::insertvertexflags ivf;
    ivf.bowywat = 1; // Use Bowyer-Watson algorithm
    ivf.lawson = 0;
    ivf.validflag = 0;
    // tetgenmesh::triface searchtet;
    // searchtet.tet = NULL;
    ivf.iloc = (int) tetgenmesh::UNKNOWN;
    auto t0 = Clock::now();
    tetgenmesh::face *splitsh = NULL;
    tetgenmesh::face *splitseg = NULL;
    // temp_mesh.setpointtype(newpt, tetgenmesh::UNUSEDVERTEX);
    // printf("start GetCaveBoundryPoint ... \n");
    // tetgenmesh::arraypool *out_cavetetvertlist ;
    std::vector<tetgenmesh::point> out_cavetetvertlist;
    tetMesh_.GetCaveBoundryPoint(pt, &searchtet, splitsh, splitseg, &ivf, out_cavetetvertlist);
    for(const auto pt : out_cavetetvertlist)
    {
        auto pt_type = tetMesh_.pointtype(pt);
        if(( pt_type != tetgenmesh::UNUSEDVERTEX) && ( pt_type != tetgenmesh::DUPLICATEDVERTEX))
        {
            candid_pts.push_back(pt);
        }
    }
}

void VoronoiGen::InsertPt(tetgenmesh::point pt)
{
    tetgenmesh::point newpt;
    // auto temp_mesh = tetgenmesh(tetMesh_);
    auto& temp_mesh = tetMesh_;
    printf("deep copy succed! \n");

    temp_mesh.makepoint(&newpt, tetgenmesh::UNUSEDVERTEX);
    newpt[0] = pt[0];
    newpt[1] = pt[1];
    newpt[2] = pt[2];
    tetgenmesh::insertvertexflags ivf;
    ivf.bowywat = 1; // Use Bowyer-Watson algorithm
    ivf.lawson = 0;

    tetgenmesh::triface searchtet;
    searchtet.tet = NULL;
  
    ivf.iloc = (int) tetgenmesh::UNKNOWN;
    auto t0 = Clock::now();
    tetgenmesh::face *splitsh = NULL;
    tetgenmesh::face *splitseg = NULL;
    temp_mesh.setpointtype(newpt, tetgenmesh::VOLVERTEX);

    printf("-----------init tet num: %lu ...\n", tetMesh_.tetrahedrons->items);
    if(temp_mesh.insertpoint(newpt, &searchtet, splitsh, splitseg,  &ivf))
    // if (tetMesh_.insert_vertex_bw(newpt, &searchtet, &ivf))
    {
        std::set<tetgenmesh::point> candidate_pts;
        GetVertexStar(newpt, candidate_pts, 1);
        printf("--------- insert pt  nei points num : %lu \n", candidate_pts.size());
        // temp_mesh.removevertexbyflips();
        // printf("insert successfully ...\n");
        printf("current pt type: %d \n", tetMesh_.pointtype(newpt));
        // printf("current pt num: %lu ...\n", tetMesh_.points->items);
        printf("      Removing a point %d in volume.\n",
             tetMesh_.pointmark(newpt));
             
        printf("-----------insert tet num: %lu ...\n", tetMesh_.tetrahedrons->items);
        
        // tetMesh_.points->dealloc(newpt);
        temp_mesh.b->fliplinklevel = 100000;
        temp_mesh.removevertexbyflips(newpt);
        // temp_mesh.removevertexbyflips(newpt);
        // temp_mesh.removevertexbyflips(newpt);
        candidate_pts.clear();
        GetVertexStar(newpt, candidate_pts, 1);
        printf("after remove insert pt  nei points num : %lu \n", candidate_pts.size());
        // tetMesh_.ed
        // tetMesh_.points->dealloc(newpt);

        // temp_mesh.removevertexbyflips(newpt);
        printf("----------remove tet num : %lu ...\n", tetMesh_.tetrahedrons->items);
        printf("current pt num after remove: %lu ...\n", tetMesh_.points->items);
        // tetMesh_.removevertexbyflips();
        // printf("remove successfully ...\n");
        //  printf("current pt num: %llu ...\n", tetMesh_.points->items);
    }
    auto t1 = Clock::now();
    // double insert_time = std::chrono::nanoseconds(t1 - t0).count()/1e9;
    // printf(" point insert time : %g \n", insert_time);
    


    // printf("start to insert ........ \n");
    // tetMesh_.insertconstrainedpoints(insertarray, arylen, rejflag);
}


void VoronoiGen::BuildPtIdMap()
{
    tetMesh_.points->traversalinit();
    tetgenmesh::point ploop = tetMesh_.pointtraverse();
    size_t pt_num_ = 0;
    points_.clear();
    while(ploop != (tetgenmesh::point)NULL)
    {
        const auto p_type = tetMesh_.pointtype(ploop);
        if ((p_type == tetgenmesh::UNUSEDVERTEX) || (p_type == tetgenmesh::DUPLICATEDVERTEX)) 
        {
            ploop = tetMesh_.pointtraverse();
            continue;
        } 
        points_.push_back(ploop);
        point_id_map_[ploop] = pt_num_;
        ploop = tetMesh_.pointtraverse();
        pt_num_ ++;
    }
    pt_adjecent_mat_.resize(pt_num_, pt_num_);
    printf("pt num : %zu \n", pt_num_);
    printf("point_id_map_ size : %zu \n", point_id_map_.size());

}

void VoronoiGen::BuildAdjecentMat()
{
    tetgenmesh::point ploop = tetMesh_.pointtraverse();
    // points_.clear()
    
    cluster_init_pids_.resize(point_id_map_.size());
    // cluster_init_pts_.resize(point_id_map_.size());
    cluster_size_vec_.resize(point_id_map_.size() + 1);
    cluster_size_vec_.zeros();
    for(auto ploop : points_)
    {
        size_t cur_p_id = point_id_map_[ploop]; 
        std::set<tetgenmesh::point> candidate_pts;
        GetVertexStar(ploop, candidate_pts, 1);
        // point_cluster_pts_map_[ploop] = candidate_pts;
        std::vector<int> cur_cluster_pids;
        cur_cluster_pids.push_back(cur_p_id);
        // std::vector<double> cur_cluster_pts;

        // cur_cluster_pts.push_back(ploop[0]);
        // cur_cluster_pts.push_back(ploop[1]);
        // cur_cluster_pts.push_back(ploop[2]);
        if(candidate_pts.size() > 1000)
        {

            std::cout << " strange pt nn size " << candidate_pts.size() << std::endl;
            singular_pts_.push_back({ploop[0], ploop[1], ploop[2]});
            // for(const auto& pt : candidate_pts)
            // {
            //     singular_pts_.push_back({pt[0], pt[1], pt[2]});
            // }
        }

        double dist_threshold = 1e-5;
        candidate_pts.erase(ploop);
        std::vector<double> pt_dists(candidate_pts.size(), 0);
        std::vector<size_t> pids(candidate_pts.size(), 0);
        size_t nn_count = 0;
        for(auto &pt : candidate_pts)
        {
            if(pt == ploop) 
            {
                // std::cout << " pt == ploop indices " << std::endl;
                nn_count ++;
                continue;
            }
            if(point_id_map_.find(pt) == point_id_map_.end()) continue;
            size_t p_id = point_id_map_[pt];
            bool too_close_pt = false;
            // for(const auto n_id : cur_cluster_pids)
            // {
            //     double cur_dist = PtDistance(points_[p_id], points_[n_id]);
            //     pt_dists[nn_count] = cur_dist;
            //     // if(cur_dist < dist_threshold)
            //     // {
            //     //     too_close_pt = true;
            //     //     break;
            //     // }
            //     
            // }
            double cur_dist = PtDistance(points_[p_id], points_[cur_p_id]);
            pt_dists[nn_count] = cur_dist;
            pids[nn_count] = p_id;
            nn_count ++;
            // if(!too_close_pt)
            // {
            //     pt_adjecent_mat_(cur_p_id, p_id) = 1;
            //     cur_cluster_pids.push_back(p_id);
            // } 
        }
        // for( auto id : indices)
        // {
        //     std::cout << " " << id << " " ;
        // }
        // std::cout << endl;
        // for( auto dist : pt_dists)
        // {
        //     std::cout << " " << dist << " " ;
        // }
        // std::cout << endl;
        // Initialize indices: 0, 1, 2, ...
        // std::cout << " nn_count indices size  " << nn_count << std::endl;

        vector<int> indices(pids.size());
        iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&pt_dists](int pa, int pb) {
                return pt_dists[pa] < pt_dists[pb];
            });
        // std::cout << " finish to sort indices " << std::endl;

        int max_nn_num = 1000;
        for(int idx = 0; idx < indices.size(); ++idx )
        {
            if(idx < max_nn_num)
            {
                size_t new_id = indices[idx];
                size_t p_id = pids[new_id];
                // std::cout << " nn p_id " << p_id<<  std::endl;
                pt_adjecent_mat_(cur_p_id, p_id) = 1;
                cur_cluster_pids.push_back(p_id);
            }
        }
        // std::cout << " finish to sort indices  22" << std::endl;

        ploop = tetMesh_.pointtraverse();
        cluster_init_pids_[cur_p_id] = cur_cluster_pids;
        // cluster_init_pts_[cur_p_id]  = cur_cluster_pts;
        cluster_size_vec_[cur_p_id + 1] = cur_cluster_pids.size();
        // printf("cur pt id 000 : %d \n", cur_p_id);
    }
    average_neighbor_num_ = size_t((double) arma::sum(cluster_size_vec_) / double(points_.size()));
    cluster_accum_size_vec_.resize(points_.size() + 1);
    cluster_accum_size_vec_[0] = 0;
    for(int i = 1; i < int(points_.size()); ++i)
    {
        cluster_accum_size_vec_[i] = cluster_accum_size_vec_[i-1] + cluster_size_vec_[i];
    }
    // printf("finish loop pt id 000\n");
}

void VoronoiGen::BuildAdjecentMatKNN()
{
    // tetgenmesh::point ploop = tetMesh_.pointtraverse();
    cluster_init_pids_.resize(point_id_map_.size());
    cluster_size_vec_.resize(point_id_map_.size() + 1);
    cluster_size_vec_.zeros();
    PicoTree kd_tree;
    kd_tree.Init(points_);
    for(auto pt : points_)
    {
        size_t cur_p_id = point_id_map_[pt]; 
        int k = 8;
        auto nn_pids = kd_tree.SearchNearestKNN(pt[0], pt[1], pt[2], k);
        std::vector<int> cluster_pids;
        cluster_pids.push_back(cur_p_id);
        for(auto n_id : nn_pids)
        {
            if(n_id != cur_p_id)
            {
                pt_adjecent_mat_(cur_p_id, n_id) = 1;
                cluster_pids.push_back(n_id);
            }
        }
        cluster_init_pids_[cur_p_id] = cluster_pids;
        cluster_size_vec_[cur_p_id + 1] = cluster_pids.size();

        // printf("cur pt id 000 : %d \n", cur_p_id);
    }
    average_neighbor_num_ = size_t((double) arma::sum(cluster_size_vec_) / double(points_.size()));
    cluster_accum_size_vec_.resize(points_.size() + 1);
    cluster_accum_size_vec_[0] = 0;
    for(int i = 1; i < int(points_.size()); ++i)
    {
        cluster_accum_size_vec_[i] = cluster_accum_size_vec_[i-1] + cluster_size_vec_[i];
    }
    // printf("finish loop pt id 000\n");
}


void VoronoiGen::InitMesh()
{
    
    Tetrahedralize();
    // BuildPtIdMap();
    
    tetlist_ = new tetgenmesh::arraypool(sizeof(tetgenmesh::triface), 10);
    ptlist_ = new tetgenmesh::arraypool(sizeof(tetgenmesh::point), 10);

    tetlist2_ = new tetgenmesh::arraypool(sizeof(tetgenmesh::triface), 10);
    ptlist2_ = new tetgenmesh::arraypool(sizeof(tetgenmesh::point), 10);

    tetlist3_ = new tetgenmesh::arraypool(sizeof(tetgenmesh::triface), 10);
    ptlist3_ = new tetgenmesh::arraypool(sizeof(tetgenmesh::point), 10);
}

void VoronoiGen::BuildPicoTree()
{
    pTree_.Init(tet_center_pts_);
}


std::unordered_map<tetgenmesh::point, int> VoronoiGen::point_id_map_;
std::vector<std::vector<int>> VoronoiGen::cluster_init_pids_;
// std::vector<std::vector<double>> VoronoiGen::cluster_init_pts_;
arma::ivec VoronoiGen::cluster_size_vec_; 
arma::ivec VoronoiGen::cluster_accum_size_vec_;