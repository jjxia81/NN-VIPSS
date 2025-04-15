#include "pico_tree.h"
#include <iostream>

void PicoTree::Init(const std::vector<double>& in_pts)
{
    points_.clear();
    for(size_t i = 0; i < in_pts.size()/3; ++i)
    {
        std::array<data_type, 3> pt{in_pts[3*i], in_pts[3*i + 1], in_pts[3*i + 2]};
        points_.push_back(pt);
    }
    kdtree_ = std::make_shared<pico_tree::KdTree<Space>>(points_, max_leaf_size);
    return;
}

void PicoTree::Init(const std::vector<double*>& in_pts)
{
    points_.clear();
    for(const auto ptr : in_pts)
    {
        std::array<data_type, 3> pt{ptr[0], ptr[1], ptr[2]};
        points_.push_back(pt);
    }
    kdtree_ = std::make_shared<pico_tree::KdTree<Space>>(points_, max_leaf_size);
    return;
}

int PicoTree::SearchNearestPt(double x, double y, double z) const 
{
    data_type query[3] = {x, y, z};
    Nn result;
    kdtree_->SearchNn(query, result);
    int pt_id = result.index;
    return pt_id;
}

std::vector<int> PicoTree::SearchNearestKNN(double x, double y, double z, int k = 8) const
{
    data_type query[3] = {x, y, z};
    std::vector<Nn> knn;
    // size_t k = 8;
    kdtree_->SearchKnn(query, k, knn);
    std::vector<int> ids;
    for(const auto& nn : knn)
    {
        ids.push_back(nn.index);
    }
    return ids;
}

double PicoTree::NearestPtDist(double x, double y, double z) const
{
    data_type query[3] = {x, y, z};
    Nn result;
    kdtree_->SearchNn(query, result);
    return result.distance;
}