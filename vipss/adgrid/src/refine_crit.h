//
//  main.cpp
//  tet_subdivision
//
//  Created by Yiwen Ju on 12/2/23.
//
#pragma once

#include <array>
#include <SmallVector.h>
// #include <convex_hull_membership/contains.h>
#include <contains.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include "implicit_functions/Hermite_RBF.h"
#include <fstream>
#include <mtet/mtet.h>

class OutTetObj{

    public:
        OutTetObj(){};
        ~OutTetObj(){};

        void AddNewTet(const Eigen::Matrix<double, 4, 3> &pts);
        void SaveTetDataToObj(const std::string& out_path);

    public:

    std::vector<std::array<double,3>> pts_;
    std::vector<std::array<int, 2>> edges_; 
    std::vector<std::array<int,3>> faces_;
    static OutTetObj unoriented_tets;
    static OutTetObj unoriented_tets_split;
    static OutTetObj unoriented_tets_unsplit;
    static OutTetObj boundary1_unorientable_tets;
    static OutTetObj boundary2_RV_tets;
    static OutTetObj boundary3_eprime_tets;
    static OutTetObj limit_orientable_tets;
    static OutTetObj rv_failed_tets;
    static OutTetObj threshold_tets;
    static OutTetObj eprime_failed_tets;
    
};


std::unordered_map<std::string, std::array<double,8>>& GetTetEdgeMidSampleMap();

/// Enums for the current settings of implicit complexes
enum geo_obj {
    IA, /*0*/
    CSG, /*1*/
    MI /*2*/
};

/// Three types of refinement criteria based on the modality.

/// This function performs two checks (zero-crossing and distance checks) under the setting of implicit arrangement(IA) and its curve network.
///
/// @param[in] pts          4 arrays of three-tuples represent the coordinate of tet vertices. Each subarray represents a coordinate.
/// @param[in] tet_info         A 4 by funcNum by 4 data structure. Frist layer represents 4 vertices of the tet. Second layer stores all values and gradients of all functions of 1 vertex. Third layer (Eigen::RowVector4d) represents the value at 0th index and gradents at {1, 2, 3} indices.
/// @param[in] funcNum          The number of functions.
/// @param[in] threshold            The user-input epsilon value for the distance check.
/// @param[in] curve_network            A  bool represents whether we only want to compute curve network.
/// @param[out] active          A `bool` represents whether the tet is containing part of the geometry, i.e., this tet passes the zero-crossing test.
/// @param[out] sub_call_two            A tracker of how many times two functions' distance check is called.
/// @param[out] sub_call_two            A tracker of how many times three functions' distance check is called.
///
/// @return         A `bool` represents whether the tet is "refinable".
///  i.e., passing the zero-crossing test and contains error greater than `threshold`.
///
bool critIA(const Eigen::Matrix<double, 4, 3> &pts,
            const std::array<llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20>,4>& tet_info,
            const size_t funcNum,
            const double threshold,
            const bool curve_network,
            bool &active,
            int &sub_call_two,
            int &sub_call_three);

//boundary_type 0, means no boundary, 1 means unorientable tet, 2 ridge valley switch tet, 3 e prime 0 crossing

bool critIARidge(
            const Eigen::Matrix<double, 4, 3> &pts,
            const std::array<llvm_vecsmall::SmallVector<CurvatureData<double>, 20>,4>& tet_info,
            const mtet::TetId tid,
            mtet::MTetMesh &grid_mesh,
            const size_t funcNum,
            const double threshold,
            const bool curve_network,
            const int crest_type,
            bool& active,
            bool& orientable,
            int &sub_call_two,
            int &sub_call_three,
            int& boundary_type);

///This function performs two checks (zero-crossing and distance checks) under the setting of constructive solid geometry(CSG) and its curve network.
///The parameters follow the same style of `critIA`. Below is the only different input.
///
///
bool critCSG(const Eigen::Matrix<double, 4, 3> &pts,
             const std::array<llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20>,4>& tet_info,
             const size_t funcNum,
             const std::function<std::pair<std::array<double, 2>, llvm_vecsmall::SmallVector<int, 20>>(llvm_vecsmall::SmallVector<std::array<double, 2>, 20>)> csg_func,
             const double threshold,
             const bool curve_network,
             bool& active,
             int &sub_call_two,
             int &sub_call_three);

/// This function performs two checks (zero-crossing and distance checks) under the setting of material interface (MI) and its curve network.
/// The parameters follow the same as in `critIA`. see above.
bool critMI(const Eigen::Matrix<double, 4, 3> &pts,
            const std::array<llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20>,4>& tet_info,
            const size_t funcNum,
            const double threshold,
            const bool curve_network,
            bool& active,
            int &sub_call_two,
            int &sub_call_three);

