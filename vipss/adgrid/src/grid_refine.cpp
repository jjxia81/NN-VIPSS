//
//  mesh_refine.cpp
//  adaptive_mesh_refinement
//
//  Created by Yiwen Ju on 8/4/24.
//

#include "grid_refine.h"

using Vec7D = Eigen::Vector<double, 7>;
using IndexMap = ankerl::unordered_dense::map<uint64_t, llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20>>;
using IndexMapRidge = ankerl::unordered_dense::map<uint64_t, llvm_vecsmall::SmallVector<CurvatureData<double>, 20>>;

using tetActive = ankerl::unordered_dense::map<std::span<VertexId, 4>, bool, TetHash, TetEqual>;


bool push_longest_edge(mtet::TetId tid, mtet::MTetMesh &grid, 
        const size_t funcNum, 
        const int mode,
        const int crest_type,
        const double tet_max_edge_len, 
        const double threshold,
        const bool curve_network,
        Eigen::Matrix<double, 4, 3>& pts,
        int &sub_call_two,
        int &sub_call_three, 
        IndexMapRidge& vertex_func_grad_map,
        std::vector<std::pair<mtet::Scalar, mtet::EdgeId>>& Q,
        tetActive &tet_active_map,
        std::array<llvm_vecsmall::SmallVector<CurvatureData<double>, 20>,4>& tet_info,
        const std::function<llvm_vecsmall::SmallVector<CurvatureData<double>, 20>(std::span<const Scalar, 3>, size_t)> func,
        const std::function<std::pair<std::array<double, 2>, llvm_vecsmall::SmallVector<int, 20>>(llvm_vecsmall::SmallVector<std::array<double, 2>, 20>)> csg_func
)
{
    std::span<VertexId, 4> vs = grid.get_tet(tid);
    {
        //Timer eval_timer(evaluation, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        for (int i = 0; i < 4; ++i)
        {
            auto vid = vs[i];
            auto coords = grid.get_vertex(vid);
            pts.row(i) = Eigen::RowVector3d({coords[0], coords[1], coords[2]});
            llvm_vecsmall::SmallVector<std::array<double, 4>, 20> func_gradList(funcNum);
            
            llvm_vecsmall::SmallVector<CurvatureData<double>, 20> func_info(funcNum);
            std::array<double, 4> func_grad;
            if (!vertex_func_grad_map.contains(value_of(vid))) {
                func_info = func(coords, funcNum);
                vertex_func_grad_map[value_of(vid)] = func_info;
            }
            else {
                //func_gradList = vertex_func_grad_map[value_of(vid)];
                func_info = vertex_func_grad_map[value_of(vid)];
            }
            tet_info[i] = func_info;
        }
        //eval_timer.Stop();
    }
    bool isActive = 0;
    bool orientable = false;
    bool subResult;
    {
        // int crest_type = 1;
        // std::cout << " start critIARidge .... " << std::endl;
        // Timer sub_timer(subdivision, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        subResult = critIARidge(pts, tet_info, funcNum, threshold, curve_network, crest_type, isActive,orientable, sub_call_two, sub_call_three);
        // std::cout << " finish critIARidge .... " << std::endl;
        // switch (mode){
        //     case IA:
        //         subResult = critIA(pts, tet_info, funcNum, threshold, curve_network, isActive, sub_call_two, sub_call_three);
        //         break;
        //     case MI:
        //         subResult = critMI(pts, tet_info, funcNum, threshold, curve_network, isActive, sub_call_two, sub_call_three);
        //         break;
        //     case CSG:
        //         subResult = critCSG(pts, tet_info, funcNum, csg_func, threshold, curve_network, isActive, sub_call_two, sub_call_three);
        //         break;
        //     default:
        //         throw std::runtime_error("no implicit complexes specified");
        //         return false;
        // }
        // sub_timer.Stop();
    }
    //vertex_active_map[vertexHash(vs)] = isActive;
    tet_active_map[vs] = isActive;
    //Timer eval_timer(evaluation, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
    if (subResult)
    {
        mtet::EdgeId longest_edge;
        mtet::Scalar longest_edge_length = 0;
        grid.foreach_edge_in_tet(tid, [&](mtet::EdgeId eid, mtet::VertexId v0, mtet::VertexId v1)
                                    {
            auto p0 = grid.get_vertex(v0);
            auto p1 = grid.get_vertex(v1);
            mtet::Scalar l = (p0[0] - p1[0]) * (p0[0] - p1[0]) + (p0[1] - p1[1]) * (p0[1] - p1[1]) +
            (p0[2] - p1[2]) * (p0[2] - p1[2]);
            if (l > longest_edge_length) {
                longest_edge_length = l;
                longest_edge = eid;
            } });
        // double tet_edge_len_limit = 0.05; 
        if(longest_edge_length > tet_max_edge_len)
        {
            Q.emplace_back(longest_edge_length, longest_edge);
        } else {
            if(orientable)
            {
                OutTetObj::limit_orientable_tets.AddNewTet(pts);
            } else {
                OutTetObj::limit_unorientable_tets.AddNewTet(pts);
            }
            
        }
        //eval_timer.Stop();
        return true;
    }
    //eval_timer.Stop();
    return false;
};



bool gridRefine(
                const int mode,
                const bool curve_network,
                const double threshold,
                const double alpha,
                const int max_elements,
                const size_t funcNum,
                const std::function<llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20>(std::span<const Scalar, 3>, size_t)> func,
                const std::function<std::pair<std::array<double, 2>, llvm_vecsmall::SmallVector<int, 20>>(llvm_vecsmall::SmallVector<std::array<double, 2>, 20>)> csg_func,
                mtet::MTetMesh &grid,
                tet_metric &metric_list,
                std::array<double, timer_amount> profileTimer
                )
{
    /// Tet Metric
    int sub_call_two = 0;
    int sub_call_three = 0;

    /// initialize vertex map: vertex index -> {{f_i, gx, gy, gz} | for all f_i in the function}
    
    IndexMap vertex_func_grad_map;
    vertex_func_grad_map.reserve(grid.get_num_vertices());
    
    /// hash for mounting a boolean that represents the activeness to a tet
    
    tetActive tet_active_map;
    tet_active_map.reserve(grid.get_num_tets());

    grid.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3> data)
                            {vertex_func_grad_map[value_of(vid)] = func(data, funcNum);});

    auto comp = [](std::pair<mtet::Scalar, mtet::EdgeId> e0,
                   std::pair<mtet::Scalar, mtet::EdgeId> e1)
    { return e0.first < e1.first; };
    std::vector<std::pair<mtet::Scalar, mtet::EdgeId>> Q;

    Eigen::Matrix<double, 4, 3> pts;
    std::array<llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20>,4> tet_info;
    for (size_t i = 0; i < tet_info.size(); i++){
        tet_info[i].resize(funcNum);
    }
    double activeTet = 0;
    auto push_longest_edge = [&](mtet::TetId tid)
    {
        std::span<VertexId, 4> vs = grid.get_tet(tid);
        {
            //Timer eval_timer(evaluation, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
            for (int i = 0; i < 4; ++i)
            {
                auto vid = vs[i];
                auto coords = grid.get_vertex(vid);
                pts.row(i) = Eigen::RowVector3d({coords[0], coords[1], coords[2]});
                llvm_vecsmall::SmallVector<std::array<double, 4>, 20> func_gradList(funcNum);
                
                llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20> func_info(funcNum);
                std::array<double, 4> func_grad;
                if (!vertex_func_grad_map.contains(value_of(vid))) {
                    func_info = func(coords, funcNum);
                    vertex_func_grad_map[value_of(vid)] = func_info;
                }
                else {
                    //func_gradList = vertex_func_grad_map[value_of(vid)];
                    func_info = vertex_func_grad_map[value_of(vid)];
                }
                tet_info[i] = func_info;
            }
            //eval_timer.Stop();
        }
        bool isActive = 0;
        bool subResult;
        {
            //Timer sub_timer(subdivision, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
            switch (mode){
                case IA:
                    subResult = critIA(pts, tet_info, funcNum, threshold, curve_network, isActive, sub_call_two, sub_call_three);
                    break;
                case MI:
                    subResult = critMI(pts, tet_info, funcNum, threshold, curve_network, isActive, sub_call_two, sub_call_three);
                    break;
                case CSG:
                    subResult = critCSG(pts, tet_info, funcNum, csg_func, threshold, curve_network, isActive, sub_call_two, sub_call_three);
                    break;
                default:
                    throw std::runtime_error("no implicit complexes specified");
                    return false;
            }
            //sub_timer.Stop();
        }
        //vertex_active_map[vertexHash(vs)] = isActive;
        tet_active_map[vs] = isActive;
        //Timer eval_timer(evaluation, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        if (subResult)
        {
            mtet::EdgeId longest_edge;
            mtet::Scalar longest_edge_length = 0;
            grid.foreach_edge_in_tet(tid, [&](mtet::EdgeId eid, mtet::VertexId v0, mtet::VertexId v1)
                                     {
                auto p0 = grid.get_vertex(v0);
                auto p1 = grid.get_vertex(v1);
                mtet::Scalar l = (p0[0] - p1[0]) * (p0[0] - p1[0]) + (p0[1] - p1[1]) * (p0[1] - p1[1]) +
                (p0[2] - p1[2]) * (p0[2] - p1[2]);
                if (l > longest_edge_length) {
                    longest_edge_length = l;
                    longest_edge = eid;
                } });
            Q.emplace_back(longest_edge_length, longest_edge);
            //eval_timer.Stop();
            return true;
        }
        //eval_timer.Stop();
        return false;
    };


    {
        Timer timer(total_time, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        
        // Initialize priority queue.
        grid.seq_foreach_tet([&](mtet::TetId tid, [[maybe_unused]] std::span<const mtet::VertexId, 4> vs)
                             { push_longest_edge(tid); });
        std::make_heap(Q.begin(), Q.end(), comp);
        
        // Keep splitting the longest edge
        while (!Q.empty())
        {
            std::pop_heap(Q.begin(), Q.end(), comp);
            auto [edge_length, eid] = Q.back();
            if (!grid.has_edge(eid)){
                Q.pop_back();
                continue;
            }
            //implement alpha value:
            mtet::Scalar comp_edge_length = alpha * edge_length;
            bool addedActive = false;
            grid.foreach_tet_around_edge(eid,[&](mtet::TetId tid){
                std::span<VertexId, 4> vs = grid.get_tet(tid);
                if(/*vertex_active_map.contains(vertexHash(vs))*/tet_active_map.contains(vs)){
                    if (tet_active_map[vs]){
                        mtet::EdgeId longest_edge;
                        mtet::Scalar longest_edge_length = 0;
                        grid.foreach_edge_in_tet(tid, [&](mtet::EdgeId eid_active, mtet::VertexId v0, mtet::VertexId v1)
                                                 {
                            auto p0 = grid.get_vertex(v0);
                            auto p1 = grid.get_vertex(v1);
                            mtet::Scalar l = (p0[0] - p1[0]) * (p0[0] - p1[0]) + (p0[1] - p1[1]) * (p0[1] - p1[1]) +
                            (p0[2] - p1[2]) * (p0[2] - p1[2]);
                            if (l > longest_edge_length) {
                                longest_edge_length = l;
                                longest_edge = eid_active;
                            }
                        });
                        if (longest_edge_length > comp_edge_length) {
                            Q.emplace_back(longest_edge_length, longest_edge);
                            addedActive = true;
                        }
                    }
                }
            });
            if(addedActive){
                std::push_heap(Q.begin(), Q.end(), comp);
                continue;
            }
            Q.pop_back();
            std::array<VertexId, 2> vs_old = grid.get_edge_vertices(eid);
            //Timer split_timer(splitting, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
            auto [vid, eid0, eid1] = grid.split_edge(eid);
            //split_timer.Stop();
            //std::cout << "Number of elements: " << mesh.get_num_tets() << std::endl;
            if (grid.get_num_tets() > max_elements) {
                break;
            }
            grid.foreach_tet_around_edge(eid0, [&](mtet::TetId tid)
                                         {
                if (push_longest_edge(tid)) {
                    std::push_heap(Q.begin(), Q.end(), comp);
                } });
            grid.foreach_tet_around_edge(eid1, [&](mtet::TetId tid)
                                         {
                if (push_longest_edge(tid)) {
                    std::push_heap(Q.begin(), Q.end(), comp);
                } });
        }
        timer.Stop();
    }
    
    std::vector<mtet::TetId> activeTetId;
    grid.seq_foreach_tet([&](mtet::TetId tid, std::span<const VertexId, 4> data) {
        std::span<VertexId, 4> vs = grid.get_tet(tid);
        std::array<std::valarray<double>,4> vallPoints;
        for (int i = 0; i < 4; i++){
            vallPoints[i] = {0.0,0.0,0.0};
        }
        for (int i = 0; i < 4; i++){
            VertexId vid = vs[i];
            std::span<Scalar, 3> coords = grid.get_vertex(vid);
            vallPoints[i][0] = coords[0];
            vallPoints[i][1] = coords[1];
            vallPoints[i][2] = coords[2];
        }
        double ratio = tet_radius_ratio(vallPoints);
        if (ratio < metric_list.min_radius_ratio){
            metric_list.min_radius_ratio = ratio;
        }
        if(tet_active_map.contains(vs)){
            if (tet_active_map[vs]){
                metric_list.active_tet++;
                activeTetId.push_back(tid);
                if (ratio < metric_list.active_radius_ratio){
                    metric_list.active_radius_ratio = ratio;
                }
            }
        }
    });
    metric_list.total_tet = grid.get_num_tets();
    metric_list.two_func_check = sub_call_two;
    metric_list.three_func_check = sub_call_three;
    metric_list.vertex_func_grad_map = vertex_func_grad_map;
    metric_list.activeTetId = activeTetId;
    //profiled time(see details in time.h) and profiled number of calls to zero
    std::cout << time_label[0] << ": " << profileTimer[0] << std::endl;
    return true;
}



bool gridRefineRidges(
                const int mode,
                const int crest_type, 
                const double tet_max_edge_len,
                const bool curve_network,
                const double threshold,
                const double alpha,
                const int max_elements,
                const size_t funcNum,
                const std::function<llvm_vecsmall::SmallVector<CurvatureData<double>, 20>(std::span<const Scalar, 3>, size_t)> func,
                const std::function<std::pair<std::array<double, 2>, llvm_vecsmall::SmallVector<int, 20>>(llvm_vecsmall::SmallVector<std::array<double, 2>, 20>)> csg_func,
                mtet::MTetMesh &grid,
                tet_metric &metric_list,
                std::array<double, timer_amount> profileTimer
                )
{
    /// Tet Metric
    int sub_call_two = 0;
    int sub_call_three = 0;

    /// initialize vertex map: vertex index -> {{f_i, gx, gy, gz} | for all f_i in the function}
    
    IndexMapRidge vertex_func_grad_map;
    vertex_func_grad_map.reserve(grid.get_num_vertices());
    
    /// hash for mounting a boolean that represents the activeness to a tet
    
    tetActive tet_active_map;
    tet_active_map.reserve(grid.get_num_tets());
    printf("start seq_foreach_vertex .... \n");
    grid.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3> data)
                            {vertex_func_grad_map[value_of(vid)] = func(data, funcNum);
                                
                            // std::cout << " vertex_func_grad_map ID " << value_of(vid) <<  " f val " << vertex_func_grad_map[value_of(vid)][0].f_val_ << std::endl;
                            // std::cout << " vertex_func_grad_map ID " << value_of(vid) <<  " e val " << vertex_func_grad_map[value_of(vid)][0].e2_ << std::endl;
                        });
  

    auto comp = [](std::pair<mtet::Scalar, mtet::EdgeId> e0,
                   std::pair<mtet::Scalar, mtet::EdgeId> e1)
    { return e0.first < e1.first; };
    std::vector<std::pair<mtet::Scalar, mtet::EdgeId>> Q;

    Eigen::Matrix<double, 4, 3> pts;
    // std::array<llvm_vecsmall::SmallVector<Eigen::RowVectorXd, 20>,4> tet_info;

    std::array<llvm_vecsmall::SmallVector<CurvatureData<double>, 20>, 4> tet_info;
    
    for (size_t i = 0; i < tet_info.size(); i++){
        tet_info[i].resize(funcNum);
    }
    double activeTet = 0;
    
    {
        Timer timer(total_time, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        

        printf("start seq_foreach_vertex and push_longest_edge .... \n");
        // Initialize priority queue.
        grid.seq_foreach_tet([&](mtet::TetId tid, [[maybe_unused]] std::span<const mtet::VertexId, 4> vs)
                             { 
                              push_longest_edge(tid, grid, funcNum,  mode, crest_type, tet_max_edge_len, threshold,
        curve_network, pts,sub_call_two, sub_call_three, vertex_func_grad_map,
        Q, tet_active_map, tet_info, func, csg_func);
 });
        printf("finish seq_foreach_vertex and push_longest_edge .... \n");

        std::make_heap(Q.begin(), Q.end(), comp);
        std::cout << " q stack size " << Q.size() << std::endl;
        int loop_id = 0;
        // Keep splitting the longest edge
        while (!Q.empty())
        {
            // std::cout << " q stack loop count " << loop_id << std::endl;
            // loop_id ++;
            // if(loop_id > max_tet_num)
            // {
            //     break;
            // }
            std::pop_heap(Q.begin(), Q.end(), comp);
            auto [edge_length, eid] = Q.back();
            if (!grid.has_edge(eid)){
                Q.pop_back();
                continue;
            }
            //implement alpha value:
            mtet::Scalar comp_edge_length = alpha * edge_length;
            // std::cout << " comp_edge_length " << comp_edge_length << std::endl;
            bool addedActive = false;
            grid.foreach_tet_around_edge(eid,[&](mtet::TetId tid){
                std::span<VertexId, 4> vs = grid.get_tet(tid);
                if(/*vertex_active_map.contains(vertexHash(vs))*/tet_active_map.contains(vs)){
                    if (tet_active_map[vs]){
                        mtet::EdgeId longest_edge;
                        mtet::Scalar longest_edge_length = 0;
                        grid.foreach_edge_in_tet(tid, [&](mtet::EdgeId eid_active, mtet::VertexId v0, mtet::VertexId v1)
                                                 {
                            auto p0 = grid.get_vertex(v0);
                            auto p1 = grid.get_vertex(v1);
                            mtet::Scalar l = (p0[0] - p1[0]) * (p0[0] - p1[0]) + (p0[1] - p1[1]) * (p0[1] - p1[1]) +
                            (p0[2] - p1[2]) * (p0[2] - p1[2]);
                            if (l > longest_edge_length) {
                                longest_edge_length = l;
                                longest_edge = eid_active;
                            }
                        });
                        if (longest_edge_length > comp_edge_length) {
                            Q.emplace_back(longest_edge_length, longest_edge);
                            addedActive = true;
                            std::cout << "add tet longest edge length : " << longest_edge_length << std::endl;
                        }
                    }
                }
            });
            if(addedActive){
                std::push_heap(Q.begin(), Q.end(), comp);
                continue;
            }
            Q.pop_back();
            std::array<VertexId, 2> vs_old = grid.get_edge_vertices(eid);
            //Timer split_timer(splitting, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
            auto [vid, eid0, eid1] = grid.split_edge(eid);
            //split_timer.Stop();
            //std::cout << "Number of elements: " << mesh.get_num_tets() << std::endl;
            if (grid.get_num_tets() > max_elements) {
                break;
            }
            grid.foreach_tet_around_edge(eid0, [&](mtet::TetId tid)
            {
                
                if (push_longest_edge(tid, grid, funcNum,  mode,  crest_type, tet_max_edge_len, threshold,
        curve_network, pts,sub_call_two,sub_call_three, vertex_func_grad_map,
         Q,tet_active_map, tet_info, func, csg_func)) {
                    std::push_heap(Q.begin(), Q.end(), comp);
                } });
            grid.foreach_tet_around_edge(eid1, [&](mtet::TetId tid)
                                         {
                if (push_longest_edge(tid, grid, funcNum, mode, crest_type, tet_max_edge_len, threshold,
        curve_network, pts,sub_call_two,sub_call_three, vertex_func_grad_map,
         Q,tet_active_map, tet_info,func, csg_func)) {
                    std::push_heap(Q.begin(), Q.end(), comp);
                } });
        }
        timer.Stop();
    }
    std::cout << " start assign split tet values" << std::endl;
    std::vector<mtet::TetId> activeTetId;
    grid.seq_foreach_tet([&](mtet::TetId tid, std::span<const VertexId, 4> data) {
        std::span<VertexId, 4> vs = grid.get_tet(tid);
        std::array<std::valarray<double>,4> vallPoints;
        for (int i = 0; i < 4; i++){
            vallPoints[i] = {0.0,0.0,0.0};
        }
        // std::cout << " finish assign split tet values" << std::endl;
        for (int i = 0; i < 4; i++){
            VertexId vid = vs[i];
            std::span<Scalar, 3> coords = grid.get_vertex(vid);
            vallPoints[i][0] = coords[0];
            vallPoints[i][1] = coords[1];
            vallPoints[i][2] = coords[2];
        }
        // std::cout << " finish assign split tet point coordinates" << std::endl;
        double ratio = tet_radius_ratio(vallPoints);
        if (ratio < metric_list.min_radius_ratio){
            metric_list.min_radius_ratio = ratio;
        }
        // std::cout << " finish assign tet min_radius_ratio" << std::endl;
        if(tet_active_map.contains(vs)){
            if (tet_active_map[vs]){
                metric_list.active_tet++;
                activeTetId.push_back(tid);
                if (ratio < metric_list.active_radius_ratio){
                    metric_list.active_radius_ratio = ratio;
                }
            }
        }
        // std::cout << " finish assign tet active_radius_ratio" << std::endl;
    });
    metric_list.total_tet = grid.get_num_tets();
    metric_list.two_func_check = sub_call_two;
    metric_list.three_func_check = sub_call_three;
    metric_list.vertex_func_grad_map_ridge = vertex_func_grad_map;
    metric_list.activeTetId = activeTetId;
    //profiled time(see details in time.h) and profiled number of calls to zero
    std::cout << time_label[0] << ": " << profileTimer[0] << std::endl;
    return true;
}
