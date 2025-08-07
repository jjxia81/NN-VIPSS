//
//  subdivide_multi.cpp
//  adaptive_mesh_refinement
//
//  Created by Yiwen Ju on 6/20/24.
//

#include "refine_crit.h"

///Stores the 2d and 3d origin for the convex hull check happened in zero-crossing criteria.
std::array<double, 2> query_2d = {0.0, 0.0}; // X, Y
std::array<double, 3> query_3d = {0.0, 0.0, 0.0}; // X, Y, Z

/// Below are the local functions servicing `critIA` , `critCSG`, and `critMI`

/// returns a `bool` value that `true` represents positive and `false` represents negative of the input value `x`.
bool get_sign(double x) {
    return x > 0;
}

/// Construct the values of one function at the bezier control points within a tet.
///
/// @param[in] vals         The function values at four tet vertices
/// @param[in] grads            The total derivative of the functions in x, y, z direction at four tet vertices.
/// @param[in] vec          sampled vectors using four tet vertices. Given tet vertices as p0, p1, p2, p3. These 6 vectors are p1 - p0, p2 - p0, p3 - p0, p2 - p1, p3 - p1, p3 - p2.
///
/// @return         The eigen vector of 20 bezier values.
Eigen::Vector<double, 20> bezierConstruct(const Eigen::RowVector4d vals,
                                                const Eigen::Matrix<double, 4, 3> grads,
                                                const Eigen::Matrix<double, 3, 6> vec)
{
    Eigen::RowVector3d v0s, v1s, v2s, v3s;
    v0s = grads.row(0) * vec(Eigen::all, {0, 1, 2}) / 3;
    v0s.array() += vals(0);
    v1s =  grads.row(1) * vec(Eigen::all, {3, 4, 0}) / 3;
    v1s = v1s.asDiagonal() * Eigen::Vector3d({1, 1, -1});
    v1s.array() += vals(1);
    v2s = grads.row(2) * vec(Eigen::all, {5, 1, 3}) / 3;
    v2s = v2s.asDiagonal() * Eigen::Vector3d({1, -1, -1});
    v2s.array() += vals(2);
    v3s = grads.row(3) * vec(Eigen::all, {2, 4, 5}) / 3;
    v3s *= -1;
    v3s.array() += vals(3);
    
    
    double vMid0 = (9 * (v1s(0) + v1s(1) + v2s(0) + v2s(2) + v3s(1) + v3s(2)) / 6 - vals(1) - vals(2) - vals(3))/ 6;
    double vMid1 =(9 * (v0s[1] + v0s[2] + v2s[0] + v2s[1] + v3s[0] + v3s[2]) / 6 - vals(0) - vals(2) - vals(3))/ 6;
    double vMid2 =(9 * (v0s[0] + v0s[2] + v1s[1] + v1s[2] + v3s[0] + v3s[1]) / 6 - vals(0) - vals(1) - vals(3))/ 6;
    double vMid3 =(9 * (v0s[0] + v0s[1] + v1s[0] + v1s[2] + v2s[1] + v2s[2]) / 6 - vals(0) - vals(1) - vals(2))/ 6;
    Eigen::RowVector<double, 20> valList;
    valList << vals, v0s, v1s, v2s, v3s, vMid0, vMid1, vMid2, vMid3;
    return valList;
}

/// Construct the value differences between linear interpolations and bezier approximations at 16 bezier control points (excluding control points at tet vertices)
/// @param[in] valList          The eigen vector of 20 bezier values.
///
/// @return         The value differences at 16 control points.
Eigen::Vector<double, 16> bezierDiff(const Eigen::Vector<double,20> valList)
{
    /// Constant coefficient to obtain linear interpolated values at each bezier control points
    const Eigen::Matrix<double, 16, 4> linear_coeff {{2, 1, 0, 0}, {2, 0, 1, 0}, {2, 0, 0, 1}, {0, 2, 1, 0},{0, 2, 0, 1}, {1, 2, 0, 0}, {0, 0, 2, 1}, {1, 0, 2, 0},{0, 1, 2, 0}, {1, 0, 0, 2}, {0, 1, 0, 2}, {0, 0, 1, 2},{0, 1, 1, 1}, {1, 0, 1, 1}, {1, 1, 0, 1}, {1, 1, 1, 0}};
    Eigen::Vector<double, 16> linear_val = (linear_coeff * valList.head(4)) / 3;
    return valList.tail(16) - linear_val;
}

/// The check whether the two functions' intersection curve lies in the tet.
/// Transforms values at 20 bezier control points for two functions into the correct format that `convex_hull_membership` library can use.
/// @param[in] valList          The eigen vector of 20 bezier values.
///
/// @return A array that convexhull memship library can use.
std::array<double, 40> parse_convex_points2d(const Eigen::Matrix<double, 2, 20> valList) {
    std::array<double, 40> transposed;
    Eigen::MatrixXd::Map(transposed.data(), 2, 20) = valList;
    return transposed;
}

/// The check whether the three functions' intersection point lies in the tet.
/// Transforms values at 20 bezier control points for three functions into the correct format that `convex_hull_membership` library can use.
/// @param[in] valList          The eigen vector of 20 bezier values.
///
/// @return A array that convexhull memship library can use.
std::array<double, 60> parse_convex_points3d(const Eigen::Matrix<double, 3, 20> valList) {
    std::array<double, 60> transposed;
    Eigen::MatrixXd::Map(transposed.data(), 3, 20) = valList;
    return transposed;
}

/// Given two functions, here is the check whether the two functions' intersection curve can be well approximated by linear interpolation.
/// @param[in] grad         The linear interpolations' gradients of these two functions within the tet.
/// @param[in] diff_matrix          The difference between linear interpolations and bezier approximations at 16 bezier control points (excluding control points at tet vertices) for these two functions
/// @param[in] sqD          The squared determinant to offset the un-normalized gradients
/// @param[in] threshold            The user-defined error threshold
///
/// @return         Whether the tet passes the check for these two functions.
bool two_func_check (Eigen::Matrix<double, 2, 3> grad,
                     const Eigen::Matrix<double, 16, 2> diff_matrix,
                     const double sqD,
                     const double threshold)
{
    Eigen::Matrix2d w;
    w << grad.row(0).squaredNorm(), grad.row(0).dot(grad.row(1)),
    grad.row(0).dot(grad.row(1)), grad.row(1).squaredNorm();
    double E = w.determinant();
    Eigen::Matrix<double, 2, 3> H = Eigen::Matrix2d({{w(1, 1), -w(1, 0)}, {-w(0, 1), w(0,0)}}) * grad;
    
    //find the largest max error (max squared gamma: the LHS of the equation) among all 16 bezier control points
    Eigen::Matrix<double, 16, 3> unNormDis = diff_matrix * H;
    Eigen::Vector<double, 16> dotProducts = sqD * unNormDis.cwiseProduct(unNormDis).rowwise().sum();
    return (dotProducts.maxCoeff() > threshold*threshold * E * E);
}

/// Given two functions, here is the check whether the three functions' intersection curve can be well approximated by linear interpolation.
/// @param[in] grad         The linear interpolations' gradients of these three functions within the tet.
/// @param[in] diff_matrix          The difference between linear interpolations and bezier approximations at 16 bezier control points (excluding control points at tet vertices) for these two functions
/// @param[in] sqD          The squared determinant to offset the un-normalized gradients
/// @param[in] threshold            The user-defined error threshold
///
/// @return         Whether the tet passes the check for these three functions.
bool three_func_check (Eigen::Matrix<double, 3, 3> grad,
                     const Eigen::Matrix<double, 16, 3> diff_matrix,
                     const double sqD,
                     const double threshold)
{
    double E = grad.determinant();
    Eigen::Matrix<double, 3, 3> H;
    H << grad.row(1).cross(grad.row(2)),
    grad.row(2).cross(grad.row(0)),
    grad.row(0).cross(grad.row(1));
    Eigen::Matrix<double, 16, 3> unNormDis_eigen = diff_matrix * H;
    Eigen::Vector<double, 16> dotProducts = sqD * unNormDis_eigen.cwiseProduct(unNormDis_eigen).rowwise().sum();
    //double maxGammaSq = dotProducts.maxCoeff();
    return (dotProducts.maxCoeff() > threshold*threshold * E * E);
}

bool critIA(
            const Eigen::Matrix<double, 4, 3> &pts,
            const std::array<llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20>,4> tet_info,
            const size_t funcNum,
            const double threshold,
            const bool curve_network,
            bool& active,
            int &sub_call_two,
            int &sub_call_three)
{
    Eigen::Matrix<double, Eigen::Dynamic, 20> valList (funcNum, 20);
    Eigen::Matrix<double, Eigen::Dynamic, 16> diffList(funcNum, 16);
    llvm_vecsmall::SmallVector<bool, 20> activeTF(funcNum);
    Eigen::Matrix<double, 20, 3> gradList;
    Eigen::Vector3d eigenVec1 = pts.row(1) - pts.row(0), eigenVec2 = pts.row(2) - pts.row(0), eigenVec3 = pts.row(3) - pts.row(0), eigenVec4 = pts.row(2) - pts.row(1), eigenVec5 = pts.row(3) - pts.row(1), eigenVec6 = pts.row(3) - pts.row(2);
    Eigen::Matrix<double, 3, 6> vec;
    vec << eigenVec1, eigenVec2, eigenVec3, eigenVec4, eigenVec5, eigenVec6;
    double D = vec.leftCols(3).determinant();
    double sqD = D*D;
    Eigen::Matrix3d crossMatrix;
    crossMatrix << eigenVec2.cross(eigenVec3), eigenVec3.cross(eigenVec1), eigenVec1.cross(eigenVec2);
    
    int activeNum = 0;
    //single function linearity check:
    for (int funcIter = 0; funcIter < funcNum; funcIter++){
        //Timer single_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        //storing bezier and linear info for later linearity comparison
        Eigen::Matrix4d func_info;
        func_info << tet_info[0][funcIter], tet_info[1][funcIter], tet_info[2][funcIter], tet_info[3][funcIter];
        Eigen::RowVector4d vals = func_info.col(0);
        Eigen::Matrix<double, 4, 3> grads_eigen = func_info.rightCols(3);
        valList.row(funcIter) = bezierConstruct(vals, grads_eigen, vec);
        //Timer single_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        activeTF[funcIter] = get_sign(valList.row(funcIter).maxCoeff()) != get_sign(valList.row(funcIter).minCoeff());
        //single_timer.Stop();
        if (activeTF[funcIter]){
            if (!active){
                active = true;
            }
            activeNum++;
            Eigen::Vector3d unNormF = Eigen::RowVector3d(vals(1)-vals(0), vals(2)-vals(0), vals(3)-vals(0)) * crossMatrix.transpose();
            gradList.row(funcIter) = unNormF;
            diffList.row(funcIter) = bezierDiff(valList.row(funcIter));
            double error = std::max(diffList.row(funcIter).maxCoeff(), -diffList.row(funcIter).minCoeff());
            //Timer single2_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
            double lhs = error * error * sqD;
            double rhs;
            if (!curve_network){
                rhs = threshold * threshold * gradList.row(funcIter).squaredNorm();
            }else{
                rhs = std::numeric_limits<double>::infinity() * gradList.row(funcIter).squaredNorm();
            }
            if (lhs > rhs) {
                //single2_timer.Stop();
                return true;
            }
            //single2_timer.Stop();
        }
    }
    //Timer single_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
    if(activeNum < 2){
        //single_timer.Stop();
        return false;
    }
    llvm_vecsmall::SmallVector<int, 20> activeFunc(activeNum);
    int activeFuncIter = 0;
    for (int funcIter = 0; funcIter < funcNum; funcIter++){
        if (activeTF[funcIter]){
            activeFunc[activeFuncIter] = funcIter;
            activeFuncIter++;
        }
    }
    llvm_vecsmall::SmallVector<llvm_vecsmall::SmallVector<bool, 20>, 20> zeroXResult(funcNum, llvm_vecsmall::SmallVector<bool, 20>(funcNum));
    //single_timer.Stop();
    const int pairNum = activeNum * (activeNum-1)/2, triNum = activeNum * (activeNum-1) * (activeNum - 2)/ 6;
    
    // 2-function checks
    int activeDouble_count = 0;
    {
        //Timer timer(twoFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        bool zeroX;
        
        for (int i = 0; i < activeNum - 1; i++){
            for (int j = i + 1; j < activeNum; j++){
                std::array<int, 2> pairIndices = {activeFunc[i], activeFunc[j]};
                std::array<double, 40> nPoints = parse_convex_points2d(valList(pairIndices, Eigen::all));
                //Timer sub_timer(sub_twoFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
                zeroX = convex_hull_membership::contains<2, double>(nPoints, query_2d);
                //sub_timer.Stop();
                
                if (zeroX){
                    activeDouble_count++;
                    sub_call_two ++;
                    zeroXResult[pairIndices[0]][pairIndices[1]] = true;
                    zeroXResult[pairIndices[1]][pairIndices[0]] = true;
                    Eigen::Matrix<double, 2, 3> grad = gradList(pairIndices, Eigen::all);
                    Eigen::Matrix<double, 16, 2> diff_matrix = diffList(pairIndices, Eigen::all).transpose();
                    // two function linearity test:
                    if (two_func_check (grad, diff_matrix, sqD, threshold)){
                        //timer.Stop();
                        return true;
                    }
                }
            }
        }
        //timer.Stop();
    }
    if(activeDouble_count < 3)
        return false;
    // 3-function checks
    {
        //Timer timer(threeFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        bool zeroX;
        for (int i = 0; i < activeNum - 2; i++){
            for (int j = i + 1; j < activeNum - 1; j++){
                for (int k = j + 1; k < activeNum; k++){
                    std::array<int, 3> tripleIndices = {activeFunc[i], activeFunc[j], activeFunc[k]};
                    if(!(zeroXResult[tripleIndices[0]][tripleIndices[1]]&&zeroXResult[tripleIndices[0]][tripleIndices[2]]&&zeroXResult[tripleIndices[1]][tripleIndices[2]]))
                        continue;
                    std::array<double, 60> nPoints = parse_convex_points3d(valList(tripleIndices, Eigen::all));
                    sub_call_three ++;
                    //Timer sub_timer(sub_threeFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
                    zeroX = convex_hull_membership::contains<3, double>(nPoints, query_3d);
                    //sub_timer.Stop();
                    
                    if (zeroX){
                        Eigen::Matrix<double, 3, 3> grad = gradList(tripleIndices, Eigen::all);
                        Eigen::Matrix<double, 16, 3> diff_matrix = diffList(tripleIndices, Eigen::all).transpose();
                        if (three_func_check (grad, diff_matrix, sqD, threshold)){
                            //timer.Stop();
                            return true;
                        }
                    }
                }
            }
        }
        //timer.Stop();
    }
    return false;
}

bool critCSG(
             const Eigen::Matrix<double, 4, 3> &pts,
             const std::array<llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20>,4> tet_info,
             const size_t funcNum,
             const std::function<std::pair<std::array<double, 2>, llvm_vecsmall::SmallVector<int, 20>>(llvm_vecsmall::SmallVector<std::array<double, 2>, 20>)> csg_func,
             const double threshold,
             const bool curve_network,
             bool& active,
             int &sub_call_two,
             int &sub_call_three)
{
    Eigen::Matrix<double, Eigen::Dynamic, 20> valList (funcNum, 20);
    Eigen::Matrix<double, Eigen::Dynamic, 16> diffList(funcNum, 16);
    llvm_vecsmall::SmallVector<bool, 20> activeTF(funcNum);
    llvm_vecsmall::SmallVector<std::array<double , 2>, 20> funcInt(funcNum);
    Eigen::Matrix<double, 20, 3> gradList;
    Eigen::Vector3d eigenVec1 = pts.row(1) - pts.row(0), eigenVec2 = pts.row(2) - pts.row(0), eigenVec3 = pts.row(3) - pts.row(0), eigenVec4 = pts.row(2) - pts.row(1), eigenVec5 = pts.row(3) - pts.row(1), eigenVec6 = pts.row(3) - pts.row(2);
    Eigen::Matrix<double, 3, 6> vec;
    vec << eigenVec1, eigenVec2, eigenVec3, eigenVec4, eigenVec5, eigenVec6;
    double D = vec.leftCols(3).determinant();
    double sqD = D*D;
    Eigen::Matrix3d crossMatrix;
    crossMatrix << eigenVec2.cross(eigenVec3), eigenVec3.cross(eigenVec1), eigenVec1.cross(eigenVec2);
    
    int activeNum = 0;
    //single function linearity check:
    for (int funcIter = 0; funcIter < funcNum; funcIter++){
        //Timer single_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        Eigen::Matrix4d func_info;
        func_info << tet_info[0][funcIter], tet_info[1][funcIter], tet_info[2][funcIter], tet_info[3][funcIter];
        Eigen::RowVector4d vals = func_info.col(0);
        Eigen::Matrix<double, 4, 3> grads_eigen = func_info.rightCols(3);
        valList.row(funcIter) = bezierConstruct(vals, grads_eigen, vec);
        funcInt[funcIter] = {valList.row(funcIter).minCoeff(), valList.row(funcIter).maxCoeff()};
    }
    
        std::pair<std::array<double, 2>, llvm_vecsmall::SmallVector<int, 20>> csgResult = csg_func(funcInt);
        if(csgResult.first[0] * csgResult.first[1] > 0){
            return false;
        }else{
            for (size_t funcIter = 0; funcIter < funcNum; funcIter++){
                //Timer single_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
                activeTF[funcIter] = !csgResult.second[funcIter];
                //single_timer.Stop();
                if (activeTF[funcIter]){
                    if (!active){
                        active = true;
                    }
                    activeNum++;
                    double v0 = tet_info[0][funcIter][0], v1 = tet_info[1][funcIter][0], v2 = tet_info[2][funcIter][0], v3 = tet_info[3][funcIter][0];
                    Eigen::Vector3d unNormF = Eigen::RowVector3d(v1-v0, v2-v0, v3-v0) * crossMatrix.transpose();
                    gradList.row(funcIter) = unNormF;
                    diffList.row(funcIter) = bezierDiff(valList.row(funcIter));
                    double error = std::max(diffList.row(funcIter).maxCoeff(), -diffList.row(funcIter).minCoeff());
                    //Timer single2_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
                    double lhs = error * error * sqD;
                    double rhs;
                    if (!curve_network){
                        rhs = threshold * threshold * gradList.row(funcIter).squaredNorm();
                    }else{
                        rhs = std::numeric_limits<double>::infinity() * gradList.row(funcIter).squaredNorm();
                    }
                    if (lhs > rhs) {
                        //single2_timer.Stop();
                        return true;
                    }
                    //single2_timer.Stop();
                }
            }
        }
    //Timer single_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
    if(activeNum < 2){
        //single_timer.Stop();
        return false;
    }
    llvm_vecsmall::SmallVector<int, 20> activeFunc(activeNum);
    int activeFuncIter = 0;
    for (int funcIter = 0; funcIter < funcNum; funcIter++){
        if (activeTF[funcIter]){
            activeFunc[activeFuncIter] = funcIter;
            activeFuncIter++;
        }
    }
    llvm_vecsmall::SmallVector<llvm_vecsmall::SmallVector<bool, 20>, 20> zeroXResult(funcNum, llvm_vecsmall::SmallVector<bool, 20>(funcNum));
    //single_timer.Stop();
    const int pairNum = activeNum * (activeNum-1)/2, triNum = activeNum * (activeNum-1) * (activeNum - 2)/ 6;
    
    // 2-function checks
    int activeDouble_count = 0;
    {
        //Timer timer(twoFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        bool zeroX;
        for (int i = 0; i < activeNum - 1; i++){
            for (int j = i + 1; j < activeNum; j++){
                std::array<int, 2> pairIndices = {activeFunc[i], activeFunc[j]};
                std::array<double, 40> nPoints = parse_convex_points2d(valList(pairIndices, Eigen::all));
                //Timer sub_timer(sub_twoFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
                zeroX = convex_hull_membership::contains<2, double>(nPoints, query_2d);
                //sub_timer.Stop();
                
                if (zeroX){
                    activeDouble_count++;
                    sub_call_two ++;
                    zeroXResult[pairIndices[0]][pairIndices[1]] = true;
                    zeroXResult[pairIndices[1]][pairIndices[0]] = true;
                    Eigen::Matrix<double, 2, 3> grad = gradList(pairIndices, Eigen::all);
                    Eigen::Matrix<double, 16, 2> diff_matrix = diffList(pairIndices, Eigen::all).transpose();
                    // two function linearity test:
                    if (two_func_check (grad, diff_matrix, sqD, threshold)){
                        //timer.Stop();
                        return true;
                    }
                }
            }
        }
        //timer.Stop();
    }
    if(activeDouble_count < 3)
        return false;
    // 3-function checks
    {
        //Timer timer(threeFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        bool zeroX;
        for (int i = 0; i < activeNum - 2; i++){
            for (int j = i + 1; j < activeNum - 1; j++){
                for (int k = j + 1; k < activeNum; k++){
                    std::array<int, 3> tripleIndices = {activeFunc[i], activeFunc[j], activeFunc[k]};
                    if(!(zeroXResult[tripleIndices[0]][tripleIndices[1]]&&zeroXResult[tripleIndices[0]][tripleIndices[2]]&&zeroXResult[tripleIndices[1]][tripleIndices[2]]))
                        continue;
                    std::array<double, 60> nPoints = parse_convex_points3d(valList(tripleIndices, Eigen::all));
                    sub_call_three ++;
                    //Timer sub_timer(sub_threeFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
                    zeroX = convex_hull_membership::contains<3, double>(nPoints, query_3d);
                    //sub_timer.Stop();
                    
                    if (zeroX){
                        Eigen::Matrix<double, 3, 3> grad = gradList(tripleIndices, Eigen::all);
                        Eigen::Matrix<double, 16, 3> diff_matrix = diffList(tripleIndices, Eigen::all).transpose();
                        if (three_func_check (grad, diff_matrix, sqD, threshold)){
                            //timer.Stop();
                            return true;
                        }
                    }
                }
            }
        }
        //timer.Stop();
    }
    return false;
}
bool critMI(
            const Eigen::Matrix<double, 4, 3> &pts,
            const std::array<llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20>,4> tet_info,
            const size_t funcNum,
            const double threshold,
            const bool curve_network,
            bool& active,
            int &sub_call_two,
            int &sub_call_three)
{
    Eigen::Vector3d eigenVec1 = pts.row(1) - pts.row(0), eigenVec2 = pts.row(2) - pts.row(0), eigenVec3 = pts.row(3) - pts.row(0), eigenVec4 = pts.row(2) - pts.row(1), eigenVec5 = pts.row(3) - pts.row(1), eigenVec6 = pts.row(3) - pts.row(2);
    Eigen::Matrix<double, 3, 6> vec;
    vec << eigenVec1, eigenVec2, eigenVec3, eigenVec4, eigenVec5, eigenVec6;
    double D = vec.leftCols(3).determinant();
    double sqD = D*D;
    Eigen::Matrix3d crossMatrix_eigen;
    crossMatrix_eigen << eigenVec2.cross(eigenVec3), eigenVec3.cross(eigenVec1), eigenVec1.cross(eigenVec2);
    Eigen::Matrix<double, 20, 3> gradList_eigen;
    Eigen::Matrix<double, Eigen::Dynamic, 20> valList (funcNum, 20);
    Eigen::Matrix<double, Eigen::Dynamic, 16> diffList(funcNum, 16);
    
    llvm_vecsmall::SmallVector<bool, 20> activeList(funcNum);
    llvm_vecsmall::SmallVector<std::array<double , 2>, 20> funcInt(funcNum);
    double maxLow = -1 * std::numeric_limits<double>::infinity();
    llvm_vecsmall::SmallVector<llvm_vecsmall::SmallVector<bool, 20>, 20> activePair(funcNum, llvm_vecsmall::SmallVector<bool, 20>(funcNum, false));
    //single function linearity check:
    for (int funcIter = 0; funcIter < funcNum; funcIter++){
        Eigen::Matrix4d func_info;
        func_info << tet_info[0][funcIter], tet_info[1][funcIter], tet_info[2][funcIter], tet_info[3][funcIter];
        Eigen::RowVector4d vals = func_info.col(0);
        Eigen::Matrix<double, 4, 3> grads_eigen = func_info.rightCols(3);
        valList.row(funcIter) = bezierConstruct(vals, grads_eigen, vec);
        funcInt[funcIter] = {valList.row(funcIter).minCoeff(), valList.row(funcIter).maxCoeff()};
        if (maxLow < funcInt[funcIter][0]){
            maxLow = funcInt[funcIter][0];
        }
    }
    llvm_vecsmall::SmallVector<int, 20> activeFunc;
    for (int funcIter = 0; funcIter < funcNum; funcIter++){
        if(funcInt[funcIter][1] > maxLow){
            activeFunc.push_back(funcIter);
        }
    }
    size_t activeNum = activeFunc.size();
    if(activeNum < 2)
        return false;

    //Timer get_func_timer(getActiveMuti, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
    const size_t pairNum = activeNum * (activeNum-1)/2, triNum = activeNum * (activeNum-1) * (activeNum - 2)/ 6, quadNum = activeNum * (activeNum - 1) * (activeNum - 2) * (activeNum - 3)/ 24;
//    get_func_timer.Stop();
    
    for (int i = 0; i < activeNum - 1; i++){
        for (int j = i + 1; j < activeNum; j++){
            //Timer single_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
            int funcIndex1 = activeFunc[i];
            int funcIndex2 = activeFunc[j];
            Eigen::Vector<double, 20> diff_at_point;
            diff_at_point = valList.row(funcIndex2) - valList.row(funcIndex1);
            bool activeTF = get_sign(diff_at_point.maxCoeff()) == get_sign(diff_at_point.minCoeff()) ? false : true;
            //single_timer.Stop();
            if (activeTF){
                if (!active){
                    active = true;
                }
                activePair[funcIndex1][funcIndex2] = true;
                activePair[funcIndex2][funcIndex1] = true;
                if (!activeList[funcIndex1]){
                    activeList[funcIndex1] = true;
                    double v0 = valList(funcIndex1, 0), v1 = valList(funcIndex1, 1), v2 = valList(funcIndex1, 2), v3 = valList(funcIndex1, 3);
                    Eigen::Vector3d unNormF_eigen = Eigen::RowVector3d(v1-v0, v2-v0, v3-v0) * crossMatrix_eigen.transpose();
                    gradList_eigen.row(funcIndex1) = unNormF_eigen;
                    
                    diffList.row(funcIndex1) = bezierDiff(valList.row(funcIndex1));
                }
                if (!activeList[funcIndex2]){
                    activeList[funcIndex2] = true;
                    double v0 = valList(funcIndex2, 0), v1 = valList(funcIndex2, 1), v2 = valList(funcIndex2, 2), v3 = valList(funcIndex2, 3);
                    Eigen::Vector3d unNormF_eigen = Eigen::RowVector3d(v1-v0, v2-v0, v3-v0) * crossMatrix_eigen.transpose();
                    gradList_eigen.row(funcIndex2) = unNormF_eigen;
                    diffList.row(funcIndex2) = bezierDiff(valList.row(funcIndex2));
                    
                }
                Eigen::Vector<double, 16> diff_twofunc;
                diff_twofunc = diffList.row(funcIndex1) - diffList.row(funcIndex2);
                //Timer single2_timer(singleFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
                double error = std::max(diff_twofunc.maxCoeff(), -diff_twofunc.minCoeff());
                Eigen::Vector3d grad_eigen;
                grad_eigen = gradList_eigen.row(funcIndex1) - gradList_eigen.row(funcIndex2);
                double lhs = error * error * sqD;
                double rhs;
                if (!curve_network){
                    rhs = threshold * threshold * grad_eigen.squaredNorm();
                }else{
                    rhs = std::numeric_limits<double>::infinity() * grad_eigen.squaredNorm();
                }
                if (lhs > rhs) {
                    //single2_timer.Stop();
                    return true;
                }
                //single2_timer.Stop();
            }
        }
    }
    
    // 2-function checks
    int activeTriple_count = 0;
    {
        //Timer timer(twoFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        bool zeroX;
        for (int i = 0; i < activeNum - 2; i++){
            for (int j = i + 1; j < activeNum - 1; j++){
                for (int k = j + 1; k < activeNum; k++){
                    int funcIndex1 = activeFunc[i];
                    int funcIndex2 = activeFunc[j];
                    int funcIndex3 = activeFunc[k];
                    if(!(activePair[funcIndex1][funcIndex2]&&activePair[funcIndex1][funcIndex3]&&activePair[funcIndex2][funcIndex3]))
                        continue;
                    Eigen::Matrix<double,2, 20> diff_mi(2, 20);
                    diff_mi.row(0) = valList.row(funcIndex1) - valList.row(funcIndex2);
                    diff_mi.row(1) =  valList.row(funcIndex2) - valList.row(funcIndex3);
                    std::array<double, 40> nPoints = parse_convex_points2d(diff_mi);
                    //Timer sub_timer(sub_twoFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
                    zeroX = convex_hull_membership::contains<2, double>(nPoints, query_2d);
                    //sub_timer.Stop();
                    if (zeroX){
                        sub_call_two ++;
                        activeTriple_count++;
                        Eigen::Matrix<double, 2, 3> grad(2, 3);
                        grad.row(0) = gradList_eigen.row(funcIndex1) - gradList_eigen.row(funcIndex2);
                        grad.row(1) = gradList_eigen.row(funcIndex2) - gradList_eigen.row(funcIndex3);
                        Eigen::Matrix<double, 2, 16> diff_matrix(2, 16);
                        diff_matrix.row(0) = diffList.row(funcIndex1) - diffList.row(funcIndex2);
                        diff_matrix.row(1) = diffList.row(funcIndex2) - diffList.row(funcIndex3);
                        if (two_func_check (grad, diff_matrix.transpose(), sqD, threshold)){
                            //timer.Stop();
                            return true;
                        }
                    }
                }
            }
        }
        //timer.Stop();
    }
    if(activeTriple_count < 4)
        return false;
    {
        //Timer timer(threeFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
        bool zeroX;
        for (int i = 0; i < activeNum - 3; i++){
            for (int j = i + 1; j < activeNum - 2; j++){
                for (int k = j + 1; k < activeNum - 1; k++){
                    for (int m = k + 1; m < activeNum; m++){
                        int funcIndex1 = activeFunc[i];
                        int funcIndex2 = activeFunc[j];
                        int funcIndex3 = activeFunc[k];
                        int funcIndex4 = activeFunc[m];
                        if(!(activePair[funcIndex1][funcIndex2]&&activePair[funcIndex1][funcIndex3]&&activePair[funcIndex1][funcIndex4]&&activePair[funcIndex2][funcIndex3]&&activePair[funcIndex2][funcIndex4]&&activePair[funcIndex3][funcIndex4]))
                            continue;
                        
                        Eigen::Matrix<double,3, 20> diff_mi(3, 20);
                        diff_mi.row(0) = valList.row(funcIndex1) - valList.row(funcIndex2);
                        diff_mi.row(1) =  valList.row(funcIndex2) - valList.row(funcIndex3);
                        diff_mi.row(2) =  valList.row(funcIndex3) - valList.row(funcIndex4);
                        std::array<double, 60> nPoints = parse_convex_points3d(diff_mi);
                        //Timer sub_timer(sub_twoFunc, [&](auto profileResult){profileTimer = combine_timer(profileTimer, profileResult);});
                        zeroX = convex_hull_membership::contains<3, double>(nPoints, query_3d);
                        //sub_timer.Stop();
                        
                        if (zeroX){
                            sub_call_three ++;
                            Eigen::Matrix<double, 3, 3> grad(3, 3);
                            grad.row(0) = gradList_eigen.row(funcIndex1) - gradList_eigen.row(funcIndex2);
                            grad.row(1) = gradList_eigen.row(funcIndex2) - gradList_eigen.row(funcIndex3);
                            grad.row(2) = gradList_eigen.row(funcIndex3) - gradList_eigen.row(funcIndex4);
                            Eigen::Matrix<double, 3, 16> diff_matrix(3, 16);
                            diff_matrix.row(0) = diffList.row(funcIndex1) - diffList.row(funcIndex2);
                            diff_matrix.row(1) = diffList.row(funcIndex2) - diffList.row(funcIndex3);
                            diff_matrix.row(2) = diffList.row(funcIndex3) - diffList.row(funcIndex4);
                            if (three_func_check (grad, diff_matrix.transpose(), sqD, threshold)){
                                //timer.Stop();
                                return true;
                            }
                        }
                    }
                }
            }
        }
        //timer.Stop();
    }
    return false;
}


