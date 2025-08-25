NN-VIPSS - Jianjun Xia (2025)

------------------------------------

This code implements the algorithm described in

  [**Variational Surface Reconstruction Using Natural Neighbors**](https://jjxia81.github.io/NN-VIPSS-Project/) 
   Jianjun Xia, Tao Ju.  
   *ACM Transactions on Graphics (Proc. ACM Siggraph 2025)*  

The primary function of this program is to predict the normal as well as the underlying surface of a given set of unoriented points.

Currently, the code is tested on Ubuntu OS 24.04, it should work at other platforms with minor modification. If you are not familiar with compilation on windows, we provide a pre-compiled [binary files](https://gowustl-my.sharepoint.com/:u:/g/personal/jianjun_x_wustl_edu/Ea108LLPWkJJrr3xBJBVrckB4ipwM4qRdXkxsJRkugdTmQ?e=cJ3dQI) for you to try. 

To build on MacOS, a [fork](https://github.com/yig/NN-VIPSS) by Yotam Gingold might be helpful.

BUILDING
======================================================================================================


The code need dependencies: 
1) [Armadillo](https://gitlab.com/conradsnicta/armadillo-code)  
2) NLOPT  
3) [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) (need to enable OpenMP support, better recompile it when using Ubuntu)  
4) Eigen  
5) [CLI11](https://github.com/CLIUtils/CLI11)  
6) [pico_tree v0.8.3](https://github.com/Jaybro/pico_tree/tree/v0.8.3)  
7) [C++ JSON (nlohmann/json)](https://github.com/nlohmann/json)

Install Openblass from source with OpenMP enabled first, then Amardillo and other libs. If using Windows, switch windows branch first and vcpkg is preferred to install the libs except OpenBlas, as you need to recompile the lib with USE_OPENMP on; pico header files are already in the source code. When compiling with Visual Studio, please set C++ version to 20+ and enable OpenMP support.   



<!-- You can download & install them by yourself, or run the env.sh script which will install homebrew first.
$source env.sh   -->

Then go to the vipss folder, build the Cmake file and make:
$mkdir ./vipss/build  
$cd ./vipss/build  
$cmake ..  
$make  

In the build directory, there should be an executable called "nnvipss" (or "nnvipss.exe" on Windows if it is successfully built).


RUNNING
======================================================================================================

To run the code from the command line, type:

$./nnvipss -i input_file_name [-l user_lambda] [-o output_file_path]

where:
1. -i: followed by the path of the input file. input_file_name is a path to the input file. currently, support file format includes ".xyz" and ".ply". The format of .xyz is:  

2. -o: optional argument. Followed by the path of the output path. output_file_path is a path to the folder for generating output files. Default the folder of the input file.

3. -l: optional argument. Followed by a float number indicating the lambda which balances the energy (see the paper for details). Default 0 (exact interpolation), you should set and tune this number according to your inputs.

4. -s: optional bool argument(false or true, true for default). If true, the program would output the surface ([input file name].ply). 

5. --alpha : optional argument. Followed by a float number indicating the soft constraints level of the gradient length. Larger value means stronger constraints, the default value is 50, you may finetune if needed.   

6. --large_input : optional bool argument(false or true, false for default). For test on dense sampling cases, setting the option true, can reduce the average nature neighbor size, which is useful when we are doing the test on the large benchmark described in the paper.    

7. --max_iter: Followed by the a integer number, the default value 10000. You can finetune the number if you want early stop or more iteration for smoother results.

8. -a : [apative grid surface](https://jurwen.github.io/Adaptive-grid-for-implicit-complexes/) threshold value for surface generation, default value is 0.003.  

9. -w : MST weight type, 0 for angle score, 1 for a conbined weight(sqrt(dist) * score), default is 0. You may try 1 if the init normals are not correctly flipped for some tricky cases, especially when the there are large missing part on dense sampling cases.  
 

Some examples have been placed at data folder for testing:
$./nnvipss -i ../../data/points/doghead.xyz -o {your_own_out_dir}/doghead.ply 

The program will generate the predicted normal in [input file name]_normal.ply.
If -s is included in the command line, the program will generate the surface as the zero-level set of the solved implicit function ([input file name]_surface.ply).

:bell: To generate all the example in the paper, please run the makefigure.sh script in the vipss folder:  
check makefigure.sh file, and replace the out_dir with your output directory.  

:mega: For further questions about the code and the paper, please contact Jianjun Xia  jianjun.x@wustl.edu (might be invalid after he graduated). You can also contact Prof. Tao Ju at taoju@wustl.edu.



