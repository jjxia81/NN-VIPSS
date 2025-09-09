
cmake_minimum_required(VERSION 3.10)
project(vipss)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a -O3 ")

# check_library_exists(atomic __atomic_compare_exchange_16 "" NEEDS_LIBATOMIC)
find_package(Eigen3 REQUIRED)
find_package(NLopt REQUIRED)
find_package(PicoTree REQUIRED)
find_package(CLI11)

# set(NLopt_LIB_DIR "")
# set(NLopt_LIB ${NLOPT_LIBRARIES})
# add_definitions(-w)
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(OpenBLAS )
# set(ARMADILLO_LIB_DIRS "")
set(ARMADILLO_LIB ${ARMADILLO_LIBRARIES})
include_directories( ${EIGEN3_INCLUDE_DIR})


find_package(nlohmann_json 3.2.0 REQUIRED)
# find_library(ATOMIC_OPS_LIB atomic_ops REQUIRED)
# aux_source_directory(./adgrid ADGRID_LIST)
# aux_source_directory(./adgrid/nanothread NANOTHREAD_LIST)
# aux_source_directory(./adgrid/mshio MSHIO_LIST)
# aux_source_directory(./adgrid/predicates PREDICATES_LIST)
file(GLOB ADGRID_LIST ./adgrid/*.c*)
file(GLOB ADGRID_LIST ./adgrid/src/*.c*)
file(GLOB NANOTHREAD_LIST ./adgrid/nanothread/*.c*)
file(GLOB MSHIO_LIST ./adgrid/mshio/*.c*)
file(GLOB MTET_LIST ./adgrid/mtet/*.c*)
file(GLOB PREDICATES_LIST ./adgrid/predicates/predicates.c)
# set(PREDICATES_LIST ./adgrid/predicates/predicates.c)

include_directories(./adgrid/include )
include_directories(./adgrid/src )
include_directories(./adgrid/include/mshio )

find_library(ATOMIC_LIB atomic_ops REQUIRED)
message(STATUS "MY_VAR is: ${ATOMIC_LIB}")
# message(STATUS  ${ATOMIC_LIB})

add_library(adgrid ${ADGRID_LIST}  ${NANOTHREAD_LIST} ${MSHIO_LIST} ${PREDICATES_LIST} ${MTET_LIST})

target_link_libraries(adgrid  nlohmann_json::nlohmann_json atomic)

# set(PicoTree_INCLUDE_DIR "/usr/local/include/pico_tree")
# include(CMakePrintHelpers)
# cmake_print_variables(${EIGEN3_INCLUDE_DIR})
# cmake_print_variables(${QHULL_INCLUDE_DIR})

# include_directories(${PicoTree_INCLUDE_DIR})
include_directories(./src )

include_directories(${NLOPT_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS} ./src/surfacer)

aux_source_directory(./src SRC_LIST)
aux_source_directory(./src/surfacer SURFACER_LIST)



LINK_DIRECTORIES(${ARMADILLO_LIB_DIRS} ${NLOPT_LIB_DIR} )


add_library(local_vipss ${SRC_LIST}  ${SURFACER_LIST})
target_link_libraries(local_vipss ${ARMADILLO_LIB} ${NLOPT_LIBRARIES} 
            OpenMP::OpenMP_CXX adgrid )

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} local_vipss ${ARMADILLO_LIB} ${NLOPT_LIBRARIES} 
            OpenMP::OpenMP_CXX adgrid )

find_library(OpenBLAS_LIB openblas REQUIRED)

add_executable(test_time test_time.cpp)
target_link_libraries(test_time local_vipss ${ARMADILLO_LIB} ${NLOPT_LIBRARIES} 
            OpenMP::OpenMP_CXX ${OpenBLAS_LIB})
