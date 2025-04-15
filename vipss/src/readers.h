#ifndef READERS_H
#define READERS_H


#include<vector>
#include<string>
#include <stdint.h>
#include "stats.h"

// using namespace std;

bool readOffFile(std::string filename,std::vector<double>&vertices,std::vector<unsigned int>&faces2vertices);


bool writeOffFile(std::string filename,const std::vector<double>&vertices,const std::vector<unsigned int>&faces2vertices);

bool writePLYFile(std::string filename, const std::vector<double>&vertices, const std::vector<unsigned int>&faces2vertices,
                  const std::vector<double> &vertices_normal, const std::vector<unsigned char>&vertices_color);
bool writePLYFile(std::string filename,const std::vector<double>&vertices);
bool writePLYFile_VF(std::string filename,const std::vector<double>&vertices,const std::vector<unsigned int>&faces2vertices);
bool writePLYFile_VN(std::string filename,const std::vector<double>&vertices, const std::vector<double>&vertices_normal);
bool writePLYFile_CO(std::string filename,const std::vector<double>&vertices,
                        const std::vector<uint8_t>&vertices_color);
bool writePLYFile_VN_CO(std::string filename,const std::vector<double>&vertices, 
                        const std::vector<double>&vertices_normal,
                        const std::vector<uint8_t>&vertices_color);


bool readPLYFile(std::string filename,  std::vector<double>&vertices, std::vector<double> &vertices_normal);

bool readObjFile(std::string filename, std::vector<double>&vertices, std::vector<unsigned int>&faces2vertices, std::vector<double> &vertices_normal);
bool readObjFile_Line(std::string filename,std::vector<double>&vertices,std::vector<unsigned int>&edges2vertices);

bool writeObjFile(std::string filename,const std::vector<double>&vertices,const std::vector<unsigned int>&faces2vertices);
bool writeObjFile_line(std::string filename,const std::vector<double>&vertices,const std::vector<unsigned int>&edge2vertices);
bool writeObjFile_vn(std::string filename,const std::vector<double>&vertices,const std::vector<double>&vn);
bool writeObjPtn_line(std::string filename, const std::vector<double>&vertices, 
                        const std::vector<uint8_t>&colors,
                       const std::vector<double>&normals);

bool readSurfFile(std::string filename,std::vector<double>&vertices,std::vector<unsigned int>&faces2vertices,std::vector<double>&vertices_field);

bool writeSurfFile(std::string filename,const std::vector<double>&vertices,const std::vector<unsigned int>&faces2vertices,const std::vector<double>&vertices_field);


bool readCurfFile(std::string filename, std::vector<double>&vertices, std::vector<unsigned int>&edges2vertices, std::vector<double>&vertices_field, std::vector<double> &vertices_tangent);


bool writeCurfFile(std::string filename, const std::vector<double>&vertices, const std::vector<unsigned int>&edges2vertices, std::vector<double>&vertices_field , std::vector<double> &vertices_tangent);


bool readVolfFile(std::string filename, std::vector<double>&vertices, std::vector<unsigned int>&tets2vertices, std::vector<double> &vertices_normal, std::vector<double> &vertices_field);


bool writeVolfFile(std::string filename, const std::vector<double>&vertices, const std::vector<unsigned int>&tets2vertices, std::vector<double> &vertices_normal, std::vector<double> &vertices_field);

bool readContourEdgeTxtFile(std::string filename, std::vector<int>&edges2vertices);
bool writeContourEdgeTxtFile(std::string filename, const std::vector<unsigned int>&edges2vertices);



bool writeVecFile(std::string filename, const std::vector<int>&vec);
bool readVecFile(std::string filename,std::vector<int>&vec);



bool readVVecFile(std::string filename, std::vector<std::vector<int>> &vvec);
bool writeVVecFile(std::string filename, const std::vector<std::vector<int>> &vvec);

bool writeVecFile(std::string filename, const std::vector<float>&vec);
bool readVecFile(std::string filename,std::vector<float>&vec);


bool readVecFile(std::string filename,std::vector<double>&vec);

bool readVVecFile(std::string filename, std::vector<std::vector<double>> &vvec);
bool writeVVecFile(std::string filename, const std::vector<std::vector<double>> &vvec);

bool writeSufFile(std::string filename, const std::vector<double>&vertices, const std::vector<unsigned int>&faces2vertices, const std::vector<int>&facesMat, const std::vector<int> &CtrEdges);
bool writeCtrGraphFile(std::string filename, const std::vector<float> &vertices, const std::vector<std::vector<int>>&edge2vertices, const std::vector<std::vector<int>>&edgeMat, const std::vector<std::vector<float>>&planepara);

bool writeCurNetFile(std::string filename, const std::vector<double> &vertices, const std::vector<std::vector<int>>&edge2vertices, const std::vector<std::vector<int>>&edgeMat, const std::vector<std::vector<double>>&planepara, const std::vector<double>&verticesNor);

bool readXYZ(std::string filename, std::vector<double>&v);
bool readXYZnormal(std::string filename, std::vector<double>&v, std::vector<double>&vn);
bool writeXYZ(std::string filename, std::vector<double>&v);
bool writeXYZnormal(std::string filename, std::vector<double>&v, std::vector<double>&vn);

bool readPlyMesh(const std::string& filename, std::vector<std::array<double, 3>>& vts, std::vector<std::vector<size_t>>& faces);

bool writePlyMesh(const std::string& filename,  const std::vector<std::array<double, 3>>& vts,
                                                const std::vector<std::vector<size_t>>& faces);
bool writePlyMeshWithColor(const std::string& filename,  const std::vector<std::array<double, 3>>& vts,
                                                const std::vector<std::array<double, 3>>& colors, 
                                                const std::vector<std::vector<size_t>>& faces);

bool SaveSphere(const std::string& filename,  const std::vector<std::array<double, 3>>& vts, 
                                              const std::vector<std::vector<size_t>>& faces,
                                              const std::array<double,3> center, const double radius);

void output_opt_pts_with_color(const std::vector<double>& pts, const std::vector<double>& s_vals, 
                               const std::string& out_dir);


void WriteStatsLog(const std::string& path, const VP_STATS& vp_stats);
void WriteStatsTimeCSV(const std::string& path, const VP_STATS& vp_stats);
void WriteVectorValsToCSV(const std::string& path, const std::vector<double>& vector_vals);
// void WriteStatsTimeCSV(const std::string& path, const VP_STATS& vp_stats);
std::vector<double> ReadVectorFromFile(const std::string& filename) ;


void SaveMeshWithQualityToPly(const std::string &filename, 
               const std::vector<std::array<double,3>> &points, 
               const std::vector<double>& point_qualities,
               const std::vector<std::array<size_t, 3>> &faces);

void SaveMeshToPly(const std::string &filename, 
               const std::vector<std::array<double,3>> &points, 
               const std::vector<std::vector<size_t>> &faces);

void SaveMeshToPly(const std::string &filename, 
                const std::vector<std::array<double,3>> &points, 
                const std::vector<double> &points_quality, 
                const std::vector<std::vector<size_t>> &faces);

void SaveMeshToPly(const std::string &filename, 
               const std::vector<std::array<double,3>> &points, 
               const std::vector<std::array<size_t, 3>> &faces);

void SaveTetMeshToPly(const std::vector<std::array<double, 3>>& vertices, 
                      const std::vector<std::array<size_t, 4>>& tets,
                     const std::vector<double>& values, const std::string& file_path);

void SavePointsWithQualityToPLY(const std::string& filename, 
                        const std::vector<std::array<double,3>>& points,
                        const std::vector<double>& qualtity);

bool SaveTetToFile(const std::vector<std::array<double, 3>>& vertices, 
    const std::vector<std::array<size_t, 4>>& tet, const std::string& filename);

bool ReadTetFromFile(std::vector<std::array<double, 3>>& vertices, 
        std::vector<std::array<size_t, 4>>& tet, const std::string& filename);     
#endif // READERS_H
