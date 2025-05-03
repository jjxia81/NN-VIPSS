#include"readers.h"
#include<iostream>
#include <iomanip>
#include<fstream>
#include<sstream>
#include<assert.h>
#include "happly.h"
#include <unordered_set>
using namespace std;

bool readOffFile(string filename,vector<double>&vertices,vector<unsigned int>&faces2vertices){
    ifstream reader(filename.data(), ofstream::in);
    if (!reader.good()) {
        cout << "Can not open the OFF file " << filename << endl;
        return false;
    }else {
        cout << "Reading: "<<filename<<endl;
    }

    string ss;

    int ivalue,ibuf[20];
    reader>>ss;
    //cout<<ss<<endl;
    if(ss!="OFF"){
        cout << "Not OFF file: " << filename << endl;
        return false;
    }


    int n_vertices,n_faces,n_edges;
    reader>>n_vertices;
    reader>>n_faces;
    reader>>n_edges;

    cout<<n_vertices<<' '<< n_faces<<' '<<n_edges<<endl;
    vertices.resize(n_vertices*3);
    //faces2vertices.resize(n_faces*3);

    for(int i =0;i<vertices.size();i++){
        reader>>vertices[i];
    }

    faces2vertices.clear();
    for(int i =0;i<n_faces;i++){
        reader>>ivalue;
        vector<int>tempvlist(ivalue);
        for(int j=0;j<ivalue;++j)reader>>tempvlist[j];
        for(int i=2;i<tempvlist.size();++i){
            faces2vertices.push_back(tempvlist[0]);
            faces2vertices.push_back(tempvlist[i-1]);
            faces2vertices.push_back(tempvlist[i]);
        }
    }




    reader.close();
    return true;



}


bool writeOffFile(string filename,const vector<double>&vertices,const vector<unsigned int>&faces2vertices){
    filename = filename + ".off";
    ofstream outer(filename.data(), ofstream::out);
    if (!outer.good()) {
        cout << "Can not create output OFF file " << filename << endl;
        return false;
    }


    size_t n_vertices = vertices.size()/3;
    size_t n_faces = faces2vertices.size()/3;

    outer<<"OFF"<<endl;
    outer<<n_vertices<<' '<<n_faces<<' '<<0<<endl;
    for(size_t i=0;i<n_vertices;++i){
        auto p_v = vertices.data()+i*3;
        outer << p_v[0] << " "<< p_v[1] << " "<< p_v[2] << endl;
    }
    for(size_t i=0;i<n_faces;++i){
        auto p_fv = faces2vertices.data()+i*3;
        outer << "3 " << p_fv[0]+1<< " "<< p_fv[1]+1 << " "<< p_fv[2]+1 << endl;
    }

    outer.close();
    cout<<"saving finish: "<<filename<<endl;
    return true;

}

bool writePLYFile(string filename,const vector<double>&vertices,const vector<unsigned int>&faces2vertices,
                  const vector<double>&vertices_normal,const vector<unsigned char>&vertices_color){
    filename = filename + ".ply";
    ofstream outer(filename.data(), ofstream::out);
    if (!outer.good()) {
        cout << "Can not create output PLY file " << filename << endl;
        return false;
    }


    size_t n_vertices = vertices.size()/3;
    size_t n_faces = faces2vertices.size()/3;
    outer << "ply" <<endl;
    outer << "format ascii 1.0"<<endl;
    outer << "element vertex " << n_vertices <<endl;
    outer << "property float x" <<endl;
    outer << "property float y" <<endl;
    outer << "property float z" <<endl;
    outer << "property float nx" <<endl;
    outer << "property float ny" <<endl;
    outer << "property float nz" <<endl;
    outer << "property uchar red" <<endl;
    outer << "property uchar green" <<endl;
    outer << "property uchar blue" <<endl;
    outer << "property uchar alpha" <<endl;
    outer << "element face " << n_faces <<endl;
    outer << "property list uchar int vertex_indices" <<endl;
    outer << "end_header" <<endl;

    for(size_t i=0;i<n_vertices;++i){
        auto p_v = vertices.data()+i*3;
        auto p_vn = vertices_normal.data()+i*3;
        auto p_vc = vertices_color.data()+i*4;
        for(size_t j=0;j<3;++j)outer << p_v[j] << " ";
        for(size_t j=0;j<3;++j)outer << p_vn[j] << " ";
        for(size_t j=0;j<4;++j)outer << size_t(p_vc[j]) << " ";
        outer << endl;
    }

    for(size_t i=0;i<n_faces;++i){
        auto p_fv = faces2vertices.data()+i*3;
        outer << "3 ";
        for(size_t j=0;j<3;++j)outer << p_fv[j] << " ";
        outer << endl;
    }
    outer.close();
    cout<<"saving finish: "<<filename<<endl;
    return true;
}

bool writePLYFile_VF(string filename,const vector<double>&vertices,const vector<unsigned int>&faces2vertices){
    filename = filename;
    ofstream outer(filename.data(), ofstream::out);
    if (!outer.good()) {
        cout << "Can not create output PLY file " << filename << endl;
        return false;
    }


    size_t n_vertices = vertices.size()/3;
    size_t n_faces = faces2vertices.size()/3;
    outer << "ply" <<endl;
    outer << "format ascii 1.0"<<endl;
    outer << "element vertex " << n_vertices <<endl;
    outer << "property float x" <<endl;
    outer << "property float y" <<endl;
    outer << "property float z" <<endl;
    outer << "element face " << n_faces <<endl;
    outer << "property list uchar int vertex_indices" <<endl;
    outer << "end_header" <<endl;

    for(size_t i=0;i<n_vertices;++i){
        auto p_v = vertices.data()+i*3;
        for(size_t j=0;j<3;++j)outer << p_v[j] << " ";
        outer << endl;
    }

    for(size_t i=0;i<n_faces;++i){
        auto p_fv = faces2vertices.data()+i*3;
        outer << "3 ";
        for(size_t j=0;j<3;++j)outer << p_fv[j] << " ";
        outer << endl;
    }
    outer.close();
    cout<<"saving finish: "<<filename<<endl;
    return true;
}


bool writePLYFile_VN(string filename,const vector<double>&vertices, const vector<double>&vertices_normal){
    filename = filename + ".ply";
    ofstream outer(filename.data(), ofstream::out);
    if (!outer.good()) {
        cout << "Can not create output PLY file " << filename << endl;
        return false;
    }

    size_t n_vertices = vertices.size()/3;
    outer << "ply" <<endl;
    outer << "format ascii 1.0"<<endl;
    outer << "element vertex " << n_vertices <<endl;
    outer << "property float x" <<endl;
    outer << "property float y" <<endl;
    outer << "property float z" <<endl;
    outer << "property float nx" <<endl;
    outer << "property float ny" <<endl;
    outer << "property float nz" <<endl;
    outer << "end_header" <<endl;

    for(size_t i=0;i<n_vertices;++i){
        auto p_v = vertices.data()+i*3;
        auto p_vn = vertices_normal.data()+i*3;
        for(size_t j=0;j<3;++j)outer << p_v[j] << " ";
        for(size_t j=0;j<3;++j)outer << p_vn[j] << " ";
        outer << endl;
    }

    outer.close();
    cout<<"saving finish: "<<filename<<endl;
    return true;
}


bool writePLYFile(string filename,const vector<double>&vertices){
    filename = filename + ".ply";
    ofstream outer(filename.data(), ofstream::out);
    if (!outer.good()) {
        cout << "Can not create output PLY file " << filename << endl;
        return false;
    }

    size_t n_vertices = vertices.size()/3;
    outer << "ply" <<endl;
    outer << "format ascii 1.0"<<endl;
    outer << "element vertex " << n_vertices <<endl;
    outer << "property float x" <<endl;
    outer << "property float y" <<endl;
    outer << "property float z" <<endl;
    outer << "end_header" <<endl;

    for(size_t i=0;i<n_vertices;++i){
        auto p_v = vertices.data()+i*3;
        for(size_t j=0;j<3;++j)outer << p_v[j] << " ";
        outer << endl;
    }

    outer.close();
    cout<<"saving finish: "<<filename<<endl;
    return true;
}


bool readPLYFile(string filename,  vector<double>&vertices, vector<double> &vertices_normal){
    ifstream fin(filename.data());
    if(fin.fail()){
        cout<<"Fail to open input file: "<<filename<<endl;
        return false;
    }

    vertices.clear();
    vertices_normal.clear();
    auto readVerticesAndNormal = [&vertices,&vertices_normal](stringstream &strs){
        double dvalue;
        for(size_t i=0;i<3;++i){strs>>dvalue;vertices.push_back(dvalue);}
        for(size_t i=0;i<3;++i){strs>>dvalue;vertices_normal.push_back(dvalue);}
    };

    string oneline;

    cout<<"reading: "<<filename<<endl;
    bool isstart = false;
    while( getline( fin, oneline ) ){
        stringstream strs( oneline );
        string prefix;

        if(isstart){
            readVerticesAndNormal( strs ); continue;
        }else{
            strs >> prefix;
            if( prefix == "end_header"  ) {isstart = true; continue; }
        }
    }
    
    fin.close();
    return true;
}



bool writePLYFile_CO(string filename,const vector<double>&vertices,
                        const vector<uint8_t>&vertices_color){
    filename = filename + ".ply";
    ofstream outer(filename.data(), ofstream::out);
    if (!outer.good()) {
        cout << "Can not create output PLY file " << filename << endl;
        return false;
    }
    size_t n_vertices = vertices.size()/3;
    outer << "ply" <<endl;
    outer << "format ascii 1.0"<<endl;
    outer << "element vertex " << n_vertices <<endl;
    outer << "property float x" <<endl;
    outer << "property float y" <<endl;
    outer << "property float z" <<endl;
    outer << "property uchar red" <<endl;
    outer << "property uchar green" <<endl;
    outer << "property uchar blue" <<endl;
    outer << "end_header" <<endl;

    for(size_t i=0;i<n_vertices;++i){
        auto p_v = vertices.data()+i*3;

        auto p_co = vertices_color.data() + i*3;
        for(size_t j=0;j<3;++j)outer << p_v[j] << " ";
        for(size_t j=0;j<3;++j)outer << to_string(p_co[j]) << " ";
        outer << endl;
    }

    outer.close();
    cout<<"saving finish: "<<filename<<endl;
    return true;
}



bool writePLYFile_VN_CO(string filename,const vector<double>&vertices, 
                        const vector<double>&vertices_normal,
                        const vector<uint8_t>&vertices_color){
    filename = filename + ".ply";
    ofstream outer(filename.data(), ofstream::out);
    if (!outer.good()) {
        cout << "Can not create output PLY file " << filename << endl;
        return false;
    }

    int n_vertices = vertices.size()/3;
    outer << "ply" <<endl;
    outer << "format ascii 1.0"<<endl;
    outer << "element vertex " << n_vertices <<endl;
    outer << "property float x" <<endl;
    outer << "property float y" <<endl;
    outer << "property float z" <<endl;
    outer << "property float nx" <<endl;
    outer << "property float ny" <<endl;
    outer << "property float nz" <<endl;
    outer << "property uchar red" <<endl;
    outer << "property uchar green" <<endl;
    outer << "property uchar blue" <<endl;
    outer << "end_header" <<endl;

    for(int i=0;i<n_vertices;++i){
        auto p_v = vertices.data()+i*3;
        auto p_vn = vertices_normal.data()+i*3;
        auto p_co = vertices_color.data() + i*3;
        for(int j=0;j<3;++j)outer << p_v[j] << " ";
        for(int j=0;j<3;++j)outer << p_vn[j] << " ";
        for(int j=0;j<3;++j)outer << to_string(p_co[j]) << " ";
        outer << endl;
    }

    outer.close();
    cout<<"saving finish: "<<filename<<endl;
    return true;
}



bool readObjFile(string filename, vector<double>&vertices, vector<unsigned int>&faces2vertices, vector<double>&vertices_normal){

    ifstream fin(filename.data());
    if(fin.fail()){
        cout<<"Fail to open input file: "<<filename<<endl;
        return false;
    }

    vertices.clear();
    faces2vertices.clear();
    vertices_normal.clear();
    auto readVertices = [&vertices](stringstream &strs){
        double dvalue;
        for(size_t i=0;i<3;++i){strs>>dvalue;vertices.push_back(dvalue);}
    };
    auto readFaces = [&faces2vertices](stringstream &strs){
        string oneset,indstring;
        size_t ivalue;
        size_t nv = 0;
        vector<int>tempvlist;
        while (strs>>oneset){
            stringstream oneset_ss( oneset );
            getline( oneset_ss, indstring, '/' );
            stringstream indstring_ss(indstring);
            indstring_ss >> ivalue;
            nv++;
            tempvlist.push_back(ivalue-1);
        }
        assert(tempvlist.size()>=3);
        for(size_t i=2;i<tempvlist.size();++i){
            faces2vertices.push_back(tempvlist[0]);
            faces2vertices.push_back(tempvlist[i-1]);
            faces2vertices.push_back(tempvlist[i]);
        }
    };
    auto readVerticesNormal = [&vertices_normal](stringstream &strs){
        double dvalue;
        for(size_t i=0;i<3;++i){strs>>dvalue;vertices_normal.push_back(dvalue);}
    };

    string oneline;

    cout<<"reading: "<<filename<<endl;

    while( getline( fin, oneline ) ){
        stringstream strs( oneline );
        string prefix;

        strs >> prefix;

        if( prefix == "v"  ) { readVertices( strs ); continue; } // vertex
        if( prefix == "vt" ) {  continue; } // texture coordinate
        if( prefix == "vn" ) {  readVerticesNormal(strs);continue; } // vertex normal
        if( prefix == "vf" ) { /*readVerticesField( ss );*/ continue; } // tangent vector
        if( prefix == "f"  ) { readFaces( strs ); continue; } // face
        if( prefix[0] == '#' ) continue; // comment
        if( prefix == "o" ) continue; // object name
        if( prefix == "g" ) continue; // group name
        if( prefix == "s" ) continue; // smoothing group
        if( prefix == "mtllib" ) continue; // material library
        if( prefix == "usemtl" ) continue; // material
        if( prefix == "k" ) continue; // field degree
        if( prefix == "fs" ) continue; // field singularity
        if( prefix == "" ) continue; // empty string
        if( prefix == "c" ) continue;

        cout << "Error: not a valid curf file!" << endl;
        cout << "(Offending line: " << oneline << ")" << endl;
        return false;
    }


    fin.close();
    return true;



}


bool readObjFile_Line(string filename,vector<double>&vertices,vector<unsigned int>&edges2vertices){


    ifstream fin(filename.data());
    if(fin.fail()){
        cout<<"Fail to open input file: "<<filename<<endl;
        exit(-1213);
        return false;
    }

    vertices.clear();
    edges2vertices.clear();
    auto readVertices = [&vertices](stringstream &strs){
        double dvalue;
        for(int i=0;i<3;++i){strs>>dvalue;vertices.push_back(dvalue);}
    };
    auto readEdges = [&edges2vertices](stringstream &strs){
        string oneset,indstring;
        unsigned int ivalue,ivalue2;
        strs>>ivalue;
        while (strs>>ivalue2){
            edges2vertices.push_back(ivalue-1);
            edges2vertices.push_back(ivalue2-1);
            ivalue = ivalue2;
        }

        //        for(int i=0;i<3;++i){
        //            strs>>ivalue;faces2vertices.push_back(ivalue-1);
        //        }
    };


    string oneline;

    cout<<"reading: "<<filename<<endl;

    while( getline( fin, oneline ) ){
        stringstream strs( oneline );
        string prefix;

        strs >> prefix;

        if( prefix == "v"  ) { readVertices( strs ); continue; } // vertex
        if( prefix == "l"  ) { readEdges( strs ); continue; } // edges
        if( prefix == "vt" ) {  continue; } // texture coordinate
        if( prefix == "vn" ) {  continue; } // vertex normal
        if( prefix == "vf" ) { /*readVerticesField( ss );*/ continue; } // tangent vector
        if( prefix == "f"  ) {  continue; } // face
        if( prefix[0] == '#' ) continue; // comment
        if( prefix == "o" ) continue; // object name
        if( prefix == "g" ) continue; // group name
        if( prefix == "s" ) continue; // smoothing group
        if( prefix == "mtllib" ) continue; // material library
        if( prefix == "usemtl" ) continue; // material
        if( prefix == "k" ) continue; // field degree
        if( prefix == "fs" ) continue; // field singularity
        if( prefix == "" ) continue; // empty string
        if( prefix == "c" ) continue;

        cout << "Error: not a valid curf file!" << endl;
        cout << "(Offending line: " << oneline << ")" << endl;
        return false;
    }


    fin.close();
    return true;




}

bool writeObjFile(string filename,const vector<double>&vertices,const vector<unsigned int>&faces2vertices){
    filename = filename + ".obj";
    ofstream outer(filename.data(), ofstream::out);
    if (!outer.good()) {
        cout << "Can not create output Obj file " << filename << endl;
        return false;
    }

    outer << setprecision(13);
    size_t n_vertices = vertices.size()/3;
    size_t n_faces = faces2vertices.size()/3;
    for(size_t i=0;i<n_vertices;++i){
        auto p_v = vertices.data()+i*3;
        outer << "v " << p_v[0] << " "<< p_v[1] << " "<< p_v[2] << endl;
    }

    for(int i=0;i<n_faces;++i){
        auto p_fv = faces2vertices.data()+i*3;
        outer << "f " << p_fv[0]+1<< " "<< p_fv[1]+1 << " "<< p_fv[2]+1 << endl;
    }


    outer.close();
    cout<<"saving finish: "<<filename<<endl;
    return true;



}

bool writeObjFile_vn(string filename,const vector<double>&vertices,const vector<double>&vn){
    filename = filename + ".obj";
    ofstream outer(filename.data(), ofstream::out);
    if (!outer.good()) {
        cout << "Can not create output Obj file " << filename << endl;
        return false;
    }

    outer << setprecision(8);
    size_t n_vertices = vertices.size()/3;
    for(size_t i=0;i<n_vertices;++i){
        auto p_v = vertices.data()+i*3;
        outer << "v " << p_v[0] << " "<< p_v[1] << " "<< p_v[2] << endl;
    }
    for(size_t i=0;i<n_vertices;++i){
        auto p_vn = vn.data()+i*3;
        outer << "vn " << p_vn[0] << " "<< p_vn[1] << " "<< p_vn[2] << endl;
    }


    outer.close();
    cout<<"saving finish: "<<filename<<endl;
    return true;


}

bool writeObjFile_line(string filename, const vector<double>&vertices, const vector<unsigned int> &edge2vertices){

    filename = filename + ".obj";
    ofstream outer(filename.data(), ofstream::out);
    if (!outer.good()) {
        cout << "Can not create output Obj file " << filename << endl;
        return false;
    }



    size_t n_vertices = vertices.size()/3;

    for(size_t i=0;i<n_vertices;++i){
        auto p_v = vertices.data()+i*3;
        outer << "v " << p_v[0] << " "<< p_v[1] << " "<< p_v[2] << endl;
    }


    {
        size_t n_edges = edge2vertices.size()/2;
        for(size_t i=0;i<n_edges;++i){
            auto p_ev = edge2vertices.data()+i*2;
            outer << "l " << p_ev[0]+1<< " "<< p_ev[1]+1 << endl;
        }
    }


    outer.close();
    cout<<"saving finish: "<<filename<<endl;
    return true;
}



bool writeObjFile_line(string filename, const vector<double>&vertices, 
                        const vector<uint8_t>& colors,
                       const vector<unsigned int> &edge2vertices){

    filename = filename + ".obj";
    ofstream outer(filename.data(), ofstream::out);
    if (!outer.good()) {
        cout << "Can not create output Obj file " << filename << endl;
        return false;
    }

    size_t n_vertices = vertices.size()/3;

    for(size_t i=0;i<n_vertices;++i){
        auto p_v = vertices.data()+i*3;
        double c_r = double(colors[3*i]) / 255.0; 
        double c_g = double(colors[3*i + 1]) / 255.0; 
        double c_b = double(colors[3*i + 2]) / 255.0; 

        outer << "v " << p_v[0] << " "<< p_v[1] << " "<< p_v[2]  
                << " " << c_r  << " " << c_g << " " << c_b << endl;
    }


    {
        size_t n_edges = edge2vertices.size()/2;
        for(size_t i=0;i<n_edges;++i){
            auto p_ev = edge2vertices.data()+i*2;
            outer << "l " << p_ev[0]+1<< " "<< p_ev[1]+1 << endl;
        }
    }
    outer.close();
    cout<<"saving finish: "<<filename<<endl;
    return true;
}



bool writeObjPtn_line(string filename, const vector<double>&vertices, 
                        const vector<uint8_t>&colors,
                       const vector<double>&normals)
{
    vector<double> new_vertices;
    vector<uint8_t> new_colors;
    vector<unsigned int> edge2vertices;

    double ratio = 0.1;
    
    filename = filename + ".obj";
    ofstream outer(filename.data(), ofstream::out);
    if (!outer.good()) {
        cout << "Can not create output Obj file " << filename << endl;
        return false;
    }

    for(size_t i =0; i < vertices.size()/3; ++i)
    {
        // new_vertices.push_back(vertices[3*i]);
        // new_vertices.push_back(vertices[3*i + 1]);
        // new_vertices.push_back(vertices[3*i + 2]);
        double p1_x = vertices[3*i] + ratio * normals[3*i];
        double p1_y = vertices[3*i + 1] + ratio * normals[3*i + 1];
        double p1_z = vertices[3*i + 2] + ratio * normals[3*i + 2];

        double r = colors[3*i]/255.0; 
        double g = colors[3*i + 1]/255.0; 
        double b = colors[3*i + 2]/255.0; 

        outer << "v " << p1_x << " "<< p1_y << " "<< p1_z  
                << " " << r << " " << g << " " << b << endl;

        p1_x = vertices[3*i] ;
        p1_y = vertices[3*i + 1] ;
        p1_z = vertices[3*i + 2] ;

        outer << "v " << p1_x << " "<< p1_y << " "<< p1_z  
                << " " << r << " " << g << " " << b << endl;

        p1_x = vertices[3*i] - ratio * normals[3*i];
        p1_y = vertices[3*i + 1] - ratio * normals[3*i + 1];
        p1_z = vertices[3*i + 2] - ratio * normals[3*i + 2];

        outer << "v " << p1_x << " "<< p1_y << " "<< p1_z  
                << " " << r << " " << g << " " << b << endl;

    }

    for(size_t i =0; i < vertices.size()/3; ++i)
    {
        outer << "f " << 3*i + 1<< " "<< 3*i + 2 << " " << 3*i + 3 << endl;
    }
    outer.close();

    // writeObjFile_line(filename, new_vertices, new_colors, edge2vertices);
}



bool readSurfFile(string filename,vector<double>&vertices,vector<unsigned int>&faces2vertices,vector<double>&vertices_field){
    ifstream fin(filename.data());
    if(fin.fail()){
        cout<<"Fail to open input file: "<<filename<<endl;
        return false;
    }

    vertices.clear();
    faces2vertices.clear();
    vertices_field.clear();
    auto readVertices = [&vertices](stringstream &strs){
        double dvalue;
        for(size_t i=0;i<3;++i){strs>>dvalue;vertices.push_back(dvalue);}
    };
    auto readFaces = [&faces2vertices](stringstream &strs){
        int ivalue;
        for(size_t i=0;i<3;++i){
            strs>>ivalue;faces2vertices.push_back(ivalue-1);
        }
    };
    auto readVerticesField = [&vertices_field](stringstream &strs){
        double dvalue;
        for(size_t i=0;i<3;++i){strs>>dvalue;vertices_field.push_back(dvalue);}
    };

    string oneline;

    cout<<"reading: "<<filename<<endl;

    while( getline( fin, oneline ) ){
        stringstream strs( oneline );
        string prefix;

        strs >> prefix;

        if( prefix == "v"  ) { readVertices( strs ); continue; } // vertex
        if( prefix == "vt" ) {  continue; } // texture coordinate
        if( prefix == "vn" ) { /* readVerticesNormal(strs);*/continue; } // vertex normal
        if( prefix == "vf" ) { readVerticesField( strs ); continue; } // tangent vector
        if( prefix == "f"  ) { readFaces( strs ); continue; } // face
        if( prefix[0] == '#' ) continue; // comment
        if( prefix == "o" ) continue; // object name
        if( prefix == "g" ) continue; // group name
        if( prefix == "s" ) continue; // smoothing group
        if( prefix == "mtllib" ) continue; // material library
        if( prefix == "usemtl" ) continue; // material
        if( prefix == "k" ) continue; // field degree
        if( prefix == "fs" ) continue; // field singularity
        if( prefix == "" ) continue; // empty string
        if( prefix == "c" ) continue;

        cout << "Error: not a valid curf file!" << endl;
        cout << "(Offending line: " << oneline << ")" << endl;
        return false;
    }


    fin.close();
    return true;


}

bool writeSurfFile(string filename,const vector<double>&vertices,const vector<unsigned int>&faces2vertices,const vector<double>&vertices_field){



    filename = filename + ".surf";
    ofstream outer(filename.data(), ofstream::out);
    if (!outer.good()) {
        cout << "Can not create output Surf file " << filename << endl;
        return false;
    }

    size_t n_vertices = vertices.size()/3;
    size_t n_faces = faces2vertices.size()/3;
    for(size_t i=0;i<n_vertices;++i){
        auto p_v = vertices.data()+i*3;
        outer << "v " << p_v[0] << " "<< p_v[1] << " "<< p_v[2] << endl;
    }

    for(int i=0;i<n_faces;++i){
        auto p_fv = faces2vertices.data()+i*3;
        outer << "f " << p_fv[0]+1<< " "<< p_fv[1]+1 << " "<< p_fv[2]+1 << endl;
    }

    for(int i=0;i<n_vertices;++i){
        auto p_vvec = vertices_field.data()+i*3;
        outer << "vf " << p_vvec[0] << " "<< p_vvec[1] << " "<< p_vvec[2] << endl;
    }

    outer.close();
    cout<<"saving finish: "<<filename<<endl;

    return true;
}


bool readCurfFile(string filename,vector<double>&vertices,vector<unsigned int>&edges2vertices,vector<double>&vertices_field,vector<double>&vertices_tangent){

    ifstream fin(filename.data());
    if(fin.fail()){
        cout<<"Fail to open input file: "<<filename<<endl;
        return false;
    }

    vertices.clear();
    edges2vertices.clear();
    vertices_field.clear();
    auto readVertices = [&vertices](stringstream &strs){
        double dvalue;
        for(int i=0;i<3;++i){strs>>dvalue;vertices.push_back(dvalue);}
    };
    auto readEdges = [&edges2vertices](stringstream &strs){
        int ivalue;
        for(int i=0;i<2;++i){
            strs>>ivalue;edges2vertices.push_back(ivalue-1);
        }
    };
    auto readVerticesField = [&vertices_field](stringstream &strs){
        double dvalue;
        for(int i=0;i<3;++i){strs>>dvalue;vertices_field.push_back(dvalue);}
    };
    auto readVerticesTangent = [&vertices_tangent](stringstream &strs){
        double dvalue;
        for(int i=0;i<3;++i){strs>>dvalue;vertices_tangent.push_back(dvalue);}
    };


    string oneline;

    cout<<"reading: "<<filename<<endl;

    while( getline( fin, oneline ) ){
        stringstream strs( oneline );
        string prefix;

        strs >> prefix;

        if( prefix == "v"  ) { readVertices( strs ); continue; } // vertex
        if( prefix == "e" ) {  readEdges(strs);continue; } // texture coordinate
        if( prefix == "vn" ) {  readVerticesTangent(strs);continue; } // vertex normal(surface/volume)/tangent(curve)/LookAt Vector(Curve)
        if( prefix == "vf" ) { readVerticesField( strs ); continue; } // vertices field
        if( prefix[0] == '#' ) continue; // comment line

        cout << "Error: not a valid curf file!" << endl;
        cout << "(Offending line: " << oneline << ")" << endl;
        return false;
    }


    fin.close();
    return true;


}


bool writeCurfFile(string filename, const vector<double>&vertices, const vector<unsigned int>&edges2vertices , vector<double> &vertices_field, vector<double>&vertices_tangent){

    filename = filename + ".curf";
    ofstream fout(filename.data());
    if(fout.fail()){
        cout<<"Fail to create output file: "<<filename<<endl;
        return false;
    }
    size_t n_vertices = vertices.size()/3;
    size_t n_edges = edges2vertices.size()/2;

    auto p_vd = vertices.data();
    auto p_evd = edges2vertices.data();
    auto p_vvecd = vertices_field.data();
    auto p_vtd = vertices_tangent.data();
    for(size_t i =0; i<n_vertices;++i){
        auto p_v = p_vd+i*3;
        fout<<"v "<<p_v[0]<<' '<<p_v[1]<<' '<<p_v[2]<<endl;

    }
    for(size_t i=0;i<n_edges;++i){
        auto p_ev = p_evd+i*2;
        fout<<"e "<<p_ev[0]+1<<' '<<p_ev[1]+1<<endl;
    }

    if(vertices_field.size()!=vertices.size()){
        cout<<"Output vertices field is invalid or empty!"<<endl;
    }else{
        for(size_t i =0; i<n_vertices;++i){
            auto p_vvec = p_vvecd+i*3;
            fout<<"vf "<<p_vvec[0]<<' '<<p_vvec[1]<<' '<<p_vvec[2]<<endl;
        }
    }

    if(vertices_tangent.size()!=vertices.size()){
        cout<<"Output vertices field is invalid or empty!"<<endl;
    }else{
        for(size_t i =0; i<n_vertices;++i){
            auto p_vt = p_vtd+i*3;
            fout<<"vn "<<p_vt[0]<<' '<<p_vt[1]<<' '<<p_vt[2]<<endl;
        }
    }
    fout.close();

    cout<<"Write: "<<filename<<endl;
    return true;


}


bool readVolfFile(string filename,vector<double>&vertices,vector<unsigned int>&tets2vertices,vector<double> &vertices_normal,vector<double> &vertices_field){

    ifstream fin(filename.data());
    if(fin.fail()){
        cout<<"Fail to open input file: "<<filename<<endl;
        return false;
    }

    vertices.clear();
    tets2vertices.clear();
    vertices_field.clear();
    auto readVertices = [&vertices](stringstream &strs){
        double dvalue;
        for(int i=0;i<3;++i){strs>>dvalue;vertices.push_back(dvalue);}
    };
    auto readTets = [&tets2vertices](stringstream &strs){
        int ivalue;
        for(int i=0;i<4;++i){
            strs>>ivalue;tets2vertices.push_back(ivalue-1);
        }
    };
    auto readVerticesField = [&vertices_field](stringstream &strs){
        double dvalue;
        for(int i=0;i<3;++i){strs>>dvalue;vertices_field.push_back(dvalue);}
    };
    auto readVerticesNormal = [&vertices_normal](stringstream &strs){
        double dvalue;
        for(int i=0;i<3;++i){strs>>dvalue;vertices_normal.push_back(dvalue);}
    };

    string oneline;

    cout<<"reading: "<<filename<<endl;

    while( getline( fin, oneline ) ){
        stringstream strs( oneline );
        string prefix;

        strs >> prefix;

        if( prefix == "v"  ) { readVertices( strs ); continue; } // vertex
        if( prefix == "vn" ) {  readVerticesNormal(strs);continue; } // vertex normal
        if( prefix == "vf" ) { readVerticesField( strs ); continue; } // tangent vector
        if( prefix == "tet"  ) { readTets( strs ); continue; } // face
        if( prefix[0] == '#' ) continue; // comment

        cout << "Error: not a valid curf file!" << endl;
        cout << "(Offending line: " << oneline << ")" << endl;
        return false;
    }


    fin.close();
    return true;

}


bool writeVolfFile(string filename, const vector<double>&vertices, const vector<unsigned int>&tets2vertices, vector<double> &vertices_normal, vector<double> &vertices_field){
    filename = filename + ".volf";
    ofstream fout(filename.data());
    if(fout.fail()){
        cout<<"Fail to create output file: "<<filename<<endl;
        return false;
    }
    size_t n_vertices = vertices.size()/3;
    size_t n_tets = tets2vertices.size()/4;

    auto p_vd = vertices.data();
    auto p_tvd = tets2vertices.data();
    auto p_vvecd = vertices_field.data();
    auto p_vnd = vertices_normal.data();
    for(size_t i =0; i<n_vertices;++i){
        auto p_v = p_vd+i*3;
        fout<<"v "<<p_v[0]<<' '<<p_v[1]<<' '<<p_v[2]<<endl;

    }
    for(size_t i=0;i<n_tets;++i){
        auto p_tv = p_tvd+i*4;
        fout<<"tet "<<p_tv[0]+1<<' '<<p_tv[1]+1<<' '<<p_tv[2]+1<<' '<<p_tv[3]+1<<endl;
    }

    if(vertices_normal.size()/3!=n_vertices){
        cout<<"Output vertices normal is invalid or empty!"<<endl;
    }else{
        for(size_t i =0; i<n_vertices;++i){
            auto p_vn = p_vnd+i*3;
            fout<<"vn "<<p_vn[0]<<' '<<p_vn[1]<<' '<<p_vn[2]<<endl;
        }
    }
    if(vertices_field.size()!=vertices.size()){
        cout<<"Output vertices field is invalid or empty!"<<endl;
    }else{
        for(size_t i =0; i<n_vertices;++i){
            auto p_vvec = p_vvecd+i*3;
            fout<<"vf "<<p_vvec[0]<<' '<<p_vvec[1]<<' '<<p_vvec[2]<<endl;
        }
    }


    fout.close();

    cout<<"Write: "<<filename<<endl;
    return true;
}
bool readContourEdgeTxtFile(string filename, vector<int>&edges2vertices){
    ifstream reader(filename.data(), ifstream::in);
    if (!reader.good()) {
        cout << "Can not open the file " << filename << endl;
        return false;
    }
    int nnum = 0;
    reader>>nnum;
    edges2vertices.resize(nnum*2);
    for(int i=0;i<nnum*2;++i)reader>>edges2vertices[i];
    reader.close();
    return true;
}
bool writeContourEdgeTxtFile(string filename, const vector<unsigned int>&edges2vertices){
    filename = filename + "_ContuorEdges.txt";
    ofstream fout(filename.data());
    if(fout.fail()){
        cout<<"Fail to create output file: "<<filename<<endl;
        return false;
    }

    size_t numofE = edges2vertices.size()/2;
    fout<<numofE<<endl;
    for(size_t i=0;i<numofE;++i){
        fout<<edges2vertices[i*2]<<' '<<edges2vertices[i*2+1]<<endl;
    }
    fout.close();

    cout<<"Write: "<<filename<<endl;
    return true;
}
bool writeVecFile(string filename, const vector<int> &vec){
    filename = filename + "_vec.txt";
    ofstream fout(filename.data());
    if(fout.fail()){
        cout<<"Fail to create output file: "<<filename<<endl;
        return false;
    }

    int numof = (int)vec.size();
    fout<<numof<<endl;
    if(numof==0)return true;
    for(auto a:vec)fout<<a<<endl;
    fout.close();

    cout<<"Write: "<<filename<<endl;
    return true;
}
bool readVecFile(string filename, vector<int> &vec){
    ifstream reader(filename.data(), ifstream::in);
    if (!reader.good()) {
        cout << "Can not open the file " << filename << endl;
        return false;
    }
    int nnum = 0;
    reader>>nnum;
    vec.resize(nnum);
    for(int i=0;i<nnum;++i)reader>>vec[i];
    reader.close();
    return true;
}
bool readVVecFile(string filename, vector<vector<int>> &vvec){
    ifstream reader(filename.data(), ifstream::in);
    if (!reader.good()) {
        cout << "Can not open the file " << filename << endl;
        return false;
    }
    int nnum = 0;
    reader>>nnum;
    vector<int>vvnum(nnum,0);
    for(int i=0;i<nnum;++i)reader>>vvnum[i];

    vvec.resize(nnum);
    for(int i=0;i<nnum;++i){
        vvec[i].resize(vvnum[i]);
        for(int j=0;j<vvnum[i];++j)reader>>vvec[i][j];
    }
    reader.close();
    return true;
}

bool writeVVecFile(string filename, const vector< vector<int> > &vvec){
    filename = filename + "_vvec.txt";
    ofstream fout(filename.data());
    if(fout.fail()){
        cout<<"Fail to create output file: "<<filename<<endl;
        return false;
    }

    int numof = (int)vvec.size();

    fout<<numof<<endl;

    if(numof==0)return true;
    for(int i=0;i<numof-1;i++){
        fout<<vvec[i].size()<<' ';
    }
    fout<<vvec[numof-1].size()<<endl;

    for(auto &a:vvec){
        for(int i=0;i<a.size()-1;i++){
            fout<<a[i]<<' ';
        }
        fout<<a[a.size()-1]<<endl;
    }
    fout.close();

    cout<<"Write: "<<filename<<endl;
    return true;
}
bool readVVecFile(string filename, vector<vector<double>> &vvec){
    ifstream reader(filename.data(), ifstream::in);
    if (!reader.good()) {
        cout << "Can not open the file " << filename << endl;
        return false;
    }
    int nnum = 0;
    reader>>nnum;
    vector<int>vvnum(nnum,0);
    for(int i=0;i<nnum;++i)reader>>vvnum[i];

    vvec.resize(nnum);
    for(int i=0;i<nnum;++i){
        vvec[i].resize(vvnum[i]);
        for(int j=0;j<vvnum[i];++j)reader>>vvec[i][j];
    }
    reader.close();
    return true;
}

bool writeVVecFile(string filename, const vector< vector<double> > &vvec){
    filename = filename + "_vvec.txt";
    ofstream fout(filename.data());
    if(fout.fail()){
        cout<<"Fail to create output file: "<<filename<<endl;
        return false;
    }

    size_t numof = vvec.size();

    fout<<numof<<endl;

    if(numof==0)return true;
    for(size_t i=0;i<numof-1;i++){
        fout<<vvec[i].size()<<' ';
    }
    fout<<vvec[numof-1].size()<<endl;

    for(auto &a:vvec){
        for(size_t i=0;i<a.size()-1;i++){
            fout<<a[i]<<' ';
        }
        fout<<a[a.size()-1]<<endl;
    }
    fout.close();

    cout<<"Write: "<<filename<<endl;
    return true;
}
bool writeVecFile(string filename, const vector<float>&vec){
    filename = filename + "_vec.txt";
    ofstream fout(filename.data());
    if(fout.fail()){
        cout<<"Fail to create output file: "<<filename<<endl;
        return false;
    }

    size_t numof = vec.size();
    fout<<numof<<endl;
    for(auto a:vec)fout<<a<<endl;
    fout.close();

    cout<<"Write: "<<filename<<endl;
    return true;
}
bool readVecFile(string filename,vector<float>&vec){
    ifstream reader(filename.data(), ifstream::in);
    if (!reader.good()) {
        cout << "Can not open the file " << filename << endl;
        return false;
    }
    int nnum = 0;
    reader>>nnum;
    vec.resize(nnum);
    for(int i=0;i<nnum;++i)reader>>vec[i];

    reader.close();
    return true;

}
bool readVecFile(string filename,vector<double>&vec){
    ifstream reader(filename.data(), ifstream::in);
    if (!reader.good()) {
        cout << "Can not open the file " << filename << endl;
        return false;
    }
    int nnum = 0;
    reader>>nnum;
    vec.resize(nnum);
    for(int i=0;i<nnum;++i)reader>>vec[i];

    reader.close();
    return true;

}
bool writeSufFile(string filename, const vector<double>&vertices, const vector<unsigned int>&faces2vertices, const vector<int>&facesMat, const vector<int> &CtrEdges){

    filename = filename + ".suf";
    ofstream outer(filename.data(), ofstream::out);
    if (!outer.good()) {
        cout << "Can not create output suf file " << filename << endl;
        return false;
    }
    size_t n_vertices = vertices.size()/3;
    size_t n_faces = faces2vertices.size()/3;


    outer<<n_vertices<<' '<<n_faces<<endl;
    for(size_t i=0;i<n_vertices;++i){
        auto p_v = vertices.data()+i*3;
        outer << p_v[0] << " "<< p_v[1] << " "<< p_v[2] << endl;
    }
    for(size_t i=0;i<n_faces;++i){
        auto p_fv = faces2vertices.data()+i*3;
        auto p_fm = facesMat.data()+i*2;
        outer<< p_fv[0]<< " "<< p_fv[1] << " "<< p_fv[2] << " "<< p_fm[0]<< " "<<p_fm[1] <<endl;
    }

    outer<<CtrEdges.size()/2<<endl;
    for(size_t i=0;i<CtrEdges.size()/2;++i){
        outer<<CtrEdges[i*2]<<' '<<CtrEdges[i*2+1]<<endl;
    }
    outer.close();
    return 1;
}
bool writeCtrGraphFile(string filename,const vector<float>&vertices,const vector<vector<int>>&edge2vertices,const vector<vector<int>>&edgeMat, const vector<vector<float>>&planepara){
    filename = filename + ".CtrGraph";
    ofstream outer(filename.data(), ofstream::out);
    if (!outer.good()) {
        cout << "Can not create output CtrGraph file " << filename << endl;
        return false;
    }

    size_t n_vertices = vertices.size()/3;
    size_t n_plane = edge2vertices.size();
    outer<<"n "<<n_vertices<<endl;
    auto p_vd = vertices.data();
    for(size_t i=0;i<n_vertices;++i){
        auto p_v = p_vd+ i*3;
        outer << "v "<<p_v[0] << " "<< p_v[1] << " "<< p_v[2] << endl;
    }

    outer<<"n "<<n_plane<<endl;
    for(size_t i=0;i<n_plane;++i){
        outer<<"p ";
        for(size_t j=0;j<4;++j)outer<<planepara[i][j]<<' ';

        auto &p_ev = edge2vertices[i];
        auto &p_em = edgeMat[i];
        size_t ne = p_ev.size()/2;
        outer<<ne<<endl;

        for(size_t j=0;j<ne;++j){
            auto ind = j*2;
            outer<<"e "<<p_ev[ind]<<' '<<p_ev[ind+1]<<' '<<p_em[ind]<<' '<<p_em[ind+1]<<endl;
        }
    }

    outer.close();


    cout<<"finish: "<<filename<<endl;
    return true;





}


bool writeCurNetFile(string filename, const vector<double> &vertices, const vector<vector<int> > &edge2vertices, const vector<vector<int> > &edgeMat, const vector<vector<double> > &planepara, const vector<double>&verticesNor){

    filename = filename + ".CurNet";
    ofstream outer(filename.data(), ofstream::out);
    if (!outer.good()) {
        cout << "Can not create output CurNet file " << filename << endl;
        return false;
    }

    size_t n_vertices = vertices.size()/3;
    size_t n_plane = edge2vertices.size();
    outer<<"n "<<n_vertices<<endl;
    auto p_vd = vertices.data();
    auto p_vnd = verticesNor.data();
    for(size_t i=0;i<n_vertices;++i){
        auto p_v = p_vd+ i*3;
        auto p_vn = p_vnd+i*3;
        outer << "v "<<p_v[0] << " "<< p_v[1] << " "<< p_v[2]  << " " <<p_vn[0] << " "<< p_vn[1] << " "<< p_vn[2] << endl;
    }

    outer<<"n "<<n_plane<<endl;
    for(size_t i=0;i<n_plane;++i){
        outer<<"p ";
        for(size_t j=0;j<4;++j)outer<<planepara[i][j]<<' ';

        auto &p_ev = edge2vertices[i];
        auto &p_em = edgeMat[i];
        size_t ne = p_ev.size()/2;
        outer<<ne<<endl;

        for(size_t j=0;j<ne;++j){
            auto ind = j*2;
            outer<<"e "<<p_ev[ind]<<' '<<p_ev[ind+1]<<' '<<p_em[ind]<<' '<<p_em[ind+1]<<endl;
        }
    }

    outer.close();

    return true;




}


bool readXYZ(string filename, vector<double>&v){

    ifstream reader(filename.data(), ofstream::in);
    if (!reader.good()) {
        cout << "Can not open the file " << filename << endl;
        return false;
    }else {
        cout << "Reading: "<<filename<<endl;
    }
    v.clear();
    std::string line;
    while (std::getline(reader, line)) {
        std::stringstream ss(line);
        double x, y, z;
        // Read the first three values as coordinates
        if (ss >> x >> y >> z) {
            v.push_back(x);
            v.push_back(y);
            v.push_back(z);
        }
        // Ignore the rest of the line (e.g., normals or other attributes)
    }
    // double val;
    // while(!reader.eof()){

    //     reader>>val;
    //     v.push_back(val);
    // }
    reader.close();
    return true;
}

// bool readXYZnormal(string filename, vector<double>&v, vector<double>&vn){

//     ifstream reader(filename.data(), ofstream::in);
//     if (!reader.good()) {
//         cout << "Can not open the file " << filename << endl;
//         return false;
//     }else {
//         cout << "Reading: "<<filename<<endl;
//     }
//     v.clear();
//     vn.clear();
//     double val;
//     while(!reader.eof()){
//         for(size_t i=0;i<3;++i){reader>>val;v.push_back(val);}
//         for(size_t i=0;i<3;++i){reader>>val;vn.push_back(val);}
//     }
//     reader.close();

//     return true;
// }


bool readXYZnormal(string filename, vector<double>&v, vector<double>&vn){

     std::ifstream xyzFile(filename);
    if (!xyzFile) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }
    std::string line;
    double x, y, z, nx, ny, nz;
    while (std::getline(xyzFile, line)) {
        std::istringstream iss(line);
        // Point3D point;
        // Try reading 6 values (x, y, z, nx, ny, nz)
        if (!(iss >> x >> y >> z)) {
            continue;
        }
        v.push_back(x); 
        v.push_back(y); 
        v.push_back(z); 
        // Check if normals are present
        if (!(iss >> nx >> ny >> nz)) {
            nx = ny = nz = 0.0;  // Default to (0,0,0) if normals are missing
        }
        vn.push_back(nx); 
        vn.push_back(ny); 
        vn.push_back(nz); 
    }
    xyzFile.close();
    std::cout << "Successfully read data from " << filename << std::endl;
    return true;
}


bool writeXYZ(string filename, vector<double>&v){

    size_t npt = v.size()/3;
    ofstream outer(filename.data(), ofstream::out);
    if (!outer.good()) {
        cout << "Can not create output file " << filename << endl;
        return false;
    }

    for(size_t i=0;i<npt;++i){
        auto p_v = v.data()+i*3;
        outer<<p_v[0]<<' '<<p_v[1]<<' '<<p_v[2]<<endl;
    }
    outer.close();
    return true;

}

bool writeXYZnormal(string filename, vector<double>&v, vector<double>&vn){

    size_t npt = v.size()/3;
    ofstream outer(filename.data(), ofstream::out);
    if (!outer.good()) {
        cout << "Can not create output file " << filename << endl;
        return false;
    }

    for(size_t i=0;i<npt;++i){
        auto p_v = v.data()+i*3;
        auto p_vn = vn.data()+i*3;
        outer<<p_v[0]<<' '<<p_v[1]<<' '<<p_v[2]<<' ';
        outer<<p_vn[0]<<' '<<p_vn[1]<<' '<<p_vn[2]<<endl;
    }
    outer.close();
    return true;

}

bool readPlyMesh(const std::string& filename, std::vector<std::array<double, 3>>& vts, std::vector<std::vector<size_t>>& faces)
{
    // Construct the data object by reading from file
    happly::PLYData plyIn(filename);

    // printf("fetch data to readPlyMesh -------- \n");
    // Get mesh-style data from the object
    vts = plyIn.getVertexPositions();
    printf("read mesh pt num : %lu \n", vts.size());
    faces = plyIn.getFaceIndices<size_t>();
    printf("read mesh face num : %lu \n", faces.size());
    return true;
}

bool writePlyMeshWithColor(const std::string& filename,  const std::vector<std::array<double, 3>>& vts,
                                                const std::vector<std::array<double, 3>>& colors, 
                                                const std::vector<std::vector<size_t>>& faces)
{
    // Create an empty object
    happly::PLYData plyOut;
    // Add mesh data (elements are created automatically)
    plyOut.addVertexPositions(vts);
    plyOut.addVertexColors(colors);
    plyOut.addFaceIndices(faces);
    // Write the object to file
    plyOut.write(filename, happly::DataFormat::ASCII);
    return true;
}

bool writePlyMesh(const std::string& filename,  const std::vector<std::array<double, 3>>& vts,
                                                const std::vector<std::vector<size_t>>& faces)
{
    // Create an empty object
    happly::PLYData plyOut;
    // Add mesh data (elements are created automatically)
    plyOut.addVertexPositions(vts);
    plyOut.addFaceIndices(faces);
    // Write the object to file
    plyOut.write(filename, happly::DataFormat::ASCII);
    return true;
}


bool SaveSphere(const std::string& filename,  const std::vector<std::array<double, 3>>& vts, 
                                              const std::vector<std::vector<size_t>>& faces,
                                              const std::array<double,3> center, const double radius)
{
    double r = ((double) rand() / (RAND_MAX));
    double g = ((double) rand() / (RAND_MAX));
    double b = ((double) rand() / (RAND_MAX));
    std::array<double, 3> col = {r, g, b};
    std::vector<std::array<double, 3>> new_vts;
    std::vector<std::array<double, 3>> colors(vts.size(), col);
    new_vts.resize(vts.size());
    for(size_t i =0; i < vts.size(); ++i)
    {
        new_vts[i][0] = vts[i][0] * radius + center[0];
        new_vts[i][1] = vts[i][1] * radius + center[1];
        new_vts[i][2] = vts[i][2] * radius + center[2];
    }

    writePlyMeshWithColor(filename, new_vts, colors, faces);
    return true;
}


void output_opt_pts_with_color(const std::vector<double>& pts, const std::vector<double>& s_vals, 
                               const std::string& out_dir)
{
    size_t ptn = pts.size()/3;
    std::vector<unsigned char> color;
    double max_val = 0.01;
    for(size_t i = 0; i < ptn; ++i)
    {
        double cur_dist = s_vals[i];
        if(cur_dist >= max_val)
        {
            color.push_back(255);
            color.push_back(0);
            color.push_back(0);
        } else if (cur_dist <= - max_val)
        {
            color.push_back(0);
            color.push_back(255);
            color.push_back(0);
        } else{
            unsigned char r = 0;
            unsigned char g = 0;
            if(cur_dist >= 0)
            {
                r = int(255 * cur_dist/max_val );
            } else {
                g = int(255 * (-cur_dist)/max_val );
            }

            color.push_back(r);
            color.push_back(g);
            color.push_back(0);
        }
    }
    writePLYFile_CO(out_dir, pts, color);
}



void WriteStatsLog(const std::string& path, const VP_STATS& vp_stats)
{
    std::ofstream log_file;
    log_file.open(path);
    if(log_file)
    {
        double time_sum = vp_stats.init_normal_total_time_ +
            vp_stats.opt_solver_time_ + vp_stats.build_H_total_time_ 
            + vp_stats.surface_total_time_;

        log_file << "time statistics for normal initializaiton : " << std::endl;
        log_file << "total time : " << time_sum << std::endl;
        log_file << "normal init time : " << vp_stats.init_normal_total_time_ << std::endl;
        log_file << "optimization time : " << vp_stats.opt_solver_time_ + vp_stats.build_H_total_time_  << std::endl;
        log_file << "surfacing time : " << vp_stats.surface_total_time_ << std::endl;

        log_file << "     " << std::endl;
        log_file << "time statistics for normal init : " << std::endl;
        log_file << "cal cluster init normal with partial vipps time : " << vp_stats.init_cluster_normal_time_ << std::endl;
        log_file << "cal cluster scores time : " << vp_stats.cal_cluster_neigbor_scores_time_ << std::endl;
        log_file << "build normal MST tree time : " << vp_stats.build_normal_MST_time_ << std::endl;
        log_file << "normal flip with MST tree time : " << vp_stats.normal_flip_time_ << std::endl;
        log_file << "Normal init total time : " << vp_stats.init_normal_total_time_ << std::endl;


        log_file << "     " << std::endl;
        log_file << "time statistics for optimization : " << std::endl;
        log_file << "cal cluster J total time : " << vp_stats.cal_cluster_J_total_time_ << std::endl;
        log_file << "add J ele to tris vector time : " << vp_stats.add_J_ele_to_triplet_vector_time_ << std::endl;
        log_file << "build H from tris vector time : " << vp_stats.build_eigen_final_h_from_tris_time_ << std::endl;
        log_file << "get final H sub block time : " << vp_stats.take_h_sub_block_time_ << std::endl;
        log_file << "build H total time : " << vp_stats.build_H_total_time_ << std::endl;
        log_file << "opt solver total time : " << vp_stats.opt_solver_time_ << std::endl;
        log_file << "opt func call total num : " << vp_stats.opt_func_call_num_ << std::endl;
        log_file << "Optimzation total time : " <<  vp_stats.opt_solver_time_ + vp_stats.build_H_total_time_ << std::endl;

        log_file << "     " << std::endl;
        log_file << "time statistics for surface : " << std::endl;
        // log_file << "generate voroi data time : " << vp_stats.generate_voroi_data_time_ << std::endl;
        log_file << "build cluster HRBF time : " << vp_stats.build_nn_rbf_time_ << std::endl;
        log_file << "natural neighbor n search time : " << vp_stats.neighbor_search_time_ << std::endl;
        log_file << "cal nn coords and HRBF time : " << vp_stats.cal_nn_coordinate_and_hbrf_time_ << std::endl;
        log_file << "voxel dist func val evaluated count : " << vp_stats.voxel_cal_num << std::endl;
        log_file << "natural neighbor surfacing total time : " << vp_stats.surface_total_time_ << std::endl;
    } 
}



void WriteStatsTimeCSV(const std::string& path, const VP_STATS& vp_stats)
{
    std::ofstream csv_file;
    csv_file.open(path);
    if(csv_file)
    {
        csv_file << "pt num, " << vp_stats.pt_num_ << std::endl;
        double normal_estimation_time = vp_stats.init_normal_total_time_ 
                                        + vp_stats.build_H_total_time_
                                        + vp_stats.opt_solver_time_;
        csv_file << "normal estimation timing, " << normal_estimation_time << std::endl;
        csv_file << "ave evaluate nn num, " << vp_stats.average_neighbor_num_ << std::endl;
        // csv_file << "ave cluster size, " << vp_stats.average_cluster_size_ << std::endl;
        csv_file << "octree dummy pt num, " << vp_stats.octree_dummy_pt_num_ << std::endl;
        csv_file << "tetgen triangulation, " << vp_stats.tetgen_triangulation_time_ << std::endl;
        csv_file << "init normal, " << vp_stats.init_normal_total_time_ << std::endl;
        // csv_file << " cal cluster scores time : " << vp_stats.cal_cluster_neigbor_scores_time_ << std::endl;
        csv_file << "construct Hmat, " << vp_stats.build_H_total_time_ << std::endl;
        // csv_file << "cal cluster J total time : " << vp_stats.cal_cluster_J_total_time_ << std::endl;
        csv_file << "optimization, " << vp_stats.opt_solver_time_ << std::endl;
        csv_file << "opt num, " << vp_stats.opt_func_call_num_ << std::endl;
        double iter_per_time = vp_stats.opt_solver_time_ / double(vp_stats.opt_func_call_num_);
        csv_file << "opt per iter time, " << iter_per_time << std::endl;
        // csv_file << "optimization total time : " << vp_stats.build_H_total_time_ << std::endl;
        csv_file << "generate voro, " << vp_stats.generate_voro_data_time_ << std::endl;
        csv_file << "build nn HRBF, " << vp_stats.build_nn_rbf_time_ << std::endl;
        // csv_file << "natural jm, " << vp_stats.neighbor_search_time_ << std::endl;
        csv_file << "evaluate count, " << vp_stats.nn_evaluate_count_ << std::endl;
        csv_file << "NN surface,  " << vp_stats.surface_total_time_ << std::endl;
        csv_file << "adgrid generation,  " << vp_stats.adgrid_gen_time_ << std::endl;
        csv_file << "global HRBF coefficient, " << vp_stats.hrbf_coefficient_time_ << std::endl;
        csv_file << "max memory, " << vp_stats.possible_max_memory << std::endl;
        csv_file << "final H memory, " << vp_stats.max_hmat_memory << std::endl;
        csv_file << "ave cluster size, " << vp_stats.ave_cluster_size << std::endl;
        csv_file << "cluster size std deviation, " << vp_stats.cluster_std_dev << std::endl;
        csv_file << "max cluster size : , " << vp_stats.max_cluster_size << std::endl;
    } 
}

void WriteVectorValsToCSV(const std::string& path, const std::vector<double>& vector_vals)
{
    std::ofstream csv_file;
    csv_file.open(path);
    if(csv_file)
    {
        for(auto val : vector_vals)
        {
            csv_file << val << std::endl;
        }
    } 
}


std::vector<double> ReadVectorFromFile(const std::string& filename) {
    std::ifstream inFile(filename);
    if (!inFile) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return {};
    }

    std::vector<double> vec;
    std::string line;
    while (std::getline(inFile, line)) {
        std::istringstream iss(line);
        double value;
        if (iss >> value) {
            vec.push_back(value);
        }
    }
    inFile.close();
    return vec;
}


void SaveMeshWithQualityToPly(const std::string &filename, 
               const std::vector<std::array<double,3>> &points, 
               const std::vector<double>& point_qualities,
               const std::vector<std::array<size_t, 3>> &faces) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error: Unable to open file for writing!\n";
        return;
    }

    // Write PLY header
    outFile << "ply\n";
    outFile << "format ascii 1.0\n";
    outFile << "element vertex " << points.size() << "\n";
    outFile << "property float x\n";
    outFile << "property float y\n";
    outFile << "property float z\n";
    outFile << "property float quality\n";  // Quality attribute
    outFile << "element face " << faces.size() << "\n";
    outFile << "property list uchar int vertex_indices\n";
    outFile << "end_header\n";

    // Write vertex data
    for (int i= 0; i < points.size(); ++i) {
        const auto& p = points[i];
        double quality = point_qualities[i];
        outFile << p[0] << " " << p[1] << " " << p[2] << " " << quality << "\n";
    }

    // Write face data
    for (const auto &face : faces) {
        outFile << face.size();
        for(auto vid : face)
        {
            outFile << " " << vid;
        }
        outFile << "\n";
    }

    outFile.close();
    std::cout << "PLY file saved successfully: " << filename << std::endl;
}



void SaveMeshToPly(const std::string &filename, 
               const std::vector<std::array<double,3>> &points, 
               const std::vector<std::vector<size_t>> &faces) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error: Unable to open file for writing!\n";
        return;
    }

    // Write PLY header
    outFile << "ply\n";
    outFile << "format ascii 1.0\n";
    outFile << "element vertex " << points.size() << "\n";
    outFile << "property float x\n";
    outFile << "property float y\n";
    outFile << "property float z\n";
    outFile << "element face " << faces.size() << "\n";
    outFile << "property list uchar int vertex_indices\n";
    outFile << "end_header\n";

    // Write vertex data
    for (int i= 0; i < points.size(); ++i) {
        const auto& p = points[i];
        outFile << p[0] << " " << p[1] << " " << p[2] << "\n";
    }

    // Write face data
    for (const auto &face : faces) {
        outFile << face.size();
        for(auto vid : face)
        {
            outFile << " " << vid;
        }
        outFile << "\n";
    }

    outFile.close();
    std::cout << "PLY file saved successfully: " << filename << std::endl;
}


void SaveMeshToPly(const std::string &filename, 
    const std::vector<std::array<double,3>> &points, 
    const std::vector<double> &points_quality, 
    const std::vector<std::vector<size_t>> &faces) {
    std::ofstream outFile(filename);
    if (!outFile) {
    std::cerr << "Error: Unable to open file for writing!\n";
    return;
    }

    // Write PLY header
    outFile << "ply\n";
    outFile << "format ascii 1.0\n";
    outFile << "element vertex " << points.size() << "\n";
    outFile << "property float x\n";
    outFile << "property float y\n";
    outFile << "property float z\n";
    outFile << "property float quality\n";
    outFile << "element face " << faces.size() << "\n";
    outFile << "property list uchar int vertex_indices\n";
    outFile << "end_header\n";

    // Write vertex data
    for (int i= 0; i < points.size(); ++i) {
        const auto& p = points[i];
        outFile << p[0] << " " << p[1] << " " << p[2] << " " << points_quality[i] << "\n";
    }

    // Write face data
    for (const auto &face : faces) {
    outFile << face.size();
    for(auto vid : face)
    {
    outFile << " " << vid;
    }
    outFile << "\n";
    }

    outFile.close();
    std::cout << "PLY file saved successfully: " << filename << std::endl;
}


void SaveMeshToPly(const std::string &filename, 
               const std::vector<std::array<double,3>> &points, 
               const std::vector<std::array<size_t, 3>> &faces) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error: Unable to open file for writing!\n";
        return;
    }

    // Write PLY header
    outFile << "ply\n";
    outFile << "format ascii 1.0\n";
    outFile << "element vertex " << points.size() << "\n";
    outFile << "property float x\n";
    outFile << "property float y\n";
    outFile << "property float z\n";
    outFile << "element face " << faces.size() << "\n";
    outFile << "property list uchar int vertex_indices\n";
    outFile << "end_header\n";

    // Write vertex data
    for (int i= 0; i < points.size(); ++i) {
        const auto& p = points[i];
        outFile << p[0] << " " << p[1] << " " << p[2] << "\n";
    }

    // Write face data
    for (const auto &face : faces) {
        outFile << "3 " <<  face[0] << " " << face[1] << " " << face[2] << std::endl;
    }

    outFile.close();
    std::cout << "PLY file saved successfully: " << filename << std::endl;
}

void GetTetEdges(const vector<array<size_t, 4>>& tets, std::vector<std::array<size_t, 2>>& edges)
{
    std::unordered_set<string> edge_keys;
    std::vector<std::array<size_t, 2>> tet_edges = {{0,1}, {0,2}, {0,3}, {1, 2}, {1, 3}, {2, 3}}; 
    for(const auto& tet : tets)
    {
        for(const auto& e : tet_edges)
        {
            size_t pa_id = tet[e[0]];  
            size_t pb_id = tet[e[1]];  
            std::array<size_t, 2> edge = pa_id < pb_id ? std::array<size_t, 2>{pa_id, pb_id} 
                                                        : std::array<size_t, 2>{pb_id, pa_id};
            string e_key = std::to_string(edge[0]) + "_" + std::to_string(edge[1]);
            if(edge_keys.find(e_key) == edge_keys.end())
            {
                edge_keys.insert(e_key);
                edges.push_back(edge);
            }
        }
    }
}


void SaveTetMeshToPly(const vector<array<double, 3>>& vertices, 
    const vector<array<size_t, 4>>& tets,
   const vector<double>& values, const std::string& filename)
{
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error: Unable to open file for writing!\n";
        return;
    }
    std::vector<std::array<size_t, 2>> edges;
    GetTetEdges(tets, edges);
    std::vector<std::string> v_color_strs; 
    vector<array<double, 3>> v_pts;
    vector<double> v_values;
    for (int i= 0; i < vertices.size(); ++i) {
        const auto& p = vertices[i];
        std::string color_str = values[i] > 0 ? " 255 0 0" : " 0 0 255";
        v_color_strs.push_back(color_str);
        v_pts.push_back(p);
        v_values.push_back(values[i]);
        // outFile << p[0] << " " << p[1] << " " << p[2] << color_str << std::endl;
    }
    std::vector<std::array<size_t, 3>> faces;  
    size_t p_count = vertices.size();
    for (int i= 0; i < edges.size(); ++i) {
        const auto& e = edges[i];
        double px = vertices[e[0]][0] / 2.0 +  vertices[e[1]][0] / 2.0;
        double py = vertices[e[0]][1] / 2.0 +  vertices[e[1]][1] / 2.0;
        double pz = vertices[e[0]][2] / 2.0 +  vertices[e[1]][2] / 2.0;
        
        // outFile << px << " " << py << " " << pz ;
        if(values[e[0]] * values[e[1]] <= 0)
        {
            double value = (values[e[0]] + values[e[1]]) /2.0;
            v_values.push_back(value);
            v_pts.push_back({px, py, pz});
            std::string color_str = " 0 255 0";
            v_color_strs.push_back(color_str);
            faces.push_back({e[0], e[1], p_count});
            p_count ++;
        }
    }
    
    // Write PLY header
    outFile << "ply\n";
    outFile << "format ascii 1.0\n";
    outFile << "element vertex " << vertices.size() + faces.size() << "\n";
    outFile << "property float x\n";
    outFile << "property float y\n";
    outFile << "property float z\n";
    outFile << "property uchar red\n";
    outFile << "property uchar green\n";
    outFile << "property uchar blue\n";
    outFile << "property float quality\n";
    outFile << "element face " << faces.size() << "\n";
    outFile << "property list uchar int vertex_indices\n";
    outFile << "end_header\n";

    for(size_t i = 0; i < v_pts.size(); ++i)
    {
        const auto& p = v_pts[i];
        const auto& color_str = v_color_strs[i];
        outFile << p[0] << " " << p[1] << " " << p[2] << color_str << " " << v_values[i]<< std::endl;
    }
    // Write vertex data
    // Write face data
    for (const auto &face : faces) {
        outFile << "3 " <<  face[0] << " " << face[1] << " " << face[2] << std::endl;
    }

    outFile.close();
    std::cout << "PLY file saved successfully: " << filename << std::endl;
}


void SavePointsWithQualityToPLY(const std::string& filename, 
    const std::vector<std::array<double,3>>& points,
    const std::vector<double>& qualtity) {
    //  const std::vector<Point>& points, const std::vector<Edge>& edges
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << points.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property float quality\n";
    file << "end_header\n";
    // Write vertex data
    for (int i =0; i < points.size(); ++i) {
        const auto& point = points[i];
        
        file << point[0] << " " << point[1] << " " << point[2] << " " << qualtity[i] << "\n";
    }

    file.close();
    std::cout << "PLY file saved: " << filename << std::endl;
}


// Save vertices and tetrahedra to text file
bool SaveTetToFile(const std::vector<std::array<double, 3>>& vertices, 
    const std::vector<std::array<size_t, 4>>& tet, const std::string& filename) {
    std::ofstream outFile(filename);
    
    if (!outFile.is_open()) {
        return false;
    }
    
    // Write number of vertices and tetrahedra as header
    outFile << vertices.size() << " " << tet.size() << "\n";
    
    // Write vertices section
    outFile << "# Vertices\n";
    for (const auto& vertex : vertices) {
        outFile << vertex[0] << " " 
                << vertex[1] << " " 
                << vertex[2] << "\n";
    }
    
    // Write tetrahedra section
    outFile << "# Tetrahedra\n";
    for (const auto& tetra : tet) {
        outFile << tetra[0] << " " 
                << tetra[1] << " " 
                << tetra[2] << " " 
                << tetra[3] << "\n";
    }
    
    outFile.close();
    return true;
}

// Read vertices and tetrahedra from text file
bool ReadTetFromFile(std::vector<std::array<double, 3>>& vertices, 
    std::vector<std::array<size_t, 4>>& tet, const std::string& filename) {
    std::ifstream inFile(filename);
    
    if (!inFile.is_open()) {
        return false;
    }
    
    // Clear existing data
    vertices.clear();
    tet.clear();
    
    std::string line;
    
    // Read header
    size_t numVertices, numTet;
    if (!std::getline(inFile, line)) {
        inFile.close();
        return false;
    }
    std::istringstream header(line);
    if (!(header >> numVertices >> numTet)) {
        inFile.close();
        throw std::runtime_error("Invalid header format");
    }
    
    // Read vertices section
    while (std::getline(inFile, line) && line != "# Tetrahedra") {
        if (line.empty() || line[0] == '#') continue;
        
        std::array<double, 3> vertex;
        std::istringstream iss(line);
        if (!(iss >> vertex[0] >> vertex[1] >> vertex[2])) {
            inFile.close();
            throw std::runtime_error("Invalid vertex format");
        }
        vertices.push_back(vertex);
    }
    
    // Read tetrahedra section
    while (std::getline(inFile, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::array<size_t, 4> tetra;
        std::istringstream iss(line);
        if (!(iss >> tetra[0] >> tetra[1] >> tetra[2] >> tetra[3])) {
            inFile.close();
            throw std::runtime_error("Invalid tetrahedron format");
        }
        tet.push_back(tetra);
    }
    
    // Verify counts match header
    if (vertices.size() != numVertices || tet.size() != numTet) {
        inFile.close();
        throw std::runtime_error("Data counts don't match header");
    }
    
    inFile.close();
    return true;
}