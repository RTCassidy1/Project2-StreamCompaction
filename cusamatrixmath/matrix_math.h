#ifndef MAIN_H
#define MAIN_H

#include <stdlib.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <thrust\scatter.h>
#include "glslUtility.h"


using namespace std;



//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv);

//-------------------------------
//----------function declarations----------
//-------------------------------
void seq_scan(float*,float*, int);
void print_array(float*, int);
void naive_scan( float*, float*, float*, float*, int);
void shared_scan( float*, float*, int);
void seq_scatter(float*, float*,  float*, int);
void string_compaction( float*, float*, int);
__global__ void glob_sec_scan( float*, float*, float*, float*,float*, int);
__global__ void shared_naive_scan( float*, float*, float*, int);
__global__ void add_segments( float*, float*,int);
__global__ void glob_excl_block_scan( float*, float*, float*, int);
__global__ void shared_block_scan( float*, float*, int);
__global__ void make_bool_arr( float*, float*, int);
__global__ void scatter( float*, float*, float*, int);


#endif
