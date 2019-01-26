//------------------------------------------------------------------------------
//
//  Include fle for the Matrix Multiply test harness
//
//  HISTORY: Written by Tim Mattson, August 2010
//           Modified by Simon McIntosh-Smith, September 2011
//           Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
//           Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//
//------------------------------------------------------------------------------

#ifndef __MULT_HDR
#define __MULT_HDR

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "util.hpp"
#include "matrix_lib.hpp"

//------------------------------------------------------------------------------
//  Constants
//------------------------------------------------------------------------------
#define ORDER 1024  // Order of the square matrices A, B, and C
#define TOL (0.001) // tolerance used in floating point comparisons
#define DIM 2       // Max dim for NDRange
#define COUNT 1     // number of times to do each multiplication
#define SUCCESS 1
#define FAILURE 0

#endif