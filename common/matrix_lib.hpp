//------------------------------------------------------------------------------
//
//  PROGRAM: Matrix library include file (function prototypes)
//
//  HISTORY: Written by Tim Mattson, August 2010
//           Modified by Simon McIntosh-Smith, September 2011
//           Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
//           Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//           Modified to assume square matrices by Simon McIntosh-Smith, Sep 2014
//
//------------------------------------------------------------------------------

#pragma once

#include <vector>

/// compute the matrix product (sequential algorithm, dot producdt)
void seq_mat_mul_sdot(int const N,
                      std::vector<float> const& A,
                      std::vector<float> const& B,
                      std::vector<float>& C);

/// initialize the input matrices A and B
void initmat(int N, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C);

/// fill Btrans(Mdim,Pdim)  with transpose of B(Pdim,Mdim)
void trans(int N, std::vector<float> const& B, std::vector<float>& Btrans);

/// compute errors of the product matrix
float error(int N, std::vector<float> const& C);

/// analyze and output results
void results(int const N, std::vector<float> const& C, double const run_time);

// Order of the square matrices A, B, and C
constexpr int ORDER = 1024;
// Max dim for NDRange
constexpr int DIM = 2;
// number of times to do each multiplication
constexpr int COUNT = 1;
