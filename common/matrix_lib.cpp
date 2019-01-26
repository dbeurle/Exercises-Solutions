//------------------------------------------------------------------------------
//
//  PROGRAM: Matrix library for the multiplication driver
//
//  PURPOSE: This is a simple set of functions to manipulate
//           matrices used with the multiplcation driver.
//
//  USAGE:   The matrices are square and the order is
//           set as a defined constant, ORDER.
//
//  HISTORY: Written by Tim Mattson, August 2010
//           Modified by Simon McIntosh-Smith, September 2011
//           Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
//           Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//           Modified to assume square matrices by Simon McIntosh-Smith, Sep 2014
//
//------------------------------------------------------------------------------

#include "matrix_lib.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

// tolerance used in floating point comparisons
constexpr float TOL = 0.001f;

void seq_mat_mul_sdot(int N, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float tmp = 0.0f;
            for (int k = 0; k < N; k++)
            {
                /* C(i,j) = sum(over k) A(i,k) * B(k,j) */
                tmp += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = tmp;
        }
    }
}

void initmat(int N, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C)
{
    std::fill(begin(A), end(A), 3.0f);
    std::fill(begin(B), end(B), 5.0f);
    std::fill(begin(C), end(C), 0.0f);
}

void zero_mat(int N, std::vector<float>& C) { std::fill(begin(C), end(C), 0.0f); }

void trans(int N, std::vector<float>& B, std::vector<float>& Btrans)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            Btrans[j * N + i] = B[i * N + j];
        }
    }
}

float error(int N, std::vector<float> const& C)
{
    float const cval = N * 15.0f;

    float errsq = 0.0f;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            errsq += std::pow(C[i * N + j] - cval, 2);
        }
    }
    return errsq;
}

void results(int const N, std::vector<float> const& C, double const run_time)
{
    float const mflops = 2 * std::pow(N, 3) / (1E6 * run_time);

    std::cout << " " << run_time << " seconds at " << mflops << " MFLOPS\n";

    float const errsq = error(N, C);

    if (std::isnan(errsq) || errsq > TOL)
    {
        std::cout << "\n Errors in multiplication: " << errsq << "\n";
    }
}
