//------------------------------------------------------------------------------
//
// Name:       vadd_chain.cpp
//
// Purpose:    Elementwise addition of two vectors (c = a + b)
//
//                   c = a + b
//
// HISTORY:    Written by Tim Mattson, June 2011
//             Ported to C++ Wrapper API by Benedict Gaster, September 2011
//             Updated to C++ Wrapper API v1.2 by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//
//------------------------------------------------------------------------------

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "util.hpp"
#include "err_code.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <random>
#include <iostream>
#include <fstream>
#include <vector>

// tolerance used in floating point comparisons
constexpr float TOL = 0.001f;
// length of vectors a, b, and c
constexpr int LENGTH = 1024;

int main()
{
    std::vector<float> h_a(LENGTH);             // a vector
    std::vector<float> h_b(LENGTH);             // b vector
    std::vector<float> h_c(LENGTH, 0xdeadbeef); // c vector (result)
    std::vector<float> h_d(LENGTH, 0xdeadbeef); // d vector (result)
    std::vector<float> h_e(LENGTH);             // e vector
    std::vector<float> h_f(LENGTH, 0xdeadbeef); // f vector (result)
    std::vector<float> h_g(LENGTH);             // g vector

    cl::Buffer d_a; // device memory used for the input a vector
    cl::Buffer d_b; // device memory used for the input b vector
    cl::Buffer d_c; // device memory used for the output c vector
    cl::Buffer d_d; // device memory used for the output d vector
    cl::Buffer d_e; // device memory used for the input e vector
    cl::Buffer d_f; // device memory used for the output f vector
    cl::Buffer d_g; // device memory used for the input g vector

    // Fill vectors a and b with random float values
    {
        std::mt19937 mersenne_engine{std::random_device{}()};

        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

        std::generate(begin(h_a), end(h_a), [&]() { return distribution(mersenne_engine); });
        std::generate(begin(h_b), end(h_b), [&]() { return distribution(mersenne_engine); });
        std::generate(begin(h_e), end(h_e), [&]() { return distribution(mersenne_engine); });
        std::generate(begin(h_g), end(h_g), [&]() { return distribution(mersenne_engine); });
    }

    try
    {
        // Create a context
        cl::Context context(DEVICE);

        // Load in kernel source, creating a program object for the context
        cl::Program program(context, util::loadProgram("vadd_chain.cl"), true);

        // Get the command queue
        cl::CommandQueue queue(context);

        // Create the kernel functor
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, size_t> vadd(program, "vadd");

        d_a = cl::Buffer(context, begin(h_a), end(h_a), true);
        d_b = cl::Buffer(context, begin(h_b), end(h_b), true);
        d_e = cl::Buffer(context, begin(h_e), end(h_e), true);
        d_g = cl::Buffer(context, begin(h_g), end(h_g), true);

        d_c = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * LENGTH);
        d_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * LENGTH);
        d_f = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

        vadd(cl::EnqueueArgs(queue, cl::NDRange(LENGTH)), d_a, d_b, d_c, LENGTH);
        vadd(cl::EnqueueArgs(queue, cl::NDRange(LENGTH)), d_e, d_c, d_d, LENGTH);
        vadd(cl::EnqueueArgs(queue, cl::NDRange(LENGTH)), d_g, d_d, d_f, LENGTH);

        cl::copy(queue, d_f, begin(h_f), end(h_f));

        // Test the results
        int correct = 0;
        for (int i = 0; i < LENGTH; i++)
        {
            // assign element i of a+b+e+g to tmp
            float const tmp = h_a[i] + h_b[i] + h_e[i] + h_g[i] - h_f[i];

            if (std::pow(tmp, 2) < std::pow(TOL, 2))
            {
                // correct if square deviation is less than tolerance squared
                correct++;
            }
            else
            {
                printf(" tmp %f h_a %f h_b %f h_e %f h_g %f h_f %f\n",
                       tmp,
                       h_a[i],
                       h_b[i],
                       h_e[i],
                       h_g[i],
                       h_f[i]);
            }
        }
        // summarize results
        printf("C = A + B + E + G:  %d out of %d results were correct.\n", correct, LENGTH);
    }
    catch (cl::Error const& err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")" << std::endl;
    }
}
