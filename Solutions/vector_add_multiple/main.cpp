//------------------------------------------------------------------------------
//
// Name:       vadd_three.cpp
//
// Purpose:    Elementwise addition of three vectors (d = a + b + c)
//
//                   d = a + b + c
//
// HISTORY:    Written by Tim Mattson, June 2011
//             Ported to C++ Wrapper API by Benedict Gaster, September 2011
//             Updated to C++ Wrapper API v1.2 by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//             Updated by Tom Deakin, October 2014
//
//------------------------------------------------------------------------------

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>

#include "load_source.hpp"
#include "err_code.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

// tolerance used in floating point comparisons
constexpr float TOL = 0.001f;
// length of vectors a, b, and c
constexpr std::size_t LENGTH = 1024;

int main()
{
    try
    {
        // Create a context
        cl::Context context(DEVICE);

        // Load in kernel source, creating a program object for the context
        cl::Program program(context, load_source("../Solutions/kernel/multiple_vector_add.cl"), true);

        // Get the command queue
        cl::CommandQueue queue(context);

        // Create the kernel functor
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, size_t> vadd(program,
                                                                                     "multiple_"
                                                                                     "vector_add");

        std::vector<float> h_a(LENGTH), h_b(LENGTH), h_c(LENGTH);
        // Fill vectors with random floats
        {
            std::mt19937 mersenne_engine{std::random_device{}()};

            std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

            std::generate(begin(h_a), end(h_a), [&]() { return distribution(mersenne_engine); });
            std::generate(begin(h_b), end(h_b), [&]() { return distribution(mersenne_engine); });
            std::generate(begin(h_c), end(h_c), [&]() { return distribution(mersenne_engine); });
        }

        cl::Buffer d_a(context, begin(h_a), end(h_a), true);
        cl::Buffer d_b(context, begin(h_b), end(h_b), true);
        cl::Buffer d_c(context, begin(h_c), end(h_c), true);

        cl::Buffer d_d(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

        auto const start_time = std::chrono::steady_clock::now();

        vadd(cl::EnqueueArgs(queue, cl::NDRange(LENGTH)), d_a, d_b, d_c, d_d, LENGTH);

        auto const end_time = std::chrono::steady_clock::now();

        std::chrono::duration<double> const elapsed_time = end_time - start_time;

        std::cout << "kernel took " << elapsed_time.count() << " seconds to run\n";

        // d vector (result)
        std::vector<float> h_d(LENGTH, 0xdeadbeef);
        cl::copy(queue, d_d, begin(h_d), end(h_d));

        // Test the results
        int correct = 0;
        for (int i = 0; i < LENGTH; i++)
        {
            float tmp = h_a[i] + h_b[i] + h_c[i] - h_d[i];

            if (std::pow(tmp, 2) < std::pow(TOL, 2))
            {
                ++correct;
            }
            else
            {
                std::cout << " tmp " << tmp << " h_a " << h_a[i] << " h_b " << h_b[i] << " h_c "
                          << h_c[i] << " h_d " << h_d[i] << "\n";
            }
        }
        std::cout << "C = A + B + C: " << correct << " out of " << LENGTH
                  << " results were correct.\n";
    }
    catch (cl::Error const& err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")" << std::endl;
        return 1;
    }
    return 0;
}
