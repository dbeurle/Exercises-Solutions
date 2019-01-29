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

#include "load_source.hpp"
#include "err_code.h"

#include <algorithm>
#include <chrono>
#include <random>
#include <iostream>
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
        cl::Program program(context, load_source("../Solutions/kernel/vector_add.cl"), true);

        // Get the command queue
        cl::CommandQueue queue(context);

        // Create the kernel functor
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, size_t> vector_add(program, "vector_add");

        std::vector<float> h_f(LENGTH, 0xdeadbeef); // f vector (result)

        std::vector<float> h_a(LENGTH); // a vector
        std::vector<float> h_b(LENGTH); // b vector
        std::vector<float> h_e(LENGTH); // e vector
        std::vector<float> h_g(LENGTH); // g vector

        // Fill vectors with random floats
        {
            std::mt19937 mersenne_engine{std::random_device{}()};

            std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

            std::generate(begin(h_a), end(h_a), [&]() { return distribution(mersenne_engine); });
            std::generate(begin(h_b), end(h_b), [&]() { return distribution(mersenne_engine); });
            std::generate(begin(h_e), end(h_e), [&]() { return distribution(mersenne_engine); });
            std::generate(begin(h_g), end(h_g), [&]() { return distribution(mersenne_engine); });
        }

        cl::Buffer d_a(context, begin(h_a), end(h_a), true);
        cl::Buffer d_b(context, begin(h_b), end(h_b), true);
        cl::Buffer d_e(context, begin(h_e), end(h_e), true);
        cl::Buffer d_g(context, begin(h_g), end(h_g), true);

        cl::Buffer d_c(context, CL_MEM_READ_WRITE, sizeof(float) * LENGTH);
        cl::Buffer d_d(context, CL_MEM_READ_WRITE, sizeof(float) * LENGTH);
        cl::Buffer d_f(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

        auto const start_time = std::chrono::steady_clock::now();

        vector_add(cl::EnqueueArgs(queue, cl::NDRange(LENGTH)), d_a, d_b, d_c, LENGTH);
        vector_add(cl::EnqueueArgs(queue, cl::NDRange(LENGTH)), d_e, d_c, d_d, LENGTH);
        vector_add(cl::EnqueueArgs(queue, cl::NDRange(LENGTH)), d_g, d_d, d_f, LENGTH);

        auto const end_time = std::chrono::steady_clock::now();

        std::chrono::duration<double> const elapsed_time = end_time - start_time;

        std::cout << "kernels took " << elapsed_time.count() << " seconds to run\n";

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
                ++correct;
            }
            else
            {
                std::cout << " tmp " << tmp << " h_a " << h_a[i] << " h_b " << h_b[i] << " h_e "
                          << h_e[i] << " h_g " << h_g[i] << " h_f " << h_f[i] << "\n";
            }
        }
        // summarize results
        std::cout << "C = A + B + E + G: " << correct << " out of " << LENGTH
                  << " results were correct.\n";
    }
    catch (cl::Error const& err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")" << std::endl;
    }
}
