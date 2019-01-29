//------------------------------------------------------------------------------
//
// Name:       vadd_cpp.cpp
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

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

// tolerance used in floating point comparisons
constexpr double TOL = 0.001;
// length of vectors a, b, and c
constexpr int LENGTH = 1024;

int main(void)
{
    std::vector<float> h_a(LENGTH), h_b(LENGTH), h_c(LENGTH, 0xdeadbeef);

    // device memory used for the input a and b vector and the output c vector
    cl::Buffer d_a, d_b, d_c;

    // Fill vectors a and b with random float values
    auto const count = LENGTH;
    for (int i = 0; i < count; i++)
    {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    try
    {
        // Create a context
        cl::Context context(DEVICE);

        // Load in kernel source, creating a program object for the context
        cl::Program program(context, load_source("vadd.cl"), true);

        // Get the command queue
        cl::CommandQueue queue(context);

        // Create the kernel functor
        auto vadd = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int>(program, "vadd");

        d_a = cl::Buffer(context, begin(h_a), end(h_a), true);
        d_b = cl::Buffer(context, begin(h_b), end(h_b), true);
        d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

        util::Timer timer;

        vadd(cl::EnqueueArgs(queue, cl::NDRange(count)), d_a, d_b, d_c, count);

        queue.finish();

        printf("\nThe kernels ran in %lf seconds\n", timer.getTimeMilliseconds() / 1000.0);

        cl::copy(queue, d_c, begin(h_c), end(h_c));

        // Test the results
        int correct = 0;
        for (int i = 0; i < count; i++)
        {
            if (std::pow(h_a[i] + h_b[i] - h_c[i], 2) < std::pow(TOL, 2))
            {
                ++correct;
            }
            else
            {
                printf(" tmp %f h_a %f h_b %f  h_c %f \n", tmp, h_a[i], h_b[i], h_c[i]);
            }
        }
        // summarize results
        printf("vector add to find C = A+B:  %d out of %d results were correct.\n", correct, count);
    }
    catch (cl::Error const& err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")" << std::endl;
    }
}
