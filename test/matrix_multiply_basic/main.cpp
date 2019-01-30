//------------------------------------------------------------------------------
//
//  PROGRAM: Matrix Multiplication driver
//
//  PURPOSE: This is a driver program to test various ways of computing
//           the product:
//
//                C  = A * B
//
//           A and B are set to constant matrices so we
//           can make a quick test of the multiplication.
//
//  USAGE:   The matrices are constant matrices, square and the order is
//           set as a constant, ORDER (see mult.h).
//
//  HISTORY: Written by Tim Mattson, August 2010
//           Modified by Simon McIntosh-Smith, September 2011
//           Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
//           Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//           Modified to assume square matrices by Simon McIntosh-Smith, Sep 2014
//
//------------------------------------------------------------------------------

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "matrix_lib.hpp"
#include "load_source.hpp"
#include "err_code.h"
#include "device_picker.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
    try
    {
        // Create a context and queue
        cl_uint deviceIndex = 1;
        parseArguments(argc, argv, &deviceIndex);

        // Get list of devices
        std::vector<cl::Device> devices;
        unsigned numDevices = getDeviceList(devices);

        // Check device index in range
        if (deviceIndex >= numDevices)
        {
            std::cout << "Invalid device index (try '--list')\n";
            return EXIT_FAILURE;
        }

        cl::Device device = devices[deviceIndex];

        std::string name;
        getDeviceName(device, name);
        std::cout << "\nUsing OpenCL device: " << name << "\n";

        std::vector<cl::Device> chosen_device;
        chosen_device.push_back(device);

        cl::Context context(chosen_device);

        cl::CommandQueue queue(context, device);

        std::size_t const N = ORDER;
        std::size_t const size = N * N;

        std::cout << "\n===== Sequential, matrix mult (dot prod), order " << N
                  << " on host CPU ======\n";

        // Host memory for Matrix A B and C
        std::vector<float> h_A(size, 3.0f), h_B(size, 5.0f), h_C(size);

        for (int i = 0; i < COUNT; i++)
        {
            std::fill(begin(h_C), end(h_C), 0.0f);

            auto const start_time = std::chrono::steady_clock::now();

            seq_mat_mul_sdot(N, h_A, h_B, h_C);

            std::chrono::duration<double> const run_time = std::chrono::steady_clock::now()
                                                           - start_time;
            results(N, h_C, run_time.count());
        }

        // Setup the matrices in device buffers, initialize matrices, and write them into global memory
        cl::Buffer d_a(context, begin(h_A), end(h_A), true);
        cl::Buffer d_b(context, begin(h_B), end(h_B), true);
        cl::Buffer d_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * size);

        // OpenCL matrix multiplication ... naive

        // Create the compute program from the source buffer
        cl::Program program(context, load_source("../test/kernel/matrix_multiply_basic.cl"), true);

        // Create the compute kernel from the program
        // clang-format off
        cl::make_kernel<std::size_t, cl::Buffer, cl::Buffer, cl::Buffer> naive_mmul(program,
                                                                                    "matrix_multiply_basic");
        // clang-format on
        std::cout << "\n===== OpenCL, matrix mult, C(i,j) per work item, order " << N << " ======\n";

        // Do the multiplication COUNT times
        for (int i = 0; i < COUNT; i++)
        {
            std::fill(begin(h_C), end(h_C), 0.0f);

            auto const start_time = std::chrono::steady_clock::now();

            // Execute the kernel over the entire range of C matrix elements ... computing
            // a dot product for each element of the product matrix.  The local work
            // group size is set to NULL ... so I'm telling the OpenCL runtime to
            // figure out a local work group size for me.
            cl::NDRange global(N, N);
            naive_mmul(cl::EnqueueArgs(queue, global), N, d_a, d_b, d_c);

            queue.finish();

            std::chrono::duration<double> const run_time = std::chrono::steady_clock::now()
                                                           - start_time;

            cl::copy(queue, d_c, begin(h_C), end(h_C));

            results(N, h_C, run_time.count());
        }
    }
    catch (cl::Error const& err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")" << std::endl;
        return 1;
    }
    return 0;
}
