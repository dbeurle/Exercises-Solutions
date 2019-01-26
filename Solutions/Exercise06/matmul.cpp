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
#include "util.hpp"
#include "err_code.h"
#include "device_picker.hpp"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

std::string const
    kernel_source = "__kernel void mmul(                                                    \n"
                    "   const int N,                                                        \n"
                    "   __global float* A,                                                  \n"
                    "   __global float* B,                                                  \n"
                    "   __global float* C)                                                  \n"
                    "{                                                                      \n"
                    "   int k;                                                              \n"
                    "   size_t i = get_global_id(0);                                        \n"
                    "   size_t j = get_global_id(1);                                        \n"
                    "   if ( (i < N) && (j <N))                                             \n"
                    "   {                                                                   \n"
                    "       float tmp = 0.0f;                                               \n"
                    "       for(k=0;k<N;k++)                                                \n"
                    "           tmp += A[i*N+k] * B[k*N+j];                                 \n"
                    "       C[i*N+j] = tmp;                                                 \n"
                    "   }                                                                   \n"
                    "}                                                                      \n"
                    "\n";

int main(int argc, char* argv[])
{
    int const N = ORDER;
    int const size = N * N;

    try
    {
        // Host memory for Matrix A B and C
        std::vector<float> h_A(size, 3.0f), h_B(size, 5.0f), h_C(size, 0.0f);
        // Matrices in device memory
        cl::Buffer d_a, d_b, d_c;

        // Create a context and queue
        cl_uint deviceIndex = 0;
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

        std::cout << "\n===== Sequential, matrix mult (dot prod), order " << N
                  << " on host CPU ======\n";

        for (int i = 0; i < COUNT; i++)
        {
            std::fill(begin(h_C), end(h_C), 0.0f);

            auto const start_time = std::chrono::steady_clock::now();

            seq_mat_mul_sdot(N, h_A, h_B, h_C);

            std::chrono::duration<double> const run_time = std::chrono::steady_clock::now()
                                                           - start_time;
            results(N, h_C, run_time.count());
        }

        // Setup the buffers, initialize matrices, and write them into global memory
        d_a = cl::Buffer(context, begin(h_A), end(h_A), true);
        d_b = cl::Buffer(context, begin(h_B), end(h_B), true);
        d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size);

        // OpenCL matrix multiplication ... Naive

        // Create the compute program from the source buffer
        cl::Program program(context, kernel_source, true);

        // Create the compute kernel from the program
        cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> naive_mmul(program, "mmul");

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
