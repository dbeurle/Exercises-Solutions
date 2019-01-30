//
// Pi reduction - vectorized
//
// Numeric integration to estimate pi
// Asks the user to select a device at runtime
// Vector size must be present as a CLI argument
//
// History: C version written by Tim Mattson, May 2010
//          Ported to the C++ Wrapper API by Benedict R. Gaster, September 2011
//          C++ version Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
//          Updated by Tom Deakin, September 2013

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>

#include "load_source.hpp"
#include "err_code.h"

#include <chrono>
#include <fstream>
#include <numeric>
#include <iostream>
#include <vector>

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: ./pi_vectorised num\n"
                  << "\twhere num = 1, 4 or 8\n";
        return 1;
    }

    int vector_size = atoi(argv[1]);

    // Define some vector size specific constants
    std::uint32_t iterations, work_group_size;
    if (vector_size == 1)
    {
        iterations = 262144;
        work_group_size = 8;
    }
    else if (vector_size == 4)
    {
        iterations = 262144 / 4;
        work_group_size = 32;
    }
    else if (vector_size == 8)
    {
        iterations = 262144 / 8;
        work_group_size = 64;
    }
    else
    {
        std::cerr << "Invalid vector size\n";
        return 1;
    }

    // Set some default values:
    // Default number of steps (updated later to device preferable)
    constexpr std::size_t in_nsteps = 512 * 512 * 512;
    // Default number of iterations
    std::size_t const niters = iterations;

    try
    {
        // Create context, queue and build program
        cl::Context context(DEVICE);
        cl::CommandQueue queue(context);
        cl::Program program(context, load_source("../test/kernel/pi_vocl.cl"), true);
        cl::Kernel kernel;

        // Now that we know the size of the work_groups, we can set the number of work
        // groups, the actual number of steps, and the step size
        unsigned int nwork_groups = in_nsteps / (work_group_size * niters);

        // Get the max work group size for the kernel pi on our device
        unsigned int max_size;
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        if (vector_size == 1)
        {
            kernel = cl::Kernel(program, "pi");
            max_size = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(devices[0]);
        }
        else if (vector_size == 4)
        {
            kernel = cl::Kernel(program, "pi_vec4");
            max_size = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(devices[0]);
        }
        else if (vector_size == 8)
        {
            kernel = cl::Kernel(program, "pi_vec8");
            max_size = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(devices[0]);
        }

        if (max_size > work_group_size)
        {
            work_group_size = max_size;
            nwork_groups = in_nsteps / (nwork_groups * niters);
        }

        if (nwork_groups < 1)
        {
            nwork_groups = devices[0].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
            work_group_size = in_nsteps / (nwork_groups * niters);
        }

        unsigned int nsteps = work_group_size * niters * nwork_groups;
        float step_size = 1.0f / (float)nsteps;

        std::cout << nwork_groups << " work groups of size " << work_group_size << ".\n"
                  << nsteps << " Integration steps\n";

        cl::Buffer d_partial_sums(context, CL_MEM_WRITE_ONLY, sizeof(float) * nwork_groups);

        // Start the timer
        auto const start = std::chrono::steady_clock::now();

        // Execute the kernel over the entire range of our 1d input data et
        // using the maximum number of work group items for this device
        cl::NDRange global(nwork_groups * work_group_size);
        cl::NDRange local(work_group_size);

        cl::LocalSpaceArg localmem = cl::Local(sizeof(float) * work_group_size);

        kernel.setArg(0, niters);
        kernel.setArg(1, step_size);
        kernel.setArg(2, localmem);
        kernel.setArg(3, d_partial_sums);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

        // Vector to hold partial sum
        std::vector<float> h_psum(nwork_groups);
        cl::copy(queue, d_partial_sums, h_psum.begin(), h_psum.end());

        // Complete the sum and compute the final integral value
        auto const pi_res = std::accumulate(begin(h_psum), end(h_psum), 0.0f) * step_size;

        auto const end = std::chrono::steady_clock::now();

        std::chrono::duration<double> const elapsed = end - start;

        // Stop the timer
        std::cout << "The calculation ran in " << elapsed.count() << " seconds\n"
                  << " pi = " << pi_res << " for " << nsteps << " steps\n";

        return 0;
    }
    catch (cl::Error const& err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")" << std::endl;
        return 1;
    }
}
