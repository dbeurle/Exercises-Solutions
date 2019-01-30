
// Name:       pi_ocl.cpp
// Purpose:    Numeric integration to estimate pi
// HISTORY:    Written by Tim Mattson, May 2010
//             Ported to the C++ Wrapper API by Benedict R. Gaster, September 2011
//             Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>

#include <numeric>
#include <chrono>
#include <string>
#include <vector>
#include <iostream>

#include "load_source.hpp"
#include "err_code.h"
#include "device_picker.hpp"

int main(int argc, char* argv[])
{
    // default number of steps (updated later to device preferable)
    constexpr std::size_t in_nsteps = 512 * 512 * 512;
    // number of iterations
    constexpr std::size_t niters = 262144;

    try
    {
        cl_uint deviceIndex = 1;
        parseArguments(argc, argv, &deviceIndex);

        // Get list of devices
        std::vector<cl::Device> devices;
        auto const numDevices = getDeviceList(devices);

        // Check device index in range
        if (deviceIndex >= numDevices)
        {
            std::cout << "Invalid device index (try '--list')\n";
            return 1;
        }

        cl::Device device = devices[deviceIndex];

        std::string name;
        getDeviceName(device, name);
        std::cout << "\nUsing OpenCL device: " << name << "\n";

        std::vector<cl::Device> chosen_device;
        chosen_device.push_back(device);
        cl::Context context(chosen_device);
        cl::CommandQueue queue(context, device);

        // Create the program object
        cl::Program program(context, load_source("../test/kernel/pi_ocl.cl"), true);

        // Create the kernel object for quering information
        cl::Kernel ko_pi(program, "pi");

        cl::make_kernel<int, float, cl::LocalSpaceArg, cl::Buffer> pi(program, "pi");

        // Get the work group size
        auto work_group_size = ko_pi.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

        // Now that we know the size of the work_groups, we can set the number of work
        // groups, the actual number of steps, and the step size
        auto nwork_groups = in_nsteps / (work_group_size * niters);

        if (nwork_groups < 1)
        {
            nwork_groups = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
            work_group_size = in_nsteps / (nwork_groups * niters);
        }

        auto const nsteps = work_group_size * niters * nwork_groups;
        float const step_size = 1.0f / static_cast<float>(nsteps);

        std::cout << " " << nwork_groups << " work groups of size " << work_group_size << ".  "
                  << nsteps << " Integration steps\n";

        cl::Buffer d_partial_sums(context, CL_MEM_WRITE_ONLY, sizeof(float) * nwork_groups);

        auto const start = std::chrono::steady_clock::now();

        // Execute the kernel over the entire range of our 1d input data set
        // using the maximum number of work group items for this device
        pi(cl::EnqueueArgs(queue, cl::NDRange(nsteps / niters), cl::NDRange(work_group_size)),
           niters,
           step_size,
           cl::Local(sizeof(float) * work_group_size),
           d_partial_sums);

        std::vector<float> h_psum(nwork_groups);
        cl::copy(queue, d_partial_sums, begin(h_psum), end(h_psum));

        // complete the sum and compute final integral value
        auto const pi_res = std::accumulate(begin(h_psum), end(h_psum), 0.0f) * step_size;

        auto const end = std::chrono::steady_clock::now();
        std::chrono::duration<double> const elapsed = end - start;

        std::cout << "\nThe calculation ran in " << elapsed.count() << " seconds\n";
        std::cout << " pi = " << pi_res << " for " << nsteps << " steps\n";
    }
    catch (cl::Error const& err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")" << std::endl;
    }
}
