/*

This program will numerically compute the integral of

                  4/(1+x*x)

from 0 to 1.  The value of this integral is pi -- which
is great since it gives us an easy way to check the answer.

The is the original sequential program.  It uses the timer
from the OpenMP runtime library

History: Written by Tim Mattson, 11/99.
         Ported to C++ by Tom Deakin, August 2013

*/

#include "util.hpp"

#include <cstdio>
#include <cmath>

int main()
{
    constexpr long num_steps = 100'000'000;

    double step = 1.0 / static_cast<double>(num_steps);

    util::Timer timer;

    double sum = 0.0;

    for (int i = 1; i <= num_steps; i++)
    {
        sum += 4.0 / (1.0 + std::pow((i - 0.5) * step, 2));
    }

    double pi = step * sum;
    double run_time = timer.getTimeMilliseconds() / 1000.0;
    printf("\n pi with %ld steps is %lf in %lf seconds\n", num_steps, pi, run_time);
}
