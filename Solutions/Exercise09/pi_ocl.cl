
// kernel:  pi
// Purpose: accumulate partial sums of pi comp
// input: float step_size
//        int   niters per work item
//        local float* an array to hold sums from each work item
// output: partial_sums   float vector of partial sums
void reduce(__local float*, __global float*);

__kernel void pi(const int niters,
                 const float step_size,
                 __local float* local_sums,
                 __global float* partial_sums)
{
    size_t const num_work_items = get_local_size(0);
    size_t const local_id = get_local_id(0);
    size_t const group_id = get_group_id(0);

    int istart = (group_id * num_work_items + local_id) * niters;
    int iend = istart + niters;

    float accum = 0.0f;

    for (int i = istart; i < iend; i++)
    {
        float x = (i + 0.5f) * step_size;
        accum += 4.0f / (1.0f + x * x);
    }

    local_sums[local_id] = accum;
    barrier(CLK_LOCAL_MEM_FENCE);

    reduce(local_sums, partial_sums);
}

// OpenCL function:  reduction
// Purpose: reduce across all the work-items in a work-group
// input: local float* an array to hold sums from each work item
// output: global float* partial_sums   float vector of partial sums
void reduce(__local float* local_sums, __global float* partial_sums)
{
    if (get_local_id(0) == 0)
    {
        size_t num_work_items = get_local_size(0);
        size_t group_id = get_group_id(0);

        float sum = 0.0f;

        for (int i = 0; i < num_work_items; i++)
        {
            sum += local_sums[i];
        }

        partial_sums[group_id] = sum;
    }
}
