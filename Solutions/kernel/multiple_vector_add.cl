
// kernel:  vadd
// Purpose: Compute the elementwise sum d = a+b+c
// input: a, b and c float vectors of length size
// output: d float vector of length size holding the sum a + b + c
__kernel void multiple_vector_add(__global float const* const restrict a,
                                  __global float const* const restrict b,
                                  __global float const* const restrict c,
                                  __global float* const restrict d,
                                  size_t const size)
{
    size_t index = get_global_id(0);

    if (index >= size)
    {
        return;
    }
    d[index] = a[index] + b[index] + c[index];
}
