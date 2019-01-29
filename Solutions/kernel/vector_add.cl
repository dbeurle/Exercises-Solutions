
/// kernel:  vector_add
/// Purpose: Compute the elementwise sum c = a+b
/// input: a and b float vectors of length size
/// output: c float vector of length size holding the sum a + b
__kernel void vector_add(__global float* const restrict a,
                         __global float* const restrict b,
                         __global float* const restrict c,
                         size_t const size)
{
    size_t index = get_global_id(0);

    if (index >= size)
    {
        return;
    }
    c[index] = a[index] + b[index];
}
