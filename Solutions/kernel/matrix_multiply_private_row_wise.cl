
__kernel void mmul(size_t const N,
                   __global float const* const restrict A,
                   __global float const* const restrict B,
                   __global float* const restrict C)
{
    size_t index = get_global_id(0);

    if (index >= N)
    {
        return;
    }

    float Awrk[1024];

    for (size_t k = 0; k < N; k++)
    {
        Awrk[k] = A[index * N + k];
    }

    for (size_t j = 0; j < N; j++)
    {
        float tmp = 0.0f;

        for (size_t k = 0; k < N; k++)
        {
            tmp += Awrk[k] * B[k * N + j];
        }
        C[index * N + j] = tmp;
    }
}
