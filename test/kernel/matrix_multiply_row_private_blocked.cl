
__kernel void mmul(size_t const N,
                   __global float* A,
                   __global float* B,
                   __global float* C,
                   __local float* Bwrk)
{
    size_t const i = get_global_id(0);

    if (i >= N)
    {
        return;
    }

    float Awrk[1024];

    size_t const iloc = get_local_id(0);
    size_t const nloc = get_local_size(0);

    for (size_t k = 0; k < N; k++)
    {
        Awrk[k] = A[i * N + k];
    }

    for (size_t j = 0; j < N; j++)
    {
        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t k = iloc; k < N; k += nloc)
        {
            Bwrk[k] = B[k * N + j];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        float tmp = 0.0f;

        for (size_t k = 0; k < N; k++)
        {
            tmp += Awrk[k] * Bwrk[k];
        }

        C[i * N + j] = tmp;

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    L
}
