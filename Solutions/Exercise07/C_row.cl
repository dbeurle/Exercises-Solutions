
__kernel void mmul(const int N,
                   __global float* restrict A,
                   __global float* restrict B,
                   __global float* restrict C)
{
    size_t index = get_global_id(0);

    if (index >= N)
    {
        return;
    }

    for (size_t j = 0; j < N; j++)
    {
        float tmp = 0.0f;
        for (size_t k = 0; k < N; k++)
        {
            tmp += A[index * N + k] * B[k * N + j];
        }
        C[index * N + j] = tmp;
    }
}
