
__kernel void mmul(const int N, __global float* A, __global float* B, __global float* C)
{
    size_t i = get_global_id(0);

    if (i >= N)
    {
        return;
    }

    float Awrk[1024];

    for (size_t k = 0; k < N; k++)
    {
        Awrk[k] = A[i * N + k];
    }

    for (size_t j = 0; j < N; j++)
    {
        float tmp = 0.0f;

        for (size_t k = 0; k < N; k++)
        {
            tmp += Awrk[k] * B[k * N + j];
        }
        C[i * N + j] = tmp;
    }
}
