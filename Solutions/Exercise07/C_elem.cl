
__kernel void mmul(int const N, __global float* A, __global float* B, __global float* C)
{
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);

    if (i < N && j < N)
    {
        int k;
        float tmp = 0.0f;
        for (k = 0; k < N; k++) tmp += A[i * N + k] * B[k * N + j];
        C[i * N + j] = tmp;
    }
}
