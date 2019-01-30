
__kernel void matrix_multiply_basic(size_t const N,
                                    __global float const* const restrict A,
                                    __global float const* const restrict B,
                                    __global float* const restrict C)
{
    size_t row = get_global_id(0);
    size_t col = get_global_id(1);

    if (row >= N || col >= N)
    {
        return;
    }

    float inner_product = 0.0f;

    for (size_t inner_index = 0; inner_index < N; inner_index++)
    {
        inner_product += A[row * N + inner_index] * B[inner_index * N + col];
    }
    C[row * N + col] = inner_product;
}
