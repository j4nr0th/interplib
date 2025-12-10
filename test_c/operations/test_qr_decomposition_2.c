#include "../../src/operations/matrices.h"
#include "../common/common.h"

static void print_matrix(const matrix_t *m)
{
    for (unsigned i = 0; i < m->rows; ++i)
    {
        for (unsigned j = 0; j < m->cols; ++j)
            printf("%8.3g ", m->values[i * m->cols + j]);
        printf("\n");
    }
}

int main()
{
    // 3 x 1 matrix
    double matrix_a[] = {
        -0.41,
        0.031,
        0.138,
    };

    const matrix_t a = {.rows = 3, .cols = 1, .values = matrix_a};
    const matrix_t q = {.rows = 3, .cols = 3, .values = (double[3 * 3]){}};
    const matrix_t r = {.rows = 3, .cols = 1, .values = (double[3 * 1]){}};
    const matrix_t ra = {.rows = 3, .cols = 1, .values = (double[3 * 1]){}};

    printf("Initial matrix A:\n");
    print_matrix(&a);

    // Copy A to R
    for (unsigned i = 0; i < 3; ++i)
    {
        for (unsigned j = 0; j < 1; ++j)
            r.values[i * 1 + j] = a.values[i * 1 + j];
    }

    interp_result_t res = matrix_qr_decompose(&r, &q);
    printf("Matrix Q^T:\n");
    print_matrix(&q);
    printf("Matrix R:\n");
    print_matrix(&r);

    TEST_ASSERTION(res == INTERP_SUCCESS, "QR decomposition failed");

    // Multiplying A with Q should give R

    res = matrix_multiply(&q, &a, &ra);
    TEST_ASSERTION(res == INTERP_SUCCESS, "Multiplication failed");
    printf("Matrix QA:\n");
    print_matrix(&ra);

    for (unsigned i = 0; i < 3; ++i)
    {
        for (unsigned j = 0; j < 1; ++j)
        {
            const double va = r.values[i * 1 + j];
            const double vb = ra.values[i * 1 + j];
            TEST_NUMBERS_CLOSE(va, vb, 1e-12, 1e-10);
        }
    }

    // Transpose matrix Q back
    for (unsigned i = 0; i < 3; ++i)
    {
        for (unsigned j = 0; j < i; ++j)
        {
            const double tmp = q.values[i * 3 + j];
            q.values[i * 3 + j] = q.values[j * 3 + i];
            q.values[j * 3 + i] = tmp;
        }
    }

    printf("Matrix Q:\n");
    print_matrix(&q);

    res = matrix_multiply(&q, &r, &ra);
    TEST_ASSERTION(res == INTERP_SUCCESS, "Multiplication failed");

    printf("Re-computed QR:\n");
    print_matrix(&ra);

    // Compare
    for (unsigned i = 0; i < 3; ++i)
    {
        for (unsigned j = 0; j < 1; ++j)
        {
            const double va = a.values[i * 1 + j];
            const double vb = ra.values[i * 1 + j];
            TEST_NUMBERS_CLOSE(va, vb, 1e-12, 1e-10);
        }
    }

    return 0;
}
