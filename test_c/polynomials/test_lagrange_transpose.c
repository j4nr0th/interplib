//
// Created by jan on 2025-09-13.
//
#include "../../src/polynomials/lagrange.h"
#include "../common/common.h"
#include <math.h>

int main()
{
    test_prng_t rng;
    test_prng_seed(&rng, 834760834);

    enum
    {
        ROOT_COUNT = 100,
        TEST_NODES = 1000,
    };

    double roots[ROOT_COUNT];
    double nodes[TEST_NODES];

    for (int i = 0; i < ROOT_COUNT; ++i)
    {
        roots[i] = 2 * test_prng_next_double(&rng) - 1;
    }

    for (unsigned i = 0; i < TEST_NODES; ++i)
    {
        nodes[i] = 2 * test_prng_next_double(&rng) - 1;
    }

    double values_normal[TEST_NODES * ROOT_COUNT];
    double values_transposed[ROOT_COUNT * TEST_NODES];
    double work_buffer[ROOT_COUNT];

    lagrange_polynomial_values(TEST_NODES, nodes, ROOT_COUNT, roots, values_normal, work_buffer);
    lagrange_polynomial_values_transposed_2(TEST_NODES, nodes, ROOT_COUNT, roots, values_transposed);

    double error_square = 0.0;
    for (unsigned i = 0; i < TEST_NODES; ++i)
    {
        double inner_error_square = 0.0;
        for (unsigned j = 0; j < ROOT_COUNT; ++j)
        {
            const double error = values_normal[j + i * ROOT_COUNT] - values_transposed[i + j * TEST_NODES];
            const double avg = (values_normal[j + i * ROOT_COUNT] + values_transposed[i + j * TEST_NODES]) / 2.0;
            inner_error_square += (error * error) / (avg * avg);
            TEST_NUMBERS_CLOSE(values_normal[j + i * ROOT_COUNT], values_transposed[i + j * TEST_NODES], 1e-12, 1e-10);
        }
        error_square += inner_error_square;
    }

    printf("RMS error: %.5e\n", sqrt(error_square / (TEST_NODES * ROOT_COUNT)));

    return 0;
}
