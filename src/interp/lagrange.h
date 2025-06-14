//
// Created by jan on 29.9.2024.

#ifndef LAGRANGE_H
#define LAGRANGE_H
#include "../common_defines.h"
#include "../error.h"

/**
 * @brief Compute common denominators of Lagrange polynomials.
 *
 * @param n Number of nodes.
 * @param nodes Array with nodes where the Lagrange polynomial is zero.
 * @param denominators Array which receives the denominators.
 */
INTERPLIB_INTERNAL
void lagrange_polynomial_denominators(unsigned n, const double INTERPLIB_ARRAY_ARG(nodes, restrict static n),
                                      double INTERPLIB_ARRAY_ARG(denominators, restrict n));
/**
 * @brief Compute values of Lagrange polynomial coefficients without dividing by the common denominator.
 *
 * @param n Number of nodes.
 * @param j Index of the Lagrange polynomial. At that node its value will be non-zero.
 * @param nodes Array with nodes where the Lagrange polynomial is zero.
 * @param coefficients Array which receives the coefficients. The term's index corresponds to it's power of x.
 */
INTERPLIB_INTERNAL
void lagrange_polynomial_coefficients(unsigned n, unsigned j,
                                      const double INTERPLIB_ARRAY_ARG(nodes, restrict static n),
                                      double INTERPLIB_ARRAY_ARG(coefficients, restrict n));

/**
 * @brief Compute values of Lagrange polynomials with given nodes at specified
 * locations. The interpolation can be computed for any function on the same
 * mesh by taking the inner product of the weight matrix with the function
 * values.
 *
 * @param n_in Number of points where polynomials should be evaluated.
 * @param pos Points where the Lagrange polynomials should be evaluated at.
 * @param n_nodes Number or roots of Lagrange polynomials, which is also the order of the polynomials.
 * @param x Roots of the lagrange polynomials.
 * @param weights Array which receives the values of Lagrange polynomials.
 * @param work Array used to store intermediate results.
 *
 */
INTERPLIB_INTERNAL
void lagrange_polynomial_values(unsigned n_in, const double INTERPLIB_ARRAY_ARG(pos, static n_in), unsigned n_nodes,
                                const double INTERPLIB_ARRAY_ARG(x, static n_nodes),
                                double INTERPLIB_ARRAY_ARG(weights, restrict n_nodes *n_in),
                                double INTERPLIB_ARRAY_ARG(work, restrict n_nodes));

/**
 * @brief Compute first derivative of Lagrange polynomials with given nodes at
 * specified locations. The interpolation can be computed for any function on
 * the same mesh by taking the inner product of the weight matrix with the
 * function values.
 *
 * @param n_in Number of points where polynomials should be evaluated.
 * @param pos Points where the Lagrange polynomials should be evaluated at.
 * @param n_nodes Number or roots of Lagrange polynomials, which is also the order of the polynomials.
 * @param x Roots of the lagrange polynomials.
 * @param weights Array which receives the values of Lagrange polynomials.
 * @param work1 Array used to store intermediate results.
 * @param work2 Array used to store intermediate results.
 */
INTERPLIB_INTERNAL
void lagrange_polynomial_first_derivative(unsigned n_in, const double INTERPLIB_ARRAY_ARG(pos, static n_in),
                                          unsigned n_nodes, const double INTERPLIB_ARRAY_ARG(x, static n_nodes),
                                          double INTERPLIB_ARRAY_ARG(weights, restrict n_nodes *n_in),
                                          double INTERPLIB_ARRAY_ARG(work1, restrict n_nodes),
                                          double INTERPLIB_ARRAY_ARG(work2, restrict n_nodes));

/**
 * @brief Compute second derivative of Lagrange polynomials with given nodes at
 * specified locations. The interpolation can be computed for any function on
 * the same mesh by taking the inner product of the weight matrix with the
 * function values.
 *
 * @param n_in Number of points where the interpolation will be needed.
 * @param pos Array of nodes where interpolation will be computed
 * @param n_nodes Number of nodes where the function is known.
 * @param x Array of x-values of nodes where the function is known which must
 * be monotonically increasing.
 * @param weights Array which receives the weights for the interpolation.
 * @param work1 Array used to store intermediate results.
 * @param work2 Array used to store intermediate results.
 *
 * @return `INTERP_SUCCESS` on success, `INTERP_ERROR_NOT_INCREASING` if `x[i +
 * 1] > x[i]` does not hold for all `i`.
 */
INTERPLIB_INTERNAL
interp_error_t lagrange_polynomial_second_derivative(unsigned n_in, const double INTERPLIB_ARRAY_ARG(pos, static n_in),
                                                     unsigned n_nodes,
                                                     const double INTERPLIB_ARRAY_ARG(x, static n_nodes),
                                                     double INTERPLIB_ARRAY_ARG(weights, restrict n_nodes *n_in),
                                                     double INTERPLIB_ARRAY_ARG(work1, restrict n_nodes),
                                                     double INTERPLIB_ARRAY_ARG(work2, restrict n_nodes));

#endif // LAGRANGE_H
