//
// Created by jan on 2025-09-08.
//

#ifndef INTERPLIB_DE_RHAM_SEQUENCE_H
#define INTERPLIB_DE_RHAM_SEQUENCE_H

#include "basis_set.h"

/* Type that describes the polynomial space used.

Based on the number of dimensions, the polynomial spaces are connected
by different operations. For `n` dimensional space they start at 0-forms
and end at `n`-forms.

For 1-dimensional case:

     strong/weak: grad/div
 H¹ ----------------------> L²

For 2-dimensional case:

        strong/weak: curl/curl         strong/weak: div/-grad
 H¹xH¹ -----------------------> H¹xL² -----------------------> L²xL²
                                H¹xL²

For 3-dimensional case:

           strong/weak: grad/div            strong/weak: curl/curl            strong/weak: div/grad
                                  L²xH¹xH¹                          H¹xL²xL²
 H¹xH¹xH¹ ----------------------> H¹xL²xH¹ -----------------------> L²xH¹xL² ----------------------> L²xL²xL²
                                  H¹xH¹xL²                          L²xL²xH¹
 */

#include "../topology/topology.h"

typedef struct
{
    unsigned order_integration;
    unsigned order_basis;
} element_order_t;

typedef struct element_collection_t element_collection_t;

INTERPLIB_INTERNAL
interp_result_t element_collection_init(element_collection_t *this, unsigned element_count, const manifold_t *manifold,
                                        const cutl_allocator_t *allocator);

typedef enum
{
    FORM_INVALID = 0,
    FORM_ORDER_0,
    FORM_ORDER_1,
    FORM_ORDER_2,
    // FORM_ORDER_3,
    FORM_ORDER_COUNT,
} form_order_t;

typedef struct
{
    const element_collection_t *collection;
    form_order_t form_order;
    double values[];
} differential_form_t;

static inline unsigned differential_form_degrees_of_freedom_count(const form_order_t form_order,
                                                                  const manifold_dimension_t dimension,
                                                                  const element_order_t orders[])
{
    switch (dimension)
    {
    case MANIFOLD_1D:
        switch (form_order)
        {
        case FORM_ORDER_0:
            return orders[0].order_basis + 1;

        case FORM_ORDER_1:
            return orders[0].order_basis;

        default:
            ASSERT(0, "Invalid form order.");
            return 0;
        }
        break;

    case MANIFOLD_2D:
        switch (form_order)
        {
        case FORM_ORDER_0:
            return (orders[0].order_basis + 1) * (orders[1].order_basis + 1);

        case FORM_ORDER_1:
            return (orders[0].order_basis + 1) * orders[1].order_basis +
                   orders[0].order_basis * (orders[1].order_basis + 1);

        case FORM_ORDER_2:
            return orders[0].order_basis * orders[1].order_basis;

        default:
            ASSERT(0, "Invalid form order.");
            return 0;
        }
        break;

    default:
        ASSERT(0, "Invalid manifold dimension.");
        return 0;
    }
}

INTERPLIB_INTERNAL
interp_result_t differential_form_new(differential_form_t **out, form_order_t form_order,
                                      element_collection_t *collection, const double *set_value);

INTERPLIB_INTERNAL
void differential_form_destroy(differential_form_t *this);

INTERPLIB_INTERNAL
double *differential_form_element_degrees_of_freedom(differential_form_t *this, unsigned index);

INTERPLIB_INTERNAL
const double *differential_form_element_degrees_of_freedom_const(const differential_form_t *this, unsigned index);

INTERPLIB_INTERNAL
unsigned differential_form_element_degrees_of_freedom_count(const differential_form_t *this, unsigned index);

INTERPLIB_INTERNAL
const manifold_t *differential_form_manifold(const differential_form_t *this);

#endif // INTERPLIB_DE_RHAM_SEQUENCE_H
