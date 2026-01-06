//
// Created by jan on 2025-09-08.
//

#include "de_rham_sequence.h"

#include <string.h>

typedef struct
{
    form_order_t order;
    unsigned offsets[];
} form_offsets_t;

typedef struct element_collection_t
{
    cutl_allocator_t allocator;
    const manifold_t *manifold;
    unsigned element_count;
    element_order_t *element_orders;
    form_offsets_t *form_offsets[FORM_ORDER_COUNT - 1];
} element_collection_t;

interp_result_t element_collection_init(element_collection_t *this, const unsigned element_count,
                                        const manifold_t *manifold, const cutl_allocator_t *allocator)
{
    this->manifold = manifold;
    this->allocator = *allocator;
    this->element_count = element_count;

    this->element_orders =
        cutl_alloc(allocator, element_count * sizeof *this->element_orders * manifold_dimension_count(this->manifold));
    if (!this->element_orders)
        return INTERP_ERROR_FAILED_ALLOCATION;

    for (unsigned i = 0; i < FORM_ORDER_COUNT - 1; ++i)
    {
        // this->form_offsets[i] = allocate(allocator, element_count * sizeof *this->form_offsets);
        // if (!this->form_offsets[i])
        // {
        //     for (unsigned j = 0; j < i; ++j)
        //         deallocate(allocator, this->form_offsets[j]);
        //     deallocate(allocator, this->element_orders);
        //     return INTERP_ERROR_FAILED_ALLOCATION;
        // }
        this->form_offsets[i] = NULL;
    }

    return INTERP_SUCCESS;
}

void element_collection_destroy(element_collection_t *this)
{
    cutl_dealloc(&this->allocator, this->element_orders);
    for (unsigned i = 0; i < FORM_ORDER_COUNT - 1; ++i)
    {
        cutl_dealloc(&this->allocator, this->form_offsets[i]);
    }
    *this = (element_collection_t){};
}

interp_result_t element_collection_form_offsets(element_collection_t *this, const form_order_t form_order,
                                                const form_offsets_t **out)
{
    if (this->form_offsets[form_order - 1] != NULL)
    {
        *out = this->form_offsets[form_order - 1];
        return INTERP_SUCCESS;
    }

    form_offsets_t *const offsets =
        cutl_alloc(&this->allocator, sizeof *offsets + sizeof *offsets->offsets * (this->element_count + 1));
    if (!offsets)
        return INTERP_ERROR_FAILED_ALLOCATION;
    offsets->order = form_order;

    offsets->offsets[0] = 0;
    for (unsigned i = 0; i < this->element_count; ++i)
    {
        offsets->offsets[i + 1] =
            offsets->offsets[i] + differential_form_degrees_of_freedom_count(
                                      form_order, this->manifold->manifold_type,
                                      this->element_orders + i * manifold_dimension_count(this->manifold));
    }
    this->form_offsets[form_order - 1] = offsets;
    *out = offsets;
    return INTERP_SUCCESS;
}

INTERPLIB_INTERNAL
interp_result_t differential_form_new(differential_form_t **out, const form_order_t form_order,
                                      element_collection_t *collection, const double *set_value)
{
    const unsigned element_count = collection->element_count;
    const form_offsets_t *offsets;
    const interp_result_t res = element_collection_form_offsets(collection, form_order, &offsets);
    if (res != INTERP_SUCCESS)
        return res;

    differential_form_t *const this =
        cutl_alloc(&collection->allocator, sizeof *this + sizeof *this->values * offsets->offsets[element_count]);
    if (!this)
        return INTERP_ERROR_FAILED_ALLOCATION;

    this->collection = collection;
    this->form_order = form_order;

    if (set_value != NULL)
    {
        if (*set_value == 0.0)
        {
            memset(this->values, 0, sizeof *this->values * offsets->offsets[element_count]);
        }
        else
        {
            for (unsigned i = 0; i < element_count; ++i)
            {
                this->values[offsets->offsets[i]] = *set_value;
            }
        }
    }

    *out = this;
    return INTERP_SUCCESS;
}

INTERPLIB_INTERNAL
void differential_form_destroy(differential_form_t *this)
{
    const cutl_allocator_t *allocator = &this->collection->allocator;
    this->collection = NULL;
    this->form_order = 0;
    cutl_dealloc(allocator, this);
}

INTERPLIB_INTERNAL
double *differential_form_element_degrees_of_freedom(differential_form_t *this, const unsigned index)
{
    const form_offsets_t *offsets = this->collection->form_offsets[this->form_order - 1];
    ASSERT(offsets != NULL, "Offsets for the form were not initialized.");
    ASSERT(index < this->collection->element_count, "Index of the element was out of bounds.");
    return this->values + offsets->offsets[index];
}

INTERPLIB_INTERNAL
const double *differential_form_element_degrees_of_freedom_const(const differential_form_t *this, const unsigned index)
{
    const form_offsets_t *offsets = this->collection->form_offsets[this->form_order - 1];
    ASSERT(offsets != NULL, "Offsets for the form were not initialized.");
    ASSERT(index < this->collection->element_count, "Index of the element was out of bounds.");
    return this->values + offsets->offsets[index];
}

INTERPLIB_INTERNAL
unsigned differential_form_element_degrees_of_freedom_count(const differential_form_t *this, const unsigned index)
{
    const form_offsets_t *offsets = this->collection->form_offsets[this->form_order - 1];
    ASSERT(offsets != NULL, "Offsets for the form were not initialized.");
    ASSERT(index < this->collection->element_count, "Index of the element was out of bounds.");
    return offsets->offsets[index + 1] - offsets->offsets[index];
}
const manifold_t *differential_form_manifold(const differential_form_t *this)
{
    return this->collection->manifold;
}
