//
// Created by jan on 2025-09-09.
//

#ifndef INTERPLIB_BASIS_LEGENDRE_H
#define INTERPLIB_BASIS_LEGENDRE_H
#include "basis_set.h"

INTERPLIB_INTERNAL
interp_result_t legendre_basis_create(basis_set_t **out, basis_spec_t spec, const integration_rule_t *rule,
                                      const cutl_allocator_t *allocator);

#endif // INTERPLIB_BASIS_LEGENDRE_H
