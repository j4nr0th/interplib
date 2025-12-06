#ifndef INTERPLIB_RECONSTRUCTION_H
#define INTERPLIB_RECONSTRUCTION_H

#include "../common/allocator.h"
#include "../common/error.h"

#include "../basis/basis_set.h"

#include "../integration/integration_rules.h"

typedef struct reconstruction_state_t reconstruction_state_t;

interp_result_t reconstruction_state_create(unsigned ndims, const basis_set_t *basis_sets[const static ndims],
                                            const allocator_callbacks *allocator, reconstruction_state_t **out);

void reconstruction_state_release(reconstruction_state_t *recon_state);

void reconstruction_state_reset(reconstruction_state_t *recon_state);

size_t reconstruction_state_integration_points_count(const reconstruction_state_t *state);

size_t reconstruction_state_basis_count(const reconstruction_state_t *state);

void reconstruction_state_basis_values_current(const reconstruction_state_t *state, double *values);

#endif // INTERPLIB_RECONSTRUCTION_H
