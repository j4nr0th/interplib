#ifndef INTERPLIB_DEGREES_OF_FREEDOM_H
#define INTERPLIB_DEGREES_OF_FREEDOM_H

#include "module.h"

typedef struct
{
    PyObject_VAR_HEAD;
    unsigned n_dims;
    basis_spec_t *basis_specs;
    double values[];
} dof_object;

INTERPLIB_INTERNAL
extern PyType_Spec degrees_of_freedom_type_spec;

INTERPLIB_INTERNAL
PyObject *dof_reconstruct_at_integration_points(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                Py_ssize_t nargs, const PyObject *kwnames);

INTERPLIB_INTERNAL
PyObject *dof_reconstruct_derivative_at_integration_points(PyObject *self, PyTypeObject *defining_class,
                                                           PyObject *const *args, Py_ssize_t nargs,
                                                           const PyObject *kwnames);

INTERPLIB_INTERNAL
int *reconstruction_derivative_indices(unsigned ndim, PyObject *py_indices);

#endif // INTERPLIB_DEGREES_OF_FREEDOM_H
