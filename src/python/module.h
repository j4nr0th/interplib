//
// Created by jan on 2025-09-07.
//

#ifndef INTERPLIB_MODULE_H
#define INTERPLIB_MODULE_H

#include "../common/allocator.h"

//  Python ssize define
#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

//  Prevent numpy from being re-imported
#ifndef PY_LIMITED_API
#define PY_LIMITED_API 0x030A0000
#endif

#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL _interp
#endif

#include <Python.h>
#include <cpyutl.h>
#include <numpy/ndarrayobject.h>

INTERPLIB_INTERNAL
extern allocator_callbacks SYSTEM_ALLOCATOR;

INTERPLIB_INTERNAL
extern allocator_callbacks PYTHON_ALLOCATOR;

INTERPLIB_INTERNAL
extern allocator_callbacks OBJECT_ALLOCATOR;

#include "../basis/basis_set.h"
#include "../integration/integration_rules.h"

typedef struct
{
    // Integration
    PyTypeObject *integration_spec_type;
    PyTypeObject *integration_registry_type;

    // Basis
    PyTypeObject *basis_registry_type;
    PyTypeObject *basis_spec_type;

    // Function Spaces
    PyTypeObject *function_space_type;

    // Topology
    PyTypeObject *geoid_type;
    PyTypeObject *line_type;
    PyTypeObject *surf_type;
    PyTypeObject *man_type;
    PyTypeObject *man1d_type;
    PyTypeObject *man2d_type;

    // Default Registries
    PyObject *registry_integration;
    PyObject *registry_basis;
} interplib_module_state_t;

INTERPLIB_INTERNAL
extern PyModuleDef interplib_module;

static inline const interplib_module_state_t *interplib_get_module_state(PyTypeObject *type)
{
    PyObject *const mod = PyType_GetModuleByDef(type, &interplib_module);
    if (!mod)
    {
        return NULL;
    }
    return PyModule_GetState(mod);
}

INTERPLIB_INTERNAL
int heap_type_traverse_type(PyObject *self, visitproc visit, void *arg);

#endif // INTERPLIB_MODULE_H
