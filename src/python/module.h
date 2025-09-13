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
    integration_rule_registry_t *integration_rule_registry;
    PyTypeObject *integration_rule_type;

    // Basis
    basis_set_registry_t *basis_registry;
    PyTypeObject *basis_set_type;

    // Topology
    PyTypeObject *geoid_type;
    PyTypeObject *line_type;
    PyTypeObject *surf_type;
    PyTypeObject *man_type;
    PyTypeObject *man1d_type;
    PyTypeObject *man2d_type;
} interplib_module_state_t;

INTERPLIB_INTERNAL
extern PyModuleDef interplib_module;

static inline integration_rule_registry_t *interplib_get_integration_registry(PyTypeObject *type)
{
    PyObject *const mod = PyType_GetModuleByDef(type, &interplib_module);
    if (!mod)
    {
        return NULL;
    }
    const interplib_module_state_t *const state = PyModule_GetState(mod);
    if (!state)
    {
        return NULL;
    }
    return state->integration_rule_registry;
}

static inline basis_set_registry_t *interplib_get_basis_registry(PyTypeObject *type)
{
    PyObject *const mod = PyType_GetModuleByDef(type, &interplib_module);
    if (!mod)
    {
        return NULL;
    }
    const interplib_module_state_t *const state = PyModule_GetState(mod);
    if (!state)
    {
        return NULL;
    }
    return state->basis_registry;
}

#endif // INTERPLIB_MODULE_H
