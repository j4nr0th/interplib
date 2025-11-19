#ifndef INTERPLIB_INTEGRATION_RULE_OBJECT_H
#define INTERPLIB_INTEGRATION_RULE_OBJECT_H
#include "../integration/integration_rules.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD;
    integration_rule_registry_t *registry;
} integration_registry_object;

typedef struct
{
    PyObject_HEAD;
    integration_rule_spec_t spec;
} integration_specs_object;

INTERPLIB_INTERNAL
extern PyType_Spec integration_registry_type_spec;

INTERPLIB_INTERNAL
extern PyType_Spec integration_specs_type_spec;

INTERPLIB_INTERNAL
integration_registry_object *integration_registry_object_create(PyTypeObject *type);

INTERPLIB_INTERNAL
integration_specs_object *integration_specs_object_create(PyTypeObject *type, integration_rule_spec_t spec);

#endif // INTERPLIB_INTEGRATION_RULE_OBJECT_H
