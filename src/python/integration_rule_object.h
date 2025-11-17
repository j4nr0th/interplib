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
extern PyType_Spec integration_rule_registry_type_spec;

INTERPLIB_INTERNAL
extern PyType_Spec integration_specs_type_spec;

INTERPLIB_INTERNAL
integration_registry_object *integration_registry_object_create(PyTypeObject *type);

#endif // INTERPLIB_INTEGRATION_RULE_OBJECT_H
