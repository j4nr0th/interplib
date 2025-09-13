//
// Created by jan on 2025-09-10.
//

#ifndef INTERPLIB_INTEGRATION_RULE_OBJECT_H
#define INTERPLIB_INTEGRATION_RULE_OBJECT_H
#include "../integration/integration_rules.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD;
    const integration_rule_t *rule;
} integration_rule_object;

INTERPLIB_INTERNAL
extern PyType_Spec integration_rule_type_spec;

#endif // INTERPLIB_INTEGRATION_RULE_OBJECT_H
