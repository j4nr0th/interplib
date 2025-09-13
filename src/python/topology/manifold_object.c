//
// Created by jan on 18.1.2025.
//

#include "manifold_object.h"

PyDoc_STRVAR(manifold_type_docstr, "A manifold of a finite number of dimensions.");

static PyType_Slot manifold_type_slots[] = {
    {.slot = Py_tp_doc, .pfunc = (void *)manifold_type_docstr},
    {0},
};

INTERPLIB_INTERNAL
PyType_Spec manifold_type_spec = {
    .name = "interplib._interp.Manifold",
    .basicsize = sizeof(PyObject),
    .itemsize = 0,
    .flags = Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_DISALLOW_INSTANTIATION,
    .slots = manifold_type_slots,
};
