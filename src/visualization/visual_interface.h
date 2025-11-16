//
// Created by jan on 2025-09-14.
//

#ifndef INTERPLIB_VISUAL_INTERFACE_H
#define INTERPLIB_VISUAL_INTERFACE_H
#include "../basis/de_rham_sequence.h"
#include "../topology/topology.h"
#include "common.h"

typedef struct viz_window_t viz_window_t;

typedef struct visual_interface_t visual_interface_t;
typedef viz_result_t (*visualization_create_function)(visual_interface_t *interface, void *in_data,
                                                      viz_window_t **out_window);
typedef viz_result_t (*visualization_await_function)(visual_interface_t *interface, viz_window_t *window);
typedef viz_result_t (*visualization_destroy_function)(visual_interface_t *interface, viz_window_t *window);

struct visual_interface_t
{
    visualization_create_function create_function;
    visualization_await_function await_function;
    visualization_destroy_function destroy_function;
};

#endif // INTERPLIB_VISUAL_INTERFACE_H
