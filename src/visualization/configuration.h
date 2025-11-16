//
// Created by jan on 2025-09-14.
//

#ifndef INTERPLIB_CONFIGURATION_H
#define INTERPLIB_CONFIGURATION_H

#include "../basis/de_rham_sequence.h"
#include "color.h"
#include "common.h"
#include "space.h"

#include "../topology/topology.h"

typedef struct
{
    unsigned initial_window_width;
    unsigned initial_window_height;
    int fixed_size;
    int start_fullscreen;
    const char *window_title;
    vec4b_t background_color;
} window_configuration_t;

typedef struct
{
    // How to display Geometry
    space_t display_space;
    const colormap_t *data_colormap;
    value_mapping_t value_mapping;

    // What to display
    // topology and element information is included in the forms here already
    const differential_form_t *form;      // data to display
    const differential_form_t *positions; // array of forms that specify positions
} scene_configuration_t;

#endif // INTERPLIB_CONFIGURATION_H
