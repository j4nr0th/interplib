//
// Created by jan on 18.1.2025.
//

#ifndef MIMETIC_MODULE_H
#define MIMETIC_MODULE_H

typedef unsigned index_t;
_Static_assert(sizeof(index_t) == 4, "This must be for the geo_id_t to make sense");
typedef struct
{
    index_t index : 31;
    index_t reverse : 1;
} geo_id_t;

enum
{
    GEO_ID_INVALID = ~(1 << 31)
};

typedef union {
    struct
    {
        geo_id_t begin;
        geo_id_t end;
    };
    geo_id_t values[2];
} line_t;

typedef struct
{
    unsigned n_lines;
    geo_id_t *values;
} surface_t;

// typedef union
// {
//     struct { double x, y, z; };
//     struct { double v0, v1, v2; };
//     double values[3];
// } double3;

#endif // MIMETIC_MODULE_H
