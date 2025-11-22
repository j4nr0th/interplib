cmake_minimum_required(VERSION 3.29)

if (NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME "InterpLib")
endif ()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C)
find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)

find_package(OpenMP)

execute_process(
        COMMAND "${Python_EXECUTABLE}"
        -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NumPy_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(BASIS_SOURCE_FILES
        src/basis/basis_set.c
        src/basis/basis_bernstein.c
        src/basis/basis_lagrange.c
        src/basis/basis_legendre.c
        src/basis/de_rham_sequence.c
)
set(BASIS_HEADER_FILES
        src/basis/basis_set.h
        src/basis/basis_bernstein.h
        src/basis/basis_lagrange.h
        src/basis/basis_legendre.h
        src/basis/de_rham_sequence.h
)
set(COMMON_SOURCE_FILES
        src/common/allocator.c
        src/common/error.c
        src/common/rw_lock.c
)
set(COMMON_HEADER_FILES
        src/common/allocator.h
        src/common/common_defines.h
        src/common/error.h
        src/common/rw_lock.h
)

set(INTEGRATION_SOURCE_FILES
        src/integration/integration_rules.c
        src/integration/gauss_legendre.c
        src/integration/gauss_lobatto.c
)

set(INTEGRATION_HEADER_FILES
        src/integration/integration_rules.h
        src/integration/gauss_legendre.h
        src/integration/gauss_lobatto.h
)

set(OPERATIONS_SOURCE_FILES
        src/operations/nd_iteration.c
)

set(OPERATIONS_HEADER_FILES
        src/operations/nd_iteration.h
)

set(POLYNOMIALS_SOURCE_FILES
        src/polynomials/bernstein.c
        src/polynomials/lagrange.c
        src/polynomials/legendre.c
)

set(POLYNOMIALS_HEADER_FILES
        src/polynomials/bernstein.h
        src/polynomials/lagrange.h
        src/polynomials/legendre.h
)

set(TOPOLOGY_SOURCE_FILES
        src/topology/topology.c
)

set(TOPOLOGY_HEADER_FILES
        src/topology/topology.h
)

add_library(interp STATIC
        ${BASIS_SOURCE_FILES}
        ${COMMON_SOURCE_FILES}
        ${INTEGRATION_SOURCE_FILES}
        ${OPERATIONS_SOURCE_FILES}
        ${TOPOLOGY_SOURCE_FILES}
        ${POLYNOMIALS_SOURCE_FILES}
)

set_property(TARGET interp PROPERTY C_STANDARD 17)
target_compile_definitions(interp PRIVATE INTERPLIB_ASSERTS)
target_link_libraries(interp PRIVATE m)
if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(interp PRIVATE -Wall -Wextra -Werror -fanalyzer -Werror)
endif ()

set(PYTHON_TOPOLOGY_SOURCE_FILES
        src/python/topology/geoid_object.c
        src/python/topology/line_object.c
        src/python/topology/manifold1d_object.c
        src/python/topology/manifold2d_object.c
        src/python/topology/manifold_object.c
        src/python/topology/surface_object.c
)

set(PYTHON_NUMERICS_SOURCE_FILES
        src/python/integration_objects.c
        src/python/basis_objects.c
)

Python_add_library(_interp MODULE
        src/python/module.c
        ${PYTHON_TOPOLOGY_SOURCE_FILES}
        ${PYTHON_NUMERICS_SOURCE_FILES}
        WITH_SOABI)

add_subdirectory(cpyutl)

target_link_libraries(_interp PRIVATE interp cpyutl)
set_property(TARGET _interp PROPERTY C_STANDARD 17)
install(TARGETS _interp DESTINATION ${SKBUILD_PROJECT_NAME})
target_compile_definitions(_interp PRIVATE INTERPLIB_ASSERTS)

if ("${NumPy_INCLUDE_DIRS}" STREQUAL "")
    message(FATAL_ERROR "NumPy_INCLUDE_DIRS was empty.")
endif ()
target_include_directories(_interp PRIVATE ${NumPy_INCLUDE_DIRS} ${Python_INCLUDE_DIRS} cpyutl/src)

if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(_interp PRIVATE -Wall -Wextra -Werror -fwhole-program -flto=auto)
endif ()


enable_testing()
add_subdirectory(test_c)

# add_subdirectory(src/visualization)
