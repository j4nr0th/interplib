find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK4 REQUIRED IMPORTED_TARGET gtk4)

set(VISUALIZATION_COMMON_SOURCE_FILES
        visual_interface.c
        common.c
        configuration.c
        configuration.h
        space.c
        space.h
        color.c
        color.h
)

add_library(interpviz STATIC
        ${VISUALIZATION_COMMON_SOURCE_FILES}
)
add_subdirectory(renderer)
target_link_libraries(interpviz PRIVATE PkgConfig::GTK4 interp interprender)
