find_package(OpenGL REQUIRED)

set(GLAD_SOURCE_FILES
        glad/src/glad.c
)

set(RENDERER_SOURCE_FILES
        renderer_1d.c
        renderer_1d.h

)

add_library(interprender STATIC ${GLAD_SOURCE_FILES} ${RENDERER_SOURCE_FILES})
target_link_libraries(interprender PRIVATE OpenGL)
target_include_directories(interprender PRIVATE glad/include)
