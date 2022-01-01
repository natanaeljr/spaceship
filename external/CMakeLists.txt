cmake_minimum_required(VERSION 3.5)
include(ExternalProject)

#########################################################################################
# SPDLOG
#########################################################################################
# SPDLog is very fast, header-only or compiled, C++ logging library.
# SPDLog is licensed under the MIT license.
message(STATUS "Adding external library: SPDLOG")
ExternalProject_Add(spdlog_external
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/spdlog
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/spdlog
    INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DCMAKE_GENERATOR=${CMAKE_GENERATOR}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DSPDLOG_INSTALL=ON
        -DSPDLOG_BUILD_SHARED=OFF
        -DSPDLOG_BUILD_TESTS=OFF
        -DSPDLOG_BUILD_EXAMPLE=OFF
        -DSPDLOG_FMT_EXTERNAL=OFF
        -DSPDLOG_NO_EXCEPTIONS=ON)

#########################################################################################
# GLFW
#########################################################################################
# GLFW is multi-platform library for OpenGL, OpenGL ES and Vulkan development on desktop.
# Provides an API for creating windows, contexts and surfaces, receiving input and events.
# GLFW is licensed under the zlib/libpng license.
message(STATUS "Adding external library: GLFW")
ExternalProject_Add(glfw_external
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/glfw
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/glfw
    INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DCMAKE_GENERATOR=${CMAKE_GENERATOR}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS=OFF
        -DGLFW_BUILD_DOCS=OFF
        -DGLFW_BUILD_TESTS=OFF
        -DGLFW_BUILD_EXAMPLES=OFF)
