cmake_minimum_required(VERSION 3.16)

project(OpenFHE-Numpy)

set(OPENFHE_NUMPY_VERSION_MAJOR 0)
set(OPENFHE_NUMPY_VERSION_MINOR 0)
set(OPENFHE_NUMPY_VERSION_PATCH 1)
set(OPENFHE_NUMPY_VERSION ${OPENFHE_NUMPY_VERSION_MAJOR}.${OPENFHE_NUMPY_VERSION_MINOR}.${OPENFHE_NUMPY_VERSION_PATCH})

# OpenFHE version can be specified externally (-DOPENFHE_REQUIRED_VERSION=1.3.0)
if(NOT DEFINED OPENFHE_REQUIRED_VERSION)
    set(OPENFHE_REQUIRED_VERSION "1.3.0" CACHE STRING "Required OpenFHE version")
else()
    # User provided OPENFHE_REQUIRED_VERSION via -D
    message(STATUS "Using user-specified OpenFHE version: ${OPENFHE_REQUIRED_VERSION}")
endif()

# Build Options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

if(APPLE)
    set(CMAKE_CXX_VISIBILITY_PRESET default)
endif()

option(BUILD_EXAMPLES "Set to ON to build C++ examples for the library" ON)
# option(BUILD_TESTING "Enable building of tests" OFF)
message(STATUS "BUILD_EXAMPLES:     ${BUILD_EXAMPLES}")

find_package(OpenFHE ${OPENFHE_REQUIRED_VERSION} REQUIRED)
message(STATUS "Building with OpenFHE version: ${OPENFHE_REQUIRED_VERSION}")

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)


# [tango]
# Must be used for easy structural changes!!!
# Renamed the source directory from 'openfhe_numpy/openfhe_numpy/' to 'openfhe_numpy/src/' so that
# 'openfhe_numpy/examples/' and 'openfhe_numpy/tests/' import the installed 'openfhe_numpy' package
# (with C++ bindings) rather than the local source folder. The local copy
# lacks the built C++ extensions and would cause import errors.

# Define OPENFHE_NUMPY_INSTALL_DIR for the main source directory.
set(OpenFHE_Numpy_MAIN_SOURCE "${CMAKE_SOURCE_DIR}/src")
set(OpenFHE_Numpy_INCLUDES "${OpenFHE_Numpy_MAIN_SOURCE}/cpp/include")
set(OpenFHE_Numpy_SOURCE "${OpenFHE_Numpy_MAIN_SOURCE}/cpp/lib")

include_directories( ${OpenFHE_Numpy_INCLUDES} )
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)

set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
link_libraries(${OpenFHE_SHARED_LIBRARIES})

add_library(openfhe_numpy_cpp STATIC
    ${OpenFHE_Numpy_INCLUDES}/numpy_constants.h
    ${OpenFHE_Numpy_INCLUDES}/numpy_array_metadata.h
    ${OpenFHE_Numpy_INCLUDES}/numpy_enc_matrix.h
    ${OpenFHE_Numpy_INCLUDES}/numpy_helper_functions.h
    ${OpenFHE_Numpy_INCLUDES}/numpy_utils.h
    ${OpenFHE_Numpy_SOURCE}/numpy_array_metadata.cpp
    ${OpenFHE_Numpy_SOURCE}/numpy_enc_matrix.cpp
    ${OpenFHE_Numpy_SOURCE}/numpy_helper_functions.cpp
    ${OpenFHE_Numpy_SOURCE}/numpy_utils.cpp
)

# Python extension module via pybind11
pybind11_add_module(openfhe_numpy ${OpenFHE_Numpy_SOURCE}/numpy_bindings.cpp)

target_link_libraries(openfhe_numpy PRIVATE openfhe_numpy_cpp)

# Allow the user to specify the path to Python executable (if not provided, find it)
option(PYTHON_EXECUTABLE_PATH "Path to Python executable" "")
if(PYTHON_EXECUTABLE_PATH)
    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE_PATH}")
endif()
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Check Python version
if(Python3_VERSION_MAJOR EQUAL 3 AND Python3_VERSION_MINOR GREATER_EQUAL 10)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "from sys import exec_prefix; print(exec_prefix)"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else()
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()


# Determine to use Python site-packages directory or user defined directory
# If user didn't override CMAKE_INSTALL_PREFIX install directly into site-packages
# Bby default, Unix-like system, CMAKE_INSTALL_PREFIX = /usr/local.

message(STATUS "Python site packages directory: ${PYTHON_SITE_PACKAGES}")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
    set(Python_Install_Location "${PYTHON_SITE_PACKAGES}")
else()
    set(Python_Install_Location "${CMAKE_INSTALL_PREFIX}")
endif()

message("***** INSTALL IS AT ${Python_Install_Location}; to change, run cmake with -DCMAKE_INSTALL_PREFIX=/your/path")


# [tango] place all files inside the openfhe_numpy folder. Flattening this structure may shadow modules from other packages.
# Define OPENFHE_NUMPY_INSTALL_DIR for easy install-path overrides.
set(OpenFHE_Numpy_INSTALL_PATH "${Python_Install_Location}/openfhe_numpy")

# install the .so with the python bindings
install(TARGETS openfhe_numpy LIBRARY DESTINATION ${OpenFHE_Numpy_INSTALL_PATH})
install(FILES ${CMAKE_SOURCE_DIR}/src/__init__.py DESTINATION ${OpenFHE_Numpy_INSTALL_PATH})

# install the required directories: operations, tensor and utils (with *.py files only)
file(GLOB OPERATIONS_DIR_FILES "${OpenFHE_Numpy_MAIN_SOURCE}/operations/*.py")
install(FILES ${OPERATIONS_DIR_FILES} DESTINATION ${OpenFHE_Numpy_INSTALL_PATH}/operations/)
file(GLOB TENSOR_DIR_FILES "${OpenFHE_Numpy_MAIN_SOURCE}/tensor/*.py")
install(FILES ${TENSOR_DIR_FILES} DESTINATION ${OpenFHE_Numpy_INSTALL_PATH}/tensor/)
file(GLOB UTILS_DIR_FILES "${OpenFHE_Numpy_MAIN_SOURCE}/utils/*.py")
install(FILES ${UTILS_DIR_FILES} DESTINATION ${OpenFHE_Numpy_INSTALL_PATH}/utils/)

if(BUILD_EXAMPLES)
    file(GLOB EXAMPLE_SOURCES "${OpenFHE_Numpy_MAIN_SOURCE}/cpp/examples/*.cpp")

    foreach(example_file ${EXAMPLE_SOURCES})
        get_filename_component(example_name ${example_file} NAME_WE)
        add_executable(${example_name} ${example_file})
        target_include_directories(${example_name} PRIVATE ${OpenFHE_Numpy_INCLUDES})
        if(TARGET OpenFHE::pke)
            target_link_libraries(${example_name} PRIVATE openfhe_numpy_cpp OpenFHE::pke OpenFHE::core OpenFHE::binfhe)
        else()
            target_link_libraries(${example_name} PRIVATE openfhe_numpy_cpp)
        endif()
    endforeach()
endif()
