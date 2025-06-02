# Minimum required CMake version - needs at least 3.16 for modern features
cmake_minimum_required(VERSION 3.16)

# Project declaration with name, version, and languages
project(OpenFHE-Numpy)

# Version variables
set(OPENFHE_NUMPY_VERSION_MAJOR 0)
set(OPENFHE_NUMPY_VERSION_MINOR 0)
set(OPENFHE_NUMPY_VERSION_PATCH 1)
set(OPENFHE_NUMPY_VERSION ${OPENFHE_NUMPY_VERSION_MAJOR}.${OPENFHE_NUMPY_VERSION_MINOR}.${OPENFHE_NUMPY_VERSION_PATCH})

set(OPENFHE_PYTHON_VERSION_MAJOR 1)
set(OPENFHE_PYTHON_VERSION_MINOR 3)
set(OPENFHE_PYTHON_VERSION_PATCH 0)
set(OPENFHE_PYTHON_VERSION_TWEAK 0)
set(OPENFHE_PYTHON_VERSION ${OPENFHE_PYTHON_VERSION_MAJOR}.${OPENFHE_PYTHON_VERSION_MINOR}.${OPENFHE_PYTHON_VERSION_PATCH}.${OPENFHE_PYTHON_VERSION_TWEAK})

# Build Options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

option(BUILD_EXAMPLES "Set to ON to build C++ examples for the library" ON)
# option(BUILD_TESTING "Enable building of tests" OFF)
# message(STATUS "BUILD_EXAMPLES:     ${BUILD_EXAMPLES}")

if(APPLE)
    set(CMAKE_CXX_VISIBILITY_PRESET default)
endif()

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

# Dependencies
find_package(OpenFHE 1.3.0 REQUIRED)
find_package(pybind11 REQUIRED)
# find_package(OpenMP QUIET)

# include_directories(${OPENMP_INCLUDES})
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)

set(OpenFHE_Numpy_INCLUDES src/cpp/include)
include_directories( ${OpenFHE_Numpy_INCLUDES} )

# always link with shared libs
set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
link_libraries(${OpenFHE_SHARED_LIBRARIES})

set(OpenFHE_Numpy_SOURCE src/cpp/lib)
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

### Pybind Modules
# pybind11_add_module(openfhe_numpy 
#                     ${OpenFHE_Numpy_SOURCE}/numpy_array_metadata.cpp
#                     ${OpenFHE_Numpy_SOURCE}/numpy_enc_matrix.cpp
#                     ${OpenFHE_Numpy_SOURCE}/numpy_helper_functions.cpp
#                     ${OpenFHE_Numpy_SOURCE}/numpy_utils.cpp
#                     ${OpenFHE_Numpy_SOURCE}/numpy_bindings.cpp
#                     )
pybind11_add_module(openfhe_numpy ${OpenFHE_Numpy_SOURCE}/numpy_bindings.cpp)

target_link_libraries(openfhe_numpy PRIVATE openfhe_numpy_cpp)

# Find Python and its development components
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Check Python version
if(${Python3_VERSION_MAJOR} EQUAL 3 AND ${Python3_VERSION_MINOR} GREATER_EQUAL 10)
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

message(STATUS "Python site packages directory: ${PYTHON_SITE_PACKAGES}")
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(Python_Install_Location "${PYTHON_SITE_PACKAGES}")
else()
    set(Python_Install_Location "${CMAKE_INSTALL_PREFIX}")
endif()
message("***** INSTALL IS AT ${Python_Install_Location}; to change, run cmake with -DCMAKE_INSTALL_PREFIX=/your/path")
install(TARGETS openfhe_numpy LIBRARY DESTINATION ${Python_Install_Location})

if(BUILD_EXAMPLES)
    file(GLOB EXAMPLE_SOURCES "src/cpp/examples/*.cpp")

    foreach(example_file ${EXAMPLE_SOURCES})
        get_filename_component(example_name ${example_file} NAME_WE)
        add_executable(${example_name} ${example_file})
        target_include_directories(${example_name} PRIVATE ${OpenFHE_Numpy_INCLUDES})
        target_link_libraries(${example_name} PRIVATE openfhe_numpy_cpp)
    endforeach()
endif()
