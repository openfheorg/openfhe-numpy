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

option(BUILD_EXAMPLES "Build C++ example executables" ON)
option(BUILD_TESTING "Enable building of tests" OFF)

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

set(OpenFHE_Numpy_INCLUDES openfhe_numpy/cpp/include)
include_directories( ${OpenFHE_Numpy_INCLUDES} )

# always link with shared libs
set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
link_libraries(${OpenFHE_SHARED_LIBRARIES})

### Pybind Modules
set(OpenFHE_Numpy_SOURCE openfhe_numpy/cpp/lib)
pybind11_add_module(openfhe_numpy 
                    ${OpenFHE_Numpy_SOURCE}/numpy_array_metadata.cpp
                    ${OpenFHE_Numpy_SOURCE}/numpy_enc_matrix.cpp
                    ${OpenFHE_Numpy_SOURCE}/numpy_helper_functions.cpp
                    ${OpenFHE_Numpy_SOURCE}/numpy_utils.cpp
                    ${OpenFHE_Numpy_SOURCE}/numpy_bindings.cpp
                    )

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

