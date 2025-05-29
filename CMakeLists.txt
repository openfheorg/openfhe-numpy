# Minimum required CMake version - needs at least 3.16 for modern features
cmake_minimum_required(VERSION 3.16)

# Project declaration with name, version, and languages
project(OpenFHE-Numpy)

# Version variables
set(OPENFHE_NUMPY_VERSION_MAJOR 0)
set(OPENFHE_NUMPY_VERSION_MINOR 0)
set(OPENFHE_NUMPY_VERSION_PATCH 1)
set(OPENFHE_NUMPY_VERSION ${OPENFHE_NUMPY_VERSION_MAJOR}.${OPENFHE_NUMPY_VERSION_MINOR}.${OPENFHE_NUMPY_VERSION_PATCH})

# Build Options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

option(ENABLE_DEBUG_LOGGING "Enable debug logging (defines DEBUG)" OFF)
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
find_package(OpenMP QUIET)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

link_directories(${OpenFHE_LIBDIR})

# Debug OpenFHE
message(STATUS "OpenFHE version: ${OpenFHE_VERSION}")
message(STATUS "OpenFHE libraries: ${OpenFHE_LIBRARIES}")
message(STATUS "OpenFHE include path: ${OpenFHE_INCLUDE}")
message(STATUS "OpenFHE library path: ${OpenFHE_LIBDIR}")

# Project Structure
set(OpenFHE_Numpy_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/openfhe_numpy/cpp/lib)
set(OpenFHE_Numpy_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/openfhe_numpy/cpp/include)
set(OpenFHE_Numpy_BINDINGS ${OpenFHE_Numpy_SOURCES}/bindings.cpp)
# Updated path to reflect new directory structure
set(OpenFHE_Numpy_PYTHON ${CMAKE_CURRENT_SOURCE_DIR})
set(OpenFHE_Numpy_BUILD_PYTHON ${CMAKE_BINARY_DIR}/openfhe_numpy)

file(MAKE_DIRECTORY ${OpenFHE_Numpy_BUILD_PYTHON})

# Configuration Header
configure_file(./config/config.h.in config/config.h)
install(FILES ${CMAKE_BINARY_DIR}/config/config_core.h DESTINATION include/config)

# Include Paths
include_directories(
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/third-party/include
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe
    ${OpenFHE_Numpy_INCLUDES})
message(STATUS "================== CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Source Files
file(GLOB_RECURSE CORE_SOURCES ${OpenFHE_Numpy_SOURCES}/*.cpp)
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*/bindings?\\.cpp$")

add_library(openfhe_numpy_cpp_core STATIC ${CORE_SOURCES})

target_include_directories(openfhe_numpy_cpp_core PUBLIC
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/third-party/include
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe
    ${OpenFHE_Numpy_INCLUDES})

target_link_libraries(openfhe_numpy_cpp_core PUBLIC
    ${OpenFHE_SHARED_LIBRARIES}
    ${OpenFHE_PKE_SHARED_LIBRARIES}
    ${OpenFHE_BINFHE_SHARED_LIBRARIES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(openfhe_numpy_cpp_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Python Extension
pybind11_add_module(_openfhe_numpy MODULE ${OpenFHE_Numpy_BINDINGS})
target_link_libraries(_openfhe_numpy PRIVATE openfhe_numpy_cpp_core)

configure_file(
    "${OpenFHE_Numpy_PYTHON}/version.py.in"
    "${OpenFHE_Numpy_BUILD_PYTHON}/version.py" @ONLY)

# Set output location for the compiled Python extension
set_target_properties(_openfhe_numpy PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${OpenFHE_Numpy_BUILD_PYTHON}/openfhe_numpy
    INSTALL_RPATH "$ORIGIN;${OpenFHE_LIBDIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Copy Python files to build directory
file(GLOB_RECURSE PYTHON_SOURCE_FILES "${OpenFHE_Numpy_PYTHON}/*.py")
foreach(PY_FILE ${PYTHON_SOURCE_FILES})
    file(RELATIVE_PATH REL_PATH "${OpenFHE_Numpy_PYTHON}" "${PY_FILE}")
    configure_file(${PY_FILE} "${OpenFHE_Numpy_BUILD_PYTHON}/${REL_PATH}" COPYONLY)
endforeach()

# Installation Configuration
include(GNUInstallDirs)

# For pip/scikit-build installations
if(DEFINED SKBUILD)
    set(PYTHON_INSTALL_PATH "${SKBUILD_PROJECT_NAME}")
    message(STATUS "Building for pip/scikit-build: ${PYTHON_INSTALL_PATH}")
else()
    # Determine installation path based on CMAKE_INSTALL_PREFIX
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
            OUTPUT_VARIABLE PYTHON_SITE_PACKAGES 
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(PYTHON_INSTALL_PATH "${PYTHON_SITE_PACKAGES}")
    else()
        set(PYTHON_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}")
    endif()
endif()

message(STATUS "Python package will be installed to: ${PYTHON_INSTALL_PATH}")

# Install the Python extension and files 
if(DEFINED SKBUILD)
    # For pip installations, keep the /openfhe_numpy subdirectory
    install(TARGETS _openfhe_numpy 
            LIBRARY DESTINATION "${PYTHON_INSTALL_PATH}/openfhe_numpy")

    install(DIRECTORY "${OpenFHE_Numpy_PYTHON}/"
            DESTINATION "${PYTHON_INSTALL_PATH}/openfhe_numpy"
            FILES_MATCHING PATTERN "*.py"
            PATTERN "__pycache__" EXCLUDE)

    install(FILES "${OpenFHE_Numpy_BUILD_PYTHON}/version.py"
            DESTINATION "${PYTHON_INSTALL_PATH}/openfhe_numpy")
else()
    # For custom installations, install directly to specified path
    install(TARGETS _openfhe_numpy 
            LIBRARY DESTINATION "${PYTHON_INSTALL_PATH}")

    install(DIRECTORY "${OpenFHE_Numpy_PYTHON}/"
            DESTINATION "${PYTHON_INSTALL_PATH}"
            FILES_MATCHING PATTERN "*.py"
            PATTERN "__pycache__" EXCLUDE)

    install(FILES "${OpenFHE_Numpy_BUILD_PYTHON}/version.py"
            DESTINATION "${PYTHON_INSTALL_PATH}")
endif()

# Build Examples
if(BUILD_EXAMPLES)
    set(EXAMPLE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/examples)
    file(MAKE_DIRECTORY ${EXAMPLE_OUTPUT_DIR})

    find_library(OPENFHE_CORE_LIB OPENFHEcore PATHS ${OpenFHE_LIBDIR} REQUIRED)
    find_library(OPENFHE_PKE_LIB OPENFHEpke PATHS ${OpenFHE_LIBDIR} REQUIRED)
    find_library(OPENFHE_BINFHE_LIB OPENFHEbinfhe PATHS ${OpenFHE_LIBDIR} REQUIRED)

    file(GLOB EXAMPLES ${CMAKE_CURRENT_SOURCE_DIR}/openfhe_numpy/cpp/examples/*.cpp)
    foreach(ex IN LISTS EXAMPLES)
        get_filename_component(name ${ex} NAME_WE)
        file(READ ${ex} FILE_CONTENT)
        if(NOT FILE_CONTENT MATCHES "^\\s*//")
            add_executable(${name} ${ex})
            target_link_libraries(${name} PRIVATE
                openfhe_numpy_cpp_core
                ${OPENFHE_PKE_LIB}
                ${OPENFHE_CORE_LIB}
                ${OPENFHE_BINFHE_LIB})
            set_target_properties(${name} PROPERTIES 
                RUNTIME_OUTPUT_DIRECTORY ${EXAMPLE_OUTPUT_DIR}
                INSTALL_RPATH "${OpenFHE_LIBDIR}")
        endif()
    endforeach()
endif()

# Testing
if(BUILD_TESTING)
    enable_testing()
endif()

# CPack configuration for creating binary packages
include(CPack)
set(CPACK_PACKAGE_NAME "openfhe-numpy")
set(CPACK_PACKAGE_VENDOR "OpenFHE Team")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "NumPy-like API for homomorphic encryption with OpenFHE")
set(CPACK_PACKAGE_VERSION "${OPENFHE_NUMPY_VERSION}")
set(CPACK_PACKAGE_VERSION_MAJOR "${OPENFHE_NUMPY_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${OPENFHE_NUMPY_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${OPENFHE_NUMPY_VERSION_PATCH}")