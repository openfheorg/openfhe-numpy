# Minimum required CMake version - needs at least 3.16 for modern features
cmake_minimum_required(VERSION 3.16)

# Project declaration with name, version, and languages
project(OpenFHE-Numpy VERSION 0.0.1 LANGUAGES CXX)

# Version variables - stored individually for component access and compatibility
set(OPENFHE_NUMPY_VERSION_MAJOR 0)
set(OPENFHE_NUMPY_VERSION_MINOR 0)
set(OPENFHE_NUMPY_VERSION_PATCH 1)
set(OPENFHE_NUMPY_VERSION ${OPENFHE_NUMPY_VERSION_MAJOR}.${OPENFHE_NUMPY_VERSION_MINOR}.${OPENFHE_NUMPY_VERSION_PATCH})

# -----------------------------------------------------------------------------
# C++ Standards and Build Options
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)               # Use C++17 standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)      # Require C++17 (don't fall back)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Make position-independent code for shared libs
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF) # Disable link-time optimization

# User-configurable options with default values
option(BUILD_STATIC "Build static versions of the library" OFF)
option(ENABLE_DEBUG_LOGGING "Enable debug logging (defines DEBUG)" OFF)
option(BUILD_EXAMPLES "Build C++ example executables" ON)
option(BUILD_TESTING "Enable building of tests" OFF)

# Platform-specific settings
if(APPLE)
    set(CMAKE_CXX_VISIBILITY_PRESET default) # Fix visibility on macOS
endif()

# Modern CMake policies
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW) # Use new Python finding mechanism
endif()

# -----------------------------------------------------------------------------
# External Dependencies
# -----------------------------------------------------------------------------
# Find required and optional libraries
find_package(OpenFHE 1.3.0 REQUIRED)     # Homomorphic Encryption library
find_package(pybind11 REQUIRED)          # Python binding generator
find_package(OpenMP QUIET)               # Optional parallel processing 
find_package(Python REQUIRED             # Python interpreter and libraries
             COMPONENTS Interpreter Development)

# Add OpenFHE library path to linker search path (critical for linking)
link_directories(${OpenFHE_LIBDIR})

# -----------------------------------------------------------------------------
# Project Directory Structure
# -----------------------------------------------------------------------------
# Define key directories for the project
set(OpenFHE_Numpy_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/core/src)       # C++ source files
set(OpenFHE_Numpy_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/core/include)  # C++ header files
set(OpenFHE_Numpy_BINDINGS ${OpenFHE_Numpy_SOURCES}/bindings.cpp)    # Python bindings
set(OpenFHE_Numpy_PYTHON ${CMAKE_CURRENT_SOURCE_DIR}/openfhe_numpy)  # Python source
set(OpenFHE_Numpy_BUILD_PYTHON ${CMAKE_BINARY_DIR}/openfhe_numpy)    # Python build output

# Create the Python build output directory
file(MAKE_DIRECTORY ${OpenFHE_Numpy_BUILD_PYTHON})

# -----------------------------------------------------------------------------
# Generate Configuration Header
# -----------------------------------------------------------------------------
# Create config.h from template with version information
configure_file(
    "${OpenFHE_Numpy_INCLUDES}/openfhe_numpy/config.h.in"  # Template header
    "${OpenFHE_Numpy_INCLUDES}/openfhe_numpy/config.h"     # Generated header
    @ONLY  # Replace only @VAR@ style variables
)

# -----------------------------------------------------------------------------
# Include Directories Setup
# -----------------------------------------------------------------------------
# Add include directories globally
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)
include_directories(${OpenFHE_Numpy_INCLUDES})

# -----------------------------------------------------------------------------
# Source Files Collection
# -----------------------------------------------------------------------------
# Find all C++ source files except bindings.cpp
file(GLOB_RECURSE CORE_SOURCES
    ${OpenFHE_Numpy_SOURCES}/*.cpp
)
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*/bindings?\\.cpp$")

# -----------------------------------------------------------------------------
# Core C++ Library Target
# -----------------------------------------------------------------------------
# Create either static or shared library based on option
if(BUILD_STATIC)
    add_library(openfhe_core STATIC ${CORE_SOURCES})
else()
    add_library(openfhe_core SHARED ${CORE_SOURCES})
endif()

# Add include directories specifically for the core library
target_include_directories(openfhe_core PRIVATE 
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/third-party/include
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe
    ${OpenFHE_Numpy_INCLUDES}
    ${OPENMP_INCLUDES}
)

# Link against OpenFHE libraries (static or shared based on option)
if(BUILD_STATIC)
    # Set linker flags for static linking
    set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    target_link_libraries(openfhe_core PRIVATE 
        ${OpenFHE_STATIC_LIBRARIES}
        ${OpenFHE_PKE_STATIC_LIBRARIES}
        ${OpenFHE_BINFHE_STATIC_LIBRARIES}
    )
else()
    # Use default linker flags
    set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
    target_link_libraries(openfhe_core PRIVATE 
        ${OpenFHE_SHARED_LIBRARIES}
        ${OpenFHE_PKE_SHARED_LIBRARIES}
        ${OpenFHE_BINFHE_SHARED_LIBRARIES}
    )
endif()

# Link OpenMP if available for parallel processing
if(OpenMP_CXX_FOUND)
    target_link_libraries(openfhe_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# -----------------------------------------------------------------------------
# Python Extension Module
# -----------------------------------------------------------------------------
# Create Python extension module from bindings.cpp
pybind11_add_module(_openfhe_numpy MODULE ${OpenFHE_Numpy_BINDINGS})
target_link_libraries(_openfhe_numpy PRIVATE 
    openfhe_core
    ${OpenFHE_PKE_SHARED_LIBRARIES}  # Explicitly link PKE library
)

# Generate version.py with Python package version info in the build directory
configure_file(
    "${OpenFHE_Numpy_PYTHON}/version.py.in"
    "${OpenFHE_Numpy_BUILD_PYTHON}/version.py"
    @ONLY
)

# Set output directory for Python extension module to build directory
set_target_properties(_openfhe_numpy PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${OpenFHE_Numpy_BUILD_PYTHON}
)

# Copy Python source files to the build directory
file(GLOB_RECURSE PYTHON_SOURCE_FILES "${OpenFHE_Numpy_PYTHON}/*.py")
foreach(PY_FILE ${PYTHON_SOURCE_FILES})
    file(RELATIVE_PATH REL_PATH "${OpenFHE_Numpy_PYTHON}" "${PY_FILE}")
    configure_file(${PY_FILE} "${OpenFHE_Numpy_BUILD_PYTHON}/${REL_PATH}" COPYONLY)
endforeach()

# -----------------------------------------------------------------------------
# Installation Configuration
# -----------------------------------------------------------------------------
include(GNUInstallDirs)  # Include standard installation directories

# Get Python site-packages directory for installation
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Determine where to install Python package
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(PYTHON_INSTALL_LOCATION "${PYTHON_SITE_PACKAGES}")
else()
    set(PYTHON_INSTALL_LOCATION "${CMAKE_INSTALL_PREFIX}")
endif()

message(STATUS "Python package will be installed to: ${PYTHON_INSTALL_LOCATION}")

# Install Python module and package files
# (Python requires all source files as it's an interpreted language)
install(TARGETS _openfhe_numpy 
    LIBRARY DESTINATION "${PYTHON_INSTALL_LOCATION}/openfhe_numpy"
)

install(DIRECTORY "${OpenFHE_Numpy_PYTHON}/" 
    DESTINATION "${PYTHON_INSTALL_LOCATION}/openfhe_numpy"
    FILES_MATCHING PATTERN "*.py"
)

# Install generated version.py file
install(FILES "${OpenFHE_Numpy_BUILD_PYTHON}/version.py"
    DESTINATION "${PYTHON_INSTALL_LOCATION}/openfhe_numpy"
)

# Install C++ library
install(TARGETS openfhe_core
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

# Install C++ header files except helper.h
install(DIRECTORY "${OpenFHE_Numpy_INCLUDES}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.h"
    PATTERN "internal/*" EXCLUDE)
#     PATTERN "*helper.h" EXCLUDE
#     PATTERN "*array_metadata.h" EXCLUDE
# )


# -----------------------------------------------------------------------------
# Examples Build Configuration
# -----------------------------------------------------------------------------
# Examples section with fixed linking
if(BUILD_EXAMPLES)
    # Create directory for example executables
    set(EXAMPLE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/examples)
    file(MAKE_DIRECTORY ${EXAMPLE_OUTPUT_DIR})
    
    # Find the exact OpenFHE library files needed
    find_library(OPENFHE_CORE_LIB OPENFHEcore PATHS ${OpenFHE_LIBDIR} REQUIRED)
    find_library(OPENFHE_PKE_LIB OPENFHEpke PATHS ${OpenFHE_LIBDIR} REQUIRED)
    find_library(OPENFHE_BINFHE_LIB OPENFHEbinfhe PATHS ${OpenFHE_LIBDIR} REQUIRED)

    # Find and build all example files
    file(GLOB EXAMPLES ${CMAKE_CURRENT_SOURCE_DIR}/core/examples/*.cpp)
    foreach(ex IN LISTS EXAMPLES)
        # Get example name without extension
        get_filename_component(name ${ex} NAME_WE)
        
        # Read file content to check for main function
        file(READ ${ex} FILE_CONTENT)
        if(NOT FILE_CONTENT MATCHES "^\\s*//")
            # Create executable if file isn't completely commented out
            add_executable(${name} ${ex})
            
            # Link directly against library files in correct order
            target_link_libraries(${name} PRIVATE 
                openfhe_core
                ${OPENFHE_PKE_LIB}
                ${OPENFHE_CORE_LIB}
                ${OPENFHE_BINFHE_LIB}
            )
            
            # Set output directory for example
            set_target_properties(${name} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${EXAMPLE_OUTPUT_DIR}
            )
        endif()
    endforeach()
endif()

# # -----------------------------------------------------------------------------
# # Testing Configuration
# # -----------------------------------------------------------------------------
# if(BUILD_TESTING)
#     enable_testing()
#     # Add test directory or specific test executables here
#     # add_subdirectory(test)
# endif()