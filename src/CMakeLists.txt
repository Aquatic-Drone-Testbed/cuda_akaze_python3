cmake_minimum_required(VERSION 3.14)
project(cuda_akaze)

# Find external libraries and dependencies
find_package(OpenCV REQUIRED)
find_package(Doxygen)
find_package(CUDA)
find_package(pybind11 REQUIRED)  # <-- Use pybind11 instead of Boost.Python

# ===================================
# Set this to yes for python bindings
option(USE_PYTHON "Build Python bindings" OFF)

if(USE_PYTHON)
  include_directories(${pybind11_INCLUDE_DIRS})
endif()

if (NOT APPLE)
  find_package(OpenMP)
endif ()

# ============================================================================ #
# Compilation flags
if(UNIX)
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -g -O0 -Wall -Wextra -Wunused-variable -DDEBUG -D_DEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g -Wall -Wextra -Wunused-variable -DDEBUG -D_DEBUG")
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_50;-lineinfo;--compiler-options;-O2")
endif()

if(OPENMP_FOUND)
  message("OpenMP found")
  if(UNIX)
    set(CMAKE_C_FLAGS_RELEASE "-O3 -Wall -Wextra -Wunused-variable -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra -Wunused-variable -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
else()
  message("OpenMP not found")
  if(UNIX)
    if(APPLE)
      set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3 -Wall -std=c++11 -Wunused-variable -Wno-unknown-pragmas -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
      set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wall -std=c++11 -stdlib=libc++ -Wno-unknown-pragmas -Wunused-variable -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
    else()
      set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3 -Wall -std=c++0x -Wunused-variable -Wno-unknown-pragmas -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
      set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wall -std=c++0x -Wno-unknown-pragmas -Wunused-variable -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
    endif()
  endif()
endif()

# ============================================================================ #
# Include directories
include_directories("lib/" ${CUDASIFT})

# AKAZE libraries
set(AKAZE_SRCS
    lib/AKAZEConfig.h
    lib/AKAZE.h                  lib/AKAZE.cpp
    lib/fed.h                    lib/fed.cpp
    lib/nldiffusion_functions.h  lib/nldiffusion_functions.cpp
    lib/utils.h                  lib/utils.cpp)
 
add_library(AKAZE ${AKAZE_SRCS})
target_link_libraries(AKAZE ${OpenCV_LIBS})

# CUDA sources
set(AKAZE_CUDA_SRCS
  lib/cudaImage.cu lib/cuda_akaze.cu)

set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_library(AKAZE_CUDA ${AKAZE_CUDA_SRCS})
target_link_libraries(AKAZE_CUDA ${OpenCV_LIBS})

# Python bindings with pybind11
if(USE_PYTHON)
  pybind11_add_module(cuda_akaze_py cuda_akaze_py.cpp)  # Your Python binding source
  target_link_libraries(cuda_akaze_py PRIVATE AKAZE AKAZE_CUDA ${OpenCV_LIBS})
endif()

# Build examples
if(BUILD_EXAMPLES)
  add_executable(akaze_features akaze_features.cpp)
  target_link_libraries(akaze_features AKAZE AKAZE_CUDA)

  add_executable(akaze_match akaze_match.cpp)
  target_link_libraries(akaze_match AKAZE AKAZE_CUDA)

  add_executable(akaze_compare akaze_compare.cpp)
  target_link_libraries(akaze_compare AKAZE AKAZE_CUDA)
endif()

# ============================================================================ #
# Library installation
install(TARGETS AKAZE AKAZE_CUDA DESTINATION lib)
install(FILES
    lib/AKAZE.h
    lib/fed.h
    lib/utils.h
    lib/nldiffusion_functions.h
    lib/AKAZEConfig.h
    lib/cudaImage.h
    lib/cuda_akaze.h
    lib/cudautils.h
    DESTINATION
    include/cuda_akaze
)

# ============================================================================ #
# Generate Doxygen-based documentation project
if(DOXYGEN_FOUND)
  add_custom_target(akaze_documentation
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif()

