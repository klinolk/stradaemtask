cmake_minimum_required(VERSION 3.10)
project(VoxelRenderer CXX)

# Set C++ standard to at least C++11 (or 17 for better features)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable more warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

# Try to find OpenMP (optional for macOS)
find_package(OpenMP QUIET)

if(APPLE)  # macOS specific settings
    # Try to find libomp from Homebrew if standard find_package fails
    if(NOT OpenMP_FOUND)
        # Homebrew installs libomp in different locations depending on architecture
        if(EXISTS "/opt/homebrew/opt/libomp/include/omp.h")  # Apple Silicon
            set(OPENMP_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
            set(OPENMP_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
            set(OpenMP_FOUND TRUE)
            message(STATUS "Found OpenMP via Homebrew (Apple Silicon): ${OPENMP_INCLUDE_DIR}")
        elseif(EXISTS "/usr/local/opt/libomp/include/omp.h")  # Intel
            set(OPENMP_INCLUDE_DIR "/usr/local/opt/libomp/include")
            set(OPENMP_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
            set(OpenMP_FOUND TRUE)
            message(STATUS "Found OpenMP via Homebrew (Intel): ${OPENMP_INCLUDE_DIR}")
        else()
            message(WARNING "OpenMP not found. Install with: brew install libomp")
            add_definitions(-DNO_OMP)
        endif()
    endif()
    
    # If OpenMP found on macOS, set proper flags
    if(OpenMP_FOUND)
        if(OPENMP_INCLUDE_DIR)
            include_directories(${OPENMP_INCLUDE_DIR})
        endif()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
    endif()
else()  # Linux/Windows
    if(OpenMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    else()
        add_definitions(-DNO_OMP)
    endif()
endif()

# Find SDL2 library
find_package(SDL2 REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${SDL2_INCLUDE_DIRS})

# Add definition for STB image
add_compile_definitions(USE_STB_IMAGE)

# Add the executable
add_executable(voxel_renderer
    main.cpp
    utils/octree.cpp
    utils/mesh.cpp)

# Link libraries
target_link_libraries(voxel_renderer ${SDL2_LIBRARIES})

# Link OpenMP if found
if(OpenMP_FOUND)
    if(OPENMP_LIBRARY)
        target_link_libraries(voxel_renderer ${OPENMP_LIBRARY})
    else()
        target_link_libraries(voxel_renderer OpenMP::OpenMP_CXX)
    endif()
endif()

# Set output path
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR})