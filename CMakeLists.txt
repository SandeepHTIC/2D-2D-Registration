cmake_minimum_required(VERSION 3.10)
project(PDProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GDCM Configuration
find_package(GDCM REQUIRED)
include_directories(${GDCM_INCLUDE_DIRS})
add_definitions(-DGDCM_USE_DYNAMIC_LIBRARIES)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen3 (auto-detect via MSYS path)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Local Include Files (json.hpp, etc.)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source Files
set(SOURCES
    src/maindcm.cpp
    src/Center_detection.cpp
    src/Outlier_angle.cpp
    src/S_F_B.cpp
    src/dbscan.cpp
    src/blob_detection.cpp
    src/icp_angle.cpp
    src/kmeans.cpp
    src/plate12icp.cpp
    src/plate12indexing.cpp
    src/crop_image.cpp
    src/Calibration.cpp
    src/ref_conversion.cpp
    src/estimate_camera_matrix.cpp
    src/decompose_camera.cpp
    src/registration_check.cpp
)


# Executable
add_executable(main ${SOURCES})

# Link Libraries
target_link_libraries(main
    ${OpenCV_LIBS}
    ${GDCM_LIBRARIES}
    gdcmMSFF
    gdcmCommon
    gdcmDSED
)