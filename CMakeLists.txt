cmake_minimum_required(VERSION 3.15)
project(PDProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Linking mode ---
# Prefer shared on MinGW to use import libs (DLLs). Static DCMTK rarely available on MSYS2.
set(BUILD_SHARED_LIBS ON)

# --- Compiler-specific runtime settings ---
if(MSVC)
    # MSVC: Use static runtime (/MT or /MTd)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# --- DCMTK ---
# Request components needed for color, JPEG, RLE decoding
find_package(DCMTK REQUIRED COMPONENTS dcmimage dcmimgle dcmjpeg dcmdata ofstd oflog ijg8 ijg12 ijg16 dcmdjpeg)
include_directories(${DCMTK_INCLUDE_DIRS})

# --- OpenCV ---
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# --- Eigen ---
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# --- Local Includes ---
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- Sources ---
set(SOURCES
    src/maindcm.cpp
    src/ImageInpaint.cpp
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
    src/JsonCheck.cpp
    src/RegistrationCheck.cpp
)

# --- Executable ---
add_executable(main ${SOURCES})

# --- Link libraries ---
target_link_libraries(main
    ${OpenCV_LIBS}
    ${DCMTK_LIBRARIES}
)
# --- Portable install (bundle DLLs from MSYS2 MinGW) ---
include(GNUInstallDirs)
install(TARGETS main RUNTIME DESTINATION .)

if(WIN32)
  install(CODE [[
    include(BundleUtilities)
    set(BU_CHMOD_BUNDLE_ITEMS ON)
    set(dirs "C:/msys64/mingw64/bin")
    fixup_bundle("${CMAKE_INSTALL_PREFIX}/main.exe" "" "${dirs}")
  ]])
endif()