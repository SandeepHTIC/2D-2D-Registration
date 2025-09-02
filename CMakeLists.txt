cmake_minimum_required(VERSION 3.15)
project(PDProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Prefer static libraries ---
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_FIND_LIBRARY_SUFFIXES .a .lib)

# --- Compiler-specific runtime settings ---
if(MSVC)
    # MSVC: Use static runtime (/MT or /MTd)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    # GCC / Clang: Link standard library statically
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -static-libgcc -static-libstdc++")
endif()

# --- GDCM ---
find_package(GDCM REQUIRED)
include_directories(${GDCM_INCLUDE_DIRS})
# Do NOT add -DGDCM_USE_DYNAMIC_LIBRARIES

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

# --- Link static libraries ---
target_link_libraries(main
    ${OpenCV_LIBS}
    ${GDCM_LIBRARIES}
    gdcmMSFF
    gdcmCommon
    gdcmDSED
)
# --- Portable install (bundle DLLs from MSYS2 MinGW) ---
include(GNUInstallDirs)
install(TARGETS main RUNTIME DESTINATION .)

if(WIN32)
  # Run at install time, not configure time
  install(CODE [[
    include(BundleUtilities)
    set(BU_CHMOD_BUNDLE_ITEMS ON)
    # Your DLLs location
    set(dirs "C:/msys64/mingw64/bin")
    # Copy non-system DLLs next to main.exe
    fixup_bundle("${CMAKE_INSTALL_PREFIX}/main.exe" "" "${dirs}")
  ]])
endif()


