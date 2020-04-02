cmake_minimum_required(VERSION 3.15)
project(Noise)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Noise main.cpp noise.cpp noise.h denoise.cpp denoise.h wavelet2d.cpp wavelet2d.h)

target_link_libraries(Noise ${OpenCV_LIBS} /usr/local/lib/libfftw3.a)