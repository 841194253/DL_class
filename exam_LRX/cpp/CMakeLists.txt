cmake_minimum_required(VERSION 3.10)  # 指定CMake最低版本
project(OpenCV_work)

# 设置C++11标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/package/opencv")
find_package(OpenCV REQUIRED)

# 包含 OpenCV 头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 显示OpenCV库路径
message(STATUS "OpenCV library status:")
message(STATUS "OpenCV Directory: ${OpenCV_DIR}")
message(STATUS "OpenCV Include Directory: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV Library Directory: ${OpenCV_LIBS}")

# 添加可执行文件
add_executable(work1 work1.cpp)

target_link_libraries(work1 ${OpenCV_LIBS})