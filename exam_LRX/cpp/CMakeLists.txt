cmake_minimum_required(VERSION 3.10)  # 指定CMake最低版本
project(OpenCV_work)

# 设置C++11标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置OpenCV的本地路径
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/package/opencv")

# 指定OpenCV头文件的路径
include_directories(${OpenCV_DIR}/include)

# 指定OpenCV库的路径
link_directories(${OpenCV_DIR}/lib)

# 显示opencv
message(STATUS "OpenCV library status:")
message(STATUS "OpenCV Include Directory: ${OpenCV_DIR}/include")
message(STATUS "OpenCV Library Directory: ${OpenCV_DIR}/lib")

# 添加可执行文件
add_executable(work1 work1.cpp)

# 链接所需的OpenCV库
target_link_libraries(work1 opencv_core opencv_imgcodecs opencv_highgui)


