cmake_minimum_required(VERSION 3.10)  # 指定CMake最低版本
project(OpenCV_work)

# 设置C++11标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB IMAGE_FILES "images/*.png")

set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/package/opencv")
find_package(OpenCV REQUIRED)
#set(OpenCV_LIBS_DIR_1 "${OpenCV_DIR}/x64/mingw/lib")
#set(OpenCV_LIBS_DIR_2 "${OpenCV_DIR}/x64/mingw/bin")

# 包含 OpenCV 头文件
#include_directories(${OpenCV_INCLUDE_DIRS})
#link_directories(${OpenCV_LIBS_DIR_1})
#link_directories(${OpenCV_LIBS_DIR_2})

# 显示OpenCV库路径
message(STATUS "OpenCV library status:")
message(STATUS "OpenCV VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV Directory: ${OpenCV_DIR}")
message(STATUS "OpenCV Include Directory: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV Library Directory: ${OpenCV_LIBS}")
message(STATUS "PNG File: ${IMAGE_FILES}")
#message(STATUS "OpenCV Library Directory: ${OpenCV_LIBS_DIR_1}")
#message(STATUS "OpenCV Library Directory: ${OpenCV_LIBS_DIR_2}")


# 添加可执行文件
add_executable(work1 work1.cpp)
add_executable(work2 work2.cpp)
add_executable(work3 work3.cpp)
add_executable(work4 work4.cpp)
add_executable(work5 work5.cpp)


target_link_libraries(work1 ${OpenCV_LIBS})
target_link_libraries(work2 ${OpenCV_LIBS})
target_link_libraries(work3 ${OpenCV_LIBS})
target_link_libraries(work4 ${OpenCV_LIBS})
target_link_libraries(work5 ${OpenCV_LIBS})