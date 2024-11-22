cmake_minimum_required(VERSION 3.10)  # 指定CMake最低版本
project(OpenCV_work)

# 设置C++11标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB IMAGE_FILES "images/*.png")

set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/package/opencv")
find_package(OpenCV REQUIRED)

# 显示OpenCV库路径
message(STATUS "OpenCV library status:")
message(STATUS "OpenCV VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV Directory: ${OpenCV_DIR}")
message(STATUS "OpenCV Include Directory: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV Library Directory: ${OpenCV_LIBS}")
message(STATUS "PNG File: ${IMAGE_FILES}")

# 添加可执行文件
add_executable(work1 work1.cpp)
add_executable(work2 work2.cpp)
add_executable(work3 work3.cpp)
add_executable(work4 work4.cpp)
#add_executable(work5 work5.cpp)
add_executable(work6 work6.cpp)
add_executable(work7 work7.cpp)
add_executable(work8 work8.cpp)
add_executable(work9 work9.cpp)
add_executable(work10 work10.cpp)

target_link_libraries(work1 ${OpenCV_LIBS})
target_link_libraries(work2 ${OpenCV_LIBS})
target_link_libraries(work3 ${OpenCV_LIBS})
target_link_libraries(work4 ${OpenCV_LIBS})
#target_link_libraries(work5 ${OpenCV_LIBS})
target_link_libraries(work6 ${OpenCV_LIBS})
target_link_libraries(work7 ${OpenCV_LIBS})
target_link_libraries(work8 ${OpenCV_LIBS})
target_link_libraries(work9 ${OpenCV_LIBS})
target_link_libraries(work10 ${OpenCV_LIBS})