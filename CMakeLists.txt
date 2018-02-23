project(traffic-sensor)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

aux_source_directory(. SRC_LIST)

add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
