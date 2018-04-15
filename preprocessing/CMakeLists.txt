project(preprocessing CXX)
cmake_minimum_required(VERSION 2.8)

set(HEADERS emavg.h wmavg.h)
set(SRCS emavg.cpp)
set(CMAKE_BUILT_TYPE Release)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(${PROJECT_NAME} SHARED ${SRCS})
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)