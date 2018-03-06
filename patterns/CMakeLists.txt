project(patterns CXX)
cmake_minimum_required(VERSION 2.8)

set(HEADERS subject.h observer.h)
set(CMAKE_BUILT_TYPE Release)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(${PROJECT_NAME} SHARED ${HEADERS})
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
