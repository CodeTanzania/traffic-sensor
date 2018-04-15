project(traffic-sensor)
cmake_minimum_required(VERSION 2.8.11)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -Wall")
set("${PROJECT_NAME}_VERSION_MAJOR" 1)
set("${PROJECT_NAME}_VERSION_MINOR" 0)

set (CMAKE_INCLUDE_CURRENT_DIR ON)
set (CMAKE_AUTOMOC ON)


# Configure a header file to pass some of the CMake settings
# to the source code.
# configure_file(
#  "${PROJECT_SOURCE_DIR}/Config.h.in"
#  "${PROJECT_BINARY_DIR}/Config.h"
#)

# turn debug on debug information
# option (DEBUG "Turn On Debug Information" ON)

# message("Include Directories for the project ${PROJECT_NAME} are located in $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")

find_package(OpenCV  REQUIRED)
# find_package(Qt5Core REQUIRED)
# find_package(sqlite3 REQUIRED)
# find_package(jsoncpp REQUIRED)
file(GLOB SOURCES "src/*.cpp")

add_subdirectory(patterns)
add_subdirectory(preprocessing)

add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
# link with Open CV and Sqlite3 libs
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})


