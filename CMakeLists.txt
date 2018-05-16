cmake_minimum_required(VERSION 2.8.3)
project(local_map)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        sensor_msgs
        cv_bridge
        image_transport)


catkin_package(
)


find_package(PCL REQUIRED)
add_definitions(${PCL_DEFINITIONS})


include_directories(${PCL_INCLUDE_DIRS})
include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(cloud_filter src/cloud_filter.cpp)
target_link_libraries(cloud_filter ${catkin_LIBRARIES}  ${PCL_LIBRARIES})
