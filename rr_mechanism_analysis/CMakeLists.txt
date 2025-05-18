cmake_minimum_required(VERSION 3.22)

# Default to C99
if(NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif()

# Default to C++20
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()

if (NOT DEFINED CMAKE_BUILD_TYPE)
    message(WARNING "Build Type not set. Defaulting to Release")
    set(CMAKE_BUILD_TYPE Release)
else()
    message("Build Type set to " ${CMAKE_BUILD_TYPE})
endif ()

project(rr_mechanism)

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(fmt REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
# find_package(pinocchio REQUIRED)
find_package(tf2_eigen REQUIRED)

##############
## Set Vars ##
##############

add_executable(rr_mechanism 
    src/rr_mechanism.cpp
    )

target_link_libraries(rr_mechanism PUBLIC
    fmt::fmt-header-only
    )

ament_target_dependencies(rr_mechanism PUBLIC
    rclcpp
    builtin_interfaces
    std_msgs
    sensor_msgs
    tf2_eigen
    )

install(TARGETS
    rr_mechanism
    DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
    launch
    resources
    DESTINATION share/${PROJECT_NAME})

ament_package()
