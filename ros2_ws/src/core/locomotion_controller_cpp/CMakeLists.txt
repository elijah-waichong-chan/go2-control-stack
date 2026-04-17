cmake_minimum_required(VERSION 3.8)
project(locomotion_controller_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(hq_pcot_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(unitree_go REQUIRED)
find_package(yaml-cpp REQUIRED)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Root directory of an ONNX Runtime C++ installation")
if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()
find_path(
  ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT}
  PATH_SUFFIXES
    include
    onnxruntime/include
    onnxruntime/core/session
    include/onnxruntime/core/session
)
find_library(
  ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  HINTS ${ONNXRUNTIME_ROOT}
  PATH_SUFFIXES lib lib64
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR
    "ONNX Runtime C++ headers/library not found. "
    "Set ONNXRUNTIME_ROOT, or provide ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIBRARY."
  )
endif()

add_library(locomotion_controller_cpp_common INTERFACE)
target_include_directories(
  locomotion_controller_cpp_common
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_executable(stand_up_init src/stand_up_init.cpp)
target_link_libraries(stand_up_init locomotion_controller_cpp_common)
ament_target_dependencies(stand_up_init rclcpp hq_pcot_msgs unitree_go)

add_executable(wireless_cmd_bridge src/wireless_cmd_bridge.cpp)
target_link_libraries(wireless_cmd_bridge locomotion_controller_cpp_common)
ament_target_dependencies(wireless_cmd_bridge rclcpp hq_pcot_msgs unitree_go)

add_executable(policy_controller src/policy_controller.cpp)
target_include_directories(policy_controller PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(
  policy_controller
  locomotion_controller_cpp_common
  yaml-cpp
  ${ONNXRUNTIME_LIBRARY}
)
ament_target_dependencies(
  policy_controller
  ament_index_cpp
  hq_pcot_msgs
  rclcpp
  unitree_go
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS
    stand_up_init
    wireless_cmd_bridge
    policy_controller
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY
    config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
