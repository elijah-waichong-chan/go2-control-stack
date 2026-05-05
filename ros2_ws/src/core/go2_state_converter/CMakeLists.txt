cmake_minimum_required(VERSION 3.8)
project(go2_state_converter)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(icon_lab_d1_ros2 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(go2_d1_integration REQUIRED)
find_package(unitree_go REQUIRED)

add_executable(state_converter_node src/state_converter_node.cpp)
ament_target_dependencies(
  state_converter_node
  icon_lab_d1_ros2
  rclcpp
  sensor_msgs
  unitree_go
)

install(TARGETS
  state_converter_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
