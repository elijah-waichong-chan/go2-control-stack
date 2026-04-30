cmake_minimum_required(VERSION 3.8)
project(icon_lab_d1_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ServoCommand.msg"
  "msg/ServoFeedback.msg"
  DEPENDENCIES builtin_interfaces std_msgs
)

install(PROGRAMS
  scripts/ros_command_udp_forwarder.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME ros_command_udp_forwarder
)

install(PROGRAMS
  scripts/ros_command_topic_repeater.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME ros_command_topic_repeater
)

install(PROGRAMS
  scripts/udp_feedback_ros_publisher.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME udp_feedback_ros_publisher
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
