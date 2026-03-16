cmake_minimum_required(VERSION 3.8)

project(go2_d1_integration)

find_package(ament_cmake REQUIRED)

foreach(dir urdf)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}")
    install(
      DIRECTORY ${dir}/
      DESTINATION share/${PROJECT_NAME}/${dir}
    )
  endif()
endforeach()

install(FILES package.xml DESTINATION share/${PROJECT_NAME})

ament_package()
