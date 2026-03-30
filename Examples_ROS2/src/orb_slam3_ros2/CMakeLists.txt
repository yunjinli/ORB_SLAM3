cmake_minimum_required(VERSION 3.8)
project(orb_slam3_ros2)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(message_filters REQUIRED)

# Link directly to the local git repo's build tree
set(ORB_SLAM3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
include_directories(
  ${ORB_SLAM3_DIR}
  ${ORB_SLAM3_DIR}/include
  ${ORB_SLAM3_DIR}/include/CameraModels
  ${ORB_SLAM3_DIR}/Thirdparty/Sophus
  ${EIGEN3_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(mono_node src/mono_node.cpp)
ament_target_dependencies(mono_node rclcpp sensor_msgs cv_bridge std_msgs geometry_msgs OpenCV tf2_ros tf2_geometry_msgs nav_msgs visualization_msgs)
target_link_libraries(mono_node 
  ${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so 
  ${ORB_SLAM3_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
  ${ORB_SLAM3_DIR}/Thirdparty/g2o/lib/libg2o.so
  ${Pangolin_LIBRARIES}
  boost_serialization
  crypto
)

add_executable(rgbd_node src/rgbd_node.cpp)
ament_target_dependencies(rgbd_node rclcpp sensor_msgs cv_bridge std_msgs geometry_msgs OpenCV tf2_ros tf2_geometry_msgs nav_msgs visualization_msgs message_filters)
target_link_libraries(rgbd_node 
  ${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so 
  ${ORB_SLAM3_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
  ${ORB_SLAM3_DIR}/Thirdparty/g2o/lib/libg2o.so
  ${Pangolin_LIBRARIES}
  boost_serialization
  crypto
)

install(TARGETS
  mono_node
  rgbd_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
