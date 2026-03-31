```
ros2 run orb_slam3_ros2 rgbd_node --ros-args   --remap camera/rgb/image_raw:=/camera/rgb/image_color   --remap camera/depth/image_raw:=/camera/depth/image   -p vocab_file:=/workspaces/ORB_SLAM3/Vocabulary/ORBvoc.txt   -p settings_file:=/workspaces/ORB_SLAM3/Examples_ROS2/TUM3_ROS2.yaml
```