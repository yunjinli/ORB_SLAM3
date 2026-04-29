## Docker

```
ros2 run orb_slam3_ros2 rgbd_node --ros-args   --remap camera/rgb/image_raw:=/camera/rgb/image_color   --remap camera/depth/image_raw:=/camera/depth/image   -p vocab_file:=/workspaces/ORB_SLAM3/Vocabulary/ORBvoc.txt   -p settings_file:=/workspaces/ORB_SLAM3/Examples_ROS2/TUM3_ROS2.yaml
```

## Pixi

```
pixi run ros2 run orb_slam3_ros2 rgbd_node --ros-args   --remap camera/rgb/image_raw:=/camera/rgb/image_color   --remap camera/depth/image_raw:=/camera/depth/image   -p vocab_file:=./Vocabulary/ORBvoc.txt   -p settings_file:=./Examples_ROS2/TUM3_ROS2.yaml
```
```
pixi run python Examples_ROS2/slam_data_saver.py --output-dir ./output --settings-file ./Examples_ROS2/TUM3_ROS2.yaml --ros-args --remap camera/rgb/image_raw:=/camera/rgb/image_color   --remap camera/depth/image_raw:=/camera/depth/image
```
```
pixi run python Examples_ROS2/slam_to_shaper.py --slam-dir ./output/ --output-dir ./data/rgbd_fr3_orb --settings ./Examples_ROS2/TUM3_ROS2.yaml --stride 100 --max-frames 8
```