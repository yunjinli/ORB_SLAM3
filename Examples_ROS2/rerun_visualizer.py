#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import rerun as rr
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2

class RerunBridge(Node):
    def __init__(self):
        super().__init__('rerun_bridge')
        
        rr.init("ORB_SLAM3_Rerun_Viewer")
        rr.connect() # Automatically beams data over TCP to localhost:9876
        # Natively configure Rerun to expect Z-forward, X-right, Y-down (OpenCV / ORB_SLAM3 Native standard)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)

        self.bridge = CvBridge()
        
        # Subscriptions to our flawless C++ Node streams
        self.create_subscription(PoseStamped, '/orb_slam3/camera_pose', self.pose_callback, 10)
        self.create_subscription(Image, '/camera/rgb/image_color', self.rgb_callback, 10)
        self.create_subscription(PointCloud2, '/orb_slam3/map_points', self.map_callback, 10)
        self.create_subscription(PointCloud2, '/orb_slam3/dense_points', self.dense_callback, 10)
        
        self.get_logger().info("Rerun Bridge listening to ORB_SLAM3 streams...")

    def pose_callback(self, msg: PoseStamped):
        rr.set_time_seconds("time", msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        t, q = msg.pose.position, msg.pose.orientation
        
        # Log the dynamic camera transform floating in the global 'world'
        rr.log("world/camera", rr.Transform3D(
            translation=[t.x, t.y, t.z],
            rotation=rr.Quaternion(xyzw=[q.x, q.y, q.z, q.w])
        ))

        # Log a frustum indicator to represent the camera's physical lens
        rr.log("world/camera/frustum", rr.Pinhole(
            resolution=[640, 480],
            focal_length=[535.4, 539.2],
            principal_point=[320.1, 247.6]
        ))

    def rgb_callback(self, msg: Image):
        rr.set_time_seconds("time", msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        # Pin the image natively into the frustum's mathematical plane perfectly
        rr.log("world/camera/frustum/image", rr.Image(cv_img))

    def map_callback(self, msg: PointCloud2):
        rr.set_time_seconds("time", msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        
        # Parse sparse map points generated purely by ORB_SLAM3 features
        data = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
        if not data.size: return
        pts = np.vstack((data['x'], data['y'], data['z'])).T
        
        # Draw ORB corners natively in bright cyan across the world
        rr.log("world/sparse_map", rr.Points3D(
            positions=pts,
            colors=[[0, 255, 255]] * len(pts), 
            radii=0.015
        ))

    def dense_callback(self, msg: PointCloud2):
        rr.set_time_seconds("time", msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        
        # Parse our beautifully optimized C++ Dense Points
        data = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)))
        if not data.size: return

        # Unpack structured arrays natively
        pts = np.vstack((data['x'], data['y'], data['z'])).T
        rgb_uint32 = data['rgb'].astype(np.float32).view(np.uint32)
        r = (rgb_uint32 >> 16) & 255
        g = (rgb_uint32 >> 8) & 255
        b = rgb_uint32 & 255
        colors = np.stack((r, g, b), axis=-1).astype(np.uint8)

        # Log directly to the global world frame (since C++ node already securely locked it to 'map')
        # Using a unique entity name per second to persist the trail over the timeline automatically!
        rr.log(f"world/dense_map/chunk_{msg.header.stamp.sec}", rr.Points3D(
            positions=pts,
            colors=colors,
            radii=0.005
        ))

def main(args=None):
    rclpy.init(args=args)
    node = RerunBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
