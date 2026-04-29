#!/usr/bin/env python3
"""
Subscribe to ORB-SLAM3 ROS2 topics and save per-frame data to disk.

Topics consumed:
  orb_slam3/camera_pose           geometry_msgs/PoseStamped   — camera Twc pose
  orb_slam3/keypoint_associations sensor_msgs/PointCloud2     — (u,v,x,y,z) per keypoint
  camera/rgb/image_raw            sensor_msgs/Image           — raw RGB
  camera/depth/image_raw          sensor_msgs/Image           — raw depth

camera_pose and keypoint_associations are ExactTime-synced (they share the same
msgRGB->header.stamp in the C++ node). RGB and depth are subscribed independently
with a rolling cache; the closest frame within --max-image-dt seconds is selected.

Output layout:
  <output_dir>/
  ├── rgb/        000000.png  ...   BGR, lossless
  ├── depth/      000000.npy  ...   raw values (float32 m or uint16 mm)
  ├── poses/      000000.npy  ...   4×4 float64 Twc (world←camera)
  ├── kp2d/       000000.npy  ...   (N,2) float32  pixel (u,v) undistorted
  ├── kp3d/       000000.npy  ...   (N,3) float32  world (x,y,z)
  └── timestamps.txt               "frame_idx  timestamp_sec" per saved frame

Usage:
  python3 slam_data_saver.py --output-dir ./output/
  python3 slam_data_saver.py --output-dir ./output/ --max-image-dt 0.05
"""

import argparse
import os
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, PointCloud2
import message_filters


def _stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=np.float64,
    )


def _parse_assoc(msg: PointCloud2) -> np.ndarray:
    """Decode keypoint_associations: u v x y z, 20 bytes/point → (N,5) float32."""
    if msg.width == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(msg.width, 5)


def _find_closest(cache: deque, stamp: float, max_dt: float):
    """Return the cached message whose stamp is closest to `stamp`, or None."""
    if not cache:
        return None
    best_t, best_msg = min(cache, key=lambda item: abs(item[0] - stamp))
    return best_msg if abs(best_t - stamp) <= max_dt else None


class SlamDataSaver(Node):
    def __init__(self, output_dir: str, max_image_dt: float,
                 settings_file: str, map_save_every: int):
        super().__init__("slam_data_saver")
        self._bridge = CvBridge()
        self._frame_idx = 0
        self._output_dir = output_dir
        self._max_image_dt = max_image_dt
        self._map_save_every = map_save_every

        # Rolling caches for image topics (keyed by timestamp).
        # Camera drivers typically publish BEST_EFFORT — use sensor_data QoS.
        self._rgb_cache: deque = deque(maxlen=30)
        self._depth_cache: deque = deque(maxlen=30)

        for subdir in ("rgb", "depth", "poses", "kp2d", "kp3d"):
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
        self._ts_file = open(os.path.join(output_dir, "timestamps.txt"), "a")
        self._map_pts_latest: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self._map_pts_recv = 0

        # Image subscriptions: sensor_data QoS to match BEST_EFFORT camera publishers.
        self.create_subscription(
            Image, "camera/rgb/image_raw", self._rgb_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, "camera/depth/image_raw", self._depth_cb, qos_profile_sensor_data
        )

        # SLAM output subscriptions: default RELIABLE QoS (matching the C++ publishers).
        # camera_pose and keypoint_associations are published in the same C++ callback
        # with the same header stamp, so ExactTime sync is safe here.
        pose_sub  = message_filters.Subscriber(self, PoseStamped, "orb_slam3/camera_pose")
        assoc_sub = message_filters.Subscriber(self, PointCloud2, "orb_slam3/keypoint_associations")
        self._sync = message_filters.TimeSynchronizer([pose_sub, assoc_sub], queue_size=20)
        self._sync.registerCallback(self._slam_cb)

        # map_points grows every frame but neighbouring frames are nearly identical —
        # subscribe independently and overwrite a single file every --map-save-every frames.
        self.create_subscription(PointCloud2, "orb_slam3/map_points", self._map_cb, 10)

        # Diagnostics: track how many messages arrive on each topic independently.
        self._recv_counts = {"pose": 0, "assoc": 0, "map": 0, "rgb": 0, "depth": 0}
        self.create_timer(5.0, self._diag_cb)

        # Save intrinsics once from the ORB-SLAM3 settings YAML.
        if settings_file:
            self._save_intrinsics(settings_file)

        self.get_logger().info(
            f"Saving to: {output_dir}  (max_image_dt={max_image_dt}s, "
            f"map_save_every={map_save_every})"
        )

    # ------------------------------------------------------------------

    def _rgb_cb(self, msg: Image):
        self._recv_counts["rgb"] += 1
        self._rgb_cache.append((_stamp_to_sec(msg.header.stamp), msg))

    def _depth_cb(self, msg: Image):
        self._recv_counts["depth"] += 1
        self._depth_cache.append((_stamp_to_sec(msg.header.stamp), msg))

    def _map_cb(self, msg: PointCloud2):
        self._recv_counts["map"] += 1
        self._map_pts_recv += 1
        if msg.width == 0:
            return
        raw = np.frombuffer(bytes(msg.data), dtype=np.float32)
        self._map_pts_latest = raw.reshape(msg.width, 3)
        if self._map_pts_recv % self._map_save_every == 0:
            np.save(os.path.join(self._output_dir, "map_points_latest.npy"), self._map_pts_latest)

    def _slam_cb(self, pose_msg: PoseStamped, assoc_msg: PointCloud2):
        self._recv_counts["pose"] += 1
        self._recv_counts["assoc"] += 1

        slam_t = _stamp_to_sec(pose_msg.header.stamp)

        rgb_msg   = _find_closest(self._rgb_cache,   slam_t, self._max_image_dt)
        depth_msg = _find_closest(self._depth_cache, slam_t, self._max_image_dt)

        if rgb_msg is None:
            self.get_logger().warn(
                f"No RGB within {self._max_image_dt}s of t={slam_t:.3f} — "
                f"cache size={len(self._rgb_cache)}. Skipping frame."
            )
            return
        if depth_msg is None:
            self.get_logger().warn(
                f"No depth within {self._max_image_dt}s of t={slam_t:.3f} — "
                f"cache size={len(self._depth_cache)}. Skipping frame."
            )
            return

        self._save(slam_t, pose_msg, assoc_msg, rgb_msg, depth_msg)

    def _save(self, stamp, pose_msg, assoc_msg, rgb_msg, depth_msg):
        idx = self._frame_idx
        self._frame_idx += 1
        tag = f"{idx:06d}"

        # Pose — 4×4 Twc
        p, q = pose_msg.pose.position, pose_msg.pose.orientation
        Twc = np.eye(4, dtype=np.float64)
        Twc[:3, :3] = _quat_to_rot(q.x, q.y, q.z, q.w)
        Twc[:3, 3] = [p.x, p.y, p.z]
        np.save(os.path.join(self._output_dir, "poses", f"{tag}.npy"), Twc)

        # Keypoints
        pts = _parse_assoc(assoc_msg)
        np.save(os.path.join(self._output_dir, "kp2d", f"{tag}.npy"), pts[:, :2])
        np.save(os.path.join(self._output_dir, "kp3d", f"{tag}.npy"), pts[:, 2:])

        # RGB
        rgb = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        cv2.imwrite(os.path.join(self._output_dir, "rgb", f"{tag}.png"), rgb)

        # Depth — raw array preserves exact values regardless of encoding
        depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        np.save(os.path.join(self._output_dir, "depth", f"{tag}.npy"), depth)

        self._ts_file.write(f"{tag}  {stamp:.9f}\n")
        self._ts_file.flush()

        if idx % 50 == 0:
            self.get_logger().info(
                f"Saved frame {tag}  t={stamp:.3f}  kps={len(pts)}  map_pts={len(self._map_pts_latest)}"
            )

    def _diag_cb(self):
        c = self._recv_counts
        self.get_logger().info(
            f"[diag] recv — pose:{c['pose']}  assoc:{c['assoc']}  map:{c['map']}  "
            f"rgb:{c['rgb']}  depth:{c['depth']}  saved:{self._frame_idx}"
        )

    def _save_intrinsics(self, settings_file: str):
        try:
            fs = cv2.FileStorage(settings_file, cv2.FileStorage_READ)
            fx = fs.getNode("Camera1.fx").real()
            fy = fs.getNode("Camera1.fy").real()
            cx = fs.getNode("Camera1.cx").real()
            cy = fs.getNode("Camera1.cy").real()
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            np.save(os.path.join(self._output_dir, "intrinsics.npy"), K)
            self.get_logger().info(f"Saved intrinsics: fx={fx} fy={fy} cx={cx} cy={cy}")
        except Exception as e:
            self.get_logger().warn(f"Could not read intrinsics from {settings_file}: {e}")

    def destroy_node(self):
        self._ts_file.close()
        # Flush the latest map on clean shutdown.
        if len(self._map_pts_latest) > 0:
            np.save(os.path.join(self._output_dir, "map_points_latest.npy"), self._map_pts_latest)
            self.get_logger().info(
                f"Saved final map: {len(self._map_pts_latest)} points → map_points_latest.npy"
            )
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description="Save ORB-SLAM3 frame data to disk.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-image-dt", type=float, default=0.05,
                        help="Max time delta (s) to match RGB/depth to SLAM stamp (default 0.05)")
    parser.add_argument("--settings-file", default="",
                        help="Path to ORB-SLAM3 settings YAML to save intrinsics.npy")
    parser.add_argument("--map-save-every", type=int, default=30,
                        help="Overwrite map_points_latest.npy every N map messages (default 30)")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args or None)
    node = SlamDataSaver(
        output_dir=os.path.abspath(args.output_dir),
        max_image_dt=args.max_image_dt,
        settings_file=args.settings_file,
        map_save_every=args.map_save_every,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Done. Saved {node._frame_idx} frames total.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
