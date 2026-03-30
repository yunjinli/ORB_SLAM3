#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <std_msgs/msg/header.hpp>
#include <sophus/se3.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>

// ORB_SLAM3 includes
#include "System.h"
#include "Atlas.h"
#include "MapPoint.h"

using std::placeholders::_1;

class MonoNode : public rclcpp::Node
{
public:
    MonoNode(ORB_SLAM3::System* pSLAM)
        : Node("orb_slam3_mono"), m_SLAM(pSLAM)
    {
        // Subscribe to standard camera topic
        m_image_sub = this->create_subscription<sensor_msgs::msg::Image>(
            "camera/image_raw", 10,
            std::bind(&MonoNode::ImageCallback, this, std::placeholders::_1));

        m_pose_pub = this->create_publisher<geometry_msgs::msg::PoseStamped>("orb_slam3/camera_pose", 10);
        m_map_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("orb_slam3/map_points", 10);
        m_path_pub = this->create_publisher<nav_msgs::msg::Path>("orb_slam3/camera_trajectory", 10);
        m_frustum_pub = this->create_publisher<visualization_msgs::msg::Marker>("orb_slam3/camera_frustum", 10);
        m_path.header.frame_id = "map";
        m_tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        RCLCPP_INFO(this->get_logger(), "ORB_SLAM3 Monocular node started! Waiting for images on /camera/image_raw ...");
    }

private:
    void ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
            
            // Extract exact timestamp from ROS message header (in seconds)
            double rTime = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

            // Pass the image block to the SLAM processing thread
            Sophus::SE3f Tcw = m_SLAM->TrackMonocular(cv_ptr->image, rTime);
            
            // Only publish if tracking is OK (state == 2)
            if (m_SLAM->GetTrackingState() == 2) {
                PublishPose(Tcw, msg->header);
                PublishMapPoints(msg->header);
            }
            
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    void PublishPose(const Sophus::SE3f& Tcw, const std_msgs::msg::Header& header) {
        Sophus::SE3f Twc = Tcw.inverse();
        Eigen::Vector3f translation = Twc.translation();
        Eigen::Quaternionf rotation = Twc.unit_quaternion();

        // 1. Publish PoseStamped
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header = header;
        pose_msg.header.frame_id = "map";
        pose_msg.pose.position.x = translation.x();
        pose_msg.pose.position.y = translation.y();
        pose_msg.pose.position.z = translation.z();
        pose_msg.pose.orientation.x = rotation.x();
        pose_msg.pose.orientation.y = rotation.y();
        pose_msg.pose.orientation.z = rotation.z();
        pose_msg.pose.orientation.w = rotation.w();
        m_pose_pub->publish(pose_msg);

        // 2. Publish TF
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header = pose_msg.header;
        tf_msg.header.frame_id = "map";
        tf_msg.child_frame_id = "camera";
        tf_msg.transform.translation.x = translation.x();
        tf_msg.transform.translation.y = translation.y();
        tf_msg.transform.translation.z = translation.z();
        tf_msg.transform.rotation = pose_msg.pose.orientation;
        m_tf_broadcaster->sendTransform(tf_msg);

        // 3. Publish Path
        m_path.header.stamp = pose_msg.header.stamp;
        m_path.poses.push_back(pose_msg);
        m_path_pub->publish(m_path);

        // 4. Publish Frustum
        visualization_msgs::msg::Marker frustum;
        frustum.header = pose_msg.header;
        frustum.header.frame_id = "map"; // Bypasses any asynchronous TF delivery latency!
        frustum.ns = "camera_frustum";
        frustum.id = 0;
        frustum.type = visualization_msgs::msg::Marker::LINE_LIST;
        frustum.action = visualization_msgs::msg::Marker::ADD;
        
        // Native Map pose rendering instantly rotates the points without relying on tf2 lookups:
        frustum.pose = pose_msg.pose;
        frustum.scale.x = 0.02; // Medium line width
        frustum.color.r = 0.0f; frustum.color.g = 1.0f; frustum.color.b = 0.0f; frustum.color.a = 1.0f;

        geometry_msgs::msg::Point o, tl, tr, br, bl;
        o.x = 0; o.y = 0; o.z = 0;
        
        // Scaled up nicely by 2x for visibility without cluttering smaller datasets
        float w = 0.3f, h = 0.2f, z = 0.4f; 
        tl.x = -w; tl.y = -h; tl.z = z;
        tr.x = w; tr.y = -h; tr.z = z;
        br.x = w; br.y = h; br.z = z;
        bl.x = -w; bl.y = h; bl.z = z;

        frustum.points = {o, tl, o, tr, o, br, o, bl, tl, tr, tr, br, br, bl, bl, tl};
        m_frustum_pub->publish(frustum);
    }

    void PublishMapPoints(const std_msgs::msg::Header& header) {
        std::vector<ORB_SLAM3::MapPoint*> mps = m_SLAM->GetTrackedMapPoints();
        if (mps.empty()) return;

        sensor_msgs::msg::PointCloud2 cloud;
        cloud.header = header;
        cloud.header.frame_id = "map";
        cloud.height = 1;
        cloud.width = 0;
        cloud.is_dense = false;
        cloud.is_bigendian = false;
        
        sensor_msgs::msg::PointField f_x, f_y, f_z;
        f_x.name = "x"; f_x.offset = 0; f_x.datatype = sensor_msgs::msg::PointField::FLOAT32; f_x.count = 1;
        f_y.name = "y"; f_y.offset = 4; f_y.datatype = sensor_msgs::msg::PointField::FLOAT32; f_y.count = 1;
        f_z.name = "z"; f_z.offset = 8; f_z.datatype = sensor_msgs::msg::PointField::FLOAT32; f_z.count = 1;
        cloud.fields = {f_x, f_y, f_z};
        cloud.point_step = 12;
        cloud.row_step = 0;

        cloud.data.reserve(mps.size() * 12);
        
        for (auto mp : mps) {
            if (mp && !mp->isBad()) {
                Eigen::Vector3f pos = mp->GetWorldPos();
                const uint8_t* pos_ptr = reinterpret_cast<const uint8_t*>(pos.data());
                cloud.data.insert(cloud.data.end(), pos_ptr, pos_ptr + 12);
                cloud.width++;
            }
        }
        cloud.row_step = cloud.width * cloud.point_step;
        m_map_pub->publish(cloud);
    }

    ORB_SLAM3::System* m_SLAM;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_image_sub;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr m_pose_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_map_pub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr m_path_pub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr m_frustum_pub;
    nav_msgs::msg::Path m_path;
    std::unique_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    // Initial configuration parser node
    auto setup_node = std::make_shared<rclcpp::Node>("orb_slam3_mono_setup");
    
    setup_node->declare_parameter("vocab_file", "");
    setup_node->declare_parameter("settings_file", "");

    std::string vocab_file = setup_node->get_parameter("vocab_file").as_string();
    std::string settings_file = setup_node->get_parameter("settings_file").as_string();

    if (vocab_file.empty() || settings_file.empty()) {
        RCLCPP_ERROR(setup_node->get_logger(), 
                     "Usage: ros2 run orb_slam3_ros2 mono_node --ros-args -p vocab_file:=<path_to_vocab> -p settings_file:=<path_to_settings>");
        return 1;
    }

    RCLCPP_INFO(setup_node->get_logger(), "Initializing ORB_SLAM3 backend...");
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(vocab_file, settings_file, ORB_SLAM3::System::MONOCULAR, true);

    auto mono_node = std::make_shared<MonoNode>(&SLAM);
    
    // Process ROS callbacks (blocks thread)
    rclcpp::spin(mono_node);

    // Stop all threads upon clean exit
    SLAM.Shutdown();
    rclcpp::shutdown();

    return 0;
}
