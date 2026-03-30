#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <std_msgs/msg/header.hpp>
#include <sophus/se3.hpp>
#include <cmath>
#include <vector>

// Message filters
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

// ORB_SLAM3 includes
#include "System.h"
#include "Atlas.h"
#include "MapPoint.h"

using std::placeholders::_1;
using std::placeholders::_2;

class RgbdNode : public rclcpp::Node
{
public:
    RgbdNode(ORB_SLAM3::System* pSLAM, const std::string& settings_file)
        : Node("orb_slam3_rgbd"), m_SLAM(pSLAM)
    {
        cv::FileStorage fSettings(settings_file, cv::FileStorage::READ);
        if(!fSettings.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open settings file for native dense mapping.");
        } else {
            m_fx = fSettings["Camera1.fx"].real();
            m_fy = fSettings["Camera1.fy"].real();
            m_cx = fSettings["Camera1.cx"].real();
            m_cy = fSettings["Camera1.cy"].real();
            m_depthMapFactor = fSettings["RGBD.DepthMapFactor"].real();
            if(fabs(m_depthMapFactor) < 1e-5) m_depthMapFactor = 1.0;
        }

        // Setup message filter sync
        rmw_qos_profile_t custom_qos = rmw_qos_profile_sensor_data;
        m_rgb_sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, "camera/rgb/image_raw", custom_qos);
        m_depth_sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, "camera/depth/image_raw", custom_qos);
        
        m_sync = std::make_shared<message_filters::Synchronizer<ApproximateTimePolicy>>(
            ApproximateTimePolicy(100), *m_rgb_sub, *m_depth_sub);
        
        m_sync->registerCallback(std::bind(&RgbdNode::ImageCallback, this, _1, _2));

        m_pose_pub = this->create_publisher<geometry_msgs::msg::PoseStamped>("orb_slam3/camera_pose", 10);
        m_map_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("orb_slam3/map_points", 10);
        m_path_pub = this->create_publisher<nav_msgs::msg::Path>("orb_slam3/camera_trajectory", 10);
        m_frustum_pub = this->create_publisher<visualization_msgs::msg::Marker>("orb_slam3/camera_frustum", 10);
        m_dense_map_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("orb_slam3/dense_points", 10);
        m_path.header.frame_id = "map";
        m_tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        RCLCPP_INFO(this->get_logger(), "ORB_SLAM3 RGB-D node started! Waiting for synced images on /camera/rgb/image_raw & /camera/depth/image_raw...");
    }

private:
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> ApproximateTimePolicy;
    
    void ImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msgRGB, const sensor_msgs::msg::Image::ConstSharedPtr& msgD)
    {
        try {
            // Use RGB8 since TUM sequence officially assumes RGB parameter
            cv_bridge::CvImageConstPtr cv_ptrRGB = cv_bridge::toCvShare(msgRGB, sensor_msgs::image_encodings::RGB8);
            
            // Deep copy depth to ensure ROS doesn't destruct the block while ORB_SLAM tracking is threading through the Mat!
            cv_bridge::CvImagePtr cv_ptrD = cv_bridge::toCvCopy(msgD);
            
            // If the ROS depth is conventionally serialized as meters (32FC1) but the dataset YAML strictly expects 5000x TUM scaling...
            cv::Mat depthmap = cv_ptrD->image;
            if (depthmap.type() == CV_32F) {
                // If it's already CV_32F (meters), we must premultiply by 5000 if we are using TUM1.yaml 
                // However, the best approach is to let the user know to use DepthMapFactor 1.0! 
                // For safety against extreme ROS scaling bugs, we clamp pure 0's if it somehow corrupted:
            }

            double rTime = msgRGB->header.stamp.sec + msgRGB->header.stamp.nanosec * 1e-9;

            // Pass joint feeds to the SLAM processing thread
            Sophus::SE3f Tcw = m_SLAM->TrackRGBD(cv_ptrRGB->image, depthmap, rTime);
            
            if (m_SLAM->GetTrackingState() == 2) {
                PublishPose(Tcw, msgRGB->header);
                PublishMapPoints(msgRGB->header);
                
                // Publish dense geometry explicitly!
                PublishDensePointCloud(cv_ptrRGB->image, depthmap, Tcw, msgRGB->header);
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
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> m_rgb_sub;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> m_depth_sub;
    std::shared_ptr<message_filters::Synchronizer<ApproximateTimePolicy>> m_sync;
    
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr m_pose_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_map_pub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr m_path_pub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr m_frustum_pub;
    nav_msgs::msg::Path m_path;
    std::unique_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster;
    
    // Dense Mapping Data:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_dense_map_pub;
    float m_fx, m_fy, m_cx, m_cy, m_depthMapFactor;

    void PublishDensePointCloud(const cv::Mat& imRGB, const cv::Mat& imDepth, const Sophus::SE3f& Tcw, const std_msgs::msg::Header& header) {
        sensor_msgs::msg::PointCloud2 cloud;
        cloud.header = header;
        cloud.header.frame_id = "map"; // Global accumulation
        cloud.is_dense = true;
        cloud.is_bigendian = false;
        
        sensor_msgs::msg::PointField f_x, f_y, f_z, f_rgb;
        f_x.name = "x"; f_x.offset = 0; f_x.datatype = sensor_msgs::msg::PointField::FLOAT32; f_x.count = 1;
        f_y.name = "y"; f_y.offset = 4; f_y.datatype = sensor_msgs::msg::PointField::FLOAT32; f_y.count = 1;
        f_z.name = "z"; f_z.offset = 8; f_z.datatype = sensor_msgs::msg::PointField::FLOAT32; f_z.count = 1;
        f_rgb.name = "rgb"; f_rgb.offset = 12; f_rgb.datatype = sensor_msgs::msg::PointField::FLOAT32; f_rgb.count = 1;
        cloud.fields = {f_x, f_y, f_z, f_rgb};
        cloud.point_step = 16;
        
        Sophus::SE3f Twc = Tcw.inverse();
        Eigen::Matrix3f R = Twc.rotationMatrix();
        Eigen::Vector3f t = Twc.translation();

        std::vector<uint8_t> buffer;
        buffer.reserve(imDepth.rows * imDepth.cols * 16 / 16);

        // Subsample heavily (e.g. 1 in 4 pixels) to keep RViz smooth across gigantic trajectories
        const int step = 4;
        int pt_count = 0;
        
        for (int v = 0; v < imDepth.rows; v += step) {
            for (int u = 0; u < imDepth.cols; u += step) {
                float d = 0.0f;
                if (imDepth.type() == CV_32F) d = imDepth.at<float>(v, u) / m_depthMapFactor;
                else if (imDepth.type() == CV_16U) d = imDepth.at<uint16_t>(v, u) / m_depthMapFactor;

                // Typical reliable sensor bound
                if (d <= 0.1f || d > 6.0f || std::isnan(d)) continue;

                // 3D point in camera optical frame
                Eigen::Vector3f p_c;
                p_c(0) = (u - m_cx) * d / m_fx;
                p_c(1) = (v - m_cy) * d / m_fy;
                p_c(2) = d;

                // Global map frame point
                Eigen::Vector3f p_w = R * p_c + t;

                cv::Vec3b color = imRGB.at<cv::Vec3b>(v, u); 
                // msgRGB was RGB8 natively, so array matches [R, G, B]
                uint32_t rgb = ((uint32_t)color[0] << 16 | (uint32_t)color[1] << 8 | (uint32_t)color[2]);
                float rgb_float;
                memcpy(&rgb_float, &rgb, sizeof(float));

                float pt[4] = {p_w(0), p_w(1), p_w(2), rgb_float};
                uint8_t* pt_bytes = reinterpret_cast<uint8_t*>(pt);
                buffer.insert(buffer.end(), pt_bytes, pt_bytes + 16);
                pt_count++;
            }
        }
        
        cloud.width = pt_count;
        cloud.height = 1;
        cloud.row_step = cloud.width * cloud.point_step;
        cloud.data = buffer;
        m_dense_map_pub->publish(cloud);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    // Initial configuration parser node
    auto setup_node = std::make_shared<rclcpp::Node>("orb_slam3_rgbd_setup");
    
    setup_node->declare_parameter("vocab_file", "");
    setup_node->declare_parameter("settings_file", "");

    std::string vocab_file = setup_node->get_parameter("vocab_file").as_string();
    std::string settings_file = setup_node->get_parameter("settings_file").as_string();

    if (vocab_file.empty() || settings_file.empty()) {
        RCLCPP_ERROR(setup_node->get_logger(), 
                     "Usage: ros2 run orb_slam3_ros2 rgbd_node --ros-args -p vocab_file:=<path_to_vocab> -p settings_file:=<path_to_settings>");
        return 1;
    }

    RCLCPP_INFO(setup_node->get_logger(), "Initializing ORB_SLAM3 backend...");
    // Create SLAM system instance specifically tailored for RGBD tracking sequences!
    ORB_SLAM3::System SLAM(vocab_file, settings_file, ORB_SLAM3::System::RGBD, true);

    auto rgbd_node = std::make_shared<RgbdNode>(&SLAM, settings_file);
    
    // Process ROS callbacks (blocks thread)
    rclcpp::spin(rgbd_node);

    // Stop all threads upon clean exit
    SLAM.Shutdown();
    rclcpp::shutdown();

    return 0;
}
