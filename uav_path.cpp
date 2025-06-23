#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <cmath>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <limits>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <chrono> 
#include <omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/centroid.h>
#include <Eigen/Core>
#include <locale.h>

class UavPathGenerator {
public:
    UavPathGenerator() 
    : ugv_received_(false), uav_received_(false), sphere_radius_(2.5), safe_radius_(1.0),
        height_offset_(2.0), marker_id_(0), num_points_penalty_weight_(0.5), process_timer_enabled_(false)
    {
        ros::NodeHandle nh;
        ros::NodeHandle private_nh("~");
        
        // Read configuration from the parameter server
        private_nh.param<double>("sphere_radius", sphere_radius_, 2.5);
        private_nh.param<double>("safe_radius", safe_radius_, 0.8);
        private_nh.param<double>("height_offset", height_offset_, 3.5);
        private_nh.param<double>("num_points_penalty", num_points_penalty_weight_, 0.5);
        private_nh.param<bool>("use_timer", process_timer_enabled_, false);
        double timer_freq;
        private_nh.param<double>("timer_frequency", timer_freq, 10.0); // Default 10Hz
        
        // Subscribe to UGV path topic
        ugv_sub_ = nh.subscribe("/path_imp", 10, &UavPathGenerator::ugvCallback, this);
        // Subscribe to UAV initial path topic
        uav_sub_ = nh.subscribe("/path_ugv_simplist", 10, &UavPathGenerator::uavCallback, this);
        // Subscribe to point cloud topic for obstacle detection
        cloud_sub_ = nh.subscribe("/points_raw", 10, &UavPathGenerator::cloudCallback, this);
        // Publish the safely generated UAV path
        safe_path_pub_ = nh.advertise<nav_msgs::Path>("/path_UAV", 10);
        // New: Publish point cloud within the cylinder
        cylinder_points_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/cylinder_points", 10);
        
        // If timer mode is enabled
        if (process_timer_enabled_) {
            process_timer_ = nh.createTimer(ros::Duration(1.0/timer_freq), &UavPathGenerator::timerCallback, this);
            ROS_INFO("Timer mode enabled, frequency: %.1f Hz", timer_freq);
        }
        
        // Initialize point cloud and KD-tree
        latest_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
        kdtree_.reset(new pcl::KdTreeFLANN<pcl::PointXYZ>);
        
        // Preallocate container space
        cylinder_segments_.reserve(100);
        
        ROS_INFO("UAV path planning and obstacle avoidance module initialized");
    }

private:
    // ROS communication
    ros::Subscriber ugv_sub_;
    ros::Subscriber uav_sub_;
    ros::Subscriber cloud_sub_;
    ros::Publisher safe_path_pub_;
    ros::Publisher cylinder_points_pub_; // New: Cylinder point cloud publisher
    ros::Timer process_timer_;           // New: Timer

    // Path and state data
    nav_msgs::Path ugv_path_;
    nav_msgs::Path uav_path_;
    bool ugv_received_;
    bool uav_received_;
    bool process_timer_enabled_; // Whether to use timer mode
    
    // Configuration parameters
    double sphere_radius_;  // Communication sphere radius
    double safe_radius_;    // UAV safety distance (cylinder radius)
    double height_offset_;  // Height offset
    double num_points_penalty_weight_; // New: Additional points penalty weight
    int marker_id_;         // Marker ID counter
    
    // Point cloud processing
    pcl::PointCloud<pcl::PointXYZ>::Ptr latest_cloud_;
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_;
    
    // Visualization cache
    std::vector<std::pair<geometry_msgs::Point, geometry_msgs::Point>> cylinder_segments_;

    // Receive UGV path
    void ugvCallback(const nav_msgs::Path::ConstPtr &msg) {
        ugv_path_ = *msg;
        ugv_received_ = true;
        
        if (!process_timer_enabled_) {
            processPaths();
        }
    }

    // Receive UAV initial path
    void uavCallback(const nav_msgs::Path::ConstPtr &msg) {
        uav_path_ = *msg;
        uav_received_ = true;
        
        if (!process_timer_enabled_) {
            processPaths();
        }
    }
    
    // Timer callback function
    void timerCallback(const ros::TimerEvent& event) {
        processPaths();
    }

    // Receive point cloud data for obstacle detection, optimized version
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg) {
        // Start performance measurement
        auto start = std::chrono::high_resolution_clock::now();
        
        // Directly convert point cloud without filtering
        pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *raw_cloud);
        
        // Directly use raw point cloud
        latest_cloud_ = raw_cloud;
        
        // Update KD-tree index
        if (!latest_cloud_->empty()) {
            kdtree_->setInputCloud(latest_cloud_);
            ROS_DEBUG("Updated point cloud data, containing %zu points", latest_cloud_->size());
        } else {
            ROS_WARN("Received point cloud is empty, cannot be used for obstacle detection");
        }
        
        // End performance measurement
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        ROS_DEBUG("Point cloud processing time: %ld ms, points: %lu", duration, latest_cloud_->size());
    }

    // Add interpolation function to find the corresponding point in the simplist path based on the X coordinate
    geometry_msgs::Point interpolatePointAtX(const nav_msgs::Path& path, double target_x) {
        geometry_msgs::Point result;
        
        // If the path is empty, return the origin
        if (path.poses.empty()) {
            result.x = target_x;
            result.y = 0.0;
            result.z = 0.0;
            return result;
        }
        
        // If X is less than the starting point of the path, use the first point
        if (target_x <= path.poses.front().pose.position.x) {
            return path.poses.front().pose.position;
        }
        
        // If X is greater than the endpoint of the path, use the last point
        if (target_x >= path.poses.back().pose.position.x) {
            return path.poses.back().pose.position;
        }
        
        // Find the nearest two points in the path and interpolate
        for (size_t i = 0; i < path.poses.size() - 1; ++i) {
            double x1 = path.poses[i].pose.position.x;
            double x2 = path.poses[i + 1].pose.position.x;
            
            if (x1 <= target_x && target_x <= x2) {
                // Found suitable point pair, perform linear interpolation
                double ratio = 0.0;
                if (std::abs(x2 - x1) > 1e-6) {
                    ratio = (target_x - x1) / (x2 - x1);
                }
                
                result.x = target_x;
                result.y = path.poses[i].pose.position.y + ratio * 
                        (path.poses[i + 1].pose.position.y - path.poses[i].pose.position.y);
                result.z = path.poses[i].pose.position.z + ratio * 
                        (path.poses[i + 1].pose.position.z - path.poses[i].pose.position.z);
                
                return result;
            }
        }
        
        // If no suitable interval is found (should not happen), return the nearest point
        double min_dist = std::numeric_limits<double>::max();
        int nearest_idx = 0;
        
        for (size_t i = 0; i < path.poses.size(); ++i) {
            double dist = std::abs(path.poses[i].pose.position.x - target_x);
            if (dist < min_dist) {
                min_dist = dist;
                nearest_idx = i;
            }
        }
        
        return path.poses[nearest_idx].pose.position;
    }

    // Optimized point cloud nearest point search using KD-tree
    inline double computeMinDistanceToCloud(const geometry_msgs::Point &cand) {
        if (!latest_cloud_ || latest_cloud_->empty()) return 9999.0;
        
        // Convert to PCL point type
        pcl::PointXYZ searchPoint;
        searchPoint.x = cand.x;
        searchPoint.y = cand.y;
        searchPoint.z = cand.z;
        
        // Search for the nearest point - modified to use vector instead of std::array
        std::vector<int> pointIdxNKNSearch(1); 
        std::vector<float> pointNKNSquaredDistance(1);
        
        if (kdtree_->nearestKSearch(searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
            return std::sqrt(pointNKNSquaredDistance[0]);
        }
        
        return 9999.0;
    }

    // Template-based point-to-line segment distance calculation, supports both PCL and geometry points
    template<typename PointT>
    inline double pointToLineDistance(
        const PointT& point,
        const geometry_msgs::Point& lineStart,
        const geometry_msgs::Point& lineEnd) {
        
        // Calculate line segment vector
        double vx = lineEnd.x - lineStart.x;
        double vy = lineEnd.y - lineStart.y;
        double vz = lineEnd.z - lineStart.z;
        double lineLenSq = vx*vx + vy*vy + vz*vz;
        
        // If line segment length is almost zero, degrade to point-to-point distance
        if (lineLenSq < 1e-6) {
            double dx = point.x - lineStart.x;
            double dy = point.y - lineStart.y;
            double dz = point.z - lineStart.z;
            return std::sqrt(dx*dx + dy*dy + dz*dz);
        }
        
        // Calculate projection ratio t (fast dot product)
        double t = ((point.x - lineStart.x) * vx + 
                   (point.y - lineStart.y) * vy + 
                   (point.z - lineStart.z) * vz) / lineLenSq;
        
        // Clamp t to [0,1]
        t = std::max(0.0, std::min(1.0, t));
        
        // Calculate projection point coordinates
        double projX = lineStart.x + t * vx;
        double projY = lineStart.y + t * vy;
        double projZ = lineStart.z + t * vz;
        
        // Calculate distance from point to projection point
        double dx = point.x - projX;
        double dy = point.y - projY;
        double dz = point.z - projZ;
        
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    // Optimized point-to-line segment distance calculation
    inline double distancePointToSegment(const geometry_msgs::Point &p,
                                        const geometry_msgs::Point &start,
                                        const geometry_msgs::Point &end) {
        return pointToLineDistance(p, start, end);
    }

    // Optimized point-to-line segment distance calculation (PCL point type version)
    inline double pointToLineSegmentDistance(
        const pcl::PointXYZ& point,
        const geometry_msgs::Point& lineStart,
        const geometry_msgs::Point& lineEnd) {
        return pointToLineDistance(point, lineStart, lineEnd);
    }

    // Optimized point-to-point distance calculation
    inline double distancePointToPoint(const geometry_msgs::Point &p1, 
                                     const geometry_msgs::Point &p2) {
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        double dz = p1.z - p2.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    // Modify hasObstacleInCylinder function to return obstacle point cloud instead of directly publishing
    bool hasObstacleInCylinder(const geometry_msgs::Point &start_pt, 
                            const geometry_msgs::Point &end_pt,
                            double safe_radius,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr &obstacle_points) {
        if (!latest_cloud_ || latest_cloud_->empty()) return false;

        // Fast bounding box check
        double dx = end_pt.x - start_pt.x;
        double dy = end_pt.y - start_pt.y;
        double dz = end_pt.z - start_pt.z;
        double seg_len_sq = dx*dx + dy*dy + dz*dz;
        
        if (seg_len_sq < 1e-6) return false;
        
        // Calculate center point and search radius
        pcl::PointXYZ centerPoint;
        centerPoint.x = (start_pt.x + end_pt.x) * 0.5;
        centerPoint.y = (start_pt.y + end_pt.y) * 0.5;
        centerPoint.z = (start_pt.z + end_pt.z) * 0.5;
        
        double search_radius = std::sqrt(seg_len_sq) * 0.5 + safe_radius * 3;
        
        // Use member variables instead of local variables to avoid repeated memory allocation
        static std::vector<int> pointIdxRadiusSearch;
        static std::vector<float> pointRadiusSquaredDistance;
        pointIdxRadiusSearch.clear();
        pointRadiusSquaredDistance.clear();
        
        // Radius search
        int found_points = kdtree_->radiusSearch(centerPoint, search_radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
        if (found_points == 0) return false;
        
        // Fast boundary check
        double minX = std::min(start_pt.x, end_pt.x) - safe_radius;
        double minY = std::min(start_pt.y, end_pt.y) - safe_radius;
        double minZ = std::min(start_pt.z, end_pt.z) - safe_radius;
        double maxX = std::max(start_pt.x, end_pt.x) + safe_radius;
        double maxY = std::max(start_pt.y, end_pt.y) + safe_radius;
        double maxZ = std::max(start_pt.z, end_pt.z) + safe_radius;
        
        // To speed up, calculate distance only when necessary
        const double safe_radius_sq = safe_radius * safe_radius;
        bool has_obstacle = false;
        
        for (int i = 0; i < found_points; ++i) {
            const auto& pt = latest_cloud_->points[pointIdxRadiusSearch[i]];
            
            if (pt.x < minX || pt.x > maxX || pt.y < minY || pt.y > maxY || pt.z < minZ || pt.z > maxZ) {
                continue;
            }
            
            // First calculate a rough squared distance from the point to the line segment
            double proj_t = ((pt.x - start_pt.x) * dx + (pt.y - start_pt.y) * dy + (pt.z - start_pt.z) * dz) / seg_len_sq;
            proj_t = std::max(0.0, std::min(1.0, proj_t));
            
            double projX = start_pt.x + proj_t * dx;
            double projY = start_pt.y + proj_t * dy;
            double projZ = start_pt.z + proj_t * dz;
            
            double dist_sq = (pt.x - projX) * (pt.x - projX) + 
                            (pt.y - projY) * (pt.y - projY) + 
                            (pt.z - projZ) * (pt.z - projZ);
            
            if (dist_sq < safe_radius_sq) {
                // Add the point to the obstacle point cloud
                obstacle_points->points.push_back(pt);
                has_obstacle = true;
            }
        }
        
        if (has_obstacle) {
            ROS_INFO("Found points within the cylinder, total %zu points", obstacle_points->points.size());
        }
        
        return has_obstacle;
    }

    // The rest of the code remains unchanged
};
