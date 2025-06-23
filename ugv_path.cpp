#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <std_msgs/Header.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <array>
#include <chrono>
#include <functional>
#include <locale.h>

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>

// Eigen
#include <Eigen/Dense>
#include <unsupported/Eigen/Splines>

// Visualization
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

class PipeShapeDetector
{
public:
    PipeShapeDetector() : nh_("~"), stop_left_(false), stop_right_(false), 
    max_x_last_(0.0), shrink_accumulation_(0.0), accumulation_count_(0)
    {
        sub_ = nh_.subscribe("/points_raw", 1, &PipeShapeDetector::callbackPointsRaw, this);
        sub2_ = nh_.subscribe("/imu", 1, &PipeShapeDetector::imuCallback, this);
        pub_ = nh_.advertise<nav_msgs::Path>("/path_imp", 10);
        pub_simplist = nh_.advertise<nav_msgs::Path>("/path_ugv_simplist", 10);
        
        ROS_INFO("PipeShapeDetector initialized.");
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Subscriber sub2_;
    Eigen::Vector3d vertical_direction_;
    ros::Publisher pub_;
    ros::Publisher pub_simplist;
    std::vector<std::array<double, 3>> points_global_;
    bool stop_left_;
    bool stop_right_;
    double max_x_last_;
    double shrink_accumulation_;
    double yaw;
    int accumulation_count_;
    const double THRESHOLD = 10;
    const int MIN_TRIGGER_COUNT = 45;
    double slope, intercept;
    double imu_roll = 0.0;
    double imu_pitch = 0.0;
    double imu_yaw = 0.0;
    double diameter = 0.0;
    nav_msgs::Path path_average;

    void imuCallback(const sensor_msgs::Imu::ConstPtr &msg)
    {
        // IMU data
        double ax = msg->linear_acceleration.x;
        double ay = msg->linear_acceleration.y;
        double az = msg->linear_acceleration.z;

        // Calculate roll and pitch angles
        imu_roll = atan2(ay, az);
        imu_pitch = atan2(-ax, sqrt(ay * ay + az * az));

        // Assume yaw angle is 0, since it cannot be calculated from acceleration data
        imu_yaw = 0.0;
    }
  
    // Optimize caluYaw function
    std::pair<double, double> caluYaw(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_yaw) {
        if(point_cloud_yaw->size() < 30) {
            ROS_ERROR("Point cloud size is less than 30!");
            return std::make_pair(0.0, 0.0);
        }
        pcl::PointCloud<pcl::Normal>::Ptr point_cloud_normal(boost::make_shared<pcl::PointCloud<pcl::Normal>>());
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> nor;
        nor.setInputCloud(point_cloud_yaw);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        nor.setSearchMethod(tree);
        nor.setKSearch(15);
        nor.setViewPoint(3, 30, 0);
        nor.compute(*point_cloud_normal);

        if (point_cloud_normal->empty()) {
            ROS_ERROR("Normal estimation failed!");
            return std::make_pair(0.0, 0.0);
        }

        auto compareByCurvature = [](const pcl::Normal& a, const pcl::Normal& b) {
            return a.curvature < b.curvature;
        };

        std::partial_sort(point_cloud_normal->begin(), point_cloud_normal->begin() + 30, point_cloud_normal->end(), compareByCurvature);

        Eigen::MatrixXf normals(3, 30);
        for (int i = 0; i < 30; ++i) {
            normals(0, i) = point_cloud_normal->points[i].normal_x;
            normals(1, i) = point_cloud_normal->points[i].normal_y;
            normals(2, i) = point_cloud_normal->points[i].normal_z;
        }
        Eigen::Vector3f vec = normals.rowwise().sum();
        vec.normalize();

        auto yaw_modify_cos = vec.dot(Eigen::Vector3f(1, 0, 0));
        auto yaw_modify = -M_PI_2 - (-acos(yaw_modify_cos));
        double ang = yaw_modify * 180 / M_PI;

        // Calculate pipe diameter
        Eigen::MatrixXf points(3, point_cloud_yaw->size());
        for (int i = 0; i < point_cloud_yaw->size(); ++i) {
            points(0, i) = point_cloud_yaw->points[i].x;
            points(1, i) = point_cloud_yaw->points[i].y;
            points(2, i) = point_cloud_yaw->points[i].z;
        }
        Eigen::VectorXf projections = (vec.transpose() * points).transpose();
        double y_max = projections.maxCoeff();
        double y_min = projections.minCoeff();
        double diameter = y_max - y_min;

        return std::make_pair(-yaw_modify, diameter);
    }

    void callbackPointsRaw(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        // Only convert point cloud once and pass to other functions
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);
        
        // Crop for yaw calculation
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cropped(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::CropBox<pcl::PointXYZ> crop_box;
        crop_box.setInputCloud(cloud);
        crop_box.setMin(Eigen::Vector4f(-2, -100, 2, 1));
        crop_box.setMax(Eigen::Vector4f(2, 100, 9, 1));
        crop_box.filter(*cloud_cropped);
        
        std::tie(yaw, diameter) = caluYaw(cloud_cropped);
        std::tie(slope, intercept) = fitLine(cloud);

        // Pass the converted point cloud to avoid repeated conversion
        processPointsRaw(msg, cloud);
    }

    // Optimize processPointsRaw function
    void processPointsRaw(const sensor_msgs::PointCloud2ConstPtr &msg, 
                                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
    {
        auto start = std::chrono::high_resolution_clock::now(); // Start timing
        if (msg->width == 0 || msg->height == 0)
        {
            ROS_WARN("Received empty point cloud.");
            return;
        }

        // Optimization: collect to local array first to avoid critical section
        std::vector<std::array<double, 3>> temp_points;
        temp_points.reserve(msg->width * msg->height);
        
        // Single-threaded collection of valid points
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
        
        for (size_t i = 0; i < msg->width * msg->height; ++i, ++iter_x, ++iter_y, ++iter_z)
        {
            if (!std::isnan(*iter_x) && !std::isnan(*iter_y) && !std::isnan(*iter_z))
            {
                temp_points.push_back({*iter_x, *iter_y, *iter_z});
            }
        }
        
        points_global_ = std::move(temp_points);

        if (points_global_.empty())
        {
            ROS_WARN("No valid points found in the point cloud.");
            return;
        }

        std::vector<std::pair<double, double>> points_2d;
        points_2d.reserve(points_global_.size());
        for (const auto &point : points_global_)
        {
            points_2d.emplace_back(point[0], point[1]);
        }

        double step_size = 3 / (1 + abs(yaw) * 2);
        auto [left_edge_points, right_edge_points] = processPointCloud(points_2d, step_size);

        if (left_edge_points.empty() || right_edge_points.empty())
        {
            ROS_WARN("No edge points detected.");
            return;
        }
        auto average_edge_points = calculateAveragePath(left_edge_points, right_edge_points);

        // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        // pcl::fromROSMsg(*msg, *cloud);

        path_average = generatePath(average_edge_points, msg->header);

        detectAndCorrectOutliers(path_average);

        generateCuboidsAndDetectPoints(path_average, cloud, msg->header);

        removeCollinearPoints(path_average);

        auto path_simplist = generatePath_simplist(average_edge_points, msg->header, slope, intercept);
        pub_.publish(path_average);
        pub_simplist.publish(path_simplist);
        // ros::Duration(1.0).sleep();

        auto end = std::chrono::high_resolution_clock::now(); // End timing
        std::chrono::duration<double> duration = end - start;
        ROS_INFO("Function took %.0f ms", duration.count() * 1000.0);
    }

    void detectAndCorrectOutliers(nav_msgs::Path& path) {
        if (path.poses.size() < 10) {
            return;
        }
        
        // Stage 1: Detect all outliers
        std::vector<bool> should_correct(path.poses.size(), false);
        std::vector<double> corrected_values(path.poses.size(), 0.0);
        std::vector<double> corrected_values_x(path.poses.size(), 0.0);
        
        const double prior_outlier_prob = 0.2;  // Prior probability of outlier
        const double measurement_noise_std = 0.1;  // Measurement noise standard deviation
        const double outlier_noise_std = 0.5;  // Outlier noise standard deviation
        const double MEASUREMENT_NOISE_VAR = measurement_noise_std * measurement_noise_std;
        const double OUTLIER_NOISE_VAR = outlier_noise_std * outlier_noise_std;
        const double NORM_FACTOR_MEASUREMENT = 1.0 / (measurement_noise_std * std::sqrt(2 * M_PI));
        const double NORM_FACTOR_OUTLIER = 1.0 / (outlier_noise_std * std::sqrt(2 * M_PI));
        const double PRIOR_NORMAL = 1.0 - prior_outlier_prob;
        
        // First mark all possible outliers
        for (size_t i = 2; i < path.poses.size() - 2; ++i) {
            double pred_from_prev = path.poses[i-1].pose.position.y + 
                                (path.poses[i-1].pose.position.y - path.poses[i-2].pose.position.y) * 
                                (path.poses[i].pose.position.x - path.poses[i-1].pose.position.x) / 
                                (path.poses[i-1].pose.position.x - path.poses[i-2].pose.position.x + 1e-10);
            
            double pred_from_next = path.poses[i+1].pose.position.y + 
                                (path.poses[i+1].pose.position.y - path.poses[i+2].pose.position.y) * 
                                (path.poses[i].pose.position.x - path.poses[i+1].pose.position.x) / 
                                (path.poses[i+1].pose.position.x - path.poses[i+2].pose.position.x + 1e-10);
            
            double predicted_y = (pred_from_prev + pred_from_next) / 2.0;
            double y_error = path.poses[i].pose.position.y - predicted_y;
            
            // Ignore very small errors
            if (std::abs(y_error) < 0.01) continue;
            
            // Bayesian outlier detection
            double error_sq = y_error * y_error;
            double normal_likelihood = std::exp(-0.5 * error_sq / MEASUREMENT_NOISE_VAR) * NORM_FACTOR_MEASUREMENT;
            double outlier_likelihood = std::exp(-0.5 * error_sq / OUTLIER_NOISE_VAR) * NORM_FACTOR_OUTLIER;
            
            double posterior_outlier_prob = (outlier_likelihood * prior_outlier_prob) / 
                                        (normal_likelihood * PRIOR_NORMAL + 
                                        outlier_likelihood * prior_outlier_prob);
            
            // Mark outlier
            if (posterior_outlier_prob > 0.3 && std::abs(y_error) > 0.05) {
                should_correct[i] = true;
            }
        }
        
        // Stage 2: Process each marked outlier
        int correction_count = 0;
        
        for (size_t i = 2; i < path.poses.size() - 2; ++i) {
            if (should_correct[i]) {
                // Find the nearest non-outlier points before and after
                int prev_idx = i - 1;
                int next_idx = i + 1;
                bool found_prev = false;
                bool found_next = false;
                
                // Expand search forward and backward simultaneously until a non-outlier is found
                while ((!found_prev || !found_next) && 
                       (prev_idx >= 0 || next_idx < path.poses.size())) {
                    
                    // Find non-outlier point forward
                    if (!found_prev && prev_idx >= 0) {
                        if (prev_idx < 2 || !should_correct[prev_idx]) {
                            found_prev = true;
                        } else {
                            prev_idx--;
                        }
                    }
                    
                    // Find non-outlier point backward
                    if (!found_next && next_idx < path.poses.size()) {
                        if (next_idx >= path.poses.size() - 2 || !should_correct[next_idx]) {
                            found_next = true;
                        } else {
                            next_idx++;
                        }
                    }
                }
                
                // Use the y coordinate of the reliable points found
                double prev_y = found_prev ? path.poses[prev_idx].pose.position.y : path.poses[0].pose.position.y;
                double next_y = found_next ? path.poses[next_idx].pose.position.y : path.poses[path.poses.size()-1].pose.position.y;
                
                // Use linear interpolation for correction, more accurately considering the x position
                double prev_x = path.poses[prev_idx].pose.position.x;
                double next_x = path.poses[next_idx].pose.position.x;
                double curr_x = path.poses[i].pose.position.x;
                
                // Only use linear interpolation if both reference points are valid and x coordinates are different
                if (found_prev && found_next && std::abs(next_x - prev_x) > 1e-6) {
                    // Interpolate based on x position
                    double weight = (curr_x - prev_x) / (next_x - prev_x);
                    corrected_values[i] = prev_y * (1.0 - weight) + next_y * weight;
                } else {
                    // Otherwise use simple average
                    corrected_values[i] = (prev_y + next_y) / 2.0;
                }
                
                // Keep X coordinate unchanged to avoid modifying the overall path shape
                corrected_values_x[i] = path.poses[i].pose.position.x;
                
            }
        }
        
        // Stage 3: Apply corrections and filter isolated corrections
        for (size_t i = 2; i < path.poses.size() - 2; ++i) {
            if (should_correct[i]) {
                // Check if adjacent points are also corrected to avoid unsmooth isolated corrections
                bool neighbors_corrected = (i > 0 && should_correct[i-1]) || 
                                         (i < path.poses.size()-1 && should_correct[i+1]);
                
                // Only apply correction if error is large or adjacent points are also corrected
                if (neighbors_corrected || std::abs(path.poses[i].pose.position.y - corrected_values[i]) > 0.03) {
                    // Apply correction value
                    path.poses[i].pose.position.y = corrected_values[i];
                    
                    // Keep X coordinate unchanged to avoid changing path shape
                    // path.poses[i].pose.position.x = corrected_values_x[i];
                    
                    correction_count++;
                } else {
                    should_correct[i] = false; // Cancel isolated small corrections
                }
            }
        }
        
    }

    // Optimize processPointCloud function
    std::pair<std::vector<std::array<double, 2>>, std::vector<std::array<double, 2>>>
    processPointCloud(const std::vector<std::pair<double, double>> &points_2d, double step_size, double filter_thre = 1, int max_pre = 55)
    {
        std::vector<std::array<double, 2>> left_edge_points, right_edge_points;
        std::vector<double> x_coords(points_2d.size()), y_coords(points_2d.size());
        bool stop_left_ = false, stop_right_ = false;
        

        #pragma omp parallel for
        for (size_t i = 0; i < points_2d.size(); ++i)
        {
            x_coords[i] = points_2d[i].first;
            y_coords[i] = points_2d[i].second;
        }

        double max_x = *std::max_element(x_coords.begin(), x_coords.end());

        // Check the change in max_x
        double shrink_amount = max_x_last_ - max_x;
        if (abs(shrink_amount) > 0.1)
        {
            if (shrink_amount > 0)
            {
                shrink_accumulation_ += shrink_amount;
                accumulation_count_++;
            }
            else
            {
                shrink_accumulation_ += shrink_amount;
                accumulation_count_--;
            }

            if (shrink_accumulation_ < 0)
            {
                shrink_accumulation_ = 0;
            }
        }

        max_x_last_ = max_x;

        // Evaluate the stop condition
        if (accumulation_count_ >= MIN_TRIGGER_COUNT)
        {
            double y_rate = std::exp(shrink_accumulation_ - 40); // Derivative of exp(x-5) is exp(x-5)
            std::cout << "shrink_accumulation_: " << shrink_accumulation_ << " y_rate: " << y_rate << " accumulation_count: " << accumulation_count_ << std::endl;
            if (y_rate > THRESHOLD)
            {
                ROS_INFO("Stopping path planning due to high shrink accumulation rate.");
                return {left_edge_points, right_edge_points}; // Returning empty vectors to stop planning
            }
        }

        std::vector<double> bins;
        bins.reserve(max_pre / step_size + 1);
        for (double i = 0; i <= max_pre; i += step_size)
        {
            bins.push_back(i);
        }

        #pragma omp parallel for
        for (size_t i = 1; i < bins.size(); ++i)
        {
            std::vector<std::array<double, 2>> region_points;
            for (size_t j = 0; j < x_coords.size(); ++j)
            {
                if (x_coords[j] >= bins[i - 1] && x_coords[j] < bins[i])
                {
                    region_points.push_back({x_coords[j], y_coords[j]});
                }
            }
            if (!region_points.empty())
            {
                auto min_y_it = std::min_element(region_points.begin(), region_points.end(), [](auto &a, auto &b)
                                                { return a[1] < b[1]; });
                auto max_y_it = std::max_element(region_points.begin(), region_points.end(), [](auto &a, auto &b)
                                                { return a[1] < b[1]; });
                std::array<double, 2> left_edge = *min_y_it;
                std::array<double, 2> right_edge = *max_y_it;
                if (!stop_left_)
                {
                    if (!left_edge_points.empty())
                    {
                        double left_y_diff = std::abs(left_edge[1] - left_edge_points.back()[1]);
                        if (left_y_diff > filter_thre)
                        {
                            stop_left_ = true;
                        }
                        else
                        {
                            left_edge_points.push_back(left_edge);
                        }
                    }
                    else
                    {
                        left_edge_points.push_back(left_edge);
                    }
                }
                if (!stop_right_)
                {
                    if (!right_edge_points.empty())
                    {
                        double right_y_diff = std::abs(right_edge[1] - right_edge_points.back()[1]);
                        if (right_y_diff > filter_thre)
                        {
                            stop_right_ = true;
                        }
                        else
                        {
                            right_edge_points.push_back(right_edge);
                        }
                    }
                    else
                    {
                        right_edge_points.push_back(right_edge);
                    }
                }
            }
        }

        if (!stop_left_ && !stop_right_)
        {
            ROS_INFO("Normal detection: Left edge: %zu, Right edge: %zu", 
                     left_edge_points.size(), right_edge_points.size());
            return {left_edge_points, right_edge_points};
        }
        else
        {
            // Calculate the maximum distance to fit (based on max_pre parameter)
            double max_distance = max_pre;
            // Calculate the maximum number of points
            size_t max_points = std::ceil(max_distance / step_size);
            
            // Detect if the pipe is curved
            bool is_curved_left = false;
            bool is_curved_right = false;
            
            if (stop_left_ && left_edge_points.size() >= 5) {
                // Detect if the left pipe has a curvature trend
                is_curved_left = detectCurvature(left_edge_points);
            }
            
            if (stop_right_ && right_edge_points.size() >= 5) {
                // Detect if the right pipe has a curvature trend
                is_curved_right = detectCurvature(right_edge_points);
            }
            
            // Handle left edge points
            if (stop_left_)
            {
                if (is_curved_left) {
                    // Use polynomial fitting for curved pipes
                    std::vector<double> x_fit(left_edge_points.size()), y_fit(left_edge_points.size());
                    for (size_t i = 0; i < left_edge_points.size(); ++i) {
                        x_fit[i] = left_edge_points[i][0];
                        y_fit[i] = left_edge_points[i][1];
                    }
                    
                    // Adaptively adjust polynomial degree based on number of points
                    int poly_degree = 2; 
                    
                    // Use higher order polynomial fitting
                    auto poly_coeffs = polynomialFit(x_fit, y_fit, poly_degree);
                    double start_x = left_edge_points.back()[0];
                                    
                    // Calculate how many points need to be predicted
                    size_t target_points = std::min(max_points, left_edge_points.size() + 100);
                    
                    for (size_t i = left_edge_points.size(); i < target_points; ++i) {
                        start_x += step_size;
                        double pred_y = polynomialPredict(poly_coeffs, start_x);
                        left_edge_points.push_back({start_x, pred_y});
                    }
                    
                    ROS_INFO("Curved left edge detected, extended with polynomial fit to %zu points", 
                             left_edge_points.size());
                } else {
                    // For straight pipes, continue to use linear fitting
                    std::vector<double> x_fit(left_edge_points.size()), y_fit(left_edge_points.size());
                    #pragma omp parallel for
                    for (size_t i = 0; i < left_edge_points.size(); ++i) {
                        x_fit[i] = left_edge_points[i][0];
                        y_fit[i] = left_edge_points[i][1];
                    }
                    auto coeffs = linearRegressionFit(x_fit, y_fit);
                    double start_x = left_edge_points.back()[0];
                    
                    size_t target_points = std::min(max_points, left_edge_points.size() + 200);
                    
                    for (size_t i = left_edge_points.size(); i < target_points; ++i) {
                        start_x += step_size;
                        double pred_y = linearRegressionPredict(coeffs, start_x);
                        left_edge_points.push_back({start_x, pred_y});
                    }
                    
                    ROS_INFO("Straight left edge, extended with linear fit to %zu points", 
                             left_edge_points.size());
                }
            }
            
            // Handle right edge points - similar logic as left
            if (stop_right_)
            {
                if (is_curved_right) {
                    // Use polynomial fitting for curved pipes
                    std::vector<double> x_fit(right_edge_points.size()), y_fit(right_edge_points.size());
                    for (size_t i = 0; i < right_edge_points.size(); ++i) {
                        x_fit[i] = right_edge_points[i][0];
                        y_fit[i] = right_edge_points[i][1];
                    }
                    
                    // Adaptively adjust polynomial degree based on number of points
                    int poly_degree = 2;

                    
                    // Use higher order polynomial fitting
                    auto poly_coeffs = polynomialFit(x_fit, y_fit, poly_degree);
                    double start_x = right_edge_points.back()[0];
                    
                    // Calculate how many points need to be predicted
                    size_t target_points = std::min(max_points, right_edge_points.size() + 100);
                    
                    for (size_t i = right_edge_points.size(); i < target_points; ++i) {
                        start_x += step_size;
                        double pred_y = polynomialPredict(poly_coeffs, start_x);
                        right_edge_points.push_back({start_x, pred_y});
                    }
                    
                    ROS_INFO("Curved right edge detected, extended with polynomial fit to %zu points", 
                             right_edge_points.size());
                } else {
                    // For straight pipes, continue to use linear fitting
                    std::vector<double> x_fit(right_edge_points.size()), y_fit(right_edge_points.size());
                    #pragma omp parallel for
                    for (size_t i = 0; i < right_edge_points.size(); ++i) {
                        x_fit[i] = right_edge_points[i][0];
                        y_fit[i] = right_edge_points[i][1];
                    }
                    auto coeffs = linearRegressionFit(x_fit, y_fit);
                    double start_x = right_edge_points.back()[0];
                    
                    size_t target_points = std::min(max_points, right_edge_points.size() + 200);
                    
                    for (size_t i = right_edge_points.size(); i < target_points; ++i) {
                        start_x += step_size;
                        double pred_y = linearRegressionPredict(coeffs, start_x);
                        right_edge_points.push_back({start_x, pred_y});
                    }
                    
                    ROS_INFO("Straight right edge, extended with linear fit to %zu points", 
                             right_edge_points.size());
                }
            }
            
            // Ensure the number of points on both sides is equal for subsequent average path calculation
            if (left_edge_points.size() > right_edge_points.size()) {
                // Trim excess points on the left
                left_edge_points.resize(right_edge_points.size());
            } else if (right_edge_points.size() > left_edge_points.size()) {
                // Trim excess points on the right
                right_edge_points.resize(left_edge_points.size());
            }
            
            ROS_INFO("Final path points: %zu", left_edge_points.size());
        }
        return {left_edge_points, right_edge_points};
    }

    Eigen::VectorXd linearRegressionFit(const std::vector<double> &x, const std::vector<double> &y)
    {
        int n = x.size();
        Eigen::MatrixXd X(n, 2);
        Eigen::VectorXd Y = Eigen::VectorXd::Map(y.data(), n);
        for (int i = 0; i < n; ++i)
        {
            X(i, 0) = 1;
            X(i, 1) = x[i];
        }
        Eigen::VectorXd coeffs = (X.transpose() * X).ldlt().solve(X.transpose() * Y);
        return coeffs;
    }

    double linearRegressionPredict(const Eigen::VectorXd &coeffs, double x)
    {
        return coeffs(0) + coeffs(1) * x;
    }

    
    // Detect if the path has significant curvature
    bool detectCurvature(const std::vector<std::array<double, 2>>& points) {
        if (points.size() < 5) return false;
        
        // Detect if the path has significant curvature
        std::vector<double> slopes;
        const int step = std::max(1, static_cast<int>(points.size() / 10));
        
        for (size_t i = step; i < points.size(); i += step) {
            if (i >= points.size()) break;
            
            double dx = points[i][0] - points[i-step][0];
            if (std::abs(dx) < 1e-6) continue;  // Avoid division by zero
            
            double slope = (points[i][1] - points[i-step][1]) / dx;
            slopes.push_back(slope);
        }
        
        if (slopes.size() < 3) return false;
        
        // Calculate slope variance
        double slope_variance = 0.0;
        double mean_slope = 0.0;
        
        // Calculate mean slope
        for (double s : slopes) {
            mean_slope += s;
        }
        mean_slope /= slopes.size();
        
        // Calculate variance
        for (double s : slopes) {
            slope_variance += (s - mean_slope) * (s - mean_slope);
        }
        slope_variance /= slopes.size();
        
        // If slope variance is significant, consider it a curved path
        return slope_variance > 0.001; 
    }

    // Polynomial fitting function
    Eigen::VectorXd polynomialFit(const std::vector<double>& x, const std::vector<double>& y, int degree) {
        int n = x.size();
        int terms = degree + 1;
        
        Eigen::MatrixXd X(n, terms);
        Eigen::VectorXd Y = Eigen::VectorXd::Map(y.data(), n);
        
        // Construct Vandermonde matrix
        for (int i = 0; i < n; ++i) {
            double xi = 1.0;
            for (int j = 0; j < terms; ++j) {
                X(i, j) = xi;
                xi *= x[i];
            }
        }
        
        // Use QR decomposition to solve the linear system, more stable than direct inversion
        Eigen::VectorXd coeffs = X.householderQr().solve(Y);
        return coeffs;
    }

    // Polynomial prediction function
    double polynomialPredict(const Eigen::VectorXd& coeffs, double x) {
        double result = 0.0;
        double xi = 1.0;
        
        for (int i = 0; i < coeffs.size(); ++i) {
            result += 1.2 * coeffs(i) * xi;
            xi *= x;
        }
        
        return result;
    }

    std::vector<std::pair<double, double>> calculateAveragePath(const std::vector<std::array<double, 2>> &left_edge_points, const std::vector<std::array<double, 2>> &right_edge_points)
    {
        std::vector<std::pair<double, double>> average_edge_points;
        for (size_t i = 0; i < left_edge_points.size() && i < right_edge_points.size(); ++i)
        {
            double avg_x = (left_edge_points[i][0] + right_edge_points[i][0]) / 2.0;
            double avg_y = (left_edge_points[i][1] + right_edge_points[i][1]) / 2.0;
            average_edge_points.emplace_back(avg_x, avg_y);
        }
        return average_edge_points;
    }

    std::pair<double, double> fitLine(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
    {
        // Sample the point cloud along the x-axis
        std::vector<double> sampled_x_coords, sampled_z_coords;
        double step_size = 0.5; 

        // Get the x-axis range of the point cloud
        double min_x = -5.0;
        double max_x = -3.0;

        // Create buckets
        int num_buckets = static_cast<int>((max_x - min_x) / step_size) + 1;
        std::vector<std::vector<pcl::PointXYZ>> buckets(num_buckets);

        // Assign point cloud to buckets
        for (const auto& point : cloud->points)
        {
            int bucket_index = static_cast<int>((point.x - min_x) / step_size);
            if (bucket_index >= 0 && bucket_index < num_buckets)
            {
                buckets[bucket_index].push_back(point);
            }
        }

        // Find the minimum z point in each bucket
        for (int i = 0; i < num_buckets; ++i)
        {
            if (!buckets[i].empty())
            {
                double min_z = std::numeric_limits<double>::max();
                double x_coord = min_x + i * step_size;
                for (const auto& point : buckets[i])
                {
                    if (point.z < min_z)
                    {
                        min_z = point.z;
                    }
                }
                sampled_x_coords.push_back(x_coord);
                sampled_z_coords.push_back(min_z);
            }
        }

        // Fit a straight line
        auto coeffs = linearRegressionFit(sampled_x_coords, sampled_z_coords);
        return {coeffs(1), coeffs(0)}; // slope and intercept
    }

    // Optimize generatePath function
    nav_msgs::Path generatePath(const std::vector<std::pair<double, double>> &points, const std_msgs::Header &header)
    {
        nav_msgs::Path path;
        path.header = header;
        path.header.frame_id = "map";

        if (points.size() < 3) {
            ROS_WARN("Points size is less than 3, no path generated.");
            return path;
        }

        tf2::Quaternion q_imu;
        // Shrink the amplitude of the yaw angle, controlled by a coefficient
        double yaw_shrink_factor = 0.3;
        double adjusted_yaw = yaw * yaw_shrink_factor;
        
        // Smooth transition for roll angle in the same way as yaw
        double roll_shrink_factor = 0.1;
        double adjusted_roll = imu_roll * roll_shrink_factor;
        
        int transition_points = std::max(1, std::min(static_cast<int>(points.size()), 
                                      static_cast<int>(points.size() / (1 + std::exp(-abs(adjusted_yaw))))));
        double transition_yaw_step = adjusted_yaw / transition_points; 
        double transition_roll_step = adjusted_roll / transition_points;
        
        path.poses.reserve(points.size()); 

        for (int i = 0; i < transition_points; ++i) {
            geometry_msgs::PoseStamped pose;
            pose.header = path.header;
            Eigen::Vector3d local_point(points[i].first, points[i].second, slope * points[i].first + intercept);
            // Apply gradual change for roll angle
            q_imu.setRPY(-(adjusted_roll - i * transition_roll_step), 0, adjusted_yaw - i * transition_yaw_step);
            tf2::Vector3 tf_local_point(local_point.x(), local_point.y(), local_point.z());
            tf2::Vector3 global_point = tf2::quatRotate(q_imu, tf_local_point);
            pose.pose.position.x = global_point.x();
            pose.pose.position.y = global_point.y();
            pose.pose.position.z = global_point.z();
            pose.pose.orientation.x = q_imu.x();
            pose.pose.orientation.y = q_imu.y();
            pose.pose.orientation.z = q_imu.z();
            pose.pose.orientation.w = q_imu.w();
            path.poses.push_back(pose);
        }
        
        q_imu.setRPY(0, 0, 0); // Fix yaw and roll angles to 0
        for (int i = transition_points; i < points.size(); ++i) {
            geometry_msgs::PoseStamped pose;
            pose.header = path.header;
            Eigen::Vector3d local_point(points[i].first, points[i].second, slope * points[i].first + intercept);
            tf2::Vector3 tf_local_point(local_point.x(), local_point.y(), local_point.z());
            tf2::Vector3 global_point = tf2::quatRotate(q_imu, tf_local_point);
            pose.pose.position.x = global_point.x();
            pose.pose.position.y = global_point.y();
            pose.pose.position.z = global_point.z();
            pose.pose.orientation.x = q_imu.x();
            pose.pose.orientation.y = q_imu.y();
            pose.pose.orientation.z = q_imu.z();
            pose.pose.orientation.w = q_imu.w();
            path.poses.push_back(pose);
        }

        return path;
    }

    void removeCollinearPoints(nav_msgs::Path& path) {
        // If the number of path points is less than 3, collinearity detection cannot be performed
        if (path.poses.size() < 3) {
            return;
        }
        
        // Threshold definition: maximum distance from point to line (for collinearity determination)
        const double collinear_threshold = 0.05; 
        
        // Use a marker array to record points to keep
        std::vector<bool> keep_point(path.poses.size(), true);
        keep_point[0] = true;
        keep_point[path.poses.size() - 1] = true;
        
        // Detect if three consecutive points are collinear
        for (size_t i = 1; i < path.poses.size() - 1; ++i) {
            const auto& prev = path.poses[i-1].pose.position;
            const auto& curr = path.poses[i].pose.position;
            const auto& next = path.poses[i+1].pose.position;
            
            // Calculate the vector formed by the previous and next points
            Eigen::Vector3d vec_prev_next(next.x - prev.x, next.y - prev.y, next.z - prev.z);
            double length_prev_next = vec_prev_next.norm();
            
            // Avoid division by zero or very small values
            if (length_prev_next < 1e-5) {
                continue;
            }
            
            Eigen::Vector3d unit_vec = vec_prev_next / length_prev_next;
            
            // Vector from prev to curr
            Eigen::Vector3d vec_prev_curr(curr.x - prev.x, curr.y - prev.y, curr.z - prev.z);
            
            // Calculate the projection length of curr in the direction of prev_next
            double proj_length = vec_prev_curr.dot(unit_vec);
            
            // Calculate the distance from curr to the line (perpendicular distance)
            Eigen::Vector3d proj_point = Eigen::Vector3d(prev.x, prev.y, prev.z) + unit_vec * proj_length;
            Eigen::Vector3d curr_point(curr.x, curr.y, curr.z);
            double distance = (proj_point - curr_point).norm();
            
            // If the distance is less than the threshold and the projection point is within the segment, curr is redundant
            if (distance < collinear_threshold && proj_length > 0 && proj_length < length_prev_next) {
                keep_point[i] = false;
            }
        }
        
        // Delete points marked as not to be kept
        nav_msgs::Path simplified_path;
        simplified_path.header = path.header;
        
        for (size_t i = 0; i < path.poses.size(); ++i) {
            if (keep_point[i]) {
                simplified_path.poses.push_back(path.poses[i]);
            }
        }
        
        // Update path
        if (simplified_path.poses.size() < path.poses.size()) {
            path = simplified_path;
        }
    }

    void generateCuboidsAndDetectPoints(nav_msgs::Path& path, const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, const std_msgs::Header& header) {
        // 0. Basic parameter settings
        const double PIPE_RADIUS = diameter / 2.0;  // Pipe radius
        const double MAX_TILT_ANGLE = 0.35;         // Maximum allowed tilt angle (about 20 degrees)
        const double SAFETY_MARGIN = 0.3;           // Safety margin
        const double VEHICLE_WIDTH = 1.2;           // Vehicle width
        const int NUM_FIREFLIES = 50;              // Number of fireflies
        const int MAX_ITERATIONS = 80;              // Maximum number of iterations
        
        // calculate the index boundary for the first 60% of the path
        size_t path_size = path.poses.size();
        size_t max_detection_idx = static_cast<size_t>(path_size * 0.8);
        ROS_INFO("Obstacle detection applied only to first 60%% of path (points 0-%zu of %zu total)", 
                max_detection_idx, path_size);
        
        // 1. Preliminary detection of path segments with obstacles - only detect the first 60% of the path
        std::vector<int> obstacle_indices;
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> obstacle_clouds; // Save obstacle point cloud

        for (size_t i = 1; i < std::min(path.poses.size() - 1, max_detection_idx); ++i) {
            const auto& p1 = path.poses[i].pose.position;
            const auto& p2 = path.poses[i + 1].pose.position;
            
            double dx = p2.x - p1.x;
            double dy = p2.y - p1.y;
            double length = std::sqrt(dx*dx + dy*dy);
            
            // Detection box parameters
            geometry_msgs::Point center;
            center.x = (p1.x + p2.x) / 2.0;
            center.y = (p1.y + p2.y) / 2.0;
            center.z = p1.z + 0.8; // Use original z coordinate plus an offset
            
            double yaw = std::atan2(dy, dx);
            
            // Use a larger detection range
            pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::CropBox<pcl::PointXYZ> crop_box;
            crop_box.setInputCloud(input_cloud);
            crop_box.setMin(Eigen::Vector4f(-length/2, -0.4, 0.2, 1.0)); 
            crop_box.setMax(Eigen::Vector4f(length/2, 0.4, 1.5, 1.0));  
            crop_box.setRotation(Eigen::Vector3f(0, 0, yaw));
            crop_box.setTranslation(Eigen::Vector3f(center.x, center.y, center.z));
            crop_box.filter(*filtered_cloud);
            
            if (!filtered_cloud->empty()) {
                obstacle_indices.push_back(i);
                obstacle_clouds.push_back(filtered_cloud); 
            }
        }

        // If no obstacles, return directly
        if (obstacle_indices.empty()) {
            ROS_INFO("No obstacles detected in the path.");
            return;
        }

        ROS_INFO("Detected %zu obstacles in the first 60%% of path", obstacle_indices.size());

        // 2. Apply improved firefly algorithm to each obstacle path segment (optimize only in Y direction)
        for (size_t i = 0; i < obstacle_indices.size(); ++i) {
            int idx = obstacle_indices[i];
            const auto& obstacle_cloud = obstacle_clouds[i];
            const auto& p_orig = path.poses[idx].pose.position;
            
            if (obstacle_cloud->empty()) {
                ROS_WARN("Empty obstacle cloud for path point %d", idx);
                continue;
            }
            
            // Obstacle point cloud statistical analysis
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*obstacle_cloud, centroid);
            
            // Project obstacles onto the y-axis to get their distribution in the y direction
            std::vector<float> obstacle_y_values;
            for (const auto& point : obstacle_cloud->points) {
                obstacle_y_values.push_back(point.y);
            }
            
            // Calculate the distribution range of obstacles in the y direction
            float obstacle_y_min = *std::min_element(obstacle_y_values.begin(), obstacle_y_values.end());
            float obstacle_y_max = *std::max_element(obstacle_y_values.begin(), obstacle_y_values.end());
            
            // 3. Firefly algorithm initialization - generate fireflies only in the Y direction of the pipe cross-section
            std::vector<double> fireflies_y;     
            std::vector<double> intensities(NUM_FIREFLIES);   // Firefly brightness
            
            // Calculate the safe range of the pipe in the y direction
            double pipe_y_min = -PIPE_RADIUS + VEHICLE_WIDTH/2 + SAFETY_MARGIN;
            double pipe_y_max = PIPE_RADIUS - VEHICLE_WIDTH/2 - SAFETY_MARGIN;
            
            // Generate initial firefly positions - evenly distributed in the Y direction of the pipe cross-section
            for (int i = 0; i < NUM_FIREFLIES; i++) {
                // Evenly distributed within the safe range
                double y = pipe_y_min + (pipe_y_max - pipe_y_min) * i / (NUM_FIREFLIES - 1);
                
                // Add some randomness
                y += (pipe_y_max - pipe_y_min) * 0.1 * (rand() / double(RAND_MAX) - 0.5);
                
                // Ensure within safe range
                y = std::max(pipe_y_min, std::min(pipe_y_max, y));
                
                // Calculate tilt - approximate using the ratio of y to pipe radius
                double tilt = std::abs(y) / PIPE_RADIUS;
                
                // If tilt exceeds the maximum, adjust position
                if (tilt > MAX_TILT_ANGLE) {
                    double adjusted_y = MAX_TILT_ANGLE * PIPE_RADIUS * (y >= 0 ? 1 : -1);
                    y = adjusted_y;
                }
                
                fireflies_y.push_back(y);
            }
            
            // 4. Firefly algorithm iteration
            double alpha = 0.5;  // Random movement factor, increase to improve global search ability, but path may oscillate

            double beta0 = 2.8;  // Base value of attractiveness, increase to speed up convergence, but diversity decreases

            double gamma = 1.5;  // Light absorption coefficient, increase to form multiple local search areas, suitable for complex obstacle avoidance

            
            for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
                // 4.1 Calculate the brightness (fitness) of each firefly
                for (int i = 0; i < NUM_FIREFLIES; i++) {
                    double y = fireflies_y[i];
                    
                    // Calculate the distance to the obstacle (the farther the better)
                    double min_dist = std::numeric_limits<double>::max();
                    for (float obs_y : obstacle_y_values) {
                        // Calculate the distance from the vehicle edge to the obstacle, not from the center to the obstacle
                        double actual_dist = std::abs(y - obs_y) - (VEHICLE_WIDTH/2 + SAFETY_MARGIN);
                        
                        // If the distance is negative, a collision will occur, apply a penalty
                        if (actual_dist < 0) {
                            actual_dist = actual_dist * 5;
                        }
                        
                        min_dist = std::min(min_dist, actual_dist);
                    }
                    
                    // Calculate tilt penalty (the smaller the better)）
                    double tilt_penalty = std::pow(std::abs(y) / PIPE_RADIUS, 2) * 2.0;
                    
                    // Calculate the penalty for distance to the pipe center (encourage staying near the center)
                    double center_dist_penalty = std::abs(y) * 0.5;
                    
                    // Comprehensive fitness (brightness) - farther distance, less tilt, closer to center is better
                    intensities[i] = min_dist - tilt_penalty - center_dist_penalty;
                }
                
                // 4.2 Fireflies attract and move each other
                std::vector<double> new_fireflies_y = fireflies_y;
                
                for (int i = 0; i < NUM_FIREFLIES; i++) {
                    for (int j = 0; j < NUM_FIREFLIES; j++) {
                        // Firefly i is attracted by a brighter firefly j
                        if (intensities[j] > intensities[i]) {
                            double r = std::abs(fireflies_y[i] - fireflies_y[j]);
                            
                            // Calculate attractiveness
                            double beta = beta0 * std::exp(-gamma * r * r);
                            
                            // Move towards the brighter firefly
                            new_fireflies_y[i] += beta * (fireflies_y[j] - fireflies_y[i]);
                        }
                    }
                    
                    // Add random movement
                    new_fireflies_y[i] += alpha * (rand() / double(RAND_MAX) - 0.5) * (pipe_y_max - pipe_y_min) * 0.2;
                    
                    // Ensure fireflies are inside the pipe and meet tilt constraints
                    double y = new_fireflies_y[i];
                    
                    // If out of safe range, adjust position
                    y = std::max(pipe_y_min, std::min(pipe_y_max, y));
                    
                    // Check tilt constraint
                    double tilt = std::abs(y) / PIPE_RADIUS;
                    if (tilt > MAX_TILT_ANGLE) {
                        y = MAX_TILT_ANGLE * PIPE_RADIUS * (y >= 0 ? 1 : -1);
                    }
                    
                    new_fireflies_y[i] = y;
                }
                
                // Update firefly positions
                fireflies_y = new_fireflies_y;
                
                // Reduce random factor
                alpha *= 0.95;
            }
            
            // 5. Select the best firefly (highest brightness)
            int best_idx = 0;
            for (int i = 1; i < NUM_FIREFLIES; i++) {
                if (intensities[i] > intensities[best_idx]) {
                    best_idx = i;
                }
            }
            
            // 6. Generate new path point - only modify Y coordinate, keep Z unchanged
            geometry_msgs::PoseStamped new_pose = path.poses[idx];
            new_pose.pose.position.y = fireflies_y[best_idx];
            
            // Since Y is changed, update Z according to pipe geometry
            // Assume the pipe is cylindrical, Z should change with Y to stay on the pipe surface
            // Calculate the Z coordinate of the pipe center
            double pipe_center_z = path.poses[idx].pose.position.z + (PIPE_RADIUS - sqrt(PIPE_RADIUS*PIPE_RADIUS - new_pose.pose.position.y*new_pose.pose.position.y));
            
            // 7. Update path
            path.poses[idx] = new_pose;
            
            // 8. Local path smoothing - adjust Y coordinates of surrounding points
            const int SMOOTH_RADIUS = 2;
            for (int offset = 1; offset <= SMOOTH_RADIUS; offset++) {
                // Ensure not to exceed the first 60% for smoothing
                if (idx + offset < path.poses.size() && idx + offset < max_detection_idx) {
                    double weight = (SMOOTH_RADIUS - offset + 1.0) / (SMOOTH_RADIUS + 1.0);
                    path.poses[idx + offset].pose.position.y = 
                        (1.0 - weight) * path.poses[idx + offset].pose.position.y + 
                        weight * new_pose.pose.position.y;
                }
                
                if (idx - offset >= 0) {
                    double weight = (SMOOTH_RADIUS - offset + 1.0) / (SMOOTH_RADIUS + 1.0);
                    path.poses[idx - offset].pose.position.y = 
                        (1.0 - weight) * path.poses[idx - offset].pose.position.y + 
                        weight * new_pose.pose.position.y;
                }
            }
        }
        
        // 9. Final global path smoothing to ensure overall coherence
        smoothAvoidancePath(path, max_detection_idx);
    }

    // Add max_detection_idx parameter to limit smoothing area
    void smoothAvoidancePath(nav_msgs::Path& path, size_t max_detection_idx) {
        if (path.poses.size() < 5) return;
        
        size_t transition_zone = std::min(static_cast<size_t>(path.poses.size() * 0.1), 
                                        static_cast<size_t>(10)); // 最多10个点的过渡区

        size_t total_smooth_area = std::min(max_detection_idx + transition_zone, path.poses.size());
        
        // Savitzky-Golay filter coefficients (five-point smoothing)
        const std::vector<double> sg_coeffs = {-3.0/35, 12.0/35, 12.0/35, 12.0/35, -3.0/35};
        const int half_window = sg_coeffs.size() / 2;
        
        // Create a temporary path to store the smoothing result
        nav_msgs::Path smoothed_path = path;
        
        // Apply the filter to all internal points that need smoothing, only smooth Y coordinate
        for (size_t i = half_window; i < total_smooth_area - half_window; ++i) {
            double smooth_y = 0.0;
            for (int j = -half_window; j <= half_window; ++j) {
                smooth_y += path.poses[i+j].pose.position.y * sg_coeffs[j+half_window];
            }
            smoothed_path.poses[i].pose.position.y = smooth_y;
        }
        
        // Apply weaker smoothing to the transition area to ensure smooth transition with the subsequent path
        for (size_t i = max_detection_idx; i < total_smooth_area; ++i) {
            
            double transition_weight = 1.0 - static_cast<double>(i - max_detection_idx) / transition_zone;
            
            
            smoothed_path.poses[i].pose.position.y = 
                transition_weight * smoothed_path.poses[i].pose.position.y + 
                (1.0 - transition_weight) * path.poses[i].pose.position.y;
        }
        
        
        path = smoothed_path;

        ROS_INFO("Path smoothed: first 60%% (%zu points) with obstacle avoidance, with %zu-point transition zone",
                max_detection_idx, transition_zone);
    }

    nav_msgs::Path generatePath_simplist(const std::vector<std::pair<double, double>> &points, const std_msgs::Header &header, double slope, double intercept)
    {
        nav_msgs::Path path;
        path.header = header;
        path.header.frame_id = "map";
        tf2::Quaternion q;
        q.setRPY(-imu_roll, 0, imu_yaw);  

        for (const auto &point : points) {
            geometry_msgs::PoseStamped pose;
            pose.header = header;
            tf2::Vector3 local_point(point.first, point.second,  slope * point.first + intercept); 
            tf2::Vector3 global_point = tf2::quatRotate(q, local_point);

            pose.pose.position.x = global_point.x();
            pose.pose.position.y = global_point.y();
            pose.pose.position.z = slope * global_point.x() + intercept + 5;
            pose.pose.orientation.x = q.x();
            pose.pose.orientation.y = q.y();
            pose.pose.orientation.z = q.z();
            pose.pose.orientation.w = q.w();
            path.poses.push_back(pose);
        }

        return path;
    }
};



int main(int argc, char **argv)
{
    setlocale(LC_ALL, "zh_CN.UTF-8");
    ros::init(argc, argv, "ugv_path");
    ROS_INFO("Node initialized");
    try
    {
        PipeShapeDetector detector;
        ros::spin();
    }
    catch (const ros::Exception &e)
    {
        ROS_ERROR("ROS exception: %s", e.what());
    }
    ROS_INFO("Node exiting");
    return 0;
}
