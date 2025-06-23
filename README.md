# Large Scale Tunnel Air-Ground Collaboration With FLISP: Fast LiDAR-IMU Synchronized Path Planner
To start FLISP, you only need to start ugv_path and uav_path in sequence to complete the path planning. ugv_path requires the lidar point cloud topic and the imu topic as inputs. uav_path requires ugv_path to run first and then subscribe to the point cloud topic to avoid obstacles.
