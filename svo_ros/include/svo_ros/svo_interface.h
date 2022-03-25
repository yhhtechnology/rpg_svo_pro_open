#pragma once

#include <thread>

#include <ros/ros.h>
#include <std_msgs/String.h>    // user-input
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <mutex>
#include <Eigen/Core>

#include <svo/common/types.h>
#include <svo/common/camera_fwd.h>
#include <svo/common/transformation.h>
#include <svo/common/imu_calibration.h>
#include <vikit/timer.h>

namespace svo {

// forward declarations
class FrameHandlerBase;
class Visualizer;
class ImuHandler;
class BackendInterface;
class CeresBackendInterface;
class CeresBackendPublisher;

enum class PipelineType {
  kMono,
  kStereo,
  kArray
};

typedef struct {
    // timestamp
    uint64_t nano_timestamp_;
    // Position and Rotation
    // SE3 pose;
    Eigen::Quaterniond q_w_imu_;
    Eigen::Vector3d p_w_imu_;
    // Velocity
    Eigen::Vector3d linear_speed_;
    // acceleration
    Eigen::Vector3d linear_acceleration_;
    // angular speed
    Eigen::Vector3d angular_speed_;
    Eigen::Vector3d gyr_bias_;
    Eigen::Vector3d acc_bias_;
    // Covariance of Position and Rotation
    Eigen::Matrix<double, 6, 6> pose_cov_;
    // state source
    // navigation status
} NavState;

/// SVO Interface
class SvoInterface
{
public:

  // ROS subscription and publishing.
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  PipelineType pipeline_type_;
  ros::Subscriber sub_remote_key_;
  std::string remote_input_;
  std::unique_ptr<std::thread> imu_thread_;
  std::unique_ptr<std::thread> image_thread_;

  // SVO modules.
  std::shared_ptr<FrameHandlerBase> svo_;
  std::shared_ptr<Visualizer> visualizer_;
  std::shared_ptr<ImuHandler> imu_handler_;
  std::shared_ptr<BackendInterface> backend_interface_;
  std::shared_ptr<CeresBackendInterface> ceres_backend_interface_;
  std::shared_ptr<CeresBackendPublisher> ceres_backend_publisher_;

  CameraBundlePtr ncam_;

  // Parameters
  bool set_initial_attitude_from_gravity_ = true;

  // System state.
  bool quit_ = false;
  bool idle_ = false;
  bool automatic_reinitialization_ = false;

  SvoInterface(const PipelineType& pipeline_type,
          const ros::NodeHandle& nh,
          const ros::NodeHandle& private_nh);

  virtual ~SvoInterface();

  // Processing
  void processImageBundle(
      const std::vector<cv::Mat>& images,
      int64_t timestamp_nanoseconds);

  bool setImuPrior(const int64_t timestamp_nanoseconds);

  void publishResults(
      const std::vector<cv::Mat>& images,
      const int64_t timestamp_nanoseconds);

  void updateLastetState(int64_t timestamp_nanosecond);
  std::mutex update_lastest_state_lock_;
  std::mutex window_lock_;
  std::mutex imu_data_lock_;
  Eigen::Vector3d g_w_;
  std::list<NavState> window_states_;
  NavState lastest_state_;
  bool frontend_stabel_ = false;
  vk::Timer imu_time_;


  bool getLastestImuPose(
      const ImuMeasurement imu_measurement,
      NavState *cur_state);
  bool smoothImuPoseOutput(
    const NavState &cur_state,
    NavState *smooth_state,
    bool *flag);
  bool processSmoothNavState(
    const NavState &cur_state,
    NavState *smooth_state);


    void printNavstate(NavState navstate){
        std::cout
            <<"navstate.q_w_imu_ = "
            << navstate.q_w_imu_.w() << ", "
            << navstate.q_w_imu_.x() << ", "
            << navstate.q_w_imu_.y() << ", "
            << navstate.q_w_imu_.z() << ", "
            <<"; navstate.p_w_imu_"<< navstate.p_w_imu_.transpose()
            <<std::endl;
    }

    void navStateInterp(const NavState &s_ns,
                        const NavState &e_ns,
                        double factor,
                        NavState *ns);

  template <typename Derived>
  static Eigen::Quaternion<typename Derived::Scalar> deltaQ(
      const Eigen::MatrixBase<Derived> &theta) {
      typedef typename Derived::Scalar Scalar_t;
      Eigen::Quaternion<Scalar_t> dq;
      Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
      half_theta /= static_cast<Scalar_t>(2.0);
      dq.w() = static_cast<Scalar_t>(1.0);
      dq.x() = half_theta.x();
      dq.y() = half_theta.y();
      dq.z() = half_theta.z();
      return dq;
  }

  // Subscription and callbacks
  void monoCallback(const sensor_msgs::ImageConstPtr& msg);
  void stereoCallback(
      const sensor_msgs::ImageConstPtr& msg0,
      const sensor_msgs::ImageConstPtr& msg1);
  void imuCallback(const sensor_msgs::ImuConstPtr& imu_msg);
  void inputKeyCallback(const std_msgs::StringConstPtr& key_input);


  // These functions are called before and after monoCallback or stereoCallback.
  // a derived class can implement some additional logic here.
  virtual void imageCallbackPreprocessing(int64_t timestamp_nanoseconds) {}
  virtual void imageCallbackPostprocessing() {}

  void subscribeImu();
  void subscribeImage();
  void subscribeRemoteKey();

  void imuLoop();
  void monoLoop();
  void stereoLoop();
};

} // namespace svo
