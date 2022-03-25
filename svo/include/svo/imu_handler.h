// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <memory>  // std::shared_ptr
#include <mutex>   // std::mutex
#include <iostream>
#include <fstream>
#include <deque>
#include <map>
#include <svo/common/types.h>
#include <svo/common/transformation.h>
#include <svo/common/imu_calibration.h>
#include <svo/vio_common/matrix_operations.hpp>

namespace svo {

class PreintegratedImuMeasurement {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d omega_bias_;
    Eigen::Vector3d acc_bias_;
    Eigen::Vector3d delta_t_ij_;
    Eigen::Vector3d delta_v_ij_;
    Quaternion delta_R_ij_;
    double dt_sum_;

    PreintegratedImuMeasurement(const Eigen::Vector3d& omega_bias,
                                const Eigen::Vector3d& acc_bias);

    /// Add single measurements to be integrated
    void addMeasurement(const ImuMeasurement& m);

    /// Add many measurements to be integrated
    void addMeasurements(const ImuMeasurements& ms);

 private:
    bool last_imu_measurement_set_;
    ImuMeasurement last_imu_measurement;
};

struct IMUHandlerOptions {
    bool temporal_stationary_check = false;
    double temporal_window_length_sec_ = 0.5;
    double stationary_acc_sigma_thresh_ = 10e-4;
    double stationary_gyr_sigma_thresh_ = 6e-5;
};

enum class IMUTemporalStatus { kStationary, kMoving, kUnkown };

extern const std::map<IMUTemporalStatus, std::string>
    imu_temporal_status_names_;

class ImuHandler {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<ImuHandler> Ptr;
    typedef std::mutex mutex_t;
    typedef std::unique_lock<mutex_t> ulock_t;

    IMUHandlerOptions options_;
    ImuCalibration imu_calib_;
    ImuInitialization imu_init_;

    // TODO: make private
    mutable mutex_t bias_mut_;
    Eigen::Vector3d acc_bias_;  //!< Accleration bias used during preintegration
    Eigen::Vector3d
        omega_bias_;  //!< Angular rate bias values used during preintegration

    ImuHandler(const ImuCalibration& imu_calib,
               const ImuInitialization& imu_init,
               const IMUHandlerOptions& options);
    ~ImuHandler();

    const Eigen::Vector3d& getAccelerometerBias() const {
        ulock_t lock(bias_mut_);
        return acc_bias_;
    }

    const Eigen::Vector3d& getGyroscopeBias() const {
        ulock_t lock(bias_mut_);
        return omega_bias_;
    }

    void setAccelerometerBias(const Eigen::Vector3d& acc_bias) {
        ulock_t lock(bias_mut_);
        acc_bias_ = acc_bias;
    }

    void setGyroscopeBias(const Eigen::Vector3d& omega_bias) {
        ulock_t lock(bias_mut_);
        omega_bias_ = omega_bias;
    }

    ImuMeasurements getMeasurementsCopy() const {
        ulock_t lock(measurements_mut_);
        return measurements_;
    }

    /// Get IMU measurements in some time interval. Note that you have to
    /// provide
    /// the camera timestamps. Internally, given the calibration it corrects the
    /// timestamps for delays.
    bool getMeasurements(const double old_cam_timestamp,  // seconds
                         const double new_cam_timestamp,  // seconds
                         const bool delete_old_measurements,
                         ImuMeasurements& measurements);

    /// Get IMU measurements up to  for the exact borders.
    /// Note that you have to provide the camera timestamp.
    /// Internally, given the calibration it corrects the timestamps for delays.
    /// The returned measurement will cover the full timeinterval
    /// (getMeasurements only gives newest measurement smaller than
    /// new_cam_timestamp
    bool getMeasurementsContainingEdges(
        const double frame_timestamp,  // seoncds
        ImuMeasurements& measurements,
        const bool remove_measurements);

    bool getClosestMeasurement(const double timestamp,
                               ImuMeasurement& measurement) const;

    // deprecated, use preintegrated imu measurement!
    /// Gets relative transformation in IMU coordinate frame
    bool getRelativeRotationPrior(const double old_cam_timestamp,
                                  const double new_cam_timestamp,
                                  bool delete_old_measurements,
                                  Quaternion& R_oldimu_newimu);

    bool getAngularVelocity(double timestamp, Eigen::Vector3d& omega) const;

    /// Assumes we are in hover condition and estimates the inital orientation
    /// by
    /// estimating the gravity direction. The yaw direction is not
    /// deterministic.
    bool getInitialAttitude(double timestamp, Quaternion& R_imu_world) const;

    bool addImuMeasurement(const ImuMeasurement& measurement);

    bool loadImuMeasurementsFromFile(const std::string& filename);

    bool loadImuMeasurementsFromCsvFile(const std::string& filename);

    static Eigen::Matrix3d integrateGyroMeasurement(
        const Eigen::Vector3d& omega_measured,
        const Eigen::Matrix3d& R_cam_imu,
        const double delta_t);

    static ImuCalibration loadCalibrationFromFile(const std::string& filename);
    static ImuInitialization loadInitializationFromFile(
        const std::string& filename);

    void reset();

    double getLatestTimestamp() const {
        ulock_t lock(measurements_mut_);
        return measurements_.front().timestamp_;
    }

    bool waitTill(const double timestamp_sec, const double timeout_sec = 1.0);

    IMUTemporalStatus checkTemporalStatus(const double time_sec);


    // to make things a bit faster than using angle-axis conversion:
    __inline__ double sinc(double x) {
        if (fabs(x) > 1e-6) {
            return sin(x) / x;
        } else {
            static const double c_2 = 1.0 / 6.0;
            static const double c_4 = 1.0 / 120.0;
            static const double c_6 = 1.0 / 5040.0;
            const double x_2 = x * x;
            const double x_4 = x_2 * x_2;
            const double x_6 = x_2 * x_2 * x_2;
            return 1.0 - c_2 * x_2 + c_4 * x_4 - c_6 * x_6;
        }
    }
    __inline__ Eigen::Quaterniond deltaQ(const Eigen::Vector3d& dAlpha) {
        Eigen::Vector4d dq;
        double halfnorm = 0.5 * dAlpha.template tail<3>().norm();
        dq.template head<3>() = sinc(halfnorm) * 0.5 * dAlpha.template tail<3>();
        dq[3] = cos(halfnorm);
        return Eigen::Quaterniond(dq);
    }

    // Propagates pose, speeds and biases with given IMU measurements.
    int propagation(const ImuMeasurements& imu_measurements,
                            Transformation& T_WS,
                            Eigen::Matrix<double, 9, 1>& speed_and_biases,
                            const double& t_start,
                            const double& t_end) {
        const double t_start_adjusted = t_start - imu_calib_.delay_imu_cam;
        const double t_end_adjusted = t_end - imu_calib_.delay_imu_cam;
        // sanity check:
        assert(imu_measurements.back().timestamp_ <= t_start_adjusted);
        if (!(imu_measurements.front().timestamp_ >= t_end_adjusted)) {
            assert(false);
            return -1;  // nothing to do...
        }
        // initial condition
        Eigen::Vector3d r_0 = T_WS.getPosition();
        Eigen::Quaterniond q_WS_0 = T_WS.getEigenQuaternion();
        Eigen::Matrix3d C_WS_0 = T_WS.getRotationMatrix();
        // increments (initialise with identity)
        Eigen::Quaterniond Delta_q(1, 0, 0, 0);
        Eigen::Matrix3d C_integral = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d C_doubleintegral = Eigen::Matrix3d::Zero();
        Eigen::Vector3d acc_integral = Eigen::Vector3d::Zero();
        Eigen::Vector3d acc_doubleintegral = Eigen::Vector3d::Zero();

        // cross matrix accumulatrion
        Eigen::Matrix3d cross = Eigen::Matrix3d::Zero();

        // sub-Jacobians
        Eigen::Matrix3d dalpha_db_g = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d dv_db_g = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d dp_db_g = Eigen::Matrix3d::Zero();

        // the Jacobian of the increment (w/o biases)
        Eigen::Matrix<double, 15, 15> P_delta =
            Eigen::Matrix<double, 15, 15>::Zero();

        double Delta_t = 0;
        bool has_started = false;
        int num_propagated = 0;

        double time = t_start_adjusted;
        for (size_t i = imu_measurements.size() - 1; i != 0u; --i) {
            Eigen::Vector3d omega_S_0 = imu_measurements[i].angular_velocity_;
            Eigen::Vector3d acc_S_0 = imu_measurements[i].linear_acceleration_;
            Eigen::Vector3d omega_S_1 = imu_measurements[i - 1].angular_velocity_;
            Eigen::Vector3d acc_S_1 = imu_measurements[i - 1].linear_acceleration_;
            double nexttime = imu_measurements[i - 1].timestamp_;

            // time delta
            double dt = nexttime - time;

            if (t_end_adjusted < nexttime) {
                double interval = nexttime - imu_measurements[i].timestamp_;
                nexttime = t_end_adjusted;
                dt = nexttime - time;
                const double r = dt / interval;
                omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
                acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
            }

            if (dt <= 0.0) {
                continue;
            }
            Delta_t += dt;

            if (!has_started) {
                has_started = true;
                const double r = dt / (nexttime - imu_measurements[i].timestamp_);
                omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
                acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
            }

            // ensure integrity
            double sigma_g_c = imu_calib_.gyro_noise_density;
            double sigma_a_c = imu_calib_.acc_noise_density;

            if (std::abs(omega_S_0[0]) > imu_calib_.saturation_omega_max ||
                std::abs(omega_S_0[1]) > imu_calib_.saturation_omega_max ||
                std::abs(omega_S_0[2]) > imu_calib_.saturation_omega_max ||
                std::abs(omega_S_1[0]) > imu_calib_.saturation_omega_max ||
                std::abs(omega_S_1[1]) > imu_calib_.saturation_omega_max ||
                std::abs(omega_S_1[2]) > imu_calib_.saturation_omega_max) {
                sigma_g_c *= 100;
                LOG(WARNING) << "gyr saturation";
            }

            if (std::abs(acc_S_0[0]) > imu_calib_.saturation_accel_max ||
                std::abs(acc_S_0[1]) > imu_calib_.saturation_accel_max ||
                std::abs(acc_S_0[2]) > imu_calib_.saturation_accel_max ||
                std::abs(acc_S_1[0]) > imu_calib_.saturation_accel_max ||
                std::abs(acc_S_1[1]) > imu_calib_.saturation_accel_max ||
                std::abs(acc_S_1[2]) > imu_calib_.saturation_accel_max) {
                sigma_a_c *= 100;
                LOG(WARNING) << "acc saturation";
            }

            // actual propagation
            // orientation:
            Eigen::Quaterniond dq;
            const Eigen::Vector3d omega_S_true =
                (0.5 * (omega_S_0 + omega_S_1) - speed_and_biases.segment<3>(3));
            const double theta_half = omega_S_true.norm() * 0.5 * dt;
            const double sinc_theta_half = sinc(theta_half);
            const double cos_theta_half = cos(theta_half);
            dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
            dq.w() = cos_theta_half;
            Eigen::Quaterniond Delta_q_1 = Delta_q * dq;
            // rotation matrix integral:
            const Eigen::Matrix3d C = Delta_q.toRotationMatrix();
            const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();
            const Eigen::Vector3d acc_S_true =
                (0.5 * (acc_S_0 + acc_S_1) - speed_and_biases.segment<3>(6));
            const Eigen::Matrix3d C_integral_1 = C_integral + 0.5 * (C + C_1) * dt;
            const Eigen::Vector3d acc_integral_1 =
                acc_integral + 0.5 * (C + C_1) * acc_S_true * dt;
            // rotation matrix double integral:
            C_doubleintegral += C_integral * dt + 0.25 * (C + C_1) * dt * dt;
            acc_doubleintegral +=
                acc_integral * dt + 0.25 * (C + C_1) * acc_S_true * dt * dt;

            // Jacobian parts
            dalpha_db_g += dt * C_1;
            const Eigen::Matrix3d cross_1 =
                dq.inverse().toRotationMatrix() * cross +
                expmapDerivativeSO3(omega_S_true * dt) * dt;
            const Eigen::Matrix3d acc_S_x = skewSymmetric(acc_S_true);
            Eigen::Matrix3d dv_db_g_1 =
                dv_db_g +
                0.5 * dt * (C * acc_S_x * cross + C_1 * acc_S_x * cross_1);
            dp_db_g +=
                dt * dv_db_g +
                0.25 * dt * dt * (C * acc_S_x * cross + C_1 * acc_S_x * cross_1);

            // memory shift
            Delta_q = Delta_q_1;
            C_integral = C_integral_1;
            acc_integral = acc_integral_1;
            cross = cross_1;
            dv_db_g = dv_db_g_1;
            time = nexttime;

            ++num_propagated;

            if (nexttime == t_end_adjusted) break;
        }

        // actual propagation output:
        const Eigen::Vector3d g_W = imu_calib_.gravity_magnitude * Eigen::Vector3d(0, 0, 1.0);
        T_WS = Transformation(
            r_0 + speed_and_biases.head<3>() * Delta_t +
                C_WS_0 *
                    (acc_doubleintegral /*-C_doubleintegral*speedAndBiases.segment<3>(6)*/) -
                0.5 * g_W * Delta_t * Delta_t,
            q_WS_0 * Delta_q);
        speed_and_biases.head<3>() +=
            C_WS_0 * (acc_integral /*-C_integral*speedAndBiases.segment<3>(6)*/) -
            g_W * Delta_t;
        return num_propagated;
    }

 private:
    mutable mutex_t measurements_mut_;
    ImuMeasurements
        measurements_;  ///< Newest measurement is at the front of the list
    ImuMeasurements temporal_imu_window_;
    std::ofstream ofs_;  //!< File stream for tracing the received measurments
};

}  // namespace svo
