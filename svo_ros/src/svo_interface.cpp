#include <svo_ros/svo_interface.h>

#include <ros/callback_queue.h>

#include <svo_ros/svo_factory.h>
#include <svo_ros/visualizer.h>
#include <svo/map.h>
#include <svo/imu_handler.h>
#include <svo/common/frame.h>
#include <svo/common/camera.h>
#include <svo/common/conversions.h>
#include <svo/common/imu_calibration.h>
#include <svo/frame_handler_mono.h>
#include <svo/frame_handler_stereo.h>
#include <svo/frame_handler_array.h>
#include <svo/initialization.h>
#include <svo/direct/depth_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/subscriber_filter.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <vikit/params_helper.h>
#include <vikit/timer.h>
#include <svo_ros/ceres_backend_factory.h>

#ifdef SVO_USE_GTSAM_BACKEND
#include <svo_ros/backend_factory.h>
#include <svo/backend/backend_interface.h>
#include <svo/backend/backend_optimizer.h>
#endif

#ifdef SVO_LOOP_CLOSING
#include <svo/online_loopclosing/loop_closing.h>
#endif

#ifdef SVO_GLOBAL_MAP
#include <svo/global_map.h>
#endif

namespace svo {

SvoInterface::SvoInterface(const PipelineType& pipeline_type,
                           const ros::NodeHandle& nh,
                           const ros::NodeHandle& private_nh)
    : nh_(nh),
      pnh_(private_nh),
      pipeline_type_(pipeline_type),
    //   set_initial_attitude_from_gravity_(
    //       vk::param<bool>(pnh_, "set_initial_attitude_from_gravity", true)),
      automatic_reinitialization_(
          vk::param<bool>(pnh_, "automatic_reinitialization", false)) {
    switch (pipeline_type) {
        case PipelineType::kMono:
            svo_ = factory::makeMono(pnh_);
            break;
        case PipelineType::kStereo:
            svo_ = factory::makeStereo(pnh_);
            break;
        case PipelineType::kArray:
            svo_ = factory::makeArray(pnh_);
            break;
        default:
            LOG(FATAL) << "Unknown pipeline";
            break;
    }
    ncam_ = svo_->getNCamera();

    visualizer_.reset(
        new Visualizer(svo_->options_.trace_dir, pnh_, ncam_->getNumCameras()));

    if (vk::param<bool>(pnh_, "use_imu", false)) {
        imu_handler_ = factory::getImuHandler(pnh_);
        svo_->imu_handler_ = imu_handler_;
    }

    if (vk::param<bool>(pnh_, "use_ceres_backend", false)) {
        ceres_backend_interface_ =
            ceres_backend_factory::makeBackend(pnh_, ncam_);
        if (imu_handler_) {
            svo_->setBundleAdjuster(ceres_backend_interface_);
            ceres_backend_interface_->setImu(imu_handler_);
            ceres_backend_interface_->makePublisher(pnh_,
                                                    ceres_backend_publisher_);
        } else {
            SVO_ERROR_STREAM("Cannot use ceres backend without using imu");
        }
    }
#ifdef SVO_USE_GTSAM_BACKEND
    if (vk::param<bool>(pnh_, "use_backend", false)) {
        backend_interface_ = svo::backend_factory::makeBackend(pnh_);
        ceres_backend_publisher_.reset(
            new CeresBackendPublisher(svo_->options_.trace_dir, pnh_));
        svo_->setBundleAdjuster(backend_interface_);
        backend_interface_->imu_handler_ = imu_handler_;
    }
#endif
    if (vk::param<bool>(pnh_, "runlc", false)) {
#ifdef SVO_LOOP_CLOSING
        LoopClosingPtr loop_closing_ptr =
            factory::getLoopClosingModule(pnh_, svo_->getNCamera());
        svo_->lc_ = std::move(loop_closing_ptr);
        CHECK(svo_->depth_filter_->options_.extra_map_points)
            << "The depth filter seems to be initialized without extra map "
               "points.";
#else
        LOG(FATAL) << "You have to enable loop closing in svo_cmake.";
#endif
    }

    if (vk::param<bool>(pnh_, "use_global_map", false)) {
#ifdef SVO_GLOBAL_MAP
        svo_->global_map_ = factory::getGlobalMap(pnh_, svo_->getNCamera());
        if (imu_handler_) {
            svo_->global_map_->initializeIMUParams(imu_handler_->imu_calib_,
                                                   imu_handler_->imu_init_);
        }
#else
        LOG(FATAL) << "You have to enable global map in cmake";
#endif
    }
    g_w_ = Eigen::Vector3d(0, 0, imu_handler_->imu_calib_.gravity_magnitude);
    svo_->start();
}

SvoInterface::~SvoInterface() {
    if (imu_thread_) imu_thread_->join();
    if (image_thread_) image_thread_->join();
    VLOG(1) << "Destructed SVO.";
}

void SvoInterface::processImageBundle(const std::vector<cv::Mat>& images,
                                      const int64_t timestamp_nanoseconds) {
    if (!svo_->isBackendValid()) {
        if (vk::param<bool>(pnh_, "use_ceres_backend", false, true)) {
            ceres_backend_interface_ =
                ceres_backend_factory::makeBackend(pnh_, ncam_);
            if (imu_handler_) {
                svo_->setBundleAdjuster(ceres_backend_interface_);
                ceres_backend_interface_->setImu(imu_handler_);
                ceres_backend_interface_->makePublisher(
                    pnh_, ceres_backend_publisher_);
            } else {
                SVO_ERROR_STREAM("Cannot use ceres backend without using imu");
            }
        }
    }
    svo_->addImageBundle(images, timestamp_nanoseconds);
}

void SvoInterface::publishResults(const std::vector<cv::Mat>& images,
                                  const int64_t timestamp_nanoseconds) {
    CHECK_NOTNULL(svo_.get());
    CHECK_NOTNULL(visualizer_.get());

    visualizer_->img_caption_.clear();
    if (svo_->isBackendValid()) {
        std::string static_str =
            ceres_backend_interface_->getStationaryStatusStr();
        visualizer_->img_caption_ = static_str;
    }

    visualizer_->publishSvoInfo(svo_.get(), timestamp_nanoseconds);
    switch (svo_->stage()) {
        case Stage::kTracking: {
            // Eigen::Matrix<double, 6, 6> covariance;
            // covariance.setZero();
            // visualizer_->publishImuPose(svo_->getLastFrames()->get_T_W_B(),
            //                             covariance, timestamp_nanoseconds);
            visualizer_->publishCameraPoses(svo_->getLastFrames(),
                                            timestamp_nanoseconds);
            visualizer_->visualizeMarkers(svo_->getLastFrames(),
                                          svo_->closeKeyframes(), svo_->map());
            visualizer_->exportToDense(svo_->getLastFrames());
            bool draw_boundary = false;
            if (svo_->isBackendValid()) {
                draw_boundary = svo_->getBundleAdjuster()->isFixedToGlobalMap();
            }
            visualizer_->publishImagesWithFeatures(
                svo_->getLastFrames(), timestamp_nanoseconds, draw_boundary);
#ifdef SVO_LOOP_CLOSING
            // detections
            if (svo_->lc_) {
                visualizer_->publishLoopClosureInfo(
                    svo_->lc_->cur_loop_check_viz_info_,
                    std::string("loop_query"),
                    Eigen::Vector3f(0.0f, 0.0f, 1.0f), 0.5);
                visualizer_->publishLoopClosureInfo(
                    svo_->lc_->loop_detect_viz_info_,
                    std::string("loop_detection"),
                    Eigen::Vector3f(1.0f, 0.0f, 0.0f), 1.0);
                if (svo_->isBackendValid()) {
                    visualizer_->publishLoopClosureInfo(
                        svo_->lc_->loop_correction_viz_info_,
                        std::string("loop_correction"),
                        Eigen::Vector3f(0.0f, 1.0f, 0.0f), 3.0);
                }
                if (svo_->getLastFrames()->at(0)->isKeyframe()) {
                    bool pc_recalculated = visualizer_->publishPoseGraph(
                        svo_->lc_->kf_list_,
                        svo_->lc_->need_to_update_pose_graph_viz_,
                        static_cast<size_t>(
                            svo_->lc_->options_.ignored_past_frames));
                    if (pc_recalculated) {
                        svo_->lc_->need_to_update_pose_graph_viz_ = false;
                    }
                }
            }
#endif
#ifdef SVO_GLOBAL_MAP
            if (svo_->global_map_) {
                visualizer_->visualizeGlobalMap(
                    *(svo_->global_map_), std::string("global_vis"),
                    Eigen::Vector3f(0.0f, 0.0f, 1.0f), 0.3);
                visualizer_->visualizeFixedLandmarks(
                    svo_->getLastFrames()->at(0));
            }
#endif
            break;
        }
        case Stage::kInitializing: {
            visualizer_->publishBundleFeatureTracks(
                svo_->initializer_->frames_ref_, svo_->getLastFrames(),
                timestamp_nanoseconds);
            break;
        }
        case Stage::kPaused:
        case Stage::kRelocalization:
            visualizer_->publishImages(images, timestamp_nanoseconds);
            break;
        default:
            LOG(FATAL) << "Unknown stage";
            break;
    }

#ifdef SVO_USE_GTSAM_BACKEND
    if (svo_->stage() == Stage::kTracking && backend_interface_) {
        if (svo_->getLastFrames()->isKeyframe()) {
            std::lock_guard<std::mutex> estimate_lock(
                backend_interface_->optimizer_->estimate_mut_);
            const gtsam::Values& state =
                backend_interface_->optimizer_->estimate_;
            ceres_backend_publisher_->visualizeFrames(state);
            if (backend_interface_->options_.add_imu_factors)
                ceres_backend_publisher_->visualizeVelocity(state);
            ceres_backend_publisher_->visualizePoints(state);
        }
    }
#endif
}

bool SvoInterface::setImuPrior(const int64_t timestamp_nanoseconds) {
    if (svo_->getBundleAdjuster()) {
        // if we use backend, this will take care of setting priors
        if (!svo_->hasStarted()) {
            // when starting up, make sure we already have IMU measurements
            if (imu_handler_->getMeasurementsCopy().size() < 10u) {
                return false;
            }
        }
        return true;
    }

    if (imu_handler_ && !svo_->hasStarted() &&
        set_initial_attitude_from_gravity_) {
        // set initial orientation
        Quaternion R_imu_world;
        if (imu_handler_->getInitialAttitude(
                timestamp_nanoseconds *
                    common::conversions::kNanoSecondsToSeconds,
                R_imu_world)) {
            VLOG(3)
                << "Set initial orientation from accelerometer measurements.";
            svo_->setRotationPrior(R_imu_world);
        } else {
            return false;
        }
    } else if (imu_handler_ && svo_->getLastFrames()) {
        // set incremental rotation prior
        Quaternion R_lastimu_newimu;
        if (imu_handler_->getRelativeRotationPrior(
                svo_->getLastFrames()->getMinTimestampNanoseconds() *
                    common::conversions::kNanoSecondsToSeconds,
                timestamp_nanoseconds *
                    common::conversions::kNanoSecondsToSeconds,
                false, R_lastimu_newimu)) {
            VLOG(3) << "Set incremental rotation prior from IMU.";
            svo_->setRotationIncrementPrior(R_lastimu_newimu);
        }
    }
    return true;
}

void SvoInterface::monoCallback(const sensor_msgs::ImageConstPtr& msg) {
    if (idle_) return;
    cv::Mat image;
    try {
        image = cv_bridge::toCvCopy(msg)->image;
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }

    std::vector<cv::Mat> images;
    images.push_back(image.clone());

    if (!setImuPrior(msg->header.stamp.toNSec())) {
        VLOG(3)
            << "Could not align gravity! Attempting again in next iteration.";
        return;
    }

    imageCallbackPreprocessing(msg->header.stamp.toNSec());
    processImageBundle(images, msg->header.stamp.toNSec());
    updateLastetState(msg->header.stamp.toNSec());
    publishResults(images, msg->header.stamp.toNSec());

    if (svo_->stage() == Stage::kPaused && automatic_reinitialization_)
        svo_->start();

    imageCallbackPostprocessing();
}

void SvoInterface::stereoCallback(const sensor_msgs::ImageConstPtr& msg0,
                                  const sensor_msgs::ImageConstPtr& msg1) {
    if (idle_) return;

    cv::Mat img0, img1;
    try {
        img0 = cv_bridge::toCvShare(msg0, "mono8")->image;
        img1 = cv_bridge::toCvShare(msg1, "mono8")->image;
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }

    if (!setImuPrior(msg0->header.stamp.toNSec())) {
        VLOG(3)
            << "Could not align gravity! Attempting again in next iteration.";
        return;
    }
    imageCallbackPreprocessing(msg0->header.stamp.toNSec());
    processImageBundle({img0, img1}, msg0->header.stamp.toNSec());
    updateLastetState(msg0->header.stamp.toNSec());
    publishResults({img0, img1}, msg0->header.stamp.toNSec());
    if (svo_->stage() == Stage::kPaused && automatic_reinitialization_)
        svo_->start();
    imageCallbackPostprocessing();
}

void SvoInterface::imuCallback(const sensor_msgs::ImuConstPtr& msg) {
    const Eigen::Vector3d omega_imu(msg->angular_velocity.x,
                                    msg->angular_velocity.y,
                                    msg->angular_velocity.z);
    const Eigen::Vector3d lin_acc_imu(msg->linear_acceleration.x,
                                      msg->linear_acceleration.y,
                                      msg->linear_acceleration.z);
    const ImuMeasurement m(msg->header.stamp.toSec(), omega_imu, lin_acc_imu);
    if (imu_handler_) {
        imu_handler_->addImuMeasurement(m);
    } else {
        SVO_ERROR_STREAM("SvoNode has no ImuHandler");
    }
    if(frontend_stabel_){
        // 1. Make sure all imu measurements are used
        // 2. To keep stable output frequency
        // 3. How to set window_states_?(TODO)
        NavState cur_state, smooth_state;
        cur_state.nano_timestamp_ = static_cast<uint64_t>(msg->header.stamp.toNSec());
        if(getLastestImuPose(m, &cur_state)) {
            // // smooth output
            // bool flag = false;
            // if(smoothImuPoseOutput(cur_state, &smooth_state, &flag)) {
            //     if(flag) {
            //         Transformation T_world_imu
            //             = Transformation(smooth_state.q_w_imu_, smooth_state.p_w_imu_);
            //         visualizer_->publishImuPose(
            //             T_world_imu, smooth_state.pose_cov_, smooth_state.nano_timestamp_);
            //         visualizer_->pubIMUPath(smooth_state);
            //     } else {
            //         Transformation T_world_imu = Transformation(cur_state.q_w_imu_, cur_state.p_w_imu_);
            //         visualizer_->publishImuPose(
            //             T_world_imu, cur_state.pose_cov_, cur_state.nano_timestamp_);
            //         visualizer_->pubIMUPath(cur_state);
            //     }
            // }
            Transformation T_world_imu = Transformation(cur_state.q_w_imu_, cur_state.p_w_imu_);
            visualizer_->publishImuPose(T_world_imu, cur_state.pose_cov_, cur_state.nano_timestamp_);
            visualizer_->pubIMUPath(cur_state);
        }
    }
}

void SvoInterface::inputKeyCallback(const std_msgs::StringConstPtr& key_input) {
    std::string remote_input = key_input->data;
    char input = remote_input.c_str()[0];
    switch (input) {
        case 'q':
            quit_ = true;
            SVO_INFO_STREAM("SVO user input: QUIT");
            break;
        case 'r':
            svo_->reset();
            idle_ = true;
            SVO_INFO_STREAM("SVO user input: RESET");
            break;
        case 's':
            svo_->start();
            idle_ = false;
            SVO_INFO_STREAM("SVO user input: START");
            break;
        case 'c':
            svo_->setCompensation(true);
            SVO_INFO_STREAM("Enabled affine compensation.");
            break;
        case 'C':
            svo_->setCompensation(false);
            SVO_INFO_STREAM("Disabled affine compensation.");
            break;
        default:;
    }
}

void SvoInterface::subscribeImu() {
    imu_thread_ = std::unique_ptr<std::thread>(
        new std::thread(&SvoInterface::imuLoop, this));
    // sleep(3);
}

void SvoInterface::subscribeImage() {
    if (pipeline_type_ == PipelineType::kMono)
        image_thread_ = std::unique_ptr<std::thread>(
            new std::thread(&SvoInterface::monoLoop, this));
    else if (pipeline_type_ == PipelineType::kStereo)
        image_thread_ = std::unique_ptr<std::thread>(
            new std::thread(&SvoInterface::stereoLoop, this));
}

void SvoInterface::subscribeRemoteKey() {
    std::string remote_key_topic =
        vk::param<std::string>(pnh_, "remote_key_topic", "svo/remote_key");
    sub_remote_key_ = nh_.subscribe(remote_key_topic, 5,
                                    &svo::SvoInterface::inputKeyCallback, this);
}

void SvoInterface::imuLoop() {
    SVO_INFO_STREAM("SvoNode: Started IMU loop.");
    ros::NodeHandle nh;
    ros::CallbackQueue queue;
    nh.setCallbackQueue(&queue);
    std::string imu_topic = vk::param<std::string>(pnh_, "imu_topic", "imu");
    ros::Subscriber sub_imu =
        nh.subscribe(imu_topic, 10, &svo::SvoInterface::imuCallback, this);
    while (ros::ok() && !quit_) {
        queue.callAvailable(ros::WallDuration(0.1));
    }
}

void SvoInterface::monoLoop() {
    SVO_INFO_STREAM("SvoNode: Started Image loop.");

    ros::NodeHandle nh;
    ros::CallbackQueue queue;
    nh.setCallbackQueue(&queue);

    image_transport::ImageTransport it(nh);
    std::string image_topic =
        vk::param<std::string>(pnh_, "cam0_topic", "camera/image_raw");
    image_transport::Subscriber it_sub =
        it.subscribe(image_topic, 5, &svo::SvoInterface::monoCallback, this);

    while (ros::ok() && !quit_) {
        queue.callAvailable(ros::WallDuration(0.1));
    }
}

void SvoInterface::stereoLoop() {
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image,
                                                      sensor_msgs::Image>
        ExactPolicy;
    typedef message_filters::Synchronizer<ExactPolicy> ExactSync;

    ros::NodeHandle nh(nh_, "image_thread");
    ros::CallbackQueue queue;
    nh.setCallbackQueue(&queue);
    // subscribe to cam msgs
    std::string cam0_topic(
        vk::param<std::string>(pnh_, "cam0_topic", "/cam0/image_raw"));
    std::string cam1_topic(
        vk::param<std::string>(pnh_, "cam1_topic", "/cam1/image_raw"));
    image_transport::ImageTransport it(nh);
    image_transport::SubscriberFilter sub0(it, cam0_topic, 1,
                                           std::string("raw"));
    image_transport::SubscriberFilter sub1(it, cam1_topic, 1,
                                           std::string("raw"));
    ExactSync sync_sub(ExactPolicy(5), sub0, sub1);
    sync_sub.registerCallback(
        boost::bind(&svo::SvoInterface::stereoCallback, this, _1, _2));

    while (ros::ok() && !quit_) {
        queue.callAvailable(ros::WallDuration(0.1));
    }
}

void SvoInterface::updateLastetState(int64_t timestamp_nanosecond){
    if(svo_->getLastFrames()->imu_measurements_deque_.size() > 0) {
        ImuMeasurement imu_measurement;
        imu_measurement = svo_->getLastFrames()->imu_measurements_deque_.at(0);
        Transformation T_world_imu = svo_->getLastFrames()->get_T_W_B();
        Eigen::Vector3d speed, gyr_bias, acc_bias;
        svo_->getLastFrames()->getIMUState(&speed, &gyr_bias, &acc_bias);
        {
            std::lock_guard<std::mutex> lock(update_lastest_state_lock_);
            lastest_state_.nano_timestamp_ = static_cast<uint64_t>(timestamp_nanosecond);
            lastest_state_.q_w_imu_ = T_world_imu.getRotation().toImplementation();
            lastest_state_.p_w_imu_ = T_world_imu.getPosition();
            lastest_state_.gyr_bias_ = gyr_bias;
            lastest_state_.acc_bias_ = acc_bias;
            lastest_state_.linear_speed_ = speed;
            lastest_state_.linear_acceleration_ = imu_measurement.linear_acceleration_;
            lastest_state_.angular_speed_ = imu_measurement.angular_velocity_;
            if(!frontend_stabel_) {
                frontend_stabel_ = true;
            }
        }
    }
}

bool SvoInterface::getLastestImuPose(
    const ImuMeasurement imu_measurement,
    NavState *cur_state) {
    std::lock_guard<std::mutex> lock(update_lastest_state_lock_);
    if( cur_state->nano_timestamp_ <= lastest_state_.nano_timestamp_) {
        return false;
    }
    double dt = (cur_state->nano_timestamp_ - lastest_state_.nano_timestamp_) * 1e-9;
    Eigen::Quaterniond q_w_i = lastest_state_.q_w_imu_;
    q_w_i.normalize();
    Eigen::Vector3d p_i = lastest_state_.p_w_imu_;
    Eigen::Vector3d un_acc_0 = q_w_i * (
        lastest_state_.linear_acceleration_ - lastest_state_.acc_bias_) - g_w_;
    Eigen::Vector3d un_gyr =
        0.5 * (lastest_state_.angular_speed_ + imu_measurement.angular_velocity_) - lastest_state_.gyr_bias_;

    Eigen::Quaterniond q_w_j = q_w_i * deltaQ(un_gyr * dt);
    q_w_j.normalize();
    {
        // std::cout<<"dt: "<<dt<<std::endl;
        // std::cout<<"un_gyr: "<<un_gyr.transpose()<<std::endl;
        // std::cout
        //     <<"q_w_i = "
        //     << q_w_i.w() << ", "
        //     << q_w_i.x() << ", "
        //     << q_w_i.y() << ", "
        //     << q_w_i.z() << ", "
        //     <<std::endl;
        // std::cout
        //     <<"q_w_j = "
        //     << q_w_j.w() << ", "
        //     << q_w_j.x() << ", "
        //     << q_w_j.y() << ", "
        //     << q_w_j.z() << ", "
        //     <<std::endl;
    }
    Eigen::Vector3d un_acc_1 = q_w_j * (imu_measurement.linear_acceleration_ - lastest_state_.acc_bias_) - g_w_;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    Eigen::Vector3d p_j = p_i + dt * lastest_state_.linear_speed_ + 0.5 * dt * dt * un_acc;
    Eigen::Vector3d v_j = lastest_state_.linear_speed_ + un_acc * dt;
    {
        cur_state->q_w_imu_ = q_w_j;
        cur_state->p_w_imu_ = p_j;
        cur_state->gyr_bias_ = lastest_state_.gyr_bias_;
        cur_state->acc_bias_ = lastest_state_.acc_bias_;
        cur_state->linear_speed_ = v_j;
        cur_state->linear_acceleration_ = imu_measurement.linear_acceleration_;
        cur_state->angular_speed_ = imu_measurement.angular_velocity_;
    }
    lastest_state_ = *cur_state;
    return true;
}

bool SvoInterface::smoothImuPoseOutput(
    const NavState &cur_state, NavState *smooth_state, bool *flag) {
    // TODO(yehonghua)
    // static constexpr uint64_t window_gap = 1e7;
    static constexpr uint64_t window_gap = 1e6; // window_gap = 1/hz*1e9
    // if(window_states_.empty()) {
    if(window_states_.size() < 10) {
        *smooth_state = cur_state;
        smooth_state->nano_timestamp_ = static_cast<uint64_t>(
            std::round(smooth_state->nano_timestamp_ * 1.0 / window_gap)) * window_gap;
        window_states_.push_back(*smooth_state);
        std::cout<<"window_states_.empty()"<<std::endl;
        *flag = false;
        return false;
    } else {
        std::cout<<"window_states_.size()"<< window_states_.size()<<std::endl;
        uint64_t last_standard_time = window_states_.back().nano_timestamp_;
        if (cur_state.nano_timestamp_ <= last_standard_time) {
            // default value
            NavState nav_state;
            nav_state.nano_timestamp_ = 0;
            *smooth_state = nav_state;
            std::cout<<"cur_state.nano_timestamp_ <= last_standard_time"<<std::endl;
            return false;
        }
        double time_gap = (static_cast<double>(cur_state.nano_timestamp_) -
            last_standard_time) / static_cast<double>(window_gap);
        int interp_num = static_cast<int>(std::round(time_gap));
        *flag = processSmoothNavState(cur_state, smooth_state);
        if (interp_num >= 1) {
            std::list<NavState> interp_nav_states;
            const NavState &s_ns = window_states_.back();
            const NavState &e_ns = *smooth_state;
            // smooth_ns 基于 smooth_ns 进行结果插值
            for (int i = 1; i <= interp_num; ++i) {
                NavState n_state;
                // 严格保证了定位查询窗口中相邻两帧的定位时间戳偏差10ms
                uint64_t c_time = last_standard_time + window_gap * i;
                navStateInterp(s_ns, e_ns, i / time_gap, &n_state);
                n_state.nano_timestamp_ = c_time;
                interp_nav_states.push_back(n_state);
            }
            // 将处理后的定位结果 interp_nav_states 插入 定位窗口中
            window_states_.insert(window_states_.end(),
                                  interp_nav_states.begin(),
                                  interp_nav_states.end());
            size_t window_size = 1000; // "window_size": 200,
            while (window_states_.size() > window_size) {
                window_states_.erase(window_states_.begin());
            }
            *smooth_state = window_states_.back();
        }
    }
    return true;
}

bool SvoInterface::processSmoothNavState(
    const NavState &cur_state, NavState *smooth_state) {
    static constexpr size_t base_num = 10;
    if (window_states_.size() < base_num) {
        (*smooth_state) = cur_state;
        std::cout<<"processSmoothNavState window_states_.size() < base_num!!!"<<std::endl;
        return false;
    }
    NavState lastest_state, last_state;
    size_t index = 0;
    for (auto iter = window_states_.rbegin(); iter != window_states_.rend();
         ++iter) {
        if (index == 0) {
            lastest_state = *iter;
        }
        if (index == base_num - 1) {
            last_state = *iter;
            break;
        }
        ++index;
    }
    double factor = 1.0 * (cur_state.nano_timestamp_ - last_state.nano_timestamp_) /
                    (lastest_state.nano_timestamp_ - last_state.nano_timestamp_);
    if (factor <= 1.0) {
        (*smooth_state) = cur_state;
        std::cout<<"processSmoothNavState factor <= 1.0!!!"<<std::endl;
        return false;
    }
    NavState predict_state;
    predict_state.nano_timestamp_ = cur_state.nano_timestamp_;
    // printNavstate(last_state);
    // printNavstate(lastest_state);
    navStateInterp(last_state, lastest_state, factor, &predict_state);
    // printNavstate(predict_state);
    smooth_state->nano_timestamp_ = cur_state.nano_timestamp_;
    navStateInterp(predict_state, cur_state, 0.6, smooth_state);
    return true;
}

void SvoInterface::navStateInterp(const NavState &s_ns,
                                         const NavState &e_ns,
                                         double factor,
                                         NavState *ns) {
    ns->q_w_imu_ = s_ns.q_w_imu_.slerp(factor, e_ns.q_w_imu_);
    ns->q_w_imu_.normalize();
    ns->p_w_imu_ = s_ns.p_w_imu_ + (e_ns.p_w_imu_ - s_ns.p_w_imu_) * factor;
    ns->linear_speed_ =
        s_ns.linear_speed_ + (e_ns.linear_speed_ - s_ns.linear_speed_) * factor;
    ns->linear_acceleration_ =
        s_ns.linear_acceleration_ +
        (e_ns.linear_acceleration_ - s_ns.linear_acceleration_) * factor;
    ns->angular_speed_ =
        s_ns.angular_speed_ + (e_ns.angular_speed_ - s_ns.angular_speed_) * factor;
    // TODO(yehonghua) How about covariance ?
    ns->pose_cov_ = e_ns.pose_cov_;
}

}  // namespace svo
