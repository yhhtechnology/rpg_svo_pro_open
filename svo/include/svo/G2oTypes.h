
#ifndef G2OTYPES_H
#define G2OTYPES_H

#include "../Thirdparty/g2o/g2o/core/base_vertex.h"
#include "../Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "../Thirdparty/g2o/g2o/types/types_sba.h"
#include "../Thirdparty/g2o/g2o/core/base_multi_edge.h"
#include "../Thirdparty/g2o/g2o/core/base_unary_edge.h"

#include <opencv2/core/core.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

// #include <Frame.h>
// #include <KeyFrame.h>
// #include "Converter.h"
#include <math.h>

namespace svo {

class KeyFrame;
class Frame;
class GeometricCamera;

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 12, 1> Vector12d;
typedef Eigen::Matrix<double, 15, 1> Vector15d;
typedef Eigen::Matrix<double, 12, 12> Matrix12d;
typedef Eigen::Matrix<double, 15, 15> Matrix15d;
typedef Eigen::Matrix<double, 9, 9> Matrix9d;

Eigen::Matrix3d ExpSO3(const double x, const double y, const double z);
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& w);

Eigen::Vector3d LogSO3(const Eigen::Matrix3d& R);

Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d& v);
Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d& v);
Eigen::Matrix3d RightJacobianSO3(const double x,
                                 const double y,
                                 const double z);

Eigen::Matrix3d Skew(const Eigen::Vector3d& w);
Eigen::Matrix3d InverseRightJacobianSO3(const double x,
                                        const double y,
                                        const double z);

Eigen::Matrix3d NormalizeRotation(const Eigen::Matrix3d& R);

class ImuCamPose {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ImuCamPose() {}
    ImuCamPose(KeyFrame* pKF);
    ImuCamPose(Frame* pF);
    ImuCamPose(Eigen::Matrix3d& _Rwc, Eigen::Vector3d& _twc, KeyFrame* pKF);

    void SetParam(const std::vector<Eigen::Matrix3d>& _Rcw,
                  const std::vector<Eigen::Vector3d>& _tcw,
                  const std::vector<Eigen::Matrix3d>& _Rbc,
                  const std::vector<Eigen::Vector3d>& _tbc,
                  const double& _bf);

    void Update(const double* pu);   // update in the imu reference
    void UpdateW(const double* pu);  // update in the world reference
    Eigen::Vector2d Project(const Eigen::Vector3d& Xw,
                            int cam_idx = 0) const;  // Mono
    Eigen::Vector3d ProjectStereo(const Eigen::Vector3d& Xw,
                                  int cam_idx = 0) const;  // Stereo
    bool isDepthPositive(const Eigen::Vector3d& Xw, int cam_idx = 0) const;

 public:
    // For IMU
    Eigen::Matrix3d Rwb;
    Eigen::Vector3d twb;

    // For set of cameras
    std::vector<Eigen::Matrix3d> Rcw;
    std::vector<Eigen::Vector3d> tcw;
    std::vector<Eigen::Matrix3d> Rcb, Rbc;
    std::vector<Eigen::Vector3d> tcb, tbc;
    double bf;
    std::vector<GeometricCamera*> pCamera;

    // For posegraph 4DoF
    Eigen::Matrix3d Rwb0;
    Eigen::Matrix3d DR;

    int its;
};


// Optimizable parameters are IMU pose
class VertexPose : public g2o::BaseVertex<6, ImuCamPose> {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPose() {}
    VertexPose(KeyFrame* pKF) { setEstimate(ImuCamPose(pKF)); }
    VertexPose(Frame* pF) { setEstimate(ImuCamPose(pF)); }

    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double* update_) {
        _estimate.Update(update_);
        updateCache();
    }
};

class VertexPose4DoF : public g2o::BaseVertex<4, ImuCamPose> {
    // Translation and yaw are the only optimizable variables
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPose4DoF() {}
    VertexPose4DoF(KeyFrame* pKF) { setEstimate(ImuCamPose(pKF)); }
    VertexPose4DoF(Frame* pF) { setEstimate(ImuCamPose(pF)); }
    VertexPose4DoF(Eigen::Matrix3d& _Rwc,
                   Eigen::Vector3d& _twc,
                   KeyFrame* pKF) {
        setEstimate(ImuCamPose(_Rwc, _twc, pKF));
    }

    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double* update_) {
        double update6DoF[6];
        update6DoF[0] = 0;
        update6DoF[1] = 0;
        update6DoF[2] = update_[0];
        update6DoF[3] = update_[1];
        update6DoF[4] = update_[2];
        update6DoF[5] = update_[3];
        _estimate.UpdateW(update6DoF);
        updateCache();
    }
};

// // scale vertex
// class VertexScale : public g2o::BaseVertex<1, double> {
//  public:
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//     VertexScale() { setEstimate(1.0); }
//     VertexScale(double ps) { setEstimate(ps); }

//     virtual bool read(std::istream& is) { return false; }
//     virtual bool write(std::ostream& os) const { return false; }

//     virtual void setToOriginImpl() { setEstimate(1.0); }

//     virtual void oplusImpl(const double* update_) {
//         setEstimate(estimate() * exp(*update_));
//     }
// };


class InvDepthPoint {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    InvDepthPoint() {}
    InvDepthPoint(double _rho, double _u, double _v, KeyFrame* pHostKF);

    void Update(const double* pu);

    double rho;
    double u, v;  // they are not variables, observation in the host frame

    double fx, fy, cx, cy, bf;  // from host frame

    int its;
};

// Inverse depth point (just one parameter, inverse depth at the host frame)
class VertexInvDepth : public g2o::BaseVertex<1, InvDepthPoint> {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexInvDepth() {}
    VertexInvDepth(double invDepth, double u, double v, KeyFrame* pHostKF) {
        setEstimate(InvDepthPoint(invDepth, u, v, pHostKF));
    }

    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double* update_) {
        _estimate.Update(update_);
        updateCache();
    }
};

class EdgeMono : public g2o::BaseBinaryEdge<2,
                                            Eigen::Vector2d,
                                            g2o::VertexSBAPointXYZ,
                                            VertexPose> {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeMono(int cam_idx_ = 0) : cam_idx(cam_idx_) {}

    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    void computeError() {
        const g2o::VertexSBAPointXYZ* VPoint =
            static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
        const Eigen::Vector2d obs(_measurement);
        _error = obs - VPose->estimate().Project(VPoint->estimate(), cam_idx);
    }

    virtual void linearizeOplus();

    bool isDepthPositive() {
        const g2o::VertexSBAPointXYZ* VPoint =
            static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
        return VPose->estimate().isDepthPositive(VPoint->estimate(), cam_idx);
    }

    Eigen::Matrix<double, 2, 9> GetJacobian() {
        linearizeOplus();
        Eigen::Matrix<double, 2, 9> J;
        J.block<2, 3>(0, 0) = _jacobianOplusXi;
        J.block<2, 6>(0, 3) = _jacobianOplusXj;
        return J;
    }

    Eigen::Matrix<double, 9, 9> GetHessian() {
        linearizeOplus();
        Eigen::Matrix<double, 2, 9> J;
        J.block<2, 3>(0, 0) = _jacobianOplusXi;
        J.block<2, 6>(0, 3) = _jacobianOplusXj;
        return J.transpose() * information() * J;
    }

 public:
    const int cam_idx;
};

class EdgeMonoOnlyPose
    : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeMonoOnlyPose(const cv::Mat& Xw_, int cam_idx_ = 0)
        : Xw(Converter::toVector3d(Xw_)), cam_idx(cam_idx_) {}

    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    void computeError() {
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
        const Eigen::Vector2d obs(_measurement);
        _error = obs - VPose->estimate().Project(Xw, cam_idx);
    }

    virtual void linearizeOplus();

    bool isDepthPositive() {
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
        return VPose->estimate().isDepthPositive(Xw, cam_idx);
    }

    Eigen::Matrix<double, 6, 6> GetHessian() {
        linearizeOplus();
        return _jacobianOplusXi.transpose() * information() * _jacobianOplusXi;
    }

 public:
    const Eigen::Vector3d Xw;
    const int cam_idx;
};

class EdgeStereo : public g2o::BaseBinaryEdge<3,
                                              Eigen::Vector3d,
                                              g2o::VertexSBAPointXYZ,
                                              VertexPose> {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeStereo(int cam_idx_ = 0) : cam_idx(cam_idx_) {}

    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    void computeError() {
        const g2o::VertexSBAPointXYZ* VPoint =
            static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
        const Eigen::Vector3d obs(_measurement);
        _error =
            obs - VPose->estimate().ProjectStereo(VPoint->estimate(), cam_idx);
    }

    virtual void linearizeOplus();

    Eigen::Matrix<double, 3, 9> GetJacobian() {
        linearizeOplus();
        Eigen::Matrix<double, 3, 9> J;
        J.block<3, 3>(0, 0) = _jacobianOplusXi;
        J.block<3, 6>(0, 3) = _jacobianOplusXj;
        return J;
    }

    Eigen::Matrix<double, 9, 9> GetHessian() {
        linearizeOplus();
        Eigen::Matrix<double, 3, 9> J;
        J.block<3, 3>(0, 0) = _jacobianOplusXi;
        J.block<3, 6>(0, 3) = _jacobianOplusXj;
        return J.transpose() * information() * J;
    }

 public:
    const int cam_idx;
};

class EdgeStereoOnlyPose
    : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose> {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeStereoOnlyPose(const cv::Mat& Xw_, int cam_idx_ = 0)
        : Xw(Converter::toVector3d(Xw_)), cam_idx(cam_idx_) {}

    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    void computeError() {
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
        const Eigen::Vector3d obs(_measurement);
        _error = obs - VPose->estimate().ProjectStereo(Xw, cam_idx);
    }

    virtual void linearizeOplus();

    Eigen::Matrix<double, 6, 6> GetHessian() {
        linearizeOplus();
        return _jacobianOplusXi.transpose() * information() * _jacobianOplusXi;
    }

 public:
    const Eigen::Vector3d Xw;  // 3D point coordinates
    const int cam_idx;
};

// Edge inertial whre gravity is included as optimizable variable and it is not
// supposed to be pointing in -z axis, as well as scale

// class ConstraintPoseImu {
//  public:
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//     ConstraintPoseImu(const Eigen::Matrix3d& Rwb_,
//                       const Eigen::Vector3d& twb_,
//                       const Eigen::Vector3d& vwb_,
//                       const Eigen::Vector3d& bg_,
//                       const Eigen::Vector3d& ba_,
//                       const Matrix15d& H_)
//         : Rwb(Rwb_), twb(twb_), vwb(vwb_), bg(bg_), ba(ba_), H(H_) {
//         H = (H + H) / 2;
//         Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 15, 15> > es(H);
//         Eigen::Matrix<double, 15, 1> eigs = es.eigenvalues();
//         for (int i = 0; i < 15; i++)
//             if (eigs[i] < 1e-12) eigs[i] = 0;
//         H = es.eigenvectors() * eigs.asDiagonal() *
//             es.eigenvectors().transpose();
//     }
//     ConstraintPoseImu(const cv::Mat& Rwb_,
//                       const cv::Mat& twb_,
//                       const cv::Mat& vwb_,
//                       const IMU::Bias& b,
//                       const cv::Mat& H_) {
//         Rwb = Converter::toMatrix3d(Rwb_);
//         twb = Converter::toVector3d(twb_);
//         vwb = Converter::toVector3d(vwb_);
//         bg << b.bwx, b.bwy, b.bwz;
//         ba << b.bax, b.bay, b.baz;
//         for (int i = 0; i < 15; i++)
//             for (int j = 0; j < 15; j++) H(i, j) = H_.at<float>(i, j);
//         H = (H + H) / 2;
//         Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 15, 15> > es(H);
//         Eigen::Matrix<double, 15, 1> eigs = es.eigenvalues();
//         for (int i = 0; i < 15; i++)
//             if (eigs[i] < 1e-12) eigs[i] = 0;
//         H = es.eigenvectors() * eigs.asDiagonal() *
//             es.eigenvectors().transpose();
//     }

//     Eigen::Matrix3d Rwb;
//     Eigen::Vector3d twb;
//     Eigen::Vector3d vwb;
//     Eigen::Vector3d bg;
//     Eigen::Vector3d ba;
//     Matrix15d H;
// };

class Edge4DoF
    : public g2o::BaseBinaryEdge<6, Vector6d, VertexPose4DoF, VertexPose4DoF> {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Edge4DoF(const Eigen::Matrix4d& deltaT) {
        dTij = deltaT;
        dRij = deltaT.block<3, 3>(0, 0);
        dtij = deltaT.block<3, 1>(0, 3);
    }

    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    void computeError() {
        const VertexPose4DoF* VPi =
            static_cast<const VertexPose4DoF*>(_vertices[0]);
        const VertexPose4DoF* VPj =
            static_cast<const VertexPose4DoF*>(_vertices[1]);
        _error << LogSO3(VPi->estimate().Rcw[0] *
                         VPj->estimate().Rcw[0].transpose() * dRij.transpose()),
            VPi->estimate().Rcw[0] * (-VPj->estimate().Rcw[0].transpose() *
                                      VPj->estimate().tcw[0]) +
                VPi->estimate().tcw[0] - dtij;
    }

    // virtual void linearizeOplus(); // numerical implementation

    Eigen::Matrix4d dTij;
    Eigen::Matrix3d dRij;
    Eigen::Vector3d dtij;
};

}  // namespace ORB_SLAM2

#endif  // G2OTYPES_H
