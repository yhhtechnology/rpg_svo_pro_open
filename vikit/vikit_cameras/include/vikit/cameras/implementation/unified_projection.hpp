#include <iostream>
#include <glog/logging.h>
#include <vikit/cameras/unified_projection.h>

namespace vk {
namespace cameras {

template<typename Distortion>
UnifiedProjection<Distortion>::UnifiedProjection(
    double xi, double fx, double fy, double cx, double cy, distortion_t distortion)
  : xi_(xi), fx_(fx), fy_(fy), fx_inv_(1.0/fx_), fy_inv_(1.0/fy_)
  , cx_(cx), cy_(cy), distortion_(distortion) {}

template<typename Distortion>
UnifiedProjection<Distortion>::UnifiedProjection(
    const double &xi,
    const Eigen::VectorXd& intrinsics, distortion_t distortion)
  : xi_(xi), distortion_(distortion) {
  CHECK(intrinsics.size() == 4);
  fx_ = intrinsics(0);
  fy_ = intrinsics(1);
  cx_ = intrinsics(2);
  cy_ = intrinsics(3);
  fx_inv_ = 1.0/fx_;
  fy_inv_ = 1.0/fy_;
}

template<typename Distortion>
bool UnifiedProjection<Distortion>::isUndistortedKeypointValid(
    const double &rho2_d) const {
  double one_over_xixi_m_1 = 1.0 / (xi_ * xi_ - 1);
  return xi_ <= 1.0 || rho2_d <= one_over_xixi_m_1;
}

template<typename Distortion>
bool UnifiedProjection<Distortion>::backProject3(
    const Eigen::Ref<const Eigen::Vector2d>& keypoint,
    Eigen::Vector3d* out_point_3d) const
{
  double x = (keypoint[0]-cx_)*fx_inv_;
  double y = (keypoint[1]-cy_)*fy_inv_;
  // Re-distort
  distortion_.undistort(x, y);
  double rho2_d = x * x + y * y;
  // if (!isUndistortedKeypointValid(rho2_d)) {
  //   printf("UndistortedKeypoint is not Valid  !!!!!!!! \n ");
  //   // return false;
  // }
  (*out_point_3d)[0] = x;
  (*out_point_3d)[1] = y;
  (*out_point_3d)[2] = 1.0 - xi_ * (rho2_d + 1.0) / (xi_ + sqrt(1.0 + (1.0 - xi_ * xi_) * rho2_d));
  // should normalize ??
  out_point_3d->normalize();
  return true;
}

template<typename Distortion>
void UnifiedProjection<Distortion>::project3(
    const Eigen::Ref<const Eigen::Vector3d>& point_3d,
    Eigen::Vector2d* out_keypoint,
    Eigen::Matrix<double, 2, 3>* out_jacobian_point) const {

    double norm = point_3d.norm();
    // {
    //   // Check if point will lead to a valid projection
    //   double _fov_parameter = (xi_ <= 1.0) ? xi_ : 1 / xi_;
    //   if (point_3d[2] <= -(_fov_parameter * norm)) {
    //     std::cout
    //       << "point_3d[2] = " << point_3d[2]
    //       << ";  xi_ = " << xi_
    //       << ";  _fov_parameter = " << _fov_parameter
    //       <<"\n";
    //     printf("UnifiedProjection<Distortion>::project3 FAILE !!!!!!!\n");
    //     // return;
    //   }
    // }
    const double z_inv = 1.0 / (point_3d(2) + xi_ * norm);
    const Eigen::Vector2d uv = point_3d.head<2>() * z_inv;
    const Eigen::Vector2d distort_uv = distortion_.distort(uv);
    (*out_keypoint)[0] = fx_ * distort_uv(0) + cx_;
    (*out_keypoint)[1] = fy_ * distort_uv(1) + cy_;
    // if((*out_keypoint)[0] < 0 || (*out_keypoint)[1] < 0) {
    //   printf("(*out_keypoint)[0] < 0 || (*out_keypoint)[1] < 0 !!!!!!!\n");
    //   // return;
    // }
    // need Check if keypoint lies on the sensor
    // isValid(*out_keypoint);

    if(out_jacobian_point) {
      out_jacobian_point->setZero();
      Eigen::Matrix<double, 2, 3> duv_dxyz;
      double inv_denom = z_inv;
      double dudx = inv_denom * (norm * point_3d(2) + xi_ * (point_3d(1) * point_3d(1) + point_3d(2) * point_3d(2)));
      double dvdx = -inv_denom * xi_ * point_3d(0) * point_3d(1);
      double dudy = dvdx;
      double dvdy = inv_denom * (norm * point_3d(2) + xi_ * (point_3d(0) * point_3d(0) + point_3d(2) * point_3d(2)));
      inv_denom = inv_denom * (-xi_ * point_3d(2) - norm); // reuse variable
      double dudz = point_3d(0) * inv_denom;
      double dvdz = point_3d(1) * inv_denom;
      duv_dxyz(0, 0) = dudx;
      duv_dxyz(0, 1) = dudy;
      duv_dxyz(0, 2) = dudz;
      duv_dxyz(1, 0) = dvdx;
      duv_dxyz(1, 1) = dvdy;
      duv_dxyz(1, 2) = dvdz;
      const Eigen::DiagonalMatrix<double, 2> focal_matrix(fx_, fy_);
      (*out_jacobian_point) = focal_matrix * distortion_.jacobian(uv) * duv_dxyz;
    }
}

template<typename Distortion>
double UnifiedProjection<Distortion>::errorMultiplier() const {
  return std::abs(fx_);
}

template<typename Distortion>
double UnifiedProjection<Distortion>::getAngleError(double img_err) const {
  return std::atan(img_err/(2.0*fx_)) + std::atan(img_err/(2.0*fy_));
}

template<typename Distortion>
UnifiedProjection<Distortion>
UnifiedProjection<Distortion>::createTestProjection(
    const size_t image_width, const size_t image_height) {
  return UnifiedProjection(
      image_width / 2, image_width / 2, image_width / 2, image_height / 2,
      Distortion::createTestDistortion());
}

template<typename Distortion>
template<typename T>
const T* UnifiedProjection<Distortion>::distortion() const
{
  return dynamic_cast<const T*>(&distortion_);
}

template<typename Distortion>
void UnifiedProjection<Distortion>::print(std::ostream& out) const
{
  out << "  Projection = Mei" << std::endl;
  out << "  xi_ = (" << xi_ << ")" << std::endl;
  out << "  Focal length = (" << fx_ << ", " << fy_ << ")" << std::endl;
  out << "  Principal point = (" << cx_ << ", " << cy_ << ")" << std::endl;
  distortion_.print(out);
}

template<typename Distortion>
Eigen::VectorXd UnifiedProjection<Distortion>::getIntrinsicParameters() const
{
  Eigen::VectorXd intrinsics(4);
  intrinsics(0) = fx_;
  intrinsics(1) = fy_;
  intrinsics(2) = cx_;
  intrinsics(3) = cy_;
  return intrinsics;
}

template<typename Distortion>
Eigen::VectorXd UnifiedProjection<Distortion>::getDistortionParameters() const
{
  return distortion_.getDistortionParameters();
}


} // namespace cameras
} // namespace vk
