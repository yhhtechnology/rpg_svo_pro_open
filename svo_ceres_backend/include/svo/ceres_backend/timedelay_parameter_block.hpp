
/**
 * @file timedelay_parameter_block.hpp
 * @brief Header file for the TimeDelayParameterBlock class.
 * @author Stefan Leutenegger
 */

#pragma once

#pragma diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
// Eigen 3.2.7 uses std::binder1st and std::binder2nd which are deprecated since
// c++11
// Fix is in 3.3 devel (http://eigen.tuxfamily.org/bz/show_bug.cgi?id=872).
#include <Eigen/Core>
#pragma diagnostic pop

#include "svo/ceres_backend/estimator_types.hpp"
#include "svo/ceres_backend/parameter_block.hpp"

namespace svo {
namespace ceres_backend {

/// \brief Wraps the parameter block for a speed / IMU biases estimate
class TimeDelayParameterBlock : public ParameterBlock {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef double* TimeDelay;
    typedef TimeDelay estimate_t;

    static constexpr size_t c_dimension = 1;
    static constexpr size_t c_minimal_dimension = 1;

    /// \brief Default constructor (assumes not fixed).
    TimeDelayParameterBlock() : ParameterBlock::ParameterBlock() {
        setFixed(false);
    }
    /// \brief Constructor with estimate and id.
    /// @param[in] speedAndBias The speed and bias estimate.
    /// @param[in] id The (unique) ID of this block.
    TimeDelayParameterBlock(const TimeDelay& time_delay, uint64_t id) {
        setEstimate(time_delay);
        setId(id);
        setFixed(false);
    }
    virtual ~TimeDelayParameterBlock() {}

    // ---------------------------------------------------------------------------
    // Setters
    virtual void setEstimate(const TimeDelay& time_delay) {
        estimate_ = time_delay;
    }

    // ---------------------------------------------------------------------------
    // Getters

    virtual const TimeDelay& estimate() const { return estimate_; }

    virtual double* parameters() { return estimate_; }

    virtual const double* parameters() const { return estimate_; }

    virtual size_t dimension() const { return c_dimension; }

    virtual size_t minimalDimension() const { return c_minimal_dimension; }

    // minimal internal parameterization
    // x0_plus_Delta=Delta_Chi[+]x0
    /// \brief Generalization of the addition operation,
    ///        x_plus_delta = Plus(x, delta)
    ///        with the condition that Plus(x, 0) = x.
    /// @param[in] x0 Variable.
    /// @param[in] Delta_Chi Perturbation.
    /// @param[out] x0_plus_Delta Perturbed x.
    virtual void plus(const double* x0,
                      const double* Delta_Chi,
                      double* x0_plus_Delta) const {
        double x0_(*x0);
        double Delta_Chi_(*Delta_Chi);
        (*x0_plus_Delta) = x0_ + Delta_Chi_;
    }

    /// \brief The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
    /// @param[in] x0 Variable.
    /// @param[out] jacobian The Jacobian.
    virtual void plusJacobian(const double* /*unused: x*/,
                              double* jacobian) const {
        Eigen::Map<Eigen::Matrix<double, 1, 1, Eigen::RowMajor> > identity(jacobian);
        identity.setIdentity();
    }

    // Delta_Chi=x0_plus_Delta[-]x0
    /// \brief Computes the minimal difference between a variable x and a
    ///        perturbed variable x_plus_delta
    /// @param[in] x0 Variable.
    /// @param[in] x0_plus_Delta Perturbed variable.
    /// @param[out] Delta_Chi Minimal difference.
    /// \return True on success.
    virtual void minus(const double* x0,
                       const double* x0_plus_Delta,
                       double* Delta_Chi) const {
        double x0_(*x0);
        double x0_plus_Delta_(*x0_plus_Delta);
        // double Delta_Chi_(Delta_Chi);
        (*Delta_Chi) = x0_plus_Delta_ - x0_;
    }

    /// \brief Computes the Jacobian from minimal space to naively
    ///        overparameterised space as used by ceres.
    /// @param[out] jacobian the Jacobian (dimension minDim x dim).
    /// \return True on success.
    virtual void liftJacobian(const double* /*unused: x*/,
                              double* jacobian) const {
        Eigen::Map<Eigen::Matrix<double, 1, 1, Eigen::RowMajor> > identity(jacobian);
        identity.setIdentity();
    }

    /// @brief Return parameter block type as string
    virtual std::string typeInfo() const {
        return "TimeDelayParameterBlock";
    }

 private:
    TimeDelay estimate_;
};

}  // namespace ceres_backend
}  // namespace svo
