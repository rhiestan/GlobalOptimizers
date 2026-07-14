// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef GLOBOPT_CORE_TYPES_HPP
#define GLOBOPT_CORE_TYPES_HPP

#include <Eigen/Dense>

#include <cstddef>
#include <functional>
#include <limits>
#include <type_traits>

namespace globopt {

using Index = Eigen::Index;

template <typename Scalar>
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <typename Scalar>
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

using VectorInt = Eigen::Matrix<int, Eigen::Dynamic, 1>;

/// Objective function: returns f(x); if grad_out is non-null it must be
/// filled with the gradient of f at x.
template <typename Scalar>
using ObjectiveFunction = std::function<Scalar(const Vector<Scalar>& x, Vector<Scalar>* grad_out)>;

namespace detail {

template <typename Scalar>
constexpr Scalar eps()
{
    return std::numeric_limits<Scalar>::epsilon();
}

template <typename Scalar>
constexpr Scalar inf()
{
    return std::numeric_limits<Scalar>::infinity();
}

// guard against division by ~zero in relative-change computations
template <typename Scalar>
constexpr Scalar smallNumber()
{
    return std::is_same<Scalar, float>::value ? Scalar(1e-05) : Scalar(1e-08);
}

} // namespace detail

} // namespace globopt

#endif
