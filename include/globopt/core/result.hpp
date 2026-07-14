// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef GLOBOPT_CORE_RESULT_HPP
#define GLOBOPT_CORE_RESULT_HPP

#include "types.hpp"

#include <cstddef>
#include <string>

namespace globopt {

enum class Status {
    NotRun,
    Success,                    ///< a convergence criterion was satisfied
    MaxIterationsReached,
    MaxFunctionEvaluationsReached,
    Stalled,                    ///< solution stopped changing before reaching the requested tolerance
    InvalidInput
};

inline const char* toString(Status status)
{
    switch (status) {
        case Status::NotRun:                        return "not run";
        case Status::Success:                       return "success";
        case Status::MaxIterationsReached:          return "maximum number of iterations reached";
        case Status::MaxFunctionEvaluationsReached: return "maximum number of function evaluations reached";
        case Status::Stalled:                       return "stalled";
        case Status::InvalidInput:                  return "invalid input";
    }
    return "unknown";
}

template <typename Scalar>
struct Result {
    Vector<Scalar> x;                        ///< best point found
    Scalar fval = detail::inf<Scalar>();     ///< objective value at x
    Scalar gradientNorm = detail::inf<Scalar>();
    std::size_t iterations = 0;
    std::size_t functionEvaluations = 0;
    Status status = Status::NotRun;
    std::string message;

    bool success() const { return status == Status::Success; }
};

} // namespace globopt

#endif
