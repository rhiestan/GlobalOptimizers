// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef GLOBOPT_CORE_OPTIMIZER_HPP
#define GLOBOPT_CORE_OPTIMIZER_HPP

#include "params.hpp"
#include "result.hpp"
#include "types.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace globopt {

/// Abstract base class for all optimizers.
///
/// Usage:
/// @code
///   auto opt = globopt::OptimizerFactory<double>::create("lbfgs");
///   opt->setParam("max_iterations", 500);
///   opt->setBounds(lower, upper);   // optional
///   auto result = opt->run(objective, x0);
/// @endcode
template <typename Scalar>
class Optimizer {
public:
    using ObjectiveFn = ObjectiveFunction<Scalar>;

    virtual ~Optimizer() = default;

    virtual const char* name() const = 0;

    // -- parameters ---------------------------------------------------------

    void setParam(const std::string& name, double value) { m_params.set(name, ParamValue(value)); }
    void setParam(const std::string& name, int value) { m_params.set(name, ParamValue(static_cast<long long>(value))); }
    void setParam(const std::string& name, long long value) { m_params.set(name, ParamValue(value)); }
    void setParam(const std::string& name, std::size_t value) { m_params.set(name, ParamValue(static_cast<long long>(value))); }
    void setParam(const std::string& name, bool value) { m_params.set(name, ParamValue(value)); }
    void setParam(const std::string& name, const char* value) { m_params.set(name, ParamValue(std::string(value))); }
    void setParam(const std::string& name, const std::string& value) { m_params.set(name, ParamValue(value)); }

    ParamValue getParam(const std::string& name) const { return m_params.get(name); }
    bool hasParam(const std::string& name) const { return m_params.contains(name); }
    std::vector<ParamInfo> listParams() const { return m_params.list(); }

    // -- box constraints ----------------------------------------------------

    void setBounds(const Vector<Scalar>& lowerBounds, const Vector<Scalar>& upperBounds)
    {
        if (lowerBounds.size() != upperBounds.size()) {
            throw std::invalid_argument("globopt: lower and upper bounds must have the same size");
        }
        m_lowerBounds = lowerBounds;
        m_upperBounds = upperBounds;
        m_boundsSet = true;
    }

    void clearBounds()
    {
        m_boundsSet = false;
        m_lowerBounds.resize(0);
        m_upperBounds.resize(0);
    }

    bool boundsSet() const { return m_boundsSet; }
    const Vector<Scalar>& lowerBounds() const { return m_lowerBounds; }
    const Vector<Scalar>& upperBounds() const { return m_upperBounds; }

    // -- run ----------------------------------------------------------------

    /// Minimize the objective starting from initialPoint.
    Result<Scalar> run(const ObjectiveFn& objective, const Vector<Scalar>& initialPoint)
    {
        Result<Scalar> result;
        result.x = initialPoint;

        if (!objective) {
            result.status = Status::InvalidInput;
            result.message = "no objective function provided";
            return result;
        }

        if (initialPoint.size() == 0) {
            result.status = Status::InvalidInput;
            result.message = "initial point is empty";
            return result;
        }

        if (!initialPoint.allFinite()) {
            result.status = Status::InvalidInput;
            result.message = "non-finite initial value(s)";
            return result;
        }

        if (m_boundsSet && m_lowerBounds.size() != initialPoint.size()) {
            result.status = Status::InvalidInput;
            result.message = "bounds size does not match the size of the initial point";
            return result;
        }

        return doOptimize(objective, initialPoint);
    }

protected:
    template <typename T>
    void registerParam(const std::string& name, T* target, const std::string& description)
    {
        m_params.add(name, target, description);
    }

    virtual Result<Scalar> doOptimize(const ObjectiveFn& objective, const Vector<Scalar>& initialPoint) = 0;

    bool m_boundsSet = false;
    Vector<Scalar> m_lowerBounds;
    Vector<Scalar> m_upperBounds;

private:
    detail::ParamRegistry m_params;
};

} // namespace globopt

#endif
