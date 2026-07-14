// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef GLOBOPT_CORE_NUMERICAL_GRADIENT_HPP
#define GLOBOPT_CORE_NUMERICAL_GRADIENT_HPP

#include "types.hpp"

#include <cmath>
#include <utility>

namespace globopt {

/// Wraps a gradient-free function f(x) -> Scalar into an ObjectiveFunction
/// whose gradient is computed by central finite differences. Note that each
/// gradient evaluation costs 2n extra calls to f.
template <typename Scalar, typename F>
ObjectiveFunction<Scalar> withNumericalGradient(F f)
{
    return [f = std::move(f)](const Vector<Scalar>& x, Vector<Scalar>* gradOut) -> Scalar {
        const Scalar fx = f(x);

        if (gradOut) {
            const Index n = x.size();
            gradOut->resize(n);

            const Scalar hBase = std::cbrt(detail::eps<Scalar>());
            Vector<Scalar> xh = x;

            for (Index i = 0; i < n; ++i) {
                const Scalar h = hBase * std::max(Scalar(1), std::abs(x(i)));

                xh(i) = x(i) + h;
                const Scalar fPlus = f(xh);
                xh(i) = x(i) - h;
                const Scalar fMinus = f(xh);
                xh(i) = x(i);

                (*gradOut)(i) = (fPlus - fMinus) / (2 * h);
            }
        }

        return fx;
    };
}

} // namespace globopt

#endif
