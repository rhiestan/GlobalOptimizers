// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <globopt/globopt.hpp>

#include <iostream>

int main()
{
    using Vec = globopt::Vector<double>;

    // Rosenbrock function with analytic gradient
    auto rosenbrock = [](const Vec& x, Vec* grad) -> double {
        const double a = 1.0, b = 100.0;

        if (grad) {
            grad->resize(2);
            (*grad)(0) = -2.0 * (a - x(0)) - 4.0 * b * x(0) * (x(1) - x(0) * x(0));
            (*grad)(1) = 2.0 * b * (x(1) - x(0) * x(0));
        }

        return (a - x(0)) * (a - x(0)) + b * (x(1) - x(0) * x(0)) * (x(1) - x(0) * x(0));
    };

    // 1. create the optimizer through the factory
    auto optimizer = globopt::OptimizerFactory<double>::create("L-BFGS");

    // 2. set parameters
    optimizer->setParam("max_iterations", 2000);
    optimizer->setParam("gradient_tolerance", 1e-10);
    optimizer->setParam("memory", 10);

    // 3. run
    Vec x0(2);
    x0 << -1.2, 1.0;

    const auto result = optimizer->run(rosenbrock, x0);

    std::cout << "optimizer:            " << optimizer->name() << "\n"
              << "status:               " << toString(result.status) << "\n"
              << "solution:             [" << result.x.transpose() << "]\n"
              << "objective value:      " << result.fval << "\n"
              << "gradient norm:        " << result.gradientNorm << "\n"
              << "iterations:           " << result.iterations << "\n"
              << "function evaluations: " << result.functionEvaluations << "\n";

    return result.success() ? 0 : 1;
}
