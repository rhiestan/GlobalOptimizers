// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Scalable (any-dimension) box-constrained test problems ported from
// DIRECTGOLib v2.0 - DIRECT Global Optimization test problems Library
// (https://github.com/blockchain-group/DIRECTGOLib, MIT License,
// Copyright (c) 2022 Blockchain Technologies Group; MATLAB coding by
// Linas Stripinis and Remigijus Paulavicius).
//
// Reference:
//   L. Stripinis, R. Paulavicius, "DIRECTGOLib - DIRECT Global Optimization
//   test problems Library", v2.0, 2024.
//
// Where go_benchmark.hpp provides 202 problems of mostly fixed (and mostly
// two-dimensional) size, these are defined for any number of variables, which
// is what makes it possible to test how an optimizer degrades with dimension.
// Bounds, global minima and formulas follow the MATLAB sources exactly,
// quirks included - see the note on Whitley below.

#ifndef GLOBOPT_BENCHMARKS_DIRECTGO_BENCHMARK_HPP
#define GLOBOPT_BENCHMARKS_DIRECTGO_BENCHMARK_HPP

#include "go_benchmark.hpp"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace globopt {
namespace benchmarks {

/// A test problem defined for any number of variables. Call at(n) to obtain
/// the ordinary fixed-dimension Problem used by the rest of the harness.
struct ScalableProblem {
    std::string name;
    double lower = 0.0; ///< lower bound, the same for every coordinate
    double upper = 0.0; ///< upper bound, the same for every coordinate
    std::function<double(Index)> fglob;              ///< global minimum at dimension n
    std::function<Vec(Index)> optimum;               ///< a global minimizer (may be empty)
    std::function<double(const Vec&)> objective;

    Problem at(Index n) const
    {
        if (n < 1) {
            throw std::invalid_argument("globopt benchmarks: dimension must be at least 1");
        }
        Problem p;
        p.name = name;
        p.lower = Vec::Constant(n, lower);
        p.upper = Vec::Constant(n, upper);
        p.fglob = fglob(n);
        if (optimum) {
            p.globalOptima.push_back(optimum(n));
        }
        p.objective = objective;
        return p;
    }
};

/// The scalable problems, in the order they appear in DIRECTGOLib.
inline const std::vector<ScalableProblem>& allScalableProblems()
{
    using std::abs; using std::sin; using std::cos; using std::exp;
    using std::sqrt; using std::log; using std::pow; using std::floor;

    static const std::vector<ScalableProblem> problems = [] {
        std::vector<ScalableProblem> ps;

        auto add = [&ps](std::string name, double lower, double upper,
                         std::function<double(Index)> fglob,
                         std::function<Vec(Index)> optimum,
                         std::function<double(const Vec&)> f) {
            ScalableProblem p;
            p.name = std::move(name);
            p.lower = lower;
            p.upper = upper;
            p.fglob = std::move(fglob);
            p.optimum = std::move(optimum);
            p.objective = std::move(f);
            ps.push_back(std::move(p));
        };

        const auto zero = [](Index) { return 0.0; };
        const auto atOrigin = [](Index n) { return Vec::Zero(n).eval(); };
        const auto atOnes = [](Index n) { return Vec::Ones(n).eval(); };

        add("Ackley", -15.0, 30.0, zero, atOrigin,
            [](const Vec& x) {
                const double n = static_cast<double>(x.size());
                return -20.0 * exp(-0.2 * sqrt(x.squaredNorm() / n))
                    - exp((2.0 * detail::pi * x.array()).cos().sum() / n)
                    + 20.0 + exp(1.0);
            });

        add("AlpineN1", -10.0, 10.0, zero, atOrigin,
            [](const Vec& x) {
                return (x.array() * x.array().sin() + 0.1 * x.array()).abs().sum();
            });

        add("ChungR", -100.0, 100.0, zero, atOrigin,
            [](const Vec& x) {
                const double s = x.squaredNorm();
                return s * s;
            });

        add("Cigar", -100.0, 100.0, zero, atOrigin,
            [](const Vec& x) {
                return x(0) * x(0) + 1e6 * (x.squaredNorm() - x(0) * x(0));
            });

        add("Csendes", -1.0, 1.0, zero, atOrigin,
            [](const Vec& x) {
                // undefined at any zero coordinate; the source returns 0 there,
                // which is also the global minimum
                if ((x.array() == 0.0).any()) {
                    return 0.0;
                }
                return (x.array().pow(6) * (2.0 + (1.0 / x.array()).sin())).sum();
            });

        add("DixonAndPrice", -10.0, 10.0, zero, nullptr,
            [](const Vec& x) {
                double s = (x(0) - 1.0) * (x(0) - 1.0);
                for (Index j = 1; j < x.size(); ++j) {
                    const double t = 2.0 * x(j) * x(j) - x(j - 1);
                    s += static_cast<double>(j + 1) * t * t;
                }
                return s;
            });

        add("Exponential", -1.0, 1.0, [](Index) { return -1.0; }, atOrigin,
            [](const Vec& x) { return -exp(-0.5 * x.squaredNorm()); });

        add("Griewank", -600.0, 600.0, zero, atOrigin,
            [](const Vec& x) {
                double product = 1.0;
                for (Index i = 0; i < x.size(); ++i) {
                    product *= cos(x(i) / sqrt(static_cast<double>(i + 1)));
                }
                return x.squaredNorm() / 4000.0 - product + 1.0;
            });

        add("Levy", -5.0, 5.0, zero, atOnes,
            [](const Vec& x) {
                const Index n = x.size();
                Vec z(n);
                for (Index i = 0; i < n; ++i) {
                    z(i) = 1.0 + (x(i) - 1.0) / 4.0;
                }
                double s = pow(sin(detail::pi * z(0)), 2);
                for (Index i = 0; i + 1 < n; ++i) {
                    s += pow(z(i) - 1.0, 2) * (1.0 + 10.0 * pow(sin(detail::pi * z(i) + 1.0), 2));
                }
                return s + pow(z(n - 1) - 1.0, 2) * (1.0 + pow(sin(2.0 * detail::pi * z(n - 1)), 2));
            });

        add("Qing", -500.0, 500.0, zero, nullptr,
            [](const Vec& x) {
                double s = 0.0;
                for (Index i = 0; i < x.size(); ++i) {
                    const double t = x(i) * x(i) - static_cast<double>(i + 1);
                    s += t * t;
                }
                return s;
            });

        add("Quintic", -10.0, 10.0, zero, nullptr,
            [](const Vec& x) {
                double s = 0.0;
                for (Index i = 0; i < x.size(); ++i) {
                    const double v = x(i);
                    s += abs(pow(v, 5) - 3.0 * pow(v, 4) + 4.0 * pow(v, 3)
                             + 2.0 * v * v - 10.0 * v - 4.0);
                }
                return s;
            });

        add("Rastrigin", -5.12, 5.12, zero, atOrigin,
            [](const Vec& x) {
                const double n = static_cast<double>(x.size());
                return 10.0 * n
                    + (x.array().square() - 10.0 * (2.0 * detail::pi * x.array()).cos()).sum();
            });

        add("Rosenbrock", -5.0, 10.0, zero, atOnes,
            [](const Vec& x) {
                double s = 0.0;
                for (Index j = 0; j + 1 < x.size(); ++j) {
                    s += 100.0 * pow(x(j) * x(j) - x(j + 1), 2) + pow(x(j) - 1.0, 2);
                }
                return s;
            });

        add("RotatedHyperEllipsoid", -65.536, 65.536, zero, atOrigin,
            [](const Vec& x) {
                double s = 0.0, partial = 0.0;
                for (Index i = 0; i < x.size(); ++i) {
                    partial += x(i) * x(i);
                    s += partial;
                }
                return s;
            });

        add("Salomon", -100.0, 100.0, zero, atOrigin,
            [](const Vec& x) {
                const double norm = x.norm();
                return 1.0 - cos(2.0 * detail::pi * norm) + 0.1 * norm;
            });

        add("Sargan", -100.0, 100.0, zero, atOrigin,
            [](const Vec& x) {
                const double n = static_cast<double>(x.size());
                const double total = x.sum();
                double s = 0.0;
                for (Index i = 0; i < x.size(); ++i) {
                    const double cross = x(i) * (total - x(i));
                    s += n * (x(i) * x(i) + 0.4 * cross);
                }
                return s;
            });

        add("Schwefel", -500.0, 500.0, zero, nullptr,
            [](const Vec& x) {
                double s = 0.0;
                for (Index i = 0; i < x.size(); ++i) {
                    s += -x(i) * sin(sqrt(abs(x(i))));
                }
                return 418.9828872724336 * static_cast<double>(x.size()) + s;
            });

        add("Schwefel220", -100.0, 100.0, zero, atOrigin,
            [](const Vec& x) { return x.cwiseAbs().sum(); });

        add("Schwefel221", -100.0, 100.0, zero, atOrigin,
            [](const Vec& x) { return x.cwiseAbs().maxCoeff(); });

        add("Schwefel222", -100.0, 100.0, zero, atOrigin,
            [](const Vec& x) {
                return x.cwiseAbs().sum() + x.cwiseAbs().prod();
            });

        add("Sphere", -5.0, 5.0, zero, atOrigin,
            [](const Vec& x) { return x.squaredNorm(); });

        add("Step2", -100.0, 100.0, zero, atOrigin,
            [](const Vec& x) {
                return (x.array() + 0.5).floor().square().sum();
            });

        add("StyblinskiTang", -5.0, 5.0,
            [](Index n) { return -39.166165703771426 * static_cast<double>(n); }, nullptr,
            [](const Vec& x) {
                return (x.array().pow(4) - 16.0 * x.array().square() + 5.0 * x.array()).sum() / 2.0;
            });

        add("SumOfPowers", -1.0, 1.0, zero, atOrigin,
            [](const Vec& x) {
                double s = 0.0;
                for (Index i = 0; i < x.size(); ++i) {
                    s += pow(abs(x(i)), static_cast<double>(i + 2));
                }
                return s;
            });

        add("SumSquare", -10.0, 10.0, zero, atOrigin,
            [](const Vec& x) {
                double s = 0.0;
                for (Index i = 0; i < x.size(); ++i) {
                    s += static_cast<double>(i + 1) * x(i) * x(i);
                }
                return s;
            });

        add("Trigonometric01", 0.0, detail::pi, zero, atOrigin,
            [](const Vec& x) {
                const double n = static_cast<double>(x.size());
                const double cosSum = x.array().cos().sum();
                double s = 0.0;
                for (Index i = 0; i < x.size(); ++i) {
                    const double t = n - cosSum
                        + static_cast<double>(i + 1) * (1.0 - cos(x(i)) - sin(x(i)));
                    s += t * t;
                }
                return s;
            });

        add("Vincent", 0.25, 10.0,
            [](Index n) { return -static_cast<double>(n); }, nullptr,
            [](const Vec& x) {
                double s = 0.0;
                for (Index i = 0; i < x.size(); ++i) {
                    s += sin(10.0 * log(x(i)));
                }
                return -s;
            });

        // NOTE: DIRECTGOLib evaluates the cosine of 100*(xi^2 - xj)^2 only and
        // adds (1 - xj)^2 outside it, where the canonical Whitley function
        // takes the cosine of the whole term. Ported as the source has it.
        add("Whitley", -10.24, 10.24, zero, atOnes,
            [](const Vec& x) {
                double y = 0.0;
                for (Index i = 0; i < x.size(); ++i) {
                    for (Index j = 0; j < x.size(); ++j) {
                        const double a = 100.0 * pow(x(i) * x(i) - x(j), 2);
                        const double b = pow(1.0 - x(j), 2);
                        y += (a + b) * (a + b) / 4000.0 - cos(a) + b + 1.0;
                    }
                }
                return y;
            });

        add("XinSheYang02", -2.0 * detail::pi, 2.0 * detail::pi, zero, atOrigin,
            [](const Vec& x) {
                return x.cwiseAbs().sum() * exp(-x.array().square().sin().sum());
            });

        add("Zakharov", -5.0, 10.0, zero, atOrigin,
            [](const Vec& x) {
                double weighted = 0.0;
                for (Index i = 0; i < x.size(); ++i) {
                    weighted += static_cast<double>(i + 1) * x(i);
                }
                const double half = 0.5 * weighted;
                return x.squaredNorm() + pow(half, 2) + pow(half, 4);
            });

        return ps;
    }();

    return problems;
}

/// Every scalable problem materialized at dimension n.
inline std::vector<Problem> scalableProblemsAt(const Index n)
{
    std::vector<Problem> ps;
    ps.reserve(allScalableProblems().size());
    for (const ScalableProblem& p : allScalableProblems()) {
        ps.push_back(p.at(n));
    }
    return ps;
}

/// Look up a single scalable problem by name.
inline const ScalableProblem& scalableProblem(const std::string& name)
{
    for (const ScalableProblem& p : allScalableProblems()) {
        if (p.name == name) {
            return p;
        }
    }
    throw std::invalid_argument("globopt benchmarks: unknown scalable problem '" + name + "'");
}

} // namespace benchmarks
} // namespace globopt

#endif
