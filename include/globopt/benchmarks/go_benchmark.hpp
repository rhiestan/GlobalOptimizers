// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Global-optimization benchmark problems, ported from go_benchmark.py
// (Copyright 2013 Andrea Gavana), the test suite accompanying AMPGO.
// Bounds, objective formulas and reference optima follow the Python source
// exactly, including its occasional deviations from the literature.

#ifndef GLOBOPT_BENCHMARKS_GO_BENCHMARK_HPP
#define GLOBOPT_BENCHMARKS_GO_BENCHMARK_HPP

#include "../core/types.hpp"

#include <cmath>
#include <functional>
#include <initializer_list>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace globopt {
namespace benchmarks {

using Vec = Vector<double>;

/// A single benchmark problem: box bounds, the known global minimum value
/// (fglob) and minimizer(s) as stated in go_benchmark.py, and the objective.
struct Problem {
    std::string name;
    Vec lower;
    Vec upper;
    double fglob = 0.0;
    std::vector<Vec> globalOptima;
    std::function<double(const Vec&)> objective;

    Index dimensions() const { return lower.size(); }

    /// Uniform random start point within the bounds (the Python suite's
    /// `generator`).
    template <typename Rng>
    Vec randomStart(Rng& rng) const
    {
        Vec x(lower.size());
        for (Index i = 0; i < lower.size(); ++i) {
            std::uniform_real_distribution<double> u(lower(i), upper(i));
            x(i) = u(rng);
        }
        return x;
    }
};

namespace detail {

inline Vec vec(std::initializer_list<double> values)
{
    Vec v(static_cast<Index>(values.size()));
    Index i = 0;
    for (const double value : values) {
        v(i++) = value;
    }
    return v;
}

inline Vec constant(Index n, double value)
{
    return Vec::Constant(n, value);
}

constexpr double pi = 3.14159265358979323846;

} // namespace detail

/// The full list of ported benchmark problems.
inline const std::vector<Problem>& allProblems()
{
    using detail::vec;
    using detail::constant;
    using detail::pi;
    using std::abs; using std::sin; using std::cos; using std::exp;
    using std::sqrt; using std::log; using std::pow; using std::tan;

    static const std::vector<Problem> problems = [] {
        std::vector<Problem> ps;

        auto add = [&ps](std::string name, Vec lower, Vec upper, double fglob,
                         std::vector<Vec> optima, std::function<double(const Vec&)> f) {
            Problem p;
            p.name = std::move(name);
            p.lower = std::move(lower);
            p.upper = std::move(upper);
            p.fglob = fglob;
            p.globalOptima = std::move(optima);
            p.objective = std::move(f);
            ps.push_back(std::move(p));
        };

        add("Ackley", constant(2, -30), constant(2, 30), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                const double n = static_cast<double>(x.size());
                return -20.0 * exp(-0.2 * sqrt(x.squaredNorm() / n))
                       - exp((2.0 * pi * x.array()).cos().sum() / n) + 20.0 + exp(1.0);
            });

        add("Adjiman", vec({-1, -1}), vec({2, 1}), -2.02180678, {vec({2.0, 0.10578})},
            [](const Vec& x) {
                return cos(x(0)) * sin(x(1)) - x(0) / (x(1) * x(1) + 1.0);
            });

        add("Alpine01", constant(2, -10), constant(2, 10), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                return (x.array() * x.array().sin() + 0.1 * x.array()).abs().sum();
            });

        add("Alpine02", constant(2, 0), constant(2, 10), -6.12950,
            {vec({7.91705268, 4.81584232})},
            [](const Vec& x) {
                return (x.array().sqrt() * x.array().sin()).prod();
            });

        add("BartelsConn", constant(2, -5), constant(2, 5), 1.0, {constant(2, 0)},
            [](const Vec& x) {
                return abs(x(0) * x(0) + x(1) * x(1) + x(0) * x(1)) + abs(sin(x(0))) + abs(cos(x(1)));
            });

        add("Beale", constant(2, -4.5), constant(2, 4.5), 0.0, {vec({3.0, 0.5})},
            [](const Vec& x) {
                const double a = 1.5 - x(0) + x(0) * x(1);
                const double b = 2.25 - x(0) + x(0) * x(1) * x(1);
                const double c = 2.625 - x(0) + x(0) * x(1) * x(1) * x(1);
                return a * a + b * b + c * c;
            });

        add("Bird", constant(2, -2 * pi), constant(2, 2 * pi), -106.7645367198034,
            {vec({4.701055751981055, 3.152946019601391}),
             vec({-1.582142172055011, -3.130246799635430})},
            [](const Vec& x) {
                const double s = 1.0 - sin(x(0));
                const double c = 1.0 - cos(x(1));
                return sin(x(0)) * exp(c * c) + cos(x(1)) * exp(s * s)
                       + (x(0) - x(1)) * (x(0) - x(1));
            });

        add("Bohachevsky", constant(2, -15), constant(2, 15), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                double y = 0;
                for (Index i = 0; i + 1 < x.size(); ++i) {
                    y += x(i) * x(i) + 2.0 * x(i + 1) * x(i + 1)
                         - 0.3 * cos(3.0 * pi * x(i)) - 0.4 * cos(4.0 * pi * x(i + 1)) + 0.7;
                }
                return y;
            });

        add("Branin01", vec({-5, 0}), vec({10, 15}), 0.39788735772973816,
            {vec({-pi, 12.275}), vec({pi, 2.275}), vec({9.42478, 2.475})},
            [](const Vec& x) {
                const double t = x(1) - (5.1 / (4.0 * pi * pi)) * x(0) * x(0) + 5.0 * x(0) / pi - 6.0;
                return t * t + 10.0 * (1.0 - 1.0 / (8.0 * pi)) * cos(x(0)) + 10.0;
            });

        add("Branin02", constant(2, -5), constant(2, 15), 5.559037, {vec({-3.2, 12.53})},
            [](const Vec& x) {
                const double t = x(1) - (5.1 / (4.0 * pi * pi)) * x(0) * x(0) + 5.0 * x(0) / pi - 6.0;
                return t * t + 10.0 * (1.0 - 1.0 / (8.0 * pi)) * cos(x(0)) * cos(x(1))
                       + log(x(0) * x(0) + x(1) * x(1) + 1.0) + 10.0;
            });

        add("Brent", constant(2, -10), constant(2, 10), 0.0, {constant(2, -10)},
            [](const Vec& x) {
                return (x(0) + 10.0) * (x(0) + 10.0) + (x(1) + 10.0) * (x(1) + 10.0)
                       + exp(-x(0) * x(0) - x(1) * x(1));
            });

        add("Brown", constant(2, -1), constant(2, 4), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                double y = 0;
                for (Index i = 0; i + 1 < x.size(); ++i) {
                    const double a = x(i) * x(i);
                    const double b = x(i + 1) * x(i + 1);
                    y += pow(a, b + 1.0) + pow(b, a + 1.0);
                }
                return y;
            });

        add("Bukin02", vec({-15, -3}), vec({-5, 3}), 0.0, {vec({-10.0, 0.0})},
            [](const Vec& x) {
                return 100.0 * (x(1) * x(1) - 0.01 * x(0) * x(0) + 1.0)
                       + 0.01 * (x(0) + 10.0) * (x(0) + 10.0);
            });

        add("Bukin04", vec({-15, -3}), vec({-5, 3}), 0.0, {vec({-10.0, 0.0})},
            [](const Vec& x) {
                return 100.0 * x(1) * x(1) + 0.01 * abs(x(0) + 10.0);
            });

        add("Bukin06", vec({-15, -3}), vec({-5, 3}), 0.0, {vec({-10.0, 1.0})},
            [](const Vec& x) {
                return 100.0 * sqrt(abs(x(1) - 0.01 * x(0) * x(0))) + 0.01 * abs(x(0) + 10.0);
            });

        add("CarromTable", constant(2, -10), constant(2, 10), -24.15681551650653,
            {vec({9.646157266348881, 9.646134286497169}),
             vec({-9.646157266348881, 9.646134286497169}),
             vec({9.646157266348881, -9.646134286497169}),
             vec({-9.646157266348881, -9.646134286497169})},
            [](const Vec& x) {
                const double t = cos(x(0)) * cos(x(1))
                                 * exp(abs(1.0 - sqrt(x(0) * x(0) + x(1) * x(1)) / pi));
                return -(t * t) / 30.0;
            });

        add("Chichinadze", constant(2, -30), constant(2, 30), -42.94438701899098,
            {vec({6.189866586965680, 0.5})},
            [](const Vec& x) {
                return x(0) * x(0) - 12.0 * x(0) + 11.0 + 10.0 * cos(pi * x(0) / 2.0)
                       + 8.0 * sin(5.0 * pi * x(0) / 2.0)
                       - 1.0 / sqrt(5.0) * exp(-(x(1) - 0.5) * (x(1) - 0.5) / 2.0);
            });

        add("Colville", constant(4, -10), constant(4, 10), 0.0, {constant(4, 1)},
            [](const Vec& x) {
                return 100.0 * pow(x(0) * x(0) - x(1), 2) + pow(x(0) - 1.0, 2) + pow(x(2) - 1.0, 2)
                       + 90.0 * pow(x(2) * x(2) - x(3), 2)
                       + 10.1 * (pow(x(1) - 1.0, 2) + pow(x(3) - 1.0, 2))
                       + 19.8 * (1.0 / x(1)) * (x(3) - 1.0);
            });

        add("CosineMixture", constant(2, -1), constant(2, 1), -0.1 * 2, {constant(2, 0)},
            [](const Vec& x) {
                return -0.1 * (5.0 * pi * x.array()).cos().sum() - x.squaredNorm();
            });

        add("CrossInTray", constant(2, -10), constant(2, 10), -2.062611870822739,
            {vec({1.349406685353340, 1.349406608602084}),
             vec({-1.349406685353340, 1.349406608602084}),
             vec({1.349406685353340, -1.349406608602084}),
             vec({-1.349406685353340, -1.349406608602084})},
            [](const Vec& x) {
                const double t = abs(sin(x(0)) * sin(x(1))
                                     * exp(abs(100.0 - sqrt(x(0) * x(0) + x(1) * x(1)) / pi))) + 1.0;
                return -0.0001 * pow(t, 0.1);
            });

        add("Cube", constant(2, -10), constant(2, 10), 0.0, {constant(2, 1)},
            [](const Vec& x) {
                return 100.0 * pow(x(1) - x(0) * x(0) * x(0), 2) + pow(1.0 - x(0), 2);
            });

        add("Deb01", constant(2, -1), constant(2, 1), -1.0, {vec({0.3, -0.3})},
            [](const Vec& x) {
                const double n = static_cast<double>(x.size());
                return -(1.0 / n) * (5.0 * pi * x.array()).sin().pow(6).sum();
            });

        add("DeckkersAarts", constant(2, -20), constant(2, 20), -24776.0, {vec({0.0, 15.0})},
            [](const Vec& x) {
                const double r = x(0) * x(0) + x(1) * x(1);
                return 1e5 * x(0) * x(0) + x(1) * x(1) - r * r + 1e-5 * pow(r, 4);
            });

        add("DixonPrice", constant(2, -10), constant(2, 10), 0.0,
            {vec({1.0, pow(2.0, -0.5)})},
            [](const Vec& x) {
                double s = 0;
                for (Index i = 1; i < x.size(); ++i) {
                    s += static_cast<double>(i) * pow(2.0 * x(i) * x(i) - x(i - 1), 2);
                }
                return s + pow(x(0) - 1.0, 2);
            });

        add("DropWave", constant(2, -5.12), constant(2, 5.12), -1.0, {constant(2, 0)},
            [](const Vec& x) {
                const double r = x.squaredNorm();
                return -(1.0 + cos(12.0 * sqrt(r))) / (0.5 * r + 2.0);
            });

        // Note: the Python suite's "Easom" is an Ackley-type formula on
        // [-100, 100], not the classic Easom function; ported as written.
        add("Easom", constant(2, -100), constant(2, 100), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                const double n = static_cast<double>(x.size());
                return -20.0 * exp(-0.2 * sqrt(x.squaredNorm() / n))
                       - exp((2.0 * pi * x.array()).cos().sum() / n) + 20.0 + exp(1.0);
            });

        add("EggCrate", constant(2, -5), constant(2, 5), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                return x(0) * x(0) + x(1) * x(1)
                       + 25.0 * (sin(x(0)) * sin(x(0)) + sin(x(1)) * sin(x(1)));
            });

        add("EggHolder", constant(2, -512.1), constant(2, 512.0), -959.640662711,
            {vec({512.0, 404.2319})},
            [](const Vec& x) {
                return -(x(1) + 47.0) * sin(sqrt(abs(x(1) + x(0) / 2.0 + 47.0)))
                       - x(0) * sin(sqrt(abs(x(0) - (x(1) + 47.0))));
            });

        add("Exp2", constant(2, 0), constant(2, 20), 0.0, {vec({1.0, 0.1})},
            [](const Vec& x) {
                double y = 0;
                for (int i = 0; i < 10; ++i) {
                    const double t = exp(-i * x(0) / 10.0) - 5.0 * exp(-i * x(1) * 10.0)
                                     - exp(-i / 10.0) + 5.0 * exp(static_cast<double>(-i));
                    y += t * t;
                }
                return y;
            });

        add("Exponential", constant(2, -1), constant(2, 1), -1.0, {constant(2, 0)},
            [](const Vec& x) {
                return -exp(-0.5 * x.squaredNorm());
            });

        add("FreudensteinRoth", constant(2, -10), constant(2, 10), 0.0, {vec({5.0, 4.0})},
            [](const Vec& x) {
                const double f1 = -13.0 + x(0) + ((5.0 - x(1)) * x(1) - 2.0) * x(1);
                const double f2 = -29.0 + x(0) + ((x(1) + 1.0) * x(1) - 14.0) * x(1);
                return f1 * f1 + f2 * f2;
            });

        add("Giunta", constant(2, -1), constant(2, 1), 0.06447042053690566,
            {vec({0.4673200277395354, 0.4673200169591304})},
            [](const Vec& x) {
                double y = 0.6;
                for (Index i = 0; i < x.size(); ++i) {
                    const double arg = 16.0 * x(i) / 15.0 - 1.0;
                    y += sin(arg) + sin(arg) * sin(arg) + sin(4.0 * arg) / 50.0;
                }
                return y;
            });

        add("GoldsteinPrice", constant(2, -2), constant(2, 2), 3.0, {vec({0.0, -1.0})},
            [](const Vec& x) {
                const double a = 1.0 + pow(x(0) + x(1) + 1.0, 2)
                    * (19.0 - 14.0 * x(0) + 3.0 * x(0) * x(0) - 14.0 * x(1)
                       + 6.0 * x(0) * x(1) + 3.0 * x(1) * x(1));
                const double b = 30.0 + pow(2.0 * x(0) - 3.0 * x(1), 2)
                    * (18.0 - 32.0 * x(0) + 12.0 * x(0) * x(0) + 48.0 * x(1)
                       - 36.0 * x(0) * x(1) + 27.0 * x(1) * x(1));
                return a * b;
            });

        add("Griewank", constant(2, -600), constant(2, 600), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                double prod = 1;
                for (Index i = 0; i < x.size(); ++i) {
                    prod *= cos(x(i) / sqrt(1.0 + static_cast<double>(i)));
                }
                return x.squaredNorm() / 4000.0 - prod + 1.0;
            });

        add("Hansen", constant(2, -10), constant(2, 10), -176.54,
            {vec({-7.58989583, -7.70831466})},
            [](const Vec& x) {
                double f1 = 0, f2 = 0;
                for (int i = 0; i < 5; ++i) {
                    f1 += (i + 1) * cos(i * x(0) + i + 1);
                    f2 += (i + 1) * cos((i + 2) * x(1) + i + 1);
                }
                return f1 * f2;
            });

        add("Hartmann3", constant(3, 0), constant(3, 1), -3.86278214782076,
            {vec({0.1, 0.55592003, 0.85218259})},
            [](const Vec& x) {
                static const double a[3][4] = {{3.0, 0.1, 3.0, 0.1},
                                               {10.0, 10.0, 10.0, 10.0},
                                               {30.0, 35.0, 30.0, 35.0}};
                static const double p[3][4] = {{0.36890, 0.46990, 0.10910, 0.03815},
                                               {0.11700, 0.43870, 0.87320, 0.57430},
                                               {0.26730, 0.74700, 0.55470, 0.88280}};
                static const double c[4] = {1.0, 1.2, 3.0, 3.2};
                double y = 0;
                for (int i = 0; i < 4; ++i) {
                    double d = 0;
                    for (int j = 0; j < 3; ++j) {
                        d += a[j][i] * pow(x(j) - p[j][i], 2);
                    }
                    y -= c[i] * exp(-d);
                }
                return y;
            });

        add("Hartmann6", constant(6, 0), constant(6, 1), -3.32236801141551,
            {vec({0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162, 0.65730054})},
            [](const Vec& x) {
                static const double a[6][4] = {{10.00, 0.05, 3.00, 17.00},
                                               {3.00, 10.00, 3.50, 8.00},
                                               {17.00, 17.00, 1.70, 0.05},
                                               {3.50, 0.10, 10.00, 10.00},
                                               {1.70, 8.00, 17.00, 0.10},
                                               {8.00, 14.00, 8.00, 14.00}};
                static const double p[6][4] = {{0.1312, 0.2329, 0.2348, 0.4047},
                                               {0.1696, 0.4135, 0.1451, 0.8828},
                                               {0.5569, 0.8307, 0.3522, 0.8732},
                                               {0.0124, 0.3736, 0.2883, 0.5743},
                                               {0.8283, 0.1004, 0.3047, 0.1091},
                                               {0.5886, 0.9991, 0.6650, 0.0381}};
                static const double c[4] = {1.0, 1.2, 3.0, 3.2};
                double y = 0;
                for (int i = 0; i < 4; ++i) {
                    double d = 0;
                    for (int j = 0; j < 6; ++j) {
                        d += a[j][i] * pow(x(j) - p[j][i], 2);
                    }
                    y -= c[i] * exp(-d);
                }
                return y;
            });

        add("HimmelBlau", constant(2, -6), constant(2, 6), 0.0, {vec({3.0, 2.0})},
            [](const Vec& x) {
                return pow(x(0) * x(0) + x(1) - 11.0, 2) + pow(x(0) + x(1) * x(1) - 7.0, 2);
            });

        add("HolderTable", constant(2, -10), constant(2, 10), -19.20850256788675,
            {vec({8.055023472141116, 9.664590028909654}),
             vec({-8.055023472141116, 9.664590028909654}),
             vec({8.055023472141116, -9.664590028909654}),
             vec({-8.055023472141116, -9.664590028909654})},
            [](const Vec& x) {
                return -abs(sin(x(0)) * cos(x(1))
                            * exp(abs(1.0 - sqrt(x(0) * x(0) + x(1) * x(1)) / pi)));
            });

        add("Hosaki", constant(2, 0), constant(2, 10), -2.3458, {vec({4.0, 2.0})},
            [](const Vec& x) {
                return (1.0 + x(0) * (-8.0 + x(0) * (7.0 + x(0) * (-7.0 / 3.0 + x(0) * 0.25))))
                       * x(1) * x(1) * exp(-x(1));
            });

        add("Keane", constant(2, 0), constant(2, 10), 0.673668, {vec({0.0, 1.39325})},
            [](const Vec& x) {
                const double s1 = sin(x(0) - x(1));
                const double s2 = sin(x(0) + x(1));
                return (s1 * s1 * s2 * s2) / sqrt(x(0) * x(0) + x(1) * x(1));
            });

        add("Leon", constant(2, -1.2), constant(2, 1.2), 0.0, {constant(2, 1)},
            [](const Vec& x) {
                return 100.0 * pow(x(1) - x(0) * x(0), 2) + pow(1.0 - x(0), 2);
            });

        add("Levy03", constant(2, -10), constant(2, 10), 0.0, {constant(2, 1)},
            [](const Vec& x) {
                const Index n = x.size();
                Vec z(n);
                for (Index i = 0; i < n; ++i) {
                    z(i) = 1.0 + (x(i) - 1.0) / 4.0;
                }
                double s = sin(pi * z(0)) * sin(pi * z(0));
                for (Index i = 0; i + 1 < n; ++i) {
                    s += pow(z(i) - 1.0, 2) * (1.0 + 10.0 * pow(sin(pi * z(i) + 1.0), 2));
                }
                return s + pow(z(n - 1) - 1.0, 2) * (1.0 + pow(sin(2.0 * pi * z(n - 1)), 2));
            });

        add("Levy05", constant(2, -10), constant(2, 10), -176.1375,
            {vec({-1.30685, -1.42485})},
            [](const Vec& x) {
                double s1 = 0, s2 = 0;
                for (int i = 1; i <= 5; ++i) {
                    s1 += i * cos((i - 1.0) * x(0) + i);
                    s2 += i * cos((i + 1.0) * x(1) + i);
                }
                return s1 * s2 + pow(x(0) + 1.42513, 2) + pow(x(1) + 0.80032, 2);
            });

        add("Levy13", constant(2, -10), constant(2, 10), 0.0, {constant(2, 1)},
            [](const Vec& x) {
                return pow(sin(3.0 * pi * x(0)), 2)
                       + pow(x(0) - 1.0, 2) * (1.0 + pow(sin(3.0 * pi * x(1)), 2))
                       + pow(x(1) - 1.0, 2) * (1.0 + pow(sin(2.0 * pi * x(1)), 2));
            });

        add("Matyas", constant(2, -10), constant(2, 10), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                return 0.26 * (x(0) * x(0) + x(1) * x(1)) - 0.48 * x(0) * x(1);
            });

        add("McCormick", vec({-1.5, -3}), vec({4, 4}), -1.913222954981037,
            {vec({-0.5471975602214493, -1.547197559268372})},
            [](const Vec& x) {
                return sin(x(0) + x(1)) + pow(x(0) - x(1), 2) - 1.5 * x(0) + 2.5 * x(1) + 1.0;
            });

        add("Michalewicz", constant(2, 0), constant(2, pi), -1.8013, {},
            [](const Vec& x) {
                double y = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    y -= sin(x(i)) * pow(sin((i + 1.0) * x(i) * x(i) / pi), 20.0);
                }
                return y;
            });

        add("MieleCantrell", constant(4, -1), constant(4, 1), 0.0,
            {vec({0.0, 1.0, 1.0, 1.0})},
            [](const Vec& x) {
                return pow(exp(-x(0)) - x(1), 4) + 100.0 * pow(x(1) - x(2), 6)
                       + pow(tan(x(2) - x(3)), 4) + pow(x(0), 8);
            });

        add("Parsopoulos", constant(2, -5), constant(2, 5), 0.0, {vec({pi / 2.0, pi})},
            [](const Vec& x) {
                return pow(cos(x(0)), 2) + pow(sin(x(1)), 2);
            });

        add("Paviani", constant(10, 2.001), constant(10, 9.999), -45.7784684040686,
            {constant(10, 9.350266)},
            [](const Vec& x) {
                double s = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    s += pow(log(x(i) - 2.0), 2) + pow(log(10.0 - x(i)), 2);
                }
                return s - pow(x.prod(), 0.2);
            });

        add("PenHolder", constant(2, -11), constant(2, 11), -0.9635348327265058,
            {vec({-9.646167708023526, 9.646167671043401})},
            [](const Vec& x) {
                const double t = abs(cos(x(0)) * cos(x(1))
                                     * exp(abs(1.0 - sqrt(x(0) * x(0) + x(1) * x(1)) / pi)));
                return -exp(-1.0 / t);
            });

        add("Price01", constant(2, -500), constant(2, 500), 0.0, {vec({5.0, 5.0})},
            [](const Vec& x) {
                return pow(abs(x(0)) - 5.0, 2) + pow(abs(x(1)) - 5.0, 2);
            });

        add("Qing", constant(2, -500), constant(2, 500), 0.0,
            {vec({1.0, sqrt(2.0)})},
            [](const Vec& x) {
                double y = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    y += pow(x(i) * x(i) - (i + 1.0), 2);
                }
                return y;
            });

        add("Quintic", constant(2, -10), constant(2, 10), 0.0, {constant(2, -1)},
            [](const Vec& x) {
                double y = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    const double v = x(i);
                    y += abs(pow(v, 5) - 3.0 * pow(v, 4) + 4.0 * pow(v, 3)
                             + 2.0 * v * v - 10.0 * v - 4.0);
                }
                return y;
            });

        add("Rastrigin", constant(2, -5.12), constant(2, 5.12), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                const double n = static_cast<double>(x.size());
                return 10.0 * n + (x.array().square() - 10.0 * (2.0 * pi * x.array()).cos()).sum();
            });

        add("Rosenbrock", constant(2, -5), constant(2, 10), 0.0, {constant(2, 1)},
            [](const Vec& x) {
                double y = 0;
                for (Index i = 0; i + 1 < x.size(); ++i) {
                    y += 100.0 * pow(x(i + 1) - x(i) * x(i), 2) + pow(1.0 - x(i), 2);
                }
                return y;
            });

        add("RotatedEllipse01", constant(2, -500), constant(2, 500), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                return 7.0 * x(0) * x(0) - 6.0 * sqrt(3.0) * x(0) * x(1) + 13.0 * x(1) * x(1);
            });

        add("Salomon", constant(2, -100), constant(2, 100), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                const double r = sqrt(x.squaredNorm());
                return 1.0 - cos(2.0 * pi * r) + 0.1 * r;
            });

        add("Schaffer01", constant(2, -100), constant(2, 100), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                const double r = x(0) * x(0) + x(1) * x(1);
                return 0.5 + (pow(sin(r), 2) - 0.5) / (1.0 + 0.001 * r * r);
            });

        add("Schaffer02", constant(2, -100), constant(2, 100), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                const double d = x(0) * x(0) - x(1) * x(1);
                const double r = x(0) * x(0) + x(1) * x(1);
                return 0.5 + (pow(sin(d), 2) - 0.5) / (1.0 + 0.001 * r * r);
            });

        add("Schwefel01", constant(2, -100), constant(2, 100), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                return pow(x.squaredNorm(), sqrt(pi));
            });

        add("Schwefel20", constant(2, -100), constant(2, 100), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                return x.array().abs().sum();
            });

        add("Schwefel26", constant(2, -500), constant(2, 500), 0.0,
            {constant(2, 420.968746)},
            [](const Vec& x) {
                const double n = static_cast<double>(x.size());
                return 418.982887 * n
                       - (x.array() * x.array().abs().sqrt().sin()).sum();
            });

        {
            static const double shekelA[10][4] = {{4, 4, 4, 4}, {1, 1, 1, 1}, {8, 8, 8, 8},
                                                  {6, 6, 6, 6}, {3, 7, 3, 7}, {2, 9, 2, 9},
                                                  {5, 5, 3, 3}, {8, 1, 8, 1}, {6, 2, 6, 2},
                                                  {7, 3.6, 7, 3.6}};
            static const double shekelC[10] = {0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5};

            auto shekel = [](const Vec& x, int m) {
                double y = 0;
                for (int i = 0; i < m; ++i) {
                    double d = shekelC[i];
                    for (int j = 0; j < 4; ++j) {
                        d += pow(x(j) - shekelA[i][j], 2);
                    }
                    y -= 1.0 / d;
                }
                return y;
            };

            // Shekel05's C differs at index 4 (0.6 instead of 0.4) in the Python source
            add("Shekel05", constant(4, 0), constant(4, 10), -10.1527, {constant(4, 4)},
                [](const Vec& x) {
                    static const double c5[5] = {0.1, 0.2, 0.2, 0.4, 0.6};
                    double y = 0;
                    for (int i = 0; i < 5; ++i) {
                        double d = c5[i];
                        for (int j = 0; j < 4; ++j) {
                            d += pow(x(j) - shekelA[i][j], 2);
                        }
                        y -= 1.0 / d;
                    }
                    return y;
                });

            add("Shekel10", constant(4, 0), constant(4, 10), -10.5319, {constant(4, 4)},
                [shekel](const Vec& x) { return shekel(x, 10); });
        }

        add("Shubert01", constant(2, -10), constant(2, 10), -186.7309,
            {vec({-7.0835, 4.8580})},
            [](const Vec& x) {
                double s1 = 0, s2 = 0;
                for (int i = 1; i <= 5; ++i) {
                    s1 += i * cos((i + 1.0) * x(0) + i);
                    s2 += i * cos((i + 1.0) * x(1) + i);
                }
                return s1 * s2;
            });

        add("SixHumpCamel", constant(2, -5), constant(2, 5), -1.031628,
            {vec({0.08984201368301331, -0.7126564032704135}),
             vec({-0.08984201368301331, 0.7126564032704135})},
            [](const Vec& x) {
                return (4.0 - 2.1 * x(0) * x(0) + pow(x(0), 4) / 3.0) * x(0) * x(0)
                       + x(0) * x(1) + (4.0 * x(1) * x(1) - 4.0) * x(1) * x(1);
            });

        add("Sodp", constant(2, -1), constant(2, 1), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                double y = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    y += pow(abs(x(i)), static_cast<double>(i) + 2.0);
                }
                return y;
            });

        add("Sphere", constant(2, -5.12), constant(2, 5.12), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                return x.squaredNorm();
            });

        add("StyblinskiTang", constant(2, -5), constant(2, 5), -39.16616570377142 * 2,
            {constant(2, -2.903534018185960)},
            [](const Vec& x) {
                return (x.array().pow(4) - 16.0 * x.array().square() + 5.0 * x.array()).sum() / 2.0;
            });

        add("TestTubeHolder", constant(2, -10), constant(2, 10), -10.87229990155800,
            {vec({-pi / 2.0, 0.0})},
            [](const Vec& x) {
                return -4.0 * abs(sin(x(0)) * cos(x(1))
                                  * exp(abs(cos((x(0) * x(0) + x(1) * x(1)) / 200.0))));
            });

        add("Treccani", constant(2, -5), constant(2, 5), 0.0, {vec({-2.0, 0.0})},
            [](const Vec& x) {
                return pow(x(0), 4) + 4.0 * pow(x(0), 3) + 4.0 * x(0) * x(0) + x(1) * x(1);
            });

        add("ThreeHumpCamel", constant(2, -5), constant(2, 5), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                return 2.0 * x(0) * x(0) - 1.05 * pow(x(0), 4) + pow(x(0), 6) / 6.0
                       + x(0) * x(1) + x(1) * x(1);
            });

        add("Trid", constant(6, 0), constant(6, 20), -50.0,
            {vec({6, 10, 12, 12, 10, 6})},
            [](const Vec& x) {
                double s = (x.array() - 1.0).square().sum();
                for (Index i = 1; i < x.size(); ++i) {
                    s -= x(i) * x(i - 1);
                }
                return s;
            });

        add("Ursem01", vec({-2.5, -2}), vec({3, 2}), -4.8168, {vec({1.69714, 0.0})},
            [](const Vec& x) {
                return -sin(2.0 * x(0) - 0.5 * pi) - 3.0 * cos(x(1)) - 0.5 * x(0);
            });

        add("VenterSobiezcczanskiSobieski", constant(2, -50), constant(2, 50), -400.0,
            {constant(2, 0)},
            [](const Vec& x) {
                return x(0) * x(0) - 100.0 * pow(cos(x(0)), 2) - 100.0 * cos(x(0) * x(0) / 30.0)
                       + x(1) * x(1) - 100.0 * pow(cos(x(1)), 2) - 100.0 * cos(x(1) * x(1) / 30.0);
            });

        add("WayburnSeader01", constant(2, -5), constant(2, 5), 0.0, {vec({1.0, 2.0})},
            [](const Vec& x) {
                return pow(pow(x(0), 6) + pow(x(1), 4) - 17.0, 2)
                       + pow(2.0 * x(0) + x(1) - 4.0, 2);
            });

        add("Wolfe", constant(3, 0), constant(3, 2), 0.0, {constant(3, 0)},
            [](const Vec& x) {
                return (4.0 / 3.0) * pow(x(0) * x(0) + x(1) * x(1) - x(0) * x(1), 0.75) + x(2);
            });

        add("XinSheYang02", constant(2, -2 * pi), constant(2, 2 * pi), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                return x.array().abs().sum() * exp(-x.array().square().sin().sum());
            });

        add("YaoLiu04", constant(2, -10), constant(2, 10), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                return abs(x.maxCoeff());
            });

        add("Zacharov", constant(2, -5), constant(2, 10), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                double s = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    s += (i + 1.0) * x(i);
                }
                return x.squaredNorm() + pow(0.5 * s, 2) + pow(0.5 * s, 4);
            });

        add("Zettl", constant(2, -1), constant(2, 5), -0.003791237220468656,
            {vec({-0.02989597760285287, 0.0})},
            [](const Vec& x) {
                return pow(x(0) * x(0) + x(1) * x(1) - 2.0 * x(0), 2) + x(0) / 4.0;
            });

        add("Zirilli", constant(2, -10), constant(2, 10), -0.3523, {vec({-1.0465, 0.0})},
            [](const Vec& x) {
                return 0.25 * pow(x(0), 4) - 0.5 * x(0) * x(0) + 0.1 * x(0) + 0.5 * x(1) * x(1);
            });

        add("AMGM", constant(2, 0), constant(2, 10), 0.0, {constant(2, 1)},
            [](const Vec& x) {
                const double n = static_cast<double>(x.size());
                const double f1 = x.sum() / n;
                const double f2 = pow(x.prod(), 1.0 / n);
                return (f1 - f2) * (f1 - f2);
            });

        add("BoxBetts", vec({0.9, 9.0, 0.9}), vec({1.2, 11.2, 1.2}), 0.0,
            {vec({1.0, 10.0, 1.0})},
            [](const Vec& x) {
                double y = 0;
                for (int i = 1; i <= 10; ++i) {
                    const double t = exp(-0.1 * i * x(0)) - exp(-0.1 * i * x(1))
                                     - (exp(-0.1 * i) - exp(-1.0 * i)) * x(2);
                    y += t * t;
                }
                return y;
            });

        add("Cigar", constant(2, -100), constant(2, 100), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                return x(0) * x(0) + 1e6 * (x.squaredNorm() - x(0) * x(0));
            });

        add("Cola", (Vec(17) << 0, constant(16, -4)).finished(),
            (Vec(17) << 4, constant(16, 4)).finished(), 11.7464,
            {vec({0.651906, 1.30194, 0.099242, -0.883791, -0.8796, 0.204651, -3.28414,
                  0.851188, -3.46245, 2.53245, -0.895246, 1.40992, -3.07367, 1.96257,
                  -2.97872, -0.807849, -1.68978})},
            [](const Vec& x) {
                static const double d[10][10] = {
                    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                    {1.27, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                    {1.69, 1.43, 0, 0, 0, 0, 0, 0, 0, 0},
                    {2.04, 2.35, 2.43, 0, 0, 0, 0, 0, 0, 0},
                    {3.09, 3.18, 3.26, 2.85, 0, 0, 0, 0, 0, 0},
                    {3.20, 3.22, 3.27, 2.88, 1.55, 0, 0, 0, 0, 0},
                    {2.86, 2.56, 2.58, 2.59, 3.12, 3.06, 0, 0, 0, 0},
                    {3.17, 3.18, 3.18, 3.12, 1.31, 1.64, 3.00, 0, 0, 0},
                    {3.21, 3.18, 3.18, 3.17, 1.70, 1.36, 2.95, 1.32, 0, 0},
                    {2.38, 2.31, 2.42, 1.94, 2.85, 2.81, 2.56, 2.91, 2.97, 0}};
                double x1[10], x2[10];
                x1[0] = 0.0; x1[1] = x(0);
                x2[0] = 0.0; x2[1] = 0.0;
                for (int i = 0; i < 8; ++i) {
                    x1[i + 2] = x(1 + 2 * i);
                    x2[i + 2] = x(2 + 2 * i);
                }
                double y = 0;
                for (int i = 1; i < 10; ++i) {
                    for (int j = 0; j < i; ++j) {
                        const double dist = sqrt(pow(x1[i] - x1[j], 2) + pow(x2[i] - x2[j], 2));
                        y += pow(dist - d[i][j], 2);
                    }
                }
                return y;
            });

        add("Corana", constant(4, -5), constant(4, 5), 0.0, {constant(4, 0)},
            [](const Vec& x) {
                static const double d[4] = {1.0, 1000.0, 10.0, 100.0};
                auto sgn = [](double v) { return (v > 0) - (v < 0); };
                double r = 0;
                for (int j = 0; j < 4; ++j) {
                    const double zj = std::floor(abs(x(j) / 0.2) + 0.49999) * sgn(x(j)) * 0.2;
                    if (abs(x(j) - zj) < 0.05) {
                        r += 0.15 * pow(zj - 0.05 * sgn(zj), 2) * d[j];
                    } else {
                        r += d[j] * x(j) * x(j);
                    }
                }
                return r;
            });

        add("CrossLegTable", constant(2, -10), constant(2, 10), -1.0, {},
            [](const Vec& x) {
                const double t = abs(sin(x(0)) * sin(x(1))
                                     * exp(abs(100.0 - sqrt(x(0) * x(0) + x(1) * x(1)) / pi))) + 1.0;
                return -pow(t, -0.1);
            });

        add("CrownedCross", constant(2, -10), constant(2, 10), 0.0001, {constant(2, 0)},
            [](const Vec& x) {
                const double t = abs(sin(x(0)) * sin(x(1))
                                     * exp(abs(100.0 - sqrt(x(0) * x(0) + x(1) * x(1)) / pi))) + 1.0;
                return 0.0001 * pow(t, 0.1);
            });

        add("Csendes", constant(2, -1), constant(2, 1), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                double y = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    y += pow(x(i), 6) * (2.0 + sin(1.0 / x(i)));
                }
                return y;
            });

        add("Damavandi", constant(2, 0), constant(2, 14), 0.0, {constant(2, 2)},
            [](const Vec& x) {
                const double numerator = sin(pi * (x(0) - 2.0)) * sin(pi * (x(1) - 2.0));
                const double denominator = pi * pi * (x(0) - 2.0) * (x(1) - 2.0);
                const double factor1 = 1.0 - pow(abs(numerator / denominator), 5);
                const double factor2 = 2.0 + pow(x(0) - 7.0, 2) + 2.0 * pow(x(1) - 7.0, 2);
                return factor1 * factor2;
            });

        add("Deb02", constant(2, 0), constant(2, 1), -1.0,
            {vec({0.93388314, 0.68141781})},
            [](const Vec& x) {
                const double n = static_cast<double>(x.size());
                double s = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    s += pow(sin(5.0 * pi * (pow(x(i), 0.75) - 0.05)), 6);
                }
                return -(1.0 / n) * s;
            });

        add("Decanomial", constant(2, -10), constant(2, 10), 0.0, {vec({2.0, -3.0})},
            [](const Vec& x) {
                const double f1 = abs(pow(x(0), 10) - 20.0 * pow(x(0), 9) + 180.0 * pow(x(0), 8)
                                      - 960.0 * pow(x(0), 7) + 3360.0 * pow(x(0), 6)
                                      - 8064.0 * pow(x(0), 5) + 13340.0 * pow(x(0), 4)
                                      - 15360.0 * pow(x(0), 3) + 11520.0 * pow(x(0), 2)
                                      - 5120.0 * x(0) + 2624.0);
                const double f2 = abs(pow(x(1), 4) + 12.0 * pow(x(1), 3) + 54.0 * pow(x(1), 2)
                                      + 108.0 * x(1) + 81.0);
                return 0.001 * pow(f1 + f2, 2);
            });

        add("Deceptive", constant(2, 0), constant(2, 1), -1.0,
            {vec({1.0 / 3.0, 2.0 / 3.0})},
            [](const Vec& x) {
                const Index n = x.size();
                double s = 0;
                for (Index i = 0; i < n; ++i) {
                    const double alpha = (i + 1.0) / (n + 1.0);
                    double g;
                    if (x(i) <= 0.0) {
                        g = x(i);
                    } else if (x(i) < 0.8 * alpha) {
                        g = -x(i) / alpha + 0.8;
                    } else if (x(i) < alpha) {
                        g = 5.0 * x(i) / alpha - 4.0;
                    } else if (x(i) < (1.0 + 4.0 * alpha) / 5.0) {
                        g = 5.0 * (x(i) - alpha) / (alpha - 1.0) + 1.0;
                    } else if (x(i) <= 1.0) {
                        g = (x(i) - 1.0) / (1.0 - alpha) + 4.0 / 5.0;
                    } else {
                        g = x(i) - 1.0;
                    }
                    s += g;
                }
                return -pow(s / static_cast<double>(n), 2);
            });

        add("DeflectedCorrugatedSpring", constant(2, 0), constant(2, 10), -1.0,
            {constant(2, 5)},
            [](const Vec& x) {
                const double K = 5.0, alpha = 5.0;
                const double r = (x.array() - alpha).square().sum();
                return -cos(K * sqrt(r)) + 0.1 * r;
            });

        add("DeVilliersGlasser01", constant(4, 1), constant(4, 100), 0.0,
            {vec({60.137, 1.371, 3.112, 1.761})},
            [](const Vec& x) {
                double y = 0;
                for (int i = 0; i < 24; ++i) {
                    const double t = 0.1 * i;
                    const double yi = 60.137 * pow(1.371, t) * sin(3.112 * t + 1.761);
                    y += pow(x(0) * pow(x(1), t) * sin(x(2) * t + x(3)) - yi, 2);
                }
                return y;
            });

        add("DeVilliersGlasser02", constant(5, 1), constant(5, 60), 0.0,
            {vec({53.81, 1.27, 3.012, 2.13, 0.507})},
            [](const Vec& x) {
                double y = 0;
                for (int i = 0; i < 16; ++i) {
                    const double t = 0.1 * i;
                    const double yi = 53.81 * pow(1.27, t) * std::tanh(3.012 * t + sin(2.13 * t))
                                      * cos(exp(0.507) * t);
                    y += pow(x(0) * pow(x(1), t) * std::tanh(x(2) * t + sin(x(3) * t))
                             * cos(t * exp(x(4))) - yi, 2);
                }
                return y;
            });

        add("Dolan", constant(5, -100), constant(5, 100), 1e-5,
            {vec({8.39045925, 4.81424707, 7.34574133, 68.88246895, 3.85470806})},
            [](const Vec& x) {
                return abs((x(0) + 1.7 * x(1)) * sin(x(0)) - 1.5 * x(2)
                           - 0.1 * x(3) * cos(x(3) + x(4) - x(0)) + 0.2 * x(4) * x(4)
                           - x(1) - 1.0);
            });

        add("ElAttarVidyasagarDutta", constant(2, -100), constant(2, 100), 1.712780354,
            {vec({3.40918683, -2.17143304})},
            [](const Vec& x) {
                return pow(x(0) * x(0) + x(1) - 10.0, 2) + pow(x(0) + x(1) * x(1) - 7.0, 2)
                       + pow(x(0) * x(0) + pow(x(1), 3) - 1.0, 2);
            });

        add("Gear", constant(4, 12), constant(4, 60), 2.7e-12,
            {vec({16.0, 19.0, 43.0, 49.0})},
            [](const Vec& x) {
                return pow(1.0 / 6.931
                           - std::floor(x(0)) * std::floor(x(1))
                             / (std::floor(x(2)) * std::floor(x(3))), 2);
            });

        add("Gulf", constant(3, 0), constant(3, 50), 0.0, {vec({50.0, 25.0, 1.5})},
            [](const Vec& x) {
                double y = 0;
                for (int i = 0; i < 30; ++i) {
                    const double ti = i * 0.01;
                    const double yi = 25.0 + pow(-50.0 * log(ti), 2.0 / 3.0);
                    const double ai = yi - x(1);
                    y += pow(exp(-(pow(abs(ai), x(2)) / x(0))) - ti, 2);
                }
                return y;
            });

        add("HelicalValley", constant(3, -100), constant(3, 100), 0.0,
            {vec({1.0, 0.0, 0.0})},
            [](const Vec& x) {
                return 100.0 * (pow(x(2) - 10.0 * std::atan2(x(1), x(0)) / 2.0 / pi, 2)
                                + pow(sqrt(x(0) * x(0) + x(1) * x(1)) - 1.0, 2))
                       + x(2) * x(2);
            });

        add("Holzman", vec({0, 0, 0}), vec({100, 25.6, 5}), 0.0,
            {vec({50.0, 25.0, 1.5})},
            [](const Vec& x) {
                double y = 0;
                for (int i = 0; i < 100; ++i) {
                    const double ui = 25.0 + pow(-50.0 * log(0.01 * (i + 1)), 2.0 / 3.0);
                    y += -0.1 * (i + 1) + exp(1.0 / x(0) * pow(ui - x(1), x(2)));
                }
                return y;
            });

        add("Infinity", constant(2, -1), constant(2, 1), 0.0, {constant(2, 1e-16)},
            [](const Vec& x) {
                double y = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    y += pow(x(i), 6) * (sin(1.0 / x(i)) + 2.0);
                }
                return y;
            });

        add("JennrichSampson", constant(2, -1), constant(2, 1), 124.3621824,
            {constant(2, 0.257825)},
            [](const Vec& x) {
                double y = 0;
                for (int k = 1; k <= 10; ++k) {
                    y += pow(2.0 + 2.0 * k - (exp(k * x(0)) + exp(k * x(1))), 2);
                }
                return y;
            });

        add("Judge", constant(2, -10), constant(2, 10), 16.0817307,
            {vec({0.86479, 1.2357})},
            [](const Vec& x) {
                static const double Y[20] = {4.284, 4.149, 3.877, 0.533, 2.211, 2.389, 2.145,
                                             3.231, 1.998, 1.379, 2.106, 1.428, 1.011, 2.179,
                                             2.858, 1.388, 1.651, 1.593, 1.046, 2.152};
                static const double X2[20] = {0.286, 0.973, 0.384, 0.276, 0.973, 0.543, 0.957,
                                              0.948, 0.543, 0.797, 0.936, 0.889, 0.006, 0.828,
                                              0.399, 0.617, 0.939, 0.784, 0.072, 0.889};
                static const double X3[20] = {0.645, 0.585, 0.310, 0.058, 0.455, 0.779, 0.259,
                                              0.202, 0.028, 0.099, 0.142, 0.296, 0.175, 0.180,
                                              0.842, 0.039, 0.103, 0.620, 0.158, 0.704};
                double y = 0;
                for (int i = 0; i < 20; ++i) {
                    y += pow(x(0) + x(1) * X2[i] + x(1) * x(1) * X3[i] - Y[i], 2);
                }
                return y;
            });

        add("Katsuura", constant(2, 0), constant(2, 100), 1.0, {constant(2, 0)},
            [](const Vec& x) {
                const int d = 32;
                double prod = 1;
                for (Index i = 0; i < x.size(); ++i) {
                    double s = 0;
                    for (int k = 1; k <= d; ++k) {
                        const double pow2 = pow(2.0, k);
                        s += std::nearbyint(pow2 * x(i)) / pow2;
                    }
                    prod *= 1.0 + (i + 1.0) * s;
                }
                return prod;
            });

        add("Kowalik", constant(4, -5), constant(4, 5), 0.00030748610,
            {vec({0.192833, 0.190836, 0.123117, 0.135766})},
            [](const Vec& x) {
                static const double b[11] = {4.0, 2.0, 1.0, 1.0 / 2.0, 1.0 / 4.0, 1.0 / 6.0,
                                             1.0 / 8.0, 1.0 / 10.0, 1.0 / 12.0, 1.0 / 14.0,
                                             1.0 / 16.0};
                static const double a[11] = {0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627,
                                             0.0456, 0.0342, 0.0323, 0.0235, 0.0246};
                double y = 0;
                for (int i = 0; i < 11; ++i) {
                    const double bb = b[i] * b[i];
                    const double t = a[i] - (x(0) * (bb + b[i] * x(1)) / (bb + b[i] * x(2) + x(3)));
                    y += t * t;
                }
                return y;
            });

        add("Langermann", constant(2, 0), constant(2, 10), -5.1621259,
            {vec({2.00299219, 1.006096})},
            [](const Vec& x) {
                static const double a[5] = {3, 5, 2, 1, 7};
                static const double b[5] = {5, 2, 1, 4, 9};
                static const double c[5] = {1, 2, 5, 2, 3};
                double y = 0;
                for (int i = 0; i < 5; ++i) {
                    const double r = pow(x(0) - a[i], 2) + pow(x(1) - b[i], 2);
                    y -= c[i] * exp(-(1.0 / pi) * r) * cos(pi * r);
                }
                return y;
            });

        add("LennardJones", constant(6, -4), constant(6, 4), -1.0, {},
            [](const Vec& x) {
                const int k = static_cast<int>(x.size()) / 3;
                double s = 0;
                for (int i = 0; i + 1 < k; ++i) {
                    for (int j = i + 1; j < k; ++j) {
                        const int a = 3 * i;
                        const int b = 3 * j;
                        const double xd = x(a) - x(b);
                        const double yd = x(a + 1) - x(b + 1);
                        const double zd = x(a + 2) - x(b + 2);
                        const double ed = xd * xd + yd * yd + zd * zd;
                        const double ud = ed * ed * ed;
                        if (ed > 0.0) {
                            s += (1.0 / ud - 2.0) / ud;
                        }
                    }
                }
                return s;
            });

        add("Mishra01", constant(2, 0), constant(2, 1.0 + 1e-9), 2.0, {constant(2, 1)},
            [](const Vec& x) {
                const double n = static_cast<double>(x.size());
                double s = 0;
                for (Index i = 0; i + 1 < x.size(); ++i) {
                    s += x(i);
                }
                const double xn = n - s;
                return pow(1.0 + xn, xn);
            });

        add("Mishra02", constant(2, 0), constant(2, 1.0 + 1e-9), 2.0, {constant(2, 1)},
            [](const Vec& x) {
                const double n = static_cast<double>(x.size());
                double s = 0;
                for (Index i = 0; i + 1 < x.size(); ++i) {
                    s += (x(i) + x(i + 1)) / 2.0;
                }
                const double xn = n - s;
                return pow(1.0 + xn, xn);
            });

        add("Mishra03", constant(2, -10), constant(2, 10), -0.18467, {constant(2, -10)},
            [](const Vec& x) {
                return sqrt(abs(cos(sqrt(abs(x(0) * x(0) + x(1) * x(1))))))
                       + 0.01 * (x(0) + x(1));
            });

        add("Mishra04", constant(2, -10), constant(2, 10), -0.199409, {constant(2, -10)},
            [](const Vec& x) {
                return sqrt(abs(sin(sqrt(abs(x(0) * x(0) + x(1) * x(1))))))
                       + 0.01 * (x(0) + x(1));
            });

        add("Mishra05", constant(2, -10), constant(2, 10), -0.119829,
            {vec({-1.98682, -10.0})},
            [](const Vec& x) {
                const double a = pow(sin(pow(cos(x(0)) + cos(x(1)), 2)), 2);
                const double b = pow(cos(pow(sin(x(0)) + sin(x(1)), 2)), 2);
                return pow(a + b + x(0), 2) + 0.01 * (x(0) + x(1));
            });

        add("Mishra06", constant(2, -10), constant(2, 10), -2.28395,
            {vec({2.88631, 1.82326})},
            [](const Vec& x) {
                const double a = pow(sin(pow(cos(x(0)) + cos(x(1)), 2)), 2);
                const double b = pow(cos(pow(sin(x(0)) + sin(x(1)), 2)), 2);
                return -log(pow(a - b + x(0), 2))
                       + 0.1 * (pow(x(0) - 1.0, 2) + pow(x(1) - 1.0, 2));
            });

        add("Mishra07", constant(2, -10), constant(2, 10), 0.0,
            {constant(2, sqrt(2.0))},
            [](const Vec& x) {
                const double factorial = std::tgamma(static_cast<double>(x.size()) + 1.0);
                return pow(x.prod() - factorial, 2);
            });

        add("Mishra08", constant(2, -10), constant(2, 10), 0.0, {vec({2.0, -3.0})},
            [](const Vec& x) {
                const double f1 = abs(pow(x(0), 10) - 20.0 * pow(x(0), 9) + 180.0 * pow(x(0), 8)
                                      - 960.0 * pow(x(0), 7) + 3360.0 * pow(x(0), 6)
                                      - 8064.0 * pow(x(0), 5) + 13340.0 * pow(x(0), 4)
                                      - 15360.0 * pow(x(0), 3) + 11520.0 * pow(x(0), 2)
                                      - 5120.0 * x(0) + 2624.0);
                const double f2 = abs(pow(x(1), 4) + 12.0 * pow(x(1), 3) + 54.0 * pow(x(1), 2)
                                      + 108.0 * x(1) + 81.0);
                return 0.001 * pow(f1 + f2, 2);
            });

        add("Mishra09", constant(3, -10), constant(3, 10), 0.0, {vec({1.0, 2.0, 3.0})},
            [](const Vec& x) {
                const double x1 = x(0), x2 = x(1), x3 = x(2);
                const double f1 = 2.0 * pow(x1, 3) + 5.0 * x1 * x2 + 4.0 * x3
                                  - 2.0 * x1 * x1 * x3 - 18.0;
                const double f2 = x1 + pow(x2, 3) + x1 * x2 * x2 + x1 * x3 * x3 - 22.0;
                const double f3 = 8.0 * x1 * x1 + 2.0 * x2 * x3 + 2.0 * x2 * x2
                                  + 3.0 * pow(x2, 3) - 52.0;
                return pow(f1 * f3 * f2 * f2 + f1 * f2 * f3 * f3 + f2 * f2
                           + pow(x1 + x2 - x3, 2), 2);
            });

        add("Mishra10", constant(2, -10), constant(2, 10), 0.0, {constant(2, 2)},
            [](const Vec& x) {
                const double x1 = static_cast<double>(static_cast<long long>(x(0)));
                const double x2 = static_cast<double>(static_cast<long long>(x(1)));
                return pow((x1 + x2) - x1 * x2, 2);
            });

        // Mishra11's second term is prod(|x|)/n in the Python source (the
        // **1.0 binds before /n); ported as written
        add("Mishra11", constant(2, -10), constant(2, 10), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                const double n = static_cast<double>(x.size());
                return pow((1.0 / n) * x.array().abs().sum()
                           - x.array().abs().prod() / n, 2);
            });

        add("MultiModal", constant(2, -10), constant(2, 10), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                return x.array().abs().sum() * x.array().abs().prod();
            });

        add("NeedleEye", constant(2, -10), constant(2, 10), 1.0, {},
            [](const Vec& x) {
                const double eye = 0.0001;
                double f = 0, fp = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    if (abs(x(i)) >= eye) {
                        fp = 1.0;
                        f += 100.0 + abs(x(i));
                    } else {
                        f += 1.0;
                    }
                }
                if (fp < 1e-6) {
                    f = f / static_cast<double>(x.size());
                }
                return f;
            });

        add("NewFunction01", constant(2, -10), constant(2, 10), -0.17894509347721144,
            {vec({-8.4666, -9.9988})},
            [](const Vec& x) {
                return pow(abs(cos(sqrt(abs(x(0) * x(0) + x(1))))), 0.5)
                       + 0.01 * x(0) + 0.01 * x(1);
            });

        add("NewFunction02", constant(2, -10), constant(2, 10), -0.1971881059905,
            {vec({-9.94112, -9.99952})},
            [](const Vec& x) {
                return pow(abs(sin(sqrt(abs(x(0) * x(0) + x(1))))), 0.5)
                       + 0.01 * x(0) + 0.01 * x(1);
            });

        add("NewFunction03", constant(2, -10), constant(2, 10), -1.019829,
            {vec({-1.98682, -10.0})},
            [](const Vec& x) {
                const double f1 = pow(sin(pow(cos(x(0)) + cos(x(1)), 2)), 2);
                const double f2 = pow(cos(pow(sin(x(0)) + sin(x(1)), 2)), 2);
                return pow(f1 + f2 + x(0), 2) + 0.01 * x(0) + 0.1 * x(1);
            });

        add("OddSquare", constant(2, -5 * pi), constant(2, 5 * pi), -1.0084,
            {vec({1.0, 1.3})},
            [](const Vec& x) {
                static const double a[2] = {1.0, 1.3};
                const double n = static_cast<double>(x.size());
                double d = 0, h = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    const double t = pow(x(i) - a[i], 2);
                    d = std::max(d, t);
                    h += t;
                }
                d *= n;
                return -exp(-d / (2.0 * pi)) * cos(pi * d) * (1.0 + 0.02 * h / (d + 0.01));
            });

        add("Pathological", constant(2, -100), constant(2, 100), -1.99600798403,
            {constant(2, 0)},
            [](const Vec& x) {
                const Index n = x.size();
                double y = 0;
                for (Index i = 0; i < n; ++i) {
                    const double xi = x(i);
                    const double xn = x((i + 1) % n);
                    y += (pow(sin(sqrt(xn * xn + 100.0 * xi * xi)), 2) - 0.5)
                         / (0.001 * (pow(xn - xi, 4) + 1.0) + 0.5);
                }
                return y;
            });

        add("Penalty01", constant(2, -50), constant(2, 50), 0.0, {constant(2, -1)},
            [](const Vec& x) {
                const Index n = x.size();
                const double a = 10.0, b = 100.0, c = 4.0;
                double u = 0;
                for (Index i = 0; i < n; ++i) {
                    const double xx = abs(x(i));
                    if (xx > a) {
                        u += b * pow(xx - a, c);
                    }
                }
                Vec y = 1.0 + (x.array() + 1.0) / 4.0;
                double s = 10.0 * pow(sin(pi * y(0)), 2);
                for (Index i = 0; i + 1 < n; ++i) {
                    s += pow(y(i) - 1.0, 2) * (1.0 + 10.0 * pow(sin(pi * y(i + 1)), 2));
                }
                s += pow(y(n - 1) - 1.0, 2);
                return u + (pi / 30.0) * s;
            });

        add("Penalty02", constant(2, -50), constant(2, 50), 0.0, {constant(2, 1)},
            [](const Vec& x) {
                const Index n = x.size();
                const double a = 5.0, b = 100.0, c = 4.0;
                double u = 0;
                for (Index i = 0; i < n; ++i) {
                    const double xx = abs(x(i));
                    if (xx > a) {
                        u += b * pow(xx - a, c);
                    }
                }
                double s = 10.0 * pow(sin(3.0 * pi * x(0)), 2);
                for (Index i = 0; i + 1 < n; ++i) {
                    s += pow(x(i) - 1.0, 2) * (1.0 + pow(sin(pi * x(i + 1)), 2));
                }
                s += pow(x(n - 1) - 1.0, 2) * (1.0 + pow(sin(2.0 * pi * x(n - 1)), 2));
                return u + 0.1 * s;
            });

        add("PermFunction01", constant(2, -2), constant(2, 3), 0.0, {vec({1.0, 2.0})},
            [](const Vec& x) {
                const Index n = x.size();
                const double b = 0.5;
                double sOut = 0;
                for (Index k = 1; k <= n; ++k) {
                    double sIn = 0;
                    for (Index j = 1; j <= n; ++j) {
                        sIn += (pow(static_cast<double>(j), static_cast<double>(k)) + b)
                               * (pow(x(j - 1) / j, static_cast<double>(k)) - 1.0);
                    }
                    sOut += sIn * sIn;
                }
                return sOut;
            });

        add("PermFunction02", constant(2, -2), constant(2, 3), 0.0, {vec({1.0, 2.0})},
            [](const Vec& x) {
                const Index n = x.size();
                const double b = 10.0;
                double sOut = 0;
                for (Index k = 1; k <= n; ++k) {
                    double sIn = 0;
                    for (Index j = 1; j <= n; ++j) {
                        sIn += (j + b) * (pow(x(j - 1), static_cast<double>(k))
                                          - pow(1.0 / j, static_cast<double>(k)));
                    }
                    sOut += sIn * sIn;
                }
                return sOut;
            });

        add("Pinter", constant(2, -10), constant(2, 10), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                const Index n = x.size();
                double f = 0;
                for (Index i = 0; i < n; ++i) {
                    const double xi = x(i);
                    const double xmi = x((i + n - 1) % n);
                    const double xpi = x((i + 1) % n);
                    const double A = xmi * sin(xi) + sin(xpi);
                    const double B = xmi * xmi - 2.0 * xi + 3.0 * xpi - cos(xi) + 1.0;
                    f += (i + 1.0) * xi * xi + 20.0 * (i + 1.0) * pow(sin(A), 2)
                         + (i + 1.0) * std::log10(1.0 + (i + 1.0) * B * B);
                }
                return f;
            });

        add("Plateau", constant(2, -5.12), constant(2, 5.12), 30.0, {constant(2, 0)},
            [](const Vec& x) {
                return 30.0 + x.array().abs().floor().sum();
            });

        add("Powell", constant(4, -4), constant(4, 5), 0.0, {constant(4, 0)},
            [](const Vec& x) {
                return pow(x(0) + 10.0 * x(1), 2) + 5.0 * pow(x(2) - x(3), 2)
                       + pow(x(1) - 2.0 * x(2), 4) + 10.0 * pow(x(0) - x(3), 4);
            });

        add("PowerSum", constant(4, 0), constant(4, 4), 0.0, {vec({1.0, 2.0, 2.0, 3.0})},
            [](const Vec& x) {
                static const double b[4] = {8.0, 18.0, 44.0, 114.0};
                double y = 0;
                for (int k = 1; k <= 4; ++k) {
                    double sIn = 0;
                    for (Index i = 0; i < x.size(); ++i) {
                        sIn += pow(x(i), k);
                    }
                    y += pow(sIn - b[k - 1], 2);
                }
                return y;
            });

        add("Price02", constant(2, -10), constant(2, 10), 0.9, {constant(2, 0)},
            [](const Vec& x) {
                return 1.0 + pow(sin(x(0)), 2) + pow(sin(x(1)), 2)
                       - 0.1 * exp(-x(0) * x(0) - x(1) * x(1));
            });

        add("Price03", constant(2, -50), constant(2, 50), 0.0, {constant(2, 1)},
            [](const Vec& x) {
                return 100.0 * pow(x(1) - x(0) * x(0), 2)
                       + pow(6.4 * pow(x(1) - 0.5, 2) - x(0) - 0.6, 2);
            });

        add("Price04", constant(2, -50), constant(2, 50), 0.0, {vec({2.0, 4.0})},
            [](const Vec& x) {
                return pow(2.0 * x(1) * pow(x(0), 3) - pow(x(1), 3), 2)
                       + pow(6.0 * x(0) - x(1) * x(1) + x(1), 2);
            });

        add("Quadratic", constant(2, -10), constant(2, 10), -3873.72418,
            {vec({0.19388, 0.48513})},
            [](const Vec& x) {
                return -3803.84 - 138.08 * x(0) - 232.92 * x(1) + 128.08 * x(0) * x(0)
                       + 203.64 * x(1) * x(1) + 182.25 * x(0) * x(1);
            });

        add("Rana", constant(2, -500.000001), constant(2, 500.000001), -928.5478,
            {constant(2, -500)},
            [](const Vec& x) {
                double y = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    const double e = x(i) + 1.0;
                    y += e * cos(sqrt(abs(e - x(i)))) * sin(sqrt(abs(e + x(i))))
                         + x(i) * cos(sqrt(abs(e + x(i)))) * sin(sqrt(abs(e - x(i))));
                }
                return y;
            });

        add("Ripple01", constant(2, 0), constant(2, 1), -2.2, {constant(2, 0.1)},
            [](const Vec& x) {
                double y = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    y += -exp(-2.0 * log(2.0) * pow((x(i) - 0.1) / 0.8, 2))
                         * (pow(sin(5.0 * pi * x(i)), 6)
                            + 0.1 * pow(cos(500.0 * pi * x(i)), 2));
                }
                return y;
            });

        add("Ripple25", constant(2, 0), constant(2, 1), -2.0, {constant(2, 0.1)},
            [](const Vec& x) {
                double y = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    y += -exp(-2.0 * log(2.0) * pow((x(i) - 0.1) / 0.8, 2))
                         * pow(sin(5.0 * pi * x(i)), 6);
                }
                return y;
            });

        add("RosenbrockModified", constant(2, -2), constant(2, 2), 34.37,
            {vec({-0.9, -0.95})},
            [](const Vec& x) {
                return 74.0 + 100.0 * pow(x(1) - x(0) * x(0), 2) + pow(1.0 - x(0), 2)
                       - 400.0 * exp(-(pow(x(0) + 1.0, 2) + pow(x(1) + 1.0, 2)) / 0.1);
            });

        add("RotatedEllipse02", constant(2, -500), constant(2, 500), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                return x(0) * x(0) - x(0) * x(1) + x(1) * x(1);
            });

        add("Sargan", constant(2, -100), constant(2, 100), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                const double n = static_cast<double>(x.size());
                double cross = 0;
                for (Index i = 0; i + 1 < x.size(); ++i) {
                    cross += x(i) * x(i + 1);
                }
                double y = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    y += n * (x(i) * x(i) + 0.4 * cross);
                }
                return y;
            });

        add("Schaffer03", constant(2, -100), constant(2, 100), 0.00156685,
            {vec({0.0, 1.253115})},
            [](const Vec& x) {
                const double d = x(0) * x(0) - x(1) * x(1);
                const double r = x(0) * x(0) + x(1) * x(1);
                return 0.5 + (pow(sin(cos(abs(d))), 2) - 0.5) / (1.0 + 0.001 * r * r);
            });

        add("Schaffer04", constant(2, -100), constant(2, 100), 0.292579,
            {vec({0.0, 1.253115})},
            [](const Vec& x) {
                const double d = x(0) * x(0) - x(1) * x(1);
                const double r = x(0) * x(0) + x(1) * x(1);
                return 0.5 + (pow(cos(sin(d)), 2) - 0.5) / (1.0 + 0.001 * r * r);
            });

        add("SchmidtVetters", constant(3, 0), constant(3, 10), 3.0, {constant(3, 0.78547)},
            [](const Vec& x) {
                return 1.0 / (1.0 + pow(x(0) - x(1), 2)) + sin((pi * x(1) + x(2)) / 2.0)
                       + exp(pow((x(0) + x(1)) / x(1) - 2.0, 2));
            });

        add("Schwefel02", constant(2, -100), constant(2, 100), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                double y = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    double partial = 0;
                    for (Index j = 0; j < i; ++j) {
                        partial += x(j);
                    }
                    y += partial * partial;
                }
                return y;
            });

        add("Schwefel04", constant(2, 0), constant(2, 10), 0.0, {constant(2, 1)},
            [](const Vec& x) {
                double y = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    y += pow(x(i) - 1.0, 2) + pow(x(0) - x(i) * x(i), 2);
                }
                return y;
            });

        add("Schwefel06", constant(2, -100), constant(2, 100), 0.0, {vec({1.0, 3.0})},
            [](const Vec& x) {
                return std::max(abs(x(0) + 2.0 * x(1) - 7.0), abs(2.0 * x(0) + x(1) - 5.0));
            });

        add("Schwefel21", constant(2, -100), constant(2, 100), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                return x.array().abs().maxCoeff();
            });

        add("Schwefel22", constant(2, -100), constant(2, 100), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                return x.array().abs().sum() + x.array().abs().prod();
            });

        add("Schwefel36", constant(2, 0), constant(2, 500), -3456.0, {constant(2, 12)},
            [](const Vec& x) {
                return -x(0) * x(1) * (72.0 - 2.0 * x(0) - 2.0 * x(1));
            });

        add("Shekel07", constant(4, 0), constant(4, 10), -10.3999, {constant(4, 4)},
            [](const Vec& x) {
                static const double A[7][4] = {{4, 4, 4, 4}, {1, 1, 1, 1}, {8, 8, 8, 8},
                                               {6, 6, 6, 6}, {3, 7, 3, 7}, {2, 9, 2, 9},
                                               {5, 5, 3, 3}};
                static const double C[7] = {0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3};
                double y = 0;
                for (int i = 0; i < 7; ++i) {
                    double d = C[i];
                    for (int j = 0; j < 4; ++j) {
                        d += pow(x(j) - A[i][j], 2);
                    }
                    y -= 1.0 / d;
                }
                return y;
            });

        add("Shubert03", constant(2, -10), constant(2, 10), -24.062499,
            {constant(2, 5.791794)},
            [](const Vec& x) {
                double y = 0;
                for (Index d = 0; d < 2; ++d) {
                    for (int i = 1; i <= 5; ++i) {
                        y -= i * sin((i + 1.0) * x(d) + i);
                    }
                }
                return y;
            });

        add("Shubert04", constant(2, -10), constant(2, 10), -29.016015,
            {vec({-0.80032121, -7.08350592})},
            [](const Vec& x) {
                double y = 0;
                for (Index d = 0; d < 2; ++d) {
                    for (int i = 1; i <= 5; ++i) {
                        y -= i * cos((i + 1.0) * x(d) + i);
                    }
                }
                return y;
            });

        add("SineEnvelope", constant(2, -100), constant(2, 100), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                double y = 0;
                for (Index i = 0; i + 1 < x.size(); ++i) {
                    const double r = x(i) * x(i) + x(i + 1) * x(i + 1);
                    y += (pow(sin(sqrt(r)), 2) - 0.5) / pow(1.0 + 0.001 * r, 2) + 0.5;
                }
                return y;
            });

        add("Step", constant(2, -100), constant(2, 100), 0.5, {constant(2, 0.5)},
            [](const Vec& x) {
                return (x.array().floor() + 0.5).square().sum();
            });

        // Stochastic and XinSheYang01 draw fresh random weights on every
        // evaluation, as in the Python source (numpy's global RNG)
        add("Stochastic", constant(2, -5), constant(2, 5), 0.0,
            {vec({1.0, 0.5})},
            [](const Vec& x) {
                static thread_local std::mt19937_64 rng(20130226);
                std::uniform_real_distribution<double> u(0.0, 1.0);
                double y = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    y += u(rng) * abs(x(i) - 1.0 / (i + 1.0));
                }
                return y;
            });

        add("StretchedV", constant(2, -10), constant(2, 10), 0.0,
            {vec({-9.38723188, 9.34026753})},
            [](const Vec& x) {
                double y = 0;
                for (Index i = 0; i + 1 < x.size(); ++i) {
                    const double t = x(i + 1) * x(i + 1) + x(i) * x(i);
                    y += pow(t, 0.25) * pow(sin(50.0 * pow(t, 0.1) + 1.0), 2);
                }
                return y;
            });

        add("Trefethen", constant(2, -10), constant(2, 10), -3.3068686474,
            {vec({-0.02440307923, 0.2106124261})},
            [](const Vec& x) {
                return exp(sin(50.0 * x(0))) + sin(60.0 * exp(x(1))) + sin(70.0 * sin(x(0)))
                       + sin(sin(80.0 * x(1))) - sin(10.0 * (x(0) + x(1)))
                       + 0.25 * (x(0) * x(0) + x(1) * x(1));
            });

        add("Trigonometric01", constant(2, 0), constant(2, pi), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                const double n = static_cast<double>(x.size());
                double inner = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    inner += cos(x(i)) + (i + 1.0) * (1.0 - cos(x(i)) - sin(x(i)));
                }
                return pow(n - inner, 2);
            });

        add("Trigonometric02", constant(2, -500), constant(2, 500), 1.0,
            {constant(2, 0.9)},
            [](const Vec& x) {
                double y = 1.0;
                for (Index i = 0; i < x.size(); ++i) {
                    const double t = pow(x(i) - 0.9, 2);
                    y += 8.0 * pow(sin(7.0 * t), 2) + 6.0 * pow(sin(14.0 * t), 2) + t;
                }
                return y;
            });

        add("Tripod", constant(2, -100), constant(2, 100), 0.0, {vec({0.0, -50.0})},
            [](const Vec& x) {
                const double p1 = x(0) >= 0 ? 1.0 : 0.0;
                const double p2 = x(1) >= 0 ? 1.0 : 0.0;
                return p2 * (1.0 + p1) + abs(x(0) + 50.0 * p2 * (1.0 - 2.0 * p1))
                       + abs(x(1) + 50.0 * (1.0 - 2.0 * p2));
            });

        add("Ursem03", vec({-2, -1.5}), vec({2, 1.5}), -3.0, {constant(2, 0)},
            [](const Vec& x) {
                auto term = [](double v) {
                    return -sin(2.2 * pi * v + 0.5 * pi)
                           * ((2.0 - abs(v)) / 2.0) * ((3.0 - abs(v)) / 2.0);
                };
                return term(x(0)) + term(x(1));
            });

        add("Ursem04", constant(2, -2), constant(2, 2), -1.5, {constant(2, 0)},
            [](const Vec& x) {
                return -3.0 * sin(0.5 * pi * x(0) + 0.5 * pi)
                       * (2.0 - sqrt(x(0) * x(0) + x(1) * x(1))) / 4.0;
            });

        add("UrsemWaves", vec({-0.9, -1.2}), vec({1.2, 1.2}), -8.5536, {constant(2, 1.2)},
            [](const Vec& x) {
                return -0.9 * x(0) * x(0)
                       + (x(1) * x(1) - 4.5 * x(1) * x(1)) * x(0) * x(1)
                       + 4.7 * cos(3.0 * x(0) - x(1) * x(1) * (2.0 + x(0)))
                         * sin(2.5 * pi * x(0));
            });

        add("Vincent", constant(2, 0.25), constant(2, 10), -2.0,
            {constant(2, 7.70628098)},
            [](const Vec& x) {
                return -(10.0 * x.array().log()).sin().sum();
            });

        add("Watson", constant(6, -5), constant(6, 5), 0.002288,
            {vec({-0.0158, 1.012, -0.2329, 1.260, -1.513, 0.9928})},
            [](const Vec& x) {
                const Index n = x.size();
                double y = 0;
                for (int m = 0; m < 29; ++m) {
                    const double t = (m + 1.0) / 29.0;
                    double s1 = 0, dx = 1.0;
                    for (Index j = 1; j < n; ++j) {
                        s1 += j * dx * x(j);
                        dx *= t;
                    }
                    double s2 = 0;
                    dx = 1.0;
                    for (Index j = 0; j < n; ++j) {
                        s2 += dx * x(j);
                        dx *= t;
                    }
                    y += pow(s1 - s2 * s2 - 1.0, 2);
                }
                y += x(0) * x(0);
                y += pow(x(1) - x(0) * x(0) - 1.0, 2);
                return y;
            });

        add("Wavy", constant(2, -pi), constant(2, pi), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                const double n = static_cast<double>(x.size());
                double s = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    s += cos(10.0 * x(i)) * exp(-x(i) * x(i) / 2.0);
                }
                return 1.0 - s / n;
            });

        add("WayburnSeader02", constant(2, -500), constant(2, 500), 0.0,
            {vec({0.2, 1.0})},
            [](const Vec& x) {
                return pow(1.613 - 4.0 * pow(x(0) - 0.3125, 2) - 4.0 * pow(x(1) - 1.625, 2), 2)
                       + pow(x(1) - 1.0, 2);
            });

        add("Weierstrass", constant(2, -0.5), constant(2, 0.5), 4.0, {constant(2, 0)},
            [](const Vec& x) {
                const int kmax = 20;
                const double a = 0.5, b = 3.0;
                const double n = static_cast<double>(x.size());
                double y = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    double s1 = 0, s2 = 0;
                    for (int k = 0; k <= kmax; ++k) {
                        const double ak = pow(a, k);
                        const double bk = pow(b, k);
                        s1 += ak * cos(2.0 * pi * bk * (x(i) + 0.5));
                        s2 += ak * cos(pi * bk);
                    }
                    y += s1 - n * s2;
                }
                return y;
            });

        add("Whitley", constant(2, -10.24), constant(2, 10.24), 0.0, {constant(2, 1)},
            [](const Vec& x) {
                const Index n = x.size();
                double y = 0;
                for (Index i = 0; i < n; ++i) {
                    for (Index j = 0; j < n; ++j) {
                        const double temp = 100.0 * (x(i) * x(i) - x(j)) + pow(1.0 - x(j), 2);
                        y += temp * temp / 4000.0 - cos(temp) + 1.0;
                    }
                }
                return y;
            });

        add("XinSheYang01", constant(2, -5), constant(2, 5), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                static thread_local std::mt19937_64 rng(20130226);
                std::uniform_real_distribution<double> u(0.0, 1.0);
                double y = 0;
                for (Index i = 0; i < x.size(); ++i) {
                    y += u(rng) * pow(abs(x(i)), i + 1.0);
                }
                return y;
            });

        add("XinSheYang03", constant(2, -20), constant(2, 20), -1.0, {constant(2, 0)},
            [](const Vec& x) {
                const double beta = 15.0, m = 5.0;
                return exp(-(x.array() / beta).pow(2.0 * m).sum())
                       - 2.0 * exp(-x.squaredNorm()) * x.array().cos().square().prod();
            });

        add("XinSheYang04", constant(2, -10), constant(2, 10), -1.0, {constant(2, 0)},
            [](const Vec& x) {
                return (x.array().sin().square().sum() - exp(-x.squaredNorm()))
                       * exp(-x.array().abs().sqrt().sin().square().sum());
            });

        add("Xor", constant(9, -1), constant(9, 1), 0.9597588,
            {vec({1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 0.421134})},
            [](const Vec& x) {
                const double f11 = x(6) / (1.0 + exp(-x(0) - x(1) - x(4)));
                const double f12 = x(7) / (1.0 + exp(-x(2) - x(3) - x(5)));
                const double f1 = pow(1.0 + exp(-f11 - f12 - x(8)), -2);
                const double f21 = x(6) / (1.0 + exp(-x(4)));
                const double f22 = x(7) / (1.0 + exp(-x(5)));
                const double f2 = pow(1.0 + exp(-f21 - f22 - x(8)), -2);
                const double f31 = x(6) / (1.0 + exp(-x(0) - x(4)));
                const double f32 = x(7) / (1.0 + exp(-x(2) - x(5)));
                const double f3 = pow(1.0 - pow(1.0 + exp(-f31 - f32 - x(8)), -1), 2);
                const double f41 = x(6) / (1.0 + exp(-x(1) - x(4)));
                const double f42 = x(7) / (1.0 + exp(-x(3) - x(5)));
                const double f4 = pow(1.0 - pow(1.0 + exp(-f41 - f42 - x(8)), -1), 2);
                return f1 + f2 + f3 + f4;
            });

        add("YaoLiu09", constant(2, -5.12), constant(2, 5.12), 0.0, {constant(2, 0)},
            [](const Vec& x) {
                return (x.array().square() - 10.0 * (2.0 * pi * x.array()).cos() + 10.0).sum();
            });

        add("ZeroSum", constant(2, -10), constant(2, 10), 0.0, {},
            [](const Vec& x) {
                const double s = x.sum();
                if (abs(s) < 3e-16) {
                    return 0.0;
                }
                return 1.0 + pow(10000.0 * abs(s), 0.5);
            });

        add("Zimmerman", constant(2, 0), constant(2, 100), 0.0, {vec({7.0, 2.0})},
            [](const Vec& x) {
                auto sgn = [](double v) { return static_cast<double>((v > 0) - (v < 0)); };
                auto zp = [](double v) { return 100.0 * (1.0 + v); };
                const double zh1 = 9.0 - x(0) - x(1);
                const double zh2 = pow(x(0) - 3.0, 2) + pow(x(1) - 2.0, 2) - 16.0;
                const double zh3 = x(0) * x(1) - 14.0;
                double m = zh1;
                m = std::max(m, zp(zh2) * sgn(zh2));
                m = std::max(m, zp(zh3) * sgn(zh3));
                m = std::max(m, zp(-x(0)) * sgn(x(0)));
                m = std::max(m, zp(-x(1)) * sgn(x(1)));
                return m;
            });

        // univariate problems

        add("Problem02", vec({2.7}), vec({7.5}), -1.899599, {vec({5.145735})},
            [](const Vec& x) {
                return sin(x(0)) + sin(10.0 / 3.0 * x(0));
            });

        add("Problem03", vec({-10}), vec({10}), -12.03124, {vec({-6.7745761})},
            [](const Vec& x) {
                double y = 0;
                for (int k = 1; k <= 5; ++k) {
                    y += k * sin((k + 1.0) * x(0) + k);
                }
                return -y;
            });

        add("Problem04", vec({1.9}), vec({3.9}), -3.85045, {vec({2.868034})},
            [](const Vec& x) {
                return -(16.0 * x(0) * x(0) - 24.0 * x(0) + 5.0) * exp(-x(0));
            });

        add("Problem05", vec({0.0}), vec({1.2}), -1.48907, {vec({0.96609})},
            [](const Vec& x) {
                return -(1.4 - 3.0 * x(0)) * sin(18.0 * x(0));
            });

        add("Problem06", vec({-10}), vec({10}), -0.824239, {vec({0.67956})},
            [](const Vec& x) {
                return -(x(0) + sin(x(0))) * exp(-x(0) * x(0));
            });

        add("Problem07", vec({2.7}), vec({7.5}), -1.6013, {vec({5.19978})},
            [](const Vec& x) {
                return sin(x(0)) + sin(10.0 / 3.0 * x(0)) + log(x(0)) - 0.84 * x(0) + 3.0;
            });

        add("Problem08", vec({-10}), vec({10}), -14.508, {vec({-7.083506})},
            [](const Vec& x) {
                double y = 0;
                for (int k = 1; k <= 5; ++k) {
                    y += k * cos((k + 1.0) * x(0) + k);
                }
                return -y;
            });

        add("Problem09", vec({3.1}), vec({20.4}), -1.90596, {vec({17.039})},
            [](const Vec& x) {
                return sin(x(0)) + sin(2.0 / 3.0 * x(0));
            });

        add("Problem10", vec({0}), vec({10}), -7.916727, {vec({7.9787})},
            [](const Vec& x) {
                return -x(0) * sin(x(0));
            });

        add("Problem11", vec({-pi / 2}), vec({2 * pi}), -1.5, {vec({2.09439})},
            [](const Vec& x) {
                return 2.0 * cos(x(0)) + cos(2.0 * x(0));
            });

        add("Problem12", vec({0}), vec({2 * pi}), -1.0, {vec({pi})},
            [](const Vec& x) {
                return pow(sin(x(0)), 3) + pow(cos(x(0)), 3);
            });

        add("Problem13", vec({0.001}), vec({0.99}), -1.5874, {vec({1.0 / sqrt(2.0)})},
            [](const Vec& x) {
                return -pow(x(0), 2.0 / 3.0) - pow(1.0 - x(0) * x(0), 1.0 / 3.0);
            });

        add("Problem14", vec({0}), vec({4}), -0.788685, {vec({0.224885})},
            [](const Vec& x) {
                return -exp(-x(0)) * sin(2.0 * pi * x(0));
            });

        add("Problem15", vec({-5}), vec({5}), -0.03553, {vec({2.41422})},
            [](const Vec& x) {
                return -(-x(0) * x(0) + 5.0 * x(0) - 6.0) / (x(0) * x(0) + 1.0);
            });

        add("Problem18", vec({0}), vec({6}), 0.0, {vec({2.0})},
            [](const Vec& x) {
                if (x(0) <= 3.0) {
                    return pow(x(0) - 2.0, 2);
                }
                return 2.0 * log(x(0) - 2.0) + 1.0;
            });

        add("Problem20", vec({-10}), vec({10}), -0.0634905, {vec({1.195137})},
            [](const Vec& x) {
                return -(x(0) - sin(x(0))) * exp(-x(0) * x(0));
            });

        add("Problem21", vec({0}), vec({10}), -9.50835, {vec({4.79507})},
            [](const Vec& x) {
                return x(0) * sin(x(0)) + x(0) * cos(2.0 * x(0));
            });

        add("Problem22", vec({0}), vec({20}), exp(-27.0 * pi / 2.0) - 1.0,
            {vec({9.0 * pi / 2.0})},
            [](const Vec& x) {
                return exp(-3.0 * x(0)) - pow(sin(x(0)), 3);
            });

        return ps;
    }();

    return problems;
}

/// Look up a single problem by name.
inline const Problem& problem(const std::string& name)
{
    for (const Problem& p : allProblems()) {
        if (p.name == name) {
            return p;
        }
    }
    throw std::invalid_argument("globopt benchmarks: unknown problem '" + name + "'");
}

} // namespace benchmarks
} // namespace globopt

#endif
