// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Runs AMPGO over the ported go_benchmark.py suite, mirroring the driver in
// go_amp.py's __main__: random start point within the bounds, target set to
// the known global optimum, budget of 20000 function evaluations.

#include <globopt/globopt.hpp>
#include <globopt/benchmarks/go_benchmark.hpp>

#include <cstdio>
#include <random>
#include <string>

int main(int argc, char** argv)
{
    const unsigned long long seed = (argc > 1) ? std::stoull(argv[1]) : 12345ULL;

    std::mt19937_64 rng(seed);

    int solved = 0, total = 0;
    long long totalEvals = 0;

    std::printf("%-30s %3s  %14s  %14s  %8s  %-9s\n",
                "problem", "n", "fglob", "best f", "fevals", "status");
    std::printf("%s\n", std::string(88, '-').c_str());

    for (const auto& problem : globopt::benchmarks::allProblems()) {
        ++total;

        auto objective = globopt::withNumericalGradient<double>(problem.objective);

        globopt::AMPGO<double> optimizer;
        optimizer.setBounds(problem.lower, problem.upper);
        optimizer.setParam("target_objective", problem.fglob);
        optimizer.setParam("tolerance", 1e-6);
        optimizer.setParam("max_function_evaluations", 20000);
        optimizer.setParam("total_iterations", 2000);
        optimizer.setParam("seed", static_cast<long long>(seed));

        const auto x0 = problem.randomStart(rng);
        const auto res = optimizer.run(objective, x0);

        totalEvals += static_cast<long long>(res.functionEvaluations);
        if (res.success()) {
            ++solved;
        }

        std::printf("%-30s %3d  %14.6g  %14.6g  %8zu  %-9s\n",
                    problem.name.c_str(), static_cast<int>(problem.dimensions()),
                    problem.fglob, res.fval, res.functionEvaluations,
                    res.success() ? "solved" : toString(res.status));
    }

    std::printf("%s\n", std::string(88, '-').c_str());
    std::printf("solved %d of %d problems (seed %llu, avg fevals %lld)\n",
                solved, total, seed, totalEvals / total);

    return 0;
}
