// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Runs a global optimizer over the ported go_benchmark.py suite.
//
//   run_ampgo_benchmarks [seed] [optimizer]
//
// optimizer is "AMPGO" (default, 20000-evaluation budget, mirroring the
// driver in go_amp.py's __main__), "LIPO" (500-evaluation budget: MaxLIPO+TR
// targets expensive objectives and spends much more model work per
// evaluation) or "EGO" (200-evaluation budget: Bayesian optimization spends
// even more model work per evaluation).

#include <globopt/globopt.hpp>
#include <globopt/benchmarks/go_benchmark.hpp>

#include <cstdio>
#include <random>
#include <string>

int main(int argc, char** argv)
{
    const unsigned long long seed = (argc > 1) ? std::stoull(argv[1]) : 12345ULL;
    const std::string optimizerName = (argc > 2) ? argv[2] : "AMPGO";
    const bool isLipo = (optimizerName == "LIPO" || optimizerName == "lipo");
    const bool isEgo = (optimizerName == "EGO" || optimizerName == "ego");
    const bool derivativeFree = isLipo || isEgo;
    const int budget = isLipo ? 500 : (isEgo ? 200 : 20000);

    std::mt19937_64 rng(seed);

    int solved = 0, total = 0;
    long long totalEvals = 0;

    std::printf("optimizer: %s, budget: %d evaluations per problem\n\n", optimizerName.c_str(), budget);
    std::printf("%-30s %3s  %14s  %14s  %8s  %-9s\n",
                "problem", "n", "fglob", "best f", "fevals", "status");
    std::printf("%s\n", std::string(88, '-').c_str());

    for (const auto& problem : globopt::benchmarks::allProblems()) {
        ++total;

        auto opt = globopt::OptimizerFactory<double>::create(optimizerName);
        opt->setBounds(problem.lower, problem.upper);
        opt->setParam("target_objective", problem.fglob);
        opt->setParam("tolerance", 1e-6);
        opt->setParam("max_function_evaluations", budget);
        opt->setParam("seed", static_cast<long long>(seed));
        if (!derivativeFree) {
            opt->setParam("total_iterations", 2000);
        }
        if (isEgo) {
            // lighter multistart settings than the library defaults (10/20)
            // to keep the 202-problem sweep tractable
            opt->setParam("hyperparameter_starts", 4);
            opt->setParam("acquisition_starts", 10);
        }

        // AMPGO needs gradients; LIPO and EGO are derivative-free
        globopt::ObjectiveFunction<double> objective;
        if (derivativeFree) {
            objective = [&problem](const globopt::Vector<double>& x, globopt::Vector<double>*) {
                return problem.objective(x);
            };
        } else {
            objective = globopt::withNumericalGradient<double>(problem.objective);
        }

        const auto x0 = problem.randomStart(rng);
        const auto res = opt->run(objective, x0);

        totalEvals += static_cast<long long>(res.functionEvaluations);
        if (res.success()) {
            ++solved;
        }

        std::printf("%-30s %3d  %14.6g  %14.6g  %8zu  %-9s\n",
                    problem.name.c_str(), static_cast<int>(problem.dimensions()),
                    problem.fglob, res.fval, res.functionEvaluations,
                    res.success() ? "solved" : toString(res.status));
        std::fflush(stdout);
    }

    std::printf("%s\n", std::string(88, '-').c_str());
    std::printf("solved %d of %d problems (seed %llu, avg fevals %lld)\n",
                solved, total, seed, totalEvals / total);

    return 0;
}
