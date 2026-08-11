// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Runs a global optimizer over the ported go_benchmark.py suite.
//
//   run_ampgo_benchmarks [seed] [optimizer] [options]
//
// optimizer is "AMPGO" (default, 20000-evaluation budget, mirroring the
// driver in go_amp.py's __main__), "LIPO" (500-evaluation budget: MaxLIPO+TR
// targets expensive objectives and spends much more model work per
// evaluation) or "EGO" (200-evaluation budget: Bayesian optimization spends
// even more model work per evaluation).
//
// Options (all optional, in any order after the positional arguments):
//   --budget N           evaluations per problem, overriding the default
//   --filter SUBSTR      only problems whose name contains SUBSTR (no case)
//   --max-dims N         skip problems with more than N variables
//   --stride N           take every Nth problem (a subset spread over the suite)
//   --limit N            stop after N problems have been run
//   --repeats N          run each problem with N consecutive seeds
//   --refit-interval N   EGO: re-estimate hyperparameters every N iterations
//   --list               print the selected problems and exit
//
// A per-problem wall-clock column makes the cost of the expensive optimizers
// visible; EGO in particular is dominated by its Kriging fits. Use --stride or
// --filter to get a quick read on a representative subset before committing to
// a full sweep, and build in Release: an -O0 build of this Eigen-heavy code is
// tens of times slower.

#include <globopt/globopt.hpp>
#include <globopt/benchmarks/go_benchmark.hpp>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

std::string lowercase(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

} // namespace

int main(int argc, char** argv)
{
    unsigned long long seed = 12345ULL;
    std::string optimizerName = "AMPGO";

    // positional: seed, then optimizer name
    int argi = 1;
    if (argi < argc && argv[argi][0] != '-') {
        seed = std::stoull(argv[argi++]);
    }
    if (argi < argc && argv[argi][0] != '-') {
        optimizerName = argv[argi++];
    }

    const bool isLipo = (lowercase(optimizerName) == "lipo");
    const bool isEgo = (lowercase(optimizerName) == "ego");
    const bool derivativeFree = isLipo || isEgo;

    int budget = isLipo ? 500 : (isEgo ? 200 : 20000);
    std::string filter;
    int maxDims = 0;    // 0 = no limit
    int stride = 1;
    int limit = 0;      // 0 = no limit
    int repeats = 1;
    int refitInterval = 0; // 0 = leave the optimizer default
    bool listOnly = false;

    for (; argi < argc; ++argi) {
        const std::string option = argv[argi];
        const bool hasValue = (argi + 1 < argc);
        auto value = [&]() -> std::string {
            if (!hasValue) {
                std::fprintf(stderr, "missing value for %s\n", option.c_str());
                std::exit(2);
            }
            return argv[++argi];
        };

        if (option == "--budget") {
            budget = std::stoi(value());
        } else if (option == "--filter") {
            filter = lowercase(value());
        } else if (option == "--max-dims") {
            maxDims = std::stoi(value());
        } else if (option == "--stride") {
            stride = std::max(1, std::stoi(value()));
        } else if (option == "--limit") {
            limit = std::stoi(value());
        } else if (option == "--repeats") {
            repeats = std::max(1, std::stoi(value()));
        } else if (option == "--refit-interval") {
            refitInterval = std::stoi(value());
        } else if (option == "--list") {
            listOnly = true;
        } else {
            std::fprintf(stderr, "unknown option '%s'\n", option.c_str());
            return 2;
        }
    }

    // -- problem selection ---------------------------------------------------
    std::vector<const globopt::benchmarks::Problem*> selected;
    {
        int matched = 0;
        for (const auto& problem : globopt::benchmarks::allProblems()) {
            if (!filter.empty() && lowercase(problem.name).find(filter) == std::string::npos) {
                continue;
            }
            if (maxDims > 0 && problem.dimensions() > maxDims) {
                continue;
            }
            if ((matched++ % stride) != 0) {
                continue;
            }
            selected.push_back(&problem);
            if (limit > 0 && static_cast<int>(selected.size()) >= limit) {
                break;
            }
        }
    }

    if (selected.empty()) {
        std::fprintf(stderr, "no problems selected\n");
        return 1;
    }

    if (listOnly) {
        for (const auto* problem : selected) {
            std::printf("%-30s n=%d\n", problem->name.c_str(),
                        static_cast<int>(problem->dimensions()));
        }
        std::printf("%zu problems selected\n", selected.size());
        return 0;
    }

    std::printf("optimizer: %s, budget: %d evaluations per problem", optimizerName.c_str(), budget);
    if (repeats > 1) {
        std::printf(", %d seeds per problem", repeats);
    }
    if (isEgo && refitInterval > 0) {
        std::printf(", refit interval: %d", refitInterval);
    }
    std::printf("\n%zu of %zu problems selected\n\n", selected.size(),
                globopt::benchmarks::allProblems().size());

    std::printf("%-30s %3s  %14s  %14s  %8s  %8s  %-9s\n",
                "problem", "n", "fglob", "best f", "fevals", "seconds", "status");
    std::printf("%s\n", std::string(98, '-').c_str());

    int solved = 0, close = 0, total = 0;
    long long totalEvals = 0;
    double totalSeconds = 0.0;

    // The strict 1e-6 target undersells optimizers that are meant to get close
    // in very few evaluations, so near misses are counted separately.
    const double closeTolerance = 1e-2;
    auto relativeGap = [](const double fval, const double fglob) {
        return std::abs(fval - fglob) / std::max(1.0, std::abs(fglob));
    };

    for (const auto* problem : selected) {
        for (int repeat = 0; repeat < repeats; ++repeat) {
            const unsigned long long runSeed = seed + static_cast<unsigned long long>(repeat);
            ++total;

            auto opt = globopt::OptimizerFactory<double>::create(optimizerName);
            opt->setBounds(problem->lower, problem->upper);
            opt->setParam("target_objective", problem->fglob);
            opt->setParam("tolerance", 1e-6);
            opt->setParam("max_function_evaluations", budget);
            opt->setParam("seed", static_cast<long long>(runSeed));
            if (!derivativeFree) {
                opt->setParam("total_iterations", 2000);
            }
            if (isEgo) {
                // lighter multistart settings than the library defaults (10/20)
                // to keep the 202-problem sweep tractable
                opt->setParam("hyperparameter_starts", 4);
                opt->setParam("acquisition_starts", 10);
                if (refitInterval > 0) {
                    opt->setParam("hyperparameter_refit_interval", refitInterval);
                }
            }

            // AMPGO needs gradients; LIPO and EGO are derivative-free
            globopt::ObjectiveFunction<double> objective;
            if (derivativeFree) {
                objective = [problem](const globopt::Vector<double>& x, globopt::Vector<double>*) {
                    return problem->objective(x);
                };
            } else {
                objective = globopt::withNumericalGradient<double>(problem->objective);
            }

            std::mt19937_64 rng(runSeed);
            const auto x0 = problem->randomStart(rng);

            const auto start = std::chrono::steady_clock::now();
            const auto res = opt->run(objective, x0);
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

            totalEvals += static_cast<long long>(res.functionEvaluations);
            totalSeconds += seconds;
            if (res.success()) {
                ++solved;
            }
            if (relativeGap(res.fval, problem->fglob) <= closeTolerance) {
                ++close;
            }

            std::printf("%-30s %3d  %14.6g  %14.6g  %8zu  %8.2f  %-9s\n",
                        problem->name.c_str(), static_cast<int>(problem->dimensions()),
                        problem->fglob, res.fval, res.functionEvaluations, seconds,
                        res.success() ? "solved" : toString(res.status));
            std::fflush(stdout);
        }
    }

    std::printf("%s\n", std::string(98, '-').c_str());
    std::printf("solved %d of %d runs (seed %llu, avg fevals %lld, %.1f s total, %.2f s per run)\n",
                solved, total, seed, totalEvals / total, totalSeconds, totalSeconds / total);
    std::printf("%d of %d runs within %g relative gap of fglob\n", close, total, closeTolerance);

    return 0;
}
