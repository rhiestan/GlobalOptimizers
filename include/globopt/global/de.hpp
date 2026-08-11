// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Differential Evolution, optionally with jDE self-adaptation. Implemented
// from the published algorithm description rather than ported from an
// existing library:
//
//   R. Storn, K. Price, "Differential Evolution - A Simple and Efficient
//   Heuristic for Global Optimization over Continuous Spaces", Journal of
//   Global Optimization 11, 1997 (the algorithm and the strategy naming).
//   K. Price, R. Storn, J. Lampinen, "Differential Evolution: A Practical
//   Approach to Global Optimization", Springer, 2005 (bound handling).
//   J. Brest et al., "Self-Adapting Control Parameters in Differential
//   Evolution", IEEE Transactions on Evolutionary Computation 10(6), 2006
//   (the jDE self-adaptation of F and CR).
//
// Design notes:
//   - Generations are synchronous, as in the original: every mutant for a
//     generation is built from the population as it stood at the start of it.
//   - jDE is on by default. It adapts F and CR per individual and removes most
//     of the tuning burden; "differential_weight" and "crossover_probability"
//     then act as the initial values rather than fixed constants.
//   - Finite box bounds are required, because the population is initialized by
//     sampling the box (as for LIPO and EGO).

#ifndef GLOBOPT_GLOBAL_DE_HPP
#define GLOBOPT_GLOBAL_DE_HPP

#include "../core/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace globopt {
namespace detail {
namespace de {

/// Mutation base vector and number of difference vectors.
enum class Mutation { Rand1, Best1, Rand2, Best2, CurrentToBest1 };
enum class Crossover { Binomial, Exponential };
enum class BoundHandling { Reflect, Clip, Random };

/// Parses a strategy string such as "rand/1/bin" or "best/2/exp".
inline bool parseStrategy(const std::string& name, Mutation& mutation, Crossover& crossover)
{
    const std::size_t first = name.find('/');
    const std::size_t second = (first == std::string::npos)
        ? std::string::npos
        : name.find('/', first + 1);
    if (first == std::string::npos || second == std::string::npos) {
        return false;
    }
    const std::string base = name.substr(0, first);
    const std::string count = name.substr(first + 1, second - first - 1);
    const std::string cross = name.substr(second + 1);

    if (base == "rand" && count == "1") {
        mutation = Mutation::Rand1;
    } else if (base == "best" && count == "1") {
        mutation = Mutation::Best1;
    } else if (base == "rand" && count == "2") {
        mutation = Mutation::Rand2;
    } else if (base == "best" && count == "2") {
        mutation = Mutation::Best2;
    } else if ((base == "current-to-best" || base == "currenttobest") && count == "1") {
        mutation = Mutation::CurrentToBest1;
    } else {
        return false;
    }

    if (cross == "bin") {
        crossover = Crossover::Binomial;
    } else if (cross == "exp") {
        crossover = Crossover::Exponential;
    } else {
        return false;
    }
    return true;
}

inline bool parseBoundHandling(const std::string& name, BoundHandling& out)
{
    if (name == "reflect") { out = BoundHandling::Reflect; return true; }
    if (name == "clip") { out = BoundHandling::Clip; return true; }
    if (name == "random") { out = BoundHandling::Random; return true; }
    return false;
}

/// Number of distinct donor indices a mutation strategy needs, excluding the
/// target (and excluding the best, which is taken from the population).
inline Index donorsRequired(const Mutation mutation)
{
    switch (mutation) {
        case Mutation::Rand1: return 3;
        case Mutation::Best1: return 2;
        case Mutation::Rand2: return 5;
        case Mutation::Best2: return 4;
        case Mutation::CurrentToBest1: return 2;
    }
    return 3;
}

} // namespace de
} // namespace detail

/// Differential Evolution (Storn & Price) with optional jDE self-adaptation.
/// Derivative-free; requires finite box bounds via setBounds(). The initial
/// point passed to run() is clamped into the box and seeded as the first
/// member of the population. If "target_objective" is set, the search stops
/// early with Status::Success once it is reached within "tolerance".
template <typename Scalar>
class DifferentialEvolution : public Optimizer<Scalar> {
public:
    using typename Optimizer<Scalar>::ObjectiveFn;

    DifferentialEvolution()
    {
        this->registerParam("max_function_evaluations", &m_maxFunctionEvaluations,
                            "maximum number of function evaluations (0 = max(1000, 200*n))");
        this->registerParam("population_size", &m_populationSize,
                            "number of individuals (0 = max(20, n))");
        this->registerParam("strategy", &m_strategy,
                            "mutation and crossover, e.g. rand/1/bin, best/1/bin, rand/2/bin, "
                            "best/2/bin or current-to-best/1/bin, with bin or exp crossover");
        this->registerParam("differential_weight", &m_differentialWeight,
                            "scaling factor F applied to the difference vectors "
                            "(the initial value when self_adaptive is on)");
        this->registerParam("crossover_probability", &m_crossoverProbability,
                            "crossover probability CR "
                            "(the initial value when self_adaptive is on)");
        this->registerParam("self_adaptive", &m_selfAdaptive,
                            "adapt F and CR per individual with the jDE scheme");
        this->registerParam("bound_handling", &m_boundHandling,
                            "how to repair a mutant outside the box: reflect, clip or random");
        this->registerParam("tolerance_function", &m_toleranceFunction,
                            "stop when the objective spread across the population falls below "
                            "this, relative to the best value");
        this->registerParam("target_objective", &m_targetObjective,
                            "value of the global optimum, if known (-inf to disable)");
        this->registerParam("tolerance", &m_tolerance,
                            "stop when the best objective is within this distance of target_objective");
        this->registerParam("seed", &m_seed,
                            "random seed (0 = non-deterministic)");
    }

    const char* name() const override { return "DE"; }

protected:
    Result<Scalar> doOptimize(const ObjectiveFn& objective, const Vector<Scalar>& initialPoint) override
    {
        namespace de = detail::de;

        Result<Scalar> result;
        result.x = initialPoint;
        result.gradientNorm = std::numeric_limits<Scalar>::quiet_NaN();

        if (!this->m_boundsSet
            || !this->m_lowerBounds.allFinite() || !this->m_upperBounds.allFinite()) {
            result.status = Status::InvalidInput;
            result.message = "DE requires finite lower and upper bounds (setBounds)";
            return result;
        }
        for (Index i = 0; i < this->m_lowerBounds.size(); ++i) {
            if (this->m_lowerBounds(i) >= this->m_upperBounds(i)) {
                result.status = Status::InvalidInput;
                result.message = "DE requires lower(i) < upper(i) for every variable";
                return result;
            }
        }

        de::Mutation mutation;
        de::Crossover crossover;
        de::BoundHandling boundHandling;
        if (!de::parseStrategy(m_strategy, mutation, crossover)) {
            result.status = Status::InvalidInput;
            result.message = "DE: unknown strategy '" + m_strategy
                + "' (use e.g. rand/1/bin, best/1/bin, rand/2/bin, best/2/bin or "
                  "current-to-best/1/bin)";
            return result;
        }
        if (!de::parseBoundHandling(m_boundHandling, boundHandling)) {
            result.status = Status::InvalidInput;
            result.message = "DE: unknown bound_handling '" + m_boundHandling
                + "' (use reflect, clip or random)";
            return result;
        }

        const Vector<Scalar>& lower = this->m_lowerBounds;
        const Vector<Scalar>& upper = this->m_upperBounds;
        const Index dims = initialPoint.size();

        const std::size_t budget = (m_maxFunctionEvaluations > 0)
            ? m_maxFunctionEvaluations
            : std::max<std::size_t>(1000, 200 * static_cast<std::size_t>(dims));

        // The population must be large enough for the strategy's donors plus
        // the target itself.
        const Index minimumPopulation = de::donorsRequired(mutation) + 1;
        // Storn and Price suggest 5n to 10n, which assumes a budget large
        // enough to evolve such a population. At the budgets this library is
        // benchmarked with, a much smaller population converges considerably
        // more often (measured across n = 2 to 40; see the README).
        Index population = (m_populationSize > 0)
            ? static_cast<Index>(m_populationSize)
            : std::max<Index>(20, dims);
        population = std::max(population, minimumPopulation);
        population = std::min<Index>(population, static_cast<Index>(budget));

        std::mt19937_64 rng(m_seed != 0 ? static_cast<std::uint64_t>(m_seed) : std::random_device{}());
        std::uniform_real_distribution<Scalar> unit(Scalar(0), Scalar(1));

        std::size_t& evaluations = result.functionEvaluations;
        Scalar bestValue = detail::inf<Scalar>();
        Vector<Scalar> bestPoint = initialPoint.cwiseMax(lower).cwiseMin(upper);

        auto randomPoint = [&]() {
            Vector<Scalar> x(dims);
            for (Index i = 0; i < dims; ++i) {
                x(i) = lower(i) + (upper(i) - lower(i)) * unit(rng);
            }
            return x;
        };

        auto reachedTarget = [&]() { return bestValue < m_targetObjective + m_tolerance; };

        // -- initial population ----------------------------------------------
        std::vector<Vector<Scalar>> xs(static_cast<std::size_t>(population));
        std::vector<Scalar> fs(static_cast<std::size_t>(population));
        std::vector<Scalar> weights(static_cast<std::size_t>(population), m_differentialWeight);
        std::vector<Scalar> crossovers(static_cast<std::size_t>(population), m_crossoverProbability);

        std::size_t generations = 0;
        Status status = Status::MaxFunctionEvaluationsReached;
        std::string reason = toString(Status::MaxFunctionEvaluationsReached);

        auto finish = [&](const Status finalStatus, const std::string& finalReason) {
            result.x = bestPoint;
            result.fval = bestValue;
            result.iterations = generations;
            result.status = reachedTarget() ? Status::Success : finalStatus;
            result.message = (result.status == Status::Success
                                  ? "optimization terminated successfully"
                                  : finalReason)
                + " (" + std::to_string(generations) + " generations, population "
                + std::to_string(population) + ")";
            return result;
        };

        for (Index i = 0; i < population; ++i) {
            // the caller's point seeds the population, the rest is uniform
            xs[static_cast<std::size_t>(i)] = (i == 0) ? bestPoint : randomPoint();
            fs[static_cast<std::size_t>(i)] = objective(xs[static_cast<std::size_t>(i)], nullptr);
            ++evaluations;
            if (fs[static_cast<std::size_t>(i)] < bestValue) {
                bestValue = fs[static_cast<std::size_t>(i)];
                bestPoint = xs[static_cast<std::size_t>(i)];
            }
            if (reachedTarget()) {
                return finish(Status::Success, "optimization terminated successfully");
            }
            if (evaluations >= budget) {
                return finish(Status::MaxFunctionEvaluationsReached,
                              toString(Status::MaxFunctionEvaluationsReached));
            }
        }

        // -- generations ------------------------------------------------------
        std::vector<Index> donors(static_cast<std::size_t>(de::donorsRequired(mutation)));
        Vector<Scalar> mutant(dims);
        Vector<Scalar> trial(dims);

        while (evaluations < budget && !reachedTarget()) {
            std::vector<Vector<Scalar>> nextXs = xs;
            std::vector<Scalar> nextFs = fs;
            std::vector<Scalar> nextWeights = weights;
            std::vector<Scalar> nextCrossovers = crossovers;

            for (Index i = 0; i < population && evaluations < budget; ++i) {
                const std::size_t self = static_cast<std::size_t>(i);

                // jDE: propose the control parameters this trial will use, and
                // keep them only if the trial survives selection.
                Scalar f = weights[self];
                Scalar cr = crossovers[self];
                if (m_selfAdaptive) {
                    if (unit(rng) < Scalar(0.1)) {
                        f = Scalar(0.1) + Scalar(0.9) * unit(rng);
                    }
                    if (unit(rng) < Scalar(0.1)) {
                        cr = unit(rng);
                    }
                }

                // distinct donor indices, all different from the target
                for (std::size_t d = 0; d < donors.size(); ++d) {
                    Index candidate = 0;
                    bool duplicate = true;
                    while (duplicate) {
                        candidate = static_cast<Index>(
                            std::uniform_int_distribution<long long>(0, population - 1)(rng));
                        duplicate = (candidate == i);
                        for (std::size_t e = 0; e < d && !duplicate; ++e) {
                            duplicate = (donors[e] == candidate);
                        }
                    }
                    donors[d] = candidate;
                }

                // -- mutation --------------------------------------------------
                const auto& xr0 = xs[static_cast<std::size_t>(donors[0])];
                const auto& xr1 = xs[static_cast<std::size_t>(donors[1])];
                switch (mutation) {
                    case de::Mutation::Rand1:
                        mutant = xr0 + f * (xr1 - xs[static_cast<std::size_t>(donors[2])]);
                        break;
                    case de::Mutation::Best1:
                        mutant = bestPoint + f * (xr0 - xr1);
                        break;
                    case de::Mutation::Rand2:
                        mutant = xr0
                            + f * (xr1 - xs[static_cast<std::size_t>(donors[2])])
                            + f * (xs[static_cast<std::size_t>(donors[3])]
                                   - xs[static_cast<std::size_t>(donors[4])]);
                        break;
                    case de::Mutation::Best2:
                        mutant = bestPoint + f * (xr0 - xr1)
                            + f * (xs[static_cast<std::size_t>(donors[2])]
                                   - xs[static_cast<std::size_t>(donors[3])]);
                        break;
                    case de::Mutation::CurrentToBest1:
                        mutant = xs[self] + f * (bestPoint - xs[self]) + f * (xr0 - xr1);
                        break;
                }

                // -- crossover -------------------------------------------------
                trial = xs[self];
                const Index forced = static_cast<Index>(
                    std::uniform_int_distribution<long long>(0, dims - 1)(rng));
                if (crossover == de::Crossover::Binomial) {
                    for (Index k = 0; k < dims; ++k) {
                        if (k == forced || unit(rng) < cr) {
                            trial(k) = mutant(k);
                        }
                    }
                } else {
                    // exponential: copy a contiguous (wrapping) run of components
                    Index k = forced;
                    for (Index taken = 0; taken < dims; ++taken) {
                        trial(k) = mutant(k);
                        k = (k + 1) % dims;
                        if (unit(rng) >= cr) {
                            break;
                        }
                    }
                }

                // -- keep the trial inside the box ------------------------------
                for (Index k = 0; k < dims; ++k) {
                    if (trial(k) >= lower(k) && trial(k) <= upper(k)) {
                        continue;
                    }
                    switch (boundHandling) {
                        case de::BoundHandling::Clip:
                            trial(k) = std::min(std::max(trial(k), lower(k)), upper(k));
                            break;
                        case de::BoundHandling::Random:
                            trial(k) = lower(k) + (upper(k) - lower(k)) * unit(rng);
                            break;
                        case de::BoundHandling::Reflect: {
                            // fold back into the box, repeating for points that
                            // overshoot the opposite bound
                            const Scalar span = upper(k) - lower(k);
                            Scalar value = trial(k);
                            for (int guard = 0; guard < 16
                                 && (value < lower(k) || value > upper(k)); ++guard) {
                                if (value < lower(k)) {
                                    value = lower(k) + (lower(k) - value);
                                } else {
                                    value = upper(k) - (value - upper(k));
                                }
                            }
                            if (value < lower(k) || value > upper(k)) {
                                value = lower(k) + span * unit(rng);
                            }
                            trial(k) = value;
                            break;
                        }
                    }
                }

                // -- selection ---------------------------------------------------
                const Scalar trialValue = objective(trial, nullptr);
                ++evaluations;

                if (trialValue <= fs[self]) {
                    nextXs[self] = trial;
                    nextFs[self] = trialValue;
                    nextWeights[self] = f;
                    nextCrossovers[self] = cr;
                    if (trialValue < bestValue) {
                        bestValue = trialValue;
                        bestPoint = trial;
                    }
                }
                if (reachedTarget()) {
                    return finish(Status::Success, "optimization terminated successfully");
                }
            }

            xs.swap(nextXs);
            fs.swap(nextFs);
            weights.swap(nextWeights);
            crossovers.swap(nextCrossovers);
            ++generations;

            // -- convergence: the population has collapsed onto one value ------
            const auto extremes = std::minmax_element(fs.begin(), fs.end());
            const Scalar spread = *extremes.second - *extremes.first;
            if (spread <= m_toleranceFunction * std::max(Scalar(1), std::abs(*extremes.first))) {
                status = Status::Stalled;
                reason = "the population converged without reaching the target";
                break;
            }
        }

        return finish(status, reason);
    }

private:
    std::size_t m_maxFunctionEvaluations = 0;
    std::size_t m_populationSize = 0;
    std::string m_strategy = "rand/1/bin";
    Scalar m_differentialWeight = Scalar(0.8);
    Scalar m_crossoverProbability = Scalar(0.9);
    bool m_selfAdaptive = true;
    std::string m_boundHandling = "reflect";
    Scalar m_toleranceFunction = Scalar(1e-12);
    Scalar m_targetObjective = -detail::inf<Scalar>();
    Scalar m_tolerance = Scalar(1e-05);
    long long m_seed = 0;
};

} // namespace globopt

#endif
