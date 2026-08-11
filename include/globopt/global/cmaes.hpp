// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with IPOP
// restarts. Implemented from the published algorithm description rather than
// ported from an existing library:
//
//   N. Hansen, "The CMA Evolution Strategy: A Tutorial", arXiv:1604.00772,
//   2016 (the reference formulation and the default strategy parameters).
//   N. Hansen, A. Ostermeier, "Completely Derandomized Self-Adaptation in
//   Evolution Strategies", Evolutionary Computation 9(2), 2001.
//   A. Auger, N. Hansen, "A Restart CMA Evolution Strategy With Increasing
//   Population Size", IEEE CEC 2005 (the IPOP restart scheme).
//
// Design notes:
//   - Plain CMA-ES is a local search: it converges to whichever basin the
//     initial distribution settles into. The IPOP restart scheme (restart on
//     convergence with a doubled population, from a fresh random mean) is what
//     makes it a global method, so it is on by default.
//   - The search runs in normalized coordinates. With box bounds the box maps
//     to [0, 1]^n, which keeps the initial distribution isotropic no matter how
//     anisotropic the box is; without bounds the coordinates are used as given.
//   - Bounds are handled by evaluating the objective at the point clamped into
//     the box and ranking by that value plus a quadratic penalty on the
//     distance outside it. The penalty is scaled by the spread of objective
//     values in the current generation, so it is invariant to a rescaling of
//     the objective and cannot swamp or vanish against it. The reported
//     solution is always feasible, and its fval is the true objective there.
//   - Weights are the standard positive ("mu") weights of the tutorial; the
//     active (negative weight) variant is not implemented.

#ifndef GLOBOPT_GLOBAL_CMAES_HPP
#define GLOBOPT_GLOBAL_CMAES_HPP

#include "../core/optimizer.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace globopt {

/// Covariance Matrix Adaptation Evolution Strategy with IPOP restarts.
/// Derivative-free. Box bounds are optional: without them the search is
/// unconstrained, with them the reported solution is always inside the box.
/// If "target_objective" is set, the search stops early with Status::Success
/// once it is reached within "tolerance".
template <typename Scalar>
class CMAES : public Optimizer<Scalar> {
public:
    using typename Optimizer<Scalar>::ObjectiveFn;

    CMAES()
    {
        this->registerParam("max_function_evaluations", &m_maxFunctionEvaluations,
                            "maximum number of function evaluations (0 = max(1000, 200*n))");
        this->registerParam("population_size", &m_populationSize,
                            "population size lambda (0 = 4 + floor(3*ln(n)))");
        this->registerParam("initial_step_size", &m_initialStepSize,
                            "initial step size sigma, in normalized coordinates "
                            "(0 = 0.3, i.e. about a third of the box)");
        this->registerParam("max_restarts", &m_maxRestarts,
                            "number of IPOP restarts after a converged run (0 = single run)");
        this->registerParam("population_increase_factor", &m_populationIncreaseFactor,
                            "factor applied to the population size on each restart (IPOP: 2)");
        this->registerParam("tolerance_function", &m_toleranceFunction,
                            "restart when the objective range within a generation and over the "
                            "recent history falls below this value");
        this->registerParam("tolerance_x", &m_toleranceX,
                            "restart when the search distribution becomes smaller than this "
                            "value in every coordinate");
        this->registerParam("max_condition_number", &m_maxConditionNumber,
                            "restart when the covariance matrix becomes worse conditioned than this");
        this->registerParam("boundary_penalty", &m_boundaryPenalty,
                            "weight of the quadratic penalty applied outside the box, relative "
                            "to the spread of objective values in the generation");
        this->registerParam("target_objective", &m_targetObjective,
                            "value of the global optimum, if known (-inf to disable)");
        this->registerParam("tolerance", &m_tolerance,
                            "stop when the best objective is within this distance of target_objective");
        this->registerParam("seed", &m_seed,
                            "random seed (0 = non-deterministic)");
    }

    const char* name() const override { return "CMA-ES"; }

protected:
    Result<Scalar> doOptimize(const ObjectiveFn& objective, const Vector<Scalar>& initialPoint) override
    {
        Result<Scalar> result;
        result.x = initialPoint;
        result.gradientNorm = std::numeric_limits<Scalar>::quiet_NaN();

        const Index dims = initialPoint.size();
        const bool bounded = this->m_boundsSet;

        if (bounded) {
            if (!this->m_lowerBounds.allFinite() || !this->m_upperBounds.allFinite()) {
                result.status = Status::InvalidInput;
                result.message = "CMA-ES requires finite bounds when bounds are set";
                return result;
            }
            for (Index i = 0; i < dims; ++i) {
                if (this->m_lowerBounds(i) >= this->m_upperBounds(i)) {
                    result.status = Status::InvalidInput;
                    result.message = "CMA-ES requires lower(i) < upper(i) for every variable";
                    return result;
                }
            }
        }
        if (m_populationIncreaseFactor < Scalar(1)) {
            result.status = Status::InvalidInput;
            result.message = "CMA-ES requires population_increase_factor >= 1";
            return result;
        }

        // -- normalized coordinates: x = offset + scale .* y ------------------
        const Vector<Scalar> offset = bounded
            ? this->m_lowerBounds
            : Vector<Scalar>(Vector<Scalar>::Zero(dims));
        const Vector<Scalar> scale = bounded
            ? Vector<Scalar>(this->m_upperBounds - this->m_lowerBounds)
            : Vector<Scalar>(Vector<Scalar>::Ones(dims));

        auto toX = [&](const Vector<Scalar>& y) {
            return (offset + scale.cwiseProduct(y)).eval();
        };
        auto clampY = [&](const Vector<Scalar>& y) {
            return bounded ? y.cwiseMax(Scalar(0)).cwiseMin(Scalar(1)).eval() : y;
        };

        const std::size_t budget = (m_maxFunctionEvaluations > 0)
            ? m_maxFunctionEvaluations
            : std::max<std::size_t>(1000, 200 * static_cast<std::size_t>(dims));

        std::mt19937_64 rng(m_seed != 0 ? static_cast<std::uint64_t>(m_seed) : std::random_device{}());
        std::normal_distribution<Scalar> gaussian(Scalar(0), Scalar(1));
        std::uniform_real_distribution<Scalar> unit(Scalar(0), Scalar(1));

        Scalar bestY = detail::inf<Scalar>();
        Vector<Scalar> bestX = bounded
            ? initialPoint.cwiseMax(this->m_lowerBounds).cwiseMin(this->m_upperBounds)
            : initialPoint;
        std::size_t evaluations = 0;
        std::size_t generations = 0;
        std::size_t restarts = 0;

        // Evaluate at the feasible (clamped) point and track the incumbent.
        auto evaluate = [&](const Vector<Scalar>& yFeasible) {
            const Vector<Scalar> x = toX(yFeasible);
            const Scalar f = objective(x, nullptr);
            ++evaluations;
            if (f < bestY) {
                bestY = f;
                bestX = x;
            }
            return f;
        };

        auto reachedTarget = [&]() { return bestY < m_targetObjective + m_tolerance; };

        const Vector<Scalar> startY =
            clampY(bounded ? (initialPoint - offset).cwiseQuotient(scale) : initialPoint);

        const Scalar sigma0 = (m_initialStepSize > Scalar(0))
            ? m_initialStepSize
            : (bounded ? Scalar(0.3)
                       : Scalar(0.3) * std::max(Scalar(1), startY.cwiseAbs().maxCoeff()));

        const Index defaultLambda =
            4 + static_cast<Index>(std::floor(Scalar(3) * std::log(static_cast<Scalar>(dims))));
        Scalar lambdaReal = static_cast<Scalar>(
            (m_populationSize > 0) ? static_cast<Index>(m_populationSize) : defaultLambda);

        std::string stopReason = toString(Status::MaxFunctionEvaluationsReached);
        Status stopStatus = Status::MaxFunctionEvaluationsReached;

        // -- IPOP restart loop -----------------------------------------------
        while (evaluations < budget && !reachedTarget()) {
            const Index lambda = std::max<Index>(4, static_cast<Index>(lambdaReal));

            // The mean starts at the requested point, and is re-randomized for
            // every restart so a new basin can be found.
            Vector<Scalar> mean = startY;
            if (restarts > 0) {
                if (bounded) {
                    for (Index i = 0; i < dims; ++i) {
                        mean(i) = unit(rng);
                    }
                } else {
                    for (Index i = 0; i < dims; ++i) {
                        mean(i) = startY(i) + Scalar(3) * sigma0 * gaussian(rng);
                    }
                }
            }

            const RunOutcome outcome =
                runSingle(mean, sigma0, lambda, dims, budget, bounded, rng, gaussian,
                          evaluate, clampY, reachedTarget, evaluations, generations);

            if (outcome == RunOutcome::TargetReached) {
                stopStatus = Status::Success;
                stopReason = "optimization terminated successfully";
                break;
            }
            if (outcome == RunOutcome::BudgetExhausted) {
                stopStatus = Status::MaxFunctionEvaluationsReached;
                stopReason = toString(Status::MaxFunctionEvaluationsReached);
                break;
            }

            // converged: restart with a larger population, if any are left
            if (restarts >= m_maxRestarts) {
                stopStatus = Status::Stalled;
                stopReason = "all restarts converged without reaching the target";
                break;
            }
            ++restarts;
            lambdaReal *= m_populationIncreaseFactor;
        }

        result.x = bestX;
        result.fval = bestY;
        result.iterations = generations;
        result.functionEvaluations = evaluations;
        result.status = reachedTarget() ? Status::Success : stopStatus;
        result.message = (result.status == Status::Success ? "optimization terminated successfully"
                                                           : stopReason)
            + " (" + std::to_string(generations) + " generations, "
            + std::to_string(restarts) + " restarts)";
        return result;
    }

private:
    enum class RunOutcome { Converged, BudgetExhausted, TargetReached };

    /// One CMA-ES run to convergence (or until the budget or the target ends
    /// it). Follows the reference formulation: sample, rank, update the mean,
    /// the two evolution paths, the covariance matrix and the step size.
    template <typename EvaluateFn, typename ClampFn, typename TargetFn, typename Gaussian>
    RunOutcome runSingle(Vector<Scalar> mean, const Scalar sigma0,
                         const Index lambda, const Index dims, const std::size_t budget,
                         const bool bounded, std::mt19937_64& rng, Gaussian& gaussian,
                         const EvaluateFn& evaluate, const ClampFn& clampY,
                         const TargetFn& reachedTarget, std::size_t& evaluations,
                         std::size_t& generations) const
    {
        using Matrix = globopt::Matrix<Scalar>;

        // -- strategy parameters (tutorial defaults) -------------------------
        const Index mu = lambda / 2;
        Vector<Scalar> weights(mu);
        for (Index i = 0; i < mu; ++i) {
            weights(i) = std::log(Scalar(mu) + Scalar(0.5)) - std::log(static_cast<Scalar>(i + 1));
        }
        weights /= weights.sum();
        const Scalar muEff = Scalar(1) / weights.squaredNorm();

        const Scalar n = static_cast<Scalar>(dims);
        const Scalar cSigma = (muEff + Scalar(2)) / (n + muEff + Scalar(5));
        const Scalar dSigma = Scalar(1)
            + Scalar(2) * std::max(Scalar(0), std::sqrt((muEff - Scalar(1)) / (n + Scalar(1))) - Scalar(1))
            + cSigma;
        const Scalar cc = (Scalar(4) + muEff / n) / (n + Scalar(4) + Scalar(2) * muEff / n);
        const Scalar c1 = Scalar(2) / ((n + Scalar(1.3)) * (n + Scalar(1.3)) + muEff);
        const Scalar cMu = std::min(Scalar(1) - c1,
                                    Scalar(2) * (muEff - Scalar(2) + Scalar(1) / muEff)
                                        / ((n + Scalar(2)) * (n + Scalar(2)) + muEff));
        // E||N(0, I)||
        const Scalar chiN = std::sqrt(n)
            * (Scalar(1) - Scalar(1) / (Scalar(4) * n) + Scalar(1) / (Scalar(21) * n * n));

        Scalar sigma = sigma0;
        Vector<Scalar> pSigma = Vector<Scalar>::Zero(dims);
        Vector<Scalar> pC = Vector<Scalar>::Zero(dims);
        Matrix C = Matrix::Identity(dims, dims);
        Matrix B = Matrix::Identity(dims, dims);
        Vector<Scalar> D = Vector<Scalar>::Ones(dims);
        Vector<Scalar> invSqrtDiag = Vector<Scalar>::Ones(dims);

        // The eigendecomposition is only needed to sample and to update the
        // path, so it is refreshed lazily (tutorial: every ~1/(10 n (c1+cmu))
        // generations).
        const Scalar eigenPeriod = Scalar(1) / (Scalar(10) * n * (c1 + cMu));
        std::size_t lastEigenUpdate = 0;
        std::size_t localGeneration = 0;

        std::vector<Scalar> bestHistory;
        Matrix arY(dims, lambda);
        std::vector<Scalar> rawValues(static_cast<std::size_t>(lambda));
        std::vector<Scalar> fitness(static_cast<std::size_t>(lambda));
        std::vector<Index> order(static_cast<std::size_t>(lambda));

        while (true) {
            if (evaluations >= budget) {
                return RunOutcome::BudgetExhausted;
            }

            // -- sample and evaluate the generation ---------------------------
            Vector<Scalar> penalties(lambda);
            for (Index k = 0; k < lambda; ++k) {
                Vector<Scalar> z(dims);
                for (Index i = 0; i < dims; ++i) {
                    z(i) = gaussian(rng);
                }
                const Vector<Scalar> y = B * (D.cwiseProduct(z));
                arY.col(k) = y;

                const Vector<Scalar> sample = mean + sigma * y;
                const Vector<Scalar> feasible = clampY(sample);
                rawValues[static_cast<std::size_t>(k)] = evaluate(feasible);
                penalties(k) = bounded ? (sample - feasible).squaredNorm() : Scalar(0);

                // The incumbent is tracked by evaluate(), so an interrupted
                // generation costs nothing but its own ranking.
                if (reachedTarget()) {
                    return RunOutcome::TargetReached;
                }
                if (evaluations >= budget) {
                    return RunOutcome::BudgetExhausted;
                }
            }

            // Rank by the objective plus a boundary penalty scaled by the
            // spread of this generation, which keeps it invariant to the scale
            // of the objective.
            Scalar fMin = detail::inf<Scalar>(), fMax = -detail::inf<Scalar>();
            for (const Scalar value : rawValues) {
                if (std::isfinite(value)) {
                    fMin = std::min(fMin, value);
                    fMax = std::max(fMax, value);
                }
            }
            Scalar spread = fMax - fMin;
            if (!(spread > Scalar(0))) {
                spread = std::max(Scalar(1), std::abs(std::isfinite(fMin) ? fMin : Scalar(0)));
            }
            for (Index k = 0; k < lambda; ++k) {
                fitness[static_cast<std::size_t>(k)] =
                    rawValues[static_cast<std::size_t>(k)] + m_boundaryPenalty * spread * penalties(k);
            }

            std::iota(order.begin(), order.end(), Index(0));
            std::sort(order.begin(), order.end(), [&fitness](const Index a, const Index b) {
                return fitness[static_cast<std::size_t>(a)] < fitness[static_cast<std::size_t>(b)];
            });

            ++generations;
            ++localGeneration;

            if (reachedTarget()) {
                return RunOutcome::TargetReached;
            }

            // -- recombination -------------------------------------------------
            Vector<Scalar> yWeighted = Vector<Scalar>::Zero(dims);
            for (Index i = 0; i < mu; ++i) {
                yWeighted += weights(i) * arY.col(order[static_cast<std::size_t>(i)]);
            }
            mean += sigma * yWeighted;

            // -- evolution paths ------------------------------------------------
            // C^{-1/2} y = B diag(1/D) B^T y
            const Vector<Scalar> cInvSqrtY =
                B * invSqrtDiag.cwiseProduct(B.transpose() * yWeighted);
            pSigma = (Scalar(1) - cSigma) * pSigma
                + std::sqrt(cSigma * (Scalar(2) - cSigma) * muEff) * cInvSqrtY;

            const Scalar pSigmaNorm = pSigma.norm();
            const Scalar denom = std::sqrt(
                Scalar(1) - std::pow(Scalar(1) - cSigma, Scalar(2) * static_cast<Scalar>(localGeneration)));
            const bool hSigma =
                (pSigmaNorm / denom) < (Scalar(1.4) + Scalar(2) / (n + Scalar(1))) * chiN;

            pC = (Scalar(1) - cc) * pC;
            if (hSigma) {
                pC += std::sqrt(cc * (Scalar(2) - cc) * muEff) * yWeighted;
            }

            // -- covariance matrix update ---------------------------------------
            // C <- (1 - c1 - cmu + c1*delta) C + c1 pc pc' + cmu sum w_i y_i y_i'
            const Scalar deltaHSigma = hSigma ? Scalar(0) : cc * (Scalar(2) - cc);
            C = (Scalar(1) - c1 - cMu + c1 * deltaHSigma) * C
                + c1 * (pC * pC.transpose());
            for (Index i = 0; i < mu; ++i) {
                const auto y = arY.col(order[static_cast<std::size_t>(i)]);
                C += cMu * weights(i) * (y * y.transpose());
            }

            // -- step size update ------------------------------------------------
            sigma *= std::exp((cSigma / dSigma) * (pSigmaNorm / chiN - Scalar(1)));

            // -- lazy eigendecomposition -----------------------------------------
            if (static_cast<Scalar>(localGeneration - lastEigenUpdate) > eigenPeriod) {
                lastEigenUpdate = localGeneration;
                C = Scalar(0.5) * (C + C.transpose()).eval(); // enforce symmetry

                Eigen::SelfAdjointEigenSolver<Matrix> solver(C);
                if (solver.info() != Eigen::Success) {
                    return RunOutcome::Converged; // degenerate: restart
                }
                B = solver.eigenvectors();
                const Vector<Scalar> eigenvalues = solver.eigenvalues();
                if (eigenvalues.minCoeff() <= Scalar(0)) {
                    return RunOutcome::Converged;
                }
                D = eigenvalues.cwiseSqrt();
                invSqrtDiag = D.cwiseInverse();

                if (eigenvalues.maxCoeff() / eigenvalues.minCoeff() > m_maxConditionNumber) {
                    return RunOutcome::Converged;
                }
            }

            // -- termination criteria for this run --------------------------------
            bestHistory.push_back(rawValues[static_cast<std::size_t>(order[0])]);
            const std::size_t historyLength =
                static_cast<std::size_t>(10 + 30 * dims / lambda);
            if (bestHistory.size() > historyLength) {
                bestHistory.erase(bestHistory.begin());
            }

            // objective is flat within the generation and over recent history
            if (std::isfinite(fMax) && (fMax - fMin) < m_toleranceFunction
                && bestHistory.size() >= historyLength) {
                const auto range = std::minmax_element(bestHistory.begin(), bestHistory.end());
                if ((*range.second - *range.first) < m_toleranceFunction) {
                    return RunOutcome::Converged;
                }
            }

            // the distribution has collapsed
            if ((sigma * D.maxCoeff()) < m_toleranceX
                && (sigma * pC.cwiseAbs().maxCoeff()) < m_toleranceX) {
                return RunOutcome::Converged;
            }

            if (!mean.allFinite() || !std::isfinite(sigma) || sigma <= Scalar(0)) {
                return RunOutcome::Converged;
            }
        }
    }

    std::size_t m_maxFunctionEvaluations = 0;
    std::size_t m_populationSize = 0;
    Scalar m_initialStepSize = Scalar(0);
    std::size_t m_maxRestarts = 9;
    Scalar m_populationIncreaseFactor = Scalar(2);
    Scalar m_toleranceFunction = Scalar(1e-12);
    Scalar m_toleranceX = Scalar(1e-12);
    Scalar m_maxConditionNumber = Scalar(1e14);
    Scalar m_boundaryPenalty = Scalar(1);
    Scalar m_targetObjective = -detail::inf<Scalar>();
    Scalar m_tolerance = Scalar(1e-05);
    long long m_seed = 0;
};

} // namespace globopt

#endif
