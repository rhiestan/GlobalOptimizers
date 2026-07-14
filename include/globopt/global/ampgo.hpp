// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// AMPGO (Adaptive Memory Programming for Global Optimization), ported from
// the Python implementation by Andrea Gavana (go_amp.py, Copyright 2014).
//
// Reference: L. Lasdon, A. Duarte, F. Glover, M. Laguna, R. Marti,
// "Adaptive memory programming for constrained global optimization",
// Computers & Operations Research 37 (2010) 1500-1509.

#ifndef GLOBOPT_GLOBAL_AMPGO_HPP
#define GLOBOPT_GLOBAL_AMPGO_HPP

#include "../core/optimizer.hpp"
#include "../local/lbfgs.hpp"
#include "../local/lbfgsb.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace globopt {

/// AMPGO global optimizer: alternates local minimization with "tabu
/// tunneling" phases that transform the objective so that already-found
/// minima become poles, driving the search into unexplored regions.
///
/// The objective must provide a gradient when grad_out is non-null (wrap a
/// gradient-free function with globopt::withNumericalGradient). The local
/// solver is L-BFGS-B, so box constraints (setBounds) are supported natively.
///
/// If the value of the global optimum is known, set "target_objective"; the
/// search then stops with Status::Success as soon as it is reached within
/// "tolerance". Otherwise the search runs until the iteration or function
/// evaluation budget is exhausted, and the best point found is returned
/// (status MaxIterationsReached / MaxFunctionEvaluationsReached).
///
/// Result::gradientNorm is not meaningful for this optimizer (NaN), and
/// Result::iterations counts global minimization phases.
template <typename Scalar>
class AMPGO : public Optimizer<Scalar> {
public:
    using typename Optimizer<Scalar>::ObjectiveFn;

    AMPGO()
    {
        this->registerParam("total_iterations", &m_totalIterations,
                            "maximum number of global iterations (minimization phases)");
        this->registerParam("tunnel_iterations", &m_tunnelIterations,
                            "maximum number of tabu tunneling attempts per global iteration");
        this->registerParam("max_function_evaluations", &m_maxFunctionEvaluations,
                            "maximum number of function evaluations (0 = max(100, 10*n))");
        this->registerParam("tolerance", &m_tolerance,
                            "stop when the best objective is within this distance of target_objective");
        this->registerParam("target_objective", &m_targetObjective,
                            "value of the global optimum, if known (-inf to disable)");
        this->registerParam("eps1", &m_eps1,
                            "constant defining the aspiration value during tunneling");
        this->registerParam("eps2", &m_eps2,
                            "perturbation factor for the tunneling start point");
        this->registerParam("tabu_list_size", &m_tabuListSize,
                            "size of the circular tabu list");
        this->registerParam("tabu_strategy", &m_tabuStrategy,
                            "point to drop when the tabu list is full: 'oldest' or 'farthest'");
        this->registerParam("local_solver", &m_localSolver,
                            "local optimizer: 'L-BFGS-B' (default) or 'L-BFGS'");
        this->registerParam("seed", &m_seed,
                            "random seed for the tunneling perturbations (0 = non-deterministic)");
    }

    const char* name() const override { return "AMPGO"; }

protected:
    Result<Scalar> doOptimize(const ObjectiveFn& objective, const Vector<Scalar>& initialPoint) override
    {
        const Index n = initialPoint.size();

        Result<Scalar> result;
        result.x = initialPoint;
        result.gradientNorm = std::numeric_limits<Scalar>::quiet_NaN();

        if (m_tabuListSize < 1) {
            result.status = Status::InvalidInput;
            result.message = "tabu_list_size must be at least 1";
            return result;
        }
        if (m_tabuStrategy != "oldest" && m_tabuStrategy != "farthest") {
            result.status = Status::InvalidInput;
            result.message = "tabu_strategy must be 'oldest' or 'farthest'";
            return result;
        }

        std::unique_ptr<Optimizer<Scalar>> local = makeLocalSolver();
        if (!local) {
            result.status = Status::InvalidInput;
            result.message = "invalid local_solver: '" + m_localSolver + "'";
            return result;
        }

        const std::size_t maxFnEvals = (m_maxFunctionEvaluations > 0)
            ? m_maxFunctionEvaluations
            : std::max<std::size_t>(100, 10 * static_cast<std::size_t>(n));

        const Scalar localTol = std::min(m_tolerance, Scalar(1e-08));
        if (local->hasParam("rel_objective_change_tolerance")) {
            local->setParam("rel_objective_change_tolerance", static_cast<double>(localTol));
        }
        if (this->m_boundsSet) {
            local->setBounds(this->m_lowerBounds, this->m_upperBounds);
        }

        std::size_t& evaluations = result.functionEvaluations;

        auto countedObjective = [&](const Vector<Scalar>& x, Vector<Scalar>* gradOut) -> Scalar {
            ++evaluations;
            return objective(x, gradOut);
        };

        auto runLocal = [&](const ObjectiveFn& fn, const Vector<Scalar>& from) {
            if (local->hasParam("max_function_evaluations")) {
                const std::size_t remaining = (evaluations < maxFnEvals) ? maxFnEvals - evaluations : 1;
                local->setParam("max_function_evaluations", std::max<std::size_t>(1, remaining));
            }
            return local->run(fn, from);
        };

        std::mt19937_64 rng(m_seed != 0 ? static_cast<std::uint64_t>(m_seed) : std::random_device{}());
        std::uniform_real_distribution<Scalar> uniform(Scalar(-1), Scalar(1));

        std::vector<Vector<Scalar>> tabuList;
        Scalar bestF = detail::inf<Scalar>();
        Vector<Scalar> bestX = initialPoint;

        std::size_t globalIter = 0;
        std::size_t allTunnel = 0, successTunnel = 0;

        Vector<Scalar> x0 = initialPoint;

        auto finish = [&](Status status, const std::string& reason) {
            result.x = bestX;
            result.fval = bestF;
            result.iterations = globalIter;
            result.status = status;
            result.message = reason
                + " (tunneling phases: " + std::to_string(allTunnel)
                + ", successful: " + std::to_string(successTunnel) + ")";
            return result;
        };

        auto targetReached = [&]() {
            return bestF < m_targetObjective + m_tolerance;
        };

        while (true) {
            // minimization phase

            const Result<Scalar> localRes = runLocal(countedObjective, x0);
            Vector<Scalar> xf = localRes.x;
            Scalar yf = localRes.fval;

            if (yf < bestF) {
                bestF = yf;
                bestX = xf;
            }

            if (targetReached()) {
                return finish(Status::Success, "optimization terminated successfully");
            }
            if (evaluations >= maxFnEvals) {
                return finish(Status::MaxFunctionEvaluationsReached,
                              "maximum number of function evaluations exceeded");
            }

            dropTabuPoints(xf, tabuList);
            tabuList.push_back(xf);

            // tunneling phase

            bool improve = false;

            for (std::size_t i = 0; i < m_tunnelIterations && !improve; ++i) {
                ++allTunnel;

                Vector<Scalar> r(n);
                for (Index k = 0; k < n; ++k) {
                    r(k) = uniform(rng);
                }

                Scalar beta = m_eps2 * xf.norm() / r.norm();
                if (!(std::abs(beta) >= Scalar(1e-08))) { // also catches NaN
                    beta = m_eps2;
                }

                x0 = xf + beta * r;
                if (this->m_boundsSet) {
                    x0 = x0.cwiseMax(this->m_lowerBounds).cwiseMin(this->m_upperBounds);
                }

                const Scalar aspiration = bestF - m_eps1 * (Scalar(1) + std::abs(bestF));

                // tabu tunneling function:
                //   T(x) = (f(x) - aspiration)^2 / prod_i ||x - tabu_i||
                // with the gradient obtained via the chain rule
                auto tunnelObjective = [&](const Vector<Scalar>& x, Vector<Scalar>* gradOut) -> Scalar {
                    Vector<Scalar> fGrad;
                    Scalar fx;

                    if (gradOut) {
                        fGrad.resize(n);
                        fx = countedObjective(x, &fGrad);
                    } else {
                        fx = countedObjective(x, nullptr);
                    }

                    const Scalar denominator = std::max(tabuDenominator(x, tabuList),
                                                        std::numeric_limits<Scalar>::denorm_min());
                    const Scalar delta = fx - aspiration;
                    const Scalar tval = delta * delta / denominator;

                    if (gradOut) {
                        Vector<Scalar> poleTerm = Vector<Scalar>::Zero(n);
                        for (const Vector<Scalar>& tabu : tabuList) {
                            const Vector<Scalar> diff = x - tabu;
                            const Scalar d2 = std::max(diff.squaredNorm(),
                                                       std::numeric_limits<Scalar>::denorm_min());
                            poleTerm += diff / d2;
                        }
                        *gradOut = (2 * delta / denominator) * fGrad - tval * poleTerm;
                    }

                    return tval;
                };

                const Result<Scalar> tunnelRes = runLocal(tunnelObjective, x0);
                xf = tunnelRes.x;
                const Scalar ytf = tunnelRes.fval;

                // recover the objective value from the tunneling value:
                // f = aspiration + sqrt(T * prod_i ||x - tabu_i||)
                yf = aspiration + std::sqrt(ytf * tabuDenominator(xf, tabuList));

                if (yf <= bestF + m_tolerance) {
                    bestF = yf;
                    bestX = xf;
                    improve = true;
                    ++successTunnel;
                }

                if (targetReached()) {
                    return finish(Status::Success, "optimization terminated successfully");
                }
                if (evaluations >= maxFnEvals) {
                    return finish(Status::MaxFunctionEvaluationsReached,
                                  "maximum number of function evaluations exceeded");
                }

                dropTabuPoints(xf, tabuList);
                tabuList.push_back(xf);
            }

            ++globalIter;
            x0 = xf;

            if (globalIter >= m_totalIterations) {
                return finish(Status::MaxIterationsReached,
                              "maximum number of global iterations exceeded");
            }
            if (targetReached()) {
                return finish(Status::Success, "optimization terminated successfully");
            }
        }
    }

private:
    std::unique_ptr<Optimizer<Scalar>> makeLocalSolver() const
    {
        std::string key;
        for (const char c : m_localSolver) {
            if (c != '-' && c != '_' && c != ' ') {
                key.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
        }

        if (key == "lbfgsb") {
            return std::make_unique<LBFGSB<Scalar>>();
        }
        if (key == "lbfgs") {
            return std::make_unique<LBFGS<Scalar>>();
        }
        return nullptr;
    }

    static Scalar tabuDenominator(const Vector<Scalar>& x, const std::vector<Vector<Scalar>>& tabuList)
    {
        Scalar denominator = 1;
        for (const Vector<Scalar>& tabu : tabuList) {
            denominator *= (x - tabu).norm();
        }
        return denominator;
    }

    void dropTabuPoints(const Vector<Scalar>& xf, std::vector<Vector<Scalar>>& tabuList) const
    {
        if (tabuList.size() < m_tabuListSize) {
            return;
        }

        if (m_tabuStrategy == "oldest") {
            tabuList.erase(tabuList.begin());
        } else {
            // drop the point farthest from the latest local minimum
            std::size_t farthest = 0;
            Scalar maxDistance = -1;
            for (std::size_t i = 0; i < tabuList.size(); ++i) {
                const Scalar distance = (tabuList[i] - xf).norm();
                if (distance > maxDistance) {
                    maxDistance = distance;
                    farthest = i;
                }
            }
            tabuList.erase(tabuList.begin() + static_cast<std::ptrdiff_t>(farthest));
        }
    }

    std::size_t m_totalIterations = 20;
    std::size_t m_tunnelIterations = 5;
    std::size_t m_maxFunctionEvaluations = 0;
    Scalar m_tolerance = Scalar(1e-05);
    Scalar m_targetObjective = -detail::inf<Scalar>();
    Scalar m_eps1 = Scalar(0.02);
    Scalar m_eps2 = Scalar(0.1);
    std::size_t m_tabuListSize = 5;
    std::string m_tabuStrategy = "farthest";
    std::string m_localSolver = "L-BFGS-B";
    long long m_seed = 0;
};

} // namespace globopt

#endif
