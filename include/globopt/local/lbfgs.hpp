// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// L-BFGS ported from OptimLib (https://github.com/kthohr/optim),
// Copyright (C) 2016-2023 Keith O'Hara, Apache License 2.0.

#ifndef GLOBOPT_LOCAL_LBFGS_HPP
#define GLOBOPT_LOCAL_LBFGS_HPP

#include "../core/line_search.hpp"
#include "../core/optimizer.hpp"
#include "lbfgsb.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace globopt {

namespace detail {

// two-loop recursion; algorithm 7.4 of Nocedal and Wright (2006)
template <typename Scalar>
inline Vector<Scalar> lbfgsTwoLoopRecursion(
    Vector<Scalar> q,
    const Matrix<Scalar>& sMat,
    const Matrix<Scalar>& yMat,
    const std::size_t M)
{
    Vector<Scalar> alpha(M);

    for (std::size_t i = 0; i < M; ++i) {
        const Scalar rho = Scalar(1) / yMat.col(i).dot(sMat.col(i));
        alpha(i) = rho * sMat.col(i).dot(q);

        q -= alpha(i) * yMat.col(i);
    }

    Vector<Scalar> r = q * (sMat.col(0).dot(yMat.col(0)) / yMat.col(0).dot(yMat.col(0)));

    for (int i = static_cast<int>(M) - 1; i >= 0; --i) {
        const Scalar rho = Scalar(1) / yMat.col(i).dot(sMat.col(i));
        const Scalar beta = rho * yMat.col(i).dot(r);

        r += (alpha(i) - beta) * sMat.col(i);
    }

    return r;
}

} // namespace detail

/// Limited-memory BFGS quasi-Newton optimizer with a Moré-Thuente line
/// search, for unconstrained problems. Requires a gradient: the objective is
/// called with a non-null grad_out on every evaluation.
///
/// If box constraints are set (via setBounds), the run is delegated to
/// LBFGSB, which handles bounds natively; max_iterations, gradient_tolerance
/// and memory carry over, while the Wolfe constants (specific to the
/// Moré-Thuente line search) do not apply.
template <typename Scalar>
class LBFGS : public Optimizer<Scalar> {
public:
    using typename Optimizer<Scalar>::ObjectiveFn;

    LBFGS()
    {
        this->registerParam("max_iterations", &m_maxIterations,
                            "maximum number of iterations");
        this->registerParam("gradient_tolerance", &m_gradientTolerance,
                            "stop when the L2 norm of the gradient falls below this value");
        this->registerParam("rel_solution_change_tolerance", &m_relSolutionChangeTolerance,
                            "stop when the relative change of the solution falls below this value");
        this->registerParam("memory", &m_memory,
                            "number of past iterations kept for the Hessian approximation");
        this->registerParam("wolfe_constant_1", &m_wolfeCons1,
                            "line search Armijo sufficient-decrease tolerance");
        this->registerParam("wolfe_constant_2", &m_wolfeCons2,
                            "line search curvature-condition tolerance");
    }

    const char* name() const override { return "L-BFGS"; }

protected:
    Result<Scalar> doOptimize(const ObjectiveFn& objective, const Vector<Scalar>& initialPoint) override
    {
        if (this->m_boundsSet) {
            return delegateToLbfgsb(objective, initialPoint);
        }

        const Index nVals = initialPoint.size();

        const std::size_t iterMax = m_maxIterations;
        const Scalar gradErrTol = m_gradientTolerance;
        const Scalar relSolChangeTol = m_relSolutionChangeTolerance;
        const std::size_t parM = std::max<std::size_t>(2, m_memory);
        const Scalar wolfeCons1 = m_wolfeCons1;
        const Scalar wolfeCons2 = m_wolfeCons2;

        Result<Scalar> result;
        std::size_t& fnEvals = result.functionEvaluations;

        auto objfn = [&](const Vector<Scalar>& valsInp, Vector<Scalar>* gradOut) -> Scalar {
            ++fnEvals;
            return objective(valsInp, gradOut);
        };

        auto finish = [&](const Vector<Scalar>& xFinal, Scalar gradErr, std::size_t iter, Status statusIfNotConverged) {
            result.x = xFinal;
            result.gradientNorm = gradErr;
            result.iterations = iter;

            if (gradErr <= gradErrTol) {
                result.status = Status::Success;
                result.message = "gradient tolerance reached";
            } else {
                result.status = statusIfNotConverged;
                result.message = toString(statusIfNotConverged);
            }

            result.fval = objfn(result.x, nullptr);

            return result;
        };

        // initialization

        Vector<Scalar> x = initialPoint;

        Vector<Scalar> grad(nVals);
        Vector<Scalar> d = Vector<Scalar>::Zero(nVals);
        Matrix<Scalar> sMat = Matrix<Scalar>::Zero(nVals, parM);
        Matrix<Scalar> yMat = Matrix<Scalar>::Zero(nVals, parM);

        objfn(x, &grad);

        Scalar gradErr = grad.norm();

        if (gradErr <= gradErrTol) {
            return finish(x, gradErr, 0, Status::Success);
        }

        // first step: gradient descent direction with line search

        d = -grad;

        Vector<Scalar> xP = x, gradP = grad;

        detail::lineSearchMoreThuente(Scalar(1), xP, gradP, d, wolfeCons1, wolfeCons2, objfn);

        Vector<Scalar> s = xP - x;

        gradErr = gradP.norm();
        Scalar relSolChange = (s.array() / (x.array().abs() + detail::smallNumber<Scalar>())).abs().sum();

        if (gradErr <= gradErrTol) {
            return finish(xP, gradErr, 0, Status::Success);
        }

        Vector<Scalar> y = gradP - grad;

        sMat.col(0) = s;
        yMat.col(0) = y;

        x = xP;
        grad = gradP;

        // main loop

        std::size_t iter = 0;

        while (gradErr > gradErrTol && relSolChange > relSolChangeTol && iter < iterMax) {
            ++iter;

            d = -detail::lbfgsTwoLoopRecursion(grad, sMat, yMat, std::min(iter, parM));

            detail::lineSearchMoreThuente(Scalar(1), xP, gradP, d, wolfeCons1, wolfeCons2, objfn);

            s = xP - x;
            y = gradP - grad;

            gradErr = gradP.norm();
            relSolChange = (s.array() / (x.array().abs() + detail::smallNumber<Scalar>())).abs().sum();

            x = xP;
            grad = gradP;

            sMat.middleCols(1, parM - 1) = sMat.middleCols(0, parM - 1).eval();
            yMat.middleCols(1, parM - 1) = yMat.middleCols(0, parM - 1).eval();

            sMat.col(0) = s;
            yMat.col(0) = y;
        }

        const Status statusIfNotConverged = (iter >= iterMax) ? Status::MaxIterationsReached : Status::Stalled;

        return finish(xP, gradErr, iter, statusIfNotConverged);
    }

private:
    Result<Scalar> delegateToLbfgsb(const ObjectiveFn& objective, const Vector<Scalar>& initialPoint)
    {
        LBFGSB<Scalar> solver;
        solver.setBounds(this->m_lowerBounds, this->m_upperBounds);
        solver.setParam("max_iterations", m_maxIterations);
        solver.setParam("gradient_tolerance", static_cast<double>(m_gradientTolerance));
        solver.setParam("memory", m_memory);

        Result<Scalar> result = solver.run(objective, initialPoint);
        result.message += " (bounds set: solved with L-BFGS-B)";
        return result;
    }

    std::size_t m_maxIterations = 2000;
    Scalar m_gradientTolerance = Scalar(1e-08);
    Scalar m_relSolutionChangeTolerance = Scalar(1e-14);
    std::size_t m_memory = 10;
    Scalar m_wolfeCons1 = Scalar(1e-03);
    Scalar m_wolfeCons2 = Scalar(0.90);
};

} // namespace globopt

#endif
