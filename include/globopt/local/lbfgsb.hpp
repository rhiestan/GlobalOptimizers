// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// L-BFGS-B ported from https://github.com/droemer7/l-bfgs-b,
// Copyright (c) 2023 Dane Roemer, MIT License.
//
// Reference: R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, "A Limited Memory
// Algorithm for Bound Constrained Optimization", Tech. Report, NAM-08,
// EECS Department, Northwestern University, 1994.

#ifndef GLOBOPT_LOCAL_LBFGSB_HPP
#define GLOBOPT_LOCAL_LBFGSB_HPP

#include "../core/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace globopt {

/// Limited-memory BFGS optimizer for box-constrained problems (L-BFGS-B).
/// Bounds are handled natively via gradient projection (generalized Cauchy
/// point search plus subspace minimization), so solutions may lie exactly on
/// a bound. Without bounds it behaves like L-BFGS. Requires a gradient.
///
/// The default Lewis-Overton line search enforces the weak Wolfe condition,
/// which also permits minimizing non-smooth objectives; set the
/// "strong_wolfe" parameter for the strong Wolfe condition.
template <typename Scalar>
class LBFGSB : public Optimizer<Scalar> {
public:
    using typename Optimizer<Scalar>::ObjectiveFn;

    LBFGSB()
    {
        this->registerParam("max_iterations", &m_maxIterations,
                            "maximum number of iterations");
        this->registerParam("max_function_evaluations", &m_maxFunctionEvaluations,
                            "maximum number of function evaluations (0 = unlimited)");
        this->registerParam("gradient_tolerance", &m_gradientTolerance,
                            "stop when the infinity norm of the projected gradient falls below this value");
        this->registerParam("rel_objective_change_tolerance", &m_relObjectiveChangeTolerance,
                            "stop when the relative change of the objective falls below this value");
        this->registerParam("rel_solution_change_tolerance", &m_relSolutionChangeTolerance,
                            "stop when the relative change of the solution falls below this value");
        this->registerParam("memory", &m_memory,
                            "number of correction pairs kept for the Hessian approximation");
        this->registerParam("strong_wolfe", &m_strongWolfe,
                            "enforce the strong Wolfe condition in the line search instead of the weak one");
        this->registerParam("line_search_max_iterations", &m_lineSearchMaxIterations,
                            "maximum number of line search iterations per step");
    }

    const char* name() const override { return "L-BFGS-B"; }

protected:
    Result<Scalar> doOptimize(const ObjectiveFn& objective, const Vector<Scalar>& initialPoint) override
    {
        const Index n = initialPoint.size();

        // effective bounds: unset or non-finite entries become the largest
        // representable values so the breakpoint computations stay finite
        Vector<Scalar> l(n), u(n);
        for (Index i = 0; i < n; ++i) {
            const Scalar li = this->m_boundsSet ? this->m_lowerBounds(i) : lowestVal();
            const Scalar ui = this->m_boundsSet ? this->m_upperBounds(i) : maxVal();
            l(i) = std::isfinite(li) ? li : lowestVal();
            u(i) = std::isfinite(ui) ? ui : maxVal();
            if (l(i) > u(i)) {
                std::swap(l(i), u(i));
            }
        }

        Result<Scalar> result;
        std::size_t& fnEvals = result.functionEvaluations;

        auto objfn = [&](const Vector<Scalar>& xInp, Vector<Scalar>* gradOut) -> Scalar {
            ++fnEvals;
            return objective(xInp, gradOut);
        };

        resetMemory(n);

        Vector<Scalar> x = initialPoint.cwiseMin(u).cwiseMax(l);
        Vector<Scalar> g(n);
        Scalar fx = objfn(x, &g);

        Scalar gNorm = infinityNorm(projectedGradient(x, g, l, u));
        Scalar dfNorm = detail::inf<Scalar>();
        Scalar dxNorm = detail::inf<Scalar>();

        std::size_t iter = 0;
        int stallCount = 0;

        Status status = Status::MaxIterationsReached;
        std::string message;

        while (true) {
            if (gNorm <= m_gradientTolerance) {
                status = Status::Success;
                message = "projected gradient tolerance reached";
                break;
            }
            if (stallCount == 0 && dfNorm <= m_relObjectiveChangeTolerance) {
                status = Status::Success;
                message = "objective change tolerance reached";
                break;
            }
            if (stallCount == 0 && dxNorm <= m_relSolutionChangeTolerance) {
                status = Status::Success;
                message = "solution change tolerance reached";
                break;
            }
            if (stallCount >= 2) {
                status = Status::Stalled;
                message = "no progress after restarting from steepest descent";
                break;
            }
            if (iter >= m_maxIterations) {
                status = Status::MaxIterationsReached;
                message = toString(Status::MaxIterationsReached);
                break;
            }
            if (m_maxFunctionEvaluations > 0 && fnEvals >= m_maxFunctionEvaluations) {
                status = Status::MaxFunctionEvaluationsReached;
                message = toString(Status::MaxFunctionEvaluationsReached);
                break;
            }

            // one optimization step
            const Vector<Scalar> xc = cauchyPoint(x, g, l, u);
            const Vector<Scalar> d = searchDir(xc, x, g, l, u);
            const Scalar tMax = maxStep(x, l, u, d);

            Scalar fNew = fx;
            Vector<Scalar> gNew(n);
            const Scalar t = lineSearch(objfn, fx, x, g, d, tMax, fNew, gNew);

            ++iter;

            if (t > Scalar(0)) {
                const Vector<Scalar> xNew = x + t * d;
                const Scalar dxMax = infinityNorm(xNew - x);

                if (dxMax == Scalar(0)) {
                    ++stallCount;
                } else {
                    stallCount = 0;
                    dfNorm = std::abs(fNew - fx)
                           / std::max(std::max(std::abs(fNew), std::abs(fx)), Scalar(1));
                    dxNorm = dxMax
                           / std::max(std::max(infinityNorm(xNew), infinityNorm(x)), Scalar(1));
                }

                updateMatrices(xNew - x, gNew - g);

                x = xNew;
                fx = fNew;
                g = gNew;
            } else {
                // no suitable step: discard all correction pairs and restart
                // along the steepest descent direction
                ++stallCount;
                resetMemory(n);
            }

            gNorm = infinityNorm(projectedGradient(x, g, l, u));
        }

        result.x = x;
        result.fval = fx;
        result.gradientNorm = gNorm;
        result.iterations = iter;
        result.status = status;
        result.message = message;

        return result;
    }

private:
    static constexpr Scalar lowestVal() { return std::numeric_limits<Scalar>::lowest(); }
    static constexpr Scalar maxVal() { return std::numeric_limits<Scalar>::max(); }

    template <typename Derived>
    static Scalar infinityNorm(const Eigen::MatrixBase<Derived>& v)
    {
        return v.size() == 0 ? Scalar(0) : v.template lpNorm<Eigen::Infinity>();
    }

    // gradient projected onto the feasible region: how much x can move along
    // the steepest descent direction while staying within the bounds
    static Vector<Scalar> projectedGradient(
        const Vector<Scalar>& x, const Vector<Scalar>& g,
        const Vector<Scalar>& l, const Vector<Scalar>& u)
    {
        return (x - g).cwiseMin(u).cwiseMax(l) - x;
    }

    // max{ t : l(i) <= x(i) + t*d(i) <= u(i) for all i }
    static Scalar maxStep(
        const Vector<Scalar>& x, const Vector<Scalar>& l,
        const Vector<Scalar>& u, const Vector<Scalar>& d)
    {
        // scaler preventing precision issues from generating a step that
        // violates the bounds by a small amount
        const Scalar s = Scalar(1) - detail::eps<Scalar>();

        Scalar tMax = maxVal();

        for (Index i = 0; i < d.size(); ++i) {
            Scalar t;
            if (d(i) != Scalar(0)) {
                t = s * std::max((u(i) - x(i)) / d(i), (l(i) - x(i)) / d(i));
                t = std::max(t, Scalar(0));
            } else {
                t = maxVal();
            }
            tMax = std::min(t, tMax);
        }

        return tMax;
    }

    void resetMemory(const Index n)
    {
        m_m = 0;
        m_th = Scalar(1);
        m_thInv = Scalar(1);

        m_I = Matrix<Scalar>::Identity(n, n);
        m_S = Matrix<Scalar>::Zero(n, 1);
        m_Y = Matrix<Scalar>::Zero(n, 1);
        m_SS = Matrix<Scalar>::Zero(1, 1);
        m_SY = Matrix<Scalar>::Zero(1, 1);
        m_YY = Matrix<Scalar>::Zero(1, 1);

        m_W = Matrix<Scalar>::Zero(n, 2);
        m_Wb = Matrix<Scalar>::Zero(n, 2);

        m_M = Matrix<Scalar>::Zero(2, 2);
        m_Mb = Matrix<Scalar>::Zero(2, 2);

        m_c = Vector<Scalar>::Zero(2);

        m_freeSet.clear();
        m_freeSet.reserve(static_cast<std::size_t>(n));
        m_activeSet.clear();
        m_activeSet.reserve(static_cast<std::size_t>(n));
    }

    // Determines the first local minimizer of the univariate, piecewise
    // quadratic q(t) = m(P(x - t*g, l, u)), where m is the quadratic model of
    // the objective and P projects onto the feasible box.
    Vector<Scalar> cauchyPoint(
        const Vector<Scalar>& x, const Vector<Scalar>& g,
        const Vector<Scalar>& l, const Vector<Scalar>& u)
    {
        const Index n = x.size();

        struct Breakpoint {
            Index i;
            Scalar t;
        };

        Vector<Scalar> xc = x;
        std::vector<Breakpoint> breakpoints;
        Vector<Scalar> t = Vector<Scalar>::Zero(n);
        Vector<Scalar> d = Vector<Scalar>::Zero(n);
        Vector<Scalar> p = Vector<Scalar>::Zero(m_W.cols());
        m_c = Vector<Scalar>::Zero(m_W.cols());
        m_freeSet.clear();
        m_activeSet.clear();

        std::size_t q = 0;
        Scalar fp = 0;
        Scalar fpp = 0;
        Scalar dtMin = 0;
        Scalar tStart = 0;
        Scalar tEnd = 0;
        Scalar dt = 0;

        // calculate breakpoints t(i) and descent directions d(i)
        for (Index i = 0; i < n; ++i) {
            if (g(i) < Scalar(0)) {
                t(i) = (x(i) - u(i)) / g(i);
            } else if (g(i) > Scalar(0)) {
                t(i) = (x(i) - l(i)) / g(i);
            } else {
                t(i) = maxVal();
            }

            // variables with t(i) == 0 stay fixed: d(i) = 0 and xc(i) = x(i)
            if (t(i) > Scalar(0)) {
                d(i) = -g(i);
                breakpoints.push_back({i, t(i)});
            }
        }

        if (!breakpoints.empty()) {
            std::sort(breakpoints.begin(), breakpoints.end(),
                      [](const Breakpoint& a, const Breakpoint& b) { return a.t < b.t; });

            p = m_W.transpose() * d;

            fp = -d.dot(d);
            fpp = -m_th * fp - p.dot(m_M * p);
            dtMin = (fpp == Scalar(0)) ? maxVal() : -fp / fpp;

            // define the first interval
            Index b = breakpoints[q].i;
            tStart = tEnd;
            tEnd = t(b);
            dt = tEnd - tStart;

            // inspect each interval in t until dt_min falls within the interval
            while (dtMin > dt) {
                // the current component's minimum is not in the interval, so it
                // must be bounded; zero out its search direction
                xc(b) = d(b) > Scalar(0) ? u(b) : l(b);
                const Scalar zb = xc(b) - x(b);
                d(b) = 0;

                // update vector c used to initialize the subspace minimization
                m_c += dt * p;

                // calculate f' and f'' for locating the minimum
                const Vector<Scalar> w = m_W.row(b).transpose();
                fp += dt * fpp + g(b) * g(b) + m_th * g(b) * zb - g(b) * w.dot(m_M * m_c);
                fpp += -m_th * g(b) * g(b) - 2 * g(b) * w.dot(m_M * p) - g(b) * g(b) * w.dot(m_M * w);
                dtMin = (fpp == Scalar(0)) ? maxVal() : -fp / fpp;

                // update p for the next calculation of f' and f''
                p += g(b) * w;

                // move to the next interval
                tStart = tEnd;

                if (++q < breakpoints.size()) {
                    b = breakpoints[q].i;
                    tEnd = t(b);
                    dt = tEnd - tStart;
                } else {
                    // no more intervals: the final component is bounded and
                    // there are no free variables
                    tEnd = tStart;
                    dt = 0;
                    break;
                }
            }
        }

        dtMin = std::max(dtMin, Scalar(0));
        const Scalar tMin = tStart + dtMin;

        // construct the free/active sets and compute the remaining xc(i).
        // (The paper says to compare against t_end here; using t_min instead is
        // a correction inherited from the ported implementation: the minimum is
        // at t_min, so xc(i) is at its bound exactly when t(i) == t_min.)
        for (Index i = 0; i < n; ++i) {
            if (q < breakpoints.size()) {
                if (t(i) >= tMin) {
                    xc(i) = x(i) + tMin * d(i);
                }

                if (t(i) > tMin) {
                    m_freeSet.push_back(i);
                } else {
                    m_activeSet.push_back(i);
                }
            } else {
                // no free variables: every xc(i) is already at x(i) or a bound
                m_activeSet.push_back(i);
            }
        }
        m_c += dtMin * p;

        return xc;
    }

    // Approximately solves the subspace minimization problem for the search
    // direction d over the variables that are free at the Cauchy point,
    // subject to the bounds on those variables.
    Vector<Scalar> searchDir(
        const Vector<Scalar>& xc, const Vector<Scalar>& x, const Vector<Scalar>& g,
        const Vector<Scalar>& l, const Vector<Scalar>& u)
    {
        // Eigen::indexing::all is the spelling that works from Eigen 3.4 through master
        const Matrix<Scalar> A = m_I(Eigen::indexing::all, m_activeSet); // unit vectors spanning the active set (n x ta)
        const Matrix<Scalar> Z = m_I(Eigen::indexing::all, m_freeSet);   // unit vectors spanning the free set (n x tf)
        Vector<Scalar> v;                                      // (2m x 1)
        Matrix<Scalar> N;                                      // (2m x 2m)

        // reduced gradient of the quadratic model at xc (tf x 1)
        const Vector<Scalar> rc = Z.transpose() * (g + m_th * (xc - x) - m_W * m_M * m_c);

        // compute v and N using the active set or free set, whichever is smaller
        if (A.cols() < Z.cols()) {
            v = m_Mb * m_W.transpose() * Z * rc;

            // if A is empty, N remains empty (mathematically N = I)
            if (A.cols() > 0) {
                N = Matrix<Scalar>::Identity(m_Mb.rows(), m_Mb.cols())
                  + m_th * m_Mb * m_Wb.transpose() * A * A.transpose() * m_Wb;
            }
        } else {
            v = m_M * m_W.transpose() * Z * rc;

            // if Z is empty, N remains empty (mathematically N = I)
            if (Z.cols() > 0) {
                N = Matrix<Scalar>::Identity(m_M.rows(), m_M.cols())
                  - m_thInv * m_M * m_W.transpose() * Z * Z.transpose() * m_W;
            }
        }

        if (N.size() > 0) {
            v = N.lu().solve(v);
        }

        // du = -B^-1*rc = -(1/th)*rc - (1/th^2)*Z^T*W*v (tf x 1)
        // (Equation 5.11 of the paper drops the minus sign of Equation 5.7;
        // the ported implementation restores it.)
        const Vector<Scalar> du = -m_thInv * rc - m_thInv * m_thInv * Z.transpose() * m_W * v;

        // a_star = max{ a : a <= 1, l(i) - xc(i) <= a*du(i) <= u(i) - xc(i), i free }
        const Vector<Scalar> xcFree = xc(m_freeSet);
        const Vector<Scalar> lFree = l(m_freeSet);
        const Vector<Scalar> uFree = u(m_freeSet);
        const Scalar aStar = std::min(Scalar(1), maxStep(xcFree, lFree, uFree, du));

        // d(i) = (xc - x)(i), plus the subspace step for free variables
        Vector<Scalar> d = xc - x;
        const Vector<Scalar> Zdu = Z * (aStar * du);

        for (const Index i : m_freeSet) {
            d(i) += Zdu(i);
        }

        return d;
    }

    // Performs the limited-memory BFGS update of th, S, Y, W, Wb, M, Mb.
    void updateMatrices(const Vector<Scalar>& s, const Vector<Scalar>& y)
    {
        // discard {s, y} if the curvature condition s^T*y > 0 is not satisfied
        if (s.dot(y) <= detail::eps<Scalar>() * y.squaredNorm()) {
            return;
        }

        const Index mMax = static_cast<Index>(std::max<std::size_t>(1, m_memory));
        m_m = std::min(m_m + 1, mMax);
        const Index m = m_m;

        // at max memory size, discard the oldest {s, y} pair by shifting data
        if (m_Y.cols() == mMax) {
            if (m > 1) {
                m_S.leftCols(m - 1) = m_S.rightCols(m - 1).eval();
                m_Y.leftCols(m - 1) = m_Y.rightCols(m - 1).eval();
                m_SS.topLeftCorner(m - 1, m - 1) = m_SS.bottomRightCorner(m - 1, m - 1).eval();
                m_SY.topLeftCorner(m - 1, m - 1) = m_SY.bottomRightCorner(m - 1, m - 1).eval();
                m_YY.topLeftCorner(m - 1, m - 1) = m_YY.bottomRightCorner(m - 1, m - 1).eval();
            }
        } else {
            m_S.conservativeResize(Eigen::NoChange, m);
            m_Y.conservativeResize(Eigen::NoChange, m);
            m_SS.conservativeResize(m, m);
            m_SY.conservativeResize(m, m);
            m_YY.conservativeResize(m, m);

            m_W.conservativeResize(Eigen::NoChange, 2 * m);
            m_Wb.conservativeResize(Eigen::NoChange, 2 * m);

            m_M.conservativeResize(2 * m, 2 * m);
            m_Mb.conservativeResize(2 * m, 2 * m);
        }

        m_th = y.dot(y) / y.dot(s);
        m_thInv = Scalar(1) / m_th;

        m_S.col(m - 1) = s;
        m_Y.col(m - 1) = y;

        m_SS.row(m - 1) = m_S.col(m - 1).transpose() * m_S;
        m_SS.col(m - 1) = m_SS.row(m - 1).transpose();

        m_SY.row(m - 1) = m_S.col(m - 1).transpose() * m_Y;
        m_SY.col(m - 1) = m_S.transpose() * m_Y.col(m - 1);

        m_YY.row(m - 1) = m_Y.col(m - 1).transpose() * m_Y;
        m_YY.col(m - 1) = m_YY.row(m - 1).transpose();

        // D (diagonal of S^T*Y) and R^-1 (inverse of the upper triangle of S^T*Y)
        const Matrix<Scalar> D = m_SY.diagonal().asDiagonal();
        const Matrix<Scalar> R = m_SY.template triangularView<Eigen::Upper>();
        const Matrix<Scalar> RInv = R.inverse();
        const Matrix<Scalar> LStrict = m_SY.template triangularView<Eigen::StrictlyLower>();

        m_W.leftCols(m) = m_Y;
        m_W.rightCols(m) = m_th * m_S;

        m_Wb.leftCols(m) = m_thInv * m_Y;
        m_Wb.rightCols(m) = m_S;

        m_M.topLeftCorner(m, m) = -D;
        m_M.topRightCorner(m, m) = LStrict.transpose();
        m_M.bottomLeftCorner(m, m) = LStrict;
        m_M.bottomRightCorner(m, m) = m_th * m_SS;
        m_M = m_M.inverse().eval();

        m_Mb.topLeftCorner(m, m) = Matrix<Scalar>::Zero(m, m);
        m_Mb.topRightCorner(m, m) = -RInv;
        m_Mb.bottomLeftCorner(m, m) = (-RInv).transpose();
        m_Mb.bottomRightCorner(m, m) = RInv.transpose() * (D + m_thInv * m_YY) * RInv;
    }

    // Lewis-Overton line search enforcing the Armijo condition plus the weak
    // or strong Wolfe condition. Returns the accepted step (0 if none was
    // found); on success fOut/gOut hold the objective and gradient at x + t*d.
    //
    // Reference: A. S. Lewis and M. L. Overton, "Nonsmooth optimization via
    // quasi-Newton methods", Mathematical Programming 141(1), 2013.
    template <typename ObjFn>
    Scalar lineSearch(
        ObjFn&& objfn, const Scalar fx,
        const Vector<Scalar>& x, const Vector<Scalar>& g, const Vector<Scalar>& d,
        const Scalar tMax, Scalar& fOut, Vector<Scalar>& gOut) const
    {
        const Scalar armijoCons = Scalar(1e-4);
        const Scalar wolfeCons = Scalar(0.9);

        std::size_t iter = 0;
        bool bSet = false;      // interval upper limit has been set at least once
        bool tOk = false;       // step passed both the Armijo and Wolfe checks
        Scalar a = 0;           // interval lower limit
        Scalar b = tMax;        // interval upper limit
        const Scalar s = g.dot(d); // directional derivative at t = 0

        // if s is not sufficiently negative, d is not a useful descent direction
        Scalar t = (s < -detail::eps<Scalar>() && tMax > Scalar(0))
                 ? std::min(Scalar(1), tMax)
                 : Scalar(0);
        Scalar tPrev = t;

        Vector<Scalar> gTrial(x.size());

        while (!tOk
               && t > Scalar(0)
               && (t < tMax || tPrev < tMax || iter == 0)
               && iter < m_lineSearchMaxIterations) {
            ++iter;

            const Scalar fTrial = objfn(x + t * d, &gTrial);
            const Scalar h = fTrial - fx;         // h(t) = f(x + t*d) - f(x)
            const Scalar hp = gTrial.dot(d);      // h'(t)

            if (h >= armijoCons * s * t) {
                // Armijo condition failed: the decrease is insufficient
                b = t;
                bSet = true;
            } else if (m_strongWolfe
                       && std::abs(hp) >= std::abs(wolfeCons * s)
                       && !(hp < Scalar(0) && t == tMax)) {
                // strong Wolfe condition failed: |h'(t)| must decrease sufficiently
                if (hp < Scalar(0)) {
                    a = t;
                } else {
                    b = t;
                    bSet = true;
                }
            } else if (!m_strongWolfe
                       && hp <= wolfeCons * s
                       && t != tMax) {
                // weak Wolfe condition failed: h'(t) must increase sufficiently
                a = t;
            } else {
                tOk = true;
                fOut = fTrial;
                gOut = gTrial;
            }

            if (!tOk) {
                tPrev = t;

                if (bSet) {
                    // upper limit known: bisect the interval
                    t = (a + b) / 2;
                } else if (t < tMax) {
                    // upper limit unknown: keep expanding
                    t = 2 * a;
                }

                t = std::min(std::max(t, Scalar(0)), tMax);
            }
        }

        return tOk ? t : Scalar(0);
    }

private:
    std::size_t m_maxIterations = 2000;
    std::size_t m_maxFunctionEvaluations = 0;
    Scalar m_gradientTolerance = Scalar(1e-06);
    Scalar m_relObjectiveChangeTolerance = Scalar(1e-11);
    Scalar m_relSolutionChangeTolerance = Scalar(1e-11);
    std::size_t m_memory = 10;
    bool m_strongWolfe = false;
    std::size_t m_lineSearchMaxIterations = 25;

    // limited-memory matrices (notation follows the L-BFGS-B paper)
    Index m_m = 0;             // current number of stored correction pairs
    Scalar m_th = Scalar(1);   // theta scaling parameter
    Scalar m_thInv = Scalar(1);

    Matrix<Scalar> m_I;        // identity (n x n)
    Matrix<Scalar> m_S;        // correction pairs x_{k+1} - x_k (n x m)
    Matrix<Scalar> m_Y;        // correction pairs g_{k+1} - g_k (n x m)
    Matrix<Scalar> m_SS;       // S^T*S (m x m)
    Matrix<Scalar> m_SY;       // S^T*Y (m x m)
    Matrix<Scalar> m_YY;       // Y^T*Y (m x m)

    Matrix<Scalar> m_W;        // W from the paper (n x 2m)
    Matrix<Scalar> m_Wb;       // W-bar from the paper (n x 2m)

    Matrix<Scalar> m_M;        // M from the paper (2m x 2m)
    Matrix<Scalar> m_Mb;       // M-bar from the paper (2m x 2m)

    Vector<Scalar> m_c;        // W^T*(xc - x) accumulated during the Cauchy search (2m x 1)

    std::vector<Index> m_freeSet;
    std::vector<Index> m_activeSet;
};

} // namespace globopt

#endif
