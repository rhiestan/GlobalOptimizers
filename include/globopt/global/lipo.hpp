// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// MaxLIPO+TR global optimizer, ported from dlib's global_function_search /
// upper_bound_function / trust-region machinery,
// Copyright (C) 2017 Davis E. King (davis@dlib.net), Boost Software License.
//
// References:
//   C. Malherbe and N. Vayatis, "Global optimization of Lipschitz functions",
//   ICML 2017.
//   M.J.D. Powell, "The NEWUOA software for unconstrained optimization
//   without derivatives" (quadratic model fitting).
//   J. Nocedal, S. Wright, "Numerical Optimization", algorithm 4.3
//   (trust region subproblem).

#ifndef GLOBOPT_GLOBAL_LIPO_HPP
#define GLOBOPT_GLOBAL_LIPO_HPP

#include "../core/optimizer.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace globopt {
namespace detail {
namespace lipo {

// ---------------------------------------------------------------------------
// Hard-margin linear SVM via dual coordinate descent, with the last weight
// fixed to 1 (dlib's svm_c_linear_dcd_trainer with C = infinity and
// force_last_weight_to_1). All labels are +1. Samples are sparse
// (index, value) pairs over a feature space of size numWeights.
// ---------------------------------------------------------------------------

template <typename Scalar>
using SparseSample = std::vector<std::pair<Index, Scalar>>;

template <typename Scalar>
inline void solveDcdLastWeight1(
    const std::vector<SparseSample<Scalar>>& x,
    const Index numWeights,
    const Scalar eps,
    std::mt19937_64& rng,
    Vector<Scalar>& w,
    std::vector<Scalar>& alpha)
{
    const std::size_t m = x.size();

    w = Vector<Scalar>::Zero(numWeights);
    w(numWeights - 1) = 1;
    alpha.assign(m, Scalar(0));

    // Q_i = squared norm of the sample excluding the forced-to-1 dimension
    std::vector<Scalar> Q(m);
    std::vector<std::size_t> index;
    index.reserve(m);
    for (std::size_t i = 0; i < m; ++i) {
        Scalar q = 0;
        for (const auto& kv : x[i]) {
            if (kv.first < numWeights - 1) {
                q += kv.second * kv.second;
            }
        }
        Q[i] = q;
        if (q != Scalar(0)) {
            index.push_back(i);
        }
    }

    const std::size_t maxIterations = 10000; // dlib's default

    for (std::size_t iter = 0; iter < maxIterations; ++iter) {
        Scalar pgMax = -detail::inf<Scalar>();
        Scalar pgMin = detail::inf<Scalar>();

        for (std::size_t i = 0; i + 1 < index.size(); ++i) {
            std::uniform_int_distribution<std::size_t> pick(i, index.size() - 1);
            std::swap(index[i], index[pick(rng)]);
        }

        for (const std::size_t i : index) {
            Scalar G = -1;
            for (const auto& kv : x[i]) {
                G += w(kv.first) * kv.second;
            }

            Scalar pg = 0;
            if (alpha[i] == Scalar(0)) {
                if (G < 0) {
                    pg = G;
                }
            } else {
                pg = G;
            }

            pgMax = std::max(pgMax, pg);
            pgMin = std::min(pgMin, pg);

            if (std::abs(pg) > Scalar(1e-12)) {
                const Scalar alphaOld = alpha[i];
                alpha[i] = std::max(alpha[i] - G / Q[i], Scalar(0));
                const Scalar delta = alpha[i] - alphaOld;
                for (const auto& kv : x[i]) {
                    w(kv.first) += delta * kv.second;
                }
                w(numWeights - 1) = 1;
            }
        }

        if (pgMax - pgMin <= eps) {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Upper bound function (port of dlib's upper_bound_function): the piecewise
// bound U(x) = min_i [ y_i + sqrt(offset_i + sum_k slope_k (x_k - x_ik)^2) ],
// with slopes and per-point noise offsets learned from pairwise constraints
// by the DCD solver above.
// ---------------------------------------------------------------------------

template <typename Scalar>
class UpperBoundFunction {
public:
    UpperBoundFunction(const Scalar relativeNoiseMagnitude, const Scalar solverEps)
        : m_relativeNoiseMagnitude(relativeNoiseMagnitude), m_solverEps(solverEps)
    {
    }

    Index numPoints() const { return static_cast<Index>(m_ys.size()); }
    const std::vector<Vector<Scalar>>& pointsX() const { return m_xs; }
    const std::vector<Scalar>& pointsY() const { return m_ys; }

    void add(const Vector<Scalar>& x, const Scalar y, std::mt19937_64& rng)
    {
        m_xs.push_back(x);
        m_ys.push_back(y);

        if (m_ys.size() < 2) {
            return;
        }

        if (m_ys.size() <= 4) {
            // rebuild from scratch using all pairwise constraints
            m_activeConstraints.clear();
        } else {
            // add constraints between the new point and the old points
            for (std::size_t i = 0; i + 1 < m_xs.size(); ++i) {
                m_activeConstraints.emplace_back(i, m_xs.size() - 1);
            }
        }

        learnParams(rng);
    }

    Scalar operator()(const Vector<Scalar>& x) const
    {
        Scalar upperBound = detail::inf<Scalar>();

        for (std::size_t i = 0; i < m_xs.size(); ++i) {
            const Scalar localBound = m_ys[i]
                + std::sqrt(m_offsets[i] + (m_slopes.array() * (x - m_xs[i]).array().square()).sum());
            upperBound = std::min(upperBound, localBound);
        }

        return upperBound;
    }

private:
    void learnParams(std::mt19937_64& rng)
    {
        const Index dims = m_xs[0].size();
        const std::size_t numPts = m_xs.size();

        // normalize the data so the optimization is well conditioned (sample
        // standard deviations, matching dlib's running_stats)
        Vector<Scalar> xMean = Vector<Scalar>::Zero(dims), xStd = Vector<Scalar>::Zero(dims);
        Scalar yMean = 0, yStd = 0;
        for (std::size_t i = 0; i < numPts; ++i) {
            xMean += m_xs[i];
            yMean += m_ys[i];
        }
        xMean /= static_cast<Scalar>(numPts);
        yMean /= static_cast<Scalar>(numPts);
        for (std::size_t i = 0; i < numPts; ++i) {
            xStd.array() += (m_xs[i] - xMean).array().square();
            yStd += (m_ys[i] - yMean) * (m_ys[i] - yMean);
        }
        xStd = (xStd / static_cast<Scalar>(numPts - 1)).cwiseSqrt();
        yStd = std::sqrt(yStd / static_cast<Scalar>(numPts - 1));

        const Scalar yScale = Scalar(1) / yStd;
        Vector<Scalar> xScale(dims);
        for (Index k = 0; k < dims; ++k) {
            xScale(k) = Scalar(1) / (xStd(k) * yScale);
        }

        std::vector<SparseSample<Scalar>> samples;

        auto addConstraint = [&](std::size_t i, std::size_t j) {
            SparseSample<Scalar> samp;
            samp.reserve(static_cast<std::size_t>(dims) + 2);
            for (Index k = 0; k < dims; ++k) {
                const Scalar temp = (m_xs[i](k) - m_xs[j](k)) * xScale(k) * yScale;
                samp.emplace_back(k, temp * temp);
            }

            if (m_ys[i] > m_ys[j]) {
                samp.emplace_back(dims + static_cast<Index>(j), m_relativeNoiseMagnitude);
            } else {
                samp.emplace_back(dims + static_cast<Index>(i), m_relativeNoiseMagnitude);
            }

            const Scalar diff = (m_ys[i] - m_ys[j]) * yScale;
            samp.emplace_back(dims + static_cast<Index>(numPts), 1 - diff * diff);

            samples.push_back(std::move(samp));
        };

        const bool allPairs = m_activeConstraints.empty();
        if (allPairs) {
            samples.reserve(numPts * (numPts - 1) / 2);
            for (std::size_t i = 0; i < numPts; ++i) {
                for (std::size_t j = i + 1; j < numPts; ++j) {
                    addConstraint(i, j);
                }
            }
        } else {
            samples.reserve(m_activeConstraints.size());
            for (const auto& p : m_activeConstraints) {
                addConstraint(p.first, p.second);
            }
        }

        const Index numWeights = dims + static_cast<Index>(numPts) + 1;
        Vector<Scalar> w;
        std::vector<Scalar> alpha;
        solveDcdLastWeight1(samples, numWeights, m_solverEps, rng, w, alpha);

        // keep only the active constraints so future add() calls stay cheap
        if (allPairs) {
            std::size_t k = 0;
            m_activeConstraints.clear();
            for (std::size_t i = 0; i < numPts; ++i) {
                for (std::size_t j = i + 1; j < numPts; ++j) {
                    if (alpha[k++] != Scalar(0)) {
                        m_activeConstraints.emplace_back(i, j);
                    }
                }
            }
        } else {
            std::vector<std::pair<std::size_t, std::size_t>> stillActive;
            for (std::size_t i = 0; i < alpha.size(); ++i) {
                if (alpha[i] != Scalar(0)) {
                    stillActive.push_back(m_activeConstraints[i]);
                }
            }
            m_activeConstraints.swap(stillActive);
        }

        m_slopes.resize(dims);
        for (Index k = 0; k < dims; ++k) {
            m_slopes(k) = w(k) * xScale(k) * xScale(k);
        }

        m_offsets.assign(numPts, Scalar(0));
        for (std::size_t i = 0; i < numPts; ++i) {
            m_offsets[i] = w(dims + static_cast<Index>(i)) * m_relativeNoiseMagnitude;
        }
    }

    Scalar m_relativeNoiseMagnitude;
    Scalar m_solverEps;
    std::vector<std::pair<std::size_t, std::size_t>> m_activeConstraints;

    std::vector<Vector<Scalar>> m_xs;
    std::vector<Scalar> m_ys;
    std::vector<Scalar> m_offsets;
    Vector<Scalar> m_slopes;
};

// ---------------------------------------------------------------------------
// Trust region subproblem: minimize 0.5 p'Bp + g'p subject to ||p|| <= radius
// (algorithm 4.3 of Nocedal and Wright), plus the box-bounded variant that
// greedily pins the most violated variable to its bound and re-solves.
// ---------------------------------------------------------------------------

template <typename Scalar>
inline void solveTrustRegionSubproblem(
    const Matrix<Scalar>& B,
    const Vector<Scalar>& g,
    const Scalar radius,
    Vector<Scalar>& p,
    const Scalar eps,
    const std::size_t maxIter)
{
    const Index n = g.size();

    p = Vector<Scalar>::Zero(n);

    const Scalar numericEps = B.diagonal().cwiseAbs().maxCoeff() * detail::eps<Scalar>();

    // Gershgorin lower bound on the eigenvalues of B
    Scalar minEigLowerBound = detail::inf<Scalar>();
    for (Index i = 0; i < n; ++i) {
        minEigLowerBound = std::min(minEigLowerBound,
                                    B(i, i) - (B.row(i).cwiseAbs().sum() - std::abs(B(i, i))));
    }

    const Scalar gNorm = g.norm();

    Scalar lambda = 0;
    Scalar lambdaMin = 0;
    Scalar lambdaMax = std::min(std::max(gNorm / radius - minEigLowerBound, Scalar(0)),
                                std::numeric_limits<Scalar>::max());

    if (gNorm < numericEps && minEigLowerBound > numericEps) {
        return;
    }

    Scalar lambdaDelta = 0;
    bool cholSucceededOnce = false;

    for (std::size_t i = 0; i < maxIter; ++i) {
        Matrix<Scalar> shifted = B;
        shifted.diagonal().array() += lambda;
        Eigen::LLT<Matrix<Scalar>> llt(shifted);

        if (llt.info() != Eigen::Success) {
            // if B is indefinite and g is zero, go straight to the
            // eigenvalue method below
            if (gNorm <= numericEps) {
                break;
            }

            lambdaMin = lambda;
            const Scalar a = Scalar(0.10);
            lambda = (1 - a) * lambda + a * lambdaMax;
            continue;
        }
        cholSucceededOnce = true;

        // solve L q = -g, then L' p = q
        const Vector<Scalar> q = llt.matrixL().solve(-g);
        const Scalar qNorm = q.norm();
        p = llt.matrixL().transpose().solve(q);
        const Scalar pNorm = p.norm();

        if (lambda == Scalar(0)) {
            if (pNorm < radius) {
                return;
            }
        } else {
            if (std::abs(pNorm - radius) / radius < eps) {
                return;
            }
        }

        if (pNorm < radius) {
            lambdaMax = lambda;
        } else {
            lambdaMin = lambda;
        }

        if (pNorm <= radius * detail::eps<Scalar>()) {
            const Scalar a = Scalar(0.01);
            lambda = (1 - a) * lambdaMin + a * lambdaMax;
            continue;
        }

        const Scalar oldLambda = lambda;

        lambda = lambda + (qNorm / pNorm) * (qNorm / pNorm) * (pNorm - radius) / radius;

        const Scalar gap = (lambdaMax - lambdaMin) * Scalar(0.01);
        lambda = std::min(std::max(lambda, lambdaMin + gap), lambdaMax - gap);

        lambdaDelta += std::abs(lambda - oldLambda);
        if (lambdaDelta > 3 * (lambdaMax - lambdaMin)) {
            lambda = (lambdaMin + lambdaMax) / 2;
            lambdaDelta = 0;
        }
    }

    // probably the "hard case": use an eigenvalue decomposition
    Eigen::SelfAdjointEigenSolver<Matrix<Scalar>> es(Scalar(0.5) * (B + B.transpose()));
    Vector<Scalar> ev = es.eigenvalues();
    const Index minEigIdx = 0; // Eigen sorts eigenvalues in increasing order

    ev.array() -= ev.minCoeff();
    const Scalar evThreshold = ev.cwiseAbs().maxCoeff() * detail::eps<Scalar>();
    for (Index i = 0; i < ev.size(); ++i) {
        ev(i) = (ev(i) > evThreshold) ? Scalar(1) / ev(i) : Scalar(0);
    }

    Vector<Scalar> pHard = es.eigenvectors().transpose() * g;
    pHard = ev.asDiagonal() * pHard;
    pHard = es.eigenvectors() * pHard;

    if (!cholSucceededOnce) {
        p = Vector<Scalar>::Zero(n);
    }

    if (pHard.norm() < radius && pHard.norm() >= p.norm()) {
        const Scalar tau = std::sqrt(radius * radius - pHard.squaredNorm());
        p = pHard + tau * es.eigenvectors().col(minEigIdx);
    }
}

template <typename Scalar>
inline bool boundsViolated(const Vector<Scalar>& v, const Vector<Scalar>& l, const Vector<Scalar>& u)
{
    for (Index i = 0; i < v.size(); ++i) {
        if (!(l(i) <= v(i) && v(i) <= u(i))) {
            return true;
        }
    }
    return false;
}

template <typename Scalar>
inline void solveTrustRegionSubproblemBounded(
    const Matrix<Scalar>& B_,
    const Vector<Scalar>& g_,
    const Scalar radius_,
    Vector<Scalar>& p_,
    const Scalar eps,
    const std::size_t maxIter,
    const Vector<Scalar>& lower_,
    const Vector<Scalar>& upper_)
{
    solveTrustRegionSubproblem(B_, g_, radius_, p_, eps, maxIter);

    if (!boundsViolated(p_, lower_, upper_)) {
        return;
    }

    // Greedily find the most violated bound constraint, pin that variable to
    // its bound, remove it from the problem, and re-solve.
    Matrix<Scalar> B = B_;
    Vector<Scalar> g = g_;
    Scalar radius = radius_;
    Vector<Scalar> p = p_;
    Vector<Scalar> lower = lower_;
    Vector<Scalar> upper = upper_;

    std::vector<Index> idxs(static_cast<std::size_t>(g.size()));
    for (std::size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = static_cast<Index>(i);
    }

    auto removeEntry = [](Vector<Scalar>& v, Index k) {
        Vector<Scalar> r(v.size() - 1);
        r << v.head(k), v.tail(v.size() - k - 1);
        v = r;
    };

    while (boundsViolated(p, lower, upper)) {
        Index mostViolated = 0;
        Scalar maxViolation = 0;
        Scalar boundedValue = 0;
        for (Index i = 0; i < lower.size(); ++i) {
            if (!(lower(i) <= p(i) && p(i) <= upper(i))) {
                if (lower(i) - p(i) > maxViolation) {
                    maxViolation = lower(i) - p(i);
                    mostViolated = i;
                    boundedValue = lower(i);
                } else if (p(i) - upper(i) > maxViolation) {
                    maxViolation = p(i) - upper(i);
                    mostViolated = i;
                    boundedValue = upper(i);
                }
            }
        }

        p_(idxs[static_cast<std::size_t>(mostViolated)]) = boundedValue;

        idxs.erase(idxs.begin() + mostViolated);
        if (idxs.empty()) {
            break;
        }

        removeEntry(lower, mostViolated);
        removeEntry(upper, mostViolated);
        g += B.col(mostViolated) * boundedValue;
        removeEntry(g, mostViolated);
        removeEntry(p, mostViolated);

        const Index nn = B.rows();
        Matrix<Scalar> reduced(nn - 1, nn - 1);
        Index rr = 0;
        for (Index r = 0; r < nn; ++r) {
            if (r == mostViolated) continue;
            Index cc = 0;
            for (Index c = 0; c < nn; ++c) {
                if (c == mostViolated) continue;
                reduced(rr, cc) = B(r, c);
                ++cc;
            }
            ++rr;
        }
        B = reduced;

        const Scalar squaredRadius = radius * radius - boundedValue * boundedValue;
        if (squaredRadius <= Scalar(0)) {
            p.setZero();
            break;
        }
        radius = std::sqrt(squaredRadius);

        solveTrustRegionSubproblem(B, g, radius, p, eps, maxIter);
    }

    for (std::size_t i = 0; i < idxs.size(); ++i) {
        p_(idxs[i]) = p(static_cast<Index>(i));
    }
}

// ---------------------------------------------------------------------------
// Quadratic model fitting (port of dlib's fit_quadratic_to_points): find
// Q(x) = 0.5 x'Hx + g'x + c interpolating the given points; with fewer points
// than needed to determine Q uniquely, the minimum-Frobenius-norm-Hessian
// solution is used (equations 3.9 - 3.12 of the NEWUOA paper).
// ---------------------------------------------------------------------------

template <typename Scalar>
inline void fitQuadraticToPointsMse(
    const Matrix<Scalar>& X,
    const Vector<Scalar>& Y,
    Matrix<Scalar>& H,
    Vector<Scalar>& g,
    Scalar& c)
{
    const Index dims = X.rows();
    const Index M = X.cols();
    const Index P = (dims + 1) * (dims + 2) / 2;

    Matrix<Scalar> W(P, M);
    W.topRows(dims) = X;
    W.row(dims).setOnes();
    for (Index col = 0; col < M; ++col) {
        Index wr = dims + 1;
        for (Index r = 0; r < dims; ++r) {
            for (Index r2 = r; r2 < dims; ++r2) {
                W(wr, col) = X(r, col) * X(r2, col);
                if (r2 == r) {
                    W(wr, col) *= Scalar(0.5);
                }
                ++wr;
            }
        }
    }

    // minimum-norm least-squares solution of W' z = Y (dlib: pinv(W') * Y)
    const Vector<Scalar> z = Matrix<Scalar>(W.transpose())
        .completeOrthogonalDecomposition().solve(Y);

    c = z(dims);
    g = z.head(dims);

    H.resize(dims, dims);
    Index wr = dims + 1;
    for (Index r = 0; r < dims; ++r) {
        for (Index r2 = r; r2 < dims; ++r2) {
            H(r, r2) = H(r2, r) = z(wr++);
        }
    }
}

template <typename Scalar>
inline void fitQuadraticToPoints(
    const Matrix<Scalar>& X,
    const Vector<Scalar>& Y,
    Matrix<Scalar>& H,
    Vector<Scalar>& g,
    Scalar& c)
{
    const Index dims = X.rows();
    const Index M = X.cols();

    if (M >= (dims + 1) * (dims + 2) / 2) {
        fitQuadraticToPointsMse(X, Y, H, g, c);
        return;
    }

    Matrix<Scalar> W = Matrix<Scalar>::Zero(M + dims + 1, M + dims + 1);

    W.topLeftCorner(M, M) = Scalar(0.5) * (X.transpose() * X).array().square().matrix();
    W.block(0, M, M, 1).setOnes();
    W.block(M, 0, 1, M).setOnes();
    W.block(0, M + 1, M, dims) = X.transpose();
    W.block(M + 1, 0, dims, M) = X;

    Vector<Scalar> r = Vector<Scalar>::Zero(M + dims + 1);
    r.head(M) = Y;

    const Vector<Scalar> z = W.partialPivLu().solve(r);

    const Vector<Scalar> lambda = z.head(M);

    c = z(M);
    g = z.tail(dims);
    H = X * lambda.asDiagonal() * X.transpose();
}

// ---------------------------------------------------------------------------
// Trust region and upper bound sampling steps (port of dlib's qopt_impl)
// ---------------------------------------------------------------------------

template <typename Scalar>
struct QuadInterpResult {
    Vector<Scalar> bestX;
    Scalar predictedImprovement = std::numeric_limits<Scalar>::quiet_NaN();
};

template <typename Scalar>
inline QuadInterpResult<Scalar> findMaxQuadraticallyInterpolatedVector(
    const Vector<Scalar>& anchor,
    const Scalar radius,
    const std::vector<Vector<Scalar>>& x,
    const std::vector<Scalar>& y,
    const Vector<Scalar>& lower,
    const Vector<Scalar>& upper)
{
    const Index dims = anchor.size();

    Matrix<Scalar> X(dims, static_cast<Index>(x.size()));
    Vector<Scalar> Y(static_cast<Index>(x.size()));
    for (std::size_t i = 0; i < x.size(); ++i) {
        X.col(static_cast<Index>(i)) = x[i] - anchor;
        Y(static_cast<Index>(i)) = y[i];
    }

    Matrix<Scalar> H;
    Vector<Scalar> g;
    Scalar c;
    fitQuadraticToPoints(X, Y, H, g, c);

    Vector<Scalar> p;
    solveTrustRegionSubproblemBounded<Scalar>(-H, -g, radius, p, Scalar(0.001), 500,
                                              lower - anchor, upper - anchor);

    // never move more than radius from the anchor, in case the subproblem
    // wasn't solved accurately enough
    if (p.norm() >= radius) {
        p *= radius / p.norm();
    }

    QuadInterpResult<Scalar> result;
    result.predictedImprovement = Scalar(0.5) * p.dot(H * p) + p.dot(g);
    result.bestX = (anchor + p).cwiseMax(lower).cwiseMin(upper);
    return result;
}

template <typename Scalar>
inline QuadInterpResult<Scalar> pickNextSampleUsingTrustRegion(
    const std::vector<Vector<Scalar>>& xs,
    const std::vector<Scalar>& ys,
    Scalar& radius,
    const Vector<Scalar>& lower,
    const Vector<Scalar>& upper)
{
    const Index dims = lower.size();

    // use enough points to fill out a quadratic model, or the max available
    const std::size_t N = std::min(xs.size(),
                                   static_cast<std::size_t>((dims + 1) * (dims + 2) / 2));

    // find the best sample
    Scalar bestVal = Scalar(-1e300);
    Vector<Scalar> bestX;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        if (ys[i] > bestVal) {
            bestVal = ys[i];
            bestX = xs[i];
        }
    }

    // find the N nearest neighbors of bestX
    std::vector<std::pair<Scalar, std::size_t>> distances;
    distances.reserve(xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i) {
        distances.emplace_back((bestX - xs[i]).norm(), i);
    }
    std::sort(distances.begin(), distances.end());
    distances.resize(N);

    std::vector<Vector<Scalar>> x;
    std::vector<Scalar> y;
    for (const auto& idx : distances) {
        x.push_back(xs[idx.second]);
        y.push_back(ys[idx.second]);
    }

    if (radius == Scalar(0)) {
        for (const auto& idx : distances) {
            radius = std::max(radius, (bestX - xs[idx.second]).norm());
        }
        // shrink a little so the sampling of points near the current best
        // point always gets smaller
        radius *= Scalar(0.95);
    }

    return findMaxQuadraticallyInterpolatedVector(bestX, radius, x, y, lower, upper);
}

template <typename Scalar>
struct MaxUpperBoundResult {
    Vector<Scalar> x;
    Scalar predictedImprovement = 0;
    Scalar upperBound = 0;
};

template <typename Scalar>
inline Vector<Scalar> makeRandomVector(
    std::mt19937_64& rng,
    const Vector<Scalar>& lower,
    const Vector<Scalar>& upper)
{
    Vector<Scalar> v(lower.size());
    for (Index i = 0; i < v.size(); ++i) {
        std::uniform_real_distribution<Scalar> u(lower(i), upper(i));
        v(i) = u(rng);
    }
    return v;
}

template <typename Scalar>
inline MaxUpperBoundResult<Scalar> pickNextSampleAsMaxUpperBound(
    std::mt19937_64& rng,
    const UpperBoundFunction<Scalar>& ub,
    const Vector<Scalar>& lower,
    const Vector<Scalar>& upper,
    const std::size_t numRandomSamples)
{
    // simple random search for the maximum of the upper bound
    Scalar bestUbSoFar = -detail::inf<Scalar>();
    Vector<Scalar> v;
    for (std::size_t rounds = 0; rounds < numRandomSamples; ++rounds) {
        const Vector<Scalar> vtemp = makeRandomVector(rng, lower, upper);
        const Scalar bound = ub(vtemp);
        if (bound > bestUbSoFar) {
            bestUbSoFar = bound;
            v = vtemp;
        }
    }

    Scalar maxValue = -detail::inf<Scalar>();
    for (const Scalar y : ub.pointsY()) {
        maxValue = std::max(maxValue, y);
    }

    MaxUpperBoundResult<Scalar> result;
    result.x = std::move(v);
    result.predictedImprovement = bestUbSoFar - maxValue;
    result.upperBound = bestUbSoFar;
    return result;
}

} // namespace lipo
} // namespace detail

// ---------------------------------------------------------------------------
// The optimizer
// ---------------------------------------------------------------------------

/// MaxLIPO+TR global optimizer (dlib's global_function_search): derivative-
/// free search that alternates between maximizing a learned piecewise
/// Lipschitz upper bound of the objective (LIPO, Malherbe & Vayatis 2017,
/// with per-dimension slopes and a noise model) and a trust-region step on a
/// quadratic model fit to the points near the current best.
///
/// Requires finite box bounds (setBounds). The objective is never asked for
/// a gradient. The first evaluation is at the initial point passed to run().
///
/// If the value of the global optimum is known, set "target_objective" to
/// stop early with Status::Success once reached within "tolerance";
/// otherwise the search runs through its evaluation budget and returns the
/// best point found (Status::MaxFunctionEvaluationsReached).
///
/// Result::gradientNorm is not meaningful for this optimizer (NaN), and
/// Result::iterations equals the number of function evaluations.
template <typename Scalar>
class LIPO : public Optimizer<Scalar> {
public:
    using typename Optimizer<Scalar>::ObjectiveFn;

    LIPO()
    {
        this->registerParam("max_function_evaluations", &m_maxFunctionEvaluations,
                            "maximum number of function evaluations (0 = max(100, 10*n))");
        this->registerParam("target_objective", &m_targetObjective,
                            "value of the global optimum, if known (-inf to disable)");
        this->registerParam("tolerance", &m_tolerance,
                            "stop when the best objective is within this distance of target_objective");
        this->registerParam("pure_random_search_probability", &m_pureRandomSearchProbability,
                            "probability of ignoring the upper bound and sampling uniformly at random");
        this->registerParam("upper_bound_samples", &m_upperBoundSamples,
                            "number of Monte Carlo samples used to maximize the upper bound per iteration");
        this->registerParam("relative_noise_magnitude", &m_relativeNoiseMagnitude,
                            "assumed relative noise in objective values when fitting the upper bound");
        this->registerParam("trust_region_epsilon", &m_trustRegionEpsilon,
                            "minimum predicted improvement required to take a trust-region step");
        this->registerParam("upper_bound_solver_epsilon", &m_upperBoundSolverEpsilon,
                            "accuracy of the QP solver that fits the upper bound");
        this->registerParam("seed", &m_seed,
                            "random seed (0 = non-deterministic)");
    }

    const char* name() const override { return "LIPO"; }

protected:
    Result<Scalar> doOptimize(const ObjectiveFn& objective, const Vector<Scalar>& initialPoint) override
    {
        namespace lp = detail::lipo;

        const Index n = initialPoint.size();

        Result<Scalar> result;
        result.x = initialPoint;
        result.gradientNorm = std::numeric_limits<Scalar>::quiet_NaN();

        if (!this->m_boundsSet
            || !this->m_lowerBounds.allFinite() || !this->m_upperBounds.allFinite()) {
            result.status = Status::InvalidInput;
            result.message = "LIPO requires finite lower and upper bounds (setBounds)";
            return result;
        }
        for (Index i = 0; i < n; ++i) {
            if (this->m_lowerBounds(i) == this->m_upperBounds(i)) {
                result.status = Status::InvalidInput;
                result.message = "LIPO requires lower(i) != upper(i) for every variable";
                return result;
            }
        }

        const Vector<Scalar>& lower = this->m_lowerBounds;
        const Vector<Scalar>& upper = this->m_upperBounds;

        const std::size_t maxFnEvals = (m_maxFunctionEvaluations > 0)
            ? m_maxFunctionEvaluations
            : std::max<std::size_t>(100, 10 * static_cast<std::size_t>(n));

        std::mt19937_64 rng(m_seed != 0 ? static_cast<std::uint64_t>(m_seed) : std::random_device{}());
        std::uniform_real_distribution<Scalar> unit(Scalar(0), Scalar(1));

        std::size_t& evaluations = result.functionEvaluations;

        // internally we maximize -f, as dlib maximizes
        lp::UpperBoundFunction<Scalar> ub(m_relativeNoiseMagnitude, m_upperBoundSolverEpsilon);

        Scalar bestY = -detail::inf<Scalar>();
        Vector<Scalar> bestX = initialPoint;
        Scalar radius = 0;
        bool doTrustRegionStep = true;

        auto evaluatePoint = [&](const Vector<Scalar>& x, bool trustRegionGenerated,
                                 Scalar predictedImprovement, Scalar anchorValue) {
            const Scalar y = -objective(x, nullptr);
            ++evaluations;

            ub.add(x, y, rng);

            if (trustRegionGenerated) {
                // adjust the trust region radius based on how good this
                // evaluation was
                const Scalar measuredImprovement = y - anchorValue;
                const Scalar rho = measuredImprovement / std::abs(predictedImprovement);
                if (rho < Scalar(0.25)) {
                    radius *= Scalar(0.5);
                } else if (rho > Scalar(0.75)) {
                    radius *= 2;
                }
            }

            if (y > bestY) {
                if (!trustRegionGenerated && (x - bestX).norm() > radius * Scalar(1.001)) {
                    // reset the trust region radius since we made a big move
                    radius = 0;
                }
                bestY = y;
                bestX = x;
            }
        };

        auto finish = [&](Status status, const std::string& reason) {
            result.x = bestX;
            result.fval = -bestY;
            result.iterations = evaluations;
            result.status = status;
            result.message = reason;
            return result;
        };

        auto targetReached = [&]() {
            return -bestY < m_targetObjective + m_tolerance;
        };

        // the first evaluation is at the initial point (dlib evaluates the
        // center of the bounds; run() already receives a start point)
        evaluatePoint(initialPoint.cwiseMax(lower).cwiseMin(upper), false, 0, 0);

        while (true) {
            if (targetReached()) {
                return finish(Status::Success, "optimization terminated successfully");
            }
            if (evaluations >= maxFnEvals) {
                return finish(Status::MaxFunctionEvaluationsReached,
                              toString(Status::MaxFunctionEvaluationsReached));
            }

            // make sure we have at least max(3, n) evaluations before using
            // the model-driven steps
            if (ub.numPoints() < std::max<Index>(3, n)) {
                evaluatePoint(lp::makeRandomVector(rng, lower, upper), false, 0, 0);
                continue;
            }

            if (doTrustRegionStep && ub.numPoints() > n + 1) {
                auto tr = lp::pickNextSampleUsingTrustRegion(ub.pointsX(), ub.pointsY(),
                                                             radius, lower, upper);
                if (tr.predictedImprovement > m_trustRegionEpsilon) {
                    doTrustRegionStep = false;
                    evaluatePoint(tr.bestX, true, tr.predictedImprovement, bestY);
                    continue;
                }
            }

            // alternate between upper-bound and trust-region steps
            doTrustRegionStep = true;

            if (unit(rng) >= m_pureRandomSearchProbability) {
                auto mub = lp::pickNextSampleAsMaxUpperBound(rng, ub, lower, upper,
                                                             m_upperBoundSamples);
                if (mub.predictedImprovement > 0) {
                    evaluatePoint(mub.x, false, 0, 0);
                    continue;
                }
            }

            // pick entirely at random
            evaluatePoint(lp::makeRandomVector(rng, lower, upper), false, 0, 0);
        }
    }

private:
    std::size_t m_maxFunctionEvaluations = 0;
    Scalar m_targetObjective = -detail::inf<Scalar>();
    Scalar m_tolerance = Scalar(1e-05);
    Scalar m_pureRandomSearchProbability = Scalar(0.02);
    std::size_t m_upperBoundSamples = 5000;
    Scalar m_relativeNoiseMagnitude = Scalar(0.001);
    Scalar m_trustRegionEpsilon = 0;
    Scalar m_upperBoundSolverEpsilon = Scalar(0.0001);
    long long m_seed = 0;
};

} // namespace globopt

#endif
