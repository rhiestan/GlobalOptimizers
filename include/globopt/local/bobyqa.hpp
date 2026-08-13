// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// BOBYQA (Bound Optimization BY Quadratic Approximation), ported from the C
// translation in NLopt, Copyright (c) 2009 M. J. D. Powell (mjdp@cam.ac.uk),
// modifications Copyright (c) 2010 Massachusetts Institute of Technology,
// MIT License. The original Fortran was released by Powell with "no
// restrictions or charges"; the C translation is by S. G. Johnson (2009).
//
// Reference:
//   M.J.D. Powell, "The BOBYQA algorithm for bound constrained optimization
//   without derivatives", DAMTP report NA2009/06, Cambridge (2009).
//
// The routine names (prelim, altmov, trsbox, update, rescue, bobyqb) and the
// variable names of the original are kept so that this file can be checked
// against Powell's code line by line; array indices are 1-based for the same
// reason, which is why every vector and matrix below is allocated with one
// extra element per dimension.

#ifndef GLOBOPT_LOCAL_BOBYQA_HPP
#define GLOBOPT_LOCAL_BOBYQA_HPP

#include "../core/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace globopt {
namespace detail {
namespace bobyqa {

/// Why the iteration stopped. Mirrors the subset of nlopt_result that the
/// original translation produces.
enum class Stop {
    None,               ///< keep going
    RhoEnd,             ///< the trust region radius reached its final value
    TargetReached,      ///< the objective fell below the requested target
    ObjectiveTolerance, ///< the objective stopped changing
    MaxEvaluations,
    RoundoffLimited     ///< a denominator was destroyed by cancellation
};

/// Powell's BOBYQA, operating on rescaled coordinates in which the initial
/// trust region radius is the same in every direction.
template <typename Scalar>
class Solver {
public:
    using ObjectiveFn = ObjectiveFunction<Scalar>;

    Solver(const ObjectiveFn& objective, Index numVariables, Index numInterpolationPoints)
        : m_objective(&objective)
        , n(numVariables)
        , npt(numInterpolationPoints)
        , ndim(numInterpolationPoints + numVariables)
        , nptm(numInterpolationPoints - numVariables - 1)
    {
        x = Vector<Scalar>::Zero(n + 1);
        xl = Vector<Scalar>::Zero(n + 1);
        xu = Vector<Scalar>::Zero(n + 1);
        scale = Vector<Scalar>::Ones(n + 1);
        unscaled = Vector<Scalar>::Zero(n);

        xbase = Vector<Scalar>::Zero(n + 1);
        xopt = Vector<Scalar>::Zero(n + 1);
        gopt = Vector<Scalar>::Zero(n + 1);
        gnew = Vector<Scalar>::Zero(n + 1);
        xnew = Vector<Scalar>::Zero(n + 1);
        xalt = Vector<Scalar>::Zero(n + 1);
        d = Vector<Scalar>::Zero(n + 1);
        sl = Vector<Scalar>::Zero(n + 1);
        su = Vector<Scalar>::Zero(n + 1);
        hq = Vector<Scalar>::Zero(n * (n + 1) / 2 + 1);
        pq = Vector<Scalar>::Zero(npt + 1);
        fval = Vector<Scalar>::Zero(npt + 1);
        vlag = Vector<Scalar>::Zero(ndim + 1);
        w = Vector<Scalar>::Zero(2 * npt + 2 * n + 2);

        xpt = Matrix<Scalar>::Zero(npt + 1, n + 1);
        bmat = Matrix<Scalar>::Zero(ndim + 1, n + 1);
        zmat = Matrix<Scalar>::Zero(npt + 1, nptm + 1);
    }

    // -- inputs, all in scaled coordinates ----------------------------------

    Vector<Scalar> x;      ///< current point (1-based)
    Vector<Scalar> xl;     ///< lower bounds (1-based)
    Vector<Scalar> xu;     ///< upper bounds (1-based)
    Vector<Scalar> scale;  ///< unscaled = scaled * scale (1-based)

    Scalar rhobeg = Scalar(0.1);
    Scalar rhoend = Scalar(1e-8);
    Scalar minfMax = -inf<Scalar>();  ///< stop as soon as f drops below this
    Scalar relObjectiveTol = 0;       ///< 0 disables the objective-change test
    std::size_t maxEvaluations = 0;   ///< 0 = unlimited

    // -- outputs -------------------------------------------------------------

    Scalar minf = inf<Scalar>();
    std::size_t evaluations = 0;
    std::size_t iterations = 0;

    /// Set the bounds on moves from the initial point (SL and SU in Powell's
    /// notation), which the driver computes when it pushes the initial point
    /// away from a bound it is too close to.
    void setInitialBoundOffsets(const Index i, const Scalar lowerOffset, const Scalar upperOffset)
    {
        sl(i) = lowerOffset;
        su(i) = upperOffset;
    }

    /// Run BOBYQB, leaving the best point in x (still in scaled coordinates).
    Stop run() { return bobyqb(); }

private:
    const ObjectiveFn* m_objective;

    const Index n;
    const Index npt;
    const Index ndim;
    const Index nptm;

    Vector<Scalar> unscaled;

    // The arrays of Powell's BOBYQB. XBASE holds a shift of origin, XPT the
    // interpolation points relative to it, FVAL the objective values there,
    // XOPT the displacement of the trust region centre from XBASE, GOPT the
    // model gradient at XBASE+XOPT, HQ and PQ the explicit and implicit second
    // derivatives of the model, BMAT the last N columns of H, ZMAT the
    // factorization of the leading NPT by NPT submatrix of H, SL and SU the
    // bounds relative to XBASE, and VLAG the Lagrange function values.
    Vector<Scalar> xbase, xopt, gopt, gnew, xnew, xalt, d, sl, su;
    Vector<Scalar> hq, pq, fval, vlag, w;
    Matrix<Scalar> xpt, bmat, zmat;

    static Scalar square(Scalar v) { return v * v; }

    // -----------------------------------------------------------------------
    // Objective evaluation and stopping tests
    // -----------------------------------------------------------------------

    Scalar calfun(const Vector<Scalar>& point)
    {
        for (Index i = 1; i <= n; ++i) {
            unscaled(i - 1) = point(i) * scale(i);
        }
        ++evaluations;
        return (*m_objective)(unscaled, nullptr);
    }

    bool evalLimitReached() const
    {
        return maxEvaluations > 0 && evaluations >= maxEvaluations;
    }

    bool objectiveToleranceReached(Scalar f, Scalar fold) const
    {
        if (relObjectiveTol <= 0 || !std::isfinite(fold)) {
            return false;
        }
        return std::abs(f - fold)
            < relObjectiveTol * Scalar(0.5) * (std::abs(f) + std::abs(fold));
    }

    // -----------------------------------------------------------------------
    // UPDATE: revise BMAT and ZMAT for the new position of point KNEW
    // -----------------------------------------------------------------------

    void update(Vector<Scalar>& work, const Scalar beta, const Scalar denom, const Index knew)
    {
        Scalar ztest = 0;
        for (Index k = 1; k <= npt; ++k) {
            for (Index j = 1; j <= nptm; ++j) {
                ztest = std::max(ztest, std::abs(zmat(k, j)));
            }
        }
        ztest *= Scalar(1e-20);

        // Apply the rotations that put zeros in the KNEW-th row of ZMAT.
        for (Index j = 2; j <= nptm; ++j) {
            if (std::abs(zmat(knew, j)) > ztest) {
                Scalar temp = std::sqrt(square(zmat(knew, 1)) + square(zmat(knew, j)));
                const Scalar tempa = zmat(knew, 1) / temp;
                const Scalar tempb = zmat(knew, j) / temp;
                for (Index i = 1; i <= npt; ++i) {
                    temp = tempa * zmat(i, 1) + tempb * zmat(i, j);
                    zmat(i, j) = tempa * zmat(i, j) - tempb * zmat(i, 1);
                    zmat(i, 1) = temp;
                }
            }
            zmat(knew, j) = 0;
        }

        // Put the first NPT components of the KNEW-th column of HLAG into W,
        // and calculate the parameters of the updating formula.
        for (Index i = 1; i <= npt; ++i) {
            work(i) = zmat(knew, 1) * zmat(i, 1);
        }
        const Scalar alpha = work(knew);
        const Scalar tau = vlag(knew);
        vlag(knew) -= Scalar(1);

        // Complete the updating of ZMAT.
        Scalar temp = std::sqrt(denom);
        Scalar tempb = zmat(knew, 1) / temp;
        Scalar tempa = tau / temp;
        for (Index i = 1; i <= npt; ++i) {
            zmat(i, 1) = tempa * zmat(i, 1) - tempb * vlag(i);
        }

        // Finally, update the matrix BMAT.
        for (Index j = 1; j <= n; ++j) {
            const Index jp = npt + j;
            work(jp) = bmat(knew, j);
            tempa = (alpha * vlag(jp) - tau * work(jp)) / denom;
            tempb = (-beta * work(jp) - tau * vlag(jp)) / denom;
            for (Index i = 1; i <= jp; ++i) {
                bmat(i, j) = bmat(i, j) + tempa * vlag(i) + tempb * work(i);
                if (i > npt) {
                    bmat(jp, i - npt) = bmat(i, j);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // RESCUE: rebuild BMAT and ZMAT from scratch when rounding errors have
    // damaged the interpolation conditions
    // -----------------------------------------------------------------------

    Stop rescue(const Scalar delta, Index& kopt)
    {
        // ptsaux(1,j) and ptsaux(2,j) are the two positions of provisional
        // interpolation points along e_j; ptsid identifies the provisional
        // points, and rw is this routine's own working space.
        Matrix<Scalar> ptsaux = Matrix<Scalar>::Zero(3, n + 1);
        Vector<Scalar> ptsid = Vector<Scalar>::Zero(npt + 1);
        Vector<Scalar> rw = Vector<Scalar>::Zero(ndim + npt + 1);

        Index i = 0, j = 0, k = 0, ih = 0, jp = 0, ip = 0, iq = 0, iw = 0;
        Index ihp = 0, ihq = 0, jpn = 0, kpt = 0, kold = 0, nrem = 0, knew = 0;
        Scalar f = 0, xp = 0, xq = 0, den = 0, sum = 0, diff = 0, beta = 0;
        Scalar winc = 0, temp = 0, bsum = 0, hdiag = 0, fbase = 0, denom = 0;
        Scalar vquad = 0, sumpq = 0, dsqmin = 0, distsq = 0, vlmxsq = 0;

        const Index np = n + 1;
        const Scalar sfrac = Scalar(0.5) / static_cast<Scalar>(np);

        // Shift the interpolation points so that XOPT becomes the origin, and
        // set the elements of ZMAT to zero. The squares of the distances from
        // XOPT to the other interpolation points are put at the end of RW.
        sumpq = 0;
        winc = 0;
        for (k = 1; k <= npt; ++k) {
            distsq = 0;
            for (j = 1; j <= n; ++j) {
                xpt(k, j) -= xopt(j);
                distsq += square(xpt(k, j));
            }
            sumpq += pq(k);
            rw(ndim + k) = distsq;
            winc = std::max(winc, distsq);
            for (j = 1; j <= nptm; ++j) {
                zmat(k, j) = 0;
            }
        }

        // Update HQ so that HQ and PQ define the second derivatives of the
        // model after XBASE has been shifted to the trust region centre.
        ih = 0;
        for (j = 1; j <= n; ++j) {
            rw(j) = Scalar(0.5) * sumpq * xopt(j);
            for (k = 1; k <= npt; ++k) {
                rw(j) += pq(k) * xpt(k, j);
            }
            for (i = 1; i <= j; ++i) {
                ++ih;
                hq(ih) = hq(ih) + rw(i) * xopt(j) + rw(j) * xopt(i);
            }
        }

        // Shift XBASE, SL, SU and XOPT, zero BMAT and set PTSAUX.
        for (j = 1; j <= n; ++j) {
            xbase(j) += xopt(j);
            sl(j) -= xopt(j);
            su(j) -= xopt(j);
            xopt(j) = 0;
            ptsaux(1, j) = std::min(delta, su(j));
            ptsaux(2, j) = std::max(-delta, sl(j));
            if (ptsaux(1, j) + ptsaux(2, j) < Scalar(0)) {
                std::swap(ptsaux(1, j), ptsaux(2, j));
            }
            if (std::abs(ptsaux(2, j)) < Scalar(0.5) * std::abs(ptsaux(1, j))) {
                ptsaux(2, j) = Scalar(0.5) * ptsaux(1, j);
            }
            for (i = 1; i <= ndim; ++i) {
                bmat(i, j) = 0;
            }
        }
        fbase = fval(kopt);

        // Set the identifiers of the artificial interpolation points that are
        // along a coordinate direction from XOPT, and the corresponding
        // nonzero elements of BMAT and ZMAT.
        ptsid(1) = sfrac;
        for (j = 1; j <= n; ++j) {
            jp = j + 1;
            jpn = jp + n;
            ptsid(jp) = static_cast<Scalar>(j) + sfrac;
            if (jpn <= npt) {
                ptsid(jpn) = static_cast<Scalar>(j) / static_cast<Scalar>(np) + sfrac;
                temp = Scalar(1) / (ptsaux(1, j) - ptsaux(2, j));
                bmat(jp, j) = -temp + Scalar(1) / ptsaux(1, j);
                bmat(jpn, j) = temp + Scalar(1) / ptsaux(2, j);
                bmat(1, j) = -bmat(jp, j) - bmat(jpn, j);
                zmat(1, j) = std::sqrt(Scalar(2)) / std::abs(ptsaux(1, j) * ptsaux(2, j));
                zmat(jp, j) = zmat(1, j) * ptsaux(2, j) * temp;
                zmat(jpn, j) = -zmat(1, j) * ptsaux(1, j) * temp;
            } else {
                bmat(1, j) = -Scalar(1) / ptsaux(1, j);
                bmat(jp, j) = Scalar(1) / ptsaux(1, j);
                bmat(j + npt, j) = -Scalar(0.5) * square(ptsaux(1, j));
            }
        }

        // Set any remaining identifiers with their nonzero elements of ZMAT.
        if (npt >= n + np) {
            for (k = 2 * np; k <= npt; ++k) {
                iw = static_cast<Index>((static_cast<Scalar>(k - np) - Scalar(0.5))
                                        / static_cast<Scalar>(n));
                ip = k - np - iw * n;
                iq = ip + iw;
                if (iq > n) {
                    iq -= n;
                }
                ptsid(k) = static_cast<Scalar>(ip)
                    + static_cast<Scalar>(iq) / static_cast<Scalar>(np) + sfrac;
                temp = Scalar(1) / (ptsaux(1, ip) * ptsaux(1, iq));
                zmat(1, k - np) = temp;
                zmat(ip + 1, k - np) = -temp;
                zmat(iq + 1, k - np) = -temp;
                zmat(k, k - np) = temp;
            }
        }
        nrem = npt;
        kold = 1;
        knew = kopt;

        // Reorder the provisional points so that PTSID(KOLD) is exchanged with
        // PTSID(KNEW).
    L80:
        for (j = 1; j <= n; ++j) {
            std::swap(bmat(kold, j), bmat(knew, j));
        }
        for (j = 1; j <= nptm; ++j) {
            std::swap(zmat(kold, j), zmat(knew, j));
        }
        ptsid(kold) = ptsid(knew);
        ptsid(knew) = 0;
        rw(ndim + knew) = 0;
        --nrem;
        if (knew != kopt) {
            std::swap(vlag(kold), vlag(knew));

            // Update BMAT and ZMAT so that the status of the KNEW-th point can
            // be changed from provisional to original.
            update(rw, beta, denom, knew);
            if (nrem == 0) {
                goto L350;
            }
            for (k = 1; k <= npt; ++k) {
                rw(ndim + k) = std::abs(rw(ndim + k));
            }
        }

        // Pick the index KNEW of an original interpolation point that has not
        // yet replaced one of the provisional interpolation points.
    L120:
        dsqmin = 0;
        for (k = 1; k <= npt; ++k) {
            if (rw(ndim + k) > Scalar(0)) {
                if (dsqmin == Scalar(0) || rw(ndim + k) < dsqmin) {
                    knew = k;
                    dsqmin = rw(ndim + k);
                }
            }
        }
        if (dsqmin == Scalar(0)) {
            goto L260;
        }

        // Form the W-vector of the chosen original interpolation point.
        for (j = 1; j <= n; ++j) {
            rw(npt + j) = xpt(knew, j);
        }
        for (k = 1; k <= npt; ++k) {
            sum = 0;
            if (k == kopt) {
                // nothing to add
            } else if (ptsid(k) == Scalar(0)) {
                for (j = 1; j <= n; ++j) {
                    sum += rw(npt + j) * xpt(k, j);
                }
            } else {
                ip = static_cast<Index>(ptsid(k));
                if (ip > 0) {
                    sum = rw(npt + ip) * ptsaux(1, ip);
                }
                iq = static_cast<Index>(static_cast<Scalar>(np) * ptsid(k)
                                        - static_cast<Scalar>(ip * np));
                if (iq > 0) {
                    iw = 1;
                    if (ip == 0) {
                        iw = 2;
                    }
                    sum += rw(npt + iq) * ptsaux(iw, iq);
                }
            }
            rw(k) = Scalar(0.5) * sum * sum;
        }

        // Calculate VLAG and BETA for the required updating of the H matrix if
        // XPT(KNEW,.) is reinstated in the set of interpolation points.
        for (k = 1; k <= npt; ++k) {
            sum = 0;
            for (j = 1; j <= n; ++j) {
                sum += bmat(k, j) * rw(npt + j);
            }
            vlag(k) = sum;
        }
        beta = 0;
        for (j = 1; j <= nptm; ++j) {
            sum = 0;
            for (k = 1; k <= npt; ++k) {
                sum += zmat(k, j) * rw(k);
            }
            beta -= sum * sum;
            for (k = 1; k <= npt; ++k) {
                vlag(k) += sum * zmat(k, j);
            }
        }
        bsum = 0;
        distsq = 0;
        for (j = 1; j <= n; ++j) {
            sum = 0;
            for (k = 1; k <= npt; ++k) {
                sum += bmat(k, j) * rw(k);
            }
            jp = j + npt;
            bsum += sum * rw(jp);
            for (ip = npt + 1; ip <= ndim; ++ip) {
                sum += bmat(ip, j) * rw(ip);
            }
            bsum += sum * rw(jp);
            vlag(jp) = sum;
            distsq += square(xpt(knew, j));
        }
        beta = Scalar(0.5) * distsq * distsq + beta - bsum;
        vlag(kopt) += Scalar(1);

        // KOLD is the index of the provisional interpolation point that is
        // going to be deleted, chosen to avoid a small denominator in UPDATE.
        denom = 0;
        vlmxsq = 0;
        for (k = 1; k <= npt; ++k) {
            if (ptsid(k) != Scalar(0)) {
                hdiag = 0;
                for (j = 1; j <= nptm; ++j) {
                    hdiag += square(zmat(k, j));
                }
                den = beta * hdiag + square(vlag(k));
                if (den > denom) {
                    kold = k;
                    denom = den;
                }
            }
            vlmxsq = std::max(vlmxsq, square(vlag(k)));
        }
        if (denom <= vlmxsq * Scalar(0.01)) {
            rw(ndim + knew) = -rw(ndim + knew) - winc;
            goto L120;
        }
        goto L80;

        // All the final positions of the interpolation points have been chosen
        // although the changes are not yet included in XPT.
    L260:
        for (kpt = 1; kpt <= npt; ++kpt) {
            if (ptsid(kpt) == Scalar(0)) {
                continue;
            }

            if (evalLimitReached()) {
                return Stop::MaxEvaluations;
            }

            ih = 0;
            for (j = 1; j <= n; ++j) {
                rw(j) = xpt(kpt, j);
                xpt(kpt, j) = 0;
                temp = pq(kpt) * rw(j);
                for (i = 1; i <= j; ++i) {
                    ++ih;
                    hq(ih) += temp * rw(i);
                }
            }
            pq(kpt) = 0;
            ip = static_cast<Index>(ptsid(kpt));
            iq = static_cast<Index>(static_cast<Scalar>(np) * ptsid(kpt)
                                    - static_cast<Scalar>(ip * np));
            if (ip > 0) {
                xp = ptsaux(1, ip);
                xpt(kpt, ip) = xp;
            }
            if (iq > 0) {
                xq = ptsaux(1, iq);
                if (ip == 0) {
                    xq = ptsaux(2, iq);
                }
                xpt(kpt, iq) = xq;
            }

            // Set VQUAD to the value of the current model at the new point.
            vquad = fbase;
            if (ip > 0) {
                ihp = (ip + ip * ip) / 2;
                vquad += xp * (gopt(ip) + Scalar(0.5) * xp * hq(ihp));
            }
            if (iq > 0) {
                ihq = (iq + iq * iq) / 2;
                vquad += xq * (gopt(iq) + Scalar(0.5) * xq * hq(ihq));
                if (ip > 0) {
                    iw = std::max(ihp, ihq) - std::abs(ip - iq);
                    vquad += xp * xq * hq(iw);
                }
            }
            for (k = 1; k <= npt; ++k) {
                temp = 0;
                if (ip > 0) {
                    temp += xp * xpt(k, ip);
                }
                if (iq > 0) {
                    temp += xq * xpt(k, iq);
                }
                vquad += Scalar(0.5) * pq(k) * temp * temp;
            }

            // Calculate F at the new interpolation point.
            for (i = 1; i <= n; ++i) {
                rw(i) = std::min(std::max(xl(i), xbase(i) + xpt(kpt, i)), xu(i));
                if (xpt(kpt, i) == sl(i)) {
                    rw(i) = xl(i);
                }
                if (xpt(kpt, i) == su(i)) {
                    rw(i) = xu(i);
                }
            }

            f = calfun(rw);
            fval(kpt) = f;
            if (f < fval(kopt)) {
                kopt = kpt;
            }
            if (f < minfMax) {
                return Stop::TargetReached;
            }
            if (evalLimitReached()) {
                return Stop::MaxEvaluations;
            }

            diff = f - vquad;

            // Update the quadratic model.
            for (i = 1; i <= n; ++i) {
                gopt(i) += diff * bmat(kpt, i);
            }
            for (k = 1; k <= npt; ++k) {
                sum = 0;
                for (j = 1; j <= nptm; ++j) {
                    sum += zmat(k, j) * zmat(kpt, j);
                }
                temp = diff * sum;
                if (ptsid(k) == Scalar(0)) {
                    pq(k) += temp;
                } else {
                    ip = static_cast<Index>(ptsid(k));
                    iq = static_cast<Index>(static_cast<Scalar>(np) * ptsid(k)
                                            - static_cast<Scalar>(ip * np));
                    ihq = (iq * iq + iq) / 2;
                    if (ip == 0) {
                        hq(ihq) += temp * square(ptsaux(2, iq));
                    } else {
                        ihp = (ip * ip + ip) / 2;
                        hq(ihp) += temp * square(ptsaux(1, ip));
                        if (iq > 0) {
                            hq(ihq) += temp * square(ptsaux(1, iq));
                            iw = std::max(ihp, ihq) - std::abs(iq - ip);
                            hq(iw) += temp * ptsaux(1, ip) * ptsaux(1, iq);
                        }
                    }
                }
            }
            ptsid(kpt) = 0;
        }
    L350:
        return Stop::None;
    }

    // -----------------------------------------------------------------------
    // ALTMOV: pick a new position for the KNEW-th interpolation point that
    // gives a large denominator in the next call of UPDATE
    // -----------------------------------------------------------------------

    void altmov(const Index kopt, const Index knew, const Scalar adelt,
                Scalar& alpha, Scalar& cauchy)
    {
        Vector<Scalar> glag = Vector<Scalar>::Zero(n + 1);
        Vector<Scalar> hcol = Vector<Scalar>::Zero(npt + 1);
        Vector<Scalar> aw = Vector<Scalar>::Zero(2 * n + 1);

        Index i = 0, j = 0, k = 0, ilbd = 0, isbd = 0, iubd = 0, iflag = 0;
        Index ksav = 0, ibdsav = 0;
        Scalar ha = 0, gw = 0, diff = 0, slbd = 0, vlag_ = 0, subd = 0, temp = 0;
        Scalar step = 0, curv = 0, scale_ = 0, csave = 0, tempa = 0, tempb = 0;
        Scalar tempd = 0, sumin = 0, ggfree = 0, dderiv = 0, bigstp = 0;
        Scalar predsq = 0, presav = 0, distsq = 0, stpsav = 0, wfixsq = 0, wsqsav = 0;

        const Scalar constant = Scalar(1) + std::sqrt(Scalar(2));

        // Set the first NPT components of HCOL to the leading elements of the
        // KNEW-th column of the H matrix.
        for (k = 1; k <= npt; ++k) {
            hcol(k) = 0;
        }
        for (j = 1; j <= nptm; ++j) {
            temp = zmat(knew, j);
            for (k = 1; k <= npt; ++k) {
                hcol(k) += temp * zmat(k, j);
            }
        }
        alpha = hcol(knew);
        ha = Scalar(0.5) * alpha;

        // Calculate the gradient of the KNEW-th Lagrange function at XOPT.
        for (i = 1; i <= n; ++i) {
            glag(i) = bmat(knew, i);
        }
        for (k = 1; k <= npt; ++k) {
            temp = 0;
            for (j = 1; j <= n; ++j) {
                temp += xpt(k, j) * xopt(j);
            }
            temp = hcol(k) * temp;
            for (i = 1; i <= n; ++i) {
                glag(i) += temp * xpt(k, i);
            }
        }

        // Search for a large denominator along the straight lines through XOPT
        // and another interpolation point.
        presav = 0;
        for (k = 1; k <= npt; ++k) {
            if (k == kopt) {
                continue;
            }
            dderiv = 0;
            distsq = 0;
            for (i = 1; i <= n; ++i) {
                temp = xpt(k, i) - xopt(i);
                dderiv += glag(i) * temp;
                distsq += temp * temp;
            }
            subd = adelt / std::sqrt(distsq);
            slbd = -subd;
            ilbd = 0;
            iubd = 0;
            sumin = std::min(Scalar(1), subd);

            // Revise SLBD and SUBD if necessary because of the bounds.
            for (i = 1; i <= n; ++i) {
                temp = xpt(k, i) - xopt(i);
                if (temp > Scalar(0)) {
                    if (slbd * temp < sl(i) - xopt(i)) {
                        slbd = (sl(i) - xopt(i)) / temp;
                        ilbd = -i;
                    }
                    if (subd * temp > su(i) - xopt(i)) {
                        subd = std::max(sumin, (su(i) - xopt(i)) / temp);
                        iubd = i;
                    }
                } else if (temp < Scalar(0)) {
                    if (slbd * temp > su(i) - xopt(i)) {
                        slbd = (su(i) - xopt(i)) / temp;
                        ilbd = i;
                    }
                    if (subd * temp < sl(i) - xopt(i)) {
                        subd = std::max(sumin, (sl(i) - xopt(i)) / temp);
                        iubd = -i;
                    }
                }
            }

            // Seek a large modulus of the KNEW-th Lagrange function.
            if (k == knew) {
                diff = dderiv - Scalar(1);
                step = slbd;
                vlag_ = slbd * (dderiv - slbd * diff);
                isbd = ilbd;
                temp = subd * (dderiv - subd * diff);
                if (std::abs(temp) > std::abs(vlag_)) {
                    step = subd;
                    vlag_ = temp;
                    isbd = iubd;
                }
                tempd = Scalar(0.5) * dderiv;
                tempa = tempd - diff * slbd;
                tempb = tempd - diff * subd;
                if (tempa * tempb < Scalar(0)) {
                    temp = tempd * tempd / diff;
                    if (std::abs(temp) > std::abs(vlag_)) {
                        step = tempd / diff;
                        vlag_ = temp;
                        isbd = 0;
                    }
                }
            } else {
                // Search along each of the other lines through XOPT.
                step = slbd;
                vlag_ = slbd * (Scalar(1) - slbd);
                isbd = ilbd;
                temp = subd * (Scalar(1) - subd);
                if (std::abs(temp) > std::abs(vlag_)) {
                    step = subd;
                    vlag_ = temp;
                    isbd = iubd;
                }
                if (subd > Scalar(0.5)) {
                    if (std::abs(vlag_) < Scalar(0.25)) {
                        step = Scalar(0.5);
                        vlag_ = Scalar(0.25);
                        isbd = 0;
                    }
                }
                vlag_ *= dderiv;
            }

            temp = step * (Scalar(1) - step) * distsq;
            predsq = vlag_ * vlag_ * (vlag_ * vlag_ + ha * temp * temp);
            if (predsq > presav) {
                presav = predsq;
                ksav = k;
                stpsav = step;
                ibdsav = isbd;
            }
        }

        // Construct XNEW in a way that satisfies the bound constraints exactly.
        for (i = 1; i <= n; ++i) {
            temp = xopt(i) + stpsav * (xpt(ksav, i) - xopt(i));
            xnew(i) = std::max(sl(i), std::min(su(i), temp));
        }
        if (ibdsav < 0) {
            xnew(-ibdsav) = sl(-ibdsav);
        }
        if (ibdsav > 0) {
            xnew(ibdsav) = su(ibdsav);
        }

        // Prepare for the iterative method that assembles the constrained
        // Cauchy step in AW.
        bigstp = adelt + adelt;
        iflag = 0;
    L100:
        wfixsq = 0;
        ggfree = 0;
        for (i = 1; i <= n; ++i) {
            aw(i) = 0;
            tempa = std::min(xopt(i) - sl(i), glag(i));
            tempb = std::max(xopt(i) - su(i), glag(i));
            if (tempa > Scalar(0) || tempb < Scalar(0)) {
                aw(i) = bigstp;
                ggfree += square(glag(i));
            }
        }
        if (ggfree == Scalar(0)) {
            cauchy = 0;
            return;
        }

        // Investigate whether more components of AW can be fixed.
    L120:
        temp = adelt * adelt - wfixsq;
        if (temp > Scalar(0)) {
            wsqsav = wfixsq;
            step = std::sqrt(temp / ggfree);
            ggfree = 0;
            for (i = 1; i <= n; ++i) {
                if (aw(i) == bigstp) {
                    temp = xopt(i) - step * glag(i);
                    if (temp <= sl(i)) {
                        aw(i) = sl(i) - xopt(i);
                        wfixsq += square(aw(i));
                    } else if (temp >= su(i)) {
                        aw(i) = su(i) - xopt(i);
                        wfixsq += square(aw(i));
                    } else {
                        ggfree += square(glag(i));
                    }
                }
            }
            if (wfixsq > wsqsav && ggfree > Scalar(0)) {
                goto L120;
            }
        }

        // Set the remaining free components of AW and all components of XALT.
        gw = 0;
        for (i = 1; i <= n; ++i) {
            if (aw(i) == bigstp) {
                aw(i) = -step * glag(i);
                xalt(i) = std::max(sl(i), std::min(su(i), xopt(i) + aw(i)));
            } else if (aw(i) == Scalar(0)) {
                xalt(i) = xopt(i);
            } else if (glag(i) > Scalar(0)) {
                xalt(i) = sl(i);
            } else {
                xalt(i) = su(i);
            }
            gw += glag(i) * aw(i);
        }

        // Set CURV to the curvature of the KNEW-th Lagrange function along AW.
        curv = 0;
        for (k = 1; k <= npt; ++k) {
            temp = 0;
            for (j = 1; j <= n; ++j) {
                temp += xpt(k, j) * aw(j);
            }
            curv += hcol(k) * temp * temp;
        }
        if (iflag == 1) {
            curv = -curv;
        }
        if (curv > -gw && curv < -constant * gw) {
            scale_ = -gw / curv;
            for (i = 1; i <= n; ++i) {
                temp = xopt(i) + scale_ * aw(i);
                xalt(i) = std::max(sl(i), std::min(su(i), temp));
            }
            cauchy = square(Scalar(0.5) * gw * scale_);
        } else {
            cauchy = square(gw + Scalar(0.5) * curv);
        }

        // If IFLAG is zero, XALT is recalculated after reversing the sign of
        // GLAG; the version giving the larger CAUCHY is kept.
        if (iflag == 0) {
            for (i = 1; i <= n; ++i) {
                glag(i) = -glag(i);
                aw(n + i) = xalt(i);
            }
            csave = cauchy;
            iflag = 1;
            goto L100;
        }
        if (csave > cauchy) {
            for (i = 1; i <= n; ++i) {
                xalt(i) = aw(n + i);
            }
            cauchy = csave;
        }
    }

    // -----------------------------------------------------------------------
    // TRSBOX: approximately minimize the quadratic model within the trust
    // region, subject to the bounds, by truncated conjugate gradients
    // -----------------------------------------------------------------------

    void trsbox(const Scalar delta, Scalar& dsq, Scalar& crvmin)
    {
        Vector<Scalar> xbdi = Vector<Scalar>::Zero(n + 1);
        Vector<Scalar> s = Vector<Scalar>::Zero(n + 1);
        Vector<Scalar> hs = Vector<Scalar>::Zero(n + 1);
        Vector<Scalar> hred = Vector<Scalar>::Zero(n + 1);

        Index i = 0, j = 0, k = 0, ih = 0, iu = 0, iact = 0, nact = 0, isav = 0;
        Index iterc = 0, itcsav = 0, itermax = 0;
        Scalar ds = 0, dhd = 0, dhs = 0, cth = 0, shs = 0, sth = 0, ssq = 0;
        Scalar beta = 0, sdec = 0, blen = 0, angt = 0, qred = 0, temp = 0;
        Scalar xsav = 0, xsum = 0, angbd = 0, dredg = 0, sredg = 0, resid = 0;
        Scalar delsq = 0, ggsav = 0, tempa = 0, tempb = 0, redmax = 0;
        Scalar dredsq = 0, redsav = 0, gredsq = 0, rednew = 0;
        Scalar rdprev = 0, rdnext = 0, stplen = 0, stepsq = 0;

        // The sign of GOPT(I) gives the sign of the change to the I-th variable
        // that will reduce Q. NACT counts the variables fixed at a bound.
        iterc = 0;
        nact = 0;
        for (i = 1; i <= n; ++i) {
            xbdi(i) = 0;
            if (xopt(i) <= sl(i)) {
                if (gopt(i) >= Scalar(0)) {
                    xbdi(i) = Scalar(-1);
                }
            } else if (xopt(i) >= su(i)) {
                if (gopt(i) <= Scalar(0)) {
                    xbdi(i) = Scalar(1);
                }
            }
            if (xbdi(i) != Scalar(0)) {
                ++nact;
            }
            d(i) = 0;
            gnew(i) = gopt(i);
        }
        delsq = delta * delta;
        qred = 0;
        crvmin = Scalar(-1);

        // Set the next search direction of the conjugate gradient method.
    L20:
        beta = 0;
    L30:
        stepsq = 0;
        for (i = 1; i <= n; ++i) {
            if (xbdi(i) != Scalar(0)) {
                s(i) = 0;
            } else if (beta == Scalar(0)) {
                s(i) = -gnew(i);
            } else {
                s(i) = beta * s(i) - gnew(i);
            }
            stepsq += square(s(i));
        }
        if (stepsq == Scalar(0)) {
            goto L190;
        }
        if (beta == Scalar(0)) {
            gredsq = stepsq;
            itermax = iterc + n - nact;
        }
        if (gredsq * delsq <= qred * Scalar(1e-4) * qred) {
            goto L190;
        }

        // Multiply the search direction by the second derivative matrix of Q.
        goto L210;
    L50:
        resid = delsq;
        ds = 0;
        shs = 0;
        for (i = 1; i <= n; ++i) {
            if (xbdi(i) == Scalar(0)) {
                resid -= square(d(i));
                ds += s(i) * d(i);
                shs += s(i) * hs(i);
            }
        }
        if (resid <= Scalar(0)) {
            goto L90;
        }
        temp = std::sqrt(stepsq * resid + ds * ds);
        if (ds < Scalar(0)) {
            blen = (temp - ds) / stepsq;
        } else {
            blen = resid / (temp + ds);
        }
        stplen = blen;
        if (shs > Scalar(0)) {
            stplen = std::min(blen, gredsq / shs);
        }

        // Reduce STPLEN if necessary in order to preserve the simple bounds.
        iact = 0;
        for (i = 1; i <= n; ++i) {
            if (s(i) != Scalar(0)) {
                xsum = xopt(i) + d(i);
                if (s(i) > Scalar(0)) {
                    temp = (su(i) - xsum) / s(i);
                } else {
                    temp = (sl(i) - xsum) / s(i);
                }
                if (temp < stplen) {
                    stplen = temp;
                    iact = i;
                }
            }
        }

        // Update CRVMIN, GNEW and D.
        sdec = 0;
        if (stplen > Scalar(0)) {
            ++iterc;
            temp = shs / stepsq;
            if (iact == 0 && temp > Scalar(0)) {
                crvmin = std::min(crvmin, temp);
                if (crvmin == Scalar(-1)) {
                    crvmin = temp;
                }
            }
            ggsav = gredsq;
            gredsq = 0;
            for (i = 1; i <= n; ++i) {
                gnew(i) += stplen * hs(i);
                if (xbdi(i) == Scalar(0)) {
                    gredsq += square(gnew(i));
                }
                d(i) += stplen * s(i);
            }
            sdec = std::max(stplen * (ggsav - Scalar(0.5) * stplen * shs), Scalar(0));
            qred += sdec;
        }

        // Restart the conjugate gradient method if it has hit a new bound.
        if (iact > 0) {
            ++nact;
            xbdi(iact) = Scalar(1);
            if (s(iact) < Scalar(0)) {
                xbdi(iact) = Scalar(-1);
            }
            delsq -= square(d(iact));
            if (delsq <= Scalar(0)) {
                goto L90;
            }
            goto L20;
        }

        // If STPLEN is less than BLEN, apply another conjugate gradient
        // iteration or return.
        if (stplen < blen) {
            if (iterc == itermax) {
                goto L190;
            }
            if (sdec <= qred * Scalar(0.01)) {
                goto L190;
            }
            beta = gredsq / ggsav;
            goto L30;
        }
    L90:
        crvmin = 0;

        // Prepare for the alternative iteration, which stays on the trust
        // region boundary in the plane spanned by D and the gradient.
    L100:
        if (nact >= n - 1) {
            goto L190;
        }
        dredsq = 0;
        dredg = 0;
        gredsq = 0;
        for (i = 1; i <= n; ++i) {
            if (xbdi(i) == Scalar(0)) {
                dredsq += square(d(i));
                dredg += d(i) * gnew(i);
                gredsq += square(gnew(i));
                s(i) = d(i);
            } else {
                s(i) = 0;
            }
        }
        itcsav = iterc;
        goto L210;

        // Let S be a linear combination of the reduced D and the reduced G
        // that is orthogonal to the reduced D.
    L120:
        ++iterc;
        temp = gredsq * dredsq - dredg * dredg;
        if (temp <= qred * Scalar(1e-4) * qred) {
            goto L190;
        }
        temp = std::sqrt(temp);
        for (i = 1; i <= n; ++i) {
            if (xbdi(i) == Scalar(0)) {
                s(i) = (dredg * d(i) - dredsq * gnew(i)) / temp;
            } else {
                s(i) = 0;
            }
        }
        sredg = -temp;

        // Calculate an upper bound on the tangent of half the angle of the
        // alternative iteration, namely ANGBD.
        angbd = Scalar(1);
        iact = 0;
        for (i = 1; i <= n; ++i) {
            if (xbdi(i) == Scalar(0)) {
                tempa = xopt(i) + d(i) - sl(i);
                tempb = su(i) - xopt(i) - d(i);
                if (tempa <= Scalar(0)) {
                    ++nact;
                    xbdi(i) = Scalar(-1);
                    goto L100;
                } else if (tempb <= Scalar(0)) {
                    ++nact;
                    xbdi(i) = Scalar(1);
                    goto L100;
                }
                ssq = square(d(i)) + square(s(i));
                temp = ssq - square(xopt(i) - sl(i));
                if (temp > Scalar(0)) {
                    temp = std::sqrt(temp) - s(i);
                    if (angbd * temp > tempa) {
                        angbd = tempa / temp;
                        iact = i;
                        xsav = Scalar(-1);
                    }
                }
                temp = ssq - square(su(i) - xopt(i));
                if (temp > Scalar(0)) {
                    temp = std::sqrt(temp) + s(i);
                    if (angbd * temp > tempb) {
                        angbd = tempb / temp;
                        iact = i;
                        xsav = Scalar(1);
                    }
                }
            }
        }

        // Calculate HHD and some curvatures for the alternative iteration.
        goto L210;
    L150:
        shs = 0;
        dhs = 0;
        dhd = 0;
        for (i = 1; i <= n; ++i) {
            if (xbdi(i) == Scalar(0)) {
                shs += s(i) * hs(i);
                dhs += d(i) * hs(i);
                dhd += d(i) * hred(i);
            }
        }

        // Seek the greatest reduction in Q for a range of equally spaced values
        // of ANGT in [0,ANGBD].
        redmax = 0;
        isav = 0;
        redsav = 0;
        iu = static_cast<Index>(angbd * Scalar(17) + Scalar(3.1));
        for (i = 1; i <= iu; ++i) {
            angt = angbd * static_cast<Scalar>(i) / static_cast<Scalar>(iu);
            sth = (angt + angt) / (Scalar(1) + angt * angt);
            temp = shs + angt * (angt * dhd - dhs - dhs);
            rednew = sth * (angt * dredg - sredg - Scalar(0.5) * sth * temp);
            if (rednew > redmax) {
                redmax = rednew;
                isav = i;
                rdprev = redsav;
            } else if (i == isav + 1) {
                rdnext = rednew;
            }
            redsav = rednew;
        }

        // Return if the reduction is zero.
        if (isav == 0) {
            goto L190;
        }
        if (isav < iu) {
            temp = (rdnext - rdprev) / (redmax + redmax - rdprev - rdnext);
            angt = angbd * (static_cast<Scalar>(isav) + Scalar(0.5) * temp)
                / static_cast<Scalar>(iu);
        }
        cth = (Scalar(1) - angt * angt) / (Scalar(1) + angt * angt);
        sth = (angt + angt) / (Scalar(1) + angt * angt);
        temp = shs + angt * (angt * dhd - dhs - dhs);
        sdec = sth * (angt * dredg - sredg - Scalar(0.5) * sth * temp);
        if (sdec <= Scalar(0)) {
            goto L190;
        }

        // Update GNEW, D and HRED.
        dredg = 0;
        gredsq = 0;
        for (i = 1; i <= n; ++i) {
            gnew(i) = gnew(i) + (cth - Scalar(1)) * hred(i) + sth * hs(i);
            if (xbdi(i) == Scalar(0)) {
                d(i) = cth * d(i) + sth * s(i);
                dredg += d(i) * gnew(i);
                gredsq += square(gnew(i));
            }
            hred(i) = cth * hred(i) + sth * hs(i);
        }
        qred += sdec;
        if (iact > 0 && isav == iu) {
            ++nact;
            xbdi(iact) = xsav;
            goto L100;
        }

        if (sdec > qred * Scalar(0.01)) {
            goto L120;
        }
    L190:
        dsq = 0;
        for (i = 1; i <= n; ++i) {
            xnew(i) = std::max(std::min(xopt(i) + d(i), su(i)), sl(i));
            if (xbdi(i) == Scalar(-1)) {
                xnew(i) = sl(i);
            }
            if (xbdi(i) == Scalar(1)) {
                xnew(i) = su(i);
            }
            d(i) = xnew(i) - xopt(i);
            dsq += square(d(i));
        }
        return;

        // The following instructions multiply the current S-vector by the
        // second derivative matrix of the quadratic model, putting the product
        // in HS. They are reached from three different places above and can be
        // regarded as an external subroutine.
    L210:
        ih = 0;
        for (j = 1; j <= n; ++j) {
            hs(j) = 0;
            for (i = 1; i <= j; ++i) {
                ++ih;
                if (i < j) {
                    hs(j) += hq(ih) * s(i);
                }
                hs(i) += hq(ih) * s(j);
            }
        }
        for (k = 1; k <= npt; ++k) {
            if (pq(k) != Scalar(0)) {
                temp = 0;
                for (j = 1; j <= n; ++j) {
                    temp += xpt(k, j) * s(j);
                }
                temp *= pq(k);
                for (i = 1; i <= n; ++i) {
                    hs(i) += temp * xpt(k, i);
                }
            }
        }
        if (crvmin != Scalar(0)) {
            goto L50;
        }
        if (iterc > itcsav) {
            goto L150;
        }
        for (i = 1; i <= n; ++i) {
            hred(i) = hs(i);
        }
        goto L120;
    }

    // -----------------------------------------------------------------------
    // PRELIM: set the initial interpolation set and quadratic model
    // -----------------------------------------------------------------------

    Stop prelim(Index& kopt)
    {
        Index i = 0, j = 0, k = 0, ih = 0, nfm = 0, nfx = 0, ipt = 0, jpt = 0;
        Index itemp = 0, nf = 0;
        Scalar f = 0, fbeg = 0, diff = 0, temp = 0, stepa = 0, stepb = 0;

        const Scalar rhosq = rhobeg * rhobeg;
        const Scalar recip = Scalar(1) / rhosq;
        const Index np = n + 1;

        // Set XBASE to the initial vector of variables, and zero XPT, BMAT,
        // HQ, PQ and ZMAT.
        for (j = 1; j <= n; ++j) {
            xbase(j) = x(j);
            for (k = 1; k <= npt; ++k) {
                xpt(k, j) = 0;
            }
            for (i = 1; i <= ndim; ++i) {
                bmat(i, j) = 0;
            }
        }
        for (ih = 1; ih <= n * np / 2; ++ih) {
            hq(ih) = 0;
        }
        for (k = 1; k <= npt; ++k) {
            pq(k) = 0;
            for (j = 1; j <= npt - np; ++j) {
                zmat(k, j) = 0;
            }
        }

        // Begin the initialization procedure. The coordinates of the
        // displacement of the next initial interpolation point from XBASE are
        // set in XPT(NF+1,.).
        nf = 0;
    L50:
        nfm = nf;
        nfx = nf - n;
        ++nf;
        if (nfm <= 2 * n) {
            if (nfm >= 1 && nfm <= n) {
                stepa = rhobeg;
                if (su(nfm) == Scalar(0)) {
                    stepa = -stepa;
                }
                xpt(nf, nfm) = stepa;
            } else if (nfm > n) {
                stepa = xpt(nf - n, nfx);
                stepb = -rhobeg;
                if (sl(nfx) == Scalar(0)) {
                    stepb = std::min(Scalar(2) * rhobeg, su(nfx));
                }
                if (su(nfx) == Scalar(0)) {
                    stepb = std::max(Scalar(-2) * rhobeg, sl(nfx));
                }
                xpt(nf, nfx) = stepb;
            }
        } else {
            itemp = (nfm - np) / n;
            jpt = nfm - itemp * n - n;
            ipt = jpt + itemp;
            if (ipt > n) {
                itemp = jpt;
                jpt = ipt - n;
                ipt = itemp;
            }
            xpt(nf, ipt) = xpt(ipt + 1, ipt);
            xpt(nf, jpt) = xpt(jpt + 1, jpt);
        }

        // Calculate the next value of F.
        for (j = 1; j <= n; ++j) {
            x(j) = std::min(std::max(xl(j), xbase(j) + xpt(nf, j)), xu(j));
            if (xpt(nf, j) == sl(j)) {
                x(j) = xl(j);
            }
            if (xpt(nf, j) == su(j)) {
                x(j) = xu(j);
            }
        }
        f = calfun(x);
        fval(nf) = f;
        if (nf == 1) {
            fbeg = f;
            kopt = 1;
        } else if (f < fval(kopt)) {
            kopt = nf;
        }

        // Set the nonzero initial elements of BMAT and the quadratic model.
        if (nf <= 2 * n + 1) {
            if (nf >= 2 && nf <= n + 1) {
                gopt(nfm) = (f - fbeg) / stepa;
                if (npt < nf + n) {
                    bmat(1, nfm) = -Scalar(1) / stepa;
                    bmat(nf, nfm) = Scalar(1) / stepa;
                    bmat(npt + nfm, nfm) = -Scalar(0.5) * rhosq;
                }
            } else if (nf >= n + 2) {
                ih = nfx * (nfx + 1) / 2;
                temp = (f - fbeg) / stepb;
                diff = stepb - stepa;
                hq(ih) = Scalar(2) * (temp - gopt(nfx)) / diff;
                gopt(nfx) = (gopt(nfx) * stepb - temp * stepa) / diff;
                if (stepa * stepb < Scalar(0)) {
                    if (f < fval(nf - n)) {
                        fval(nf) = fval(nf - n);
                        fval(nf - n) = f;
                        if (kopt == nf) {
                            kopt = nf - n;
                        }
                        xpt(nf - n, nfx) = stepb;
                        xpt(nf, nfx) = stepa;
                    }
                }
                bmat(1, nfx) = -(stepa + stepb) / (stepa * stepb);
                bmat(nf, nfx) = -Scalar(0.5) / xpt(nf - n, nfx);
                bmat(nf - n, nfx) = -bmat(1, nfx) - bmat(nf, nfx);
                zmat(1, nfx) = std::sqrt(Scalar(2)) / (stepa * stepb);
                zmat(nf, nfx) = std::sqrt(Scalar(0.5)) / rhosq;
                zmat(nf - n, nfx) = -zmat(1, nfx) - zmat(nf, nfx);
            }
        } else {
            // Set the off-diagonal second derivatives of the Lagrange functions
            // and the initial quadratic model.
            ih = ipt * (ipt - 1) / 2 + jpt;
            zmat(1, nfx) = recip;
            zmat(nf, nfx) = recip;
            zmat(ipt + 1, nfx) = -recip;
            zmat(jpt + 1, nfx) = -recip;
            temp = xpt(nf, ipt) * xpt(nf, jpt);
            hq(ih) = (fbeg - fval(ipt + 1) - fval(jpt + 1) + f) / temp;
        }

        if (f < minfMax) {
            return Stop::TargetReached;
        }
        if (evalLimitReached()) {
            return Stop::MaxEvaluations;
        }
        if (nf < npt) {
            goto L50;
        }
        return Stop::None;
    }

    // -----------------------------------------------------------------------
    // BOBYQB: the main iteration
    // -----------------------------------------------------------------------

    Stop bobyqb()
    {
        Index i = 0, j = 0, k = 0, ih = 0, jj = 0, ip = 0, jp = 0;
        Index knew = 0, kopt = 0, ksav = 0, kbase = 0, itest = 0, ntrits = 0;
        Scalar f = 0, dx = 0, den = 0, dsq = 0, rho = 0, sum = 0, diff = 0;
        Scalar beta = 0, gisq = 0, temp = 0, suma = 0, sumb = 0, bsum = 0;
        Scalar fopt = 0, curv = 0, gqsq = 0, dist = 0, sumw = 0, sumz = 0;
        Scalar diffa = 0, diffb = 0, diffc = 0, hdiag = 0, alpha = 0, delta = 0;
        Scalar adelt = 0, denom = 0, bdtol = 0, delsq = 0;
        Scalar ratio = 0, dnorm = 0, vquad = 0, pqold = 0, sumpq = 0;
        Scalar scaden = 0, errbig = 0, cauchy = 0, fracsq = 0, biglsq = 0;
        Scalar densav = 0, bdtest = 0, crvmin = 0, frhosq = 0, distsq = 0;
        Scalar xoptsq = 0;
        std::size_t nresc = 0, nfsav = 0;

        const Index np = n + 1;
        const Index nh = n * np / 2;

        Stop rc = Stop::None;

        // PRELIM sets XBASE, XPT, FVAL, GOPT, HQ, PQ, BMAT and ZMAT for the
        // first iteration, together with KOPT.
        const Stop rc2 = prelim(kopt);
        xoptsq = 0;
        for (i = 1; i <= n; ++i) {
            xopt(i) = xpt(kopt, i);
            xoptsq += square(xopt(i));
        }
        if (rc2 != Stop::None) {
            rc = rc2;
            goto L720;
        }
        kbase = 1;

        rho = rhobeg;
        delta = rho;
        nresc = evaluations;
        ntrits = 0;
        diffa = 0;
        diffb = 0;
        itest = 0;
        nfsav = evaluations;

        // Update GOPT if necessary before the first iteration and after each
        // call of RESCUE that makes a call of CALFUN.
    L20:
        if (kopt != kbase) {
            ih = 0;
            for (j = 1; j <= n; ++j) {
                for (i = 1; i <= j; ++i) {
                    ++ih;
                    if (i < j) {
                        gopt(j) += hq(ih) * xopt(i);
                    }
                    gopt(i) += hq(ih) * xopt(j);
                }
            }
            if (evaluations > static_cast<std::size_t>(npt)) {
                for (k = 1; k <= npt; ++k) {
                    temp = 0;
                    for (j = 1; j <= n; ++j) {
                        temp += xpt(k, j) * xopt(j);
                    }
                    temp = pq(k) * temp;
                    for (i = 1; i <= n; ++i) {
                        gopt(i) += temp * xpt(k, i);
                    }
                }
            }
        }

        // Generate the next point in the trust region that provides a small
        // value of the quadratic model subject to the bounds. NTRITS counts the
        // trust region iterations since the last alternative iteration.
    L60:
        trsbox(delta, dsq, crvmin);
        dnorm = std::min(delta, std::sqrt(dsq));
        if (dnorm < Scalar(0.5) * rho) {
            ntrits = -1;
            distsq = square(Scalar(10) * rho);
            if (evaluations <= nfsav + 2) {
                goto L650;
            }

            // Either RHO is decreased or termination occurs if the errors in
            // the quadratic model at the last three interpolation points
            // compare favourably with predictions of likely improvements.
            errbig = std::max(std::max(diffa, diffb), diffc);
            frhosq = rho * Scalar(0.125) * rho;
            if (crvmin > Scalar(0) && errbig > frhosq * crvmin) {
                goto L650;
            }
            bdtol = errbig / rho;
            for (j = 1; j <= n; ++j) {
                bdtest = bdtol;
                if (xnew(j) == sl(j)) {
                    bdtest = gnew(j);
                }
                if (xnew(j) == su(j)) {
                    bdtest = -gnew(j);
                }
                if (bdtest < bdtol) {
                    curv = hq((j + j * j) / 2);
                    for (k = 1; k <= npt; ++k) {
                        curv += pq(k) * square(xpt(k, j));
                    }
                    bdtest += Scalar(0.5) * curv * rho;
                    if (bdtest < bdtol) {
                        goto L650;
                    }
                }
            }
            goto L680;
        }
        ++ntrits;

        // Severe cancellation is likely to occur if XOPT is too far from
        // XBASE, so shift XBASE to make XOPT zero.
    L90:
        if (dsq <= xoptsq * Scalar(0.001)) {
            fracsq = xoptsq * Scalar(0.25);
            sumpq = 0;
            for (k = 1; k <= npt; ++k) {
                sumpq += pq(k);
                sum = -Scalar(0.5) * xoptsq;
                for (i = 1; i <= n; ++i) {
                    sum += xpt(k, i) * xopt(i);
                }
                w(npt + k) = sum;
                temp = fracsq - Scalar(0.5) * sum;
                for (i = 1; i <= n; ++i) {
                    w(i) = bmat(k, i);
                    vlag(i) = sum * xpt(k, i) + temp * xopt(i);
                    ip = npt + i;
                    for (j = 1; j <= i; ++j) {
                        bmat(ip, j) = bmat(ip, j) + w(i) * vlag(j) + vlag(i) * w(j);
                    }
                }
            }

            // Then the revisions of BMAT that depend on ZMAT.
            for (jj = 1; jj <= nptm; ++jj) {
                sumz = 0;
                sumw = 0;
                for (k = 1; k <= npt; ++k) {
                    sumz += zmat(k, jj);
                    vlag(k) = w(npt + k) * zmat(k, jj);
                    sumw += vlag(k);
                }
                for (j = 1; j <= n; ++j) {
                    sum = (fracsq * sumz - Scalar(0.5) * sumw) * xopt(j);
                    for (k = 1; k <= npt; ++k) {
                        sum += vlag(k) * xpt(k, j);
                    }
                    w(j) = sum;
                    for (k = 1; k <= npt; ++k) {
                        bmat(k, j) += sum * zmat(k, jj);
                    }
                }
                for (i = 1; i <= n; ++i) {
                    ip = i + npt;
                    temp = w(i);
                    for (j = 1; j <= i; ++j) {
                        bmat(ip, j) += temp * w(j);
                    }
                }
            }

            // Complete the shift, including the changes to the second
            // derivative parameters of the quadratic model.
            ih = 0;
            for (j = 1; j <= n; ++j) {
                w(j) = -Scalar(0.5) * sumpq * xopt(j);
                for (k = 1; k <= npt; ++k) {
                    w(j) += pq(k) * xpt(k, j);
                    xpt(k, j) -= xopt(j);
                }
                for (i = 1; i <= j; ++i) {
                    ++ih;
                    hq(ih) = hq(ih) + w(i) * xopt(j) + xopt(i) * w(j);
                    bmat(npt + i, j) = bmat(npt + j, i);
                }
            }
            for (i = 1; i <= n; ++i) {
                xbase(i) += xopt(i);
                xnew(i) -= xopt(i);
                sl(i) -= xopt(i);
                su(i) -= xopt(i);
                xopt(i) = 0;
            }
            xoptsq = 0;
        }
        if (ntrits == 0) {
            goto L210;
        }
        goto L230;

        // XBASE is also moved to XOPT by a call of RESCUE, which regenerates
        // BMAT and ZMAT from scratch. It is called only if rounding errors have
        // reduced the denominator of the updating formula by at least a factor
        // of two.
    L190:
        nfsav = evaluations;
        kbase = kopt;
        {
            const Stop rc3 = rescue(delta, kopt);

            // XOPT is updated now in case the branch below to L720 is taken.
            xoptsq = 0;
            if (kopt != kbase) {
                for (i = 1; i <= n; ++i) {
                    xopt(i) = xpt(kopt, i);
                    xoptsq += square(xopt(i));
                }
            }
            if (rc3 != Stop::None) {
                rc = rc3;
                goto L720;
            }
        }
        nresc = evaluations;
        if (nfsav < evaluations) {
            nfsav = evaluations;
            goto L20;
        }
        if (ntrits > 0) {
            goto L60;
        }

        // Pick two alternative vectors of variables, relative to XBASE, that
        // are suitable as new positions of the KNEW-th interpolation point.
    L210:
        altmov(kopt, knew, adelt, alpha, cauchy);
        for (i = 1; i <= n; ++i) {
            d(i) = xnew(i) - xopt(i);
        }

        // Calculate VLAG and BETA for the current choice of D.
    L230:
        for (k = 1; k <= npt; ++k) {
            suma = 0;
            sumb = 0;
            sum = 0;
            for (j = 1; j <= n; ++j) {
                suma += xpt(k, j) * d(j);
                sumb += xpt(k, j) * xopt(j);
                sum += bmat(k, j) * d(j);
            }
            w(k) = suma * (Scalar(0.5) * suma + sumb);
            vlag(k) = sum;
            w(npt + k) = suma;
        }
        beta = 0;
        for (jj = 1; jj <= nptm; ++jj) {
            sum = 0;
            for (k = 1; k <= npt; ++k) {
                sum += zmat(k, jj) * w(k);
            }
            beta -= sum * sum;
            for (k = 1; k <= npt; ++k) {
                vlag(k) += sum * zmat(k, jj);
            }
        }
        dsq = 0;
        bsum = 0;
        dx = 0;
        for (j = 1; j <= n; ++j) {
            dsq += square(d(j));
            sum = 0;
            for (k = 1; k <= npt; ++k) {
                sum += w(k) * bmat(k, j);
            }
            bsum += sum * d(j);
            jp = npt + j;
            for (i = 1; i <= n; ++i) {
                sum += bmat(jp, i) * d(i);
            }
            vlag(jp) = sum;
            bsum += sum * d(j);
            dx += d(j) * xopt(j);
        }
        beta = dx * dx + dsq * (xoptsq + dx + dx + Scalar(0.5) * dsq) + beta - bsum;
        vlag(kopt) += Scalar(1);

        // If NTRITS is zero, the denominator may be increased by replacing the
        // step D of ALTMOV by a Cauchy step.
        if (ntrits == 0) {
            denom = square(vlag(knew)) + alpha * beta;
            if (denom < cauchy && cauchy > Scalar(0)) {
                for (i = 1; i <= n; ++i) {
                    xnew(i) = xalt(i);
                    d(i) = xnew(i) - xopt(i);
                }
                cauchy = 0;
                goto L230;
            }
            if (denom <= Scalar(0.5) * square(vlag(knew))) {
                if (evaluations > nresc) {
                    goto L190;
                }
                // Much cancellation in a denominator.
                rc = Stop::RoundoffLimited;
                goto L720;
            }
        } else {
            // Set KNEW to the index of the next interpolation point to be
            // deleted to make room for a trust region step.
            delsq = delta * delta;
            scaden = 0;
            biglsq = 0;
            knew = 0;
            for (k = 1; k <= npt; ++k) {
                if (k == kopt) {
                    continue;
                }
                hdiag = 0;
                for (jj = 1; jj <= nptm; ++jj) {
                    hdiag += square(zmat(k, jj));
                }
                den = beta * hdiag + square(vlag(k));
                distsq = 0;
                for (j = 1; j <= n; ++j) {
                    distsq += square(xpt(k, j) - xopt(j));
                }
                temp = std::max(Scalar(1), square(distsq / delsq));
                if (temp * den > scaden) {
                    scaden = temp * den;
                    knew = k;
                    denom = den;
                }
                biglsq = std::max(biglsq, temp * square(vlag(k)));
            }
            if (scaden <= Scalar(0.5) * biglsq) {
                if (evaluations > nresc) {
                    goto L190;
                }
                // Much cancellation in a denominator.
                rc = Stop::RoundoffLimited;
                goto L720;
            }
        }

        // Calculate the value of the objective function at XBASE+XNEW, unless
        // the limit on the number of evaluations has been reached.
    L360:
        for (i = 1; i <= n; ++i) {
            x(i) = std::min(std::max(xl(i), xbase(i) + xnew(i)), xu(i));
            if (xnew(i) == sl(i)) {
                x(i) = xl(i);
            }
            if (xnew(i) == su(i)) {
                x(i) = xu(i);
            }
        }

        if (evalLimitReached()) {
            rc = Stop::MaxEvaluations;
            goto L720;
        }

        ++iterations;
        f = calfun(x);
        if (ntrits == -1) {
            rc = Stop::RhoEnd;
            if (f < fval(kopt)) {
                minf = f;
                return rc;
            }
            goto L720;
        }

        if (f < minfMax) {
            minf = f;
            return Stop::TargetReached;
        }

        // Use the quadratic model to predict the change in F due to the step D,
        // and set DIFF to the error of this prediction.
        fopt = fval(kopt);
        vquad = 0;
        ih = 0;
        for (j = 1; j <= n; ++j) {
            vquad += d(j) * gopt(j);
            for (i = 1; i <= j; ++i) {
                ++ih;
                temp = d(i) * d(j);
                if (i == j) {
                    temp = Scalar(0.5) * temp;
                }
                vquad += hq(ih) * temp;
            }
        }
        for (k = 1; k <= npt; ++k) {
            vquad += Scalar(0.5) * pq(k) * square(w(npt + k));
        }
        diff = f - fopt - vquad;
        diffc = diffb;
        diffb = diffa;
        diffa = std::abs(diff);
        if (dnorm > rho) {
            nfsav = evaluations;
        }

        // Pick the next value of DELTA after a trust region step.
        if (ntrits > 0) {
            if (vquad >= Scalar(0)) {
                // A trust region step has failed to reduce Q.
                rc = Stop::RoundoffLimited;
                goto L720;
            }
            ratio = (f - fopt) / vquad;
            if (ratio <= Scalar(0.1)) {
                delta = std::min(Scalar(0.5) * delta, dnorm);
            } else if (ratio <= Scalar(0.7)) {
                delta = std::max(Scalar(0.5) * delta, dnorm);
            } else {
                delta = std::max(Scalar(0.5) * delta, dnorm + dnorm);
            }
            if (delta <= rho * Scalar(1.5)) {
                delta = rho;
            }

            // Recalculate KNEW and DENOM if the new F is less than FOPT.
            if (f < fopt) {
                ksav = knew;
                densav = denom;
                delsq = delta * delta;
                scaden = 0;
                biglsq = 0;
                knew = 0;
                for (k = 1; k <= npt; ++k) {
                    hdiag = 0;
                    for (jj = 1; jj <= nptm; ++jj) {
                        hdiag += square(zmat(k, jj));
                    }
                    den = beta * hdiag + square(vlag(k));
                    distsq = 0;
                    for (j = 1; j <= n; ++j) {
                        distsq += square(xpt(k, j) - xnew(j));
                    }
                    temp = std::max(Scalar(1), square(distsq / delsq));
                    if (temp * den > scaden) {
                        scaden = temp * den;
                        knew = k;
                        denom = den;
                    }
                    biglsq = std::max(biglsq, temp * square(vlag(k)));
                }
                if (scaden <= Scalar(0.5) * biglsq) {
                    knew = ksav;
                    denom = densav;
                }
            }
        }

        // Update BMAT and ZMAT, so that the KNEW-th interpolation point can be
        // moved. Also update the second derivative terms of the model.
        update(w, beta, denom, knew);
        ih = 0;
        pqold = pq(knew);
        pq(knew) = 0;
        for (i = 1; i <= n; ++i) {
            temp = pqold * xpt(knew, i);
            for (j = 1; j <= i; ++j) {
                ++ih;
                hq(ih) += temp * xpt(knew, j);
            }
        }
        for (jj = 1; jj <= nptm; ++jj) {
            temp = diff * zmat(knew, jj);
            for (k = 1; k <= npt; ++k) {
                pq(k) += temp * zmat(k, jj);
            }
        }

        // Include the new interpolation point, and make the changes to GOPT at
        // the old XOPT that are caused by the updating of the quadratic model.
        fval(knew) = f;
        for (i = 1; i <= n; ++i) {
            xpt(knew, i) = xnew(i);
            w(i) = bmat(knew, i);
        }
        for (k = 1; k <= npt; ++k) {
            suma = 0;
            for (jj = 1; jj <= nptm; ++jj) {
                suma += zmat(knew, jj) * zmat(k, jj);
            }
            if (!std::isfinite(suma)) {
                // Detect singularity, which happens if we run for too many
                // iterations (this test was added by S. G. Johnson).
                rc = Stop::RoundoffLimited;
                goto L720;
            }
            sumb = 0;
            for (j = 1; j <= n; ++j) {
                sumb += xpt(k, j) * xopt(j);
            }
            temp = suma * sumb;
            for (i = 1; i <= n; ++i) {
                w(i) += temp * xpt(k, i);
            }
        }
        for (i = 1; i <= n; ++i) {
            gopt(i) += diff * w(i);
        }

        // Update XOPT, GOPT and KOPT if the new calculated F is less than FOPT.
        if (f < fopt) {
            kopt = knew;
            xoptsq = 0;
            ih = 0;
            for (j = 1; j <= n; ++j) {
                xopt(j) = xnew(j);
                xoptsq += square(xopt(j));
                for (i = 1; i <= j; ++i) {
                    ++ih;
                    if (i < j) {
                        gopt(j) += hq(ih) * d(i);
                    }
                    gopt(i) += hq(ih) * d(j);
                }
            }
            for (k = 1; k <= npt; ++k) {
                temp = 0;
                for (j = 1; j <= n; ++j) {
                    temp += xpt(k, j) * d(j);
                }
                temp = pq(k) * temp;
                for (i = 1; i <= n; ++i) {
                    gopt(i) += temp * xpt(k, i);
                }
            }
            if (objectiveToleranceReached(f, fopt)) {
                rc = Stop::ObjectiveTolerance;
                goto L720;
            }
        }

        // Calculate the parameters of the least Frobenius norm interpolant to
        // the current data, the gradient of this interpolant at XOPT being put
        // into VLAG(NPT+I).
        if (ntrits > 0) {
            for (k = 1; k <= npt; ++k) {
                vlag(k) = fval(k) - fval(kopt);
                w(k) = 0;
            }
            for (j = 1; j <= nptm; ++j) {
                sum = 0;
                for (k = 1; k <= npt; ++k) {
                    sum += zmat(k, j) * vlag(k);
                }
                for (k = 1; k <= npt; ++k) {
                    w(k) += sum * zmat(k, j);
                }
            }
            for (k = 1; k <= npt; ++k) {
                sum = 0;
                for (j = 1; j <= n; ++j) {
                    sum += xpt(k, j) * xopt(j);
                }
                w(k + npt) = w(k);
                w(k) = sum * w(k);
            }
            gqsq = 0;
            gisq = 0;
            for (i = 1; i <= n; ++i) {
                sum = 0;
                for (k = 1; k <= npt; ++k) {
                    sum = sum + bmat(k, i) * vlag(k) + xpt(k, i) * w(k);
                }
                if (xopt(i) == sl(i)) {
                    gqsq += square(std::min(Scalar(0), gopt(i)));
                    gisq += square(std::min(Scalar(0), sum));
                } else if (xopt(i) == su(i)) {
                    gqsq += square(std::max(Scalar(0), gopt(i)));
                    gisq += square(std::max(Scalar(0), sum));
                } else {
                    gqsq += square(gopt(i));
                    gisq += sum * sum;
                }
                vlag(npt + i) = sum;
            }

            // Test whether to replace the new quadratic model by the least
            // Frobenius norm interpolant.
            ++itest;
            if (gqsq < Scalar(10) * gisq) {
                itest = 0;
            }
            if (itest >= 3) {
                for (i = 1; i <= std::max(npt, nh); ++i) {
                    if (i <= n) {
                        gopt(i) = vlag(npt + i);
                    }
                    if (i <= npt) {
                        pq(i) = w(npt + i);
                    }
                    if (i <= nh) {
                        hq(i) = 0;
                    }
                    itest = 0;
                }
            }
        }

        // If a trust region step has provided a sufficient decrease in F, then
        // branch for another trust region calculation.
        if (ntrits == 0) {
            goto L60;
        }
        if (f <= fopt + Scalar(0.1) * vquad) {
            goto L60;
        }

        // Alternatively, find out if the interpolation points are close enough
        // to the best point so far.
        distsq = std::max(square(Scalar(2) * delta), square(Scalar(10) * rho));
    L650:
        knew = 0;
        for (k = 1; k <= npt; ++k) {
            sum = 0;
            for (j = 1; j <= n; ++j) {
                sum += square(xpt(k, j) - xopt(j));
            }
            if (sum > distsq) {
                knew = k;
                distsq = sum;
            }
        }

        // If KNEW is positive, then ALTMOV finds alternative new positions for
        // the KNEW-th interpolation point within distance ADELT of XOPT.
        if (knew > 0) {
            dist = std::sqrt(distsq);
            if (ntrits == -1) {
                delta = std::min(Scalar(0.1) * delta, Scalar(0.5) * dist);
                if (delta <= rho * Scalar(1.5)) {
                    delta = rho;
                }
            }
            ntrits = 0;
            adelt = std::max(std::min(Scalar(0.1) * dist, delta), rho);
            dsq = adelt * adelt;
            goto L90;
        }
        if (ntrits == -1) {
            goto L680;
        }
        if (ratio > Scalar(0)) {
            goto L60;
        }
        if (std::max(delta, dnorm) > rho) {
            goto L60;
        }

        // The calculations with the current value of RHO are complete. Pick the
        // next values of RHO and DELTA.
    L680:
        if (rho > rhoend) {
            delta = Scalar(0.5) * rho;
            ratio = rho / rhoend;
            if (ratio <= Scalar(16)) {
                rho = rhoend;
            } else if (ratio <= Scalar(250)) {
                rho = std::sqrt(ratio) * rhoend;
            } else {
                rho = Scalar(0.1) * rho;
            }
            delta = std::max(delta, rho);
            ntrits = 0;
            nfsav = evaluations;
            goto L60;
        }

        // Return from the calculation, after another Newton-Raphson step, if it
        // is too short to have been tried before.
        if (ntrits == -1) {
            goto L360;
        }
        if (rc == Stop::None) {
            rc = Stop::RhoEnd;
        }
    L720:
        for (i = 1; i <= n; ++i) {
            x(i) = std::min(std::max(xl(i), xbase(i) + xopt(i)), xu(i));
            if (xopt(i) == sl(i)) {
                x(i) = xl(i);
            }
            if (xopt(i) == su(i)) {
                x(i) = xu(i);
            }
        }
        minf = fval(kopt);
        return rc;
    }
};

} // namespace bobyqa
} // namespace detail

/// BOBYQA: derivative-free local minimization by quadratic interpolation
/// within a trust region, honouring box constraints (Powell, 2009).
template <typename Scalar = double>
class BOBYQA : public Optimizer<Scalar> {
public:
    using typename Optimizer<Scalar>::ObjectiveFn;

    BOBYQA()
    {
        this->registerParam("max_function_evaluations", &m_maxFunctionEvaluations,
                            "maximum number of function evaluations (0 = max(1000, 200*n))");
        this->registerParam("interpolation_points", &m_interpolationPoints,
                            "number of interpolation conditions npt, in [n+2, (n+1)(n+2)/2] (0 = 2n+1)");
        this->registerParam("initial_trust_region_radius", &m_rhobeg,
                            "initial trust region radius, relative to the scale of each variable");
        this->registerParam("final_trust_region_radius", &m_rhoend,
                            "final trust region radius: the accuracy required in the variables");
        this->registerParam("rel_objective_change_tolerance", &m_relObjectiveTol,
                            "stop when the relative change of the objective falls below this (0 = disabled)");
        this->registerParam("target_objective", &m_targetObjective,
                            "value of the global optimum, if known (-inf to disable)");
        this->registerParam("tolerance", &m_tolerance,
                            "stop when the best objective is within this distance of target_objective");
    }

    const char* name() const override { return "BOBYQA"; }

protected:
    Result<Scalar> doOptimize(const ObjectiveFn& objective, const Vector<Scalar>& initialPoint) override
    {
        namespace bq = detail::bobyqa;

        const Index n = initialPoint.size();

        Result<Scalar> result;
        result.x = initialPoint;
        result.gradientNorm = std::numeric_limits<Scalar>::quiet_NaN();

        if (m_rhobeg <= Scalar(0) || m_rhoend <= Scalar(0) || m_rhoend > m_rhobeg) {
            result.status = Status::InvalidInput;
            result.message = "BOBYQA requires 0 < final_trust_region_radius <= initial_trust_region_radius";
            return result;
        }

        const Index npt = (m_interpolationPoints > 0)
            ? static_cast<Index>(m_interpolationPoints)
            : 2 * n + 1;
        if (npt < n + 2 || npt > (n + 1) * (n + 2) / 2) {
            result.status = Status::InvalidInput;
            result.message = "BOBYQA requires n+2 <= interpolation_points <= (n+1)(n+2)/2";
            return result;
        }

        // Bounds are optional; a missing bound is infinite.
        Vector<Scalar> lower = Vector<Scalar>::Constant(n, -detail::inf<Scalar>());
        Vector<Scalar> upper = Vector<Scalar>::Constant(n, detail::inf<Scalar>());
        if (this->m_boundsSet) {
            lower = this->m_lowerBounds;
            upper = this->m_upperBounds;
        }
        for (Index i = 0; i < n; ++i) {
            if (!(lower(i) < upper(i))) {
                result.status = Status::InvalidInput;
                result.message = "BOBYQA requires lower(i) < upper(i) for every variable";
                return result;
            }
        }

        bq::Solver<Scalar> solver(objective, n, npt);

        // Rescale the variables so that the initial trust region radius is the
        // same in every direction: a variable with finite bounds is measured in
        // units of the width of its box, an unbounded one in units of its own
        // magnitude. This is what makes an anisotropic box need no manual
        // rescaling by the caller.
        for (Index i = 0; i < n; ++i) {
            const Scalar width = upper(i) - lower(i);
            const Scalar s = std::isfinite(width)
                ? width
                : std::max(Scalar(1), std::abs(initialPoint(i)));
            solver.scale(i + 1) = s;
            solver.x(i + 1) = initialPoint(i) / s;
            solver.xl(i + 1) = lower(i) / s;
            solver.xu(i + 1) = upper(i) / s;
        }

        // The construction of the first quadratic model needs room for two
        // steps of length rhobeg between the bounds.
        for (Index i = 1; i <= n; ++i) {
            if (solver.xu(i) - solver.xl(i) < m_rhobeg + m_rhobeg) {
                result.status = Status::InvalidInput;
                result.message = "BOBYQA requires upper(i) - lower(i) >= "
                                 "2 * initial_trust_region_radius * scale(i) for every variable";
                return result;
            }
        }

        solver.rhobeg = m_rhobeg;
        solver.rhoend = m_rhoend;
        solver.relObjectiveTol = m_relObjectiveTol;
        solver.maxEvaluations = (m_maxFunctionEvaluations > 0)
            ? m_maxFunctionEvaluations
            : std::max<std::size_t>(1000, 200 * static_cast<std::size_t>(n));
        solver.minfMax = (m_targetObjective > -detail::inf<Scalar>())
            ? m_targetObjective + m_tolerance
            : -detail::inf<Scalar>();

        // Move the initial point away from a bound it is too close to, and set
        // the bounds on moves from it.
        for (Index i = 1; i <= n; ++i) {
            const Scalar width = solver.xu(i) - solver.xl(i);
            Scalar lo = solver.xl(i) - solver.x(i);
            Scalar up = solver.xu(i) - solver.x(i);
            if (lo >= -m_rhobeg) {
                if (lo >= Scalar(0)) {
                    solver.x(i) = solver.xl(i);
                    lo = 0;
                    up = width;
                } else {
                    solver.x(i) = solver.xl(i) + m_rhobeg;
                    lo = -m_rhobeg;
                    up = std::max(solver.xu(i) - solver.x(i), m_rhobeg);
                }
            } else if (up <= m_rhobeg) {
                if (up <= Scalar(0)) {
                    solver.x(i) = solver.xu(i);
                    lo = -width;
                    up = 0;
                } else {
                    solver.x(i) = solver.xu(i) - m_rhobeg;
                    lo = std::min(solver.xl(i) - solver.x(i), -m_rhobeg);
                    up = m_rhobeg;
                }
            }
            solver.setInitialBoundOffsets(i, lo, up);
        }

        const bq::Stop stop = solver.run();

        for (Index i = 0; i < n; ++i) {
            result.x(i) = solver.x(i + 1) * solver.scale(i + 1);
        }
        result.fval = solver.minf;
        result.functionEvaluations = solver.evaluations;
        result.iterations = solver.iterations;

        switch (stop) {
            case bq::Stop::RhoEnd:
                result.status = Status::Success;
                result.message = "optimization terminated successfully";
                break;
            case bq::Stop::TargetReached:
                result.status = Status::Success;
                result.message = "target objective reached";
                break;
            case bq::Stop::ObjectiveTolerance:
                result.status = Status::Success;
                result.message = "relative objective change below tolerance";
                break;
            case bq::Stop::MaxEvaluations:
                result.status = Status::MaxFunctionEvaluationsReached;
                result.message = toString(Status::MaxFunctionEvaluationsReached);
                break;
            case bq::Stop::RoundoffLimited:
                result.status = Status::Stalled;
                result.message = "progress limited by rounding errors in the interpolation model";
                break;
            case bq::Stop::None:
                result.status = Status::Stalled;
                result.message = toString(Status::Stalled);
                break;
        }

        return result;
    }

private:
    std::size_t m_maxFunctionEvaluations = 0;
    std::size_t m_interpolationPoints = 0;
    Scalar m_rhobeg = Scalar(0.1);
    Scalar m_rhoend = Scalar(1e-8);
    Scalar m_relObjectiveTol = 0;
    Scalar m_targetObjective = -detail::inf<Scalar>();
    Scalar m_tolerance = Scalar(1e-05);
};

} // namespace globopt

#endif
