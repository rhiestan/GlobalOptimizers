// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Moré and Thuente line search, ported from OptimLib
// (https://github.com/kthohr/optim), Copyright (C) 2016-2023 Keith O'Hara,
// Apache License 2.0; itself based on MINPACK and Dianne P. O'Leary's
// Matlab translation of MINPACK.

#ifndef GLOBOPT_CORE_LINE_SEARCH_HPP
#define GLOBOPT_CORE_LINE_SEARCH_HPP

#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace globopt {
namespace detail {

template <typename Scalar>
inline Scalar mtSupNorm(const Scalar a, const Scalar b, const Scalar c)
{
    return std::max(std::max(std::abs(a), std::abs(b)), std::abs(c));
}

// update the 'interval of uncertainty'
template <typename Scalar>
inline unsigned int mtStep(
    Scalar& stBest, Scalar& fBest, Scalar& dBest,
    Scalar& stOther, Scalar& fOther, Scalar& dOther,
    Scalar& step, Scalar& fStep, Scalar& dStep,
    bool& bracket, Scalar stepMin, Scalar stepMax)
{
    bool bound = false;
    unsigned int info = 0;
    const Scalar sgnd = dStep * (dBest / std::abs(dBest));

    Scalar theta, s, gamma, p, q, r, stepC, stepQ, stepF;

    if (fStep > fBest) {
        info = 1;
        bound = true;

        theta = 3 * (fBest - fStep) / (step - stBest) + dBest + dStep;
        s = mtSupNorm(theta, dBest, dStep);

        gamma = s * std::sqrt(std::pow(theta / s, Scalar(2)) - (dBest / s) * (dStep / s));
        if (step < stBest) {
            gamma = -gamma;
        }

        p = (gamma - dBest) + theta;
        q = ((gamma - dBest) + gamma) + dStep;
        r = p / q;

        stepC = stBest + r * (step - stBest);
        stepQ = stBest + ((dBest / ((fBest - fStep) / (step - stBest) + dBest)) / Scalar(2)) * (step - stBest);

        if (std::abs(stepC - stBest) < std::abs(stepQ - stBest)) {
            stepF = stepC;
        } else {
            stepF = stepC + (stepQ - stepC) / Scalar(2);
        }

        bracket = true;
    } else if (sgnd < Scalar(0)) {
        info = 2;
        bound = false;

        theta = 3 * (fBest - fStep) / (step - stBest) + dBest + dStep;
        s = mtSupNorm(theta, dBest, dStep);

        gamma = s * std::sqrt(std::pow(theta / s, Scalar(2)) - (dBest / s) * (dStep / s));
        if (step > stBest) {
            gamma = -gamma;
        }

        p = (gamma - dStep) + theta;
        q = ((gamma - dStep) + gamma) + dBest;
        r = p / q;

        stepC = step + r * (stBest - step);
        stepQ = step + (dStep / (dStep - dBest)) * (stBest - step);

        if (std::abs(stepC - step) > std::abs(stepQ - step)) {
            stepF = stepC;
        } else {
            stepF = stepQ;
        }

        bracket = true;
    } else if (std::abs(dStep) < std::abs(dBest)) {
        info = 3;
        bound = true;

        theta = 3 * (fBest - fStep) / (step - stBest) + dBest + dStep;
        s = mtSupNorm(theta, dBest, dStep);

        gamma = s * std::sqrt(std::max(Scalar(0), Scalar(std::pow(theta / s, Scalar(2)) - (dBest / s) * (dStep / s))));
        if (step > stBest) {
            gamma = -gamma;
        }

        p = (gamma - dStep) + theta;
        q = (gamma + (dBest - dStep)) + gamma;
        r = p / q;

        if (r < Scalar(0) && gamma != Scalar(0)) {
            stepC = step + r * (stBest - step);
        } else if (step > stBest) {
            stepC = stepMax;
        } else {
            stepC = stepMin;
        }

        stepQ = step + (dStep / (dStep - dBest)) * (stBest - step);

        if (bracket) {
            if (std::abs(step - stepC) < std::abs(step - stepQ)) {
                stepF = stepC;
            } else {
                stepF = stepQ;
            }
        } else {
            if (std::abs(step - stepC) > std::abs(step - stepQ)) {
                stepF = stepC;
            } else {
                stepF = stepQ;
            }
        }
    } else {
        info = 4;
        bound = false;

        if (bracket) {
            theta = 3 * (fStep - fOther) / (stOther - step) + dOther + dStep;
            s = mtSupNorm(theta, dOther, dStep);

            gamma = s * std::sqrt(std::pow(theta / s, Scalar(2)) - (dOther / s) * (dStep / s));
            if (step > stOther) {
                gamma = -gamma;
            }

            p = (gamma - dStep) + theta;
            q = ((gamma - dStep) + gamma) + dOther;
            r = p / q;

            stepC = step + r * (stOther - step);
            stepF = stepC;
        } else if (step > stBest) {
            stepF = stepMax;
        } else {
            stepF = stepMin;
        }
    }

    // update the interval of uncertainty

    if (fStep > fBest) {
        stOther = step;
        fOther = fStep;
        dOther = dStep;
    } else {
        if (sgnd < Scalar(0)) {
            stOther = stBest;
            fOther = fBest;
            dOther = dBest;
        }

        stBest = step;
        fBest = fStep;
        dBest = dStep;
    }

    // compute the new step and safeguard it

    stepF = std::max(stepMin, std::min(stepMax, stepF));
    step = stepF;

    if (bracket && bound) {
        if (stOther > stBest) {
            step = std::min(stBest + Scalar(0.66) * (stOther - stBest), step);
        } else {
            step = std::max(stBest + Scalar(0.66) * (stOther - stBest), step);
        }
    }

    return info;
}

/// Moré-Thuente line search satisfying the strong Wolfe conditions.
/// On entry x is the current point and direc the search direction; on exit
/// x and grad hold the accepted point and its gradient.
template <typename Scalar, typename ObjFn>
inline Scalar lineSearchMoreThuente(
    Scalar step,
    Vector<Scalar>& x,
    Vector<Scalar>& grad,
    const Vector<Scalar>& direc,
    const Scalar wolfeCons1,   // Armijo sufficient-decrease tolerance ('mu')
    const Scalar wolfeCons2,   // curvature-condition tolerance ('eta')
    ObjFn&& objFn)
{
    const std::size_t iterMax = 100;

    const Scalar stepMin = Scalar(0);
    const Scalar stepMax = Scalar(10);
    const Scalar xtol = Scalar(1e-04);

    unsigned int info = 0, infoc = 1;
    const Scalar extrapDelta = 4; // 'delta' on page 20 of Moré-Thuente (1994)

    const Vector<Scalar> x0 = x;

    Scalar fStep = objFn(x, &grad);

    const Scalar dgradInit = grad.dot(direc);

    if (dgradInit >= Scalar(0)) {
        return step;
    }

    Scalar dgrad = dgradInit;

    std::size_t iter = 0;

    bool bracket = false, stage1 = true;

    const Scalar fInit = fStep;
    const Scalar dgradTest = wolfeCons1 * dgradInit;
    Scalar width = stepMax - stepMin, widthOld = 2 * width;

    Scalar stBest = Scalar(0), fBest = fInit, dgradBest = dgradInit;
    Scalar stOther = Scalar(0), fOther = fInit, dgradOther = dgradInit;

    while (true) {
        ++iter;

        Scalar stMin, stMax;

        if (bracket) {
            stMin = std::min(stBest, stOther);
            stMax = std::max(stBest, stOther);
        } else {
            stMin = stBest;
            stMax = step + extrapDelta * (step - stBest);
        }

        step = std::min(std::max(step, stepMin), stepMax);

        if ((bracket && (step <= stMin || step >= stMax))
                || iter >= iterMax - 1 || infoc == 0 || (bracket && stMax - stMin <= xtol * stMax)) {
            step = stBest;
        }

        x = x0 + step * direc;
        fStep = objFn(x, &grad);

        dgrad = grad.dot(direc);
        const Scalar armijoCheckVal = fInit + step * dgradTest;

        // check stop conditions

        if ((bracket && (step <= stMin || step >= stMax)) || infoc == 0) {
            info = 6;
        }
        if (step == stepMax && fStep <= armijoCheckVal && dgrad <= dgradTest) {
            info = 5;
        }
        if (step == stepMin && (fStep > armijoCheckVal || dgrad >= dgradTest)) {
            info = 4;
        }
        if (iter >= iterMax) {
            info = 3;
        }
        if (bracket && stMax - stMin <= xtol * stMax) {
            info = 2;
        }

        if (fStep <= armijoCheckVal && std::abs(dgrad) <= wolfeCons2 * (-dgradInit)) {
            // strong Wolfe conditions
            info = 1;
        }

        if (info != 0) {
            return step;
        }

        if (stage1 && fStep <= armijoCheckVal && dgrad >= std::min(wolfeCons1, wolfeCons2) * dgradInit) {
            stage1 = false;
        }

        if (stage1 && fStep <= fBest && fStep > armijoCheckVal) {
            Scalar fMod = fStep - step * dgradTest;
            Scalar fBestMod = fBest - stBest * dgradTest;
            Scalar fOtherMod = fOther - stOther * dgradTest;

            Scalar dgradMod = dgrad - dgradTest;
            Scalar dgradBestMod = dgradBest - dgradTest;
            Scalar dgradOtherMod = dgradOther - dgradTest;

            infoc = mtStep(stBest, fBestMod, dgradBestMod, stOther, fOtherMod, dgradOtherMod,
                           step, fMod, dgradMod, bracket, stMin, stMax);

            fBest = fBestMod + stBest * dgradTest;
            fOther = fOtherMod + stOther * dgradTest;

            dgradBest = dgradBestMod + dgradTest;
            dgradOther = dgradOtherMod + dgradTest;
        } else {
            infoc = mtStep(stBest, fBest, dgradBest, stOther, fOther, dgradOther,
                           step, fStep, dgrad, bracket, stMin, stMax);
        }

        if (bracket) {
            if (std::abs(stOther - stBest) >= Scalar(0.66) * widthOld) {
                step = stBest + Scalar(0.5) * (stOther - stBest);
            }

            widthOld = width;
            width = std::abs(stOther - stBest);
        }
    }
}

} // namespace detail
} // namespace globopt

#endif
