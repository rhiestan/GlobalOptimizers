// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// EGO (Efficient Global Optimization / Bayesian optimization) with an
// internal Kriging (Gaussian process) surrogate, ported from SMT
// (Surrogate Modeling Toolbox, https://github.com/SMTorg/smt):
// smt/applications/ego.py and the Kriging core in
// smt/surrogate_models/krg_based (New BSD License, Copyright SMT
// developers: N. Bartoli, R. Priem, R. Lafage, E. Roux, M. Bouhlel et al.).
//
// References:
//   D.R. Jones, M. Schonlau, W.J. Welch, "Efficient Global Optimization of
//   Expensive Black-Box Functions", Journal of Global Optimization 13, 1998.
//   M.A. Bouhlel et al., "A Python surrogate modeling framework with
//   derivatives", Advances in Engineering Software, 2019.
//
// Deviations from the Python original (documented, intentional):
//   - Hyperparameters and the acquisition function are optimized with this
//     library's L-BFGS-B (central-difference gradients) instead of scipy's
//     TNC / SLSQP.
//   - The initial design of experiments uses a plain (randomized) Latin
//     hypercube instead of SMT's ESE-optimized LHS, and the (clamped)
//     initial point passed to run() is included as the first DOE sample.
//   - The previous iteration's optimal theta is added to the hyperparameter
//     multistart set (warm start).
//   - Continuous variables only; the qEI parallel enrichment, EI tunneling,
//     noise estimation (eval_noise) and reinterpolation paths are not ported.
//     A fixed noise term on the correlation diagonal is available instead.

#ifndef GLOBOPT_GLOBAL_EGO_HPP
#define GLOBOPT_GLOBAL_EGO_HPP

#include "../core/numerical_gradient.hpp"
#include "../core/optimizer.hpp"
#include "../local/lbfgsb.hpp"

#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <Eigen/SVD>

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
namespace detail {
namespace ego {

enum class Kernel { SquarExp, AbsExp, Matern32, Matern52 };
enum class RegressionType { Constant, Linear, Quadratic };
enum class Criterion { EI, SBO, LCB };

inline bool parseKernel(const std::string& name, Kernel& out)
{
    if (name == "squar_exp") { out = Kernel::SquarExp; return true; }
    if (name == "abs_exp") { out = Kernel::AbsExp; return true; }
    if (name == "matern32") { out = Kernel::Matern32; return true; }
    if (name == "matern52") { out = Kernel::Matern52; return true; }
    return false;
}

inline bool parseRegression(const std::string& name, RegressionType& out)
{
    if (name == "constant") { out = RegressionType::Constant; return true; }
    if (name == "linear") { out = RegressionType::Linear; return true; }
    if (name == "quadratic") { out = RegressionType::Quadratic; return true; }
    return false;
}

inline bool parseCriterion(const std::string& name, Criterion& out)
{
    if (name == "EI") { out = Criterion::EI; return true; }
    if (name == "SBO") { out = Criterion::SBO; return true; }
    if (name == "LCB") { out = Criterion::LCB; return true; }
    return false;
}

/// Randomized Latin hypercube: numPoints samples in (0,1)^dims, one point in
/// each of the numPoints strata per dimension.
template <typename Scalar>
inline Matrix<Scalar> latinHypercube(std::mt19937_64& rng, const Index numPoints, const Index dims)
{
    Matrix<Scalar> samples(numPoints, dims);
    std::uniform_real_distribution<Scalar> unit(Scalar(0), Scalar(1));
    std::vector<Index> perm(static_cast<std::size_t>(numPoints));

    for (Index k = 0; k < dims; ++k) {
        std::iota(perm.begin(), perm.end(), Index(0));
        std::shuffle(perm.begin(), perm.end(), rng);
        for (Index i = 0; i < numPoints; ++i) {
            samples(i, k) = (Scalar(perm[static_cast<std::size_t>(i)]) + unit(rng)) / Scalar(numPoints);
        }
    }
    return samples;
}

inline Index regressionSize(const RegressionType type, const Index dims)
{
    switch (type) {
        case RegressionType::Constant: return 1;
        case RegressionType::Linear: return 1 + dims;
        case RegressionType::Quadratic: return 1 + dims + dims * (dims + 1) / 2;
    }
    return 1;
}

/// Polynomial regression basis f(x) (SMT's poly = constant / linear / quadratic).
template <typename Scalar>
inline Vector<Scalar> regressionRow(const RegressionType type, const Vector<Scalar>& x)
{
    const Index dims = x.size();
    Vector<Scalar> f(regressionSize(type, dims));
    f(0) = Scalar(1);
    if (type == RegressionType::Linear || type == RegressionType::Quadratic) {
        f.segment(1, dims) = x;
    }
    if (type == RegressionType::Quadratic) {
        Index idx = 1 + dims;
        for (Index i = 0; i < dims; ++i) {
            for (Index j = i; j < dims; ++j) {
                f(idx++) = x(i) * x(j);
            }
        }
    }
    return f;
}

/// Correlations for a batch of point pairs. absDx holds one componentwise
/// absolute distance |x - x'| (in normalized coordinates) per column; the
/// result has one correlation value per column.
template <typename Scalar, typename Derived>
inline Vector<Scalar> correlationVector(const Kernel kernel, const Vector<Scalar>& theta,
                                        const Eigen::MatrixBase<Derived>& absDx)
{
    using ArrayXX = Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    switch (kernel) {
        case Kernel::SquarExp: {
            const Vector<Scalar> s = absDx.cwiseProduct(absDx).transpose() * theta;
            return (-s.array()).exp();
        }
        case Kernel::AbsExp: {
            const Vector<Scalar> s = absDx.transpose() * theta;
            return (-s.array()).exp();
        }
        case Kernel::Matern32: {
            const ArrayXX ll =
                std::sqrt(Scalar(3)) * (absDx.array().colwise() * theta.array());
            return ((Scalar(1) + ll).colwise().prod()
                    * (-ll.colwise().sum()).exp()).matrix().transpose();
        }
        case Kernel::Matern52: {
            const ArrayXX ll =
                std::sqrt(Scalar(5)) * (absDx.array().colwise() * theta.array());
            return ((Scalar(1) + ll + ll.square() / Scalar(3)).colwise().prod()
                    * (-ll.colwise().sum()).exp()).matrix().transpose();
        }
    }
    return Vector<Scalar>::Zero(absDx.cols());
}

// ---------------------------------------------------------------------------
// Kriging model (SMT's KRG with p = q = 0): standardization, correlation
// matrix, reduced likelihood via Cholesky + QR (likelihood_eval.py), and
// multistart hyperparameter optimization in log10(theta) space
// (krg_based._optimize_hyperparam).
// ---------------------------------------------------------------------------

template <typename Scalar>
class KrigingModel {
public:
    struct Options {
        Kernel kernel = Kernel::SquarExp;
        RegressionType regression = RegressionType::Constant;
        Scalar theta0 = Scalar(0.01);
        Scalar thetaLower = Scalar(1e-6);
        Scalar thetaUpper = Scalar(20);
        Scalar nugget = Scalar(100) * detail::eps<Scalar>();
        Scalar noise = Scalar(0);
        std::size_t multistarts = 10;
    };

    explicit KrigingModel(const Options& options) : m_options(options) {}

    const Vector<Scalar>& theta() const { return m_theta; }

    /// Train on the given samples. Returns false if the model cannot be fit
    /// (too few points or no feasible hyperparameters found).
    bool fit(const std::vector<Vector<Scalar>>& xs, const std::vector<Scalar>& ys, std::mt19937_64& rng)
    {
        const Index nt = static_cast<Index>(xs.size());
        if (nt < 2) {
            return false;
        }
        const Index dims = xs[0].size();
        const Index p = regressionSize(m_options.regression, dims);
        if (nt <= p) {
            return false;
        }

        // -- standardization (SMT utils standardization: population std) ----
        Matrix<Scalar> X(nt, dims);
        Vector<Scalar> y(nt);
        for (Index i = 0; i < nt; ++i) {
            X.row(i) = xs[static_cast<std::size_t>(i)].transpose();
            y(i) = ys[static_cast<std::size_t>(i)];
        }

        m_xOffset = X.colwise().mean();
        Vector<Scalar> xVar = (X.rowwise() - m_xOffset.transpose()).array().square().colwise().mean();
        m_xScale = xVar.array().sqrt();
        for (Index k = 0; k < dims; ++k) {
            if (m_xScale(k) <= Scalar(0)) {
                m_xScale(k) = Scalar(1);
            }
        }
        m_yMean = y.mean();
        m_yStd = std::sqrt((y.array() - m_yMean).square().mean());
        if (m_yStd <= Scalar(0)) {
            m_yStd = Scalar(1);
        }

        m_X = (X.rowwise() - m_xOffset.transpose()).array().rowwise() / m_xScale.transpose().array();
        m_y = (y.array() - m_yMean) / m_yStd;

        // -- componentwise cross distances |x_i - x_j| (i < j) ---------------
        const Index numPairs = nt * (nt - 1) / 2;
        m_pairI.resize(static_cast<std::size_t>(numPairs));
        m_pairJ.resize(static_cast<std::size_t>(numPairs));
        m_absDx.resize(dims, numPairs);
        {
            Index k = 0;
            for (Index i = 0; i < nt; ++i) {
                for (Index j = i + 1; j < nt; ++j, ++k) {
                    m_pairI[static_cast<std::size_t>(k)] = i;
                    m_pairJ[static_cast<std::size_t>(k)] = j;
                    m_absDx.col(k) = (m_X.row(i) - m_X.row(j)).cwiseAbs().transpose();
                }
            }
        }

        // -- regression matrix F ---------------------------------------------
        m_F.resize(nt, p);
        for (Index i = 0; i < nt; ++i) {
            m_F.row(i) = regressionRow<Scalar>(m_options.regression, m_X.row(i).transpose()).transpose();
        }

        // -- multistart likelihood maximization in log10(theta) --------------
        const Scalar lo = std::log10(m_options.thetaLower);
        const Scalar hi = std::log10(m_options.thetaUpper);

        std::vector<Vector<Scalar>> starts;
        const Scalar t0 = std::min(std::max(std::log10(m_options.theta0), lo), hi);
        starts.push_back(Vector<Scalar>::Constant(dims, t0));
        if (m_theta.size() == dims) { // warm start from the previous fit
            starts.push_back(m_theta.array().log10().max(lo).min(hi).matrix());
        }
        const Matrix<Scalar> lhs = latinHypercube<Scalar>(
            rng, static_cast<Index>(m_options.multistarts), dims);
        for (Index s = 0; s < lhs.rows(); ++s) {
            starts.push_back((lo + (hi - lo) * lhs.row(s).array()).matrix().transpose());
        }

        auto negLikelihood = withNumericalGradient<Scalar>([this](const Vector<Scalar>& log10t) -> Scalar {
            const Vector<Scalar> theta = Eigen::pow(Scalar(10), log10t.array());
            const Scalar rlf = reducedLikelihood(theta, nullptr);
            return std::isfinite(rlf) ? -rlf : Scalar(1e10);
        });

        // SMT limits the likelihood optimizer to max(12*dims, 50) objective
        // evaluations (TNC maxfun); each of ours costs 2*dims+1 likelihood
        // computations for the central-difference gradient.
        LBFGSB<Scalar> solver;
        solver.setBounds(Vector<Scalar>::Constant(dims, lo), Vector<Scalar>::Constant(dims, hi));
        solver.setParam("max_function_evaluations",
                        static_cast<long long>(std::max<Index>(12 * dims, 50)));

        Scalar bestRlf = -detail::inf<Scalar>();
        Vector<Scalar> bestTheta;
        auto consider = [&](const Vector<Scalar>& log10t) {
            const Vector<Scalar> theta =
                Eigen::pow(Scalar(10), log10t.array().max(lo).min(hi));
            const Scalar rlf = reducedLikelihood(theta, nullptr);
            if (rlf > bestRlf) {
                bestRlf = rlf;
                bestTheta = theta;
            }
        };

        for (const auto& start : starts) {
            consider(start);
            const auto res = solver.run(negLikelihood, start);
            if (res.x.size() == dims && res.x.allFinite()) {
                consider(res.x);
            }
        }

        if (!std::isfinite(bestRlf)) {
            return false;
        }

        m_theta = bestTheta;
        return std::isfinite(reducedLikelihood(m_theta, &m_pars));
    }

    /// Posterior mean and variance at x (original, un-normalized units).
    void predict(const Vector<Scalar>& x, Scalar* mean, Scalar* variance) const
    {
        const Index nt = m_X.rows();
        const Vector<Scalar> xn = (x - m_xOffset).cwiseQuotient(m_xScale);

        const Matrix<Scalar> absDx =
            (m_X.rowwise() - xn.transpose()).cwiseAbs().transpose();
        const Vector<Scalar> r = correlationVector(m_options.kernel, m_theta, absDx);
        const Vector<Scalar> f = regressionRow<Scalar>(m_options.regression, xn);

        if (mean) {
            *mean = m_yMean + m_yStd * (f.dot(m_pars.beta) + r.dot(m_pars.gamma));
        }
        if (variance) {
            const Vector<Scalar> rt = m_pars.L.template triangularView<Eigen::Lower>().solve(r);
            const Vector<Scalar> u = m_pars.G.transpose().template triangularView<Eigen::Lower>()
                                         .solve(m_pars.Ft.transpose() * rt - f);
            const Scalar B = Scalar(1) - rt.squaredNorm() + u.squaredNorm();
            *variance = std::max(Scalar(0), m_pars.sigma2 * B) * m_yStd * m_yStd;
        }
    }

private:
    struct Pars {
        Matrix<Scalar> L;  // Cholesky factor of R
        Matrix<Scalar> Ft; // C^-1 F
        Matrix<Scalar> G;  // R factor of the QR of Ft
        Vector<Scalar> beta;
        Vector<Scalar> gamma;
        Scalar sigma2 = Scalar(0); // GP variance in normalized-y units
    };

    /// SMT's reduced log-likelihood (likelihood_eval.py, standard GLS path,
    /// p = q = 0). Returns -inf when theta is infeasible.
    Scalar reducedLikelihood(const Vector<Scalar>& theta, Pars* parsOut) const
    {
        const Index nt = m_X.rows();
        const Index p = m_F.cols();
        const Scalar minusInf = -detail::inf<Scalar>();

        Matrix<Scalar> R = Matrix<Scalar>::Identity(nt, nt)
            * (Scalar(1) + m_options.nugget + m_options.noise);
        const Index numPairs = m_absDx.cols();
        const Vector<Scalar> corr = correlationVector(m_options.kernel, theta, m_absDx);
        for (Index k = 0; k < numPairs; ++k) {
            R(m_pairI[static_cast<std::size_t>(k)], m_pairJ[static_cast<std::size_t>(k)]) = corr(k);
            R(m_pairJ[static_cast<std::size_t>(k)], m_pairI[static_cast<std::size_t>(k)]) = corr(k);
        }

        const Eigen::LLT<Matrix<Scalar>> llt(R);
        if (llt.info() != Eigen::Success) {
            return minusInf;
        }
        const Matrix<Scalar> L = llt.matrixL();

        const Matrix<Scalar> Ft = L.template triangularView<Eigen::Lower>().solve(m_F);
        const Eigen::HouseholderQR<Matrix<Scalar>> qr(Ft);
        const Matrix<Scalar> Q = qr.householderQ() * Matrix<Scalar>::Identity(nt, p);
        const Matrix<Scalar> G = qr.matrixQR().topRows(p).template triangularView<Eigen::Upper>();

        const Eigen::JacobiSVD<Matrix<Scalar>> svd(G);
        const auto& sv = svd.singularValues();
        if (sv(0) <= Scalar(0) || sv(sv.size() - 1) / sv(0) < Scalar(1e-10)) {
            return minusInf; // ill-conditioned regression at this theta
        }

        const Vector<Scalar> Yt = L.template triangularView<Eigen::Lower>().solve(m_y);
        const Vector<Scalar> beta =
            G.template triangularView<Eigen::Upper>().solve(Q.transpose() * Yt);
        const Vector<Scalar> rho = Yt - Ft * beta;

        Scalar sigma2 = rho.squaredNorm() / Scalar(nt);
        sigma2 = std::max(sigma2, std::numeric_limits<Scalar>::min());

        // -nt*log10(sigma2) - nt*log10(detR), with
        // log10(detR) = (2/nt) * sum(log10(diag(L)))
        Scalar rlf = -Scalar(nt) * std::log10(sigma2)
            - Scalar(2) * L.diagonal().array().log10().sum();
        rlf = std::min(rlf, Scalar(1e15));

        if (parsOut) {
            parsOut->L = L;
            parsOut->Ft = Ft;
            parsOut->G = G;
            parsOut->beta = beta;
            parsOut->gamma = L.transpose().template triangularView<Eigen::Upper>().solve(rho);
            parsOut->sigma2 = sigma2;
        }
        return rlf;
    }

    Options m_options;

    Matrix<Scalar> m_X; // normalized training inputs (nt x dims)
    Vector<Scalar> m_y; // normalized training outputs
    Vector<Scalar> m_xOffset, m_xScale;
    Scalar m_yMean = Scalar(0), m_yStd = Scalar(1);

    Matrix<Scalar> m_absDx; // componentwise |dx| per pair (dims x numPairs)
    std::vector<Index> m_pairI, m_pairJ;
    Matrix<Scalar> m_F;

    Vector<Scalar> m_theta;
    Pars m_pars;
};

/// Expected improvement of x over the current best value fmin (ego.py EI).
template <typename Scalar>
inline Scalar expectedImprovement(const KrigingModel<Scalar>& model, const Scalar fmin,
                                  const Vector<Scalar>& x)
{
    Scalar mean = 0, variance = 0;
    model.predict(x, &mean, &variance);
    const Scalar sig = std::sqrt(std::max(variance, Scalar(0)));
    if (sig <= Scalar(1e-12)) {
        return Scalar(0);
    }
    const Scalar z = (fmin - mean) / sig;
    const Scalar cdf = Scalar(0.5) * std::erfc(-z / std::sqrt(Scalar(2)));
    const Scalar pdf = std::exp(-Scalar(0.5) * z * z)
        / std::sqrt(Scalar(2) * Scalar(3.14159265358979323846));
    return (fmin - mean) * cdf + sig * pdf;
}

} // namespace ego
} // namespace detail

// ---------------------------------------------------------------------------
// EGO optimizer
// ---------------------------------------------------------------------------

/// Efficient Global Optimization (Bayesian optimization with a Kriging
/// surrogate). Derivative-free; requires finite box bounds via setBounds().
/// The initial point passed to run() is clamped into the box and used as the
/// first DOE sample. If "target_objective" is set, the search stops early
/// with Status::Success once reached within "tolerance"; otherwise it spends
/// the whole evaluation budget and returns the best point found
/// (Status::MaxFunctionEvaluationsReached).
template <typename Scalar>
class EGO : public Optimizer<Scalar> {
public:
    using typename Optimizer<Scalar>::ObjectiveFn;

    EGO()
    {
        this->registerParam("max_function_evaluations", &m_maxFunctionEvaluations,
                            "maximum number of function evaluations, DOE included (0 = max(100, 10*n))");
        this->registerParam("doe_size", &m_doeSize,
                            "number of initial design-of-experiments samples (0 = max(5, 2*n))");
        this->registerParam("criterion", &m_criterion,
                            "infill criterion: EI (expected improvement), SBO (surrogate mean) or LCB (mean - 3*sigma)");
        this->registerParam("kernel", &m_kernel,
                            "correlation kernel: squar_exp, abs_exp, matern32 or matern52");
        this->registerParam("regression", &m_regression,
                            "regression (trend) model: constant, linear or quadratic");
        this->registerParam("theta0", &m_theta0,
                            "initial correlation hyperparameter");
        this->registerParam("theta_bound_lower", &m_thetaBoundLower,
                            "lower bound for the correlation hyperparameters");
        this->registerParam("theta_bound_upper", &m_thetaBoundUpper,
                            "upper bound for the correlation hyperparameters");
        this->registerParam("nugget", &m_nugget,
                            "jitter added to the correlation diagonal for numerical stability");
        this->registerParam("noise", &m_noise,
                            "fixed noise variance added to the correlation diagonal (normalized-y units)");
        this->registerParam("hyperparameter_starts", &m_hyperparameterStarts,
                            "number of multistart points for the likelihood maximization");
        this->registerParam("acquisition_starts", &m_acquisitionStarts,
                            "number of multistart points for the acquisition maximization");
        this->registerParam("target_objective", &m_targetObjective,
                            "value of the global optimum, if known (-inf to disable)");
        this->registerParam("tolerance", &m_tolerance,
                            "stop when the best objective is within this distance of target_objective");
        this->registerParam("seed", &m_seed,
                            "random seed (0 = non-deterministic)");
    }

    const char* name() const override { return "EGO"; }

protected:
    Result<Scalar> doOptimize(const ObjectiveFn& objective, const Vector<Scalar>& initialPoint) override
    {
        namespace eg = detail::ego;

        Result<Scalar> result;
        result.x = initialPoint;
        result.gradientNorm = std::numeric_limits<Scalar>::quiet_NaN();

        if (!this->m_boundsSet
            || !this->m_lowerBounds.allFinite() || !this->m_upperBounds.allFinite()) {
            result.status = Status::InvalidInput;
            result.message = "EGO requires finite lower and upper bounds (setBounds)";
            return result;
        }
        for (Index i = 0; i < this->m_lowerBounds.size(); ++i) {
            if (this->m_lowerBounds(i) >= this->m_upperBounds(i)) {
                result.status = Status::InvalidInput;
                result.message = "EGO requires lower(i) < upper(i) for every variable";
                return result;
            }
        }

        eg::Criterion criterion;
        eg::Kernel kernel;
        eg::RegressionType regression;
        if (!eg::parseCriterion(m_criterion, criterion)) {
            result.status = Status::InvalidInput;
            result.message = "EGO: unknown criterion '" + m_criterion + "' (use EI, SBO or LCB)";
            return result;
        }
        if (!eg::parseKernel(m_kernel, kernel)) {
            result.status = Status::InvalidInput;
            result.message = "EGO: unknown kernel '" + m_kernel
                + "' (use squar_exp, abs_exp, matern32 or matern52)";
            return result;
        }
        if (!eg::parseRegression(m_regression, regression)) {
            result.status = Status::InvalidInput;
            result.message = "EGO: unknown regression '" + m_regression
                + "' (use constant, linear or quadratic)";
            return result;
        }
        if (!(m_thetaBoundLower > Scalar(0)) || !(m_thetaBoundUpper > m_thetaBoundLower)) {
            result.status = Status::InvalidInput;
            result.message = "EGO requires 0 < theta_bound_lower < theta_bound_upper";
            return result;
        }

        const Vector<Scalar>& lower = this->m_lowerBounds;
        const Vector<Scalar>& upper = this->m_upperBounds;
        const Index dims = initialPoint.size();

        const std::size_t budget = (m_maxFunctionEvaluations > 0)
            ? m_maxFunctionEvaluations
            : std::max<std::size_t>(100, 10 * static_cast<std::size_t>(dims));
        std::size_t doeSize = (m_doeSize > 0)
            ? m_doeSize
            : std::max<std::size_t>(5, 2 * static_cast<std::size_t>(dims));
        // the GP needs strictly more points than regression coefficients
        doeSize = std::max(doeSize,
                           static_cast<std::size_t>(eg::regressionSize(regression, dims)) + 1);
        doeSize = std::min(doeSize, budget);

        std::mt19937_64 rng(m_seed != 0 ? static_cast<std::uint64_t>(m_seed) : std::random_device{}());

        std::vector<Vector<Scalar>> xs;
        std::vector<Scalar> ys;
        Scalar bestY = detail::inf<Scalar>();
        Vector<Scalar> bestX = initialPoint.cwiseMax(lower).cwiseMin(upper);
        std::size_t& evaluations = result.functionEvaluations;
        std::size_t infill = 0;

        auto evaluate = [&](const Vector<Scalar>& x) {
            const Scalar y = objective(x, nullptr);
            ++evaluations;
            xs.push_back(x);
            ys.push_back(y);
            if (y < bestY) {
                bestY = y;
                bestX = x;
            }
        };

        auto reachedTarget = [&]() { return bestY < m_targetObjective + m_tolerance; };

        auto finish = [&](const Status status, const std::string& reason) {
            result.x = bestX;
            result.fval = bestY;
            result.iterations = infill;
            result.status = status;
            result.message = reason + " (" + std::to_string(doeSize) + " DOE + "
                + std::to_string(infill) + " infill evaluations)";
            return result;
        };

        auto randomPoint = [&]() {
            Vector<Scalar> x(dims);
            for (Index i = 0; i < dims; ++i) {
                std::uniform_real_distribution<Scalar> u(lower(i), upper(i));
                x(i) = u(rng);
            }
            return x;
        };

        // -- initial design of experiments -----------------------------------
        evaluate(bestX);
        if (doeSize > 1) {
            const Matrix<Scalar> lhs =
                eg::latinHypercube<Scalar>(rng, static_cast<Index>(doeSize) - 1, dims);
            for (Index s = 0; s < lhs.rows(); ++s) {
                if (reachedTarget()) {
                    return finish(Status::Success, "optimization terminated successfully");
                }
                const Vector<Scalar> x = lower.array()
                    + (upper - lower).array() * lhs.row(s).transpose().array();
                evaluate(x);
            }
        }

        // -- EGO loop: fit surrogate, maximize acquisition, evaluate ---------
        typename eg::KrigingModel<Scalar>::Options modelOptions;
        modelOptions.kernel = kernel;
        modelOptions.regression = regression;
        modelOptions.theta0 = m_theta0;
        modelOptions.thetaLower = m_thetaBoundLower;
        modelOptions.thetaUpper = m_thetaBoundUpper;
        modelOptions.nugget = m_nugget;
        modelOptions.noise = m_noise;
        modelOptions.multistarts = m_hyperparameterStarts;
        eg::KrigingModel<Scalar> model(modelOptions);

        while (!reachedTarget() && evaluations < budget) {
            Vector<Scalar> xNext;
            if (model.fit(xs, ys, rng)) {
                xNext = nextInfillPoint(model, criterion, bestY, rng);
            }
            if (xNext.size() != dims || !xNext.allFinite() || tooClose(xNext, xs)) {
                xNext = randomPoint();
            }
            evaluate(xNext);
            ++infill;
        }

        if (reachedTarget()) {
            return finish(Status::Success, "optimization terminated successfully");
        }
        return finish(Status::MaxFunctionEvaluationsReached,
                      toString(Status::MaxFunctionEvaluationsReached));
    }

private:
    /// ego.py _find_best_point: multistart bounded minimization of the
    /// acquisition (-EI, surrogate mean, or LCB).
    Vector<Scalar> nextInfillPoint(const detail::ego::KrigingModel<Scalar>& model,
                                   const detail::ego::Criterion criterion,
                                   const Scalar fmin, std::mt19937_64& rng) const
    {
        namespace eg = detail::ego;
        const Vector<Scalar>& lower = this->m_lowerBounds;
        const Vector<Scalar>& upper = this->m_upperBounds;
        const Index dims = lower.size();

        auto acquisition = [&](const Vector<Scalar>& x) -> Scalar {
            switch (criterion) {
                case eg::Criterion::EI:
                    return -eg::expectedImprovement(model, fmin, x);
                case eg::Criterion::SBO: {
                    Scalar mean = 0;
                    model.predict(x, &mean, nullptr);
                    return mean;
                }
                case eg::Criterion::LCB: {
                    Scalar mean = 0, variance = 0;
                    model.predict(x, &mean, &variance);
                    return mean - Scalar(3) * std::sqrt(std::max(variance, Scalar(0)));
                }
            }
            return Scalar(0);
        };
        auto acquisitionFn = withNumericalGradient<Scalar>(acquisition);

        LBFGSB<Scalar> solver;
        solver.setBounds(lower, upper);
        solver.setParam("max_iterations", 100LL);
        solver.setParam("max_function_evaluations", 300LL);

        const Matrix<Scalar> lhs = eg::latinHypercube<Scalar>(
            rng, static_cast<Index>(m_acquisitionStarts), dims);

        Scalar bestValue = detail::inf<Scalar>();
        Vector<Scalar> bestX;
        auto consider = [&](const Vector<Scalar>& x) {
            const Vector<Scalar> xc = x.cwiseMax(lower).cwiseMin(upper);
            const Scalar value = acquisition(xc);
            if (value < bestValue) {
                bestValue = value;
                bestX = xc;
            }
        };

        for (Index s = 0; s < lhs.rows(); ++s) {
            const Vector<Scalar> start = lower.array()
                + (upper - lower).array() * lhs.row(s).transpose().array();
            consider(start);
            const auto res = solver.run(acquisitionFn, start);
            if (res.x.size() == dims && res.x.allFinite()) {
                consider(res.x);
            }
        }
        return bestX;
    }

    /// True when x nearly coincides with an already evaluated point (would
    /// make the correlation matrix singular).
    bool tooClose(const Vector<Scalar>& x, const std::vector<Vector<Scalar>>& xs) const
    {
        const Vector<Scalar> range = this->m_upperBounds - this->m_lowerBounds;
        for (const auto& xi : xs) {
            if (((x - xi).array() / range.array()).abs().maxCoeff() < Scalar(1e-8)) {
                return true;
            }
        }
        return false;
    }

    std::size_t m_maxFunctionEvaluations = 0;
    std::size_t m_doeSize = 0;
    std::string m_criterion = "EI";
    std::string m_kernel = "squar_exp";
    std::string m_regression = "constant";
    Scalar m_theta0 = Scalar(0.01);
    Scalar m_thetaBoundLower = Scalar(1e-6);
    Scalar m_thetaBoundUpper = Scalar(20);
    Scalar m_nugget = Scalar(100) * detail::eps<Scalar>();
    Scalar m_noise = Scalar(0);
    std::size_t m_hyperparameterStarts = 10;
    std::size_t m_acquisitionStarts = 20;
    Scalar m_targetObjective = -detail::inf<Scalar>();
    Scalar m_tolerance = Scalar(1e-05);
    long long m_seed = 0;
};

} // namespace globopt

#endif
