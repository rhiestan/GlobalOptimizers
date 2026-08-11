// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <globopt/globopt.hpp>
#include <globopt/benchmarks/go_benchmark.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

bool check(bool condition, const std::string& label)
{
    if (condition) {
        std::printf("[ OK ] %s\n", label.c_str());
    } else {
        std::printf("[FAIL] %s\n", label.c_str());
        ++g_failures;
    }
    return condition;
}

template <typename Scalar>
Scalar sphere(const globopt::Vector<Scalar>& x, globopt::Vector<Scalar>* grad)
{
    if (grad) {
        *grad = 2 * x;
    }
    return x.squaredNorm();
}

template <typename Scalar>
Scalar rosenbrock(const globopt::Vector<Scalar>& x, globopt::Vector<Scalar>* grad)
{
    const Scalar a = 1, b = 100;

    if (grad) {
        grad->resize(2);
        (*grad)(0) = -2 * (a - x(0)) - 4 * b * x(0) * (x(1) - x(0) * x(0));
        (*grad)(1) = 2 * b * (x(1) - x(0) * x(0));
    }

    return (a - x(0)) * (a - x(0)) + b * (x(1) - x(0) * x(0)) * (x(1) - x(0) * x(0));
}

// Booth: f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2, minimum at (1, 3)
template <typename Scalar>
Scalar booth(const globopt::Vector<Scalar>& x, globopt::Vector<Scalar>* grad)
{
    const Scalar u = x(0) + 2 * x(1) - 7;
    const Scalar v = 2 * x(0) + x(1) - 5;

    if (grad) {
        grad->resize(2);
        (*grad)(0) = 2 * u + 4 * v;
        (*grad)(1) = 4 * u + 2 * v;
    }

    return u * u + v * v;
}

void testSphere()
{
    auto opt = globopt::OptimizerFactory<double>::create("lbfgs");

    globopt::Vector<double> x0(5);
    x0 << 2.5, -1.0, 4.0, 0.5, -3.0;

    const auto res = opt->run(&sphere<double>, x0);

    check(res.success(), "sphere: converged (" + std::string(toString(res.status)) + ")");
    check(res.x.norm() < 1e-6, "sphere: solution near origin");
    check(std::abs(res.fval) < 1e-12, "sphere: objective near zero");
    check(res.functionEvaluations > 0, "sphere: function evaluations counted");
}

void testRosenbrock()
{
    auto opt = globopt::OptimizerFactory<double>::create(globopt::Algorithm::LBFGS);
    opt->setParam("max_iterations", 5000);

    globopt::Vector<double> x0(2);
    x0 << -1.2, 1.0;

    const auto res = opt->run(&rosenbrock<double>, x0);

    check(res.success(), "rosenbrock: converged (" + std::string(toString(res.status)) + ")");
    check((res.x - globopt::Vector<double>::Ones(2)).norm() < 1e-5, "rosenbrock: solution near (1, 1)");
}

void testBoundedBooth()
{
    // bounds contain the unconstrained minimum (1, 3) in the interior, so the
    // bounded run must find it while staying inside the box
    auto opt = globopt::OptimizerFactory<double>::create("L-BFGS");

    globopt::Vector<double> lb(2), ub(2);
    lb << -5.0, -5.0;
    ub << 5.0, 5.0;
    opt->setBounds(lb, ub);

    globopt::Vector<double> x0(2);
    x0 << 0.0, 0.0;

    const auto res = opt->run(&booth<double>, x0);

    check(res.success(), "booth (bounded): converged (" + std::string(toString(res.status)) + ")");
    check((res.x - (globopt::Vector<double>(2) << 1.0, 3.0).finished()).norm() < 1e-5,
          "booth (bounded): solution near (1, 3)");
    check((res.x.array() >= lb.array()).all() && (res.x.array() <= ub.array()).all(),
          "booth (bounded): bounds respected");
}

// N-dimensional Rosenbrock, f = sum 100*(x(i+1) - x(i)^2)^2 + (x(i) - 1)^2
template <typename Scalar>
Scalar rosenbrockN(const globopt::Vector<Scalar>& x, globopt::Vector<Scalar>* grad)
{
    const auto n = x.size();
    Scalar value = 0;

    for (Eigen::Index i = 0; i + 1 < n; ++i) {
        const Scalar t1 = x(i + 1) - x(i) * x(i);
        const Scalar t2 = x(i) - 1;
        value += 100 * t1 * t1 + t2 * t2;
    }

    if (grad) {
        grad->setZero(n);
        for (Eigen::Index i = 0; i + 1 < n; ++i) {
            (*grad)(i) += -400 * x(i) * (x(i + 1) - x(i) * x(i)) + 2 * (x(i) - 1);
            (*grad)(i + 1) += 200 * (x(i + 1) - x(i) * x(i));
        }
    }

    return value;
}

void testLbfgsbUnconstrained()
{
    auto opt = globopt::OptimizerFactory<double>::create("L-BFGS-B");

    globopt::Vector<double> x0(2);
    x0 << -1.2, 1.0;

    const auto res = opt->run(&rosenbrock<double>, x0);

    check(res.success(), "lbfgsb rosenbrock: converged (" + std::string(toString(res.status)) + ")");
    check((res.x - globopt::Vector<double>::Ones(2)).norm() < 1e-3, "lbfgsb rosenbrock: solution near (1, 1)");
    check(res.fval < 1e-8, "lbfgsb rosenbrock: objective near zero");
}

void testLbfgsbConstrainedRosenbrock()
{
    // constrained 3D Rosenbrock: the minimum subject to these bounds is
    // f = 7.75 at (0.5, 0.5, 0.35), on the boundary in every coordinate
    auto opt = globopt::OptimizerFactory<double>::create(globopt::Algorithm::LBFGSB);

    globopt::Vector<double> x0(3), lb(3), ub(3);
    x0 << 0.0, 5.0, 5.0;
    lb << -0.5, 0.5, 0.35;
    ub << 0.5, 10.0, 10.0;
    opt->setBounds(lb, ub);

    const auto res = opt->run(&rosenbrockN<double>, x0);

    check(res.success(), "lbfgsb rosenbrock3 (bounded): converged (" + std::string(toString(res.status)) + ")");
    check(std::abs(res.fval - 7.75) < 1e-6, "lbfgsb rosenbrock3 (bounded): objective near 7.75");
    check((res.x - (globopt::Vector<double>(3) << 0.5, 0.5, 0.35).finished()).norm() < 1e-4,
          "lbfgsb rosenbrock3 (bounded): solution near (0.5, 0.5, 0.35)");
    check((res.x.array() >= lb.array()).all() && (res.x.array() <= ub.array()).all(),
          "lbfgsb rosenbrock3 (bounded): bounds respected");
}

void testBoundaryOptimum()
{
    // regression test: the constrained minimum of Booth subject to y <= 2 lies
    // ON the boundary, at (1.8, 2) with f = 1.8. The former transform-based
    // bounds handling falsely converged at a corner of the box here.
    globopt::Vector<double> lb(2), ub(2), expected(2);
    lb << -5.0, -5.0;
    ub << 5.0, 2.0;
    expected << 1.8, 2.0;

    globopt::Vector<double> x0(2);
    x0 << 0.0, 0.0;

    for (const char* name : {"L-BFGS-B", "L-BFGS"}) {
        auto opt = globopt::OptimizerFactory<double>::create(name);
        opt->setBounds(lb, ub);

        const auto res = opt->run(&booth<double>, x0);
        const std::string label = std::string(name) + " booth (boundary optimum)";

        check(res.success(), label + ": converged (" + std::string(toString(res.status)) + ")");
        check((res.x - expected).norm() < 1e-4, label + ": solution near (1.8, 2)");
        check(std::abs(res.fval - 1.8) < 1e-6, label + ": objective near 1.8");
        check((res.x.array() >= lb.array()).all() && (res.x.array() <= ub.array()).all(),
              label + ": bounds respected");
    }
}

void testLbfgsDelegatesBounded()
{
    auto opt = globopt::OptimizerFactory<double>::create("lbfgs");

    globopt::Vector<double> lb(2), ub(2);
    lb << -5.0, -5.0;
    ub << 5.0, 5.0;
    opt->setBounds(lb, ub);

    globopt::Vector<double> x0(2);
    x0 << 0.0, 0.0;

    const auto res = opt->run(&booth<double>, x0);

    check(res.message.find("L-BFGS-B") != std::string::npos,
          "lbfgs (bounded): delegation to L-BFGS-B reported in message");
}

void testLbfgsbFloatScalar()
{
    auto opt = globopt::OptimizerFactory<float>::create("lbfgsb");
    opt->setParam("gradient_tolerance", 1e-3);

    globopt::Vector<float> x0(3);
    x0 << 1.0f, -2.0f, 3.0f;

    const auto res = opt->run(&sphere<float>, x0);

    check(res.success(), "lbfgsb sphere<float>: converged");
    check(res.x.norm() < 1e-2f, "lbfgsb sphere<float>: solution near origin");
}

void testAmpgoBird()
{
    // mirrors go_amp.py's __main__: Bird function, target set to the known
    // global optimum, 20000 evaluations budget
    const auto& bird = globopt::benchmarks::problem("Bird");

    auto objective = globopt::withNumericalGradient<double>(bird.objective);

    auto opt = globopt::OptimizerFactory<double>::create("AMPGO");
    opt->setBounds(bird.lower, bird.upper);
    opt->setParam("target_objective", bird.fglob);
    opt->setParam("tolerance", 1e-6);
    opt->setParam("max_function_evaluations", 20000);
    opt->setParam("total_iterations", 2000);
    opt->setParam("seed", 7);

    std::mt19937_64 rng(7);
    const auto res = opt->run(objective, bird.randomStart(rng));

    check(res.success(), "ampgo bird: global optimum found (" + std::string(toString(res.status)) + ")");
    check(res.fval < bird.fglob + 1e-6, "ampgo bird: objective at global optimum");
    check(res.functionEvaluations <= 20000 + 100, "ampgo bird: evaluation budget respected");
}

void testAmpgoSixHumpCamel()
{
    const auto& camel = globopt::benchmarks::problem("SixHumpCamel");

    auto objective = globopt::withNumericalGradient<double>(camel.objective);

    globopt::AMPGO<double> opt;
    opt.setBounds(camel.lower, camel.upper);
    opt.setParam("target_objective", camel.fglob);
    opt.setParam("tolerance", 1e-5);
    opt.setParam("max_function_evaluations", 20000);
    opt.setParam("total_iterations", 2000);
    opt.setParam("seed", 11);

    std::mt19937_64 rng(11);
    const auto res = opt.run(objective, camel.randomStart(rng));

    check(res.success(), "ampgo six-hump camel: global optimum found");
}

void testAmpgoAnalyticGradient()
{
    // AMPGO on Rosenbrock with an analytic gradient and no target: it should
    // run its budget and still land on the global minimum (1, 1)
    globopt::AMPGO<double> opt;
    opt.setParam("max_function_evaluations", 5000);
    opt.setParam("seed", 3);

    globopt::Vector<double> x0(2);
    x0 << -1.2, 1.0;

    const auto res = opt.run(&rosenbrock<double>, x0);

    check(res.fval < 1e-8, "ampgo rosenbrock (analytic gradient): minimum found");
    check(!res.message.empty(), "ampgo rosenbrock: tunneling stats reported");
}

void testAmpgoParamValidation()
{
    globopt::AMPGO<double> opt;
    opt.setParam("tabu_strategy", "sideways");

    globopt::Vector<double> x0(2);
    x0 << 0.0, 0.0;

    check(opt.run(&sphere<double>, x0).status == globopt::Status::InvalidInput,
          "ampgo: invalid tabu_strategy rejected");

    globopt::AMPGO<double> opt2;
    opt2.setParam("local_solver", "nelder-mead");
    check(opt2.run(&sphere<double>, x0).status == globopt::Status::InvalidInput,
          "ampgo: invalid local_solver rejected");
}

void testBenchmarkSuite()
{
    const auto& problems = globopt::benchmarks::allProblems();

    check(problems.size() == 202, "benchmarks: all 202 problems ported ("
          + std::to_string(problems.size()) + ")");

    std::mt19937_64 rng(1);
    bool boundsOk = true, evalOk = true, optimaOk = true;

    for (const auto& p : problems) {
        if (p.lower.size() != p.upper.size() || p.lower.size() == 0
            || !(p.lower.array() <= p.upper.array()).all()) {
            boundsOk = false;
        }

        // every objective must be callable at an in-bounds point
        const auto x = p.randomStart(rng);
        const double f = p.objective(x);
        if (std::isnan(f) && p.name != "Csendes" && p.name != "Damavandi") {
            evalOk = false;
            std::printf("       NaN at random start: %s\n", p.name.c_str());
        }

        for (const auto& opt : p.globalOptima) {
            if (opt.size() != p.lower.size()) {
                optimaOk = false;
                std::printf("       optimum size mismatch: %s\n", p.name.c_str());
            }
        }
    }

    check(boundsOk, "benchmarks: bounds well-formed");
    check(evalOk, "benchmarks: objectives evaluable in bounds");
    check(optimaOk, "benchmarks: stated optima match dimensions");
}

void testLipoBird()
{
    const auto& bird = globopt::benchmarks::problem("Bird");

    auto opt = globopt::OptimizerFactory<double>::create("LIPO");
    opt->setBounds(bird.lower, bird.upper);
    opt->setParam("target_objective", bird.fglob);
    opt->setParam("tolerance", 1e-4);
    opt->setParam("max_function_evaluations", 300);
    opt->setParam("seed", 5);

    auto objective = [&](const globopt::Vector<double>& x, globopt::Vector<double>*) {
        return bird.objective(x);
    };

    std::mt19937_64 rng(5);
    const auto res = opt->run(objective, bird.randomStart(rng));

    check(res.success(), "lipo bird: global optimum found (" + std::string(toString(res.status))
          + ", f = " + std::to_string(res.fval) + ")");
    check(res.functionEvaluations <= 300, "lipo bird: evaluation budget respected");
}

void testLipoHolderTable()
{
    const auto& holder = globopt::benchmarks::problem("HolderTable");

    globopt::LIPO<double> opt;
    opt.setBounds(holder.lower, holder.upper);
    opt.setParam("target_objective", holder.fglob);
    opt.setParam("tolerance", 1e-4);
    opt.setParam("max_function_evaluations", 300);
    opt.setParam("seed", 9);

    auto objective = [&](const globopt::Vector<double>& x, globopt::Vector<double>*) {
        return holder.objective(x);
    };

    std::mt19937_64 rng(9);
    const auto res = opt.run(objective, holder.randomStart(rng));

    check(res.success(), "lipo holder table: global optimum found (f = "
          + std::to_string(res.fval) + ")");
}

void testLipoRequiresBounds()
{
    globopt::LIPO<double> opt;

    globopt::Vector<double> x0(2);
    x0 << 0.0, 0.0;

    check(opt.run(&sphere<double>, x0).status == globopt::Status::InvalidInput,
          "lipo: missing bounds rejected");

    globopt::Vector<double> lb(2), ub(2);
    lb << -1.0, -1.0;
    ub << 1.0, std::numeric_limits<double>::infinity();
    opt.setBounds(lb, ub);
    check(opt.run(&sphere<double>, x0).status == globopt::Status::InvalidInput,
          "lipo: non-finite bounds rejected");
}

void testEgoSphere()
{
    auto opt = globopt::OptimizerFactory<double>::create("EGO");

    globopt::Vector<double> lb(2), ub(2);
    lb << -2.0, -2.0;
    ub << 3.0, 3.0;
    opt->setBounds(lb, ub);
    opt->setParam("target_objective", 0.0);
    opt->setParam("tolerance", 1e-4);
    opt->setParam("max_function_evaluations", 80);
    opt->setParam("seed", 3);

    globopt::Vector<double> x0(2);
    x0 << 2.0, -1.5;
    const auto res = opt->run(&sphere<double>, x0);

    check(res.success(), "ego sphere: global optimum found (" + std::string(toString(res.status))
          + ", f = " + std::to_string(res.fval) + ")");
    check(res.functionEvaluations <= 80, "ego sphere: evaluation budget respected");
}

void testEgoBranin()
{
    const auto& branin = globopt::benchmarks::problem("Branin01");

    auto opt = globopt::OptimizerFactory<double>::create("EGO");
    opt->setBounds(branin.lower, branin.upper);
    opt->setParam("target_objective", branin.fglob);
    opt->setParam("tolerance", 1e-3);
    opt->setParam("max_function_evaluations", 120);
    opt->setParam("seed", 7);

    auto objective = [&](const globopt::Vector<double>& x, globopt::Vector<double>*) {
        return branin.objective(x);
    };

    std::mt19937_64 rng(7);
    const auto res = opt->run(objective, branin.randomStart(rng));

    check(res.success(), "ego branin: global optimum found (" + std::string(toString(res.status))
          + ", f = " + std::to_string(res.fval) + ")");
    check(res.functionEvaluations <= 120, "ego branin: evaluation budget respected");
}

void testEgoParamValidation()
{
    globopt::EGO<double> opt;

    globopt::Vector<double> x0(2);
    x0 << 0.0, 0.0;

    check(opt.run(&sphere<double>, x0).status == globopt::Status::InvalidInput,
          "ego: missing bounds rejected");

    globopt::Vector<double> lb(2), ub(2);
    lb << -1.0, -1.0;
    ub << 1.0, std::numeric_limits<double>::infinity();
    opt.setBounds(lb, ub);
    check(opt.run(&sphere<double>, x0).status == globopt::Status::InvalidInput,
          "ego: non-finite bounds rejected");

    ub << 1.0, 1.0;
    opt.setBounds(lb, ub);
    opt.setParam("criterion", "nonsense");
    check(opt.run(&sphere<double>, x0).status == globopt::Status::InvalidInput,
          "ego: unknown criterion rejected");
    opt.setParam("criterion", "EI");
    opt.setParam("kernel", "nonsense");
    check(opt.run(&sphere<double>, x0).status == globopt::Status::InvalidInput,
          "ego: unknown kernel rejected");
}

// A smooth 2-D test function for the surrogate tests below.
double krigingTestFunction(const globopt::Vector<double>& x)
{
    return std::sin(x(0)) * std::cos(x(1)) + 0.1 * x(0) * x(0);
}

// Grid samples of krigingTestFunction over [-2, 2]^2.
void krigingTestSamples(std::vector<globopt::Vector<double>>& xs, std::vector<double>& ys,
                        const int perAxis = 5)
{
    for (int i = 0; i < perAxis; ++i) {
        for (int j = 0; j < perAxis; ++j) {
            globopt::Vector<double> x(2);
            x << -2.0 + 4.0 * i / (perAxis - 1), -2.0 + 4.0 * j / (perAxis - 1);
            xs.push_back(x);
            ys.push_back(krigingTestFunction(x));
        }
    }
}

// The Kriging surrogate is an interpolator: at a training point the posterior
// mean must reproduce the observed value and the posterior variance must
// vanish. This exercises the ported likelihood / GLS core directly.
void testKrigingInterpolates()
{
    namespace eg = globopt::detail::ego;

    std::vector<globopt::Vector<double>> xs;
    std::vector<double> ys;
    krigingTestSamples(xs, ys);

    eg::KrigingModel<double>::Options options;
    eg::KrigingModel<double> model(options);
    std::mt19937_64 rng(11);

    check(model.fit(xs, ys, rng), "kriging: fit succeeds");

    double maxError = 0.0, maxVariance = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        double mean = 0.0, variance = 0.0;
        model.predict(xs[i], &mean, &variance);
        maxError = std::max(maxError, std::abs(mean - ys[i]));
        maxVariance = std::max(maxVariance, variance);
    }

    check(maxError < 1e-6, "kriging: interpolates the training points (max error "
          + std::to_string(maxError) + ")");
    check(maxVariance < 1e-6, "kriging: vanishing variance at the training points (max "
          + std::to_string(maxVariance) + ")");

    // held-out point: the GP should generalize and report positive uncertainty
    globopt::Vector<double> xt(2);
    xt << 0.4, -0.7;
    double mean = 0.0, variance = 0.0;
    model.predict(xt, &mean, &variance);

    check(std::abs(mean - krigingTestFunction(xt)) < 0.1,
          "kriging: predicts a held-out point (error "
          + std::to_string(std::abs(mean - krigingTestFunction(xt))) + ")");
    check(variance > 0.0, "kriging: positive variance away from the training points");
}

// refresh() re-conditions the model on new samples with the hyperparameters of
// the previous fit(); it must interpolate the enlarged sample set just as
// fit() would, and must refuse to run before any fit().
void testKrigingRefresh()
{
    namespace eg = globopt::detail::ego;

    std::vector<globopt::Vector<double>> xs;
    std::vector<double> ys;
    krigingTestSamples(xs, ys);

    eg::KrigingModel<double>::Options options;
    eg::KrigingModel<double> fresh(options);
    check(!fresh.refresh(xs, ys), "kriging: refresh before fit is rejected");

    eg::KrigingModel<double> model(options);
    std::mt19937_64 rng(11);
    check(model.fit(xs, ys, rng), "kriging: fit before refresh succeeds");
    const globopt::Vector<double> thetaAfterFit = model.theta();

    // add samples the model has not seen and re-condition without a refit
    for (const double offset : {0.3, -1.1, 1.7}) {
        globopt::Vector<double> x(2);
        x << offset, 0.5 * offset;
        xs.push_back(x);
        ys.push_back(krigingTestFunction(x));
    }
    check(model.refresh(xs, ys), "kriging: refresh on enlarged sample set succeeds");
    check((model.theta() - thetaAfterFit).norm() == 0.0,
          "kriging: refresh keeps the fitted hyperparameters");

    double maxError = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        double mean = 0.0;
        model.predict(xs[i], &mean, nullptr);
        maxError = std::max(maxError, std::abs(mean - ys[i]));
    }
    check(maxError < 1e-6, "kriging: refresh interpolates the new points too (max error "
          + std::to_string(maxError) + ")");
}

// Every kernel and trend must produce a usable interpolating surrogate.
void testKrigingKernelsAndRegressions()
{
    namespace eg = globopt::detail::ego;

    std::vector<globopt::Vector<double>> xs;
    std::vector<double> ys;
    krigingTestSamples(xs, ys, 6);

    const std::pair<eg::Kernel, const char*> kernels[] = {
        {eg::Kernel::SquarExp, "squar_exp"},
        {eg::Kernel::AbsExp, "abs_exp"},
        {eg::Kernel::Matern32, "matern32"},
        {eg::Kernel::Matern52, "matern52"},
    };
    const std::pair<eg::RegressionType, const char*> regressions[] = {
        {eg::RegressionType::Constant, "constant"},
        {eg::RegressionType::Linear, "linear"},
        {eg::RegressionType::Quadratic, "quadratic"},
    };

    for (const auto& kernel : kernels) {
        for (const auto& regression : regressions) {
            eg::KrigingModel<double>::Options options;
            options.kernel = kernel.first;
            options.regression = regression.first;
            eg::KrigingModel<double> model(options);
            std::mt19937_64 rng(5);

            const std::string label =
                std::string(kernel.second) + " / " + regression.second;
            if (!check(model.fit(xs, ys, rng), "kriging (" + label + "): fit succeeds")) {
                continue;
            }

            double maxError = 0.0;
            for (std::size_t i = 0; i < xs.size(); ++i) {
                double mean = 0.0;
                model.predict(xs[i], &mean, nullptr);
                maxError = std::max(maxError, std::abs(mean - ys[i]));
            }
            check(maxError < 1e-5, "kriging (" + label + "): interpolates (max error "
                  + std::to_string(maxError) + ")");
        }
    }
}

// Independent check of the ported Kriging algebra: the model computes its
// posterior through a Cholesky factorization of R and a QR of C^-1 F, which is
// numerically convenient but hard to eyeball. Here the same posterior is
// recomputed from the textbook universal-kriging formulas with a dense solve
// and a hand-written squar_exp kernel, so a mistake in the standardization,
// the GLS trend or the variance formula would show up as a mismatch.
void testKrigingAgainstReferenceFormulas()
{
    namespace eg = globopt::detail::ego;
    using Matrix = globopt::Matrix<double>;
    using Vector = globopt::Vector<double>;

    std::vector<Vector> xs;
    std::vector<double> ys;
    krigingTestSamples(xs, ys, 3);
    const globopt::Index nt = static_cast<globopt::Index>(xs.size());

    eg::KrigingModel<double>::Options options; // squar_exp / constant trend
    eg::KrigingModel<double> model(options);
    std::mt19937_64 rng(9);
    if (!check(model.fit(xs, ys, rng), "kriging reference: fit succeeds")) {
        return;
    }
    const Vector theta = model.theta(); // compare at the hyperparameters it chose

    // -- standardization, as SMT does it (population standard deviation) -----
    Matrix X(nt, 2);
    Vector y(nt);
    for (globopt::Index i = 0; i < nt; ++i) {
        X.row(i) = xs[static_cast<std::size_t>(i)].transpose();
        y(i) = ys[static_cast<std::size_t>(i)];
    }
    const Vector xOffset = X.colwise().mean();
    const Vector xScale =
        (X.rowwise() - xOffset.transpose()).array().square().colwise().mean().sqrt();
    const double yMean = y.mean();
    const double yStd = std::sqrt((y.array() - yMean).square().mean());

    const Matrix Xn =
        (X.rowwise() - xOffset.transpose()).array().rowwise() / xScale.transpose().array();
    const Vector yn = (y.array() - yMean) / yStd;

    // -- correlation matrix, hand-written squar_exp --------------------------
    auto correlation = [&theta](const Vector& a, const Vector& b) {
        double s = 0.0;
        for (globopt::Index k = 0; k < a.size(); ++k) {
            s += theta(k) * (a(k) - b(k)) * (a(k) - b(k));
        }
        return std::exp(-s);
    };

    Matrix R(nt, nt);
    for (globopt::Index i = 0; i < nt; ++i) {
        for (globopt::Index j = 0; j < nt; ++j) {
            R(i, j) = (i == j) ? 1.0 + options.nugget + options.noise
                               : correlation(Xn.row(i).transpose(), Xn.row(j).transpose());
        }
    }

    // -- generalized least squares trend and the kriging weights -------------
    const Matrix F = Matrix::Ones(nt, 1);
    const Matrix Rinv = R.inverse();
    const Matrix FtRinvF = F.transpose() * Rinv * F;
    const Vector beta = FtRinvF.inverse() * (F.transpose() * Rinv * yn);
    const Vector residual = yn - F * beta;
    const Vector gamma = Rinv * residual;
    const double sigma2 = residual.dot(Rinv * residual) / static_cast<double>(nt);

    double maxMeanError = 0.0, maxVarianceError = 0.0;
    for (const double a : {-1.7, -0.6, 0.0, 0.8, 1.9}) {
        for (const double b : {-1.3, 0.4, 1.5}) {
            Vector x(2);
            x << a, b;
            const Vector xn = (x - xOffset).cwiseQuotient(xScale);

            Vector r(nt);
            for (globopt::Index i = 0; i < nt; ++i) {
                r(i) = correlation(Xn.row(i).transpose(), xn);
            }

            const double referenceMean = yMean + yStd * (beta(0) + r.dot(gamma));
            const Vector u = F.transpose() * Rinv * r - Vector::Ones(1);
            const double referenceVariance =
                sigma2 * (1.0 - r.dot(Rinv * r) + u.dot(FtRinvF.inverse() * u)) * yStd * yStd;

            double mean = 0.0, variance = 0.0;
            model.predict(x, &mean, &variance);

            maxMeanError = std::max(maxMeanError, std::abs(mean - referenceMean));
            maxVarianceError =
                std::max(maxVarianceError, std::abs(variance - std::max(0.0, referenceVariance)));
        }
    }

    check(maxMeanError < 1e-8, "kriging reference: posterior mean matches the GLS formula "
          "(max error " + std::to_string(maxMeanError) + ")");
    check(maxVarianceError < 1e-8, "kriging reference: posterior variance matches the "
          "universal-kriging formula (max error " + std::to_string(maxVarianceError) + ")");
}

// A non-zero noise term turns the interpolator into a regressor: the posterior
// no longer has to pass exactly through the observations.
void testKrigingNoise()
{
    namespace eg = globopt::detail::ego;

    std::vector<globopt::Vector<double>> xs;
    std::vector<double> ys;
    krigingTestSamples(xs, ys, 4);

    eg::KrigingModel<double>::Options options;
    options.noise = 1e-2;
    eg::KrigingModel<double> model(options);
    std::mt19937_64 rng(13);
    check(model.fit(xs, ys, rng), "kriging noise: fit succeeds");

    double maxError = 0.0, maxVariance = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        double mean = 0.0, variance = 0.0;
        model.predict(xs[i], &mean, &variance);
        maxError = std::max(maxError, std::abs(mean - ys[i]));
        maxVariance = std::max(maxVariance, variance);
    }

    check(maxError > 1e-6, "kriging noise: smooths rather than interpolates (max deviation "
          + std::to_string(maxError) + ")");
    check(maxError < 0.5, "kriging noise: still follows the data");
    check(maxVariance > 0.0, "kriging noise: keeps positive variance at the samples");
}

// Expected improvement is non-negative everywhere, vanishes at an already
// sampled point (no uncertainty, no improvement) and is positive in an
// unexplored region.
void testExpectedImprovement()
{
    namespace eg = globopt::detail::ego;

    // A coarse design on purpose: on a dense grid this smooth function is
    // fitted so well that the posterior is numerically certain everywhere and
    // the expected improvement vanishes over the whole box.
    std::vector<globopt::Vector<double>> xs;
    std::vector<double> ys;
    krigingTestSamples(xs, ys, 3);

    eg::KrigingModel<double>::Options options;
    eg::KrigingModel<double> model(options);
    std::mt19937_64 rng(3);
    check(model.fit(xs, ys, rng), "ei: surrogate fit succeeds");

    const double fmin = *std::min_element(ys.begin(), ys.end());

    bool allNonNegative = true;
    for (double x0 = -2.0; x0 <= 2.0; x0 += 0.25) {
        for (double x1 = -2.0; x1 <= 2.0; x1 += 0.25) {
            globopt::Vector<double> x(2);
            x << x0, x1;
            if (eg::expectedImprovement(model, fmin, x) < 0.0) {
                allNonNegative = false;
            }
        }
    }
    check(allNonNegative, "ei: non-negative over the domain");

    // a point between grid samples is uncertain, so improvement is possible
    globopt::Vector<double> gap(2);
    gap << -1.0, -1.0;
    const double gapEi = eg::expectedImprovement(model, fmin, gap);
    check(gapEi > 0.0, "ei: positive in an unexplored region");

    // At an already sampled point the posterior is (numerically) certain, so
    // the expected improvement collapses: not exactly zero in floating point,
    // but negligible next to an unexplored point.
    double maxTrainingEi = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        maxTrainingEi = std::max(maxTrainingEi, eg::expectedImprovement(model, fmin, xs[i]));
    }
    check(maxTrainingEi < 1e-3 * gapEi,
          "ei: negligible at the training points (" + std::to_string(maxTrainingEi)
          + " vs " + std::to_string(gapEi) + " unexplored)");
}

// EGO must accept every documented kernel, trend and infill criterion and make
// real progress on a simple bowl with each of them.
void testEgoOptions()
{
    globopt::Vector<double> lb(2), ub(2);
    lb << -2.0, -2.0;
    ub << 3.0, 3.0;
    globopt::Vector<double> x0(2);
    x0 << 2.0, -1.5;
    const double startValue = x0.squaredNorm();

    auto runWith = [&](const char* param, const char* value, const int doeSize) {
        auto opt = globopt::OptimizerFactory<double>::create("EGO");
        opt->setBounds(lb, ub);
        opt->setParam("max_function_evaluations", 40);
        opt->setParam("seed", 17);
        opt->setParam(param, value);
        if (doeSize > 0) {
            opt->setParam("doe_size", doeSize);
        }
        const auto res = opt->run(&sphere<double>, x0);

        const std::string label =
            std::string("ego (") + param + " = " + value + ")";
        check(res.status != globopt::Status::InvalidInput, label + ": accepted");
        check(res.fval < 0.1 * startValue, label + ": improves on the start point (f = "
              + std::to_string(res.fval) + ")");
        check((res.x.array() >= lb.array()).all() && (res.x.array() <= ub.array()).all(),
              label + ": stays inside the box");
        check(res.functionEvaluations <= 40, label + ": budget respected");
    };

    for (const char* kernel : {"squar_exp", "abs_exp", "matern32", "matern52"}) {
        runWith("kernel", kernel, 0);
    }
    for (const char* criterion : {"EI", "SBO", "LCB"}) {
        runWith("criterion", criterion, 0);
    }
    // linear and quadratic trends need more DOE points than trend coefficients
    for (const char* regression : {"constant", "linear", "quadratic"}) {
        runWith("regression", regression, 12);
    }
}

// The evaluation budget, the DOE size and the reported result must be
// consistent: this is what makes benchmark numbers trustworthy.
void testEgoBudgetAndResult()
{
    globopt::Vector<double> lb(2), ub(2);
    lb << -2.0, -2.0;
    ub << 3.0, 3.0;
    globopt::Vector<double> x0(2);
    x0 << 2.0, -1.5;

    std::size_t observed = 0;
    auto counted = [&](const globopt::Vector<double>& x, globopt::Vector<double>*) {
        ++observed;
        return x.squaredNorm();
    };

    auto opt = globopt::OptimizerFactory<double>::create("EGO");
    opt->setBounds(lb, ub);
    opt->setParam("max_function_evaluations", 30);
    opt->setParam("doe_size", 8);
    opt->setParam("seed", 23);
    const auto res = opt->run(counted, x0);

    check(res.status == globopt::Status::MaxFunctionEvaluationsReached,
          "ego budget: spends the whole budget without a target");
    check(res.functionEvaluations == 30, "ego budget: exactly the requested evaluations ("
          + std::to_string(res.functionEvaluations) + ")");
    check(observed == res.functionEvaluations,
          "ego budget: reported evaluations match actual objective calls");
    check(res.iterations == 30 - 8, "ego budget: DOE size honoured ("
          + std::to_string(res.iterations) + " infill iterations)");
    check(std::abs(res.fval - res.x.squaredNorm()) < 1e-12,
          "ego budget: reported fval matches the reported x");
    check(res.message.find("8 DOE") != std::string::npos,
          "ego budget: message reports the DOE split (" + res.message + ")");
}

// A fixed seed must give a reproducible run - benchmark sweeps depend on it.
void testEgoDeterminism()
{
    globopt::Vector<double> lb(2), ub(2);
    lb << -2.0, -2.0;
    ub << 3.0, 3.0;
    globopt::Vector<double> x0(2);
    x0 << 2.0, -1.5;

    auto runSeeded = [&](const long long seed) {
        auto opt = globopt::OptimizerFactory<double>::create("EGO");
        opt->setBounds(lb, ub);
        opt->setParam("max_function_evaluations", 30);
        opt->setParam("seed", seed);
        return opt->run(&sphere<double>, x0);
    };

    const auto first = runSeeded(31);
    const auto again = runSeeded(31);
    const auto other = runSeeded(32);

    check(first.fval == again.fval && (first.x - again.x).norm() == 0.0,
          "ego determinism: same seed reproduces the run exactly");
    check(other.fval != first.fval,
          "ego determinism: a different seed explores differently");
}

// hyperparameter_refit_interval trades hyperparameter re-estimation for speed.
// Interval 1 must reproduce the every-iteration behaviour exactly, and a larger
// interval must still optimize while doing measurably less work.
void testEgoRefitInterval()
{
    globopt::Vector<double> lb(2), ub(2);
    lb << -2.0, -2.0;
    ub << 3.0, 3.0;
    globopt::Vector<double> x0(2);
    x0 << 2.0, -1.5;

    auto runWithInterval = [&](const long long interval, double& seconds) {
        auto opt = globopt::OptimizerFactory<double>::create("EGO");
        opt->setBounds(lb, ub);
        opt->setParam("max_function_evaluations", 60);
        opt->setParam("seed", 41);
        if (interval > 0) {
            opt->setParam("hyperparameter_refit_interval", interval);
        }
        const auto start = std::chrono::steady_clock::now();
        const auto res = opt->run(&sphere<double>, x0);
        seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return res;
    };

    double defaultSeconds = 0.0, oneSeconds = 0.0, tenSeconds = 0.0;
    const auto byDefault = runWithInterval(0, defaultSeconds);
    const auto interval1 = runWithInterval(1, oneSeconds);
    const auto interval10 = runWithInterval(10, tenSeconds);

    check(byDefault.fval == interval1.fval && (byDefault.x - interval1.x).norm() == 0.0,
          "ego refit interval: 1 is the default and refits every iteration");
    check(interval10.functionEvaluations == interval1.functionEvaluations,
          "ego refit interval: budget unaffected");
    check(interval10.fval < 1e-2, "ego refit interval: 10 still optimizes (f = "
          + std::to_string(interval10.fval) + ")");
    check(tenSeconds < oneSeconds,
          "ego refit interval: 10 is cheaper than 1 (" + std::to_string(tenSeconds)
          + " s vs " + std::to_string(oneSeconds) + " s)");
}

void testUnboundedBooth()
{
    auto opt = globopt::OptimizerFactory<double>::create("l_bfgs");

    globopt::Vector<double> x0(2);
    x0 << 0.0, 0.0;

    const auto res = opt->run(&booth<double>, x0);

    check(res.success(), "booth: converged");
    check((res.x - (globopt::Vector<double>(2) << 1.0, 3.0).finished()).norm() < 1e-6,
          "booth: solution near (1, 3)");
}

void testFloatScalar()
{
    auto opt = globopt::OptimizerFactory<float>::create("lbfgs");
    opt->setParam("gradient_tolerance", 1e-4);

    globopt::Vector<float> x0(3);
    x0 << 1.0f, -2.0f, 3.0f;

    const auto res = opt->run(&sphere<float>, x0);

    check(res.success(), "sphere<float>: converged");
    check(res.x.norm() < 1e-3f, "sphere<float>: solution near origin");
}

void testParamInterface()
{
    auto opt = globopt::OptimizerFactory<double>::create("lbfgs");

    opt->setParam("memory", 5);
    check(std::get<long long>(opt->getParam("memory")) == 5, "params: integer round-trip");

    opt->setParam("gradient_tolerance", 1e-10);
    check(std::get<double>(opt->getParam("gradient_tolerance")) == 1e-10, "params: double round-trip");

    bool threw = false;
    try {
        opt->setParam("no_such_param", 1.0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "params: unknown parameter throws");

    threw = false;
    try {
        globopt::OptimizerFactory<double>::create("no-such-optimizer");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "factory: unknown optimizer throws");

    check(!opt->listParams().empty(), "params: listParams non-empty");
    check(std::string(opt->name()) == "L-BFGS", "optimizer: name");

    check(globopt::OptimizerFactory<double>::available().size() == 5, "factory: five optimizers available");
    check(std::string(globopt::OptimizerFactory<double>::create("MaxLIPO")->name()) == "LIPO",
          "factory: LIPO by name");
    check(std::string(globopt::OptimizerFactory<double>::create("l-bfgs-b")->name()) == "L-BFGS-B",
          "factory: L-BFGS-B by name");
    check(std::string(globopt::OptimizerFactory<double>::create("ampgo")->name()) == "AMPGO",
          "factory: AMPGO by name");
    check(std::string(globopt::OptimizerFactory<double>::create("Bayesian")->name()) == "EGO",
          "factory: EGO by name");
}

void testInvalidInput()
{
    auto opt = globopt::OptimizerFactory<double>::create("lbfgs");

    globopt::Vector<double> empty;
    check(opt->run(&sphere<double>, empty).status == globopt::Status::InvalidInput,
          "run: empty initial point rejected");

    globopt::Vector<double> nonFinite(2);
    nonFinite << 1.0, std::nan("");
    check(opt->run(&sphere<double>, nonFinite).status == globopt::Status::InvalidInput,
          "run: non-finite initial point rejected");
}

} // namespace

int main()
{
    testSphere();
    testRosenbrock();
    testUnboundedBooth();
    testBoundedBooth();
    testLbfgsbUnconstrained();
    testLbfgsbConstrainedRosenbrock();
    testBoundaryOptimum();
    testLbfgsDelegatesBounded();
    testBenchmarkSuite();
    testAmpgoBird();
    testAmpgoSixHumpCamel();
    testAmpgoAnalyticGradient();
    testAmpgoParamValidation();
    testLipoBird();
    testLipoHolderTable();
    testLipoRequiresBounds();
    testEgoSphere();
    testEgoBranin();
    testEgoParamValidation();
    testKrigingInterpolates();
    testKrigingRefresh();
    testKrigingKernelsAndRegressions();
    testKrigingAgainstReferenceFormulas();
    testKrigingNoise();
    testExpectedImprovement();
    testEgoOptions();
    testEgoBudgetAndResult();
    testEgoDeterminism();
    testEgoRefitInterval();
    testFloatScalar();
    testLbfgsbFloatScalar();
    testParamInterface();
    testInvalidInput();

    if (g_failures > 0) {
        std::printf("\n%d test(s) FAILED\n", g_failures);
        return 1;
    }

    std::printf("\nAll tests passed\n");
    return 0;
}
