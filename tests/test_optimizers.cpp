// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <globopt/globopt.hpp>
#include <globopt/benchmarks/go_benchmark.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>

namespace {

int g_failures = 0;

void check(bool condition, const std::string& label)
{
    if (condition) {
        std::printf("[ OK ] %s\n", label.c_str());
    } else {
        std::printf("[FAIL] %s\n", label.c_str());
        ++g_failures;
    }
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
