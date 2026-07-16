# GlobalOptimizers

A header-only, templated C++17 suite of global and local optimization algorithms based on [Eigen](https://eigen.tuxfamily.org), developed with Claude.

## Design

- **Header-only**: add `include/` to your include path and `#include <globopt/globopt.hpp>`.
- **Templated**: every optimizer is templated on the scalar type (`double`, `float`, ...).
- **Uniform workflow**: create an optimizer through the factory, set parameters, run.

```cpp
#include <globopt/globopt.hpp>

using Vec = globopt::Vector<double>;

auto rosenbrock = [](const Vec& x, Vec* grad) -> double {
    if (grad) {
        grad->resize(2);
        (*grad)(0) = -2.0 * (1.0 - x(0)) - 400.0 * x(0) * (x(1) - x(0) * x(0));
        (*grad)(1) = 200.0 * (x(1) - x(0) * x(0));
    }
    return std::pow(1.0 - x(0), 2) + 100.0 * std::pow(x(1) - x(0) * x(0), 2);
};

// 1. create
auto optimizer = globopt::OptimizerFactory<double>::create("L-BFGS");

// 2. configure
optimizer->setParam("max_iterations", 2000);
optimizer->setParam("gradient_tolerance", 1e-10);

// 3. run
Vec x0(2);
x0 << -1.2, 1.0;
globopt::Result<double> result = optimizer->run(rosenbrock, x0);

std::cout << result.x.transpose() << "  f = " << result.fval << "\n";
```

The objective is a callable `Scalar(const Vector& x, Vector* grad_out)`; when `grad_out` is non-null the callable must fill it with the gradient. `Result` reports the solution, objective value, gradient norm, iteration and evaluation counts, and a `Status`.

Box constraints are supported through `optimizer->setBounds(lower, upper)` (use `±infinity` entries for one-sided bounds). Bounded problems are handled by L-BFGS-B, which treats the bounds natively via gradient projection, so solutions may lie exactly on a bound; if bounds are set on the plain L-BFGS optimizer, the run is transparently delegated to L-BFGS-B.

## Available optimizers

| Name | Kind | Notes |
|------|------|-------|
| `L-BFGS` | local, gradient-based | limited-memory BFGS with Moré-Thuente line search; ported from [OptimLib](https://github.com/kthohr/optim) |
| `L-BFGS-B` | local, gradient-based, box constraints | limited-memory BFGS for bound-constrained problems (Byrd, Lu, Nocedal, Zhu) with Lewis-Overton line search; ported from [l-bfgs-b](https://github.com/droemer7/l-bfgs-b) |
| `AMPGO` | global | Adaptive Memory Programming for Global Optimization (Lasdon, Duarte, Glover, Laguna, Martí): local minimization alternated with tabu tunneling; ported from Andrea Gavana's Python implementation |
| `LIPO` | global, derivative-free, requires finite bounds | MaxLIPO+TR (Malherbe & Vayatis LIPO upper bound + trust-region refinement); ported from [dlib](http://dlib.net)'s `global_function_search` |
| `EGO` | global, derivative-free, requires finite bounds | Efficient Global Optimization (Jones, Schonlau, Welch): Bayesian optimization with a Kriging surrogate and expected-improvement infill; ported from [SMT](https://github.com/SMTorg/smt)'s `EGO` application and Kriging core |

Planned: further global optimizers.

### L-BFGS parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | 2000 | maximum number of iterations |
| `gradient_tolerance` | 1e-8 | stop when the L2 norm of the gradient falls below this value |
| `rel_solution_change_tolerance` | 1e-14 | stop when the relative change of the solution falls below this value |
| `memory` | 10 | number of past iterations kept for the Hessian approximation |
| `wolfe_constant_1` | 1e-3 | line search Armijo sufficient-decrease tolerance |
| `wolfe_constant_2` | 0.9 | line search curvature-condition tolerance |

When bounds are set, the run is delegated to L-BFGS-B; `max_iterations`, `gradient_tolerance` and `memory` carry over, while the Wolfe constants (specific to the Moré-Thuente line search) do not apply.

### L-BFGS-B parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | 2000 | maximum number of iterations |
| `max_function_evaluations` | 0 | maximum number of function evaluations (0 = unlimited) |
| `gradient_tolerance` | 1e-6 | stop when the infinity norm of the projected gradient falls below this value |
| `rel_objective_change_tolerance` | 1e-11 | stop when the relative change of the objective falls below this value |
| `rel_solution_change_tolerance` | 1e-11 | stop when the relative change of the solution falls below this value |
| `memory` | 10 | number of correction pairs kept for the Hessian approximation |
| `strong_wolfe` | false | enforce the strong Wolfe condition in the line search instead of the weak one |
| `line_search_max_iterations` | 25 | maximum number of line search iterations per step |

The default weak Wolfe line search also permits minimizing non-smooth objectives.

### AMPGO parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_iterations` | 20 | maximum number of global iterations (minimization phases) |
| `tunnel_iterations` | 5 | maximum number of tabu tunneling attempts per global iteration |
| `max_function_evaluations` | 0 | maximum number of function evaluations (0 = max(100, 10·n)) |
| `tolerance` | 1e-5 | stop when the best objective is within this distance of `target_objective` |
| `target_objective` | -inf | value of the global optimum, if known (-inf to disable) |
| `eps1` | 0.02 | constant defining the aspiration value during tunneling |
| `eps2` | 0.1 | perturbation factor for the tunneling start point |
| `tabu_list_size` | 5 | size of the circular tabu list |
| `tabu_strategy` | `farthest` | point to drop when the tabu list is full: `oldest` or `farthest` |
| `local_solver` | `L-BFGS-B` | local optimizer: `L-BFGS-B` or `L-BFGS` |
| `seed` | 0 | random seed for the tunneling perturbations (0 = non-deterministic) |

AMPGO needs gradients (for the local solver and the tunneling chain rule). For a gradient-free objective, wrap it with `globopt::withNumericalGradient<double>(f)`, which adds central finite differences.

### LIPO parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_function_evaluations` | 0 | maximum number of function evaluations (0 = max(100, 10·n)) |
| `target_objective` | -inf | value of the global optimum, if known (-inf to disable) |
| `tolerance` | 1e-5 | stop when the best objective is within this distance of `target_objective` |
| `pure_random_search_probability` | 0.02 | probability of ignoring the upper bound and sampling uniformly at random |
| `upper_bound_samples` | 5000 | Monte Carlo samples used to maximize the upper bound per iteration |
| `relative_noise_magnitude` | 0.001 | assumed relative noise in objective values when fitting the upper bound |
| `trust_region_epsilon` | 0 | minimum predicted improvement required to take a trust-region step |
| `upper_bound_solver_epsilon` | 1e-4 | accuracy of the QP solver that fits the upper bound |
| `seed` | 0 | random seed (0 = non-deterministic) |

LIPO is derivative-free (the objective is never asked for a gradient) and requires finite box bounds. It targets *expensive* objectives: each iteration spends considerable model-fitting work to squeeze the most out of every function evaluation, so use it with budgets of a few hundred evaluations rather than tens of thousands. Integer-variable support from the dlib original is not ported.

### EGO parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_function_evaluations` | 0 | maximum number of function evaluations, DOE included (0 = max(100, 10·n)) |
| `doe_size` | 0 | number of initial design-of-experiments samples (0 = max(5, 2·n)) |
| `criterion` | `EI` | infill criterion: `EI` (expected improvement), `SBO` (surrogate mean) or `LCB` (mean − 3σ) |
| `kernel` | `squar_exp` | correlation kernel: `squar_exp`, `abs_exp`, `matern32` or `matern52` |
| `regression` | `constant` | regression (trend) model: `constant`, `linear` or `quadratic` |
| `theta0` | 0.01 | initial correlation hyperparameter |
| `theta_bound_lower` | 1e-6 | lower bound for the correlation hyperparameters |
| `theta_bound_upper` | 20 | upper bound for the correlation hyperparameters |
| `nugget` | 100·ε | jitter added to the correlation diagonal for numerical stability |
| `noise` | 0 | fixed noise variance added to the correlation diagonal (normalized-y units) |
| `hyperparameter_starts` | 10 | multistart points for the likelihood maximization |
| `acquisition_starts` | 20 | multistart points for the acquisition maximization |
| `target_objective` | -inf | value of the global optimum, if known (-inf to disable) |
| `tolerance` | 1e-5 | stop when the best objective is within this distance of `target_objective` |
| `seed` | 0 | random seed (0 = non-deterministic) |

EGO is derivative-free and requires finite box bounds. It evaluates an initial Latin-hypercube design (the initial point passed to `run` is clamped into the box and used as the first sample), then repeatedly fits an anisotropic Kriging model — maximizing the reduced likelihood over the length-scale hyperparameters with multistart L-BFGS-B in log₁₀ space — and evaluates the point that maximizes the infill criterion. It is the strongest choice when every function evaluation is expensive: use budgets of tens to a few hundred evaluations (each iteration costs a Kriging fit, which grows cubically with the number of samples). The qEI parallel enrichment, mixed-integer support and noise estimation of the SMT original are not ported.

## Benchmarks

`include/globopt/benchmarks/go_benchmark.hpp` contains all 202 problems of the `go_benchmark.py` suite accompanying AMPGO (bounds, formulas and reference optima follow the Python source exactly, quirks included — e.g. its "Easom" is Ackley-shaped, and a few `fglob` values differ from the formula's true minimum). The `run_ampgo_benchmarks` executable runs AMPGO over the full set:

```sh
./build/run_ampgo_benchmarks [seed]
```

With the default seed, AMPGO solves 179 of 202 problems within a 20000-evaluation budget per problem (average ~3200 evaluations), and LIPO solves 146 of 202 within a 500-evaluation budget (median 41 evaluations on solved problems). The misses are plateau/stair-step functions (Corana, Gear, Mishra10), noisy objectives that draw random weights on every evaluation (Stochastic, XinSheYang01), and highly oscillatory landscapes (Bukin06, Easom, Griewank, Salomon, Schaffer01/02, Ripple01, SineEnvelope, Trefethen, Wavy, Weierstrass, Pathological, CrownedCross, Deceptive, DeVilliersGlasser02, Mishra04, XinSheYang03, Zimmerman) that are hard for any gradient-based tunneling method.

## Building the tests and examples

The library itself needs no build. Tests and examples use CMake:

```sh
cmake -B build -S .            # finds Eigen3 >= 3.4, or falls back to ../eigen
cmake --build build
ctest --test-dir build
```

Eigen >= 3.4 is required (L-BFGS-B uses Eigen's indexed views). If Eigen is not installed system-wide, point CMake at a source tree with `-DGLOBOPT_EIGEN_DIR=/path/to/eigen`.

## License

MPL-2.0. The L-BFGS implementation and the Moré-Thuente line search are ported from [OptimLib](https://github.com/kthohr/optim) (Apache License 2.0, Copyright Keith O'Hara); the L-BFGS-B implementation is ported from [l-bfgs-b](https://github.com/droemer7/l-bfgs-b) (MIT License, Copyright Dane Roemer); AMPGO and the benchmark suite are ported from Andrea Gavana's Python implementation (go_amp.py / go_benchmark.py); LIPO is ported from [dlib](http://dlib.net) (Boost Software License, Copyright Davis E. King); EGO and its Kriging model are ported from [SMT](https://github.com/SMTorg/smt) (New BSD License, Copyright the SMT developers); attribution is retained in the corresponding headers.
