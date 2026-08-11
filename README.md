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
| `CMA-ES` | global, derivative-free, bounds optional | Covariance Matrix Adaptation Evolution Strategy (Hansen & Ostermeier) with IPOP restarts (Auger & Hansen); implemented from the published algorithm |

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
| `hyperparameter_refit_interval` | 1 | re-run the likelihood maximization every k infill iterations (1 = every iteration, as in SMT) |
| `target_objective` | -inf | value of the global optimum, if known (-inf to disable) |
| `tolerance` | 1e-5 | stop when the best objective is within this distance of `target_objective` |
| `seed` | 0 | random seed (0 = non-deterministic) |

EGO is derivative-free and requires finite box bounds. It evaluates an initial Latin-hypercube design (the initial point passed to `run` is clamped into the box and used as the first sample), then repeatedly fits an anisotropic Kriging model — maximizing the reduced likelihood over the length-scale hyperparameters with multistart L-BFGS-B in log₁₀ space — and evaluates the point that maximizes the infill criterion. It is the strongest choice when every function evaluation is expensive: use budgets of tens to a few hundred evaluations (each iteration costs a Kriging fit, which grows cubically with the number of samples). The qEI parallel enrichment, mixed-integer support and noise estimation of the SMT original are not ported.

The hyperparameter search dominates EGO's own runtime — over 95% of it on the benchmark suite, since every likelihood evaluation factorizes the correlation matrix of all samples so far. `hyperparameter_refit_interval` trades some of that away: with interval *k* the length scales are re-estimated only every *k*-th infill iteration and the surrogate is merely re-conditioned on the new sample in between, which cuts EGO's overhead almost exactly by a factor of *k*. The default 1 reproduces SMT's every-iteration behaviour; values of 5–10 are a good trade when the objective is cheap enough that EGO's own cost is what limits you.

### CMA-ES parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_function_evaluations` | 0 | maximum number of function evaluations (0 = max(1000, 200·n)) |
| `population_size` | 0 | population size λ (0 = 4 + ⌊3·ln n⌋) |
| `initial_step_size` | 0 | initial step size σ in normalized coordinates (0 = 0.3, about a third of the box) |
| `max_restarts` | 9 | number of IPOP restarts after a converged run (0 = a single run) |
| `population_increase_factor` | 2 | factor applied to the population size on each restart (IPOP: 2) |
| `tolerance_function` | 1e-12 | restart when the objective range within a generation and over recent history falls below this |
| `tolerance_x` | 1e-12 | restart when the search distribution collapses below this in every coordinate |
| `max_condition_number` | 1e14 | restart when the covariance matrix becomes worse conditioned than this |
| `boundary_penalty` | 1 | weight of the quadratic penalty outside the box, relative to the spread of objective values |
| `target_objective` | -inf | value of the global optimum, if known (-inf to disable) |
| `tolerance` | 1e-5 | stop when the best objective is within this distance of `target_objective` |
| `seed` | 0 | random seed (0 = non-deterministic) |

CMA-ES is derivative-free and, unlike LIPO and EGO, does not require bounds — it is the right default when the objective is cheap and no gradient is available. It samples each generation from a multivariate normal whose covariance is adapted to the local landscape, which is what lets it handle ill-conditioned and non-separable problems that defeat coordinate-wise methods: a 10-D ellipsoid with condition number 10⁶ is solved in about 12000 evaluations.

Plain CMA-ES is a *local* search, so the IPOP restart scheme (restart on convergence with a doubled population from a fresh random mean) is enabled by default and is what makes it global; set `max_restarts` to 0 for a single run. When bounds are set the search runs in coordinates that map the box to [0, 1]ⁿ, so an anisotropic box needs no manual rescaling; points outside the box are evaluated at their clamped position and ranked with a quadratic penalty, and the reported solution is always feasible. The active (negative-weight) covariance update and separable/diagonal variants are not implemented.

## Benchmarks

`include/globopt/benchmarks/go_benchmark.hpp` contains all 202 problems of the `go_benchmark.py` suite accompanying AMPGO (bounds, formulas and reference optima follow the Python source exactly, quirks included — e.g. its "Easom" is Ackley-shaped, and a few `fglob` values differ from the formula's true minimum). Four problems (`Keane`, `CosineMixture`, `Holzman`, `SchmidtVetters`) state an `fglob` that the objective beats immediately — `Keane` is a maximization problem in the Python suite — so every optimizer scores a free "solved" on them in one to three evaluations; the counts below include those four. The `run_ampgo_benchmarks` executable runs AMPGO over the full set:

```sh
./build/run_ampgo_benchmarks [seed] [optimizer] [options]
```

`optimizer` is `AMPGO` (default), `LIPO` or `EGO`; each has its own default budget. The options select a subset of the suite, which keeps the expensive optimizers testable without committing to a full sweep:

| Option | Description |
|--------|-------------|
| `--budget N` | evaluations per problem, overriding the optimizer's default |
| `--filter SUBSTR` | only problems whose name contains `SUBSTR` (case-insensitive) |
| `--max-dims N` | skip problems with more than `N` variables |
| `--stride N` | take every `N`-th problem (a subset spread over the suite) |
| `--limit N` | stop after `N` problems |
| `--repeats N` | run each problem with `N` consecutive seeds |
| `--refit-interval N` | EGO: re-estimate hyperparameters every `N` iterations |
| `--scalable DIMS` | run the scalable DIRECTGOLib suite at the given comma-separated dimensions (e.g. `--scalable 2,5,10,20`) instead of the fixed-size suite |
| `--list` | print the selected problems and exit |

Each run reports its wall-clock time, and the summary counts both problems solved to the strict 1e-6 target and runs that land within a 1% relative gap of `fglob` — the latter matters for EGO, which is built to get close in very few evaluations rather than to converge to the last digit.

```sh
./build/run_ampgo_benchmarks 12345 EGO --stride 20   # ~11 problems, a couple of minutes
```

### Scalable problems

`include/globopt/benchmarks/directgo_benchmark.hpp` adds 30 problems ported from [DIRECTGOLib](https://github.com/blockchain-group/DIRECTGOLib) v2.0 (MIT). Unlike the `go_benchmark` suite — which is 156/202 two-dimensional — these are defined for *any* number of variables, which is what makes it possible to measure how an optimizer degrades with dimension:

```sh
./build/run_ampgo_benchmarks 12345 CMA-ES --scalable 2,5,10,20,40
```

Each problem exposes bounds, the global minimum and a minimizer as functions of `n` (`ScalableProblem::at(n)` materializes an ordinary `Problem`). The test suite checks that every stated optimum really attains its stated `fglob` at several dimensions — the check that the fixed-size suite fails on four problems.

Problems solved (of 30) at the default seed, per optimizer and dimension:

| Optimizer | budget | n=2 | n=5 | n=10 | n=20 | n=40 |
|-----------|-------:|----:|----:|-----:|-----:|-----:|
| `CMA-ES` | 20000 | 28 | 25 | 25 | 24 | 19 |
| `AMPGO` (with gradients) | 20000 | 28 | 23 | 23 | 23 | — |
| `LIPO` | 500 | 26 | 13 | 5 | 4 | — |
| `EGO` | 200 | 4 | 0 | 0 | — | — |

Read the budgets before the columns: LIPO's 500 and EGO's 200 evaluations are deliberate design points — both target objectives too expensive to sample 20000 times — so the table shows how each method behaves *in its own regime*, not a like-for-like race. EGO's row in particular is close to meaningless as a ranking: 200 evaluations spread over 10 dimensions is roughly 20 per dimension, and the strict 1e-6 target asks for a last-digit convergence it never attempts. What it does show cleanly is the shape of the degradation: CMA-ES loses ground slowly and its average cost grows from ~1800 to ~12400 evaluations, while the Lipschitz upper bound LIPO fits becomes uninformative in higher dimensions. The problems CMA-ES does not solve are the deceptive multimodal ones — Schwefel, Rastrigin, Salomon and Dixon-Price.

With the default seed, AMPGO solves 179 of 202 problems within a 20000-evaluation budget per problem (average ~3200 evaluations), and LIPO solves 146 of 202 within a 500-evaluation budget (median 41 evaluations on solved problems). CMA-ES solves 182 of 202 within a 20000-evaluation budget (average ~3100 evaluations, 0.3 s for the whole sweep) — the best result here, and without needing gradients. EGO solves 39 of 202 within a 200-evaluation budget (median 24 evaluations on solved problems) and lands within a 1% relative gap of `fglob` on 89 of 202 — the fairer figure at a budget 2.5× smaller than LIPO's and 100× smaller than AMPGO's, since expected-improvement search closes in on the basin quickly but has no mechanism for polishing the last digits. Its successes are concentrated in low dimension (15 of 18 one-dimensional and 20 of 156 two-dimensional problems, none above four), and it is markedly seed-sensitive: over five seeds, Ackley ranges from 0.44 to 3.05. The full EGO sweep takes ~2.5 h; `--refit-interval 5` brings that down to ~35 min and, on this suite, costs nothing measurable (better on 94 problems, worse on 74 — within seed noise). The misses are plateau/stair-step functions (Corana, Gear, Mishra10), noisy objectives that draw random weights on every evaluation (Stochastic, XinSheYang01), and highly oscillatory landscapes (Bukin06, Easom, Griewank, Salomon, Schaffer01/02, Ripple01, SineEnvelope, Trefethen, Wavy, Weierstrass, Pathological, CrownedCross, Deceptive, DeVilliersGlasser02, Mishra04, XinSheYang03, Zimmerman) that are hard for any gradient-based tunneling method.

## Building the tests and examples

The library itself needs no build. Tests and examples use CMake:

```sh
cmake -B build -S .            # finds Eigen3 >= 3.4, or falls back to ../eigen
cmake --build build
ctest --test-dir build
```

Build with optimization. This code is Eigen-heavy, and an `-O0` build runs the EGO benchmark sweep roughly 37× slower than `-O3` (Eigen's assertions are active as well), which is the difference between a half-hour sweep and an overnight one. Single-config generators therefore default to `CMAKE_BUILD_TYPE=Release` here; pass `-DCMAKE_BUILD_TYPE=Debug` explicitly when you actually want to debug.

Eigen >= 3.4 is required (L-BFGS-B uses Eigen's indexed views). If Eigen is not installed system-wide, point CMake at a source tree with `-DGLOBOPT_EIGEN_DIR=/path/to/eigen`.

## License

MPL-2.0. The L-BFGS implementation and the Moré-Thuente line search are ported from [OptimLib](https://github.com/kthohr/optim) (Apache License 2.0, Copyright Keith O'Hara); the L-BFGS-B implementation is ported from [l-bfgs-b](https://github.com/droemer7/l-bfgs-b) (MIT License, Copyright Dane Roemer); AMPGO and the benchmark suite are ported from Andrea Gavana's Python implementation (go_amp.py / go_benchmark.py); LIPO is ported from [dlib](http://dlib.net) (Boost Software License, Copyright Davis E. King); EGO and its Kriging model are ported from [SMT](https://github.com/SMTorg/smt) (New BSD License, Copyright the SMT developers); the scalable benchmark problems are ported from [DIRECTGOLib](https://github.com/blockchain-group/DIRECTGOLib) (MIT License, Copyright the Blockchain Technologies Group); attribution is retained in the corresponding headers. CMA-ES is not a port: it is implemented from the published algorithm description.
