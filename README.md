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

Planned: AMPGO (Adaptive Memory Programming for Global Optimization), LIPO, and further global optimizers.

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

## Building the tests and examples

The library itself needs no build. Tests and examples use CMake:

```sh
cmake -B build -S .            # finds Eigen3 >= 3.4, or falls back to ../eigen
cmake --build build
ctest --test-dir build
```

Eigen >= 3.4 is required (L-BFGS-B uses Eigen's indexed views). If Eigen is not installed system-wide, point CMake at a source tree with `-DGLOBOPT_EIGEN_DIR=/path/to/eigen`.

## License

MPL-2.0. The L-BFGS implementation and the Moré-Thuente line search are ported from [OptimLib](https://github.com/kthohr/optim) (Apache License 2.0, Copyright Keith O'Hara); the L-BFGS-B implementation is ported from [l-bfgs-b](https://github.com/droemer7/l-bfgs-b) (MIT License, Copyright Dane Roemer); attribution is retained in the corresponding headers.
