// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef GLOBOPT_FACTORY_HPP
#define GLOBOPT_FACTORY_HPP

#include "core/optimizer.hpp"
#include "global/ampgo.hpp"
#include "local/lbfgs.hpp"
#include "local/lbfgsb.hpp"

#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace globopt {

enum class Algorithm {
    LBFGS,
    LBFGSB,
    AMPGO
};

template <typename Scalar = double>
class OptimizerFactory {
public:
    static std::unique_ptr<Optimizer<Scalar>> create(Algorithm algorithm)
    {
        switch (algorithm) {
            case Algorithm::LBFGS:
                return std::make_unique<LBFGS<Scalar>>();
            case Algorithm::LBFGSB:
                return std::make_unique<LBFGSB<Scalar>>();
            case Algorithm::AMPGO:
                return std::make_unique<globopt::AMPGO<Scalar>>();
        }
        throw std::invalid_argument("globopt: unknown algorithm");
    }

    /// Create an optimizer by name (case-insensitive; '-' and '_' are ignored,
    /// so "L-BFGS", "lbfgs" and "l_bfgs" are all accepted).
    static std::unique_ptr<Optimizer<Scalar>> create(const std::string& name)
    {
        const std::string key = normalize(name);

        if (key == "lbfgs") {
            return create(Algorithm::LBFGS);
        }
        if (key == "lbfgsb") {
            return create(Algorithm::LBFGSB);
        }
        if (key == "ampgo") {
            return create(Algorithm::AMPGO);
        }

        throw std::invalid_argument("globopt: unknown optimizer '" + name + "'");
    }

    static std::vector<std::string> available()
    {
        return {"L-BFGS", "L-BFGS-B", "AMPGO"};
    }

private:
    static std::string normalize(const std::string& name)
    {
        std::string result;
        result.reserve(name.size());
        for (const char c : name) {
            if (c == '-' || c == '_' || c == ' ') {
                continue;
            }
            result.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }
        return result;
    }
};

} // namespace globopt

#endif
