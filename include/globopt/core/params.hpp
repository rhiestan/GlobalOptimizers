// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef GLOBOPT_CORE_PARAMS_HPP
#define GLOBOPT_CORE_PARAMS_HPP

#include <cstddef>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace globopt {

/// Value type accepted by Optimizer::setParam.
using ParamValue = std::variant<bool, long long, double, std::string>;

/// Name/description pair returned by Optimizer::listParams.
struct ParamInfo {
    std::string name;
    std::string description;
};

namespace detail {

template <typename T>
struct is_param_integral
    : std::integral_constant<bool, std::is_integral<T>::value && !std::is_same<T, bool>::value> {};

/// Maps parameter names to member variables of an optimizer, so that
/// parameters can be set through a uniform string-based interface.
class ParamRegistry {
public:
    template <typename T>
    void add(const std::string& name, T* target, const std::string& description)
    {
        Entry entry;
        entry.description = description;

        entry.setter = [name, target](const ParamValue& value) {
            assign(name, *target, value);
        };
        entry.getter = [target]() -> ParamValue {
            return toParamValue(*target);
        };

        m_order.push_back(name);
        m_entries[name] = std::move(entry);
    }

    void set(const std::string& name, const ParamValue& value)
    {
        auto it = m_entries.find(name);
        if (it == m_entries.end()) {
            throw std::invalid_argument("globopt: unknown parameter '" + name + "'");
        }
        it->second.setter(value);
    }

    ParamValue get(const std::string& name) const
    {
        auto it = m_entries.find(name);
        if (it == m_entries.end()) {
            throw std::invalid_argument("globopt: unknown parameter '" + name + "'");
        }
        return it->second.getter();
    }

    bool contains(const std::string& name) const
    {
        return m_entries.count(name) != 0;
    }

    std::vector<ParamInfo> list() const
    {
        std::vector<ParamInfo> result;
        result.reserve(m_order.size());
        for (const auto& name : m_order) {
            result.push_back({name, m_entries.at(name).description});
        }
        return result;
    }

private:
    struct Entry {
        std::function<void(const ParamValue&)> setter;
        std::function<ParamValue()> getter;
        std::string description;
    };

    template <typename T>
    static typename std::enable_if<std::is_floating_point<T>::value>::type
    assign(const std::string& name, T& target, const ParamValue& value)
    {
        if (const auto* d = std::get_if<double>(&value)) {
            target = static_cast<T>(*d);
        } else if (const auto* i = std::get_if<long long>(&value)) {
            target = static_cast<T>(*i);
        } else {
            throw std::invalid_argument("globopt: parameter '" + name + "' expects a numeric value");
        }
    }

    template <typename T>
    static typename std::enable_if<is_param_integral<T>::value>::type
    assign(const std::string& name, T& target, const ParamValue& value)
    {
        if (const auto* i = std::get_if<long long>(&value)) {
            target = static_cast<T>(*i);
        } else {
            throw std::invalid_argument("globopt: parameter '" + name + "' expects an integer value");
        }
    }

    static void assign(const std::string& name, bool& target, const ParamValue& value)
    {
        if (const auto* b = std::get_if<bool>(&value)) {
            target = *b;
        } else {
            throw std::invalid_argument("globopt: parameter '" + name + "' expects a boolean value");
        }
    }

    static void assign(const std::string& name, std::string& target, const ParamValue& value)
    {
        if (const auto* s = std::get_if<std::string>(&value)) {
            target = *s;
        } else {
            throw std::invalid_argument("globopt: parameter '" + name + "' expects a string value");
        }
    }

    template <typename T>
    static typename std::enable_if<std::is_floating_point<T>::value, ParamValue>::type
    toParamValue(const T& value)
    {
        return static_cast<double>(value);
    }

    template <typename T>
    static typename std::enable_if<is_param_integral<T>::value, ParamValue>::type
    toParamValue(const T& value)
    {
        return static_cast<long long>(value);
    }

    static ParamValue toParamValue(const bool& value) { return value; }
    static ParamValue toParamValue(const std::string& value) { return value; }

    std::map<std::string, Entry> m_entries;
    std::vector<std::string> m_order;
};

} // namespace detail

} // namespace globopt

#endif
