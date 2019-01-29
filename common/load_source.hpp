
#pragma once

#include <fstream>
#include <stdexcept>
#include <string>

inline std::string load_source(std::string const input) noexcept(false)
{
    std::ifstream stream(input.c_str());
    if (!stream.is_open())
    {
        throw std::domain_error("Cannot open file: " + input);
    }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}
