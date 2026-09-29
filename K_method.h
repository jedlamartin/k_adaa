#ifndef K_METHOD_H
#define K_METHOD_H

#include <array>
#include <vector>

#include "ReSinc.hpp"
#include "constants.h"

class K_method {
public:
    K_method(float fs);
    void process_Standard(std::vector<float>& samples);
    void process_ADAA(std::vector<float>& samples);

    struct PAPIMetrics {
        long long sp_instructions = 0;
        long long total_instructions = 0;
        long long total_cycles = 0;
    };

    PAPIMetrics getStandardMetrics() const { return standard_metrics; }
    PAPIMetrics getADAAMetrics() const { return adaa_metrics; }

private:
    Oversampler<float, 4, 256> oversampler_standard;
    Oversampler<float, 4, 256> oversampler_adaa;

    Matrix<float, 2, 1> G;
    Matrix<float, 2, 2> H;
    Matrix<float, 2, 1> J;
    float K;

    std::vector<float> p;
    std::vector<float> y;
    std::vector<float> Y;

    PAPIMetrics standard_metrics;
    PAPIMetrics adaa_metrics;
};

#endif    // K_METHOD_H