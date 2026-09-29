#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "K_method.h"

// Define mathematical Pi if not already defined
#ifndef M_PI
    #define M_PI 3.14159265358979323846f
#endif

int main() {
    // Configuration parameters
    constexpr float fs = 48000.f;       // Sample rate matching constants.h
    constexpr float freq = 1000.f;      // Sine wave frequency (1 kHz)
    constexpr float duration = 1.0f;    // Duration in seconds
    size_t num_samples = static_cast<size_t>(fs * duration);

    std::cout << "Generating " << duration << " second(s) of a " << freq
              << " Hz sine wave at " << fs << " Hz...\n";

    // 1. Generate the base input sine wave vector
    std::vector<float> input_signal(num_samples);
    for(size_t i = 0; i < num_samples; ++i) {
        float t = static_cast<float>(i) / fs;
        input_signal[i] = std::sin(2.0f * static_cast<float>(M_PI) * freq * t);
    }

    // Create independent copies for both processing methods
    std::vector<float> standard_output = input_signal;
    std::vector<float> adaa_output = input_signal;

    // 2. Instantiate the K_method processor (reads char.txt and builds lookup
    // tables)
    std::cout << "Initializing K_method processor and loading char.txt...\n";
    K_method k_processor(fs);

    // 3. Run Standard Processing
    std::cout << "Running process_Standard()...\n";
    k_processor.process_Standard(standard_output);

    // 4. Run ADAA Processing
    std::cout << "Running process_ADAA()...\n";
    k_processor.process_ADAA(adaa_output);

    // 5. Output a few results to verify processing happened
    std::cout << "\n--- Processing Complete ---" << std::endl;
    std::cout << "First 5 samples of Standard output: ";
    for(int i = 0; i < 5; ++i) {
        std::cout << standard_output[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "First 5 samples of ADAA output:     ";
    for(int i = 0; i < 5; ++i) {
        std::cout << adaa_output[i] << " ";
    }
    std::cout << std::endl;

    auto std_metrics = k_processor.getStandardMetrics();
    auto adaa_metrics = k_processor.getADAAMetrics();

    std::cout
        << "\n================ PAPI PERFORMANCE COMPARISON ================\n";
    std::cout << std::setw(20) << "Metric" << " | " << std::setw(18)
              << "Standard Mode" << " | " << std::setw(18) << "ADAA Mode"
              << "\n";
    std::cout
        << "-------------------------------------------------------------\n";
    std::cout << std::setw(20) << "FP Instructions" << " | " << std::setw(18)
              << std_metrics.fp_instructions << " | " << std::setw(18)
              << adaa_metrics.fp_instructions << "\n";
    std::cout << std::setw(20) << "Total Cycles" << "    | " << std::setw(18)
              << std_metrics.total_cycles << " | " << std::setw(18)
              << adaa_metrics.total_cycles << "\n";
    std::cout
        << "=============================================================\n";

    return 0;

    return 0;
}