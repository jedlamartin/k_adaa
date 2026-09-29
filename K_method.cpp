#include "K_method.h"

#include <papi.h>

#include <fstream>
#include <string>

static float closestElementLinearInterpolate(const float x,
                                             const std::vector<float>& xArray,
                                             const std::vector<float>& yArray) {
    size_t size = xArray.size();
    if(x < xArray[0]) {
        return yArray[0];
    } else if(x > xArray[size - 1]) {
        return yArray[size - 1];
    }
    int i = 0, j = static_cast<int>(size), mid = 0;
    while(i < j) {
        mid = (i + j) / 2;

        if(xArray[mid] == x) {
            return yArray[mid];
        }

        if(x < xArray[mid]) {
            if(mid > 0 && x > xArray[mid - 1]) {
                float div = xArray[mid] - xArray[mid - 1];
                float mull = (xArray[mid] - x) / div;
                float mulr = (x - xArray[mid - 1]) / div;
                return yArray[mid - 1] * mull + yArray[mid] * mulr;
            }
            j = mid;
        } else {
            if(mid < size - 1 && x < xArray[mid + 1]) {
                float div = xArray[mid + 1] - xArray[mid];
                float mull = (xArray[mid + 1] - x) / div;
                float mulr = (x - xArray[mid]) / div;
                return yArray[mid] * mull + yArray[mid + 1] * mulr;
            }
            i = mid + 1;
        }
    }
    return yArray[mid];
}

K_method::K_method(float fs) {
    oversampler_standard.configure(fs, 1, dur * fs);
    oversampler_adaa.configure(fs, 1, dur * fs);

    float h = 2 * fs * upsample;
    Matrix<float, 2, 2> A_inv =
        (h * Matrix<float, 2, 2>::Identity() - A).inverse();
    G = A_inv * C;
    H = A_inv * (h * Matrix<float, 2, 2>::Identity() + A);
    J = A_inv * B;
    K = D * G;

    std::ifstream file("char.txt");
    if(file.is_open()) {
        std::string line;
        while(std::getline(file, line)) {
            if(line.empty()) continue;
            size_t tab_pos = line.find('\t');
            if(tab_pos != std::string::npos) {
                float x_val = std::stof(line.substr(0, tab_pos));
                float y_val = std::stof(line.substr(tab_pos + 1));

                float p_val = x_val - K * y_val;

                p.push_back(p_val);
                y.push_back(y_val);

                size_t idx = p.size() - 1;
                if(idx == 0) {
                    this->Y.push_back(0.0f);
                } else {
                    float dp = this->p[idx] - this->p[idx - 1];
                    float avg_y = 0.5f * (this->y[idx] + this->y[idx - 1]);
                    float next_Y = this->Y.back() + avg_y * dp;
                    this->Y.push_back(next_Y);
                }
            }
        }
        file.close();
    } else {
        throw std::runtime_error("Fatal Error: Could not open char.txt!");
    }
}

void K_method::process_Standard(std::vector<float>& samples) {
    int EventSet = PAPI_NULL;
    long long values[2] = {0, 0};

    if(PAPI_create_eventset(&EventSet) == PAPI_OK) {
        PAPI_add_event(EventSet, PAPI_TOT_INS);
        PAPI_add_event(EventSet, PAPI_TOT_CYC);
        PAPI_start(EventSet);
    }

    oversampler_standard.interpolate(samples);
    oversampler_standard.processEach([&](std::vector<float>& upsampled) {
        Matrix<float, 2, 1> w = Matrix<float, 2, 1>::Zero();
        Matrix<float, 2, 1> pk = Matrix<float, 2, 1>::Zero();
        float p, y, uBuffer;
        uBuffer = y = 0.f;
        for(size_t i = 0; i < upsampled.size(); ++i) {
            pk = H * w + J * (upsampled[i] + uBuffer) + G * y;
            p = D * pk;
            y = closestElementLinearInterpolate(p, this->p, this->y);
            w = G * y + pk;
            uBuffer = upsampled[i];
            upsampled[i] = w(1, 0) + upsampled[i];
        }
    });
    oversampler_standard.decimate(samples);

    if(PAPI_stop(EventSet, values) == PAPI_OK) {
        standard_metrics.fp_instructions = values[0];
        standard_metrics.total_cycles = values[1];
    }
    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
}

void K_method::process_ADAA(std::vector<float>& samples) {
    int EventSet = PAPI_NULL;
    long long values[2] = {0, 0};

    if(PAPI_create_eventset(&EventSet) == PAPI_OK) {
        PAPI_add_event(EventSet, PAPI_TOT_INS);
        PAPI_add_event(EventSet, PAPI_TOT_CYC);
        PAPI_start(EventSet);
    }

    oversampler_adaa.interpolate(samples);
    oversampler_adaa.processEach([&](std::vector<float>& upsampled) {
        Matrix<float, 2, 1> w = Matrix<float, 2, 1>::Zero();
        Matrix<float, 2, 1> pk = Matrix<float, 2, 1>::Zero();
        Matrix<float, 2, 1> pkBuffer = Matrix<float, 2, 1>::Zero();
        float p, pBuffer, y;
        std::array<float, 2> uBuffer {0.f};
        uBuffer[0] = uBuffer[1] = pBuffer = y = 0.f;
        for(size_t i = 0; i < upsampled.size(); ++i) {
            pk = H * w + J * (upsampled[i] + 0.5f * (uBuffer[0] + uBuffer[1])) +
                 G * y;
            p = D * pk;
            if(p == pBuffer) {
                float y1 = closestElementLinearInterpolate(p, this->p, this->y);
                float y2 =
                    closestElementLinearInterpolate(pBuffer, this->p, this->y);
                y = (y1 + y2) / 2;
            } else {
                float y1 = closestElementLinearInterpolate(p, this->p, this->Y);
                float y2 =
                    closestElementLinearInterpolate(pBuffer, this->p, this->Y);
                y = (y1 - y2) / (p - pBuffer);
            }

            w = G * y + 0.5f * (pk + pkBuffer);
            uBuffer[1] = uBuffer[0];
            uBuffer[0] = upsampled[i];
            pBuffer = p;
            pkBuffer = pk;
            upsampled[i] = w(1, 0) + upsampled[i];
        }
    });

    oversampler_adaa.decimate(samples);

    if(PAPI_stop(EventSet, values) == PAPI_OK) {
        adaa_metrics.fp_instructions = values[0];
        adaa_metrics.total_cycles = values[1];
    }
    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
}