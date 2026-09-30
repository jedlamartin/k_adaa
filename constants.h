#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <eigen3/Eigen/Dense>

using Eigen::Matrix;

constexpr float fs = 48.e3;
constexpr size_t upsample_standard = 8;
constexpr size_t upsample_adaa = 2;
constexpr float dur = 1.f;
constexpr float R1 = 4700.f;      // 4.7k
constexpr float R2 = 51000.f;     // 51k
constexpr float C1 = 47.e-6;      // 47uF
constexpr float C2 = 51.e-12;     // 51pF
constexpr float Is0 = 2.52e-9;    // 2.52nA
constexpr float UT = 26.e-3;      // 26mV

// State-space matrices
// w' = Aw + Bu + Cy
// x  = Dw + Eu + Fy
// y  = f(x)

// A = [-1/(R1*C1) 0; -1/(R1*C2) -1/(R2*C2)];
// B = [1/(R1*C1); 1/(R1*C2)];
// C = [0; -1/C2];
// D = [0 1];
inline const Matrix<float, 2, 2> A {{-1.f / (R1 * C1), 0.f},
                                    {-1.f / (R1 * C2), -1.f / (R2 * C2)}};
inline const Matrix<float, 2, 1> B {{1.f / (R1 * C1)}, {1.f / (R1 * C2)}};
inline const Matrix<float, 2, 1> C {{0.f}, {-1.f / C2}};
inline const Matrix<float, 1, 2> D {{0.f, 1.f}};

#endif    // CONSTANTS_H