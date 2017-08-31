#include <iostream>
#include <stdint.h>
#include <stdio.h>

extern "C" void cross(const float* waveforms, size_t n_pix, size_t n_samples,
    float* cleaned, const float* ref, size_t nref, int cen)
{
    for (int p=0; p<n_pix; ++p) {
        for (int j=0; j<n_samples; ++j) {
            float sum = 0.0;
            for (int i=0; i<nref; ++i) {
                int ii = j + i - cen;
                if (ii>=0 && ii<n_samples) {
                    float v = waveforms[p * n_samples + ii];
                    sum += v * ref[i];
                }
            }
            cleaned[p * n_samples + j] = sum;
        }
    }
}