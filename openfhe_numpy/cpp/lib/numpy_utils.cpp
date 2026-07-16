//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2014-2025, NJIT, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================
#include "numpy_utils.h"
#include "utils/exception.h"

#include <cmath>

void RoundVector(std::vector<double>& vector) {
    for (double& e : vector)
        e = std::round(e);
}

uint32_t NextPow2(uint32_t x) {
    return std::pow(2, std::ceil(std::log(x) / std::log(2.0)));
};


/*
Compute diagonals for the permutation matrix Sigma.
B[i,j] = A[i, i +j]
*/
std::vector<double> GenSigmaDiag(size_t slots, size_t numCols, int32_t k) {
    if (numCols == 0) {
        OPENFHE_THROW("numCols must be positive");
    }

    const int32_t d = static_cast<int32_t>(numCols);
    const size_t n = numCols * numCols;

    if (slots < n || slots % n != 0) {
        OPENFHE_THROW("slots must be a multiple of numCols * numCols");
    }

    std::vector<double> diag(slots, 0.0);

    for (size_t t = 0; t < slots / n; ++t) {
        const size_t base = t * n;

        if (k >= 0) {
            for (int32_t i = 0; i < static_cast<int32_t>(n); ++i) {
                int32_t tmp = i - d * k;
                if ((0 <= tmp) && (tmp < d - k)) {
                    diag[base + i] = 1.0;
                }
            }
        } else {
            for (int32_t i = 0; i < static_cast<int32_t>(n); ++i) {
                int32_t tmp = i - (d + k) * d;
                if ((-k <= tmp) && (tmp < d)) {
                    diag[base + i] = 1.0;
                }
            }
        }
    }

    return diag;
}

/*
Compute diagonals  for the permutation matrix Tau.
B[i,j] = A[i + j,i]
u_[d.k][k + d*i] = 1 for all 0 <= i < d
*/

std::vector<double> GenTauDiag(size_t totalSlots, size_t numCols, int32_t k) {
    if (numCols == 0) {
        OPENFHE_THROW("numCols must be positive");
    }

    const size_t n = numCols * numCols;

    if (totalSlots < n || totalSlots % n != 0) {
        OPENFHE_THROW("slots must be a multiple of numCols * numCols");
    }

    if (k < 0 || static_cast<size_t>(k) >= numCols) {
        OPENFHE_THROW("Tau diagonal index k is out of range");
    }

    std::vector<double> diag(totalSlots, 0.0);

    for (size_t t = 0; t < totalSlots / n; ++t) {
        const size_t base = t * n;
        for (size_t i = 0; i < numCols; ++i) {
            diag[base + static_cast<size_t>(k) + numCols * i] = 1.0;
        }
    }

    return diag;
}

/**
 *Compute diagonals for the permutation matrix Phi (V).
 *B[i,j] = A[i,j+1]
 *There are two diagonals in the matrix Phi.
 *Type = 0 correspond for the k-th diagonal, and type = 1 is for the (k-d)-th
 *diagonal
 */
std::vector<double> GenPhiDiag(size_t slots, size_t numCols, int32_t k, int type) {
    if (numCols == 0) {
        OPENFHE_THROW("numCols must be positive");
    }

    const size_t d = numCols;
    const size_t n = d * d;

    if (slots < n || slots % n != 0) {
        OPENFHE_THROW("slots must be a multiple of numCols * numCols");
    }

    std::vector<double> diag(slots, 0.0);

    for (size_t t = 0; t < slots / n; ++t) {
        const size_t base = t * n;

        if (type == 0) {
            for (size_t i = 0; i < n; ++i) {
                if ((i % d) < d - static_cast<size_t>(k)) {
                    diag[base + i] = 1.0;
                }
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                if ((i % d) >= d - static_cast<size_t>(k)) {
                    diag[base + i] = 1.0;
                }
            }
        }
    }

    return diag;
}

/**
 *Compute diagonals for the permutation Psi (W).
 *B[i,j] = A[i+1,j]
 */
std::vector<double> GenPsiDiag(size_t slots, size_t numCols) {
    std::vector<double> diag(slots, 0.0);  // all zeros
    std::fill(diag.begin(),  diag.begin()+ numCols * numCols, 1);
    return diag;
}

std::vector<double> GenTransposeDiag(size_t totalSlots, size_t numCols, int32_t i) {
    if (static_cast<int32_t>(numCols) < i)
        OPENFHE_THROW("numCols cannot be less than the index");

    size_t start = 0;
    size_t max   = 0;
    if (i < 0) {
        start = -i;
        max   = numCols;
    }
    else {
        max = numCols - i;
    }

    size_t n = numCols * numCols;
    std::vector<double> diag(totalSlots, 0);
    for (size_t t = 0; t < totalSlots / n; ++t) {
        for (size_t j = start; j < max; j++) {
            size_t idx = t * n + (numCols + 1) * j + i;
            if (idx < totalSlots)
                diag[idx] = 1;
        }
    }
    return diag;
}
