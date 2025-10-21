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
std::vector<double> GenSigmaDiag(uint32_t numCols, int32_t k) {
    // the cast below is necessary as we don't want to mix signed and unsigned integers in calculations
    int32_t nCols = static_cast<int32_t>(numCols);
    int32_t n = nCols * nCols;
    std::vector<double> diag(n, 0);

    if (k >= 0) {
        for (int32_t i = 0; i < n; i++) {
            int32_t tmp = i - nCols * k;
            if ((0 <= tmp) && (tmp < nCols - k)) {
                diag[i] = 1;
            }
        }
    }
    else {
        for (int32_t i = 0; i < n; i++) {
            int32_t tmp = i - (nCols + k) * nCols;
            if ((-k <= tmp) && (tmp < nCols)) {
                diag[i] = 1;
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

std::vector<double> GenTauDiag(uint32_t totalSlots, uint32_t numCols, int32_t k) {
    uint32_t n = numCols * numCols;
    std::vector<double> diag(totalSlots, 0);

    for (uint32_t t = 0; t < totalSlots / n; t++) {
        for (uint32_t i = 0; i < numCols; i++) {
            diag[(t * n) + k + numCols * i] = 1;
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
std::vector<double> GenPhiDiag(uint32_t numCols, int32_t k, int type) {
    uint32_t n = numCols * numCols;
    std::vector<double> diag(n, 0);

    if (type == 0) {
        for (uint32_t i = 0; i < n; i++)
            if ((i % numCols >= 0) && ((i % numCols) < numCols - k))
                diag[i] = 1;
    }
    else {
        for (uint32_t i = 0; i < n; i++)
            if ((i % numCols >= numCols - k) && (i % numCols < numCols)) {
                diag[i] = 1;
            }
    }

    return diag;
}

/**
 *Compute diagonals for the permutation Psi (W).
 *B[i,j] = A[i+1,j]
 */
std::vector<double> GenPsiDiag(uint32_t numCols, int32_t k) {
    uint32_t n = numCols * numCols;
    std::vector<double> diag(n, 1);
    return diag;
}

std::vector<double> GenTransposeDiag(uint32_t totalSlots, uint32_t numCols, int32_t i) {
    if (static_cast<int32_t>(numCols) < i)
        OPENFHE_THROW("numCols cannot be less than the index");

    uint32_t start = 0;
    uint32_t max   = 0;
    if (i < 0) {
        start = -i;
        max   = numCols;
    }
    else {
        max = numCols - i;
    }

    uint32_t n = numCols * numCols;
    std::vector<double> diag(totalSlots, 0);
    for (uint32_t t = 0; t < totalSlots / n; ++t) {
        for (uint32_t j = start; j < max; j++) {
            uint32_t idx = t * n + (numCols + 1) * j + i;
            if (idx < totalSlots)
                diag[idx] = 1;
        }
    }
    return diag;
}
