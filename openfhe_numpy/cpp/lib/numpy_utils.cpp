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

#include <cmath>

void RoundVector(std::vector<double>& vector) {
    for (double& e : vector)
        e = std::round(e);
}

uint32_t NextPow2(uint32_t x) {
    return pow(2, std::ceil(log(double(x)) / log(2.0)));
};


/*
Compute diagonals for the permutation matrix Sigma.
B[i,j] = A[i, i +j]
*/
std::vector<double> GenSigmaDiag(std::size_t numCols, int32_t k) {
    int32_t n = numCols * numCols;
    std::vector<double> diag(n, 0);

    if (k >= 0) {
        for (int32_t i = 0; i < n; i++) {
            int32_t tmp = i - numCols * k;
            if ((0 <= tmp) and (tmp < numCols - k)) {
                diag[i] = 1;
            }
        }
    }
    else {
        for (int32_t i = 0; i < n; i++) {
            int32_t tmp = i - (numCols + k) * numCols;
            if ((-k <= tmp) and (tmp < numCols)) {
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

std::vector<double> GenTauDiag(std::size_t totalSlots, std::size_t numCols, int32_t k) {
    std::size_t n = numCols * numCols;
    std::vector<double> diag(totalSlots, 0);

    for (std::size_t t = 0; t < totalSlots / n; t++) {
        for (std::size_t i = 0; i < numCols; i++) {
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
std::vector<double> GenPhiDiag(std::size_t numCols, int32_t k, int type) {
    std::size_t n = numCols * numCols;
    std::vector<double> diag(n, 0);

    if (type == 0) {
        for (std::size_t i = 0; i < n; i++)
            if ((i % numCols >= 0) and ((i % numCols) < numCols - k))
                diag[i] = 1;
        return diag;
    }
    for (std::size_t i = 0; i < n; i++)
        if ((i % numCols >= numCols - k) and (i % numCols < numCols)) {
            diag[i] = 1;
        }

    return diag;
}

/**
 *Compute diagonals for the permutation Psi (W).
 *B[i,j] = A[i+1,j]
 */
std::vector<double> GenPsiDiag(std::size_t numCols, int32_t k) {
    std::size_t n = numCols * numCols;
    std::vector<double> diag(n, 1);
    return diag;
}

std::vector<double> GenTransposeDiag(std::size_t totalSlots, std::size_t numCols, int32_t i) {
    std::size_t n = numCols * numCols;
    std::vector<double> diag(totalSlots, 0);
    for (auto t = 0; t < totalSlots / n; t++) {
        if (i >= 0) {
            for (auto j = 0; j < numCols - i; j++) {
                auto idx = t * n + (numCols + 1) * j + i;
                if (idx < totalSlots)
                    diag[idx] = 1;
            }
        }
        else {
            for (auto j = -i; j < numCols; j++) {
                auto idx = t * n + (numCols + 1) * j + i;
                if (idx < totalSlots)
                    diag[idx] = 1;
            }
        }
    }
    return diag;
}
