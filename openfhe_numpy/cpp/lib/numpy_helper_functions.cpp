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
#include "numpy_helper_functions.h"
#include "utils/utilities.h"
#include "utils/exception.h"

#include <iostream>

void PrintMatrix(const std::vector<std::vector<DCRTPoly>>& mat) {
    for (const auto& row : mat) {
        for (const auto& val : row)
            std::cout << val << " ";
        std::cout << std::endl;
    }
};

void PrintVector(const std::vector<DCRTPoly>& vec) {
    std::cout << std::fixed << std::setprecision(2);
    for (const auto& val : vec)
        std::cout << val << " ";
    std::cout << std::endl;
};

std::vector<std::vector<double>> MulMats(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& B) {
    uint32_t nA = A.size();
    uint32_t mA = A[0].size();

    uint32_t nB = B.size();
    uint32_t mB = B[0].size();

    if (mA != nB)
        OPENFHE_THROW("Mismatched vector sizes");

    std::vector<std::vector<double>> result(nA, std::vector<double>(mB, 0));
    for (uint32_t i = 0; i < nA; i++) {
        for (uint32_t j = 0; j < mB; j++) {
            for (uint32_t k = 0; k < mA; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
};

std::vector<double> MulMatVec(std::vector<std::vector<double>> mat, std::vector<double> vec) {
    uint32_t n = mat.size();
    uint32_t m = mat[0].size();
    uint32_t k = vec.size();

    if (m != k)
        OPENFHE_THROW("Mismatched vector sizes");

    std::vector<double> result(m, 0);
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < m; j++) {
            result[i] += mat[i][j] * vec[j];
        }
    }

    return result;
};

std::vector<double> RandVec(int n, int modulus, bool verbose) {
    std::vector<double> vec(n, 0);
    std::srand(static_cast<unsigned>(std::time(0)));
    for (int i = 0; i < n; i++)
        vec[i] = rand() % modulus;

    // Print the generated vector
    if (verbose) {
        std::cout << "Random Vector:" << std::endl;
        for (const auto& element : vec)
            std::cout << element << " ";
    }
    return vec;
};

std::vector<std::vector<double>> RandMatrix(int nrows, int numCols, double min_val, double max_val, bool verbose) {
    std::vector<std::vector<double>> matrix(nrows, std::vector<double>(numCols));
    std::srand(static_cast<unsigned>(std::time(0)));

    // Fill the matrix with random numbers in the range [minVal, maxVal]
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            matrix[i][j] = min_val + std::fmod(std::rand(), (max_val - min_val + 1));
        }
    }

    // Print the generated matrix
    if (verbose) {
        std::cout << "Random Matrix:" << std::endl;
        for (const auto& row : matrix) {
            for (const auto& element : row) {
                std::cout << element << " ";
            }
            std::cout << std::endl;
        }
    }
    return matrix;
};

std::vector<DCRTPoly> EncodeMatrix(const std::vector<std::vector<DCRTPoly>>& mat, long total_slots) {
    uint32_t n = mat.size();
    uint32_t m = mat[0].size();

    uint32_t size   = n * m;
    uint32_t blocks = total_slots / size;

    std::vector<DCRTPoly> vec;
    vec.reserve(total_slots);
    for (uint32_t t = 0; t < blocks; ++t) {
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t j = 0; j < m; ++j)
                vec.push_back(mat[i][j]);
        }
    }
    return vec;
}

std::vector<DCRTPoly> PackVecRowWise(const std::vector<DCRTPoly>& v, std::size_t block_size, std::size_t num_slots) {
    // Check input parameters
    size_t n = v.size();

    // Check power of two constraints
    if (!lbcrypto::IsPowerOfTwo(block_size)) {
        OPENFHE_THROW("BlockSize must be a power of two");
    }

    if (!lbcrypto::IsPowerOfTwo(num_slots)) {
        OPENFHE_THROW("NumSlots must be a power of two");
    }

    // Check size constraints
    if (num_slots < n) {
        OPENFHE_THROW("vector is longer than total slots");
    }

    if (num_slots == n) {
        if (num_slots / block_size > 1) {
            OPENFHE_THROW("vector is too long, can't duplicate");
        }
        return v;
    }

    if (num_slots % block_size != 0)
        OPENFHE_THROW("num_slots % block_size");

    // Calculate blocks and available space
    uint32_t total_blocks = num_slots / block_size;
    uint32_t free_slots   = num_slots - n * block_size;

    // Create and fill packed vector
    std::vector<DCRTPoly> packed;
    packed.reserve(num_slots);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < block_size; j++)
            packed.push_back(v[i]);
    }

    return packed;
}

template <typename T>
std::vector<T> PackVecColWise(const std::vector<T>& v, std::size_t block_size, std::size_t num_slots) {
    // Check input parameters
    std::size_t n = v.size();

    // Check power of two constraints
    if (!lbcrypto::IsPowerOfTwo(block_size)) {
        OPENFHE_THROW("BlockSize must be a power of two");
    }

    if (!lbcrypto::IsPowerOfTwo(num_slots)) {
        OPENFHE_THROW("NumSlots must be a power of two");
    }

    // Check size constraints
    if (block_size < n) {
        OPENFHE_THROW("vector of size (" + std::to_string(n) +
                                 ") is longer than size of a slot (" + std::to_string(block_size) + ")");
    }

    if (num_slots < n) {
        OPENFHE_THROW("vector is longer than total slots");
    }

    if (num_slots == n) {
        return v;
    }

    // Calculate blocks
    if (num_slots % block_size != 0)
        OPENFHE_THROW("num_slots % block_size");

    // Create and fill packed vector
    std::vector<T> packed(num_slots, 0);

    std::size_t total_blocks = num_slots / block_size;
    std::size_t free_slots   = num_slots - n * total_blocks;

    // Pack the vector column-wise
    std::size_t k = 0;  // index into vector to write
    for (std::size_t i = 0; i < total_blocks; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            packed[k] = v[j];
            ++k;
        }
        k += block_size - n;  // Skip remaining slots in the block
    }

    return packed;
}

void print_range(const std::vector<DCRTPoly>& v, std::size_t start, std::size_t end) {
    if (start > end || end > v.size()) {
        std::cerr << "Invalid range\n";
        return;
    }
    
    for (std::size_t i = start; i < end; ++i) {
        std::cout << v[i] << (i + 1 < end ? ' ' : '\n');
    }
}
