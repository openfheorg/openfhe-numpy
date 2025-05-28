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
#ifndef HELPER_H
#define HELPER_H

#include "../config.h"
#include <iostream>
#include <vector>

// Internal helper functions - NOT part of the public API
// Do not use directly in client code

/**
 * @brief Check if a number is a power of two
 */
inline bool IsPowerOfTwo(uint32_t x) {
    return x > 0 && (x & (x - 1)) == 0;
}
/*
Displaying the 2D vector
*/
template <typename Element>
inline void PrintMatrix(const std::vector<std::vector<Element>>& mat) {
    for (uint32_t i = 0; i < mat.size(); i++) {
        for (uint32_t j = 0; j < mat[i].size(); j++)
            std::cout << mat[i][j] << " ";
        std::cout << std::endl;
    }
};

/*
Displaying the 1D vector
*/

template <typename Element>
inline void PrintVector(const std::vector<Element> vec) {
    std::cout << std::fixed << std::setprecision(2);
    for (uint32_t i = 0; i < vec.size(); i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl << std::endl;
};

/*
Normal Matrix-Matrix Product
*/

inline std::vector<std::vector<double>> MulMats(std::vector<std::vector<double>>& A,
                                                std::vector<std::vector<double>>& B) {
    uint32_t nA = A.size();
    uint32_t mA = A[0].size();

    uint32_t nB = B.size();
    uint32_t mB = B[0].size();

    std::vector<std::vector<double>> result(nA, std::vector<double>(mB, 0));

    try {
        if (mA == nB) {
            for (uint32_t i = 0; i < nA; i++) {
                for (uint32_t j = 0; j < mB; j++) {
                    for (uint32_t k = 0; k < mA; k++) {
                        result[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }
        else {
            throw("Error:: size problems!!!");
        }
    }
    catch (...) {
        OPENFHE_THROW("ERROR ::: Normal Matrix Multiplication ::: size is different");
    }
    return result;
};

/*
Normal Matrix Vector Multiplication
*/

inline std::vector<double> MulMatVec(std::vector<std::vector<double>> mat, std::vector<double> vec) {
    uint32_t n = mat.size();
    uint32_t m = mat[0].size();
    uint32_t k = vec.size();

    std::vector<double> result(m, 0);

    try {
        if (m == k) {
            for (uint32_t i = 0; i < n; i++) {
                for (uint32_t j = 0; j < m; j++) {
                    result[i] += mat[i][j] * vec[j];
                }
            }
        }
        else {
            throw("Error:: size problems!!!");
        }
    }
    catch (...) {
        OPENFHE_THROW("ERROR ::: Normal Matrix Multiplication ::: size is different");
    }
    return result;
};

/*
Sample a rational random vector
*/

inline std::vector<double> RandVec(const int n, const int modulus = 5, const bool verbose = true) {
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

/*
Sample a rational random matrix
*/

inline std::vector<std::vector<double>> RandMatrix(const int nrows,
                                                   const int numCols,
                                                   const double min_val = 0,
                                                   const double max_val = 10,
                                                   const bool verbose   = true) {
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

template <class Element>
inline std::vector<Element> EncodeMatrix(std::vector<std::vector<Element>> mat, const long total_slots) {
    uint32_t n = mat.size();
    uint32_t m = mat[0].size();

    uint32_t size   = n * m;
    uint32_t blocks = total_slots / size;

    std::vector<Element> vec(total_slots, 0);
    long k = 0;
    for (uint32_t t = 0; t < blocks; ++t) {
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t j = 0; j < m; ++j) {
                vec[k] = mat[i][j];
                k += 1;
            }
        }
    }
    return vec;
}

/**
 * @brief Clone a vector to fill num_slots by repeating each element block_size times
 * 
 * @param v Input vector to be packed
 * @param block_size Size of each block (must be power of two)
 * @param num_slots Total number of slots to fill (must be power of two)
 * @return std::vector<double> Packed vector
 * 
 * Example: For v=[1,2,3], block_size=4, num_slots=12
 * Result: [1,1,1,1, 2,2,2,2, 3,3,3,3]
 */
template <class Element>
inline std::vector<Element> PackVecRowWise(const std::vector<Element>& v,
                                           std::size_t block_size,
                                           std::size_t num_slots) {
    // Check input parameters
    size_t n = v.size();

    // Check power of two constraints
    if (!IsPowerOfTwo(block_size)) {
        throw std::invalid_argument("BlockSize must be a power of two");
    }

    if (!IsPowerOfTwo(num_slots)) {
        throw std::invalid_argument("NumSlots must be a power of two");
    }

    // Check size constraints
    if (num_slots < n) {
        throw std::runtime_error("ERROR ::: [row_wise_vector] vector is longer than total slots");
    }

    if (num_slots == n) {
        if (num_slots / block_size > 1) {
            throw std::runtime_error("ERROR ::: [row_wise_vector] vector is too long, can't duplicate");
        }
        return v;
    }

    if (num_slots % block_size != 0)
        throw std::runtime_error("ERROR ::: num_slots % block_size");

    // Calculate blocks and available space
    uint32_t total_blocks = num_slots / block_size;
    uint32_t free_slots   = num_slots - n * block_size;

    // Create and fill packed vector
    std::vector<Element> packed(num_slots, 0.0);
    size_t k = 0;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < block_size; j++) {
            packed[k] = v[i];
            k++;
        }
    }

    return packed;
}


/**
 * @brief Clone a vector to fill num_slots by repeating elements column-wise
 * 
 * For example: For v=[1,2,3], block_size=4, num_slots=12
 * Result: [1,2,3,0, 1,2,3,0, 1,2,3,0]
 * 
 * @param v Input vector to be packed
 * @param block_size Size of each block (must be power of two)
 * @param num_slots Total number of slots to fill (must be power of two)
 * @return std::vector<T> Packed vector
 */
template <typename T>
inline std::vector<T> PackVecColWise(const std::vector<T>& v, 
                                   std::size_t block_size, 
                                   std::size_t num_slots) {
    // Check input parameters
    std::size_t n = v.size();
    
    // Check power of two constraints
    if (!IsPowerOfTwo(block_size)) {
        throw std::invalid_argument("BlockSize must be a power of two");
    }
    
    if (!IsPowerOfTwo(num_slots)) {
        throw std::invalid_argument("NumSlots must be a power of two");
    }
    
    // Check size constraints
    if (block_size < n) {
        throw std::runtime_error(
            "ERROR ::: [col_wise_vector] vector of size (" + 
            std::to_string(n) + 
            ") is longer than size of a slot (" + 
            std::to_string(block_size) + ")"
        );
    }
    
    if (num_slots < n) {
        throw std::runtime_error("ERROR ::: [col_wise_vector] vector is longer than total slots");
    }
    
    if (num_slots == n) {
        return v;
    }
    
    // Create and fill packed vector
    std::vector<T> packed(num_slots, 0);
    
    // Calculate blocks
    if (num_slots % block_size != 0)
        throw std::runtime_error("ERROR ::: num_slots % block_size");
        
    std::size_t total_blocks = num_slots / block_size;
    std::size_t free_slots = num_slots - n * total_blocks;
    
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
template <class Element>
inline void print_range(const std::vector<Element>& v,
                        std::size_t start,
                        std::size_t end)  
{
    if (start > end || end > v.size()) {
        std::cerr << "Invalid range\n";
        return;
    }
    for (std::size_t i = start; i < end; ++i) {
        std::cout << v[i] << (i + 1 < end ? ' ' : '\n');
    }
}

#endif  // HELPER_H
