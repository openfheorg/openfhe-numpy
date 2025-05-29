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
#ifndef __NUMPY_HELPER_FUNCTIONS_H__
#define __NUMPY_HELPER_FUNCTIONS_H__

#include "lattice/lat-hal.h"

#include <cstdint>
#include <vector>

// Internal helper functions - NOT part of the public API
// Do not use directly in client code

using namespace lbcrypto;
/*
 * Displaying the 2D vector
 */
void PrintMatrix(const std::vector<std::vector<DCRTPoly>>& mat);

/*
 * Displaying the 1D vector
 */
void PrintVector(const std::vector<DCRTPoly>& vec);

/*
 * Normal Matrix-Matrix Product
 */
std::vector<std::vector<double>> MulMats(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& B);

/*
 * Normal Matrix Vector Multiplication
 */
std::vector<double> MulMatVec(std::vector<std::vector<double>> mat, std::vector<double> vec);

/*
 * Sample a rational random vector
 */
std::vector<double> RandVec(int n, int modulus = 5, bool verbose = true);

/*
 * Sample a rational random matrix
 */
std::vector<std::vector<double>> RandMatrix(int nrows,
                                            int numCols,
                                            double min_val = 0,
                                            double max_val = 10,
                                            bool verbose   = true);

std::vector<DCRTPoly> EncodeMatrix(const std::vector<std::vector<DCRTPoly>>& mat, long total_slots);

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
std::vector<DCRTPoly> PackVecRowWise(const std::vector<DCRTPoly>& v, std::size_t block_size, std::size_t num_slots);

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
std::vector<T> PackVecColWise(const std::vector<T>& v, std::size_t block_size, std::size_t num_slots);

void print_range(const std::vector<DCRTPoly>& v, std::size_t start, std::size_t end);

#endif  // __NUMPY_HELPER_FUNCTIONS_H__
