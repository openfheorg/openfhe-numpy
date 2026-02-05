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
#ifndef __NUMPY_ENC_MATRIX_H__
#define __NUMPY_ENC_MATRIX_H__

#include "numpy_constants.h"
#include "openfhe.h"

using namespace lbcrypto;

// -----------------------------------------------------------------------------
// OpenFHE Matrix Encryption API
// -----------------------------------------------------------------------------

// This API provides functions for matrix encryption and operations using the OpenFHE library.
// It includes functions for generating rotation keys, performing matrix-vector multiplication,
// linear transformations, and more. The API is designed to work with the OpenFHE library's
// encryption scheme and provides a high-level interface for matrix operations.
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

namespace openfhe_numpy {
//TODO: Change from ConstCiphertext to ConstCiphertext
//TODO: using references

uint32_t MulDepthAccumulation(uint32_t numRows, uint32_t numCols, bool isSumRows);

void EvalLinTransKeyGen(PrivateKey<DCRTPoly>& secretKey, uint32_t numCols, LinTransType type, uint32_t numRepeats = 0);

void EvalSquareMatMultRotateKeyGen(PrivateKey<DCRTPoly>& secretKey, uint32_t numCols);

void EvalSumCumRowsKeyGen(PrivateKey<DCRTPoly>& secretKey, uint32_t numCols);

void EvalSumCumColsKeyGen(PrivateKey<DCRTPoly>& secretKey, uint32_t numCols);

Ciphertext<DCRTPoly> EvalMultMatVec(std::shared_ptr<std::map<uint32_t, lbcrypto::EvalKey<DCRTPoly>>>& evalKeys,
                                   MatVecEncoding encodeType,
                                   uint32_t numCols,
                                   ConstCiphertext<DCRTPoly>& ctVector,
                                   ConstCiphertext<DCRTPoly>& ctMatrix);

Ciphertext<DCRTPoly> EvalLinTransSigma(PrivateKey<DCRTPoly>& secretKey,
                                      ConstCiphertext<DCRTPoly>& ciphertext,
                                      uint32_t numCols);

Ciphertext<DCRTPoly> EvalLinTransSigma(ConstCiphertext<DCRTPoly>& ciphertext, uint32_t numCols);

Ciphertext<DCRTPoly> EvalLinTransTau(ConstCiphertext<DCRTPoly>& ctVector, uint32_t numCols);

Ciphertext<DCRTPoly> EvalLinTransTau(PrivateKey<DCRTPoly>& secretKey,
                                    ConstCiphertext<DCRTPoly>& ciphertext,
                                    uint32_t numCols);

Ciphertext<DCRTPoly> EvalLinTransPhi(ConstCiphertext<DCRTPoly>& ctVector, uint32_t numCols, uint32_t numRepeats);

Ciphertext<DCRTPoly> EvalLinTransPhi(PrivateKey<DCRTPoly>& secretKey,
                                    ConstCiphertext<DCRTPoly>& ctVector,
                                    uint32_t numCols,
                                    uint32_t numRepeats);

Ciphertext<DCRTPoly> EvalLinTransPsi(PrivateKey<DCRTPoly>& secretKey,
                                    ConstCiphertext<DCRTPoly>& ctVector,
                                    uint32_t numCols,
                                    uint32_t numRepeats);

Ciphertext<DCRTPoly> EvalLinTransPsi(ConstCiphertext<DCRTPoly>& ctVector, uint32_t numCols, uint32_t numRepeats);

Ciphertext<DCRTPoly> EvalMatMulSquare(ConstCiphertext<DCRTPoly>& ctMatA,
                                      ConstCiphertext<DCRTPoly>& ctMatB,
                                      uint32_t numCols);

Ciphertext<DCRTPoly> EvalMatMulSquare(ConstCiphertext<DCRTPoly>& ctMatA,
                                      ConstPlaintext& ptMatB,
                                      uint32_t numCols);

Ciphertext<DCRTPoly> EvalMatMulSquare(ConstCiphertext<DCRTPoly>& ctMatA,
                                      const std::vector<double>& rMatB,
                                      uint32_t numCols);

Ciphertext<DCRTPoly> EvalMatMulSquare(ConstPlaintext& ptMatA,
                                      ConstCiphertext<DCRTPoly>& ctMatB,
                                      uint32_t numCols);


Ciphertext<DCRTPoly> EvalMatMulSquare(const std::vector<double>& rMatA,
                                      ConstCiphertext<DCRTPoly>& ctMatB,
                                      uint32_t numCols);

Ciphertext<DCRTPoly> EvalTranspose(PrivateKey<DCRTPoly>& secretKey,
                                  ConstCiphertext<DCRTPoly>& ciphertext,
                                  uint32_t numCols);

Ciphertext<DCRTPoly> EvalTranspose(ConstCiphertext<DCRTPoly>& ciphertext, uint32_t numCols);

Ciphertext<DCRTPoly> EvalSumCumRows(ConstCiphertext<DCRTPoly>& ciphertext,
                                          uint32_t numCols,
                                          uint32_t numRows = 0,
                                          uint32_t slots   = 0);

Ciphertext<DCRTPoly> EvalSumCumCols(ConstCiphertext<DCRTPoly>& ciphertext,
                                          uint32_t numCols,
                                          uint32_t subringDim = 0);

Ciphertext<DCRTPoly> EvalReduceCumRows(ConstCiphertext<DCRTPoly>& ciphertext,
                                          uint32_t numCols,
                                          uint32_t numRows = 0,
                                          uint32_t slots   = 0);

Ciphertext<DCRTPoly> EvalReduceCumCols(ConstCiphertext<DCRTPoly>& ciphertext,
                                          uint32_t numCols,
                                          uint32_t subringDim = 0);
}  // namespace openfhe_numpy

#endif  // __NUMPY_ENC_MATRIX_H__
