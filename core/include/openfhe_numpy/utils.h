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
#ifndef UTILS_H
#define UTILS_H

// load necessary library
#include "openfhe_numpy/config.h"
#include "openfhe_numpy/internal/helper.h"

using namespace lbcrypto;

uint32_t NextPow2(const uint32_t x);
bool IsPow2(uint32_t x);
// template <class Element>
// void Debug(CryptoContext<Element> cc, KeyPair<Element> keys, Ciphertext<Element> ct, std::string msg, int length = 16);
void Debug(CryptoContext<DCRTPoly> cc, KeyPair<DCRTPoly> keys, Ciphertext<DCRTPoly> ct, std::string msg, int length=16);
// void Debug(CryptoContext, KeyPair keys, Ciphertext, std::string msg, int length = 16);
std::vector<double> GenSigmaDiag(std::size_t rowsize, int32_t k);
std::vector<double> GenTauDiag(std::size_t total_slots, std::size_t rowsize, int32_t k);
std::vector<double> GenPhiDiag(std::size_t rowsize, int32_t k, int type);
std::vector<double> GenPsiDiag(std::size_t rowsize, int32_t k);
std::vector<double> GenTransposeDiag(std::size_t total_slots, std::size_t rowsize, int32_t i);
void RoundVector(std::vector<double>& vector);
#endif  // UTILS_H