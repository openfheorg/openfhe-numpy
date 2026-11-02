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
#ifndef __NUMPY_LINTRANS_H__
#define __NUMPY_LINTRANS_H__

#include "openfhe.h"

#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <vector>

using namespace lbcrypto;

// -----------------------------------------------------------------------------
// Generic homomorphic linear transform (issue #41)
// -----------------------------------------------------------------------------
// Linear transforms: y = sum_t diag_t (.) Rot_{offset_t}(x)
// Default: BSGS (baby-step giant-step) with hoisted rotations. No setup is needed.
// Matrix products under FIXEDMANUAL scaling keep the legacy JKLS method.
// Optional controls:
//   SetLinTransAlgo(NAIVE) or OPENFHE_NUMPY_LT_ALGO=naive  legacy method: one rotation per diagonal
//   SetLinTransBabyStep(b) or OPENFHE_NUMPY_LT_BSTEP=b     fixed baby step (default: heuristic)
//   SetLinTransThreads(t)  or OPENFHE_NUMPY_LT_THREADS=t   upper limit on the outer threads
//   LinearTransformRotationIndices(offsets, b)             minimal rotation keys for EvalLinearTransform
//   OPENFHE_NUMPY_CANONICAL_KEYS=0                         keep the wrap keys d*k - d^2. By default they
//   or SetCanonicalRowShiftKeys(false)                     are skipped when the batch size is d^2. A
//                                                          ciphertext with more slots than the batch size
//                                                          then raises an error that says how to resolve it
//   OPENFHE_NUMPY_LT_VERBOSE=1                             print the chosen plan
// -----------------------------------------------------------------------------

namespace openfhe_numpy {

enum class LinTransAlgo : uint8_t {
    AUTO    = 0,
    NAIVE   = 1,
    BSGS    = 2,
};

void SetLinTransAlgo(LinTransAlgo algo);
LinTransAlgo GetLinTransAlgo();

// Baby-step size used by BSGS; 0 selects the heuristic (env OPENFHE_NUMPY_LT_BSTEP).
void SetLinTransBabyStep(uint32_t bStep);
uint32_t GetLinTransBabyStep();

// Upper bound on the threads used by the outer parallel loops; 0 = OpenMP maximum
// (env OPENFHE_NUMPY_LT_THREADS).
void SetLinTransThreads(uint32_t numThreads);
uint32_t GetLinTransThreads();

// Strategy that will actually run for this ciphertext when `requested` is asked for.
LinTransAlgo ResolveLinTransAlgo(ConstCiphertext<DCRTPoly>& ciphertext, LinTransAlgo requested = LinTransAlgo::AUTO);

// When the batch size equals numCols^2 and the ciphertext data repeats with a period dividing it,
// rotations by d*k and d*k - d^2 act identically on the packed data, so only the canonical d*k
// keys are needed.
// Key generation: when the batch size is d^2, skip the d*k - d^2 keys by default (automatic
// wrap-key pruning: d - 2 fewer keys for square matmul, 126 keys and about 16 GB at d = 128,
// ring 2^16). A ciphertext with more slots than the batch size needs them and raises an error that
// says how to resolve it. OPENFHE_NUMPY_CANONICAL_KEYS=0 or SetCanonicalRowShiftKeys(false) keeps them.
bool UseCanonicalRowShifts(size_t slots, uint32_t numCols);
void SetCanonicalRowShiftKeys(bool prune);
bool GetCanonicalRowShiftKeys();
// Evaluation: rotate by d*k whenever the batch size is d^2 and the ciphertext data repeats with a
// period dividing it. The d*k keys exist under both key-generation modes.
bool UseCanonicalRowShifts(ConstCiphertext<DCRTPoly>& ciphertext, uint32_t numCols);

// Throws (serially, before any parallel region) if a rotation key is missing.
void RequireRotationKeys(ConstCiphertext<DCRTPoly>& ciphertext, const std::vector<int32_t>& indices);
// Throws for a non-relinearized (3+ element) ciphertext, as EvalRotate does.
void RequireRelinearized(ConstCiphertext<DCRTPoly>& ciphertext);

Ciphertext<DCRTPoly> EvalLinearTransform(ConstCiphertext<DCRTPoly>& ciphertext,
                                         const std::vector<int32_t>& offsets,
                                         const std::vector<std::vector<double>>& diagonals,
                                         LinTransAlgo algo = LinTransAlgo::AUTO,
                                         uint32_t bStep    = 0);

// Rotation indices the BSGS plan for these offsets needs (babies plus the giant stride),
// ignoring key availability. Useful for generating a minimal key set: EvalLinearTransform plans
// from the same offsets, so these keys suffice whichever diagonals turn out to be zero, provided
// the same baby step is passed to both (or a split with these keys is found by the search).
std::vector<int32_t> LinearTransformRotationIndices(const std::vector<int32_t>& offsets, uint32_t bStep = 0);

// One digit decomposition of a ciphertext shared by many hoisted rotations.
// The input is rescaled once if its noise-scale degree is above one (auto rescaling modes only).
class HoistedCiphertext {
public:
    explicit HoistedCiphertext(ConstCiphertext<DCRTPoly>& ciphertext);

    const Ciphertext<DCRTPoly>& Get() const {
        return m_ct;
    }

    Ciphertext<DCRTPoly> Rotate(int32_t index) const;

    Plaintext Encode(const std::vector<double>& values) const;     // at the ciphertext level

    // sum_t diag_t (.) Rot_{offset_t}(x), all rotations hoisted, no giant steps
    Ciphertext<DCRTPoly> MaskedRotationSum(const std::vector<int32_t>& offsets,
                                           const std::vector<std::vector<double>>& diagonals) const;

private:
    Ciphertext<DCRTPoly> m_ct;
    std::shared_ptr<std::vector<DCRTPoly>> m_digits;
    uint32_t m_m;
    uint32_t m_slots;
};

// Rescale (auto modes only) so the noise-scale degree is one.
Ciphertext<DCRTPoly> PrepareLinTransInput(ConstCiphertext<DCRTPoly>& ciphertext);

// An exception escaping an OpenMP parallel region terminates the process; run each loop body
// through Run() and call Rethrow() after the region so errors reach the caller (and Python).
class ParallelExceptionGuard {
public:
    template <typename F>
    void Run(F&& body) {
        try {
            body();
        }
        catch (...) {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_exception)
                m_exception = std::current_exception();
        }
    }
    void Rethrow() const {
        if (m_exception)
            std::rethrow_exception(m_exception);
    }

private:
    std::mutex m_mutex;
    std::exception_ptr m_exception;
};

uint32_t LinTransThreadLimit(uint32_t work);

}  // namespace openfhe_numpy

#endif  // __NUMPY_LINTRANS_H__
