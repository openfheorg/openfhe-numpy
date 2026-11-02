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
#include "numpy_lintrans.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <string>

#ifdef _OPENMP
    #include <omp.h>
#endif

using namespace lbcrypto;

namespace openfhe_numpy {

namespace {

// -----------------------------------------------------------------------------
// Settings (initialised once from the environment)
// -----------------------------------------------------------------------------

std::string ToLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

uint32_t EnvUInt(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0')
        return 0;
    return static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
}

LinTransAlgo ParseAlgo(const std::string& value) {
    const std::string v = ToLower(value);
    if (v.empty() || v == "auto")
        return LinTransAlgo::AUTO;
    if (v == "naive")
        return LinTransAlgo::NAIVE;
    if (v == "bsgs")
        return LinTransAlgo::BSGS;
    OPENFHE_THROW("OPENFHE_NUMPY_LT_ALGO must be one of naive|bsgs|auto, got [" + value + "]");
}

struct Settings {
    std::atomic<uint8_t> algo;
    std::atomic<uint32_t> bStep;
    std::atomic<uint32_t> threads;
    uint32_t parallelMin;
    std::atomic<bool> canonical;
    bool verbose;

    Settings() {
        LinTransAlgo a = LinTransAlgo::AUTO;
        if (const char* v = std::getenv("OPENFHE_NUMPY_LT_ALGO")) {
            try {
                a = ParseAlgo(v);
            }
            catch (const std::exception& e) {
                std::cerr << "[openfhe_numpy] " << e.what() << "; using auto" << std::endl;
            }
        }
        algo.store(static_cast<uint8_t>(a));
        bStep.store(EnvUInt("OPENFHE_NUMPY_LT_BSTEP"));
        threads.store(EnvUInt("OPENFHE_NUMPY_LT_THREADS"));
        // Loops with fewer independent items than this run sequentially so that OpenFHE's own
        // per-tower parallelism stays active (nested OpenMP regions run on one thread).
        parallelMin = EnvUInt("OPENFHE_NUMPY_LT_PAR_MIN");
        if (parallelMin == 0)
            parallelMin = 4;
        // Automatic wrap-key pruning, on unless OPENFHE_NUMPY_CANONICAL_KEYS=0. A ciphertext with more
        // slots than the batch size needs the d*k - d^2 keys and raises an error that says so.
        const char* c = std::getenv("OPENFHE_NUMPY_CANONICAL_KEYS");
        canonical.store(!(c && std::string(c) == "0"));
        const char* vb = std::getenv("OPENFHE_NUMPY_LT_VERBOSE");
        verbose = (vb && *vb && std::string(vb) != "0");
    }
};

Settings& GetSettings() {
    static Settings settings;
    return settings;
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

std::shared_ptr<CryptoParametersCKKSRNS> CkksParams(ConstCiphertext<DCRTPoly>& ct) {
    auto cp = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(ct->GetCryptoParameters());
    if (!cp)
        OPENFHE_THROW("Linear transforms require CKKS RNS crypto parameters.");
    return cp;
}

bool AutoRescale(ScalingTechnique st) {
    return st == FLEXIBLEAUTO || st == FLEXIBLEAUTOEXT || st == FIXEDAUTO || st == COMPOSITESCALINGAUTO;
}

// out[p] = v[(p + r) mod S]   (same convention as EvalRotate)
std::vector<double> RotateLeftCyclic(const std::vector<double>& v, int64_t r) {
    const int64_t S = static_cast<int64_t>(v.size());
    if (S == 0)
        return v;
    int64_t sh = r % S;
    if (sh < 0)
        sh += S;
    std::vector<double> out(v.size());
    std::rotate_copy(v.begin(), v.begin() + sh, v.end(), out.begin());
    return out;
}

bool AllZero(const std::vector<double>& v) {
    return std::all_of(v.begin(), v.end(), [](double x) { return x == 0.0; });
}

size_t NumTowers(ConstPlaintext& pt) {
    return pt->GetElement<DCRTPoly>().GetNumOfElements();
}

// Diagonal plaintext at the ciphertext level (Q_l basis). Falls back to the legacy level-0
// encoding (EvalMult then adjusts) when levels cannot be matched directly.
Plaintext EncodeAtLevel(ConstCiphertext<DCRTPoly>& ct, const std::vector<double>& values, uint32_t slots) {
    auto cc       = ct->GetCryptoContext();
    auto cp       = CkksParams(ct);
    uint32_t lvl  = (cp->GetCompositeDegree() == 1) ? static_cast<uint32_t>(ct->GetLevel()) : 0;
    Plaintext pt  = cc->MakeCKKSPackedPlaintext(values, 1, lvl, nullptr, slots);
    if (lvl != 0 && NumTowers(pt) != ct->GetElements()[0].GetNumOfElements())
        pt = cc->MakeCKKSPackedPlaintext(values, 1, 0, nullptr, slots);
    return pt;
}

const std::map<uint32_t, EvalKey<DCRTPoly>>* TryGetKeyMap(ConstCiphertext<DCRTPoly>& ct) {
    try {
        return &CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(ct->GetKeyTag());
    }
    catch (...) {
        return nullptr;
    }
}

// -----------------------------------------------------------------------------
// BSGS planning
// -----------------------------------------------------------------------------

struct Term {
    int32_t offset;
    std::vector<double> diag;  // empty when `zero`
    bool zero = false;         // the merged diagonal is all zero: planned, never evaluated
};

struct Plan {
    uint32_t b     = 1;
    uint32_t G     = 1;
    int32_t giant  = 0;                // s * b
    std::vector<int32_t> babyOffsets;  // size b
    std::vector<int32_t> cell;         // size G * b, index into terms or -1
};

// Baby steps are hoisted and evaluated in parallel, giant steps are sequential full key switches,
// so a few large groups (G ~ 4) beat the classic sqrt(span) balance on many-core machines
// (BERT shapes, ring 2^16, 72 cores: b = 64 vs 16 is 1.7x faster for 255 diagonals).
uint32_t DefaultBabyStep(int64_t span) {
    uint32_t b = 1;
    const double target = std::max(std::sqrt(static_cast<double>(span)), static_cast<double>(span) / 4.0);
    while (static_cast<double>(b) < target)
        b <<= 1;
    return b;
}

Plan SingleGroupPlan(const std::vector<Term>& terms) {
    Plan plan;
    plan.b = static_cast<uint32_t>(terms.size());
    plan.G = 1;
    plan.babyOffsets.resize(terms.size());
    plan.cell.resize(terms.size());
    for (size_t t = 0; t < terms.size(); ++t) {
        plan.babyOffsets[t] = terms[t].offset;
        plan.cell[t]        = static_cast<int32_t>(t);
    }
    return plan;
}

// The baby steps and giant groups that the non-zero terms of a plan actually use.
struct PlanUsage {
    std::vector<char> baby;  // size plan.b
    uint32_t maxGroup   = 0;  // highest group index holding a non-zero term
    size_t keySwitches  = 0;  // used non-zero baby rotations + maxGroup giant rotations
};

PlanUsage UsageOf(const Plan& plan, const std::vector<Term>& terms) {
    PlanUsage u;
    u.baby.assign(plan.b, 0);
    for (size_t c = 0; c < plan.cell.size(); ++c) {
        const int32_t t = plan.cell[c];
        if (t < 0 || terms[static_cast<size_t>(t)].zero)
            continue;
        u.baby[c % plan.b] = 1;
        u.maxGroup         = std::max(u.maxGroup, static_cast<uint32_t>(c / plan.b));
    }
    u.keySwitches = u.maxGroup;
    for (uint32_t i = 0; i < plan.b; ++i)
        if (u.baby[i] && plan.babyOffsets[i] != 0)
            ++u.keySwitches;
    return u;
}

// The non-zero rotation indices a plan performs for these terms (unsorted, may repeat).
std::vector<int32_t> RotationsOf(const Plan& plan, const std::vector<Term>& terms) {
    const PlanUsage u = UsageOf(plan, terms);
    std::vector<int32_t> out;
    for (uint32_t i = 0; i < plan.b; ++i)
        if (u.baby[i] && plan.babyOffsets[i] != 0)
            out.push_back(plan.babyOffsets[i]);
    if (u.maxGroup > 0)
        out.push_back(plan.giant);
    return out;
}

// terms: unique offsets in increasing order, zero diagonals included, so that the plan depends on
// the offsets alone and matches LinearTransformRotationIndices for the same offsets and baby step.
// keyOk(r) tells whether rotation r is available. The single hoisted group costs one key switch
// per non-zero term with a non-zero offset and is infeasible when one of those keys is missing.
// A baby/giant split is accepted when every rotation it performs has a key and it needs fewer key
// switches than that. Candidates start at the requested (or default) baby step and go down, then
// up, so that a key set generated for another baby step is still found.
template <typename KeyOk>
Plan MakePlan(const std::vector<Term>& terms, bool allowGiant, uint32_t bRequested, KeyOk keyOk) {
    const size_t m = terms.size();
    Plan single    = SingleGroupPlan(terms);
    if (m <= 2 || !allowGiant)
        return single;

    const int64_t kmin = terms.front().offset;
    int64_t s          = 0;
    for (const auto& t : terms)
        s = std::gcd(s, static_cast<int64_t>(t.offset) - kmin);
    if (s <= 0)
        return single;
    const int64_t span = (static_cast<int64_t>(terms.back().offset) - kmin) / s + 1;
    // the offsets must fill their progression reasonably densely for giant steps to pay off
    if (span > 8 * static_cast<int64_t>(m) + 64)
        return single;

    size_t singleCost = 0;
    for (const auto& t : terms) {
        if (t.zero || t.offset == 0)
            continue;
        if (!keyOk(t.offset)) {
            singleCost = std::numeric_limits<size_t>::max();
            break;
        }
        ++singleCost;
    }

    const int64_t b0 = std::min<int64_t>(bRequested ? bRequested : DefaultBabyStep(span), span - 1);
    std::vector<int64_t> candidates;
    for (int64_t b = b0; b >= 2; --b)
        candidates.push_back(b);
    for (int64_t b = b0 + 1; b < span; ++b)
        candidates.push_back(b);

    for (const int64_t b : candidates) {
        const int64_t G     = (span + b - 1) / b;
        const int64_t giant = s * b;
        if (G < 2 || giant > std::numeric_limits<int32_t>::max())
            continue;

        Plan plan;
        plan.b     = static_cast<uint32_t>(b);
        plan.G     = static_cast<uint32_t>(G);
        plan.giant = static_cast<int32_t>(giant);
        plan.babyOffsets.resize(plan.b);
        for (uint32_t i = 0; i < plan.b; ++i)
            plan.babyOffsets[i] = static_cast<int32_t>(kmin + s * static_cast<int64_t>(i));
        plan.cell.assign(static_cast<size_t>(plan.G) * plan.b, -1);
        for (size_t t = 0; t < m; ++t) {
            const int64_t idx = (static_cast<int64_t>(terms[t].offset) - kmin) / s;
            plan.cell[static_cast<size_t>(idx / b) * plan.b + static_cast<size_t>(idx % b)] =
                static_cast<int32_t>(t);
        }

        const PlanUsage usage = UsageOf(plan, terms);
        if (usage.keySwitches >= singleCost)
            continue;
        bool ok = (usage.maxGroup == 0) || keyOk(plan.giant);
        for (uint32_t i = 0; i < plan.b && ok; ++i)
            ok = !usage.baby[i] || keyOk(plan.babyOffsets[i]);
        if (ok)
            return plan;
    }
    return single;
}

// Merge equal offsets, zero-pad diagonals to the batch size (as encoding does), sort by offset.
// A merged diagonal that is all zero (or cancels out) stays as a zero term: it takes part in
// planning but is never rotated or multiplied.
std::vector<Term> NormalizeTerms(const std::vector<int32_t>& offsets,
                                 const std::vector<std::vector<double>>& diagonals, uint32_t slots) {
    std::map<int32_t, std::vector<double>> merged;
    for (size_t t = 0; t < offsets.size(); ++t) {
        if (diagonals[t].size() > slots)
            OPENFHE_THROW("EvalLinearTransform: a diagonal is longer than the batch size.");
        auto it = merged.find(offsets[t]);
        if (it == merged.end())
            it = merged.emplace(offsets[t], std::vector<double>(slots, 0.0)).first;
        for (size_t i = 0; i < diagonals[t].size(); ++i)
            it->second[i] += diagonals[t][i];
    }
    std::vector<Term> terms;
    terms.reserve(merged.size());
    for (auto& kv : merged) {
        Term term{kv.first, {}, AllZero(kv.second)};
        if (!term.zero)
            term.diag = std::move(kv.second);
        terms.push_back(std::move(term));
    }
    return terms;
}

void LogPlan(const char* algo, const Plan& plan, size_t numTerms, int64_t level) {
    if (!GetSettings().verbose)
        return;
    std::cerr << "[onp-lt] " << algo << " terms=" << numTerms << " b=" << plan.b << " G=" << plan.G
              << " giant=" << plan.giant << " level=" << level << std::endl;
}

// -----------------------------------------------------------------------------
// The three evaluation strategies
// -----------------------------------------------------------------------------

Ciphertext<DCRTPoly> EvalNaive(ConstCiphertext<DCRTPoly>& ct, const std::vector<int32_t>& offsets,
                               const std::vector<std::vector<double>>& diagonals) {
    auto cc = ct->GetCryptoContext();
    Ciphertext<DCRTPoly> result;
    for (size_t t = 0; t < offsets.size(); ++t) {
        Plaintext pt                   = cc->MakeCKKSPackedPlaintext(diagonals[t]);
        Ciphertext<DCRTPoly> ctRotated = (offsets[t] == 0) ? ct->Clone() : cc->EvalRotate(ct, offsets[t]);
        Ciphertext<DCRTPoly> ctProd    = cc->EvalMult(ctRotated, pt);
        if (!result)
            result = ctProd;
        else
            cc->EvalAddInPlace(result, ctProd);
    }
    return result;
}

Ciphertext<DCRTPoly> SumInPlace(std::vector<Ciphertext<DCRTPoly>>& parts) {
    Ciphertext<DCRTPoly> acc;
    for (auto& p : parts) {
        if (!p)
            continue;
        if (!acc) {
            acc = p;
        }
        else {
            acc->GetCryptoContext()->EvalAddInPlace(acc, p);
        }
        p.reset();
    }
    return acc;
}

Ciphertext<DCRTPoly> RotateHoisted(const CryptoContext<DCRTPoly>& cc, const Ciphertext<DCRTPoly>& x, int32_t r,
                                   const std::shared_ptr<std::vector<DCRTPoly>>& digits) {
    return (r == 0) ? x : cc->EvalFastRotation(x, static_cast<uint32_t>(r), cc->GetCyclotomicOrder(), digits);
}

Ciphertext<DCRTPoly> MultDiagonal(const Ciphertext<DCRTPoly>& rotated, const Ciphertext<DCRTPoly>& x,
                                  const std::vector<double>& diag, uint32_t slots) {
    return rotated->GetCryptoContext()->EvalMult(rotated, EncodeAtLevel(x, diag, slots));
}

void AddTo(Ciphertext<DCRTPoly>& acc, const Ciphertext<DCRTPoly>& term) {
    if (!acc)
        acc = term;
    else
        acc->GetCryptoContext()->EvalAddInPlace(acc, term);
}

// Single group (no giant steps): every rotation is hoisted from one digit decomposition;
// terms are streamed into per-thread partial sums so only O(threads) ciphertexts are alive.
Ciphertext<DCRTPoly> EvalSingleGroup(const Ciphertext<DCRTPoly>& x, const std::vector<Term>& terms,
                                     const std::shared_ptr<std::vector<DCRTPoly>>& digits, uint32_t slots) {
    auto cc                   = x->GetCryptoContext();
    const uint32_t numThreads = LinTransThreadLimit(static_cast<uint32_t>(terms.size()));
    std::vector<Ciphertext<DCRTPoly>> partial(numThreads);
    ParallelExceptionGuard guard;
#pragma omp parallel for schedule(dynamic) num_threads(numThreads)
    for (int32_t t = 0; t < static_cast<int32_t>(terms.size()); ++t) {
        if (terms[t].zero)
            continue;
        guard.Run([&] {
#ifdef _OPENMP
            const size_t tid = static_cast<size_t>(omp_get_thread_num());
#else
            const size_t tid = 0;
#endif
            auto rotated = RotateHoisted(cc, x, terms[t].offset, digits);
            AddTo(partial[tid], MultDiagonal(rotated, x, terms[t].diag, slots));
        });
    }
    guard.Rethrow();
    return SumInPlace(partial);
}

Ciphertext<DCRTPoly> EvalHoisted(const Ciphertext<DCRTPoly>& x, const std::vector<Term>& terms, const Plan& plan,
                                 uint32_t slots) {
    auto cc          = x->GetCryptoContext();
    const uint32_t b = plan.b;
    const uint32_t G = plan.G;

    auto digits = cc->EvalFastRotationPrecompute(x);

    Ciphertext<DCRTPoly> acc;
    if (G == 1) {
        acc = EvalSingleGroup(x, terms, digits, slots);
    }
    else {
        const PlanUsage usage         = UsageOf(plan, terms);
        const std::vector<char>& used = usage.baby;

        // Baby steps: all hoisted from the single digit decomposition
        std::vector<Ciphertext<DCRTPoly>> baby(b);
        ParallelExceptionGuard babyGuard;
#pragma omp parallel for schedule(dynamic) num_threads(LinTransThreadLimit(b))
        for (int32_t i = 0; i < static_cast<int32_t>(b); ++i) {
            if (!used[i])
                continue;
            babyGuard.Run([&] { baby[i] = RotateHoisted(cc, x, plan.babyOffsets[i], digits); });
        }
        babyGuard.Rethrow();
        digits.reset();

        // Giant steps: Horner accumulation from the top group down, with diagonals pre-rotated
        // by -j*giant so that Rot_{j*giant}(Rot_{-j*giant}(d) . Rot_baby(x)) = d . Rot_{j*giant+baby}(x).
        for (int32_t j = static_cast<int32_t>(usage.maxGroup); j >= 0; --j) {
            const int64_t shift = -static_cast<int64_t>(j) * plan.giant;
            std::vector<Ciphertext<DCRTPoly>> prods(b);
            ParallelExceptionGuard groupGuard;
#pragma omp parallel for schedule(dynamic) num_threads(LinTransThreadLimit(b))
            for (int32_t i = 0; i < static_cast<int32_t>(b); ++i) {
                const int32_t t = plan.cell[static_cast<size_t>(j) * b + i];
                if (t < 0 || terms[t].zero)
                    continue;
                groupGuard.Run([&] {
                    const std::vector<double> diag =
                        (shift == 0) ? terms[t].diag : RotateLeftCyclic(terms[t].diag, shift);
                    prods[i] = MultDiagonal(baby[i], x, diag, slots);
                });
            }
            groupGuard.Rethrow();
            Ciphertext<DCRTPoly> inner = SumInPlace(prods);

            if (acc)
                acc = cc->EvalRotate(acc, plan.giant);
            if (inner)
                AddTo(acc, inner);
        }
    }
    return acc;
}

}  // namespace

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

void SetLinTransAlgo(LinTransAlgo algo) {
    GetSettings().algo.store(static_cast<uint8_t>(algo));
}

LinTransAlgo GetLinTransAlgo() {
    return static_cast<LinTransAlgo>(GetSettings().algo.load());
}

void SetLinTransBabyStep(uint32_t bStep) {
    GetSettings().bStep.store(bStep);
}

uint32_t GetLinTransBabyStep() {
    return GetSettings().bStep.load();
}

void SetLinTransThreads(uint32_t numThreads) {
    GetSettings().threads.store(numThreads);
}

uint32_t GetLinTransThreads() {
    return GetSettings().threads.load();
}

uint32_t LinTransThreadLimit(uint32_t work) {
#ifdef _OPENMP
    uint32_t t = GetSettings().threads.load();
    if (t == 0)
        t = static_cast<uint32_t>(omp_get_max_threads());
#else
    uint32_t t = 1;
#endif
    if (work < GetSettings().parallelMin)
        return 1;
    return std::max<uint32_t>(1, std::min<uint32_t>(t, std::max<uint32_t>(work, 1)));
}

bool UseCanonicalRowShifts(size_t slots, uint32_t numCols) {
    const size_t d = static_cast<size_t>(numCols);
    return GetSettings().canonical.load() && d > 0 && slots == d * d;
}

void SetCanonicalRowShiftKeys(bool prune) {
    GetSettings().canonical.store(prune);
}

bool GetCanonicalRowShiftKeys() {
    return GetSettings().canonical.load();
}

bool UseCanonicalRowShifts(ConstCiphertext<DCRTPoly>& ciphertext, uint32_t numCols) {
    const size_t d     = static_cast<size_t>(numCols);
    const size_t slots = ciphertext->GetCryptoContext()->GetEncodingParams()->GetBatchSize();
    const size_t ctSlots = ciphertext->GetSlots();
    // the data must repeat with a period dividing the batch size
    return d > 0 && slots == d * d && (ctSlots == 0 || slots % ctSlots == 0);
}

void RequireRotationKeys(ConstCiphertext<DCRTPoly>& ciphertext, const std::vector<int32_t>& indices) {
    const auto* keyMap = TryGetKeyMap(ciphertext);
    const uint32_t M   = ciphertext->GetCryptoContext()->GetCyclotomicOrder();
    for (int32_t r : indices) {
        if (r == 0)
            continue;
        if (keyMap == nullptr || keyMap->count(FindAutomorphismIndex2nComplex(r, M)) == 0)
            OPENFHE_THROW("EvalKey for rotation index [" + std::to_string(r) + "] is not found.");
    }
}

void RequireRelinearized(ConstCiphertext<DCRTPoly>& ciphertext) {
    if (ciphertext->NumberCiphertextElements() != 2)
        OPENFHE_THROW("Linear transforms require a relinearized ciphertext (2 elements), got " +
                      std::to_string(ciphertext->NumberCiphertextElements()) + ".");
}

LinTransAlgo ResolveLinTransAlgo(ConstCiphertext<DCRTPoly>& ciphertext, LinTransAlgo requested) {
    LinTransAlgo algo = (requested == LinTransAlgo::AUTO) ? GetLinTransAlgo() : requested;
    if (algo == LinTransAlgo::AUTO)
        algo = LinTransAlgo::BSGS;
    return algo;
}

Ciphertext<DCRTPoly> PrepareLinTransInput(ConstCiphertext<DCRTPoly>& ciphertext) {
    auto x        = ciphertext->Clone();
    auto cp       = CkksParams(ciphertext);
    const auto st = cp->GetScalingTechnique();
    if (AutoRescale(st) && x->GetNoiseScaleDeg() > 1) {
        const size_t levels = static_cast<size_t>(x->GetNoiseScaleDeg() - 1) * cp->GetCompositeDegree();
        x->GetCryptoContext()->GetScheme()->ModReduceInternalInPlace(x, levels);
    }
    return x;
}

Ciphertext<DCRTPoly> EvalLinearTransform(ConstCiphertext<DCRTPoly>& ciphertext, const std::vector<int32_t>& offsets,
                                         const std::vector<std::vector<double>>& diagonals, LinTransAlgo algo,
                                         uint32_t bStep) {
    if (offsets.empty() || offsets.size() != diagonals.size())
        OPENFHE_THROW("EvalLinearTransform: offsets and diagonals must be non-empty and of equal length.");

    const LinTransAlgo resolved = ResolveLinTransAlgo(ciphertext, algo);
    if (resolved == LinTransAlgo::NAIVE)
        return EvalNaive(ciphertext, offsets, diagonals);

    auto cc              = ciphertext->GetCryptoContext();
    const uint32_t slots = cc->GetEncodingParams()->GetBatchSize();

    RequireRelinearized(ciphertext);
    std::vector<Term> terms = NormalizeTerms(offsets, diagonals, slots);
    if (std::all_of(terms.begin(), terms.end(), [](const Term& t) { return t.zero; })) {
        // every diagonal is zero, or the diagonals cancel: an explicit zero product keeps the
        // legacy output shape (one level consumed) without reusing any input mask
        return EvalNaive(ciphertext, {0}, {std::vector<double>(slots, 0.0)});
    }

    // Pre-rotating diagonals for the giant steps is valid for any ciphertext slot count: a
    // diagonal encoded at the batch size repeats with that period in the full slot layout.
    Ciphertext<DCRTPoly> x = PrepareLinTransInput(ciphertext);

    const auto* keyMap = TryGetKeyMap(x);
    const uint32_t M   = cc->GetCyclotomicOrder();
    auto keyOk         = [&](int32_t r) {
        return r == 0 || (keyMap && keyMap->count(FindAutomorphismIndex2nComplex(r, M)) > 0);
    };
    const uint32_t b = bStep ? bStep : GetLinTransBabyStep();
    Plan plan        = MakePlan(terms, keyMap != nullptr, b, keyOk);
    LogPlan("BSGS", plan, terms.size(), x->GetLevel());

    // check every key serially: an error inside the parallel loops would only be rethrown later
    RequireRotationKeys(x, RotationsOf(plan, terms));

    return EvalHoisted(x, terms, plan, slots);
}

std::vector<int32_t> LinearTransformRotationIndices(const std::vector<int32_t>& offsets, uint32_t bStep) {
    std::vector<Term> terms;
    std::vector<int32_t> sorted(offsets);
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    for (int32_t o : sorted)
        terms.push_back(Term{o, {}});
    if (terms.empty())
        return {};
    Plan plan = MakePlan(terms, true, bStep ? bStep : GetLinTransBabyStep(), [](int32_t) { return true; });
    std::vector<int32_t> indices = RotationsOf(plan, terms);
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    return indices;
}

// -----------------------------------------------------------------------------
// HoistedCiphertext
// -----------------------------------------------------------------------------

HoistedCiphertext::HoistedCiphertext(ConstCiphertext<DCRTPoly>& ciphertext)
    : m_ct(PrepareLinTransInput(ciphertext)), m_m(0), m_slots(0) {
    RequireRelinearized(m_ct);
    auto cc   = m_ct->GetCryptoContext();
    m_m       = cc->GetCyclotomicOrder();
    m_slots   = cc->GetEncodingParams()->GetBatchSize();
    m_digits  = cc->EvalFastRotationPrecompute(m_ct);
}

Ciphertext<DCRTPoly> HoistedCiphertext::Rotate(int32_t index) const {
    if (index == 0)
        return m_ct;
    return m_ct->GetCryptoContext()->EvalFastRotation(m_ct, static_cast<uint32_t>(index), m_m, m_digits);
}

Plaintext HoistedCiphertext::Encode(const std::vector<double>& values) const {
    return EncodeAtLevel(m_ct, values, m_slots);
}

Ciphertext<DCRTPoly> HoistedCiphertext::MaskedRotationSum(const std::vector<int32_t>& offsets,
                                                          const std::vector<std::vector<double>>& diagonals) const {
    if (offsets.empty() || offsets.size() != diagonals.size())
        OPENFHE_THROW("MaskedRotationSum: offsets and diagonals must be non-empty and of equal length.");
    auto cc = m_ct->GetCryptoContext();
    Ciphertext<DCRTPoly> acc;
    for (size_t t = 0; t < offsets.size(); ++t) {
        if (AllZero(diagonals[t]))
            continue;
        Ciphertext<DCRTPoly> term = cc->EvalMult(Rotate(offsets[t]), Encode(diagonals[t]));
        if (!acc)
            acc = term;
        else
            cc->EvalAddInPlace(acc, term);
    }
    if (!acc)  // all-zero masks: an explicit zero product
        acc = cc->EvalMult(m_ct, Encode(std::vector<double>(m_slots, 0.0)));
    return acc;
}

}  // namespace openfhe_numpy
