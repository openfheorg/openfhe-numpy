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

#ifndef __NUMPY_ARRAY_METADATA_H__
#define __NUMPY_ARRAY_METADATA_H__

#include "metadata.h"
#include "numpy_constants.h"

#include "constants-defs.h"
#include "ciphertext.h"

using namespace lbcrypto;


// ------------------------------------------------------------------
//  ArrayMetadata declaration
// ------------------------------------------------------------------
namespace openfhe_numpy {

class ArrayMetadata : public lbcrypto::Metadata {
public:
    /* constructors / destructor */
    ArrayMetadata() = default;

    ArrayMetadata(std::array<int, 2> initialShape,
                                 int32_t ndim,
                                 int32_t rowsize,
                                 int32_t batchSize,
                                 ArrayEncodingType enc = ArrayEncodingType::ROW_MAJOR)
        : m_initialShape(initialShape), m_ndim(ndim), m_ncols(rowsize), m_batchSize(batchSize), m_encodeType(enc) {}
        
    ~ArrayMetadata() override = default;

    std::array<int, 2>& initialShape() noexcept {
        return m_initialShape;
    }
    constexpr std::array<int, 2> initialShape() const noexcept {
        return m_initialShape;
    }

    int32_t& ndim() noexcept {
        return m_ndim;
    }
    constexpr int32_t ndim() const noexcept {
        return m_ndim;
    }

    int32_t& rowsize() noexcept {
        return m_ncols;
    }
    constexpr int32_t rowsize() const noexcept {
        return m_ncols;
    }

    int32_t& nrows() noexcept {
        return m_nrows;
    }
    constexpr int32_t nrows() const noexcept {
        return m_nrows;
    }

    int32_t& batchSize() noexcept {
        return m_batchSize;
    }
    constexpr int32_t batchSize() const noexcept {
        return m_batchSize;
    }

    ArrayEncodingType& encodeType() noexcept {
        return m_encodeType;
    }
    constexpr ArrayEncodingType encodeType() const noexcept {
        return m_encodeType;
    }

    /* Metadata interface overrides */
    std::shared_ptr<lbcrypto::Metadata> Clone() const override;
    bool operator==(const lbcrypto::Metadata& rhs) const override;
    std::ostream& print(std::ostream& os) const;

    template <class Archive>
    void save(Archive& ar, std::uint32_t) const;
    template <class Archive>
    void load(Archive& ar, std::uint32_t);

    std::string SerializedObjectName() const {
        return "ArrayMetadata";
    }
    static uint32_t SerializedVersion() {
        return 1;
    }

    /* helper to attach / get from ciphertext  (must stay in header) */
    static std::shared_ptr<ArrayMetadata> GetMetadata(
        const std::shared_ptr<const lbcrypto::CiphertextImpl<DCRTPoly>>& ct);

    static void StoreMetadata(std::shared_ptr<lbcrypto::CiphertextImpl<DCRTPoly>> ct,
                              std::shared_ptr<ArrayMetadata> meta);

private:
    std::array<int, 2> m_initialShape{0};
    int32_t m_ndim{0};
    int32_t m_ncols{0};
    int32_t m_nrows{0};
    int32_t m_batchSize{0};
    ArrayEncodingType m_encodeType{ArrayEncodingType::ROW_MAJOR};
};

}  // namespace openfhe_numpy
#endif  // __NUMPY_ARRAY_METADATA_H__