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

#include "numpy_array_metadata.h"

using namespace lbcrypto;


namespace openfhe_numpy {

const static std::string METADATA_ARRAYINFO_TAG{"arrayInfo"};

// ---- Metadata interface -------------------------------------
std::shared_ptr<Metadata> ArrayMetadata::Clone() const {
    return std::make_shared<ArrayMetadata>(*this);
}
bool ArrayMetadata::operator==(const Metadata& rhs) const {
    auto p = dynamic_cast<const ArrayMetadata*>(&rhs);
    return p && m_initialShape == p->m_initialShape && m_ndim == p->m_ndim && m_ncols == p->m_ncols &&
           m_nrows == p->m_nrows && m_batchSize == p->m_batchSize && m_encodeType == p->m_encodeType;
}
std::ostream& ArrayMetadata::print(std::ostream& os) const {
    return os << "[shape= (" << m_initialShape[0] << ", " << m_initialShape[1] << "), ndim=" << m_ndim
              << ", cols=" << m_ncols << ", rows=" << m_nrows << ", batch=" << m_batchSize
              << ", encodeType=" << static_cast<int>(m_encodeType) << ']';
}

// ---------- template bodies (header‑only) ------------------------
template <class Archive>
inline void ArrayMetadata::save(Archive& ar, std::uint32_t) const {
    ar(cereal::base_class<lbcrypto::Metadata>(this),
       m_initialShape,
       m_ndim,
       m_ncols,
       m_nrows,
       m_batchSize,
       m_encodeType);
}
template <class Archive>
inline void ArrayMetadata::load(Archive& ar, std::uint32_t ver) {
    if (ver > SerializedVersion())
        OPENFHE_THROW("ArrayMetadata: incompatible version");
    ar(cereal::base_class<lbcrypto::Metadata>(this),
       m_initialShape,
       m_ndim,
       m_ncols,
       m_nrows,
       m_batchSize,
       m_encodeType);
}

// helper templates
std::shared_ptr<ArrayMetadata> ArrayMetadata::GetMetadata(
    const std::shared_ptr<const lbcrypto::CiphertextImpl<DCRTPoly>>& ct) {
    auto it = ct->FindMetadataByKey(METADATA_ARRAYINFO_TAG);
    if (!ct->MetadataFound(it))
        OPENFHE_THROW("ArrayMetadata not set");
    return std::dynamic_pointer_cast<ArrayMetadata>(ct->GetMetadata(it));
}
void ArrayMetadata::StoreMetadata(std::shared_ptr<lbcrypto::CiphertextImpl<DCRTPoly>> ct,
                                  std::shared_ptr<ArrayMetadata> meta) {
    ct->SetMetadataByKey(METADATA_ARRAYINFO_TAG, std::move(meta));
}

}  // namespace openfhe_numpy