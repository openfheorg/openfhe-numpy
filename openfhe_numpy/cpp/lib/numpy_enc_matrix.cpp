//==============================================================================
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
//==============================================================================
#include "numpy_enc_matrix.h"

#include "numpy_utils.h"

using namespace lbcrypto;


namespace openfhe_numpy {

/**
* @brief Generate rotation indices required for linear transformation based on transformation
* type.
*
* @param numCols   The row size (number of columns) of the matrix.
* @param type The linear transformation type (SIGMA, TAU, PHI, PSI, TRANSPOSE).
* @param numRepeats   Optional offset used by PHI and PSI types.
* @return std::vector<int32_t> List of rotation indices to be used for EvalRotateKeyGen.
**/
static std::vector<int32_t> GenLinTransIndices(uint32_t numCols, LinTransType type, uint32_t numRepeats = 0) {
    if (numCols < 0) {
        OPENFHE_THROW("numCols must be positive");
    }

    if (numCols > std::numeric_limits<int32_t>::max() / 2 ||  // conservative upper bound
        numRepeats < 0 || numRepeats > std::numeric_limits<int32_t>::max() / 2) {
        OPENFHE_THROW("numCols or numRepeats too large");
    }

    std::vector<int32_t> rotationIndices;
    int32_t nCols = static_cast<int32_t>(numCols);

    switch (type) {
        case LinTransType::SIGMA:
            // Generate indices from -numCols to numCols - 1
            rotationIndices.reserve(2*nCols);
            for (int32_t k = -nCols; k < nCols; ++k) {
                rotationIndices.push_back(k);
            }
            break;

        case LinTransType::TAU:
            // Generate indices: 0, numCols, 2*numCols, ..., (numCols-1)*numCols
            rotationIndices.reserve(nCols);
            for (int32_t k = 0; k < nCols; ++k) {
                rotationIndices.push_back(nCols * k);
            }
            break;

        case LinTransType::PHI:
            // Generate indices: numRepeats, numRepeats - numCols
            rotationIndices.reserve(2);
            for (int32_t k = 0; k < 2; ++k) {
                rotationIndices.push_back(numRepeats - k * nCols);
            }
            break;

        case LinTransType::PSI:
            // Generate a single index based on offset
            rotationIndices.push_back(nCols * numRepeats);
            break;

        case LinTransType::TRANSPOSE:
            // Generate indices for transposing a square matrix via diagonals
            rotationIndices.reserve(2*nCols);
            for (int32_t k = -nCols + 1; k < nCols; ++k) {
                rotationIndices.push_back((nCols - 1) * k);
            }
            break;

        default:
            OPENFHE_THROW("Linear transformation is undefined");
            break;
    }

    return rotationIndices;
}

/**
* @brief Generates rotation keys needed for a specific linear transformation type.
*
* This function wraps the EvalRotateKeyGen call using the appropriate rotation indices
* computed via GenLinTransIndices. It ensures the crypto context is properly prepared
* for applying a matrix-based linear transformation.
*
* @param secretKey   The KeyPair<DCRTPoly> containing the secret key used to generate rotation keys.
* @param numCols   The row size of the matrix being transformed.
* @param type The type of linear transformation.
* @param numRepeats  Optional numRepeats used by PHI and PSI transformations.
**/

void EvalLinTransKeyGen(PrivateKey<DCRTPoly>& secretKey, uint32_t numCols, LinTransType type, uint32_t numRepeats) {
    std::vector<int32_t> rotationIndices = GenLinTransIndices(numCols, type, numRepeats);
    // CryptoContext<DCRTPoly>  cc   = secretKey->GetCryptoContext();
    secretKey->GetCryptoContext()->EvalRotateKeyGen(secretKey, rotationIndices);
}

/**
 * @brief Generates rotation keys for square matrix multiplication.
 * @param secretKey The private key used for key generation.
 * @param numCols The number of columns in the square matrix.
 */
void EvalSquareMatMultRotateKeyGen(PrivateKey<DCRTPoly>& secretKey, uint32_t numCols) {
    std::vector<int32_t> indicesSigma = GenLinTransIndices(numCols, LinTransType::SIGMA);
    std::vector<int32_t> indicesTau   = GenLinTransIndices(numCols, LinTransType::TAU);

    CryptoContext<DCRTPoly>  cc = secretKey->GetCryptoContext();
    cc->EvalRotateKeyGen(secretKey, indicesSigma);
    cc->EvalRotateKeyGen(secretKey, indicesTau);

    for (uint32_t numRepeats = 1; numRepeats < numCols; ++numRepeats) {
        std::vector<int32_t> indicesPhi = GenLinTransIndices(numCols, LinTransType::PHI, numRepeats);
        std::vector<int32_t> indicesPsi = GenLinTransIndices(numCols, LinTransType::PSI, numRepeats);

        cc->EvalRotateKeyGen(secretKey, indicesPhi);
        cc->EvalRotateKeyGen(secretKey, indicesPsi);
    }
}

/**
 * @brief Generates keys for cumulative row summation.
 * @param secretKey The private key used for key generation.
 * @param numCols The number of columns in the matrix.
 */
void EvalSumCumRowsKeyGen(PrivateKey<DCRTPoly>& secretKey, uint32_t numCols) {
    int32_t nCols = static_cast<int32_t>(numCols);
    secretKey->GetCryptoContext()->EvalRotateKeyGen(secretKey, {-nCols});
}

/**
 * @brief Generates keys for cumulative column summation.
 * @param secretKey The private key used for key generation.
 * @param numCols The number of columns in the matrix.
 */
void EvalSumCumColsKeyGen(PrivateKey<DCRTPoly>& secretKey, uint32_t numCols) {
    secretKey->GetCryptoContext()->EvalRotateKeyGen(secretKey, {-1});
}

/**
* @brief Performs encrypted matrix-vector multiplication using the specified
* encoding style.This function multiplies an encrypted matrix with an encrypted
* vector using homomorphic multiplication from the paper
* https://eprint.iacr.org/2018/254
*
* @param evalKeys  The evaluation keys used for rotations (row/column
* summation).
* @param encodeType The encoding strategy (e.g., MM_CRC for column-wise,
* MM_RCR for row-wise).
* @param numCols   The number of padded cols in the encoded matrix
* @param ctVector  The ciphertext encoding the input vector.
* @param ctMatrix  The ciphertext encoding the input matrix.
*
* @return The ciphertext resulting from the matrix-vector product.
*
*/

Ciphertext<DCRTPoly> EvalMultMatVec(std::shared_ptr<std::map<uint32_t,
                                    lbcrypto::EvalKey<DCRTPoly>>>& evalKeys,
                                    MatVecEncoding encodeType,
                                    uint32_t numCols,
                                    ConstCiphertext<DCRTPoly>& ctVector,
                                    ConstCiphertext<DCRTPoly>& ctMatrix) {
    if (numCols < 0) {
        OPENFHE_THROW("numCols must be positive");
    }

    CryptoContext<DCRTPoly>  cc = ctVector->GetCryptoContext();
    Ciphertext<DCRTPoly> ctMultiplied    = cc->EvalMult(ctMatrix, ctVector);
    Ciphertext<DCRTPoly> ctProd;
    if (encodeType == MatVecEncoding::MM_CRC) {
        ctProd = cc->EvalSumCols(ctMultiplied, numCols, *evalKeys);
    }
    else if (encodeType == MatVecEncoding::MM_RCR) {
        ctProd = cc->EvalSumRows(ctMultiplied, numCols, *evalKeys);
    }
    else {
        OPENFHE_THROW("Unsupported encoding style selected.");
    }

    return ctProd;
}

/**
* @brief Linear Transformation (Sigma) as described in the paper:
* https://eprint.iacr.org/2018/1041
*
* The Sigma transformation corresponds to the permutation:
*   sigma(A)_{i,j} = A_{i, i + j}
* Its matrix representation is given by:
*   U_{d·i + j, l} = 1 if l = d·i + (i + j) mod d, and 0 otherwise.
* where d is the number of columns of the matrix 0 <= i,j < d and
 * @param secretKey The private key used for the transformation.
 * @param ciphertext The input ciphertext.
 * @param numCols The number of columns in the transformation matrix.
 * @return The resulting ciphertext after the transformation.
*/

Ciphertext<DCRTPoly> EvalLinTransSigma(PrivateKey<DCRTPoly>& secretKey,
                                      ConstCiphertext<DCRTPoly>& ciphertext,
                                      uint32_t numCols) {
    EvalLinTransKeyGen(secretKey, numCols, LinTransType::SIGMA);
    return EvalLinTransSigma(ciphertext, numCols);
}

/**
 * @brief Applies a linear transformation (Sigma) to a ciphertext without a private key.
 * @param ciphertext The input ciphertext.
 * @param numCols The number of columns in the transformation matrix.
 * @return The resulting ciphertext after the transformation.
 */
Ciphertext<DCRTPoly> EvalLinTransSigma(ConstCiphertext<DCRTPoly>& ciphertext, uint32_t numCols) {
    CryptoContext<DCRTPoly>  cc  = ciphertext->GetCryptoContext();
    size_t slots                = cc->GetEncodingParams()->GetBatchSize();
    int32_t nCols = static_cast<int32_t>(numCols);
    bool flag                    = true;
    Ciphertext<DCRTPoly> ctResult;

    for (int32_t k = -nCols; k < nCols; ++k) {
        std::vector<double> diag        = GenSigmaDiag(slots,nCols, k);
        Plaintext ptDiagonal            = cc->MakeCKKSPackedPlaintext(diag);
        Ciphertext<DCRTPoly> ctRotated  = cc->EvalRotate(ciphertext, k);
        Ciphertext<DCRTPoly> ctProd     = cc->EvalMult(ctRotated, ptDiagonal);
        if (flag) {
            ctResult = ctProd;
            flag     = false;
        }
        else
            cc->EvalAddInPlace(ctResult, ctProd);
    }

    return ctResult;
}


/**
 * @brief Applies a linear transformation (Sigma) to a raw packed data.
 * @param ciphertext The input ciphertext.
 * @param numCols The number of columns in the transformation matrix.
 * @return The resulting ciphertext after the transformation.
 */
std::vector<double> EvalLinTransSigma(const std::vector<double>& vector, uint32_t numCols) {
    const size_t slots = vector.size();
    std::vector<double> result(slots, 0.0);

    const int32_t nCols = static_cast<int32_t>(numCols);
    for (int32_t k = -nCols; k < nCols; ++k) {
        int32_t shift = ((k % static_cast<int32_t>(slots)) + slots) % slots;
        std::vector<double> diag = GenSigmaDiag(slots, numCols, k);
        for (size_t i = 0; i < slots; ++i) {
            result[i] += vector[(i + shift) % slots] * diag[i];
        }
    }
    return result;
}

/**
* @brief Linear Transformation (Tau) as described in the paper:
* https://eprint.iacr.org/2018/1041
*
* The Tau transformation corresponds to the permutation:
* tau(A)_{i,j} = A_{i + j, j}
* Its matrix representation is given by:
* U_{d·i + j, l} = 1 if l = d.(i + j) mod d + j, and 0 otherwise.
*
 * @param ciphertext The input ciphertext.
 * @param numCols The number of columns in the transformation matrix.
 * @return The resulting ciphertext after the transformation.
*/
Ciphertext<DCRTPoly> EvalLinTransTau(ConstCiphertext<DCRTPoly>& ctVector, uint32_t numCols) {
    CryptoContext<DCRTPoly>  cc = ctVector->GetCryptoContext();
    bool flag          = true;
    Ciphertext<DCRTPoly> ctResult;

    const size_t slots = cc->GetEncodingParams()->GetBatchSize();

    for (size_t k = 0; k < numCols; ++k) {
        size_t shift = (static_cast<size_t>(numCols) * k) % slots;
        Ciphertext<DCRTPoly> ctRotated  = cc->EvalRotate(ctVector, shift);
        Plaintext ptDiagonal = cc->MakeCKKSPackedPlaintext(GenTauDiag(slots, numCols, k));
        Ciphertext<DCRTPoly> ctProd     = cc->EvalMult(ctRotated, ptDiagonal);

        if (flag) {
            ctResult = ctProd;
            flag     = false;
        }
        else {
            cc->EvalAddInPlace(ctResult, ctProd);
        }
    }
    return ctResult;
}


/**
* @brief LinTrans on a packed matrix
*/
std::vector<double> EvalLinTransTau(const std::vector<double>& vector, uint32_t numCols) {
    const size_t slots = vector.size();
    std::vector<double> result(slots, 0);

    for (size_t k = 0; k < numCols; ++k) {
        size_t shift = (numCols * k) % slots;
        std::vector<double> diag = GenTauDiag(slots, numCols, k);
        for (size_t i = 0; i < slots; ++i) {
            result[i] += vector[(i + shift) % slots] * diag[i];
        }
    }
    return result;
}

Ciphertext<DCRTPoly> EvalLinTransTau(PrivateKey<DCRTPoly>& secretKey,
                                    ConstCiphertext<DCRTPoly>& ciphertext,
                                    uint32_t numCols) {
    EvalLinTransKeyGen(secretKey, numCols, LinTransType::TAU);
    return EvalLinTransTau(ciphertext, numCols);
}

/**
* @brief Linear Transformation (Phi) as described in the paper:
* https://eprint.iacr.org/2018/1041
*
* The Phi transformation corresponds to the permutation:
* phi(A)_{i,j} = A_{i, j+1}
* Its k-th matrix representation is given by:
* U_{d·i + j, l}^k = 1 if l = d.i + (j + k) mod d, and 0 otherwise.
*
* @param numCols   The number of padded cols in the encoded matrix
*/
Ciphertext<DCRTPoly> EvalLinTransPhi(ConstCiphertext<DCRTPoly>& ctVector, uint32_t numCols, uint32_t numRepeats) {

    CryptoContext<DCRTPoly>  cc = ctVector->GetCryptoContext();
    const size_t slots = cc->GetEncodingParams()->GetBatchSize();

    bool flag          = true;
    Ciphertext<DCRTPoly> ctResult;

    for (size_t i = 0; i < 2; ++i) {
        int32_t rotateIdx  = numRepeats - i * numCols;
        Plaintext ptDiagonal = cc->MakeCKKSPackedPlaintext(
                                    GenPhiDiag(slots, numCols, numRepeats, i));
        Ciphertext<DCRTPoly> ctRotated  = cc->EvalRotate(ctVector, rotateIdx);
        Ciphertext<DCRTPoly> ctProd  = cc->EvalMult(ctRotated, ptDiagonal);
        if (flag) {
            ctResult = ctProd;
            flag     = false;
        }
        else
            cc->EvalAddInPlace(ctResult, ctProd);
    }

    return ctResult;
}

std::vector<double> EvalLinTransPhi(const std::vector<double>& vector, uint32_t numCols, uint32_t numRepeats) {
    const size_t slots = vector.size();
    std::vector<double> result(slots, 0.0);

    for (size_t i = 0; i < 2; ++i) {
        size_t shift = (numRepeats + slots - i * numCols) % slots;
        std::vector<double> diag = GenPhiDiag(slots, numCols, numRepeats, i);
        for (size_t j = 0; j < slots; ++j) {
            result[j] += vector[(j + shift) % slots] * diag[j];
        }
    }
    return result;
}
Ciphertext<DCRTPoly> EvalLinTransPhi(PrivateKey<DCRTPoly>& secretKey,
                                    ConstCiphertext<DCRTPoly>& ctVector,
                                    uint32_t numCols,
                                    uint32_t numRepeats) {
    EvalLinTransKeyGen(secretKey, numCols, LinTransType::PHI, numRepeats);
    return EvalLinTransPhi(ctVector, numCols, numRepeats);
}

/**
* @brief Linear Transformation (Psi) as described in the paper:
* https://eprint.iacr.org/2018/1041
*
* The Psi transformation corresponds to the permutation:
*   psi(A)_{i,j} = A_{i+1, j}
* Its k-th matrix representation is given by:
*   U_{d·i + j, l}^k = 1 if l = d.(i + k) + j mod d, and 0 otherwise.
*
* @param numCols   The number of padded cols in the encoded matrix
*/
Ciphertext<DCRTPoly> EvalLinTransPsi(PrivateKey<DCRTPoly>& secretKey,
                                    ConstCiphertext<DCRTPoly>& ctVector,
                                    uint32_t numCols,
                                    uint32_t numRepeats) {
    EvalLinTransKeyGen(secretKey, numCols, LinTransType::PSI, numRepeats);
    return EvalLinTransPsi(ctVector, numCols, numRepeats);
}

const std::vector<double> EvalLinTransPsi(const std::vector<double>& input,
                                           uint32_t numCols,
                                           uint32_t numRepeats) {
    return RotateVector<std::vector<double>>(input, numCols * numRepeats);
}

Ciphertext<DCRTPoly> EvalLinTransPsi(ConstCiphertext<DCRTPoly>& ctVector, uint32_t numCols, uint32_t numRepeats) {
    CryptoContext<DCRTPoly> cc = ctVector->GetCryptoContext();
    return cc->EvalRotate(ctVector, numCols * numRepeats);
}

/**
 * @brief Multiplies two square matrices: CT x CT.
 * Implementation is based on https://eprint.iacr.org/2018/1041)
 * @param matrixA The first ciphertext matrix.
 * @param matrixB The second ciphertext matrix.
 * @param numCols The number of columns in the matrices.
 * @return The resulting ciphertext after the matrix multiplication.
 */
Ciphertext<DCRTPoly> EvalMatMulSquare(ConstCiphertext<DCRTPoly>& matrixA,
                                      ConstCiphertext<DCRTPoly>& matrixB,
                                      uint32_t numCols) {
    CryptoContext<DCRTPoly> cc      = matrixA->GetCryptoContext();
    Ciphertext<DCRTPoly> ctSigmaA   = EvalLinTransSigma(matrixA, numCols);
    Ciphertext<DCRTPoly> ctTauB     = EvalLinTransTau(matrixB, numCols);
    Ciphertext<DCRTPoly> ctProd     = cc->EvalMult(ctSigmaA, ctTauB);

    for (uint32_t k = 1; k < numCols; ++k) {
        Ciphertext<DCRTPoly> ctPhiAk = EvalLinTransPhi(ctSigmaA, numCols, k);
        Ciphertext<DCRTPoly> ctPsiBk = EvalLinTransPsi(ctTauB, numCols, k);
        ctProd = cc->EvalAdd(ctProd, cc->EvalMult(ctPhiAk, ctPsiBk));
    }

    return ctProd;
}


/**
 * @brief Multiplies two square matrices: CT @ Raw Packed Array
 * @param ctMatA The first ciphertext matrix.
 * @param rMatB The second raw packed matrix.
 * @param numCols The number of columns in the matrices.
 * @return The resulting ciphertext after the matrix multiplication.
 */
Ciphertext<DCRTPoly> EvalMatMulSquare(ConstCiphertext<DCRTPoly>& ctMatA,
                                      const std::vector<double>& rMatB,
                                      uint32_t numCols) {

    CryptoContext<DCRTPoly> cc          = ctMatA->GetCryptoContext();
    Ciphertext<DCRTPoly> ctSigmaA       = EvalLinTransSigma(ctMatA, numCols);
    std::vector<double> rTauB           = EvalLinTransTau(rMatB, numCols);
    Plaintext ptTauB                    = cc->MakeCKKSPackedPlaintext(rTauB);
    Ciphertext<DCRTPoly> ctProd      = cc->EvalMult(ctSigmaA, ptTauB);

    for (uint32_t k = 1; k < numCols; ++k) {
        Ciphertext<DCRTPoly> ctPhiAk = EvalLinTransPhi(ctSigmaA, numCols, k);
        Plaintext ptPsiBk = cc->MakeCKKSPackedPlaintext(EvalLinTransPsi(rTauB, numCols, k));
        ctProd = cc->EvalAdd(ctProd,cc->EvalMult(ctPhiAk, ptPsiBk));
    }

    return ctProd;
}

/**
 * @brief Multiplies two square matrices: Raw Packed Array @ CT Matrix
 * (based on https://eprint.iacr.org/2018/1041)
 * @param ctMatA The first ciphertext matrix.
 * @param rMatB The second raw packed matrix.
 * @param numCols The number of columns in the matrices.
 * @return The resulting ciphertext after the matrix multiplication.
 */
Ciphertext<DCRTPoly> EvalMatMulSquare(const std::vector<double>& rMatA,
                                      ConstCiphertext<DCRTPoly>& ctMatB,
                                      uint32_t numCols) {
    CryptoContext<DCRTPoly> cc      = ctMatB->GetCryptoContext();
    std::vector<double>rSigmaA      = EvalLinTransSigma(rMatA, numCols);
    Plaintext ptSigmaA              = cc->MakeCKKSPackedPlaintext(rSigmaA);
    Ciphertext<DCRTPoly> ctTauB     = EvalLinTransTau(ctMatB, numCols);
    Ciphertext<DCRTPoly> ctProd   = cc->EvalMult(ptSigmaA, ctTauB);

    for (uint32_t k = 1; k < numCols; ++k) {
        Plaintext ptPhiAk = cc->MakeCKKSPackedPlaintext(EvalLinTransPhi(rSigmaA, numCols, k));
        Ciphertext<DCRTPoly> ctPsiBk = EvalLinTransPsi(ctTauB, numCols, k);
        ctProd = cc->EvalAdd(ctProd,cc->EvalMult(ptPhiAk, ctPsiBk));
    }

    return ctProd;
}

/**
 * @brief Multiplies two square matrices: CT @ PT
 * @param matrixA The first ciphertext matrix.
 * @param matrixB The second plaintext matrix.
 * @param numCols The number of columns in the matrices.
 * @return The resulting ciphertext after the matrix multiplication.
 */
Ciphertext<DCRTPoly> EvalMatMulSquare(ConstCiphertext<DCRTPoly>& ctMatA,
                                      ConstPlaintext& ptMatB,
                                      uint32_t numCols) {
    return EvalMatMulSquare(ctMatA, ptMatB->GetRealPackedValue(), numCols);
}

/**
 * @brief Multiplies two square matrices: PT @ CT
 * @param matrixA The first plaintext matrix.
 * @param matrixB The second ciphertext matrix.
 * @param numCols The number of columns in the matrices.
 * @return The resulting ciphertext after the matrix multiplication.
 */
Ciphertext<DCRTPoly> EvalMatMulSquare(ConstPlaintext& ptMatA,
                                      ConstCiphertext<DCRTPoly>& ctMatB,
                                     uint32_t numCols) {
    return EvalMatMulSquare(ptMatA->GetRealPackedValue(), ctMatB, numCols);
}
/**
 * @brief Computes the transpose of a ciphertext matrix using a private key.
 * @param secretKey The private key used for the operation.
 * @param ciphertext The input ciphertext matrix.
 * @param numCols The number of columns in the matrix.
 * @return The resulting ciphertext after the transpose operation.
 */
Ciphertext<DCRTPoly> EvalTranspose(PrivateKey<DCRTPoly>& secretKey,
                                  ConstCiphertext<DCRTPoly>& ciphertext,
                                  uint32_t numCols) {
    EvalLinTransKeyGen(secretKey, numCols, LinTransType::TRANSPOSE);
    return EvalTranspose(ciphertext, numCols);
}
Ciphertext<DCRTPoly> EvalTranspose(ConstCiphertext<DCRTPoly>& ciphertext, uint32_t numCols) {
    try {
        if (numCols < 0) {
            OPENFHE_THROW("numCols must be positive");
        }
        CryptoContext<DCRTPoly>  cc = ciphertext->GetCryptoContext();
        uint32_t slots     = cc->GetEncodingParams()->GetBatchSize();
        bool flag          = true;
        Ciphertext<DCRTPoly> ctResult;
        const int32_t nCols = static_cast<int32_t>(numCols);

        for (int32_t index = -nCols + 1; index < nCols; ++index) {
            int32_t rotationIndex = (numCols - 1) * index;
            auto diagonalVector   = GenTransposeDiag(slots, numCols, index);
            auto ptDiagonal       = cc->MakeCKKSPackedPlaintext(diagonalVector);
            auto ctRotated        = cc->EvalRotate(ciphertext, rotationIndex);
            auto ctProd           = cc->EvalMult(ctRotated, ptDiagonal);
            if (flag) {
                ctResult = ctProd;
                flag     = false;
            }
            else
                cc->EvalAddInPlace(ctResult, ctProd);
        }

        return ctResult;
    }
    catch (const std::exception& e) {
        OPENFHE_THROW("EvalTranspose: Homomorphic operation failed. Details: " + std::string(e.what()));
    }
};
// -------------------------------------------------------------
// EvalSumAccumulate
// -------------------------------------------------------------
// std::vector<std::complex<double>> GenMaskSumCols(uint32_t k, uint32_t slots, uint32_t numCols) {
std::vector<double> GenMaskSumCols(uint32_t k, uint32_t slots, uint32_t numCols) {
    uint32_t n = slots / numCols;
    std::vector<double> result(slots, 0);

    for (uint32_t i = 0; i < n; ++i) {
        result[i * numCols + k] = 1.0;
    }
    return result;
};

std::vector<double> GenMaskSumRows(uint32_t k, uint32_t slots, uint32_t numRows, uint32_t numCols) {

    uint32_t blockSize = numCols * numRows;
    uint32_t n         = slots / blockSize;
    std::vector<double> mask(slots, 0);

    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < numCols; j++) {
            if (i * blockSize + numCols * k + j < slots)
                mask[i * blockSize + numCols * k + j] = 1;
        }
    }
    return mask;
};

uint32_t MulDepthAccumulation(uint32_t numRows, uint32_t numCols, bool isSumRows) {
    if (isSumRows) {
        return numRows;
    }
    else
        return numCols;
};

/**
 * @brief Reduces the cumulative sum of rows in a ciphertext matrix.
 * @param ciphertext The input ciphertext matrix.
 * @param numCols The number of columns in the matrix.
 * @param numRows The number of rows in the matrix (optional).
 * @param slots The number of slots in the ciphertext (optional).
 * @return The resulting ciphertext after the reduction.
 */

Ciphertext<DCRTPoly> EvalSumCumCols(ConstCiphertext<DCRTPoly>& ciphertext, uint32_t numCols, uint32_t subringDim) {
    if (ciphertext->GetEncodingType() != CKKS_PACKED_ENCODING)
        OPENFHE_THROW("Matrix summation of row-vectors is only supported for CKKS packed encoding.");

    const auto cryptoParams = ciphertext->GetCryptoParameters();
    const auto cc           = ciphertext->GetCryptoContext();

    subringDim = (subringDim == 0) ? cryptoParams->GetElementParams()->GetCyclotomicOrder() / 4 : subringDim;

    std::vector<double> mask = GenMaskSumCols(0, subringDim, numCols);

    Ciphertext<DCRTPoly> ctSum = ciphertext->Clone();

    for (uint32_t i = 1; i < numCols; ++i) {
        std::vector<double> mask = GenMaskSumCols(i, subringDim, numCols);
        Plaintext ptmask = cc->MakeCKKSPackedPlaintext(mask, 1, 0, nullptr, subringDim);
        Ciphertext<DCRTPoly> rotated = cc->EvalRotate(ctSum, -1);
        Ciphertext<DCRTPoly> maskedRotated = cc->EvalMult(rotated, ptmask);
        cc->EvalAddInPlace(ctSum, maskedRotated);
    }
    return ctSum;
};

/**
 * @brief Computes the cumulative sum of columns in a ciphertext matrix.
 * @param ciphertext The input ciphertext matrix.
 * @param numCols The number of columns in the matrix.
 * @param subringDim The subring dimension (optional).
 * @return The resulting ciphertext after the cumulative column sum.
 */
Ciphertext<DCRTPoly> EvalReduceCumCols(ConstCiphertext<DCRTPoly>& ciphertext, uint32_t numCols, uint32_t subringDim) {
    if (ciphertext->GetEncodingType() != CKKS_PACKED_ENCODING)
        OPENFHE_THROW("Matrix summation of row-vectors is only supported for CKKS packed encoding.");

    const auto cryptoParams = ciphertext->GetCryptoParameters();
    const auto cc           = ciphertext->GetCryptoContext();

    subringDim = (subringDim == 0) ? cryptoParams->GetElementParams()->GetCyclotomicOrder() / 4 : subringDim;

    std::vector<double> mask = GenMaskSumCols(0, subringDim, numCols);

    auto ctSum = ciphertext->Clone();

    for (uint32_t i = 1; i < numCols; ++i) {
        auto mask          = GenMaskSumCols(i, subringDim, numCols);
        auto ptmask        = cc->MakeCKKSPackedPlaintext(mask, 1, 0, nullptr, subringDim);
        auto rotated       = cc->EvalRotate(ctSum, -1);
        auto maskedRotated = cc->EvalMult(rotated, ptmask);
        cc->EvalSubInPlace(ctSum, maskedRotated);
    }
    return ctSum;
};

Ciphertext<DCRTPoly> EvalSumCumRows(ConstCiphertext<DCRTPoly>& ciphertext,
                                   uint32_t numCols,
                                   uint32_t numRows,
                                   uint32_t slots) {
    if (ciphertext->GetEncodingType() != CKKS_PACKED_ENCODING)
        OPENFHE_THROW("Matrix summation of row-vectors is only supported for CKKS packed encoding.");

    const auto cryptoParams   = ciphertext->GetCryptoParameters();
    const auto encodingParams = cryptoParams->GetEncodingParams();
    const auto cc             = ciphertext->GetCryptoContext();

    slots = (slots == 0) ? cryptoParams->GetElementParams()->GetCyclotomicOrder() / 4 : slots;

    if (numRows * numCols > slots)
        OPENFHE_THROW("The size of the matrix is bigger than the total slots.");

    std::vector<double> mask = GenMaskSumRows(0, slots, numRows, numCols);

    auto ctSum = ciphertext->Clone();

    for (uint32_t i = 1; i < numRows; ++i) {
        mask               = GenMaskSumRows(i, slots, numRows, numCols);
        auto ptmask        = cc->MakeCKKSPackedPlaintext(mask, 1, 0, nullptr, slots);
        auto rotated       = cc->EvalRotate(ctSum, -numCols);
        auto maskedRotated = cc->EvalMult(rotated, ptmask);
        cc->EvalAddInPlace(ctSum, maskedRotated);
    }
    return ctSum;
};

Ciphertext<DCRTPoly> EvalReduceCumRows(ConstCiphertext<DCRTPoly>& ciphertext,
                                      uint32_t numCols,
                                      uint32_t numRows,
                                      uint32_t slots) {
    if (ciphertext->GetEncodingType() != CKKS_PACKED_ENCODING)
        OPENFHE_THROW("Matrix summation of row-vectors is only supported for CKKS packed encoding.");

    const auto cryptoParams   = ciphertext->GetCryptoParameters();
    const auto encodingParams = cryptoParams->GetEncodingParams();
    const auto cc             = ciphertext->GetCryptoContext();

    if (numCols)
        slots = (slots == 0) ? cryptoParams->GetElementParams()->GetCyclotomicOrder() / 4 : slots;

    numRows = (numRows == 0) ? slots / numCols : numRows;

    if (numRows * numCols > slots)
        OPENFHE_THROW("The size of the matrix is bigger than the total slots.");

    std::vector<double> mask = GenMaskSumRows(0, slots, numRows, numCols);

    auto ctSum = ciphertext->Clone();

    for (uint32_t i = 1; i < numRows; ++i) {
        mask               = GenMaskSumRows(i, slots, numRows, numCols);
        auto ptmask        = cc->MakeCKKSPackedPlaintext(mask, 1, 0, nullptr, slots);
        auto rotated       = cc->EvalRotate(ctSum, -numCols);
        auto maskedRotated = cc->EvalMult(rotated, ptmask);
        cc->EvalSubInPlace(ctSum, maskedRotated);
    }
    return ctSum;
};


}  // namespace openfhe_numpy
