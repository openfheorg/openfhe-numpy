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
static std::vector<int32_t> GenLinTransIndices(int32_t numCols, LinTransType type, int32_t numRepeats = 0) {
    if (numCols < 0) {
        OPENFHE_THROW("numCols must be positive");
    }

    if (numCols > std::numeric_limits<int32_t>::max() / 2 ||  // conservative upper bound
        numRepeats < 0 || numRepeats > std::numeric_limits<int32_t>::max() / 2) {
        OPENFHE_THROW("numCols or numRepeats too large");
    }

    std::vector<int32_t> rotationIndices;
    switch (type) {
        case LinTransType::SIGMA:
            // Generate indices from -numCols to numCols - 1
            rotationIndices.reserve(2*numCols);
            for (int32_t k = -numCols; k < numCols; ++k) {
                rotationIndices.push_back(k);
            }
            break;

        case LinTransType::TAU:
            // Generate indices: 0, numCols, 2*numCols, ..., (numCols-1)*numCols
            rotationIndices.reserve(numCols);
            for (int32_t k = 0; k < numCols; ++k) {
                rotationIndices.push_back(numCols * k);
            }
            break;

        case LinTransType::PHI:
            // Generate indices: numRepeats, numRepeats - numCols
            rotationIndices.reserve(2);
            for (int32_t k = 0; k < 2; ++k) {
                rotationIndices.push_back(numRepeats - k * numCols);
            }
            break;

        case LinTransType::PSI:
            // Generate a single index based on offset
            rotationIndices.push_back(numCols * numRepeats);
            break;

        case LinTransType::TRANSPOSE:
            // Generate indices for transposing a square matrix via diagonals
            rotationIndices.reserve(2*numCols);
            for (int32_t k = -numCols + 1; k < numCols; ++k) {
                rotationIndices.push_back((numCols - 1) * k);
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

void EvalLinTransKeyGen(PrivateKey<DCRTPoly>& secretKey, int32_t numCols, LinTransType type, int32_t numRepeats) {
    auto rotationIndices = GenLinTransIndices(numCols, type, numRepeats);
    auto cryptoContext   = secretKey->GetCryptoContext();
    cryptoContext->EvalRotateKeyGen(secretKey, rotationIndices);
}

/**
 * @brief Generates rotation keys for square matrix multiplication.
 * @param secretKey The private key used for key generation.
 * @param numCols The number of columns in the square matrix.
 */
void EvalSquareMatMultRotateKeyGen(PrivateKey<DCRTPoly>& secretKey, int32_t numCols) {
    auto indicesSigma = GenLinTransIndices(numCols, LinTransType::SIGMA);
    auto indicesTau   = GenLinTransIndices(numCols, LinTransType::TAU);

    auto cryptoContext = secretKey->GetCryptoContext();
    cryptoContext->EvalRotateKeyGen(secretKey, indicesSigma);
    cryptoContext->EvalRotateKeyGen(secretKey, indicesTau);

    for (int32_t numRepeats = 1; numRepeats < numCols; ++numRepeats) {
        auto indicesPhi = GenLinTransIndices(numCols, LinTransType::PHI, numRepeats);
        auto indicesPsi = GenLinTransIndices(numCols, LinTransType::PSI, numRepeats);

        cryptoContext->EvalRotateKeyGen(secretKey, indicesPhi);
        cryptoContext->EvalRotateKeyGen(secretKey, indicesPsi);
    }
}

/**
 * @brief Generates keys for cumulative row summation.
 * @param secretKey The private key used for key generation.
 * @param numCols The number of columns in the matrix.
 */
void EvalSumCumRowsKeyGen(PrivateKey<DCRTPoly>& secretKey, int32_t numCols) {
    auto cryptoContext = secretKey->GetCryptoContext();
    cryptoContext->EvalRotateKeyGen(secretKey, {-numCols});
}

/**
 * @brief Generates keys for cumulative column summation.
 * @param secretKey The private key used for key generation.
 * @param numCols The number of columns in the matrix.
 */
void EvalSumCumColsKeyGen(PrivateKey<DCRTPoly>& secretKey, int32_t numCols) {
    auto cryptoContext = secretKey->GetCryptoContext();
    cryptoContext->EvalRotateKeyGen(secretKey, {-1});
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

Ciphertext<DCRTPoly> EvalMultMatVec(std::shared_ptr<std::map<uint32_t, lbcrypto::EvalKey<DCRTPoly>>>& evalKeys,
                                   MatVecEncoding encodeType,
                                   int32_t numCols,
                                   const Ciphertext<DCRTPoly>& ctVector,
                                   const Ciphertext<DCRTPoly>& ctMatrix) {
    if (numCols < 0) {
        OPENFHE_THROW("numCols must be positive");
    }
    Ciphertext<DCRTPoly> ctProduct;
    auto cryptoContext = ctVector->GetCryptoContext();
    auto multiplied    = cryptoContext->EvalMult(ctMatrix, ctVector);
    if (encodeType == MatVecEncoding::MM_CRC) {
        ctProduct = cryptoContext->EvalSumCols(multiplied, numCols, *evalKeys);
    }
    else if (encodeType == MatVecEncoding::MM_RCR) {
        ctProduct = cryptoContext->EvalSumRows(multiplied, numCols, *evalKeys);
    }
    else {
        OPENFHE_THROW("Unsupported encoding style selected.");
    }

    return ctProduct;
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
                                      const Ciphertext<DCRTPoly>& ciphertext,
                                      int32_t numCols) {
    EvalLinTransKeyGen(secretKey, numCols, LinTransType::SIGMA);
    return EvalLinTransSigma(ciphertext, numCols);
}

/**
 * @brief Applies a linear transformation (Sigma) to a ciphertext without a private key.
 * @param ciphertext The input ciphertext.
 * @param numCols The number of columns in the transformation matrix.
 * @return The resulting ciphertext after the transformation.
 */
Ciphertext<DCRTPoly> EvalLinTransSigma(const Ciphertext<DCRTPoly>& ciphertext, int32_t numCols) {
    if (numCols < 0) {
        OPENFHE_THROW("numCols must be positive");
    }

    auto cryptoContext = ciphertext->GetCryptoContext();
    bool flag          = true;
    Ciphertext<DCRTPoly> ctResult;

    for (int k = -numCols; k < numCols; ++k) {
        auto diag       = GenSigmaDiag(numCols, k);  // returns std::vector<double>
        auto ptDiagonal = cryptoContext->MakeCKKSPackedPlaintext(diag);
        auto ctRotated  = cryptoContext->EvalRotate(ciphertext, k);
        auto ctProduct  = cryptoContext->EvalMult(ctRotated, ptDiagonal);
        if (flag) {
            ctResult = ctProduct;
            flag     = false;
        }
        else
            cryptoContext->EvalAddInPlace(ctResult, ctProduct);
    }

    return ctResult;
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
Ciphertext<DCRTPoly> EvalLinTransTau(const Ciphertext<DCRTPoly>& ctVector, int32_t numCols) {
    if (numCols < 0) {
        OPENFHE_THROW("numCols must be positive");
    }
    auto cryptoContext = ctVector->GetCryptoContext();
    bool flag          = true;
    Ciphertext<DCRTPoly> ctResult;

    int32_t slots = cryptoContext->GetEncodingParams()->GetBatchSize();
    for (auto k = 0; k < numCols; ++k) {
        auto diag       = GenTauDiag(slots, numCols, k);
        auto ptDiagonal = cryptoContext->MakeCKKSPackedPlaintext(diag);
        auto ctRotated  = cryptoContext->EvalRotate(ctVector, numCols * k);
        auto ctProduct  = cryptoContext->EvalMult(ctRotated, ptDiagonal);
        if (flag) {
            ctResult = ctProduct;
            flag     = false;
        }
        else {
            cryptoContext->EvalAddInPlace(ctResult, ctProduct);
        }
    }

    return ctResult;
}
Ciphertext<DCRTPoly> EvalLinTransTau(PrivateKey<DCRTPoly>& secretKey,
                                    const Ciphertext<DCRTPoly>& ciphertext,
                                    int32_t numCols) {
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
Ciphertext<DCRTPoly> EvalLinTransPhi(const Ciphertext<DCRTPoly>& ctVector, int32_t numCols, int32_t numRepeats) {
    if (numCols < 0 or numRepeats < 0) {
        OPENFHE_THROW("numCols must be positive");
    }
    auto cryptoContext = ctVector->GetCryptoContext();
    bool flag          = true;
    Ciphertext<DCRTPoly> ctResult;

    for (auto i = 0; i < 2; ++i) {
        auto rotateIdx  = numRepeats - i * numCols;
        auto diag       = GenPhiDiag(numCols, numRepeats, i);
        auto ptDiagonal = cryptoContext->MakeCKKSPackedPlaintext(diag);
        auto ctRotated  = cryptoContext->EvalRotate(ctVector, rotateIdx);
        auto ctProduct  = cryptoContext->EvalMult(ctRotated, ptDiagonal);
        if (flag) {
            ctResult = ctProduct;
            flag     = false;
        }
        else
            cryptoContext->EvalAddInPlace(ctResult, ctProduct);
    }

    return ctResult;
}
Ciphertext<DCRTPoly> EvalLinTransPhi(PrivateKey<DCRTPoly>& secretKey,
                                    const Ciphertext<DCRTPoly>& ctVector,
                                    int32_t numCols,
                                    int32_t numRepeats) {
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
                                    const Ciphertext<DCRTPoly>& ctVector,
                                    int32_t numCols,
                                    int32_t numRepeats) {
    EvalLinTransKeyGen(secretKey, numCols, LinTransType::PSI, numRepeats);
    return EvalLinTransPsi(ctVector, numCols, numRepeats);
}
Ciphertext<DCRTPoly> EvalLinTransPsi(const Ciphertext<DCRTPoly>& ctVector, int32_t numCols, int32_t numRepeats) {
    if (numCols < 0 or numRepeats < 0) {
        OPENFHE_THROW("numCols must be positive");
    }
    auto cryptoContext = ctVector->GetCryptoContext();
    return cryptoContext->EvalRotate(ctVector, numCols * numRepeats);
}

/**
 * @brief Multiplies two square matrices represented as ciphertexts.
 * (based on https://eprint.iacr.org/2018/1041)
 * @param matrixA The first matrix ciphertext.
 * @param matrixB The second matrix ciphertext.
 * @param numCols The number of columns in the matrices.
 * @return The resulting ciphertext after the matrix multiplication.
 */
Ciphertext<DCRTPoly> EvalMatMulSquare(const Ciphertext<DCRTPoly>& matrixA,
                                     const Ciphertext<DCRTPoly>& matrixB,
                                     int32_t numCols) {
    if (numCols < 0) {
        OPENFHE_THROW("numCols must be positive");
    }
    auto cryptoContext               = matrixA->GetCryptoContext();
    Ciphertext<DCRTPoly> transformedA = EvalLinTransSigma(matrixA, numCols);
    Ciphertext<DCRTPoly> transformedB = EvalLinTransTau(matrixB, numCols);
    Ciphertext<DCRTPoly> ctProduct    = cryptoContext->EvalMult(transformedA, transformedB);

    for (auto k = 1; k < numCols; ++k) {
        auto transformedAk = EvalLinTransPhi(transformedA, numCols, k);
        auto transformedBk = EvalLinTransPsi(transformedB, numCols, k);
        ctProduct          = cryptoContext->EvalAdd(ctProduct, cryptoContext->EvalMult(transformedAk, transformedBk));
    }

    return ctProduct;
}

/**
 * @brief Computes the transpose of a ciphertext matrix using a private key.
 * @param secretKey The private key used for the operation.
 * @param ciphertext The input ciphertext matrix.
 * @param numCols The number of columns in the matrix.
 * @return The resulting ciphertext after the transpose operation.
 */
Ciphertext<DCRTPoly> EvalTranspose(PrivateKey<DCRTPoly>& secretKey,
                                  const Ciphertext<DCRTPoly>& ciphertext,
                                  int32_t numCols) {
    EvalLinTransKeyGen(secretKey, numCols, LinTransType::TRANSPOSE);
    return EvalTranspose(ciphertext, numCols);
}
Ciphertext<DCRTPoly> EvalTranspose(const Ciphertext<DCRTPoly>& ciphertext, int32_t numCols) {
    try {
        if (numCols < 0) {
            OPENFHE_THROW("numCols must be positive");
        }
        auto cryptoContext = ciphertext->GetCryptoContext();
        uint32_t slots     = cryptoContext->GetEncodingParams()->GetBatchSize();
        bool flag          = true;
        Ciphertext<DCRTPoly> ctResult;

        for (int32_t index = -numCols + 1; index < numCols; ++index) {
            int32_t rotationIndex = (numCols - 1) * index;
            auto diagonalVector   = GenTransposeDiag(slots, numCols, index);
            auto ptDiagonal       = cryptoContext->MakeCKKSPackedPlaintext(diagonalVector);
            auto ctRotated        = cryptoContext->EvalRotate(ciphertext, rotationIndex);
            auto ctProduct        = cryptoContext->EvalMult(ctRotated, ptDiagonal);
            if (flag) {
                ctResult = ctProduct;
                flag     = false;
            }
            else
                cryptoContext->EvalAddInPlace(ctResult, ctProduct);
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
std::vector<std::complex<double>> GenMaskSumCols(int k, int slots, int numCols) {
    if (numCols < 0 or slots < 0) {
        OPENFHE_THROW("parameters must be positive");
    }
    auto n = (int)(slots / numCols);

    std::vector<std::complex<double>> result(slots, 0);

    for (int i = 0; i < n; ++i) {
        result[i * numCols + k] = 1.0;
    }
    return result;
};

std::vector<std::complex<double>> GenMaskSumRows(int k, int slots, int numRows, int numCols) {
    if (numCols < 0 or slots < 0 or numRows < 0) {
        OPENFHE_THROW("parameters must be positive");
    }
    auto blockSize = numCols * numRows;
    auto n         = slots / blockSize;
    std::vector<std::complex<double>> mask(slots, 0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < numCols; j++) {
            if (i * blockSize + numCols * k + j < slots)
                mask[i * blockSize + numCols * k + j] = 1;
        }
    }
    return mask;
};

std::size_t MulDepthAccumulation(std::size_t numRows, std::size_t numCols, bool isSumRows) {
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

Ciphertext<DCRTPoly> EvalSumCumCols(const Ciphertext<DCRTPoly>& ciphertext, uint32_t numCols, uint32_t subringDim) {
    if (ciphertext->GetEncodingType() != CKKS_PACKED_ENCODING)
        OPENFHE_THROW("Matrix summation of row-vectors is only supported for CKKS packed encoding.");

    if (numCols < 0 or subringDim < 0) {
        OPENFHE_THROW("parameters must be positive");
    }

    const auto cryptoParams = ciphertext->GetCryptoParameters();
    const auto cc           = ciphertext->GetCryptoContext();

    subringDim = (subringDim == 0) ? cryptoParams->GetElementParams()->GetCyclotomicOrder() / 4 : subringDim;

    std::vector<std::complex<double>> mask = GenMaskSumCols(0, subringDim, numCols);

    auto ctSum = ciphertext->Clone();

    for (size_t i = 1; i < static_cast<size_t>(numCols); ++i) {
        auto mask          = GenMaskSumCols(i, subringDim, numCols);
        auto ptmask        = cc->MakeCKKSPackedPlaintext(mask, 1, 0, nullptr, subringDim);
        auto rotated       = cc->EvalRotate(ctSum, -1);
        auto maskedRotated = cc->EvalMult(rotated, ptmask);
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
Ciphertext<DCRTPoly> EvalReduceCumCols(const Ciphertext<DCRTPoly>& ciphertext, uint32_t numCols, uint32_t subringDim) {
    if (ciphertext->GetEncodingType() != CKKS_PACKED_ENCODING)
        OPENFHE_THROW("Matrix summation of row-vectors is only supported for CKKS packed encoding.");

    const auto cryptoParams = ciphertext->GetCryptoParameters();
    const auto cc           = ciphertext->GetCryptoContext();

    if (numCols < 0 or subringDim < 0) {
        OPENFHE_THROW("parameters must be positive");
    }
    subringDim = (subringDim == 0) ? cryptoParams->GetElementParams()->GetCyclotomicOrder() / 4 : subringDim;

    std::vector<std::complex<double>> mask = GenMaskSumCols(0, subringDim, numCols);

    auto ctSum = ciphertext->Clone();

    for (size_t i = 1; i < static_cast<size_t>(numCols); ++i) {
        auto mask          = GenMaskSumCols(i, subringDim, numCols);
        auto ptmask        = cc->MakeCKKSPackedPlaintext(mask, 1, 0, nullptr, subringDim);
        auto rotated       = cc->EvalRotate(ctSum, -1);
        auto maskedRotated = cc->EvalMult(rotated, ptmask);
        cc->EvalSubInPlace(ctSum, maskedRotated);
    }
    return ctSum;
};

Ciphertext<DCRTPoly> EvalSumCumRows(const Ciphertext<DCRTPoly>& ciphertext,
                                   uint32_t numCols,
                                   uint32_t numRows,
                                   uint32_t slots) {
    if (ciphertext->GetEncodingType() != CKKS_PACKED_ENCODING)
        OPENFHE_THROW("Matrix summation of row-vectors is only supported for CKKS packed encoding.");

    if (numCols < 0 or slots < 0 or numRows < 0) {
        OPENFHE_THROW("parameters must be positive");
    }
    const auto cryptoParams   = ciphertext->GetCryptoParameters();
    const auto encodingParams = cryptoParams->GetEncodingParams();
    const auto cc             = ciphertext->GetCryptoContext();

    slots = (slots == 0) ? cryptoParams->GetElementParams()->GetCyclotomicOrder() / 4 : slots;

    if (numRows * numCols > slots)
        OPENFHE_THROW("The size of the matrix is bigger than the total slots.");

    std::vector<std::complex<double>> mask = GenMaskSumRows(0, slots, numRows, numCols);

    auto ctSum = ciphertext->Clone();

    for (size_t i = 1; i < static_cast<size_t>(numRows); ++i) {
        mask               = GenMaskSumRows(i, slots, numRows, numCols);
        auto ptmask        = cc->MakeCKKSPackedPlaintext(mask, 1, 0, nullptr, slots);
        auto rotated       = cc->EvalRotate(ctSum, -numCols);
        auto maskedRotated = cc->EvalMult(rotated, ptmask);
        cc->EvalAddInPlace(ctSum, maskedRotated);
    }
    return ctSum;
};

Ciphertext<DCRTPoly> EvalReduceCumRows(const Ciphertext<DCRTPoly>& ciphertext,
                                      uint32_t numCols,
                                      uint32_t numRows,
                                      uint32_t slots) {
    if (ciphertext->GetEncodingType() != CKKS_PACKED_ENCODING)
        OPENFHE_THROW("Matrix summation of row-vectors is only supported for CKKS packed encoding.");

    const auto cryptoParams   = ciphertext->GetCryptoParameters();
    const auto encodingParams = cryptoParams->GetEncodingParams();
    const auto cc             = ciphertext->GetCryptoContext();
    if (numCols < 0 or slots < 0 or numRows < 0) {
        OPENFHE_THROW("parameters must be positive");
    }
    if (numCols)
        slots = (slots == 0) ? cryptoParams->GetElementParams()->GetCyclotomicOrder() / 4 : slots;
    numRows = (numRows == 0) ? slots / numCols : numRows;

    std::cout << numCols << " " << numRows << " " << slots;

    if (numRows * numCols > slots)
        OPENFHE_THROW("The size of the matrix is bigger than the total slots.");

    std::vector<std::complex<double>> mask = GenMaskSumRows(0, slots, numRows, numCols);

    auto ctSum = ciphertext->Clone();

    for (size_t i = 1; i < static_cast<size_t>(numRows); ++i) {
        mask               = GenMaskSumRows(i, slots, numRows, numCols);
        auto ptmask        = cc->MakeCKKSPackedPlaintext(mask, 1, 0, nullptr, slots);
        auto rotated       = cc->EvalRotate(ctSum, -numCols);
        auto maskedRotated = cc->EvalMult(rotated, ptmask);
        cc->EvalSubInPlace(ctSum, maskedRotated);
    }
    return ctSum;
};


}  // namespace openfhe_numpy
