#ifndef ENC_MATRIX_H
#define ENC_MATRIX_H

#include "openfhe_numpy/config.h"
#include "openfhe_numpy/helper.h"
#include "openfhe_numpy/utils.h"

#define ENC_MATRIX_API

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

using namespace lbcrypto;

//TODO: Change from const Ciphertext to ConstCiphertext
//TODO: using references

template <typename Element>
void EvalLinTransKeyGen(PrivateKey<Element>& secretKey, int32_t numCols, LinTransType type, int32_t numRepeats = 0);

template <typename Element>
void EvalSquareMatMultRotateKeyGen(PrivateKey<Element>& secretKey, int32_t numCols);

template <typename Element>
void EvalSumCumRowsKeyGen(PrivateKey<Element>& secretKey, int32_t numCols);

template <typename Element>
void EvalSumCumColsKeyGen(PrivateKey<Element>& secretKey, int32_t numCols);

template <typename Element>
Ciphertext<Element> EvalMultMatVec(MatKeys<Element> evalKeys,
                                   MatVecEncoding encodeType,
                                   int32_t numCols,
                                   const Ciphertext<Element>& ctVector,
                                   const Ciphertext<Element>& ctMatrix);

template <typename Element>
Ciphertext<Element> EvalLinTransSigma(PrivateKey<Element>& secretKey,
                                      const Ciphertext<Element>& ciphertext,
                                      int32_t numCols);

template <typename Element>
Ciphertext<Element> EvalLinTransSigma(const Ciphertext<Element>& ciphertext, int32_t numCols);

template <typename Element>
Ciphertext<Element> EvalLinTransTau(const Ciphertext<Element>& ctVector, int32_t numCols);

template <typename Element>
Ciphertext<Element> EvalLinTransTau(PrivateKey<Element>& secretKey,
                                    const Ciphertext<Element>& ciphertext,
                                    int32_t numCols);

template <typename Element>
Ciphertext<Element> EvalLinTransPhi(const Ciphertext<Element>& ctVector, int32_t numCols, int32_t numRepeats);

template <typename Element>
Ciphertext<Element> EvalLinTransPhi(PrivateKey<Element>& secretKey,
                                    const Ciphertext<Element>& ctVector,
                                    int32_t numCols,
                                    int32_t numRepeats);

template <typename Element>
Ciphertext<Element> EvalLinTransPsi(PrivateKey<Element>& secretKey,
                                    const Ciphertext<Element>& ctVector,
                                    int32_t numCols,
                                    int32_t numRepeats);

template <typename Element>
Ciphertext<Element> EvalLinTransPsi(const Ciphertext<Element>& ctVector, int32_t numCols, int32_t numRepeats);

template <typename Element>
Ciphertext<Element> EvalMatMulSquare(const Ciphertext<Element>& matrixA,
                                     const Ciphertext<Element>& matrixB,
                                     int32_t numCols);

template <typename Element>
Ciphertext<Element> EvalTranspose(PrivateKey<Element>& secretKey,
                                  const Ciphertext<Element>& ciphertext,
                                  int32_t numCols);

template <typename Element>
Ciphertext<Element> EvalTranspose(const Ciphertext<Element>& ciphertext, int32_t numCols);

template <typename Element>
Ciphertext<Element> EvalSumCumRows(const Ciphertext<Element>& ciphertext,
                                          uint32_t numCols,
                                          uint32_t numRows = 0,
                                          uint32_t slots   = 0);

template <typename Element>
Ciphertext<Element> EvalSumCumCols(const Ciphertext<Element>& ciphertext,
                                          uint32_t numCols,
                                          uint32_t subringDim = 0);

template <typename Element>
Ciphertext<Element> EvalReduceCumRows(const Ciphertext<Element>& ciphertext,
                                          uint32_t numCols,
                                          uint32_t numRows = 0,
                                          uint32_t slots   = 0);

template <typename Element>
Ciphertext<Element> EvalReduceCumCols(const Ciphertext<Element>& ciphertext,
                                          uint32_t numCols,
                                          uint32_t subringDim = 0);
}  // namespace openfhe_numpy
#endif  // ENC_MATRIX_H