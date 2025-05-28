#include "openfhe_numpy/enc_matrix.h"
#include "openfhe.h"
#include "openfhe_numpy/utils.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

using namespace openfhe_numpy;
using namespace lbcrypto;

/**
 * @brief Creates a CryptoContext for CKKS operations
 * 
 * @param multDepth Multiplicative depth
 * @param batchSize Batch size (default: 0)
 * @return Configured CryptoContext
 */
CryptoContext<DCRTPoly> GenerateCryptoContext(uint32_t multDepth, uint32_t batchSize = 0) {
    // Create parameter object with security level
    uint32_t ptModulus           = 0;
    uint32_t digitSize           = 0;
    uint32_t standardDeviation   = 3.19;
    SecretKeyDist secretKeyDist  = UNIFORM_TERNARY;
    uint32_t maxRelinSkDeg       = 2;
    KeySwitchTechnique ksTech    = HYBRID;
    // ScalingTechnique scalTech    = FIXEDMANUAL;
    ScalingTechnique scalTech    = FLEXIBLEAUTO;
    uint32_t firstModSize        = 60;
    batchSize                    = 512;
    uint32_t numLargeDigits      = 3;
    uint32_t multiplicativeDepth = 9;
    uint32_t scalingModSize      = 59;
    SecurityLevel securityLevel  = HEStd_NotSet;
    uint32_t ringDim             = 1024;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetRingDim(ringDim);
    parameters.SetMultiplicativeDepth(multiplicativeDepth);
    parameters.SetScalingModSize(scalingModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetFirstModSize(firstModSize);
    parameters.SetStandardDeviation(standardDeviation);
    parameters.SetSecretKeyDist(secretKeyDist);
    parameters.SetScalingTechnique(scalTech);
    parameters.SetKeySwitchTechnique(ksTech);
    parameters.SetSecurityLevel(securityLevel);
    parameters.SetNumLargeDigits(numLargeDigits);
    parameters.SetMaxRelinSkDeg(maxRelinSkDeg);
    parameters.SetDigitSize(digitSize);

    // Generate the context
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    // Enable features
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::LEVELEDSHE);
    cc->Enable(lbcrypto::ADVANCEDSHE);

    return cc;
}

/**
 * @brief Demonstrate row and column accumulation operations on encrypted matrices
 */
void DemoAccumulationOperations() {
    std::cout << "\n=== DEMO: Matrix Row & Column Accumulation ===\n";

    // Define test matrix - can be modified for different test cases
    // std::vector<std::vector<double>> matA = {
    //     {0, 7, 8, 10, 1, 2, 7, 6},
    //     {0, 1, 1, 9, 7, 5, 1, 7},
    //     {8, 8, 4, 5, 8, 2, 6, 1},
    //     {1, 0, 0, 1, 10, 3, 1, 7},
    //     {7, 8, 2, 5, 3, 2, 10, 9},
    //     {0, 3, 4, 10, 10, 5, 2, 5},
    //     {2, 5, 0, 2, 8, 8, 5, 9},
    //     {5, 1, 10, 6, 2, 8, 6, 3},
    // };
    std::vector<std::vector<double>> matA = {{6.808982, 5.7441207}, {3.20381916, 9.88113614}};

    // Initialize crypto system
    std::cout << "Initializing cryptosystem...\n";
    CryptoContext<DCRTPoly> cc = GenerateCryptoContext(20);
    std::size_t ringDim        = cc->GetRingDimension();
    std::cout << "CKKS scheme using ring dimension " << ringDim << "\n";

    // Generate keys
    std::cout << "Generating keys...\n";
    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    cc->EvalSumKeyGen(keyPair.secretKey);

    // Matrix dimensions and padding
    auto nRows     = matA.size();
    auto nCols     = matA[0].size();
    auto batchSize = ringDim / 2;

    // Pad to power of 2 dimensions for efficient operations
    auto paddedRows = NextPow2(nRows);
    auto paddedCols = NextPow2(nCols);

    // Display matrix parameters
    std::cout << "\nMatrix Parameters:\n"
              << "  Original: " << nRows << " × " << nCols << "\n"
              << "  Padded:   " << paddedRows << " × " << paddedCols << "\n"
              << "  Slots:    " << batchSize << "\n\n";

    // Display original matrix for reference
    std::cout << "Original Matrix:\n";
    for (const auto& row : matA) {
        for (double val : row) {
            std::cout << std::setw(4) << val;
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Encoding the matrix (row-major format)
    std::vector<double> encMatA = EncodeMatrix<double>(matA, batchSize);

    // Encrypt the matrix
    Plaintext ptMatA = cc->MakeCKKSPackedPlaintext(encMatA);
    auto ciphertext  = cc->Encrypt(keyPair.publicKey, ptMatA);

    // ======= ROW ACCUMULATION OPERATION =======
    std::cout << "\n--- Computing Encrypted Row Sums ---\n";

    TimeVar t;

    // Generate rotation keys for row operations
    std::cout << "\n--- 1. Generate rotation keys for row operations ---\n";
     std::cout << "\n--- paddedCols = "<<paddedCols;
     std::cout << "\n--- paddedRows = "<<paddedRows;
     std::cout << "\n--- batchSize = "<<batchSize;
    TIC(t);
    EvalSumCumRowsKeyGen(keyPair.secretKey, paddedCols);
    double timeKeyGen = TOC(t);

    // Perform homomorphic row accumulation
    std::cout << "\n--- 2.  Perform homomorphic row accumulation ---\n";
    TIC(t);
    auto ctRowSums  = EvalSumCumRows(ciphertext, paddedCols, paddedRows, batchSize);
    double timeEval = TOC(t);

    // Decrypt and display results
    std::cout << "\n--- 3.  Decrypt and display results ---\n";
    Plaintext ptResult;
    cc->Decrypt(keyPair.secretKey, ctRowSums, &ptResult);
    ptResult->SetLength(paddedCols * paddedRows);
    auto rowSumVector = ptResult->GetRealPackedValue();

    std::cout << "Row Sum Results:\n";
    RoundVector(rowSumVector);

    // Format output as matrix for easier reading
    for (std::size_t i = 0; i < paddedRows; i++) {
        for (std::size_t j = 0; j < paddedCols; j++) {
            if (i < nRows && j < nCols) {
                // Only show non-padding values
                std::cout << std::setw(4) << rowSumVector[i * paddedCols + j];
            }
        }
        std::cout << "\n";
    }

    std::cout << "\nRow Accumulation Time (KeyGen): " << timeKeyGen << " ms\n";
    std::cout << "Row Accumulation Time (Eval): " << timeEval << " ms\n";

    // ======= COLUMN ACCUMULATION OPERATION =======
    std::cout << "\n--- Computing Encrypted Column Sums ---\n";

    // Generate rotation keys for column operations
    TIC(t);
    EvalSumCumColsKeyGen(keyPair.secretKey, paddedCols);
    timeKeyGen = TOC(t);

    // Perform homomorphic column accumulation
    TIC(t);
    auto ctColSums = EvalSumCumCols(ciphertext, paddedCols);
    timeEval       = TOC(t);

    // Decrypt and display results
    cc->Decrypt(keyPair.secretKey, ctColSums, &ptResult);
    ptResult->SetLength(paddedCols * paddedRows);
    auto colSumVector = ptResult->GetRealPackedValue();

    std::cout << "Column Sum Results:\n";
    RoundVector(colSumVector);

    // Format output as matrix for easier reading
    for (std::size_t i = 0; i < paddedRows; i++) {
        for (std::size_t j = 0; j < paddedCols; j++) {
            if (i < nRows && j < nCols) {
                // Only show non-padding values
                std::cout << std::setw(4) << colSumVector[i * paddedCols + j];
            }
        }
        std::cout << "\n";
    }
    std::cout << "\nColumn Accumulation Time (KeyGen): " << timeKeyGen << " ms\n";
    std::cout << "Column Accumulation Time (Eval): " << timeEval << " ms\n";

    std::cout << "\nDemo Complete!\n";
}

/**
 * @brief Demonstrates matrix transpose operation on encrypted data
 */
void DemoMatrixTranspose() {
    std::cout << "\n=== DEMO: Matrix Transpose Operation ===\n";

    // Define test matrix
    // std::vector<std::vector<double>> matA = {
    //     {1, 1, 1, 0},
    //     {2, 2, 2, 0},
    //     {3, 3, 3, 0},
    //     {4, 4, 4, 0},
    // };

    std::vector<std::vector<double>> matA = {
        {0, 7, 8, 10, 1, 2, 7, 6},
        {0, 1, 1, 9, 7, 5, 1, 7},
        {8, 8, 4, 5, 8, 2, 6, 1},
        {1, 0, 0, 1, 10, 3, 1, 7},
        {7, 8, 2, 5, 3, 2, 10, 9},
        {0, 3, 4, 10, 10, 5, 2, 5},
        {2, 5, 0, 2, 8, 8, 5, 9},
        {5, 1, 10, 6, 2, 8, 6, 3},
    };

    printf("\nMatrix: \n");
    PrintMatrix(matA);

    // Initialize crypto system
    std::cout << "Initializing cryptosystem...\n";
    CryptoContext<DCRTPoly> cc = GenerateCryptoContext(20);
    std::size_t ringDim        = cc->GetRingDimension();
    std::cout << "CKKS scheme using ring dimension " << ringDim << "\n";

    // Generate keys
    std::cout << "Generating keys...\n";
    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    cc->EvalSumKeyGen(keyPair.secretKey);

    // Matrix dimensions and padding
    auto nRows     = matA.size();
    auto nCols     = matA[0].size();
    auto batchSize = ringDim / 2;

    // Pad to power of 2 dimensions for efficient operations
    auto paddedRows = NextPow2(nRows);
    auto paddedCols = NextPow2(nCols);

    // Display matrix parameters
    std::cout << "\nMatrix Parameters:\n"
              << "  Original: " << nRows << " × " << nCols << "\n"
              << "  Padded:   " << paddedRows << " × " << paddedCols << "\n"
              << "  Slots:    " << batchSize << "\n\n";

    // Display original matrix for reference
    std::cout << "Original Matrix:\n";
    PrintMatrix(matA);

    // Encoding the matrix
    std::vector<double> encMatA = EncodeMatrix<double>(matA, batchSize);

    // Encrypt the matrix
    Plaintext ptMatA = cc->MakeCKKSPackedPlaintext(encMatA);
    auto ctMatA      = cc->Encrypt(keyPair.publicKey, ptMatA);

    // ======= MATRIX TRANSPOSE OPERATION =======
    std::cout << "\n--- Computing Encrypted Matrix Transpose ---\n";

    TimeVar t;

    // Perform homomorphic transpose
    TIC(t);
    auto encryptedTranspose = EvalTranspose(keyPair.secretKey, ctMatA, paddedCols);
    double timeEval         = TOC(t);

    // Decrypt and display results
    Plaintext ptResult;
    cc->Decrypt(keyPair.secretKey, encryptedTranspose, &ptResult);
    ptResult->SetLength(paddedCols * paddedRows);
    auto transposeVector = ptResult->GetRealPackedValue();

    std::cout << "Transposed Matrix Result:\n";
    RoundVector(transposeVector);

    // Format output as matrix for easier reading
    for (std::size_t i = 0; i < nCols; i++) {
        for (std::size_t j = 0; j < nRows; j++) {
            // Only show non-padding values
            std::cout << std::setw(4) << transposeVector[i * nRows + j];
        }
        std::cout << "\n";
    }

    std::cout << "Transpose Operation Time (Eval): " << timeEval << " ms\n";

    std::cout << "\nDemo Complete!\n";
}

/**
 * @brief Main function with menu for selecting demos
 */
int main(int argc, char* argv[]) {
    int choice = 0;

    if (argc > 1) {
        choice = atoi(argv[1]);
    }
    else {
        std::cout << "OpenFHE Matrix Operations Demo\n"
                  << "-------------------------------\n"
                  << "1. Row & Column Accumulation\n"
                  << "2. Matrix Transpose\n"
                  << "Enter choice (default=1): ";
        std::cin >> choice;
    }

    switch (choice) {
        case 2:
            DemoMatrixTranspose();
            break;
        case 1:
        default:
            DemoAccumulationOperations();
            break;
    }

    return 0;
}