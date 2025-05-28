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
 * @brief Generate a CKKS crypto context with specified parameters
 * 
 * @param multDepth Multiplicative depth
 * @param batchSize Optional batch size (default: 0)
 * @return CryptoContext<DCRTPoly> Configured crypto context
 */
CryptoContext<DCRTPoly> GenerateCryptoContext(uint32_t multDepth, uint32_t batchSize = 0) {
    // Step 1: Setup CryptoContext

    // A. Specify main parameters

    /* A2) Bit-length of scaling factor.
    * CKKS works for real numbers, but these numbers are encoded as integers.
    * For instance, real number m=0.01 is encoded as m'=round(m*D), where D is
    * a scheme parameter called scaling factor. Suppose D=1000, then m' is 10 (an
    * integer). Say the result of a computation based on m' is 130, then at
    * decryption, the scaling factor is removed so the user is presented with
    * the real number result of 0.13.
    */
    uint32_t scaleModSize = 50;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    // Enable the features that you wish to use
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    return cc;
}

/**
 * @brief Demonstrate matrix-vector multiplication using CRC encoding
 */
void MatrixVectorProduct_CRC() {
    std::cout << "=== DEMO: Matrix-Vector Product with CRC Encoding ===" << std::endl;

    // Input plaintext data
    // std::vector<std::vector<double>> inputMatrix = {
    //     {1, 1, 1, 0},
    //     {2, 2, 2, 0},
    // };
    // std::vector<double> inputVector = {1, 2, 3, 0};
    // std::vector<double> flatMat     = {1, 1, 1, 0, 2, 2, 2, 0};
    // std::vector<double> flatVec     = {1, 2, 3, 0, 1, 2, 3, 0};

    std::vector<std::vector<double>> inputMatrix = {{0, 7, 8, 10, 1, 2, 7, 6},
                                                    {0, 1, 1, 9, 7, 5, 1, 7},
                                                    {8, 8, 4, 5, 8, 2, 6, 1},
                                                    {1, 0, 0, 1, 10, 3, 1, 7},
                                                    {7, 8, 2, 5, 3, 2, 10, 9},
                                                    {0, 3, 4, 10, 10, 5, 2, 5},
                                                    {2, 5, 0, 2, 8, 8, 5, 9},
                                                    {5, 1, 10, 6, 2, 8, 6, 3}};

    std::vector<double> inputVector = {7, 0, 1, 3, 5, 0, 1, 8};

    // Result: 98.00 120.00 129.00 117.00 163.00 126.00 137.00 103.00 


    uint multDepth = 10 ;

    printf("\nMatrix: \n");
    PrintMatrix(inputMatrix);

    printf("\nVector: \n");
    PrintVector(inputVector);

    std::cout << "Initializing CryptoContext...\n";
    TimeVar t_setup;
    TIC(t_setup);
    CryptoContext<DCRTPoly> cc = GenerateCryptoContext(multDepth);
    double time_setup          = TOC(t_setup);
    std::cout << "Setup time: " << time_setup << " ms" << std::endl;

    // Generate keys
    std::cout << "Generating keys...\n";
    TimeVar t_keygen;
    TIC(t_keygen);
    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    auto sumColsKey    = cc->EvalSumColsKeyGen(keyPair.secretKey);
    double time_keygen = TOC(t_keygen);
    std::cout << "Key generation time: " << time_keygen << " ms" << std::endl;

    // Encode and encrypt mat and vector

    std::size_t nRows          = inputMatrix.size();
    std::size_t nCols          = !inputMatrix.empty() ? inputMatrix[0].size() : 0;
    std::size_t paddedRowCount = NextPow2(nCols);
    
    std::size_t batchSize         = cc->GetRingDimension() / 2;

    std::vector<double> flatMat = EncodeMatrix<double>(inputMatrix, batchSize);
    std::vector<double> flatVec = PackVecColWise<double>(inputVector, nCols, batchSize);


    print_range(flatMat, 0 , 32);
    print_range(flatVec, 0 , 32);

    std::cout << "Encrypting inputs...\n";
    TimeVar t_encrypt;
    TIC(t_encrypt);
    auto ptMat          = cc->MakeCKKSPackedPlaintext(flatMat);
    auto ptVec          = cc->MakeCKKSPackedPlaintext(flatVec);
    auto ctMat          = cc->Encrypt(keyPair.publicKey, ptMat);
    auto ctVec          = cc->Encrypt(keyPair.publicKey, ptVec);
    double time_encrypt = TOC(t_encrypt);
    std::cout << "Encryption time: " << time_encrypt << " ms" << std::endl;

    std::cout << "\n--- Plaintext Matrix-Vector Product ---\n";
    PrintVector(MulMatVec(inputMatrix, inputVector));

    // Perform encrypted mat-vector multiplication
    std::cout << "\nPerforming homomorphic matrix-vector multiplication...";
    TimeVar t_mult;
    TIC(t_mult);
    auto ctProd      = EvalMultMatVec(sumColsKey, MatVecEncoding::MM_CRC, paddedRowCount, ctVec, ctMat);
    double time_mult = TOC(t_mult);
    std::cout << "\nHomomorphic multiplication time: " << time_mult << " ms" << std::endl;

    // Decrypt result
    std::cout << "Decrypting result...\n";
    TimeVar t_decrypt;
    TIC(t_decrypt);
    Plaintext ptResult;
    cc->Decrypt(keyPair.secretKey, ctProd, &ptResult);
    ptResult->SetLength(nCols * nRows);
    std::vector<double> resultVector = ptResult->GetRealPackedValue();
    double time_decrypt              = TOC(t_decrypt);

    std::cout << "--- Homomorphic Computation Result ---\n";
    PrintVector(resultVector);
    std::cout << "Decryption time: " << time_decrypt << " ms" << std::endl;
    std::cout << "Matrix-Vector Demo Complete.\n";
}

/**
 * @brief Demonstrate homomorphic matrix-matrix multiplication
 * 
 * @param verbose Whether to print detailed output (default: true)
 */
void DemoMatrixProduct(bool verbose = true) {
    std::cout << "\n=== DEMO: Matrix-Matrix Multiplication ===" << std::endl;

    std::vector<std::vector<double>> matA = {
        {1, 1, 1, 0},
        {2, 2, 2, 0},
        {3, 3, 3, 0},
        {4, 4, 4, 0},
    };

    std::vector<std::vector<double>> matB = {
        {1, 0, 1, 0},
        {1, 1, 0, 0},
        {3, 0, 3, 0},
        {3, 0, 2, 0},
    };

    printf("\nMatrix A: \n");
    PrintMatrix(matA);

    printf("\nMatrix B: \n");
    PrintMatrix(matB);

    // Set CKKS encryption parameters
    std::cout << "\nInitializing CryptoContext ...\n";
    TimeVar t_setup;
    TIC(t_setup);
    CryptoContext<DCRTPoly> cc = GenerateCryptoContext(20);
    usint ringDim              = cc->GetRingDimension();
    double time_setup          = TOC(t_setup);
    std::cout << "Setup time: " << time_setup << " ms" << std::endl;

    // Generate keys
    std::cout << "\nGenerating key pair ...\n";
    TimeVar t_keygen;
    TIC(t_keygen);
    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    cc->EvalSumKeyGen(keyPair.secretKey);
    double time_keygen = TOC(t_keygen);
    std::cout << "Key generation time: " << time_keygen << " ms" << std::endl;

    // Encrypt
    auto nRows          = matA.size();
    auto nCols          = matA[0].size();
    auto paddedRowCount = NextPow2(nCols);
    auto batchSize      = ringDim / 2;

    std::vector<double> encMatA = EncodeMatrix<double>(matA, batchSize);
    std::vector<double> encMatB = EncodeMatrix<double>(matB, batchSize);

    std::cout << "\nMatrix Parameters ...\n";
    std::cout << "  nRows = " << nRows << std::endl;
    std::cout << "  nCols = " << nCols << std::endl;
    std::cout << "  paddedRowCount = " << paddedRowCount << std::endl;
    std::cout << "  batchSize = " << batchSize << std::endl;

    // Encrypt matrices
    std::cout << "Encrypting matrices...\n";
    TimeVar t_encrypt;
    TIC(t_encrypt);
    Plaintext ptMatA    = cc->MakeCKKSPackedPlaintext(encMatA);
    Plaintext ptMatB    = cc->MakeCKKSPackedPlaintext(encMatB);
    auto ctMatA         = cc->Encrypt(keyPair.publicKey, ptMatA);
    auto ctMatB         = cc->Encrypt(keyPair.publicKey, ptMatB);
    double time_encrypt = TOC(t_encrypt);
    std::cout << "Encryption time: " << time_encrypt << " ms" << std::endl;

    if (verbose) {
        std::cout << "\n--- [Plaintext] Matrix Product Result ---\n";
        PrintMatrix(MulMats(matA, matB));
    }

    // Generate rotation keys for matrix multiplication
    std::cout << "\nGenerating rotation keys for matrix multiplication...\n";
    TimeVar t_rotkey;
    TIC(t_rotkey);
    EvalSquareMatMultRotateKeyGen(keyPair.secretKey, paddedRowCount);
    double time_rotkey = TOC(t_rotkey);
    std::cout << "Rotation key generation time: " << time_rotkey << " ms" << std::endl;

    // Perform encrypted mat-mat multiplication
    std::cout << "Performing homomorphic matrix multiplication...\n";
    TimeVar t_matmul;
    TIC(t_matmul);
    auto ctProduct     = EvalMatMulSquare(ctMatA, ctMatB, paddedRowCount);
    double time_matmul = TOC(t_matmul);
    std::cout << "Homomorphic matrix multiplication time: " << time_matmul << " ms" << std::endl;

    // Decrypt result
    std::cout << "Decrypting result...\n";
    TimeVar t_decrypt;
    TIC(t_decrypt);
    Plaintext ptResult;
    cc->Decrypt(keyPair.secretKey, ctProduct, &ptResult);
    ptResult->SetLength(nCols * nRows);
    std::vector<double> res = ptResult->GetRealPackedValue();
    double time_decrypt     = TOC(t_decrypt);
    std::cout << "Decryption time: " << time_decrypt << " ms" << std::endl;

    if (verbose) {
        std::cout << "\n--- [Ciphertext] Square Matrix Product Result ---\n";
        RoundVector(res);
        PrintVector(res);
    }

    std::cout << "Matrix-Matrix Demo Complete.\n";
}

/**
 * @brief Main function with demo selection
 */
int main(int argc, char* argv[]) {
    int choice = 0;

    if (argc > 1) {
        choice = atoi(argv[1]);
    }
    else {
        std::cout << "OpenFHE Matrix Operations Demo\n"
                  << "-------------------------------\n"
                  << "1. Matrix-Vector Product\n"
                  << "2. Matrix-Matrix Product\n"
                  << "Enter choice (default=1): ";
        std::cin >> choice;
    }

    switch (choice) {
        case 2:
            DemoMatrixProduct();
            break;
        case 1:
        default:
            MatrixVectorProduct_CRC();
            break;
    }

    return 0;
}