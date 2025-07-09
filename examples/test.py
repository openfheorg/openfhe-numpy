import numpy as np
from openfhe import *
import openfhe_numpy as onp


def validate_and_print_results(computed, expected, operation_name):
    """Helper function to validate and print vector results."""
    print("\n" + "*" * 60)
    print(f"{operation_name}")
    print("*" * 60)
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{computed}")

    is_match, error = onp.check_equality(computed, expected)
    print(f"\nMatch: {is_match}, Total Error: {error}")
    return is_match, error


params = {
    "batchSize": 512,
    "digitSize": 0,
    "firstModSize": 60,
    "ksTech": "HYBRID",
    "maxRelinSkDeg": 2,
    "multiplicativeDepth": 9,
    "numLargeDigits": 3,
    "ptModulus": "0",
    "ringDim": 1024,
    "scalTech": "FLEXIBLEAUTO",
    "scalingModSize": 59,
    "secretKeyDist": "UNIFORM_TERNARY",
    "securityLevel": "HEStd_NotSet",
    "standardDeviation": 3.19,
}


def main():
    # Cryptographic setup
    p = CCParamsCKKSRNS()
    p.SetRingDim(params["ringDim"])
    p.SetMultiplicativeDepth(params["multiplicativeDepth"])
    p.SetScalingModSize(params["scalingModSize"])
    p.SetBatchSize(params["batchSize"])
    p.SetFirstModSize(params["firstModSize"])
    p.SetStandardDeviation(params["standardDeviation"])
    p.SetSecretKeyDist(UNIFORM_TERNARY)
    p.SetScalingTechnique(FLEXIBLEAUTOEXT)
    p.SetKeySwitchTechnique(HYBRID)
    p.SetSecurityLevel(HEStd_NotSet)
    p.SetNumLargeDigits(params["numLargeDigits"])
    p.SetMaxRelinSkDeg(params["maxRelinSkDeg"])
    p.SetDigitSize(params["digitSize"])

    cc = GenCryptoContext(p)

    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    ring_dim = cc.GetRingDimension()
    print(f"\nCrypto context: ring_dim={ring_dim}, slots={ring_dim // 2}")

    # Case 1: power-of-two (8x8)
    A8 = np.array(
        [
            [0, 7, 8, 10, 1, 2, 7, 6],
            [0, 1, 1, 9, 7, 5, 1, 7],
            [8, 8, 4, 5, 8, 2, 6, 1],
            [1, 0, 0, 1, 10, 3, 1, 7],
            [7, 8, 2, 5, 3, 2, 10, 9],
            [0, 3, 4, 10, 10, 5, 2, 5],
            [2, 5, 0, 2, 8, 8, 5, 9],
            [5, 1, 10, 6, 2, 8, 6, 3],
        ]
    )
    B8 = np.array(
        [
            [6, 5, 4, 3, 2, 1, 0, 7],
            [7, 1, 1, 2, 7, 5, 9, 3],
            [4, 8, 8, 10, 8, 2, 1, 6],
            [7, 0, 0, 5, 10, 3, 4, 2],
            [9, 3, 2, 8, 3, 2, 1, 0],
            [5, 2, 4, 1, 10, 5, 8, 2],
            [9, 8, 0, 2, 8, 8, 7, 5],
            [3, 6, 10, 1, 2, 8, 4, 0],
        ]
    )
    validate_and_print_results(
        res_sum_decrypted, np.sum(vector_a), "Sum of vector\n" + str(vector_a)
    )


if __name__ == "__main__":
    main()
