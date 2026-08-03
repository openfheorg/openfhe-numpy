import numpy as np
from openfhe import *
import openfhe_numpy as onp

params = CCParamsCKKSRNS()
params.SetMultiplicativeDepth(4)
params.SetScalingModSize(59)
params.SetFirstModSize(60)
params.SetScalingTechnique(FIXEDAUTO)
cc = GenCryptoContext(params)
cc.Enable(PKESchemeFeature.PKE)
cc.Enable(PKESchemeFeature.LEVELEDSHE)
cc.Enable(PKESchemeFeature.ADVANCEDSHE)
keys = cc.KeyGen()
cc.EvalMultKeyGen(keys.secretKey)
cc.EvalSumKeyGen(keys.secretKey)
batch_size = cc.GetRingDimension() // 2

# Non-square: 2 rows, 4 cols
A = np.array(
    [
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
    ]
)
print("A shape:", A.shape)
print("np.cumsum(A, axis=0):\n", np.cumsum(A, axis=0))

ctm_a = onp.array(
    cc=cc,
    data=A,
    batch_size=batch_size,
    order=onp.ROW_MAJOR,
    fhe_type="C",
    mode="tile",
    public_key=keys.publicKey,
)
print("ctm_a.ncols (padded):", ctm_a.ncols, "original_shape:", ctm_a.original_shape)

onp.gen_accumulate_rows_key(keys.secretKey, ctm_a.ncols)

try:
    res = ctm_a.cumsum(axis=0)
    decrypted = res.decrypt(keys.secretKey, unpack_type="original")
    print("\ncumsum(axis=0) result:\n", decrypted)
    print("\nMatches numpy?", np.allclose(decrypted, np.cumsum(A, axis=0), atol=1e-1))
except Exception as e:
    print("cumsum(axis=0) FAILED:", type(e).__name__, e)


# ------------------------------------------------------------------
# R1: 1-D vector cumsum -- this is the one that actually crashes.
# ------------------------------------------------------------------
print("\n--- 1-D vector cumsum ---")
v = np.array([1.0, 2.0, 3.0, 4.0])
print("v:", v)
print("np.cumsum(v):", np.cumsum(v))

ctm_v = onp.array(
    cc=cc,
    data=v,
    batch_size=batch_size,
    order=onp.ROW_MAJOR,
    fhe_type="C",
    mode="tile",
    public_key=keys.publicKey,
)
print(
    "ctm_v.ndim:", ctm_v.ndim, "ctm_v.ncols:", ctm_v.ncols, "original_shape:", ctm_v.original_shape
)

onp.gen_accumulate_cols_key(keys.secretKey, ctm_v.ncols)

try:
    res_v = ctm_v.cumsum()  # default axis=0 -- same crash as axis=None
    decrypted_v = res_v.decrypt(keys.secretKey, unpack_type="original")
    print("cumsum() result:", decrypted_v)
except Exception as e:
    print("cumsum() FAILED:", type(e).__name__, e)
