"""Every linear transform must give the same result with NAIVE and with BSGS.

Each operation runs with both methods on the same ciphertext. Both outputs must match the numpy
reference and each other. Batch sizes d^2 and 4*d^2 cover the canonical and the tiled layouts.
Under FIXEDMANUAL the C++ matrix products expect a manual rescale, so only the single transforms
run there.
"""
import unittest

import numpy as np
import openfhe as fhe
import openfhe_numpy as onp

from core.lintrans import (ATOL, decrypt, encode, encrypt, lintrans_algo, lt_ref, make_context,
                           phi_ref, psi_ref, sigma_ref, tau_ref, tile)

SHAPES = ((4, 16), (4, 64), (8, 64), (8, 256))  # (d, batch size)
SCALINGS = (
    ("FLEXIBLEAUTO", fhe.FLEXIBLEAUTO),
    ("FIXEDAUTO", fhe.FIXEDAUTO),
    ("FIXEDMANUAL", fhe.FIXEDMANUAL),
)
ALGOS = (onp.LinTransAlgo.NAIVE, onp.LinTransAlgo.BSGS)


class NaiveVersusBsgs(unittest.TestCase):
    def test_all_transforms(self):
        rng = np.random.default_rng(41)
        for scaling_name, scaling in SCALINGS:
            for d, slots in SHAPES:
                cc, keys = make_context(slots, scaling)
                onp.EvalSquareMatMultRotateKeyGen(keys.secretKey, d)
                onp.EvalTransposeKeyGen(keys.secretKey, d)
                A = rng.uniform(-1, 1, (d, d))
                B = rng.uniform(-1, 1, (d, d))
                ctA = encrypt(cc, keys, A, slots)
                ctB = encrypt(cc, keys, B, slots)
                k = d - 1

                # generic transform over SIGMA offsets (their keys exist), one diagonal all zero
                offsets = sorted(rng.choice(np.arange(-(d - 1), d), size=5, replace=False).tolist())
                diagonals = [rng.uniform(-1, 1, slots).tolist() for _ in offsets]
                diagonals[1] = [0.0] * slots

                cases = {
                    "sigma": (lambda: onp.EvalLinTransSigma(ctA, d), tile(sigma_ref(A), slots)),
                    "tau": (lambda: onp.EvalLinTransTau(ctA, d), tile(tau_ref(A), slots)),
                    "phi": (lambda: onp.EvalLinTransPhi(ctA, d, k), tile(phi_ref(A, k), slots)),
                    "psi": (lambda: onp.EvalLinTransPsi(ctA, d, k), tile(psi_ref(A, k), slots)),
                    "transpose": (lambda: onp.EvalTranspose(ctA, d), tile(A.T, slots)),
                    "generic": (lambda: onp.EvalLinearTransform(ctA, offsets, diagonals),
                                lt_ref(tile(A, slots), offsets, diagonals)),
                }
                if scaling != fhe.FIXEDMANUAL:
                    product = tile(A @ B, slots)
                    cases["ct@ct"] = (lambda: onp.EvalMatMulSquare(ctA, ctB, d), product)
                    cases["ct@pt"] = (lambda: onp.EvalMatMulSquare(ctA, encode(cc, B, slots), d), product)
                    cases["pt@ct"] = (lambda: onp.EvalMatMulSquare(encode(cc, A, slots), ctB, d), product)

                for name, (run, expected) in cases.items():
                    with self.subTest(scaling=scaling_name, d=d, slots=slots, op=name):
                        outputs = []
                        for algo in ALGOS:
                            with lintrans_algo(algo):
                                outputs.append(decrypt(cc, keys, run(), slots))
                        naive, bsgs = outputs
                        np.testing.assert_allclose(naive, expected, atol=ATOL, err_msg="NAIVE")
                        np.testing.assert_allclose(bsgs, expected, atol=ATOL, err_msg="BSGS")
                        np.testing.assert_allclose(bsgs, naive, atol=ATOL, err_msg="BSGS vs NAIVE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
