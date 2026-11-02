"""Linear transforms when the batch size is d^2 (the BERT layout).

Here Tau, Psi and the square matrix product use the canonical row shifts d*k, and key generation
skips the wrap keys d*k - d^2 by default. Only the keys from the standard key-generation helpers
are generated, so a missing key fails the test.
"""
import unittest

import numpy as np
import openfhe as fhe
import openfhe_numpy as onp

from core.lintrans import (ATOL, decrypt, encrypt, make_context, psi_ref, tau_ref, tile)

DIMS = (4, 8)
SCALINGS = (("FLEXIBLEAUTO", fhe.FLEXIBLEAUTO), ("FIXEDAUTO", fhe.FIXEDAUTO))


class LinearTransformBatchD2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(2026)
        cls.cases = []
        for scaling_name, scaling in SCALINGS:
            for d in DIMS:
                slots = d * d
                cc, keys = make_context(slots, scaling)
                onp.EvalSquareMatMultRotateKeyGen(keys.secretKey, d)
                onp.EvalTransposeKeyGen(keys.secretKey, d)
                A = rng.uniform(-1, 1, (d, d))
                B = rng.uniform(-1, 1, (d, d))
                cls.cases.append(
                    dict(name=scaling_name, d=d, slots=slots, cc=cc, keys=keys, A=A, B=B,
                         ctA=encrypt(cc, keys, A, slots), ctB=encrypt(cc, keys, B, slots))
                )

    def check(self, case, ciphertext, expected):
        actual = decrypt(case["cc"], case["keys"], ciphertext, case["slots"])
        np.testing.assert_allclose(actual, tile(expected, case["slots"]), atol=ATOL)

    def test_tau(self):
        for c in self.cases:
            with self.subTest(scaling=c["name"], d=c["d"]):
                self.check(c, onp.EvalLinTransTau(c["ctA"], c["d"]), tau_ref(c["A"]))

    def test_psi(self):
        for c in self.cases:
            for k in range(1, c["d"]):
                with self.subTest(scaling=c["name"], d=c["d"], k=k):
                    self.check(c, onp.EvalLinTransPsi(c["ctA"], c["d"], k), psi_ref(c["A"], k))

    def test_transpose(self):
        for c in self.cases:
            with self.subTest(scaling=c["name"], d=c["d"]):
                self.check(c, onp.EvalTranspose(c["ctA"], c["d"]), c["A"].T)

    def test_matmul(self):
        for c in self.cases:
            with self.subTest(scaling=c["name"], d=c["d"]):
                self.check(c, onp.EvalMatMulSquare(c["ctA"], c["ctB"], c["d"]), c["A"] @ c["B"])

    @unittest.skipUnless(onp.GetCanonicalRowShiftKeys(), "wrap-key pruning turned off")
    def test_wrap_keys_are_pruned(self):
        # d - d^2 is the first wrap rotation; with automatic pruning no helper generates its key
        for c in self.cases:
            with self.subTest(scaling=c["name"], d=c["d"]):
                with self.assertRaises(RuntimeError):
                    c["cc"].EvalRotate(c["ctA"], c["d"] - c["d"] * c["d"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
