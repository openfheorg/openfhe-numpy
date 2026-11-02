"""Square matrix products with one plaintext operand.

Under BSGS, ct @ pt and pt @ ct are one direct linear transform and consume one level. NAIVE keeps
the legacy JKLS product, which consumes three. Both must give A @ B. Under FIXEDMANUAL the C++
products expect a manual rescale, so that technique is not tested here.
"""
import unittest

import numpy as np
import openfhe as fhe
import openfhe_numpy as onp

from core.lintrans import (ATOL, decrypt, depth_used, encode, encrypt, lintrans_algo,
                           make_context, tile)

SHAPES = ((4, 16), (4, 64), (8, 64))  # (d, batch size): d^2 and tiled layouts
SCALINGS = (("FLEXIBLEAUTO", fhe.FLEXIBLEAUTO), ("FIXEDAUTO", fhe.FIXEDAUTO))
EXPECTED_DEPTH = {onp.LinTransAlgo.BSGS: 1, onp.LinTransAlgo.NAIVE: 3}


class PlaintextOperandMatMul(unittest.TestCase):
    def test_result_and_depth(self):
        rng = np.random.default_rng(97)
        for scaling_name, scaling in SCALINGS:
            for d, slots in SHAPES:
                cc, keys = make_context(slots, scaling)
                onp.EvalSquareMatMultRotateKeyGen(keys.secretKey, d)
                A = rng.uniform(-1, 1, (d, d))
                B = rng.uniform(-1, 1, (d, d))
                ctA = encrypt(cc, keys, A, slots)
                ctB = encrypt(cc, keys, B, slots)
                expected = tile(A @ B, slots)

                cases = {
                    "ct@pt": (ctA, lambda: onp.EvalMatMulSquare(ctA, encode(cc, B, slots), d)),
                    "pt@ct": (ctB, lambda: onp.EvalMatMulSquare(encode(cc, A, slots), ctB, d)),
                }
                for name, (source, run) in cases.items():
                    for algo, depth in EXPECTED_DEPTH.items():
                        with self.subTest(scaling=scaling_name, d=d, slots=slots, op=name, algo=str(algo)):
                            with lintrans_algo(algo):
                                result = run()
                            np.testing.assert_allclose(
                                decrypt(cc, keys, result, slots), expected, atol=ATOL
                            )
                            self.assertEqual(depth_used(source, result), depth)


if __name__ == "__main__":
    unittest.main(verbosity=2)
