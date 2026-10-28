"""Regression tests for the generic linear transform (opt-lt review findings).

Each case uses a fresh key pair and generates only the rotation keys under test, because extra
rotation keys hide planning mistakes. The small ring and HEStd_NotSet are for speed only.

1. Diagonals that cancel after merging equal offsets must give zero, not the input.
2. The keys from LinearTransformRotationIndices must suffice whichever diagonals are zero.
3. The same keys must suffice for a ciphertext with more slots than the batch size.
4. Wrap-key pruning is automatic. With pruning off, EvalSquareMatMultRotateKeyGen keys must cover
   Tau on such a ciphertext. With pruning on, a ciphertext packed at the batch size must work, and
   one with more slots must raise an error that says how to resolve it.
"""
import os
import unittest

import numpy as np
import openfhe as fhe
import openfhe_numpy as onp

OPTIMIZED = (onp.LinTransAlgo.BSGS,)
ALL_ALGORITHMS = (onp.LinTransAlgo.NAIVE,) + OPTIMIZED
BATCH = 16


def fixture(slots=BATCH):
    params = fhe.CCParamsCKKSRNS()
    params.SetSecurityLevel(fhe.HEStd_NotSet)
    params.SetRingDim(1024)
    params.SetBatchSize(BATCH)
    params.SetMultiplicativeDepth(8)
    params.SetScalingModSize(40)
    params.SetFirstModSize(50)
    params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
    params.SetKeySwitchTechnique(fhe.HYBRID)

    cc = fhe.GenCryptoContext(params)
    for feature in (
        fhe.PKESchemeFeature.PKE,
        fhe.PKESchemeFeature.KEYSWITCH,
        fhe.PKESchemeFeature.LEVELEDSHE,
        fhe.PKESchemeFeature.ADVANCEDSHE,
    ):
        cc.Enable(feature)

    keys = cc.KeyGen()
    x = (np.arange(slots, dtype=float) - 5.0) / 8.0
    pt = cc.MakeCKKSPackedPlaintext(x.tolist(), 1, 0, None, slots)
    ct = cc.Encrypt(keys.publicKey, pt)
    assert ct.GetSlots() == slots
    return cc, keys, ct, x


def decrypt(cc, keys, ciphertext, length):
    pt = cc.Decrypt(keys.secretKey, ciphertext)
    pt.SetLength(length)
    return np.array(pt.GetRealPackedValue())


def tau_expected(x, d=4):
    """Tau applied to every d*d frame of x."""
    frames = len(x) // (d * d)
    return np.array(
        [x[f * d * d + d * ((i + j) % d) + j] for f in range(frames) for i in range(d) for j in range(d)]
    )


class LinearTransformRegressions(unittest.TestCase):
    def assertDecrypts(self, cc, keys, ciphertext, expected):
        np.testing.assert_allclose(
            decrypt(cc, keys, ciphertext, len(expected)), expected, rtol=1e-5, atol=1e-5
        )

    def test_cancelling_duplicate_diagonals(self):
        for algo in ALL_ALGORITHMS:
            with self.subTest(algo=str(algo)):
                cc, keys, ct, x = fixture()
                result = onp.EvalLinearTransform(
                    ct, [0, 0], [[1.0] * BATCH, [-1.0] * BATCH], algo, 4
                )
                self.assertDecrypts(cc, keys, result, np.zeros_like(x))

    def test_all_zero_diagonals(self):
        for algo in ALL_ALGORITHMS:
            with self.subTest(algo=str(algo)):
                cc, keys, ct, x = fixture()
                cc.EvalRotateKeyGen(keys.secretKey, [3])  # NAIVE rotates even for a zero mask
                result = onp.EvalLinearTransform(ct, [0, 3], [[0.0] * BATCH] * 2, algo, 4)
                self.assertDecrypts(cc, keys, result, np.zeros_like(x))

    def test_helper_keys_after_zero_diagonal_removal(self):
        offsets = list(range(8))
        for algo in OPTIMIZED:
            for nonzero in ([7], [5, 7], [0, 7], [1, 2, 3]):
                with self.subTest(algo=str(algo), nonzero=nonzero):
                    cc, keys, ct, x = fixture()
                    cc.EvalRotateKeyGen(
                        keys.secretKey, onp.LinearTransformRotationIndices(offsets, 4)
                    )
                    masks = [[0.0] * BATCH for _ in offsets]
                    for r in nonzero:
                        masks[r] = [1.0] * BATCH

                    result = onp.EvalLinearTransform(ct, offsets, masks, algo, 4)
                    expected = sum(np.roll(x, -r) for r in nonzero)
                    self.assertDecrypts(cc, keys, result, expected)

    def test_helper_keys_with_other_baby_step(self):
        # keys planned for baby step 4, evaluation asked for baby step 3: the planner searches
        # the other baby steps for one whose keys exist
        offsets = list(range(8))
        cc, keys, ct, x = fixture()
        cc.EvalRotateKeyGen(keys.secretKey, onp.LinearTransformRotationIndices(offsets, 4))
        masks = [[0.125] * BATCH for _ in offsets]
        result = onp.EvalLinearTransform(ct, offsets, masks, onp.LinTransAlgo.BSGS, 3)
        expected = sum(np.roll(x, -r) for r in offsets) / 8.0
        self.assertDecrypts(cc, keys, result, expected)

    def test_helper_keys_with_explicit_ciphertext_slots(self):
        for algo in OPTIMIZED:
            with self.subTest(algo=str(algo)):
                cc, keys, ct, x = fixture(slots=2 * BATCH)
                offsets = list(range(8))
                cc.EvalRotateKeyGen(
                    keys.secretKey, onp.LinearTransformRotationIndices(offsets, 4)
                )
                masks = [[0.125] * BATCH for _ in offsets]

                result = onp.EvalLinearTransform(ct, offsets, masks, algo, 4)
                expected = sum(np.roll(x, -r) for r in offsets) / 8.0
                self.assertDecrypts(cc, keys, result, expected)

    def test_matrix_keys_with_explicit_ciphertext_slots(self):
        previous_algo = onp.GetLinTransAlgo()
        previous_prune = onp.GetCanonicalRowShiftKeys()
        try:
            onp.SetCanonicalRowShiftKeys(False)  # this layout needs the wrap keys
            for algo in ALL_ALGORITHMS:
                with self.subTest(algo=str(algo)):
                    onp.SetLinTransAlgo(algo)
                    cc, keys, ct, x = fixture(slots=2 * BATCH)
                    onp.EvalSquareMatMultRotateKeyGen(keys.secretKey, 4)

                    result = onp.EvalLinTransTau(ct, 4)
                    self.assertDecrypts(cc, keys, result, tau_expected(x))
        finally:
            onp.SetLinTransAlgo(previous_algo)
            onp.SetCanonicalRowShiftKeys(previous_prune)

    @unittest.skipIf(os.environ.get("OPENFHE_NUMPY_CANONICAL_KEYS") == "0", "pruning turned off")
    def test_pruning_is_default(self):
        self.assertTrue(onp.GetCanonicalRowShiftKeys())

    def test_pruned_matrix_keys(self):
        previous_algo = onp.GetLinTransAlgo()
        previous_prune = onp.GetCanonicalRowShiftKeys()
        try:
            onp.SetCanonicalRowShiftKeys(True)
            for algo in ALL_ALGORITHMS:
                onp.SetLinTransAlgo(algo)
                # packed at the batch size: the pruned key set is enough
                with self.subTest(algo=str(algo), slots=BATCH):
                    cc, keys, ct, x = fixture()
                    onp.EvalSquareMatMultRotateKeyGen(keys.secretKey, 4)
                    result = onp.EvalLinTransTau(ct, 4)
                    self.assertDecrypts(cc, keys, result, tau_expected(x))
                # more slots than the batch size: the error names the cause
                with self.subTest(algo=str(algo), slots=2 * BATCH):
                    cc, keys, ct, x = fixture(slots=2 * BATCH)
                    onp.EvalSquareMatMultRotateKeyGen(keys.secretKey, 4)
                    with self.assertRaisesRegex(RuntimeError, "OPENFHE_NUMPY_CANONICAL_KEYS=0"):
                        onp.EvalLinTransTau(ct, 4)
        finally:
            onp.SetLinTransAlgo(previous_algo)
            onp.SetCanonicalRowShiftKeys(previous_prune)


if __name__ == "__main__":
    unittest.main(verbosity=2)
