import math
import numpy as np
from openfhe import *
import openfhe_numpy as onp
from core import *


# Block matrix tests
# - CT/CT elementwise binary operators: add, subtract, multiply
# - CT/PT elementwise binary operators: add, subtract, multiply
SCALAR = 7.9

ORDERS = [("row_major", onp.ROW_MAJOR)]
MODES = ["zero"]


def make_block_matrix_cases(full_batch_size):
    """Build block-matrix test cases."""
    cases = [
        {
            "name": "small_3x3_by_4",
            "shape": (3, 3),
            "batch_size": 4,
        },
        {
            "name": "small_5x4_by_4",
            "shape": (5, 4),
            "batch_size": 4,
        },
        {
            "name": "small_6x7_by_4",
            "shape": (6, 7),
            "batch_size": 4,
        },
        {
            "name": "small_5x5_by_8",
            "shape": (5, 5),
            "batch_size": 8,
        },
        {
            "name": "larger_than_one_full_ciphertext",
            "shape": (7, 7),
            "batch_size": 16,
        },
    ]

    valid_cases = []
    for case in cases:
        if case["batch_size"] <= full_batch_size:
            valid_cases.append(case)

    return valid_cases


def assert_block_matrix_metadata(testcase, ctm, shape, batch_size):
    """Validate block matrix metadata."""
    rows, cols = shape
    testcase.assertEqual(ctm.ndim, 2)
    testcase.assertEqual(ctm.original_shape, shape)
    testcase.assertEqual(ctm.batch_size, batch_size)
    testcase.assertEqual(len(ctm.block_shape), 2)

    br, bc = ctm.block_shape
    testcase.assertLessEqual(br * bc, batch_size)

    expected_grid = (math.ceil(rows / br), math.ceil(cols / bc))
    testcase.assertEqual(ctm.grid_shape, expected_grid)
    testcase.assertEqual(ctm.num_blocks, expected_grid[0] * expected_grid[1])
    testcase.assertEqual(ctm.shape, (expected_grid[0] * br, expected_grid[1] * bc))


class TestBlockMatrixBinaryOps(MainUnittest):
    """Test CT/CT binary elementwise operations on block matrices."""

    def test_block_matrix_binary_operations(self):
        ops = [
            ("add", lambda x, y: x + y, lambda a, b: onp.add(a, b)),
            ("sub", lambda x, y: x - y, lambda a, b: onp.subtract(a, b)),
            ("mul", lambda x, y: x * y, lambda a, b: onp.multiply(a, b)),
        ]

        ckks_params = load_ckks_params()

        for _, p in enumerate(ckks_params):
            full_batch_size = p["ringDim"] // 2

            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            for tag, np_fn, fhe_fn in ops:
                for case in make_block_matrix_cases(full_batch_size):
                    shape = case["shape"]
                    batch_size = case["batch_size"]
                    case_name = case["name"]

                    a = generate_random_array(rows=shape[0], cols=shape[1])
                    b = generate_random_array(rows=shape[0], cols=shape[1])
                    expected = np_fn(a, b)

                    for order_name, order_value in ORDERS:
                        for mode in MODES:
                            with self.subTest(
                                op=tag,
                                case=case_name,
                                shape=shape,
                                batch_size=batch_size,
                                order=order_name,
                                mode=mode,
                                ringDim=p["ringDim"],
                            ):
                                result = None

                                try:
                                    ctm_a = onp.block_array(
                                        cc=cc,
                                        data=a,
                                        batch_size=batch_size,
                                        block_shape=None,
                                        order=order_value,
                                        fhe_type="C",
                                        mode=mode,
                                        public_key=keys.publicKey,
                                    )

                                    ctm_b = onp.block_array(
                                        cc=cc,
                                        data=b,
                                        batch_size=batch_size,
                                        block_shape=None,
                                        order=order_value,
                                        fhe_type="C",
                                        mode=mode,
                                        public_key=keys.publicKey,
                                    )

                                    assert_block_matrix_metadata(
                                        self,
                                        ctm_a,
                                        shape,
                                        batch_size,
                                    )
                                    assert_block_matrix_metadata(
                                        self,
                                        ctm_b,
                                        shape,
                                        batch_size,
                                    )

                                    ctm_res = fhe_fn(ctm_a, ctm_b)
                                    result = ctm_res.decrypt(
                                        keys.secretKey,
                                        unpack_type="original",
                                    )

                                    self.assertArrayClose(
                                        actual=result,
                                        expected=expected,
                                    )

                                except Exception:
                                    self._record_case(
                                        params={
                                            "case": "block_matrix_binary_ct_ct",
                                            "op": tag,
                                            "case_name": case_name,
                                            "shape": shape,
                                            "batch_size": batch_size,
                                            "ringDim": p["ringDim"],
                                        },
                                        input_data={"a": a, "b": b},
                                        expected=expected,
                                        result=result,
                                    )
                                    raise


class TestBlockMatrixCipherPlainOps(MainUnittest):
    """Test CT/PT binary elementwise operations on block matrices."""

    def test_block_matrix_cipher_plain_operations(self):
        ops = [
            ("add_ct_pt", lambda x, y: x + y, lambda a, b: onp.add(a, b)),
            ("sub_ct_pt", lambda x, y: x - y, lambda a, b: onp.subtract(a, b)),
            ("mul_ct_pt", lambda x, y: x * y, lambda a, b: onp.multiply(a, b)),
        ]

        ckks_params = load_ckks_params()

        for _, p in enumerate(ckks_params):
            full_batch_size = p["ringDim"] // 2

            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            for tag, np_fn, fhe_fn in ops:
                for case in make_block_matrix_cases(full_batch_size):
                    shape = case["shape"]
                    batch_size = case["batch_size"]
                    case_name = case["name"]

                    a = generate_random_array(rows=shape[0], cols=shape[1])
                    b = generate_random_array(rows=shape[0], cols=shape[1])
                    expected = np_fn(a, b)

                    for order_name, order_value in ORDERS:
                        for mode in MODES:
                            with self.subTest(
                                op=tag,
                                case=case_name,
                                shape=shape,
                                batch_size=batch_size,
                                order=order_name,
                                mode=mode,
                                ringDim=p["ringDim"],
                            ):
                                result = None

                                try:
                                    ctm_a = onp.block_array(
                                        cc=cc,
                                        data=a,
                                        batch_size=batch_size,
                                        block_shape=None,
                                        order=order_value,
                                        fhe_type="C",
                                        mode=mode,
                                        public_key=keys.publicKey,
                                    )

                                    ptm_b = onp.block_array(
                                        cc=cc,
                                        data=b,
                                        batch_size=batch_size,
                                        block_shape=None,
                                        order=order_value,
                                        fhe_type="P",
                                        mode=mode,
                                    )

                                    assert_block_matrix_metadata(
                                        self,
                                        ctm_a,
                                        shape,
                                        batch_size,
                                    )
                                    assert_block_matrix_metadata(
                                        self,
                                        ptm_b,
                                        shape,
                                        batch_size,
                                    )

                                    ctm_res = fhe_fn(ctm_a, ptm_b)
                                    result = ctm_res.decrypt(
                                        keys.secretKey,
                                        unpack_type="original",
                                    )

                                    self.assertArrayClose(
                                        actual=result,
                                        expected=expected,
                                    )

                                except Exception:
                                    self._record_case(
                                        params={
                                            "case": "block_matrix_binary_ct_pt",
                                            "op": tag,
                                            "case_name": case_name,
                                            "shape": shape,
                                            "batch_size": batch_size,
                                            "ringDim": p["ringDim"],
                                        },
                                        input_data={"a": a, "b": b},
                                        expected=expected,
                                        result=result,
                                    )
                                    raise


if __name__ == "__main__":
    TestBlockMatrixBinaryOps.run_test_summary()
    TestBlockMatrixCipherPlainOps.run_test_summary()
