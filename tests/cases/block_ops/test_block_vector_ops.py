import math
import numpy as np
from openfhe import *
import openfhe_numpy as onp
from core import *


# Block vector tests
# - CT/CT elementwise binary operators: add, subtract, multiply
# - CT/PT elementwise binary operators: add, subtract, multiply
# - scalar multiplication
# - block vector dot product via onp.dot and @
# - layout mismatch validation
SCALAR = 7.9

ORDERS = [("row_major", onp.ROW_MAJOR)]
MODES = ["zero"]


def make_block_vector_cases(full_batch_size):
    """
    Build block-vector test cases.

    The first cases are small and fast.
    The final case is the important stress case:
        vector size = ringDim // 2 + 1

    This proves the block vector can store data larger than one ciphertext.
    """
    cases = [
        {
            "name": "small_17_by_4",
            "size": 17,
            "block_batch_size": 4,
        },
        {
            "name": "small_33_by_8",
            "size": 33,
            "block_batch_size": 8,
        },
        {
            "name": "small_65_by_16",
            "size": 65,
            "block_batch_size": 16,
        },
        {
            "name": "larger_than_one_full_ciphertext",
            "size": full_batch_size + 1,
            "block_batch_size": full_batch_size,
        },
    ]

    valid_cases = []
    for case in cases:
        if case["block_batch_size"] <= full_batch_size:
            valid_cases.append(case)

    return valid_cases


def assert_block_vector_metadata(testcase, ctv, size, block_batch_size):
    """Check that block vector metadata is correct."""
    expected_num_blocks = math.ceil(size / block_batch_size)
    expected_padded_size = expected_num_blocks * block_batch_size

    testcase.assertEqual(ctv.ndim, 1)
    testcase.assertEqual(ctv.original_shape, (size,))
    testcase.assertEqual(ctv.block_shape, (block_batch_size,))
    testcase.assertEqual(ctv.grid_shape, (expected_num_blocks,))
    testcase.assertEqual(ctv.shape, (expected_padded_size,))
    testcase.assertEqual(ctv.num_blocks, expected_num_blocks)
    testcase.assertEqual(ctv.batch_size, block_batch_size)

    # Most important check: this must really be multiple ciphertext blocks.
    testcase.assertGreater(ctv.num_blocks, 1)


class TestBlockVectorConstruction(MainUnittest):
    """Test construction and decryption of long block vectors."""

    def test_block_vector_construction(self):
        ckks_params = load_ckks_params()

        for _, p in enumerate(ckks_params):
            full_batch_size = p["ringDim"] // 2

            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            for case in make_block_vector_cases(full_batch_size):
                size = case["size"]
                block_batch_size = case["block_batch_size"]
                case_name = case["name"]

                a = generate_random_array(rows=size)

                for order_name, order_value in ORDERS:
                    for mode in MODES:
                        with self.subTest(
                            case=case_name,
                            order=order_name,
                            size=size,
                            block_batch_size=block_batch_size,
                            mode=mode,
                            ringDim=p["ringDim"],
                        ):
                            result = None

                            try:
                                ctv = onp.block_array(
                                    cc=cc,
                                    data=a,
                                    batch_size=block_batch_size,
                                    block_shape=None,
                                    order=order_value,
                                    fhe_type="C",
                                    mode=mode,
                                    public_key=keys.publicKey,
                                )

                                assert_block_vector_metadata(
                                    self,
                                    ctv,
                                    size,
                                    block_batch_size,
                                )

                                result = ctv.decrypt(
                                    keys.secretKey,
                                    unpack_type="original",
                                )

                                self.assertArrayClose(
                                    actual=result,
                                    expected=a,
                                )

                            except Exception:
                                self._record_case(
                                    params={
                                        "case": "block_vector_construction",
                                        "case_name": case_name,
                                        "size": size,
                                        "block_batch_size": block_batch_size,
                                        "ringDim": p["ringDim"],
                                    },
                                    input_data={"a": a},
                                    expected=a,
                                    result=result,
                                )
                                raise


class TestBlockVectorUnaryOps(MainUnittest):
    """Test unary/scalar operations on long block vectors."""

    def test_block_vector_unary_operations(self):
        # Do not include transpose or sum here yet.
        # Block transpose/sum need separate block-level implementations.
        ops = [
            ("scalar_mul", lambda x, s: x * s, lambda x, s: x * s),
        ]

        ckks_params = load_ckks_params()

        for _, p in enumerate(ckks_params):
            full_batch_size = p["ringDim"] // 2

            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            for tag, np_fn, fhe_fn in ops:
                for case in make_block_vector_cases(full_batch_size):
                    size = case["size"]
                    block_batch_size = case["block_batch_size"]
                    case_name = case["name"]

                    a = generate_random_array(rows=size)

                    for order_name, order_value in ORDERS:
                        for mode in MODES:
                            with self.subTest(
                                op=tag,
                                case=case_name,
                                order=order_name,
                                size=size,
                                block_batch_size=block_batch_size,
                                mode=mode,
                                ringDim=p["ringDim"],
                            ):
                                result = None
                                expected = np_fn(a, SCALAR)

                                try:
                                    ctv = onp.block_array(
                                        cc=cc,
                                        data=a,
                                        batch_size=block_batch_size,
                                        block_shape=None,
                                        order=order_value,
                                        fhe_type="C",
                                        mode=mode,
                                        public_key=keys.publicKey,
                                    )

                                    assert_block_vector_metadata(
                                        self,
                                        ctv,
                                        size,
                                        block_batch_size,
                                    )

                                    ctv_res = fhe_fn(ctv, SCALAR)

                                    self.assertEqual(ctv_res.num_blocks, ctv.num_blocks)
                                    self.assertEqual(ctv_res.original_shape, ctv.original_shape)
                                    self.assertEqual(ctv_res.block_shape, ctv.block_shape)
                                    self.assertEqual(ctv_res.grid_shape, ctv.grid_shape)

                                    result = ctv_res.decrypt(
                                        keys.secretKey,
                                        unpack_type="original",
                                    )

                                    self.assertArrayClose(
                                        actual=result,
                                        expected=expected,
                                        rtol=1e-7,
                                        atol=1e-10,
                                    )

                                except Exception:
                                    self._record_case(
                                        params={
                                            "case": "block_vector_unary",
                                            "op": tag,
                                            "case_name": case_name,
                                            "size": size,
                                            "block_batch_size": block_batch_size,
                                            "ringDim": p["ringDim"],
                                        },
                                        input_data={"a": a},
                                        expected=expected,
                                        result=result,
                                    )
                                    raise


class TestBlockVectorBinaryOps(MainUnittest):
    """Test CT/CT binary elementwise operations on long block vectors."""

    def test_block_vector_binary_operations(self):
        # Do not include dot here.
        # Block dot product needs block-level reduction across ciphertext blocks.
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
                for case in make_block_vector_cases(full_batch_size):
                    size = case["size"]
                    block_batch_size = case["block_batch_size"]
                    case_name = case["name"]

                    a = generate_random_array(rows=size)
                    b = generate_random_array(rows=size)

                    for order_name, order_value in ORDERS:
                        expected = np_fn(a, b)

                        for mode in MODES:
                            with self.subTest(
                                op=tag,
                                case=case_name,
                                order=order_name,
                                size=size,
                                block_batch_size=block_batch_size,
                                mode=mode,
                                ringDim=p["ringDim"],
                            ):
                                result = None

                                try:
                                    ctv_a = onp.block_array(
                                        cc=cc,
                                        data=a,
                                        batch_size=block_batch_size,
                                        block_shape=None,
                                        order=order_value,
                                        fhe_type="C",
                                        mode=mode,
                                        public_key=keys.publicKey,
                                    )

                                    ctv_b = onp.block_array(
                                        cc=cc,
                                        data=b,
                                        batch_size=block_batch_size,
                                        block_shape=None,
                                        order=order_value,
                                        fhe_type="C",
                                        mode=mode,
                                        public_key=keys.publicKey,
                                    )

                                    assert_block_vector_metadata(
                                        self,
                                        ctv_a,
                                        size,
                                        block_batch_size,
                                    )

                                    assert_block_vector_metadata(
                                        self,
                                        ctv_b,
                                        size,
                                        block_batch_size,
                                    )

                                    ctv_res = fhe_fn(ctv_a, ctv_b)

                                    self.assertEqual(ctv_res.num_blocks, ctv_a.num_blocks)
                                    self.assertEqual(ctv_res.original_shape, ctv_a.original_shape)
                                    self.assertEqual(ctv_res.block_shape, ctv_a.block_shape)
                                    self.assertEqual(ctv_res.grid_shape, ctv_a.grid_shape)

                                    result = ctv_res.decrypt(
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
                                            "case": "block_vector_binary_ct_ct",
                                            "op": tag,
                                            "case_name": case_name,
                                            "size": size,
                                            "block_batch_size": block_batch_size,
                                            "ringDim": p["ringDim"],
                                        },
                                        input_data={"a": a, "b": b},
                                        expected=expected,
                                        result=result,
                                    )
                                    raise


class TestBlockVectorCipherPlainOps(MainUnittest):
    """Test CT/PT binary elementwise operations on long block vectors."""

    def test_block_vector_cipher_plain_operations(self):
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
                for case in make_block_vector_cases(full_batch_size):
                    size = case["size"]
                    block_batch_size = case["block_batch_size"]
                    case_name = case["name"]

                    a = generate_random_array(rows=size)
                    b = generate_random_array(rows=size)

                    for order_name, order_value in ORDERS:
                        expected = np_fn(a, b)

                        for mode in MODES:
                            with self.subTest(
                                op=tag,
                                case=case_name,
                                order=order_name,
                                size=size,
                                block_batch_size=block_batch_size,
                                mode=mode,
                                ringDim=p["ringDim"],
                            ):
                                result = None

                                try:
                                    ctv_a = onp.block_array(
                                        cc=cc,
                                        data=a,
                                        batch_size=block_batch_size,
                                        block_shape=None,
                                        order=order_value,
                                        fhe_type="C",
                                        mode=mode,
                                        public_key=keys.publicKey,
                                    )

                                    ptv_b = onp.block_array(
                                        cc=cc,
                                        data=b,
                                        batch_size=block_batch_size,
                                        block_shape=None,
                                        order=order_value,
                                        fhe_type="P",
                                        mode=mode,
                                    )

                                    assert_block_vector_metadata(
                                        self,
                                        ctv_a,
                                        size,
                                        block_batch_size,
                                    )

                                    assert_block_vector_metadata(
                                        self,
                                        ptv_b,
                                        size,
                                        block_batch_size,
                                    )

                                    ctv_res = fhe_fn(ctv_a, ptv_b)

                                    self.assertEqual(ctv_res.num_blocks, ctv_a.num_blocks)
                                    self.assertEqual(ctv_res.original_shape, ctv_a.original_shape)
                                    self.assertEqual(ctv_res.block_shape, ctv_a.block_shape)
                                    self.assertEqual(ctv_res.grid_shape, ctv_a.grid_shape)

                                    result = ctv_res.decrypt(
                                        keys.secretKey,
                                        unpack_type="original",
                                    )

                                    self.assertArrayClose(
                                        actual=result,
                                        expected=expected,
                                        rtol=1e-7,
                                        atol=1e-10,
                                    )

                                except Exception:
                                    self._record_case(
                                        params={
                                            "case": "block_vector_binary_ct_pt",
                                            "op": tag,
                                            "case_name": case_name,
                                            "size": size,
                                            "block_batch_size": block_batch_size,
                                            "ringDim": p["ringDim"],
                                        },
                                        input_data={"a": a, "b": b},
                                        expected=expected,
                                        result=result,
                                    )
                                    raise


class TestBlockVectorDotOps(MainUnittest):
    """Test block vector dot-product and matmul operations."""

    def test_block_vector_dot_operations(self):
        ops = [
            ("dot", lambda x, y: np.dot(x, y), lambda a, b: onp.dot(a, b)),
            ("matmul", lambda x, y: np.dot(x, y), lambda a, b: a @ b),
        ]

        ckks_params = load_ckks_params()

        for _, p in enumerate(ckks_params):
            full_batch_size = p["ringDim"] // 2

            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            for tag, np_fn, fhe_fn in ops:
                for case in make_block_vector_cases(full_batch_size):
                    size = case["size"]
                    block_batch_size = case["block_batch_size"]
                    case_name = case["name"]

                    a = generate_random_array(rows=size)
                    b = generate_random_array(rows=size)
                    expected = np_fn(a, b)

                    for order_name, order_value in ORDERS:
                        for mode in MODES:
                            with self.subTest(
                                op=tag,
                                case=case_name,
                                order=order_name,
                                size=size,
                                block_batch_size=block_batch_size,
                                mode=mode,
                                ringDim=p["ringDim"],
                            ):
                                result = None

                                try:
                                    ctv_a = onp.block_array(
                                        cc=cc,
                                        data=a,
                                        batch_size=block_batch_size,
                                        block_shape=None,
                                        order=order_value,
                                        fhe_type="C",
                                        mode=mode,
                                        public_key=keys.publicKey,
                                    )

                                    ctv_b = onp.block_array(
                                        cc=cc,
                                        data=b,
                                        batch_size=block_batch_size,
                                        block_shape=None,
                                        order=order_value,
                                        fhe_type="C",
                                        mode=mode,
                                        public_key=keys.publicKey,
                                    )

                                    assert_block_vector_metadata(
                                        self,
                                        ctv_a,
                                        size,
                                        block_batch_size,
                                    )

                                    assert_block_vector_metadata(
                                        self,
                                        ctv_b,
                                        size,
                                        block_batch_size,
                                    )

                                    ctv_res = fhe_fn(ctv_a, ctv_b)
                                    result = ctv_res.decrypt(
                                        keys.secretKey,
                                        unpack_type="original",
                                    )

                                    self.assertAlmostEqual(
                                        float(np.asarray(result).reshape(-1)[0]),
                                        float(expected),
                                    )

                                except Exception:
                                    self._record_case(
                                        params={
                                            "case": "block_vector_dot",
                                            "op": tag,
                                            "case_name": case_name,
                                            "size": size,
                                            "block_batch_size": block_batch_size,
                                            "ringDim": p["ringDim"],
                                        },
                                        input_data={"a": a, "b": b},
                                        expected=expected,
                                        result=result,
                                    )
                                    raise


class TestBlockVectorLayoutMismatch(MainUnittest):
    """Test that incompatible block layouts are rejected."""

    def test_block_vector_layout_mismatch(self):
        ckks_params = load_ckks_params()

        for _, p in enumerate(ckks_params):
            full_batch_size = p["ringDim"] // 2

            # Need enough room for both block sizes.
            if full_batch_size < 8:
                continue

            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            size = 33
            a = generate_random_array(rows=size)
            b = generate_random_array(rows=size)

            for order_name, order_value in ORDERS:
                for mode in MODES:
                    with self.subTest(
                        case="block_vector_layout_mismatch",
                        order=order_name,
                        size=size,
                        mode=mode,
                        ringDim=p["ringDim"],
                    ):
                        ctv_a = onp.block_array(
                            cc=cc,
                            data=a,
                            batch_size=4,
                            block_shape=None,
                            order=order_value,
                            fhe_type="C",
                            mode=mode,
                            public_key=keys.publicKey,
                        )

                        ctv_b = onp.block_array(
                            cc=cc,
                            data=b,
                            batch_size=8,
                            block_shape=None,
                            order=order_value,
                            fhe_type="C",
                            mode=mode,
                            public_key=keys.publicKey,
                        )

                        self.assertNotEqual(ctv_a.block_shape, ctv_b.block_shape)
                        self.assertNotEqual(ctv_a.grid_shape, ctv_b.grid_shape)

                        with self.assertRaises(Exception):
                            _ = onp.add(ctv_a, ctv_b)


if __name__ == "__main__":
    TestBlockVectorConstruction.run_test_summary()
    TestBlockVectorUnaryOps.run_test_summary()
    TestBlockVectorBinaryOps.run_test_summary()
    TestBlockVectorCipherPlainOps.run_test_summary()
    TestBlockVectorLayoutMismatch.run_test_summary()
