# tests/test_vector_ops.py
import numpy as np
import openfhe_numpy as onp
from core import *


class TestVectorSimpleOperations(MainUnittest):
    sizes = [5, 8, 16]

    def test_unary_operations(self):
        ops = [
            ("transpose", lambda x: x.T, lambda a: a.T),
            ("scalar_mul", lambda x, s: x * s, lambda a, s: a * s),
            ("sum", lambda x: np.sum(x), lambda a: onp.sum(a)),
        ]

        ckks_params = load_ckks_params()
        for _, p in enumerate(ckks_params):
            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            batch_size = p["ringDim"] // 2
            scalar = 7.2  # used by scalar_mul

            for tag, np_fn, fhe_fn in ops:
                for size in self.sizes:
                    A = generate_random_array(rows=size, cols=1)

                    with self.subTest(tag=tag, size=size, ringDim=p["ringDim"]):
                        # encrypt A
                        ctv_a = onp.array(
                            cc=cc,
                            data=A,
                            batch_size=batch_size,
                            order=onp.ROW_MAJOR,
                            fhe_type="C",
                            mode="zero",
                            public_key=keys.publicKey,
                        )

                        # run encrypted op
                        if tag == "scalar_mul":
                            expected = np_fn(A, scalar)
                            ctv_res = fhe_fn(ctv_a, scalar)
                        else:
                            expected = np_fn(A)
                            ctv_res = fhe_fn(ctv_a)

                        # decrypt and compare
                        result = ctv_res.decrypt(keys.secretKey, unpack_type="original")

                        self.assertArrayClose(
                            params={
                                "case": "vector_unary_op_" + tag,
                                "size": size,
                                "ringDim": p["ringDim"],
                                "order": onp.ROW_MAJOR,
                            },
                            input_data={"A": A},
                            actual=result,
                            expected=expected,
                        )

    def test_binary_operations(self):
        # Only binary ops here.
        ops = [
            ("add", lambda x, y: x + y, lambda a, b: onp.add(a, b)),
            ("sub", lambda x, y: x - y, lambda a, b: onp.subtract(a, b)),
            ("mul", lambda x, y: x * y, lambda a, b: onp.multiply(a, b)),
            ("dot", lambda x, y: np.dot(A, B), lambda a, b: onp.dot(a, b)),
        ]

        ckks_params = load_ckks_params()
        for _, p in enumerate(ckks_params):
            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            batch_size = p["ringDim"] // 2

            for tag, np_fn, fhe_fn in ops:
                for size in self.sizes:
                    A = generate_random_array(rows=size, cols=1)
                    B = generate_random_array(rows=size, cols=1)
                    expected = np_fn(A, B)

                    with self.subTest(tag=tag, size=size, ringDim=p["ringDim"]):
                        ctv_a = onp.array(
                            cc=cc,
                            data=A,
                            batch_size=batch_size,
                            order=onp.ROW_MAJOR,
                            fhe_type="C",
                            mode="zero",
                            public_key=keys.publicKey,
                        )
                        ctv_b = onp.array(
                            cc=cc,
                            data=B,
                            batch_size=batch_size,
                            order=onp.ROW_MAJOR,
                            fhe_type="C",
                            mode="zero",
                            public_key=keys.publicKey,
                        )

                        ctv_res = fhe_fn(ctv_a, ctv_b)

                        result = ctv_res.decrypt(keys.secretKey, unpack_type="original")

                        self.assertArrayClose(
                            params={
                                "case": "vector_binary_op_" + tag,
                                "size": size,
                                "ringDim": p["ringDim"],
                                "order": onp.ROW_MAJOR,
                            },
                            input_data={"A": A, "B": B},
                            actual=result,
                            expected=expected,
                        )


# In test_simple_vector_operation.py
if __name__ == "__main__":
    TestVectorSimpleOperations.run_test_summary("Vector Operations", verbose=True)
