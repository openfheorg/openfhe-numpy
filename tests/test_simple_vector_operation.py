# tests/test_vector_ops.py

import numpy as np
import openfhe_numpy as onp

from core.test_framework import MainUnittest
from core.test_utils import generate_random_array
from core.test_crypto_context import load_ckks_params, gen_crypto_context


def fhe_vector_op(params, data, op_name):
    """
    Generic runner for FHE vector operations.
    - params: CKKS params dict
    - data: list, either [A, B] or [A, scalar] or [A]
    - op_name: one of "add", "sub", "transpose", "mul", "scalar_mul", "dot", "sum"
    """

    cc, keys = gen_crypto_context(params)
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    A = np.array(data[0])
    batch_size = params["ringDim"] // 2

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

    # optionally encrypt B
    if op_name in ("add", "sub", "mul", "dot"):
        B = np.array(data[1])
        ctv_b = onp.array(
            cc=cc,
            data=B,
            batch_size=batch_size,
            order=onp.ROW_MAJOR,
            fhe_type="C",
            mode="zero",
            public_key=keys.publicKey,
        )

    # dispatch
    if op_name == "add":
        ctv_res = onp.add(ctv_a, ctv_b)
    elif op_name == "sub":
        ctv_res = onp.subtract(ctv_a, ctv_b)
    elif op_name == "transpose":
        onp.gen_transpose_keys(keys.secretKey, ctv_a)
        ctv_res = onp.transpose(ctv_a)
    elif op_name == "mul":
        ctv_res = onp.multiply(ctv_a, ctv_b)
    elif op_name == "scalar_mul":
        scalar = data[1]
        ctv_res = ctv_a * scalar
    elif op_name == "dot":
        # ctv_res = ctv_a @ ctv_b
        ctv_res = onp.dot(ctv_a, ctv_b)
    elif op_name == "sum":
        ctv_res = onp.sum(ctv_a)
    else:
        raise ValueError(f"Unknown op: {op_name}")

    # decrypt
    return ctv_res.decrypt(keys.secretKey, unpack_type="original")


class TestVectorOperations(MainUnittest):
    """Dynamically parameterized tests for all FHE vector ops."""

    @classmethod
    def _generate_test_cases(cls):
        ops = [
            ("add", lambda A, B: A + B),
            ("sub", lambda A, B: A - B),
            ("mul", lambda A, B: A * B),
            ("dot", lambda A, B: np.dot(A, B)),
            ("transpose", lambda A: A.T),
            ("scalar_mul", lambda A, s: A * s),
            ("sum", lambda A: np.sum(A)),
        ]

        ckks_params = load_ckks_params()
        sizes = [5, 8, 16]
        scalar = 7.0
        test_id = 1

        for op_name, np_fn in ops:
            for params in ckks_params:
                for size in sizes:
                    A = generate_random_array(rows=size, cols=1)
                    # prepare data & expected
                    if op_name in ("add", "sub", "mul", "dot"):
                        B = generate_random_array(rows=size, cols=1)
                        expected = np_fn(A, B)
                        data = [A, B]
                    elif op_name == "scalar_mul":
                        expected = np_fn(A, scalar)
                        data = [A, scalar]
                    else:  # transpose or sum
                        expected = np_fn(A)
                        data = [A]

                    name = f"{op_name}_{test_id:03d}_ring_{params['ringDim']}_n{size}"
                    cls.generate_test_case(
                        func=lambda p, d, op=op_name: fhe_vector_op(p, d, op),
                        test_name=name,
                        params=params,
                        input_data=data,
                        expected=expected,
                        compare_fn=onp.check_equality,
                        debug=False,
                    )
                    test_id += 1


if __name__ == "__main__":
    TestVectorOperations.run_test_summary("Vector Ops", debug=True)
