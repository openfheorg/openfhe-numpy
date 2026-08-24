import gc

import numpy as np

import openfhe_numpy as onp
from core import (
    ATOL,
    MainUnittest,
    generate_random_array,
    gen_crypto_context,
    load_ckks_params,
)


SHAPES = [(4, 4), (5, 7)]
BLOCK_SHAPE = (2, 2)
ORDERS = [("row_major", onp.ROW_MAJOR), ("col_major", onp.COL_MAJOR)]
MODES = ["zero"]
VECTOR_BLOCK_SHAPE = (BLOCK_SHAPE[1],)
VECTOR_SIZES = [4, 7]
BROADCAST_SHAPES = [(3, 6)]
BROADCAST_VECTOR_BLOCK_SHAPE = (BLOCK_SHAPE[0], 1)
RANDOM_SEED = 20260824


def _generate_case_array(rows, cols=None, *, operand=0):
    logical_cols = 0 if cols is None else cols
    seed = RANDOM_SEED + rows * 1_000 + logical_cols * 10 + operand
    return generate_random_array(rows=rows, cols=cols, seed=seed)


class BlockMatrixTestFramework(MainUnittest):
    def _run_unary(
        self,
        tag,
        np_fn,
        fhe_fn,
        *,
        prepare_context=None,
        prepare_tensor=None,
    ):
        ckks_params = load_ckks_params()

        for p in ckks_params:
            batch_size = p["ringDim"] // 2
            if np.prod(BLOCK_SHAPE) > batch_size:
                continue

            cc, keys = gen_crypto_context(p)

            try:
                if prepare_context is not None:
                    prepare_context(cc, keys)

                for rows, cols in SHAPES:
                    matrix = _generate_case_array(rows, cols)
                    expected = np.asarray(np_fn(matrix))

                    for order_name, order_value in ORDERS:
                        for mode in MODES:
                            with self.subTest(
                                op=tag,
                                order=order_name,
                                rows=rows,
                                cols=cols,
                                block_shape=BLOCK_SHAPE,
                                mode=mode,
                                ringDim=p["ringDim"],
                            ):
                                result = None
                                ctm = None
                                ctm_res = None

                                try:
                                    ctm = onp.block_array(
                                        cc=cc,
                                        data=matrix,
                                        block_shape=BLOCK_SHAPE,
                                        batch_size=batch_size,
                                        order=order_value,
                                        fhe_type="C",
                                        mode=mode,
                                        public_key=keys.publicKey,
                                    )

                                    if prepare_tensor is not None:
                                        prepare_tensor(keys, ctm)

                                    ctm_res = fhe_fn(ctm)
                                    result = ctm_res.decrypt(
                                        keys.secretKey,
                                        unpack_type="original",
                                    )

                                    self.assertArrayClose(
                                        actual=np.asarray(result),
                                        expected=expected,
                                        atol=ATOL,
                                    )

                                except Exception:
                                    self._record_case(
                                        params={
                                            "case": "block_matrix_unary",
                                            "op": tag,
                                            "rows": rows,
                                            "cols": cols,
                                            "block_shape": BLOCK_SHAPE,
                                            "order": order_name,
                                            "mode": mode,
                                            "ringDim": p["ringDim"],
                                        },
                                        input_data={"matrix": matrix},
                                        expected=expected,
                                        result=result,
                                    )
                                    raise

                                finally:
                                    del ctm
                                    del ctm_res
                                    del result
                                    gc.collect()
            finally:
                del cc, keys
                gc.collect()

    def _run_binary(self, tag, np_fn, fhe_fn, *, prepare_context=None):
        ckks_params = load_ckks_params()

        for p in ckks_params:
            batch_size = p["ringDim"] // 2
            if np.prod(BLOCK_SHAPE) > batch_size:
                continue

            cc, keys = gen_crypto_context(p)

            try:
                if prepare_context is not None:
                    prepare_context(cc, keys)

                for rows, cols in SHAPES:
                    lhs = _generate_case_array(rows, cols, operand=0)
                    rhs = _generate_case_array(rows, cols, operand=1)
                    expected = np.asarray(np_fn(lhs, rhs))

                    for order_name, order_value in ORDERS:
                        for mode in MODES:
                            with self.subTest(
                                op=tag,
                                order=order_name,
                                rows=rows,
                                cols=cols,
                                block_shape=BLOCK_SHAPE,
                                mode=mode,
                                ringDim=p["ringDim"],
                            ):
                                result = None
                                ctm_lhs = None
                                ctm_rhs = None
                                ctm_res = None

                                try:
                                    ctm_lhs = onp.block_array(
                                        cc=cc,
                                        data=lhs,
                                        block_shape=BLOCK_SHAPE,
                                        batch_size=batch_size,
                                        order=order_value,
                                        fhe_type="C",
                                        mode=mode,
                                        public_key=keys.publicKey,
                                    )
                                    ctm_rhs = onp.block_array(
                                        cc=cc,
                                        data=rhs,
                                        block_shape=BLOCK_SHAPE,
                                        batch_size=batch_size,
                                        order=order_value,
                                        fhe_type="C",
                                        mode=mode,
                                        public_key=keys.publicKey,
                                    )

                                    ctm_res = fhe_fn(ctm_lhs, ctm_rhs)
                                    result = ctm_res.decrypt(
                                        keys.secretKey,
                                        unpack_type="original",
                                    )

                                    self.assertArrayClose(
                                        actual=np.asarray(result),
                                        expected=expected,
                                        atol=ATOL,
                                    )

                                except Exception:
                                    self._record_case(
                                        params={
                                            "case": "block_matrix_binary",
                                            "op": tag,
                                            "rows": rows,
                                            "cols": cols,
                                            "block_shape": BLOCK_SHAPE,
                                            "order": order_name,
                                            "mode": mode,
                                            "ringDim": p["ringDim"],
                                        },
                                        input_data={"lhs": lhs, "rhs": rhs},
                                        expected=expected,
                                        result=result,
                                    )
                                    raise

                                finally:
                                    del ctm_lhs
                                    del ctm_rhs
                                    del ctm_res
                                    del result
                                    gc.collect()
            finally:
                del cc, keys
                gc.collect()

    def _run_broadcast(
        self,
        tag,
        np_fn,
        fhe_fn,
        *,
        prepare_context=None,
        prepare_operands=None,
    ):
        ckks_params = load_ckks_params()

        for p in ckks_params:
            batch_size = p["ringDim"] // 2
            if np.prod(BLOCK_SHAPE) > batch_size:
                continue

            cc, keys = gen_crypto_context(p)

            try:
                if prepare_context is not None:
                    prepare_context(cc, keys)

                for rows, cols in BROADCAST_SHAPES:
                    matrix = _generate_case_array(rows, cols, operand=0)
                    vector = _generate_case_array(rows, 1, operand=1)
                    expected = np.asarray(np_fn(vector, matrix))

                    for order_name, order_value in ORDERS:
                        for mode in MODES:
                            with self.subTest(
                                op=tag,
                                order=order_name,
                                rows=rows,
                                cols=cols,
                                matrix_block_shape=BLOCK_SHAPE,
                                vector_block_shape=BROADCAST_VECTOR_BLOCK_SHAPE,
                                mode=mode,
                                ringDim=p["ringDim"],
                            ):
                                result = None
                                ctm = None
                                ctv = None
                                ctm_res = None

                                try:
                                    ctm = onp.block_array(
                                        cc=cc,
                                        data=matrix,
                                        block_shape=BLOCK_SHAPE,
                                        batch_size=batch_size,
                                        order=order_value,
                                        fhe_type="C",
                                        mode=mode,
                                        public_key=keys.publicKey,
                                    )
                                    ctv = onp.block_array(
                                        cc=cc,
                                        data=vector,
                                        block_shape=BROADCAST_VECTOR_BLOCK_SHAPE,
                                        batch_size=batch_size,
                                        order=order_value,
                                        fhe_type="C",
                                        mode=mode,
                                        public_key=keys.publicKey,
                                    )

                                    if prepare_operands is not None:
                                        prepare_operands(keys, ctv, ctm)

                                    ctm_res = fhe_fn(ctv, ctm)
                                    result = ctm_res.decrypt(
                                        keys.secretKey,
                                        unpack_type="original",
                                    )

                                    self.assertArrayClose(
                                        actual=np.asarray(result),
                                        expected=expected,
                                        atol=ATOL,
                                    )

                                except Exception:
                                    self._record_case(
                                        params={
                                            "case": "block_vector_broadcasting",
                                            "op": tag,
                                            "rows": rows,
                                            "cols": cols,
                                            "matrix_block_shape": BLOCK_SHAPE,
                                            "vector_block_shape": (BROADCAST_VECTOR_BLOCK_SHAPE),
                                            "order": order_name,
                                            "mode": mode,
                                            "ringDim": p["ringDim"],
                                        },
                                        input_data={
                                            "matrix": matrix,
                                            "vector": vector,
                                        },
                                        expected=expected,
                                        result=result,
                                    )
                                    raise

                                finally:
                                    del ctm
                                    del ctv
                                    del ctm_res
                                    del result
                                    gc.collect()
            finally:
                del cc, keys
                gc.collect()

    def _run_matrix_vector(
        self,
        tag,
        np_fn,
        fhe_fn,
        *,
        matrix_order_name,
        matrix_order,
        vector_order_name,
        vector_order,
        prepare_context=None,
        prepare_operands=None,
    ):
        ckks_params = load_ckks_params()

        for p in ckks_params:
            batch_size = p["ringDim"] // 2
            if np.prod(BLOCK_SHAPE) > batch_size:
                continue

            cc, keys = gen_crypto_context(p)

            try:
                if prepare_context is not None:
                    prepare_context(cc, keys)

                for rows, cols in SHAPES:
                    matrix = _generate_case_array(rows, cols, operand=0)
                    vector = _generate_case_array(cols, operand=1)
                    expected = np.asarray(np_fn(matrix, vector))

                    for mode in MODES:
                        with self.subTest(
                            op=tag,
                            matrix_order=matrix_order_name,
                            vector_order=vector_order_name,
                            rows=rows,
                            cols=cols,
                            matrix_block_shape=BLOCK_SHAPE,
                            vector_block_shape=VECTOR_BLOCK_SHAPE,
                            mode=mode,
                            ringDim=p["ringDim"],
                        ):
                            result = None
                            ctm = None
                            ctv = None
                            ctv_res = None

                            try:
                                ctm = onp.block_array(
                                    cc=cc,
                                    data=matrix,
                                    block_shape=BLOCK_SHAPE,
                                    batch_size=batch_size,
                                    order=matrix_order,
                                    fhe_type="C",
                                    mode=mode,
                                    public_key=keys.publicKey,
                                )
                                ctv = onp.block_array(
                                    cc=cc,
                                    data=vector,
                                    block_shape=VECTOR_BLOCK_SHAPE,
                                    batch_size=batch_size,
                                    order=vector_order,
                                    fhe_type="C",
                                    mode=mode,
                                    public_key=keys.publicKey,
                                    target_cols=BLOCK_SHAPE[1],
                                )

                                if prepare_operands is not None:
                                    prepare_operands(keys, ctm, ctv)

                                ctv_res = fhe_fn(ctm, ctv)
                                result = ctv_res.decrypt(
                                    keys.secretKey,
                                    unpack_type="original",
                                )

                                self.assertArrayClose(
                                    actual=np.asarray(result),
                                    expected=expected,
                                    atol=ATOL,
                                )

                            except Exception:
                                self._record_case(
                                    params={
                                        "case": "block_matrix_vector",
                                        "op": tag,
                                        "rows": rows,
                                        "cols": cols,
                                        "matrix_block_shape": BLOCK_SHAPE,
                                        "vector_block_shape": VECTOR_BLOCK_SHAPE,
                                        "matrix_order": matrix_order_name,
                                        "vector_order": vector_order_name,
                                        "mode": mode,
                                        "ringDim": p["ringDim"],
                                    },
                                    input_data={
                                        "matrix": matrix,
                                        "vector": vector,
                                    },
                                    expected=expected,
                                    result=result,
                                )
                                raise

                            finally:
                                del ctm
                                del ctv
                                del ctv_res
                                del result
                                gc.collect()
            finally:
                del cc, keys
                gc.collect()

    def _run_vector_inner_product(
        self,
        tag,
        np_fn,
        fhe_fn,
        *,
        order_name,
        order,
        prepare_context=None,
    ):
        ckks_params = load_ckks_params()

        for p in ckks_params:
            batch_size = p["ringDim"] // 2
            if np.prod(VECTOR_BLOCK_SHAPE) > batch_size:
                continue

            cc, keys = gen_crypto_context(p)

            try:
                if prepare_context is not None:
                    prepare_context(cc, keys)

                for size in VECTOR_SIZES:
                    lhs = _generate_case_array(size, operand=0)
                    rhs = _generate_case_array(size, operand=1)
                    expected = np.asarray(np_fn(lhs, rhs))

                    for mode in MODES:
                        with self.subTest(
                            op=tag,
                            order=order_name,
                            size=size,
                            block_shape=VECTOR_BLOCK_SHAPE,
                            mode=mode,
                            ringDim=p["ringDim"],
                        ):
                            result = None
                            ctv_lhs = None
                            ctv_rhs = None
                            ctv_res = None

                            try:
                                ctv_lhs = onp.block_array(
                                    cc=cc,
                                    data=lhs,
                                    block_shape=VECTOR_BLOCK_SHAPE,
                                    batch_size=batch_size,
                                    order=order,
                                    fhe_type="C",
                                    mode=mode,
                                    public_key=keys.publicKey,
                                )
                                ctv_rhs = onp.block_array(
                                    cc=cc,
                                    data=rhs,
                                    block_shape=VECTOR_BLOCK_SHAPE,
                                    batch_size=batch_size,
                                    order=order,
                                    fhe_type="C",
                                    mode=mode,
                                    public_key=keys.publicKey,
                                )

                                ctv_res = fhe_fn(ctv_lhs, ctv_rhs)
                                result = ctv_res.decrypt(
                                    keys.secretKey,
                                    unpack_type="original",
                                )

                                self.assertArrayClose(
                                    actual=np.asarray(result),
                                    expected=expected,
                                    atol=ATOL,
                                )

                            except Exception:
                                self._record_case(
                                    params={
                                        "case": "block_vector_inner_product",
                                        "op": tag,
                                        "size": size,
                                        "block_shape": VECTOR_BLOCK_SHAPE,
                                        "order": order_name,
                                        "mode": mode,
                                        "ringDim": p["ringDim"],
                                    },
                                    input_data={"lhs": lhs, "rhs": rhs},
                                    expected=expected,
                                    result=result,
                                )
                                raise

                            finally:
                                del ctv_lhs
                                del ctv_rhs
                                del ctv_res
                                del result
                                gc.collect()
            finally:
                del cc, keys
                gc.collect()
