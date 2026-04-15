import gc
import numpy as np
from openfhe import *
import openfhe_numpy as onp
from core import *

###
# Note: Column-/row-wise cumulative sum may require deeper multiplicative
# depth or larger ring dimensions for accurate results. Small ring dimensions
# (<4096) might introduce approximation errors.
###

SIZES = [2, 3, 8, 16]
ORDERS = [("row_major", onp.ROW_MAJOR), ("col_major", onp.COL_MAJOR)]


def _ensure_depth(p: dict, min_depth: int = 3) -> dict:
    params = p.copy()
    if params.get("multiplicativeDepth", 0) < min_depth:
        params["multiplicativeDepth"] = min_depth
    return params


class TestMatrixSum(MainUnittest):
    def test_total_sum(self):
        ckks_params = load_ckks_params()

        for p in ckks_params:
            params = _ensure_depth(p, 3)
            batch_size = params["ringDim"] // 2
            cc, keys = gen_crypto_context(params)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            try:
                for size in SIZES:
                    if params["ringDim"] < 4096 and size > 2:
                        continue
                    if size > batch_size:
                        continue

                    A = generate_random_array(rows=size, cols=size)
                    expected = np.sum(A)
                    onp.gen_sum_key(keys.secretKey)

                    for order_name, order_value in ORDERS:
                        with self.subTest(order=order_name, size=size, ringDim=params["ringDim"]):
                            result = None
                            ctm = None
                            ctm_result = None
                            try:
                                ctm = onp.array(
                                    cc=cc,
                                    data=A,
                                    batch_size=batch_size,
                                    order=order_value,
                                    fhe_type="C",
                                    mode="zero",
                                    public_key=keys.publicKey,
                                )
                                ctm_result = onp.sum(ctm)
                                result = ctm_result.decrypt(keys.secretKey, unpack_type="original")
                                self.assertArrayClose(actual=result, expected=expected)
                            except Exception:
                                self._record_case(
                                    params={
                                        "case": "total_sum",
                                        "size": size,
                                        "ringDim": p["ringDim"],
                                    },
                                    input_data={"A": A},
                                    expected=expected,
                                    result=result,
                                )
                                raise
                            finally:
                                del ctm, ctm_result, result
                                gc.collect()
            finally:
                del cc, keys
                gc.collect()


class TestMatrixRowSum(MainUnittest):
    def test_row_sum(self):
        ckks_params = load_ckks_params()

        for p in ckks_params:
            params = _ensure_depth(p, 3)
            batch_size = params["ringDim"] // 2
            cc, keys = gen_crypto_context(params)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            try:
                for size in SIZES:
                    if params["ringDim"] < 4096 and size > 2:
                        continue
                    if size > batch_size:
                        continue

                    A = generate_random_array(rows=size, cols=size)
                    expected = np.sum(A, axis=0)

                    for order_name, order_value in ORDERS:
                        with self.subTest(order=order_name, size=size, ringDim=params["ringDim"]):
                            result = None
                            ctm = None
                            ctm_result = None
                            try:
                                ctm = onp.array(
                                    cc=cc,
                                    data=A,
                                    batch_size=batch_size,
                                    order=order_value,
                                    fhe_type="C",
                                    mode="zero",
                                    public_key=keys.publicKey,
                                )
                                if order_value == onp.ROW_MAJOR:
                                    ctm.extra["rowkey"] = onp.sum_row_keys(
                                        keys.secretKey,
                                        ctm.ncols,
                                        ctm.batch_size,
                                    )
                                else:
                                    ctm.extra["colkey"] = onp.sum_col_keys(
                                        keys.secretKey, ctm.nrows
                                    )
                                ctm_result = onp.sum(ctm, axis=0)
                                result = ctm_result.decrypt(keys.secretKey, unpack_type="original")
                                self.assertArrayClose(actual=result, expected=expected)
                            except Exception:
                                self._record_case(
                                    params={
                                        "case": "row_sum",
                                        "size": size,
                                        "ringDim": p["ringDim"],
                                    },
                                    input_data={"A": A},
                                    expected=expected,
                                    result=result,
                                )
                                raise
                            finally:
                                del ctm, ctm_result, result
                                gc.collect()
            finally:
                del cc, keys
                gc.collect()


class TestMatrixColSum(MainUnittest):
    def test_col_sum(self):
        ckks_params = load_ckks_params()

        for p in ckks_params:
            params = _ensure_depth(p, 3)
            batch_size = params["ringDim"] // 2
            cc, keys = gen_crypto_context(params)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            try:
                for size in SIZES:
                    if params["ringDim"] < 4096 and size > 2:
                        continue
                    if size > batch_size:
                        continue

                    A = generate_random_array(rows=size, cols=size)
                    expected = np.sum(A, axis=1)

                    for order_name, order_value in ORDERS:
                        with self.subTest(order=order_name, size=size, ringDim=params["ringDim"]):
                            result = None
                            ctm = None
                            ctm_result = None
                            try:
                                ctm = onp.array(
                                    cc=cc,
                                    data=A,
                                    batch_size=batch_size,
                                    order=order_value,
                                    fhe_type="C",
                                    mode="zero",
                                    public_key=keys.publicKey,
                                )
                                if order_value == onp.ROW_MAJOR:
                                    ctm.extra["colkey"] = onp.sum_col_keys(
                                        keys.secretKey, ctm.ncols
                                    )
                                else:
                                    ctm.extra["rowkey"] = onp.sum_row_keys(
                                        keys.secretKey,
                                        ctm.nrows,
                                        ctm.batch_size,
                                    )
                                ctm_result = onp.sum(ctm, axis=1)
                                result = ctm_result.decrypt(keys.secretKey, unpack_type="original")
                                self.assertArrayClose(actual=result, expected=expected)
                            except Exception:
                                self._record_case(
                                    params={
                                        "case": "column_sum",
                                        "size": size,
                                        "ringDim": p["ringDim"],
                                    },
                                    input_data={"A": A},
                                    expected=expected,
                                    result=result,
                                )
                                raise
                            finally:
                                del ctm, ctm_result, result
                                gc.collect()
            finally:
                del cc, keys
                gc.collect()
