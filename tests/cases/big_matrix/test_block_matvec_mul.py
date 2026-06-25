import gc
import numpy as np
from openfhe import *
import openfhe_numpy as onp
from core import *


SIZES = [5, 8]
BLOCK_BATCH_SIZE = 4


def _ensure_depth(params: dict, min_depth: int = 4) -> dict:
    p = params.copy()
    if p.get("multiplicativeDepth", 0) < min_depth:
        p["multiplicativeDepth"] = min_depth
    return p


def _make_block_array(
    cc,
    data,
    batch_size,
    order,
    mode,
    fhe_type,
    public_key=None,
    target_cols=None,
    compact=False,
):
    kwargs = {
        "cc": cc,
        "data": data,
        "batch_size": batch_size,
        "block_shape": None,
        "order": order,
        "mode": mode,
        "fhe_type": fhe_type,
        "compact": compact,
    }

    if public_key is not None:
        kwargs["public_key"] = public_key

    if target_cols is not None:
        kwargs["target_cols"] = target_cols

    return onp.block_array(**kwargs)


class BlockMatVecTestBase:
    """Base class for block matrix-vector multiplication tests."""

    matrix_order = None
    vector_order = None
    matrix_mode = None
    vector_mode = None
    case_name = None

    def _attach_matvec_keys(self, secret_key, block_matrix):
        """Attach sum keys to each matrix block for block matvec."""
        for block in block_matrix.data:
            if block.order == onp.ROW_MAJOR:
                block.extra["colkey"] = onp.sum_col_keys(secret_key, block.ncols)
            elif block.order == onp.COL_MAJOR:
                block.extra["rowkey"] = onp.sum_row_keys(
                    secret_key,
                    block.nrows,
                    block.batch_size * 4,
                )
            else:
                raise ValueError(f"Unsupported order: {block.order}")

    def test_block_matrix_vector_product(self):
        ckks_params = load_ckks_params()

        for p in ckks_params:
            params = _ensure_depth(p, 4)
            max_batch_size = params["ringDim"] // 2
            batch_size = min(BLOCK_BATCH_SIZE, max_batch_size)

            if batch_size <= 0:
                continue

            cc, keys = gen_crypto_context(params)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            try:
                for size in SIZES:
                    if size > max_batch_size:
                        continue
                    size = 5
                    A = generate_random_array(rows=size, cols=size)
                    b = generate_random_array(rows=size)

                    result = None
                    ctm = None
                    ctv = None
                    ctv_result = None

                    try:
                        ctm = _make_block_array(
                            cc=cc,
                            data=A,
                            batch_size=batch_size,
                            order=self.matrix_order,
                            mode=self.matrix_mode,
                            fhe_type="C",
                            public_key=keys.publicKey,
                        )

                        vector_kwargs = {
                            "cc": cc,
                            "data": b,
                            "batch_size": batch_size,
                            "order": self.vector_order,
                            "mode": self.vector_mode,
                            "fhe_type": "C",
                            "public_key": keys.publicKey,
                            "compact": True,
                        }

                        if self.vector_order == onp.ROW_MAJOR:
                            vector_kwargs["target_cols"] = ctm.get_block(0, 0).nrows

                        ctv = _make_block_array(**vector_kwargs)

                        onp.attach_block_matvec_keys(ctm, keys.secretKey)

                        ctv_result = ctm @ ctv
                        result = ctv_result.decrypt(keys.secretKey, unpack_type="original")

                        self.assertArrayClose(actual=result, expected=expected)

                    except Exception:
                        print(A, b, size, batch_size, params["ringDim"])
                        print(f"expected={expected}")
                        print(f"result={result}")
                        self._record_case(
                            params={
                                "case": self.case_name,
                                "size": size,
                                "batch_size": batch_size,
                                "ringDim": params["ringDim"],
                            },
                            input_data={"A": A, "b": b},
                            expected=expected,
                            result=result,
                        )
                        raise

                    finally:
                        del ctm, ctv, ctv_result, result
                        gc.collect()

            finally:
                del cc, keys
                gc.collect()


class TestBlockRowMajorColMajorMatVec(BlockMatVecTestBase, MainUnittest):
    """Block row-major matrix times block column-major vector."""

    matrix_order = onp.ROW_MAJOR
    vector_order = onp.COL_MAJOR
    matrix_mode = "zero"
    vector_mode = "tile"
    case_name = "block_rowmajor_colmajor_matvec"


class TestBlockColMajorRowMajorMatVec(BlockMatVecTestBase, MainUnittest):
    """Block column-major matrix times block row-major vector."""

    matrix_order = onp.COL_MAJOR
    vector_order = onp.ROW_MAJOR
    matrix_mode = "zero"
    vector_mode = "tile"
    case_name = "block_colmajor_rowmajor_matvec"
