import gc
import numpy as np
from openfhe import *
import openfhe_numpy as onp
from core import *


# ckks_params_block.csv uses ringDim=512, so the CKKS slot capacity is 256.
# A 17 x 17 matrix has 289 entries, forcing block packing because 289 > 256.
SIZES = [17]
BLOCK_PARAMS_CSV = CRYPTO_PARAMS_DIR / "ckks_params_block.csv"


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

    def test_block_matrix_vector_product(self):
        ckks_params = load_ckks_params(BLOCK_PARAMS_CSV)

        for p in ckks_params:
            params = _ensure_depth(p, 4)
            slot_capacity = params["ringDim"] // 2
            batch_size = slot_capacity

            if batch_size <= 0:
                continue

            cc, keys = gen_crypto_context(params)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            try:
                for size in SIZES:
                    self.assertGreater(size * size, slot_capacity)

                    A = generate_random_array(rows=size, cols=size)
                    b = generate_random_array(rows=size)
                    expected = A @ b

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
                        self._record_case(
                            params={
                                "case": self.case_name,
                                "size": size,
                                "matrix_slots": size * size,
                                "slot_capacity": slot_capacity,
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
