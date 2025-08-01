import numpy as np
import openfhe_numpy as onp

from tests.core.test_framework import MainUnittest
from tests.core.test_utils import generate_random_array, suppress_stdout
from tests.core.test_crypto_context import load_ckks_params, gen_crypto_context


"""
Note: Mean operations may require sufficient multiplicative depth
and ring dimension to accommodate division operations. Small ring
dimensions (<4096) may result in higher approximation errors.
"""


def fhe_matrix_mean(params, data, axis=None, order=onp.ROW_MAJOR):
    """
    Generic matrix mean operation.
    - params: CKKS parameters dictionary
    - data: list containing the input matrix
    - axis: None for total mean, 0 for row-wise, 1 for column-wise mean
    - order: ROW_MAJOR or COLUMN_MAJOR ordering
    """
    params_copy = params.copy()
    matrix = np.array(data[0])

    # Ensure sufficient multiplicative depth for division
    if params_copy["multiplicativeDepth"] < 3:
        params_copy["multiplicativeDepth"] = 3

    with suppress_stdout(False):
        # Generate crypto context
        cc, keys = gen_crypto_context(params_copy)
        total_slots = params_copy["ringDim"] // 2

        # Encrypt matrix
        ctm_x = onp.array(
            cc=cc,
            data=matrix,
            batch_size=total_slots,
            order=order,
            fhe_type="C",
            mode="zero",
            public_key=keys.publicKey,
        )

        # Generate appropriate keys based on axis
        if axis is None:  # Total mean
            cc.EvalSumKeyGen(keys.secretKey)
        elif axis == 0:  # Row mean (mean along rows)
            if order == onp.ROW_MAJOR:
                onp.sum_row_keys(keys.secretKey, ctm_x.ncols, ctm_x.batch_size)
            elif order == onp.COL_MAJOR:
                onp.sum_col_keys(keys.secretKey, ctm_x.nrows)
            else:
                raise ValueError("Invalid order.")
        elif axis == 1:  # Column mean (mean along columns)
            if order == onp.ROW_MAJOR:
                onp.sum_col_keys(keys.secretKey, ctm_x.ncols)
            elif order == onp.COL_MAJOR:
                onp.sum_row_keys(keys.secretKey, ctm_x.nrows, ctm_x.batch_size)
            else:
                raise ValueError("Invalid order.")

        # Perform mean operation
        ctm_result = onp.mean(ctm_x, axis)

        # Decrypt result
        result = ctm_result.decrypt(keys.secretKey, unpack_type="original")

    return result


class TestMatrixMean(MainUnittest):
    """Test class for matrix mean operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for matrix mean operations."""
        operations = [
            ("total", None, lambda A: np.mean(A)),  # Total mean
            ("rows", 0, lambda A: np.mean(A, axis=0)),  # Row mean
            ("cols", 1, lambda A: np.mean(A, axis=1)),  # Column mean
        ]

        # Add ordering options
        orders = [("row_major", onp.ROW_MAJOR), ("col_major", onp.COLUMN_MAJOR)]

        ckks_param_list = load_ckks_params()
        matrix_sizes = [2, 3, 4]  # Smaller sizes for mean operations
        test_counter = 1

        for op_name, axis, np_fn in operations:
            for order_name, order_value in orders:
                for param in ckks_param_list:
                    for size in matrix_sizes:
                        # Skip tests with very small ring dimensions for stability
                        if param["ringDim"] < 4096 and size > 2:
                            continue

                        # Generate random test matrix
                        A = generate_random_array(size)

                        # Calculate expected result directly
                        expected = np_fn(A)

                        # Create test name with descriptive format
                        test_name = f"mean_{op_name}_{order_name}_{test_counter:03d}_ring_{param['ringDim']}_size_{size}"

                        # Create a closure to capture the current axis and ordering values
                        def make_func(current_axis, current_order):
                            return lambda p, d: fhe_matrix_mean(
                                p, d, current_axis, current_order
                            )

                        # Generate the test case
                        cls.generate_test_case(
                            func=make_func(axis, order_value),
                            test_name=test_name,
                            params=param,
                            input_data=[A],
                            expected=expected,
                            compare_fn=onp.check_equality,
                            debug=True,
                        )

                        test_counter += 1


if __name__ == "__main__":
    TestMatrixMean.run_test_summary("Matrix Mean", debug=True)
