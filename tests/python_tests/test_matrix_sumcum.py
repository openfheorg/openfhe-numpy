import numpy as np
import openfhe_numpy as onp

from .core import *


def fhe_matrix_cumulative_sum(params, data, axis=0, order=onp.ROW_MAJOR):
    """
    Generic matrix cumulative sum operation.
    - params: CKKS parameters dictionary
    - data: list containing the input matrix
    - axis: 0 for row-wise, 1 for column-wise cumulative sum
    - order: ROW_MAJOR or COLUMN_MAJOR ordering
    """
    params_copy = params.copy()
    matrix = np.array(data[0])

    # Ensure sufficient multiplicative depth
    required_depth = len(matrix) if axis == 0 else len(matrix[0])
    if params_copy["multiplicativeDepth"] < required_depth:
        params_copy["multiplicativeDepth"] = required_depth + 1

    with suppress_stdout(False):
        # Generate crypto context
        cc, keys = gen_crypto_context(params_copy)
        total_slots = params_copy["ringDim"] // 2

        # Encrypt matrix
        ctm_matrix = onp.array(
            cc=cc,
            data=matrix,
            batch_size=total_slots,
            order=order,
            fhe_type="C",
            mode="zero",
            public_key=keys.publicKey,
        )

        # Generate appropriate keys based on axis
        if axis == 0:  # cumulative_sum rows
            if order == onp.ROW_MAJOR:
                onp.gen_accumulate_rows_key(keys.secretKey, ctm_matrix.ncols)
            else:  # order == onp.COL_MAJOR:
                onp.gen_accumulate_cols_key(keys.secretKey, ctm_matrix.ncols)
        else:  # cumulative_sum cols
            if order == onp.ROW_MAJOR:
                onp.gen_accumulate_cols_key(keys.secretKey, ctm_matrix.ncols)
            else:  # order == onp.COL_MAJOR:
                onp.gen_accumulate_rows_key(keys.secretKey, ctm_matrix.ncols)

        # Perform cumulative sum
        ctm_result = onp.cumulative_sum(ctm_matrix, axis=axis)

        # Decrypt result
        result = ctm_result.decrypt(keys.secretKey, unpack_type="original")

    return result


class TestMatrixcumulative_sum(MainUnittest):
    """Test class for matrix cumulative sum operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for matrix cumulative sum operations."""
        # use Numpy API < 2.0 for testing purpose. Update in the next release
        operations = [
            ("rows", 0, lambda A: np.cumsum(A, axis=0)),  # Row-wise
            ("cols", 1, lambda A: np.cumsum(A, axis=1)),  # Column-wise
        ]

        # Add ordering options
        orders = [("row_major", onp.ROW_MAJOR), ("col_major", onp.COL_MAJOR)]

        ckks_param_list = load_ckks_params()
        matrix_sizes = [2, 3, 8, 16]
        test_counter = 1

        for op_name, axis, np_fn in operations:
            for order_name, order_value in orders:
                for param in ckks_param_list:
                    for size in matrix_sizes:
                        # Generate random test matrix
                        A = generate_random_array(size)

                        # Calculate expected result directly
                        expected = np_fn(A)

                        # Create test name with descriptive format
                        test_name = f"cumulative_sum_{op_name}_{order_name}_{test_counter:03d}_ring_{param['ringDim']}_size_{size}"

                        # Create a closure to capture the current axis and ordering values
                        def func(current_axis, current_order):
                            return lambda p, d: fhe_matrix_cumulative_sum(
                                p, d, current_axis, current_order
                            )

                        # Generate the test case
                        cls.generate_test_case(
                            func=func(axis, order_value),
                            test_name=test_name,
                            params=param,
                            input_data=[A],
                            expected=expected,
                            compare_fn=onp.check_equality,
                            debug=True,
                        )

                        test_counter += 1


if __name__ == "__main__":
    TestMatrixcumulative_sum.run_test_summary("Matrix Cumulative Sum", verbose=True)
