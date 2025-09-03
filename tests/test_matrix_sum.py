import numpy as np
import openfhe_numpy as onp

from core import *

###
# Note: Column-/row-wise cumulative sum may require deeper multiplicative
# depth or larger ring dimensions for accurate results. Small ring dimensions
# (<4096) might introduce approximation errors.
###


def fhe_matrix_sum(params, data, axis=None, order=onp.ROW_MAJOR):
    """
    Generic matrix sum operation.
    - params: CKKS parameters dictionary
    - data: list containing the input matrix
    - axis: None for total sum, 0 for row-wise, 1 for column-wise sum
    - order: ROW_MAJOR or COLUMN_MAJOR ordering
    """
    params_copy = params.copy()
    matrix = np.array(data[0])

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
        if axis is None:  # Total sum
            onp.gen_sum_key(keys.secretKey)

        elif axis == 0:  # Row sum (sum along rows)
            if ctm_x.order == onp.ROW_MAJOR:
                ctm_x.extra["rowkey"] = onp.sum_row_keys(
                    keys.secretKey, ctm_x.ncols, ctm_x.batch_size
                )
            elif ctm_x.order == onp.COL_MAJOR:
                ctm_x.extra["colkey"] = onp.sum_col_keys(keys.secretKey, ctm_x.nrows)

            else:
                raise ValueError("Invalid order.")
        elif axis == 1:  # Column sum (sum along columns)
            if ctm_x.order == onp.ROW_MAJOR:
                ctm_x.extra["colkey"] = onp.sum_col_keys(keys.secretKey, ctm_x.ncols)
            elif ctm_x.order == onp.COL_MAJOR:
                ctm_x.extra["rowkey"] = onp.sum_row_keys(
                    keys.secretKey, ctm_x.nrows, ctm_x.batch_size
                )
            else:
                raise ValueError("Invalid order.")

        # Perform sum operation
        if axis is None:
            ctm_result = onp.sum(ctm_x)  # Total sum without axis parameter
        else:
            assert isinstance(axis, int), f"axis should be int but got {axis!r}"
            ctm_result = onp.sum(ctm_x, axis)  # Sum along specified axis

        # Decrypt result
        result = ctm_result.decrypt(keys.secretKey, unpack_type="original")

    return result


class TestMatrixSum(MainUnittest):
    """Test class for matrix sum operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for matrix sum operations."""
        operations = [
            ("total", None, lambda A: np.sum(A)),  # Total sum
            ("rows", 0, lambda A: np.sum(A, axis=0)),  # Row sum
            ("cols", 1, lambda A: np.sum(A, axis=1)),  # Column sum
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
                        # Skip large matrices for total sum tests
                        if op_name == "total" and size > 3:
                            continue

                        # Generate random test matrix
                        A = generate_random_array(size)

                        # Calculate expected result directly
                        expected = np_fn(A)

                        # Create test name with descriptive format
                        test_name = f"sum_{op_name}_{order_name}_{test_counter:03d}_ring_{param['ringDim']}_size_{size}"

                        # Create a closure to capture the current axis and ordering values
                        def func(current_axis, current_order):
                            return lambda p, d: fhe_matrix_sum(p, d, current_axis, current_order)

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
    TestMatrixSum.run_test_summary("Matrix Sum", debug=True)
