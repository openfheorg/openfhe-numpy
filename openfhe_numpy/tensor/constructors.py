import numpy as np
import openfhe
import copy

from .tensor import BaseTensor, FHETensor
from .ptarray import PTArray
from .ctarray import CTArray
from .block_tensor import BlockFHETensor
from .block_ctarray import BlockCTArray
from openfhe_numpy.utils import utils
from openfhe_numpy.config import MatrixOrder, DataType


# TODO: constructor for block matrix
def _get_block_dimensions(data, slots) -> tuple[int, int]:
    pass
    # org_rows, org_cols, ndim = utils.get_shape(data)
    # nrows = next_power_of_two(org_rows)
    # ncols = next_power_of_two(org_cols)

    # if ndim == 1:
    #     if ncols > slots:
    #         nblocks = ncols // (slots // 2)

    #     cell_rows = 1
    #     cell_cols = slots // 2
    # else:
    #     if nrows > slots:
    #         nblocks = nrows // (slots // 2)

    #     cell_rows = slots // 2
    #     cell_cols = ncols


def array(
    cc: openfhe.CryptoContext,
    data: list | tuple | np.ndarray | int,
    slots: int = -1,
    ncols: int = 0,
    order: int = MatrixOrder.ROW_MAJOR,
    type: str = DataType.CIPHERTEXT,
    public_key=None,
) -> FHETensor:
    """
    Construct either a ciphertext (FHETensor) or plaintext (PTArray) from raw input data.

    Parameters
    ----------
    cc : CryptoContext
        The OpenFHE CryptoContext.
    data : list
        Input list or matrix data.
    slots : int
        Number of total CKKS slots.
    ncols : int
        Size of a row.
    order : int
        MatrixOrder.ROW_MAJOR or COL_MAJOR.
    type : str
        DataType.CIPHERTEXT or PLAINTEXT.
    public_key : optional
        Public key needed for encryption.

    Returns
    -------
    FHETensor Object
    """

    org_rows, org_cols, ndim = utils.get_shape(data)

    if slots == -1:
        slots = cc.GetBatchSize()

    if isinstance(data, int):
        packed_data = [data] * slots
    else:
        if ndim == 2:
            if ncols == 0:
                ncols = utils.next_power_of_two(org_cols)
            packed_data = ravel_matrix(data, slots, ncols, order)
        else:
            if ncols == 0:
                ncols = 1
            packed_data = ravel_vector(data, slots, ncols, order)
    print("==================================")
    print("packed_data = ", packed_data[:128])
    print("slots = ", slots)
    print("ncols = ", ncols)
    print("==================================")
    plaintext = cc.MakeCKKSPackedPlaintext(packed_data)

    if type == DataType.PLAINTEXT:
        result = PTArray(plaintext, (org_rows, org_cols), ndim, slots, ncols, order)
    else:
        if public_key is None:
            raise ValueError("Public key must be provided for ciphertext encoding.")

        ciphertext = cc.Encrypt(public_key, plaintext)
        result = CTArray(ciphertext, (org_rows, org_cols), slots, ncols, order)

    result.set_ncols(ncols)
    result.set_batch_size(slots)
    return result


def ravel_matrix(data, slots, ncols=1, order=MatrixOrder.ROW_MAJOR, reps=1):
    """
    Encode a 2D matrix into a CKKS plaintext.

    Parameters
    ----------
    data : list of list
    ncols : int
        Block size per row.
    order : MatrixOrder
    reps : int
        Number of repetitions

    Returns
    -------
    Plaintext
    """

    if order == MatrixOrder.ROW_MAJOR:
        packed_data = utils.pack_mat_row_wise(data, ncols, slots, reps)
    elif order == MatrixOrder.COL_MAJOR:
        packed_data = utils.pack_mat_col_wise(data, ncols, slots, reps)
    else:
        raise ValueError("Unsupported encoding order")

    return packed_data


def ravel_vector(data, slots, ncols=1, order=MatrixOrder.ROW_MAJOR):
    """
    Encode a 1D vector into a CKKS plaintext.

    Parameters
    ----------
    data : list
    ncols : int
        Number of repetitions.
    order : MatrixOrder

    Returns
    -------
    Plaintext
    """

    if ncols < 1:
        raise ValueError("Number of repetitions must be > 0")

    # if ncols == 1 and order == MatrixOrder.ROW_MAJOR:
    #     raise ValueError("Can't encode row-wise with ncols = 1")

    if not utils.is_power_of_two(ncols):
        raise ValueError("Repetition count must be a power of two")

    if order == MatrixOrder.ROW_MAJOR:
        packed_data = utils.pack_vec_row_wise(data, ncols, slots)
    elif order == MatrixOrder.COL_MAJOR:
        packed_data = utils.pack_vec_col_wise(data, ncols, slots)
    else:
        raise ValueError("Unsupported encoding order")

    return packed_data
