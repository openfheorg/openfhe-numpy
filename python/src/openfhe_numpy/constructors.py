from openfhe_numpy.utils import get_shape, next_power_of_two, is_power_of_two
from openfhe_numpy.tensor import ctarray, ptarray
from openfhe_numpy.config import MatrixEncoding, DataType
from openfhe_numpy import utils


def array(
    cc,
    data: list,
    total_slots: int,
    batch_size: int = 1,
    encoding_type: int = MatrixEncoding.ROW_MAJOR,
    type: str = DataType.CIPHERTEXT,
    pub_key=None,
):
    """
    Construct either a ciphertext (ctarray) or plaintext (ptarray) from raw input data.

    Parameters
    ----------
    cc : CryptoContext
        The OpenFHE CryptoContext.
    data : list
        Input list or matrix data.
    total_slots : int
        Number of total CKKS slots.
    block_size : int
        Repetition or column block size.
    encoding_type : int
        MatrixEncoding.ROW_MAJOR or COL_MAJOR.
    type : str
        DataType.CIPHERTEXT or PLAINTEXT.
    pub_key : optional
        Public key needed for encryption.

    Returns
    -------
    ctarray or ptarray
    """
    org_rows, org_cols, ndim = get_shape(data)

    if ndim == 2:
        ncols = next_power_of_two(org_cols)
        packed_data = ravel_mat(data, total_slots, ncols, encoding_type)
    else:
        ncols = batch_size
        packed_data = ravel_vec(data, total_slots, ncols, encoding_type)

    plaintext = cc.MakeCKKSPackedPlaintext(packed_data)

    if type == DataType.PLAINTEXT:
        return ptarray(plaintext, (org_rows, org_cols), ndim, total_slots, ncols, encoding_type)

    if pub_key is None:
        raise ValueError("Public key must be provided for ciphertext encoding.")

    ciphertext = cc.Encrypt(pub_key, plaintext)
    return ctarray(ciphertext, (org_rows, org_cols), ndim, total_slots, ncols, encoding_type)


def ravel_mat(data, total_slots, row_size=1, order=MatrixEncoding.ROW_MAJOR, reps=1):
    """
    Encode a 2D matrix into a CKKS plaintext.

    Parameters
    ----------
    data : list of list
    row_size : int
        Block size per row.
    order : MatrixEncoding
    reps : int
        Number of repetitions

    Returns
    -------
    Plaintext
    """

    if order == MatrixEncoding.ROW_MAJOR:
        packed_data = utils.pack_mat_row_wise(data, row_size, total_slots, reps)
    elif order == MatrixEncoding.COL_MAJOR:
        packed_data = utils.pack_mat_col_wise(data, row_size, total_slots, reps)
    else:
        raise ValueError("Unsupported encoding order")

    return packed_data


def ravel_vec(data, total_slots, row_size=1, order=MatrixEncoding.ROW_MAJOR):
    """
    Encode a 1D vector into a CKKS plaintext.

    Parameters
    ----------
    data : list
    row_size : int
        Number of repetitions.
    order : MatrixEncoding

    Returns
    -------
    Plaintext
    """
    from openfhe_numpy import utils

    if row_size < 1:
        raise ValueError("Number of repetitions must be > 0")

    if row_size == 1 and order == MatrixEncoding.ROW_MAJOR:
        raise ValueError("Can't encode row-wise with row_size = 1")

    if not is_power_of_two(row_size):
        raise ValueError("Repetition count must be a power of two")

    if order == MatrixEncoding.ROW_MAJOR:
        packed_data = utils.pack_vec_row_wise(data, row_size, total_slots)
    elif order == MatrixEncoding.COL_MAJOR:
        packed_data = utils.pack_vec_col_wise(data, row_size, total_slots)
    else:
        raise ValueError("Unsupported encoding order")

    return packed_data
