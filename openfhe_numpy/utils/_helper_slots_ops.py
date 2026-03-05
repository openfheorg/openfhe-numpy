import openfhe


def _create_masking(indices, size):
    """
    Create a binary mask with 1s at specified indices

    Args:
        indices: List/array of indices to set to 1
        size: Total size of the mask

    Returns:
        List with 1s at indices, 0s elsewhere
    """
    mask = [0] * size
    for idx in indices:
        mask[idx] = 1
    return mask


def _get_single_element(cc, x, idx, batch_size):
    mask = _create_masking([idx], batch_size)
    pt_mask = cc.MakeCKKSPackedPlaintext(mask)
    ct_res = cc.EvalMult(x, pt_mask)
    ct_res = cc.EvalRotate(ct_res, idx)
    return ct_res


def _duplicate_block(x, duplicate_count, block_size, pt_mask=None):
    cc = x.GetCryptoContext()
    rotation = block_size
    ct_res = x

    while rotation < block_size * duplicate_count:
        ct_rotated = cc.EvalRotate(ct_res, -rotation)
        ct_res = cc.EvalAdd(ct_res, ct_rotated)
        rotation *= 2
    if pt_mask is not None:
        ct_res = cc.EvalMult(pt_mask, ct_res)
    return ct_res
