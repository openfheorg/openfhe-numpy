from operator import index as operator_index


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


def _replication_steps(copies: int, stride: int):
    """Yield ``(rotation, use_seed)`` for exactly ``copies`` patterns."""
    copies = operator_index(copies)
    stride = operator_index(stride)

    if copies < 1:
        raise ValueError(f"copies must be positive, got {copies!r}")
    if stride < 1:
        raise ValueError(f"stride must be positive, got {stride!r}")

    produced = 1
    for bit in bin(copies)[3:]:
        yield -(produced * stride), False
        produced *= 2

        if bit == "1":
            yield -(produced * stride), True
            produced += 1


def _replicate_pattern(seed, copies: int, stride: int):
    """Replicate an isolated slot pattern exactly ``copies`` times."""
    cc = seed.GetCryptoContext()
    result = seed

    for rotation, use_seed in _replication_steps(copies, stride):
        operand = seed if use_seed else result
        result = cc.EvalAdd(result, cc.EvalRotate(operand, rotation))

    return result
