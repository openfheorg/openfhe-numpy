from operator import index as operator_index

from openfhe import FIXEDMANUAL

from .packing import _is_col_major, _is_row_major


def _get_slot_index(row, col, shape, order) -> int:
    """Return the slot index for ``(row, col)`` within one packed frame."""
    row = operator_index(row)
    col = operator_index(col)
    num_rows, num_cols = shape

    if not (0 <= row < num_rows and 0 <= col < num_cols):
        raise IndexError(f"matrix position ({row}, {col}) is outside packed shape {shape}.")
    if _is_row_major(order):
        return row * num_cols + col
    if _is_col_major(order):
        return col * num_rows + row
    raise ValueError(f"unsupported packing order {order!r}.")


def _get_packed_slot_index(coord, physical_shape, order) -> int:
    """Return the packed slot for a rank-0, rank-1, or rank-2 coordinate.

    Rank 0 (``()``) is slot 0; rank 1 (``(i,)``) is contiguous at ``i``; rank 2
    ``(row, col)`` uses the order-aware matrix formula ``_get_slot_index``.
    """
    if not coord:
        return 0
    if len(coord) == 1:
        return coord[0]
    if len(coord) == 2:
        return _get_slot_index(coord[0], coord[1], physical_shape, order)
    raise ValueError(f"packed coordinates support rank 0, 1, or 2; got rank {len(coord)}")


def _get_cell_index(slot, shape, order) -> tuple[int, int]:
    """Return ``(row, col)`` for a slot within one packed frame."""
    slot = operator_index(slot)
    num_rows, num_cols = shape

    if not 0 <= slot < num_rows * num_cols:
        raise IndexError(f"slot index {slot} is outside packed shape {shape}.")
    if _is_row_major(order):
        return divmod(slot, num_cols)
    if _is_col_major(order):
        col, row = divmod(slot, num_rows)
        return row, col
    raise ValueError(f"unsupported packing order {order!r}.")


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


def _get_elements_at_slots(ciphertext, source_slots, batch_size, rotation, mask_cache=None):
    """Keep selected ciphertext slots and optionally rotate them into position."""
    cc = ciphertext.GetCryptoContext()
    source_slots = tuple(sorted(set(source_slots)))
    if any(slot < 0 or slot >= batch_size for slot in source_slots):
        raise IndexError("slot mask contains an index outside the ciphertext batch")

    level = ciphertext.GetLevel()
    cache_key = (id(cc), source_slots, batch_size, level)
    plaintext = None if mask_cache is None else mask_cache.get(cache_key)

    if plaintext is None:
        mask = _create_masking(source_slots, batch_size)
        plaintext = cc.MakeCKKSPackedPlaintext(mask, 1, level, None, batch_size)
        if mask_cache is not None:
            mask_cache[cache_key] = plaintext

    result = cc.EvalMult(ciphertext, plaintext)
    if cc.GetScalingTechnique() == FIXEDMANUAL:
        cc.ModReduceInPlace(result)

    return result if rotation == 0 else cc.EvalRotate(result, rotation)


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
