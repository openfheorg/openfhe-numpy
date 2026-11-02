"""Shared helpers for the linear-transform tests.

Contexts use a small ring and HEStd_NotSet for speed only. Matrices are packed row by row and
tiled to fill the batch, as EncodeMatrix does.
"""
import contextlib

import numpy as np
import openfhe as fhe
import openfhe_numpy as onp

RING_DIM = 1024
ATOL = 1e-5


def make_context(batch_size, scaling=fhe.FLEXIBLEAUTO, depth=8):
    params = fhe.CCParamsCKKSRNS()
    params.SetSecurityLevel(fhe.HEStd_NotSet)
    params.SetRingDim(RING_DIM)
    params.SetBatchSize(batch_size)
    params.SetMultiplicativeDepth(depth)
    params.SetScalingModSize(40)
    params.SetFirstModSize(50)
    params.SetScalingTechnique(scaling)
    params.SetKeySwitchTechnique(fhe.HYBRID)
    cc = fhe.GenCryptoContext(params)
    for feature in (
        fhe.PKESchemeFeature.PKE,
        fhe.PKESchemeFeature.KEYSWITCH,
        fhe.PKESchemeFeature.LEVELEDSHE,
    ):
        cc.Enable(feature)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    return cc, keys


def tile(matrix, slots):
    flat = np.asarray(matrix, dtype=float).reshape(-1)
    return np.tile(flat, slots // flat.size)


def encode(cc, matrix, slots):
    return cc.MakeCKKSPackedPlaintext(tile(matrix, slots).tolist(), 1, 0, None, slots)


def encrypt(cc, keys, matrix, slots):
    return cc.Encrypt(keys.publicKey, encode(cc, matrix, slots))


def decrypt(cc, keys, ciphertext, length):
    pt = cc.Decrypt(keys.secretKey, ciphertext)
    pt.SetLength(length)
    return np.array(pt.GetRealPackedValue())


def depth_used(before, after):
    """Multiplicative levels consumed, counting a pending rescale (noise scale degree 2) as one."""
    return (after.GetLevel() + after.GetNoiseScaleDeg()) - (before.GetLevel() + before.GetNoiseScaleDeg())


# Plain references for the JKLS permutations of a d x d matrix A.
def sigma_ref(A):
    d = len(A)
    return np.array([[A[i][(i + j) % d] for j in range(d)] for i in range(d)])


def tau_ref(A):
    d = len(A)
    return np.array([[A[(i + j) % d][j] for j in range(d)] for i in range(d)])


def phi_ref(A, k):
    d = len(A)
    return np.array([[A[i][(j + k) % d] for j in range(d)] for i in range(d)])


def psi_ref(A, k):
    d = len(A)
    return np.array([[A[(i + k) % d][j] for j in range(d)] for i in range(d)])


def lt_ref(x, offsets, diagonals):
    """sum_t diagonals[t] * Rot(offsets[t])(x), with Rot_r(x)[i] = x[(i + r) mod len(x)]."""
    y = np.zeros(len(x))
    for offset, diag in zip(offsets, diagonals):
        y += np.asarray(diag) * np.roll(x, -offset)
    return y


@contextlib.contextmanager
def lintrans_algo(algo):
    previous = onp.GetLinTransAlgo()
    onp.SetLinTransAlgo(algo)
    try:
        yield
    finally:
        onp.SetLinTransAlgo(previous)
