# ==================================================================================
#  BSD 2-Clause License
#
#  Copyright (c) 2014-2025, NJIT, Duality Technologies Inc. and other contributors
#
#  All rights reserved.
#
#  Author TPOC: contact@openfhe.org
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==================================================================================

"""
Crypto context and parameter loading for OpenFHE-NumPy tests.
Suppress_stdout has been moved to utils.py.
"""

import os
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openfhe import (
    CCParamsCKKSRNS,
    GenCryptoContext,
    PKESchemeFeature,
    UNIFORM_TERNARY,
    FIXEDAUTO,
    FLEXIBLEAUTOEXT,
    FLEXIBLEAUTO,
    FIXEDMANUAL,
    HYBRID,
    BV,
    HEStd_128_classic,
    HEStd_192_classic,
    HEStd_256_classic,
    HEStd_NotSet,
)

from .test_utils import suppress_stdout

# ===============================
# Paths and CSV
# ===============================
MODULE_DIR: Path = Path(__file__).parent.resolve()
TEST_DIR: Path = Path(__file__).parent.parent.resolve()
PARAMS_CSV: Path = TEST_DIR / "crypto_params" / "ckks_params_auto.csv"

# print("PARAMS_CSV = ", PARAMS_CSV)
# ===============================
# CSV Field Converters
# ===============================
_CONVERTERS: Dict[str, Any] = {
    "ringDim": int,
    "multiplicativeDepth": int,
    "scalingModSize": int,
    "batchSize": int,
    "firstModSize": int,
    "numLargeDigits": int,
    "maxRelinSkDeg": int,
    "digitSize": int,
    "standardDeviation": float,
}

# ===============================
# Caches and Mappings
# ===============================
CRYPTO_CONTEXT_CACHE: Dict[Tuple[Tuple[str, Any], ...], Tuple[Any, Any]] = {}

SECURITY_LEVEL_MAP: Dict[str, Any] = {
    "HEStd_128_classic": HEStd_128_classic,
    "HEStd_192_classic": HEStd_192_classic,
    "HEStd_256_classic": HEStd_256_classic,
    "HEStd_NotSet": HEStd_NotSet,
}

SECRET_KEY_DIST_MAP: Dict[str, Any] = {
    "UNIFORM_TERNARY": UNIFORM_TERNARY,
}

SCALING_TECHNIQUE_MAP: Dict[str, Any] = {
    "FIXEDAUTO": FIXEDAUTO,
    "FLEXIBLEAUTOEXT": FLEXIBLEAUTOEXT,
    "FLEXIBLEAUTO": FLEXIBLEAUTO,
    "FIXEDMANUAL": FIXEDMANUAL,
}

KEY_SWITCH_TECHNIQUE_MAP: Dict[str, Any] = {
    "HYBRID": HYBRID,
    "BV": BV,
}

# ===============================
# Parameter Loading
# ===============================


def load_ckks_params() -> List[Dict[str, Any]]:
    """Load and parse CKKS parameter sets from CSV."""
    if not PARAMS_CSV.exists():
        raise FileNotFoundError(f"Missing CSV file: {PARAMS_CSV}")
    params_list: List[Dict[str, Any]] = []
    with PARAMS_CSV.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row_num, row in enumerate(reader, start=1):
            entry: Dict[str, Any] = {}
            for key, val in row.items():
                converter = _CONVERTERS.get(key, lambda x: x)
                try:
                    entry[key] = converter(val)
                except ValueError as e:
                    raise ValueError(f"Error parsing '{key}' at row {row_num}: {e}")
            params_list.append(entry)
    return params_list


# ===============================
# Crypto Context Functions
# ===============================


def gen_crypto_context(params: Dict[str, Any]) -> Tuple[Any, Any]:
    """Generate a new CryptoContext and key pair from parameters."""
    p = CCParamsCKKSRNS()
    p.SetRingDim(params["ringDim"])
    p.SetMultiplicativeDepth(params["multiplicativeDepth"])
    p.SetScalingModSize(params["scalingModSize"])
    p.SetBatchSize(params["batchSize"])
    p.SetFirstModSize(params["firstModSize"])
    p.SetStandardDeviation(params["standardDeviation"])
    p.SetSecretKeyDist(SECRET_KEY_DIST_MAP[params["secretKeyDist"]])
    p.SetScalingTechnique(SCALING_TECHNIQUE_MAP[params["scalTech"]])
    p.SetKeySwitchTechnique(KEY_SWITCH_TECHNIQUE_MAP[params["ksTech"]])
    p.SetSecurityLevel(SECURITY_LEVEL_MAP[params["securityLevel"]])
    p.SetNumLargeDigits(params["numLargeDigits"])
    p.SetMaxRelinSkDeg(params["maxRelinSkDeg"])
    p.SetDigitSize(params["digitSize"])

    cc = GenCryptoContext(p)
    for feat in (
        PKESchemeFeature.PKE,
        PKESchemeFeature.LEVELEDSHE,
        PKESchemeFeature.ADVANCEDSHE,
    ):
        cc.Enable(feat)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)
    return cc, keys


def get_cached_crypto_context(params: Dict[str, Any], use_cache: bool = True) -> Tuple[Any, Any]:
    """Get a cached CryptoContext or generate a new one."""
    if not use_cache:
        return gen_crypto_context(params)
    key = tuple(sorted(params.items()))
    if key not in CRYPTO_CONTEXT_CACHE:
        with suppress_stdout():
            CRYPTO_CONTEXT_CACHE[key] = gen_crypto_context(params)
    return CRYPTO_CONTEXT_CACHE[key]
