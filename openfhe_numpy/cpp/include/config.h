#ifndef OPENFHE_NUMPY_CONFIG_H
#define OPENFHE_NUMPY_CONFIG_H

#include "openfhe.h"


#define OPENFHE_NUMPY_VERSION "0.0.1"
#define OPENFHE_NUMPY_VERSION_MAJOR 0
#define OPENFHE_NUMPY_VERSION_MINOR 0
#define OPENFHE_NUMPY_VERSION_PATCH 1

namespace openfhe_numpy {

inline const std::string METADATA_ARRAYINFO_TAG = "arrayInfo";

enum class MatVecEncoding : std::uint8_t {
    MM_CRC  = 0,
    MM_RCR  = 1,
    MM_DIAG = 2
};

enum class LinTransType : std::uint8_t { SIGMA = 0, TAU = 1, PHI = 2, PSI = 3, TRANSPOSE = 4 };

enum class ArrayEncodingType : std::uint8_t { ROW_MAJOR = 0, COL_MAJOR = 1, DIAG_MAJOR = 2 };

template <typename Element>
using MatKeys = std::shared_ptr<std::map<uint32_t, lbcrypto::EvalKey<Element>>>;

}  // namespace openfhe_numpy
#endif  // OPENFHE_NUMPY_CONFIG_H
