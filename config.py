import openfhe
from enum import Enum


# Type Aliases for OpenFHE Types
Plaintext = openfhe.Plaintext
Ciphertext = openfhe.Ciphertext
CryptoContext = openfhe.CryptoContext
KeyPair = openfhe.KeyPair


# Encoding strategy for matrix packing
class MatrixOrder:
    ROW_MAJOR = "R"  # Encode data row-wise (default)
    COL_MAJOR = "C"  # Encode data column-wise
    DIAG_MAJOR = "D"  # Optional: encode data diagonally (future use)


# Types of data representation
class DataType:
    PLAINTEXT = "P"
    CIPHERTEXT = "C"


# Numerical constants
EPSILON = 1e-8
EPSILON_HIGH = 1e-4


# default format result
class FormatType(Enum):
    RAW = "raw"
    RESHAPE = "reshape"
    ROUND = "round"
    AUTO = "auto"
