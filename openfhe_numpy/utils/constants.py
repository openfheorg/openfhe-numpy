from enum import Enum

# Numerical constants
EPSILON = 1e-8
EPSILON_HIGH = 1e-4


# Types of data representation
class DataType(str, Enum):
    PLAINTEXT = "P"
    CIPHERTEXT = "C"

    def __str__(self) -> str:
        return self.value


# Default format result
class UnpackType(Enum):
    RAW = "raw"
    ORIGINAL = "original"
    RESHAPE = "reshape"

    def __str__(self) -> str:
        return self.value
