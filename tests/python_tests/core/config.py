from pathlib import Path
import logging

# ===============================
# Paths and Directories
# ===============================
TESTS_DIR = Path(__file__).parent.parent.resolve()
PROJECT_ROOT = TESTS_DIR.parent

LOG_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "log.txt"

ERROR_DIR = PROJECT_ROOT / "errors"
ERROR_PATH = ERROR_DIR / "error.txt"

LOGGER_NAME = "openfhe_tests"

FORMATTER = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
EPSILON = 1e-8
