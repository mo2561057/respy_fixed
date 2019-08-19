"""
This file contains configuration for this repo.
"""
"""General configuration for respy."""
from pathlib import Path

# Obtain the root directory of the package. Do not import respy which creates a circular
# import.
ROOT_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = ROOT_DIR / "ressources"

ORIGINAL_MOMENTS_DIR = ROOT_DIR / "estimation/resources"

#Hard coded large integer
HUGE_INT = 1000000000
