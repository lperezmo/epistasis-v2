"""Type stubs for the Rust extension module `epistasis._core`."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

def version() -> str: ...
def encode_vectors(
    binary_packed: NDArray[np.uint8],
    model_type: Literal["global", "local"] = "global",
) -> NDArray[np.int8]: ...
def build_model_matrix(
    encoded: NDArray[np.int8],
    sites_flat: NDArray[np.int64],
    sites_offsets: NDArray[np.int64],
) -> NDArray[np.int8]: ...
def fwht(data: NDArray[np.float64]) -> NDArray[np.float64]: ...
