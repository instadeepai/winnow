from jaxtyping import Float, Integer
import numpy as np


Peptide = Integer[np.ndarray, "token"]
Spectrum = Float[np.ndarray, "peak 2"]
