"""Provides routines for comparing GPNH convex coding method."""

# License: MIT

from .archetypal_analysis import ArchetypalAnalysis, KernelAA
from .furthest_sum import furthest_sum
from .gpnh_convex_coding import GPNHConvexCoding
from .kmeans import gap_statistic
from .simplex_projection import (simplex_project_rows, simplex_project_columns)
from .spg import spg
from .stochastic_matrices import left_stochastic_matrix, right_stochastic_matrix
