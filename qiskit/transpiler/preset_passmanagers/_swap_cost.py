# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Two-qubit cost of a 4x4 block, as the optimization stage's synthesis counts it.

Used by :class:`._AbsorbIntoSwaps` to decide whether a move saves a two-qubit gate.  The cost reads
the Weyl coordinates with the synthesis' own default specialization, so very small interactions
(for example ``cp(1e-4)``) cost 0 as they do after resynthesis, not 2 as the exact count says.  A
coordinate that is neither clearly zero nor clearly non-zero makes the cost unknown (``None``); an
unknown cost never justifies a change.  The synthesis' ``num_basis_gates`` estimate is not used
(it reports 0 for ``h; swap``).  With a continuous ``rzz`` family native (``fractional``), a block
with one non-zero coordinate costs one gate.
"""

from __future__ import annotations

import math

import numpy as np

from qiskit.circuit import Gate
from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition

_QUARTER = math.pi / 4
# Coordinates below _ZERO are zero, above _NONZERO non-zero, in between unknown.
_ZERO = 1e-12
_NONZERO = 1e-9

_SWAP_MAT = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def _count(coords, fractional):
    a, b, c = coords
    if c < _ZERO:
        if b < _ZERO:
            if a < _ZERO:
                return 0
            if abs(a - _QUARTER) < _ZERO or fractional:
                return 1
        return 2
    return 3


def two_qubit_cost(matrix, fractional=False):
    """Two-qubit gate count of the optimization stage's synthesis of ``matrix``, or ``None``
    when it cannot be decided."""
    weyl = TwoQubitWeylDecomposition(matrix)
    coords = (abs(weyl.a), abs(weyl.b), abs(weyl.c), abs(abs(weyl.a) - _QUARTER))
    if any(_ZERO <= x < _NONZERO for x in coords):
        return None
    return _count(coords[:3], fractional)


def _matrix(node):
    """Numeric matrix of a parameter-free unitary gate, or ``None``."""
    if node.is_standard_gate():
        if node.cargs or node.is_parameterized():
            return None
        try:
            return np.asarray(node.matrix, dtype=complex)
        except Exception:  # pylint: disable=broad-except
            return None
    op = node.op
    if not isinstance(op, Gate) or node.cargs or node.is_parameterized():
        return None
    try:
        return np.asarray(op.to_matrix(), dtype=complex)
    except Exception:  # pylint: disable=broad-except
        return None
