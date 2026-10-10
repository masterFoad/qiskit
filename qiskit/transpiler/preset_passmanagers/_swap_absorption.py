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

"""Move two-qubit interactions next to SWAPs present after routing on the same pair."""

from __future__ import annotations

from qiskit._accelerate.swap_absorption import absorb_into_swaps, swap_absorption_moves
from qiskit.transpiler.basepasses import TransformationPass

# The tail rule leaves a SWAP alone on a qubit with a T2 below this many seconds (see
# ``_short_t2_qubits``).
_TAIL_T2_THRESHOLD = 10e-6


class _AbsorbIntoSwaps(TransformationPass):
    """Place a two-qubit interaction next to a ``swap`` present after routing on the same pair.

    The interaction is moved, exactly, across the operations in between, which the two-qubit
    peephole optimization then lets synthesize ``SWAP * U`` as one block (two CX for ``U = CX``).
    This is motivated by optimization-aware SWAP costs in NASSC (Liu, Li and Zhou,
    arXiv:2205.10596) and dressed-SWAP synthesis in 2QAN (Lao and Browne, arXiv:2108.02099),
    applied after routing. A unit is a
    standard gate, a diagonal ``UnitaryGate`` or ``cx(a, b); D(b); cx(a, b)`` with ``D`` diagonal;
    it only passes units that commute with the same Pauli on each shared qubit, never a SWAP.
    For a SWAP no unit reaches that way, the single-qubit gates right next to it on a unit's side
    may first cross it onto its other qubit, as ``SWAP (g x I) = (I x g) SWAP``.
    An existing CZ block can move as a whole, preserving its internal order, when every member
    commutes with every external operation it crosses.

    A SWAP is left alone when one of its qubits would end on the merged block's single-qubit
    gates: when the qubit does no further multi-qubit work after it, or when, given a ``target``
    that reports the qubit's T2 below ``_TAIL_T2_THRESHOLD``, the qubit is not measured after it
    and all its remaining multi-qubit operations act on the same set of qubits.  An
    as-late-as-possible schedule parks such a trailing single-qubit run at the end of the circuit,
    and the qubit dephases while it waits.  The preset pipeline runs this pass at optimization
    levels 2 and 3, and gives it the Target, which turns on the T2 part, only at level 3, whose
    optimization loop re-synthesizes the merged blocks' neighbours.

    The search for the moves and the rewrite are implemented in Rust
    (``qiskit._accelerate.swap_absorption``).
    """

    def __init__(self, target=None):
        super().__init__()
        self.target = target

    def run(self, dag):
        """Run the pass on ``dag``.

        Args:
            dag (DAGCircuit): the DAG to rewrite.

        Returns:
            DAGCircuit: the rewritten DAG.
        """
        absorb_into_swaps(dag, sorted(_short_t2_qubits(self.target, dag.num_qubits())))
        return dag


def _short_t2_qubits(target, num_qubits):
    """Indices of the qubits whose ``target`` T2 is known and below ``_TAIL_T2_THRESHOLD``."""
    properties = getattr(target, "qubit_properties", None) or ()
    return frozenset(
        index
        for index, prop in enumerate(properties[:num_qubits])
        if getattr(prop, "t2", None) is not None and prop.t2 < _TAIL_T2_THRESHOLD
    )


def _find_moves(dag, short_t2=frozenset()):
    """Return the operations, the units and the moves the pass would make on ``dag``.

    This is the plan of the Rust implementation, exposed for testing. ``order`` is
    ``dag.topological_op_nodes()``; units are ``(members, qubits, paulis)`` with the members as
    indices into ``order``, the qubits as a ``frozenset`` of qubit indices and the Pauli each
    qubit commutes with as a dict (``"-"`` for none); moves are
    ``(unit, swap, step, hopped_nodes)``.
    """
    order = list(dag.topological_op_nodes())
    units, moves = swap_absorption_moves(dag, sorted(short_t2))
    units = [
        (members, frozenset(qubits), dict(zip(qubits, paulis))) for members, qubits, paulis in units
    ]
    return order, units, moves
