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

"""Move two-qubit interactions next to routing SWAPs on the same qubit pair."""

from __future__ import annotations

import numpy as np

from qiskit.circuit import Qubit
from qiskit.circuit.library import UnitaryGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass

# For each standard gate, the single-qubit Pauli it commutes with on each of its qubits ("-" for
# none).  Two operations that, on every qubit they share, commute with the same Pauli commute
# with each other: both are block diagonal in that Pauli's eigenbasis on the shared qubits.
_WIRE_PAULIS = {
    **dict.fromkeys(("rz", "z", "s", "sdg", "t", "tdg", "p", "u1"), "Z"),
    **dict.fromkeys(("rx", "x", "sx", "sxdg"), "X"),
    **dict.fromkeys(("ry", "y"), "Y"),
    **dict.fromkeys(("cz", "cp", "crz", "cu1", "cs", "csdg", "rzz"), "ZZ"),
    **dict.fromkeys(("cx", "crx", "csx", "rzx"), "ZX"),
    "rxx": "XX",
    "ryy": "YY",
}
_CONTROLLED_PHASE_GATES = frozenset(("cp", "cu1", "cs", "csdg"))


class _AbsorbIntoSwaps(TransformationPass):
    """Move a two-qubit interaction next to a ``swap`` on the same qubit pair.

    The interaction is moved exactly, across operations it commutes with, so that the two-qubit
    peephole optimization can synthesize the ``swap`` and the interaction as a single block
    [1, 2], for example two CX gates for a ``swap`` followed by a ``cx``.  A unit that can be
    moved is a standard gate, a diagonal two-qubit :class:`.UnitaryGate`, or
    ``cx(a, b); D(b); cx(a, b)`` with ``D`` diagonal.  It is only moved past units that commute
    with the same Pauli on every shared qubit, and never past a ``swap``.

    A ``swap`` is left alone when one of its qubits does no further multi-qubit work after it,
    because that qubit would end on the merged block's single-qubit gates, which an
    as-late-as-possible schedule leaves idle until the end of the circuit.

    References:

    [1] Liu, Li and Zhou. Not All SWAPs Have the Same Cost: A Case for Optimization-Aware Qubit
    Routing. `arXiv:2205.10596 <https://arxiv.org/abs/2205.10596>`_

    [2] Lao and Browne. 2QAN: A quantum compiler for 2-local qubit Hamiltonian simulation
    algorithms. `arXiv:2108.02099 <https://arxiv.org/abs/2108.02099>`_
    """

    def __init__(self, absorb_controlled_phase: bool = True):
        """
        Args:
            absorb_controlled_phase: Whether controlled-phase units (``cp``, ``cu1``, ``cs`` and
                ``csdg``) are moved.
        """
        super().__init__()
        self.absorb_controlled_phase = absorb_controlled_phase

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        if "swap" not in dag.count_ops(recurse=False):
            return dag
        order, units, moves = _find_moves(dag, self.absorb_controlled_phase)
        return _apply_moves(dag, order, units, moves) if moves else dag


def _find_moves(dag: DAGCircuit, absorb_controlled_phase: bool = True):
    """Find the units to move next to a ``swap``.

    Args:
        dag: The circuit to search.
        absorb_controlled_phase: Whether controlled-phase units are moved.

    Returns:
        tuple: The operations in topological order, the units, and the moves as
        ``(unit, swap, step)`` triples, where ``step`` is ``1`` if the unit is placed after the
        ``swap`` and ``-1`` if before.  Operations, units and qubits are referred to by index.
    """
    order = list(dag.topological_op_nodes())
    qubit_index = {qubit: index for index, qubit in enumerate(dag.qubits)}
    qargs = [tuple(qubit_index[qubit] for qubit in node.qargs) for node in order]
    wires = [[] for _ in qubit_index]
    position = []  # ``position[node][i]``: index of ``node`` on the wire of its ``i``-th qubit.
    for node, qubits in enumerate(qargs):
        position.append(tuple(len(wires[qubit]) for qubit in qubits))
        for qubit in qubits:
            wires[qubit].append(node)
    # Position on each wire of its last multi-qubit operation.
    last_multi = [
        max((i for i, node in enumerate(nodes) if len(qargs[node]) > 1), default=-1)
        for nodes in wires
    ]

    def after(node, i):
        """The operation after ``node`` on the wire of its ``i``-th qubit."""
        nodes = wires[qargs[node][i]]
        return nodes[position[node][i] + 1] if position[node][i] + 1 < len(nodes) else None

    # Group the operations into units, each with the Pauli it commutes with on each of its qubits.
    unit_of = [None] * len(order)
    units = []
    for node, op_node in enumerate(order):
        if unit_of[node] is not None:
            continue
        members = [node]
        paulis = dict(zip(qargs[node], _wire_paulis(op_node)))
        if op_node.name == "cx" and op_node.is_standard_gate():
            middle = after(node, 1)
            if (
                middle is not None
                and _wire_paulis(order[middle]) == "Z"
                and (last := after(middle, 0)) is not None
                and order[last].name == "cx"
                and order[last].is_standard_gate()
                and qargs[last] == qargs[node]
                and after(node, 0) == last
            ):
                members = [node, middle, last]
                paulis = dict.fromkeys(qargs[node], "Z")
        for member in members:
            unit_of[member] = len(units)
        units.append((members, frozenset(qargs[node]), paulis))

    sequence = [[unit_of[node] for node in nodes] for nodes in wires]
    sequence = [[u for i, u in enumerate(seq) if i == 0 or u != seq[i - 1]] for seq in sequence]
    unit_position = {
        (qubit, unit): index for qubit, seq in enumerate(sequence) for index, unit in enumerate(seq)
    }

    def commutes_between(qubit, unit, swap):
        """Whether ``unit`` commutes with every unit between it and ``swap`` on ``qubit``."""
        pauli = units[unit][2][qubit]
        low, high = sorted((unit_position[(qubit, unit)], unit_position[(qubit, swap)]))
        return pauli != "-" and all(
            units[other][2][qubit] == pauli for other in sequence[qubit][low + 1 : high]
        )

    swaps = [
        unit
        for unit, (members, _, _) in enumerate(units)
        if order[members[0]].name == "swap" and order[members[0]].is_standard_gate()
    ]
    # A SWAP already adjacent (on both wires) to another unit of its pair is left alone, and so is
    # that unit; the peephole optimization merges them as they are.
    taken = set()
    pending = []
    for swap in swaps:
        pair = units[swap][1]
        touching = [
            unit
            for qubit in pair
            for index in (unit_position[(qubit, swap)] - 1, unit_position[(qubit, swap)] + 1)
            if 0 <= index < len(sequence[qubit])
            and units[unit := sequence[qubit][index]][1] == pair
            and all(abs(unit_position[(q, unit)] - unit_position[(q, swap)]) == 1 for q in pair)
        ]
        if touching:
            taken.update(touching)
        else:
            pending.append(swap)
    moves = []
    for swap in pending:
        pair = units[swap][1]
        # A qubit with no further multi-qubit work would end on the merged block's single-qubit
        # gates, which an as-late-as-possible schedule leaves idle until the end of the circuit.
        node = units[swap][0][0]
        if any(last_multi[q] <= i for q, i in zip(qargs[node], position[node])):
            continue
        qubit = qargs[node][0]
        seq = sequence[qubit]
        for step in (1, -1):
            index = unit_position[(qubit, swap)] + step
            # Walk to the nearest unit of the pair, stopping at a unit nothing can be moved past.
            while (
                0 <= index < len(seq)
                and units[seq[index]][1] != pair
                and units[seq[index]][2][qubit] != "-"
            ):
                index += step
            unit = seq[index] if 0 <= index < len(seq) else None
            if (
                unit is not None
                and units[unit][1] == pair
                and (
                    absorb_controlled_phase
                    or order[units[unit][0][0]].name not in _CONTROLLED_PHASE_GATES
                )
                and unit not in taken
                and all(commutes_between(q, unit, swap) for q in pair)
            ):
                taken.add(unit)
                moves.append((unit, swap, step))
                break
    return order, units, moves


def _apply_moves(dag: DAGCircuit, order, units, moves) -> DAGCircuit:
    """Move every unit in ``moves`` in place, right next to its SWAP."""
    for unit, _, _ in moves:
        for member in units[unit][0]:
            dag.remove_op_node(order[member])
    for unit, swap, step in moves:
        swap_node = order[units[swap][0][0]]
        block = DAGCircuit()
        block.add_qubits(qubits := [Qubit(), Qubit()])
        wire_map = dict(zip(swap_node.qargs, qubits))
        members = [order[member] for member in units[unit][0]]
        for node in [swap_node, *members] if step > 0 else [*members, swap_node]:
            block.apply_operation_back(node.op, [wire_map[q] for q in node.qargs], check=False)
        dag.substitute_node_with_dag(swap_node, block, wires=qubits)
    return dag


def _wire_paulis(node: DAGOpNode) -> str:
    """The Pauli ``node`` commutes with on each of its qubits, as a string ("-" for none)."""
    if node.is_standard_gate():
        return _WIRE_PAULIS.get(node.name, "-" * len(node.qargs))
    if node.name == "unitary" and len(node.qargs) == 2 and isinstance(node.op, UnitaryGate):
        matrix = node.op.to_matrix()
        if not np.count_nonzero(matrix - np.diag(np.diagonal(matrix))):
            return "ZZ"
    return "-" * len(node.qargs)
