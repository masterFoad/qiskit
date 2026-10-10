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

import numpy as np

from qiskit.circuit import Qubit
from qiskit.circuit.library import UnitaryGate
from qiskit.dagcircuit import DAGCircuit
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

    A SWAP is left alone when one of its qubits would end on the merged block's single-qubit
    gates: when the qubit does no further multi-qubit work after it, or when, given a ``target``
    that reports the qubit's T2 below ``_TAIL_T2_THRESHOLD``, the qubit is not measured after it
    and all its remaining multi-qubit operations act on the same set of qubits.  An
    as-late-as-possible schedule parks such a trailing single-qubit run at the end of the circuit,
    and the qubit dephases while it waits.  The preset pipeline runs this pass at optimization
    levels 2 and 3, and gives it the Target, which turns on the T2 part, only at level 3, whose
    optimization loop re-synthesizes the merged blocks' neighbours.
    """

    def __init__(self, target=None):
        super().__init__()
        self.target = target

    def run(self, dag):
        if "swap" not in dag.count_ops(recurse=False):
            return dag
        order, units, moves = _find_moves(dag, _short_t2_qubits(self.target, dag.num_qubits()))
        return _apply_moves(dag, order, units, moves) if moves else dag


def _short_t2_qubits(target, num_qubits):
    """Indices of the qubits whose ``target`` T2 is known and below ``_TAIL_T2_THRESHOLD``."""
    properties = getattr(target, "qubit_properties", None) or ()
    return frozenset(
        index
        for index, prop in enumerate(properties[:num_qubits])
        if getattr(prop, "t2", None) is not None and prop.t2 < _TAIL_T2_THRESHOLD
    )


def _find_moves(dag, short_t2=frozenset()):
    """Return the operations in topological order, their units and the ``(unit, swap, side)`` moves.

    Operations and qubits are referred to by index; ``short_t2`` holds the qubits the T2 part of
    the tail rule applies to.
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
    # Position on each wire of its last measurement, for the qubits the tail rule's T2 part covers.
    last_measure = {
        qubit: max(
            (i for i, node in enumerate(wires[qubit]) if order[node].name == "measure"), default=-1
        )
        for qubit in short_t2
        if qubit < len(wires)
    }

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

    def ends_on_one_pair(qubit, index, unit):
        """Whether the tail rule's T2 part leaves the SWAP at ``index`` on ``qubit`` alone.

        That is when ``qubit`` has a short T2, is not measured after the SWAP, and its multi-qubit
        units after the SWAP other than ``unit`` all act on one set of qubits.
        """
        if qubit not in short_t2 or last_measure[qubit] > index:
            return False
        seen = None
        for node in wires[qubit][index + 1 :]:
            if (
                len(qargs[node]) < 2
                or unit_of[node] == unit
                or getattr(order[node].op, "_directive", False)
            ):
                continue
            if seen is None:
                seen = units[unit_of[node]][1]
            elif units[unit_of[node]][1] != seen:
                return False
        return True

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
        # A qubit that does no further multi-qubit work after the SWAP would end on the merged
        # block's trailing single-qubit gates, which an as-late-as-possible schedule leaves waiting
        # until the end of the circuit; leave the SWAP decomposition there.
        node = units[swap][0][0]
        if any(last_multi[q] <= i for q, i in zip(qargs[node], position[node])):
            continue
        # Look for the nearest unit of the pair on either side of the SWAP.
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
                and unit not in taken
                and all(commutes_between(q, unit, swap) for q in pair)
            ):
                # A SWAP the tail rule leaves alone is not tried on its other side either.
                if not any(
                    ends_on_one_pair(q, i, unit) for q, i in zip(qargs[node], position[node])
                ):
                    taken.add(unit)
                    moves.append((unit, swap, step))
                break
    return order, units, moves


def _apply_moves(dag, order, units, moves):
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


def _wire_paulis(node):
    """The Pauli ``node`` commutes with on each of its qubits, as a string ("-" for none)."""
    if node.is_standard_gate():
        return _WIRE_PAULIS.get(node.name, "-" * len(node.qargs))
    if node.name == "unitary" and len(node.qargs) == 2 and isinstance(node.op, UnitaryGate):
        matrix = node.op.to_matrix()
        if not np.count_nonzero(matrix - np.diag(np.diagonal(matrix))):
            return "ZZ"
    return "-" * len(node.qargs)
