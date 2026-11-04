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

"""Move two-qubit gates next to routing SWAPs on the same pair of qubits."""

from __future__ import annotations

import numpy as np

from qiskit.circuit import Qubit
from qiskit.circuit.commutation_library import SessionCommutationChecker
from qiskit.circuit.library import (
    CXGate,
    CZGate,
    PhaseGate,
    SwapGate,
    UnitaryGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass

# The gates are listed explicitly, so that new commutation table entries do not enable new moves.
_TWO_QUBIT_GATES = frozenset("cz cp crz cu1 cs csdg rzz cx crx csx rzx rxx ryy".split())
_SUPPORTED_GATES = frozenset("rz z s sdg t tdg p u1 rx x sx sxdg ry y".split()) | _TWO_QUBIT_GATES
_PAULI_PROBES = (("X", XGate()), ("Y", YGate()), ("Z", ZGate()))
# Gates used in place of a parameterized gate for the commutation table lookup only; the circuit
# is not changed.  RX, RY and RZ are functions of their Pauli, P and U1 of Z, and CP and CU1 of CZ.
_LOOKUP_REPRESENTATIVES = {
    "rx": XGate(),
    "ry": YGate(),
    "rz": ZGate(),
    "p": ZGate(),
    "u1": ZGate(),
    "cp": CZGate(),
    "cu1": CZGate(),
}
# Two-qubit units use a non-negative key encoding their unordered qubit pair.
_SINGLE_QUBIT_UNIT = -2
_OTHER_UNIT = -1
_SWAP_MATRIX = SwapGate().to_matrix()


class _AbsorbIntoSwaps(TransformationPass):
    """Move a two-qubit gate next to a ``swap`` on the same pair of qubits.

    The gate is moved across operations it commutes with, so that
    :class:`.TwoQubitPeepholeOptimization` can synthesize the ``swap`` and the gate as one block
    [1, 2], for example two CX gates for a ``swap`` followed by a ``cx``.  What can be moved is a
    supported standard two-qubit gate such as ``cx``, ``cz``, ``cp``, ``rzz`` or ``rxx``, a
    diagonal two-qubit :class:`.UnitaryGate`, or ``cx(a, b); D(b); cx(a, b)`` with ``D`` a
    diagonal single-qubit standard gate.  It is only moved past operations that commute with the
    same Pauli on every shared qubit, and never past a ``swap``.

    A ``swap`` with a ``cx`` is written as two CX gates, and a ``swap`` with a gate that is
    diagonal and not parameterized as three CX gates and at most three phase gates.  Both forms are
    exact, including the global phase.  Other pairs are left next to each other for
    :class:`.TwoQubitPeepholeOptimization`, whose general decomposition can need more
    single-qubit gates.

    A ``swap`` is left alone when no multi-qubit operation follows it on one of its qubits.  That
    qubit would then end on the single-qubit gates of the merged block, which an
    as-late-as-possible schedule leaves idle until the end of the circuit.  A directive on two or
    more qubits, such as a ``barrier``, counts as a multi-qubit operation here; single-qubit
    operations, including measurements, do not.

    With ``move_diagonal=False``, diagonal gates and ``cx(a, b); D(b); cx(a, b)`` are not moved.
    The preset pass managers do this for targets with ``rzz``, which runs such a gate as one
    two-qubit gate, so that merging it with a ``swap`` costs more than it saves.

    References:

    [1] Liu, Li and Zhou. Not All SWAPs Have the Same Cost: A Case for Optimization-Aware Qubit
    Routing. `arXiv:2205.10596 <https://arxiv.org/abs/2205.10596>`_

    [2] Lao and Browne. 2QAN: A quantum compiler for 2-local qubit Hamiltonian simulation
    algorithms. `arXiv:2108.02099 <https://arxiv.org/abs/2108.02099>`_
    """

    def __init__(self, move_diagonal: bool = True):
        super().__init__()
        self.move_diagonal = move_diagonal

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        ops = dag.count_ops(recurse=False)
        if "swap" not in ops:
            return dag
        # Only a supported two-qubit gate or a diagonal unitary can be moved.
        if _TWO_QUBIT_GATES.isdisjoint(ops) and all(
            _wire_paulis(node) != "ZZ" for node in dag.named_nodes("unitary")
        ):
            return dag
        order, members, moves = _find_moves(dag)
        if not self.move_diagonal:
            moves = [
                move
                for move in moves
                if len(members[move[0]]) == 1 and _wire_paulis(order[move[0]]) != "ZZ"
            ]
        for unit, _, _ in moves:
            for member in members[unit]:
                dag.remove_op_node(order[member])
        for unit, swap, step in moves:
            swap_node = order[swap]
            block = DAGCircuit()
            block.add_qubits(qubits := [Qubit(), Qubit()])
            wire_map = dict(zip(swap_node.qargs, qubits))
            nodes = [order[member] for member in members[unit]]
            if (fused := _fused_block(swap_node, nodes, step)) is None:
                for node in [swap_node, *nodes] if step > 0 else [*nodes, swap_node]:
                    block.apply_operation_back(
                        node.op, [wire_map[q] for q in node.qargs], check=False
                    )
            else:
                for op, indices in fused[0]:
                    block.apply_operation_back(op, [qubits[i] for i in indices], check=False)
                block.global_phase = fused[1]
            dag.substitute_node_with_dag(swap_node, block, wires=qubits)
        return dag


def _find_moves(dag: DAGCircuit):
    """Return the nodes in topological order, the members of each moved unit, and the moves.

    A unit is one node or a ``cx(a, b); D(b); cx(a, b)`` group, named by the topological index of
    its first node.  ``members[unit]`` lists the indices of its nodes.  Each move is
    ``(unit, swap, step)``: a positive step moves a later unit just after the SWAP, a negative
    step moves an earlier unit just before it.  All decisions use the original DAG.
    """
    order = list(dag.topological_op_nodes())
    qubit_index = {qubit: index for index, qubit in enumerate(dag.qubits)}
    num_wires = len(qubit_index)
    # For each wire: its units in order, the Pauli each unit commutes with on the wire ("-" for
    # none), and for each position the start of the run of equal Paulis that contains it.
    sequences = [[] for _ in range(num_wires)]
    labels = [[] for _ in range(num_wires)]
    run_starts = [[] for _ in range(num_wires)]
    last_multi = [-1] * num_wires  # wire position of the last multi-qubit unit
    # Per unit: its pair key, and for two-qubit units the qubits and wire positions in qarg order.
    pair_keys = [_OTHER_UNIT] * len(order)
    first_qubit = [-1] * len(order)
    second_qubit = [-1] * len(order)
    first_position = [-1] * len(order)
    second_position = [-1] * len(order)
    groups = {}  # members of the cx(a, b); D(b); cx(a, b) units
    swaps = []
    cache = {}

    def add_to_wire(qubit, unit, pauli):
        """Add ``unit`` to the end of a wire and return its position there."""
        wire_labels = labels[qubit]
        starts = run_starts[qubit]
        position = len(starts)
        starts.append(starts[-1] if position and wire_labels[-1] == pauli else position)
        wire_labels.append(pauli)
        sequences[qubit].append(unit)
        return position

    for node, op_node in enumerate(order):
        qargs = op_node.qargs
        label = _wire_paulis(op_node, cache)
        if len(qargs) == 1:
            pair_keys[node] = _SINGLE_QUBIT_UNIT
            add_to_wire(qubit_index[qargs[0]], node, label)
            continue
        if len(qargs) != 2:
            for qubit in qargs:
                qubit = qubit_index[qubit]
                last_multi[qubit] = add_to_wire(qubit, node, "-")
            continue
        qubit_a, qubit_b = qubit_index[qargs[0]], qubit_index[qargs[1]]
        target = sequences[qubit_b]
        # Group ``cx(a, b); D(b); cx(a, b)`` when the first CX and D are still units of their
        # own and nothing else is between them on either wire.
        if (
            op_node.name == "cx"
            and op_node.is_standard_gate()
            and len(target) > 1
            and labels[qubit_b][-1] == "Z"
            and pair_keys[middle := target[-1]] == _SINGLE_QUBIT_UNIT
            and first_qubit[head := target[-2]] == qubit_a
            and sequences[qubit_a][-1] == head
            and head not in groups
            and order[head].name == "cx"
            and order[head].is_standard_gate()
            and labels[qubit_a][-1] + labels[qubit_b][-2] == "ZX"
        ):
            # The group replaces the first CX and D, and commutes with Z on the target too.
            groups[head] = [head, middle, node]
            wire_labels, starts = labels[qubit_b], run_starts[qubit_b]
            for wire in (target, wire_labels, starts):
                wire.pop()
            wire_labels[-1] = "Z"
            starts[-1] = (
                starts[-2] if len(starts) > 1 and wire_labels[-2] == "Z" else len(starts) - 1
            )
            continue
        pair_keys[node] = (
            qubit_a * num_wires + qubit_b if qubit_a < qubit_b else qubit_b * num_wires + qubit_a
        )
        first_qubit[node], second_qubit[node] = qubit_a, qubit_b
        first_position[node] = last_multi[qubit_a] = add_to_wire(qubit_a, node, label[0])
        second_position[node] = last_multi[qubit_b] = add_to_wire(qubit_b, node, label[1])
        if op_node.name == "swap" and op_node.is_standard_gate():
            swaps.append(node)

    def position_on_wire(qubit, unit):
        return first_position[unit] if first_qubit[unit] == qubit else second_position[unit]

    def commutes_between(qubit, unit, swap):
        """Check the open interval between a candidate and its SWAP on one wire."""
        unit_position = position_on_wire(qubit, unit)
        swap_position = position_on_wire(qubit, swap)
        low, high = sorted((unit_position, swap_position))
        wire_labels = labels[qubit]
        pauli = wire_labels[unit_position]
        # Every label in between must be in one run of the candidate's Pauli.
        return pauli != "-" and (
            high - low <= 1
            or (wire_labels[high - 1] == pauli and run_starts[qubit][high - 1] <= low + 1)
        )

    def next_multi(qubit, unit, step):
        """The nearest unit past ``unit`` on ``qubit`` that is not single-qubit, if any."""
        sequence = sequences[qubit]
        index = position_on_wire(qubit, unit) + step
        while 0 <= index < len(sequence) and pair_keys[sequence[index]] == _SINGLE_QUBIT_UNIT:
            index += step
        return sequence[index] if 0 <= index < len(sequence) else None

    # A same-pair unit that is already next to a SWAP, whatever its Paulis, is not moved, and
    # that SWAP takes no other unit.  A unit next to the SWAP on both wires is next to it on the
    # first wire, so only that wire is scanned.
    taken = set()
    pending = []
    for swap in swaps:
        sequence = sequences[first_qubit[swap]]
        at = first_position[swap]
        touching = [
            unit
            for index in (at - 1, at + 1)
            if 0 <= index < len(sequence)
            and pair_keys[unit := sequence[index]] == pair_keys[swap]
            and abs(position_on_wire(second_qubit[swap], unit) - second_position[swap]) == 1
        ]
        if touching:
            taken.update(touching)
        else:
            pending.append(swap)

    moves = []
    for swap in pending:
        qubit_a, qubit_b = first_qubit[swap], second_qubit[swap]
        if (
            last_multi[qubit_a] <= first_position[swap]
            or last_multi[qubit_b] <= second_position[swap]
        ):
            continue
        sequence = sequences[qubit_a]
        wire_labels = labels[qubit_a]
        # Search the SWAP's first wire forward, then backward.  In each direction, stop at the
        # first same-pair unit or the first unit with no Pauli, even if the checks below fail.
        for step in (1, -1):
            index = first_position[swap] + step
            while (
                0 <= index < len(sequence)
                and pair_keys[sequence[index]] != pair_keys[swap]
                and wire_labels[index] != "-"
            ):
                index += step
            if not 0 <= index < len(sequence):
                continue
            unit = sequence[index]
            if (
                pair_keys[unit] == pair_keys[swap]
                and unit not in taken
                and commutes_between(qubit_a, unit, swap)
                and commutes_between(qubit_b, unit, swap)
                # Do not split the candidate from a same-pair unit farther away.
                and not (
                    (far := next_multi(qubit_a, unit, step)) is not None
                    and far == next_multi(qubit_b, unit, step)
                    and pair_keys[far] == pair_keys[unit]
                )
            ):
                taken.add(unit)
                moves.append((unit, swap, step))
                break
    members = {unit: groups.get(unit) or [unit] for unit, _, _ in moves}
    return order, members, moves


def _wire_paulis(node: DAGOpNode, cache: dict | None = None) -> str:
    """The Pauli ``node`` commutes with on each of its qubits, as a string ("-" for none).

    Two operations that commute with the same Pauli on every shared qubit are block diagonal in
    the eigenbases of those Paulis, so they commute exactly.  The supported standard gates are
    functions of single-qubit Paulis, diagonal controlled gates or Pauli rotations.  Their labels
    come only from :meth:`.CommutationChecker.check_commutation_entries`, never from the
    checker's numerical check, and are stored per gate name and width in ``cache``.
    """
    width = len(node.qargs)
    if node.is_standard_gate():
        if node.name not in _SUPPORTED_GATES:
            return "-" * width
        key = (node.name, width)
        if cache is not None and key in cache:
            return cache[key]
        representative = _LOOKUP_REPRESENTATIVES.get(node.name, node.op)
        qargs = tuple(range(width))
        found = []
        for wire in qargs:
            axes = [
                axis
                for axis, pauli in _PAULI_PROBES
                if SessionCommutationChecker.check_commutation_entries(
                    representative, qargs, pauli, (wire,)
                )
                is True
            ]
            found.append(axes[0] if len(axes) == 1 else "-")
        label = "".join(found)
        if cache is not None:
            cache[key] = label
        return label
    if node.name == "unitary" and width == 2 and isinstance(node.op, UnitaryGate):
        matrix = node.op.to_matrix()
        if not np.count_nonzero(matrix - np.diag(np.diagonal(matrix))):
            return "ZZ"
    return "-" * width


def _fused_block(swap_node: DAGOpNode, nodes: list[DAGOpNode], step: int):
    """The SWAP and the unit as ``(operations, global_phase)`` in a short exact form, or ``None``.

    The operations are ``(gate, indices)`` pairs, with indices into the SWAP's qubits.  A CX
    unit gives two CX gates.  A diagonal unit gives three CX gates and up to three phase gates.
    """
    wires = {qubit: index for index, qubit in enumerate(swap_node.qargs)}
    if len(nodes) == 1 and nodes[0].name == "cx" and nodes[0].is_standard_gate():
        control, target = (wires[qubit] for qubit in nodes[0].qargs)
        pairs = [(control, target), (target, control)]
        return [(CXGate(), pair) for pair in (pairs if step > 0 else pairs[::-1])], 0.0
    matrix = np.eye(4, dtype=complex)
    for node in nodes:
        if node.is_parameterized():
            return None
        gate = node.op.to_matrix()
        if len(node.qargs) == 1:
            gate = np.kron(gate, np.eye(2)) if wires[node.qargs[0]] else np.kron(np.eye(2), gate)
        elif wires[node.qargs[0]]:
            gate = _SWAP_MATRIX @ gate @ _SWAP_MATRIX
        matrix = gate @ matrix
    diagonal = np.diagonal(matrix)
    if np.count_nonzero(matrix - np.diag(diagonal)):
        return None
    # With phases (c, c + a + g, c + b + g, c + a + b) on |00>, |01>, |10>, |11> (qubit 0 is
    # the low bit), the unit is P(a) on qubit 0, P(b) on qubit 1, and a phase g on odd parity.
    phase_00, phase_01, phase_10, phase_11 = np.angle(diagonal)
    parity = (phase_01 + phase_10 - phase_00 - phase_11) / 2

    def phase(angle, index):
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return [(PhaseGate(angle), (index,))] if abs(angle) > 1e-12 else []

    middle = phase(parity, 1)
    local = phase(phase_01 - phase_00 - parity, 0) + phase(phase_10 - phase_00 - parity, 1)
    if step > 0:
        operations = [(CXGate(), (0, 1)), (CXGate(), (1, 0)), *middle, (CXGate(), (0, 1)), *local]
    else:
        operations = [*local, (CXGate(), (0, 1)), *middle, (CXGate(), (1, 0)), (CXGate(), (0, 1))]
    return operations, float(phase_00)
