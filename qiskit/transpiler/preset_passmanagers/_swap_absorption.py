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

from dataclasses import dataclass

import numpy as np

from qiskit.circuit import Qubit
from qiskit.circuit.commutation_library import SessionCommutationChecker
from qiskit.circuit.library import CZGate, UnitaryGate, XGate, YGate, ZGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass

# Keep the admitted gate families explicit: extra table coverage must not enable new moves.
_SUPPORTED_GATES = frozenset(
    (
        "rz",
        "z",
        "s",
        "sdg",
        "t",
        "tdg",
        "p",
        "u1",
        "rx",
        "x",
        "sx",
        "sxdg",
        "ry",
        "y",
        "cz",
        "cp",
        "crz",
        "cu1",
        "cs",
        "csdg",
        "rzz",
        "cx",
        "crx",
        "csx",
        "rzx",
        "rxx",
        "ryy",
    )
)
_PAULI_PROBES = (("X", XGate()), ("Y", YGate()), ("Z", ZGate()))
# Parameter-independent proof representatives, never replacements in the circuit. RX/RY/RZ
# are functions of their Paulis; P/U1 are functions of Z; CP/CU1 use (I - CZ) / 2.
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


class _AbsorbIntoSwaps(TransformationPass):
    """Move a two-qubit interaction next to a ``swap`` on the same qubit pair.

    The interaction is moved exactly, across operations it commutes with, so that the two-qubit
    peephole optimization can synthesize the ``swap`` and the interaction as a single block
    [1, 2], for example two CX gates for a ``swap`` followed by a ``cx``.  A unit that can be
    moved is a standard gate, a diagonal two-qubit :class:`.UnitaryGate`, or
    ``cx(a, b); D(b); cx(a, b)`` with ``D`` diagonal.  It is only moved past units that commute
    with the same Pauli on every shared qubit, and never past a ``swap``.

    A ``swap`` is left alone when one of its qubits would end on the merged block's single-qubit
    gates, which an as-late-as-possible schedule leaves idle until the end of the circuit.  That
    is the case when no multi-qubit operation follows the ``swap`` on the qubit.  Directives such
    as a ``barrier`` count as multi-qubit operations here, as they fence the qubit's schedule.

    References:

    [1] Liu, Li and Zhou. Not All SWAPs Have the Same Cost: A Case for Optimization-Aware Qubit
    Routing. `arXiv:2205.10596 <https://arxiv.org/abs/2205.10596>`_

    [2] Lao and Browne. 2QAN: A quantum compiler for 2-local qubit Hamiltonian simulation
    algorithms. `arXiv:2108.02099 <https://arxiv.org/abs/2108.02099>`_
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        if "swap" not in dag.count_ops(recurse=False):
            return dag
        order, units, moves = _find_moves(dag)
        return _apply_moves(dag, order, units, moves) if moves else dag


class _Members:
    """Lazy view of each unit's ``(member node indices, qubit indices)``.

    Only CX-diagonal-CX groups need stored member lists; other units contain their first node.
    """

    __slots__ = ("first", "groups", "q0", "q1", "qargs")

    def __init__(self, first, groups, q0, q1, qargs):
        self.first = first
        self.groups = groups
        self.q0 = q0
        self.q1 = q1
        self.qargs = qargs

    def __len__(self):
        return len(self.first)

    def __getitem__(self, unit):
        head = self.first[unit]
        if self.q1[head] >= 0:
            qubits = frozenset((self.q0[head], self.q1[head]))
        else:
            qubits = frozenset(self.qargs(head))
        return (self.groups.get(unit) or [head], qubits)


@dataclass(frozen=True, slots=True)
class _NodeIndex:
    """Node indices along each wire and per-node qubits and wire positions.

    First-qubit entries are populated for one- and two-qubit nodes, second-qubit entries only
    for two-qubit nodes. Other entries remain -1. ``last_multi[q]`` is the wire position of the
    last multi-qubit node on qubit ``q``, or -1 if there is none.
    """

    wires: list[list[int]]
    last_multi: list[int]
    first_qubit: list[int]
    second_qubit: list[int]
    first_position: list[int]
    second_position: list[int]
    widths: list[int]


@dataclass(frozen=True, slots=True)
class _UnitIndex:
    """Nodes grouped into units, numbered in the topological order of their first nodes.

    A unit is one node or a CX-diagonal-CX group. Labels follow the first node's qubit order;
    pair keys ignore that order. ``groups`` stores members only for the three-node units.
    """

    node_to_unit: list[int]
    first_node: list[int]
    groups: dict[int, list[int]]
    labels: list[str]
    pair_keys: list[int]
    swaps: list[int]


@dataclass(frozen=True, slots=True)
class _WireIndex:
    """Unit indices and local Pauli labels along each wire, with consecutive repeats merged.

    ``run_starts[q][p]`` starts the equal-label run containing wire position ``p`` on qubit ``q``.
    The first/second positions are per two-qubit unit, rather than per node as in ``_NodeIndex``.
    """

    sequences: list[list[int]]
    labels: list[list[str]]
    run_starts: list[list[int]]
    first_position: list[int]
    second_position: list[int]


def _find_moves(dag: DAGCircuit):
    """Return the topological nodes, their unit membership, and the selected moves.

    Each move is ``(unit, swap, step)``. A positive step moves a later unit just after the SWAP;
    a negative step moves an earlier unit just before it. All decisions use the original DAG.
    """
    order = list(dag.topological_op_nodes())
    qubit_index = {qubit: index for index, qubit in enumerate(dag.qubits)}
    nodes = _index_nodes(order, qubit_index, dag.num_qubits())
    units = _group_nodes(order, nodes)
    wires = _index_unit_wires(nodes, units)
    moves = _select_moves(nodes, units, wires)

    def qubits_for_node(node):
        return [qubit_index[qubit] for qubit in order[node].qargs]

    members = _Members(
        units.first_node, units.groups, nodes.first_qubit, nodes.second_qubit, qubits_for_node
    )
    return order, members, moves


def _index_nodes(order, qubit_index, num_wires: int) -> _NodeIndex:
    """Record node positions and the last multi-qubit operation on every wire."""
    wires = [[] for _ in range(num_wires)]
    last_multi = [-1] * num_wires
    num_nodes = len(order)
    first_qubit = [-1] * num_nodes
    second_qubit = [-1] * num_nodes
    first_position = [-1] * num_nodes
    second_position = [-1] * num_nodes
    widths = [0] * num_nodes
    for node, op_node in enumerate(order):
        qubits = op_node.qargs
        width = len(qubits)
        widths[node] = width
        if width == 1:
            qubit = qubit_index[qubits[0]]
            wire = wires[qubit]
            first_qubit[node] = qubit
            first_position[node] = len(wire)
            wire.append(node)
        elif width == 2:
            qubit_a = qubit_index[qubits[0]]
            qubit_b = qubit_index[qubits[1]]
            wire_a = wires[qubit_a]
            wire_b = wires[qubit_b]
            first_qubit[node] = qubit_a
            second_qubit[node] = qubit_b
            first_position[node] = last_multi[qubit_a] = len(wire_a)
            second_position[node] = last_multi[qubit_b] = len(wire_b)
            wire_a.append(node)
            wire_b.append(node)
        else:
            for qubit in qubits:
                wire = wires[qubit_index[qubit]]
                if width > 1:
                    last_multi[qubit_index[qubit]] = len(wire)
                wire.append(node)
    return _NodeIndex(
        wires, last_multi, first_qubit, second_qubit, first_position, second_position, widths
    )


def _group_nodes(order, nodes: _NodeIndex) -> _UnitIndex:
    """Treat each node as a unit, grouping exactly ``CX(a, b); D(b); CX(a, b)`` for diagonal D."""
    cache = {}
    num_wires = len(nodes.wires)

    def next_on_wire(qubit, position):
        wire = nodes.wires[qubit]
        return wire[position + 1] if position + 1 < len(wire) else None

    node_to_unit = [-1] * len(order)
    first_node = []
    groups = {}
    labels = []
    pair_keys = []
    swaps = []
    for node, op_node in enumerate(order):
        if node_to_unit[node] >= 0:
            continue
        width = nodes.widths[node]
        label = _wire_paulis(op_node, cache)
        unit = len(first_node)
        if label == "ZX" and op_node.name == "cx" and op_node.is_standard_gate():
            control, target = nodes.first_qubit[node], nodes.second_qubit[node]
            middle = next_on_wire(target, nodes.second_position[node])
            if (
                middle is not None
                and _wire_paulis(order[middle], cache) == "Z"
                and (last := next_on_wire(target, nodes.first_position[middle])) is not None
                and order[last].name == "cx"
                and order[last].is_standard_gate()
                and nodes.first_qubit[last] == control
                and nodes.second_qubit[last] == target
                and nodes.widths[last] == 2
                and next_on_wire(control, nodes.first_position[node]) == last
            ):
                groups[unit] = [node, middle, last]
                node_to_unit[middle] = node_to_unit[last] = unit
                label = "ZZ"
        node_to_unit[node] = unit
        first_node.append(node)
        labels.append(label)
        if width == 2:
            qubit_a, qubit_b = nodes.first_qubit[node], nodes.second_qubit[node]
            pair_keys.append(
                qubit_a * num_wires + qubit_b
                if qubit_a < qubit_b
                else qubit_b * num_wires + qubit_a
            )
            if op_node.name == "swap" and op_node.is_standard_gate():
                swaps.append(unit)
        else:
            pair_keys.append(_SINGLE_QUBIT_UNIT if width == 1 else _OTHER_UNIT)
    return _UnitIndex(node_to_unit, first_node, groups, labels, pair_keys, swaps)


def _index_unit_wires(nodes: _NodeIndex, units: _UnitIndex) -> _WireIndex:
    """Index units on each wire and precompute equal-Pauli runs for commutation checks."""
    num_units = len(units.first_node)
    first_position = [-1] * num_units
    second_position = [-1] * num_units
    sequences = []
    wire_labels = []
    run_starts = []
    for qubit, wire in enumerate(nodes.wires):
        sequence = []
        labels = []
        starts = []
        previous_unit = -1
        for node in wire:
            unit = units.node_to_unit[node]
            if unit == previous_unit:
                continue
            previous_unit = unit
            position = len(sequence)
            label = units.labels[unit]
            head = units.first_node[unit]
            if units.pair_keys[unit] >= 0:
                if nodes.first_qubit[head] == qubit:
                    first_position[unit] = position
                    label = label[0]
                else:
                    second_position[unit] = position
                    label = label[1]
            else:
                label = label[0] if label else "-"
            starts.append(starts[-1] if position and labels[-1] == label else position)
            sequence.append(unit)
            labels.append(label)
        sequences.append(sequence)
        wire_labels.append(labels)
        run_starts.append(starts)
    return _WireIndex(sequences, wire_labels, run_starts, first_position, second_position)


def _select_moves(nodes: _NodeIndex, units: _UnitIndex, wires: _WireIndex):
    """Reserve existing SWAP neighbors, then choose at most one movable unit per remaining SWAP."""

    def position_on_wire(qubit, unit):
        if nodes.first_qubit[units.first_node[unit]] == qubit:
            return wires.first_position[unit]
        return wires.second_position[unit]

    def commutes_between(qubit, unit, swap):
        """Check the open interval between a candidate and its SWAP on one wire."""
        unit_position = position_on_wire(qubit, unit)
        swap_position = position_on_wire(qubit, swap)
        low, high = (
            (unit_position, swap_position)
            if unit_position < swap_position
            else (swap_position, unit_position)
        )
        labels = wires.labels[qubit]
        pauli = labels[unit_position]
        if pauli == "-":
            return False
        # Every intervening label must belong to one run matching the candidate's Pauli.
        return high - low <= 1 or (
            labels[high - 1] == pauli and wires.run_starts[qubit][high - 1] <= low + 1
        )

    def splits_same_pair_run(unit, step, qubit_a, qubit_b):
        """Check whether removal separates the candidate from a same-pair neighbor farther away."""
        neighbours = []
        for qubit in (qubit_a, qubit_b):
            sequence = wires.sequences[qubit]
            index = position_on_wire(qubit, unit) + step
            while (
                0 <= index < len(sequence)
                and units.pair_keys[sequence[index]] == _SINGLE_QUBIT_UNIT
            ):
                index += step
            neighbours.append(sequence[index] if 0 <= index < len(sequence) else None)
        return (
            neighbours[0] is not None
            and neighbours[0] == neighbours[1]
            and units.pair_keys[neighbours[0]] == units.pair_keys[unit]
        )

    # Reserve all existing neighbors before selecting any moves, including neighbors of later
    # or terminal SWAPs. Any same-pair neighbor counts, regardless of its Pauli labels.
    taken = set()
    pending = []
    for swap in units.swaps:
        pair_key = units.pair_keys[swap]
        head = units.first_node[swap]
        qubit_a, qubit_b = nodes.first_qubit[head], nodes.second_qubit[head]
        touching = []
        # A unit next to the SWAP on both wires is a neighbor on the first wire, so scan only
        # that wire and check the second.
        sequence = wires.sequences[qubit_a]
        at = wires.first_position[swap]
        for index in (at - 1, at + 1):
            if (
                0 <= index < len(sequence)
                and units.pair_keys[unit := sequence[index]] == pair_key
                and abs(position_on_wire(qubit_b, unit) - wires.second_position[swap]) == 1
            ):
                touching.append(unit)
        if touching:
            taken.update(touching)
        else:
            pending.append(swap)

    moves = []
    for swap in pending:
        head = units.first_node[swap]
        qubit_a, qubit_b = nodes.first_qubit[head], nodes.second_qubit[head]
        if (
            nodes.last_multi[qubit_a] <= nodes.first_position[head]
            or nodes.last_multi[qubit_b] <= nodes.second_position[head]
        ):
            continue
        pair_key = units.pair_keys[swap]
        sequence = wires.sequences[qubit_a]
        labels = wires.labels[qubit_a]
        size = len(sequence)
        # Search the SWAP's first-qubit wire, preferring a later candidate. In either direction,
        # stop at the first same-pair unit or unknown label, even if it fails a remaining guard.
        for step in (1, -1):
            index = wires.first_position[swap] + step
            while (
                0 <= index < size
                and units.pair_keys[sequence[index]] != pair_key
                and labels[index] != "-"
            ):
                index += step
            if not 0 <= index < size:
                continue
            unit = sequence[index]
            if (
                units.pair_keys[unit] == pair_key
                and unit not in taken
                and commutes_between(qubit_a, unit, swap)
                and commutes_between(qubit_b, unit, swap)
                and not splits_same_pair_run(unit, step, qubit_a, qubit_b)
            ):
                taken.add(unit)
                moves.append((unit, swap, step))
                break
    return moves


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


def _standard_wire_paulis(op, width: int) -> str:
    """Derive local labels for a verified, allowlisted standard gate using only table rules.

    Two operations commuting with the same local Pauli on every shared wire are block
    diagonal in those Pauli eigenbases and hence commute exactly. The admitted families
    are functions of local Paulis, diagonal controlled gates, or Pauli-product rotations.
    This uses the repaired raw lookup API, never the checker's approximate acceptance paths.
    """
    representative = _LOOKUP_REPRESENTATIVES.get(op.name, op)
    qargs = tuple(range(width))
    labels = []
    for wire in range(width):
        axes = [
            axis
            for axis, pauli in _PAULI_PROBES
            if SessionCommutationChecker.check_commutation_entries(
                representative, qargs, pauli, (wire,)
            )
            is True
        ]
        labels.append(axes[0] if len(axes) == 1 else "-")
    return "".join(labels)


def _wire_paulis(node: DAGOpNode, cache: dict | None = None) -> str:
    """The Pauli ``node`` commutes with on each of its qubits, as a string ("-" for none)."""
    width = len(node.qargs)
    if node.is_standard_gate():
        if node.name not in _SUPPORTED_GATES:
            return "-" * width
        key = (node.name, width)
        if cache is not None and key in cache:
            return cache[key]
        label = _standard_wire_paulis(node.op, width)
        if cache is not None:
            cache[key] = label
        return label
    if node.name == "unitary" and width == 2 and isinstance(node.op, UnitaryGate):
        matrix = node.op.to_matrix()
        if not np.count_nonzero(matrix - np.diag(np.diagonal(matrix))):
            return "ZZ"
    return "-" * width
