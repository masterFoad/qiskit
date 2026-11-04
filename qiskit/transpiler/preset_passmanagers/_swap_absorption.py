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
from qiskit.circuit.library import CXGate, CZGate, PhaseGate, SwapGate, UnitaryGate
from qiskit.circuit.library import XGate, YGate, ZGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

# The gates are listed explicitly, so that new commutation table entries do not enable
# new moves.
_TWO_QUBIT_GATES = frozenset("cz cp crz cu1 cs csdg rzz cx crx csx rzx rxx ryy".split())
_SINGLE_QUBIT_GATES = frozenset("rz z s sdg t tdg p u1 rx x sx sxdg ry y".split())
_SUPPORTED_GATES = _SINGLE_QUBIT_GATES | _TWO_QUBIT_GATES
_PAULI_PROBES = (("X", XGate()), ("Y", YGate()), ("Z", ZGate()))
# Gates used in place of a parameterized gate for the commutation table lookup only; the
# circuit is not changed.  RX, RY and RZ are functions of their Pauli, P and U1 of Z, and
# CP and CU1 of CZ.
_LOOKUP_REPRESENTATIVES = {
    "rx": XGate(),
    "ry": YGate(),
    "rz": ZGate(),
    "p": ZGate(),
    "u1": ZGate(),
    "cp": CZGate(),
    "cu1": CZGate(),
}
_SWAP_MATRIX = SwapGate().to_matrix()


class _AbsorbIntoSwaps(TransformationPass):
    """Move a two-qubit gate next to a ``swap`` on the same pair of qubits.

    The gate is moved across operations it commutes with, so that
    :class:`.TwoQubitPeepholeOptimization` can synthesize the ``swap`` and the gate as one
    block [1, 2].  What can be moved is a supported standard two-qubit gate such as ``cx``,
    ``cz``, ``cp``, ``rzz`` or ``rxx``, a diagonal two-qubit :class:`.UnitaryGate`, or
    ``cx(a, b); D(b); cx(a, b)`` with ``D`` a diagonal single-qubit standard gate.  It is
    only moved past operations that commute with the same Pauli on every shared qubit, and
    never past a ``swap``.  A ``swap`` with a ``cx`` is written as two CX gates, and a
    ``swap`` with a diagonal gate without parameters as three CX gates and at most three
    phase gates.

    A ``swap`` is left alone when no multi-qubit operation follows it on one of its qubits,
    since that qubit would end on the single-qubit gates of the merged block, which an
    as-late-as-possible schedule leaves idle until the end of the circuit.  A directive on
    two or more qubits, such as a ``barrier``, counts as a multi-qubit operation here.

    With ``move_diagonal=False``, diagonal gates and ``cx(a, b); D(b); cx(a, b)`` are not
    moved.  The preset pass managers do this for targets with ``rzz``, which runs such a
    gate as one two-qubit gate, so that merging it with a ``swap`` costs more than it saves,
    and for targets that cannot express the CX and phase gates of the merged form.

    [1] Liu, Li and Zhou. Not All SWAPs Have the Same Cost: A Case for Optimization-Aware
    Qubit Routing. `arXiv:2205.10596 <https://arxiv.org/abs/2205.10596>`_

    [2] Lao and Browne. 2QAN: A quantum compiler for 2-local qubit Hamiltonian simulation
    algorithms. `arXiv:2108.02099 <https://arxiv.org/abs/2108.02099>`_
    """

    def __init__(self, move_diagonal: bool = True):
        super().__init__()
        self.move_diagonal = move_diagonal

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        ops = dag.count_ops(recurse=False)
        if "swap" not in ops or (
            _TWO_QUBIT_GATES.isdisjoint(ops)
            and not any(map(_is_diagonal_unitary, dag.named_nodes("unitary")))
        ):
            return dag
        moves = _find_moves(dag, self.move_diagonal)
        for nodes, _, _ in moves:
            for node in nodes:
                dag.remove_op_node(node)
        blocks = {}  # equal units share one block
        for nodes, swap, step in moves:
            key = _block_key(swap, nodes, step)
            block = blocks.get(key)
            if block is None:
                block = DAGCircuit()
                qubits = [Qubit(), Qubit()]
                block.add_qubits(qubits)
                operations, block.global_phase = _block_operations(swap, nodes, step)
                for gate, indices in operations:
                    gate_qubits = [qubits[index] for index in indices]
                    block.apply_operation_back(gate, gate_qubits, check=False)
                if key is not None:
                    blocks[key] = block
            dag.substitute_node_with_dag(swap, block, wires=block.qubits)
        return dag


def _find_moves(dag: DAGCircuit, move_diagonal: bool) -> list:
    """Return the moves as ``(nodes, swap, step)``: the nodes of the moved unit, the SWAP
    node, and a positive step for a later unit moved just after the SWAP or a negative one
    for an earlier unit moved just before it.  A unit is one node or a
    ``cx(a, b); D(b); cx(a, b)`` group, named by the topological index of its first node.
    All decisions use the original DAG."""
    order = list(dag.topological_op_nodes())
    wire_of = {qubit: wire for wire, qubit in enumerate(dag.qubits)}
    # Per wire, at each position: the unit there, the Pauli it commutes with on this wire
    # ("-" for none), and the first position of the run of equal Paulis that ends there.
    wires = [[] for _ in wire_of]
    paulis = [[] for _ in wire_of]
    runs = [[] for _ in wire_of]
    last_multi = [-1] * len(wire_of)  # position of the last unit that is not single-qubit
    # Per unit, in flat lists for speed:
    #   pair: None for a single-qubit unit, -1 for a wider one, else the number of its pair;
    #   first_wire, second_wire: the wires of a two-qubit unit, in qarg order;
    #   first_position, second_position: its positions on those wires.
    pair = [None] * len(order)
    first_wire = [None] * len(order)
    second_wire = [None] * len(order)
    first_position = [None] * len(order)
    second_position = [None] * len(order)
    pair_numbers = {}  # each pair of wires is numbered when it is first seen
    groups = {}  # first node of a group -> its nodes
    swaps = []
    cache = {}

    def add(unit, wire, pauli):
        units, labels, starts = wires[wire], paulis[wire], runs[wire]
        position = len(units)
        starts.append(starts[-1] if position and labels[-1] == pauli else position)
        units.append(unit)
        labels.append(pauli)
        return position

    def position(unit, wire):
        return first_position[unit] if first_wire[unit] == wire else second_position[unit]

    def group_head(a, b):
        """The first CX of a group ``cx(a, b); D(b); cx(a, b)`` that the next unit would
        close, if that CX is still a unit of its own and nothing else is between the three
        on either wire."""
        if len(wires[b]) < 2 or paulis[b][-1] != "Z":
            return None
        middle = wires[b][-1]
        head = wires[b][-2]
        if pair[middle] is not None or first_wire[head] != a or wires[a][-1] != head:
            return None
        if order[head].name != "cx" or paulis[a][-1] != "Z" or paulis[b][-2] != "X":
            return None
        return head

    for unit, node in enumerate(order):
        qargs = node.qargs
        label = _wire_paulis(node, cache)
        if len(qargs) == 1:
            add(unit, wire_of[qargs[0]], label)
            continue
        if len(qargs) != 2:
            pair[unit] = -1
            for qubit, pauli in zip(qargs, label):
                last_multi[wire_of[qubit]] = add(unit, wire_of[qubit], pauli)
            continue
        a = wire_of[qargs[0]]
        b = wire_of[qargs[1]]
        head = group_head(a, b) if label == "ZX" and node.name == "cx" else None
        if head is not None:
            # The first CX stands for the group from now on: D leaves wire b, and the label
            # of the CX there becomes Z, which the whole group commutes with.
            groups[head] = [head, wires[b][-1], unit]
            wires[b].pop()
            paulis[b].pop()
            runs[b].pop()
            last = len(runs[b]) - 1
            paulis[b][last] = "Z"
            if last and paulis[b][last - 1] == "Z":
                runs[b][last] = runs[b][last - 1]
            else:
                runs[b][last] = last
            continue
        sorted_pair = (a, b) if a < b else (b, a)
        pair[unit] = pair_numbers.setdefault(sorted_pair, len(pair_numbers))
        first_wire[unit] = a
        second_wire[unit] = b
        if node.name == "swap" and node.is_standard_gate():
            swaps.append(unit)
        first_position[unit] = last_multi[a] = add(unit, a, label[0])
        second_position[unit] = last_multi[b] = add(unit, b, label[1])

    def next_multi(wire, unit, step):
        """The nearest unit past ``unit`` on ``wire`` that is not single-qubit, if any."""
        units = wires[wire]
        at = position(unit, wire) + step
        while 0 <= at < len(units) and pair[units[at]] is None:
            at += step
        return units[at] if 0 <= at < len(units) else None

    def commutes_between(wire, unit, swap):
        """Whether ``unit`` commutes with everything between it and ``swap`` on ``wire``."""
        low, high = sorted((position(unit, wire), position(swap, wire)))
        pauli = paulis[wire][position(unit, wire)]
        if pauli == "-":
            return False
        # Every Pauli in between must be in one run of the unit's Pauli.
        return high - low <= 1 or (
            paulis[wire][high - 1] == pauli and runs[wire][high - 1] <= low + 1
        )

    def can_move(unit, swap, step):
        """Whether ``unit`` commutes with everything between it and ``swap``, and is not
        split from a same-pair unit farther away."""
        a, b = first_wire[swap], second_wire[swap]
        if not (commutes_between(a, unit, swap) and commutes_between(b, unit, swap)):
            return False
        far = next_multi(a, unit, step)
        return far is None or far != next_multi(b, unit, step) or pair[far] != pair[unit]

    # A unit already next to a SWAP on both wires is not moved, and that SWAP takes no
    # other unit.
    taken = set()
    pending = []
    for swap in swaps:
        a, b = first_wire[swap], second_wire[swap]
        units = wires[a]
        at = first_position[swap]
        touching = [
            units[near]
            for near in (at - 1, at + 1)
            if 0 <= near < len(units)
            and pair[units[near]] == pair[swap]
            and abs(position(units[near], b) - second_position[swap]) == 1
        ]
        if touching:
            taken.update(touching)
        else:
            pending.append(swap)

    moves = []
    for swap in pending:
        a, b = first_wire[swap], second_wire[swap]
        if last_multi[a] <= first_position[swap] or last_multi[b] <= second_position[swap]:
            continue
        units, labels, swap_pair = wires[a], paulis[a], pair[swap]
        for step in (1, -1):
            # On the SWAP's first wire, stop at the first same-pair unit or the first unit
            # with no Pauli.
            at = first_position[swap] + step
            while 0 <= at < len(units) and pair[units[at]] != swap_pair and labels[at] != "-":
                at += step
            if not 0 <= at < len(units):
                continue
            unit = units[at]
            if pair[unit] != swap_pair or unit in taken:
                continue
            # A diagonal unit that must stay blocks only this side of the SWAP.
            if not move_diagonal and labels[at] == "Z" and paulis[b][position(unit, b)] == "Z":
                continue
            if can_move(unit, swap, step):
                taken.add(unit)
                moves.append((unit, swap, step))
                break
    return [
        ([order[member] for member in groups.get(unit, [unit])], order[swap], step)
        for unit, swap, step in moves
    ]


def _wire_paulis(node, cache: dict) -> str:
    """The Pauli ``node`` commutes with on each of its qubits, as a string ("-" for none).

    Two operations that commute with the same Pauli on every shared qubit are block
    diagonal in the eigenbases of those Paulis, so they commute exactly.  The labels of the
    supported standard gates come only from
    :meth:`.CommutationChecker.check_commutation_entries`, never from the checker's
    numerical check, and are stored per gate name in ``cache``.
    """
    if not node.is_standard_gate():
        return "ZZ" if _is_diagonal_unitary(node) else "-" * len(node.qargs)
    name = node.name
    if name not in _SUPPORTED_GATES:
        return "-" * len(node.qargs)
    if name not in cache:
        gate = _LOOKUP_REPRESENTATIVES.get(name, node.op)
        qargs = tuple(range(len(node.qargs)))
        label = ""
        for wire in qargs:
            axes = []
            for axis, pauli in _PAULI_PROBES:
                entry = SessionCommutationChecker.check_commutation_entries(
                    gate, qargs, pauli, (wire,)
                )
                if entry is True:
                    axes.append(axis)
            label += axes[0] if len(axes) == 1 else "-"
        cache[name] = label
    return cache[name]


def _is_diagonal_unitary(node) -> bool:
    if node.name != "unitary" or len(node.qargs) != 2:
        return False
    if not isinstance(node.op, UnitaryGate):
        return False
    matrix = node.op.to_matrix()
    return not np.count_nonzero(matrix - np.diag(np.diagonal(matrix)))


def _block_key(swap, nodes: list, step: int):
    """A key that determines the block of the SWAP and the unit, or ``None``.

    The key holds the direction, whether the unit's first qubit is the SWAP's second qubit,
    and the gate names and exact angles (``float.hex`` tells 0.0 and -0.0 apart).  A unit
    with a gate that is not a standard gate with float angles, such as a
    :class:`.UnitaryGate`, has no key.
    """
    key = [step, nodes[0].qargs[0] != swap.qargs[0]]
    for node in nodes:
        if not node.is_standard_gate():
            return None
        key.append(node.name)
        for angle in node.params:
            if type(angle) is not float:
                return None
            key.append(angle.hex())
    return tuple(key)


def _block_operations(swap, nodes: list, step: int):
    """The SWAP and the unit as ``(operations, global_phase)``.

    The operations are ``(gate, indices)`` pairs, with indices into the SWAP's qubits.  A CX
    unit gives two CX gates, and a diagonal unit without parameters three CX gates and up
    to three phase gates; both forms are exact.  Any other unit keeps its gates, next to
    the SWAP.
    """
    wires = {qubit: index for index, qubit in enumerate(swap.qargs)}
    if len(nodes) == 1 and nodes[0].name == "cx" and nodes[0].is_standard_gate():
        control, target = (wires[qubit] for qubit in nodes[0].qargs)
        if step > 0:
            return [_cx(control, target), _cx(target, control)], 0.0
        return [_cx(target, control), _cx(control, target)], 0.0
    phases = None if any(node.is_parameterized() for node in nodes) else _phases(nodes, wires)
    if phases is None:
        ordered = [swap, *nodes] if step > 0 else [*nodes, swap]
        return [(node.op, [wires[qubit] for qubit in node.qargs]) for node in ordered], 0.0
    # With phases (c, c + a + g, c + b + g, c + a + b) on |00>, |01>, |10>, |11> (qubit 0
    # is the low bit), the unit is P(a) on qubit 0, P(b) on qubit 1, and a phase g on odd
    # parity.
    phase_00, phase_01, phase_10, phase_11 = phases
    parity = (phase_01 + phase_10 - phase_00 - phase_11) / 2

    def phase(angle, index):
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return [(PhaseGate(angle), (index,))] if angle else []

    middle = phase(parity, 1)
    local = phase(phase_01 - phase_00 - parity, 0) + phase(phase_10 - phase_00 - parity, 1)
    if step > 0:
        operations = [_cx(0, 1), _cx(1, 0), *middle, _cx(0, 1), *local]
    else:
        operations = [*local, _cx(0, 1), *middle, _cx(1, 0), _cx(0, 1)]
    return operations, float(phase_00)


def _cx(control: int, target: int):
    return CXGate(), (control, target)


def _phases(nodes: list, wires: dict):
    """The phases on the diagonal of the unit, or ``None`` when it is not diagonal."""
    matrix = np.eye(4, dtype=complex)
    for node in nodes:
        gate = node.op.to_matrix()
        if len(node.qargs) == 1:
            if wires[node.qargs[0]]:
                gate = np.kron(gate, np.eye(2))
            else:
                gate = np.kron(np.eye(2), gate)
        elif wires[node.qargs[0]]:
            gate = _SWAP_MATRIX @ gate @ _SWAP_MATRIX
        matrix = gate @ matrix
    diagonal = np.diagonal(matrix)
    return None if np.count_nonzero(matrix - np.diag(diagonal)) else np.angle(diagonal)
