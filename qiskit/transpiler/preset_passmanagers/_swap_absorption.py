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

import logging

import numpy as np

from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.circuit.commutation_library import SessionCommutationChecker
from qiskit.circuit.library import CXGate, CZGate, PhaseGate, SwapGate, UnitaryGate
from qiskit.circuit.library import XGate, YGate, ZGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.exceptions import QiskitError
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passmanager import PassManager

logger = logging.getLogger(__name__)

# Explicit allowlists, so that new commutation table entries do not enable new moves.
_TWO_QUBIT_GATES = frozenset("cz cp crz cu1 cs csdg rzz cx crx csx rzx rxx ryy".split())
_SINGLE_QUBIT_GATES = frozenset("rz z s sdg t tdg p u1 rx x sx sxdg ry y".split())
_SUPPORTED_GATES = _SINGLE_QUBIT_GATES | _TWO_QUBIT_GATES
_PAULI_PROBES = (("X", XGate()), ("Y", YGate()), ("Z", ZGate()))
# Parameter-free stand-ins used only for the commutation table lookup: RX, RY and RZ are
# functions of their Pauli, P and U1 of Z, and CP and CU1 of CZ.
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
    """Move two-qubit gates next to a ``swap`` on the same pair of qubits.

    A supported standard gate, a diagonal two-qubit :class:`.UnitaryGate`, or a
    ``cx(a, b); D(b); cx(a, b)`` group with ``D`` a diagonal single-qubit standard gate, is
    moved across operations that commute with the same Pauli on every shared qubit, so that the
    ``swap`` and the gate can be synthesized as one block [1, 2].  A ``swap`` and a ``cx`` are
    written as two CX gates, and a ``swap`` and bound diagonal standard gates as three CX gates
    and up to three phase gates; other gates stay next to the ``swap``.

    With ``move_diagonal=False``, diagonal gates are not moved; the preset pass managers use it
    for targets with ``rzz``, where a diagonal gate is already cheap.  With ``check``, a list of
    passes, the moves are made only if these passes, followed by ``synthesis`` for blocks that
    keep the ``swap``, turn every block on the ``swap``'s qubits into instructions that
    ``target`` (or ``basis_gates`` if there is no target) supports.

    [1] Liu, Li and Zhou, `arXiv:2205.10596 <https://arxiv.org/abs/2205.10596>`_

    [2] Lao and Browne, `arXiv:2108.02099 <https://arxiv.org/abs/2108.02099>`_
    """

    def __init__(
        self, *, move_diagonal=True, target=None, basis_gates=None, check=None, synthesis=()
    ):
        super().__init__()
        self.move_diagonal = move_diagonal
        self.target = target
        self.basis_gates = basis_gates
        self.check = None if check is None else PassManager(check)
        self.synthesis = PassManager(synthesis)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        ops = dag.count_ops(recurse=False)
        if "swap" not in ops or (
            _TWO_QUBIT_GATES.isdisjoint(ops)
            and not any(map(_is_diagonal_unitary, dag.named_nodes("unitary")))
        ):
            return dag
        blocks = {}  # equal moves share one block
        to_check = {}  # each block once on each pair of physical qubits
        moves = []
        for nodes, swap, step in _find_moves(dag, self.move_diagonal):
            key = _block_key(swap, nodes, step) or object()  # no key: a block of its own
            if key not in blocks:
                blocks[key] = _block_operations(swap, nodes, step)
            qubits = tuple(dag.find_bit(qubit).index for qubit in swap.qargs)
            to_check[key, qubits] = (blocks[key][0], qubits)
            moves.append((nodes, swap, key))
        if not moves or (
            self.check is not None and not self._supported(dag.num_qubits(), to_check.values())
        ):
            return dag
        for nodes, _, _ in moves:
            for node in nodes:
                dag.remove_op_node(node)
        dags = {}
        for _, swap, key in moves:
            if key not in dags:
                operations, global_phase = blocks[key]
                block = dags[key] = DAGCircuit()
                block.add_qubits([Qubit(), Qubit()])
                block.global_phase = global_phase
                for gate, indices in operations:
                    gate_qubits = [block.qubits[index] for index in indices]
                    block.apply_operation_back(gate, gate_qubits, check=False)
            dag.substitute_node_with_dag(swap, dags[key], wires=dags[key].qubits)
        return dag

    def _supported(self, num_qubits: int, blocks) -> bool:
        """Whether the check turns the blocks into instructions supported on their qubits."""
        # Blocks that keep the swap also go through ``synthesis``, which merges the moved gates
        # with the swap.  Barriers keep the blocks apart.
        rewritten, with_swap = QuantumCircuit(num_qubits), QuantumCircuit(num_qubits)
        for operations, qubits in blocks:
            checked = with_swap if any(gate.name == "swap" for gate, _ in operations) else rewritten
            for gate, indices in operations:
                checked.append(gate, [qubits[index] for index in indices], copy=False)
            checked.barrier(qubits)
        try:
            checked = self.check.run(rewritten)
            if with_swap.data:
                checked.compose(self.synthesis.run(self.check.run(with_swap)), inplace=True)
        except QiskitError as error:
            logger.debug("Not moving gates next to swaps: the check raised %r", error)
            return False
        for instruction in checked.data:
            name = instruction.operation.name
            if name == "barrier":
                continue
            if self.target is None:
                supported = name in self.basis_gates
            else:
                qargs = tuple(checked.find_bit(qubit).index for qubit in instruction.qubits)
                params = instruction.operation.params
                supported = self.target.instruction_supported(name, qargs, parameters=params)
            if not supported:
                return False
        return True


def _find_moves(dag: DAGCircuit, move_diagonal: bool) -> list:
    """Return the moves as ``(nodes, swap, step)``, all decided on the original DAG.

    ``nodes`` are the nodes of a unit: one node or a ``cx(a, b); D(b); cx(a, b)`` group,
    numbered by the topological index of its first node.  ``step`` is +1 for a later unit
    moved to just after the SWAP and -1 for an earlier unit moved to just before it.
    """
    order = list(dag.topological_op_nodes())
    wire_of = {qubit: wire for wire, qubit in enumerate(dag.qubits)}
    # Per wire, at each position: the unit there, the Pauli it commutes with on this wire
    # ("-" for none), and the first position of the run of equal Paulis that ends there.
    wires = [[] for _ in wire_of]
    paulis = [[] for _ in wire_of]
    runs = [[] for _ in wire_of]
    last_multi = [-1] * len(wire_of)  # position of the last unit that is not single-qubit
    # Per unit, in flat lists for speed: its pair number (None for a single-qubit unit, -1 for
    # a wider one), its two wires in qarg order, and its positions on those wires.
    pair = [None] * len(order)
    first_wire = [None] * len(order)
    second_wire = [None] * len(order)
    first_position = [None] * len(order)
    second_position = [None] * len(order)
    pair_numbers = {}
    groups = {}  # First CX's topological index -> all three nodes' indices.
    swaps = []
    cache = {}  # gate name -> Pauli label

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
        """Return the first CX of an adjacent ``cx(a, b); D(b)``, or ``None``."""
        if len(wires[b]) < 2 or paulis[b][-1] != "Z":
            return None
        head, middle = wires[b][-2:]
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
        a, b = wire_of[qargs[0]], wire_of[qargs[1]]
        head = group_head(a, b) if label == "ZX" and node.name == "cx" else None
        if head is not None:
            # From now on the first CX stands for the group, which commutes with Z on wire b.
            groups[head] = [head, wires[b][-1], unit]
            # Drop D(b) from wire b's record; the CX's entry now stands for the group and
            # commutes with Z on b, so relabel it and recompute the start of its Z run.
            wires[b].pop()
            paulis[b].pop()
            runs[b].pop()
            last = len(runs[b]) - 1
            paulis[b][last] = "Z"
            runs[b][last] = runs[b][last - 1] if last and paulis[b][last - 1] == "Z" else last
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
        """Whether ``unit`` can move to ``swap``, not splitting it from a same-pair unit beyond."""
        a, b = first_wire[swap], second_wire[swap]
        if not (commutes_between(a, unit, swap) and commutes_between(b, unit, swap)):
            return False
        far = next_multi(a, unit, step)
        return far is None or far != next_multi(b, unit, step) or pair[far] != pair[unit]

    # A unit already next to a SWAP on both wires stays, and that SWAP takes no other unit.
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
        # Skip a SWAP with no later multi-qubit operation (a barrier counts) on one of its
        # wires, where a fused diagonal form would leave trailing phase gates.  The skip is
        # deliberate for all forms, so that a SWAP at the end of a wire is left as it is.
        if last_multi[a] <= first_position[swap] or last_multi[b] <= second_position[swap]:
            continue
        units, labels, swap_pair = wires[a], paulis[a], pair[swap]
        for step in (1, -1):
            # Walk the first wire to the nearest same-pair unit, stopping at one with no Pauli.
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

    Operations that commute with the same Pauli on every shared qubit commute exactly.  For
    standard gates, labels come only from :meth:`.CommutationChecker.check_commutation_entries`,
    never from its numerical check.  The label of an allowlisted gate does not depend on its
    angle, so the cache is keyed by gate name.
    """
    if not node.is_standard_gate():
        return "ZZ" if _is_diagonal_unitary(node) else "-" * len(node.qargs)
    name = node.name
    if name not in _SUPPORTED_GATES:
        return "-" * len(node.qargs)
    if name not in cache:
        gate = _LOOKUP_REPRESENTATIVES.get(name) or node.op
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
    if node.name != "unitary" or len(node.qargs) != 2 or not isinstance(node.op, UnitaryGate):
        return False
    matrix = node.op.to_matrix()
    return not np.count_nonzero(matrix - np.diag(np.diagonal(matrix)))


def _block_key(swap, nodes: list, step: int):
    """Return a key that determines the block of ``swap`` and ``nodes``, or ``None``.

    The key holds the direction, the qubit order, and the gate names and angles.  It exists
    only for standard gates with float angles; ``float.hex`` tells 0.0 and -0.0 apart.
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
    """Return the SWAP and ``nodes`` as ``(operations, global_phase)``.

    The operations are ``(gate, indices)`` pairs on the SWAP's qubits.  A CX and bound diagonal
    standard gates are fused without approximation; the angles carry floating-point rounding.
    Other gates, such as a :class:`.UnitaryGate`, whose matrix may be unitary only within a
    tolerance, are kept next to the SWAP.
    """
    index_of = {qubit: index for index, qubit in enumerate(swap.qargs)}
    if len(nodes) == 1 and nodes[0].name == "cx" and nodes[0].is_standard_gate():
        control, target = (index_of[qubit] for qubit in nodes[0].qargs)
        if step > 0:
            return [_cx(control, target), _cx(target, control)], 0.0
        return [_cx(target, control), _cx(control, target)], 0.0
    can_fuse = all(node.is_standard_gate() and not node.is_parameterized() for node in nodes)
    phases = _phases(nodes, index_of) if can_fuse else None
    if phases is None:
        ordered = [swap, *nodes] if step > 0 else [*nodes, swap]
        return [(node.op, [index_of[qubit] for qubit in node.qargs]) for node in ordered], 0.0
    # Phases (c, c + a + g, c + b + g, c + a + b) on |00>, |01>, |10>, |11>, with qubit 0 the
    # low bit, are P(a) on qubit 0, P(b) on qubit 1 and a phase g on odd parity.
    phase_00, phase_01, phase_10, phase_11 = phases
    parity = (phase_01 + phase_10 - phase_00 - phase_11) / 2

    def phase(angle, index):
        # Wrap to [-pi, pi), unless adding pi would erase the angle.
        if np.pi + angle != np.pi:
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


def _phases(nodes: list, index_of: dict):
    """Return the diagonal phases of the gates' product, or ``None`` if it is not diagonal."""
    matrix = np.eye(4, dtype=complex)
    for node in nodes:
        gate = node.op.to_matrix()
        if len(node.qargs) == 1:
            gate = np.kron(gate, np.eye(2)) if index_of[node.qargs[0]] else np.kron(np.eye(2), gate)
        elif index_of[node.qargs[0]]:
            gate = _SWAP_MATRIX @ gate @ _SWAP_MATRIX
        matrix = gate @ matrix
    diagonal = np.diagonal(matrix)
    return None if np.count_nonzero(matrix - np.diag(diagonal)) else np.angle(diagonal)
