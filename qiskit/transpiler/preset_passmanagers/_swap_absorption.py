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
    """``units[u]`` view: ``(members, qubits)`` of unit ``u``, the node indices and qubit indices
    the guard's unit tuples start with (the guard's per-wire Pauli dict is not kept)."""

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


def _find_moves(dag: DAGCircuit):
    order = list(dag.topological_op_nodes())
    getq = {qubit: index for index, qubit in enumerate(dag.qubits)}.__getitem__
    num_wires = dag.num_qubits()
    wires = [[] for _ in range(num_wires)]
    last_multi = [-1] * num_wires
    n = len(order)
    q0 = [-1] * n
    q1 = [-1] * n
    p0 = [-1] * n
    p1 = [-1] * n
    nq = [0] * n
    for node, op_node in enumerate(order):
        qs = op_node.qargs
        k = len(qs)
        nq[node] = k
        if k == 1:
            a = getq(qs[0])
            wa = wires[a]
            q0[node] = a
            p0[node] = len(wa)
            wa.append(node)
        elif k == 2:
            a = getq(qs[0])
            b = getq(qs[1])
            wa = wires[a]
            wb = wires[b]
            q0[node] = a
            q1[node] = b
            p0[node] = last_multi[a] = len(wa)
            p1[node] = last_multi[b] = len(wb)
            wa.append(node)
            wb.append(node)
        else:
            for qubit in qs:
                w = wires[getq(qubit)]
                if k > 1:
                    last_multi[getq(qubit)] = len(w)
                w.append(node)

    cache = {}

    def labels(op_node, k):
        if op_node.is_standard_gate():
            key = (op_node.name, k)
            label = cache.get(key)
            if label is None:
                label = cache[key] = _WIRE_PAULIS.get(op_node.name, "-" * k)
            return label
        return _wire_paulis(op_node)

    def next_on(node, a, pa):
        nodes = wires[a]
        return nodes[pa + 1] if pa + 1 < len(nodes) else None

    # Units, numbered as in the guard.  ``ukey``: a pair key for two-qubit units, -2 for
    # single-qubit units, -1 otherwise.
    unit_of = [-1] * n
    first = []
    groups = {}
    ulab = []
    ukey = []
    swaps = []
    for node, op_node in enumerate(order):
        if unit_of[node] >= 0:
            continue
        k = nq[node]
        label = labels(op_node, k)
        unit = len(first)
        if label == "ZX" and op_node.name == "cx" and op_node.is_standard_gate():
            a, b = q0[node], q1[node]
            middle = next_on(node, b, p1[node])
            if (
                middle is not None
                and labels(order[middle], nq[middle]) == "Z"
                and (last := next_on(middle, b, p0[middle])) is not None
                and order[last].name == "cx"
                and order[last].is_standard_gate()
                and q0[last] == a
                and q1[last] == b
                and nq[last] == 2
                and next_on(node, a, p0[node]) == last
            ):
                groups[unit] = [node, middle, last]
                unit_of[middle] = unit_of[last] = unit
                label = "ZZ"
        unit_of[node] = unit
        first.append(node)
        ulab.append(label)
        if k == 2:
            a, b = q0[node], q1[node]
            ukey.append(a * num_wires + b if a < b else b * num_wires + a)
            if op_node.name == "swap" and op_node.is_standard_gate():
                swaps.append(unit)
        else:
            ukey.append(-2 if k == 1 else -1)

    # Per wire: units in order (consecutive repeats merged), their label on the wire, and the
    # start of the run of equal labels each position belongs to.  ``pos0``/``pos1``: position of a
    # two-qubit unit on the wire of its first/second qubit.
    num_units = len(first)
    pos0 = [-1] * num_units
    pos1 = [-1] * num_units
    sequence = []
    seq_lab = []
    run_start = []
    for qubit in range(num_wires):
        seq = []
        labs = []
        starts = []
        prev = -1
        for node in wires[qubit]:
            unit = unit_of[node]
            if unit == prev:
                continue
            prev = unit
            i = len(seq)
            label = ulab[unit]
            head = first[unit]
            if ukey[unit] >= 0:
                if q0[head] == qubit:
                    pos0[unit] = i
                    label = label[0]
                else:
                    pos1[unit] = i
                    label = label[1]
            else:
                label = label[0] if label else "-"
            starts.append(starts[-1] if i and labs[-1] == label else i)
            seq.append(unit)
            labs.append(label)
        sequence.append(seq)
        seq_lab.append(labs)
        run_start.append(starts)

    def place(qubit, unit):
        return pos0[unit] if q0[first[unit]] == qubit else pos1[unit]

    def commutes_between(qubit, unit, swap):
        a, b = place(qubit, unit), place(qubit, swap)
        low, high = (a, b) if a < b else (b, a)
        labs = seq_lab[qubit]
        pauli = labs[a]
        if pauli == "-":
            return False
        return high - low <= 1 or (
            labs[high - 1] == pauli and run_start[qubit][high - 1] <= low + 1
        )

    def splits_same_pair_run(unit, step, qa, qb):
        neighbours = []
        for qubit in (qa, qb):
            seq = sequence[qubit]
            index = place(qubit, unit) + step
            while 0 <= index < len(seq) and ukey[seq[index]] == -2:
                index += step
            neighbours.append(seq[index] if 0 <= index < len(seq) else None)
        return (
            neighbours[0] is not None
            and neighbours[0] == neighbours[1]
            and ukey[neighbours[0]] == ukey[unit]
        )

    taken = set()
    pending = []
    for swap in swaps:
        key = ukey[swap]
        head = first[swap]
        qa, qb = q0[head], q1[head]
        touching = []
        for qubit, at in ((qa, pos0[swap]), (qb, pos1[swap])):
            seq = sequence[qubit]
            for index in (at - 1, at + 1):
                if (
                    0 <= index < len(seq)
                    and ukey[unit := seq[index]] == key
                    and abs(place(qa, unit) - pos0[swap]) == 1
                    and abs(place(qb, unit) - pos1[swap]) == 1
                ):
                    touching.append(unit)
        if touching:
            taken.update(touching)
        else:
            pending.append(swap)
    moves = []
    for swap in pending:
        head = first[swap]
        qa, qb = q0[head], q1[head]
        if last_multi[qa] <= p0[head] or last_multi[qb] <= p1[head]:
            continue
        key = ukey[swap]
        seq = sequence[qa]
        labs = seq_lab[qa]
        size = len(seq)
        for step in (1, -1):
            index = pos0[swap] + step
            while 0 <= index < size and ukey[seq[index]] != key and labs[index] != "-":
                index += step
            if not 0 <= index < size:
                continue
            unit = seq[index]
            if (
                ukey[unit] == key
                and unit not in taken
                and commutes_between(qa, unit, swap)
                and commutes_between(qb, unit, swap)
                and not splits_same_pair_run(unit, step, qa, qb)
            ):
                taken.add(unit)
                moves.append((unit, swap, step))
                break
    getall = lambda node: [getq(qubit) for qubit in order[node].qargs]
    return order, _Members(first, groups, q0, q1, getall), moves


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
