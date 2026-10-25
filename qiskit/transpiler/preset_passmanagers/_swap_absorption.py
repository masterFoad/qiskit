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

import gc
from typing import NamedTuple

import numpy as np

from qiskit.circuit import Qubit
from qiskit.circuit.commutation_library import SessionCommutationChecker
from qiskit.circuit.library import CZGate, UnitaryGate, XGate, YGate, ZGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.preset_passmanagers._swap_cost import _SWAP_MAT, _matrix, two_qubit_cost
from qiskit.transpiler.preset_passmanagers import _swap_sandwich

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
# CRZ, CRX, RZZ, RXX, RYY and RZX are looked up under their own names, and the raw lookup
# ignores their parameters. Their labels hold at every angle, because each gate is a function of
# the local Paulis in its label: with c = cos(theta / 2) and s = sin(theta / 2), CRZ and CRX are
# (I + Z) / 2 (x) I + (I - Z) / 2 (x) (c I - i s P) for P = Z or X on the target, and RZZ, RXX,
# RYY and RZX are c I - i s P for P = ZZ, XX, YY or ZX (Z on the first qubit).
# test_standard_wire_labels pins the labels at a symbolic angle and at 0, 1e-6 and 0.37.
_LOOKUP_REPRESENTATIVES = {
    "rx": XGate(),
    "ry": YGate(),
    "rz": ZGate(),
    "p": ZGate(),
    "u1": ZGate(),
    "cp": CZGate(),
    "cu1": CZGate(),
}
# Unit keys of units that are not two-qubit units; a two-qubit unit's pair key is non-negative.
_SINGLE_QUBIT_UNIT = -2
_UNSUPPORTED_UNIT = -1


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

    A move is only made when it is predicted to save at least one two-qubit gate: the summed cost
    (see :mod:`._swap_cost`) of the same-pair blocks around the ``swap`` and the moved unit before
    the move, minus their cost after it.  A move whose blocks have an unknown cost, or are longer
    than 256 units on a wire, is not made: a move that does not lower the two-qubit count can
    still add single-qubit gates and depth to the resynthesized block.

    Args:
        target: used only to cost a continuous ``rzz`` as one two-qubit gate when the target
            supports it.
        sandwich: after the moves, insert exact cancelling pairs ``g(P, Q)``, ``g(P, Q)`` of
            CZ-class gates between a run that holds a ``swap`` and the next run on the same pair
            (the "SWAPs sandwiching a commuting set" case of [1], pp. 7-8, applied after routing;
            see :mod:`._swap_sandwich`), where the cost predicts a saving of at least 1 per qubit
            pair.  Off by default.
    """

    def __init__(self, target=None, sandwich: bool = False):
        super().__init__()
        self.fractional = target is not None and "rzz" in target.operation_names
        self.sandwich = bool(sandwich)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        if "swap" not in dag.count_ops(recurse=False):
            return dag
        # The analysis allocates many short-lived containers, and collections triggered by them
        # dominated the runtime on large circuits.  The previous collector state is restored.
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            order, units, moves = _find_moves(dag, gainful=True, fractional=self.fractional)
            if moves:
                dag = _apply_moves(dag, order, units, moves)
            if self.sandwich:
                dag = _swap_sandwich.insert_swap_sandwiches(dag, fractional=self.fractional)
            return dag
        finally:
            if gc_was_enabled:
                gc.enable()


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


def _find_moves(dag: DAGCircuit, gainful: bool = False, fractional: bool = False):
    """The exact moves of ``dag``, or with ``gainful`` only those with a predicted saving."""
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

    def next_on(node, a, pa):
        nodes = wires[a]
        return nodes[pa + 1] if pa + 1 < len(nodes) else None

    # Units, numbered as in the guard.  ``ukey``: a pair key for two-qubit units,
    # _SINGLE_QUBIT_UNIT for single-qubit units, _UNSUPPORTED_UNIT otherwise.
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
        label = _wire_paulis(op_node, cache)
        unit = len(first)
        if label == "ZX" and op_node.name == "cx" and op_node.is_standard_gate():
            a, b = q0[node], q1[node]
            middle = next_on(node, b, p1[node])
            if (
                middle is not None
                and _wire_paulis(order[middle], cache) == "Z"
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
            ukey.append(_SINGLE_QUBIT_UNIT if k == 1 else _UNSUPPORTED_UNIT)

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
            while 0 <= index < len(seq) and ukey[seq[index]] == _SINGLE_QUBIT_UNIT:
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
    units = _Members(first, groups, q0, q1, getall)
    if gainful:
        # A dropped move's unit stays taken, so no other SWAP moves it instead.
        context = _GainContext(sequence, ukey, pos0, pos1, first, q0, q1, nq)
        reuse = _GainReuse() if moves else None
        moves = [
            move
            for move in moves
            if (gain := _move_gain(order, units, context, move, fractional, reuse)) is not None
            and gain >= 1
        ]
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


# Labels by (name, width) for the whole process. The session commutation table is built once
# from the fixed standard library, and the raw lookup reads only names and the placement.
_WIRE_PAULI_CACHE: dict[tuple[str, int], str] = {}


def _standard_wire_paulis(op, width: int) -> str:
    """Derive local labels for a verified, allowlisted standard gate using only table rules.

    Two operations commuting with the same local Pauli on every shared wire are block
    diagonal in those Pauli eigenbases and hence commute exactly. The admitted families
    are functions of local Paulis, diagonal controlled gates, or Pauli-product rotations.
    This uses the repaired raw lookup API, never the checker's approximate acceptance paths.
    Each ``(op.name, width)`` is looked up once, then read from ``_WIRE_PAULI_CACHE``.
    """
    key = (op.name, width)
    if (label := _WIRE_PAULI_CACHE.get(key)) is not None:
        return label
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
    label = _WIRE_PAULI_CACHE[key] = "".join(labels)
    return label


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


# A move whose slice holds more than _MAX_SLICE units on a wire is not costed.  Widening stops at
# the cap, so a slice takes at most _MAX_SLICE steps per wire, however long its run of soft units.
_MAX_SLICE = 256
_NOT_CACHED = object()


class _GainContext(NamedTuple):
    """The :func:`_find_moves` tables that :func:`_move_gain` reads, shared, not copied."""

    sequence: list  # Per wire: its units in order.
    ukey: list  # Per unit: pair key, _SINGLE_QUBIT_UNIT or _UNSUPPORTED_UNIT.
    pos0: list  # Per two-qubit unit: position on the wire of its first qubit.
    pos1: list  # Per two-qubit unit: position on the wire of its second qubit.
    first: list  # Per unit: its first node.
    q0: list  # Per node: its first qubit.
    q1: list  # Per node: its second qubit, or -1.
    nq: list  # Per node: its number of qubits.


class _GainReuse:
    """Reuse one-qubit lifts and block costs on byte-identical inputs, across the moves of one
    gainful :func:`_find_moves` call."""

    __slots__ = ("costs", "identity", "lifted")

    def __init__(self):
        self.identity = np.eye(2)
        self.lifted = {}
        self.costs = {}

    def lift(self, matrix, first_wire):
        """Use the original kron operation on misses, preserving its dtype and values."""
        key = (first_wire, matrix.shape, matrix.dtype.str, matrix.tobytes())
        value = self.lifted.get(key)
        if value is None:
            value = np.kron(self.identity, matrix) if first_wire else np.kron(matrix, self.identity)
            self.lifted[key] = value
        return value

    def cost(self, matrix, fractional):
        """Cache unknown costs too: an unchanged unknown block must still reject a move."""
        key = (fractional, matrix.shape, matrix.dtype.str, matrix.tobytes())
        value = self.costs.get(key, _NOT_CACHED)
        if value is _NOT_CACHED:
            value = two_qubit_cost(matrix, fractional)
            self.costs[key] = value
        return value


def _move_gain(order, units, context, move, fractional, reuse=None):
    """Predicted two-qubit saving of ``move``.

    On each wire of the pair, the slice from the ``swap`` to the moved unit is widened over
    same-pair units and parameter-free single-qubit gates up to the next other operation (no
    same-pair block crosses such an operation).  The slices are cut into blocks (same-pair units
    joined by single-qubit gates only, on both wires) before and after the move, and the gain is
    the summed block cost before minus after.  ``None`` when a block cost is unknown or a slice is
    longer than ``_MAX_SLICE`` units.
    """
    unit, swap, step = move
    head = context.first[swap]
    qa, qb = context.q0[head], context.q1[head]
    key = context.ukey[swap]
    cache = {}
    if reuse is None:
        reuse = _GainReuse()

    def matrix(u):
        if u not in cache:
            cache[u] = _unit_matrix(order, units, context, u, key, qa, reuse)
        return cache[u]

    slices = []
    for qubit in (qa, qb):
        wire = _widen_slice(context, qubit, swap, unit, key, matrix)
        if wire is None:
            return None
        slices.append(wire)
    before = _blocks(slices, context.ukey, key, matrix, reuse)
    after = _blocks(_moved_slices(slices, unit, swap, step), context.ukey, key, matrix, reuse)
    if before is None or after is None:
        return None
    return _cost_difference(before, after, reuse, fractional)


def _place(context, qubit, u):
    """Position of unit ``u`` on the wire of ``qubit``."""
    return context.pos0[u] if context.q0[context.first[u]] == qubit else context.pos1[u]


def _unit_matrix(order, units, context, u, key, qa, reuse):
    """4x4 matrix of a pair unit on (qa, qb) (qa least significant), or 2x2 of a 1q unit, or
    ``None`` when a member has no matrix or ``u`` is neither."""
    if context.ukey[u] == key:
        mat = np.eye(4, dtype=complex)
        for member in units[u][0]:
            m = _matrix(order[member])
            if m is None:
                return None
            if context.nq[member] == 1:
                m = reuse.lift(m, context.q0[member] == qa)
            elif context.q0[member] != qa:
                m = _SWAP_MAT @ m @ _SWAP_MAT
            mat = m @ mat
        return mat
    if context.ukey[u] == _SINGLE_QUBIT_UNIT:
        members = units[u][0]
        if len(members) == 1:
            return _matrix(order[members[0]])
    return None


def _soft(u, ukey, key, matrix):
    """Whether a slice is widened over unit ``u``: a same-pair unit or a 1q unit with a matrix."""
    return ukey[u] == key or (ukey[u] == _SINGLE_QUBIT_UNIT and matrix(u) is not None)


def _widen_slice(context, qubit, swap, unit, key, matrix):
    """The units on ``qubit`` from ``swap`` to ``unit``, widened over soft units on both sides,
    or ``None`` when the slice is longer than ``_MAX_SLICE`` units."""
    seq = context.sequence[qubit]
    ukey = context.ukey
    low, high = sorted((_place(context, qubit, swap), _place(context, qubit, unit)))
    while low > 0 and high - low < _MAX_SLICE and _soft(seq[low - 1], ukey, key, matrix):
        low -= 1
    while (
        high + 1 < len(seq) and high - low < _MAX_SLICE and _soft(seq[high + 1], ukey, key, matrix)
    ):
        high += 1
    if high - low >= _MAX_SLICE:
        return None
    return seq[low : high + 1]


def _moved_slices(slices, unit, swap, step):
    """Fresh copies of ``slices`` with ``unit`` placed right next to ``swap``."""
    moved = []
    for wire in slices:
        wire = [u for u in wire if u != unit]
        at = wire.index(swap)
        wire.insert(at + 1 if step > 0 else at, unit)
        moved.append(wire)
    return moved


def _blocks(slices, ukey, key, matrix, reuse):
    """Matrices of the same-pair blocks of the two ``slices``, or ``None`` if one is unknown."""
    wire_a, wire_b = slices
    index_b = {u: i for i, u in enumerate(wire_b)}
    index_a = {u: i for i, u in enumerate(wire_a)}
    pair = [u for u in wire_a if ukey[u] == key and u in index_b]
    out = []
    current = None
    for k, u in enumerate(pair):
        unit_matrix = matrix(u)
        if unit_matrix is None:
            return None
        if current is not None:
            prev = pair[k - 1]
            gap_a = wire_a[index_a[prev] + 1 : index_a[u]]
            gap_b = wire_b[index_b[prev] + 1 : index_b[u]]
            if all(ukey[g] == _SINGLE_QUBIT_UNIT and matrix(g) is not None for g in gap_a + gap_b):
                for g in gap_a:
                    current = reuse.lift(matrix(g), True) @ current
                for g in gap_b:
                    current = reuse.lift(matrix(g), False) @ current
                current = unit_matrix @ current
                continue
            out.append(current)
        current = unit_matrix
    if current is not None:
        out.append(current)
    return out


def _cost_difference(before, after, reuse, fractional):
    """Summed cost of the ``before`` blocks minus that of the ``after`` blocks, or ``None`` if a
    cost is unknown."""
    total = 0
    for mats, sign in ((before, 1), (after, -1)):
        for mat in mats:
            value = reuse.cost(mat, fractional)
            if value is None:
                return None
            total += sign * value
    return total
