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
# Author: Foad Abo Dahood (@masterFoad)

"""Resynthesize pairs of two-qubit blocks on the same qubits across a commuting gap.

``ConsolidateBlocks`` and ``TwoQubitPeepholeOptimization`` collect two-qubit gates on the same
pair of qubits into one block only when nothing but single-qubit gates separates them.  After
routing, two blocks ``B`` and ``P`` on qubits ``(a, b)`` are often separated by a gap of gates
that act on ``a`` or ``b`` and on another qubit.  When every gap gate commutes with one Pauli
``s_a`` on ``a`` and one Pauli ``s_b`` on ``b``, the Clifford gate ``g`` that is ``CZ`` written in
the ``(s_a, s_b)`` basis commutes with the whole gap.  ``g`` is ``CZ`` itself for ``(Z, Z)`` and
``CX`` for ``(Z, X)``.  In circuit order::

    B; gap; P    ==    (B; g); gap; (g^-1; P)

The identity is exact.  The pass decides commutation numerically: a commutator counts as zero
when its largest entry is at most 1e-12.  Of the gates ``g`` that commute with the gap, the pass
takes the one that lowers the number of CX gates the two blocks need the most.  That number is
``TwoQubitBasisDecomposer.num_basis_gates`` with ``cx`` as the basis gate.  The pass writes
``B; g`` and ``g^-1; P`` back as ``UnitaryGate`` instructions for ``UnitarySynthesis``.

The simplest case is a ``swap(a, b)`` followed later by a ``cx(a, b)``, with ``g = cx(a, b)``.
The ``swap`` block becomes ``swap(a, b); cx(a, b)``, which needs two CX gates instead of three.
The ``cx`` block becomes ``cx(a, b); cx(a, b)``, which is the identity, instead of one CX.

The pass has three steps.  It collects the blocks that ``ConsolidateBlocks`` collects
(``DAGCircuit.collect_2q_runs``), sweeps them once in topological order, and writes the changed
blocks back.  In the sweep each block looks forward along both of its wires (the walk) for the
next block on the same pair.  Single-qubit gates next to either block join that block.  A block
is rewritten again until no rewrite lowers the count.  A rewrite needs a strictly lower count, so
the number of CX gates the two rewritten blocks need never increases.  Blocks without a rewrite
are left as they are.

The pass ranks a gate ``g`` by ``num_basis_gates``.  It accepts ``g`` only after
``TwoQubitBasisDecomposer`` has synthesized both new blocks with exactly that many CX gates and
reproduced each to an absolute tolerance of 1e-10.  The count alone can be lower than what the
synthesis needs; see https://github.com/Qiskit/qiskit/issues/17443.  The pass does not use
``approximation_degree``.

The count is the number of two-qubit gates ``UnitarySynthesis`` needs when every two-qubit basis
gate of the target is locally equivalent to a CX (``cx``, ``cz``, ``ecr``).  For any other target
the pass does nothing.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from qiskit.circuit.library import CXGate, CZGate, HGate, SGate, SwapGate, UnitaryGate
from qiskit.circuit.library import XGate, YGate, ZGate
from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.target import Target

# A commutator whose largest entry is below this is zero: four orders above the rounding of a
# product of 4x4 unitaries, far below any rotation angle a circuit carries on purpose.
_COMMUTE_ATOL = 1e-12
# Units a walk passes on one wire before it gives up.  It bounds the pass at
# ``blocks * _MAX_GAP`` steps; a longer gap is left alone.
_MAX_GAP = 400
# The Paulis in the order the gates ``g`` are tried; among equal costs the first one stays.
_LETTERS = "ZXY"
# The two-qubit basis gates that are locally equivalent to a CX.  With only these in the
# target, a block costs as many basis gates as CX gates, and the CX decomposer used here and the
# decomposer ``UnitarySynthesis`` picks for the target agree on that number.
_CX_EQUIVALENT = frozenset({"cx", "cz", "ecr"})
# Largest entry of (synthesized block - block) still read as an exact synthesis.  A block that
# needs the counted number of CX gates gives about 1e-15; a block that needs more, synthesized
# with fewer, gives 1e-7 or more.
_SYNTHESIS_ATOL = 1e-10

_PAULIS = dict(zip(_LETTERS, (ZGate().to_matrix(), XGate().to_matrix(), YGate().to_matrix())))
_ID = np.eye(2, dtype=complex)
_SWAP = SwapGate().to_matrix()
# ``W`` with ``W Z W^-1`` equal to the Pauli.
_TO_LETTER = {"Z": _ID, "X": HGate().to_matrix(), "Y": SGate().to_matrix() @ HGate().to_matrix()}
# The Pauli on qubit 0 (the low bit) and on qubit 1 of a two-qubit matrix.
_PAIR_PAULIS = [
    {letter: np.kron(_ID, pauli) for letter, pauli in _PAULIS.items()},
    {letter: np.kron(pauli, _ID) for letter, pauli in _PAULIS.items()},
]
# Every subset of the Paulis, in trial order, and the intersection of any two of them.
_SUBSETS = ["".join(c) for size in range(len(_LETTERS) + 1) for c in combinations(_LETTERS, size)]
_COMMON = {(s, t): "".join(x for x in s if x in t) for s in _SUBSETS for t in _SUBSETS}


def _frame(letter_a: str, letter_b: str) -> np.ndarray:
    """``CZ`` in the basis where ``letter_a`` on qubit 0 and ``letter_b`` on qubit 1 are ``Z``."""
    basis = np.kron(_TO_LETTER[letter_b], _TO_LETTER[letter_a])
    return basis @ CZGate().to_matrix() @ basis.conj().T


# Each gate ``g`` with its inverse, by pair of Paulis.
_FRAMES = {
    (la, lb): (_frame(la, lb), _frame(la, lb).conj().T) for la in _LETTERS for lb in _LETTERS
}
_DECOMPOSER = TwoQubitBasisDecomposer(CXGate())
_CX_COUNT = _DECOMPOSER.num_basis_gates


class _Unit:
    """A two-qubit block, a single-qubit gate, or an operation without a matrix (``matrix`` is
    None), which ends every walk.  ``at`` is the unit's position on each of its wires."""

    __slots__ = ("absorbed", "at", "changed", "cost", "matrix", "nodes", "wires")

    def __init__(self, wires: tuple[int, ...], matrix: np.ndarray | None, nodes: list[DAGOpNode]):
        self.wires, self.matrix, self.nodes = wires, matrix, nodes
        self.at, self.cost, self.absorbed, self.changed = (), None, False, False


class _ResynthesizeBlockPairs(TransformationPass):
    """Lower the two-qubit gate count of pairs of blocks on the same qubits across a commuting gap.

    The pass works on a physical circuit and needs ``UnitarySynthesis`` after it: it leaves
    ``UnitaryGate`` instructions on the two qubits of every block it rewrites.
    """

    def __init__(self, target: Target | None = None):
        """
        Args:
            target: the target of the circuit.  The pass leaves the circuit alone when the target
                has a two-qubit gate other than ``cx``, ``cz`` and ``ecr``; without a target it
                assumes one of these.
        """
        super().__init__()
        self._enabled = target is None or all(
            name in _CX_EQUIVALENT
            for name in target.operation_names
            if getattr(target.operation_from_name(name), "num_qubits", None) == 2
        )

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        if not self._enabled:
            return dag
        # Routed circuits repeat few matrices, so every numerical answer is kept by matrix.
        self._paulis_of, self._cost_of, self._frames_of, self._synthesis_of = {}, {}, {}, {}
        blocks, sequences = _collect_units(dag)
        for block in blocks:
            while self._rewrite(block, sequences):
                pass
        _write_back(dag, blocks)
        return dag

    def _rewrite(self, block: _Unit, sequences: list[list[_Unit]]) -> bool:
        """Resynthesize ``block`` and the next block on its qubits; False when nothing is gained."""
        found = self._find_partner(block, sequences)
        if found is None:
            return False
        partner, letters_a, letters_b, lead, trail = found
        first = _with_gates(block, lead, after=True)
        second = _with_gates(partner, trail, after=False)
        for unit in (block, partner):
            if unit.cost is None:
                unit.cost = self._cost(unit.matrix)
        best = self._best_frame(first, second, letters_a, letters_b, block.cost + partner.cost)
        if best is None:
            return False
        (block.matrix, partner.matrix), (block.cost, partner.cost) = best
        block.changed = partner.changed = True
        for gate in lead + trail:
            gate.absorbed = True
        # The absorbed gates sit directly after the block and directly before the partner.
        block.nodes = block.nodes + [node for gate in lead for node in gate.nodes]
        partner.nodes = [node for gate in trail for node in gate.nodes] + partner.nodes
        return True

    def _find_partner(self, block: _Unit, sequences: list[list[_Unit]]):
        """The next block on the qubits of ``block`` when the gap up to it commutes with a Pauli
        on each wire: ``(partner, Paulis on wire a, Paulis on wire b, lead, trail)``.

        ``lead`` are the single-qubit gates directly after ``block`` and ``trail`` those directly
        before the partner; each joins its block.  Everything in between is the gap.
        """
        partner, letters, lead, trail = None, [], [], []
        for wire, start in zip(block.wires, block.at):
            common, pending, in_gap, found = _LETTERS, [], False, None
            sequence = sequences[wire]
            for position in range(start + 1, min(len(sequence), start + 1 + _MAX_GAP)):
                unit = sequence[position]
                if unit.absorbed:
                    continue
                if unit.matrix is None:
                    return None
                if len(unit.wires) == 1:
                    pending.append(unit)
                    continue
                if unit.wires == block.wires:
                    found = unit
                    break
                # A block on another pair: the single-qubit gates since the last one are in the gap.
                for gate in pending if in_gap else ():
                    common = _COMMON[common, self._paulis(gate, wire)]
                if not in_gap:
                    lead.extend(pending)
                pending, in_gap = [], True
                common = _COMMON[common, self._paulis(unit, wire)]
                if not common:
                    return None
            if found is None or (partner is not None and found is not partner):
                return None
            partner = found
            (trail if in_gap else lead).extend(pending)
            letters.append(common)
        return partner, letters[0], letters[1], lead, trail

    def _best_frame(self, first, second, letters_a: str, letters_b: str, cost: int):
        """The cheapest ``((first g, g^-1 second), their costs)`` below ``cost`` whose costs the
        synthesis confirms, or None."""
        key = (first.tobytes(), second.tobytes(), letters_a, letters_b, cost)
        if key not in self._frames_of:
            candidates = []
            for letter_a in letters_a:
                for letter_b in letters_b:
                    frame, inverse = _FRAMES[letter_a, letter_b]
                    matrices = frame @ first, second @ inverse
                    costs = self._cost(matrices[0]), self._cost(matrices[1])
                    if sum(costs) < cost:
                        candidates.append((matrices, costs))
            # Cheapest first, the trial order among equals; only the winner is synthesized.
            candidates.sort(key=lambda candidate: sum(candidate[1]))
            confirmed = (c for c in candidates if all(map(self._synthesis_confirms, *c)))
            self._frames_of[key] = next(confirmed, None)
        return self._frames_of[key]

    def _synthesis_confirms(self, matrix: np.ndarray, cost: int) -> bool:
        """Whether the synthesis of ``matrix`` uses ``cost`` CX gates and reproduces it."""
        key = matrix.tobytes()
        if key not in self._synthesis_of:
            circuit = _DECOMPOSER(matrix)
            synthesised = np.exp(1j * circuit.global_phase) * np.eye(4, dtype=complex)
            for instruction in circuit.data:
                qargs = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
                gate = _on_pair(instruction.operation.to_matrix(), qargs, (0, 1))
                synthesised = gate @ synthesised
            exact = np.abs(synthesised - matrix).max() <= _SYNTHESIS_ATOL
            self._synthesis_of[key] = (circuit.count_ops().get("cx", 0), bool(exact))
        return self._synthesis_of[key] == (cost, True)

    def _paulis(self, unit: _Unit, wire: int) -> str:
        """The Paulis ``unit`` commutes with on ``wire``."""
        index = unit.wires.index(wire)
        key = (unit.matrix.tobytes(), index)
        if key not in self._paulis_of:
            probes = _PAULIS if len(unit.wires) == 1 else _PAIR_PAULIS[index]
            self._paulis_of[key] = "".join(
                letter
                for letter, pauli in probes.items()
                if np.abs(unit.matrix @ pauli - pauli @ unit.matrix).max() <= _COMMUTE_ATOL
            )
        return self._paulis_of[key]

    def _cost(self, matrix: np.ndarray) -> int:
        """The number of CX gates ``matrix`` needs, from ``num_basis_gates`` with a ``cx`` basis."""
        key = matrix.tobytes()
        if key not in self._cost_of:
            self._cost_of[key] = _CX_COUNT(matrix)
        return self._cost_of[key]


def _collect_units(dag: DAGCircuit) -> tuple[list[_Unit], list[list[_Unit]]]:
    """The two-qubit blocks in topological order and, per wire, the sequence of units."""
    wire_of = {qubit: index for index, qubit in enumerate(dag.qubits)}
    block_of = {}
    for run in dag.collect_2q_runs():
        two_qubit = [index for index, node in enumerate(run) if len(node.qargs) == 2]
        if two_qubit:
            # The single-qubit gates before the first and after the last two-qubit gate of a run
            # stay outside the block: they can join either neighbour.
            inside = run[two_qubit[0] : two_qubit[-1] + 1]
            wires = tuple(sorted(wire_of[qubit] for qubit in inside[0].qargs))
            block = _Unit(wires, None, inside)
            for node in inside:
                block_of[node] = block
    blocks, opaque, sequences = [], set(), [[] for _ in dag.qubits]

    def place(unit):
        unit.at = tuple(len(sequences[wire]) for wire in unit.wires)
        for wire in unit.wires:
            sequences[wire].append(unit)

    for node in dag.topological_op_nodes():
        block = block_of.get(node)
        qargs = tuple(wire_of[qubit] for qubit in node.qargs)
        if block is None:
            place(_Unit(qargs, _matrix(node) if len(qargs) == 1 else None, [node]))
            continue
        if node == block.nodes[0]:
            blocks.append(block)
            place(block)
        if id(block) in opaque:
            continue
        matrix = _matrix(node)
        if matrix is None:
            opaque.add(id(block))
            block.matrix = None
        else:
            matrix = _on_pair(matrix, qargs, block.wires)
            block.matrix = matrix if block.matrix is None else matrix @ block.matrix
    return [block for block in blocks if id(block) not in opaque], sequences


def _write_back(dag: DAGCircuit, blocks: list[_Unit]):
    """Replace the nodes of every changed block by one ``UnitaryGate``."""
    for block in blocks:
        if block.changed:
            wire_pos_map = {dag.qubits[wire]: index for index, wire in enumerate(block.wires)}
            gate = UnitaryGate(block.matrix, check_input=False)
            dag.replace_block_with_op(block.nodes, gate, wire_pos_map, cycle_check=False)


def _matrix(node: DAGOpNode) -> np.ndarray | None:
    """The matrix of a gate, through its definition when it has no ``to_matrix``; None for a
    gate without one and for everything that is not a gate (``delay``, ``barrier``, ...)."""
    matrix = node.matrix  # only gates have one
    if matrix is not None:
        return np.asarray(matrix)
    if node.is_parameterized() or not isinstance(node.op, Gate):
        return None
    try:
        return Operator(node.op).data
    except QiskitError:
        return None


def _on_pair(matrix: np.ndarray, qargs: tuple[int, ...], wires: tuple[int, int]) -> np.ndarray:
    """The matrix of a gate on ``qargs`` as a matrix on ``wires`` (qubit 0 is ``wires[0]``)."""
    if len(qargs) == 2:
        return matrix if qargs[0] == wires[0] else _SWAP @ matrix @ _SWAP
    lifted = np.zeros((4, 4), dtype=complex)
    if qargs[0] == wires[0]:  # identity on qubit 1: the two diagonal 2x2 blocks
        lifted[:2, :2] = lifted[2:, 2:] = matrix
    else:  # identity on qubit 0: the entries of equal low bit
        lifted[::2, ::2] = lifted[1::2, 1::2] = matrix
    return lifted


def _with_gates(block: _Unit, gates: list[_Unit], after: bool) -> np.ndarray:
    """The matrix of ``block`` followed (``after``) or preceded by the single-qubit ``gates``,
    which are in circuit order on each wire."""
    matrix = block.matrix
    for gate in gates if after else reversed(gates):
        lifted = _on_pair(gate.matrix, gate.wires, block.wires)
        matrix = lifted @ matrix if after else matrix @ lifted
    return matrix
