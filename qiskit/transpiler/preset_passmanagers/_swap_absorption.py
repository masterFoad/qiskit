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

"""Absorb a commuting Clifford into ``swap`` gates to save CX gates.

After routing, a ``swap`` on qubits ``(a, b)`` often has another two-qubit gate on the same pair
of qubits nearby.  This text takes that gate to come after the ``swap`` and calls it the later
gate.  The gates in between each act on one of the two qubits and possibly on other qubits, so
two-qubit resynthesis does not see the ``swap`` and the later gate together.  The pass acts on
every ``swap`` in the circuit, whether routing inserted it or not.

When each of the gates in between commutes with one Pauli on each of the two qubits, there is a
two-qubit Clifford ``g`` on ``(a, b)`` that commutes with the gates in between.  ``g`` is
``cx(c, t)`` when they commute with ``Z`` on ``c`` and with ``X`` on ``t``.  Otherwise ``g`` is
``cz`` when they commute with ``Z`` on both qubits.  A qubit that none of them acts on fits any
Pauli.  ``g`` is self-inverse, so ``swap; gates in between; later gate`` equals
``(swap; g); gates in between; (g; later gate)`` in circuit order, and both blocks in parentheses
need fewer CX gates [1].  When the other gate comes before the ``swap``, both blocks are
reversed.

====================  ======  =================================  ==================================
later gate            ``g``   ``swap`` becomes                   later gate becomes
====================  ======  =================================  ==================================
``cx(c, t)``          ``cx``  ``cx(c, t); cx(t, c)``             removed
``swap``              ``cx``  ``cx(c, t); cx(t, c)``             ``cx(t, c); cx(c, t)``
``swap``              ``cz``  ``h(a); cx(a, b); cx(b, a); h(b)`` the same block on its qubits
``cz``                ``cz``  ``h(a); cx(a, b); cx(b, a); h(b)`` removed
diagonal gate         none    three CX gates and phase gates     removed
====================  ======  =================================  ==================================

The last row is a move, not an instance of the identity above.  It applies to a diagonal standard
gate with bound parameters (``cp``, ``cu1``, ``crz``, ``cs``, ``csdg``, ``rzz``) and to a
``cx(a, b); D(b); cx(a, b)`` group, where ``D`` is a diagonal single-qubit gate.  When the gates
in between commute with ``Z`` on both qubits, the pass moves the gate next to the ``swap`` and
merges the two into three CX gates and phase gates (the dressed SWAP of [2]).

Every rewrite is exact, global phase included.  The pass counts two-qubit gates as CX
equivalents: ``swap`` 3, ``cx``, ``cz`` and ``ecr`` 1, other two-qubit gates 2.  By this count
each rewrite removes exactly two.  Each new block counts at most as much as the gate it replaces,
so by this count no rewrite adds depth.  Counted per gate, with a ``swap`` as one gate, the depth
can grow until translation expands the ``swap`` gates.

Four conditions, C1 to C4, skip the rewrites that the optimization loop could undo or do better
itself with ``CommutativeCancellation`` or ``TwoQubitPeepholeOptimization``.  They are not related
to the ``check`` argument of :class:`._AbsorbIntoSwaps`.

* C1: neither the ``swap`` nor the later gate has another gate on the same pair of qubits with only
  single-qubit gates in between.  ``TwoQubitPeepholeOptimization`` would collect them into one
  block.  With ``peephole_in_loop``, two ``swap`` gates whose gates in between commute with ``Z``
  are also left alone in one gate pattern: beyond one of them, the next multi-qubit gate on one
  of its qubits is a third ``swap`` on the same pair, and the next multi-qubit gate beyond that
  one has the same name and qubits as the first multi-qubit gate in between.
* C2: removing the later gate does not make two gates on the same pair of qubits adjacent when
  either of them could cancel with a gate further away.
* C3: no gate that could cancel with an end of a new block is reachable from that end across
  single-qubit gates and two-qubit gates that commute with the gate at that end.  A gate can
  cancel with the end when its translation has a ``cx`` with the same control and target at the
  side that faces the end, or when both are ``cz``.
* C4: a diagonal gate is not moved when ``CommutativeCancellation`` would cancel the ``cx`` at the
  facing end of the ``swap`` with the diagonal gate's own ``cx``.  This happens when the two have
  the same control and target and no gate in between acts on the target qubit.

- [1] Liu, Li and Zhou, Not all SWAPs have the same cost: a case for optimization-aware qubit
  routing, `arXiv:2205.10596 <https://arxiv.org/abs/2205.10596>`_
- [2] Lao and Browne, 2QAN: a quantum compiler for 2-local qubit Hamiltonian simulation
  algorithms, `arXiv:2108.02099 <https://arxiv.org/abs/2108.02099>`_
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.circuit.commutation_library import SessionCommutationChecker
from qiskit.circuit.library import CXGate, CZGate, HGate, PhaseGate, SwapGate, XGate, YGate, ZGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.exceptions import QiskitError
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passmanager import PassManager

# Explicit allowlists, so that new commutation table entries do not enable new rewrites.
_TWO_QUBIT_GATES = frozenset("cz cp crz cu1 cs csdg rzz cx crx csx rzx rxx ryy".split())
_SINGLE_QUBIT_GATES = frozenset("rz z s sdg t tdg p u1 rx x sx sxdg ry y".split())
_SUPPORTED_GATES = _SINGLE_QUBIT_GATES | _TWO_QUBIT_GATES
_DIAGONAL_GATES = frozenset("cp cu1 crz cs csdg rzz".split())
# Gates whose translation has ``cx(q0, q1)`` at an end (``ecr``, ``cry``: at both ends).
# An entry here can only make C3 reject a rewrite.
_ENDS_WITH_CX = _DIAGONAL_GATES | {"cx", "swap", "ecr", "dcx", "cry"}
_PAULI_PROBES = (("X", XGate()), ("Y", YGate()), ("Z", ZGate()))
_ENTRY = SessionCommutationChecker.check_commutation_entries
# Parameter-free stand-ins for the table lookup; the Paulis found do not depend on the angle.
_LOOKUP_REPRESENTATIVES = {"rx": XGate(), "ry": YGate(), "rz": ZGate(), "p": ZGate(), "u1": ZGate()}
_LOOKUP_REPRESENTATIVES |= {"cp": CZGate(), "cu1": CZGate()}
_SWAP_MATRIX = SwapGate().to_matrix()
# No Pauli: the gate commutes with none on this qubit, or no gate in between acts on it.
_NO_PAULI = "-"
_SINGLE, _WIDE, _CZ = None, -1, ()  # ``pair``: single-qubit, wider unit; ``exposed``: a cz end
_C1, _C2, _C3, _C4 = "C1", "C2", "C3", "C4"  # stats keys
_NO_CLIFFORD, _PARTNER = "no_clifford", "partner"  # stats keys


def _cx(control: int, target: int):
    return CXGate(), (control, target)


# ``swap; cz`` on the block's qubits; ``cz`` and ``swap`` commute, so it is ``cz; swap`` too.
_CZ_BLOCK = [(HGate(), (0,)), _cx(0, 1), _cx(1, 0), (HGate(), (1,))], 0.0
# ``swap; cx(c, 1 - c)`` at (c, 1), ``cx(c, 1 - c); swap`` at (c, -1); ``[::s]`` reverses the pair
_CX_BLOCKS = {(c, s): ([_cx(c, 1 - c), _cx(1 - c, c)][::s], 0.0) for c in (0, 1) for s in (1, -1)}


class _AbsorbIntoSwaps(TransformationPass):
    """Absorb a commuting two-qubit Clifford into each ``swap`` and into the nearest two-qubit
    gate on the same pair of qubits.

    See the module docstring.  Only the top-level DAG is rewritten; control-flow bodies are not
    entered.  With ``move_diagonal=False`` diagonal gates are not moved; the preset pass managers
    use this on targets with ``rzz``.  When ``target`` or ``basis_gates`` has ``cz`` and no
    ``cx``, a translated ``swap`` ends with a ``cz``, and C3 takes this into account.

    ``check`` is a list of translation passes.  The pass runs them once, on all distinct new
    blocks together.  If any block does not become instructions that the target supports on its
    qubits, the pass returns the input DAG.  The output is either the full rewrite or the input.

    ``peephole_in_loop`` says that the optimization loop that follows runs
    :class:`.TwoQubitPeepholeOptimization` in every iteration, as the preset pass manager does
    at optimization level 3.  The loop then handles the gate pattern that C1 names for
    ``peephole_in_loop`` by itself: once the two ``swap`` gates meet, the loop cancels the two
    gates with the same name and qubits.  The pass leaves that pattern alone."""

    def __init__(
        self,
        *,
        move_diagonal=True,
        target=None,
        basis_gates=None,
        check=None,
        peephole_in_loop=False,
    ):
        super().__init__()
        if check is not None and target is None and basis_gates is None:
            raise TranspilerError("check needs a target or basis_gates")
        self.move_diagonal, self.target, self.basis_gates = move_diagonal, target, basis_gates
        self.peephole_in_loop = peephole_in_loop
        self.check = None if check is None else PassManager(check)
        names = target.operation_names if target is not None else basis_gates or ()
        self.cz_native = "cz" in names and "cx" not in names  # a swap translates to cz gates

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        if "swap" not in dag.count_ops(recurse=False):
            return dag
        rewrites = _rewrites(
            dag, self.move_diagonal, self.cz_native, Counter(), self.peephole_in_loop
        )
        if self.check is not None:
            to_check = {}  # each block once on each pair of physical qubits
            for swap, blocks, _ in rewrites:
                qubits = tuple(dag.find_bit(qubit).index for qubit in swap.qargs)
                for block in filter(None, blocks):
                    to_check[_key(block), qubits] = block, qubits
            if to_check and not self._supported(dag.num_qubits(), to_check.values()):
                return dag
        dags = {}
        for swap, (swap_block, partner_block), partner in rewrites:
            if partner_block is None:
                for node in partner:
                    dag.remove_op_node(node)
            else:
                block = _block_dag(partner_block, dags)
                wires = [block.qubits[swap.qargs.index(qubit)] for qubit in partner[0].qargs]
                dag.substitute_node_with_dag(partner[0], block, wires=wires)
            block = _block_dag(swap_block, dags)
            dag.substitute_node_with_dag(swap, block, wires=block.qubits)
        return dag

    def _supported(self, num_qubits: int, blocks) -> bool:
        """Whether the check turns the blocks into instructions supported on their qubits."""
        circuit = QuantumCircuit(num_qubits)  # barriers keep the blocks apart
        for (operations, _), qubits in blocks:
            for gate, indices in operations:
                circuit.append(gate, [qubits[index] for index in indices], copy=False)
            circuit.barrier(qubits)
        try:
            checked = self.check.run(circuit)
        except QiskitError:
            return False
        for instruction in checked.data:
            name, params = instruction.operation.name, instruction.operation.params
            qargs = tuple(checked.find_bit(qubit).index for qubit in instruction.qubits)
            if self.target is None:
                supported = name in self.basis_gates
            else:
                supported = self.target.instruction_supported(name, qargs, parameters=params)
            if name != "barrier" and not supported:
                return False
        return True


def _key(block):  # ``(phase, ((name, indices, *params), ...))`` of ``(operations, phase)``
    return block[1], tuple((g.name, i, *map(float, g.params)) for g, i in block[0])


def _block_dag(block, dags: dict) -> DAGCircuit:
    """The two-qubit :class:`.DAGCircuit` of ``block``, built once per distinct block."""
    key = _key(block)
    if key not in dags:
        circuit = dags[key] = DAGCircuit()
        circuit.add_qubits([Qubit(), Qubit()])
        circuit.global_phase = block[1]
        for gate, indices in block[0]:
            circuit.apply_operation_back(gate, [circuit.qubits[i] for i in indices], check=False)
    return dags[key]


def _rewrites(
    dag: DAGCircuit, move_diagonal: bool, cz_native: bool, stats: Counter, peephole_in_loop=False
) -> list:
    """Return the rewrites as ``(swap, (swap_block, partner_block), partner)``, where the
    partner is the other gate of the pair (the later gate of the module docstring): the partner's
    nodes (one, or a ``cx; D; cx`` group) are removed when ``partner_block`` is ``None``, and
    the blocks are ``(operations, global_phase)`` on the qargs of the ``swap``.  Everything is
    decided on the original DAG; each ``swap`` and each partner is in at most one rewrite.
    ``stats`` counts the applied rewrites and the rejections by condition; ``cz_native``: a
    translated ``swap`` ends with a ``cz``; ``peephole_in_loop``: skip the ``swap``-``swap``
    rewrites in the C1 gate pattern (module docstring)."""
    wire_of = {qubit: wire for wire, qubit in enumerate(dag.qubits)}
    width, count = len(wire_of), dag.num_ops()
    # Per wire and position: the unit there (a node, or a ``cx(a, b); D(b); cx(a, b)`` group under
    # its first node's topological index) and the Pauli it commutes with there.  Per unit: pair
    # (``_SINGLE``, ``_WIDE``, or one number per unordered pair of wires), wires and positions.
    wires, paulis = [[] for _ in range(width)], [[] for _ in range(width)]
    last_multi, pair = [-1] * width, [None] * count  # per wire: last multi-qubit unit's position
    first_wire, second_wire, first_position, second_position = ([None] * count for _ in range(4))
    swaps, labels_of = [], {}  # swap units; standard gate name -> Paulis per qubit
    groups, blocks = {}, {}  # first CX's index -> the three nodes' indices; blocks by value
    # The node list's tuples force a garbage collection on a large circuit; it finds these empty.
    order = list(dag.topological_op_nodes())

    def position(unit, wire):
        return first_position[unit] if first_wire[unit] == wire else second_position[unit]

    for unit, node in enumerate(order):
        qargs, name = node.qargs, node.name
        standard = node.is_standard_gate()
        label = labels_of.get(name) if standard else _NO_PAULI * len(qargs)
        if label is None:
            label = labels_of[name] = _wire_paulis(node)
        if len(qargs) != 2:  # single-qubit, or wider (a barrier, ``ccx``)
            pair[unit] = _SINGLE if len(qargs) == 1 else _WIDE
            for qubit, pauli in zip(qargs, label):
                wire = wire_of[qubit]
                wires[wire].append(unit)
                paulis[wire].append(pauli)
                if pair[unit] == _WIDE:
                    last_multi[wire] = len(wires[wire]) - 1
            continue
        a, b = wire_of[qargs[0]], wire_of[qargs[1]]
        # A standard ``cx`` (Paulis Z, X) after ``cx(a, b); D(b)``: a group commuting with Z on b.
        if name == "cx" and label == "ZX" and len(paulis[b]) > 1 and paulis[b][-1] == "Z":
            head, middle = wires[b][-2:]
            if (
                pair[middle] is _SINGLE
                and first_wire[head] == a
                and wires[a][-1] == head
                and order[head].name == "cx"
                and paulis[a][-1] == "Z"
                and paulis[b][-2] == "X"
            ):
                groups[head] = [head, wires[b].pop(), unit]
                paulis[b].pop()
                paulis[b][-1] = "Z"
                continue
        pair[unit] = a * width + b if a < b else b * width + a  # one number per unordered pair
        first_wire[unit], second_wire[unit] = a, b
        if name == "swap" and standard:
            swaps.append(unit)
        wires[a].append(unit)
        paulis[a].append(label[0])
        first_position[unit] = last_multi[a] = len(wires[a]) - 1
        wires[b].append(unit)
        paulis[b].append(label[1])
        second_position[unit] = last_multi[b] = len(wires[b]) - 1

    def scan(wire, at, same, step):
        """One pass over ``wire`` from position ``at`` in direction ``step``: the nearest unit
        that is not single-qubit (or ``None``), the nearest unit of pair ``same`` if every gate
        before it commutes with one Pauli on the wire (the walk, else ``None``), and that Pauli."""
        units, labels, size = wires[wire], paulis[wire], len(wires[wire])
        nearest = letter = None  # the Pauli all gates in between so far commute with, or _NO_PAULI
        at += step
        while 0 <= at < size:
            unit = units[at]
            if pair[unit] == same:
                hit = None if letter == _NO_PAULI else unit
                return (unit if nearest is None else nearest), hit, letter
            if letter != _NO_PAULI:
                letter = labels[at] if letter in (None, labels[at]) else _NO_PAULI
            if nearest is None and pair[unit] is not _SINGLE:
                nearest = unit
            if letter == _NO_PAULI and nearest is not None:
                break
            at += step
        return nearest, None, None

    def exposed(unit, step):
        """What the end of ``unit`` reached by a walk in direction ``step`` exposes (``step < 0``:
        the walk comes from later gates and meets the unit's last gate): a CX as its ``(control,
        target)``, a bare ``cz`` as ``_CZ``, nothing as ``None``."""
        if order[unit].name == "swap" and cz_native:
            return _CZ if step < 0 else None
        if order[unit].name in _ENDS_WITH_CX:  # ``dcx`` = ``cx(q0, q1); cx(q1, q0)``: reversed end
            ends = first_wire[unit], second_wire[unit]
            return ends[::-1] if order[unit].name == "dcx" and step < 0 else ends
        return _CZ if order[unit].name == "cz" else None

    def partner_free(unit, orientation, step):
        """C3: no gate that cancels against what ``unit`` exposes is reachable in direction
        ``step`` across single-qubit gates (the optimization loop may merge or move them) and
        two-qubit gates that commute with it gate-wise: ``Z`` on a CX's control and ``X`` on its
        target, ``Z`` on both wires of a ``cz``."""
        found = None
        control, target = orientation or (first_wire[unit], second_wire[unit])
        for wire, wanted in zip((control, target), "ZZ" if orientation == _CZ else "ZX"):
            units, labels = wires[wire], paulis[wire]
            at = position(unit, wire) + step
            while 0 <= at < len(units) and pair[units[at]] != pair[unit]:
                if pair[units[at]] is not _SINGLE and labels[at] != wanted:
                    return True
                at += step
            if not 0 <= at < len(units) or found not in (None, units[at]):
                return True
            found = units[at]
        return exposed(found, step) != orientation

    def no_new_adjacency(unit, swap):
        """C2: removing ``unit`` makes no two gates on the pair adjacent if either could cancel."""
        for wire in (first_wire[unit], second_wire[unit]):
            at, same = position(unit, wire), pair[unit]
            before, after = scan(wire, at, same, -1)[0], scan(wire, at, same, 1)[0]
            if before is None or after is None or swap in (before, after) or pair[before] == _WIDE:
                continue
            other = second_wire[before] if first_wire[before] == wire else first_wire[before]
            far = scan(other, position(before, other), same, 1)[0]  # the nearest multi-qubit unit
            if pair[before] != pair[after] or far != after:
                continue
            for end, step in ((before, -1), (after, 1)):
                orientation = exposed(end, -step)  # the end that faces the walk
                if orientation is not None and not partner_free(end, orientation, step):
                    return False
        return True

    def choose(swap, unit, step, letter_a, letter_b):
        """``(swap_block, partner_block)`` for ``swap`` and ``unit``, or the rejecting check."""
        a, b = first_wire[swap], second_wire[swap]
        far = scan(first_wire[unit], first_position[unit], pair[unit], step)[0]  # C1 on the partner
        if far is not None and pair[far] == pair[unit]:
            if far == scan(second_wire[unit], second_position[unit], pair[unit], step)[0]:
                return _C1
        name = order[unit].name if order[unit].is_standard_gate() else None
        orientation = first_wire[unit], second_wire[unit]
        fits_cz = letter_a in (None, "Z") and letter_b in (None, "Z")
        # ``g = cx`` by control wire: Z or nothing on the control, X or nothing on the target.
        fits_cx = {a: letter_a in (None, "Z") and letter_b in (None, "X")}
        fits_cx[b] = letter_b in (None, "Z") and letter_a in (None, "X")
        swap_end, unit_end = exposed(swap, step), exposed(unit, -step)  # the far ends
        swap_free = swap_end is None or partner_free(swap, swap_end, -step)
        unit_free = unit_end is None or partner_free(unit, unit_end, step)
        if name == "cx" and unit not in groups:
            if not fits_cx[orientation[0]]:
                return _NO_CLIFFORD
            if not (unit_free and (orientation == swap_end or swap_free)):
                return _C3
            if not no_new_adjacency(unit, swap):
                return _C2
            return _CX_BLOCKS[int(orientation[0] != a), step], None
        if name == "swap":
            if peephole_in_loop and "Z" in (letter_a, letter_b):  # C1 with ``peephole_in_loop``
                for u, s in ((swap, -step), (unit, step)):
                    for w in (first_wire[u], second_wire[u]):
                        n = scan(w, position(u, w), pair[unit], s)[0]  # a third swap on the pair
                        if n is None or pair[n] != pair[unit] or order[n].name != "swap":
                            continue
                        far = scan(w, position(n, w), pair[unit], s)[0]  # beyond the third swap
                        near = scan(w, position(u, w), pair[unit], -s)[
                            0
                        ]  # the first gate in between
                        if None not in (far, near) and order[far].qargs == order[near].qargs:
                            if order[far].name == order[near].name:  # the pair the loop cancels
                                return _C1
            for control, target in ((a, b), (b, a)):
                if fits_cx[control] and (swap_free or (control, target) == swap_end):
                    if unit_free or (control, target) == unit_end:
                        c = int(control != a)
                        return _CX_BLOCKS[c, step], _CX_BLOCKS[c, -step]
            if not (fits_cz or fits_cx[a] or fits_cx[b]):
                return _NO_CLIFFORD
            return (_CZ_BLOCK, _CZ_BLOCK) if fits_cz and swap_free and unit_free else _C3
        if not fits_cz:
            return _NO_CLIFFORD
        if name == "cz":
            if not (swap_free and unit_free):
                return _C3
            return (_CZ_BLOCK, None) if no_new_adjacency(unit, swap) else _C2
        if not move_diagonal or (name not in _DIAGONAL_GATES and unit not in groups):
            return _PARTNER
        nodes = [order[member] for member in groups.get(unit, [unit])]
        if any(member.is_parameterized() for member in nodes):
            return _PARTNER  # an unbound angle has no matrix; the Paulis do not depend on it
        # C4: the swap's end has the same cx control and target and no gate in between acts
        # on the target qubit: the loop cancels the two cx gates.
        if orientation == (a, b) and letter_b is None:
            return _C4
        if not unit_free:
            return _C3
        if not no_new_adjacency(unit, swap):
            return _C2
        qargs = order[swap].qargs  # the block depends only on the gates' values and qubits
        key = (step, *((n.name, *map(float, n.params), *map(qargs.index, n.qargs)) for n in nodes))
        if key not in blocks:
            blocks[key] = _diagonal_block(order[swap], nodes, step)
        return blocks[key], None

    used, rewrites = set(), []  # swap gates and partners already in a rewrite
    for swap in (swap for swap in swaps if swap not in used):
        a, b = first_wire[swap], second_wire[swap]
        # A swap that is the last multi-qubit gate on one of its qubits is not rewritten: a merged
        # diagonal gate would leave phase gates after it.
        if last_multi[a] <= first_position[swap] or last_multi[b] <= second_position[swap]:
            continue
        same, hits = pair[swap], []  # the scans' hits as (unit, step, letter_a, letter_b)
        for step in (1, -1):
            near, hit, letter_a = scan(a, first_position[swap], same, step)
            if near is None or (hit is None and pair[near] != same):
                continue
            near_b, hit_b, letter_b = scan(b, second_position[swap], same, step)
            if near == near_b and pair[near] == same:  # C1 on the swap itself
                stats[_C1] += 1
                break
            if hit is not None and hit == hit_b:
                hits.append((hit, step, letter_a, letter_b))
        else:
            best = None  # every rewrite saves two: the first hit with a plan wins, forward first
            for unit, step, letter_a, letter_b in hits:
                if unit in used:
                    continue
                plan = choose(swap, unit, step, letter_a, letter_b)
                if isinstance(plan, str):
                    stats[plan] += 1  # also for the backward hit after a forward plan
                elif best is None:
                    best = plan, unit
            if best is not None:
                plan, unit = best
                used.update((swap, unit))
                stats["applied"] += 1
                rewrites.append((order[swap], plan, [order[m] for m in groups.get(unit, [unit])]))
    return rewrites


def _wire_paulis(node) -> str:
    """The Pauli the standard gate ``node`` commutes with on each of its qubits (``_NO_PAULI`` for
    none), from the commutation table only, never from a numerical check; operations that commute
    with the same Pauli on every shared qubit commute exactly."""
    if node.name not in _SUPPORTED_GATES:
        return _NO_PAULI * len(node.qargs)
    gate = _LOOKUP_REPRESENTATIVES.get(node.name) or node.op
    qargs = tuple(range(len(node.qargs)))
    label = ""
    for wire in qargs:
        axes = [axis for axis, p in _PAULI_PROBES if _ENTRY(gate, qargs, p, (wire,)) is True]
        label += axes[0] if len(axes) == 1 else _NO_PAULI
    return label


def _diagonal_block(swap, nodes: list, step: int):
    """``swap; nodes`` (``step > 0``) or ``nodes; swap`` as ``(operations, global_phase)``.  The
    diagonal gates are merged without approximation; the computed angles carry rounding."""
    matrix = np.eye(4, dtype=complex)
    for node in nodes:
        gate, index = node.op.to_matrix(), swap.qargs.index(node.qargs[0])
        if len(node.qargs) == 1:
            gate = np.kron(gate, np.eye(2)) if index else np.kron(np.eye(2), gate)
        elif index:
            gate = _SWAP_MATRIX @ gate @ _SWAP_MATRIX
        matrix = gate @ matrix
    # Phases (c, c + a + g, c + b + g, c + a + b) on |00>, |01>, |10>, |11>, with qubit 0 the
    # low bit, are P(a) on qubit 0, P(b) on qubit 1 and a phase g on odd parity.
    phase_00, phase_01, phase_10, phase_11 = np.angle(np.diagonal(matrix))
    parity = (phase_01 + phase_10 - phase_00 - phase_11) / 2

    def phase(angle, index):
        if np.pi + angle != np.pi:  # wrap to [-pi, pi), unless adding pi would erase the angle
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return [(PhaseGate(angle), (index,))] if angle else []

    middle = phase(parity, 1)
    local = phase(phase_01 - phase_00 - parity, 0) + phase(phase_10 - phase_00 - parity, 1)
    if step > 0:
        return [_cx(0, 1), _cx(1, 0), *middle, _cx(0, 1), *local], float(phase_00)
    return [*local, _cx(0, 1), *middle, _cx(1, 0), _cx(0, 1)], float(phase_00)
