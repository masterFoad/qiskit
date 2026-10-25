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

"""Insert cancelling pairs of CZ-class gates between same-pair two-qubit runs around routing SWAPs.

This is the ``sandwich`` step of :class:`._AbsorbIntoSwaps`.

For a qubit pair ``(a, b)`` and Paulis ``P, Q`` in ``{X, Y, Z}`` let

    g(P, Q) = (I + P_a + Q_b - P_a Q_b) / 2 = (U (x) V) CZ (U (x) V)^dagger,

with ``U Z U^dagger = P`` and ``V Z V^dagger = Q``.  ``g`` is Hermitian and ``g g = I``.  If every
operation strictly between two consecutive runs ``R`` and ``R'`` of the pair commutes with ``P`` on
``a`` and with ``Q`` on ``b`` (exact Pauli-letter table, no tolerance), then ``g`` commutes with
everything between them and ``R . R' = (R g) . (g R')`` exactly.  ``SWAP g`` is always in the
iSWAP class (2 two-qubit gates instead of 3), so a SWAP next to such a window gets cheaper.  This
is the "set of commutable gates sandwiched by two SWAP gates" case of NASSC [1] (p. 8; p. 7, Fig. 7
for the orientation of the SWAP decomposition), where it is a term of the routing cost and the
cancellation itself is left to later passes.  Here it is applied after routing, in all nine Pauli
frames and between a run that holds a SWAP and the next run on the same pair (either side); per
qubit pair a dynamic program over its windows picks the frames that minimise the summed two-qubit
cost of :mod:`._swap_cost`.  The insertion is only applied where that cost drops; the resynthesis
itself is left to the optimization stage (``ConsolidateBlocks`` / ``TwoQubitPeepholeOptimization``).

The last run of a wire (after which the wire has only parameter-free single-qubit gates) never
receives a ``g``: its resynthesis tends to leave trailing single-qubit gates, which an
as-late-as-possible schedule moves to the end of the circuit, lengthening that qubit's idle time.

References:

[1] Liu, Li and Zhou. Not All SWAPs Have the Same Cost: A Case for Optimization-Aware Qubit
Routing. `arXiv:2205.10596 <https://arxiv.org/abs/2205.10596>`_
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import CZGate, HGate, SdgGate, SGate, UnitaryGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.preset_passmanagers import _swap_absorption
from qiskit.transpiler.preset_passmanagers._swap_cost import _SWAP_MAT, _matrix, two_qubit_cost

# Standard gates a window may contain besides the movers' families (``_SUPPORTED_GATES`` of
# :mod:`._swap_absorption`).  Their letters come from the same commutation-table lookup: ``cy`` and
# ``cry`` commute with Z on the control and Y on the target, ``ch`` with Z on the control, ``ecr``
# (``(IX - XY) / sqrt(2)``) with X on its second qubit.
_EXTRA_GATES = frozenset(("cy", "cry", "ch", "ecr"))
_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _g_matrix(p, q):
    """``g(P, Q)`` on (a, b) with ``a`` the least significant qubit."""
    pa = np.kron(_PAULI["I"], _PAULI[p])
    qb = np.kron(_PAULI[q], _PAULI["I"])
    return 0.5 * (np.eye(4, dtype=complex) + pa + qb - pa @ qb)


_G = {(p, q): _g_matrix(p, q) for p in "XYZ" for q in "XYZ"}

# Providers observed at import time; this detects later method replacements.
_GATE_TO_MATRIX = Gate.to_matrix
_UNITARY_ARRAY = UnitaryGate.__array__


def _is_canonical_unitary_gate(op):
    """Whether ``op`` is the unmodified concrete ``UnitaryGate`` matrix provider.

    This deliberately rejects subclasses and per-instance replacements of either method that
    participates in ``Gate.to_matrix``.  Method identities are compared to import-time references.
    This does not certify arbitrary parameter payloads or patches made before import.
    """
    if type(op) is not UnitaryGate:
        return False
    overrides = getattr(op, "__dict__", {})
    return (
        "to_matrix" not in overrides
        and "__array__" not in overrides
        and Gate.to_matrix is _GATE_TO_MATRIX
        and UnitaryGate.to_matrix is _GATE_TO_MATRIX
        and UnitaryGate.__array__ is _UNITARY_ARRAY
    )


def _wire_labels(node, cache):
    """The Pauli ``node`` commutes with on each of its qubits ("-" for none).

    This is the movers' labelling extended to ``_EXTRA_GATES``.  ``id`` commutes with every Pauli;
    it is labelled Z, so it never blocks a Z frame and blocks the X and Y frames.
    """
    if node.is_standard_gate():
        if node.name == "id":
            return "Z"
        if node.name in _EXTRA_GATES:
            key = (node.name, len(node.qargs))
            if key not in cache:
                cache[key] = _swap_absorption._standard_wire_paulis(node.op, len(node.qargs))
            return cache[key]
    return _swap_absorption._wire_paulis(node, cache)


class _Nodes:
    """Per-node wire data of the circuit, indexed by topological position.

    ``q0``/``q1`` are the wire indices of a one- or two-qubit node and ``p0``/``p1`` its positions
    on those wires; wider nodes keep their wires in ``wide_qargs``.  ``plain`` marks parameter-free
    gates with a known matrix, ``light`` the plain one-qubit gates, and ``mats`` holds the matrices
    read so far (standard-gate matrices are read on first use).
    """

    __slots__ = (
        "light",
        "mats",
        "nq",
        "order",
        "p0",
        "p1",
        "plain",
        "q0",
        "q1",
        "wide_qargs",
        "wires",
    )

    def __init__(self, order, num_wires):
        n = len(order)
        self.order = order
        self.wires = [[] for _ in range(num_wires)]
        self.q0 = [-1] * n
        self.q1 = [-1] * n
        self.p0 = [-1] * n
        self.p1 = [-1] * n
        self.nq = [0] * n
        self.wide_qargs = {}
        self.light = bytearray(n)
        self.mats = [None] * n
        self.plain = bytearray(n)

    def position(self, i, q):
        """Position of node ``i`` (one or two qubits) on wire ``q``."""
        return self.p0[i] if self.q0[i] == q else self.p1[i]


class _Run:
    """A maximal same-pair sequence of parameter-free two-qubit gates linked by light gates.

    ``first`` and ``last`` are its first and last member; ``lo``/``hi`` map each wire of the pair
    to the extent of the run on it (its own leading and trailing light gates included).
    """

    __slots__ = ("first", "hi", "index", "last", "lo", "matrix", "pair", "swap")

    def __init__(self, pair, index, first):
        self.pair = pair
        self.index = index
        self.first = self.last = first
        # Only runs of searched pairs get per-wire extent dictionaries.
        self.lo = None
        self.hi = None
        self.swap = False
        self.matrix = None


class _Runs:
    """The runs of a circuit and the qubit pairs worth searching."""

    __slots__ = ("by_pair", "last_hard", "pairs", "run_of", "searched_runs")

    def __init__(self, run_of, by_pair, last_hard):
        self.run_of = run_of
        self.by_pair = by_pair
        self.last_hard = last_hard
        # A pair is searched only if it has a SWAP and more than one run.
        self.pairs = [
            key for key, seq in by_pair.items() if len(seq) > 1 and any(r.swap for r in seq)
        ]
        self.searched_runs = set()
        for key in self.pairs:
            self.searched_runs.update(run.index for run in by_pair[key])


class _Window(NamedTuple):
    """A window between consecutive runs of a pair: the usable frames and the runs inside it."""

    choices: list
    inner: frozenset


def insert_swap_sandwiches(dag: DAGCircuit, fractional: bool = False) -> DAGCircuit:
    """Apply the exact g-insertions that lower the two-qubit cost of SWAP-adjacent blocks.

    Args:
        dag: a routed circuit that still contains its routing ``swap`` gates.
        fractional: whether a continuous ZZ rotation is native (changes the cost only).
    """
    order = list(dag.topological_op_nodes())
    nodes = _index_nodes(dag, order)
    runs = _find_runs(nodes)
    if not runs.pairs:
        return dag
    ends_wire = _set_extents(nodes, runs)
    windows = _find_windows(nodes, runs)
    if not windows:
        return dag
    costs = _RunCosts(nodes, fractional)
    inserts = _choose_insertions(runs, windows, ends_wire, costs.cost)
    # Same pair, consecutive windows: both sides of the inner run are modified, apply in order.
    qubits = dag.qubits
    for run, after, frame in inserts:
        _insert_g(dag, nodes, run, after, frame, qubits)
    return dag


def _standard_matrix(node):
    try:
        return np.asarray(node.matrix, dtype=complex)
    except Exception:  # pylint: disable=broad-except
        return None


def _index_nodes(dag, order):
    """Record each node's wires and positions and decide which nodes are plain and light.

    Standard-gate matrices are not read here (they are read when a run needs them), until the first
    custom ``Gate``: its matrix callback may mutate earlier nodes, so all earlier standard matrices
    are captured just before it, and every later matrix is read in order.  A canonical
    ``UnitaryGate`` runs no user code and does not trigger this.
    """
    getq = {qubit: i for i, qubit in enumerate(dag.qubits)}.__getitem__
    nodes = _Nodes(order, dag.num_qubits())
    wires, q0, q1, p0, p1 = nodes.wires, nodes.q0, nodes.q1, nodes.p0, nodes.p1
    nq, wide_qargs, light, mats, plain = (
        nodes.nq,
        nodes.wide_qargs,
        nodes.light,
        nodes.mats,
        nodes.plain,
    )
    eager = False
    for i, node in enumerate(order):
        qs = node.qargs
        width = nq[i] = len(qs)
        if width == 1:
            a = q0[i] = getq(qs[0])
            p0[i] = len(wires[a])
            wires[a].append(i)
        elif width == 2:
            a = q0[i] = getq(qs[0])
            b = q1[i] = getq(qs[1])
            p0[i] = len(wires[a])
            p1[i] = len(wires[b])
            wires[a].append(i)
            wires[b].append(i)
        elif width > 2:
            wide_qargs[i] = tuple(getq(q) for q in qs)
            for q in wide_qargs[i]:
                wires[q].append(i)
        if width <= 2:
            if node.is_standard_gate():
                plain[i] = not node.cargs and not node.is_parameterized()
                if eager and plain[i]:
                    # The native DAG node may have an uncached Python operation.  Materialize it
                    # before reading ``matrix`` so a prior custom callback cannot make the
                    # operation and matrix views disagree.  This is deliberately only the eager
                    # path: the lazy path avoids touching standard operations during admission.
                    _ = node.op
                    mats[i] = _standard_matrix(node)
                    plain[i] = mats[i] is not None
            else:
                op = node.op
                if not eager and isinstance(op, Gate) and not _is_canonical_unitary_gate(op):
                    # Custom matrix callbacks can mutate earlier nodes.  Capture their matrices
                    # before the first callback, then keep the original eager sampling order.
                    for j in range(i):
                        if plain[j] and mats[j] is None:
                            _ = order[j].op
                            mats[j] = _standard_matrix(order[j])
                            plain[j] = mats[j] is not None
                            light[j] = nq[j] == 1 and plain[j]
                    eager = True
                mats[i] = _matrix(node)
                plain[i] = mats[i] is not None
        light[i] = width == 1 and plain[i]
    return nodes


def _find_runs(nodes):
    """Group plain two-qubit gates into runs and note the last non-light node on each wire.

    Runs are keyed by their sorted qubit pair.  A run never spans a non-light node on either of
    its wires: such a node ends it.
    """
    q0, q1, nq, light, plain = nodes.q0, nodes.q1, nodes.nq, nodes.light, nodes.plain
    wide_qargs = nodes.wide_qargs
    run_of = [-1] * len(nodes.order)
    runs = []
    last_hard = [-1] * len(nodes.wires)
    by_pair = {}
    for i, node in enumerate(nodes.order):
        if light[i]:
            continue
        width = nq[i]
        if width == 2 and plain[i] and q0[i] != q1[i]:
            a, b = q0[i], q1[i]
            prev = last_hard[a]
            if prev >= 0 and prev == last_hard[b] and run_of[prev] >= 0:
                run = runs[run_of[prev]]
            else:
                key = (a, b) if a < b else (b, a)
                run = _Run(key, len(runs), i)
                runs.append(run)
                by_pair.setdefault(key, []).append(run)
            run.last = i
            run_of[i] = run.index
            if node.name == "swap" and node.is_standard_gate():
                run.swap = True
        if width == 1:
            last_hard[q0[i]] = i
        elif width == 2:
            last_hard[q0[i]] = last_hard[q1[i]] = i
        elif width > 2:
            for q in wide_qargs[i]:
                last_hard[q] = i
    return _Runs(run_of, by_pair, last_hard)


def _set_extents(nodes, runs):
    """Set ``lo``/``hi`` of every run of a searched pair; return the runs that end their wire.

    A stretch of light gates between two hard gates on a wire is owned by exactly one run: the run
    before it if that run belongs to a searched pair, else the run after it.  So no two runs ever
    claim the same light gate.  A run ends its wire when only light gates follow it there (any
    other operation, such as a measurement or a barrier, counts as further work).
    """
    wires, light, q0, p0, p1 = nodes.wires, nodes.light, nodes.q0, nodes.p0, nodes.p1
    run_of, last_hard, searched_runs = runs.run_of, runs.last_hard, runs.searched_runs
    ends_wire = set()
    for key in runs.pairs:
        for run in runs.by_pair[key]:
            first, last = run.first, run.last
            if any(last_hard[q] == last for q in key):
                ends_wire.add(run.index)
            run.lo = {}
            run.hi = {}
            for q in key:
                seq = wires[q]
                first_pos = p0[first] if q0[first] == q else p1[first]
                lo = first_pos
                while lo > 0 and light[seq[lo - 1]]:
                    lo -= 1
                if lo > 0 and lo < first_pos and run_of[seq[lo - 1]] in searched_runs:
                    lo = first_pos
                hi = p0[last] if q0[last] == q else p1[last]
                while hi + 1 < len(seq) and light[seq[hi + 1]]:
                    hi += 1
                run.lo[q] = lo
                run.hi[q] = hi
    return ends_wire


def _find_windows(nodes, runs):
    """The windows between consecutive runs of a pair, one of them holding a SWAP.

    Returns ``{(pair, t): (choices, inner)}`` for the window after run ``t`` of ``pair``:
    ``choices`` are the frames ``(P, Q)`` that commute with every operation strictly between the two
    extents, and ``inner`` the runs inside the window.
    """
    order, wires, q0, nq, wide_qargs = (
        nodes.order,
        nodes.wires,
        nodes.q0,
        nodes.nq,
        nodes.wide_qargs,
    )
    run_of = runs.run_of
    letter_cache = {}
    label_cache = {}

    def letter(i, q):
        val = letter_cache.get((i, q))
        if val is None:
            if nq[i] <= 2:
                axis = 0 if q0[i] == q else 1
            else:
                axis = wide_qargs[i].index(q)
            val = letter_cache[(i, q)] = _wire_labels(order[i], label_cache)[axis]
        return val

    windows = {}
    for key in runs.pairs:
        seq = runs.by_pair[key]
        for t in range(len(seq) - 1):
            left, right = seq[t], seq[t + 1]
            if not (left.swap or right.swap):
                continue
            allowed = []
            inner = set()
            for q in key:
                wire = wires[q]
                pauli = None
                for at in range(left.hi[q] + 1, right.lo[q]):
                    i = wire[at]
                    value = letter(i, q)
                    if value not in "XYZ" or (pauli is not None and value != pauli):
                        break
                    pauli = value
                    if run_of[i] >= 0:
                        inner.add(run_of[i])
                else:
                    allowed.append(list("XYZ") if pauli is None else [pauli])
                    continue
                break
            else:
                # A frame is the pair (left run's g, right run's g): the same g on both sides.
                choices = [((p, q), (p, q)) for p in allowed[0] for q in allowed[1]]
                windows[(key, t)] = _Window(choices, frozenset(inner))
    return windows


class _RunCosts:
    """Two-qubit cost of a run, optionally with a ``g`` in front of and behind it (cached)."""

    def __init__(self, nodes, fractional):
        self._nodes = nodes
        self._fractional = fractional
        self._cache = {}

    def matrix(self, run):
        """The 4x4 matrix of the run's extent, on (a, b) with ``a`` the least significant qubit."""
        if run.matrix is None:
            nodes = self._nodes
            order, wires, q0, q1, nq, mats = (
                nodes.order,
                nodes.wires,
                nodes.q0,
                nodes.q1,
                nodes.nq,
                nodes.mats,
            )
            a, b = run.pair
            members = set()
            for q in run.pair:
                members.update(wires[q][run.lo[q] : run.hi[q] + 1])
            # Work on a (b, a, b', a') tensor: one-qubit gates act on one axis, no kron.
            mat = np.eye(4, dtype=complex).reshape(2, 2, 2, 2)
            for i in sorted(members):
                m = mats[i]
                if m is None:
                    m = mats[i] = _standard_matrix(order[i])
                if nq[i] == 1:
                    if q0[i] == a:
                        mat = np.einsum("ij,bjcd->bicd", m, mat)
                    else:
                        mat = np.einsum("ij,jacd->iacd", m, mat)
                else:
                    if q0[i] != a or q1[i] != b:
                        m = _SWAP_MAT @ m @ _SWAP_MAT
                    mat = (m @ mat.reshape(4, 4)).reshape(2, 2, 2, 2)
            run.matrix = mat.reshape(4, 4)
        return run.matrix

    def cost(self, run, before, after):
        """Cost of ``run`` with ``g(before)`` in front and ``g(after)`` behind, or ``None``."""
        cache = self._cache
        key = (run.index, before, after)
        if key not in cache:
            mat = self.matrix(run)
            if before is not None:
                mat = mat @ _G[before]
            if after is not None:
                mat = _G[after] @ mat
            cache[key] = two_qubit_cost(mat, self._fractional)
        return cache[key]


def _blocked(seq, t, window, banned):
    """Whether window ``t`` of ``seq`` touches a banned run (either side or inside)."""
    return (
        seq[t].index in banned or seq[t + 1].index in banned or not window.inner.isdisjoint(banned)
    )


def _plan_pair(seq, pair_windows, ends_wire, banned, cost):
    """Dynamic program over the frame (or none) of each window of one pair.

    ``seq`` are the pair's runs and ``pair_windows[t]`` the window after run ``t`` (or ``None``).
    Returns ``(gain, path)`` with ``path[t]`` the chosen frame of window ``t``, or ``(0, None)``.
    """
    options = []
    for t in range(len(seq) - 1):
        opts = [None]
        win = pair_windows[t]
        if (
            win is not None
            and seq[t + 1].index not in ends_wire
            and not _blocked(seq, t, win, banned)
            and cost(seq[t], None, None) is not None
            and cost(seq[t + 1], None, None) is not None
        ):
            opts.extend(win.choices)
        options.append(opts)
    if all(len(o) == 1 for o in options):
        return 0, None
    # Runs with no usable window on either side keep their cost; the DP only needs to see
    # the others (each maximal stretch of usable windows is independent).
    active = [
        (t > 0 and len(options[t - 1]) > 1) or (t < len(options) and len(options[t]) > 1)
        for t in range(len(seq))
    ]
    base = 0
    total = 0
    path = [None] * len(options)
    prev = None
    for t, run in enumerate(seq):
        if not active[t]:
            continue
        base += cost(run, None, None)
        if prev is None:
            prev = {None: (0, 0)}
            start = t
            # back[k][f_right] = the best frame of the window before run start + k.
            back = []
        cur = {}
        links = {}
        right = options[t] if t < len(options) else (None,)
        for f_right in right:
            best = None
            for f_left, score in prev.items():
                # Ties: fewer insertions.  An unknown cost makes the combination unusable;
                # the all-None path always stays usable, as the base cost of every active run
                # is known.
                here = cost(
                    run,
                    None if f_left is None else f_left[1],
                    None if f_right is None else f_right[0],
                )
                if here is None:
                    continue
                val = (score[0] + here, score[1] + (f_right is not None))
                if best is None or val < best:
                    best = val
                    links[f_right] = f_left
            if best is not None:
                cur[f_right] = best
        back.append(links)
        prev = cur
        if len(right) == 1:  # end of a stretch: walk the links back once
            total += prev[None][0]
            frame = None
            for k in range(len(back) - 1, 0, -1):
                frame = back[k][frame]
                path[start + k - 1] = frame
            prev = None
    gain = base - total
    return (gain, path) if gain > 0 else (0, None)


def _choose_insertions(runs, windows, ends_wire, cost):
    """Plan every pair, then accept plans in order of gain while they stay jointly exact.

    Joint exactness: a run that receives a ``g`` never lies inside an accepted window, and an
    accepted window never contains a run that receives a ``g``.  Returns ``(run, after, frame)``
    insertions in application order.
    """
    pair_windows = {
        key: [windows.get((key, t)) for t in range(len(runs.by_pair[key]) - 1)]
        for key in runs.pairs
    }
    ranked = []
    initial_plans = {}
    for key in runs.pairs:
        gain, path = _plan_pair(runs.by_pair[key], pair_windows[key], ends_wire, frozenset(), cost)
        if gain > 0:
            ranked.append((-gain, key))
            initial_plans[key] = (gain, path)
    ranked.sort()
    banned = set()
    inserts = []
    for _, key in ranked:
        seq = runs.by_pair[key]
        gain, path = initial_plans.pop(key)
        # Bans only remove choices.  An initial optimum that is still feasible remains optimal,
        # with the same stable tie-breaking among its choices.
        if any(
            _blocked(seq, t, windows[(key, t)], banned)
            for t, frame in enumerate(path)
            if frame is not None
        ):
            gain, path = _plan_pair(seq, pair_windows[key], ends_wire, banned, cost)
            if not gain:
                continue
        for t, frame in enumerate(path):
            if frame is None:
                continue
            banned.add(seq[t].index)
            banned.add(seq[t + 1].index)
            banned.update(windows[(key, t)].inner)
            inserts.append((seq[t], True, frame[0]))
            inserts.append((seq[t + 1], False, frame[1]))
    return inserts


def _insert_g(dag, nodes, run, after, frame, qubits):
    """Put ``g(frame)`` right after (``after``) or right before the extent of ``run``.

    The boundary member of the run is replaced by ``member; light; g`` (or ``g; light; member``),
    where ``light`` are the run's own light gates beyond that member, removed from their place.
    """
    order, wires = nodes.order, nodes.wires
    a, b = run.pair
    anchor = run.last if after else run.first
    stretch = []
    for q in (a, b):
        at = nodes.position(anchor, q)
        stretch.extend(wires[q][at + 1 : run.hi[q] + 1] if after else wires[q][run.lo[q] : at])
    stretch.sort()
    anchor_node = order[anchor]
    wa, wb = Qubit(), Qubit()
    wire_map = {qubits[a]: wa, qubits[b]: wb}
    block = DAGCircuit()
    block.add_qubits([wa, wb])
    stretch_nodes = [order[i] for i in stretch]
    for node in stretch_nodes:
        dag.remove_op_node(node)

    def emit(node):
        return block.apply_operation_back(
            node.op, [wire_map[x] for x in node.qargs], check=False
        )._node_id

    if after:
        anchor_id = emit(anchor_node)
        stretch_ids = [emit(node) for node in stretch_nodes]
        _emit_g(block, wa, wb, frame)
    else:
        _emit_g(block, wa, wb, frame)
        stretch_ids = [emit(node) for node in stretch_nodes]
        anchor_id = emit(anchor_node)
    new = dag.substitute_node_with_dag(
        anchor_node, block, wires=[wire_map[x] for x in anchor_node.qargs]
    )
    # Keep ``order`` valid for a later insertion on the other side of the same run: ``new`` maps
    # each block node id to its node in ``dag``.
    order[anchor] = new[anchor_id]
    for i, block_id in zip(stretch, stretch_ids):
        order[i] = new[block_id]


def _emit_g(block, wa, wb, frame):
    """Append ``g(P, Q) = (U (x) V) CZ (U (x) V)^dagger`` to ``block``.

    Here ``U Z U^dagger = P`` and ``V Z V^dagger = Q``.
    """
    # Time order: U^dagger, CZ, U.
    pre = {"X": [HGate()], "Y": [SdgGate(), HGate()], "Z": []}
    post = {"X": [HGate()], "Y": [HGate(), SGate()], "Z": []}
    for wire, letter in ((wa, frame[0]), (wb, frame[1])):
        for gate in pre[letter]:
            block.apply_operation_back(gate, [wire], check=False)
    block.apply_operation_back(CZGate(), [wa, wb], check=False)
    for wire, letter in ((wa, frame[0]), (wb, frame[1])):
        for gate in post[letter]:
            block.apply_operation_back(gate, [wire], check=False)
