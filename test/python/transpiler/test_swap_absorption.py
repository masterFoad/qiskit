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

"""Test the placement of two-qubit interactions next to routing SWAPs."""

import collections
import contextlib
import copy
import math
import random
from unittest.mock import patch

import numpy as np
from ddt import ddt, data

from qiskit.circuit import Barrier, CommutationChecker, Gate, Parameter, QuantumCircuit
from qiskit.circuit.library import (
    CPhaseGate,
    CU1Gate,
    CXGate,
    CZGate,
    HGate,
    QFTGate,
    RYGate,
    RYYGate,
    RZGate,
    RZZGate,
    SwapGate,
    U1Gate,
    UnitaryGate,
    qaoa_ansatz,
)
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.transpiler import CouplingMap, PassManager, Target, generate_preset_pass_manager
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers import _swap_absorption
from qiskit.transpiler.preset_passmanagers.plugin import (
    PassManagerStagePlugin,
    PassManagerStagePluginManager,
)
from qiskit.transpiler.preset_passmanagers._swap_absorption import (
    _AbsorbIntoSwaps,
    _SUPPORTED_GATES,
    _find_moves,
    _wire_paulis,
)
from test import QiskitTestCase


# Finite specification, independent of the shared lookup table. Diagonal gates are functions
# of Z, X/Y families are functions of their Paulis, and controlled X families have Z-diagonal
# control projectors and X-like target actions. Pauli-product generators commute with their
# local factors (RZX is ZX in qargs order). CX-D-CX for diagonal D is a function of Z tensor Z.
_EXPECTED_WIRE_PAULIS = {
    **dict.fromkeys(("rz", "z", "s", "sdg", "t", "tdg", "p", "u1"), "Z"),
    **dict.fromkeys(("rx", "x", "sx", "sxdg"), "X"),
    **dict.fromkeys(("ry", "y"), "Y"),
    **dict.fromkeys(("cz", "cp", "crz", "cu1", "cs", "csdg", "rzz"), "ZZ"),
    **dict.fromkeys(("cx", "crx", "csx", "rzx"), "ZX"),
    "rxx": "XX",
    "ryy": "YY",
}


def _run(circuit):
    """Run ``_AbsorbIntoSwaps`` without a target on ``circuit``."""
    return dag_to_circuit(_AbsorbIntoSwaps().run(circuit_to_dag(circuit)))


def _swap_circuit(*between, interaction=None, swap=None):
    """A SWAP on (0, 1), the ``(operation, qubits)`` pairs ``between``, an interaction on (0, 1)
    (``cp`` by default) and more two-qubit work on both qubits."""
    qc = QuantumCircuit(3)
    qc.append(SwapGate() if swap is None else swap, [0, 1])
    for operation, qubits in between:
        qc.append(operation, qubits)
    qc.append(CPhaseGate(0.3) if interaction is None else interaction, [0, 1])
    qc.cx(0, 2)
    qc.cx(1, 2)
    return qc


def _custom_gate(name, definition):
    """A custom gate called ``name`` with the given ``definition``."""
    gate = Gate(name, definition.num_qubits, [])
    gate.definition = definition
    return gate


def _append_standard(circuit, name, qubits, angle=0.37):
    """Add a gate from the pass's commutation table, including legacy gate classes."""
    if name in ("cu1", "u1"):
        circuit.append(CU1Gate(angle) if name == "cu1" else U1Gate(angle), qubits)
    else:
        parameterized = {"p", "rz", "rx", "ry", "cp", "crz", "crx", "rzz", "rxx", "ryy", "rzx"}
        arguments = (angle,) if name in parameterized else ()
        getattr(circuit, name)(*arguments, *qubits)


def _isolated_label_cache():
    """Empty the module's label cache for a test that replaces or must exercise the lookup, and
    restore it afterwards, so labels from a replaced lookup never reach other tests."""
    return patch.dict(_swap_absorption._WIRE_PAULI_CACHE, clear=True)


def _neighbour_on_both_wires(circuit, index, step=1):
    """Index of the nearest instruction after (or before, for ``step=-1``) ``index`` on its
    qubits, if that instruction acts on both of them."""
    qubits = set(circuit.data[index].qubits)
    stop = len(circuit.data) if step > 0 else -1
    for other in range(index + step, stop, step):
        if qubits & set(circuit.data[other].qubits):
            return other if qubits <= set(circuit.data[other].qubits) else None
    return None


def _line_target(num_qubits, angle_bounded=False, two_qubit="cz"):
    """A line Target, optionally with a fractional ``rzz`` bounded to ``[0, pi/2]``."""
    coupling = CouplingMap.from_line(num_qubits)
    target = Target.from_configuration(
        [two_qubit, "rz", "sx", "x"], num_qubits=num_qubits, coupling_map=coupling
    )
    if angle_bounded:
        target.add_instruction(
            RZZGate(Parameter("theta")),
            dict.fromkeys(coupling.get_edges()),
            angle_bounds=[(0, math.pi / 2)],
        )
    return target


def _heavy_hex_backend(backend_class=GenericBackendV2):
    """A 19-qubit heavy-hex CZ backend."""
    return backend_class(
        num_qubits=19,
        basis_gates=["cz", "rz", "sx", "x"],
        coupling_map=CouplingMap.from_heavy_hex(3),
        seed=7,
    )


def _phase_and_cx_circuit(num_qubits=5):
    """All-to-all controlled phases and CX interactions, which need routing on a line."""
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            qc.cp(0.3 + 0.1 * (i + j), i, j)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            qc.cx(i, j)
    return qc


def _qft(num_qubits, measure=False):
    """A ``QFTGate`` circuit, optionally measured."""
    qc = QuantumCircuit(num_qubits)
    qc.append(QFTGate(num_qubits), range(num_qubits))
    if measure:
        qc.measure_all()
    return qc


def _qaoa(reps):
    """A bound QAOA ansatz for all-to-all ``ZZ`` terms on 5 qubits."""
    cost = SparsePauliOp.from_sparse_list(
        [("ZZ", [i, j], 1.0) for i in range(5) for j in range(i + 1, 5)], num_qubits=5
    )
    return qaoa_ansatz(cost, reps=reps).assign_parameters([0.4, 0.9, 0.2, 0.7][: 2 * reps])


class _Snapshot(AnalysisPass):
    """Append a copy of the circuit and of its layout to ``snapshots``."""

    def __init__(self, snapshots):
        super().__init__()
        self.snapshots = snapshots

    def run(self, dag):
        self.snapshots.append((copy.deepcopy(dag), copy.deepcopy(self.property_set["layout"])))


def _compile(level, circuit, absorb=True, snapshots=None, **kwargs):
    """Compile with the preset pass manager.

    Return the output and the names of the moved interactions.  ``absorb=False`` makes the pass
    leave every circuit unchanged, as a control.  A ``snapshots`` list receives the circuit after
    the init and the routing stages.
    """
    moved = []
    original = _swap_absorption._find_moves

    def recording(dag, *args, **options):
        order, units, moves = original(dag, *args, **options)
        moved.extend(order[units[unit][0][0]].name for unit, _, _ in moves)
        return order, units, moves

    with (
        patch.object(_swap_absorption, "_find_moves", recording),
        (
            contextlib.nullcontext()
            if absorb
            else patch.object(_AbsorbIntoSwaps, "run", lambda self, dag: dag)
        ),
    ):
        pm = generate_preset_pass_manager(level, **kwargs)
        if snapshots is not None:
            pm.post_init = PassManager([_Snapshot(snapshots)])
            pm.post_routing = PassManager([_Snapshot(snapshots)])
        out = pm.run(circuit)
    return out, moved


def _translation_absorb(pm):
    """The ``_AbsorbIntoSwaps`` step that starts the translation stage of ``pm``, or ``None``."""
    first = pm.translation._tasks[0][0] if pm.translation._tasks else None
    return first if isinstance(first, _AbsorbIntoSwaps) else None


def _routing_absorbs(pm):
    """Whether the routing stage of ``pm`` contains ``_AbsorbIntoSwaps``."""
    tasks = [task for tasks in pm.routing._tasks for task in tasks] if pm.routing else []
    return any(isinstance(task, _AbsorbIntoSwaps) for task in tasks)


def _replays(virtual, routed, layout):
    """Whether ``routed`` is ``virtual`` placed by ``layout``, with only swaps added."""
    pending = {qubit: collections.deque() for qubit in virtual.qubits}
    for node in virtual.topological_op_nodes():
        for qubit in node.qargs:
            pending[qubit].append(node)
    held = {index: qubit for qubit, index in layout.get_virtual_bits().items()}
    for node in routed.topological_op_nodes():
        physical = [routed.find_bit(qubit).index for qubit in node.qargs]
        qubits = [held[index] for index in physical]
        heads = {pending[q][0] if pending.get(q) else None for q in qubits}
        head = heads.pop() if len(heads) == 1 else None
        if head is not None and head.op == node.op and list(head.qargs) == qubits:
            for qubit in qubits:
                pending[qubit].popleft()
        elif node.name == "swap":
            held[physical[0]], held[physical[1]] = held[physical[1]], held[physical[0]]
        else:
            return False
    return not any(pending.values())


class _MockTranslationPlugin(PassManagerStagePlugin):
    """A translation plugin built like the ones of external providers."""

    def __init__(self, pass_level):
        self.pass_level = pass_level

    def pass_manager(self, pass_manager_config, optimization_level=None):
        return common.generate_translation_passmanager(
            pass_manager_config.target,
            basis_gates=pass_manager_config.basis_gates,
            approximation_degree=pass_manager_config.approximation_degree,
            coupling_map=pass_manager_config.coupling_map,
            hls_config=pass_manager_config.hls_config,
            qubits_initially_zero=pass_manager_config.qubits_initially_zero,
            optimization_level=optimization_level if self.pass_level else None,
        )


class _ProviderPluginBackend(GenericBackendV2):
    """A backend that selects its own translation stage plugin, as hardware providers do."""

    plugin = "mock_provider"

    def get_translation_stage_plugin(self):
        return self.plugin


def _with_mock_translation_plugins():
    """Make ``"mock_provider"`` and ``"mock_provider_no_level"`` build ``_MockTranslationPlugin``,
    which passes its optimization level to the helper only in the first."""
    plugins = {"mock_provider": True, "mock_provider_no_level": False}
    original = PassManagerStagePluginManager.get_passmanager_stage

    def get_passmanager_stage(self, stage_name, plugin_name, pm_config, optimization_level=None):
        if stage_name == "translation" and plugin_name in plugins:
            plugin = _MockTranslationPlugin(plugins[plugin_name])
            return plugin.pass_manager(pm_config, optimization_level)
        return original(self, stage_name, plugin_name, pm_config, optimization_level)

    return patch.object(
        PassManagerStagePluginManager, "get_passmanager_stage", get_passmanager_stage
    )


def _put_interaction(circuit, kind, qubits, angle):
    """Add a ``cp``, an ``rzz``, or a ``cx``-``rz``-``cx`` block on ``qubits``."""
    if kind == "cx_rz_cx":
        circuit.cx(*qubits)
        circuit.rz(angle, qubits[1])
        circuit.cx(*qubits)
    else:
        getattr(circuit, kind)(angle, *qubits)


def _interleaved(first, second, extra):
    """Wire 1 reads ``swap(0, 1)``, B on (1, 2), A on (0, 1), ``swap(1, 2)``: A moves back to the
    first SWAP across B, and B moves forward to the second SWAP across A."""
    qc = QuantumCircuit(4)
    qc.h(range(4))
    qc.swap(0, 1)
    if extra:
        qc.rz(0.13, 1)
    _put_interaction(qc, second, (1, 2), 0.41)
    if extra:
        qc.rz(0.29, 1)
    _put_interaction(qc, first, (0, 1), -0.37)
    qc.swap(1, 2)
    qc.cx(2, 3)
    qc.cx(0, 1)
    return qc


def _spans_overlap(dag):
    """The moves of ``dag`` and whether two of them cross overlapping stretches of one wire."""
    order, units, moves = _find_moves(dag)
    index = {qubit: i for i, qubit in enumerate(dag.qubits)}
    wires = collections.defaultdict(list)
    for n, node in enumerate(order):
        for qubit in node.qargs:
            wires[index[qubit]].append(n)
    spans = []
    for unit, swap, _ in moves:
        span = {}
        for qubit in units[swap][1]:
            wire = wires[qubit]
            positions = [wire.index(m) for m in (*units[unit][0], units[swap][0][0]) if m in wire]
            span[qubit] = (min(positions), max(positions))
        spans.append(span)
    overlap = any(
        a[q][0] <= b[q][1] and b[q][0] <= a[q][1]
        for i, a in enumerate(spans)
        for b in spans[i + 1 :]
        for q in set(a) & set(b)
    )
    return moves, overlap


@ddt
class TestAbsorbIntoSwaps(QiskitTestCase):
    """Test the ``_AbsorbIntoSwaps`` pass."""

    def test_standard_wire_labels(self):
        """All 27 families retain their labels for every angle and local qubit order."""
        self.assertEqual(_SUPPORTED_GATES, _EXPECTED_WIRE_PAULIS.keys())
        for name, expected in _EXPECTED_WIRE_PAULIS.items():
            for angle in (Parameter("theta"), 0, 1e-6, 0.37):
                for qubits in ((2,),) if len(expected) == 1 else ((0, 2), (2, 0)):
                    with self.subTest(gate=name, angle=angle, qubits=qubits):
                        circuit = QuantumCircuit(3)
                        _append_standard(circuit, name, qubits, angle)
                        node = next(circuit_to_dag(circuit).topological_op_nodes())
                        # Derive each label again, so every angle is looked up.
                        with _isolated_label_cache():
                            self.assertEqual(_wire_paulis(node), expected)

    def test_unsupported_gates_do_not_query(self):
        """Identity and allowlist checks precede lookup, including when a cache is populated."""
        circuit = QuantumCircuit(2)
        for name, width in (("cx", 2), ("rz", 1), ("cp", 2), ("unitary", 2)):
            circuit.append(Gate(name, width, []), range(width))
        circuit.append(HGate(), [0])
        circuit.append(CXGate(ctrl_state=0), [0, 1])
        circuit.append(CPhaseGate(0.3, ctrl_state=0), [0, 1])
        cache = {(name, len(label)): label for name, label in _EXPECTED_WIRE_PAULIS.items()}
        original_cache = cache.copy()
        with (
            _isolated_label_cache(),
            patch.object(
                _swap_absorption.SessionCommutationChecker,
                "check_commutation_entries",
                side_effect=AssertionError("unsupported gate queried"),
            ),
        ):
            for node in circuit_to_dag(circuit).topological_op_nodes():
                self.assertEqual(_wire_paulis(node, cache), "-" * len(node.qargs))
        self.assertEqual(cache, original_cache)

    def test_missing_or_ambiguous_axes(self):
        """A missing, negative, or ambiguous rule must conservatively block a wire."""
        circuit = QuantumCircuit(2)
        circuit.cp(0.3, 0, 1)
        node = next(circuit_to_dag(circuit).topological_op_nodes())
        for result in (None, False, True):
            with (
                _isolated_label_cache(),
                patch.object(
                    _swap_absorption.SessionCommutationChecker,
                    "check_commutation_entries",
                    return_value=result,
                ) as lookup,
            ):
                self.assertEqual(_wire_paulis(node), "--")
                self.assertEqual(lookup.call_count, 6)

    def test_exact_unitary_labels_without_lookup(self):
        """Only exactly diagonal custom unitaries receive ZZ, without table queries."""
        matrix = np.diag([1, 1j, -1, -1j])
        almost = matrix.copy()
        almost[0, 1] = 1e-13
        with (
            _isolated_label_cache(),
            patch.object(
                _swap_absorption.SessionCommutationChecker,
                "check_commutation_entries",
                side_effect=AssertionError("custom unitary queried"),
            ),
        ):
            for value, label in ((matrix, "ZZ"), (almost, "--")):
                circuit = QuantumCircuit(2)
                circuit.append(UnitaryGate(value, check_input=False), [0, 1])
                node = next(circuit_to_dag(circuit).topological_op_nodes())
                self.assertEqual(_wire_paulis(node), label)

    def test_small_rotation_crossing(self):
        """A tiny RY blocks CP while RZ permits a move with the scheduling guard satisfied."""
        for gate, count in ((RYGate, 0), (RZGate, 1)):
            circuit = _swap_circuit((gate(1e-6), [0]))
            circuit.global_phase = 0.123
            self.assertEqual(len(_find_moves(circuit_to_dag(circuit))[2]), count)
            out = _run(circuit)
            if count == 0:
                self.assertEqual(out, circuit)
            np.testing.assert_allclose(
                Operator(out).data, Operator(circuit).data, atol=1e-12, rtol=0
            )

    def test_lookup_cache_across_runs(self):
        """Only preprocessing queries rules, once per kind and local axis, and a second run of the
        pass reuses the labels without querying again."""
        circuit = QuantumCircuit(3)
        circuit.swap(0, 1)
        for angle in (0, 1e-6, 0.37):
            for name, label in _EXPECTED_WIRE_PAULIS.items():
                _append_standard(circuit, name, (2,) if len(label) == 1 else (2, 0), angle)
        circuit.cx(0, 1)
        with (
            _isolated_label_cache(),
            patch.object(
                _swap_absorption.SessionCommutationChecker,
                "check_commutation_entries",
                wraps=_swap_absorption.SessionCommutationChecker.check_commutation_entries,
            ) as lookup,
            patch.object(
                CommutationChecker, "commute", side_effect=AssertionError("commute called")
            ),
            patch.object(
                CommutationChecker,
                "commute_nodes",
                side_effect=AssertionError("commute_nodes called"),
            ),
        ):
            for count in (120, 0):
                lookup.reset_mock()
                _run(circuit)
                self.assertEqual(lookup.call_count, count)
                for call in lookup.call_args_list:
                    op, qargs, _, probe = call.args
                    self.assertEqual(qargs, tuple(range(op.num_qubits)))
                    self.assertIn(probe, [(wire,) for wire in qargs])
            # Exercise range checks along many repeated RZs, with an actual successful move.
            _swap_absorption._WIRE_PAULI_CACHE.clear()
            lookup.reset_mock()
            crossing = _swap_circuit(*[(RZGate(0.2), [0])] * 100)
            self.assertEqual(len(_find_moves(circuit_to_dag(crossing))[2]), 1)
            self.assertEqual(lookup.call_count, 15)  # RZ, CP, CX, independent of range length.
            lookup.reset_mock()
            no_swap = QuantumCircuit(2)
            no_swap.cp(0.3, 0, 1)
            self.assertEqual(_run(no_swap), no_swap)
            lookup.assert_not_called()

    @data(
        (0, False, False),
        (1, False, False),
        (2, False, True),
        (2, True, True),
        (3, False, True),
        (3, True, True),
    )
    def test_preset_absorption_levels(self, setting):
        """O2 and O3 start the translation stage with the pass, and routing does not run it."""
        level, angle_bounded, enabled = setting
        target = _line_target(4, angle_bounded)
        self.assertEqual(target.has_angle_bounds(), angle_bounded)
        pm = generate_preset_pass_manager(level, target=target)
        absorb = _translation_absorb(pm)
        self.assertEqual(absorb is not None, enabled)
        self.assertFalse(_routing_absorbs(pm))

    def test_preset_absorption_without_target(self):
        """Basis gates, with or without a coupling map, also get the pass at O2 and O3."""
        basis = ["cx", "rz", "sx", "x"]
        for level in (2, 3):
            for coupling_map in (CouplingMap.from_line(4), None):
                pm = generate_preset_pass_manager(
                    level, basis_gates=basis, coupling_map=coupling_map
                )
                self.assertIsInstance(_translation_absorb(pm), _AbsorbIntoSwaps)

    @data("translator", "synthesis")
    def test_builtin_translation_plugins(self, method):
        """The built-in translation plugins pass their level, so they also run the pass."""
        for level in (1, 2, 3):
            pm = generate_preset_pass_manager(
                level, target=_line_target(4), translation_method=method
            )
            self.assertEqual(_translation_absorb(pm) is not None, level in (2, 3))

    @data(None, 0, 1, 2, 3)
    def test_translation_passmanager_optimization_level(self, level):
        """``generate_translation_passmanager`` adds the pass only when given level 2 or 3."""
        target = _line_target(4)
        for translation in (
            common.generate_translation_passmanager(target, optimization_level=level),
            common.generate_translation_passmanager(None, optimization_level=level),
        ):
            first = translation._tasks[0][0] if translation._tasks else None
            self.assertEqual(isinstance(first, _AbsorbIntoSwaps), level in (2, 3))

    def test_level_three_moves_controlled_phase(self):
        """O3 moves controlled phases exactly."""
        target = _line_target(5)
        qc = _phase_and_cx_circuit()
        for seed in (3, 7, 11):
            out, moved = _compile(3, qc, target=target, seed_transpiler=seed)
            self.assertTrue(set(moved) & {"cp", "cu1", "cs", "csdg"}, moved)
            self.assertTrue(Operator.from_circuit(out).equiv(Operator(qc)))

    def test_wire_end_with_and_without_barrier(self):
        """A SWAP whose qubit does no further multi-qubit work is left alone, and a later
        multi-qubit barrier on that qubit counts as such work."""
        qc = QuantumCircuit(3)
        qc.cp(0.3, 0, 1)
        qc.cz(0, 2)
        qc.swap(0, 1)
        qc.cx(0, 2)
        qc.h(1)
        self.assertEqual(_run(qc), qc)
        fenced = qc.copy()
        fenced.barrier()
        out = _run(fenced)
        self.assertNotEqual(out, fenced)
        self.assertTrue(Operator(out).equiv(Operator(qc)))
        swap = [inst.name for inst in out.data].index("swap")
        self.assertEqual(out.data[_neighbour_on_both_wires(out, swap, -1)].name, "cp")

    def test_every_governed_gate_moves_equivalently(self):
        """Each table entry is exercised as a candidate or a commuting intervening gate."""
        single = {
            "Z": ("rz", "z", "s", "sdg", "t", "tdg", "p", "u1"),
            "X": ("rx", "x", "sx", "sxdg"),
            "Y": ("ry", "y"),
        }
        double = {
            "ZZ": ("cz", "cp", "crz", "cu1", "cs", "csdg", "rzz"),
            "ZX": ("cx", "crx", "csx", "rzx"),
            "XX": ("rxx",),
            "YY": ("ryy",),
        }
        # Every gate of the table is exercised below.
        exercised = {name for names in (*single.values(), *double.values()) for name in names}
        self.assertEqual(_SUPPORTED_GATES, exercised)

        for pauli, names in single.items():
            for name in names:
                with self.subTest(gate=name):
                    qc = QuantumCircuit(3)
                    qc.swap(0, 1)
                    wire = 1 if pauli == "X" else 0
                    _append_standard(qc, name, (wire,))
                    candidate = {"Z": "cp", "X": "cx", "Y": "ryy"}[pauli]
                    _append_standard(qc, candidate, (0, 1))
                    qc.cx(0, 2)
                    qc.cx(1, 2)
                    self.assertEqual(len(_find_moves(circuit_to_dag(qc))[2]), 1)
                    self.assertTrue(Operator(_run(qc)).equiv(Operator(qc)))
                    np.testing.assert_allclose(
                        Operator(_run(qc)).data, Operator(qc).data, atol=1e-12, rtol=0
                    )

        for paulis, names in double.items():
            for name in names:
                for qubits in ((0, 1), (1, 0)):
                    with self.subTest(gate=name, qubits=qubits):
                        qc = QuantumCircuit(3)
                        qc.swap(0, 1)
                        for wire, pauli in zip(qubits, paulis):
                            _append_standard(qc, {"Z": "rz", "X": "rx", "Y": "ry"}[pauli], (wire,))
                        _append_standard(qc, name, qubits)
                        qc.cx(0, 2)
                        qc.cx(1, 2)
                        self.assertEqual(len(_find_moves(circuit_to_dag(qc))[2]), 1)
                        self.assertTrue(Operator(_run(qc)).equiv(Operator(qc)))
                        np.testing.assert_allclose(
                            Operator(_run(qc)).data, Operator(qc).data, atol=1e-12, rtol=0
                        )

    def test_moves_next_to_swap(self):
        """An interaction reaches its SWAP through gates it commutes with, exactly."""
        cp_circuit = QuantumCircuit(3)
        cp_circuit.h(0)
        cp_circuit.compose(
            _swap_circuit((RZGate(0.2), [0]), (CZGate(), [0, 2]), (RZZGate(0.4), [1, 2])),
            inplace=True,
        )
        earlier = QuantumCircuit(3)
        earlier.cp(0.3, 0, 1)
        earlier.cz(0, 2)
        earlier.swap(0, 1)
        earlier.cx(0, 2)
        earlier.cx(1, 2)
        unitary = QuantumCircuit(3)
        unitary.swap(0, 1)
        unitary.p(0.1, 1)
        unitary.cp(0.5, 1, 2)
        unitary.append(UnitaryGate(Operator.from_label("ZZ").to_matrix() * 1j), [0, 1])
        unitary.cx(0, 2)
        unitary.cx(2, 1)
        cx_circuit = QuantumCircuit(4)
        cx_circuit.h([0, 1])
        cx_circuit.swap(0, 1)
        cx_circuit.rz(0.2, 0)
        cx_circuit.cx(0, 2)
        cx_circuit.sx(1)
        cx_circuit.cx(3, 1)
        cx_circuit.cx(0, 1)
        cx_circuit.cx(0, 2)
        cx_circuit.cx(1, 3)
        zz_after = QuantumCircuit(3)
        zz_after.h([0, 1, 2])
        zz_after.swap(0, 1)
        zz_after.cz(1, 2)
        zz_after.cx(0, 1)
        zz_after.rz(0.7, 1)
        zz_after.cx(0, 1)
        zz_after.h(0)
        zz_after.cx(0, 2)
        zz_after.cx(1, 2)
        zz_shared = QuantumCircuit(3)
        zz_shared.swap(0, 1)
        zz_shared.cx(0, 2)
        zz_shared.cx(0, 1)
        zz_shared.rz(0.7, 1)
        zz_shared.cx(0, 1)
        zz_shared.cx(1, 2)
        zz_shared.cx(0, 2)
        cases = {
            "cp through diagonal gates": (cp_circuit, "cp", 1),
            "earlier term delayed onto the swap": (earlier, "cp", -1),
            "diagonal UnitaryGate": (unitary, "unitary", 1),
            "cx through diagonal control and X-like target": (cx_circuit, "cx", 1),
            "ryy through Y rotations": (
                _swap_circuit((RYGate(0.3), [0]), (RYYGate(0.2), [1, 2]), interaction=RYYGate(0.5)),
                "ryy",
                1,
            ),
            "cx rz cx block after the swap": (zz_after, "cx", 1),
            "cx rz cx block through a shared control": (zz_shared, "cx", 1),
        }
        for name, (qc, moved, step) in cases.items():
            with self.subTest(name):
                out = _run(qc)
                self.assertEqual(Operator(out), Operator(qc))
                swap = [inst.name for inst in out.data].index("swap")
                self.assertEqual(out.data[_neighbour_on_both_wires(out, swap, step)].name, moved)

    def test_blocked_moves_leave_circuit_unchanged(self):
        """No move crosses an operation that blocks it, and lookalike gates are not moved."""
        one, two = QuantumCircuit(1), QuantumCircuit(2)
        one.h(0)
        two.h(0)
        cx_definition = QuantumCircuit(2)
        cx_definition.cx(0, 1)
        nearly_zz = UnitaryGate(Operator.from_label("ZZ")).to_matrix()
        nearly_zz[0, 1] = nearly_zz[1, 0] = 1e-13
        control_flow = QuantumCircuit(3, 1)
        control_flow.swap(0, 1)
        with control_flow.if_test((control_flow.clbits[0], 1)):
            control_flow.rz(0.2, 0)
        control_flow.cp(0.3, 0, 1)
        control_flow.cx(0, 2)
        control_flow.cx(1, 2)
        wire_end = QuantumCircuit(3)
        wire_end.cp(0.3, 0, 1)
        wire_end.cz(0, 2)
        wire_end.swap(0, 1)
        wire_end.cx(0, 2)
        wire_end.h(1)
        non_commuting = QuantumCircuit(2)
        non_commuting.cp(0.3, 0, 1)
        non_commuting.h(0)
        non_commuting.swap(0, 1)
        non_commuting.measure_all()
        adjacent = QuantumCircuit(3)
        adjacent.cz(0, 1)
        adjacent.compose(_swap_circuit((RZGate(0.2), [0])), inplace=True)
        cases = {
            "control flow": control_flow,
            "barrier": _swap_circuit((Barrier(2), [0, 1])),
            "swap ending a wire": wire_end,
            "non-commuting gate": non_commuting,
            "swap already next to a term": adjacent,
            "cx after a gate diagonal on its target": _swap_circuit(
                (RZGate(0.2), [1]), interaction=CXGate()
            ),
            "custom gate named swap": _swap_circuit(
                (RZGate(0.2), [0]), swap=_custom_gate("swap", cx_definition)
            ),
            "nearly diagonal UnitaryGate": _swap_circuit(
                (UnitaryGate(nearly_zz, check_input=False), [0, 2])
            ),
        }
        for name in ("cp", "cx", "unitary"):
            cases[f"custom gate named {name}"] = _swap_circuit((_custom_gate(name, two), [0, 2]))
        cases["custom gate named rz"] = _swap_circuit((_custom_gate("rz", one), [0]))
        for name, qc in cases.items():
            with self.subTest(name):
                self.assertEqual(_run(qc), qc)

    def test_two_overlapping_moves(self):
        """Two CP moves may share one qubit across their SWAP pairs."""
        qc = QuantumCircuit(4)
        qc.swap(0, 1)
        qc.rz(0.11, 0)
        qc.cp(0.23, 0, 1)
        qc.swap(1, 2)
        qc.rz(0.17, 1)
        qc.cp(-0.31, 1, 2)
        qc.cx(0, 3)
        qc.cx(1, 3)
        qc.cx(2, 3)
        _, units, moves = _find_moves(circuit_to_dag(qc))
        self.assertEqual(len(moves), 2)
        self.assertEqual(len(units[moves[0][1]][1] & units[moves[1][1]][1]), 1)
        self.assertTrue(Operator(_run(qc)).equiv(Operator(qc)))

    def test_two_opposite_direction_moves(self):
        """One interaction moves forward and another moves backward in the same pass."""
        qc = QuantumCircuit(5)
        qc.cp(0.23, 0, 1)
        qc.rz(0.11, 0)
        qc.swap(0, 1)
        qc.cx(0, 4)
        qc.cx(1, 4)
        qc.swap(2, 3)
        qc.rz(0.17, 2)
        qc.cp(-0.31, 2, 3)
        qc.cx(2, 4)
        qc.cx(3, 4)
        _, _, moves = _find_moves(circuit_to_dag(qc))
        self.assertEqual(len(moves), 2)
        self.assertEqual({step for _, _, step in moves}, {-1, 1})
        self.assertTrue(Operator(_run(qc)).equiv(Operator(qc)))

    def test_symbolic_controlled_phase(self):
        """The pass moves symbolic CP without changing its parameter or bound operator."""
        theta = Parameter("theta")
        qc = _swap_circuit((RZGate(0.2), [0]), interaction=CPhaseGate(theta))
        out = _run(qc)
        self.assertNotEqual(out, qc)
        self.assertEqual(out.parameters, qc.parameters)
        self.assertTrue(
            Operator(out.assign_parameters({theta: 0.37})).equiv(
                Operator(qc.assign_parameters({theta: 0.37}))
            )
        )

    def test_asymmetric_crz_is_not_moved_across_swap(self):
        """A ``crz`` (which does not commute with SWAP) is moved up to a SWAP, never through one."""
        for control, target in ((0, 1), (1, 0)):
            qc = QuantumCircuit(3)
            qc.h([0, 1])
            qc.swap(0, 1)
            qc.rz(0.2, 0)
            qc.crz(0.7, control, target)
            qc.swap(0, 1)
            qc.crz(0.5, target, control)
            qc.cx(0, 2)
            qc.cx(1, 2)
            out = _run(qc)
            self.assertEqual(Operator(out), Operator(qc))
            self.assertEqual([inst.name for inst in out.data].count("swap"), 2)

    def test_each_term_absorbed_once(self):
        """Two SWAPs never claim the same diagonal term."""
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        qc.rz(0.1, 0)
        qc.cp(0.3, 0, 1)
        qc.rz(0.2, 1)
        qc.swap(0, 1)
        qc.cx(0, 1)
        out = _run(qc)
        self.assertEqual(Operator(out), Operator(qc))
        self.assertEqual(out.count_ops(), qc.count_ops())

    def test_random_small_circuit_operators(self):
        """Sample interacting gates after seeded CX and CP move opportunities."""
        rng = random.Random(404)
        selected = 0
        singles = ("rx", "ry", "rz", "sx", "s", "h")
        doubles = ("cx", "cz", "cp", "crz", "rxx", "rzz", "swap")
        for _ in range(128):
            qc = QuantumCircuit(4)
            qc.swap(0, 1)
            qc.rz(0.2, 0)
            qc.cx(0, 1)
            qc.cx(0, 2)
            qc.cx(1, 2)
            qc.swap(2, 3)
            qc.rz(0.1, 2)
            qc.cp(0.3, 2, 3)
            qc.cx(2, 0)
            qc.cx(3, 0)
            for _ in range(rng.randint(6, 14)):
                if rng.random() < 0.4:
                    name = rng.choice(singles)
                    qubit = rng.randrange(4)
                    angle = (rng.uniform(-1, 1),) if name in ("rx", "ry", "rz") else ()
                    getattr(qc, name)(*angle, qubit)
                else:
                    name = rng.choice(doubles)
                    a, b = rng.sample(range(4), 2)
                    angle = (rng.uniform(-1, 1),) if name in ("cp", "crz", "rxx", "rzz") else ()
                    getattr(qc, name)(*angle, a, b)
            selected += len(_find_moves(circuit_to_dag(qc))[2])
            self.assertTrue(Operator(_run(qc)).equiv(Operator(qc)))
        self.assertGreater(selected, 0)

    def test_moves_on_overlapping_stretches(self):
        """Two moves whose stretches overlap on a shared wire, each unit crossing the other, give
        the same operator, also for ``cx``-``rz``-``cx`` blocks."""
        kinds = ("cp", "rzz", "cx_rz_cx")
        for first in kinds:
            for second in kinds:
                for extra in (False, True):
                    with self.subTest(first=first, second=second, extra=extra):
                        qc = _interleaved(first, second, extra)
                        moves, overlap = _spans_overlap(circuit_to_dag(qc))
                        self.assertEqual(len(moves), 2)
                        self.assertTrue(overlap)
                        out = _run(qc)
                        self.assertNotEqual(out, qc)
                        self.assertTrue(Operator(out).equiv(Operator(qc)))

    def test_random_overlapping_moves(self):
        """Random circuits with several SWAPs and diagonal interactions keep their operator,
        including those with moves on overlapping stretches."""
        rng = random.Random(8)
        singles = ("rz", "rz", "sx", "h", "s", "x")
        doubles = ("cx", "cz", "cp", "rzz", "rxx", "swap", "swap", "cx_rz_cx")
        moved = overlapping = 0
        for _ in range(400):
            num_qubits = rng.choice((3, 4))
            qc = QuantumCircuit(num_qubits)
            for _ in range(rng.randint(8, 18)):
                if rng.random() < 0.3:
                    name = rng.choice(singles)
                    qubit = rng.randrange(num_qubits)
                    getattr(qc, name)(*((rng.uniform(-1, 1),) if name == "rz" else ()), qubit)
                else:
                    name = rng.choice(doubles)
                    qubits = rng.sample(range(num_qubits), 2)
                    if name in ("cp", "rzz", "rxx", "cx_rz_cx"):
                        _put_interaction(qc, name, qubits, rng.uniform(-1, 1))
                    else:
                        getattr(qc, name)(*qubits)
            for qubit in range(num_qubits - 1):
                qc.cx(qubit, qubit + 1)
            moves, overlap = _spans_overlap(circuit_to_dag(qc))
            moved += bool(moves)
            overlapping += overlap
            if moves:
                self.assertTrue(Operator(_run(qc)).equiv(Operator(qc)))
        self.assertGreater(moved, 0)
        self.assertGreater(overlapping, 0)

    @data(2, 3)
    def test_preset_exact(self, level):
        """A custom gate named ``cx`` and routed ``crz`` in both directions compile exactly."""
        definition = QuantumCircuit(2)
        definition.cx(1, 0)
        definition.ry(0.3, 0)
        custom_cx = _custom_gate("cx", definition)
        cx_circuit = QuantumCircuit(4)
        cx_circuit.h(range(4))
        for i in range(4):
            for j in range(i + 1, 4):
                cx_circuit.cp(0.2 * (i + j), i, j)
                cx_circuit.append(custom_cx, [j, i])
        crz_circuit = QuantumCircuit(4)
        crz_circuit.h(range(4))
        for i in range(4):
            for j in range(4):
                if i != j:
                    crz_circuit.crz(0.1 * (i + 2 * j + 1), i, j)
        for qc, two_qubit, seed in ((cx_circuit, "cx", 7), (crz_circuit, "cz", 3)):
            with self.subTest(two_qubit=two_qubit):
                target = _line_target(4, two_qubit=two_qubit)
                out = generate_preset_pass_manager(level, target=target, seed_transpiler=seed).run(
                    qc
                )
                self.assertTrue(Operator.from_circuit(out).equiv(Operator(qc)))

    def test_preset_pass_manager_qaoa(self):
        """Routed QAOA on a line needs fewer two-qubit gates, and stays equivalent."""
        qc = _qaoa(reps=1)
        target = _line_target(5)
        out = _compile(2, qc, target=target, seed_transpiler=7)[0]
        self.assertEqual(Operator.from_circuit(out), Operator(qc))
        without = _compile(2, qc, absorb=False, target=target, seed_transpiler=7)[0]
        self.assertLess(out.count_ops()["cz"], without.count_ops()["cz"])

    def test_level_two_layout_unchanged(self):
        """At O2 the pass runs after VF2PostLayout, so the layout equals that of the control."""
        backend = _heavy_hex_backend()
        for qc in (_qft(8, measure=True), _phase_and_cx_circuit(6)):
            for seed in (1, 2, 3):
                out, moved = _compile(2, qc, backend=backend, seed_transpiler=seed)
                self.assertGreater(len(moved), 0)
                control = _compile(2, qc, absorb=False, backend=backend, seed_transpiler=seed)[0]
                self.assertNotEqual(out, control)
                self.assertEqual(out.layout, control.layout)

    def test_level_three_initial_layout_kept(self):
        """At O3 with an ``initial_layout`` the layout equals that of the control."""
        target = _line_target(6)
        qc = _phase_and_cx_circuit(6)
        initial_layout = [5, 3, 1, 0, 2, 4]
        for seed in (1, 2, 3):
            kwargs = {"target": target, "initial_layout": initial_layout, "seed_transpiler": seed}
            out, moved = _compile(3, qc, **kwargs)
            self.assertGreater(len(moved), 0)
            control = _compile(3, qc, absorb=False, **kwargs)[0]
            self.assertEqual(out.layout, control.layout)
            self.assertEqual(out.layout.initial_index_layout()[:6], initial_layout)
            self.assertTrue(Operator.from_circuit(out).equiv(Operator(qc)))

    def test_level_three_default_layout_equivalent(self):
        """At O3 the final VF2PostLayout can choose another layout, so only check equivalence."""
        target = _line_target(6)
        qc = _qft(6)
        for seed in (1, 2, 3):
            out, moved = _compile(3, qc, target=target, seed_transpiler=seed)
            self.assertGreater(len(moved), 0)
            self.assertTrue(Operator.from_circuit(out).equiv(Operator(qc)))

    @data(2, 3)
    def test_provider_translation_plugin(self, level):
        """A provider translation plugin that passes its level gets the default plugin's output."""
        qc = _qft(8, measure=True)
        expected, default_moved = _compile(
            level, qc, backend=_heavy_hex_backend(), seed_transpiler=1
        )
        with _with_mock_translation_plugins():
            backend = _heavy_hex_backend(_ProviderPluginBackend)
            pm = generate_preset_pass_manager(level, backend=backend, seed_transpiler=1)
            self.assertIsInstance(_translation_absorb(pm), _AbsorbIntoSwaps)
            out, moved = _compile(level, qc, backend=backend, seed_transpiler=1)
            backend.plugin = "mock_provider_no_level"
            pm = generate_preset_pass_manager(level, backend=backend, seed_transpiler=1)
            self.assertIsNone(_translation_absorb(pm))
            _, unmoved = _compile(level, qc, backend=backend, seed_transpiler=1)
        self.assertGreater(len(moved), 0)
        self.assertEqual(moved, default_moved)
        self.assertEqual(out, expected)
        self.assertEqual(unmoved, [])

    @data(2, 3)
    def test_routing_stage_adds_only_swaps(self, level):
        """The routing stage's output is the init stage's output, laid out, plus swaps only."""
        for qc, kwargs in (
            (_phase_and_cx_circuit(6), {"target": _line_target(6)}),
            (_qft(8, measure=True), {"backend": _heavy_hex_backend()}),
        ):
            snapshots = []
            _, moved = _compile(level, qc, snapshots=snapshots, seed_transpiler=1, **kwargs)
            self.assertGreater(len(moved), 0)
            (virtual, _), (routed, layout) = snapshots
            self.assertIn("swap", routed.count_ops())
            self.assertTrue(_replays(virtual, routed, layout))
