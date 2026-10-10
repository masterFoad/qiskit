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
from ddt import ddt, data, unpack

from qiskit.circuit import Barrier, Gate, Parameter, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import (
    CPhaseGate,
    CU1Gate,
    CXGate,
    CZGate,
    QFTGate,
    RYGate,
    RYYGate,
    RZGate,
    RZZGate,
    SwapGate,
    U1Gate,
    UnitaryGate,
    XGate,
    qaoa_ansatz,
)
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager.flow_controllers import DoWhileController
from qiskit.providers import QubitProperties
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.transpiler import CouplingMap, PassManager, Target, generate_preset_pass_manager
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passes import TwoQubitPeepholeOptimization, WrapAngles
from qiskit.transpiler.passes.utils.wrap_angles import WrapAngleRegistry
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers import _swap_absorption
from qiskit.transpiler.preset_passmanagers.plugin import (
    PassManagerStagePlugin,
    PassManagerStagePluginManager,
)
from qiskit.transpiler.preset_passmanagers._swap_absorption import (
    _AbsorbIntoSwaps,
    _WIRE_PAULIS,
    _find_moves,
    _short_t2_qubits,
)
from test import QiskitTestCase


def _run(circuit):
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


def _append_standard(circuit, name, qubits):
    """Add a gate from the pass's commutation table, including legacy gate classes."""
    if name in ("cu1", "u1"):
        circuit.append(CU1Gate(0.37) if name == "cu1" else U1Gate(0.37), qubits)
    else:
        parameterized = {"p", "rz", "rx", "ry", "cp", "crz", "crx", "rzz", "rxx", "ryy", "rzx"}
        arguments = (0.37,) if name in parameterized else ()
        getattr(circuit, name)(*arguments, *qubits)


def _neighbour_on_both_wires(circuit, index, step=1):
    """Index of the nearest instruction after (or before, for ``step=-1``) ``index`` on its
    qubits, if that instruction acts on both of them."""
    qubits = set(circuit.data[index].qubits)
    stop = len(circuit.data) if step > 0 else -1
    for other in range(index + step, stop, step):
        if qubits & set(circuit.data[other].qubits):
            return other if qubits <= set(circuit.data[other].qubits) else None
    return None


def _fold_rzz(angles, _qubits):
    """Exact rewrite of an ``rzz`` angle into ``[0, pi/2]``, up to global phase."""
    theta = (float(angles[0]) + math.pi) % (2 * math.pi) - math.pi
    dag = DAGCircuit()
    dag.add_qreg(register := QuantumRegister(2))
    first, second = register
    if abs(theta) > math.pi / 2:
        # RZZ(theta) = RZZ(theta -+ pi) (Z x Z) up to global phase.
        theta -= math.copysign(math.pi, theta)
        dag.apply_operation_back(RZGate(math.pi), (first,))
        dag.apply_operation_back(RZGate(math.pi), (second,))
    flip = theta < 0
    if flip:
        dag.apply_operation_back(XGate(), (first,))
    dag.apply_operation_back(RZZGate(abs(theta)), (first, second))
    if flip:
        dag.apply_operation_back(XGate(), (first,))
    return dag


def _line_target(num_qubits, angle_bounded=False, two_qubit="cz", t2=None):
    """A line Target, optionally with a fractional ``rzz`` bounded to ``[0, pi/2]`` and qubit T2s
    (seconds, ``None`` for unknown)."""
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
    if t2 is not None:
        target.qubit_properties = [QubitProperties(t1=1e-4, t2=value) for value in t2]
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

    Return the output, the names of the moved interactions and the number of level 3 loop
    iterations (runs of the in-loop two-qubit peephole).  ``absorb=False`` makes the pass leave
    every circuit unchanged, as a control.  A ``snapshots`` list receives the circuit after the
    init and the routing stages.
    """
    moved = []
    original = _swap_absorption._find_moves

    def recording(dag, *args, **options):
        order, units, moves = original(dag, *args, **options)
        moved.extend(order[units[unit][0][0]].name for unit, _, _, _ in moves)
        return order, units, moves

    peepholes = []

    def callback(pass_, **_):
        if isinstance(pass_, TwoQubitPeepholeOptimization):
            peepholes.append(pass_)

    registry = WrapAngleRegistry()
    registry.add_wrapper("rzz", _fold_rzz)
    with (
        patch.object(_swap_absorption, "_find_moves", recording),
        patch.object(WrapAngles, "DEFAULT_REGISTRY", registry),
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
        out = pm.run(circuit, callback=callback)
    return out, moved, len(peepholes) - (level == 2)


def _rzz_in_bounds(circuit):
    """Whether every ``rzz`` angle of ``circuit`` lies in ``[0, pi/2]``."""
    return all(
        0 <= float(inst.operation.params[0]) <= math.pi / 2 + 1e-12
        for inst in circuit.data
        if inst.operation.name == "rzz"
    )


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


def _tail_circuit(measure=False, other_pair=False, final_barrier=False):
    """A SWAP on (0, 1) whose ``cp`` moves next to it; then qubit 1 only works with qubit 2."""
    qc = QuantumCircuit(4, 1)
    qc.h(0)
    qc.swap(0, 1)
    qc.rz(0.2, 0)
    qc.cz(0, 2)
    qc.cp(0.3, 0, 1)
    qc.cx(0, 2)
    qc.cx(0, 3)
    qc.cx(1, 2)
    if other_pair:
        qc.cx(1, 3)
    if final_barrier:
        qc.barrier()
    if measure:
        qc.measure(1, 0)
    return qc


@ddt
class TestAbsorbIntoSwaps(QiskitTestCase):
    """Test the ``_AbsorbIntoSwaps`` pass."""

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
        if enabled:
            # Only level 3 gives the pass the target, for the tail rule.
            self.assertIs(absorb.target, target if level == 3 else None)
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

    @data(False, True)
    def test_level_three_moves_controlled_phase(self, angle_bounded):
        """O3 moves controlled phases exactly, and within the angle bounds of the Target."""
        target = _line_target(5, angle_bounded)
        qc = _phase_and_cx_circuit()
        for seed in (3, 7, 11):
            out, moved, _ = _compile(3, qc, target=target, seed_transpiler=seed)
            self.assertTrue(set(moved) & {"cp", "cu1", "cs", "csdg"}, moved)
            self.assertTrue(Operator.from_circuit(out).equiv(Operator(qc)))
            self.assertTrue(_rzz_in_bounds(out))

    @data(3, 7, 11, 22, 33, 44, 55, 66)
    def test_level_three_angle_bounded_operational_contract(self, seed):
        """The routed QFT regression remains exact and obeys angle bounds at level 3."""
        target = _line_target(6, angle_bounded=True)
        qc = _qft(6).decompose()
        out, _, _ = _compile(3, qc, target=target, seed_transpiler=seed)
        self.assertTrue(_rzz_in_bounds(out))
        actual = Operator.from_circuit(out).data
        expected = Operator(qc).data
        overlap = np.vdot(expected, actual)
        np.testing.assert_allclose(actual, expected * overlap / abs(overlap), atol=1e-12, rtol=0)

    @data(False, True)
    def test_level_two_ignores_target_t2(self, angle_bounded):
        """At level 2 the step gets no Target, so its output matches a step built without one."""
        target = _line_target(5, angle_bounded)
        for qc in (_phase_and_cx_circuit(), _qaoa(reps=2)):
            for seed in (1, 2, 3):
                out = _compile(2, qc, target=target, seed_transpiler=seed)[0]
                with patch.object(
                    common, "_AbsorbIntoSwaps", lambda target=None: _AbsorbIntoSwaps()
                ):
                    expected = _compile(2, qc, target=target, seed_transpiler=seed)[0]
                self.assertEqual(out, expected)
                self.assertEqual(out.layout, expected.layout)

    @data(
        ({}, 0),
        ({"measure": True}, 1),
        ({"other_pair": True}, 1),
        ({"final_barrier": True}, 0),
        ({"final_barrier": True, "measure": True}, 1),
    )
    @unpack
    def test_tail_rule(self, variant, expected):
        """A short-T2 qubit that is unmeasured and ends on one more pair keeps its SWAP alone."""
        qc = _tail_circuit(**variant)
        dag = circuit_to_dag(qc)
        self.assertEqual(len(_find_moves(dag)[2]), 1)
        self.assertEqual(len(_find_moves(dag, frozenset({2, 3}))[2]), 1)
        self.assertEqual(len(_find_moves(dag, frozenset({1}))[2]), expected)
        short = _line_target(4, t2=[100e-6, 8e-6, 100e-6, 100e-6])
        for target in (None, short):
            out = dag_to_circuit(_AbsorbIntoSwaps(target=target).run(circuit_to_dag(qc)))
            self.assertEqual(out == qc, target is short and expected == 0)
            if not variant.get("measure"):
                self.assertEqual(Operator(out), Operator(qc))

    def test_short_t2_qubits(self):
        """The T2 part of the tail rule reads the Target's qubit properties."""
        threshold = _swap_absorption._TAIL_T2_THRESHOLD
        target = _line_target(4, t2=[threshold / 2, threshold, None, 2 * threshold])
        self.assertEqual(_short_t2_qubits(target, 4), frozenset({0}))
        self.assertEqual(_short_t2_qubits(target, 0), frozenset())
        self.assertEqual(_short_t2_qubits(_line_target(4), 4), frozenset())
        self.assertEqual(_short_t2_qubits(None, 4), frozenset())

    @data(2, 3)
    def test_preset_tail_rule_uses_target(self, level):
        """The preset pipeline applies the T2 part of the tail rule at O3 only."""
        qc = _tail_circuit()
        for t2, expected in ((8e-6, 0 if level == 3 else 1), (100e-6, 1)):
            target = _line_target(4, t2=[100e-6, t2, 100e-6, 100e-6])
            pm = generate_preset_pass_manager(level, target=target, initial_layout=[0, 1, 2, 3])
            absorb = _translation_absorb(pm)
            moves = _find_moves(circuit_to_dag(qc), _short_t2_qubits(absorb.target, 4))[2]
            self.assertEqual(len(moves), expected)

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
        expected = {
            name: pauli for pauli, names in (*single.items(), *double.items()) for name in names
        }
        self.assertEqual(_WIRE_PAULIS, expected)

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

    def test_hop_single_qubit_gates_across_swap(self):
        """Single-qubit gates next to a SWAP cross it onto the other qubit to let a term reach it."""
        lookalike = QuantumCircuit(1)
        lookalike.h(0)
        for name, gate in (("h", None), ("custom gate named rz", _custom_gate("rz", lookalike))):
            for step in (1, -1):
                with self.subTest(name, step=step):
                    qc = QuantumCircuit(3)
                    qc.h(2)
                    if step < 0:
                        qc.cp(0.3, 0, 1)
                        qc.cz(0, 2)
                        qc.sx(1)
                        qc.h(0) if gate is None else qc.append(gate, [0])
                        qc.swap(0, 1)
                    else:
                        qc.swap(0, 1)
                        qc.h(0) if gate is None else qc.append(gate, [0])
                        qc.sx(1)
                        qc.cz(0, 2)
                        qc.cp(0.3, 0, 1)
                    qc.cx(0, 2)
                    qc.cx(1, 2)
                    _, _, moves = _find_moves(circuit_to_dag(qc))
                    self.assertEqual([(move[2], len(move[3])) for move in moves], [(step, 2)])
                    out = _run(qc)
                    np.testing.assert_allclose(
                        Operator(out).data, Operator(qc).data, atol=1e-12, rtol=0
                    )
                    swap = [inst.name for inst in out.data].index("swap")
                    self.assertEqual(out.data[_neighbour_on_both_wires(out, swap, step)].name, "cp")

    def test_hop_preserves_same_pair_donor_block(self):
        """A hop must not split a donor from its existing two-qubit resynthesis run."""
        for reverse in (False, True):
            qc = QuantumCircuit(3)
            qc.swap(0, 1)
            qc.h(0)
            qc.cz(0, 2)
            qc.cp(0.3, 0, 1)
            qc.rz(0.2, 1)
            qc.cp(0.4, 0, 1)
            qc.cx(0, 2)
            qc.cx(1, 2)
            if reverse:
                qc = qc.reverse_ops()
                qc.cx(0, 2)
                qc.cx(1, 2)
            self.assertFalse(_find_moves(circuit_to_dag(qc))[2])
            self.assertEqual(_run(qc), qc)

    def test_hop_skips_gap_cleared_by_existing_move(self):
        """A move that clears the only multi-qubit blocker makes a further hop unnecessary."""
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                qc = QuantumCircuit(4)
                qc.swap(0, 1)
                qc.h(0)
                qc.cz(0, 2)
                qc.cp(0.3, 0, 1)
                qc.cz(2, 3)
                qc.swap(0, 2)
                qc.cx(0, 3)
                qc.cx(1, 3)
                qc.cx(2, 3)
                if reverse:
                    qc = qc.reverse_ops()
                    qc.cx(0, 3)
                    qc.cx(1, 3)
                    qc.cx(2, 3)
                order, units, moves = _find_moves(circuit_to_dag(qc))
                self.assertEqual(len(moves), 1)
                unit, swap, step, hops = moves[0]
                self.assertEqual(order[units[unit][0][0]].op.base_class, CZGate)
                self.assertEqual(
                    [qc.find_bit(qubit).index for qubit in order[units[swap][0][0]].qargs],
                    [0, 2],
                )
                self.assertEqual(step, 1 if reverse else -1)
                self.assertEqual(hops, [])
                out = _run(qc)
                np.testing.assert_allclose(
                    Operator(out).data, Operator(qc).data, atol=1e-12, rtol=0
                )

    def test_hop_moves_complete_cz_block(self):
        """Preserve a block's internal noncommuting gates when its external crossings commute."""
        for reverse in (False, True):
            for unsafe in (False, True):
                with self.subTest(reverse=reverse, unsafe=unsafe):
                    qc = QuantumCircuit(3)
                    qc.swap(0, 1)
                    qc.h(0)
                    qc.cz(0, 2)
                    qc.cz(0, 1)
                    qc.sx(0 if unsafe else 1)
                    qc.rz(0.2, 1)
                    qc.cz(0, 1)
                    qc.cx(0, 2)
                    qc.cx(1, 2)
                    if reverse:
                        qc = qc.reverse_ops()
                        qc.cx(0, 2)
                        qc.cx(1, 2)
                    order, units, moves = _find_moves(circuit_to_dag(qc))
                    self.assertEqual(bool(moves), not unsafe)
                    if moves:
                        self.assertEqual(len(moves), 1)
                        members = [order[member].op.base_class for member in units[moves[0][0]][0]]
                        self.assertEqual(members.count(CZGate), 2)
                    np.testing.assert_allclose(
                        Operator(_run(qc)).data, Operator(qc).data, atol=1e-12, rtol=0
                    )

    def test_hop_complete_cz_block_respects_short_t2_tail(self):
        """The T2 tail check excludes every transported source, not only the compound ID."""
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.h(0)
        qc.cz(0, 2)
        qc.cz(0, 1)
        qc.sx(1)
        qc.cz(0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        _, units, moves = _find_moves(circuit_to_dag(qc))
        self.assertEqual(len(moves), 1)
        self.assertEqual(len(units[moves[0][0]][0]), 3)
        for qubit in (0, 1):
            with self.subTest(qubit=qubit):
                self.assertEqual(_find_moves(circuit_to_dag(qc), frozenset({qubit}))[2], [])
        np.testing.assert_allclose(Operator(_run(qc)).data, Operator(qc).data, atol=1e-12, rtol=0)

    def test_hop_complete_cz_block_short_t2_measurement(self):
        """A measured short-T2 wire is not vetoed by the unmeasured-tail rule."""
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.h(0)
        qc.cz(0, 2)
        qc.cz(0, 1)
        qc.sx(1)
        qc.cz(0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        qc.measure_all()
        for qubit in (0, 1):
            with self.subTest(qubit=qubit):
                _, units, moves = _find_moves(circuit_to_dag(qc), frozenset({qubit}))
                self.assertEqual(len(moves), 1)
                self.assertEqual(len(units[moves[0][0]][0]), 3)
        expected = qc.remove_final_measurements(inplace=False)
        actual = _run(qc).remove_final_measurements(inplace=False)
        np.testing.assert_allclose(
            Operator(actual).data, Operator(expected).data, atol=1e-12, rtol=0
        )

    def test_hop_cz_block_rejects_lookalike_and_directive(self):
        """Unknown shared-wire operations and directives must not be part of a moved block."""
        custom = QuantumCircuit(2)
        custom.cx(1, 0)
        custom.h(0)
        for barrier in (True, False):
            qc = QuantumCircuit(3)
            qc.swap(0, 1)
            qc.h(0)
            qc.cz(0, 2)
            qc.cz(0, 1)
            if barrier:
                qc.barrier(1)
            else:
                qc.append(_custom_gate("cz", custom), [0, 1])
            qc.cz(0, 1)
            qc.cx(0, 2)
            qc.cx(1, 2)
            self.assertFalse(_find_moves(circuit_to_dag(qc))[2])
            self.assertEqual(_run(qc), qc)

    def test_hop_checks_both_shared_qubits(self):
        """Relabelling one-qubit gates does not permit a noncommuting two-qubit crossing."""
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.h(0)
        qc.cz(0, 2)
        qc.cx(2, 1)
        qc.cp(0.3, 0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        self.assertFalse(_find_moves(circuit_to_dag(qc))[2])
        self.assertEqual(_run(qc), qc)

    def test_hop_keeps_moves_without_hops(self):
        """Hopping is only tried for a SWAP no unit reaches otherwise."""
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.rz(0.2, 0)
        qc.cz(0, 2)
        qc.cp(0.3, 0, 1)
        qc.cz(1, 2)
        qc.swap(0, 1)
        qc.h(0)
        qc.cz(0, 2)
        qc.cp(0.4, 0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        _, _, moves = _find_moves(circuit_to_dag(qc))
        self.assertEqual([len(move[3]) for move in moves], [0, 1])
        self.assertEqual(Operator(_run(qc)), Operator(qc))

    def test_random_hop_operators(self):
        """Seeded circuits with single-qubit gates between SWAPs and their terms stay exact."""
        rng = random.Random(505)
        hops = 0
        singles = ("rx", "ry", "rz", "sx", "s", "h", "t")
        doubles = ("cx", "cz", "cp", "rzz", "swap")
        for _ in range(96):
            qc = QuantumCircuit(4)
            for _ in range(rng.randint(10, 18)):
                if rng.random() < 0.15:
                    qc.swap(*rng.sample(range(4), 2))
                elif rng.random() < 0.5:
                    name = rng.choice(singles)
                    angle = (rng.uniform(-1, 1),) if name in ("rx", "ry", "rz") else ()
                    getattr(qc, name)(*angle, rng.randrange(4))
                else:
                    name = rng.choice(doubles)
                    angle = (rng.uniform(-1, 1),) if name in ("cp", "rzz") else ()
                    getattr(qc, name)(*angle, *rng.sample(range(4), 2))
            qc.cx(0, 1)
            qc.cx(2, 3)
            qc.cx(1, 2)
            qc.cx(3, 0)
            hops += sum(len(move[3]) for move in _find_moves(circuit_to_dag(qc))[2])
            self.assertTrue(Operator(_run(qc)).equiv(Operator(qc)))
        self.assertGreater(hops, 0)

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
        self.assertEqual({step for _, _, step, _ in moves}, {-1, 1})
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
                out, moved, _ = _compile(2, qc, backend=backend, seed_transpiler=seed)
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
            out, moved, _ = _compile(3, qc, **kwargs)
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
            out, moved, _ = _compile(3, qc, target=target, seed_transpiler=seed)
            self.assertGreater(len(moved), 0)
            self.assertTrue(Operator.from_circuit(out).equiv(Operator(qc)))

    @data(2, 3)
    def test_provider_translation_plugin(self, level):
        """A provider translation plugin that passes its level gets the default plugin's output."""
        qc = _qft(8, measure=True)
        expected, default_moved, _ = _compile(
            level, qc, backend=_heavy_hex_backend(), seed_transpiler=1
        )
        with _with_mock_translation_plugins():
            backend = _heavy_hex_backend(_ProviderPluginBackend)
            pm = generate_preset_pass_manager(level, backend=backend, seed_transpiler=1)
            self.assertIsInstance(_translation_absorb(pm), _AbsorbIntoSwaps)
            out, moved, _ = _compile(level, qc, backend=backend, seed_transpiler=1)
            backend.plugin = "mock_provider_no_level"
            pm = generate_preset_pass_manager(level, backend=backend, seed_transpiler=1)
            self.assertIsNone(_translation_absorb(pm))
            _, unmoved, _ = _compile(level, qc, backend=backend, seed_transpiler=1)
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
            _, moved, _ = _compile(level, qc, snapshots=snapshots, seed_transpiler=1, **kwargs)
            self.assertGreater(len(moved), 0)
            (virtual, _), (routed, layout) = snapshots
            self.assertIn("swap", routed.count_ops())
            self.assertTrue(_replays(virtual, routed, layout))
