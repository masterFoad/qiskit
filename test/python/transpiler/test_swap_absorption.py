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

import math
import random
from unittest.mock import patch

from ddt import ddt, data

from qiskit.circuit import Gate, Parameter, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import CU1Gate, RZGate, RZZGate, U1Gate, UnitaryGate, XGate
from qiskit.circuit.library import qaoa_ansatz
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.transpiler import CouplingMap, PassManager, Target, generate_preset_pass_manager
from qiskit.transpiler.passes import WrapAngles
from qiskit.transpiler.passes.utils.wrap_angles import WrapAngleRegistry
from qiskit.transpiler.preset_passmanagers import builtin_plugins
from qiskit.transpiler.preset_passmanagers import _swap_absorption
from qiskit.transpiler.preset_passmanagers._swap_absorption import (
    _AbsorbIntoSwaps,
    _CONTROLLED_PHASE_GATES,
    _WIRE_PAULIS,
    _find_moves,
)
from test import QiskitTestCase


def _run(circuit):
    """Run ``_AbsorbIntoSwaps`` without a target on ``circuit``."""
    return dag_to_circuit(_AbsorbIntoSwaps().run(circuit_to_dag(circuit)))


def _append_standard(circuit, name, qubits):
    """Add a gate from the pass's commutation table, including legacy gate classes."""
    if name in ("cu1", "u1"):
        circuit.append(CU1Gate(0.37) if name == "cu1" else U1Gate(0.37), qubits)
    else:
        parameterized = {"p", "rz", "rx", "ry", "cp", "crz", "crx", "rzz", "rxx", "ryy", "rzx"}
        arguments = (0.37,) if name in parameterized else ()
        getattr(circuit, name)(*arguments, *qubits)


def _next_on_both_wires(circuit, index):
    """Index of the next instruction after ``index`` if it acts on both qubits of ``index``."""
    qubits = set(circuit.data[index].qubits)
    for later, instruction in enumerate(circuit.data[index + 1 :], start=index + 1):
        if qubits & set(instruction.qubits):
            return later if qubits <= set(instruction.qubits) else None
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


def _line_target(num_qubits, angle_bounded):
    """A CZ line Target, optionally with a fractional ``rzz`` bounded to ``[0, pi/2]``."""
    coupling = CouplingMap.from_line(num_qubits)
    target = Target.from_configuration(
        ["cz", "rz", "sx", "x"], num_qubits=num_qubits, coupling_map=coupling
    )
    if angle_bounded:
        target.add_instruction(
            RZZGate(Parameter("theta")),
            dict.fromkeys(coupling.get_edges()),
            angle_bounds=[(0, math.pi / 2)],
        )
    return target


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


def _compile_recording_moves(level, target, circuit, seed=7):
    """Compile with the preset pass manager; return the output and the names of moved units."""
    moved = []
    original = _swap_absorption._find_moves

    def recording(dag, *args, **kwargs):
        order, units, moves = original(dag, *args, **kwargs)
        moved.extend(order[units[unit][0][0]].name for unit, _, _ in moves)
        return order, units, moves

    registry = WrapAngleRegistry()
    registry.add_wrapper("rzz", _fold_rzz)
    with (
        patch.object(_swap_absorption, "_find_moves", recording),
        patch.object(WrapAngles, "DEFAULT_REGISTRY", registry),
    ):
        out = generate_preset_pass_manager(level, target=target, seed_transpiler=seed).run(circuit)
    return out, moved


def _translation_stage_with_default_pass(self, pass_manager_config, optimization_level=None):
    """The default translation stage with the level 2 pass in its default configuration."""
    translation = builtin_plugins.BasisTranslatorPassManager().pass_manager(
        pass_manager_config, optimization_level
    )
    if optimization_level == 2:
        translation = PassManager([_AbsorbIntoSwaps()]) + translation
    return translation


@ddt
class TestAbsorbIntoSwaps(QiskitTestCase):
    """Test the ``_AbsorbIntoSwaps`` pass."""

    @data(
        (0, False, None),
        (1, False, None),
        (2, False, True),
        (2, True, True),
        (3, False, True),
        (3, True, False),
    )
    def test_preset_absorption_levels(self, setting):
        """O2 and O3 include the pass; O3 keeps controlled phases only on angle-bounded Targets."""
        level, angle_bounded, absorb_controlled_phase = setting
        target = _line_target(4, angle_bounded)
        self.assertEqual(target.has_angle_bounds(), angle_bounded)
        pm = generate_preset_pass_manager(level, target=target)
        first = pm.translation._tasks[0][0]
        self.assertEqual(isinstance(first, _AbsorbIntoSwaps), absorb_controlled_phase is not None)
        if absorb_controlled_phase is not None:
            self.assertEqual(first.absorb_controlled_phase, absorb_controlled_phase)

    def test_preset_absorption_without_target(self):
        """Basis gates alone (no angle bounds) absorb controlled phases at O2 and O3."""
        for level in (2, 3):
            pm = generate_preset_pass_manager(level, basis_gates=["cx", "rz", "sx", "x"])
            self.assertTrue(pm.translation._tasks[0][0].absorb_controlled_phase)

    def test_controlled_phase_option(self):
        """The option only stops controlled-phase units; it is on by default."""
        self.assertTrue(_AbsorbIntoSwaps().absorb_controlled_phase)
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.swap(0, 1)
        qc.rz(0.2, 0)
        qc.cz(0, 2)
        qc.cp(0.3, 0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        for name in sorted(_CONTROLLED_PHASE_GATES):
            phase = qc.copy_empty_like()
            for instruction in qc.data:
                if instruction.name == "cp":
                    _append_standard(phase, name, instruction.qubits)
                else:
                    phase.append(instruction)
            dag = circuit_to_dag(phase)
            self.assertEqual(len(_find_moves(dag)[2]), 1)
            self.assertEqual(len(_find_moves(dag, absorb_controlled_phase=False)[2]), 0)
            off = dag_to_circuit(_AbsorbIntoSwaps(absorb_controlled_phase=False).run(dag))
            self.assertEqual(off, phase)
        # A non-phase unit still moves with the option off.
        cz = QuantumCircuit(3)
        cz.h(0)
        cz.swap(0, 1)
        cz.rz(0.2, 0)
        cz.cz(0, 2)
        cz.cz(0, 1)
        cz.cx(0, 2)
        cz.cx(1, 2)
        self.assertEqual(len(_find_moves(circuit_to_dag(cz), absorb_controlled_phase=False)[2]), 1)

    def test_level_three_angle_bounded_moves_no_controlled_phase(self):
        """O3 on an angle-bounded Target never moves a controlled phase, but O2 there does."""
        target = _line_target(5, angle_bounded=True)
        qc = _phase_and_cx_circuit()
        for seed in (3, 7, 11):
            out, moved = _compile_recording_moves(3, target, qc, seed)
            self.assertFalse(set(moved) & _CONTROLLED_PHASE_GATES, moved)
            self.assertTrue(Operator.from_circuit(out).equiv(Operator(qc)))
            _, moved_o2 = _compile_recording_moves(2, target, qc, seed)
            self.assertTrue(set(moved_o2) & _CONTROLLED_PHASE_GATES, moved_o2)

    def test_level_three_unbounded_absorbs_as_level_two(self):
        """O3 on a Target without angle bounds moves controlled phases like O2."""
        target = _line_target(5, angle_bounded=False)
        qc = _phase_and_cx_circuit()
        for seed in (3, 7, 11):
            out, moved = _compile_recording_moves(3, target, qc, seed)
            self.assertTrue(set(moved) & _CONTROLLED_PHASE_GATES, moved)
            self.assertTrue(Operator.from_circuit(out).equiv(Operator(qc)))
        # The O3 pass has the O2 configuration, so a routed input gets the same moves.
        o2, o3 = (generate_preset_pass_manager(level, target=target) for level in (2, 3))
        self.assertTrue(o2.translation._tasks[0][0].absorb_controlled_phase)
        self.assertTrue(o3.translation._tasks[0][0].absorb_controlled_phase)

    @data(False, True)
    def test_level_two_uses_default_pass(self, angle_bounded):
        """O2 output equals that of the pass in its default configuration, on both Target kinds."""
        target = _line_target(5, angle_bounded)
        cost = SparsePauliOp.from_sparse_list(
            [("ZZ", [i, j], 1.0) for i in range(5) for j in range(i + 1, 5)], num_qubits=5
        )
        circuits = [
            _phase_and_cx_circuit(),
            qaoa_ansatz(cost, reps=2).assign_parameters([0.4, 0.9, 0.2, 0.7]),
        ]
        registry = WrapAngleRegistry()
        registry.add_wrapper("rzz", _fold_rzz)
        with patch.object(WrapAngles, "DEFAULT_REGISTRY", registry):
            for qc in circuits:
                for seed in (1, 2, 3):
                    out = generate_preset_pass_manager(2, target=target, seed_transpiler=seed).run(
                        qc
                    )
                    with patch.object(
                        builtin_plugins.DefaultTranslationPassManager,
                        "pass_manager",
                        _translation_stage_with_default_pass,
                    ):
                        expected = generate_preset_pass_manager(
                            2, target=target, seed_transpiler=seed
                        ).run(qc)
                    self.assertEqual(out, expected)
                    self.assertEqual(out.layout, expected.layout)

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
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.rz(0.2, 0)
        qc.cp(theta, 0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        out = _run(qc)
        self.assertEqual(out.parameters, qc.parameters)
        self.assertTrue(
            Operator(out.assign_parameters({theta: 0.37})).equiv(
                Operator(qc.assign_parameters({theta: 0.37}))
            )
        )
        self.assertNotEqual(out, qc)

    def test_control_flow_blocks_move(self):
        """An if/else boundary is never crossed by the CP candidate."""
        qc = QuantumCircuit(3, 1)
        qc.swap(0, 1)
        with qc.if_test((qc.clbits[0], 1)):
            qc.rz(0.2, 0)
        qc.cp(0.3, 0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        self.assertEqual(_run(qc), qc)

    def test_barrier_blocks_move(self):
        """A barrier between SWAP and CP prevents absorption."""
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.barrier(0, 1)
        qc.cp(0.3, 0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        self.assertEqual(_run(qc), qc)

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

    def test_moves_controlled_phase_to_swap(self):
        """A later controlled phase reaches the SWAP through commuting diagonal gates."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.swap(0, 1)
        qc.rz(0.2, 0)
        qc.cz(0, 2)
        qc.rzz(0.4, 1, 2)
        qc.cp(0.3, 0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        out = _run(qc)
        self.assertEqual(Operator(out), Operator(qc))
        swap_index = [inst.name for inst in out.data].index("swap")
        self.assertEqual(out.data[_next_on_both_wires(out, swap_index)].name, "cp")

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

    def test_moves_earlier_term_onto_swap(self):
        """An earlier diagonal term is delayed onto the SWAP when no later one is available."""
        qc = QuantumCircuit(3)
        qc.cp(0.3, 0, 1)
        qc.cz(0, 2)
        qc.swap(0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        out = _run(qc)
        self.assertEqual(Operator(out), Operator(qc))
        self.assertEqual([inst.name for inst in out.data[:3]], ["cz", "cp", "swap"])

    def test_moves_zz_block_after_swap(self):
        """A ``cx rz cx`` block after the SWAP is moved back onto it."""
        qc = QuantumCircuit(3)
        qc.h([0, 1, 2])
        qc.swap(0, 1)
        qc.cz(1, 2)
        qc.cx(0, 1)
        qc.rz(0.7, 1)
        qc.cx(0, 1)
        qc.h(0)
        qc.cx(0, 2)
        qc.cx(1, 2)
        out = _run(qc)
        self.assertEqual(Operator(out), Operator(qc))
        swap_index = [inst.name for inst in out.data].index("swap")
        self.assertEqual(out.data[swap_index + 1].name, "cx")
        self.assertEqual(_next_on_both_wires(out, swap_index), swap_index + 1)

    def test_diagonal_unitary_block(self):
        """A diagonal two-qubit ``UnitaryGate`` counts as a diagonal interaction."""
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.p(0.1, 1)
        qc.cp(0.5, 1, 2)
        qc.append(UnitaryGate(Operator.from_label("ZZ").to_matrix() * 1j), [0, 1])
        qc.cx(0, 2)
        qc.cx(2, 1)
        out = _run(qc)
        self.assertEqual(Operator(out), Operator(qc))
        swap_index = [inst.name for inst in out.data].index("swap")
        self.assertEqual(out.data[_next_on_both_wires(out, swap_index)].name, "unitary")

    def test_swap_ending_a_wire_untouched(self):
        """A SWAP after which one of its qubits does no more multi-qubit work is left alone."""
        qc = QuantumCircuit(3)
        qc.cp(0.3, 0, 1)
        qc.cz(0, 2)
        qc.swap(0, 1)
        qc.cx(0, 2)
        qc.h(1)
        self.assertEqual(_run(qc), qc)

    def test_non_commuting_gate_blocks(self):
        """Nothing moves across an operation it does not commute with."""
        qc = QuantumCircuit(2)
        qc.cp(0.3, 0, 1)
        qc.h(0)
        qc.swap(0, 1)
        qc.measure_all()
        self.assertEqual(_run(qc), qc)

    def test_adjacent_pair_untouched(self):
        """A SWAP that already touches a diagonal term of its pair is left as it is."""
        qc = QuantumCircuit(3)
        qc.cz(0, 1)
        qc.swap(0, 1)
        qc.rz(0.2, 0)
        qc.cp(0.3, 0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        self.assertEqual(_run(qc), qc)

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

    def test_moves_cx_through_commuting_gates(self):
        """A CX reaches the SWAP through gates diagonal on its control and X-like on its target."""
        qc = QuantumCircuit(4)
        qc.h([0, 1])
        qc.swap(0, 1)
        qc.rz(0.2, 0)
        qc.cx(0, 2)
        qc.sx(1)
        qc.cx(3, 1)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(1, 3)
        out = _run(qc)
        self.assertEqual(Operator(out), Operator(qc))
        swap_index = [inst.name for inst in out.data].index("swap")
        self.assertEqual(out.data[swap_index + 1].name, "cx")
        self.assertEqual(_next_on_both_wires(out, swap_index), swap_index + 1)

    def test_moves_ryy_through_y_rotations(self):
        """An ``ryy`` passes gates that commute with ``Y`` on the shared qubits."""
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.ry(0.3, 0)
        qc.ryy(0.2, 1, 2)
        qc.ryy(0.5, 0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        out = _run(qc)
        self.assertEqual(Operator(out), Operator(qc))
        self.assertEqual([inst.name for inst in out.data[:2]], ["swap", "ryy"])
        self.assertEqual(_next_on_both_wires(out, 0), 1)

    def test_reversed_cx_blocks(self):
        """A CX is not moved across a gate that is diagonal on its target qubit."""
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.rz(0.2, 1)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        self.assertEqual(_run(qc), qc)

    def test_moves_zz_block_through_shared_control(self):
        """A ``cx rz cx`` block passes a CX whose control is on a shared qubit."""
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 1)
        qc.rz(0.7, 1)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(0, 2)
        out = _run(qc)
        self.assertEqual(Operator(out), Operator(qc))
        self.assertEqual([inst.name for inst in out.data[:4]], ["swap", "cx", "rz", "cx"])

    @data("cp", "cx", "rz", "unitary")
    def test_custom_gate_with_standard_name_blocks(self, name):
        """A custom gate that reuses a standard name is not treated as the standard gate."""
        custom = QuantumCircuit(2 if name != "rz" else 1)
        custom.h(0)
        gate = Gate(name, custom.num_qubits, [])
        gate.definition = custom
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.append(gate, [0] if name == "rz" else [0, 2])
        qc.cp(0.3, 0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        self.assertEqual(_run(qc), qc)

    def test_custom_swap_is_not_a_swap(self):
        """A custom gate named ``swap`` is not a routing SWAP."""
        definition = QuantumCircuit(2)
        definition.cx(0, 1)
        gate = Gate("swap", 2, [])
        gate.definition = definition
        qc = QuantumCircuit(3)
        qc.append(gate, [0, 1])
        qc.rz(0.2, 0)
        qc.cp(0.3, 0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        self.assertEqual(_run(qc), qc)

    def test_nearly_diagonal_unitary_blocks(self):
        """A ``UnitaryGate`` with any nonzero off-diagonal entry is not diagonal."""
        matrix = UnitaryGate(Operator.from_label("ZZ")).to_matrix()
        matrix[0, 1] = matrix[1, 0] = 1e-13
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.append(UnitaryGate(matrix, check_input=False), [0, 2])
        qc.cp(0.3, 0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        self.assertEqual(_run(qc), qc)

    @data(2, 3)
    def test_preset_custom_cx_exact(self, level):
        """A custom gate named ``cx`` next to routing SWAPs compiles exactly at O2 and O3."""
        definition = QuantumCircuit(2)
        definition.cx(1, 0)
        definition.ry(0.3, 0)
        gate = Gate("cx", 2, [])
        gate.definition = definition
        qc = QuantumCircuit(4)
        qc.h(range(4))
        for i in range(4):
            for j in range(i + 1, 4):
                qc.cp(0.2 * (i + j), i, j)
                qc.append(gate, [j, i])
        target = Target.from_configuration(
            ["cx", "rz", "sx", "x"], num_qubits=4, coupling_map=CouplingMap.from_line(4)
        )
        out = generate_preset_pass_manager(level, target=target, seed_transpiler=7).run(qc)
        self.assertTrue(Operator.from_circuit(out).equiv(Operator(qc)))

    @data(2, 3)
    def test_preset_crz_through_swaps_exact(self, level):
        """Routed controlled-RZ interactions in both directions compile exactly at O2 and O3."""
        qc = QuantumCircuit(4)
        qc.h(range(4))
        for i in range(4):
            for j in range(4):
                if i != j:
                    qc.crz(0.1 * (i + 2 * j + 1), i, j)
        target = Target.from_configuration(
            ["cz", "rz", "sx", "x"], num_qubits=4, coupling_map=CouplingMap.from_line(4)
        )
        out = generate_preset_pass_manager(level, target=target, seed_transpiler=3).run(qc)
        self.assertTrue(Operator.from_circuit(out).equiv(Operator(qc)))

    def test_preset_pass_manager_qaoa(self):
        """Routed QAOA on a line needs fewer two-qubit gates, and stays equivalent."""
        cost = SparsePauliOp.from_sparse_list(
            [("ZZ", [i, j], 1.0) for i in range(5) for j in range(i + 1, 5)], num_qubits=5
        )
        qc = qaoa_ansatz(cost, reps=1).assign_parameters([0.4, 0.9])
        target = Target.from_configuration(
            ["cz", "rz", "sx", "x"], num_qubits=5, coupling_map=CouplingMap.from_line(5)
        )
        out = generate_preset_pass_manager(2, target=target, seed_transpiler=7).run(qc)
        self.assertEqual(Operator.from_circuit(out), Operator(qc))
        # The "translator" plugin is the same translation stage without the absorption step.
        without = generate_preset_pass_manager(
            2, target=target, seed_transpiler=7, translation_method="translator"
        ).run(qc)
        self.assertLess(out.count_ops()["cz"], without.count_ops()["cz"])
