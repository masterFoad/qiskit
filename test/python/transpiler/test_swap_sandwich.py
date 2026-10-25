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

"""Tests for the ``sandwich`` option of the post-routing SWAP absorption pass."""

import itertools
from unittest import mock

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate, Parameter, Qubit
from qiskit.circuit.library import (
    CYGate,
    CZGate,
    RZGate,
    UnitaryGate,
    get_standard_gate_name_mapping,
)
from qiskit.circuit.random import random_circuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap, PassManager, generate_preset_pass_manager
from qiskit.transpiler.passes import ConsolidateBlocks, UnitarySynthesis
from qiskit.transpiler.preset_passmanagers import _swap_absorption, _swap_cost, _swap_sandwich
from qiskit.transpiler.preset_passmanagers._swap_absorption import _AbsorbIntoSwaps
from test import QiskitTestCase  # pylint: disable=wrong-import-order


def _run_with(circuit, **options):
    return dag_to_circuit(_AbsorbIntoSwaps(**options).run(circuit_to_dag(circuit)))


def _routed(num_qubits, depth, seed, basis):
    qc = random_circuit(num_qubits, depth, max_operands=2, seed=seed)
    return transpile(
        qc,
        coupling_map=CouplingMap.from_line(num_qubits),
        basis_gates=list(basis),
        optimization_level=0,
        seed_transpiler=seed,
        routing_method="sabre",
    )


_BASES = (
    ("cx", "rz", "sx", "x", "swap"),
    ("cx", "cz", "h", "s", "sdg", "rz", "ry", "rx", "swap", "cp", "rzz", "cy", "rxx"),
)
_CASES = [(n, d, s) for n in (3, 4, 5) for d in (6, 12) for s in range(3)]


def _sandwich_window(clifford):
    """A SWAP, a window on (0, 1) that commutes with Z on both wires, a SWAP, then more work."""
    qc = QuantumCircuit(3)
    qc.swap(0, 1)
    qc.cz(0, 2)
    if clifford:
        qc.h(0)
        qc.cx(2, 0)
    else:
        qc.t(0)
    qc.rz(0.3, 1)
    qc.swap(0, 1)
    qc.cx(1, 2)
    qc.cx(0, 2)
    return qc


def _two_swaps(*between):
    """Two SWAPs on (0, 1) around ``between`` and no further work: the second SWAP ends both
    wires."""
    qc = QuantumCircuit(3)
    qc.swap(0, 1)
    for name, args in between:
        getattr(qc, name)(*args)
    qc.swap(0, 1)
    return qc


def _peephole(circuit):
    return PassManager(
        [ConsolidateBlocks(basis_gates=["cx", "u"]), UnitarySynthesis(basis_gates=["cx", "u"])]
    ).run(circuit)


class _LoggedUnitary(UnitaryGate):
    """A ``UnitaryGate`` subclass whose ``to_matrix`` calls are logged, the third one raising."""

    log = []
    fail_at = None

    def to_matrix(self):
        self.log.append(("to_matrix", self.label))
        if len(self.log) == self.fail_at:
            raise ValueError("third call")
        return super().to_matrix()


class _LoggedGate(Gate):
    """A custom two-qubit gate whose ``to_matrix`` returns CZ and logs the call."""

    log = []
    hook = None

    def __init__(self, label):
        super().__init__("logged_cz", 2, [], label=label)

    def to_matrix(self):
        self.log.append(("to_matrix", self.label))
        if self.hook is not None:
            self.hook()  # pylint: disable=not-callable
        return np.diag([1, 1, 1, -1]).astype(complex)


def _sandwich_with(*between):
    """``swap(0, 1)``, ``between`` on (0, 1, 2), ``swap(0, 1)``, then work on both wires."""
    qc = QuantumCircuit(3)
    qc.swap(0, 1)
    for gate, qargs in between:
        qc.append(gate, qargs)
    qc.swap(0, 1)
    qc.cx(1, 2)
    qc.cx(0, 2)
    return qc


def _dag_of(circuit):
    """The DAG of ``circuit``, keeping the Python operation objects."""
    dag = DAGCircuit()
    dag.add_qubits(circuit.qubits)
    for instruction in circuit.data:
        dag.apply_operation_back(instruction.operation, instruction.qubits, check=False)
    return dag


class TestSwapSandwich(QiskitTestCase):
    """The sandwich option is exact and off by default."""

    def test_g_matrices(self):
        """The emitted g(P, Q) circuits equal the closed form exactly and square to identity."""
        self.assertEqual(len(_swap_sandwich._G), 9)
        for frame, matrix in _swap_sandwich._G.items():
            block = DAGCircuit()
            wires = [Qubit(), Qubit()]
            block.add_qubits(wires)
            _swap_sandwich._emit_g(block, wires[0], wires[1], frame)
            emitted = Operator(dag_to_circuit(block)).data
            np.testing.assert_allclose(emitted, matrix, atol=1e-12)
            np.testing.assert_allclose(matrix @ matrix, np.eye(4), atol=1e-12)
            p, q = (_swap_sandwich._PAULI[x] for x in frame)
            # g commutes with P on its first qubit and with Q on its second.
            for local in (np.kron(np.eye(2), p), np.kron(q, np.eye(2))):
                np.testing.assert_allclose(matrix @ local, local @ matrix, atol=1e-12)

    def test_window_labels_come_from_the_table(self):
        """Window letters are the movers' table-backed labels plus four more table-backed gates;
        ``id`` is Z; other standard gates block every wire."""
        expected = {
            **dict.fromkeys(("rz", "z", "s", "sdg", "t", "tdg", "p", "u1", "id"), "Z"),
            **dict.fromkeys(("rx", "x", "sx", "sxdg"), "X"),
            **dict.fromkeys(("ry", "y"), "Y"),
            **dict.fromkeys(("cz", "cp", "crz", "cu1", "cs", "csdg", "rzz"), "ZZ"),
            **dict.fromkeys(("cx", "crx", "csx", "rzx"), "ZX"),
            **dict.fromkeys(("cy", "cry"), "ZY"),
            "ch": "Z-",
            "ecr": "-X",
            "rxx": "XX",
            "ryy": "YY",
        }
        for name, gate in get_standard_gate_name_mapping().items():
            width = getattr(gate, "num_qubits", 0)
            if not width or name in ("measure", "reset", "delay", "barrier", "global_phase"):
                continue
            for angle in (Parameter("theta"), 0.0, 0.37):
                with self.subTest(gate=name, angle=angle):
                    op = type(gate)(*([angle] * len(gate.params))) if gate.params else gate
                    qc = QuantumCircuit(width)
                    qc.append(op, range(width))
                    node = next(circuit_to_dag(qc).topological_op_nodes())
                    label = _swap_sandwich._wire_labels(node, {})
                    self.assertEqual(label, expected.get(name, "-" * width))

    def test_off_by_default(self):
        """The default pass and ``sandwich=False`` are the same pass."""
        self.assertFalse(_AbsorbIntoSwaps().sandwich)
        for basis, (n, depth, seed) in itertools.product(_BASES, _CASES):
            qc = _routed(n, depth, seed, basis)
            self.assertEqual(_run_with(qc, sandwich=False), _run_with(qc))

    def test_two_swaps_sandwich(self):
        """Two SWAPs around a Z-diagonal window, with work after them: 7 CX become 5 after
        resynthesis, and the operator is unchanged."""
        qc = _two_swaps(("cz", (0, 2)), ("rz", (0.3, 1)), ("t", (0,)))
        qc.cx(0, 2)
        qc.cx(1, 2)
        out = _run_with(qc, sandwich=True)
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))
        self.assertEqual(out.count_ops().get("cz", 0), 3)
        self.assertEqual(_peephole(out).count_ops().get("cx", 0), 7)
        self.assertEqual(_peephole(qc).count_ops().get("cx", 0), 9)

    def test_wire_end_guard(self):
        """The last two-qubit run of a wire never receives a g."""
        qc = _two_swaps(("cz", (0, 2)), ("rz", (0.3, 1)))
        self.assertEqual(_run_with(qc, sandwich=True), _run_with(qc))
        out = _run_with(_sandwich_window(False), sandwich=True)
        self.assertNotEqual(out, _run_with(_sandwich_window(False)))
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(_sandwich_window(False)))

    def test_measure_or_barrier_after_counts_as_work(self):
        """A measurement or a barrier after the second SWAP is further work on the wire, so the
        wire-end guard does not apply."""
        for directive in ("measure", "barrier"):
            with self.subTest(directive):
                qc = QuantumCircuit(3, 2)
                qc.swap(0, 1)
                qc.cz(0, 2)
                qc.rz(0.3, 1)
                qc.swap(0, 1)
                if directive == "measure":
                    qc.measure([0, 1], [0, 1])
                else:
                    qc.barrier([0, 1])
                out = _run_with(qc, sandwich=True)
                self.assertNotEqual(out, qc)
                if directive == "barrier":
                    self.assertEqual(Operator(out), Operator(qc))

    def test_clifford_in_window_blocks(self):
        """A Hadamard in the window blocks every frame on that wire."""
        qc = _sandwich_window(True)
        self.assertEqual(_run_with(qc, sandwich=True), _run_with(qc))

    def test_no_insertion_when_window_blocks(self):
        """An X rotation on the window's Z wire blocks every frame on that wire."""
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.cz(0, 2)
        qc.rx(0.3, 0)
        qc.cz(0, 2)
        qc.swap(0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        self.assertEqual(_run_with(qc, sandwich=True), _run_with(qc))

    def test_non_unitary_or_custom_window_blocks(self):
        """A measurement, a reset, control flow or a custom gate in the window is never crossed."""
        custom = QuantumCircuit(1)
        custom.rz(0.2, 0)
        custom_gate = custom.to_gate(label="custom_rz")
        cases = {}
        for name in ("measure", "reset", "if", "custom"):
            qc = QuantumCircuit(3, 1)
            qc.swap(0, 1)
            if name == "measure":
                qc.measure(0, 0)
            elif name == "reset":
                qc.reset(0)
            elif name == "if":
                with qc.if_test((qc.clbits[0], 1)):
                    qc.rz(0.2, 0)
            else:
                qc.append(custom_gate, [0])
            qc.swap(0, 1)
            qc.cx(0, 2)
            qc.cx(1, 2)
            cases[name] = qc
        for name, qc in cases.items():
            with self.subTest(name):
                self.assertEqual(_run_with(qc, sandwich=True), _run_with(qc))

    def test_symbolic_gate_in_window(self):
        """A symbolic standard gate commutes with its letter for every parameter value, so it
        does not block the window; the output is exact for any bound value."""
        theta = Parameter("theta")
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.cp(theta, 0, 2)
        qc.rz(theta, 1)
        qc.swap(0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        out = _run_with(qc, sandwich=True)
        self.assertNotEqual(out, _run_with(qc))
        self.assertEqual(out.parameters, qc.parameters)
        for value in (0.0, 0.37, -2.1):
            self.assertEqual(
                Operator(out.assign_parameters({theta: value})),
                Operator(qc.assign_parameters({theta: value})),
            )

    def test_random_routed_circuits_keep_operator(self):
        """Exact operator, including the global phase, on random routed circuits."""
        changed = 0
        for basis, (n, depth, seed), fractional in itertools.product(_BASES, _CASES, (False, True)):
            qc = _routed(n, depth, seed, basis)
            pass_ = _AbsorbIntoSwaps(sandwich=True)
            pass_.fractional = fractional
            out = dag_to_circuit(pass_.run(circuit_to_dag(qc)))
            changed += out != _run_with(qc)
            # Operator(), not Operator.from_circuit(): the routed input carries a layout, and the
            # pass output (without one) is compared as the same physical circuit.
            self.assertEqual(Operator(out), Operator(qc))
        self.assertGreater(changed, 0)

    def test_layout_semantics_of_equality(self):
        """With the routed layout attached to the output, ``Operator.from_circuit`` (which undoes
        the layout permutations) of the sandwich output equals the logical circuit's operator."""
        qc = QuantumCircuit(5)
        qc.h(range(5))
        for i in range(5):
            for j in range(i + 1, 5):
                qc.cp(0.3 + 0.1 * (i + j), i, j)
        for i in range(5):
            for j in range(i + 1, 5):
                qc.cx(i, j)
        pm = generate_preset_pass_manager(
            2,
            coupling_map=CouplingMap.from_line(5),
            basis_gates=["cx", "rz", "sx", "x"],
            seed_transpiler=3,
        )
        pm.translation = PassManager([])
        pm.optimization = PassManager([])
        routed = pm.run(qc)
        out = _run_with(routed, sandwich=True)
        self.assertNotEqual(out, _run_with(routed))
        self.assertEqual(Operator(out), Operator(routed))
        out._layout = routed.layout
        self.assertEqual(Operator.from_circuit(out), Operator(qc))

    def test_preset_keeps_sandwich_off(self):
        """The preset pass managers build the pass with the sandwich off and the target's cost."""
        for backend, fractional in (
            (GenericBackendV2(5, seed=2), False),
            (GenericBackendV2(5, basis_gates=["cz", "rz", "sx", "x", "rzz"], seed=2), True),
        ):
            pm = generate_preset_pass_manager(2, backend, seed_transpiler=1)
            found = [
                task
                for task in pm.translation.to_flow_controller().tasks
                if isinstance(task, _AbsorbIntoSwaps)
            ]
            self.assertEqual(len(found), 1)
            self.assertFalse(found[0].sandwich)
            self.assertEqual(found[0].fractional, fractional)

    def test_custom_matrices_are_read_in_topological_order(self):
        """Every non-standard gate of width <= 2 is asked for its matrix once, in topological
        order, before any window label; a stored ``UnitaryGate`` subclass keeps its class."""
        diag = UnitaryGate(np.diag([1, 1j, 1j, 1]), label="u")
        qc = _sandwich_with(
            (_LoggedUnitary(diag.to_matrix(), label="a"), [0, 2]),
            (_LoggedUnitary(diag.to_matrix(), label="c"), [1, 2]),
            (_LoggedGate("b"), [1, 2]),
        )
        # ``circuit_to_dag`` stores a ``UnitaryGate`` natively and drops the subclass, so build the
        # DAG directly.
        dag = _dag_of(qc)
        self.assertEqual(
            [type(node.op) for node in dag.op_nodes() if node.name != "swap"][:3],
            [_LoggedUnitary, _LoggedUnitary, _LoggedGate],
        )
        expected = Operator(qc)
        log = []
        _LoggedUnitary.log = _LoggedGate.log = log
        out = _swap_sandwich.insert_swap_sandwiches(dag)
        self.assertEqual(log[:3], [("to_matrix", "a"), ("to_matrix", "c"), ("to_matrix", "b")])
        # Then the window letters: ``a`` on wire 0, ``c`` on wire 1 (``b`` has no label call).
        self.assertEqual(log[3:], [("to_matrix", "a"), ("to_matrix", "c")])
        self.assertEqual(Operator(dag_to_circuit(out)), expected)

    def test_raising_label_matrix_propagates(self):
        """A ``to_matrix`` failure during the matrix pass is swallowed (no matrix); the same
        failure while labelling the window propagates."""
        diag = np.diag([1, 1j, 1j, 1])
        for fail_at, raises in ((1, False), (2, True)):
            with self.subTest(fail_at=fail_at):
                dag = _dag_of(_sandwich_with((_LoggedUnitary(diag, label="a"), [0, 2])))
                _LoggedUnitary.log = []
                with mock.patch.object(_LoggedUnitary, "fail_at", fail_at):
                    if raises:
                        with self.assertRaisesRegex(ValueError, "third call"):
                            _swap_sandwich.insert_swap_sandwiches(dag)
                    else:
                        _swap_sandwich.insert_swap_sandwiches(dag)
                self.assertEqual(len(_LoggedUnitary.log), 2)

    def test_label_lookups_cold_and_warm(self):
        """``_wire_labels`` is looked up in the module at every letter miss, the same number of
        times with a cold and a warm label cache; only the cold run asks the commutation table."""
        qc = _sandwich_with((CZGate(), [0, 2]), (CYGate(), [1, 2]), (RZGate(0.3), [1]))
        calls = {}
        with mock.patch.dict(_swap_absorption._WIRE_PAULI_CACHE, clear=True):
            for state in ("cold", "warm"):
                with (
                    mock.patch.object(
                        _swap_sandwich, "_wire_labels", wraps=_swap_sandwich._wire_labels
                    ) as labels,
                    mock.patch.object(
                        _swap_absorption,
                        "SessionCommutationChecker",
                        wraps=_swap_absorption.SessionCommutationChecker,
                    ) as checker,
                ):
                    out = _swap_sandwich.insert_swap_sandwiches(circuit_to_dag(qc))
                    calls[state] = (
                        labels.call_count,
                        checker.check_commutation_entries.call_count,
                    )
                self.assertEqual(Operator(dag_to_circuit(out)), Operator(qc))
        self.assertGreater(calls["cold"][0], 0)
        self.assertEqual(calls["cold"][0], calls["warm"][0])
        self.assertGreater(calls["cold"][1], 0)
        self.assertEqual(calls["warm"][1], 0)


class TestTwoQubitCost(QiskitTestCase):
    """The cost used by the moves and the sandwich."""

    def test_cost_on_tiny_angles(self):
        """The cost follows the synthesis on tiny controlled phases (``cp(1e-5)`` is synthesized
        with no two-qubit gate), and is unknown when a Weyl coordinate lies between the zero and
        non-zero thresholds."""
        synthesis = PassManager([UnitarySynthesis(basis_gates=["cx", "u"])])
        for theta, expected in ((0.3, 2), (1e-3, 2), (1e-5, 0), (1e-13, 0)):
            with self.subTest(theta=theta):
                qc = QuantumCircuit(2)
                qc.cp(theta, 0, 1)
                matrix = Operator(qc).data
                self.assertEqual(_swap_cost.two_qubit_cost(matrix), expected)
                unitary = QuantumCircuit(2)
                unitary.unitary(matrix, [0, 1])
                self.assertEqual(synthesis.run(unitary).count_ops().get("cx", 0), expected)
        qc = QuantumCircuit(2)
        qc.rxx(0.3, 0, 1)
        qc.ryy(0.3, 0, 1)
        qc.rzz(1e-10, 0, 1)
        self.assertIsNone(_swap_cost.two_qubit_cost(Operator(qc).data))

    def test_cost_of_swap_classes(self):
        """SWAP costs 3, SWAP times a CZ-class gate 2, and with a native rzz a ZZ rotation 1."""
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        swap = Operator(qc).data
        self.assertEqual(_swap_cost.two_qubit_cost(swap), 3)
        for frame, g in _swap_sandwich._G.items():
            with self.subTest(frame=frame):
                self.assertEqual(_swap_cost.two_qubit_cost(swap @ g), 2)
        rzz = QuantumCircuit(2)
        rzz.rzz(0.3, 0, 1)
        self.assertEqual(_swap_cost.two_qubit_cost(Operator(rzz).data), 2)
        self.assertEqual(_swap_cost.two_qubit_cost(Operator(rzz).data, fractional=True), 1)
