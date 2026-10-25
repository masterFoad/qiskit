# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file or at
# https://www.apache.org/licenses/LICENSE-2.0.

"""Output contracts for lazy matrix admission in the SWAP sandwich pass.

Standard-gate matrices are read only when a run needs them, until the first custom ``Gate``.
These tests check that this changes no output.  Without a reference they check that the output
implements the input exactly.  Setting ``SWAP_SANDWICH_REFERENCE`` to the path of another
``_swap_sandwich.py`` additionally compares the returned DAG with that implementation's (callback
order and count may differ; only the returned DAG is compared).
"""

import importlib.util
import os
import sys
import unittest
from unittest import mock

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Operation, Parameter, Qubit
from qiskit.circuit.library import (
    Barrier,
    CPhaseGate,
    CXGate,
    CYGate,
    RZGate,
    Reset,
    SwapGate,
    UnitaryGate,
    XGate,
)
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Operator
from qiskit.transpiler.preset_passmanagers import _swap_sandwich


def _reference_module():
    """Load the module named by ``SWAP_SANDWICH_REFERENCE``, or ``None``."""
    path = os.environ.get("SWAP_SANDWICH_REFERENCE")
    if not path:
        return None
    spec = importlib.util.spec_from_file_location("swap_sandwich_reference", path)
    if spec is None or spec.loader is None:
        raise AssertionError("reference module is not loadable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dag(operations):
    dag = DAGCircuit()
    qubits = [Qubit() for _ in range(3)]
    dag.add_qubits(qubits)
    for operation, indices in operations:
        dag.apply_operation_back(operation, [qubits[index] for index in indices], check=False)
    return dag


def _sandwich_locals():
    frame = sys._getframe(1)  # pylint: disable=protected-access
    while frame.f_code.co_name != "insert_swap_sandwiches":
        frame = frame.f_back
    return frame.f_locals


def _window(extra):
    """A candidate sandwich with a deterministic custom operation in its window."""
    return [
        (SwapGate(), (0, 1)),
        (extra, (0, 2)),
        (RZGate(0.25), (1,)),
        (SwapGate(), (0, 1)),
        (CXGate(), (0, 2)),
        (CXGate(), (1, 2)),
    ]


def _spoof_window(operation):
    """A SWAP-adjacent one-qubit window, where the packed Spoof acts as X."""
    return [
        (SwapGate(), (0, 1)),
        (operation, (0,)),
        (RZGate(0.25), (1,)),
        (SwapGate(), (0, 1)),
        (CXGate(), (0, 2)),
        (CXGate(), (1, 2)),
    ]


class _SnapshotPoke(Gate):
    """Shared operation type, so that two independently built DAGs compare equal."""

    def __init__(self, theta):
        super().__init__("poke", 1, [])
        self._theta = theta

    def to_matrix(self):
        _sandwich_locals()["order"][0].params = [self._theta]
        return np.eye(2, dtype=complex)


class _NativeSnapshotPoke(Gate):
    """Mutate a native-built CPhase only while sandwich snapshots its matrix."""

    calls = 0

    def __init__(self):
        super().__init__("native_snapshot_poke", 1, [])

    def to_matrix(self):
        type(self).calls += 1
        frame = sys._getframe(1)  # pylint: disable=protected-access
        while frame is not None and frame.f_code.co_name != "insert_swap_sandwiches":
            frame = frame.f_back
        if frame is not None:
            cp = next(node for node in frame.f_locals["order"] if node.name == "cp")
            cp.params = [0.5]
        return np.eye(2, dtype=complex)


def _snapshot_mutation_window():
    """A window whose custom callback changes an earlier CPhase; built afresh for each run."""
    theta = Parameter("theta")

    return [
        (CPhaseGate(0), (0, 1)),
        (SwapGate(), (0, 1)),
        (_SnapshotPoke(theta), (0,)),
        (CYGate(), (0, 2)),
        (CYGate(), (1, 2)),
        (SwapGate(), (0, 1)),
        (Barrier(3), (0, 1, 2)),
    ]


def _canonicalize_spoof(dag):
    """Replace this fixture's cached non-Gate by its canonical packed X operation."""
    normalized = DAGCircuit()
    qubits = [Qubit() for _ in dag.qubits]
    normalized.add_qubits(qubits)
    normalized.global_phase = dag.global_phase
    lookup = {qubit: index for index, qubit in enumerate(dag.qubits)}
    for node in dag.topological_op_nodes():
        operation = node.op
        if isinstance(operation, _Spoof):
            np.testing.assert_array_equal(node.matrix, XGate().to_matrix())
            operation = XGate()
        normalized.apply_operation_back(
            operation, [qubits[lookup[qubit]] for qubit in node.qargs], check=False
        )
    return normalized


class _DeterministicCZ(Gate):
    def __init__(self, label="custom"):
        super().__init__("deterministic_cz", 2, [], label=label)

    def to_matrix(self):
        return np.diag([1, 1, 1, -1]).astype(complex)


class _PretendReset(Gate):
    """A Gate whose packed tag resembles Reset but whose original matrix is unitary."""

    _standard_instruction_type = Reset._standard_instruction_type

    def __init__(self):
        super().__init__("pretend_reset", 1, [])

    def to_matrix(self):
        return np.eye(2, dtype=complex)


class _Spoof(Operation):
    """A cached non-Gate operation that carries the packed X tag."""

    _standard_gate = XGate._standard_gate
    name = "spoof"
    label = None
    num_qubits = 1
    num_clbits = 0
    params = []


class TestSwapSandwichLazyMatrix(unittest.TestCase):
    """Small output fixtures for the standard-gate lazy path."""

    def _assert_reference_equal(self, make_operations, operator_check=True):
        reference = _reference_module()
        if reference is None and not operator_check:
            # These fixtures have no exact operator outside the pass (a reset-like tag, or a
            # callback that only works while the pass runs); they need a reference to compare.
            self.skipTest("SWAP_SANDWICH_REFERENCE is not set")
        out = _swap_sandwich.insert_swap_sandwiches(_dag(make_operations()))
        if reference is None:
            expected = Operator(dag_to_circuit(_dag(make_operations())))
            self.assertTrue(Operator(dag_to_circuit(out)).equiv(expected))
        else:
            self.assertEqual(out, reference.insert_swap_sandwiches(_dag(make_operations())))

    def test_deterministic_custom_gate_output_matches_reference(self):
        """A custom deterministic matrix keeps the returned DAG, regardless of callback order."""
        self._assert_reference_equal(lambda: _window(_DeterministicCZ()))

    def test_pretend_reset_output_matches_reference(self):
        """A cached reset-like Gate remains governed by its original unitary semantics."""
        self._assert_reference_equal(lambda: _window(_PretendReset()), operator_check=False)

    def test_previous_standard_matrix_snapshot_matches_reference(self):
        """A custom callback changing a prior numeric CPhase cannot invalidate its snapshot."""
        self._assert_reference_equal(_snapshot_mutation_window, operator_check=False)

    def test_native_standard_operation_snapshot_keeps_emitted_cphase(self):
        """The eager catch-up snapshots a native CPhase before a callback mutates its params."""
        _NativeSnapshotPoke.calls = 0
        circuit = QuantumCircuit(3)
        circuit.swap(0, 1)
        circuit.cy(0, 2)
        circuit.cy(1, 2)
        circuit.cp(0.0, 0, 1)
        circuit.append(_NativeSnapshotPoke(), [0])
        circuit.swap(0, 1)
        circuit.barrier(0, 1, 2)

        out = _swap_sandwich.insert_swap_sandwiches(circuit_to_dag(circuit))
        calls_after_pass = _NativeSnapshotPoke.calls
        emitted_cp = next(node for node in out.topological_op_nodes() if node.name == "cp")
        self.assertEqual(emitted_cp.op.params, [0.0])
        self.assertEqual(
            [
                (node.name, tuple(out.find_bit(qubit).index for qubit in node.qargs))
                for node in out.topological_op_nodes()
            ],
            [
                ("swap", (0, 1)),
                ("cz", (0, 1)),
                ("cy", (0, 2)),
                ("cy", (1, 2)),
                ("cz", (0, 1)),
                ("cp", (0, 1)),
                ("native_snapshot_poke", (0,)),
                ("swap", (0, 1)),
                ("barrier", (0, 1, 2)),
            ],
        )
        expected = QuantumCircuit(3)
        expected.swap(0, 1)
        expected.cz(0, 1)
        expected.cy(0, 2)
        expected.cy(1, 2)
        expected.cz(0, 1)
        expected.cp(0.0, 0, 1)
        expected.append(_NativeSnapshotPoke(), [0])
        expected.swap(0, 1)
        expected.barrier(0, 1, 2)
        self.assertEqual(Operator(dag_to_circuit(out)), Operator(expected))
        self.assertEqual(calls_after_pass, 1)

    def test_standard_matrices_are_not_read_before_a_candidate_run_needs_them(self):
        """The prepass does not invoke the eager admission helper for standard gates."""
        dag = _dag([(XGate(), (0,))])
        with mock.patch.object(_swap_sandwich, "_matrix", side_effect=AssertionError("eager")):
            self.assertEqual(_swap_sandwich.insert_swap_sandwiches(dag), dag)

    def test_cached_non_gate_follows_packed_standard_gate_view(self):
        """A cached non-Gate with the packed X tag is treated as the packed X gate."""
        spoof = _Spoof()
        spoof_dag = _dag(_spoof_window(spoof))
        packed_dag = _dag(_spoof_window(XGate()))
        spoof_node = next(node for node in spoof_dag.topological_op_nodes() if node.op is spoof)
        self.assertTrue(spoof_node.is_standard_gate())
        self.assertFalse(isinstance(spoof_node.op, Gate))
        self.assertEqual(
            _canonicalize_spoof(_swap_sandwich.insert_swap_sandwiches(spoof_dag)),
            _swap_sandwich.insert_swap_sandwiches(packed_dag),
        )

    def test_canonical_unitary_gate_predicate_rejects_subclass_and_instance_overrides(self):
        """Only an unmodified concrete UnitaryGate skips the first custom catch-up."""
        matrix = np.diag([1, 1, 1, -1]).astype(complex)

        class SubUnitary(UnitaryGate):
            pass

        to_matrix_override = UnitaryGate(matrix)
        to_matrix_override.to_matrix = lambda: matrix
        array_override = UnitaryGate(matrix)
        array_override.__array__ = lambda dtype=None, copy=None: matrix
        is_canonical = _swap_sandwich._is_canonical_unitary_gate
        self.assertTrue(is_canonical(UnitaryGate(matrix)))
        self.assertFalse(is_canonical(SubUnitary(matrix)))
        self.assertFalse(is_canonical(to_matrix_override))
        self.assertFalse(is_canonical(array_override))
        self.assertFalse(is_canonical(_DeterministicCZ()))

    def test_canonical_unitary_gate_predicate_rejects_replaced_providers(self):
        """A class-level replacement of either matrix provider keeps the custom catch-up."""
        gate = UnitaryGate(np.diag([1, 1, 1, -1]).astype(complex))
        is_canonical = _swap_sandwich._is_canonical_unitary_gate
        array = UnitaryGate.__array__
        to_matrix = Gate.to_matrix

        def replaced_array(self, *args, **kwargs):
            return array(self, *args, **kwargs)

        def replaced_to_matrix(self):
            return to_matrix(self)

        with mock.patch.object(UnitaryGate, "__array__", replaced_array):
            self.assertFalse(is_canonical(gate))
        with mock.patch.object(Gate, "to_matrix", replaced_to_matrix):
            self.assertFalse(is_canonical(gate))
        self.assertTrue(is_canonical(gate))

    def test_canonical_unitary_gate_output_matches_reference(self):
        """A canonical UnitaryGate keeps the returned DAG without the custom catch-up."""
        self._assert_reference_equal(
            lambda: _window(UnitaryGate(np.diag([1, 1, 1, -1]).astype(complex)))
        )
